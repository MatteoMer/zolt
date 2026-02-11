//! Prefix-Suffix Decomposition Prover for Jolt-Compatible Sumcheck
//!
//! This module implements the prefix-suffix decomposition state and computation
//! for the LookupsReadRaf sumcheck during address rounds.
//!
//! The decomposition efficiently computes:
//!   Σ_tables Σ_b table.combine(P(c, b), Q[b])
//!
//! Where:
//!   - P(c, b) is the prefix MLE evaluated at challenge c and remaining bits b
//!   - Q[b] = Σ_{j: prefix_bits[j] == b} u_eval[j] * suffix_mle(suffix_bits[j])
//!
//! Reference: jolt-core/src/zkvm/instruction_lookups/read_raf_checking.rs

const std = @import("std");

// Debug output control - set to true to enable verbose debug prints
const debug_verbose = false;
fn dbg(comptime fmt: []const u8, args: anytype) void {
    if (debug_verbose) std.debug.print(fmt, args);
}

const Allocator = std.mem.Allocator;

const prefixes_mod = @import("prefixes.zig");
const suffixes_mod = @import("suffixes.zig");
const LookupBits = prefixes_mod.LookupBits;
const Prefixes = prefixes_mod.Prefixes;
const Suffixes = suffixes_mod.Suffixes;
const suffixMle = suffixes_mod.suffixMle;
const tableSuffixes = suffixes_mod.tableSuffixes;

/// Number of lookup tables in Jolt
pub const NUM_TABLES: usize = 41;

/// Maximum number of suffixes any table can have (ValidSignedRemainderTable has 5)
pub const MAX_SUFFIXES_PER_TABLE: usize = 5;

/// Compute 2^exp as a field element
/// Handles large exponents (up to 128) that don't fit in u64
fn fieldPow2(comptime F: type, exp: usize) F {
    if (exp == 0) return F.one();

    // For small exponents, use direct computation
    if (exp < 64) {
        return F.fromU64(@as(u64, 1) << @intCast(exp));
    }

    // For large exponents, use repeated squaring
    // 2^exp = 2^64 * 2^(exp-64)
    const two_pow_64 = F.fromU64(1).sub(F.fromU64(1)).add(F.fromBytes(&[_]u8{
        0, 0, 0, 0, 0, 0, 0, 0, // Lower 8 bytes = 0
        1, 0, 0, 0, 0, 0, 0, 0, // 2^64 in little-endian
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
    }));

    var result = two_pow_64;
    var remaining = exp - 64;

    while (remaining >= 64) {
        result = result.mul(two_pow_64);
        remaining -= 64;
    }

    if (remaining > 0) {
        result = result.mul(F.fromU64(@as(u64, 1) << @intCast(remaining)));
    }

    return result;
}

/// LOG_K = 128 for RV64 (2*XLEN for interleaved operands)
pub const LOG_K: usize = 128;

/// Default number of phases for prefix-suffix decomposition
pub const DEFAULT_PHASES: usize = 8;

/// Suffix polynomial storage for a single table
/// Q[suffix_idx][prefix_idx] = Σ u_eval[j] * suffix_mle(suffix_bits[j])
pub fn TableSuffixPolys(comptime F: type) type {
    return struct {
        const Self = @This();

        /// One polynomial per suffix type used by this table
        polys: [][]F,
        /// Number of suffixes for this table
        num_suffixes: usize,
        /// Effective length of each polynomial (halves on each bind)
        /// This tracks the "active" portion of the polynomial after binding.
        effective_len: usize,
        allocator: Allocator,

        pub fn init(allocator: Allocator, num_suffixes: usize, poly_size: usize) !Self {
            const polys = try allocator.alloc([]F, num_suffixes);
            for (polys) |*poly| {
                poly.* = try allocator.alloc(F, poly_size);
                @memset(poly.*, F.zero());
            }
            return .{
                .polys = polys,
                .num_suffixes = num_suffixes,
                .effective_len = poly_size,
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            for (self.polys) |poly| {
                self.allocator.free(poly);
            }
            self.allocator.free(self.polys);
        }

        /// Get the polynomial for a specific suffix
        pub fn getSuffix(self: *const Self, suffix_idx: usize) []F {
            return self.polys[suffix_idx];
        }

        /// Get the effective length of the polynomial (may be smaller than allocated)
        pub fn getEffectiveLen(self: *const Self) usize {
            return self.effective_len;
        }

        /// Reset the effective length (used after re-initialization at phase transitions)
        pub fn resetEffectiveLen(self: *Self, new_len: usize) void {
            self.effective_len = new_len;
        }

        /// Bind a challenge (halves the effective polynomial size)
        /// Uses HighToLow binding order: new[i] = left[i] + r * (right[i] - left[i])
        /// where left = poly[0..n], right = poly[n..2n]
        pub fn bind(self: *Self, r: F) void {
            const half_size = self.effective_len / 2;
            for (self.polys) |poly| {
                // HighToLow: left half [0..half_size], right half [half_size..effective_len]
                for (0..half_size) |j| {
                    const low = poly[j];
                    const high = poly[j + half_size];
                    poly[j] = low.add(r.mul(high.sub(low)));
                }
            }
            self.effective_len = half_size;
        }
    };
}

/// All suffix polynomials for all tables
pub fn AllSuffixPolys(comptime F: type) type {
    return struct {
        const Self = @This();

        /// Suffix polynomials per table
        tables: [NUM_TABLES]?TableSuffixPolys(F),
        allocator: Allocator,

        pub fn init(allocator: Allocator) Self {
            return .{
                .tables = [_]?TableSuffixPolys(F){null} ** NUM_TABLES,
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            for (&self.tables) |*maybe_table| {
                if (maybe_table.*) |*table| {
                    table.deinit();
                }
            }
        }

        /// Initialize suffix polynomials for a specific phase
        pub fn initPhase(
            self: *Self,
            phase: usize,
            phases: usize,
            u_evals: []const F,
            lookup_indices: []const u128,
            cycle_table_indices: []const i8,
        ) !void {
            const log_m = LOG_K / phases;
            const m: usize = @as(usize, 1) << @intCast(log_m);
            const m_mask: u128 = m - 1;
            const suffix_len = LOG_K - (phase + 1) * log_m;

            // Initialize each table's suffix polynomials
            for (0..NUM_TABLES) |table_idx| {
                const table_suffixes = tableSuffixes(table_idx);
                const num_suffixes = table_suffixes.len;

                // Allocate if not already done
                if (self.tables[table_idx] == null) {
                    self.tables[table_idx] = try TableSuffixPolys(F).init(
                        self.allocator,
                        num_suffixes,
                        m,
                    );
                } else {
                    // Reset existing polynomials and restore effective length
                    for (self.tables[table_idx].?.polys) |poly| {
                        @memset(poly[0..m], F.zero());
                    }
                    self.tables[table_idx].?.resetEffectiveLen(m);
                }
            }

            // Accumulate contributions from each cycle
            var cycles_processed: usize = 0;
            var non_zero_suffix_mle: usize = 0;
            for (0..u_evals.len) |j| {
                const table_idx = cycle_table_indices[j];
                if (table_idx < 0) continue; // No table for this cycle

                const t_idx: usize = @intCast(table_idx);
                if (t_idx >= NUM_TABLES) continue;

                cycles_processed += 1;

                const k = lookup_indices[j];
                const prefix_bits = (k >> @intCast(suffix_len)) & m_mask;
                const suffix_bits_raw = k & ((@as(u128, 1) << @intCast(suffix_len)) - 1);
                const suffix_bits = LookupBits(128).new(suffix_bits_raw, suffix_len);

                const u = u_evals[j];
                const table_suffixes = tableSuffixes(t_idx);

                // Debug: print first few cycles in phase 0
                if (phase == 0 and cycles_processed <= 5) {
                    dbg("[SUFFIX INIT] cycle j={} t_idx={} k=0x{x:0>32} prefix_bits={} suffix_len={} suffix_raw=0x{x:0>32}\n", .{
                        j, t_idx, k, prefix_bits, suffix_len, suffix_bits_raw,
                    });
                }

                // Accumulate for each suffix type
                for (table_suffixes, 0..) |suffix, s_idx| {
                    const t = suffixMle(suffix, suffix_bits);
                    if (t != 0) {
                        non_zero_suffix_mle += 1;
                        const q_poly = self.tables[t_idx].?.polys[s_idx];
                        const idx: usize = @intCast(prefix_bits);
                        if (suffixes_mod.is01Valued(suffix)) {
                            // {0,1}-valued: t is 1, just add u
                            q_poly[idx] = q_poly[idx].add(u);
                        } else {
                            // General suffix: multiply by t
                            q_poly[idx] = q_poly[idx].add(u.mul(F.fromU64(t)));
                        }
                        // Debug: print which suffix types contribute
                        if (phase == 0 and cycles_processed <= 5) {
                            dbg("  suffix[{}]={s} t={} idx={}\n", .{ s_idx, @tagName(suffix), t, idx });
                        }
                    }
                }
            }
            if (phase == 0) {
                dbg("[SUFFIX INIT] phase={}: cycles_processed={}, non_zero_suffix_mle={}\n", .{
                    phase, cycles_processed, non_zero_suffix_mle,
                });
            }
        }

        /// Bind a challenge in all suffix polynomials
        pub fn bindAll(self: *Self, r: F) void {
            for (&self.tables) |*maybe_table| {
                if (maybe_table.*) |*table| {
                    table.bind(r);
                }
            }
        }
    };
}

/// Prefix checkpoints for all prefix types
pub fn PrefixCheckpointsState(comptime F: type) type {
    return struct {
        const Self = @This();

        /// Checkpoint value for each prefix type
        checkpoints: [Prefixes.COUNT]?F,

        pub fn init() Self {
            return .{
                .checkpoints = [_]?F{null} ** Prefixes.COUNT,
            };
        }

        /// Update checkpoints after binding two rounds (r_x, r_y)
        pub fn update(self: *Self, r_x: F, r_y: F, round: usize, suffix_len: usize) void {
            for (0..Prefixes.COUNT) |i| {
                const prefix: Prefixes = @enumFromInt(i);
                self.checkpoints[i] = prefixes_mod.updatePrefixCheckpoint(
                    F,
                    prefix,
                    &self.checkpoints,
                    r_x,
                    r_y,
                    round,
                    suffix_len,
                );
            }
        }
    };
}

/// Compute prover message for read-checking (address rounds)
/// Returns [eval_0, eval_2] for the degree-2 polynomial
pub fn proverMsgReadChecking(
    comptime F: type,
    round: usize,
    suffix_polys: *const AllSuffixPolys(F),
    prefix_checkpoints: *const PrefixCheckpointsState(F),
    r_x: ?F,
) [2]F {
    const len = blk: {
        // Find the current EFFECTIVE Q length from any initialized table
        // This reflects the length after binding (halves each round)
        for (suffix_polys.tables) |maybe_table| {
            if (maybe_table) |table| {
                if (table.effective_len > 0) {
                    break :blk table.effective_len;
                }
            }
        }
        break :blk 1; // Fallback
    };

    // Debug: count non-zero Q values for round 0
    if (round == 0) {
        var total_non_zero: usize = 0;
        var non_zero_tables: usize = 0;
        for (0..NUM_TABLES) |table_idx| {
            if (suffix_polys.tables[table_idx]) |table| {
                var table_non_zero: usize = 0;
                for (table.polys) |poly| {
                    for (poly) |v| {
                        if (!v.eql(F.zero())) table_non_zero += 1;
                    }
                }
                if (table_non_zero > 0) {
                    non_zero_tables += 1;
                    total_non_zero += table_non_zero;
                    dbg("[READ_CHECK ROUND 0] table {} has {} non-zero Q values\n", .{ table_idx, table_non_zero });

                    // Print ALL non-zero Q values with indices
                    {
                        const suffixes_list = tableSuffixes(table_idx);
                        var right_half_nonzero: usize = 0;
                        for (table.polys, 0..) |poly, s_idx| {
                            for (0..len) |idx| {
                                if (!poly[idx].eql(F.zero())) {
                                    const in_right = idx >= len / 2;
                                    if (in_right) right_half_nonzero += 1;
                                    dbg("  T{}:Q[{s}][{}] = {x} {s}\n", .{
                                        table_idx, @tagName(suffixes_list[s_idx]), idx,
                                        poly[idx].toBytesBE()[24..32].*,
                                        if (in_right) "RIGHT_HALF!" else "",
                                    });
                                }
                            }
                        }
                        if (right_half_nonzero > 0) {
                            dbg("[READ_CHECK ROUND 0] TABLE {} HAS {} RIGHT-HALF ENTRIES!\n", .{ table_idx, right_half_nonzero });
                        }
                    }
                }
            }
        }
        dbg("[READ_CHECK ROUND 0] Q poly stats: total_non_zero={}, non_zero_tables={}, len={}\n", .{
            total_non_zero, non_zero_tables, len,
        });
    }

    if (round < 3) {
        dbg("[READ_CHECK R{}] effective_len={}\n", .{ round, len });
    }

    const log_len = @ctz(len);
    const half_len = len / 2;

    var eval_0 = F.zero();
    var eval_2_left = F.zero();
    var eval_2_right = F.zero();
    var eval_0_per_table: [NUM_TABLES]F = [_]F{F.zero()} ** NUM_TABLES;

    // Sum over all remaining bits b
    for (0..half_len) |b_idx| {
        const b = LookupBits(128).new(@as(u128, b_idx), log_len - 1);

        // Compute prefix evaluations at c=0 and c=2 for all prefix types
        var prefixes_c0: [Prefixes.COUNT]F = undefined;
        var prefixes_c2: [Prefixes.COUNT]F = undefined;

        for (0..Prefixes.COUNT) |i| {
            const prefix: Prefixes = @enumFromInt(i);
            var b_copy = b;
            prefixes_c0[i] = prefixes_mod.prefixMle(F, prefix, &prefix_checkpoints.checkpoints, r_x, 0, &b_copy, round);
            b_copy = b;
            prefixes_c2[i] = prefixes_mod.prefixMle(F, prefix, &prefix_checkpoints.checkpoints, r_x, 2, &b_copy, round);
        }

        // Sum contributions from all tables
        for (0..NUM_TABLES) |table_idx| {
            if (suffix_polys.tables[table_idx]) |table| {
                const table_suffixes = tableSuffixes(table_idx);

                // Get suffix values at left and right positions
                var suffixes_left: [MAX_SUFFIXES_PER_TABLE]F = undefined;
                var suffixes_right: [MAX_SUFFIXES_PER_TABLE]F = undefined;

                for (table_suffixes, 0..) |_, s_idx| {
                    const poly = table.polys[s_idx];
                    suffixes_left[s_idx] = poly[b_idx];
                    suffixes_right[s_idx] = poly[b_idx + half_len];
                }

                // Combine using table-specific formula
                const combined_0 = tableCombine(F, table_idx, &prefixes_c0, suffixes_left[0..table_suffixes.len]);
                const combined_2_left = tableCombine(F, table_idx, &prefixes_c2, suffixes_left[0..table_suffixes.len]);
                const combined_2_right = tableCombine(F, table_idx, &prefixes_c2, suffixes_right[0..table_suffixes.len]);

                eval_0 = eval_0.add(combined_0);
                eval_0_per_table[table_idx] = eval_0_per_table[table_idx].add(combined_0);
                eval_2_left = eval_2_left.add(combined_2_left);
                eval_2_right = eval_2_right.add(combined_2_right);
            }
        }
    }

    // Debug: print per-table eval_0 at round 0
    if (round == 0) {
        var sum_per_table = F.zero();
        for (0..NUM_TABLES) |t_idx| {
            if (!eval_0_per_table[t_idx].eql(F.zero())) {
                dbg("[READ_CHECK R0] eval_0_per_table[{}]={x}\n", .{ t_idx, eval_0_per_table[t_idx].toBytesBE()[16..32].* });
                sum_per_table = sum_per_table.add(eval_0_per_table[t_idx]);
            }
        }
        dbg("[READ_CHECK R0] sum_per_table={x}\n", .{sum_per_table.toBytesBE()[16..32].*});
        dbg("[READ_CHECK R0] eval_0={x}\n", .{eval_0.toBytesBE()[16..32].*});
        dbg("[READ_CHECK R0] sum==eval_0: {}\n", .{sum_per_table.eql(eval_0)});
    }

    // Quadratic interpolation: eval_2 = 2*eval_2_right - eval_2_left
    const eval_2 = eval_2_right.add(eval_2_right).sub(eval_2_left);

    if (round == 0) {
        dbg("[READ_CHECK R0] eval_2_left={x}\n", .{eval_2_left.toBytesBE()[16..32].*});
        dbg("[READ_CHECK R0] eval_2_right={x}\n", .{eval_2_right.toBytesBE()[16..32].*});
        dbg("[READ_CHECK R0] eval_2_right==0: {}\n", .{eval_2_right.eql(F.zero())});
        // Also compute eval_1 independently to verify the total sum
        var eval_1_indep = F.zero();
        for (0..half_len) |b_idx2| {
            const b2 = LookupBits(128).new(@as(u128, b_idx2), log_len - 1);
            var prefixes_c1: [Prefixes.COUNT]F = undefined;
            for (0..Prefixes.COUNT) |i| {
                const prefix2: Prefixes = @enumFromInt(i);
                var b_copy2 = b2;
                prefixes_c1[i] = prefixes_mod.prefixMle(F, prefix2, &prefix_checkpoints.checkpoints, r_x, 1, &b_copy2, round);
            }
            for (0..NUM_TABLES) |table_idx2| {
                if (suffix_polys.tables[table_idx2]) |table2| {
                    const ts2 = tableSuffixes(table_idx2);
                    var suf_l2: [MAX_SUFFIXES_PER_TABLE]F = undefined;
                    var suf_r2: [MAX_SUFFIXES_PER_TABLE]F = undefined;
                    for (ts2, 0..) |_, s_idx2| {
                        suf_l2[s_idx2] = table2.polys[s_idx2][b_idx2];
                        suf_r2[s_idx2] = table2.polys[s_idx2][b_idx2 + half_len];
                    }
                    // At c=1, suffix value is: Q_left * (1-1) + Q_right * 1 = Q_right  -- wait no
                    // Actually at c=1, the linear interpolation is: Q[b_idx] * (1 - 1) + Q[b_idx + half_len] * 1 = Q_right
                    // But that's only if binding convention is: new[i] = old[i]*(1-r) + old[i+half]*r
                    // At c=1: new[i] = old[i+half] = Q_right
                    const combined_1 = tableCombine(F, table_idx2, &prefixes_c1, suf_r2[0..ts2.len]);
                    eval_1_indep = eval_1_indep.add(combined_1);
                }
            }
        }
        const total_read_check = eval_0.add(eval_1_indep);
        dbg("[READ_CHECK ROUND 0] eval_0={x}\n", .{eval_0.toBytesBE()[16..32].*});
        dbg("[READ_CHECK ROUND 0] eval_1_indep={x}\n", .{eval_1_indep.toBytesBE()[16..32].*});
        dbg("[READ_CHECK ROUND 0] eval_0+eval_1={x}\n", .{total_read_check.toBytesBE()[16..32].*});
        dbg("[READ_CHECK ROUND 0] eval_2={x}\n", .{eval_2.toBytesBE()[16..32].*});
    }

    return .{ eval_0, eval_2 };
}

/// Table-specific combine function
/// Combines prefix and suffix evaluations according to each table's formula
/// Reference: jolt-core/src/zkvm/lookup_table/*.rs for each table's combine() implementation
fn tableCombine(comptime F: type, table_idx: usize, prefixes: []const F, suffixes: []const F) F {
    return switch (table_idx) {
        // 0: RangeCheck: prefixes[LowerWord] * one + lower_word
        0 => prefixes[@intFromEnum(Prefixes.LowerWord)].mul(suffixes[0]).add(suffixes[1]),
        // 1: RangeCheckAligned: (prefixes[LowerWord] * one + lower_word) - prefixes[Lsb] * lsb
        1 => blk: {
            const lower_word_contrib = prefixes[@intFromEnum(Prefixes.LowerWord)].mul(suffixes[0]).add(suffixes[1]);
            if (suffixes.len >= 3) {
                const lsb_contrib = prefixes[@intFromEnum(Prefixes.Lsb)].mul(suffixes[2]);
                break :blk lower_word_contrib.sub(lsb_contrib);
            }
            break :blk lower_word_contrib;
        },
        // 2: And: prefixes[And] * one + and
        2 => prefixes[@intFromEnum(Prefixes.And)].mul(suffixes[0]).add(suffixes[1]),
        // 3: Andn: prefixes[Andn] * one + andn
        3 => prefixes[@intFromEnum(Prefixes.Andn)].mul(suffixes[0]).add(suffixes[1]),
        // 4: Or: prefixes[Or] * one + or
        4 => prefixes[@intFromEnum(Prefixes.Or)].mul(suffixes[0]).add(suffixes[1]),
        // 5: Xor: prefixes[Xor] * one + xor
        5 => prefixes[@intFromEnum(Prefixes.Xor)].mul(suffixes[0]).add(suffixes[1]),
        // 6: Equal: prefixes[Eq] * eq
        6 => prefixes[@intFromEnum(Prefixes.Eq)].mul(suffixes[0]),
        // 7: SignedGreaterThanEqual: one + RightMsb*one - LeftMsb*one - LessThan*one - Eq*less_than
        7 => blk: {
            var result = suffixes[0]; // one
            result = result.add(prefixes[@intFromEnum(Prefixes.RightOperandMsb)].mul(suffixes[0]));
            result = result.sub(prefixes[@intFromEnum(Prefixes.LeftOperandMsb)].mul(suffixes[0]));
            result = result.sub(prefixes[@intFromEnum(Prefixes.LessThan)].mul(suffixes[0]));
            if (suffixes.len >= 2) {
                result = result.sub(prefixes[@intFromEnum(Prefixes.Eq)].mul(suffixes[1]));
            }
            break :blk result;
        },
        // 8: UnsignedGreaterThanEqual: one - LessThan*one - Eq*less_than
        8 => blk: {
            var result = suffixes[0]; // one
            result = result.sub(prefixes[@intFromEnum(Prefixes.LessThan)].mul(suffixes[0]));
            if (suffixes.len >= 2) {
                result = result.sub(prefixes[@intFromEnum(Prefixes.Eq)].mul(suffixes[1]));
            }
            break :blk result;
        },
        // 9: NotEqual: one - prefixes[Eq] * eq
        9 => blk: {
            if (suffixes.len >= 2) {
                break :blk suffixes[0].sub(prefixes[@intFromEnum(Prefixes.Eq)].mul(suffixes[1]));
            }
            break :blk suffixes[0];
        },
        // 10: SignedLessThan: LeftMsb*one - RightMsb*one + LessThan*one + Eq*less_than
        10 => blk: {
            var result = prefixes[@intFromEnum(Prefixes.LeftOperandMsb)].mul(suffixes[0]);
            result = result.sub(prefixes[@intFromEnum(Prefixes.RightOperandMsb)].mul(suffixes[0]));
            result = result.add(prefixes[@intFromEnum(Prefixes.LessThan)].mul(suffixes[0]));
            if (suffixes.len >= 2) {
                result = result.add(prefixes[@intFromEnum(Prefixes.Eq)].mul(suffixes[1]));
            }
            break :blk result;
        },
        // 11: UnsignedLessThan: LessThan*one + Eq*less_than
        11 => blk: {
            var result = prefixes[@intFromEnum(Prefixes.LessThan)].mul(suffixes[0]);
            if (suffixes.len >= 2) {
                result = result.add(prefixes[@intFromEnum(Prefixes.Eq)].mul(suffixes[1]));
            }
            break :blk result;
        },
        // 12: Movsign: (2^XLEN - 1) * LeftMsb * one
        12 => blk: {
            const ones: u64 = 0xFFFFFFFF_FFFFFFFF; // 2^64 - 1 for RV64
            break :blk F.fromU64(ones).mul(prefixes[@intFromEnum(Prefixes.LeftOperandMsb)]).mul(suffixes[0]);
        },
        // 13: UpperWord: prefixes[UpperWord] * one + upper_word
        13 => prefixes[@intFromEnum(Prefixes.UpperWord)].mul(suffixes[0]).add(suffixes[1]),
        // 14: LessThanEqual (UnsignedLessThanEqual): LessThan*one + Eq*less_than + Eq*eq
        14 => blk: {
            var result = prefixes[@intFromEnum(Prefixes.LessThan)].mul(suffixes[0]);
            if (suffixes.len >= 2) {
                result = result.add(prefixes[@intFromEnum(Prefixes.Eq)].mul(suffixes[1]));
            }
            if (suffixes.len >= 3) {
                result = result.add(prefixes[@intFromEnum(Prefixes.Eq)].mul(suffixes[2]));
            }
            break :blk result;
        },
        // 15: ValidSignedRemainder:
        // RightOperandIsZero*right_is_zero + PositiveRemainderEqualsDivisor*less_than
        // + PositiveRemainderLessThanDivisor*one + NegativeDivisorZeroRemainder*left_is_zero
        // + NegativeDivisorEqualsRemainder*greater_than + NegativeDivisorGreaterThanRemainder*one
        15 => blk: {
            // Suffixes: [one, right_operand_is_zero, less_than, left_operand_is_zero, greater_than]
            var result = F.zero();
            if (suffixes.len >= 1) {
                // PositiveRemainderLessThanDivisor*one + NegativeDivisorGreaterThanRemainder*one
                result = result.add(prefixes[@intFromEnum(Prefixes.PositiveRemainderLessThanDivisor)].mul(suffixes[0]));
                result = result.add(prefixes[@intFromEnum(Prefixes.NegativeDivisorGreaterThanRemainder)].mul(suffixes[0]));
            }
            if (suffixes.len >= 2) {
                result = result.add(prefixes[@intFromEnum(Prefixes.RightOperandIsZero)].mul(suffixes[1]));
            }
            if (suffixes.len >= 3) {
                result = result.add(prefixes[@intFromEnum(Prefixes.PositiveRemainderEqualsDivisor)].mul(suffixes[2]));
            }
            if (suffixes.len >= 4) {
                result = result.add(prefixes[@intFromEnum(Prefixes.NegativeDivisorZeroRemainder)].mul(suffixes[3]));
            }
            if (suffixes.len >= 5) {
                result = result.add(prefixes[@intFromEnum(Prefixes.NegativeDivisorEqualsRemainder)].mul(suffixes[4]));
            }
            break :blk result;
        },
        // 16: ValidUnsignedRemainder: RightOperandIsZero*right_is_zero + LessThan*one + Eq*less_than
        16 => blk: {
            // Suffixes: [one, right_operand_is_zero, less_than]
            var result = F.zero();
            if (suffixes.len >= 1) {
                result = result.add(prefixes[@intFromEnum(Prefixes.LessThan)].mul(suffixes[0]));
            }
            if (suffixes.len >= 2) {
                result = result.add(prefixes[@intFromEnum(Prefixes.RightOperandIsZero)].mul(suffixes[1]));
            }
            if (suffixes.len >= 3) {
                result = result.add(prefixes[@intFromEnum(Prefixes.Eq)].mul(suffixes[2]));
            }
            break :blk result;
        },
        // 17: ValidDiv0: one - LeftOperandIsZero*left_is_zero + DivByZero*div_by_zero
        17 => blk: {
            // Suffixes: [one, left_operand_is_zero, div_by_zero]
            var result = F.zero();
            if (suffixes.len >= 1) {
                result = result.add(suffixes[0]); // one
            }
            if (suffixes.len >= 2) {
                result = result.sub(prefixes[@intFromEnum(Prefixes.LeftOperandIsZero)].mul(suffixes[1]));
            }
            if (suffixes.len >= 3) {
                result = result.add(prefixes[@intFromEnum(Prefixes.DivByZero)].mul(suffixes[2]));
            }
            break :blk result;
        },
        // 18: HalfwordAlignment: one - Lsb*lsb
        18 => blk: {
            // Suffixes: [one, lsb]
            var result = F.zero();
            if (suffixes.len >= 1) {
                result = result.add(suffixes[0]); // one
            }
            if (suffixes.len >= 2) {
                result = result.sub(prefixes[@intFromEnum(Prefixes.Lsb)].mul(suffixes[1]));
            }
            break :blk result;
        },
        // 19: WordAlignment: TwoLsb*two_lsb
        19 => blk: {
            // Suffixes: [two_lsb]
            if (suffixes.len >= 1) {
                break :blk prefixes[@intFromEnum(Prefixes.TwoLsb)].mul(suffixes[0]);
            }
            break :blk F.zero();
        },
        // 20: LowerHalfWord: LowerHalfWord*one + lower_half_word
        20 => blk: {
            // Suffixes: [one, lower_half_word]
            var result = F.zero();
            if (suffixes.len >= 1) {
                result = prefixes[@intFromEnum(Prefixes.LowerHalfWord)].mul(suffixes[0]);
            }
            if (suffixes.len >= 2) {
                result = result.add(suffixes[1]);
            }
            break :blk result;
        },
        // 21: SignExtendHalfWord: LowerHalfWord*one + lower_half_word + SignExtensionUpperHalf*sign_ext
        21 => blk: {
            // Suffixes: [one, lower_half_word, sign_extension_upper_half]
            var result = F.zero();
            if (suffixes.len >= 1) {
                result = prefixes[@intFromEnum(Prefixes.LowerHalfWord)].mul(suffixes[0]);
            }
            if (suffixes.len >= 2) {
                result = result.add(suffixes[1]);
            }
            if (suffixes.len >= 3) {
                result = result.add(prefixes[@intFromEnum(Prefixes.SignExtensionUpperHalf)].mul(suffixes[2]));
            }
            break :blk result;
        },
        // 22: Pow2: Pow2*pow2
        22 => blk: {
            // Suffixes: [pow2]
            if (suffixes.len >= 1) {
                break :blk prefixes[@intFromEnum(Prefixes.Pow2)].mul(suffixes[0]);
            }
            break :blk F.zero();
        },
        // 23: Pow2W: Pow2W*pow2w
        23 => blk: {
            // Suffixes: [pow2w]
            if (suffixes.len >= 1) {
                break :blk prefixes[@intFromEnum(Prefixes.Pow2W)].mul(suffixes[0]);
            }
            break :blk F.zero();
        },
        // 24: ShiftRightBitmask: 2^XLEN * one - Pow2*pow2
        24 => blk: {
            // Suffixes: [one, pow2]
            const two_pow_xlen = F.fromU64(1).add(F.fromU64(0xFFFFFFFF_FFFFFFFF)); // 2^64
            var result = F.zero();
            if (suffixes.len >= 1) {
                result = two_pow_xlen.mul(suffixes[0]);
            }
            if (suffixes.len >= 2) {
                result = result.sub(prefixes[@intFromEnum(Prefixes.Pow2)].mul(suffixes[1]));
            }
            break :blk result;
        },
        // 25: VirtualRev8W: Rev8W*one + rev8w
        25 => blk: {
            // Suffixes: [one, rev8w]
            var result = F.zero();
            if (suffixes.len >= 1) {
                result = prefixes[@intFromEnum(Prefixes.Rev8W)].mul(suffixes[0]);
            }
            if (suffixes.len >= 2) {
                result = result.add(suffixes[1]);
            }
            break :blk result;
        },
        // 26: VirtualSRL: RightShift*right_shift_helper + right_shift
        26 => blk: {
            // Suffixes: [right_shift_helper, right_shift]
            var result = F.zero();
            if (suffixes.len >= 1) {
                result = prefixes[@intFromEnum(Prefixes.RightShift)].mul(suffixes[0]);
            }
            if (suffixes.len >= 2) {
                result = result.add(suffixes[1]);
            }
            break :blk result;
        },
        // 27: VirtualSRA: RightShift*helper + right_shift + LeftOperandMsb*sign_ext + SignExtension*one
        27 => blk: {
            // Suffixes: [right_shift_helper, right_shift, sign_extension, one]
            var result = F.zero();
            if (suffixes.len >= 1) {
                result = prefixes[@intFromEnum(Prefixes.RightShift)].mul(suffixes[0]);
            }
            if (suffixes.len >= 2) {
                result = result.add(suffixes[1]);
            }
            if (suffixes.len >= 3) {
                result = result.add(prefixes[@intFromEnum(Prefixes.LeftOperandMsb)].mul(suffixes[2]));
            }
            if (suffixes.len >= 4) {
                result = result.add(prefixes[@intFromEnum(Prefixes.SignExtension)].mul(suffixes[3]));
            }
            break :blk result;
        },
        // 28: VirtualROTR: RightShift*helper + right_shift + LeftShiftHelper*left_shift + LeftShift*one
        28 => blk: {
            // Suffixes: [right_shift_helper, right_shift, left_shift, one]
            var result = F.zero();
            if (suffixes.len >= 1) {
                result = prefixes[@intFromEnum(Prefixes.RightShift)].mul(suffixes[0]);
            }
            if (suffixes.len >= 2) {
                result = result.add(suffixes[1]);
            }
            if (suffixes.len >= 3) {
                result = result.add(prefixes[@intFromEnum(Prefixes.LeftShiftHelper)].mul(suffixes[2]));
            }
            if (suffixes.len >= 4) {
                result = result.add(prefixes[@intFromEnum(Prefixes.LeftShift)].mul(suffixes[3]));
            }
            break :blk result;
        },
        // 29: VirtualROTRW: RightShiftW*helper + right_shift_w + LeftShiftWHelper*left_shift_w + LeftShiftW*one
        29 => blk: {
            // Suffixes: [right_shift_w_helper, right_shift_w, left_shift_w, one]
            var result = F.zero();
            if (suffixes.len >= 1) {
                result = prefixes[@intFromEnum(Prefixes.RightShiftW)].mul(suffixes[0]);
            }
            if (suffixes.len >= 2) {
                result = result.add(suffixes[1]);
            }
            if (suffixes.len >= 3) {
                result = result.add(prefixes[@intFromEnum(Prefixes.LeftShiftWHelper)].mul(suffixes[2]));
            }
            if (suffixes.len >= 4) {
                result = result.add(prefixes[@intFromEnum(Prefixes.LeftShiftW)].mul(suffixes[3]));
            }
            break :blk result;
        },
        // 30: VirtualChangeDivisor: RightOperand*one + right_operand + ChangeDivisor*change_divisor
        30 => blk: {
            // Suffixes: [one, right_operand, change_divisor]
            var result = F.zero();
            if (suffixes.len >= 1) {
                result = prefixes[@intFromEnum(Prefixes.RightOperand)].mul(suffixes[0]);
            }
            if (suffixes.len >= 2) {
                result = result.add(suffixes[1]);
            }
            if (suffixes.len >= 3) {
                result = result.add(prefixes[@intFromEnum(Prefixes.ChangeDivisor)].mul(suffixes[2]));
            }
            break :blk result;
        },
        // 31: VirtualChangeDivisorW: RightOperandW*one + right_op_w + ChangeDivisorW*change + SignExtRightOp*sign_ext
        31 => blk: {
            // Suffixes: [one, right_operand_w, change_divisor_w, sign_extension]
            var result = F.zero();
            if (suffixes.len >= 1) {
                result = prefixes[@intFromEnum(Prefixes.RightOperandW)].mul(suffixes[0]);
            }
            if (suffixes.len >= 2) {
                result = result.add(suffixes[1]);
            }
            if (suffixes.len >= 3) {
                result = result.add(prefixes[@intFromEnum(Prefixes.ChangeDivisorW)].mul(suffixes[2]));
            }
            if (suffixes.len >= 4) {
                result = result.add(prefixes[@intFromEnum(Prefixes.SignExtensionRightOperand)].mul(suffixes[3]));
            }
            break :blk result;
        },
        // 32: MulUNoOverflow: OverflowBitsZero*overflow_bits_zero
        32 => blk: {
            // Suffixes: [overflow_bits_zero]
            if (suffixes.len >= 1) {
                break :blk prefixes[@intFromEnum(Prefixes.OverflowBitsZero)].mul(suffixes[0]);
            }
            break :blk F.zero();
        },
        // 33: VirtualXORROT32: XorRot32*one + xor_rot
        33 => blk: {
            // Suffixes: [one, xor_rot]
            var result = F.zero();
            if (suffixes.len >= 1) {
                result = prefixes[@intFromEnum(Prefixes.XorRot32)].mul(suffixes[0]);
            }
            if (suffixes.len >= 2) {
                result = result.add(suffixes[1]);
            }
            break :blk result;
        },
        // 34: VirtualXORROT24: XorRot24*one + xor_rot
        34 => blk: {
            var result = F.zero();
            if (suffixes.len >= 1) {
                result = prefixes[@intFromEnum(Prefixes.XorRot24)].mul(suffixes[0]);
            }
            if (suffixes.len >= 2) {
                result = result.add(suffixes[1]);
            }
            break :blk result;
        },
        // 35: VirtualXORROT16: XorRot16*one + xor_rot
        35 => blk: {
            var result = F.zero();
            if (suffixes.len >= 1) {
                result = prefixes[@intFromEnum(Prefixes.XorRot16)].mul(suffixes[0]);
            }
            if (suffixes.len >= 2) {
                result = result.add(suffixes[1]);
            }
            break :blk result;
        },
        // 36: VirtualXORROT63: XorRot63*one + xor_rot
        36 => blk: {
            var result = F.zero();
            if (suffixes.len >= 1) {
                result = prefixes[@intFromEnum(Prefixes.XorRot63)].mul(suffixes[0]);
            }
            if (suffixes.len >= 2) {
                result = result.add(suffixes[1]);
            }
            break :blk result;
        },
        // 37: VirtualXORROTW16: XorRotW16*one + xor_rot
        37 => blk: {
            var result = F.zero();
            if (suffixes.len >= 1) {
                result = prefixes[@intFromEnum(Prefixes.XorRotW16)].mul(suffixes[0]);
            }
            if (suffixes.len >= 2) {
                result = result.add(suffixes[1]);
            }
            break :blk result;
        },
        // 38: VirtualXORROTW12: XorRotW12*one + xor_rot
        38 => blk: {
            var result = F.zero();
            if (suffixes.len >= 1) {
                result = prefixes[@intFromEnum(Prefixes.XorRotW12)].mul(suffixes[0]);
            }
            if (suffixes.len >= 2) {
                result = result.add(suffixes[1]);
            }
            break :blk result;
        },
        // 39: VirtualXORROTW8: XorRotW8*one + xor_rot
        39 => blk: {
            var result = F.zero();
            if (suffixes.len >= 1) {
                result = prefixes[@intFromEnum(Prefixes.XorRotW8)].mul(suffixes[0]);
            }
            if (suffixes.len >= 2) {
                result = result.add(suffixes[1]);
            }
            break :blk result;
        },
        // 40: VirtualXORROTW7: XorRotW7*one + xor_rot
        40 => blk: {
            var result = F.zero();
            if (suffixes.len >= 1) {
                result = prefixes[@intFromEnum(Prefixes.XorRotW7)].mul(suffixes[0]);
            }
            if (suffixes.len >= 2) {
                result = result.add(suffixes[1]);
            }
            break :blk result;
        },
        // Fallback for any undefined tables
        else => F.zero(),
    };
}

/// RAF (Read-Address-Flag) Decomposition State
/// RAF Polynomial Type - determines binding behavior
pub const RafPolyType = enum {
    LeftOperand,
    RightOperand,
    Identity,
};

/// This handles the identity/operand polynomial decomposition for RAF sumcheck
/// Tracks bound_value for proper prefix polynomial evaluation.
///
/// For OperandPolynomial:
/// - Left binds on even rounds (0, 2, 4, ...): bound_value = 2*bound_value + r
/// - Right binds on odd rounds (1, 3, 5, ...): bound_value = 2*bound_value + r
///
/// For IdentityPolynomial:
/// - Always binds: bound_value = 2*bound_value + r
pub fn RafDecomposition(comptime F: type) type {
    return struct {
        const Self = @This();

        /// Q accumulators: [shift_suffix, operand/identity_suffix]
        Q: [2][]F,
        /// Current Q size
        Q_size: usize,
        /// Total number of rounds (LOG_K = 128)
        total_len: usize,
        /// Rounds per phase (LOG_K / phases = 16)
        chunk_len: usize,
        /// Current phase (0..phases)
        phase: usize,
        /// Current round (global, 0..LOG_K)
        round: usize,
        /// Number of bound variables (global, across all phases)
        num_bound_vars: usize,
        /// Bound value (accumulated from challenges, used as checkpoint for prefix)
        bound_value: F,
        /// Type of RAF polynomial (determines binding behavior)
        poly_type: RafPolyType,
        /// Allocator
        allocator: Allocator,
        /// Materialized prefix MLE table (size = 2^chunk_len at phase start, halves each round)
        prefix_mle: []F,
        /// Current prefix MLE size (halves each round within a phase)
        prefix_mle_size: usize,
        /// Maximum prefix MLE size (2^chunk_len, allocated once)
        prefix_mle_max_size: usize,

        pub fn init(allocator: Allocator, initial_size: usize, chunk_len: usize, total_len: usize, poly_type: RafPolyType) !Self {
            var Q: [2][]F = undefined;
            Q[0] = try allocator.alloc(F, initial_size);
            Q[1] = try allocator.alloc(F, initial_size);
            @memset(Q[0], F.zero());
            @memset(Q[1], F.zero());

            // Allocate prefix MLE table: 2^chunk_len entries
            const prefix_max = @as(usize, 1) << @intCast(chunk_len);
            const prefix_mle = try allocator.alloc(F, prefix_max);
            @memset(prefix_mle, F.zero());

            return .{
                .Q = Q,
                .Q_size = initial_size,
                .total_len = total_len,
                .chunk_len = chunk_len,
                .phase = 0,
                .round = 0,
                .num_bound_vars = 0,
                .bound_value = F.zero(),
                .poly_type = poly_type,
                .allocator = allocator,
                .prefix_mle = prefix_mle,
                .prefix_mle_size = prefix_max,
                .prefix_mle_max_size = prefix_max,
            };
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.Q[0]);
            self.allocator.free(self.Q[1]);
            self.allocator.free(self.prefix_mle);
        }

        pub fn QLen(self: *const Self) usize {
            return self.Q_size;
        }

        /// Get suffix length for current phase
        pub fn suffixLen(self: *const Self) usize {
            return self.total_len - (self.phase + 1) * self.chunk_len;
        }

        /// Reset Q accumulators for new phase
        pub fn resetQ(self: *Self) void {
            @memset(self.Q[0], F.zero());
            @memset(self.Q[1], F.zero());
        }

        /// Reset for a new phase: restore Q_size to initial and reset accumulators
        /// Call this at phase transitions before calling initQRaf
        /// NOTE: Preserves num_bound_vars, round, and bound_value which track the prefix state
        pub fn resetForPhase(self: *Self, new_phase: usize, initial_size: usize) void {
            self.Q_size = initial_size;
            self.phase = new_phase;
            // Zero out all allocated elements, not just Q_size
            @memset(self.Q[0][0..initial_size], F.zero());
            @memset(self.Q[1][0..initial_size], F.zero());
            // NOTE: Do NOT reset num_bound_vars, round, or bound_value!
            // These track the accumulated prefix state across phases.
        }

        /// Should this polynomial bind on the current round?
        fn shouldBind(self: *const Self) bool {
            const is_even = (self.num_bound_vars % 2 == 0);
            return switch (self.poly_type) {
                .LeftOperand => is_even, // Left binds on even rounds
                .RightOperand => !is_even, // Right binds on odd rounds
                .Identity => true, // Identity always binds
            };
        }

        /// Materialize the prefix MLE table at the start of a new phase.
        /// This creates a full evaluation table of 2^chunk_len entries matching
        /// Jolt's CachedPolynomial / prefix_polynomial approach.
        ///
        /// For OperandPolynomial:
        ///   P[i] = bound_value * 2^(chunk_len/2) + uninterleave(i, side)
        /// For IdentityPolynomial:
        ///   P[i] = bound_value * 2^chunk_len + i
        pub fn initPrefix(self: *Self) void {
            const size = self.prefix_mle_max_size; // 2^chunk_len
            self.prefix_mle_size = size;

            switch (self.poly_type) {
                .LeftOperand, .RightOperand => {
                    const is_left = (self.poly_type == .LeftOperand);
                    const half_chunk = self.chunk_len / 2;
                    const scale = fieldPow2(F, half_chunk); // 2^(chunk_len/2)
                    const base = self.bound_value.mul(scale);

                    for (0..size) |i| {
                        // Uninterleave to get operand bits
                        // Left uses odd positions (1,3,5,...), Right uses even positions (0,2,4,...)
                        var operand_val: u64 = 0;
                        for (0..@min(half_chunk, 32)) |bit_idx| {
                            if (is_left) {
                                const bit: u64 = @truncate((i >> @intCast(2 * bit_idx + 1)) & 1);
                                operand_val |= bit << @intCast(bit_idx);
                            } else {
                                const bit: u64 = @truncate((i >> @intCast(2 * bit_idx)) & 1);
                                operand_val |= bit << @intCast(bit_idx);
                            }
                        }
                        self.prefix_mle[i] = base.add(F.fromU64(operand_val));
                    }
                },
                .Identity => {
                    const scale = fieldPow2(F, self.chunk_len); // 2^chunk_len
                    const base = self.bound_value.mul(scale);

                    for (0..size) |i| {
                        self.prefix_mle[i] = base.add(F.fromU64(@intCast(i)));
                    }
                },
            }
        }

        /// Save the prefix checkpoint (final single value after all chunk_len rounds)
        /// and update bound_value for the next phase.
        /// Call this at phase boundaries after all rounds in the chunk are done.
        pub fn updateCheckpoint(self: *Self) void {
            // After chunk_len rounds of binding, prefix_mle_size should be 1
            // The single remaining value IS the checkpoint
            std.debug.assert(self.prefix_mle_size == 1);
            self.bound_value = self.prefix_mle[0];
        }

        /// Bind a challenge to Q polynomials AND prefix MLE table
        /// Uses HighToLow binding order: new[j] = old[j] + r * (old[j+half] - old[j])
        pub fn bind(self: *Self, r: F) void {
            // Bind Q polynomials
            const half_size = self.Q_size / 2;
            for (0..2) |i| {
                for (0..half_size) |j| {
                    const low = self.Q[i][j];
                    const high = self.Q[i][j + half_size];
                    self.Q[i][j] = low.add(r.mul(high.sub(low)));
                }
            }
            self.Q_size = half_size;

            // Bind prefix MLE table (HighToLow)
            const prefix_half = self.prefix_mle_size / 2;
            for (0..prefix_half) |j| {
                const low = self.prefix_mle[j];
                const high = self.prefix_mle[j + prefix_half];
                self.prefix_mle[j] = low.add(r.mul(high.sub(low)));
            }
            self.prefix_mle_size = prefix_half;

            self.num_bound_vars += 1;
            self.round += 1;
            if (self.round % self.chunk_len == 0) {
                self.phase += 1;
            }
        }

        /// Compute sumcheck evaluations at index b using the materialized prefix MLE table.
        /// Returns (P(0), P(2)) where P is the prefix polynomial.
        ///
        /// Uses standard MLE sumcheck_evals (table lookup + linear interpolation):
        ///   P(0) = prefix_mle[b]
        ///   P(1) = prefix_mle[b + half_len]
        ///   m = P(1) - P(0)
        ///   P(2) = P(1) + m = 2*P(1) - P(0)
        pub fn prefixEvals(self: *const Self, b: usize) [2]F {
            const half_len = self.prefix_mle_size / 2;
            const eval_0 = self.prefix_mle[b];
            const eval_1 = self.prefix_mle[b + half_len];
            // Linear extrapolation: P(2) = P(1) + (P(1) - P(0)) = 2*P(1) - P(0)
            const eval_2 = eval_1.add(eval_1).sub(eval_0);
            return .{ eval_0, eval_2 };
        }
    };
}

/// Initialize Q accumulators for all three RAF decompositions (left, right, identity)
/// This is a fused initialization matching Jolt's init_Q_raf
pub fn initQRaf(
    comptime F: type,
    left: *RafDecomposition(F),
    right: *RafDecomposition(F),
    identity: *RafDecomposition(F),
    u_evals: []const F,
    lookup_indices: []const u128,
    is_interleaved_operands: []const bool,
) void {
    std.debug.assert(left.Q_size == right.Q_size);
    std.debug.assert(left.Q_size == identity.Q_size);

    const poly_len = left.Q_size;
    const suffix_len = left.suffixLen();
    const half_suffix_len = suffix_len / 2;

    // Constants for this phase
    const shift_half: u128 = @as(u128, 1) << @intCast(half_suffix_len);
    const shift_full: u128 = @as(u128, 1) << @intCast(suffix_len);
    const shift_half_f = F.fromU128(shift_half);
    const shift_full_f = F.fromU128(shift_full);

    // Reset all Q accumulators
    left.resetQ();
    right.resetQ();
    identity.resetQ();

    // Accumulators for the 5 distinct Q components:
    // - sh: ShiftHalfSuffix for operands (left.Q[0], right.Q[0])
    // - l: Left operand suffix (left.Q[1])
    // - r: Right operand suffix (right.Q[1])
    // - sf: ShiftFullSuffix for identity (identity.Q[0])
    // - id: Identity suffix (identity.Q[1])

    for (lookup_indices, 0..) |k, j| {
        const u = u_evals[j];
        const suffix_bits = k & ((@as(u128, 1) << @intCast(suffix_len)) - 1);
        const prefix_bits = (k >> @intCast(suffix_len)) & (@as(u128, poly_len) - 1);
        const r_index: usize = @intCast(prefix_bits);

        if (is_interleaved_operands[j]) {
            // Operand path: accumulate ShiftHalf * u and operand suffix * u
            // ShiftHalf contribution: u * 2^{suffix_len/2}
            left.Q[0][r_index] = left.Q[0][r_index].add(u.mul(shift_half_f));
            right.Q[0][r_index] = right.Q[0][r_index].add(u.mul(shift_half_f));

            // Uninterleave suffix bits to get left and right operand suffixes
            const lo_bits = uninterleaveBitsLeft(suffix_bits, suffix_len);
            const ro_bits = uninterleaveBitsRight(suffix_bits, suffix_len);

            if (lo_bits != 0) {
                left.Q[1][r_index] = left.Q[1][r_index].add(u.mul(F.fromU64(lo_bits)));
            }
            if (ro_bits != 0) {
                right.Q[1][r_index] = right.Q[1][r_index].add(u.mul(F.fromU64(ro_bits)));
            }
        } else {
            // Identity path: accumulate ShiftFull * u and identity suffix * u
            // ShiftFull contribution: u * 2^{suffix_len}
            identity.Q[0][r_index] = identity.Q[0][r_index].add(u.mul(shift_full_f));

            // Identity suffix contribution
            if (suffix_bits != 0) {
                if (suffix_len <= 64) {
                    identity.Q[1][r_index] = identity.Q[1][r_index].add(u.mul(F.fromU64(@truncate(suffix_bits))));
                } else {
                    identity.Q[1][r_index] = identity.Q[1][r_index].add(u.mul(F.fromU128(suffix_bits)));
                }
            }
        }
    }
}

/// Uninterleave bits to get the left operand
/// In Jolt's interleave format: interleaved = (left << 1) | right
/// So left operand bits are at ODD positions (1, 3, 5, ...)
fn uninterleaveBitsLeft(bits: u128, num_bits: usize) u64 {
    var left: u64 = 0;
    const half_bits = num_bits / 2;
    var i: usize = 0;
    while (i < half_bits and i < 64) : (i += 1) {
        // Left operand at ODD positions (1, 3, 5, ...)
        const shift_amt: u7 = @intCast(@min(2 * i + 1, 127));
        const bit = (bits >> shift_amt) & 1;
        left |= @as(u64, @truncate(bit)) << @as(u6, @intCast(i));
    }
    return left;
}

/// Uninterleave bits to get the right operand
/// In Jolt's interleave format: interleaved = (left << 1) | right
/// So right operand bits are at EVEN positions (0, 2, 4, ...)
fn uninterleaveBitsRight(bits: u128, num_bits: usize) u64 {
    var right: u64 = 0;
    const half_bits = num_bits / 2;
    var i: usize = 0;
    while (i < half_bits and i < 64) : (i += 1) {
        // Right operand at EVEN positions (0, 2, 4, ...)
        const shift_amt: u7 = @intCast(@min(2 * i, 127));
        const bit = (bits >> shift_amt) & 1;
        right |= @as(u64, @truncate(bit)) << @as(u6, @intCast(i));
    }
    return right;
}

/// Compute prover message for RAF (Read-Address-Flag) contribution
/// Returns [eval_0, eval_2] for the degree-2 polynomial
///
/// This computes: γ*left + γ²*(identity + right)
/// Where left, right, identity are prefix-suffix decompositions.
///
/// Jolt's formula (per index b):
///   For each (P, Q) pair in decomposition:
///     p_evals = P.sumcheck_evals(b)  // (P(0), P(2))
///     eval_0 += p_evals.0 * Q[b]
///     eval_2_left += p_evals.1 * Q[b]
///     eval_2_right += p_evals.1 * Q[b + half_len]
///   final: (eval_0, 2*eval_2_right - eval_2_left)
///
/// For RAF, each decomposition has 2 (P, Q) pairs:
/// - (Operand/Identity prefix P, Q[0]) - prefix depends on bound_value and round
/// - (None=constant 1, Q[1]) - prefix is constant 1
///
/// Reference: jolt-core/src/zkvm/instruction_lookups/read_raf_checking.rs:932
pub fn proverMsgRaf(
    comptime F: type,
    left_ps: *const RafDecomposition(F),
    right_ps: *const RafDecomposition(F),
    identity_ps: *const RafDecomposition(F),
    gamma: F,
    gamma_sqr: F,
) [2]F {
    const len = identity_ps.QLen();
    const half_len = len / 2;

    // Debug: show state at round 0 and 1
    if (left_ps.round == 0 or left_ps.round == 1) {
        dbg("[RAF_DEBUG R{}] Q_size={}, bound_value_left={x}\n", .{
            left_ps.round,
            len,
            left_ps.bound_value.toBytesBE()[16..32].*,
        });
        dbg("[RAF_DEBUG R{}] bound_value_right={x}, bound_value_identity={x}\n", .{
            left_ps.round,
            right_ps.bound_value.toBytesBE()[16..32].*,
            identity_ps.bound_value.toBytesBE()[16..32].*,
        });

        // Print Q array sums for comparison with Jolt
        var left_q0_sum = F.zero();
        var left_q1_sum = F.zero();
        var right_q0_sum = F.zero();
        var right_q1_sum = F.zero();
        var identity_q0_sum = F.zero();
        var identity_q1_sum = F.zero();
        for (0..len) |b| {
            left_q0_sum = left_q0_sum.add(left_ps.Q[0][b]);
            left_q1_sum = left_q1_sum.add(left_ps.Q[1][b]);
            right_q0_sum = right_q0_sum.add(right_ps.Q[0][b]);
            right_q1_sum = right_q1_sum.add(right_ps.Q[1][b]);
            identity_q0_sum = identity_q0_sum.add(identity_ps.Q[0][b]);
            identity_q1_sum = identity_q1_sum.add(identity_ps.Q[1][b]);
        }
        dbg("[RAF_DEBUG R{}] Q_SUM: left[0]={x}, left[1]={x}\n", .{
            left_ps.round,
            left_q0_sum.toBytesBE()[16..32].*,
            left_q1_sum.toBytesBE()[16..32].*,
        });
        dbg("[RAF_DEBUG R{}] Q_SUM: right[0]={x}, right[1]={x}\n", .{
            left_ps.round,
            right_q0_sum.toBytesBE()[16..32].*,
            right_q1_sum.toBytesBE()[16..32].*,
        });
        dbg("[RAF_DEBUG R{}] Q_SUM: identity[0]={x}, identity[1]={x}\n", .{
            left_ps.round,
            identity_q0_sum.toBytesBE()[16..32].*,
            identity_q1_sum.toBytesBE()[16..32].*,
        });

        // Print Q values at index 0 specifically
        dbg("[RAF_DEBUG R{}] Q_AT_0: left_Q0[0]={x}, left_Q1[0]={x}\n", .{
            left_ps.round,
            left_ps.Q[0][0].toBytesBE()[16..32].*,
            left_ps.Q[1][0].toBytesBE()[16..32].*,
        });
        dbg("[RAF_DEBUG R{}] Q_AT_0: right_Q1[0]={x}, identity_Q0[0]={x}, identity_Q1[0]={x}\n", .{
            left_ps.round,
            right_ps.Q[1][0].toBytesBE()[16..32].*,
            identity_ps.Q[0][0].toBytesBE()[16..32].*,
            identity_ps.Q[1][0].toBytesBE()[16..32].*,
        });

        // Print prefix MLE values for debugging
        dbg("[RAF_DEBUG R{}] left prefix_mle_size={}, prefix_mle[0]={x}\n", .{
            left_ps.round,
            left_ps.prefix_mle_size,
            left_ps.prefix_mle[0].toBytesBE()[16..32].*,
        });
        if (left_ps.prefix_mle_size >= 2) {
            const lhalf = left_ps.prefix_mle_size / 2;
            dbg("[RAF_DEBUG R{}] left prefix_mle[half={d}]={x}\n", .{
                left_ps.round,
                lhalf,
                left_ps.prefix_mle[lhalf].toBytesBE()[16..32].*,
            });
        }
        dbg("[RAF_DEBUG R{}] identity prefix_mle_size={}, prefix_mle[0]={x}\n", .{
            left_ps.round,
            identity_ps.prefix_mle_size,
            identity_ps.prefix_mle[0].toBytesBE()[16..32].*,
        });
        if (identity_ps.prefix_mle_size >= 2) {
            const ihalf = identity_ps.prefix_mle_size / 2;
            dbg("[RAF_DEBUG R{}] identity prefix_mle[half={d}]={x}\n", .{
                left_ps.round,
                ihalf,
                identity_ps.prefix_mle[ihalf].toBytesBE()[16..32].*,
            });
        }

        // Print prefix evals at b=0
        const l_pf_0 = left_ps.prefixEvals(0);
        const r_pf_0 = right_ps.prefixEvals(0);
        const i_pf_0 = identity_ps.prefixEvals(0);
        dbg("[RAF_DEBUG R{}] prefix_evals(0): left=({x}, {x})\n", .{
            left_ps.round,
            l_pf_0[0].toBytesBE()[16..32].*,
            l_pf_0[1].toBytesBE()[16..32].*,
        });
        dbg("[RAF_DEBUG R{}] prefix_evals(0): right=({x}, {x})\n", .{
            left_ps.round,
            r_pf_0[0].toBytesBE()[16..32].*,
            r_pf_0[1].toBytesBE()[16..32].*,
        });
        dbg("[RAF_DEBUG R{}] prefix_evals(0): identity=({x}, {x})\n", .{
            left_ps.round,
            i_pf_0[0].toBytesBE()[16..32].*,
            i_pf_0[1].toBytesBE()[16..32].*,
        });

        // Compute explicit sum to verify
        var explicit_left_sum_0 = F.zero();
        var explicit_right_sum_0 = F.zero();
        for (0..half_len) |b| {
            const l_prefix_b = left_ps.prefixEvals(b);
            const r_prefix_b = right_ps.prefixEvals(b);
            const i_prefix_b = identity_ps.prefixEvals(b);

            // left contribution: prefix * Q0 + 1 * Q1
            const l_contrib = l_prefix_b[0].mul(left_ps.Q[0][b]).add(left_ps.Q[1][b]);
            explicit_left_sum_0 = explicit_left_sum_0.add(l_contrib);

            // right+identity contribution
            const r_contrib = r_prefix_b[0].mul(right_ps.Q[0][b]).add(right_ps.Q[1][b]);
            const i_contrib = i_prefix_b[0].mul(identity_ps.Q[0][b]).add(identity_ps.Q[1][b]);
            explicit_right_sum_0 = explicit_right_sum_0.add(r_contrib).add(i_contrib);
        }
        const explicit_raf_0 = gamma.mul(explicit_left_sum_0).add(gamma_sqr.mul(explicit_right_sum_0));
        dbg("[RAF_DEBUG R{}] explicit_raf_0={x} (should match raf_evals[0])\n", .{
            left_ps.round,
            explicit_raf_0.toBytesBE()[16..32].*,
        });
    }

    // Accumulators for the sums
    var left_sum_0 = F.zero();
    var left_sum_2_left = F.zero();
    var left_sum_2_right = F.zero();
    var right_sum_0 = F.zero();
    var right_sum_2_left = F.zero();
    var right_sum_2_right = F.zero();

    // For each half-index b, compute sumcheck evaluations
    for (0..half_len) |b| {
        // Get Q values at left (b) and right (b + half_len) positions
        // Q[0] = shift*u accumulator (prefix = constant 1)
        // Q[1] = suffix*u accumulator (prefix = OperandPolynomial/IdentityPolynomial)
        const l_q0_left = left_ps.Q[0][b];
        const l_q0_right = left_ps.Q[0][b + half_len];
        const l_q1_left = left_ps.Q[1][b];
        const l_q1_right = left_ps.Q[1][b + half_len];

        const r_q0_left = right_ps.Q[0][b];
        const r_q0_right = right_ps.Q[0][b + half_len];
        const r_q1_left = right_ps.Q[1][b];
        const r_q1_right = right_ps.Q[1][b + half_len];

        const i_q0_left = identity_ps.Q[0][b];
        const i_q0_right = identity_ps.Q[0][b + half_len];
        const i_q1_left = identity_ps.Q[1][b];
        const i_q1_right = identity_ps.Q[1][b + half_len];

        // Compute prefix evaluations at c=0 and c=2
        // In Jolt's PrefixSuffixDecomposition with ORDER=2:
        //   P[0] = Some(prefix polynomial), Q[0] = shift suffix accumulator
        //   P[1] = None (constant 1),       Q[1] = operand/identity suffix accumulator
        // So prefix evals multiply Q[0], and constant 1 multiplies Q[1].
        const l_prefix = left_ps.prefixEvals(b);
        const r_prefix = right_ps.prefixEvals(b);
        const i_prefix = identity_ps.prefixEvals(b);

        // Left operand contribution: sum of (prefix, Q[0]) and (1, Q[1]) pairs
        // Pair 0: P = LeftOperand prefix, Q = Q[0] (shift half accumulator)
        const l_pair0_0 = l_prefix[0].mul(l_q0_left);
        const l_pair0_2_left = l_prefix[1].mul(l_q0_left);
        const l_pair0_2_right = l_prefix[1].mul(l_q0_right);
        // Pair 1: P = constant 1, Q = Q[1] (operand suffix accumulator)
        const l_pair1_0 = l_q1_left;
        const l_pair1_2_left = l_q1_left;
        const l_pair1_2_right = l_q1_right;

        // Right operand contribution
        // Pair 0: P = RightOperand prefix, Q = Q[0] (shift half accumulator)
        const r_pair0_0 = r_prefix[0].mul(r_q0_left);
        const r_pair0_2_left = r_prefix[1].mul(r_q0_left);
        const r_pair0_2_right = r_prefix[1].mul(r_q0_right);
        // Pair 1: P = constant 1, Q = Q[1] (operand suffix accumulator)
        const r_pair1_0 = r_q1_left;
        const r_pair1_2_left = r_q1_left;
        const r_pair1_2_right = r_q1_right;

        // Identity contribution
        // Pair 0: P = Identity prefix, Q = Q[0] (shift full accumulator)
        const i_pair0_0 = i_prefix[0].mul(i_q0_left);
        const i_pair0_2_left = i_prefix[1].mul(i_q0_left);
        const i_pair0_2_right = i_prefix[1].mul(i_q0_right);
        // Pair 1: P = constant 1, Q = Q[1] (identity suffix accumulator)
        const i_pair1_0 = i_q1_left;
        const i_pair1_2_left = i_q1_left;
        const i_pair1_2_right = i_q1_right;

        // Accumulate left operand totals
        left_sum_0 = left_sum_0.add(l_pair0_0).add(l_pair1_0);
        left_sum_2_left = left_sum_2_left.add(l_pair0_2_left).add(l_pair1_2_left);
        left_sum_2_right = left_sum_2_right.add(l_pair0_2_right).add(l_pair1_2_right);

        // Accumulate (identity + right) totals
        const combo_0 = i_pair0_0.add(i_pair1_0).add(r_pair0_0).add(r_pair1_0);
        const combo_2_left = i_pair0_2_left.add(i_pair1_2_left).add(r_pair0_2_left).add(r_pair1_2_left);
        const combo_2_right = i_pair0_2_right.add(i_pair1_2_right).add(r_pair0_2_right).add(r_pair1_2_right);

        right_sum_0 = right_sum_0.add(combo_0);
        right_sum_2_left = right_sum_2_left.add(combo_2_left);
        right_sum_2_right = right_sum_2_right.add(combo_2_right);
    }

    // Apply quadratic interpolation: eval_2 = 2*eval_2_right - eval_2_left
    const left_sum_2 = left_sum_2_right.add(left_sum_2_right).sub(left_sum_2_left);
    const right_sum_2 = right_sum_2_right.add(right_sum_2_right).sub(right_sum_2_left);

    // Final result: γ*left + γ²*(identity + right)
    const eval_0 = gamma.mul(left_sum_0).add(gamma_sqr.mul(right_sum_0));
    const eval_2 = gamma.mul(left_sum_2).add(gamma_sqr.mul(right_sum_2));

    return .{ eval_0, eval_2 };
}

// ============================================================================
// Expanding Table for Address-Binding Condensation
// ============================================================================

/// Expanding Table: accumulates EQ(x_1, ..., x_j, r_1, ..., r_j) as challenges come in.
/// Used to track the per-address eq weights during address rounds.
///
/// At each phase transition, u_evals are multiplied by the appropriate expanding table
/// entries to "condense" the prior phase's eq contributions.
///
/// Reference: jolt-core/src/utils/expanding_table.rs
pub fn ExpandingTable(comptime F: type) type {
    return struct {
        const Self = @This();

        /// The accumulated eq values (length doubles with each update)
        values: []F,
        /// Current length (starts at 1)
        len: usize,
        /// Scratch space for HighToLow binding
        scratch: []F,
        /// Allocator
        allocator: Allocator,

        /// Initialize with given capacity (should be max size = 2^log_m)
        pub fn init(allocator: Allocator, capacity: usize) !Self {
            const values = try allocator.alloc(F, capacity);
            const scratch = try allocator.alloc(F, capacity);
            @memset(values, F.zero());
            @memset(scratch, F.zero());
            return .{
                .values = values,
                .len = 0,
                .scratch = scratch,
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.values);
            self.allocator.free(self.scratch);
        }

        /// Reset to length 1 containing the given value (typically F.one())
        pub fn reset(self: *Self, value: F) void {
            self.values[0] = value;
            self.len = 1;
        }

        /// Get current length
        pub fn getLen(self: *const Self) usize {
            return self.len;
        }

        /// Get value at index
        pub fn get(self: *const Self, index: usize) F {
            std.debug.assert(index < self.len);
            return self.values[index];
        }

        /// Update the table (doubles length) with new challenge r_j.
        /// Uses HighToLow binding:
        ///   For each existing entry v[i]:
        ///     new[2*i] = v[i] * (1 - r_j) = v[i] - v[i]*r_j
        ///     new[2*i + 1] = v[i] * r_j
        pub fn update(self: *Self, r_j: F) void {
            // HighToLow: expand each entry into two
            for (0..self.len) |i| {
                const v_i = self.values[i];
                const eval_1 = r_j.mul(v_i); // v[i] * r
                self.scratch[2 * i] = v_i.sub(eval_1); // v[i] * (1 - r)
                self.scratch[2 * i + 1] = eval_1; // v[i] * r
            }

            // Swap values and scratch
            const tmp = self.values;
            self.values = self.scratch;
            self.scratch = tmp;

            self.len *= 2;
        }

        /// Update the table (doubles length) with new challenge r_j.
        /// Uses LowToHigh binding:
        ///   values[i] becomes values[i] * (1 - r_j) (contribution when NEW bit = 0)
        ///   values[i + old_len] becomes values[i] * r_j (contribution when NEW bit = 1)
        /// This places the OLDEST bound bit in the LSB of the index.
        pub fn updateLowToHigh(self: *Self, r_j: F) void {
            // LowToHigh: for each existing entry v[i]:
            //   new[i] = v[i] * (1 - r_j)
            //   new[i + len] = v[i] * r_j
            const old_len = self.len;
            for (0..old_len) |i| {
                const v_i = self.values[i];
                const eval_1 = r_j.mul(v_i); // v[i] * r
                self.values[i + old_len] = eval_1; // new entry when new bit = 1
                self.values[i] = v_i.sub(eval_1); // old entry becomes v[i] * (1-r)
            }
            self.len *= 2;
        }

        /// Clone the current values into a new slice
        pub fn cloneValues(self: *const Self, allocator: Allocator) ![]F {
            const result = try allocator.alloc(F, self.len);
            @memcpy(result, self.values[0..self.len]);
            return result;
        }
    };
}

/// Reverse the bits in a value (up to num_bits bits)
fn reverseBits(value: u128, num_bits: usize) u128 {
    var result: u128 = 0;
    var i: usize = 0;
    while (i < num_bits) : (i += 1) {
        const bit = (value >> @intCast(i)) & 1;
        result |= bit << @intCast(num_bits - 1 - i);
    }
    return result;
}

/// Condense u_evals using the expanding table values from the previous phase.
///
/// For each cycle j:
///   - Extract the bound prefix bits from lookup_index
///   - Multiply u_evals[j] by v[k_bound] where k_bound = prefix with bits reversed
///
/// This accumulates the eq contributions from the previous phase into u_evals.
///
/// IMPORTANT: The expanding table uses HighToLow binding, so:
///   - v[i] where i[0]=1 means k[MSB]=1 (first bound variable matched r[0])
///   - v[i] where i[1]=1 means k[MSB-1]=1 (second bound variable matched r[1])
/// But the prefix bits extracted from k are in the opposite order:
///   - prefix[0] = k[suffix_bits] (the lowest of the bound bits)
///   - prefix[log_m-1] = k[127] (the highest bound bit)
/// So we need to REVERSE the bit order of the prefix to get the correct v index.
///
/// Args:
///   u_evals: Per-cycle eq(j, r_reduction) values to be condensed
///   v: Expanding table from previous phase
///   lookup_indices: Per-cycle lookup indices
///   phase: Current phase number (1-indexed when condensing)
///   phases: Total number of phases (typically 8)
pub fn condenseUEvals(
    comptime F: type,
    u_evals: []F,
    v: *const ExpandingTable(F),
    lookup_indices: []const u128,
    phase: usize,
    phases: usize,
) void {
    const log_m = LOG_K / phases;
    const m_mask: u128 = (@as(u128, 1) << @intCast(log_m)) - 1;

    // Number of suffix bits remaining after this phase
    const suffix_bits = (phases - phase) * log_m;

    var max_k_bound: u128 = 0;
    for (lookup_indices, 0..) |k, j| {
        // Extract the bound prefix (the bits that have been bound in previous phases)
        const prefix = k >> @intCast(suffix_bits);
        const k_prefix_raw = prefix & m_mask;

        // NOTE: Jolt uses the prefix bits directly without reversal
        // The expanding table indexing in Jolt appears to match this convention
        const k_bound = k_prefix_raw;

        if (k_bound > max_k_bound) max_k_bound = k_bound;

        // Multiply by the expanding table value
        if (k_bound >= v.getLen()) {
            dbg("[CONDENSE] ERROR: k_bound={} >= v.len={} at j={}, phase={}, suffix_bits={}, k=0x{x:0>32}\n", .{ k_bound, v.getLen(), j, phase, suffix_bits, k });
            @panic("k_bound out of range");
        }
        u_evals[j] = u_evals[j].mul(v.get(@intCast(k_bound)));
    }
    dbg("[CONDENSE] phase={}, suffix_bits={}, max_k_bound={}\n", .{ phase, suffix_bits, max_k_bound });
}

// ============================================================================
// Table MLE Computation at r_address (for cycle rounds)
// ============================================================================

/// Compute table_values_at_r_addr for all 41 tables after address rounds complete.
///
/// At this point:
/// - All 128 address variables have been bound via prefix-suffix decomposition
/// - The suffix variable set is empty (suffix_len = 0)
/// - Each suffix MLE is evaluated on an empty bitstring
///
/// For each table t:
///   table_values_at_r_addr[t] = table.combine(&prefixes, &suffix_evals)
///
/// where:
///   - prefixes are the bound prefix checkpoint values
///   - suffix_evals[s] = F::from_u64(suffix.suffix_mle(empty_bits))
///
/// Reference: jolt-core/src/zkvm/instruction_lookups/read_raf_checking.rs:641-671
pub fn computeTableValuesAtRAddress(
    comptime F: type,
    prefix_checkpoints: *const PrefixCheckpointsState(F),
) [NUM_TABLES]F {
    var result: [NUM_TABLES]F = undefined;

    // Empty suffix bits (0 bits)
    const empty_suffix_bits = LookupBits(128).new(0, 0);

    // Convert prefix checkpoints to an array of F values (using 0 for None)
    var prefix_values: [Prefixes.COUNT]F = undefined;
    var non_zero_prefixes: usize = 0;
    for (0..Prefixes.COUNT) |i| {
        prefix_values[i] = prefix_checkpoints.checkpoints[i] orelse F.zero();
        if (!prefix_values[i].eql(F.zero())) non_zero_prefixes += 1;
    }
    dbg("[computeTableValuesAtRAddress] non_zero_prefixes={}/{}\n", .{ non_zero_prefixes, Prefixes.COUNT });

    // Compute MLE value for each table
    for (0..NUM_TABLES) |table_idx| {
        // Get the suffixes for this table
        const table_suff = tableSuffixes(table_idx);

        // Evaluate each suffix MLE at empty bits and convert to field element
        var suffix_evals: [MAX_SUFFIXES_PER_TABLE]F = undefined;
        for (table_suff, 0..) |suff, s_idx| {
            const mle_val = suffixMle(suff, empty_suffix_bits);
            suffix_evals[s_idx] = F.fromU64(mle_val);
        }

        // Combine using the table-specific formula
        result[table_idx] = tableCombine(F, table_idx, &prefix_values, suffix_evals[0..table_suff.len]);

        // Debug: print first few non-zero table values
        if (table_idx < 5 or !result[table_idx].eql(F.zero())) {
            dbg("[computeTableValuesAtRAddress] table[{}]: num_suffixes={}, combined={x}\n", .{
                table_idx,
                table_suff.len,
                result[table_idx].toBytesBE()[24..32].*,
            });
        }
    }

    return result;
}

// ============================================================================
// Tests
// ============================================================================

test "ExpandingTable basic operations" {
    const F = @import("../../field/mod.zig").BN254Scalar;
    const allocator = std.testing.allocator;

    var table = try ExpandingTable(F).init(allocator, 16);
    defer table.deinit();

    // Reset to 1
    table.reset(F.one());
    try std.testing.expectEqual(@as(usize, 1), table.getLen());
    try std.testing.expect(table.get(0).eql(F.one()));

    // Update with r = 0.5 (approx) - just test with a simple value
    const r = F.fromU64(123456789);
    table.update(r);

    try std.testing.expectEqual(@as(usize, 2), table.getLen());
    // After update: values[0] = 1 - r, values[1] = r
    const one_minus_r = F.one().sub(r);
    try std.testing.expect(table.get(0).eql(one_minus_r));
    try std.testing.expect(table.get(1).eql(r));
}

test "AllSuffixPolys init and deinit" {
    const F = @import("../../field/mod.zig").BN254Scalar;
    const allocator = std.testing.allocator;

    var polys = AllSuffixPolys(F).init(allocator);
    defer polys.deinit();

    // Initialize with empty data
    const empty_u_evals = [_]F{};
    const empty_indices = [_]u128{};
    const empty_tables = [_]i8{};

    try polys.initPhase(0, 8, &empty_u_evals, &empty_indices, &empty_tables);
}

test "TableSuffixPolys bind" {
    const F = @import("../../field/mod.zig").BN254Scalar;
    const allocator = std.testing.allocator;

    var table = try TableSuffixPolys(F).init(allocator, 2, 4);
    defer table.deinit();

    // Set some values
    table.polys[0][0] = F.fromU64(1);
    table.polys[0][1] = F.fromU64(2);
    table.polys[0][2] = F.fromU64(3);
    table.polys[0][3] = F.fromU64(4);

    // Bind with r = 0 (should select low values)
    table.bind(F.zero());

    // After binding: poly[0] = (1-0)*1 + 0*2 = 1, poly[1] = (1-0)*3 + 0*4 = 3
    try std.testing.expect(table.polys[0][0].eql(F.fromU64(1)));
    try std.testing.expect(table.polys[0][1].eql(F.fromU64(3)));
}

test "RafDecomposition init and deinit" {
    const F = @import("../../field/mod.zig").BN254Scalar;
    const allocator = std.testing.allocator;

    var raf = try RafDecomposition(F).init(allocator, 16, 16, 128, .LeftOperand);
    defer raf.deinit();

    try std.testing.expectEqual(@as(usize, 16), raf.QLen());
    try std.testing.expectEqual(@as(usize, 112), raf.suffixLen()); // 128 - (0+1)*16 = 112
}

test "uninterleaveBits" {
    // Test that uninterleave correctly separates bits per Jolt's format
    // In Jolt: interleaved = (left << 1) | right
    // So left operand is at ODD positions, right operand is at EVEN positions
    //
    // bits = 0b1010 = 10
    // Binary: bit3=1, bit2=0, bit1=1, bit0=0
    // Odd positions (1, 3): bits 1 and 1 -> left = 0b11 = 3
    // Even positions (0, 2): bits 0 and 0 -> right = 0b00 = 0
    const bits: u128 = 0b1010;
    const left = uninterleaveBitsLeft(bits, 4);
    const right = uninterleaveBitsRight(bits, 4);

    // Left from odd bits (positions 1, 3) = 1, 1 -> 0b11 = 3
    try std.testing.expectEqual(@as(u64, 3), left);
    // Right from even bits (positions 0, 2) = 0, 0 -> 0b00 = 0
    try std.testing.expectEqual(@as(u64, 0), right);
}

test "initQRaf basic" {
    const F = @import("../../field/mod.zig").BN254Scalar;
    const allocator = std.testing.allocator;

    var left = try RafDecomposition(F).init(allocator, 4, 2, 8, .LeftOperand);
    defer left.deinit();
    var right = try RafDecomposition(F).init(allocator, 4, 2, 8, .RightOperand);
    defer right.deinit();
    var identity = try RafDecomposition(F).init(allocator, 4, 2, 8, .Identity);
    defer identity.deinit();

    // Create simple test data: one interleaved cycle, one identity cycle
    const u_evals = [_]F{ F.one(), F.one() };
    const lookup_indices = [_]u128{ 0x0, 0x0 }; // Both at index 0
    const is_interleaved = [_]bool{ true, false };

    initQRaf(F, &left, &right, &identity, &u_evals, &lookup_indices, &is_interleaved);

    // Interleaved cycle should contribute to left/right Q[0] (shift) and Q[1] (operand)
    // Identity cycle should contribute to identity Q[0] (shift) and Q[1] (identity)
    // With suffix_len = 6 (8 - (0+1)*2), shift_half = 2^3 = 8, shift_full = 2^6 = 64
    const shift_half = F.fromU64(8);
    const shift_full = F.fromU64(64);

    try std.testing.expect(left.Q[0][0].eql(shift_half));
    try std.testing.expect(identity.Q[0][0].eql(shift_full));
}
