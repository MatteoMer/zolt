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

        /// Bind a challenge (halves the polynomial size)
        /// Uses HighToLow binding order: new[i] = left[i] + r * (right[i] - left[i])
        /// where left = poly[0..n], right = poly[n..2n]
        pub fn bind(self: *Self, r: F) void {
            for (self.polys) |poly| {
                const half_size = poly.len / 2;
                // HighToLow: left half [0..n], right half [n..2n]
                for (0..half_size) |j| {
                    const low = poly[j];
                    const high = poly[j + half_size];
                    poly[j] = low.add(r.mul(high.sub(low)));
                }
            }
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
                    // Reset existing polynomials
                    for (self.tables[table_idx].?.polys) |poly| {
                        @memset(poly, F.zero());
                    }
                }
            }

            // Accumulate contributions from each cycle
            for (0..u_evals.len) |j| {
                const table_idx = cycle_table_indices[j];
                if (table_idx < 0) continue; // No table for this cycle

                const t_idx: usize = @intCast(table_idx);
                if (t_idx >= NUM_TABLES) continue;

                const k = lookup_indices[j];
                const prefix_bits = (k >> @intCast(suffix_len)) & m_mask;
                const suffix_bits_raw = k & ((@as(u128, 1) << @intCast(suffix_len)) - 1);
                const suffix_bits = LookupBits(128).new(suffix_bits_raw, suffix_len);

                const u = u_evals[j];
                const table_suffixes = tableSuffixes(t_idx);

                // Accumulate for each suffix type
                for (table_suffixes, 0..) |suffix, s_idx| {
                    const t = suffixMle(suffix, suffix_bits);
                    if (t != 0) {
                        const q_poly = self.tables[t_idx].?.polys[s_idx];
                        const idx: usize = @intCast(prefix_bits);
                        if (suffixes_mod.is01Valued(suffix)) {
                            // {0,1}-valued: t is 1, just add u
                            q_poly[idx] = q_poly[idx].add(u);
                        } else {
                            // General suffix: multiply by t
                            q_poly[idx] = q_poly[idx].add(u.mul(F.fromU64(t)));
                        }
                    }
                }
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
        // Find the current Q length from any initialized table
        for (suffix_polys.tables) |maybe_table| {
            if (maybe_table) |table| {
                if (table.polys.len > 0 and table.polys[0].len > 0) {
                    break :blk table.polys[0].len;
                }
            }
        }
        break :blk 1; // Fallback
    };

    const log_len = @ctz(len);
    const half_len = len / 2;

    var eval_0 = F.zero();
    var eval_2_left = F.zero();
    var eval_2_right = F.zero();

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
                eval_2_left = eval_2_left.add(combined_2_left);
                eval_2_right = eval_2_right.add(combined_2_right);
            }
        }
    }

    // Quadratic interpolation: eval_2 = 2*eval_2_right - eval_2_left
    const eval_2 = eval_2_right.add(eval_2_right).sub(eval_2_left);

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
        /// Number of bound variables
        num_bound_vars: usize,
        /// Bound value (accumulated from challenges)
        bound_value: F,
        /// Type of RAF polynomial (determines binding behavior)
        poly_type: RafPolyType,
        /// Allocator
        allocator: Allocator,

        pub fn init(allocator: Allocator, initial_size: usize, chunk_len: usize, total_len: usize, poly_type: RafPolyType) !Self {
            var Q: [2][]F = undefined;
            Q[0] = try allocator.alloc(F, initial_size);
            Q[1] = try allocator.alloc(F, initial_size);
            @memset(Q[0], F.zero());
            @memset(Q[1], F.zero());
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
            };
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.Q[0]);
            self.allocator.free(self.Q[1]);
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

        /// Bind a challenge to Q polynomials and update bound_value
        /// Uses HighToLow binding order: new[j] = Q[j] + r * (Q[j+half] - Q[j])
        pub fn bind(self: *Self, r: F) void {
            const half_size = self.Q_size / 2;
            // HighToLow: left half [0..half_size], right half [half_size..Q_size]
            for (0..2) |i| {
                for (0..half_size) |j| {
                    const low = self.Q[i][j];
                    const high = self.Q[i][j + half_size];
                    self.Q[i][j] = low.add(r.mul(high.sub(low)));
                }
            }
            self.Q_size = half_size;

            // Update bound_value based on polynomial type
            // Jolt's binding rule:
            // - OperandPolynomial: binds when round parity matches side
            // - IdentityPolynomial: always binds
            if (self.shouldBind()) {
                self.bound_value = self.bound_value.add(self.bound_value).add(r);
            }

            self.num_bound_vars += 1;
            self.round += 1;
            if (self.round % self.chunk_len == 0) {
                self.phase += 1;
            }
        }

        /// Compute sumcheck evaluations at index b
        /// Returns (P(0), P(2)) where P is the prefix polynomial
        ///
        /// For OperandPolynomial:
        ///   If binding this round:
        ///     P(c) = bound_value * 2^unbound_pairs + operand_bits(b) + c * m
        ///   Else:
        ///     P(c) = bound_value * 2^unbound_pairs + operand_bits(b) (constant in c)
        ///
        /// For IdentityPolynomial:
        ///   P(c) = bound_value * 2^remaining + b + c * m (always linear)
        pub fn prefixEvals(self: *const Self, b: usize) [2]F {
            const num_vars = self.total_len;
            const remaining = num_vars - self.num_bound_vars;

            return switch (self.poly_type) {
                .LeftOperand => self.operandPrefixEvals(b, remaining, true),
                .RightOperand => self.operandPrefixEvals(b, remaining, false),
                .Identity => self.identityPrefixEvals(b, remaining),
            };
        }

        fn operandPrefixEvals(self: *const Self, b: usize, remaining: usize, is_left: bool) [2]F {
            // Uninterleave b to get the operand bits
            // In Jolt's interleave format: odd positions -> left, even positions -> right
            const half_bits = remaining / 2;
            var operand_bits: u64 = 0;
            if (half_bits > 0) {
                if (is_left) {
                    // Left operand: extract odd positions (1, 3, 5, ...)
                    for (0..@min(half_bits, 32)) |i| {
                        const bit: u64 = @truncate((b >> @intCast(2 * i + 1)) & 1);
                        operand_bits |= bit << @intCast(i);
                    }
                } else {
                    // Right operand: extract even positions (0, 2, 4, ...)
                    for (0..@min(half_bits, 32)) |i| {
                        const bit: u64 = @truncate((b >> @intCast(2 * i)) & 1);
                        operand_bits |= bit << @intCast(i);
                    }
                }
            }

            // In Jolt's format: even positions = right operand, odd positions = left operand
            // When num_bound_vars is even, we're binding an even position (right operand)
            // When num_bound_vars is odd, we're binding an odd position (left operand)
            const is_even_round = (self.num_bound_vars % 2 == 0);
            const is_binding_round = if (is_left) !is_even_round else is_even_round;

            if (is_binding_round and half_bits > 0) {
                // Linear polynomial: P(c) = base + c * m
                // where base = bound_value * 2^unbound_pairs + operand_bits
                // and m = 2^(unbound_pairs - 1)
                const unbound_pairs = remaining / 2;
                // Use field element arithmetic to handle large scales (unbound_pairs can be up to 64)
                const scale = fieldPow2(F, unbound_pairs);
                const m = fieldPow2(F, unbound_pairs - 1);

                const base = self.bound_value.mul(scale).add(F.fromU64(operand_bits));
                const eval_0 = base;
                const eval_2 = base.add(m.add(m)); // 2 * m using field arithmetic
                return .{ eval_0, eval_2 };
            } else {
                // Constant polynomial (not binding or remaining=0)
                const unbound_pairs = remaining / 2;
                const scale = if (unbound_pairs > 0) fieldPow2(F, unbound_pairs) else F.one();
                const base = self.bound_value.mul(scale).add(F.fromU64(operand_bits));
                return .{ base, base }; // P(0) = P(2) = base
            }
        }

        fn identityPrefixEvals(self: *const Self, b: usize, remaining: usize) [2]F {
            // Identity polynomial always binds
            // P(c) = bound_value * 2^remaining + b + c * m
            // where m = 2^(remaining - 1)
            if (remaining > 0) {
                // Use field element arithmetic to handle large scales (remaining can be up to 128)
                const scale = fieldPow2(F, remaining);
                const m = fieldPow2(F, remaining - 1);
                const base = self.bound_value.mul(scale).add(F.fromU64(@intCast(b)));
                const eval_0 = base;
                const eval_2 = base.add(m.add(m)); // 2 * m using field arithmetic
                return .{ eval_0, eval_2 };
            } else {
                // All variables bound
                const base = self.bound_value.add(F.fromU64(@intCast(b)));
                return .{ base, base };
            }
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
/// - (ShiftSuffix prefix=1, Q[0]) - prefix is constant 1
/// - (OperandSuffix prefix P, Q[1]) - prefix depends on bound_value and round
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
        // For shift suffix (Q[0]): prefix is constant 1
        const shift_p0 = F.one();
        const shift_p2 = F.one();

        // For operand/identity suffix (Q[1]): use proper prefix evaluation
        const l_prefix = left_ps.prefixEvals(b);
        const r_prefix = right_ps.prefixEvals(b);
        const i_prefix = identity_ps.prefixEvals(b);

        // Left operand contribution: sum of (shift, Q[0]) and (operand, Q[1]) pairs
        // eval_0 = P(0)*Q[left], eval_2_left = P(2)*Q[left], eval_2_right = P(2)*Q[right]
        const l_shift_0 = shift_p0.mul(l_q0_left);
        const l_shift_2_left = shift_p2.mul(l_q0_left);
        const l_shift_2_right = shift_p2.mul(l_q0_right);

        const l_op_0 = l_prefix[0].mul(l_q1_left);
        const l_op_2_left = l_prefix[1].mul(l_q1_left);
        const l_op_2_right = l_prefix[1].mul(l_q1_right);

        // Right operand contribution
        const r_shift_0 = shift_p0.mul(r_q0_left);
        const r_shift_2_left = shift_p2.mul(r_q0_left);
        const r_shift_2_right = shift_p2.mul(r_q0_right);

        const r_op_0 = r_prefix[0].mul(r_q1_left);
        const r_op_2_left = r_prefix[1].mul(r_q1_left);
        const r_op_2_right = r_prefix[1].mul(r_q1_right);

        // Identity contribution
        const i_shift_0 = shift_p0.mul(i_q0_left);
        const i_shift_2_left = shift_p2.mul(i_q0_left);
        const i_shift_2_right = shift_p2.mul(i_q0_right);

        const i_op_0 = i_prefix[0].mul(i_q1_left);
        const i_op_2_left = i_prefix[1].mul(i_q1_left);
        const i_op_2_right = i_prefix[1].mul(i_q1_right);

        // Accumulate left operand totals
        left_sum_0 = left_sum_0.add(l_shift_0).add(l_op_0);
        left_sum_2_left = left_sum_2_left.add(l_shift_2_left).add(l_op_2_left);
        left_sum_2_right = left_sum_2_right.add(l_shift_2_right).add(l_op_2_right);

        // Accumulate (identity + right) totals
        const combo_0 = i_shift_0.add(i_op_0).add(r_shift_0).add(r_op_0);
        const combo_2_left = i_shift_2_left.add(i_op_2_left).add(r_shift_2_left).add(r_op_2_left);
        const combo_2_right = i_shift_2_right.add(i_op_2_right).add(r_shift_2_right).add(r_op_2_right);

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

        /// Clone the current values into a new slice
        pub fn cloneValues(self: *const Self, allocator: Allocator) ![]F {
            const result = try allocator.alloc(F, self.len);
            @memcpy(result, self.values[0..self.len]);
            return result;
        }
    };
}

/// Condense u_evals using the expanding table values from the previous phase.
///
/// For each cycle j:
///   - Extract the bound prefix bits from lookup_index
///   - Multiply u_evals[j] by v[k_bound] where k_bound = prefix & m_mask
///
/// This accumulates the eq contributions from the previous phase into u_evals.
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
        const k_bound = prefix & m_mask;

        if (k_bound > max_k_bound) max_k_bound = k_bound;

        // Multiply by the expanding table value
        if (k_bound >= v.getLen()) {
            std.debug.print("[CONDENSE] ERROR: k_bound={} >= v.len={} at j={}, phase={}, suffix_bits={}, k=0x{x:0>32}\n", .{ k_bound, v.getLen(), j, phase, suffix_bits, k });
            @panic("k_bound out of range");
        }
        u_evals[j] = u_evals[j].mul(v.get(@intCast(k_bound)));
    }
    std.debug.print("[CONDENSE] phase={}, suffix_bits={}, max_k_bound={}\n", .{ phase, suffix_bits, max_k_bound });
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
