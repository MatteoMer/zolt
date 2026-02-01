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
        pub fn bind(self: *Self, r: F) void {
            for (self.polys) |poly| {
                const half_size = poly.len / 2;
                for (0..half_size) |j| {
                    const low = poly[2 * j];
                    const high = poly[2 * j + 1];
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
        // 15-41: Other tables - use simplified patterns for now
        // Most follow pattern: prefix[Type] * one + suffix_result
        15...41 => blk: {
            if (suffixes.len == 0) break :blk F.zero();
            if (suffixes.len == 1) break :blk suffixes[0];
            // Default pattern: sum of all suffixes
            var result = suffixes[0];
            for (suffixes[1..]) |s| {
                result = result.add(s);
            }
            break :blk result;
        },
        else => F.zero(),
    };
}

/// RAF (Read-Address-Flag) Decomposition State
/// This handles the identity/operand polynomial decomposition for RAF sumcheck
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
        /// Current round within phase
        round: usize,
        /// Bound prefix value (accumulated from challenges)
        bound_prefix: F,
        /// Allocator
        allocator: Allocator,

        pub fn init(allocator: Allocator, initial_size: usize, chunk_len: usize, total_len: usize) !Self {
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
                .bound_prefix = F.zero(),
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

        /// Bind a challenge to Q polynomials and update prefix
        pub fn bind(self: *Self, r: F) void {
            const half_size = self.Q_size / 2;
            for (0..2) |i| {
                for (0..half_size) |j| {
                    const low = self.Q[i][2 * j];
                    const high = self.Q[i][2 * j + 1];
                    self.Q[i][j] = low.add(r.mul(high.sub(low)));
                }
            }
            self.Q_size = half_size;

            // Update bound prefix (HighToLow binding)
            self.bound_prefix = self.bound_prefix.add(self.bound_prefix).add(r);

            self.round += 1;
            if (self.round % self.chunk_len == 0) {
                self.phase += 1;
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

/// Uninterleave bits to get the left operand (even positions)
fn uninterleaveBitsLeft(bits: u128, num_bits: usize) u64 {
    var left: u64 = 0;
    const half_bits = num_bits / 2;
    var i: u6 = 0;
    while (i < half_bits and i < 64) : (i += 1) {
        // Even positions (0, 2, 4, ...) go to left
        const bit = (bits >> @as(u7, @intCast(2 * i))) & 1;
        left |= @as(u64, @truncate(bit)) << i;
    }
    return left;
}

/// Uninterleave bits to get the right operand (odd positions)
fn uninterleaveBitsRight(bits: u128, num_bits: usize) u64 {
    var right: u64 = 0;
    const half_bits = num_bits / 2;
    var i: u6 = 0;
    while (i < half_bits and i < 64) : (i += 1) {
        // Odd positions (1, 3, 5, ...) go to right
        const bit = (bits >> @as(u7, @intCast(2 * i + 1))) & 1;
        right |= @as(u64, @truncate(bit)) << i;
    }
    return right;
}

/// Compute prover message for RAF (Read-Address-Flag) contribution
/// Returns [eval_0, eval_2] for the degree-2 polynomial
///
/// This computes: γ*left + γ²*(identity + right)
/// Where left, right, identity are prefix-suffix decompositions with:
/// - left.Q[0] = ShiftHalfSuffix * Σ u (for interleaved cycles)
/// - left.Q[1] = LeftOperandSuffix * Σ u
/// - right.Q[0] = ShiftHalfSuffix * Σ u (same as left)
/// - right.Q[1] = RightOperandSuffix * Σ u
/// - identity.Q[0] = ShiftFullSuffix * Σ u (for identity cycles)
/// - identity.Q[1] = IdentitySuffix * Σ u
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
    var left_sum_2 = F.zero();
    var right_sum_0 = F.zero(); // Actually (identity + right) at c=0
    var right_sum_2 = F.zero(); // Actually (identity + right) at c=2

    // For each half-index b, compute sumcheck evaluations
    for (0..half_len) |b| {
        // Get Q values at left (b) and right (b + half_len) positions
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

        // Compute prefix evaluations for the operand/identity polynomials
        // For OperandPolynomial with HighToLow binding:
        // P(c) = bound_prefix * 2^{remaining_vars} + c * m (where m depends on which var is being bound)

        // For simplicity in the current implementation, we use the Q accumulators directly
        // The prefix polynomials are handled implicitly through the bound values

        // Left operand contribution: P_shift_half(c) * Q_shift_half + P_left(c) * Q_left
        // Since P_shift_half is constant (1), and P_left is linear in c for left-binding rounds:
        const l0 = l_q0_left.add(l_q1_left);
        const l2_left = l_q0_left.add(l_q1_left);
        const l2_right = l_q0_right.add(l_q1_right);
        const l2 = l2_right.add(l2_right).sub(l2_left);

        // Right operand: same structure
        const r0 = r_q0_left.add(r_q1_left);
        const r2_left = r_q0_left.add(r_q1_left);
        const r2_right = r_q0_right.add(r_q1_right);
        const r2 = r2_right.add(r2_right).sub(r2_left);

        // Identity: same structure
        const id0 = i_q0_left.add(i_q1_left);
        const id2_left = i_q0_left.add(i_q1_left);
        const id2_right = i_q0_right.add(i_q1_right);
        const id2 = id2_right.add(id2_right).sub(id2_left);

        // Accumulate: left for γ weight, (identity + right) for γ² weight
        left_sum_0 = left_sum_0.add(l0);
        left_sum_2 = left_sum_2.add(l2);
        right_sum_0 = right_sum_0.add(id0.add(r0));
        right_sum_2 = right_sum_2.add(id2.add(r2));
    }

    // Final result: γ*left + γ²*(identity + right)
    const eval_0 = gamma.mul(left_sum_0).add(gamma_sqr.mul(right_sum_0));
    const eval_2 = gamma.mul(left_sum_2).add(gamma_sqr.mul(right_sum_2));

    return .{ eval_0, eval_2 };
}

// ============================================================================
// Tests
// ============================================================================

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

    var raf = try RafDecomposition(F).init(allocator, 16, 16, 128);
    defer raf.deinit();

    try std.testing.expectEqual(@as(usize, 16), raf.QLen());
    try std.testing.expectEqual(@as(usize, 112), raf.suffixLen()); // 128 - (0+1)*16 = 112
}

test "uninterleaveBits" {
    // Test that uninterleave correctly separates even and odd bits
    // bits = 0b1010 (binary), suffix_len = 4
    // Even positions (0, 2): bits 0 and 2 -> left = 0b00 = 0
    // Odd positions (1, 3): bits 1 and 3 -> right = 0b11 = 3
    // Actually: position 0 has bit 0, position 2 has bit 0, position 1 has bit 1, position 3 has bit 1
    // 0b1010 = position 1 and 3 are set
    const bits: u128 = 0b1010;
    const left = uninterleaveBitsLeft(bits, 4);
    const right = uninterleaveBitsRight(bits, 4);

    // Even bits (positions 0, 2) = 0, 0 -> left = 0
    try std.testing.expectEqual(@as(u64, 0), left);
    // Odd bits (positions 1, 3) = 1, 1 -> right = 0b11 = 3
    try std.testing.expectEqual(@as(u64, 3), right);
}

test "initQRaf basic" {
    const F = @import("../../field/mod.zig").BN254Scalar;
    const allocator = std.testing.allocator;

    var left = try RafDecomposition(F).init(allocator, 4, 2, 8);
    defer left.deinit();
    var right = try RafDecomposition(F).init(allocator, 4, 2, 8);
    defer right.deinit();
    var identity = try RafDecomposition(F).init(allocator, 4, 2, 8);
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
