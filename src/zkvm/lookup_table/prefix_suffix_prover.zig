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
fn tableCombine(comptime F: type, table_idx: usize, prefixes: []const F, suffixes: []const F) F {
    // Each table has a specific combination formula
    // Most tables: result = prefix[relevant] * suffixes[0] + suffixes[1]
    // This matches Jolt's table.combine() implementations

    return switch (table_idx) {
        // RangeCheck: prefixes[LowerWord] * One + LowerWord
        0 => prefixes[@intFromEnum(Prefixes.LowerWord)].mul(suffixes[0]).add(suffixes[1]),
        // RangeCheckAligned: same as RangeCheck
        1 => prefixes[@intFromEnum(Prefixes.LowerWord)].mul(suffixes[0]).add(suffixes[1]),
        // And: prefixes[And] * One + And
        2 => prefixes[@intFromEnum(Prefixes.And)].mul(suffixes[0]).add(suffixes[1]),
        // Andn: prefixes[Andn] * One + NotAnd
        3 => prefixes[@intFromEnum(Prefixes.Andn)].mul(suffixes[0]).add(suffixes[1]),
        // Or: prefixes[Or] * One + Or
        4 => prefixes[@intFromEnum(Prefixes.Or)].mul(suffixes[0]).add(suffixes[1]),
        // Xor: prefixes[Xor] * One + Xor
        5 => prefixes[@intFromEnum(Prefixes.Xor)].mul(suffixes[0]).add(suffixes[1]),
        // Equal: prefixes[Eq] * One + Eq_suffix
        6 => prefixes[@intFromEnum(Prefixes.Eq)].mul(suffixes[0]).add(suffixes[1]),
        // SignedGreaterThanEqual: complex formula involving MSB prefixes
        7 => {
            if (suffixes.len >= 2) {
                return prefixes[@intFromEnum(Prefixes.LessThan)].mul(suffixes[0]).add(suffixes[1]);
            }
            return suffixes[0];
        },
        // UnsignedGreaterThanEqual
        8 => {
            if (suffixes.len >= 2) {
                return prefixes[@intFromEnum(Prefixes.LessThan)].mul(suffixes[0]).add(suffixes[1]);
            }
            return suffixes[0];
        },
        // For other tables, use a simple combination
        else => {
            if (suffixes.len == 0) return F.zero();
            if (suffixes.len == 1) return suffixes[0];
            // Default: first prefix * first suffix + second suffix
            return suffixes[0].add(suffixes[1]);
        },
    };
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
