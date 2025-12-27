//! Expanding Table for Lasso Lookup Arguments
//!
//! This module implements the ExpandingTable data structure used in the Lasso
//! lookup argument protocol. The ExpandingTable incrementally builds EQ polynomial
//! evaluations as sumcheck rounds progress.
//!
//! The key insight is that during sumcheck, we need to evaluate:
//!   eq(r, x) = prod_{i=1}^{n} (r_i * x_i + (1-r_i) * (1-x_i))
//!
//! Rather than recomputing this from scratch each round, the ExpandingTable
//! maintains a running table that doubles in size each round:
//!   - Round 0: [1] (before any binding)
//!   - Round 1: [(1-r_0), r_0] (after binding first variable)
//!   - Round 2: [(1-r_0)(1-r_1), (1-r_0)*r_1, r_0*(1-r_1), r_0*r_1]
//!   - etc.
//!
//! This gives O(2^i) storage for round i, which is much smaller than the full
//! table size O(2^n) during early rounds.

const std = @import("std");
const Allocator = std.mem.Allocator;

/// ExpandingTable - Incrementally builds EQ polynomial evaluations
///
/// The table stores evaluations of eq(r[0..k], x) for all x in {0,1}^k,
/// where k is the current round number.
pub fn ExpandingTable(comptime F: type) type {
    return struct {
        const Self = @This();

        /// The current evaluations. Length = 2^round.
        values: []F,
        /// Current round (number of variables bound)
        round: usize,
        /// Maximum rounds (total number of variables)
        max_rounds: usize,
        /// Allocator for resizing
        allocator: Allocator,

        /// Create a new ExpandingTable for a polynomial with n variables
        pub fn init(allocator: Allocator, max_rounds: usize) !Self {
            // Start with a single value: 1 (empty product)
            const values = try allocator.alloc(F, 1);
            values[0] = F.one();

            return Self{
                .values = values,
                .round = 0,
                .max_rounds = max_rounds,
                .allocator = allocator,
            };
        }

        /// Initialize with a specific starting value (not 1)
        pub fn initWithValue(allocator: Allocator, max_rounds: usize, initial: F) !Self {
            const values = try allocator.alloc(F, 1);
            values[0] = initial;

            return Self{
                .values = values,
                .round = 0,
                .max_rounds = max_rounds,
                .allocator = allocator,
            };
        }

        /// Free the table
        pub fn deinit(self: *Self) void {
            self.allocator.free(self.values);
        }

        /// Get the current size (2^round)
        pub fn size(self: *const Self) usize {
            return self.values.len;
        }

        /// Bind the next variable to challenge value r
        ///
        /// This doubles the table size:
        /// For each existing value v at index i, we create:
        ///   - new_values[2*i] = v * (1 - r)
        ///   - new_values[2*i + 1] = v * r
        pub fn bind(self: *Self, r: F) !void {
            std.debug.assert(self.round < self.max_rounds);

            const new_size = self.values.len * 2;
            const new_values = try self.allocator.alloc(F, new_size);

            const one_minus_r = F.one().sub(r);

            for (self.values, 0..) |v, i| {
                new_values[2 * i] = v.mul(one_minus_r);
                new_values[2 * i + 1] = v.mul(r);
            }

            self.allocator.free(self.values);
            self.values = new_values;
            self.round += 1;
        }

        /// Bind variable with a precomputed pair (1-r, r)
        /// This is slightly more efficient when the pair is already computed
        pub fn bindWithPair(self: *Self, one_minus_r: F, r: F) !void {
            std.debug.assert(self.round < self.max_rounds);

            const new_size = self.values.len * 2;
            const new_values = try self.allocator.alloc(F, new_size);

            for (self.values, 0..) |v, i| {
                new_values[2 * i] = v.mul(one_minus_r);
                new_values[2 * i + 1] = v.mul(r);
            }

            self.allocator.free(self.values);
            self.values = new_values;
            self.round += 1;
        }

        /// Get the evaluation at a specific index
        pub fn get(self: *const Self, index: usize) F {
            std.debug.assert(index < self.values.len);
            return self.values[index];
        }

        /// Get all current values
        pub fn getAll(self: *const Self) []const F {
            return self.values;
        }

        /// Compute the sum of all current values
        /// This equals eq(r, x) summed over all x in {0,1}^k
        /// which equals 1 (for any r)
        pub fn sum(self: *const Self) F {
            var result = F.zero();
            for (self.values) |v| {
                result = result.add(v);
            }
            return result;
        }

        /// Condense the table by applying a weighting function
        /// Returns an array where condensed[j] = sum_i (values[i] * weights[i])
        /// where the sum is over indices i whose upper bits equal j
        pub fn condense(self: *const Self, allocator: Allocator, weights: []const F, out_bits: usize) ![]F {
            std.debug.assert(weights.len == self.values.len);
            std.debug.assert(out_bits <= self.round);

            const out_size = @as(usize, 1) << @intCast(out_bits);
            const in_chunk = @as(usize, 1) << @intCast(self.round - out_bits);

            const result = try allocator.alloc(F, out_size);
            @memset(result, F.zero());

            for (0..self.values.len) |i| {
                const out_idx = i / in_chunk;
                result[out_idx] = result[out_idx].add(self.values[i].mul(weights[i]));
            }

            return result;
        }

        /// Clone the table
        pub fn clone(self: *const Self) !Self {
            const new_values = try self.allocator.alloc(F, self.values.len);
            @memcpy(new_values, self.values);

            return Self{
                .values = new_values,
                .round = self.round,
                .max_rounds = self.max_rounds,
                .allocator = self.allocator,
            };
        }

        /// Scale all values by a constant
        pub fn scale(self: *Self, scalar: F) void {
            for (self.values) |*v| {
                v.* = v.mul(scalar);
            }
        }

        /// Add another expanding table (must be same size)
        pub fn addAssign(self: *Self, other: *const Self) void {
            std.debug.assert(self.values.len == other.values.len);
            for (self.values, other.values) |*v, o| {
                v.* = v.add(o);
            }
        }
    };
}

test "expanding table init and bind" {
    const F = @import("../../field/mod.zig").BN254Scalar;
    const allocator = std.testing.allocator;

    var table = try ExpandingTable(F).init(allocator, 3);
    defer table.deinit();

    // Initially: [1]
    try std.testing.expectEqual(@as(usize, 1), table.size());
    try std.testing.expect(table.get(0).eql(F.one()));

    // Bind r0 = 3
    const r0 = F.fromU64(3);
    try table.bind(r0);

    // Now: [(1-3), 3] = [-2, 3]
    try std.testing.expectEqual(@as(usize, 2), table.size());
    const expected_0 = F.one().sub(r0);
    const expected_1 = r0;
    try std.testing.expect(table.get(0).eql(expected_0));
    try std.testing.expect(table.get(1).eql(expected_1));

    // Sum should be 1 (always)
    const sum = table.sum();
    try std.testing.expect(sum.eql(F.one()));
}

test "expanding table multiple binds" {
    const F = @import("../../field/mod.zig").BN254Scalar;
    const allocator = std.testing.allocator;

    var table = try ExpandingTable(F).init(allocator, 4);
    defer table.deinit();

    // Bind several variables
    const challenges = [_]F{
        F.fromU64(2),
        F.fromU64(3),
        F.fromU64(5),
    };

    for (challenges) |r| {
        try table.bind(r);
    }

    // Should have 2^3 = 8 entries
    try std.testing.expectEqual(@as(usize, 8), table.size());

    // Sum should still be 1
    const sum = table.sum();
    try std.testing.expect(sum.eql(F.one()));

    // Verify the table contains eq(r, x) evaluations
    // eq(r, 000) = (1-r0)(1-r1)(1-r2) = (1-2)(1-3)(1-5) = (-1)(-2)(-4) = -8
    // eq(r, 001) = (r0)(1-r1)(1-r2) = (2)(-2)(-4) = 16
    // etc.
    const one = F.one();
    const r0 = challenges[0];
    const r1 = challenges[1];
    const r2 = challenges[2];

    const one_m_r0 = one.sub(r0);
    const one_m_r1 = one.sub(r1);
    const one_m_r2 = one.sub(r2);

    // Index 0 = 000: (1-r0)(1-r1)(1-r2)
    const expected_000 = one_m_r0.mul(one_m_r1).mul(one_m_r2);
    try std.testing.expect(table.get(0).eql(expected_000));

    // Index 1 = 001: r0 * (1-r1) * (1-r2)
    const expected_001 = r0.mul(one_m_r1).mul(one_m_r2);
    try std.testing.expect(table.get(1).eql(expected_001));

    // Index 7 = 111: r0 * r1 * r2
    const expected_111 = r0.mul(r1).mul(r2);
    try std.testing.expect(table.get(7).eql(expected_111));
}

test "expanding table condense" {
    const F = @import("../../field/mod.zig").BN254Scalar;
    const allocator = std.testing.allocator;

    var table = try ExpandingTable(F).init(allocator, 4);
    defer table.deinit();

    // Bind 2 variables
    try table.bind(F.fromU64(2));
    try table.bind(F.fromU64(3));

    // Now table has 4 entries
    try std.testing.expectEqual(@as(usize, 4), table.size());

    // Create weights: [1, 2, 3, 4]
    const weights = [_]F{
        F.fromU64(1),
        F.fromU64(2),
        F.fromU64(3),
        F.fromU64(4),
    };

    // Condense to 1 bit (2 outputs)
    const condensed = try table.condense(allocator, &weights, 1);
    defer allocator.free(condensed);

    try std.testing.expectEqual(@as(usize, 2), condensed.len);

    // condensed[0] = table[0]*w[0] + table[1]*w[1]
    // condensed[1] = table[2]*w[2] + table[3]*w[3]
    const expected_0 = table.get(0).mul(weights[0]).add(table.get(1).mul(weights[1]));
    const expected_1 = table.get(2).mul(weights[2]).add(table.get(3).mul(weights[3]));

    try std.testing.expect(condensed[0].eql(expected_0));
    try std.testing.expect(condensed[1].eql(expected_1));
}
