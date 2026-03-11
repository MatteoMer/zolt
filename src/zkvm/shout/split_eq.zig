//! Split EQ Polynomial (Gruen's Optimization)
//!
//! This module implements Gruen's optimization for evaluating the EQ polynomial
//! efficiently during sumcheck. The key insight is that for variables split into
//! "outer" and "inner" groups, we can factor:
//!
//!   eq(w, x) = eq(w_out, x_out) * eq(w_in, x_in)
//!
//! By caching the prefix tables for both groups, we can evaluate sums of the form:
//!   sum_j eq(r, j) * f(j)
//! more efficiently than the naive O(T) approach.
//!
//! The split structure is particularly useful for Jolt's multi-phase sumcheck,
//! where the number of cycle variables (log_T) is typically much smaller than
//! the number of address variables (LOG_K).
//!
//! Reference: Dao-Thaler (2024) and Gruen's optimizations

const std = @import("std");
const Allocator = std.mem.Allocator;

/// SplitEqPolynomial - Caches EQ evaluations for efficient inner product computation
///
/// Splits variables into "outer" (high-order) and "inner" (low-order) groups,
/// caching the prefix tables for each group.
pub fn SplitEqPolynomial(comptime F: type) type {
    return struct {
        const Self = @This();

        /// Pair type for precomputed (1-w_i, w_i) values
        pub const WPair = struct {
            one_minus: F,
            val: F,
        };

        /// Total number of variables
        num_vars: usize,
        /// Number of outer variables (high-order bits)
        num_outer: usize,
        /// Number of inner variables (low-order bits)
        num_inner: usize,

        /// Challenge point w (the "target" for eq(w, x))
        w: []F,

        /// Cached table for outer variables
        /// E_out[k] = eq(w_out[0..k], ·) evaluated over {0,1}^k
        E_out: []F,

        /// Cached table for inner variables
        /// E_in[k] = eq(w_in[0..k], ·) evaluated over {0,1}^k
        E_in: []F,

        /// Precomputed (1-w_i, w_i) pairs for efficient binding
        w_pairs: []WPair,

        /// Current round within outer/inner groups
        outer_round: usize,
        inner_round: usize,

        allocator: Allocator,

        /// Create a new SplitEqPolynomial for challenge point w
        ///
        /// @param num_outer: Number of high-order (outer) variables
        /// @param num_inner: Number of low-order (inner) variables
        /// @param w: Challenge point of length (num_outer + num_inner)
        pub fn init(allocator: Allocator, num_outer: usize, num_inner: usize, w: []const F) !Self {
            const num_vars = num_outer + num_inner;
            std.debug.assert(w.len == num_vars);

            // Copy challenge point
            const w_copy = try allocator.alloc(F, num_vars);
            @memcpy(w_copy, w);

            // Precompute (1-w_i, w_i) pairs
            const w_pairs = try allocator.alloc(WPair, num_vars);
            for (w, 0..) |wi, i| {
                w_pairs[i] = WPair{
                    .one_minus = F.one().sub(wi),
                    .val = wi,
                };
            }

            // Allocate outer table (size 2^num_outer)
            const outer_size = @as(usize, 1) << @intCast(num_outer);
            const E_out = try allocator.alloc(F, outer_size);

            // Allocate inner table (size 2^num_inner)
            const inner_size = @as(usize, 1) << @intCast(num_inner);
            const E_in = try allocator.alloc(F, inner_size);

            // Initialize both tables
            var result = Self{
                .num_vars = num_vars,
                .num_outer = num_outer,
                .num_inner = num_inner,
                .w = w_copy,
                .E_out = E_out,
                .E_in = E_in,
                .w_pairs = w_pairs,
                .outer_round = 0,
                .inner_round = 0,
                .allocator = allocator,
            };

            try result.buildTables();
            return result;
        }

        /// Free all resources
        pub fn deinit(self: *Self) void {
            self.allocator.free(self.w);
            self.allocator.free(self.E_out);
            self.allocator.free(self.E_in);
            self.allocator.free(self.w_pairs);
        }

        /// Build the initial EQ tables
        fn buildTables(self: *Self) !void {
            // Build outer table: eq(w_out, x_out) for all x_out in {0,1}^num_outer
            self.E_out[0] = F.one();
            for (0..self.num_outer) |i| {
                const half = @as(usize, 1) << @intCast(i);
                const pair = self.w_pairs[i];
                for (0..half) |j| {
                    self.E_out[j + half] = self.E_out[j].mul(pair.val);
                    self.E_out[j] = self.E_out[j].mul(pair.one_minus);
                }
            }

            // Build inner table: eq(w_in, x_in) for all x_in in {0,1}^num_inner
            self.E_in[0] = F.one();
            for (0..self.num_inner) |i| {
                const half = @as(usize, 1) << @intCast(i);
                const pair = self.w_pairs[self.num_outer + i];
                for (0..half) |j| {
                    self.E_in[j + half] = self.E_in[j].mul(pair.val);
                    self.E_in[j] = self.E_in[j].mul(pair.one_minus);
                }
            }
        }

        /// Get eq(w_out, x_out) for a specific outer index
        pub fn getOuterEq(self: *const Self, outer_idx: usize) F {
            std.debug.assert(outer_idx < self.E_out.len);
            return self.E_out[outer_idx];
        }

        /// Get eq(w_in, x_in) for a specific inner index
        pub fn getInnerEq(self: *const Self, inner_idx: usize) F {
            std.debug.assert(inner_idx < self.E_in.len);
            return self.E_in[inner_idx];
        }

        /// Get the full eq(w, x) = eq(w_out, x_out) * eq(w_in, x_in)
        pub fn getFullEq(self: *const Self, outer_idx: usize, inner_idx: usize) F {
            return self.getOuterEq(outer_idx).mul(self.getInnerEq(inner_idx));
        }

        /// Get eq(w, j) for a linear index j (where j encodes both outer and inner)
        pub fn getEq(self: *const Self, j: usize) F {
            const inner_mask = (@as(usize, 1) << @intCast(self.num_inner)) - 1;
            const inner_idx = j & inner_mask;
            const outer_idx = j >> @intCast(self.num_inner);
            return self.getFullEq(outer_idx, inner_idx);
        }

        /// Compute sum_{j=0}^{T-1} eq(w, j) * f(j)
        /// This is a weighted inner product using the cached EQ values
        pub fn innerProduct(self: *const Self, f: []const F) F {
            std.debug.assert(f.len == self.E_out.len * self.E_in.len);

            var result = F.zero();
            const inner_size = self.E_in.len;

            for (0..self.E_out.len) |outer_idx| {
                const e_out = self.E_out[outer_idx];
                const base = outer_idx * inner_size;

                for (0..inner_size) |inner_idx| {
                    const e_in = self.E_in[inner_idx];
                    const eq_val = e_out.mul(e_in);
                    result = result.add(eq_val.mul(f[base + inner_idx]));
                }
            }
            return result;
        }

        /// Compute inner product over just the outer variables
        /// Returns an array of partial sums indexed by inner_idx
        pub fn outerInnerProduct(self: *const Self, allocator: Allocator, f: []const F) ![]F {
            std.debug.assert(f.len == self.E_out.len * self.E_in.len);

            const inner_size = self.E_in.len;
            const result = try allocator.alloc(F, inner_size);
            @memset(result, F.zero());

            for (0..self.E_out.len) |outer_idx| {
                const e_out = self.E_out[outer_idx];
                const base = outer_idx * inner_size;

                for (0..inner_size) |inner_idx| {
                    result[inner_idx] = result[inner_idx].add(e_out.mul(f[base + inner_idx]));
                }
            }

            return result;
        }

        /// Bind an outer variable to a challenge value
        /// Updates the E_out table for the next round
        pub fn bindOuter(self: *Self, r: F) void {
            std.debug.assert(self.outer_round < self.num_outer);

            const current_size = self.E_out.len >> @intCast(self.outer_round);
            const new_size = current_size / 2;
            const offset = self.E_out.len - current_size;

            const one_minus_r = F.one().sub(r);

            // Combine pairs: new[i] = old[2i] * (1-r) + old[2i+1] * r
            for (0..new_size) |i| {
                const old_0 = self.E_out[offset + 2 * i];
                const old_1 = self.E_out[offset + 2 * i + 1];
                self.E_out[offset + i] = old_0.mul(one_minus_r).add(old_1.mul(r));
            }

            self.outer_round += 1;
        }

        /// Bind an inner variable to a challenge value
        pub fn bindInner(self: *Self, r: F) void {
            std.debug.assert(self.inner_round < self.num_inner);

            const current_size = self.E_in.len >> @intCast(self.inner_round);
            const new_size = current_size / 2;
            const offset = self.E_in.len - current_size;

            const one_minus_r = F.one().sub(r);

            for (0..new_size) |i| {
                const old_0 = self.E_in[offset + 2 * i];
                const old_1 = self.E_in[offset + 2 * i + 1];
                self.E_in[offset + i] = old_0.mul(one_minus_r).add(old_1.mul(r));
            }

            self.inner_round += 1;
        }

        /// Get the current outer table slice (after bindings)
        pub fn currentOuterSlice(self: *const Self) []const F {
            const current_size = self.E_out.len >> @intCast(self.outer_round);
            const offset = self.E_out.len - current_size;
            return self.E_out[offset..];
        }

        /// Get the current inner table slice (after bindings)
        pub fn currentInnerSlice(self: *const Self) []const F {
            const current_size = self.E_in.len >> @intCast(self.inner_round);
            const offset = self.E_in.len - current_size;
            return self.E_in[offset..];
        }

        /// Clone the SplitEqPolynomial
        pub fn clone(self: *const Self) !Self {
            const w_copy = try self.allocator.alloc(F, self.w.len);
            @memcpy(w_copy, self.w);

            const E_out_copy = try self.allocator.alloc(F, self.E_out.len);
            @memcpy(E_out_copy, self.E_out);

            const E_in_copy = try self.allocator.alloc(F, self.E_in.len);
            @memcpy(E_in_copy, self.E_in);

            const w_pairs_copy = try self.allocator.alloc(struct { one_minus: F, val: F }, self.w_pairs.len);
            @memcpy(w_pairs_copy, self.w_pairs);

            return Self{
                .num_vars = self.num_vars,
                .num_outer = self.num_outer,
                .num_inner = self.num_inner,
                .w = w_copy,
                .E_out = E_out_copy,
                .E_in = E_in_copy,
                .w_pairs = w_pairs_copy,
                .outer_round = self.outer_round,
                .inner_round = self.inner_round,
                .allocator = self.allocator,
            };
        }
    };
}

test "split eq basic" {
    const F = @import("../../field/mod.zig").BN254Scalar;
    const allocator = std.testing.allocator;

    // Create with 2 outer and 2 inner variables
    const w = [_]F{
        F.fromU64(2), // w_out[0]
        F.fromU64(3), // w_out[1]
        F.fromU64(5), // w_in[0]
        F.fromU64(7), // w_in[1]
    };

    var split_eq = try SplitEqPolynomial(F).init(allocator, 2, 2, &w);
    defer split_eq.deinit();

    // Verify eq(w, 0) = eq(w_out, 0) * eq(w_in, 0)
    // = (1-2)(1-3) * (1-5)(1-7)
    // = (-1)(-2) * (-4)(-6) = 2 * 24 = 48
    const eq_0 = split_eq.getEq(0);
    const one = F.one();
    const expected_out = one.sub(w[0]).mul(one.sub(w[1]));
    const expected_in = one.sub(w[2]).mul(one.sub(w[3]));
    const expected = expected_out.mul(expected_in);
    try std.testing.expect(eq_0.eql(expected));

    // Verify sum of all EQ values = 1
    var sum = F.zero();
    for (0..16) |j| {
        sum = sum.add(split_eq.getEq(j));
    }
    try std.testing.expect(sum.eql(F.one()));
}

test "split eq inner product" {
    const F = @import("../../field/mod.zig").BN254Scalar;
    const allocator = std.testing.allocator;

    // Simple case: 1 outer, 1 inner
    const w = [_]F{
        F.fromU64(2), // w_out
        F.fromU64(3), // w_in
    };

    var split_eq = try SplitEqPolynomial(F).init(allocator, 1, 1, &w);
    defer split_eq.deinit();

    // f = [1, 2, 3, 4]
    const f = [_]F{
        F.fromU64(1),
        F.fromU64(2),
        F.fromU64(3),
        F.fromU64(4),
    };

    const result = split_eq.innerProduct(&f);

    // Manual calculation:
    // eq(w, 0) = (1-2)(1-3) = 2
    // eq(w, 1) = (2)(1-3) = -4
    // eq(w, 2) = (1-2)(3) = -3
    // eq(w, 3) = (2)(3) = 6
    // inner product = 1*2 + 2*(-4) + 3*(-3) + 4*6 = 2 - 8 - 9 + 24 = 9
    const one = F.one();
    const e00 = one.sub(w[0]).mul(one.sub(w[1]));
    const e01 = w[0].mul(one.sub(w[1]));
    const e10 = one.sub(w[0]).mul(w[1]);
    const e11 = w[0].mul(w[1]);

    const expected = e00.mul(f[0]).add(e01.mul(f[1])).add(e10.mul(f[2])).add(e11.mul(f[3]));
    try std.testing.expect(result.eql(expected));
}

test "split eq binding" {
    const F = @import("../../field/mod.zig").BN254Scalar;
    const allocator = std.testing.allocator;

    // 2 outer, 2 inner variables
    const w = [_]F{
        F.fromU64(2),
        F.fromU64(3),
        F.fromU64(5),
        F.fromU64(7),
    };

    var split_eq = try SplitEqPolynomial(F).init(allocator, 2, 2, &w);
    defer split_eq.deinit();

    // Bind first outer variable to r0 = 4
    split_eq.bindOuter(F.fromU64(4));

    // Current outer slice should have 2 elements
    const outer_slice = split_eq.currentOuterSlice();
    try std.testing.expectEqual(@as(usize, 2), outer_slice.len);

    // After binding, the values should be:
    // new[i] = old[2i] * (1-r) + old[2i+1] * r
    // = eq(w_out, 2i) * (1-4) + eq(w_out, 2i+1) * 4
}
