//! Polynomial operations for Jolt
//!
//! This module provides polynomial types and operations used in the Jolt proof system,
//! including multilinear polynomials, equality polynomials, and commitment schemes.

const std = @import("std");
const Allocator = std.mem.Allocator;
const field = @import("../field/mod.zig");

pub const commitment = @import("commitment/mod.zig");
pub const split_eq = @import("split_eq.zig");
pub const multiquadratic = @import("multiquadratic.zig");

// Re-export commonly used types
pub const GruenSplitEqPolynomial = split_eq.GruenSplitEqPolynomial;
pub const MultiquadraticPolynomial = multiquadratic.MultiquadraticPolynomial;
pub const GridValue = multiquadratic.GridValue;

/// Dense multilinear polynomial representation
///
/// A multilinear polynomial over n variables is stored as 2^n evaluations
/// at the boolean hypercube points.
pub fn DensePolynomial(comptime F: type) type {
    return struct {
        const Self = @This();

        /// Evaluations at boolean hypercube points
        evaluations: []F,
        /// Number of variables
        num_vars: usize,
        /// Allocator used for this polynomial
        allocator: Allocator,

        /// Create a new polynomial from evaluations
        pub fn init(allocator: Allocator, evaluations: []const F) !Self {
            const num_vars = std.math.log2_int(usize, evaluations.len);
            std.debug.assert(evaluations.len == (@as(usize, 1) << num_vars));

            const evals = try allocator.alloc(F, evaluations.len);
            @memcpy(evals, evaluations);

            return .{
                .evaluations = evals,
                .num_vars = num_vars,
                .allocator = allocator,
            };
        }

        /// Create a zero polynomial with n variables
        pub fn zero(allocator: Allocator, num_vars: usize) !Self {
            const size = @as(usize, 1) << num_vars;
            const evals = try allocator.alloc(F, size);
            @memset(evals, F.zero());

            return .{
                .evaluations = evals,
                .num_vars = num_vars,
                .allocator = allocator,
            };
        }

        /// Free the polynomial
        pub fn deinit(self: *Self) void {
            self.allocator.free(self.evaluations);
        }

        /// Get the number of evaluations (2^num_vars)
        pub fn len(self: *const Self) usize {
            return self.evaluations.len;
        }

        /// Evaluate the polynomial at a point
        pub fn evaluate(self: *const Self, point: []const F) F {
            std.debug.assert(point.len == self.num_vars);

            // Use the multilinear extension formula
            var result = F.zero();
            for (self.evaluations, 0..) |eval, i| {
                var term = eval;
                for (0..self.num_vars) |j| {
                    const shift_amount: u6 = @intCast(j);
                    const bit = (i >> shift_amount) & 1;
                    if (bit == 1) {
                        term = term.mul(point[j]);
                    } else {
                        term = term.mul(F.one().sub(point[j]));
                    }
                }
                result = result.add(term);
            }
            return result;
        }

        /// Add two polynomials
        pub fn add(self: *const Self, other: *const Self) !Self {
            std.debug.assert(self.num_vars == other.num_vars);
            const evals = try self.allocator.alloc(F, self.evaluations.len);

            for (0..self.evaluations.len) |i| {
                evals[i] = self.evaluations[i].add(other.evaluations[i]);
            }

            return .{
                .evaluations = evals,
                .num_vars = self.num_vars,
                .allocator = self.allocator,
            };
        }

        /// Multiply by a scalar
        pub fn scale(self: *const Self, scalar: F) !Self {
            const evals = try self.allocator.alloc(F, self.evaluations.len);

            for (0..self.evaluations.len) |i| {
                evals[i] = self.evaluations[i].mul(scalar);
            }

            return .{
                .evaluations = evals,
                .num_vars = self.num_vars,
                .allocator = self.allocator,
            };
        }

        /// Bind the first variable to a value
        /// This reduces the polynomial from n to n-1 variables
        pub fn bindFirst(self: *const Self, value: F) !Self {
            std.debug.assert(self.num_vars > 0);

            const new_size = self.evaluations.len / 2;
            const evals = try self.allocator.alloc(F, new_size);

            const one_minus_value = F.one().sub(value);

            for (0..new_size) |i| {
                // f(x_1, ..., x_n) evaluated at x_1 = value
                // = (1 - value) * f(0, x_2, ..., x_n) + value * f(1, x_2, ..., x_n)
                const low = self.evaluations[i].mul(one_minus_value);
                const high = self.evaluations[i + new_size].mul(value);
                evals[i] = low.add(high);
            }

            return .{
                .evaluations = evals,
                .num_vars = self.num_vars - 1,
                .allocator = self.allocator,
            };
        }
    };
}

/// Equality polynomial eq(x, r)
///
/// eq(x, r) = prod_{i=1}^{n} (x_i * r_i + (1 - x_i) * (1 - r_i))
///
/// This polynomial evaluates to 1 when x = r and 0 when x and r differ
/// in any coordinate on the boolean hypercube.
pub fn EqPolynomial(comptime F: type) type {
    return struct {
        const Self = @This();

        /// The point r that defines the equality polynomial
        r: []F,
        allocator: Allocator,

        /// Create an equality polynomial for point r
        pub fn init(allocator: Allocator, r: []const F) !Self {
            const r_copy = try allocator.alloc(F, r.len);
            @memcpy(r_copy, r);

            return .{
                .r = r_copy,
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.r);
        }

        /// Evaluate eq(x, r) at a point x
        pub fn evaluate(self: *const Self, x: []const F) F {
            std.debug.assert(x.len == self.r.len);

            var result = F.one();
            for (0..self.r.len) |i| {
                // x_i * r_i + (1 - x_i) * (1 - r_i)
                const xi_ri = x[i].mul(self.r[i]);
                const one_minus_xi = F.one().sub(x[i]);
                const one_minus_ri = F.one().sub(self.r[i]);
                const term = xi_ri.add(one_minus_xi.mul(one_minus_ri));
                result = result.mul(term);
            }
            return result;
        }

        /// Compute all evaluations on the boolean hypercube
        pub fn evals(self: *const Self, allocator: Allocator) ![]F {
            const n = self.r.len;
            const size = @as(usize, 1) << @as(u6, @intCast(n));
            const result = try allocator.alloc(F, size);

            // Start with eq evaluated at all zeros
            result[0] = F.one();
            for (0..n) |i| {
                const one_minus_ri = F.one().sub(self.r[i]);
                result[0] = result[0].mul(one_minus_ri);
            }

            // Build up using the recurrence relation
            for (0..n) |i| {
                const half = @as(usize, 1) << @as(u6, @intCast(i));
                const factor = self.r[i].mul(F.one().sub(self.r[i]).inverse().?);

                for (0..half) |j| {
                    result[j + half] = result[j].mul(factor);
                }
            }

            return result;
        }
    };
}

/// Univariate polynomial (used in sumcheck)
pub fn UniPoly(comptime F: type) type {
    return struct {
        const Self = @This();

        /// Coefficients in monomial basis (constant term first)
        coeffs: []F,
        allocator: Allocator,

        /// Create from coefficients
        pub fn init(allocator: Allocator, coeffs: []const F) !Self {
            const c = try allocator.alloc(F, coeffs.len);
            @memcpy(c, coeffs);

            return .{
                .coeffs = c,
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.coeffs);
        }

        /// Evaluate at a point using Horner's method
        pub fn evaluate(self: *const Self, x: F) F {
            if (self.coeffs.len == 0) return F.zero();

            var result = self.coeffs[self.coeffs.len - 1];
            var i = self.coeffs.len - 1;
            while (i > 0) {
                i -= 1;
                result = result.mul(x).add(self.coeffs[i]);
            }
            return result;
        }

        /// Get the degree
        pub fn degree(self: *const Self) usize {
            if (self.coeffs.len == 0) return 0;
            return self.coeffs.len - 1;
        }

        /// Interpolate a degree-3 polynomial from evaluations at 0, 1, 2, 3
        ///
        /// Given p(0), p(1), p(2), p(3), returns coefficients [c0, c1, c2, c3]
        /// where p(X) = c0 + c1*X + c2*X² + c3*X³
        ///
        /// Uses the explicit inverse of the Vandermonde matrix.
        pub fn interpolateDegree3(evals: [4]F) [4]F {
            const p0 = evals[0];
            const p1 = evals[1];
            const p2 = evals[2];
            const p3 = evals[3];

            // c0 = p(0)
            const c0 = p0;

            // For the other coefficients, we solve the Vandermonde system
            // The inverse of the 4x4 Vandermonde matrix at points 0,1,2,3 gives:
            //
            // c1 = (-11*p0 + 18*p1 - 9*p2 + 2*p3) / 6
            // c2 = (2*p0 - 5*p1 + 4*p2 - p3) / 2
            // c3 = (-p0 + 3*p1 - 3*p2 + p3) / 6
            //
            // We compute these using field arithmetic

            // Compute 1/6 and 1/2 as field inverses
            const inv6 = F.fromU64(6).inverse().?;
            const inv2 = F.fromU64(2).inverse().?;

            // c1 = (-11*p0 + 18*p1 - 9*p2 + 2*p3) / 6
            const c1_num = F.zero()
                .sub(F.fromU64(11).mul(p0))
                .add(F.fromU64(18).mul(p1))
                .sub(F.fromU64(9).mul(p2))
                .add(F.fromU64(2).mul(p3));
            const c1 = c1_num.mul(inv6);

            // c2 = (2*p0 - 5*p1 + 4*p2 - p3) / 2
            const c2_num = F.fromU64(2).mul(p0)
                .sub(F.fromU64(5).mul(p1))
                .add(F.fromU64(4).mul(p2))
                .sub(p3);
            const c2 = c2_num.mul(inv2);

            // c3 = (-p0 + 3*p1 - 3*p2 + p3) / 6
            const c3_num = F.zero().sub(p0)
                .add(F.fromU64(3).mul(p1))
                .sub(F.fromU64(3).mul(p2))
                .add(p3);
            const c3 = c3_num.mul(inv6);

            return [4]F{ c0, c1, c2, c3 };
        }

        /// Convert evaluations at 0,1,2,3 to Jolt's compressed format [c0, c2, c3]
        ///
        /// Jolt stores coefficients except the linear term, which is recovered from the hint.
        pub fn evalsToCompressed(evals: [4]F) [3]F {
            const coeffs = interpolateDegree3(evals);
            return [3]F{ coeffs[0], coeffs[2], coeffs[3] };
        }
    };
}

test "unipoly interpolate degree 3" {
    const F = field.BN254Scalar;

    // Test: p(X) = 1 + 2X + 3X² + 4X³
    // p(0) = 1
    // p(1) = 1 + 2 + 3 + 4 = 10
    // p(2) = 1 + 4 + 12 + 32 = 49
    // p(3) = 1 + 6 + 27 + 108 = 142
    const evals = [4]F{
        F.fromU64(1),
        F.fromU64(10),
        F.fromU64(49),
        F.fromU64(142),
    };

    const coeffs = UniPoly(F).interpolateDegree3(evals);

    try std.testing.expect(coeffs[0].eql(F.fromU64(1)));
    try std.testing.expect(coeffs[1].eql(F.fromU64(2)));
    try std.testing.expect(coeffs[2].eql(F.fromU64(3)));
    try std.testing.expect(coeffs[3].eql(F.fromU64(4)));
}

test "unipoly compressed format" {
    const F = field.BN254Scalar;

    // Same polynomial: p(X) = 1 + 2X + 3X² + 4X³
    const evals = [4]F{
        F.fromU64(1),
        F.fromU64(10),
        F.fromU64(49),
        F.fromU64(142),
    };

    const compressed = UniPoly(F).evalsToCompressed(evals);

    // Should be [c0, c2, c3] = [1, 3, 4]
    try std.testing.expect(compressed[0].eql(F.fromU64(1)));
    try std.testing.expect(compressed[1].eql(F.fromU64(3)));
    try std.testing.expect(compressed[2].eql(F.fromU64(4)));
}

test "dense polynomial basic" {
    const F = field.BN254Scalar;
    const allocator = std.testing.allocator;

    // Create a simple polynomial with 2 variables (4 evaluations)
    const evals = [_]F{
        F.fromU64(1), // f(0,0)
        F.fromU64(2), // f(1,0)
        F.fromU64(3), // f(0,1)
        F.fromU64(4), // f(1,1)
    };

    var poly = try DensePolynomial(F).init(allocator, &evals);
    defer poly.deinit();

    try std.testing.expectEqual(@as(usize, 2), poly.num_vars);
    try std.testing.expectEqual(@as(usize, 4), poly.len());
}
