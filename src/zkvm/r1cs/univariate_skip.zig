//! Univariate Skip Optimization for Jolt-Compatible Sumcheck
//!
//! This module implements Jolt's "univariate skip" optimization for the first round
//! of sumcheck in stages 1 and 2. The optimization reduces the number of rounds by
//! encoding multiple constraint evaluations into a single high-degree polynomial.
//!
//! ## Key Concepts
//!
//! 1. **Constraint Grouping**: 19 R1CS constraints are split into:
//!    - First group: 10 constraints (boolean guards, ~64-bit Bz)
//!    - Second group: 9 constraints (wider arithmetic, ~128-bit Bz)
//!
//! 2. **Extended Domain**: Instead of evaluating at just {0, 1}, we evaluate
//!    on a symmetric domain {-DEGREE, ..., -1, 0, 1, ..., DEGREE}
//!
//! 3. **First-Round Polynomial**: s1(Y) = L(tau_high, Y) * t1(Y)
//!    - t1(Y) is interpolated from constraint evaluations on extended domain
//!    - L(tau_high, Y) is the Lagrange kernel polynomial
//!    - Result has degree at most 3*DEGREE (= 27 for DEGREE=9)
//!
//! Reference: jolt-core/src/subprotocols/univariate_skip.rs

const std = @import("std");
const Allocator = std.mem.Allocator;

/// Number of R1CS constraints
pub const NUM_R1CS_CONSTRAINTS: usize = 19;

/// Degree of univariate skip = (NUM_R1CS_CONSTRAINTS - 1) / 2 = 9
pub const OUTER_UNIVARIATE_SKIP_DEGREE: usize = (NUM_R1CS_CONSTRAINTS - 1) / 2;

/// Domain size = DEGREE + 1 = 10
pub const OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE: usize = OUTER_UNIVARIATE_SKIP_DEGREE + 1;

/// Extended domain size = 2 * DEGREE + 1 = 19
pub const OUTER_UNIVARIATE_SKIP_EXTENDED_DOMAIN_SIZE: usize = 2 * OUTER_UNIVARIATE_SKIP_DEGREE + 1;

/// Number of coefficients in first-round polynomial = 3 * DEGREE + 1 = 28
pub const OUTER_FIRST_ROUND_POLY_NUM_COEFFS: usize = 3 * OUTER_UNIVARIATE_SKIP_DEGREE + 1;

/// Degree bound of first-round polynomial = 27
pub const OUTER_FIRST_ROUND_POLY_DEGREE_BOUND: usize = OUTER_FIRST_ROUND_POLY_NUM_COEFFS - 1;

/// Number of remaining constraints in second group = 9
pub const NUM_REMAINING_R1CS_CONSTRAINTS: usize = NUM_R1CS_CONSTRAINTS - OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE;

/// Product virtualization constants (Stage 2)
pub const NUM_PRODUCT_VIRTUAL: usize = 5;
pub const PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DOMAIN_SIZE: usize = NUM_PRODUCT_VIRTUAL;
pub const PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DEGREE: usize = NUM_PRODUCT_VIRTUAL - 1;
pub const PRODUCT_VIRTUAL_UNIVARIATE_SKIP_EXTENDED_DOMAIN_SIZE: usize = 2 * PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DEGREE + 1;
pub const PRODUCT_VIRTUAL_FIRST_ROUND_POLY_NUM_COEFFS: usize = 3 * PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DEGREE + 1;
pub const PRODUCT_VIRTUAL_FIRST_ROUND_POLY_DEGREE_BOUND: usize = PRODUCT_VIRTUAL_FIRST_ROUND_POLY_NUM_COEFFS - 1;

/// Univariate polynomial representation
pub fn UniPoly(comptime F: type) type {
    return struct {
        const Self = @This();

        /// Polynomial coefficients in ascending order: c0 + c1*x + c2*x^2 + ...
        coeffs: []F,
        allocator: Allocator,

        pub fn init(allocator: Allocator, coeffs: []const F) !Self {
            const owned = try allocator.alloc(F, coeffs.len);
            @memcpy(owned, coeffs);
            return Self{
                .coeffs = owned,
                .allocator = allocator,
            };
        }

        pub fn initZero(allocator: Allocator, deg: usize) !Self {
            const coeffs = try allocator.alloc(F, deg + 1);
            @memset(coeffs, F.zero());
            return Self{
                .coeffs = coeffs,
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.coeffs);
        }

        /// Evaluate polynomial at point x using Horner's method
        pub fn evaluate(self: *const Self, x: F) F {
            if (self.coeffs.len == 0) return F.zero();

            var result = self.coeffs[self.coeffs.len - 1];
            var i: usize = self.coeffs.len - 1;
            while (i > 0) {
                i -= 1;
                result = result.mul(x).add(self.coeffs[i]);
            }
            return result;
        }

        /// Return degree of polynomial
        pub fn degree(self: *const Self) usize {
            if (self.coeffs.len == 0) return 0;
            return self.coeffs.len - 1;
        }

        /// Multiply two polynomials
        pub fn mul(self: *const Self, other: *const Self, allocator: Allocator) !Self {
            if (self.coeffs.len == 0 or other.coeffs.len == 0) {
                return Self.initZero(allocator, 0);
            }

            const new_degree = self.degree() + other.degree();
            var result = try Self.initZero(allocator, new_degree);

            for (self.coeffs, 0..) |a, i| {
                for (other.coeffs, 0..) |b, j| {
                    result.coeffs[i + j] = result.coeffs[i + j].add(a.mul(b));
                }
            }

            return result;
        }

        /// Sum of evaluations on symmetric domain {-k, ..., -1, 0, 1, ..., k}
        /// where k = domain_half
        pub fn sumOverSymmetricDomain(self: *const Self, domain_half: usize) F {
            var sum = F.zero();

            // Evaluate at 0
            sum = sum.add(self.evaluate(F.zero()));

            // Evaluate at +-1, +-2, ..., +-domain_half
            var i: i64 = 1;
            while (i <= @as(i64, @intCast(domain_half))) : (i += 1) {
                // Positive point
                var x_pos = F.zero();
                var j: i64 = 0;
                while (j < i) : (j += 1) {
                    x_pos = x_pos.add(F.one());
                }
                sum = sum.add(self.evaluate(x_pos));

                // Negative point
                const x_neg = F.zero().sub(x_pos);
                sum = sum.add(self.evaluate(x_neg));
            }

            return sum;
        }
    };
}

/// Compute the interleaved symmetric univariate-skip target indices outside the base window.
///
/// For domain size 10 and degree 9:
/// - Base window: {-4, -3, -2, -1, 0, 1, 2, 3, 4, 5}
/// - Extended points: {-9, 6, -8, 7, -7, 8, -6, 9, -5} (interleaved)
pub fn uniskipTargets(
    comptime DOMAIN_SIZE: usize,
    comptime DEGREE: usize,
) [DEGREE]i64 {
    const base_left: i64 = -@as(i64, @intCast((DOMAIN_SIZE - 1) / 2));
    const base_right: i64 = base_left + @as(i64, @intCast(DOMAIN_SIZE)) - 1;
    const ext_left: i64 = -@as(i64, @intCast(DEGREE));
    const ext_right: i64 = @as(i64, @intCast(DEGREE));

    var targets: [DEGREE]i64 = undefined;
    var idx: usize = 0;
    var n = base_left - 1;
    var p = base_right + 1;

    while (n >= ext_left and p <= ext_right and idx < DEGREE) {
        targets[idx] = n;
        idx += 1;
        if (idx >= DEGREE) break;
        targets[idx] = p;
        idx += 1;
        n -= 1;
        p += 1;
    }

    while (idx < DEGREE and n >= ext_left) {
        targets[idx] = n;
        idx += 1;
        n -= 1;
    }

    while (idx < DEGREE and p <= ext_right) {
        targets[idx] = p;
        idx += 1;
        p += 1;
    }

    return targets;
}

/// Lagrange polynomial utilities
pub fn LagrangePolynomial(comptime F: type) type {
    return struct {
        const Self = @This();

        /// Compute Lagrange basis polynomial evaluations at a point tau.
        /// For a symmetric domain {-k, ..., -1, 0, 1, ..., k} where k = (DOMAIN_SIZE-1)/2
        /// Returns L_i(tau) for i = 0, 1, ..., DOMAIN_SIZE-1
        pub fn evals(comptime DOMAIN_SIZE: usize, tau: F, allocator: Allocator) ![]F {
            const result = try allocator.alloc(F, DOMAIN_SIZE);

            // Domain points: {base_left, base_left+1, ..., base_right}
            const base_left_i: i64 = -@as(i64, @intCast((DOMAIN_SIZE - 1) / 2));

            // Compute L_i(tau) = prod_{j != i} (tau - x_j) / (x_i - x_j)
            for (0..DOMAIN_SIZE) |i| {
                const x_i = Self.fieldFromI64(base_left_i + @as(i64, @intCast(i)));
                var num = F.one();
                var den = F.one();

                for (0..DOMAIN_SIZE) |j| {
                    if (i == j) continue;
                    const x_j = Self.fieldFromI64(base_left_i + @as(i64, @intCast(j)));
                    num = num.mul(tau.sub(x_j));
                    den = den.mul(x_i.sub(x_j));
                }

                result[i] = num.mul(den.inv());
            }

            return result;
        }

        /// Interpolate polynomial coefficients from evaluations on symmetric domain
        /// Uses Lagrange interpolation
        pub fn interpolateCoeffs(comptime SIZE: usize, vals: []const F, allocator: Allocator) ![]F {
            if (vals.len != SIZE) return error.InvalidLength;

            // Domain points
            const base_left_i: i64 = -@as(i64, @intCast((SIZE - 1) / 2));

            // Result coefficients
            const coeffs = try allocator.alloc(F, SIZE);
            @memset(coeffs, F.zero());

            // For each evaluation point, compute its Lagrange basis polynomial
            // and add the weighted contribution
            for (0..SIZE) |i| {
                const x_i = Self.fieldFromI64(base_left_i + @as(i64, @intCast(i)));
                const y_i = vals[i];

                // Compute L_i(X) = prod_{j != i} (X - x_j) / (x_i - x_j)
                // We build this as a polynomial and accumulate

                // First compute the denominator (scalar)
                var den = F.one();
                for (0..SIZE) |j| {
                    if (i == j) continue;
                    const x_j = Self.fieldFromI64(base_left_i + @as(i64, @intCast(j)));
                    den = den.mul(x_i.sub(x_j));
                }
                const scale = y_i.mul(den.inv());

                // Build the numerator polynomial prod_{j != i} (X - x_j)
                // Start with 1, then multiply by (X - x_j) for each j != i
                var basis = try allocator.alloc(F, SIZE);
                defer allocator.free(basis);
                @memset(basis, F.zero());
                basis[0] = F.one();
                var basis_deg: usize = 0;

                for (0..SIZE) |j| {
                    if (i == j) continue;
                    const x_j = Self.fieldFromI64(base_left_i + @as(i64, @intCast(j)));
                    const neg_x_j = F.zero().sub(x_j);

                    // Multiply basis by (X - x_j) = X*basis + (-x_j)*basis
                    // Work backwards to avoid overwriting
                    var k: usize = basis_deg + 1;
                    while (k > 0) {
                        k -= 1;
                        const new_val = if (k > 0) basis[k - 1] else F.zero();
                        basis[k + 1] = basis[k].add(if (k + 1 <= basis_deg + 1) basis[k + 1].add(new_val) else new_val);
                    }
                    // Actually, let's do this properly with a temp array
                    var temp = try allocator.alloc(F, SIZE);
                    defer allocator.free(temp);
                    @memset(temp, F.zero());

                    for (0..basis_deg + 1) |idx| {
                        // (X - x_j) * basis = X * basis - x_j * basis
                        temp[idx + 1] = temp[idx + 1].add(basis[idx]); // X * c_k = c_k * X^{k+1}
                        temp[idx] = temp[idx].add(basis[idx].mul(neg_x_j)); // -x_j * c_k
                    }

                    @memcpy(basis[0..SIZE], temp);
                    basis_deg += 1;
                }

                // Add scaled basis to result
                for (0..SIZE) |k| {
                    coeffs[k] = coeffs[k].add(basis[k].mul(scale));
                }
            }

            return coeffs;
        }

        /// Convert i64 to field element (handling negatives)
        fn fieldFromI64(val: i64) F {
            if (val >= 0) {
                return F.fromU64(@intCast(val));
            } else {
                return F.zero().sub(F.fromU64(@intCast(-val)));
            }
        }
    };
}

/// Build the uni-skip first-round polynomial s1 from base and extended evaluations.
///
/// s1(Y) = L(tau_high, Y) * t1(Y)
/// where:
/// - t1(Y) is the underlying sumcheck polynomial (degree at most 2*DEGREE)
/// - L(tau_high, Y) is the Lagrange kernel over the base window (degree DOMAIN_SIZE-1)
/// - Result has degree at most 3*DEGREE (NUM_COEFFS = 3*DEGREE + 1)
pub fn buildUniskipFirstRoundPoly(
    comptime F: type,
    comptime DOMAIN_SIZE: usize,
    comptime DEGREE: usize,
    comptime EXTENDED_SIZE: usize,
    comptime NUM_COEFFS: usize,
    base_evals: ?[]const F,
    extended_evals: []const F,
    tau_high: F,
    allocator: Allocator,
) !UniPoly(F) {
    comptime {
        std.debug.assert(EXTENDED_SIZE == 2 * DEGREE + 1);
        std.debug.assert(NUM_COEFFS == 3 * DEGREE + 1);
    }

    // Get the target indices for extended evaluations
    const targets = comptime uniskipTargets(DOMAIN_SIZE, DEGREE);

    // Build t1 evaluations on the full extended symmetric window
    var t1_vals = try allocator.alloc(F, EXTENDED_SIZE);
    defer allocator.free(t1_vals);
    @memset(t1_vals, F.zero());

    // Fill in base window evaluations (if provided)
    if (base_evals) |base| {
        const base_left: i64 = -@as(i64, @intCast((DOMAIN_SIZE - 1) / 2));
        for (base, 0..) |val, i| {
            const z = base_left + @as(i64, @intCast(i));
            const pos: usize = @intCast(z + @as(i64, @intCast(DEGREE)));
            t1_vals[pos] = val;
        }
    }

    // Fill in extended evaluations (outside base window)
    for (extended_evals, 0..) |val, idx| {
        const z = targets[idx];
        const pos: usize = @intCast(z + @as(i64, @intCast(DEGREE)));
        t1_vals[pos] = val;
    }

    // Interpolate t1 coefficients from evaluations on extended grid
    const t1_coeffs = try LagrangePolynomial(F).interpolateCoeffs(EXTENDED_SIZE, t1_vals, allocator);
    defer allocator.free(t1_coeffs);

    // Compute Lagrange kernel values at tau_high
    const lagrange_values = try LagrangePolynomial(F).evals(DOMAIN_SIZE, tau_high, allocator);
    defer allocator.free(lagrange_values);

    // Interpolate Lagrange kernel coefficients
    const lagrange_coeffs = try LagrangePolynomial(F).interpolateCoeffs(DOMAIN_SIZE, lagrange_values, allocator);
    defer allocator.free(lagrange_coeffs);

    // Multiply polynomials: s1(Y) = L(tau_high, Y) * t1(Y)
    var s1_coeffs = try allocator.alloc(F, NUM_COEFFS);
    @memset(s1_coeffs, F.zero());

    for (lagrange_coeffs, 0..) |a, i| {
        for (t1_coeffs, 0..) |b, j| {
            if (i + j < NUM_COEFFS) {
                s1_coeffs[i + j] = s1_coeffs[i + j].add(a.mul(b));
            }
        }
    }

    return UniPoly(F){
        .coeffs = s1_coeffs,
        .allocator = allocator,
    };
}

/// UniSkipFirstRoundProof - Proof for univariate skip first round
pub fn UniSkipFirstRoundProof(comptime F: type) type {
    return struct {
        const Self = @This();

        /// The high-degree univariate polynomial for the first round
        uni_poly: UniPoly(F),

        pub fn init(uni_poly: UniPoly(F)) Self {
            return Self{ .uni_poly = uni_poly };
        }

        pub fn deinit(self: *Self) void {
            self.uni_poly.deinit();
        }

        /// Serialize in Jolt-compatible format
        /// All coefficients are written (no compression)
        pub fn serialize(self: *const Self, writer: anytype, comptime writeFieldElement: fn (anytype, F) anyerror!void) !void {
            // Write number of coefficients as u64
            try writer.writeInt(u64, @intCast(self.uni_poly.coeffs.len), .little);

            // Write all coefficients
            for (self.uni_poly.coeffs) |coeff| {
                try writeFieldElement(writer, coeff);
            }
        }
    };
}

// ============================================================================
// Tests
// ============================================================================

test "uniskip targets for stage 1 (DEGREE=9, DOMAIN_SIZE=10)" {
    const targets = comptime uniskipTargets(10, 9);

    // Base window is {-4, -3, -2, -1, 0, 1, 2, 3, 4, 5}
    // Extended points should be outside this range
    // Interleaved from left and right: {-5, 6, -6, 7, -7, 8, -8, 9, -9}
    try std.testing.expectEqual(@as(i64, -5), targets[0]);
    try std.testing.expectEqual(@as(i64, 6), targets[1]);
    try std.testing.expectEqual(@as(i64, -6), targets[2]);
    try std.testing.expectEqual(@as(i64, 7), targets[3]);
    try std.testing.expectEqual(@as(i64, -7), targets[4]);
    try std.testing.expectEqual(@as(i64, 8), targets[5]);
    try std.testing.expectEqual(@as(i64, -8), targets[6]);
    try std.testing.expectEqual(@as(i64, 9), targets[7]);
    try std.testing.expectEqual(@as(i64, -9), targets[8]);
}

test "uniskip targets for stage 2 (DEGREE=4, DOMAIN_SIZE=5)" {
    const targets = comptime uniskipTargets(5, 4);

    // Base window is {-2, -1, 0, 1, 2}
    // Extended points: {-3, 3, -4, 4}
    try std.testing.expectEqual(@as(i64, -3), targets[0]);
    try std.testing.expectEqual(@as(i64, 3), targets[1]);
    try std.testing.expectEqual(@as(i64, -4), targets[2]);
    try std.testing.expectEqual(@as(i64, 4), targets[3]);
}

test "lagrange polynomial evaluation" {
    const field = @import("../../field/mod.zig");
    const F = field.BN254Scalar;
    const allocator = std.testing.allocator;

    // For domain size 3: {-1, 0, 1}
    // L_0(x) = (x-0)(x-1) / ((-1)-0)((-1)-1) = x(x-1) / 2
    // L_1(x) = (x-(-1))(x-1) / ((0)-(-1))((0)-1) = (x+1)(x-1) / (-1) = -(x^2-1)
    // L_2(x) = (x-(-1))(x-0) / ((1)-(-1))((1)-0) = (x+1)x / 2

    const tau = F.fromU64(2); // Evaluate at x=2
    const evals = try LagrangePolynomial(F).evals(3, tau, allocator);
    defer allocator.free(evals);

    // L_0(2) = 2*1 / 2 = 1
    // L_1(2) = -(4-1) = -3
    // L_2(2) = 3*2 / 2 = 3

    // Sum should be 1 (partition of unity)
    var sum = F.zero();
    for (evals) |e| {
        sum = sum.add(e);
    }
    try std.testing.expect(sum.eql(F.one()));
}

test "unipoly evaluation" {
    const field = @import("../../field/mod.zig");
    const F = field.BN254Scalar;
    const allocator = std.testing.allocator;

    // p(x) = 1 + 2x + 3x^2
    const coeffs = [_]F{ F.fromU64(1), F.fromU64(2), F.fromU64(3) };
    var poly = try UniPoly(F).init(allocator, &coeffs);
    defer poly.deinit();

    // p(0) = 1
    try std.testing.expect(poly.evaluate(F.zero()).eql(F.fromU64(1)));

    // p(1) = 1 + 2 + 3 = 6
    try std.testing.expect(poly.evaluate(F.one()).eql(F.fromU64(6)));

    // p(2) = 1 + 4 + 12 = 17
    try std.testing.expect(poly.evaluate(F.fromU64(2)).eql(F.fromU64(17)));
}

test "constants match Jolt" {
    // Verify our constants match Jolt's for Stage 1
    try std.testing.expectEqual(@as(usize, 19), NUM_R1CS_CONSTRAINTS);
    try std.testing.expectEqual(@as(usize, 9), OUTER_UNIVARIATE_SKIP_DEGREE);
    try std.testing.expectEqual(@as(usize, 10), OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE);
    try std.testing.expectEqual(@as(usize, 19), OUTER_UNIVARIATE_SKIP_EXTENDED_DOMAIN_SIZE);
    try std.testing.expectEqual(@as(usize, 28), OUTER_FIRST_ROUND_POLY_NUM_COEFFS);
    try std.testing.expectEqual(@as(usize, 27), OUTER_FIRST_ROUND_POLY_DEGREE_BOUND);
    try std.testing.expectEqual(@as(usize, 9), NUM_REMAINING_R1CS_CONSTRAINTS);

    // Verify Stage 2 constants
    try std.testing.expectEqual(@as(usize, 5), NUM_PRODUCT_VIRTUAL);
    try std.testing.expectEqual(@as(usize, 4), PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DEGREE);
    try std.testing.expectEqual(@as(usize, 5), PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DOMAIN_SIZE);
    try std.testing.expectEqual(@as(usize, 9), PRODUCT_VIRTUAL_UNIVARIATE_SKIP_EXTENDED_DOMAIN_SIZE);
    try std.testing.expectEqual(@as(usize, 13), PRODUCT_VIRTUAL_FIRST_ROUND_POLY_NUM_COEFFS);
    try std.testing.expectEqual(@as(usize, 12), PRODUCT_VIRTUAL_FIRST_ROUND_POLY_DEGREE_BOUND);
}
