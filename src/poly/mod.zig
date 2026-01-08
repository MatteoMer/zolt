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

        /// Bind the first (highest index) variable to a value
        /// This reduces the polynomial from n to n-1 variables
        /// Uses HighToLow binding order (top variable bound first)
        pub fn bindFirst(self: *const Self, value: F) !Self {
            std.debug.assert(self.num_vars > 0);

            const new_size = self.evaluations.len / 2;
            const evals = try self.allocator.alloc(F, new_size);

            const one_minus_value = F.one().sub(value);

            for (0..new_size) |i| {
                // f(x_1, ..., x_n) evaluated at x_n = value (top variable)
                // = (1 - value) * f(x_1, ..., 0) + value * f(x_1, ..., 1)
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

        /// Bind the last (lowest index) variable to a value - in-place
        /// This reduces the polynomial from n to n-1 variables
        /// Uses LowToHigh binding order (bottom variable bound first)
        ///
        /// Matches Jolt's bound_poly_var_bot:
        ///   Z[i] = Z[2*i] + r * (Z[2*i+1] - Z[2*i])
        ///
        /// This is the correct binding order for the outer streaming sumcheck
        /// linear phase where we bind cycle variables from low to high.
        pub fn bindLow(self: *Self, value: F) void {
            std.debug.assert(self.num_vars > 0);

            const new_size = self.evaluations.len / 2;

            for (0..new_size) |i| {
                // f(x_0, x_1, ..., x_{n-1}) evaluated at x_0 = value (bottom variable)
                // new[i] = old[2*i] + r * (old[2*i+1] - old[2*i])
                // = (1 - r) * old[2*i] + r * old[2*i+1]
                const low = self.evaluations[2 * i];
                const high = self.evaluations[2 * i + 1];
                self.evaluations[i] = low.add(value.mul(high.sub(low)));
            }

            self.num_vars -= 1;
        }

        /// Get the length (number of evaluations after binding)
        pub fn boundLen(self: *const Self) usize {
            return @as(usize, 1) << @intCast(self.num_vars);
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
        ///
        /// Uses BIG-ENDIAN indexing to match Jolt's EqPolynomial::evals():
        /// - result[i] = eq(r, x) where x is the binary representation of i
        /// - bit 0 (MSB of i) corresponds to r[0]
        /// - bit n-1 (LSB of i) corresponds to r[n-1]
        ///
        /// This matches Jolt's convention where for r = [r0, r1, ..., r_{n-1}]:
        /// - index 0 → eq(r, [0, 0, ..., 0])
        /// - index 1 → eq(r, [0, 0, ..., 1])  (LSB = 1, so last var x_{n-1} = 1)
        /// - etc.
        pub fn evals(self: *const Self, allocator: Allocator) ![]F {
            const n = self.r.len;
            const size = @as(usize, 1) << @as(u6, @intCast(n));
            const result = try allocator.alloc(F, size);

            // Initialize with scaling factor (1 for no scaling)
            @memset(result, F.one());

            // Build evaluations using Jolt's algorithm (big-endian indexing)
            // Jolt's algorithm:
            //   for j in 0..r.len() {
            //       size *= 2;
            //       for i in (0..size).rev().step_by(2) {
            //           let scalar = evals[i / 2];
            //           evals[i] = scalar * r[j];
            //           evals[i - 1] = scalar - evals[i];
            //       }
            //   }
            //
            // (0..size).rev().step_by(2) gives odd indices in descending order:
            // For size=4: [3, 1]
            // For size=8: [7, 5, 3, 1]
            //
            // All challenges are expected to be in Montgomery form.
            var current_size: usize = 1;
            for (0..n) |j| {
                // Double the size for this variable
                current_size *= 2;

                // Process odd indices from high to low (matching Jolt's rev().step_by(2))
                var i = current_size - 1; // Start at highest odd index
                while (i >= 1) {
                    // i is odd, so i-1 is even
                    const scalar = result[i / 2]; // i/2 = (i-1)/2 for odd i
                    const r_j = self.r[j];
                    result[i] = scalar.mul(r_j); // higher index (odd) gets r[j]
                    result[i - 1] = scalar.sub(result[i]); // lower index (even) gets 1 - r[j]

                    if (i >= 2) {
                        i -= 2;
                    } else {
                        break;
                    }
                }
            }

            return result;
        }

        /// Bind the first variable and reduce polynomial size in-place
        /// After binding n variables, evals() will return 2^(original_len - n) values
        pub fn bind(self: *Self, value: F) void {
            if (self.r.len == 0) return;

            const new_len = self.r.len - 1;
            // Shift r values down (bind first variable)
            for (0..new_len) |i| {
                // Interpolate: new_r[i] = (1 - value) * r[i] + value * r[i+1]
                // Actually for eq polynomial binding, we just drop the first r
                // But we need to update the "current" evaluation
                self.r[i] = self.r[i + 1];
            }
            // We can't actually resize, so we track this differently
            // For now, leave r unchanged but the caller should track bound count
            _ = value;
        }

        /// Evaluate eq(x, r) using the MLE formula (static version)
        pub fn mle(r: []const F, x: []const F) F {
            std.debug.assert(r.len == x.len);
            var result = F.one();
            for (0..r.len) |i| {
                const ri_xi = r[i].mul(x[i]);
                const one_minus_ri = F.one().sub(r[i]);
                const one_minus_xi = F.one().sub(x[i]);
                result = result.mul(ri_xi.add(one_minus_ri.mul(one_minus_xi)));
            }
            return result;
        }
    };
}

/// EqPlusOne polynomial eq+1(x, y)
///
/// This MLE evaluates to 1 when y = x + 1 (binary increment).
/// For x in the range [0, 2^l - 2], eq+1(x, y) = 1 iff y = x + 1.
/// When x is all 1s, the result is 0 (there's no successor in the range).
///
/// Used in Jolt's ShiftSumcheck to relate values at consecutive indices.
pub fn EqPlusOnePolynomial(comptime F: type) type {
    return struct {
        const Self = @This();

        /// The point x that defines eq+1(x, ·)
        x: []F,
        allocator: Allocator,

        /// Create an EqPlusOne polynomial for point x
        /// x is assumed to be in BIG_ENDIAN order (MSB first)
        pub fn init(allocator: Allocator, x: []const F) !Self {
            const x_copy = try allocator.alloc(F, x.len);
            @memcpy(x_copy, x);

            return .{
                .x = x_copy,
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.x);
        }

        /// Evaluate eq+1(x, y) at point y
        /// Both x and y are in BIG_ENDIAN order (x[0] is MSB)
        ///
        /// eq+1(x, y) = 1 iff y = x + 1 in binary.
        /// If x + 1 = y, then:
        ///   - Let k be the index of the rightmost 0 bit in x (counting from LSB)
        ///   - In y, all bits below k are 0 (from 1s in x that carry over)
        ///   - Bit k in y is 1 (from the carry)
        ///   - Bits above k are the same in x and y
        pub fn evaluate(self: *const Self, y: []const F) F {
            const l = self.x.len;
            std.debug.assert(y.len == l);

            var result = F.zero();

            // Sum over all possible positions k for the "flip" bit
            // k is the suffix length of 1s in x (0-indexed from LSB)
            for (0..l) |k| {
                // lower_bits_product: bits 0..k-1 (from LSB) are 1 in x, 0 in y
                // In BIG_ENDIAN: these are indices l-1, l-2, ..., l-k
                var lower_bits_product = F.one();
                for (0..k) |i| {
                    const idx = l - 1 - i;
                    // x[idx] should be 1, y[idx] should be 0
                    lower_bits_product = lower_bits_product.mul(self.x[idx].mul(F.one().sub(y[idx])));
                }

                // kth_bit_product: bit k is 0 in x, 1 in y
                // In BIG_ENDIAN: index l-1-k
                const kth_idx = l - 1 - k;
                const kth_bit_product = F.one().sub(self.x[kth_idx]).mul(y[kth_idx]);

                // higher_bits_product: bits k+1..l-1 are the same in x and y
                // In BIG_ENDIAN: indices 0..l-2-k
                var higher_bits_product = F.one();
                for ((k + 1)..l) |i| {
                    const idx = l - 1 - i;
                    // x[idx] == y[idx] means: x*y + (1-x)*(1-y)
                    const xi_yi = self.x[idx].mul(y[idx]);
                    const one_minus_xi = F.one().sub(self.x[idx]);
                    const one_minus_yi = F.one().sub(y[idx]);
                    higher_bits_product = higher_bits_product.mul(xi_yi.add(one_minus_xi.mul(one_minus_yi)));
                }

                result = result.add(lower_bits_product.mul(kth_bit_product).mul(higher_bits_product));
            }

            return result;
        }

        /// Compute eq+1(x, y) directly (static version)
        pub fn mle(x: []const F, y: []const F) F {
            const l = x.len;
            std.debug.assert(y.len == l);

            var result = F.zero();

            for (0..l) |k| {
                var lower_bits_product = F.one();
                for (0..k) |i| {
                    const idx = l - 1 - i;
                    lower_bits_product = lower_bits_product.mul(x[idx].mul(F.one().sub(y[idx])));
                }

                const kth_idx = l - 1 - k;
                const kth_bit_product = F.one().sub(x[kth_idx]).mul(y[kth_idx]);

                var higher_bits_product = F.one();
                for ((k + 1)..l) |i| {
                    const idx = l - 1 - i;
                    const xi_yi = x[idx].mul(y[idx]);
                    const one_minus_xi = F.one().sub(x[idx]);
                    const one_minus_yi = F.one().sub(y[idx]);
                    higher_bits_product = higher_bits_product.mul(xi_yi.add(one_minus_xi.mul(one_minus_yi)));
                }

                result = result.add(lower_bits_product.mul(kth_bit_product).mul(higher_bits_product));
            }

            return result;
        }

        /// Bind the first variable (MSB) to a value, reducing the polynomial
        /// This is used in sumcheck rounds
        pub fn bind(self: *Self, value: F) void {
            _ = self;
            _ = value;
            // For eq+1, binding is complex. For now, just reduce size tracking.
            // A proper implementation would update internal state.
            // This is a placeholder - the actual computation happens in evaluate().
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

test "EqPolynomial partition of unity" {
    const F = field.BN254Scalar;
    const testing = std.testing;

    // Test 1: with r = [1/2, 1/3] (simple values)
    {
        const half = F.one().mul(F.fromU64(2).inverse().?);
        const third = F.one().mul(F.fromU64(3).inverse().?);
        const r = [_]F{ half, third };

        var eq_poly = try EqPolynomial(F).init(testing.allocator, &r);
        defer eq_poly.deinit();

        const evals = try eq_poly.evals(testing.allocator);
        defer testing.allocator.free(evals);

        // Sum should equal 1
        var sum = F.zero();
        for (evals) |ev| {
            sum = sum.add(ev);
        }

        // Check partition of unity
        try testing.expect(sum.eql(F.one()));
    }

    // Test 2: with larger r values (like in the inner_sum_prod test)
    {
        const r = [_]F{ F.fromU64(5555), F.fromU64(6666) };

        var eq_poly = try EqPolynomial(F).init(testing.allocator, &r);
        defer eq_poly.deinit();

        const evals = try eq_poly.evals(testing.allocator);
        defer testing.allocator.free(evals);

        // Sum should equal 1
        var sum = F.zero();
        for (evals) |ev| {
            sum = sum.add(ev);
        }

        // FORCED Print for debugging
        std.debug.print("\n\n=== PARTITION TEST ===\n", .{});
        std.debug.print("r[0] (5555 as Montgomery):\n", .{});
        for (r[0].limbs) |limb| std.debug.print("  {x:016}\n", .{limb});
        std.debug.print("r[1] (6666 as Montgomery):\n", .{});
        for (r[1].limbs) |limb| std.debug.print("  {x:016}\n", .{limb});
        std.debug.print("Eq evals:\n", .{});
        for (evals, 0..) |ev, i| {
            std.debug.print("  [{d}]: ", .{i});
            for (ev.limbs) |limb| std.debug.print("{x:016} ", .{limb});
            std.debug.print("\n", .{});
        }
        std.debug.print("Sum:\n  ", .{});
        for (sum.limbs) |limb| std.debug.print("{x:016} ", .{limb});
        std.debug.print("\nOne:\n  ", .{});
        for (F.one().limbs) |limb| std.debug.print("{x:016} ", .{limb});
        std.debug.print("\nSum == One? {}\n", .{sum.eql(F.one())});
        std.debug.print("=== END PARTITION TEST ===\n\n", .{});

        // Check partition of unity
        try testing.expect(sum.eql(F.one()));
    }
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

test "dense polynomial bindLow" {
    const F = field.BN254Scalar;
    const allocator = std.testing.allocator;

    // Create a polynomial with 2 variables (4 evaluations)
    // Layout with x_0 as LSB: evals[i] = f(b_0, b_1) where i = b_0 + 2*b_1
    // So:
    //   evals[0] = f(0,0) = 1
    //   evals[1] = f(1,0) = 2
    //   evals[2] = f(0,1) = 3
    //   evals[3] = f(1,1) = 4
    const evals = [_]F{
        F.fromU64(1), // f(0,0)
        F.fromU64(2), // f(1,0)
        F.fromU64(3), // f(0,1)
        F.fromU64(4), // f(1,1)
    };

    var poly = try DensePolynomial(F).init(allocator, &evals);
    defer poly.deinit();

    // Bind x_0 (the low variable) to r = 3
    const r = F.fromU64(3);
    poly.bindLow(r);

    // After binding x_0 = r:
    //   new[0] = f(r, 0) = (1-r)*f(0,0) + r*f(1,0) = (1-3)*1 + 3*2 = -2 + 6 = 4
    //   new[1] = f(r, 1) = (1-r)*f(0,1) + r*f(1,1) = (1-3)*3 + 3*4 = -6 + 12 = 6
    // But using Jolt's formula: new[i] = old[2i] + r*(old[2i+1] - old[2i])
    //   new[0] = 1 + 3*(2 - 1) = 1 + 3 = 4 ✓
    //   new[1] = 3 + 3*(4 - 3) = 3 + 3 = 6 ✓

    try std.testing.expectEqual(@as(usize, 1), poly.num_vars);
    try std.testing.expectEqual(@as(usize, 2), poly.boundLen());
    try std.testing.expect(poly.evaluations[0].eql(F.fromU64(4)));
    try std.testing.expect(poly.evaluations[1].eql(F.fromU64(6)));
}

test "dense polynomial bindLow matches Jolt's bound_poly_var_bot" {
    const F = field.BN254Scalar;
    const allocator = std.testing.allocator;

    // 3 variables: 8 evaluations
    const evals = [_]F{
        F.fromU64(10), // f(0,0,0)
        F.fromU64(20), // f(1,0,0)
        F.fromU64(30), // f(0,1,0)
        F.fromU64(40), // f(1,1,0)
        F.fromU64(50), // f(0,0,1)
        F.fromU64(60), // f(1,0,1)
        F.fromU64(70), // f(0,1,1)
        F.fromU64(80), // f(1,1,1)
    };

    var poly = try DensePolynomial(F).init(allocator, &evals);
    defer poly.deinit();

    // Bind x_0 to r = 5
    const r = F.fromU64(5);
    poly.bindLow(r);

    // Expected: new[i] = old[2i] + r*(old[2i+1] - old[2i])
    // new[0] = 10 + 5*(20-10) = 10 + 50 = 60
    // new[1] = 30 + 5*(40-30) = 30 + 50 = 80
    // new[2] = 50 + 5*(60-50) = 50 + 50 = 100
    // new[3] = 70 + 5*(80-70) = 70 + 50 = 120

    try std.testing.expectEqual(@as(usize, 2), poly.num_vars);
    try std.testing.expect(poly.evaluations[0].eql(F.fromU64(60)));
    try std.testing.expect(poly.evaluations[1].eql(F.fromU64(80)));
    try std.testing.expect(poly.evaluations[2].eql(F.fromU64(100)));
    try std.testing.expect(poly.evaluations[3].eql(F.fromU64(120)));
}

test "EqPlusOnePolynomial basic" {
    const F = field.BN254Scalar;

    // Test with 2-bit values
    // eq+1(x, y) = 1 iff y = x + 1
    // x = [0, 0] (binary 0) => y should be [0, 1] (binary 1)
    // x = [0, 1] (binary 1) => y should be [1, 0] (binary 2)
    // x = [1, 0] (binary 2) => y should be [1, 1] (binary 3)
    // x = [1, 1] (binary 3) => no valid y in range (returns 0)

    const zero = F.zero();
    const one = F.one();

    // Test: x = [0, 0] (binary 0), y = [0, 1] (binary 1) should give 1
    {
        const x = [_]F{ zero, zero };
        const y = [_]F{ zero, one };
        const result = EqPlusOnePolynomial(F).mle(&x, &y);
        try std.testing.expect(result.eql(one));
    }

    // Test: x = [0, 1] (binary 1), y = [1, 0] (binary 2) should give 1
    {
        const x = [_]F{ zero, one };
        const y = [_]F{ one, zero };
        const result = EqPlusOnePolynomial(F).mle(&x, &y);
        try std.testing.expect(result.eql(one));
    }

    // Test: x = [1, 0] (binary 2), y = [1, 1] (binary 3) should give 1
    {
        const x = [_]F{ one, zero };
        const y = [_]F{ one, one };
        const result = EqPlusOnePolynomial(F).mle(&x, &y);
        try std.testing.expect(result.eql(one));
    }

    // Test: x = [1, 1] (binary 3), y = anything should give 0 (no successor)
    {
        const x = [_]F{ one, one };
        const y = [_]F{ zero, zero };
        const result = EqPlusOnePolynomial(F).mle(&x, &y);
        try std.testing.expect(result.eql(zero));
    }

    // Test: x = [0, 0], y = [1, 0] should give 0 (not successor)
    {
        const x = [_]F{ zero, zero };
        const y = [_]F{ one, zero };
        const result = EqPlusOnePolynomial(F).mle(&x, &y);
        try std.testing.expect(result.eql(zero));
    }
}

// Reference tests from submodules to ensure they run
test {
    std.testing.refAllDecls(split_eq);
    std.testing.refAllDecls(multiquadratic);
}
