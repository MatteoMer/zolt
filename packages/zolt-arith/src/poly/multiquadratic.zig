//! Multiquadratic Polynomial
//!
//! A multiquadratic polynomial is a multilinear polynomial represented over
//! the grid {0, 1, ∞}^d instead of the standard boolean hypercube {0, 1}^d.
//!
//! Key insight: For a degree-1 multilinear polynomial, values on {0, 1}^d
//! uniquely determine the polynomial. We can extrapolate to ∞ along each
//! dimension using: f(∞) = f(1) - f(0) (the slope of the line).
//!
//! Grid encoding: Base-3 layout with least-significant variable (z_0) fastest-varying.
//! For dimension d, grid size is 3^d.
//! Index calculation: idx = Σ_{i=0}^{d-1} enc(x_i) * 3^i where enc: {0,1,∞} → {0,1,2}
//!
//! Reference: jolt-core/src/poly/multiquadratic_poly.rs

const std = @import("std");
const Allocator = std.mem.Allocator;

/// Encoding for ternary grid values
pub const GridValue = enum(u8) {
    Zero = 0,
    One = 1,
    Infinity = 2,
};

/// Multiquadratic polynomial over {0, 1, ∞}^num_vars
pub fn MultiquadraticPolynomial(comptime F: type) type {
    return struct {
        const Self = @This();

        /// Number of variables
        num_vars: usize,
        /// Evaluations on the ternary grid {0, 1, ∞}^num_vars
        /// Size is 3^num_vars, indexed in base-3 with z_0 fastest-varying
        evaluations: []F,
        allocator: Allocator,

        /// Create from existing ternary grid evaluations
        pub fn init(allocator: Allocator, num_vars: usize, evaluations: []const F) !Self {
            const expected_size = pow3(num_vars);
            std.debug.assert(evaluations.len == expected_size);

            const evals = try allocator.alloc(F, expected_size);
            @memcpy(evals, evaluations);

            return Self{
                .num_vars = num_vars,
                .evaluations = evals,
                .allocator = allocator,
            };
        }

        /// Create from linear (boolean hypercube) evaluations
        ///
        /// Expands {0, 1}^num_vars evaluations to {0, 1, ∞}^num_vars by
        /// computing f(∞) = f(1) - f(0) along each dimension
        pub fn fromLinear(allocator: Allocator, num_vars: usize, linear_evals: []const F) !Self {
            const linear_size: usize = @as(usize, 1) << @intCast(num_vars);
            std.debug.assert(linear_evals.len == linear_size);

            const ternary_size = pow3(num_vars);
            const result = try allocator.alloc(F, ternary_size);

            // Use temporary buffer for expansion
            const buffer = try allocator.alloc(F, ternary_size);
            defer allocator.free(buffer);

            // Copy linear evaluations to result (will be expanded in place)
            // Linear evals are at indices where all coordinates are in {0, 1}
            @memset(result, F.zero());

            // Map linear indices to ternary indices
            for (0..linear_size) |linear_idx| {
                var ternary_idx: usize = 0;
                var pow3_factor: usize = 1;
                var idx = linear_idx;

                for (0..num_vars) |_| {
                    const bit = idx & 1;
                    ternary_idx += bit * pow3_factor;
                    pow3_factor *= 3;
                    idx >>= 1;
                }

                result[ternary_idx] = linear_evals[linear_idx];
            }

            // Expand dimension by dimension
            expandLinearToTernaryGeneric(F, num_vars, result);

            return Self{
                .num_vars = num_vars,
                .evaluations = result,
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.evaluations);
        }

        /// Get evaluation at a ternary grid point
        pub fn get(self: *const Self, point: []const GridValue) F {
            std.debug.assert(point.len == self.num_vars);
            return self.evaluations[ternaryIndex(point)];
        }

        /// Set evaluation at a ternary grid point
        pub fn set(self: *Self, point: []const GridValue, value: F) void {
            std.debug.assert(point.len == self.num_vars);
            self.evaluations[ternaryIndex(point)] = value;
        }

        /// Get evaluation at index 0 (all variables = 0)
        pub fn getZero(self: *const Self) F {
            return self.evaluations[0];
        }

        /// Get evaluation at index 2 (first variable = ∞, rest = 0)
        /// This is the "infinity" term for the first variable
        pub fn getInfinity(self: *const Self) F {
            if (self.num_vars == 0) return F.zero();
            return self.evaluations[2]; // Index for (∞, 0, 0, ...)
        }

        /// Bind the first (least-significant) variable z_0 := r, reducing the
        /// dimension from w to w-1 and keeping the base-3 layout invariant.
        ///
        /// For each assignment to (z_1, ..., z_{w-1}), we have three stored values
        ///   f(0, ..), f(1, ..), f(∞, ..)
        /// and interpolate the unique quadratic in z_0 that matches them, then
        /// evaluate it at z_0 = r.
        ///
        /// Formula: f(r, rest) = f(0, rest) * (1 - r) + f(1, rest) * r + f(∞, rest) * r(r - 1)
        ///
        /// Reference: jolt-core/src/poly/multiquadratic_poly.rs bind_first_variable
        pub fn bind(self: *Self, r: F) void {
            if (self.num_vars == 0) return;

            const new_size = pow3(self.num_vars - 1);
            const one = F.one();

            // r_term = r * (r - 1)
            const r_term = r.mul(r.sub(one));

            for (0..new_size) |new_idx| {
                const old_base_idx = new_idx * 3;
                const eval_at_0 = self.evaluations[old_base_idx]; // z_0 = 0
                const eval_at_1 = self.evaluations[old_base_idx + 1]; // z_0 = 1
                const eval_at_inf = self.evaluations[old_base_idx + 2]; // z_0 = ∞

                // f(r) = f(0) * (1 - r) + f(1) * r + f(∞) * r(r - 1)
                self.evaluations[new_idx] = eval_at_0.mul(one.sub(r))
                    .add(eval_at_1.mul(r))
                    .add(eval_at_inf.mul(r_term));
            }

            self.num_vars -= 1;
            // Note: We don't reallocate/truncate the array - just reduce logical size
            // The extra elements beyond pow3(num_vars) are now unused
        }

        /// Check if the polynomial has been fully bound (no remaining variables)
        pub fn isBound(self: *const Self) bool {
            return self.num_vars == 0;
        }

        /// Get the final claim after all variables have been bound
        pub fn finalSumcheckClaim(self: *const Self) F {
            std.debug.assert(self.isBound());
            return self.evaluations[0];
        }

        /// Project to first variable, summing over remaining variables with eq weights
        ///
        /// Computes:
        ///   t'(z_0) = Σ_{x_1,...,x_{n-1}} eq_weights[x_1...x_{n-1}] * self[z_0, x_1, ..., x_{n-1}]
        ///
        /// for z_0 ∈ {0, ∞}
        ///
        /// Returns (t'(0), t'(∞))
        pub fn projectToFirstVariable(
            self: *const Self,
            eq_weights: []const F,
        ) struct { t_zero: F, t_infinity: F } {
            if (self.num_vars == 0) {
                return .{ .t_zero = self.evaluations[0], .t_infinity = F.zero() };
            }

            const remaining_vars = self.num_vars - 1;
            const remaining_size = pow3(remaining_vars);

            var t_zero = F.zero();
            var t_infinity = F.zero();

            // Sum over all remaining variable assignments in {0, 1, ∞}^{n-1}
            for (0..remaining_size) |rest_idx| {
                // Get the eq weight for this assignment
                // eq_weights is indexed over {0, 1}^{n-1}, so we need to map
                // Only sum over boolean assignments for eq weights
                const is_boolean = isBooleanTernaryIndex(rest_idx, remaining_vars);
                if (!is_boolean) continue;

                const linear_idx = ternaryToBinaryIndex(rest_idx, remaining_vars);
                if (linear_idx >= eq_weights.len) continue;

                const weight = eq_weights[linear_idx];

                // Index for (0, rest) and (∞, rest) in the ternary grid
                const idx_zero = rest_idx * 3 + 0; // z_0 = 0
                const idx_inf = rest_idx * 3 + 2; // z_0 = ∞

                t_zero = t_zero.add(weight.mul(self.evaluations[idx_zero]));
                t_infinity = t_infinity.add(weight.mul(self.evaluations[idx_inf]));
            }

            return .{ .t_zero = t_zero, .t_infinity = t_infinity };
        }
    };
}

/// Compute 3^n
fn pow3(n: usize) usize {
    var result: usize = 1;
    for (0..n) |_| {
        result *= 3;
    }
    return result;
}

/// Compute ternary index from grid point
fn ternaryIndex(point: []const GridValue) usize {
    var idx: usize = 0;
    var factor: usize = 1;
    for (point) |v| {
        idx += @intFromEnum(v) * factor;
        factor *= 3;
    }
    return idx;
}

/// Check if a ternary index corresponds to a boolean point (all coords in {0, 1})
fn isBooleanTernaryIndex(ternary_idx: usize, num_vars: usize) bool {
    var idx = ternary_idx;
    for (0..num_vars) |_| {
        if ((idx % 3) == 2) return false; // Found an ∞
        idx /= 3;
    }
    return true;
}

/// Convert ternary index to binary index (assuming all coords are in {0, 1})
fn ternaryToBinaryIndex(ternary_idx: usize, num_vars: usize) usize {
    var result: usize = 0;
    var idx = ternary_idx;
    var bit: usize = 0;
    for (0..num_vars) |_| {
        const coord = idx % 3;
        if (coord == 1) {
            result |= (@as(usize, 1) << @intCast(bit));
        }
        idx /= 3;
        bit += 1;
    }
    return result;
}

/// Expand linear evaluations to ternary grid in place
///
/// For each dimension d from 0 to num_vars-1:
///   For each slice along dimension d, compute f(∞) = f(1) - f(0)
fn expandLinearToTernaryGeneric(comptime F: type, num_vars: usize, grid: []F) void {
    if (num_vars == 0) return;

    // Process each dimension
    for (0..num_vars) |dim| {
        const stride_before = pow3(dim); // 3^dim
        const stride_after = pow3(num_vars - dim - 1); // 3^(n-dim-1)

        // For each "slice" perpendicular to dimension dim
        for (0..stride_after) |after| {
            for (0..stride_before) |before| {
                // Indices for the three values along dimension dim
                const base = after * stride_before * 3 + before;
                const idx_0 = base; // coord[dim] = 0
                const idx_1 = base + stride_before; // coord[dim] = 1
                const idx_2 = base + 2 * stride_before; // coord[dim] = ∞

                // f(∞) = f(1) - f(0)
                grid[idx_2] = grid[idx_1].sub(grid[idx_0]);
            }
        }
    }
}

/// Expand a grid of linear evaluations to multiquadratic in-place
///
/// Input: grid contains linear evaluations at boolean points,
///        padded with zeros at non-boolean indices
/// Output: grid contains full multiquadratic evaluations
pub fn expandGrid(comptime F: type, num_vars: usize, grid: []F) void {
    const buffer: []F = undefined; // Not needed for in-place version
    _ = buffer;
    if (num_vars == 0) return;

    // Process dimension by dimension
    for (0..num_vars) |dim| {
        expandDimension(F, num_vars, dim, grid);
    }
}

/// Expand a single dimension from linear to include infinity
fn expandDimension(comptime F: type, num_vars: usize, dim: usize, grid: []F) void {
    const stride = pow3(dim);
    const outer_stride = pow3(num_vars - dim - 1);

    // For each slice perpendicular to dimension dim
    for (0..outer_stride) |outer| {
        for (0..stride) |inner| {
            const base = outer * stride * 3 + inner;
            const idx_0 = base;
            const idx_1 = base + stride;
            const idx_2 = base + 2 * stride;

            // f(∞) = f(1) - f(0) (slope)
            grid[idx_2] = grid[idx_1].sub(grid[idx_0]);
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

const testing = std.testing;
const BN254Scalar = @import("../field/mod.zig").BN254Scalar;

test "MultiquadraticPolynomial: pow3" {
    try testing.expectEqual(@as(usize, 1), pow3(0));
    try testing.expectEqual(@as(usize, 3), pow3(1));
    try testing.expectEqual(@as(usize, 9), pow3(2));
    try testing.expectEqual(@as(usize, 27), pow3(3));
    try testing.expectEqual(@as(usize, 81), pow3(4));
}

test "MultiquadraticPolynomial: ternaryIndex" {
    // 1 variable
    try testing.expectEqual(@as(usize, 0), ternaryIndex(&[_]GridValue{.Zero}));
    try testing.expectEqual(@as(usize, 1), ternaryIndex(&[_]GridValue{.One}));
    try testing.expectEqual(@as(usize, 2), ternaryIndex(&[_]GridValue{.Infinity}));

    // 2 variables (z_0 fastest)
    // (0, 0) -> 0*1 + 0*3 = 0
    try testing.expectEqual(@as(usize, 0), ternaryIndex(&[_]GridValue{ .Zero, .Zero }));
    // (1, 0) -> 1*1 + 0*3 = 1
    try testing.expectEqual(@as(usize, 1), ternaryIndex(&[_]GridValue{ .One, .Zero }));
    // (∞, 0) -> 2*1 + 0*3 = 2
    try testing.expectEqual(@as(usize, 2), ternaryIndex(&[_]GridValue{ .Infinity, .Zero }));
    // (0, 1) -> 0*1 + 1*3 = 3
    try testing.expectEqual(@as(usize, 3), ternaryIndex(&[_]GridValue{ .Zero, .One }));
    // (1, 1) -> 1*1 + 1*3 = 4
    try testing.expectEqual(@as(usize, 4), ternaryIndex(&[_]GridValue{ .One, .One }));
    // (0, ∞) -> 0*1 + 2*3 = 6
    try testing.expectEqual(@as(usize, 6), ternaryIndex(&[_]GridValue{ .Zero, .Infinity }));
}

test "MultiquadraticPolynomial: fromLinear 1 variable" {
    const F = BN254Scalar;

    // Linear polynomial: f(0) = 3, f(1) = 7
    const linear = [_]F{ F.fromU64(3), F.fromU64(7) };

    var poly = try MultiquadraticPolynomial(F).fromLinear(testing.allocator, 1, &linear);
    defer poly.deinit();

    // Ternary: f(0) = 3, f(1) = 7, f(∞) = 7 - 3 = 4
    try testing.expect(poly.get(&[_]GridValue{.Zero}).eql(F.fromU64(3)));
    try testing.expect(poly.get(&[_]GridValue{.One}).eql(F.fromU64(7)));
    try testing.expect(poly.get(&[_]GridValue{.Infinity}).eql(F.fromU64(4)));
}

test "MultiquadraticPolynomial: fromLinear 2 variables" {
    const F = BN254Scalar;

    // Linear polynomial over {0,1}^2:
    // f(0,0) = 1, f(1,0) = 2, f(0,1) = 3, f(1,1) = 4
    const linear = [_]F{
        F.fromU64(1), // (0,0)
        F.fromU64(2), // (1,0)
        F.fromU64(3), // (0,1)
        F.fromU64(4), // (1,1)
    };

    var poly = try MultiquadraticPolynomial(F).fromLinear(testing.allocator, 2, &linear);
    defer poly.deinit();

    // Check boolean points are correct
    try testing.expect(poly.get(&[_]GridValue{ .Zero, .Zero }).eql(F.fromU64(1)));
    try testing.expect(poly.get(&[_]GridValue{ .One, .Zero }).eql(F.fromU64(2)));
    try testing.expect(poly.get(&[_]GridValue{ .Zero, .One }).eql(F.fromU64(3)));
    try testing.expect(poly.get(&[_]GridValue{ .One, .One }).eql(F.fromU64(4)));

    // Check infinity along first dimension
    // f(∞, 0) = f(1,0) - f(0,0) = 2 - 1 = 1
    try testing.expect(poly.get(&[_]GridValue{ .Infinity, .Zero }).eql(F.fromU64(1)));
    // f(∞, 1) = f(1,1) - f(0,1) = 4 - 3 = 1
    try testing.expect(poly.get(&[_]GridValue{ .Infinity, .One }).eql(F.fromU64(1)));

    // Check infinity along second dimension
    // f(0, ∞) = f(0,1) - f(0,0) = 3 - 1 = 2
    try testing.expect(poly.get(&[_]GridValue{ .Zero, .Infinity }).eql(F.fromU64(2)));
    // f(1, ∞) = f(1,1) - f(1,0) = 4 - 2 = 2
    try testing.expect(poly.get(&[_]GridValue{ .One, .Infinity }).eql(F.fromU64(2)));

    // f(∞, ∞) = f(1,∞) - f(0,∞) = 2 - 2 = 0
    // OR equivalently: f(∞,1) - f(∞,0) = 1 - 1 = 0
    try testing.expect(poly.get(&[_]GridValue{ .Infinity, .Infinity }).eql(F.fromU64(0)));
}

test "MultiquadraticPolynomial: isBooleanTernaryIndex" {
    // 2 variables
    try testing.expect(isBooleanTernaryIndex(0, 2)); // (0, 0)
    try testing.expect(isBooleanTernaryIndex(1, 2)); // (1, 0)
    try testing.expect(!isBooleanTernaryIndex(2, 2)); // (∞, 0)
    try testing.expect(isBooleanTernaryIndex(3, 2)); // (0, 1)
    try testing.expect(isBooleanTernaryIndex(4, 2)); // (1, 1)
    try testing.expect(!isBooleanTernaryIndex(5, 2)); // (∞, 1)
    try testing.expect(!isBooleanTernaryIndex(6, 2)); // (0, ∞)
}

test "MultiquadraticPolynomial: ternaryToBinaryIndex" {
    // 2 variables
    try testing.expectEqual(@as(usize, 0), ternaryToBinaryIndex(0, 2)); // (0, 0) -> 0b00
    try testing.expectEqual(@as(usize, 1), ternaryToBinaryIndex(1, 2)); // (1, 0) -> 0b01
    try testing.expectEqual(@as(usize, 2), ternaryToBinaryIndex(3, 2)); // (0, 1) -> 0b10
    try testing.expectEqual(@as(usize, 3), ternaryToBinaryIndex(4, 2)); // (1, 1) -> 0b11
}

test "MultiquadraticPolynomial: bind 1 variable" {
    const F = BN254Scalar;

    // Linear polynomial: f(0) = 3, f(1) = 7
    // f(x) = 3 + 4x (linear interpolation)
    // After binding at r = 2: f(2) = 3 + 4*2 = 11
    const linear = [_]F{ F.fromU64(3), F.fromU64(7) };

    var poly = try MultiquadraticPolynomial(F).fromLinear(testing.allocator, 1, &linear);
    defer poly.deinit();

    // Verify initial state
    try testing.expectEqual(@as(usize, 1), poly.num_vars);

    // Bind at r = 2
    // Formula: f(r) = f(0)*(1-r) + f(1)*r + f(∞)*r(r-1)
    // f(∞) = 7 - 3 = 4
    // f(2) = 3*(1-2) + 7*2 + 4*2*(2-1) = 3*(-1) + 14 + 4*2 = -3 + 14 + 8 = 19
    const r = F.fromU64(2);
    poly.bind(r);

    try testing.expectEqual(@as(usize, 0), poly.num_vars);
    try testing.expect(poly.isBound());

    // Expected: f(2) = 3 + 4*2 = 11 for linear, but for multiquadratic:
    // f(r) = f(0)*(1-r) + f(1)*r + f(∞)*r(r-1)
    // = 3*(-1) + 7*2 + 4*2*1 = -3 + 14 + 8 = 19
    const expected = F.fromU64(19);
    try testing.expect(poly.finalSumcheckClaim().eql(expected));
}

test "MultiquadraticPolynomial: bind 2 variables" {
    const F = BN254Scalar;

    // Linear polynomial over {0,1}^2:
    // f(0,0) = 1, f(1,0) = 2, f(0,1) = 3, f(1,1) = 4
    const linear = [_]F{
        F.fromU64(1), // (0,0)
        F.fromU64(2), // (1,0)
        F.fromU64(3), // (0,1)
        F.fromU64(4), // (1,1)
    };

    var poly = try MultiquadraticPolynomial(F).fromLinear(testing.allocator, 2, &linear);
    defer poly.deinit();

    // Verify initial state
    try testing.expectEqual(@as(usize, 2), poly.num_vars);
    try testing.expect(!poly.isBound());

    // Bind first variable (z_0) at r = 2
    // For z_1 = 0: f(0,0)=1, f(1,0)=2, f(∞,0)=1
    //   f(2,0) = 1*(1-2) + 2*2 + 1*2*(2-1) = -1 + 4 + 2 = 5
    // For z_1 = 1: f(0,1)=3, f(1,1)=4, f(∞,1)=1
    //   f(2,1) = 3*(-1) + 4*2 + 1*2*1 = -3 + 8 + 2 = 7
    // For z_1 = ∞: f(0,∞)=2, f(1,∞)=2, f(∞,∞)=0
    //   f(2,∞) = 2*(-1) + 2*2 + 0*2*1 = -2 + 4 + 0 = 2
    const r1 = F.fromU64(2);
    poly.bind(r1);

    try testing.expectEqual(@as(usize, 1), poly.num_vars);
    try testing.expect(!poly.isBound());

    // After first bind, we should have 3 values:
    // evals[0] = f(2, 0) = 5
    // evals[1] = f(2, 1) = 7
    // evals[2] = f(2, ∞) = 2
    try testing.expect(poly.evaluations[0].eql(F.fromU64(5)));
    try testing.expect(poly.evaluations[1].eql(F.fromU64(7)));
    try testing.expect(poly.evaluations[2].eql(F.fromU64(2)));

    // Bind second variable (z_1) at r = 3
    // f(2, 3) = f(2,0)*(1-3) + f(2,1)*3 + f(2,∞)*3*(3-1)
    //         = 5*(-2) + 7*3 + 2*3*2
    //         = -10 + 21 + 12 = 23
    const r2 = F.fromU64(3);
    poly.bind(r2);

    try testing.expectEqual(@as(usize, 0), poly.num_vars);
    try testing.expect(poly.isBound());
    try testing.expect(poly.finalSumcheckClaim().eql(F.fromU64(23)));
}

test "MultiquadraticPolynomial: bind with field challenge" {
    const F = BN254Scalar;

    // Linear polynomial: f(0) = 5, f(1) = 10
    // f(∞) = 5
    const linear = [_]F{ F.fromU64(5), F.fromU64(10) };

    var poly = try MultiquadraticPolynomial(F).fromLinear(testing.allocator, 1, &linear);
    defer poly.deinit();

    // Bind at r = 0 should give f(0) = 5
    const r_zero = F.zero();
    var poly_copy_zero = try MultiquadraticPolynomial(F).init(testing.allocator, 1, poly.evaluations[0..3]);
    defer poly_copy_zero.deinit();
    poly_copy_zero.bind(r_zero);
    try testing.expect(poly_copy_zero.finalSumcheckClaim().eql(F.fromU64(5)));

    // Bind at r = 1 should give f(1) = 10
    const r_one = F.one();
    var poly_copy_one = try MultiquadraticPolynomial(F).init(testing.allocator, 1, poly.evaluations[0..3]);
    defer poly_copy_one.deinit();
    poly_copy_one.bind(r_one);
    try testing.expect(poly_copy_one.finalSumcheckClaim().eql(F.fromU64(10)));
}
