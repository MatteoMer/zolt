const std = @import("std");
const Allocator = std.mem.Allocator;

/// Gruen Split Equality Polynomial
///
/// Implements the Dao-Thaler + Gruen optimization for efficient sumcheck.
/// Factors eq(w, x) into three parts:
///   eq(w, x) = E_out(x_out) * E_in(x_in) * eq_linear(x_last)
///
/// This allows computing round polynomials efficiently without full polynomial expansion.
///
/// Variable Layout (LowToHigh binding):
///   w = [w_0, w_1, ..., w_{m-1}, w_m, ..., w_{n-2}, w_{n-1}]
///        |------w_out-------|    |-----w_in-----|   w_last
///              m vars              (n-1-m) vars     1 var
///
/// where m = n / 2 and n = w.len()
pub fn GruenSplitEqPolynomial(comptime F: type) type {
    return struct {
        const Self = @This();

        allocator: Allocator,

        /// Number of unbound variables remaining (decrements each round)
        current_index: usize,

        /// Accumulated eq(w_bound, r_bound) from already-bound variables
        current_scalar: F,

        /// The full challenge vector w (big-endian: w[0] = MSB)
        w: []const F,

        /// Prefix eq tables for w_in. E_in_vec[k] = eq(w_in[..k], .) over {0,1}^k.
        /// Invariant: always non-empty; E_in_vec[0] = [1].
        E_in_vec: [][]F,

        /// Prefix eq tables for w_out. E_out_vec[k] = eq(w_out[..k], .) over {0,1}^k.
        /// Invariant: always non-empty; E_out_vec[0] = [1].
        E_out_vec: [][]F,

        /// Track how many tables we've popped from E_in
        E_in_active: usize,
        /// Track how many tables we've popped from E_out
        E_out_active: usize,

        pub fn init(allocator: Allocator, w: []const F) !Self {
            return initWithScaling(allocator, w, F.one());
        }

        pub fn initWithScaling(allocator: Allocator, w: []const F, scaling_factor: F) !Self {
            const n = w.len;
            if (n == 0) return error.EmptyChallenge;

            const m = n / 2;

            // w = [w_out (m vars), w_in ((n-1-m) vars), w_last (1 var)]
            // w_out = w[0..m]
            // w_in = w[m..n-1]
            // w_last = w[n-1]
            const w_out = w[0..m];
            const w_in = if (n > m + 1) w[m .. n - 1] else &[_]F{};

            // Build prefix eq tables
            const E_out_vec = try evalsCached(F, allocator, w_out);
            const E_in_vec = try evalsCached(F, allocator, w_in);

            // Copy w to owned slice
            const w_copy = try allocator.alloc(F, n);
            @memcpy(w_copy, w);

            return Self{
                .allocator = allocator,
                .current_index = n,
                .current_scalar = scaling_factor,
                .w = w_copy,
                .E_in_vec = E_in_vec,
                .E_out_vec = E_out_vec,
                .E_in_active = E_in_vec.len,
                .E_out_active = E_out_vec.len,
            };
        }

        pub fn deinit(self: *Self) void {
            // Free all E_out tables
            for (self.E_out_vec) |table| {
                self.allocator.free(table);
            }
            self.allocator.free(self.E_out_vec);

            // Free all E_in tables
            for (self.E_in_vec) |table| {
                self.allocator.free(table);
            }
            self.allocator.free(self.E_in_vec);

            // Free w copy
            self.allocator.free(@constCast(self.w));
        }

        /// Return the current E_in table (last active table)
        pub fn E_in_current(self: *const Self) []const F {
            if (self.E_in_active == 0) return &[_]F{F.one()};
            return self.E_in_vec[self.E_in_active - 1];
        }

        /// Return the current E_out table (last active table)
        pub fn E_out_current(self: *const Self) []const F {
            if (self.E_out_active == 0) return &[_]F{F.one()};
            return self.E_out_vec[self.E_out_active - 1];
        }

        /// Get the current w value being bound
        pub fn get_current_w(self: *const Self) F {
            return self.w[self.current_index - 1];
        }

        /// Bind a variable with the given challenge r
        /// Updates current_scalar with eq(w[current_index-1], r)
        pub fn bind(self: *Self, r: F) void {
            // eq(w, r) = (1 - w) * (1 - r) + w * r = 1 - w - r + 2*w*r
            const w_i = self.w[self.current_index - 1];
            const prod_w_r = w_i.mul(r);
            const eq_val = F.one().sub(w_i).sub(r).add(prod_w_r).add(prod_w_r);
            self.current_scalar = self.current_scalar.mul(eq_val);

            // Decrement current_index
            self.current_index -= 1;

            // Pop from E_in or E_out based on which region we're in
            const n = self.w.len;
            const m = n / 2;

            // E_in covers variables m to n-2 (indices m..n-1 exclusive of last)
            // E_out covers variables 0 to m-1 (indices 0..m)
            // After binding, current_index tells us how many vars remain
            // E_in is active for current_index > m, E_out is active for current_index <= m

            if (m < self.current_index and self.E_in_active > 1) {
                self.E_in_active -= 1;
            } else if (0 < self.current_index and self.E_out_active > 1) {
                self.E_out_active -= 1;
            }
        }

        /// Compute cubic polynomial s(X) = l(X) * q(X) from quadratic coefficients
        ///
        /// Arguments:
        /// - q_constant: q(0), the constant term of the quadratic q(X)
        /// - q_quadratic_coeff: coefficient of X^2 in q(X)
        /// - s_0_plus_s_1: the previous round claim (s(0) + s(1))
        ///
        /// Returns: UniPoly coefficients [c0, c1, c2, c3] such that
        ///          s(X) = c0 + c1*X + c2*X^2 + c3*X^3
        pub fn gruenPolyDeg3(self: *const Self, q_constant: F, q_quadratic_coeff: F, s_0_plus_s_1: F) [4]F {
            // Linear eq polynomial: l(X) = eq_eval_0 + (eq_eval_1 - eq_eval_0) * X
            // where eq_eval_1 = current_scalar * w[current_index-1]
            //       eq_eval_0 = current_scalar - eq_eval_1
            const eq_eval_1 = self.current_scalar.mul(self.w[self.current_index - 1]);
            const eq_eval_0 = self.current_scalar.sub(eq_eval_1);
            const eq_m = eq_eval_1.sub(eq_eval_0); // slope of l(X)
            const eq_eval_2 = eq_eval_1.add(eq_m);
            const eq_eval_3 = eq_eval_2.add(eq_m);

            // Quadratic polynomial q(X) = c + d*X + e*X^2
            // We have: c = q_constant, e = q_quadratic_coeff
            // We need to find d using the constraint s(0) + s(1) = s_0_plus_s_1

            const quadratic_eval_0 = q_constant;
            const cubic_eval_0 = eq_eval_0.mul(quadratic_eval_0);
            const cubic_eval_1 = s_0_plus_s_1.sub(cubic_eval_0);

            // q(1) = c + d + e => cubic_eval_1 = eq_eval_1 * q(1)
            // => q(1) = cubic_eval_1 / eq_eval_1
            const quadratic_eval_1 = cubic_eval_1.mul(eq_eval_1.inverse() orelse F.one());

            // q(2) = c + 2d + 4e = q(1) + q(1) - q(0) + 2e
            const e_times_2 = q_quadratic_coeff.add(q_quadratic_coeff);
            const quadratic_eval_2 = quadratic_eval_1.add(quadratic_eval_1).sub(quadratic_eval_0).add(e_times_2);

            // q(3) = c + 3d + 9e = q(2) + q(1) - q(0) + 4e
            const quadratic_eval_3 = quadratic_eval_2.add(quadratic_eval_1).sub(quadratic_eval_0).add(e_times_2).add(e_times_2);

            // Cubic evaluations
            const cubic_eval_2 = eq_eval_2.mul(quadratic_eval_2);
            const cubic_eval_3 = eq_eval_3.mul(quadratic_eval_3);

            // Convert evaluations to coefficients using Lagrange interpolation
            return fromEvals(.{ cubic_eval_0, cubic_eval_1, cubic_eval_2, cubic_eval_3 });
        }

        /// Convert [p(0), p(1), p(2), p(3)] evaluations to coefficients [c0, c1, c2, c3]
        fn fromEvals(evals: [4]F) [4]F {
            const c0 = evals[0];

            const six = F.fromU64(6);
            const six_inv = six.inverse() orelse F.one();
            const two = F.fromU64(2);
            const two_inv = two.inverse() orelse F.one();

            // c3 = (-e0 + 3*e1 - 3*e2 + e3) / 6
            const c3 = evals[0].neg()
                .add(evals[1].mul(F.fromU64(3)))
                .sub(evals[2].mul(F.fromU64(3)))
                .add(evals[3])
                .mul(six_inv);

            // c2 = (2*e0 - 5*e1 + 4*e2 - e3) / 2
            const c2 = evals[0].mul(two)
                .sub(evals[1].mul(F.fromU64(5)))
                .add(evals[2].mul(F.fromU64(4)))
                .sub(evals[3])
                .mul(two_inv);

            // c1 = e1 - e0 - c2 - c3
            const c1 = evals[1].sub(evals[0]).sub(c2).sub(c3);

            return .{ c0, c1, c2, c3 };
        }
    };
}

/// Build prefix eq tables: evals_cached[k] = eq(w[..k], .) over {0,1}^k
/// Returns (num_vars + 1) tables where index k corresponds to eq over k variables.
/// evals_cached[0] = [1] (eq over 0 vars)
fn evalsCached(comptime F: type, allocator: Allocator, w: []const F) ![][]F {
    const num_vars = w.len;
    const num_tables = num_vars + 1;

    var tables = try allocator.alloc([]F, num_tables);

    // Table 0: eq over 0 vars = [1]
    tables[0] = try allocator.alloc(F, 1);
    tables[0][0] = F.one();

    // Build each prefix table incrementally
    for (0..num_vars) |k| {
        const prev = tables[k];
        const prev_len = prev.len;
        const new_len = prev_len * 2;

        tables[k + 1] = try allocator.alloc(F, new_len);
        const curr = tables[k + 1];

        // Extend: for each entry in prev, create two entries in curr
        // Using Jolt's evals_serial pattern:
        // for i in (0..size).rev().step_by(2):
        //     scalar = evals[i / 2]
        //     evals[i] = scalar * w[j]
        //     evals[i - 1] = scalar - evals[i]
        var i: usize = new_len - 1;
        while (i > 0) : (i -= 2) {
            const scalar = prev[i / 2];
            curr[i] = scalar.mul(w[k]);
            curr[i - 1] = scalar.sub(curr[i]);
            if (i < 2) break;
        }
    }

    return tables;
}

/// Compute group index from x_out and x_in
pub fn groupIndex(x_out: usize, x_in: usize, num_x_in_bits: usize) usize {
    return (x_out << num_x_in_bits) | x_in;
}

test "gruen eq polynomial initialization" {
    const BN254Scalar = @import("../../field/mod.zig").BN254Scalar;
    const allocator = std.testing.allocator;

    // Test with 4 variables
    const w = [_]BN254Scalar{
        BN254Scalar.fromU64(1),
        BN254Scalar.fromU64(2),
        BN254Scalar.fromU64(3),
        BN254Scalar.fromU64(4),
    };

    var gruen = try GruenSplitEqPolynomial(BN254Scalar).init(allocator, &w);
    defer gruen.deinit();

    // Check initial state
    try std.testing.expectEqual(@as(usize, 4), gruen.current_index);
    try std.testing.expect(gruen.current_scalar.eql(BN254Scalar.one()));
}

test "gruen eq polynomial bind" {
    const BN254Scalar = @import("../../field/mod.zig").BN254Scalar;
    const allocator = std.testing.allocator;

    const w = [_]BN254Scalar{
        BN254Scalar.fromU64(2),
        BN254Scalar.fromU64(3),
    };

    var gruen = try GruenSplitEqPolynomial(BN254Scalar).init(allocator, &w);
    defer gruen.deinit();

    // Bind with r=5
    const r = BN254Scalar.fromU64(5);
    gruen.bind(r);

    // After binding, current_index should be 1
    try std.testing.expectEqual(@as(usize, 1), gruen.current_index);

    // current_scalar should be eq(w[1], r) = eq(3, 5) = 1 - 3 - 5 + 2*3*5 = 1 - 8 + 30 = 23
    // In field arithmetic (mod p)
    const expected = BN254Scalar.one()
        .sub(BN254Scalar.fromU64(3))
        .sub(BN254Scalar.fromU64(5))
        .add(BN254Scalar.fromU64(30));
    try std.testing.expect(gruen.current_scalar.eql(expected));
}

test "gruen poly deg 3" {
    const BN254Scalar = @import("../../field/mod.zig").BN254Scalar;
    const allocator = std.testing.allocator;

    const w = [_]BN254Scalar{
        BN254Scalar.fromU64(1),
        BN254Scalar.fromU64(2),
    };

    var gruen = try GruenSplitEqPolynomial(BN254Scalar).init(allocator, &w);
    defer gruen.deinit();

    // Test gruen_poly_deg_3 with simple values
    const q_constant = BN254Scalar.fromU64(10);
    const q_quadratic = BN254Scalar.fromU64(5);
    const claim = BN254Scalar.fromU64(100);

    const coeffs = gruen.gruenPolyDeg3(q_constant, q_quadratic, claim);

    // Verify p(0) + p(1) = claim
    // p(X) = c0 + c1*X + c2*X^2 + c3*X^3
    // p(0) = c0
    // p(1) = c0 + c1 + c2 + c3
    const p0 = coeffs[0];
    const p1 = coeffs[0].add(coeffs[1]).add(coeffs[2]).add(coeffs[3]);
    const sum = p0.add(p1);

    try std.testing.expect(sum.eql(claim));
}
