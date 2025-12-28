//! Gruen Split Eq Polynomial
//!
//! Implements efficient eq polynomial evaluation for streaming sumcheck.
//! The key optimization is pre-computing prefix eq tables that allow
//! factored evaluation: eq(τ, x) = eq(τ_out, x_out) * eq(τ_in, x_in)
//!
//! Reference: jolt-core/src/poly/split_eq_poly.rs

const std = @import("std");
const Allocator = std.mem.Allocator;

/// Gruen's optimized split eq polynomial for streaming sumcheck
///
/// Maintains:
/// - Prefix eq tables E_out_vec[k] = eq(τ_out[0..k], ·) over {0,1}^k
/// - Prefix eq tables E_in_vec[k] = eq(τ_in[0..k], ·) over {0,1}^k
/// - A scalar accumulating bound variables: eq(τ_bound, r_bound)
///
/// This allows efficient window-based evaluation without recomputing
/// the full eq polynomial each round.
pub fn GruenSplitEqPolynomial(comptime F: type) type {
    return struct {
        const Self = @This();

        /// Index of next variable to bind (decreases each round)
        current_index: usize,
        /// Accumulated scalar from bound variables: prod eq(τ_i, r_i)
        current_scalar: F,
        /// Full challenge vector τ
        tau: []F,
        /// Prefix eq tables for outer variables (cycle index)
        /// E_out_vec[k] has 2^k entries for eq(τ_out[0..k], ·)
        E_out_vec: std.ArrayListUnmanaged([]F),
        /// Prefix eq tables for inner variables (constraint group)
        /// E_in_vec[k] has 2^k entries for eq(τ_in[0..k], ·)
        E_in_vec: std.ArrayListUnmanaged([]F),
        /// Number of outer variables (cycle bits)
        num_x_out: usize,
        /// Number of inner variables (group bits + constraint bits)
        num_x_in: usize,
        /// Allocator
        allocator: Allocator,

        /// Initialize with challenge vector τ
        ///
        /// τ is split into:
        /// - τ_in: first num_x_in variables (constraint group)
        /// - τ_out: remaining variables (cycle index)
        pub fn init(allocator: Allocator, tau: []const F, num_x_in: usize) !Self {
            return initWithScaling(allocator, tau, num_x_in, null);
        }

        /// Initialize with challenge vector τ and an optional initial scaling factor
        ///
        /// Following Jolt's LowToHigh binding order:
        /// - Skip the last element of τ (w_last)
        /// - Split the rest into two halves: w_out (first half) and w_in (second half)
        /// - E_out_vec contains eq tables for w_out
        /// - E_in_vec contains eq tables for w_in
        ///
        /// The scaling_factor (e.g., Lagrange kernel from UniSkip) becomes the
        /// initial current_scalar and is multiplied into all eq evaluations.
        pub fn initWithScaling(allocator: Allocator, tau: []const F, num_x_in: usize, scaling_factor: ?F) !Self {
            // Match Jolt's LowToHigh structure:
            // tau = [w_out, w_in, w_last]
            // w_out = first half of tau[0..tau.len-1]
            // w_in = second half of tau[0..tau.len-1]
            _ = num_x_in; // Ignored - we use Jolt's split

            if (tau.len == 0) {
                return Self{
                    .current_index = 0,
                    .current_scalar = scaling_factor orelse F.one(),
                    .tau = &[_]F{},
                    .E_out_vec = .{},
                    .E_in_vec = .{},
                    .num_x_out = 0,
                    .num_x_in = 0,
                    .allocator = allocator,
                };
            }

            // Split like Jolt: m = len/2, skip last, split rest
            const m = tau.len / 2;
            const num_x_out = m; // First half
            const num_x_in_actual = if (tau.len > 1) tau.len - 1 - m else 0; // Second half

            // Copy tau
            const tau_copy = try allocator.alloc(F, tau.len);
            @memcpy(tau_copy, tau);

            // Build prefix eq tables
            var E_in_vec: std.ArrayListUnmanaged([]F) = .{};
            var E_out_vec: std.ArrayListUnmanaged([]F) = .{};

            // Build outer prefix tables (E_out_vec) for tau[0..m]
            // E_out_vec[0] = [1] (empty eq is always 1)
            try E_out_vec.append(allocator, try allocator.alloc(F, 1));
            E_out_vec.items[0][0] = F.one();

            for (0..num_x_out) |k| {
                const prev_size: usize = @as(usize, 1) << @intCast(k);
                const new_size: usize = prev_size * 2;
                const prev = E_out_vec.items[k];
                const next = try allocator.alloc(F, new_size);

                const tau_k = tau_copy[k]; // w_out uses tau[0..m]
                const one_minus_tau_k = F.one().sub(tau_k);

                // eq(τ[0..k+1], (x, 0)) = eq(τ[0..k], x) * (1 - τ[k])
                // eq(τ[0..k+1], (x, 1)) = eq(τ[0..k], x) * τ[k]
                for (0..prev_size) |i| {
                    next[i] = prev[i].mul(one_minus_tau_k);
                    next[i + prev_size] = prev[i].mul(tau_k);
                }

                try E_out_vec.append(allocator, next);
            }

            // Build inner prefix tables (E_in_vec) for tau[m..tau.len-1]
            // E_in_vec[0] = [1]
            try E_in_vec.append(allocator, try allocator.alloc(F, 1));
            E_in_vec.items[0][0] = F.one();

            for (0..num_x_in_actual) |k| {
                const prev_size: usize = @as(usize, 1) << @intCast(k);
                const new_size: usize = prev_size * 2;
                const prev = E_in_vec.items[k];
                const next = try allocator.alloc(F, new_size);

                const tau_k = tau_copy[m + k]; // w_in uses tau[m..tau.len-1]
                const one_minus_tau_k = F.one().sub(tau_k);

                for (0..prev_size) |i| {
                    next[i] = prev[i].mul(one_minus_tau_k);
                    next[i + prev_size] = prev[i].mul(tau_k);
                }

                try E_in_vec.append(allocator, next);
            }

            return Self{
                .current_index = tau.len,
                .current_scalar = scaling_factor orelse F.one(),
                .tau = tau_copy,
                .E_out_vec = E_out_vec,
                .E_in_vec = E_in_vec,
                .num_x_out = num_x_out,
                .num_x_in = num_x_in_actual,
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.tau);

            for (self.E_in_vec.items) |table| {
                self.allocator.free(table);
            }
            self.E_in_vec.deinit(self.allocator);

            for (self.E_out_vec.items) |table| {
                self.allocator.free(table);
            }
            self.E_out_vec.deinit(self.allocator);
        }

        /// Bind the current variable to challenge r
        ///
        /// Updates current_scalar with eq(τ[current_index-1], r)
        /// and decrements current_index
        pub fn bind(self: *Self, r: F) void {
            if (self.current_index == 0) return;

            self.current_index -= 1;
            const tau_i = self.tau[self.current_index];

            // eq(τ_i, r) = τ_i * r + (1 - τ_i) * (1 - r)
            //            = τ_i * r + 1 - τ_i - r + τ_i * r
            //            = 2 * τ_i * r - τ_i - r + 1
            const eq_val = tau_i.mul(r).add(F.one().sub(tau_i).mul(F.one().sub(r)));
            self.current_scalar = self.current_scalar.mul(eq_val);
        }

        /// Get the full eq table for the remaining unbound variables
        ///
        /// Returns eq(τ_unbound, ·) scaled by current_scalar
        pub fn getFullEqTable(self: *const Self, allocator: Allocator) ![]F {
            const size = @as(usize, 1) << @intCast(self.current_index);
            const result = try allocator.alloc(F, size);

            // Start with the product of all (1 - τ_i) for unbound variables
            var base = self.current_scalar;
            for (0..self.current_index) |i| {
                base = base.mul(F.one().sub(self.tau[i]));
            }
            result[0] = base;

            // Build up using the factor ratios
            for (0..self.current_index) |i| {
                const half = @as(usize, 1) << @intCast(i);
                const tau_i = self.tau[i];
                const one_minus_tau_i = F.one().sub(tau_i);

                // factor = τ_i / (1 - τ_i)
                const factor = if (one_minus_tau_i.eql(F.zero()))
                    F.zero()
                else
                    tau_i.mul(one_minus_tau_i.inverse().?);

                for (0..half) |j| {
                    result[j + half] = result[j].mul(factor);
                }
            }

            return result;
        }

        /// Get the high bit challenge τ_high
        ///
        /// In Jolt's univariate skip, τ_high is the last element of the tau vector,
        /// which corresponds to the constraint group selector bit.
        pub fn getTauHigh(self: *const Self) F {
            if (self.tau.len == 0) return F.zero();
            return self.tau[self.tau.len - 1];
        }

        /// Get eq tables for a window of variables
        ///
        /// This matches Jolt's E_out_in_for_window function:
        /// - window_size: number of variables being processed (typically 1 for sumcheck)
        /// - Returns (E_out, E_in) for the factorized eq evaluation
        ///
        /// The factorization computes eq weights as:
        ///   eq[i] = E_out[i >> head_in_bits] * E_in[i & ((1 << head_in_bits) - 1)]
        ///
        /// For the streaming round with 1024 cycles (10 vars) and window_size=1:
        /// - num_unbound = current_index = 11 (tau_low length)
        /// - head_len = 11 - 1 = 10
        /// - m = 11 / 2 = 5
        /// - head_out_bits = min(10, 5) = 5  → E_out has 32 entries
        /// - head_in_bits = 10 - 5 = 5       → E_in has 32 entries
        pub fn getWindowEqTables(
            self: *const Self,
            _: usize, // num_unbound_vars - ignored, use current_index like Jolt
            window_size: usize,
        ) struct { E_out: []const F, E_in: []const F, head_in_bits: usize } {
            // Following Jolt's E_out_in_for_window logic exactly
            // Use current_index as num_unbound, NOT capped by any external parameter
            const num_unbound = self.current_index;
            const actual_window = @min(window_size, num_unbound);
            const head_len = num_unbound -| actual_window;

            // Split into out and in parts
            const n = self.tau.len;
            const m = n / 2;
            const head_out_bits = @min(head_len, m);
            const head_in_bits = head_len -| head_out_bits;

            // Get tables of appropriate sizes
            const E_out = if (head_out_bits <= self.num_x_out)
                self.E_out_vec.items[head_out_bits]
            else
                self.E_out_vec.items[self.num_x_out];

            const E_in = if (head_in_bits <= self.num_x_in)
                self.E_in_vec.items[head_in_bits]
            else
                self.E_in_vec.items[self.num_x_in];

            return .{ .E_out = E_out, .E_in = E_in, .head_in_bits = head_in_bits };
        }

        /// Compute Gruen's degree-3 round polynomial from multiquadratic evaluations
        ///
        /// Given:
        /// - q_constant = t'(0) (sum over all x with current var = 0)
        /// - q_quadratic_coeff = t'(∞) (slope term)
        /// - previous_claim = s(0) + s(1)
        ///
        /// Returns evaluations [s(0), s(1), s(2), s(3)] for degree-3 polynomial
        pub fn computeCubicRoundPoly(
            self: *const Self,
            q_constant: F,
            q_quadratic_coeff: F,
            previous_claim: F,
        ) [4]F {
            if (self.current_index == 0) {
                return [4]F{ previous_claim, F.zero(), F.zero(), F.zero() };
            }

            // Linear eq polynomial: l(X) = eq(0) + (eq(1) - eq(0)) * X
            // where eq(b) = current_scalar * product of eq(τ_i, b) for current var
            const tau_curr = self.tau[self.current_index - 1];

            // eq(0) for current variable: current_scalar * (1 - τ_curr)
            const eq_0 = self.current_scalar.mul(F.one().sub(tau_curr));
            // eq(1) for current variable: current_scalar * τ_curr
            const eq_1 = self.current_scalar.mul(tau_curr);

            // l(X) = eq_0 + (eq_1 - eq_0) * X
            const l_slope = eq_1.sub(eq_0);

            // l(0), l(1), l(2), l(3)
            const l_0 = eq_0;
            const l_1 = eq_1;
            const l_2 = eq_0.add(l_slope.mul(F.fromU64(2)));
            const l_3 = eq_0.add(l_slope.mul(F.fromU64(3)));

            // Quadratic q(X) = c + d*X + e*X²
            // We know:
            // - q(0) = c = q_constant
            // - q(∞) = e (coefficient of X² extrapolated)
            // - s(0) + s(1) = l(0)*q(0) + l(1)*q(1) = previous_claim
            //
            // Solve for d:
            // l(1)*q(1) = previous_claim - l(0)*q(0)
            // q(1) = (previous_claim - l(0)*q(0)) / l(1)
            // q(1) = c + d + e
            // d = q(1) - c - e

            const l0_q0 = l_0.mul(q_constant);
            const q_1 = if (l_1.eql(F.zero()))
                F.zero()
            else
                previous_claim.sub(l0_q0).mul(l_1.inverse().?);

            // Now we have q(0), q(1), and the quadratic coefficient e
            // q(X) = c + d*X + e*X²
            // c = q_constant
            // d = q(1) - c - e
            const c = q_constant;
            const e = q_quadratic_coeff;
            const d = q_1.sub(c).sub(e);

            // q(2) = c + 2d + 4e
            const q_2 = c.add(d.mul(F.fromU64(2))).add(e.mul(F.fromU64(4)));
            // q(3) = c + 3d + 9e
            const q_3 = c.add(d.mul(F.fromU64(3))).add(e.mul(F.fromU64(9)));

            // s(X) = l(X) * q(X)
            const s_0 = l_0.mul(c); // l(0) * q(0)
            const s_1 = l_1.mul(q_1); // l(1) * q(1)
            const s_2 = l_2.mul(q_2); // l(2) * q(2)
            const s_3 = l_3.mul(q_3); // l(3) * q(3)

            return [4]F{ s_0, s_1, s_2, s_3 };
        }

        /// Compute the linear eq factor values for the current variable
        ///
        /// Returns (eq_0, eq_1) where:
        /// - eq_0 = value when current variable = 0
        /// - eq_1 = value when current variable = 1
        pub fn getCurrentEqFactors(self: *const Self) struct { eq_0: F, eq_1: F } {
            if (self.current_index == 0) {
                return .{ .eq_0 = self.current_scalar, .eq_1 = self.current_scalar };
            }

            const tau_curr = self.tau[self.current_index - 1];
            const eq_0 = self.current_scalar.mul(F.one().sub(tau_curr));
            const eq_1 = self.current_scalar.mul(tau_curr);

            return .{ .eq_0 = eq_0, .eq_1 = eq_1 };
        }
    };
}

// ============================================================================
// Tests
// ============================================================================

const testing = std.testing;
const BN254Scalar = @import("../field/mod.zig").BN254Scalar;

test "GruenSplitEqPolynomial: initialization" {
    const F = BN254Scalar;

    // tau = [τ_0, τ_1, τ_2] with 1 inner variable and 2 outer
    const tau = [_]F{ F.fromU64(2), F.fromU64(3), F.fromU64(5) };

    var split_eq = try GruenSplitEqPolynomial(F).init(testing.allocator, &tau, 1);
    defer split_eq.deinit();

    try testing.expectEqual(@as(usize, 3), split_eq.current_index);
    try testing.expect(split_eq.current_scalar.eql(F.one()));
    try testing.expectEqual(@as(usize, 1), split_eq.num_x_in);
    try testing.expectEqual(@as(usize, 2), split_eq.num_x_out);

    // Check prefix table sizes
    try testing.expectEqual(@as(usize, 2), split_eq.E_in_vec.items.len); // [1], [2]
    try testing.expectEqual(@as(usize, 3), split_eq.E_out_vec.items.len); // [1], [2], [4]
}

test "GruenSplitEqPolynomial: bind updates scalar" {
    const F = BN254Scalar;

    const tau = [_]F{ F.fromU64(2), F.fromU64(3) };

    var split_eq = try GruenSplitEqPolynomial(F).init(testing.allocator, &tau, 1);
    defer split_eq.deinit();

    // Bind with challenge r = 5
    const r = F.fromU64(5);
    split_eq.bind(r);

    try testing.expectEqual(@as(usize, 1), split_eq.current_index);

    // eq(τ[1], r) = eq(3, 5) = 3*5 + (1-3)*(1-5) = 15 + (-2)*(-4) = 15 + 8 = 23
    const tau_1 = F.fromU64(3);
    const expected_eq = tau_1.mul(r).add(F.one().sub(tau_1).mul(F.one().sub(r)));
    try testing.expect(split_eq.current_scalar.eql(expected_eq));
}

test "GruenSplitEqPolynomial: prefix tables correctness" {
    const F = BN254Scalar;

    // Simple case: tau = [2, 3]
    const tau = [_]F{ F.fromU64(2), F.fromU64(3) };

    var split_eq = try GruenSplitEqPolynomial(F).init(testing.allocator, &tau, 1);
    defer split_eq.deinit();

    // E_in_vec[1] should be [1-τ_0, τ_0] = [1-2, 2] = [-1, 2]
    const E_in_1 = split_eq.E_in_vec.items[1];
    try testing.expectEqual(@as(usize, 2), E_in_1.len);

    const expected_e_in_0 = F.one().sub(F.fromU64(2)); // 1 - 2 = -1
    const expected_e_in_1 = F.fromU64(2);
    try testing.expect(E_in_1[0].eql(expected_e_in_0));
    try testing.expect(E_in_1[1].eql(expected_e_in_1));
}

test "GruenSplitEqPolynomial: cubic round poly basic" {
    const F = BN254Scalar;

    const tau = [_]F{ F.fromU64(1), F.fromU64(2) };

    var split_eq = try GruenSplitEqPolynomial(F).init(testing.allocator, &tau, 1);
    defer split_eq.deinit();

    // Compute a cubic round poly with some test values
    const q_constant = F.fromU64(10);
    const q_quadratic = F.fromU64(3);
    const previous_claim = F.fromU64(100);

    const round_poly = split_eq.computeCubicRoundPoly(q_constant, q_quadratic, previous_claim);

    // s(0) + s(1) should equal previous_claim
    const sum = round_poly[0].add(round_poly[1]);
    try testing.expect(sum.eql(previous_claim));
}
