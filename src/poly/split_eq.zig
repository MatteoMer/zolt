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
        /// Full challenge vector τ (tau_low, excluding tau_high)
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
        pub fn init(allocator: Allocator, tau: []const F) !Self {
            return initWithScaling(allocator, tau, null);
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
        pub fn initWithScaling(allocator: Allocator, tau: []const F, scaling_factor: ?F) !Self {
            // Match Jolt's LowToHigh structure exactly:
            //   let m = w.len() / 2;
            //   let (_w_last, wprime) = w.split_last().unwrap();
            //   let (w_out, w_in) = wprime.split_at(m);
            //
            // So:
            //   m = tau.len / 2
            //   w_last = tau[tau.len - 1] (skipped from E tables)
            //   w_out = tau[0..m]
            //   w_in = tau[m..tau.len - 1]

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

            // Split like Jolt: m = tau.len / 2
            // Jolt does: let m = w.len() / 2; let (_w_last, wprime) = w.split_last(); let (w_out, w_in) = wprime.split_at(m);
            // So w_out = tau[0..m], w_in = tau[m..tau.len-1], w_last = tau[tau.len-1] is skipped
            const m = tau.len / 2;
            const num_x_out = m; // First half: tau[0..m]
            const num_x_in_actual = if (tau.len > 1) @min(tau.len - 1 - m, tau.len - 1) else 0; // Second half: tau[m..tau.len-1]

            // Copy tau
            const tau_copy = try allocator.alloc(F, tau.len);
            @memcpy(tau_copy, tau);

            // Build prefix eq tables
            var E_in_vec: std.ArrayListUnmanaged([]F) = .{};
            var E_out_vec: std.ArrayListUnmanaged([]F) = .{};

            // Build outer prefix tables (E_out_vec) for tau[0..m]
            // E_out_vec[0] = [1] (empty eq is always 1)
            //
            // IMPORTANT: Use BIG-ENDIAN indexing to match Jolt's EqPolynomial::evals.
            // For index i with bit-decomposition i = Σ b_j * 2^{n-1-j}:
            //   evals[i] = eq(τ, b₀...b_{n-1}) where b₀ is MSB
            //
            // This means τ[0] controls the MSB (high bit) of the index.
            // When adding τ[k], the existing table (size 2^k) is doubled:
            //   - indices 0..size-1: multiply by (1 - τ[k])
            //   - indices size..2*size-1: multiply by τ[k]
            //
            // But we iterate in reverse to maintain the correct ordering:
            // For each existing entry at index i (which represents some bit pattern),
            // we create two entries:
            //   - index 2*i (append 0 as new LSB): multiplied by (1 - τ[k])
            //   - index 2*i+1 (append 1 as new LSB): multiplied by τ[k]
            //
            // This builds the table such that the MSB corresponds to τ[0].
            try E_out_vec.append(allocator, try allocator.alloc(F, 1));
            E_out_vec.items[0][0] = F.one();

            for (0..num_x_out) |k| {
                const prev_size: usize = @as(usize, 1) << @intCast(k);
                const new_size: usize = prev_size * 2;
                const prev = E_out_vec.items[k];
                const next = try allocator.alloc(F, new_size);

                const tau_k = tau_copy[k]; // w_out uses tau[0..m]
                const one_minus_tau_k = F.one().sub(tau_k);

                // Jolt's big-endian ordering: iterate backwards, doubling indices
                // For each prev[i/2], we set:
                //   next[i] = prev[i/2] * τ[k]       (odd index, bit = 1)
                //   next[i-1] = prev[i/2] * (1-τ[k]) (even index, bit = 0)
                //
                // Equivalently in forward iteration:
                //   next[2*i] = prev[i] * (1-τ[k])   (even index, appended 0)
                //   next[2*i+1] = prev[i] * τ[k]    (odd index, appended 1)
                for (0..prev_size) |i| {
                    next[2 * i] = prev[i].mul(one_minus_tau_k);
                    next[2 * i + 1] = prev[i].mul(tau_k);
                }

                try E_out_vec.append(allocator, next);
            }

            // Build inner prefix tables (E_in_vec) for tau[m..tau.len-1]
            // Same big-endian ordering as E_out_vec
            try E_in_vec.append(allocator, try allocator.alloc(F, 1));
            E_in_vec.items[0][0] = F.one();

            for (0..num_x_in_actual) |k| {
                const prev_size: usize = @as(usize, 1) << @intCast(k);
                const new_size: usize = prev_size * 2;
                const prev = E_in_vec.items[k];
                const next = try allocator.alloc(F, new_size);

                const tau_k = tau_copy[m + k]; // w_in uses tau[m..tau.len-1]
                const one_minus_tau_k = F.one().sub(tau_k);

                // Big-endian: append new bit as LSB
                for (0..prev_size) |i| {
                    next[2 * i] = prev[i].mul(one_minus_tau_k);
                    next[2 * i + 1] = prev[i].mul(tau_k);
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
        /// Returns eq(τ_unbound, ·) scaled by current_scalar.
        /// Uses BIG-ENDIAN indexing: τ[0] controls MSB of index.
        pub fn getFullEqTable(self: *const Self, allocator: Allocator) ![]F {
            const size = @as(usize, 1) << @intCast(self.current_index);
            const result = try allocator.alloc(F, size);

            // Use big-endian construction matching Jolt's EqPolynomial::evals_serial
            // Start with just the scaling factor
            result[0] = self.current_scalar;
            var current_size: usize = 1;

            // For each variable, double the table size
            // τ[0] is first (controls MSB), τ[n-1] is last (controls LSB)
            for (0..self.current_index) |k| {
                const tau_k = self.tau[k];
                const one_minus_tau_k = F.one().sub(tau_k);

                // Iterate in reverse to avoid overwriting needed values
                // For big-endian: even indices get (1-τ), odd indices get τ
                var i: usize = current_size;
                while (i > 0) {
                    i -= 1;
                    const scalar = result[i];
                    // Big-endian: new bit is appended as LSB
                    // Index 2*i (even, bit=0) gets (1-τ)
                    // Index 2*i+1 (odd, bit=1) gets τ
                    result[2 * i + 1] = scalar.mul(tau_k);
                    result[2 * i] = scalar.mul(one_minus_tau_k);
                }
                current_size *= 2;
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
        /// - m = tau.len / 2 = 11 / 2 = 5
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

            // Split into out and in parts using m = tau.len / 2
            // This matches Jolt's calculation: m = w.len() / 2
            const m = self.tau.len / 2;
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

        /// Get the eq table for the "active" window bits (all window bits except the current Gruen variable)
        ///
        /// This is used when projecting a multiquadratic polynomial t'(z_0, ..., z_{w-1}) down to
        /// a univariate in the first variable by summing against eq(tau_active, ·) over the
        /// remaining coordinates.
        ///
        /// For window_size = 1: returns [1] (eq over zero variables)
        /// For window_size > 1: returns eq(w_active, ·) where w_active is the first (window_size-1) bits of the window
        ///
        /// The complete factorization satisfies:
        ///   log2(|E_out|) + log2(|E_in|) + log2(|E_active|) + 1 = #unbound bits
        ///
        /// Reference: jolt-core/src/poly/split_eq_poly.rs E_active_for_window
        pub fn getEActiveForWindow(self: *const Self, allocator: Allocator, window_size: usize) ![]F {
            // No active bits in a size-0/1 window; eq over zero vars is [1].
            if (window_size <= 1) {
                const result = try allocator.alloc(F, 1);
                result[0] = F.one();
                return result;
            }

            const num_unbound = self.current_index;

            // Clamp to the maximum meaningful window size at this round.
            if (window_size > num_unbound) {
                const result = try allocator.alloc(F, 1);
                result[0] = F.one();
                return result;
            }

            // Extract the active window bits: w[window_start..window_start + window_size - 1]
            // where window_start = remaining_w.len - window_size = num_unbound - window_size
            const window_start = num_unbound - window_size;
            const active_bits = window_size - 1; // All window bits except the current (last) one

            // Compute full eq table over the active window bits
            // w_active = tau[window_start..window_start + active_bits]
            const table_size = @as(usize, 1) << @intCast(active_bits);
            const result = try allocator.alloc(F, table_size);

            // Build eq table using big-endian indexing
            result[0] = F.one();
            var current_size: usize = 1;

            for (0..active_bits) |k| {
                const tau_k = self.tau[window_start + k];
                const one_minus_tau_k = F.one().sub(tau_k);

                // Iterate in reverse to avoid overwriting needed values
                var i: usize = current_size;
                while (i > 0) {
                    i -= 1;
                    const scalar = result[i];
                    // Big-endian: new bit is appended as LSB
                    result[2 * i + 1] = scalar.mul(tau_k);
                    result[2 * i] = scalar.mul(one_minus_tau_k);
                }
                current_size *= 2;
            }

            return result;
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

    // tau = [τ_0, τ_1, τ_2] (length 3)
    // Split: m = 3/2 = 1
    // w_last = tau[2] (skipped)
    // w_out = tau[0..1] = [τ_0]
    // w_in = tau[1..2] = [τ_1]
    const tau = [_]F{ F.fromU64(2), F.fromU64(3), F.fromU64(5) };

    var split_eq = try GruenSplitEqPolynomial(F).init(testing.allocator, &tau);
    defer split_eq.deinit();

    try testing.expectEqual(@as(usize, 3), split_eq.current_index);
    try testing.expect(split_eq.current_scalar.eql(F.one()));

    // m = 3/2 = 1
    // num_x_out = 1 (tau[0..1])
    // num_x_in = 3 - 1 - 1 = 1 (tau[1..2])
    try testing.expectEqual(@as(usize, 1), split_eq.num_x_in);
    try testing.expectEqual(@as(usize, 1), split_eq.num_x_out);

    // Check prefix table sizes
    // E_in_vec: [0] = [1], [1] = [2] -> 2 tables
    // E_out_vec: [0] = [1], [1] = [2] -> 2 tables
    try testing.expectEqual(@as(usize, 2), split_eq.E_in_vec.items.len);
    try testing.expectEqual(@as(usize, 2), split_eq.E_out_vec.items.len);
}

test "GruenSplitEqPolynomial: bind updates scalar" {
    const F = BN254Scalar;

    // tau with length 2
    const tau = [_]F{ F.fromU64(2), F.fromU64(3) };

    var split_eq = try GruenSplitEqPolynomial(F).init(testing.allocator, &tau);
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

    // tau = [2, 3, 5] with length 3
    // m = 3/2 = 1
    // w_last = tau[2] = 5 (skipped)
    // w_out = tau[0..1] = [2]
    // w_in = tau[1..2] = [3]
    const tau = [_]F{ F.fromU64(2), F.fromU64(3), F.fromU64(5) };

    var split_eq = try GruenSplitEqPolynomial(F).init(testing.allocator, &tau);
    defer split_eq.deinit();

    // E_out_vec[1] should have 2 entries for tau[0] = 2
    // In big-endian: E_out[0] = (1-τ_0), E_out[1] = τ_0
    const E_out_1 = split_eq.E_out_vec.items[1];
    try testing.expectEqual(@as(usize, 2), E_out_1.len);

    const expected_e_out_0 = F.one().sub(F.fromU64(2)); // 1 - 2 = -1
    const expected_e_out_1 = F.fromU64(2);
    try testing.expect(E_out_1[0].eql(expected_e_out_0));
    try testing.expect(E_out_1[1].eql(expected_e_out_1));

    // E_in_vec[1] should have 2 entries for tau[1] = 3 (since m=1, w_in starts at tau[1])
    const E_in_1 = split_eq.E_in_vec.items[1];
    try testing.expectEqual(@as(usize, 2), E_in_1.len);

    const expected_e_in_0 = F.one().sub(F.fromU64(3)); // 1 - 3 = -2
    const expected_e_in_1 = F.fromU64(3);
    try testing.expect(E_in_1[0].eql(expected_e_in_0));
    try testing.expect(E_in_1[1].eql(expected_e_in_1));
}

test "GruenSplitEqPolynomial: cubic round poly basic" {
    const F = BN254Scalar;

    // tau with length 2
    const tau = [_]F{ F.fromU64(1), F.fromU64(2) };

    var split_eq = try GruenSplitEqPolynomial(F).init(testing.allocator, &tau);
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

test "GruenSplitEqPolynomial: big-endian eq table correctness" {
    const F = BN254Scalar;

    // tau = [τ_0, τ_1, τ_2, τ_3] with length 4
    // m = 4/2 = 2
    // w_last = tau[3] = 11 (skipped)
    // E_out uses tau[0..2] = [3, 5] and E_in uses tau[2..3] = [7]
    // getFullEqTable combines all unbound variables
    const tau = [_]F{ F.fromU64(3), F.fromU64(5), F.fromU64(7), F.fromU64(11) };

    var split_eq = try GruenSplitEqPolynomial(F).init(testing.allocator, &tau);
    defer split_eq.deinit();

    // Get the full eq table (should have 16 entries since current_index = 4)
    const eq_table = try split_eq.getFullEqTable(testing.allocator);
    defer testing.allocator.free(eq_table);

    try testing.expectEqual(@as(usize, 16), eq_table.len);

    // In big-endian ordering, index i encodes bits (b_0, b_1, b_2, b_3) where b_0 is MSB
    // eq(τ, x) = Π_j (τ_j * b_j + (1-τ_j) * (1-b_j))

    const tau0 = F.fromU64(3);
    const tau1 = F.fromU64(5);
    const tau2 = F.fromU64(7);
    const tau3 = F.fromU64(11);
    const one_minus_tau0 = F.one().sub(tau0);
    const one_minus_tau1 = F.one().sub(tau1);
    const one_minus_tau2 = F.one().sub(tau2);
    const one_minus_tau3 = F.one().sub(tau3);

    // Index 0 (0000): all bits 0 → (1-τ_0)*(1-τ_1)*(1-τ_2)*(1-τ_3)
    const expected_0 = one_minus_tau0.mul(one_minus_tau1).mul(one_minus_tau2).mul(one_minus_tau3);

    // Index 15 (1111): all bits 1 → τ_0*τ_1*τ_2*τ_3
    const expected_15 = tau0.mul(tau1).mul(tau2).mul(tau3);

    // Index 5 (0101): bits = (0,1,0,1) → (1-τ_0)*τ_1*(1-τ_2)*τ_3
    const expected_5 = one_minus_tau0.mul(tau1).mul(one_minus_tau2).mul(tau3);

    // Index 10 (1010): bits = (1,0,1,0) → τ_0*(1-τ_1)*τ_2*(1-τ_3)
    const expected_10 = tau0.mul(one_minus_tau1).mul(tau2).mul(one_minus_tau3);

    try testing.expect(eq_table[0].eql(expected_0));
    try testing.expect(eq_table[5].eql(expected_5));
    try testing.expect(eq_table[10].eql(expected_10));
    try testing.expect(eq_table[15].eql(expected_15));
}

test "GruenSplitEqPolynomial: getEActiveForWindow" {
    const F = BN254Scalar;

    // tau = [τ_0, τ_1, τ_2, τ_3] with length 4
    const tau = [_]F{ F.fromU64(3), F.fromU64(5), F.fromU64(7), F.fromU64(11) };

    var split_eq = try GruenSplitEqPolynomial(F).init(testing.allocator, &tau);
    defer split_eq.deinit();

    // With 4 unbound variables and window_size = 1:
    // - No active bits (window_size - 1 = 0)
    // - Should return [1]
    const e_active_1 = try split_eq.getEActiveForWindow(testing.allocator, 1);
    defer testing.allocator.free(e_active_1);
    try testing.expectEqual(@as(usize, 1), e_active_1.len);
    try testing.expect(e_active_1[0].eql(F.one()));

    // With 4 unbound variables and window_size = 2:
    // - window starts at index 4-2=2, so window = tau[2..4] = [7, 11]
    // - active = tau[2..3] = [7] (1 bit)
    // - E_active should have 2 entries: [1-7, 7] = [-6, 7]
    const e_active_2 = try split_eq.getEActiveForWindow(testing.allocator, 2);
    defer testing.allocator.free(e_active_2);
    try testing.expectEqual(@as(usize, 2), e_active_2.len);
    const expected_0_2 = F.one().sub(F.fromU64(7)); // 1 - 7 = -6
    const expected_1_2 = F.fromU64(7);
    try testing.expect(e_active_2[0].eql(expected_0_2));
    try testing.expect(e_active_2[1].eql(expected_1_2));

    // With 4 unbound variables and window_size = 3:
    // - window starts at index 4-3=1, so window = tau[1..4] = [5, 7, 11]
    // - active = tau[1..3] = [5, 7] (2 bits)
    // - E_active should have 4 entries: eq([5,7], {0,0}), eq([5,7], {0,1}), eq([5,7], {1,0}), eq([5,7], {1,1})
    const e_active_3 = try split_eq.getEActiveForWindow(testing.allocator, 3);
    defer testing.allocator.free(e_active_3);
    try testing.expectEqual(@as(usize, 4), e_active_3.len);

    const tau_active_0 = F.fromU64(5);
    const tau_active_1 = F.fromU64(7);
    const one_minus_5 = F.one().sub(tau_active_0);
    const one_minus_7 = F.one().sub(tau_active_1);

    // Index 0 (00): (1-5)*(1-7) = -4 * -6 = 24
    const exp_00 = one_minus_5.mul(one_minus_7);
    // Index 1 (01): (1-5)*7 = -4 * 7 = -28
    const exp_01 = one_minus_5.mul(tau_active_1);
    // Index 2 (10): 5*(1-7) = 5 * -6 = -30
    const exp_10 = tau_active_0.mul(one_minus_7);
    // Index 3 (11): 5*7 = 35
    const exp_11 = tau_active_0.mul(tau_active_1);

    try testing.expect(e_active_3[0].eql(exp_00));
    try testing.expect(e_active_3[1].eql(exp_01));
    try testing.expect(e_active_3[2].eql(exp_10));
    try testing.expect(e_active_3[3].eql(exp_11));
}
