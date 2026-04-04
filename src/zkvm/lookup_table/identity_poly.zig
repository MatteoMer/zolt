//! Identity and Operand Polynomials for Jolt-Compatible Sumcheck
//!
//! This module implements the IdentityPolynomial and OperandPolynomial types used
//! in Jolt's RAF (Read-Address-Flag) computation during the LookupsReadRaf sumcheck.
//!
//! ## IdentityPolynomial
//!
//! Evaluates to the binary index encoded by the input bits:
//!   Identity(r_0, r_1, ..., r_{n-1}) = Σ r_i * 2^(n-1-i)
//!
//! For HighToLow binding: bound_value += bound_value; bound_value += r;
//!
//! ## OperandPolynomial
//!
//! Evaluates to either the left or right operand extracted from interleaved bits.
//! In Jolt's interleave format: interleaved = (left << 1) | right
//! - Left: sum_{i=0}^{n/2-1} r[2i+1] * 2^(n/2-1-i)  (odd positions)
//! - Right: sum_{i=0}^{n/2-1} r[2i] * 2^(n/2-1-i) (even positions)
//!
//! Reference: jolt-core/src/poly/identity_poly.rs

const std = @import("std");
const Allocator = std.mem.Allocator;
const prefixes_mod = @import("prefixes.zig");
const LookupBits = prefixes_mod.LookupBits;

/// Binding order for polynomial evaluation
pub const BindingOrder = enum {
    LowToHigh,
    HighToLow,
};

/// Operand side for OperandPolynomial
pub const OperandSide = enum {
    Left,
    Right,
};

/// IdentityPolynomial evaluates to the binary index encoded by the input bits
pub fn IdentityPolynomial(comptime F: type) type {
    return struct {
        const Self = @This();

        num_vars: usize,
        num_bound_vars: usize,
        bound_value: F,

        /// Create a new identity polynomial
        pub fn init(num_vars: usize) Self {
            return .{
                .num_vars = num_vars,
                .num_bound_vars = 0,
                .bound_value = F.zero(),
            };
        }

        /// Bind the next variable to a challenge
        pub fn bind(self: *Self, r: F, order: BindingOrder) void {
            std.debug.assert(self.num_bound_vars < self.num_vars);

            switch (order) {
                .LowToHigh => {
                    // bound_value += 2^num_bound_vars * r
                    const shift = @as(u128, 1) << @intCast(self.num_bound_vars);
                    self.bound_value = self.bound_value.add(F.fromU128(shift).mul(r));
                },
                .HighToLow => {
                    // bound_value = bound_value * 2 + r
                    self.bound_value = self.bound_value.add(self.bound_value);
                    self.bound_value = self.bound_value.add(r);
                },
            }
            self.num_bound_vars += 1;
        }

        /// Get the final sumcheck claim after all variables are bound
        pub fn finalSumcheckClaim(self: *const Self) F {
            std.debug.assert(self.num_vars == self.num_bound_vars);
            return self.bound_value;
        }

        /// Evaluate the identity polynomial at a point
        pub fn evaluate(self: *const Self, r: []const F) F {
            std.debug.assert(r.len == self.num_vars);
            var result = F.zero();
            for (r, 0..) |ri, i| {
                const shift = @as(u128, 1) << @intCast(self.num_vars - 1 - i);
                result = result.add(ri.mul(F.fromU128(shift)));
            }
            return result;
        }

        /// Compute sumcheck evaluations at index for given degree
        /// Returns [p(1), p(2), ..., p(degree)] where p(0) is derived
        pub fn sumcheckEvals(self: *const Self, index: usize, degree: usize, order: BindingOrder) [3]F {
            var evals: [3]F = .{ F.zero(), F.zero(), F.zero() };
            std.debug.assert(degree <= 3);

            const m: F = switch (order) {
                .LowToHigh => blk: {
                    const m = F.fromU128(@as(u128, 1) << @intCast(self.num_bound_vars));
                    evals[0] = self.bound_value.add(m.add(m).mul(F.fromU64(@as(u64, @intCast(index)))));
                    break :blk m;
                },
                .HighToLow => blk: {
                    const m = F.fromU128(@as(u128, 1) << @intCast(self.num_vars - 1 - self.num_bound_vars));
                    evals[0] = self.bound_value.mul(m.add(m)).add(F.fromU64(@as(u64, @intCast(index))));
                    break :blk m;
                },
            };

            var eval = evals[0].add(m);
            var i: usize = 1;
            while (i < degree) : (i += 1) {
                eval = eval.add(m);
                evals[i] = eval;
            }

            return evals;
        }
    };
}

/// OperandPolynomial evaluates to either the left or right operand from uninterleaved bits
pub fn OperandPolynomial(comptime F: type) type {
    return struct {
        const Self = @This();

        num_vars: usize,
        num_bound_vars: usize,
        bound_value: F,
        side: OperandSide,

        /// Create a new operand polynomial
        pub fn init(num_vars: usize, side: OperandSide) Self {
            std.debug.assert(num_vars % 2 == 0); // num_vars must be even
            return .{
                .num_vars = num_vars,
                .num_bound_vars = 0,
                .bound_value = F.zero(),
                .side = side,
            };
        }

        /// Bind the next variable to a challenge (HighToLow binding only)
        pub fn bind(self: *Self, r: F) void {
            std.debug.assert(self.num_bound_vars < self.num_vars);

            // Only update bound_value when binding the relevant operand variable
            // In Jolt's interleave format: odd positions are left, even positions are right
            const is_left_var = self.num_bound_vars % 2 == 1;
            const is_right_var = self.num_bound_vars % 2 == 0;

            if ((is_left_var and self.side == .Left) or
                (is_right_var and self.side == .Right))
            {
                self.bound_value = self.bound_value.add(self.bound_value);
                self.bound_value = self.bound_value.add(r);
            }

            self.num_bound_vars += 1;
        }

        /// Get the final sumcheck claim
        pub fn finalSumcheckClaim(self: *const Self) F {
            std.debug.assert(self.num_vars == self.num_bound_vars);
            return self.bound_value;
        }

        /// Evaluate the operand polynomial at a point
        pub fn evaluate(self: *const Self, r: []const F) F {
            std.debug.assert(r.len == self.num_vars);
            std.debug.assert(r.len % 2 == 0);

            var result = F.zero();
            const half_len = r.len / 2;

            switch (self.side) {
                .Left => {
                    // Sum over odd positions (left operand in Jolt's format)
                    for (0..half_len) |i| {
                        const shift = @as(u128, 1) << @intCast(half_len - 1 - i);
                        result = result.add(r[2 * i + 1].mul(F.fromU128(shift)));
                    }
                },
                .Right => {
                    // Sum over even positions (right operand in Jolt's format)
                    for (0..half_len) |i| {
                        const shift = @as(u128, 1) << @intCast(half_len - 1 - i);
                        result = result.add(r[2 * i].mul(F.fromU128(shift)));
                    }
                },
            }

            return result;
        }

        /// Uninterleave bits to get left and right operand values
        /// In Jolt's interleave format: interleaved = (left << 1) | right
        /// So left operand bits are at ODD positions (1, 3, 5, ...)
        /// and right operand bits are at EVEN positions (0, 2, 4, ...)
        fn uninterleaveBits(index: u128) struct { left: u64, right: u64 } {
            var left: u64 = 0;
            var right: u64 = 0;
            var i: u6 = 0;
            while (i < 64) : (i += 1) {
                // Left operand at odd positions (1, 3, 5, ...)
                const left_bit = (index >> @as(u7, @intCast(2 * i + 1))) & 1;
                // Right operand at even positions (0, 2, 4, ...)
                const right_bit = (index >> @as(u7, @intCast(2 * i))) & 1;
                left |= @as(u64, @truncate(left_bit)) << i;
                right |= @as(u64, @truncate(right_bit)) << i;
            }
            return .{ .left = left, .right = right };
        }

        /// Compute sumcheck evaluations (HighToLow binding only)
        pub fn sumcheckEvals(self: *const Self, index: usize, degree: usize) [3]F {
            var evals: [3]F = .{ F.zero(), F.zero(), F.zero() };
            std.debug.assert(degree <= 3);

            const uninterleaved = uninterleaveBits(@as(u128, index));
            const operand_index = switch (self.side) {
                .Left => F.fromU64(uninterleaved.left),
                .Right => F.fromU64(uninterleaved.right),
            };

            const is_even_bound = self.num_bound_vars % 2 == 0;

            if (is_even_bound) {
                const unbound_pairs = (self.num_vars - self.num_bound_vars) / 2;
                std.debug.assert(unbound_pairs > 0);

                evals[0] = self.bound_value.mul(F.fromU128(@as(u128, 1) << @intCast(unbound_pairs))).add(operand_index);

                if (self.side == .Left) {
                    const m = F.fromU128(@as(u128, 1) << @intCast(unbound_pairs - 1));
                    var eval = evals[0].add(m);
                    for (1..degree) |i| {
                        eval = eval.add(m);
                        evals[i] = eval;
                    }
                } else {
                    // Right operand - constant for this variable
                    for (1..degree) |i| {
                        evals[i] = evals[0];
                    }
                }
            } else if (self.side == .Right) {
                // Binding right operand variable
                const unbound_pairs = (self.num_vars - self.num_bound_vars + 1) / 2;
                evals[0] = self.bound_value.mul(F.fromU128(@as(u128, 1) << @intCast(unbound_pairs))).add(operand_index);
                const m = F.fromU128(@as(u128, 1) << @intCast(unbound_pairs - 1));
                var eval = evals[0].add(m);
                for (1..degree) |i| {
                    eval = eval.add(m);
                    evals[i] = eval;
                }
            } else {
                // Binding right variable but this is LeftOperand - constant
                const unbound_pairs = (self.num_vars - self.num_bound_vars) / 2;
                evals[0] = self.bound_value.mul(F.fromU128(@as(u128, 1) << @intCast(unbound_pairs))).add(operand_index);
                for (1..degree) |i| {
                    evals[i] = evals[0];
                }
            }

            return evals;
        }
    };
}

/// PrefixSuffixDecomposition for Identity/Operand polynomials
pub fn PrefixSuffixDecomposition(comptime F: type, comptime ORDER: usize) type {
    return struct {
        const Self = @This();

        /// Q accumulator for each suffix term
        Q: [ORDER][]F,
        /// Current Q size
        Q_size: usize,

        allocator: Allocator,

        pub fn init(allocator: Allocator, initial_size: usize) !Self {
            var Q: [ORDER][]F = undefined;
            for (0..ORDER) |i| {
                Q[i] = try allocator.alloc(F, initial_size);
                @memset(Q[i], F.zero());
            }
            return .{
                .Q = Q,
                .Q_size = initial_size,
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            for (0..ORDER) |i| {
                self.allocator.free(self.Q[i]);
            }
        }

        /// Get the current Q length
        pub fn QLen(self: *const Self) usize {
            return self.Q_size;
        }

        /// Compute sumcheck evaluations at index b
        /// Returns (eval_at_0, eval_at_2) for degree-2 interpolation
        pub fn sumcheckEvals(self: *const Self, b: usize) struct { e0: F, e2: F } {
            // For identity/operand: combine Q[0] (shift suffix) and Q[1] (operand suffix)
            // eval_at_c = Σ_i P_i(c) * Q_i[b]
            // For now, simplified: just return Q[0][b] as a placeholder

            // In full implementation, this would combine prefix evaluations with Q accumulators
            const e0 = self.Q[0][2 * b];
            const e2 = self.Q[1][2 * b]; // Simplified - actual impl uses prefix_mle

            return .{ .e0 = e0, .e2 = e2 };
        }

        /// Bind a challenge (halves the Q arrays)
        pub fn bind(self: *Self, r: F) void {
            const half_size = self.Q_size / 2;
            for (0..ORDER) |i| {
                for (0..half_size) |j| {
                    const low = self.Q[i][2 * j];
                    const high = self.Q[i][2 * j + 1];
                    self.Q[i][j] = low.add(r.mul(high.sub(low)));
                }
            }
            self.Q_size = half_size;
        }

        /// Initialize Q accumulators for RAF computation
        pub fn initQRaf(
            self: *Self,
            u_evals: []const F,
            lookup_indices: []const u128,
            is_interleaved_operands: []const bool,
        ) void {
            // Reset Q accumulators
            for (0..ORDER) |i| {
                @memset(self.Q[i], F.zero());
            }

            // Accumulate weighted contributions
            for (u_evals, 0..) |u, j| {
                const idx = lookup_indices[j];
                const chunk_idx = @as(usize, @truncate(idx)) % self.Q_size;

                // Accumulate based on path (interleaved vs identity)
                if (is_interleaved_operands[j]) {
                    // Interleaved path: accumulate into Q[0] (left) and Q[1] (right)
                    self.Q[0][chunk_idx] = self.Q[0][chunk_idx].add(u);
                } else {
                    // Identity path: accumulate into Q[1] (identity)
                    self.Q[1][chunk_idx] = self.Q[1][chunk_idx].add(u);
                }
            }
        }
    };
}

// ============================================================================
// Tests
// ============================================================================

test "IdentityPolynomial evaluate" {
    const F = @import("zolt_arith").field.BN254Scalar;

    const poly = IdentityPolynomial(F).init(4);

    // Identity at (1, 0, 1, 1) should be 1*8 + 0*4 + 1*2 + 1*1 = 11
    const point = [_]F{ F.one(), F.zero(), F.one(), F.one() };
    const result = poly.evaluate(&point);
    try std.testing.expect(result.eql(F.fromU64(11)));
}

test "IdentityPolynomial bind HighToLow" {
    const F = @import("zolt_arith").field.BN254Scalar;

    var poly = IdentityPolynomial(F).init(4);

    // Bind variables one by one: r_0=1, r_1=0, r_2=1, r_3=1
    poly.bind(F.one(), .HighToLow); // bound = 0*2 + 1 = 1
    poly.bind(F.zero(), .HighToLow); // bound = 1*2 + 0 = 2
    poly.bind(F.one(), .HighToLow); // bound = 2*2 + 1 = 5
    poly.bind(F.one(), .HighToLow); // bound = 5*2 + 1 = 11

    const claim = poly.finalSumcheckClaim();
    try std.testing.expect(claim.eql(F.fromU64(11)));
}

test "OperandPolynomial evaluate" {
    const F = @import("zolt_arith").field.BN254Scalar;

    const left_poly = OperandPolynomial(F).init(8, .Left);
    const right_poly = OperandPolynomial(F).init(8, .Right);

    // For interleaved (x0, y0, x1, y1, x2, y2, x3, y3)
    // Left = x0*8 + x1*4 + x2*2 + x3*1
    // Right = y0*8 + y1*4 + y2*2 + y3*1
    const point = [_]F{
        F.one(), F.zero(), // (x0=1, y0=0)
        F.one(), F.one(), // (x1=1, y1=1)
        F.zero(), F.one(), // (x2=0, y2=1)
        F.one(), F.zero(), // (x3=1, y3=0)
    };

    // Left = 1*8 + 1*4 + 0*2 + 1*1 = 13
    const left_result = left_poly.evaluate(&point);
    try std.testing.expect(left_result.eql(F.fromU64(13)));

    // Right = 0*8 + 1*4 + 1*2 + 0*1 = 6
    const right_result = right_poly.evaluate(&point);
    try std.testing.expect(right_result.eql(F.fromU64(6)));
}
