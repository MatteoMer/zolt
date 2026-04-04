//! Register Value Evaluation Sumcheck
//!
//! Proves the relation:
//!   Val(r) = Σ_{j=0}^{T-1} inc(j) · wa(r_address, j) · LT(r_cycle, j)
//!
//! Where:
//! - r = (r_address, r_cycle) is the evaluation point from Stage 4 (RegistersRWC)
//! - Val(r) is the claimed register value at address r_address and time r_cycle
//! - inc(j) is rd_inc: the change in destination register value at cycle j
//! - wa(r_address, j) is rd's write-address indicator (1 if rd matches r_address)
//! - LT(r_cycle, j) is the less-than MLE: 1 iff j < r_cycle
//!
//! This is Stage 5, Instance 0 in Jolt's verification.
//!
//! Reference: jolt-core/src/zkvm/registers/val_evaluation.rs

const std = @import("std");
const Allocator = std.mem.Allocator;

const poly_mod = @import("zolt_arith").poly;
const EqPolynomial = poly_mod.EqPolynomial;

/// Constants
const LOG_K: usize = 7; // log2(128 registers)
const REGISTER_COUNT: usize = 128; // Jolt uses 128 register slots

/// Sumcheck instance parameters for RegistersValEvaluation
pub fn RegistersValEvaluationParams(comptime F: type) type {
    return struct {
        const Self = @This();

        /// r_address from Stage 4 (LOG_K bits)
        r_address: []const F,
        /// r_cycle from Stage 4 (log_T bits)
        r_cycle: []const F,
        /// Number of rounds = r_cycle.len
        num_rounds: usize,
        /// Input claim: Val(r) from Stage 4 accumulator
        input_claim: F,

        allocator: Allocator,

        pub fn init(
            allocator: Allocator,
            r_address: []const F,
            r_cycle: []const F,
            input_claim: F,
        ) !Self {
            const r_addr_copy = try allocator.alloc(F, r_address.len);
            @memcpy(r_addr_copy, r_address);

            const r_cycle_copy = try allocator.alloc(F, r_cycle.len);
            @memcpy(r_cycle_copy, r_cycle);

            return Self{
                .r_address = r_addr_copy,
                .r_cycle = r_cycle_copy,
                .num_rounds = r_cycle.len,
                .input_claim = input_claim,
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.r_address);
            self.allocator.free(self.r_cycle);
        }
    };
}

/// Prover for RegistersValEvaluation sumcheck
///
/// Computes: Σ_j inc(j) · wa(j) · LT(j, r_cycle)
/// where j ranges over cycles 0..T-1
pub fn RegistersValEvaluationProver(comptime F: type) type {
    return struct {
        const Self = @This();

        /// inc evaluations: rd_inc(j) for each cycle j
        inc_evals: []F,
        /// wa evaluations: eq(r_address, rd[j]) for each cycle j
        wa_evals: []F,
        /// LT evaluations: LT(j, r_cycle) for each cycle j
        lt_evals: []F,

        /// Current effective length (shrinks after binding)
        num_vars: usize,
        /// Current round
        round: usize,
        /// Current claim
        current_claim: F,

        /// Parameters
        params: RegistersValEvaluationParams(F),

        allocator: Allocator,

        /// Initialize the prover from trace data
        ///
        /// @param rd_inc: The rd_inc polynomial evaluations (one per cycle)
        /// @param rd_addresses: The rd register address (0-127) for each cycle
        /// @param params: Sumcheck parameters including r_address, r_cycle
        pub fn init(
            allocator: Allocator,
            rd_inc: []const F,
            rd_addresses: []const ?u8, // null if no rd write
            params: RegistersValEvaluationParams(F),
        ) !Self {
            const trace_len = rd_inc.len;
            const padded_len = if (trace_len == 0)
                1
            else
                std.math.ceilPowerOfTwo(usize, trace_len) catch trace_len;
            const num_vars = if (padded_len <= 1) 0 else std.math.log2_int(usize, padded_len);

            // Allocate evaluation arrays
            const inc_evals = try allocator.alloc(F, padded_len);
            const wa_evals = try allocator.alloc(F, padded_len);
            const lt_evals = try allocator.alloc(F, padded_len);

            // Initialize with zeros
            for (inc_evals) |*e| e.* = F.zero();
            for (wa_evals) |*e| e.* = F.zero();
            for (lt_evals) |*e| e.* = F.zero();

            // Fill in actual values
            for (0..trace_len) |j| {
                // inc(j) = rd_inc value at cycle j
                inc_evals[j] = rd_inc[j];

                // wa(j) = eq(r_address, rd[j]) if rd[j] is valid
                if (rd_addresses[j]) |rd| {
                    wa_evals[j] = computeEqAtPoint(F, params.r_address, rd);
                } else {
                    wa_evals[j] = F.zero();
                }

                // LT(j, r_cycle) = strict less-than MLE
                lt_evals[j] = computeLtAtIndex(F, params.r_cycle, j);
            }

            // Compute initial claim: Σ_j inc(j) · wa(j) · LT(j)
            var initial_claim = F.zero();
            for (0..padded_len) |j| {
                initial_claim = initial_claim.add(
                    inc_evals[j].mul(wa_evals[j]).mul(lt_evals[j]),
                );
            }

            return Self{
                .inc_evals = inc_evals,
                .wa_evals = wa_evals,
                .lt_evals = lt_evals,
                .num_vars = num_vars,
                .round = 0,
                .current_claim = initial_claim,
                .params = params,
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.inc_evals);
            self.allocator.free(self.wa_evals);
            self.allocator.free(self.lt_evals);
            self.params.deinit();
        }

        /// Get number of sumcheck rounds
        pub fn numRounds(self: *const Self) usize {
            return self.num_vars;
        }

        /// Get the input claim
        pub fn getInputClaim(self: *const Self) F {
            return self.params.input_claim;
        }

        /// Get effective array length after binding
        fn effectiveLen(self: *const Self) usize {
            return self.inc_evals.len >> @intCast(self.round);
        }

        /// Compute round polynomial evaluations [p(0), p(1), p(2), p(3)]
        ///
        /// For degree-3 sumcheck (product of 3 multilinear), we need 4 evaluations.
        pub fn computeRoundPolynomial(self: *Self) [4]F {
            var evals: [4]F = .{ F.zero(), F.zero(), F.zero(), F.zero() };
            const n = self.effectiveLen();
            const half = n / 2;

            if (half == 0) {
                if (n > 0) {
                    evals[0] = self.inc_evals[0].mul(self.wa_evals[0]).mul(self.lt_evals[0]);
                }
                return evals;
            }

            for (0..half) |i| {
                // For LowToHigh binding: x=0 at index 2*i, x=1 at index 2*i+1
                const inc_0 = self.inc_evals[2 * i];
                const wa_0 = self.wa_evals[2 * i];
                const lt_0 = self.lt_evals[2 * i];

                const inc_1 = self.inc_evals[2 * i + 1];
                const wa_1 = self.wa_evals[2 * i + 1];
                const lt_1 = self.lt_evals[2 * i + 1];

                // p(0): product at x = 0
                evals[0] = evals[0].add(inc_0.mul(wa_0).mul(lt_0));

                // p(1): product at x = 1
                evals[1] = evals[1].add(inc_1.mul(wa_1).mul(lt_1));

                // Extrapolate to x=2 for degree-3 polynomial
                const two = F.fromU64(2);

                // f(2) = 2*f(1) - f(0) for multilinear
                const inc_2 = two.mul(inc_1).sub(inc_0);
                const wa_2 = two.mul(wa_1).sub(wa_0);
                const lt_2 = two.mul(lt_1).sub(lt_0);
                evals[2] = evals[2].add(inc_2.mul(wa_2).mul(lt_2));

                // For Toom-Cook format, evals[3] = p_inf = c3 (leading coefficient)
                // For product of 3 multilinear polynomials, c3 = product of slopes
                // slope = f(1) - f(0) for each multilinear
                const inc_slope = inc_1.sub(inc_0);
                const wa_slope = wa_1.sub(wa_0);
                const lt_slope = lt_1.sub(lt_0);
                evals[3] = evals[3].add(inc_slope.mul(wa_slope).mul(lt_slope));
            }

            return evals;
        }

        /// Bind the current variable to challenge r
        pub fn bindChallenge(self: *Self, r: F) void {
            const n = self.effectiveLen();
            const half = n / 2;
            if (half == 0) {
                self.round += 1;
                return;
            }

            const one_minus_r = F.one().sub(r);

            // Fold all three polynomials: new[i] = (1-r)*old[2*i] + r*old[2*i+1]
            for (0..half) |i| {
                self.inc_evals[i] = one_minus_r.mul(self.inc_evals[2 * i]).add(r.mul(self.inc_evals[2 * i + 1]));
                self.wa_evals[i] = one_minus_r.mul(self.wa_evals[2 * i]).add(r.mul(self.wa_evals[2 * i + 1]));
                self.lt_evals[i] = one_minus_r.mul(self.lt_evals[2 * i]).add(r.mul(self.lt_evals[2 * i + 1]));
            }

            // Zero out upper half
            for (half..n) |i| {
                self.inc_evals[i] = F.zero();
                self.wa_evals[i] = F.zero();
                self.lt_evals[i] = F.zero();
            }

            self.round += 1;
        }

        /// Get final polynomial openings after all rounds
        pub fn getFinalOpenings(self: *const Self) struct { inc: F, wa: F } {
            return .{
                .inc = if (self.inc_evals.len > 0) self.inc_evals[0] else F.zero(),
                .wa = if (self.wa_evals.len > 0) self.wa_evals[0] else F.zero(),
            };
        }

        /// Get the final claim (product at the bound point)
        pub fn getFinalClaim(self: *const Self) F {
            if (self.inc_evals.len == 0) return F.zero();
            return self.inc_evals[0].mul(self.wa_evals[0]).mul(self.lt_evals[0]);
        }
    };
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Compute eq(r, k) for a specific index k (LE bit order)
fn computeEqAtPoint(comptime F: type, r: []const F, k: anytype) F {
    const k_val: usize = @intCast(k);
    var result = F.one();
    for (r, 0..) |ri, i| {
        const ki = (k_val >> @intCast(i)) & 1;
        if (ki == 1) {
            result = result.mul(ri);
        } else {
            result = result.mul(F.one().sub(ri));
        }
    }
    return result;
}

/// Compute LT(j, r_cycle) for index j
/// LT(x, y) = 1 iff x < y as bitstrings
fn computeLtAtIndex(comptime F: type, r_cycle: []const F, j: usize) F {
    // LT(j, r) = Σ_{i} (1-j_i) * r[i] * Π_{k>i} eq(j_k, r[k])
    var result = F.zero();
    const num_vars = r_cycle.len;

    for (0..num_vars) |i| {
        const ji = (j >> @intCast(i)) & 1;
        if (ji == 0) {
            // This bit position contributes
            var contrib = r_cycle[i];

            // Multiply by eq for higher bits
            for ((i + 1)..num_vars) |k| {
                const jk = (j >> @intCast(k)) & 1;
                const rk = r_cycle[k];
                if (jk == 1) {
                    contrib = contrib.mul(rk);
                } else {
                    contrib = contrib.mul(F.one().sub(rk));
                }
            }

            result = result.add(contrib);
        }
    }

    return result;
}

// ============================================================================
// Tests
// ============================================================================

test "registers val evaluation params" {
    const allocator = std.testing.allocator;
    const field = @import("zolt_arith").field;
    const F = field.BN254Scalar;

    const r_address = [_]F{ F.one(), F.zero(), F.one() };
    const r_cycle = [_]F{ F.zero(), F.one(), F.one() };

    var params = try RegistersValEvaluationParams(F).init(
        allocator,
        &r_address,
        &r_cycle,
        F.fromU64(42),
    );
    defer params.deinit();

    try std.testing.expectEqual(@as(usize, 3), params.num_rounds);
    try std.testing.expect(params.input_claim.eql(F.fromU64(42)));
}

test "computeLtAtIndex basic" {
    const field = @import("zolt_arith").field;
    const F = field.BN254Scalar;

    // r_cycle = [1, 0] represents cycle 1 in binary: bit0=1, bit1=0
    const r_cycle = [_]F{ F.one(), F.zero() };

    // LT(0, 1) should be non-zero (0 < 1)
    const lt_0 = computeLtAtIndex(F, &r_cycle, 0);
    try std.testing.expect(!lt_0.eql(F.zero()));

    // LT(1, 1) should be zero (1 is not < 1)
    const lt_1 = computeLtAtIndex(F, &r_cycle, 1);
    try std.testing.expect(lt_1.eql(F.zero()));

    // LT(2, 1) should be zero (2 is not < 1)
    const lt_2 = computeLtAtIndex(F, &r_cycle, 2);
    try std.testing.expect(lt_2.eql(F.zero()));
}
