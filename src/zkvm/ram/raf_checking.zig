//! RAM RAF (Read-After-Final) Checking
//!
//! This module implements the RAF (Read-After-Final) checking protocol for
//! memory consistency verification. RAF checking ensures that:
//!
//! 1. Every memory read returns the value from the most recent write to that address
//! 2. The final memory state is consistent with all writes
//!
//! The protocol uses a sumcheck argument to prove the relation:
//!   Σ_{k=0}^{K-1} ra(k) ⋅ unmap(k) = raf_claim
//!
//! Where:
//! - ra(k) = Σ_j eq(r_cycle, j) ⋅ 1[address(j) = k] aggregates access counts per address k
//! - unmap(k) converts remapped address k back to original address
//! - raf_claim is from the outer Spartan sumcheck
//!
//! Reference: jolt-core/src/zkvm/ram/raf_evaluation.rs

const std = @import("std");
const Allocator = std.mem.Allocator;

const mod = @import("mod.zig");
const MemoryOp = mod.MemoryOp;
const MemoryAccess = mod.MemoryAccess;
const MemoryTrace = mod.MemoryTrace;

/// Parameters for RAF evaluation sumcheck
pub fn RafEvaluationParams(comptime F: type) type {
    return struct {
        const Self = @This();

        /// log2 of the number of memory addresses (K)
        log_k: usize,
        /// Start address for unmapping
        start_address: u64,
        /// Random challenge for cycle binding (r_cycle)
        r_cycle: []const F,
        /// Allocator for dynamic memory
        allocator: Allocator,

        pub fn init(
            allocator: Allocator,
            log_k: usize,
            start_address: u64,
            r_cycle: []const F,
        ) !Self {
            const r_cycle_copy = try allocator.alloc(F, r_cycle.len);
            @memcpy(r_cycle_copy, r_cycle);

            return Self{
                .log_k = log_k,
                .start_address = start_address,
                .r_cycle = r_cycle_copy,
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.r_cycle);
        }

        /// Get the number of sumcheck rounds
        pub fn numRounds(self: *const Self) usize {
            return self.log_k;
        }

        /// Get the degree bound (product of two linear polynomials)
        pub fn degreeBound() usize {
            return 2;
        }
    };
}

/// Memory address counts per address slot
/// ra(k) = Σ_j eq(r_cycle, j) ⋅ 1[address(j) = k]
pub fn RaPolynomial(comptime F: type) type {
    return struct {
        const Self = @This();

        /// Evaluations ra[k] for k in [0, 2^log_k)
        evals: []F,
        /// Number of variables
        num_vars: usize,
        allocator: Allocator,

        /// Initialize from a memory trace
        /// Computes ra(k) = Σ_j eq(r_cycle, j) ⋅ 1[address(j) = k]
        pub fn fromTrace(
            allocator: Allocator,
            trace: *const MemoryTrace,
            r_cycle: []const F,
            start_address: u64,
            log_k: usize,
        ) !Self {
            const k_size: usize = @as(usize, 1) << @intCast(log_k);
            const evals = try allocator.alloc(F, k_size);
            for (evals) |*e| {
                e.* = F.zero();
            }

            // Pre-compute eq(r_cycle, j) for all cycles j in the trace
            const trace_len = trace.accesses.items.len;
            const log_t = if (trace_len == 0) 0 else std.math.log2_int_ceil(usize, trace_len);
            const eq_evals = try computeEqEvals(allocator, F, r_cycle, log_t);
            defer allocator.free(eq_evals);

            // Accumulate ra(k) for each access
            for (trace.accesses.items, 0..) |access, j| {
                // Remap address to slot k
                if (access.address >= start_address) {
                    const k = (access.address - start_address) / 8;
                    if (k < k_size) {
                        // Add eq(r_cycle, j) to ra(k)
                        const eq_val = if (j < eq_evals.len) eq_evals[j] else F.zero();
                        evals[@intCast(k)] = evals[@intCast(k)].add(eq_val);
                    }
                }
            }

            return Self{
                .evals = evals,
                .num_vars = log_k,
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.evals);
        }

        /// Evaluate the polynomial at point r
        pub fn evaluate(self: *const Self, r: []const F) F {
            if (r.len != self.num_vars) {
                return F.zero();
            }

            // Multilinear evaluation: Σ_k ra(k) ⋅ eq(r, k)
            var result = F.zero();
            for (self.evals, 0..) |eval, k| {
                const eq_val = computeEqAtPoint(F, r, k);
                result = result.add(eval.mul(eq_val));
            }
            return result;
        }

        /// Get evaluation at index k
        pub fn get(self: *const Self, k: usize) F {
            if (k >= self.evals.len) return F.zero();
            return self.evals[k];
        }

        /// Bind the first variable to value r
        /// Reduces from n variables to n-1 variables
        pub fn bind(self: *Self, r: F) void {
            const half = self.evals.len / 2;
            for (0..half) |i| {
                // Interpolate: (1-r) * evals[i] + r * evals[i + half]
                const lo = self.evals[i];
                const hi = self.evals[i + half];
                const one_minus_r = F.one().sub(r);
                self.evals[i] = one_minus_r.mul(lo).add(r.mul(hi));
            }
            self.num_vars -= 1;
        }

        /// Get final claim after all bindings
        pub fn finalClaim(self: *const Self) F {
            if (self.evals.len == 0) return F.zero();
            return self.evals[0];
        }
    };
}

/// Unmap polynomial: converts remapped address k to original address
pub fn UnmapPolynomial(comptime F: type) type {
    return struct {
        const Self = @This();

        /// Number of variables (log2 of address space)
        num_vars: usize,
        /// Start address for unmapping
        start_address: u64,

        pub fn init(num_vars: usize, start_address: u64) Self {
            return Self{
                .num_vars = num_vars,
                .start_address = start_address,
            };
        }

        /// Evaluate unmap at index k: returns original address = start_address + k * 8
        pub fn evaluateAtIndex(self: *const Self, k: usize) u64 {
            return self.start_address + @as(u64, @intCast(k)) * 8;
        }

        /// Evaluate the unmap polynomial at point r
        /// unmap(r) = Σ_k eq(r, k) ⋅ (start_address + k * 8)
        pub fn evaluate(self: *const Self, r: []const F) F {
            if (r.len != self.num_vars) {
                return F.zero();
            }

            // This is an identity polynomial shifted by start_address
            // unmap(r) = start_address + 8 * Σ_i r[i] * 2^i
            var result = F.fromU64(self.start_address);
            var power: u64 = 8; // Start with 8 for word alignment

            for (r) |ri| {
                // Add r[i] * 2^(i+3) (multiply by 8 for word size)
                result = result.add(ri.mul(F.fromU64(power)));
                power *= 2;
            }

            return result;
        }

        /// Get sumcheck evaluations at index i for binding
        pub fn sumcheckEvals(self: *const Self, i: usize, bound_values: []const F) [3]F {
            // The unmap polynomial is linear in each variable
            // At index i, we compute evaluations at x=0, x=1, x=2

            // Base contribution from already-bound variables
            var base = F.fromU64(self.start_address);
            var current_power: u64 = 8;
            for (bound_values) |v| {
                base = base.add(v.mul(F.fromU64(current_power)));
                current_power *= 2;
            }

            // Power for variable i
            const power_i = current_power << @intCast(i);

            // Evaluations at x=0, x=1, x=2
            return [3]F{
                base, // x=0
                base.add(F.fromU64(power_i)), // x=1
                base.add(F.fromU64(power_i * 2)), // x=2
            };
        }
    };
}

/// RAF Evaluation Sumcheck Prover
pub fn RafEvaluationProver(comptime F: type) type {
    return struct {
        const Self = @This();

        /// Ra polynomial (access counts)
        ra: RaPolynomial(F),
        /// Unmap polynomial
        unmap: UnmapPolynomial(F),
        /// Parameters
        params: RafEvaluationParams(F),
        /// Current round
        round: usize,
        /// Bound values from previous rounds
        bound_values: std.ArrayListUnmanaged(F),
        allocator: Allocator,

        pub fn init(
            allocator: Allocator,
            trace: *const MemoryTrace,
            params: RafEvaluationParams(F),
        ) !Self {
            const ra = try RaPolynomial(F).fromTrace(
                allocator,
                trace,
                params.r_cycle,
                params.start_address,
                params.log_k,
            );

            const unmap = UnmapPolynomial(F).init(params.log_k, params.start_address);

            return Self{
                .ra = ra,
                .unmap = unmap,
                .params = params,
                .round = 0,
                .bound_values = .{},
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            self.ra.deinit();
            self.bound_values.deinit(self.allocator);
        }

        /// Compute the initial claim: Σ_k ra(k) ⋅ unmap(k)
        pub fn computeInitialClaim(self: *const Self) F {
            var claim = F.zero();
            for (0..self.ra.evals.len) |k| {
                const ra_k = self.ra.get(k);
                const unmap_k = F.fromU64(self.unmap.evaluateAtIndex(k));
                claim = claim.add(ra_k.mul(unmap_k));
            }
            return claim;
        }

        /// Compute the round polynomial for the current round
        /// Returns [p(0), p(1)] where p(x) = Σ_{k: k[round]=x} ra(k) ⋅ unmap(k)
        pub fn computeRoundPolynomial(self: *Self) [2]F {
            const half = self.ra.evals.len / 2;
            var evals: [2]F = .{ F.zero(), F.zero() };

            for (0..half) |i| {
                // Index with bit[round] = 0
                const ra_lo = self.ra.evals[i];
                // Index with bit[round] = 1
                const ra_hi = self.ra.evals[i + half];

                // Unmap evaluations
                const unmap_lo = F.fromU64(self.unmap.evaluateAtIndex(i));
                const unmap_hi = F.fromU64(self.unmap.evaluateAtIndex(i + half));

                evals[0] = evals[0].add(ra_lo.mul(unmap_lo));
                evals[1] = evals[1].add(ra_hi.mul(unmap_hi));
            }

            return evals;
        }

        /// Process a challenge and bind the current variable
        pub fn bindChallenge(self: *Self, challenge: F) !void {
            self.ra.bind(challenge);
            try self.bound_values.append(self.allocator, challenge);
            self.round += 1;
        }

        /// Get the final evaluation claim
        pub fn getFinalClaim(self: *const Self) F {
            return self.ra.finalClaim();
        }

        /// Check if all rounds are complete
        pub fn isComplete(self: *const Self) bool {
            return self.round >= self.params.log_k;
        }
    };
}

/// RAF Evaluation Sumcheck Verifier
pub fn RafEvaluationVerifier(comptime F: type) type {
    return struct {
        const Self = @This();

        params: RafEvaluationParams(F),
        /// Current claim
        current_claim: F,
        /// Bound challenges
        challenges: std.ArrayListUnmanaged(F),
        /// Current round
        round: usize,
        allocator: Allocator,

        pub fn init(allocator: Allocator, params: RafEvaluationParams(F), initial_claim: F) Self {
            return Self{
                .params = params,
                .current_claim = initial_claim,
                .challenges = .{},
                .round = 0,
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            self.challenges.deinit(self.allocator);
        }

        /// Verify a round polynomial and update state
        /// round_poly contains [p(0), p(1)]
        /// Returns the challenge to use for this round
        pub fn verifyRound(self: *Self, round_poly: [2]F, transcript: anytype) !F {
            // Check: p(0) + p(1) = current_claim
            const sum = round_poly[0].add(round_poly[1]);
            if (!sum.eql(self.current_claim)) {
                return error.SumcheckVerificationFailed;
            }

            // Get challenge from transcript
            const challenge = transcript.challengeScalar("raf_challenge");

            // Update claim: p(challenge) = (1-challenge) * p(0) + challenge * p(1)
            const one_minus_r = F.one().sub(challenge);
            self.current_claim = one_minus_r.mul(round_poly[0]).add(challenge.mul(round_poly[1]));

            try self.challenges.append(self.allocator, challenge);
            self.round += 1;

            return challenge;
        }

        /// Get the final claim after all rounds
        pub fn getFinalClaim(self: *const Self) F {
            return self.current_claim;
        }

        /// Verify the final claim against oracle evaluations
        /// ra_eval: evaluation of ra polynomial at challenges
        /// unmap_eval: evaluation of unmap polynomial at challenges
        pub fn verifyFinalClaim(self: *const Self, ra_eval: F, unmap_eval: F) bool {
            const expected = ra_eval.mul(unmap_eval);
            return expected.eql(self.current_claim);
        }

        /// Get all challenges for opening proof
        pub fn getChallenges(self: *const Self) []const F {
            return self.challenges.items;
        }
    };
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Compute eq(r, j) for all j in [0, 2^n)
fn computeEqEvals(allocator: Allocator, comptime F: type, r: []const F, n: usize) ![]F {
    if (n == 0) {
        const result = try allocator.alloc(F, 1);
        result[0] = F.one();
        return result;
    }

    const size: usize = @as(usize, 1) << @intCast(n);
    const result = try allocator.alloc(F, size);

    // Start with eq(r[0..0], 0) = 1
    result[0] = F.one();
    var current_size: usize = 1;

    // Build up using: eq(r[0..i+1], j) = eq(r[0..i], j[0..i]) * eq_i(r[i], j[i])
    for (0..n) |i| {
        const ri = if (i < r.len) r[i] else F.zero();
        const one_minus_ri = F.one().sub(ri);

        // Process in reverse to avoid overwriting
        var j = current_size;
        while (j > 0) {
            j -= 1;
            // j with bit i = 0
            result[j] = result[j].mul(one_minus_ri);
            // j with bit i = 1
            result[j + current_size] = result[j].mul(ri).mul(one_minus_ri.inverse());
        }
        current_size *= 2;
    }

    return result;
}

/// Compute eq(r, k) for a specific index k
fn computeEqAtPoint(comptime F: type, r: []const F, k: usize) F {
    var result = F.one();
    for (r, 0..) |ri, i| {
        const ki = (k >> @intCast(i)) & 1;
        if (ki == 1) {
            result = result.mul(ri);
        } else {
            result = result.mul(F.one().sub(ri));
        }
    }
    return result;
}

// ============================================================================
// Tests
// ============================================================================

test "ra polynomial from empty trace" {
    const allocator = std.testing.allocator;
    const field = @import("../../field/mod.zig");
    const F = field.BN254Scalar;

    var trace = MemoryTrace.init(allocator);
    defer trace.deinit();

    const r_cycle = [_]F{ F.one(), F.zero() };
    var ra = try RaPolynomial(F).fromTrace(
        allocator,
        &trace,
        &r_cycle,
        0x80000000, // start address
        4, // log_k = 4 -> 16 slots
    );
    defer ra.deinit();

    // Empty trace should give all zeros
    for (ra.evals) |e| {
        try std.testing.expect(e.eql(F.zero()));
    }
}

test "ra polynomial from trace with single access" {
    const allocator = std.testing.allocator;
    const field = @import("../../field/mod.zig");
    const F = field.BN254Scalar;

    var trace = MemoryTrace.init(allocator);
    defer trace.deinit();

    // Single write at address start + 8 (slot 1)
    try trace.recordWrite(0x80000008, 42, 0);

    const r_cycle = [_]F{F.one()};
    var ra = try RaPolynomial(F).fromTrace(
        allocator,
        &trace,
        &r_cycle,
        0x80000000,
        4,
    );
    defer ra.deinit();

    // Slot 1 should have non-zero value
    try std.testing.expect(!ra.get(1).eql(F.zero()));
    // Other slots should be zero
    try std.testing.expect(ra.get(0).eql(F.zero()));
    try std.testing.expect(ra.get(2).eql(F.zero()));
}

test "unmap polynomial evaluation" {
    const field = @import("../../field/mod.zig");
    const F = field.BN254Scalar;

    const unmap = UnmapPolynomial(F).init(4, 0x80000000);

    // unmap(0) = start_address
    try std.testing.expectEqual(@as(u64, 0x80000000), unmap.evaluateAtIndex(0));

    // unmap(1) = start_address + 8
    try std.testing.expectEqual(@as(u64, 0x80000008), unmap.evaluateAtIndex(1));

    // unmap(10) = start_address + 80
    try std.testing.expectEqual(@as(u64, 0x80000050), unmap.evaluateAtIndex(10));
}

test "raf evaluation prover init" {
    const allocator = std.testing.allocator;
    const field = @import("../../field/mod.zig");
    const F = field.BN254Scalar;

    var trace = MemoryTrace.init(allocator);
    defer trace.deinit();

    try trace.recordWrite(0x80000000, 1, 0);
    try trace.recordRead(0x80000000, 1, 1);

    const r_cycle = [_]F{ F.fromU64(2), F.fromU64(3) };
    var params = try RafEvaluationParams(F).init(
        allocator,
        4,
        0x80000000,
        &r_cycle,
    );
    defer params.deinit();

    var prover = try RafEvaluationProver(F).init(allocator, &trace, params);
    defer prover.deinit();

    // Initial claim should be computable
    const claim = prover.computeInitialClaim();
    _ = claim; // Just verify it doesn't crash

    try std.testing.expect(!prover.isComplete());
}

test "eq polynomial computation" {
    const allocator = std.testing.allocator;
    const field = @import("../../field/mod.zig");
    const F = field.BN254Scalar;

    // Test with n=2: eq(r, j) for j in [0, 4)
    const r = [_]F{ F.fromU64(2), F.fromU64(3) };
    const evals = try computeEqEvals(allocator, F, &r, 2);
    defer allocator.free(evals);

    try std.testing.expectEqual(@as(usize, 4), evals.len);

    // Verify: eq(r, 0) = (1-r0)(1-r1) = (1-2)(1-3) = (-1)(-2) = 2
    const expected_0 = F.one().sub(F.fromU64(2)).mul(F.one().sub(F.fromU64(3)));
    try std.testing.expect(evals[0].eql(expected_0));
}
