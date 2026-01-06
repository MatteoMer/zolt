//! Batched Sumcheck Protocol for Stage 2
//!
//! This module implements the batched sumcheck protocol that combines multiple
//! sumcheck instances into a single proof. This is used in Stage 2 to batch:
//!
//! 1. ProductVirtualRemainder - n_cycle_vars rounds, degree 3
//! 2. RamRafEvaluation - log_ram_k rounds, degree 2
//! 3. RamReadWriteChecking - log_ram_k + n_cycle_vars rounds, degree 3
//! 4. OutputSumcheck - log_ram_k rounds, degree 3
//! 5. InstructionLookupsClaimReduction - n_cycle_vars rounds, degree 2
//!
//! ## Batching Protocol
//!
//! 1. Each instance provides its input_claim
//! 2. All input_claims are appended to transcript
//! 3. Batching coefficients α₀, α₁, ... are sampled from transcript
//! 4. Claims are scaled: scaled_claim[i] = claim[i] * 2^(max_rounds - rounds[i])
//! 5. Combined: batched_claim = Σᵢ αᵢ * scaled_claim[i]
//! 6. Run max_rounds rounds, combining polynomials: h(z) = Σᵢ αᵢ * hᵢ(z)
//!
//! Reference: jolt-core/src/subprotocols/sumcheck.rs (BatchedSumcheck)

const std = @import("std");
const Allocator = std.mem.Allocator;

const poly_mod = @import("../poly/mod.zig");
const UniPoly = poly_mod.UniPoly;
const transcripts = @import("../transcripts/mod.zig");
const Blake2bTranscript = transcripts.Blake2bTranscript;

/// Sumcheck instance interface
///
/// Each batched instance must implement this interface
pub fn SumcheckInstance(comptime F: type) type {
    return struct {
        const Self = @This();

        /// Number of rounds for this instance
        num_rounds: usize,
        /// Degree bound for round polynomials
        degree: usize,
        /// Input claim for this instance
        input_claim: F,

        /// Compute the round polynomial for the given round
        /// Returns evaluations [s(0), s(1), s(2), ...] up to degree+1 points
        computeRoundPolyFn: *const fn (ctx: *anyopaque, round: usize) anyerror![4]F,

        /// Bind the challenge and advance to next round
        bindChallengeFn: *const fn (ctx: *anyopaque, challenge: F) anyerror!void,

        /// Cache opening claims after all rounds complete
        /// r_sumcheck: The final evaluation point (all challenges)
        cacheOpeningsFn: *const fn (ctx: *anyopaque, r_sumcheck: []const F) anyerror!void,

        /// Context pointer for the actual instance
        ctx: *anyopaque,

        /// Compute round polynomial
        pub fn computeRoundPoly(self: *Self, round: usize) ![4]F {
            return self.computeRoundPolyFn(self.ctx, round);
        }

        /// Bind challenge
        pub fn bindChallenge(self: *Self, challenge: F) !void {
            return self.bindChallengeFn(self.ctx, challenge);
        }

        /// Cache openings
        pub fn cacheOpenings(self: *Self, r_sumcheck: []const F) !void {
            return self.cacheOpeningsFn(self.ctx, r_sumcheck);
        }
    };
}

/// Batched sumcheck prover
pub fn BatchedSumcheckProver(comptime F: type) type {
    return struct {
        const Self = @This();

        /// The sumcheck instances being batched
        instances: std.ArrayListUnmanaged(SumcheckInstance(F)),
        /// Batching coefficients (one per instance)
        batching_coeffs: std.ArrayListUnmanaged(F),
        /// Maximum number of rounds across all instances
        max_num_rounds: usize,
        /// Current round
        current_round: usize,
        /// Current batched claim
        current_claim: F,
        /// Accumulated challenges
        challenges: std.ArrayListUnmanaged(F),
        /// Allocator
        allocator: Allocator,

        /// Initialize the batched sumcheck prover
        pub fn init(allocator: Allocator) Self {
            return Self{
                .instances = .{},
                .batching_coeffs = .{},
                .max_num_rounds = 0,
                .current_round = 0,
                .current_claim = F.zero(),
                .challenges = .{},
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            self.instances.deinit(self.allocator);
            self.batching_coeffs.deinit(self.allocator);
            self.challenges.deinit(self.allocator);
        }

        /// Add a sumcheck instance to the batch
        pub fn addInstance(self: *Self, instance: SumcheckInstance(F)) !void {
            try self.instances.append(self.allocator, instance);
            if (instance.num_rounds > self.max_num_rounds) {
                self.max_num_rounds = instance.num_rounds;
            }
        }

        /// Setup the batching by sampling coefficients and computing initial claim
        ///
        /// This must be called after all instances are added and before generating proofs.
        /// The transcript should have already received all input_claims.
        pub fn setupBatching(self: *Self, transcript: *Blake2bTranscript(F)) !void {
            // Append all input claims to transcript
            for (self.instances.items) |instance| {
                transcript.appendScalar(instance.input_claim);
            }

            // Sample batching coefficients
            for (0..self.instances.items.len) |_| {
                const coeff = transcript.challengeScalarFull();
                try self.batching_coeffs.append(self.allocator, coeff);
            }

            // Compute initial batched claim
            // scaled_claim[i] = claim[i] * 2^(max_rounds - rounds[i])
            // batched_claim = Σᵢ coeff[i] * scaled_claim[i]
            var batched_claim = F.zero();
            for (self.instances.items, 0..) |instance, i| {
                const scale_power = self.max_num_rounds - instance.num_rounds;
                var scaled_claim = instance.input_claim;
                // Multiply by 2^scale_power
                for (0..scale_power) |_| {
                    scaled_claim = scaled_claim.add(scaled_claim);
                }
                batched_claim = batched_claim.add(scaled_claim.mul(self.batching_coeffs.items[i]));
            }
            self.current_claim = batched_claim;
        }

        /// Compute the combined round polynomial for the current round
        ///
        /// Returns compressed coefficients [c0, c2, c3] for degree-3 polynomial
        pub fn computeRoundPolynomial(self: *Self) ![3]F {
            var combined_evals = [4]F{ F.zero(), F.zero(), F.zero(), F.zero() };

            for (self.instances.items, 0..) |*instance, i| {
                // Check if this instance has started (its rounds begin when max_rounds - num_rounds have passed)
                const start_round = self.max_num_rounds - instance.num_rounds;

                if (self.current_round >= start_round) {
                    // This instance is active
                    const instance_round = self.current_round - start_round;
                    const evals = try instance.computeRoundPoly(instance_round);

                    // Scale by batching coefficient
                    const coeff = self.batching_coeffs.items[i];
                    for (0..4) |j| {
                        combined_evals[j] = combined_evals[j].add(evals[j].mul(coeff));
                    }
                } else {
                    // Instance hasn't started yet - contribute constant polynomial = scaled_input_claim
                    // For a constant polynomial, s(X) = c for all X
                    const scale_power = self.max_num_rounds - instance.num_rounds - self.current_round;
                    var scaled = instance.input_claim;
                    for (0..scale_power) |_| {
                        scaled = scaled.add(scaled);
                    }
                    const weighted = scaled.mul(self.batching_coeffs.items[i]);
                    // Constant polynomial: s(0) = s(1) = s(2) = s(3) = c
                    for (0..4) |j| {
                        combined_evals[j] = combined_evals[j].add(weighted);
                    }
                }
            }

            // Convert to compressed coefficients
            return UniPoly(F).evalsToCompressed(combined_evals);
        }

        /// Bind the challenge for this round
        pub fn bindChallenge(self: *Self, challenge: F) !void {
            try self.challenges.append(self.allocator, challenge);

            // Bind challenge in all active instances
            for (self.instances.items) |*instance| {
                const start_round = self.max_num_rounds - instance.num_rounds;
                if (self.current_round >= start_round) {
                    try instance.bindChallenge(challenge);
                }
            }

            self.current_round += 1;
        }

        /// Update claim after receiving challenge
        pub fn updateClaim(self: *Self, round_evals: [4]F, challenge: F) void {
            // Evaluate combined polynomial at challenge using Lagrange interpolation
            self.current_claim = evaluateCubicAtPointGeneric(F, round_evals, challenge);
        }

        /// Cache openings for all instances after sumcheck completes
        pub fn cacheOpenings(self: *Self) !void {
            const r_sumcheck = self.challenges.items;

            for (self.instances.items) |*instance| {
                // Each instance gets the suffix of challenges corresponding to its rounds
                const start_idx = self.max_num_rounds - instance.num_rounds;
                const instance_challenges = r_sumcheck[start_idx..];
                try instance.cacheOpenings(instance_challenges);
            }
        }

        /// Get all accumulated challenges
        pub fn getChallenges(self: *const Self) []const F {
            return self.challenges.items;
        }

        /// Get the final claim after all rounds
        pub fn getFinalClaim(self: *const Self) F {
            return self.current_claim;
        }

        /// Number of total rounds
        pub fn numRounds(self: *const Self) usize {
            return self.max_num_rounds;
        }
    };
}

/// Result of batched sumcheck proof generation
pub fn BatchedSumcheckProof(comptime F: type) type {
    return struct {
        /// Compressed round polynomials [c0, c2, c3] for each round
        round_polys: std.ArrayListUnmanaged([3]F),
        /// Challenges for each round
        challenges: std.ArrayListUnmanaged(F),
        /// Final claim after all rounds
        final_claim: F,
        /// Allocator
        allocator: Allocator,

        pub fn init(allocator: Allocator) @This() {
            return .{
                .round_polys = .{},
                .challenges = .{},
                .final_claim = F.zero(),
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *@This()) void {
            self.round_polys.deinit(self.allocator);
            self.challenges.deinit(self.allocator);
        }
    };
}

/// Generate a complete batched sumcheck proof
pub fn generateBatchedProof(
    comptime F: type,
    prover: *BatchedSumcheckProver(F),
    transcript: *Blake2bTranscript(F),
) !BatchedSumcheckProof(F) {
    var proof = BatchedSumcheckProof(F).init(prover.allocator);

    const num_rounds = prover.numRounds();

    for (0..num_rounds) |round_idx| {
        // Compute round polynomial
        const round_evals = try prover.computeRoundPolynomial();

        // Store in proof
        try proof.round_polys.append(prover.allocator, round_evals);

        // Append to transcript (matching Jolt's UniPoly format)
        transcript.appendMessage("UniPoly_begin");
        transcript.appendScalar(round_evals[0]); // c0
        transcript.appendScalar(round_evals[1]); // c2
        transcript.appendScalar(round_evals[2]); // c3
        transcript.appendMessage("UniPoly_end");

        // Get challenge from transcript
        const challenge = transcript.challengeScalar();
        try proof.challenges.append(prover.allocator, challenge);

        // Compute full evaluations for claim update
        // Need to recover s(0), s(1), s(2), s(3) from compressed [c0, c2, c3]
        // s(0) = c0
        // s(0) + s(1) = current_claim => s(1) = current_claim - c0
        // Use Lagrange to get s(2), s(3) from the cubic
        const s0 = round_evals[0];
        const s1 = prover.current_claim.sub(s0);

        // For cubic interpolation, we need to reconstruct s(2), s(3)
        // The compressed form stores [c0, c2, c3] where the polynomial is:
        // s(X) = c0 + c1*X + c2*X^2 + c3*X^3
        // We have c0 = round_evals[0], and we stored c2, c3 in positions 1, 2
        // c1 can be recovered from s(0) + s(1) = current_claim

        // Full evaluations for updateClaim
        const full_evals = [4]F{ s0, s1, round_evals[1], round_evals[2] };

        // Update prover state
        prover.updateClaim(full_evals, challenge);
        try prover.bindChallenge(challenge);

        _ = round_idx;
    }

    proof.final_claim = prover.getFinalClaim();

    // Cache openings in all instances
    try prover.cacheOpenings();

    return proof;
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Evaluate cubic polynomial at a point using Lagrange interpolation
fn evaluateCubicAtPointGeneric(comptime F: type, evals: [4]F, x: F) F {
    // Lagrange interpolation at points 0, 1, 2, 3
    const x_minus_0 = x;
    const x_minus_1 = x.sub(F.one());
    const x_minus_2 = x.sub(F.fromU64(2));
    const x_minus_3 = x.sub(F.fromU64(3));

    // L_0(x) = (x-1)(x-2)(x-3) / (-6)
    const L0 = x_minus_1.mul(x_minus_2).mul(x_minus_3).mul(F.fromU64(6).neg().inverse().?);
    // L_1(x) = x(x-2)(x-3) / 2
    const L1 = x_minus_0.mul(x_minus_2).mul(x_minus_3).mul(F.fromU64(2).inverse().?);
    // L_2(x) = x(x-1)(x-3) / (-2)
    const L2 = x_minus_0.mul(x_minus_1).mul(x_minus_3).mul(F.fromU64(2).neg().inverse().?);
    // L_3(x) = x(x-1)(x-2) / 6
    const L3 = x_minus_0.mul(x_minus_1).mul(x_minus_2).mul(F.fromU64(6).inverse().?);

    return evals[0].mul(L0)
        .add(evals[1].mul(L1))
        .add(evals[2].mul(L2))
        .add(evals[3].mul(L3));
}

// ============================================================================
// Tests
// ============================================================================

const testing = std.testing;
const BN254Scalar = @import("../field/mod.zig").BN254Scalar;

test "batched sumcheck: initialization" {
    const F = BN254Scalar;
    var prover = BatchedSumcheckProver(F).init(testing.allocator);
    defer prover.deinit();

    try testing.expectEqual(@as(usize, 0), prover.instances.items.len);
    try testing.expectEqual(@as(usize, 0), prover.max_num_rounds);
}

test "batched sumcheck: cubic evaluation" {
    const F = BN254Scalar;

    // Test polynomial s(X) = 1 + 2X + 3X^2 + 4X^3
    // s(0) = 1, s(1) = 10, s(2) = 49, s(3) = 142
    const evals = [4]F{
        F.fromU64(1),
        F.fromU64(10),
        F.fromU64(49),
        F.fromU64(142),
    };

    // Evaluate at X = 2
    const result = evaluateCubicAtPointGeneric(F, evals, F.fromU64(2));
    try testing.expect(result.eql(F.fromU64(49)));

    // Evaluate at X = 0
    const result0 = evaluateCubicAtPointGeneric(F, evals, F.zero());
    try testing.expect(result0.eql(F.fromU64(1)));
}
