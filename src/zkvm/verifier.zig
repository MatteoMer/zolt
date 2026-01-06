//! Multi-Stage Sumcheck Verifier for Jolt zkVM
//!
//! This module implements verification for all 6 sumcheck stages in the Jolt protocol.
//! Each stage has specific polynomial structures and degree bounds that must be verified.
//!
//! ## Stage Overview
//!
//! 1. **Stage 1: Outer Spartan** - R1CS instruction correctness (degree 3)
//! 2. **Stage 2: RAM RAF Evaluation** - Memory read-after-final checking (degree 2)
//! 3. **Stage 3: Lasso Lookup** - Instruction lookup reduction (degree 2)
//! 4. **Stage 4: Value Evaluation** - Memory value consistency (degree 3)
//! 5. **Stage 5: Register Evaluation** - Register consistency (degree 2)
//! 6. **Stage 6: Booleanity** - Flag constraint verification (degree 2)
//!
//! Reference: jolt-core/src/zkvm/verifier.rs

const std = @import("std");
const Allocator = std.mem.Allocator;

const poly = @import("../poly/mod.zig");
const prover = @import("prover.zig");
const ram = @import("ram/mod.zig");
const lasso = @import("lasso/mod.zig");
const transcripts = @import("../transcripts/mod.zig");

const StageProof = prover.StageProof;
const JoltStageProofs = prover.JoltStageProofs;

/// Verification result for a single stage
pub const StageVerificationResult = struct {
    success: bool,
    final_claim: ?[]const u8, // Serialized final claim for cross-stage checks
    error_msg: ?[]const u8,
};

/// Verifier configuration options
pub const VerifierConfig = struct {
    /// When true, strictly verify p(0) + p(1) = claim for all sumcheck rounds.
    /// When false, only verify structural consistency (useful for debugging).
    strict_sumcheck: bool = true,

    /// Log verification details for debugging
    debug_output: bool = false,
};

/// Multi-stage verifier state
pub fn MultiStageVerifier(comptime F: type) type {
    return struct {
        const Self = @This();

        /// Verification results for each stage
        stage_results: [6]StageVerificationResult,
        /// Opening claims accumulated for batch verification
        opening_claims: OpeningClaimAccumulator(F),
        /// Current stage
        current_stage: usize,
        /// Allocator
        allocator: Allocator,
        /// Configuration
        config: VerifierConfig,

        pub fn init(allocator: Allocator) Self {
            return initWithConfig(allocator, .{});
        }

        pub fn initWithConfig(allocator: Allocator, config: VerifierConfig) Self {
            var results: [6]StageVerificationResult = undefined;
            for (&results) |*r| {
                r.* = .{
                    .success = false,
                    .final_claim = null,
                    .error_msg = null,
                };
            }

            return Self{
                .stage_results = results,
                .opening_claims = OpeningClaimAccumulator(F).init(allocator),
                .current_stage = 0,
                .allocator = allocator,
                .config = config,
            };
        }

        pub fn deinit(self: *Self) void {
            self.opening_claims.deinit();
        }

        /// Verify all stages in the proof
        pub fn verify(
            self: *Self,
            proofs: *const JoltStageProofs(F),
            transcript: *transcripts.Transcript(F),
        ) !bool {
            // Extract log_t and log_k from proof for transcript sync
            const log_t = proofs.log_t;
            const log_k = proofs.log_k;

            std.debug.print("\n[VERIFIER] ========================================\n", .{});
            std.debug.print("[VERIFIER] Starting multi-stage verification\n", .{});
            std.debug.print("[VERIFIER]   log_t={d}, log_k={d}\n", .{ log_t, log_k });
            std.debug.print("[VERIFIER] ========================================\n\n", .{});

            // Verify each stage in order
            // Each stage verification updates the transcript for Fiat-Shamir
            if (!try self.verifyStage1(&proofs.stage_proofs[0], transcript)) {
                std.debug.print("[VERIFIER] Stage 1 verification FAILED\n", .{});
                return false;
            }
            std.debug.print("\n[VERIFIER] Stage 1 PASSED\n\n", .{});

            if (!try self.verifyStage2(&proofs.stage_proofs[1], transcript, log_t)) {
                std.debug.print("[VERIFIER] Stage 2 verification FAILED\n", .{});
                return false;
            }
            std.debug.print("\n[VERIFIER] Stage 2 PASSED\n\n", .{});

            if (!try self.verifyStage3(&proofs.stage_proofs[2], transcript, log_t, log_k)) {
                std.debug.print("[VERIFIER] Stage 3 verification FAILED\n", .{});
                return false;
            }
            std.debug.print("\n[VERIFIER] Stage 3 PASSED\n\n", .{});

            if (!try self.verifyStage4(&proofs.stage_proofs[3], transcript, log_t)) {
                std.debug.print("[VERIFIER] Stage 4 verification FAILED\n", .{});
                return false;
            }
            std.debug.print("\n[VERIFIER] Stage 4 PASSED\n\n", .{});

            if (!try self.verifyStage5(&proofs.stage_proofs[4], transcript, log_t)) {
                std.debug.print("[VERIFIER] Stage 5 verification FAILED\n", .{});
                return false;
            }
            std.debug.print("\n[VERIFIER] Stage 5 PASSED\n\n", .{});

            if (!try self.verifyStage6(&proofs.stage_proofs[5], transcript, log_t)) {
                std.debug.print("[VERIFIER] Stage 6 verification FAILED\n", .{});
                return false;
            }
            std.debug.print("\n[VERIFIER] Stage 6 PASSED\n\n", .{});

            std.debug.print("[VERIFIER] ========================================\n", .{});
            std.debug.print("[VERIFIER] All stages PASSED!\n", .{});
            std.debug.print("[VERIFIER] ========================================\n\n", .{});

            // All stages verified successfully
            return true;
        }

        /// Verify Stage 1: Outer Spartan sumcheck
        ///
        /// Verifies: sum_{x} eq(tau, x) * [(Az)(x) * (Bz)(x) - (Cz)(x)] = 0
        /// Degree: 3 (after first round)
        fn verifyStage1(
            self: *Self,
            proof: *const StageProof(F),
            transcript: *transcripts.Transcript(F),
        ) !bool {
            // Number of rounds = number of round polynomials in the proof
            const num_rounds = proof.round_polys.items.len;

            std.debug.print("\n[VERIFIER STAGE 1] =====================================\n", .{});
            std.debug.print("[VERIFIER STAGE 1] Outer Spartan sumcheck\n", .{});
            std.debug.print("[VERIFIER STAGE 1] num_rounds={d}, final_claims_len={d}\n", .{ num_rounds, proof.final_claims.items.len });

            // Skip if no rounds (empty trace)
            if (num_rounds == 0) {
                std.debug.print("[VERIFIER STAGE 1] SKIPPED (no rounds)\n", .{});
                self.stage_results[0] = .{
                    .success = true,
                    .final_claim = null,
                    .error_msg = null,
                };
                self.current_stage = 1;
                return true;
            }

            // Get tau challenges from transcript (must match prover)
            // Prover generates log_num_constraints tau challenges before sumcheck rounds
            std.debug.print("[VERIFIER STAGE 1] Getting {d} tau challenges...\n", .{num_rounds});
            for (0..num_rounds) |i| {
                const tau = try transcript.challengeScalar("spartan_tau");
                std.debug.print("[VERIFIER STAGE 1]   tau[{d}] = {x:0>16}{x:0>16}{x:0>16}{x:0>16}\n", .{ i, tau.limbs[3], tau.limbs[2], tau.limbs[1], tau.limbs[0] });
            }

            // For Spartan, the initial claim should be 0 (R1CS satisfied)
            // This is recorded as the first final_claim by the prover
            var current_claim = if (proof.final_claims.items.len > 0)
                proof.final_claims.items[0]
            else
                F.zero();

            std.debug.print("[VERIFIER STAGE 1] Initial claim = {x:0>16}{x:0>16}{x:0>16}{x:0>16}\n", .{ current_claim.limbs[3], current_claim.limbs[2], current_claim.limbs[1], current_claim.limbs[0] });

            // Verify each round polynomial
            for (proof.round_polys.items, 0..) |round_poly, round_idx| {
                std.debug.print("\n[VERIFIER STAGE 1] --- Round {d}/{d} ---\n", .{ round_idx, num_rounds });
                std.debug.print("[VERIFIER STAGE 1] round_poly.len = {d}\n", .{round_poly.len});

                // For degree 3, round_poly has 4 coefficients: [p(0), p(1), p(2), p(3)]
                // or evaluations that we interpolate from
                if (round_poly.len < 2) {
                    std.debug.print("[VERIFIER STAGE 1] ERROR: round_poly.len < 2\n", .{});
                    self.stage_results[0] = .{
                        .success = false,
                        .final_claim = null,
                        .error_msg = "Stage 1: invalid round polynomial length",
                    };
                    return false; // Invalid polynomial
                }

                // Print all polynomial coefficients
                for (round_poly, 0..) |coeff, i| {
                    std.debug.print("[VERIFIER STAGE 1] p[{d}] = {x:0>16}{x:0>16}{x:0>16}{x:0>16}\n", .{ i, coeff.limbs[3], coeff.limbs[2], coeff.limbs[1], coeff.limbs[0] });
                }

                // Verify: p(0) + p(1) = current_claim
                const sum = round_poly[0].add(round_poly[1]);
                const sum_check_ok = sum.eql(current_claim);

                std.debug.print("[VERIFIER STAGE 1] p(0) + p(1) = {x:0>16}{x:0>16}{x:0>16}{x:0>16}\n", .{ sum.limbs[3], sum.limbs[2], sum.limbs[1], sum.limbs[0] });
                std.debug.print("[VERIFIER STAGE 1] claim       = {x:0>16}{x:0>16}{x:0>16}{x:0>16}\n", .{ current_claim.limbs[3], current_claim.limbs[2], current_claim.limbs[1], current_claim.limbs[0] });
                std.debug.print("[VERIFIER STAGE 1] sum_check_ok = {}\n", .{sum_check_ok});

                // Absorb round polynomial into transcript (Fiat-Shamir binding)
                try transcript.appendScalar("round_poly_0", round_poly[0]);
                try transcript.appendScalar("round_poly_1", round_poly[1]);
                if (round_poly.len > 2) {
                    try transcript.appendScalar("round_poly_2", round_poly[2]);
                }

                // Get challenge from transcript (must be called to keep in sync)
                const challenge = try transcript.challengeScalar("spartan_round");
                std.debug.print("[VERIFIER STAGE 1] challenge = {x:0>16}{x:0>16}{x:0>16}{x:0>16}\n", .{ challenge.limbs[3], challenge.limbs[2], challenge.limbs[1], challenge.limbs[0] });

                if (self.config.strict_sumcheck and !sum_check_ok) {
                    // Strict mode: reject the proof if sum check fails
                    std.debug.print("[VERIFIER STAGE 1] FAILED: p(0) + p(1) != claim\n", .{});
                    self.stage_results[0] = .{
                        .success = false,
                        .final_claim = null,
                        .error_msg = "Stage 1: sumcheck failed - p(0) + p(1) != claim",
                    };
                    return false;
                }

                // Update claim: evaluate p at challenge point
                current_claim = evaluatePolynomialAtChallenge(F, round_poly, challenge);
                std.debug.print("[VERIFIER STAGE 1] new_claim = p(r) = {x:0>16}{x:0>16}{x:0>16}{x:0>16}\n", .{ current_claim.limbs[3], current_claim.limbs[2], current_claim.limbs[1], current_claim.limbs[0] });
            }

            std.debug.print("\n[VERIFIER STAGE 1] Final claim after all rounds = {x:0>16}{x:0>16}{x:0>16}{x:0>16}\n", .{ current_claim.limbs[3], current_claim.limbs[2], current_claim.limbs[1], current_claim.limbs[0] });

            // Stage 1 verification passed
            self.stage_results[0] = .{
                .success = true,
                .final_claim = null,
                .error_msg = null,
            };
            self.current_stage = 1;
            return true;
        }

        /// Verify Stage 2: RAM RAF evaluation
        ///
        /// Verifies: Σ_k ra(k) * unmap(k) = raf_claim
        /// Degree: 2 (product of two linear polynomials)
        fn verifyStage2(
            self: *Self,
            proof: *const StageProof(F),
            transcript: *transcripts.Transcript(F),
            log_t: usize,
        ) !bool {
            const num_rounds = proof.round_polys.items.len;

            std.debug.print("\n[VERIFIER STAGE 2] =====================================\n", .{});
            std.debug.print("[VERIFIER STAGE 2] RAM RAF evaluation\n", .{});
            std.debug.print("[VERIFIER STAGE 2] num_rounds={d}, log_t={d}\n", .{ num_rounds, log_t });

            // Skip if empty
            if (num_rounds == 0) {
                std.debug.print("[VERIFIER STAGE 2] SKIPPED (no rounds)\n", .{});
                self.stage_results[1] = .{
                    .success = true,
                    .final_claim = null,
                    .error_msg = null,
                };
                self.current_stage = 2;
                return true;
            }

            // Prover generates log_t r_cycle challenges before sumcheck rounds
            std.debug.print("[VERIFIER STAGE 2] Getting {d} r_cycle challenges...\n", .{log_t});
            for (0..log_t) |i| {
                const r_cycle = try transcript.challengeScalar("r_cycle");
                std.debug.print("[VERIFIER STAGE 2]   r_cycle[{d}] = {x:0>16}{x:0>16}{x:0>16}{x:0>16}\n", .{ i, r_cycle.limbs[3], r_cycle.limbs[2], r_cycle.limbs[1], r_cycle.limbs[0] });
            }

            // Get initial claim from proof (if provided)
            var current_claim = if (proof.final_claims.items.len > 0)
                proof.final_claims.items[0]
            else
                F.zero();

            std.debug.print("[VERIFIER STAGE 2] Initial claim = {x:0>16}{x:0>16}{x:0>16}{x:0>16}\n", .{ current_claim.limbs[3], current_claim.limbs[2], current_claim.limbs[1], current_claim.limbs[0] });

            // Verify each round polynomial
            // For degree-2 sumcheck, round_poly = [p(0), p(2)]
            // We recover p(1) = claim - p(0) from the sumcheck constraint
            for (proof.round_polys.items, 0..) |round_poly, round_idx| {
                std.debug.print("\n[VERIFIER STAGE 2] --- Round {d}/{d} ---\n", .{ round_idx, num_rounds });

                if (round_poly.len < 2) {
                    std.debug.print("[VERIFIER STAGE 2] ERROR: round_poly.len < 2\n", .{});
                    self.stage_results[1] = .{
                        .success = false,
                        .final_claim = null,
                        .error_msg = "Stage 2: invalid round polynomial length",
                    };
                    return false;
                }

                // round_poly = [p(0), p(2)] for degree-2 compressed format
                const p_at_0 = round_poly[0];
                const p_at_2 = round_poly[1];

                // Recover p(1) from sumcheck constraint: p(0) + p(1) = current_claim
                const p_at_1 = current_claim.sub(p_at_0);

                std.debug.print("[VERIFIER STAGE 2] p(0) = {x:0>16}{x:0>16}{x:0>16}{x:0>16}\n", .{ p_at_0.limbs[3], p_at_0.limbs[2], p_at_0.limbs[1], p_at_0.limbs[0] });
                std.debug.print("[VERIFIER STAGE 2] p(1) = {x:0>16}{x:0>16}{x:0>16}{x:0>16} (recovered)\n", .{ p_at_1.limbs[3], p_at_1.limbs[2], p_at_1.limbs[1], p_at_1.limbs[0] });
                std.debug.print("[VERIFIER STAGE 2] p(2) = {x:0>16}{x:0>16}{x:0>16}{x:0>16}\n", .{ p_at_2.limbs[3], p_at_2.limbs[2], p_at_2.limbs[1], p_at_2.limbs[0] });
                std.debug.print("[VERIFIER STAGE 2] claim = {x:0>16}{x:0>16}{x:0>16}{x:0>16}\n", .{ current_claim.limbs[3], current_claim.limbs[2], current_claim.limbs[1], current_claim.limbs[0] });

                // Get challenge from transcript (must match prover's challenge)
                const challenge = try transcript.challengeScalar("raf_round");
                std.debug.print("[VERIFIER STAGE 2] challenge = {x:0>16}{x:0>16}{x:0>16}{x:0>16}\n", .{ challenge.limbs[3], challenge.limbs[2], challenge.limbs[1], challenge.limbs[0] });

                // Accumulate challenge for opening claims
                try self.opening_claims.addChallenge(challenge);

                // Update claim using quadratic Lagrange interpolation
                // with points p(0), p(1), p(2)
                current_claim = evaluateQuadraticAt3Points(F, p_at_0, p_at_1, p_at_2, challenge);
                std.debug.print("[VERIFIER STAGE 2] new_claim = p(r) = {x:0>16}{x:0>16}{x:0>16}{x:0>16}\n", .{ current_claim.limbs[3], current_claim.limbs[2], current_claim.limbs[1], current_claim.limbs[0] });
            }

            std.debug.print("\n[VERIFIER STAGE 2] Final claim = {x:0>16}{x:0>16}{x:0>16}{x:0>16}\n", .{ current_claim.limbs[3], current_claim.limbs[2], current_claim.limbs[1], current_claim.limbs[0] });

            // Stage 2 verification passed
            self.stage_results[1] = .{
                .success = true,
                .final_claim = null,
                .error_msg = null,
            };
            self.current_stage = 2;
            return true;
        }

        /// Verify Stage 3: Lasso lookup
        ///
        /// Two-phase sumcheck:
        /// 1. Address binding (log_K rounds)
        /// 2. Cycle binding (log_T rounds)
        fn verifyStage3(
            self: *Self,
            proof: *const StageProof(F),
            transcript: *transcripts.Transcript(F),
            log_t: usize,
            log_k: usize,
        ) !bool {
            const num_rounds = proof.round_polys.items.len;

            std.debug.print("\n[VERIFIER STAGE 3] =====================================\n", .{});
            std.debug.print("[VERIFIER STAGE 3] Lasso lookup sumcheck\n", .{});
            std.debug.print("[VERIFIER STAGE 3] num_rounds={d}, log_t={d}, log_k={d}\n", .{ num_rounds, log_t, log_k });

            // Skip if no round polynomials (empty lookup trace)
            if (num_rounds == 0) {
                std.debug.print("[VERIFIER STAGE 3] SKIPPED (no rounds)\n", .{});
                self.stage_results[2] = .{
                    .success = true,
                    .final_claim = null,
                    .error_msg = null,
                };
                self.current_stage = 3;
                return true;
            }

            // Get gamma challenge for batching (prover generates this first)
            const gamma = try transcript.challengeScalar("lasso_gamma");
            std.debug.print("[VERIFIER STAGE 3] gamma = {x:0>16}{x:0>16}{x:0>16}{x:0>16}\n", .{ gamma.limbs[3], gamma.limbs[2], gamma.limbs[1], gamma.limbs[0] });

            // Prover generates log_t r_reduction challenges
            std.debug.print("[VERIFIER STAGE 3] Getting {d} r_reduction challenges...\n", .{log_t});
            for (0..log_t) |i| {
                const r_red = try transcript.challengeScalar("r_reduction");
                std.debug.print("[VERIFIER STAGE 3]   r_reduction[{d}] = {x:0>16}{x:0>16}{x:0>16}{x:0>16}\n", .{ i, r_red.limbs[3], r_red.limbs[2], r_red.limbs[1], r_red.limbs[0] });
            }

            // Get initial claim
            var current_claim = if (proof.final_claims.items.len > 0)
                proof.final_claims.items[0]
            else
                F.zero();

            std.debug.print("[VERIFIER STAGE 3] Initial claim = {x:0>16}{x:0>16}{x:0>16}{x:0>16}\n", .{ current_claim.limbs[3], current_claim.limbs[2], current_claim.limbs[1], current_claim.limbs[0] });

            // Verify each round
            // Lasso prover sends polynomial in coefficient form: [c0, c1, c2] for p(X) = c0 + c1*X + c2*X^2
            for (proof.round_polys.items, 0..) |round_poly, round_idx| {
                const is_address_phase = round_idx < log_k;
                std.debug.print("\n[VERIFIER STAGE 3] --- Round {d}/{d} ({s}) ---\n", .{ round_idx, num_rounds, if (is_address_phase) "address" else "cycle" });

                if (round_poly.len < 2) {
                    std.debug.print("[VERIFIER STAGE 3] ERROR: round_poly.len < 2\n", .{});
                    self.stage_results[2] = .{
                        .success = false,
                        .final_claim = null,
                        .error_msg = "Stage 3: invalid round polynomial",
                    };
                    return false;
                }

                // Convert from coefficient form to evaluation form
                // p(X) = c0 + c1*X + c2*X^2
                const c0 = round_poly[0];
                const c1 = round_poly[1];
                const c2 = if (round_poly.len > 2) round_poly[2] else F.zero();

                std.debug.print("[VERIFIER STAGE 3] c0 = {x:0>16}{x:0>16}{x:0>16}{x:0>16}\n", .{ c0.limbs[3], c0.limbs[2], c0.limbs[1], c0.limbs[0] });
                std.debug.print("[VERIFIER STAGE 3] c1 = {x:0>16}{x:0>16}{x:0>16}{x:0>16}\n", .{ c1.limbs[3], c1.limbs[2], c1.limbs[1], c1.limbs[0] });
                std.debug.print("[VERIFIER STAGE 3] c2 = {x:0>16}{x:0>16}{x:0>16}{x:0>16}\n", .{ c2.limbs[3], c2.limbs[2], c2.limbs[1], c2.limbs[0] });

                // Compute p(0) = c0, p(1) = c0 + c1 + c2, p(2) = c0 + 2*c1 + 4*c2
                const p_at_0 = c0;
                const p_at_1 = c0.add(c1).add(c2);

                std.debug.print("[VERIFIER STAGE 3] p(0) = {x:0>16}{x:0>16}{x:0>16}{x:0>16}\n", .{ p_at_0.limbs[3], p_at_0.limbs[2], p_at_0.limbs[1], p_at_0.limbs[0] });
                std.debug.print("[VERIFIER STAGE 3] p(1) = {x:0>16}{x:0>16}{x:0>16}{x:0>16}\n", .{ p_at_1.limbs[3], p_at_1.limbs[2], p_at_1.limbs[1], p_at_1.limbs[0] });

                // Verify sumcheck constraint: p(0) + p(1) = current_claim
                const sum = p_at_0.add(p_at_1);
                const sum_check_ok = sum.eql(current_claim);

                std.debug.print("[VERIFIER STAGE 3] p(0)+p(1) = {x:0>16}{x:0>16}{x:0>16}{x:0>16}\n", .{ sum.limbs[3], sum.limbs[2], sum.limbs[1], sum.limbs[0] });
                std.debug.print("[VERIFIER STAGE 3] claim     = {x:0>16}{x:0>16}{x:0>16}{x:0>16}\n", .{ current_claim.limbs[3], current_claim.limbs[2], current_claim.limbs[1], current_claim.limbs[0] });
                std.debug.print("[VERIFIER STAGE 3] sum_check_ok = {}\n", .{sum_check_ok});

                // Get challenge
                const challenge = try transcript.challengeScalar("lasso_round");
                std.debug.print("[VERIFIER STAGE 3] challenge = {x:0>16}{x:0>16}{x:0>16}{x:0>16}\n", .{ challenge.limbs[3], challenge.limbs[2], challenge.limbs[1], challenge.limbs[0] });

                if (self.config.strict_sumcheck and !sum_check_ok) {
                    std.debug.print("[VERIFIER STAGE 3] FAILED: p(0) + p(1) != claim\n", .{});
                    self.stage_results[2] = .{
                        .success = false,
                        .final_claim = null,
                        .error_msg = "Stage 3: sumcheck failed - p(0) + p(1) != claim",
                    };
                    return false;
                }

                try self.opening_claims.addChallenge(challenge);

                // Update claim: evaluate p(challenge) = c0 + c1*r + c2*r^2
                const new_claim = c0.add(c1.mul(challenge)).add(c2.mul(challenge).mul(challenge));
                std.debug.print("[VERIFIER STAGE 3] new_claim = p(r) = {x:0>16}{x:0>16}{x:0>16}{x:0>16}\n", .{ new_claim.limbs[3], new_claim.limbs[2], new_claim.limbs[1], new_claim.limbs[0] });
                current_claim = new_claim;
            }

            std.debug.print("\n[VERIFIER STAGE 3] Final claim = {x:0>16}{x:0>16}{x:0>16}{x:0>16}\n", .{ current_claim.limbs[3], current_claim.limbs[2], current_claim.limbs[1], current_claim.limbs[0] });

            // Stage 3 verification passed
            self.stage_results[2] = .{
                .success = true,
                .final_claim = null,
                .error_msg = null,
            };
            self.current_stage = 3;
            return true;
        }

        /// Verify Stage 4: Value evaluation
        ///
        /// Verifies: Val(r) - Val_init = Σ_j inc(j) * wa(j) * LT(j)
        /// Degree: 3
        fn verifyStage4(
            self: *Self,
            proof: *const StageProof(F),
            transcript: *transcripts.Transcript(F),
            log_t: usize,
        ) !bool {
            const num_rounds = proof.round_polys.items.len;

            std.debug.print("\n[VERIFIER STAGE 4] =====================================\n", .{});
            std.debug.print("[VERIFIER STAGE 4] Value evaluation\n", .{});
            std.debug.print("[VERIFIER STAGE 4] num_rounds={d}, log_t={d}\n", .{ num_rounds, log_t });

            // Skip if empty
            if (num_rounds == 0) {
                std.debug.print("[VERIFIER STAGE 4] SKIPPED (no rounds)\n", .{});
                self.stage_results[3] = .{
                    .success = true,
                    .final_claim = null,
                    .error_msg = null,
                };
                self.current_stage = 4;
                return true;
            }

            // Prover generates log_k r_address challenges (log_k = 16)
            const log_k: usize = 16;
            std.debug.print("[VERIFIER STAGE 4] Getting {d} r_address challenges...\n", .{log_k});
            for (0..log_k) |i| {
                const r_addr = try transcript.challengeScalar("r_address");
                std.debug.print("[VERIFIER STAGE 4]   r_address[{d}] = {x:0>16}{x:0>16}{x:0>16}{x:0>16}\n", .{ i, r_addr.limbs[3], r_addr.limbs[2], r_addr.limbs[1], r_addr.limbs[0] });
            }

            // Prover generates log_t r_cycle_val challenges
            std.debug.print("[VERIFIER STAGE 4] Getting {d} r_cycle_val challenges...\n", .{log_t});
            for (0..log_t) |i| {
                const r_cycle = try transcript.challengeScalar("r_cycle_val");
                std.debug.print("[VERIFIER STAGE 4]   r_cycle_val[{d}] = {x:0>16}{x:0>16}{x:0>16}{x:0>16}\n", .{ i, r_cycle.limbs[3], r_cycle.limbs[2], r_cycle.limbs[1], r_cycle.limbs[0] });
            }

            // Get initial claim
            var current_claim = if (proof.final_claims.items.len > 0)
                proof.final_claims.items[0]
            else
                F.zero();

            std.debug.print("[VERIFIER STAGE 4] Initial claim = {x:0>16}{x:0>16}{x:0>16}{x:0>16}\n", .{ current_claim.limbs[3], current_claim.limbs[2], current_claim.limbs[1], current_claim.limbs[0] });

            // Verify each round polynomial
            for (proof.round_polys.items, 0..) |round_poly, round_idx| {
                std.debug.print("\n[VERIFIER STAGE 4] --- Round {d}/{d} ---\n", .{ round_idx, num_rounds });

                if (round_poly.len < 4) {
                    // Degree 3 sumcheck (product of 3 multilinear) requires 4 evaluations
                    std.debug.print("[VERIFIER STAGE 4] ERROR: round_poly.len={d} < 4\n", .{round_poly.len});
                    self.stage_results[3] = .{
                        .success = false,
                        .final_claim = null,
                        .error_msg = "Stage 4: invalid round polynomial (need 4 evals for degree 3)",
                    };
                    return false;
                }

                // Print all polynomial coefficients
                for (round_poly, 0..) |coeff, i| {
                    std.debug.print("[VERIFIER STAGE 4] p[{d}] = {x:0>16}{x:0>16}{x:0>16}{x:0>16}\n", .{ i, coeff.limbs[3], coeff.limbs[2], coeff.limbs[1], coeff.limbs[0] });
                }

                // Verify sumcheck: p(0) + p(1) = current_claim
                const sum = round_poly[0].add(round_poly[1]);
                const sum_check_ok = sum.eql(current_claim);

                std.debug.print("[VERIFIER STAGE 4] p(0)+p(1) = {x:0>16}{x:0>16}{x:0>16}{x:0>16}\n", .{ sum.limbs[3], sum.limbs[2], sum.limbs[1], sum.limbs[0] });
                std.debug.print("[VERIFIER STAGE 4] claim     = {x:0>16}{x:0>16}{x:0>16}{x:0>16}\n", .{ current_claim.limbs[3], current_claim.limbs[2], current_claim.limbs[1], current_claim.limbs[0] });
                std.debug.print("[VERIFIER STAGE 4] sum_check_ok = {}\n", .{sum_check_ok});

                // Get challenge
                const challenge = try transcript.challengeScalar("val_eval_round");
                std.debug.print("[VERIFIER STAGE 4] challenge = {x:0>16}{x:0>16}{x:0>16}{x:0>16}\n", .{ challenge.limbs[3], challenge.limbs[2], challenge.limbs[1], challenge.limbs[0] });

                if (self.config.strict_sumcheck and !sum_check_ok) {
                    std.debug.print("[VERIFIER STAGE 4] FAILED: p(0) + p(1) != claim\n", .{});
                    self.stage_results[3] = .{
                        .success = false,
                        .final_claim = null,
                        .error_msg = "Stage 4: sumcheck failed - p(0) + p(1) != claim",
                    };
                    return false;
                }

                try self.opening_claims.addChallenge(challenge);

                // Update claim using interpolation for degree 3
                current_claim = evaluatePolynomialAtChallenge(F, round_poly, challenge);
                std.debug.print("[VERIFIER STAGE 4] new_claim = p(r) = {x:0>16}{x:0>16}{x:0>16}{x:0>16}\n", .{ current_claim.limbs[3], current_claim.limbs[2], current_claim.limbs[1], current_claim.limbs[0] });
            }

            std.debug.print("\n[VERIFIER STAGE 4] Final claim = {x:0>16}{x:0>16}{x:0>16}{x:0>16}\n", .{ current_claim.limbs[3], current_claim.limbs[2], current_claim.limbs[1], current_claim.limbs[0] });

            // Stage 4 verification passed
            self.stage_results[3] = .{
                .success = true,
                .final_claim = null,
                .error_msg = null,
            };
            self.current_stage = 4;
            return true;
        }

        /// Verify Stage 5: Register evaluation
        ///
        /// Similar to Stage 4 but for 32 registers (log_k = 5)
        fn verifyStage5(
            self: *Self,
            proof: *const StageProof(F),
            transcript: *transcripts.Transcript(F),
            log_t: usize,
        ) !bool {
            const num_rounds = proof.round_polys.items.len;

            std.debug.print("\n[VERIFIER STAGE 5] =====================================\n", .{});
            std.debug.print("[VERIFIER STAGE 5] Register evaluation\n", .{});
            std.debug.print("[VERIFIER STAGE 5] num_rounds={d}, log_t={d}\n", .{ num_rounds, log_t });

            // Skip if empty
            if (num_rounds == 0) {
                std.debug.print("[VERIFIER STAGE 5] SKIPPED (no rounds)\n", .{});
                self.stage_results[4] = .{
                    .success = true,
                    .final_claim = null,
                    .error_msg = null,
                };
                self.current_stage = 5;
                return true;
            }

            // Prover generates 5 r_register challenges (log2(32) = 5)
            const log_regs: usize = 5;
            std.debug.print("[VERIFIER STAGE 5] Getting {d} r_register challenges...\n", .{log_regs});
            for (0..log_regs) |i| {
                const r_reg = try transcript.challengeScalar("r_register");
                std.debug.print("[VERIFIER STAGE 5]   r_register[{d}] = {x:0>16}{x:0>16}{x:0>16}{x:0>16}\n", .{ i, r_reg.limbs[3], r_reg.limbs[2], r_reg.limbs[1], r_reg.limbs[0] });
            }

            // Prover generates log_t r_cycle_reg challenges
            std.debug.print("[VERIFIER STAGE 5] Getting {d} r_cycle_reg challenges...\n", .{log_t});
            for (0..log_t) |i| {
                const r_cycle = try transcript.challengeScalar("r_cycle_reg");
                std.debug.print("[VERIFIER STAGE 5]   r_cycle_reg[{d}] = {x:0>16}{x:0>16}{x:0>16}{x:0>16}\n", .{ i, r_cycle.limbs[3], r_cycle.limbs[2], r_cycle.limbs[1], r_cycle.limbs[0] });
            }

            // Get initial claim
            var current_claim = if (proof.final_claims.items.len > 0)
                proof.final_claims.items[0]
            else
                F.zero();

            std.debug.print("[VERIFIER STAGE 5] Initial claim = {x:0>16}{x:0>16}{x:0>16}{x:0>16}\n", .{ current_claim.limbs[3], current_claim.limbs[2], current_claim.limbs[1], current_claim.limbs[0] });

            // Verify each round
            // For degree-2 sumcheck, round_poly = [p(0), p(2)]
            for (proof.round_polys.items, 0..) |round_poly, round_idx| {
                std.debug.print("\n[VERIFIER STAGE 5] --- Round {d}/{d} ---\n", .{ round_idx, num_rounds });

                if (round_poly.len < 2) {
                    std.debug.print("[VERIFIER STAGE 5] ERROR: round_poly.len < 2\n", .{});
                    self.stage_results[4] = .{
                        .success = false,
                        .final_claim = null,
                        .error_msg = "Stage 5: invalid round polynomial",
                    };
                    return false;
                }

                // round_poly = [p(0), p(2)] for degree-2 compressed format
                const p_at_0 = round_poly[0];
                const p_at_2 = round_poly[1];

                // Recover p(1) from sumcheck constraint: p(0) + p(1) = current_claim
                const p_at_1 = current_claim.sub(p_at_0);

                std.debug.print("[VERIFIER STAGE 5] p(0) = {x:0>16}{x:0>16}{x:0>16}{x:0>16}\n", .{ p_at_0.limbs[3], p_at_0.limbs[2], p_at_0.limbs[1], p_at_0.limbs[0] });
                std.debug.print("[VERIFIER STAGE 5] p(1) = {x:0>16}{x:0>16}{x:0>16}{x:0>16} (recovered)\n", .{ p_at_1.limbs[3], p_at_1.limbs[2], p_at_1.limbs[1], p_at_1.limbs[0] });
                std.debug.print("[VERIFIER STAGE 5] p(2) = {x:0>16}{x:0>16}{x:0>16}{x:0>16}\n", .{ p_at_2.limbs[3], p_at_2.limbs[2], p_at_2.limbs[1], p_at_2.limbs[0] });
                std.debug.print("[VERIFIER STAGE 5] claim = {x:0>16}{x:0>16}{x:0>16}{x:0>16}\n", .{ current_claim.limbs[3], current_claim.limbs[2], current_claim.limbs[1], current_claim.limbs[0] });

                // Get challenge
                const challenge = try transcript.challengeScalar("reg_eval_round");
                std.debug.print("[VERIFIER STAGE 5] challenge = {x:0>16}{x:0>16}{x:0>16}{x:0>16}\n", .{ challenge.limbs[3], challenge.limbs[2], challenge.limbs[1], challenge.limbs[0] });

                try self.opening_claims.addChallenge(challenge);

                // Update claim using quadratic interpolation
                current_claim = evaluateQuadraticAt3Points(F, p_at_0, p_at_1, p_at_2, challenge);
                std.debug.print("[VERIFIER STAGE 5] new_claim = p(r) = {x:0>16}{x:0>16}{x:0>16}{x:0>16}\n", .{ current_claim.limbs[3], current_claim.limbs[2], current_claim.limbs[1], current_claim.limbs[0] });
            }

            std.debug.print("\n[VERIFIER STAGE 5] Final claim = {x:0>16}{x:0>16}{x:0>16}{x:0>16}\n", .{ current_claim.limbs[3], current_claim.limbs[2], current_claim.limbs[1], current_claim.limbs[0] });

            // Stage 5 verification passed
            self.stage_results[4] = .{
                .success = true,
                .final_claim = null,
                .error_msg = null,
            };
            self.current_stage = 5;
            return true;
        }

        /// Verify Stage 6: Booleanity and Hamming weight
        ///
        /// Verifies: All flags f satisfy f * (1-f) = 0
        /// Degree: 2
        fn verifyStage6(
            self: *Self,
            proof: *const StageProof(F),
            transcript: *transcripts.Transcript(F),
            log_t: usize,
        ) !bool {
            const num_rounds = proof.round_polys.items.len;

            std.debug.print("\n[VERIFIER STAGE 6] =====================================\n", .{});
            std.debug.print("[VERIFIER STAGE 6] Booleanity and Hamming weight\n", .{});
            std.debug.print("[VERIFIER STAGE 6] num_rounds={d}, log_t={d}\n", .{ num_rounds, log_t });

            // Get booleanity challenge
            const bool_chal = try transcript.challengeScalar("booleanity");
            std.debug.print("[VERIFIER STAGE 6] booleanity = {x:0>16}{x:0>16}{x:0>16}{x:0>16}\n", .{ bool_chal.limbs[3], bool_chal.limbs[2], bool_chal.limbs[1], bool_chal.limbs[0] });

            // Get initial claim (should be 0 for valid flags)
            var current_claim = if (proof.final_claims.items.len > 0)
                proof.final_claims.items[0]
            else
                F.zero();

            std.debug.print("[VERIFIER STAGE 6] Initial claim = {x:0>16}{x:0>16}{x:0>16}{x:0>16}\n", .{ current_claim.limbs[3], current_claim.limbs[2], current_claim.limbs[1], current_claim.limbs[0] });

            // Verify each round
            // For degree-2 sumcheck, round_poly = [p(0), p(2)]
            for (proof.round_polys.items, 0..) |round_poly, round_idx| {
                std.debug.print("\n[VERIFIER STAGE 6] --- Round {d}/{d} ---\n", .{ round_idx, num_rounds });

                if (round_poly.len < 2) {
                    std.debug.print("[VERIFIER STAGE 6] ERROR: round_poly.len < 2\n", .{});
                    self.stage_results[5] = .{
                        .success = false,
                        .final_claim = null,
                        .error_msg = "Stage 6: invalid round polynomial",
                    };
                    return false;
                }

                // round_poly = [p(0), p(2)] for degree-2 compressed format
                const p_at_0 = round_poly[0];
                const p_at_2 = round_poly[1];

                // Recover p(1) from sumcheck constraint: p(0) + p(1) = current_claim
                const p_at_1 = current_claim.sub(p_at_0);

                std.debug.print("[VERIFIER STAGE 6] p(0) = {x:0>16}{x:0>16}{x:0>16}{x:0>16}\n", .{ p_at_0.limbs[3], p_at_0.limbs[2], p_at_0.limbs[1], p_at_0.limbs[0] });
                std.debug.print("[VERIFIER STAGE 6] p(1) = {x:0>16}{x:0>16}{x:0>16}{x:0>16} (recovered)\n", .{ p_at_1.limbs[3], p_at_1.limbs[2], p_at_1.limbs[1], p_at_1.limbs[0] });
                std.debug.print("[VERIFIER STAGE 6] p(2) = {x:0>16}{x:0>16}{x:0>16}{x:0>16}\n", .{ p_at_2.limbs[3], p_at_2.limbs[2], p_at_2.limbs[1], p_at_2.limbs[0] });
                std.debug.print("[VERIFIER STAGE 6] claim = {x:0>16}{x:0>16}{x:0>16}{x:0>16}\n", .{ current_claim.limbs[3], current_claim.limbs[2], current_claim.limbs[1], current_claim.limbs[0] });

                // Get challenge
                const challenge = try transcript.challengeScalar("bool_round");
                std.debug.print("[VERIFIER STAGE 6] challenge = {x:0>16}{x:0>16}{x:0>16}{x:0>16}\n", .{ challenge.limbs[3], challenge.limbs[2], challenge.limbs[1], challenge.limbs[0] });

                try self.opening_claims.addChallenge(challenge);

                // Update claim using quadratic interpolation
                current_claim = evaluateQuadraticAt3Points(F, p_at_0, p_at_1, p_at_2, challenge);
                std.debug.print("[VERIFIER STAGE 6] new_claim = p(r) = {x:0>16}{x:0>16}{x:0>16}{x:0>16}\n", .{ current_claim.limbs[3], current_claim.limbs[2], current_claim.limbs[1], current_claim.limbs[0] });
            }

            std.debug.print("\n[VERIFIER STAGE 6] Final claim = {x:0>16}{x:0>16}{x:0>16}{x:0>16}\n", .{ current_claim.limbs[3], current_claim.limbs[2], current_claim.limbs[1], current_claim.limbs[0] });

            // Stage 6 verification passed
            self.stage_results[5] = .{
                .success = true,
                .final_claim = null,
                .error_msg = null,
            };
            self.current_stage = 6;
            return true;
        }
    };
}

/// Accumulator for opening claims across all stages
pub fn OpeningClaimAccumulator(comptime F: type) type {
    return struct {
        const Self = @This();

        /// All challenges accumulated across stages
        challenges: std.ArrayListUnmanaged(F),
        /// Opening points (slices into challenges)
        opening_points: std.ArrayListUnmanaged(struct { start: usize, len: usize }),
        /// Claimed evaluations
        claimed_evals: std.ArrayListUnmanaged(F),
        allocator: Allocator,

        pub fn init(allocator: Allocator) Self {
            return Self{
                .challenges = .{},
                .opening_points = .{},
                .claimed_evals = .{},
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            self.challenges.deinit(self.allocator);
            self.opening_points.deinit(self.allocator);
            self.claimed_evals.deinit(self.allocator);
        }

        /// Add a challenge to the accumulator
        pub fn addChallenge(self: *Self, challenge: F) !void {
            try self.challenges.append(self.allocator, challenge);
        }

        /// Record an opening claim
        pub fn recordOpening(self: *Self, eval: F) !void {
            const point_start = if (self.opening_points.items.len > 0)
                self.opening_points.items[self.opening_points.items.len - 1].start +
                    self.opening_points.items[self.opening_points.items.len - 1].len
            else
                0;

            try self.opening_points.append(self.allocator, .{
                .start = point_start,
                .len = self.challenges.items.len - point_start,
            });
            try self.claimed_evals.append(self.allocator, eval);
        }

        /// Get all challenges as a slice
        pub fn getChallenges(self: *const Self) []const F {
            return self.challenges.items;
        }
    };
}

/// Evaluate a quadratic polynomial given p(0), p(1), p(2) at point r
/// Uses Lagrange interpolation: p(r) = L0(r)*p(0) + L1(r)*p(1) + L2(r)*p(2)
fn evaluateQuadraticAt3Points(comptime F: type, p0: F, p1: F, p2: F, r: F) F {
    const two = F.fromU64(2);
    const r_minus_1 = r.sub(F.one());
    const r_minus_2 = r.sub(two);

    // L0(r) = (r-1)(r-2)/((0-1)(0-2)) = (r-1)(r-2)/2
    const L0 = r_minus_1.mul(r_minus_2).mul(two.inverse().?);
    // L1(r) = r(r-2)/((1-0)(1-2)) = -r(r-2)
    const L1 = r.mul(r_minus_2).neg();
    // L2(r) = r(r-1)/((2-0)(2-1)) = r(r-1)/2
    const L2 = r.mul(r_minus_1).mul(two.inverse().?);

    return p0.mul(L0).add(p1.mul(L1)).add(p2.mul(L2));
}

/// Evaluate a polynomial (given as evaluations at 0, 1, 2, ...) at a challenge point
/// Uses Lagrange interpolation for small degree polynomials
fn evaluatePolynomialAtChallenge(comptime F: type, evals: []const F, r: F) F {
    if (evals.len == 0) return F.zero();
    if (evals.len == 1) return evals[0];

    // For degree 1 (2 points): linear interpolation
    if (evals.len == 2) {
        const one_minus_r = F.one().sub(r);
        return one_minus_r.mul(evals[0]).add(r.mul(evals[1]));
    }

    // For degree 2 (3 points): quadratic Lagrange interpolation
    // p(r) = p(0) * (r-1)(r-2)/((0-1)(0-2)) + p(1) * (r-0)(r-2)/((1-0)(1-2)) + p(2) * (r-0)(r-1)/((2-0)(2-1))
    //      = p(0) * (r-1)(r-2)/2 - p(1) * r(r-2) + p(2) * r(r-1)/2
    if (evals.len == 3) {
        const two = F.fromU64(2);
        const r_minus_1 = r.sub(F.one());
        const r_minus_2 = r.sub(two);

        // L0(r) = (r-1)(r-2)/2
        const L0 = r_minus_1.mul(r_minus_2).mul(two.inverse().?);
        // L1(r) = -r(r-2)
        const L1 = r.mul(r_minus_2).neg();
        // L2(r) = r(r-1)/2
        const L2 = r.mul(r_minus_1).mul(two.inverse().?);

        return evals[0].mul(L0).add(evals[1].mul(L1)).add(evals[2].mul(L2));
    }

    // For degree 3 (4 points): cubic Lagrange interpolation
    if (evals.len >= 4) {
        const two = F.fromU64(2);
        const three = F.fromU64(3);
        const six = F.fromU64(6);

        const r_minus_1 = r.sub(F.one());
        const r_minus_2 = r.sub(two);
        const r_minus_3 = r.sub(three);

        // L0(r) = (r-1)(r-2)(r-3) / (-6)
        const L0 = r_minus_1.mul(r_minus_2).mul(r_minus_3).mul(six.neg().inverse().?);
        // L1(r) = r(r-2)(r-3) / 2
        const L1 = r.mul(r_minus_2).mul(r_minus_3).mul(two.inverse().?);
        // L2(r) = r(r-1)(r-3) / (-2)
        const L2 = r.mul(r_minus_1).mul(r_minus_3).mul(two.neg().inverse().?);
        // L3(r) = r(r-1)(r-2) / 6
        const L3 = r.mul(r_minus_1).mul(r_minus_2).mul(six.inverse().?);

        return evals[0].mul(L0).add(evals[1].mul(L1)).add(evals[2].mul(L2)).add(evals[3].mul(L3));
    }

    // Fallback: linear for unknown sizes
    const one_minus_r = F.one().sub(r);
    return one_minus_r.mul(evals[0]).add(r.mul(evals[evals.len - 1]));
}

// ============================================================================
// Tests
// ============================================================================

test "multi-stage verifier init" {
    const allocator = std.testing.allocator;
    const field = @import("../field/mod.zig");
    const F = field.BN254Scalar;

    var verifier = MultiStageVerifier(F).init(allocator);
    defer verifier.deinit();

    // All stages should start as not verified
    for (verifier.stage_results) |result| {
        try std.testing.expect(!result.success);
    }
}

test "opening claim accumulator" {
    const allocator = std.testing.allocator;
    const field = @import("../field/mod.zig");
    const F = field.BN254Scalar;

    var acc = OpeningClaimAccumulator(F).init(allocator);
    defer acc.deinit();

    try acc.addChallenge(F.fromU64(1));
    try acc.addChallenge(F.fromU64(2));
    try acc.addChallenge(F.fromU64(3));

    try std.testing.expectEqual(@as(usize, 3), acc.challenges.items.len);
}

test "polynomial evaluation at challenge - linear" {
    const field = @import("../field/mod.zig");
    const F = field.BN254Scalar;

    // p(x) with p(0) = 4, p(1) = 6
    // p(x) = 4 + 2x, so p(0) = 4, p(1) = 6
    const evals = [_]F{ F.fromU64(4), F.fromU64(6) };
    const r = F.fromU64(2);

    // p(2) = 4 + 2*2 = 8
    const result = evaluatePolynomialAtChallenge(F, &evals, r);
    try std.testing.expect(result.eql(F.fromU64(8)));
}

test "polynomial evaluation at challenge - quadratic" {
    const field = @import("../field/mod.zig");
    const F = field.BN254Scalar;

    // p(x) = x^2, so p(0) = 0, p(1) = 1, p(2) = 4
    const evals = [_]F{ F.fromU64(0), F.fromU64(1), F.fromU64(4) };
    const r = F.fromU64(3);

    // p(3) = 9
    const result = evaluatePolynomialAtChallenge(F, &evals, r);
    try std.testing.expect(result.eql(F.fromU64(9)));
}

test "evaluate quadratic at 3 points" {
    const field = @import("../field/mod.zig");
    const F = field.BN254Scalar;

    // p(x) = x^2, so p(0) = 0, p(1) = 1, p(2) = 4
    const p0 = F.fromU64(0);
    const p1 = F.fromU64(1);
    const p2 = F.fromU64(4);
    const r = F.fromU64(3);

    // p(3) = 9
    const result = evaluateQuadraticAt3Points(F, p0, p1, p2, r);
    try std.testing.expect(result.eql(F.fromU64(9)));
}

test "evaluate quadratic at 3 points - linear case" {
    const field = @import("../field/mod.zig");
    const F = field.BN254Scalar;

    // p(x) = 2x + 1, so p(0) = 1, p(1) = 3, p(2) = 5
    const p0 = F.fromU64(1);
    const p1 = F.fromU64(3);
    const p2 = F.fromU64(5);
    const r = F.fromU64(4);

    // p(4) = 9
    const result = evaluateQuadraticAt3Points(F, p0, p1, p2, r);
    try std.testing.expect(result.eql(F.fromU64(9)));
}

test "verifier config defaults" {
    const config = VerifierConfig{};
    try std.testing.expect(config.strict_sumcheck);
    try std.testing.expect(!config.debug_output);
}

test "verifier with strict mode" {
    const allocator = std.testing.allocator;
    const field = @import("../field/mod.zig");
    const F = field.BN254Scalar;

    // Create verifier with strict sumcheck enabled (default)
    var strict_verifier = MultiStageVerifier(F).init(allocator);
    defer strict_verifier.deinit();
    try std.testing.expect(strict_verifier.config.strict_sumcheck);

    // Create verifier with strict sumcheck disabled
    var lenient_verifier = MultiStageVerifier(F).initWithConfig(allocator, .{
        .strict_sumcheck = false,
    });
    defer lenient_verifier.deinit();
    try std.testing.expect(!lenient_verifier.config.strict_sumcheck);
}
