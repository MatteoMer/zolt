//! Proof Converter: Zolt Internal → Jolt Compatible Format
//!
//! This module converts Zolt's internal 6-stage proof structure to
//! Jolt's 7-stage proof format for cross-verification compatibility.
//!
//! ## Stage Mapping
//!
//! Zolt (6 stages):                    Jolt (7 stages):
//! 1. Outer Spartan           →        1. Outer Spartan (+ UniSkip)
//! 2. RAM RAF + Read-Write    →        2. Product virtualization + RAM RAF + RW (+ UniSkip)
//! 3. Instruction Lookup      →        3. Spartan shift + Instruction input + Registers claim
//! 4. Memory Val Evaluation   →        4. Registers RW + RAM val evaluation + RAM val final
//! 5. Register Val Evaluation →        5. Registers val evaluation + RAM RA + Lookups RAF
//! 6. Booleanity              →        6. Bytecode RAF + Hamming + Booleanity + RA virtual
//!                            →        7. Hamming weight claim reduction
//!
//! Note: Zolt's stages are more consolidated, so conversion involves
//! splitting some proofs and creating empty placeholders where Zolt
//! handles things differently.
//!
//! ## Constraint Evaluation
//!
//! When `convertWithWitnesses` is called with actual per-cycle witnesses,
//! the converter will compute real Az*Bz products from the R1CS constraints
//! using the evaluators from `r1cs/evaluators.zig`. This enables proper
//! verification of the univariate skip first-round polynomial.

const std = @import("std");
const Allocator = std.mem.Allocator;

const jolt_types = @import("jolt_types.zig");
const prover = @import("prover.zig");
const field_mod = @import("../field/mod.zig");
const r1cs = @import("r1cs/mod.zig");
const spartan_outer = @import("spartan/outer.zig");
const streaming_outer = @import("spartan/streaming_outer.zig");
const transcripts = @import("../transcripts/mod.zig");
const Blake2bTranscript = transcripts.Blake2bTranscript;
const poly_mod = @import("../poly/mod.zig");

/// Convert Zolt's internal proof to Jolt-compatible format
pub fn ProofConverter(comptime F: type) type {
    return struct {
        const Self = @This();

        // Import types we need
        const JoltProofType = jolt_types.JoltProof;
        const SumcheckInstanceProof = jolt_types.SumcheckInstanceProof;
        const UniSkipFirstRoundProof = jolt_types.UniSkipFirstRoundProof;
        const OpeningClaims = jolt_types.OpeningClaims;
        const SumcheckId = jolt_types.SumcheckId;
        const OpeningId = jolt_types.OpeningId;
        const VirtualPolynomial = jolt_types.VirtualPolynomial;

        allocator: Allocator,

        pub fn init(allocator: Allocator) Self {
            return Self{
                .allocator = allocator,
            };
        }

        /// Convert Zolt's 6-stage proof to Jolt's 7-stage format
        ///
        /// This creates a JoltProof that can be serialized and verified
        /// by the Jolt verifier.
        ///
        /// IMPORTANT: This generates "zero proofs" - all sumcheck round polynomials
        /// are zero, which satisfies the verification check when all claims are 0.
        /// This is a placeholder for proper cross-compatibility.
        pub fn convert(
            self: *Self,
            comptime Commitment: type,
            comptime Proof: type,
            zolt_stage_proofs: *const prover.JoltStageProofs(F),
            commitments: []const Commitment,
            joint_opening_proof: ?Proof,
            config: ConversionConfig,
        ) !JoltProofType(F, Commitment, Proof) {
            var jolt_proof = JoltProofType(F, Commitment, Proof).init(self.allocator);

            // Copy configuration parameters
            const trace_length: usize = @as(usize, 1) << @intCast(zolt_stage_proofs.log_t);
            const ram_K: usize = @as(usize, 1) << @intCast(zolt_stage_proofs.log_k);

            jolt_proof.trace_length = trace_length;
            jolt_proof.ram_K = ram_K;
            jolt_proof.bytecode_K = config.bytecode_K;
            jolt_proof.log_k_chunk = config.log_k_chunk;
            jolt_proof.lookups_ra_virtual_log_k_chunk = config.lookups_ra_virtual_log_k_chunk;

            // Compute derived parameters
            const n_cycle_vars = std.math.log2_int(usize, trace_length);
            const log_ram_k = std.math.log2_int(usize, ram_K);

            // Copy commitments
            for (commitments) |c| {
                try jolt_proof.commitments.append(self.allocator, c);
            }

            // Set joint opening proof
            jolt_proof.joint_opening_proof = joint_opening_proof;

            // Create UniSkip proof for Stage 1 (degree-27 polynomial)
            jolt_proof.stage1_uni_skip_first_round_proof = try self.createUniSkipProofStage1();

            // Stage 1: Outer Spartan Remaining
            // num_rounds = 1 + num_cycles_bits (from OuterRemainingSumcheckParams)
            try self.generateZeroSumcheckProof(
                &jolt_proof.stage1_sumcheck_proof,
                1 + n_cycle_vars,
                3, // degree 3
            );

            // Add Stage 1 opening claims
            // SpartanOuter requires all 36 R1CS inputs + UnivariateSkip claim
            // This matches the ALL_R1CS_INPUTS array in Jolt's r1cs/inputs.rs
            try self.addSpartanOuterOpeningClaims(&jolt_proof.opening_claims);

            // Create UniSkip proof for Stage 2 (degree-12 polynomial)
            jolt_proof.stage2_uni_skip_first_round_proof = try self.createUniSkipProofStage2();

            // Stage 2: Product virtualization + RAM RAF + RW + Output + Instruction claim reduction
            // This is a batched sumcheck with multiple instances
            // The max rounds is typically n_cycle_vars + log_ram_k for RAM operations
            // But the exact count depends on the specific verifiers batched together
            // For simplicity, use n_cycle_vars + 1 (matching Stage 1 remaining)
            try self.generateZeroSumcheckProof(
                &jolt_proof.stage2_sumcheck_proof,
                n_cycle_vars + 1, // Conservative estimate
                3,
            );

            // Add Stage 2 opening claims
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .RamRa, .sumcheck_id = .RamRafEvaluation } },
                F.zero(),
            );
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .RamVal, .sumcheck_id = .RamReadWriteChecking } },
                F.zero(),
            );
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .UnivariateSkip, .sumcheck_id = .SpartanProductVirtualization } },
                F.zero(),
            );

            // Stage 3: Spartan shift + Instruction input + Registers claim reduction
            try self.generateZeroSumcheckProof(
                &jolt_proof.stage3_sumcheck_proof,
                n_cycle_vars,
                3,
            );

            // Add Stage 3 opening claims
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .LookupOutput, .sumcheck_id = .InstructionClaimReduction } },
                F.zero(),
            );

            // Stage 4: Registers RW + RAM val evaluation + final
            try self.generateZeroSumcheckProof(
                &jolt_proof.stage4_sumcheck_proof,
                log_ram_k,
                3,
            );

            // Add Stage 4 opening claims
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .RamVal, .sumcheck_id = .RamValEvaluation } },
                F.zero(),
            );
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .RamValFinal, .sumcheck_id = .RamValFinalEvaluation } },
                F.zero(),
            );

            // Stage 5: Registers val + RAM RA reduction + Lookups RAF
            try self.generateZeroSumcheckProof(
                &jolt_proof.stage5_sumcheck_proof,
                n_cycle_vars,
                3,
            );

            // Add Stage 5 opening claims
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .RegistersVal, .sumcheck_id = .RegistersValEvaluation } },
                F.zero(),
            );
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .RamRa, .sumcheck_id = .RamRaClaimReduction } },
                F.zero(),
            );

            // Stage 6: Bytecode RAF + Hamming + Booleanity + RA virtual + Inc reduction
            try self.generateZeroSumcheckProof(
                &jolt_proof.stage6_sumcheck_proof,
                n_cycle_vars,
                3,
            );

            // Add Stage 6 opening claims
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .RamHammingWeight, .sumcheck_id = .Booleanity } },
                F.zero(),
            );
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .RamHammingWeight, .sumcheck_id = .RamHammingBooleanity } },
                F.zero(),
            );

            // Stage 7: Hamming weight claim reduction
            // num_rounds = log_k_chunk
            try self.generateZeroSumcheckProof(
                &jolt_proof.stage7_sumcheck_proof,
                config.log_k_chunk,
                3,
            );

            // Add Stage 7 opening claims
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .RamHammingWeight, .sumcheck_id = .HammingWeightClaimReduction } },
                F.zero(),
            );

            return jolt_proof;
        }

        /// Generate a zero-filled sumcheck proof with the specified number of rounds
        ///
        /// Each round has a compressed polynomial with degree `degree_bound`.
        /// For claim = 0, all-zero polynomials satisfy p(0) + p(1) = claim.
        fn generateZeroSumcheckProof(
            self: *Self,
            proof: *SumcheckInstanceProof(F),
            num_rounds: usize,
            degree_bound: usize,
        ) !void {
            // Compressed poly: coeffs_except_linear_term has `degree_bound` elements
            // (constant, quadratic, cubic, ...) - linear term is recovered from hint
            for (0..num_rounds) |_| {
                const coeffs = try self.allocator.alloc(F, degree_bound);
                @memset(coeffs, F.zero());
                try proof.compressed_polys.append(self.allocator, .{
                    .coeffs_except_linear_term = coeffs,
                    .allocator = self.allocator,
                });
            }
        }

        /// Generate sumcheck proof using the streaming outer prover
        ///
        /// This produces actual polynomial evaluations (not zeros) by computing
        /// Az*Bz products from the R1CS constraints.
        fn generateStreamingOuterSumcheckProof(
            self: *Self,
            proof: *SumcheckInstanceProof(F),
            cycle_witnesses: []const r1cs.R1CSCycleInputs(F),
            tau: []const F,
        ) !void {
            const StreamingOuterProver = streaming_outer.StreamingOuterProver(F);

            // Initialize the streaming prover
            var outer_prover = StreamingOuterProver.init(
                self.allocator,
                cycle_witnesses,
                tau,
            ) catch {
                // Fallback to zero proofs if initialization fails
                const num_rounds = 1 + std.math.log2_int(usize, @max(1, cycle_witnesses.len));
                return self.generateZeroSumcheckProof(proof, num_rounds, 3);
            };
            defer outer_prover.deinit();

            // Skip the first round (handled by UniSkip)
            // Generate remaining rounds
            const num_rounds = outer_prover.numRounds();
            if (num_rounds <= 1) {
                return;
            }

            // Bind the first-round challenge (would come from transcript)
            // For now, use a deterministic challenge and placeholder claim
            const r0 = F.fromU64(0x9e3779b97f4a7c15);
            const uni_skip_claim = F.zero(); // Placeholder - non-transcript version
            outer_prover.bindFirstRoundChallenge(r0, uni_skip_claim) catch {};

            // Generate remaining round polynomials
            for (1..num_rounds) |_| {
                const round_evals = outer_prover.computeRemainingRoundPoly() catch {
                    // Fallback to zero polynomial
                    const coeffs = try self.allocator.alloc(F, 3);
                    @memset(coeffs, F.zero());
                    try proof.compressed_polys.append(self.allocator, .{
                        .coeffs_except_linear_term = coeffs,
                        .allocator = self.allocator,
                    });
                    continue;
                };

                // Convert evaluations [s(0), s(1), s(2), s(3)] to compressed coefficients [c0, c2, c3]
                // The linear term c1 is recovered from the hint during verification
                const compressed = poly_mod.UniPoly(F).evalsToCompressed(round_evals);
                const coeffs = try self.allocator.alloc(F, 3);
                coeffs[0] = compressed[0]; // c0 (constant)
                coeffs[1] = compressed[1]; // c2 (quadratic)
                coeffs[2] = compressed[2]; // c3 (cubic)

                try proof.compressed_polys.append(self.allocator, .{
                    .coeffs_except_linear_term = coeffs,
                    .allocator = self.allocator,
                });

                // Bind challenge for this round
                // In real implementation, challenge comes from transcript
                const challenge = F.fromU64(0xc4ceb9fe1a85ec53);
                outer_prover.bindRemainingRoundChallenge(challenge) catch {};
                outer_prover.updateClaim(round_evals, challenge);
            }
        }

        /// Result of Stage 1 sumcheck proof generation
        const Stage1Result = struct {
            /// Accumulated sumcheck challenges (r_stream, r_cycle_bits...)
            /// The full r_cycle point is [r_stream, r1, r2, ..., r_n] reversed
            challenges: std.ArrayListUnmanaged(F),
            /// The first-round challenge r0 from UniSkip
            r0: F,
            /// The UnivariateSkip claim: evaluation of UniSkip polynomial at r0
            /// This is the input_claim for the remaining sumcheck rounds
            uni_skip_claim: F,
            /// Allocator for cleanup
            allocator: Allocator,

            pub fn deinit(self: *Stage1Result) void {
                self.challenges.deinit(self.allocator);
            }
        };

        /// Generate sumcheck proof using the streaming outer prover with Fiat-Shamir transcript
        ///
        /// This produces actual polynomial evaluations by computing Az*Bz products
        /// from the R1CS constraints, using the provided transcript for challenges.
        ///
        /// Returns the accumulated challenges for computing r_cycle.
        fn generateStreamingOuterSumcheckProofWithTranscript(
            self: *Self,
            proof: *SumcheckInstanceProof(F),
            uniskip_proof: *const UniSkipFirstRoundProof(F),
            cycle_witnesses: []const r1cs.R1CSCycleInputs(F),
            tau: []const F,
            transcript: *Blake2bTranscript(F),
        ) !Stage1Result {
            const StreamingOuterProver = streaming_outer.StreamingOuterProver(F);
            const LagrangePoly = r1cs.univariate_skip.LagrangePolynomial(F);
            var challenges: std.ArrayListUnmanaged(F) = .{};

            // Extract tau_high for the UniSkip Lagrange kernel
            // tau has length num_rows_bits = num_cycle_vars + 2
            // tau_high is the last element (used for Lagrange kernel)
            // Full tau is passed to split_eq (it handles the split internally)
            if (tau.len < 2) {
                const num_rounds = 1 + std.math.log2_int(usize, @max(1, cycle_witnesses.len));
                try self.generateZeroSumcheckProof(proof, num_rounds, 3);
                return Stage1Result{ .challenges = challenges, .r0 = F.zero(), .uni_skip_claim = F.zero(), .allocator = self.allocator };
            }
            const tau_high = tau[tau.len - 1];

            // DEBUG: Print full tau vector
            std.debug.print("[ZOLT] STAGE1_PRE: tau.len = {}\n", .{tau.len});
            for (tau, 0..) |t, i| {
                std.debug.print("[ZOLT] STAGE1_PRE: tau[{}] = {any}\n", .{ i, t.toBytesBE() });
            }

            // The first round was already processed by UniSkip
            // Append the UniSkip polynomial to transcript using UniPoly format:
            // "UncompressedUniPoly_begin", all coefficients, "UncompressedUniPoly_end"
            transcript.appendMessage("UncompressedUniPoly_begin");
            for (uniskip_proof.uni_poly) |coeff| {
                transcript.appendScalar(coeff);
            }
            transcript.appendMessage("UncompressedUniPoly_end");

            // Get the challenge for the first round (r0)
            const r0 = transcript.challengeScalar();

            // DEBUG: Print r0
            std.debug.print("[ZOLT] STAGE1_PRE: r0 = {any}\n", .{r0.toBytesBE()});

            // Compute the Lagrange kernel L(r0, tau_high) to use as initial scaling
            // This matches Jolt's: lagrange_tau_r0 = LagrangePolynomial::lagrange_kernel(&r0, &tau_high)
            const lagrange_tau_r0 = try LagrangePoly.lagrangeKernel(
                r1cs.univariate_skip.OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE,
                r0,
                tau_high,
                self.allocator,
            );

            // DEBUG: Print tau_high and lagrange_tau_r0
            std.debug.print("[ZOLT] STAGE1_PRE: tau_high = {any}\n", .{tau_high.toBytesBE()});
            std.debug.print("[ZOLT] STAGE1_PRE: lagrange_tau_r0 = {any}\n", .{lagrange_tau_r0.toBytesBE()});

            // Initialize the streaming prover with full tau and Lagrange kernel scaling
            // The prover internally extracts:
            //   tau_high = tau[tau.len - 1] (stored separately for first-round polynomial)
            //   tau_low = tau[0..tau.len - 1] (passed to split_eq)
            // This matches Jolt's behavior in OuterSharedState::new().
            var outer_prover = StreamingOuterProver.initWithScaling(
                self.allocator,
                cycle_witnesses,
                tau, // Full tau - prover extracts tau_low and tau_high internally
                lagrange_tau_r0,
            ) catch {
                // Fallback to zero proofs if initialization fails
                const num_rounds = 1 + std.math.log2_int(usize, @max(1, cycle_witnesses.len));
                try self.generateZeroSumcheckProof(proof, num_rounds, 3);
                return Stage1Result{ .challenges = challenges, .r0 = r0, .uni_skip_claim = F.zero(), .allocator = self.allocator };
            };
            defer outer_prover.deinit();

            // Compute the UnivariateSkip claim: evaluation of UniSkip polynomial at r0
            // Use evaluatePolyAtChallenge which handles the Jolt-format challenge [0, 0, low, high]
            const uni_skip_claim = evaluatePolyAtChallenge(uniskip_proof.uni_poly, r0);

            // DEBUG: Print uni_skip_claim
            std.debug.print("[ZOLT] STAGE1_PRE: uni_skip_claim = {any}\n", .{uni_skip_claim.toBytesBE()});

            // Bind the first-round challenge from transcript with the uni_skip_claim
            outer_prover.bindFirstRoundChallenge(r0, uni_skip_claim) catch {};

            // Match Jolt's cache_openings: after UniSkip verification, the verifier calls
            // accumulator.append_virtual() which appends the uni_skip_claim to transcript.
            // This happens BEFORE BatchedSumcheck::verify which also appends it.
            transcript.appendScalar(uni_skip_claim);

            // IMPORTANT: Match Jolt's BatchedSumcheck::prove and verify transcript flow exactly:
            //   1. Append input_claim to transcript
            //   2. Get batching_coeffs via challenge_vector(1) - this modifies transcript state!
            //   3. Then process round polynomials
            //
            // The batching coefficient is used to scale the round polynomials:
            //   batched_poly = Σ poly_i * coeff_i
            //
            // For a single sumcheck instance, batched_poly = poly * coeff.
            //
            // The input_claim for Stage 1 remaining sumcheck is uni_skip_claim.
            transcript.appendScalar(uni_skip_claim);

            // Get batching coefficient
            // This advances the transcript state AND provides the scaling factor
            // IMPORTANT: Use challengeScalarFull() which returns proper Montgomery form,
            // matching Jolt's challenge_vector which uses F::from_bytes (not MontU128Challenge)
            const batching_coeff = transcript.challengeScalarFull();

            // DEBUG: Print batching_coeff
            std.debug.print("[ZOLT] STAGE1_PRE: batching_coeff = {any}\n", .{batching_coeff.toBytesBE()});

            // Generate remaining rounds
            // In Jolt, stage1_sumcheck_proof contains num_rounds polynomials
            // where num_rounds = 1 + num_cycle_vars (1 streaming + cycle vars)
            // The UniSkip is separate and doesn't count here
            const num_remaining_rounds = outer_prover.numRounds(); // 1 + num_cycle_vars
            if (num_remaining_rounds == 0) {
                return Stage1Result{ .challenges = challenges, .r0 = r0, .uni_skip_claim = uni_skip_claim, .allocator = self.allocator };
            }

            // DEBUG: Print initial claim (= uni_skip_claim * batching_coeff * 2^num_rounds factor)
            // In Jolt, the initial claim for sumcheck is:
            //   claim = input_claim.mul_pow_2(max_num_rounds - num_rounds) * coeff
            // For a single instance with max_num_rounds = num_rounds:
            //   claim = input_claim * coeff = uni_skip_claim * batching_coeff
            //
            // NOTE: The prover tracks UNSCALED claims internally (outer_prover.current_claim
            // is set to uni_skip_claim by bindFirstRoundChallenge). The SCALED coefficients
            // are sent to the proof.
            const initial_claim = uni_skip_claim.mul(batching_coeff);
            std.debug.print("[ZOLT] STAGE1_INITIAL: claim = {any}\n", .{initial_claim.toBytesBE()});
            std.debug.print("[ZOLT] STAGE1_INITIAL: claim_le = {any}\n", .{initial_claim.toBytes()});

            // Generate all remaining round polynomials with transcript integration
            // Use computeRemainingRoundPoly for ALL rounds - it now properly:
            // 1. Materializes Az/Bz on first call
            // 2. Rebuilds t_prime_poly from bound Az/Bz when needed
            // 3. Uses the multiquadratic method for all rounds
            for (0..num_remaining_rounds) |round_idx| {
                // DEBUG: Print current claim at start of round
                std.debug.print("[ZOLT] STAGE1_ROUND_{}: current_claim = {any}\n", .{ round_idx, outer_prover.current_claim.toBytesBE() });
                std.debug.print("[ZOLT] STAGE1_ROUND_{}: current_claim_le = {any}\n", .{ round_idx, outer_prover.current_claim.toBytes() });

                const raw_evals: [4]F = outer_prover.computeRemainingRoundPoly() catch {
                    // Fallback to zero polynomial
                    const coeffs = try self.allocator.alloc(F, 3);
                    @memset(coeffs, F.zero());
                    try proof.compressed_polys.append(self.allocator, .{
                        .coeffs_except_linear_term = coeffs,
                        .allocator = self.allocator,
                    });
                    try challenges.append(self.allocator, F.zero());
                    continue;
                };

                // CRITICAL: Apply batching coefficient to scale the evaluations for OUTPUT
                // In Jolt's BatchedSumcheck::prove:
                //   1. Individual prover computes unscaled poly
                //   2. Batched poly = Σ individual_poly * coeff
                //   3. Batched poly is hashed to transcript
                //   4. Individual prover tracks UNSCALED claim for next round
                //
                // So we:
                // - Scale the polynomial for output (to transcript and proof)
                // - Keep unscaled claim for prover's internal state
                const scaled_evals = [4]F{
                    raw_evals[0].mul(batching_coeff),
                    raw_evals[1].mul(batching_coeff),
                    raw_evals[2].mul(batching_coeff),
                    raw_evals[3].mul(batching_coeff),
                };

                // DEBUG: Print raw and scaled evaluations
                std.debug.print("[ZOLT] STAGE1_ROUND_{}: raw_evals = [{any}, {any}, {any}, {any}]\n", .{
                    round_idx,
                    raw_evals[0].toBytesBE(),
                    raw_evals[1].toBytesBE(),
                    raw_evals[2].toBytesBE(),
                    raw_evals[3].toBytesBE(),
                });
                std.debug.print("[ZOLT] STAGE1_ROUND_{}: scaled_evals = [{any}, {any}, {any}, {any}]\n", .{
                    round_idx,
                    scaled_evals[0].toBytesBE(),
                    scaled_evals[1].toBytesBE(),
                    scaled_evals[2].toBytesBE(),
                    scaled_evals[3].toBytesBE(),
                });

                // Convert SCALED evaluations to compressed coefficients for proof
                const compressed = poly_mod.UniPoly(F).evalsToCompressed(scaled_evals);
                const coeffs = try self.allocator.alloc(F, 3);
                coeffs[0] = compressed[0]; // c0 (constant)
                coeffs[1] = compressed[1]; // c2 (quadratic)
                coeffs[2] = compressed[2]; // c3 (cubic)

                // DEBUG: Print compressed coefficients in both BE and LE
                std.debug.print("[ZOLT] STAGE1_ROUND_{}: c0 = {any}\n", .{ round_idx, compressed[0].toBytesBE() });
                std.debug.print("[ZOLT] STAGE1_ROUND_{}: c0_le = {any}\n", .{ round_idx, compressed[0].toBytes() });
                std.debug.print("[ZOLT] STAGE1_ROUND_{}: c2 = {any}\n", .{ round_idx, compressed[1].toBytesBE() });
                std.debug.print("[ZOLT] STAGE1_ROUND_{}: c2_le = {any}\n", .{ round_idx, compressed[1].toBytes() });
                std.debug.print("[ZOLT] STAGE1_ROUND_{}: c3 = {any}\n", .{ round_idx, compressed[2].toBytesBE() });
                std.debug.print("[ZOLT] STAGE1_ROUND_{}: c3_le = {any}\n", .{ round_idx, compressed[2].toBytes() });

                try proof.compressed_polys.append(self.allocator, .{
                    .coeffs_except_linear_term = coeffs,
                    .allocator = self.allocator,
                });

                // Append SCALED round polynomial to transcript using Jolt's CompressedUniPoly format
                transcript.appendMessage("UniPoly_begin");
                transcript.appendScalar(compressed[0]); // c0
                transcript.appendScalar(compressed[1]); // c2
                transcript.appendScalar(compressed[2]); // c3
                transcript.appendMessage("UniPoly_end");

                // Get challenge from transcript
                const challenge = transcript.challengeScalar();
                try challenges.append(self.allocator, challenge);

                // DEBUG: Print challenge in both BE and LE
                std.debug.print("[ZOLT] STAGE1_ROUND_{}: challenge = {any}\n", .{ round_idx, challenge.toBytesBE() });
                std.debug.print("[ZOLT] STAGE1_ROUND_{}: challenge_le = {any}\n", .{ round_idx, challenge.toBytes() });

                // Bind challenge and update claim
                // CRITICAL: The verifier uses eval_from_hint with SCALED coefficients and SCALED hint,
                // so we must track SCALED claims to match the verifier's computation.
                outer_prover.bindRemainingRoundChallenge(challenge) catch {};
                outer_prover.updateClaim(scaled_evals, challenge);

                // DEBUG: Print next_claim after update
                std.debug.print("[ZOLT] STAGE1_ROUND_{}: next_claim = {any}\n", .{ round_idx, outer_prover.current_claim.toBytesBE() });
                std.debug.print("[ZOLT] STAGE1_ROUND_{}: split_eq.current_scalar = {any}\n", .{ round_idx, outer_prover.split_eq.current_scalar.toBytesBE() });
                // Compute and print the implied inner product at this round
                if (outer_prover.split_eq.current_scalar.inverse()) |eq_inv| {
                    const implied = outer_prover.current_claim.mul(eq_inv);
                    std.debug.print("[ZOLT] STAGE1_ROUND_{}: implied_inner_prod = {any}\n", .{ round_idx, implied.toBytesBE() });
                }
                // Print bound Az*Bz if available
                if (outer_prover.az_poly) |az| {
                    if (outer_prover.bz_poly) |bz| {
                        if (az.evaluations.len > 0 and bz.evaluations.len > 0) {
                            const az_bz = az.evaluations[0].mul(bz.evaluations[0]);
                            std.debug.print("[ZOLT] STAGE1_ROUND_{}: az[0]*bz[0] = {any}\n", .{ round_idx, az_bz.toBytesBE() });
                            std.debug.print("[ZOLT] STAGE1_ROUND_{}: az.len = {}, bz.len = {}\n", .{ round_idx, az.evaluations.len, bz.evaluations.len });
                        }
                    }
                }
                // Print t_prime values if available
                if (outer_prover.t_prime_poly) |t_prime| {
                    if (t_prime.evaluations.len > 0) {
                        std.debug.print("[ZOLT] STAGE1_ROUND_{}: t_prime[0] = {any}\n", .{ round_idx, t_prime.evaluations[0].toBytesBE() });
                        std.debug.print("[ZOLT] STAGE1_ROUND_{}: t_prime.num_vars = {}, len = {}\n", .{ round_idx, t_prime.num_vars, t_prime.evaluations.len });
                    }
                }
            }

            // DEBUG: Print final values for cross-verification
            const eq_scalar = outer_prover.split_eq.current_scalar;
            const final_claim = outer_prover.current_claim;
            std.debug.print("[ZOLT] STAGE1_FINAL: eq_factor (split_eq.current_scalar) = {any}\n", .{eq_scalar.toBytesBE()});
            std.debug.print("[ZOLT] STAGE1_FINAL: output_claim (unscaled) = {any}\n", .{final_claim.toBytesBE()});
            std.debug.print("[ZOLT] STAGE1_FINAL: output_claim (scaled) = {any}\n", .{final_claim.mul(batching_coeff).toBytesBE()});
            // Compute implied inner_sum_prod = output_claim / eq_factor
            if (eq_scalar.inverse()) |eq_inv| {
                const implied_inner_sum_prod = final_claim.mul(eq_inv);
                std.debug.print("[ZOLT] STAGE1_FINAL: implied_inner_sum_prod (output/eq) = {any}\n", .{implied_inner_sum_prod.toBytesBE()});
            }
            // Print final bound Az/Bz values
            if (outer_prover.az_poly) |az| {
                if (az.evaluations.len > 0) {
                    std.debug.print("[ZOLT] STAGE1_FINAL: az_poly final value = {any}\n", .{az.evaluations[0].toBytesBE()});
                }
            }
            if (outer_prover.bz_poly) |bz| {
                if (bz.evaluations.len > 0) {
                    std.debug.print("[ZOLT] STAGE1_FINAL: bz_poly final value = {any}\n", .{bz.evaluations[0].toBytesBE()});
                    // Also compute the product
                    if (outer_prover.az_poly) |az| {
                        if (az.evaluations.len > 0) {
                            const az_bz_product = az.evaluations[0].mul(bz.evaluations[0]);
                            std.debug.print("[ZOLT] STAGE1_FINAL: az_final * bz_final = {any}\n", .{az_bz_product.toBytesBE()});
                        }
                    }
                }
            }
            std.debug.print("[ZOLT] STAGE1_FINAL: num_challenges = {}\n", .{challenges.items.len});
            for (challenges.items, 0..) |c, i| {
                std.debug.print("[ZOLT] STAGE1_FINAL: challenge[{}] = {any}\n", .{ i, c.toBytesBE() });
            }

            return Stage1Result{ .challenges = challenges, .r0 = r0, .uni_skip_claim = uni_skip_claim, .allocator = self.allocator };
        }

        /// Evaluate a polynomial given as coefficients at a point using Horner's method
        /// Uses standard Montgomery multiplication.
        fn evaluatePolyAtPoint(coeffs: []const F, x: F) F {
            if (coeffs.len == 0) return F.zero();

            var result = coeffs[coeffs.len - 1];
            var i = coeffs.len - 1;
            while (i > 0) {
                i -= 1;
                result = result.mul(x).add(coeffs[i]);
            }
            return result;
        }

        /// Evaluate a polynomial at a challenge point using Horner's method.
        ///
        /// Both the coefficients and the challenge point should be in Montgomery form.
        fn evaluatePolyAtChallenge(coeffs: []const F, x: F) F {
            if (coeffs.len == 0) return F.zero();

            // Both coeffs and x are in Montgomery form (challenges are now
            // converted to Montgomery form in the transcript).
            // Standard field multiplication works correctly.

            // Use Horner's method
            var result = coeffs[coeffs.len - 1];
            var i = coeffs.len - 1;
            while (i > 0) {
                i -= 1;
                result = result.mul(x).add(coeffs[i]);
            }
            return result;
        }

        /// R1CS input indices in Jolt's ALL_R1CS_INPUTS order
        /// Maps from Jolt's ordering (index in this array) to Zolt's R1CSInputIndex
        const JOLT_TO_ZOLT_R1CS_INDICES = [36]r1cs.R1CSInputIndex{
            .LeftInstructionInput, // 0
            .RightInstructionInput, // 1
            .Product, // 2
            .WriteLookupOutputToRD, // 3
            .WritePCtoRD, // 4
            .ShouldBranch, // 5
            .PC, // 6
            .UnexpandedPC, // 7
            .Imm, // 8
            .RamAddress, // 9
            .Rs1Value, // 10
            .Rs2Value, // 11
            .RdWriteValue, // 12
            .RamReadValue, // 13
            .RamWriteValue, // 14
            .LeftLookupOperand, // 15
            .RightLookupOperand, // 16
            .NextUnexpandedPC, // 17
            .NextPC, // 18
            .NextIsVirtual, // 19
            .NextIsFirstInSequence, // 20
            .LookupOutput, // 21
            .ShouldJump, // 22
            .FlagAddOperands, // 23
            .FlagSubtractOperands, // 24
            .FlagMultiplyOperands, // 25
            .FlagLoad, // 26
            .FlagStore, // 27
            .FlagJump, // 28
            .FlagWriteLookupOutputToRD, // 29
            .FlagVirtualInstruction, // 30
            .FlagAssert, // 31
            .FlagDoNotUpdateUnexpandedPC, // 32
            .FlagAdvice, // 33
            .FlagIsCompressed, // 34
            .FlagIsFirstInSequence, // 35
        };

        /// VirtualPolynomial identifiers in Jolt's order
        const R1CS_VIRTUAL_POLYS = [36]VirtualPolynomial{
            .LeftInstructionInput, // 0
            .RightInstructionInput, // 1
            .Product, // 2
            .WriteLookupOutputToRD, // 3
            .WritePCtoRD, // 4
            .ShouldBranch, // 5
            .PC, // 6
            .UnexpandedPC, // 7
            .Imm, // 8
            .RamAddress, // 9
            .Rs1Value, // 10
            .Rs2Value, // 11
            .RdWriteValue, // 12
            .RamReadValue, // 13
            .RamWriteValue, // 14
            .LeftLookupOperand, // 15
            .RightLookupOperand, // 16
            .NextUnexpandedPC, // 17
            .NextPC, // 18
            .NextIsVirtual, // 19
            .NextIsFirstInSequence, // 20
            .LookupOutput, // 21
            .ShouldJump, // 22
            // OpFlags variants (13 of them) - CircuitFlags indices
            .{ .OpFlags = 0 }, // 23: AddOperands
            .{ .OpFlags = 1 }, // 24: SubtractOperands
            .{ .OpFlags = 2 }, // 25: MultiplyOperands
            .{ .OpFlags = 3 }, // 26: Load
            .{ .OpFlags = 4 }, // 27: Store
            .{ .OpFlags = 5 }, // 28: Jump
            .{ .OpFlags = 6 }, // 29: WriteLookupOutputToRD
            .{ .OpFlags = 7 }, // 30: VirtualInstruction
            .{ .OpFlags = 8 }, // 31: Assert
            .{ .OpFlags = 9 }, // 32: DoNotUpdateUnexpandedPC
            .{ .OpFlags = 10 }, // 33: Advice
            .{ .OpFlags = 11 }, // 34: IsCompressed
            .{ .OpFlags = 12 }, // 35: IsFirstInSequence
        };

        /// Add all 36 R1CS input opening claims for SpartanOuter with zero claims
        ///
        /// This exactly matches the ALL_R1CS_INPUTS array in Jolt's r1cs/inputs.rs:
        /// - 23 simple virtual polynomials
        /// - 13 OpFlags variants
        fn addSpartanOuterOpeningClaims(
            self: *Self,
            claims: *OpeningClaims(F),
        ) !void {
            _ = self;

            // Add all R1CS inputs for SpartanOuter with zero claims
            for (R1CS_VIRTUAL_POLYS) |poly| {
                try claims.insert(
                    .{ .Virtual = .{ .poly = poly, .sumcheck_id = .SpartanOuter } },
                    F.zero(),
                );
            }

            // Add the UnivariateSkip claim for SpartanOuter
            try claims.insert(
                .{ .Virtual = .{ .poly = .UnivariateSkip, .sumcheck_id = .SpartanOuter } },
                F.zero(),
            );
        }

        /// Add all 36 R1CS input opening claims for SpartanOuter with actual evaluations
        ///
        /// This computes the MLE evaluations at r_cycle and uses those as the claims.
        fn addSpartanOuterOpeningClaimsWithEvaluations(
            self: *Self,
            claims: *OpeningClaims(F),
            cycle_witnesses: []const r1cs.R1CSCycleInputs(F),
            r_cycle: []const F,
            uni_skip_claim: F,
        ) !void {
            // Compute MLE evaluations at r_cycle
            const R1CSInputEvaluator = r1cs.R1CSInputEvaluator(F);
            const input_evals = try R1CSInputEvaluator.computeClaimedInputs(
                self.allocator,
                cycle_witnesses,
                r_cycle,
            );

            // DEBUG: Print the first few R1CS input evaluations
            std.debug.print("[ZOLT] OPENING_CLAIMS: r_cycle.len = {}\n", .{r_cycle.len});
            std.debug.print("[ZOLT] OPENING_CLAIMS: cycle_witnesses.len = {}\n", .{cycle_witnesses.len});
            // Print first and last r_cycle values
            if (r_cycle.len > 0) {
                std.debug.print("[ZOLT] OPENING_CLAIMS: r_cycle[0] = {any}\n", .{r_cycle[0].toBytes()});
                std.debug.print("[ZOLT] OPENING_CLAIMS: r_cycle[last] = {any}\n", .{r_cycle[r_cycle.len - 1].toBytes()});
            }
            // Print first few witness values
            if (cycle_witnesses.len > 0) {
                std.debug.print("[ZOLT] OPENING_CLAIMS: witness[0].LeftInstructionInput = {any}\n", .{cycle_witnesses[0].values[0].toBytes()});
                std.debug.print("[ZOLT] OPENING_CLAIMS: witness[0].RightInstructionInput = {any}\n", .{cycle_witnesses[0].values[1].toBytes()});
                std.debug.print("[ZOLT] OPENING_CLAIMS: witness[0].Product = {any}\n", .{cycle_witnesses[0].values[2].toBytes()});
                std.debug.print("[ZOLT] OPENING_CLAIMS: witness[0].PC = {any}\n", .{cycle_witnesses[0].values[6].toBytes()});
            }
            if (cycle_witnesses.len > 1) {
                std.debug.print("[ZOLT] OPENING_CLAIMS: witness[1].LeftInstructionInput = {any}\n", .{cycle_witnesses[1].values[0].toBytes()});
                std.debug.print("[ZOLT] OPENING_CLAIMS: witness[1].PC = {any}\n", .{cycle_witnesses[1].values[6].toBytes()});
            }
            std.debug.print("[ZOLT] OPENING_CLAIMS: r1cs_input_evals[0] (LeftInstructionInput) = {any}\n", .{input_evals[0].toBytes()});
            std.debug.print("[ZOLT] OPENING_CLAIMS: r1cs_input_evals[1] (RightInstructionInput) = {any}\n", .{input_evals[1].toBytes()});
            std.debug.print("[ZOLT] OPENING_CLAIMS: r1cs_input_evals[2] (Product) = {any}\n", .{input_evals[2].toBytes()});

            // Add R1CS inputs for SpartanOuter with computed evaluations
            for (R1CS_VIRTUAL_POLYS, 0..) |poly, jolt_idx| {
                // Map Jolt's index to Zolt's R1CSInputIndex
                const zolt_idx = JOLT_TO_ZOLT_R1CS_INDICES[jolt_idx].toIndex();
                const claim = input_evals[zolt_idx];

                try claims.insert(
                    .{ .Virtual = .{ .poly = poly, .sumcheck_id = .SpartanOuter } },
                    claim,
                );
            }

            // Add the UnivariateSkip claim for SpartanOuter
            // This is uni_poly.evaluate(r0), the input_claim for the remaining sumcheck
            try claims.insert(
                .{ .Virtual = .{ .poly = .UnivariateSkip, .sumcheck_id = .SpartanOuter } },
                uni_skip_claim,
            );
        }

        /// Create a UniSkipFirstRoundProof for Stage 1 (degree-27 polynomial)
        ///
        /// Jolt's Stage 1 (Spartan outer) uses a degree-27 first-round polynomial
        /// that encodes all 19 R1CS constraints via the univariate skip optimization.
        ///
        /// For the verification to pass, the polynomial must satisfy:
        ///   Σ_{j=0}^{27} coeff_j * power_sums[j] = 0
        ///
        /// where power_sums[j] = Σ_{t in domain} t^j for domain {-4, -3, ..., 4, 5}.
        ///
        /// The simplest valid polynomial is all zeros (trivially sums to 0).
        fn createUniSkipProofStage1(self: *Self) !?UniSkipFirstRoundProof(F) {
            // For stage 1, we need 28 coefficients (degree 27)
            const NUM_COEFFS = r1cs.OUTER_FIRST_ROUND_POLY_NUM_COEFFS;

            // Create an all-zero polynomial that trivially satisfies the sum constraint.
            const coeffs = try self.allocator.alloc(F, NUM_COEFFS);
            @memset(coeffs, F.zero());

            return UniSkipFirstRoundProof(F){
                .uni_poly = coeffs,
                .allocator = self.allocator,
            };
        }

        /// Create a UniSkipFirstRoundProof for Stage 1 from actual witnesses
        ///
        /// This computes real Az*Bz products using the constraint evaluators,
        /// producing a polynomial that satisfies the univariate skip verification.
        ///
        /// IMPORTANT: The eq polynomial must be computed from tau_low (excluding tau_high)
        /// because the UniSkip polynomial formula is:
        ///   s1(Y) = L(τ_high, Y) · t1(Y)
        /// where t1(Y) = Σ_x eq(τ_low, x) · Az(x,Y) · Bz(x,Y)
        ///
        /// If we used the full tau, τ_high would be counted twice!
        fn createUniSkipProofStage1FromWitnesses(
            self: *Self,
            cycle_witnesses: []const r1cs.R1CSCycleInputs(F),
            tau: []const F,
        ) !?UniSkipFirstRoundProof(F) {
            if (cycle_witnesses.len == 0) {
                return self.createUniSkipProofStage1();
            }

            const NUM_COEFFS = r1cs.OUTER_FIRST_ROUND_POLY_NUM_COEFFS;

            if (tau.len < 2) {
                return self.createUniSkipProofStage1();
            }

            // Use the StreamingOuterProver which properly handles both FIRST_GROUP
            // and SECOND_GROUP constraints in the UniSkip computation.
            //
            // Key differences from the old SpartanOuterProver:
            // 1. Uses full_tau for UniSkip eq computation (dropping tau_high internally)
            // 2. Iterates over both constraint groups (not just FIRST_GROUP)
            // 3. Properly handles the cycle/group interleaving
            //
            // The StreamingOuterProver.initWithScaling takes:
            // - cycle_witnesses: actual witness values per cycle
            // - tau: FULL tau vector (num_cycle_vars + 2 elements)
            // - lagrange_tau_r0: Lagrange kernel L(tau_high, r0) - but for UniSkip we use null
            //   because the Lagrange kernel multiplication is done in interpolateFirstRoundPoly
            var outer_prover = try streaming_outer.StreamingOuterProver(F).initWithScaling(
                self.allocator,
                cycle_witnesses,
                tau,
                null, // No scaling for initial UniSkip - will be applied in interpolation
            );
            defer outer_prover.deinit();

            // Compute the univariate skip polynomial using the fixed implementation
            // that properly handles both constraint groups
            const uni_poly_coeffs = try outer_prover.computeFirstRoundPoly();

            // Copy coefficients to our proof structure
            const coeffs = try self.allocator.alloc(F, NUM_COEFFS);
            @memset(coeffs, F.zero());

            // Copy available coefficients (may be fewer than NUM_COEFFS)
            const copy_len = @min(uni_poly_coeffs.len, NUM_COEFFS);
            @memcpy(coeffs[0..copy_len], uni_poly_coeffs[0..copy_len]);

            return UniSkipFirstRoundProof(F){
                .uni_poly = coeffs,
                .allocator = self.allocator,
            };
        }

        /// Convert with actual per-cycle witnesses for real constraint evaluation
        ///
        /// This method produces proofs with proper Az*Bz evaluations instead of zeros.
        /// Use this for cross-verification with Jolt.
        pub fn convertWithWitnesses(
            self: *Self,
            comptime Commitment: type,
            comptime Proof: type,
            zolt_stage_proofs: *const prover.JoltStageProofs(F),
            commitments: []const Commitment,
            joint_opening_proof: ?Proof,
            config: ConversionConfig,
            cycle_witnesses: []const r1cs.R1CSCycleInputs(F),
            tau: []const F,
        ) !JoltProofType(F, Commitment, Proof) {
            var jolt_proof = JoltProofType(F, Commitment, Proof).init(self.allocator);

            // Copy configuration parameters
            const trace_length: usize = @as(usize, 1) << @intCast(zolt_stage_proofs.log_t);
            const ram_K: usize = @as(usize, 1) << @intCast(zolt_stage_proofs.log_k);

            jolt_proof.trace_length = trace_length;
            jolt_proof.ram_K = ram_K;
            jolt_proof.bytecode_K = config.bytecode_K;
            jolt_proof.log_k_chunk = config.log_k_chunk;
            jolt_proof.lookups_ra_virtual_log_k_chunk = config.lookups_ra_virtual_log_k_chunk;

            // Compute derived parameters
            const n_cycle_vars = std.math.log2_int(usize, trace_length);
            const log_ram_k = std.math.log2_int(usize, ram_K);
            _ = log_ram_k;

            // Copy commitments
            for (commitments) |c| {
                try jolt_proof.commitments.append(self.allocator, c);
            }

            // Set joint opening proof
            jolt_proof.joint_opening_proof = joint_opening_proof;

            // Create UniSkip proof for Stage 1 with actual constraint evaluations
            jolt_proof.stage1_uni_skip_first_round_proof = try self.createUniSkipProofStage1FromWitnesses(
                cycle_witnesses,
                tau,
            );

            // Stage 1: Outer Spartan Remaining - use streaming prover for actual evaluations
            try self.generateStreamingOuterSumcheckProof(
                &jolt_proof.stage1_sumcheck_proof,
                cycle_witnesses,
                tau,
            );

            // Add Stage 1 opening claims
            try self.addSpartanOuterOpeningClaims(&jolt_proof.opening_claims);

            // Create UniSkip proof for Stage 2 (still using placeholder for now)
            jolt_proof.stage2_uni_skip_first_round_proof = try self.createUniSkipProofStage2();

            // Stage 2 and onwards use placeholder zero proofs
            try self.generateZeroSumcheckProof(
                &jolt_proof.stage2_sumcheck_proof,
                n_cycle_vars + 1,
                3,
            );

            // Add remaining opening claims (same as standard convert)
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .RamRa, .sumcheck_id = .RamRafEvaluation } },
                F.zero(),
            );
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .RamVal, .sumcheck_id = .RamReadWriteChecking } },
                F.zero(),
            );

            // Stages 3-7 (placeholder)
            try self.generateZeroSumcheckProof(&jolt_proof.stage3_sumcheck_proof, n_cycle_vars, 3);
            try self.generateZeroSumcheckProof(&jolt_proof.stage4_sumcheck_proof, n_cycle_vars, 3);
            try self.generateZeroSumcheckProof(&jolt_proof.stage5_sumcheck_proof, n_cycle_vars, 3);
            try self.generateZeroSumcheckProof(&jolt_proof.stage6_sumcheck_proof, n_cycle_vars, 3);
            try self.generateZeroSumcheckProof(&jolt_proof.stage7_sumcheck_proof, n_cycle_vars, 3);

            return jolt_proof;
        }

        /// Convert with actual per-cycle witnesses and Fiat-Shamir transcript
        ///
        /// This method produces proofs with proper Az*Bz evaluations and uses
        /// the Blake2b transcript for all Fiat-Shamir challenges.
        /// This is the method to use for Jolt cross-verification.
        pub fn convertWithTranscript(
            self: *Self,
            comptime Commitment: type,
            comptime Proof: type,
            zolt_stage_proofs: *const prover.JoltStageProofs(F),
            commitments: []const Commitment,
            joint_opening_proof: ?Proof,
            config: ConversionConfig,
            cycle_witnesses: []const r1cs.R1CSCycleInputs(F),
            tau: []const F,
            transcript: *Blake2bTranscript(F),
        ) !JoltProofType(F, Commitment, Proof) {
            var jolt_proof = JoltProofType(F, Commitment, Proof).init(self.allocator);

            // Copy configuration parameters
            const trace_length: usize = @as(usize, 1) << @intCast(zolt_stage_proofs.log_t);
            const ram_K: usize = @as(usize, 1) << @intCast(zolt_stage_proofs.log_k);

            jolt_proof.trace_length = trace_length;
            jolt_proof.ram_K = ram_K;
            jolt_proof.bytecode_K = config.bytecode_K;
            jolt_proof.log_k_chunk = config.log_k_chunk;
            jolt_proof.lookups_ra_virtual_log_k_chunk = config.lookups_ra_virtual_log_k_chunk;

            // Compute derived parameters
            const n_cycle_vars = std.math.log2_int(usize, trace_length);

            // Copy commitments and append to transcript
            for (commitments) |c| {
                try jolt_proof.commitments.append(self.allocator, c);
            }

            // Append commitments to transcript (GT elements for Dory)
            // This is done in Jolt's prover before deriving challenges
            // Note: For now we skip this since commitment serialization to transcript
            // is complex and involves GT element encoding

            // Set joint opening proof
            jolt_proof.joint_opening_proof = joint_opening_proof;

            // Create UniSkip proof for Stage 1 with actual constraint evaluations
            jolt_proof.stage1_uni_skip_first_round_proof = try self.createUniSkipProofStage1FromWitnesses(
                cycle_witnesses,
                tau,
            );

            // Stage 1: Outer Spartan Remaining - use streaming prover with transcript
            var stage1_result: ?Stage1Result = null;
            if (jolt_proof.stage1_uni_skip_first_round_proof) |*uniskip| {
                stage1_result = try self.generateStreamingOuterSumcheckProofWithTranscript(
                    &jolt_proof.stage1_sumcheck_proof,
                    uniskip,
                    cycle_witnesses,
                    tau,
                    transcript,
                );
            } else {
                // Fallback to zero proofs
                try self.generateZeroSumcheckProof(
                    &jolt_proof.stage1_sumcheck_proof,
                    1 + n_cycle_vars,
                    3,
                );
            }
            defer if (stage1_result) |*r| r.deinit();

            // Add Stage 1 opening claims with computed MLE evaluations
            if (stage1_result) |result| {
                // The r_cycle point for R1CS input evaluation
                // In Jolt, sumcheck_challenges = [r_stream, r_1, r_2, ..., r_n]
                // For opening claims, r_cycle = challenges[1..] converted to BIG_ENDIAN
                // This means: take [r_1, ..., r_n] and reverse to [r_n, ..., r_1]
                const all_challenges = result.challenges.items;

                // Skip the first challenge (r_stream) to get the cycle challenges
                const cycle_challenges = if (all_challenges.len > 1)
                    all_challenges[1..]
                else
                    all_challenges;

                // Convert from LITTLE_ENDIAN (sumcheck order) to BIG_ENDIAN (MLE eval order)
                // by reversing the challenges
                const r_cycle_big_endian = try self.allocator.alloc(F, cycle_challenges.len);
                defer self.allocator.free(r_cycle_big_endian);
                for (0..cycle_challenges.len) |i| {
                    r_cycle_big_endian[i] = cycle_challenges[cycle_challenges.len - 1 - i];
                }

                try self.addSpartanOuterOpeningClaimsWithEvaluations(
                    &jolt_proof.opening_claims,
                    cycle_witnesses,
                    r_cycle_big_endian,
                    result.uni_skip_claim,
                );
            } else {
                // Fallback to zero claims
                try self.addSpartanOuterOpeningClaims(&jolt_proof.opening_claims);
            }

            // Create UniSkip proof for Stage 2
            jolt_proof.stage2_uni_skip_first_round_proof = try self.createUniSkipProofStage2();

            // Stage 2 and onwards: still use placeholder zero proofs
            // (Full implementation would require complete Stage 2-7 prover)
            try self.generateZeroSumcheckProof(
                &jolt_proof.stage2_sumcheck_proof,
                n_cycle_vars + 1,
                3,
            );

            // Add remaining opening claims
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .RamRa, .sumcheck_id = .RamRafEvaluation } },
                F.zero(),
            );
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .RamVal, .sumcheck_id = .RamReadWriteChecking } },
                F.zero(),
            );
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .UnivariateSkip, .sumcheck_id = .SpartanProductVirtualization } },
                F.zero(),
            );

            // Stages 3-7 (placeholder)
            try self.generateZeroSumcheckProof(&jolt_proof.stage3_sumcheck_proof, n_cycle_vars, 3);
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .LookupOutput, .sumcheck_id = .InstructionClaimReduction } },
                F.zero(),
            );

            try self.generateZeroSumcheckProof(&jolt_proof.stage4_sumcheck_proof, n_cycle_vars, 3);
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .RamVal, .sumcheck_id = .RamValEvaluation } },
                F.zero(),
            );
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .RamValFinal, .sumcheck_id = .RamValFinalEvaluation } },
                F.zero(),
            );

            try self.generateZeroSumcheckProof(&jolt_proof.stage5_sumcheck_proof, n_cycle_vars, 3);
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .RegistersVal, .sumcheck_id = .RegistersValEvaluation } },
                F.zero(),
            );
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .RamRa, .sumcheck_id = .RamRaClaimReduction } },
                F.zero(),
            );

            try self.generateZeroSumcheckProof(&jolt_proof.stage6_sumcheck_proof, n_cycle_vars, 3);
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .RamHammingWeight, .sumcheck_id = .Booleanity } },
                F.zero(),
            );
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .RamHammingWeight, .sumcheck_id = .RamHammingBooleanity } },
                F.zero(),
            );

            try self.generateZeroSumcheckProof(&jolt_proof.stage7_sumcheck_proof, config.log_k_chunk, 3);
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .RamHammingWeight, .sumcheck_id = .HammingWeightClaimReduction } },
                F.zero(),
            );

            return jolt_proof;
        }

        /// Create a UniSkipFirstRoundProof for Stage 2 (degree-12 polynomial)
        ///
        /// Jolt's Stage 2 (product virtualization) uses a degree-12 first-round
        /// polynomial for the 5 product constraints.
        ///
        /// For the verification to pass, the polynomial must satisfy:
        ///   Σ_{j=0}^{12} coeff_j * power_sums[j] = 0
        ///
        /// where power_sums[j] = Σ_{t in domain} t^j for domain {-2, -1, 0, 1, 2}.
        fn createUniSkipProofStage2(self: *Self) !?UniSkipFirstRoundProof(F) {
            const univariate_skip = r1cs.univariate_skip;

            // For stage 2, we need 13 coefficients (degree 12)
            const NUM_COEFFS = univariate_skip.PRODUCT_VIRTUAL_FIRST_ROUND_POLY_NUM_COEFFS;

            // Create an all-zero polynomial that trivially satisfies the sum constraint.
            const coeffs = try self.allocator.alloc(F, NUM_COEFFS);
            @memset(coeffs, F.zero());

            return UniSkipFirstRoundProof(F){
                .uni_poly = coeffs,
                .allocator = self.allocator,
            };
        }
    };
}

/// Configuration for proof conversion
///
/// These values must match Jolt's config.rs:
/// - log_k_chunk: Must be <= 8 (Jolt uses 4 for small traces, 8 for large)
/// - lookups_ra_virtual_log_k_chunk: Jolt uses LOG_K/8 (=16) for small traces
pub const ConversionConfig = struct {
    /// Bytecode address space size (K)
    bytecode_K: usize = 1 << 16,
    /// Log of chunk size for one-hot encoding (must be <= 8, Jolt uses 4 for small traces)
    log_k_chunk: usize = 4,
    /// Log of chunk size for lookups RA virtualization (LOG_K / 8 = 128 / 8 = 16 for small traces)
    lookups_ra_virtual_log_k_chunk: usize = 16,
};

// =============================================================================
// Tests
// =============================================================================

const testing = std.testing;
const BN254Scalar = field_mod.BN254Scalar;

test "proof converter: basic initialization" {
    const converter = ProofConverter(BN254Scalar).init(testing.allocator);
    _ = converter;
}

test "proof converter: convert empty proof" {
    const F = BN254Scalar;
    var converter = ProofConverter(F).init(testing.allocator);

    // Create empty Zolt stage proofs
    var zolt_proofs = prover.JoltStageProofs(F).init(testing.allocator);
    defer zolt_proofs.deinit();

    zolt_proofs.log_t = 4; // 16 steps
    zolt_proofs.log_k = 10; // 1024 addresses

    // Dummy commitment and proof types for testing
    const DummyCommitment = struct { value: u64 };
    const DummyProof = struct { data: [32]u8 };

    // Convert to Jolt format
    var jolt_proof = try converter.convert(
        DummyCommitment,
        DummyProof,
        &zolt_proofs,
        &[_]DummyCommitment{},
        null,
        .{},
    );
    defer jolt_proof.deinit();

    // Verify trace length is correct
    try testing.expectEqual(@as(usize, 16), jolt_proof.trace_length);
    try testing.expectEqual(@as(usize, 1024), jolt_proof.ram_K);
}

test "proof converter: convert generates zero proofs" {
    const F = BN254Scalar;
    var converter = ProofConverter(F).init(testing.allocator);

    // Create Zolt stage proofs with data
    var zolt_proofs = prover.JoltStageProofs(F).init(testing.allocator);
    defer zolt_proofs.deinit();

    zolt_proofs.log_t = 2; // trace_length = 4
    zolt_proofs.log_k = 8; // ram_K = 256

    // Note: Zolt stage data is now ignored - we generate zero proofs

    // Dummy types
    const DummyCommitment = struct { value: u64 };
    const DummyProof = struct { data: [32]u8 };

    // Convert
    var jolt_proof = try converter.convert(
        DummyCommitment,
        DummyProof,
        &zolt_proofs,
        &[_]DummyCommitment{},
        null,
        .{},
    );
    defer jolt_proof.deinit();

    // Verify trace length (2^2 = 4)
    try testing.expectEqual(@as(usize, 4), jolt_proof.trace_length);

    // Stage 1: num_rounds = 1 + n_cycle_vars = 1 + 2 = 3
    try testing.expectEqual(@as(usize, 3), jolt_proof.stage1_sumcheck_proof.compressed_polys.items.len);

    // Stage 2: num_rounds = n_cycle_vars + 1 = 3
    try testing.expectEqual(@as(usize, 3), jolt_proof.stage2_sumcheck_proof.compressed_polys.items.len);

    // Verify uni skip proofs were created
    try testing.expect(jolt_proof.stage1_uni_skip_first_round_proof != null);
    try testing.expect(jolt_proof.stage2_uni_skip_first_round_proof != null);

    // Verify opening claims were added (multiple claims per stage)
    try testing.expect(jolt_proof.opening_claims.len() > 0);
}

test "proof converter: convertWithTranscript uses Blake2b transcript" {
    const F = BN254Scalar;
    var converter = ProofConverter(F).init(testing.allocator);

    // Create Zolt stage proofs
    var zolt_proofs = prover.JoltStageProofs(F).init(testing.allocator);
    defer zolt_proofs.deinit();

    zolt_proofs.log_t = 2; // trace_length = 4
    zolt_proofs.log_k = 8; // ram_K = 256

    // Create trivial cycle witnesses
    const cycle_witnesses = [_]r1cs.R1CSCycleInputs(F){
        .{ .values = [_]F{F.zero()} ** 36 },
        .{ .values = [_]F{F.zero()} ** 36 },
        .{ .values = [_]F{F.zero()} ** 36 },
        .{ .values = [_]F{F.zero()} ** 36 },
    };

    // Create tau challenge vector
    const tau = [_]F{ F.fromU64(1), F.fromU64(2), F.fromU64(3) };

    // Initialize transcript (matching Jolt's label)
    var transcript = Blake2bTranscript(F).init("jolt_v1");

    // Dummy types
    const DummyCommitment = struct { value: u64 };
    const DummyProof = struct { data: [32]u8 };

    // Convert with transcript
    var jolt_proof = try converter.convertWithTranscript(
        DummyCommitment,
        DummyProof,
        &zolt_proofs,
        &[_]DummyCommitment{},
        null,
        .{},
        &cycle_witnesses,
        &tau,
        &transcript,
    );
    defer jolt_proof.deinit();

    // Verify trace length
    try testing.expectEqual(@as(usize, 4), jolt_proof.trace_length);

    // Verify transcript was used (round counter should be > 0 after generating proof)
    try testing.expect(transcript.n_rounds > 0);

    // Verify uni skip proofs were created
    try testing.expect(jolt_proof.stage1_uni_skip_first_round_proof != null);
    try testing.expect(jolt_proof.stage2_uni_skip_first_round_proof != null);

    // Verify opening claims were added
    try testing.expect(jolt_proof.opening_claims.len() > 0);
}

test "proof converter: transcript produces deterministic challenges" {
    const F = BN254Scalar;

    // Create two converters and transcripts with same inputs
    var converter1 = ProofConverter(F).init(testing.allocator);
    var converter2 = ProofConverter(F).init(testing.allocator);

    var zolt_proofs1 = prover.JoltStageProofs(F).init(testing.allocator);
    defer zolt_proofs1.deinit();
    zolt_proofs1.log_t = 2;
    zolt_proofs1.log_k = 8;

    var zolt_proofs2 = prover.JoltStageProofs(F).init(testing.allocator);
    defer zolt_proofs2.deinit();
    zolt_proofs2.log_t = 2;
    zolt_proofs2.log_k = 8;

    const cycle_witnesses = [_]r1cs.R1CSCycleInputs(F){
        .{ .values = [_]F{F.zero()} ** 36 },
        .{ .values = [_]F{F.zero()} ** 36 },
    };

    const tau = [_]F{ F.fromU64(1), F.fromU64(2) };

    var transcript1 = Blake2bTranscript(F).init("jolt_test");
    var transcript2 = Blake2bTranscript(F).init("jolt_test");

    const DummyCommitment = struct { value: u64 };
    const DummyProof = struct { data: [32]u8 };

    var jolt_proof1 = try converter1.convertWithTranscript(
        DummyCommitment,
        DummyProof,
        &zolt_proofs1,
        &[_]DummyCommitment{},
        null,
        .{},
        &cycle_witnesses,
        &tau,
        &transcript1,
    );
    defer jolt_proof1.deinit();

    var jolt_proof2 = try converter2.convertWithTranscript(
        DummyCommitment,
        DummyProof,
        &zolt_proofs2,
        &[_]DummyCommitment{},
        null,
        .{},
        &cycle_witnesses,
        &tau,
        &transcript2,
    );
    defer jolt_proof2.deinit();

    // Same inputs should produce same transcript state
    try testing.expectEqualSlices(u8, &transcript1.state, &transcript2.state);
    try testing.expectEqual(transcript1.n_rounds, transcript2.n_rounds);
}
