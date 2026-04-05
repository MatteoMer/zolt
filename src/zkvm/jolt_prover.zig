//! Jolt Prover: Direct Jolt-Compatible 7-Stage Proof Generation
//!
//! This module generates proofs directly in Jolt's 7-stage format
//! for cross-verification compatibility with the upstream Jolt verifier.
//!
//! ## Stage Layout (Jolt 7 stages)
//!
//! 1. Outer Spartan (+ UniSkip)
//! 2. Product virtualization + RAM RAF + Read-Write (+ UniSkip)
//! 3. Spartan shift + Instruction input + Registers claim
//! 4. Registers RW + RAM val evaluation + RAM val final
//! 5. Registers val evaluation + RAM RA + Lookups RAF
//! 6. Bytecode RAF + Hamming + Booleanity + RA virtual
//! 7. Hamming weight claim reduction
//!
//! ## Constraint Evaluation

const std = @import("std");

const Allocator = std.mem.Allocator;
const ThreadPool = @import("zolt_pool").ThreadPool;

const jolt_types = @import("jolt_types.zig");
const field_mod = @import("zolt_arith").field;
const UnreducedProductAccum = field_mod.UnreducedProductAccum;
const r1cs = @import("r1cs/mod.zig");
const streaming_outer = @import("spartan/streaming_outer.zig");
const product_remainder = @import("spartan/product_remainder.zig");
const transcripts = @import("zolt_arith").transcripts;
const Blake2bTranscript = transcripts.Blake2bTranscript;
const poly_mod = @import("zolt_arith").poly;
const jolt_device = @import("jolt_device.zig");
const constants = @import("../common/constants.zig");
const ram = @import("ram/mod.zig");
const instruction = @import("instruction/mod.zig");
const spartan_mod = @import("spartan/mod.zig");
const Stage3Prover = spartan_mod.Stage3Prover;
const Stage5BatchedProver = spartan_mod.Stage5BatchedProver;
const Stage6BatchedProver = spartan_mod.Stage6BatchedProver;
const preprocessing = @import("preprocessing.zig");
const r1cs_evaluators = @import("r1cs/evaluators.zig");

const zkvm_debug = @import("debug.zig");
const debug_verbose = zkvm_debug.verbose;
const stage_timing_enabled = false;

/// Direct Jolt-compatible 7-stage prover
pub fn JoltProver(comptime F: type) type {
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

        const gpu_mod = @import("zolt_arith").gpu;

        allocator: Allocator,
        thread_pool: ?*ThreadPool = null,
        gpu_ops: ?*gpu_mod.GpuPolyOps = null,
        gpu_msm: ?*gpu_mod.GpuMsmOps = null,
        // Heap-allocated GPU resources (must not be inline — struct is returned by value)
        _gpu_accel: ?*gpu_mod.GpuAccelerator = null,
        _gpu_poly: ?*gpu_mod.GpuPolyOps = null,
        _gpu_msm: ?*gpu_mod.GpuMsmOps = null,

        pub fn init(allocator: Allocator) Self {
            return Self{ .allocator = allocator };
        }

        pub fn initWithThreadPool(allocator: Allocator, tp: *ThreadPool) Self {
            return Self{
                .allocator = allocator,
                .thread_pool = tp,
            };
        }

        pub fn deinit(self: *Self) void {
            if (self._gpu_msm) |m| {
                m.deinit();
                self.allocator.destroy(m);
            }
            if (self._gpu_poly) |p| {
                p.deinit();
                self.allocator.destroy(p);
            }
            if (self._gpu_accel) |a| {
                a.deinit();
                self.allocator.destroy(a);
            }
            self.gpu_ops = null;
            self.gpu_msm = null;
        }

        /// Lazy GPU init — only called when trace size justifies GPU overhead.
        /// Call from prove path after trace length is known.
        pub fn enableGpu(self: *Self) void {
            if (self.gpu_ops != null) return; // already initialized
            const accel = self.allocator.create(gpu_mod.GpuAccelerator) catch return;
            accel.* = gpu_mod.GpuAccelerator.init(self.allocator) catch {
                self.allocator.destroy(accel);
                return;
            };
            const poly = self.allocator.create(gpu_mod.GpuPolyOps) catch {
                accel.deinit();
                self.allocator.destroy(accel);
                return;
            };
            poly.* = gpu_mod.GpuPolyOps.init(accel) catch {
                accel.deinit();
                self.allocator.destroy(accel);
                self.allocator.destroy(poly);
                return;
            };
            self._gpu_accel = accel;
            self._gpu_poly = poly;
            self.gpu_ops = poly;

            // Also init GPU MSM ops (shares the same accelerator)
            const msm_ops = self.allocator.create(gpu_mod.GpuMsmOps) catch return;
            msm_ops.* = gpu_mod.GpuMsmOps.init(accel) catch {
                self.allocator.destroy(msm_ops);
                return;
            };
            self._gpu_msm = msm_ops;
            self.gpu_msm = msm_ops;
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
            tau: []const F,
            transcript: *Blake2bTranscript(F),
            compact_witnesses: []const r1cs_evaluators.CompactWitness,
        ) !Stage1Result {
            const StreamingOuterProver = streaming_outer.StreamingOuterProver(F);
            const LagrangePoly = r1cs.univariate_skip.LagrangePolynomial(F);
            var challenges: std.ArrayListUnmanaged(F) = .{};

            // Extract tau_high for the UniSkip Lagrange kernel
            // tau has length num_rows_bits = num_cycle_vars + 2
            // tau_high is the last element (used for Lagrange kernel)
            // Full tau is passed to split_eq (it handles the split internally)
            if (tau.len < 2) {
                return error.InvalidTauLength;
            }
            const tau_high = tau[tau.len - 1];

            // DEBUG: Print tau length (challenges from transcript)

            // The first round was already processed by UniSkip
            // Append the UniSkip polynomial to transcript
            transcript.appendScalars("uniskip_poly", uniskip_proof.uni_poly);

            // Get the challenge for the first round (r0)
            const r0 = transcript.challengeScalar();

            // Compute the Lagrange kernel L(r0, tau_high) to use as initial scaling
            const lagrange_tau_r0 = try LagrangePoly.lagrangeKernel(
                r1cs.univariate_skip.OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE,
                r0,
                tau_high,
                self.allocator,
            );

            // Initialize the streaming prover with full tau and Lagrange kernel scaling
            // The prover internally extracts:
            //   tau_high = tau[tau.len - 1] (stored separately for first-round polynomial)
            //   tau_low = tau[0..tau.len - 1] (passed to split_eq)
            // This matches Jolt's behavior in OuterSharedState::new().
            var outer_prover = try StreamingOuterProver.initWithScaling(
                self.allocator,
                compact_witnesses,
                tau, // Full tau - prover extracts tau_low and tau_high internally
                lagrange_tau_r0,
            );
            outer_prover.thread_pool = self.thread_pool;
            defer outer_prover.deinit();

            // Compute the UnivariateSkip claim: evaluation of UniSkip polynomial at r0
            const uni_skip_claim = evaluatePolyAtChallenge(uniskip_proof.uni_poly, r0);

            // (Debug decomposition block removed — was gated by `comptime false`)

            // Bind the first-round challenge from transcript with the uni_skip_claim
            try outer_prover.bindFirstRoundChallenge(r0, uni_skip_claim);

            // Match Jolt's cache_openings: after UniSkip verification, the verifier calls
            // accumulator.append_virtual() which appends the uni_skip_claim to transcript.
            // This happens BEFORE BatchedSumcheck::verify which also appends it.
            // flush_to_transcript: uni_skip opening claim
            transcript.appendScalar("opening_claim", uni_skip_claim);
            if (comptime debug_verbose) {
                std.debug.print("[ZOLT-PROVER] after_flush transcript_state = ", .{});
                for (transcript.state[0..8]) |b| std.debug.print("{x:0>2}", .{b});
                std.debug.print(" round={}\n", .{transcript.n_rounds});
            }

            // BatchedSumcheck::verify: append input_claim then get batching coefficients
            transcript.appendScalar("sumcheck_claim", uni_skip_claim);

            // Get batching coefficient - advances transcript state AND provides scaling factor
            const batching_coeff = transcript.challengeScalarFull();
            if (comptime debug_verbose) {
                std.debug.print("[ZOLT-PROVER] input_claim (uni_skip_claim) = {any}\n", .{uni_skip_claim.toBytesBE()});
                std.debug.print("[ZOLT-PROVER] batching_coeff = {any}\n", .{batching_coeff.toBytesBE()});
                std.debug.print("[ZOLT-PROVER] transcript state: ", .{});
                for (transcript.state[0..8]) |b| std.debug.print("{x:0>2} ", .{b});
                std.debug.print("round={}\n", .{transcript.n_rounds});
            }

            // Generate remaining rounds
            // In Jolt, stage1_sumcheck_proof contains num_rounds polynomials
            // where num_rounds = 1 + num_cycle_vars (1 streaming + cycle vars)
            // The UniSkip is separate and doesn't count here
            const num_remaining_rounds = outer_prover.numRounds(); // 1 + num_cycle_vars
            if (num_remaining_rounds == 0) {
                return Stage1Result{ .challenges = challenges, .r0 = r0, .uni_skip_claim = uni_skip_claim, .allocator = self.allocator };
            }

            // Compute initial claim = uni_skip_claim * batching_coeff (for Jolt compatibility)
            const initial_claim = uni_skip_claim.mul(batching_coeff);
            if (comptime debug_verbose) {
                std.debug.print("[ZOLT-PROVER] batched_claim = {any}\n", .{initial_claim.toBytesBE()});
            }

            // Generate all remaining round polynomials with transcript integration
            for (0..num_remaining_rounds) |_| {
                const raw_evals: [4]F = try outer_prover.computeRemainingRoundPoly();

                // Scale evaluations by batching coefficient for output
                const scaled_evals = [4]F{
                    raw_evals[0].mul(batching_coeff),
                    raw_evals[1].mul(batching_coeff),
                    raw_evals[2].mul(batching_coeff),
                    raw_evals[3].mul(batching_coeff),
                };

                // Convert to compressed coefficients for proof
                const compressed = poly_mod.UniPoly(F).evalsToCompressed(scaled_evals);
                const coeffs = try self.allocator.alloc(F, 3);
                coeffs[0] = compressed[0]; // c0
                coeffs[1] = compressed[1]; // c2
                coeffs[2] = compressed[2]; // c3

                // DEBUG: Print round polynomial coefficients (LE bytes for Jolt comparison)

                try proof.compressed_polys.append(self.allocator, .{
                    .coeffs_except_linear_term = coeffs,
                    .allocator = self.allocator,
                });

                // Append round polynomial to transcript
                transcript.appendScalars("sumcheck_poly", coeffs);

                // Get challenge from transcript
                const challenge = transcript.challengeScalar();
                try challenges.append(self.allocator, challenge);

                // DEBUG: Print challenge (LE bytes for Jolt comparison)

                // Bind challenge and update claim
                try outer_prover.bindRemainingRoundChallenge(challenge);
                outer_prover.updateClaim(raw_evals, challenge);
            }

            return Stage1Result{ .challenges = challenges, .r0 = r0, .uni_skip_claim = uni_skip_claim, .allocator = self.allocator };
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

        /// R1CS input indices in upstream Jolt's ALL_R1CS_INPUTS order (35 inputs)
        /// Maps from Jolt's ordering (index in this array) to Zolt's R1CSInputIndex
        const JOLT_TO_ZOLT_R1CS_INDICES = [35]r1cs.R1CSInputIndex{
            .LeftInstructionInput, // 0
            .RightInstructionInput, // 1
            .Product, // 2
            .ShouldBranch, // 3
            .PC, // 4
            .UnexpandedPC, // 5
            .Imm, // 6
            .RamAddress, // 7
            .Rs1Value, // 8
            .Rs2Value, // 9
            .RdWriteValue, // 10
            .RamReadValue, // 11
            .RamWriteValue, // 12
            .LeftLookupOperand, // 13
            .RightLookupOperand, // 14
            .NextUnexpandedPC, // 15
            .NextPC, // 16
            .NextIsVirtual, // 17
            .NextIsFirstInSequence, // 18
            .LookupOutput, // 19
            .ShouldJump, // 20
            .FlagAddOperands, // 21
            .FlagSubtractOperands, // 22
            .FlagMultiplyOperands, // 23
            .FlagLoad, // 24
            .FlagStore, // 25
            .FlagJump, // 26
            .FlagWriteLookupOutputToRD, // 27
            .FlagVirtualInstruction, // 28
            .FlagAssert, // 29
            .FlagDoNotUpdateUnexpandedPC, // 30
            .FlagAdvice, // 31
            .FlagIsCompressed, // 32
            .FlagIsFirstInSequence, // 33
            .FlagIsLastInSequence, // 34
        };

        /// VirtualPolynomial identifiers in upstream Jolt's order (35 inputs)
        const R1CS_VIRTUAL_POLYS = [35]VirtualPolynomial{
            .LeftInstructionInput, // 0
            .RightInstructionInput, // 1
            .Product, // 2
            .ShouldBranch, // 3
            .PC, // 4
            .UnexpandedPC, // 5
            .Imm, // 6
            .RamAddress, // 7
            .Rs1Value, // 8
            .Rs2Value, // 9
            .RdWriteValue, // 10
            .RamReadValue, // 11
            .RamWriteValue, // 12
            .LeftLookupOperand, // 13
            .RightLookupOperand, // 14
            .NextUnexpandedPC, // 15
            .NextPC, // 16
            .NextIsVirtual, // 17
            .NextIsFirstInSequence, // 18
            .LookupOutput, // 19
            .ShouldJump, // 20
            // OpFlags variants (14 of them) - CircuitFlags indices
            .{ .OpFlags = 0 }, // 21: AddOperands
            .{ .OpFlags = 1 }, // 22: SubtractOperands
            .{ .OpFlags = 2 }, // 23: MultiplyOperands
            .{ .OpFlags = 3 }, // 24: Load
            .{ .OpFlags = 4 }, // 25: Store
            .{ .OpFlags = 5 }, // 26: Jump
            .{ .OpFlags = 6 }, // 27: WriteLookupOutputToRD
            .{ .OpFlags = 7 }, // 28: VirtualInstruction
            .{ .OpFlags = 8 }, // 29: Assert
            .{ .OpFlags = 9 }, // 30: DoNotUpdateUnexpandedPC
            .{ .OpFlags = 10 }, // 31: Advice
            .{ .OpFlags = 11 }, // 32: IsCompressed
            .{ .OpFlags = 12 }, // 33: IsFirstInSequence
            .{ .OpFlags = 13 }, // 34: IsLastInSequence
        };

        /// Add all 35 R1CS input opening claims for SpartanOuter with zero claims
        ///
        /// This exactly matches the ALL_R1CS_INPUTS array in upstream Jolt's r1cs/inputs.rs:
        /// - 21 simple virtual polynomials
        /// - 14 OpFlags variants
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

        /// Add all 35 R1CS input opening claims for SpartanOuter with actual evaluations
        ///
        /// This computes the MLE evaluations at r_cycle and uses those as the claims.
        ///
        /// IMPORTANT: This also appends all 35 R1CS input claims to the transcript
        /// in Jolt's order (ALL_R1CS_INPUTS). This is required for Fiat-Shamir
        /// consistency before deriving Stage 2's tau_high challenge.
        fn addSpartanOuterOpeningClaimsWithEvaluations(
            self: *Self,
            claims: *OpeningClaims(F),
            raw_r1cs_inputs: []const r1cs_evaluators.RawR1CSInputs,
            padded_trace_len: usize,
            r_cycle: []const F,
            uni_skip_claim: F,
            transcript: *Blake2bTranscript(F),
            _: F, // r_stream (unused after debug removal)
            r0: F,
        ) !void {
            // Compute MLE evaluations at r_cycle using typed accumulators
            const R1CSInputEvaluator = r1cs.R1CSInputEvaluator(F);
            const input_evals = try R1CSInputEvaluator.computeClaimedInputsTyped(
                self.allocator,
                raw_r1cs_inputs,
                padded_trace_len,
                r_cycle,
                self.thread_pool,
            );

            // Compute Lagrange weights at r0
            const FIRST_GROUP_SIZE = 10;
            const SECOND_GROUP_SIZE = 9;
            var lagrange_weights: [FIRST_GROUP_SIZE]F = undefined;
            const base_left: i64 = -@as(i64, (FIRST_GROUP_SIZE - 1) / 2); // = -4

            for (0..FIRST_GROUP_SIZE) |i| {
                var numer = F.one();
                var denom = F.one();

                for (0..FIRST_GROUP_SIZE) |j| {
                    if (i != j) {
                        const x_j: i64 = base_left + @as(i64, @intCast(j));
                        const x_j_field = if (x_j >= 0)
                            F.fromU64(@intCast(x_j))
                        else
                            F.zero().sub(F.fromU64(@intCast(-x_j)));
                        numer = numer.mul(r0.sub(x_j_field));

                        const diff: i64 = @as(i64, @intCast(i)) - @as(i64, @intCast(j));
                        if (diff > 0) {
                            denom = denom.mul(F.fromU64(@intCast(diff)));
                        } else {
                            denom = denom.mul(F.zero().sub(F.fromU64(@intCast(-diff))));
                        }
                    }
                }

                lagrange_weights[i] = if (!denom.eql(F.zero()))
                    numer.mul(denom.inverse().?)
                else
                    F.zero();
            }

            // Build z vector with trailing 1 (like Jolt)
            var z: [r1cs.R1CSInputIndex.NUM_INPUTS + 1]F = undefined;
            @memcpy(z[0..r1cs.R1CSInputIndex.NUM_INPUTS], &input_evals);
            z[r1cs.R1CSInputIndex.NUM_INPUTS] = F.one(); // constant column

            // Compute az_g0, bz_g0 from first group
            var az_g0 = F.zero();
            var bz_g0 = F.zero();
            for (0..FIRST_GROUP_SIZE) |i| {
                const constraint_idx = r1cs.FIRST_GROUP_INDICES[i];
                const constraint = r1cs.UNIFORM_CONSTRAINTS[constraint_idx];

                // az_contrib = condition.dot_product(z)
                const az_contrib = constraint.condition.evaluateWithConstant(F, &z);
                // bz_contrib = (left - right).dot_product(z)
                const bz_contrib = constraint.left.evaluateWithConstant(F, &z)
                    .sub(constraint.right.evaluateWithConstant(F, &z));

                az_g0 = az_g0.add(lagrange_weights[i].mul(az_contrib));
                bz_g0 = bz_g0.add(lagrange_weights[i].mul(bz_contrib));
            }

            // Compute az_g1, bz_g1 from second group
            var az_g1 = F.zero();
            var bz_g1 = F.zero();
            const g1_len = @min(SECOND_GROUP_SIZE, FIRST_GROUP_SIZE);
            for (0..g1_len) |i| {
                const constraint_idx = r1cs.SECOND_GROUP_INDICES[i];
                const constraint = r1cs.UNIFORM_CONSTRAINTS[constraint_idx];

                const az_contrib = constraint.condition.evaluateWithConstant(F, &z);
                const bz_contrib = constraint.left.evaluateWithConstant(F, &z)
                    .sub(constraint.right.evaluateWithConstant(F, &z));

                az_g1 = az_g1.add(lagrange_weights[i].mul(az_contrib));
                bz_g1 = bz_g1.add(lagrange_weights[i].mul(bz_contrib));
            }

            // Blend with r_stream

            // Add R1CS inputs for SpartanOuter with computed evaluations
            // AND append each claim to transcript in Jolt's order (for Fiat-Shamir)

            for (R1CS_VIRTUAL_POLYS, 0..) |poly, jolt_idx| {
                // Map Jolt's index to Zolt's R1CSInputIndex
                const zolt_idx = JOLT_TO_ZOLT_R1CS_INDICES[jolt_idx].toIndex();
                const claim = input_evals[zolt_idx];

                try claims.insert(
                    .{ .Virtual = .{ .poly = poly, .sumcheck_id = .SpartanOuter } },
                    claim,
                );

                // flush_to_transcript: opening claim
                transcript.appendScalar("opening_claim", claim);
            }

            // Add the UnivariateSkip claim for SpartanOuter
            // This is uni_poly.evaluate(r0), the input_claim for the remaining sumcheck
            // NOTE: Do NOT append to transcript here - the UniSkip claim was already appended
            // twice earlier (once in cache_openings after r0 sampling, once in BatchedSumcheck::prove)
            try claims.insert(
                .{ .Virtual = .{ .poly = .UnivariateSkip, .sumcheck_id = .SpartanOuter } },
                uni_skip_claim,
            );
        }

        /// Create a UniSkipFirstRoundProof for Stage 1 (degree-27 polynomial)
        ///
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
            tau: []const F,
            compact_witnesses: []const r1cs_evaluators.CompactWitness,
        ) !?UniSkipFirstRoundProof(F) {
            if (compact_witnesses.len == 0) {
                return error.EmptyWitnesses;
            }

            const NUM_COEFFS = r1cs.OUTER_FIRST_ROUND_POLY_NUM_COEFFS;

            if (tau.len < 2) {
                return error.InvalidTauLength;
            }

            var outer_prover = try streaming_outer.StreamingOuterProver(F).initWithScaling(
                self.allocator,
                compact_witnesses,
                tau,
                null, // No scaling for initial UniSkip - will be applied in interpolation
            );
            outer_prover.thread_pool = self.thread_pool;
            outer_prover.gpu_ops = self.gpu_ops;
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

        // =================================================================
        // Stage output structs — carry data between proveWithTranscript stages
        // =================================================================

        /// Data produced by Stage 1 and consumed by later stages.
        const ProveStage1Output = struct {
            stage1_result: ?Stage1Result,
            /// r_spartan_original (BIG_ENDIAN) — used by Stages 3, 5, 6.
            r_spartan_original: []F,
        };

        /// Data produced by Stage 2 and consumed by later stages.
        const ProveStage2Output = struct {
            stage2_result: Stage2Result,
        };

        /// Data produced by Stage 3 and consumed by later stages.
        const ProveStage3Output = struct {
            stage3_result: spartan_mod.Stage3Result(F),
        };

        /// Data produced by Stage 4 and consumed by later stages.
        const ProveStage4Output = struct {
            stage4_regs_r_address: ?[]F,
            stage4_regs_r_cycle: ?[]F,
            stage4_r_cycle_val: ?[]F,
            r_reduction_be: ?[]F,
            stage4_inc_poly_copy: ?[]F,
            allocator: Allocator,

            pub fn deinit(self: *ProveStage4Output) void {
                if (self.stage4_regs_r_address) |arr| self.allocator.free(arr);
                if (self.stage4_regs_r_cycle) |arr| self.allocator.free(arr);
                if (self.stage4_r_cycle_val) |arr| self.allocator.free(arr);
                if (self.r_reduction_be) |arr| self.allocator.free(arr);
                if (self.stage4_inc_poly_copy) |arr| self.allocator.free(arr);
            }
        };

        /// Data produced by Stage 5 and consumed by later stages.
        const ProveStage5Output = struct {
            stage5_result: spartan_mod.Stage5Result(F),
            /// s_cycle_stage5 (BIG_ENDIAN) from RegistersValEvaluation opening point.
            s_cycle_stage5: []F,
            allocator: Allocator,

            pub fn deinit(self: *ProveStage5Output) void {
                self.stage5_result.deinit();
                self.allocator.free(self.s_cycle_stage5);
            }
        };

        /// Data produced by Stage 6 and consumed by Stage 7.
        const ProveStage6Output = struct {
            stage6_result: spartan_mod.Stage6Result(F),
        };

        /// Convert with actual per-cycle witnesses and Fiat-Shamir transcript
        ///
        /// This method produces proofs with proper Az*Bz evaluations and uses
        /// the Blake2b transcript for all Fiat-Shamir challenges.
        /// This is the method to use for Jolt cross-verification.
        pub fn proveWithTranscript(
            self: *Self,
            comptime Commitment: type,
            comptime Proof: type,
            log_t: u8,
            log_k: u8,
            commitments: []const Commitment,
            joint_opening_proof: ?Proof,
            config: JoltProverConfig,
            tau: []const F,
            transcript: *Blake2bTranscript(F),
        ) !JoltProofType(F, Commitment, Proof) {
            var jolt_proof = JoltProofType(F, Commitment, Proof).init(self.allocator);

            // Copy configuration parameters
            const trace_length: usize = @as(usize, 1) << @intCast(log_t);
            const ram_K: usize = @as(usize, 1) << @intCast(log_k);

            jolt_proof.trace_length = trace_length;
            jolt_proof.ram_K = ram_K;

            // Enable GPU for large traces where persistent-buffer operations win.
            if (trace_length >= 16384) self.enableGpu();

            jolt_proof.log_k_chunk = config.log_k_chunk;
            jolt_proof.lookups_ra_virtual_log_k_chunk = config.lookups_ra_virtual_log_k_chunk;

            // Set config structs (matching Jolt's serialization format)
            jolt_proof.rw_config = jolt_types.ReadWriteConfig.default(log_t, log_k);
            jolt_proof.one_hot_config = .{
                .log_k_chunk = @intCast(config.log_k_chunk),
                .lookups_ra_virtual_log_k_chunk = @intCast(config.lookups_ra_virtual_log_k_chunk),
            };
            jolt_proof.dory_layout = 0; // Wide layout

            // Compute derived parameters
            const n_cycle_vars = std.math.log2_int(usize, trace_length);
            const log_ram_k = std.math.log2_int(usize, ram_K);

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

            // Per-stage timing
            var stage_timer = std.time.Timer.start() catch unreachable;
            var bench_timer = std.time.Timer.start() catch unreachable;

            // Use pre-built compact/raw witnesses from config (built during witness gen,
            // outside Stage 1 timing).
            const compact_witnesses = config.prebuilt_compact;
            const raw_r1cs_inputs = config.prebuilt_raw;

            // ==================================================================
            // Execute 7 proving stages, threading data through output structs
            // ==================================================================

            var s1_out = try self.executeStage1(&jolt_proof, transcript, &config, tau, compact_witnesses, raw_r1cs_inputs, n_cycle_vars, trace_length, &stage_timer, &bench_timer);
            defer if (s1_out.stage1_result) |*r| r.deinit();
            defer self.allocator.free(s1_out.r_spartan_original);

            var s2_out = try self.executeStage2(&jolt_proof, transcript, &config, &s1_out, raw_r1cs_inputs, n_cycle_vars, log_ram_k, &stage_timer, &bench_timer);
            defer s2_out.stage2_result.deinit();

            var s3_out = try self.executeStage3(&jolt_proof, transcript, &config, &s2_out, &s1_out, raw_r1cs_inputs, n_cycle_vars, log_ram_k, &stage_timer, &bench_timer);
            defer s3_out.stage3_result.deinit();

            var s4_out = try self.executeStage4(&jolt_proof, transcript, &config, &s2_out, &s3_out, n_cycle_vars, log_ram_k, ram_K, trace_length, &stage_timer, &bench_timer);
            defer s4_out.deinit();

            var s5_out = try self.executeStage5(&jolt_proof, transcript, &config, &s1_out, &s2_out, &s4_out, n_cycle_vars, log_ram_k, ram_K, &stage_timer, &bench_timer);
            defer s5_out.deinit();

            var s6_out = try self.executeStage6(&jolt_proof, transcript, &config, &s1_out, &s2_out, &s3_out, &s4_out, &s5_out, n_cycle_vars, log_ram_k, ram_K, &stage_timer, &bench_timer);
            defer s6_out.stage6_result.deinit();

            try self.executeStage7(&jolt_proof, transcript, &config, &s2_out, &s5_out, &s6_out, n_cycle_vars, &stage_timer, &bench_timer);

            return jolt_proof;
        }

        /// Stage 1: UniSkip + Outer Spartan sumcheck + opening claims.
        fn executeStage1(
            self: *Self,
            jolt_proof: anytype,
            transcript: *Blake2bTranscript(F),
            config: *const JoltProverConfig,
            tau: []const F,
            compact_witnesses: []const r1cs_evaluators.CompactWitness,
            raw_r1cs_inputs: []const r1cs_evaluators.RawR1CSInputs,
            n_cycle_vars: usize,
            trace_length: usize,
            stage_timer: *std.time.Timer,
            bench_timer: *std.time.Timer,
        ) !ProveStage1Output {
            const bench = config.bench_output;

            jolt_proof.stage1_uni_skip_first_round_proof = try self.createUniSkipProofStage1FromWitnesses(
                tau,
                compact_witnesses,
            );
            const s1_init_ns = bench_timer.read();
            bench_timer.reset();

            // Stage 1: Outer Spartan Remaining - use streaming prover with transcript
            var stage1_result: ?Stage1Result = null;
            if (jolt_proof.stage1_uni_skip_first_round_proof) |*uniskip| {
                stage1_result = try self.generateStreamingOuterSumcheckProofWithTranscript(
                    &jolt_proof.stage1_sumcheck_proof,
                    uniskip,
                    tau,
                    transcript,
                    compact_witnesses,
                );
            } else {
                return error.MissingUniSkipProof;
            }
            errdefer if (stage1_result) |*r| r.deinit();
            const s1_sumcheck_ns = bench_timer.read();
            bench_timer.reset();

            // Add Stage 1 opening claims with computed MLE evaluations
            if (stage1_result) |result| {
                // The r_cycle point for R1CS input MLE evaluation
                // Sumcheck challenges = [r_stream, r_1, r_2, ..., r_n]
                // r_cycle = challenges[1..] converted to BIG_ENDIAN (reversed)
                //
                // Both Zolt's EqPolynomial.evals and Jolt's EqPolynomial::evals use
                // BIG_ENDIAN convention (r[0]→MSB). The sumcheck challenges are in
                // binding order (LowToHigh), so we reverse them to BIG_ENDIAN.
                const all_challenges = result.challenges.items;

                // Skip the first challenge (r_stream) to get the cycle challenges
                const cycle_challenges = if (all_challenges.len > 1)
                    all_challenges[1..]
                else
                    all_challenges;

                // Convert from sumcheck order to BIG_ENDIAN (MLE eval order)
                const r_cycle_big_endian = try self.allocator.alloc(F, cycle_challenges.len);
                defer self.allocator.free(r_cycle_big_endian);
                for (0..cycle_challenges.len) |i| {
                    r_cycle_big_endian[i] = cycle_challenges[cycle_challenges.len - 1 - i];
                }

                // Get r_stream (first challenge) and r0 from Stage 1 result
                const r_stream = if (all_challenges.len > 0) all_challenges[0] else F.zero();

                try self.addSpartanOuterOpeningClaimsWithEvaluations(
                    &jolt_proof.opening_claims,
                    raw_r1cs_inputs,
                    trace_length,
                    r_cycle_big_endian,
                    result.uni_skip_claim,
                    transcript,
                    r_stream,
                    result.r0,
                );
            } else {
                // Fallback to zero claims
                try self.addSpartanOuterOpeningClaims(&jolt_proof.opening_claims);
            }

            {
                const s1_claims_ns = bench_timer.read();
                const s1_total_ns = stage_timer.read();
                if (comptime stage_timing_enabled) {
                    std.debug.print("    [STAGE-TIMING] Stage 1: {d:.1} ms\n", .{@as(f64, @floatFromInt(s1_total_ns)) / 1_000_000.0});
                }
                if (bench) {
                    const ms = 1_000_000.0;
                    std.debug.print("[BENCH] stage=1 total={d:.1} init={d:.1} sumcheck={d:.1} claims={d:.1}\n", .{
                        @as(f64, @floatFromInt(s1_total_ns)) / ms,
                        @as(f64, @floatFromInt(s1_init_ns)) / ms,
                        @as(f64, @floatFromInt(s1_sumcheck_ns)) / ms,
                        @as(f64, @floatFromInt(s1_claims_ns)) / ms,
                    });
                }
            }
            stage_timer.reset();

            // Compute r_spartan_original (BIG_ENDIAN) from Stage 1 challenges.
            // Used by InstructionLookupsClaimReduction and later stages.
            var r_spartan_original = try self.allocator.alloc(F, n_cycle_vars);

            if (stage1_result) |result| {
                const all_chals = result.challenges.items;
                const cycle_chals = if (all_chals.len > 1)
                    all_chals[1..]
                else
                    all_chals;

                // Store r_spartan_original in BIG_ENDIAN order (like Jolt's opening point)
                for (0..n_cycle_vars) |i| {
                    const src_idx = n_cycle_vars - 1 - i;
                    if (src_idx < cycle_chals.len) {
                        r_spartan_original[i] = cycle_chals[src_idx];
                    } else {
                        r_spartan_original[i] = F.zero();
                    }
                }
            } else {
                for (0..n_cycle_vars) |i| {
                    r_spartan_original[i] = F.zero();
                }
            }

            return ProveStage1Output{
                .stage1_result = stage1_result,
                .r_spartan_original = r_spartan_original,
            };
        }

        /// Stage 2: Product virtualization + RAM RAF + Read-Write + output + instruction claim reduction.
        fn executeStage2(
            self: *Self,
            jolt_proof: anytype,
            transcript: *Blake2bTranscript(F),
            config: *const JoltProverConfig,
            s1_out: *const ProveStage1Output,
            raw_r1cs_inputs: []const r1cs_evaluators.RawR1CSInputs,
            n_cycle_vars: usize,
            log_ram_k: usize,
            stage_timer: *std.time.Timer,
            bench_timer: *std.time.Timer,
        ) !ProveStage2Output {
            const bench = config.bench_output;
            const stage1_result = s1_out.stage1_result;
            const r_spartan_original = s1_out.r_spartan_original;

            // Create UniSkip proof for Stage 2
            // Jolt samples a NEW tau_high for Stage 2 from the transcript (see ProductVirtualUniSkipParams::new)
            // tau = [r_cycle_outer, tau_high] where tau_high is freshly sampled
            const tau_high_stage2 = transcript.challengeScalar();

            // Get the 3 product claims from Stage 1's opening claims
            // Order: Product, ShouldBranch, ShouldJump
            const PRODUCT_VIRTUALS = [3]VirtualPolynomial{
                .Product,
                .ShouldBranch,
                .ShouldJump,
            };

            var base_evals_stage2: [3]F = [_]F{F.zero()} ** 3;
            for (PRODUCT_VIRTUALS, 0..) |poly, i| {
                const claim_key = OpeningId{ .Virtual = .{ .poly = poly, .sumcheck_id = .SpartanOuter } };
                if (jolt_proof.opening_claims.get(claim_key)) |claim| {
                    base_evals_stage2[i] = claim;
                }
            }

            // Debug: Print Stage 2 setup

            // Build tau_stage2 BEFORE calling createUniSkipProofStage2WithClaims
            // tau_stage2 = [r_cycle_reversed, tau_high_stage2]
            const tau_stage2_early = try self.allocator.alloc(F, n_cycle_vars + 1);
            defer self.allocator.free(tau_stage2_early);

            if (stage1_result) |result| {
                const all_challenges = result.challenges.items;
                // Skip the first challenge (r_stream) to get r_cycle
                const cycle_challenges = if (all_challenges.len > 1)
                    all_challenges[1..]
                else
                    all_challenges;

                // r_cycle reversed (BIG_ENDIAN)
                for (0..n_cycle_vars) |i| {
                    const src_idx = n_cycle_vars - 1 - i;
                    if (src_idx < cycle_challenges.len) {
                        tau_stage2_early[i] = cycle_challenges[src_idx];
                    } else {
                        tau_stage2_early[i] = F.zero();
                    }
                }
            } else {
                for (0..n_cycle_vars) |i| {
                    tau_stage2_early[i] = F.zero();
                }
            }
            tau_stage2_early[n_cycle_vars] = tau_high_stage2;

            jolt_proof.stage2_uni_skip_first_round_proof = try self.createUniSkipProofStage2WithClaims(
                &base_evals_stage2,
                tau_high_stage2,
                raw_r1cs_inputs,
                tau_stage2_early,
            );

            // CRITICAL: Append Stage 2 UniSkip polynomial to transcript (matching Jolt verifier flow)
            // The verifier calls UniSkipFirstRoundProof::verify which:
            // 1. Appends the polynomial coefficients to transcript
            // 2. Derives r0 challenge
            // 3. Calls cache_openings which appends UnivariateSkip claim
            var r0_stage2: F = F.zero();
            var uni_skip_claim_stage2: F = F.zero();

            if (jolt_proof.stage2_uni_skip_first_round_proof) |proof| {
                // Append polynomial
                transcript.appendScalars("uniskip_poly", proof.uni_poly);

                // Derive r0 challenge
                r0_stage2 = transcript.challengeScalar();

                // Compute UnivariateSkip claim = poly(r0)
                // uni_poly = [c0, c1, c2, ..., c12] -> poly(x) = c0 + c1*x + c2*x^2 + ...
                var r_power = F.one();
                for (proof.uni_poly) |coeff| {
                    uni_skip_claim_stage2 = uni_skip_claim_stage2.add(coeff.mul(r_power));
                    r_power = r_power.mul(r0_stage2);
                }

                // flush_to_transcript: uni_skip opening claim
                transcript.appendScalar("opening_claim", uni_skip_claim_stage2);

                // Update the opening claim for UnivariateSkip at SpartanProductVirtualization
                try jolt_proof.opening_claims.insert(
                    .{ .Virtual = .{ .poly = .UnivariateSkip, .sumcheck_id = .SpartanProductVirtualization } },
                    uni_skip_claim_stage2,
                );

                // Debug: verify the claim was inserted correctly
                const inserted_claim = jolt_proof.opening_claims.get(.{ .Virtual = .{ .poly = .UnivariateSkip, .sumcheck_id = .SpartanProductVirtualization } });
                if (inserted_claim) |_| {} else {}
            }

            // Stage 2 batches 5 sumcheck instances:
            // 1. ProductVirtualRemainder: n_cycle_vars rounds
            // 2. RamRafEvaluation: log_ram_k rounds
            // 3. RamReadWriteChecking: log_ram_k + n_cycle_vars rounds (max!)
            // 4. RamOutputCheck: log_ram_k rounds
            // 5. InstructionLookupsClaimReduction: n_cycle_vars rounds
            // max_num_rounds = log_ram_k + n_cycle_vars
            //
            // CRITICAL: Stage 2's tau is NOT the original tau from Stage 1!
            // It's built from [r_cycle_stage1, tau_high_stage2] where:
            // - r_cycle_stage1 = the sumcheck challenges from Stage 1 (opening point)
            // - tau_high_stage2 = freshly sampled challenge
            // See Jolt's ProductVirtualUniSkipParams::new

            // Build tau_stage2 from Stage 1 challenges (r_spartan_original is already BIG_ENDIAN)
            var tau_stage2 = try self.allocator.alloc(F, n_cycle_vars + 1);
            defer self.allocator.free(tau_stage2);
            for (0..n_cycle_vars) |i| {
                tau_stage2[i] = r_spartan_original[i];
            }
            tau_stage2[n_cycle_vars] = tau_high_stage2;

            const s2_init_ns = bench_timer.read();
            bench_timer.reset();

            var stage2_result = try self.generateStage2BatchedSumcheckProof(
                &jolt_proof.stage2_sumcheck_proof,
                transcript,
                r0_stage2,
                uni_skip_claim_stage2,
                tau_stage2,
                r_spartan_original,
                raw_r1cs_inputs,
                n_cycle_vars,
                log_ram_k,
                &jolt_proof.opening_claims,
                config.*,
            );
            const s2_sumcheck_ns = bench_timer.read();
            bench_timer.reset();

            // Add remaining opening claims - use actual final claims from provers
            // Instance 1 (RAF): RamRa opening is the final claim
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .RamRa, .sumcheck_id = .RamRafEvaluation } },
                stage2_result.raf_final_claim,
            );
            // Instance 2 (RWC): Individual opening claims for ra, val, inc
            // NOTE: The rwc_val_claim is the evaluation of RamVal at the Stage 2 opening point.
            // For Stage 4's RamValEvaluation, Jolt computes input_claim = rwc_val_claim - init_eval.
            // For programs without RAM ops, this should be 0 only if rwc_val_claim equals init_eval.
            // However, changing this claim would break Stage 2's expected_output_claim check.
            // The root cause needs to be fixed in the RWC prover's polynomial computation.
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .RamVal, .sumcheck_id = .RamReadWriteChecking } },
                stage2_result.rwc_val_claim, // RamVal evaluation at opening point
            );
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .RamRa, .sumcheck_id = .RamReadWriteChecking } },
                stage2_result.rwc_ra_claim, // RamRa evaluation at opening point
            );
            // RamInc is a committed polynomial needed by RamReadWriteChecking
            try jolt_proof.opening_claims.insert(
                .{ .Committed = .{ .poly = .RamInc, .sumcheck_id = .RamReadWriteChecking } },
                stage2_result.rwc_inc_claim, // RamInc evaluation at r_cycle
            );
            // Note: UnivariateSkip for SpartanProductVirtualization was already set above with the actual claim value

            // Add PRODUCT_UNIQUE_FACTOR_VIRTUALS claims for SpartanProductVirtualization
            // These 8 virtual polynomials match upstream's PRODUCT_UNIQUE_FACTOR_VIRTUALS:
            // [0] LeftInstructionInput, [1] RightInstructionInput, [2] OpFlags(Jump),
            // [3] OpFlags(WriteLookupOutputToRD), [4] LookupOutput, [5] InstructionFlags(Branch),
            // [6] NextIsNoop, [7] OpFlags(VirtualInstruction)
            if (comptime debug_verbose) {
                std.debug.print("[INSERT] LeftInstructionInput@ProdVirt = {any}\n", .{stage2_result.factor_evals[0].toBytesBE()});
            }
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .LeftInstructionInput, .sumcheck_id = .SpartanProductVirtualization } },
                stage2_result.factor_evals[0], // LeftInstructionInput
            );
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .RightInstructionInput, .sumcheck_id = .SpartanProductVirtualization } },
                stage2_result.factor_evals[1], // RightInstructionInput
            );
            // OpFlags::Jump = CircuitFlags index 5
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .{ .OpFlags = 5 }, .sumcheck_id = .SpartanProductVirtualization } },
                stage2_result.factor_evals[2], // Jump
            );
            // OpFlags::WriteLookupOutputToRD = CircuitFlags index 6
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .{ .OpFlags = 6 }, .sumcheck_id = .SpartanProductVirtualization } },
                stage2_result.factor_evals[3], // WriteLookupOutputToRD
            );
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .LookupOutput, .sumcheck_id = .SpartanProductVirtualization } },
                stage2_result.factor_evals[4], // LookupOutput
            );
            // InstructionFlags::Branch = index 4
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .{ .InstructionFlags = 4 }, .sumcheck_id = .SpartanProductVirtualization } },
                stage2_result.factor_evals[5], // Branch
            );
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .NextIsNoop, .sumcheck_id = .SpartanProductVirtualization } },
                stage2_result.factor_evals[6], // NextIsNoop
            );
            // OpFlags::VirtualInstruction = CircuitFlags index 7
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .{ .OpFlags = 7 }, .sumcheck_id = .SpartanProductVirtualization } },
                stage2_result.factor_evals[7], // VirtualInstruction
            );

            // Stage 2: OutputSumcheckVerifier claims
            // val_final_claim is the MLE evaluation Val_final(r') at the opening point
            // This comes from the OutputSumcheck prover's val_final polynomial after binding
            //
            // NOTE: We keep the original output_val_final_claim here because it's used in Stage 2's
            // expected_output_claim computation. The sumcheck polynomial rounds were generated based
            // on this value, so changing it would break Stage 2 verification.
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .RamValFinal, .sumcheck_id = .RamOutputCheck } },
                stage2_result.output_val_final_claim,
            );
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .RamValInit, .sumcheck_id = .RamOutputCheck } },
                stage2_result.output_val_init_claim,
            );

            errdefer stage2_result.deinit();

            // Stage 2: InstructionLookupsClaimReductionSumcheckVerifier claims
            // These are the MLE evaluations of the lookup polynomials at the sumcheck challenges
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .LookupOutput, .sumcheck_id = .InstructionClaimReduction } },
                stage2_result.instr_lookup_output_claim,
            );
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .LeftLookupOperand, .sumcheck_id = .InstructionClaimReduction } },
                stage2_result.instr_left_operand_claim,
            );
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .RightLookupOperand, .sumcheck_id = .InstructionClaimReduction } },
                stage2_result.instr_right_operand_claim,
            );
            if (comptime debug_verbose) {
                std.debug.print("[INSERT] LeftInstructionInput@InstrClaimRed = {any}\n", .{stage2_result.instr_left_instr_input_claim.toBytesBE()});
            }
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .LeftInstructionInput, .sumcheck_id = .InstructionClaimReduction } },
                stage2_result.instr_left_instr_input_claim,
            );
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .RightInstructionInput, .sumcheck_id = .InstructionClaimReduction } },
                stage2_result.instr_right_instr_input_claim,
            );

            // CRITICAL: Append Stage 2 cache_openings claims to transcript
            // Order follows upstream instance ordering:
            // [0] RamReadWriteChecking: 3 claims (RamVal, RamRa, RamInc)
            // [1] ProductVirtualRemainder: 8 claims (PRODUCT_UNIQUE_FACTOR_VIRTUALS)
            // [2] InstructionLookupsClaimReduction: 5 claims
            // [3] RamRafEvaluation: 1 claim (RamRa)
            // [4] OutputSumcheck: 1 claim (RamValFinal only; RamValInit is not opened)

            // Instance 0: RamReadWriteChecking - 3 claims
            transcript.appendScalar("opening_claim", stage2_result.rwc_val_claim);
            transcript.appendScalar("opening_claim", stage2_result.rwc_ra_claim);
            transcript.appendScalar("opening_claim", stage2_result.rwc_inc_claim);

            // Instance 1: ProductVirtualRemainder - 8 claims (PRODUCT_UNIQUE_FACTOR_VIRTUALS)
            for (stage2_result.factor_evals) |eval| {
                transcript.appendScalar("opening_claim", eval);
            }

            // Instance 2: InstructionLookupsClaimReduction - 2 flushed claims
            // LookupOutput, LeftInstructionInput, RightInstructionInput are aliased to
            // ProductVirtualRemainder's openings at the same point, so NOT flushed.
            // Only LeftLookupOperand and RightLookupOperand are new.
            transcript.appendScalar("opening_claim", stage2_result.instr_left_operand_claim);
            transcript.appendScalar("opening_claim", stage2_result.instr_right_operand_claim);

            // Instance 3: RamRafEvaluation - 1 claim
            transcript.appendScalar("opening_claim", stage2_result.raf_final_claim);

            // Instance 4: OutputSumcheck - 1 claim (only RamValFinal; RamValInit is NOT opened)
            transcript.appendScalar("opening_claim", stage2_result.output_val_final_claim);

            {
                const s2_claims_ns = bench_timer.read();
                const s2_total_ns = stage_timer.read();
                if (comptime stage_timing_enabled) {
                    std.debug.print("    [STAGE-TIMING] Stage 2: {d:.1} ms\n", .{@as(f64, @floatFromInt(s2_total_ns)) / 1_000_000.0});
                }
                if (bench) {
                    const ms = 1_000_000.0;
                    std.debug.print("[BENCH] stage=2 total={d:.1} init={d:.1} sumcheck={d:.1} claims={d:.1}\n", .{
                        @as(f64, @floatFromInt(s2_total_ns)) / ms,
                        @as(f64, @floatFromInt(s2_init_ns)) / ms,
                        @as(f64, @floatFromInt(s2_sumcheck_ns)) / ms,
                        @as(f64, @floatFromInt(s2_claims_ns)) / ms,
                    });
                }
            }
            stage_timer.reset();
            bench_timer.reset();

            return ProveStage2Output{
                .stage2_result = stage2_result,
            };
        }

        /// Stage 3: SpartanShift, InstructionInput, RegistersClaimReduction.
        fn executeStage3(
            self: *Self,
            jolt_proof: anytype,
            transcript: *Blake2bTranscript(F),
            config: *const JoltProverConfig,
            s2_out: *const ProveStage2Output,
            s1_out: *const ProveStage1Output,
            raw_r1cs_inputs: []const r1cs_evaluators.RawR1CSInputs,
            n_cycle_vars: usize,
            log_ram_k: usize,
            stage_timer: *std.time.Timer,
            bench_timer: *std.time.Timer,
        ) !ProveStage3Output {
            const bench = config.bench_output;
            const stage2_result = &s2_out.stage2_result;
            const r_spartan_original = s1_out.r_spartan_original;
            _ = log_ram_k;

            // Stage 3: SpartanShift, InstructionInput, RegistersClaimReduction
            // Extract r_product from Stage 2 challenges (last n_cycle_vars in BIG_ENDIAN)
            var r_product = try self.allocator.alloc(F, n_cycle_vars);
            defer self.allocator.free(r_product);
            {
                // Stage 2 challenges are in sumcheck order (LITTLE_ENDIAN)
                // ProductVirtualRemainder uses the last n_cycle_vars challenges
                // We need to reverse to get BIG_ENDIAN order
                const stage2_chals = stage2_result.challenges;
                const product_start = if (stage2_chals.len > n_cycle_vars) stage2_chals.len - n_cycle_vars else 0;
                for (0..n_cycle_vars) |i| {
                    const src_idx = n_cycle_vars - 1 - i;
                    if (product_start + src_idx < stage2_chals.len) {
                        r_product[i] = stage2_chals[product_start + src_idx];
                    } else {
                        r_product[i] = F.zero();
                    }
                }
            }

            const s3_init_ns = bench_timer.read();
            bench_timer.reset();

            // Generate Stage 3 proof using the proper sumcheck prover
            var stage3_prover_instance = Stage3Prover(F).init(self.allocator);
            stage3_prover_instance.thread_pool = self.thread_pool;
            stage3_prover_instance.gpu_ops = self.gpu_ops;
            var stage3_result = try stage3_prover_instance.generateStage3Proof(
                &jolt_proof.stage3_sumcheck_proof,
                transcript,
                &jolt_proof.opening_claims,
                raw_r1cs_inputs,
                n_cycle_vars,
                r_spartan_original, // r_outer in BIG_ENDIAN
                r_product, // r_product in BIG_ENDIAN
            );
            errdefer stage3_result.deinit();
            const s3_sumcheck_ns = bench_timer.read();
            bench_timer.reset();

            // Debug: Print Stage 3 challenges for comparison with Jolt
            // NOTE: Stage 3 challenges are MontU128Challenge-style [0, 0, low, high] limbs
            // where the limbs ARE the Montgomery representation directly.
            // To compare with Jolt's params.r_cycle, we need to look at limbs[2] and limbs[3].
            // Also print in the format that matches Jolt's params.r_cycle (16 zero bytes + 16 data bytes)
            for (0..stage3_result.challenges.len) |i| {
                const c = stage3_result.challenges[stage3_result.challenges.len - 1 - i];
                // Jolt's Challenge serializes as [0, 0, low_LE, high_LE] where each u64 is in LE bytes
                var jolt_format: [32]u8 = [_]u8{0} ** 32;
                std.mem.writeInt(u64, jolt_format[16..24], c.limbs[2], .little);
                std.mem.writeInt(u64, jolt_format[24..32], c.limbs[3], .little);
            }

            // DEBUG: Check challenges immediately before claiming them

            // SpartanShift claims (from Stage 3 prover)
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .UnexpandedPC, .sumcheck_id = .SpartanShift } },
                stage3_result.shift_unexpanded_pc_claim,
            );
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .PC, .sumcheck_id = .SpartanShift } },
                stage3_result.shift_pc_claim,
            );
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .{ .OpFlags = @intFromEnum(instruction.CircuitFlags.VirtualInstruction) }, .sumcheck_id = .SpartanShift } },
                stage3_result.shift_is_virtual_claim,
            );
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .{ .OpFlags = @intFromEnum(instruction.CircuitFlags.IsFirstInSequence) }, .sumcheck_id = .SpartanShift } },
                stage3_result.shift_is_first_in_sequence_claim,
            );
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .{ .InstructionFlags = @intFromEnum(instruction.InstructionFlags.IsNoop) }, .sumcheck_id = .SpartanShift } },
                stage3_result.shift_is_noop_claim,
            );

            // InstructionInputVirtualization claims (from Stage 3 prover)
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .{ .InstructionFlags = @intFromEnum(instruction.InstructionFlags.LeftOperandIsRs1Value) }, .sumcheck_id = .InstructionInputVirtualization } },
                stage3_result.instr_left_is_rs1_claim,
            );
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .Rs1Value, .sumcheck_id = .InstructionInputVirtualization } },
                stage3_result.instr_rs1_value_claim,
            );
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .{ .InstructionFlags = @intFromEnum(instruction.InstructionFlags.LeftOperandIsPC) }, .sumcheck_id = .InstructionInputVirtualization } },
                stage3_result.instr_left_is_pc_claim,
            );
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .UnexpandedPC, .sumcheck_id = .InstructionInputVirtualization } },
                stage3_result.instr_unexpanded_pc_claim,
            );
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .{ .InstructionFlags = @intFromEnum(instruction.InstructionFlags.RightOperandIsRs2Value) }, .sumcheck_id = .InstructionInputVirtualization } },
                stage3_result.instr_right_is_rs2_claim,
            );
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .Rs2Value, .sumcheck_id = .InstructionInputVirtualization } },
                stage3_result.instr_rs2_value_claim,
            );
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .{ .InstructionFlags = @intFromEnum(instruction.InstructionFlags.RightOperandIsImm) }, .sumcheck_id = .InstructionInputVirtualization } },
                stage3_result.instr_right_is_imm_claim,
            );
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .Imm, .sumcheck_id = .InstructionInputVirtualization } },
                stage3_result.instr_imm_claim,
            );

            // RegistersClaimReduction claims (from Stage 3 prover)
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .RdWriteValue, .sumcheck_id = .RegistersClaimReduction } },
                stage3_result.reg_rd_write_value_claim,
            );
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .Rs1Value, .sumcheck_id = .RegistersClaimReduction } },
                stage3_result.reg_rs1_value_claim,
            );
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .Rs2Value, .sumcheck_id = .RegistersClaimReduction } },
                stage3_result.reg_rs2_value_claim,
            );

            // BytecodeReadRaf claims (InstructionReadRaf in Jolt)
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .InstructionRafFlag, .sumcheck_id = .BytecodeReadRaf } },
                F.zero(),
            );
            // InstructionRa chunks (just add first chunk for now)
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .{ .InstructionRa = 0 }, .sumcheck_id = .BytecodeReadRaf } },
                F.zero(),
            );

            // IncClaimReduction claims (Increment checking)
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .RamRa, .sumcheck_id = .IncClaimReduction } },
                F.zero(),
            );

            // LookupOutput at InstructionClaimReduction was already added in Stage 2

            {
                const s3_claims_ns = bench_timer.read();
                const s3_total_ns = stage_timer.read();
                if (comptime stage_timing_enabled) {
                    std.debug.print("    [STAGE-TIMING] Stage 3: {d:.1} ms\n", .{@as(f64, @floatFromInt(s3_total_ns)) / 1_000_000.0});
                }
                if (bench) {
                    const ms = 1_000_000.0;
                    std.debug.print("[BENCH] stage=3 total={d:.1} init={d:.1} sumcheck={d:.1} claims={d:.1}\n", .{
                        @as(f64, @floatFromInt(s3_total_ns)) / ms,
                        @as(f64, @floatFromInt(s3_init_ns)) / ms,
                        @as(f64, @floatFromInt(s3_sumcheck_ns)) / ms,
                        @as(f64, @floatFromInt(s3_claims_ns)) / ms,
                    });
                }
            }
            stage_timer.reset();
            bench_timer.reset();

            return ProveStage3Output{
                .stage3_result = stage3_result,
            };
        }

        /// Stage 4: RegistersReadWriteChecking, RamValEvaluation, RamValFinalEvaluation.
        fn executeStage4(
            self: *Self,
            jolt_proof: anytype,
            transcript: *Blake2bTranscript(F),
            config: *const JoltProverConfig,
            s2_out: *const ProveStage2Output,
            s3_out: *const ProveStage3Output,
            n_cycle_vars: usize,
            log_ram_k: usize,
            ram_K: usize,
            trace_length: usize,
            stage_timer: *std.time.Timer,
            bench_timer: *std.time.Timer,
        ) !ProveStage4Output {
            const bench = config.bench_output;
            const stage2_result = &s2_out.stage2_result;
            const stage3_result = &s3_out.stage3_result;
            _ = trace_length;

            const trace = config.execution_trace orelse return error.MissingExecutionTrace;
            const memory_trace = config.memory_trace orelse return error.MissingMemoryTrace;

            // Compute init_eval for ValEvaluation at the RWC r_address
            // (computeInitialRamEval stays here since stage2_sumcheck.zig also uses it)
            const init_eval_for_val_eval = blk: {
                if (config.memory_layout) |ml| {
                    // Extract r_address_be from Stage 2 RWC phase challenges for init_eval computation
                    const phase1 = jolt_proof.rw_config.ram_rw_phase1_num_rounds;
                    const phase2 = jolt_proof.rw_config.ram_rw_phase2_num_rounds;
                    const phase3_cycle_len = n_cycle_vars - phase1;
                    const phase3_address_len = log_ram_k - phase2;

                    var r_address_be = try self.allocator.alloc(F, log_ram_k);
                    defer self.allocator.free(r_address_be);
                    @memset(r_address_be, F.zero());

                    const phase2_start = phase1;
                    for (0..phase2) |i| {
                        const src_idx = phase2_start + i;
                        if (src_idx < stage2_result.challenges.len) {
                            const dest_idx = phase3_address_len + (phase2 - 1 - i);
                            if (dest_idx < log_ram_k) {
                                r_address_be[dest_idx] = stage2_result.challenges[src_idx];
                            }
                        }
                    }
                    const phase3_addr_start = phase1 + phase2 + phase3_cycle_len;
                    for (0..phase3_address_len) |i| {
                        const src_idx = phase3_addr_start + i;
                        if (src_idx < stage2_result.challenges.len) {
                            const dest_idx = phase3_address_len - 1 - i;
                            r_address_be[dest_idx] = stage2_result.challenges[src_idx];
                        }
                    }

                    break :blk computeInitialRamEval(
                        config.bytecode_words,
                        config.min_bytecode_address,
                        ml,
                        r_address_be,
                        log_ram_k,
                        config.program_inputs,
                    );
                }
                break :blk F.zero();
            };

            const start_address: u64 = if (config.memory_layout) |ml|
                ml.getLowestAddress()
            else
                constants.RAM_START_ADDRESS;

            bench_timer.reset();

            // Delegate to the extracted Stage 4 prover
            const Stage4ProverType = spartan_mod.stage4_prover_mod.Stage4Prover(F);
            const Stage3ClaimsType = spartan_mod.stage4_gruen_prover.Stage3Claims(F);

            var stage4_prover = Stage4ProverType.init(self.allocator);
            stage4_prover.thread_pool = self.thread_pool;
            stage4_prover.gpu_ops = self.gpu_ops;

            const stage4_result = try stage4_prover.generateStage4Proof(
                &jolt_proof.stage4_sumcheck_proof,
                transcript,
                stage2_result.challenges,
                stage2_result.rwc_val_claim,
                stage2_result.output_val_final_claim,
                jolt_proof.rw_config.ram_rw_phase1_num_rounds,
                jolt_proof.rw_config.ram_rw_phase2_num_rounds,
                Stage3ClaimsType{
                    .rd_write_value = stage3_result.reg_rd_write_value_claim,
                    .rs1_value = stage3_result.reg_rs1_value_claim,
                    .rs2_value = stage3_result.reg_rs2_value_claim,
                },
                stage3_result.challenges,
                trace,
                memory_trace,
                config.initial_ram,
                n_cycle_vars,
                log_ram_k,
                ram_K,
                init_eval_for_val_eval,
                config.memory_layout,
                start_address,
            );

            const s4_sumcheck_ns = bench_timer.read();
            bench_timer.reset();

            // Insert opening claims into accumulator
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .RegistersVal, .sumcheck_id = .RegistersReadWriteChecking } },
                stage4_result.regs_val_claim,
            );
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .Rs1Ra, .sumcheck_id = .RegistersReadWriteChecking } },
                stage4_result.regs_rs1_ra_claim,
            );
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .Rs2Ra, .sumcheck_id = .RegistersReadWriteChecking } },
                stage4_result.regs_rs2_ra_claim,
            );
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .RdWa, .sumcheck_id = .RegistersReadWriteChecking } },
                stage4_result.regs_rd_wa_claim,
            );
            try jolt_proof.opening_claims.insert(
                .{ .Committed = .{ .poly = .RdInc, .sumcheck_id = .RegistersReadWriteChecking } },
                stage4_result.regs_inc_claim,
            );
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .RamRa, .sumcheck_id = .RamValCheck } },
                stage4_result.val_eval_wa_claim,
            );
            try jolt_proof.opening_claims.insert(
                .{ .Committed = .{ .poly = .RamInc, .sumcheck_id = .RamValCheck } },
                stage4_result.val_eval_inc_claim,
            );

            {
                const s4_total_ns = stage_timer.read();
                if (comptime stage_timing_enabled) {
                    std.debug.print("    [STAGE-TIMING] Stage 4: {d:.1} ms\n", .{@as(f64, @floatFromInt(s4_total_ns)) / 1_000_000.0});
                }
                if (bench) {
                    const ms = 1_000_000.0;
                    const s4_cl = bench_timer.read();
                    const s4_init = s4_total_ns -| s4_sumcheck_ns -| s4_cl;
                    std.debug.print("[BENCH] stage=4 total={d:.1} init={d:.1} sumcheck={d:.1} claims={d:.1}\n", .{
                        @as(f64, @floatFromInt(s4_total_ns)) / ms,
                        @as(f64, @floatFromInt(s4_init)) / ms,
                        @as(f64, @floatFromInt(s4_sumcheck_ns)) / ms,
                        @as(f64, @floatFromInt(s4_cl)) / ms,
                    });
                }
            }
            stage_timer.reset();
            bench_timer.reset();

            return ProveStage4Output{
                .stage4_regs_r_address = stage4_result.regs_r_address,
                .stage4_regs_r_cycle = stage4_result.regs_r_cycle,
                .stage4_r_cycle_val = stage4_result.r_cycle_val,
                .r_reduction_be = stage4_result.r_reduction_be,
                .stage4_inc_poly_copy = stage4_result.inc_poly_copy,
                .allocator = self.allocator,
            };
        }

        /// Stage 5: RegistersValEvaluation, RamRaClaimReduction, LookupsReadRaf.
        fn executeStage5(
            self: *Self,
            jolt_proof: anytype,
            transcript: *Blake2bTranscript(F),
            config: *const JoltProverConfig,
            s1_out: *const ProveStage1Output,
            s2_out: *const ProveStage2Output,
            s4_out: *const ProveStage4Output,
            n_cycle_vars: usize,
            log_ram_k: usize,
            ram_K: usize,
            stage_timer: *std.time.Timer,
            bench_timer: *std.time.Timer,
        ) !ProveStage5Output {
            const bench = config.bench_output;
            const stage2_result = &s2_out.stage2_result;
            const r_spartan_original = s1_out.r_spartan_original;
            const stage4_regs_r_address = s4_out.stage4_regs_r_address;
            const stage4_regs_r_cycle = s4_out.stage4_regs_r_cycle;
            const stage4_r_cycle_val = s4_out.stage4_r_cycle_val;
            const r_reduction_be = s4_out.r_reduction_be;
            _ = ram_K;

            // Stage 5: RegistersValEvaluation, RamRaClaimReduction, LookupsReadRaf
            // LookupsReadRaf has max rounds: LOG_K + log_T where LOG_K = XLEN * 2 = 128
            // For RV64: max_num_rounds = 128 + log_T = 128 + 8 = 136
            const lookups_log_k: usize = 128; // XLEN * 2 for RV64

            // CRITICAL: Jolt samples TWO separate gammas for Stage 5 instances.
            // The verifier creates instances in this order:
            //   1. InstructionReadRafSumcheckVerifier::new() → squeezes gamma_lookups_raf
            //   2. RamRaClaimReductionSumcheckVerifier::new() → squeezes gamma_ram_ra
            // So we must squeeze in the SAME order.
            const gamma_lookups_raf = transcript.challengeScalarFull();
            const gamma_ram_ra = transcript.challengeScalarFull();

            const s5_init_ns = bench_timer.read();
            bench_timer.reset();

            // Generate Stage 5 proof using the batched sumcheck prover
            var stage5_prover_instance = Stage5BatchedProver(F).init(self.allocator);
            stage5_prover_instance.thread_pool = self.thread_pool;
            stage5_prover_instance.gpu_ops = self.gpu_ops;
            var stage5_result: spartan_mod.Stage5Result(F) = undefined;

            // Use trace-aware prover if we have trace data and Stage 4 opening point
            if (config.execution_trace != null and stage4_regs_r_address != null and stage4_regs_r_cycle != null and r_reduction_be != null) {
                stage5_result = try stage5_prover_instance.generateStage5ProofWithTrace(
                    &jolt_proof.stage5_sumcheck_proof,
                    transcript,
                    &jolt_proof.opening_claims,
                    n_cycle_vars,
                    log_ram_k,
                    gamma_ram_ra,
                    gamma_lookups_raf,
                    config.lookups_ra_virtual_log_k_chunk,
                    config.execution_trace.?,
                    config.memory_trace, // RAM trace for ram_ra_claim computation
                    config.memory_layout, // Memory layout for address remapping
                    stage4_regs_r_address.?,
                    stage4_regs_r_cycle.?,
                    r_reduction_be.?, // Stage 3 challenges in BIG_ENDIAN for LookupsReadRaf eq computation
                    // RamRaClaimReduction opening points:
                    stage2_result.r_address_raf, // r_address_1 from RamRafEvaluation
                    stage2_result.r_address_rw, // r_address_2 from RamReadWriteChecking
                    r_spartan_original, // r_cycle_raf from SpartanOuter (Stage 1)
                    stage2_result.r_cycle_rw, // r_cycle_rw from RamReadWriteChecking
                    stage4_r_cycle_val.?, // r_cycle_val from RamValEvaluation (Stage 4)
                );
            } else {
                // Fallback to zero prover for programs without trace
                stage5_result = try stage5_prover_instance.generateStage5Proof(
                    &jolt_proof.stage5_sumcheck_proof,
                    transcript,
                    &jolt_proof.opening_claims,
                    n_cycle_vars,
                    log_ram_k,
                    gamma_ram_ra,
                    gamma_lookups_raf,
                    config.lookups_ra_virtual_log_k_chunk,
                );
            }
            errdefer stage5_result.deinit();
            const s5_sumcheck_ns = bench_timer.read();
            bench_timer.reset();

            // RegistersValEvaluation claims
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .RdWa, .sumcheck_id = .RegistersValEvaluation } },
                stage5_result.regs_val_wa_claim,
            );
            try jolt_proof.opening_claims.insert(
                .{ .Committed = .{ .poly = .RdInc, .sumcheck_id = .RegistersValEvaluation } },
                stage5_result.regs_val_inc_claim,
            );

            // RamRaClaimReduction claims
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .RamRa, .sumcheck_id = .RamRaClaimReduction } },
                stage5_result.ram_ra_claim,
            );

            // LookupsReadRaf claims (Stage 5 - LookupsReadRafSumcheckVerifier)
            // LookupTableFlag(i) for each of the 40 lookup tables
            const num_lookup_tables: usize = 40; // LookupTables::<XLEN>::COUNT (40 variants, ValidSignedRemainder removed)
            for (0..num_lookup_tables) |i| {
                const flag_value = stage5_result.lookups_table_flags[i];
                if (!flag_value.eql(F.zero())) {
                    // Convert to standard form for printing (same as serialization)
                    const standard = flag_value.fromMontgomery();
                    var buf: [32]u8 = undefined;
                    for (0..4) |j| {
                        std.mem.writeInt(u64, buf[j * 8 ..][0..8], standard.limbs[j], .little);
                    }
                }
                try jolt_proof.opening_claims.insert(
                    .{ .Virtual = .{ .poly = .{ .LookupTableFlag = i }, .sumcheck_id = .InstructionReadRaf } },
                    flag_value,
                );
            }

            // InstructionRa(i) chunks for LookupsReadRaf (LOG_K / ra_virtual_log_k_chunk = 128 / 16 = 8 chunks)
            const lookups_ra_d: usize = lookups_log_k / config.lookups_ra_virtual_log_k_chunk;
            for (0..lookups_ra_d) |i| {
                try jolt_proof.opening_claims.insert(
                    .{ .Virtual = .{ .poly = .{ .InstructionRa = i }, .sumcheck_id = .InstructionReadRaf } },
                    stage5_result.lookups_ra_chunks[i],
                );
            }

            // InstructionRafFlag for LookupsReadRaf
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .InstructionRafFlag, .sumcheck_id = .InstructionReadRaf } },
                stage5_result.lookups_raf_flag,
            );

            // Append Stage 5 cache openings to transcript
            // Order must match upstream instance order: InstructionReadRaf, RamRaClaimReduction, RegistersValEvaluation

            // Instance 0: LookupsReadRaf (LookupTableFlag(0..41), InstructionRa(0..8), InstructionRafFlag)
            for (stage5_result.lookups_table_flags) |flag| {
                transcript.appendScalar("opening_claim", flag);
            }
            for (stage5_result.lookups_ra_chunks) |chunk| {
                transcript.appendScalar("opening_claim", chunk);
            }
            transcript.appendScalar("opening_claim", stage5_result.lookups_raf_flag);

            // Instance 1: RamRaClaimReduction (RamRa)
            transcript.appendScalar("opening_claim", stage5_result.ram_ra_claim);

            // Instance 2: RegistersValEvaluation (RdInc, RdWa)
            transcript.appendScalar("opening_claim", stage5_result.regs_val_inc_claim);
            transcript.appendScalar("opening_claim", stage5_result.regs_val_wa_claim);

            {
                const s5_claims_ns = bench_timer.read();
                const s5_total_ns = stage_timer.read();
                if (comptime stage_timing_enabled) {
                    std.debug.print("    [STAGE-TIMING] Stage 5: {d:.1} ms\n", .{@as(f64, @floatFromInt(s5_total_ns)) / 1_000_000.0});
                }
                if (bench) {
                    const ms = 1_000_000.0;
                    std.debug.print("[BENCH] stage=5 total={d:.1} init={d:.1} sumcheck={d:.1} claims={d:.1}\n", .{
                        @as(f64, @floatFromInt(s5_total_ns)) / ms,
                        @as(f64, @floatFromInt(s5_init_ns)) / ms,
                        @as(f64, @floatFromInt(s5_sumcheck_ns)) / ms,
                        @as(f64, @floatFromInt(s5_claims_ns)) / ms,
                    });
                }
            }
            stage_timer.reset();
            bench_timer.reset();

            // Compute Stage 5 RegistersValEvaluation opening point (s_cycle_stage5)
            const lookups_log_k_local: usize = 128;
            const stage5_lookups_num_rounds = lookups_log_k_local + n_cycle_vars;
            const stage5_regs_val_num_rounds = n_cycle_vars;
            var s_cycle_stage5 = try self.allocator.alloc(F, n_cycle_vars);
            for (0..n_cycle_vars) |i| {
                const stage5_idx = stage5_lookups_num_rounds - stage5_regs_val_num_rounds + i;
                s_cycle_stage5[n_cycle_vars - 1 - i] = stage5_result.challenges[stage5_idx];
            }

            return ProveStage5Output{
                .stage5_result = stage5_result,
                .s_cycle_stage5 = s_cycle_stage5,
                .allocator = self.allocator,
            };
        }

        /// Stage 6: BytecodeReadRaf, RamHammingBooleanity, Booleanity, RamRaVirtual, LookupsRaVirtual, IncClaimReduction.
        fn executeStage6(
            self: *Self,
            jolt_proof: anytype,
            transcript: *Blake2bTranscript(F),
            config: *const JoltProverConfig,
            s1_out: *const ProveStage1Output,
            s2_out: *const ProveStage2Output,
            s3_out: *const ProveStage3Output,
            s4_out: *const ProveStage4Output,
            s5_out: *const ProveStage5Output,
            n_cycle_vars: usize,
            log_ram_k: usize,
            ram_K: usize,
            stage_timer: *std.time.Timer,
            bench_timer: *std.time.Timer,
        ) !ProveStage6Output {
            const bench = config.bench_output;
            const stage2_result = &s2_out.stage2_result;
            const stage3_result = &s3_out.stage3_result;
            const stage4_regs_r_address = s4_out.stage4_regs_r_address;
            const stage4_regs_r_cycle = s4_out.stage4_regs_r_cycle;
            const stage4_r_cycle_val = s4_out.stage4_r_cycle_val;
            const stage4_inc_poly_copy = s4_out.stage4_inc_poly_copy;
            const stage5_result = &s5_out.stage5_result;
            const s_cycle_stage5 = s5_out.s_cycle_stage5;
            const r_spartan_original = s1_out.r_spartan_original;
            _ = log_ram_k;

            // Stage 6: BytecodeReadRaf, RamHammingBooleanity, Booleanity, RamRaVirtual, LookupsRaVirtual, IncClaimReduction
            const lookups_log_k: usize = 128; // XLEN * 2 for RV64
            const bytecode_log_k = std.math.log2_int(usize, config.bytecode_K);
            const ram_log_k = std.math.log2_int(usize, ram_K);
            const instruction_d: usize = (lookups_log_k + config.log_k_chunk - 1) / config.log_k_chunk;
            const bytecode_d_val: usize = (bytecode_log_k + config.log_k_chunk - 1) / config.log_k_chunk;
            const ram_d_val: usize = (ram_log_k + config.log_k_chunk - 1) / config.log_k_chunk;

            // Generate Stage 6 proof using the batched sumcheck prover
            const the_trace = config.execution_trace orelse return error.ExecutionTraceRequired;
            const the_memory_layout = config.memory_layout orelse return error.MemoryLayoutRequired;

            // Compute SpartanShift r_cycle in BIG_ENDIAN from Stage 3 challenges (reversed)
            const r_cycle_shift_be = try self.allocator.alloc(F, stage3_result.challenges.len);
            defer self.allocator.free(r_cycle_shift_be);
            for (0..stage3_result.challenges.len) |i| {
                r_cycle_shift_be[i] = stage3_result.challenges[stage3_result.challenges.len - 1 - i];
            }

            // Build bytecode entry table from static ELF + execution trace overlay
            const bytecode_K_val: usize = @as(usize, 1) << @intCast(bytecode_log_k);
            const stage6_mod = @import("spartan/stage6_prover.zig");
            // Get pc_map for converting ELF addresses to bytecode array indices
            const pc_map_ptr = config.bytecode_pc_map orelse return error.MissingBytecodepcMap;
            const bytecode_entries = try stage6_mod.buildBytecodeEntries(self.allocator, the_trace, bytecode_K_val, pc_map_ptr, config.program_code_bytes, config.code_base_address, the_memory_layout.termination, config.text_size, config.bytecode_preprocessing);
            defer self.allocator.free(bytecode_entries);

            // Get register address opening points for Stages 4 and 5
            // Stage 4: from RegistersReadWriteChecking (address portion)
            const r_register_4 = stage4_regs_r_address orelse &[_]F{};
            // Stage 5: use same as Stage 4 (both address 32 registers)
            // In Jolt, this comes from RegistersValEvaluation's opening point split,
            // but the address variables are the SAME as Stage 4's since they share
            // the same register address space.
            const r_register_5 = stage4_regs_r_address orelse &[_]F{};

            const s6_init_ns = bench_timer.read();
            bench_timer.reset();

            var stage6_prover_instance = Stage6BatchedProver(F).init(self.allocator);
            stage6_prover_instance.thread_pool = self.thread_pool;
            stage6_prover_instance.gpu_ops = self.gpu_ops;
            var stage6_result = try stage6_prover_instance.generateStage6Proof(
                &jolt_proof.stage6_sumcheck_proof,
                transcript,
                &jolt_proof.opening_claims,
                n_cycle_vars,
                bytecode_log_k,
                config.log_k_chunk,
                bytecode_d_val,
                ram_d_val,
                instruction_d,
                config.lookups_ra_virtual_log_k_chunk,
                // Execution trace
                the_trace,
                // BytecodeReadRaf r_cycles (all BIG_ENDIAN)
                r_spartan_original, // r_cycle_bc1: SpartanOuter
                stage2_result.r_cycle_product, // r_cycle_bc2: SpartanProductVirtualization
                r_cycle_shift_be, // r_cycle_bc3: SpartanShift
                if (stage4_regs_r_cycle) |v| v else s_cycle_stage5, // r_cycle_bc4: RegistersReadWriteChecking
                s_cycle_stage5, // r_cycle_bc5: RegistersValEvaluation
                // IncClaimReduction r_cycles (all BIG_ENDIAN)
                stage2_result.r_cycle_rw, // r_cycle_inc: RamReadWriteChecking
                if (stage4_r_cycle_val) |v| v else s_cycle_stage5, // r_cycle_inc: RamValEvaluation
                // Stage 5 challenges for deriving LookupsRaVirtual and RamRaVirtual points
                stage5_result.challenges,
                // RAM r_address from Stage 2 (BIG_ENDIAN) — aligned address used by RamRaClaimReduction
                stage2_result.r_address_raf,
                // Memory layout for address remapping
                the_memory_layout,
                // Bytecode entries for Val polynomial computation
                bytecode_entries,
                // Register address opening points
                r_register_4,
                r_register_5,
                // BytecodePCMapper for converting ELF addresses to bytecode array indices
                pc_map_ptr,
                // ELF entry point address for entry-point constraint (PR #1335)
                config.entry_address,
                // Stage 4 inc_poly copy for diagnostic
                if (stage4_inc_poly_copy) |v| v else &[_]F{},
            );
            errdefer stage6_result.deinit();
            const s6_sumcheck_ns = bench_timer.read();
            bench_timer.reset();

            // Insert Stage 6 opening claims into accumulator

            // HammingBooleanity: virtual RamHammingWeight
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .RamHammingWeight, .sumcheck_id = .Booleanity } },
                stage6_result.hamming_weight_claim,
            );
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .RamHammingWeight, .sumcheck_id = .RamHammingBooleanity } },
                stage6_result.hamming_weight_claim,
            );

            // IncClaimReduction: committed RamInc, RdInc
            try jolt_proof.opening_claims.insert(
                .{ .Committed = .{ .poly = .RamInc, .sumcheck_id = .IncClaimReduction } },
                stage6_result.ram_inc_claim,
            );
            try jolt_proof.opening_claims.insert(
                .{ .Committed = .{ .poly = .RdInc, .sumcheck_id = .IncClaimReduction } },
                stage6_result.rd_inc_claim,
            );

            // BytecodeReadRaf cache_openings: BytecodeRa(i)
            for (0..bytecode_d_val) |i| {
                try jolt_proof.opening_claims.insert(
                    .{ .Committed = .{ .poly = .{ .BytecodeRa = i }, .sumcheck_id = .BytecodeReadRaf } },
                    stage6_result.bytecode_ra_claims[i],
                );
            }

            // Booleanity cache_openings: all RA polys
            // Order: InstructionRa(0..instruction_d), BytecodeRa(0..bytecode_d), RamRa(0..ram_d)
            var bool_idx: usize = 0;
            for (0..instruction_d) |i| {
                try jolt_proof.opening_claims.insert(
                    .{ .Committed = .{ .poly = .{ .InstructionRa = i }, .sumcheck_id = .Booleanity } },
                    stage6_result.booleanity_ra_claims[bool_idx],
                );
                bool_idx += 1;
            }
            for (0..bytecode_d_val) |i| {
                try jolt_proof.opening_claims.insert(
                    .{ .Committed = .{ .poly = .{ .BytecodeRa = i }, .sumcheck_id = .Booleanity } },
                    stage6_result.booleanity_ra_claims[bool_idx],
                );
                bool_idx += 1;
            }
            for (0..ram_d_val) |i| {
                try jolt_proof.opening_claims.insert(
                    .{ .Committed = .{ .poly = .{ .RamRa = i }, .sumcheck_id = .Booleanity } },
                    stage6_result.booleanity_ra_claims[bool_idx],
                );
                bool_idx += 1;
            }

            // RamRaVirtualization cache_openings: RamRa(i)
            for (0..ram_d_val) |i| {
                try jolt_proof.opening_claims.insert(
                    .{ .Committed = .{ .poly = .{ .RamRa = i }, .sumcheck_id = .RamRaVirtualization } },
                    stage6_result.ram_ra_virtual_claims[i],
                );
            }

            // InstructionRaVirtualization cache_openings: InstructionRa(i)
            for (0..instruction_d) |i| {
                try jolt_proof.opening_claims.insert(
                    .{ .Committed = .{ .poly = .{ .InstructionRa = i }, .sumcheck_id = .InstructionRaVirtualization } },
                    stage6_result.instruction_ra_virtual_claims[i],
                );
            }

            // Stage 6 cache_openings are already appended by Stage 6 prover
            // (stage6_prover.zig lines 4055-4083)
            // Do NOT re-append them here.

            {
                const s6_claims_ns = bench_timer.read();
                const s6_total_ns = stage_timer.read();
                if (comptime stage_timing_enabled) {
                    std.debug.print("    [STAGE-TIMING] Stage 6: {d:.1} ms\n", .{@as(f64, @floatFromInt(s6_total_ns)) / 1_000_000.0});
                }
                if (bench) {
                    const ms = 1_000_000.0;
                    std.debug.print("[BENCH] stage=6 total={d:.1} init={d:.1} sumcheck={d:.1} claims={d:.1}\n", .{
                        @as(f64, @floatFromInt(s6_total_ns)) / ms,
                        @as(f64, @floatFromInt(s6_init_ns)) / ms,
                        @as(f64, @floatFromInt(s6_sumcheck_ns)) / ms,
                        @as(f64, @floatFromInt(s6_claims_ns)) / ms,
                    });
                }
            }
            stage_timer.reset();
            bench_timer.reset();

            return ProveStage6Output{
                .stage6_result = stage6_result,
            };
        }

        /// Stage 7: HammingWeightClaimReduction sumcheck.
        fn executeStage7(
            self: *Self,
            jolt_proof: anytype,
            transcript: *Blake2bTranscript(F),
            config: *const JoltProverConfig,
            s2_out: *const ProveStage2Output,
            s5_out: *const ProveStage5Output,
            s6_out: *const ProveStage6Output,
            n_cycle_vars: usize,
            stage_timer: *std.time.Timer,
            bench_timer: *std.time.Timer,
        ) !void {
            const bench = config.bench_output;
            const stage2_result = &s2_out.stage2_result;
            const stage5_result = &s5_out.stage5_result;
            const stage6_result = &s6_out.stage6_result;
            const the_trace = config.execution_trace orelse return error.ExecutionTraceRequired;
            const the_memory_layout = config.memory_layout orelse return error.MemoryLayoutRequired;
            const pc_map_ptr = config.bytecode_pc_map orelse return error.MissingBytecodepcMap;
            _ = n_cycle_vars;

            // Delegate to the extracted Stage 7 prover
            bench_timer.reset();

            const Stage7ProverType = spartan_mod.stage7_prover_mod.Stage7Prover(F);
            var stage7_prover = Stage7ProverType.init(self.allocator);
            stage7_prover.thread_pool = self.thread_pool;

            var stage7_result = try stage7_prover.generateStage7Proof(
                &jolt_proof.stage7_sumcheck_proof,
                transcript,
                stage6_result,
                stage5_result.challenges,
                stage2_result.r_address_raf,
                the_trace,
                the_memory_layout,
                pc_map_ptr,
            );
            defer stage7_result.deinit();

            // Store opening point on proof
            jolt_proof.opening_point = stage7_result.opening_point;

            // Insert G_i claims into opening_claims accumulator
            const s6_instruction_d_val = stage6_result.instruction_d;
            const s6_bytecode_d_val = stage6_result.bytecode_d;
            for (0..stage7_result.g_claims.len) |i| {
                const key: jolt_types.OpeningId = blk: {
                    if (i < s6_instruction_d_val) {
                        break :blk .{ .Committed = .{ .poly = .{ .InstructionRa = i }, .sumcheck_id = .HammingWeightClaimReduction } };
                    } else if (i < s6_instruction_d_val + s6_bytecode_d_val) {
                        break :blk .{ .Committed = .{ .poly = .{ .BytecodeRa = i - s6_instruction_d_val }, .sumcheck_id = .HammingWeightClaimReduction } };
                    } else {
                        break :blk .{ .Committed = .{ .poly = .{ .RamRa = i - s6_instruction_d_val - s6_bytecode_d_val }, .sumcheck_id = .HammingWeightClaimReduction } };
                    }
                };
                try jolt_proof.opening_claims.insert(key, stage7_result.g_claims[i]);
            }

            {
                const s7_sumcheck_ns = bench_timer.read();
                const s7_total_ns = stage_timer.read();
                if (comptime stage_timing_enabled) {
                    std.debug.print("    [STAGE-TIMING] Stage 7: {d:.1} ms\n", .{@as(f64, @floatFromInt(s7_total_ns)) / 1_000_000.0});
                }
                if (bench) {
                    const ms = 1_000_000.0;
                    std.debug.print("[BENCH] stage=7 total={d:.1} sumcheck={d:.1}\n", .{
                        @as(f64, @floatFromInt(s7_total_ns)) / ms,
                        @as(f64, @floatFromInt(s7_sumcheck_ns)) / ms,
                    });
                }
            }
        }

        const stage2_sumcheck = @import("spartan/stage2_sumcheck.zig").Stage2Sumcheck(F);

        /// Re-export from stage2_sumcheck.zig
        const Stage2Result = stage2_sumcheck.Stage2Result;

        /// Generate Stage 2 batched sumcheck proof — delegates to stage2_sumcheck.zig
        fn generateStage2BatchedSumcheckProof(
            self: *Self,
            proof: *SumcheckInstanceProof(F),
            transcript: *Blake2bTranscript(F),
            r0_stage2: F,
            uni_skip_claim_stage2: F,
            tau: []const F,
            r_spartan_for_instr: []const F,
            raw_r1cs_inputs: []const r1cs_evaluators.RawR1CSInputs,
            n_cycle_vars: usize,
            log_ram_k: usize,
            opening_claims: *OpeningClaims(F),
            config: JoltProverConfig,
        ) !Stage2Result {
            return stage2_sumcheck.generateBatchedSumcheckProof(
                .{
                    .allocator = self.allocator,
                    .thread_pool = self.thread_pool,
                    .gpu_ops = self.gpu_ops,
                },
                proof,
                transcript,
                r0_stage2,
                uni_skip_claim_stage2,
                tau,
                r_spartan_for_instr,
                raw_r1cs_inputs,
                n_cycle_vars,
                log_ram_k,
                opening_claims,
                config,
            );
        }

        /// Evaluate polynomial at challenge using Jolt's eval_from_hint formula
        /// Delegates to the shared UniPoly implementation.
        fn evalFromHint(compressed: [3]F, hint: F, x: F) F {
            return poly_mod.UniPoly(F).evalFromHint(compressed, hint, x);
        }

        /// Compute eq(r, idx) where r is in BIG_ENDIAN order (MSB first).
        pub fn computeEqAtPointBigEndian(r: []const F, idx: usize) F {
            return @import("eq_utils.zig").computeEqAtPointBE(F, r, idx);
        }

        /// Compute eq(r, idx) where r is in LITTLE_ENDIAN order (LSB first).
        /// bit i of idx corresponds to r[i].
        fn computeEqAtPointLE(r: []const F, idx: usize) F {
            return @import("eq_utils.zig").computeEqAtPointLE(F, r, idx);
        }

        /// Evaluate the initial RAM polynomial at r_address (BIG_ENDIAN).
        ///
        /// This matches Jolt's `eval_initial_ram_mle` which evaluates:
        ///   sum_k bytecode_words[k] * eq(r_address, bytecode_start + k)
        /// where bytecode_start = remap_address(min_bytecode_address)
        ///
        /// NOTE: Unlike the old implementation that used initial_ram hashmap (stack data),
        /// this now uses bytecode_words (program bytecode) like Jolt does.
        pub fn computeInitialRamEval(
            bytecode_words: ?[]const u64,
            min_bytecode_address: u64,
            memory_layout: *const jolt_device.MemoryLayout,
            r_address_be: []const F,
            log_ram_k: usize,
            program_inputs: ?[]const u8,
        ) F {
            const lowest_address = memory_layout.getLowestAddress();

            var result = F.zero();
            const max_idx: usize = @as(usize, 1) << @intCast(log_ram_k);

            // Evaluate bytecode region (like Jolt's eval_initial_ram_mle)
            if (bytecode_words) |words| {
                if (words.len > 0) {
                    // bytecode_start = remap_address(min_bytecode_address)
                    // remap_address = (address - lowest_address) / 8
                    const bytecode_start: usize = @intCast((min_bytecode_address - lowest_address) / 8);

                    // Sum: bytecode_words[k] * eq(r_address, bytecode_start + k)
                    for (words, 0..) |word, k| {
                        const idx = bytecode_start + k;
                        if (idx >= max_idx) break;

                        const eq_val = computeEqAtPointBigEndian(r_address_be, idx);
                        const val = F.fromU64(word);
                        result = result.add(eq_val.mul(val));
                    }
                }
            }

            // Also add inputs region (like Jolt does)
            if (program_inputs) |inputs| {
                if (inputs.len > 0) {
                    // input_start = remap_address(memory_layout.input_start)
                    const input_start: usize = @intCast((memory_layout.input_start - lowest_address) / 8);

                    // Pack inputs into u64 words (little-endian)
                    var idx = input_start;
                    var off: usize = 0;
                    while (off < inputs.len) {
                        if (idx >= max_idx) break;

                        var word: u64 = 0;
                        const chunk_end = @min(off + 8, inputs.len);
                        for (off..chunk_end) |i| {
                            const byte_pos = i - off;
                            word |= @as(u64, inputs[i]) << @intCast(byte_pos * 8);
                        }

                        const eq_val = computeEqAtPointBigEndian(r_address_be, idx);
                        const val = F.fromU64(word);
                        result = result.add(eq_val.mul(val));

                        idx += 1;
                        off = chunk_end;
                    }
                }
            }

            return result;
        }

        /// Evaluate cubic polynomial at a challenge point from Toom-Cook evaluations
        /// Delegates to the shared UniPoly implementation.
        fn evaluateCubicAtChallengeFromEvals(evals: [4]F, x: F) F {
            return poly_mod.UniPoly(F).evaluateToomCookAt(evals, x);
        }

        /// Evaluate quadratic polynomial at a challenge point using Lagrange interpolation
        /// Input: evals = [p(0), p(1), p(2), _] where only the first 3 are used
        ///
        /// For ValFinal (degree-2 polynomial), we use Lagrange interpolation through 3 points
        /// [p(0), p(1), p(2)] at x = 0, 1, 2. This matches Jolt's from_evals_and_hint which
        /// uses Vandermonde interpolation through 3 points.
        ///
        /// Lagrange basis polynomials for points 0, 1, 2:
        /// L_0(x) = (x-1)(x-2) / ((0-1)(0-2)) = (x-1)(x-2) / 2
        /// L_1(x) = (x-0)(x-2) / ((1-0)(1-2)) = x(x-2) / (-1) = x(2-x)
        /// L_2(x) = (x-0)(x-1) / ((2-0)(2-1)) = x(x-1) / 2
        ///
        /// p(x) = p(0)*L_0(x) + p(1)*L_1(x) + p(2)*L_2(x)
        fn evaluateQuadraticAtChallengeFromEvals(evals: [4]F, x: F) F {
            const p0 = evals[0];
            const p1 = evals[1];
            const p2 = evals[2];

            const one = F.one();
            const two = F.fromU64(2);
            const inv2 = two.inverse().?;

            // L_0(x) = (x-1)(x-2) / 2
            const x_minus_1 = x.sub(one);
            const x_minus_2 = x.sub(two);
            const L_0 = x_minus_1.mul(x_minus_2).mul(inv2);

            // L_1(x) = x(2-x) = x(2-x)
            const two_minus_x = two.sub(x);
            const L_1 = x.mul(two_minus_x);

            // L_2(x) = x(x-1) / 2
            const L_2 = x.mul(x_minus_1).mul(inv2);

            // p(x) = p(0)*L_0(x) + p(1)*L_1(x) + p(2)*L_2(x)
            return p0.mul(L_0).add(p1.mul(L_1)).add(p2.mul(L_2));
        }

        /// Create a UniSkipFirstRoundProof for Stage 2 with actual base claims and extended evaluations
        ///
        /// This constructs the polynomial s1(Y) = L(tau_high, Y) * t1(Y) where:
        /// - L is the Lagrange kernel over the 5-point domain {-2, -1, 0, 1, 2}
        /// - t1 is interpolated from base_evals (at base domain) and extended_evals
        ///
        /// For product virtualization, the base_evals are the 3 product claims from Stage 1:
        /// [Product, ShouldBranch, ShouldJump]
        ///
        /// The extended_evals are the fused products at extended points {-2, 2}.
        ///
        /// The polynomial satisfies: Σ_t s1(t) = Σ_i L_i(tau_high) * base_evals[i] = input_claim
        fn createUniSkipProofStage2WithClaims(
            self: *Self,
            base_evals: *const [3]F,
            tau_high: F,
            raw_inputs: []const r1cs_evaluators.RawR1CSInputs,
            tau_stage2: []const F,
        ) !?UniSkipFirstRoundProof(F) {
            const univariate_skip = r1cs.univariate_skip;

            const DOMAIN_SIZE = univariate_skip.PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DOMAIN_SIZE; // 5
            const DEGREE = univariate_skip.PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DEGREE; // 4
            const EXTENDED_SIZE = univariate_skip.PRODUCT_VIRTUAL_UNIVARIATE_SKIP_EXTENDED_DOMAIN_SIZE; // 9
            const NUM_COEFFS = univariate_skip.PRODUCT_VIRTUAL_FIRST_ROUND_POLY_NUM_COEFFS; // 13

            // Compute extended evaluations from cycle witnesses using the 5 product constraints
            // Extended points {-3, 3, -4, 4} require the fused products computed from witness data
            const extended_evals: [DEGREE]F = blk: {
                if (raw_inputs.len == 0) {
                    break :blk [_]F{F.zero()} ** DEGREE;
                }

                // Compute extended evaluations using GruenSplitEq + RawR1CSInputs integer arithmetic
                break :blk try univariate_skip.computeProductVirtualExtendedEvals(
                    F,
                    raw_inputs,
                    tau_stage2,
                    self.allocator,
                    self.thread_pool,
                );
            };

            // Use the existing buildUniskipFirstRoundPoly function
            const uni_poly = try univariate_skip.buildUniskipFirstRoundPoly(
                F,
                DOMAIN_SIZE,
                DEGREE,
                EXTENDED_SIZE,
                NUM_COEFFS,
                base_evals,
                &extended_evals,
                tau_high,
                self.allocator,
            );

            // Verify the polynomial satisfies the sum constraint
            // input_claim = Σ L_i(tau_high) * base_evals[i]
            const LagrangePoly = univariate_skip.LagrangePolynomial(F);
            const lagrange_evals = try LagrangePoly.evals(DOMAIN_SIZE, tau_high, self.allocator);
            defer self.allocator.free(lagrange_evals);

            var input_claim = F.zero();
            for (base_evals, 0..) |eval, i| {
                input_claim = input_claim.add(lagrange_evals[i].mul(eval));
            }

            // Check domain sum
            const power_sums = univariate_skip.computePowerSums(DOMAIN_SIZE, NUM_COEFFS);
            var domain_sum = F.zero();
            for (uni_poly.coeffs, 0..) |coeff, j| {
                domain_sum = domain_sum.add(coeff.mulI128(power_sums[j]));
            }

            // Return as UniSkipFirstRoundProof
            return UniSkipFirstRoundProof(F){
                .uni_poly = uni_poly.coeffs,
                .allocator = self.allocator,
            };
        }
    };
}

/// Re-exported from stage2_sumcheck.zig
const extractProductFactors = @import("spartan/stage2_sumcheck.zig").extractProductFactors;

/// Configuration for proof conversion
///
/// These values must match Jolt's config.rs:
/// - log_k_chunk: Must be <= 8 (Jolt uses 4 for small traces, 8 for large)
/// - lookups_ra_virtual_log_k_chunk: Jolt uses LOG_K/8 (=16) for small traces
const tracer = @import("../tracer/mod.zig");

pub const JoltProverConfig = struct {
    /// Enable [BENCH] output lines for fine-grained stage timing
    /// Set via ZOLT_BENCH=1 environment variable
    bench_output: bool = false,
    /// Bytecode address space size (K) - must match Jolt's BytecodePreprocessing.code_size
    /// Use computeBytecodeCodeSize() in mod.zig to compute from raw program bytes
    bytecode_K: usize = 2048,
    /// Log of chunk size for one-hot encoding (must be <= 8, Jolt uses 4 for small traces)
    log_k_chunk: usize = 4,
    /// Log of chunk size for lookups RA virtualization (LOG_K / 8 = 128 / 8 = 16 for small traces)
    lookups_ra_virtual_log_k_chunk: usize = 16,
    /// Memory layout for computing I/O polynomial evaluations
    /// If null, OutputSumcheck will use zero claims (which will fail verification)
    memory_layout: ?*const jolt_device.MemoryLayout = null,
    /// Initial RAM state (before execution)
    initial_ram: ?*const std.AutoHashMapUnmanaged(u64, u64) = null,
    /// Final RAM state (after execution)
    final_ram: ?*const std.AutoHashMapUnmanaged(u64, u64) = null,
    /// Memory trace for RAF evaluation sumcheck
    /// If null, uses zero-polynomial approach (may fail for non-zero claims)
    memory_trace: ?*const ram.MemoryTrace = null,
    /// Program input bytes (for OutputSumcheck's ProgramIOPolynomial)
    program_inputs: ?[]const u8 = null,
    /// Program output bytes (for OutputSumcheck's ProgramIOPolynomial)
    program_outputs: ?[]const u8 = null,
    /// Whether the program panicked (for OutputSumcheck's ProgramIOPolynomial)
    is_panicking: bool = false,
    /// Execution trace for Stage 4 RegistersReadWriteChecking
    /// If null, Stage 4 uses zero-polynomial approach (which fails verification)
    execution_trace: ?*const tracer.ExecutionTrace = null,
    /// Bytecode words for initial RAM MLE evaluation (packed into 8-byte words)
    /// This is required for Stage 4 to compute the correct init_eval for RamValEvaluation
    bytecode_words: ?[]const u64 = null,
    /// Minimum bytecode address (word-aligned: actual_min_address / 8 * 8)
    /// Used to compute the remapped bytecode_start index
    min_bytecode_address: u64 = 0,
    /// BytecodePCMapper for converting ELF addresses to bytecode array indices
    /// Required for Stage 6 BytecodeReadRaf to correctly map cycle PCs to bytecode rows
    bytecode_pc_map: ?*const preprocessing.BytecodePCMapper = null,
    /// Raw ELF code bytes (text section) for populating static bytecode entries in Stage 6
    /// Required so buildBytecodeEntries can fill entries for ALL instructions, not just executed ones
    program_code_bytes: ?[]const u8 = null,
    /// Base address of the code section (typically 0x80000000)
    code_base_address: u64 = 0x80000000,
    /// Size of the .text section in bytes (instructions only, excluding .rodata)
    text_size: usize = 0,
    /// Preprocessing bytecode (source of truth for verifier; used to build val_polys)
    bytecode_preprocessing: ?*const @import("preprocessing.zig").BytecodePreprocessing = null,
    /// ELF entry point address (e_entry). Used for Stage 6 entry-point constraint.
    entry_address: u64 = 0,
    /// Pre-built compact integer witnesses (built by buildFromTrace during witness gen).
    prebuilt_compact: []const r1cs_evaluators.CompactWitness,
    /// Pre-built raw R1CS integer inputs for typed-accumulator claims.
    prebuilt_raw: []const r1cs_evaluators.RawR1CSInputs,
};

// =============================================================================
// Tests
// =============================================================================

const testing = std.testing;
const BN254Scalar = field_mod.BN254Scalar;

test "proof converter: basic initialization" {
    const converter = JoltProver(BN254Scalar).init(testing.allocator);
    _ = converter;
}

test "proof converter: proveWithTranscript uses Blake2b transcript" {
    const F = BN254Scalar;
    var converter = JoltProver(F).init(testing.allocator);

    // Create trivial compact/raw witnesses (4 noop cycles matching trace_length)
    const compact_witnesses = [_]r1cs_evaluators.CompactWitness{
        r1cs_evaluators.CompactWitness.noop(),
        r1cs_evaluators.CompactWitness.noop(),
        r1cs_evaluators.CompactWitness.noop(),
        r1cs_evaluators.CompactWitness.noop(),
    };
    const raw_witnesses = [_]r1cs_evaluators.RawR1CSInputs{
        r1cs_evaluators.RawR1CSInputs.noop(),
        r1cs_evaluators.RawR1CSInputs.noop(),
        r1cs_evaluators.RawR1CSInputs.noop(),
        r1cs_evaluators.RawR1CSInputs.noop(),
    };

    // Create tau challenge vector
    const tau = [_]F{ F.fromU64(1), F.fromU64(2), F.fromU64(3) };

    // Initialize transcript (matching Jolt's label)
    var transcript = Blake2bTranscript(F).init("jolt_v1");

    // Dummy types
    const DummyCommitment = struct { value: u64 };
    const DummyProof = struct { data: [32]u8 };

    // Convert with transcript
    var jolt_proof = try converter.proveWithTranscript(
        DummyCommitment,
        DummyProof,
        2, // log_t: trace_length = 4
        8, // log_k: ram_K = 256
        &[_]DummyCommitment{},
        null,
        .{ .prebuilt_compact = &compact_witnesses, .prebuilt_raw = &raw_witnesses },
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
    var converter1 = JoltProver(F).init(testing.allocator);
    var converter2 = JoltProver(F).init(testing.allocator);

    // Create compact/raw witnesses (4 noop cycles for trace_length = 4)
    const compact_witnesses = [_]r1cs_evaluators.CompactWitness{
        r1cs_evaluators.CompactWitness.noop(),
        r1cs_evaluators.CompactWitness.noop(),
        r1cs_evaluators.CompactWitness.noop(),
        r1cs_evaluators.CompactWitness.noop(),
    };
    const raw_witnesses = [_]r1cs_evaluators.RawR1CSInputs{
        r1cs_evaluators.RawR1CSInputs.noop(),
        r1cs_evaluators.RawR1CSInputs.noop(),
        r1cs_evaluators.RawR1CSInputs.noop(),
        r1cs_evaluators.RawR1CSInputs.noop(),
    };

    const tau = [_]F{ F.fromU64(1), F.fromU64(2) };

    var transcript1 = Blake2bTranscript(F).init("jolt_test");
    var transcript2 = Blake2bTranscript(F).init("jolt_test");

    const DummyCommitment = struct { value: u64 };
    const DummyProof = struct { data: [32]u8 };

    var jolt_proof1 = try converter1.proveWithTranscript(
        DummyCommitment,
        DummyProof,
        2, // log_t
        8, // log_k
        &[_]DummyCommitment{},
        null,
        .{ .prebuilt_compact = &compact_witnesses, .prebuilt_raw = &raw_witnesses },
        &tau,
        &transcript1,
    );
    defer jolt_proof1.deinit();

    var jolt_proof2 = try converter2.proveWithTranscript(
        DummyCommitment,
        DummyProof,
        2, // log_t
        8, // log_k
        &[_]DummyCommitment{},
        null,
        .{ .prebuilt_compact = &compact_witnesses, .prebuilt_raw = &raw_witnesses },
        &tau,
        &transcript2,
    );
    defer jolt_proof2.deinit();

    // Same inputs should produce same transcript state
    try testing.expectEqualSlices(u8, &transcript1.state, &transcript2.state);
    try testing.expectEqual(transcript1.n_rounds, transcript2.n_rounds);
}
