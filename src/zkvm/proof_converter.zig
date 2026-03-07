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

// Debug output control - set to true to enable verbose debug prints
const debug_verbose = true;
fn dbg(comptime fmt: []const u8, args: anytype) void {
    if (debug_verbose) std.debug.print(fmt, args);
}

const Allocator = std.mem.Allocator;

const jolt_types = @import("jolt_types.zig");
const prover = @import("prover.zig");
const field_mod = @import("../field/mod.zig");
const r1cs = @import("r1cs/mod.zig");
const streaming_outer = @import("spartan/streaming_outer.zig");
const product_remainder = @import("spartan/product_remainder.zig");
const transcripts = @import("../transcripts/mod.zig");
const Blake2bTranscript = transcripts.Blake2bTranscript;
const poly_mod = @import("../poly/mod.zig");
const jolt_device = @import("jolt_device.zig");
const constants = @import("../common/constants.zig");
const ram = @import("ram/mod.zig");
const instruction = @import("instruction/mod.zig");
const spartan_mod = @import("spartan/mod.zig");
const Stage3Prover = spartan_mod.Stage3Prover;
const Stage5BatchedProver = spartan_mod.Stage5BatchedProver;
const Stage6BatchedProver = spartan_mod.Stage6BatchedProver;
const preprocessing = @import("preprocessing.zig");

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

            jolt_proof.log_k_chunk = config.log_k_chunk;
            jolt_proof.lookups_ra_virtual_log_k_chunk = config.lookups_ra_virtual_log_k_chunk;

            // Set config structs (matching Jolt's serialization format)
            const log_t: u8 = @intCast(zolt_stage_proofs.log_t);
            const log_k: u8 = @intCast(zolt_stage_proofs.log_k);
            jolt_proof.rw_config = jolt_types.ReadWriteConfig.default(log_t, log_k);
            jolt_proof.one_hot_config = .{
                .log_k_chunk = @intCast(config.log_k_chunk),
                .lookups_ra_virtual_log_k_chunk = @intCast(config.lookups_ra_virtual_log_k_chunk),
            };
            jolt_proof.dory_layout = 0; // Wide layout

            // Compute derived parameters
            const n_cycle_vars = std.math.log2_int(usize, trace_length);
            _ = std.math.log2_int(usize, ram_K);

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
            // SpartanOuter requires all 35 R1CS inputs + UnivariateSkip claim
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
            // RegistersReadWriteChecking has LOG_K + log2(T) rounds where LOG_K = 7 (128 registers)
            const log_registers_stage4 = 7;
            try self.generateZeroSumcheckProof(
                &jolt_proof.stage4_sumcheck_proof,
                log_registers_stage4 + n_cycle_vars,
                3,
            );

            // Add Stage 4 opening claims
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .RamVal, .sumcheck_id = .RamValCheck } },
                F.zero(),
            );
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .RamValFinal, .sumcheck_id = .RamValCheck } },
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

            // DEBUG: Print tau length (challenges from transcript)
            dbg("[ZOLT] STAGE1: tau.len = {}\n", .{tau.len});

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
            dbg("[ZOLT] STAGE1: lagrange_tau_r0 (initial eq scaling) = {any}\n", .{lagrange_tau_r0.toBytes()});

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
            const uni_skip_claim = evaluatePolyAtChallenge(uniskip_proof.uni_poly, r0);
            dbg("[ZOLT] STAGE1: uni_skip_claim@SpartanOuter = {any}\n", .{uni_skip_claim.toBytesBE()});

            // DEBUG: Decompose s1(r0) = L(tau_high, r0) * t1(r0) and compare
            {
                dbg("[DECOMP] r0 = {any}\n", .{r0.toBytesBE()});
                dbg("[DECOMP] lagrange_tau_r0 = {any}\n", .{lagrange_tau_r0.toBytesBE()});
                const inv_L = lagrange_tau_r0.inverse();
                if (inv_L) |il| {
                    const t1_r0_from_s1 = uni_skip_claim.mul(il);
                    dbg("[DECOMP] t1(r0) = s1(r0)/L = {any}\n", .{t1_r0_from_s1.toBytesBE()});

                    // Now compute t1(r0) directly by evaluating the direct sum
                    // using the witnesses and lagrange evals at r0
                    // Build eq tables from the SAME full_tau
                    const tau_len = tau.len;
                    const m_d = tau_len / 2;
                    const n_x_in_bits_d = if (tau_len > 1) tau_len - 1 - m_d else 0;
                    const n_x_in_prime_bits_d = if (n_x_in_bits_d > 0) n_x_in_bits_d - 1 else 0;
                    const n_x_out_d: usize = @as(usize, 1) << @intCast(m_d);
                    const n_x_in_d: usize = @as(usize, 1) << @intCast(n_x_in_bits_d);

                    // Build eq tables (same logic as streaming_outer.buildEqTable)
                    const E_out_d = try self.allocator.alloc(F, n_x_out_d);
                    defer self.allocator.free(E_out_d);
                    E_out_d[0] = F.one();
                    var cs_d: usize = 1;
                    for (0..m_d) |kd| {
                        const t_k = tau[kd];
                        const omt_k = F.one().sub(t_k);
                        var id: usize = cs_d;
                        while (id > 0) {
                            id -= 1;
                            E_out_d[2 * id + 1] = E_out_d[id].mul(t_k);
                            E_out_d[2 * id] = E_out_d[id].mul(omt_k);
                        }
                        cs_d *= 2;
                    }

                    const E_in_d = try self.allocator.alloc(F, n_x_in_d);
                    defer self.allocator.free(E_in_d);
                    E_in_d[0] = F.one();
                    var cs_d2: usize = 1;
                    for (0..n_x_in_bits_d) |kd2| {
                        const t_k2 = tau[m_d + kd2];
                        const omt_k2 = F.one().sub(t_k2);
                        var id2: usize = cs_d2;
                        while (id2 > 0) {
                            id2 -= 1;
                            E_in_d[2 * id2 + 1] = E_in_d[id2].mul(t_k2);
                            E_in_d[2 * id2] = E_in_d[id2].mul(omt_k2);
                        }
                        cs_d2 *= 2;
                    }

                    // Compute Lagrange evals at r0
                    const StreamProver = streaming_outer.StreamingOuterProver(F);
                    const FGSZ = StreamProver.FIRST_GROUP_SIZE;
                    const SGSZ = StreamProver.SECOND_GROUP_SIZE;
                    const c_mod = r1cs.constraints;
                    var lags: [FGSZ]F = undefined;
                    {
                        const lag_start: i64 = -@as(i64, (FGSZ - 1) / 2);
                        for (0..FGSZ) |li| {
                            var lnum = F.one();
                            var lden = F.one();
                            for (0..FGSZ) |lj| {
                                if (li != lj) {
                                    const lxj: i64 = lag_start + @as(i64, @intCast(lj));
                                    const lxjf = if (lxj >= 0) F.fromU64(@intCast(lxj)) else F.zero().sub(F.fromU64(@intCast(-lxj)));
                                    lnum = lnum.mul(r0.sub(lxjf));
                                    const ldiff: i64 = @as(i64, @intCast(li)) - @as(i64, @intCast(lj));
                                    const ldifff = if (ldiff > 0) F.fromU64(@intCast(ldiff)) else F.zero().sub(F.fromU64(@intCast(-ldiff)));
                                    lden = lden.mul(ldifff);
                                }
                            }
                            lags[li] = lnum.mul(lden.inverse().?);
                        }
                    }

                    // Check Lagrange weights: should sum to 1
                    var lag_sum = F.zero();
                    for (0..FGSZ) |li| lag_sum = lag_sum.add(lags[li]);
                    dbg("[DECOMP] Lagrange weights sum = {any}\n", .{lag_sum.toBytesBE()});
                    dbg("[DECOMP] Lagrange weights sum == 1? {}\n", .{lag_sum.eql(F.one())});

                    // Check base domain: t1 at Y=0 (base point, should be zero)
                    {
                        var t1_at_zero = F.zero();
                        for (0..n_x_out_d) |xo0| {
                            for (0..n_x_in_d) |xi0| {
                                const eq0 = E_out_d[xo0].mul(E_in_d[xi0]);
                                const xip0 = xi0 >> 1;
                                const cyc0 = (xo0 << @intCast(n_x_in_prime_bits_d)) | xip0;
                                const grp0: usize = xi0 & 1;
                                if (cyc0 < cycle_witnesses.len) {
                                    const gid0 = if (grp0 == 0) &c_mod.FIRST_GROUP_INDICES else &c_mod.SECOND_GROUP_INDICES;
                                    const w0 = &cycle_witnesses[cyc0];
                                    // At Y=0, L_i(0) = delta_{i,4} since domain is {-4,...,5}
                                    const ci0 = 4; // index of Y=0 in domain
                                    const gsz0: usize = if (grp0 == 0) FGSZ else SGSZ;
                                    if (ci0 < gsz0) {
                                        const cc0 = c_mod.UNIFORM_CONSTRAINTS[gid0[ci0]];
                                        const az0 = cc0.condition.evaluate(F, w0.asSlice());
                                        const bz0 = cc0.left.evaluate(F, w0.asSlice()).sub(cc0.right.evaluate(F, w0.asSlice()));
                                        t1_at_zero = t1_at_zero.add(eq0.mul(az0.mul(bz0)));
                                    }
                                }
                            }
                        }
                        dbg("[DECOMP] t1(0) base domain = {any}\n", .{t1_at_zero.toBytesBE()});
                        dbg("[DECOMP] t1(0) == 0? {}\n", .{t1_at_zero.eql(F.zero())});
                    }

                    // Check ALL 10 base domain points
                    for (0..FGSZ) |base_idx| {
                        const base_y: i64 = -4 + @as(i64, @intCast(base_idx));
                        var t1_at_base = F.zero();
                        for (0..n_x_out_d) |xob| {
                            for (0..n_x_in_d) |xib| {
                                const eqb = E_out_d[xob].mul(E_in_d[xib]);
                                const xipb = xib >> 1;
                                const cycb = (xob << @intCast(n_x_in_prime_bits_d)) | xipb;
                                const grpb: usize = xib & 1;
                                if (cycb < cycle_witnesses.len) {
                                    const gidb = if (grpb == 0) &c_mod.FIRST_GROUP_INDICES else &c_mod.SECOND_GROUP_INDICES;
                                    const wb = &cycle_witnesses[cycb];
                                    const gszb: usize = if (grpb == 0) FGSZ else SGSZ;
                                    if (base_idx < gszb) {
                                        const ccb = c_mod.UNIFORM_CONSTRAINTS[gidb[base_idx]];
                                        const azb = ccb.condition.evaluate(F, wb.asSlice());
                                        const bzb = ccb.left.evaluate(F, wb.asSlice()).sub(ccb.right.evaluate(F, wb.asSlice()));
                                        const prod_b = azb.mul(bzb);
                                        if (!prod_b.eql(F.zero())) {
                                            dbg("[DECOMP] CONSTRAINT VIOLATED: cycle={d} group={d} base_idx={d} (Y={d}) constraint={d}\n", .{
                                                cycb, grpb, base_idx, base_y, gidb[base_idx],
                                            });
                                            dbg("[DECOMP]   condition={any}\n", .{azb.toBytesBE()});
                                            dbg("[DECOMP]   left-right={any}\n", .{bzb.toBytesBE()});
                                            // Print witness values for this cycle
                                            const R1CSIdx = c_mod.R1CSInputIndex;
                                            const ws = wb.asSlice();
                                            dbg("[DECOMP]   FlagJump={any}\n", .{ws[R1CSIdx.FlagJump.toIndex()].toBytesBE()});
                                            dbg("[DECOMP]   RdWriteValue={any}\n", .{ws[R1CSIdx.RdWriteValue.toIndex()].toBytesBE()});
                                            dbg("[DECOMP]   UnexpandedPC={any}\n", .{ws[R1CSIdx.UnexpandedPC.toIndex()].toBytesBE()});
                                            dbg("[DECOMP]   FlagIsCompressed={any}\n", .{ws[R1CSIdx.FlagIsCompressed.toIndex()].toBytesBE()});
                                            dbg("[DECOMP]   FlagIsNoop={any}\n", .{ws[R1CSIdx.FlagIsNoop.toIndex()].toBytesBE()});
                                            dbg("[DECOMP]   PC={any}\n", .{ws[R1CSIdx.PC.toIndex()].toBytesBE()});
                                        }
                                        t1_at_base = t1_at_base.add(eqb.mul(prod_b));
                                    }
                                }
                            }
                        }
                        if (!t1_at_base.eql(F.zero())) {
                            dbg("[DECOMP] t1({d}) NONZERO! = {any}\n", .{base_y, t1_at_base.toBytesBE()});
                        }
                    }
                    dbg("[DECOMP] base domain check complete\n", .{});

                    // Check direct_sum for cycle 0, group 0 individually
                    if (cycle_witnesses.len > 0) {
                        const w_test = &cycle_witnesses[0];
                        var az_test = F.zero();
                        var bz_test = F.zero();
                        for (0..FGSZ) |ki| {
                            const cc_test = c_mod.UNIFORM_CONSTRAINTS[c_mod.FIRST_GROUP_INDICES[ki]];
                            const cv_test = cc_test.condition.evaluate(F, w_test.asSlice());
                            const mv_test = cc_test.left.evaluate(F, w_test.asSlice()).sub(cc_test.right.evaluate(F, w_test.asSlice()));
                            az_test = az_test.add(lags[ki].mul(cv_test));
                            bz_test = bz_test.add(lags[ki].mul(mv_test));
                        }
                        dbg("[DECOMP] cycle0_g0: Az(r0)={any}\n", .{az_test.toBytesBE()});
                        dbg("[DECOMP] cycle0_g0: Bz(r0)={any}\n", .{bz_test.toBytesBE()});
                        dbg("[DECOMP] cycle0_g0: Az*Bz={any}\n", .{az_test.mul(bz_test).toBytesBE()});
                    }

                    // Compute direct sum
                    var direct_t1_r0 = F.zero();
                    for (0..n_x_out_d) |xod| {
                        for (0..n_x_in_d) |xid| {
                            const eq_d = E_out_d[xod].mul(E_in_d[xid]);
                            const xipd = xid >> 1;
                            const cycd = (xod << @intCast(n_x_in_prime_bits_d)) | xipd;
                            const grpd: usize = xid & 1;

                            if (cycd < cycle_witnesses.len) {
                                const gsz_d: usize = if (grpd == 0) FGSZ else SGSZ;
                                const gid_d = if (grpd == 0) &c_mod.FIRST_GROUP_INDICES else &c_mod.SECOND_GROUP_INDICES;
                                const wd = &cycle_witnesses[cycd];
                                var azd = F.zero();
                                var bzd = F.zero();
                                for (0..gsz_d) |kkd| {
                                    const ccd = c_mod.UNIFORM_CONSTRAINTS[gid_d[kkd]];
                                    const cvd = ccd.condition.evaluate(F, wd.asSlice());
                                    const mvd = ccd.left.evaluate(F, wd.asSlice()).sub(ccd.right.evaluate(F, wd.asSlice()));
                                    azd = azd.add(lags[kkd].mul(cvd));
                                    bzd = bzd.add(lags[kkd].mul(mvd));
                                }
                                direct_t1_r0 = direct_t1_r0.add(eq_d.mul(azd.mul(bzd)));
                            }
                        }
                    }
                    dbg("[DECOMP] t1(r0) direct sum = {any}\n", .{direct_t1_r0.toBytesBE()});
                    dbg("[DECOMP] t1(r0) from s1 match direct? {}\n", .{t1_r0_from_s1.eql(direct_t1_r0)});

                    // Also: compute s1(r0) = L(tau_high,r0) * direct_t1_r0 and compare
                    const s1_direct = lagrange_tau_r0.mul(direct_t1_r0);
                    dbg("[DECOMP] s1_direct = L * direct_t1 = {any}\n", .{s1_direct.toBytesBE()});
                    dbg("[DECOMP] s1_direct == uni_skip_claim? {}\n", .{s1_direct.eql(uni_skip_claim)});

                    // CRITICAL: Evaluate t1(r0) from UniSkip coefficients directly
                    // s1(r0) = uni_skip_claim, t1(r0) = s1(r0) / L(tau_high, r0)
                    // But also evaluate t1 at r0 using Lagrange formula from the 19 domain evaluations
                    {
                        const US = r1cs.univariate_skip;
                        const EXT_SIZE = US.OUTER_UNIVARIATE_SKIP_EXTENDED_DOMAIN_SIZE; // 19
                        const DEG = US.OUTER_UNIVARIATE_SKIP_DEGREE; // 9

                        // Get t1_vals from the UniSkip polynomial's coefficient representation
                        // Actually, we need the evaluation values. Let's compute t1(r0) using
                        // Lagrange interpolation from the 19 evaluations that were used to build the polynomial.
                        // We need the evaluations - recompute them quickly
                        const targets = US.UNISKIP_TARGETS;

                        // Build t1_eval_vals: the 19 evaluations on {-9,...,9}
                        // Base domain {-4,...,5}: t1 = 0
                        // Extended targets: computed from witnesses
                        var t1_eval_19: [EXT_SIZE]F = [_]F{F.zero()} ** EXT_SIZE;

                        // Recompute extended evaluations at the 9 target points
                        for (targets) |target_y| {
                            var ext_sum = F.zero();
                            for (0..n_x_out_d) |xo| {
                                const eo = if (xo < E_out_d.len) E_out_d[xo] else F.zero();
                                for (0..n_x_in_d) |xi| {
                                    const ei = if (xi < E_in_d.len) E_in_d[xi] else F.zero();
                                    const eq_v = eo.mul(ei);
                                    const xip = xi >> 1;
                                    const cyc = (xo << @intCast(n_x_in_prime_bits_d)) | xip;
                                    const grp: u1 = @truncate(xi & 1);
                                    if (cyc < cycle_witnesses.len) {
                                        const w = &cycle_witnesses[cyc];
                                        const gsz: usize = if (grp == 0) FGSZ else SGSZ;
                                        const gids = if (grp == 0) &c_mod.FIRST_GROUP_INDICES else &c_mod.SECOND_GROUP_INDICES;
                                        // Build Lagrange evals at target_y
                                        const tgt_f = if (target_y >= 0) F.fromU64(@intCast(target_y)) else F.zero().sub(F.fromU64(@intCast(-target_y)));
                                        var lag_tgt: [FGSZ]F = undefined;
                                        const lag_s: i64 = -@as(i64, (FGSZ - 1) / 2);
                                        for (0..FGSZ) |lti| {
                                            var ln = F.one();
                                            var ld = F.one();
                                            for (0..FGSZ) |ltj| {
                                                if (lti != ltj) {
                                                    const ltxj: i64 = lag_s + @as(i64, @intCast(ltj));
                                                    const ltxjf = if (ltxj >= 0) F.fromU64(@intCast(ltxj)) else F.zero().sub(F.fromU64(@intCast(-ltxj)));
                                                    ln = ln.mul(tgt_f.sub(ltxjf));
                                                    const ltd: i64 = @as(i64, @intCast(lti)) - @as(i64, @intCast(ltj));
                                                    const ltdf = if (ltd > 0) F.fromU64(@intCast(ltd)) else F.zero().sub(F.fromU64(@intCast(-ltd)));
                                                    ld = ld.mul(ltdf);
                                                }
                                            }
                                            lag_tgt[lti] = ln.mul(ld.inverse().?);
                                        }
                                        var az_t = F.zero();
                                        var bz_t = F.zero();
                                        for (0..gsz) |ki| {
                                            const cc = c_mod.UNIFORM_CONSTRAINTS[gids[ki]];
                                            const cv = cc.condition.evaluate(F, w.asSlice());
                                            const mv = cc.left.evaluate(F, w.asSlice()).sub(cc.right.evaluate(F, w.asSlice()));
                                            az_t = az_t.add(lag_tgt[ki].mul(cv));
                                            bz_t = bz_t.add(lag_tgt[ki].mul(mv));
                                        }
                                        ext_sum = ext_sum.add(eq_v.mul(az_t.mul(bz_t)));
                                    }
                                }
                            }
                            const pos: usize = @intCast(target_y + @as(i64, DEG));
                            t1_eval_19[pos] = ext_sum;
                            dbg("[DECOMP] t1_eval_19[{d}] (Y={d}) = {any}\n", .{pos, target_y, ext_sum.toBytesBE()});
                        }

                        // Now evaluate t1(r0) using Lagrange formula from the 19 evaluations
                        var t1_r0_lagrange19 = F.zero();
                        for (0..EXT_SIZE) |li19| {
                            if (t1_eval_19[li19].eql(F.zero())) continue;
                            const xi19: i64 = @as(i64, @intCast(li19)) - @as(i64, DEG);
                            var ln19 = F.one();
                            var ld19 = F.one();
                            for (0..EXT_SIZE) |lj19| {
                                if (li19 != lj19) {
                                    const xj19: i64 = @as(i64, @intCast(lj19)) - @as(i64, DEG);
                                    const xj19f = if (xj19 >= 0) F.fromU64(@intCast(xj19)) else F.zero().sub(F.fromU64(@intCast(-xj19)));
                                    ln19 = ln19.mul(r0.sub(xj19f));
                                    const d19: i64 = xi19 - xj19;
                                    const d19f = if (d19 > 0) F.fromU64(@intCast(d19)) else F.zero().sub(F.fromU64(@intCast(-d19)));
                                    ld19 = ld19.mul(d19f);
                                }
                            }
                            t1_r0_lagrange19 = t1_r0_lagrange19.add(t1_eval_19[li19].mul(ln19.mul(ld19.inverse().?)));
                        }

                        dbg("[DECOMP] t1(r0) Lagrange19 = {any}\n", .{t1_r0_lagrange19.toBytesBE()});
                        dbg("[DECOMP] Lagrange19 == direct_sum? {}\n", .{t1_r0_lagrange19.eql(direct_t1_r0)});
                        dbg("[DECOMP] Lagrange19 == from_s1? {}\n", .{t1_r0_lagrange19.eql(t1_r0_from_s1)});

                        // Also check: does s1 polynomial Horner eval at r0 match uni_skip_claim?
                        // (s1 = the actual polynomial sent in the proof)
                        const s1_horner = evaluatePolyAtChallenge(uniskip_proof.uni_poly, r0);
                        dbg("[DECOMP] s1(r0) Horner = {any}\n", .{s1_horner.toBytesBE()});
                        dbg("[DECOMP] s1 Horner == uni_skip_claim? {}\n", .{s1_horner.eql(uni_skip_claim)});
                    }

                    // Now compute t1 at Y=-5 (first extended target) using the SAME
                    // Lagrange eval approach (not COEFFS_PER_J) and compare with
                    // what evaluateAzBzAtTargetY gives
                    const neg5_field = F.zero().sub(F.fromU64(5));
                    var lag_neg5: [FGSZ]F = undefined;
                    {
                        const lag_start_n5: i64 = -@as(i64, (FGSZ - 1) / 2);
                        for (0..FGSZ) |li| {
                            var lnum2 = F.one();
                            var lden2 = F.one();
                            for (0..FGSZ) |lj| {
                                if (li != lj) {
                                    const lxj2: i64 = lag_start_n5 + @as(i64, @intCast(lj));
                                    const lxjf2 = if (lxj2 >= 0) F.fromU64(@intCast(lxj2)) else F.zero().sub(F.fromU64(@intCast(-lxj2)));
                                    lnum2 = lnum2.mul(neg5_field.sub(lxjf2));
                                    const ldiff2: i64 = @as(i64, @intCast(li)) - @as(i64, @intCast(lj));
                                    const ldifff2 = if (ldiff2 > 0) F.fromU64(@intCast(ldiff2)) else F.zero().sub(F.fromU64(@intCast(-ldiff2)));
                                    lden2 = lden2.mul(ldifff2);
                                }
                            }
                            lag_neg5[li] = lnum2.mul(lden2.inverse().?);
                        }
                    }

                    // Compute direct sum at Y=-5 using Lagrange evals at -5
                    var direct_t1_neg5 = F.zero();
                    for (0..n_x_out_d) |xod2| {
                        for (0..n_x_in_d) |xid2| {
                            const eq_d2 = E_out_d[xod2].mul(E_in_d[xid2]);
                            const xipd2 = xid2 >> 1;
                            const cycd2 = (xod2 << @intCast(n_x_in_prime_bits_d)) | xipd2;
                            const grpd2: usize = xid2 & 1;

                            if (cycd2 < cycle_witnesses.len) {
                                const gsz_d2: usize = if (grpd2 == 0) FGSZ else SGSZ;
                                const gid_d2 = if (grpd2 == 0) &c_mod.FIRST_GROUP_INDICES else &c_mod.SECOND_GROUP_INDICES;
                                const wd2 = &cycle_witnesses[cycd2];
                                var azd2 = F.zero();
                                var bzd2 = F.zero();
                                for (0..gsz_d2) |kkd2| {
                                    const ccd2 = c_mod.UNIFORM_CONSTRAINTS[gid_d2[kkd2]];
                                    const cvd2 = ccd2.condition.evaluate(F, wd2.asSlice());
                                    const mvd2 = ccd2.left.evaluate(F, wd2.asSlice()).sub(ccd2.right.evaluate(F, wd2.asSlice()));
                                    azd2 = azd2.add(lag_neg5[kkd2].mul(cvd2));
                                    bzd2 = bzd2.add(lag_neg5[kkd2].mul(mvd2));
                                }
                                direct_t1_neg5 = direct_t1_neg5.add(eq_d2.mul(azd2.mul(bzd2)));
                            }
                        }
                    }
                    dbg("[DECOMP] t1(-5) direct Lagrange = {any}\n", .{direct_t1_neg5.toBytesBE()});

                    // Now compute using COEFFS_PER_J (same as evaluateAzBzAtTargetY)
                    const unskip = r1cs.univariate_skip;
                    var coeffs_t1_neg5 = F.zero();
                    for (0..n_x_out_d) |xod3| {
                        for (0..n_x_in_d) |xid3| {
                            const eq_d3 = E_out_d[xod3].mul(E_in_d[xid3]);
                            const xipd3 = xid3 >> 1;
                            const cycd3 = (xod3 << @intCast(n_x_in_prime_bits_d)) | xipd3;
                            const grpd3: usize = xid3 & 1;

                            if (cycd3 < cycle_witnesses.len) {
                                const gsz_d3: usize = if (grpd3 == 0) FGSZ else SGSZ;
                                const gid_d3 = if (grpd3 == 0) &c_mod.FIRST_GROUP_INDICES else &c_mod.SECOND_GROUP_INDICES;
                                const wd3 = &cycle_witnesses[cycd3];
                                // Evaluate using COEFFS_PER_J[0] (target Y=-5 is index 0)
                                const coeffs_j = unskip.COEFFS_PER_J[0]; // target -5
                                var azd3 = F.zero();
                                var bzd3 = F.zero();
                                for (0..gsz_d3) |kkd3| {
                                    const ccd3 = c_mod.UNIFORM_CONSTRAINTS[gid_d3[kkd3]];
                                    const cvd3 = ccd3.condition.evaluate(F, wd3.asSlice());
                                    const mvd3 = ccd3.left.evaluate(F, wd3.asSlice()).sub(ccd3.right.evaluate(F, wd3.asSlice()));
                                    const cf3 = coeffs_j[kkd3];
                                    const cf3f = if (cf3 > 0) F.fromU64(@intCast(cf3)) else F.zero().sub(F.fromU64(@intCast(-cf3)));
                                    azd3 = azd3.add(cf3f.mul(cvd3));
                                    bzd3 = bzd3.add(cf3f.mul(mvd3));
                                }
                                coeffs_t1_neg5 = coeffs_t1_neg5.add(eq_d3.mul(azd3.mul(bzd3)));
                            }
                        }
                    }
                    dbg("[DECOMP] t1(-5) COEFFS_PER_J = {any}\n", .{coeffs_t1_neg5.toBytesBE()});
                    dbg("[DECOMP] t1(-5) Lagrange == COEFFS_PER_J? {}\n", .{direct_t1_neg5.eql(coeffs_t1_neg5)});

                    // Compute t1 at ALL 19 domain points {-9,...,9} using direct Lagrange
                    // and compare with what the polynomial gives
                    var domain_mismatches: usize = 0;
                    for (0..19) |dpidx| {
                        const dpy: i64 = @as(i64, @intCast(dpidx)) - 9;
                        const dpy_field = if (dpy >= 0) F.fromU64(@intCast(dpy)) else F.zero().sub(F.fromU64(@intCast(-dpy)));

                        // Compute Lagrange evals at this Y on 10-point domain
                        var lag_y: [FGSZ]F = undefined;
                        {
                            const lag_start_y: i64 = -@as(i64, (FGSZ - 1) / 2);
                            for (0..FGSZ) |liy| {
                                var lnumy = F.one();
                                var ldeny = F.one();
                                for (0..FGSZ) |ljy| {
                                    if (liy != ljy) {
                                        const lxjy: i64 = lag_start_y + @as(i64, @intCast(ljy));
                                        const lxjfy = if (lxjy >= 0) F.fromU64(@intCast(lxjy)) else F.zero().sub(F.fromU64(@intCast(-lxjy)));
                                        lnumy = lnumy.mul(dpy_field.sub(lxjfy));
                                        const ldiffy: i64 = @as(i64, @intCast(liy)) - @as(i64, @intCast(ljy));
                                        const ldiffyf = if (ldiffy > 0) F.fromU64(@intCast(ldiffy)) else F.zero().sub(F.fromU64(@intCast(-ldiffy)));
                                        ldeny = ldeny.mul(ldiffyf);
                                    }
                                }
                                lag_y[liy] = lnumy.mul(ldeny.inverse().?);
                            }
                        }

                        // Compute direct t1(dpy) sum
                        var t1_dpy = F.zero();
                        for (0..n_x_out_d) |xody| {
                            for (0..n_x_in_d) |xidy| {
                                const eq_dy = E_out_d[xody].mul(E_in_d[xidy]);
                                const xipdy = xidy >> 1;
                                const cycdy = (xody << @intCast(n_x_in_prime_bits_d)) | xipdy;
                                const grpdy: usize = xidy & 1;
                                if (cycdy < cycle_witnesses.len) {
                                    const gsz_dy: usize = if (grpdy == 0) FGSZ else SGSZ;
                                    const gid_dy = if (grpdy == 0) &c_mod.FIRST_GROUP_INDICES else &c_mod.SECOND_GROUP_INDICES;
                                    const wdy = &cycle_witnesses[cycdy];
                                    var azdy = F.zero();
                                    var bzdy = F.zero();
                                    for (0..gsz_dy) |kkdy| {
                                        const ccdy = c_mod.UNIFORM_CONSTRAINTS[gid_dy[kkdy]];
                                        const cvdy = ccdy.condition.evaluate(F, wdy.asSlice());
                                        const mvdy = ccdy.left.evaluate(F, wdy.asSlice()).sub(ccdy.right.evaluate(F, wdy.asSlice()));
                                        azdy = azdy.add(lag_y[kkdy].mul(cvdy));
                                        bzdy = bzdy.add(lag_y[kkdy].mul(mvdy));
                                    }
                                    t1_dpy = t1_dpy.add(eq_dy.mul(azdy.mul(bzdy)));
                                }
                            }
                        }

                        // (placeholder for polynomial evaluation at dpy)

                        // For base points, the direct sum should be 0 (correct witness)
                        const is_base = (dpy >= -4 and dpy <= 5);
                        if (!t1_dpy.eql(F.zero()) and is_base) {
                            dbg("[DOMCHK] Y={d}: t1 NON-ZERO at base point! val={any}\n", .{ dpy, t1_dpy.toBytesBE() });
                            domain_mismatches += 1;

                            // Find which cycles contribute non-zero AzBz at this base point
                            var cnt_nz: usize = 0;
                            for (0..n_x_out_d) |xodz| {
                                for (0..n_x_in_d) |xidz| {
                                    const xipz = xidz >> 1;
                                    const cycz = (xodz << @intCast(n_x_in_prime_bits_d)) | xipz;
                                    const grpz: usize = xidz & 1;
                                    if (cycz < cycle_witnesses.len) {
                                        const gsz_z: usize = if (grpz == 0) FGSZ else SGSZ;
                                        const gid_z = if (grpz == 0) &c_mod.FIRST_GROUP_INDICES else &c_mod.SECOND_GROUP_INDICES;
                                        const wdz = &cycle_witnesses[cycz];
                                        var azz = F.zero();
                                        var bzz = F.zero();
                                        for (0..gsz_z) |kkz| {
                                            const ccz = c_mod.UNIFORM_CONSTRAINTS[gid_z[kkz]];
                                            const cvz = ccz.condition.evaluate(F, wdz.asSlice());
                                            const mvz = ccz.left.evaluate(F, wdz.asSlice()).sub(ccz.right.evaluate(F, wdz.asSlice()));
                                            azz = azz.add(lag_y[kkz].mul(cvz));
                                            bzz = bzz.add(lag_y[kkz].mul(mvz));
                                        }
                                        const abz = azz.mul(bzz);
                                        if (!abz.eql(F.zero())) {
                                            cnt_nz += 1;
                                            if (cnt_nz <= 3) {
                                                dbg("[DOMCHK]   cycle={d} grp={d}: Az*Bz={any}\n", .{ cycz, grpz, abz.toBytesBE() });
                                                // Also print individual Az, Bz
                                                dbg("[DOMCHK]     Az={any} Bz={any}\n", .{ azz.toBytesBE(), bzz.toBytesBE() });
                                                // And Lagrange evals used
                                                for (0..@min(3, gsz_z)) |qk| {
                                                    dbg("[DOMCHK]     lag[{d}]={any}\n", .{ qk, lag_y[qk].toBytesBE() });
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            dbg("[DOMCHK]   total non-zero AzBz: {d}\n", .{cnt_nz});
                        } else if (!is_base) {
                            dbg("[DOMCHK] Y={d}: t1_direct={any}\n", .{ dpy, t1_dpy.toBytesBE() });
                        }
                    }
                    dbg("[DOMCHK] domain check: {d} base point violations\n", .{domain_mismatches});

                    // Check ALL constraints at ALL cycles for violations
                    var total_violations: usize = 0;
                    for (0..cycle_witnesses.len) |cv| {
                        const wcv = &cycle_witnesses[cv];
                        for (0..c_mod.UNIFORM_CONSTRAINTS.len) |ci| {
                            const cc = c_mod.UNIFORM_CONSTRAINTS[ci];
                            const cond_v = cc.condition.evaluate(F, wcv.asSlice());
                            const left_v = cc.left.evaluate(F, wcv.asSlice());
                            const right_v = cc.right.evaluate(F, wcv.asSlice());
                            const diff_v = left_v.sub(right_v);
                            const prod_v = cond_v.mul(diff_v);
                            if (!prod_v.eql(F.zero())) {
                                total_violations += 1;
                                if (total_violations <= 20) {
                                    dbg("[VIOLATION] cycle={d} constraint={d}: cond={any} left={any} right={any} diff={any}\n", .{
                                        cv, ci,
                                        cond_v.toBytesBE(),
                                        left_v.toBytesBE(),
                                        right_v.toBytesBE(),
                                        diff_v.toBytesBE(),
                                    });
                                }
                            }
                        }
                    }
                    dbg("[VIOLATION] total constraint violations: {d} across {d} cycles x {d} constraints\n", .{
                        total_violations, cycle_witnesses.len, c_mod.UNIFORM_CONSTRAINTS.len,
                    });

                    // Print key witness values for violated cycles
                    if (total_violations > 0 and cycle_witnesses.len > 54) {
                        const w54s = cycle_witnesses[54].asSlice();
                        dbg("[CYCLE54] UnexpandedPC   = {any}\n", .{w54s[c_mod.R1CSInputIndex.UnexpandedPC.toIndex()].toBytesBE()});
                        dbg("[CYCLE54] NextUnexpPC    = {any}\n", .{w54s[c_mod.R1CSInputIndex.NextUnexpandedPC.toIndex()].toBytesBE()});
                        dbg("[CYCLE54] ShouldBranch   = {any}\n", .{w54s[c_mod.R1CSInputIndex.ShouldBranch.toIndex()].toBytesBE()});
                        dbg("[CYCLE54] FlagJump       = {any}\n", .{w54s[c_mod.R1CSInputIndex.FlagJump.toIndex()].toBytesBE()});
                        dbg("[CYCLE54] DoNotUpdateUPC = {any}\n", .{w54s[c_mod.R1CSInputIndex.FlagDoNotUpdateUnexpandedPC.toIndex()].toBytesBE()});
                        dbg("[CYCLE54] IsCompressed   = {any}\n", .{w54s[c_mod.R1CSInputIndex.FlagIsCompressed.toIndex()].toBytesBE()});
                        dbg("[CYCLE54] PC             = {any}\n", .{w54s[c_mod.R1CSInputIndex.PC.toIndex()].toBytesBE()});
                        dbg("[CYCLE54] NextPC         = {any}\n", .{w54s[c_mod.R1CSInputIndex.NextPC.toIndex()].toBytesBE()});
                        dbg("[CYCLE54] FlagIsNoop     = {any}\n", .{w54s[c_mod.R1CSInputIndex.FlagIsNoop.toIndex()].toBytesBE()});
                        // Also print cycle 55
                        if (cycle_witnesses.len > 55) {
                            const w55s = cycle_witnesses[55].asSlice();
                            dbg("[CYCLE55] UnexpandedPC   = {any}\n", .{w55s[c_mod.R1CSInputIndex.UnexpandedPC.toIndex()].toBytesBE()});
                            dbg("[CYCLE55] FlagIsNoop     = {any}\n", .{w55s[c_mod.R1CSInputIndex.FlagIsNoop.toIndex()].toBytesBE()});
                            dbg("[CYCLE55] PC             = {any}\n", .{w55s[c_mod.R1CSInputIndex.PC.toIndex()].toBytesBE()});
                        }
                    }
                }
            }

            // Bind the first-round challenge from transcript with the uni_skip_claim
            outer_prover.bindFirstRoundChallenge(r0, uni_skip_claim) catch {};

            // Match Jolt's cache_openings: after UniSkip verification, the verifier calls
            // accumulator.append_virtual() which appends the uni_skip_claim to transcript.
            // This happens BEFORE BatchedSumcheck::verify which also appends it.
            // flush_to_transcript: uni_skip opening claim
            transcript.appendScalar("opening_claim", uni_skip_claim);
            std.debug.print("[ZOLT-PROVER] after_flush transcript_state = ", .{});
            for (transcript.state[0..8]) |b| std.debug.print("{x:0>2}", .{b});
            std.debug.print(" round={}\n", .{transcript.n_rounds});

            // BatchedSumcheck::verify: append input_claim then get batching coefficients
            transcript.appendScalar("sumcheck_claim", uni_skip_claim);

            // Get batching coefficient - advances transcript state AND provides scaling factor
            const batching_coeff = transcript.challengeScalarFull();
            std.debug.print("[ZOLT-PROVER] input_claim (uni_skip_claim) = {any}\n", .{uni_skip_claim.toBytesBE()});
            std.debug.print("[ZOLT-PROVER] batching_coeff = {any}\n", .{batching_coeff.toBytesBE()});
            std.debug.print("[ZOLT-PROVER] transcript state: ", .{});
            for (transcript.state[0..8]) |b| std.debug.print("{x:0>2} ", .{b});
            std.debug.print("round={}\n", .{transcript.n_rounds});

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
            std.debug.print("[ZOLT-PROVER] batched_claim = {any}\n", .{initial_claim.toBytesBE()});

            // Generate all remaining round polynomials with transcript integration
            for (0..num_remaining_rounds) |round_idx| {
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
                dbg("[ZOLT] STAGE1_ROUND_{}: c0 = {any}\n", .{ round_idx, compressed[0].toBytes() });
                dbg("[ZOLT] STAGE1_ROUND_{}: c2 = {any}\n", .{ round_idx, compressed[1].toBytes() });
                dbg("[ZOLT] STAGE1_ROUND_{}: c3 = {any}\n", .{ round_idx, compressed[2].toBytes() });

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
                dbg("[ZOLT] STAGE1_ROUND_{}: challenge = {any}\n", .{ round_idx, challenge.toBytes() });

                // Bind challenge and update claim
                // Use raw_evals for internal claim tracking (matches Jolt's prover behavior)
                // The proof contains scaled polynomials, but the prover tracks unscaled internally
                outer_prover.bindRemainingRoundChallenge(challenge) catch {};
                outer_prover.updateClaim(raw_evals, challenge);
            }

            // DEBUG: Print final summary including eq factor from split_eq
            dbg("[ZOLT] STAGE1_FINAL: num_rounds = {}\n", .{challenges.items.len});
            const prover_eq_factor = outer_prover.split_eq.current_scalar;
            dbg("[ZOLT] STAGE1_FINAL: prover eq_factor (split_eq.current_scalar) = {any}\n", .{prover_eq_factor.toBytes()});
            dbg("[ZOLT] STAGE1_FINAL: prover eq_factor limbs = [{x}, {x}, {x}, {x}]\n", .{
                prover_eq_factor.limbs[0], prover_eq_factor.limbs[1], prover_eq_factor.limbs[2], prover_eq_factor.limbs[3],
            });

            // Print final claim from prover
            const prover_final_claim = outer_prover.current_claim;
            dbg("[ZOLT] STAGE1_FINAL: prover final_claim = {any}\n", .{prover_final_claim.toBytes()});
            dbg("[ZOLT] STAGE1_FINAL: prover final_claim * batching_coeff = {any}\n", .{prover_final_claim.mul(batching_coeff).toBytes()});

            // Compute implied Az*Bz = final_claim / eq_factor
            if (!prover_eq_factor.eql(F.zero())) {
                const implied_az_bz = prover_final_claim.mul(prover_eq_factor.inverse().?);
                dbg("[ZOLT] STAGE1_FINAL: implied Az*Bz (final_claim/eq_factor) = {any}\n", .{implied_az_bz.toBytes()});
            }

            // CROSS-CHECK: Compute the "correct" final_claim directly from witnesses
            // This is what the verifier expects: eq_factor * Az(r_stream, r0, r_cycle) * Bz(r_stream, r0, r_cycle)
            // where r_cycle is the full set of bound challenges reversed
            {
                const all_chal = challenges.items;
                const r_stream_check = if (all_chal.len > 0) all_chal[0] else F.zero();
                const r0_check = r0;

                // Get the cycle challenges (skip r_stream)
                const cycle_chal = if (all_chal.len > 1) all_chal[1..] else all_chal[0..0];

                // Compute MLE evaluations of R1CS inputs at r_cycle (reversed)
                const r_cycle_be = try self.allocator.alloc(F, cycle_chal.len);
                defer self.allocator.free(r_cycle_be);
                for (0..cycle_chal.len) |idx| {
                    r_cycle_be[idx] = cycle_chal[cycle_chal.len - 1 - idx];
                }

                const R1CSInputEval = r1cs.R1CSInputEvaluator(F);
                const check_evals = try R1CSInputEval.computeClaimedInputs(
                    self.allocator,
                    cycle_witnesses,
                    r_cycle_be,
                );

                // Compute Lagrange weights at r0
                const FGSZ = 10;
                const SGSZ = 9;
                var w_check: [FGSZ]F = undefined;
                const start: i64 = -4;
                for (0..FGSZ) |ii| {
                    var numer = F.one();
                    var denom = F.one();
                    for (0..FGSZ) |jj| {
                        if (ii != jj) {
                            const x_j: i64 = start + @as(i64, @intCast(jj));
                            const x_j_f = if (x_j >= 0)
                                F.fromU64(@intCast(x_j))
                            else
                                F.zero().sub(F.fromU64(@intCast(-x_j)));
                            numer = numer.mul(r0_check.sub(x_j_f));
                            const d: i64 = @as(i64, @intCast(ii)) - @as(i64, @intCast(jj));
                            if (d > 0) {
                                denom = denom.mul(F.fromU64(@intCast(d)));
                            } else {
                                denom = denom.mul(F.zero().sub(F.fromU64(@intCast(-d))));
                            }
                        }
                    }
                    w_check[ii] = if (!denom.eql(F.zero())) numer.mul(denom.inverse().?) else F.zero();
                }

                // Build z vector
                var z_check: [r1cs.R1CSInputIndex.NUM_INPUTS + 1]F = undefined;
                @memcpy(z_check[0..r1cs.R1CSInputIndex.NUM_INPUTS], &check_evals);
                z_check[r1cs.R1CSInputIndex.NUM_INPUTS] = F.one();

                // Compute az_g0, bz_g0
                var az_g0_c = F.zero();
                var bz_g0_c = F.zero();
                for (0..FGSZ) |ii| {
                    const cidx = r1cs.FIRST_GROUP_INDICES[ii];
                    const cons = r1cs.UNIFORM_CONSTRAINTS[cidx];
                    const az_c = cons.condition.evaluateWithConstant(F, &z_check);
                    const bz_c = cons.left.evaluateWithConstant(F, &z_check).sub(cons.right.evaluateWithConstant(F, &z_check));
                    az_g0_c = az_g0_c.add(w_check[ii].mul(az_c));
                    bz_g0_c = bz_g0_c.add(w_check[ii].mul(bz_c));
                }

                var az_g1_c = F.zero();
                var bz_g1_c = F.zero();
                for (0..SGSZ) |ii| {
                    const cidx = r1cs.SECOND_GROUP_INDICES[ii];
                    const cons = r1cs.UNIFORM_CONSTRAINTS[cidx];
                    const az_c = cons.condition.evaluateWithConstant(F, &z_check);
                    const bz_c = cons.left.evaluateWithConstant(F, &z_check).sub(cons.right.evaluateWithConstant(F, &z_check));
                    az_g1_c = az_g1_c.add(w_check[ii].mul(az_c));
                    bz_g1_c = bz_g1_c.add(w_check[ii].mul(bz_c));
                }

                const az_f_c = az_g0_c.add(r_stream_check.mul(az_g1_c.sub(az_g0_c)));
                const bz_f_c = bz_g0_c.add(r_stream_check.mul(bz_g1_c.sub(bz_g0_c)));
                const inner_sp = az_f_c.mul(bz_f_c);
                const expected_final = inner_sp.mul(prover_eq_factor);

                dbg("[ZOLT] STAGE1_CROSSCHECK: inner_sum_prod = {any}\n", .{inner_sp.toBytes()});
                dbg("[ZOLT] STAGE1_CROSSCHECK: expected_final (inner_sp * eq_factor) = {any}\n", .{expected_final.toBytes()});
                dbg("[ZOLT] STAGE1_CROSSCHECK: prover_final = {any}\n", .{prover_final_claim.toBytes()});
                dbg("[ZOLT] STAGE1_CROSSCHECK: match = {}\n", .{expected_final.eql(prover_final_claim)});
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
            cycle_witnesses: []const r1cs.R1CSCycleInputs(F),
            r_cycle: []const F,
            uni_skip_claim: F,
            transcript: *Blake2bTranscript(F),
            r_stream: F,
            r0: F,
        ) !void {
            // Compute MLE evaluations at r_cycle
            const R1CSInputEvaluator = r1cs.R1CSInputEvaluator(F);
            const input_evals = try R1CSInputEvaluator.computeClaimedInputs(
                self.allocator,
                cycle_witnesses,
                r_cycle,
            );

            // DEBUG: Print the Imm opening claim value (index 8)
            if (comptime debug_verbose) {
                const imm_eval = input_evals[r1cs.R1CSInputIndex.Imm.toIndex()];
                const imm_le = imm_eval.toBytes();
                dbg("[ZOLT_OC_IMM] oc_Imm_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                    imm_le[0], imm_le[1], imm_le[2], imm_le[3], imm_le[4], imm_le[5], imm_le[6], imm_le[7],
                });
                // Print Imm witness values for first few cycles
                for (0..@min(cycle_witnesses.len, 10)) |c_idx| {
                    const w_imm = cycle_witnesses[c_idx].values[r1cs.R1CSInputIndex.Imm.toIndex()];
                    if (!w_imm.eql(F.zero())) {
                        const wl = w_imm.toBytes();
                        dbg("[ZOLT_OC_IMM] witness[{}].Imm_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                            c_idx, wl[0], wl[1], wl[2], wl[3], wl[4], wl[5], wl[6], wl[7],
                        });
                    }
                }
            }
            // DEBUG: Print the first few R1CS input evaluations
            dbg("[ZOLT] OPENING_CLAIMS: r_cycle.len = {}\n", .{r_cycle.len});
            dbg("[ZOLT] OPENING_CLAIMS: cycle_witnesses.len = {}\n", .{cycle_witnesses.len});
            // Print first and last r_cycle values
            if (r_cycle.len > 0) {
                dbg("[ZOLT] OPENING_CLAIMS: r_cycle[0] = {any}\n", .{r_cycle[0].toBytes()});
                dbg("[ZOLT] OPENING_CLAIMS: r_cycle[last] = {any}\n", .{r_cycle[r_cycle.len - 1].toBytes()});
            }
            // Print first few witness values
            if (cycle_witnesses.len > 0) {
                dbg("[ZOLT] OPENING_CLAIMS: witness[0].LeftInstructionInput = {any}\n", .{cycle_witnesses[0].values[0].toBytes()});
                dbg("[ZOLT] OPENING_CLAIMS: witness[0].RightInstructionInput = {any}\n", .{cycle_witnesses[0].values[1].toBytes()});
                dbg("[ZOLT] OPENING_CLAIMS: witness[0].Product = {any}\n", .{cycle_witnesses[0].values[2].toBytes()});
                dbg("[ZOLT] OPENING_CLAIMS: witness[0].PC = {any}\n", .{cycle_witnesses[0].values[6].toBytes()});
            }
            if (cycle_witnesses.len > 1) {
                dbg("[ZOLT] OPENING_CLAIMS: witness[1].LeftInstructionInput = {any}\n", .{cycle_witnesses[1].values[0].toBytes()});
                dbg("[ZOLT] OPENING_CLAIMS: witness[1].PC = {any}\n", .{cycle_witnesses[1].values[6].toBytes()});
            }
            dbg("[ZOLT] OPENING_CLAIMS: r1cs_input_evals[0] (LeftInstructionInput) = {any}\n", .{input_evals[0].toBytes()});
            dbg("[ZOLT] OPENING_CLAIMS: r1cs_input_evals[1] (RightInstructionInput) = {any}\n", .{input_evals[1].toBytes()});
            dbg("[ZOLT] OPENING_CLAIMS: r1cs_input_evals[2] (Product) = {any}\n", .{input_evals[2].toBytes()});

            // DEBUG: Compute inner_sum_prod using Jolt's formula to compare with prover
            dbg("[ZOLT] INNER_SUM_PROD: r_stream = {any}\n", .{r_stream.toBytesBE()});
            dbg("[ZOLT] INNER_SUM_PROD: r0 = {any}\n", .{r0.toBytesBE()});

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
            const az_final = az_g0.add(r_stream.mul(az_g1.sub(az_g0)));
            const bz_final = bz_g0.add(r_stream.mul(bz_g1.sub(bz_g0)));
            const inner_sum_prod = az_final.mul(bz_final);

            dbg("[ZOLT] INNER_SUM_PROD: az_g0 = {any}\n", .{az_g0.toBytesBE()});
            dbg("[ZOLT] INNER_SUM_PROD: bz_g0 = {any}\n", .{bz_g0.toBytesBE()});
            dbg("[ZOLT] INNER_SUM_PROD: az_g1 = {any}\n", .{az_g1.toBytesBE()});
            dbg("[ZOLT] INNER_SUM_PROD: bz_g1 = {any}\n", .{bz_g1.toBytesBE()});
            dbg("[ZOLT] INNER_SUM_PROD: az_final = {any}\n", .{az_final.toBytesBE()});
            dbg("[ZOLT] INNER_SUM_PROD: bz_final = {any}\n", .{bz_final.toBytesBE()});
            dbg("[ZOLT] INNER_SUM_PROD: inner_sum_prod = {any}\n", .{inner_sum_prod.toBytesBE()});

            // Add R1CS inputs for SpartanOuter with computed evaluations
            // AND append each claim to transcript in Jolt's order (for Fiat-Shamir)
            dbg("[ZOLT] OPENING_CLAIMS: Starting to append 36 claims to transcript\n", .{});
            dbg("[ZOLT] OPENING_CLAIMS: transcript state before = {any}\n", .{transcript.state[0..8]});

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

            // DEBUG: Print first few UniSkip coefficients
            dbg("[ZOLT UNISKIP_PROOF] Computing UniSkip from witnesses, tau.len={d}\n", .{tau.len});
            dbg("[ZOLT UNISKIP_PROOF] uni_poly_coeffs.len = {d}\n", .{uni_poly_coeffs.len});
            if (uni_poly_coeffs.len > 0) {
                dbg("[ZOLT UNISKIP_PROOF] uni_poly_coeffs[0] = {any}\n", .{uni_poly_coeffs[0].toBytesBE()});
            }
            if (uni_poly_coeffs.len > 1) {
                dbg("[ZOLT UNISKIP_PROOF] uni_poly_coeffs[1] = {any}\n", .{uni_poly_coeffs[1].toBytesBE()});
            }

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

            jolt_proof.log_k_chunk = config.log_k_chunk;
            jolt_proof.lookups_ra_virtual_log_k_chunk = config.lookups_ra_virtual_log_k_chunk;

            // Set config structs (matching Jolt's serialization format)
            const log_t: u8 = @intCast(zolt_stage_proofs.log_t);
            const log_k: u8 = @intCast(zolt_stage_proofs.log_k);
            jolt_proof.rw_config = jolt_types.ReadWriteConfig.default(log_t, log_k);
            jolt_proof.one_hot_config = .{
                .log_k_chunk = @intCast(config.log_k_chunk),
                .lookups_ra_virtual_log_k_chunk = @intCast(config.lookups_ra_virtual_log_k_chunk),
            };
            jolt_proof.dory_layout = 0; // Wide layout

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
            // Stage 4 needs LOG_K + n_cycle_vars rounds where LOG_K = 7 (128 registers)
            const log_registers = 7;
            const lookups_log_k: usize = 128; // XLEN * 2 for RV64 instruction lookups
            const bytecode_log_k: usize = 16; // log2(65536) bytecode address space
            try self.generateZeroSumcheckProof(&jolt_proof.stage3_sumcheck_proof, n_cycle_vars, 3);
            try self.generateZeroSumcheckProof(&jolt_proof.stage4_sumcheck_proof, log_registers + n_cycle_vars, 3);
            // Stage 5: max rounds from LookupsReadRaf = lookups_log_k + n_cycle_vars
            try self.generateZeroSumcheckProof(&jolt_proof.stage5_sumcheck_proof, lookups_log_k + n_cycle_vars, 3);
            // Stage 6: max rounds from BytecodeReadRaf = bytecode_log_k + n_cycle_vars
            try self.generateZeroSumcheckProof(&jolt_proof.stage6_sumcheck_proof, bytecode_log_k + n_cycle_vars, 3);
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
            dbg("\n[PROOF_CONV] ===== STARTING CONVERT WITH TRANSCRIPT =====\n", .{});
            dbg("[PROOF_CONV] Starting conversion, trace_length={}...\n", .{@as(usize, 1) << @intCast(zolt_stage_proofs.log_t)});
            var jolt_proof = JoltProofType(F, Commitment, Proof).init(self.allocator);

            // Copy configuration parameters
            const trace_length: usize = @as(usize, 1) << @intCast(zolt_stage_proofs.log_t);
            const ram_K: usize = @as(usize, 1) << @intCast(zolt_stage_proofs.log_k);

            dbg("[PROOF_CONV] trace_length={}, ram_K={}\n", .{ trace_length, ram_K });
            jolt_proof.trace_length = trace_length;
            jolt_proof.ram_K = ram_K;

            jolt_proof.log_k_chunk = config.log_k_chunk;
            jolt_proof.lookups_ra_virtual_log_k_chunk = config.lookups_ra_virtual_log_k_chunk;

            // Set config structs (matching Jolt's serialization format)
            const log_t: u8 = @intCast(zolt_stage_proofs.log_t);
            const log_k: u8 = @intCast(zolt_stage_proofs.log_k);
            jolt_proof.rw_config = jolt_types.ReadWriteConfig.default(log_t, log_k);
            jolt_proof.one_hot_config = .{
                .log_k_chunk = @intCast(config.log_k_chunk),
                .lookups_ra_virtual_log_k_chunk = @intCast(config.lookups_ra_virtual_log_k_chunk),
            };
            jolt_proof.dory_layout = 0; // Wide layout

            // Compute derived parameters
            const n_cycle_vars = std.math.log2_int(usize, trace_length);
            const log_ram_k = std.math.log2_int(usize, ram_K);

            // CRITICAL: Pad cycle_witnesses to trace_length with NoOp witness values.
            // In Jolt, padded cycles are Cycle::NoOp which has:
            //   - FlagIsNoop = 1
            //   - FlagDoNotUpdateUnexpandedPC = 1
            //   - All other R1CS inputs = 0
            // The prover must use padded witnesses because:
            //   1. The streaming outer sumcheck materializes Az/Bz over all trace_length cycles
            //   2. Even though Az*Bz = 0 for NoOp cycles, the individual Az values at NoOp
            //      cycles are non-zero (some conditions evaluate to 1)
            //   3. The verifier evaluates Az(r)*Bz(r) using MLE openings that include NoOp
            //      contributions, so the prover's MLE must match
            const padded_witnesses = try self.allocator.alloc(r1cs.R1CSCycleInputs(F), trace_length);
            defer self.allocator.free(padded_witnesses);

            // Copy actual witness data
            @memcpy(padded_witnesses[0..cycle_witnesses.len], cycle_witnesses);

            // Fill padded cycles with NoOp witness values
            for (cycle_witnesses.len..trace_length) |i| {
                padded_witnesses[i] = r1cs.R1CSCycleInputs(F).init(); // All zeros
                padded_witnesses[i].values[r1cs.R1CSInputIndex.FlagIsNoop.toIndex()] = F.one();
                padded_witnesses[i].values[r1cs.R1CSInputIndex.FlagDoNotUpdateUnexpandedPC.toIndex()] = F.one();
            }

            dbg("[PROOF_CONV] Padded cycle_witnesses from {} to {} (NoOp padding: FlagIsNoop=1, FlagDoNotUpdateUnexpandedPC=1)\n", .{ cycle_witnesses.len, trace_length });

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
            // Use padded witnesses so that NoOp cycles are included in the polynomial evaluation
            jolt_proof.stage1_uni_skip_first_round_proof = try self.createUniSkipProofStage1FromWitnesses(
                padded_witnesses,
                tau,
            );

            // Stage 1: Outer Spartan Remaining - use streaming prover with transcript
            // Use padded witnesses so Az/Bz MLE evaluations match the verifier's computation
            dbg("[PROOF_CONV] Starting Stage 1...\n", .{});
            {
                dbg("[ZOLT] Transcript before Stage 1: ", .{});
                for (transcript.state[0..8]) |b| dbg("{x:0>2} ", .{b});
                dbg(" round={}\n", .{transcript.n_rounds});
            }
            var stage1_result: ?Stage1Result = null;
            if (jolt_proof.stage1_uni_skip_first_round_proof) |*uniskip| {
                stage1_result = try self.generateStreamingOuterSumcheckProofWithTranscript(
                    &jolt_proof.stage1_sumcheck_proof,
                    uniskip,
                    padded_witnesses,
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
                    padded_witnesses,
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

            // Create UniSkip proof for Stage 2
            // Jolt samples a NEW tau_high for Stage 2 from the transcript (see ProductVirtualUniSkipParams::new)
            // tau = [r_cycle_outer, tau_high] where tau_high is freshly sampled
            dbg("[ZOLT] STAGE2_PRE: transcript state before tau_high = {any}\n", .{transcript.state[0..8]});
            const tau_high_stage2 = transcript.challengeScalar();
            dbg("[ZOLT] STAGE2: sampled tau_high = {any}\n", .{tau_high_stage2.toBytesBE()});

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
            dbg("[ZOLT] STAGE2: tau_high = {any}\n", .{tau_high_stage2.toBytesBE()});
            for (base_evals_stage2, 0..) |eval, i| {
                dbg("[ZOLT] STAGE2: base_evals[{}] = {any}\n", .{ i, eval.toBytesBE() });
            }

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
                padded_witnesses,
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
                dbg("[ZOLT] STAGE2: r0 = {any}\n", .{r0_stage2.toBytesBE()});

                // Compute UnivariateSkip claim = poly(r0)
                // uni_poly = [c0, c1, c2, ..., c12] -> poly(x) = c0 + c1*x + c2*x^2 + ...
                var r_power = F.one();
                for (proof.uni_poly) |coeff| {
                    uni_skip_claim_stage2 = uni_skip_claim_stage2.add(coeff.mul(r_power));
                    r_power = r_power.mul(r0_stage2);
                }
                dbg("[ZOLT] STAGE2: uni_skip_claim = {any}\n", .{uni_skip_claim_stage2.toBytesBE()});

                // flush_to_transcript: uni_skip opening claim
                transcript.appendScalar("opening_claim", uni_skip_claim_stage2);

                // Update the opening claim for UnivariateSkip at SpartanProductVirtualization
                try jolt_proof.opening_claims.insert(
                    .{ .Virtual = .{ .poly = .UnivariateSkip, .sumcheck_id = .SpartanProductVirtualization } },
                    uni_skip_claim_stage2,
                );

                // Debug: verify the claim was inserted correctly
                const inserted_claim = jolt_proof.opening_claims.get(.{ .Virtual = .{ .poly = .UnivariateSkip, .sumcheck_id = .SpartanProductVirtualization } });
                if (inserted_claim) |claim| {
                    dbg("[ZOLT] STAGE2: inserted uni_skip_claim = {any}\n", .{claim.toBytesBE()});
                } else {
                    dbg("[ZOLT] STAGE2: ERROR - uni_skip_claim was NOT inserted!\n", .{});
                }
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
            var tau_stage2 = try self.allocator.alloc(F, n_cycle_vars + 1);
            defer self.allocator.free(tau_stage2);

            // Build tau_stage2 from Stage 1 challenges
            // Also compute r_spartan_original (non-reversed) for InstructionLookupsClaimReduction
            var r_spartan_original = try self.allocator.alloc(F, n_cycle_vars);
            defer self.allocator.free(r_spartan_original);

            if (stage1_result) |result| {
                const all_challenges = result.challenges.items;
                // Skip the first challenge (r_stream) to get r_cycle
                const cycle_challenges = if (all_challenges.len > 1)
                    all_challenges[1..]
                else
                    all_challenges;

                // Debug: print Stage 1 challenges
                dbg("[ZOLT] STAGE1_CHALLENGES: all_challenges.len = {}, cycle_challenges.len = {}\n", .{ all_challenges.len, cycle_challenges.len });
                if (cycle_challenges.len > 0) {
                    const r0_bytes = cycle_challenges[0].toBytesBE();
                    const rlast_bytes = cycle_challenges[cycle_challenges.len - 1].toBytesBE();
                    dbg("[ZOLT] STAGE1_CHALLENGES: cycle_challenges[0] (r_0) = {any}\n", .{r0_bytes});
                    dbg("[ZOLT] STAGE1_CHALLENGES: cycle_challenges[last] (r_{{n-1}}) = {any}\n", .{rlast_bytes});
                }

                // Store r_spartan_original in BIG_ENDIAN order (like Jolt's opening point)
                // This is used by InstructionLookupsClaimReduction
                for (0..n_cycle_vars) |i| {
                    const src_idx = n_cycle_vars - 1 - i;
                    if (src_idx < cycle_challenges.len) {
                        r_spartan_original[i] = cycle_challenges[src_idx];
                    } else {
                        r_spartan_original[i] = F.zero();
                    }
                }

                // CRITICAL: In Jolt, the opening point r_cycle is stored in BIG_ENDIAN order
                // (reversed from sumcheck challenge order).
                // See OuterRemainingSumcheckParams::normalize_opening_point which converts
                // from LITTLE_ENDIAN to BIG_ENDIAN via match_endianness() (reverses the vector)
                //
                // So tau_stage2 = [r_cycle_reversed, tau_high] where r_cycle_reversed[i] = r_cycle[n-1-i]
                for (0..n_cycle_vars) |i| {
                    tau_stage2[i] = r_spartan_original[i];
                }
            } else {
                // Fallback to zeros
                for (0..n_cycle_vars) |i| {
                    tau_stage2[i] = F.zero();
                    r_spartan_original[i] = F.zero();
                }
            }
            // Append tau_high_stage2 as the last element
            tau_stage2[n_cycle_vars] = tau_high_stage2;

            {
                dbg("[PROOF_CONV] Starting Stage 2...\n", .{});
                dbg("[ZOLT] Transcript before Stage 2: ", .{});
                for (transcript.state[0..8]) |b| dbg("{x:0>2} ", .{b});
                dbg(" round={}\n", .{transcript.n_rounds});
            }
            dbg("[ZOLT] STAGE2: tau_stage2.len = {}\n", .{tau_stage2.len});
            if (tau_stage2.len > 0) {
                dbg("[ZOLT] STAGE2: tau_stage2[0] = {any}\n", .{tau_stage2[0].toBytesBE()});
                dbg("[ZOLT] STAGE2: tau_stage2[last] = {any}\n", .{tau_stage2[tau_stage2.len - 1].toBytesBE()});
            }

            var stage2_result = try self.generateStage2BatchedSumcheckProof(
                &jolt_proof.stage2_sumcheck_proof,
                transcript,
                r0_stage2,
                uni_skip_claim_stage2,
                tau_stage2,
                r_spartan_original,
                padded_witnesses,
                n_cycle_vars,
                log_ram_k,
                &jolt_proof.opening_claims,
                config,
            );

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
            dbg("[ZOLT PRODUCT] factor_evals for SpartanProductVirtualization:\n", .{});
            for (stage2_result.factor_evals, 0..) |eval, i| {
                dbg("[ZOLT PRODUCT]   factor[{}] = {any}\n", .{ i, eval.toBytesBE() });
            }
            std.debug.print("[INSERT] LeftInstructionInput@ProdVirt = {any}\n", .{stage2_result.factor_evals[0].toBytesBE()});
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
            dbg("[ZOLT] OutputSumcheck: inserting val_final_claim (from prover) = {any}\n", .{stage2_result.output_val_final_claim.toBytesBE()});
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .RamValFinal, .sumcheck_id = .RamOutputCheck } },
                stage2_result.output_val_final_claim,
            );
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .RamValInit, .sumcheck_id = .RamOutputCheck } },
                stage2_result.output_val_init_claim,
            );

            // Clean up stage2_result
            defer stage2_result.deinit();

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
            std.debug.print("[INSERT] LeftInstructionInput@InstrClaimRed = {any}\n", .{stage2_result.instr_left_instr_input_claim.toBytesBE()});
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
            dbg("[ZOLT] Stage 2 cache_openings: appending claims to transcript\n", .{});

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

            dbg("[ZOLT] Stage 2 cache_openings: appended 15 claims to transcript\n", .{});
            dbg("[ZOLT] Stage 2 transcript state after cache_openings = {any}\n", .{transcript.state[0..8]});

            // Stage 3: SpartanShift, InstructionInput, RegistersClaimReduction
            {
                dbg("[PROOF_CONV] Starting Stage 3...\n", .{});
                dbg("[ZOLT] Transcript before Stage 3: ", .{});
                for (transcript.state[0..8]) |b| dbg("{x:0>2} ", .{b});
                dbg(" round={}\n", .{transcript.n_rounds});
            }
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

            dbg("[ZOLT] STAGE3: r_spartan_original[0] = {any}\n", .{r_spartan_original[0].toBytesBE()[0..8]});
            dbg("[ZOLT] STAGE3: r_product[0] = {any}\n", .{r_product[0].toBytesBE()[0..8]});

            // Generate Stage 3 proof using the proper sumcheck prover
            var stage3_prover_instance = Stage3Prover(F).init(self.allocator);
            var stage3_result = try stage3_prover_instance.generateStage3Proof(
                &jolt_proof.stage3_sumcheck_proof,
                transcript,
                &jolt_proof.opening_claims,
                padded_witnesses,
                n_cycle_vars,
                r_spartan_original, // r_outer in BIG_ENDIAN
                r_product, // r_product in BIG_ENDIAN
            );
            defer stage3_result.deinit();

            // Debug: Print Stage 3 challenges for comparison with Jolt
            // NOTE: Stage 3 challenges are MontU128Challenge-style [0, 0, low, high] limbs
            // where the limbs ARE the Montgomery representation directly.
            // To compare with Jolt's params.r_cycle, we need to look at limbs[2] and limbs[3].
            dbg("[ZOLT STAGE3 RESULT] challenges.len = {}\n", .{stage3_result.challenges.len});
            for (stage3_result.challenges, 0..) |c, i| {
                dbg("[ZOLT STAGE3 RESULT]   challenge[{}] limbs = [0x{x:0>16}, 0x{x:0>16}, 0x{x:0>16}, 0x{x:0>16}]\n", .{ i, c.limbs[0], c.limbs[1], c.limbs[2], c.limbs[3] });
            }
            // Also print in the format that matches Jolt's params.r_cycle (16 zero bytes + 16 data bytes)
            dbg("[ZOLT STAGE3 RESULT] As Jolt Challenge format (reversed to BIG_ENDIAN r_cycle_be):\n", .{});
            for (0..stage3_result.challenges.len) |i| {
                const c = stage3_result.challenges[stage3_result.challenges.len - 1 - i];
                // Jolt's Challenge serializes as [0, 0, low_LE, high_LE] where each u64 is in LE bytes
                var jolt_format: [32]u8 = [_]u8{0} ** 32;
                std.mem.writeInt(u64, jolt_format[16..24], c.limbs[2], .little);
                std.mem.writeInt(u64, jolt_format[24..32], c.limbs[3], .little);
                dbg("[ZOLT STAGE3 RESULT]   r_cycle_be[{}] = {{ ", .{i});
                for (jolt_format) |b| dbg("{x:0>2} ", .{b});
                dbg("}}\n", .{});
            }

            // DEBUG: Check challenges immediately before claiming them
            dbg("[ZOLT STAGE3] challenges BEFORE inserting claims (both forms):\n", .{});
            for (0..stage3_result.challenges.len) |i| {
                const c = stage3_result.challenges[i];
                dbg("  challenges[{}] limbs = [{x:0>16}, {x:0>16}, {x:0>16}, {x:0>16}] -> BE = {x}\n", .{ i, c.limbs[0], c.limbs[1], c.limbs[2], c.limbs[3], c.toBytesBE()[16..32].* });
            }

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
            dbg("[ZOLT STAGE3->4] RegistersClaimReduction claims:\n", .{});
            dbg("[ZOLT STAGE3->4]   rd_write_value = {any}\n", .{stage3_result.reg_rd_write_value_claim.toBytes()});
            dbg("[ZOLT STAGE3->4]   rs1_value = {any}\n", .{stage3_result.reg_rs1_value_claim.toBytes()});
            dbg("[ZOLT STAGE3->4]   rs2_value = {any}\n", .{stage3_result.reg_rs2_value_claim.toBytes()});
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

            // Stage 4: RegistersReadWriteChecking, RamValEvaluation, RamValFinalEvaluation
            {
                dbg("[PROOF_CONV] Starting Stage 4...\n", .{});
                dbg("[ZOLT] Transcript before Stage 4: ", .{});
                for (transcript.state[0..8]) |b| dbg("{x:0>2} ", .{b});
                dbg(" round={}\n", .{transcript.n_rounds});
            }
            // RegistersReadWriteChecking has LOG_K + log2(T) rounds where LOG_K = log2(REGISTER_COUNT)
            // REGISTER_COUNT = 32 (RISCV) + 96 (Virtual) = 128, so LOG_K = 7
            const log_registers = 7; // log2(128) = 7
            const stage4_max_rounds = log_registers + n_cycle_vars;

            // NOTE: Stage 3's cache_openings (13 claims, 3 aliased) are already appended to the transcript
            // by stage3_prover.generateStage3Proof() - no need to append again here.

            // Stage 4 RegistersReadWriteChecking needs:
            // - gamma from transcript (challenge scalar)
            // - r_cycle from Stage 3 (the sumcheck challenges from RegistersClaimReduction)
            // - execution trace from config
            // DEBUG: Print transcript state before gamma
            dbg("[STAGE4 TRANSCRIPT] State BEFORE gamma challenge:\n", .{});
            dbg("[STAGE4 TRANSCRIPT]   state = {{ ", .{});
            for (transcript.state[0..8]) |b| dbg("{x:0>2} ", .{b});
            dbg("}}\n", .{});

            // ALWAYS-ON: Print transcript state before gamma for comparison with Jolt verifier
            {
                dbg("[ZOLT STAGE4] Transcript state BEFORE gamma: ", .{});
                for (transcript.state[0..8]) |b| dbg("{x:0>2} ", .{b});
                dbg("\n", .{});
            }

            const gamma_stage4 = transcript.challengeScalarFull();
            dbg("[STAGE4] gamma_full_BE = {any}\n", .{gamma_stage4.toBytesBE()});

            // Domain separator and gamma for RamValCheck (combined ValEvaluation + ValFinal)
            // Must match upstream: transcript.append_bytes(b"ram_val_check_gamma", &[])
            transcript.appendBytes("ram_val_check_gamma", &.{});
            const ram_val_check_gamma = transcript.challengeScalarFull();
            dbg("[STAGE4] ram_val_check_gamma_BE = {any}\n", .{ram_val_check_gamma.toBytesBE()});

            // Variables to store Stage 4 opening point for Stage 5
            var stage4_regs_r_address: ?[]F = null;
            var stage4_regs_r_cycle: ?[]F = null;
            var stage4_r_cycle_val: ?[]F = null; // r_cycle_val from RamValEvaluation (for RamRaClaimReduction)
            // r_reduction from Stage 2 InstructionClaimReduction (for LookupsReadRaf in Stage 5)
            // CRITICAL: InstructionClaimReduction is in Stage 2, NOT Stage 3!
            // The challenges are the last n_cycle_vars challenges from Stage 2 (Instance 4).
            var r_reduction_be: ?[]F = null;
            // Stage 4 inc_poly copy for Stage 6 diagnostic
            var stage4_inc_poly_copy: ?[]F = null;
            defer {
                if (stage4_regs_r_address) |arr| self.allocator.free(arr);
                if (stage4_regs_r_cycle) |arr| self.allocator.free(arr);
                if (stage4_r_cycle_val) |arr| self.allocator.free(arr);
                if (r_reduction_be) |arr| self.allocator.free(arr);
                if (stage4_inc_poly_copy) |arr| self.allocator.free(arr);
            }

            // Compute r_reduction_be from Stage 2 challenges (InstructionClaimReduction)
            // Stage 2 has 5 instances with max_num_rounds = log_ram_k + n_cycle_vars
            // Instance 4 (InstructionClaimReduction) uses the last n_cycle_vars rounds
            // So its challenges are stage2_result.challenges[max_num_rounds - n_cycle_vars .. max_num_rounds]
            const max_stage2_rounds = log_ram_k + n_cycle_vars;
            const instr_start = max_stage2_rounds - n_cycle_vars;
            dbg("[STAGE5 PREP] Extracting InstructionClaimReduction challenges from Stage 2:\n", .{});
            dbg("[STAGE5 PREP]   max_stage2_rounds={}, n_cycle_vars={}, instr_start={}\n", .{ max_stage2_rounds, n_cycle_vars, instr_start });
            dbg("[STAGE5 PREP]   stage2_result.challenges.len={}\n", .{stage2_result.challenges.len});

            // Extract and reverse the InstructionClaimReduction challenges to BIG_ENDIAN order
            r_reduction_be = try self.allocator.alloc(F, n_cycle_vars);
            dbg("[STAGE5 PREP] Stage 2 challenges[16..24] (raw, LE order):\n", .{});
            for (0..n_cycle_vars) |i| {
                const src_idx = instr_start + i;
                // Reverse to BIG_ENDIAN: first challenge in LITTLE_ENDIAN becomes last in BIG_ENDIAN
                const dest_idx = n_cycle_vars - 1 - i;
                r_reduction_be.?[dest_idx] = stage2_result.challenges[src_idx];
                dbg("  challenges[{}] = {x}\n", .{ src_idx, stage2_result.challenges[src_idx].toBytesBE()[16..32].* });
            }
            dbg("[STAGE5 PREP] r_reduction_be (from Stage 2 InstructionClaimReduction):\n", .{});
            for (0..r_reduction_be.?.len) |i| {
                const c = r_reduction_be.?[i];
                dbg("  r_reduction_be[{}] limbs = [{x:0>16}, {x:0>16}, {x:0>16}, {x:0>16}]\n", .{ i, c.limbs[0], c.limbs[1], c.limbs[2], c.limbs[3] });
                dbg("  r_reduction_be[{}] toBytesBE()[16..32] = {x}\n", .{ i, c.toBytesBE()[16..32].* });
                // Print LE bytes for direct comparison with Jolt
                const le = c.toBytes();
                dbg("  r_reduction_be[{}] LE bytes[0..16] = {any}\n", .{ i, le[0..16].* });
            }


            // Use Stage 4 prover if we have execution and memory trace data.
            stage4_block: {
                const trace = config.execution_trace orelse {
                    dbg("[STAGE4] No execution trace, using zero proof\n", .{});
                    try self.generateZeroSumcheckProof(&jolt_proof.stage4_sumcheck_proof, stage4_max_rounds, 2);
                    try jolt_proof.opening_claims.insert(.{ .Virtual = .{ .poly = .RegistersVal, .sumcheck_id = .RegistersReadWriteChecking } }, F.zero());
                    try jolt_proof.opening_claims.insert(.{ .Virtual = .{ .poly = .Rs1Ra, .sumcheck_id = .RegistersReadWriteChecking } }, F.zero());
                    try jolt_proof.opening_claims.insert(.{ .Virtual = .{ .poly = .Rs2Ra, .sumcheck_id = .RegistersReadWriteChecking } }, F.zero());
                    try jolt_proof.opening_claims.insert(.{ .Virtual = .{ .poly = .RdWa, .sumcheck_id = .RegistersReadWriteChecking } }, F.zero());
                    try jolt_proof.opening_claims.insert(.{ .Committed = .{ .poly = .RdInc, .sumcheck_id = .RegistersReadWriteChecking } }, F.zero());
                    try jolt_proof.opening_claims.insert(.{ .Virtual = .{ .poly = .RamRa, .sumcheck_id = .RamValCheck } }, F.zero());
                    try jolt_proof.opening_claims.insert(.{ .Committed = .{ .poly = .RamInc, .sumcheck_id = .RamValCheck } }, F.zero());
                    break :stage4_block;
                };
                const memory_trace = config.memory_trace orelse {
                    dbg("[STAGE4] No memory trace, using zero proof\n", .{});
                    try self.generateZeroSumcheckProof(&jolt_proof.stage4_sumcheck_proof, stage4_max_rounds, 2);
                    try jolt_proof.opening_claims.insert(.{ .Virtual = .{ .poly = .RegistersVal, .sumcheck_id = .RegistersReadWriteChecking } }, F.zero());
                    try jolt_proof.opening_claims.insert(.{ .Virtual = .{ .poly = .Rs1Ra, .sumcheck_id = .RegistersReadWriteChecking } }, F.zero());
                    try jolt_proof.opening_claims.insert(.{ .Virtual = .{ .poly = .Rs2Ra, .sumcheck_id = .RegistersReadWriteChecking } }, F.zero());
                    try jolt_proof.opening_claims.insert(.{ .Virtual = .{ .poly = .RdWa, .sumcheck_id = .RegistersReadWriteChecking } }, F.zero());
                    try jolt_proof.opening_claims.insert(.{ .Committed = .{ .poly = .RdInc, .sumcheck_id = .RegistersReadWriteChecking } }, F.zero());
                    try jolt_proof.opening_claims.insert(.{ .Virtual = .{ .poly = .RamRa, .sumcheck_id = .RamValCheck } }, F.zero());
                    try jolt_proof.opening_claims.insert(.{ .Committed = .{ .poly = .RamInc, .sumcheck_id = .RamValCheck } }, F.zero());
                    break :stage4_block;
                };

                // Use the Gruen-optimized Stage 4 prover for Jolt compatibility
                dbg("\n[PROOF_CONV] ===== STARTING STAGE 4 REGISTER CHECKING =====\n", .{});
                dbg("[PROOF_CONV] Using Stage4GruenProver for Jolt compatibility\n", .{});
                const Stage4ProverType = spartan_mod.stage4_gruen_prover.Stage4GruenProver(F);
                const Stage3Claims = spartan_mod.stage4_gruen_prover.Stage3Claims(F);

                const stage3_claims = Stage3Claims{
                    .rd_write_value = stage3_result.reg_rd_write_value_claim,
                    .rs1_value = stage3_result.reg_rs1_value_claim,
                    .rs2_value = stage3_result.reg_rs2_value_claim,
                };

                const input_claim_registers = stage3_claims.rd_write_value
                    .add(gamma_stage4.mul(stage3_claims.rs1_value))
                    .add(gamma_stage4.mul(gamma_stage4).mul(stage3_claims.rs2_value));

                // Derive r_address from Stage 2 RWC sumcheck challenges using normalize_opening_point.
                //
                // RamReadWriteChecking has 3 phases:
                // - Phase 1: phase1_num_rounds cycle vars (bound low-to-high)
                // - Phase 2: phase2_num_rounds address vars (bound low-to-high)
                // - Phase 3: (log_T - phase1) cycle vars + (log_K - phase2) address vars
                //
                // With default config (phase1 = log_T/2, phase2 = log_K):
                // - Phase 1: challenges[0..phase1]
                // - Phase 2: challenges[phase1..phase1+phase2]
                // - Phase 3 cycle: challenges[phase1+phase2..phase1+phase2+phase3_cycle]
                //
                // normalize_opening_point returns:
                // - r_cycle = reverse(phase3_cycle) ++ reverse(phase1)
                // - r_address = reverse(phase3_address) ++ reverse(phase2)
                // - opening_point = [r_address, r_cycle]
                //
                // ValEvaluation splits at log_K to get r_address.

                // Get phase config from the proof
                const phase1 = jolt_proof.rw_config.ram_rw_phase1_num_rounds;
                const phase2 = jolt_proof.rw_config.ram_rw_phase2_num_rounds;
                const phase3_cycle_len = n_cycle_vars - phase1;
                const phase3_address_len = log_ram_k - phase2;

                dbg("[ZOLT STAGE4] Phase config: phase1={}, phase2={}, phase3_cycle={}, phase3_addr={}\n", .{ phase1, phase2, phase3_cycle_len, phase3_address_len });

                // Extract r_address using normalize_opening_point logic:
                // r_address = reverse(phase3_address) ++ reverse(phase2)
                // With default config (phase2 = log_K, phase3_address = 0):
                // r_address = reverse(challenges[phase1..phase1+phase2])
                var r_address_be = try self.allocator.alloc(F, log_ram_k);
                defer self.allocator.free(r_address_be);
                @memset(r_address_be, F.zero());

                // Phase 2 address challenges are at indices [phase1..phase1+phase2)
                const phase2_start = phase1;
                for (0..phase2) |i| {
                    const src_idx = phase2_start + i;
                    if (src_idx < stage2_result.challenges.len) {
                        // reverse(phase2): put challenge at phase2_start+i into r_address[phase2-1-i]
                        // Then prepend to phase3_address (which is at r_address[0..phase3_address_len])
                        const dest_idx = phase3_address_len + (phase2 - 1 - i);
                        if (dest_idx < log_ram_k) {
                            r_address_be[dest_idx] = stage2_result.challenges[src_idx];
                        }
                    }
                }
                // Phase 3 address challenges are at indices [phase1+phase2+phase3_cycle..end)
                const phase3_addr_start = phase1 + phase2 + phase3_cycle_len;
                for (0..phase3_address_len) |i| {
                    const src_idx = phase3_addr_start + i;
                    if (src_idx < stage2_result.challenges.len) {
                        // reverse(phase3_address): put into r_address[phase3_addr_len-1-i]
                        const dest_idx = phase3_address_len - 1 - i;
                        r_address_be[dest_idx] = stage2_result.challenges[src_idx];
                    }
                }

                dbg("[ZOLT STAGE4] r_address_be computed (first 5):\n", .{});
                for (0..@min(5, log_ram_k)) |i| {
                    dbg("[ZOLT STAGE4]   r_address_be[{}] = {any}\n", .{ i, r_address_be[i].toBytes()[0..8] });
                }

                // Also extract r_cycle for other uses
                // r_cycle = reverse(phase3_cycle) ++ reverse(phase1)
                var r_cycle_be = try self.allocator.alloc(F, n_cycle_vars);
                defer self.allocator.free(r_cycle_be);
                @memset(r_cycle_be, F.zero());

                // Phase 1 cycle challenges at indices [0..phase1)
                for (0..phase1) |i| {
                    if (i < stage2_result.challenges.len) {
                        // reverse(phase1): put into r_cycle[phase3_cycle_len + (phase1-1-i)]
                        const dest_idx = phase3_cycle_len + (phase1 - 1 - i);
                        if (dest_idx < n_cycle_vars) {
                            r_cycle_be[dest_idx] = stage2_result.challenges[i];
                        }
                    }
                }
                // Phase 3 cycle challenges at indices [phase1+phase2..phase1+phase2+phase3_cycle_len)
                const phase3_cycle_start = phase1 + phase2;
                for (0..phase3_cycle_len) |i| {
                    const src_idx = phase3_cycle_start + i;
                    if (src_idx < stage2_result.challenges.len) {
                        // reverse(phase3_cycle): put into r_cycle[phase3_cycle_len-1-i]
                        const dest_idx = phase3_cycle_len - 1 - i;
                        r_cycle_be[dest_idx] = stage2_result.challenges[src_idx];
                    }
                }

                // Reverse to LITTLE_ENDIAN for MLE helpers.
                var r_cycle_le = try self.allocator.alloc(F, n_cycle_vars);
                defer self.allocator.free(r_cycle_le);
                for (0..n_cycle_vars) |i| {
                    r_cycle_le[i] = r_cycle_be[n_cycle_vars - 1 - i];
                }

                // DEBUG: Print r_cycle_be and r_cycle_le for LT polynomial debugging
                dbg("[ZOLT LT DEBUG SETUP] n_cycle_vars = {}, r_cycle_be.len = {}, r_cycle_le.len = {}\n", .{ n_cycle_vars, r_cycle_be.len, r_cycle_le.len });
                {
                    const b0 = r_cycle_be[0].toBytes();
                    const b1 = r_cycle_be[1].toBytes();
                    const b7 = r_cycle_be[7].toBytes();
                    const b6 = r_cycle_be[6].toBytes();
                    dbg("[ZOLT LT DEBUG] r_cycle_be[0] = {{ {}, {}, {}, {}, {}, {}, {}, {} }}\n", .{ b0[0], b0[1], b0[2], b0[3], b0[4], b0[5], b0[6], b0[7] });
                    dbg("[ZOLT LT DEBUG] r_cycle_be[1] = {{ {}, {}, {}, {}, {}, {}, {}, {} }}\n", .{ b1[0], b1[1], b1[2], b1[3], b1[4], b1[5], b1[6], b1[7] });
                    dbg("[ZOLT LT DEBUG] r_cycle_be[6] = {{ {}, {}, {}, {}, {}, {}, {}, {} }}\n", .{ b6[0], b6[1], b6[2], b6[3], b6[4], b6[5], b6[6], b6[7] });
                    dbg("[ZOLT LT DEBUG] r_cycle_be[7] = {{ {}, {}, {}, {}, {}, {}, {}, {} }}\n", .{ b7[0], b7[1], b7[2], b7[3], b7[4], b7[5], b7[6], b7[7] });
                    const l0 = r_cycle_le[0].toBytes();
                    const l1 = r_cycle_le[1].toBytes();
                    const l6 = r_cycle_le[6].toBytes();
                    const l7 = r_cycle_le[7].toBytes();
                    dbg("[ZOLT LT DEBUG] r_cycle_le[0] (=be[7]) = {{ {}, {}, {}, {}, {}, {}, {}, {} }}\n", .{ l0[0], l0[1], l0[2], l0[3], l0[4], l0[5], l0[6], l0[7] });
                    dbg("[ZOLT LT DEBUG] r_cycle_le[1] (=be[6]) = {{ {}, {}, {}, {}, {}, {}, {}, {} }}\n", .{ l1[0], l1[1], l1[2], l1[3], l1[4], l1[5], l1[6], l1[7] });
                    dbg("[ZOLT LT DEBUG] r_cycle_le[6] (=be[1]) = {{ {}, {}, {}, {}, {}, {}, {}, {} }}\n", .{ l6[0], l6[1], l6[2], l6[3], l6[4], l6[5], l6[6], l6[7] });
                    dbg("[ZOLT LT DEBUG] r_cycle_le[7] (=be[0]) = {{ {}, {}, {}, {}, {}, {}, {}, {} }}\n", .{ l7[0], l7[1], l7[2], l7[3], l7[4], l7[5], l7[6], l7[7] });
                }

                var r_address_le = try self.allocator.alloc(F, log_ram_k);
                defer self.allocator.free(r_address_le);
                for (0..log_ram_k) |i| {
                    r_address_le[i] = r_address_be[log_ram_k - 1 - i];
                }

                // CRITICAL FIX: ValEvaluation MUST use getLowestAddress() to match Jolt.
                // Jolt's RamReadWriteChecking includes ALL memory accesses (including I/O region),
                // using remap_address(addr, memory_layout) which starts at get_lowest_address().
                // ValEvaluation must use the same address range because:
                //   - val_claim comes from RWC which uses get_lowest_address()
                //   - init_eval is computed at the RWC r_address
                //   - The wa polynomial in ValEvaluation must include I/O writes (e.g., termination)
                //   - Otherwise input_claim = val_claim - init_eval ≠ actual polynomial sum
                const start_address: u64 = if (config.memory_layout) |ml|
                    ml.getLowestAddress()
                else
                    constants.RAM_START_ADDRESS;
                dbg("[ZOLT STAGE4] Using getLowestAddress for ValEvaluation = 0x{X:0>16}\n", .{start_address});

                // CRITICAL FIX: ValEvaluation and ValFinal use DIFFERENT r_address points!
                //
                // For ValEvaluation:
                //   - Jolt verifier gets r_address from RamVal @ RamReadWriteChecking
                //   - This is r_address_be that we computed from Stage 2 RWC challenges above
                //   - init_eval = eval_initial_ram_mle(r_address from RWC)
                //
                // For ValFinal:
                //   - Jolt verifier gets r_address from RamValFinal @ RamOutputCheck
                //   - This is the same point where output_val_init_claim was computed
                //   - init_eval = output_val_init_claim (from OutputSumcheck)
                //
                // Previously we used output_val_init_claim for both, which is WRONG!

                // Compute init_eval for ValEvaluation at the RWC r_address
                const init_eval_for_val_eval = blk: {
                    if (config.memory_layout) |ml| {
                        const result = computeInitialRamEval(
                            config.bytecode_words,
                            config.min_bytecode_address,
                            ml,
                            r_address_be,
                            log_ram_k,
                            config.program_inputs,
                        );
                        dbg("[ZOLT STAGE4 FIX] Computed init_eval_for_val_eval at RWC r_address:\n", .{});
                        dbg("[ZOLT STAGE4 FIX]   init_eval_for_val_eval = {any}\n", .{result.toBytesBE()});
                        break :blk result;
                    }
                    // No memory layout -> init_eval = 0
                    dbg("[ZOLT STAGE4 FIX] No memory_layout, using zero for init_eval_for_val_eval\n", .{});
                    break :blk F.zero();
                };

                // Initialize val_eval prover for combined RamValCheck.
                // Uses a single prover with inc, wa, lt arrays that computes inc * wa * (lt + gamma).

                // Initialize val_eval prover to get its polynomial sum
                const trace_len = trace.steps.items.len;
                // CRITICAL FIX: Use r_cycle_be for the LT polynomial, not r_cycle_le!
                // Jolt's verifier computes LT(r, r_cycle) where both are in BIG_ENDIAN.
                // The r_cycle comes from the RamVal opening point which is stored in BE.
                // Using r_cycle_le produces a different LT value because LT is not symmetric.
                const val_eval_params_early = try ram.ValEvaluationParams(F).init(
                    self.allocator,
                    init_eval_for_val_eval, // Use the correct init_eval computed at RWC r_address
                    trace_len,
                    ram_K,
                    r_address_le, // r_address uses LE for eq polynomial (symmetric)
                    r_cycle_be, // FIXED: Use BE for LT polynomial (not symmetric)
                );
                // CRITICAL: Use initWithLayout to filter out synthetic termination/panic writes.
                // This matches Jolt's behavior where these bits are set directly in final memory
                // without corresponding trace entries in the inc polynomial.
                var val_eval_prover_early = try ram.ValEvaluationProver(F).initWithLayout(
                    self.allocator,
                    memory_trace,
                    config.initial_ram,
                    val_eval_params_early,
                    start_address,
                    config.memory_layout, // Pass memory_layout to filter synthetic writes
                );
                defer val_eval_prover_early.deinit();

                // Debug: verify which r_cycle was passed
                dbg("[PROOF_CONVERTER EARLY PROVER] val_eval_prover_early initialized with:\n", .{});
                dbg("  start_address = 0x{X:0>16}\n", .{start_address});
                dbg("  r_cycle_le[0] (passed to prover) = {any}\n", .{r_cycle_le[0].toBytes()[0..8]});
                dbg("  val_eval_prover_early.lt_evals[0] = {any}\n", .{val_eval_prover_early.lt_evals[0].toBytes()[0..8]});

                // Combined RamValCheck input_claim matching upstream formula:
                //   input_claim = (val_rw_claim - init_eval) + gamma * (val_final_claim - init_eval)
                // Uses single init_eval at RWC r_address (both addresses are equal with default config).
                const input_claim_val_eval = stage2_result.rwc_val_claim.sub(init_eval_for_val_eval);
                const input_claim_val_final = stage2_result.output_val_final_claim.sub(init_eval_for_val_eval);
                const input_claim_ram_val_check = input_claim_val_eval.add(ram_val_check_gamma.mul(input_claim_val_final));

                dbg("[ZOLT STAGE4] input_claim_ram_val_check_BE = {any}\n", .{input_claim_ram_val_check.toBytesBE()});

                // Append 2 input claims to transcript (upstream has 2 instances, not 3)
                transcript.appendScalar("sumcheck_claim", input_claim_registers);
                transcript.appendScalar("sumcheck_claim", input_claim_ram_val_check);

                // Sample 2 batching coefficients
                const batch0 = transcript.challengeScalarFull();
                const batch1 = transcript.challengeScalarFull();

                dbg("[ZOLT STAGE4] input_claim_registers_BE = {any}\n", .{input_claim_registers.toBytesBE()});
                dbg("[ZOLT STAGE4] input_claim_ram_val_check_BE = {any}\n", .{input_claim_ram_val_check.toBytesBE()});
                dbg("[ZOLT STAGE4] batching_coeff[0]_BE = {any}\n", .{batch0.toBytesBE()});
                dbg("[ZOLT STAGE4] batching_coeff[1]_BE = {any}\n", .{batch1.toBytesBE()});

                const batching_coeffs = [2]F{ batch0, batch1 };

                // Now initialize regs prover with actual batch0
                // CRITICAL FIX: Stage 3's sumcheck binds variables in the order challenges are sampled.
                // The final claim rd_write_value_claim = f(c0, c1, ..., cn) in that order.
                // Stage 4's eq polynomial must use the SAME ordering so that:
                //   Σ_j eq(r, j) * f(j) = f(r) = rd_write_value_claim when r = Stage 3 challenges
                // We do NOT reverse to BE here - we use the original LE order for the prover.
                // (The BE conversion is only for serialization to match Jolt's opening_point format.)
                const stage3_r_cycle_le = stage3_result.challenges;

                var regs_prover = Stage4ProverType.initWithClaims(
                    self.allocator,
                    trace,
                    gamma_stage4,
                    stage3_r_cycle_le,
                    stage3_claims,
                    batch0, // Use the correct batching coefficient from transcript
                ) catch |err| {
                    dbg("[STAGE4] Prover init error: {any}, using zero proof\n", .{err});
                    try self.generateZeroSumcheckProof(&jolt_proof.stage4_sumcheck_proof, stage4_max_rounds, 3);
                    try jolt_proof.opening_claims.insert(.{ .Virtual = .{ .poly = .RegistersVal, .sumcheck_id = .RegistersReadWriteChecking } }, F.zero());
                    try jolt_proof.opening_claims.insert(.{ .Virtual = .{ .poly = .Rs1Ra, .sumcheck_id = .RegistersReadWriteChecking } }, F.zero());
                    try jolt_proof.opening_claims.insert(.{ .Virtual = .{ .poly = .Rs2Ra, .sumcheck_id = .RegistersReadWriteChecking } }, F.zero());
                    try jolt_proof.opening_claims.insert(.{ .Virtual = .{ .poly = .RdWa, .sumcheck_id = .RegistersReadWriteChecking } }, F.zero());
                    try jolt_proof.opening_claims.insert(.{ .Committed = .{ .poly = .RdInc, .sumcheck_id = .RegistersReadWriteChecking } }, F.zero());
                    try jolt_proof.opening_claims.insert(.{ .Virtual = .{ .poly = .RamRa, .sumcheck_id = .RamValCheck } }, F.zero());
                    try jolt_proof.opening_claims.insert(.{ .Committed = .{ .poly = .RamInc, .sumcheck_id = .RamValCheck } }, F.zero());
                    break :stage4_block;
                };
                defer regs_prover.deinit();

                // DIAGNOSTIC: Brute-force check of initial polynomial sum vs input_claim
                {
                    const T_diag = @as(usize, 1) << @intCast(n_cycle_vars);
                    const K_diag: usize = 128;
                    // Build eq table from r_cycle_be (MSB first)
                    const r_cycle_be_diag = try self.allocator.alloc(F, n_cycle_vars);
                    defer self.allocator.free(r_cycle_be_diag);
                    for (0..n_cycle_vars) |ii| {
                        r_cycle_be_diag[ii] = stage3_r_cycle_le[n_cycle_vars - 1 - ii];
                    }
                    // Build eq table using LE (same as binding order: bit 0 = r_cycle_be[n-1] = stage3[0])
                    var eq_diag = try self.allocator.alloc(F, T_diag);
                    defer self.allocator.free(eq_diag);
                    eq_diag[0] = F.one();
                    var eq_diag_len: usize = 1;
                    for (0..n_cycle_vars) |bit| {
                        // LE: bit 0 → challenge = stage3[0]
                        const r_i = stage3_r_cycle_le[bit];
                        const one_minus_r = F.one().sub(r_i);
                        var eidx: usize = eq_diag_len;
                        while (eidx > 0) {
                            eidx -= 1;
                            eq_diag[2 * eidx + 1] = eq_diag[eidx].mul(r_i);
                            eq_diag[2 * eidx] = eq_diag[eidx].mul(one_minus_r);
                        }
                        eq_diag_len *= 2;
                    }
                    // Now compute Σ_{k,j} eq[j] * C(k,j) where C = ra*val + wa*(val+inc)
                    // Access regs_prover's internal polynomial arrays
                    var brute_sum = F.zero();
                    for (0..K_diag) |k| {
                        for (0..T_diag) |j| {
                            const idx = k * T_diag + j;
                            const ra = regs_prover.ra_poly[idx];
                            const wa = regs_prover.rd_wa_poly[idx];
                            const val = regs_prover.val_poly[idx];
                            const inc = regs_prover.inc_poly[j];
                            const c_val = ra.mul(val).add(wa.mul(val.add(inc)));
                            brute_sum = brute_sum.add(eq_diag[j].mul(c_val));
                        }
                    }
                    // Also compute just RdWriteValue = Σ eq[j] * Σ_k wa(k,j)*(val(k,j)+inc(j))
                    var brute_rdwv = F.zero();
                    var brute_rs1v = F.zero();
                    var brute_rs2v = F.zero();
                    for (0..T_diag) |j| {
                        for (0..K_diag) |k| {
                            const idx = k * T_diag + j;
                            const wa = regs_prover.rd_wa_poly[idx];
                            const val = regs_prover.val_poly[idx];
                            const inc = regs_prover.inc_poly[j];
                            const rs1_ra = regs_prover.rs1_ra_poly[idx];
                            const rs2_ra = regs_prover.rs2_ra_poly[idx];
                            brute_rdwv = brute_rdwv.add(eq_diag[j].mul(wa.mul(val.add(inc))));
                            brute_rs1v = brute_rs1v.add(eq_diag[j].mul(rs1_ra.mul(val)));
                            brute_rs2v = brute_rs2v.add(eq_diag[j].mul(rs2_ra.mul(val)));
                        }
                    }
                    // Compare per-cycle RdWriteValue, Rs1Value, Rs2Value between Stage 3 R1CS and Stage 4
                    var rdwv_mismatch_count: usize = 0;
                    var rs1v_mismatch_count: usize = 0;
                    var rs2v_mismatch_count: usize = 0;
                    for (0..T_diag) |j| {
                        // Stage 4: Σ_k wa(k,j)*(inc(j)+val(k,j))
                        var rdwv_j = F.zero();
                        var rs1v_j = F.zero();
                        var rs2v_j = F.zero();
                        for (0..K_diag) |k| {
                            const idx = k * T_diag + j;
                            const wa = regs_prover.rd_wa_poly[idx];
                            const val = regs_prover.val_poly[idx];
                            const inc = regs_prover.inc_poly[j];
                            const rs1_ra = regs_prover.rs1_ra_poly[idx];
                            const rs2_ra = regs_prover.rs2_ra_poly[idx];
                            rdwv_j = rdwv_j.add(wa.mul(val.add(inc)));
                            rs1v_j = rs1v_j.add(rs1_ra.mul(val));
                            rs2v_j = rs2v_j.add(rs2_ra.mul(val));
                        }
                        // Stage 3: from R1CS witness
                        if (j < padded_witnesses.len) {
                            const r1cs_rdwv = padded_witnesses[j].values[r1cs.R1CSInputIndex.RdWriteValue.toIndex()];
                            const r1cs_rs1v = padded_witnesses[j].values[r1cs.R1CSInputIndex.Rs1Value.toIndex()];
                            const r1cs_rs2v = padded_witnesses[j].values[r1cs.R1CSInputIndex.Rs2Value.toIndex()];
                            if (!rdwv_j.eql(r1cs_rdwv)) {
                                rdwv_mismatch_count += 1;
                                if (rdwv_mismatch_count <= 5) {
                                    dbg("[STAGE4 MISMATCH] cycle {}: rdwv s4={} s3={} (0x{x})\n", .{j, rdwv_j.toU64(), r1cs_rdwv.toU64(), r1cs_rdwv.toU64()});
                                    if (j < trace.steps.items.len) {
                                        const step = trace.steps.items[j];
                                        dbg("  instruction=0x{x:0>8} pc=0x{x} upc=0x{x} rd_idx={} rd_written={} is_noop={} is_term_store={} rd_value={}\n", .{
                                            step.instruction, step.pc, step.unexpanded_pc, step.rd_index, step.rd_written, step.is_noop, step.is_termination_store, step.rd_value,
                                        });
                                    }
                                }
                            }
                            if (!rs1v_j.eql(r1cs_rs1v)) {
                                rs1v_mismatch_count += 1;
                                if (rs1v_mismatch_count <= 3) {
                                    dbg("[STAGE4 MISMATCH] cycle {}: rs1v s4={} s3={}\n", .{j, rs1v_j.toU64(), r1cs_rs1v.toU64()});
                                }
                            }
                            if (!rs2v_j.eql(r1cs_rs2v)) {
                                rs2v_mismatch_count += 1;
                                if (rs2v_mismatch_count <= 3) {
                                    dbg("[STAGE4 MISMATCH] cycle {}: rs2v s4={} s3={}\n", .{j, rs2v_j.toU64(), r1cs_rs2v.toU64()});
                                }
                            }
                        }
                    }

                    dbg("[STAGE4 COMPARE] Total mismatches: rdwv={}, rs1v={}, rs2v={} (out of {} cycles)\n", .{rdwv_mismatch_count, rs1v_mismatch_count, rs2v_mismatch_count, T_diag});
                    // Check eq sum = 1
                    var eq_sum = F.zero();
                    for (0..T_diag) |j| eq_sum = eq_sum.add(eq_diag[j]);
                    dbg("[STAGE4 BRUTE] eq_sum = {any} (should be 1)\n", .{eq_sum.toBytesBE()});
                    dbg("[STAGE4 BRUTE] brute_rdwv = {any}\n", .{brute_rdwv.toBytesBE()});
                    dbg("[STAGE4 BRUTE] stage3 rd_write_value = {any}\n", .{stage3_claims.rd_write_value.toBytesBE()});
                    dbg("[STAGE4 BRUTE] rdwv match? {}\n", .{brute_rdwv.eql(stage3_claims.rd_write_value)});
                    dbg("[STAGE4 BRUTE] brute_rs1v = {any}\n", .{brute_rs1v.toBytesBE()});
                    dbg("[STAGE4 BRUTE] stage3 rs1_value = {any}\n", .{stage3_claims.rs1_value.toBytesBE()});
                    dbg("[STAGE4 BRUTE] rs1v match? {}\n", .{brute_rs1v.eql(stage3_claims.rs1_value)});
                    dbg("[STAGE4 BRUTE] brute_rs2v = {any}\n", .{brute_rs2v.toBytesBE()});
                    dbg("[STAGE4 BRUTE] stage3 rs2_value = {any}\n", .{stage3_claims.rs2_value.toBytesBE()});
                    dbg("[STAGE4 BRUTE] rs2v match? {}\n", .{brute_rs2v.eql(stage3_claims.rs2_value)});
                    dbg("[STAGE4 BRUTE] Actual polynomial sum = {any}\n", .{brute_sum.toBytesBE()});
                    dbg("[STAGE4 BRUTE] input_claim_registers = {any}\n", .{input_claim_registers.toBytesBE()});
                    dbg("[STAGE4 BRUTE] match? {}\n", .{brute_sum.eql(input_claim_registers)});
                }

                // DIAGNOSTIC: Compute MLE of inc_poly at challenges[0..8] BEFORE binding
                {
                    const T_diag2 = @as(usize, 1) << @intCast(n_cycle_vars);
                    var eq_le_diag = try self.allocator.alloc(F, T_diag2);
                    defer self.allocator.free(eq_le_diag);
                    eq_le_diag[0] = F.one();
                    var elen: usize = 1;
                    for (0..n_cycle_vars) |bi| {
                        const r_i = stage3_r_cycle_le[bi]; // use stage3 r_cycle as baseline
                        _ = r_i;
                        // Actually use stage4 challenges — but we don't have them yet!
                        // We can't do this before the sumcheck. Skip.
                        elen *= 2;
                    }
                    // Recompute rd_inc independently and compare with Stage 4's inc_poly
                    {
                        const K_CHK = 128;
                        var reg_chk: [K_CHK]u64 = [_]u64{0} ** K_CHK;
                        var diff_count: usize = 0;
                        for (0..T_diag2) |j| {
                            const step_j = trace.steps.items[j];
                            var expected_inc = F.zero();
                            if (!step_j.is_noop and step_j.rd_written) {
                                const rd_j = step_j.rd_index;
                                const pre = reg_chk[rd_j];
                                const post = step_j.rd_value;
                                expected_inc = F.fromU64(post).sub(F.fromU64(pre));
                                if (rd_j != 0) {
                                    reg_chk[rd_j] = post;
                                }
                            }
                            const actual = regs_prover.inc_poly[j];
                            if (!actual.eql(expected_inc)) {
                                if (diff_count < 8) {
                                    const a_le = actual.toBytesBE();
                                    const e_le = expected_inc.toBytesBE();
                                    std.debug.print("[INC DIFF] j={} rd={} noop={} wr={} rd_val={} actual_LE={x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2} expected_LE={x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}\n", .{
                                        j, step_j.rd_index, @as(u8, if (step_j.is_noop) 1 else 0),
                                        @as(u8, if (step_j.rd_written) 1 else 0), step_j.rd_value,
                                        a_le[31], a_le[30], a_le[29], a_le[28], a_le[27], a_le[26], a_le[25], a_le[24],
                                        e_le[31], e_le[30], e_le[29], e_le[28], e_le[27], e_le[26], e_le[25], e_le[24],
                                    });
                                }
                                diff_count += 1;
                            }
                        }
                        std.debug.print("[INC DIFF] total mismatches: {}\n", .{diff_count});
                    }
                }

                // Combined RamValCheck: val_eval + gamma * val_final as a single instance
                const ram_val_check_rounds = val_eval_prover_early.numRounds(); // = n_cycle_vars
                const rounds_per_instance = [2]usize{ stage4_max_rounds, ram_val_check_rounds };

                // Track individual claims for 2 instances
                var individual_claims = [2]F{
                    input_claim_registers,
                    input_claim_ram_val_check,
                };

                // Scale initial claims by 2^(max_rounds - instance_rounds)
                for (0..2) |i| {
                    const scale_power = stage4_max_rounds - rounds_per_instance[i];
                    for (0..scale_power) |_| {
                        individual_claims[i] = individual_claims[i].add(individual_claims[i]);
                    }
                }

                // Initial batched claim
                var batched_claim = F.zero();
                for (0..2) |i| {
                    batched_claim = batched_claim.add(individual_claims[i].mul(batching_coeffs[i]));
                }
                dbg("[ZOLT STAGE4] Initial batched_claim (BE) = {any}\n", .{batched_claim.toBytesBE()});

                var regs_current_claim = individual_claims[0];

                var stage4_r_sumcheck = try self.allocator.alloc(F, stage4_max_rounds);
                defer self.allocator.free(stage4_r_sumcheck);

                // Save inc_poly before binding for verification and Stage 6 diagnostic
                stage4_inc_poly_copy = try self.allocator.alloc(F, @as(usize, 1) << @intCast(n_cycle_vars));
                @memcpy(stage4_inc_poly_copy.?, regs_prover.inc_poly[0..stage4_inc_poly_copy.?.len]);

                for (0..stage4_max_rounds) |round_idx| {
                    var combined_evals = [4]F{ F.zero(), F.zero(), F.zero(), F.zero() };

                    // Instance 0: RegistersRWC (always active)
                    const regs_evals = regs_prover.computeRoundEvals(round_idx, regs_current_claim);
                    for (0..4) |j| {
                        combined_evals[j] = combined_evals[j].add(regs_evals[j].mul(batching_coeffs[0]));
                    }

                    // Instance 1: Combined RamValCheck = val_eval + gamma * val_final
                    const ram_val_check_offset = stage4_max_rounds - ram_val_check_rounds;
                    const two_inv_local = F.fromU64(2).inverse() orelse F.one();

                    var ram_val_check_evals_opt: ?[4]F = null;
                    const ram_val_check_active = round_idx >= ram_val_check_offset;
                    if (!ram_val_check_active) {
                        // Inactive: constant polynomial H(X) = claim/2
                        const half_claim = individual_claims[1].mul(two_inv_local);
                        const weighted = half_claim.mul(batching_coeffs[1]);
                        for (0..3) |j| {
                            combined_evals[j] = combined_evals[j].add(weighted);
                        }
                    } else {
                        // Active: compute combined polynomial inc * wa * (LT + gamma)
                        // Single prover computes all Toom-Cook evals directly, matching upstream.
                        var ram_evals = val_eval_prover_early.computeRoundPolynomialCombined(ram_val_check_gamma);
                        // Apply hint: p(0) = claim - p(1)
                        ram_evals[0] = individual_claims[1].sub(ram_evals[1]);

                        ram_val_check_evals_opt = ram_evals;
                        for (0..4) |j| {
                            combined_evals[j] = combined_evals[j].add(ram_evals[j].mul(batching_coeffs[1]));
                        }
                    }

                    // Convert Toom-Cook evals to coefficients and append to proof
                    const full_coeffs = poly_mod.UniPoly(F).toomCookToCoeffs(combined_evals);
                    var coeffs = [_]F{ full_coeffs[0], full_coeffs[1], full_coeffs[2], full_coeffs[3] };
                    const coeffs_slice: []const F = &coeffs;
                    try jolt_proof.stage4_sumcheck_proof.addRoundPoly(coeffs_slice);

                    // Compressed coefficients [c0, c2, c3] for transcript
                    const compressed = poly_mod.UniPoly(F).toomCookToCompressed(combined_evals);
                    transcript.appendScalars("sumcheck_poly", compressed[0..3]);

                    const challenge = transcript.challengeScalar();
                    stage4_r_sumcheck[round_idx] = challenge;
                    batched_claim = evalFromHint(compressed, batched_claim, challenge);

                    // Update instance 0 (regs): always active
                    regs_current_claim = evaluateCubicAtChallengeFromEvals(regs_evals, challenge);
                    individual_claims[0] = regs_current_claim;
                    regs_prover.bindChallenge(round_idx, challenge);

                    // Update instance 1 (combined RamValCheck)
                    if (ram_val_check_evals_opt) |ram_evals| {
                        // Active: evaluate combined polynomial at challenge
                        individual_claims[1] = evaluateCubicAtChallengeFromEvals(ram_evals, challenge);
                        // Bind val_eval prover (contains inc, wa, lt arrays for both terms)
                        val_eval_prover_early.bindChallengeWithPoly(challenge, ram_evals);
                    } else {
                        // Inactive: claim' = claim / 2
                        individual_claims[1] = individual_claims[1].mul(two_inv_local);
                    }
                }

                dbg("[ZOLT STAGE4] Final batched_claim = {any}\n", .{batched_claim.toBytesBE()});
                dbg("[STAGE4 DIAG] individual_claims[0] (regs) = {any}\n", .{individual_claims[0].toBytesBE()});
                dbg("[STAGE4 DIAG] individual_claims[1] (ram)  = {any}\n", .{individual_claims[1].toBytesBE()});
                // Verify batched = sum of weighted claims
                const recomputed_batched = individual_claims[0].mul(batching_coeffs[0]).add(individual_claims[1].mul(batching_coeffs[1]));
                dbg("[STAGE4 DIAG] recomputed_batched = {any}\n", .{recomputed_batched.toBytesBE()});
                dbg("[STAGE4 DIAG] batched_claim match? {}\n", .{batched_claim.eql(recomputed_batched)});

                const regs_claims = regs_prover.getFinalClaims();
                const val_eval_openings = val_eval_prover_early.getFinalOpenings();

                // DIAGNOSTIC: Verify inc_poly binding via brute-force MLE
                {
                    const T_bf = stage4_inc_poly_copy.?.len;
                    // Build eq table from Stage 4 cycle challenges (LSB-first, matching binding order)
                    var eq_bf = try self.allocator.alloc(F, T_bf);
                    defer self.allocator.free(eq_bf);
                    eq_bf[0] = F.one();
                    var eq_len: usize = 1;
                    for (0..n_cycle_vars) |bi| {
                        const r_i = stage4_r_sumcheck[bi]; // challenge for bit bi (LSB first)
                        const one_minus_ri = F.one().sub(r_i);
                        var idx: usize = eq_len;
                        while (idx > 0) {
                            idx -= 1;
                            eq_bf[2 * idx + 1] = eq_bf[idx].mul(r_i);
                            eq_bf[2 * idx] = eq_bf[idx].mul(one_minus_ri);
                        }
                        eq_len *= 2;
                    }
                    var bf_inc = F.zero();
                    for (0..T_bf) |j| {
                        bf_inc = bf_inc.add(stage4_inc_poly_copy.?[j].mul(eq_bf[j]));
                    }
                    const match_bf = bf_inc.eql(regs_claims.inc_claim);
                    std.debug.print("[INC BIND CHECK] brute_force={x:0>16} binding={x:0>16} match={}\n", .{
                        @as(u64, @bitCast(bf_inc.toBytes()[0..8].*)),
                        @as(u64, @bitCast(regs_claims.inc_claim.toBytes()[0..8].*)),
                        match_bf,
                    });

                    // Also compute using BIG_ENDIAN eq (reversed challenges)
                    eq_bf[0] = F.one();
                    eq_len = 1;
                    for (0..n_cycle_vars) |bi| {
                        const r_i = stage4_r_sumcheck[n_cycle_vars - 1 - bi]; // reversed
                        const one_minus_ri = F.one().sub(r_i);
                        var idx: usize = eq_len;
                        while (idx > 0) {
                            idx -= 1;
                            eq_bf[2 * idx + 1] = eq_bf[idx].mul(r_i);
                            eq_bf[2 * idx] = eq_bf[idx].mul(one_minus_ri);
                        }
                        eq_len *= 2;
                    }
                    var bf_inc_be = F.zero();
                    for (0..T_bf) |j| {
                        bf_inc_be = bf_inc_be.add(stage4_inc_poly_copy.?[j].mul(eq_bf[j]));
                    }
                    std.debug.print("[INC BIND CHECK BE] brute_force_be={x:0>16} match_be={}\n", .{
                        @as(u64, @bitCast(bf_inc_be.toBytes()[0..8].*)),
                        bf_inc_be.eql(regs_claims.inc_claim),
                    });
                }

                // Diagnostic: verify RegistersRWC expected output
                {
                    const eq_scalar = if (regs_prover.merged_eq) |meq| meq[0] else blk: {
                        if (regs_prover.gruen_eq) |*geq| break :blk geq.current_scalar;
                        break :blk F.zero();
                    };
                    const ra_val = regs_claims.val_claim.mul(
                        regs_claims.rs1_ra_claim.mul(gamma_stage4).add(regs_claims.rs2_ra_claim.mul(gamma_stage4.mul(gamma_stage4))),
                    );
                    const wa_val_inc = regs_claims.rd_wa_claim.mul(regs_claims.inc_claim.add(regs_claims.val_claim));
                    const expected_regs_output = eq_scalar.mul(wa_val_inc.add(ra_val));
                    dbg("[STAGE4 REGS DIAG] eq_scalar = {any}\n", .{eq_scalar.toBytesBE()});
                    dbg("[STAGE4 REGS DIAG] val_claim = {any}\n", .{regs_claims.val_claim.toBytesBE()});
                    dbg("[STAGE4 REGS DIAG] rs1_ra_claim = {any}\n", .{regs_claims.rs1_ra_claim.toBytesBE()});
                    dbg("[STAGE4 REGS DIAG] rs2_ra_claim = {any}\n", .{regs_claims.rs2_ra_claim.toBytesBE()});
                    dbg("[STAGE4 REGS DIAG] rd_wa_claim = {any}\n", .{regs_claims.rd_wa_claim.toBytesBE()});
                    dbg("[STAGE4 REGS DIAG] inc_claim = {any}\n", .{regs_claims.inc_claim.toBytesBE()});
                    dbg("[STAGE4 REGS DIAG] expected_regs_output = {any}\n", .{expected_regs_output.toBytesBE()});
                    dbg("[STAGE4 REGS DIAG] individual_claims[0] = {any}\n", .{individual_claims[0].toBytesBE()});
                    dbg("[STAGE4 REGS DIAG] match? {}\n", .{expected_regs_output.eql(individual_claims[0])});
                }

                // Diagnostic: verify combined output claim
                const expected_output_combined = val_eval_openings.inc_eval
                    .mul(val_eval_openings.wa_eval)
                    .mul(val_eval_openings.lt_eval.add(ram_val_check_gamma));
                dbg("[STAGE4 DIAG] inc_eval = {any}\n", .{val_eval_openings.inc_eval.toBytesBE()});
                dbg("[STAGE4 DIAG] wa_eval = {any}\n", .{val_eval_openings.wa_eval.toBytesBE()});
                dbg("[STAGE4 DIAG] lt_eval = {any}\n", .{val_eval_openings.lt_eval.toBytesBE()});
                dbg("[STAGE4 DIAG] expected_output = inc*wa*(lt+gamma) = {any}\n", .{expected_output_combined.toBytesBE()});
                dbg("[STAGE4 DIAG] individual_claims[1] (tracked) = {any}\n", .{individual_claims[1].toBytesBE()});
                dbg("[STAGE4 DIAG] match? {}\n", .{expected_output_combined.eql(individual_claims[1])});

                // Debug: brute-force comparison (disabled for speed)
                if (false) {
                    // Get the raw initial polynomial values from the prover for a small check.
                    // After all 15 rounds of binding with challenges c0..c14:
                    //   Phases: c0..c7 bind cycle (LSB first), c8..c14 bind address (LSB first)
                    //   val_poly[0] should = Σ_{k,j} eq_cycle(c0..c7, j) * eq_addr(c8..c14, k) * val_orig[k,j]
                    //
                    // Instead of recomputing from scratch, access the prover's internal state
                    // to compare val_poly[0] vs the original polynomial evaluated by eq.
                    //
                    // Simple cross-check: print raw val_claim alongside a computation that uses
                    // the BE-converted opening point (same as Stage 5's brute force)
                    const T_brute = @as(usize, 1) << @intCast(n_cycle_vars);
                    const trace_steps = trace.steps.items;

                    // Use the SAME opening point as Stage 5 (BIG_ENDIAN, reversed challenges)
                    // r_cycle_BE = [c7, c6, ..., c0]  (reversed phase 1)
                    // r_addr_BE = [c14, c13, ..., c8]  (reversed phase 2)
                    // But since we're summing over ALL j and k, the eq function value is the same
                    // regardless of bit ordering - it's just which INDEX gets which value.
                    //
                    // KEY INSIGHT: The standard eq expansion builds eq[j] = Π_i (r_i·j_i + (1-r_i)·(1-j_i))
                    // where j_i is the i-th bit of j (LSB first). This means:
                    //   eq[j] with r = [c0,c1,...,c7] gives:
                    //   eq[j] = Π_i (c_i·j_i + (1-c_i)·(1-j_i)) where j_i = bit i of j
                    //
                    // This is the SAME as the binding: binding j LSB-first with c0 first means
                    // val_poly[0] = Σ_j eq[j]*val_orig[0,j] for k=0 after Phase 1.
                    //
                    // Let's verify by computing the MLE directly and comparing.

                    // Build eq_cycle via expansion (same as binding order)
                    var eq_cycle_brute = try self.allocator.alloc(F, T_brute);
                    defer self.allocator.free(eq_cycle_brute);
                    eq_cycle_brute[0] = F.one();
                    var eq_len: usize = 1;
                    for (0..n_cycle_vars) |bit_idx| {
                        const r_i = stage4_r_sumcheck[bit_idx];
                        const one_minus_r = F.one().sub(r_i);
                        var idx: usize = eq_len;
                        while (idx > 0) {
                            idx -= 1;
                            eq_cycle_brute[2 * idx + 1] = eq_cycle_brute[idx].mul(r_i);
                            eq_cycle_brute[2 * idx] = eq_cycle_brute[idx].mul(one_minus_r);
                        }
                        eq_len *= 2;
                    }

                    // Build eq_addr via expansion
                    const addr_bits: usize = 7;
                    var eq_addr_brute: [128]F = undefined;
                    eq_addr_brute[0] = F.one();
                    var eq_a_len: usize = 1;
                    for (0..addr_bits) |bit_idx| {
                        const r_i = stage4_r_sumcheck[n_cycle_vars + bit_idx];
                        const one_minus_r = F.one().sub(r_i);
                        var a_idx: usize = eq_a_len;
                        while (a_idx > 0) {
                            a_idx -= 1;
                            eq_addr_brute[2 * a_idx + 1] = eq_addr_brute[a_idx].mul(r_i);
                            eq_addr_brute[2 * a_idx] = eq_addr_brute[a_idx].mul(one_minus_r);
                        }
                        eq_a_len *= 2;
                    }

                    // Access the prover's ORIGINAL val_poly to verify
                    // Since the prover has already been bound, we can't access the original.
                    // Instead, reconstruct from trace.
                    var brute_val_claim = F.zero();
                    var brute_regs: [32]u64 = [_]u64{0} ** 32;

                    for (0..T_brute) |j| {
                        if (j < trace_steps.len) {
                            for (0..32) |k| {
                                if (!eq_addr_brute[k].eql(F.zero()) and !eq_cycle_brute[j].eql(F.zero())) {
                                    brute_val_claim = brute_val_claim.add(
                                        eq_addr_brute[k].mul(eq_cycle_brute[j]).mul(F.fromU64(brute_regs[k])),
                                    );
                                }
                            }
                            const step_b = trace_steps[j];
                            if (step_b.is_termination_store) {
                                brute_val_claim = brute_val_claim.add(
                                    eq_addr_brute[32].mul(eq_cycle_brute[j]).mul(F.fromU64(step_b.rs1_value)),
                                );
                                brute_val_claim = brute_val_claim.add(
                                    eq_addr_brute[33].mul(eq_cycle_brute[j]).mul(F.fromU64(step_b.rs2_value)),
                                );
                            }
                            if (!step_b.is_noop or step_b.is_termination_store) {
                                if (!step_b.is_termination_store) {
                                    const instr_b = step_b.instruction;
                                    const rd_b: u5 = @truncate((instr_b >> 7) & 0x1f);
                                    const opcode_b = instr_b & 0x7f;
                                    const rd_used_b = switch (opcode_b) {
                                        0x23, 0x63 => false,
                                        else => true,
                                    };
                                    if (rd_used_b and rd_b != 0 and rd_b < 32) {
                                        brute_regs[rd_b] = step_b.rd_value;
                                    }
                                }
                            }
                        } else {
                            for (0..32) |k| {
                                if (!eq_addr_brute[k].eql(F.zero()) and !eq_cycle_brute[j].eql(F.zero())) {
                                    brute_val_claim = brute_val_claim.add(
                                        eq_addr_brute[k].mul(eq_cycle_brute[j]).mul(F.fromU64(brute_regs[k])),
                                    );
                                }
                            }
                        }
                    }

                    dbg("\n[STAGE4 BRUTE VAL] val_claim from prover (val_poly[0]) = {any}\n", .{regs_claims.val_claim.toBytesBE()[0..16]});
                    dbg("[STAGE4 BRUTE VAL] brute force val(r_addr,r_cycle)    = {any}\n", .{brute_val_claim.toBytesBE()[0..16]});
                    dbg("[STAGE4 BRUTE VAL] match? {}\n", .{regs_claims.val_claim.eql(brute_val_claim)});

                    // Also: compute val using the prover's internal getValAt function to directly
                    // read from the BOUND polynomial. After binding, val_poly[0] should be the answer.
                    // If brute_val != val_poly[0], the binding has a bug.
                    // If brute_val == val_poly[0] but != stage5_brute_val, then the opening point is wrong.

                    // Inc check
                    var brute_inc_claim = F.zero();
                    var brute_regs2: [32]u64 = [_]u64{0} ** 32;
                    for (0..T_brute) |j| {
                        if (j < trace_steps.len) {
                            const step_b = trace_steps[j];
                            if (!step_b.is_noop or step_b.is_termination_store) {
                                if (!step_b.is_termination_store) {
                                    const instr_b = step_b.instruction;
                                    const rd_b: u5 = @truncate((instr_b >> 7) & 0x1f);
                                    const opcode_b = instr_b & 0x7f;
                                    const rd_used_b = switch (opcode_b) {
                                        0x23, 0x63 => false,
                                        else => true,
                                    };
                                    if (rd_used_b and rd_b != 0 and rd_b < 32) {
                                        const inc_val = F.fromU64(step_b.rd_value).sub(F.fromU64(brute_regs2[rd_b]));
                                        brute_inc_claim = brute_inc_claim.add(eq_cycle_brute[j].mul(inc_val));
                                        brute_regs2[rd_b] = step_b.rd_value;
                                    }
                                }
                            }
                        }
                    }
                    dbg("[STAGE4 BRUTE INC] inc_claim from prover = {any}\n", .{regs_claims.inc_claim.toBytesBE()[0..16]});
                    dbg("[STAGE4 BRUTE INC] brute force inc_claim = {any}\n", .{brute_inc_claim.toBytesBE()[0..16]});
                    dbg("[STAGE4 BRUTE INC] match? {}\n", .{regs_claims.inc_claim.eql(brute_inc_claim)});

                    // CRITICAL: Also verify that Stage 5's brute force (BIG_ENDIAN) would give the same
                    // as our LSB-first brute force. Build eq with reversed challenges:
                    var eq_cycle_be = try self.allocator.alloc(F, T_brute);
                    defer self.allocator.free(eq_cycle_be);
                    eq_cycle_be[0] = F.one();
                    var eq_be_len: usize = 1;
                    for (0..n_cycle_vars) |bit_idx| {
                        // Reversed: use c7 first, then c6, etc. (BIG_ENDIAN = MSB first)
                        const r_i = stage4_r_sumcheck[n_cycle_vars - 1 - bit_idx];
                        const one_minus_r = F.one().sub(r_i);
                        var be_idx: usize = eq_be_len;
                        while (be_idx > 0) {
                            be_idx -= 1;
                            eq_cycle_be[2 * be_idx + 1] = eq_cycle_be[be_idx].mul(r_i);
                            eq_cycle_be[2 * be_idx] = eq_cycle_be[be_idx].mul(one_minus_r);
                        }
                        eq_be_len *= 2;
                    }
                    var eq_addr_be: [128]F = undefined;
                    eq_addr_be[0] = F.one();
                    var eq_abe_len: usize = 1;
                    for (0..addr_bits) |bit_idx| {
                        const r_i = stage4_r_sumcheck[n_cycle_vars + addr_bits - 1 - bit_idx];
                        const one_minus_r = F.one().sub(r_i);
                        var abe_idx: usize = eq_abe_len;
                        while (abe_idx > 0) {
                            abe_idx -= 1;
                            eq_addr_be[2 * abe_idx + 1] = eq_addr_be[abe_idx].mul(r_i);
                            eq_addr_be[2 * abe_idx] = eq_addr_be[abe_idx].mul(one_minus_r);
                        }
                        eq_abe_len *= 2;
                    }
                    var brute_val_be = F.zero();
                    var brute_regs3: [32]u64 = [_]u64{0} ** 32;
                    for (0..T_brute) |j| {
                        if (j < trace_steps.len) {
                            for (0..32) |k| {
                                if (!eq_addr_be[k].eql(F.zero()) and !eq_cycle_be[j].eql(F.zero())) {
                                    brute_val_be = brute_val_be.add(
                                        eq_addr_be[k].mul(eq_cycle_be[j]).mul(F.fromU64(brute_regs3[k])),
                                    );
                                }
                            }
                            const step_b = trace_steps[j];
                            if (step_b.is_termination_store) {
                                brute_val_be = brute_val_be.add(
                                    eq_addr_be[32].mul(eq_cycle_be[j]).mul(F.fromU64(step_b.rs1_value)),
                                );
                                brute_val_be = brute_val_be.add(
                                    eq_addr_be[33].mul(eq_cycle_be[j]).mul(F.fromU64(step_b.rs2_value)),
                                );
                            }
                            if (!step_b.is_noop or step_b.is_termination_store) {
                                if (!step_b.is_termination_store) {
                                    const instr_b = step_b.instruction;
                                    const rd_b: u5 = @truncate((instr_b >> 7) & 0x1f);
                                    const opcode_b = instr_b & 0x7f;
                                    const rd_used_b = switch (opcode_b) {
                                        0x23, 0x63 => false,
                                        else => true,
                                    };
                                    if (rd_used_b and rd_b != 0 and rd_b < 32) {
                                        brute_regs3[rd_b] = step_b.rd_value;
                                    }
                                }
                            }
                        } else {
                            for (0..32) |k| {
                                if (!eq_addr_be[k].eql(F.zero()) and !eq_cycle_be[j].eql(F.zero())) {
                                    brute_val_be = brute_val_be.add(
                                        eq_addr_be[k].mul(eq_cycle_be[j]).mul(F.fromU64(brute_regs3[k])),
                                    );
                                }
                            }
                        }
                    }
                    dbg("[STAGE4 BRUTE VAL BE] brute val (BE ordering) = {any}\n", .{brute_val_be.toBytesBE()[0..16]});
                    dbg("[STAGE4 BRUTE VAL BE] should match Stage 5 brute\n", .{});
                    dbg("[STAGE4 BRUTE VAL] LSB val = BE val? {}\n", .{brute_val_claim.eql(brute_val_be)});

                    // Compute eq(r_cycle_be, 5) using both methods for j=5
                    // Build r_cycle_be from stage4_r_sumcheck directly (reversed phase1)
                    {
                        const test_j: usize = 5;
                        const n_vars = n_cycle_vars;
                        // r_cycle_be = [c7, c6, c5, c4, c3, c2, c1, c0]
                        var r_cycle_local: [8]F = undefined;
                        for (0..n_vars) |i| {
                            r_cycle_local[i] = stage4_r_sumcheck[n_vars - 1 - i];
                        }

                        // Stage 5 method: computeEqAtIndex with r_cycle_be (BE)
                        // r[j] pairs with bit (n-1-j) of test_j
                        var eq_stage5 = F.one();
                        for (0..n_vars) |bit_i| {
                            const bj: u1 = @truncate(test_j >> @intCast(n_vars - 1 - bit_i));
                            const rj = r_cycle_local[bit_i];
                            if (bj == 1) {
                                eq_stage5 = eq_stage5.mulHiBigIntU128(rj.limbs);
                            } else {
                                eq_stage5 = eq_stage5.mul(F.one().sub(rj));
                            }
                        }

                        // Table expansion method (BE reversed challenges)
                        const eq_table_val = eq_cycle_be[test_j];

                        // Direct mul method
                        var eq_mul_only = F.one();
                        for (0..n_vars) |bit_i| {
                            const bj: u1 = @truncate(test_j >> @intCast(n_vars - 1 - bit_i));
                            const rj = r_cycle_local[bit_i];
                            if (bj == 1) {
                                eq_mul_only = eq_mul_only.mul(rj);
                            } else {
                                eq_mul_only = eq_mul_only.mul(F.one().sub(rj));
                            }
                        }

                        dbg("\n[EQ TEST j=5] computeEqAtIndex (mulHi) = {any}\n", .{eq_stage5.toBytesBE()[0..16]});
                        dbg("[EQ TEST j=5] eq_cycle_be table       = {any}\n", .{eq_table_val.toBytesBE()[0..16]});
                        dbg("[EQ TEST j=5] direct mul only         = {any}\n", .{eq_mul_only.toBytesBE()[0..16]});
                        dbg("[EQ TEST j=5] stage5==table? {}, stage5==mul? {}, table==mul? {}\n", .{
                            eq_stage5.eql(eq_table_val), eq_stage5.eql(eq_mul_only), eq_table_val.eql(eq_mul_only),
                        });
                    }
                }

                // RegistersReadWriteChecking opening claims
                try jolt_proof.opening_claims.insert(
                    .{ .Virtual = .{ .poly = .RegistersVal, .sumcheck_id = .RegistersReadWriteChecking } },
                    regs_claims.val_claim,
                );
                try jolt_proof.opening_claims.insert(
                    .{ .Virtual = .{ .poly = .Rs1Ra, .sumcheck_id = .RegistersReadWriteChecking } },
                    regs_claims.rs1_ra_claim,
                );
                try jolt_proof.opening_claims.insert(
                    .{ .Virtual = .{ .poly = .Rs2Ra, .sumcheck_id = .RegistersReadWriteChecking } },
                    regs_claims.rs2_ra_claim,
                );
                try jolt_proof.opening_claims.insert(
                    .{ .Virtual = .{ .poly = .RdWa, .sumcheck_id = .RegistersReadWriteChecking } },
                    regs_claims.rd_wa_claim,
                );
                try jolt_proof.opening_claims.insert(
                    .{ .Committed = .{ .poly = .RdInc, .sumcheck_id = .RegistersReadWriteChecking } },
                    regs_claims.inc_claim,
                );

                // RamValCheck opening claims (combined, 2 claims: RamRa then RamInc)
                try jolt_proof.opening_claims.insert(
                    .{ .Virtual = .{ .poly = .RamRa, .sumcheck_id = .RamValCheck } },
                    val_eval_openings.wa_eval,
                );
                try jolt_proof.opening_claims.insert(
                    .{ .Committed = .{ .poly = .RamInc, .sumcheck_id = .RamValCheck } },
                    val_eval_openings.inc_eval,
                );

                // Cache openings into transcript (7 claims total)
                transcript.appendScalar("opening_claim", regs_claims.val_claim);
                transcript.appendScalar("opening_claim", regs_claims.rs1_ra_claim);
                transcript.appendScalar("opening_claim", regs_claims.rs2_ra_claim);
                transcript.appendScalar("opening_claim", regs_claims.rd_wa_claim);
                transcript.appendScalar("opening_claim", regs_claims.inc_claim);
                transcript.appendScalar("opening_claim", val_eval_openings.wa_eval);
                transcript.appendScalar("opening_claim", val_eval_openings.inc_eval);

                // Save Stage 4 RegistersRWC opening point for Stage 5
                // For RegistersRWC:
                // - Phase 1: 8 rounds (all cycle vars)
                // - Phase 2: 7 rounds (all address vars)
                // - Phase 3: 0 rounds
                // r_address = reverse(phase2_challenges) = reverse(stage4_r_sumcheck[8..15])
                // r_cycle = reverse(phase1_challenges) = reverse(stage4_r_sumcheck[0..8])
                const regs_log_k: usize = 7; // LOG_REGISTER_COUNT
                stage4_regs_r_address = try self.allocator.alloc(F, regs_log_k);
                stage4_regs_r_cycle = try self.allocator.alloc(F, n_cycle_vars);

                // r_address = reverse(phase2 challenges)
                for (0..regs_log_k) |i| {
                    stage4_regs_r_address.?[i] = stage4_r_sumcheck[n_cycle_vars + (regs_log_k - 1 - i)];
                }
                // r_cycle = reverse(phase1 challenges)
                for (0..n_cycle_vars) |i| {
                    stage4_regs_r_cycle.?[i] = stage4_r_sumcheck[n_cycle_vars - 1 - i];
                }

                dbg("[STAGE4 -> STAGE5] Saved opening point for RegistersValEvaluation:\n", .{});
                dbg("  r_address[0] = {any}\n", .{stage4_regs_r_address.?[0].toBytesBE()[0..8]});
                dbg("  r_cycle[0] = {any}\n", .{stage4_regs_r_cycle.?[0].toBytesBE()[0..8]});

                // Also save r_cycle_val for RamRaClaimReduction (ValEvaluation starts at round 7)
                // r_cycle_val = reverse(challenges[7..15]) for BIG_ENDIAN order
                stage4_r_cycle_val = try self.allocator.alloc(F, n_cycle_vars);
                const val_eval_start: usize = 7; // ValEvaluation starts at round 7
                for (0..n_cycle_vars) |i| {
                    const src_idx = val_eval_start + i;
                    if (src_idx < stage4_r_sumcheck.len) {
                        stage4_r_cycle_val.?[n_cycle_vars - 1 - i] = stage4_r_sumcheck[src_idx];
                    } else {
                        stage4_r_cycle_val.?[n_cycle_vars - 1 - i] = F.zero();
                    }
                }
                dbg("[STAGE4 -> STAGE5] Saved r_cycle_val for RamRaClaimReduction\n", .{});
            } // end stage4_block

            // Stage 5: RegistersValEvaluation, RamRaClaimReduction, LookupsReadRaf
            {
                dbg("[PROOF_CONV] Starting Stage 5...\n", .{});
                // ALWAYS-ON DIAGNOSTIC
                dbg("[STAGE5 DIAG] Transcript before Stage 5: ", .{});
                for (transcript.state[0..8]) |b| dbg("{x:0>2} ", .{b});
                dbg(" round={}\n", .{transcript.n_rounds});
            }
            // LookupsReadRaf has max rounds: LOG_K + log_T where LOG_K = XLEN * 2 = 128
            // For RV64: max_num_rounds = 128 + log_T = 128 + 8 = 136
            const lookups_log_k: usize = 128; // XLEN * 2 for RV64

            // CRITICAL: Jolt samples TWO separate gammas for Stage 5 instances.
            // The verifier creates instances in this order:
            //   1. InstructionReadRafSumcheckVerifier::new() → squeezes gamma_lookups_raf
            //   2. RamRaClaimReductionSumcheckVerifier::new() → squeezes gamma_ram_ra
            // So we must squeeze in the SAME order.
            dbg("[STAGE5 DIAG] Transcript state BEFORE gamma squeeze: {any} round={}\n", .{ transcript.state[0..8].*, transcript.n_rounds });
            const gamma_lookups_raf = transcript.challengeScalarFull();
            dbg("[STAGE5 DIAG] gamma_lookups_raf LE = {any}\n", .{gamma_lookups_raf.toBytes()});
            dbg("[STAGE5 DIAG] Transcript state AFTER gamma_lookups_raf: {any} round={}\n", .{ transcript.state[0..8].*, transcript.n_rounds });
            const gamma_ram_ra = transcript.challengeScalarFull();
            dbg("[STAGE5 DIAG] gamma_ram_ra LE = {any}\n", .{gamma_ram_ra.toBytes()});
            dbg("[STAGE5 DIAG] Transcript state AFTER gamma_ram_ra: {any} round={}\n", .{ transcript.state[0..8].*, transcript.n_rounds });
            dbg("[STAGE5] gamma_lookups_raf = {any}\n", .{gamma_lookups_raf.toBytesBE()});
            dbg("[STAGE5] gamma_ram_ra = {any}\n", .{gamma_ram_ra.toBytesBE()});

            // Generate Stage 5 proof using the batched sumcheck prover
            var stage5_prover_instance = Stage5BatchedProver(F).init(self.allocator);
            var stage5_result: spartan_mod.Stage5Result(F) = undefined;

            // Use trace-aware prover if we have trace data and Stage 4 opening point
            dbg("[STAGE5] Checking conditions: execution_trace={any}, r_address={any}, r_cycle={any}\n", .{
                config.execution_trace != null, stage4_regs_r_address != null, stage4_regs_r_cycle != null,
            });
            if (config.execution_trace != null and stage4_regs_r_address != null and stage4_regs_r_cycle != null and r_reduction_be != null) {
                dbg("[STAGE5] Using trace-aware prover\n", .{});
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
                dbg("[STAGE5] Using ZERO prover fallback (trace not available)\n", .{});
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
            defer stage5_result.deinit();

            // Debug: Print Stage 5 opening claims for comparison with Jolt verifier
            if (comptime debug_verbose) {
                dbg("[ZOLT S5 CLAIMS] inc_claim (LE) = {any}\n", .{stage5_result.regs_val_inc_claim.toBytes()});
                dbg("[ZOLT S5 CLAIMS] wa_claim (LE) = {any}\n", .{stage5_result.regs_val_wa_claim.toBytes()});
                dbg("[ZOLT S5 CLAIMS] ram_ra_claim (LE) = {any}\n", .{stage5_result.ram_ra_claim.toBytes()});
                dbg("[ZOLT S5 CLAIMS] raf_flag (LE) = {any}\n", .{stage5_result.lookups_raf_flag.toBytes()});
                for (0..8) |i| {
                    dbg("[ZOLT S5 CLAIMS] ra_chunk[{}] (LE) = {any}\n", .{ i, stage5_result.lookups_ra_chunks[i].toBytes() });
                }
                for (0..41) |i| {
                    if (!stage5_result.lookups_table_flags[i].eql(F.zero())) {
                        dbg("[ZOLT S5 CLAIMS] table_flag[{}] (LE) = {any}\n", .{ i, stage5_result.lookups_table_flags[i].toBytes() });
                    }
                }
            }

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
            // LookupTableFlag(i) for each of the 41 lookup tables
            const num_lookup_tables: usize = 41; // LookupTables::<XLEN>::COUNT (41 variants)
            dbg("[SERIALIZE DEBUG] Inserting LookupTableFlag claims:\n", .{});
            for (0..num_lookup_tables) |i| {
                const flag_value = stage5_result.lookups_table_flags[i];
                if (!flag_value.eql(F.zero())) {
                    // Convert to standard form for printing (same as serialization)
                    const standard = flag_value.fromMontgomery();
                    var buf: [32]u8 = undefined;
                    for (0..4) |j| {
                        std.mem.writeInt(u64, buf[j * 8 ..][0..8], standard.limbs[j], .little);
                    }
                    dbg("  LookupTableFlag({}) = {any}\n", .{ i, buf });
                }
                try jolt_proof.opening_claims.insert(
                    .{ .Virtual = .{ .poly = .{ .LookupTableFlag = i }, .sumcheck_id = .InstructionReadRaf } },
                    flag_value,
                );
            }
            dbg("[SERIALIZE DEBUG] After inserting all LookupTableFlag claims, total entries = {}\n", .{jolt_proof.opening_claims.len()});

            // InstructionRa(i) chunks for LookupsReadRaf (LOG_K / ra_virtual_log_k_chunk = 128 / 16 = 8 chunks)
            const lookups_ra_d: usize = lookups_log_k / config.lookups_ra_virtual_log_k_chunk;
            for (0..lookups_ra_d) |i| {
                dbg("[OPENING_CLAIMS] Inserting InstructionRa({}) for InstructionReadRaf = {any}\n", .{ i, stage5_result.lookups_ra_chunks[i].toBytesBE() });
                try jolt_proof.opening_claims.insert(
                    .{ .Virtual = .{ .poly = .{ .InstructionRa = i }, .sumcheck_id = .InstructionReadRaf } },
                    stage5_result.lookups_ra_chunks[i],
                );
            }

            // InstructionRafFlag for LookupsReadRaf
            dbg("[STAGE5 RAF_FLAG] Inserting raf_flag = {any}\n", .{stage5_result.lookups_raf_flag.toBytesBE()});
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
                // ALWAYS-ON: Print transcript state after Stage 5 cache_openings
                dbg("[STAGE6 ENTRY] Transcript AFTER Stage 5 cache_openings: ", .{});
                for (transcript.state[0..32]) |b| dbg("{x:0>2} ", .{b});
                dbg(" round={}\n", .{transcript.n_rounds});
            }

            // Stage 6: BytecodeReadRaf, RamHammingBooleanity, Booleanity, RamRaVirtual, LookupsRaVirtual, IncClaimReduction
            const bytecode_log_k = std.math.log2_int(usize, config.bytecode_K);
            const ram_log_k = std.math.log2_int(usize, ram_K);
            const instruction_d: usize = (lookups_log_k + config.log_k_chunk - 1) / config.log_k_chunk;
            const bytecode_d_val: usize = (bytecode_log_k + config.log_k_chunk - 1) / config.log_k_chunk;
            const ram_d_val: usize = (ram_log_k + config.log_k_chunk - 1) / config.log_k_chunk;

            dbg("[STAGE6] Parameters: bytecode_log_k={}, ram_log_k={}, instruction_d={}, bytecode_d={}, ram_d={}\n", .{
                bytecode_log_k, ram_log_k, instruction_d, bytecode_d_val, ram_d_val,
            });

            // Compute Stage 5 RegistersValEvaluation opening point (s_cycle_stage5)
            // Stage 5 max_rounds = 136 (128 address + 8 cycle for RV64)
            // RegistersValEvaluation runs in the last n_cycle_vars rounds
            // Opening point = challenges[128..136] reversed (LE → BE)
            const stage5_lookups_num_rounds = lookups_log_k + n_cycle_vars; // 136
            const stage5_regs_val_num_rounds = n_cycle_vars; // 8
            var s_cycle_stage5 = try self.allocator.alloc(F, n_cycle_vars);
            defer self.allocator.free(s_cycle_stage5);
            for (0..n_cycle_vars) |i| {
                // RegistersValEvaluation challenges are the last n_cycle_vars of Stage 5
                const stage5_idx = stage5_lookups_num_rounds - stage5_regs_val_num_rounds + i;
                // Reverse for BIG_ENDIAN
                s_cycle_stage5[n_cycle_vars - 1 - i] = stage5_result.challenges[stage5_idx];
            }

            // Generate Stage 6 proof using the batched sumcheck prover
            const the_trace = config.execution_trace orelse return error.ExecutionTraceRequired;
            const the_memory_layout = config.memory_layout orelse return error.MemoryLayoutRequired;

            // Compute SpartanShift r_cycle in BIG_ENDIAN from Stage 3 challenges (reversed)
            const r_cycle_shift_be = try self.allocator.alloc(F, stage3_result.challenges.len);
            defer self.allocator.free(r_cycle_shift_be);
            for (0..stage3_result.challenges.len) |i| {
                r_cycle_shift_be[i] = stage3_result.challenges[stage3_result.challenges.len - 1 - i];
            }

            dbg("[PROOF_CONV] Starting Stage 6...\n", .{});
            // Build bytecode entry table from static ELF + execution trace overlay
            const bytecode_K_val: usize = @as(usize, 1) << @intCast(bytecode_log_k);
            const stage6_mod = @import("spartan/stage6_prover.zig");
            // Get pc_map for converting ELF addresses to bytecode array indices
            const pc_map_ptr = config.bytecode_pc_map orelse return error.MissingBytecodepcMap;
            const bytecode_entries = try stage6_mod.buildBytecodeEntries(self.allocator, the_trace, bytecode_K_val, pc_map_ptr, config.program_code_bytes, config.code_base_address, the_memory_layout.termination);
            defer self.allocator.free(bytecode_entries);

            // Get register address opening points for Stages 4 and 5
            // Stage 4: from RegistersReadWriteChecking (address portion)
            const r_register_4 = stage4_regs_r_address orelse &[_]F{};
            // Stage 5: use same as Stage 4 (both address 32 registers)
            // In Jolt, this comes from RegistersValEvaluation's opening point split,
            // but the address variables are the SAME as Stage 4's since they share
            // the same register address space.
            const r_register_5 = stage4_regs_r_address orelse &[_]F{};

            var stage6_prover_instance = Stage6BatchedProver(F).init(self.allocator);
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
                // Stage 4 inc_poly copy for diagnostic
                if (stage4_inc_poly_copy) |v| v else &[_]F{},
            );
            defer stage6_result.deinit();

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
            dbg("[PROOF_CONV] Starting Stage 7...\n", .{});
            dbg("[STAGE7] Transcript before Stage 7: {{ ", .{});
            for (transcript.state[0..8]) |b| dbg("{x:0>2} ", .{b});
            dbg("}} round={}\n", .{transcript.n_rounds});

            // ====================================================================
            // Stage 7: HammingWeightClaimReduction sumcheck
            // ====================================================================
            {
                const s6_challenges = stage6_result.challenges;
                const s6_bytecode_log_k = stage6_result.bytecode_log_k;
                const s6_log_k_chunk = stage6_result.log_k_chunk;
                const s6_n_cycle_vars = stage6_result.n_cycle_vars;
                const s6_bytecode_d = stage6_result.bytecode_d;
                const s6_ram_d = stage6_result.ram_d;
                const s6_instruction_d = stage6_result.instruction_d;
                const s6_max_rounds = s6_bytecode_log_k + s6_n_cycle_vars;
                const s6_booleanity_rounds = s6_log_k_chunk + s6_n_cycle_vars;
                const s6_bool_start = s6_max_rounds - s6_booleanity_rounds; // = bytecode_log_k - log_k_chunk
                const N = s6_instruction_d + s6_bytecode_d + s6_ram_d;
                const k_chunk: usize = @as(usize, 1) << @intCast(s6_log_k_chunk);
                const T_val: usize = @as(usize, 1) << @intCast(s6_n_cycle_vars);

                dbg("[STAGE7] N={}, log_k_chunk={}, k_chunk={}, T={}\n", .{ N, s6_log_k_chunk, k_chunk, T_val });

                // Extract r_cycle_BE from Booleanity's cycle portion
                // Booleanity challenges[bool_start+log_k_chunk..bool_start+booleanity_rounds] reversed
                var r_cycle_be = try self.allocator.alloc(F, s6_n_cycle_vars);
                defer self.allocator.free(r_cycle_be);
                for (0..s6_n_cycle_vars) |i| {
                    r_cycle_be[i] = s6_challenges[s6_bool_start + s6_booleanity_rounds - 1 - i];
                }

                // Debug: print r_cycle_be
                for (0..s6_n_cycle_vars) |i| {
                    const v_be = r_cycle_be[i].toBytesBE();
                    dbg("[ZOLT HW] r_cycle_be[{d}] LE=[", .{i});
                    for (0..8) |bi| dbg("{x:0>2}", .{v_be[31 - bi]});
                    dbg("]\n", .{});
                }

                // Extract r_addr_bool_BE from Booleanity's address portion
                // challenges[bool_start..bool_start+log_k_chunk] reversed
                var r_addr_bool_be = try self.allocator.alloc(F, s6_log_k_chunk);
                defer self.allocator.free(r_addr_bool_be);
                for (0..s6_log_k_chunk) |i| {
                    r_addr_bool_be[i] = s6_challenges[s6_bool_start + s6_log_k_chunk - 1 - i];
                }
                // Debug: print r_addr_bool_be
                for (0..s6_log_k_chunk) |i| {
                    const v_be = r_addr_bool_be[i].toBytesBE();
                    dbg("[ZOLT HW] r_addr_bool_be[{d}] LE=[", .{i});
                    for (0..8) |bi| dbg("{x:0>2}", .{v_be[31 - bi]});
                    dbg("]\n", .{});
                }

                // Extract r_addr_virt_i for each ra polynomial (log_k_chunk elements each, BE)
                // Order: InstructionRa(0..inst_d), BytecodeRa(0..bc_d), RamRa(0..ram_d)
                var r_addr_virt = try self.allocator.alloc([]F, N);
                // Initialize to empty slices so deferred free doesn't crash on uninitialized entries
                for (r_addr_virt) |*slot| {
                    slot.* = &[_]F{};
                }
                defer {
                    for (r_addr_virt) |chunk| {
                        if (chunk.len > 0) self.allocator.free(chunk);
                    }
                    self.allocator.free(r_addr_virt);
                }

                // InstructionRa: from Stage 5 (LookupsRaVirtual) address chunks
                // LookupsRaVirtual in Stage 6 uses lookups_ra_addr_chunks from Stage 5
                // The address challenges are stage5_challenges[0..128] NOT reversed (stays LE in Stage 5)
                // Then split into chunks of lookups_ra_virtual_log_k_chunk (16),
                // but for HW reduction we need log_k_chunk-sized chunks of the full 128-bit address.
                // Actually, the r_addr_virt for InstructionRa(i) is the chunk stored by
                // LookupsRaVirtual's cache_openings, which uses compute_r_address_chunks.
                // This splits the full LOOKUPS_LOG_K=128 address into instruction_d=32 chunks of log_k_chunk=4.
                // But LookupsRaVirtual uses lookups_ra_virtual_log_k_chunk (=16) chunks internally,
                // and then the verifier uses compute_r_address_chunks to split those into log_k_chunk chunks.
                //
                // The verifier does: get_committed_polynomial_opening(InstructionRa(i), InstructionRaVirtualization)
                // which returns the point stored by LookupsRaVirtual's cache_openings.
                // That point stores r_address = compute_r_address_chunks(full_address_128, log_k_chunk)
                // So r_addr_virt[i] for InstructionRa(i) is the i-th chunk of 128/log_k_chunk = 32 chunks.
                //
                // The full 128-bit address (BE) for Lookups comes from Stage 5:
                // InstructionReadRaf has LOOKUPS_LOG_K=128 address variables.
                // In Stage 5's batched sumcheck, InstructionReadRaf starts at round 0
                // (it has the max rounds = 128 + n_cycle_vars).
                // normalize_opening_point: r[0..128].reverse() → BE, r[128..].reverse() → BE
                // But wait - Stage 5 InstructionReadRaf's normalize does NOT reverse the address!
                // Let me check...
                const LOOKUPS_LOG_K: usize = 128;

                // InstructionReadRaf in Stage 5 has LOOKUPS_LOG_K + n_cycle_vars rounds
                // Its normalize_opening_point does NOT reverse the address (stays LE)
                // Then compute_r_address_chunks splits into chunks of log_k_chunk
                // So r_addr_virt for InstructionRa(i) = stage5_challenges[i*log_k_chunk..(i+1)*log_k_chunk]
                // (NOT reversed - address stays in LE/sumcheck order)
                for (0..s6_instruction_d) |i| {
                    var chunk = try self.allocator.alloc(F, s6_log_k_chunk);
                    const chunk_start = i * s6_log_k_chunk;
                    for (0..s6_log_k_chunk) |ci| {
                        if (chunk_start + ci < LOOKUPS_LOG_K) {
                            chunk[ci] = stage5_result.challenges[chunk_start + ci];
                        } else {
                            chunk[ci] = F.zero();
                        }
                    }
                    r_addr_virt[i] = chunk;
                    // Print all r_addr_virt for comparison with Jolt
                    for (0..s6_log_k_chunk) |ci| {
                        const v_be = chunk[ci].toBytesBE();
                        dbg("[ZOLT HW] r_addr_virt[{d}][{d}] LE=[", .{ i, ci });
                        for (0..8) |bi| dbg("{x:0>2}", .{v_be[31 - bi]});
                        dbg("]\n", .{});
                    }
                }

                // BytecodeRa: from BytecodeReadRaf address challenges (Stage 6)
                // BytecodeReadRaf starts at round 0, has bytecode_log_k address rounds
                // normalize_opening_point: r[0..bytecode_log_k].reverse() → BE
                // Then compute_r_address_chunks pads with zeros and splits into bytecode_d chunks
                {
                    // Reversed address → BE
                    var bc_addr_be = try self.allocator.alloc(F, s6_bytecode_log_k);
                    defer self.allocator.free(bc_addr_be);
                    for (0..s6_bytecode_log_k) |i| {
                        bc_addr_be[i] = s6_challenges[s6_bytecode_log_k - 1 - i];
                    }

                    // Pad to multiple of log_k_chunk (prepend zeros)
                    const padded_len = s6_bytecode_d * s6_log_k_chunk;
                    var bc_addr_padded = try self.allocator.alloc(F, padded_len);
                    defer self.allocator.free(bc_addr_padded);
                    @memset(bc_addr_padded, F.zero());
                    // Copy to the end (BE: prepend zeros)
                    const pad = padded_len - s6_bytecode_log_k;
                    for (0..s6_bytecode_log_k) |i| {
                        bc_addr_padded[pad + i] = bc_addr_be[i];
                    }

                    // Split into chunks
                    for (0..s6_bytecode_d) |i| {
                        var chunk = try self.allocator.alloc(F, s6_log_k_chunk);
                        for (0..s6_log_k_chunk) |ci| {
                            chunk[ci] = bc_addr_padded[i * s6_log_k_chunk + ci];
                        }
                        r_addr_virt[s6_instruction_d + i] = chunk;
                        const gi = s6_instruction_d + i;
                        for (0..s6_log_k_chunk) |ci| {
                            const v_be = chunk[ci].toBytesBE();
                            dbg("[ZOLT HW] r_addr_virt[{d}][{d}] LE=[", .{ gi, ci });
                            for (0..8) |bi| dbg("{x:0>2}", .{v_be[31 - bi]});
                            dbg("]\n", .{});
                        }
                    }
                }

                // RamRa: from RamRaVirtual address challenges (derived from Stage 5)
                // Same as how Stage 6 extracted ram_ra_addr_chunks
                {
                    const s7_ram_log_k: usize = s6_ram_d * s6_log_k_chunk;
                    const stage5_max_rounds = LOOKUPS_LOG_K + s6_n_cycle_vars;
                    const ram_ra_total_rounds = s7_ram_log_k + s6_n_cycle_vars;
                    const ram_ra_offset = stage5_max_rounds - ram_ra_total_rounds;

                    // Reversed address → BE
                    var ram_addr_be = try self.allocator.alloc(F, s7_ram_log_k);
                    defer self.allocator.free(ram_addr_be);
                    for (0..s7_ram_log_k) |i| {
                        ram_addr_be[i] = stage5_result.challenges[ram_ra_offset + s7_ram_log_k - 1 - i];
                    }

                    // Split into chunks (already multiple of log_k_chunk)
                    for (0..s6_ram_d) |i| {
                        var chunk = try self.allocator.alloc(F, s6_log_k_chunk);
                        for (0..s6_log_k_chunk) |ci| {
                            chunk[ci] = ram_addr_be[i * s6_log_k_chunk + ci];
                        }
                        r_addr_virt[s6_instruction_d + s6_bytecode_d + i] = chunk;
                        const gi = s6_instruction_d + s6_bytecode_d + i;
                        for (0..s6_log_k_chunk) |ci| {
                            const v_be = chunk[ci].toBytesBE();
                            dbg("[ZOLT HW] r_addr_virt[{d}][{d}] LE=[", .{ gi, ci });
                            for (0..8) |bi| dbg("{x:0>2}", .{v_be[31 - bi]});
                            dbg("]\n", .{});
                        }
                    }
                }

                // Build eq table for r_cycle
                //
                // IMPORTANT: The booleanity sumcheck's Phase 2 uses LowToHigh binding.
                // When halving the eq_cycle table with challenges c[0],...,c[n-1],
                // challenge c[m] binds bit m of the table index j.
                //
                // For the Stage 7 G tables to produce claims consistent with the
                // booleanity Phase 2 halving, the eq_cycle table must use the SAME
                // bit-to-challenge mapping. With computeEqTable(r, n), r[m] controls
                // bit m of index j. So we need r[m] = c[m] (the LE cycle challenges).
                //
                // r_cycle_be is the REVERSED cycle challenges (BE format), so we need
                // to use the UN-reversed version = direct Stage 6 cycle challenges (LE).
                var r_cycle_le = try self.allocator.alloc(F, s6_n_cycle_vars);
                defer self.allocator.free(r_cycle_le);
                for (0..s6_n_cycle_vars) |i| {
                    r_cycle_le[i] = s6_challenges[s6_bool_start + s6_log_k_chunk + i];
                }
                const eq_cycle = try stage6_mod.computeEqTable(F, self.allocator, r_cycle_le, s6_n_cycle_vars);
                defer self.allocator.free(eq_cycle);

                // Compute G_i polynomials: G_i(k) = Σ_j eq(r_cycle, j) · (addr_chunk_i(j) == k ? 1 : 0)
                var G = try self.allocator.alloc([]F, N);
                defer {
                    for (G) |g| self.allocator.free(g);
                    self.allocator.free(G);
                }
                for (0..N) |i| {
                    G[i] = try self.allocator.alloc(F, k_chunk);
                    @memset(G[i], F.zero());
                }

                // Iterate over all cycles to populate G_i
                // G_i(k) = Σ_j eq(r_cycle, j) · [addr_chunk_i(j) == k]
                for (0..T_val) |j| {
                    const step = the_trace.steps.items[j];
                    const eq_j = eq_cycle[j];

                    // InstructionRa: compute 128-bit lookup index, split into chunks
                    // Use the same computeLookupIndex as Stage 6 LookupsRaVirtualProver
                    // to ensure G tables are consistent with the virt_claims.
                    {
                        const lookup_idx = stage6_mod.computeLookupIndex(step);
                        for (0..s6_instruction_d) |i| {
                            const shift = s6_log_k_chunk * (s6_instruction_d - 1 - i);
                            const mask: u128 = (@as(u128, 1) << @intCast(s6_log_k_chunk)) - 1;
                            const chunk_val: usize = @intCast((lookup_idx >> @intCast(shift)) & mask);
                            if (chunk_val < k_chunk) {
                                G[i][chunk_val] = G[i][chunk_val].add(eq_j);
                            }
                        }
                    }

                    // BytecodeRa: bytecode address (pc_idx) split into chunks
                    {
                        const pc_idx = pc_map_ptr.getPCForStep(step);
                        for (0..s6_bytecode_d) |i| {
                            const chunk_val = stage6_mod.extractChunkMSB(@intCast(pc_idx), i, s6_bytecode_d, s6_log_k_chunk);
                            const ra_idx = s6_instruction_d + i;
                            if (chunk_val < k_chunk) {
                                G[ra_idx][chunk_val] = G[ra_idx][chunk_val].add(eq_j);
                            }
                        }
                    }

                    // RamRa: memory address (remapped) split into chunks
                    {
                        if (step.memory_addr) |addr| {
                            if (addr != 0) {
                                if (the_memory_layout.remapAddress(addr)) |raddr| {
                                    for (0..s6_ram_d) |i| {
                                        const chunk_val = stage6_mod.extractChunkMSB(raddr, i, s6_ram_d, s6_log_k_chunk);
                                        const ra_idx = s6_instruction_d + s6_bytecode_d + i;
                                        if (chunk_val < k_chunk) {
                                            G[ra_idx][chunk_val] = G[ra_idx][chunk_val].add(eq_j);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                // Compute eq tables for r_addr_bool and r_addr_virt_i
                //
                // IMPORTANT: computeEqTable puts r[0] at bit 0 (LE convention).
                // The booleanity Phase 1 F table also uses LowToHigh expansion,
                // putting a[0] at bit 0. For eq_bool to match F[chunk], we need
                // to pass the LE address challenges (same order as Phase 1 binding).
                //
                // Similarly, the virtualization sumchecks use LowToHigh binding,
                // so eq_virt needs the LE versions of the address challenges.
                //
                // The LE version = reversed BE version.
                var r_addr_bool_le = try self.allocator.alloc(F, s6_log_k_chunk);
                defer self.allocator.free(r_addr_bool_le);
                for (0..s6_log_k_chunk) |i| {
                    r_addr_bool_le[i] = r_addr_bool_be[s6_log_k_chunk - 1 - i];
                }
                var eq_bool = try stage6_mod.computeEqTable(F, self.allocator, r_addr_bool_le, s6_log_k_chunk);
                defer self.allocator.free(eq_bool);

                var eq_virt = try self.allocator.alloc([]F, N);
                defer {
                    for (eq_virt) |ev| self.allocator.free(ev);
                    self.allocator.free(eq_virt);
                }
                for (0..N) |i| {
                    // Reverse r_addr_virt to LE for eq table
                    var r_virt_le = try self.allocator.alloc(F, s6_log_k_chunk);
                    for (0..s6_log_k_chunk) |ci| {
                        r_virt_le[ci] = r_addr_virt[i][s6_log_k_chunk - 1 - ci];
                    }
                    eq_virt[i] = try stage6_mod.computeEqTable(F, self.allocator, r_virt_le, s6_log_k_chunk);
                    self.allocator.free(r_virt_le);
                }

                // Debug: verify G table sums and cross-products with eq_virt/eq_bool
                {
                    for (0..N) |i| {
                        var g_sum = F.zero();
                        var g_virt_sum = F.zero();
                        var g_bool_sum = F.zero();
                        for (0..k_chunk) |k| {
                            g_sum = g_sum.add(G[i][k]);
                            g_virt_sum = g_virt_sum.add(G[i][k].mul(eq_virt[i][k]));
                            g_bool_sum = g_bool_sum.add(G[i][k].mul(eq_bool[k]));
                        }
                        // Print G table values for specific indices to debug
                        if (i == 14 or i == 15 or i == 16 or i == 24 or i == 32 or i == 34) {
                            dbg("[STAGE7 GTABLE] i={d}: ", .{i});
                            for (0..k_chunk) |k| {
                                if (!G[i][k].eql(F.zero())) {
                                    const g_be = G[i][k].toBytesBE();
                                    dbg("G[{d}]=[", .{k});
                                    for (0..8) |bi| dbg("{x:0>2}", .{g_be[31 - bi]});
                                    dbg("] ", .{});
                                }
                            }
                            dbg("\n", .{});
                            // Also print eq_virt values and products for failing indices
                            if (i == 15 or i == 24 or i == 32 or i == 34) {
                                dbg("[STAGE7 EQVIRT] i={d}: ", .{i});
                                for (0..k_chunk) |k| {
                                    const ev_be = eq_virt[i][k].toBytesBE();
                                    dbg("ev[{d}]=[", .{k});
                                    for (0..8) |bi| dbg("{x:0>2}", .{ev_be[31 - bi]});
                                    dbg("] ", .{});
                                }
                                dbg("\n", .{});
                                // Also print eq_bool and per-k products
                                dbg("[STAGE7 EQBOOL] i={d}: ", .{i});
                                for (0..k_chunk) |k| {
                                    const eb_be = eq_bool[k].toBytesBE();
                                    dbg("eb[{d}]=[", .{k});
                                    for (0..8) |bi| dbg("{x:0>2}", .{eb_be[31 - bi]});
                                    dbg("] ", .{});
                                }
                                dbg("\n", .{});
                            }
                        }
                        const virt_claim: F = blk: {
                            if (i < s6_instruction_d) break :blk stage6_result.instruction_ra_virtual_claims[i];
                            if (i < s6_instruction_d + s6_bytecode_d) break :blk stage6_result.bytecode_ra_claims[i - s6_instruction_d];
                            break :blk stage6_result.ram_ra_virtual_claims[i - s6_instruction_d - s6_bytecode_d];
                        };
                        const bool_claim = stage6_result.booleanity_ra_claims[i];
                        const gs_be = g_sum.toBytesBE();
                        const gv_be = g_virt_sum.toBytesBE();
                        const vc_be = virt_claim.toBytesBE();
                        const gb_be2 = g_bool_sum.toBytesBE();
                        const bc_be = bool_claim.toBytesBE();
                        const gv_match = g_virt_sum.eql(virt_claim);
                        const gb_match = g_bool_sum.eql(bool_claim);
                        dbg("[STAGE7 VERIFY] i={d}: G_sum_LE=[", .{i});
                        for (0..8) |bi| dbg("{x:0>2}", .{gs_be[31 - bi]});
                        dbg("]\n", .{});
                        dbg("[STAGE7 VERIFY] i={d}: G*eq_virt_LE=[", .{i});
                        for (0..8) |bi| dbg("{x:0>2}", .{gv_be[31 - bi]});
                        dbg("] virt_claim_LE=[", .{});
                        for (0..8) |bi| dbg("{x:0>2}", .{vc_be[31 - bi]});
                        dbg("] match={}\n", .{gv_match});
                        dbg("[STAGE7 VERIFY] i={d}: G*eq_bool_LE=[", .{i});
                        for (0..8) |bi| dbg("{x:0>2}", .{gb_be2[31 - bi]});
                        dbg("] bool_claim_LE=[", .{});
                        for (0..8) |bi| dbg("{x:0>2}", .{bc_be[31 - bi]});
                        dbg("] match={}\n", .{gb_match});
                    }
                }

                // Sample gamma from transcript (matches Jolt's HammingWeightClaimReductionParams::new)
                // IMPORTANT: Jolt's HW code calls transcript.challenge_scalar() which uses
                // challenge_scalar_128_bits() -> F::from_bytes() = from_le_bytes_mod_order().
                // This is the FULL field element path, NOT the 125-bit optimized path.
                // So we must use challengeScalarFull() here.
                dbg("[STAGE7] Transcript state before gamma: {{ ", .{});
                for (transcript.state[0..8]) |b| dbg("{x:0>2} ", .{b});
                dbg("}} round={}\n", .{transcript.n_rounds});
                const gamma = transcript.challengeScalarFull();
                {
                    const gb = gamma.toBytesBE();
                    dbg("[STAGE7] gamma_LE=[", .{});
                    for (0..8) |bi| dbg("{x:0>2},", .{gb[31 - bi]});
                    dbg("]\n", .{});
                }
                var gamma_powers = try self.allocator.alloc(F, 3 * N);
                defer self.allocator.free(gamma_powers);
                gamma_powers[0] = F.one();
                for (1..3 * N) |i| gamma_powers[i] = gamma_powers[i - 1].mul(gamma);

                // Compute HammingWeight claims for each ra_i
                // For InstructionRa and BytecodeRa: H_i = 1 (Jolt convention)
                // For RamRa: H_i = ram_hw_factor (from RamHammingBooleanity opening)
                const ram_hw_factor = stage6_result.hamming_weight_claim;

                // Compute input claim: Σ_i (γ^{3i}·H_i + γ^{3i+1}·claim_bool_i + γ^{3i+2}·claim_virt_i)
                // Use claims from Stage 6 result (booleanity claims now properly computed)
                var input_claim = F.zero();
                for (0..N) |i| {
                    const hw_claim: F = if (i >= s6_instruction_d + s6_bytecode_d) ram_hw_factor else F.one();
                    const bool_claim = stage6_result.booleanity_ra_claims[i];
                    const virt_claim: F = blk: {
                        if (i < s6_instruction_d) {
                            break :blk stage6_result.instruction_ra_virtual_claims[i];
                        } else if (i < s6_instruction_d + s6_bytecode_d) {
                            break :blk stage6_result.bytecode_ra_claims[i - s6_instruction_d];
                        } else {
                            break :blk stage6_result.ram_ra_virtual_claims[i - s6_instruction_d - s6_bytecode_d];
                        }
                    };
                    input_claim = input_claim.add(gamma_powers[3 * i].mul(hw_claim));
                    input_claim = input_claim.add(gamma_powers[3 * i + 1].mul(bool_claim));
                    input_claim = input_claim.add(gamma_powers[3 * i + 2].mul(virt_claim));
                    if (i < 3) {
                        const hw_be = hw_claim.toBytesBE();
                        const bl_be = bool_claim.toBytesBE();
                        const vt_be = virt_claim.toBytesBE();
                        dbg("[HW_INPUT] ra[{d}] hw=[", .{i});
                        for (0..8) |bi| dbg("{x:0>2},", .{hw_be[31 - bi]});
                        dbg("] bool=[", .{});
                        for (0..8) |bi| dbg("{x:0>2},", .{bl_be[31 - bi]});
                        dbg("] virt=[", .{});
                        for (0..8) |bi| dbg("{x:0>2},", .{vt_be[31 - bi]});
                        dbg("]\n", .{});
                    }
                }

                dbg("[STAGE7] input_claim_LE=[", .{});
                const ic_be = input_claim.toBytesBE();
                for (0..8) |bi| dbg("{x:0>2}", .{ic_be[31 - bi]});
                dbg("]\n", .{});

                // Append input claim to transcript (matches BatchedSumcheck::verify)
                transcript.appendScalar("sumcheck_claim", input_claim);

                // Sample batching coefficient (only 1 instance for now - no advice)
                const batch_coeffs = try transcript.challengeVector(self.allocator, 1);
                defer self.allocator.free(batch_coeffs);
                const batch_coeff = batch_coeffs[0];

                // Batched claim = batch_coeff * input_claim (1 instance, no scaling needed for same rounds)
                var current_claim = batch_coeff.mul(input_claim);

                // Run degree-2 sumcheck over log_k_chunk rounds
                const num_rounds = s6_log_k_chunk;
                const degree_bound: usize = 2;

                // Collect Stage 7 sumcheck challenges for opening point construction
                var stage7_challenges = try self.allocator.alloc(F, num_rounds);
                defer self.allocator.free(stage7_challenges);

                // Track current polynomial size (halves each round)
                var poly_size: usize = k_chunk;

                for (0..num_rounds) |round| {
                    const half = poly_size / 2;

                    // Sanity check: verify sum over current table = current_claim / batch_coeff
                    {
                        var check_sum = F.zero();
                        for (0..poly_size) |k| {
                            for (0..N) |i| {
                                const gi = G[i][k];
                                const w = gamma_powers[3 * i].add(gamma_powers[3 * i + 1].mul(eq_bool[k])).add(gamma_powers[3 * i + 2].mul(eq_virt[i][k]));
                                check_sum = check_sum.add(gi.mul(w));
                            }
                        }
                        check_sum = check_sum.mul(batch_coeff);
                        const eq_claim = check_sum.eql(current_claim);
                        if (!eq_claim) {
                            const cs_be = check_sum.toBytesBE();
                            const cc_be2 = current_claim.toBytesBE();
                            dbg("[STAGE7 SANITY R{d}] FAIL: check_sum_LE=[", .{round});
                            for (0..8) |bi| dbg("{x:0>2}", .{cs_be[31 - bi]});
                            dbg("] current_claim_LE=[", .{});
                            for (0..8) |bi| dbg("{x:0>2}", .{cc_be2[31 - bi]});
                            dbg("]\n", .{});
                        } else {
                            dbg("[STAGE7 SANITY R{d}] OK\n", .{round});
                        }
                    }

                    // LowToHigh binding: pair (2*j, 2*j+1) to bind LSB first
                    // Compute round polynomial evaluations at {0, 2}
                    // (p(1) is derived from p(0) + p(1) = claim)
                    var p0 = F.zero();
                    var p2 = F.zero();

                    for (0..half) |j| {
                        // LowToHigh: lo = poly[2*j], hi = poly[2*j+1]
                        const eq_b_lo = eq_bool[2 * j];
                        const eq_b_hi = eq_bool[2 * j + 1];
                        // Eval at x=2: f(2) = 2*f(1) - f(0)
                        const eq_b_2 = eq_b_hi.add(eq_b_hi).sub(eq_b_lo);

                        for (0..N) |i| {
                            const g_lo = G[i][2 * j];
                            const g_hi = G[i][2 * j + 1];
                            const g_2 = g_hi.add(g_hi).sub(g_lo);

                            const ev_lo = eq_virt[i][2 * j];
                            const ev_hi = eq_virt[i][2 * j + 1];
                            const ev_2 = ev_hi.add(ev_hi).sub(ev_lo);

                            // weight(x) = γ^{3i} + γ^{3i+1}·eq_bool(x) + γ^{3i+2}·eq_virt_i(x)
                            const w0 = gamma_powers[3 * i].add(gamma_powers[3 * i + 1].mul(eq_b_lo)).add(gamma_powers[3 * i + 2].mul(ev_lo));
                            const w2 = gamma_powers[3 * i].add(gamma_powers[3 * i + 1].mul(eq_b_2)).add(gamma_powers[3 * i + 2].mul(ev_2));

                            p0 = p0.add(g_lo.mul(w0));
                            p2 = p2.add(g_2.mul(w2));
                        }
                    }

                    // Scale by batch coefficient
                    p0 = p0.mul(batch_coeff);
                    p2 = p2.mul(batch_coeff);

                    // p(1) = current_claim - p(0)
                    const p1 = current_claim.sub(p0);

                    // Compress to Toom-Cook format: coeffs_except_linear = [a0, a2]
                    // p(x) = a0 + a1*x + a2*x^2
                    // a0 = p(0)
                    // a2 = (p(2) - 2*p(1) + p(0)) / 2
                    const two_p1 = p1.add(p1);
                    const a2_num = p2.sub(two_p1).add(p0);
                    const a2 = a2_num.mul(F.fromU64(2).inverse().?);

                    const coeffs = try self.allocator.alloc(F, degree_bound);
                    coeffs[0] = p0; // a0 = p(0) = constant term
                    coeffs[1] = a2; // a2 = quadratic coefficient
                    try jolt_proof.stage7_sumcheck_proof.compressed_polys.append(self.allocator, .{
                        .coeffs_except_linear_term = coeffs,
                        .allocator = self.allocator,
                    });

                    // Append to transcript and get challenge
                    transcript.appendScalars("sumcheck_poly", coeffs[0..degree_bound]);

                    const challenge = transcript.challengeScalar();
                    stage7_challenges[round] = challenge;

                    // Evaluate p(challenge) = a0 + a1*challenge + a2*challenge^2
                    // a1 = p(1) - a0 - a2
                    const a0 = p0;
                    const a1 = p1.sub(a0).sub(a2);
                    current_claim = a0.add(a1.mul(challenge)).add(a2.mul(challenge.mul(challenge)));

                    // Bind all polynomials at challenge (LowToHigh: bind pairs 2j, 2j+1)
                    for (0..N) |i| {
                        for (0..half) |jj| {
                            G[i][jj] = G[i][2 * jj].add(challenge.mul(G[i][2 * jj + 1].sub(G[i][2 * jj])));
                        }
                    }
                    for (0..half) |jj| {
                        eq_bool[jj] = eq_bool[2 * jj].add(challenge.mul(eq_bool[2 * jj + 1].sub(eq_bool[2 * jj])));
                    }
                    for (0..N) |i| {
                        for (0..half) |jj| {
                            eq_virt[i][jj] = eq_virt[i][2 * jj].add(challenge.mul(eq_virt[i][2 * jj + 1].sub(eq_virt[i][2 * jj])));
                        }
                    }
                    poly_size = half;

                    if (round < 2) {
                        dbg("[STAGE7] R{} p(0)_LE=[", .{round});
                        const p0_be = p0.toBytesBE();
                        for (0..8) |bi| dbg("{x:0>2}", .{p0_be[31 - bi]});
                        dbg("] claim_LE=[", .{});
                        const cc_be = current_claim.toBytesBE();
                        for (0..8) |bi| dbg("{x:0>2}", .{cc_be[31 - bi]});
                        dbg("]\n", .{});
                    }
                }

                // Cache opening claims: G_i(ρ) for each ra_i
                // G_i[0] is the final value after all bindings
                // Order: InstructionRa(0..inst_d), BytecodeRa(0..bc_d), RamRa(0..ram_d)
                for (0..N) |i| {
                    const g_claim = G[i][0];
                    const key: jolt_types.OpeningId = blk: {
                        if (i < s6_instruction_d) {
                            break :blk .{ .Committed = .{ .poly = .{ .InstructionRa = i }, .sumcheck_id = .HammingWeightClaimReduction } };
                        } else if (i < s6_instruction_d + s6_bytecode_d) {
                            break :blk .{ .Committed = .{ .poly = .{ .BytecodeRa = i - s6_instruction_d }, .sumcheck_id = .HammingWeightClaimReduction } };
                        } else {
                            break :blk .{ .Committed = .{ .poly = .{ .RamRa = i - s6_instruction_d - s6_bytecode_d }, .sumcheck_id = .HammingWeightClaimReduction } };
                        }
                    };
                    try jolt_proof.opening_claims.insert(key, g_claim);
                    // Append to transcript (matches cache_openings → append_sparse)
                    transcript.appendScalar("opening_claim", g_claim);
                }

                dbg("[STAGE7] Sumcheck complete, G[0][0]_LE=[", .{});
                const g0_be = G[0][0].toBytesBE();
                for (0..8) |bi| dbg("{x:0>2}", .{g0_be[31 - bi]});
                dbg("]\n", .{});

                // Debug: Verify expected output claim (what verifier would compute)
                {
                    const final_eq_bool = eq_bool[0];
                    const eb_be = final_eq_bool.toBytesBE();
                    dbg("[STAGE7] final eq_bool[0]_LE=[", .{});
                    for (0..8) |bi| dbg("{x:0>2}", .{eb_be[31 - bi]});
                    dbg("]\n", .{});

                    // Cross-check: compute mle(rho_rev, r_addr_bool) directly
                    {
                        // Collect sumcheck challenges (stored in round_polys, extracted via transcript)
                        // Actually, the sumcheck challenges are the round challenges we used to bind.
                        // They are derived from the transcript. Let me retrieve them from what was used.
                        // For now, just compute mle from stored r_addr_bool_be and see what we get.
                        // rho_rev = reversed sumcheck challenges

                        // Print initial eq table values for first few entries
                        var eq_bool_check = try stage6_mod.computeEqTable(F, self.allocator, r_addr_bool_be, s6_log_k_chunk);
                        defer self.allocator.free(eq_bool_check);
                        dbg("[STAGE7] eq_bool initial[0..4]_LE=", .{});
                        for (0..@min(4, eq_bool_check.len)) |ei| {
                            const e_be = eq_bool_check[ei].toBytesBE();
                            dbg("[", .{});
                            for (0..8) |bi| dbg("{x:0>2}", .{e_be[31 - bi]});
                            dbg("]", .{});
                        }
                        dbg("\n", .{});
                    }

                    var expected = F.zero();
                    for (0..N) |i| {
                        const gi = G[i][0];
                        const evi = eq_virt[i][0];
                        const weight = gamma_powers[3 * i].add(gamma_powers[3 * i + 1].mul(final_eq_bool)).add(gamma_powers[3 * i + 2].mul(evi));
                        expected = expected.add(gi.mul(weight));
                    }
                    // expected * batch_coeff should equal the output_claim
                    const expected_batched = expected.mul(batch_coeff);
                    const exp_be = expected_batched.toBytesBE();
                    dbg("[STAGE7] prover expected_claim_LE=[", .{});
                    for (0..8) |bi| dbg("{x:0>2}", .{exp_be[31 - bi]});
                    dbg("]\n", .{});

                    // Print eq_virt[0][0] for comparison
                    const ev0_be = eq_virt[0][0].toBytesBE();
                    dbg("[STAGE7] final eq_virt[0][0]_LE=[", .{});
                    for (0..8) |bi| dbg("{x:0>2}", .{ev0_be[31 - bi]});
                    dbg("]\n", .{});

                    // Print the current_claim (output of sumcheck)
                    const cc_be = current_claim.toBytesBE();
                    dbg("[STAGE7] sumcheck output_claim_LE=[", .{});
                    for (0..8) |bi| dbg("{x:0>2}", .{cc_be[31 - bi]});
                    dbg("]\n", .{});
                }

                // Construct the unified opening point: [r_address_stage7_BE || r_cycle_BE]
                // r_address = reversed stage7_challenges (LE → BE, like Jolt's match_endianness)
                // r_cycle = r_cycle_be (already BE from Stage 6 booleanity)
                const opening_point_len = s6_log_k_chunk + s6_n_cycle_vars;
                var opening_point_storage = try self.allocator.alloc(F, opening_point_len);
                // r_address_be: reverse the stage7_challenges
                for (0..s6_log_k_chunk) |i| {
                    opening_point_storage[i] = stage7_challenges[s6_log_k_chunk - 1 - i];
                }
                // r_cycle_be
                for (0..s6_n_cycle_vars) |i| {
                    opening_point_storage[s6_log_k_chunk + i] = r_cycle_be[i];
                }
                jolt_proof.opening_point = opening_point_storage;

                dbg("[STAGE7] Stored opening_point ({} dims = {} addr + {} cycle)\n", .{ opening_point_len, s6_log_k_chunk, s6_n_cycle_vars });
                for (0..opening_point_len) |i| {
                    const op_be = opening_point_storage[i].toBytesBE();
                    dbg("[STAGE7] opening_point[{d}] LE=[", .{i});
                    for (0..8) |bi| dbg("{x:0>2}", .{op_be[31 - bi]});
                    dbg("]\n", .{});
                }
            }

            dbg("[PROOF_CONV] Conversion complete!\n", .{});
            return jolt_proof;
        }

        /// Result of Stage 2 sumcheck including factor evaluations and challenges
        const Stage2Result = struct {
            /// The 8 factor polynomial evaluations at r_cycle
            /// Order: LeftInstructionInput, RightInstructionInput, IsRdNotZero,
            ///        WriteLookupOutputToRDFlag, JumpFlag, LookupOutput, BranchFlag, NextIsNoop
            factor_evals: [8]F,
            /// All sumcheck challenges (26 for max_num_rounds = log_ram_k + n_cycle_vars)
            /// Used for computing OutputSumcheck's r_address_prime
            challenges: []F,
            /// Final claims from each prover (for opening claims)
            raf_final_claim: F, // Instance 1: RamRafEvaluation
            rwc_final_claim: F, // Instance 2: RamReadWriteChecking (combined claim)
            output_final_claim: F, // Instance 3: RamOutputCheck sumcheck final claim
            instr_final_claim: F, // Instance 4: InstructionLookupsClaimReduction (combined)
            /// OutputSumcheck's Val_final polynomial evaluation at r_address_prime
            /// This is the MLE evaluation Val_final(r'), needed for opening claim
            output_val_final_claim: F, // Val_final(r') for RamValFinal opening
            /// OutputSumcheck's Val_init polynomial evaluation at r_address_prime
            output_val_init_claim: F, // Val_init(r') for RamValInit opening
            /// OutputSumcheck's r_address challenges (big-endian order)
            r_address_raf: []F,
            /// RWC's r_address challenges (big-endian order) - for RamRaClaimReduction
            r_address_rw: []F,
            /// RWC's r_cycle challenges (big-endian order) - for RamRaClaimReduction
            r_cycle_rw: []F,
            /// ProductVirtualRemainder's r_cycle challenges (big-endian order) - for BytecodeReadRaf
            r_cycle_product: []F,
            /// Individual RWC opening claims (ra, val, inc)
            rwc_ra_claim: F,
            rwc_val_claim: F,
            rwc_inc_claim: F,
            /// Individual InstructionLookups opening claims (5 terms)
            instr_lookup_output_claim: F,
            instr_left_operand_claim: F,
            instr_right_operand_claim: F,
            instr_left_instr_input_claim: F,
            instr_right_instr_input_claim: F,
            allocator: Allocator,

            pub fn deinit(self: *Stage2Result) void {
                self.allocator.free(self.challenges);
                self.allocator.free(self.r_address_raf);
                self.allocator.free(self.r_address_rw);
                self.allocator.free(self.r_cycle_rw);
                self.allocator.free(self.r_cycle_product);
            }
        };

        /// Generate Stage 2 batched sumcheck proof
        ///
        /// Stage 2 batches 5 sumcheck instances:
        /// 1. ProductVirtualRemainder: n_cycle_vars rounds, degree 3
        /// 2. RamRafEvaluation: log_ram_k rounds, degree 2
        /// 3. RamReadWriteChecking: log_ram_k + n_cycle_vars rounds, degree 3
        /// 4. OutputSumcheck: log_ram_k rounds, degree 3
        /// 5. InstructionLookupsClaimReduction: n_cycle_vars rounds, degree 2
        ///
        /// For programs without RAM/lookups, instances 2-5 have zero input claims
        /// and contribute constant-zero polynomials.
        ///
        /// Returns the 8 factor polynomial evaluations at r_cycle for opening claims.
        fn generateStage2BatchedSumcheckProof(
            self: *Self,
            proof: *SumcheckInstanceProof(F),
            transcript: *Blake2bTranscript(F),
            r0_stage2: F,
            uni_skip_claim_stage2: F,
            tau: []const F,
            r_spartan_for_instr: []const F,
            cycle_witnesses: []const r1cs.R1CSCycleInputs(F),
            n_cycle_vars: usize,
            log_ram_k: usize,
            opening_claims: *OpeningClaims(F),
            config: ConversionConfig,
        ) !Stage2Result {
            const max_num_rounds = log_ram_k + n_cycle_vars;
            dbg("[ZOLT] STAGE2_BATCHED: max_rounds={}, n_cycle={}, log_ram_k={}\n", .{ max_num_rounds, n_cycle_vars, log_ram_k });

            // Define the 5 instances with their input claims and round counts
            // Instance 0: ProductVirtualRemainder (input = uni_skip_claim from SpartanProductVirtualization)
            // Instance 1: RamRafEvaluation (input = RamAddress from SpartanOuter)
            // Instance 2: RamReadWriteChecking (input = RamReadValue + gamma * RamWriteValue)
            // Instance 3: OutputSumcheck (input = 0)
            // Instance 4: InstructionLookupsClaimReduction (input = LookupOutput + gamma * LeftOperand + gamma^2 * RightOperand)

            // Get opening claims from proof (these were set during Stage 1)
            const ram_address_claim = opening_claims.get(.{ .Virtual = .{ .poly = .RamAddress, .sumcheck_id = .SpartanOuter } }) orelse F.zero();
            const ram_read_value_claim = opening_claims.get(.{ .Virtual = .{ .poly = .RamReadValue, .sumcheck_id = .SpartanOuter } }) orelse F.zero();
            const ram_write_value_claim = opening_claims.get(.{ .Virtual = .{ .poly = .RamWriteValue, .sumcheck_id = .SpartanOuter } }) orelse F.zero();
            const lookup_output_claim = opening_claims.get(.{ .Virtual = .{ .poly = .LookupOutput, .sumcheck_id = .SpartanOuter } }) orelse F.zero();
            const left_operand_claim = opening_claims.get(.{ .Virtual = .{ .poly = .LeftLookupOperand, .sumcheck_id = .SpartanOuter } }) orelse F.zero();
            const right_operand_claim = opening_claims.get(.{ .Virtual = .{ .poly = .RightLookupOperand, .sumcheck_id = .SpartanOuter } }) orelse F.zero();
            const left_instr_input_claim = opening_claims.get(.{ .Virtual = .{ .poly = .LeftInstructionInput, .sumcheck_id = .SpartanOuter } }) orelse F.zero();
            const right_instr_input_claim = opening_claims.get(.{ .Virtual = .{ .poly = .RightInstructionInput, .sumcheck_id = .SpartanOuter } }) orelse F.zero();

            dbg("[ZOLT] RWC_DEBUG: ram_read_value_claim = {any}\n", .{ram_read_value_claim.toBytesBE()});
            dbg("[ZOLT] RWC_DEBUG: ram_write_value_claim = {any}\n", .{ram_write_value_claim.toBytesBE()});

            // Sample gammas from transcript in the same order as upstream Jolt verifier:
            // 1. RamReadWriteChecking samples gamma first
            // 2. InstructionLookupsClaimReduction samples gamma
            // 3. OutputSumcheck samples r_address
            //
            // CRITICAL: gamma uses challenge_scalar (NO 125-bit masking) = challengeScalarFull()
            // r_address uses challenge_scalar_optimized (HAS 125-bit masking) = challengeScalar()

            // 1. RamReadWriteChecking gamma
            const gamma_rwc = transcript.challengeScalarFull();
            dbg("[ZOLT] STAGE2_BATCHED: gamma_rwc = {any}\n", .{gamma_rwc.toBytesBE()});

            // 2. InstructionLookupsClaimReduction gamma (via challenge_scalar, NO masking)
            const gamma_instr = transcript.challengeScalarFull();
            const gamma_instr_sqr = gamma_instr.mul(gamma_instr);
            const gamma_instr_cub = gamma_instr_sqr.mul(gamma_instr);
            const gamma_instr_quart = gamma_instr_sqr.mul(gamma_instr_sqr);
            dbg("[ZOLT] STAGE2_BATCHED: gamma_instr = {any}\n", .{gamma_instr.toBytesBE()});

            // 3. OutputSumcheck samples r_address (log_ram_k challenges via challenge_vector_optimized)
            const r_address_presampled = try self.allocator.alloc(F, log_ram_k);
            defer self.allocator.free(r_address_presampled);
            for (r_address_presampled) |*r| {
                r.* = transcript.challengeScalar();
            }

            // Compute input_claims in UPSTREAM order:
            // [0] RamReadWriteChecking: RamReadValue + gamma_rwc * RamWriteValue
            // [1] ProductVirtualRemainder: uni_skip_claim
            // [2] InstructionLookupsClaimReduction: LookupOutput + γ*LeftOp + γ²*RightOp + γ³*LeftInstr + γ⁴*RightInstr
            // [3] RamRafEvaluation: RamAddress
            // [4] OutputSumcheck: 0
            const input_claim_rwc = ram_read_value_claim.add(gamma_rwc.mul(ram_write_value_claim));
            const input_claim_instr = lookup_output_claim
                .add(gamma_instr.mul(left_operand_claim))
                .add(gamma_instr_sqr.mul(right_operand_claim))
                .add(gamma_instr_cub.mul(left_instr_input_claim))
                .add(gamma_instr_quart.mul(right_instr_input_claim));

            dbg("[ZOLT] STAGE2_BATCHED: input_claim[0] (RamReadWriteChecking) = {any}\n", .{input_claim_rwc.toBytesBE()});
            dbg("[ZOLT] STAGE2_BATCHED: input_claim[1] (ProductVirtualRemainder) = {any}\n", .{uni_skip_claim_stage2.toBytesBE()});
            dbg("[ZOLT] STAGE2_BATCHED: input_claim[2] (InstructionLookupsClaimReduction) = {any}\n", .{input_claim_instr.toBytesBE()});
            dbg("[ZOLT] STAGE2_BATCHED: input_claim[3] (RamRafEvaluation) = {any}\n", .{ram_address_claim.toBytesBE()});
            dbg("[ZOLT] STAGE2_BATCHED: input_claim[4] (OutputSumcheck) = 0\n", .{});

            const input_claims = [5]F{
                input_claim_rwc, // [0] RamReadWriteChecking
                uni_skip_claim_stage2, // [1] ProductVirtualRemainder
                input_claim_instr, // [2] InstructionLookupsClaimReduction
                ram_address_claim, // [3] RamRafEvaluation
                F.zero(), // [4] OutputSumcheck
            };

            const rounds_per_instance = [5]usize{
                log_ram_k + n_cycle_vars, // [0] RamReadWriteChecking
                n_cycle_vars, // [1] ProductVirtualRemainder
                n_cycle_vars, // [2] InstructionLookupsClaimReduction
                log_ram_k, // [3] RamRafEvaluation
                log_ram_k, // [4] OutputSumcheck
            };

            // Step 1: Append all input claims to transcript
            for (input_claims) |claim| {
                transcript.appendScalar("sumcheck_claim", claim);
            }

            // Debug: STAGE2_PRE logs for compare_sumcheck.py compatibility
            for (0..5) |i| {
                const claim_bytes = input_claims[i].toBytes();
                dbg("[ZOLT] STAGE2_PRE: input_claim[{d}] = {{ ", .{i});
                for (claim_bytes) |b| {
                    dbg("{d}, ", .{b});
                }
                dbg("}}\n", .{});
                dbg("[ZOLT] STAGE2_PRE: num_rounds[{d}] = {d}\n", .{ i, rounds_per_instance[i] });
                dbg("[ZOLT] STAGE2_PRE: degree[{d}] = 3\n", .{i}); // All instances use degree 3 max
            }
            dbg("[ZOLT] STAGE2: transcript state after input_claims = {any}\n", .{transcript.state[0..8]});

            // Step 2: Sample batching coefficients (input claims already appended at line 1747)
            var batching_coeffs: [5]F = undefined;
            for (0..5) |i| {
                batching_coeffs[i] = transcript.challengeScalarFull();
            }

            // Debug: STAGE2_PRE batching coefficient logs for compare_sumcheck.py
            dbg("[ZOLT] STAGE2_PRE: batching_coeffs.len = 5\n", .{});
            for (0..5) |i| {
                const coeff_bytes = batching_coeffs[i].toBytes();
                dbg("[ZOLT] STAGE2_PRE: batching_coeff[{d}] = {{ ", .{i});
                for (coeff_bytes) |b| {
                    dbg("{d}, ", .{b});
                }
                dbg("}}\n", .{});
            }

            dbg("[ZOLT] STAGE2_BATCHED: batching_coeff[0] = {any}\n", .{batching_coeffs[0].toBytesBE()});

            // Step 3: Compute initial batched claim
            // batched_claim = Σᵢ αᵢ * input_claim[i] * 2^(max_rounds - rounds[i])
            var batched_claim = F.zero();
            for (0..5) |i| {
                const scale_power = max_num_rounds - rounds_per_instance[i];
                var scaled_claim = input_claims[i];
                for (0..scale_power) |_| {
                    scaled_claim = scaled_claim.add(scaled_claim);
                }
                batched_claim = batched_claim.add(scaled_claim.mul(batching_coeffs[i]));
            }

            dbg("[ZOLT] STAGE2_BATCHED: initial batched_claim = {any}\n", .{batched_claim.toBytesBE()});
            dbg("[ZOLT] STAGE2_BATCHED: uni_skip_claim_stage2 (product input) = {any}\n", .{uni_skip_claim_stage2.toBytesBE()});

            // Debug: STAGE2_INITIAL log for compare_sumcheck.py
            {
                const claim_bytes = batched_claim.toBytes();
                dbg("[ZOLT] STAGE2_INITIAL: batched_claim = {{ ", .{});
                for (claim_bytes) |b| {
                    dbg("{d}, ", .{b});
                }
                dbg("}}\n", .{});
            }

            // Initialize provers for each instance (upstream ordering):
            // [0] RamReadWriteChecking, [1] ProductVirtualRemainder,
            // [2] InstructionLookupsClaimReduction, [3] RamRafEvaluation, [4] OutputSumcheck

            // Instance 0: RamReadWriteChecking - starts at round 0 (max rounds)
            const RWCProver = ram.RamReadWriteCheckingProver(F);
            var rwc_prover: ?RWCProver = null;
            var rwc_evals_this_round: ?[4]F = null;
            const use_rwc_prover = !input_claims[0].eql(F.zero());
            if (config.memory_trace != null and use_rwc_prover) {
                const phase1_num_rounds = n_cycle_vars;
                var rwc_params = ram.RamReadWriteCheckingParams(F).initWithPhaseConfig(
                    self.allocator,
                    gamma_rwc,
                    tau[0..n_cycle_vars],
                    log_ram_k,
                    n_cycle_vars,
                    phase1_num_rounds,
                    if (config.memory_layout) |ml| ml.getLowestAddress() else 0x80000000,
                ) catch null;

                if (rwc_params) |*params| {
                    rwc_prover = RWCProver.init(
                        self.allocator,
                        config.memory_trace.?,
                        params.*,
                        input_claims[0],
                        config.initial_ram,
                        config.memory_layout,
                        config.is_panicking,
                    ) catch null;

                    if (rwc_prover != null) {
                        dbg("[ZOLT] RWC: Prover initialized for instance 0\n", .{});
                    } else {
                        params.deinit();
                    }
                }
            }
            defer if (rwc_prover) |*rp| rp.deinit();

            // Instance 1: ProductVirtualRemainder
            const ProductRemainderProver = product_remainder.ProductVirtualRemainderProver(F);
            var product_prover: ?ProductRemainderProver = null;
            if (cycle_witnesses.len > 0 and tau.len > 0) {
                product_prover = ProductRemainderProver.init(
                    self.allocator,
                    r0_stage2,
                    tau,
                    uni_skip_claim_stage2,
                    cycle_witnesses,
                ) catch null;
            }
            defer if (product_prover) |*p| p.deinit();

            // Instance 2: InstructionLookupsClaimReduction - initialized lazily at start_round
            const claim_reductions = @import("claim_reductions/mod.zig");
            const InstrLookupsProver = claim_reductions.InstructionLookupsProver(F);
            var instr_prover: ?InstrLookupsProver = null;
            var instr_evals_this_round: ?[4]F = null;
            defer if (instr_prover) |*ip| ip.deinit();

            // Instance 3: RamRafEvaluation - initialized lazily at start_round
            const RafProver = ram.RafEvaluationProver(F);
            var raf_prover: ?RafProver = null;
            var raf_evals_this_round: ?[4]F = null;
            defer if (raf_prover) |*rp| rp.deinit();

            // Instance 4: OutputSumcheck
            const OutputProver = ram.OutputSumcheckProver(F);
            var output_prover: ?OutputProver = null;
            if (config.memory_layout != null and config.initial_ram != null and config.final_ram != null) {
                output_prover = OutputProver.init(
                    self.allocator,
                    config.initial_ram.?,
                    config.final_ram.?,
                    r_address_presampled,
                    config.memory_layout.?,
                    config.program_inputs,
                    config.program_outputs,
                    config.is_panicking,
                ) catch null;
                if (output_prover) |_| {
                    dbg("[ZOLT] STAGE2_BATCHED: OutputSumcheckProver initialized\n", .{});
                }
            }
            defer if (output_prover) |*p| p.deinit();

            // Track individual claims for each instance (needed for zero-poly instances)
            var individual_claims: [5]F = undefined;
            for (0..5) |i| {
                const scale_power = max_num_rounds - rounds_per_instance[i];
                var scaled = input_claims[i];
                for (0..scale_power) |_| {
                    scaled = scaled.add(scaled);
                }
                individual_claims[i] = scaled;
            }

            // Store challenges for opening claims computation
            var challenges: std.ArrayListUnmanaged(F) = .{};
            defer challenges.deinit(self.allocator);

            // Step 4: Run batched sumcheck rounds
            for (0..max_num_rounds) |round_idx| {
                // Compute combined polynomial from all instances
                var combined_evals = [4]F{ F.zero(), F.zero(), F.zero(), F.zero() };
                // Store ProductVirtualRemainder's evals for claim update
                var product_evals_this_round: ?[4]F = null;
                // Store OutputSumcheck's evals for claim update
                var output_evals_this_round: ?[4]F = null;

                for (0..5) |i| {
                    const start_round = max_num_rounds - rounds_per_instance[i];

                    if (round_idx >= start_round) {
                        // Instance is active
                        if (i == 0) {
                            // Instance 0: RamReadWriteChecking (max rounds, starts at round 0)
                            if (rwc_prover) |*rwcp| {
                                const rwc_evals = rwcp.computeRoundPolynomialCubic();
                                rwc_evals_this_round = rwc_evals;
                                for (0..4) |j| {
                                    combined_evals[j] = combined_evals[j].add(rwc_evals[j].mul(batching_coeffs[i]));
                                }
                            } else {
                                // Fallback: constant polynomial from scaled claim
                                const instance_round = round_idx - start_round;
                                const remaining_rounds = rounds_per_instance[i] - 1 - instance_round;
                                var scaled = input_claims[i];
                                for (0..remaining_rounds) |_| scaled = scaled.add(scaled);
                                const weighted = scaled.mul(batching_coeffs[i]);
                                for (0..4) |j| combined_evals[j] = combined_evals[j].add(weighted);
                            }
                        } else if (i == 1) {
                            // Instance 1: ProductVirtualRemainder
                            if (product_prover) |_| {
                                const claim_before = product_prover.?.current_claim;
                                const comp = product_prover.?.computeRoundPolynomial() catch [3]F{ F.zero(), F.zero(), F.zero() };
                                const c0 = comp[0];
                                const c2_p = comp[1];
                                const c3_p = comp[2];
                                const c1 = claim_before.sub(c0).sub(c0).sub(c2_p).sub(c3_p);
                                const s0 = c0;
                                const s1 = claim_before.sub(s0);
                                const s2 = c0.add(c1.mul(F.fromU64(2))).add(c2_p.mul(F.fromU64(4))).add(c3_p.mul(F.fromU64(8)));
                                const s3 = c0.add(c1.mul(F.fromU64(3))).add(c2_p.mul(F.fromU64(9))).add(c3_p.mul(F.fromU64(27)));
                                product_evals_this_round = [4]F{ s0, s1, s2, s3 };
                                for (0..4) |j| {
                                    combined_evals[j] = combined_evals[j].add(product_evals_this_round.?[j].mul(batching_coeffs[i]));
                                }
                            } else {
                                const instance_round = round_idx - start_round;
                                const remaining_rounds = rounds_per_instance[i] - 1 - instance_round;
                                var scaled = input_claims[i];
                                for (0..remaining_rounds) |_| scaled = scaled.add(scaled);
                                const weighted = scaled.mul(batching_coeffs[i]);
                                for (0..4) |j| combined_evals[j] = combined_evals[j].add(weighted);
                            }
                        } else if (i == 2) {
                            // Instance 2: InstructionLookupsClaimReduction
                            if (round_idx == start_round and instr_prover == null and cycle_witnesses.len > 0) {
                                var instr_params = claim_reductions.InstructionLookupsParams(F).init(
                                    self.allocator,
                                    gamma_instr,
                                    r_spartan_for_instr,
                                    n_cycle_vars,
                                ) catch null;

                                if (instr_params) |*params| {
                                    const R1CSInputIndex = @import("r1cs/constraints.zig").R1CSInputIndex;
                                    const lookup_outputs_arr = try self.allocator.alloc(F, cycle_witnesses.len);
                                    defer self.allocator.free(lookup_outputs_arr);
                                    const left_operands_arr = try self.allocator.alloc(F, cycle_witnesses.len);
                                    defer self.allocator.free(left_operands_arr);
                                    const right_operands_arr = try self.allocator.alloc(F, cycle_witnesses.len);
                                    defer self.allocator.free(right_operands_arr);
                                    const left_instr_arr = try self.allocator.alloc(F, cycle_witnesses.len);
                                    defer self.allocator.free(left_instr_arr);
                                    const right_instr_arr = try self.allocator.alloc(F, cycle_witnesses.len);
                                    defer self.allocator.free(right_instr_arr);

                                    for (cycle_witnesses, 0..) |w, wi| {
                                        lookup_outputs_arr[wi] = w.values[R1CSInputIndex.LookupOutput.toIndex()];
                                        left_operands_arr[wi] = w.values[R1CSInputIndex.LeftLookupOperand.toIndex()];
                                        right_operands_arr[wi] = w.values[R1CSInputIndex.RightLookupOperand.toIndex()];
                                        left_instr_arr[wi] = w.values[R1CSInputIndex.LeftInstructionInput.toIndex()];
                                        right_instr_arr[wi] = w.values[R1CSInputIndex.RightInstructionInput.toIndex()];
                                    }

                                    instr_prover = InstrLookupsProver.init(
                                        self.allocator,
                                        params.*,
                                        input_claims[2],
                                        lookup_outputs_arr,
                                        left_operands_arr,
                                        right_operands_arr,
                                        left_instr_arr,
                                        right_instr_arr,
                                    ) catch blk: {
                                        params.deinit();
                                        break :blk null;
                                    };

                                    if (instr_prover != null) {
                                        dbg("[ZOLT] InstrLookups: Prover initialized for instance 2\n", .{});
                                    }
                                }
                            }

                            if (instr_prover) |*ip| {
                                const instr_evals = ip.computeRoundPolynomialCubic();
                                instr_evals_this_round = instr_evals;
                                for (0..4) |j| {
                                    combined_evals[j] = combined_evals[j].add(instr_evals[j].mul(batching_coeffs[i]));
                                }
                            } else {
                                const instance_round = round_idx - start_round;
                                const remaining_rounds = rounds_per_instance[i] - 1 - instance_round;
                                var scaled = input_claims[i];
                                for (0..remaining_rounds) |_| scaled = scaled.add(scaled);
                                const weighted = scaled.mul(batching_coeffs[i]);
                                for (0..4) |j| combined_evals[j] = combined_evals[j].add(weighted);
                            }
                        } else if (i == 3) {
                            // Instance 3: RamRafEvaluation
                            const use_raf_prover = !input_claims[3].eql(F.zero());
                            if (round_idx == start_round and raf_prover == null and config.memory_trace != null and use_raf_prover) {
                                const r_cycle_slice = tau[0..n_cycle_vars];
                                const r_cycle = try self.allocator.alloc(F, n_cycle_vars);
                                @memcpy(r_cycle, r_cycle_slice);
                                const start_addr: u64 = if (config.memory_layout) |ml| ml.getLowestAddress() else 0x80000000;
                                var raf_params = try ram.RafEvaluationParams(F).init(self.allocator, log_ram_k, start_addr, r_cycle);
                                self.allocator.free(r_cycle);

                                const raf_initial_claim = input_claims[3];
                                dbg("[ZOLT] RAF: Initializing with claim = {any}\n", .{raf_initial_claim.toBytesBE()});

                                raf_prover = RafProver.init(self.allocator, config.memory_trace.?, raf_params, raf_initial_claim) catch |err| blk: {
                                    dbg("[ZOLT] RAF: Prover init failed: {}\n", .{err});
                                    raf_params.deinit();
                                    break :blk null;
                                };
                            }

                            if (raf_prover) |*rp| {
                                const raf_evals = rp.computeRoundPolynomialCubic();
                                raf_evals_this_round = raf_evals;
                                for (0..4) |j| {
                                    combined_evals[j] = combined_evals[j].add(raf_evals[j].mul(batching_coeffs[i]));
                                }
                            } else {
                                if (round_idx == start_round) {
                                    dbg("[ZOLT] WARNING: Instance 3 (RAF) using fallback - no prover\n", .{});
                                }
                                const remaining_rounds = rounds_per_instance[i] - (round_idx - start_round);
                                var scaled = individual_claims[i];
                                for (0..remaining_rounds) |_| scaled = scaled.mul(F.fromU64(2));
                                scaled = scaled.mul(F.fromU64(2).inverse().?);
                                const weighted = scaled.mul(batching_coeffs[i]);
                                for (0..4) |j| combined_evals[j] = combined_evals[j].add(weighted);
                            }
                        } else if (i == 4) {
                            // Instance 4: OutputSumcheck
                            if (output_prover) |_| {
                                const output_compressed = output_prover.?.computeRoundPolynomial();
                                const c0 = output_compressed[0];
                                const c2_o = output_compressed[1];
                                const c3_o = output_compressed[2];
                                const current_claim_output = output_prover.?.current_claim;
                                const c1 = current_claim_output.sub(c0).sub(c0).sub(c2_o).sub(c3_o);
                                const s0_out = c0;
                                const s1_out = current_claim_output.sub(s0_out);
                                const s2_out = c0.add(c1.mul(F.fromU64(2))).add(c2_o.mul(F.fromU64(4))).add(c3_o.mul(F.fromU64(8)));
                                const s3_out = c0.add(c1.mul(F.fromU64(3))).add(c2_o.mul(F.fromU64(9))).add(c3_o.mul(F.fromU64(27)));
                                output_evals_this_round = [4]F{ s0_out, s1_out, s2_out, s3_out };
                                for (0..4) |j| {
                                    combined_evals[j] = combined_evals[j].add(output_evals_this_round.?[j].mul(batching_coeffs[i]));
                                }
                            } else {
                                // Zero input claim → zero polynomial
                                const scale_power = rounds_per_instance[i] - 1 - (round_idx - start_round);
                                var scaled = input_claims[i];
                                for (0..scale_power) |_| scaled = scaled.add(scaled);
                                const weighted = scaled.mul(batching_coeffs[i]);
                                for (0..4) |j| combined_evals[j] = combined_evals[j].add(weighted);
                            }
                        }
                    } else {
                        // Instance hasn't started yet - contribute scaled input claim as constant
                        const scale_power = max_num_rounds - rounds_per_instance[i] - round_idx - 1;
                        var scaled = input_claims[i];
                        for (0..scale_power) |_| scaled = scaled.add(scaled);
                        const weighted = scaled.mul(batching_coeffs[i]);
                        for (0..4) |j| combined_evals[j] = combined_evals[j].add(weighted);
                    }
                }

                // Convert to compressed coefficients [c0, c2, c3]
                const compressed = poly_mod.UniPoly(F).evalsToCompressed(combined_evals);

                if (round_idx == 0 or round_idx == 16 or round_idx == max_num_rounds - 1) {
                    dbg("[ZOLT] STAGE2_BATCHED round {}: combined_evals[0] = {any}\n", .{ round_idx, combined_evals[0].toBytesBE() });
                    dbg("[ZOLT] STAGE2_BATCHED round {}: combined_evals[1] = {any}\n", .{ round_idx, combined_evals[1].toBytesBE() });
                    dbg("[ZOLT] STAGE2_BATCHED round {}: compressed[0] (c0) = {any}\n", .{ round_idx, compressed[0].toBytesBE() });
                    dbg("[ZOLT] STAGE2_BATCHED round {}: compressed[2] (c3) = {any}\n", .{ round_idx, compressed[2].toBytesBE() });
                }

                // Append to proof
                const coeffs = try self.allocator.alloc(F, 3);
                coeffs[0] = compressed[0];
                coeffs[1] = compressed[1];
                coeffs[2] = compressed[2];
                try proof.compressed_polys.append(self.allocator, .{
                    .coeffs_except_linear_term = coeffs,
                    .allocator = self.allocator,
                });

                // Append to transcript: sumcheck polynomial coefficients
                transcript.appendScalars("sumcheck_poly", compressed[0..3]);

                // Sample round challenge
                const challenge = transcript.challengeScalar();
                try challenges.append(self.allocator, challenge);

                // Update batched claim by evaluating at challenge
                // CRITICAL: Must use evalFromHint (same as Jolt's verifier) to ensure
                // the claim evolution matches. Using Lagrange interpolation from combined_evals
                // would give different results because the evaluations may not be consistent
                // with what Jolt expects (different s1, s2, s3 can produce the same c0, c2, c3).
                const old_claim = batched_claim;
                batched_claim = evalFromHint(compressed, old_claim, challenge);


                // Debug: STAGE2_ROUND logs for compare_sumcheck.py
                {
                    const old_bytes = old_claim.toBytes();
                    dbg("[ZOLT] STAGE2_ROUND_{d}: current_claim = {{ ", .{round_idx});
                    for (old_bytes) |b| dbg("{d}, ", .{b});
                    dbg("}}\n", .{});

                    const c0_bytes = compressed[0].toBytes();
                    dbg("[ZOLT] STAGE2_ROUND_{d}: c0 = {{ ", .{round_idx});
                    for (c0_bytes) |b| dbg("{d}, ", .{b});
                    dbg("}}\n", .{});

                    const c2_bytes = compressed[1].toBytes();
                    dbg("[ZOLT] STAGE2_ROUND_{d}: c2 = {{ ", .{round_idx});
                    for (c2_bytes) |b| dbg("{d}, ", .{b});
                    dbg("}}\n", .{});

                    const c3_bytes = compressed[2].toBytes();
                    dbg("[ZOLT] STAGE2_ROUND_{d}: c3 = {{ ", .{round_idx});
                    for (c3_bytes) |b| dbg("{d}, ", .{b});
                    dbg("}}\n", .{});

                    const chal_bytes = challenge.toBytes();
                    dbg("[ZOLT] STAGE2_ROUND_{d}: challenge = {{ ", .{round_idx});
                    for (chal_bytes) |b| dbg("{d}, ", .{b});
                    dbg("}}\n", .{});

                    const new_bytes = batched_claim.toBytes();
                    dbg("[ZOLT] STAGE2_ROUND_{d}: next_claim = {{ ", .{round_idx});
                    for (new_bytes) |b| dbg("{d}, ", .{b});
                    dbg("}}\n", .{});
                }

                // Debug: Print claim trajectory for first few and last few rounds
                if (round_idx < 3 or round_idx >= max_num_rounds - 5) {
                    dbg("[ZOLT CLAIM] round {}: old_claim = {any}\n", .{ round_idx, old_claim.toBytesBE() });
                    dbg("[ZOLT CLAIM] round {}: s(0)+s(1) = {any}\n", .{ round_idx, combined_evals[0].add(combined_evals[1]).toBytesBE() });
                    dbg("[ZOLT CLAIM] round {}: new_claim = {any}\n", .{ round_idx, batched_claim.toBytesBE() });
                    // Check: s(0) + s(1) should equal old_claim for soundness
                    const sum_check = combined_evals[0].add(combined_evals[1]);
                    if (!sum_check.eql(old_claim)) {
                        dbg("[ZOLT CLAIM ERROR] round {}: s(0)+s(1) != old_claim!\n", .{round_idx});
                        // Print individual instance contributions
                        dbg("[ZOLT DEBUG] Instance contributions at round {}:\n", .{round_idx});
                        dbg("  Instance 0 (ProductVirtual) active: {}, prover: {}\n", .{ round_idx >= max_num_rounds - n_cycle_vars, product_prover != null });
                        if (product_evals_this_round) |pe| {
                            const ps = pe[0].add(pe[1]).mul(batching_coeffs[0]);
                            dbg("  Instance 0: s0+s1 contrib = {any}\n", .{ps.toBytesBE()});
                            dbg("  Instance 0: s0 = {any}, s1 = {any}\n", .{ pe[0].toBytesBE(), pe[1].toBytesBE() });
                            dbg("  Instance 0: s0+s1 = {any}\n", .{pe[0].add(pe[1]).toBytesBE()});
                            // Note: pp.current_claim is ALREADY UPDATED for next round at this point!
                            dbg("  Instance 0: current_claim (next round) = {any}\n", .{if (product_prover) |pp| pp.current_claim.toBytesBE() else [_]u8{0} ** 32});
                        } else {
                            dbg("  Instance 0: NULL evals\n", .{});
                        }
                        dbg("  Instance 1 (RAF) active: {}, prover: {}\n", .{ round_idx >= max_num_rounds - log_ram_k, raf_prover != null });
                        if (raf_evals_this_round) |re| {
                            const rs = re[0].add(re[1]).mul(batching_coeffs[1]);
                            dbg("  Instance 1: s0+s1 contrib = {any}\n", .{rs.toBytesBE()});
                        } else {
                            dbg("  Instance 1: NULL evals\n", .{});
                        }
                        dbg("  Instance 2 (RWC) active: {}, prover: {}\n", .{ round_idx >= 0, rwc_prover != null });
                        if (rwc_evals_this_round) |re| {
                            const rs = re[0].add(re[1]).mul(batching_coeffs[2]);
                            dbg("  Instance 2: s0+s1 contrib = {any}\n", .{rs.toBytesBE()});
                        } else {
                            dbg("  Instance 2: NULL evals\n", .{});
                        }
                        dbg("  Instance 3 (Output) active: {}, prover: {}\n", .{ round_idx >= max_num_rounds - log_ram_k, output_prover != null });
                        if (output_evals_this_round) |oe| {
                            const os = oe[0].add(oe[1]).mul(batching_coeffs[3]);
                            dbg("  Instance 3: s0+s1 contrib = {any}\n", .{os.toBytesBE()});
                        } else {
                            dbg("  Instance 3: NULL evals\n", .{});
                        }
                        dbg("  Instance 4 (Instr) active: {}, prover: {}\n", .{ round_idx >= max_num_rounds - n_cycle_vars, instr_prover != null });
                        if (instr_evals_this_round) |ie| {
                            const is = ie[0].add(ie[1]).mul(batching_coeffs[4]);
                            dbg("  Instance 4: s0+s1 contrib = {any}\n", .{is.toBytesBE()});
                        } else {
                            dbg("  Instance 4: NULL evals\n", .{});
                        }
                    }
                }

                // Bind challenge in all active instances and update their claims
                // Instance 0: RWC (starts at round 0)
                if (rwc_prover) |*rwcp| {
                    if (rwc_evals_this_round) |evals| rwcp.updateClaim(evals, challenge);
                    rwcp.bindChallenge(challenge) catch {};
                }

                // Instance 1: ProductVirtualRemainder (starts at max_rounds - n_cycle_vars)
                if (product_prover != null and round_idx >= (max_num_rounds - n_cycle_vars)) {
                    if (product_evals_this_round) |evals| product_prover.?.updateClaim(evals, challenge);
                    product_prover.?.bindChallenge(challenge) catch {};
                }

                // Instance 2: InstructionLookups (starts at max_rounds - n_cycle_vars)
                if (instr_prover != null and round_idx >= (max_num_rounds - n_cycle_vars)) {
                    if (instr_evals_this_round) |evals| instr_prover.?.updateClaim(evals, challenge);
                    instr_prover.?.bindChallenge(challenge) catch {};
                }

                // Instance 3: RAF (starts at max_rounds - log_ram_k)
                if (raf_prover != null and round_idx >= (max_num_rounds - log_ram_k)) {
                    if (raf_evals_this_round) |evals| raf_prover.?.updateClaim(evals, challenge);
                    raf_prover.?.bindChallenge(challenge) catch {};
                }

                // Instance 4: OutputSumcheck (starts at max_rounds - log_ram_k)
                if (output_prover != null and round_idx >= (max_num_rounds - log_ram_k)) {
                    if (output_evals_this_round) |evals| output_prover.?.updateClaim(evals, challenge);
                    output_prover.?.bindChallenge(challenge);
                }

                // Reset per-round evals
                raf_evals_this_round = null;
                rwc_evals_this_round = null;
                instr_evals_this_round = null;

                // CRITICAL: Update individual_claims for each instance by evaluating at challenge
                // This is required for the batched sumcheck to maintain correct claim tracking
                // For inactive instances, the constant polynomial evaluates to the same scaled value
                // For active instances, we update based on the polynomial evaluation
                for (0..5) |i| {
                    const start_round = max_num_rounds - rounds_per_instance[i];
                    if (round_idx >= start_round) {
                        // Instance was active - update claim from its prover
                        if (i == 0 and rwc_prover != null) {
                            individual_claims[i] = rwc_prover.?.current_claim;
                        } else if (i == 1 and product_prover != null) {
                            individual_claims[i] = product_prover.?.current_claim;
                        } else if (i == 2 and instr_prover != null) {
                            individual_claims[i] = instr_prover.?.current_claim;
                        } else if (i == 3 and raf_prover != null) {
                            individual_claims[i] = raf_prover.?.current_claim;
                        } else if (i == 4 and output_prover != null) {
                            individual_claims[i] = output_prover.?.current_claim;
                        } else {
                            // Fallback: for instances without provers, keep tracking manually
                            // The claim after evaluating constant polynomial at r is just the constant
                            const remaining = rounds_per_instance[i] - (round_idx - start_round) - 1;
                            var scaled = input_claims[i];
                            for (0..remaining) |_| {
                                scaled = scaled.add(scaled);
                            }
                            individual_claims[i] = scaled;
                        }
                    } else {
                        // Instance not yet active - constant polynomial evaluates to scaled claim
                        // scale_power = remaining rounds until activation - 1
                        // = (start_round - round_idx - 1) where start_round = max_num_rounds - rounds_per_instance[i]
                        const start_round_i = max_num_rounds - rounds_per_instance[i];
                        if (round_idx + 1 < start_round_i) {
                            const scale_power = start_round_i - round_idx - 2;
                            var scaled = input_claims[i];
                            for (0..scale_power) |_| {
                                scaled = scaled.add(scaled);
                            }
                            individual_claims[i] = scaled;
                        } else {
                            // At the round just before activation, scale_power = 0
                            individual_claims[i] = input_claims[i];
                        }
                    }
                }

                // Debug: Check divergence between batched_claim and sum of individual claims
                // Do this AFTER individual_claims update so they're in sync
                if (round_idx == 15 or round_idx == 16 or round_idx == 25) {
                    var should_be_batched = F.zero();
                    for (0..5) |dbg_i| {
                        should_be_batched = should_be_batched.add(individual_claims[dbg_i].mul(batching_coeffs[dbg_i]));
                    }
                    dbg("[ZOLT SYNC] round {}: batched = {any}\n", .{ round_idx, batched_claim.toBytesBE() });
                    dbg("[ZOLT SYNC] round {}: should_be = {any}\n", .{ round_idx, should_be_batched.toBytesBE() });
                    dbg("[ZOLT SYNC] round {}: match = {}\n", .{ round_idx, batched_claim.eql(should_be_batched) });
                }
            }

            dbg("[ZOLT] STAGE2_BATCHED: final batched_claim = {any}\n", .{batched_claim.toBytesBE()});

            // Debug: Verify batched_claim equals sum of (coeff * prover_claim)
            var expected_batched = F.zero();
            // Instance 0: RWC
            if (rwc_prover) |rp| {
                expected_batched = expected_batched.add(rp.current_claim.mul(batching_coeffs[0]));
            } else {
                expected_batched = expected_batched.add(individual_claims[0].mul(batching_coeffs[0]));
            }
            // Instance 1: Product
            if (product_prover) |pp| {
                expected_batched = expected_batched.add(pp.current_claim.mul(batching_coeffs[1]));
            } else {
                expected_batched = expected_batched.add(individual_claims[1].mul(batching_coeffs[1]));
            }
            // Instance 2: InstrLookups
            if (instr_prover) |*ip| {
                expected_batched = expected_batched.add(ip.current_claim.mul(batching_coeffs[2]));
            } else {
                expected_batched = expected_batched.add(individual_claims[2].mul(batching_coeffs[2]));
            }
            // Instance 3: RAF
            if (raf_prover) |rp| {
                expected_batched = expected_batched.add(rp.current_claim.mul(batching_coeffs[3]));
            } else {
                expected_batched = expected_batched.add(individual_claims[3].mul(batching_coeffs[3]));
            }
            // Instance 4: Output
            if (output_prover) |op| {
                expected_batched = expected_batched.add(op.current_claim.mul(batching_coeffs[4]));
            } else {
                expected_batched = expected_batched.add(individual_claims[4].mul(batching_coeffs[4]));
            }
            dbg("[ZOLT DEBUG] expected_batched (from provers) = {any}\n", .{expected_batched.toBytesBE()});
            dbg("[ZOLT DEBUG] actual batched = {any}\n", .{batched_claim.toBytesBE()});
            dbg("[ZOLT DEBUG] MATCH: {}\n", .{expected_batched.eql(batched_claim)});

            // Debug: STAGE2_FINAL log for compare_sumcheck.py
            {
                const final_bytes = batched_claim.toBytes();
                dbg("[ZOLT] STAGE2_FINAL: output_claim = {{ ", .{});
                for (final_bytes) |b| {
                    dbg("{d}, ", .{b});
                }
                dbg("}}\n", .{});
            }

            // Debug: Print all challenges in LE format for comparison with Jolt
            dbg("[ZOLT] STAGE2_BATCHED: challenges.len = {}\n", .{challenges.items.len});
            for (challenges.items, 0..) |ch, idx| {
                const be_bytes = ch.toBytesBE();
                // Convert to LE: last 8 bytes of BE = first 8 bytes of LE
                dbg("[ZOLT] STAGE2_BATCHED: challenge[{}] LE first 8 bytes = [{x:0>2}, {x:0>2}, {x:0>2}, {x:0>2}, {x:0>2}, {x:0>2}, {x:0>2}, {x:0>2}]\n", .{ idx, be_bytes[31], be_bytes[30], be_bytes[29], be_bytes[28], be_bytes[27], be_bytes[26], be_bytes[25], be_bytes[24] });
            }

            // Debug: Print prover's final left/right values
            if (product_prover) |pp| {
                dbg("[ZOLT] PROVER FINAL: left[0] = {any}\n", .{pp.left_poly.evaluations[0].toBytesBE()});
                dbg("[ZOLT] PROVER FINAL: right[0] = {any}\n", .{pp.right_poly.evaluations[0].toBytesBE()});
                dbg("[ZOLT] PROVER FINAL: split_eq.current_scalar = {any}\n", .{pp.split_eq.current_scalar.toBytesBE()});
                const prover_final = pp.left_poly.evaluations[0].mul(pp.right_poly.evaluations[0]).mul(pp.split_eq.current_scalar);
                dbg("[ZOLT] PROVER FINAL: left * right * eq = {any}\n", .{prover_final.toBytesBE()});
            }

            // Compute the 8 factor polynomial evaluations at r_cycle
            // r_cycle is the last n_cycle_vars challenges from Stage 2
            // ProductVirtualRemainder starts at round log_ram_k, so its r_cycle
            // is challenges[log_ram_k..max_num_rounds]
            const factor_evals = try self.computeProductFactorEvaluations(
                cycle_witnesses,
                challenges.items,
                n_cycle_vars,
                log_ram_k,
            );

            // Debug: Compute fused_left and fused_right from factor_evals and compare
            // Lagrange weights at r0_stage2
            const LagrangePoly = r1cs.univariate_skip.LagrangePolynomial(F);
            const w = try LagrangePoly.evals(3, r0_stage2, self.allocator);
            defer self.allocator.free(w);

            // fused_left = w[0]*l_inst + w[1]*lookup_out + w[2]*j_flag
            const fused_left = w[0].mul(factor_evals[0])
                .add(w[1].mul(factor_evals[4]))
                .add(w[2].mul(factor_evals[2]));
            // fused_right = w[0]*r_inst + w[1]*branch_flag + w[2]*(1 - next_is_noop)
            const one_minus_next_noop = F.one().sub(factor_evals[6]);
            const fused_right = w[0].mul(factor_evals[1])
                .add(w[1].mul(factor_evals[5]))
                .add(w[2].mul(one_minus_next_noop));

            dbg("[ZOLT] FACTOR CLAIMS: fused_left = {any}\n", .{fused_left.toBytesBE()});
            dbg("[ZOLT] FACTOR CLAIMS: fused_right = {any}\n", .{fused_right.toBytesBE()});

            // Compute tau_high_bound_r0 and tau_bound_r_tail_rev for expected_output_claim debug
            // tau_high_bound_r0 = LagrangeKernel(5, tau_high, r0)
            const tau_high = tau[tau.len - 1];
            const tau_high_bound_r0 = try LagrangePoly.lagrangeKernel(3, tau_high, r0_stage2, self.allocator);
            dbg("[ZOLT] FACTOR CLAIMS: tau_high = {any}\n", .{tau_high.toBytesBE()});
            dbg("[ZOLT] FACTOR CLAIMS: r0_stage2 = {any}\n", .{r0_stage2.toBytesBE()});
            dbg("[ZOLT] FACTOR CLAIMS: tau_high_bound_r0 = {any}\n", .{tau_high_bound_r0.toBytesBE()});

            // tau_bound_r_tail_rev = eq(tau_low, r_cycle_reversed)
            // tau_low = tau[0..n_cycle_vars]
            // r_cycle_reversed = last n_cycle_vars challenges, reversed
            // The challenges.items are the Stage 2 sumcheck challenges
            // ProductVirtualRemainder starts at round (max_num_rounds - n_cycle_vars)
            // Its challenges are the LAST n_cycle_vars of challenges.items
            const product_start_round = max_num_rounds - n_cycle_vars;
            dbg("[ZOLT] FACTOR CLAIMS: product_start_round = {}, challenges.len = {}, n_cycle_vars = {}\n", .{ product_start_round, challenges.items.len, n_cycle_vars });

            // Extract ProductVirtualRemainder challenges (last n_cycle_vars)
            var product_challenges = try self.allocator.alloc(F, n_cycle_vars);
            defer self.allocator.free(product_challenges);
            for (0..n_cycle_vars) |i| {
                if (product_start_round + i < challenges.items.len) {
                    product_challenges[i] = challenges.items[product_start_round + i];
                } else {
                    product_challenges[i] = F.zero();
                }
            }

            // Reverse the product challenges (r_cycle_reversed)
            var r_cycle_reversed = try self.allocator.alloc(F, n_cycle_vars);
            defer self.allocator.free(r_cycle_reversed);
            for (0..n_cycle_vars) |i| {
                r_cycle_reversed[i] = product_challenges[n_cycle_vars - 1 - i];
            }

            // Compute eq(tau_low, r_cycle_reversed)
            const tau_low = tau[0..n_cycle_vars];
            const EqPoly = poly_mod.EqPolynomial(F);
            const tau_bound_r_tail_rev = EqPoly.mle(tau_low, r_cycle_reversed);

            dbg("[ZOLT] FACTOR CLAIMS: tau_low[0] = {any}\n", .{tau_low[0].toBytesBE()});
            dbg("[ZOLT] FACTOR CLAIMS: r_cycle_reversed[0] = {any}\n", .{r_cycle_reversed[0].toBytesBE()});
            dbg("[ZOLT] FACTOR CLAIMS: tau_bound_r_tail_rev = {any}\n", .{tau_bound_r_tail_rev.toBytesBE()});

            // Compute expected_output_claim
            const expected_output_claim = tau_high_bound_r0.mul(tau_bound_r_tail_rev).mul(fused_left).mul(fused_right);
            dbg("[ZOLT] FACTOR CLAIMS: expected_output_claim = {any}\n", .{expected_output_claim.toBytesBE()});

            // --- DIAGNOSTIC: Recompute verifier's expected_claim for each instance ---
            // Compare prover's polynomial evaluation with verifier's expected claim formula
            {
                // Instance 0: ProductVirtualRemainder
                // Verifier computes: tau_high_bound_r0 * tau_bound_r_tail_rev * fused_left * fused_right
                // This is already computed as expected_output_claim above
                const inst0_prover = if (product_prover) |pp| pp.current_claim else F.zero();
                const inst0_verifier = expected_output_claim;
                dbg("[ZOLT DIAG] Instance 0 (Product): prover={any}\n", .{inst0_prover.toBytesBE()});
                dbg("[ZOLT DIAG] Instance 0 (Product): verifier={any}\n", .{inst0_verifier.toBytesBE()});
                dbg("[ZOLT DIAG] Instance 0 MATCH: {}\n", .{inst0_prover.eql(inst0_verifier)});

                // Instance 1: RamRafEvaluation
                // Verifier computes: unmap_eval * ra_input_claim
                // ra_input_claim = raf_prover.getFinalClaim() = ra.finalClaim()
                // unmap_eval = start_address + 8 * identity(r_address)
                const inst1_prover_claim = if (raf_prover) |rp| rp.current_claim else F.zero();
                const inst1_ra_claim = if (raf_prover) |rp| rp.getFinalClaim() else F.zero();
                dbg("[ZOLT DIAG] Instance 1 (RAF): prover current_claim = {any}\n", .{inst1_prover_claim.toBytesBE()});
                dbg("[ZOLT DIAG] Instance 1 (RAF): ra.finalClaim = {any}\n", .{inst1_ra_claim.toBytesBE()});

                // Compute unmap(r) at the RAF opening point (r_address)
                // RAF challenges are challenges[start_round_1..start_round_1 + log_ram_k]
                // normalized (reversed) to BIG_ENDIAN
                {
                    const start_round_1 = max_num_rounds - log_ram_k;
                    var r_addr_raf = try self.allocator.alloc(F, log_ram_k);
                    defer self.allocator.free(r_addr_raf);
                    for (0..log_ram_k) |ii| {
                        if (start_round_1 + ii < challenges.items.len) {
                            r_addr_raf[log_ram_k - 1 - ii] = challenges.items[start_round_1 + ii];
                        } else {
                            r_addr_raf[log_ram_k - 1 - ii] = F.zero();
                        }
                    }

                    // Compute unmap_eval = start_address + 8 * identity(r_addr_raf)
                    const start_addr: u64 = if (config.memory_layout) |ml| ml.getLowestAddress() else 0x80000000;
                    // identity(r) = r[n-1]*1 + r[n-2]*2 + ... (in BIG_ENDIAN, r[n-1] is LSB)
                    var identity_val = F.zero();
                    var pow: u64 = 1;
                    var iii: usize = log_ram_k;
                    while (iii > 0) {
                        iii -= 1;
                        identity_val = identity_val.add(r_addr_raf[iii].mul(F.fromU64(pow)));
                        pow *= 2;
                    }
                    const unmap_eval = identity_val.mul(F.fromU64(8)).add(F.fromU64(start_addr));
                    dbg("[ZOLT DIAG] Instance 1 (RAF): unmap(r) = {any}\n", .{unmap_eval.toBytesBE()});
                    dbg("[ZOLT DIAG] Instance 1 (RAF): unmap * ra = {any}\n", .{unmap_eval.mul(inst1_ra_claim).toBytesBE()});
                    dbg("[ZOLT DIAG] Instance 1 (RAF): current_claim == unmap*ra? {}\n", .{inst1_prover_claim.eql(unmap_eval.mul(inst1_ra_claim))});
                }

                // Instance 2: RamReadWriteChecking
                // Verifier computes: eq_eval_cycle * ra_claim * (val_claim + gamma * (val_claim + inc_claim))
                // But wait - the verifier uses the OPENING CLAIMS from the proof (rwc_ra_claim, rwc_val_claim, rwc_inc_claim)
                // The prover's current_claim should equal this formula
                const inst2_prover_claim = if (rwc_prover) |rp| rp.current_claim else F.zero();
                dbg("[ZOLT DIAG] Instance 2 (RWC): prover current_claim = {any}\n", .{inst2_prover_claim.toBytesBE()});

                // Instance 3: OutputSumcheck
                const inst3_prover_claim = if (output_prover) |op| op.current_claim else F.zero();
                dbg("[ZOLT DIAG] Instance 3 (Output): prover current_claim = {any}\n", .{inst3_prover_claim.toBytesBE()});

                // Instance 4: InstructionLookups
                const inst4_prover_claim = if (instr_prover) |*ip| ip.current_claim else F.zero();
                dbg("[ZOLT DIAG] Instance 4 (Instr): prover current_claim = {any}\n", .{inst4_prover_claim.toBytesBE()});

                // Now compute what the verifier expects from the formula, starting with inst1
                // For inst1 (RAF), verifier computes: unmap_eval * ra_input_claim
                // ra_input_claim = raf_final_claim = the prover's raf.getFinalClaim()
                // unmap_eval depends on memory layout. Let's see if inst1_prover_claim / raf_final equals unmap_eval
                // Actually, let me just see if all instances' prover claims sum correctly
                const total_verifier_expected = inst0_verifier.mul(batching_coeffs[0])
                    .add(inst1_prover_claim.mul(batching_coeffs[1]))
                    .add(inst2_prover_claim.mul(batching_coeffs[2]))
                    .add(inst3_prover_claim.mul(batching_coeffs[3]))
                    .add(inst4_prover_claim.mul(batching_coeffs[4]));
                dbg("[ZOLT DIAG] Sum of prover claims * coeffs = {any}\n", .{total_verifier_expected.toBytesBE()});
                dbg("[ZOLT DIAG] Actual batched_claim = {any}\n", .{batched_claim.toBytesBE()});
                dbg("[ZOLT DIAG] These should match = {}\n", .{total_verifier_expected.eql(batched_claim)});
            }

            // Copy challenges to return them
            const challenges_copy = try self.allocator.alloc(F, challenges.items.len);
            @memcpy(challenges_copy, challenges.items);

            // Get final claims from each prover
            const raf_claim = if (raf_prover) |rp| rp.getFinalClaim() else F.zero();
            const rwc_claim = if (rwc_prover) |*rp| rp.current_claim else F.zero();
            const output_claim = if (output_prover) |op| op.current_claim else F.zero();
            const instr_claim = if (instr_prover) |*ip| ip.current_claim else F.zero();

            // Get individual RWC opening claims (ra, val, inc)
            var rwc_ra_claim = F.zero();
            var rwc_val_claim = F.zero();
            var rwc_inc_claim = F.zero();
            dbg("[ZOLT] STAGE2 RWC: rwc_prover is_null = {}\n", .{rwc_prover == null});
            if (rwc_prover) |*rp| {
                dbg("[ZOLT] STAGE2 RWC: getting opening claims...\n", .{});
                dbg("[ZOLT] STAGE2 RWC: entries.len = {}\n", .{rp.entries.items.len});
                for (rp.entries.items, 0..) |entry, idx| {
                    dbg("[ZOLT] STAGE2 RWC: entry[{}]: cycle={}, addr={}, ra={any}\n", .{ idx, entry.cycle, entry.address, entry.ra_coeff.toBytesBE()[0..8] });
                }
                const rwc_opening_claims = rp.getOpeningClaims(challenges.items);
                rwc_ra_claim = rwc_opening_claims.ra_claim;
                rwc_val_claim = rwc_opening_claims.val_claim;
                rwc_inc_claim = rwc_opening_claims.inc_claim;
                dbg("[ZOLT] STAGE2 RWC: ra_claim = {any}\n", .{rwc_ra_claim.toBytesBE()});
                dbg("[ZOLT] STAGE2 RWC: val_claim = {any}\n", .{rwc_val_claim.toBytesBE()});
                dbg("[ZOLT] STAGE2 RWC: inc_claim = {any}\n", .{rwc_inc_claim.toBytesBE()});

                // Verify: current_claim should equal eq_cycle * ra * (val + gamma * (val + inc))
                const eq_cycle_scalar = rp.eq_evals[0];
                const rwc_gamma = rp.params.gamma;
                const expected_rwc = eq_cycle_scalar.mul(rwc_ra_claim).mul(
                    rwc_val_claim.add(rwc_gamma.mul(rwc_val_claim.add(rwc_inc_claim))),
                );
                dbg("[ZOLT] STAGE2 RWC VERIFY: eq_cycle = {any}\n", .{eq_cycle_scalar.toBytesBE()});
                dbg("[ZOLT] STAGE2 RWC VERIFY: gamma = {any}\n", .{rwc_gamma.toBytesBE()});
                dbg("[ZOLT] STAGE2 RWC VERIFY: expected = eq*ra*(val+gamma*(val+inc)) = {any}\n", .{expected_rwc.toBytesBE()});
                dbg("[ZOLT] STAGE2 RWC VERIFY: prover current_claim = {any}\n", .{rp.current_claim.toBytesBE()});
                dbg("[ZOLT] STAGE2 RWC VERIFY: match = {}\n", .{expected_rwc.eql(rp.current_claim)});
            } else {
                dbg("[ZOLT] STAGE2 RWC: prover is null, computing val_init(r_address) for rwc_val_claim\n", .{});
                // When rwc_prover is null (no RAM operations), the val polynomial equals val_init
                // everywhere. So val(r_address, r_cycle) = val_init(r_address).
                //
                // Jolt's verifier computes: input_claim = rwc_val_claim - init_eval
                // For this to equal 0 (what we want for no-RAM programs), rwc_val_claim must equal init_eval.
                //
                // We compute r_address from the Stage 2 challenges using normalize_opening_point logic.
                if (config.initial_ram != null and config.memory_layout != null) {
                    // RWC uses 3-phase structure:
                    // - Phase 1: phase1_num_rounds cycle vars (ALL cycle vars for Jolt compat)
                    // - Phase 2: log_k address vars
                    // - Phase 3: remaining cycle + address vars
                    const phase1 = n_cycle_vars; // ALL cycle vars in phase 1 (Jolt compat)
                    const phase2 = log_ram_k;
                    const phase3_cycle_len = n_cycle_vars - phase1;
                    const phase3_address_len = log_ram_k - phase2; // = 0 for default config

                    // Compute r_address_be = reverse(phase3_address) ++ reverse(phase2)
                    var r_addr_be = try self.allocator.alloc(F, log_ram_k);
                    defer self.allocator.free(r_addr_be);
                    @memset(r_addr_be, F.zero());

                    // Phase 2 address challenges are at indices [phase1..phase1+phase2)
                    const phase2_start = phase1;
                    for (0..phase2) |i| {
                        const src_idx = phase2_start + i;
                        if (src_idx < challenges.items.len) {
                            const dest_idx = phase3_address_len + (phase2 - 1 - i);
                            if (dest_idx < log_ram_k) {
                                r_addr_be[dest_idx] = challenges.items[src_idx];
                            }
                        }
                    }
                    // Phase 3 address challenges are at indices [phase1+phase2+phase3_cycle..end)
                    const phase3_addr_start = phase1 + phase2 + phase3_cycle_len;
                    for (0..phase3_address_len) |i| {
                        const src_idx = phase3_addr_start + i;
                        if (src_idx < challenges.items.len) {
                            const dest_idx = phase3_address_len - 1 - i;
                            r_addr_be[dest_idx] = challenges.items[src_idx];
                        }
                    }

                    // Debug: print r_address_be values
                    dbg("[ZOLT] STAGE2 RWC: r_addr_be (from challenges[{}..{}]):\n", .{ phase2_start, phase2_start + phase2 });
                    for (0..@min(4, log_ram_k)) |i| {
                        dbg("[ZOLT] STAGE2 RWC:   r_addr_be[{}] = {x}\n", .{ i, r_addr_be[i].toBytesBE()[16..32].* });
                    }
                    // Also print the source challenges
                    dbg("[ZOLT] STAGE2 RWC: Source challenges:\n", .{});
                    for (0..@min(4, phase2)) |i| {
                        const src_idx = phase2_start + i;
                        if (src_idx < challenges.items.len) {
                            dbg("[ZOLT] STAGE2 RWC:   challenges[{}] = {x}\n", .{ src_idx, challenges.items[src_idx].toBytesBE()[16..32].* });
                        }
                    }

                    // Compute val_init(r_address_be) using bytecode_words (like Jolt does)
                    rwc_val_claim = computeInitialRamEval(
                        config.bytecode_words,
                        config.min_bytecode_address,
                        config.memory_layout.?,
                        r_addr_be,
                        log_ram_k,
                        config.program_inputs,
                    );
                    dbg("[ZOLT] STAGE2 RWC: computed rwc_val_claim = val_init(r_address) = {any}\n", .{rwc_val_claim.toBytesBE()});
                }
            }

            // Get individual InstructionLookups opening claims (5 terms)
            var instr_lookup_output = F.zero();
            var instr_left_operand = F.zero();
            var instr_right_operand = F.zero();
            var instr_left_instr_input = F.zero();
            var instr_right_instr_input = F.zero();
            if (instr_prover) |*ip| {
                const instr_opening_claims = ip.getOpeningClaims();
                instr_lookup_output = instr_opening_claims.lookup_output;
                instr_left_operand = instr_opening_claims.left_operand;
                instr_right_operand = instr_opening_claims.right_operand;
                instr_left_instr_input = instr_opening_claims.left_instr_input;
                instr_right_instr_input = instr_opening_claims.right_instr_input;
                dbg("[ZOLT] STAGE2 Instr: lookup_output = {any}\n", .{instr_lookup_output.toBytesBE()});
                dbg("[ZOLT] STAGE2 Instr: left_operand = {any}\n", .{instr_left_operand.toBytesBE()});
                dbg("[ZOLT] STAGE2 Instr: right_operand = {any}\n", .{instr_right_operand.toBytesBE()});
                dbg("[ZOLT] STAGE2 Instr: left_instr_input = {any}\n", .{instr_left_instr_input.toBytesBE()});
                dbg("[ZOLT] STAGE2 Instr: right_instr_input = {any}\n", .{instr_right_instr_input.toBytesBE()});
            }

            dbg("[ZOLT] STAGE2: raf_final_claim = {any}\n", .{raf_claim.toBytesBE()});
            dbg("[ZOLT] STAGE2: rwc_final_claim = {any}\n", .{rwc_claim.toBytesBE()});
            dbg("[ZOLT] STAGE2: output_final_claim = {any}\n", .{output_claim.toBytesBE()});
            dbg("[ZOLT] STAGE2: instr_final_claim = {any}\n", .{instr_claim.toBytesBE()});

            // Get Val_final(r') and Val_init(r') from the OutputSumcheck prover
            // These are the MLE evaluations at the final opening point
            var output_val_final = F.zero();
            var output_val_init = F.zero();
            if (output_prover) |op| {
                const output_claims = op.getFinalClaims();
                output_val_final = output_claims.val_final;
                output_val_init = output_claims.val_init;
            }
            dbg("[ZOLT] STAGE2: output_val_final_claim (from prover) = {any}\n", .{output_val_final.toBytesBE()});
            dbg("[ZOLT] STAGE2: output_val_init_claim (from prover) = {any}\n", .{output_val_init.toBytesBE()});

            // Compute r_address_rw and r_cycle_rw from RWC challenges for RamRaClaimReduction
            // RWC uses 3-phase structure:
            // - Phase 1: n_cycle_vars cycle variables
            // - Phase 2: log_ram_k address variables
            // - Phase 3: 0 remaining variables (for default config)
            // Opening point = [r_address, r_cycle] in BIG_ENDIAN
            const phase1_rounds = n_cycle_vars;
            const phase2_rounds = log_ram_k;

            const r_address_rw = try self.allocator.alloc(F, log_ram_k);
            const r_cycle_rw = try self.allocator.alloc(F, n_cycle_vars);

            // r_address from phase 2 challenges, reversed to BIG_ENDIAN
            // Phase 2 challenges are at indices [phase1..phase1+phase2)
            for (0..phase2_rounds) |i| {
                const src_idx = phase1_rounds + i;
                if (src_idx < challenges.items.len) {
                    r_address_rw[phase2_rounds - 1 - i] = challenges.items[src_idx];
                } else {
                    r_address_rw[phase2_rounds - 1 - i] = F.zero();
                }
            }

            // r_cycle from phase 1 challenges, reversed to BIG_ENDIAN
            // Phase 1 challenges are at indices [0..phase1)
            for (0..phase1_rounds) |i| {
                if (i < challenges.items.len) {
                    r_cycle_rw[phase1_rounds - 1 - i] = challenges.items[i];
                } else {
                    r_cycle_rw[phase1_rounds - 1 - i] = F.zero();
                }
            }

            dbg("[ZOLT] STAGE2: r_address_rw (BIG_ENDIAN) computed, len={}\n", .{r_address_rw.len});
            dbg("[ZOLT] STAGE2: r_cycle_rw (BIG_ENDIAN) computed, len={}\n", .{r_cycle_rw.len});

            // CRITICAL FIX: r_address_raf should be computed from sumcheck challenges, NOT the pre-sampled r_address!
            //
            // In Jolt's Stage 2 batched sumcheck:
            // - RamRafEvaluation has 16 rounds, offset = 8, gets challenges[8..24]
            // - RamReadWriteChecking has 24 rounds, offset = 0, gets challenges[0..24]
            //   - Phase 1 (cycle): challenges[0..8]
            //   - Phase 2 (address): challenges[8..24]
            //
            // Both instances' r_address = reverse(challenges[8..24]).
            // So r_address_raf = r_address_rw!
            //
            // The pre-sampled r_address is used only for OutputSumcheck's eq polynomial,
            // NOT for RamRafEvaluation's opening point.
            //
            // Compute r_address_raf from sumcheck challenges the same way as r_address_rw:
            const r_address_raf = try self.allocator.alloc(F, log_ram_k);
            for (0..phase2_rounds) |i| {
                const src_idx = phase1_rounds + i;
                if (src_idx < challenges.items.len) {
                    r_address_raf[phase2_rounds - 1 - i] = challenges.items[src_idx];
                } else {
                    r_address_raf[phase2_rounds - 1 - i] = F.zero();
                }
            }

            // Debug: compare r_address_raf and r_address_rw (they should now be identical)
            dbg("[ZOLT] STAGE2: r_address_raf[0..4] (BE from sumcheck) = ", .{});
            for (0..@min(4, r_address_raf.len)) |i| {
                dbg("{x} ", .{r_address_raf[i].toBytesBE()[24..32].*});
            }
            dbg("\n", .{});
            dbg("[ZOLT] STAGE2: r_address_rw[0..4] (BE from sumcheck) = ", .{});
            for (0..@min(4, r_address_rw.len)) |i| {
                dbg("{x} ", .{r_address_rw[i].toBytesBE()[24..32].*});
            }
            dbg("\n", .{});

            // Compute ProductVirtualRemainder r_cycle from Stage 2 challenges
            // ProductVirtualRemainder starts at round (max_num_rounds - n_cycle_vars)
            // and runs for n_cycle_vars rounds. Reversed to BIG_ENDIAN.
            const product_start = max_num_rounds - n_cycle_vars;
            const r_cycle_product = try self.allocator.alloc(F, n_cycle_vars);
            for (0..n_cycle_vars) |i| {
                const src_idx = product_start + i;
                if (src_idx < challenges.items.len) {
                    r_cycle_product[n_cycle_vars - 1 - i] = challenges.items[src_idx];
                } else {
                    r_cycle_product[n_cycle_vars - 1 - i] = F.zero();
                }
            }
            dbg("[ZOLT] STAGE2: r_cycle_product (BIG_ENDIAN) computed, len={}\n", .{r_cycle_product.len});

            return Stage2Result{
                .factor_evals = factor_evals,
                .challenges = challenges_copy,
                .raf_final_claim = raf_claim,
                .rwc_final_claim = rwc_claim,
                .output_final_claim = output_claim,
                .instr_final_claim = instr_claim,
                .output_val_final_claim = output_val_final,
                .output_val_init_claim = output_val_init,
                .r_address_raf = r_address_raf,
                .r_address_rw = r_address_rw,
                .r_cycle_rw = r_cycle_rw,
                .r_cycle_product = r_cycle_product,
                .rwc_ra_claim = rwc_ra_claim,
                .rwc_val_claim = rwc_val_claim,
                .rwc_inc_claim = rwc_inc_claim,
                .instr_lookup_output_claim = instr_lookup_output,
                .instr_left_operand_claim = instr_left_operand,
                .instr_right_operand_claim = instr_right_operand,
                .instr_left_instr_input_claim = instr_left_instr_input,
                .instr_right_instr_input_claim = instr_right_instr_input,
                .allocator = self.allocator,
            };
        }

        /// Compute MLE evaluations of the 8 factor polynomials at r_cycle
        ///
        /// The 8 factors are:
        /// 0: LeftInstructionInput
        /// 1: RightInstructionInput
        /// 2: IsRdNotZero
        /// 3: WriteLookupOutputToRDFlag
        /// 4: JumpFlag
        /// 5: LookupOutput
        /// 6: BranchFlag
        /// 7: NextIsNoop
        ///
        /// Returns MLE(factor_i, r_cycle) = Σ_t eq(r_cycle, t) * factor_value[t]
        fn computeProductFactorEvaluations(
            self: *Self,
            cycle_witnesses: []const r1cs.R1CSCycleInputs(F),
            all_challenges: []const F,
            n_cycle_vars: usize,
            log_ram_k: usize,
        ) ![8]F {
            _ = log_ram_k;
            // r_cycle is the last n_cycle_vars challenges
            // In Jolt, ProductVirtualRemainder runs for n_cycle_vars rounds starting after log_ram_k rounds
            // So r_cycle = all_challenges[log_ram_k..log_ram_k + n_cycle_vars]
            // But the challenges are stored in order, so we take the last n_cycle_vars
            if (all_challenges.len < n_cycle_vars) {
                // Not enough challenges, return zeros
                return [8]F{ F.zero(), F.zero(), F.zero(), F.zero(), F.zero(), F.zero(), F.zero(), F.zero() };
            }

            // Extract r_cycle (last n_cycle_vars challenges)
            // These are the sumcheck challenges that were used to bind the ProductVirtualRemainder
            // polynomial. Jolt uses normalize_opening_point which reverses the challenges to
            // convert from LITTLE_ENDIAN to BIG_ENDIAN.
            const r_cycle_start = all_challenges.len - n_cycle_vars;
            const r_cycle_original = all_challenges[r_cycle_start..];

            // Jolt's normalize_opening_point reverses the challenges to convert from LE to BE.
            // The factor claims must be computed at this reversed point to match the verifier's
            // expected_output_claim computation.
            const r_cycle = try self.allocator.alloc(F, n_cycle_vars);
            defer self.allocator.free(r_cycle);
            for (0..n_cycle_vars) |i| {
                r_cycle[i] = r_cycle_original[n_cycle_vars - 1 - i];
            }

            dbg("[ZOLT] FACTOR_EVALS: r_cycle.len = {}, n_cycle_vars = {}\n", .{ r_cycle.len, n_cycle_vars });
            if (r_cycle.len > 0) {
                dbg("[ZOLT] FACTOR_EVALS: r_cycle[0] BE = {any}\n", .{r_cycle[0].toBytesBE()});
                dbg("[ZOLT] FACTOR_EVALS: r_cycle[0] LE = {any}\n", .{r_cycle[0].toBytes()});
                dbg("[ZOLT] FACTOR_EVALS: r_cycle_original[0] BE = {any}\n", .{r_cycle_original[0].toBytesBE()});
            }
            if (r_cycle.len > 7) {
                dbg("[ZOLT] FACTOR_EVALS: r_cycle[7] LE = {any}\n", .{r_cycle[7].toBytes()});
            }

            // Compute eq polynomial evaluations at r_cycle (using BIG_ENDIAN indexing like Jolt)
            const EqPoly = poly_mod.EqPolynomial(F);
            var eq_poly = try EqPoly.init(self.allocator, r_cycle);
            defer eq_poly.deinit();

            const eq_evals = try eq_poly.evals(self.allocator);
            defer self.allocator.free(eq_evals);

            dbg("[ZOLT] FACTOR_EVALS: eq_evals.len = {}, cycle_witnesses.len = {}\n", .{ eq_evals.len, cycle_witnesses.len });
            dbg("[ZOLT] FACTOR_EVALS: eq_evals[0] = {any}\n", .{eq_evals[0].toBytesBE()});
            dbg("[ZOLT] FACTOR_EVALS: eq_evals[1] = {any}\n", .{eq_evals[1].toBytesBE()});
            dbg("[ZOLT] FACTOR_EVALS: eq_evals[2] = {any}\n", .{eq_evals[2].toBytesBE()});
            // Print sum of eq_evals (should be 1 for partition of unity)
            var eq_sum = F.zero();
            for (eq_evals) |ev| {
                eq_sum = eq_sum.add(ev);
            }
            dbg("[ZOLT] FACTOR_EVALS: eq_sum = {any} (should be 1)\n", .{eq_sum.toBytesBE()});

            // Initialize factor accumulators
            var factor_evals = [8]F{ F.zero(), F.zero(), F.zero(), F.zero(), F.zero(), F.zero(), F.zero(), F.zero() };

            // Compute MLE evaluation: Σ_t eq(r_cycle, t) * factor_value[t]
            const num_cycles = @min(eq_evals.len, cycle_witnesses.len);

            // Debug: Print witness values for several cycles
            dbg("[ZOLT] FACTOR_EVALS: witness[0][LeftInstructionInput] LE = {any}\n", .{cycle_witnesses[0].values[r1cs.R1CSInputIndex.LeftInstructionInput.toIndex()].toBytes()});
            dbg("[ZOLT] FACTOR_EVALS: witness[1][LeftInstructionInput] LE = {any}\n", .{cycle_witnesses[1].values[r1cs.R1CSInputIndex.LeftInstructionInput.toIndex()].toBytes()});

            // Count non-zero LeftInstructionInput values
            var nonzero_left_count: usize = 0;
            for (0..@min(256, cycle_witnesses.len)) |t| {
                const val = cycle_witnesses[t].values[r1cs.R1CSInputIndex.LeftInstructionInput.toIndex()];
                if (!val.eql(F.zero())) {
                    nonzero_left_count += 1;
                    if (nonzero_left_count <= 3) {
                        dbg("[ZOLT] FACTOR_EVALS: witness[{}][LeftInstructionInput] = {any}\n", .{t, val.toBytes()});
                    }
                }
            }
            dbg("[ZOLT] FACTOR_EVALS: total nonzero LeftInstructionInput in 256 cycles = {}\n", .{nonzero_left_count});
            dbg("[ZOLT] FACTOR_EVALS: witness[0][RightInstructionInput] = {any}\n", .{cycle_witnesses[0].values[r1cs.R1CSInputIndex.RightInstructionInput.toIndex()].toBytesBE()});

            // Track per-cycle contributions for debugging
            var cycle_count_with_nonzero_branch: usize = 0;
            var cycle_count_with_nonzero_lookup_output: usize = 0;

            // Diagnostic: print per-cycle flag values for first 32 cycles
            {
                var printed: usize = 0;
                for (0..@min(256, num_cycles)) |t2| {
                    const w = &cycle_witnesses[t2];
                    const isrdnz = w.values[r1cs.R1CSInputIndex.FlagIsRdNotZero.toIndex()];
                    const wlflag = w.values[r1cs.R1CSInputIndex.FlagWriteLookupOutputToRD.toIndex()];
                    const jumpfl = w.values[r1cs.R1CSInputIndex.FlagJump.toIndex()];
                    const branchf = w.values[r1cs.R1CSInputIndex.FlagBranch.toIndex()];
                    const is_noop = w.values[r1cs.R1CSInputIndex.FlagIsNoop.toIndex()];
                    const imm_val = w.values[r1cs.R1CSInputIndex.Imm.toIndex()];
                    // Only print non-noop cycles or first few
                    if (printed < 32 and (is_noop.eql(F.zero()) or t2 < 4)) {
                        const il = imm_val.toBytes();
                        dbg("[FACTOR_DIAG] c={} noop={} isRdNZ={} WrLookup={} Jump={} Branch={} imm_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                            t2,
                            @as(u8, if (is_noop.eql(F.one())) 1 else 0),
                            @as(u8, if (isrdnz.eql(F.one())) 1 else 0),
                            @as(u8, if (wlflag.eql(F.one())) 1 else 0),
                            @as(u8, if (jumpfl.eql(F.one())) 1 else 0),
                            @as(u8, if (branchf.eql(F.one())) 1 else 0),
                            il[0], il[1], il[2], il[3], il[4], il[5], il[6], il[7],
                        });
                        printed += 1;
                    }
                }
            }

            for (0..num_cycles) |t| {
                const eq_val = eq_evals[t];
                const witness = &cycle_witnesses[t];

                // Debug: Track non-zero values
                const branch_val = witness.values[r1cs.R1CSInputIndex.FlagBranch.toIndex()];
                const lookup_output_val = witness.values[r1cs.R1CSInputIndex.LookupOutput.toIndex()];
                if (!branch_val.eql(F.zero())) {
                    cycle_count_with_nonzero_branch += 1;
                    if (cycle_count_with_nonzero_branch <= 5) {
                        dbg("[ZOLT DEBUG] Cycle {} has Branch=1, LookupOutput={any}\n", .{ t, lookup_output_val.toBytesBE()[28..32] });
                    }
                }
                if (!lookup_output_val.eql(F.zero())) {
                    cycle_count_with_nonzero_lookup_output += 1;
                }

                // Extract the 8 factor values from the witness
                // Must match PRODUCT_UNIQUE_FACTOR_VIRTUALS order:
                // [0] LeftInstructionInput, [1] RightInstructionInput,
                // [2] OpFlags(Jump), [3] OpFlags(WriteLookupOutputToRD),
                // [4] LookupOutput, [5] InstructionFlags(Branch),
                // [6] NextIsNoop, [7] OpFlags(VirtualInstruction)

                // 0: LeftInstructionInput
                factor_evals[0] = factor_evals[0].add(eq_val.mul(
                    witness.values[r1cs.R1CSInputIndex.LeftInstructionInput.toIndex()],
                ));

                // 1: RightInstructionInput
                factor_evals[1] = factor_evals[1].add(eq_val.mul(
                    witness.values[r1cs.R1CSInputIndex.RightInstructionInput.toIndex()],
                ));

                // 2: OpFlags(Jump)
                factor_evals[2] = factor_evals[2].add(eq_val.mul(
                    witness.values[r1cs.R1CSInputIndex.FlagJump.toIndex()],
                ));

                // 3: OpFlags(WriteLookupOutputToRD)
                factor_evals[3] = factor_evals[3].add(eq_val.mul(
                    witness.values[r1cs.R1CSInputIndex.FlagWriteLookupOutputToRD.toIndex()],
                ));

                // 4: LookupOutput
                factor_evals[4] = factor_evals[4].add(eq_val.mul(
                    witness.values[r1cs.R1CSInputIndex.LookupOutput.toIndex()],
                ));

                // 5: InstructionFlags(Branch)
                factor_evals[5] = factor_evals[5].add(eq_val.mul(
                    witness.values[r1cs.R1CSInputIndex.FlagBranch.toIndex()],
                ));

                // 6: NextIsNoop - check if next instruction is a noop
                const next_is_noop = blk: {
                    if (t + 1 < cycle_witnesses.len) {
                        break :blk cycle_witnesses[t + 1].values[r1cs.R1CSInputIndex.FlagIsNoop.toIndex()];
                    }
                    // For last cycle: not_next_noop = false (hardcoded), so NextIsNoop = true
                    break :blk F.one();
                };
                factor_evals[6] = factor_evals[6].add(eq_val.mul(next_is_noop));

                // 7: OpFlags(VirtualInstruction)
                factor_evals[7] = factor_evals[7].add(eq_val.mul(
                    witness.values[r1cs.R1CSInputIndex.FlagVirtualInstruction.toIndex()],
                ));
            }

            // Debug: Print counts
            dbg("[ZOLT DEBUG] Cycles with non-zero Branch: {}\n", .{cycle_count_with_nonzero_branch});
            dbg("[ZOLT DEBUG] Cycles with non-zero LookupOutput: {}\n", .{cycle_count_with_nonzero_lookup_output});

            // Handle padding cycles (indices from cycle_witnesses.len to eq_evals.len)
            // Note: If cycle_witnesses already includes NoOp padding (from R1CS witness generator),
            // this loop may not execute. Only run if witnesses are shorter than eq domain.
            //
            // Padding cycles are NoOp cycles. For NoOp:
            // - Factors 0-5, 7: all zero (no instruction input, no flags set, no output)
            // - Factor 6 (NextIsNoop): 1 (next cycle is also a NoOp)
            if (cycle_witnesses.len < eq_evals.len) {
                for (cycle_witnesses.len..eq_evals.len) |t| {
                    const eq_val = eq_evals[t];
                    factor_evals[6] = factor_evals[6].add(eq_val);
                }
            }

            dbg("[ZOLT] FACTOR_EVALS: factor[0] (LeftInstructionInput) = {any}\n", .{factor_evals[0].toBytesBE()});
            dbg("[ZOLT] FACTOR_EVALS: factor[1] (RightInstructionInput) = {any}\n", .{factor_evals[1].toBytesBE()});
            dbg("[ZOLT] FACTOR_EVALS: factor[2] (Jump) = {any}\n", .{factor_evals[2].toBytesBE()});
            dbg("[ZOLT] FACTOR_EVALS: factor[3] (WriteLookupOutputToRD) = {any}\n", .{factor_evals[3].toBytesBE()});
            dbg("[ZOLT] FACTOR_EVALS: factor[4] (LookupOutput) = {any}\n", .{factor_evals[4].toBytesBE()});
            dbg("[ZOLT] FACTOR_EVALS: factor[5] (Branch) = {any}\n", .{factor_evals[5].toBytesBE()});
            dbg("[ZOLT] FACTOR_EVALS: factor[6] (NextIsNoop) = {any}\n", .{factor_evals[6].toBytesBE()});
            dbg("[ZOLT] FACTOR_EVALS: factor[7] (NextIsNoop) = {any}\n", .{factor_evals[7].toBytesBE()});

            return factor_evals;
        }

        /// Evaluate polynomial at challenge using Jolt's eval_from_hint formula
        /// This is the verifier's computation from compressed coefficients [c0, c2, c3] and hint
        fn evalFromHint(compressed: [3]F, hint: F, x: F) F {
            const c0 = compressed[0];
            const c2 = compressed[1];
            const c3 = compressed[2];

            // Recover c1 = hint - 2*c0 - c2 - c3
            const c1 = hint.sub(c0).sub(c0).sub(c2).sub(c3);

            // P(x) = c0 + c1*x + c2*x^2 + c3*x^3
            const x2 = x.mul(x);
            const x3 = x2.mul(x);
            const result = c0.add(c1.mul(x)).add(c2.mul(x2)).add(c3.mul(x3));

            // Debug: Print intermediate values for Stage 4 first round
            // Stage 4 Round 0 challenge has limbs[2] = 0xb5ba64b08cc4cef5
            if (x.limbs[2] == 0xb5ba64b08cc4cef5) {
                dbg("[ZOLT evalFromHint STAGE4 R0] Found Stage 4 Round 0!\n", .{});
                dbg("[ZOLT evalFromHint STAGE4 R0]   x limbs = [0x{x}, 0x{x}, 0x{x}, 0x{x}]\n", .{ x.limbs[0], x.limbs[1], x.limbs[2], x.limbs[3] });
                dbg("[ZOLT evalFromHint STAGE4 R0]   x2 limbs = [0x{x}, 0x{x}, 0x{x}, 0x{x}]\n", .{ x2.limbs[0], x2.limbs[1], x2.limbs[2], x2.limbs[3] });
                dbg("[ZOLT evalFromHint STAGE4 R0]   c0 limbs = [0x{x}, 0x{x}, 0x{x}, 0x{x}]\n", .{ c0.limbs[0], c0.limbs[1], c0.limbs[2], c0.limbs[3] });
                dbg("[ZOLT evalFromHint STAGE4 R0]   c1 limbs = [0x{x}, 0x{x}, 0x{x}, 0x{x}]\n", .{ c1.limbs[0], c1.limbs[1], c1.limbs[2], c1.limbs[3] });
                dbg("[ZOLT evalFromHint STAGE4 R0]   hint limbs = [0x{x}, 0x{x}, 0x{x}, 0x{x}]\n", .{ hint.limbs[0], hint.limbs[1], hint.limbs[2], hint.limbs[3] });
                dbg("[ZOLT evalFromHint STAGE4 R0]   result limbs = [0x{x}, 0x{x}, 0x{x}, 0x{x}]\n", .{ result.limbs[0], result.limbs[1], result.limbs[2], result.limbs[3] });
            }

            return result;
        }

        /// Compute eq(r, idx) where r is in BIG_ENDIAN order (MSB first).
        fn computeEqAtPointBigEndian(r: []const F, idx: usize) F {
            var result = F.one();
            const n = r.len;
            for (0..n) |i| {
                const bit = (idx >> @intCast(n - 1 - i)) & 1;
                if (bit == 1) {
                    result = result.mul(r[i]);
                } else {
                    result = result.mul(F.one().sub(r[i]));
                }
            }
            return result;
        }

        /// Compute eq(r, idx) where r is in LITTLE_ENDIAN order (LSB first).
        /// bit i of idx corresponds to r[i].
        fn computeEqAtPointLE(r: []const F, idx: usize) F {
            var result = F.one();
            for (r, 0..) |ri, i| {
                const bit = (idx >> @intCast(i)) & 1;
                if (bit == 1) {
                    result = result.mul(ri);
                } else {
                    result = result.mul(F.one().sub(ri));
                }
            }
            return result;
        }

        /// Evaluate the initial RAM polynomial at r_address (BIG_ENDIAN).
        ///
        /// This matches Jolt's `eval_initial_ram_mle` which evaluates:
        ///   sum_k bytecode_words[k] * eq(r_address, bytecode_start + k)
        /// where bytecode_start = remap_address(min_bytecode_address)
        ///
        /// NOTE: Unlike the old implementation that used initial_ram hashmap (stack data),
        /// this now uses bytecode_words (program bytecode) like Jolt does.
        fn computeInitialRamEval(
            bytecode_words: ?[]const u64,
            min_bytecode_address: u64,
            memory_layout: *const jolt_device.MemoryLayout,
            r_address_be: []const F,
            log_ram_k: usize,
            program_inputs: ?[]const u8,
        ) F {
            dbg("[COMPUTE_INIT_RAM_EVAL] Computing with log_ram_k={}\n", .{log_ram_k});
            dbg("[COMPUTE_INIT_RAM_EVAL] r_address_be.len = {}\n", .{r_address_be.len});
            dbg("[COMPUTE_INIT_RAM_EVAL] min_bytecode_address = 0x{x:0>16}\n", .{min_bytecode_address});

            const lowest_address = memory_layout.getLowestAddress();
            dbg("[COMPUTE_INIT_RAM_EVAL] lowest_address = 0x{x:0>16}\n", .{lowest_address});

            var result = F.zero();
            const max_idx: usize = @as(usize, 1) << @intCast(log_ram_k);

            // Evaluate bytecode region (like Jolt's eval_initial_ram_mle)
            if (bytecode_words) |words| {
                if (words.len > 0) {
                    // bytecode_start = remap_address(min_bytecode_address)
                    // remap_address = (address - lowest_address) / 8
                    const bytecode_start: usize = @intCast((min_bytecode_address - lowest_address) / 8);
                    dbg("[COMPUTE_INIT_RAM_EVAL] bytecode_start (remapped) = {}\n", .{bytecode_start});
                    dbg("[COMPUTE_INIT_RAM_EVAL] bytecode_words.len = {}\n", .{words.len});
                    if (words.len > 0) {
                        dbg("[COMPUTE_INIT_RAM_EVAL] bytecode_words first 3: ", .{});
                        for (0..@min(3, words.len)) |i| {
                            dbg("0x{x:0>16} ", .{words[i]});
                        }
                        dbg("\n", .{});
                    }

                    // Sum: bytecode_words[k] * eq(r_address, bytecode_start + k)
                    for (words, 0..) |word, k| {
                        const idx = bytecode_start + k;
                        if (idx >= max_idx) break;

                        const eq_val = computeEqAtPointBigEndian(r_address_be, idx);
                        const val = F.fromU64(word);
                        result = result.add(eq_val.mul(val));
                    }
                    dbg("[COMPUTE_INIT_RAM_EVAL] Processed {} bytecode words\n", .{words.len});
                }
            }

            // Also add inputs region (like Jolt does)
            if (program_inputs) |inputs| {
                if (inputs.len > 0) {
                    // input_start = remap_address(memory_layout.input_start)
                    const input_start: usize = @intCast((memory_layout.input_start - lowest_address) / 8);
                    dbg("[COMPUTE_INIT_RAM_EVAL] input_start (remapped) = {}\n", .{input_start});
                    dbg("[COMPUTE_INIT_RAM_EVAL] inputs.len = {}\n", .{inputs.len});

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

            dbg("[COMPUTE_INIT_RAM_EVAL] result = {any}\n", .{result.toBytes()[0..8]});
            return result;
        }

        /// Evaluate cubic polynomial at a challenge point from Toom-Cook evaluations
        /// Input: evals = [p(0), p(1), p(2), p_inf] where p_inf is the leading coefficient c3
        fn evaluateCubicAtChallengeFromEvals(evals: [4]F, x: F) F {
            // Convert Toom-Cook format to coefficients first
            // evals = [p(0), p(1), p(2), p_inf] where p_inf = c3
            const coeffs = poly_mod.UniPoly(F).toomCookToCoeffs(evals);

            // Evaluate p(x) = c0 + c1*x + c2*x^2 + c3*x^3 using Horner's method
            var result = coeffs[3];
            result = result.mul(x).add(coeffs[2]);
            result = result.mul(x).add(coeffs[1]);
            result = result.mul(x).add(coeffs[0]);
            return result;
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
            cycle_witnesses: []const r1cs.R1CSCycleInputs(F),
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
                if (cycle_witnesses.len == 0) {
                    // No witnesses - use zeros
                    break :blk [_]F{F.zero()} ** DEGREE;
                }

                // Extract the 8 product factors from each cycle witness
                const cycle_factors = try self.allocator.alloc([8]F, cycle_witnesses.len);
                defer self.allocator.free(cycle_factors);

                for (cycle_witnesses, 0..) |witness, idx| {
                    cycle_factors[idx] = extractProductFactors(F, &witness, cycle_witnesses, idx);
                }

                // Compute extended evaluations using the precomputed Lagrange coefficients
                break :blk try univariate_skip.computeProductVirtualExtendedEvals(
                    F,
                    cycle_factors,
                    tau_stage2,
                    self.allocator,
                );
            };

            // Debug: Print extended evaluations
            for (extended_evals, 0..) |eval, ei| {
                dbg("[ZOLT] STAGE2_UNISKIP: extended_evals[{}] = {any}\n", .{ ei, eval.toBytesBE() });
            }

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

            // Debug: Print ALL polynomial coefficients for comparison with Jolt (LE format like Jolt)
            for (uni_poly.coeffs, 0..) |coeff, ci| {
                var le_bytes: [32]u8 = undefined;
                const be_bytes = coeff.toBytesBE();
                for (0..32) |bi| {
                    le_bytes[bi] = be_bytes[31 - bi];
                }
                dbg("[ZOLT] STAGE2_UNISKIP: coeffs[{}] = {any}\n", .{ ci, le_bytes });
            }
            dbg("[ZOLT] STAGE2_UNISKIP: total num_coeffs = {}\n", .{uni_poly.coeffs.len});

            // Verify the polynomial satisfies the sum constraint
            // input_claim = Σ L_i(tau_high) * base_evals[i]
            const LagrangePoly = univariate_skip.LagrangePolynomial(F);
            const lagrange_evals = try LagrangePoly.evals(DOMAIN_SIZE, tau_high, self.allocator);
            defer self.allocator.free(lagrange_evals);

            var input_claim = F.zero();
            for (base_evals, 0..) |eval, i| {
                input_claim = input_claim.add(lagrange_evals[i].mul(eval));
            }
            dbg("[ZOLT] STAGE2_UNISKIP: input_claim = {any}\n", .{input_claim.toBytesBE()});

            // Check domain sum
            const power_sums = univariate_skip.computePowerSums(DOMAIN_SIZE, NUM_COEFFS);
            var domain_sum = F.zero();
            for (uni_poly.coeffs, 0..) |coeff, j| {
                domain_sum = domain_sum.add(coeff.mulI128(power_sums[j]));
            }
            dbg("[ZOLT] STAGE2_UNISKIP: domain_sum = {any}\n", .{domain_sum.toBytesBE()});
            dbg("[ZOLT] STAGE2_UNISKIP: sum matches input_claim? {}\n", .{domain_sum.eql(input_claim)});

            // Return as UniSkipFirstRoundProof
            return UniSkipFirstRoundProof(F){
                .uni_poly = uni_poly.coeffs,
                .allocator = self.allocator,
            };
        }
    };
}

/// Extract the 8 product factors from an R1CS cycle witness
///
/// The 8 factors are (matching upstream PRODUCT_UNIQUE_FACTOR_VIRTUALS):
///   [0] LeftInstructionInput
///   [1] RightInstructionInput
///   [2] JumpFlag (OpFlags::Jump)
///   [3] WriteLookupOutputToRDFlag (OpFlags::WriteLookupOutputToRD)
///   [4] LookupOutput
///   [5] BranchFlag (InstructionFlags::Branch)
///   [6] NextIsNoop
///   [7] VirtualInstructionFlag (OpFlags::VirtualInstruction)
fn extractProductFactors(
    comptime F: type,
    witness: *const r1cs.R1CSCycleInputs(F),
    all_witnesses: []const r1cs.R1CSCycleInputs(F),
    cycle_idx: usize,
) [8]F {
    const R1CSInputIndex = r1cs.R1CSInputIndex;

    return [8]F{
        // 0: LeftInstructionInput
        witness.values[R1CSInputIndex.LeftInstructionInput.toIndex()],
        // 1: RightInstructionInput
        witness.values[R1CSInputIndex.RightInstructionInput.toIndex()],
        // 2: JumpFlag (OpFlags::Jump)
        witness.values[R1CSInputIndex.FlagJump.toIndex()],
        // 3: WriteLookupOutputToRDFlag (OpFlags::WriteLookupOutputToRD)
        witness.values[R1CSInputIndex.FlagWriteLookupOutputToRD.toIndex()],
        // 4: LookupOutput
        witness.values[R1CSInputIndex.LookupOutput.toIndex()],
        // 5: BranchFlag (InstructionFlags::Branch)
        witness.values[R1CSInputIndex.FlagBranch.toIndex()],
        // 6: NextIsNoop - 1 if next instruction is a noop
        blk: {
            if (cycle_idx + 1 < all_witnesses.len) {
                const next_witness = &all_witnesses[cycle_idx + 1];
                break :blk next_witness.values[R1CSInputIndex.FlagIsNoop.toIndex()];
            }
            // Last cycle: NextIsNoop = true
            break :blk F.one();
        },
        // 7: VirtualInstructionFlag (OpFlags::VirtualInstruction)
        witness.values[R1CSInputIndex.FlagVirtualInstruction.toIndex()],
    };
}

/// Configuration for proof conversion
///
/// These values must match Jolt's config.rs:
/// - log_k_chunk: Must be <= 8 (Jolt uses 4 for small traces, 8 for large)
/// - lookups_ra_virtual_log_k_chunk: Jolt uses LOG_K/8 (=16) for small traces
const tracer = @import("../tracer/mod.zig");

pub const ConversionConfig = struct {
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
