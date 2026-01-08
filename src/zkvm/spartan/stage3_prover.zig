//! Stage 3 Batched Sumcheck Prover for Jolt Compatibility
//!
//! Stage 3 in Jolt consists of 3 batched sumcheck instances:
//! 1. ShiftSumcheck - proves shift polynomial relations (degree 2)
//! 2. InstructionInputSumcheck - proves operand computation (degree 3)
//! 3. RegistersClaimReduction - reduces register value claims (degree 2)
//!
//! All three instances have n_cycle_vars rounds.
//!
//! ## Expected Output Claim Formulas
//!
//! ShiftSumcheck:
//!   (gamma^0 * upc + gamma^1 * pc + gamma^2 * virt + gamma^3 * first) * eq+1(r_outer, r)
//!   + gamma^4 * (1 - noop) * eq+1(r_product, r)
//!
//! InstructionInputSumcheck:
//!   (eq(r, r_stage1) + gamma^2 * eq(r, r_stage2)) * (right + gamma * left)
//!   where left = left_is_rs1 * rs1 + left_is_pc * pc
//!         right = right_is_rs2 * rs2 + right_is_imm * imm
//!
//! RegistersClaimReduction:
//!   eq(r, r_spartan) * (rd + gamma * rs1 + gamma^2 * rs2)

const std = @import("std");
const Allocator = std.mem.Allocator;
const poly_mod = @import("../../poly/mod.zig");
const transcripts = @import("../../transcripts/mod.zig");
const jolt_types = @import("../jolt_types.zig");
const r1cs = @import("../r1cs/mod.zig");
const R1CSInputIndex = r1cs.R1CSInputIndex;
const instruction_mod = @import("../instruction/mod.zig");

/// Stage 3 prover result
pub fn Stage3Result(comptime F: type) type {
    return struct {
        const Self = @This();

        /// All sumcheck challenges (n_cycle_vars of them)
        challenges: []F,
        /// Final round claims for batched sumcheck
        /// These are p(r_n) for each polynomial in the last round
        shift_final_claim: F,
        instr_final_claim: F,
        reg_final_claim: F,

        /// Shift sumcheck opening claims
        shift_unexpanded_pc_claim: F,
        shift_pc_claim: F,
        shift_is_virtual_claim: F,
        shift_is_first_in_sequence_claim: F,
        shift_is_noop_claim: F,
        /// InstructionInput sumcheck opening claims
        instr_left_is_rs1_claim: F,
        instr_rs1_value_claim: F,
        instr_left_is_pc_claim: F,
        instr_unexpanded_pc_claim: F,
        instr_right_is_rs2_claim: F,
        instr_rs2_value_claim: F,
        instr_right_is_imm_claim: F,
        instr_imm_claim: F,
        /// RegistersClaimReduction opening claims
        reg_rd_write_value_claim: F,
        reg_rs1_value_claim: F,
        reg_rs2_value_claim: F,

        allocator: Allocator,

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.challenges);
        }
    };
}

/// Generate Stage 3 batched sumcheck proof
pub fn Stage3Prover(comptime F: type) type {
    return struct {
        const Self = @This();
        const SumcheckInstanceProof = jolt_types.SumcheckInstanceProof;
        const OpeningClaims = jolt_types.OpeningClaims;
        const Blake2bTranscript = transcripts.Blake2bTranscript;
        const EqPolynomial = poly_mod.EqPolynomial;
        const EqPlusOnePolynomial = poly_mod.EqPlusOnePolynomial;
        const MleEvaluation = poly_mod.MleEvaluation;

        allocator: Allocator,

        pub fn init(allocator: Allocator) Self {
            return Self{
                .allocator = allocator,
            };
        }

        /// Generate Stage 3 sumcheck proof with proper transcript flow
        ///
        /// This implements the actual sumcheck protocol where:
        /// - Round j polynomial = sum over remaining variables of the instance polynomial
        /// - After all rounds, output claims are the MLE evaluations at the challenge point
        ///
        /// Transcript flow (matching Jolt verifier):
        /// 1. Derive 5 gamma powers for ShiftSumcheck
        /// 2. Derive 1 gamma for InstructionInputSumcheck
        /// 3. Derive 1 gamma for RegistersClaimReduction
        /// 4. Compute and append 3 input claims
        /// 5. Derive 3 batching coefficients
        /// 6. For each round: compute round poly, compress, append to transcript, derive challenge
        /// 7. Compute and append 16 opening claims
        pub fn generateStage3Proof(
            self: *Self,
            proof: *SumcheckInstanceProof(F),
            transcript: *Blake2bTranscript(F),
            opening_claims: *OpeningClaims(F),
            cycle_witnesses: []const r1cs.R1CSCycleInputs(F),
            n_cycle_vars: usize,
            r_outer: []const F, // r_cycle from Stage 1 (BIG_ENDIAN)
            r_product: []const F, // r_cycle from Stage 2 product sumcheck (BIG_ENDIAN)
        ) !Stage3Result(F) {
            const num_rounds = n_cycle_vars;
            const trace_len = cycle_witnesses.len;

            // Debug: Check what witnesses we received
            std.debug.print("[STAGE3] generateStage3Proof: cycle_witnesses.len = {}\n", .{cycle_witnesses.len});
            if (cycle_witnesses.len > 0) {
                std.debug.print("[STAGE3] generateStage3Proof: witness[0].PC (idx 6) = {any}\n", .{cycle_witnesses[0].values[6].toBytesBE()});
                std.debug.print("[STAGE3] generateStage3Proof: witness[0].UPC (idx 7) = {any}\n", .{cycle_witnesses[0].values[7].toBytesBE()});
            }

            std.debug.print("[STAGE3] Starting with {} rounds, trace_len={}\n", .{ num_rounds, trace_len });

            // DEBUG: Print transcript state BEFORE gamma derivation
            std.debug.print("\n[ZOLT] ========== STAGE 3 BEGIN ==========\n", .{});
            std.debug.print("[ZOLT] STAGE3_PRE: transcript_state = {{ {any} }}\n", .{transcript.state[0..16]});

            // DEBUG: Print r_outer and r_product inputs
            std.debug.print("[ZOLT] STAGE3_PRE: r_outer.len = {}\n", .{r_outer.len});
            std.debug.print("[ZOLT] STAGE3_PRE: r_product.len = {}\n", .{r_product.len});
            if (r_outer.len > 0) {
                std.debug.print("[ZOLT] STAGE3_PRE: r_outer[0] = {{ {any} }}\n", .{r_outer[0].toBytes()});
                if (r_outer.len > 1) {
                    std.debug.print("[ZOLT] STAGE3_PRE: r_outer[1] = {{ {any} }}\n", .{r_outer[1].toBytes()});
                }
                std.debug.print("[ZOLT] STAGE3_PRE: r_outer[last] = {{ {any} }}\n", .{r_outer[r_outer.len - 1].toBytes()});
            }
            if (r_product.len > 0) {
                std.debug.print("[ZOLT] STAGE3_PRE: r_product[0] = {{ {any} }}\n", .{r_product[0].toBytes()});
                if (r_product.len > 1) {
                    std.debug.print("[ZOLT] STAGE3_PRE: r_product[1] = {{ {any} }}\n", .{r_product[1].toBytes()});
                }
                std.debug.print("[ZOLT] STAGE3_PRE: r_product[last] = {{ {any} }}\n", .{r_product[r_product.len - 1].toBytes()});
            }

            // Phase 1: Derive parameters (BEFORE BatchedSumcheck::verify)
            // NOTE: Stage 3 uses challenge_scalar (NOT challenge_scalar_optimized) which means
            // we need challengeScalarFull (no 125-bit masking) to match Jolt's behavior.
            //
            // ShiftSumcheckParams::new - derive 5 gamma powers
            const shift_gamma_powers = try self.deriveGammaPowersFull(transcript, 5);
            defer self.allocator.free(shift_gamma_powers);

            // Debug: Print all 5 gamma powers in LE bytes format for comparison
            std.debug.print("[ZOLT] STAGE3_SHIFT: gamma_powers[0] = {{ {any} }}\n", .{shift_gamma_powers[0].toBytes()});
            std.debug.print("[ZOLT] STAGE3_SHIFT: gamma_powers[1] = {{ {any} }}\n", .{shift_gamma_powers[1].toBytes()});
            std.debug.print("[ZOLT] STAGE3_SHIFT: gamma_powers[2] = {{ {any} }}\n", .{shift_gamma_powers[2].toBytes()});
            std.debug.print("[ZOLT] STAGE3_SHIFT: gamma_powers[3] = {{ {any} }}\n", .{shift_gamma_powers[3].toBytes()});
            std.debug.print("[ZOLT] STAGE3_SHIFT: gamma_powers[4] = {{ {any} }}\n", .{shift_gamma_powers[4].toBytes()});

            // InstructionInputParams::new - derive 1 gamma
            const instr_gamma = transcript.challengeScalarFull();
            const instr_gamma_sqr = instr_gamma.mul(instr_gamma);
            std.debug.print("[ZOLT] STAGE3_INSTR: gamma = {{ {any} }}\n", .{instr_gamma.toBytes()});
            std.debug.print("[ZOLT] STAGE3_INSTR: gamma_sqr = {{ {any} }}\n", .{instr_gamma_sqr.toBytes()});

            // RegistersClaimReductionSumcheckParams::new - derive 1 gamma
            const reg_gamma = transcript.challengeScalarFull();
            const reg_gamma_sqr = reg_gamma.mul(reg_gamma);
            std.debug.print("[ZOLT] STAGE3_REG: gamma = {{ {any} }}\n", .{reg_gamma.toBytes()});
            std.debug.print("[ZOLT] STAGE3_REG: gamma_sqr = {{ {any} }}\n", .{reg_gamma_sqr.toBytes()});

            // Compute input claims for each sumcheck instance
            const shift_input_claim = self.computeShiftInputClaim(
                opening_claims,
                shift_gamma_powers,
            );
            std.debug.print("[ZOLT] STAGE3_PRE: input_claim[0] (Shift) = {{ {any} }}\n", .{shift_input_claim.toBytes()});

            const instr_input_claim = self.computeInstructionInputClaim(
                opening_claims,
                instr_gamma,
                instr_gamma_sqr,
            );
            std.debug.print("[ZOLT] STAGE3_PRE: input_claim[1] (InstrInput) = {{ {any} }}\n", .{instr_input_claim.toBytes()});

            const reg_input_claim = self.computeRegistersInputClaim(
                opening_claims,
                reg_gamma,
                reg_gamma_sqr,
            );
            std.debug.print("[ZOLT] STAGE3_PRE: input_claim[2] (Registers) = {{ {any} }}\n", .{reg_input_claim.toBytes()});

            // Phase 2: BatchedSumcheck::verify protocol

            // Append input claims to transcript (line 201 in sumcheck.rs)
            transcript.appendScalar(shift_input_claim);
            transcript.appendScalar(instr_input_claim);
            transcript.appendScalar(reg_input_claim);

            // Derive batching coefficients (line 204 in sumcheck.rs)
            // NOTE: Jolt's challenge_vector uses challenge_scalar (full 128 bits, no masking)
            var batching_coeffs: [3]F = undefined;
            for (0..3) |i| {
                batching_coeffs[i] = transcript.challengeScalarFull();
            }
            std.debug.print("[ZOLT] STAGE3_PRE: batching_coeff[0] = {{ {any} }}\n", .{batching_coeffs[0].toBytes()});
            std.debug.print("[ZOLT] STAGE3_PRE: batching_coeff[1] = {{ {any} }}\n", .{batching_coeffs[1].toBytes()});
            std.debug.print("[ZOLT] STAGE3_PRE: batching_coeff[2] = {{ {any} }}\n", .{batching_coeffs[2].toBytes()});

            // Compute the combined initial claim
            var combined_claim = shift_input_claim.mul(batching_coeffs[0]);
            combined_claim = combined_claim.add(instr_input_claim.mul(batching_coeffs[1]));
            combined_claim = combined_claim.add(reg_input_claim.mul(batching_coeffs[2]));
            std.debug.print("[ZOLT] STAGE3_INITIAL: batched_claim = {{ {any} }}\n", .{combined_claim.toBytes()});

            // Allocate challenges
            var challenges = try self.allocator.alloc(F, num_rounds);

            // Build MLEs from trace data
            var shift_mles = try self.buildShiftMLEs(cycle_witnesses, trace_len);
            defer shift_mles.deinit(self.allocator);

            var instr_mles = try self.buildInstructionInputMLEs(cycle_witnesses, trace_len);
            defer instr_mles.deinit(self.allocator);

            var reg_mles = try self.buildRegistersMLEs(cycle_witnesses, trace_len);
            defer reg_mles.deinit(self.allocator);

            // Build eq/eq+1 polynomial evaluations tables
            // For each instance, we need eq(r_prev, x) or eq+1(r_prev, x) weights
            var eq_r_outer_poly = try EqPolynomial(F).init(self.allocator, r_outer);
            defer eq_r_outer_poly.deinit();
            const eq_r_outer_evals = try eq_r_outer_poly.evals(self.allocator);
            defer self.allocator.free(eq_r_outer_evals);

            var eq_r_product_poly = try EqPolynomial(F).init(self.allocator, r_product);
            defer eq_r_product_poly.deinit();
            const eq_r_product_evals = try eq_r_product_poly.evals(self.allocator);
            defer self.allocator.free(eq_r_product_evals);

            // For eq+1, we compute tables eq_plus_one(r, j) for all j
            const eq_plus_one_outer_evals = try self.computeEqPlusOneEvals(r_outer, trace_len);
            defer self.allocator.free(eq_plus_one_outer_evals);

            const eq_plus_one_product_evals = try self.computeEqPlusOneEvals(r_product, trace_len);
            defer self.allocator.free(eq_plus_one_product_evals);

            // DEBUG: Print MLE sample values
            std.debug.print("\n[ZOLT] STAGE3_MLE: shift_mles sample values:\n", .{});
            if (trace_len > 0) {
                std.debug.print("[ZOLT] STAGE3_MLE: unexpanded_pc[0] = {{ {any} }}\n", .{shift_mles.unexpanded_pc[0].toBytes()});
                std.debug.print("[ZOLT] STAGE3_MLE: pc[0] = {{ {any} }}\n", .{shift_mles.pc[0].toBytes()});
                std.debug.print("[ZOLT] STAGE3_MLE: is_virtual[0] = {{ {any} }}\n", .{shift_mles.is_virtual[0].toBytes()});
                std.debug.print("[ZOLT] STAGE3_MLE: is_first_in_sequence[0] = {{ {any} }}\n", .{shift_mles.is_first_in_sequence[0].toBytes()});
                std.debug.print("[ZOLT] STAGE3_MLE: is_noop[0] = {{ {any} }}\n", .{shift_mles.is_noop[0].toBytes()});
                if (trace_len > 1) {
                    std.debug.print("[ZOLT] STAGE3_MLE: unexpanded_pc[1] = {{ {any} }}\n", .{shift_mles.unexpanded_pc[1].toBytes()});
                    std.debug.print("[ZOLT] STAGE3_MLE: is_noop[1] = {{ {any} }}\n", .{shift_mles.is_noop[1].toBytes()});
                }
            }

            std.debug.print("\n[ZOLT] STAGE3_MLE: instr_mles sample values:\n", .{});
            if (trace_len > 0) {
                std.debug.print("[ZOLT] STAGE3_MLE: left_is_rs1[0] = {{ {any} }}\n", .{instr_mles.left_is_rs1[0].toBytes()});
                std.debug.print("[ZOLT] STAGE3_MLE: rs1_value[0] = {{ {any} }}\n", .{instr_mles.rs1_value[0].toBytes()});
                std.debug.print("[ZOLT] STAGE3_MLE: right_is_rs2[0] = {{ {any} }}\n", .{instr_mles.right_is_rs2[0].toBytes()});
                std.debug.print("[ZOLT] STAGE3_MLE: rs2_value[0] = {{ {any} }}\n", .{instr_mles.rs2_value[0].toBytes()});
                std.debug.print("[ZOLT] STAGE3_MLE: imm[0] = {{ {any} }}\n", .{instr_mles.imm[0].toBytes()});
            }

            std.debug.print("\n[ZOLT] STAGE3_MLE: reg_mles sample values:\n", .{});
            if (trace_len > 0) {
                std.debug.print("[ZOLT] STAGE3_MLE: rd_write_value[0] = {{ {any} }}\n", .{reg_mles.rd_write_value[0].toBytes()});
                std.debug.print("[ZOLT] STAGE3_MLE: rs1_value[0] = {{ {any} }}\n", .{reg_mles.rs1_value[0].toBytes()});
                std.debug.print("[ZOLT] STAGE3_MLE: rs2_value[0] = {{ {any} }}\n", .{reg_mles.rs2_value[0].toBytes()});
            }

            // DEBUG: Print eq polynomial sample values
            std.debug.print("\n[ZOLT] STAGE3_EQ: eq polynomial evaluations:\n", .{});
            if (eq_r_outer_evals.len > 0) {
                std.debug.print("[ZOLT] STAGE3_EQ: eq_r_outer[0] = {{ {any} }}\n", .{eq_r_outer_evals[0].toBytes()});
                if (eq_r_outer_evals.len > 1) {
                    std.debug.print("[ZOLT] STAGE3_EQ: eq_r_outer[1] = {{ {any} }}\n", .{eq_r_outer_evals[1].toBytes()});
                }
            }
            if (eq_r_product_evals.len > 0) {
                std.debug.print("[ZOLT] STAGE3_EQ: eq_r_product[0] = {{ {any} }}\n", .{eq_r_product_evals[0].toBytes()});
            }
            if (eq_plus_one_outer_evals.len > 0) {
                std.debug.print("[ZOLT] STAGE3_EQ: eq_plus_one_outer[0] = {{ {any} }}\n", .{eq_plus_one_outer_evals[0].toBytes()});
                if (eq_plus_one_outer_evals.len > 1) {
                    std.debug.print("[ZOLT] STAGE3_EQ: eq_plus_one_outer[1] = {{ {any} }}\n", .{eq_plus_one_outer_evals[1].toBytes()});
                }
            }
            if (eq_plus_one_product_evals.len > 0) {
                std.debug.print("[ZOLT] STAGE3_EQ: eq_plus_one_product[0] = {{ {any} }}\n", .{eq_plus_one_product_evals[0].toBytes()});
            }

            // Track current claims for each instance
            var current_shift_claim = shift_input_claim;
            var current_instr_claim = instr_input_claim;
            var current_reg_claim = reg_input_claim;

            // Run sumcheck rounds
            for (0..num_rounds) |round| {
                // Current size of MLEs is trace_len >> round
                const round_shift: u6 = @intCast(round);
                const current_size = trace_len >> round_shift;
                const half_size = current_size >> 1;

                // Compute round polynomial for each instance
                // Following Jolt's approach:
                // - Compute p(0), p(2) for degree-2 or p(0), p(2), p(3) for degree-3
                // - Derive p(1) = previous_claim - p(0)
                // - Convert to coefficients and compress

                // ShiftSumcheck: degree 2, returns [p(0), p(2)]
                const shift_evals_02 = self.computeShiftRoundEvals(
                    &shift_mles,
                    eq_plus_one_outer_evals,
                    eq_plus_one_product_evals,
                    shift_gamma_powers,
                    half_size,
                );
                // Derive p(1) from claim
                const shift_p1 = current_shift_claim.sub(shift_evals_02[0]);
                const shift_evals: [3]F = .{ shift_evals_02[0], shift_p1, shift_evals_02[1] };

                // InstructionInputSumcheck: degree 3, returns [p(0), p(2), p(3)]
                const instr_evals_023 = self.computeInstrRoundEvals(
                    &instr_mles,
                    eq_r_outer_evals,
                    eq_r_product_evals,
                    instr_gamma,
                    instr_gamma_sqr,
                    half_size,
                );
                // Derive p(1) from claim
                const instr_p1 = current_instr_claim.sub(instr_evals_023[0]);
                const instr_evals: [4]F = .{ instr_evals_023[0], instr_p1, instr_evals_023[1], instr_evals_023[2] };

                // RegistersClaimReduction: degree 2, returns [p(0), p(2)]
                const reg_evals_02 = self.computeRegRoundEvals(
                    &reg_mles,
                    eq_r_outer_evals, // r_spartan = r_outer
                    reg_gamma,
                    reg_gamma_sqr,
                    half_size,
                );
                // Derive p(1) from claim
                const reg_p1 = current_reg_claim.sub(reg_evals_02[0]);
                const reg_evals: [3]F = .{ reg_evals_02[0], reg_p1, reg_evals_02[1] };

                // Debug round evaluations (all rounds for comprehensive analysis)
                // First 10 rounds: full detail; rest: summary only
                const detailed = round < 10;

                std.debug.print("\n[ZOLT] STAGE3_ROUND_{}: current_claim = {{ {any} }}\n", .{ round, combined_claim.toBytes() });
                std.debug.print("[ZOLT] STAGE3_ROUND_{}: shift_claim = {{ {any} }}\n", .{ round, current_shift_claim.toBytes() });
                std.debug.print("[ZOLT] STAGE3_ROUND_{}: instr_claim = {{ {any} }}\n", .{ round, current_instr_claim.toBytes() });
                std.debug.print("[ZOLT] STAGE3_ROUND_{}: reg_claim = {{ {any} }}\n", .{ round, current_reg_claim.toBytes() });

                if (detailed) {
                    // Shift evaluations
                    std.debug.print("[ZOLT] STAGE3_ROUND_{}: shift_p0 = {{ {any} }}\n", .{ round, shift_evals[0].toBytes() });
                    std.debug.print("[ZOLT] STAGE3_ROUND_{}: shift_p1 = {{ {any} }}\n", .{ round, shift_evals[1].toBytes() });
                    std.debug.print("[ZOLT] STAGE3_ROUND_{}: shift_p2 = {{ {any} }}\n", .{ round, shift_evals[2].toBytes() });

                    // Instruction input evaluations
                    std.debug.print("[ZOLT] STAGE3_ROUND_{}: instr_p0 = {{ {any} }}\n", .{ round, instr_evals[0].toBytes() });
                    std.debug.print("[ZOLT] STAGE3_ROUND_{}: instr_p1 = {{ {any} }}\n", .{ round, instr_evals[1].toBytes() });
                    std.debug.print("[ZOLT] STAGE3_ROUND_{}: instr_p2 = {{ {any} }}\n", .{ round, instr_evals[2].toBytes() });
                    std.debug.print("[ZOLT] STAGE3_ROUND_{}: instr_p3 = {{ {any} }}\n", .{ round, instr_evals[3].toBytes() });

                    // Registers evaluations
                    std.debug.print("[ZOLT] STAGE3_ROUND_{}: reg_p0 = {{ {any} }}\n", .{ round, reg_evals[0].toBytes() });
                    std.debug.print("[ZOLT] STAGE3_ROUND_{}: reg_p1 = {{ {any} }}\n", .{ round, reg_evals[1].toBytes() });
                    std.debug.print("[ZOLT] STAGE3_ROUND_{}: reg_p2 = {{ {any} }}\n", .{ round, reg_evals[2].toBytes() });
                }

                // Combine round polynomials (all evaluated at 0, 1, 2, 3)
                // batched_poly = coeff[0] * shift_poly + coeff[1] * instr_poly + coeff[2] * reg_poly
                var combined_evals: [4]F = undefined;
                for (0..4) |i| {
                    const shift_val = if (i < 3) shift_evals[i] else F.zero();
                    const instr_val = instr_evals[i];
                    const reg_val = if (i < 3) reg_evals[i] else F.zero();
                    combined_evals[i] = shift_val.mul(batching_coeffs[0])
                        .add(instr_val.mul(batching_coeffs[1]))
                        .add(reg_val.mul(batching_coeffs[2]));
                }

                // Convert evaluations to coefficients
                const combined_coeffs = try self.evalsToCoeffs(&combined_evals, 3);
                defer self.allocator.free(combined_coeffs);

                // Compress: [c0, c2, c3] (c1 recovered from hint = combined_claim)
                const compressed = try self.allocator.alloc(F, 3);
                compressed[0] = combined_coeffs[0];
                compressed[1] = combined_coeffs[2];
                compressed[2] = combined_coeffs[3];

                // Append to proof
                try proof.compressed_polys.append(self.allocator, .{
                    .coeffs_except_linear_term = compressed,
                    .allocator = self.allocator,
                });

                // Append compressed poly to transcript
                // NOTE: Jolt uses "UniPoly_begin/end" (NOT "CompressedUniPoly") for CompressedUniPoly
                transcript.appendMessage("UniPoly_begin");
                for (compressed) |coeff| {
                    transcript.appendScalar(coeff);
                }
                transcript.appendMessage("UniPoly_end");

                // Debug: Print compressed coefficients before transcript append
                std.debug.print("[ZOLT] STAGE3_ROUND_{}: c0 = {{ {any} }}\n", .{ round, compressed[0].toBytes() });
                std.debug.print("[ZOLT] STAGE3_ROUND_{}: c2 = {{ {any} }}\n", .{ round, compressed[1].toBytes() });
                std.debug.print("[ZOLT] STAGE3_ROUND_{}: c3 = {{ {any} }}\n", .{ round, compressed[2].toBytes() });

                // Also print the full combined polynomial coefficients for debugging
                if (detailed) {
                    std.debug.print("[ZOLT] STAGE3_ROUND_{}: combined_coeffs = [{{ {any} }}, {{ {any} }}, {{ {any} }}, {{ {any} }}]\n", .{
                        round,
                        combined_coeffs[0].toBytes(),
                        combined_coeffs[1].toBytes(),
                        combined_coeffs[2].toBytes(),
                        combined_coeffs[3].toBytes(),
                    });

                    // Also print combined evals for verification
                    std.debug.print("[ZOLT] STAGE3_ROUND_{}: combined_evals = [{{ {any} }}, {{ {any} }}, {{ {any} }}, {{ {any} }}]\n", .{
                        round,
                        combined_evals[0].toBytes(),
                        combined_evals[1].toBytes(),
                        combined_evals[2].toBytes(),
                        combined_evals[3].toBytes(),
                    });
                }

                // Derive challenge
                const r_j = transcript.challengeScalar();
                challenges[round] = r_j;

                std.debug.print("[ZOLT] STAGE3_ROUND_{}: challenge = {{ {any} }}\n", .{ round, r_j.toBytes() });

                // Evaluate combined polynomial at r_j to get next claim
                combined_claim = self.evaluatePolyAtPoint(combined_coeffs, r_j);
                std.debug.print("[ZOLT] STAGE3_ROUND_{}: next_claim = {{ {any} }}\n", .{ round, combined_claim.toBytes() });

                // Update individual claims by evaluating their polynomials at r_j
                const shift_coeffs = try self.evalsToCoeffs(&shift_evals, 2);
                defer self.allocator.free(shift_coeffs);
                current_shift_claim = self.evaluatePolyAtPoint(shift_coeffs, r_j);

                const instr_coeffs = try self.evalsToCoeffs(&instr_evals, 3);
                defer self.allocator.free(instr_coeffs);
                current_instr_claim = self.evaluatePolyAtPoint(instr_coeffs, r_j);

                const reg_coeffs = try self.evalsToCoeffs(&reg_evals, 2);
                defer self.allocator.free(reg_coeffs);
                current_reg_claim = self.evaluatePolyAtPoint(reg_coeffs, r_j);

                // Bind all MLEs at r_j
                shift_mles.bind(r_j);
                instr_mles.bind(r_j);
                reg_mles.bind(r_j);

                // Bind eq evaluations
                self.bindEqEvals(eq_r_outer_evals, r_j);
                self.bindEqEvals(eq_r_product_evals, r_j);
                self.bindEqEvals(eq_plus_one_outer_evals, r_j);
                self.bindEqEvals(eq_plus_one_product_evals, r_j);
            }

            std.debug.print("\n[ZOLT] STAGE3_FINAL: output_claim = {{ {any} }}\n", .{combined_claim.toBytes()});
            std.debug.print("[ZOLT] STAGE3_FINAL: shift_claim = {{ {any} }}\n", .{current_shift_claim.toBytes()});
            std.debug.print("[ZOLT] STAGE3_FINAL: instr_claim = {{ {any} }}\n", .{current_instr_claim.toBytes()});
            std.debug.print("[ZOLT] STAGE3_FINAL: reg_claim = {{ {any} }}\n", .{current_reg_claim.toBytes()});
            std.debug.print("[ZOLT] STAGE3_FINAL: num_rounds = {}\n", .{num_rounds});

            // Print final challenges
            std.debug.print("[ZOLT] STAGE3_FINAL: challenges[0] = {{ {any} }}\n", .{challenges[0].toBytes()});
            if (num_rounds > 1) {
                std.debug.print("[ZOLT] STAGE3_FINAL: challenges[last] = {{ {any} }}\n", .{challenges[num_rounds - 1].toBytes()});
            }

            // Phase 3: Compute and cache opening claims
            // After all rounds, the MLEs are bound to single values
            const shift_claims = shift_mles.finalClaims();
            const instr_claims = instr_mles.finalClaims();
            const reg_claims = reg_mles.finalClaims();

            // DEBUG: Print opening claims
            std.debug.print("\n[ZOLT] STAGE3_OPENING: Shift sumcheck claims:\n", .{});
            std.debug.print("[ZOLT] STAGE3_OPENING: unexpanded_pc = {{ {any} }}\n", .{shift_claims.unexpanded_pc.toBytes()});
            std.debug.print("[ZOLT] STAGE3_OPENING: pc = {{ {any} }}\n", .{shift_claims.pc.toBytes()});
            std.debug.print("[ZOLT] STAGE3_OPENING: is_virtual = {{ {any} }}\n", .{shift_claims.is_virtual.toBytes()});
            std.debug.print("[ZOLT] STAGE3_OPENING: is_first_in_sequence = {{ {any} }}\n", .{shift_claims.is_first_in_sequence.toBytes()});
            std.debug.print("[ZOLT] STAGE3_OPENING: is_noop = {{ {any} }}\n", .{shift_claims.is_noop.toBytes()});

            std.debug.print("\n[ZOLT] STAGE3_OPENING: InstructionInput sumcheck claims:\n", .{});
            std.debug.print("[ZOLT] STAGE3_OPENING: left_is_rs1 = {{ {any} }}\n", .{instr_claims.left_is_rs1.toBytes()});
            std.debug.print("[ZOLT] STAGE3_OPENING: rs1_value = {{ {any} }}\n", .{instr_claims.rs1_value.toBytes()});
            std.debug.print("[ZOLT] STAGE3_OPENING: left_is_pc = {{ {any} }}\n", .{instr_claims.left_is_pc.toBytes()});
            std.debug.print("[ZOLT] STAGE3_OPENING: unexpanded_pc = {{ {any} }}\n", .{instr_claims.unexpanded_pc.toBytes()});
            std.debug.print("[ZOLT] STAGE3_OPENING: right_is_rs2 = {{ {any} }}\n", .{instr_claims.right_is_rs2.toBytes()});
            std.debug.print("[ZOLT] STAGE3_OPENING: rs2_value = {{ {any} }}\n", .{instr_claims.rs2_value.toBytes()});
            std.debug.print("[ZOLT] STAGE3_OPENING: right_is_imm = {{ {any} }}\n", .{instr_claims.right_is_imm.toBytes()});
            std.debug.print("[ZOLT] STAGE3_OPENING: imm = {{ {any} }}\n", .{instr_claims.imm.toBytes()});

            std.debug.print("\n[ZOLT] STAGE3_OPENING: RegistersClaimReduction claims:\n", .{});
            std.debug.print("[ZOLT] STAGE3_OPENING: rd_write_value = {{ {any} }}\n", .{reg_claims.rd_write_value.toBytes()});
            std.debug.print("[ZOLT] STAGE3_OPENING: rs1_value = {{ {any} }}\n", .{reg_claims.rs1_value.toBytes()});
            std.debug.print("[ZOLT] STAGE3_OPENING: rs2_value = {{ {any} }}\n", .{reg_claims.rs2_value.toBytes()});

            // Append opening claims to transcript (cache_openings)
            // ShiftSumcheck: 5 claims
            transcript.appendScalar(shift_claims.unexpanded_pc);
            transcript.appendScalar(shift_claims.pc);
            transcript.appendScalar(shift_claims.is_virtual);
            transcript.appendScalar(shift_claims.is_first_in_sequence);
            transcript.appendScalar(shift_claims.is_noop);

            // InstructionInputSumcheck: 8 claims
            transcript.appendScalar(instr_claims.left_is_rs1);
            transcript.appendScalar(instr_claims.rs1_value);
            transcript.appendScalar(instr_claims.left_is_pc);
            transcript.appendScalar(instr_claims.unexpanded_pc);
            transcript.appendScalar(instr_claims.right_is_rs2);
            transcript.appendScalar(instr_claims.rs2_value);
            transcript.appendScalar(instr_claims.right_is_imm);
            transcript.appendScalar(instr_claims.imm);

            // RegistersClaimReduction: 3 claims
            transcript.appendScalar(reg_claims.rd_write_value);
            transcript.appendScalar(reg_claims.rs1_value);
            transcript.appendScalar(reg_claims.rs2_value);

            return Stage3Result(F){
                .challenges = challenges,
                .shift_final_claim = current_shift_claim,
                .instr_final_claim = current_instr_claim,
                .reg_final_claim = current_reg_claim,
                .shift_unexpanded_pc_claim = shift_claims.unexpanded_pc,
                .shift_pc_claim = shift_claims.pc,
                .shift_is_virtual_claim = shift_claims.is_virtual,
                .shift_is_first_in_sequence_claim = shift_claims.is_first_in_sequence,
                .shift_is_noop_claim = shift_claims.is_noop,
                .instr_left_is_rs1_claim = instr_claims.left_is_rs1,
                .instr_rs1_value_claim = instr_claims.rs1_value,
                .instr_left_is_pc_claim = instr_claims.left_is_pc,
                .instr_unexpanded_pc_claim = instr_claims.unexpanded_pc,
                .instr_right_is_rs2_claim = instr_claims.right_is_rs2,
                .instr_rs2_value_claim = instr_claims.rs2_value,
                .instr_right_is_imm_claim = instr_claims.right_is_imm,
                .instr_imm_claim = instr_claims.imm,
                .reg_rd_write_value_claim = reg_claims.rd_write_value,
                .reg_rs1_value_claim = reg_claims.rs1_value,
                .reg_rs2_value_claim = reg_claims.rs2_value,
                .allocator = self.allocator,
            };
        }

        /// Derive n gamma powers from transcript (uses 125-bit masked scalars)
        fn deriveGammaPowers(self: *Self, transcript: *Blake2bTranscript(F), n: usize) ![]F {
            const powers = try self.allocator.alloc(F, n);
            const gamma = transcript.challengeScalar();
            powers[0] = F.one();
            if (n > 1) {
                powers[1] = gamma;
                for (2..n) |i| {
                    powers[i] = powers[i - 1].mul(gamma);
                }
            }
            return powers;
        }

        /// Derive n gamma powers from transcript (uses full 128-bit scalars)
        /// This matches Jolt's challenge_scalar_powers which calls challenge_scalar (not optimized)
        fn deriveGammaPowersFull(self: *Self, transcript: *Blake2bTranscript(F), n: usize) ![]F {
            const powers = try self.allocator.alloc(F, n);
            const gamma = transcript.challengeScalarFull();
            powers[0] = F.one();
            if (n > 1) {
                powers[1] = gamma;
                for (2..n) |i| {
                    powers[i] = powers[i - 1].mul(gamma);
                }
            }
            return powers;
        }

        /// Compute ShiftSumcheck input claim from opening accumulator
        fn computeShiftInputClaim(
            self: *Self,
            opening_claims: *const OpeningClaims(F),
            gamma_powers: []const F,
        ) F {
            _ = self;
            // input_claim = NextUnexpandedPC + gamma*NextPC + gamma^2*NextIsVirtual
            //             + gamma^3*NextIsFirstInSequence + gamma^4*(1 - NextIsNoop)
            const next_unexpanded_pc = opening_claims.get(.{ .Virtual = .{ .poly = .NextUnexpandedPC, .sumcheck_id = .SpartanOuter } }) orelse F.zero();
            const next_pc = opening_claims.get(.{ .Virtual = .{ .poly = .NextPC, .sumcheck_id = .SpartanOuter } }) orelse F.zero();
            const next_is_virtual = opening_claims.get(.{ .Virtual = .{ .poly = .NextIsVirtual, .sumcheck_id = .SpartanOuter } }) orelse F.zero();
            const next_is_first = opening_claims.get(.{ .Virtual = .{ .poly = .NextIsFirstInSequence, .sumcheck_id = .SpartanOuter } }) orelse F.zero();
            const next_is_noop = opening_claims.get(.{ .Virtual = .{ .poly = .NextIsNoop, .sumcheck_id = .SpartanProductVirtualization } }) orelse F.zero();

            // DEBUG: Print the individual claims with consistent format
            std.debug.print("[ZOLT] STAGE3_SHIFT_INPUT: NextUnexpandedPC = {{ {any} }}\n", .{next_unexpanded_pc.toBytes()});
            std.debug.print("[ZOLT] STAGE3_SHIFT_INPUT: NextPC = {{ {any} }}\n", .{next_pc.toBytes()});
            std.debug.print("[ZOLT] STAGE3_SHIFT_INPUT: NextIsVirtual = {{ {any} }}\n", .{next_is_virtual.toBytes()});
            std.debug.print("[ZOLT] STAGE3_SHIFT_INPUT: NextIsFirstInSequence = {{ {any} }}\n", .{next_is_first.toBytes()});
            std.debug.print("[ZOLT] STAGE3_SHIFT_INPUT: NextIsNoop = {{ {any} }}\n", .{next_is_noop.toBytes()});

            var result = next_unexpanded_pc;
            result = result.add(gamma_powers[1].mul(next_pc));
            result = result.add(gamma_powers[2].mul(next_is_virtual));
            result = result.add(gamma_powers[3].mul(next_is_first));
            result = result.add(gamma_powers[4].mul(F.one().sub(next_is_noop)));
            return result;
        }

        /// Compute InstructionInputSumcheck input claim
        fn computeInstructionInputClaim(
            self: *Self,
            opening_claims: *const OpeningClaims(F),
            gamma: F,
            gamma_sqr: F,
        ) F {
            _ = self;
            // input_claim = (right_1 + gamma*left_1) + gamma^2*(right_2 + gamma*left_2)
            const left_1 = opening_claims.get(.{ .Virtual = .{ .poly = .LeftInstructionInput, .sumcheck_id = .SpartanOuter } }) orelse F.zero();
            const right_1 = opening_claims.get(.{ .Virtual = .{ .poly = .RightInstructionInput, .sumcheck_id = .SpartanOuter } }) orelse F.zero();
            const left_2 = opening_claims.get(.{ .Virtual = .{ .poly = .LeftInstructionInput, .sumcheck_id = .SpartanProductVirtualization } }) orelse F.zero();
            const right_2 = opening_claims.get(.{ .Virtual = .{ .poly = .RightInstructionInput, .sumcheck_id = .SpartanProductVirtualization } }) orelse F.zero();

            // DEBUG: Print the individual claims
            std.debug.print("[ZOLT] STAGE3_INSTR_INPUT: LeftInstructionInput(Outer) = {{ {any} }}\n", .{left_1.toBytes()});
            std.debug.print("[ZOLT] STAGE3_INSTR_INPUT: RightInstructionInput(Outer) = {{ {any} }}\n", .{right_1.toBytes()});
            std.debug.print("[ZOLT] STAGE3_INSTR_INPUT: LeftInstructionInput(Product) = {{ {any} }}\n", .{left_2.toBytes()});
            std.debug.print("[ZOLT] STAGE3_INSTR_INPUT: RightInstructionInput(Product) = {{ {any} }}\n", .{right_2.toBytes()});

            const claim_1 = right_1.add(gamma.mul(left_1));
            const claim_2 = right_2.add(gamma.mul(left_2));
            return claim_1.add(gamma_sqr.mul(claim_2));
        }

        /// Compute RegistersClaimReduction input claim
        fn computeRegistersInputClaim(
            self: *Self,
            opening_claims: *const OpeningClaims(F),
            gamma: F,
            gamma_sqr: F,
        ) F {
            _ = self;
            // input_claim = rd + gamma*rs1 + gamma^2*rs2
            const rd = opening_claims.get(.{ .Virtual = .{ .poly = .RdWriteValue, .sumcheck_id = .SpartanOuter } }) orelse F.zero();
            const rs1 = opening_claims.get(.{ .Virtual = .{ .poly = .Rs1Value, .sumcheck_id = .SpartanOuter } }) orelse F.zero();
            const rs2 = opening_claims.get(.{ .Virtual = .{ .poly = .Rs2Value, .sumcheck_id = .SpartanOuter } }) orelse F.zero();

            // DEBUG: Print the individual claims
            std.debug.print("[ZOLT] STAGE3_REG_INPUT: RdWriteValue = {{ {any} }}\n", .{rd.toBytes()});
            std.debug.print("[ZOLT] STAGE3_REG_INPUT: Rs1Value = {{ {any} }}\n", .{rs1.toBytes()});
            std.debug.print("[ZOLT] STAGE3_REG_INPUT: Rs2Value = {{ {any} }}\n", .{rs2.toBytes()});

            var result = rd;
            result = result.add(gamma.mul(rs1));
            result = result.add(gamma_sqr.mul(rs2));
            return result;
        }

        /// MLE structure for Shift sumcheck
        const ShiftMLEs = struct {
            unexpanded_pc: []F,
            pc: []F,
            is_virtual: []F,
            is_first_in_sequence: []F,
            is_noop: []F,

            fn deinit(self: *ShiftMLEs, allocator: Allocator) void {
                allocator.free(self.unexpanded_pc);
                allocator.free(self.pc);
                allocator.free(self.is_virtual);
                allocator.free(self.is_first_in_sequence);
                allocator.free(self.is_noop);
            }

            fn bind(self: *ShiftMLEs, r: F) void {
                bindMLE(self.unexpanded_pc, r);
                bindMLE(self.pc, r);
                bindMLE(self.is_virtual, r);
                bindMLE(self.is_first_in_sequence, r);
                bindMLE(self.is_noop, r);
            }

            const FinalClaims = struct {
                unexpanded_pc: F,
                pc: F,
                is_virtual: F,
                is_first_in_sequence: F,
                is_noop: F,
            };

            fn finalClaims(self: *const ShiftMLEs) FinalClaims {
                return .{
                    .unexpanded_pc = if (self.unexpanded_pc.len > 0) self.unexpanded_pc[0] else F.zero(),
                    .pc = if (self.pc.len > 0) self.pc[0] else F.zero(),
                    .is_virtual = if (self.is_virtual.len > 0) self.is_virtual[0] else F.zero(),
                    .is_first_in_sequence = if (self.is_first_in_sequence.len > 0) self.is_first_in_sequence[0] else F.zero(),
                    .is_noop = if (self.is_noop.len > 0) self.is_noop[0] else F.zero(),
                };
            }
        };

        /// Build Shift MLEs from trace
        fn buildShiftMLEs(self: *Self, cycle_witnesses: []const r1cs.R1CSCycleInputs(F), trace_len: usize) !ShiftMLEs {
            const unexpanded_pc = try self.allocator.alloc(F, trace_len);
            const pc = try self.allocator.alloc(F, trace_len);
            const is_virtual = try self.allocator.alloc(F, trace_len);
            const is_first_in_sequence = try self.allocator.alloc(F, trace_len);
            const is_noop = try self.allocator.alloc(F, trace_len);

            // Debug: print index mapping
            std.debug.print("[STAGE3] buildShiftMLEs: UnexpandedPC idx={}, PC idx={}\n", .{
                R1CSInputIndex.UnexpandedPC.toIndex(),
                R1CSInputIndex.PC.toIndex(),
            });
            std.debug.print("[STAGE3] buildShiftMLEs: cycle_witnesses.len={}, trace_len={}\n", .{
                cycle_witnesses.len,
                trace_len,
            });

            for (0..trace_len) |i| {
                if (i < cycle_witnesses.len) {
                    const values = &cycle_witnesses[i].values;
                    unexpanded_pc[i] = values[R1CSInputIndex.UnexpandedPC.toIndex()];
                    pc[i] = values[R1CSInputIndex.PC.toIndex()];
                    is_virtual[i] = values[R1CSInputIndex.FlagVirtualInstruction.toIndex()];
                    is_first_in_sequence[i] = values[R1CSInputIndex.FlagIsFirstInSequence.toIndex()];
                    is_noop[i] = values[R1CSInputIndex.FlagIsNoop.toIndex()];

                    // Debug first few cycles - show last 8 bytes where small values reside
                    if (i < 3) {
                        std.debug.print("[STAGE3] cycle[{}]: raw_upc_idx7={any}, raw_pc_idx6={any}\n", .{
                            i,
                            values[7].toBytesBE()[24..32], // UnexpandedPC = 7, last 8 bytes
                            values[6].toBytesBE()[24..32], // PC = 6, last 8 bytes
                        });
                        std.debug.print("[STAGE3] cycle[{}]: via_enum: upc={any}, pc={any}, noop={any}\n", .{
                            i,
                            unexpanded_pc[i].toBytesBE()[24..32],
                            pc[i].toBytesBE()[24..32],
                            is_noop[i].toBytesBE()[24..32],
                        });
                    }
                } else {
                    // Padding
                    unexpanded_pc[i] = F.zero();
                    pc[i] = F.zero();
                    is_virtual[i] = F.zero();
                    is_first_in_sequence[i] = F.zero();
                    is_noop[i] = F.one(); // Padding cycles are noops
                }
            }

            return ShiftMLEs{
                .unexpanded_pc = unexpanded_pc,
                .pc = pc,
                .is_virtual = is_virtual,
                .is_first_in_sequence = is_first_in_sequence,
                .is_noop = is_noop,
            };
        }

        /// MLE structure for InstructionInput sumcheck
        const InstructionInputMLEs = struct {
            left_is_rs1: []F,
            rs1_value: []F,
            left_is_pc: []F,
            unexpanded_pc: []F,
            right_is_rs2: []F,
            rs2_value: []F,
            right_is_imm: []F,
            imm: []F,

            fn deinit(self: *InstructionInputMLEs, allocator: Allocator) void {
                allocator.free(self.left_is_rs1);
                allocator.free(self.rs1_value);
                allocator.free(self.left_is_pc);
                allocator.free(self.unexpanded_pc);
                allocator.free(self.right_is_rs2);
                allocator.free(self.rs2_value);
                allocator.free(self.right_is_imm);
                allocator.free(self.imm);
            }

            fn bind(self: *InstructionInputMLEs, r: F) void {
                bindMLE(self.left_is_rs1, r);
                bindMLE(self.rs1_value, r);
                bindMLE(self.left_is_pc, r);
                bindMLE(self.unexpanded_pc, r);
                bindMLE(self.right_is_rs2, r);
                bindMLE(self.rs2_value, r);
                bindMLE(self.right_is_imm, r);
                bindMLE(self.imm, r);
            }

            const FinalClaims = struct {
                left_is_rs1: F,
                rs1_value: F,
                left_is_pc: F,
                unexpanded_pc: F,
                right_is_rs2: F,
                rs2_value: F,
                right_is_imm: F,
                imm: F,
            };

            fn finalClaims(self: *const InstructionInputMLEs) FinalClaims {
                return .{
                    .left_is_rs1 = if (self.left_is_rs1.len > 0) self.left_is_rs1[0] else F.zero(),
                    .rs1_value = if (self.rs1_value.len > 0) self.rs1_value[0] else F.zero(),
                    .left_is_pc = if (self.left_is_pc.len > 0) self.left_is_pc[0] else F.zero(),
                    .unexpanded_pc = if (self.unexpanded_pc.len > 0) self.unexpanded_pc[0] else F.zero(),
                    .right_is_rs2 = if (self.right_is_rs2.len > 0) self.right_is_rs2[0] else F.zero(),
                    .rs2_value = if (self.rs2_value.len > 0) self.rs2_value[0] else F.zero(),
                    .right_is_imm = if (self.right_is_imm.len > 0) self.right_is_imm[0] else F.zero(),
                    .imm = if (self.imm.len > 0) self.imm[0] else F.zero(),
                };
            }
        };

        /// Build InstructionInput MLEs from trace
        /// Uses explicit instruction flags computed in R1CSCycleInputs.fromTraceStep
        fn buildInstructionInputMLEs(self: *Self, cycle_witnesses: []const r1cs.R1CSCycleInputs(F), trace_len: usize) !InstructionInputMLEs {
            const left_is_rs1 = try self.allocator.alloc(F, trace_len);
            const rs1_value = try self.allocator.alloc(F, trace_len);
            const left_is_pc = try self.allocator.alloc(F, trace_len);
            const unexpanded_pc = try self.allocator.alloc(F, trace_len);
            const right_is_rs2 = try self.allocator.alloc(F, trace_len);
            const rs2_value = try self.allocator.alloc(F, trace_len);
            const right_is_imm = try self.allocator.alloc(F, trace_len);
            const imm = try self.allocator.alloc(F, trace_len);

            for (0..trace_len) |i| {
                if (i < cycle_witnesses.len) {
                    const values = &cycle_witnesses[i].values;
                    // Use explicit instruction flags computed from opcode
                    left_is_rs1[i] = values[R1CSInputIndex.FlagLeftOperandIsRs1.toIndex()];
                    left_is_pc[i] = values[R1CSInputIndex.FlagLeftOperandIsPC.toIndex()];
                    right_is_rs2[i] = values[R1CSInputIndex.FlagRightOperandIsRs2.toIndex()];
                    right_is_imm[i] = values[R1CSInputIndex.FlagRightOperandIsImm.toIndex()];
                    // Get operand values
                    rs1_value[i] = values[R1CSInputIndex.Rs1Value.toIndex()];
                    unexpanded_pc[i] = values[R1CSInputIndex.UnexpandedPC.toIndex()];
                    rs2_value[i] = values[R1CSInputIndex.Rs2Value.toIndex()];
                    imm[i] = values[R1CSInputIndex.Imm.toIndex()];
                } else {
                    // Padding - all flags and values are zero
                    left_is_rs1[i] = F.zero();
                    rs1_value[i] = F.zero();
                    left_is_pc[i] = F.zero();
                    unexpanded_pc[i] = F.zero();
                    right_is_rs2[i] = F.zero();
                    rs2_value[i] = F.zero();
                    right_is_imm[i] = F.zero();
                    imm[i] = F.zero();
                }
            }

            return InstructionInputMLEs{
                .left_is_rs1 = left_is_rs1,
                .rs1_value = rs1_value,
                .left_is_pc = left_is_pc,
                .unexpanded_pc = unexpanded_pc,
                .right_is_rs2 = right_is_rs2,
                .rs2_value = rs2_value,
                .right_is_imm = right_is_imm,
                .imm = imm,
            };
        }

        /// MLE structure for RegistersClaimReduction sumcheck
        const RegistersMLEs = struct {
            rd_write_value: []F,
            rs1_value: []F,
            rs2_value: []F,

            fn deinit(self: *RegistersMLEs, allocator: Allocator) void {
                allocator.free(self.rd_write_value);
                allocator.free(self.rs1_value);
                allocator.free(self.rs2_value);
            }

            fn bind(self: *RegistersMLEs, r: F) void {
                bindMLE(self.rd_write_value, r);
                bindMLE(self.rs1_value, r);
                bindMLE(self.rs2_value, r);
            }

            const FinalClaims = struct {
                rd_write_value: F,
                rs1_value: F,
                rs2_value: F,
            };

            fn finalClaims(self: *const RegistersMLEs) FinalClaims {
                return .{
                    .rd_write_value = if (self.rd_write_value.len > 0) self.rd_write_value[0] else F.zero(),
                    .rs1_value = if (self.rs1_value.len > 0) self.rs1_value[0] else F.zero(),
                    .rs2_value = if (self.rs2_value.len > 0) self.rs2_value[0] else F.zero(),
                };
            }
        };

        /// Build Registers MLEs from trace
        fn buildRegistersMLEs(self: *Self, cycle_witnesses: []const r1cs.R1CSCycleInputs(F), trace_len: usize) !RegistersMLEs {
            const rd_write_value = try self.allocator.alloc(F, trace_len);
            const rs1_value = try self.allocator.alloc(F, trace_len);
            const rs2_value = try self.allocator.alloc(F, trace_len);

            for (0..trace_len) |i| {
                if (i < cycle_witnesses.len) {
                    const values = &cycle_witnesses[i].values;
                    rd_write_value[i] = values[R1CSInputIndex.RdWriteValue.toIndex()];
                    rs1_value[i] = values[R1CSInputIndex.Rs1Value.toIndex()];
                    rs2_value[i] = values[R1CSInputIndex.Rs2Value.toIndex()];
                } else {
                    // Padding
                    rd_write_value[i] = F.zero();
                    rs1_value[i] = F.zero();
                    rs2_value[i] = F.zero();
                }
            }

            return RegistersMLEs{
                .rd_write_value = rd_write_value,
                .rs1_value = rs1_value,
                .rs2_value = rs2_value,
            };
        }

        /// Compute eq+1(r, j) evaluations for all j in [0, trace_len)
        fn computeEqPlusOneEvals(self: *Self, r: []const F, trace_len: usize) ![]F {
            const evals = try self.allocator.alloc(F, trace_len);

            const n_vars = r.len;
            for (0..trace_len) |j| {
                // Convert j to binary (BIG_ENDIAN to match r)
                var j_bits = try self.allocator.alloc(F, n_vars);
                defer self.allocator.free(j_bits);
                for (0..n_vars) |k| {
                    const bit_pos: u6 = @intCast(n_vars - 1 - k);
                    j_bits[k] = if ((j >> bit_pos) & 1 == 1) F.one() else F.zero();
                }
                evals[j] = EqPlusOnePolynomial(F).mle(r, j_bits);
            }

            return evals;
        }

        /// Compute round evaluations for ShiftSumcheck
        /// Returns [p(0), p(2)] for degree-2 polynomial (p(1) derived from claim)
        /// Following Jolt's approach: extrapolate each MLE first, then compute products
        fn computeShiftRoundEvals(
            self: *Self,
            mles: *const ShiftMLEs,
            eq_plus_one_outer: []const F,
            eq_plus_one_product: []const F,
            gamma: []const F,
            half_size: usize,
        ) [2]F {
            _ = self;
            var evals: [2]F = .{ F.zero(), F.zero() };

            for (0..half_size) |j| {
                // Get values at j*2 (bit=0) and j*2+1 (bit=1)
                const upc_0 = mles.unexpanded_pc[2 * j];
                const upc_1 = mles.unexpanded_pc[2 * j + 1];
                const pc_0 = mles.pc[2 * j];
                const pc_1 = mles.pc[2 * j + 1];
                const virt_0 = mles.is_virtual[2 * j];
                const virt_1 = mles.is_virtual[2 * j + 1];
                const first_0 = mles.is_first_in_sequence[2 * j];
                const first_1 = mles.is_first_in_sequence[2 * j + 1];
                const noop_0 = mles.is_noop[2 * j];
                const noop_1 = mles.is_noop[2 * j + 1];

                const eq_out_0 = eq_plus_one_outer[2 * j];
                const eq_out_1 = eq_plus_one_outer[2 * j + 1];
                const eq_prod_0 = eq_plus_one_product[2 * j];
                const eq_prod_1 = eq_plus_one_product[2 * j + 1];

                // Linear extrapolation for each MLE: f(2) = 2*f(1) - f(0)
                const upc_2 = upc_1.add(upc_1).sub(upc_0);
                const pc_2 = pc_1.add(pc_1).sub(pc_0);
                const virt_2 = virt_1.add(virt_1).sub(virt_0);
                const first_2 = first_1.add(first_1).sub(first_0);
                const noop_2 = noop_1.add(noop_1).sub(noop_0);
                const eq_out_2 = eq_out_1.add(eq_out_1).sub(eq_out_0);
                const eq_prod_2 = eq_prod_1.add(eq_prod_1).sub(eq_prod_0);

                // f(j, 0) = eq+1_outer(j) * (upc + gamma*pc + gamma^2*virt + gamma^3*first)
                //         + gamma^4 * (1-noop) * eq+1_prod(j)
                const val_0 = upc_0.add(gamma[1].mul(pc_0)).add(gamma[2].mul(virt_0)).add(gamma[3].mul(first_0));
                const term1_0 = eq_out_0.mul(val_0);
                const term2_0 = gamma[4].mul(F.one().sub(noop_0)).mul(eq_prod_0);
                const f_0 = term1_0.add(term2_0);

                // f(j, 2) - compute using extrapolated values
                const val_2 = upc_2.add(gamma[1].mul(pc_2)).add(gamma[2].mul(virt_2)).add(gamma[3].mul(first_2));
                const term1_2 = eq_out_2.mul(val_2);
                const term2_2 = gamma[4].mul(F.one().sub(noop_2)).mul(eq_prod_2);
                const f_2 = term1_2.add(term2_2);

                evals[0] = evals[0].add(f_0);
                evals[1] = evals[1].add(f_2);
            }

            return evals;
        }

        /// Compute round evaluations for InstructionInputSumcheck
        /// Returns [p(0), p(2), p(3)] for degree-3 polynomial (p(1) derived from claim)
        /// Following Jolt's approach: extrapolate each MLE first, then compute products
        fn computeInstrRoundEvals(
            self: *Self,
            mles: *const InstructionInputMLEs,
            eq_outer: []const F,
            eq_product: []const F,
            gamma: F,
            gamma_sqr: F,
            half_size: usize,
        ) [3]F {
            _ = self;
            var evals: [3]F = .{ F.zero(), F.zero(), F.zero() };

            for (0..half_size) |j| {
                // Get values at bit=0 and bit=1
                const left_is_rs1_0 = mles.left_is_rs1[2 * j];
                const left_is_rs1_1 = mles.left_is_rs1[2 * j + 1];
                const rs1_0 = mles.rs1_value[2 * j];
                const rs1_1 = mles.rs1_value[2 * j + 1];
                const left_is_pc_0 = mles.left_is_pc[2 * j];
                const left_is_pc_1 = mles.left_is_pc[2 * j + 1];
                const pc_0 = mles.unexpanded_pc[2 * j];
                const pc_1 = mles.unexpanded_pc[2 * j + 1];
                const right_is_rs2_0 = mles.right_is_rs2[2 * j];
                const right_is_rs2_1 = mles.right_is_rs2[2 * j + 1];
                const rs2_0 = mles.rs2_value[2 * j];
                const rs2_1 = mles.rs2_value[2 * j + 1];
                const right_is_imm_0 = mles.right_is_imm[2 * j];
                const right_is_imm_1 = mles.right_is_imm[2 * j + 1];
                const imm_0 = mles.imm[2 * j];
                const imm_1 = mles.imm[2 * j + 1];

                const eq_out_0 = eq_outer[2 * j];
                const eq_out_1 = eq_outer[2 * j + 1];
                const eq_prod_0 = eq_product[2 * j];
                const eq_prod_1 = eq_product[2 * j + 1];

                // Linear extrapolation for each MLE: f(2) = 2*f(1) - f(0), f(3) = 3*f(1) - 2*f(0)
                const left_is_rs1_2 = left_is_rs1_1.add(left_is_rs1_1).sub(left_is_rs1_0);
                const left_is_rs1_3 = left_is_rs1_2.add(left_is_rs1_1).sub(left_is_rs1_0);
                const rs1_2 = rs1_1.add(rs1_1).sub(rs1_0);
                const rs1_3 = rs1_2.add(rs1_1).sub(rs1_0);
                const left_is_pc_2 = left_is_pc_1.add(left_is_pc_1).sub(left_is_pc_0);
                const left_is_pc_3 = left_is_pc_2.add(left_is_pc_1).sub(left_is_pc_0);
                const pc_2 = pc_1.add(pc_1).sub(pc_0);
                const pc_3 = pc_2.add(pc_1).sub(pc_0);
                const right_is_rs2_2 = right_is_rs2_1.add(right_is_rs2_1).sub(right_is_rs2_0);
                const right_is_rs2_3 = right_is_rs2_2.add(right_is_rs2_1).sub(right_is_rs2_0);
                const rs2_2 = rs2_1.add(rs2_1).sub(rs2_0);
                const rs2_3 = rs2_2.add(rs2_1).sub(rs2_0);
                const right_is_imm_2 = right_is_imm_1.add(right_is_imm_1).sub(right_is_imm_0);
                const right_is_imm_3 = right_is_imm_2.add(right_is_imm_1).sub(right_is_imm_0);
                const imm_2 = imm_1.add(imm_1).sub(imm_0);
                const imm_3 = imm_2.add(imm_1).sub(imm_0);
                const eq_out_2 = eq_out_1.add(eq_out_1).sub(eq_out_0);
                const eq_out_3 = eq_out_2.add(eq_out_1).sub(eq_out_0);
                const eq_prod_2 = eq_prod_1.add(eq_prod_1).sub(eq_prod_0);
                const eq_prod_3 = eq_prod_2.add(eq_prod_1).sub(eq_prod_0);

                // left = left_is_rs1 * rs1 + left_is_pc * pc
                // right = right_is_rs2 * rs2 + right_is_imm * imm
                // f = (eq_outer + gamma^2 * eq_product) * (right + gamma * left)
                // Degree: 1 (eq) * 2 (left*rs1) = 3

                // Compute at x_j = 0
                const left_0 = left_is_rs1_0.mul(rs1_0).add(left_is_pc_0.mul(pc_0));
                const right_0 = right_is_rs2_0.mul(rs2_0).add(right_is_imm_0.mul(imm_0));
                const eq_weight_0 = eq_out_0.add(gamma_sqr.mul(eq_prod_0));
                const f_0 = eq_weight_0.mul(right_0.add(gamma.mul(left_0)));

                // Compute at x_j = 2
                const left_2 = left_is_rs1_2.mul(rs1_2).add(left_is_pc_2.mul(pc_2));
                const right_2 = right_is_rs2_2.mul(rs2_2).add(right_is_imm_2.mul(imm_2));
                const eq_weight_2 = eq_out_2.add(gamma_sqr.mul(eq_prod_2));
                const f_2 = eq_weight_2.mul(right_2.add(gamma.mul(left_2)));

                // Compute at x_j = 3
                const left_3 = left_is_rs1_3.mul(rs1_3).add(left_is_pc_3.mul(pc_3));
                const right_3 = right_is_rs2_3.mul(rs2_3).add(right_is_imm_3.mul(imm_3));
                const eq_weight_3 = eq_out_3.add(gamma_sqr.mul(eq_prod_3));
                const f_3 = eq_weight_3.mul(right_3.add(gamma.mul(left_3)));

                evals[0] = evals[0].add(f_0);
                evals[1] = evals[1].add(f_2);
                evals[2] = evals[2].add(f_3);
            }

            return evals;
        }

        /// Compute round evaluations for RegistersClaimReduction
        /// Returns [p(0), p(2)] for degree-2 polynomial (p(1) derived from claim)
        /// Following Jolt's approach: extrapolate each MLE first, then compute products
        fn computeRegRoundEvals(
            self: *Self,
            mles: *const RegistersMLEs,
            eq_spartan: []const F,
            gamma: F,
            gamma_sqr: F,
            half_size: usize,
        ) [2]F {
            _ = self;
            var evals: [2]F = .{ F.zero(), F.zero() };

            for (0..half_size) |j| {
                const rd_0 = mles.rd_write_value[2 * j];
                const rd_1 = mles.rd_write_value[2 * j + 1];
                const rs1_0 = mles.rs1_value[2 * j];
                const rs1_1 = mles.rs1_value[2 * j + 1];
                const rs2_0 = mles.rs2_value[2 * j];
                const rs2_1 = mles.rs2_value[2 * j + 1];

                const eq_0 = eq_spartan[2 * j];
                const eq_1 = eq_spartan[2 * j + 1];

                // Linear extrapolation for each MLE: f(2) = 2*f(1) - f(0)
                const rd_2 = rd_1.add(rd_1).sub(rd_0);
                const rs1_2 = rs1_1.add(rs1_1).sub(rs1_0);
                const rs2_2 = rs2_1.add(rs2_1).sub(rs2_0);
                const eq_2 = eq_1.add(eq_1).sub(eq_0);

                // f = eq(r_spartan, x) * (rd + gamma * rs1 + gamma^2 * rs2)
                // Degree: 1 * 1 = 2 (eq is linear, register poly is linear)

                const reg_val_0 = rd_0.add(gamma.mul(rs1_0)).add(gamma_sqr.mul(rs2_0));
                const reg_val_2 = rd_2.add(gamma.mul(rs1_2)).add(gamma_sqr.mul(rs2_2));

                const f_0 = eq_0.mul(reg_val_0);
                const f_2 = eq_2.mul(reg_val_2);

                evals[0] = evals[0].add(f_0);
                evals[1] = evals[1].add(f_2);
            }

            return evals;
        }

        /// Convert evaluations at 0, 1, 2, ... to polynomial coefficients
        fn evalsToCoeffs(self: *Self, evals: []const F, degree: usize) ![]F {
            const coeffs = try self.allocator.alloc(F, degree + 1);

            if (degree == 2) {
                // For degree 2: p(x) = c0 + c1*x + c2*x^2
                // p(0) = c0
                // p(1) = c0 + c1 + c2
                // p(2) = c0 + 2*c1 + 4*c2
                //
                // c0 = p(0)
                // c1 + c2 = p(1) - p(0)
                // 2*c1 + 4*c2 = p(2) - p(0)
                //
                // From last two: 2*c2 = p(2) - p(0) - 2*(p(1) - p(0)) = p(2) - 2*p(1) + p(0)
                // c2 = (p(2) - 2*p(1) + p(0)) / 2
                // c1 = p(1) - p(0) - c2
                const p0 = evals[0];
                const p1 = evals[1];
                const p2 = evals[2];

                const two = F.fromU64(2);
                const two_inv = two.inverse() orelse F.one();

                coeffs[0] = p0;
                const c2 = p2.sub(p1.add(p1)).add(p0).mul(two_inv);
                coeffs[2] = c2;
                coeffs[1] = p1.sub(p0).sub(c2);
            } else if (degree == 3) {
                // For degree 3: p(x) = c0 + c1*x + c2*x^2 + c3*x^3
                // p(0) = c0
                // p(1) = c0 + c1 + c2 + c3
                // p(2) = c0 + 2*c1 + 4*c2 + 8*c3
                // p(3) = c0 + 3*c1 + 9*c2 + 27*c3
                //
                // This is a Vandermonde system. Solve directly.
                const p0 = evals[0];
                const p1 = evals[1];
                const p2 = evals[2];
                const p3 = evals[3];

                // Using Lagrange basis conversion
                // c0 = p0
                // c1, c2, c3 from solving the system
                //
                // For simplicity, use the Lagrange formula:
                // p(x) = sum_i p_i * L_i(x)
                // where L_i(x) = prod_{j != i} (x - j) / (i - j)
                //
                // L_0(x) = (x-1)(x-2)(x-3) / ((-1)(-2)(-3)) = (x-1)(x-2)(x-3) / (-6)
                // L_1(x) = (x-0)(x-2)(x-3) / ((1)(-1)(-2)) = x(x-2)(x-3) / 2
                // L_2(x) = (x-0)(x-1)(x-3) / ((2)(1)(-1)) = x(x-1)(x-3) / (-2)
                // L_3(x) = (x-0)(x-1)(x-2) / ((3)(2)(1)) = x(x-1)(x-2) / 6
                //
                // Expand and collect:
                // c0 = p0*(-1/6)(1*2*3) + p1*(1/2)(0) + p2*(-1/2)(0) + p3*(1/6)(0) = p0
                // ... (tedious to expand)
                //
                // Use a different approach: finite differences
                // d1 = p1 - p0, d2 = p2 - p1, d3 = p3 - p2 (first differences)
                // dd1 = d2 - d1, dd2 = d3 - d2 (second differences)
                // ddd = dd2 - dd1 (third difference)
                //
                // c3 = ddd / 6
                // c2 = (dd1 - 3*c3) / 2 = dd1/2 - 3*c3/2
                // c1 = d1 - c2 - c3
                // c0 = p0

                const d1 = p1.sub(p0);
                const d2 = p2.sub(p1);
                const d3 = p3.sub(p2);
                const dd1 = d2.sub(d1);
                const dd2 = d3.sub(d2);
                const ddd = dd2.sub(dd1);

                const six_inv = F.fromU64(6).inverse() orelse F.one();
                const two_inv = F.fromU64(2).inverse() orelse F.one();

                const c3 = ddd.mul(six_inv);
                // c2 = (dd1 - 6*c3) / 2 = dd1/2 - 3*c3
                const c2 = dd1.mul(two_inv).sub(c3.mul(F.fromU64(3)));
                const c1 = d1.sub(c2).sub(c3);

                coeffs[0] = p0;
                coeffs[1] = c1;
                coeffs[2] = c2;
                coeffs[3] = c3;
            } else {
                // Fallback: linear
                coeffs[0] = evals[0];
                if (degree >= 1 and evals.len > 1) {
                    coeffs[1] = evals[1].sub(evals[0]);
                } else {
                    for (1..degree + 1) |i| {
                        coeffs[i] = F.zero();
                    }
                }
            }

            return coeffs;
        }

        /// Evaluate polynomial at a point
        fn evaluatePolyAtPoint(self: *Self, coeffs: []const F, x: F) F {
            _ = self;
            var result = F.zero();
            var x_pow = F.one();
            for (coeffs) |coeff| {
                result = result.add(coeff.mul(x_pow));
                x_pow = x_pow.mul(x);
            }
            return result;
        }

        /// Bind eq evaluations (halve the table)
        fn bindEqEvals(self: *Self, evals: []F, r: F) void {
            _ = self;
            const half = evals.len / 2;
            for (0..half) |i| {
                // eq(r, x) after binding at r: eq_new[i] = (1-r)*eq[2i] + r*eq[2i+1]
                const low = evals[2 * i];
                const high = evals[2 * i + 1];
                evals[i] = F.one().sub(r).mul(low).add(r.mul(high));
            }
        }

        /// Bind MLE polynomial (in-place, halve size)
        fn bindMLE(mle: []F, r: F) void {
            const half = mle.len / 2;
            for (0..half) |i| {
                const low = mle[2 * i];
                const high = mle[2 * i + 1];
                mle[i] = F.one().sub(r).mul(low).add(r.mul(high));
            }
        }
    };
}
