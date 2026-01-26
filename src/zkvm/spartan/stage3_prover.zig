//! Stage 3 Batched Sumcheck Prover for Jolt Compatibility
//!
//! Stage 3 in Jolt consists of 3 batched sumcheck instances:
//! 1. ShiftSumcheck - proves shift polynomial relations (degree 2) - uses prefix-suffix
//! 2. InstructionInputSumcheck - proves operand computation (degree 3) - uses GruenSplitEq
//! 3. RegistersClaimReduction - reduces register value claims (degree 2) - uses prefix-suffix
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
        const EqPlusOnePrefixSuffixPoly = poly_mod.EqPlusOnePrefixSuffixPoly;
        const MleEvaluation = poly_mod.MleEvaluation;

        // Degree bound for round polynomials
        const SHIFT_DEGREE: usize = 2; // ShiftSumcheck is degree 2
        const INSTR_DEGREE: usize = 3; // InstructionInputSumcheck is degree 3
        const REG_DEGREE: usize = 2; // RegistersClaimReduction is degree 2

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
            std.debug.print("[ZOLT] STAGE3_SHIFT: gamma_powers[4] = {{ {any} }}\n", .{shift_gamma_powers[4].toBytes()});

            // InstructionInputParams::new - derive 1 gamma
            const instr_gamma = transcript.challengeScalarFull();
            const instr_gamma_sqr = instr_gamma.mul(instr_gamma);

            // RegistersClaimReductionSumcheckParams::new - derive 1 gamma
            const reg_gamma = transcript.challengeScalarFull();
            const reg_gamma_sqr = reg_gamma.mul(reg_gamma);

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

            // Compute the combined initial claim
            var combined_claim = shift_input_claim.mul(batching_coeffs[0]);
            combined_claim = combined_claim.add(instr_input_claim.mul(batching_coeffs[1]));
            combined_claim = combined_claim.add(reg_input_claim.mul(batching_coeffs[2]));

            // Allocate challenges
            var challenges = try self.allocator.alloc(F, num_rounds);

            // =========================================================================
            // Initialize Prefix-Suffix Provers for Shift and Registers
            // =========================================================================

            // ShiftSumcheck uses EqPlusOnePrefixSuffixPoly decomposition with 4 (P,Q) pairs
            var shift_prover = try ShiftPrefixSuffixProver(F).init(
                self.allocator,
                cycle_witnesses,
                trace_len,
                r_outer,
                r_product,
                shift_gamma_powers,
            );
            defer shift_prover.deinit();

            // RegistersClaimReduction uses EqPolynomial prefix-suffix with 1 (P,Q) pair
            var reg_prover = try RegistersPrefixSuffixProver(F).init(
                self.allocator,
                cycle_witnesses,
                trace_len,
                r_outer, // r_spartan = r_outer
                reg_gamma,
                reg_gamma_sqr,
            );
            defer reg_prover.deinit();

            // InstructionInputSumcheck uses direct computation (no prefix-suffix in Jolt)
            var instr_prover = try InstructionInputProver(F).init(
                self.allocator,
                cycle_witnesses,
                trace_len,
                r_outer,
                r_product,
                instr_gamma,
                instr_gamma_sqr,
            );
            defer instr_prover.deinit();

            // DEBUG: Check initial witness values and compute initial sum
            std.debug.print("\n[ZOLT] INSTR_INIT: trace_len = {}, prover.current_size = {}\n", .{ trace_len, instr_prover.current_size });
            // Compute the full sum to verify it equals input_claim
            var full_sum = F.zero();
            var left_sum = F.zero();
            var right_sum = F.zero();
            for (0..trace_len) |i| {
                const left_i = instr_prover.left_is_rs1[i].mul(instr_prover.rs1_value[i])
                    .add(instr_prover.left_is_pc[i].mul(instr_prover.unexpanded_pc[i]));
                const right_i = instr_prover.right_is_rs2[i].mul(instr_prover.rs2_value[i])
                    .add(instr_prover.right_is_imm[i].mul(instr_prover.imm[i]));
                const eq_weight_i = instr_prover.eq_outer[i].add(instr_gamma_sqr.mul(instr_prover.eq_product[i]));
                full_sum = full_sum.add(eq_weight_i.mul(right_i.add(instr_gamma.mul(left_i))));

                // Also compute eq-weighted sums of left and right separately for each stage
                left_sum = left_sum.add(instr_prover.eq_outer[i].mul(left_i));
                right_sum = right_sum.add(instr_prover.eq_outer[i].mul(right_i));
            }
            std.debug.print("[ZOLT] INSTR_INIT: full_sum = {{ {any} }}\n", .{full_sum.toBytes()[0..8]});
            std.debug.print("[ZOLT] INSTR_INIT: instr_input_claim = {{ {any} }}\n", .{instr_input_claim.toBytes()[0..8]});
            std.debug.print("[ZOLT] INSTR_INIT: sum_equals_claim = {}\n", .{full_sum.eql(instr_input_claim)});

            // The eq-weighted left_sum should equal left_1 from opening claims
            // left_1 = LeftInstructionInput evaluated at r_outer
            const left_1_from_openings = opening_claims.get(.{ .Virtual = .{ .poly = .LeftInstructionInput, .sumcheck_id = .SpartanOuter } }) orelse F.zero();
            const right_1_from_openings = opening_claims.get(.{ .Virtual = .{ .poly = .RightInstructionInput, .sumcheck_id = .SpartanOuter } }) orelse F.zero();
            std.debug.print("[ZOLT] INSTR_INIT: eq_weighted_left_sum = {{ {any} }}\n", .{left_sum.toBytes()[0..8]});
            std.debug.print("[ZOLT] INSTR_INIT: left_1_from_openings = {{ {any} }}\n", .{left_1_from_openings.toBytes()[0..8]});
            std.debug.print("[ZOLT] INSTR_INIT: left_match = {}\n", .{left_sum.eql(left_1_from_openings)});
            std.debug.print("[ZOLT] INSTR_INIT: eq_weighted_right_sum = {{ {any} }}\n", .{right_sum.toBytes()[0..8]});
            std.debug.print("[ZOLT] INSTR_INIT: right_1_from_openings = {{ {any} }}\n", .{right_1_from_openings.toBytes()[0..8]});
            std.debug.print("[ZOLT] INSTR_INIT: right_match = {}\n", .{right_sum.eql(right_1_from_openings)});

            // Debug: find mismatches
            {
                var mismatch_count: usize = 0;
                for (0..trace_len) |idx| {
                    const right_computed = instr_prover.right_is_rs2[idx].mul(instr_prover.rs2_value[idx])
                        .add(instr_prover.right_is_imm[idx].mul(instr_prover.imm[idx]));
                    const right_from_witness = if (idx < cycle_witnesses.len)
                        cycle_witnesses[idx].values[R1CSInputIndex.RightInstructionInput.toIndex()]
                    else
                        F.zero();
                    if (!right_computed.eql(right_from_witness)) {
                        mismatch_count += 1;
                        if (mismatch_count <= 5) {
                            std.debug.print("[ZOLT] INSTR_INIT: MISMATCH at cycle {}: computed = {{ {any} }}, witness = {{ {any} }}\n", .{ idx, right_computed.toBytes()[0..8], right_from_witness.toBytes()[0..8] });
                            std.debug.print("[ZOLT]   right_is_rs2 = {{ {any} }}, rs2 = {{ {any} }}\n", .{ instr_prover.right_is_rs2[idx].toBytes()[0..8], instr_prover.rs2_value[idx].toBytes()[0..8] });
                            std.debug.print("[ZOLT]   right_is_imm = {{ {any} }}, imm = {{ {any} }}\n", .{ instr_prover.right_is_imm[idx].toBytes()[0..8], instr_prover.imm[idx].toBytes()[0..8] });
                            // Also print the instruction for this cycle
                            if (idx < cycle_witnesses.len) {
                                const instr = cycle_witnesses[idx].values[R1CSInputIndex.Product.toIndex()]; // Using Product as proxy (need actual instruction)
                                _ = instr;
                                // Get opcode from witness if available
                            }
                        }
                    }
                }
                std.debug.print("[ZOLT] INSTR_INIT: right mismatch_count = {} / {}\n", .{ mismatch_count, trace_len });
            }

            // Track current claims for each instance
            var current_shift_claim = shift_input_claim;
            var current_instr_claim = instr_input_claim;
            var current_reg_claim = reg_input_claim;

            // Run sumcheck rounds
            for (0..num_rounds) |round| {
                std.debug.print("\n[ZOLT] STAGE3_ROUND_{}: current_claim = {{ {any} }}\n", .{ round, combined_claim.toBytes() });

                // Compute round polynomial for each instance
                // ShiftSumcheck: degree 2
                const shift_evals = shift_prover.computeRoundEvals(current_shift_claim);

                // InstructionInputSumcheck: degree 3
                const instr_evals = instr_prover.computeRoundEvals(current_instr_claim);

                // DEBUG: Verify instr_evals at round 0
                if (round == 0) {
                    // Manually compute p(0) and p(1) sums
                    var manual_p0 = F.zero();
                    var manual_p1 = F.zero();
                    const half = instr_prover.current_size / 2;
                    for (0..half) |j| {
                        const left_0 = instr_prover.left_is_rs1[2 * j].mul(instr_prover.rs1_value[2 * j])
                            .add(instr_prover.left_is_pc[2 * j].mul(instr_prover.unexpanded_pc[2 * j]));
                        const right_0 = instr_prover.right_is_rs2[2 * j].mul(instr_prover.rs2_value[2 * j])
                            .add(instr_prover.right_is_imm[2 * j].mul(instr_prover.imm[2 * j]));
                        const eq_w_0 = instr_prover.eq_outer[2 * j].add(instr_gamma_sqr.mul(instr_prover.eq_product[2 * j]));
                        manual_p0 = manual_p0.add(eq_w_0.mul(right_0.add(instr_gamma.mul(left_0))));

                        const left_1 = instr_prover.left_is_rs1[2 * j + 1].mul(instr_prover.rs1_value[2 * j + 1])
                            .add(instr_prover.left_is_pc[2 * j + 1].mul(instr_prover.unexpanded_pc[2 * j + 1]));
                        const right_1 = instr_prover.right_is_rs2[2 * j + 1].mul(instr_prover.rs2_value[2 * j + 1])
                            .add(instr_prover.right_is_imm[2 * j + 1].mul(instr_prover.imm[2 * j + 1]));
                        const eq_w_1 = instr_prover.eq_outer[2 * j + 1].add(instr_gamma_sqr.mul(instr_prover.eq_product[2 * j + 1]));
                        manual_p1 = manual_p1.add(eq_w_1.mul(right_1.add(instr_gamma.mul(left_1))));
                    }
                    std.debug.print("[ZOLT] ROUND0_VERIFY: manual_p0 = {{ {any} }}\n", .{manual_p0.toBytes()[0..8]});
                    std.debug.print("[ZOLT] ROUND0_VERIFY: instr_evals[0] = {{ {any} }}\n", .{instr_evals[0].toBytes()[0..8]});
                    std.debug.print("[ZOLT] ROUND0_VERIFY: p0_match = {}\n", .{manual_p0.eql(instr_evals[0])});
                    std.debug.print("[ZOLT] ROUND0_VERIFY: manual_p1 = {{ {any} }}\n", .{manual_p1.toBytes()[0..8]});
                    std.debug.print("[ZOLT] ROUND0_VERIFY: derived p1 = {{ {any} }}\n", .{instr_evals[1].toBytes()[0..8]});
                    std.debug.print("[ZOLT] ROUND0_VERIFY: p0+p1 = {{ {any} }}\n", .{manual_p0.add(manual_p1).toBytes()[0..8]});
                    std.debug.print("[ZOLT] ROUND0_VERIFY: input_claim = {{ {any} }}\n", .{current_instr_claim.toBytes()[0..8]});
                }

                // RegistersClaimReduction: degree 2
                const reg_evals = reg_prover.computeRoundEvals(current_reg_claim);

                // DEBUG: After last round, manually check the formula
                if (round == num_rounds - 1) {
                    std.debug.print("[ZOLT] LAST_ROUND: instr_evals = [p0={{ {any} }}, p1={{ {any} }}, p2={{ {any} }}, p3={{ {any} }}]\n", .{
                        instr_evals[0].toBytes()[0..8],
                        instr_evals[1].toBytes()[0..8],
                        instr_evals[2].toBytes()[0..8],
                        instr_evals[3].toBytes()[0..8],
                    });

                    // Manually compute what the polynomial value should be at different points
                    // The prover should have current_size = 2 at this point
                    std.debug.print("[ZOLT] LAST_ROUND: instr_prover.current_size = {}\n", .{instr_prover.current_size});

                    // Check the sumcheck invariant: p(0) + p(1) = previous_claim
                    const p0_plus_p1 = instr_evals[0].add(instr_evals[1]);
                    std.debug.print("[ZOLT] LAST_ROUND: p0+p1 = {{ {any} }}\n", .{p0_plus_p1.toBytes()[0..8]});
                    std.debug.print("[ZOLT] LAST_ROUND: current_instr_claim = {{ {any} }}\n", .{current_instr_claim.toBytes()[0..8]});
                    std.debug.print("[ZOLT] LAST_ROUND: sumcheck_invariant_ok = {}\n", .{p0_plus_p1.eql(current_instr_claim)});

                    // Manually compute what f(0) and f(1) should be from the raw values
                    // Before bind, current_size = 2, so we have values at indices 0 and 1
                    const left_0 = instr_prover.left_is_rs1[0].mul(instr_prover.rs1_value[0])
                        .add(instr_prover.left_is_pc[0].mul(instr_prover.unexpanded_pc[0]));
                    const right_0 = instr_prover.right_is_rs2[0].mul(instr_prover.rs2_value[0])
                        .add(instr_prover.right_is_imm[0].mul(instr_prover.imm[0]));
                    const eq_weight_0 = instr_prover.eq_outer[0].add(instr_gamma_sqr.mul(instr_prover.eq_product[0]));
                    const f_0 = eq_weight_0.mul(right_0.add(instr_gamma.mul(left_0)));

                    const left_1 = instr_prover.left_is_rs1[1].mul(instr_prover.rs1_value[1])
                        .add(instr_prover.left_is_pc[1].mul(instr_prover.unexpanded_pc[1]));
                    const right_1 = instr_prover.right_is_rs2[1].mul(instr_prover.rs2_value[1])
                        .add(instr_prover.right_is_imm[1].mul(instr_prover.imm[1]));
                    const eq_weight_1 = instr_prover.eq_outer[1].add(instr_gamma_sqr.mul(instr_prover.eq_product[1]));
                    const f_1 = eq_weight_1.mul(right_1.add(instr_gamma.mul(left_1)));

                    std.debug.print("[ZOLT] LAST_ROUND: manual_f0 = {{ {any} }}\n", .{f_0.toBytes()[0..8]});
                    std.debug.print("[ZOLT] LAST_ROUND: manual_f1 = {{ {any} }}\n", .{f_1.toBytes()[0..8]});
                    std.debug.print("[ZOLT] LAST_ROUND: f0_match = {}, f1_match = {}\n", .{ f_0.eql(instr_evals[0]), f_1.eql(instr_evals[1]) });

                    // Check actual witness values at index 1
                    std.debug.print("[ZOLT] LAST_ROUND: left_is_rs1[1] = {{ {any} }}\n", .{instr_prover.left_is_rs1[1].toBytes()[0..8]});
                    std.debug.print("[ZOLT] LAST_ROUND: rs1_value[1] = {{ {any} }}\n", .{instr_prover.rs1_value[1].toBytes()[0..8]});
                    std.debug.print("[ZOLT] LAST_ROUND: eq_outer[1] = {{ {any} }}\n", .{instr_prover.eq_outer[1].toBytes()[0..8]});
                    std.debug.print("[ZOLT] LAST_ROUND: eq_product[1] = {{ {any} }}\n", .{instr_prover.eq_product[1].toBytes()[0..8]});
                    std.debug.print("[ZOLT] LAST_ROUND: eq_weight_1 = {{ {any} }}\n", .{eq_weight_1.toBytes()[0..8]});
                    std.debug.print("[ZOLT] LAST_ROUND: left_1 = {{ {any} }}\n", .{left_1.toBytes()[0..8]});
                    std.debug.print("[ZOLT] LAST_ROUND: right_1 = {{ {any} }}\n", .{right_1.toBytes()[0..8]});
                }

                // Debug: Check individual prover invariants
                if (round < 3) {
                    const shift_sum = shift_evals[0].add(shift_evals[1]);
                    const instr_sum = instr_evals[0].add(instr_evals[1]);
                    const reg_sum = reg_evals[0].add(reg_evals[1]);
                    std.debug.print("[ZOLT] STAGE3_ROUND_{}: shift_p0+p1 = {{ {any} }}, shift_claim = {{ {any} }}, match={}\n", .{ round, shift_sum.toBytes()[0..8], current_shift_claim.toBytes()[0..8], shift_sum.eql(current_shift_claim) });
                    std.debug.print("[ZOLT] STAGE3_ROUND_{}: instr_p0+p1 = {{ {any} }}, instr_claim = {{ {any} }}, match={}\n", .{ round, instr_sum.toBytes()[0..8], current_instr_claim.toBytes()[0..8], instr_sum.eql(current_instr_claim) });
                    std.debug.print("[ZOLT] STAGE3_ROUND_{}: reg_p0+p1 = {{ {any} }}, reg_claim = {{ {any} }}, match={}\n", .{ round, reg_sum.toBytes()[0..8], current_reg_claim.toBytes()[0..8], reg_sum.eql(current_reg_claim) });
                }
                std.debug.print("[ZOLT] STAGE3_ROUND_{}: shift_p0 = {{ {any} }}\n", .{ round, shift_evals[0].toBytes() });
                std.debug.print("[ZOLT] STAGE3_ROUND_{}: shift_p1 = {{ {any} }}\n", .{ round, shift_evals[1].toBytes() });

                // Combine round polynomials (all evaluated at 0, 1, 2, 3)
                // batched_poly = coeff[0] * shift_poly + coeff[1] * instr_poly + coeff[2] * reg_poly
                // NOTE: shift and reg are degree-2, but we need their values at x=3 via extrapolation
                // Linear extrapolation: p(3) = 3*p(1) - 3*p(0) + p(-1) is wrong for degree-2
                // Quadratic extrapolation: p(3) = 3*p(2) - 3*p(1) + p(0)
                const shift_p3 = shift_evals[2].mul(F.fromU64(3)).sub(shift_evals[1].mul(F.fromU64(3))).add(shift_evals[0]);
                const reg_p3 = reg_evals[2].mul(F.fromU64(3)).sub(reg_evals[1].mul(F.fromU64(3))).add(reg_evals[0]);

                var combined_evals: [4]F = undefined;
                for (0..4) |i| {
                    const shift_val = if (i < 3) shift_evals[i] else shift_p3;
                    const instr_val = instr_evals[i];
                    const reg_val = if (i < 3) reg_evals[i] else reg_p3;
                    combined_evals[i] = shift_val.mul(batching_coeffs[0])
                        .add(instr_val.mul(batching_coeffs[1]))
                        .add(reg_val.mul(batching_coeffs[2]));
                }

                // Debug: Print evaluations
                if (round < 3) {
                    std.debug.print("[ZOLT] STAGE3_ROUND_{}: p0 = {{ {any} }}\n", .{ round, combined_evals[0].toBytes() });
                    std.debug.print("[ZOLT] STAGE3_ROUND_{}: p1 = {{ {any} }}\n", .{ round, combined_evals[1].toBytes() });
                    std.debug.print("[ZOLT] STAGE3_ROUND_{}: p0+p1 = {{ {any} }}\n", .{ round, combined_evals[0].add(combined_evals[1]).toBytes() });
                    std.debug.print("[ZOLT] STAGE3_ROUND_{}: current_claim (should match p0+p1) = {{ {any} }}\n", .{ round, combined_claim.toBytes() });
                }

                // Convert evaluations to coefficients
                const combined_coeffs = try self.evalsToCoeffs(&combined_evals, 3);
                defer self.allocator.free(combined_coeffs);

                // Debug: Print all coefficients including c1
                if (round < 3) {
                    std.debug.print("[ZOLT] STAGE3_ROUND_{}: c1 = {{ {any} }}\n", .{ round, combined_coeffs[1].toBytes() });
                }

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

                // Debug: Print compressed coefficients
                std.debug.print("[ZOLT] STAGE3_ROUND_{}: c0 = {{ {any} }}\n", .{ round, compressed[0].toBytes() });
                std.debug.print("[ZOLT] STAGE3_ROUND_{}: c2 = {{ {any} }}\n", .{ round, compressed[1].toBytes() });
                std.debug.print("[ZOLT] STAGE3_ROUND_{}: c3 = {{ {any} }}\n", .{ round, compressed[2].toBytes() });

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

                // DEBUG: Verify combined_claim equals batched sum of individual claims
                if (round < 3) {
                    // Test: evaluate combined poly at point 0 - should equal combined_evals[0]
                    const test_p0 = self.evaluatePolyAtPoint(combined_coeffs, F.zero());
                    const expect_p0 = combined_evals[0];
                    std.debug.print("[ZOLT] STAGE3_ROUND_{}: combined_poly(0) = {{ {any} }}, expect = {{ {any} }}, match={}\n", .{ round, test_p0.toBytes()[0..8], expect_p0.toBytes()[0..8], test_p0.eql(expect_p0) });

                    // Direct sum check
                    const direct_combined = batching_coeffs[0].mul(self.evaluatePolyAtPoint(shift_coeffs, r_j))
                        .add(batching_coeffs[1].mul(self.evaluatePolyAtPoint(instr_coeffs, r_j)))
                        .add(batching_coeffs[2].mul(self.evaluatePolyAtPoint(reg_coeffs, r_j)));
                    std.debug.print("[ZOLT] STAGE3_ROUND_{}: direct_combined = {{ {any} }}\n", .{ round, direct_combined.toBytes()[0..8] });
                    std.debug.print("[ZOLT] STAGE3_ROUND_{}: combined_claim = {{ {any} }}\n", .{ round, combined_claim.toBytes()[0..8] });

                    const batched_sum = batching_coeffs[0].mul(current_shift_claim)
                        .add(batching_coeffs[1].mul(current_instr_claim))
                        .add(batching_coeffs[2].mul(current_reg_claim));
                    std.debug.print("[ZOLT] STAGE3_ROUND_{}: batched_sum = {{ {any} }}\n", .{ round, batched_sum.toBytes()[0..8] });
                    if (!batched_sum.eql(combined_claim)) {
                        std.debug.print("[ZOLT] STAGE3_ROUND_{}: MISMATCH: batched_sum != combined_claim!\n", .{round});
                    }
                }

                // Bind all provers at r_j
                shift_prover.bind(r_j);
                instr_prover.bind(r_j);
                reg_prover.bind(r_j);

                // DEBUG: Track nonzero count and verify sumcheck invariant after each bind
                {
                    // Verify sumcheck invariant: does the actual f(0) + f(1) sum match?
                    // At this point we've just bound with r_j, so current_size is halved
                    // Let's check the NEXT round's invariant by computing f(0) and f(1) from the new bound values
                    if (round < num_rounds - 1) {
                        // After binding, current_size is halved
                        const next_half = instr_prover.current_size / 2;
                        if (next_half > 0) {
                            // Compute f(0) sum over the next round's indices
                            var f0_sum = F.zero();
                            var f1_sum = F.zero();
                            for (0..next_half) |j| {
                                const left_0 = instr_prover.left_is_rs1[2 * j].mul(instr_prover.rs1_value[2 * j])
                                    .add(instr_prover.left_is_pc[2 * j].mul(instr_prover.unexpanded_pc[2 * j]));
                                const right_0 = instr_prover.right_is_rs2[2 * j].mul(instr_prover.rs2_value[2 * j])
                                    .add(instr_prover.right_is_imm[2 * j].mul(instr_prover.imm[2 * j]));
                                const eq_w_0 = instr_prover.eq_outer[2 * j].add(instr_gamma_sqr.mul(instr_prover.eq_product[2 * j]));
                                const contrib_0 = eq_w_0.mul(right_0.add(instr_gamma.mul(left_0)));
                                f0_sum = f0_sum.add(contrib_0);

                                const left_1 = instr_prover.left_is_rs1[2 * j + 1].mul(instr_prover.rs1_value[2 * j + 1])
                                    .add(instr_prover.left_is_pc[2 * j + 1].mul(instr_prover.unexpanded_pc[2 * j + 1]));
                                const right_1 = instr_prover.right_is_rs2[2 * j + 1].mul(instr_prover.rs2_value[2 * j + 1])
                                    .add(instr_prover.right_is_imm[2 * j + 1].mul(instr_prover.imm[2 * j + 1]));
                                const eq_w_1 = instr_prover.eq_outer[2 * j + 1].add(instr_gamma_sqr.mul(instr_prover.eq_product[2 * j + 1]));
                                const contrib_1 = eq_w_1.mul(right_1.add(instr_gamma.mul(left_1)));
                                f1_sum = f1_sum.add(contrib_1);
                            }
                            const total_sum = f0_sum.add(f1_sum);
                            // Compare with the updated current_instr_claim (which was just set to p(r_j))
                            const matches = total_sum.eql(current_instr_claim);
                            if (round >= 5 or !matches) {
                                std.debug.print("[ZOLT] VERIFY_ROUND_{}: actual_f0+f1 = {{ {any} }}, current_instr_claim = {{ {any} }}, match={}\n", .{ round + 1, total_sum.toBytes()[0..8], current_instr_claim.toBytes()[0..8], matches });
                            }
                        }
                    }
                }
            }

            std.debug.print("\n[ZOLT] STAGE3_FINAL: output_claim = {{ {any} }}\n", .{combined_claim.toBytes()});

            // DEBUG: Compute expected_output_claim like verifier
            {
                // Get final opening claims from provers
                const s_claims = shift_prover.finalClaims();
                const i_claims = instr_prover.finalClaims();
                const r_claims = reg_prover.finalClaims();

                // Compute eq+1 evaluations at final challenge point
                // NOTE: Jolt's verifier uses normalize_opening_point which REVERSES challenges for BigEndian
                const reversed_challenges = try self.allocator.alloc(F, challenges.len);
                defer self.allocator.free(reversed_challenges);
                for (0..challenges.len) |i| {
                    reversed_challenges[i] = challenges[challenges.len - 1 - i];
                }

                var eq_plus_one_outer = try poly_mod.EqPlusOnePolynomial(F).init(self.allocator, r_outer);
                defer eq_plus_one_outer.deinit();
                const eq_plus_one_r_outer = eq_plus_one_outer.evaluate(reversed_challenges);

                var eq_plus_one_prod = try poly_mod.EqPlusOnePolynomial(F).init(self.allocator, r_product);
                defer eq_plus_one_prod.deinit();
                const eq_plus_one_r_prod = eq_plus_one_prod.evaluate(reversed_challenges);

                std.debug.print("[ZOLT] STAGE3_DEBUG: challenges[0] = {{ {any} }}\n", .{challenges[0].toBytes()[0..8]});
                std.debug.print("[ZOLT] STAGE3_DEBUG: reversed_challenges[0] = {{ {any} }}\n", .{reversed_challenges[0].toBytes()[0..8]});

                // Compute shift_expected = eq+1(r_outer, r_final) * [upc + γ*pc + γ²*virt + γ³*first] + γ⁴*(1-noop)*eq+1(r_prod, r_final)
                const shift_val = s_claims.unexpanded_pc
                    .add(shift_gamma_powers[1].mul(s_claims.pc))
                    .add(shift_gamma_powers[2].mul(s_claims.is_virtual))
                    .add(shift_gamma_powers[3].mul(s_claims.is_first_in_sequence));
                const shift_expected = eq_plus_one_r_outer.mul(shift_val)
                    .add(shift_gamma_powers[4].mul(F.one().sub(s_claims.is_noop)).mul(eq_plus_one_r_prod));

                std.debug.print("\n[ZOLT] STAGE3_DEBUG: shift_val = {{ {any} }}\n", .{shift_val.toBytes()[0..8]});
                std.debug.print("[ZOLT] STAGE3_DEBUG: shift_expected = {{ {any} }}\n", .{shift_expected.toBytes()});
                std.debug.print("[ZOLT] STAGE3_DEBUG: current_shift_claim = {{ {any} }}\n", .{current_shift_claim.toBytes()});
                std.debug.print("[ZOLT] STAGE3_DEBUG: shift_match = {}\n", .{shift_expected.eql(current_shift_claim)});

                // Check prover's eq+1 values
                const prover_eq_plus_one_outer = shift_prover.phase2_eq_plus_one_outer.?[0];
                const prover_eq_plus_one_prod = shift_prover.phase2_eq_plus_one_prod.?[0];
                std.debug.print("[ZOLT] STAGE3_DEBUG: prover eq+1_outer = {{ {any} }}\n", .{prover_eq_plus_one_outer.toBytes()});
                std.debug.print("[ZOLT] STAGE3_DEBUG: verifier eq+1_outer = {{ {any} }}\n", .{eq_plus_one_r_outer.toBytes()});
                std.debug.print("[ZOLT] STAGE3_DEBUG: eq+1_outer match = {}\n", .{prover_eq_plus_one_outer.eql(eq_plus_one_r_outer)});
                std.debug.print("[ZOLT] STAGE3_DEBUG: prover eq+1_prod = {{ {any} }}\n", .{prover_eq_plus_one_prod.toBytes()});
                std.debug.print("[ZOLT] STAGE3_DEBUG: verifier eq+1_prod = {{ {any} }}\n", .{eq_plus_one_r_prod.toBytes()});
                std.debug.print("[ZOLT] STAGE3_DEBUG: eq+1_prod match = {}\n", .{prover_eq_plus_one_prod.eql(eq_plus_one_r_prod)});

                // Compute InstructionInput expected_output_claim
                var eq_outer = try poly_mod.EqPolynomial(F).init(self.allocator, r_outer);
                defer eq_outer.deinit();
                const eq_r_stage_1 = eq_outer.evaluate(reversed_challenges);

                var eq_prod = try poly_mod.EqPolynomial(F).init(self.allocator, r_product);
                defer eq_prod.deinit();
                const eq_r_stage_2 = eq_prod.evaluate(reversed_challenges);

                const left_instr = i_claims.left_is_rs1.mul(i_claims.rs1_value)
                    .add(i_claims.left_is_pc.mul(i_claims.unexpanded_pc));
                const right_instr = i_claims.right_is_rs2.mul(i_claims.rs2_value)
                    .add(i_claims.right_is_imm.mul(i_claims.imm));
                const instr_expected = (eq_r_stage_1.add(instr_gamma_sqr.mul(eq_r_stage_2)))
                    .mul(right_instr.add(instr_gamma.mul(left_instr)));

                std.debug.print("\n[ZOLT] STAGE3_DEBUG: eq_r_stage_1 = {{ {any} }}\n", .{eq_r_stage_1.toBytes()});
                std.debug.print("[ZOLT] STAGE3_DEBUG: eq_r_stage_2 = {{ {any} }}\n", .{eq_r_stage_2.toBytes()});
                std.debug.print("[ZOLT] STAGE3_DEBUG: left_instr (from i_claims) = {{ {any} }}\n", .{left_instr.toBytes()});
                std.debug.print("[ZOLT] STAGE3_DEBUG: right_instr (from i_claims) = {{ {any} }}\n", .{right_instr.toBytes()});

                // Compute directly from prover's final witness values
                const direct_left = instr_prover.left_is_rs1[0].mul(instr_prover.rs1_value[0])
                    .add(instr_prover.left_is_pc[0].mul(instr_prover.unexpanded_pc[0]));
                const direct_right = instr_prover.right_is_rs2[0].mul(instr_prover.rs2_value[0])
                    .add(instr_prover.right_is_imm[0].mul(instr_prover.imm[0]));
                std.debug.print("[ZOLT] STAGE3_DEBUG: direct_left = {{ {any} }}\n", .{direct_left.toBytes()});
                std.debug.print("[ZOLT] STAGE3_DEBUG: direct_right = {{ {any} }}\n", .{direct_right.toBytes()});
                std.debug.print("[ZOLT] STAGE3_DEBUG: left_match = {}, right_match = {}\n", .{ direct_left.eql(left_instr), direct_right.eql(right_instr) });

                // Now recompute instr_expected using prover's eq values
                const prover_eq_weight = instr_prover.eq_outer[0].add(instr_gamma_sqr.mul(instr_prover.eq_product[0]));
                const prover_f = prover_eq_weight.mul(direct_right.add(instr_gamma.mul(direct_left)));
                std.debug.print("[ZOLT] STAGE3_DEBUG: prover_f = {{ {any} }}\n", .{prover_f.toBytes()});

                // Check the individual claim components
                std.debug.print("[ZOLT] STAGE3_DEBUG: i_claims.left_is_rs1 = {{ {any} }}\n", .{i_claims.left_is_rs1.toBytes()[0..8]});
                std.debug.print("[ZOLT] STAGE3_DEBUG: i_claims.rs1_value = {{ {any} }}\n", .{i_claims.rs1_value.toBytes()[0..8]});
                std.debug.print("[ZOLT] STAGE3_DEBUG: i_claims.left_is_pc = {{ {any} }}\n", .{i_claims.left_is_pc.toBytes()[0..8]});
                std.debug.print("[ZOLT] STAGE3_DEBUG: i_claims.unexpanded_pc = {{ {any} }}\n", .{i_claims.unexpanded_pc.toBytes()[0..8]});

                // Check individual witness MLE values
                std.debug.print("[ZOLT] STAGE3_DEBUG: instr_prover.left_is_rs1[0] = {{ {any} }}\n", .{instr_prover.left_is_rs1[0].toBytes()[0..8]});
                std.debug.print("[ZOLT] STAGE3_DEBUG: instr_prover.rs1_value[0] = {{ {any} }}\n", .{instr_prover.rs1_value[0].toBytes()[0..8]});
                std.debug.print("[ZOLT] STAGE3_DEBUG: instr_prover.left_is_pc[0] = {{ {any} }}\n", .{instr_prover.left_is_pc[0].toBytes()[0..8]});
                std.debug.print("[ZOLT] STAGE3_DEBUG: instr_prover.unexpanded_pc[0] = {{ {any} }}\n", .{instr_prover.unexpanded_pc[0].toBytes()[0..8]});

                std.debug.print("[ZOLT] STAGE3_DEBUG: instr_prover_eq_outer[0] = {{ {any} }}\n", .{instr_prover.eq_outer[0].toBytes()});
                std.debug.print("[ZOLT] STAGE3_DEBUG: instr_prover_eq_prod[0] = {{ {any} }}\n", .{instr_prover.eq_product[0].toBytes()});
                std.debug.print("[ZOLT] STAGE3_DEBUG: instr_expected = {{ {any} }}\n", .{instr_expected.toBytes()});
                std.debug.print("[ZOLT] STAGE3_DEBUG: current_instr_claim = {{ {any} }}\n", .{current_instr_claim.toBytes()});
                std.debug.print("[ZOLT] STAGE3_DEBUG: instr_match = {}\n", .{instr_expected.eql(current_instr_claim)});

                // Compute Registers expected_output_claim
                // eq(r, r_spartan) * (rd + gamma*rs1 + gamma^2*rs2)
                const reg_val = r_claims.rd_write_value
                    .add(reg_gamma.mul(r_claims.rs1_value))
                    .add(reg_gamma_sqr.mul(r_claims.rs2_value));
                const reg_expected = eq_r_stage_1.mul(reg_val);

                std.debug.print("\n[ZOLT] STAGE3_DEBUG: r_claims.rd_write_value = {{ {any} }}\n", .{r_claims.rd_write_value.toBytes()});
                std.debug.print("[ZOLT] STAGE3_DEBUG: r_claims.rs1_value = {{ {any} }}\n", .{r_claims.rs1_value.toBytes()});
                std.debug.print("[ZOLT] STAGE3_DEBUG: r_claims.rs2_value = {{ {any} }}\n", .{r_claims.rs2_value.toBytes()});
                std.debug.print("[ZOLT] STAGE3_DEBUG: reg_gamma = {{ {any} }}\n", .{reg_gamma.toBytes()});
                std.debug.print("[ZOLT] STAGE3_DEBUG: reg_val = {{ {any} }}\n", .{reg_val.toBytes()});
                std.debug.print("[ZOLT] STAGE3_DEBUG: eq_r_stage_1 = {{ {any} }}\n", .{eq_r_stage_1.toBytes()});
                std.debug.print("[ZOLT] STAGE3_DEBUG: reg_expected = {{ {any} }}\n", .{reg_expected.toBytes()});
                std.debug.print("[ZOLT] STAGE3_DEBUG: current_reg_claim = {{ {any} }}\n", .{current_reg_claim.toBytes()});
                std.debug.print("[ZOLT] STAGE3_DEBUG: reg_match = {}\n", .{reg_expected.eql(current_reg_claim)});

                // Also compute what the prover's eq polynomial should be
                const prover_eq_final = reg_prover.phase2_eq.?[0];
                std.debug.print("[ZOLT] STAGE3_DEBUG: prover_eq_final = {{ {any} }}\n", .{prover_eq_final.toBytes()});
                std.debug.print("[ZOLT] STAGE3_DEBUG: prover_eq vs eq_r_stage_1: {}\n", .{prover_eq_final.eql(eq_r_stage_1)});

                // Compute final expected_output_claim
                const final_expected = batching_coeffs[0].mul(shift_expected)
                    .add(batching_coeffs[1].mul(instr_expected))
                    .add(batching_coeffs[2].mul(reg_expected));
                std.debug.print("\n[ZOLT] STAGE3_DEBUG: final_expected = {{ {any} }}\n", .{final_expected.toBytes()});
                std.debug.print("[ZOLT] STAGE3_DEBUG: combined_claim = {{ {any} }}\n", .{combined_claim.toBytes()});
                std.debug.print("[ZOLT] STAGE3_DEBUG: final_match = {}\n", .{final_expected.eql(combined_claim)});
            }

            // Phase 3: Compute and cache opening claims
            // After all rounds, the MLEs are bound to single values
            const shift_claims = shift_prover.finalClaims();
            const instr_claims = instr_prover.finalClaims();
            const reg_claims = reg_prover.finalClaims();

            // DEBUG: Print opening claims
            std.debug.print("\n[ZOLT] STAGE3_OPENING: Shift sumcheck claims:\n", .{});
            std.debug.print("[ZOLT] STAGE3_OPENING: unexpanded_pc = {{ {any} }}\n", .{shift_claims.unexpanded_pc.toBytes()});
            std.debug.print("[ZOLT] STAGE3_OPENING: pc = {{ {any} }}\n", .{shift_claims.pc.toBytes()});
            std.debug.print("[ZOLT] STAGE3_OPENING: is_noop = {{ {any} }}\n", .{shift_claims.is_noop.toBytes()});

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

            std.debug.print("[ZOLT] SHIFT_INPUT: next_unexpanded_pc = {{ {any} }}\n", .{next_unexpanded_pc.toBytes()});
            std.debug.print("[ZOLT] SHIFT_INPUT: next_pc = {{ {any} }}\n", .{next_pc.toBytes()});
            std.debug.print("[ZOLT] SHIFT_INPUT: next_is_virtual = {{ {any} }}\n", .{next_is_virtual.toBytes()});
            std.debug.print("[ZOLT] SHIFT_INPUT: next_is_first = {{ {any} }}\n", .{next_is_first.toBytes()});
            std.debug.print("[ZOLT] SHIFT_INPUT: next_is_noop = {{ {any} }}\n", .{next_is_noop.toBytes()});
            std.debug.print("[ZOLT] SHIFT_INPUT: gamma_powers[4] = {{ {any} }}\n", .{gamma_powers[4].toBytes()});
            // Also verify the input claim is correctly computed
            std.debug.print("[ZOLT] SHIFT_INPUT: 1 - next_is_noop = {{ {any} }}\n", .{F.one().sub(next_is_noop).toBytes()});
            std.debug.print("[ZOLT] SHIFT_INPUT: gamma^4 * (1-noop) = {{ {any} }}\n", .{gamma_powers[4].mul(F.one().sub(next_is_noop)).toBytes()});

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

            var result = rd;
            result = result.add(gamma.mul(rs1));
            result = result.add(gamma_sqr.mul(rs2));
            return result;
        }

        /// Convert evaluations at 0, 1, 2, ... to polynomial coefficients
        fn evalsToCoeffs(self: *Self, evals: []const F, degree: usize) ![]F {
            const coeffs = try self.allocator.alloc(F, degree + 1);

            if (degree == 2) {
                // For degree 2: p(x) = c0 + c1*x + c2*x^2
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
                // For degree 3: use finite differences
                const p0 = evals[0];
                const p1 = evals[1];
                const p2 = evals[2];
                const p3 = evals[3];

                const d1 = p1.sub(p0);
                const d2 = p2.sub(p1);
                const d3 = p3.sub(p2);
                const dd1 = d2.sub(d1);
                const dd2 = d3.sub(d2);
                const ddd = dd2.sub(dd1);

                const six_inv = F.fromU64(6).inverse() orelse F.one();
                const two_inv = F.fromU64(2).inverse() orelse F.one();

                const c3 = ddd.mul(six_inv);
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
    };
}

// =============================================================================
// ShiftSumcheck Prefix-Suffix Prover
// =============================================================================
//
// Uses EqPlusOnePrefixSuffixPoly decomposition with 4 (P,Q) pairs:
// - 2 pairs for r_outer (prefix_0/suffix_0, prefix_1/suffix_1)
// - 2 pairs for r_product (prefix_0/suffix_0, prefix_1/suffix_1)
//
// Phase1: First half of rounds use prefix-suffix optimization
// Phase2: Second half of rounds use materialized MLEs
// Transition: When prefix buffer size == 2

fn ShiftPrefixSuffixProver(comptime F: type) type {
    return struct {
        const Self = @This();
        const EqPlusOnePrefixSuffixPoly = poly_mod.EqPlusOnePrefixSuffixPoly;
        const EqPolynomial = poly_mod.EqPolynomial;
        const EqPlusOnePolynomial = poly_mod.EqPlusOnePolynomial;

        // P buffers (prefix polynomials) and Q buffers (accumulated witness * suffix)
        // 4 pairs: (P_0_outer, Q_0_outer), (P_1_outer, Q_1_outer),
        //          (P_0_prod, Q_0_prod), (P_1_prod, Q_1_prod)
        P_0_outer: []F,
        Q_0_outer: []F,
        P_1_outer: []F,
        Q_1_outer: []F,
        P_0_prod: []F,
        Q_0_prod: []F,
        P_1_prod: []F,
        Q_1_prod: []F,

        // Gamma powers for batching
        gamma_powers: []const F,

        // Witness MLEs (for final claims computation)
        unexpanded_pc: []F,
        pc: []F,
        is_virtual: []F,
        is_first_in_sequence: []F,
        is_noop: []F,

        // State tracking
        prefix_n_vars: usize,
        suffix_n_vars: usize,
        current_prefix_size: usize,
        current_witness_size: usize, // Track witness MLE size separately
        sumcheck_challenges: std.ArrayList(F),
        in_phase2: bool,

        // Original points (needed for Phase 2 transition)
        r_outer: []const F,
        r_product: []const F,

        // Original trace (needed for Phase 2 witness MLE reconstruction)
        cycle_witnesses: []const r1cs.R1CSCycleInputs(F),
        trace_len: usize,

        // Phase2 materialized polynomials (only allocated when transitioning)
        phase2_eq_plus_one_outer: ?[]F,
        phase2_eq_plus_one_prod: ?[]F,

        allocator: Allocator,

        pub fn init(
            allocator: Allocator,
            cycle_witnesses: []const r1cs.R1CSCycleInputs(F),
            trace_len: usize,
            r_outer: []const F,
            r_product: []const F,
            gamma_powers: []const F,
        ) !Self {
            const n_vars = r_outer.len;
            // Split r into hi (first half) and lo (second half)
            // Jolt convention: PREFIX uses r_lo, SUFFIX uses r_hi
            const split_point = n_vars / 2;
            const r_outer_hi = r_outer[0..split_point]; // First half -> used for SUFFIX
            const r_outer_lo = r_outer[split_point..]; // Second half -> used for PREFIX
            const r_prod_hi = r_product[0..split_point];
            const r_prod_lo = r_product[split_point..];

            // Sizes: prefix_size = 2^len(r_lo), suffix_size = 2^len(r_hi)
            const prefix_n_vars = r_outer_lo.len; // = n_vars - split_point
            const suffix_n_vars = r_outer_hi.len; // = split_point
            const prefix_size: usize = @as(usize, 1) << @intCast(prefix_n_vars);
            const suffix_size: usize = @as(usize, 1) << @intCast(suffix_n_vars);

            // Initialize P buffers (prefix polynomials)
            // PREFIX uses r_lo (Jolt convention)
            // P_0 = eq+1(r_lo, j) for j in [0, prefix_size)
            // P_1 = is_max(r_lo) * delta(j=0)
            const P_0_outer = try allocator.alloc(F, prefix_size);
            const P_1_outer = try allocator.alloc(F, prefix_size);
            const P_0_prod = try allocator.alloc(F, prefix_size);
            const P_1_prod = try allocator.alloc(F, prefix_size);

            // Compute eq+1(r_lo, j) - PREFIX uses r_lo (Jolt convention)
            try computeEqPlusOneEvals(allocator, r_outer_lo, P_0_outer);
            try computeEqPlusOneEvals(allocator, r_prod_lo, P_0_prod);

            // Compute is_max(r_lo) for prefix_1
            // is_max(x) = eq((1,1,...,1), x) = product of x[i]
            var is_max_outer = F.one();
            for (r_outer_lo) |r_i| {
                is_max_outer = is_max_outer.mul(r_i);
            }
            @memset(P_1_outer, F.zero());
            P_1_outer[0] = is_max_outer;

            var is_max_prod = F.one();
            for (r_prod_lo) |r_i| {
                is_max_prod = is_max_prod.mul(r_i);
            }
            @memset(P_1_prod, F.zero());
            P_1_prod[0] = is_max_prod;

            // Compute suffix evaluations (needed for Q buffer construction)
            // SUFFIX uses r_hi (Jolt convention)
            // suffix evaluations are indexed by x_hi in [0, suffix_size)
            const suffix_0_outer = try allocator.alloc(F, suffix_size);
            defer allocator.free(suffix_0_outer);
            const suffix_1_outer = try allocator.alloc(F, suffix_size);
            defer allocator.free(suffix_1_outer);
            const suffix_0_prod = try allocator.alloc(F, suffix_size);
            defer allocator.free(suffix_0_prod);
            const suffix_1_prod = try allocator.alloc(F, suffix_size);
            defer allocator.free(suffix_1_prod);

            // SUFFIX uses r_hi (Jolt convention)
            try computeEqAndEqPlusOneEvals(allocator, r_outer_hi, suffix_0_outer, suffix_1_outer);
            try computeEqAndEqPlusOneEvals(allocator, r_prod_hi, suffix_0_prod, suffix_1_prod);

            // Initialize Q buffers to zero
            const Q_0_outer = try allocator.alloc(F, prefix_size);
            const Q_1_outer = try allocator.alloc(F, prefix_size);
            const Q_0_prod = try allocator.alloc(F, prefix_size);
            const Q_1_prod = try allocator.alloc(F, prefix_size);
            @memset(Q_0_outer, F.zero());
            @memset(Q_1_outer, F.zero());
            @memset(Q_0_prod, F.zero());
            @memset(Q_1_prod, F.zero());

            // Allocate witness MLEs
            const unexpanded_pc = try allocator.alloc(F, trace_len);
            const pc = try allocator.alloc(F, trace_len);
            const is_virtual = try allocator.alloc(F, trace_len);
            const is_first_in_sequence = try allocator.alloc(F, trace_len);
            const is_noop = try allocator.alloc(F, trace_len);

            // Compute Q buffers by accumulating witness * suffix
            // Q[x_lo] = sum over x_hi of: witness(x) * suffix[x_hi]
            // where x = x_lo + (x_hi << prefix_n_vars)
            for (0..prefix_size) |x_lo| {
                var q_0_outer_acc = F.zero();
                var q_1_outer_acc = F.zero();
                var q_0_prod_acc = F.zero();
                var q_1_prod_acc = F.zero();

                for (0..suffix_size) |x_hi| {
                    const x = x_lo + (x_hi << @intCast(prefix_n_vars));
                    if (x >= trace_len) continue;

                    // Get witness values at this cycle
                    const witness = &cycle_witnesses[x].values;
                    const upc = witness[R1CSInputIndex.UnexpandedPC.toIndex()];
                    const pc_val = witness[R1CSInputIndex.PC.toIndex()];
                    const virt = witness[R1CSInputIndex.FlagVirtualInstruction.toIndex()];
                    const first = witness[R1CSInputIndex.FlagIsFirstInSequence.toIndex()];
                    const noop = witness[R1CSInputIndex.FlagIsNoop.toIndex()];

                    // Store witness values for final claims
                    unexpanded_pc[x] = upc;
                    pc[x] = pc_val;
                    is_virtual[x] = virt;
                    is_first_in_sequence[x] = first;
                    is_noop[x] = noop;

                    // Compute batched witness value v = upc + gamma*pc + gamma^2*virt + gamma^3*first
                    var v = upc;
                    v = v.add(gamma_powers[1].mul(pc_val));
                    v = v.add(gamma_powers[2].mul(virt));
                    v = v.add(gamma_powers[3].mul(first));

                    // Accumulate: Q_outer += v * suffix
                    q_0_outer_acc = q_0_outer_acc.add(v.mul(suffix_0_outer[x_hi]));
                    q_1_outer_acc = q_1_outer_acc.add(v.mul(suffix_1_outer[x_hi]));

                    // For product term: Q_prod += (1 - noop) * suffix
                    const one_minus_noop = F.one().sub(noop);
                    q_0_prod_acc = q_0_prod_acc.add(one_minus_noop.mul(suffix_0_prod[x_hi]));
                    q_1_prod_acc = q_1_prod_acc.add(one_minus_noop.mul(suffix_1_prod[x_hi]));
                }

                Q_0_outer[x_lo] = q_0_outer_acc;
                Q_1_outer[x_lo] = q_1_outer_acc;
                Q_0_prod[x_lo] = q_0_prod_acc.mul(gamma_powers[4]);
                Q_1_prod[x_lo] = q_1_prod_acc.mul(gamma_powers[4]);
            }

            // DEBUG: Print initial witness MLE values
            std.debug.print("\n[ZOLT] SHIFT_INIT: trace_len={d}, prefix_size={d}, suffix_size={d}\n", .{ trace_len, prefix_size, suffix_size });
            std.debug.print("[ZOLT] SHIFT_INIT: unexpanded_pc[0..4] = ", .{});
            for (0..@min(4, trace_len)) |i| {
                std.debug.print("{any} ", .{unexpanded_pc[i].toBytes()[0..8]});
            }
            std.debug.print("\n", .{});

            // DEBUG: Print last cycle's Next values (should be 0 for last cycle)
            const last_idx = trace_len - 1;
            const last_witness = &cycle_witnesses[last_idx].values;
            std.debug.print("[ZOLT] SHIFT_INIT: cycle_witnesses[{}].NextUPC = {any}\n", .{ last_idx, last_witness[R1CSInputIndex.NextUnexpandedPC.toIndex()].toBytes()[0..8] });
            std.debug.print("[ZOLT] SHIFT_INIT: cycle_witnesses[{}].NextPC = {any}\n", .{ last_idx, last_witness[R1CSInputIndex.NextPC.toIndex()].toBytes()[0..8] });
            std.debug.print("[ZOLT] SHIFT_INIT: cycle_witnesses[{}].NextIsVirtual = {any}\n", .{ last_idx, last_witness[R1CSInputIndex.NextIsVirtual.toIndex()].toBytes()[0..8] });
            std.debug.print("[ZOLT] SHIFT_INIT: cycle_witnesses[{}].NextIsFirst = {any}\n", .{ last_idx, last_witness[R1CSInputIndex.NextIsFirstInSequence.toIndex()].toBytes()[0..8] });

            // DEBUG: Verify NextUPC[j] = UPC[j+1] relationship for all j
            var next_shift_mismatch_count: usize = 0;
            for (0..trace_len - 1) |check_j| {
                const next_upc_j = cycle_witnesses[check_j].values[R1CSInputIndex.NextUnexpandedPC.toIndex()];
                const upc_j_plus_1 = cycle_witnesses[check_j + 1].values[R1CSInputIndex.UnexpandedPC.toIndex()];
                if (!next_upc_j.eql(upc_j_plus_1)) {
                    if (next_shift_mismatch_count < 5) {
                        std.debug.print("[ZOLT] SHIFT_INIT: MISMATCH NextUPC[{}] != UPC[{}]: {any} != {any}\n", .{ check_j, check_j + 1, next_upc_j.toBytes()[0..8], upc_j_plus_1.toBytes()[0..8] });
                    }
                    next_shift_mismatch_count += 1;
                }
            }
            if (next_shift_mismatch_count > 0) {
                std.debug.print("[ZOLT] SHIFT_INIT: Found {} mismatches in NextUPC[j] = UPC[j+1] relationship!\n", .{next_shift_mismatch_count});
            } else {
                std.debug.print("[ZOLT] SHIFT_INIT: NextUPC[j] = UPC[j+1] verified for all {} cycles\n", .{trace_len - 1});
            }

            // DEBUG: Verify grand sum = Σ P[j]*Q[j]
            var grand_sum = F.zero();
            for (0..prefix_size) |j| {
                grand_sum = grand_sum.add(P_0_outer[j].mul(Q_0_outer[j]));
                grand_sum = grand_sum.add(P_1_outer[j].mul(Q_1_outer[j]));
                grand_sum = grand_sum.add(P_0_prod[j].mul(Q_0_prod[j]));
                grand_sum = grand_sum.add(P_1_prod[j].mul(Q_1_prod[j]));
            }
            std.debug.print("[ZOLT] SHIFT_INIT: grand_sum(P*Q) = {{ {any} }}\n", .{grand_sum.toBytes()});

            // DEBUG: Compute direct sum without prefix-suffix optimization
            // sum = Σ_j eq+1(r_outer, j) * [upc(j) + γ*pc(j) + γ²*virt(j) + γ³*first(j)]
            //     + γ⁴ * Σ_j eq+1(r_prod, j) * (1 - noop(j))
            var direct_sum = F.zero();
            const j_bits = try allocator.alloc(F, n_vars);
            defer allocator.free(j_bits);
            for (0..trace_len) |j| {
                // Convert j to BIG_ENDIAN bits
                for (0..n_vars) |k| {
                    const bit_pos: u6 = @intCast(n_vars - 1 - k);
                    j_bits[k] = if ((j >> bit_pos) & 1 == 1) F.one() else F.zero();
                }

                const eq_plus_one_outer = poly_mod.EqPlusOnePolynomial(F).mle(r_outer, j_bits);
                const eq_plus_one_prod = poly_mod.EqPlusOnePolynomial(F).mle(r_product, j_bits);

                const witness = &cycle_witnesses[j].values;
                const upc = witness[R1CSInputIndex.UnexpandedPC.toIndex()];
                const pc_val = witness[R1CSInputIndex.PC.toIndex()];
                const virt = witness[R1CSInputIndex.FlagVirtualInstruction.toIndex()];
                const first = witness[R1CSInputIndex.FlagIsFirstInSequence.toIndex()];
                const noop = witness[R1CSInputIndex.FlagIsNoop.toIndex()];

                var v = upc;
                v = v.add(gamma_powers[1].mul(pc_val));
                v = v.add(gamma_powers[2].mul(virt));
                v = v.add(gamma_powers[3].mul(first));

                direct_sum = direct_sum.add(eq_plus_one_outer.mul(v));
                direct_sum = direct_sum.add(gamma_powers[4].mul(eq_plus_one_prod).mul(F.one().sub(noop)));
            }
            std.debug.print("[ZOLT] SHIFT_INIT: direct_sum = {{ {any} }}\n", .{direct_sum.toBytes()});

            // DEBUG: Compute what the input_claim should be based on "Next" polynomial evaluations
            // This uses the SAME witness but reads from NextUnexpandedPC, NextPC, etc. with EQ weighting
            var next_sum = F.zero();
            for (0..trace_len) |jj| {
                // Convert jj to BIG_ENDIAN bits
                for (0..n_vars) |k| {
                    const bit_pos: u6 = @intCast(n_vars - 1 - k);
                    j_bits[k] = if ((jj >> bit_pos) & 1 == 1) F.one() else F.zero();
                }

                const eq_outer = poly_mod.EqPolynomial(F).mle(r_outer, j_bits);
                const eq_prod = poly_mod.EqPolynomial(F).mle(r_product, j_bits);

                const witness = &cycle_witnesses[jj].values;
                const next_upc = witness[R1CSInputIndex.NextUnexpandedPC.toIndex()];
                const next_pc = witness[R1CSInputIndex.NextPC.toIndex()];
                const next_virt = witness[R1CSInputIndex.NextIsVirtual.toIndex()];
                const next_first = witness[R1CSInputIndex.NextIsFirstInSequence.toIndex()];
                const next_noop = witness[R1CSInputIndex.FlagIsNoop.toIndex()]; // FlagIsNoop is the "NextIsNoop" from product virtualization

                var next_v = next_upc;
                next_v = next_v.add(gamma_powers[1].mul(next_pc));
                next_v = next_v.add(gamma_powers[2].mul(next_virt));
                next_v = next_v.add(gamma_powers[3].mul(next_first));

                next_sum = next_sum.add(eq_outer.mul(next_v));
                next_sum = next_sum.add(gamma_powers[4].mul(eq_prod).mul(F.one().sub(next_noop)));
            }
            std.debug.print("[ZOLT] SHIFT_INIT: next_sum (using Next polys with eq) = {{ {any} }}\n", .{next_sum.toBytes()});

            // DEBUG: Compute the difference and the expected boundary term
            const diff = next_sum.sub(direct_sum);
            std.debug.print("[ZOLT] SHIFT_INIT: next_sum - direct_sum = {{ {any} }}\n", .{diff.toBytes()});

            // The boundary term should be eq(r, N-1) * (batched Next values at index N-1)
            // This is the term that's in next_sum but not in direct_sum

            // Also compare next_sum to input_claim - if r_outer is correct, they should match
            // (Assuming the opening claims were computed at r_outer)
            std.debug.print("[ZOLT] SHIFT_INIT: r_outer[0] = {{ {any} }}\n", .{r_outer[0].toBytes()[0..8]});
            std.debug.print("[ZOLT] SHIFT_INIT: r_outer[last] = {{ {any} }}\n", .{r_outer[r_outer.len - 1].toBytes()[0..8]});

            // DEBUG: Verify the relationship Next[j] = Current[j+1]
            std.debug.print("[ZOLT] SHIFT_INIT: Checking Next[j] = Current[j+1] relationship:\n", .{});
            for (0..@min(5, trace_len - 1)) |test_j| {
                _ = cycle_witnesses[test_j].values[R1CSInputIndex.UnexpandedPC.toIndex()]; // Current j
                const next_upc_j = cycle_witnesses[test_j].values[R1CSInputIndex.NextUnexpandedPC.toIndex()];
                const curr_upc_j1 = cycle_witnesses[test_j + 1].values[R1CSInputIndex.UnexpandedPC.toIndex()];
                std.debug.print("  j={d}: NextUPC[j]={any}, UPC[j+1]={any}, match={}\n", .{
                    test_j,
                    next_upc_j.toBytes()[0..8],
                    curr_upc_j1.toBytes()[0..8],
                    next_upc_j.eql(curr_upc_j1),
                });
            }

            // DEBUG: Verify eq(r, k-1) = eq+1(r, k) relationship and boundary behavior
            std.debug.print("[ZOLT] SHIFT_INIT: Verifying eq(r, k-1) = eq+1(r, k):\n", .{});

            // Check eq+1(r, 0) - this is the boundary case
            @memset(j_bits, F.zero()); // j = 0 in bits
            const eq_plus_one_at_0 = poly_mod.EqPlusOnePolynomial(F).mle(r_outer, j_bits);
            std.debug.print("  eq+1(r, 0) = {any} (should be ~0 unless r=max)\n", .{eq_plus_one_at_0.toBytes()[0..8]});

            // Check eq+1(r, N-1) where N = trace_len - this is also a boundary case
            const n_minus_1 = trace_len - 1;
            for (0..n_vars) |bit_idx| {
                const bit_pos: u6 = @intCast(n_vars - 1 - bit_idx);
                j_bits[bit_idx] = if ((n_minus_1 >> bit_pos) & 1 == 1) F.one() else F.zero();
            }
            const eq_plus_one_at_n_minus_1 = poly_mod.EqPlusOnePolynomial(F).mle(r_outer, j_bits);
            std.debug.print("  eq+1(r, N-1={d}) = {any} (should be 0 by design)\n", .{ n_minus_1, eq_plus_one_at_n_minus_1.toBytes()[0..8] });

            // Check eq(r, N-1) for comparison
            const eq_at_n_minus_1 = poly_mod.EqPolynomial(F).mle(r_outer, j_bits);
            std.debug.print("  eq(r, N-1={d}) = {any}\n", .{ n_minus_1, eq_at_n_minus_1.toBytes()[0..8] });

            for (1..@min(5, trace_len)) |k| {
                // Compute eq(r_outer, k-1)
                for (0..n_vars) |bit_idx| {
                    const bit_pos: u6 = @intCast(n_vars - 1 - bit_idx);
                    j_bits[bit_idx] = if (((k - 1) >> bit_pos) & 1 == 1) F.one() else F.zero();
                }
                const eq_k_minus_1 = poly_mod.EqPolynomial(F).mle(r_outer, j_bits);

                // Compute eq+1(r_outer, k)
                for (0..n_vars) |bit_idx| {
                    const bit_pos: u6 = @intCast(n_vars - 1 - bit_idx);
                    j_bits[bit_idx] = if ((k >> bit_pos) & 1 == 1) F.one() else F.zero();
                }
                const eq_plus_one_k = poly_mod.EqPlusOnePolynomial(F).mle(r_outer, j_bits);

                const match = eq_k_minus_1.eql(eq_plus_one_k);
                std.debug.print("  k={d}: eq(r,k-1)={any}, eq+1(r,k)={any}, match={}\n", .{
                    k,
                    eq_k_minus_1.toBytes()[0..8],
                    eq_plus_one_k.toBytes()[0..8],
                    match,
                });
            }

            return Self{
                .P_0_outer = P_0_outer,
                .Q_0_outer = Q_0_outer,
                .P_1_outer = P_1_outer,
                .Q_1_outer = Q_1_outer,
                .P_0_prod = P_0_prod,
                .Q_0_prod = Q_0_prod,
                .P_1_prod = P_1_prod,
                .Q_1_prod = Q_1_prod,
                .gamma_powers = gamma_powers,
                .unexpanded_pc = unexpanded_pc,
                .pc = pc,
                .is_virtual = is_virtual,
                .is_first_in_sequence = is_first_in_sequence,
                .is_noop = is_noop,
                .prefix_n_vars = prefix_n_vars,
                .suffix_n_vars = suffix_n_vars,
                .current_prefix_size = prefix_size,
                .current_witness_size = trace_len,
                .sumcheck_challenges = std.ArrayList(F).init(allocator),
                .in_phase2 = false,
                .r_outer = r_outer,
                .r_product = r_product,
                .cycle_witnesses = cycle_witnesses,
                .trace_len = trace_len,
                .phase2_eq_plus_one_outer = null,
                .phase2_eq_plus_one_prod = null,
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.P_0_outer);
            self.allocator.free(self.Q_0_outer);
            self.allocator.free(self.P_1_outer);
            self.allocator.free(self.Q_1_outer);
            self.allocator.free(self.P_0_prod);
            self.allocator.free(self.Q_0_prod);
            self.allocator.free(self.P_1_prod);
            self.allocator.free(self.Q_1_prod);
            self.allocator.free(self.unexpanded_pc);
            self.allocator.free(self.pc);
            self.allocator.free(self.is_virtual);
            self.allocator.free(self.is_first_in_sequence);
            self.allocator.free(self.is_noop);
            self.sumcheck_challenges.deinit();
            if (self.phase2_eq_plus_one_outer) |p| self.allocator.free(p);
            if (self.phase2_eq_plus_one_prod) |p| self.allocator.free(p);
        }

        /// Compute round evaluations [p(0), p(1), p(2)] for degree-2 polynomial
        pub fn computeRoundEvals(self: *Self, previous_claim: F) [3]F {
            if (self.in_phase2) {
                return self.computeRoundEvalsPhase2(previous_claim);
            } else {
                return self.computeRoundEvalsPhase1(previous_claim);
            }
        }

        fn computeRoundEvalsPhase1(self: *Self, previous_claim: F) [3]F {
            // Phase1: Use P*Q formula for prefix-suffix sumcheck
            // For LowToHigh binding, we're binding the first variable X
            // H(X) = sum_j P[2j + X] * Q[2j + X]
            // So H(0) = sum_j P[2j] * Q[2j]    (X=0)
            //    H(1) = sum_j P[2j+1] * Q[2j+1] (X=1)
            const half = self.current_prefix_size / 2;
            var evals: [3]F = .{ F.zero(), F.zero(), F.zero() }; // p(0), p(1), p(2)

            // Process all 4 (P, Q) pairs
            const pairs: [4]struct { P: []F, Q: []F } = .{
                .{ .P = self.P_0_outer, .Q = self.Q_0_outer },
                .{ .P = self.P_1_outer, .Q = self.Q_1_outer },
                .{ .P = self.P_0_prod, .Q = self.Q_0_prod },
                .{ .P = self.P_1_prod, .Q = self.Q_1_prod },
            };

            for (pairs) |pair| {
                for (0..half) |i| {
                    // Get P and Q values at indices 2i and 2i+1
                    const p_at_0 = pair.P[2 * i];       // P evaluated when X=0
                    const p_at_1 = pair.P[2 * i + 1];   // P evaluated when X=1
                    const q_at_0 = pair.Q[2 * i];       // Q evaluated when X=0
                    const q_at_1 = pair.Q[2 * i + 1];   // Q evaluated when X=1

                    // Linear extrapolation for X=2: f(2) = 2*f(1) - f(0)
                    const p_at_2 = p_at_1.add(p_at_1).sub(p_at_0);
                    const q_at_2 = q_at_1.add(q_at_1).sub(q_at_0);

                    // H(X) = Σ_j P_j(X) * Q_j(X)
                    // H(0) = Σ_j P_j(0) * Q_j(0) = Σ_j P[2j] * Q[2j]
                    // H(1) = Σ_j P_j(1) * Q_j(1) = Σ_j P[2j+1] * Q[2j+1]
                    // H(2) = Σ_j P_j(2) * Q_j(2) (extrapolated)
                    evals[0] = evals[0].add(p_at_0.mul(q_at_0));
                    evals[1] = evals[1].add(p_at_1.mul(q_at_1));
                    evals[2] = evals[2].add(p_at_2.mul(q_at_2));
                }
            }

            // DEBUG: Verify sumcheck invariant p(0) + p(1) = previous_claim
            const computed_sum = evals[0].add(evals[1]);
            if (!computed_sum.eql(previous_claim)) {
                std.debug.print("[ZOLT] SHIFT INVARIANT FAIL: p(0)+p(1) = {{ {any} }}, expected = {{ {any} }}\n", .{ computed_sum.toBytes(), previous_claim.toBytes() });
            }

            return evals;
        }

        fn computeRoundEvalsPhase2(self: *Self, previous_claim: F) [3]F {
            // Phase2: Use materialized eq+1 polynomials with witness MLEs
            const eq_outer = self.phase2_eq_plus_one_outer.?;
            const eq_prod = self.phase2_eq_plus_one_prod.?;
            const half = eq_outer.len / 2;

            var evals: [2]F = .{ F.zero(), F.zero() };

            for (0..half) |j| {
                const eq_out_0 = eq_outer[2 * j];
                const eq_out_1 = eq_outer[2 * j + 1];
                const eq_prod_0 = eq_prod[2 * j];
                const eq_prod_1 = eq_prod[2 * j + 1];

                const upc_0 = self.unexpanded_pc[2 * j];
                const upc_1 = self.unexpanded_pc[2 * j + 1];
                const pc_0 = self.pc[2 * j];
                const pc_1 = self.pc[2 * j + 1];
                const virt_0 = self.is_virtual[2 * j];
                const virt_1 = self.is_virtual[2 * j + 1];
                const first_0 = self.is_first_in_sequence[2 * j];
                const first_1 = self.is_first_in_sequence[2 * j + 1];
                const noop_0 = self.is_noop[2 * j];
                const noop_1 = self.is_noop[2 * j + 1];

                // Extrapolate to X=2
                const eq_out_2 = eq_out_1.add(eq_out_1).sub(eq_out_0);
                const eq_prod_2 = eq_prod_1.add(eq_prod_1).sub(eq_prod_0);
                const upc_2 = upc_1.add(upc_1).sub(upc_0);
                const pc_2 = pc_1.add(pc_1).sub(pc_0);
                const virt_2 = virt_1.add(virt_1).sub(virt_0);
                const first_2 = first_1.add(first_1).sub(first_0);
                const noop_2 = noop_1.add(noop_1).sub(noop_0);

                // Compute f at X=0
                const val_0 = upc_0.add(self.gamma_powers[1].mul(pc_0))
                    .add(self.gamma_powers[2].mul(virt_0))
                    .add(self.gamma_powers[3].mul(first_0));
                const term1_0 = eq_out_0.mul(val_0);
                const term2_0 = self.gamma_powers[4].mul(F.one().sub(noop_0)).mul(eq_prod_0);
                const f_0 = term1_0.add(term2_0);

                // Compute f at X=2
                const val_2 = upc_2.add(self.gamma_powers[1].mul(pc_2))
                    .add(self.gamma_powers[2].mul(virt_2))
                    .add(self.gamma_powers[3].mul(first_2));
                const term1_2 = eq_out_2.mul(val_2);
                const term2_2 = self.gamma_powers[4].mul(F.one().sub(noop_2)).mul(eq_prod_2);
                const f_2 = term1_2.add(term2_2);

                evals[0] = evals[0].add(f_0);
                evals[1] = evals[1].add(f_2);
            }

            const p_1 = previous_claim.sub(evals[0]);
            return [3]F{ evals[0], p_1, evals[1] };
        }

        /// Bind the prover at challenge r_j
        pub fn bind(self: *Self, r_j: F) void {
            if (self.in_phase2) {
                self.bindPhase2(r_j);
            } else {
                // Check if we should transition to Phase2
                if (self.shouldTransitionToPhase2()) {
                    // transitionToPhase2 handles appending the challenge itself
                    self.transitionToPhase2(r_j);
                } else {
                    // Append challenge for Phase 1 binding
                    self.sumcheck_challenges.append(r_j) catch unreachable;
                    self.bindPhase1(r_j);
                }
            }
        }

        fn shouldTransitionToPhase2(self: *Self) bool {
            // Transition when prefix size is 2 (log2 == 1)
            return std.math.log2_int(usize, self.current_prefix_size) == 1;
        }

        fn bindPhase1(self: *Self, r_j: F) void {
            // Bind P and Q buffers: new[i] = old[2i] + r * (old[2i+1] - old[2i])
            const new_prefix_size = self.current_prefix_size / 2;

            for (0..new_prefix_size) |i| {
                // P_0_outer
                self.P_0_outer[i] = self.P_0_outer[2 * i].add(r_j.mul(self.P_0_outer[2 * i + 1].sub(self.P_0_outer[2 * i])));
                self.Q_0_outer[i] = self.Q_0_outer[2 * i].add(r_j.mul(self.Q_0_outer[2 * i + 1].sub(self.Q_0_outer[2 * i])));

                // P_1_outer
                self.P_1_outer[i] = self.P_1_outer[2 * i].add(r_j.mul(self.P_1_outer[2 * i + 1].sub(self.P_1_outer[2 * i])));
                self.Q_1_outer[i] = self.Q_1_outer[2 * i].add(r_j.mul(self.Q_1_outer[2 * i + 1].sub(self.Q_1_outer[2 * i])));

                // P_0_prod
                self.P_0_prod[i] = self.P_0_prod[2 * i].add(r_j.mul(self.P_0_prod[2 * i + 1].sub(self.P_0_prod[2 * i])));
                self.Q_0_prod[i] = self.Q_0_prod[2 * i].add(r_j.mul(self.Q_0_prod[2 * i + 1].sub(self.Q_0_prod[2 * i])));

                // P_1_prod
                self.P_1_prod[i] = self.P_1_prod[2 * i].add(r_j.mul(self.P_1_prod[2 * i + 1].sub(self.P_1_prod[2 * i])));
                self.Q_1_prod[i] = self.Q_1_prod[2 * i].add(r_j.mul(self.Q_1_prod[2 * i + 1].sub(self.Q_1_prod[2 * i])));
            }

            self.current_prefix_size = new_prefix_size;
            // Note: Witness MLEs are NOT bound in Phase 1. They are reconstructed
            // from scratch during transitionToPhase2 using Eq(r_prefix, i) weighting.
        }

        fn transitionToPhase2(self: *Self, r_j: F) void {
            // The transition happens AFTER binding with the final Phase 1 challenge r_j
            // First, bind the P/Q buffers one last time (they become size 1)
            const new_prefix_size = self.current_prefix_size / 2;
            for (0..new_prefix_size) |i| {
                self.P_0_outer[i] = self.P_0_outer[2 * i].add(r_j.mul(self.P_0_outer[2 * i + 1].sub(self.P_0_outer[2 * i])));
                self.Q_0_outer[i] = self.Q_0_outer[2 * i].add(r_j.mul(self.Q_0_outer[2 * i + 1].sub(self.Q_0_outer[2 * i])));
                self.P_1_outer[i] = self.P_1_outer[2 * i].add(r_j.mul(self.P_1_outer[2 * i + 1].sub(self.P_1_outer[2 * i])));
                self.Q_1_outer[i] = self.Q_1_outer[2 * i].add(r_j.mul(self.Q_1_outer[2 * i + 1].sub(self.Q_1_outer[2 * i])));
                self.P_0_prod[i] = self.P_0_prod[2 * i].add(r_j.mul(self.P_0_prod[2 * i + 1].sub(self.P_0_prod[2 * i])));
                self.Q_0_prod[i] = self.Q_0_prod[2 * i].add(r_j.mul(self.Q_0_prod[2 * i + 1].sub(self.Q_0_prod[2 * i])));
                self.P_1_prod[i] = self.P_1_prod[2 * i].add(r_j.mul(self.P_1_prod[2 * i + 1].sub(self.P_1_prod[2 * i])));
                self.Q_1_prod[i] = self.Q_1_prod[2 * i].add(r_j.mul(self.Q_1_prod[2 * i + 1].sub(self.Q_1_prod[2 * i])));
            }
            self.current_prefix_size = new_prefix_size;

            // Store final challenge
            self.sumcheck_challenges.append(r_j) catch unreachable;
            self.in_phase2 = true;

            // Collect all Phase 1 challenges as r_prefix
            // CRITICAL: Jolt converts from LITTLE_ENDIAN (sumcheck order) to BIG_ENDIAN (MLE indexing)
            // by reversing the challenges array. We must do the same.
            // sumcheck_challenges[0] = first round = binds LSB variable
            // After reversal: r_prefix_be[0] = last challenge = MSB variable
            const r_prefix_le = self.sumcheck_challenges.items;
            const r_prefix_be = self.allocator.alloc(F, r_prefix_le.len) catch unreachable;
            defer self.allocator.free(r_prefix_be);
            for (0..r_prefix_le.len) |i| {
                r_prefix_be[i] = r_prefix_le[r_prefix_le.len - 1 - i];
            }
            const n_remaining_rounds = self.suffix_n_vars;
            const suffix_size: usize = @as(usize, 1) << @intCast(n_remaining_rounds);

            std.debug.print("\n[ZOLT] SHIFT_PHASE2_START: n_remaining_rounds={d}, suffix_size={d}\n", .{ n_remaining_rounds, suffix_size });
            std.debug.print("[ZOLT] SHIFT_PHASE2_START: r_prefix_be.len={d}\n", .{r_prefix_be.len});

            // =====================================================================
            // Step 1: Regenerate prefix-suffix decomposition from original r_outer/r_product
            // and evaluate prefix at r_prefix to get scalar values
            // =====================================================================

            // For r_outer: split into hi and lo parts (Jolt convention)
            // r_hi (first half) -> used for SUFFIX
            // r_lo (second half) -> used for PREFIX
            // split_point = suffix_n_vars (original n_vars / 2)
            const r_outer_hi = self.r_outer[0..self.suffix_n_vars]; // For SUFFIX
            const r_outer_lo = self.r_outer[self.suffix_n_vars..]; // For PREFIX

            // Regenerate prefix polynomials for r_outer
            // PREFIX uses r_lo (Jolt convention)
            const prefix_size_outer: usize = @as(usize, 1) << @intCast(r_outer_lo.len);
            const prefix_0_outer = self.allocator.alloc(F, prefix_size_outer) catch unreachable;
            defer self.allocator.free(prefix_0_outer);
            computeEqPlusOneEvals(self.allocator, r_outer_lo, prefix_0_outer) catch unreachable;

            const prefix_1_outer = self.allocator.alloc(F, prefix_size_outer) catch unreachable;
            defer self.allocator.free(prefix_1_outer);
            @memset(prefix_1_outer, F.zero());
            var is_max_outer = F.one();
            for (r_outer_lo) |r_i| {
                is_max_outer = is_max_outer.mul(r_i);
            }
            prefix_1_outer[0] = is_max_outer;

            // Evaluate prefix polynomials at r_prefix
            // NOTE: evaluateMle expects LITTLE_ENDIAN order (point[0] binds LSB)
            // r_prefix_le (original from sumcheck_challenges) is already in little-endian order
            // r_prefix_be is the big-endian version we created above by reversing
            // For MLE evaluation, we need little-endian, so use r_prefix_le directly
            const prefix_0_eval_outer = evaluateMle(prefix_0_outer, r_prefix_le);
            const prefix_1_eval_outer = evaluateMle(prefix_1_outer, r_prefix_le);

            // DEBUG: Print prefix evaluation details
            std.debug.print("[ZOLT] SHIFT_PREFIX: r_prefix_be[0] = {any}\n", .{r_prefix_be[0].toBytes()[0..8]});
            std.debug.print("[ZOLT] SHIFT_PREFIX: r_prefix_be[last] = {any}\n", .{r_prefix_be[r_prefix_be.len - 1].toBytes()[0..8]});
            std.debug.print("[ZOLT] SHIFT_PREFIX: prefix_0_eval_outer = {any}\n", .{prefix_0_eval_outer.toBytes()[0..8]});
            std.debug.print("[ZOLT] SHIFT_PREFIX: prefix_1_eval_outer = {any}\n", .{prefix_1_eval_outer.toBytes()[0..8]});

            // DEBUG: Print r_outer_hi and r_outer_lo (the fixed points from Stage 1)
            // Using BE for comparison with STAGE1_CHALLENGES output
            std.debug.print("[ZOLT] SHIFT_PREFIX: r_outer_hi[0] (BE) = {any}\n", .{r_outer_hi[0].toBytesBE()[0..8]});
            std.debug.print("[ZOLT] SHIFT_PREFIX: r_outer_hi[last] (BE) = {any}\n", .{r_outer_hi[r_outer_hi.len - 1].toBytesBE()[0..8]});
            std.debug.print("[ZOLT] SHIFT_PREFIX: r_outer_lo[0] (BE) = {any}\n", .{r_outer_lo[0].toBytesBE()[0..8]});
            std.debug.print("[ZOLT] SHIFT_PREFIX: r_outer_lo[last] (BE) = {any}\n", .{r_outer_lo[r_outer_lo.len - 1].toBytesBE()[0..8]});

            // Regenerate suffix polynomials for r_outer
            // SUFFIX uses r_hi (Jolt convention)
            const suffix_0_outer = self.allocator.alloc(F, suffix_size) catch unreachable;
            defer self.allocator.free(suffix_0_outer);
            const suffix_1_outer = self.allocator.alloc(F, suffix_size) catch unreachable;
            defer self.allocator.free(suffix_1_outer);
            computeEqAndEqPlusOneEvals(self.allocator, r_outer_hi, suffix_0_outer, suffix_1_outer) catch unreachable;

            // Same for r_product
            const r_prod_hi = self.r_product[0..self.suffix_n_vars]; // For SUFFIX
            const r_prod_lo = self.r_product[self.suffix_n_vars..]; // For PREFIX

            // PREFIX uses r_lo (Jolt convention)
            const prefix_size_prod: usize = @as(usize, 1) << @intCast(r_prod_lo.len);
            const prefix_0_prod = self.allocator.alloc(F, prefix_size_prod) catch unreachable;
            defer self.allocator.free(prefix_0_prod);
            computeEqPlusOneEvals(self.allocator, r_prod_lo, prefix_0_prod) catch unreachable;

            const prefix_1_prod = self.allocator.alloc(F, prefix_size_prod) catch unreachable;
            defer self.allocator.free(prefix_1_prod);
            @memset(prefix_1_prod, F.zero());
            var is_max_prod = F.one();
            for (r_prod_lo) |r_i| {
                is_max_prod = is_max_prod.mul(r_i);
            }
            prefix_1_prod[0] = is_max_prod;

            // Use r_prefix_le for MLE evaluation (little-endian for evaluateMle)
            const prefix_0_eval_prod = evaluateMle(prefix_0_prod, r_prefix_le);
            const prefix_1_eval_prod = evaluateMle(prefix_1_prod, r_prefix_le);

            // SUFFIX uses r_hi (Jolt convention)
            const suffix_0_prod = self.allocator.alloc(F, suffix_size) catch unreachable;
            defer self.allocator.free(suffix_0_prod);
            const suffix_1_prod = self.allocator.alloc(F, suffix_size) catch unreachable;
            defer self.allocator.free(suffix_1_prod);
            computeEqAndEqPlusOneEvals(self.allocator, r_prod_hi, suffix_0_prod, suffix_1_prod) catch unreachable;

            // =====================================================================
            // Step 2: Construct eq+1(r_outer, (r_prefix, j)) for all j in suffix domain
            // eq+1(r, (r_prefix, j)) = prefix_0_eval * suffix_0[j] + prefix_1_eval * suffix_1[j]
            // =====================================================================

            self.phase2_eq_plus_one_outer = self.allocator.alloc(F, suffix_size) catch unreachable;
            self.phase2_eq_plus_one_prod = self.allocator.alloc(F, suffix_size) catch unreachable;

            for (0..suffix_size) |j| {
                self.phase2_eq_plus_one_outer.?[j] = prefix_0_eval_outer.mul(suffix_0_outer[j])
                    .add(prefix_1_eval_outer.mul(suffix_1_outer[j]));
                self.phase2_eq_plus_one_prod.?[j] = prefix_0_eval_prod.mul(suffix_0_prod[j])
                    .add(prefix_1_eval_prod.mul(suffix_1_prod[j]));
            }

            // =====================================================================
            // Step 3: Construct witness MLEs by summing over prefix domain weighted by Eq(r_prefix, i)
            // poly[j] = Σ_i Eq(r_prefix, i) * witness[i * suffix_size + j]
            // =====================================================================

            // Compute Eq(r_prefix, i) for all i in prefix domain (using BIG_ENDIAN version)
            const prefix_domain_size: usize = @as(usize, 1) << @intCast(r_prefix_be.len);
            const eq_evals = self.allocator.alloc(F, prefix_domain_size) catch unreachable;
            defer self.allocator.free(eq_evals);
            computeEqEvals(self.allocator, r_prefix_be, eq_evals) catch unreachable;

            // Reallocate witness MLEs to suffix_size
            self.allocator.free(self.unexpanded_pc);
            self.allocator.free(self.pc);
            self.allocator.free(self.is_virtual);
            self.allocator.free(self.is_first_in_sequence);
            self.allocator.free(self.is_noop);

            self.unexpanded_pc = self.allocator.alloc(F, suffix_size) catch unreachable;
            self.pc = self.allocator.alloc(F, suffix_size) catch unreachable;
            self.is_virtual = self.allocator.alloc(F, suffix_size) catch unreachable;
            self.is_first_in_sequence = self.allocator.alloc(F, suffix_size) catch unreachable;
            self.is_noop = self.allocator.alloc(F, suffix_size) catch unreachable;

            @memset(self.unexpanded_pc, F.zero());
            @memset(self.pc, F.zero());
            @memset(self.is_virtual, F.zero());
            @memset(self.is_first_in_sequence, F.zero());
            @memset(self.is_noop, F.zero());

            // Sum over prefix domain
            for (0..suffix_size) |j| {
                var upc_acc = F.zero();
                var pc_acc = F.zero();
                var virt_acc = F.zero();
                var first_acc = F.zero();
                var noop_acc = F.zero();

                for (0..prefix_domain_size) |i| {
                    // Trace index = i * suffix_size + j (interleaved layout)
                    // But Jolt uses trace.par_chunks(eq_evals.len()), meaning:
                    // For suffix index j, the trace indices are j*prefix_domain_size + i
                    const trace_idx = j * prefix_domain_size + i;
                    if (trace_idx >= self.trace_len) continue;

                    const witness = &self.cycle_witnesses[trace_idx].values;
                    const eq_eval = eq_evals[i];

                    upc_acc = upc_acc.add(eq_eval.mul(witness[R1CSInputIndex.UnexpandedPC.toIndex()]));
                    pc_acc = pc_acc.add(eq_eval.mul(witness[R1CSInputIndex.PC.toIndex()]));
                    virt_acc = virt_acc.add(eq_eval.mul(witness[R1CSInputIndex.FlagVirtualInstruction.toIndex()]));
                    first_acc = first_acc.add(eq_eval.mul(witness[R1CSInputIndex.FlagIsFirstInSequence.toIndex()]));
                    noop_acc = noop_acc.add(eq_eval.mul(witness[R1CSInputIndex.FlagIsNoop.toIndex()]));
                }

                self.unexpanded_pc[j] = upc_acc;
                self.pc[j] = pc_acc;
                self.is_virtual[j] = virt_acc;
                self.is_first_in_sequence[j] = first_acc;
                self.is_noop[j] = noop_acc;
            }

            self.current_witness_size = suffix_size;

            std.debug.print("[ZOLT] SHIFT_PHASE2_START: eq+1_outer[0] = {{ {any} }}\n", .{self.phase2_eq_plus_one_outer.?[0].toBytes()[0..8]});
            std.debug.print("[ZOLT] SHIFT_PHASE2_START: unexpanded_pc[0] = {{ {any} }}\n", .{self.unexpanded_pc[0].toBytes()[0..8]});

            // DEBUG: Verify eq+1_outer initialization by direct evaluation
            {
                // eq+1_outer[0] should equal eq+1(r_outer, (r_prefix_be, [0,0,...,0]))
                // where [0,0,...,0] is all zeros (suffix_n_vars zeros)
                // This is: prefix_0_eval * suffix_0[0] + prefix_1_eval * suffix_1[0]

                // First, verify suffix_0[0] = eq(r_outer_hi, [0,...,0])
                // eq([r0,r1,...], [0,0,...]) should be prod(1-ri)
                var expected_suffix_0_at_0 = F.one();
                for (r_outer_hi) |ri| {
                    expected_suffix_0_at_0 = expected_suffix_0_at_0.mul(F.one().sub(ri));
                }
                std.debug.print("[ZOLT] SHIFT_DEBUG: expected suffix_0[0] = {any}\n", .{expected_suffix_0_at_0.toBytes()[0..8]});
                std.debug.print("[ZOLT] SHIFT_DEBUG: actual suffix_0_outer[0] = {any}\n", .{suffix_0_outer[0].toBytes()[0..8]});

                // eq+1([r0,r1,...], [0,0,...]) should be 0 (because y=0 is not a successor of any x >= 0)
                // Actually no, eq+1(x, y) = 1 iff y = x+1
                // For y=[0,...,0] (binary 0), we need x = -1 which doesn't exist in unsigned
                // So eq+1(anything, [0,...,0]) = 0
                std.debug.print("[ZOLT] SHIFT_DEBUG: suffix_1_outer[0] should be ~0: {any}\n", .{suffix_1_outer[0].toBytes()[0..8]});

                // CRITICAL TEST: Evaluate eq+1(r_outer, (r_prefix_be, [0,0,0,0])) directly
                // and compare with phase2_eq_plus_one_outer[0]
                //
                // Construct full y = (r_prefix_be, zeros_4) in big-endian
                // Wait no, the formula is eq+1(r_outer, (y_hi, y_lo)) where y_lo is bound first.
                // At index j=0 (all zeros for suffix), y = (zeros_4, r_prefix_be)
                // So full_y = (zeros_suffix, r_prefix_be) where zeros_suffix has suffix_n_vars zeros
                const full_y = self.allocator.alloc(F, self.r_outer.len) catch unreachable;
                defer self.allocator.free(full_y);

                // Big-endian: first half = suffix = zeros (for j=0), second half = prefix = r_prefix_be
                for (0..self.suffix_n_vars) |i| {
                    full_y[i] = F.zero();
                }
                for (0..r_prefix_be.len) |i| {
                    full_y[self.suffix_n_vars + i] = r_prefix_be[i];
                }

                // Direct evaluation
                const direct_eq_plus_one = poly_mod.EqPlusOnePolynomial(F).mle(self.r_outer, full_y);
                std.debug.print("[ZOLT] SHIFT_CRITICAL: direct eq+1(r_outer, (zeros, r_prefix)) = {any}\n", .{direct_eq_plus_one.toBytes()});
                std.debug.print("[ZOLT] SHIFT_CRITICAL: phase2_eq+1_outer[0] = {any}\n", .{self.phase2_eq_plus_one_outer.?[0].toBytes()});
                std.debug.print("[ZOLT] SHIFT_CRITICAL: match = {}\n", .{direct_eq_plus_one.eql(self.phase2_eq_plus_one_outer.?[0])});

                // Debug: check formula components
                // phase2_eq+1_outer[0] = prefix_0_eval * suffix_0[0] + prefix_1_eval * suffix_1[0]
                const expected_from_formula = prefix_0_eval_outer.mul(suffix_0_outer[0])
                    .add(prefix_1_eval_outer.mul(suffix_1_outer[0]));
                std.debug.print("[ZOLT] SHIFT_CRITICAL: from_formula = {any}\n", .{expected_from_formula.toBytes()});
                std.debug.print("[ZOLT] SHIFT_CRITICAL: formula_match = {}\n", .{expected_from_formula.eql(self.phase2_eq_plus_one_outer.?[0])});

                // Debug: prefix_0 and suffix_0 individually
                // Direct eq+1(r_lo, y_lo) where y_lo = r_prefix_be
                const direct_prefix_eval = poly_mod.EqPlusOnePolynomial(F).mle(r_outer_lo, r_prefix_be);
                std.debug.print("[ZOLT] SHIFT_CRITICAL: direct eq+1(r_lo, y_lo) = {any}\n", .{direct_prefix_eval.toBytes()});
                std.debug.print("[ZOLT] SHIFT_CRITICAL: prefix_0_eval_outer = {any}\n", .{prefix_0_eval_outer.toBytes()});
                std.debug.print("[ZOLT] SHIFT_CRITICAL: prefix_match = {}\n", .{direct_prefix_eval.eql(prefix_0_eval_outer)});

                // Direct eq(r_hi, y_hi) where y_hi = zeros
                const zeros_hi = self.allocator.alloc(F, self.suffix_n_vars) catch unreachable;
                defer self.allocator.free(zeros_hi);
                @memset(zeros_hi, F.zero());
                const direct_suffix_eval = poly_mod.EqPolynomial(F).mle(r_outer_hi, zeros_hi);
                std.debug.print("[ZOLT] SHIFT_CRITICAL: direct eq(r_hi, zeros) = {any}\n", .{direct_suffix_eval.toBytes()});
                std.debug.print("[ZOLT] SHIFT_CRITICAL: suffix_0[0] = {any}\n", .{suffix_0_outer[0].toBytes()});
                std.debug.print("[ZOLT] SHIFT_CRITICAL: suffix_match = {}\n", .{direct_suffix_eval.eql(suffix_0_outer[0])});
            }
        }

        fn bindPhase2(self: *Self, r_j: F) void {
            // Bind witness MLEs and eq+1 polynomials using LowToHigh order
            // Formula: new[i] = old[2*i] + r * (old[2*i+1] - old[2*i])
            const new_size = self.current_witness_size / 2;

            // Debug: print current state before binding
            if (self.current_witness_size <= 16) {
                std.debug.print("[ZOLT] SHIFT_BIND: size={}, r_j={any}\n", .{ self.current_witness_size, r_j.toBytes()[0..8] });
                if (self.phase2_eq_plus_one_outer) |eq| {
                    std.debug.print("[ZOLT] SHIFT_BIND: eq+1_outer[0]={any}, eq+1_outer[1]={any}\n", .{ eq[0].toBytes()[0..8], eq[1].toBytes()[0..8] });
                }
            }

            for (0..new_size) |i| {
                self.unexpanded_pc[i] = self.unexpanded_pc[2 * i].add(r_j.mul(self.unexpanded_pc[2 * i + 1].sub(self.unexpanded_pc[2 * i])));
                self.pc[i] = self.pc[2 * i].add(r_j.mul(self.pc[2 * i + 1].sub(self.pc[2 * i])));
                self.is_virtual[i] = self.is_virtual[2 * i].add(r_j.mul(self.is_virtual[2 * i + 1].sub(self.is_virtual[2 * i])));
                self.is_first_in_sequence[i] = self.is_first_in_sequence[2 * i].add(r_j.mul(self.is_first_in_sequence[2 * i + 1].sub(self.is_first_in_sequence[2 * i])));

                // Debug after last binding
                if (new_size == 1 and self.phase2_eq_plus_one_outer != null) {
                    const eq = self.phase2_eq_plus_one_outer.?;
                    const new_val = eq[0].add(r_j.mul(eq[1].sub(eq[0])));
                    std.debug.print("[ZOLT] SHIFT_BIND_FINAL: new eq+1_outer[0]={any}\n", .{new_val.toBytes()});
                }
                self.is_noop[i] = self.is_noop[2 * i].add(r_j.mul(self.is_noop[2 * i + 1].sub(self.is_noop[2 * i])));

                if (self.phase2_eq_plus_one_outer) |eq| {
                    eq[i] = eq[2 * i].add(r_j.mul(eq[2 * i + 1].sub(eq[2 * i])));
                }
                if (self.phase2_eq_plus_one_prod) |eq| {
                    eq[i] = eq[2 * i].add(r_j.mul(eq[2 * i + 1].sub(eq[2 * i])));
                }
            }
            self.current_witness_size = new_size;
        }

        // Helper: Evaluate MLE at a point
        fn evaluateMle(coeffs: []const F, point: []const F) F {
            if (coeffs.len == 1) return coeffs[0];
            if (point.len == 0) return coeffs[0];

            const temp = std.heap.page_allocator.alloc(F, coeffs.len) catch unreachable;
            defer std.heap.page_allocator.free(temp);
            @memcpy(temp, coeffs);

            var current_len = coeffs.len;
            for (point) |r_i| {
                const half = current_len / 2;
                for (0..half) |i| {
                    temp[i] = temp[2 * i].add(r_i.mul(temp[2 * i + 1].sub(temp[2 * i])));
                }
                current_len = half;
            }
            return temp[0];
        }

        // Helper: Compute Eq(r, j) for all j
        fn computeEqEvals(allocator: Allocator, r: []const F, out: []F) !void {
            const n = r.len;
            const size = out.len;
            std.debug.assert(size == @as(usize, 1) << @intCast(n));

            const j_bits = try allocator.alloc(F, n);
            defer allocator.free(j_bits);

            for (0..size) |j| {
                // Convert j to binary (BIG_ENDIAN: bit 0 is MSB)
                for (0..n) |k| {
                    const bit_pos: u6 = @intCast(n - 1 - k);
                    j_bits[k] = if ((j >> bit_pos) & 1 == 1) F.one() else F.zero();
                }
                out[j] = poly_mod.EqPolynomial(F).mle(r, j_bits);
            }
        }

        /// Get final claims after all rounds
        /// After all rounds, current_witness_size should be 1
        pub fn finalClaims(self: *const Self) struct {
            unexpanded_pc: F,
            pc: F,
            is_virtual: F,
            is_first_in_sequence: F,
            is_noop: F,
        } {
            std.debug.assert(self.current_witness_size == 1);
            return .{
                .unexpanded_pc = self.unexpanded_pc[0],
                .pc = self.pc[0],
                .is_virtual = self.is_virtual[0],
                .is_first_in_sequence = self.is_first_in_sequence[0],
                .is_noop = self.is_noop[0],
            };
        }

        // Helper: Compute eq+1(r, j) for all j
        fn computeEqPlusOneEvals(allocator: Allocator, r: []const F, out: []F) !void {
            const n = r.len;
            const size = out.len;
            std.debug.assert(size == @as(usize, 1) << @intCast(n));

            const j_bits = try allocator.alloc(F, n);
            defer allocator.free(j_bits);

            for (0..size) |j| {
                // Convert j to binary (BIG_ENDIAN: bit 0 is MSB)
                for (0..n) |k| {
                    const bit_pos: u6 = @intCast(n - 1 - k);
                    j_bits[k] = if ((j >> bit_pos) & 1 == 1) F.one() else F.zero();
                }
                out[j] = poly_mod.EqPlusOnePolynomial(F).mle(r, j_bits);
            }
        }

        // Helper: Compute both eq and eq+1 evaluations
        fn computeEqAndEqPlusOneEvals(allocator: Allocator, r: []const F, eq_out: []F, eq_plus_one_out: []F) !void {
            const n = r.len;
            const size = eq_out.len;
            std.debug.assert(size == @as(usize, 1) << @intCast(n));

            const j_bits = try allocator.alloc(F, n);
            defer allocator.free(j_bits);

            for (0..size) |j| {
                for (0..n) |k| {
                    const bit_pos: u6 = @intCast(n - 1 - k);
                    j_bits[k] = if ((j >> bit_pos) & 1 == 1) F.one() else F.zero();
                }
                eq_out[j] = poly_mod.EqPolynomial(F).mle(r, j_bits);
                eq_plus_one_out[j] = poly_mod.EqPlusOnePolynomial(F).mle(r, j_bits);
            }
        }
    };
}

// =============================================================================
// InstructionInput Prover (No Prefix-Suffix, uses direct computation)
// =============================================================================

fn InstructionInputProver(comptime F: type) type {
    return struct {
        const Self = @This();

        // Witness MLEs
        left_is_rs1: []F,
        rs1_value: []F,
        left_is_pc: []F,
        unexpanded_pc: []F,
        right_is_rs2: []F,
        rs2_value: []F,
        right_is_imm: []F,
        imm: []F,

        // Eq polynomial evaluations
        eq_outer: []F,
        eq_product: []F,

        gamma: F,
        gamma_sqr: F,

        current_size: usize,
        allocator: Allocator,

        pub fn init(
            allocator: Allocator,
            cycle_witnesses: []const r1cs.R1CSCycleInputs(F),
            trace_len: usize,
            r_outer: []const F,
            r_product: []const F,
            gamma: F,
            gamma_sqr: F,
        ) !Self {
            // Allocate MLEs
            const left_is_rs1 = try allocator.alloc(F, trace_len);
            const rs1_value = try allocator.alloc(F, trace_len);
            const left_is_pc = try allocator.alloc(F, trace_len);
            const unexpanded_pc = try allocator.alloc(F, trace_len);
            const right_is_rs2 = try allocator.alloc(F, trace_len);
            const rs2_value = try allocator.alloc(F, trace_len);
            const right_is_imm = try allocator.alloc(F, trace_len);
            const imm = try allocator.alloc(F, trace_len);

            // Fill from witnesses
            for (0..trace_len) |i| {
                if (i < cycle_witnesses.len) {
                    const values = &cycle_witnesses[i].values;
                    left_is_rs1[i] = values[R1CSInputIndex.FlagLeftOperandIsRs1.toIndex()];
                    left_is_pc[i] = values[R1CSInputIndex.FlagLeftOperandIsPC.toIndex()];
                    right_is_rs2[i] = values[R1CSInputIndex.FlagRightOperandIsRs2.toIndex()];
                    right_is_imm[i] = values[R1CSInputIndex.FlagRightOperandIsImm.toIndex()];
                    rs1_value[i] = values[R1CSInputIndex.Rs1Value.toIndex()];
                    unexpanded_pc[i] = values[R1CSInputIndex.UnexpandedPC.toIndex()];
                    rs2_value[i] = values[R1CSInputIndex.Rs2Value.toIndex()];
                    imm[i] = values[R1CSInputIndex.Imm.toIndex()];
                } else {
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

            // Compute eq evaluations
            var eq_outer_poly = try poly_mod.EqPolynomial(F).init(allocator, r_outer);
            defer eq_outer_poly.deinit();
            const eq_outer = try eq_outer_poly.evals(allocator);

            var eq_product_poly = try poly_mod.EqPolynomial(F).init(allocator, r_product);
            defer eq_product_poly.deinit();
            const eq_product = try eq_product_poly.evals(allocator);

            return Self{
                .left_is_rs1 = left_is_rs1,
                .rs1_value = rs1_value,
                .left_is_pc = left_is_pc,
                .unexpanded_pc = unexpanded_pc,
                .right_is_rs2 = right_is_rs2,
                .rs2_value = rs2_value,
                .right_is_imm = right_is_imm,
                .imm = imm,
                .eq_outer = eq_outer,
                .eq_product = eq_product,
                .gamma = gamma,
                .gamma_sqr = gamma_sqr,
                .current_size = trace_len,
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.left_is_rs1);
            self.allocator.free(self.rs1_value);
            self.allocator.free(self.left_is_pc);
            self.allocator.free(self.unexpanded_pc);
            self.allocator.free(self.right_is_rs2);
            self.allocator.free(self.rs2_value);
            self.allocator.free(self.right_is_imm);
            self.allocator.free(self.imm);
            self.allocator.free(self.eq_outer);
            self.allocator.free(self.eq_product);
        }

        /// Compute round evaluations [p(0), p(1), p(2), p(3)] for degree-3 polynomial
        pub fn computeRoundEvals(self: *Self, previous_claim: F) [4]F {
            const half = self.current_size / 2;
            var evals: [3]F = .{ F.zero(), F.zero(), F.zero() }; // p(0), p(2), p(3)

            for (0..half) |j| {
                // Get values at bit=0 and bit=1
                const left_is_rs1_0 = self.left_is_rs1[2 * j];
                const left_is_rs1_1 = self.left_is_rs1[2 * j + 1];
                const rs1_0 = self.rs1_value[2 * j];
                const rs1_1 = self.rs1_value[2 * j + 1];
                const left_is_pc_0 = self.left_is_pc[2 * j];
                const left_is_pc_1 = self.left_is_pc[2 * j + 1];
                const pc_0 = self.unexpanded_pc[2 * j];
                const pc_1 = self.unexpanded_pc[2 * j + 1];
                const right_is_rs2_0 = self.right_is_rs2[2 * j];
                const right_is_rs2_1 = self.right_is_rs2[2 * j + 1];
                const rs2_0 = self.rs2_value[2 * j];
                const rs2_1 = self.rs2_value[2 * j + 1];
                const right_is_imm_0 = self.right_is_imm[2 * j];
                const right_is_imm_1 = self.right_is_imm[2 * j + 1];
                const imm_0 = self.imm[2 * j];
                const imm_1 = self.imm[2 * j + 1];
                const eq_out_0 = self.eq_outer[2 * j];
                const eq_out_1 = self.eq_outer[2 * j + 1];
                const eq_prod_0 = self.eq_product[2 * j];
                const eq_prod_1 = self.eq_product[2 * j + 1];

                // Extrapolate to X=2 and X=3
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

                // Compute at X=0
                const left_0 = left_is_rs1_0.mul(rs1_0).add(left_is_pc_0.mul(pc_0));
                const right_0 = right_is_rs2_0.mul(rs2_0).add(right_is_imm_0.mul(imm_0));
                const eq_weight_0 = eq_out_0.add(self.gamma_sqr.mul(eq_prod_0));
                const f_0 = eq_weight_0.mul(right_0.add(self.gamma.mul(left_0)));

                // Compute at X=2
                const left_2 = left_is_rs1_2.mul(rs1_2).add(left_is_pc_2.mul(pc_2));
                const right_2 = right_is_rs2_2.mul(rs2_2).add(right_is_imm_2.mul(imm_2));
                const eq_weight_2 = eq_out_2.add(self.gamma_sqr.mul(eq_prod_2));
                const f_2 = eq_weight_2.mul(right_2.add(self.gamma.mul(left_2)));

                // Compute at X=3
                const left_3 = left_is_rs1_3.mul(rs1_3).add(left_is_pc_3.mul(pc_3));
                const right_3 = right_is_rs2_3.mul(rs2_3).add(right_is_imm_3.mul(imm_3));
                const eq_weight_3 = eq_out_3.add(self.gamma_sqr.mul(eq_prod_3));
                const f_3 = eq_weight_3.mul(right_3.add(self.gamma.mul(left_3)));

                evals[0] = evals[0].add(f_0);
                evals[1] = evals[1].add(f_2);
                evals[2] = evals[2].add(f_3);
            }

            // Derive p(1) from previous_claim
            const p_1 = previous_claim.sub(evals[0]);

            return [4]F{ evals[0], p_1, evals[1], evals[2] };
        }

        pub fn bind(self: *Self, r_j: F) void {
            // Bind in LowToHigh order to match computeRoundEvals indexing (2*i, 2*i+1)
            // new[i] = old[2*i] + r * (old[2*i+1] - old[2*i])
            const new_size = self.current_size / 2;

            for (0..new_size) |i| {
                self.left_is_rs1[i] = self.left_is_rs1[2 * i].add(r_j.mul(self.left_is_rs1[2 * i + 1].sub(self.left_is_rs1[2 * i])));
                self.rs1_value[i] = self.rs1_value[2 * i].add(r_j.mul(self.rs1_value[2 * i + 1].sub(self.rs1_value[2 * i])));
                self.left_is_pc[i] = self.left_is_pc[2 * i].add(r_j.mul(self.left_is_pc[2 * i + 1].sub(self.left_is_pc[2 * i])));
                self.unexpanded_pc[i] = self.unexpanded_pc[2 * i].add(r_j.mul(self.unexpanded_pc[2 * i + 1].sub(self.unexpanded_pc[2 * i])));
                self.right_is_rs2[i] = self.right_is_rs2[2 * i].add(r_j.mul(self.right_is_rs2[2 * i + 1].sub(self.right_is_rs2[2 * i])));
                self.rs2_value[i] = self.rs2_value[2 * i].add(r_j.mul(self.rs2_value[2 * i + 1].sub(self.rs2_value[2 * i])));
                self.right_is_imm[i] = self.right_is_imm[2 * i].add(r_j.mul(self.right_is_imm[2 * i + 1].sub(self.right_is_imm[2 * i])));
                self.imm[i] = self.imm[2 * i].add(r_j.mul(self.imm[2 * i + 1].sub(self.imm[2 * i])));
                self.eq_outer[i] = self.eq_outer[2 * i].add(r_j.mul(self.eq_outer[2 * i + 1].sub(self.eq_outer[2 * i])));
                self.eq_product[i] = self.eq_product[2 * i].add(r_j.mul(self.eq_product[2 * i + 1].sub(self.eq_product[2 * i])));
            }

            self.current_size = new_size;
        }

        pub fn finalClaims(self: *const Self) struct {
            left_is_rs1: F,
            rs1_value: F,
            left_is_pc: F,
            unexpanded_pc: F,
            right_is_rs2: F,
            rs2_value: F,
            right_is_imm: F,
            imm: F,
        } {
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
}

// =============================================================================
// RegistersClaimReduction Prefix-Suffix Prover
// =============================================================================

fn RegistersPrefixSuffixProver(comptime F: type) type {
    return struct {
        const Self = @This();

        // Single (P, Q) pair for eq polynomial
        P: []F, // Prefix eq evals
        Q: []F, // Accumulated witness * suffix

        // Witness MLEs
        rd_write_value: []F,
        rs1_value: []F,
        rs2_value: []F,

        gamma: F,
        gamma_sqr: F,

        prefix_n_vars: usize,
        current_prefix_size: usize,
        current_witness_size: usize,
        in_phase2: bool,

        // Phase 2 eq polynomial
        phase2_eq: ?[]F,

        // r_hi for Phase 2 initialization (eq suffix)
        r_hi: []const F,
        // r_lo for prefix evaluation in Phase 2
        r_lo: []const F,
        // Accumulated prefix challenges
        prefix_challenges: std.ArrayList(F),

        allocator: Allocator,

        pub fn init(
            allocator: Allocator,
            cycle_witnesses: []const r1cs.R1CSCycleInputs(F),
            trace_len: usize,
            r_spartan: []const F,
            gamma: F,
            gamma_sqr: F,
        ) !Self {
            const n_vars = r_spartan.len;
            // Split r into hi (first half) and lo (second half)
            // Jolt convention: PREFIX uses r_lo, SUFFIX uses r_hi
            const split_point = n_vars / 2;
            const r_hi = r_spartan[0..split_point]; // First half -> used for SUFFIX
            const r_lo = r_spartan[split_point..]; // Second half -> used for PREFIX

            // Sizes: prefix_size = 2^len(r_lo), suffix_size = 2^len(r_hi)
            const prefix_n_vars = r_lo.len; // = n_vars - split_point
            const suffix_n_vars = r_hi.len; // = split_point
            const prefix_size: usize = @as(usize, 1) << @intCast(prefix_n_vars);
            const suffix_size: usize = @as(usize, 1) << @intCast(suffix_n_vars);

            // P = eq(r_lo, j) for PREFIX (Jolt convention)
            const P = try allocator.alloc(F, prefix_size);
            var eq_lo = try poly_mod.EqPolynomial(F).init(allocator, r_lo);
            defer eq_lo.deinit();
            const eq_lo_evals = try eq_lo.evals(allocator);
            defer allocator.free(eq_lo_evals);
            @memcpy(P, eq_lo_evals);

            // Suffix evals = eq(r_hi, j) for SUFFIX (Jolt convention)
            var eq_hi = try poly_mod.EqPolynomial(F).init(allocator, r_hi);
            defer eq_hi.deinit();
            const suffix_evals = try eq_hi.evals(allocator);
            defer allocator.free(suffix_evals);

            // Allocate witness MLEs
            const rd_write_value = try allocator.alloc(F, trace_len);
            const rs1_value = try allocator.alloc(F, trace_len);
            const rs2_value = try allocator.alloc(F, trace_len);

            // Initialize Q buffer and fill witness MLEs
            const Q = try allocator.alloc(F, prefix_size);
            @memset(Q, F.zero());

            // Debug: Print first few R1CS witness values
            std.debug.print("[STAGE3] RegistersClaimReduction: trace_len={}, prefix_size={}, suffix_size={}\n", .{ trace_len, prefix_size, suffix_size });

            for (0..prefix_size) |x_lo| {
                var q_acc = F.zero();

                for (0..suffix_size) |x_hi| {
                    const x = x_lo + (x_hi << @intCast(prefix_n_vars));
                    if (x >= trace_len) continue;

                    const witness = &cycle_witnesses[x].values;
                    const rd = witness[R1CSInputIndex.RdWriteValue.toIndex()];
                    const rs1 = witness[R1CSInputIndex.Rs1Value.toIndex()];
                    const rs2 = witness[R1CSInputIndex.Rs2Value.toIndex()];

                    // Debug: Print first few cycles
                    if (x < 5) {
                        // Extract raw u64 values for comparison with Stage 4
                        const rd_limbs = rd.toBytes();
                        const rd_u64: u64 = @as(u64, rd_limbs[0]) |
                            (@as(u64, rd_limbs[1]) << 8) |
                            (@as(u64, rd_limbs[2]) << 16) |
                            (@as(u64, rd_limbs[3]) << 24) |
                            (@as(u64, rd_limbs[4]) << 32) |
                            (@as(u64, rd_limbs[5]) << 40) |
                            (@as(u64, rd_limbs[6]) << 48) |
                            (@as(u64, rd_limbs[7]) << 56);
                        const rs1_limbs = rs1.toBytes();
                        const rs1_u64: u64 = @as(u64, rs1_limbs[0]) |
                            (@as(u64, rs1_limbs[1]) << 8) |
                            (@as(u64, rs1_limbs[2]) << 16) |
                            (@as(u64, rs1_limbs[3]) << 24) |
                            (@as(u64, rs1_limbs[4]) << 32) |
                            (@as(u64, rs1_limbs[5]) << 40) |
                            (@as(u64, rs1_limbs[6]) << 48) |
                            (@as(u64, rs1_limbs[7]) << 56);
                        const rs2_limbs = rs2.toBytes();
                        const rs2_u64: u64 = @as(u64, rs2_limbs[0]) |
                            (@as(u64, rs2_limbs[1]) << 8) |
                            (@as(u64, rs2_limbs[2]) << 16) |
                            (@as(u64, rs2_limbs[3]) << 24) |
                            (@as(u64, rs2_limbs[4]) << 32) |
                            (@as(u64, rs2_limbs[5]) << 40) |
                            (@as(u64, rs2_limbs[6]) << 48) |
                            (@as(u64, rs2_limbs[7]) << 56);
                        std.debug.print("[STAGE3] Cycle {}: rd_wv={}, rs1_v={}, rs2_v={}\n", .{ x, rd_u64, rs1_u64, rs2_u64 });
                    }

                    // Store witness values
                    rd_write_value[x] = rd;
                    rs1_value[x] = rs1;
                    rs2_value[x] = rs2;

                    // v = rd + gamma*rs1 + gamma^2*rs2
                    const v = rd.add(gamma.mul(rs1)).add(gamma_sqr.mul(rs2));

                    // Accumulate: Q[x_lo] += v * suffix[x_hi]
                    q_acc = q_acc.add(v.mul(suffix_evals[x_hi]));
                }

                Q[x_lo] = q_acc;
            }

            return Self{
                .P = P,
                .Q = Q,
                .rd_write_value = rd_write_value,
                .rs1_value = rs1_value,
                .rs2_value = rs2_value,
                .gamma = gamma,
                .gamma_sqr = gamma_sqr,
                .prefix_n_vars = prefix_n_vars,
                .current_prefix_size = prefix_size,
                .current_witness_size = trace_len,
                .in_phase2 = false,
                .phase2_eq = null,
                .r_hi = r_hi,
                .r_lo = r_lo,
                .prefix_challenges = std.ArrayList(F).initCapacity(allocator, @intCast(prefix_n_vars)) catch unreachable,
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.P);
            self.allocator.free(self.Q);
            self.prefix_challenges.deinit();
            self.allocator.free(self.rd_write_value);
            self.allocator.free(self.rs1_value);
            self.allocator.free(self.rs2_value);
            if (self.phase2_eq) |eq| self.allocator.free(eq);
        }

        pub fn computeRoundEvals(self: *Self, previous_claim: F) [3]F {
            if (self.in_phase2) {
                return self.computeRoundEvalsPhase2(previous_claim);
            } else {
                return self.computeRoundEvalsPhase1(previous_claim);
            }
        }

        fn computeRoundEvalsPhase1(self: *Self, previous_claim: F) [3]F {
            const half = self.current_prefix_size / 2;
            var evals: [2]F = .{ F.zero(), F.zero() }; // p(0), p(2)

            for (0..half) |i| {
                const p_0 = self.P[2 * i];
                const p_1 = self.P[2 * i + 1];
                const q_0 = self.Q[2 * i];
                const q_1 = self.Q[2 * i + 1];

                // Extrapolate to X=2
                const p_2 = p_1.add(p_1).sub(p_0);
                const q_2 = q_1.add(q_1).sub(q_0);

                evals[0] = evals[0].add(p_0.mul(q_0));
                evals[1] = evals[1].add(p_2.mul(q_2));
            }

            const p_1 = previous_claim.sub(evals[0]);
            return [3]F{ evals[0], p_1, evals[1] };
        }

        fn computeRoundEvalsPhase2(self: *Self, previous_claim: F) [3]F {
            const eq = self.phase2_eq.?;
            const half = eq.len / 2;
            var evals: [2]F = .{ F.zero(), F.zero() };

            for (0..half) |j| {
                const eq_0 = eq[2 * j];
                const eq_1 = eq[2 * j + 1];
                const rd_0 = self.rd_write_value[2 * j];
                const rd_1 = self.rd_write_value[2 * j + 1];
                const rs1_0 = self.rs1_value[2 * j];
                const rs1_1 = self.rs1_value[2 * j + 1];
                const rs2_0 = self.rs2_value[2 * j];
                const rs2_1 = self.rs2_value[2 * j + 1];

                // Extrapolate
                const eq_2 = eq_1.add(eq_1).sub(eq_0);
                const rd_2 = rd_1.add(rd_1).sub(rd_0);
                const rs1_2 = rs1_1.add(rs1_1).sub(rs1_0);
                const rs2_2 = rs2_1.add(rs2_1).sub(rs2_0);

                const v_0 = rd_0.add(self.gamma.mul(rs1_0)).add(self.gamma_sqr.mul(rs2_0));
                const v_2 = rd_2.add(self.gamma.mul(rs1_2)).add(self.gamma_sqr.mul(rs2_2));

                evals[0] = evals[0].add(eq_0.mul(v_0));
                evals[1] = evals[1].add(eq_2.mul(v_2));
            }

            const p_1 = previous_claim.sub(evals[0]);
            return [3]F{ evals[0], p_1, evals[1] };
        }

        pub fn bind(self: *Self, r_j: F) void {
            if (self.in_phase2) {
                self.bindPhase2(r_j);
            } else {
                if (self.shouldTransitionToPhase2()) {
                    self.transitionToPhase2(r_j);
                } else {
                    self.bindPhase1(r_j);
                }
            }
        }

        fn shouldTransitionToPhase2(self: *Self) bool {
            return std.math.log2_int(usize, self.current_prefix_size) == 1;
        }

        fn bindPhase1(self: *Self, r_j: F) void {
            const new_prefix_size = self.current_prefix_size / 2;

            for (0..new_prefix_size) |i| {
                self.P[i] = self.P[2 * i].add(r_j.mul(self.P[2 * i + 1].sub(self.P[2 * i])));
                self.Q[i] = self.Q[2 * i].add(r_j.mul(self.Q[2 * i + 1].sub(self.Q[2 * i])));
            }

            self.current_prefix_size = new_prefix_size;

            // Also bind witness MLEs in LowToHigh order to match computeRoundEvals indexing
            const witness_new_size = self.current_witness_size / 2;
            for (0..witness_new_size) |i| {
                self.rd_write_value[i] = self.rd_write_value[2 * i].add(r_j.mul(self.rd_write_value[2 * i + 1].sub(self.rd_write_value[2 * i])));
                self.rs1_value[i] = self.rs1_value[2 * i].add(r_j.mul(self.rs1_value[2 * i + 1].sub(self.rs1_value[2 * i])));
                self.rs2_value[i] = self.rs2_value[2 * i].add(r_j.mul(self.rs2_value[2 * i + 1].sub(self.rs2_value[2 * i])));
            }
            self.current_witness_size = witness_new_size;

            // Record challenge for Phase 2 initialization
            self.prefix_challenges.append(r_j) catch unreachable;
        }

        fn transitionToPhase2(self: *Self, r_j: F) void {
            // Final bind and record challenge
            self.bindPhase1(r_j);
            self.in_phase2 = true;

            // Materialize eq polynomial for Phase 2:
            // eq_suffix = eq(r_hi, j) * eq(r_prefix, r_lo)
            // where r_prefix = accumulated prefix challenges

            const remaining_size = self.current_witness_size;
            self.phase2_eq = self.allocator.alloc(F, remaining_size) catch unreachable;

            // Compute eq(r_prefix, r_lo) - the prefix evaluation
            // r_prefix = prefix challenges (reversed from little-endian to big-endian)
            // r_lo = second half of r_spartan (already in big-endian order)
            //
            // Jolt converts prefix_challenges from LITTLE_ENDIAN to BIG_ENDIAN via match_endianness(),
            // which reverses the vector. We need to do the same.
            const reversed_prefix = self.allocator.alloc(F, self.prefix_challenges.items.len) catch unreachable;
            defer self.allocator.free(reversed_prefix);
            for (0..self.prefix_challenges.items.len) |i| {
                reversed_prefix[i] = self.prefix_challenges.items[self.prefix_challenges.items.len - 1 - i];
            }

            var eq_prefix = poly_mod.EqPolynomial(F).init(self.allocator, self.r_lo) catch unreachable;
            defer eq_prefix.deinit();
            const eq_prefix_eval = eq_prefix.evaluate(reversed_prefix);

            // Compute eq(r_hi, j) for each j in [0, remaining_size)
            var eq_suffix = poly_mod.EqPolynomial(F).init(self.allocator, self.r_hi) catch unreachable;
            defer eq_suffix.deinit();
            const eq_suffix_evals = eq_suffix.evals(self.allocator) catch unreachable;
            defer self.allocator.free(eq_suffix_evals);

            // phase2_eq[j] = eq_suffix[j] * eq_prefix_eval
            for (0..remaining_size) |j| {
                self.phase2_eq.?[j] = eq_suffix_evals[j].mul(eq_prefix_eval);
            }
        }

        fn bindPhase2(self: *Self, r_j: F) void {
            // Bind in LowToHigh order to match computeRoundEvalsPhase2 indexing (2*j, 2*j+1)
            const new_size = self.current_witness_size / 2;

            for (0..new_size) |i| {
                self.rd_write_value[i] = self.rd_write_value[2 * i].add(r_j.mul(self.rd_write_value[2 * i + 1].sub(self.rd_write_value[2 * i])));
                self.rs1_value[i] = self.rs1_value[2 * i].add(r_j.mul(self.rs1_value[2 * i + 1].sub(self.rs1_value[2 * i])));
                self.rs2_value[i] = self.rs2_value[2 * i].add(r_j.mul(self.rs2_value[2 * i + 1].sub(self.rs2_value[2 * i])));

                if (self.phase2_eq) |eq| {
                    eq[i] = eq[2 * i].add(r_j.mul(eq[2 * i + 1].sub(eq[2 * i])));
                }
            }
            self.current_witness_size = new_size;
        }

        pub fn finalClaims(self: *const Self) struct {
            rd_write_value: F,
            rs1_value: F,
            rs2_value: F,
        } {
            std.debug.assert(self.current_witness_size == 1);
            return .{
                .rd_write_value = self.rd_write_value[0],
                .rs1_value = self.rs1_value[0],
                .rs2_value = self.rs2_value[0],
            };
        }
    };
}
