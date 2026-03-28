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

// Debug output control - set to true to enable verbose debug prints
const debug_verbose = false;
fn dbg(comptime fmt: []const u8, args: anytype) void {
    if (debug_verbose) std.debug.print(fmt, args);
}

const Allocator = std.mem.Allocator;
const ThreadPool = @import("../../utils/thread_pool.zig").ThreadPool;
const GpuPolyOps = @import("../../gpu/mod.zig").GpuPolyOps;
const poly_mod = @import("../../poly/mod.zig");
const transcripts = @import("../../transcripts/mod.zig");
const jolt_types = @import("../jolt_types.zig");
const r1cs = @import("../r1cs/mod.zig");
const R1CSInputIndex = r1cs.R1CSInputIndex;
const instruction_mod = @import("../instruction/mod.zig");
const field_mod = @import("../../field/mod.zig");
const UnreducedProductAccum = field_mod.UnreducedProductAccum;
const FoldedMulU64 = field_mod.FoldedMulU64;
const RawR1CSInputs = @import("../r1cs/evaluators.zig").RawR1CSInputs;

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
        thread_pool: ?*@import("../../utils/thread_pool.zig").ThreadPool = null,
        gpu_ops: ?*GpuPolyOps = null,

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
            raw_inputs: []const RawR1CSInputs,
            n_cycle_vars: usize,
            r_outer: []const F, // r_cycle from Stage 1 (BIG_ENDIAN)
            r_product: []const F, // r_cycle from Stage 2 product sumcheck (BIG_ENDIAN)
        ) !Stage3Result(F) {
            const num_rounds = n_cycle_vars;
            const trace_len = cycle_witnesses.len;

            // Debug: Check what witnesses we received
            if (comptime debug_verbose) {
                dbg("[STAGE3] generateStage3Proof: cycle_witnesses.len = {}\n", .{cycle_witnesses.len});
                if (cycle_witnesses.len > 0) {
                    dbg("[STAGE3] generateStage3Proof: witness[0].PC (idx 6) = {any}\n", .{cycle_witnesses[0].values[6].toBytesBE()});
                    dbg("[STAGE3] generateStage3Proof: witness[0].UPC (idx 7) = {any}\n", .{cycle_witnesses[0].values[7].toBytesBE()});
                }

                dbg("[STAGE3] Starting with {} rounds, trace_len={}\n", .{ num_rounds, trace_len });

                // DEBUG: Print transcript state BEFORE gamma derivation
                dbg("\n[ZOLT] ========== STAGE 3 BEGIN ==========\n", .{});
                dbg("[ZOLT] STAGE3_PRE: transcript_state = {{ {any} }}\n", .{transcript.state[0..16]});
            }

            // Phase 1: Derive parameters (BEFORE BatchedSumcheck::verify)
            // NOTE: Stage 3 uses challenge_scalar (NOT challenge_scalar_optimized) which means
            // we need challengeScalarFull (no 125-bit masking) to match Jolt's behavior.
            //
            // ShiftSumcheckParams::new - derive 5 gamma powers
            const shift_gamma_powers = try self.deriveGammaPowersFull(transcript, 5);
            defer self.allocator.free(shift_gamma_powers);

            // Debug: Print all 5 gamma powers in LE bytes format for comparison
            if (comptime debug_verbose) {
                dbg("[ZOLT] STAGE3_SHIFT: gamma_powers[0] = {{ {any} }}\n", .{shift_gamma_powers[0].toBytes()});
                dbg("[ZOLT] STAGE3_SHIFT: gamma_powers[1] = {{ {any} }}\n", .{shift_gamma_powers[1].toBytes()});
                dbg("[ZOLT] STAGE3_SHIFT: gamma_powers[4] = {{ {any} }}\n", .{shift_gamma_powers[4].toBytes()});
            }

            // InstructionInputParams::new - derive 1 gamma
            const instr_gamma = transcript.challengeScalarFull();
            // RegistersClaimReductionSumcheckParams::new - derive 1 gamma
            const reg_gamma = transcript.challengeScalarFull();
            const reg_gamma_sqr = reg_gamma.mul(reg_gamma);

            // Compute input claims for each sumcheck instance
            const shift_input_claim = self.computeShiftInputClaim(
                opening_claims,
                shift_gamma_powers,
            );
            if (comptime debug_verbose) {
                dbg("[ZOLT] STAGE3_PRE: input_claim[0] (Shift) = {{ {any} }}\n", .{shift_input_claim.toBytes()});
            }

            const instr_input_claim = self.computeInstructionInputClaim(
                opening_claims,
                instr_gamma,
                instr_gamma,
            );
            if (comptime debug_verbose) {
                dbg("[ZOLT] STAGE3_PRE: input_claim[1] (InstrInput) = {{ {any} }}\n", .{instr_input_claim.toBytes()});
            }

            const reg_input_claim = self.computeRegistersInputClaim(
                opening_claims,
                reg_gamma,
                reg_gamma_sqr,
            );
            if (comptime debug_verbose) {
                dbg("[ZOLT] STAGE3_PRE: input_claim[2] (Registers) = {{ {any} }}\n", .{reg_input_claim.toBytes()});
            }

            // Phase 2: BatchedSumcheck::verify protocol

            // Append input claims to transcript (line 201 in sumcheck.rs)
            transcript.appendScalar("sumcheck_claim", shift_input_claim);
            transcript.appendScalar("sumcheck_claim", instr_input_claim);
            transcript.appendScalar("sumcheck_claim", reg_input_claim);

            // Derive batching coefficients (line 204 in sumcheck.rs)
            // NOTE: Jolt's challenge_vector uses challenge_scalar (full 128 bits, no masking)
            var batching_coeffs: [3]F = undefined;
            for (0..3) |i| {
                batching_coeffs[i] = transcript.challengeScalarFull();
            }
            if (comptime debug_verbose) {
                dbg("[ZOLT] STAGE3_PRE: batching_coeff[0] = {{ {any} }}\n", .{batching_coeffs[0].toBytes()});
            }

            // Compute the combined initial claim
            var combined_claim = shift_input_claim.mul(batching_coeffs[0]);
            combined_claim = combined_claim.add(instr_input_claim.mul(batching_coeffs[1]));
            combined_claim = combined_claim.add(reg_input_claim.mul(batching_coeffs[2]));

            // Allocate challenges
            var challenges = try self.allocator.alloc(F, num_rounds);

            // =========================================================================
            // Initialize Prefix-Suffix Provers for Shift and Registers
            // =========================================================================

            const bench_s3_init = (std.posix.getenv("ZOLT_BENCH") != null);
            const t_init_start = if (bench_s3_init) std.time.nanoTimestamp() else 0;

            // ShiftSumcheck uses EqPlusOnePrefixSuffixPoly decomposition with 4 (P,Q) pairs
            var shift_prover = try ShiftPrefixSuffixProver(F).init(
                self.allocator,
                cycle_witnesses,
                raw_inputs,
                trace_len,
                r_outer,
                r_product,
                shift_gamma_powers,
                self.thread_pool,
            );
            shift_prover.thread_pool = self.thread_pool;
            shift_prover.gpu_ops = self.gpu_ops;
            defer shift_prover.deinit();

            const t_after_shift = if (bench_s3_init) std.time.nanoTimestamp() else 0;

            // RegistersClaimReduction uses EqPolynomial prefix-suffix with 1 (P,Q) pair
            var reg_prover = try RegistersPrefixSuffixProver(F).init(
                self.allocator,
                raw_inputs,
                trace_len,
                r_outer, // r_spartan = r_outer
                reg_gamma,
                reg_gamma_sqr,
                self.thread_pool,
            );
            reg_prover.thread_pool = self.thread_pool;
            reg_prover.gpu_ops = self.gpu_ops;
            defer reg_prover.deinit();

            const t_after_reg = if (bench_s3_init) std.time.nanoTimestamp() else 0;

            // InstructionInputSumcheck uses direct computation (no prefix-suffix in Jolt)
            var instr_prover = try InstructionInputProver(F).init(
                self.allocator,
                cycle_witnesses,
                raw_inputs,
                trace_len,
                r_outer,
                r_product,
                instr_gamma,
                self.thread_pool,
            );
            instr_prover.thread_pool = self.thread_pool;
            instr_prover.gpu_ops = self.gpu_ops;
            defer instr_prover.deinit();

            const t_after_instr = if (bench_s3_init) std.time.nanoTimestamp() else 0;

            if (bench_s3_init) {
                const to_ms_i = struct {
                    fn f(ns: i128) f64 {
                        return @as(f64, @floatFromInt(ns)) / 1_000_000.0;
                    }
                }.f;
                std.debug.print("[BENCH] stage=3 init: shift={d:.1}ms reg={d:.1}ms instr={d:.1}ms\n", .{
                    to_ms_i(t_after_shift - t_init_start),
                    to_ms_i(t_after_reg - t_after_shift),
                    to_ms_i(t_after_instr - t_after_reg),
                });
            }

            if (comptime debug_verbose) {
                // DEBUG: Check initial witness values and compute initial sum
                dbg("\n[ZOLT] INSTR_INIT: trace_len = {}, prover.current_size = {}\n", .{ trace_len, instr_prover.current_size });
                var full_sum = F.zero();
                var left_sum = F.zero();
                var right_sum = F.zero();
                for (0..trace_len) |i| {
                    const left_i = instr_prover.left_is_rs1[i].mul(instr_prover.rs1_value[i])
                        .add(instr_prover.left_is_pc[i].mul(instr_prover.unexpanded_pc[i]));
                    const right_i = instr_prover.right_is_rs2[i].mul(instr_prover.rs2_value[i])
                        .add(instr_prover.right_is_imm[i].mul(instr_prover.imm[i]));
                    const eq_weight_i = instr_prover.eq_stage2[i];
                    full_sum = full_sum.add(eq_weight_i.mul(right_i.add(instr_gamma.mul(left_i))));
                    left_sum = left_sum.add(instr_prover.eq_stage2[i].mul(left_i));
                    right_sum = right_sum.add(instr_prover.eq_stage2[i].mul(right_i));
                }
                dbg("[ZOLT] INSTR_INIT: full_sum = {{ {any} }}\n", .{full_sum.toBytes()[0..8]});
                dbg("[ZOLT] INSTR_INIT: instr_input_claim = {{ {any} }}\n", .{instr_input_claim.toBytes()[0..8]});
                dbg("[ZOLT] INSTR_INIT: sum_equals_claim = {}\n", .{full_sum.eql(instr_input_claim)});

                const left_1_from_openings = opening_claims.get(.{ .Virtual = .{ .poly = .LeftInstructionInput, .sumcheck_id = .SpartanOuter } }) orelse F.zero();
                const right_1_from_openings = opening_claims.get(.{ .Virtual = .{ .poly = .RightInstructionInput, .sumcheck_id = .SpartanOuter } }) orelse F.zero();
                dbg("[ZOLT] INSTR_INIT: eq_weighted_left_sum = {{ {any} }}\n", .{left_sum.toBytes()[0..8]});
                dbg("[ZOLT] INSTR_INIT: left_1_from_openings = {{ {any} }}\n", .{left_1_from_openings.toBytes()[0..8]});
                dbg("[ZOLT] INSTR_INIT: left_match = {}\n", .{left_sum.eql(left_1_from_openings)});
                dbg("[ZOLT] INSTR_INIT: eq_weighted_right_sum = {{ {any} }}\n", .{right_sum.toBytes()[0..8]});
                dbg("[ZOLT] INSTR_INIT: right_1_from_openings = {{ {any} }}\n", .{right_1_from_openings.toBytes()[0..8]});
                dbg("[ZOLT] INSTR_INIT: right_match = {}\n", .{right_sum.eql(right_1_from_openings)});

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
                                dbg("[ZOLT] INSTR_INIT: MISMATCH at cycle {}: computed = {{ {any} }}, witness = {{ {any} }}\n", .{ idx, right_computed.toBytes()[0..8], right_from_witness.toBytes()[0..8] });
                                dbg("[ZOLT]   right_is_rs2 = {{ {any} }}, rs2 = {{ {any} }}\n", .{ instr_prover.right_is_rs2[idx].toBytes()[0..8], instr_prover.rs2_value[idx].toBytes()[0..8] });
                                dbg("[ZOLT]   right_is_imm = {{ {any} }}, imm = {{ {any} }}\n", .{ instr_prover.right_is_imm[idx].toBytes()[0..8], instr_prover.imm[idx].toBytes()[0..8] });
                                if (idx < cycle_witnesses.len) {
                                    const instr = cycle_witnesses[idx].values[R1CSInputIndex.Product.toIndex()];
                                    _ = instr;
                                }
                            }
                        }
                    }
                    dbg("[ZOLT] INSTR_INIT: right mismatch_count = {} / {}\n", .{ mismatch_count, trace_len });
                }
            }

            // (debug removed)

            // Track current claims for each instance
            var current_shift_claim = shift_input_claim;
            var current_instr_claim = instr_input_claim;
            var current_reg_claim = reg_input_claim;

            // Bench accumulators for stage 3 sub-timing
            const bench_s3 = (std.posix.getenv("ZOLT_BENCH") != null);
            var s3_shift_compute_ns: u64 = 0;
            var s3_instr_compute_ns: u64 = 0;
            var s3_reg_compute_ns: u64 = 0;
            var s3_shift_bind_ns: u64 = 0;
            var s3_instr_bind_ns: u64 = 0;
            var s3_reg_bind_ns: u64 = 0;
            var s3_overhead_ns: u64 = 0;
            _ = &s3_overhead_ns;

            // Run sumcheck rounds
            for (0..num_rounds) |round| {
                if (comptime debug_verbose) {
                    dbg("\n[ZOLT] STAGE3_ROUND_{}: current_claim = {{ {any} }}\n", .{ round, combined_claim.toBytes() });

                    // Print current shift/reg claims at phase transitions
                    if (round == 3) {
                        dbg("[ZOLT] PHASE2_START_CLAIM: current_shift_claim = {{ {any} }}\n", .{current_shift_claim.toBytes()});
                        dbg("[ZOLT] PHASE2_START_CLAIM: current_reg_claim = {{ {any} }}\n", .{current_reg_claim.toBytes()});
                    }
                }

                // Compute round polynomial for each instance
                // ShiftSumcheck: degree 2
                const t_shift_c = if (bench_s3) std.time.nanoTimestamp() else 0;
                const shift_evals = shift_prover.computeRoundEvals(current_shift_claim);
                if (bench_s3) s3_shift_compute_ns += @intCast(@as(i128, std.time.nanoTimestamp() - t_shift_c));

                // InstructionInputSumcheck: degree 3
                const t_instr_c = if (bench_s3) std.time.nanoTimestamp() else 0;
                const instr_evals = instr_prover.computeRoundEvals(current_instr_claim);
                if (bench_s3) s3_instr_compute_ns += @intCast(@as(i128, std.time.nanoTimestamp() - t_instr_c));

                if (comptime debug_verbose) {
                    // DEBUG: Verify instr_evals at round 0
                    if (round == 0) {
                        var manual_p0 = F.zero();
                        var manual_p1 = F.zero();
                        const half = instr_prover.current_size / 2;
                        for (0..half) |j| {
                            const left_0 = instr_prover.left_is_rs1[2 * j].mul(instr_prover.rs1_value[2 * j])
                                .add(instr_prover.left_is_pc[2 * j].mul(instr_prover.unexpanded_pc[2 * j]));
                            const right_0 = instr_prover.right_is_rs2[2 * j].mul(instr_prover.rs2_value[2 * j])
                                .add(instr_prover.right_is_imm[2 * j].mul(instr_prover.imm[2 * j]));
                            const eq_w_0 = instr_prover.eq_stage2[2 * j];
                            manual_p0 = manual_p0.add(eq_w_0.mul(right_0.add(instr_gamma.mul(left_0))));

                            const left_1 = instr_prover.left_is_rs1[2 * j + 1].mul(instr_prover.rs1_value[2 * j + 1])
                                .add(instr_prover.left_is_pc[2 * j + 1].mul(instr_prover.unexpanded_pc[2 * j + 1]));
                            const right_1 = instr_prover.right_is_rs2[2 * j + 1].mul(instr_prover.rs2_value[2 * j + 1])
                                .add(instr_prover.right_is_imm[2 * j + 1].mul(instr_prover.imm[2 * j + 1]));
                            const eq_w_1 = instr_prover.eq_stage2[2 * j + 1];
                            manual_p1 = manual_p1.add(eq_w_1.mul(right_1.add(instr_gamma.mul(left_1))));
                        }
                        dbg("[ZOLT] ROUND0_VERIFY: manual_p0 = {{ {any} }}\n", .{manual_p0.toBytes()[0..8]});
                        dbg("[ZOLT] ROUND0_VERIFY: instr_evals[0] = {{ {any} }}\n", .{instr_evals[0].toBytes()[0..8]});
                        dbg("[ZOLT] ROUND0_VERIFY: p0_match = {}\n", .{manual_p0.eql(instr_evals[0])});
                        dbg("[ZOLT] ROUND0_VERIFY: manual_p1 = {{ {any} }}\n", .{manual_p1.toBytes()[0..8]});
                        dbg("[ZOLT] ROUND0_VERIFY: derived p1 = {{ {any} }}\n", .{instr_evals[1].toBytes()[0..8]});
                        dbg("[ZOLT] ROUND0_VERIFY: p0+p1 = {{ {any} }}\n", .{manual_p0.add(manual_p1).toBytes()[0..8]});
                        dbg("[ZOLT] ROUND0_VERIFY: input_claim = {{ {any} }}\n", .{current_instr_claim.toBytes()[0..8]});
                    }
                }

                // RegistersClaimReduction: degree 2
                const t_reg_c = if (bench_s3) std.time.nanoTimestamp() else 0;
                const reg_evals = reg_prover.computeRoundEvals(current_reg_claim);
                if (bench_s3) s3_reg_compute_ns += @intCast(@as(i128, std.time.nanoTimestamp() - t_reg_c));

                // DEBUG: After last round, manually check the formula
                if (comptime debug_verbose) {
                    if (round == num_rounds - 1) {
                        dbg("[ZOLT] LAST_ROUND: instr_evals = [p0={{ {any} }}, p1={{ {any} }}, p2={{ {any} }}, p3={{ {any} }}]\n", .{
                            instr_evals[0].toBytes()[0..8],
                            instr_evals[1].toBytes()[0..8],
                            instr_evals[2].toBytes()[0..8],
                            instr_evals[3].toBytes()[0..8],
                        });

                        // Manually compute what the polynomial value should be at different points
                        // The prover should have current_size = 2 at this point
                        dbg("[ZOLT] LAST_ROUND: instr_prover.current_size = {}\n", .{instr_prover.current_size});

                        // Check the sumcheck invariant: p(0) + p(1) = previous_claim
                        const p0_plus_p1 = instr_evals[0].add(instr_evals[1]);
                        dbg("[ZOLT] LAST_ROUND: p0+p1 = {{ {any} }}\n", .{p0_plus_p1.toBytes()[0..8]});
                        dbg("[ZOLT] LAST_ROUND: current_instr_claim = {{ {any} }}\n", .{current_instr_claim.toBytes()[0..8]});
                        dbg("[ZOLT] LAST_ROUND: sumcheck_invariant_ok = {}\n", .{p0_plus_p1.eql(current_instr_claim)});

                        // Manually compute what f(0) and f(1) should be from the raw values
                        // Before bind, current_size = 2, so we have values at indices 0 and 1
                        const left_0 = instr_prover.left_is_rs1[0].mul(instr_prover.rs1_value[0])
                            .add(instr_prover.left_is_pc[0].mul(instr_prover.unexpanded_pc[0]));
                        const right_0 = instr_prover.right_is_rs2[0].mul(instr_prover.rs2_value[0])
                            .add(instr_prover.right_is_imm[0].mul(instr_prover.imm[0]));
                        const eq_weight_0 = instr_prover.eq_stage2[0];
                        const f_0 = eq_weight_0.mul(right_0.add(instr_gamma.mul(left_0)));

                        const left_1 = instr_prover.left_is_rs1[1].mul(instr_prover.rs1_value[1])
                            .add(instr_prover.left_is_pc[1].mul(instr_prover.unexpanded_pc[1]));
                        const right_1 = instr_prover.right_is_rs2[1].mul(instr_prover.rs2_value[1])
                            .add(instr_prover.right_is_imm[1].mul(instr_prover.imm[1]));
                        const eq_weight_1 = instr_prover.eq_stage2[1];
                        const f_1 = eq_weight_1.mul(right_1.add(instr_gamma.mul(left_1)));

                        dbg("[ZOLT] LAST_ROUND: manual_f0 = {{ {any} }}\n", .{f_0.toBytes()[0..8]});
                        dbg("[ZOLT] LAST_ROUND: manual_f1 = {{ {any} }}\n", .{f_1.toBytes()[0..8]});
                        dbg("[ZOLT] LAST_ROUND: f0_match = {}, f1_match = {}\n", .{ f_0.eql(instr_evals[0]), f_1.eql(instr_evals[1]) });

                        // Check actual witness values at index 1
                        dbg("[ZOLT] LAST_ROUND: left_is_rs1[1] = {{ {any} }}\n", .{instr_prover.left_is_rs1[1].toBytes()[0..8]});
                        dbg("[ZOLT] LAST_ROUND: rs1_value[1] = {{ {any} }}\n", .{instr_prover.rs1_value[1].toBytes()[0..8]});
                        dbg("[ZOLT] LAST_ROUND: eq_outer[1] = {{ {any} }}\n", .{instr_prover.eq_stage2[1].toBytes()[0..8]});
                        dbg("[ZOLT] LAST_ROUND: eq_product[1] = {{ {any} }}\n", .{instr_prover.eq_stage2[1].toBytes()[0..8]});
                        dbg("[ZOLT] LAST_ROUND: eq_weight_1 = {{ {any} }}\n", .{eq_weight_1.toBytes()[0..8]});
                        dbg("[ZOLT] LAST_ROUND: left_1 = {{ {any} }}\n", .{left_1.toBytes()[0..8]});
                        dbg("[ZOLT] LAST_ROUND: right_1 = {{ {any} }}\n", .{right_1.toBytes()[0..8]});
                    }
                }

                // Debug: Check individual prover invariants (ALL rounds)
                if (comptime debug_verbose) {
                    {
                        const shift_sum = shift_evals[0].add(shift_evals[1]);
                        const instr_sum = instr_evals[0].add(instr_evals[1]);
                        const reg_sum = reg_evals[0].add(reg_evals[1]);
                        dbg("[ZOLT] STAGE3_ROUND_{}: shift_p0+p1 = {{ {any} }}, shift_claim = {{ {any} }}, match={}\n", .{ round, shift_sum.toBytes()[0..8], current_shift_claim.toBytes()[0..8], shift_sum.eql(current_shift_claim) });
                        dbg("[ZOLT] STAGE3_ROUND_{}: instr_p0+p1 = {{ {any} }}, instr_claim = {{ {any} }}, match={}\n", .{ round, instr_sum.toBytes()[0..8], current_instr_claim.toBytes()[0..8], instr_sum.eql(current_instr_claim) });
                        dbg("[ZOLT] STAGE3_ROUND_{}: reg_p0+p1 = {{ {any} }}, reg_claim = {{ {any} }}, match={}\n", .{ round, reg_sum.toBytes()[0..8], current_reg_claim.toBytes()[0..8], reg_sum.eql(current_reg_claim) });
                        dbg("[ZOLT] STAGE3_ROUND_{}: shift_phase = {s}, reg_phase = {s}\n", .{ round, if (shift_prover.in_phase2) "PHASE2" else "PHASE1", if (reg_prover.in_phase2) "PHASE2" else "PHASE1" });
                    }
                    dbg("[ZOLT] STAGE3_ROUND_{}: shift_p0 = {{ {any} }}\n", .{ round, shift_evals[0].toBytes() });
                    dbg("[ZOLT] STAGE3_ROUND_{}: shift_p1 = {{ {any} }}\n", .{ round, shift_evals[1].toBytes() });
                }

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
                if (comptime debug_verbose) {
                    if (round < 3) {
                        dbg("[ZOLT] STAGE3_ROUND_{}: p0 = {{ {any} }}\n", .{ round, combined_evals[0].toBytes() });
                        dbg("[ZOLT] STAGE3_ROUND_{}: p1 = {{ {any} }}\n", .{ round, combined_evals[1].toBytes() });
                        dbg("[ZOLT] STAGE3_ROUND_{}: p0+p1 = {{ {any} }}\n", .{ round, combined_evals[0].add(combined_evals[1]).toBytes() });
                        dbg("[ZOLT] STAGE3_ROUND_{}: current_claim (should match p0+p1) = {{ {any} }}\n", .{ round, combined_claim.toBytes() });
                    }
                }

                // Compress evaluations to [c0, c2, c3] using finite differences (no allocation for interp)
                const inv2 = F.fromU64(2).inverse().?;
                const inv6 = F.fromU64(6).inverse().?;
                const d1_c = combined_evals[1].sub(combined_evals[0]);
                const d2_c = combined_evals[2].sub(combined_evals[1]);
                const d3_c = combined_evals[3].sub(combined_evals[2]);
                const dd1_c = d2_c.sub(d1_c);
                const dd2_c = d3_c.sub(d2_c);
                const c3_val = dd2_c.sub(dd1_c).mul(inv6);
                const c2_val = dd1_c.mul(inv2).sub(c3_val.mul(F.fromU64(3)));

                const compressed = try self.allocator.alloc(F, 3);
                compressed[0] = combined_evals[0]; // c0
                compressed[1] = c2_val;
                compressed[2] = c3_val;

                // Append to proof
                try proof.compressed_polys.append(self.allocator, .{
                    .coeffs_except_linear_term = compressed,
                    .allocator = self.allocator,
                });

                // Append compressed poly to transcript
                transcript.appendScalars("sumcheck_poly", compressed);

                // Debug: Print compressed coefficients
                if (comptime debug_verbose) {
                    dbg("[ZOLT] STAGE3_ROUND_{}: c0 = {{ {any} }}\n", .{ round, compressed[0].toBytes() });
                    dbg("[ZOLT] STAGE3_ROUND_{}: c2 = {{ {any} }}\n", .{ round, compressed[1].toBytes() });
                    dbg("[ZOLT] STAGE3_ROUND_{}: c3 = {{ {any} }}\n", .{ round, compressed[2].toBytes() });
                }

                // Derive challenge
                const r_j = transcript.challengeScalar();
                challenges[round] = r_j;

                if (comptime debug_verbose) {
                    dbg("[ZOLT] STAGE3_ROUND_{}: challenge = {{ {any} }}\n", .{ round, r_j.toBytes() });
                }

                // Evaluate combined polynomial at r_j using evalFromHint (no allocation)
                const UniPolyF = poly_mod.UniPoly(F);
                combined_claim = UniPolyF.evalFromHint(.{ compressed[0], compressed[1], compressed[2] }, combined_claim, r_j);
                if (comptime debug_verbose) {
                    dbg("[ZOLT] STAGE3_ROUND_{}: next_claim = {{ {any} }}\n", .{ round, combined_claim.toBytes() });
                }

                // Update individual claims using direct evaluation from evals (no allocation)
                current_shift_claim = UniPolyF.evalFromEvalsDeg2(shift_evals, r_j);
                current_instr_claim = UniPolyF.evalFromEvalsDeg3(instr_evals, r_j);
                current_reg_claim = UniPolyF.evalFromEvalsDeg2(reg_evals, r_j);

                // DEBUG: Verify combined_claim equals batched sum of individual claims
                if (comptime debug_verbose) {
                    if (round < 3) {
                        const batched_sum = batching_coeffs[0].mul(current_shift_claim)
                            .add(batching_coeffs[1].mul(current_instr_claim))
                            .add(batching_coeffs[2].mul(current_reg_claim));
                        dbg("[ZOLT] STAGE3_ROUND_{}: batched_sum = {{ {any} }}\n", .{ round, batched_sum.toBytes()[0..8] });
                        if (!batched_sum.eql(combined_claim)) {
                            dbg("[ZOLT] STAGE3_ROUND_{}: MISMATCH: batched_sum != combined_claim!\n", .{round});
                        }
                    }
                }

                // Bind all provers at r_j
                const t_shift_b = if (bench_s3) std.time.nanoTimestamp() else 0;
                shift_prover.bind(r_j);
                if (bench_s3) s3_shift_bind_ns += @intCast(@as(i128, std.time.nanoTimestamp() - t_shift_b));

                const t_instr_b = if (bench_s3) std.time.nanoTimestamp() else 0;
                instr_prover.bind(r_j);
                if (bench_s3) s3_instr_bind_ns += @intCast(@as(i128, std.time.nanoTimestamp() - t_instr_b));

                const t_reg_b = if (bench_s3) std.time.nanoTimestamp() else 0;
                reg_prover.bind(r_j);
                if (bench_s3) s3_reg_bind_ns += @intCast(@as(i128, std.time.nanoTimestamp() - t_reg_b));

                if (comptime debug_verbose) {
                    // DEBUG: Verify shift prover's accumulated claim after each Phase 2 bind
                    if (shift_prover.in_phase2 and round < num_rounds - 1) {
                        const shift_ws = shift_prover.current_witness_size;
                        var shift_total = F.zero();
                        for (0..shift_ws) |j| {
                            const eq_out = shift_prover.phase2_eq_plus_one_outer.?[j];
                            const eq_prod_val = shift_prover.phase2_eq_plus_one_prod.?[j];
                            const upc_v = shift_prover.unexpanded_pc[j];
                            const pc_val = shift_prover.pc[j];
                            const virt = shift_prover.is_virtual[j];
                            const first = shift_prover.is_first_in_sequence[j];
                            const noop = shift_prover.is_noop[j];
                            const val = upc_v.add(shift_prover.gamma_powers[1].mul(pc_val))
                                .add(shift_prover.gamma_powers[2].mul(virt))
                                .add(shift_prover.gamma_powers[3].mul(first));
                            const term1 = eq_out.mul(val);
                            const term2 = shift_prover.gamma_powers[4].mul(F.one().sub(noop)).mul(eq_prod_val);
                            shift_total = shift_total.add(term1).add(term2);
                        }
                        const shift_verify_match = shift_total.eql(current_shift_claim);
                        dbg("[ZOLT] SHIFT_PHASE2_VERIFY_ROUND_{}: total_sum = {{ {any} }}, claim = {{ {any} }}, match={}\n", .{ round, shift_total.toBytes()[0..8], current_shift_claim.toBytes()[0..8], shift_verify_match });
                    }

                    // DEBUG: Track nonzero count and verify sumcheck invariant after each bind
                    if (round < num_rounds - 1) {
                        const next_half = instr_prover.current_size / 2;
                        if (next_half > 0) {
                            var f0_sum = F.zero();
                            var f1_sum = F.zero();
                            for (0..next_half) |j| {
                                const left_0 = instr_prover.left_is_rs1[2 * j].mul(instr_prover.rs1_value[2 * j])
                                    .add(instr_prover.left_is_pc[2 * j].mul(instr_prover.unexpanded_pc[2 * j]));
                                const right_0 = instr_prover.right_is_rs2[2 * j].mul(instr_prover.rs2_value[2 * j])
                                    .add(instr_prover.right_is_imm[2 * j].mul(instr_prover.imm[2 * j]));
                                const eq_w_0 = instr_prover.eq_stage2[2 * j];
                                const contrib_0 = eq_w_0.mul(right_0.add(instr_gamma.mul(left_0)));
                                f0_sum = f0_sum.add(contrib_0);

                                const left_1 = instr_prover.left_is_rs1[2 * j + 1].mul(instr_prover.rs1_value[2 * j + 1])
                                    .add(instr_prover.left_is_pc[2 * j + 1].mul(instr_prover.unexpanded_pc[2 * j + 1]));
                                const right_1 = instr_prover.right_is_rs2[2 * j + 1].mul(instr_prover.rs2_value[2 * j + 1])
                                    .add(instr_prover.right_is_imm[2 * j + 1].mul(instr_prover.imm[2 * j + 1]));
                                const eq_w_1 = instr_prover.eq_stage2[2 * j + 1];
                                const contrib_1 = eq_w_1.mul(right_1.add(instr_gamma.mul(left_1)));
                                f1_sum = f1_sum.add(contrib_1);
                            }
                            const total_sum = f0_sum.add(f1_sum);
                            const matches = total_sum.eql(current_instr_claim);
                            if (round >= 5 or !matches) {
                                dbg("[ZOLT] VERIFY_ROUND_{}: actual_f0+f1 = {{ {any} }}, current_instr_claim = {{ {any} }}, match={}\n", .{ round + 1, total_sum.toBytes()[0..8], current_instr_claim.toBytes()[0..8], matches });
                            }
                        }
                    }
                }
            }

            if (bench_s3) {
                const to_ms = struct {
                    fn f(ns: u64) f64 {
                        return @as(f64, @floatFromInt(ns)) / 1_000_000.0;
                    }
                }.f;
                std.debug.print("[BENCH] stage=3 shift_compute={d:.1}ms instr_compute={d:.1}ms reg_compute={d:.1}ms shift_bind={d:.1}ms instr_bind={d:.1}ms reg_bind={d:.1}ms\n", .{
                    to_ms(s3_shift_compute_ns), to_ms(s3_instr_compute_ns), to_ms(s3_reg_compute_ns),
                    to_ms(s3_shift_bind_ns),    to_ms(s3_instr_bind_ns),    to_ms(s3_reg_bind_ns),
                });
            }

            if (comptime debug_verbose) {
                dbg("\n[ZOLT] STAGE3_FINAL: output_claim = {{ {any} }}\n", .{combined_claim.toBytes()});
            }

            // DEBUG: Compute expected_output_claim like verifier
            if (comptime debug_verbose) {
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

                dbg("[ZOLT] STAGE3_DEBUG: challenges[0] = {{ {any} }}\n", .{challenges[0].toBytes()[0..8]});
                dbg("[ZOLT] STAGE3_DEBUG: reversed_challenges[0] = {{ {any} }}\n", .{reversed_challenges[0].toBytes()[0..8]});

                // Compute shift_expected = eq+1(r_outer, r_final) * [upc + γ*pc + γ²*virt + γ³*first] + γ⁴*(1-noop)*eq+1(r_prod, r_final)
                const shift_val = s_claims.unexpanded_pc
                    .add(shift_gamma_powers[1].mul(s_claims.pc))
                    .add(shift_gamma_powers[2].mul(s_claims.is_virtual))
                    .add(shift_gamma_powers[3].mul(s_claims.is_first_in_sequence));
                const shift_expected = eq_plus_one_r_outer.mul(shift_val)
                    .add(shift_gamma_powers[4].mul(F.one().sub(s_claims.is_noop)).mul(eq_plus_one_r_prod));

                dbg("\n[ZOLT] STAGE3_DEBUG: shift_val = {{ {any} }}\n", .{shift_val.toBytes()[0..8]});
                dbg("[ZOLT] STAGE3_DEBUG: shift_expected = {{ {any} }}\n", .{shift_expected.toBytes()});
                dbg("[ZOLT] STAGE3_DEBUG: current_shift_claim = {{ {any} }}\n", .{current_shift_claim.toBytes()});
                dbg("[ZOLT] STAGE3_DEBUG: shift_match = {}\n", .{shift_expected.eql(current_shift_claim)});

                // Check prover's eq+1 values
                const prover_eq_plus_one_outer = shift_prover.phase2_eq_plus_one_outer.?[0];
                const prover_eq_plus_one_prod = shift_prover.phase2_eq_plus_one_prod.?[0];
                dbg("[ZOLT] STAGE3_DEBUG: prover eq+1_outer = {{ {any} }}\n", .{prover_eq_plus_one_outer.toBytes()});
                dbg("[ZOLT] STAGE3_DEBUG: verifier eq+1_outer = {{ {any} }}\n", .{eq_plus_one_r_outer.toBytes()});
                dbg("[ZOLT] STAGE3_DEBUG: eq+1_outer match = {}\n", .{prover_eq_plus_one_outer.eql(eq_plus_one_r_outer)});
                dbg("[ZOLT] STAGE3_DEBUG: prover eq+1_prod = {{ {any} }}\n", .{prover_eq_plus_one_prod.toBytes()});
                dbg("[ZOLT] STAGE3_DEBUG: verifier eq+1_prod = {{ {any} }}\n", .{eq_plus_one_r_prod.toBytes()});
                dbg("[ZOLT] STAGE3_DEBUG: eq+1_prod match = {}\n", .{prover_eq_plus_one_prod.eql(eq_plus_one_r_prod)});

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
                const instr_expected = eq_r_stage_2
                    .mul(right_instr.add(instr_gamma.mul(left_instr)));

                dbg("\n[ZOLT] STAGE3_DEBUG: eq_r_stage_1 = {{ {any} }}\n", .{eq_r_stage_1.toBytes()});
                dbg("[ZOLT] STAGE3_DEBUG: eq_r_stage_2 = {{ {any} }}\n", .{eq_r_stage_2.toBytes()});
                dbg("[ZOLT] STAGE3_DEBUG: left_instr (from i_claims) = {{ {any} }}\n", .{left_instr.toBytes()});
                dbg("[ZOLT] STAGE3_DEBUG: right_instr (from i_claims) = {{ {any} }}\n", .{right_instr.toBytes()});

                // Compute directly from prover's final witness values
                const direct_left = instr_prover.left_is_rs1[0].mul(instr_prover.rs1_value[0])
                    .add(instr_prover.left_is_pc[0].mul(instr_prover.unexpanded_pc[0]));
                const direct_right = instr_prover.right_is_rs2[0].mul(instr_prover.rs2_value[0])
                    .add(instr_prover.right_is_imm[0].mul(instr_prover.imm[0]));
                dbg("[ZOLT] STAGE3_DEBUG: direct_left = {{ {any} }}\n", .{direct_left.toBytes()});
                dbg("[ZOLT] STAGE3_DEBUG: direct_right = {{ {any} }}\n", .{direct_right.toBytes()});
                dbg("[ZOLT] STAGE3_DEBUG: left_match = {}, right_match = {}\n", .{ direct_left.eql(left_instr), direct_right.eql(right_instr) });

                // Now recompute instr_expected using prover's eq values
                const prover_eq_weight = instr_prover.eq_stage2[0];
                const prover_f = prover_eq_weight.mul(direct_right.add(instr_gamma.mul(direct_left)));
                dbg("[ZOLT] STAGE3_DEBUG: prover_f = {{ {any} }}\n", .{prover_f.toBytes()});

                // Check the individual claim components
                dbg("[ZOLT] STAGE3_DEBUG: i_claims.left_is_rs1 = {{ {any} }}\n", .{i_claims.left_is_rs1.toBytes()[0..8]});
                dbg("[ZOLT] STAGE3_DEBUG: i_claims.rs1_value = {{ {any} }}\n", .{i_claims.rs1_value.toBytes()[0..8]});
                dbg("[ZOLT] STAGE3_DEBUG: i_claims.left_is_pc = {{ {any} }}\n", .{i_claims.left_is_pc.toBytes()[0..8]});
                dbg("[ZOLT] STAGE3_DEBUG: i_claims.unexpanded_pc = {{ {any} }}\n", .{i_claims.unexpanded_pc.toBytes()[0..8]});

                // Check individual witness MLE values
                dbg("[ZOLT] STAGE3_DEBUG: instr_prover.left_is_rs1[0] = {{ {any} }}\n", .{instr_prover.left_is_rs1[0].toBytes()[0..8]});
                dbg("[ZOLT] STAGE3_DEBUG: instr_prover.rs1_value[0] = {{ {any} }}\n", .{instr_prover.rs1_value[0].toBytes()[0..8]});
                dbg("[ZOLT] STAGE3_DEBUG: instr_prover.left_is_pc[0] = {{ {any} }}\n", .{instr_prover.left_is_pc[0].toBytes()[0..8]});
                dbg("[ZOLT] STAGE3_DEBUG: instr_prover.unexpanded_pc[0] = {{ {any} }}\n", .{instr_prover.unexpanded_pc[0].toBytes()[0..8]});

                dbg("[ZOLT] STAGE3_DEBUG: instr_prover_eq_outer[0] = {{ {any} }}\n", .{instr_prover.eq_stage2[0].toBytes()});
                dbg("[ZOLT] STAGE3_DEBUG: instr_prover_eq_prod[0] = {{ {any} }}\n", .{instr_prover.eq_stage2[0].toBytes()});
                dbg("[ZOLT] STAGE3_DEBUG: instr_expected = {{ {any} }}\n", .{instr_expected.toBytes()});
                dbg("[ZOLT] STAGE3_DEBUG: current_instr_claim = {{ {any} }}\n", .{current_instr_claim.toBytes()});
                dbg("[ZOLT] STAGE3_DEBUG: instr_match = {}\n", .{instr_expected.eql(current_instr_claim)});

                // Compute Registers expected_output_claim
                // eq(r, r_spartan) * (rd + gamma*rs1 + gamma^2*rs2)
                const reg_val = r_claims.rd_write_value
                    .add(reg_gamma.mul(r_claims.rs1_value))
                    .add(reg_gamma_sqr.mul(r_claims.rs2_value));
                const reg_expected = eq_r_stage_1.mul(reg_val);

                dbg("\n[ZOLT] STAGE3_DEBUG: r_claims.rd_write_value = {{ {any} }}\n", .{r_claims.rd_write_value.toBytes()});
                dbg("[ZOLT] STAGE3_DEBUG: r_claims.rs1_value = {{ {any} }}\n", .{r_claims.rs1_value.toBytes()});
                dbg("[ZOLT] STAGE3_DEBUG: r_claims.rs2_value = {{ {any} }}\n", .{r_claims.rs2_value.toBytes()});
                dbg("[ZOLT] STAGE3_DEBUG: reg_gamma = {{ {any} }}\n", .{reg_gamma.toBytes()});
                dbg("[ZOLT] STAGE3_DEBUG: reg_val = {{ {any} }}\n", .{reg_val.toBytes()});
                dbg("[ZOLT] STAGE3_DEBUG: eq_r_stage_1 = {{ {any} }}\n", .{eq_r_stage_1.toBytes()});
                dbg("[ZOLT] STAGE3_DEBUG: reg_expected = {{ {any} }}\n", .{reg_expected.toBytes()});
                dbg("[ZOLT] STAGE3_DEBUG: current_reg_claim = {{ {any} }}\n", .{current_reg_claim.toBytes()});
                dbg("[ZOLT] STAGE3_DEBUG: reg_match = {}\n", .{reg_expected.eql(current_reg_claim)});

                // Also compute what the prover's eq polynomial should be
                const prover_eq_final = reg_prover.phase2_eq.?[0];
                dbg("[ZOLT] STAGE3_DEBUG: prover_eq_final = {{ {any} }}\n", .{prover_eq_final.toBytes()});
                dbg("[ZOLT] STAGE3_DEBUG: prover_eq vs eq_r_stage_1: {}\n", .{prover_eq_final.eql(eq_r_stage_1)});

                // Compute final expected_output_claim
                const final_expected = batching_coeffs[0].mul(shift_expected)
                    .add(batching_coeffs[1].mul(instr_expected))
                    .add(batching_coeffs[2].mul(reg_expected));
                dbg("\n[ZOLT] STAGE3_DEBUG: final_expected = {{ {any} }}\n", .{final_expected.toBytes()});
                dbg("[ZOLT] STAGE3_DEBUG: combined_claim = {{ {any} }}\n", .{combined_claim.toBytes()});
                dbg("[ZOLT] STAGE3_DEBUG: final_match = {}\n", .{final_expected.eql(combined_claim)});
            }

            // Phase 3: Compute and cache opening claims
            // After all rounds, the MLEs are bound to single values
            const shift_claims = shift_prover.finalClaims();
            const instr_claims = instr_prover.finalClaims();
            const reg_claims = reg_prover.finalClaims();

            // CRITICAL DIAGNOSTIC: Verify reg_claims match MLE at challenges
            if (comptime debug_verbose) {
                // Build eq(challenges, j) table
                const eq_check = self.allocator.alloc(F, trace_len) catch unreachable;
                defer self.allocator.free(eq_check);
                eq_check[0] = F.one();
                var eq_check_sz: usize = 1;
                // Process challenges in REVERSE order so that challenges[0] binds LSB
                var ri_rev: usize = num_rounds;
                while (ri_rev > 0) {
                    ri_rev -= 1;
                    const c_i = challenges[ri_rev];
                    const one_m_c = F.one().sub(c_i);
                    var idx: usize = eq_check_sz;
                    while (idx > 0) {
                        idx -= 1;
                        eq_check[2 * idx + 1] = eq_check[idx].mul(c_i);
                        eq_check[2 * idx] = eq_check[idx].mul(one_m_c);
                    }
                    eq_check_sz *= 2;
                }

                // Print eq_check entries for comparison with Stage 4
                dbg("[STAGE3]   eq_check[0] = {any}\n", .{eq_check[0].toBytes()[0..8]});
                dbg("[STAGE3]   eq_check[1] = {any}\n", .{eq_check[1].toBytes()[0..8]});
                dbg("[STAGE3]   eq_check[63] = {any}\n", .{eq_check[63].toBytes()[0..8]});
                dbg("[STAGE3]   eq_check[64] = {any}\n", .{eq_check[64].toBytes()[0..8]});
                dbg("[STAGE3]   challenges[0] = {any}\n", .{challenges[0].toBytes()[0..8]});
                dbg("[STAGE3]   challenges[7] = {any}\n", .{challenges[7].toBytes()[0..8]});
                dbg("[STAGE3]   num_rounds = {}\n", .{num_rounds});
                dbg("[STAGE3]   trace_len = {}\n", .{trace_len});
                // Print first few non-zero rs1 witness contributions with their eq weight
                var rs1_partial_contributions: usize = 0;
                var rs1_nonzero_count: usize = 0;
                for (0..trace_len) |jj| {
                    if (jj < cycle_witnesses.len) {
                        const rs1_val = cycle_witnesses[jj].values[R1CSInputIndex.Rs1Value.toIndex()];
                        if (!rs1_val.eql(F.zero())) {
                            rs1_nonzero_count += 1;
                            if (rs1_partial_contributions < 8) {
                                const contrib = eq_check[jj].mul(rs1_val);
                                dbg("[STAGE3]   rs1_contrib[{}]: val={any}, eq={any}, contrib={any}\n", .{
                                    jj, rs1_val.toBytes()[0..8], eq_check[jj].toBytes()[0..8], contrib.toBytes()[0..8],
                                });
                                rs1_partial_contributions += 1;
                            }
                        }
                    }
                }
                dbg("[STAGE3]   total rs1 nonzero entries: {}\n", .{rs1_nonzero_count});
                // Print ALL non-zero rs1 entries with their cycle index
                var rs1_all_count: usize = 0;
                for (0..trace_len) |jj| {
                    if (jj < cycle_witnesses.len) {
                        const rs1_val = cycle_witnesses[jj].values[R1CSInputIndex.Rs1Value.toIndex()];
                        if (!rs1_val.eql(F.zero())) {
                            rs1_all_count += 1;
                            dbg("[STAGE3]   rs1_all[{}/cycle={}]: val={any}\n", .{
                                rs1_all_count, jj, rs1_val.toBytes()[0..8],
                            });
                        }
                    }
                }

                // Compute MLE of rd_write_value at challenges point
                // We need the ORIGINAL (unbound) witness, but it's been modified by binding.
                // Instead, build from the R1CS witnesses directly.
                var rd_mle_sum = F.zero();
                var rs1_mle_sum = F.zero();
                var rs2_mle_sum = F.zero();
                for (0..trace_len) |jj| {
                    if (jj < cycle_witnesses.len) {
                        const rd_val = cycle_witnesses[jj].values[R1CSInputIndex.RdWriteValue.toIndex()];
                        const rs1_val = cycle_witnesses[jj].values[R1CSInputIndex.Rs1Value.toIndex()];
                        const rs2_val = cycle_witnesses[jj].values[R1CSInputIndex.Rs2Value.toIndex()];
                        rd_mle_sum = rd_mle_sum.add(eq_check[jj].mul(rd_val));
                        rs1_mle_sum = rs1_mle_sum.add(eq_check[jj].mul(rs1_val));
                        rs2_mle_sum = rs2_mle_sum.add(eq_check[jj].mul(rs2_val));
                    }
                }

                // Also compute MLE by manual binding (to check the binding logic)
                var manual_rd = self.allocator.alloc(F, trace_len) catch unreachable;
                defer self.allocator.free(manual_rd);
                for (0..trace_len) |jj| {
                    if (jj < cycle_witnesses.len) {
                        manual_rd[jj] = cycle_witnesses[jj].values[R1CSInputIndex.RdWriteValue.toIndex()];
                    } else {
                        manual_rd[jj] = F.zero();
                    }
                }
                var manual_sz: usize = trace_len;
                for (0..num_rounds) |ri| {
                    const c_i = challenges[ri];
                    const new_sz = manual_sz / 2;
                    for (0..new_sz) |idx| {
                        manual_rd[idx] = manual_rd[2 * idx].add(c_i.mul(manual_rd[2 * idx + 1].sub(manual_rd[2 * idx])));
                    }
                    manual_sz = new_sz;
                }

                dbg("\n[STAGE3 MLE CHECK]\n", .{});
                dbg("[STAGE3]   rd_mle_sum  = {any}\n", .{rd_mle_sum.toBytes()});
                dbg("[STAGE3]   manual_bind = {any}\n", .{manual_rd[0].toBytes()});
                dbg("[STAGE3]   reg_claim   = {any}\n", .{reg_claims.rd_write_value.toBytes()});
                dbg("[STAGE3]   rd MATCH? {}\n", .{rd_mle_sum.eql(reg_claims.rd_write_value)});
                dbg("[STAGE3]   manual==mle? {}\n", .{manual_rd[0].eql(rd_mle_sum)});
                dbg("[STAGE3]   manual==claim? {}\n", .{manual_rd[0].eql(reg_claims.rd_write_value)});
                dbg("[STAGE3]   rs1_mle_sum = {any}\n", .{rs1_mle_sum.toBytes()});
                dbg("[STAGE3]   rs1_claim   = {any}\n", .{reg_claims.rs1_value.toBytes()});
                dbg("[STAGE3]   rs1 MATCH? {}\n", .{rs1_mle_sum.eql(reg_claims.rs1_value)});
                dbg("[STAGE3]   rs2_mle_sum = {any}\n", .{rs2_mle_sum.toBytes()});
                dbg("[STAGE3]   rs2_claim   = {any}\n", .{reg_claims.rs2_value.toBytes()});
                dbg("[STAGE3]   rs2 MATCH? {}\n", .{rs2_mle_sum.eql(reg_claims.rs2_value)});
            }

            // DEBUG: Print opening claims
            if (comptime debug_verbose) {
                dbg("\n[ZOLT] STAGE3_OPENING: Shift sumcheck claims:\n", .{});
                dbg("[ZOLT] STAGE3_OPENING: unexpanded_pc = {{ {any} }}\n", .{shift_claims.unexpanded_pc.toBytes()});
                dbg("[ZOLT] STAGE3_OPENING: pc = {{ {any} }}\n", .{shift_claims.pc.toBytes()});
                dbg("[ZOLT] STAGE3_OPENING: is_noop = {{ {any} }}\n", .{shift_claims.is_noop.toBytes()});
            }

            // Append opening claims to transcript (cache_openings)
            // ShiftSumcheck: 5 claims
            if (comptime debug_verbose) {
                dbg("\n[ZOLT cache_openings] ShiftSumcheck claims appended to transcript:\n", .{});
                dbg("  [0] unexpanded_pc LE = {{ ", .{});
                for (shift_claims.unexpanded_pc.toBytes()) |b| dbg("{x:0>2}, ", .{b});
                dbg("}}\n", .{});
            }
            transcript.appendScalar("opening_claim", shift_claims.unexpanded_pc);

            if (comptime debug_verbose) {
                dbg("  [1] pc LE = {{ ", .{});
                for (shift_claims.pc.toBytes()) |b| dbg("{x:0>2}, ", .{b});
                dbg("}}\n", .{});
            }
            transcript.appendScalar("opening_claim", shift_claims.pc);

            if (comptime debug_verbose) {
                dbg("  [2] is_virtual LE = {{ ", .{});
                for (shift_claims.is_virtual.toBytes()) |b| dbg("{x:0>2}, ", .{b});
                dbg("}}\n", .{});
            }
            transcript.appendScalar("opening_claim", shift_claims.is_virtual);

            if (comptime debug_verbose) {
                dbg("  [3] is_first_in_sequence LE = {{ ", .{});
                for (shift_claims.is_first_in_sequence.toBytes()) |b| dbg("{x:0>2}, ", .{b});
                dbg("}}\n", .{});
            }
            transcript.appendScalar("opening_claim", shift_claims.is_first_in_sequence);

            if (comptime debug_verbose) {
                dbg("  [4] is_noop LE = {{ ", .{});
                for (shift_claims.is_noop.toBytes()) |b| dbg("{x:0>2}, ", .{b});
                dbg("}}\n", .{});
            }
            transcript.appendScalar("opening_claim", shift_claims.is_noop);

            // InstructionInputSumcheck: 8 claims
            if (comptime debug_verbose) {
                dbg("[ZOLT cache_openings] InstructionInputVirtualization claims:\n", .{});
                dbg("  [5] left_is_rs1 LE = {{ ", .{});
                for (instr_claims.left_is_rs1.toBytes()) |b| dbg("{x:0>2}, ", .{b});
                dbg("}}\n", .{});
            }
            transcript.appendScalar("opening_claim", instr_claims.left_is_rs1);

            if (comptime debug_verbose) {
                dbg("  [6] rs1_value LE = {{ ", .{});
                for (instr_claims.rs1_value.toBytes()) |b| dbg("{x:0>2}, ", .{b});
                dbg("}}\n", .{});
            }
            transcript.appendScalar("opening_claim", instr_claims.rs1_value);

            if (comptime debug_verbose) {
                dbg("  [7] left_is_pc LE = {{ ", .{});
                for (instr_claims.left_is_pc.toBytes()) |b| dbg("{x:0>2}, ", .{b});
                dbg("}}\n", .{});
            }
            transcript.appendScalar("opening_claim", instr_claims.left_is_pc);

            if (comptime debug_verbose) {
                dbg("  [8] unexpanded_pc LE = {{ ", .{});
                for (instr_claims.unexpanded_pc.toBytes()) |b| dbg("{x:0>2}, ", .{b});
                dbg("}} (ALIASED - same poly UnexpandedPC at same point as Shift)\n", .{});
            }
            // ALIASED: UnexpandedPC already opened at same point by ShiftSumcheck — not flushed to transcript

            if (comptime debug_verbose) {
                dbg("  [9] right_is_rs2 LE = {{ ", .{});
                for (instr_claims.right_is_rs2.toBytes()) |b| dbg("{x:0>2}, ", .{b});
                dbg("}}\n", .{});
            }
            transcript.appendScalar("opening_claim", instr_claims.right_is_rs2);

            if (comptime debug_verbose) {
                dbg("  [10] rs2_value LE = {{ ", .{});
                for (instr_claims.rs2_value.toBytes()) |b| dbg("{x:0>2}, ", .{b});
                dbg("}}\n", .{});
            }
            transcript.appendScalar("opening_claim", instr_claims.rs2_value);

            if (comptime debug_verbose) {
                dbg("  [11] right_is_imm LE = {{ ", .{});
                for (instr_claims.right_is_imm.toBytes()) |b| dbg("{x:0>2}, ", .{b});
                dbg("}}\n", .{});
            }
            transcript.appendScalar("opening_claim", instr_claims.right_is_imm);

            if (comptime debug_verbose) {
                dbg("  [12] imm LE = {{ ", .{});
                for (instr_claims.imm.toBytes()) |b| dbg("{x:0>2}, ", .{b});
                dbg("}}\n", .{});
            }
            transcript.appendScalar("opening_claim", instr_claims.imm);

            // RegistersClaimReduction: 3 claims
            if (comptime debug_verbose) {
                dbg("[ZOLT cache_openings] RegistersClaimReduction claims:\n", .{});
                dbg("  [13] rd_write_value LE = {{ ", .{});
                for (reg_claims.rd_write_value.toBytes()) |b| dbg("{x:0>2}, ", .{b});
                dbg("}}\n", .{});
            }
            transcript.appendScalar("opening_claim", reg_claims.rd_write_value);

            if (comptime debug_verbose) {
                dbg("  [14] rs1_value LE = {{ ", .{});
                for (reg_claims.rs1_value.toBytes()) |b| dbg("{x:0>2}, ", .{b});
                dbg("}} (ALIASED - same poly Rs1Value at same point as InstrInput)\n", .{});
            }
            // ALIASED: Rs1Value already opened at same point by InstructionInputSumcheck — not flushed

            if (comptime debug_verbose) {
                dbg("  [15] rs2_value LE = {{ ", .{});
                for (reg_claims.rs2_value.toBytes()) |b| dbg("{x:0>2}, ", .{b});
                dbg("}} (ALIASED - same poly Rs2Value at same point as InstrInput)\n", .{});
            }
            // ALIASED: Rs2Value already opened at same point by InstructionInputSumcheck — not flushed

            // Print transcript state after cache_openings
            if (comptime debug_verbose) {
                dbg("[ZOLT cache_openings] Transcript state AFTER all 13 claims (3 aliased): {{ ", .{});
                for (transcript.state[0..8]) |b| dbg("{x:0>2} ", .{b});
                dbg("}}\n", .{});
            }

            if (comptime debug_verbose) {
                // Print transcript state and key claims for comparison with Jolt
                const std_io = @import("std");
                std_io.debug.print("[ZOLT Stage3] Transcript state AFTER cache_openings: ", .{});
                for (transcript.state[0..8]) |b| std_io.debug.print("{x:0>2} ", .{b});
                std_io.debug.print("\n", .{});
                // Print all 16 claims in LE format for byte-by-byte comparison with Jolt
                std_io.debug.print("[ZOLT Stage3 cache_openings] 16 claims (LE bytes, first 16):\n", .{});
                const claims_arr = [_]struct { name: []const u8, val: F }{
                    .{ .name = "UnexpandedPC/Shift", .val = shift_claims.unexpanded_pc },
                    .{ .name = "PC/Shift", .val = shift_claims.pc },
                    .{ .name = "VirtualInst/Shift", .val = shift_claims.is_virtual },
                    .{ .name = "IsFirstInSeq/Shift", .val = shift_claims.is_first_in_sequence },
                    .{ .name = "IsNoop/Shift", .val = shift_claims.is_noop },
                    .{ .name = "LeftIsRs1/IIV", .val = instr_claims.left_is_rs1 },
                    .{ .name = "Rs1Value/IIV", .val = instr_claims.rs1_value },
                    .{ .name = "LeftIsPC/IIV", .val = instr_claims.left_is_pc },
                    .{ .name = "UnexpandedPC/IIV", .val = instr_claims.unexpanded_pc },
                    .{ .name = "RightIsRs2/IIV", .val = instr_claims.right_is_rs2 },
                    .{ .name = "Rs2Value/IIV", .val = instr_claims.rs2_value },
                    .{ .name = "RightIsImm/IIV", .val = instr_claims.right_is_imm },
                    .{ .name = "Imm/IIV", .val = instr_claims.imm },
                    .{ .name = "RdWriteValue/Reg", .val = reg_claims.rd_write_value },
                    .{ .name = "Rs1Value/Reg", .val = reg_claims.rs1_value },
                    .{ .name = "Rs2Value/Reg", .val = reg_claims.rs2_value },
                };
                for (claims_arr, 0..) |entry, i| {
                    const le = entry.val.toBytes();
                    std_io.debug.print("  [{d:>2}] {s}: ", .{ i, entry.name });
                    for (le[0..16]) |b| std_io.debug.print("{x:0>2} ", .{b});
                    std_io.debug.print("\n", .{});
                }
            }

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

            if (comptime debug_verbose) {
                dbg("[ZOLT] SHIFT_INPUT: next_unexpanded_pc = {{ {any} }}\n", .{next_unexpanded_pc.toBytes()});
                dbg("[ZOLT] SHIFT_INPUT: next_pc = {{ {any} }}\n", .{next_pc.toBytes()});
                dbg("[ZOLT] SHIFT_INPUT: next_is_virtual = {{ {any} }}\n", .{next_is_virtual.toBytes()});
                dbg("[ZOLT] SHIFT_INPUT: next_is_first = {{ {any} }}\n", .{next_is_first.toBytes()});
                dbg("[ZOLT] SHIFT_INPUT: next_is_noop = {{ {any} }}\n", .{next_is_noop.toBytes()});
                dbg("[ZOLT] SHIFT_INPUT: gamma_powers[4] = {{ {any} }}\n", .{gamma_powers[4].toBytes()});
                // Also verify the input claim is correctly computed
                dbg("[ZOLT] SHIFT_INPUT: 1 - next_is_noop = {{ {any} }}\n", .{F.one().sub(next_is_noop).toBytes()});
                dbg("[ZOLT] SHIFT_INPUT: gamma^4 * (1-noop) = {{ {any} }}\n", .{gamma_powers[4].mul(F.one().sub(next_is_noop)).toBytes()});
            }

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
            _ = gamma_sqr;
            // Upstream: input_claim = right_claim_stage_2 + gamma * left_claim_stage_2
            // Uses only SpartanProductVirtualization claims (aliased to InstructionClaimReduction)
            const left = opening_claims.get(.{ .Virtual = .{ .poly = .LeftInstructionInput, .sumcheck_id = .SpartanProductVirtualization } }) orelse F.zero();
            const right = opening_claims.get(.{ .Virtual = .{ .poly = .RightInstructionInput, .sumcheck_id = .SpartanProductVirtualization } }) orelse F.zero();

            return right.add(gamma.mul(left));
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
        sumcheck_challenges: std.ArrayListUnmanaged(F),
        in_phase2: bool,

        // Original points (needed for Phase 2 transition)
        r_outer: []const F,
        r_product: []const F,

        // Original trace (needed for Phase 2 witness MLE reconstruction)
        cycle_witnesses: []const r1cs.R1CSCycleInputs(F),
        raw_inputs: []const RawR1CSInputs,
        trace_len: usize,

        // Phase2 materialized polynomials (only allocated when transitioning)
        phase2_eq_plus_one_outer: ?[]F,
        phase2_eq_plus_one_prod: ?[]F,

        allocator: Allocator,
        thread_pool: ?*ThreadPool = null,
        gpu_ops: ?*GpuPolyOps = null,

        pub fn init(
            allocator: Allocator,
            cycle_witnesses: []const r1cs.R1CSCycleInputs(F),
            raw_inputs: []const RawR1CSInputs,
            trace_len: usize,
            r_outer: []const F,
            r_product: []const F,
            gamma_powers: []const F,
            thread_pool: ?*ThreadPool,
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
            poly_mod.EqPolynomial(F).buildEqPlusOneTableInPlace(r_outer_lo, P_0_outer);
            poly_mod.EqPolynomial(F).buildEqPlusOneTableInPlace(r_prod_lo, P_0_prod);

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
            poly_mod.EqPolynomial(F).buildEqAndEqPlusOneInPlace(r_outer_hi, suffix_0_outer, suffix_1_outer);
            poly_mod.EqPolynomial(F).buildEqAndEqPlusOneInPlace(r_prod_hi, suffix_0_prod, suffix_1_prod);

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
            // Each x_lo writes to disjoint Q indices and disjoint witness indices.
            // Optimization: read from RawR1CSInputs (u64/bool) and use mulU64Unreduced
            // for ~4x fewer multiply instructions per accumulation.
            const ShiftInitCtx = struct {
                raw_inputs: []const RawR1CSInputs,
                suffix_0_outer: []const F,
                suffix_1_outer: []const F,
                suffix_0_prod: []const F,
                suffix_1_prod: []const F,
                Q_0_outer: []F,
                Q_1_outer: []F,
                Q_0_prod: []F,
                Q_1_prod: []F,
                unexpanded_pc: []F,
                pc_arr: []F,
                is_virtual: []F,
                is_first_in_sequence: []F,
                is_noop: []F,
                gamma_powers: []const F,
                prefix_n_vars: usize,
                suffix_size: usize,
                trace_len: usize,
            };
            const shift_init_ctx = ShiftInitCtx{
                .raw_inputs = raw_inputs,
                .suffix_0_outer = suffix_0_outer,
                .suffix_1_outer = suffix_1_outer,
                .suffix_0_prod = suffix_0_prod,
                .suffix_1_prod = suffix_1_prod,
                .Q_0_outer = Q_0_outer,
                .Q_1_outer = Q_1_outer,
                .Q_0_prod = Q_0_prod,
                .Q_1_prod = Q_1_prod,
                .unexpanded_pc = unexpanded_pc,
                .pc_arr = pc,
                .is_virtual = is_virtual,
                .is_first_in_sequence = is_first_in_sequence,
                .is_noop = is_noop,
                .gamma_powers = gamma_powers,
                .prefix_n_vars = prefix_n_vars,
                .suffix_size = suffix_size,
                .trace_len = trace_len,
            };
            const shiftInitWorker = struct {
                fn f(c: ShiftInitCtx, x_lo: usize) void {
                    // Use deferred-reduction accumulators for Q buffers
                    var q_0_outer_acc = UnreducedProductAccum.zero();
                    var q_1_outer_acc = UnreducedProductAccum.zero();
                    // For prod Q buffers we just sum suffix values (when !noop), use FoldedMulU64
                    var q_0_prod_acc = FoldedMulU64.zero();
                    var q_1_prod_acc = FoldedMulU64.zero();

                    for (0..c.suffix_size) |x_hi| {
                        const x = x_lo + (x_hi << @intCast(c.prefix_n_vars));
                        if (x >= c.trace_len) continue;

                        // Read from RawR1CSInputs (u64/bool) instead of F
                        const raw = &c.raw_inputs[x];
                        const upc = raw.u64_values[2]; // UnexpandedPC
                        const pc_val = raw.u64_values[1]; // PC
                        const virt = raw.bool_flags[11]; // FlagVirtualInstruction
                        const first_flag = raw.bool_flags[16]; // FlagIsFirstInSequence
                        const noop = raw.bool_flags[20]; // FlagIsNoop

                        // Fill witness MLE arrays (needed for Phase 2 claims)
                        c.unexpanded_pc[x] = F.fromU64(upc);
                        c.pc_arr[x] = F.fromU64(pc_val);
                        c.is_virtual[x] = if (virt) F.one() else F.zero();
                        c.is_first_in_sequence[x] = if (first_flag) F.one() else F.zero();
                        c.is_noop[x] = if (noop) F.one() else F.zero();

                        // Compute v = gamma[0]*upc + gamma[1]*pc + gamma[2]*virt + gamma[3]*first
                        // using mulU64Unreduced (4 mulq each) + bool conditional-add (0 mulq)
                        var v_accum = field_mod.mulU64Unreduced(c.gamma_powers[0], upc);
                        v_accum.addAssign(field_mod.mulU64Unreduced(c.gamma_powers[1], pc_val));
                        if (virt) v_accum.addBigInt4(c.gamma_powers[2].limbs);
                        if (first_flag) v_accum.addBigInt4(c.gamma_powers[3].limbs);
                        const v = field_mod.reduceMulU64(v_accum);

                        // Accumulate v * suffix into Q buffers (deferred Montgomery)
                        q_0_outer_acc.addAssign(v.mulToProductAccum(c.suffix_0_outer[x_hi]));
                        q_1_outer_acc.addAssign(v.mulToProductAccum(c.suffix_1_outer[x_hi]));

                        // For prod Q buffers: (1-noop) * suffix
                        // When noop=false (the common case), just add the suffix value directly
                        if (!noop) {
                            q_0_prod_acc.addBigInt4(c.suffix_0_prod[x_hi].limbs);
                            q_1_prod_acc.addBigInt4(c.suffix_1_prod[x_hi].limbs);
                        }
                    }

                    c.Q_0_outer[x_lo] = q_0_outer_acc.reduce();
                    c.Q_1_outer[x_lo] = q_1_outer_acc.reduce();
                    c.Q_0_prod[x_lo] = field_mod.reduceMulU64(q_0_prod_acc).mul(c.gamma_powers[4]);
                    c.Q_1_prod[x_lo] = field_mod.reduceMulU64(q_1_prod_acc).mul(c.gamma_powers[4]);
                }
            }.f;

            if (thread_pool) |tp| {
                tp.parallelForForce(prefix_size, shift_init_ctx, shiftInitWorker);
            } else {
                for (0..prefix_size) |x_lo| shiftInitWorker(shift_init_ctx, x_lo);
            }

            if (comptime debug_verbose) {
                // DEBUG: Print initial witness MLE values
                dbg("\n[ZOLT] SHIFT_INIT: trace_len={d}, prefix_size={d}, suffix_size={d}\n", .{ trace_len, prefix_size, suffix_size });
                dbg("[ZOLT] SHIFT_INIT: unexpanded_pc[0..4] = ", .{});
                for (0..@min(4, trace_len)) |i| {
                    dbg("{any} ", .{unexpanded_pc[i].toBytes()[0..8]});
                }
                dbg("\n", .{});

                // DEBUG: Print last cycle's Next values (should be 0 for last cycle)
                const last_idx = trace_len - 1;
                const last_witness = &cycle_witnesses[last_idx].values;
                dbg("[ZOLT] SHIFT_INIT: cycle_witnesses[{}].NextUPC = {any}\n", .{ last_idx, last_witness[R1CSInputIndex.NextUnexpandedPC.toIndex()].toBytes()[0..8] });
                dbg("[ZOLT] SHIFT_INIT: cycle_witnesses[{}].NextPC = {any}\n", .{ last_idx, last_witness[R1CSInputIndex.NextPC.toIndex()].toBytes()[0..8] });
                dbg("[ZOLT] SHIFT_INIT: cycle_witnesses[{}].NextIsVirtual = {any}\n", .{ last_idx, last_witness[R1CSInputIndex.NextIsVirtual.toIndex()].toBytes()[0..8] });
                dbg("[ZOLT] SHIFT_INIT: cycle_witnesses[{}].NextIsFirst = {any}\n", .{ last_idx, last_witness[R1CSInputIndex.NextIsFirstInSequence.toIndex()].toBytes()[0..8] });

                // DEBUG: Verify NextUPC[j] = UPC[j+1] relationship for all j
                var next_shift_mismatch_count: usize = 0;
                for (0..trace_len - 1) |check_j| {
                    const next_upc_j = cycle_witnesses[check_j].values[R1CSInputIndex.NextUnexpandedPC.toIndex()];
                    const upc_j_plus_1 = cycle_witnesses[check_j + 1].values[R1CSInputIndex.UnexpandedPC.toIndex()];
                    if (!next_upc_j.eql(upc_j_plus_1)) {
                        if (next_shift_mismatch_count < 5) {
                            dbg("[ZOLT] SHIFT_INIT: MISMATCH NextUPC[{}] != UPC[{}]: {any} != {any}\n", .{ check_j, check_j + 1, next_upc_j.toBytes()[0..8], upc_j_plus_1.toBytes()[0..8] });
                        }
                        next_shift_mismatch_count += 1;
                    }
                }
                if (next_shift_mismatch_count > 0) {
                    dbg("[ZOLT] SHIFT_INIT: Found {} mismatches in NextUPC[j] = UPC[j+1] relationship!\n", .{next_shift_mismatch_count});
                } else {
                    dbg("[ZOLT] SHIFT_INIT: NextUPC[j] = UPC[j+1] verified for all {} cycles\n", .{trace_len - 1});
                }


                // DEBUG: Verify grand sum = Σ P[j]*Q[j]
                var grand_sum = F.zero();
                for (0..prefix_size) |j| {
                    grand_sum = grand_sum.add(P_0_outer[j].mul(Q_0_outer[j]));
                    grand_sum = grand_sum.add(P_1_outer[j].mul(Q_1_outer[j]));
                    grand_sum = grand_sum.add(P_0_prod[j].mul(Q_0_prod[j]));
                    grand_sum = grand_sum.add(P_1_prod[j].mul(Q_1_prod[j]));
                }
                dbg("[ZOLT] SHIFT_INIT: grand_sum(P*Q) = {{ {any} }}\n", .{grand_sum.toBytes()});

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
                dbg("[ZOLT] SHIFT_INIT: direct_sum = {{ {any} }}\n", .{direct_sum.toBytes()});

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
                dbg("[ZOLT] SHIFT_INIT: next_sum (using Next polys with eq) = {{ {any} }}\n", .{next_sum.toBytes()});

                // DEBUG: Compute the difference and the expected boundary term
                const diff = next_sum.sub(direct_sum);
                dbg("[ZOLT] SHIFT_INIT: next_sum - direct_sum = {{ {any} }}\n", .{diff.toBytes()});

                // The boundary term should be eq(r, N-1) * (batched Next values at index N-1)
                // This is the term that's in next_sum but not in direct_sum

                // Also compare next_sum to input_claim - if r_outer is correct, they should match
                // (Assuming the opening claims were computed at r_outer)
                dbg("[ZOLT] SHIFT_INIT: r_outer[0] = {{ {any} }}\n", .{r_outer[0].toBytes()[0..8]});
                dbg("[ZOLT] SHIFT_INIT: r_outer[last] = {{ {any} }}\n", .{r_outer[r_outer.len - 1].toBytes()[0..8]});

                // DEBUG: Verify the relationship Next[j] = Current[j+1]
                dbg("[ZOLT] SHIFT_INIT: Checking Next[j] = Current[j+1] relationship:\n", .{});
                for (0..@min(5, trace_len - 1)) |test_j| {
                    _ = cycle_witnesses[test_j].values[R1CSInputIndex.UnexpandedPC.toIndex()]; // Current j
                    const next_upc_j = cycle_witnesses[test_j].values[R1CSInputIndex.NextUnexpandedPC.toIndex()];
                    const curr_upc_j1 = cycle_witnesses[test_j + 1].values[R1CSInputIndex.UnexpandedPC.toIndex()];
                    dbg("  j={d}: NextUPC[j]={any}, UPC[j+1]={any}, match={}\n", .{
                        test_j,
                        next_upc_j.toBytes()[0..8],
                        curr_upc_j1.toBytes()[0..8],
                        next_upc_j.eql(curr_upc_j1),
                    });
                }

                // DEBUG: Verify eq(r, k-1) = eq+1(r, k) relationship and boundary behavior
                dbg("[ZOLT] SHIFT_INIT: Verifying eq(r, k-1) = eq+1(r, k):\n", .{});

                // Check eq+1(r, 0) - this is the boundary case
                @memset(j_bits, F.zero()); // j = 0 in bits
                const eq_plus_one_at_0 = poly_mod.EqPlusOnePolynomial(F).mle(r_outer, j_bits);
                dbg("  eq+1(r, 0) = {any} (should be ~0 unless r=max)\n", .{eq_plus_one_at_0.toBytes()[0..8]});

                // Check eq+1(r, N-1) where N = trace_len - this is also a boundary case
                const n_minus_1 = trace_len - 1;
                for (0..n_vars) |bit_idx| {
                    const bit_pos: u6 = @intCast(n_vars - 1 - bit_idx);
                    j_bits[bit_idx] = if ((n_minus_1 >> bit_pos) & 1 == 1) F.one() else F.zero();
                }
                const eq_plus_one_at_n_minus_1 = poly_mod.EqPlusOnePolynomial(F).mle(r_outer, j_bits);
                dbg("  eq+1(r, N-1={d}) = {any} (should be 0 by design)\n", .{ n_minus_1, eq_plus_one_at_n_minus_1.toBytes()[0..8] });

                // Check eq(r, N-1) for comparison
                const eq_at_n_minus_1 = poly_mod.EqPolynomial(F).mle(r_outer, j_bits);
                dbg("  eq(r, N-1={d}) = {any}\n", .{ n_minus_1, eq_at_n_minus_1.toBytes()[0..8] });

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

                    const match_ = eq_k_minus_1.eql(eq_plus_one_k);
                    dbg("  k={d}: eq(r,k-1)={any}, eq+1(r,k)={any}, match={}\n", .{
                        k,
                        eq_k_minus_1.toBytes()[0..8],
                        eq_plus_one_k.toBytes()[0..8],
                        match_,
                    });
                }
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
                .sumcheck_challenges = .{},
                .in_phase2 = false,
                .r_outer = r_outer,
                .r_product = r_product,
                .cycle_witnesses = cycle_witnesses,
                .raw_inputs = raw_inputs,
                .trace_len = trace_len,
                .phase2_eq_plus_one_outer = null,
                .phase2_eq_plus_one_prod = null,
                .allocator = allocator,
                .thread_pool = thread_pool,
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
            self.sumcheck_challenges.deinit(self.allocator);
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
            const half = self.current_prefix_size / 2;

            const P1Ctx = struct {
                P0o: []const F, Q0o: []const F,
                P1o: []const F, Q1o: []const F,
                P0p: []const F, Q0p: []const F,
                P1p: []const F, Q1p: []const F,
            };
            const ctx = P1Ctx{
                .P0o = self.P_0_outer, .Q0o = self.Q_0_outer,
                .P1o = self.P_1_outer, .Q1o = self.Q_1_outer,
                .P0p = self.P_0_prod, .Q0p = self.Q_0_prod,
                .P1p = self.P_1_prod, .Q1p = self.Q_1_prod,
            };

            const mapFn = struct {
                fn f(c: P1Ctx, start: usize, end: usize) [3]F {
                    @setEvalBranchQuota(10000);
                    const use_deferred = comptime @hasDecl(F, "mulToProductAccum");
                    if (use_deferred) {
                        var accum: [3]UnreducedProductAccum = .{ UnreducedProductAccum.zero(), UnreducedProductAccum.zero(), UnreducedProductAccum.zero() };
                        const all_pairs = [4][2][]const F{
                            .{ c.P0o, c.Q0o }, .{ c.P1o, c.Q1o },
                            .{ c.P0p, c.Q0p }, .{ c.P1p, c.Q1p },
                        };
                        for (all_pairs) |pair| {
                            const P = pair[0];
                            const Q = pair[1];
                            for (start..end) |i| {
                                const p0 = P[2 * i];
                                const p1 = P[2 * i + 1];
                                const q0 = Q[2 * i];
                                const q1 = Q[2 * i + 1];
                                const p2 = p1.add(p1).sub(p0);
                                const q2 = q1.add(q1).sub(q0);
                                accum[0].addAssign(p0.mulToProductAccum(q0));
                                accum[1].addAssign(p1.mulToProductAccum(q1));
                                accum[2].addAssign(p2.mulToProductAccum(q2));
                            }
                        }
                        return .{ accum[0].reduce(), accum[1].reduce(), accum[2].reduce() };
                    } else {
                        var local: [3]F = .{ F.zero(), F.zero(), F.zero() };
                        const all_pairs = [4][2][]const F{
                            .{ c.P0o, c.Q0o }, .{ c.P1o, c.Q1o },
                            .{ c.P0p, c.Q0p }, .{ c.P1p, c.Q1p },
                        };
                        for (all_pairs) |pair| {
                            const P = pair[0];
                            const Q = pair[1];
                            for (start..end) |i| {
                                const p0 = P[2 * i];
                                const p1 = P[2 * i + 1];
                                const q0 = Q[2 * i];
                                const q1 = Q[2 * i + 1];
                                const p2 = p1.add(p1).sub(p0);
                                const q2 = q1.add(q1).sub(q0);
                                local[0] = local[0].add(p0.mul(q0));
                                local[1] = local[1].add(p1.mul(q1));
                                local[2] = local[2].add(p2.mul(q2));
                            }
                        }
                        return local;
                    }
                }
            }.f;

            const reduceFn = struct {
                fn f(a: [3]F, b: [3]F) [3]F {
                    return .{ a[0].add(b[0]), a[1].add(b[1]), a[2].add(b[2]) };
                }
            }.f;

            const identity = [3]F{ F.zero(), F.zero(), F.zero() };
            const evals = if (self.thread_pool) |tp|
                tp.parallelReduce([3]F, half, identity, ctx, mapFn, reduceFn)
            else
                mapFn(ctx, 0, half);

            // DEBUG: Verify sumcheck invariant p(0) + p(1) = previous_claim
            if (comptime debug_verbose) {
                const computed_sum = evals[0].add(evals[1]);
                if (!computed_sum.eql(previous_claim)) {
                    dbg("[ZOLT] SHIFT INVARIANT FAIL: p(0)+p(1) = {{ {any} }}, expected = {{ {any} }}\n", .{ computed_sum.toBytes(), previous_claim.toBytes() });
                }
            }

            return evals;
        }

        fn computeRoundEvalsPhase2(self: *Self, previous_claim: F) [3]F {
            // Phase2: Use materialized eq+1 polynomials with witness MLEs
            const eq_outer = self.phase2_eq_plus_one_outer.?;
            const eq_prod = self.phase2_eq_plus_one_prod.?;
            // CRITICAL FIX: Use current_witness_size, NOT eq_outer.len!
            // eq_outer.len is the original suffix_size allocation, but current_witness_size
            // shrinks after each bindPhase2 call.
            const half = self.current_witness_size / 2;

            var evals: [2]F = .{ F.zero(), F.zero() };

            // Debug: Print arrays when they're size 2 (last Phase 2 round)
            if (comptime debug_verbose) {
                if (self.current_witness_size == 2) {
                    dbg("[ZOLT] SHIFT_LAST_ROUND: eq_outer[0] = {{ {any} }}\n", .{eq_outer[0].toBytes()});
                    dbg("[ZOLT] SHIFT_LAST_ROUND: eq_outer[1] = {{ {any} }}\n", .{eq_outer[1].toBytes()});
                    dbg("[ZOLT] SHIFT_LAST_ROUND: eq_prod[0] = {{ {any} }}\n", .{eq_prod[0].toBytes()});
                    dbg("[ZOLT] SHIFT_LAST_ROUND: eq_prod[1] = {{ {any} }}\n", .{eq_prod[1].toBytes()});
                    dbg("[ZOLT] SHIFT_LAST_ROUND: upc[0] = {{ {any} }}\n", .{self.unexpanded_pc[0].toBytes()});
                    dbg("[ZOLT] SHIFT_LAST_ROUND: upc[1] = {{ {any} }}\n", .{self.unexpanded_pc[1].toBytes()});
                    dbg("[ZOLT] SHIFT_LAST_ROUND: noop[0] = {{ {any} }}\n", .{self.is_noop[0].toBytes()});
                    dbg("[ZOLT] SHIFT_LAST_ROUND: noop[1] = {{ {any} }}\n", .{self.is_noop[1].toBytes()});
                    dbg("[ZOLT] SHIFT_LAST_ROUND: previous_claim = {{ {any} }}\n", .{previous_claim.toBytes()});
                }
            }

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

                // Debug: last round details
                if (comptime debug_verbose) {
                    if (eq_outer.len == 2) {
                        dbg("[ZOLT] SHIFT_LAST_ROUND: f(0) = {{ {any} }}\n", .{f_0.toBytes()});
                        dbg("[ZOLT] SHIFT_LAST_ROUND: f(2) = {{ {any} }}\n", .{f_2.toBytes()});

                        // Compute f(1) = eq_out_1 * val_1 + γ⁴*(1-noop_1)*eq_prod_1
                        const val_1 = upc_1.add(self.gamma_powers[1].mul(pc_1))
                            .add(self.gamma_powers[2].mul(virt_1))
                            .add(self.gamma_powers[3].mul(first_1));
                        const actual_f1 = eq_out_1.mul(val_1)
                            .add(self.gamma_powers[4].mul(F.one().sub(noop_1)).mul(eq_prod_1));
                        const derived_f1 = previous_claim.sub(f_0);
                        dbg("[ZOLT] SHIFT_LAST_ROUND: actual_f(1) = {{ {any} }}\n", .{actual_f1.toBytes()});
                        dbg("[ZOLT] SHIFT_LAST_ROUND: derived_f(1) = {{ {any} }}\n", .{derived_f1.toBytes()});
                        dbg("[ZOLT] SHIFT_LAST_ROUND: f(1) match = {}\n", .{actual_f1.eql(derived_f1)});
                        dbg("[ZOLT] SHIFT_LAST_ROUND: f(0)+actual_f(1) = {{ {any} }}\n", .{f_0.add(actual_f1).toBytes()});
                        dbg("[ZOLT] SHIFT_LAST_ROUND: claim = {{ {any} }}\n", .{previous_claim.toBytes()});
                        dbg("[ZOLT] SHIFT_LAST_ROUND: f(0)+actual_f(1)==claim: {}\n", .{f_0.add(actual_f1).eql(previous_claim)});
                    }
                }
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
                    self.sumcheck_challenges.append(self.allocator, r_j) catch unreachable;
                    self.bindPhase1(r_j);
                }
            }
        }

        fn shouldTransitionToPhase2(self: *Self) bool {
            // Transition when prefix size is 2 (log2 == 1)
            return std.math.log2_int(usize, self.current_prefix_size) == 1;
        }

        fn bindPhase1(self: *Self, r_j: F) void {
            const new_prefix_size = self.current_prefix_size / 2;

            const BindP1Ctx = struct {
                slices: [8][]F,
                r: F,
                n: usize,
            };
            const bctx = BindP1Ctx{
                .slices = .{
                    self.P_0_outer, self.Q_0_outer,
                    self.P_1_outer, self.Q_1_outer,
                    self.P_0_prod, self.Q_0_prod,
                    self.P_1_prod, self.Q_1_prod,
                },
                .r = r_j,
                .n = new_prefix_size,
            };
            const bindOneFn = struct {
                fn f(c: BindP1Ctx, idx: usize) void {
                    const arr = c.slices[idx];
                    for (0..c.n) |i| {
                        arr[i] = arr[2 * i].add(c.r.mul(arr[2 * i + 1].sub(arr[2 * i])));
                    }
                }
            }.f;

            if (self.gpu_ops) |gpu| {
                if (new_prefix_size >= 16384) {
                    for (bctx.slices) |arr| {
                        gpu.polyBindLow(arr[0 .. new_prefix_size * 2], r_j, arr[0..new_prefix_size]) catch {
                            for (0..new_prefix_size) |i| {
                                arr[i] = arr[2 * i].add(r_j.montgomeryMul(arr[2 * i + 1].sub(arr[2 * i])));
                            }
                        };
                    }
                } else {
                    for (0..8) |idx| bindOneFn(bctx, idx);
                }
            } else if (self.thread_pool) |tp| {
                tp.parallelForForce(8, bctx, bindOneFn);
            } else {
                for (0..8) |idx| bindOneFn(bctx, idx);
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
            self.sumcheck_challenges.append(self.allocator, r_j) catch unreachable;
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

            if (comptime debug_verbose) {
                dbg("\n[ZOLT] SHIFT_PHASE2_START: n_remaining_rounds={d}, suffix_size={d}\n", .{ n_remaining_rounds, suffix_size });
                dbg("[ZOLT] SHIFT_PHASE2_START: r_prefix_be.len={d}\n", .{r_prefix_be.len});
            }

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
            poly_mod.EqPolynomial(F).buildEqPlusOneTableInPlace(r_outer_lo, prefix_0_outer);

            const prefix_1_outer = self.allocator.alloc(F, prefix_size_outer) catch unreachable;
            defer self.allocator.free(prefix_1_outer);
            @memset(prefix_1_outer, F.zero());
            var is_max_outer = F.one();
            for (r_outer_lo) |r_i| {
                is_max_outer = is_max_outer.mul(r_i);
            }
            prefix_1_outer[0] = is_max_outer;

            // Evaluate prefix polynomials at r_prefix
            // NOTE: evaluateMle binds point[0] to the LSB of the table index.
            // The prefix table is in BIG_ENDIAN order (buildEqPlusOneTableInPlace uses BIG_ENDIAN bits).
            // So evaluateMle(table, [r_0, r_1, r_2]) binds r_0→LSB, r_1→mid, r_2→MSB.
            // This matches Jolt's MultilinearPolynomial::evaluate(r_prefix_BE) which also
            // evaluates Σ table[i] * Eq(r_prefix_BE, i_BE), since eq_evals uses BIG_ENDIAN.
            // r_prefix_le = [r_0, r_1, r_2] (LowToHigh: LSB challenge first)
            const prefix_0_eval_outer = evaluateMle(prefix_0_outer, r_prefix_le);
            const prefix_1_eval_outer = evaluateMle(prefix_1_outer, r_prefix_le);

            // DEBUG: Print prefix evaluation details
            if (comptime debug_verbose) {
                dbg("[ZOLT] SHIFT_PREFIX: r_prefix_be[0] = {any}\n", .{r_prefix_be[0].toBytes()[0..8]});
                dbg("[ZOLT] SHIFT_PREFIX: r_prefix_be[last] = {any}\n", .{r_prefix_be[r_prefix_be.len - 1].toBytes()[0..8]});
                dbg("[ZOLT] SHIFT_PREFIX: prefix_0_eval_outer = {any}\n", .{prefix_0_eval_outer.toBytes()[0..8]});
                dbg("[ZOLT] SHIFT_PREFIX: prefix_1_eval_outer = {any}\n", .{prefix_1_eval_outer.toBytes()[0..8]});

                // DEBUG: Print r_outer_hi and r_outer_lo (the fixed points from Stage 1)
                // Using BE for comparison with STAGE1_CHALLENGES output
                dbg("[ZOLT] SHIFT_PREFIX: r_outer_hi[0] (BE) = {any}\n", .{r_outer_hi[0].toBytesBE()[0..8]});
                dbg("[ZOLT] SHIFT_PREFIX: r_outer_hi[last] (BE) = {any}\n", .{r_outer_hi[r_outer_hi.len - 1].toBytesBE()[0..8]});
                dbg("[ZOLT] SHIFT_PREFIX: r_outer_lo[0] (BE) = {any}\n", .{r_outer_lo[0].toBytesBE()[0..8]});
                dbg("[ZOLT] SHIFT_PREFIX: r_outer_lo[last] (BE) = {any}\n", .{r_outer_lo[r_outer_lo.len - 1].toBytesBE()[0..8]});
            }

            // Regenerate suffix polynomials for r_outer
            // SUFFIX uses r_hi (Jolt convention)
            const suffix_0_outer = self.allocator.alloc(F, suffix_size) catch unreachable;
            defer self.allocator.free(suffix_0_outer);
            const suffix_1_outer = self.allocator.alloc(F, suffix_size) catch unreachable;
            defer self.allocator.free(suffix_1_outer);
            poly_mod.EqPolynomial(F).buildEqAndEqPlusOneInPlace(r_outer_hi, suffix_0_outer, suffix_1_outer);

            // Same for r_product
            const r_prod_hi = self.r_product[0..self.suffix_n_vars]; // For SUFFIX
            const r_prod_lo = self.r_product[self.suffix_n_vars..]; // For PREFIX

            // PREFIX uses r_lo (Jolt convention)
            const prefix_size_prod: usize = @as(usize, 1) << @intCast(r_prod_lo.len);
            const prefix_0_prod = self.allocator.alloc(F, prefix_size_prod) catch unreachable;
            defer self.allocator.free(prefix_0_prod);
            poly_mod.EqPolynomial(F).buildEqPlusOneTableInPlace(r_prod_lo, prefix_0_prod);

            const prefix_1_prod = self.allocator.alloc(F, prefix_size_prod) catch unreachable;
            defer self.allocator.free(prefix_1_prod);
            @memset(prefix_1_prod, F.zero());
            var is_max_prod = F.one();
            for (r_prod_lo) |r_i| {
                is_max_prod = is_max_prod.mul(r_i);
            }
            prefix_1_prod[0] = is_max_prod;

            // Use r_prefix_le for MLE evaluation (LSB challenge first to match evaluateMle convention)
            const prefix_0_eval_prod = evaluateMle(prefix_0_prod, r_prefix_le);
            const prefix_1_eval_prod = evaluateMle(prefix_1_prod, r_prefix_le);

            // SUFFIX uses r_hi (Jolt convention)
            const suffix_0_prod = self.allocator.alloc(F, suffix_size) catch unreachable;
            defer self.allocator.free(suffix_0_prod);
            const suffix_1_prod = self.allocator.alloc(F, suffix_size) catch unreachable;
            defer self.allocator.free(suffix_1_prod);
            poly_mod.EqPolynomial(F).buildEqAndEqPlusOneInPlace(r_prod_hi, suffix_0_prod, suffix_1_prod);

            // =====================================================================
            // Step 2: Construct eq+1(r_outer, (r_prefix, j)) for all j in suffix domain
            // eq+1(r, (r_prefix, j)) = prefix_0_eval * suffix_0[j] + prefix_1_eval * suffix_1[j]
            // =====================================================================

            self.phase2_eq_plus_one_outer = self.allocator.alloc(F, suffix_size) catch unreachable;
            self.phase2_eq_plus_one_prod = self.allocator.alloc(F, suffix_size) catch unreachable;

            // Parallelize eq+1 materialization
            const EqMatCtx = struct {
                eq_outer: []F,
                eq_prod: []F,
                s0o: []const F,
                s1o: []const F,
                s0p: []const F,
                s1p: []const F,
                p0o: F,
                p1o: F,
                p0p: F,
                p1p: F,
            };
            const eq_mat_ctx = EqMatCtx{
                .eq_outer = self.phase2_eq_plus_one_outer.?,
                .eq_prod = self.phase2_eq_plus_one_prod.?,
                .s0o = suffix_0_outer, .s1o = suffix_1_outer,
                .s0p = suffix_0_prod, .s1p = suffix_1_prod,
                .p0o = prefix_0_eval_outer, .p1o = prefix_1_eval_outer,
                .p0p = prefix_0_eval_prod, .p1p = prefix_1_eval_prod,
            };
            const eqMatWorker = struct {
                fn f(c: EqMatCtx, j: usize) void {
                    c.eq_outer[j] = c.p0o.mul(c.s0o[j]).add(c.p1o.mul(c.s1o[j]));
                    c.eq_prod[j] = c.p0p.mul(c.s0p[j]).add(c.p1p.mul(c.s1p[j]));
                }
            }.f;
            if (self.thread_pool) |tp| {
                tp.parallelForForce(suffix_size, eq_mat_ctx, eqMatWorker);
            } else {
                for (0..suffix_size) |j| eqMatWorker(eq_mat_ctx, j);
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

            // Sum over prefix domain (parallelized — each j is independent)
            // Optimization: use mulU64Unreduced for u64 witnesses and conditional-add for booleans
            const WitReconCtx = struct {
                raw_inputs_ptr: []const RawR1CSInputs,
                eq_ev: []const F,
                upc_out: []F,
                pc_out: []F,
                virt_out: []F,
                first_out: []F,
                noop_out: []F,
                prefix_dom_size: usize,
                tl: usize,
            };
            const wit_ctx = WitReconCtx{
                .raw_inputs_ptr = self.raw_inputs,
                .eq_ev = eq_evals,
                .upc_out = self.unexpanded_pc,
                .pc_out = self.pc,
                .virt_out = self.is_virtual,
                .first_out = self.is_first_in_sequence,
                .noop_out = self.is_noop,
                .prefix_dom_size = prefix_domain_size,
                .tl = self.trace_len,
            };
            const witReconWorker = struct {
                fn f(c: WitReconCtx, j: usize) void {
                    // Use FoldedMulU64 accumulators for u64 witnesses (4 mulq each)
                    var upc_acc = FoldedMulU64.zero();
                    var pc_acc = FoldedMulU64.zero();
                    // Use FoldedMulU64 for booleans too (conditional addBigInt4, 0 mulq)
                    var virt_acc = FoldedMulU64.zero();
                    var first_acc = FoldedMulU64.zero();
                    var noop_acc = FoldedMulU64.zero();

                    for (0..c.prefix_dom_size) |i| {
                        const trace_idx = j * c.prefix_dom_size + i;
                        if (trace_idx >= c.tl) continue;

                        const raw = &c.raw_inputs_ptr[trace_idx];
                        const eq_eval = c.eq_ev[i];

                        // u64 witnesses: mulU64Unreduced (4 mulq each)
                        upc_acc.addAssign(field_mod.mulU64Unreduced(eq_eval, raw.u64_values[2])); // UnexpandedPC
                        pc_acc.addAssign(field_mod.mulU64Unreduced(eq_eval, raw.u64_values[1])); // PC

                        // Boolean witnesses: conditional add (0 mulq)
                        if (raw.bool_flags[11]) virt_acc.addBigInt4(eq_eval.limbs); // FlagVirtualInstruction
                        if (raw.bool_flags[16]) first_acc.addBigInt4(eq_eval.limbs); // FlagIsFirstInSequence
                        if (raw.bool_flags[20]) noop_acc.addBigInt4(eq_eval.limbs); // FlagIsNoop
                    }

                    c.upc_out[j] = field_mod.reduceMulU64(upc_acc);
                    c.pc_out[j] = field_mod.reduceMulU64(pc_acc);
                    c.virt_out[j] = field_mod.reduceMulU64(virt_acc);
                    c.first_out[j] = field_mod.reduceMulU64(first_acc);
                    c.noop_out[j] = field_mod.reduceMulU64(noop_acc);
                }
            }.f;
            if (self.thread_pool) |tp| {
                tp.parallelForForce(suffix_size, wit_ctx, witReconWorker);
            } else {
                for (0..suffix_size) |j| witReconWorker(wit_ctx, j);
            }

            self.current_witness_size = suffix_size;

            if (comptime debug_verbose) {
                dbg("[ZOLT] SHIFT_PHASE2_START: eq+1_outer[0] = {{ {any} }}\n", .{self.phase2_eq_plus_one_outer.?[0].toBytes()[0..8]});
                dbg("[ZOLT] SHIFT_PHASE2_START: unexpanded_pc[0] = {{ {any} }}\n", .{self.unexpanded_pc[0].toBytes()[0..8]});

                // CRITICAL VERIFICATION: Compute Σ_j f(j) using Phase 2 data
                // This should equal the current_shift_claim at the start of Phase 2
                {
                    var phase2_total_sum = F.zero();
                    for (0..suffix_size) |j| {
                        const eq_out = self.phase2_eq_plus_one_outer.?[j];
                        const eq_prod_v = self.phase2_eq_plus_one_prod.?[j];
                        const upc = self.unexpanded_pc[j];
                        const pc_val = self.pc[j];
                        const virt = self.is_virtual[j];
                        const first = self.is_first_in_sequence[j];
                        const noop = self.is_noop[j];

                        const val = upc.add(self.gamma_powers[1].mul(pc_val))
                            .add(self.gamma_powers[2].mul(virt))
                            .add(self.gamma_powers[3].mul(first));
                        const term1 = eq_out.mul(val);
                        const term2 = self.gamma_powers[4].mul(F.one().sub(noop)).mul(eq_prod_v);
                        phase2_total_sum = phase2_total_sum.add(term1).add(term2);
                    }
                    dbg("[ZOLT] SHIFT_PHASE2_VERIFY: phase2_total_sum = {{ {any} }}\n", .{phase2_total_sum.toBytes()});
                    dbg("[ZOLT] SHIFT_PHASE2_VERIFY: (compare with current_shift_claim at Phase2 start)\n", .{});
                }

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
                    dbg("[ZOLT] SHIFT_DEBUG: expected suffix_0[0] = {any}\n", .{expected_suffix_0_at_0.toBytes()[0..8]});
                    dbg("[ZOLT] SHIFT_DEBUG: actual suffix_0_outer[0] = {any}\n", .{suffix_0_outer[0].toBytes()[0..8]});

                    // eq+1([r0,r1,...], [0,0,...]) should be 0 (because y=0 is not a successor of any x >= 0)
                    // Actually no, eq+1(x, y) = 1 iff y = x+1
                    // For y=[0,...,0] (binary 0), we need x = -1 which doesn't exist in unsigned
                    // So eq+1(anything, [0,...,0]) = 0
                    dbg("[ZOLT] SHIFT_DEBUG: suffix_1_outer[0] should be ~0: {any}\n", .{suffix_1_outer[0].toBytes()[0..8]});

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
                    dbg("[ZOLT] SHIFT_CRITICAL: direct eq+1(r_outer, (zeros, r_prefix)) = {any}\n", .{direct_eq_plus_one.toBytes()});
                    dbg("[ZOLT] SHIFT_CRITICAL: phase2_eq+1_outer[0] = {any}\n", .{self.phase2_eq_plus_one_outer.?[0].toBytes()});
                    dbg("[ZOLT] SHIFT_CRITICAL: match = {}\n", .{direct_eq_plus_one.eql(self.phase2_eq_plus_one_outer.?[0])});

                    // Debug: check formula components
                    // phase2_eq+1_outer[0] = prefix_0_eval * suffix_0[0] + prefix_1_eval * suffix_1[0]
                    const expected_from_formula = prefix_0_eval_outer.mul(suffix_0_outer[0])
                        .add(prefix_1_eval_outer.mul(suffix_1_outer[0]));
                    dbg("[ZOLT] SHIFT_CRITICAL: from_formula = {any}\n", .{expected_from_formula.toBytes()});
                    dbg("[ZOLT] SHIFT_CRITICAL: formula_match = {}\n", .{expected_from_formula.eql(self.phase2_eq_plus_one_outer.?[0])});

                    // Debug: prefix_0 and suffix_0 individually
                    // Direct eq+1(r_lo, y_lo) where y_lo = r_prefix_be
                    const direct_prefix_eval = poly_mod.EqPlusOnePolynomial(F).mle(r_outer_lo, r_prefix_be);
                    dbg("[ZOLT] SHIFT_CRITICAL: direct eq+1(r_lo, y_lo) = {any}\n", .{direct_prefix_eval.toBytes()});
                    dbg("[ZOLT] SHIFT_CRITICAL: prefix_0_eval_outer = {any}\n", .{prefix_0_eval_outer.toBytes()});
                    dbg("[ZOLT] SHIFT_CRITICAL: prefix_match = {}\n", .{direct_prefix_eval.eql(prefix_0_eval_outer)});

                    // Direct eq(r_hi, y_hi) where y_hi = zeros
                    const zeros_hi = self.allocator.alloc(F, self.suffix_n_vars) catch unreachable;
                    defer self.allocator.free(zeros_hi);
                    @memset(zeros_hi, F.zero());
                    const direct_suffix_eval = poly_mod.EqPolynomial(F).mle(r_outer_hi, zeros_hi);
                    dbg("[ZOLT] SHIFT_CRITICAL: direct eq(r_hi, zeros) = {any}\n", .{direct_suffix_eval.toBytes()});
                    dbg("[ZOLT] SHIFT_CRITICAL: suffix_0[0] = {any}\n", .{suffix_0_outer[0].toBytes()});
                    dbg("[ZOLT] SHIFT_CRITICAL: suffix_match = {}\n", .{direct_suffix_eval.eql(suffix_0_outer[0])});
                }
            }
        }

        fn bindPhase2(self: *Self, r_j: F) void {
            const new_size = self.current_witness_size / 2;

            var num_arrays: usize = 5;
            if (self.phase2_eq_plus_one_outer != null) num_arrays += 1;
            if (self.phase2_eq_plus_one_prod != null) num_arrays += 1;

            const BindP2Ctx = struct {
                slices: [7]?[]F,
                r: F,
                n: usize,
            };
            const bctx = BindP2Ctx{
                .slices = .{
                    self.unexpanded_pc, self.pc, self.is_virtual,
                    self.is_first_in_sequence, self.is_noop,
                    self.phase2_eq_plus_one_outer,
                    self.phase2_eq_plus_one_prod,
                },
                .r = r_j,
                .n = new_size,
            };
            const bindOneFn = struct {
                fn f(c: BindP2Ctx, idx: usize) void {
                    const arr = c.slices[idx] orelse return;
                    for (0..c.n) |i| {
                        arr[i] = arr[2 * i].add(c.r.mul(arr[2 * i + 1].sub(arr[2 * i])));
                    }
                }
            }.f;

            if (self.gpu_ops) |gpu| {
                if (new_size >= 16384) {
                    for (bctx.slices) |maybe_arr| {
                        const arr = maybe_arr orelse continue;
                        gpu.polyBindLow(arr[0 .. new_size * 2], r_j, arr[0..new_size]) catch {
                            for (0..new_size) |i| {
                                arr[i] = arr[2 * i].add(r_j.montgomeryMul(arr[2 * i + 1].sub(arr[2 * i])));
                            }
                        };
                    }
                } else {
                    for (0..num_arrays) |idx| bindOneFn(bctx, idx);
                }
            } else if (self.thread_pool) |tp| {
                tp.parallelForForce(num_arrays, bctx, bindOneFn);
            } else {
                for (0..num_arrays) |idx| bindOneFn(bctx, idx);
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

    };
}

// =============================================================================
// InstructionInput Prover (No Prefix-Suffix, uses direct computation)
// =============================================================================

fn InstructionInputProver(comptime F: type) type {
    return struct {
        const Self = @This();

        // Witness MLEs (double-buffered: main + scratch for parallel bind)
        left_is_rs1: []F,
        rs1_value: []F,
        left_is_pc: []F,
        unexpanded_pc: []F,
        right_is_rs2: []F,
        rs2_value: []F,
        right_is_imm: []F,
        imm: []F,

        // Scratch buffers for double-buffer parallel bind
        scratch: [8][]F,

        // Gruen split eq polynomial (factored eq at r_cycle_stage_2)
        gruen_eq: poly_mod.GruenSplitEqPolynomial(F),

        gamma: F,

        current_size: usize,
        allocator: Allocator,
        thread_pool: ?*ThreadPool = null,
        gpu_ops: ?*GpuPolyOps = null,

        pub fn init(
            allocator: Allocator,
            cycle_witnesses: []const r1cs.R1CSCycleInputs(F),
            raw_inputs: []const RawR1CSInputs,
            trace_len: usize,
            r_outer: []const F,
            r_product: []const F,
            gamma: F,
            thread_pool: ?*ThreadPool,
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

            // Fill from RawR1CSInputs for better cache locality (~100 bytes vs 1344 bytes per entry)
            // Booleans use F.one()/F.zero(), u64s use F.fromU64(), Imm from field witnesses (i128)
            const InstrFillCtx = struct {
                raw_inputs_ptr: []const RawR1CSInputs,
                cycle_witnesses_ptr: []const r1cs.R1CSCycleInputs(F),
                left_is_rs1: []F,
                rs1_value_arr: []F,
                left_is_pc: []F,
                unexpanded_pc: []F,
                right_is_rs2: []F,
                rs2_value_arr: []F,
                right_is_imm: []F,
                imm_arr: []F,
                raw_len: usize,
            };
            const fill_ctx = InstrFillCtx{
                .raw_inputs_ptr = raw_inputs,
                .cycle_witnesses_ptr = cycle_witnesses,
                .left_is_rs1 = left_is_rs1,
                .rs1_value_arr = rs1_value,
                .left_is_pc = left_is_pc,
                .unexpanded_pc = unexpanded_pc,
                .right_is_rs2 = right_is_rs2,
                .rs2_value_arr = rs2_value,
                .right_is_imm = right_is_imm,
                .imm_arr = imm,
                .raw_len = raw_inputs.len,
            };
            const fillWorker = struct {
                fn f(c: InstrFillCtx, i: usize) void {
                    if (i < c.raw_len) {
                        const raw = &c.raw_inputs_ptr[i];
                        // Booleans: conditional F.one()/F.zero() (no Montgomery mul)
                        c.left_is_rs1[i] = if (raw.bool_flags[21]) F.one() else F.zero();
                        c.left_is_pc[i] = if (raw.bool_flags[22]) F.one() else F.zero();
                        c.right_is_rs2[i] = if (raw.bool_flags[23]) F.one() else F.zero();
                        c.right_is_imm[i] = if (raw.bool_flags[24]) F.one() else F.zero();
                        // u64 values
                        c.rs1_value_arr[i] = F.fromU64(raw.u64_values[4]); // Rs1Value
                        c.unexpanded_pc[i] = F.fromU64(raw.u64_values[2]); // UnexpandedPC
                        c.rs2_value_arr[i] = F.fromU64(raw.u64_values[5]); // Rs2Value
                        // Imm is i128 — read from field witnesses (correct Montgomery encoding)
                        c.imm_arr[i] = c.cycle_witnesses_ptr[i].values[R1CSInputIndex.Imm.toIndex()];
                    } else {
                        c.left_is_rs1[i] = F.zero();
                        c.rs1_value_arr[i] = F.zero();
                        c.left_is_pc[i] = F.zero();
                        c.unexpanded_pc[i] = F.zero();
                        c.right_is_rs2[i] = F.zero();
                        c.rs2_value_arr[i] = F.zero();
                        c.right_is_imm[i] = F.zero();
                        c.imm_arr[i] = F.zero();
                    }
                }
            }.f;

            if (thread_pool) |tp| {
                tp.parallelForForce(trace_len, fill_ctx, fillWorker);
            } else {
                for (0..trace_len) |i| fillWorker(fill_ctx, i);
            }

            // Initialize Gruen split eq polynomial at r_product (= r_cycle_stage_2)
            _ = r_outer; // Not used for InstructionInput
            const gruen_eq = try poly_mod.GruenSplitEqPolynomial(F).init(allocator, r_product);

            // Allocate scratch buffers for double-buffer parallel bind
            var scratch: [8][]F = undefined;
            for (0..8) |i| {
                scratch[i] = try allocator.alloc(F, trace_len);
            }

            return Self{
                .left_is_rs1 = left_is_rs1,
                .rs1_value = rs1_value,
                .left_is_pc = left_is_pc,
                .unexpanded_pc = unexpanded_pc,
                .right_is_rs2 = right_is_rs2,
                .rs2_value = rs2_value,
                .right_is_imm = right_is_imm,
                .imm = imm,
                .scratch = scratch,
                .gruen_eq = gruen_eq,
                .gamma = gamma,
                .current_size = trace_len,
                .allocator = allocator,
                .thread_pool = thread_pool,
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
            for (0..8) |i| self.allocator.free(self.scratch[i]);
            self.gruen_eq.deinit();
        }

        /// Compute round evaluations [p(0), p(1), p(2), p(3)] for degree-3 polynomial
        /// Uses Gruen split eq: factored E_out × E_in, evaluate at {0, ∞}, reconstruct cubic.
        pub fn computeRoundEvals(self: *Self, previous_claim: F) [4]F {
            const half = self.current_size / 2;
            const tables = self.gruen_eq.getWindowEqTables(self.gruen_eq.current_index, 1);
            const E_out = tables.E_out;
            const E_in = tables.E_in;
            const head_in_bits = tables.head_in_bits;
            const in_mask = (@as(usize, 1) << @intCast(head_in_bits)) -| 1;

            const Ctx = struct {
                left_is_rs1: []const F,
                rs1_value: []const F,
                left_is_pc: []const F,
                unexpanded_pc: []const F,
                right_is_rs2: []const F,
                rs2_value: []const F,
                right_is_imm: []const F,
                imm: []const F,
                E_out: []const F,
                E_in: []const F,
                head_in_bits: usize,
                in_mask: usize,
                gamma: F,
            };

            const ctx = Ctx{
                .left_is_rs1 = self.left_is_rs1,
                .rs1_value = self.rs1_value,
                .left_is_pc = self.left_is_pc,
                .unexpanded_pc = self.unexpanded_pc,
                .right_is_rs2 = self.right_is_rs2,
                .rs2_value = self.rs2_value,
                .right_is_imm = self.right_is_imm,
                .imm = self.imm,
                .E_out = E_out,
                .E_in = E_in,
                .head_in_bits = head_in_bits,
                .in_mask = in_mask,
                .gamma = self.gamma,
            };

            const mapFn = struct {
                fn map(c: Ctx, start: usize, end: usize) [2]F {
                    var q_constant = F.zero();
                    var q_quadratic = F.zero();

                    for (start..end) |j| {
                        // Factored eq: E_prefix = E_out[j >> head_in_bits] * E_in[j & mask]
                        const x_out = j >> @intCast(c.head_in_bits);
                        const x_in = j & c.in_mask;
                        const E_prefix = (if (x_out < c.E_out.len) c.E_out[x_out] else F.one())
                            .mul(if (x_in < c.E_in.len) c.E_in[x_in] else F.one());

                        // Inner polynomial at X=0: val(0) = right(0) + γ*left(0)
                        const left_0 = c.left_is_rs1[2 * j].mul(c.rs1_value[2 * j])
                            .add(c.left_is_pc[2 * j].mul(c.unexpanded_pc[2 * j]));
                        const right_0 = c.right_is_rs2[2 * j].mul(c.rs2_value[2 * j])
                            .add(c.right_is_imm[2 * j].mul(c.imm[2 * j]));
                        const val_0 = right_0.add(c.gamma.mul(left_0));

                        // Inner polynomial "eval at ∞" = coefficient of X² in val(X).
                        // val(X) = right(X) + γ*left(X) where each is sum of products of linears.
                        // For f(X)*g(X) with f linear, g linear: coeff of X² = f_slope * g_slope.
                        const lis1_s = c.left_is_rs1[2 * j + 1].sub(c.left_is_rs1[2 * j]);
                        const rs1_s = c.rs1_value[2 * j + 1].sub(c.rs1_value[2 * j]);
                        const lipc_s = c.left_is_pc[2 * j + 1].sub(c.left_is_pc[2 * j]);
                        const pc_s = c.unexpanded_pc[2 * j + 1].sub(c.unexpanded_pc[2 * j]);
                        const ris2_s = c.right_is_rs2[2 * j + 1].sub(c.right_is_rs2[2 * j]);
                        const rs2_s = c.rs2_value[2 * j + 1].sub(c.rs2_value[2 * j]);
                        const riim_s = c.right_is_imm[2 * j + 1].sub(c.right_is_imm[2 * j]);
                        const imm_s = c.imm[2 * j + 1].sub(c.imm[2 * j]);

                        const left_inf = lis1_s.mul(rs1_s).add(lipc_s.mul(pc_s));
                        const right_inf = ris2_s.mul(rs2_s).add(riim_s.mul(imm_s));
                        const val_inf = right_inf.add(c.gamma.mul(left_inf));

                        // Accumulate: q_constant = Σ E_prefix * val(0)
                        //             q_quadratic = Σ E_prefix * val(∞)  [X² coefficient]
                        q_constant = q_constant.add(E_prefix.mul(val_0));
                        q_quadratic = q_quadratic.add(E_prefix.mul(val_inf));
                    }
                    return .{ q_constant, q_quadratic };
                }
            }.map;

            const reduceFn = struct {
                fn reduce(a: [2]F, b: [2]F) [2]F {
                    return .{ a[0].add(b[0]), a[1].add(b[1]) };
                }
            }.reduce;

            const sums = if (self.thread_pool) |tp|
                tp.parallelReduce([2]F, half, .{ F.zero(), F.zero() }, ctx, mapFn, reduceFn)
            else
                mapFn(ctx, 0, half);

            // Reconstruct degree-3 polynomial from {0, ∞} evaluations
            return self.gruen_eq.computeCubicRoundPoly(sums[0], sums[1], previous_claim);
        }

        /// Get main arrays as a fixed-size array of pointers (for parallel bind)
        fn getMainSlices(self: *Self) [8][]F {
            return .{
                self.left_is_rs1, self.rs1_value,
                self.left_is_pc, self.unexpanded_pc,
                self.right_is_rs2, self.rs2_value,
                self.right_is_imm, self.imm,
            };
        }

        /// Swap main ↔ scratch pointers after double-buffer bind
        fn swapBuffers(self: *Self) void {
            const pairs = .{
                .{ &self.left_is_rs1, &self.scratch[0] },
                .{ &self.rs1_value, &self.scratch[1] },
                .{ &self.left_is_pc, &self.scratch[2] },
                .{ &self.unexpanded_pc, &self.scratch[3] },
                .{ &self.right_is_rs2, &self.scratch[4] },
                .{ &self.rs2_value, &self.scratch[5] },
                .{ &self.right_is_imm, &self.scratch[6] },
                .{ &self.imm, &self.scratch[7] },
            };
            inline for (pairs) |pair| {
                const tmp = pair[0].*;
                pair[0].* = pair[1].*;
                pair[1].* = tmp;
            }
        }

        pub fn bind(self: *Self, r_j: F) void {
            const new_size = self.current_size / 2;

            // GPU bind path removed — InstructionInput uses GruenSplitEq for eq (no flat array)

            if (self.thread_pool) |tp| {
                if (new_size >= 256) {
                    // Double-buffer parallel bind: read from main, write to scratch,
                    // then swap pointers. No data races since src != dst.
                    const BindCtx = struct {
                        src: [8][]const F,
                        dst: [8][]F,
                        r: F,
                    };
                    const main = self.getMainSlices();
                    var src: [8][]const F = undefined;
                    for (0..8) |i| src[i] = main[i];
                    const ctx = BindCtx{
                        .src = src,
                        .dst = self.scratch,
                        .r = r_j,
                    };
                    tp.parallelForForce(new_size, ctx, struct {
                        fn run(c: BindCtx, i: usize) void {
                            inline for (0..8) |a| {
                                const low = c.src[a][2 * i];
                                const high = c.src[a][2 * i + 1];
                                c.dst[a][i] = low.add(c.r.mul(high.sub(low)));
                            }
                        }
                    }.run);
                    self.swapBuffers();
                    self.gruen_eq.bind(r_j);
                    self.current_size = new_size;
                    return;
                }
            }

            // Sequential fallback (small arrays — in-place is safe when sequential)
            for (0..new_size) |i| {
                self.left_is_rs1[i] = self.left_is_rs1[2 * i].add(r_j.mul(self.left_is_rs1[2 * i + 1].sub(self.left_is_rs1[2 * i])));
                self.rs1_value[i] = self.rs1_value[2 * i].add(r_j.mul(self.rs1_value[2 * i + 1].sub(self.rs1_value[2 * i])));
                self.left_is_pc[i] = self.left_is_pc[2 * i].add(r_j.mul(self.left_is_pc[2 * i + 1].sub(self.left_is_pc[2 * i])));
                self.unexpanded_pc[i] = self.unexpanded_pc[2 * i].add(r_j.mul(self.unexpanded_pc[2 * i + 1].sub(self.unexpanded_pc[2 * i])));
                self.right_is_rs2[i] = self.right_is_rs2[2 * i].add(r_j.mul(self.right_is_rs2[2 * i + 1].sub(self.right_is_rs2[2 * i])));
                self.rs2_value[i] = self.rs2_value[2 * i].add(r_j.mul(self.rs2_value[2 * i + 1].sub(self.rs2_value[2 * i])));
                self.right_is_imm[i] = self.right_is_imm[2 * i].add(r_j.mul(self.right_is_imm[2 * i + 1].sub(self.right_is_imm[2 * i])));
                self.imm[i] = self.imm[2 * i].add(r_j.mul(self.imm[2 * i + 1].sub(self.imm[2 * i])));
            }
            self.gruen_eq.bind(r_j);
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

        // Witness MLEs (only allocated at Phase 2 transition, suffix_size)
        rd_write_value: ?[]F,
        rs1_value: ?[]F,
        rs2_value: ?[]F,

        gamma: F,
        gamma_sqr: F,

        prefix_n_vars: usize,
        suffix_n_vars: usize,
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
        prefix_challenges: std.ArrayListUnmanaged(F),

        // Raw integer witness data (for Phase 2 reconstruction)
        raw_inputs: []const RawR1CSInputs,
        trace_len: usize,

        allocator: Allocator,
        thread_pool: ?*ThreadPool = null,
        gpu_ops: ?*GpuPolyOps = null,

        pub fn init(
            allocator: Allocator,
            raw_inputs: []const RawR1CSInputs,
            trace_len: usize,
            r_spartan: []const F,
            gamma: F,
            gamma_sqr: F,
            thread_pool: ?*ThreadPool,
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

            // Initialize Q buffer using RawR1CSInputs with mulU64Unreduced
            // No witness MLEs allocated here — they're reconstructed at Phase 2 transition
            const Q = try allocator.alloc(F, prefix_size);
            @memset(Q, F.zero());

            if (comptime debug_verbose) {
                dbg("[STAGE3] RegistersClaimReduction: trace_len={}, prefix_size={}, suffix_size={}\n", .{ trace_len, prefix_size, suffix_size });
            }

            const RegInitCtx = struct {
                raw_inputs: []const RawR1CSInputs,
                suffix_evals: []const F,
                Q_buf: []F,
                gamma_val: F,
                gamma_sqr_val: F,
                prefix_n_vars_val: usize,
                suffix_size_val: usize,
                trace_len_val: usize,
            };
            const reg_init_ctx = RegInitCtx{
                .raw_inputs = raw_inputs,
                .suffix_evals = suffix_evals,
                .Q_buf = Q,
                .gamma_val = gamma,
                .gamma_sqr_val = gamma_sqr,
                .prefix_n_vars_val = prefix_n_vars,
                .suffix_size_val = suffix_size,
                .trace_len_val = trace_len,
            };
            const regInitWorker = struct {
                fn f(c: RegInitCtx, x_lo: usize) void {
                    var q_acc = UnreducedProductAccum.zero();
                    for (0..c.suffix_size_val) |x_hi| {
                        const x = x_lo + (x_hi << @intCast(c.prefix_n_vars_val));
                        if (x >= c.trace_len_val) continue;

                        const raw = &c.raw_inputs[x];
                        const rd = raw.u64_values[6]; // RdWriteValue
                        const rs1 = raw.u64_values[4]; // Rs1Value
                        const rs2 = raw.u64_values[5]; // Rs2Value

                        // v = rd + gamma*rs1 + gamma^2*rs2 using mulU64Unreduced
                        var v_accum = FoldedMulU64.zero();
                        v_accum.addAssign(field_mod.mulU64Unreduced(F.one(), rd));
                        v_accum.addAssign(field_mod.mulU64Unreduced(c.gamma_val, rs1));
                        v_accum.addAssign(field_mod.mulU64Unreduced(c.gamma_sqr_val, rs2));
                        const v = field_mod.reduceMulU64(v_accum);

                        // Accumulate v * suffix (deferred Montgomery)
                        q_acc.addAssign(v.mulToProductAccum(c.suffix_evals[x_hi]));
                    }
                    c.Q_buf[x_lo] = q_acc.reduce();
                }
            }.f;

            if (thread_pool) |tp| {
                tp.parallelForForce(prefix_size, reg_init_ctx, regInitWorker);
            } else {
                for (0..prefix_size) |x_lo| regInitWorker(reg_init_ctx, x_lo);
            }

            const padded_size = prefix_size * suffix_size;
            return Self{
                .P = P,
                .Q = Q,
                .rd_write_value = null, // Allocated at Phase 2 transition
                .rs1_value = null,
                .rs2_value = null,
                .gamma = gamma,
                .gamma_sqr = gamma_sqr,
                .prefix_n_vars = prefix_n_vars,
                .suffix_n_vars = suffix_n_vars,
                .current_prefix_size = prefix_size,
                .current_witness_size = padded_size,
                .in_phase2 = false,
                .phase2_eq = null,
                .r_hi = r_hi,
                .r_lo = r_lo,
                .prefix_challenges = std.ArrayListUnmanaged(F).initCapacity(allocator, @intCast(prefix_n_vars)) catch unreachable,
                .raw_inputs = raw_inputs,
                .trace_len = trace_len,
                .allocator = allocator,
                .thread_pool = thread_pool,
            };
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.P);
            self.allocator.free(self.Q);
            self.prefix_challenges.deinit(self.allocator);
            if (self.rd_write_value) |v| self.allocator.free(v);
            if (self.rs1_value) |v| self.allocator.free(v);
            if (self.rs2_value) |v| self.allocator.free(v);
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
            const use_deferred = comptime @hasDecl(F, "mulToProductAccum");
            var evals: [2]F = undefined;

            if (use_deferred) {
                var accum: [2]UnreducedProductAccum = .{ UnreducedProductAccum.zero(), UnreducedProductAccum.zero() };
                for (0..half) |i| {
                    const p_0 = self.P[2 * i];
                    const p_1 = self.P[2 * i + 1];
                    const q_0 = self.Q[2 * i];
                    const q_1 = self.Q[2 * i + 1];
                    const p_2 = p_1.add(p_1).sub(p_0);
                    const q_2 = q_1.add(q_1).sub(q_0);

                    accum[0].addAssign(p_0.mulToProductAccum(q_0));
                    accum[1].addAssign(p_2.mulToProductAccum(q_2));
                }
                evals = .{ accum[0].reduce(), accum[1].reduce() };
            } else {
                evals = .{ F.zero(), F.zero() };
                for (0..half) |i| {
                    const p_0 = self.P[2 * i];
                    const p_1 = self.P[2 * i + 1];
                    const q_0 = self.Q[2 * i];
                    const q_1 = self.Q[2 * i + 1];
                    const p_2 = p_1.add(p_1).sub(p_0);
                    const q_2 = q_1.add(q_1).sub(q_0);

                    evals[0] = evals[0].add(p_0.mul(q_0));
                    evals[1] = evals[1].add(p_2.mul(q_2));
                }
            }

            const p_1 = previous_claim.sub(evals[0]);
            return [3]F{ evals[0], p_1, evals[1] };
        }

        fn computeRoundEvalsPhase2(self: *Self, previous_claim: F) [3]F {
            const eq = self.phase2_eq.?;
            const rd = self.rd_write_value.?;
            const rs1 = self.rs1_value.?;
            const rs2 = self.rs2_value.?;
            const half = self.current_witness_size / 2;
            var evals: [2]F = .{ F.zero(), F.zero() };

            for (0..half) |j| {
                const eq_0 = eq[2 * j];
                const eq_1 = eq[2 * j + 1];
                const rd_0 = rd[2 * j];
                const rd_1 = rd[2 * j + 1];
                const rs1_0 = rs1[2 * j];
                const rs1_1 = rs1[2 * j + 1];
                const rs2_0 = rs2[2 * j];
                const rs2_1 = rs2[2 * j + 1];

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

            // Only bind P and Q (prefix-sized arrays)
            // Witness MLEs are NOT bound during Phase 1 — they're reconstructed at transition
            for (0..new_prefix_size) |i| {
                self.P[i] = self.P[2 * i].add(r_j.mul(self.P[2 * i + 1].sub(self.P[2 * i])));
                self.Q[i] = self.Q[2 * i].add(r_j.mul(self.Q[2 * i + 1].sub(self.Q[2 * i])));
            }

            self.current_prefix_size = new_prefix_size;

            // Record challenge for Phase 2 initialization
            self.prefix_challenges.append(self.allocator, r_j) catch unreachable;
        }

        fn transitionToPhase2(self: *Self, r_j: F) void {
            // Final bind and record challenge
            self.bindPhase1(r_j);
            self.in_phase2 = true;

            const suffix_size: usize = @as(usize, 1) << @intCast(self.suffix_n_vars);

            // Materialize eq polynomial for Phase 2:
            // eq(r_spartan, (r_prefix, j)) = eq(r_lo, r_prefix) * eq(r_hi, j)

            // Reverse prefix challenges (LE → BE) for Jolt compatibility
            const reversed_prefix = self.allocator.alloc(F, self.prefix_challenges.items.len) catch unreachable;
            defer self.allocator.free(reversed_prefix);
            for (0..self.prefix_challenges.items.len) |i| {
                reversed_prefix[i] = self.prefix_challenges.items[self.prefix_challenges.items.len - 1 - i];
            }

            var eq_prefix = poly_mod.EqPolynomial(F).init(self.allocator, self.r_lo) catch unreachable;
            defer eq_prefix.deinit();
            const eq_prefix_eval = eq_prefix.evaluate(reversed_prefix);

            // Compute eq(r_hi, j) for each j in suffix domain
            var eq_suffix = poly_mod.EqPolynomial(F).init(self.allocator, self.r_hi) catch unreachable;
            defer eq_suffix.deinit();
            const eq_suffix_evals = eq_suffix.evals(self.allocator) catch unreachable;
            defer self.allocator.free(eq_suffix_evals);

            // phase2_eq[j] = eq_suffix[j] * eq_prefix_eval
            self.phase2_eq = self.allocator.alloc(F, suffix_size) catch unreachable;
            for (0..suffix_size) |j| {
                self.phase2_eq.?[j] = eq_suffix_evals[j].mul(eq_prefix_eval);
            }

            // Reconstruct witness MLEs at suffix_size by contracting over prefix domain
            // using raw_inputs + mulU64Unreduced
            const prefix_domain_size: usize = @as(usize, 1) << @intCast(self.prefix_challenges.items.len);

            // Compute eq(r_prefix_be, i) for all prefix indices using EqPolynomial
            var eq_poly = poly_mod.EqPolynomial(F).init(self.allocator, reversed_prefix) catch unreachable;
            defer eq_poly.deinit();
            const eq_evals_alloc = eq_poly.evals(self.allocator) catch unreachable;
            defer self.allocator.free(eq_evals_alloc);
            const eq_evals = eq_evals_alloc[0..prefix_domain_size];

            // Allocate witness MLEs at suffix_size (not T)
            self.rd_write_value = self.allocator.alloc(F, suffix_size) catch unreachable;
            self.rs1_value = self.allocator.alloc(F, suffix_size) catch unreachable;
            self.rs2_value = self.allocator.alloc(F, suffix_size) catch unreachable;

            // Contract: witness_mle[j] = Sum_i eq_evals[i] * raw_inputs[j*prefix + i].u64_values[idx]
            const RegReconCtx = struct {
                raw_inputs_ptr: []const RawR1CSInputs,
                eq_ev: []const F,
                rd_out: []F,
                rs1_out: []F,
                rs2_out: []F,
                prefix_dom_size: usize,
                tl: usize,
            };
            const recon_ctx = RegReconCtx{
                .raw_inputs_ptr = self.raw_inputs,
                .eq_ev = eq_evals,
                .rd_out = self.rd_write_value.?,
                .rs1_out = self.rs1_value.?,
                .rs2_out = self.rs2_value.?,
                .prefix_dom_size = prefix_domain_size,
                .tl = self.trace_len,
            };
            const regReconWorker = struct {
                fn f(c: RegReconCtx, j: usize) void {
                    var rd_acc = FoldedMulU64.zero();
                    var rs1_acc = FoldedMulU64.zero();
                    var rs2_acc = FoldedMulU64.zero();

                    for (0..c.prefix_dom_size) |i| {
                        const trace_idx = j * c.prefix_dom_size + i;
                        if (trace_idx >= c.tl) continue;

                        const raw = &c.raw_inputs_ptr[trace_idx];
                        const eq_eval = c.eq_ev[i];

                        rd_acc.addAssign(field_mod.mulU64Unreduced(eq_eval, raw.u64_values[6])); // RdWriteValue
                        rs1_acc.addAssign(field_mod.mulU64Unreduced(eq_eval, raw.u64_values[4])); // Rs1Value
                        rs2_acc.addAssign(field_mod.mulU64Unreduced(eq_eval, raw.u64_values[5])); // Rs2Value
                    }

                    c.rd_out[j] = field_mod.reduceMulU64(rd_acc);
                    c.rs1_out[j] = field_mod.reduceMulU64(rs1_acc);
                    c.rs2_out[j] = field_mod.reduceMulU64(rs2_acc);
                }
            }.f;
            if (self.thread_pool) |tp| {
                tp.parallelForForce(suffix_size, recon_ctx, regReconWorker);
            } else {
                for (0..suffix_size) |j| regReconWorker(recon_ctx, j);
            }

            self.current_witness_size = suffix_size;
        }

        fn bindPhase2(self: *Self, r_j: F) void {
            const new_size = self.current_witness_size / 2;

            var num_arrays: usize = 3;
            if (self.phase2_eq != null) num_arrays = 4;

            const RegBP2Ctx = struct {
                slices: [4]?[]F,
                r: F,
                n: usize,
            };
            const bctx = RegBP2Ctx{
                .slices = .{ self.rd_write_value, self.rs1_value, self.rs2_value, self.phase2_eq },
                .r = r_j,
                .n = new_size,
            };
            const bindOneFn = struct {
                fn f(c: RegBP2Ctx, idx: usize) void {
                    const arr = c.slices[idx] orelse return;
                    for (0..c.n) |i| {
                        arr[i] = arr[2 * i].add(c.r.mul(arr[2 * i + 1].sub(arr[2 * i])));
                    }
                }
            }.f;

            if (self.gpu_ops) |gpu| {
                if (new_size >= 16384) {
                    for (bctx.slices) |maybe_arr| {
                        const arr = maybe_arr orelse continue;
                        gpu.polyBindLow(arr[0 .. new_size * 2], r_j, arr[0..new_size]) catch {
                            for (0..new_size) |i| {
                                arr[i] = arr[2 * i].add(r_j.montgomeryMul(arr[2 * i + 1].sub(arr[2 * i])));
                            }
                        };
                    }
                } else {
                    for (0..num_arrays) |idx| bindOneFn(bctx, idx);
                }
            } else if (self.thread_pool) |tp| {
                tp.parallelForForce(num_arrays, bctx, bindOneFn);
            } else {
                for (0..num_arrays) |idx| bindOneFn(bctx, idx);
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
                .rd_write_value = self.rd_write_value.?[0],
                .rs1_value = self.rs1_value.?[0],
                .rs2_value = self.rs2_value.?[0],
            };
        }
    };
}
