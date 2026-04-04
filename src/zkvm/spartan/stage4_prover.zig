//! Stage 4 Batched Sumcheck Prover: RegistersReadWriteChecking + RamValCheck
//!
//! Stage 4 in Jolt consists of 2 batched sumcheck instances:
//! 1. RegistersReadWriteChecking: LOG_REGISTER_COUNT + n_cycle_vars rounds (degree 3)
//! 2. RamValCheck (combined ValEvaluation + ValFinal): n_cycle_vars rounds (degree 3)
//!
//! The RegistersReadWriteChecking instance uses the Gruen-optimized prover,
//! while RamValCheck uses a combined prover computing inc * wa * (lt + gamma).

const std = @import("std");

const Allocator = std.mem.Allocator;
const ThreadPool = @import("zolt_pool").ThreadPool;
const GpuPolyOps = @import("zolt_arith").gpu.GpuPolyOps;
const poly_mod = @import("zolt_arith").poly;
const transcripts = @import("zolt_arith").transcripts;
const jolt_types = @import("../jolt_types.zig");
const constants = @import("../../common/constants.zig");
const ram = @import("../ram/mod.zig");
const jolt_device = @import("../jolt_device.zig");

const stage4_gruen = @import("stage4_gruen_prover.zig");

/// log2(REGISTER_COUNT) where REGISTER_COUNT = 32 (RISCV) + 96 (Virtual) = 128
const LOG_REGISTERS: usize = 7;

pub fn Stage4Result(comptime F: type) type {
    return struct {
        const Self = @This();

        // RegistersRWC claims
        regs_val_claim: F,
        regs_rs1_ra_claim: F,
        regs_rs2_ra_claim: F,
        regs_rd_wa_claim: F,
        regs_inc_claim: F,

        // RamValCheck claims
        val_eval_wa_claim: F,
        val_eval_inc_claim: F,

        // Derived opening points (BIG_ENDIAN)
        regs_r_address: []F,
        regs_r_cycle: []F,
        r_cycle_val: []F,
        r_reduction_be: []F,
        inc_poly_copy: []F,

        allocator: Allocator,

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.regs_r_address);
            self.allocator.free(self.regs_r_cycle);
            self.allocator.free(self.r_cycle_val);
            self.allocator.free(self.r_reduction_be);
            self.allocator.free(self.inc_poly_copy);
        }
    };
}

pub fn Stage4Prover(comptime F: type) type {
    return struct {
        const Self = @This();

        const Blake2bTranscript = transcripts.Blake2bTranscript(F);
        const SumcheckInstanceProof = jolt_types.SumcheckInstanceProof(F);
        const Stage4GruenProver = stage4_gruen.Stage4GruenProver(F);
        const Stage3Claims = stage4_gruen.Stage3Claims(F);
        const ValEvaluationParams = ram.ValEvaluationParams(F);
        const ValEvaluationProver = ram.ValEvaluationProver(F);
        const UniPoly = poly_mod.UniPoly(F);

        allocator: Allocator,
        thread_pool: ?*ThreadPool = null,
        gpu_ops: ?*GpuPolyOps = null,

        pub fn init(allocator: Allocator) Self {
            return Self{ .allocator = allocator };
        }

        /// Generate Stage 4 batched sumcheck proof.
        ///
        /// Batches 2 instances:
        ///   0. RegistersReadWriteChecking (LOG_K + n_cycle_vars rounds)
        ///   1. RamValCheck: combined ValEvaluation + gamma * ValFinal (n_cycle_vars rounds)
        ///
        /// Appends 7 cache_opening claims to transcript before returning.
        pub fn generateStage4Proof(
            self: *Self,
            proof: *SumcheckInstanceProof,
            transcript: *Blake2bTranscript,
            // Stage 2 data
            stage2_challenges: []const F,
            rwc_val_claim: F,
            output_val_final_claim: F,
            rw_phase1: usize,
            rw_phase2: usize,
            // Stage 3 data
            stage3_claims: Stage3Claims,
            stage3_challenges: []const F,
            // Trace data
            trace: anytype,
            memory_trace: anytype,
            initial_ram: ?*const std.AutoHashMapUnmanaged(u64, u64),
            // Dimensions
            n_cycle_vars: usize,
            log_ram_k: usize,
            ram_K: usize,
            // Init eval (precomputed by caller)
            init_eval_for_val_eval: F,
            // Memory config
            memory_layout: ?*const jolt_device.MemoryLayout,
            start_address: u64,
        ) !Stage4Result(F) {
            const stage4_max_rounds = LOG_REGISTERS + n_cycle_vars;

            // ============================================================
            // Derive gamma challenges from transcript
            // ============================================================
            const gamma_stage4 = transcript.challengeScalarFull();

            // Domain separator and gamma for RamValCheck
            transcript.appendBytes("ram_val_check_gamma", &.{});
            const ram_val_check_gamma = transcript.challengeScalarFull();

            // ============================================================
            // Compute r_reduction_be from Stage 2 InstructionClaimReduction challenges
            // ============================================================
            const max_stage2_rounds = log_ram_k + n_cycle_vars;
            const instr_start = max_stage2_rounds - n_cycle_vars;

            var r_reduction_be = try self.allocator.alloc(F, n_cycle_vars);
            for (0..n_cycle_vars) |i| {
                const src_idx = instr_start + i;
                const dest_idx = n_cycle_vars - 1 - i;
                r_reduction_be[dest_idx] = stage2_challenges[src_idx];
            }

            // ============================================================
            // Extract r_address and r_cycle from Stage 2 RWC phases
            // ============================================================
            const phase1 = rw_phase1;
            const phase2 = rw_phase2;
            const phase3_cycle_len = n_cycle_vars - phase1;
            const phase3_address_len = log_ram_k - phase2;

            var r_address_be = try self.allocator.alloc(F, log_ram_k);
            defer self.allocator.free(r_address_be);
            @memset(r_address_be, F.zero());

            const phase2_start = phase1;
            for (0..phase2) |i| {
                const src_idx = phase2_start + i;
                if (src_idx < stage2_challenges.len) {
                    const dest_idx = phase3_address_len + (phase2 - 1 - i);
                    if (dest_idx < log_ram_k) {
                        r_address_be[dest_idx] = stage2_challenges[src_idx];
                    }
                }
            }
            const phase3_addr_start = phase1 + phase2 + phase3_cycle_len;
            for (0..phase3_address_len) |i| {
                const src_idx = phase3_addr_start + i;
                if (src_idx < stage2_challenges.len) {
                    const dest_idx = phase3_address_len - 1 - i;
                    r_address_be[dest_idx] = stage2_challenges[src_idx];
                }
            }

            var r_cycle_be = try self.allocator.alloc(F, n_cycle_vars);
            defer self.allocator.free(r_cycle_be);
            @memset(r_cycle_be, F.zero());

            for (0..phase1) |i| {
                if (i < stage2_challenges.len) {
                    const dest_idx = phase3_cycle_len + (phase1 - 1 - i);
                    if (dest_idx < n_cycle_vars) {
                        r_cycle_be[dest_idx] = stage2_challenges[i];
                    }
                }
            }
            const phase3_cycle_start = phase1 + phase2;
            for (0..phase3_cycle_len) |i| {
                const src_idx = phase3_cycle_start + i;
                if (src_idx < stage2_challenges.len) {
                    const dest_idx = phase3_cycle_len - 1 - i;
                    r_cycle_be[dest_idx] = stage2_challenges[src_idx];
                }
            }

            var r_cycle_le = try self.allocator.alloc(F, n_cycle_vars);
            defer self.allocator.free(r_cycle_le);
            for (0..n_cycle_vars) |i| {
                r_cycle_le[i] = r_cycle_be[n_cycle_vars - 1 - i];
            }

            var r_address_le = try self.allocator.alloc(F, log_ram_k);
            defer self.allocator.free(r_address_le);
            for (0..log_ram_k) |i| {
                r_address_le[i] = r_address_be[log_ram_k - 1 - i];
            }

            // ============================================================
            // Initialize ValEvaluation prover
            // ============================================================
            const trace_len = trace.steps.items.len;
            const val_eval_params = try ValEvaluationParams.init(
                self.allocator,
                init_eval_for_val_eval,
                trace_len,
                ram_K,
                r_address_le,
                r_cycle_be,
            );
            var val_eval_prover = try ValEvaluationProver.initWithLayoutAndPool(
                self.allocator,
                memory_trace,
                initial_ram,
                val_eval_params,
                start_address,
                memory_layout,
                self.thread_pool,
            );
            val_eval_prover.thread_pool = self.thread_pool;
            defer val_eval_prover.deinit();

            // ============================================================
            // Compute input claims
            // ============================================================
            const input_claim_registers = stage3_claims.rd_write_value
                .add(gamma_stage4.mul(stage3_claims.rs1_value))
                .add(gamma_stage4.mul(gamma_stage4).mul(stage3_claims.rs2_value));

            const input_claim_val_eval = rwc_val_claim.sub(init_eval_for_val_eval);
            const input_claim_val_final = output_val_final_claim.sub(init_eval_for_val_eval);
            const input_claim_ram_val_check = input_claim_val_eval.add(ram_val_check_gamma.mul(input_claim_val_final));

            // ============================================================
            // Transcript: append input claims and derive batching coefficients
            // ============================================================
            transcript.appendScalar("sumcheck_claim", input_claim_registers);
            transcript.appendScalar("sumcheck_claim", input_claim_ram_val_check);

            const batch0 = transcript.challengeScalarFull();
            const batch1 = transcript.challengeScalarFull();
            const batching_coeffs = [2]F{ batch0, batch1 };

            // ============================================================
            // Initialize Registers RWC prover
            // ============================================================
            const stage3_r_cycle_le = stage3_challenges;

            var regs_prover = try Stage4GruenProver.initWithClaims(
                self.allocator,
                trace,
                gamma_stage4,
                stage3_r_cycle_le,
                stage3_claims,
                batch0,
                self.thread_pool,
            );
            regs_prover.thread_pool = self.thread_pool;
            regs_prover.gpu_ops = self.gpu_ops;
            defer regs_prover.deinit();

            // ============================================================
            // Save inc_poly copy for Stage 6
            // ============================================================
            const inc_poly_copy = try self.allocator.alloc(F, @as(usize, 1) << @intCast(n_cycle_vars));
            @memcpy(inc_poly_copy, regs_prover.inc_poly[0..inc_poly_copy.len]);

            // ============================================================
            // Batched sumcheck loop (2 instances)
            // ============================================================
            const ram_val_check_rounds = val_eval_prover.numRounds();
            const rounds_per_instance = [2]usize{ stage4_max_rounds, ram_val_check_rounds };

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

            var batched_claim = F.zero();
            for (0..2) |i| {
                batched_claim = batched_claim.add(individual_claims[i].mul(batching_coeffs[i]));
            }

            var regs_current_claim = individual_claims[0];
            var stage4_r_sumcheck = try self.allocator.alloc(F, stage4_max_rounds);
            defer self.allocator.free(stage4_r_sumcheck);

            for (0..stage4_max_rounds) |round_idx| {
                var combined_evals = [4]F{ F.zero(), F.zero(), F.zero(), F.zero() };

                // Instance 0: RegistersRWC (always active)
                const regs_evals = regs_prover.computeRoundEvals(round_idx, regs_current_claim);
                for (0..4) |j| {
                    combined_evals[j] = combined_evals[j].add(regs_evals[j].mul(batching_coeffs[0]));
                }

                // Instance 1: Combined RamValCheck = val_eval + gamma * val_final
                const ram_val_check_offset = stage4_max_rounds - ram_val_check_rounds;
                const two_inv_local = UniPoly.INV2;

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
                    var ram_evals = val_eval_prover.computeRoundPolynomialCombined(ram_val_check_gamma);
                    // Apply hint: p(0) = claim - p(1)
                    ram_evals[0] = individual_claims[1].sub(ram_evals[1]);

                    ram_val_check_evals_opt = ram_evals;
                    for (0..4) |j| {
                        combined_evals[j] = combined_evals[j].add(ram_evals[j].mul(batching_coeffs[1]));
                    }
                }

                // Convert Toom-Cook evals to coefficients and append to proof
                const full_coeffs = UniPoly.toomCookToCoeffs(combined_evals);
                var coeffs = [_]F{ full_coeffs[0], full_coeffs[1], full_coeffs[2], full_coeffs[3] };
                const coeffs_slice: []const F = &coeffs;
                try proof.addRoundPoly(coeffs_slice);

                // Compressed coefficients [c0, c2, c3] for transcript
                const compressed = UniPoly.toomCookToCompressed(combined_evals);
                transcript.appendScalars("sumcheck_poly", compressed[0..3]);

                const challenge = transcript.challengeScalar();
                stage4_r_sumcheck[round_idx] = challenge;
                batched_claim = UniPoly.evalFromHint(compressed, batched_claim, challenge);

                // Update instance 0 (regs): always active
                regs_current_claim = UniPoly.evaluateToomCookAt(regs_evals, challenge);
                individual_claims[0] = regs_current_claim;
                regs_prover.bindChallenge(round_idx, challenge);

                // Update instance 1 (combined RamValCheck)
                if (ram_val_check_evals_opt) |ram_evals| {
                    individual_claims[1] = UniPoly.evaluateToomCookAt(ram_evals, challenge);
                    val_eval_prover.bindChallengeWithPoly(challenge, ram_evals);
                } else {
                    individual_claims[1] = individual_claims[1].mul(two_inv_local);
                }
            }

            // ============================================================
            // Extract final claims
            // ============================================================
            const regs_claims = regs_prover.getFinalClaims();
            const val_eval_openings = val_eval_prover.getFinalOpenings();

            // ============================================================
            // Append cache_openings to transcript (7 claims)
            // ============================================================
            transcript.appendScalar("opening_claim", regs_claims.val_claim);
            transcript.appendScalar("opening_claim", regs_claims.rs1_ra_claim);
            transcript.appendScalar("opening_claim", regs_claims.rs2_ra_claim);
            transcript.appendScalar("opening_claim", regs_claims.rd_wa_claim);
            transcript.appendScalar("opening_claim", regs_claims.inc_claim);
            transcript.appendScalar("opening_claim", val_eval_openings.wa_eval);
            transcript.appendScalar("opening_claim", val_eval_openings.inc_eval);

            // ============================================================
            // Derive opening points from sumcheck challenges
            // ============================================================
            const regs_log_k: usize = LOG_REGISTERS;

            var regs_r_address = try self.allocator.alloc(F, regs_log_k);
            var regs_r_cycle = try self.allocator.alloc(F, n_cycle_vars);

            // r_address = reverse(phase2 challenges)
            for (0..regs_log_k) |i| {
                regs_r_address[i] = stage4_r_sumcheck[n_cycle_vars + (regs_log_k - 1 - i)];
            }
            // r_cycle = reverse(phase1 challenges)
            for (0..n_cycle_vars) |i| {
                regs_r_cycle[i] = stage4_r_sumcheck[n_cycle_vars - 1 - i];
            }

            // r_cycle_val for RamRaClaimReduction (ValEvaluation starts at round 7)
            var r_cycle_val = try self.allocator.alloc(F, n_cycle_vars);
            const val_eval_start: usize = LOG_REGISTERS;
            for (0..n_cycle_vars) |i| {
                const src_idx = val_eval_start + i;
                if (src_idx < stage4_r_sumcheck.len) {
                    r_cycle_val[n_cycle_vars - 1 - i] = stage4_r_sumcheck[src_idx];
                } else {
                    r_cycle_val[n_cycle_vars - 1 - i] = F.zero();
                }
            }

            return Stage4Result(F){
                .regs_val_claim = regs_claims.val_claim,
                .regs_rs1_ra_claim = regs_claims.rs1_ra_claim,
                .regs_rs2_ra_claim = regs_claims.rs2_ra_claim,
                .regs_rd_wa_claim = regs_claims.rd_wa_claim,
                .regs_inc_claim = regs_claims.inc_claim,
                .val_eval_wa_claim = val_eval_openings.wa_eval,
                .val_eval_inc_claim = val_eval_openings.inc_eval,
                .regs_r_address = regs_r_address,
                .regs_r_cycle = regs_r_cycle,
                .r_cycle_val = r_cycle_val,
                .r_reduction_be = r_reduction_be,
                .inc_poly_copy = inc_poly_copy,
                .allocator = self.allocator,
            };
        }
    };
}
