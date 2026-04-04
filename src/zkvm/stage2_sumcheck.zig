//! Stage 2 Batched Sumcheck — extracted from jolt_prover.zig
//!
//! Batches 5 sumcheck instances for Stage 2:
//!   1. ProductVirtualRemainder:  n_cycle_vars rounds, degree 3
//!   2. RamRafEvaluation:        log_ram_k rounds, degree 2
//!   3. RamReadWriteChecking:    log_ram_k + n_cycle_vars rounds, degree 3
//!   4. OutputSumcheck:          log_ram_k rounds, degree 3
//!   5. InstructionLookupsClaimReduction: n_cycle_vars rounds, degree 2

const std = @import("std");
const Allocator = std.mem.Allocator;
const ThreadPool = @import("zolt_pool").ThreadPool;

const jolt_types = @import("jolt_types.zig");
const field_mod = @import("zolt_arith").field;
const r1cs = @import("r1cs/mod.zig");
const product_remainder = @import("spartan/product_remainder.zig");
const transcripts = @import("zolt_arith").transcripts;
const Blake2bTranscript = transcripts.Blake2bTranscript;
const poly_mod = @import("zolt_arith").poly;
const jolt_device = @import("jolt_device.zig");
const ram = @import("ram/mod.zig");
const claim_reductions = @import("claim_reductions/mod.zig");
const jolt_prover = @import("jolt_prover.zig");

/// Generic Stage 2 sumcheck namespace, parameterised on the field.
pub fn Stage2Sumcheck(comptime F: type) type {
    return struct {
        const Self = @This();

        // Re-export type aliases used by callers.
        const SumcheckInstanceProof = jolt_types.SumcheckInstanceProof;
        const OpeningClaims = jolt_types.OpeningClaims;
        const OpeningId = jolt_types.OpeningId;
        const VirtualPolynomial = jolt_types.VirtualPolynomial;

        const gpu_mod = @import("zolt_arith").gpu;
        const JoltProverConfig = jolt_prover.JoltProverConfig;

        // -----------------------------------------------------------------
        // Stage2Result — return type of generateBatchedSumcheckProof
        // -----------------------------------------------------------------

        pub const Stage2Result = struct {
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

        // -----------------------------------------------------------------
        // Context — replaces the old `self: *JoltProver` receiver
        // -----------------------------------------------------------------

        pub const Context = struct {
            allocator: Allocator,
            thread_pool: ?*ThreadPool = null,
            gpu_ops: ?*gpu_mod.GpuPolyOps = null,
        };

        // -----------------------------------------------------------------
        // generateBatchedSumcheckProof
        // -----------------------------------------------------------------

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
        pub fn generateBatchedSumcheckProof(
            ctx: Context,
            proof: *SumcheckInstanceProof(F),
            transcript: *Blake2bTranscript(F),
            r0_stage2: F,
            uni_skip_claim_stage2: F,
            tau: []const F,
            r_spartan_for_instr: []const F,
            raw_r1cs_inputs: []const @import("r1cs/evaluators.zig").RawR1CSInputs,
            n_cycle_vars: usize,
            log_ram_k: usize,
            opening_claims: *OpeningClaims(F),
            config: JoltProverConfig,
        ) !Stage2Result {
            const allocator = ctx.allocator;
            const max_num_rounds = log_ram_k + n_cycle_vars;

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


            // Sample gammas from transcript in the same order as upstream Jolt verifier:
            // 1. RamReadWriteChecking samples gamma first
            // 2. InstructionLookupsClaimReduction samples gamma
            // 3. OutputSumcheck samples r_address
            //
            // CRITICAL: gamma uses challenge_scalar (NO 125-bit masking) = challengeScalarFull()
            // r_address uses challenge_scalar_optimized (HAS 125-bit masking) = challengeScalar()

            // 1. RamReadWriteChecking gamma
            const gamma_rwc = transcript.challengeScalarFull();

            // 2. InstructionLookupsClaimReduction gamma (via challenge_scalar, NO masking)
            const gamma_instr = transcript.challengeScalarFull();
            const gamma_instr_sqr = gamma_instr.mul(gamma_instr);
            const gamma_instr_cub = gamma_instr_sqr.mul(gamma_instr);
            const gamma_instr_quart = gamma_instr_sqr.mul(gamma_instr_sqr);

            // 3. OutputSumcheck samples r_address (log_ram_k challenges via challenge_vector_optimized)
            const r_address_presampled = try allocator.alloc(F, log_ram_k);
            defer allocator.free(r_address_presampled);
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


            // Step 2: Sample batching coefficients (input claims already appended at line 1747)
            var batching_coeffs: [5]F = undefined;
            for (0..5) |i| {
                batching_coeffs[i] = transcript.challengeScalarFull();
            }

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


            // Initialize provers for each instance (upstream ordering):
            // [0] RamReadWriteChecking, [1] ProductVirtualRemainder,
            // [2] InstructionLookupsClaimReduction, [3] RamRafEvaluation, [4] OutputSumcheck
            const s2_fn_t0 = if (config.bench_output) std.time.nanoTimestamp() else 0;

            // Instance 0: RamReadWriteChecking - starts at round 0 (max rounds)
            const RWCProver = ram.RamReadWriteCheckingProver(F);
            var rwc_prover: ?RWCProver = null;
            var rwc_evals_this_round: ?[4]F = null;
            const use_rwc_prover = !input_claims[0].eql(F.zero());
            if (config.memory_trace != null and use_rwc_prover) {
                const phase1_num_rounds = n_cycle_vars;
                var rwc_params = ram.RamReadWriteCheckingParams(F).initWithPhaseConfig(
                    allocator,
                    gamma_rwc,
                    tau[0..n_cycle_vars],
                    log_ram_k,
                    n_cycle_vars,
                    phase1_num_rounds,
                    if (config.memory_layout) |ml| ml.getLowestAddress() else 0x80000000,
                ) catch null;

                if (rwc_params) |*params| {
                    rwc_prover = RWCProver.init(
                        allocator,
                        config.memory_trace.?,
                        params.*,
                        input_claims[0],
                        config.initial_ram,
                        config.memory_layout,
                        config.is_panicking,
                    ) catch null;

                    if (rwc_prover != null) {
                        rwc_prover.?.thread_pool = ctx.thread_pool;
                    } else {
                        params.deinit();
                    }
                }
            }
            defer if (rwc_prover) |*rp| rp.deinit();

            // Instance 1: ProductVirtualRemainder
            const ProductRemainderProver = product_remainder.ProductVirtualRemainderProver(F);
            var product_prover: ?ProductRemainderProver = null;
            if (raw_r1cs_inputs.len > 0 and tau.len > 0) {
                product_prover = ProductRemainderProver.init(
                    allocator,
                    r0_stage2,
                    tau,
                    uni_skip_claim_stage2,
                    raw_r1cs_inputs,
                    ctx.thread_pool,
                ) catch null;
            }
            if (product_prover != null) {
                product_prover.?.thread_pool = ctx.thread_pool;
                product_prover.?.gpu_ops = ctx.gpu_ops;
                product_prover.?.initGpuShadows();
            }
            defer if (product_prover) |*p| p.deinit();

            // Instance 2: InstructionLookupsClaimReduction - initialized lazily at start_round
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
                    allocator,
                    config.initial_ram.?,
                    config.final_ram.?,
                    r_address_presampled,
                    config.memory_layout.?,
                    config.program_inputs,
                    config.program_outputs,
                    config.is_panicking,
                ) catch null;
                if (output_prover != null) {
                    output_prover.?.thread_pool = ctx.thread_pool;
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
            defer challenges.deinit(allocator);

            // Per-instance timing accumulators (bench only)
            const bench_s2 = config.bench_output;
            var inst_compute_ns: [5]u64 = .{ 0, 0, 0, 0, 0 };
            var inst_bind_ns: [5]u64 = .{ 0, 0, 0, 0, 0 };
            var inst_active_rounds: [5]u64 = .{ 0, 0, 0, 0, 0 };

            const s2_init_done_t = if (config.bench_output) std.time.nanoTimestamp() else 0;

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
                    const inst_t0 = if (bench_s2) std.time.nanoTimestamp() else 0;

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
                            if (round_idx == start_round and instr_prover == null and raw_r1cs_inputs.len > 0) {
                                var instr_params = claim_reductions.InstructionLookupsParams(F).init(
                                    allocator,
                                    gamma_instr,
                                    r_spartan_for_instr,
                                    n_cycle_vars,
                                ) catch null;

                                if (instr_params) |*params| {
                                    instr_prover = InstrLookupsProver.init(
                                        allocator,
                                        params.*,
                                        input_claims[2],
                                        raw_r1cs_inputs,
                                        ctx.thread_pool,
                                    ) catch blk: {
                                        params.deinit();
                                        break :blk null;
                                    };
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
                                const r_cycle = try allocator.alloc(F, n_cycle_vars);
                                @memcpy(r_cycle, r_cycle_slice);
                                const start_addr: u64 = if (config.memory_layout) |ml| ml.getLowestAddress() else 0x80000000;
                                var raf_params = try ram.RafEvaluationParams(F).init(allocator, log_ram_k, start_addr, r_cycle);
                                allocator.free(r_cycle);

                                const raf_initial_claim = input_claims[3];

                                raf_prover = RafProver.init(allocator, config.memory_trace.?, raf_params, raf_initial_claim) catch blk: {
                                    raf_params.deinit();
                                    break :blk null;
                                };
                                if (raf_prover != null) raf_prover.?.thread_pool = ctx.thread_pool;
                            }

                            if (raf_prover) |*rp| {
                                const raf_evals = rp.computeRoundPolynomialCubic();
                                raf_evals_this_round = raf_evals;
                                for (0..4) |j| {
                                    combined_evals[j] = combined_evals[j].add(raf_evals[j].mul(batching_coeffs[i]));
                                }
                            } else {
                                if (round_idx == start_round) {
                                }
                                const remaining_rounds = rounds_per_instance[i] - (round_idx - start_round);
                                var scaled = individual_claims[i];
                                for (0..remaining_rounds) |_| scaled = scaled.mul(F.fromU64(2));
                                scaled = scaled.mul(poly_mod.UniPoly(F).INV2);
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
                                // Zero input claim -> zero polynomial
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

                    if (bench_s2) {
                        const inst_t1 = std.time.nanoTimestamp();
                        inst_compute_ns[i] += @intCast(@as(i128, inst_t1 - inst_t0));
                        if (round_idx >= start_round) inst_active_rounds[i] += 1;
                    }
                }

                // Convert to compressed coefficients [c0, c2, c3]
                const compressed = poly_mod.UniPoly(F).evalsToCompressed(combined_evals);

                if (round_idx == 0 or round_idx == 16 or round_idx == max_num_rounds - 1) {
                }

                // Append to proof
                const coeffs = try allocator.alloc(F, 3);
                coeffs[0] = compressed[0];
                coeffs[1] = compressed[1];
                coeffs[2] = compressed[2];
                try proof.compressed_polys.append(allocator, .{
                    .coeffs_except_linear_term = coeffs,
                    .allocator = allocator,
                });

                // Append to transcript: sumcheck polynomial coefficients
                transcript.appendScalars("sumcheck_poly", compressed[0..3]);

                // Sample round challenge
                const challenge = transcript.challengeScalar();
                try challenges.append(allocator, challenge);

                // Update batched claim by evaluating at challenge
                // CRITICAL: Must use evalFromHint (same as Jolt's verifier) to ensure
                // the claim evolution matches. Using Lagrange interpolation from combined_evals
                // would give different results because the evaluations may not be consistent
                // with what Jolt expects (different s1, s2, s3 can produce the same c0, c2, c3).
                const old_claim = batched_claim;
                batched_claim = evalFromHint(compressed, old_claim, challenge);


                // Debug: Check claim trajectory for first few and last few rounds
                if (round_idx < 3 or round_idx >= max_num_rounds - 5) {
                    // Check: s(0) + s(1) should equal old_claim for soundness
                    const sum_check = combined_evals[0].add(combined_evals[1]);
                    if (!sum_check.eql(old_claim)) {
                        // Print individual instance contributions
                        if (product_evals_this_round) |_| {
                            // Note: pp.current_claim is ALREADY UPDATED for next round at this point!
                        } else {
                        }
                        if (raf_evals_this_round) |_| {
                        } else {
                        }
                        if (rwc_evals_this_round) |_| {
                        } else {
                        }
                        if (output_evals_this_round) |_| {
                        } else {
                        }
                        if (instr_evals_this_round) |_| {
                        } else {
                        }
                    }
                }

                // Bind challenge in all active instances and update their claims
                // Instance 0: RWC (starts at round 0)
                if (rwc_prover) |*rwcp| {
                    const bt0 = if (bench_s2) std.time.nanoTimestamp() else 0;
                    if (rwc_evals_this_round) |evals| rwcp.updateClaim(evals, challenge);
                    rwcp.bindChallenge(challenge) catch {};
                    if (bench_s2) inst_bind_ns[0] += @intCast(@as(i128, std.time.nanoTimestamp() - bt0));
                }

                // Instance 1: ProductVirtualRemainder (starts at max_rounds - n_cycle_vars)
                if (product_prover != null and round_idx >= (max_num_rounds - n_cycle_vars)) {
                    const bt1 = if (bench_s2) std.time.nanoTimestamp() else 0;
                    if (product_evals_this_round) |evals| product_prover.?.updateClaim(evals, challenge);
                    product_prover.?.bindChallenge(challenge) catch {};
                    if (bench_s2) inst_bind_ns[1] += @intCast(@as(i128, std.time.nanoTimestamp() - bt1));
                }

                // Instance 2: InstructionLookups (starts at max_rounds - n_cycle_vars)
                if (instr_prover != null and round_idx >= (max_num_rounds - n_cycle_vars)) {
                    const bt2 = if (bench_s2) std.time.nanoTimestamp() else 0;
                    if (instr_evals_this_round) |evals| instr_prover.?.updateClaim(evals, challenge);
                    instr_prover.?.bindChallenge(challenge) catch {};
                    if (bench_s2) inst_bind_ns[2] += @intCast(@as(i128, std.time.nanoTimestamp() - bt2));
                }

                // Instance 3: RAF (starts at max_rounds - log_ram_k)
                if (raf_prover != null and round_idx >= (max_num_rounds - log_ram_k)) {
                    const bt3 = if (bench_s2) std.time.nanoTimestamp() else 0;
                    if (raf_evals_this_round) |evals| raf_prover.?.updateClaim(evals, challenge);
                    raf_prover.?.bindChallenge(challenge) catch {};
                    if (bench_s2) inst_bind_ns[3] += @intCast(@as(i128, std.time.nanoTimestamp() - bt3));
                }

                // Instance 4: OutputSumcheck (starts at max_rounds - log_ram_k)
                if (output_prover != null and round_idx >= (max_num_rounds - log_ram_k)) {
                    const bt4 = if (bench_s2) std.time.nanoTimestamp() else 0;
                    if (output_evals_this_round) |evals| output_prover.?.updateClaim(evals, challenge);
                    output_prover.?.bindChallenge(challenge);
                    if (bench_s2) inst_bind_ns[4] += @intCast(@as(i128, std.time.nanoTimestamp() - bt4));
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
                }
            }

            const s2_loop_done_t = if (config.bench_output) std.time.nanoTimestamp() else 0;

            // Print per-instance timing breakdown
            if (bench_s2) {
                const names = [5][]const u8{ "RWC", "Product", "InstrLookups", "RAF", "Output" };
                const ms = 1_000_000.0;
                for (0..5) |bi| {
                    std.debug.print("[BENCH] stage=2 instance={d}({s}) compute={d:.1}ms bind={d:.1}ms rounds={d}\n", .{
                        bi,
                        names[bi],
                        @as(f64, @floatFromInt(inst_compute_ns[bi])) / ms,
                        @as(f64, @floatFromInt(inst_bind_ns[bi])) / ms,
                        inst_active_rounds[bi],
                    });
                }
            }

            // Print prover's per-instance final claims for comparison with verifier
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


            // Debug: Print all challenges in LE format for comparison with Jolt

            // Debug: Print prover's final left/right values
            if (product_prover) |_| {
            }

            // Compute the 8 factor polynomial evaluations at r_cycle
            // r_cycle is the last n_cycle_vars challenges from Stage 2
            // ProductVirtualRemainder starts at round log_ram_k, so its r_cycle
            // is challenges[log_ram_k..max_num_rounds]
            const factor_evals = try computeProductFactorEvaluations(
                ctx,
                raw_r1cs_inputs,
                challenges.items,
                n_cycle_vars,
                log_ram_k,
            );

            // Debug: Compute fused_left and fused_right from factor_evals and compare
            // Lagrange weights at r0_stage2
            const LagrangePoly = r1cs.univariate_skip.LagrangePolynomial(F);
            const w = try LagrangePoly.evals(3, r0_stage2, allocator);
            defer allocator.free(w);

            // fused_left = w[0]*l_inst + w[1]*lookup_out + w[2]*j_flag
            // fused_right = w[0]*r_inst + w[1]*branch_flag + w[2]*(1 - next_is_noop)


            // Compute tau_high_bound_r0 and tau_bound_r_tail_rev for expected_output_claim debug
            // tau_high_bound_r0 = LagrangeKernel(5, tau_high, r0)

            // tau_bound_r_tail_rev = eq(tau_low, r_cycle_reversed)
            // tau_low = tau[0..n_cycle_vars]
            // r_cycle_reversed = last n_cycle_vars challenges, reversed
            // The challenges.items are the Stage 2 sumcheck challenges
            // ProductVirtualRemainder starts at round (max_num_rounds - n_cycle_vars)
            // Its challenges are the LAST n_cycle_vars of challenges.items
            const product_start_round = max_num_rounds - n_cycle_vars;

            // Extract ProductVirtualRemainder challenges (last n_cycle_vars)
            var product_challenges = try allocator.alloc(F, n_cycle_vars);
            defer allocator.free(product_challenges);
            for (0..n_cycle_vars) |i| {
                if (product_start_round + i < challenges.items.len) {
                    product_challenges[i] = challenges.items[product_start_round + i];
                } else {
                    product_challenges[i] = F.zero();
                }
            }

            // Reverse the product challenges (r_cycle_reversed)
            var r_cycle_reversed = try allocator.alloc(F, n_cycle_vars);
            defer allocator.free(r_cycle_reversed);
            for (0..n_cycle_vars) |i| {
                r_cycle_reversed[i] = product_challenges[n_cycle_vars - 1 - i];
            }

            // Compute eq(tau_low, r_cycle_reversed)


            // Compute expected_output_claim


            // Copy challenges to return them
            const challenges_copy = try allocator.alloc(F, challenges.items.len);
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
            if (rwc_prover) |*rp| {
                const rwc_opening_claims = rp.getOpeningClaims(challenges.items);
                rwc_ra_claim = rwc_opening_claims.ra_claim;
                rwc_val_claim = rwc_opening_claims.val_claim;
                rwc_inc_claim = rwc_opening_claims.inc_claim;

                // Verify: current_claim should equal eq_cycle * ra * (val + gamma * (val + inc))
            } else {
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
                    var r_addr_be = try allocator.alloc(F, log_ram_k);
                    defer allocator.free(r_addr_be);
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

                    // Compute val_init(r_address_be) using bytecode_words (like Jolt does)
                    rwc_val_claim = jolt_prover.JoltProver(F).computeInitialRamEval(
                        config.bytecode_words,
                        config.min_bytecode_address,
                        config.memory_layout.?,
                        r_addr_be,
                        log_ram_k,
                        config.program_inputs,
                    );
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
            }


            // Get Val_final(r') and Val_init(r') from the OutputSumcheck prover
            // These are the MLE evaluations at the final opening point
            var output_val_final = F.zero();
            var output_val_init = F.zero();
            if (output_prover) |op| {
                const output_claims_val = op.getFinalClaims();
                output_val_final = output_claims_val.val_final;
                output_val_init = output_claims_val.val_init;
            }

            // Compute r_address_rw and r_cycle_rw from RWC challenges for RamRaClaimReduction
            // RWC uses 3-phase structure:
            // - Phase 1: n_cycle_vars cycle variables
            // - Phase 2: log_ram_k address variables
            // - Phase 3: 0 remaining variables (for default config)
            // Opening point = [r_address, r_cycle] in BIG_ENDIAN
            const phase1_rounds = n_cycle_vars;
            const phase2_rounds = log_ram_k;

            const r_address_rw = try allocator.alloc(F, log_ram_k);
            const r_cycle_rw = try allocator.alloc(F, n_cycle_vars);

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
            const r_address_raf = try allocator.alloc(F, log_ram_k);
            for (0..phase2_rounds) |i| {
                const src_idx = phase1_rounds + i;
                if (src_idx < challenges.items.len) {
                    r_address_raf[phase2_rounds - 1 - i] = challenges.items[src_idx];
                } else {
                    r_address_raf[phase2_rounds - 1 - i] = F.zero();
                }
            }

            // Debug: compare r_address_raf and r_address_rw (they should now be identical)

            // Compute ProductVirtualRemainder r_cycle from Stage 2 challenges
            // ProductVirtualRemainder starts at round (max_num_rounds - n_cycle_vars)
            // and runs for n_cycle_vars rounds. Reversed to BIG_ENDIAN.
            const product_start = max_num_rounds - n_cycle_vars;
            const r_cycle_product = try allocator.alloc(F, n_cycle_vars);
            for (0..n_cycle_vars) |i| {
                const src_idx = product_start + i;
                if (src_idx < challenges.items.len) {
                    r_cycle_product[n_cycle_vars - 1 - i] = challenges.items[src_idx];
                } else {
                    r_cycle_product[n_cycle_vars - 1 - i] = F.zero();
                }
            }

            if (config.bench_output) {
                const ms2 = 1_000_000.0;
                const t_now = std.time.nanoTimestamp();
                std.debug.print("[BENCH] stage=2 breakdown: prover_init={d:.1}ms round_loop={d:.1}ms post_loop={d:.1}ms\n", .{
                    @as(f64, @floatFromInt(@as(i128, s2_init_done_t - s2_fn_t0))) / ms2,
                    @as(f64, @floatFromInt(@as(i128, s2_loop_done_t - s2_init_done_t))) / ms2,
                    @as(f64, @floatFromInt(@as(i128, t_now - s2_loop_done_t))) / ms2,
                });
            }

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
                .allocator = allocator,
            };
        }

        // -----------------------------------------------------------------
        // computeProductFactorEvaluations
        // -----------------------------------------------------------------

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
        pub fn computeProductFactorEvaluations(
            ctx: Context,
            raw_r1cs_inputs: []const @import("r1cs/evaluators.zig").RawR1CSInputs,
            all_challenges: []const F,
            n_cycle_vars: usize,
            log_ram_k: usize,
        ) ![8]F {
            _ = log_ram_k;
            const allocator = ctx.allocator;
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
            const r_cycle_buf = try allocator.alloc(F, n_cycle_vars);
            defer allocator.free(r_cycle_buf);
            for (0..n_cycle_vars) |i| {
                r_cycle_buf[i] = r_cycle_original[n_cycle_vars - 1 - i];
            }

            // Compute eq polynomial evaluations at r_cycle (using BIG_ENDIAN indexing like Jolt)
            const EqPoly = poly_mod.EqPolynomial(F);
            const eq_evals = try EqPoly.evalsSliceWithScaling(F, allocator, r_cycle_buf, null);
            defer allocator.free(eq_evals);

            // Compute MLE evaluation: Σ_t eq(r_cycle, t) * factor_value[t]
            // Parallelized with UnreducedProductAccum for deferred Montgomery reduction.
            const num_cycles = @min(eq_evals.len, raw_r1cs_inputs.len);

            const FactorCtx = struct {
                eq: []const F,
                raw: []const @import("r1cs/evaluators.zig").RawR1CSInputs,
            };
            const fctx = FactorCtx{
                .eq = eq_evals,
                .raw = raw_r1cs_inputs,
            };

            // Use typed integer accumulators matching Jolt's compute_claimed_factors:
            // u64 factors -> fmaddU64 (4 integer muls), bool factors -> fmaddBool (0 muls),
            // i128 factors -> fmaddI128 (8 integer muls). ~4x fewer ops than field*field.
            const MedAccumS = field_mod.MedAccumS;
            const factorMapFn = struct {
                fn f(c: FactorCtx, start: usize, end: usize) [8]MedAccumS {
                    @setEvalBranchQuota(10000);
                    var acc: [8]MedAccumS = .{MedAccumS.zero()} ** 8;
                    for (start..end) |t| {
                        const eq_val = c.eq[t];
                        const raw = &c.raw[t];

                        // Factor 0: LeftInstructionInput (u64)
                        acc[0].fmaddU64(eq_val, raw.u64_values[0]);
                        // Factor 1: RightInstructionInput (i128)
                        acc[1].fmaddI128(eq_val, raw.signed_values[0]);
                        // Factor 2: FlagJump (bool)
                        acc[2].fmaddBool(eq_val, raw.bool_flags[9]);
                        // Factor 3: FlagWriteLookupOutputToRD (bool)
                        acc[3].fmaddBool(eq_val, raw.bool_flags[10]);
                        // Factor 4: LookupOutput (u64)
                        acc[4].fmaddU64(eq_val, raw.u64_values[12]);
                        // Factor 5: FlagBranch (bool)
                        acc[5].fmaddBool(eq_val, raw.bool_flags[19]);
                        // Factor 6: NextIsNoop (bool from next cycle)
                        const next_is_noop: bool = if (t + 1 < c.raw.len)
                            c.raw[t + 1].bool_flags[20]
                        else
                            true; // Last cycle: NextIsNoop = true
                        acc[6].fmaddBool(eq_val, next_is_noop);
                        // Factor 7: FlagVirtualInstruction (bool)
                        acc[7].fmaddBool(eq_val, raw.bool_flags[11]);
                    }
                    return acc;
                }
            }.f;

            const factorReduceFn = struct {
                fn f(a: [8]MedAccumS, b: [8]MedAccumS) [8]MedAccumS {
                    var result: [8]MedAccumS = undefined;
                    inline for (0..8) |fi| {
                        result[fi] = a[fi];
                        result[fi].pos.addAssign(b[fi].pos);
                        result[fi].neg.addAssign(b[fi].neg);
                    }
                    return result;
                }
            }.f;

            const identity: [8]MedAccumS = .{MedAccumS.zero()} ** 8;
            const accum = if (ctx.thread_pool) |tp|
                tp.parallelReduce([8]MedAccumS, num_cycles, identity, fctx, factorMapFn, factorReduceFn)
            else
                factorMapFn(fctx, 0, num_cycles);

            var factor_evals = [8]F{ F.zero(), F.zero(), F.zero(), F.zero(), F.zero(), F.zero(), F.zero(), F.zero() };
            inline for (0..8) |fi| {
                factor_evals[fi] = accum[fi].barrettReduce();
            }

            // Handle padding cycles (only NextIsNoop=1 for NoOp padding)
            if (raw_r1cs_inputs.len < eq_evals.len) {
                for (raw_r1cs_inputs.len..eq_evals.len) |t| {
                    factor_evals[6] = factor_evals[6].add(eq_evals[t]);
                }
            }

            return factor_evals;
        }

        // -----------------------------------------------------------------
        // Private helpers
        // -----------------------------------------------------------------

        /// Evaluate polynomial at challenge using Jolt's eval_from_hint formula.
        /// Delegates to the shared UniPoly implementation.
        fn evalFromHint(compressed: [3]F, hint: F, x: F) F {
            return poly_mod.UniPoly(F).evalFromHint(compressed, hint, x);
        }
    };
}

// =========================================================================
// Standalone helpers (outside the generic struct)
// =========================================================================

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
pub fn extractProductFactors(
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
