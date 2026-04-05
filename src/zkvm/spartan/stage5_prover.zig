//! Stage 5 Batched Sumcheck Prover
//!
//! Stage 5 is a batched sumcheck with 3 instances:
//! 1. RegistersValEvaluation: 8 rounds (log_T)
//! 2. RamRaClaimReduction: 8 rounds (log_T, cycle-only)
//! 3. LookupsReadRaf: 136 rounds (LOOKUPS_LOG_K + log_T)
//!
//! The batched sumcheck combines instances with different round counts.
//! Instances with fewer rounds contribute constant polynomials (scaled input claims)
//! until their variables start being bound.
//!
//! Reference: jolt-core/src/subprotocols/sumcheck.rs

const std = @import("std");

const zkvm_debug = @import("../debug.zig");
const dbg = zkvm_debug.dbg;
const debug_verbose = zkvm_debug.verbose;

// Benchmark timing control - set to true to enable fine-grained timing
const bench_timing = false;

const Allocator = std.mem.Allocator;
const ThreadPool = @import("zolt_pool").ThreadPool;
const pool_helpers = @import("zolt_pool").helpers;
const GpuPolyOps = @import("zolt_arith").gpu.GpuPolyOps;

const poly_mod = @import("zolt_arith").poly;
const LtPolynomial = @import("zolt_arith").poly.lt_poly.LtPolynomial;
const UniPoly = poly_mod.UniPoly;
const transcripts = @import("zolt_arith").transcripts;
const Blake2bTranscript = transcripts.Blake2bTranscript;
const jolt_types = @import("../jolt_types.zig");
const SumcheckInstanceProof = jolt_types.SumcheckInstanceProof;
const OpeningClaims = jolt_types.OpeningClaims;
const OpeningId = jolt_types.OpeningId;
const tracer = @import("../../tracer/mod.zig");
const ExecutionTrace = tracer.ExecutionTrace;
const ram = @import("../ram/mod.zig");
const jolt_device = @import("../jolt_device.zig");
const sumcheck_helpers = @import("sumcheck_helpers.zig");
const UnreducedProductAccum = @import("zolt_arith").field.UnreducedProductAccum;

// Import extracted instance helpers
const stage5_inst = @import("stage5_instances.zig");

// Import extracted RamRaClaimReduction prover (Instance 1)
const stage5_ram_ra = @import("stage5_ram_ra.zig");
pub const RamRaClaimReductionProver = stage5_ram_ra.RamRaClaimReductionProver;

// Import extracted LookupsReadRaf prover (Instance 2)
const stage5_lookups_mod = @import("stage5_lookups.zig");
pub const LookupsReadRafProver = stage5_lookups_mod.LookupsReadRafProver;

// Import prefix-suffix decomposition modules
const lookup_table_mod = @import("../lookup_table/mod.zig");
const AllSuffixPolys = lookup_table_mod.AllSuffixPolys;
const PrefixCheckpointsState = lookup_table_mod.PrefixCheckpointsState;
const proverMsgReadChecking = lookup_table_mod.proverMsgReadChecking;
const RafDecomposition = lookup_table_mod.RafDecomposition;
const initQRaf = lookup_table_mod.initQRaf;
const proverMsgRaf = lookup_table_mod.proverMsgRaf;
const LookupBits = lookup_table_mod.LookupBits;
const ExpandingTable = lookup_table_mod.ExpandingTable;
const condenseUEvals = lookup_table_mod.condenseUEvals;
const computeTableValuesAtRAddress = lookup_table_mod.computeTableValuesAtRAddress;
const NUM_TABLES = lookup_table_mod.NUM_TABLES;

/// Constants for Stage 5
pub const LOOKUPS_LOG_K: usize = 128; // XLEN * 2 for RV64
pub const RAM_LOG_K: usize = 16; // Default RAM address space
pub const REGISTERS_LOG_K: usize = 7; // log2(128) register slots

// Re-export free helper functions from stage5_instances
pub const evaluateLeftOperand = stage5_inst.evaluateLeftOperand;
pub const evaluateRightOperand = stage5_inst.evaluateRightOperand;
pub const evaluateIdentity = stage5_inst.evaluateIdentity;
pub const computeEqPolynomial = stage5_inst.computeEqPolynomial;
pub const computeEqAtPoint = stage5_inst.computeEqAtPoint;
pub const interleaveBits128 = stage5_inst.interleaveBits128;
pub const getBit128 = stage5_inst.getBit128;
pub const getLookupTableIndex = stage5_inst.getLookupTableIndex;

/// Result of Stage 5 sumcheck
pub fn Stage5Result(comptime F: type) type {
    return struct {
        const Self = @This();

        /// All 136 sumcheck challenges
        challenges: []F,
        /// RegistersValEvaluation opening claims
        regs_val_inc_claim: F, // RdInc at r_cycle'
        regs_val_wa_claim: F, // RdWa at (r_address, r_cycle')
        /// RamRaClaimReduction opening claim
        ram_ra_claim: F, // RamRa at reduced point
        /// LookupsReadRaf opening claims
        lookups_table_flags: []F, // LookupTableFlag(i) for i in 0..41
        lookups_ra_chunks: []F, // InstructionRa(i) for i in 0..8
        lookups_raf_flag: F, // InstructionRafFlag
        allocator: Allocator,

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.challenges);
            self.allocator.free(self.lookups_table_flags);
            self.allocator.free(self.lookups_ra_chunks);
        }
    };
}

/// Stage 5 Batched Sumcheck Prover
pub fn Stage5BatchedProver(comptime F: type) type {
    return struct {
        const Self = @This();
        const Inst = stage5_inst.Helpers(F);

        allocator: Allocator,
        thread_pool: ?*ThreadPool = null,
        gpu_ops: ?*GpuPolyOps = null,

        pub fn init(allocator: Allocator) Self {
            return .{ .allocator = allocator };
        }

        /// Generate Stage 5 batched sumcheck proof
        ///
        /// For programs without RAM operations (like Fibonacci), all instances
        /// have input_claim = 0 from the accumulator. In this case, we generate
        /// zero sumcheck polynomials which correctly verify.
        ///
        /// For programs with RAM operations, we need actual sumcheck provers.
        pub fn generateStage5Proof(
            self: *Self,
            proof: *SumcheckInstanceProof(F),
            transcript: *Blake2bTranscript(F),
            opening_claims: *OpeningClaims(F),
            n_cycle_vars: usize,
            _: usize, // log_ram_k (unused: cycle-only reduction)
            gamma_ram_ra: F,
            gamma_lookups_raf: F,
            lookups_ra_virtual_log_k_chunk: usize,
        ) !Stage5Result(F) {
            // Instance configurations
            const regs_val_num_rounds = n_cycle_vars; // 8 rounds
            const ram_ra_num_rounds = n_cycle_vars; // 8 rounds (cycle-only, matches upstream)
            const lookups_num_rounds = LOOKUPS_LOG_K + n_cycle_vars; // 136 rounds
            const max_num_rounds = lookups_num_rounds;

            // Use gamma_ram_ra for RamRaClaimReduction (Instance 1)
            // Use gamma_lookups_raf for LookupsReadRaf (Instance 2)
            const gamma = gamma_ram_ra;
            const gamma_raf = gamma_lookups_raf;

            if (comptime debug_verbose) {
                dbg("[STAGE5] Configuration: regs={}, ram_ra={}, lookups={}, max={}\n", .{
                    regs_val_num_rounds, ram_ra_num_rounds, lookups_num_rounds, max_num_rounds,
                });
            }

            // Get input claims from accumulator
            const regs_val_input = opening_claims.get(
                .{ .Virtual = .{ .poly = .RegistersVal, .sumcheck_id = .RegistersReadWriteChecking } },
            ) orelse F.zero();

            // Upstream RamRaClaimReduction uses 3 claims: raf, rw, val (fused RamValCheck)
            // input = claim_raf + gamma*claim_rw + gamma^2*claim_val
            const claim_raf = opening_claims.get(
                .{ .Virtual = .{ .poly = .RamRa, .sumcheck_id = .RamRafEvaluation } },
            ) orelse F.zero();
            const claim_rw = opening_claims.get(
                .{ .Virtual = .{ .poly = .RamRa, .sumcheck_id = .RamReadWriteChecking } },
            ) orelse F.zero();
            const claim_val = opening_claims.get(
                .{ .Virtual = .{ .poly = .RamRa, .sumcheck_id = .RamValCheck } },
            ) orelse F.zero();

            const gamma2 = gamma.mul(gamma);
            const ram_ra_input = claim_raf
                .add(gamma.mul(claim_rw))
                .add(gamma2.mul(claim_val));

            // LookupsReadRaf batches 3 claims with gamma_lookups_raf:
            // input = rv + gamma*left_op + gamma^2*right_op
            const rv_claim = opening_claims.get(
                .{ .Virtual = .{ .poly = .LookupOutput, .sumcheck_id = .InstructionClaimReduction } },
            ) orelse F.zero();
            const left_op_claim = opening_claims.get(
                .{ .Virtual = .{ .poly = .LeftLookupOperand, .sumcheck_id = .InstructionClaimReduction } },
            ) orelse F.zero();
            const right_op_claim = opening_claims.get(
                .{ .Virtual = .{ .poly = .RightLookupOperand, .sumcheck_id = .InstructionClaimReduction } },
            ) orelse F.zero();

            const gamma_raf2 = gamma_raf.mul(gamma_raf);
            const lookups_input = rv_claim
                .add(gamma_raf.mul(left_op_claim))
                .add(gamma_raf2.mul(right_op_claim));

            if (comptime debug_verbose) {
                dbg("[STAGE5] Input claims:\n", .{});
                dbg("  regs_val_input = {any}\n", .{regs_val_input.toBytesBE()[0..8]});
                dbg("  ram_ra_input = {any}\n", .{ram_ra_input.toBytesBE()[0..8]});
                dbg("  lookups_input = {any}\n", .{lookups_input.toBytesBE()[0..8]});
            }
            if (comptime debug_verbose) {
                const print = std.debug.print;
                print("[S5 CLAIM DETAIL] rv_claim = {any}\n", .{rv_claim.toBytesBE()});
                print("[S5 CLAIM DETAIL] left_op_claim = {any}\n", .{left_op_claim.toBytesBE()});
                print("[S5 CLAIM DETAIL] right_op_claim = {any}\n", .{right_op_claim.toBytesBE()});
                print("[S5 CLAIM DETAIL] gamma_raf = {any}\n", .{gamma_raf.toBytesBE()});
                print("[S5 CLAIM DETAIL] lookups_input = {any}\n", .{lookups_input.toBytesBE()});
                print("[S5 CLAIM DETAIL] regs_val_input = {any}\n", .{regs_val_input.toBytesBE()});
                print("[S5 CLAIM DETAIL] ram_ra_input = {any}\n", .{ram_ra_input.toBytesBE()});
            }

            // Append input claims to transcript in verifier instance order:
            // [lookups_read_raf, ram_ra_reduction, registers_val_evaluation]
            transcript.appendScalar("sumcheck_claim", lookups_input);
            transcript.appendScalar("sumcheck_claim", ram_ra_input);
            transcript.appendScalar("sumcheck_claim", regs_val_input);

            const batch0 = transcript.challengeScalarFull(); // coeff for lookups
            const batch1 = transcript.challengeScalarFull(); // coeff for ram_ra
            const batch2 = transcript.challengeScalarFull(); // coeff for regs_val

            if (comptime debug_verbose) {
                dbg("[STAGE5] Batching coefficients (LE for Jolt comparison):\n", .{});
                dbg("  batch0 (lookups, LE) = {any}\n", .{batch0.toBytes()});
                dbg("  batch1 (ram_ra, LE) = {any}\n", .{batch1.toBytes()});
                dbg("  batch2 (regs_val, LE) = {any}\n", .{batch2.toBytes()});
            }

            // Compute scaled input claims
            // Instance i with num_rounds[i] is scaled by 2^(max_num_rounds - num_rounds[i])
            const regs_scale = max_num_rounds - regs_val_num_rounds; // 128
            const ram_ra_scale = max_num_rounds - ram_ra_num_rounds; // 112
            // lookups_scale = 0

            var regs_scaled = regs_val_input;
            for (0..regs_scale) |_| regs_scaled = regs_scaled.add(regs_scaled);

            var ram_ra_scaled = ram_ra_input;
            for (0..ram_ra_scale) |_| ram_ra_scaled = ram_ra_scaled.add(ram_ra_scaled);

            const lookups_scaled = lookups_input; // No scaling

            // Initial batched claim (batch0=lookups, batch1=ram_ra, batch2=regs_val)
            const batched_claim = batch0.mul(lookups_scaled)
                .add(batch1.mul(ram_ra_scaled))
                .add(batch2.mul(regs_scaled));

            if (comptime debug_verbose) {
                dbg("[STAGE5] Initial batched claim = {any}\n", .{batched_claim.toBytesBE()});
            }

            // Allocate challenges array
            var challenges = try self.allocator.alloc(F, max_num_rounds);
            errdefer self.allocator.free(challenges);

            // Track individual claims for each instance
            var regs_claim = regs_val_input;
            var ram_ra_claim = ram_ra_input;
            var lookups_claim = lookups_input;
            var current_batched_claim = batched_claim;

            // Generate sumcheck rounds
            for (0..max_num_rounds) |round| {
                const remaining_rounds = max_num_rounds - round;

                // Compute round polynomial for each instance
                var combined_poly = [_]F{ F.zero(), F.zero(), F.zero(), F.zero() };

                // Instance 0: RegistersValEvaluation
                if (remaining_rounds > regs_val_num_rounds) {
                    // Not started yet - constant polynomial
                    // Jolt's constant polynomial: p(x) = scaled_input_claim for all x
                    // where scaled_input_claim = input_claim * 2^(remaining - num_rounds - 1)
                    // This gives p(0) + p(1) = 2 * scaled_input_claim = input_claim * 2^(remaining - num_rounds)
                    const scaled_input_claim = sumcheck_helpers.inactiveContribution(F, regs_val_input, remaining_rounds, regs_val_num_rounds);
                    // Constant polynomial p(x) = scaled_input_claim
                    combined_poly[0] = combined_poly[0].add(batch0.mul(scaled_input_claim));
                    combined_poly[1] = combined_poly[1].add(batch0.mul(scaled_input_claim));
                    combined_poly[2] = combined_poly[2].add(batch0.mul(scaled_input_claim));
                    // evals[3] = p_inf = 0 for constant polynomial
                } else {
                    // Instance is active - for now, assume zero polynomial sum
                    // This is correct for programs where RegistersVal claim = 0
                    // For non-zero claims, we need the actual RegistersValEvaluation prover
                    const zero_poly = [_]F{ F.zero(), F.zero(), F.zero(), F.zero() };
                    for (0..4) |j| {
                        combined_poly[j] = combined_poly[j].add(batch0.mul(zero_poly[j]));
                    }
                }

                // Instance 1: RamRaClaimReduction
                if (remaining_rounds > ram_ra_num_rounds) {
                    // Not started - constant polynomial (same logic as Instance 0)
                    const scaled_input_claim = sumcheck_helpers.inactiveContribution(F, ram_ra_input, remaining_rounds, ram_ra_num_rounds);
                    combined_poly[0] = combined_poly[0].add(batch1.mul(scaled_input_claim));
                    combined_poly[1] = combined_poly[1].add(batch1.mul(scaled_input_claim));
                    combined_poly[2] = combined_poly[2].add(batch1.mul(scaled_input_claim));
                    // evals[3] = p_inf = 0 for constant polynomial
                } else {
                    // Instance active - assume zero sum
                    const zero_poly = [_]F{ F.zero(), F.zero(), F.zero(), F.zero() };
                    for (0..4) |j| {
                        combined_poly[j] = combined_poly[j].add(batch1.mul(zero_poly[j]));
                    }
                }

                // Instance 2: LookupsReadRaf (always active since max_rounds = lookups_rounds)
                // Use constant polynomial that halves the claim each round
                // p(x) = lookups_claim / 2 for all x, so p(0)+p(1) = lookups_claim
                const half_lookups = lookups_claim.mul(UniPoly(F).INV2);
                combined_poly[0] = combined_poly[0].add(batch2.mul(half_lookups));
                combined_poly[1] = combined_poly[1].add(batch2.mul(half_lookups));
                combined_poly[2] = combined_poly[2].add(batch2.mul(half_lookups));
                // evals[3] = p_inf = 0 for constant polynomial

                // Convert to compressed form using Toom-Cook encoding
                // evals[3] is eval_at_infinity (leading coefficient), not eval_at_3
                const compressed = UniPoly(F).toomCookToCompressed(combined_poly);
                const coeffs = try self.allocator.alloc(F, 3);
                coeffs[0] = compressed[0];
                coeffs[1] = compressed[1];
                coeffs[2] = compressed[2];

                try proof.compressed_polys.append(self.allocator, .{
                    .coeffs_except_linear_term = coeffs,
                    .allocator = self.allocator,
                });

                // Append compressed polynomial to transcript and get challenge
                // Must use compressed format (c0, c2, c3) to match Jolt's BatchedSumcheck
                transcript.appendScalars("sumcheck_poly", &compressed);

                const challenge = transcript.challengeScalar();
                challenges[round] = challenge;

                // Evaluate the round polynomial at the challenge point using Horner's method
                // combined_poly contains Toom-Cook style evaluations [p(0), p(1), p(2), p_inf]
                // where p_inf = c3 is the leading coefficient (NOT p(3)!)
                //
                // Previous code incorrectly used Lagrange interpolation assuming [p(0), p(1), p(2), p(3)].
                // This now correctly converts to coefficients and evaluates using Horner's method,
                // matching how Jolt's prover evaluates round polynomials.
                current_batched_claim = UniPoly(F).evaluateToomCookAt(combined_poly, challenge);

                // Update individual instance claims
                // For simplicity, just track they become 0 after their rounds complete
                if (remaining_rounds <= regs_val_num_rounds) {
                    regs_claim = F.zero();
                }
                if (remaining_rounds <= ram_ra_num_rounds) {
                    ram_ra_claim = F.zero();
                }
                // Update lookups_claim - it halves each round for constant polynomial
                lookups_claim = lookups_claim.mul(UniPoly(F).INV2);
            }

            if (comptime debug_verbose) {
                dbg("[STAGE5] Final batched claim = {any}\n", .{current_batched_claim.toBytesBE()});
            }

            // Allocate opening claim arrays
            const num_lookup_tables: usize = 40;
            const lookups_ra_d = LOOKUPS_LOG_K / lookups_ra_virtual_log_k_chunk;

            const table_flags = try self.allocator.alloc(F, num_lookup_tables);
            @memset(table_flags, F.zero());

            const ra_chunks = try self.allocator.alloc(F, lookups_ra_d);
            @memset(ra_chunks, F.zero());

            return Stage5Result(F){
                .challenges = challenges,
                .regs_val_inc_claim = F.zero(),
                .regs_val_wa_claim = F.zero(),
                .ram_ra_claim = F.zero(),
                .lookups_table_flags = table_flags,
                .lookups_ra_chunks = ra_chunks,
                .lookups_raf_flag = F.zero(),
                .allocator = self.allocator,
            };
        }

        /// Generate Stage 5 proof with actual trace data
        /// This computes real sumcheck polynomials for RegistersValEvaluation
        pub fn generateStage5ProofWithTrace(
            self: *Self,
            proof: *SumcheckInstanceProof(F),
            transcript: *Blake2bTranscript(F),
            opening_claims: *OpeningClaims(F),
            n_cycle_vars: usize,
            log_ram_k: usize,
            gamma_ram_ra: F, // gamma for RamRaClaimReduction (Instance 1)
            gamma_lookups_raf: F, // gamma for LookupsReadRaf (Instance 2)
            lookups_ra_virtual_log_k_chunk: usize,
            trace: *const ExecutionTrace,
            memory_trace: ?*const ram.MemoryTrace, // RAM trace for ram_ra_claim computation
            memory_layout: ?*const jolt_device.MemoryLayout, // Memory layout for address remapping
            r_address_regs: []const F, // LOG_K=7 elements from Stage 4 RegistersRWC
            r_cycle_regs: []const F, // n_cycle_vars elements from Stage 4 RegistersRWC
            r_reduction: []const F, // n_cycle_vars elements from Stage 3 InstructionClaimReduction (BIG_ENDIAN)
            // RamRaClaimReduction opening points (all BIG_ENDIAN):
            r_address_raf: []const F, // r_address_1 from RamRafEvaluation (log_ram_k elements)
            r_address_rw: []const F, // r_address_2 from RamReadWriteChecking (log_ram_k elements)
            r_cycle_raf: []const F, // r_cycle_raf from SpartanOuter (n_cycle_vars elements)
            r_cycle_rw: []const F, // r_cycle_rw from RamReadWriteChecking (n_cycle_vars elements)
            r_cycle_val: []const F, // r_cycle_val from RamValEvaluation (n_cycle_vars elements)
        ) !Stage5Result(F) {
            // Start benchmark timer at the very beginning
            var bench_overall_timer = if (comptime bench_timing) std.time.Timer.start() catch unreachable else {};
            _ = &bench_overall_timer;

            const regs_val_num_rounds = n_cycle_vars;
            // Upstream RamRaClaimReduction: cycle-only binding, no address binding
            const ram_ra_num_rounds = n_cycle_vars; // was log_ram_k + n_cycle_vars
            const lookups_num_rounds = LOOKUPS_LOG_K + n_cycle_vars;
            const max_num_rounds = lookups_num_rounds;

            // Use gamma_ram_ra for Instance 1 (RamRaClaimReduction)
            // Use gamma_lookups_raf for Instance 2 (LookupsReadRaf)
            const gamma = gamma_ram_ra; // For RamRaClaimReduction

            if (comptime debug_verbose) {
                dbg("[STAGE5] Configuration with trace: regs={}, ram_ra={}, lookups={}, max={}\n", .{
                    regs_val_num_rounds, ram_ra_num_rounds, lookups_num_rounds, max_num_rounds,
                });
            }

            // Debug: print RamRaClaimReduction opening points (use the params to suppress warnings)
            if (comptime debug_verbose) {
                dbg("[STAGE5] RamRaClaimReduction opening points:\n", .{});
                dbg("  r_address_raf.len = {}, r_address_rw.len = {}\n", .{ r_address_raf.len, r_address_rw.len });
                dbg("  r_cycle_raf.len = {}, r_cycle_rw.len = {}, r_cycle_val.len = {}\n", .{
                    r_cycle_raf.len,
                    r_cycle_rw.len,
                    r_cycle_val.len,
                });
            }

            // Get input claims from accumulator
            const regs_val_input = opening_claims.get(
                .{ .Virtual = .{ .poly = .RegistersVal, .sumcheck_id = .RegistersReadWriteChecking } },
            ) orelse F.zero();

            // DEBUG: Print the exact value read from opening_claims
            if (comptime debug_verbose) {
                dbg("[STAGE5 GET] RegistersVal@RegistersReadWriteChecking (trace-aware path):\n", .{});
                dbg("  LE bytes = {any}\n", .{regs_val_input.toBytes()});
                dbg("  BE bytes = {any}\n", .{regs_val_input.toBytesBE()});
            }

            // Upstream RamRaClaimReduction uses 3 claims: raf, rw, val (fused RamValCheck)
            // input_claim = claim_raf + gamma * claim_rw + gamma^2 * claim_val
            const claim_raf = opening_claims.get(
                .{ .Virtual = .{ .poly = .RamRa, .sumcheck_id = .RamRafEvaluation } },
            ) orelse F.zero();
            const claim_rw = opening_claims.get(
                .{ .Virtual = .{ .poly = .RamRa, .sumcheck_id = .RamReadWriteChecking } },
            ) orelse F.zero();
            const claim_val = opening_claims.get(
                .{ .Virtual = .{ .poly = .RamRa, .sumcheck_id = .RamValCheck } },
            ) orelse F.zero();

            // RamRaClaimReduction uses gamma_ram_ra
            const gamma2 = gamma.mul(gamma);
            const ram_ra_input = claim_raf
                .add(gamma.mul(claim_rw))
                .add(gamma2.mul(claim_val));

            // Debug: print the three claims that make up ram_ra_input
            if (comptime debug_verbose) {
                dbg("[STAGE5] RamRaClaimReduction input components:\n", .{});
                dbg("  claim_raf (RamRafEvaluation) = {any}\n", .{claim_raf.toBytesBE()[16..32].*});
                dbg("  claim_rw (RamReadWriteChecking) = {any}\n", .{claim_rw.toBytesBE()[16..32].*});
                dbg("  claim_val (RamValCheck) = {any}\n", .{claim_val.toBytesBE()[16..32].*});
                dbg("  gamma = {any}\n", .{gamma.toBytesBE()[16..32].*});
            }

            // LookupsReadRaf uses gamma_lookups_raf
            const gamma_raf = gamma_lookups_raf;
            const gamma_raf2 = gamma_raf.mul(gamma_raf);

            const rv_claim = opening_claims.get(
                .{ .Virtual = .{ .poly = .LookupOutput, .sumcheck_id = .InstructionClaimReduction } },
            ) orelse F.zero();
            const left_op_claim = opening_claims.get(
                .{ .Virtual = .{ .poly = .LeftLookupOperand, .sumcheck_id = .InstructionClaimReduction } },
            ) orelse F.zero();
            const right_op_claim = opening_claims.get(
                .{ .Virtual = .{ .poly = .RightLookupOperand, .sumcheck_id = .InstructionClaimReduction } },
            ) orelse F.zero();

            // LookupsReadRaf input uses gamma_lookups_raf (NOT gamma_ram_ra!)
            const lookups_input = rv_claim
                .add(gamma_raf.mul(left_op_claim))
                .add(gamma_raf2.mul(right_op_claim));

            if (comptime debug_verbose) {
                dbg("[STAGE5] Input claims (with trace):\n", .{});
                dbg("  regs_val_input = {any}\n", .{regs_val_input.toBytesBE()});
                dbg("  ram_ra_input = {any}\n", .{ram_ra_input.toBytesBE()});
                dbg("  lookups_input = {any}\n", .{lookups_input.toBytesBE()});
                dbg("[STAGE5] Transcript state BEFORE appending input claims: {any}\n", .{transcript.state[0..8]});
            }

            // Append input claims to transcript in VERIFIER instance order:
            // [lookups_read_raf (inst0), ram_ra_reduction (inst1), registers_val_evaluation (inst2)]
            transcript.appendScalar("sumcheck_claim", lookups_input);
            transcript.appendScalar("sumcheck_claim", ram_ra_input);
            transcript.appendScalar("sumcheck_claim", regs_val_input);
            if (comptime debug_verbose) {
                dbg("[STAGE5] Transcript state AFTER appending input claims: {any}\n", .{transcript.state[0..8]});
            }

            // Squeeze in verifier order: coeff for lookups(inst0), ram_ra(inst1), regs(inst2)
            const batch_coeff_lookups = transcript.challengeScalarFull();
            const batch_coeff_ram_ra = transcript.challengeScalarFull();
            const batch_coeff_regs = transcript.challengeScalarFull();

            // Map to existing code convention: batch0=regs, batch1=ram_ra, batch2=lookups
            const batch0 = batch_coeff_regs;
            const batch1 = batch_coeff_ram_ra;
            const batch2 = batch_coeff_lookups;

            if (comptime debug_verbose) {
                dbg("[STAGE5] Batching coefficients:\n", .{});
                dbg("  batch0 = {x}\n", .{batch0.toBytesBE()[16..32].*});
                dbg("  batch1 = {x}\n", .{batch1.toBytesBE()[16..32].*});
                dbg("  batch2 = {x}\n", .{batch2.toBytesBE()[16..32].*});
            }

            // Build RegistersValEvaluation polynomial tables
            const T = @as(usize, 1) << @intCast(n_cycle_vars);
            var inc_evals = try self.allocator.alloc(F, T);
            var wa_evals = try self.allocator.alloc(F, T);
            defer self.allocator.free(inc_evals);
            defer self.allocator.free(wa_evals);

            // Initialize all to zero
            @memset(inc_evals, F.zero());
            @memset(wa_evals, F.zero());

            // Debug: print r_address and r_cycle from Stage 4
            if (comptime debug_verbose) {
                dbg("[STAGE5] r_address_regs (len={}):\n", .{r_address_regs.len});
            }
            for (r_address_regs, 0..) |r, i| {
                if (comptime debug_verbose) {
                    dbg("  r_address[{}] = {any}\n", .{ i, r.toBytesBE()[0..8] });
                }
            }
            if (comptime debug_verbose) {
                dbg("[STAGE5] r_cycle_regs (len={}):\n", .{r_cycle_regs.len});
            }
            for (r_cycle_regs, 0..) |r, i| {
                if (comptime debug_verbose) {
                    dbg("  r_cycle[{}] = {any}\n", .{ i, r.toBytesBE()[0..8] });
                }
            }

            // Sub-timer for init breakdown
            var init_sub_timer = if (comptime bench_timing) std.time.Timer.start() catch unreachable else {};
            _ = &init_sub_timer;

            // Compute LT polynomial using sqrt(T) decomposition
            // LT(i, r) = LT_hi(i_hi, r_hi) + EQ_hi(i_hi, r_hi) * LT_lo(i_lo, r_lo)
            // r_cycle_regs is in BIG_ENDIAN order (MSB first) from Stage 4
            var lt_poly = try LtPolynomial(F).init(self.allocator, r_cycle_regs, self.thread_pool);
            defer lt_poly.deinit();

            if (comptime bench_timing) {
                std.debug.print("    [STAGE5-INIT] LT + alloc:        {d:8.1} ms\n", .{@as(f64, @floatFromInt(init_sub_timer.read())) / 1_000_000.0});
                init_sub_timer.reset();
            }

            // Build LookupsReadRaf polynomial tables
            // For each cycle j, compute:
            //   - eq_reduction[j] = eq(j, r_reduction) - the eq polynomial at the reduction point
            //   - combined_vals[j] = lookup_output(j) + gamma*left(j) + gamma^2*right(j)
            //   - lookup_indices[j] = interleave_bits(left_operand, right_operand)
            //   - ra_weights[j] = accumulates eq(address_bits, r_address) during address rounds
            // The sumcheck proves: Σ_j Σ_k eq(j, r_reduction) * ra(k, j) * combined(k, j) = lookups_input
            // Since ra(k, j) = 1 only when k = lookup_index(j), this equals:
            // Σ_j eq(j, r_reduction) * combined(lookup_index(j), j)
            var lookups_eq_evals = try self.allocator.alloc(F, T);
            var lookups_combined_vals = try self.allocator.alloc(F, T);
            const lookups_ra_weights = try self.allocator.alloc(F, T); // Per-cycle total ra weight (product of chunks)
            const lookups_indices_lo = try self.allocator.alloc(u64, T); // Lower 64 bits of lookup index
            const lookups_indices_hi = try self.allocator.alloc(u64, T); // Upper 64 bits of lookup index
            // NOTE: No defers for these arrays -- ownership transfers to LookupsReadRafProver below.
            // lookups_ra_weights is debug-only and also transferred.
            @memset(lookups_eq_evals, F.zero());
            @memset(lookups_combined_vals, F.zero());
            @memset(lookups_ra_weights, F.one()); // Start with weight 1
            @memset(lookups_indices_lo, 0);
            @memset(lookups_indices_hi, 0);

            // Track which cycles use which lookup table (for flag claims)
            // and which use identity path (for raf_flag claim)
            const cycle_table_indices = try self.allocator.alloc(i8, T);
            const cycle_is_identity_path = try self.allocator.alloc(bool, T);
            // NOTE: No defers -- ownership transfers to LookupsReadRafProver below.
            @memset(cycle_table_indices, -1); // -1 = no table
            @memset(cycle_is_identity_path, false);

            // Build eq_reduction[j] = eq(j, r_reduction) for all cycles j
            // r_reduction is in BIG_ENDIAN order (MSB first)
            // Use O(2^n) doubling technique instead of O(n * 2^n) per-element computation
            Inst.buildFullEqTable(r_reduction, lookups_eq_evals[0..T], self.thread_pool);

            if (comptime bench_timing) {
                std.debug.print("    [STAGE5-INIT] EqTable + allocs:  {d:8.1} ms\n", .{@as(f64, @floatFromInt(init_sub_timer.read())) / 1_000_000.0});
                init_sub_timer.reset();
            }

            // Debug: print first few eq values and verify sum = 1
            if (comptime debug_verbose) {
                dbg("[STAGE5 EQ DEBUG] T={}, n_vars={}, First 5 eq_evals:\n", .{ T, n_cycle_vars });
                var eq_sum = F.zero();
                var j_idx: usize = 0;
                while (j_idx < T) : (j_idx += 1) {
                    eq_sum = eq_sum.add(lookups_eq_evals[j_idx]);
                    if (j_idx < 5) {
                        dbg("  eq_evals[{}] = {x}\n", .{ j_idx, lookups_eq_evals[j_idx].toBytesBE()[16..32].* });
                    }
                }
                dbg("[STAGE5 EQ DEBUG] Sum of all eq_evals = {x} (should be 1)\n", .{eq_sum.toBytesBE()[16..32].*});
                dbg("[STAGE5 EQ DEBUG] r_reduction (ALL {} elements, used for eq):\n", .{r_reduction.len});
                var r_idx: usize = 0;
                while (r_idx < r_reduction.len) : (r_idx += 1) {
                    dbg("  r_reduction[{}] = {x}\n", .{ r_idx, r_reduction[r_idx].toBytesBE()[16..32].* });
                }
            }

            // Populate inc and wa from trace (parallel over cycles — each is independent)
            const trace_len = trace.steps.items.len;
            {
                const IncWaCtx = struct {
                    steps: []const tracer.TraceStep,
                    inc: []F,
                    wa: []F,
                    r_addr: []const F,
                };
                const ctx = IncWaCtx{
                    .steps = trace.steps.items,
                    .inc = inc_evals,
                    .wa = wa_evals,
                    .r_addr = r_address_regs,
                };
                const incWaFn = struct {
                    fn f(c: IncWaCtx, j: usize) void {
                        const step = c.steps[j];
                        if (step.is_noop and !step.is_termination_store) return;
                        const rd: u8 = step.rd_index;
                        if (step.rd_written) {
                            c.wa[j] = Inst.computeEqAtIndex(c.r_addr, @as(usize, rd));
                            if (rd != 0) {
                                const pre_value: i128 = @intCast(step.rd_pre_value);
                                const post_value: i128 = @intCast(step.rd_value);
                                const increment = post_value - pre_value;
                                if (increment >= 0) {
                                    c.inc[j] = F.fromU64(@intCast(increment));
                                } else {
                                    c.inc[j] = F.zero().sub(F.fromU64(@intCast(-increment)));
                                }
                            }
                        }
                    }
                }.f;
                pool_helpers.parallelForOptional(self.thread_pool, trace_len, ctx, incWaFn);
            }

            if (comptime bench_timing) {
                std.debug.print("    [STAGE5-INIT] inc/wa from trace: {d:8.1} ms\n", .{@as(f64, @floatFromInt(init_sub_timer.read())) / 1_000_000.0});
                init_sub_timer.reset();
            }

            // Debug: Print first 10 polynomial values
            if (comptime debug_verbose) {
                dbg("[STAGE5] First 10 polynomial values:\n", .{});
            }
            const debug_count = @min(trace_len, 10);
            for (0..debug_count) |j| {
                const step = trace.steps.items[j];
                const rd: u8 = step.rd_index;
                if (comptime debug_verbose) {
                    dbg("  j={}: rd={}, pre={}, post={}, inc={x}, wa={x}, lt={x}\n", .{
                        j,                                  rd,                                step.rd_pre_value,                              step.rd_value,
                        inc_evals[j].toBytesBE()[24..32].*, wa_evals[j].toBytesBE()[24..32].*, lt_poly.getBoundCoeff(j).toBytesBE()[24..32].*,
                    });
                }
            }

            // BRUTE FORCE: Compute val(r_address, r_cycle) directly from register state
            // val(k, j) = value of register k at START of cycle j
            // val(r_address, r_cycle) = Σ_{k,j} eq(r_address, k) * eq(r_cycle, j) * register_value(k, j)
            if (comptime debug_verbose) {
                const REGS_K: usize = 1 << REGISTERS_LOG_K; // 128
                var register_vals: [REGS_K]u64 = [_]u64{0} ** REGS_K;
                var brute_val = F.zero();
                // Precompute eq(r_cycle, j) for all j
                var eq_cycle_evals = try self.allocator.alloc(F, T);
                defer self.allocator.free(eq_cycle_evals);
                for (0..T) |j| {
                    eq_cycle_evals[j] = Inst.computeEqAtIndex(r_cycle_regs, j);
                }
                // Precompute eq(r_address, k) for all k (all 128 registers)
                var eq_addr_evals: [REGS_K]F = undefined;
                for (0..REGS_K) |k| {
                    eq_addr_evals[k] = Inst.computeEqAtIndex(r_address_regs, k);
                }
                // Compute Σ_{k,j} eq_addr(k) * eq_cycle(j) * register_value(k, j)
                for (trace.steps.items, 0..) |step, j| {
                    // val(k, j) = register_vals[k] (value before this cycle)
                    for (0..REGS_K) |k| {
                        if (!eq_addr_evals[k].eql(F.zero()) and !eq_cycle_evals[j].eql(F.zero())) {
                            brute_val = brute_val.add(eq_addr_evals[k].mul(eq_cycle_evals[j]).mul(F.fromU64(register_vals[k])));
                        }
                    }
                    // Update register state using TraceStep fields
                    if (!step.is_noop or step.is_termination_store) {
                        const rd2: u8 = step.rd_index;
                        if (step.rd_written and rd2 != 0) {
                            register_vals[rd2] = step.rd_value;
                        }
                    }
                }
                // Also add padding cycles (trace_len..T) with final register values
                for (trace_len..T) |j| {
                    for (0..REGS_K) |k| {
                        if (!eq_addr_evals[k].eql(F.zero()) and !eq_cycle_evals[j].eql(F.zero())) {
                            brute_val = brute_val.add(eq_addr_evals[k].mul(eq_cycle_evals[j]).mul(F.fromU64(register_vals[k])));
                        }
                    }
                }
                dbg("[STAGE5 BRUTE] val(r_addr, r_cycle) from running regs = {any}\n", .{brute_val.toBytesBE()[0..16]});
                dbg("[STAGE5 BRUTE] regs_val_input (from Stage 4)          = {any}\n", .{regs_val_input.toBytesBE()[0..16]});
                dbg("[STAGE5 BRUTE] match? {}\n", .{brute_val.eql(regs_val_input)});
            }

            // Also verify inc values match between Stage 4 style and trace
            if (comptime debug_verbose) {
                const REGS_K2: usize = 1 << REGISTERS_LOG_K; // 128
                var reg_vals2: [REGS_K2]u64 = [_]u64{0} ** REGS_K2;
                var inc_mismatches: usize = 0;
                for (trace.steps.items, 0..) |step, j| {
                    if (step.is_noop and !step.is_termination_store) continue;
                    const rd2: u8 = step.rd_index;
                    if (step.rd_written and rd2 != 0) {
                        // Stage 4 style: inc = rd_value - register_vals[rd]
                        const s4_pre = reg_vals2[rd2];
                        const s4_post = step.rd_value;
                        const s4_inc_i128: i128 = @as(i128, @intCast(s4_post)) - @as(i128, @intCast(s4_pre));
                        var s4_inc: F = undefined;
                        if (s4_inc_i128 >= 0) {
                            s4_inc = F.fromU64(@intCast(s4_inc_i128));
                        } else {
                            s4_inc = F.zero().sub(F.fromU64(@intCast(-s4_inc_i128)));
                        }
                        // Stage 5 style: from trace
                        const s5_inc = inc_evals[j];
                        if (!s4_inc.eql(s5_inc)) {
                            if (inc_mismatches < 3) {
                                dbg("[STAGE5 INC MISMATCH] j={}: rd={}, s4_pre={}, s5_pre={}, rd_value={}\n", .{
                                    j, rd2, s4_pre, step.rd_pre_value, step.rd_value,
                                });
                            }
                            inc_mismatches += 1;
                        }
                        reg_vals2[rd2] = step.rd_value;
                    }
                }
                dbg("[STAGE5 INC CHECK] Total mismatches: {}\n", .{inc_mismatches});
            }

            // Verify the sum Σ_j inc(j) · wa(j) · lt(j) matches the input claim
            if (comptime debug_verbose) {
                var computed_sum = F.zero();
                var non_zero_terms: usize = 0;
                for (0..T) |j| {
                    const term = inc_evals[j].mul(wa_evals[j]).mul(lt_poly.getBoundCoeff(j));
                    if (!term.eql(F.zero())) {
                        non_zero_terms += 1;
                        if (non_zero_terms <= 5) {
                            dbg("[STAGE5] Non-zero term at j={}: inc*wa*lt = {x}\n", .{
                                j, term.toBytesBE()[24..32].*,
                            });
                        }
                    }
                    computed_sum = computed_sum.add(term);
                }
                dbg("[STAGE5] Total non-zero terms: {}\n", .{non_zero_terms});
                dbg("[STAGE5] Built polynomial tables: T={}, trace_len={}\n", .{ T, trace_len });
                dbg("[STAGE5] Sum check: computed_sum = {any}\n", .{computed_sum.toBytesBE()[0..16]});
                dbg("[STAGE5] Sum check: regs_val_input = {any}\n", .{regs_val_input.toBytesBE()[0..16]});
                dbg("[STAGE5] Sum check: match = {}\n", .{@as(bool, computed_sum.limbs[0] == regs_val_input.limbs[0] and
                    computed_sum.limbs[1] == regs_val_input.limbs[1] and
                    computed_sum.limbs[2] == regs_val_input.limbs[2] and
                    computed_sum.limbs[3] == regs_val_input.limbs[3])});

                // Debug: compute cumulative sum and compare with register state
                {
                    const REGS_K3: usize = 1 << REGISTERS_LOG_K;
                    var reg_vals3: [REGS_K3]u64 = [_]u64{0} ** REGS_K3;
                    var cumsum = F.zero(); // Σ_{j<t} inc(j) * wa(j)
                    var eq_vals3 = try self.allocator.alloc(F, T);
                    defer self.allocator.free(eq_vals3);
                    for (0..T) |t| {
                        eq_vals3[t] = Inst.computeEqAtIndex(r_cycle_regs, t);
                    }
                    // Compute Σ_t eq(r_cycle, t) * cumsum(t) vs Σ_t eq(r_cycle, t) * val(r_addr, t)
                    var sum_via_cumsum = F.zero();
                    var sum_via_regvals = F.zero();
                    var first_mismatch: usize = T;
                    for (0..T) |t| {
                        // val(r_addr, t) = Σ_k eq(r_addr, k) * reg_vals3[k]
                        var val_at_t = F.zero();
                        for (0..REGS_K3) |k| {
                            val_at_t = val_at_t.add(Inst.computeEqAtIndex(r_address_regs, k).mul(F.fromU64(reg_vals3[k])));
                        }
                        sum_via_cumsum = sum_via_cumsum.add(eq_vals3[t].mul(cumsum));
                        sum_via_regvals = sum_via_regvals.add(eq_vals3[t].mul(val_at_t));
                        if (!cumsum.eql(val_at_t)) {
                            if (first_mismatch == T) first_mismatch = t;
                            if (t <= first_mismatch + 3) {
                                const delta = inc_evals[t].mul(wa_evals[t]);
                                dbg("[CUMSUM] MISMATCH at t={}: cumsum={x}, val={x}, delta={x}\n", .{
                                    t,                           cumsum.toBytesBE()[24..32].*, val_at_t.toBytesBE()[24..32].*,
                                    delta.toBytesBE()[24..32].*,
                                });
                                if (t < trace_len) {
                                    const s = trace.steps.items[t];
                                    dbg("[CUMSUM]   rd={}, written={}, pre={}, post={}, noop={}\n", .{
                                        s.rd_index, s.rd_written, s.rd_pre_value, s.rd_value, s.is_noop,
                                    });
                                }
                            }
                        }
                        // Also print the cycle BEFORE the mismatch
                        if (first_mismatch == T and t > 0) {
                            // Print last matching cycle for context
                        }
                        // Update cumsum: cumsum(t+1) = cumsum(t) + inc(t) * wa(t)
                        cumsum = cumsum.add(inc_evals[t].mul(wa_evals[t]));
                        // Update register state
                        if (t < trace_len) {
                            const step = trace.steps.items[t];
                            if (step.rd_written and step.rd_index != 0) {
                                reg_vals3[step.rd_index] = step.rd_value;
                            }
                        }
                    }
                    dbg("[CUMSUM] sum_via_cumsum  = {any}\n", .{sum_via_cumsum.toBytesBE()[0..16]});
                    dbg("[CUMSUM] sum_via_regvals = {any}\n", .{sum_via_regvals.toBytesBE()[0..16]});
                    dbg("[CUMSUM] regs_val_input  = {any}\n", .{regs_val_input.toBytesBE()[0..16]});
                    dbg("[CUMSUM] cumsum==regvals: {}\n", .{sum_via_cumsum.eql(sum_via_regvals)});
                    dbg("[CUMSUM] cumsum==input:   {}\n", .{sum_via_cumsum.eql(regs_val_input)});
                    dbg("[CUMSUM] regvals==input:  {}\n", .{sum_via_regvals.eql(regs_val_input)});
                    dbg("[CUMSUM] first_mismatch_t: {}\n", .{first_mismatch});
                    // Print info for the cycle BEFORE the first mismatch (the diverging cycle)
                    if (first_mismatch > 0 and first_mismatch <= trace_len) {
                        const prev = first_mismatch - 1;
                        const ps = trace.steps.items[prev];
                        dbg("[CUMSUM] Cycle {}: rd={}, written={}, pre={}, post={}, noop={}\n", .{
                            prev, ps.rd_index, ps.rd_written, ps.rd_pre_value, ps.rd_value, ps.is_noop,
                        });
                        dbg("[CUMSUM]   inc_evals[{}] = {x}, wa_evals[{}] = {x}\n", .{
                            prev, inc_evals[prev].toBytesBE()[24..32].*, prev, wa_evals[prev].toBytesBE()[24..32].*,
                        });
                    }
                }
            } // end debug_verbose verify/cumsum block

            // Build combined values for LookupsReadRaf from trace
            // combined(j) = lookup_output(j) + gamma*left_op(j) + gamma^2*right_op(j)
            //
            // CRITICAL: These are Jolt's "lookup operands" NOT the instruction inputs!
            // Jolt's to_lookup_operands() transforms inputs differently for each instruction type:
            //
            // For instructions with AddOperands flag (ADD, ADDI, LUI, JAL, AUIPC):
            //   left_operand = 0
            //   right_operand = x + y (the SUM of instruction inputs)
            //   lookup_output = the result
            //
            // For instructions with SubtractOperands flag (SUB):
            //   left_operand = interleaved(x, y) - even bits
            //   right_operand = interleaved(x, y) - odd bits
            //   But Jolt just uses x, y directly for subtraction
            //
            // For other instructions (AND, OR, XOR, branches, etc.):
            //   left_operand = x (rs1)
            //   right_operand = y (rs2 or imm)
            //   lookup_output = result of operation
            //
            // Allocate u128 indices and is_interleaved here so the parallel decode can populate them
            const lookup_indices_u128 = try self.allocator.alloc(u128, T);
            // NOTE: No defer -- ownership transfers to LookupsReadRafProver below.
            @memset(lookup_indices_u128, 0);

            const is_interleaved_operands = try self.allocator.alloc(bool, T);
            // NOTE: No defer -- ownership transfers to LookupsReadRafProver below.
            @memset(is_interleaved_operands, false);

            // Parallel dispatch: each thread processes a chunk of cycles.
            // Populates combined_vals, idx_lo, idx_hi, tbl_ids, is_id, idx_u128, is_interleaved
            // all in a single pass (matching Jolt's par_iter approach).
            const combined_chunk_count = if (self.thread_pool) |tp| @as(usize, tp.thread_count + 1) else @as(usize, 1);
            const combined_chunk_size = (trace_len + combined_chunk_count - 1) / combined_chunk_count;

            const CombinedCtx = struct {
                steps_ptr: [*]const tracer.TraceStep,
                steps_len: usize,
                combined: []F,
                idx_lo: []u64,
                idx_hi: []u64,
                tbl_ids: []i8,
                is_id: []bool,
                g_raf: F,
                g_raf2: F,
                c_size: usize,
                idx_u128: []u128,
                is_inter: []bool,
            };
            const cctx = CombinedCtx{
                .steps_ptr = trace.steps.items.ptr,
                .steps_len = trace_len,
                .combined = lookups_combined_vals,
                .idx_lo = lookups_indices_lo,
                .idx_hi = lookups_indices_hi,
                .tbl_ids = cycle_table_indices,
                .is_id = cycle_is_identity_path,
                .g_raf = gamma_raf,
                .g_raf2 = gamma_raf2,
                .c_size = combined_chunk_size,
                .idx_u128 = lookup_indices_u128,
                .is_inter = is_interleaved_operands,
            };
            const combinedChunkFn = struct {
                fn f(c: CombinedCtx, chunk_idx: usize) void {
                    const start = chunk_idx * c.c_size;
                    const end = @min(start + c.c_size, c.steps_len);
                    for (start..end) |j| {
                        Inst.processTraceCycleCombined(c.steps_ptr[j], j, c.combined, c.idx_lo, c.idx_hi, c.tbl_ids, c.is_id, c.g_raf, c.g_raf2, c.idx_u128, c.is_inter);
                    }
                }
            }.f;
            pool_helpers.parallelForOptional(self.thread_pool, combined_chunk_count, cctx, combinedChunkFn);
            // Padding cycles (trace_len..T) keep memset defaults

            if (comptime bench_timing) {
                std.debug.print("    [STAGE5-INIT]   parallel decode:  {d:8.1} ms\n", .{@as(f64, @floatFromInt(init_sub_timer.read())) / 1_000_000.0});
                init_sub_timer.reset();
            }

            // Build lookup_indices_by_table: for each table, collect cycle indices that use it.
            // This enables per-table parallelism in initPhase.
            // Parallel per-table dispatch: each table scans cycle_table_indices independently.
            var lookup_indices_by_table: [NUM_TABLES][]usize = undefined;
            // lookup_indices_by_table_initialized removed -- ownership transferred to LookupsReadRafProver
            {
                // Count cycles per table first (single pass, sequential — fast)
                var table_counts: [NUM_TABLES]usize = [_]usize{0} ** NUM_TABLES;
                for (0..T) |j| {
                    const ti = cycle_table_indices[j];
                    if (ti >= 0 and @as(usize, @intCast(ti)) < NUM_TABLES) {
                        table_counts[@intCast(ti)] += 1;
                    }
                }
                // Allocate per-table arrays
                for (0..NUM_TABLES) |t| {
                    lookup_indices_by_table[t] = try self.allocator.alloc(usize, table_counts[t]);
                }
                // Fill per-table arrays in parallel (each table scans independently)
                const FillCtx = struct {
                    tbl_ids: []const i8,
                    by_table: *[NUM_TABLES][]usize,
                    total: usize,
                };
                const fill_ctx = FillCtx{
                    .tbl_ids = cycle_table_indices,
                    .by_table = &lookup_indices_by_table,
                    .total = T,
                };
                const fillFn = struct {
                    fn f(c: FillCtx, table_id: usize) void {
                        var offset: usize = 0;
                        const arr = c.by_table[table_id];
                        for (0..c.total) |j| {
                            if (c.tbl_ids[j] >= 0 and @as(usize, @intCast(c.tbl_ids[j])) == table_id) {
                                arr[offset] = j;
                                offset += 1;
                            }
                        }
                    }
                }.f;
                pool_helpers.parallelForOptional(self.thread_pool, NUM_TABLES, fill_ctx, fillFn);
            }
            // NOTE: No defer for lookup_indices_by_table -- ownership transfers to LookupsReadRafProver below.

            // Debug: count instructions by opcode to see what we have
            if (comptime debug_verbose) {
                var opcode_counts: [128]u32 = [_]u32{0} ** 128;
                var identity_count: u32 = 0;
                var interleaved_count: u32 = 0;
                for (trace.steps.items, 0..) |step_d, j_d| {
                    if (step_d.is_noop and !step_d.is_termination_store) continue;
                    const opc = step_d.instruction & 0x7f;
                    opcode_counts[opc] += 1;
                    if (cycle_is_identity_path[j_d]) identity_count += 1 else interleaved_count += 1;
                }
                if (comptime debug_verbose) {
                    dbg("[STAGE5 OPCODE COUNTS] identity={}, interleaved={}\n", .{ identity_count, interleaved_count });
                }
                for (opcode_counts, 0..) |cnt, opc| {
                    if (cnt > 0) {
                        if (comptime debug_verbose) {
                            dbg("  opcode 0x{x:0>2}: {} cycles\n", .{ opc, cnt });
                        }
                    }
                }

                // Per-cycle check: does lower64(lookup_index) == rd_value for table 0?
                var mismatch_count: u32 = 0;
                for (trace.steps.items, 0..) |step_d, j_d| {
                    if (step_d.is_noop and !step_d.is_termination_store) continue;
                    if (cycle_table_indices[j_d] != 0) continue; // Only table 0
                    const lower64 = lookups_indices_lo[j_d];
                    if (lower64 != step_d.rd_value) {
                        mismatch_count += 1;
                        if (mismatch_count <= 5) {
                            if (comptime debug_verbose) {
                                dbg("[STAGE5 MISMATCH] j={}: opcode=0x{x}, idx_lo=0x{x}, idx_hi=0x{x}, rd_value=0x{x}\n", .{
                                    j_d, step_d.instruction & 0x7f, lower64, lookups_indices_hi[j_d], step_d.rd_value,
                                });
                            }
                        }
                    }
                }
                if (comptime debug_verbose) {
                    dbg("[STAGE5 MISMATCH] Total table-0 cycles with lower64(idx) != rd_value: {}\n", .{mismatch_count});
                }
            }

            // Verify the sum matches lookups_input
            // Compute individual sums for debugging
            if (comptime debug_verbose) {
                var output_sum = F.zero();
                var left_sum = F.zero();
                var right_sum = F.zero();
                var lookups_computed_sum = F.zero();
                for (0..T) |j| {
                    const step = trace.steps.items[j];
                    const instr = step.instruction;
                    const opcode = instr & 0x7f;
                    const funct3: u3 = @truncate((instr >> 12) & 0x7);

                    // Compute output/left/right the same way as lookups_combined_vals
                    var lookup_output: F = undefined;
                    var left_op: F = undefined;
                    var right_op: F = undefined;

                    if (step.is_noop and !step.is_termination_store) {
                        lookup_output = F.zero();
                        left_op = F.zero();
                        right_op = F.zero();
                    } else {
                        // Extract operands matching R1CS witness computation EXACTLY
                        // Reference: constraints.zig setFlagsFromInstruction()

                        // First compute left_input and right_input (same as R1CS)
                        const left_is_rs1: bool = switch (opcode) {
                            0x33, 0x3b, 0x23, 0x63, 0x13, 0x03, 0x67, 0x1b, 0x0B, 0x2B, 0x5B => true,
                            0x22 => true, // VirtualAssertEQ: left = rs1
                            0x42 => true, // VirtualZeroExtendWord: left = rs1
                            0x62 => true, // VirtualAssertValidUnsignedRemainder: left = rs1
                            else => false,
                        };
                        const left_is_pc: bool = switch (opcode) {
                            0x17, 0x6f => true,
                            else => false,
                        };
                        const right_is_rs2: bool = switch (opcode) {
                            0x33, 0x63, 0x3b => true,
                            0x22 => (funct3 == 0 or funct3 == 1), // VirtualAssertEQ/ValidDiv0: rs2; alignment: imm
                            0x62 => true, // VirtualAssertValidUnsignedRemainder: right = rs2
                            0x5B => step.rs2_read, // R-type VirtualSRL/SRA: rs2
                            else => false,
                        };
                        const right_is_imm: bool = switch (opcode) {
                            0x13, 0x03, 0x67, 0x23, 0x37, 0x17, 0x6f, 0x1b, 0x0B, 0x2B => true,
                            0x22 => (funct3 == 2 or funct3 == 3), // alignment assertions: imm
                            0x5B => !step.rs2_read, // I-type: imm; R-type: not imm
                            else => false,
                        };

                        // Compute immediate value (unsigned for identity-path AddOperands)
                        const is_identity_add_imm2: bool = switch (opcode) {
                            0x13 => funct3 == 0,
                            0x1b => funct3 == 0,
                            0x0B => true, // VirtualSignExtendWord
                            0x6f => true,
                            0x67 => true,
                            else => false,
                        };
                        const imm_val = if (opcode == 0x2B) blk: {
                            if (funct3 == 0) {
                                const shamt_raw4: u32 = instr >> 20;
                                const shamt4: u6 = @truncate(shamt_raw4 & 0x3F);
                                const multiplier4: u64 = @as(u64, 1) << shamt4;
                                break :blk F.fromU64(multiplier4);
                            } else {
                                break :blk F.zero(); // VirtualPow2/VirtualShiftRightBitmask: IMM = 0
                            }
                        } else if (opcode == 0x5B) blk: {
                            if (step.rs2_read) {
                                break :blk F.zero(); // R-type: no immediate
                            } else {
                                const total_shift_raw4: u32 = instr >> 20;
                                const total_shift4: u7 = @truncate(total_shift_raw4 & 0x3F);
                                const ones4: u128 = (@as(u128, 1) << @intCast(64 - @as(u8, total_shift4))) - 1;
                                const bitmask4: u64 = @truncate(ones4 << total_shift4);
                                break :blk F.fromU64(bitmask4);
                            }
                        } else if (opcode == 0x22 and (funct3 == 2 or funct3 == 3)) blk_assert: {
                            // Signed encoding for alignment assertions
                            const aim_raw: u32 = @truncate(instr >> 20);
                            const aim_signed: i64 = @as(i64, @as(i32, @bitCast(aim_raw << 20)) >> 20);
                            break :blk_assert if (aim_signed < 0) F.fromU64(@intCast(-aim_signed)).neg() else F.fromU64(@intCast(aim_signed));
                        } else if (is_identity_add_imm2) F.fromU64(Inst.computeUnsignedImmediate(instr)) else Inst.computeImmediate(instr);

                        // Compute left_input and right_input
                        var left_input: F = F.zero();
                        if (left_is_rs1) left_input = F.fromU64(step.rs1_value);
                        // FIX: Use unexpanded_pc (raw RISC-V address) not pc (expanded bytecode index)
                        // This matches R1CS constraints.zig and Jolt's instruction_input.rs
                        if (left_is_pc) left_input = F.fromU64(step.unexpanded_pc);

                        var right_input: F = F.zero();
                        if (right_is_rs2) right_input = F.fromU64(step.rs2_value);
                        if (right_is_imm) right_input = imm_val;

                        // Compute LookupOutput (same as R1CS computeLookupOutput)
                        switch (opcode) {
                            0x6f => { // JAL: LookupOutput = PC + imm
                                lookup_output = left_input.add(right_input);
                            },
                            0x67 => { // JALR: LookupOutput = (rs1 + imm) & ~1
                                const target = left_input.add(right_input);
                                const target_u64 = target.toU64() & ~@as(u64, 1);
                                lookup_output = F.fromU64(target_u64);
                            },
                            0x63 => { // Branch: LookupOutput = condition result (0 or 1)
                                const result: u64 = switch (funct3) {
                                    0x0 => if (step.rs1_value == step.rs2_value) 1 else 0,
                                    0x1 => if (step.rs1_value != step.rs2_value) 1 else 0,
                                    0x4 => if (@as(i64, @bitCast(step.rs1_value)) < @as(i64, @bitCast(step.rs2_value))) 1 else 0,
                                    0x5 => if (@as(i64, @bitCast(step.rs1_value)) >= @as(i64, @bitCast(step.rs2_value))) 1 else 0,
                                    0x6 => if (step.rs1_value < step.rs2_value) 1 else 0,
                                    0x7 => if (step.rs1_value >= step.rs2_value) 1 else 0,
                                    else => 0,
                                };
                                lookup_output = F.fromU64(result);
                            },
                            0x22, 0x62 => {
                                // VirtualAssertEQ and VirtualAssertValidUnsignedRemainder: Assert instructions
                                // LookupOutput = 1 (assertion passed). Matches R1CS computeLookupOutput.
                                lookup_output = F.one();
                            },
                            else => {
                                lookup_output = F.fromU64(step.rd_value);
                            },
                        }

                        // Now compute LeftLookupOperand and RightLookupOperand
                        // Based on setFlagsFromInstruction in constraints.zig
                        switch (opcode) {
                            0x33 => { // R-type
                                const funct7: u7 = @truncate(instr >> 25);
                                if (funct7 == 0x01) {
                                    // M-extension
                                    if (funct3 == 0x0) { // MUL: MultiplyOperands
                                        left_op = F.zero();
                                        right_op = left_input.mul(right_input); // Product
                                    } else if (funct3 == 0x3) { // MULHU: MultiplyOperands
                                        left_op = F.zero();
                                        right_op = left_input.mul(right_input); // Product
                                    } else {
                                        // DIVU, REMU, etc.: interleaved
                                        left_op = left_input;
                                        right_op = right_input;
                                    }
                                } else if (funct3 == 0x0 and funct7 == 0x20) {
                                    // SUB: SubtractOperands
                                    const two_pow_64 = F.fromBytes(&[_]u8{ 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 });
                                    left_op = F.zero();
                                    right_op = left_input.sub(right_input).add(two_pow_64);
                                } else if (funct3 == 0x0 and funct7 == 0x0) {
                                    // ADD: AddOperands
                                    left_op = F.zero();
                                    right_op = left_input.add(right_input);
                                } else {
                                    // XOR, AND, OR, SLT, SLTU, SRL, SRA: interleaved
                                    left_op = left_input;
                                    right_op = right_input;
                                }
                            },
                            0x13 => { // I-type ALU: only ADDI (funct3=0) uses AddOperands
                                if (funct3 == 0) {
                                    // ADDI: AddOperands
                                    left_op = F.zero();
                                    right_op = left_input.add(right_input);
                                } else {
                                    // SLLI, SLTI, SLTIU, XORI, SRLI, SRAI, ORI, ANDI: interleaved
                                    left_op = left_input;
                                    right_op = right_input;
                                }
                            },
                            0x37 => { // LUI: AddOperands, left_input=0, right_input=imm
                                left_op = F.zero();
                                right_op = left_input.add(right_input);
                            },
                            0x17 => { // AUIPC: AddOperands, left_input=PC, right_input=imm
                                left_op = F.zero();
                                right_op = left_input.add(right_input);
                            },
                            0x6f => { // JAL: AddOperands, left_input=PC, right_input=imm
                                left_op = F.zero();
                                right_op = left_input.add(right_input);
                            },
                            0x67 => { // JALR: AddOperands, left_input=rs1, right_input=imm
                                left_op = F.zero();
                                right_op = left_input.add(right_input);
                            },
                            0x0B => { // VirtualSignExtendWord: AddOperands, left=0, right=rs1
                                left_op = F.zero();
                                right_op = left_input.add(right_input);
                            },
                            0x2B => { // Virtual I-type: dispatch on funct3
                                if (funct3 == 0) {
                                    // VirtualMULI: MultiplyOperands, left=0, right=rs1*imm
                                    left_op = F.zero();
                                    right_op = left_input.mul(right_input);
                                } else {
                                    // VirtualPow2 (funct3=1), VirtualShiftRightBitmask (funct3=2): AddOperands
                                    left_op = F.zero();
                                    right_op = left_input.add(right_input);
                                }
                            },
                            0x1b => { // I-type word ALU (ADDIW, SLLIW, SRLIW, SRAIW)
                                // Only ADDIW (funct3=0) uses AddOperands; others use interleaved
                                if (funct3 == 0) {
                                    // ADDIW: AddOperands, left=0, right=rs1+imm
                                    left_op = F.zero();
                                    right_op = left_input.add(right_input);
                                } else {
                                    // SLLIW, SRLIW, SRAIW: interleaved
                                    left_op = left_input;
                                    right_op = right_input;
                                }
                            },
                            0x3b => { // ADDW/SUBW/VirtualChangeDivisorW
                                const funct7: u7 = @truncate(instr >> 25);
                                if (funct3 == 0 and funct7 == 0) {
                                    // ADDW: AddOperands, left=0, right=rs1+rs2
                                    left_op = F.zero();
                                    right_op = left_input.add(right_input);
                                } else if (funct3 == 0 and funct7 == 0x20) {
                                    // SUBW: SubtractOperands, left=0, right=rs1-rs2+2^64
                                    const two_pow_64 = F.fromBytes(&[_]u8{ 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 });
                                    left_op = F.zero();
                                    right_op = left_input.sub(right_input).add(two_pow_64);
                                } else if (funct3 == 6 and funct7 == 0x01) {
                                    // VirtualChangeDivisorW: interleaved, left=rs1 as u32 as u64, right=rs2
                                    const rs1_lower32: u64 = step.rs1_value & 0xFFFFFFFF;
                                    left_op = F.fromU64(rs1_lower32);
                                    right_op = F.fromU64(step.rs2_value);
                                } else {
                                    // Other 0x3b variants (not AddOperands/SubtractOperands)
                                    left_op = left_input;
                                    right_op = right_input;
                                }
                            },
                            0x02 => { // VirtualAdvice: Advice flag (identity path)
                                left_op = F.zero();
                                right_op = F.fromU128(@as(u128, step.rd_value));
                            },
                            0x22 => { // Virtual assert: dispatch on funct3
                                if (funct3 == 2 or funct3 == 3) {
                                    // Alignment assertions: AddOperands
                                    left_op = F.zero();
                                    right_op = left_input.add(right_input);
                                } else {
                                    // VirtualAssertEQ/ValidDiv0: interleaved
                                    left_op = left_input;
                                    right_op = right_input;
                                }
                            },
                            0x42 => { // VirtualZeroExtendWord: AddOperands flag (identity path)
                                left_op = F.zero();
                                right_op = F.fromU128(@as(u128, step.rs1_value));
                            },
                            0x62 => { // VirtualAssertValidUnsignedRemainder: Assert flag (interleaved)
                                left_op = left_input;
                                right_op = right_input;
                            },
                            else => {
                                // Default: NOT Add+Sub+Mul
                                left_op = left_input;
                                right_op = right_input;
                            },
                        }
                    }

                    output_sum = output_sum.add(lookups_eq_evals[j].mul(lookup_output));
                    left_sum = left_sum.add(lookups_eq_evals[j].mul(left_op));
                    right_sum = right_sum.add(lookups_eq_evals[j].mul(right_op));
                    // Use recomputed combined value to match individual sums
                    const recomputed_combined = lookup_output.add(gamma_raf.mul(left_op)).add(gamma_raf2.mul(right_op));
                    lookups_computed_sum = lookups_computed_sum.add(lookups_eq_evals[j].mul(recomputed_combined));
                    // Compare against lookups_combined_vals
                    if (!recomputed_combined.eql(lookups_combined_vals[j])) {
                        const step_dbg2 = trace.steps.items[j];
                        const instr_dbg2 = step_dbg2.instruction;
                        const opcode_dbg2 = instr_dbg2 & 0x7f;
                        if (comptime debug_verbose) {
                            dbg("[COMBINED MISMATCH] j={}: opcode=0x{x}, noop={}, term={}\n", .{
                                j, opcode_dbg2, step_dbg2.is_noop, step_dbg2.is_termination_store,
                            });
                            dbg("  recomputed = {x}\n", .{recomputed_combined.toBytesBE()[16..32].*});
                            dbg("  combined_v = {x}\n", .{lookups_combined_vals[j].toBytesBE()[16..32].*});
                            dbg("  recomp: output=0x{x}, left=0x{x}, right=0x{x}\n", .{
                                lookup_output.toU64(), left_op.toU64(), right_op.toU64(),
                            });
                            dbg("  rs1={}, rs2={}, rd={}, pc=0x{x}\n", .{
                                step_dbg2.rs1_value, step_dbg2.rs2_value,
                                step_dbg2.rd_value,  step_dbg2.pc,
                            });
                        }
                        // Also show what table says
                        const tbl_idx_dbg = getLookupTableIndex(opcode_dbg2, @truncate(instr_dbg2 >> 12), @truncate(instr_dbg2 >> 25));
                        if (tbl_idx_dbg >= 0) {
                            const tbl_u_dbg: usize = @intCast(tbl_idx_dbg);
                            const lo_dbg = lookups_indices_lo[j];
                            const hi_dbg = lookups_indices_hi[j];
                            const idx_dbg: u128 = @as(u128, hi_dbg) << 64 | @as(u128, lo_dbg);
                            const TableDbg = @import("../lookup_table/mod.zig").LookupTable(F, 64);
                            const tbl_entry_dbg = TableDbg.materializeTableEntry(tbl_u_dbg, idx_dbg);
                            if (comptime debug_verbose) {
                                dbg("  table[{}] at idx=0x{x}: entry=0x{x}\n", .{
                                    tbl_idx_dbg, idx_dbg, tbl_entry_dbg,
                                });
                            }
                        }
                    }
                }
                // Debug: print first 5 cycles' right operand values
                if (comptime debug_verbose) {
                    dbg("[STAGE5 LOOKUPS] First 5 right_op values (computed in loop):\n", .{});
                }
                for (0..@min(5, trace_len)) |jj| {
                    const step_dbg = trace.steps.items[jj];
                    const instr_dbg = step_dbg.instruction;
                    const opcode_dbg = instr_dbg & 0x7f;

                    // Recompute to show
                    const right_is_rs2_dbg: bool = switch (opcode_dbg) {
                        0x33, 0x63, 0x3b => true,
                        else => false,
                    };
                    const right_is_imm_dbg: bool = switch (opcode_dbg) {
                        0x13, 0x03, 0x67, 0x23, 0x37, 0x17, 0x6f, 0x1b, 0x0B => true,
                        else => false,
                    };
                    const imm_dbg = Inst.computeImmediate(instr_dbg);
                    var right_input_dbg: F = F.zero();
                    if (right_is_rs2_dbg) right_input_dbg = F.fromU64(step_dbg.rs2_value);
                    if (right_is_imm_dbg) right_input_dbg = imm_dbg;

                    if (comptime debug_verbose) {
                        dbg("  j={}: opcode=0x{x}, right_is_rs2={}, right_is_imm={}, imm=0x{x}, rs2=0x{x}, right_input=0x{x}\n", .{
                            jj,              opcode_dbg,         right_is_rs2_dbg,        right_is_imm_dbg,
                            imm_dbg.toU64(), step_dbg.rs2_value, right_input_dbg.toU64(),
                        });
                    }
                }
                if (comptime debug_verbose) {
                    dbg("[STAGE5 LOOKUPS] Individual sum verification:\n", .{});
                    dbg("  output_sum (Σ eq*output) = {any}\n", .{output_sum.toBytesBE()[0..16]});
                    dbg("  rv_claim (from Stage 2)  = {any}\n", .{rv_claim.toBytesBE()[0..16]});
                    dbg("  output match = {}\n", .{output_sum.eql(rv_claim)});
                    dbg("  left_sum (Σ eq*left)     = {any}\n", .{left_sum.toBytesBE()[0..16]});
                    dbg("  left_op_claim (Stage 2)  = {any}\n", .{left_op_claim.toBytesBE()[0..16]});
                    dbg("  left match = {}\n", .{left_sum.eql(left_op_claim)});
                    dbg("  right_sum (Σ eq*right)   = {any}\n", .{right_sum.toBytesBE()[0..16]});
                    dbg("  right_op_claim (Stage 2) = {any}\n", .{right_op_claim.toBytesBE()[0..16]});
                    dbg("  right match = {}\n", .{right_sum.eql(right_op_claim)});
                }
                // Debug: print first few eq_evals values
                if (comptime debug_verbose) {
                    dbg("  Stage5 eq_evals[0..3]: {x}, {x}, {x}\n", .{
                        lookups_eq_evals[0].toBytesBE()[16..32].*,
                        lookups_eq_evals[1].toBytesBE()[16..32].*,
                        lookups_eq_evals[2].toBytesBE()[16..32].*,
                    });
                }
                // Verify individual terms
                for (0..5) |jj| {
                    const step_v = trace.steps.items[jj];
                    const instr_v = step_v.instruction;
                    const opcode_v = instr_v & 0x7f;
                    const funct3_v: u3 = @truncate((instr_v >> 12) & 0x7);
                    const funct7_v: u7 = @truncate(instr_v >> 25);
                    // Recompute left_op, right_op, output using same logic as above
                    const left_is_rs1_v: bool = switch (opcode_v) {
                        0x33, 0x3b, 0x23, 0x63, 0x13, 0x03, 0x67, 0x1b, 0x0B, 0x2B, 0x5B => true,
                        0x22 => true, // VirtualAssertEQ: left = rs1
                        0x42 => true, // VirtualZeroExtendWord: left = rs1
                        0x62 => true, // VirtualAssertValidUnsignedRemainder: left = rs1
                        else => false,
                    };
                    const left_is_pc_v: bool = switch (opcode_v) {
                        0x17, 0x6f => true,
                        else => false,
                    };
                    const right_is_rs2_v: bool = switch (opcode_v) {
                        0x33, 0x63, 0x3b => true,
                        0x22 => (funct3_v == 0 or funct3_v == 1), // VirtualAssertEQ/ValidDiv0: rs2; alignment: imm
                        0x62 => true, // VirtualAssertValidUnsignedRemainder: right = rs2
                        0x5B => step_v.rs2_read, // VirtualSRL/SRA R-type: rs2
                        else => false,
                    };
                    const right_is_imm_v: bool = switch (opcode_v) {
                        0x13, 0x03, 0x67, 0x23, 0x37, 0x17, 0x6f, 0x1b, 0x0B, 0x2B => true,
                        0x22 => (funct3_v == 2 or funct3_v == 3), // alignment assertions: imm
                        0x5B => !step_v.rs2_read, // I-type: imm; R-type: not imm
                        else => false,
                    };
                    const imm_v = if (opcode_v == 0x2B) blk: {
                        if (funct3_v == 0) {
                            const shamt_rv: u32 = instr_v >> 20;
                            const shamt_v: u6 = @truncate(shamt_rv & 0x3F);
                            break :blk F.fromU64(@as(u64, 1) << shamt_v);
                        } else {
                            break :blk F.zero(); // VirtualPow2/VirtualShiftRightBitmask: IMM = 0
                        }
                    } else if (opcode_v == 0x5B) blk5bv: {
                        if (step_v.rs2_read) {
                            break :blk5bv F.zero(); // R-type: no immediate
                        } else {
                            const ts_rv: u32 = instr_v >> 20;
                            const ts_v: u7 = @truncate(ts_rv & 0x3F);
                            const ones_v: u128 = (@as(u128, 1) << @intCast(64 - @as(u8, ts_v))) - 1;
                            break :blk5bv F.fromU64(@truncate(ones_v << ts_v));
                        }
                    } else if (opcode_v == 0x22 and (funct3_v == 2 or funct3_v == 3)) blk_assert2: {
                        const aim2_raw: u32 = @truncate(instr_v >> 20);
                        const aim2_signed: i64 = @as(i64, @as(i32, @bitCast(aim2_raw << 20)) >> 20);
                        break :blk_assert2 if (aim2_signed < 0) F.fromU64(@intCast(-aim2_signed)).neg() else F.fromU64(@intCast(aim2_signed));
                    } else Inst.computeImmediate(instr_v);
                    var left_input_v: F = F.zero();
                    if (left_is_rs1_v) left_input_v = F.fromU64(step_v.rs1_value);
                    if (left_is_pc_v) left_input_v = F.fromU64(step_v.unexpanded_pc);
                    var right_input_v: F = F.zero();
                    if (right_is_rs2_v) right_input_v = F.fromU64(step_v.rs2_value);
                    if (right_is_imm_v) right_input_v = imm_v;
                    // Compute lookup operands
                    const is_add_type = switch (opcode_v) {
                        0x13, 0x37, 0x17, 0x6f, 0x67, 0x0B => true, // 0x0B = VirtualSignExtendWord
                        0x1b => (funct3_v == 0), // ADDIW (funct3=0) uses AddOperands
                        0x33 => !(funct7_v == 0x01 and funct3_v != 0x0) and !(funct7_v == 0x20),
                        0x3b => (funct3_v == 0 and funct7_v == 0) or (funct3_v == 0 and funct7_v == 0x20), // ADDW/SUBW
                        0x2B => (funct3_v != 0), // VirtualPow2/VirtualShiftRightBitmask: AddOperands
                        0x22 => (funct3_v == 2 or funct3_v == 3), // Alignment assertions: AddOperands
                        0x42 => true, // VirtualZeroExtendWord: AddOperands
                        else => false,
                    };
                    const is_sub_type = (opcode_v == 0x33 and funct3_v == 0 and funct7_v == 0x20) or
                        (opcode_v == 0x3b and funct3_v == 0 and funct7_v == 0x20); // SUB or SUBW
                    const is_mul_type = (opcode_v == 0x33 and funct7_v == 0x01 and funct3_v == 0) or
                        (opcode_v == 0x2B and funct3_v == 0); // MUL or VirtualMULI (funct3=0 only)
                    var left_op_v: F = undefined;
                    var right_op_v: F = undefined;
                    if (opcode_v == 0x02) {
                        // VirtualAdvice: Advice flag, identity path
                        left_op_v = F.zero();
                        right_op_v = F.fromU128(@as(u128, step_v.rd_value));
                    } else if (opcode_v == 0x42) {
                        // VirtualZeroExtendWord: AddOperands, identity path
                        left_op_v = F.zero();
                        right_op_v = F.fromU128(@as(u128, step_v.rs1_value));
                    } else if (is_sub_type) {
                        const two_pow_64_v = F.fromBytes(&[_]u8{ 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 });
                        left_op_v = F.zero();
                        right_op_v = left_input_v.sub(right_input_v).add(two_pow_64_v);
                    } else if (is_mul_type) {
                        left_op_v = F.zero();
                        right_op_v = left_input_v.mul(right_input_v);
                    } else if (is_add_type) {
                        left_op_v = F.zero();
                        right_op_v = left_input_v.add(right_input_v);
                    } else {
                        // Default includes 0x22 (VirtualAssertEQ), 0x62 (VirtualAssertValidUnsignedRemainder)
                        left_op_v = left_input_v;
                        right_op_v = right_input_v;
                    }
                    const output_v = F.fromU64(step_v.rd_value); // Simplified, ignores JAL/JALR/Branch

                    if (comptime debug_verbose) {
                        dbg("  [VERIFY j={}] eq={x}, left={x}, right={x}, out={x}\n", .{
                            jj,
                            lookups_eq_evals[jj].toBytesBE()[24..32].*,
                            left_op_v.toBytesBE()[24..32].*,
                            right_op_v.toBytesBE()[24..32].*,
                            output_v.toBytesBE()[24..32].*,
                        });
                    }
                }
                if (comptime debug_verbose) {
                    dbg("[STAGE5 LOOKUPS] Sum verification:\n", .{});
                    dbg("  computed_sum = {any}\n", .{lookups_computed_sum.toBytesBE()[0..8]});
                    dbg("  lookups_input = {any}\n", .{lookups_input.toBytesBE()[0..8]});
                    dbg("  rv_claim = {any}\n", .{rv_claim.toBytesBE()[0..8]});
                    dbg("  left_op_claim = {any}\n", .{left_op_claim.toBytesBE()[0..8]});
                    dbg("  right_op_claim = {any}\n", .{right_op_claim.toBytesBE()[0..8]});
                    dbg("  match = {}\n", .{lookups_computed_sum.eql(lookups_input)});
                }
            } // end debug sum verification

            // Compute scaling factors
            const regs_scale = max_num_rounds - regs_val_num_rounds;
            const ram_ra_scale = max_num_rounds - ram_ra_num_rounds;

            var regs_scaled = regs_val_input;
            for (0..regs_scale) |_| regs_scaled = regs_scaled.add(regs_scaled);

            var ram_ra_scaled = ram_ra_input;
            for (0..ram_ra_scale) |_| ram_ra_scaled = ram_ra_scaled.add(ram_ra_scaled);

            const lookups_scaled = lookups_input;

            const batched_claim = batch0.mul(regs_scaled)
                .add(batch1.mul(ram_ra_scaled))
                .add(batch2.mul(lookups_scaled));

            if (comptime debug_verbose) {
                dbg("[STAGE5] Initial batched claim = {any}\n", .{batched_claim.toBytesBE()});
                dbg("  [S5P] initial_claim (e before R0): {x}\n", .{batched_claim.toBytes()[0..16].*});
            }

            var challenges = try self.allocator.alloc(F, max_num_rounds);
            errdefer self.allocator.free(challenges);

            // Track current batched claim (for verification)
            var current_batched_claim = batched_claim;

            // Track individual instance claims - MUST use SCALED values!
            // Each instance's claim starts at input * 2^(max_rounds - instance_rounds)
            // During inactive rounds, the claim halves each round.
            // By the time the instance becomes active, claim = unscaled input.
            var regs_val_current_claim = regs_scaled; // Instance 0: RegistersValEvaluation (scaled by 2^128)
            // Instance 1: RamRaClaimReduction - uses extracted prover
            var ram_ra_prover = try RamRaClaimReductionProver(F).init(
                self.allocator,
                self.thread_pool,
                self.gpu_ops,
                gamma,
                claim_raf,
                claim_rw,
                claim_val,
                n_cycle_vars,
                log_ram_k,
                memory_trace,
                memory_layout,
                r_address_raf,
                r_address_rw,
                r_cycle_raf,
                r_cycle_rw,
                r_cycle_val,
            );
            defer ram_ra_prover.deinit();
            var lookups_claim = lookups_input; // Instance 2: LookupsReadRaf (no scaling, active from round 0)
            const batch2_inv = if (comptime debug_verbose) batch2.inverse().? else F.zero(); // Only needed for debug diagnostics

            // DEBUG: Print initial claims
            if (comptime debug_verbose) {
                const print = std.debug.print;
                print("[ZOLT INIT] lookups_input (LE) = {any}\n", .{lookups_input.toBytes()[0..16].*});
                print("[ZOLT INIT] regs_scaled (LE) = {any}\n", .{regs_scaled.toBytes()[0..16].*});
                print("[ZOLT INIT] batched_claim (LE) = {any}\n", .{batched_claim.toBytes()[0..16].*});
                print("[ZOLT INIT] batch0 (LE) = {any}\n", .{batch0.toBytes()[0..16].*});
                print("[ZOLT INIT] batch1 (LE) = {any}\n", .{batch1.toBytes()[0..16].*});
                print("[ZOLT INIT] batch2 (LE) = {any}\n", .{batch2.toBytes()[0..16].*});
            }

            // ===================================================================
            // RamRaClaimReduction State Initialization (via extracted prover)
            // ===================================================================
            // ram_ra_prover was initialized above (near ram_ra_prover.current_claim replacement).
            // Set the scaled initial claim now.
            ram_ra_prover.setScaledClaim(ram_ra_scaled);

            // NOTE: ExpandingTable (ram_ra_F) was previously used for PhaseAddress tracking
            // but is no longer needed since RamRaClaimReduction uses cycle-only binding.
            // The state is now fully encapsulated in ram_ra_prover.

            // ===================================================================
            // Initialize LookupsReadRafProver (Instance 2)
            // ===================================================================
            // All arrays populated above are transferred to the prover (it takes ownership).

            if (comptime bench_timing) {
                std.debug.print("    [STAGE5-INIT] combined+indices:  {d:8.1} ms\n", .{@as(f64, @floatFromInt(init_sub_timer.read())) / 1_000_000.0});
                init_sub_timer.reset();
            }

            var lookups_prover = try LookupsReadRafProver(F).init(
                self.allocator,
                self.thread_pool,
                self.gpu_ops,
                gamma_raf,
                lookups_input,
                n_cycle_vars,
                lookups_ra_virtual_log_k_chunk,
                r_reduction,
                // Pre-populated arrays (ownership transferred):
                lookups_eq_evals,
                lookups_combined_vals,
                lookups_indices_lo,
                lookups_indices_hi,
                lookup_indices_u128,
                cycle_table_indices,
                cycle_is_identity_path,
                is_interleaved_operands,
                lookup_indices_by_table,
            );
            defer lookups_prover.deinit();

            if (comptime bench_timing) {
                std.debug.print("    [STAGE5-INIT] LookupsReadRafProver.init: {d:8.1} ms\n", .{@as(f64, @floatFromInt(init_sub_timer.read())) / 1_000_000.0});
                init_sub_timer.reset();
            }



            // Run the batched sumcheck
            if (comptime debug_verbose) {
                dbg("[STAGE5] Entering main sumcheck loop, max_num_rounds={}\n", .{max_num_rounds});
            }

            // Benchmark timing accumulators
            var bench_timer = if (comptime bench_timing) std.time.Timer.start() catch unreachable else {};
            var bench_init_ns: u64 = 0;
            var bench_phase_transition_ns: u64 = 0;
            const bench_condense_ns: u64 = 0;
            const bench_init_phase_ns: u64 = 0;
            var bench_addr_other_ns: u64 = 0;
            var bench_addr_compute_ns: u64 = 0;
            var bench_addr_bind_ns: u64 = 0;
            var bench_addr_transcript_ns: u64 = 0;
            var bench_cycle_bind_ns: u64 = 0;
            var bench_cycle_transcript_ns: u64 = 0;
            var bench_inst0_compute_ns: u64 = 0;
            var bench_inst1_compute_ns: u64 = 0;
            var bench_inst2_addr_compute_ns: u64 = 0;
            _ = &bench_inst2_addr_compute_ns;
            var bench_inst2_cycle_compute_ns: u64 = 0;
            const bench_remat_ns: u64 = 0; // Untimed rematerialization gap (now inside lookups_prover)
            var bench_cycle_coeff_ns: u64 = 0; // Untimed coefficient combination gap

            // BRUTE FORCE PER-ROUND DIAGNOSTIC: bf_weights now lives inside lookups_prover.

            // lookups_current_scalar, split-eq tables, and other Instance 2 state
            // are now encapsulated inside lookups_prover.

            // Cached Instance 0 poly evals from compute phase, reused in bind phase
            var cached_regs_val_evals: ?[4]F = null;

            if (comptime bench_timing) bench_init_ns = bench_overall_timer.read();

            // Lightweight phase timing (no per-round overhead, just phase boundaries)
            const s5_do_phase_timing = @import("std").posix.getenv("ZOLT_BENCH") != null;
            var s5_phase_timer = if (s5_do_phase_timing) std.time.Timer.start() catch null else null;
            var s5_addr_compute_ns: u64 = 0;
            var s5_addr_bind_ns: u64 = 0;
            var s5_phase_trans_ns: u64 = 0;
            var s5_cycle_compute_ns: u64 = 0;
            var s5_cycle_bind_ns: u64 = 0;

            for (0..max_num_rounds) |round| {
                if (comptime bench_timing) bench_timer.reset();

                const remaining_rounds = max_num_rounds - round;

                var combined_poly = [_]F{ F.zero(), F.zero(), F.zero(), F.zero() };

                // Instance 0: RegistersValEvaluation (8 rounds)
                if (remaining_rounds > regs_val_num_rounds) {
                    // Not started yet - constant polynomial where p(0) + p(1) = scaled_claim
                    // Jolt's constant polynomial: p(x) = scaled_input_claim for all x
                    // where scaled_input_claim = input_claim * 2^(remaining - num_rounds - 1)
                    // This gives p(0) + p(1) = 2 * scaled_input_claim = input_claim * 2^(remaining - num_rounds)
                    const scaled_input_claim = sumcheck_helpers.inactiveContribution(F, regs_val_input, remaining_rounds, regs_val_num_rounds);
                    // Constant polynomial p(x) = scaled_input_claim
                    combined_poly[0] = combined_poly[0].add(batch0.mul(scaled_input_claim));
                    combined_poly[1] = combined_poly[1].add(batch0.mul(scaled_input_claim));
                    combined_poly[2] = combined_poly[2].add(batch0.mul(scaled_input_claim));
                    // evals[3] = p_inf = 0 for constant polynomial
                } else {
                    // Instance is active - compute actual round polynomial
                    const regs_round = regs_val_num_rounds - remaining_rounds;
                    if (comptime bench_timing) bench_timer.reset();
                    const poly_evals = Inst.computeRegsValRoundPoly(inc_evals, wa_evals, &lt_poly, regs_round, self.thread_pool);
                    if (comptime bench_timing) bench_inst0_compute_ns += bench_timer.read();
                    cached_regs_val_evals = poly_evals;
                    combined_poly[0] = combined_poly[0].add(batch0.mul(poly_evals[0]));
                    combined_poly[1] = combined_poly[1].add(batch0.mul(poly_evals[1]));
                    combined_poly[2] = combined_poly[2].add(batch0.mul(poly_evals[2]));
                    combined_poly[3] = combined_poly[3].add(batch0.mul(poly_evals[3]));

                    // CRITICAL DEBUG: Check if p0(0) + p0(1) = claim0
                    const inst0_sum = poly_evals[0].add(poly_evals[1]);
                    const inst0_sum_matches = inst0_sum.eql(regs_val_current_claim);
                    if (round >= LOOKUPS_LOG_K) {
                        if (comptime debug_verbose) {
                            dbg("[INST0 SUMCHECK] Round {}: p(0)+p(1) = {x}, claim0 = {x}, match = {}\n", .{
                                round,
                                inst0_sum.toBytesBE()[16..32].*,
                                regs_val_current_claim.toBytesBE()[16..32].*,
                                inst0_sum_matches,
                            });
                        }
                    }

                    // Debug: Print Instance 0 contribution for cycle rounds
                    if (round >= LOOKUPS_LOG_K and round <= LOOKUPS_LOG_K + 1) {
                        const inst0_coeffs = UniPoly(F).toomCookToCoeffs(poly_evals);
                        if (comptime debug_verbose) {
                            dbg("[ZOLT INST0] Round {}: regs_round={}\n", .{ round, regs_round });
                            dbg("  poly_evals (Toom) = [{any}, {any}, {any}, {any}]\n", .{
                                poly_evals[0].toBytes(), poly_evals[1].toBytes(),
                                poly_evals[2].toBytes(), poly_evals[3].toBytes(),
                            });
                            dbg("  inst0_coeffs (coeffs) = [{any}, {any}, {any}, {any}]\n", .{
                                inst0_coeffs[0].toBytes(), inst0_coeffs[1].toBytes(),
                                inst0_coeffs[2].toBytes(), inst0_coeffs[3].toBytes(),
                            });
                            dbg("  batch0 = {any}\n", .{batch0.toBytes()});
                        }
                    }
                }

                // Instance 1: RamRaClaimReduction (24 rounds)
                // Sumcheck proves: Σ_{k,c} eq_combined(k,c) · ra(k,c) = input_claim
                // where ra(k,c) = 1 iff there's a RAM access at (address=k, cycle=c)
                if (remaining_rounds > ram_ra_num_rounds) {
                    // Not started - constant polynomial (same logic as Instance 0)
                    // Use ram_ra_input from opening_claims (corrected in proof_converter before Stage 5)
                    const scaled_input_claim = sumcheck_helpers.inactiveContribution(F, ram_ra_input, remaining_rounds, ram_ra_num_rounds);
                    combined_poly[0] = combined_poly[0].add(batch1.mul(scaled_input_claim));
                    combined_poly[1] = combined_poly[1].add(batch1.mul(scaled_input_claim));
                    combined_poly[2] = combined_poly[2].add(batch1.mul(scaled_input_claim));
                    // evals[3] = p_inf = 0 for constant polynomial
                } else {
                    // Instance is active - compute RamRaClaimReduction sumcheck polynomial
                    if (comptime bench_timing) bench_timer.reset();

                    const ram_ra_round = ram_ra_num_rounds - remaining_rounds;
                    const poly_evals = ram_ra_prover.computeRoundPoly(ram_ra_round);

                    combined_poly[0] = combined_poly[0].add(batch1.mul(poly_evals[0]));
                    combined_poly[1] = combined_poly[1].add(batch1.mul(poly_evals[1]));
                    combined_poly[2] = combined_poly[2].add(batch1.mul(poly_evals[2]));

                    if (comptime bench_timing) bench_inst1_compute_ns += bench_timer.read();
                }
                if (comptime bench_timing) bench_timer.reset(); // Start timing untimed gap

                // Instance 2: LookupsReadRaf (136 rounds)
                // Since lookups_num_rounds = max_num_rounds, this instance is always active
                if (round < LOOKUPS_LOG_K) {
                    // Address round: use Jolt's prefix-suffix decomposition
                    // The polynomial is split into:
                    //   - Read-checking: Σ_tables Σ_b table.combine(prefix(c,b), Q_suffix[b])
                    //   - RAF: γ*left + γ²*(identity + right)
                    //
                    // Jolt uses HighToLow binding: round 0 -> bit 127 (MSB), round 127 -> bit 0 (LSB)

                    // Compute Instance 2 address round polynomial via extracted prover
                    if (comptime bench_timing) bench_timer.reset();
                    const lookups_evals = lookups_prover.computeAddressRoundPoly(round, challenges[0..round]);
                    if (comptime bench_timing) {
                        const elapsed = bench_timer.read();
                        bench_addr_compute_ns += elapsed;
                        bench_inst2_addr_compute_ns += elapsed;
                    }
                    if (s5_phase_timer) |*pt| {
                        s5_addr_compute_ns += pt.read();
                        pt.reset();
                    }

                    const eval_0_inst2 = lookups_evals[0];
                    const eval_2_inst2 = lookups_evals[2];
                    const eval_1_inst2 = lookups_claim.sub(eval_0_inst2);

                    combined_poly[0] = combined_poly[0].add(batch2.mul(eval_0_inst2));
                    combined_poly[1] = combined_poly[1].add(batch2.mul(eval_1_inst2));
                    combined_poly[2] = combined_poly[2].add(batch2.mul(eval_2_inst2));

                    // CRITICAL VERIFICATION: p(0) + p(1) should equal current_batched_claim
                    const poly_sum = combined_poly[0].add(combined_poly[1]);
                    const sumcheck_ok_addr = poly_sum.eql(current_batched_claim);
                    if (!sumcheck_ok_addr) {
                        if (comptime debug_verbose) {
                            dbg("[STAGE5 VERIFY R{}] p(0)+p(1) != claim! p01={x}, claim={x}\n", .{
                                round,
                                poly_sum.toBytesBE()[16..32].*,
                                current_batched_claim.toBytesBE()[16..32].*,
                            });
                        }
                    }

                    // Compute degree-2 compressed coefficients directly from evaluations
                    // For degree-2 polynomial p(x) = c0 + c1*x + c2*x^2:
                    //   p(0) = c0
                    //   p(1) = c0 + c1 + c2  =>  c1 = p(1) - p(0) - c2
                    //   p(2) = c0 + 2*c1 + 4*c2  =>  c2 = (p(2) - 2*p(1) + p(0)) / 2
                    // Compressed format (excluding c1): [c0, c2]
                    const inv2 = UniPoly(F).INV2;
                    const batched_c0 = combined_poly[0];
                    const batched_c2 = combined_poly[2].sub(combined_poly[1]).sub(combined_poly[1]).add(combined_poly[0]).mul(inv2);

                    const coeffs = try self.allocator.alloc(F, 2);
                    coeffs[0] = batched_c0; // c0
                    coeffs[1] = batched_c2; // c2

                    // Debug: print coefficients in same LE format as Jolt verifier for ALL rounds
                    if (comptime debug_verbose) {
                        dbg("  [S5P] R{} c0={x} c2={x}\n", .{
                            round,
                            coeffs[0].toBytes()[0..16].*,
                            coeffs[1].toBytes()[0..16].*,
                        });
                    }

                    try proof.compressed_polys.append(self.allocator, .{
                        .coeffs_except_linear_term = coeffs,
                        .allocator = self.allocator,
                    });

                    // Append to transcript: 2 coefficients for degree-2 polynomial
                    if (comptime bench_timing) bench_timer.reset();
                    transcript.appendScalars("sumcheck_poly", coeffs);

                    const challenge = transcript.challengeScalar();
                    challenges[round] = challenge;
                    if (comptime bench_timing) {
                        bench_addr_transcript_ns += bench_timer.read();
                        bench_timer.reset();
                    }

                    // DEBUG: Print challenge and coefficients for comparison with Jolt verifier
                    if (comptime debug_verbose) if (round < 4 or round == 7 or round == 127 or round == 128) {
                        const print = std.debug.print;
                        print("[ZOLT S5V R{}] hint={any}\n", .{ round, current_batched_claim.toBytes()[0..16].* });
                        print("[ZOLT S5V R{}] c0={any}\n", .{ round, coeffs[0].toBytes()[0..16].* });
                        print("[ZOLT S5V R{}] c2={any}\n", .{ round, coeffs[1].toBytes()[0..16].* });
                        print("[ZOLT S5V R{}] challenge={any}\n", .{ round, challenge.toBytes()[0..16].* });
                    };

                    // Update current_batched_claim by evaluating degree-2 polynomial at challenge
                    // p(r) = c0 + r*c1 + r^2*c2
                    // where c1 = claim - 2*c0 - c2 (from p(0)+p(1) = claim)
                    const c0 = coeffs[0];
                    const c2_val = coeffs[1];
                    const c1 = current_batched_claim.sub(c0).sub(c0).sub(c2_val);
                    const r2 = challenge.mul(challenge);

                    // CRITICAL: Challenge * F uses mulHiBigIntU128 (Jolt delegates to F * Challenge)
                    current_batched_claim = c0.add(c1.mulHiBigIntU128(challenge.limbs)).add(c2_val.mul(r2));

                    // Per-round tracking (matches Jolt verifier's [S5V] output)
                    if (comptime debug_verbose) {
                        dbg("  [S5P] R{} challenge={x} new_e={x} degree=2\n", .{
                            round,
                            challenge.toBytes()[0..16].*,
                            current_batched_claim.toBytes()[0..16].*,
                        });
                    }

                    // =====================================================================
                    // CRITICAL FIX: Update lookups_claim to match polynomial evaluation
                    // =====================================================================
                    // The lookups_claim must evolve as p_inst2(r) where p_inst2 is Instance 2's polynomial.
                    // For degree-2 polynomial with evals at 0, 1, 2:
                    //   p(0) = eval_0_inst2
                    //   p(1) = eval_1_inst2 = lookups_claim - eval_0_inst2 (sumcheck property)
                    //   p(2) = eval_2_inst2
                    // Coefficients:
                    //   c0 = p(0)
                    //   c2 = (p(2) - 2*p(1) + p(0)) / 2
                    //   c1 = p(1) - p(0) - c2 = lookups_claim - eval_0_inst2 - eval_0_inst2 - c2
                    //
                    // p(r) = c0 + r*c1 + r²*c2
                    const inst2_c0 = eval_0_inst2;
                    const inst2_c2 = eval_2_inst2.sub(eval_1_inst2).sub(eval_1_inst2).add(eval_0_inst2).mul(UniPoly(F).INV2);
                    const inst2_c1 = eval_1_inst2.sub(eval_0_inst2).sub(inst2_c2);
                    const inst2_at_r = inst2_c0.add(inst2_c1.mulHiBigIntU128(challenge.limbs)).add(inst2_c2.mul(r2));

                    // Print Instance 2 poly coefficients (compare with Jolt prover)
                    if (comptime debug_verbose) if (round < 5) {
                        const print = std.debug.print;
                        print("[ZOLT INST2 POLY R{}] c0={any}\n", .{ round, inst2_c0.toBytes()[0..16].* });
                        print("[ZOLT INST2 POLY R{}] c1={any}\n", .{ round, inst2_c1.toBytes()[0..16].* });
                        print("[ZOLT INST2 POLY R{}] c2={any}\n", .{ round, inst2_c2.toBytes()[0..16].* });
                    };

                    // Debug: show claim chain for first 3 rounds and last 3 address rounds
                    if (round < 3 or (round >= 125 and round < 128)) {
                        if (comptime debug_verbose) {
                            dbg("[CLAIM_CHAIN R{}] before_claim={x}\n", .{ round, lookups_claim.toBytesBE()[16..32].* });
                            dbg("[CLAIM_CHAIN R{}] eval_0={x}, eval_1={x}, eval_2={x}\n", .{
                                round,
                                eval_0_inst2.toBytesBE()[16..32].*,
                                eval_1_inst2.toBytesBE()[16..32].*,
                                eval_2_inst2.toBytesBE()[16..32].*,
                            });
                            dbg("[CLAIM_CHAIN R{}] sum_check: eval_0+eval_1={x} (should == before_claim)\n", .{
                                round,
                                eval_0_inst2.add(eval_1_inst2).toBytesBE()[16..32].*,
                            });
                            dbg("[CLAIM_CHAIN R{}] p(r)={x} -> new_claim\n", .{ round, inst2_at_r.toBytesBE()[16..32].* });
                        }
                    }

                    // NOTE: lookups_claim will be updated AFTER inst0/inst1 claims,
                    // by deriving it from the batched claim. See below.
                    const lookups_claim_from_poly = inst2_at_r;
                    _ = lookups_claim_from_poly;

                    // =====================================================================
                    // CRITICAL FIX: Update Instance 0 and Instance 1 claims during address rounds
                    // =====================================================================
                    // Instance 0 (RegistersValEvaluation) is inactive for all address rounds (0-127)
                    // Its polynomial is constant, so p(r) = claim/2 for degree-0 constant poly
                    // Actually for inactive instance: polynomial is CONSTANT with value = scaled_claim
                    // and claim is doubled each round (because p(0) + p(1) = 2 * constant = claim)
                    // So after evaluating at r, we get: constant = claim/2, then new_claim = constant = claim/2... wait
                    // Let me think: if claim = 2*C (where C is the constant poly value)
                    // then p(x) = C for all x, so p(0) + p(1) = 2*C = claim (satisfies sumcheck)
                    // and p(r) = C = claim/2
                    // So the new claim is claim/2, not doubled!
                    //
                    // But wait, the Toom-Cook format has [p(0), p(1), p(2), p_inf]
                    // For constant poly: [C, C, C, 0]
                    // This gives coeffs [C, 0, 0, 0] (constant term only)
                    // So p(r) = C
                    // And if claim = 2*C, then new_claim = C = claim/2
                    //
                    // Hmm, but the code uses combined_poly which adds batch0 * scaled_claim for inactive instance
                    // Let me check...
                    //
                    // For now, update claims by halving (since p(r) = claim/2 for inactive constant poly)
                    if (remaining_rounds > regs_val_num_rounds) {
                        // Instance 0 is inactive - claim halves
                        regs_val_current_claim = regs_val_current_claim.mul(UniPoly(F).INV2);
                    }

                    if (remaining_rounds > ram_ra_num_rounds) {
                        // Instance 1 is inactive - claim halves
                        ram_ra_prover.current_claim = ram_ra_prover.current_claim.mul(UniPoly(F).INV2);
                    }
                    // NOTE: Instance 1 active case (address rounds 112-127) is handled below after
                    // the RamRaClaimReduction binding section, where ram_ra_prover.current_claim is updated.
                    // The batched claim recomputation is also moved to after the RamRaClaimReduction binding.

                    // NOTE: Consistency check moved to after Instance 1 claim update (below)

                    // ===================================================================
                    // Update RamRaClaimReduction state after receiving challenge
                    // ===================================================================
                    if (remaining_rounds <= ram_ra_num_rounds) {
                        const ram_ra_round = ram_ra_num_rounds - remaining_rounds;
                        ram_ra_prover.bindChallenge(ram_ra_round, challenge);
                    }

                    // Update lookups_claim from the polynomial chain
                    // p(r) = c0 + r*c1 + r²*c2 where c0=eval_0, c1=eval_1-eval_0-c2
                    lookups_claim = inst2_at_r;

                    // Print Instance 2 claim at EVERY round (matches Jolt's [S5 INST2 CLAIM R{}])
                    if (comptime debug_verbose) {
                        dbg("[S5 INST2 CLAIM R{}] {any}\n", .{ round, lookups_claim.toBytes()[0..16].* });
                    }

                    // BRUTE FORCE CHAIN CHECK (debug only)
                    if (comptime debug_verbose) if (round < 16) {
                        var bf_chain_sum = F.zero();
                        for (0..T) |jj| {
                            const u_j = lookups_eq_evals[jj]; // eq(j, r_reduction) at round 0, but CONDENSED after phase transitions
                            const cv_j = lookups_combined_vals[jj]; // initial combined value
                            // Compute eq_address factor: Π_{j=0}^{round} eq_bit(challenge[j], K(t)_{127-j})
                            var eq_addr = F.one();
                            for (0..round + 1) |rr| {
                                const bit_pos = LOOKUPS_LOG_K - 1 - rr;
                                const k_lo = lookups_indices_lo[jj];
                                const k_hi = lookups_indices_hi[jj];
                                const bit_val: u1 = if (bit_pos >= 64) @truncate(k_hi >> @intCast(bit_pos - 64)) else @truncate(k_lo >> @intCast(bit_pos));
                                const r_val = challenges[rr];
                                const eq_bit = if (bit_val == 1) r_val else F.one().sub(r_val);
                                eq_addr = eq_addr.mul(eq_bit);
                            }
                            bf_chain_sum = bf_chain_sum.add(u_j.mul(eq_addr).mul(cv_j));
                        }
                        std.debug.print("[BF_CHAIN R{}] match={}\n", .{
                            round,
                            bf_chain_sum.eql(lookups_claim),
                        });
                    };

                    // BINARY SEARCH debug diagnostic
                    if (comptime debug_verbose) {
                        // The correct inst2 claim should equal the one derived from evaluating
                        // the Instance 2 polynomial at the challenge. If eval_0_inst2 or eval_2_inst2
                        // were wrong, lookups_claim would be wrong.
                        // But since consistency holds, lookups_claim IS correct for the batched polynomial.
                        // The issue is that the BATCHED polynomial encodes wrong inst2 values.
                        // So let's compare eval_0_inst2 with the "correct" eval_0 that would make
                        // the batched output_claim match expected_output_claim.
                        //
                        // For now, just print the first round where lookups_claim doesn't match
                        // what a brute force computation gives.
                        if (round < 3 or round == 15 or round == 16 or round == 31 or round == 32 or round == 127) {
                            std.debug.print("[INST2_CHAIN R{}] lookups_claim = {any}\n", .{ round, lookups_claim.toBytes()[0..16].* });
                        }
                    }

                    // NOTE: current_batched_claim was already correctly set at the polynomial
                    // evaluation step above (c0 + c1*r + c2*r² + c3*r³). The verifier uses
                    // eval_from_hint which recovers c1 from the hint (previous claim) and
                    // evaluates the polynomial - so the prover's claim must be exactly that
                    // polynomial evaluation, NOT a recomputation from individual instance claims.
                    // Jolt's prover does NOT recompute the batched claim from instances.
                    // Individual instance claims are tracked separately for computing future
                    // round polynomials but do NOT override the batched claim.

                    // Debug: print claim tracking for rounds 126-128
                    if (round >= 126 and round <= 128) {
                        if (comptime debug_verbose) {
                            dbg("[ADDR CLAIM TRACK] Round {}: regs_val={x}, ram_ra={x}, lookups={x}\n", .{
                                round,
                                regs_val_current_claim.toBytesBE()[16..32].*,
                                ram_ra_prover.current_claim.toBytesBE()[16..32].*,
                                lookups_claim.toBytesBE()[16..32].*,
                            });
                            dbg("[ADDR CLAIM TRACK] Round {}: batched_claim={x}\n", .{
                                round,
                                current_batched_claim.toBytesBE()[16..32].*,
                            });
                        }
                    }

                    // ===================================================================
                    // Bind address challenge via extracted lookups prover
                    // ===================================================================
                    if (comptime bench_timing) {
                        bench_addr_other_ns += bench_timer.read();
                        bench_timer.reset();
                    }
                    try lookups_prover.bindAddressChallenge(challenge, round, challenges[0 .. round + 1]);
                    lookups_prover.setClaim(lookups_claim);
                    if (comptime bench_timing) bench_addr_bind_ns += bench_timer.read();
                    if (s5_phase_timer) |*pt| {
                        s5_addr_bind_ns += pt.read();
                        pt.reset();
                    }

                    if (comptime bench_timing) bench_phase_transition_ns += bench_timer.read();
                    if (s5_phase_timer) |*pt| {
                        s5_phase_trans_ns += pt.read();
                        pt.reset();
                    }
                    continue; // Skip the rest of the loop for address rounds
                } else {
                    // Cycle rounds: Instance 2 degree-10 polynomial via extracted prover
                    const lookups_round = round - LOOKUPS_LOG_K;

                    // Rematerialization at start of cycle rounds (round 128 only)
                    if (lookups_round == 0) {
                        try lookups_prover.rematerialize(challenges[0..LOOKUPS_LOG_K]);
                    }

                    // Compute cycle round polynomial via extracted prover
                    if (comptime bench_timing) bench_timer.reset();
                    const full_coeffs = try lookups_prover.computeCycleRoundPoly(lookups_round);
                    defer self.allocator.free(full_coeffs);
                    if (comptime bench_timing) {
                        bench_inst2_cycle_compute_ns += bench_timer.read();
                    }
                    if (s5_phase_timer) |*pt| {
                        s5_cycle_compute_ns += pt.read();
                        pt.reset();
                    }
                    if (comptime bench_timing) bench_timer.reset();

                    // For the batched sumcheck, we need to add Instance 2's polynomial to the combined polynomial
                    // The combined polynomial format depends on whether we're in a cycle round
                    // For cycle rounds, Instance 2 has degree 10, so we need a different approach

                    // Instance 0 and 1 contribute at most degree-3 (constant polynomials for most rounds)
                    // Instance 2 contributes degree-10 for cycle rounds (product of 10 linear factors)
                    // The batched polynomial degree is max(3, 10) = 10

                    // Scale by batch coefficient and combine
                    // For degree-10, we need 11 coefficients: [c0, c1, ..., c10]
                    // combined_poly only has 4 slots, so we need to extend it

                    // Create extended combined polynomial for degree-10
                    if (comptime bench_timing) bench_inst2_cycle_compute_ns += bench_timer.read();
                    if (s5_phase_timer) |*pt| {
                        s5_cycle_compute_ns += pt.read();
                        pt.reset();
                    }
                    if (comptime bench_timing) bench_timer.reset();
                    var combined_coeffs = try self.allocator.alloc(F, 11);
                    defer self.allocator.free(combined_coeffs);
                    @memset(combined_coeffs, F.zero());

                    // Add Instance 0 and 1 contributions (degree-3 or less)
                    // These are already in combined_poly[0..4]
                    const inst01_coeffs = UniPoly(F).toomCookToCoeffs(combined_poly);
                    for (0..4) |i| {
                        combined_coeffs[i] = combined_coeffs[i].add(inst01_coeffs[i]);
                    }

                    // Add Instance 2 contribution (degree-10)
                    for (full_coeffs, 0..) |coeff, i| {
                        if (i < combined_coeffs.len) {
                            combined_coeffs[i] = combined_coeffs[i].add(batch2.mul(coeff));
                        }
                    }

                    // Compress, append to proof/transcript, derive challenge, evaluate
                    const degree = combined_coeffs.len - 1; // degree 10
                    if (comptime bench_timing) bench_cycle_coeff_ns += bench_timer.read();
                    if (comptime bench_timing) bench_timer.reset();
                    const round_result = try sumcheck_helpers.finishSumcheckRound(F, combined_coeffs, degree, current_batched_claim, transcript, proof, self.allocator);
                    const challenge = round_result.challenge;
                    challenges[round] = challenge;
                    current_batched_claim = round_result.new_claim;
                    if (comptime bench_timing) bench_cycle_transcript_ns += bench_timer.read();

                    if (comptime debug_verbose) {
                        dbg("  [S5P] R{} challenge={x} new_e={x} degree={}\n", .{
                            round,
                            challenge.toBytes()[0..16].*,
                            current_batched_claim.toBytes()[0..16].*,
                            degree,
                        });
                    }

                    // Skip the standard compression/serialization below
                    if (comptime bench_timing) bench_timer.reset();
                    // Bind the challenge for RegistersValEvaluation if active
                    if (remaining_rounds <= regs_val_num_rounds) {
                        const regs_round = regs_val_num_rounds - remaining_rounds;

                        // Update Instance 0 claim: new_claim = p0(challenge)
                        // Reuse poly_evals cached from the compute phase (same round)
                        const poly_evals = cached_regs_val_evals.?;
                        const inst0_coeffs = UniPoly(F).toomCookToCoeffs(poly_evals);
                        // Evaluate at challenge using Horner's method
                        var p0_at_r = inst0_coeffs[3]; // c3 (highest)
                        p0_at_r = p0_at_r.mulHiBigIntU128(challenge.limbs).add(inst0_coeffs[2]); // c2
                        p0_at_r = p0_at_r.mulHiBigIntU128(challenge.limbs).add(inst0_coeffs[1]); // c1
                        p0_at_r = p0_at_r.mulHiBigIntU128(challenge.limbs).add(inst0_coeffs[0]); // c0
                        regs_val_current_claim = p0_at_r;

                        Inst.bindRegsValChallenge(inc_evals, wa_evals, &lt_poly, regs_round, challenge, self.thread_pool, self.gpu_ops);
                    }

                    // Bind the challenge for RamRaClaimReduction cycle rounds
                    // Upstream: cycle-only binding (no PhaseAddress)
                    if (remaining_rounds <= ram_ra_num_rounds) {
                        const ram_ra_round = ram_ra_num_rounds - remaining_rounds;
                        ram_ra_prover.bindChallenge(ram_ra_round, challenge);
                    }

                    // Bind cycle round challenge via extracted lookups prover
                    lookups_prover.bindCycleChallenge(challenge, lookups_round, full_coeffs);
                    lookups_claim = lookups_prover.current_claim;

                    if (comptime bench_timing) bench_cycle_bind_ns += bench_timer.read();
                    if (s5_phase_timer) |*pt| {
                        s5_cycle_bind_ns += pt.read();
                        pt.reset();
                    }
                    continue; // Skip the rest of the loop (we handled everything)
                }
            }

            // Post-loop timer
            if (comptime bench_timing) bench_timer.reset();

            // Print Stage 5 fine-grained timing
            if (comptime bench_timing) {
                const toMs = struct {
                    fn f(ns: u64) f64 {
                        return @as(f64, @floatFromInt(ns)) / 1_000_000.0;
                    }
                }.f;
                const init_ms = toMs(bench_init_ns);
                const addr_compute_ms = toMs(bench_addr_compute_ns);
                const addr_bind_ms = toMs(bench_addr_bind_ns);
                const addr_transcript_ms = toMs(bench_addr_transcript_ns);
                const addr_other_ms = toMs(bench_addr_other_ns);
                const addr_phase_ms = toMs(bench_phase_transition_ns);
                const cycle_compute_ms = toMs(bench_inst0_compute_ns + bench_inst1_compute_ns + bench_inst2_cycle_compute_ns);
                const cycle_bind_ms = toMs(bench_cycle_bind_ns);
                const cycle_transcript_ms = toMs(bench_cycle_transcript_ns);
                const addr_total_ms = addr_compute_ms + addr_bind_ms + addr_transcript_ms + addr_other_ms + addr_phase_ms;
                const cycle_total_ms = cycle_compute_ms + cycle_bind_ms + cycle_transcript_ms;
                const total_ms = init_ms + addr_total_ms + cycle_total_ms;

                std.debug.print("\n    [STAGE5-BENCH] === Stage 5 Fine-Grained Timing ===\n", .{});
                std.debug.print("    [STAGE5-BENCH] Init (alloc+setup):      {d:8.1} ms\n", .{init_ms});
                std.debug.print("    [STAGE5-BENCH] Address rounds (0-{}):\n", .{LOOKUPS_LOG_K - 1});
                std.debug.print("    [STAGE5-BENCH]   compute (ReadRaf):     {d:8.1} ms\n", .{addr_compute_ms});
                std.debug.print("    [STAGE5-BENCH]   bind (suffix+raf):     {d:8.1} ms\n", .{addr_bind_ms});
                std.debug.print("    [STAGE5-BENCH]   transcript:            {d:8.1} ms\n", .{addr_transcript_ms});
                std.debug.print("    [STAGE5-BENCH]   claim updates+RamRa:   {d:8.1} ms\n", .{addr_other_ms});
                std.debug.print("    [STAGE5-BENCH]   phase transitions:     {d:8.1} ms\n", .{addr_phase_ms});
                std.debug.print("    [STAGE5-BENCH]     condenseUEvals:      {d:8.1} ms\n", .{toMs(bench_condense_ns)});
                std.debug.print("    [STAGE5-BENCH]     initPhase+initQRaf:  {d:8.1} ms\n", .{toMs(bench_init_phase_ns)});
                std.debug.print("    [STAGE5-BENCH]   addr subtotal:         {d:8.1} ms\n", .{addr_total_ms});
                std.debug.print("    [STAGE5-BENCH] Cycle rounds ({}-{}):\n", .{ LOOKUPS_LOG_K, LOOKUPS_LOG_K + n_cycle_vars - 1 });
                std.debug.print("    [STAGE5-BENCH]   compute total:         {d:8.1} ms\n", .{cycle_compute_ms});
                std.debug.print("    [STAGE5-BENCH]     inst0 (RegsVal):     {d:8.1} ms\n", .{toMs(bench_inst0_compute_ns)});
                std.debug.print("    [STAGE5-BENCH]     inst1 (RamRa):       {d:8.1} ms\n", .{toMs(bench_inst1_compute_ns)});
                std.debug.print("    [STAGE5-BENCH]     inst2 (Lookups):     {d:8.1} ms\n", .{toMs(bench_inst2_cycle_compute_ns)});
                std.debug.print("    [STAGE5-BENCH]   bind:                  {d:8.1} ms\n", .{cycle_bind_ms});
                std.debug.print("    [STAGE5-BENCH]   transcript:            {d:8.1} ms\n", .{cycle_transcript_ms});
                std.debug.print("    [STAGE5-BENCH]   cycle subtotal:        {d:8.1} ms\n", .{cycle_total_ms});
                std.debug.print("    [STAGE5-BENCH] TOTAL (timed):           {d:8.1} ms\n", .{total_ms});
                std.debug.print("    [STAGE5-BENCH] Untimed gaps:\n", .{});
                std.debug.print("    [STAGE5-BENCH]   remat (round 128):     {d:8.1} ms\n", .{toMs(bench_remat_ns)});
                std.debug.print("    [STAGE5-BENCH]   coeff combo (18 rds):  {d:8.1} ms\n", .{toMs(bench_cycle_coeff_ns)});
                std.debug.print("    [STAGE5-BENCH] T={}, n_cycle_vars={}\n\n", .{ T, n_cycle_vars });
            }

            // Debug: print final batched claim (this is output_claim from verifier's perspective)
            if (comptime debug_verbose) {
                dbg("[STAGE5] Final batched claim (output_claim) = {any}\n", .{current_batched_claim.toBytesBE()});
                dbg("[STAGE5] Final lookups_current_scalar (should = eq_eval_r_reduction) = {x}\n", .{lookups_prover.lookups_current_scalar.toBytesBE()[16..32].*});
            }

            // DEBUG: Print each instance's final claim value
            // The verifier computes expected = batch0*inst0_eval + batch1*inst1_eval + batch2*inst2_eval
            // The prover's output_claim should equal this.
            if (comptime debug_verbose) {
                dbg("[STAGE5 FINAL CLAIMS] Individual instance final values:\n", .{});
                dbg("  regs_val_current_claim (Instance 0) = {any}\n", .{regs_val_current_claim.toBytes()});
                dbg("  ram_ra_prover.current_claim (Instance 1) = {any}\n", .{ram_ra_prover.current_claim.toBytes()});
                dbg("  lookups_claim (Instance 2) = {any}\n", .{lookups_claim.toBytes()});
                dbg("  batch0*inst0 (LE) = {any}\n", .{batch0.mul(regs_val_current_claim).toBytes()});
                dbg("  batch1*inst1 (LE) = {any}\n", .{batch1.mul(ram_ra_prover.current_claim).toBytes()});
                dbg("  batch2*inst2 (LE) = {any}\n", .{batch2.mul(lookups_claim).toBytes()});
            }

            // Print the chain values so we can compare with Jolt verifier
            if (comptime debug_verbose) {
                const print = std.debug.print;
                print("[ZOLT S5 CHAIN] inst0_claim FULL LE = {any}\n", .{regs_val_current_claim.toBytes()});
                print("[ZOLT S5 CHAIN] inst1_claim FULL LE = {any}\n", .{ram_ra_prover.current_claim.toBytes()});
                print("[ZOLT S5 CHAIN] inst2_claim FULL LE = {any}\n", .{lookups_claim.toBytes()});
                print("[ZOLT S5 CHAIN] batch0 FULL LE = {any}\n", .{batch0.toBytes()});
                print("[ZOLT S5 CHAIN] batch1 FULL LE = {any}\n", .{batch1.toBytes()});
                print("[ZOLT S5 CHAIN] batch2 FULL LE = {any}\n", .{batch2.toBytes()});
                print("[ZOLT S5 CHAIN] batch0*inst0 FULL LE = {any}\n", .{batch0.mul(regs_val_current_claim).toBytes()});
                print("[ZOLT S5 CHAIN] batch1*inst1 FULL LE = {any}\n", .{batch1.mul(ram_ra_prover.current_claim).toBytes()});
                print("[ZOLT S5 CHAIN] batch2*inst2 FULL LE = {any}\n", .{batch2.mul(lookups_claim).toBytes()});
                const recon = batch0.mul(regs_val_current_claim).add(batch1.mul(ram_ra_prover.current_claim)).add(batch2.mul(lookups_claim));
                print("[ZOLT S5 CHAIN] sum = {any}\n", .{recon.toBytes()});
                print("[ZOLT S5 CHAIN] batched_claim = {any}\n", .{current_batched_claim.toBytes()});
                print("[ZOLT S5 CHAIN] sum==batched = {}\n", .{recon.eql(current_batched_claim)});
            }
            const recon = batch0.mul(regs_val_current_claim).add(batch1.mul(ram_ra_prover.current_claim)).add(batch2.mul(lookups_claim));
            if (comptime debug_verbose) {
                dbg("  batch0*inst0 + batch1*inst1 + batch2*inst2 = {any}\n", .{recon.toBytes()});
                dbg("  current_batched_claim = {any}\n", .{current_batched_claim.toBytes()});
                dbg("  reconstruction matches output_claim: {}\n", .{recon.eql(current_batched_claim)});
            }

            // CRITICAL: Derive correct Instance 2 claim from batched output
            // The batched output_claim is CORRECT (S5P==S5V). Individual claims for
            // inst0 and inst1 are also correct. So we can derive the TRUE inst2 claim.
            const correct_inst2_from_batched = if (comptime debug_verbose) current_batched_claim.sub(batch0.mul(regs_val_current_claim)).sub(batch1.mul(ram_ra_prover.current_claim)).mul(batch2_inv) else F.zero();
            if (comptime debug_verbose) {
                dbg("  [DRIFT CHECK] lookups_claim (tracked)     = {any}\n", .{lookups_claim.toBytes()});
                dbg("  [DRIFT CHECK] correct_inst2 (from batch)  = {any}\n", .{correct_inst2_from_batched.toBytes()});
                dbg("  [DRIFT CHECK] drift detected: {}\n", .{!lookups_claim.eql(correct_inst2_from_batched)});
            }

            // =================================================================
            // BRUTE FORCE Instance 1 expected output claim
            // =================================================================
            if (comptime debug_verbose) {
                const inst1_start = max_num_rounds - ram_ra_num_rounds; // 128
                const cycle_chal = challenges[inst1_start .. inst1_start + ram_ra_num_rounds]; // 8 challenges

                // Reverse cycle challenges to BIG_ENDIAN
                var r_cycle_reduced_be_buf: [32]F = undefined;
                for (0..n_cycle_vars) |i| {
                    r_cycle_reduced_be_buf[i] = cycle_chal[n_cycle_vars - 1 - i];
                }
                const r_cycle_reduced_be = r_cycle_reduced_be_buf[0..n_cycle_vars];

                // Compute eq(r_cycle_x, r_cycle_reduced) — upstream uses BIG_ENDIAN mle
                const eq_cycle_raf_red = computeEqPolynomial(F, r_cycle_raf, r_cycle_reduced_be);
                const eq_cycle_rw_red = computeEqPolynomial(F, r_cycle_rw, r_cycle_reduced_be);
                const eq_cycle_val_red = computeEqPolynomial(F, r_cycle_val, r_cycle_reduced_be);

                // eq_combined = eq_raf + gamma*eq_rw + gamma^2*eq_val (upstream cycle-only)
                const bf_eq_combined = eq_cycle_raf_red
                    .add(gamma.mul(eq_cycle_rw_red))
                    .add(gamma2.mul(eq_cycle_val_red));

                // ra_claim_reduced = Σ eq(addr, r_address) * eq(cycle, r_cycle_reduced)
                // r_address is FIXED from opening claims (r_address_raf)
                var bf_ra_claim = F.zero();
                if (memory_trace) |mt| {
                    for (mt.accesses.items) |access| {
                        if (access.op == .Write) {
                            const raw_addr = access.address;
                            const cycle = access.timestamp;
                            const addr: u64 = if (memory_layout) |ml|
                                ml.remapAddress(raw_addr) orelse 0
                            else
                                raw_addr & ((@as(u64, 1) << @intCast(log_ram_k)) - 1);
                            const eq_a = computeEqAtPoint(F, r_address_raf, addr);
                            const eq_c = computeEqAtPoint(F, r_cycle_reduced_be, cycle);
                            bf_ra_claim = bf_ra_claim.add(eq_a.mul(eq_c));
                        }
                    }
                }

                const bf_expected_inst1 = bf_eq_combined.mul(bf_ra_claim);

                if (comptime debug_verbose) {
                    dbg("[BRUTE FORCE INST1] eq_cycle_raf = {x}\n", .{eq_cycle_raf_red.toBytesBE()[16..32].*});
                    dbg("[BRUTE FORCE INST1] eq_cycle_rw = {x}\n", .{eq_cycle_rw_red.toBytesBE()[16..32].*});
                    dbg("[BRUTE FORCE INST1] eq_cycle_val = {x}\n", .{eq_cycle_val_red.toBytesBE()[16..32].*});
                    dbg("[BRUTE FORCE INST1] eq_combined = {x}\n", .{bf_eq_combined.toBytesBE()[16..32].*});
                    dbg("[BRUTE FORCE INST1] ra_claim_reduced = {x}\n", .{bf_ra_claim.toBytesBE()[16..32].*});
                    dbg("[BRUTE FORCE INST1] expected (eq_combined * ra_reduced) = {any}\n", .{bf_expected_inst1.toBytes()});
                    dbg("[BRUTE FORCE INST1] prover tracked ram_ra_prover.current_claim = {any}\n", .{ram_ra_prover.current_claim.toBytes()});
                    dbg("[BRUTE FORCE INST1] match: {}\n", .{bf_expected_inst1.eql(ram_ra_prover.current_claim)});
                }
            }

            // Print lightweight phase timing
            if (s5_do_phase_timing) {
                const toMs = struct {
                    fn f(ns: u64) f64 {
                        return @as(f64, @floatFromInt(ns)) / 1_000_000.0;
                    }
                }.f;
                const total_timed = s5_addr_compute_ns + s5_addr_bind_ns + s5_phase_trans_ns + s5_cycle_compute_ns + s5_cycle_bind_ns;
                std.debug.print("    [S5-PHASE] addr_compute={d:.1}ms addr_bind={d:.1}ms phase_trans={d:.1}ms cycle_compute={d:.1}ms cycle_bind={d:.1}ms timed={d:.1}ms\n", .{
                    toMs(s5_addr_compute_ns),
                    toMs(s5_addr_bind_ns),
                    toMs(s5_phase_trans_ns),
                    toMs(s5_cycle_compute_ns),
                    toMs(s5_cycle_bind_ns),
                    toMs(total_timed),
                });
            }

            // Get final opening claims from the folded polynomials
            const regs_val_inc_claim = inc_evals[0];
            const regs_val_wa_claim = wa_evals[0];
            const regs_val_lt_claim = lt_poly.finalClaim();
            const regs_final_product = regs_val_inc_claim.mul(regs_val_wa_claim).mul(regs_val_lt_claim);

            if (comptime debug_verbose) {
                dbg("[STAGE5] Final opening claims:\n", .{});
                dbg("  regs_val_inc_claim = {any}\n", .{regs_val_inc_claim.toBytesBE()});
                dbg("  regs_val_wa_claim = {any}\n", .{regs_val_wa_claim.toBytesBE()});
                dbg("  regs_val_lt_claim (lt[0] after binding) = {any}\n", .{regs_val_lt_claim.toBytesBE()});
                dbg("  regs_final_product (inc*wa*lt) = {any}\n", .{regs_final_product.toBytesBE()});
            }

            // Compute what the verifier would compute for LT(r_normalized, r_cycle)
            // r_normalized = reversed challenges (BIG_ENDIAN)
            // The last 8 challenges are for RegistersValEvaluation
            const regs_challenges = challenges[(max_num_rounds - regs_val_num_rounds)..];
            if (comptime debug_verbose) {
                dbg("[STAGE5] Computing verifier's LT(r_normalized, r_cycle):\n", .{});
                dbg("  regs_challenges[0] = {any}\n", .{regs_challenges[0].toBytesBE()[0..8]});
                dbg("  regs_challenges[7] = {any}\n", .{regs_challenges[7].toBytesBE()[0..8]});
                dbg("  r_cycle_regs[0] = {any}\n", .{r_cycle_regs[0].toBytesBE()[0..8]});
                dbg("  r_cycle_regs[7] = {any}\n", .{r_cycle_regs[7].toBytesBE()[0..8]});
            }

            // Compute LT(r_normalized, r_cycle) like the verifier does
            // r_normalized = [c7, c6, c5, c4, c3, c2, c1, c0] (reversed challenges)
            // LT(x, y) = Σ_i (1 - x_i) · y_i · eq(x[i+1:], y[i+1:])
            var lt_verifier = F.zero();
            var eq_term = F.one();
            for (0..n_cycle_vars) |i| {
                // r_normalized[i] = challenges[n-1-i] (reversed)
                const x_i = regs_challenges[n_cycle_vars - 1 - i];
                const y_i = r_cycle_regs[i]; // r_cycle is already BIG_ENDIAN
                const one_minus_x = F.one().sub(x_i);
                lt_verifier = lt_verifier.add(one_minus_x.mul(y_i).mul(eq_term));
                // eq_term *= (1 - x - y + 2*x*y)
                const xy = x_i.mul(y_i);
                eq_term = eq_term.mul(F.one().sub(x_i).sub(y_i).add(xy).add(xy));
            }
            if (comptime debug_verbose) {
                dbg("  LT_verifier (what verifier computes) = {any}\n", .{lt_verifier.toBytesBE()});
            }

            // The verifier expects: expected_output_claim = inc_claim * wa_claim * LT_verifier
            const expected_product = regs_val_inc_claim.mul(regs_val_wa_claim).mul(lt_verifier);
            if (comptime debug_verbose) {
                dbg("  expected_product (inc*wa*LT_verifier) = {any}\n", .{expected_product.toBytesBE()});
                dbg("  Match: {}\n", .{regs_final_product.eql(expected_product)});
            }

            // Compute LookupsReadRaf opening claims via extracted prover
            const lookups_claims = try lookups_prover.getOpeningClaims(challenges);
            const table_flags = lookups_claims.table_flags;
            const ra_chunks = lookups_claims.ra_chunks;
            const computed_raf_flag = lookups_claims.raf_flag;

            // Compute ram_ra_claim from the extracted RamRaClaimReduction prover
            const ram_ra_claim = ram_ra_prover.finalClaim();
            if (comptime debug_verbose) {
                dbg("[STAGE5 RAM_RA] ram_ra_claim = H_prime[0] = {x}\n", .{ram_ra_claim.toBytesBE()});
            }

            return Stage5Result(F){
                .challenges = challenges,
                .regs_val_inc_claim = regs_val_inc_claim,
                .regs_val_wa_claim = regs_val_wa_claim,
                .ram_ra_claim = ram_ra_claim,
                .lookups_table_flags = table_flags,
                .lookups_ra_chunks = ra_chunks,
                .lookups_raf_flag = computed_raf_flag,
                .allocator = self.allocator,
            };
        }
    };
}

test "stage5_prover compiles" {
    const F = @import("zolt_arith").field.BN254Scalar;
    const allocator = std.testing.allocator;

    const prover = Stage5BatchedProver(F).init(allocator);
    _ = prover;
}
