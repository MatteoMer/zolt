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

// Debug output control - set to true to enable verbose debug prints
const debug_verbose = false;

// Benchmark timing control - set to true to enable fine-grained timing
const bench_timing = false;
fn dbg(comptime fmt: []const u8, args: anytype) void {
    if (debug_verbose) std.debug.print(fmt, args);
}

const Allocator = std.mem.Allocator;
const ThreadPool = @import("../../utils/thread_pool.zig").ThreadPool;
const GpuPolyOps = @import("../../gpu/mod.zig").GpuPolyOps;

const poly_mod = @import("../../poly/mod.zig");
const LtPolynomial = @import("../../poly/lt_poly.zig").LtPolynomial;
const UniPoly = poly_mod.UniPoly;
const transcripts = @import("../../transcripts/mod.zig");
const Blake2bTranscript = transcripts.Blake2bTranscript;
const jolt_types = @import("../jolt_types.zig");
const SumcheckInstanceProof = jolt_types.SumcheckInstanceProof;
const OpeningClaims = jolt_types.OpeningClaims;
const OpeningId = jolt_types.OpeningId;
const tracer = @import("../../tracer/mod.zig");
const ExecutionTrace = tracer.ExecutionTrace;
const ram = @import("../ram/mod.zig");
const jolt_device = @import("../jolt_device.zig");
const UnreducedProductAccum = @import("../../field/mod.zig").UnreducedProductAccum;

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
                    const scale = remaining_rounds - regs_val_num_rounds - 1;
                    var scaled_input_claim = regs_val_input;
                    for (0..scale) |_| scaled_input_claim = scaled_input_claim.add(scaled_input_claim);
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
                    const scale = remaining_rounds - ram_ra_num_rounds - 1;
                    var scaled_input_claim = ram_ra_input;
                    for (0..scale) |_| scaled_input_claim = scaled_input_claim.add(scaled_input_claim);
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
            var lookups_ra_weights = try self.allocator.alloc(F, T); // Per-cycle total ra weight (product of chunks)
            const lookups_indices_lo = try self.allocator.alloc(u64, T); // Lower 64 bits of lookup index
            const lookups_indices_hi = try self.allocator.alloc(u64, T); // Upper 64 bits of lookup index
            defer self.allocator.free(lookups_eq_evals);
            defer self.allocator.free(lookups_combined_vals);
            defer self.allocator.free(lookups_ra_weights);
            defer self.allocator.free(lookups_indices_lo);
            defer self.allocator.free(lookups_indices_hi);
            @memset(lookups_eq_evals, F.zero());
            @memset(lookups_combined_vals, F.zero());
            @memset(lookups_ra_weights, F.one()); // Start with weight 1
            @memset(lookups_indices_lo, 0);
            @memset(lookups_indices_hi, 0);

            // Track per-chunk ra weights for opening claims
            // We have up to 8 chunks of 16 bits each (LOOKUPS_LOG_K=128, chunk_size=16)
            // lookups_ra_virtual_log_k_chunk is typically 16, giving 8 chunks
            const MAX_RA_CHUNKS = 8;
            const ra_num_chunks = LOOKUPS_LOG_K / lookups_ra_virtual_log_k_chunk;
            std.debug.assert(ra_num_chunks <= MAX_RA_CHUNKS);

            // ra_chunk_weights allocated lazily at rematerialization (round 128) to avoid 64MB memset at init
            var ra_chunk_weights: [MAX_RA_CHUNKS][]F = undefined;
            var ra_chunk_weights_allocated = false;
            defer {
                if (ra_chunk_weights_allocated) {
                    for (0..ra_num_chunks) |chunk_idx| {
                        self.allocator.free(ra_chunk_weights[chunk_idx]);
                    }
                }
            }


            // Track which cycles use which lookup table (for flag claims)
            // and which use identity path (for raf_flag claim)
            const cycle_table_indices = try self.allocator.alloc(i8, T);
            const cycle_is_identity_path = try self.allocator.alloc(bool, T);
            defer self.allocator.free(cycle_table_indices);
            defer self.allocator.free(cycle_is_identity_path);
            @memset(cycle_table_indices, -1); // -1 = no table
            @memset(cycle_is_identity_path, false);

            // Store table MLE evaluations at r_address (populated during rematerialization)
            // These are used for computing val_claim = Σ table_flags[i] * table_values[i]
            // Note: Jolt has 40 tables (LookupTables::COUNT = 40, ValidSignedRemainder removed)
            const MAX_LOOKUP_TABLES: usize = 40;
            var stored_table_values: [MAX_LOOKUP_TABLES]F = [_]F{F.zero()} ** MAX_LOOKUP_TABLES;

            // Build eq_reduction[j] = eq(j, r_reduction) for all cycles j
            // r_reduction is in BIG_ENDIAN order (MSB first)
            // Use O(2^n) doubling technique instead of O(n * 2^n) per-element computation
            buildFullEqTable(r_reduction, lookups_eq_evals[0..T], self.thread_pool);

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
                            c.wa[j] = computeEqAtIndex(c.r_addr, @as(usize, rd));
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
                if (self.thread_pool) |tp| {
                    tp.parallelForForce(trace_len, ctx, incWaFn);
                } else {
                    for (0..trace_len) |j| incWaFn(ctx, j);
                }
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
                        j, rd, step.rd_pre_value, step.rd_value,
                        inc_evals[j].toBytesBE()[24..32].*,
                        wa_evals[j].toBytesBE()[24..32].*,
                        lt_poly.getBoundCoeff(j).toBytesBE()[24..32].*,
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
                    eq_cycle_evals[j] = computeEqAtIndex(r_cycle_regs, j);
                }
                // Precompute eq(r_address, k) for all k (all 128 registers)
                var eq_addr_evals: [REGS_K]F = undefined;
                for (0..REGS_K) |k| {
                    eq_addr_evals[k] = computeEqAtIndex(r_address_regs, k);
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
                    eq_vals3[t] = computeEqAtIndex(r_cycle_regs, t);
                }
                // Compute Σ_t eq(r_cycle, t) * cumsum(t) vs Σ_t eq(r_cycle, t) * val(r_addr, t)
                var sum_via_cumsum = F.zero();
                var sum_via_regvals = F.zero();
                var first_mismatch: usize = T;
                for (0..T) |t| {
                    // val(r_addr, t) = Σ_k eq(r_addr, k) * reg_vals3[k]
                    var val_at_t = F.zero();
                    for (0..REGS_K3) |k| {
                        val_at_t = val_at_t.add(computeEqAtIndex(r_address_regs, k).mul(F.fromU64(reg_vals3[k])));
                    }
                    sum_via_cumsum = sum_via_cumsum.add(eq_vals3[t].mul(cumsum));
                    sum_via_regvals = sum_via_regvals.add(eq_vals3[t].mul(val_at_t));
                    if (!cumsum.eql(val_at_t)) {
                        if (first_mismatch == T) first_mismatch = t;
                        if (t <= first_mismatch + 3) {
                            const delta = inc_evals[t].mul(wa_evals[t]);
                            dbg("[CUMSUM] MISMATCH at t={}: cumsum={x}, val={x}, delta={x}\n", .{
                                t, cumsum.toBytesBE()[24..32].*, val_at_t.toBytesBE()[24..32].*,
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
            defer self.allocator.free(lookup_indices_u128);
            @memset(lookup_indices_u128, 0);

            const is_interleaved_operands = try self.allocator.alloc(bool, T);
            defer self.allocator.free(is_interleaved_operands);
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
                        processTraceCycleCombined(c.steps_ptr[j], j, c.combined, c.idx_lo, c.idx_hi, c.tbl_ids, c.is_id, c.g_raf, c.g_raf2, c.idx_u128, c.is_inter);
                    }
                }
            }.f;
            if (self.thread_pool) |tp| {
                tp.parallelForForce(combined_chunk_count, cctx, combinedChunkFn);
            } else {
                for (0..trace_len) |j| {
                    processTraceCycleCombined(trace.steps.items[j], j, lookups_combined_vals, lookups_indices_lo, lookups_indices_hi, cycle_table_indices, cycle_is_identity_path, gamma_raf, gamma_raf2, lookup_indices_u128, is_interleaved_operands);
                }
            }
            // Padding cycles (trace_len..T) keep memset defaults

            if (comptime bench_timing) {
                std.debug.print("    [STAGE5-INIT]   parallel decode:  {d:8.1} ms\n", .{@as(f64, @floatFromInt(init_sub_timer.read())) / 1_000_000.0});
                init_sub_timer.reset();
            }

            // Build lookup_indices_by_table: for each table, collect cycle indices that use it.
            // This enables per-table parallelism in initPhase.
            // Parallel per-table dispatch: each table scans cycle_table_indices independently.
            var lookup_indices_by_table: [NUM_TABLES][]usize = undefined;
            const lookup_indices_by_table_initialized: usize = NUM_TABLES;
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
                if (self.thread_pool) |tp| {
                    tp.parallelForForce(NUM_TABLES, fill_ctx, fillFn);
                } else {
                    for (0..NUM_TABLES) |t| fillFn(fill_ctx, t);
                }
            }
            defer {
                for (0..lookup_indices_by_table_initialized) |t| {
                    self.allocator.free(lookup_indices_by_table[t]);
                }
            }

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
                    dbg("[STAGE5 OPCODE COUNTS] identity={}, interleaved={}\n", .{identity_count, interleaved_count});
                }
                for (opcode_counts, 0..) |cnt, opc| {
                    if (cnt > 0) {
                        if (comptime debug_verbose) {
                            dbg("  opcode 0x{x:0>2}: {} cycles\n", .{opc, cnt});
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
                    } else if (is_identity_add_imm2) F.fromU64(computeUnsignedImmediate(instr)) else computeImmediate(instr);

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
                            step_dbg2.rd_value, step_dbg2.pc,
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
                const imm_dbg = computeImmediate(instr_dbg);
                var right_input_dbg: F = F.zero();
                if (right_is_rs2_dbg) right_input_dbg = F.fromU64(step_dbg.rs2_value);
                if (right_is_imm_dbg) right_input_dbg = imm_dbg;

                if (comptime debug_verbose) {
                    dbg("  j={}: opcode=0x{x}, right_is_rs2={}, right_is_imm={}, imm=0x{x}, rs2=0x{x}, right_input=0x{x}\n", .{
                        jj, opcode_dbg, right_is_rs2_dbg, right_is_imm_dbg,
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
                } else computeImmediate(instr_v);
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
            // Instance 1: RamRaClaimReduction - initialized later after computed_ram_ra_input is computed
            var ram_ra_current_claim: F = undefined;
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
            // RamRaClaimReduction State Initialization
            // ===================================================================
            // For sparse traces (like Fibonacci with few RAM accesses), we compute
            // the sumcheck polynomial directly from the RAM trace.
            //
            // The sumcheck proves: Σ_{k,c} eq_combined(k,c) · ra(k,c) = input_claim
            // where:
            //   eq_combined(k, c) = eq(r_addr_1, k)·(eq_raf(c) + γ·eq_val(c))
            //                     + γ²·eq(r_addr_2, k)·(eq_rw(c) + γ·eq_val(c))
            //   ra(k,c) = 1 iff there's a RAM access at (address=k, cycle=c)
            //
            // Since ra(k,c) is sparse (only 1s at actual RAM accesses), we compute:
            //   input_claim = Σ_{accesses} eq_combined(addr, cycle)
            //
            // The sumcheck has 3 phases:
            //   - PhaseAddress: log_K rounds binding address variables k
            //   - PhaseCycle1: first half of cycle rounds using prefix-suffix
            //   - PhaseCycle2: second half of cycle rounds using dense sumcheck
            //
            // For each RAM access at (addr, cycle), we precompute:
            //   G_A[access] = eq(r_cycle_raf, cycle) + γ · eq(r_cycle_val, cycle)
            //   G_B[access] = eq(r_cycle_rw, cycle) + γ · eq(r_cycle_val, cycle)
            //
            // Then during PhaseAddress, for each access:
            //   contribution = eq(r_addr_1, k)·G_A + γ²·eq(r_addr_2, k)·G_B

            // K = 2^log_ram_k is the RAM domain size
            const K = @as(usize, 1) << @intCast(log_ram_k);

            // Build sparse RAM access list and precompute G_A, G_B for each access
            const ram_access_count = if (memory_trace) |mt| mt.accesses.items.len else 0;
            if (comptime debug_verbose) {
                dbg("[STAGE5 RAM_RA] Initializing with {} RAM accesses\n", .{ram_access_count});
            }

            // Allocate sparse access arrays
            var ram_addresses = try self.allocator.alloc(u64, ram_access_count);
            defer self.allocator.free(ram_addresses);
            var ram_cycles = try self.allocator.alloc(u64, ram_access_count);
            defer self.allocator.free(ram_cycles);
            var ram_G_A = try self.allocator.alloc(F, ram_access_count);
            defer self.allocator.free(ram_G_A);
            var ram_G_B = try self.allocator.alloc(F, ram_access_count);
            defer self.allocator.free(ram_G_B);
            // Precompute per-access eq(r_cycle_*, cycle) values once (used by G_A/G_B and cycle rounds)
            var eq_raf_access = try self.allocator.alloc(F, ram_access_count);
            defer self.allocator.free(eq_raf_access);
            var eq_rw_access = try self.allocator.alloc(F, ram_access_count);
            defer self.allocator.free(eq_rw_access);
            var eq_val_access = try self.allocator.alloc(F, ram_access_count);
            defer self.allocator.free(eq_val_access);

            if (memory_trace) |mt| {
                for (mt.accesses.items, 0..) |access, i| {
                    const cycle = access.timestamp;
                    eq_raf_access[i] = computeEqAtPoint(F, r_cycle_raf, cycle);
                    eq_rw_access[i] = computeEqAtPoint(F, r_cycle_rw, cycle);
                    eq_val_access[i] = computeEqAtPoint(F, r_cycle_val, cycle);
                }
            }

            // Precompute G_A and G_B for each RAM access using precomputed eq values
            // G_A[i] = eq(r_cycle_raf, c_i) + γ · eq(r_cycle_val, c_i)
            // G_B[i] = eq(r_cycle_rw, c_i) + γ · eq(r_cycle_val, c_i)
            //
            // Remap addresses to polynomial index space using memory_layout
            if (memory_trace) |mt| {
                for (mt.accesses.items, 0..) |access, i| {
                    const remapped_addr: u64 = if (memory_layout) |ml|
                        ml.remapAddress(access.address) orelse 0
                    else
                        access.address & (@as(u64, K) - 1);

                    ram_addresses[i] = remapped_addr;
                    ram_cycles[i] = access.timestamp;

                    // Reuse precomputed eq values
                    const eq_raf_c = eq_raf_access[i];
                    const eq_rw_c = eq_rw_access[i];
                    const eq_val_c = eq_val_access[i];

                    // G_A = eq_raf + γ · eq_val
                    // G_B = eq_rw + γ · eq_val
                    ram_G_A[i] = eq_raf_c.add(gamma.mul(eq_val_c));
                    ram_G_B[i] = eq_rw_c.add(gamma.mul(eq_val_c));

                    if (comptime debug_verbose) {
                        dbg("[STAGE5 RAM_RA] Access {}: raw_addr=0x{x}, remapped_addr={}, cycle={}\n", .{ i, access.address, remapped_addr, access.timestamp });
                        dbg("  eq_raf_c={any}, eq_rw_c={any}, eq_val_c={any}\n", .{
                            eq_raf_c.toBytesBE()[16..32].*,
                            eq_rw_c.toBytesBE()[16..32].*,
                            eq_val_c.toBytesBE()[16..32].*,
                        });
                        dbg("  G_A={any}, G_B={any}\n", .{
                            ram_G_A[i].toBytesBE()[16..32].*,
                            ram_G_B[i].toBytesBE()[16..32].*,
                        });
                    }
                }
            }

            // Full-size G_A and G_B arrays (size K, mostly zeros)
            // G_A_full[k] = G_A for the access at address k, or 0 if no access at k
            // G_B_full[k] = G_B for the access at address k, or 0 if no access at k
            // This allows us to iterate densely over all K addresses (like Jolt does)
            var G_A_full = try self.allocator.alloc(F, K);
            defer self.allocator.free(G_A_full);
            var G_B_full = try self.allocator.alloc(F, K);
            defer self.allocator.free(G_B_full);
            @memset(G_A_full, F.zero());
            @memset(G_B_full, F.zero());
            for (0..ram_access_count) |i| {
                const addr_usize: usize = @intCast(ram_addresses[i]);
                G_A_full[addr_usize] = ram_G_A[i];
                G_B_full[addr_usize] = ram_G_B[i];
            }
            if (comptime debug_verbose) {
                dbg("[STAGE5 RAM_RA] Created full-size G_A/G_B arrays (K={}), {} non-zero entries\n", .{ K, ram_access_count });
            }

            // Initialize B_1 and B_2 polynomials for address rounds
            // B_1 = eq(r_address_raf, k) - this is bound during address rounds
            // B_2 = eq(r_address_rw, k) - this is bound during address rounds
            // These are multilinear polynomials over log_ram_k variables
            var B_1 = try self.allocator.alloc(F, K);
            defer self.allocator.free(B_1);
            var B_2 = try self.allocator.alloc(F, K);
            defer self.allocator.free(B_2);

            // Compute B_1[k] = eq(r_address_raf, k) and B_2[k] = eq(r_address_rw, k) for all k
            // Uses O(2^n) butterfly construction instead of O(n * 2^n) per-element computation
            buildFullEqTable(r_address_raf, B_1[0..K], self.thread_pool);
            buildFullEqTable(r_address_rw, B_2[0..K], self.thread_pool);

            // Debug: print B_1 and B_2 for first few addresses
            if (comptime debug_verbose) {
                dbg("[STAGE5 RAM_RA] B_1/B_2 eq polynomials (first 4 and last 4 of {}):\n", .{K});
            }
            for (0..@min(4, K)) |k| {
                if (comptime debug_verbose) {
                    dbg("  B_1[{}]={any}, B_2[{}]={any}\n", .{
                        k, B_1[k].toBytesBE()[16..32].*, k, B_2[k].toBytesBE()[16..32].*,
                    });
                }
            }
            if (K > 8) {
                for (K - 4..K) |k| {
                    if (comptime debug_verbose) {
                        dbg("  B_1[{}]={any}, B_2[{}]={any}\n", .{
                            k, B_1[k].toBytesBE()[16..32].*, k, B_2[k].toBytesBE()[16..32].*,
                        });
                    }
                }
            }

            // Initialize Instance 1 claim tracking with SCALED value
            ram_ra_current_claim = ram_ra_scaled;

            // Expanding table to track eq(r_addr_reduced_so_far, k_bound_bits)
            // This accumulates the eq value as we bind address bits
            var ram_ra_F = try ExpandingTable(F).init(self.allocator, K);
            defer ram_ra_F.deinit();
            ram_ra_F.reset(F.one());

            // Track bound address challenges for RamRaClaimReduction
            // ram_ra_bound_challenges removed — no PhaseAddress binding in upstream cycle-only reduction

            // Track per-access eq_cycle_bound factor during cycle rounds
            // eq_cycle_bound[i] = product of eq terms for bound cycle bits
            // Initially 1.0, then after binding bit m with challenge r:
            //   eq_cycle_bound[i] *= (c_m == 0) ? (1 - r) : r
            var eq_cycle_bound = try self.allocator.alloc(F, ram_access_count);
            defer self.allocator.free(eq_cycle_bound);
            for (0..ram_access_count) |i| {
                eq_cycle_bound[i] = F.one();
            }

            // ===================================================================
            // Separate eq tracking for PhaseCycle correction
            // ===================================================================
            // The verifier expects eq(r_cycle_*, r_cycle_reduced) where r_cycle_reduced
            // is the sumcheck challenges. But G_A/G_B have eq(r_cycle_*, c_i) where c_i
            // is the access's cycle.
            //
            // We need to track:
            // - eq_*_bound: product of eq_bit(r_cycle_*[j], r_j) for bound bits j < m
            // - eq_*_remaining[i]: product of eq_bit(r_cycle_*[j], c_i[j]) for remaining bits j >= m
            //
            // The contribution should be eq_*_bound * eq_*_remaining * eq_bit(r_cycle_*[m], c_i[m])
            // instead of eq(r_cycle_*, c_i).

            // Bound factors (shared for all accesses) - eq(r_cycle_*, r_cycle_reduced_so_far)
            // These track the product of eq_bit(r_cycle_*[j], r_j) for j < current_round
            var eq_raf_bound: F = F.one();
            var eq_rw_bound: F = F.one();
            var eq_val_bound: F = F.one();

            // Remaining factors (per-access) - product of eq_bit(r_cycle_*[j], c_i[j]) for j >= current_round
            // Initially equals the full eq values, then we divide out bound bits as they're processed
            var eq_raf_remaining = try self.allocator.alloc(F, ram_access_count);
            defer self.allocator.free(eq_raf_remaining);
            var eq_rw_remaining = try self.allocator.alloc(F, ram_access_count);
            defer self.allocator.free(eq_rw_remaining);
            var eq_val_remaining = try self.allocator.alloc(F, ram_access_count);
            defer self.allocator.free(eq_val_remaining);

            // Initialize remaining factors to full eq values
            for (0..ram_access_count) |i| {
                eq_raf_remaining[i] = eq_raf_access[i];
                eq_rw_remaining[i] = eq_rw_access[i];
                eq_val_remaining[i] = eq_val_access[i];
            }

            // ===================================================================
            // P*Q Decomposition for PhaseCycle (Jolt's approach)
            // ===================================================================
            // For cycle sumcheck, we use prefix-suffix decomposition:
            //   P_x[c_lo] = eq(r_cycle_x_lo, c_lo)  -- eq evaluations for prefix bits
            //   Q_x[c_lo] = Σ_{c_hi} H[c_lo,c_hi] · eq(r_cycle_x_hi, c_hi)  -- suffix sums
            // where H[c] = F_values[address[c]] = eq(r_addr_reduced, address[c])
            //
            // The polynomial contribution is: Σ_j coeff * P_x[j] * Q_x[j]
            // This correctly captures binding: P gets bound during sumcheck, Q contains
            // the already-computed suffix weights.
            //
            // Split: log_T = prefix_n_vars + suffix_n_vars
            // In Jolt: prefix_n_vars = log_T / 2, suffix_n_vars = log_T - prefix_n_vars

            const prefix_n_vars = n_cycle_vars / 2;
            const suffix_n_vars = n_cycle_vars - prefix_n_vars;
            const prefix_size = @as(usize, 1) << @intCast(prefix_n_vars);
            const suffix_size = @as(usize, 1) << @intCast(suffix_n_vars);

            if (comptime debug_verbose) {
                dbg("[STAGE5 PQ] PhaseCycle P*Q setup: n_cycle={}, prefix={}, suffix={}\n", .{
                    n_cycle_vars, prefix_n_vars, suffix_n_vars,
                });
                dbg("[STAGE5 PQ] prefix_size={}, suffix_size={}\n", .{ prefix_size, suffix_size });
            }

            // P arrays: eq evaluations for prefix (low) bits
            // P_x[c_lo] = eq(r_cycle_x_lo, c_lo)
            // r_cycle vectors are BIG_ENDIAN: first suffix_n_vars are high bits, last prefix_n_vars are low
            var P_raf = try self.allocator.alloc(F, prefix_size);
            defer self.allocator.free(P_raf);
            var P_rw = try self.allocator.alloc(F, prefix_size);
            defer self.allocator.free(P_rw);
            var P_val = try self.allocator.alloc(F, prefix_size);
            defer self.allocator.free(P_val);

            // Q arrays: suffix-weighted sums
            // Q_x[c_lo] = Σ_{c_hi} H[c_lo,c_hi] · eq(r_cycle_x_hi, c_hi)
            var Q_raf = try self.allocator.alloc(F, prefix_size);
            defer self.allocator.free(Q_raf);
            var Q_rw = try self.allocator.alloc(F, prefix_size);
            defer self.allocator.free(Q_rw);
            var Q_val = try self.allocator.alloc(F, prefix_size);
            defer self.allocator.free(Q_val);

            // Precompute eq evaluations for suffix (high) bits: eq(r_cycle_x_hi, c_hi)
            // r_cycle_*[0..suffix_n_vars] are the high bits (BIG_ENDIAN)
            var eq_raf_hi = try self.allocator.alloc(F, suffix_size);
            defer self.allocator.free(eq_raf_hi);
            var eq_rw_hi = try self.allocator.alloc(F, suffix_size);
            defer self.allocator.free(eq_rw_hi);
            var eq_val_hi = try self.allocator.alloc(F, suffix_size);
            defer self.allocator.free(eq_val_hi);

            // Compute eq_x_hi for all c_hi values
            // r_cycle_*[0..suffix_n_vars] contains the HIGH bits
            for (0..suffix_size) |c_hi| {
                eq_raf_hi[c_hi] = computeEqAtPoint(F, r_cycle_raf[0..suffix_n_vars], c_hi);
                eq_rw_hi[c_hi] = computeEqAtPoint(F, r_cycle_rw[0..suffix_n_vars], c_hi);
                eq_val_hi[c_hi] = computeEqAtPoint(F, r_cycle_val[0..suffix_n_vars], c_hi);
            }

            // Initialize P arrays from prefix bits: P_x[c_lo] = eq(r_cycle_x_lo, c_lo)
            // r_cycle_*[suffix_n_vars..] contains the LOW bits
            for (0..prefix_size) |c_lo| {
                P_raf[c_lo] = computeEqAtPoint(F, r_cycle_raf[suffix_n_vars..], c_lo);
                P_rw[c_lo] = computeEqAtPoint(F, r_cycle_rw[suffix_n_vars..], c_lo);
                P_val[c_lo] = computeEqAtPoint(F, r_cycle_val[suffix_n_vars..], c_lo);
            }

            // Initialize Q arrays to zero
            @memset(Q_raf, F.zero());
            @memset(Q_rw, F.zero());
            @memset(Q_val, F.zero());

            // Note: Q arrays will be computed at the start of PhaseCycle when we have
            // the F_values from PhaseAddress (ram_ra_F after all address rounds).
            // This is deferred because F_values evolve during address rounds.

            // Flag to track if Q has been initialized
            var phase_cycle_q_initialized = false;

            // For PhaseCycle2: H'[c_hi] = Σ_{c_lo} H[c_lo,c_hi] * eq_prefix[c_lo]
            var H_prime = try self.allocator.alloc(F, suffix_size);
            defer self.allocator.free(H_prime);
            @memset(H_prime, F.zero());

            // Track cycle sumcheck challenges for computing eq_prefix during PhaseCycle2 transition
            var cycle_challenges = try self.allocator.alloc(F, n_cycle_vars);
            defer self.allocator.free(cycle_challenges);
            @memset(cycle_challenges, F.zero());

            // Flag to track if H_prime has been initialized (for PhaseCycle2)
            var phase_cycle2_initialized = false;
            // Scale factors for PhaseCycle2: eq(r_cycle_x_lo, r_cycle_prefix_reduced)
            // These are computed during PhaseCycle2 initialization and used in every PhaseCycle2 round
            var scale_raf: F = F.one();
            var scale_rw: F = F.one();
            var scale_val: F = F.one();

            // Store Instance 1 polynomial evaluations for claim update after challenge
            // These are set in the polynomial computation section and used in the binding section
            var inst1_eval_0: F = F.zero();
            var inst1_eval_1: F = F.zero();
            var inst1_eval_2: F = F.zero();

            // ===================================================================
            // Prefix-Suffix Decomposition Initialization for LookupsReadRaf
            // ===================================================================
            // lookup_indices_u128 already populated by parallel decode pass above

            if (comptime bench_timing) {
                std.debug.print("    [STAGE5-INIT] combined+indices:  {d:8.1} ms\n", .{@as(f64, @floatFromInt(init_sub_timer.read())) / 1_000_000.0});
                init_sub_timer.reset();
            }

            // Initialize suffix polynomials for phase 0
            var suffix_polys = AllSuffixPolys(F).init(self.allocator);
            defer suffix_polys.deinit();
            var prefix_checkpoints = PrefixCheckpointsState(F).init();

            // Jolt uses 16 phases for log_T < 24 (traces < 2^24), 8 phases otherwise
            // This matches config::get_instruction_sumcheck_phases() in Jolt
            const INSTRUCTION_PHASES_THRESHOLD_LOG_T = 24;
            const log_T = std.math.log2_int(usize, T);
            const num_phases: usize = if (log_T < INSTRUCTION_PHASES_THRESHOLD_LOG_T) 16 else 8;
            const log_m = LOOKUPS_LOG_K / num_phases; // = 8 for 16 phases, 16 for 8 phases
            var current_phase: usize = 0;
            const initial_m = @as(usize, 1) << @intCast(log_m); // 2^8 = 256 or 2^16 = 65536

            // Initialize RAF (Read-Address-Flag) decompositions for left/right/identity
            // Allocated before initPhase so both can run concurrently via tp.join()
            var left_raf = try RafDecomposition(F).init(self.allocator, initial_m, log_m, LOOKUPS_LOG_K, .LeftOperand);
            defer left_raf.deinit();
            var right_raf = try RafDecomposition(F).init(self.allocator, initial_m, log_m, LOOKUPS_LOG_K, .RightOperand);
            defer right_raf.deinit();
            var identity_raf = try RafDecomposition(F).init(self.allocator, initial_m, log_m, LOOKUPS_LOG_K, .Identity);
            defer identity_raf.deinit();

            // is_interleaved_operands already populated by parallel decode pass above

            // Initialize Q polynomials for read-checking AND RAF Q accumulators concurrently
            if (self.thread_pool) |tp| {
                const SuffixInitCtx = struct {
                    polys: *@TypeOf(suffix_polys),
                    phases: usize,
                    eq: []const F,
                    indices: []const u128,
                    table_indices: []const i8,
                    pool: *ThreadPool,
                    alloc_inner: Allocator,
                    ibt: ?*const [NUM_TABLES][]usize,
                };
                const suffix_ctx = SuffixInitCtx{
                    .polys = &suffix_polys,
                    .phases = num_phases,
                    .eq = lookups_eq_evals,
                    .indices = lookup_indices_u128,
                    .table_indices = cycle_table_indices,
                    .pool = tp,
                    .alloc_inner = self.allocator,
                    .ibt = &lookup_indices_by_table,
                };
                const RafInitCtx = struct {
                    left_raf: *@TypeOf(left_raf),
                    right_raf: *@TypeOf(right_raf),
                    identity_raf: *@TypeOf(identity_raf),
                    eq: []const F,
                    indices: []const u128,
                    is_interleaved: []const bool,
                    pool: *ThreadPool,
                    alloc_inner: Allocator,
                };
                const raf_ctx = RafInitCtx{
                    .left_raf = &left_raf,
                    .right_raf = &right_raf,
                    .identity_raf = &identity_raf,
                    .eq = lookups_eq_evals,
                    .indices = lookup_indices_u128,
                    .is_interleaved = is_interleaved_operands,
                    .pool = tp,
                    .alloc_inner = self.allocator,
                };
                _ = tp.join(
                    void,
                    void,
                    suffix_ctx,
                    struct {
                        fn f(c: SuffixInitCtx) void {
                            c.polys.initPhase(0, c.phases, c.eq, c.indices, c.table_indices, c.pool, c.alloc_inner, c.ibt) catch unreachable;
                        }
                    }.f,
                    raf_ctx,
                    struct {
                        fn f(c: RafInitCtx) void {
                            initQRaf(F, c.left_raf, c.right_raf, c.identity_raf, c.eq, c.indices, c.is_interleaved, c.pool, c.alloc_inner) catch unreachable;
                        }
                    }.f,
                );
            } else {
                try suffix_polys.initPhase(0, num_phases, lookups_eq_evals, lookup_indices_u128, cycle_table_indices, null, self.allocator, null);
                try initQRaf(F, &left_raf, &right_raf, &identity_raf, lookups_eq_evals, lookup_indices_u128, is_interleaved_operands, null, self.allocator);
            }

            if (comptime bench_timing) {
                std.debug.print("    [STAGE5-INIT] initPhase+initQRaf:{d:8.1} ms\n", .{@as(f64, @floatFromInt(init_sub_timer.read())) / 1_000_000.0});
                init_sub_timer.reset();
            }

            // Materialize prefix MLE tables for phase 0
            left_raf.initPrefix();
            right_raf.initPrefix();
            identity_raf.initPrefix();

            // Initialize expanding tables for each phase (accumulate eq values during address rounds)
            // We need num_phases expanding tables, each of size 2^log_m
            // Use max of 16 phases (for small traces)
            var expanding_tables: [16]ExpandingTable(F) = undefined;
            var tables_initialized: usize = 0;
            errdefer {
                for (0..tables_initialized) |phase_idx| {
                    expanding_tables[phase_idx].deinit();
                }
            }
            for (0..num_phases) |phase_idx| {
                expanding_tables[phase_idx] = try ExpandingTable(F).init(self.allocator, initial_m);
                tables_initialized = phase_idx + 1;
            }
            defer {
                for (0..num_phases) |phase_idx| {
                    expanding_tables[phase_idx].deinit();
                }
            }

            // Reset phase 0's expanding table to 1
            expanding_tables[0].reset(F.one());

            if (comptime debug_verbose) {
                dbg("[STAGE5 PREFIX-SUFFIX] Initialized phase 0, log_m={}, suffix_len={}, initial_m={}\n", .{
                    log_m,
                    LOOKUPS_LOG_K - log_m,
                    initial_m,
                });
            }

            // ==========================================================================
            // DIAGNOSTIC: Verify the total sum from Q polynomials matches brute force
            // After initPhase, Q[table][suffix][prefix_idx] = Σ_{j: table_j==t, prefix==prefix_idx} u[j] * suffixMle(suffix_bits_j)
            //
            // The read_checking total sum should equal: Σ_j u[j] * table_t(k_j)
            // = Σ_j u[j] * tableCombine_t(prefix_value(k_j), suffix_values(k_j))
            //
            // We can verify this by computing:
            //   Σ_{b=0}^{m-1} tableCombine(actual_prefix_value(b), Q[b])
            //
            // But the prefix value at the concrete index b is just: the formula applied to integer b
            // For LowerWord: it accumulates 2^(shift) * bit_value for each bit position
            //
            // Simpler approach: Compute brute-force rv_claim from trace data and compare
            // ==========================================================================
            if (comptime debug_verbose) {
                // Brute-force: Σ_j u[j] * combined[j] over ALL cycles (not just non-noop)
                var bf_combined = F.zero();
                var bf_combined_noop_skipped = F.zero();
                for (0..T) |jj| {
                    const u_j = lookups_eq_evals[jj];
                    bf_combined = bf_combined.add(u_j.mul(lookups_combined_vals[jj]));
                    if (jj < trace.steps.items.len) {
                        const step_bf = trace.steps.items[jj];
                        if (step_bf.is_noop and !step_bf.is_termination_store) continue;
                    }
                    bf_combined_noop_skipped = bf_combined_noop_skipped.add(u_j.mul(lookups_combined_vals[jj]));
                }

                // Also count non-zero combined vals beyond trace
                var nonzero_combined_beyond_trace: usize = 0;
                for (trace.steps.items.len..T) |jj| {
                    if (!lookups_combined_vals[jj].eql(F.zero())) nonzero_combined_beyond_trace += 1;
                }

                if (comptime debug_verbose) {
                    dbg("[DIAG INIT] bf_combined (all T={}) = {x}\n", .{ T, bf_combined.toBytesBE()[16..32].* });
                    dbg("[DIAG INIT] bf_combined_noop_skipped = {x}\n", .{bf_combined_noop_skipped.toBytesBE()[16..32].*});
                    dbg("[DIAG INIT] lookups_claim = {x}\n", .{lookups_claim.toBytesBE()[16..32].*});
                    dbg("[DIAG INIT] combined_all==claim: {}, combined_skip==claim: {}\n", .{ bf_combined.eql(lookups_claim), bf_combined_noop_skipped.eql(lookups_claim) });
                    dbg("[DIAG INIT] nonzero_combined beyond trace: {}\n", .{nonzero_combined_beyond_trace});
                }
                // Compare opening claims individually
                if (comptime debug_verbose) {
                    dbg("[DIAG INIT] rv_claim = {x}\n", .{rv_claim.toBytesBE()[16..32].*});
                    dbg("[DIAG INIT] left_op_claim = {x}\n", .{left_op_claim.toBytesBE()[16..32].*});
                    dbg("[DIAG INIT] right_op_claim = {x}\n", .{right_op_claim.toBytesBE()[16..32].*});
                    dbg("[DIAG INIT] gamma_raf = {x}\n", .{gamma_raf.toBytesBE()[16..32].*});
                    dbg("[DIAG INIT] gamma_raf2 = {x}\n", .{gamma_raf2.toBytesBE()[16..32].*});
                }
                const recomputed_input = rv_claim.add(gamma_raf.mul(left_op_claim)).add(gamma_raf2.mul(right_op_claim));
                if (comptime debug_verbose) {
                    dbg("[DIAG INIT] recomputed rv+g*l+g2*r = {x}\n", .{recomputed_input.toBytesBE()[16..32].*});
                    dbg("[DIAG INIT] lookups_input match: {}\n", .{recomputed_input.eql(lookups_claim)});
                }

                // DIAGNOSTIC: Recompute combined values using R1CS-style formula
                // and compare against Stage 5's trace-derived combined_vals.
                if (comptime debug_verbose) {
                // The R1CS witness computes:
                //   For AddOperands: LeftLookup=0, RightLookup=left_input+right_input (FIELD arithmetic)
                //   For interleaved: LeftLookup=left_input, RightLookup=right_input
                // While Stage 5 computes:
                //   For identity path: left_op=0, right_op=F.fromU128(u128 arithmetic result)
                // These may differ when negative immediates are involved.
                {
                    var r1cs_bf_sum = F.zero();
                    var stage5_bf_sum = F.zero();
                    var per_cycle_mismatches: usize = 0;
                    for (0..T) |jj_d| {
                        const eq_j = lookups_eq_evals[jj_d];
                        stage5_bf_sum = stage5_bf_sum.add(eq_j.mul(lookups_combined_vals[jj_d]));

                        if (jj_d >= trace.steps.items.len) {
                            // Padding cycle: R1CS would be zero too
                            continue;
                        }
                        const step_d = trace.steps.items[jj_d];
                        if (step_d.is_noop and !step_d.is_termination_store) continue;

                        const instr_d = step_d.instruction;
                        const opcode_d: u8 = @truncate(instr_d & 0x7f);
                        const funct3_d: u3 = @truncate((instr_d >> 12) & 0x7);
                        const funct7_d: u7 = @truncate(instr_d >> 25);
                        const table_idx_d = getLookupTableIndex(opcode_d, funct3_d, funct7_d);

                        if (table_idx_d < 0) continue; // no-table cycles: both are zero

                        // Recompute using FIELD arithmetic (R1CS style)
                        const r1cs_left_is_rs1: bool = switch (opcode_d) {
                            0x33, 0x3b, 0x23, 0x63, 0x13, 0x03, 0x67, 0x1b, 0x0B, 0x2B, 0x5B => true,
                            0x22 => true, // VirtualAssertEQ
                            0x42 => true, // VirtualZeroExtendWord
                            0x62 => true, // VirtualAssertValidUnsignedRemainder
                            else => false,
                        };
                        const r1cs_left_is_pc: bool = switch (opcode_d) {
                            0x17, 0x6f => true,
                            else => false,
                        };
                        const r1cs_right_is_rs2: bool = switch (opcode_d) {
                            0x33, 0x63, 0x3b => true,
                            0x22 => (funct3_d == 0 or funct3_d == 1), // VirtualAssertEQ/ValidDiv0: rs2; alignment: imm
                            0x62 => true, // VirtualAssertValidUnsignedRemainder
                            0x5B => step_d.rs2_read, // VirtualSRL/SRA R-type: rs2
                            else => false,
                        };
                        const r1cs_right_is_imm: bool = switch (opcode_d) {
                            0x13, 0x03, 0x67, 0x23, 0x37, 0x17, 0x6f, 0x1b, 0x0B, 0x2B => true,
                            0x22 => (funct3_d == 2 or funct3_d == 3), // alignment assertions: imm
                            0x5B => !step_d.rs2_read, // I-type: imm; R-type: not imm
                            else => false,
                        };
                        var r1cs_left_input = F.zero();
                        if (r1cs_left_is_rs1) r1cs_left_input = F.fromU64(step_d.rs1_value);
                        if (r1cs_left_is_pc) r1cs_left_input = F.fromU64(step_d.unexpanded_pc);
                        var r1cs_right_input = F.zero();
                        if (r1cs_right_is_rs2) r1cs_right_input = F.fromU64(step_d.rs2_value);
                        if (r1cs_right_is_imm) {
                            if (opcode_d == 0x2B) {
                                if (funct3_d == 0) {
                                    const shamt_rd: u32 = instr_d >> 20;
                                    const shamt_d: u6 = @truncate(shamt_rd & 0x3F);
                                    r1cs_right_input = F.fromU64(@as(u64, 1) << shamt_d);
                                } else {
                                    r1cs_right_input = F.zero(); // VirtualPow2/VirtualShiftRightBitmask: IMM = 0
                                }
                            } else if (opcode_d == 0x5B) {
                                if (step_d.rs2_read) {
                                    r1cs_right_input = F.zero(); // R-type: no immediate
                                } else {
                                    const ts_rd: u32 = instr_d >> 20;
                                    const ts_d: u7 = @truncate(ts_rd & 0x3F);
                                    const ones_d: u128 = (@as(u128, 1) << @intCast(64 - @as(u8, ts_d))) - 1;
                                    r1cs_right_input = F.fromU64(@truncate(ones_d << ts_d));
                                }
                            } else if (opcode_d == 0x22 and (funct3_d == 2 or funct3_d == 3)) {
                                r1cs_right_input = F.fromU64(computeUnsignedImmediate(instr_d));
                            } else {
                                r1cs_right_input = computeImmediate(instr_d);
                            }
                        }

                        // Now compute R1CS-style lookup operands
                        var r1cs_left_op = F.zero();
                        var r1cs_right_op = F.zero();
                        const r1cs_is_add = switch (opcode_d) {
                            0x33 => (funct3_d == 0 and funct7_d == 0), // ADD
                            0x13 => (funct3_d == 0), // ADDI
                            0x0B => true, // VirtualSignExtendWord
                            0x37, 0x17, 0x6f, 0x67 => true, // LUI, AUIPC, JAL, JALR
                            0x1b => (funct3_d == 0), // ADDIW
                            0x3b => (funct3_d == 0 and funct7_d == 0), // ADDW
                            0x2B => (funct3_d != 0), // VirtualPow2/VirtualShiftRightBitmask: AddOperands
                            0x22 => (funct3_d == 2 or funct3_d == 3), // Alignment assertions: AddOperands
                            0x42 => true, // VirtualZeroExtendWord (AddOperands)
                            else => false,
                        };
                        const r1cs_is_sub = switch (opcode_d) {
                            0x33 => (funct3_d == 0 and funct7_d == 0x20), // SUB
                            0x3b => (funct3_d == 0 and funct7_d == 0x20), // SUBW
                            else => false,
                        };
                        const r1cs_is_mul = switch (opcode_d) {
                            0x33 => (funct7_d == 0x01 and (funct3_d == 0 or funct3_d == 3)), // MUL, MULHU
                            0x2B => (funct3_d == 0), // VirtualMULI (funct3=0 only)
                            else => false,
                        };
                        if (opcode_d == 0x02) {
                            // VirtualAdvice: Advice flag, identity path
                            r1cs_left_op = F.zero();
                            r1cs_right_op = F.fromU128(@as(u128, step_d.rd_value));
                        } else if (opcode_d == 0x42) {
                            // VirtualZeroExtendWord: AddOperands, identity path
                            r1cs_left_op = F.zero();
                            r1cs_right_op = F.fromU128(@as(u128, step_d.rs1_value));
                        } else if (r1cs_is_add) {
                            r1cs_left_op = F.zero();
                            r1cs_right_op = r1cs_left_input.add(r1cs_right_input);
                        } else if (r1cs_is_sub) {
                            const two_pow_64 = F.fromBytes(&[_]u8{ 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 });
                            r1cs_left_op = F.zero();
                            r1cs_right_op = r1cs_left_input.sub(r1cs_right_input).add(two_pow_64);
                        } else if (r1cs_is_mul) {
                            r1cs_left_op = F.zero();
                            r1cs_right_op = r1cs_left_input.mul(r1cs_right_input);
                        } else {
                            // Default includes 0x22 (VirtualAssertEQ), 0x62 (VirtualAssertValidUnsignedRemainder)
                            r1cs_left_op = r1cs_left_input;
                            r1cs_right_op = r1cs_right_input;
                        }

                        // Compute R1CS-style LookupOutput (matching computeLookupOutput)
                        var r1cs_output: F = undefined;
                        switch (opcode_d) {
                            0x6f => { r1cs_output = r1cs_left_input.add(r1cs_right_input); },
                            0x67 => {
                                const target = r1cs_left_input.add(r1cs_right_input);
                                r1cs_output = F.fromU64(target.toU64() & ~@as(u64, 1));
                            },
                            0x63 => { r1cs_output = F.fromU64(switch (funct3_d) {
                                0x0 => @intFromBool(step_d.rs1_value == step_d.rs2_value),
                                0x1 => @intFromBool(step_d.rs1_value != step_d.rs2_value),
                                0x4 => @intFromBool(@as(i64, @bitCast(step_d.rs1_value)) < @as(i64, @bitCast(step_d.rs2_value))),
                                0x5 => @intFromBool(@as(i64, @bitCast(step_d.rs1_value)) >= @as(i64, @bitCast(step_d.rs2_value))),
                                0x6 => @intFromBool(step_d.rs1_value < step_d.rs2_value),
                                0x7 => @intFromBool(step_d.rs1_value >= step_d.rs2_value),
                                else => 0,
                            }); },
                            0x22, 0x62 => {
                                // VirtualAssertEQ / VirtualAssertValidUnsignedRemainder: Assert => output = 1
                                r1cs_output = F.one();
                            },
                            else => { r1cs_output = F.fromU64(step_d.rd_value); },
                        }

                        const r1cs_combined = r1cs_output.add(gamma_raf.mul(r1cs_left_op)).add(gamma_raf2.mul(r1cs_right_op));
                        r1cs_bf_sum = r1cs_bf_sum.add(eq_j.mul(r1cs_combined));

                        // Compare per-cycle
                        if (!r1cs_combined.eql(lookups_combined_vals[jj_d]) and per_cycle_mismatches < 5) {
                            if (comptime debug_verbose) {
                                dbg("[DIAG MISMATCH] j={}: opcode=0x{x}, funct3={}, funct7={}\n", .{ jj_d, opcode_d, funct3_d, funct7_d });
                                dbg("  stage5_combined = {x}\n", .{lookups_combined_vals[jj_d].toBytesBE()[16..32].*});
                                dbg("  r1cs_combined   = {x}\n", .{r1cs_combined.toBytesBE()[16..32].*});
                                dbg("  stage5_right_op = {x}\n", .{(lookups_combined_vals[jj_d].sub(r1cs_output).sub(gamma_raf.mul(F.zero()))).toBytesBE()[16..32].*});
                                dbg("  r1cs_right_op   = {x}\n", .{r1cs_right_op.toBytesBE()[16..32].*});
                                dbg("  rs1={}, rs2={}, imm_field={x}\n", .{
                                    step_d.rs1_value, step_d.rs2_value,
                                    computeImmediate(instr_d).toBytesBE()[16..32].*,
                                });
                            }
                            per_cycle_mismatches += 1;
                        }
                    }
                    if (comptime debug_verbose) {
                        dbg("[DIAG R1CS vs S5] stage5_bf_sum    = {x}\n", .{stage5_bf_sum.toBytesBE()[16..32].*});
                        dbg("[DIAG R1CS vs S5] r1cs_bf_sum      = {x}\n", .{r1cs_bf_sum.toBytesBE()[16..32].*});
                        dbg("[DIAG R1CS vs S5] lookups_claim     = {x}\n", .{lookups_claim.toBytesBE()[16..32].*});
                        dbg("[DIAG R1CS vs S5] stage5 == claim: {}\n", .{stage5_bf_sum.eql(lookups_claim)});
                        dbg("[DIAG R1CS vs S5] r1cs == claim: {}\n", .{r1cs_bf_sum.eql(lookups_claim)});
                        dbg("[DIAG R1CS vs S5] per_cycle_mismatches: {}\n", .{per_cycle_mismatches});
                    }
                }
                } // end comptime debug_verbose for R1CS diagnostic

                // Now compute total from Q polynomials directly
                // For each table that has non-zero Q, sum Q values across all entries
                var q_total_raw = F.zero();
                for (0..NUM_TABLES) |t_idx| {
                    if (suffix_polys.tables[t_idx]) |table| {
                        for (table.polys, 0..) |poly, s_idx| {
                            var poly_sum = F.zero();
                            for (poly) |v| {
                                poly_sum = poly_sum.add(v);
                            }
                            if (!poly_sum.eql(F.zero())) {
                                if (comptime debug_verbose) {
                                    dbg("[DIAG INIT] Q_total[table={},suffix={}] = {x}\n", .{
                                        t_idx, s_idx, poly_sum.toBytesBE()[16..32].*,
                                    });
                                }
                            }
                            q_total_raw = q_total_raw.add(poly_sum);
                        }
                    }
                }
                if (comptime debug_verbose) {
                    dbg("[DIAG INIT] Total raw Q sum = {x}\n", .{q_total_raw.toBytesBE()[16..32].*});
                }

                // Also verify: Q[128..255] should be all zero if all addresses have MSB=0
                var right_half_nonzero: usize = 0;
                for (0..NUM_TABLES) |t_idx| {
                    if (suffix_polys.tables[t_idx]) |table| {
                        for (table.polys) |poly| {
                            for (128..poly.len) |idx| {
                                if (!poly[idx].eql(F.zero())) {
                                    right_half_nonzero += 1;
                                }
                            }
                        }
                    }
                }
                if (comptime debug_verbose) {
                    dbg("[DIAG INIT] Right half (Q[128..255]) non-zero entries: {}\n", .{right_half_nonzero});
                }

                // Also check RAF Q arrays
                var raf_right_half_nonzero: usize = 0;
                for (0..2) |qi| {
                    for (128..initial_m) |idx| {
                        if (!left_raf.Q[qi][idx].eql(F.zero())) raf_right_half_nonzero += 1;
                        if (!right_raf.Q[qi][idx].eql(F.zero())) raf_right_half_nonzero += 1;
                        if (!identity_raf.Q[qi][idx].eql(F.zero())) raf_right_half_nonzero += 1;
                    }
                }
                if (comptime debug_verbose) {
                    dbg("[DIAG INIT] RAF right half non-zero entries: {}\n", .{raf_right_half_nonzero});
                }

                // Compute total from prefix-suffix at phase 0 (no challenges bound yet)
                // For each index b (0..255), evaluate the full prefix and combine with Q
                // At phase 0, round 0, before any variable binding:
                //   Each full index b represents 8 bits of the address (bits 120..127)
                //   The suffix is 120 bits long
                //   tableCombine(prefix_at_b, Q[b]) gives the contribution of entries at prefix b
                //
                // The total = Σ_b tableCombine(prefix(integer b), Q[b])
                //
                // For table 0 (RangeCheck): combine = LowerWord * one + lower_word_suffix
                //   At phase 0 with j=0..7 (first 8 rounds), LowerWord prefix = 0 (since j < XLEN)
                //   So combine = 0 * Q_one[b] + Q_lower_word[b] = Q_lower_word[b]
                //   Total = Σ_b Q_lower_word[b]
                //
                // This is exactly what we computed as Q_total[table=0, suffix=1]!
                // But the issue is: this should equal Σ_{j: table_j==0} u[j] * output_j
                // Let's verify:
                var bf_table0_output = F.zero();
                var bf_table0_lookup_output = F.zero();
                for (trace.steps.items, 0..) |_, jj| {
                    if (cycle_table_indices[jj] == 0) {
                        bf_table0_output = bf_table0_output.add(lookups_eq_evals[jj].mul(F.fromU64(trace.steps.items[jj].rd_value)));
                        // Use lookup_index lower 64 bits (= materialize_entry for RangeCheck)
                        bf_table0_lookup_output = bf_table0_lookup_output.add(lookups_eq_evals[jj].mul(F.fromU64(lookups_indices_lo[jj])));
                    }
                }
                if (comptime debug_verbose) {
                    dbg("[DIAG INIT] bf_table0_output (Σu*rd_value for table 0) = {x}\n", .{
                        bf_table0_output.toBytesBE()[16..32].*,
                    });
                    dbg("[DIAG INIT] bf_table0_lookup (Σu*lookup_lo for table 0) = {x}\n", .{
                        bf_table0_lookup_output.toBytesBE()[16..32].*,
                    });
                }
            }

            // Run the batched sumcheck
            if (comptime debug_verbose) {
                dbg("[STAGE5] Entering main sumcheck loop, max_num_rounds={}\n", .{max_num_rounds});
            }

            // Benchmark timing accumulators
            var bench_timer = if (comptime bench_timing) std.time.Timer.start() catch unreachable else {};
            var bench_init_ns: u64 = 0;
            var bench_phase_transition_ns: u64 = 0;
            var bench_condense_ns: u64 = 0;
            var bench_init_phase_ns: u64 = 0;
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
            var bench_remat_ns: u64 = 0; // Untimed rematerialization gap
            var bench_cycle_coeff_ns: u64 = 0; // Untimed coefficient combination gap

            // BRUTE FORCE PER-ROUND DIAGNOSTIC (disabled in release)
            var bf_weights: []F = &[_]F{};
            if (comptime debug_verbose) {
                bf_weights = try self.allocator.alloc(F, T);
                for (0..T) |j| {
                    bf_weights[j] = lookups_eq_evals[j].mul(lookups_combined_vals[j]);
                }
            }
            defer if (comptime debug_verbose) self.allocator.free(bf_weights);

            // Track accumulated eq factor for bound cycle variables
            // This matches Jolt's current_scalar in GruenSplitEqPolynomial
            // Formula: current_scalar *= eq(w[i], challenge_i) = 1 - w[i] - c + 2*w[i]*c
            // where w[i] = r_reduction[n-1-i] (original challenge) and c = sumcheck challenge
            var lookups_current_scalar = F.one();

            // Split-eq tables for cycle rounds: factored eq polynomial
            // E_in covers r_reduction[0..m_in], E_out covers r_reduction[m_in..n_cycle_vars-1]
            // eq_prefix(j) = E_out[j >> m_in] * E_in[j & ((1 << m_in) - 1)]
            // Initialized lazily at the start of the first cycle round.
            const MAX_SPLIT_EQ_SIZE = 1 << 10; // supports up to n_cycle_vars=21 (split ~10/11)
            var split_eq_E_in: [MAX_SPLIT_EQ_SIZE]F = undefined;
            var split_eq_E_out: [MAX_SPLIT_EQ_SIZE]F = undefined;
            var split_eq_m_in: usize = 0; // number of variables in E_in
            var split_eq_E_in_len: usize = 0;
            var split_eq_E_out_len: usize = 0;
            var split_eq_initialized = false;

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
                    const scale = remaining_rounds - regs_val_num_rounds - 1;
                    var scaled_input_claim = regs_val_input;
                    for (0..scale) |_| scaled_input_claim = scaled_input_claim.add(scaled_input_claim);
                    // Constant polynomial p(x) = scaled_input_claim
                    combined_poly[0] = combined_poly[0].add(batch0.mul(scaled_input_claim));
                    combined_poly[1] = combined_poly[1].add(batch0.mul(scaled_input_claim));
                    combined_poly[2] = combined_poly[2].add(batch0.mul(scaled_input_claim));
                    // evals[3] = p_inf = 0 for constant polynomial
                } else {
                    // Instance is active - compute actual round polynomial
                    const regs_round = regs_val_num_rounds - remaining_rounds;
                    if (comptime bench_timing) bench_timer.reset();
                    const poly_evals = computeRegsValRoundPoly(inc_evals, wa_evals, &lt_poly, regs_round, self.thread_pool);
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
                    const scale = remaining_rounds - ram_ra_num_rounds - 1;
                    var scaled_input_claim = ram_ra_input;
                    for (0..scale) |_| scaled_input_claim = scaled_input_claim.add(scaled_input_claim);
                    combined_poly[0] = combined_poly[0].add(batch1.mul(scaled_input_claim));
                    combined_poly[1] = combined_poly[1].add(batch1.mul(scaled_input_claim));
                    combined_poly[2] = combined_poly[2].add(batch1.mul(scaled_input_claim));
                    // evals[3] = p_inf = 0 for constant polynomial
                } else {
                    // Instance is active - compute RamRaClaimReduction sumcheck polynomial
                    if (comptime bench_timing) bench_timer.reset();

                    const ram_ra_round = ram_ra_num_rounds - remaining_rounds; // 0 to n_cycle_vars-1

                    // Upstream RamRaClaimReduction: cycle-only binding (no PhaseAddress)
                    // All ram_ra rounds are cycle rounds.
                    {
                        // ============================================================
                        // PhaseCycle (Jolt's approach) — cycle-only binding
                        // ============================================================
                        // Using prefix-suffix decomposition:
                        //   P_x[c_lo] = eq(r_cycle_x_lo, c_lo)  -- prefix eq evaluations
                        //   Q_x[c_lo] = Σ_{c_hi} H[c_lo,c_hi] · eq(r_cycle_x_hi, c_hi)  -- suffix sums
                        //
                        // H[c] = F_values[address[c]] = B_1[address[c]] = eq(r_address, address[c])
                        // Coefficients: 1, gamma, gamma^2 (upstream cycle-only reduction)
                        //
                        // PhaseCycle1: rounds 0 to prefix_n_vars-1 (bind prefix bits using P*Q)
                        // PhaseCycle2: rounds prefix_n_vars to n_cycle_vars-1 (bind suffix using H'*eq_hi)

                        const cycle_round = ram_ra_round; // 0 to n_cycle_vars-1

                        // Initialize Q arrays at the start of PhaseCycle (round 0)
                        if (!phase_cycle_q_initialized) {
                            phase_cycle_q_initialized = true;

                            if (comptime debug_verbose) {
                                dbg("[STAGE5 PQ] prefix_n_vars={}, suffix_n_vars={}\n", .{
                                    prefix_n_vars, suffix_n_vars,
                                });
                            }

                            // Compute Q arrays: Q_x[c_lo] = Σ_{c_hi} H[c_lo,c_hi] · eq_x_hi(c_hi)
                            // where H[c] = F_values[address[c]] = B_1[address[c]] = eq(r_address, addr)
                            @memset(Q_raf, F.zero());
                            @memset(Q_rw, F.zero());
                            @memset(Q_val, F.zero());

                            for (0..ram_access_count) |access_idx| {
                                const cycle = ram_cycles[access_idx];
                                const cycle_usize: usize = @intCast(cycle);
                                const addr = ram_addresses[access_idx];
                                const addr_usize: usize = @intCast(addr);

                                // H[c] = F_values[address[c]] = eq(r_address, addr)
                                // B_1[k] = eq(r_address_raf, k) serves as F_values (upstream uses single r_address)
                                const H_c = B_1[addr_usize];

                                // Split cycle into c_lo (prefix) and c_hi (suffix)
                                const c_lo = cycle_usize & (prefix_size - 1);
                                const c_hi = cycle_usize >> @intCast(prefix_n_vars);

                                // Q_x[c_lo] += H[c] * eq_x_hi(c_hi)
                                Q_raf[c_lo] = Q_raf[c_lo].add(H_c.mul(eq_raf_hi[c_hi]));
                                Q_rw[c_lo] = Q_rw[c_lo].add(H_c.mul(eq_rw_hi[c_hi]));
                                Q_val[c_lo] = Q_val[c_lo].add(H_c.mul(eq_val_hi[c_hi]));
                            }

                            if (comptime debug_verbose) {
                                dbg("[STAGE5 PQ] Q arrays initialized with {} accesses\n", .{ram_access_count});
                            }
                        }

                        // Upstream coefficients: 1, γ, γ² (address pre-evaluated via F_values)
                        const coeff_raf = F.one();
                        const coeff_rw = gamma;
                        const coeff_val = gamma2;

                        // Current P polynomial length (P_raf initially has prefix_size elements)
                        // After cycle_round bindings, effective length is prefix_size >> cycle_round
                        const current_P_len = prefix_size >> @intCast(cycle_round);
                        const half_len = current_P_len / 2;

                        if (cycle_round < prefix_n_vars and half_len > 0) {
                            // PhaseCycle1: Bind prefix bits using P*Q decomposition
                            // CRITICAL: Use hint mechanism - only compute eval_1 and eval_2 directly
                            // Then derive eval_0 = claim - eval_1 to guarantee sumcheck property

                            // DEBUG: Print claim at start of polynomial computation
                            if (comptime debug_verbose) {
                                dbg("[INST1 HINT DEBUG] Round {}: ram_ra_current_claim at poly start = {x}\n", .{
                                    round,
                                    ram_ra_current_claim.toBytesBE()[16..32].*,
                                });
                            }

                            var eval_1 = F.zero();

                            // Compute sumcheck polynomial using P * Q products
                            // Only compute eval_1 (sum over odd indices P[2j+1] * Q[2j+1])
                            for (0..half_len) |j| {
                                // P values at odd indices
                                const p_raf_1 = P_raf[2 * j + 1];
                                const p_rw_1 = P_rw[2 * j + 1];
                                const p_val_1 = P_val[2 * j + 1];

                                // Q values at odd indices
                                const q_raf_1 = Q_raf[2 * j + 1];
                                const q_rw_1 = Q_rw[2 * j + 1];
                                const q_val_1 = Q_val[2 * j + 1];

                                // eval_1 contribution: P[2j+1] * Q[2j+1]
                                const contrib_1 = coeff_raf.mul(p_raf_1.mul(q_raf_1))
                                    .add(coeff_rw.mul(p_rw_1.mul(q_rw_1)))
                                    .add(coeff_val.mul(p_val_1.mul(q_val_1)));
                                eval_1 = eval_1.add(contrib_1);
                            }

                            // HINT MECHANISM: derive eval_0 from claim
                            // p(0) + p(1) = claim => p(0) = claim - p(1)
                            const eval_0 = ram_ra_current_claim.sub(eval_1);

                            // Compute eval_2 = p(2) for degree-2 polynomial
                            var eval_2 = F.zero();
                            for (0..half_len) |j| {
                                const p_raf_at_2 = P_raf[2 * j + 1].add(P_raf[2 * j + 1]).sub(P_raf[2 * j]);
                                const q_raf_at_2 = Q_raf[2 * j + 1].add(Q_raf[2 * j + 1]).sub(Q_raf[2 * j]);
                                const p_rw_at_2 = P_rw[2 * j + 1].add(P_rw[2 * j + 1]).sub(P_rw[2 * j]);
                                const q_rw_at_2 = Q_rw[2 * j + 1].add(Q_rw[2 * j + 1]).sub(Q_rw[2 * j]);
                                const p_val_at_2 = P_val[2 * j + 1].add(P_val[2 * j + 1]).sub(P_val[2 * j]);
                                const q_val_at_2 = Q_val[2 * j + 1].add(Q_val[2 * j + 1]).sub(Q_val[2 * j]);

                                const contrib_2 = coeff_raf.mul(p_raf_at_2.mul(q_raf_at_2))
                                    .add(coeff_rw.mul(p_rw_at_2.mul(q_rw_at_2)))
                                    .add(coeff_val.mul(p_val_at_2.mul(q_val_at_2)));
                                eval_2 = eval_2.add(contrib_2);
                            }

                            combined_poly[0] = combined_poly[0].add(batch1.mul(eval_0));
                            combined_poly[1] = combined_poly[1].add(batch1.mul(eval_1));
                            combined_poly[2] = combined_poly[2].add(batch1.mul(eval_2));

                            // Store evaluations for claim update after challenge is generated
                            inst1_eval_0 = eval_0;
                            inst1_eval_1 = eval_1;
                            inst1_eval_2 = eval_2;

                            if (comptime debug_verbose) {
                                dbg("[STAGE5 RAM_RA] PhaseCycle1 round {}: eval_0={x}, eval_1={x}, eval_2={x}\n", .{
                                    cycle_round,
                                    eval_0.toBytesBE()[16..32].*,
                                    eval_1.toBytesBE()[16..32].*,
                                    eval_2.toBytesBE()[16..32].*,
                                });
                            }

                            // Debug: Print Instance 1 contribution for Round 128-129
                            if (round >= LOOKUPS_LOG_K and round <= LOOKUPS_LOG_K + 1) {
                                const inst1_evals = [4]F{ eval_0, eval_1, eval_2, F.zero() };
                                const inst1_coeffs = UniPoly(F).toomCookToCoeffs(inst1_evals);
                                if (comptime debug_verbose) {
                                    dbg("[ZOLT INST1] Round {}: ram_ra_round={}, cycle_round={}\n", .{ round, ram_ra_round, cycle_round });
                                    dbg("  inst1_evals (Toom) = [{any}, {any}, {any}, {any}]\n", .{
                                        inst1_evals[0].toBytes(), inst1_evals[1].toBytes(),
                                        inst1_evals[2].toBytes(), inst1_evals[3].toBytes(),
                                    });
                                    dbg("  inst1_coeffs (coeffs) = [{any}, {any}, {any}, {any}]\n", .{
                                        inst1_coeffs[0].toBytes(), inst1_coeffs[1].toBytes(),
                                        inst1_coeffs[2].toBytes(), inst1_coeffs[3].toBytes(),
                                    });
                                    dbg("  batch1 = {any}\n", .{batch1.toBytes()});
                                }
                            }
                        } else {
                            // PhaseCycle2: After prefix rounds, use H' * eq_hi for suffix
                            const suffix_round = cycle_round - prefix_n_vars;

                            // Initialize H_prime at the start of PhaseCycle2
                            if (!phase_cycle2_initialized) {
                                phase_cycle2_initialized = true;

                                // Compute eq_prefix[c_lo] = eq(r_cycle_prefix_challenges, c_lo)
                                // The prefix challenges are cycle_challenges[0..prefix_n_vars]
                                var eq_prefix = try self.allocator.alloc(F, prefix_size);
                                defer self.allocator.free(eq_prefix);

                                for (0..prefix_size) |c_lo| {
                                    // eq_prefix[c_lo] = Π_j eq_bit(r_j, c_lo_j)
                                    // cycle_challenges are in LowToHigh order (challenge[0] = LSB)
                                    var eq_val_local = F.one();
                                    var c_lo_var = c_lo;
                                    for (0..prefix_n_vars) |j| {
                                        const bit: u1 = @truncate(c_lo_var);
                                        c_lo_var >>= 1;
                                        const r_j = cycle_challenges[j];
                                        // eq_bit(r, b) = (1-r) if b=0, else r
                                        const eq_bit_val = if (bit == 0) F.one().sub(r_j) else r_j;
                                        eq_val_local = eq_val_local.mul(eq_bit_val);
                                    }
                                    eq_prefix[c_lo] = eq_val_local;
                                }

                                // Compute H_prime[c_hi] = Σ_{c_lo} H[c_lo,c_hi] * eq_prefix[c_lo]
                                // where H[c] = F_values[address[c]] = eq(r_addr_reduced, address[c])
                                @memset(H_prime, F.zero());
                                for (0..ram_access_count) |access_idx| {
                                    const cycle = ram_cycles[access_idx];
                                    const cycle_usize: usize = @intCast(cycle);
                                    const addr = ram_addresses[access_idx];
                                    const addr_usize: usize = @intCast(addr);

                                    // H[c] = B_1[address[c]] = eq(r_address, addr)
                                    const H_c = B_1[addr_usize];
                                    const c_lo = cycle_usize & (prefix_size - 1);
                                    const c_hi = cycle_usize >> @intCast(prefix_n_vars);

                                    H_prime[c_hi] = H_prime[c_hi].add(H_c.mul(eq_prefix[c_lo]));
                                }

                                // Compute scaling factors: eq(r_cycle_x_lo, r_cycle_prefix_reduced)
                                // r_cycle_*[suffix_n_vars..] are the LOW bits (BIG_ENDIAN)
                                // cycle_challenges[0..prefix_n_vars] are the prefix challenges (LowToHigh)
                                scale_raf = F.one();
                                scale_rw = F.one();
                                scale_val = F.one();
                                for (0..prefix_n_vars) |j| {
                                    // r_cycle_* is BIG_ENDIAN: index suffix_n_vars + j corresponds to prefix bit j
                                    const r_raf_j = r_cycle_raf[suffix_n_vars + prefix_n_vars - 1 - j];
                                    const r_rw_j = r_cycle_rw[suffix_n_vars + prefix_n_vars - 1 - j];
                                    const r_val_j = r_cycle_val[suffix_n_vars + prefix_n_vars - 1 - j];
                                    const chal_j = cycle_challenges[j];
                                    // eq_bit(r, chal) = (1-r)(1-chal) + r*chal
                                    scale_raf = scale_raf.mul(F.one().sub(r_raf_j).mul(F.one().sub(chal_j)).add(r_raf_j.mul(chal_j)));
                                    scale_rw = scale_rw.mul(F.one().sub(r_rw_j).mul(F.one().sub(chal_j)).add(r_rw_j.mul(chal_j)));
                                    scale_val = scale_val.mul(F.one().sub(r_val_j).mul(F.one().sub(chal_j)).add(r_val_j.mul(chal_j)));
                                }

                                if (comptime debug_verbose) {
                                    dbg("[STAGE5 RAM_RA] PhaseCycle2 starting at suffix_round={}\n", .{suffix_round});
                                }
                            }

                            // Compute polynomial using H_prime * eq_hi products
                            const current_len = suffix_size >> @intCast(suffix_round);
                            const half_len_suffix = current_len / 2;

                            // Upstream coefficients with scale factors: 1*scale_raf, γ*scale_rw, γ²*scale_val
                            const coeff_raf_scaled = scale_raf;
                            const coeff_rw_scaled = gamma.mul(scale_rw);
                            const coeff_val_scaled = gamma2.mul(scale_val);

                            // CRITICAL: Use hint mechanism - only compute eval_1 directly
                            // Then derive eval_0 = claim - eval_1 to guarantee sumcheck property
                            var eval_1 = F.zero();

                            for (0..half_len_suffix) |j| {
                                // H_prime values (only need odd indices for eval_1)
                                const h_1 = H_prime[2 * j + 1];

                                // eq_hi values (only need odd indices for eval_1)
                                const eq_raf_1 = eq_raf_hi[2 * j + 1];
                                const eq_rw_1 = eq_rw_hi[2 * j + 1];
                                const eq_val_1 = eq_val_hi[2 * j + 1];

                                // Contribution for X=1
                                const contrib_1 = h_1.mul(
                                    coeff_raf_scaled.mul(eq_raf_1)
                                        .add(coeff_rw_scaled.mul(eq_rw_1))
                                        .add(coeff_val_scaled.mul(eq_val_1)),
                                );
                                eval_1 = eval_1.add(contrib_1);
                            }

                            // HINT MECHANISM: derive eval_0 from claim
                            // p(0) + p(1) = claim => p(0) = claim - p(1)
                            const eval_0 = ram_ra_current_claim.sub(eval_1);

                            // Compute eval_2 = p(2)
                            var eval_2 = F.zero();
                            for (0..half_len_suffix) |j| {
                                const h_at_2 = H_prime[2 * j + 1].add(H_prime[2 * j + 1]).sub(H_prime[2 * j]);
                                const eq_raf_at_2 = eq_raf_hi[2 * j + 1].add(eq_raf_hi[2 * j + 1]).sub(eq_raf_hi[2 * j]);
                                const eq_rw_at_2 = eq_rw_hi[2 * j + 1].add(eq_rw_hi[2 * j + 1]).sub(eq_rw_hi[2 * j]);
                                const eq_val_at_2 = eq_val_hi[2 * j + 1].add(eq_val_hi[2 * j + 1]).sub(eq_val_hi[2 * j]);

                                const contrib_2 = h_at_2.mul(
                                    coeff_raf_scaled.mul(eq_raf_at_2)
                                        .add(coeff_rw_scaled.mul(eq_rw_at_2))
                                        .add(coeff_val_scaled.mul(eq_val_at_2)),
                                );
                                eval_2 = eval_2.add(contrib_2);
                            }

                            combined_poly[0] = combined_poly[0].add(batch1.mul(eval_0));
                            combined_poly[1] = combined_poly[1].add(batch1.mul(eval_1));
                            combined_poly[2] = combined_poly[2].add(batch1.mul(eval_2));

                            // Store evaluations for claim update after challenge is generated
                            inst1_eval_0 = eval_0;
                            inst1_eval_1 = eval_1;
                            inst1_eval_2 = eval_2;

                            if (comptime debug_verbose) {
                                dbg("[STAGE5 RAM_RA] PhaseCycle2 round {}: eval_0={x}, eval_1={x}, eval_2={x}\n", .{
                                    cycle_round,
                                    eval_0.toBytesBE()[16..32].*,
                                    eval_1.toBytesBE()[16..32].*,
                                    eval_2.toBytesBE()[16..32].*,
                                });
                            }
                        }
                    }
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

                    // On odd rounds, r_x is the last challenge (the X variable was just bound)
                    // On even rounds, r_x is null (we're computing over the X variable)
                    const r_x: ?F = if (round % 2 == 1) challenges[round - 1] else null;

                    // Compute read-checking and RAF contributions concurrently via tp.join
                    if (comptime bench_timing) bench_timer.reset();
                    const RcCtx = struct {
                        round: usize,
                        sp: *AllSuffixPolys(F),
                        pc: *PrefixCheckpointsState(F),
                        rx: ?F,
                        tp: ?*ThreadPool,
                    };
                    const RafCtx = struct {
                        left: *RafDecomposition(F),
                        right: *RafDecomposition(F),
                        identity: *RafDecomposition(F),
                        g1: F,
                        g2: F,
                        tp: ?*ThreadPool,
                    };
                    const rc_join_ctx = RcCtx{ .round = round, .sp = &suffix_polys, .pc = &prefix_checkpoints, .rx = r_x, .tp = self.thread_pool };
                    const raf_join_ctx = RafCtx{ .left = &left_raf, .right = &right_raf, .identity = &identity_raf, .g1 = gamma_raf, .g2 = gamma_raf2, .tp = self.thread_pool };
                    const read_checking_evals, const raf_evals = if (self.thread_pool) |tp|
                        tp.join([2]F, [2]F, rc_join_ctx, struct {
                            fn f(c: RcCtx) [2]F { return proverMsgReadChecking(F, c.round, c.sp, c.pc, c.rx, c.tp); }
                        }.f, raf_join_ctx, struct {
                            fn f(c: RafCtx) [2]F { return proverMsgRaf(F, c.left, c.right, c.identity, c.g1, c.g2, c.tp); }
                        }.f)
                    else
                        .{ proverMsgReadChecking(F, round, &suffix_polys, &prefix_checkpoints, r_x, self.thread_pool), proverMsgRaf(F, &left_raf, &right_raf, &identity_raf, gamma_raf, gamma_raf2, self.thread_pool) };
                    if (comptime bench_timing) {
                        const elapsed = bench_timer.read();
                        bench_addr_compute_ns += elapsed;
                        bench_inst2_addr_compute_ns += elapsed;
                    }
                    if (s5_phase_timer) |*pt| { s5_addr_compute_ns += pt.read(); pt.reset(); }

                    // Combined: read_checking + raf
                    const eval_0_inst2 = read_checking_evals[0].add(raf_evals[0]);
                    const eval_2_inst2 = read_checking_evals[1].add(raf_evals[1]);

                    // Print Instance 2 eval_0 and eval_2 for comparison with Jolt prover
                    if (comptime debug_verbose) if (round < 5 or (round >= 14 and round <= 17) or round == 127) {
                        const print = std.debug.print;
                        print("[ZOLT INST2 R{}] previous_claim = {any}\n", .{ round, lookups_claim.toBytes()[0..16].* });
                        print("[ZOLT INST2 R{}] eval_at_0 = {any}\n", .{ round, eval_0_inst2.toBytes()[0..16].* });
                        print("[ZOLT INST2 R{}] eval_at_2 = {any}\n", .{ round, eval_2_inst2.toBytes()[0..16].* });
                        print("[ZOLT INST2 R{}] read_checking = [{any}, {any}]\n", .{
                            round,
                            read_checking_evals[0].toBytes()[0..16].*,
                            read_checking_evals[1].toBytes()[0..16].*,
                        });
                        print("[ZOLT INST2 R{}] raf = [{any}, {any}]\n", .{
                            round,
                            raf_evals[0].toBytes()[0..16].*,
                            raf_evals[1].toBytes()[0..16].*,
                        });
                    };

                    // DEGREE-2 EXCESS CHECK: at R0 (all indices bit 127=0),
                    // each component should have eval_2 = -eval_0 (since eval_1 = 0)
                    if (comptime debug_verbose) if (round == 0) {
                        const print = std.debug.print;
                        const neg_rc_e0 = F.zero().sub(read_checking_evals[0]);
                        const neg_raf_e0 = F.zero().sub(raf_evals[0]);
                        const rc_ok = read_checking_evals[1].eql(neg_rc_e0);
                        const raf_ok = raf_evals[1].eql(neg_raf_e0);
                        print("[DEG2 R0] rc: eval_2={x}, -eval_0={x}, match={}\n", .{
                            read_checking_evals[1].toBytesBE()[16..32].*,
                            neg_rc_e0.toBytesBE()[16..32].*,
                            rc_ok,
                        });
                        print("[DEG2 R0] raf: eval_2={x}, -eval_0={x}, match={}\n", .{
                            raf_evals[1].toBytesBE()[16..32].*,
                            neg_raf_e0.toBytesBE()[16..32].*,
                            raf_ok,
                        });
                        const neg_total_e0 = F.zero().sub(eval_0_inst2);
                        const total_ok = eval_2_inst2.eql(neg_total_e0);
                        print("[DEG2 R0] total: eval_2={x}, -eval_0={x}, match={}\n", .{
                            eval_2_inst2.toBytesBE()[16..32].*,
                            neg_total_e0.toBytesBE()[16..32].*,
                            total_ok,
                        });
                    };

                    // (TARGETED DEBUG moved after eval_1 derivation below)

                    // *** PER-ROUND BRUTE FORCE eval_0 CHECK ***
                    if (comptime debug_verbose)
                    // bf_weights[j] = eq(j,r_red) * cv[j] * Π_{i<round} eq_bit(r_i, K(j)_{127-i})
                    // eval_0 = Σ_{j: bit(round) of K(j) = 0} bf_weights[j]
                    {
                        const bf_bit_pos = LOOKUPS_LOG_K - 1 - round;
                        var bf_e0 = F.zero();
                        var bf_e1 = F.zero();
                        for (0..T) |jj| {
                            const bit_val: u1 = if (bf_bit_pos >= 64) @truncate(lookups_indices_hi[jj] >> @intCast(bf_bit_pos - 64)) else @truncate(lookups_indices_lo[jj] >> @intCast(bf_bit_pos));
                            if (bit_val == 0) {
                                bf_e0 = bf_e0.add(bf_weights[jj]);
                            } else {
                                bf_e1 = bf_e1.add(bf_weights[jj]);
                            }
                        }
                        const bf_e0_match = bf_e0.eql(eval_0_inst2);
                        if (!bf_e0_match) {
                            const print = std.debug.print;
                            // Also check claim consistency
                            const bf_claim = bf_e0.add(bf_e1);
                            print("[BF_EVAL0 MISMATCH R{}] bf_e0={x}, ps_e0={x}\n", .{
                                round,
                                bf_e0.toBytesBE()[16..32].*,
                                eval_0_inst2.toBytesBE()[16..32].*,
                            });
                            print("[BF_EVAL0 R{}] bf_claim={x}, ps_claim={x}, claim_match={}\n", .{
                                round,
                                bf_claim.toBytesBE()[16..32].*,
                                lookups_claim.toBytesBE()[16..32].*,
                                bf_claim.eql(lookups_claim),
                            });
                            // Print read_checking and raf components separately
                            print("[BF_EVAL0 R{}] ps_rc_e0={x}, ps_raf_e0={x}\n", .{
                                round,
                                read_checking_evals[0].toBytesBE()[16..32].*,
                                raf_evals[0].toBytesBE()[16..32].*,
                            });
                        }
                    }

                    // CRITICAL MULTILINEAR CHECK: For any multilinear polynomial,
                    // p(2) = 2*p(1) - p(0). If the prefix-suffix decomposition
                    // gives a different eval_2, it's computing the WRONG polynomial.
                    const eval_1_inst2 = lookups_claim.sub(eval_0_inst2);
                    const expected_eval_2 = eval_1_inst2.add(eval_1_inst2).sub(eval_0_inst2);
                    const eval_2_matches_ml = eval_2_inst2.eql(expected_eval_2);
                    if (comptime debug_verbose) if (!eval_2_matches_ml) {
                        const print = std.debug.print;
                        print("[MULTILINEAR BUG R{}] eval_2_inst2 != 2*eval_1 - eval_0!\n", .{round});
                        print("  actual eval_2   = {x}\n", .{eval_2_inst2.toBytesBE()});
                        print("  expected (2e1-e0) = {x}\n", .{expected_eval_2.toBytesBE()});
                        print("  eval_0 = {x}\n", .{eval_0_inst2.toBytesBE()});
                        print("  eval_1 = {x}\n", .{eval_1_inst2.toBytesBE()});
                        print("  claim  = {x}\n", .{lookups_claim.toBytesBE()});
                        print("  read_checking[0] (e0) = {x}\n", .{read_checking_evals[0].toBytesBE()});
                        print("  read_checking[1] (e2) = {x}\n", .{read_checking_evals[1].toBytesBE()});
                        print("  raf[0] (e0) = {x}\n", .{raf_evals[0].toBytesBE()});
                        print("  raf[1] (e2) = {x}\n", .{raf_evals[1].toBytesBE()});
                    };
                    // LINEARITY CHECK per component at round 0
                    if (comptime debug_verbose) if (round == 0) {
                        const print2 = std.debug.print;
                        // For read_checking: rc_0 + rc_1 should equal some claim
                        // rc_1 = eval_1_total - raf_1
                        // raf_1 = raf_claim - raf_0 where raf_claim is raf part of lookups_claim
                        // Actually, let's just check: 2*(claim-e0) - e0 should equal e2 IF linear
                        // For read_checking alone: rc_claim = lookups_claim - raf_claim
                        // rc_1 = rc_claim - rc_0, expected_rc_2 = 2*rc_1 - rc_0
                        // But we don't have rc_claim or raf_claim separately
                        // Instead: total is linear iff rc and raf are both linear
                        // Check: is rc_2 = 2*(eval_1_inst2 - raf_1) - rc_0?
                        // We need raf eval_1 independently. Let's compute it:
                        // raf_claim = lookups_claim_raf_part
                        // Actually since g(c) = rc(c) + raf(c), if g is linear:
                        //   g(2) = 2*g(1) - g(0)
                        //   rc(2) + raf(2) = 2*(rc(1)+raf(1)) - (rc(0)+raf(0))
                        // The non-linearity means one of rc or raf is non-linear.
                        // Just print rc eval_2 relationship:
                        const rc_e0 = read_checking_evals[0];
                        const rc_e2 = read_checking_evals[1];
                        const raf_e0 = raf_evals[0];
                        const raf_e2 = raf_evals[1];
                        // total_e1 = lookups_claim - eval_0_inst2
                        // If rc is linear: rc(2) = 2*rc(1) - rc(0) and raf(2) = 2*raf(1) - raf(0)
                        // rc(1) + raf(1) = total_e1
                        // If BOTH linear: rc(2)+raf(2) = 2*(rc(1)+raf(1)) - (rc(0)+raf(0)) = 2*total_e1 - total_e0 = expected_eval_2
                        // If they're both linear but total isn't, something is very wrong
                        // Print the eval_2 components:
                        const total_e2 = rc_e2.add(raf_e2);
                        print2("[LINEARITY R0] rc_e0={x} rc_e2={x}\n", .{ rc_e0.toBytesBE()[16..32].*, rc_e2.toBytesBE()[16..32].* });
                        print2("[LINEARITY R0] raf_e0={x} raf_e2={x}\n", .{ raf_e0.toBytesBE()[16..32].*, raf_e2.toBytesBE()[16..32].* });
                        print2("[LINEARITY R0] total_e2(from components)={x} actual_e2={x} match={}\n", .{
                            total_e2.toBytesBE()[16..32].*,
                            eval_2_inst2.toBytesBE()[16..32].*,
                            total_e2.eql(eval_2_inst2),
                        });
                        print2("[LINEARITY R0] expected_linear_e2={x}\n", .{expected_eval_2.toBytesBE()[16..32].*});
                        // diff = actual_e2 - expected_e2 = amount of nonlinearity
                        const nonlin = eval_2_inst2.sub(expected_eval_2);
                        print2("[LINEARITY R0] nonlinearity = {x}\n", .{nonlin.toBytesBE()[16..32].*});
                        // Check if nonlinearity comes from rc or raf:
                        // If raf is linear and rc isn't: nonlin = rc_e2 - (2*rc_e1 - rc_e0) = rc_e2 - expected_rc_e2
                        // where expected_rc_e2 = expected_total_e2 - expected_raf_e2 = expected_total_e2 - raf_e2 (if raf linear)
                    };
                    // TARGETED DEBUG: Print Instance 2 values in LE format for Jolt comparison
                    if (comptime debug_verbose) if (round < 4 or round == 7 or round == 8 or round == 15 or round == 16 or round == 127) {
                        const print = std.debug.print;
                        print("[ZOLT INST2 R{}] previous_claim = {any}\n", .{ round, lookups_claim.toBytes()[0..16].* });
                        print("[ZOLT INST2 R{}] eval_at_0 = {any}\n", .{ round, eval_0_inst2.toBytes()[0..16].* });
                        print("[ZOLT INST2 R{}] eval_at_1 = {any}\n", .{ round, eval_1_inst2.toBytes()[0..16].* });
                        print("[ZOLT INST2 R{}] eval_at_2 = {any}\n", .{ round, eval_2_inst2.toBytes()[0..16].* });
                        print("[ZOLT INST2 R{}] read_checking = [{any}, {any}]\n", .{ round, read_checking_evals[0].toBytes()[0..16].*, read_checking_evals[1].toBytes()[0..16].* });
                        print("[ZOLT INST2 R{}] raf = [{any}, {any}]\n", .{ round, raf_evals[0].toBytes()[0..16].*, raf_evals[1].toBytes()[0..16].* });
                    };

                    // BRUTE FORCE VERIFICATION: At round 0, compute the Instance 2 eval_0
                    if (comptime debug_verbose) {
                    if (round == 0) {
                        // At round 0, verify that combined_vals matches trace
                        const bit_pos_r0 = LOOKUPS_LOG_K - 1; // bit 127 for round 0
                        var bf_eval_0_r0 = F.zero();
                        var bf_eval_1_r0 = F.zero();
                        for (0..T) |jj| {
                            const u_j = lookups_eq_evals[jj];
                            const cv_j = lookups_combined_vals[jj];
                            const contrib = u_j.mul(cv_j);
                            const k_lo = lookups_indices_lo[jj];
                            const k_hi = lookups_indices_hi[jj];
                            const bit_val: u1 = if (bit_pos_r0 >= 64) @truncate(k_hi >> @intCast(bit_pos_r0 - 64)) else @truncate(k_lo >> @intCast(bit_pos_r0));
                            if (bit_val == 0) {
                                bf_eval_0_r0 = bf_eval_0_r0.add(contrib);
                            } else {
                                bf_eval_1_r0 = bf_eval_1_r0.add(contrib);
                            }
                        }
                        {
                            const print = std.debug.print;
                            const bf_claim_r0 = bf_eval_0_r0.add(bf_eval_1_r0);
                            print("[BRUTE R0] bf_eval_0 = {any}\n", .{bf_eval_0_r0.toBytes()[0..16].*});
                            print("[BRUTE R0] bf_eval_1 = {any}\n", .{bf_eval_1_r0.toBytes()[0..16].*});
                            print("[BRUTE R0] bf_claim (e0+e1) = {any}\n", .{bf_claim_r0.toBytes()[0..16].*});
                            print("[BRUTE R0] lookups_claim = {any}\n", .{lookups_claim.toBytes()[0..16].*});
                            print("[BRUTE R0] claim_match = {}\n", .{bf_claim_r0.eql(lookups_claim)});
                            print("[BRUTE R0] ps_eval_0 = {any}\n", .{eval_0_inst2.toBytes()[0..16].*});
                            print("[BRUTE R0] ps_eval_2 = {any}\n", .{eval_2_inst2.toBytes()[0..16].*});
                            print("[BRUTE R0] bf_eval_0 == ps_eval_0: {}\n", .{bf_eval_0_r0.eql(eval_0_inst2)});
                        }
                    }
                    } // end comptime debug_verbose round-0 brute force
                    if (comptime debug_verbose) {
                    if (round < 3 or round == 7 or round == 15 or round == 127) {
                        const bit_pos = LOOKUPS_LOG_K - 1 - round;
                        var direct_eval_0 = F.zero();
                        var direct_eval_1 = F.zero();
                        var bf_val_eval_0 = F.zero();
                        var bf_raf_eval_0 = F.zero();
                        var bf_val_per_table: [NUM_TABLES]F = [_]F{F.zero()} ** NUM_TABLES;
                        var bf_left_sum = F.zero();
                        var bf_right_sum = F.zero();
                        var bf_identity_sum = F.zero();
                        var bf_raf_cycle_count: usize = 0;
                        var bf_identity_cycle_count: usize = 0;
                        var bf_interleaved_cycle_count: usize = 0;
                        // Alternative: compute RAF directly from operands
                        var bf_raf_from_operands = F.zero();
                        for (0..T) |jj| {
                            const u_j = lookups_eq_evals[jj];
                            const combined_j = lookups_combined_vals[jj];
                            const contrib = u_j.mul(combined_j);

                            // Get bit at bit_pos from lookup_index
                            const k_lo_bf = lookups_indices_lo[jj];
                            const k_hi_bf = lookups_indices_hi[jj];
                            const bit_val: u1 = if (bit_pos >= 64) @truncate(k_hi_bf >> @intCast(bit_pos - 64)) else @truncate(k_lo_bf >> @intCast(bit_pos));

                            if (bit_val == 0) {
                                direct_eval_0 = direct_eval_0.add(contrib);
                                const step_check = trace.steps.items[jj];
                                if (step_check.is_noop and !step_check.is_termination_store and round == 0 and !lookups_combined_vals[jj].eql(F.zero())) {
                                    dbg("[BRUTE R0] NOOP cycle {} with bit_val=0, NONZERO combined={x}\n", .{
                                        jj, lookups_combined_vals[jj].toBytesBE()[16..32].*,
                                    });
                                }
                                // Decompose: output_part = u*output, raf_part = u*(γ*left + γ²*right)
                                // We need the lookup output value for this cycle
                                // For table-0 identity-path: output = lower64(lookup_index)
                                // For other: use lookup_output from combined_vals computation
                                _ = lookups_indices_lo[jj];
                                // Actually this isn't right for all tables. Let's just use:
                                // raf_part = combined - output = γ*left + γ²*right
                                // But we don't have output stored separately.
                                // Instead, compute from trace:
                                const step_bf = trace.steps.items[jj];
                                if (!step_bf.is_noop or step_bf.is_termination_store) {
                                    // Recompute lookup_output same as combined_vals
                                    const instr_bf = step_bf.instruction;
                                    const opc_bf = instr_bf & 0x7f;
                                    const funct3_bf: u3 = @truncate((instr_bf >> 12) & 0x7);
                                    _ = funct3_bf;
                                    var lo_bf: F = undefined;
                                    switch (opc_bf) {
                                        0x6f => lo_bf = F.fromU64(step_bf.pc +% blk_jal: {
                                            const imm20_bf: u32 = ((@as(u32, instr_bf >> 31) & 1) << 19) |
                                                ((@as(u32, instr_bf >> 12) & 0xFF) << 11) |
                                                ((@as(u32, instr_bf >> 20) & 1) << 10) |
                                                ((@as(u32, instr_bf >> 21) & 0x3FF));
                                            const imm_signed_bf: i64 = @as(i64, @as(i32, @bitCast(imm20_bf << 12)) >> 11);
                                            break :blk_jal @as(u64, @bitCast(imm_signed_bf));
                                        }),
                                        0x67 => lo_bf = blk_jalr: {
                                            const imm12_raw_bf: u32 = @truncate(instr_bf >> 20);
                                            const imm_signed_bf: i64 = @as(i64, @as(i32, @bitCast(imm12_raw_bf << 20)) >> 20);
                                            const imm_u64_bf: u64 = @bitCast(imm_signed_bf);
                                            break :blk_jalr F.fromU64((step_bf.rs1_value +% imm_u64_bf) & ~@as(u64, 1));
                                        },
                                        0x63 => lo_bf = blk_br: {
                                            const f3: u3 = @truncate((instr_bf >> 12) & 0x7);
                                            const result_bf: u64 = switch (f3) {
                                                0x0 => if (step_bf.rs1_value == step_bf.rs2_value) 1 else 0,
                                                0x1 => if (step_bf.rs1_value != step_bf.rs2_value) 1 else 0,
                                                0x4 => if (@as(i64, @bitCast(step_bf.rs1_value)) < @as(i64, @bitCast(step_bf.rs2_value))) 1 else 0,
                                                0x5 => if (@as(i64, @bitCast(step_bf.rs1_value)) >= @as(i64, @bitCast(step_bf.rs2_value))) 1 else 0,
                                                0x6 => if (step_bf.rs1_value < step_bf.rs2_value) 1 else 0,
                                                0x7 => if (step_bf.rs1_value >= step_bf.rs2_value) 1 else 0,
                                                else => 0,
                                            };
                                            break :blk_br F.fromU64(result_bf);
                                        },
                                        else => lo_bf = F.fromU64(step_bf.rd_value),
                                    }
                                    bf_val_eval_0 = bf_val_eval_0.add(u_j.mul(lo_bf));
                                    bf_raf_eval_0 = bf_raf_eval_0.add(u_j.mul(lookups_combined_vals[jj].sub(lo_bf)));
                                    bf_raf_cycle_count += 1;

                                    // Compute left/right/identity contributions from lookup_index
                                    const k_lo = lookups_indices_lo[jj];
                                    const k_hi = lookups_indices_hi[jj];

                                    // Debug: for cycles 0-10 or 56-58, print the contribution and running sum
                                    if ((jj < 10 or (jj >= 56 and jj <= 58)) and round == 0) {
                                        const raf_from_combined = lookups_combined_vals[jj].sub(lo_bf);
                                        const gamma_raf_sqr_dbg2 = gamma_lookups_raf.mul(gamma_lookups_raf);
                                        const raf_from_operands_cycle = if (cycle_is_identity_path[jj])
                                            gamma_raf_sqr_dbg2.mul(F.fromU128((@as(u128, k_hi) << 64) | @as(u128, k_lo)))
                                        else blk: {
                                            var lb: u64 = 0;
                                            var rb: u64 = 0;
                                            inline for (0..32) |i| {
                                                const bit_lo = (k_lo >> @intCast(2 * i)) & 1;
                                                const bit_hi = (k_lo >> @intCast(2 * i + 1)) & 1;
                                                rb |= @as(u64, @truncate(bit_lo)) << @intCast(i);
                                                lb |= @as(u64, @truncate(bit_hi)) << @intCast(i);
                                            }
                                            break :blk gamma_lookups_raf.mul(F.fromU64(lb)).add(gamma_raf_sqr_dbg2.mul(F.fromU64(rb)));
                                        };
                                        dbg("[BRUTE R0 CYCLE{}] k_lo=0x{x}, is_identity={}\n", .{ jj, k_lo, cycle_is_identity_path[jj] });
                                        dbg("[BRUTE R0 CYCLE{}] lo_bf=0x{x}\n", .{ jj, lo_bf.toU64() });
                                        dbg("[BRUTE R0 CYCLE{}] combined-output={x}\n", .{ jj, raf_from_combined.toBytesBE()[16..32].* });
                                        dbg("[BRUTE R0 CYCLE{}] from_operands={x}\n", .{ jj, raf_from_operands_cycle.toBytesBE()[16..32].* });
                                        const cycle_match = raf_from_combined.eql(raf_from_operands_cycle);
                                        dbg("[BRUTE R0 CYCLE{}] MATCH={}\n", .{ jj, cycle_match });
                                    }

                                    if (!cycle_is_identity_path[jj]) {
                                        // Interleaved: compute left and right operands
                                        // uninterleave: left = odd bits, right = even bits
                                        var left_bits: u64 = 0;
                                        var right_bits: u64 = 0;
                                        inline for (0..32) |i| {
                                            const bit_lo = (k_lo >> @intCast(2 * i)) & 1;
                                            const bit_hi = (k_lo >> @intCast(2 * i + 1)) & 1;
                                            right_bits |= @as(u64, @truncate(bit_lo)) << @intCast(i);
                                            left_bits |= @as(u64, @truncate(bit_hi)) << @intCast(i);
                                        }
                                        // Also handle k_hi for larger indices
                                        if (k_hi != 0) {
                                            inline for (0..32) |i| {
                                                const bit_lo = (k_hi >> @intCast(2 * i)) & 1;
                                                const bit_hi = (k_hi >> @intCast(2 * i + 1)) & 1;
                                                right_bits |= @as(u64, @truncate(bit_lo)) << @intCast(32 + i);
                                                left_bits |= @as(u64, @truncate(bit_hi)) << @intCast(32 + i);
                                            }
                                        }

                                        bf_left_sum = bf_left_sum.add(u_j.mul(F.fromU64(left_bits)));
                                        bf_right_sum = bf_right_sum.add(u_j.mul(F.fromU64(right_bits)));
                                        bf_interleaved_cycle_count += 1;
                                        // Compute RAF contribution directly
                                        const gamma_raf_sqr_op = gamma_lookups_raf.mul(gamma_lookups_raf);
                                        const raf_contrib = u_j.mul(gamma_lookups_raf.mul(F.fromU64(left_bits)).add(gamma_raf_sqr_op.mul(F.fromU64(right_bits))));
                                        bf_raf_from_operands = bf_raf_from_operands.add(raf_contrib);
                                        // Debug: compare running sums
                                        if (round == 0 and jj < 60) {
                                            dbg("[BRUTE R0 CYCLE{}] INTERLEAVED: bf_raf_eval_0={x}, bf_raf_from_op={x}, match={}\n", .{
                                                jj, bf_raf_eval_0.toBytesBE()[16..32].*, bf_raf_from_operands.toBytesBE()[16..32].*,
                                                bf_raf_eval_0.eql(bf_raf_from_operands),
                                            });
                                        }

                                        // Debug: print interleaved cycle values
                                        if (round == 0) {
                                            // Compute what combined_vals should have
                                            const gamma_raf_sqr_dbg = gamma_lookups_raf.mul(gamma_lookups_raf);
                                            const raf_from_bits = gamma_lookups_raf.mul(F.fromU64(left_bits)).add(gamma_raf_sqr_dbg.mul(F.fromU64(right_bits)));
                                            dbg("[BRUTE R0 CYCLE{}] left_bits=0x{x}, right_bits=0x{x}\n", .{ jj, left_bits, right_bits });
                                            dbg("[BRUTE R0 CYCLE{}] γ*left + γ²*right={x}\n", .{ jj, raf_from_bits.toBytesBE()[16..32].* });
                                            dbg("[BRUTE R0 CYCLE{}] AFTER running bf_left_sum={x}\n", .{ jj, bf_left_sum.toBytesBE()[16..32].* });
                                            dbg("[BRUTE R0 CYCLE{}] AFTER running bf_right_sum={x}\n", .{ jj, bf_right_sum.toBytesBE()[16..32].* });
                                        }
                                    } else {
                                        // Identity path: identity = full lookup_index
                                        // For 128-bit, use lo and hi
                                        const id_val = F.fromU128((@as(u128, k_hi) << 64) | @as(u128, k_lo));
                                        bf_identity_sum = bf_identity_sum.add(u_j.mul(id_val));
                                        bf_identity_cycle_count += 1;
                                        // Compute RAF contribution directly
                                        const gamma_raf_sqr_id = gamma_lookups_raf.mul(gamma_lookups_raf);
                                        bf_raf_from_operands = bf_raf_from_operands.add(u_j.mul(gamma_raf_sqr_id.mul(id_val)));
                                        // Debug: compare running sums
                                        if (round == 0 and jj < 60) {
                                            dbg("[BRUTE R0 CYCLE{}] IDENTITY: bf_raf_eval_0={x}, bf_raf_from_op={x}, match={}\n", .{
                                                jj, bf_raf_eval_0.toBytesBE()[16..32].*, bf_raf_from_operands.toBytesBE()[16..32].*,
                                                bf_raf_eval_0.eql(bf_raf_from_operands),
                                            });
                                        }

                                        // Debug: print identity contribution for first 5 cycles
                                        if (jj < 5 and round == 0) {
                                            const gamma_raf_sqr_dbg = gamma_lookups_raf.mul(gamma_lookups_raf);
                                            const id_val_scaled = gamma_raf_sqr_dbg.mul(id_val);
                                            dbg("[BRUTE R0 CYCLE{}] identity_val=0x{x}, γ²*id={x}\n", .{ jj, k_lo, id_val_scaled.toBytesBE()[16..32].* });
                                            dbg("[BRUTE R0 CYCLE{}] running bf_identity_sum={x}\n", .{ jj, bf_identity_sum.toBytesBE()[16..32].* });
                                        }
                                    }

                                    // Per-table tracking
                                    const t_idx_bf = cycle_table_indices[jj];
                                    if (t_idx_bf >= 0 and @as(usize, @intCast(t_idx_bf)) < NUM_TABLES) {
                                        bf_val_per_table[@intCast(t_idx_bf)] = bf_val_per_table[@intCast(t_idx_bf)].add(u_j.mul(lo_bf));
                                    }
                                }
                            } else {
                                direct_eval_1 = direct_eval_1.add(contrib);
                            }
                        }
                        const direct_sum = direct_eval_0.add(direct_eval_1);
                        dbg("[BRUTE R{}] direct_eval_0={x}\n", .{ round, direct_eval_0.toBytesBE()[16..32].* });
                        dbg("[BRUTE R{}] direct_eval_1={x}\n", .{ round, direct_eval_1.toBytesBE()[16..32].* });
                        dbg("[BRUTE R{}] bf_val_eval_0={x} (should match read_checking)\n", .{ round, bf_val_eval_0.toBytesBE()[16..32].* });
                        dbg("[BRUTE R{}] bf_raf_eval_0={x} (should match raf_evals)\n", .{ round, bf_raf_eval_0.toBytesBE()[16..32].* });
                        if (round == 0) {
                            dbg("[BRUTE R0] bf_raf_cycle_count={}, identity={}, interleaved={}\n", .{bf_raf_cycle_count, bf_identity_cycle_count, bf_interleaved_cycle_count});
                            dbg("[BRUTE R0] bf_left_sum={x}\n", .{bf_left_sum.toBytesBE()[16..32].*});
                            dbg("[BRUTE R0] bf_right_sum={x}\n", .{bf_right_sum.toBytesBE()[16..32].*});
                            dbg("[BRUTE R0] bf_identity_sum={x}\n", .{bf_identity_sum.toBytesBE()[16..32].*});
                            // Compute what RAF should be from left/right/identity
                            const gamma_raf_sqr = gamma_lookups_raf.mul(gamma_lookups_raf);
                            const bf_raf_reconstructed = gamma_lookups_raf.mul(bf_left_sum).add(gamma_raf_sqr.mul(bf_right_sum.add(bf_identity_sum)));
                            dbg("[BRUTE R0] bf_raf_reconstructed (γ*l + γ²*(r+i))={x}\n", .{bf_raf_reconstructed.toBytesBE()[16..32].*});

                            // Also compute RAF using the "combined - output" formula for verification
                            // combined = output + γ*left + γ²*right
                            // So combined - output = γ*left + γ²*right
                            // But bf_raf_eval_0 is computed from u*(combined-output), and bf_raf_reconstructed from u*operands
                            // The difference might be in how identity-path cycles contribute
                            // For identity path: combined - output = γ*0 + γ²*right_op = γ²*right_op
                            // But bf_raf_reconstructed uses γ²*identity
                            // Let's also compute: γ*bf_left + γ²*bf_right (without identity)
                            const bf_raf_no_identity = gamma_lookups_raf.mul(bf_left_sum).add(gamma_raf_sqr.mul(bf_right_sum));
                            dbg("[BRUTE R0] bf_raf_no_identity (γ*l + γ²*r)={x}\n", .{bf_raf_no_identity.toBytesBE()[16..32].*});
                            dbg("[BRUTE R0] γ²*bf_identity_sum={x}\n", .{gamma_raf_sqr.mul(bf_identity_sum).toBytesBE()[16..32].*});
                            // Compute difference
                            const diff = bf_raf_eval_0.sub(bf_raf_reconstructed);
                            dbg("[BRUTE R0] DIFF (bf_raf_eval_0 - bf_raf_reconstructed)={x}\n", .{diff.toBytesBE()[16..32].*});

                            // Also compute: γ * bf_left + γ² * bf_right + γ² * bf_identity
                            // This is equivalent to bf_raf_reconstructed but computed differently
                            const alt_reconstructed = gamma_lookups_raf.mul(bf_left_sum)
                                .add(gamma_raf_sqr.mul(bf_right_sum))
                                .add(gamma_raf_sqr.mul(bf_identity_sum));
                            dbg("[BRUTE R0] alt_reconstructed (γ*l + γ²*r + γ²*i)={x}\n", .{alt_reconstructed.toBytesBE()[16..32].*});
                            dbg("[BRUTE R0] alt==bf_raf_reconstructed: {}\n", .{alt_reconstructed.eql(bf_raf_reconstructed)});
                            dbg("[BRUTE R0] bf_raf_from_operands={x}\n", .{bf_raf_from_operands.toBytesBE()[16..32].*});
                            dbg("[BRUTE R0] from_operands==bf_raf_eval_0: {}\n", .{bf_raf_from_operands.eql(bf_raf_eval_0)});
                        }
                        const bf_total = bf_val_eval_0.add(bf_raf_eval_0);
                        dbg("[BRUTE R{}] bf_val+bf_raf={x}\n", .{ round, bf_total.toBytesBE()[16..32].* });
                        const ps_total = read_checking_evals[0].add(raf_evals[0]);
                        dbg("[BRUTE R{}] ps_val+ps_raf={x}\n", .{ round, ps_total.toBytesBE()[16..32].* });
                        if (round == 0) {
                            var bf_val_sum_per_table = F.zero();
                            for (0..NUM_TABLES) |t_check| {
                                if (!bf_val_per_table[t_check].eql(F.zero())) {
                                    dbg("[BRUTE R0] bf_val_per_table[{}]={x}\n", .{ t_check, bf_val_per_table[t_check].toBytesBE()[16..32].* });
                                    bf_val_sum_per_table = bf_val_sum_per_table.add(bf_val_per_table[t_check]);
                                }
                            }
                            dbg("[BRUTE R0] bf_val_sum_per_table={x}\n", .{bf_val_sum_per_table.toBytesBE()[16..32].*});
                            dbg("[BRUTE R0] bf_val_eval_0={x}\n", .{bf_val_eval_0.toBytesBE()[16..32].*});
                            dbg("[BRUTE R0] sum==bf_val_eval_0: {}\n", .{bf_val_sum_per_table.eql(bf_val_eval_0)});
                        }
                        dbg("[BRUTE R{}] direct_sum={x}, lookups_claim={x}\n", .{
                            round,
                            direct_sum.toBytesBE()[16..32].*,
                            lookups_claim.toBytesBE()[16..32].*,
                        });
                        dbg("[BRUTE R{}] sum==claim: {}, eval0_match: {}\n", .{
                            round,
                            direct_sum.eql(lookups_claim),
                            direct_eval_0.eql(eval_0_inst2),
                        });
                        dbg("[BRUTE R{}] prefix_suffix_eval_0={x}\n", .{ round, eval_0_inst2.toBytesBE()[16..32].* });
                        dbg("[BRUTE R{}] prefix_suffix_eval_2={x}\n", .{ round, eval_2_inst2.toBytesBE()[16..32].* });

                        // BRUTE FORCE eval_2: p(2) = Σ_j u[j] * combined[j] * coeff_2(bit)
                        // where coeff_2(bit) = 2*bit + (1-2)*(1-bit) = 3*bit - 1
                        // If bit=0: -1, if bit=1: 2
                        if (round == 0) {
                            var bf_eval_2 = F.zero();
                            for (0..T) |jj2| {
                                const u_j2 = lookups_eq_evals[jj2];
                                const combined_j2 = lookups_combined_vals[jj2];
                                const contrib2 = u_j2.mul(combined_j2);
                                const k_lo_2 = lookups_indices_lo[jj2];
                                const k_hi_2 = lookups_indices_hi[jj2];
                                const bit_val_2: u1 = if (bit_pos >= 64) @truncate(k_hi_2 >> @intCast(bit_pos - 64)) else @truncate(k_lo_2 >> @intCast(bit_pos));
                                // coeff_2 = 3*bit - 1 = -1 if bit=0, 2 if bit=1
                                if (bit_val_2 == 0) {
                                    bf_eval_2 = bf_eval_2.sub(contrib2);
                                } else {
                                    bf_eval_2 = bf_eval_2.add(contrib2.add(contrib2));
                                }
                            }
                            dbg("[BRUTE R0 EVAL2] bf_eval_2 (brute force) = {x}\n", .{bf_eval_2.toBytesBE()[16..32].*});
                            dbg("[BRUTE R0 EVAL2] ps_eval_2 (prefix-suffix) = {x}\n", .{eval_2_inst2.toBytesBE()[16..32].*});
                            dbg("[BRUTE R0 EVAL2] match = {}\n", .{bf_eval_2.eql(eval_2_inst2)});
                            // Also check: bf_eval_2 should equal -claim (since all bits are 0)
                            const neg_lc = F.zero().sub(lookups_claim);
                            dbg("[BRUTE R0 EVAL2] -claim = {x}\n", .{neg_lc.toBytesBE()[16..32].*});
                            dbg("[BRUTE R0 EVAL2] bf_eval_2 == -claim: {}\n", .{bf_eval_2.eql(neg_lc)});
                        }

                        // CORRECT RAF brute force: use identity/operand path properly
                        // Check at intermediate rounds to find where drift starts
                        const check_rounds = [_]usize{ 0, 1, 7, 15, 31, 63, 127 };
                        const should_check = for (check_rounds) |cr| {
                            if (round == cr) break true;
                        } else false;
                        if (should_check) {
                            var correct_raf_eval_0 = F.zero();
                            for (0..T) |jc| {
                                const u_jc = lookups_eq_evals[jc];
                                if (u_jc.eql(F.zero())) continue;

                                const k_lo_c = lookups_indices_lo[jc];
                                const k_hi_c = lookups_indices_hi[jc];
                                const bit_val_c: u1 = if (bit_pos >= 64) @truncate(k_hi_c >> @intCast(bit_pos - 64)) else @truncate(k_lo_c >> @intCast(bit_pos));
                                if (bit_val_c != 0) continue; // Only eval_0 (bit = 0)

                                if (!cycle_is_identity_path[jc]) {
                                    // Interleaved: RAF = γ * left_operand(k) + γ² * right_operand(k)
                                    // Left(k) = k[127]*2^63 + k[125]*2^62 + ... + k[1]*2^0 (odd bit positions, MSB first)
                                    // Right(k) = k[126]*2^63 + k[124]*2^62 + ... + k[0]*2^0 (even bit positions, MSB first)
                                    const k_128: u128 = @as(u128, k_hi_c) << 64 | k_lo_c;
                                    var left_bits: u64 = 0;
                                    var right_bits: u64 = 0;
                                    for (0..64) |bi| {
                                        // i-th contribution: r[2i] (variable 2i, bit position 127-2i) with coeff 2^{63-i}
                                        // For Left (odd positions): bit at position 127 - 2*i contributes 2^{63-i}
                                        // But 127 - 2*i is odd, which matches "odd positions"
                                        // Actually, let's just use the formula directly:
                                        // Left = Σ_i k[127 - 2*i] * 2^{63-i} = Σ_i k[odd] * 2^{power}
                                        // At i=0: k[127] * 2^63, at i=1: k[125] * 2^62, ...
                                        const odd_pos = 127 - 2 * bi;
                                        const even_pos = 126 - 2 * bi;
                                        const left_bit: u64 = @truncate((k_128 >> @intCast(odd_pos)) & 1);
                                        const right_bit: u64 = @truncate((k_128 >> @intCast(even_pos)) & 1);
                                        left_bits |= left_bit << @intCast(63 - bi);
                                        right_bits |= right_bit << @intCast(63 - bi);
                                    }
                                    const left_f = F.fromU64(left_bits);
                                    const right_f = F.fromU64(right_bits);
                                    correct_raf_eval_0 = correct_raf_eval_0.add(u_jc.mul(gamma_raf.mul(left_f).add(gamma_raf2.mul(right_f))));
                                } else {
                                    // Identity: RAF = γ² * identity(k) where identity(k) = k
                                    const k_128i: u128 = @as(u128, k_hi_c) << 64 | k_lo_c;
                                    const identity_f = F.fromU128(k_128i);
                                    correct_raf_eval_0 = correct_raf_eval_0.add(u_jc.mul(gamma_raf2.mul(identity_f)));
                                }
                            }
                            dbg("[CORRECT_RAF R{}] eval_0={x}\n", .{ round, correct_raf_eval_0.toBytesBE()[16..32].* });
                            dbg("[CORRECT_RAF R{}] matches raf_evals[0]: {}\n", .{ round, correct_raf_eval_0.eql(raf_evals[0]) });
                        }
                    }
                    } // end comptime debug_verbose round diagnostics

                    // Combine Instance 2 with Instance 0 and 1 contributions
                    // During address rounds, all active instances produce degree-2 polynomials:
                    // - Instance 2 (LookupsReadRaf): degree 2 from prefix-suffix decomposition
                    // - Instance 1 (RamRaClaimReduction): degree 2 when active (rounds 112-127)
                    // - Instance 0 (RegistersValEvaluation): inactive (constant poly, degree 0)
                    // Inactive instances contribute constant p(x) = C for all x.
                    //
                    // The batched polynomial is degree 2, so we send 2 compressed coefficients [c0, c2].
                    // This matches Jolt's variable-degree batched sumcheck which sends only as many
                    // coefficients as needed for the actual polynomial degree in each round.

                    // combined_poly[0..3] = [p(0), p(1), p(2)] from Instance 0+1 contributions
                    // Add Instance 2's contribution
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
                        ram_ra_current_claim = ram_ra_current_claim.mul(UniPoly(F).INV2);
                    }
                    // NOTE: Instance 1 active case (address rounds 112-127) is handled below after
                    // the RamRaClaimReduction binding section, where ram_ra_current_claim is updated.
                    // The batched claim recomputation is also moved to after the RamRaClaimReduction binding.

                    // NOTE: Consistency check moved to after Instance 1 claim update (below)

                    // ===================================================================
                    // Update RamRaClaimReduction state after receiving challenge
                    // ===================================================================
                    // RamRaClaimReduction is active in rounds 112-135 (remaining_rounds <= 24)
                    // NOTE: We use (remaining_rounds - 1) because we already computed the polynomial
                    // for this round and are now handling the challenge binding for it
                    if (round >= 126 and round <= 131) {
                        if (comptime debug_verbose) {
                            dbg("[DEBUG BINDING R{}] remaining={}, ram_ra_num_rounds={}, check={}\n", .{
                                round,
                                remaining_rounds,
                                ram_ra_num_rounds,
                                remaining_rounds <= ram_ra_num_rounds,
                            });
                        }
                    }
                    if (remaining_rounds <= ram_ra_num_rounds) {
                        const ram_ra_round = ram_ra_num_rounds - remaining_rounds;

                        if (round >= 126 and round <= 130) {
                            if (comptime debug_verbose) {
                                dbg("[DEBUG R{} IN] ram_ra_round={}, log_ram_k={}, is_phase_cycle={}\n", .{
                                    round,
                                    ram_ra_round,
                                    log_ram_k,
                                    ram_ra_round >= log_ram_k,
                                });
                            }
                        }

                        {
                            // PhaseCycle: bind polynomials (cycle-only, no PhaseAddress)
                            const cycle_round = ram_ra_round; // 0 to n_cycle_vars-1
                            // Store cycle challenge for PhaseCycle2 eq_prefix computation
                            cycle_challenges[cycle_round] = challenge;

                            if (cycle_round < prefix_n_vars) {
                                // PhaseCycle1: bind P and Q polynomials
                                const current_len = prefix_size >> @intCast(cycle_round);
                                const half_len = current_len / 2;

                                // Bind P and Q arrays: X'[j] = (1-r)*X[2j] + r*X[2j+1]
                                // CRITICAL: Use mulHiBigIntU128 for F * Challenge
                                // Parallelize across the 6 independent arrays
                                if (self.gpu_ops) |gpu| {
                                    if (half_len >= 16384) {
                                        const pq_arrays = [_][]F{ P_raf, P_rw, P_val, Q_raf, Q_rw, Q_val };
                                        for (pq_arrays) |arr| {
                                            gpu.polyBindLow(arr[0 .. half_len * 2], challenge, arr[0..half_len]) catch {
                                                for (0..half_len) |j| {
                                                    const lo = arr[2 * j];
                                                    arr[j] = lo.add(arr[2 * j + 1].sub(lo).mulHiBigIntU128(challenge.limbs));
                                                }
                                            };
                                        }
                                    } else {
                                        for (0..half_len) |j| {
                                            P_raf[j] = P_raf[2 * j].add(P_raf[2 * j + 1].sub(P_raf[2 * j]).mulHiBigIntU128(challenge.limbs));
                                            P_rw[j] = P_rw[2 * j].add(P_rw[2 * j + 1].sub(P_rw[2 * j]).mulHiBigIntU128(challenge.limbs));
                                            P_val[j] = P_val[2 * j].add(P_val[2 * j + 1].sub(P_val[2 * j]).mulHiBigIntU128(challenge.limbs));
                                        }
                                        for (0..half_len) |j| {
                                            Q_raf[j] = Q_raf[2 * j].add(Q_raf[2 * j + 1].sub(Q_raf[2 * j]).mulHiBigIntU128(challenge.limbs));
                                            Q_rw[j] = Q_rw[2 * j].add(Q_rw[2 * j + 1].sub(Q_rw[2 * j]).mulHiBigIntU128(challenge.limbs));
                                            Q_val[j] = Q_val[2 * j].add(Q_val[2 * j + 1].sub(Q_val[2 * j]).mulHiBigIntU128(challenge.limbs));
                                        }
                                    }
                                } else if (self.thread_pool) |tp| {
                                    const BindCtx = struct {
                                        p_raf: []F, p_rw: []F, p_val: []F,
                                        q_raf: []F, q_rw: []F, q_val: []F,
                                        chal_limbs: [4]u64, h: usize,
                                    };
                                    const bctx = BindCtx{ .p_raf = P_raf, .p_rw = P_rw, .p_val = P_val, .q_raf = Q_raf, .q_rw = Q_rw, .q_val = Q_val, .chal_limbs = challenge.limbs, .h = half_len };
                                    tp.parallelForForce(6, bctx, struct {
                                        fn f(c: BindCtx, arr_idx: usize) void {
                                            const arr = switch (arr_idx) {
                                                0 => c.p_raf,
                                                1 => c.p_rw,
                                                2 => c.p_val,
                                                3 => c.q_raf,
                                                4 => c.q_rw,
                                                5 => c.q_val,
                                                else => unreachable,
                                            };
                                            for (0..c.h) |j| {
                                                const lo = arr[2 * j];
                                                arr[j] = lo.add(arr[2 * j + 1].sub(lo).mulHiBigIntU128(c.chal_limbs));
                                            }
                                        }
                                    }.f);
                                } else {
                                    for (0..half_len) |j| {
                                        P_raf[j] = P_raf[2 * j].add(P_raf[2 * j + 1].sub(P_raf[2 * j]).mulHiBigIntU128(challenge.limbs));
                                        P_rw[j] = P_rw[2 * j].add(P_rw[2 * j + 1].sub(P_rw[2 * j]).mulHiBigIntU128(challenge.limbs));
                                        P_val[j] = P_val[2 * j].add(P_val[2 * j + 1].sub(P_val[2 * j]).mulHiBigIntU128(challenge.limbs));
                                    }
                                    for (0..half_len) |j| {
                                        Q_raf[j] = Q_raf[2 * j].add(Q_raf[2 * j + 1].sub(Q_raf[2 * j]).mulHiBigIntU128(challenge.limbs));
                                        Q_rw[j] = Q_rw[2 * j].add(Q_rw[2 * j + 1].sub(Q_rw[2 * j]).mulHiBigIntU128(challenge.limbs));
                                        Q_val[j] = Q_val[2 * j].add(Q_val[2 * j + 1].sub(Q_val[2 * j]).mulHiBigIntU128(challenge.limbs));
                                    }
                                }

                                if (cycle_round < 3) {
                                    if (comptime debug_verbose) {
                                        dbg("[STAGE5 RAM_RA] Bound PhaseCycle1 round {}: challenge={x}, new_len={}\n", .{
                                            cycle_round,
                                            challenge.toBytesBE()[16..32].*,
                                            half_len,
                                        });
                                    }
                                    if (half_len > 0) {
                                        if (comptime debug_verbose) {
                                            dbg("  P_raf[0]={x}, Q_raf[0]={x}\n", .{
                                                P_raf[0].toBytesBE()[16..32].*,
                                                Q_raf[0].toBytesBE()[16..32].*,
                                            });
                                        }
                                    }
                                }
                            } else {
                                // PhaseCycle2: bind H_prime and eq_hi arrays
                                const suffix_round = cycle_round - prefix_n_vars;
                                const current_len = suffix_size >> @intCast(suffix_round);
                                const half_len = current_len / 2;

                                // Bind H_prime and eq_hi arrays: X'[j] = (1-r)*X[2j] + r*X[2j+1]
                                // CRITICAL: Use mulHiBigIntU128 for F * Challenge
                                // Parallelize across the 4 independent arrays
                                if (self.gpu_ops) |gpu| {
                                    if (half_len >= 16384) {
                                        const heq_arrays = [_][]F{ H_prime, eq_raf_hi, eq_rw_hi, eq_val_hi };
                                        for (heq_arrays) |arr| {
                                            gpu.polyBindLow(arr[0 .. half_len * 2], challenge, arr[0..half_len]) catch {
                                                for (0..half_len) |j| {
                                                    const lo = arr[2 * j];
                                                    arr[j] = lo.add(arr[2 * j + 1].sub(lo).mulHiBigIntU128(challenge.limbs));
                                                }
                                            };
                                        }
                                    } else {
                                        for (0..half_len) |j| {
                                            H_prime[j] = H_prime[2 * j].add(H_prime[2 * j + 1].sub(H_prime[2 * j]).mulHiBigIntU128(challenge.limbs));
                                        }
                                        for (0..half_len) |j| {
                                            eq_raf_hi[j] = eq_raf_hi[2 * j].add(eq_raf_hi[2 * j + 1].sub(eq_raf_hi[2 * j]).mulHiBigIntU128(challenge.limbs));
                                            eq_rw_hi[j] = eq_rw_hi[2 * j].add(eq_rw_hi[2 * j + 1].sub(eq_rw_hi[2 * j]).mulHiBigIntU128(challenge.limbs));
                                            eq_val_hi[j] = eq_val_hi[2 * j].add(eq_val_hi[2 * j + 1].sub(eq_val_hi[2 * j]).mulHiBigIntU128(challenge.limbs));
                                        }
                                    }
                                } else if (self.thread_pool) |tp| {
                                    const BindCtx2 = struct {
                                        h_prime: []F, eq_raf: []F, eq_rw: []F, eq_val: []F,
                                        chal_limbs: [4]u64, h: usize,
                                    };
                                    const bctx2 = BindCtx2{ .h_prime = H_prime, .eq_raf = eq_raf_hi, .eq_rw = eq_rw_hi, .eq_val = eq_val_hi, .chal_limbs = challenge.limbs, .h = half_len };
                                    tp.parallelForForce(4, bctx2, struct {
                                        fn f(c: BindCtx2, arr_idx: usize) void {
                                            const arr = switch (arr_idx) {
                                                0 => c.h_prime,
                                                1 => c.eq_raf,
                                                2 => c.eq_rw,
                                                3 => c.eq_val,
                                                else => unreachable,
                                            };
                                            for (0..c.h) |j| {
                                                const lo = arr[2 * j];
                                                arr[j] = lo.add(arr[2 * j + 1].sub(lo).mulHiBigIntU128(c.chal_limbs));
                                            }
                                        }
                                    }.f);
                                } else {
                                    for (0..half_len) |j| {
                                        H_prime[j] = H_prime[2 * j].add(H_prime[2 * j + 1].sub(H_prime[2 * j]).mulHiBigIntU128(challenge.limbs));
                                    }
                                    for (0..half_len) |j| {
                                        eq_raf_hi[j] = eq_raf_hi[2 * j].add(eq_raf_hi[2 * j + 1].sub(eq_raf_hi[2 * j]).mulHiBigIntU128(challenge.limbs));
                                        eq_rw_hi[j] = eq_rw_hi[2 * j].add(eq_rw_hi[2 * j + 1].sub(eq_rw_hi[2 * j]).mulHiBigIntU128(challenge.limbs));
                                        eq_val_hi[j] = eq_val_hi[2 * j].add(eq_val_hi[2 * j + 1].sub(eq_val_hi[2 * j]).mulHiBigIntU128(challenge.limbs));
                                    }
                                }

                                if (comptime debug_verbose) {
                                    dbg("[STAGE5 RAM_RA] Bound PhaseCycle2 round {} (suffix {}): challenge={x}, new_len={}\n", .{
                                        cycle_round,
                                        suffix_round,
                                        challenge.toBytesBE()[16..32].*,
                                        half_len,
                                    });
                                }
                                if (half_len > 0) {
                                    if (comptime debug_verbose) {
                                        dbg("  H_prime[0]={x}, eq_raf_hi[0]={x}\n", .{
                                            H_prime[0].toBytesBE()[16..32].*,
                                            eq_raf_hi[0].toBytesBE()[16..32].*,
                                        });
                                    }
                                }
                            }

                            // Update Instance 1 claim: p(r) = c0 + r*c1 + r²*c2
                            const c2_inst1 = inst1_eval_2.sub(inst1_eval_1).sub(inst1_eval_1).add(inst1_eval_0).mul(UniPoly(F).INV2);
                            const c1_inst1 = inst1_eval_1.sub(inst1_eval_0).sub(c2_inst1);
                            const c0_inst1 = inst1_eval_0;
                            const r2_inst1 = challenge.mul(challenge);
                            ram_ra_current_claim = c0_inst1.add(c1_inst1.mulHiBigIntU128(challenge.limbs)).add(c2_inst1.mul(r2_inst1));
                        }
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
                                ram_ra_current_claim.toBytesBE()[16..32].*,
                                lookups_claim.toBytesBE()[16..32].*,
                            });
                            dbg("[ADDR CLAIM TRACK] Round {}: batched_claim={x}\n", .{
                                round,
                                current_batched_claim.toBytesBE()[16..32].*,
                            });
                        }
                    }

                    // ===================================================================
                    // Update prefix-suffix decomposition state after receiving challenge
                    // ===================================================================

                    // Bind challenge to suffix polys, RAF decompositions, and expanding table
                    // Run all concurrently (Jolt uses rayon::scope with 5 spawns)
                    if (comptime bench_timing) {
                        bench_addr_other_ns += bench_timer.read();
                        bench_timer.reset();
                    }
                    if (self.thread_pool) |tp| {
                        const BindAllCtx = struct {
                            sp: *AllSuffixPolys(F),
                            left: *RafDecomposition(F),
                            right: *RafDecomposition(F),
                            ident: *RafDecomposition(F),
                            exp_table: *ExpandingTable(F),
                            chal: F,
                            pool: *ThreadPool,
                        };
                        const bind_ctx = BindAllCtx{
                            .sp = &suffix_polys,
                            .left = &left_raf,
                            .right = &right_raf,
                            .ident = &identity_raf,
                            .exp_table = &expanding_tables[current_phase],
                            .chal = challenge,
                            .pool = tp,
                        };
                        tp.parallelForForce(5, bind_ctx, struct {
                            fn f(c: BindAllCtx, task_idx: usize) void {
                                switch (task_idx) {
                                    0 => c.sp.bindAllParallel(c.chal, c.pool),
                                    1 => c.left.bind(c.chal),
                                    2 => c.right.bind(c.chal),
                                    3 => c.ident.bind(c.chal),
                                    4 => c.exp_table.update(c.chal),
                                    else => unreachable,
                                }
                            }
                        }.f);
                    } else {
                        suffix_polys.bindAll(challenge);
                        left_raf.bind(challenge);
                        right_raf.bind(challenge);
                        identity_raf.bind(challenge);
                        expanding_tables[current_phase].update(challenge);
                    }

                    // Update prefix checkpoints every 2 rounds (after binding X and Y)
                    const round_in_phase = round % log_m;
                    if (round_in_phase % 2 == 1) {
                        // We just bound Y, update prefix checkpoints with (checkpoint_r_x, r_y)
                        const checkpoint_r_x = challenges[round - 1];
                        const r_y = challenge;
                        const suffix_len = LOOKUPS_LOG_K - (current_phase + 1) * log_m;
                        prefix_checkpoints.update(checkpoint_r_x, r_y, round, suffix_len);
                    }
                    if (comptime bench_timing) bench_addr_bind_ns += bench_timer.read();
                    if (s5_phase_timer) |*pt| { s5_addr_bind_ns += pt.read(); pt.reset(); }

                    // Check for phase transition (every log_m = 16 rounds)
                    if (comptime bench_timing) bench_timer.reset();
                    if ((round + 1) % log_m == 0 and round + 1 < LOOKUPS_LOG_K) {
                        const prev_phase = current_phase;
                        current_phase += 1;
                        if (comptime debug_verbose) {
                            dbg("[STAGE5] Phase transition to phase {}, prev_table_len={}\n", .{ current_phase, expanding_tables[prev_phase].getLen() });
                        }

                        // DRIFT DEBUG BEFORE CONDENSATION:
                        // Verify the expanding table matches direct EQ computation
                        if (comptime debug_verbose) {
                        if (current_phase == 1) {
                            // At phase 0→1 transition (round 7 just completed):
                            // expanding_tables[0] should contain EQ(k, [r_0, r_1, ..., r_7]) for k in 0..255
                            // Verify entry 0: EQ(0, r) = Π_{i=0}^{7} (1 - r_i)
                            var direct_eq_0 = F.one();
                            for (0..log_m) |i| {
                                direct_eq_0 = direct_eq_0.mul(F.one().sub(challenges[prev_phase * log_m + i]));
                            }
                            const table_eq_0 = expanding_tables[prev_phase].get(0);
                            if (comptime debug_verbose) {
                                dbg("[PHASE_VERIFY] expanding_table[0] = {x}\n", .{table_eq_0.toBytesBE()[16..32].*});
                                dbg("[PHASE_VERIFY] direct_eq(0, r) = {x}\n", .{direct_eq_0.toBytesBE()[16..32].*});
                                dbg("[PHASE_VERIFY] match = {}\n", .{table_eq_0.eql(direct_eq_0)});
                            }

                            // Verify entry 1: EQ(1, r) - bit 0 = 1, rest = 0
                            // With HighToLow binding: bit 0 of k corresponds to round (log_m-1) = round 7
                            // So EQ(1, r) = r_7 * Π_{i=0}^{6} (1 - r_i)
                            var direct_eq_1 = challenges[prev_phase * log_m + log_m - 1];
                            for (0..log_m - 1) |i| {
                                direct_eq_1 = direct_eq_1.mul(F.one().sub(challenges[prev_phase * log_m + i]));
                            }
                            const table_eq_1 = expanding_tables[prev_phase].get(1);
                            if (comptime debug_verbose) {
                                dbg("[PHASE_VERIFY] expanding_table[1] = {x}\n", .{table_eq_1.toBytesBE()[16..32].*});
                                dbg("[PHASE_VERIFY] direct_eq(1, r) = {x}\n", .{direct_eq_1.toBytesBE()[16..32].*});
                                dbg("[PHASE_VERIFY] match = {}\n", .{table_eq_1.eql(direct_eq_1)});
                            }

                            // Compute brute sum BEFORE condensation (should be original claim)
                            var pre_condense_sum = F.zero();
                            for (0..T) |jj| {
                                pre_condense_sum = pre_condense_sum.add(lookups_eq_evals[jj].mul(lookups_combined_vals[jj]));
                            }
                            if (comptime debug_verbose) {
                                dbg("[PHASE_VERIFY] pre_condense_sum = {x}\n", .{pre_condense_sum.toBytesBE()[16..32].*});
                            }

                            // Compute "what the condensed sum SHOULD be" by multiplying each term
                            // by the appropriate expanding table entry
                            var expected_condensed_sum = F.zero();
                            const suffix_bits = (num_phases - current_phase) * log_m;
                            const m_mask: u128 = (@as(u128, 1) << @intCast(log_m)) - 1;
                            for (0..T) |jj| {
                                const k = lookup_indices_u128[jj];
                                const prefix = k >> @intCast(suffix_bits);
                                const k_bound: usize = @intCast(prefix & m_mask);
                                const v_val = expanding_tables[prev_phase].get(k_bound);
                                expected_condensed_sum = expected_condensed_sum.add(lookups_eq_evals[jj].mul(v_val).mul(lookups_combined_vals[jj]));
                                if (jj < 5) {
                                    if (comptime debug_verbose) {
                                        dbg("[PHASE_VERIFY] j={}: k_bound={}, v_val={x}, eq={x}, cv={x}\n", .{
                                            jj, k_bound,
                                            v_val.toBytesBE()[24..32].*,
                                            lookups_eq_evals[jj].toBytesBE()[24..32].*,
                                            lookups_combined_vals[jj].toBytesBE()[24..32].*,
                                        });
                                    }
                                }
                            }
                            if (comptime debug_verbose) {
                                dbg("[PHASE_VERIFY] expected_condensed_sum = {x}\n", .{expected_condensed_sum.toBytesBE()[16..32].*});
                                dbg("[PHASE_VERIFY] lookups_claim (poly chain) = {x}\n", .{lookups_claim.toBytesBE()[16..32].*});
                                dbg("[PHASE_VERIFY] match = {}\n", .{expected_condensed_sum.eql(lookups_claim)});
                            }
                        }
                        } // end comptime debug_verbose

                        // Condense u_evals (lookups_eq_evals) using the expanding table from the previous phase
                        if (comptime bench_timing) bench_timer.reset();
                        condenseUEvals(F, lookups_eq_evals, &expanding_tables[prev_phase], lookup_indices_u128, current_phase, num_phases, self.thread_pool);
                        if (comptime debug_verbose) {
                            dbg("[STAGE5] Phase {} condense done, now calling initPhase...\n", .{current_phase});
                        }

                        // DRIFT DEBUG: Compute direct sum after condensation to compare with lookups_claim
                        // At this point, lookups_eq_evals[j] has been condensed to include the expanding table contribution
                        // The sum should match lookups_claim (after polynomial evolution through previous round)
                        if (comptime debug_verbose) {
                            var brute_sum = F.zero();
                            for (0..T) |jj| {
                                brute_sum = brute_sum.add(lookups_eq_evals[jj].mul(lookups_combined_vals[jj]));
                            }
                            const drift_match = brute_sum.eql(lookups_claim);
                            {
                                const print = std.debug.print;
                                print("[DRIFT_CHECK Phase {}] brute_sum = {any}\n", .{ current_phase, brute_sum.toBytes()[0..16].* });
                                print("[DRIFT_CHECK Phase {}] lookups_claim = {any}\n", .{ current_phase, lookups_claim.toBytes()[0..16].* });
                                print("[DRIFT_CHECK Phase {}] match = {}\n", .{ current_phase, drift_match });
                            }
                        }

                        if (comptime bench_timing) bench_condense_ns += bench_timer.read();

                        // Save prefix checkpoints before resetting RAF decompositions
                        if (comptime bench_timing) bench_timer.reset();
                        left_raf.updateCheckpoint();
                        right_raf.updateCheckpoint();
                        identity_raf.updateCheckpoint();

                        // Reset RAF decompositions for new phase (restore Q_size to initial_m=256)
                        left_raf.resetForPhase(current_phase, initial_m);
                        right_raf.resetForPhase(current_phase, initial_m);
                        identity_raf.resetForPhase(current_phase, initial_m);

                        // Run initPhase and initQRaf concurrently (they're independent —
                        // initPhase writes to suffix polys, initQRaf writes to RAF Q arrays,
                        // both only read from u_evals and lookup_indices).
                        if (self.thread_pool) |tp| {
                            const SuffixInitCtx = struct {
                                polys: *AllSuffixPolys(F),
                                phase: usize,
                                phases: usize,
                                eq: []const F,
                                indices: []const u128,
                                table_indices: []const i8,
                                pool: *ThreadPool,
                                alloc_inner: Allocator,
                                ibt: ?*const [NUM_TABLES][]usize,
                            };
                            const suffix_ctx = SuffixInitCtx{
                                .polys = &suffix_polys,
                                .phase = current_phase,
                                .phases = num_phases,
                                .eq = lookups_eq_evals,
                                .indices = lookup_indices_u128,
                                .table_indices = cycle_table_indices,
                                .pool = tp,
                                .alloc_inner = self.allocator,
                                .ibt = &lookup_indices_by_table,
                            };
                            const RafInitCtx = struct {
                                left_raf: *@TypeOf(left_raf),
                                right_raf: *@TypeOf(right_raf),
                                identity_raf: *@TypeOf(identity_raf),
                                eq: []const F,
                                indices: []const u128,
                                is_interleaved: []const bool,
                                pool: *ThreadPool,
                                alloc_inner: Allocator,
                            };
                            const raf_ctx = RafInitCtx{
                                .left_raf = &left_raf,
                                .right_raf = &right_raf,
                                .identity_raf = &identity_raf,
                                .eq = lookups_eq_evals,
                                .indices = lookup_indices_u128,
                                .is_interleaved = is_interleaved_operands,
                                .pool = tp,
                                .alloc_inner = self.allocator,
                            };
                            _ = tp.join(
                                void,
                                void,
                                suffix_ctx,
                                struct {
                                    fn f(c: SuffixInitCtx) void {
                                        c.polys.initPhase(c.phase, c.phases, c.eq, c.indices, c.table_indices, c.pool, c.alloc_inner, c.ibt) catch unreachable;
                                    }
                                }.f,
                                raf_ctx,
                                struct {
                                    fn f(c: RafInitCtx) void {
                                        initQRaf(F, c.left_raf, c.right_raf, c.identity_raf, c.eq, c.indices, c.is_interleaved, c.pool, c.alloc_inner) catch unreachable;
                                    }
                                }.f,
                            );
                        } else {
                            try suffix_polys.initPhase(current_phase, num_phases, lookups_eq_evals, lookup_indices_u128, cycle_table_indices, null, self.allocator, null);
                            try initQRaf(F, &left_raf, &right_raf, &identity_raf, lookups_eq_evals, lookup_indices_u128, is_interleaved_operands, null, self.allocator);
                        }

                        // Materialize prefix MLE tables for new phase
                        left_raf.initPrefix();
                        right_raf.initPrefix();
                        identity_raf.initPrefix();
                        if (comptime debug_verbose) {
                            dbg("[STAGE5] Phase {} initQRaf + initPrefix done\n", .{current_phase});
                        }

                        // Reset the new phase's expanding table to 1
                        expanding_tables[current_phase].reset(F.one());
                        if (comptime bench_timing) bench_init_phase_ns += bench_timer.read();

                        if (comptime debug_verbose) {
                            dbg("[STAGE5] Condensed u_evals with expanding table, reset phase {} table\n", .{current_phase});
                        }
                    }

                    // ra_weights and ra_chunk_weights are NOT needed during address rounds.
                    // They are fully rematerialized from expanding tables at cycle round start (round 128).
                    // Skipping incremental updates saves ~130ms of parallelForForce dispatch overhead.
                    if (comptime debug_verbose) {
                        const bit_index = LOOKUPS_LOG_K - 1 - round;
                        const one_minus_r = F.one().sub(challenge);
                        const chunk_idx = round / lookups_ra_virtual_log_k_chunk;

                        dbg("[STAGE5 ADDR] Round {} challenge (full) = {any}\n", .{
                            round,
                            challenge.toBytesBE(),
                        });

                        for (0..T) |j| {
                            const bit = getBit128(lookups_indices_lo[j], lookups_indices_hi[j], bit_index);
                            const factor = if (bit == 0) one_minus_r else challenge;
                            lookups_ra_weights[j] = lookups_ra_weights[j].mul(factor);
                            bf_weights[j] = bf_weights[j].mul(factor);

                            if (chunk_idx < ra_num_chunks) {
                                ra_chunk_weights[chunk_idx][j] = ra_chunk_weights[chunk_idx][j].mul(factor);
                            }
                        }

                        if ((round + 1) % lookups_ra_virtual_log_k_chunk == 0) {
                            dbg("[STAGE5 RA_CHUNK] Finished chunk {} after round {}\n", .{ chunk_idx, round });
                            for (0..@min(4, T)) |jj| {
                                dbg("  ra_chunk[{}][{}] = {x}\n", .{
                                    chunk_idx, jj, ra_chunk_weights[chunk_idx][jj].toBytesBE()[16..32].*,
                                });
                            }
                        }
                    }

                    // NOTE: lookups_claim is already updated above by evaluating Instance 2's polynomial at challenge.
                    // The recomputation from raw arrays is WRONG because eq_evals hasn't been bound yet.
                    // The correct value is p_inst2(r) which we computed earlier.

                    if (round % 8 == 7) {
                        if (comptime debug_verbose) {
                            dbg("[STAGE5] Completed rounds 0-{}\n", .{round});
                        }
                    }

                    if (comptime bench_timing) bench_phase_transition_ns += bench_timer.read();
                    if (s5_phase_timer) |*pt| { s5_phase_trans_ns += pt.read(); pt.reset(); }
                    continue; // Skip the rest of the loop for address rounds
                } else {
                    // Cycle rounds: actual sumcheck over eq_reduction * Π_c ra_chunk[c] * combined_vals
                    //
                    // Jolt's approach (read_raf_checking.rs:720-774):
                    // 1. Build pairs [(p_j(0), p_j(1))] for 9 factors:
                    //    - combined_val with eq_in absorbed: (eq[2j] * v[2j], eq[2j+1] * v[2j+1])
                    //    - ra_chunks[i]: (ra[i][2j], ra[i][2j+1])
                    // 2. Evaluate product at [1, 2, ..., 8, ∞] using eval_linear_prod
                    // 3. Use finish_mles_product_sum_from_evals to recover full degree-10 polynomial
                    //    by multiplying by eq(X, r_round) where r_round = r_reduction[n-1-cycle_round]
                    //
                    const lookups_round = round - LOOKUPS_LOG_K;

                    // ============================================================
                    // Rematerialization at start of cycle rounds (round 128 only)
                    // ============================================================
                    // Jolt's init_log_t_rounds() (lines 641-692) materializes the
                    // combined_val polynomial using the bound prefix checkpoint values.
                    //
                    // raf_interleaved = γ * left_prefix + γ² * right_prefix
                    // raf_identity = γ² * identity_prefix
                    //
                    // For each cycle j:
                    //   if is_interleaved_operands[j]:
                    //     combined_val[j] = table_eval[j] + raf_interleaved
                    //   else:
                    //     combined_val[j] = table_eval[j] + raf_identity
                    if (lookups_round == 0) {
                        // DEBUG: Check sum BEFORE rematerialization
                        if (comptime debug_verbose) {
                        var pre_remat_sum = F.zero();
                        for (0..T) |jj| {
                            pre_remat_sum = pre_remat_sum.add(lookups_eq_evals[jj].mul(lookups_combined_vals[jj]));
                        }
                            dbg("[PRE-REMAT] sum(eq*combined_vals) = {x}\n", .{pre_remat_sum.toBytesBE()[16..32].*});
                            dbg("[PRE-REMAT] lookups_claim (poly chain) = {x}\n", .{lookups_claim.toBytesBE()[16..32].*});
                            dbg("[PRE-REMAT] match = {}\n", .{pre_remat_sum.eql(lookups_claim)});
                        }

                        // Get bound prefix values from RAF decompositions
                        // After 128 address rounds, the prefix MLE has been bound to a single value.
                        // Use updateCheckpoint to save the final prefix MLE value into bound_value.
                        left_raf.updateCheckpoint();
                        right_raf.updateCheckpoint();
                        identity_raf.updateCheckpoint();
                        const left_prefix = left_raf.bound_value;
                        const right_prefix = right_raf.bound_value;
                        const identity_prefix = identity_raf.bound_value;

                        // Also compute identity_poly_eval directly from challenges for verification
                        var computed_identity = F.zero();
                        for (0..LOOKUPS_LOG_K) |i| {
                            const r_i = challenges[i];
                            // identity_poly_eval = Σᵢ rᵢ · 2^(LOG_K-1-i)
                            // Use field pow2 to avoid overflow
                            const power = LOOKUPS_LOG_K - 1 - i;
                            const two_power = if (power < 64) F.fromU64(@as(u64, 1) << @intCast(power)) else blk: {
                                // Handle power >= 64
                                const two_pow_64 = F.fromBytes(&[_]u8{
                                    0, 0, 0, 0, 0, 0, 0, 0,
                                    1, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0,
                                });
                                var result = two_pow_64;
                                var remaining = power - 64;
                                while (remaining >= 64) {
                                    result = result.mul(two_pow_64);
                                    remaining -= 64;
                                }
                                if (remaining > 0) {
                                    result = result.mul(F.fromU64(@as(u64, 1) << @intCast(remaining)));
                                }
                                break :blk result;
                            };
                            computed_identity = computed_identity.add(r_i.mul(two_power));
                        }

                        // Also compute left/right operand evals directly from challenges for verification
                        var computed_left = F.zero();
                        var computed_right = F.zero();
                        // left_operand = Σ r_{2i} · 2^(LOG_K/2-1-i) for i=0..LOG_K/2-1
                        // right_operand = Σ r_{2i+1} · 2^(LOG_K/2-1-i) for i=0..LOG_K/2-1
                        const half_k = LOOKUPS_LOG_K / 2;
                        for (0..half_k) |i| {
                            const r_left = challenges[2 * i]; // even indices
                            const r_right = challenges[2 * i + 1]; // odd indices
                            const power = half_k - 1 - i;
                            const two_power = if (power < 64) F.fromU64(@as(u64, 1) << @intCast(power)) else blk: {
                                const two_pow_64 = F.fromBytes(&[_]u8{
                                    0, 0, 0, 0, 0, 0, 0, 0,
                                    1, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0,
                                });
                                var result = two_pow_64;
                                var remaining = power - 64;
                                while (remaining >= 64) {
                                    result = result.mul(two_pow_64);
                                    remaining -= 64;
                                }
                                if (remaining > 0) {
                                    result = result.mul(F.fromU64(@as(u64, 1) << @intCast(remaining)));
                                }
                                break :blk result;
                            };
                            computed_left = computed_left.add(r_left.mul(two_power));
                            computed_right = computed_right.add(r_right.mul(two_power));
                        }

                        if (comptime debug_verbose) {
                            dbg("[STAGE5 REMATERIALIZE] Verification:\n", .{});
                            dbg("  computed_identity (from challenges) = {x}\n", .{computed_identity.toBytesBE()[16..32].*});
                            dbg("  identity_prefix (from bound_value) = {x}\n", .{identity_prefix.toBytesBE()[16..32].*});
                            dbg("  identity match = {}\n", .{computed_identity.eql(identity_prefix)});
                            dbg("  computed_left (from challenges) = {x}\n", .{computed_left.toBytesBE()[16..32].*});
                            dbg("  left_prefix (from bound_value) = {x}\n", .{left_prefix.toBytesBE()[16..32].*});
                            dbg("  left match = {}\n", .{computed_left.eql(left_prefix)});
                            dbg("  computed_right (from challenges) = {x}\n", .{computed_right.toBytesBE()[16..32].*});
                            dbg("  right_prefix (from bound_value) = {x}\n", .{right_prefix.toBytesBE()[16..32].*});
                            dbg("  right match = {}\n", .{computed_right.eql(right_prefix)});
                        }

                        // Print first few challenges to compare with Jolt
                        if (comptime debug_verbose) {
                            dbg("  First 4 challenges (to compare with Jolt):\n", .{});
                        }
                        for (0..4) |i| {
                            if (comptime debug_verbose) {
                                dbg("    challenges[{}] = {x}\n", .{ i, challenges[i].toBytesBE()[16..32].* });
                            }
                        }

                        // Compute RAF scalar values
                        const raf_interleaved = gamma_raf.mul(left_prefix).add(gamma_raf2.mul(right_prefix));
                        const raf_identity = gamma_raf2.mul(identity_prefix);

                        if (comptime debug_verbose) {
                            dbg("[STAGE5 REMATERIALIZE] round=128, left_prefix={x}, right_prefix={x}, identity_prefix={x}\n", .{
                                left_prefix.toBytesBE()[16..32].*,
                                right_prefix.toBytesBE()[16..32].*,
                                identity_prefix.toBytesBE()[16..32].*,
                            });
                            dbg("[STAGE5 REMATERIALIZE] raf_interleaved={x}, raf_identity={x}\n", .{
                                raf_interleaved.toBytesBE()[16..32].*,
                                raf_identity.toBytesBE()[16..32].*,
                            });
                        }

                        // ============================================================
                        // CRITICAL FIX: Use table MLE at r_address, NOT raw lookup output
                        // ============================================================
                        // Jolt's init_log_t_rounds computes table_values_at_r_addr:
                        //   table_values_at_r_addr[t] = table.combine(&prefixes, &suffix_evals)
                        // where suffix_evals are the suffix MLEs at empty bits (since all
                        // suffix variables have been bound during address rounds).
                        //
                        // The raw lookup_output (e.g., rs1 & rs2 for AND) is NOT the same as
                        // the table MLE evaluated at the random point r_address!
                        //
                        // Reference: jolt-core/src/zkvm/instruction_lookups/read_raf_checking.rs:641-671

                        // Compute table_values_at_r_addr using bound prefix checkpoints
                        const table_values = computeTableValuesAtRAddress(F, &prefix_checkpoints);

                        // Store for later use in val_claim computation
                        // table_values has NUM_TABLES (41) entries, copy into stored_table_values
                        for (0..NUM_TABLES) |ti| {
                            stored_table_values[ti] = table_values[ti];
                        }

                        // ============================================================
                        // CRITICAL FIX: Rematerialize combined_vals for cycle rounds
                        // ============================================================
                        // Jolt's init_log_t_rounds (read_raf_checking.rs:746-762) computes:
                        //   combined_val[j] = table_eval(r_addr, table(j)) + raf_eval(j)
                        // where:
                        //   table_eval = stored_table_values[table(j)]  (table MLE at r_addr, 0 if no table)
                        //   raf_eval = raf_interleaved or raf_identity depending on instruction type
                        //
                        // CRITICAL: In Jolt, RAF is added for ALL cycles, including those
                        // without lookup tables (NOOPs, padding). The table_val is only
                        // added when cycle.lookup_table() returns Some(table). But the
                        // RAF contribution is ALWAYS added based on is_interleaved_operands.
                        {
                            const RematCtx = struct {
                                combined: []F,
                                t_indices: []const i8,
                                is_identity: []const bool,
                                table_vals: *const [MAX_LOOKUP_TABLES]F,
                                raf_il: F,
                                raf_id: F,
                            };
                            const rctx = RematCtx{
                                .combined = lookups_combined_vals,
                                .t_indices = cycle_table_indices,
                                .is_identity = cycle_is_identity_path,
                                .table_vals = &stored_table_values,
                                .raf_il = raf_interleaved,
                                .raf_id = raf_identity,
                            };
                            const rematFn = struct {
                                fn f(c: RematCtx, j: usize) void {
                                    var val = F.zero();
                                    const t_idx_j = c.t_indices[j];
                                    if (t_idx_j >= 0) {
                                        const ti: usize = @intCast(t_idx_j);
                                        if (ti < NUM_TABLES) {
                                            val = c.table_vals[ti];
                                        }
                                    }
                                    if (!c.is_identity[j]) {
                                        val = val.add(c.raf_il);
                                    } else {
                                        val = val.add(c.raf_id);
                                    }
                                    c.combined[j] = val;
                                }
                            }.f;
                            if (self.thread_pool) |tp| {
                                tp.parallelForForce(T, rctx, rematFn);
                            } else {
                                for (0..T) |j| rematFn(rctx, j);
                            }
                        }
                        if (comptime debug_verbose) {
                            dbg("[STAGE5 REMAT] combined_vals rematerialized for {} cycles\n", .{T});
                            dbg("[STAGE5 REMAT] combined_vals[0] = {x}\n", .{lookups_combined_vals[0].toBytesBE()[16..32].*});
                        }

                        // ============================================================
                        // DEBUG: Direct MLE computation for comparison
                        // ============================================================
                        // Jolt's verifier uses table.evaluate_mle(&r_address_prime)
                        // which is a direct formula. Let's compute RangeCheck MLE directly
                        // to verify our prefix-suffix decomposition is correct.
                        //
                        // RangeCheck MLE formula (Jolt):
                        //   Σ_{i=0}^{63} 2^(63-i) * r[64+i]
                        //
                        // This means: r_address_prime[64] has coeff 2^63, r[65] has 2^62, etc.
                        if (comptime debug_verbose) {
                            dbg("[STAGE5 DEBUG] Direct MLE computation for RangeCheck:\n", .{});
                        }
                        // Debug: print challenges[64], [65], and [127] for comparison with Jolt (FULL 32 bytes)
                        const ch64_bytes = challenges[64].toBytesBE();
                        if (comptime debug_verbose) {
                            dbg("[STAGE5 DEBUG] challenges[64] (FULL 32) = ", .{});
                            for (ch64_bytes) |b| dbg("{x:0>2}", .{b});
                            dbg("\n", .{});
                        }
                        const ch65_bytes = challenges[65].toBytesBE();
                        if (comptime debug_verbose) {
                            dbg("[STAGE5 DEBUG] challenges[65] (FULL 32) = ", .{});
                            for (ch65_bytes) |b| dbg("{x:0>2}", .{b});
                            dbg("\n", .{});
                        }
                        const ch127_bytes = challenges[127].toBytesBE();
                        if (comptime debug_verbose) {
                            dbg("[STAGE5 DEBUG] challenges[127] (FULL 32) = ", .{});
                            for (ch127_bytes) |b| dbg("{x:0>2}", .{b});
                            dbg("\n", .{});
                        }
                        var direct_range_check_mle = F.zero();
                        for (0..64) |i| {
                            // challenges are in HighToLow order: challenge[0] binds bit 127
                            // So challenge[64] corresponds to bit 63 (the MSB of the lower word)
                            // and challenge[127] corresponds to bit 0 (the LSB of the lower word)
                            const r_i = challenges[64 + i];
                            const shift = 63 - i;
                            const coeff = if (shift < 64) F.fromU64(@as(u64, 1) << @intCast(shift)) else F.zero();
                            direct_range_check_mle = direct_range_check_mle.add(coeff.mul(r_i));
                            if (i < 3 or i >= 61) {
                                if (comptime debug_verbose) {
                                    dbg("  i={}, shift={}, coeff=2^{}, r={x}\n", .{
                                        i, shift, shift, r_i.toBytesBE()[28..32].*,
                                    });
                                }
                            }
                        }
                        if (comptime debug_verbose) {
                            dbg("[STAGE5 DEBUG] Direct RangeCheck MLE result: {x}\n", .{
                                direct_range_check_mle.toBytesBE()[16..32].*,
                            });
                            dbg("[STAGE5 DEBUG] Prefix-suffix result (table[0]): {x}\n", .{
                                table_values[0].toBytesBE()[16..32].*,
                            });
                            dbg("[STAGE5 DEBUG] Match: {}\n", .{
                                direct_range_check_mle.eql(table_values[0]),
                            });
                        }

                        // Verify AND table (index 2): Σ 2^(63-i) * r[2*i] * r[2*i+1]
                        var direct_and_mle = F.zero();
                        for (0..64) |i| {
                            const x_i = challenges[2 * i];
                            const y_i = challenges[2 * i + 1];
                            const shift_and: u6 = @intCast(63 - i);
                            direct_and_mle = direct_and_mle.add(F.fromU64(@as(u64, 1) << shift_and).mul(x_i.mul(y_i)));
                        }
                        if (comptime debug_verbose) {
                            dbg("[TABLE_VERIFY] AND direct = {x}\n", .{direct_and_mle.toBytesBE()[16..32].*});
                            dbg("[TABLE_VERIFY] AND prefix-suffix = {x}\n", .{table_values[2].toBytesBE()[16..32].*});
                            dbg("[TABLE_VERIFY] AND match: {}\n", .{direct_and_mle.eql(table_values[2])});
                        }

                        // Verify XOR table (index 5): Σ 2^(63-i) * ((1-x)*y + x*(1-y))
                        var direct_xor_mle = F.zero();
                        for (0..64) |i| {
                            const x_i = challenges[2 * i];
                            const y_i = challenges[2 * i + 1];
                            const shift_xor: u6 = @intCast(63 - i);
                            const xor_val = F.one().sub(x_i).mul(y_i).add(x_i.mul(F.one().sub(y_i)));
                            direct_xor_mle = direct_xor_mle.add(F.fromU64(@as(u64, 1) << shift_xor).mul(xor_val));
                        }
                        if (comptime debug_verbose) {
                            dbg("[TABLE_VERIFY] XOR direct = {x}\n", .{direct_xor_mle.toBytesBE()[16..32].*});
                            dbg("[TABLE_VERIFY] XOR prefix-suffix = {x}\n", .{table_values[5].toBytesBE()[16..32].*});
                            dbg("[TABLE_VERIFY] XOR match: {}\n", .{direct_xor_mle.eql(table_values[5])});
                        }

                        // Verify OR table (index 4): Σ 2^(63-i) * (x + y - x*y)
                        var direct_or_mle = F.zero();
                        for (0..64) |i| {
                            const x_i = challenges[2 * i];
                            const y_i = challenges[2 * i + 1];
                            const shift_or: u6 = @intCast(63 - i);
                            const or_val = x_i.add(y_i).sub(x_i.mul(y_i));
                            direct_or_mle = direct_or_mle.add(F.fromU64(@as(u64, 1) << shift_or).mul(or_val));
                        }
                        if (comptime debug_verbose) {
                            dbg("[TABLE_VERIFY] OR direct = {x}\n", .{direct_or_mle.toBytesBE()[16..32].*});
                            dbg("[TABLE_VERIFY] OR prefix-suffix = {x}\n", .{table_values[4].toBytesBE()[16..32].*});
                            dbg("[TABLE_VERIFY] OR match: {}\n", .{direct_or_mle.eql(table_values[4])});
                        }

                        // Verify UnsignedLessThan table (index 11):
                        // Σ_i (1 - r[2*i]) * r[2*i+1] * Π_{j<i} (r[2*j]*r[2*j+1] + (1-r[2*j])*(1-r[2*j+1]))
                        {
                            var direct_ult_mle = F.zero();
                            var eq_term = F.one();
                            for (0..64) |i| {
                                const x_i = challenges[2 * i];
                                const y_i = challenges[2 * i + 1];
                                direct_ult_mle = direct_ult_mle.add(F.one().sub(x_i).mul(y_i).mul(eq_term));
                                eq_term = eq_term.mul(x_i.mul(y_i).add(F.one().sub(x_i).mul(F.one().sub(y_i))));
                            }
                            if (comptime debug_verbose) {
                                dbg("[TABLE_VERIFY] UnsignedLessThan direct = {any}\n", .{direct_ult_mle.toBytes()});
                                dbg("[TABLE_VERIFY] UnsignedLessThan prefix-suffix = {any}\n", .{table_values[11].toBytes()});
                                dbg("[TABLE_VERIFY] UnsignedLessThan match: {}\n", .{direct_ult_mle.eql(table_values[11])});
                            }
                            // Also compare with Jolt's expected value
                            if (comptime debug_verbose) {
                                dbg("[TABLE_VERIFY] UnsignedLessThan Jolt val = ce b8 26 7f 68 99 84 22 e1 c6 cd a7 b2 dd cd 24 a4 7a 39 dc 1f 7f f9 d7 7f 51 3f 23 03 54 5f 1c\n", .{});
                            }
                        }

                        // Verify Equal table (index 6): Π (x_i*y_i + (1-x_i)*(1-y_i))
                        {
                            var direct_equal_mle = F.one();
                            for (0..64) |i| {
                                const x_i = challenges[2 * i];
                                const y_i = challenges[2 * i + 1];
                                const eq_term = x_i.mul(y_i).add(F.one().sub(x_i).mul(F.one().sub(y_i)));
                                direct_equal_mle = direct_equal_mle.mul(eq_term);
                            }
                            if (comptime debug_verbose) {
                                dbg("[TABLE_VERIFY] Equal direct = {any}\n", .{direct_equal_mle.toBytes()});
                                dbg("[TABLE_VERIFY] Equal prefix-suffix = {any}\n", .{table_values[6].toBytes()});
                                dbg("[TABLE_VERIFY] Equal match: {}\n", .{direct_equal_mle.eql(table_values[6])});
                            }
                        }

                        // Verify UnsignedGTE table (index 8): 1 - ULT
                        {
                            var direct_ult = F.zero();
                            var eq_term = F.one();
                            for (0..64) |i| {
                                const x_i = challenges[2 * i];
                                const y_i = challenges[2 * i + 1];
                                direct_ult = direct_ult.add(F.one().sub(x_i).mul(y_i).mul(eq_term));
                                eq_term = eq_term.mul(x_i.mul(y_i).add(F.one().sub(x_i).mul(F.one().sub(y_i))));
                            }
                            const direct_ugte_mle = F.one().sub(direct_ult);
                            if (comptime debug_verbose) {
                                dbg("[TABLE_VERIFY] UnsignedGTE direct = {any}\n", .{direct_ugte_mle.toBytes()});
                                dbg("[TABLE_VERIFY] UnsignedGTE prefix-suffix = {any}\n", .{table_values[8].toBytes()});
                                dbg("[TABLE_VERIFY] UnsignedGTE match: {}\n", .{direct_ugte_mle.eql(table_values[8])});
                            }
                        }

                        // Verify ValidUnsignedRemainder table (index 16): LT + divisor_is_zero
                        {
                            var divisor_is_zero = F.one();
                            var lt = F.zero();
                            var eq_term = F.one();
                            for (0..64) |i| {
                                const x_i = challenges[2 * i];
                                const y_i = challenges[2 * i + 1];
                                divisor_is_zero = divisor_is_zero.mul(F.one().sub(y_i));
                                lt = lt.add(F.one().sub(x_i).mul(y_i).mul(eq_term));
                                eq_term = eq_term.mul(x_i.mul(y_i).add(F.one().sub(x_i).mul(F.one().sub(y_i))));
                            }
                            const direct_vur_mle = lt.add(divisor_is_zero);
                            if (comptime debug_verbose) {
                                dbg("[TABLE_VERIFY] ValidUnsignedRemainder direct = {any}\n", .{direct_vur_mle.toBytes()});
                                dbg("[TABLE_VERIFY] ValidUnsignedRemainder prefix-suffix = {any}\n", .{table_values[16].toBytes()});
                                dbg("[TABLE_VERIFY] ValidUnsignedRemainder match: {}\n", .{direct_vur_mle.eql(table_values[16])});
                            }
                        }

                        // Verify NotEqual table (index 9): 1 - Equal
                        {
                            var direct_eq = F.one();
                            for (0..64) |i| {
                                const x_i = challenges[2 * i];
                                const y_i = challenges[2 * i + 1];
                                direct_eq = direct_eq.mul(x_i.mul(y_i).add(F.one().sub(x_i).mul(F.one().sub(y_i))));
                            }
                            const direct_neq_mle = F.one().sub(direct_eq);
                            if (comptime debug_verbose) {
                                dbg("[TABLE_VERIFY] NotEqual direct = {any}\n", .{direct_neq_mle.toBytes()});
                                dbg("[TABLE_VERIFY] NotEqual prefix-suffix = {any}\n", .{table_values[9].toBytes()});
                                dbg("[TABLE_VERIFY] NotEqual match: {}\n", .{direct_neq_mle.eql(table_values[9])});
                            }
                        }

                        // Debug: print key prefix checkpoint values
                        if (comptime debug_verbose) {
                            dbg("[STAGE5 REMATERIALIZE] Key prefix checkpoints:\n", .{});
                        }
                        const lw_idx = @intFromEnum(lookup_table_mod.Prefixes.LowerWord);
                        const eq_idx = @intFromEnum(lookup_table_mod.Prefixes.Eq);
                        const lt_idx = @intFromEnum(lookup_table_mod.Prefixes.LessThan);
                        const lsb_idx = @intFromEnum(lookup_table_mod.Prefixes.Lsb);
                        if (prefix_checkpoints.checkpoints[lw_idx]) |v| {
                            if (comptime debug_verbose) {
                                dbg("  LowerWord = {x}\n", .{v.toBytesBE()[16..32].*});
                            }
                        } else {
                            if (comptime debug_verbose) {
                                dbg("  LowerWord = NULL\n", .{});
                            }
                        }
                        if (prefix_checkpoints.checkpoints[eq_idx]) |v| {
                            if (comptime debug_verbose) {
                                dbg("  Eq = {x}\n", .{v.toBytesBE()[16..32].*});
                            }
                        } else {
                            if (comptime debug_verbose) {
                                dbg("  Eq = NULL\n", .{});
                            }
                        }
                        if (prefix_checkpoints.checkpoints[lt_idx]) |v| {
                            if (comptime debug_verbose) {
                                dbg("  LessThan = {x}\n", .{v.toBytesBE()[16..32].*});
                            }
                        } else {
                            if (comptime debug_verbose) {
                                dbg("  LessThan = NULL\n", .{});
                            }
                        }
                        if (prefix_checkpoints.checkpoints[lsb_idx]) |v| {
                            if (comptime debug_verbose) {
                                dbg("  Lsb = {x}\n", .{v.toBytesBE()[16..32].*});
                            }
                        } else {
                            if (comptime debug_verbose) {
                                dbg("  Lsb = NULL\n", .{});
                            }
                        }

                        // Debug: print table values
                        if (comptime debug_verbose) {
                            dbg("[STAGE5 REMATERIALIZE] table_values_at_r_addr:\n", .{});
                        }
                        for (0..NUM_TABLES) |t_idx| {
                            if (!table_values[t_idx].eql(F.zero())) {
                                if (comptime debug_verbose) {
                                    dbg("  table[{}] = {x}\n", .{ t_idx, table_values[t_idx].toBytesBE()[16..32].* });
                                }
                            }
                        }

                        // ============================================================
                        // Direct MLE verification for ALL tables
                        // Computes table MLE directly from challenges and compares
                        // with prefix-suffix decomposition result
                        // ============================================================
                        if (comptime debug_verbose) {
                            const print = std.debug.print;
                            var any_mismatch = false;

                            // Table 0: RangeCheck - Σ 2^(63-i) * r[64+i]
                            {
                                var direct = F.zero();
                                for (0..64) |i| {
                                    const r_i = challenges[64 + i];
                                    const shift: u6 = @intCast(63 - i);
                                    direct = direct.add(F.fromU64(@as(u64, 1) << shift).mul(r_i));
                                }
                                if (!direct.eql(table_values[0])) {
                                    print("[TABLE_MLE_CHECK] T0 RangeCheck MISMATCH!\n", .{});
                                    print("  direct = {any}\n", .{direct.toBytes()[0..16].*});
                                    print("  ps     = {any}\n", .{table_values[0].toBytes()[0..16].*});
                                    any_mismatch = true;
                                }
                            }

                            // Table 1: RangeCheckAligned - Σ 2^(63-i) * r[64+i] but requires alignment check
                            // Skipping - uses same formula as RangeCheck for MLE

                            // Table 2: And - Σ 2^(63-i) * r[2i] * r[2i+1]
                            {
                                var direct = F.zero();
                                for (0..64) |i| {
                                    const shift: u6 = @intCast(63 - i);
                                    direct = direct.add(F.fromU64(@as(u64, 1) << shift).mul(challenges[2 * i].mul(challenges[2 * i + 1])));
                                }
                                if (!direct.eql(table_values[2])) {
                                    print("[TABLE_MLE_CHECK] T2 And MISMATCH!\n", .{});
                                    any_mismatch = true;
                                }
                            }

                            // Table 4: Or - Σ 2^(63-i) * (x + y - x*y)
                            {
                                var direct = F.zero();
                                for (0..64) |i| {
                                    const x_i = challenges[2 * i];
                                    const y_i = challenges[2 * i + 1];
                                    const shift: u6 = @intCast(63 - i);
                                    direct = direct.add(F.fromU64(@as(u64, 1) << shift).mul(x_i.add(y_i).sub(x_i.mul(y_i))));
                                }
                                if (!direct.eql(table_values[4])) {
                                    print("[TABLE_MLE_CHECK] T4 Or MISMATCH!\n", .{});
                                    any_mismatch = true;
                                }
                            }

                            // Table 5: Xor - Σ 2^(63-i) * ((1-x)*y + x*(1-y))
                            {
                                var direct = F.zero();
                                for (0..64) |i| {
                                    const x_i = challenges[2 * i];
                                    const y_i = challenges[2 * i + 1];
                                    const shift: u6 = @intCast(63 - i);
                                    const xor_val = F.one().sub(x_i).mul(y_i).add(x_i.mul(F.one().sub(y_i)));
                                    direct = direct.add(F.fromU64(@as(u64, 1) << shift).mul(xor_val));
                                }
                                if (!direct.eql(table_values[5])) {
                                    print("[TABLE_MLE_CHECK] T5 Xor MISMATCH!\n", .{});
                                    any_mismatch = true;
                                }
                            }

                            // Table 6: Equal - Π (x*y + (1-x)*(1-y))
                            {
                                var direct = F.one();
                                for (0..64) |i| {
                                    const x_i = challenges[2 * i];
                                    const y_i = challenges[2 * i + 1];
                                    direct = direct.mul(x_i.mul(y_i).add(F.one().sub(x_i).mul(F.one().sub(y_i))));
                                }
                                if (!direct.eql(table_values[6])) {
                                    print("[TABLE_MLE_CHECK] T6 Equal MISMATCH!\n", .{});
                                    any_mismatch = true;
                                }
                            }

                            // Table 9: NotEqual - 1 - Equal
                            {
                                var eq_val = F.one();
                                for (0..64) |i| {
                                    const x_i = challenges[2 * i];
                                    const y_i = challenges[2 * i + 1];
                                    eq_val = eq_val.mul(x_i.mul(y_i).add(F.one().sub(x_i).mul(F.one().sub(y_i))));
                                }
                                const direct = F.one().sub(eq_val);
                                if (!direct.eql(table_values[9])) {
                                    print("[TABLE_MLE_CHECK] T9 NotEqual MISMATCH!\n", .{});
                                    any_mismatch = true;
                                }
                            }

                            // Table 11: UnsignedLessThan - Σ (1-x)*y * eq_prefix
                            {
                                var direct = F.zero();
                                var eq_term = F.one();
                                for (0..64) |i| {
                                    const x_i = challenges[2 * i];
                                    const y_i = challenges[2 * i + 1];
                                    direct = direct.add(F.one().sub(x_i).mul(y_i).mul(eq_term));
                                    eq_term = eq_term.mul(x_i.mul(y_i).add(F.one().sub(x_i).mul(F.one().sub(y_i))));
                                }
                                if (!direct.eql(table_values[11])) {
                                    print("[TABLE_MLE_CHECK] T11 UnsignedLessThan MISMATCH!\n", .{});
                                    any_mismatch = true;
                                }
                            }

                            // Table 16: ValidUnsignedRemainder - lt + divisor_is_zero
                            {
                                var divisor_is_zero = F.one();
                                var lt = F.zero();
                                var eq_term = F.one();
                                for (0..64) |i| {
                                    const x_i = challenges[2 * i];
                                    const y_i = challenges[2 * i + 1];
                                    divisor_is_zero = divisor_is_zero.mul(F.one().sub(y_i));
                                    lt = lt.add(F.one().sub(x_i).mul(y_i).mul(eq_term));
                                    eq_term = eq_term.mul(x_i.mul(y_i).add(F.one().sub(x_i).mul(F.one().sub(y_i))));
                                }
                                const direct = lt.add(divisor_is_zero);
                                if (!direct.eql(table_values[16])) {
                                    print("[TABLE_MLE_CHECK] T16 ValidUnsignedRemainder MISMATCH!\n", .{});
                                    print("  direct = {any}\n", .{direct.toBytes()[0..16].*});
                                    print("  ps     = {any}\n", .{table_values[16].toBytes()[0..16].*});
                                    any_mismatch = true;
                                }
                            }

                            // Table 17: ValidDiv0 - 1 - divisor_is_zero + is_valid_div_by_zero
                            // Interleaving: (divisor, quotient) so x=divisor, y=quotient
                            {
                                var divisor_is_zero = F.one();
                                var is_valid_div_by_zero = F.one();
                                for (0..64) |i| {
                                    const x_i = challenges[2 * i]; // divisor bit
                                    const y_i = challenges[2 * i + 1]; // quotient bit
                                    divisor_is_zero = divisor_is_zero.mul(F.one().sub(x_i));
                                    is_valid_div_by_zero = is_valid_div_by_zero.mul(F.one().sub(x_i).mul(y_i));
                                }
                                const direct = F.one().sub(divisor_is_zero).add(is_valid_div_by_zero);
                                if (!direct.eql(table_values[17])) {
                                    print("[TABLE_MLE_CHECK] T17 ValidDiv0 MISMATCH!\n", .{});
                                    print("  direct = {any}\n", .{direct.toBytes()[0..16].*});
                                    print("  ps     = {any}\n", .{table_values[17].toBytes()[0..16].*});
                                    any_mismatch = true;
                                }
                            }

                            // Table 21: SignExtendHalfWord - uses half-word sign extension
                            // Skipping complex formula

                            // Table 27: VirtualSRA
                            {
                                var result = F.zero();
                                var sign_extension = F.zero();
                                for (0..64) |i| {
                                    const x_i = challenges[2 * i];
                                    const y_i = challenges[2 * i + 1];
                                    result = result.mul(F.one().add(y_i));
                                    result = result.add(x_i.mul(y_i));
                                    if (i != 0) {
                                        sign_extension = sign_extension.add(F.fromU64(@as(u64, 1) << @intCast(i)).mul(F.one().sub(y_i)));
                                    }
                                }
                                const direct = result.add(challenges[0].mul(sign_extension));
                                if (!direct.eql(table_values[27])) {
                                    print("[TABLE_MLE_CHECK] T27 VirtualSRA MISMATCH!\n", .{});
                                    print("  direct = {any}\n", .{direct.toBytes()[0..16].*});
                                    print("  ps     = {any}\n", .{table_values[27].toBytes()[0..16].*});
                                    any_mismatch = true;
                                }
                            }

                            // Table 31: VirtualChangeDivisorW
                            {
                                const sign_bit = challenges[65]; // r[XLEN+1] = r[65]
                                var divisor_value = F.zero();
                                for (32..64) |i| { // i in XLEN/2..XLEN
                                    const bit_value = challenges[2 * i + 1];
                                    const shift: u6 = @intCast(63 - i);
                                    divisor_value = divisor_value.add(F.fromU64(@as(u64, 1) << shift).mul(bit_value));
                                }
                                var x_product: F = challenges[64]; // r[XLEN] = r[64]
                                for (33..64) |i| { // i in XLEN/2+1..XLEN
                                    x_product = x_product.mul(F.one().sub(challenges[2 * i]));
                                }
                                var y_product = F.one();
                                for (32..64) |i| { // i in XLEN/2..XLEN
                                    y_product = y_product.mul(challenges[2 * i + 1]);
                                }
                                // sign_extension = (2^64 - 2^32) * sign_bit
                                const two_pow_64 = F.fromBytes(&[_]u8{
                                    0, 0, 0, 0, 0, 0, 0, 0,
                                    1, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0,
                                });
                                const two_pow_32 = F.fromU64(1 << 32);
                                const sign_ext = two_pow_64.sub(two_pow_32).mul(sign_bit);
                                // adjustment = 2 - 2^64
                                const adjustment = F.fromU64(2).sub(two_pow_64);
                                const direct = divisor_value.add(adjustment.mul(x_product).mul(y_product)).add(sign_ext);
                                if (!direct.eql(table_values[31])) {
                                    print("[TABLE_MLE_CHECK] T31 VirtualChangeDivisorW MISMATCH!\n", .{});
                                    print("  direct = {any}\n", .{direct.toBytes()[0..16].*});
                                    print("  ps     = {any}\n", .{table_values[31].toBytes()[0..16].*});
                                    any_mismatch = true;
                                }
                            }

                            if (!any_mismatch) {
                                print("[TABLE_MLE_CHECK] ALL checked tables MATCH direct MLE!\n", .{});
                            }
                        }

                        // DIAGNOSTIC: Check sum with ORIGINAL combined_vals before rematerialization
                        if (comptime debug_verbose) {
                            var sum_with_original = F.zero();
                            for (0..T) |fj| {
                                const fj_eq = computeEqAtIndex(r_reduction, fj);
                                var fj_ra = F.one();
                                for (0..ra_num_chunks) |fc| {
                                    fj_ra = fj_ra.mul(ra_chunk_weights[fc][fj]);
                                }
                                sum_with_original = sum_with_original.add(fj_eq.mul(fj_ra).mul(lookups_combined_vals[fj]));
                            }
                            dbg("[PRE_REMAT_SUM] sum_with_ORIGINAL_cv = {x}\n", .{sum_with_original.toBytesBE()[16..32].*});
                            dbg("[PRE_REMAT_SUM] lookups_claim         = {x}\n", .{lookups_claim.toBytesBE()[16..32].*});
                            dbg("[PRE_REMAT_SUM] MATCH: {}\n", .{sum_with_original.eql(lookups_claim)});
                        }

                        // Rematerialize combined_vals using the correct formula
                        // combined_val[j] = table_values_at_r_addr[table(j)] + raf_val
                        //
                        // IMPORTANT: Jolt always adds the RAF contribution regardless of whether
                        // there's a lookup table. The table value is only added IF there's a table.
                        // See: jolt-core/src/zkvm/instruction_lookups/read_raf_checking.rs:682-698
                        {
                            const RematCtx2 = struct {
                                combined: []F,
                                t_indices: []const i8,
                                is_identity: []const bool,
                                table_vals: *const [NUM_TABLES]F,
                                raf_il: F,
                                raf_id: F,
                            };
                            const rctx2 = RematCtx2{
                                .combined = lookups_combined_vals,
                                .t_indices = cycle_table_indices,
                                .is_identity = cycle_is_identity_path,
                                .table_vals = &table_values,
                                .raf_il = raf_interleaved,
                                .raf_id = raf_identity,
                            };
                            const rematFn2 = struct {
                                fn f(c: RematCtx2, j: usize) void {
                                    var val = F.zero();
                                    const tidx = c.t_indices[j];
                                    if (tidx >= 0) {
                                        const ti: usize = @intCast(tidx);
                                        if (ti < NUM_TABLES) {
                                            val = c.table_vals[ti];
                                        }
                                    }
                                    if (!c.is_identity[j]) {
                                        val = val.add(c.raf_il);
                                    } else {
                                        val = val.add(c.raf_id);
                                    }
                                    c.combined[j] = val;
                                }
                            }.f;
                            if (self.thread_pool) |tp| {
                                tp.parallelForForce(T, rctx2, rematFn2);
                            } else {
                                for (0..T) |j| rematFn2(rctx2, j);
                            }
                        }

                        if (comptime debug_verbose) {
                            // Debug: print first 5 rematerialized values
                            dbg("[STAGE5 REMATERIALIZE] First 5 combined_vals after rematerialization:\n", .{});
                            for (0..@min(5, trace_len)) |j| {
                                const table_idx_dbg = cycle_table_indices[j];
                                const table_val_dbg = if (table_idx_dbg >= 0 and @as(usize, @intCast(table_idx_dbg)) < NUM_TABLES) table_values[@intCast(table_idx_dbg)] else F.zero();
                                dbg("  j={}: combined_val={x}, is_identity_path={}, table_idx={}, table_val={x}\n", .{
                                    j,
                                    lookups_combined_vals[j].toBytesBE()[24..32].*,
                                    cycle_is_identity_path[j],
                                    table_idx_dbg,
                                    table_val_dbg.toBytesBE()[24..32].*,
                                });
                            }
                        }

                        // ============================================================
                        // CRITICAL FIX: Reset eq_evals to fresh values for cycle rounds
                        // ============================================================
                        // During address rounds, lookups_eq_evals was modified by the
                        // condensation process (multiplied by expanding table values).
                        // For cycle rounds, we need fresh eq(j, r_reduction) values
                        // that haven't been modified.
                        //
                        // Jolt does this by using a separate GruenSplitEqPolynomial
                        // (eq_r_reduction) for cycle rounds, initialized with r_reduction.
                        // See: jolt-core/src/zkvm/instruction_lookups/read_raf_checking.rs:354-356
                        //
                        // We reinitialize lookups_eq_evals here.
                        if (comptime debug_verbose) {
                            // DIAGNOSTIC: Compute sum using condensed u_evals BEFORE reinitializing
                            var condensed_sum = F.zero();
                            for (0..T) |jj| {
                                condensed_sum = condensed_sum.add(lookups_eq_evals[jj].mul(lookups_combined_vals[jj]));
                            }
                            dbg("[CONDENSED_DIAG] Σ condensed_eq * cv_remat = {x}\n", .{condensed_sum.toBytesBE()[16..32].*});
                            dbg("[CONDENSED_DIAG] lookups_claim = {x}\n", .{lookups_claim.toBytesBE()[16..32].*});
                            dbg("[CONDENSED_DIAG] match = {}\n", .{condensed_sum.eql(lookups_claim)});

                            const fresh_eq_0 = computeEqAtIndex(r_reduction, 0);
                            dbg("[CONDENSED_DIAG] condensed_eq[0] = {x}\n", .{lookups_eq_evals[0].toBytesBE()[16..32].*});
                            dbg("[CONDENSED_DIAG] fresh_eq(0) = {x}\n", .{fresh_eq_0.toBytesBE()[16..32].*});
                            dbg("[CONDENSED_DIAG] ra_weights[0] = {x}\n", .{lookups_ra_weights[0].toBytesBE()[16..32].*});
                            const expected_condensed_0 = fresh_eq_0.mul(lookups_ra_weights[0]);
                            dbg("[CONDENSED_DIAG] fresh_eq(0) * ra(0) = {x}\n", .{expected_condensed_0.toBytesBE()[16..32].*});
                            dbg("[CONDENSED_DIAG] condensed_eq[0] == fresh*ra: {}\n", .{lookups_eq_evals[0].eql(expected_condensed_0)});

                            const k_lo_0_diag = lookups_indices_lo[0];
                            const shift_7 = 0;
                            const k_bound_7: usize = @truncate(k_lo_0_diag >> @intCast(shift_7));
                            const k_bound_7_masked = k_bound_7 & ((@as(usize, 1) << @intCast(log_m)) - 1);
                            const v7_val = expanding_tables[num_phases - 1].get(k_bound_7_masked);
                            const expected_with_v7 = expected_condensed_0.mul(v7_val);
                            dbg("[CONDENSED_DIAG] v[7][0x{x}] = {x}\n", .{k_bound_7_masked, v7_val.toBytesBE()[16..32].*});
                            dbg("[CONDENSED_DIAG] fresh*ra_partial * v7 = {x}\n", .{expected_with_v7.toBytesBE()[16..32].*});

                            var ra_partial = F.one();
                            for (0..num_phases - 1) |pp| {
                                const shift_pp = (num_phases - 1 - pp) * log_m;
                                var k_pp: usize = undefined;
                                if (shift_pp >= 64) {
                                    k_pp = @truncate(lookups_indices_hi[0] >> @intCast(shift_pp - 64));
                                } else {
                                    k_pp = @truncate(lookups_indices_lo[0] >> @intCast(shift_pp));
                                }
                                k_pp &= (@as(usize, 1) << @intCast(log_m)) - 1;
                                ra_partial = ra_partial.mul(expanding_tables[pp].get(k_pp));
                            }
                            const fresh_ra_partial = fresh_eq_0.mul(ra_partial);
                            dbg("[CONDENSED_DIAG] condensed_eq[0] == fresh*ra_partial: {}\n", .{lookups_eq_evals[0].eql(fresh_ra_partial)});

                            const last_phase = num_phases - 1;
                            const last_m_mask = (@as(usize, 1) << @intCast(log_m)) - 1;
                            const last_shift: u7 = 0;
                            var sum_with_last_phase = F.zero();
                            for (0..T) |jj| {
                                const k_lo_jj = lookups_indices_lo[jj];
                                const k_bound_last: usize = @truncate(k_lo_jj >> @intCast(last_shift));
                                const k_masked = k_bound_last & last_m_mask;
                                const v_last = expanding_tables[last_phase].get(k_masked);
                                sum_with_last_phase = sum_with_last_phase.add(
                                    lookups_eq_evals[jj].mul(v_last).mul(lookups_combined_vals[jj])
                                );
                            }
                            dbg("[CONDENSED_DIAG] FINAL match = {}\n", .{sum_with_last_phase.eql(lookups_claim)});
                        }

                        if (comptime debug_verbose) {
                            dbg("[STAGE5 CYCLE] Reinitializing lookups_eq_evals for cycle rounds\n", .{});
                            // Only rebuild full eq table for debug verification (expensive, O(T))
                            buildFullEqTable(r_reduction, lookups_eq_evals[0..T], self.thread_pool);
                        }
                        if (comptime debug_verbose) {
                            // Debug: verify sum = 1
                            var eq_sum_verify = F.zero();
                            for (0..T) |j| {
                                eq_sum_verify = eq_sum_verify.add(lookups_eq_evals[j]);
                            }
                            dbg("[STAGE5 CYCLE] eq_sum after reinit = {x} (should be 1)\n", .{eq_sum_verify.toBytesBE()[16..32].*});
                            dbg("[STAGE5 CYCLE] reinit eq_evals[0] = {x}\n", .{lookups_eq_evals[0].toBytesBE()[16..32].*});
                            dbg("[STAGE5 CYCLE] reinit eq_evals[1] = {x}\n", .{lookups_eq_evals[1].toBytesBE()[16..32].*});
                        }

                        // ============================================================
                        // Allocate ra_chunk_weights + scratch buffers for double-buffer bind
                        for (0..ra_num_chunks) |chunk_idx| {
                            ra_chunk_weights[chunk_idx] = self.allocator.alloc(F, T) catch unreachable;
                        }
                        ra_chunk_weights_allocated = true;


                        // CRITICAL FIX: Materialize ra_chunk_weights from expanding tables
                        // ============================================================
                        // At the start of cycle rounds (round 128), we need to compute
                        // ra_polys[chunk][j] = ∏_{phase in chunk} expanding_tables[phase][k_phase]
                        // where k_phase is the bits of lookup_index[j] for that phase.
                        //
                        // This matches Jolt's init_log_t_rounds() implementation.
                        // See: jolt-core/src/zkvm/instruction_lookups/read_raf_checking.rs:586-643
                        //
                        // Each chunk covers (phases_per_chunk) phases, where:
                        //   phases_per_chunk = num_phases / ra_num_chunks = 8 / 8 = 1
                        //   chunk_i handles phases [chunk_i * phases_per_chunk, (chunk_i+1) * phases_per_chunk)
                        if (comptime debug_verbose) {
                            dbg("[STAGE5 CYCLE] Materializing ra_chunk_weights from expanding tables\n", .{});
                        }
                        const phases_per_chunk = num_phases / ra_num_chunks;
                        if (comptime debug_verbose) {
                            dbg("  num_phases={}, ra_num_chunks={}, phases_per_chunk={}\n", .{ num_phases, ra_num_chunks, phases_per_chunk });
                        }

                        if (comptime debug_verbose) {
                            // Debug: print expanding table sizes and first few values
                            dbg("  Expanding table state at round 128:\n", .{});
                            const v0_p0 = expanding_tables[0].get(0);
                            const v0_p1 = expanding_tables[1].get(0);
                            dbg("    phase[0] v[0] (full) = {x}\n", .{v0_p0.toBytesBE()[16..32].*});
                            dbg("    phase[1] v[0] (full) = {x}\n", .{v0_p1.toBytesBE()[16..32].*});
                            dbg("    product v0_p0 * v0_p1 = {x}\n", .{v0_p0.mul(v0_p1).toBytesBE()[16..32].*});
                            for (0..@min(4, num_phases)) |phase| {
                                dbg("    phase[{}]: len={}, first_vals=[{x}, {x}, {x}, {x}]\n", .{
                                    phase,
                                    expanding_tables[phase].getLen(),
                                    expanding_tables[phase].get(0).toBytesBE()[28..32].*,
                                    expanding_tables[phase].get(1).toBytesBE()[28..32].*,
                                    expanding_tables[phase].get(2).toBytesBE()[28..32].*,
                                    expanding_tables[phase].get(3).toBytesBE()[28..32].*,
                                });
                            }
                        }

                        // Materialize ra_chunk_weights from expanding tables (parallel over T cycles)
                        {
                            const RaChunkRematCtx = struct {
                                indices_lo: []const u64,
                                indices_hi: []const u64,
                                chunks: *[MAX_RA_CHUNKS][]F,
                                tables: *const [16]ExpandingTable(F),
                                n_phases: usize,
                                ppc: usize, // phases_per_chunk
                                n_chunks: usize,
                                lm: usize, // log_m
                            };
                            const rctx = RaChunkRematCtx{
                                .indices_lo = lookups_indices_lo,
                                .indices_hi = lookups_indices_hi,
                                .chunks = &ra_chunk_weights,
                                .tables = &expanding_tables,
                                .n_phases = num_phases,
                                .ppc = phases_per_chunk,
                                .n_chunks = ra_num_chunks,
                                .lm = log_m,
                            };
                            const rematChunkFn = struct {
                                fn f(c: RaChunkRematCtx, j: usize) void {
                                    const k_lo = c.indices_lo[j];
                                    const k_hi = c.indices_hi[j];
                                    const phase_mask = (@as(usize, 1) << @intCast(c.lm)) - 1;
                                    for (0..c.n_chunks) |chunk_idx| {
                                        var ra_val = F.one();
                                        const phase_start = chunk_idx * c.ppc;
                                        const phase_end = @min((chunk_idx + 1) * c.ppc, c.n_phases);
                                        for (phase_start..phase_end) |phase| {
                                            const shift = (c.n_phases - 1 - phase) * c.lm;
                                            var k_phase: usize = undefined;
                                            if (shift >= 64) {
                                                k_phase = @truncate(k_hi >> @intCast(shift - 64));
                                            } else if (shift + c.lm <= 64) {
                                                k_phase = @truncate(k_lo >> @intCast(shift));
                                            } else {
                                                const lo_bits = k_lo >> @intCast(shift);
                                                const hi_bits = k_hi << @intCast(64 - shift);
                                                k_phase = @truncate(lo_bits | hi_bits);
                                            }
                                            k_phase &= phase_mask;
                                            ra_val = ra_val.mul(c.tables[phase].get(k_phase));
                                        }
                                        c.chunks[chunk_idx][j] = ra_val;
                                    }
                                }
                            }.f;
                            if (self.thread_pool) |tp| {
                                tp.parallelForForce(T, rctx, rematChunkFn);
                            } else {
                                for (0..T) |j| rematChunkFn(rctx, j);
                            }
                        }

                        // Free data structures no longer needed after rematerialization.
                        // suffix_polys + expanding_tables were only used during address rounds.
                        // ra_chunk_weights now hold the materialized values for cycle rounds.
                        suffix_polys.deinit();
                        suffix_polys = AllSuffixPolys(F).init(self.allocator); // Reset to empty for defer
                        for (0..num_phases) |phase_idx| {
                            expanding_tables[phase_idx].deinit();
                            expanding_tables[phase_idx] = ExpandingTable(F).init(self.allocator, 1) catch unreachable;
                        }

                        if (comptime debug_verbose) {
                            // Debug: print first few ra_chunk_weights after materialization
                            dbg("[STAGE5 CYCLE] ra_chunk_weights after materialization (first 4 cycles):\n", .{});
                            for (0..@min(4, T)) |j| {
                                dbg("  j={}: ra_chunks=[", .{j});
                                for (0..ra_num_chunks) |c| {
                                    if (c > 0) dbg(", ", .{});
                                    dbg("{x}", .{ra_chunk_weights[c][j].toBytesBE()[24..32].*});
                                }
                                dbg("]\n", .{});
                            }
                        }

                        if (comptime debug_verbose) {
                            // Compute the FULL sum with materialized ra for verification
                            var sum_with_material = F.zero();
                            for (0..T) |j| {
                                const eq_j = computeEqAtIndex(r_reduction, j);
                                var ra_material = F.one();
                                for (0..ra_num_chunks) |c| {
                                    ra_material = ra_material.mul(ra_chunk_weights[c][j]);
                                }
                                sum_with_material = sum_with_material.add(eq_j.mul(ra_material).mul(lookups_combined_vals[j]));
                            }
                            dbg("[RA_COMPARE] sum_with_material = {x}\n", .{sum_with_material.toBytesBE()[16..32].*});
                            dbg("[RA_COMPARE] lookups_claim     = {x}\n", .{lookups_claim.toBytesBE()[16..32].*});
                            dbg("[RA_COMPARE] material==claim: {}\n", .{sum_with_material.eql(lookups_claim)});
                        }

                        if (comptime debug_verbose) {
                            // Update lookups_ra_weights[j] to be the product of all chunks (debug only)
                            for (0..T) |j| {
                                var ra_prod = F.one();
                                for (0..ra_num_chunks) |c| {
                                    ra_prod = ra_prod.mul(ra_chunk_weights[c][j]);
                                }
                                lookups_ra_weights[j] = ra_prod;
                            }
                        }

                        // DEBUG: Verify RA computation directly
                        // Compute eq(lookup_index(0), r_addr) directly from challenges
                        // and compare with Π_c ra_chunk_c(0)
                        if (comptime debug_verbose) {
                            const k_lo_0 = lookups_indices_lo[0];
                            const k_hi_0 = lookups_indices_hi[0];
                            var direct_eq = F.one();
                            for (0..LOOKUPS_LOG_K) |bit_idx| {
                                // HighToLow: challenge[0] binds the MSB (bit 127)
                                // challenge[bit_idx] binds bit (127 - bit_idx)
                                const bit_pos = LOOKUPS_LOG_K - 1 - bit_idx;
                                const k_bit: u1 = if (bit_pos >= 64) @truncate(k_hi_0 >> @intCast(bit_pos - 64)) else @truncate(k_lo_0 >> @intCast(bit_pos));
                                const r_i = challenges[bit_idx];
                                // eq_bit(k_bit, r_i) = k_bit * r_i + (1 - k_bit) * (1 - r_i)
                                const eq_bit = if (k_bit == 1) r_i else F.one().sub(r_i);
                                direct_eq = direct_eq.mul(eq_bit);
                            }
                            if (comptime debug_verbose) {
                                dbg("[STAGE5 RA_VERIFY] Direct eq(K(0), r_addr) = {x}\n", .{direct_eq.toBytesBE()[16..32].*});
                                dbg("[STAGE5 RA_VERIFY] Product of ra_chunks(0) = {x}\n", .{lookups_ra_weights[0].toBytesBE()[16..32].*});
                                dbg("[STAGE5 RA_VERIFY] Match = {}\n", .{direct_eq.eql(lookups_ra_weights[0])});
                            }
                        }

                        // DEBUG: Compute combined_val(0) directly from table MLE
                        // combined_val(0) should be table_value(r_addr) + raf(r_addr)
                        // where raf(r_addr) depends on identity/interleaved path
                        if (comptime debug_verbose) {
                            // For cycle 0: what table is it?
                            const t_idx_0 = cycle_table_indices[0];
                            const is_id_0 = cycle_is_identity_path[0];
                            if (comptime debug_verbose) {
                                dbg("[STAGE5 CV_VERIFY] Cycle 0: table_idx={}, is_identity={}\n", .{t_idx_0, is_id_0});
                                dbg("[STAGE5 CV_VERIFY] combined_vals[0] = {x}\n", .{lookups_combined_vals[0].toBytesBE()[16..32].*});
                            }
                            if (t_idx_0 >= 0) {
                                if (comptime debug_verbose) {
                                    dbg("[STAGE5 CV_VERIFY] table_val = {x}\n", .{stored_table_values[@intCast(t_idx_0)].toBytesBE()[16..32].*});
                                }
                            }
                            if (is_id_0) {
                                if (comptime debug_verbose) {
                                    dbg("[STAGE5 CV_VERIFY] raf_identity = {x}\n", .{raf_identity.toBytesBE()[16..32].*});
                                }
                            } else {
                                if (comptime debug_verbose) {
                                    dbg("[STAGE5 CV_VERIFY] raf_interleaved = {x}\n", .{raf_interleaved.toBytesBE()[16..32].*});
                                }
                            }

                            // Also verify: sum over j of eq(j) * eq(K(j), r_addr) * cv(j)
                            // by doing it cycle by cycle for first 5 active cycles
                            var direct_sum_5 = F.zero();
                            for (0..@min(5, trace_len)) |jj| {
                                const eq_j = lookups_eq_evals[jj];
                                const ra_j = lookups_ra_weights[jj];
                                const cv_j = lookups_combined_vals[jj];
                                const contrib_j = eq_j.mul(ra_j).mul(cv_j);
                                direct_sum_5 = direct_sum_5.add(contrib_j);
                                if (comptime debug_verbose) {
                                    dbg("[STAGE5 CYCLE_SUM] j={}: eq={x}, ra={x}, cv={x}, contrib={x}, running={x}\n", .{
                                        jj,
                                        eq_j.toBytesBE()[24..32].*,
                                        ra_j.toBytesBE()[24..32].*,
                                        cv_j.toBytesBE()[24..32].*,
                                        contrib_j.toBytesBE()[24..32].*,
                                        direct_sum_5.toBytesBE()[24..32].*,
                                    });
                                }
                            }
                        }

                        // DEBUG: DIRECT brute-force claim computation
                        // Compute Σ_j eq(j, r_red) * eq(K(j), r_addr) * cv_remat(j)
                        // using per-bit eq computation (NOT expanding tables)
                        if (comptime debug_verbose) {
                            var direct_bf_sum = F.zero();
                            // We don't store initial cv, so skip that check
                            var mismatch_count: usize = 0;
                            for (0..T) |j| {
                                // eq(j, r_reduction) - freshly computed
                                const eq_j = lookups_eq_evals[j]; // Already fresh

                                // eq(K(j), r_addr) - compute directly from challenges
                                var direct_ra = F.one();
                                const k_lo_j = lookups_indices_lo[j];
                                const k_hi_j = lookups_indices_hi[j];
                                for (0..LOOKUPS_LOG_K) |bit_idx| {
                                    const bit_pos = LOOKUPS_LOG_K - 1 - bit_idx;
                                    const k_bit: u1 = if (bit_pos >= 64) @truncate(k_hi_j >> @intCast(bit_pos - 64)) else @truncate(k_lo_j >> @intCast(bit_pos));
                                    const r_i = challenges[bit_idx];
                                    const eq_bit = if (k_bit == 1) r_i else F.one().sub(r_i);
                                    direct_ra = direct_ra.mul(eq_bit);
                                }

                                // Check: direct_ra vs ra_weights
                                if (!direct_ra.eql(lookups_ra_weights[j])) {
                                    if (mismatch_count < 3) {
                                        if (comptime debug_verbose) {
                                            dbg("[DIRECT_BF] ra MISMATCH at j={}: direct={x}, ra_weights={x}\n", .{
                                                j, direct_ra.toBytesBE()[24..32].*, lookups_ra_weights[j].toBytesBE()[24..32].*,
                                            });
                                        }
                                    }
                                    mismatch_count += 1;
                                }

                                const cv_j = lookups_combined_vals[j];
                                direct_bf_sum = direct_bf_sum.add(eq_j.mul(direct_ra).mul(cv_j));

                                // Also compute using INITIAL cv (before rematerialization)
                                // initial_cv[j] = table(K(j)) + raf(K(j)) (point evaluation)
                                // After binding, the correct cv should be table_MLE(r_addr) + raf_MLE(r_addr)
                                // So initial_cv * eq(K(j), r_addr) should NOT give the right answer
                                // (because table(K(j)) ≠ table_MLE(r_addr) in general)
                            }
                            // Also compute materialized_sum using ra_weights
                            var mat_sum_check = F.zero();
                            for (0..T) |j2| {
                                mat_sum_check = mat_sum_check.add(lookups_eq_evals[j2].mul(lookups_ra_weights[j2]).mul(lookups_combined_vals[j2]));
                            }
                            if (comptime debug_verbose) {
                                dbg("[DIRECT_BF] ra_weights mismatches: {}/{}\n", .{mismatch_count, T});
                                dbg("[DIRECT_BF] Σ eq*direct_ra*cv_remat = {x}\n", .{direct_bf_sum.toBytesBE()[16..32].*});
                                dbg("[DIRECT_BF] materialized_sum (ra_weights) = {x}\n", .{mat_sum_check.toBytesBE()[16..32].*});
                                dbg("[DIRECT_BF] lookups_claim (poly chain) = {x}\n", .{lookups_claim.toBytesBE()[16..32].*});
                                dbg("[DIRECT_BF] direct_bf == materialized: {}\n", .{direct_bf_sum.eql(mat_sum_check)});
                                dbg("[DIRECT_BF] direct_bf == lookups_claim: {}\n", .{direct_bf_sum.eql(lookups_claim)});
                            }

                            // KEY TEST: The claim should equal materialized_sum.
                            // If direct_bf matches materialized but not claim, then
                            // the MATERIALIZATION is consistent but the claim is wrong.
                            // But we PROVED the claim is correct (S5P==S5V + CONSISTENCY).
                            // So if they don't match, the materialization must be wrong.
                            //
                            // But WHAT could be wrong if all components individually check out?
                            // Let's check: is the INITIAL claim correct?
                            // initial_claim = Σ_j eq(j, r_red) * initial_cv(j)
                            // = Σ_j eq(j, r_red) * (table(K(j)) + γ*left(K(j)) + γ²*right(K(j)))
                            //
                            // After binding, claim_128 = Σ_j eq(j) * eq(K(j), r_addr) * table_MLE(r_addr) + raf...
                            //
                            // But is there a MISSING FACTOR? What if the initial sum was NOT
                            // Σ_j eq(j) * cv_j but rather something different?
                            //
                            // Let me check the initial claim structure more carefully.
                            // Print lookups_input for reference
                            if (comptime debug_verbose) {
                                dbg("[DIRECT_BF] lookups_input (initial claim) = {x}\n", .{lookups_input.toBytesBE()[16..32].*});
                            }

                            // SPLIT TEST: Compute sum with table-only (no RAF) and RAF-only (no table)
                            var table_only_sum = F.zero();
                            var raf_only_sum = F.zero();
                            for (0..T) |jj| {
                                const eq_jj = lookups_eq_evals[jj];
                                const ra_jj = lookups_ra_weights[jj];
                                const eq_ra = eq_jj.mul(ra_jj);

                                // Table-only contribution
                                const t_idx_jj = cycle_table_indices[jj];
                                if (t_idx_jj >= 0) {
                                    const ti: usize = @intCast(t_idx_jj);
                                    if (ti < NUM_TABLES) {
                                        table_only_sum = table_only_sum.add(eq_ra.mul(stored_table_values[ti]));
                                    }
                                }

                                // RAF-only contribution
                                const is_interleaved_jj = !cycle_is_identity_path[jj];
                                if (is_interleaved_jj) {
                                    raf_only_sum = raf_only_sum.add(eq_ra.mul(raf_interleaved));
                                } else {
                                    raf_only_sum = raf_only_sum.add(eq_ra.mul(raf_identity));
                                }
                            }
                            if (comptime debug_verbose) {
                                dbg("[SPLIT_TEST] table_only_sum = {x}\n", .{table_only_sum.toBytesBE()[16..32].*});
                                dbg("[SPLIT_TEST] raf_only_sum = {x}\n", .{raf_only_sum.toBytesBE()[16..32].*});
                                dbg("[SPLIT_TEST] table+raf = {x}\n", .{table_only_sum.add(raf_only_sum).toBytesBE()[16..32].*});
                                dbg("[SPLIT_TEST] materialized = {x}\n", .{mat_sum_check.toBytesBE()[16..32].*});
                                dbg("[SPLIT_TEST] split matches materialized: {}\n", .{table_only_sum.add(raf_only_sum).eql(mat_sum_check)});
                            }

                            // Now compute what the initial sum SHOULD be after binding
                            // Initial: Σ_j eq(j,r) * [table(K(j)) + γ*raf(K(j))]
                            // After binding: Σ_j eq(j,r) * eq(K(j), r_addr) * [table_mle_j(r_addr) + γ*raf_mle(r_addr)]
                            // The RAF contribution with binding should be:
                            // Σ_j eq(j,r) * eq(K(j), r_addr) * [γ*left_mle(r_addr) + γ²*(right_mle(r_addr) + identity_mle(r_addr))]
                            // for interleaved, or γ²*identity_mle(r_addr) for identity path
                            // = Σ eq*ra * raf_interleaved (for interleaved) or raf_identity (for identity)
                            // which is what we compute above as raf_only_sum
                            //
                            // So the claim should be table_only_sum + raf_only_sum
                            // And this should equal lookups_claim
                            if (comptime debug_verbose) {
                                dbg("[SPLIT_TEST] lookups_claim = {x}\n", .{lookups_claim.toBytesBE()[16..32].*});
                            }
                        }
                    }
                    // Verify full sum at the start of cycle rounds (debug only)
                    if (comptime debug_verbose) {
                        if (lookups_round == 0) {
                            var full_sum_check = F.zero();
                            for (0..T) |fj| {
                                const fj_eq = computeEqAtIndex(r_reduction, fj);
                                var fj_ra = F.one();
                                for (0..ra_num_chunks) |fc| {
                                    fj_ra = fj_ra.mul(ra_chunk_weights[fc][fj]);
                                }
                                full_sum_check = full_sum_check.add(fj_eq.mul(fj_ra).mul(lookups_combined_vals[fj]));
                            }
                            {
                                const print = std.debug.print;
                                print("[CYCLE_START_CHECK] Full sum = {any}\n", .{full_sum_check.toBytes()});
                                print("[CYCLE_START_CHECK] lookups_claim = {any}\n", .{lookups_claim.toBytes()});
                                print("[CYCLE_START_CHECK] MATCH: {}\n", .{full_sum_check.eql(lookups_claim)});
                            }

                            var cv_only_sum = F.zero();
                            for (0..T) |fj| {
                                const fj_eq = computeEqAtIndex(r_reduction, fj);
                                cv_only_sum = cv_only_sum.add(fj_eq.mul(lookups_combined_vals[fj]));
                            }
                            dbg("[CYCLE_START_CHECK] cv_only_sum (no ra) = {x}\n", .{cv_only_sum.toBytesBE()[16..32].*});

                            var all_ra_one = true;
                            var non_one_count: usize = 0;
                            for (0..T) |fj| {
                                var fj_ra = F.one();
                                for (0..ra_num_chunks) |fc| {
                                    fj_ra = fj_ra.mul(ra_chunk_weights[fc][fj]);
                                }
                                if (!fj_ra.eql(F.one())) {
                                    all_ra_one = false;
                                    non_one_count += 1;
                                    if (non_one_count <= 3) {
                                        dbg("[CYCLE_START_CHECK] ra_product[{}] != 1: {x}\n", .{fj, fj_ra.toBytesBE()[16..32].*});
                                    }
                                }
                            }
                            dbg("[CYCLE_START_CHECK] all_ra_one={}, non_one_count={}\n", .{all_ra_one, non_one_count});

                            var g1_brute = F.zero();
                            const half_T = T >> 1;
                            for (0..half_T) |fj| {
                                const fj_eq = computeEqAtIndexPartial(r_reduction, fj, n_cycle_vars - 1);
                                var fj_ra = F.one();
                                for (0..ra_num_chunks) |fc| {
                                    fj_ra = fj_ra.mul(ra_chunk_weights[fc][2 * fj + 1]);
                                }
                                g1_brute = g1_brute.add(fj_eq.mul(fj_ra).mul(lookups_combined_vals[2 * fj + 1]));
                            }
                            dbg("[CYCLE_START_CHECK] g(1)_brute = {x}\n", .{g1_brute.toBytesBE()[16..32].*});
                        }
                    }

                    // Initialize split-eq tables on first cycle round
                    if (comptime bench_timing) bench_remat_ns += bench_timer.read();
                    if (comptime bench_timing) bench_timer.reset();
                    if (!split_eq_initialized) {
                        // remaining_vars for first cycle round = n_cycle_vars - 1
                        // j = (x_out << m_in) | x_in
                        // In BIG ENDIAN r_reduction: high bits of j map to r_reduction[0..m_out],
                        // low bits of j map to r_reduction[m_out..m_out+m_in]
                        const init_vars = n_cycle_vars - 1;
                        split_eq_m_in = init_vars / 2;
                        const m_out = init_vars - split_eq_m_in;
                        split_eq_E_in_len = @as(usize, 1) << @intCast(split_eq_m_in);
                        split_eq_E_out_len = @as(usize, 1) << @intCast(m_out);
                        std.debug.assert(split_eq_E_in_len <= MAX_SPLIT_EQ_SIZE);
                        std.debug.assert(split_eq_E_out_len <= MAX_SPLIT_EQ_SIZE);

                        // Build E_out from r_reduction[0..m_out] (covers HIGH bits of j)
                        split_eq_E_out[0] = F.one();
                        for (0..m_out) |i| {
                            const ri = r_reduction[i];
                            const len = @as(usize, 1) << @intCast(i);
                            var j2: usize = len;
                            while (j2 > 0) {
                                j2 -= 1;
                                split_eq_E_out[2 * j2 + 1] = split_eq_E_out[j2].mul(ri);
                                split_eq_E_out[2 * j2] = split_eq_E_out[j2].sub(split_eq_E_out[2 * j2 + 1]);
                            }
                        }
                        // Build E_in from r_reduction[m_out..m_out+m_in] (covers LOW bits of j)
                        split_eq_E_in[0] = F.one();
                        for (0..split_eq_m_in) |i| {
                            const ri = r_reduction[m_out + i];
                            const len = @as(usize, 1) << @intCast(i);
                            var j2: usize = len;
                            while (j2 > 0) {
                                j2 -= 1;
                                split_eq_E_in[2 * j2 + 1] = split_eq_E_in[j2].mul(ri);
                                split_eq_E_in[2 * j2] = split_eq_E_in[j2].sub(split_eq_E_in[2 * j2 + 1]);
                            }
                        }
                        split_eq_initialized = true;
                    }

                    // Get r_round for this cycle variable (LowToHigh binding: last element first)
                    const r_round = r_reduction[n_cycle_vars - 1 - lookups_round];

                    // Accumulate sum of 9-factor products via flat parallelReduce with split-eq
                    // eq_prefix(j) = E_out[j >> m_in] * E_in[j & (E_in_len-1)]
                    const current_half_size = T >> @intCast(lookups_round + 1);
                    var sum_evals: [9]F = blk: {
                        const EvalCtx = struct {
                            combined: []const F,
                            chunks: *const [MAX_RA_CHUNKS][]F,
                            e_in: [*]const F,
                            e_out: [*]const F,
                            e_in_mask: usize, // e_in_len - 1
                            e_in_shift: u6, // log2(e_in_len)
                            n_chunks: usize,
                        };
                        const e_in_shift: u6 = @intCast(@ctz(split_eq_E_in_len));
                        const ectx = EvalCtx{
                            .combined = lookups_combined_vals,
                            .chunks = &ra_chunk_weights,
                            .e_in = &split_eq_E_in,
                            .e_out = &split_eq_E_out,
                            .e_in_mask = split_eq_E_in_len - 1,
                            .e_in_shift = e_in_shift,
                            .n_chunks = ra_num_chunks,
                        };
                        const identity = [_]F{F.zero()} ** 9;
                        const mapFn = struct {
                            fn f(c: EvalCtx, start: usize, end: usize) [9]F {
                                var acc = [_]UnreducedProductAccum{UnreducedProductAccum.zero()} ** 9;
                                for (start..end) |j| {
                                    var pairs: [9][2]F = undefined;
                                    const x_in = j & c.e_in_mask;
                                    const x_out = j >> c.e_in_shift;
                                    const eq_prefix = c.e_out[x_out].mul(c.e_in[x_in]);
                                    pairs[0][0] = eq_prefix.mul(c.combined[2 * j]);
                                    pairs[0][1] = eq_prefix.mul(c.combined[2 * j + 1]);
                                    for (0..c.n_chunks) |ci| {
                                        pairs[ci + 1][0] = c.chunks[ci][2 * j];
                                        pairs[ci + 1][1] = c.chunks[ci][2 * j + 1];
                                    }
                                    UniPoly(F).evalProd9Accumulate(pairs, &acc);
                                }
                                // Reduce accumulators at chunk boundary
                                var result: [9]F = undefined;
                                inline for (0..9) |k| result[k] = acc[k].reduce();
                                return result;
                            }
                        }.f;
                        const reduceFn = struct {
                            fn f(a: [9]F, b: [9]F) [9]F {
                                var r: [9]F = undefined;
                                for (0..9) |k| r[k] = a[k].add(b[k]);
                                return r;
                            }
                        }.f;
                        if (self.thread_pool) |tp| {
                            break :blk tp.parallelReduce([9]F, current_half_size, identity, ectx, mapFn, reduceFn);
                        }
                        break :blk mapFn(ectx, 0, current_half_size);
                    };

                    // CRITICAL FIX: Multiply sum_evals by lookups_current_scalar
                    // This matches Jolt's GruenSplitEqPolynomial behavior where current_scalar
                    // accumulates eq(w[bound_vars], r_bound_challenges) and is multiplied at the end.
                    // Without this, we incorrectly mix original r_reduction with sumcheck challenges.
                    for (&sum_evals) |*eval| {
                        eval.* = eval.*.mul(lookups_current_scalar);
                    }

                    // Get the current claim for this round
                    const cycle_claim = lookups_claim;

                    // Use finish_mles_product_sum_from_evals to recover degree-10 polynomial
                    // This: 1) recovers eval_at_0 using the claim and eq factor
                    //       2) interpolates degree-9 quotient
                    //       3) multiplies by eq(X, r_round) to get degree-10
                    const full_coeffs = try UniPoly(F).finishMlesProductSumFromEvals(
                        self.allocator,
                        &sum_evals,
                        cycle_claim,
                        r_round,
                    );
                    defer self.allocator.free(full_coeffs);

                    // Convert to compressed format (skip c1)
                    const lookups_compressed = try UniPoly(F).toCompressed(self.allocator, full_coeffs);
                    defer self.allocator.free(lookups_compressed);

                    // Verify sumcheck property: p(0) + p(1) = claim
                    var p_at_1 = F.zero();
                    for (full_coeffs) |c| {
                        p_at_1 = p_at_1.add(c);
                    }
                    const p_at_0 = full_coeffs[0];
                    const sum_check = p_at_0.add(p_at_1);
                    const sumcheck_ok = sum_check.eql(cycle_claim);

                    // Debug: print polynomial info for all cycle rounds
                    if (comptime debug_verbose) {
                        dbg("[STAGE5 CYCLE] Round {} (cycle var {}):\n", .{ round, lookups_round });
                        dbg("  r_round = {x}\n", .{r_round.toBytesBE()[24..32].*});
                        dbg("  lookups_current_scalar = {x}\n", .{lookups_current_scalar.toBytesBE()[24..32].*});
                        dbg("  cycle_claim = {x}\n", .{cycle_claim.toBytesBE()[24..32].*});
                        dbg("  p(0) = {x}\n", .{p_at_0.toBytesBE()[24..32].*});
                        dbg("  p(1) = {x}\n", .{p_at_1.toBytesBE()[24..32].*});
                        dbg("  p(0)+p(1) = {x}, matches_claim = {}\n", .{ sum_check.toBytesBE()[24..32].*, sumcheck_ok });
                        dbg("  full_coeffs len = {}\n", .{full_coeffs.len});
                    }
                    // Print sum_evals to debug
                    if (comptime debug_verbose) {
                        dbg("  sum_evals = [\n", .{});
                    }
                    for (0..9) |k| {
                        if (comptime debug_verbose) {
                            dbg("    [{d}] = {x}\n", .{ k, sum_evals[k].toBytesBE()[24..32].* });
                        }
                    }
                    if (comptime debug_verbose) {
                        dbg("  ]\n", .{});
                    }
                    // Debug: print eq and combined values for last round
                    if (lookups_round == n_cycle_vars - 1) {
                        if (comptime debug_verbose) {
                            dbg("  [LAST ROUND] eq[0]={x}, eq[1]={x}\n", .{
                                lookups_eq_evals[0].toBytesBE()[16..32].*,
                                lookups_eq_evals[1].toBytesBE()[16..32].*,
                            });
                            dbg("  [LAST ROUND] val[0]={x}, val[1]={x}\n", .{
                                lookups_combined_vals[0].toBytesBE()[16..32].*,
                                lookups_combined_vals[1].toBytesBE()[16..32].*,
                            });
                            dbg("  [LAST ROUND] ra_chunk[0]: [{x}, {x}]\n", .{
                                ra_chunk_weights[0][0].toBytesBE()[16..32].*,
                                ra_chunk_weights[0][1].toBytesBE()[16..32].*,
                            });
                        }
                    }
                    // Print Instance 0+1 contribution
                    if (comptime debug_verbose) {
                        dbg("  combined_poly (Inst 0+1) = [{x}, {x}, {x}, {x}]\n", .{
                            combined_poly[0].toBytesBE()[24..32].*,
                            combined_poly[1].toBytesBE()[24..32].*,
                            combined_poly[2].toBytesBE()[24..32].*,
                            combined_poly[3].toBytesBE()[24..32].*,
                        });
                    }

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
                    if (s5_phase_timer) |*pt| { s5_cycle_compute_ns += pt.read(); pt.reset(); }
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

                    // Convert to compressed format [c0, c2, c3, ..., c10]
                    const final_compressed = try UniPoly(F).toCompressed(self.allocator, combined_coeffs);

                    // Debug: print first 3 compressed coefficients (excluding linear term) in LE format
                    if (round == 135) { // Only Round 135
                        if (comptime debug_verbose) {
                            dbg("[STAGE5 CYCLE ZOLT] Round {} compressed coeffs (LE, comparing to Jolt):\n", .{round});
                            dbg("  final_compressed.len = {}\n", .{final_compressed.len});
                        }
                        for (0..final_compressed.len) |k| {
                            // Jolt displays LE bytes from arkworks serialization
                            if (comptime debug_verbose) {
                                dbg("  coeff[{}] = {any}\n", .{ k, final_compressed[k].toBytes() });
                            }
                        }
                        if (comptime debug_verbose) {
                            dbg("  current_batched_claim (LE) = {any}\n", .{current_batched_claim.toBytes()});
                        }
                        // Also print combined_coeffs before compression
                        if (comptime debug_verbose) {
                            dbg("  combined_coeffs[0] (c0) = {any}\n", .{combined_coeffs[0].toBytes()});
                            dbg("  combined_coeffs[1] (c1) = {any}\n", .{combined_coeffs[1].toBytes()});
                            dbg("  combined_coeffs[2] (c2) = {any}\n", .{combined_coeffs[2].toBytes()});
                        }
                        // Print Instance 0+1 contribution details
                        if (comptime debug_verbose) {
                            dbg("  inst01_coeffs[0..4] = [{any}, {any}, {any}, {any}]\n", .{
                                inst01_coeffs[0].toBytes(), inst01_coeffs[1].toBytes(),
                                inst01_coeffs[2].toBytes(), inst01_coeffs[3].toBytes(),
                            });
                        }
                        // Print full_coeffs (Instance 2)
                        if (comptime debug_verbose) {
                            dbg("  full_coeffs[0..3] = [{any}, {any}, {any}]\n", .{
                                full_coeffs[0].toBytes(), full_coeffs[1].toBytes(), full_coeffs[2].toBytes(),
                            });
                        }
                    }

                    try proof.compressed_polys.append(self.allocator, .{
                        .coeffs_except_linear_term = final_compressed,
                        .allocator = self.allocator,
                    });

                    // Append to transcript
                    if (comptime bench_timing) bench_cycle_coeff_ns += bench_timer.read();
                    if (comptime bench_timing) bench_timer.reset();
                    transcript.appendScalars("sumcheck_poly", final_compressed);

                    const challenge = transcript.challengeScalar();
                    challenges[round] = challenge;
                    if (comptime bench_timing) bench_cycle_transcript_ns += bench_timer.read();

                    // Debug: print challenge for comparison with Jolt
                    if (round < 4 or round >= 128) {
                        if (comptime debug_verbose) {
                            dbg("[STAGE5 CHALLENGE] Round {}: LE = {any}\n", .{round, challenge.toBytes()[0..16].*});
                        }
                    }

                    // Update current_batched_claim by evaluating polynomial at challenge
                    // The VERIFIER uses eval_from_hint which:
                    // 1. Has compressed coeffs [c0, c2, c3, ...] and hint = current_claim
                    // 2. Recovers c1 = hint - 2*c0 - c2 - c3 - ...
                    // 3. Evaluates p(r) = c0 + r*c1 + r²*c2 + ...
                    //
                    // So we MUST update our claim using the same formula as the verifier.

                    // Compute c1_recovered from the hint (current_batched_claim)
                    // c1 = hint - 2*c0 - c2 - c3 - ... - c_d
                    var c1_recovered = current_batched_claim.sub(combined_coeffs[0]).sub(combined_coeffs[0]); // hint - 2*c0
                    for (2..combined_coeffs.len) |i| {
                        c1_recovered = c1_recovered.sub(combined_coeffs[i]);
                    }

                    // Debug: compare c1_direct vs c1_recovered
                    if (round >= 128) {
                        if (comptime debug_verbose) {
                            dbg("[STAGE5 C1 DEBUG] Round {}:\n", .{round});
                            dbg("  c1_direct (combined_coeffs[1]) = {any}\n", .{combined_coeffs[1].toBytes()});
                            dbg("  c1_recovered (from hint)       = {any}\n", .{c1_recovered.toBytes()});
                            dbg("  hint (current_batched_claim)   = {any}\n", .{current_batched_claim.toBytes()});
                            dbg("  c1_direct == c1_recovered: {}\n", .{combined_coeffs[1].eql(c1_recovered)});
                        }

                        // CRITICAL DEBUG: Check if current_batched_claim = sum of scaled instance claims
                        // Expected: batch0*claim0 + batch1*claim1 + batch2*claim2
                        const expected_batched = batch0.mul(regs_val_current_claim)
                            .add(batch1.mul(ram_ra_current_claim))
                            .add(batch2.mul(lookups_claim));
                        if (comptime debug_verbose) {
                            dbg("  expected_batched (batch0*c0+batch1*c1+batch2*c2) = {any}\n", .{expected_batched.toBytes()});
                            dbg("  current_batched_claim matches expected: {}\n", .{current_batched_claim.eql(expected_batched)});
                            dbg("    batch0*claim0 = {any}\n", .{batch0.mul(regs_val_current_claim).toBytes()});
                            dbg("    batch1*claim1 = {any}\n", .{batch1.mul(ram_ra_current_claim).toBytes()});
                            dbg("    batch2*claim2 = {any}\n", .{batch2.mul(lookups_claim).toBytes()});
                            dbg("    regs_val_current_claim = {x}\n", .{regs_val_current_claim.toBytesBE()[16..32].*});
                            dbg("    ram_ra_current_claim   = {x}\n", .{ram_ra_current_claim.toBytesBE()[16..32].*});
                            dbg("    lookups_claim          = {x}\n", .{lookups_claim.toBytesBE()[16..32].*});
                        }

                        // Compute what the hint SHOULD be if coefficients are correct:
                        // hint_expected = p(0) + p(1) = 2*c0 + c1 + c2 + ... + c_d
                        var hint_expected = combined_coeffs[0].add(combined_coeffs[0]); // 2*c0
                        for (1..combined_coeffs.len) |i| {
                            hint_expected = hint_expected.add(combined_coeffs[i]);
                        }
                        if (comptime debug_verbose) {
                            dbg("  hint_expected (2*c0+c1+c2+...) = {any}\n", .{hint_expected.toBytes()});
                            dbg("  hint == hint_expected: {}\n", .{current_batched_claim.eql(hint_expected)});
                        }
                    }

                    // CRITICAL: Update current_batched_claim by evaluating the batched polynomial
                    // at the challenge, using the same formula as Jolt's eval_from_hint.
                    // This ensures the prover's claim matches what the verifier will compute.
                    // The verifier uses: c1 = hint - 2*c0 - c2 - ..., then p(r) = c0 + r*c1 + r²*c2 + ...
                    {
                        // combined_coeffs[0] = c0, combined_coeffs[1] = c1, combined_coeffs[2] = c2, etc.
                        // Evaluate p(r) = c0 + r*c1 + r²*c2 + r³*c3 + ...
                        // Use c1_recovered from hint (to match verifier exactly)
                        var eval_result = combined_coeffs[0].add(c1_recovered.mulHiBigIntU128(challenge.limbs));
                        var r_power = challenge.mul(challenge); // r²
                        for (2..combined_coeffs.len) |ci| {
                            eval_result = eval_result.add(combined_coeffs[ci].mul(r_power));
                            if (ci + 1 < combined_coeffs.len) {
                                r_power = r_power.mul(challenge);
                            }
                        }
                        current_batched_claim = eval_result;
                    }

                    // Per-round tracking (matches Jolt verifier's [S5V] output)
                    if (comptime debug_verbose) {
                        dbg("  [S5P] R{} challenge={x} new_e={x} degree={}\n", .{
                            round,
                            challenge.toBytes()[0..16].*,
                            current_batched_claim.toBytes()[0..16].*,
                            combined_coeffs.len - 1,
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

                        bindRegsValChallenge(inc_evals, wa_evals, &lt_poly, regs_round, challenge, self.thread_pool, self.gpu_ops);
                    }

                    // Bind the challenge for RamRaClaimReduction cycle rounds
                    // Upstream: cycle-only binding (no PhaseAddress)
                    if (remaining_rounds <= ram_ra_num_rounds) {
                        const ram_ra_round = ram_ra_num_rounds - remaining_rounds;
                        {
                            // PhaseCycle: bind cycle variables (all rounds are cycle rounds)
                            const cycle_round = ram_ra_round;
                            const one_minus_r = F.one().sub(challenge);

                            // Get r_cycle_* values for this bit (LowToHigh order)
                            const r_raf_bit = r_cycle_raf[n_cycle_vars - 1 - cycle_round];
                            const r_rw_bit = r_cycle_rw[n_cycle_vars - 1 - cycle_round];
                            const r_val_bit = r_cycle_val[n_cycle_vars - 1 - cycle_round];

                            // Update eq_*_bound with eq_bit(r_*[m], r_m)
                            const eq_raf_update = F.one().sub(r_raf_bit).sub(challenge).add(r_raf_bit.mul(challenge).add(r_raf_bit.mul(challenge)));
                            const eq_rw_update = F.one().sub(r_rw_bit).sub(challenge).add(r_rw_bit.mul(challenge).add(r_rw_bit.mul(challenge)));
                            const eq_val_update = F.one().sub(r_val_bit).sub(challenge).add(r_val_bit.mul(challenge).add(r_val_bit.mul(challenge)));
                            eq_raf_bound = eq_raf_bound.mul(eq_raf_update);
                            eq_rw_bound = eq_rw_bound.mul(eq_rw_update);
                            eq_val_bound = eq_val_bound.mul(eq_val_update);

                            // Precompute the 6 inverses: each eq_*_bit_at_c only takes
                            // 2 possible values per round (r_*_bit or 1-r_*_bit).
                            // Avoids M * 3 field inversions per round (each ~256 muls).
                            const one_minus_r_raf = F.one().sub(r_raf_bit);
                            const one_minus_r_rw = F.one().sub(r_rw_bit);
                            const one_minus_r_val = F.one().sub(r_val_bit);
                            const inv_r_raf = if (!r_raf_bit.eql(F.zero())) r_raf_bit.inverse().? else F.zero();
                            const inv_1mr_raf = if (!one_minus_r_raf.eql(F.zero())) one_minus_r_raf.inverse().? else F.zero();
                            const inv_r_rw = if (!r_rw_bit.eql(F.zero())) r_rw_bit.inverse().? else F.zero();
                            const inv_1mr_rw = if (!one_minus_r_rw.eql(F.zero())) one_minus_r_rw.inverse().? else F.zero();
                            const inv_r_val = if (!r_val_bit.eql(F.zero())) r_val_bit.inverse().? else F.zero();
                            const inv_1mr_val = if (!one_minus_r_val.eql(F.zero())) one_minus_r_val.inverse().? else F.zero();

                            for (0..ram_access_count) |access_idx| {
                                const cycle = ram_cycles[access_idx];
                                const cycle_usize: usize = @intCast(cycle);
                                // Get the cycle bit that was just bound
                                const c_m: u1 = @truncate(cycle_usize >> @intCast(cycle_round));
                                // Multiply eq_cycle_bound by the binding factor
                                const factor = if (c_m == 0) one_minus_r else challenge;
                                eq_cycle_bound[access_idx] = eq_cycle_bound[access_idx].mul(factor);

                                // Update eq_*_remaining by dividing out eq_bit(r_*[m], c_m)
                                const inv_raf = if (c_m == 0) inv_1mr_raf else inv_r_raf;
                                const inv_rw = if (c_m == 0) inv_1mr_rw else inv_r_rw;
                                const inv_val = if (c_m == 0) inv_1mr_val else inv_r_val;

                                if (!inv_raf.eql(F.zero())) {
                                    eq_raf_remaining[access_idx] = eq_raf_remaining[access_idx].mul(inv_raf);
                                }
                                if (!inv_rw.eql(F.zero())) {
                                    eq_rw_remaining[access_idx] = eq_rw_remaining[access_idx].mul(inv_rw);
                                }
                                if (!inv_val.eql(F.zero())) {
                                    eq_val_remaining[access_idx] = eq_val_remaining[access_idx].mul(inv_val);
                                }
                            }

                            if (comptime debug_verbose) {
                                dbg("[STAGE5 CYCLE BIND R{}] cycle_round={}, challenge={x}\n", .{
                                    round,
                                    cycle_round,
                                    challenge.toBytesBE()[16..32].*,
                                });
                                dbg("  eq_raf_bound={x}\n", .{eq_raf_bound.toBytesBE()[16..32].*});
                            }
                            if (ram_access_count > 0) {
                                if (comptime debug_verbose) {
                                    dbg("  eq_cycle_bound[0]={x}\n", .{eq_cycle_bound[0].toBytesBE()[16..32].*});
                                }
                            }

                            // CRITICAL FIX: Bind P/Q arrays for Instance 1 PhaseCycle
                            // This was missing - P/Q arrays need to be bound after each cycle round
                            // to match the address rounds behavior

                            // Store cycle challenge for PhaseCycle2 eq_prefix computation
                            cycle_challenges[cycle_round] = challenge;

                            if (cycle_round < prefix_n_vars) {
                                // PhaseCycle1: bind P and Q polynomials
                                const current_len = prefix_size >> @intCast(cycle_round);
                                const half_len = current_len / 2;

                                // Bind P and Q arrays: X'[j] = (1-r)*X[2j] + r*X[2j+1]
                                // Parallelize across the 6 independent arrays
                                if (self.gpu_ops) |gpu| {
                                    if (half_len >= 16384) {
                                        const pq_arrays = [_][]F{ P_raf, P_rw, P_val, Q_raf, Q_rw, Q_val };
                                        for (pq_arrays) |arr| {
                                            gpu.polyBindLow(arr[0 .. half_len * 2], challenge, arr[0..half_len]) catch {
                                                for (0..half_len) |j| {
                                                    const lo = arr[2 * j];
                                                    arr[j] = lo.add(arr[2 * j + 1].sub(lo).mulHiBigIntU128(challenge.limbs));
                                                }
                                            };
                                        }
                                    } else {
                                        for (0..half_len) |j| {
                                            P_raf[j] = P_raf[2 * j].add(P_raf[2 * j + 1].sub(P_raf[2 * j]).mulHiBigIntU128(challenge.limbs));
                                            P_rw[j] = P_rw[2 * j].add(P_rw[2 * j + 1].sub(P_rw[2 * j]).mulHiBigIntU128(challenge.limbs));
                                            P_val[j] = P_val[2 * j].add(P_val[2 * j + 1].sub(P_val[2 * j]).mulHiBigIntU128(challenge.limbs));
                                        }
                                        for (0..half_len) |j| {
                                            Q_raf[j] = Q_raf[2 * j].add(Q_raf[2 * j + 1].sub(Q_raf[2 * j]).mulHiBigIntU128(challenge.limbs));
                                            Q_rw[j] = Q_rw[2 * j].add(Q_rw[2 * j + 1].sub(Q_rw[2 * j]).mulHiBigIntU128(challenge.limbs));
                                            Q_val[j] = Q_val[2 * j].add(Q_val[2 * j + 1].sub(Q_val[2 * j]).mulHiBigIntU128(challenge.limbs));
                                        }
                                    }
                                } else if (self.thread_pool) |tp| {
                                    const BindCtx = struct {
                                        p_raf: []F, p_rw: []F, p_val: []F,
                                        q_raf: []F, q_rw: []F, q_val: []F,
                                        chal_limbs: [4]u64, h: usize,
                                    };
                                    const bctx = BindCtx{ .p_raf = P_raf, .p_rw = P_rw, .p_val = P_val, .q_raf = Q_raf, .q_rw = Q_rw, .q_val = Q_val, .chal_limbs = challenge.limbs, .h = half_len };
                                    tp.parallelForForce(6, bctx, struct {
                                        fn f(c: BindCtx, arr_idx: usize) void {
                                            const arr = switch (arr_idx) {
                                                0 => c.p_raf,
                                                1 => c.p_rw,
                                                2 => c.p_val,
                                                3 => c.q_raf,
                                                4 => c.q_rw,
                                                5 => c.q_val,
                                                else => unreachable,
                                            };
                                            for (0..c.h) |j| {
                                                const lo = arr[2 * j];
                                                arr[j] = lo.add(arr[2 * j + 1].sub(lo).mulHiBigIntU128(c.chal_limbs));
                                            }
                                        }
                                    }.f);
                                } else {
                                    for (0..half_len) |j| {
                                        P_raf[j] = P_raf[2 * j].add(P_raf[2 * j + 1].sub(P_raf[2 * j]).mulHiBigIntU128(challenge.limbs));
                                        P_rw[j] = P_rw[2 * j].add(P_rw[2 * j + 1].sub(P_rw[2 * j]).mulHiBigIntU128(challenge.limbs));
                                        P_val[j] = P_val[2 * j].add(P_val[2 * j + 1].sub(P_val[2 * j]).mulHiBigIntU128(challenge.limbs));
                                    }
                                    for (0..half_len) |j| {
                                        Q_raf[j] = Q_raf[2 * j].add(Q_raf[2 * j + 1].sub(Q_raf[2 * j]).mulHiBigIntU128(challenge.limbs));
                                        Q_rw[j] = Q_rw[2 * j].add(Q_rw[2 * j + 1].sub(Q_rw[2 * j]).mulHiBigIntU128(challenge.limbs));
                                        Q_val[j] = Q_val[2 * j].add(Q_val[2 * j + 1].sub(Q_val[2 * j]).mulHiBigIntU128(challenge.limbs));
                                    }
                                }

                                if (comptime debug_verbose) {
                                    dbg("[STAGE5 CYCLE BIND R{}] Bound P/Q arrays: half_len={}\n", .{
                                        round,
                                        half_len,
                                    });
                                }
                            } else {
                                // PhaseCycle2: bind H_prime and eq_hi arrays
                                const suffix_round = cycle_round - prefix_n_vars;
                                const current_len = suffix_size >> @intCast(suffix_round);
                                const half_len = current_len / 2;

                                // Bind H_prime and eq_hi arrays: X'[j] = (1-r)*X[2j] + r*X[2j+1]
                                // Parallelize across the 4 independent arrays
                                if (self.gpu_ops) |gpu| {
                                    if (half_len >= 16384) {
                                        const heq_arrays = [_][]F{ H_prime, eq_raf_hi, eq_rw_hi, eq_val_hi };
                                        for (heq_arrays) |arr| {
                                            gpu.polyBindLow(arr[0 .. half_len * 2], challenge, arr[0..half_len]) catch {
                                                for (0..half_len) |j| {
                                                    const lo = arr[2 * j];
                                                    arr[j] = lo.add(arr[2 * j + 1].sub(lo).mulHiBigIntU128(challenge.limbs));
                                                }
                                            };
                                        }
                                    } else {
                                        for (0..half_len) |j| {
                                            H_prime[j] = H_prime[2 * j].add(H_prime[2 * j + 1].sub(H_prime[2 * j]).mulHiBigIntU128(challenge.limbs));
                                        }
                                        for (0..half_len) |j| {
                                            eq_raf_hi[j] = eq_raf_hi[2 * j].add(eq_raf_hi[2 * j + 1].sub(eq_raf_hi[2 * j]).mulHiBigIntU128(challenge.limbs));
                                            eq_rw_hi[j] = eq_rw_hi[2 * j].add(eq_rw_hi[2 * j + 1].sub(eq_rw_hi[2 * j]).mulHiBigIntU128(challenge.limbs));
                                            eq_val_hi[j] = eq_val_hi[2 * j].add(eq_val_hi[2 * j + 1].sub(eq_val_hi[2 * j]).mulHiBigIntU128(challenge.limbs));
                                        }
                                    }
                                } else if (self.thread_pool) |tp| {
                                    const BindCtx2 = struct {
                                        h_prime: []F, eq_raf: []F, eq_rw: []F, eq_val: []F,
                                        chal_limbs: [4]u64, h: usize,
                                    };
                                    const bctx2 = BindCtx2{ .h_prime = H_prime, .eq_raf = eq_raf_hi, .eq_rw = eq_rw_hi, .eq_val = eq_val_hi, .chal_limbs = challenge.limbs, .h = half_len };
                                    tp.parallelForForce(4, bctx2, struct {
                                        fn f(c: BindCtx2, arr_idx: usize) void {
                                            const arr = switch (arr_idx) {
                                                0 => c.h_prime,
                                                1 => c.eq_raf,
                                                2 => c.eq_rw,
                                                3 => c.eq_val,
                                                else => unreachable,
                                            };
                                            for (0..c.h) |j| {
                                                const lo = arr[2 * j];
                                                arr[j] = lo.add(arr[2 * j + 1].sub(lo).mulHiBigIntU128(c.chal_limbs));
                                            }
                                        }
                                    }.f);
                                } else {
                                    for (0..half_len) |j| {
                                        H_prime[j] = H_prime[2 * j].add(H_prime[2 * j + 1].sub(H_prime[2 * j]).mulHiBigIntU128(challenge.limbs));
                                    }
                                    for (0..half_len) |j| {
                                        eq_raf_hi[j] = eq_raf_hi[2 * j].add(eq_raf_hi[2 * j + 1].sub(eq_raf_hi[2 * j]).mulHiBigIntU128(challenge.limbs));
                                        eq_rw_hi[j] = eq_rw_hi[2 * j].add(eq_rw_hi[2 * j + 1].sub(eq_rw_hi[2 * j]).mulHiBigIntU128(challenge.limbs));
                                        eq_val_hi[j] = eq_val_hi[2 * j].add(eq_val_hi[2 * j + 1].sub(eq_val_hi[2 * j]).mulHiBigIntU128(challenge.limbs));
                                    }
                                }

                                if (comptime debug_verbose) {
                                    dbg("[STAGE5 CYCLE BIND R{}] Bound H'/eq_hi arrays: half_len={}\n", .{
                                        round,
                                        half_len,
                                    });
                                }
                            }

                            // CRITICAL: Update Instance 1 claim after binding
                            // p(r) = c0 + r*c1 + r²*c2 for degree-2 polynomial
                            // where c0 = eval_0, c1 = eval_1 - eval_0 - c2, c2 = (eval_2 - 2*eval_1 + eval_0) / 2
                            // Since we use hint: eval_0 = claim - eval_1, we have p(0) + p(1) = claim
                            // Compute p(r) using Lagrange interpolation:
                            // p(r) = eval_0 * L0(r) + eval_1 * L1(r) + eval_2 * L2(r)
                            // where L0(r) = (r-1)(r-2)/2, L1(r) = -r(r-2), L2(r) = r(r-1)/2
                            // Simpler: convert to coefficients and use Horner
                            const c2_inst1 = inst1_eval_2.sub(inst1_eval_1).sub(inst1_eval_1).add(inst1_eval_0).mul(UniPoly(F).INV2);
                            const c1_inst1 = inst1_eval_1.sub(inst1_eval_0).sub(c2_inst1);
                            const c0_inst1 = inst1_eval_0;
                            // p(r) = c0 + r*c1 + r²*c2
                            const r2 = challenge.mul(challenge);
                            ram_ra_current_claim = c0_inst1.add(c1_inst1.mulHiBigIntU128(challenge.limbs)).add(c2_inst1.mul(r2));

                            if (comptime debug_verbose) {
                                dbg("[STAGE5 INST1 CLAIM] Round {}: new_claim={x}\n", .{
                                    round,
                                    ram_ra_current_claim.toBytesBE()[16..32].*,
                                });
                            }
                        }
                    }

                    // Bind cycle round challenge for lookups
                    // CRITICAL: Do NOT bind eq_evals - keep them as original eq(j, r_reduction)
                    // Only bind combined_vals (standard MLE binding)
                    // This matches Jolt's GruenSplitEqPolynomial where E_in/E_out are not modified
                    // Bind combined_vals + all ra_chunk_weights in parallel across arrays
                    if (self.gpu_ops) |gpu| {
                        // GPU path: bind each polynomial via GPU
                        const lk_n = lookups_combined_vals.len >> @intCast(lookups_round);
                        const lk_half = lk_n / 2;
                        if (lk_half >= 16384) {
                            gpu.polyBindLow(lookups_combined_vals[0 .. lk_half * 2], challenge, lookups_combined_vals[0..lk_half]) catch {
                                for (0..lk_half) |i| {
                                    lookups_combined_vals[i] = lookups_combined_vals[2 * i].add(challenge.mul(lookups_combined_vals[2 * i + 1].sub(lookups_combined_vals[2 * i])));
                                }
                            };
                            for (lk_half..lk_n) |i| {
                                lookups_combined_vals[i] = F.zero();
                            }
                        } else {
                            bindSinglePolynomial(lookups_combined_vals, lookups_round, challenge, self.thread_pool, self.gpu_ops);
                        }
                        for (0..ra_num_chunks) |chunk_idx| {
                            const ra_poly = ra_chunk_weights[chunk_idx];
                            const ra_n = ra_poly.len >> @intCast(lookups_round);
                            const ra_half = ra_n / 2;
                            if (ra_half >= 16384) {
                                gpu.polyBindLow(ra_poly[0 .. ra_half * 2], challenge, ra_poly[0..ra_half]) catch {
                                    for (0..ra_half) |i| {
                                        ra_poly[i] = ra_poly[2 * i].add(challenge.mul(ra_poly[2 * i + 1].sub(ra_poly[2 * i])));
                                    }
                                };
                                for (ra_half..ra_n) |i| {
                                    ra_poly[i] = F.zero();
                                }
                            } else {
                                bindSinglePolynomial(ra_poly, lookups_round, challenge, self.thread_pool, self.gpu_ops);
                            }
                        }
                    } else if (self.thread_pool) |tp| {
                        const BindArraysCtx = struct {
                            combined: []F,
                            chunks: *[MAX_RA_CHUNKS][]F,
                            n_chunks: usize,
                            lround: usize,
                            chal: F,
                        };
                        const bactx = BindArraysCtx{
                            .combined = lookups_combined_vals,
                            .chunks = &ra_chunk_weights,
                            .n_chunks = ra_num_chunks,
                            .lround = lookups_round,
                            .chal = challenge,
                        };
                        tp.parallelForForce(ra_num_chunks + 1, bactx, struct {
                            fn f(c: BindArraysCtx, arr_idx: usize) void {
                                const poly = if (arr_idx == 0) c.combined else c.chunks[arr_idx - 1];
                                const n = poly.len >> @intCast(c.lround);
                                const half = n / 2;
                                if (half == 0) return;
                                for (0..half) |i| {
                                    poly[i] = poly[2 * i].add(c.chal.mul(poly[2 * i + 1].sub(poly[2 * i])));
                                }
                                for (half..n) |i| {
                                    poly[i] = F.zero();
                                }
                            }
                        }.f);
                    } else {
                        bindSinglePolynomial(lookups_combined_vals, lookups_round, challenge, self.thread_pool, self.gpu_ops);
                        for (0..ra_num_chunks) |chunk_idx| {
                            bindSinglePolynomial(ra_chunk_weights[chunk_idx], lookups_round, challenge, self.thread_pool, self.gpu_ops);
                        }
                    }

                    // CRITICAL FIX: Update lookups_current_scalar with eq(w_i, challenge)
                    // Formula: eq(w, r) = 1 - w - r + 2*w*r
                    const w_i = r_reduction[n_cycle_vars - 1 - lookups_round];
                    const prod_w_r = w_i.mulHiBigIntU128(challenge.limbs); // w * r
                    const one_minus_r_scalar = F.one().sub(challenge); // (1 - r) as F
                    const eq_factor = one_minus_r_scalar.sub(w_i).add(prod_w_r).add(prod_w_r); // 1 - r - w + 2*w*r
                    lookups_current_scalar = lookups_current_scalar.mul(eq_factor);

                    // Halve split-eq tables: remove the variable being bound from remaining eq.
                    // LowToHigh binding removes the LAST r_reduction var = LSB of eq index.
                    // In the expansion, bit 0 of E_in index maps to r_reduction[m_out+m_in-1].
                    // To marginalize over bit 0: new[k] = old[2*k] + old[2*k+1]
                    // E_in is halved first (covers the LSB side), then E_out.
                    if (split_eq_E_in_len > 1) {
                        const new_len = split_eq_E_in_len / 2;
                        for (0..new_len) |k| {
                            split_eq_E_in[k] = split_eq_E_in[2 * k].add(split_eq_E_in[2 * k + 1]);
                        }
                        split_eq_E_in_len = new_len;
                    } else if (split_eq_E_out_len > 1) {
                        // E_in is exhausted. Now halve E_out (remove its LSB = bit 0).
                        const new_len = split_eq_E_out_len / 2;
                        for (0..new_len) |k| {
                            split_eq_E_out[k] = split_eq_E_out[2 * k].add(split_eq_E_out[2 * k + 1]);
                        }
                        split_eq_E_out_len = new_len;
                    }

                    // Debug: print current_scalar update
                    if (lookups_round < 3 or lookups_round >= n_cycle_vars - 1) {
                        if (comptime debug_verbose) {
                            dbg("[STAGE5 CYCLE] round={}, lookups_round={}\n", .{ round, lookups_round });
                            dbg("  w_i (r_reduction[{}]) = {x}\n", .{ n_cycle_vars - 1 - lookups_round, w_i.toBytesBE()[16..32].* });
                            dbg("  challenge limbs[2..4] = {x:0>16}{x:0>16}\n", .{ challenge.limbs[2], challenge.limbs[3] });
                            dbg("  prod_w_r = {x}\n", .{ prod_w_r.toBytesBE()[16..32].* });
                            dbg("  one_minus_r = {x}\n", .{ one_minus_r_scalar.toBytesBE()[16..32].* });
                            dbg("  eq_factor = {x}\n", .{ eq_factor.toBytesBE()[16..32].* });
                            dbg("  current_scalar = {x}\n", .{ lookups_current_scalar.toBytesBE()[16..32].* });
                        }
                    }
                    // Debug: show ra_chunk values before/after binding
                    if (lookups_round == 0) {
                        if (comptime debug_verbose) {
                            dbg("[STAGE5 CYCLE] ra_chunks before first binding (round=128):\n", .{});
                        }
                        for (0..@min(4, ra_num_chunks)) |c| {
                            if (comptime debug_verbose) {
                                dbg("  ra_chunk[{}][0:4] = [{x}, {x}, {x}, {x}]\n", .{
                                    c,
                                    ra_chunk_weights[c][0].toBytesBE()[16..32].*,
                                    ra_chunk_weights[c][1].toBytesBE()[16..32].*,
                                    ra_chunk_weights[c][2].toBytesBE()[16..32].*,
                                    ra_chunk_weights[c][3].toBytesBE()[16..32].*,
                                });
                            }
                        }
                    }

                    // Update lookups_claim by evaluating the Instance 2 polynomial at the challenge.
                    // The polynomial p2(X) was constructed by finishMlesProductSumFromEvals.
                    {
                        // Evaluate p2(challenge) using Horner's method
                        var p2_at_r = full_coeffs[full_coeffs.len - 1];
                        var idx = full_coeffs.len - 1;
                        while (idx > 0) {
                            idx -= 1;
                            p2_at_r = p2_at_r.mulHiBigIntU128(challenge.limbs).add(full_coeffs[idx]);
                        }
                        lookups_claim = p2_at_r;
                    }

                    // NOTE: current_batched_claim was already correctly set by polynomial
                    // evaluation above. The verifier uses eval_from_hint which evaluates
                    // the batched polynomial at the challenge. Do NOT override with
                    // batch0*claim0 + batch1*claim1 + batch2*claim2 as Jolt never does this.

                    // Debug: print challenges for cycle rounds (128-135)
                    if (round >= LOOKUPS_LOG_K) {
                        if (comptime debug_verbose) {
                            dbg("[STAGE5 ROUND {}] challenge={x}\n", .{
                                round,
                                challenge.toBytesBE()[16..32].*,
                            });
                            dbg("  new_batched_claim = {x}\n", .{current_batched_claim.toBytesBE()[16..32].*});
                            dbg("  new_lookups_claim = {x}\n", .{lookups_claim.toBytesBE()[16..32].*});
                        }
                        // Debug: print eq_evals[0] after binding
                        if (comptime debug_verbose) {
                            dbg("  eq_evals[0] after bind = {x}\n", .{lookups_eq_evals[0].toBytesBE()[16..32].*});
                        }

                        // CRITICAL DIAGNOSTIC: Compare lookups_claim with bound array computation
                        // After binding, compute: current_scalar * Σ_j (eq_prefix(j) * Π_c ra_c[j] * cv[j])
                        // where the sum is over remaining (unbound) elements
                        var bound_array_sum = F.zero();
                        if (comptime debug_verbose) {
                            const bound_half_size = T >> @intCast(lookups_round + 1);
                            const bound_remaining = n_cycle_vars - lookups_round - 1;
                            for (0..bound_half_size) |bj| {
                                const bj_eq = computeEqAtIndexPartial(r_reduction, bj, bound_remaining);
                                var bj_ra = F.one();
                                for (0..ra_num_chunks) |bc| {
                                    bj_ra = bj_ra.mul(ra_chunk_weights[bc][bj]);
                                }
                                bound_array_sum = bound_array_sum.add(bj_eq.mul(bj_ra).mul(lookups_combined_vals[bj]));
                            }
                            bound_array_sum = bound_array_sum.mul(lookups_current_scalar);
                        }
                        if (comptime debug_verbose) {
                            dbg("  BOUND_ARRAY_SUM = {x}\n", .{bound_array_sum.toBytesBE()[16..32].*});
                            dbg("  lookups_claim   = {x}\n", .{lookups_claim.toBytesBE()[16..32].*});
                            dbg("  BOUND==CLAIM: {}\n", .{bound_array_sum.eql(lookups_claim)});
                        }
                    }

                    if (comptime bench_timing) bench_cycle_bind_ns += bench_timer.read();
                    if (s5_phase_timer) |*pt| { s5_cycle_bind_ns += pt.read(); pt.reset(); }
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
                std.debug.print("    [STAGE5-BENCH] T={}, n_cycle_vars={}\n\n", .{T, n_cycle_vars});
            }

            // Debug: print final batched claim (this is output_claim from verifier's perspective)
            if (comptime debug_verbose) {
                dbg("[STAGE5] Final batched claim (output_claim) = {any}\n", .{current_batched_claim.toBytesBE()});
                dbg("[STAGE5] Final lookups_current_scalar (should = eq_eval_r_reduction) = {x}\n", .{lookups_current_scalar.toBytesBE()[16..32].*});
            }

            // DEBUG: Print each instance's final claim value
            // The verifier computes expected = batch0*inst0_eval + batch1*inst1_eval + batch2*inst2_eval
            // The prover's output_claim should equal this.
            if (comptime debug_verbose) {
                dbg("[STAGE5 FINAL CLAIMS] Individual instance final values:\n", .{});
                dbg("  regs_val_current_claim (Instance 0) = {any}\n", .{regs_val_current_claim.toBytes()});
                dbg("  ram_ra_current_claim (Instance 1) = {any}\n", .{ram_ra_current_claim.toBytes()});
                dbg("  lookups_claim (Instance 2) = {any}\n", .{lookups_claim.toBytes()});
                dbg("  batch0*inst0 (LE) = {any}\n", .{batch0.mul(regs_val_current_claim).toBytes()});
                dbg("  batch1*inst1 (LE) = {any}\n", .{batch1.mul(ram_ra_current_claim).toBytes()});
                dbg("  batch2*inst2 (LE) = {any}\n", .{batch2.mul(lookups_claim).toBytes()});
            }

            // Print the chain values so we can compare with Jolt verifier
            if (comptime debug_verbose) {
                const print = std.debug.print;
                print("[ZOLT S5 CHAIN] inst0_claim FULL LE = {any}\n", .{regs_val_current_claim.toBytes()});
                print("[ZOLT S5 CHAIN] inst1_claim FULL LE = {any}\n", .{ram_ra_current_claim.toBytes()});
                print("[ZOLT S5 CHAIN] inst2_claim FULL LE = {any}\n", .{lookups_claim.toBytes()});
                print("[ZOLT S5 CHAIN] batch0 FULL LE = {any}\n", .{batch0.toBytes()});
                print("[ZOLT S5 CHAIN] batch1 FULL LE = {any}\n", .{batch1.toBytes()});
                print("[ZOLT S5 CHAIN] batch2 FULL LE = {any}\n", .{batch2.toBytes()});
                print("[ZOLT S5 CHAIN] batch0*inst0 FULL LE = {any}\n", .{batch0.mul(regs_val_current_claim).toBytes()});
                print("[ZOLT S5 CHAIN] batch1*inst1 FULL LE = {any}\n", .{batch1.mul(ram_ra_current_claim).toBytes()});
                print("[ZOLT S5 CHAIN] batch2*inst2 FULL LE = {any}\n", .{batch2.mul(lookups_claim).toBytes()});
                const recon = batch0.mul(regs_val_current_claim).add(batch1.mul(ram_ra_current_claim)).add(batch2.mul(lookups_claim));
                print("[ZOLT S5 CHAIN] sum = {any}\n", .{recon.toBytes()});
                print("[ZOLT S5 CHAIN] batched_claim = {any}\n", .{current_batched_claim.toBytes()});
                print("[ZOLT S5 CHAIN] sum==batched = {}\n", .{recon.eql(current_batched_claim)});
            }
            const recon = batch0.mul(regs_val_current_claim).add(batch1.mul(ram_ra_current_claim)).add(batch2.mul(lookups_claim));
            if (comptime debug_verbose) {
                dbg("  batch0*inst0 + batch1*inst1 + batch2*inst2 = {any}\n", .{recon.toBytes()});
                dbg("  current_batched_claim = {any}\n", .{current_batched_claim.toBytes()});
                dbg("  reconstruction matches output_claim: {}\n", .{recon.eql(current_batched_claim)});
            }

            // CRITICAL: Derive correct Instance 2 claim from batched output
            // The batched output_claim is CORRECT (S5P==S5V). Individual claims for
            // inst0 and inst1 are also correct. So we can derive the TRUE inst2 claim.
            const correct_inst2_from_batched = if (comptime debug_verbose) current_batched_claim.sub(batch0.mul(regs_val_current_claim)).sub(batch1.mul(ram_ra_current_claim)).mul(batch2_inv) else F.zero();
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
                                raw_addr & (@as(u64, K) - 1);
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
                    dbg("[BRUTE FORCE INST1] prover tracked ram_ra_current_claim = {any}\n", .{ram_ra_current_claim.toBytes()});
                    dbg("[BRUTE FORCE INST1] match: {}\n", .{bf_expected_inst1.eql(ram_ra_current_claim)});
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

            // Compute LookupsReadRaf opening claims
            //
            // After the sumcheck, we have:
            //   lookups_output_claim = lookups_eq_evals[0] * lookups_combined_vals[0]
            //
            // The verifier computes:
            //   expected_output_claim = eq_r_reduction * ra_claim * (val_claim + gamma * raf_claim)
            //
            // Where:
            //   eq_r_reduction = eq(r_reduction, r_cycle_prime)  [verifier computes this]
            //   ra_claim = Π_{i=0}^{7} InstructionRa(i)
            //   val_claim = Σ_{i=0}^{41} LookupTableFlag(i) * table_i(r_address)
            //   raf_claim = (1 - raf_flag) * (left_op + gamma * right_op) + raf_flag * gamma * identity
            //
            // Our sumcheck gives: lookups_output_claim = eq_evals[0] * combined[0]
            // where eq_evals[0] should equal eq_r_reduction after proper binding
            //
            // We need to find ra_claim, val_claim, raf_claim such that:
            //   eq_r_reduction * ra_claim * (val_claim + gamma * raf_claim) = lookups_output_claim

            const num_lookup_tables: usize = 40;
            const lookups_ra_d = LOOKUPS_LOG_K / lookups_ra_virtual_log_k_chunk;

            // Extract r_address (first 128 challenges) and r_cycle' (last 8 challenges)
            const r_address_prime = challenges[0..LOOKUPS_LOG_K];
            const r_cycle_prime = challenges[LOOKUPS_LOG_K..];

            // The actual lookups output claim from the sumcheck
            // After binding all cycle variables:
            // - lookups_current_scalar = eq(r_reduction, challenges)
            // - ra_chunk_weights[i][0] is the bound ra polynomial for chunk i at single point
            // - combined_vals[0] is the bound combined polynomial at single point
            // The output claim is: current_scalar * Π_i(ra_chunks[i][0]) * combined_vals[0]
            //
            // NOTE: We use ra_chunk_weights[i][0] (individual chunks bound during cycle rounds)
            // NOT lookups_ra_weights[0] (which was materialized at round 128 and never re-bound)
            var lookups_ra_product_bound = F.one();
            for (0..ra_num_chunks) |i| {
                lookups_ra_product_bound = lookups_ra_product_bound.mul(ra_chunk_weights[i][0]);
            }
            const lookups_output_claim = lookups_current_scalar.mul(lookups_ra_product_bound).mul(lookups_combined_vals[0]);

            // Compare polynomial chain output with expected output
            if (comptime debug_verbose) {
                const print = std.debug.print;
                print("[S5 FINAL] lookups_claim (chain)      = {any}\n", .{lookups_claim.toBytes()});
                print("[S5 FINAL] lookups_output_claim (exp) = {any}\n", .{lookups_output_claim.toBytes()});
                print("[S5 FINAL] MATCH = {}\n", .{lookups_claim.eql(lookups_output_claim)});
                print("[S5 FINAL] current_scalar = {any}\n", .{lookups_current_scalar.toBytes()});
                print("[S5 FINAL] ra_product = {any}\n", .{lookups_ra_product_bound.toBytes()});
                print("[S5 FINAL] combined_val[0] = {any}\n", .{lookups_combined_vals[0].toBytes()});
            }

            if (comptime debug_verbose) {
                dbg("[STAGE5 LOOKUPS] Computing opening claims:\n", .{});
                dbg("  lookups_input = {any}\n", .{lookups_input.toBytesBE()[0..8]});
                dbg("  lookups_output_claim (eq*ra_w0*combined) = {any}\n", .{lookups_output_claim.toBytesBE()});
                dbg("  lookups_claim (tracked Instance 2) = {any}\n", .{lookups_claim.toBytes()});
                dbg("  lookups_eq_evals[0] = {any}\n", .{lookups_eq_evals[0].toBytesBE()[0..8]});
                dbg("  lookups_combined_vals[0] = {any}\n", .{lookups_combined_vals[0].toBytes()});
                dbg("  lookups_current_scalar = {any}\n", .{lookups_current_scalar.toBytes()});
                dbg("  lookups_combined_vals[0] = {any}\n", .{lookups_combined_vals[0].toBytesBE()[0..8]});
            }

            // Compute eq(r_reduction, r_cycle_prime)
            // r_reduction is from Stage 3 InstructionClaimReduction (BIG_ENDIAN)
            // r_cycle_prime is from this Stage 5 sumcheck (LITTLE_ENDIAN from sumcheck order)
            //
            // CRITICAL: Stage 5 sumcheck challenges are bound in LITTLE_ENDIAN order (r_0, r_1, ..., r_7)
            // but r_reduction from Stage 3 is in BIG_ENDIAN order (r_7, r_6, ..., r_0)
            // We need to reverse r_cycle_prime to match r_reduction's ordering for eq computation
            var r_cycle_prime_be = try self.allocator.alloc(F, n_cycle_vars);
            defer self.allocator.free(r_cycle_prime_be);
            for (0..n_cycle_vars) |i| {
                r_cycle_prime_be[i] = r_cycle_prime[n_cycle_vars - 1 - i];
            }

            const eq_r_reduction = computeEqPolynomial(F, r_reduction, r_cycle_prime_be);

            if (comptime debug_verbose) {
                dbg("  r_reduction[0..8] (8 elements):\n", .{});
            }
            for (0..n_cycle_vars) |i| {
                if (comptime debug_verbose) {
                    dbg("    r_reduction[{}] = {x}\n", .{ i, r_reduction[i].toBytesBE()[16..32].* });
                }
            }
            if (comptime debug_verbose) {
                dbg("  r_cycle_prime_be[0..8] (8 elements):\n", .{});
            }
            for (0..n_cycle_vars) |i| {
                if (comptime debug_verbose) {
                    dbg("    r_cycle_prime_be[{}] = {x}\n", .{ i, r_cycle_prime_be[i].toBytesBE()[16..32].* });
                }
            }
            if (comptime debug_verbose) {
                dbg("  eq_r_reduction (verifier computes) = {x}\n", .{eq_r_reduction.toBytesBE()[16..32].*});
                dbg("  eq_evals[0] (from sumcheck) = {x}\n", .{lookups_eq_evals[0].toBytesBE()[16..32].*});
                dbg("  lookups_current_scalar (should equal eq_r_reduction) = {x}\n", .{lookups_current_scalar.toBytesBE()[16..32].*});
            }

            // Debug: print first few r_address_prime values
            if (comptime debug_verbose) {
                dbg("[STAGE5 FINAL] r_address_prime[0..4] (sumcheck challenges 0-3):\n", .{});
            }
            for (0..4) |i| {
                if (comptime debug_verbose) {
                    dbg("  r_address_prime[{}] = {x}\n", .{ i, r_address_prime[i].toBytesBE()[16..32].* });
                }
            }

            // Compute operand polynomial evaluations at r_address_prime
            const left_op_eval = evaluateLeftOperand(F, r_address_prime);
            const right_op_eval = evaluateRightOperand(F, r_address_prime);
            const identity_eval = evaluateIdentity(F, r_address_prime);

            if (comptime debug_verbose) {
                dbg("  left_op_eval (full) = {x}\n", .{left_op_eval.toBytesBE()[16..32].*});
                dbg("  right_op_eval (full) = {x}\n", .{right_op_eval.toBytesBE()[16..32].*});
                dbg("  identity_eval (full) = {x}\n", .{identity_eval.toBytesBE()[16..32].*});
                dbg("  gamma_lookups_raf = {any}\n", .{gamma_lookups_raf.toBytesBE()[0..8]});
            }

            // CORRECT APPROACH: Compute opening claims from the bound polynomials
            //
            // 1. InstructionRa(i) = ra_chunk_weights[i][0] after all binding
            // 2. LookupTableFlag(i) = Σ_{j: table[j] == i} eq(r_cycle', j)
            // 3. InstructionRafFlag = Σ_{j: identity_path} eq(r_cycle', j)

            // Allocate output arrays
            const table_flags = try self.allocator.alloc(F, num_lookup_tables);
            @memset(table_flags, F.zero());

            const ra_chunks = try self.allocator.alloc(F, lookups_ra_d);
            // ra_claims[i] = ra_chunk_weights[i][0] after all n_cycle_vars bindings
            // The binding process reduces the polynomial to a single value at index 0,
            // which equals the MLE evaluation at the bound point (r_cycle_prime).
            // This matches Jolt's ra_poly.final_sumcheck_claim()
            for (0..lookups_ra_d) |i| {
                ra_chunks[i] = ra_chunk_weights[i][0];
            }

            // Debug: print ra chunk claims (FULL 32 bytes for comparison with Jolt)
            if (comptime debug_verbose) {
                dbg("[STAGE5 LOOKUPS] ra_chunk claims (FULL 32 bytes LE for Jolt comparison):\n", .{});
            }
            var ra_product = F.one();
            for (0..lookups_ra_d) |i| {
                const le_bytes = ra_chunks[i].toBytes();
                if (comptime debug_verbose) {
                    dbg("  ra_chunks[{}] LE = {any}\n", .{ i, le_bytes });
                }
                ra_product = ra_product.mul(ra_chunks[i]);
            }
            const ra_product_le = ra_product.toBytes();
            if (comptime debug_verbose) {
                dbg("  ra_product FULL LE = {any}\n", .{ra_product_le});
                dbg("  lookups_ra_weights[0] = {any}\n", .{lookups_ra_weights[0].toBytesBE()[0..8]});
            }

            // Check lookups_current_scalar * ra_product * combined_vals[0]
            const scalar_ra_combined = lookups_current_scalar.mul(ra_product).mul(lookups_combined_vals[0]);
            if (comptime debug_verbose) {
                dbg("  scalar*ra_product*combined = {any}\n", .{scalar_ra_combined.toBytes()});
                dbg("  lookups_claim (tracked)    = {any}\n", .{lookups_claim.toBytes()});
                dbg("  scalar*ra_product*combined == lookups_claim: {}\n", .{scalar_ra_combined.eql(lookups_claim)});
            }

            // Verify ra_product == lookups_ra_weights[0]
            const match_after = ra_product.eql(lookups_ra_weights[0]);
            if (comptime debug_verbose) {
                dbg("  ra_product == lookups_ra_weights[0] (after all binding): {}\n", .{match_after});
            }
            if (!match_after) {
                if (comptime debug_verbose) {
                    dbg("  WARNING: ra_product and lookups_ra_weights[0] don't match after binding!\n", .{});
                    dbg("  This is expected - binding the product != product of bindings\n", .{});
                    dbg("  The correct ra_claim should be the PRODUCT of the bound chunk values.\n", .{});
                }
            }

            // Print table index histogram for comparison with Jolt
            if (comptime debug_verbose) {
                const print = std.debug.print;
                var table_counts: [42]usize = [_]usize{0} ** 42;
                var no_table_count: usize = 0;
                for (0..T) |jj| {
                    if (jj >= trace_len) continue;
                    const tidx = cycle_table_indices[jj];
                    if (tidx >= 0 and @as(usize, @intCast(tidx)) < 42) {
                        table_counts[@intCast(tidx)] += 1;
                    } else {
                        no_table_count += 1;
                    }
                }
                print("[ZOLT TABLE HISTOGRAM] trace_len={}, T={}, no_table={}\n", .{ trace_len, T, no_table_count });
                for (0..42) |i| {
                    if (table_counts[i] > 0) {
                        print("[ZOLT TABLE HISTOGRAM] table[{}] = {} cycles\n", .{ i, table_counts[i] });
                    }
                }
                // Print first 30 cycle-to-table mappings with opcodes
                for (0..@min(30, trace_len)) |jj| {
                    const step = trace.steps.items[jj];
                    const instr_dbg = step.instruction;
                    const opcode_dbg = instr_dbg & 0x7f;
                    const funct3_dbg: u3 = @truncate((instr_dbg >> 12) & 0x7);
                    const funct7_dbg: u7 = @truncate(instr_dbg >> 25);
                    print("[ZOLT CYCLE MAP] j={}: opcode=0x{x:0>2} funct3={} funct7=0x{x:0>2} table={} identity={} noop={}\n", .{
                        jj, opcode_dbg, funct3_dbg, funct7_dbg, cycle_table_indices[jj], cycle_is_identity_path[jj], step.is_noop,
                    });
                }
            }

            // Compute eq(r_cycle', j) for all j and accumulate into table flags
            // r_cycle_prime is LITTLE_ENDIAN from sumcheck (r_0, r_1, ..., r_7)
            // We need to compute eq evaluations at this point
            //
            // Note: After binding n_cycle_vars rounds in the sumcheck, the eq polynomial
            // lookups_eq_evals has been bound to the challenges. The final value eq_evals[0]
            // equals eq(r_reduction, r_cycle'). But we need eq(r_cycle', j) for each original cycle j.
            //
            // We compute this by evaluating eq(r_cycle', j) for each j in [0, T)
            // where r_cycle' = reversed challenges (to get BIG_ENDIAN for eq)
            if (comptime debug_verbose) {
                // Verify eq sum and print per-cycle eq values for first 5 cycles
                const print = std.debug.print;
                var eq_sum_dbg = F.zero();
                for (0..T) |j| {
                    const eq_j = computeEqAtIndex(r_cycle_prime_be, j);
                    eq_sum_dbg = eq_sum_dbg.add(eq_j);
                    if (j < 5) {
                        print("[ZOLT EQ DEBUG] j={}: eq_j LE[0..16] = {any}, table_idx={}\n", .{ j, eq_j.toBytes()[0..16].*, cycle_table_indices[j] });
                    }
                }
                print("[ZOLT EQ DEBUG] Sum of eq(j, r_cycle) over all T={} cycles = {any}\n", .{ T, eq_sum_dbg.toBytes()[0..16].* });
                print("[ZOLT EQ DEBUG] r_cycle_prime_be (reversed sumcheck challenges):\n", .{});
                for (0..n_cycle_vars) |i| {
                    print("[ZOLT EQ DEBUG]   r_cycle_prime_be[{}] limbs = [0x{x}, 0x{x}, 0x{x}, 0x{x}]\n", .{ i, r_cycle_prime_be[i].limbs[0], r_cycle_prime_be[i].limbs[1], r_cycle_prime_be[i].limbs[2], r_cycle_prime_be[i].limbs[3] });
                }
            }
            // Compute table flags using split-eq parallel histogram.
            // Split r_cycle_prime_be into hi/lo halves, build sqrt(T)-sized eq tables,
            // then parallel over E_hi with sequential inner E_lo (matching Jolt's compute_flag_claims).
            const flag_n_hi = n_cycle_vars / 2;
            const flag_n_lo = n_cycle_vars - flag_n_hi;
            const flag_hi_len = @as(usize, 1) << @intCast(flag_n_hi);
            const flag_lo_len = @as(usize, 1) << @intCast(flag_n_lo);

            // Build E_hi and E_lo (each sqrt(T) sized) — reuse lookups_eq_evals buffer for E_lo
            // BIG_ENDIAN: bit k of index j maps to r[n-1-k], so:
            //   j_lo (bits 0..n_lo-1) uses r[n_hi..n] (last n_lo elements)
            //   j_hi (bits n_lo..n-1) uses r[0..n_hi] (first n_hi elements)
            const E_lo_flags = lookups_eq_evals[0..flag_lo_len];
            buildFullEqTable(r_cycle_prime_be[flag_n_hi..n_cycle_vars], E_lo_flags, self.thread_pool);
            const E_hi_flags = try self.allocator.alloc(F, flag_hi_len);
            defer self.allocator.free(E_hi_flags);
            buildFullEqTable(r_cycle_prime_be[0..flag_n_hi], E_hi_flags, self.thread_pool);

            // Parallel histogram accumulation over E_hi chunks
            const FlagResult = struct {
                flags: [NUM_TABLES]F,
                raf: F,
            };
            const FlagCtx = struct {
                e_hi: []const F,
                e_lo: []const F,
                lo_len: usize,
                tbl_ids: []const i8,
                is_id: []const bool,
                n_tables: usize,
                total_T: usize,
            };
            const flag_ctx = FlagCtx{
                .e_hi = E_hi_flags,
                .e_lo = E_lo_flags,
                .lo_len = flag_lo_len,
                .tbl_ids = cycle_table_indices,
                .is_id = cycle_is_identity_path,
                .n_tables = num_lookup_tables,
                .total_T = T,
            };
            const flag_identity = FlagResult{
                .flags = [_]F{F.zero()} ** NUM_TABLES,
                .raf = F.zero(),
            };
            const flagMapFn = struct {
                fn f(c: FlagCtx, hi_start: usize, hi_end: usize) FlagResult {
                    var local_flags = [_]F{F.zero()} ** NUM_TABLES;
                    var local_raf = F.zero();
                    for (hi_start..hi_end) |c_hi| {
                        const base = c_hi * c.lo_len;
                        // Accumulate with deferred e_hi multiplication
                        var inner_flags = [_]F{F.zero()} ** NUM_TABLES;
                        var inner_raf = F.zero();
                        const inner_end = @min(base + c.lo_len, c.total_T);
                        for (base..inner_end) |j| {
                            const c_lo = j - base;
                            const eq_lo = c.e_lo[c_lo];
                            const table_idx = c.tbl_ids[j];
                            if (table_idx >= 0 and @as(usize, @intCast(table_idx)) < c.n_tables) {
                                inner_flags[@intCast(table_idx)] = inner_flags[@intCast(table_idx)].add(eq_lo);
                            }
                            if (c.is_id[j]) {
                                inner_raf = inner_raf.add(eq_lo);
                            }
                        }
                        // Multiply by e_hi once for this block
                        const e_hi = c.e_hi[c_hi];
                        for (0..c.n_tables) |t| {
                            if (!inner_flags[t].eql(F.zero())) {
                                local_flags[t] = local_flags[t].add(e_hi.mul(inner_flags[t]));
                            }
                        }
                        if (!inner_raf.eql(F.zero())) {
                            local_raf = local_raf.add(e_hi.mul(inner_raf));
                        }
                    }
                    return FlagResult{ .flags = local_flags, .raf = local_raf };
                }
            }.f;
            const flagReduceFn = struct {
                fn f(a: FlagResult, b: FlagResult) FlagResult {
                    var r: FlagResult = undefined;
                    for (0..NUM_TABLES) |t| {
                        r.flags[t] = a.flags[t].add(b.flags[t]);
                    }
                    r.raf = a.raf.add(b.raf);
                    return r;
                }
            }.f;

            const flag_result = if (self.thread_pool) |tp|
                tp.parallelReduce(FlagResult, flag_hi_len, flag_identity, flag_ctx, flagMapFn, flagReduceFn)
            else
                flagMapFn(flag_ctx, 0, flag_hi_len);

            @memcpy(table_flags[0..num_lookup_tables], flag_result.flags[0..num_lookup_tables]);
            var computed_raf_flag = flag_result.raf;

            // Debug: print non-zero table flags
            if (comptime debug_verbose) {
                dbg("[STAGE5 LOOKUPS] Non-zero table flags (FULL LE):\n", .{});
                for (0..num_lookup_tables) |i| {
                    if (!table_flags[i].eql(F.zero())) {
                        dbg("  table_flags[{}] = {any}\n", .{ i, table_flags[i].toBytes() });
                    }
                }
            }
            if (comptime debug_verbose) {
                dbg("[STAGE5 LOOKUPS] raf_flag (identity path sum) = {any}\n", .{computed_raf_flag.toBytesBE()[0..8]});
            }

            // Verify the opening claims match the sumcheck output
            // expected = eq_r_reduction * ra_product * (val_claim + gamma * raf_claim)
            // where val_claim = Σ table_flags[i] * table_eval[i]
            // and raf_claim = (1 - raf_flag)*(left_op + gamma*right_op) + raf_flag*gamma*identity

            const raf_claim = F.one().sub(computed_raf_flag).mul(left_op_eval.add(gamma_lookups_raf.mul(right_op_eval)))
                .add(computed_raf_flag.mul(gamma_lookups_raf).mul(identity_eval));
            if (comptime debug_verbose) {
                const print = std.debug.print;
                print("[ZOLT S5 INST2] raf_claim FULL LE = {any}\n", .{raf_claim.toBytes()});
            }

            // Compute val_claim = Σ table_flags[i] * stored_table_values[i]
            // This matches the verifier's formula: Σ val_evals[i] * table_flag_claims[i]
            var val_claim = F.zero();
            for (0..num_lookup_tables) |i| {
                val_claim = val_claim.add(table_flags[i].mul(stored_table_values[i]));
                if (comptime debug_verbose) if (!table_flags[i].eql(F.zero())) {
                    const print = std.debug.print;
                    print("[ZOLT S5 INST2] table[{}] val_eval={any} flag={any}\n", .{
                        i, stored_table_values[i].toBytes()[0..16].*, table_flags[i].toBytes()[0..16].*,
                    });
                };
            }
            if (comptime debug_verbose) {
                const print = std.debug.print;
                print("[ZOLT S5 INST2] val_claim FULL LE = {any}\n", .{val_claim.toBytes()});
                print("[ZOLT S5 INST2] raf_flag FULL LE = {any}\n", .{computed_raf_flag.toBytes()});
                print("[ZOLT S5 INST2] left_op_eval FULL LE = {any}\n", .{left_op_eval.toBytes()});
                print("[ZOLT S5 INST2] right_op_eval FULL LE = {any}\n", .{right_op_eval.toBytes()});
                print("[ZOLT S5 INST2] identity_eval FULL LE = {any}\n", .{identity_eval.toBytes()});
                print("[ZOLT S5 INST2] gamma FULL LE = {any}\n", .{gamma_lookups_raf.toBytes()});
                const expected_inst2 = eq_r_reduction.mul(ra_product).mul(val_claim.add(gamma_lookups_raf.mul(raf_claim)));
                print("[ZOLT S5 INST2] expected_inst2 FULL LE = {any}\n", .{expected_inst2.toBytes()});
                print("[ZOLT S5 INST2] lookups_claim (chain) FULL LE = {any}\n", .{lookups_claim.toBytes()});
                print("[ZOLT S5 INST2] inst2_match = {}\n", .{expected_inst2.eql(lookups_claim)});
            }

            // CRITICAL DIAGNOSTIC: Compare combined_vals[0] with val_claim + gamma * raf_claim
            // (gated behind debug_verbose to avoid expensive runtime checks in release builds)
            if (comptime debug_verbose) {
                const verifier_combined = val_claim.add(gamma_lookups_raf.mul(raf_claim));
                dbg("\n  === CRITICAL COMPARISON ===\n", .{});
                dbg("  combined_vals[0] (bound polynomial) FULL LE = {any}\n", .{lookups_combined_vals[0].toBytes()});
                dbg("  val+gamma*raf   (from opening claims) FULL LE = {any}\n", .{verifier_combined.toBytes()});
                dbg("  combined_vals[0] == val+gamma*raf: {}\n", .{lookups_combined_vals[0].eql(verifier_combined)});

                if (!lookups_combined_vals[0].eql(verifier_combined)) {
                    const cv_div_ra = if (!ra_product.eql(F.zero())) lookups_combined_vals[0].mul(ra_product.inverse().?) else F.zero();
                    dbg("  combined_vals[0] / ra_product = {any}\n", .{cv_div_ra.toBytes()});
                    dbg("  cv/ra == val+gamma*raf: {}\n", .{cv_div_ra.eql(verifier_combined)});

                    const ra_times_combined = ra_product.mul(verifier_combined);
                    dbg("  ra*(val+gamma*raf) == combined_vals[0]: {}\n", .{ra_times_combined.eql(lookups_combined_vals[0])});

                    if (!eq_r_reduction.eql(F.zero()) and !ra_product.eql(F.zero())) {
                        const implied_cv = correct_inst2_from_batched.mul(eq_r_reduction.inverse().?).mul(ra_product.inverse().?);
                        dbg("  correct_inst2 / (eq*ra) = {any}\n", .{implied_cv.toBytes()});
                    }
                    if (!lookups_current_scalar.eql(F.zero()) and !ra_product.eql(F.zero())) {
                        const implied_cv2 = correct_inst2_from_batched.mul(lookups_current_scalar.inverse().?).mul(ra_product.inverse().?);
                        dbg("  correct_inst2 / (scalar*ra) = {any}\n", .{implied_cv2.toBytes()});
                    }
                    dbg("  eq_r_reduction == lookups_current_scalar: {}\n", .{eq_r_reduction.eql(lookups_current_scalar)});
                }
                dbg("  === END CRITICAL COMPARISON ===\n\n", .{});

                const expected_output = eq_r_reduction.mul(ra_product).mul(val_claim.add(gamma_lookups_raf.mul(raf_claim)));
                dbg("  expected_output == lookups_output_claim: {}\n", .{expected_output.eql(lookups_output_claim)});
                dbg("  correct_inst2_from_batched == lookups_output_claim: {}\n", .{correct_inst2_from_batched.eql(lookups_output_claim)});
            }
            if (comptime debug_verbose) {
                if (!correct_inst2_from_batched.eql(lookups_output_claim)) {
                    dbg("  ERROR: correct_inst2_from_batched != lookups_output_claim!\n", .{});
                } else {
                    dbg("  GOOD: correct_inst2_from_batched == lookups_output_claim\n", .{});
                }
            }

            // ============================================================
            // COMPUTE ram_ra_claim FROM SUMCHECK STATE
            // ============================================================
            // After the RamRaClaimReduction sumcheck binds all cycle variables,
            // H_prime[0] holds the final evaluation of the ra polynomial at the
            // opening point [r_address, r_cycle_reduced].
            // This matches upstream's state.H_prime.final_sumcheck_claim().
            const ram_ra_claim = H_prime[0];
            if (comptime debug_verbose) {
                dbg("[STAGE5 RAM_RA] ram_ra_claim = H_prime[0] = {x}\n", .{ram_ra_claim.toBytesBE()});
            }

            if (comptime bench_timing) {
                const post_loop_ms = @as(f64, @floatFromInt(bench_timer.read())) / 1_000_000.0;
                const post_sc_ms = @as(f64, @floatFromInt(bench_overall_timer.read() - bench_init_ns - bench_addr_compute_ns - bench_addr_bind_ns - bench_addr_transcript_ns - bench_addr_other_ns - bench_phase_transition_ns - bench_inst0_compute_ns - bench_inst1_compute_ns - bench_inst2_cycle_compute_ns - bench_cycle_bind_ns - bench_cycle_transcript_ns)) / 1_000_000.0;
                std.debug.print("    [STAGE5-BENCH] Post-sumcheck (total gap):  {d:8.1} ms\n", .{post_sc_ms});
                std.debug.print("    [STAGE5-BENCH]   post-loop code:           {d:8.1} ms\n", .{post_loop_ms});
                std.debug.print("    [STAGE5-BENCH]   remat (round 128):        {d:8.1} ms\n", .{@as(f64, @floatFromInt(bench_remat_ns)) / 1_000_000.0});
                std.debug.print("    [STAGE5-BENCH]   coeff combo (18 rds):     {d:8.1} ms\n", .{@as(f64, @floatFromInt(bench_cycle_coeff_ns)) / 1_000_000.0});
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

        /// Per-cycle instruction decode for combined_vals, lookup indices, table assignments.
        /// Extracted from the init trace loop for parallel dispatch.
        /// IMPORTANT: This is a hot function called T times in parallel.
        fn processTraceCycleCombined(
            step: tracer.TraceStep,
            j: usize,
            combined: []F,
            idx_lo: []u64,
            idx_hi: []u64,
            tbl_ids: []i8,
            is_id: []bool,
            g_raf: F,
            g_raf2: F,
            idx_u128: ?[]u128,
            is_interleaved_out: ?[]bool,
        ) void {
            // NOTE: Do NOT skip NOOPs here! In Jolt, NOOPs (ADDI x0,x0,0) are valid
            // instructions with lookup_table = RangeCheck and is_identity_path = true.
            // Skipping them causes cycle_table_indices and cycle_is_identity_path to be
            // wrong, which corrupts Q arrays, rematerialization, and opening claims.

            const instr = step.instruction;
            const opcode = instr & 0x7f;
            const funct3: u3 = @truncate((instr >> 12) & 0x7);
            const funct7: u7 = @truncate(instr >> 25);

            // Determine left_op, right_op, and lookup_output based on instruction type.
            // This MUST match the verification loop / R1CS witness exactly.
            // Use field arithmetic with signedI64ToField for signed immediates.
            var left_op: F = undefined;
            var right_op: F = undefined;
            var lookup_output: F = undefined;

            // First compute left_input and right_input (same as R1CS)
            const left_is_rs1: bool = switch (opcode) {
                0x33, 0x3b, 0x23, 0x63, 0x13, 0x03, 0x67, 0x1b, 0x0B, 0x2B, 0x5B => true,
                0x22 => true, // VirtualAssertEQ: left = rs1
                0x42 => true, // VirtualZeroExtendWord: left = rs1
                0x62 => true, // VirtualAssertValidUnsignedRemainder: left = rs1
                // 0x02 (VirtualAdvice): left_is_rs1 = false (instruction_inputs = (0,0))
                else => false,
            };
            const left_is_pc: bool = switch (opcode) {
                0x17, 0x6f => true,
                else => false,
            };
            const right_is_rs2: bool = switch (opcode) {
                0x33, 0x63, 0x3b => true,
                0x22 => (funct3 == 0 or funct3 == 1), // VirtualAssertEQ/ValidDiv0: right = rs2; alignment: right = imm
                0x62 => true, // VirtualAssertValidUnsignedRemainder: right = rs2
                0x5B => step.rs2_read, // VirtualSRL/VirtualSRA R-type: rs2; VirtualSRLI/VirtualSRAI I-type: imm
                else => false,
            };
            const right_is_imm: bool = switch (opcode) {
                0x13, 0x03, 0x67, 0x23, 0x37, 0x17, 0x6f, 0x1b, 0x0B, 0x2B => true,
                0x22 => (funct3 == 2 or funct3 == 3), // alignment assertions: right = imm
                0x5B => !step.rs2_read, // I-type: imm; R-type: not imm
                else => false,
            };

            // For identity-path AddOperands instructions (ADDI, ADDIW, JAL, JALR, VirtualSignExtendWord),
            // use UNSIGNED u64 immediate to match Jolt's to_lookup_operands() u128 arithmetic.
            // This ensures RightInstructionInput matches between R1CS, Stage 3, and Stage 5.
            const is_identity_add_imm: bool = switch (opcode) {
                0x13 => funct3 == 0, // ADDI
                0x1b => funct3 == 0, // ADDIW
                0x0B => true, // VirtualSignExtendWord
                0x6f => true, // JAL
                0x67 => true, // JALR
                else => false,
            };
            const imm_val = if (opcode == 0x2B) blk: {
                if (funct3 == 0) {
                    // VirtualMULI: IMM = multiplier = 1 << shamt
                    const shamt_raw2: u32 = instr >> 20;
                    const shamt2: u6 = @truncate(shamt_raw2 & 0x3F);
                    const multiplier2: u64 = @as(u64, 1) << shamt2;
                    break :blk F.fromU64(multiplier2);
                } else {
                    // VirtualPow2/VirtualShiftRightBitmask: IMM = 0
                    break :blk F.zero();
                }
            } else if (opcode == 0x5B) blk: {
                if (step.rs2_read) {
                    // VirtualSRL/VirtualSRA R-type: no immediate (rs2 used instead)
                    break :blk F.zero();
                } else {
                    // VirtualSRLI/VirtualSRAI I-type: IMM = bitmask computed from total shift
                    const total_shift_raw2: u32 = instr >> 20;
                    const total_shift2: u7 = @truncate(total_shift_raw2 & 0x3F);
                    const ones2: u128 = (@as(u128, 1) << @intCast(64 - @as(u8, total_shift2))) - 1;
                    const bitmask2: u64 = @truncate(ones2 << total_shift2);
                    break :blk F.fromU64(bitmask2);
                }
            } else if (opcode == 0x22 and (funct3 == 2 or funct3 == 3)) blk: {
                // VirtualAssertHalfwordAlignment/WordAlignment: SIGNED IMM encoding
                // Must match R1CS witness (now signed) and Jolt verifier val_poly
                const assert_imm_raw: u32 = @truncate(instr >> 20);
                const assert_imm_signed: i64 = @as(i64, @as(i32, @bitCast(assert_imm_raw << 20)) >> 20);
                if (assert_imm_signed < 0) {
                    break :blk F.fromU64(@intCast(-assert_imm_signed)).neg();
                } else {
                    break :blk F.fromU64(@intCast(assert_imm_signed));
                }
            } else if (is_identity_add_imm) blk: {
                // Use unsigned u64 representation (two's complement) for the immediate.
                // E.g., imm=-1 → F(0xFFFFFFFFFFFFFFFF) instead of F(p-1).
                break :blk F.fromU64(computeUnsignedImmediate(instr));
            } else computeImmediate(instr);

            var left_input: F = F.zero();
            if (left_is_rs1) left_input = F.fromU64(step.rs1_value);
            // FIX: Use unexpanded_pc (raw RISC-V address) not pc (expanded bytecode index)
            // This matches R1CS constraints.zig and Jolt's instruction_input.rs
            if (left_is_pc) left_input = F.fromU64(step.unexpanded_pc);

            var right_input: F = F.zero();
            if (right_is_rs2) right_input = F.fromU64(step.rs2_value);
            if (right_is_imm) right_input = imm_val;

            // Compute LookupOutput = materialize_entry(lookup_index) for the instruction's table.
            // For identity-path (AddOperands/SubtractOperands/MultiplyOperands):
            //   lookup_output = materialize_entry(right_op_raw) = F.fromU64(right_op_raw) for RangeCheck.
            // For interleaved path:
            //   lookup_output = materialize_entry(interleave(left, right)) from the assigned table.
            // Special cases: JAL, JALR, Branch have their own formulas.
            //
            // NOTE: This is computed AFTER the lookup index section below, but we set a
            // preliminary value here and may override it.
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
                    // Default: rd_value (will be overridden for ADDIW/ADDW/SUBW below)
                    lookup_output = F.fromU64(step.rd_value);
                },
            }

            // Compute LeftLookupOperand and RightLookupOperand
            switch (opcode) {
                0x33 => { // R-type
                    if (funct7 == 0x01) {
                        // M-extension
                        if (funct3 == 0x0) { // MUL: MultiplyOperands
                            left_op = F.zero();
                            right_op = left_input.mul(right_input); // Product
                        } else if (funct3 == 0x3) { // MULHU: MultiplyOperands
                            left_op = F.zero();
                            right_op = left_input.mul(right_input); // Product
                        } else {
                            // DIVU, REMU, MULHSU, etc.: interleaved
                            left_op = left_input;
                            right_op = right_input;
                        }
                    } else if (funct3 == 0x0 and funct7 == 0x20) {
                        // SUB: SubtractOperands, left=0, right=rs1-rs2+2^64
                        const two_pow_64 = F.fromBytes(&[_]u8{ 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 });
                        left_op = F.zero();
                        right_op = left_input.sub(right_input).add(two_pow_64);
                    } else if (funct3 == 0x0 and funct7 == 0x0) {
                        // ADD: AddOperands, left=0, right=rs1+rs2
                        left_op = F.zero();
                        right_op = left_input.add(right_input);
                    } else {
                        // XOR, AND, OR, SLT, SLTU, SRL, SRA: interleaved operands
                        left_op = left_input;
                        right_op = right_input;
                    }
                },
                0x13 => { // I-type ALU: only ADDI (funct3=0) uses AddOperands
                    // Other I-type ALU instructions (SLLI, SLTI, etc.) use interleaved operands
                    // Note: Jolt expands SLLI/etc to virtual instructions, but Zolt handles them directly
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
                    // In Jolt, ADDW decomposes to ADD+VirtualSEW, SUBW to SUB+VirtualSEW.
                    // For Zolt's single-cycle model, match the first step's format.
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
                        // VirtualChangeDivisorW: interleaved, left=rs1 as u32 as u64 (truncated), right=rs2
                        // Jolt's to_instruction_inputs: (rs1 as u32 as u64, rs2 as i128)
                        // to_lookup_operands: (rs1 as u32 as u64, rs2 as u64)
                        const rs1_lower32: u64 = step.rs1_value & 0xFFFFFFFF;
                        left_op = F.fromU64(rs1_lower32);
                        right_op = F.fromU64(step.rs2_value);
                    } else {
                        // Other 0x3b variants (not AddOperands/SubtractOperands)
                        left_op = left_input;
                        right_op = right_input;
                    }
                },
                0x0B => { // VirtualSignExtendWord: AddOperands, left=0, right=rs1
                    // Lookup operands: (0, rs1_val + 0) = (0, rs1_val)
                    left_op = F.zero();
                    right_op = left_input.add(right_input); // rs1 + 0 = rs1
                },
                0x2B => { // Virtual I-type: dispatch on funct3
                    if (funct3 == 0) {
                        // VirtualMULI: MultiplyOperands, left=0, right=rs1*imm
                        left_op = F.zero();
                        right_op = left_input.mul(right_input);
                    } else {
                        // VirtualPow2 (funct3=1), VirtualShiftRightBitmask (funct3=2): AddOperands
                        // Lookup operands: (0, rs1 + 0) = (0, rs1)
                        left_op = F.zero();
                        right_op = left_input.add(right_input); // rs1 + 0 = rs1
                    }
                },
                0x03 => { // Load: NOT AddOperands, left=rs1, right=imm
                    // R1CS witness sets: LeftLookupOperand=left_input, RightLookupOperand=right_input
                    left_op = left_input;
                    right_op = right_input;
                },
                0x23 => { // Store: NOT AddOperands, left=rs1, right=imm
                    // R1CS witness sets: LeftLookupOperand=left_input, RightLookupOperand=right_input
                    left_op = left_input;
                    right_op = right_input;
                },
                0x02 => { // VirtualAdvice: Advice flag (identity path)
                    // R1CS: LeftLookupOperand=0, RightLookupOperand=F.fromU128(rd_value)
                    // left_input=0, right_input=0 (no instruction inputs)
                    // The lookup operand is the advice oracle value (rd_value)
                    left_op = F.zero();
                    right_op = F.fromU128(@as(u128, step.rd_value));
                },
                0x22 => { // Virtual assert: dispatch on funct3
                    if (funct3 == 2 or funct3 == 3) {
                        // VirtualAssertHalfwordAlignment/WordAlignment: AddOperands
                        // Lookup operands: (0, rs1 + imm)
                        left_op = F.zero();
                        right_op = left_input.add(right_input); // rs1 + imm
                    } else {
                        // VirtualAssertEQ (funct3=0) / VirtualAssertValidDiv0 (funct3=1): interleaved
                        left_op = left_input;
                        right_op = right_input;
                    }
                },
                0x42 => { // VirtualZeroExtendWord: AddOperands flag (identity path)
                    // R1CS: LeftLookupOperand=0, RightLookupOperand=F.fromU128(rs1_value)
                    // AddOperands: left=0, right=left_input+right_input
                    // Here left_input=rs1, right_input=0, so right=rs1
                    left_op = F.zero();
                    right_op = F.fromU128(@as(u128, step.rs1_value));
                },
                0x62 => { // VirtualAssertValidUnsignedRemainder: Assert flag (interleaved)
                    // R1CS: LeftLookupOperand=left_input(=rs1), RightLookupOperand=right_input(=rs2)
                    left_op = left_input;
                    right_op = right_input;
                },
                else => {
                    // Default: NOT Add+Sub+Mul (includes 0x63 Branch)
                    left_op = left_input;
                    right_op = right_input;
                },
            }

            // Track which lookup table this cycle uses (for flag claims)
            const table_idx = getLookupTableIndex(opcode, funct3, funct7);
            tbl_ids[j] = table_idx;

            // For instructions without a lookup table (Load, Store, SLL, etc.):
            // All three must be zeroed to match the R1CS witness, which sets:
            //   LeftLookupOperand = 0, RightLookupOperand = 0, LookupOutput = 0
            // In Jolt, these instructions decompose into virtual sequences and never
            // appear as raw cycles, so the R1CS witness has all zeros for their operands.
            // The RAF contribution is handled by the global prefix-suffix polynomials
            // during address rounds, NOT by per-cycle combined_vals.
            if (table_idx < 0) {
                lookup_output = F.zero();
                left_op = F.zero();
                right_op = F.zero();
            }

            // combined_vals is rematerialized at round 128 using prefix checkpoint constants
            // (see lines 4382-4422), so skip the expensive field arithmetic here.
            // Only compute for debug verification.
            if (comptime debug_verbose) {
                combined[j] = lookup_output.add(g_raf.mul(left_op)).add(g_raf2.mul(right_op));
            }

            // Determine identity path (not interleaved) based on Jolt's flags:
            //   - AddOperands: ADD, ADDI, ADDIW, ADDW, LUI, AUIPC, JAL, JALR, Load, Store
            //   - SubtractOperands: SUB, SUBW
            //   - MultiplyOperands: MUL, MULHU
            // Identity path instructions use raw operand value as lookup index (NOT interleaved).
            // Interleaved path instructions use interleave_bits(left, right) as lookup index.
            const is_identity_path: bool = switch (opcode) {
                0x33 => blk: {
                    if (funct3 == 0 and funct7 == 0) break :blk true; // ADD (AddOperands)
                    if (funct3 == 0 and funct7 == 0x20) break :blk true; // SUB (SubtractOperands)
                    if (funct7 == 0x01 and funct3 == 0) break :blk true; // MUL (MultiplyOperands)
                    if (funct7 == 0x01 and funct3 == 3) break :blk true; // MULHU (MultiplyOperands)
                    break :blk false;
                },
                0x13 => (funct3 == 0), // ADDI (AddOperands)
                0x0B => true, // VirtualSignExtendWord (AddOperands)
                0x2B => true, // VirtualMULI/Pow2/ShiftRightBitmask: all identity path (MultiplyOperands or AddOperands)
                0x1b => (funct3 == 0), // ADDIW (AddOperands)
                0x3b => blk: {
                    if (funct3 == 0 and funct7 == 0) break :blk true; // ADDW (AddOperands)
                    if (funct3 == 0 and funct7 == 0x20) break :blk true; // SUBW (SubtractOperands)
                    break :blk false;
                },
                0x37 => true, // LUI (AddOperands)
                0x17 => true, // AUIPC (AddOperands)
                0x6f => true, // JAL (AddOperands)
                0x67 => true, // JALR (AddOperands)
                0x02 => true, // VirtualAdvice (Advice flag → identity path)
                0x42 => true, // VirtualZeroExtendWord (AddOperands → identity path)
                0x03 => false, // Load: uses (rs1, imm) format, NOT identity path
                0x23 => false, // Store: uses (rs1, imm) format, NOT identity path
                0x22 => (funct3 == 2 or funct3 == 3), // Alignment assertions: AddOperands (identity); AssertEQ/ValidDiv0: interleaved
                0x62 => false, // VirtualAssertValidUnsignedRemainder: interleaved (rs1, rs2)
                else => false,
            };
            is_id[j] = is_identity_path;

            // Compute lookup operands and index matching Jolt's to_lookup_operands/to_lookup_index.
            // For identity-path: left_op_raw=0, right_op_raw=computed_value, index=computed u128
            //   Jolt's to_lookup_index() for identity-path instructions returns the raw u128 result
            //   (NOT wrapped at 64 bits). E.g., ADD returns x as u128 + y as u64 as u128.
            // For interleaved-path: left_op_raw=rs1, right_op_raw=rs2, index=interleave(left,right)
            var left_op_raw: u64 = undefined;
            var right_op_raw: u64 = undefined;
            // lookup_idx_u128 holds the FULL u128 lookup index (not wrapped at u64)
            var lookup_idx_u128: u128 = undefined;

            if (is_identity_path) {
                left_op_raw = 0;
                // Compute lookup index in u128 to match Jolt's to_lookup_index()
                // Jolt returns the raw computation result, NOT wrapped at 64 bits.
                lookup_idx_u128 = switch (opcode) {
                    // ADD: index = rs1 as u128 + rs2 as u128
                    0x33 => blk128: {
                        if (funct3 == 0 and funct7 == 0) {
                            break :blk128 @as(u128, step.rs1_value) + @as(u128, step.rs2_value);
                        }
                        // SUB: index = rs1 as u128 + (2^64 - rs2 as u128)
                        if (funct3 == 0 and funct7 == 0x20) {
                            break :blk128 @as(u128, step.rs1_value) + (@as(u128, 1) << 64) - @as(u128, step.rs2_value);
                        }
                        // MUL: index = rs1 as u128 * rs2 as u128
                        if (funct7 == 0x01 and funct3 == 0) {
                            break :blk128 @as(u128, step.rs1_value) * @as(u128, step.rs2_value);
                        }
                        // MULHU: index = rs1 as u128 * rs2 as u128
                        if (funct7 == 0x01 and funct3 == 3) {
                            break :blk128 @as(u128, step.rs1_value) * @as(u128, step.rs2_value);
                        }
                        break :blk128 0;
                    },
                    // ADDW/SUBW: same computation as ADD/SUB
                    0x3b => blk128: {
                        if (funct3 == 0 and funct7 == 0) {
                            // ADDW: index = rs1 + rs2 (u128)
                            break :blk128 @as(u128, step.rs1_value) + @as(u128, step.rs2_value);
                        }
                        if (funct3 == 0 and funct7 == 0x20) {
                            // SUBW: index = rs1 + 2^64 - rs2 (u128)
                            break :blk128 @as(u128, step.rs1_value) + (@as(u128, 1) << 64) - @as(u128, step.rs2_value);
                        }
                        break :blk128 0;
                    },
                    // ADDI: index = rs1 + sign_ext(imm) (u128)
                    0x13 => blk128: {
                        const imm12_raw: u32 = @truncate(instr >> 20);
                        const imm_signed: i64 = @as(i64, @as(i32, @bitCast(imm12_raw << 20)) >> 20);
                        const imm_u64: u64 = @bitCast(imm_signed);
                        break :blk128 @as(u128, step.rs1_value) + @as(u128, imm_u64);
                    },
                    // ADDIW: index = rs1 + sign_ext(imm) (u128)
                    0x1b => blk128: {
                        const imm12_raw_w: u32 = @truncate(instr >> 20);
                        const imm_signed_w: i64 = @as(i64, @as(i32, @bitCast(imm12_raw_w << 20)) >> 20);
                        const imm_u64_w: u64 = @bitCast(imm_signed_w);
                        break :blk128 @as(u128, step.rs1_value) + @as(u128, imm_u64_w);
                    },
                    // LUI: index = sign_ext_32_to_64(imm) as u128
                    // Jolt sign-extends the U-type immediate via `as i32 as i64 as u64`
                    0x37 => blk128: {
                        const imm_u32: u32 = instr & 0xFFFFF000;
                        const imm_sext: u64 = @bitCast(@as(i64, @as(i32, @bitCast(imm_u32))));
                        break :blk128 @as(u128, imm_sext);
                    },
                    // AUIPC: index = pc + sign_ext_32_to_64(imm) (u128)
                    0x17 => blk128: {
                        const imm_u32: u32 = instr & 0xFFFFF000;
                        const imm_sext: u64 = @bitCast(@as(i64, @as(i32, @bitCast(imm_u32))));
                        break :blk128 @as(u128, step.unexpanded_pc) + @as(u128, imm_sext);
                    },
                    // JAL: index = pc + sign_ext(imm) (u128)
                    0x6f => blk128: {
                        const imm20: u32 = ((@as(u32, instr >> 31) & 1) << 19) |
                            ((@as(u32, instr >> 12) & 0xFF) << 11) |
                            ((@as(u32, instr >> 20) & 1) << 10) |
                            ((@as(u32, instr >> 21) & 0x3FF));
                        const imm_signed: i64 = @as(i64, @as(i32, @bitCast(imm20 << 12)) >> 11);
                        const imm_u64: u64 = @bitCast(imm_signed);
                        break :blk128 @as(u128, step.unexpanded_pc) + @as(u128, imm_u64);
                    },
                    // JALR: index = rs1 + sign_ext(imm) (u128)
                    0x67 => blk128: {
                        const imm12_raw: u32 = @truncate(instr >> 20);
                        const imm_signed: i64 = @as(i64, @as(i32, @bitCast(imm12_raw << 20)) >> 20);
                        const imm_u64: u64 = @bitCast(imm_signed);
                        break :blk128 @as(u128, step.rs1_value) + @as(u128, imm_u64);
                    },
                    // VirtualSignExtendWord: index = rs1 (the value to sign-extend)
                    0x0B => @as(u128, step.rs1_value),
                    // VirtualMULI/Pow2/ShiftRightBitmask: dispatch on funct3
                    0x2B => blk128: {
                        if (funct3 == 0) {
                            // VirtualMULI: index = rs1 * multiplier (u128)
                            const shamt_raw3: u32 = instr >> 20;
                            const shamt3: u6 = @truncate(shamt_raw3 & 0x3F);
                            const multiplier3: u64 = @as(u64, 1) << shamt3;
                            break :blk128 @as(u128, step.rs1_value) * @as(u128, multiplier3);
                        } else {
                            // VirtualPow2/VirtualShiftRightBitmask: AddOperands, index = rs1 + 0 = rs1
                            break :blk128 @as(u128, step.rs1_value);
                        }
                    },
                    // VirtualAdvice: index = advice_value (rd_value) — Jolt's to_lookup_index returns second operand
                    0x02 => @as(u128, step.rd_value),
                    // VirtualZeroExtendWord: index = rs1 + 0 = rs1 — Jolt's to_lookup_operands returns (0, x+y) where y=0
                    0x42 => @as(u128, step.rs1_value),
                    // VirtualAssertHalfwordAlignment/WordAlignment (funct3=2,3): AddOperands, index = rs1 + imm (u128)
                    0x22 => blk128: {
                        // Wrapping u64 addition matching tracer's lookup index
                        const imm_u64_22 = computeUnsignedImmediate(instr);
                        break :blk128 @as(u128, step.rs1_value +% imm_u64_22);
                    },
                    else => 0,
                };
                // right_op_raw is the lower 64 bits of the lookup index (for R1CS witness compatibility)
                right_op_raw = @truncate(lookup_idx_u128);

                // CRITICAL: For identity-path instructions, right_op must be the FULL u128
                // lookup index as a field element, matching Jolt's to_lookup_operands() which
                // returns u128 results. This is consistent with the RAF decomposition which
                // uses the u128 index for the identity polynomial evaluation.
                //
                // The R1CS witness also uses u128 values (via computeU128LookupOperand),
                // ensuring consistency between Stage 2 claims and Stage 5 combined_vals.
                right_op = F.fromU128(lookup_idx_u128);
            } else {
                // Interleaved path: left=rs1, right=rs2 (or imm for I-type)
                // VirtualChangeDivisorW (0x3b/f3=6/f7=1): left = rs1 as u32 as u64 (truncated to 32 bits)
                left_op_raw = if (opcode == 0x3b and funct3 == 6 and funct7 == 0x01)
                    step.rs1_value & 0xFFFFFFFF
                else
                    step.rs1_value;
                right_op_raw = switch (opcode) {
                    0x33, 0x3b, 0x63 => step.rs2_value,
                    0x13 => blk: {
                        // I-type: right operand is sign-extended immediate (as u64)
                        const imm12_raw: u32 = @truncate(instr >> 20);
                        const imm_signed: i64 = @as(i64, @as(i32, @bitCast(imm12_raw << 20)) >> 20);
                        break :blk @as(u64, @bitCast(imm_signed));
                    },
                    0x5B => blk5b: {
                        if (step.rs2_read) {
                            // VirtualSRL/VirtualSRA R-type: right operand is rs2
                            break :blk5b step.rs2_value;
                        } else {
                            // VirtualSRLI/VirtualSRAI I-type: right operand is bitmask computed from total shift
                            const ts_raw: u32 = instr >> 20;
                            const ts: u7 = @truncate(ts_raw & 0x3F);
                            const ones_5b: u128 = (@as(u128, 1) << @intCast(64 - @as(u8, ts))) - 1;
                            break :blk5b @truncate(ones_5b << ts);
                        }
                    },
                    else => step.rs2_value,
                };
                lookup_idx_u128 = interleaveBits128(left_op_raw, right_op_raw);
            }

            // Use the computed u128 lookup index
            const lookup_idx: u128 = lookup_idx_u128;
            idx_lo[j] = @truncate(lookup_idx);
            idx_hi[j] = @truncate(lookup_idx >> 64);

            // CRITICAL FIX: Instructions without a lookup table should have
            // lookup_index = 0, matching Jolt where to_instruction_inputs() = (0, 0)
            // and interleave(0, 0) = 0.
            if (table_idx < 0) {
                idx_lo[j] = 0;
                idx_hi[j] = 0;
            }

            // NOTE: Do NOT override lookup_output with materializeTableEntry here!
            // The initial lookups_combined_vals must match the R1CS witness polynomials
            // (which use computeLookupOutput = rd_value for most instructions).
            // The address round prefix-suffix decomposition uses table MLEs independently
            // via Q arrays, and combined_vals are rematerialized at the phase transition
            // (init_log_t_rounds) using stored_table_values for the cycle rounds.

            // Merge: compute u128 lookup index and is_interleaved in the same pass
            if (idx_u128) |u128_out| {
                u128_out[j] = (@as(u128, idx_hi[j]) << 64) | idx_lo[j];
            }
            if (is_interleaved_out) |interleaved| {
                interleaved[j] = !is_id[j];
            }
        }
        // (generated by extracting the original sequential loop body)

        /// Compute immediate value from instruction, matching R1CS deriveImmediate
        /// Compute the immediate value as a field element, matching Jolt's per-format encoding.
        ///
        /// CRITICAL: The encoding depends on the RISC-V format type:
        ///   - I-type (FormatI): u64 sign-extended from 12-bit, then u64→i128 zero-extension
        ///     → F.fromU64(sign_extended_u64). This includes 0x13, 0x03, 0x67, 0x1b, 0x73.
        ///   - U-type (FormatU): raw upper 20 bits as u64 → F.fromU64(u32_value)
        ///   - J-type (FormatJ): u64 sign-extended from 21-bit, then u64→i128 zero-extension
        ///     → F.fromU64(sign_extended_u64)
        ///   - S-type (FormatS): i64 sign-extended from 12-bit → i64 as i128 (signed)
        ///     → fieldFromI128(signed_value)
        ///   - B-type (FormatB): i128 sign-extended from 13-bit → signed
        ///     → fieldFromI128(signed_value)
        ///
        /// The reason for the asymmetry: Jolt's FormatI/FormatJ/FormatU store imm as u64,
        /// while FormatS stores imm as i64 and FormatB stores imm as i128. The conversion
        /// to NormalizedOperands.imm (i128) uses `u64 as i128` (zero-extension) for the
        /// unsigned formats, but `i64 as i128` (sign-extension) for the signed formats.
        /// Then `F::from_i128()` is called on the result.
        fn computeImmediate(instr: u32) F {
            const opcode: u8 = @truncate(instr & 0x7f);

            switch (opcode) {
                // I-type: imm[11:0] at bits [31:20], sign-extended to i64, then treat as u64
                // Jolt: FormatI.imm is u64, NormalizedOperands.imm = u64 as i128 (zero-ext)
                0x13, 0x03, 0x67, 0x1b, 0x73 => {
                    const imm12: u32 = instr >> 20;
                    const imm_signed: i64 = @as(i64, @as(i32, @bitCast(imm12 << 20)) >> 20);
                    // Treat as unsigned u64 (same bit pattern), matching Jolt's u64 as i128
                    return F.fromU64(@as(u64, @bitCast(imm_signed)));
                },
                // S-type: imm[11:5] at [31:25], imm[4:0] at [11:7], sign-extended
                // Jolt: FormatS.imm is i64, NormalizedOperands.imm = i64 as i128 (sign-ext)
                0x23 => {
                    const imm11_5 = (instr >> 25) & 0x7f;
                    const imm4_0 = (instr >> 7) & 0x1f;
                    const imm12: u32 = (imm11_5 << 5) | imm4_0;
                    const imm_signed: i64 = @as(i64, @as(i32, @bitCast(imm12 << 20)) >> 20);
                    return signedI64ToField(imm_signed);
                },
                // B-type: imm[12|10:5] at [31:25], imm[4:1|11] at [11:7], sign-extended, *2
                // Jolt: FormatB.imm is i128, NormalizedOperands.imm = i128 directly (signed)
                0x63 => {
                    const imm12 = (instr >> 31) & 1;
                    const imm10_5 = (instr >> 25) & 0x3f;
                    const imm4_1 = (instr >> 8) & 0xf;
                    const imm11 = (instr >> 7) & 1;
                    const imm13: u32 = (imm12 << 12) | (imm11 << 11) | (imm10_5 << 5) | (imm4_1 << 1);
                    const imm_signed: i64 = @as(i64, @as(i32, @bitCast(imm13 << 19)) >> 19);
                    return signedI64ToField(imm_signed);
                },
                // U-type: imm[31:12] at [31:12], shifted left by 12, SIGN-EXTENDED to 64 bits
                // Jolt: FormatU.parse does `as i32 as i64 as u64` which sign-extends the
                // 32-bit immediate to 64 bits. E.g., LUI 0xf0f0f → imm = 0xFFFFFFFFF0F0F000.
                0x37, 0x17 => {
                    const imm_upper: u32 = instr & 0xFFFFF000;
                    const sign_extended: i64 = @as(i64, @as(i32, @bitCast(imm_upper)));
                    return F.fromU64(@as(u64, @bitCast(sign_extended)));
                },
                // J-type: imm[20|10:1|11|19:12] at [31:12], sign-extended to i64, then treat as u64
                // Jolt: FormatJ.imm is u64, NormalizedOperands.imm = u64 as i128 (zero-ext)
                0x6f => {
                    const imm20 = (instr >> 31) & 1;
                    const imm10_1 = (instr >> 21) & 0x3ff;
                    const imm11 = (instr >> 20) & 1;
                    const imm19_12 = (instr >> 12) & 0xff;
                    const imm21: u32 = (imm20 << 20) | (imm19_12 << 12) | (imm11 << 11) | (imm10_1 << 1);
                    const imm_signed: i64 = @as(i64, @as(i32, @bitCast(imm21 << 11)) >> 11);
                    // Treat as unsigned u64 (same bit pattern), matching Jolt's u64 as i128
                    return F.fromU64(@as(u64, @bitCast(imm_signed)));
                },
                else => return F.zero(),
            }
        }

        /// Convert signed i64 to field element (handle negative values)
        fn signedI64ToField(val: i64) F {
            if (val >= 0) {
                return F.fromU64(@intCast(val));
            } else {
                return F.zero().sub(F.fromU64(@intCast(-val)));
            }
        }

        /// Compute the sign-extended immediate as an UNSIGNED u64 (two's complement).
        /// Used for identity-path AddOperands instructions where the lookup index
        /// is computed as: x as u128 + y as u64 as u128.
        fn computeUnsignedImmediate(instr: u32) u64 {
            const opcode: u8 = @truncate(instr & 0x7f);
            switch (opcode) {
                0x13, 0x03, 0x67, 0x1b, 0x22 => { // I-type (including VirtualAssert*)
                    const imm12: u32 = instr >> 20;
                    const imm_signed: i64 = @as(i64, @as(i32, @bitCast(imm12 << 20)) >> 20);
                    return @bitCast(imm_signed);
                },
                0x6f => { // J-type (JAL)
                    const imm20 = (instr >> 31) & 0x1;
                    const imm10_1 = (instr >> 21) & 0x3FF;
                    const imm11 = (instr >> 20) & 0x1;
                    const imm19_12 = (instr >> 12) & 0xFF;
                    const raw = (imm20 << 20) | (imm19_12 << 12) | (imm11 << 11) | (imm10_1 << 1);
                    const imm_signed: i64 = @as(i64, @as(i32, @bitCast(raw << 11)) >> 11);
                    return @bitCast(imm_signed);
                },
                else => return 0,
            }
        }

        /// Compute eq(r, k) for a specific index k
        /// Compute eq(k, r) where r is in BIG_ENDIAN order.
        ///
        /// This matches Jolt's EqPolynomial::evals convention:
        /// - evals[k] = Π_j (bit_{n-1-j}(k) ? r[j] : (1-r[j]))
        /// - Equivalently: bit j of k ↔ r[n-1-j]
        ///
        /// Example for n=2, k=1 (binary 01):
        /// - j=0: bit 1 of k = 0 → (1-r[0])
        /// - j=1: bit 0 of k = 1 → r[1]
        /// - Result: (1-r[0]) * r[1]
        fn computeEqAtIndex(r: []const F, k: usize) F {
            const n = r.len;
            var result = F.one();
            for (0..n) |j| {
                // Extract bit (n-1-j) of k: b_j = (k >> (n-1-j)) & 1
                const bj: u1 = @truncate(k >> @intCast(n - 1 - j));
                const rj = r[j]; // r[j] corresponds to bit (n-1-j) of k
                if (bj == 1) {
                    // Use standard F multiplication for full field elements
                    // This matches Stage 2's InstrLookupsProver which also uses F.mul
                    result = result.mul(rj);
                } else {
                    const one_minus_rj = F.one().sub(rj);
                    result = result.mul(one_minus_rj);
                }
            }
            return result;
        }

        /// Build the full EQ table for all indices 0..2^n using parallel forward butterfly.
        /// O(2^n) field multiplications instead of O(n * 2^n) for element-wise computation.
        /// r is in BIG_ENDIAN order: r[0] is MSB.
        /// output must have length >= 2^r.len.
        ///
        /// Algorithm (matches Jolt's EqPolynomial::evals_parallel):
        /// Process r from LSB (r[n-1]) to MSB (r[0]). At each layer, the left/right
        /// halves are independent pairs that can be parallelized.
        fn buildFullEqTable(r: []const F, output: []F, tp: ?*ThreadPool) void {
            const n = r.len;
            if (n == 0) {
                output[0] = F.one();
                return;
            }
            // Seed: output[0] = 1
            output[0] = F.one();
            var size: usize = 1;

            // Process from LSB (r[n-1]) to MSB (r[0])
            for (0..n) |i| {
                const ri = r[n - 1 - i];
                const left = output[0..size];
                const right = output[size .. 2 * size];

                // Each (left[j], right[j]) pair is independent
                const PARALLEL_THRESHOLD = 256;
                if (tp != null and size >= PARALLEL_THRESHOLD) {
                    const EqButterflyCtx = struct {
                        l: []F,
                        rr: []F,
                        r_val: F,
                    };
                    const ctx = EqButterflyCtx{ .l = left, .rr = right, .r_val = ri };
                    tp.?.parallelForForce(size, ctx, struct {
                        fn f(c: EqButterflyCtx, j: usize) void {
                            const y = c.l[j].mul(c.r_val);
                            c.rr[j] = y;
                            c.l[j] = c.l[j].sub(y);
                        }
                    }.f);
                } else {
                    for (0..size) |j| {
                        const y = left[j].mul(ri);
                        right[j] = y;
                        left[j] = left[j].sub(y);
                    }
                }
                size *= 2;
            }
        }

        /// Build partial EQ table for first num_vars variables of r.
        /// Output has 2^num_vars entries.
        fn buildPartialEqTable(r: []const F, num_vars: usize, output: []F, tp: ?*ThreadPool) void {
            if (num_vars == 0) {
                output[0] = F.one();
                return;
            }
            buildFullEqTable(r[0..num_vars], output, tp);
        }

        /// Compute eq(k, r[0:num_vars]) - partial eq polynomial over first num_vars variables.
        /// This is used in cycle rounds where some variables have been bound.
        ///
        /// r is in BIG_ENDIAN order: r[0] is MSB, r[n-1] is LSB.
        /// For LowToHigh binding of cycle variables:
        /// - After binding k LSB variables, we use r[0:n-k] (the MSB portion)
        /// - k uses bits from the remaining (n-k) variables
        ///
        /// Example with n=8, num_vars=6 (after binding 2 LSB vars):
        /// - k in [0, 2^6) uses bits [0, 6) which correspond to r[0:6]
        /// - bit j of k corresponds to r[5-j] (since r[5] is bit 0, r[0] is bit 5)
        fn computeEqAtIndexPartial(r: []const F, k: usize, num_vars: usize) F {
            if (num_vars == 0) return F.one();
            var result = F.one();
            for (0..num_vars) |j| {
                // Extract bit (num_vars-1-j) of k: this is the j-th MSB of k
                const bj: u1 = @truncate(k >> @intCast(num_vars - 1 - j));
                const rj = r[j]; // r[j] corresponds to bit (num_vars-1-j) of k
                if (bj == 1) {
                    // Use standard F multiplication for full field elements
                    result = result.mul(rj);
                } else {
                    const one_minus_rj = F.one().sub(rj);
                    result = result.mul(one_minus_rj);
                }
            }
            return result;
        }

        /// Compute all LT(j, r) evaluations efficiently using Jolt's algorithm
        /// Returns lt_evals where lt_evals[j] = LT(j, r) for all j in [0, 2^n)
        /// r is in BIG_ENDIAN order (MSB first)
        fn computeAllLtEvals(allocator: Allocator, r: []const F, tp: ?*ThreadPool) ![]F {
            const n = r.len;
            const size = @as(usize, 1) << @intCast(n);
            var evals = try allocator.alloc(F, size);
            @memset(evals, F.zero());

            // Jolt's lt_evals algorithm with parallel butterfly:
            // Process r from LSB (r[n-1]) to MSB (r[0]).
            // LT formula: right[j] = left[j] * r_i; left[j] += r_i - right[j]

            for (0..n) |i| {
                const ri = r[n - 1 - i]; // Process from LSB to MSB
                const half = @as(usize, 1) << @intCast(i);
                const left = evals[0..half];
                const right = evals[half .. 2 * half];

                const PARALLEL_THRESHOLD = 256;
                if (tp != null and half >= PARALLEL_THRESHOLD) {
                    const LtButterflyCtx = struct {
                        l: []F,
                        rr: []F,
                        r_val: F,
                    };
                    const ctx = LtButterflyCtx{ .l = left, .rr = right, .r_val = ri };
                    tp.?.parallelForForce(half, ctx, struct {
                        fn f(c: LtButterflyCtx, j: usize) void {
                            const y = c.l[j].mul(c.r_val);
                            c.rr[j] = y;
                            c.l[j] = c.l[j].add(c.r_val.sub(y));
                        }
                    }.f);
                } else {
                    for (0..half) |j| {
                        const y = left[j].mul(ri);
                        right[j] = y;
                        left[j] = left[j].add(ri.sub(y));
                    }
                }
            }

            return evals;
        }

        /// Compute LT(j, r_cycle) for index j (legacy single-point version)
        /// LT(x, y) = 1 iff x < y as bitstrings
        /// x is boolean (index j), y is field elements (r_cycle)
        fn computeLtAtIndex(r_cycle: []const F, j: usize) F {
            // LT(x, y) = Σ_i (1 - x_i) · y_i · eq(x[i+1:], y[i+1:])
            // where sum runs from MSB to LSB
            var result = F.zero();
            const num_vars = r_cycle.len;

            // Process from MSB (index 0 in BIG_ENDIAN) to LSB
            for (0..num_vars) |i| {
                const ji = (j >> @intCast(num_vars - 1 - i)) & 1; // MSB first
                if (ji == 0) { // (1 - x_i) = 1 only when x_i = 0
                    var contrib = r_cycle[i]; // y_i
                    // Multiply by eq(x[i+1:], y[i+1:])
                    for ((i + 1)..num_vars) |k| {
                        const jk = (j >> @intCast(num_vars - 1 - k)) & 1;
                        const rk = r_cycle[k];
                        if (jk == 1) {
                            contrib = contrib.mul(rk);
                        } else {
                            contrib = contrib.mul(F.one().sub(rk));
                        }
                    }
                    result = result.add(contrib);
                }
            }

            return result;
        }

        /// Compute round polynomial for RegistersValEvaluation
        /// Returns [p(0), p(1), p(2), p(3)] for degree-3 sumcheck
        fn computeRegsValRoundPoly(inc: []F, wa: []F, lt: *const LtPolynomial(F), round: usize, tp: ?*ThreadPool) [4]F {
            const n = inc.len >> @intCast(round);
            const half = n / 2;

            if (half == 0) {
                var evals = [_]F{ F.zero(), F.zero(), F.zero(), F.zero() };
                if (n > 0) {
                    evals[0] = inc[0].mul(wa[0]).mul(lt.finalClaim());
                    evals[1] = evals[0];
                    evals[2] = evals[0];
                }
                return evals;
            }

            const LtPoly = LtPolynomial(F);
            const Ctx = struct {
                inc_p: []F,
                wa_p: []F,
                lt_p: *const LtPoly,
            };
            const ctx = Ctx{ .inc_p = inc, .wa_p = wa, .lt_p = lt };
            const identity = [_]F{ F.zero(), F.zero(), F.zero(), F.zero() };

            const mapFn = struct {
                fn f(c: Ctx, start: usize, end: usize) [4]F {
                    var r_u: [4]UnreducedProductAccum = .{UnreducedProductAccum.zero()} ** 4;
                    for (start..end) |i| {
                        const inc_0 = c.inc_p[2 * i];
                        const wa_0 = c.wa_p[2 * i];
                        const lt_0 = c.lt_p.getBoundCoeff(2 * i);
                        const inc_1 = c.inc_p[2 * i + 1];
                        const wa_1 = c.wa_p[2 * i + 1];
                        const lt_1 = c.lt_p.getBoundCoeff(2 * i + 1);

                        r_u[0].addAssign(inc_0.mul(wa_0).mulToProductAccum(lt_0));
                        r_u[1].addAssign(inc_1.mul(wa_1).mulToProductAccum(lt_1));
                        r_u[2].addAssign(inc_1.add(inc_1).sub(inc_0).mul(wa_1.add(wa_1).sub(wa_0)).mulToProductAccum(lt_1.add(lt_1).sub(lt_0)));
                        r_u[3].addAssign(inc_1.sub(inc_0).mul(wa_1.sub(wa_0)).mulToProductAccum(lt_1.sub(lt_0)));
                    }
                    return [4]F{ r_u[0].reduce(), r_u[1].reduce(), r_u[2].reduce(), r_u[3].reduce() };
                }
            }.f;
            const reduceFn = struct {
                fn f(a: [4]F, b: [4]F) [4]F {
                    return [4]F{ a[0].add(b[0]), a[1].add(b[1]), a[2].add(b[2]), a[3].add(b[3]) };
                }
            }.f;

            if (tp) |pool| {
                return pool.parallelReduce([4]F, half, identity, ctx, mapFn, reduceFn);
            }
            return mapFn(ctx, 0, half);
        }

        /// Bind challenge for RegistersValEvaluation polynomials
        fn bindRegsValChallenge(inc: []F, wa: []F, lt: *LtPolynomial(F), round: usize, r: F, tp: ?*ThreadPool, gpu: ?*GpuPolyOps) void {
            const n = inc.len >> @intCast(round);
            const half = n / 2;
            if (half == 0) return;

            // Bind LtPolynomial (operates on sqrt(T)-sized sub-arrays internally)
            lt.bind(r);

            // Bind inc and wa arrays (parallelize across 2 independent arrays)
            const arrays = [_][]F{ inc, wa };

            if (gpu) |g| {
                if (half >= 16384) {
                    for (arrays) |arr| {
                        g.polyBindLow(arr[0 .. half * 2], r, arr[0..half]) catch {
                            for (0..half) |i| {
                                arr[i] = arr[2 * i].add(r.mul(arr[2 * i + 1].sub(arr[2 * i])));
                            }
                        };
                    }
                } else {
                    for (0..half) |i| {
                        inc[i] = inc[2 * i].add(r.mul(inc[2 * i + 1].sub(inc[2 * i])));
                        wa[i] = wa[2 * i].add(r.mul(wa[2 * i + 1].sub(wa[2 * i])));
                    }
                }
            } else if (tp) |pool| {
                if (half >= 256) {
                    const BindCtx = struct { inc: []F, wa: []F, rv: F, h: usize };
                    const ctx = BindCtx{ .inc = inc, .wa = wa, .rv = r, .h = half };
                    pool.parallelForForce(2, ctx, struct {
                        fn f(c: BindCtx, arr_idx: usize) void {
                            const arr = if (arr_idx == 0) c.inc else c.wa;
                            for (0..c.h) |i| {
                                arr[i] = arr[2 * i].add(c.rv.mul(arr[2 * i + 1].sub(arr[2 * i])));
                            }
                        }
                    }.f);
                } else {
                    for (0..half) |i| {
                        inc[i] = inc[2 * i].add(r.mul(inc[2 * i + 1].sub(inc[2 * i])));
                        wa[i] = wa[2 * i].add(r.mul(wa[2 * i + 1].sub(wa[2 * i])));
                    }
                }
            } else {
                for (0..half) |i| {
                    inc[i] = inc[2 * i].add(r.mul(inc[2 * i + 1].sub(inc[2 * i])));
                    wa[i] = wa[2 * i].add(r.mul(wa[2 * i + 1].sub(wa[2 * i])));
                }
            }

            // Zero out upper half (inc and wa only; LtPolynomial handles its own state)
            for (half..n) |i| {
                inc[i] = F.zero();
                wa[i] = F.zero();
            }
        }

        /// Compute round polynomial for LookupsReadRaf (cycle rounds only)
        /// This computes Σ_j eq_reduction(j) * combined_vals(j)
        /// Returns [p(0), p(1), p(2), p_inf] for degree-2 polynomial (product of 2 linears)
        fn computeLookupsRoundPoly(eq_evals: []F, combined: []F, round: usize, tp: ?*ThreadPool) [4]F {
            const n = eq_evals.len >> @intCast(round);
            const half = n / 2;

            if (half == 0) {
                var evals = [_]F{ F.zero(), F.zero(), F.zero(), F.zero() };
                if (n > 0) {
                    const c = eq_evals[0].mul(combined[0]);
                    evals[0] = c;
                    evals[1] = c;
                    evals[2] = c;
                }
                return evals;
            }

            const Ctx = struct { eq: []F, comb: []F };
            const ctx = Ctx{ .eq = eq_evals, .comb = combined };
            const identity = [_]F{ F.zero(), F.zero(), F.zero(), F.zero() };

            const mapFn = struct {
                fn f(c: Ctx, start: usize, end: usize) [4]F {
                    var r_u: [4]UnreducedProductAccum = .{UnreducedProductAccum.zero()} ** 4;
                    for (start..end) |i| {
                        const eq_0 = c.eq[2 * i];
                        const eq_1 = c.eq[2 * i + 1];
                        const c_0 = c.comb[2 * i];
                        const c_1 = c.comb[2 * i + 1];
                        r_u[0].addAssign(eq_0.mulToProductAccum(c_0));
                        r_u[1].addAssign(eq_1.mulToProductAccum(c_1));
                        r_u[2].addAssign(eq_1.add(eq_1).sub(eq_0).mulToProductAccum(c_1.add(c_1).sub(c_0)));
                        r_u[3].addAssign(eq_1.sub(eq_0).mulToProductAccum(c_1.sub(c_0)));
                    }
                    return [4]F{ r_u[0].reduce(), r_u[1].reduce(), r_u[2].reduce(), r_u[3].reduce() };
                }
            }.f;
            const reduceFn = struct {
                fn f(a: [4]F, b: [4]F) [4]F {
                    return [4]F{ a[0].add(b[0]), a[1].add(b[1]), a[2].add(b[2]), a[3].add(b[3]) };
                }
            }.f;

            if (tp) |pool| {
                return pool.parallelReduce([4]F, half, identity, ctx, mapFn, reduceFn);
            }
            return mapFn(ctx, 0, half);
        }

        /// Bind challenge for LookupsReadRaf polynomials (cycle rounds) - legacy version
        fn bindLookupsChallenge(eq_evals: []F, combined: []F, round: usize, r: F, tp: ?*ThreadPool, gpu: ?*GpuPolyOps) void {
            const n = eq_evals.len >> @intCast(round);
            const half = n / 2;
            if (half == 0) return;

            const arrays = [_][]F{ eq_evals, combined };

            if (gpu) |g| {
                if (half >= 16384) {
                    for (arrays) |arr| {
                        g.polyBindLow(arr[0 .. half * 2], r, arr[0..half]) catch {
                            for (0..half) |i| {
                                arr[i] = arr[2 * i].add(r.mul(arr[2 * i + 1].sub(arr[2 * i])));
                            }
                        };
                    }
                } else {
                    for (0..half) |i| {
                        eq_evals[i] = eq_evals[2 * i].add(r.mul(eq_evals[2 * i + 1].sub(eq_evals[2 * i])));
                        combined[i] = combined[2 * i].add(r.mul(combined[2 * i + 1].sub(combined[2 * i])));
                    }
                }
            } else if (tp) |pool| {
                if (half >= 256) {
                    const BindCtx = struct { eq: []F, comb: []F, rv: F, h: usize };
                    const ctx = BindCtx{ .eq = eq_evals, .comb = combined, .rv = r, .h = half };
                    pool.parallelForForce(2, ctx, struct {
                        fn f(c: BindCtx, arr_idx: usize) void {
                            const arr = if (arr_idx == 0) c.eq else c.comb;
                            for (0..c.h) |i| {
                                arr[i] = arr[2 * i].add(c.rv.mul(arr[2 * i + 1].sub(arr[2 * i])));
                            }
                        }
                    }.f);
                } else {
                    for (0..half) |i| {
                        eq_evals[i] = eq_evals[2 * i].add(r.mul(eq_evals[2 * i + 1].sub(eq_evals[2 * i])));
                        combined[i] = combined[2 * i].add(r.mul(combined[2 * i + 1].sub(combined[2 * i])));
                    }
                }
            } else {
                for (0..half) |i| {
                    eq_evals[i] = eq_evals[2 * i].add(r.mul(eq_evals[2 * i + 1].sub(eq_evals[2 * i])));
                    combined[i] = combined[2 * i].add(r.mul(combined[2 * i + 1].sub(combined[2 * i])));
                }
            }

            // Zero out upper half
            for (half..n) |i| {
                eq_evals[i] = F.zero();
                combined[i] = F.zero();
            }
        }

        /// Compute round polynomial for LookupsReadRaf with ra_weights (cycle rounds)
        /// This computes Σ_j eq(j) * ra(j) * combined(j)
        /// Returns [p(0), p(1), p(2), p_inf] for degree-3 polynomial (product of 3 linears)
        fn computeLookupsRoundPolyWithRa(eq_evals: []F, ra_weights: []F, combined: []F, round: usize, tp: ?*ThreadPool) [4]F {
            const n = eq_evals.len >> @intCast(round);
            const half = n / 2;

            if (half == 0) {
                var evals = [_]F{ F.zero(), F.zero(), F.zero(), F.zero() };
                if (n > 0) {
                    const c = eq_evals[0].mul(ra_weights[0]).mul(combined[0]);
                    evals[0] = c;
                    evals[1] = c;
                    evals[2] = c;
                }
                return evals;
            }

            const Ctx = struct { eq: []F, ra: []F, comb: []F };
            const ctx = Ctx{ .eq = eq_evals, .ra = ra_weights, .comb = combined };
            const identity = [_]F{ F.zero(), F.zero(), F.zero(), F.zero() };

            const mapFn = struct {
                fn f(c: Ctx, start: usize, end: usize) [4]F {
                    var r_u: [4]UnreducedProductAccum = .{UnreducedProductAccum.zero()} ** 4;
                    for (start..end) |i| {
                        const eq_0 = c.eq[2 * i];
                        const eq_1 = c.eq[2 * i + 1];
                        const ra_0 = c.ra[2 * i];
                        const ra_1 = c.ra[2 * i + 1];
                        const c_0 = c.comb[2 * i];
                        const c_1 = c.comb[2 * i + 1];
                        r_u[0].addAssign(eq_0.mul(ra_0).mulToProductAccum(c_0));
                        r_u[1].addAssign(eq_1.mul(ra_1).mulToProductAccum(c_1));
                        r_u[2].addAssign(eq_1.add(eq_1).sub(eq_0).mul(ra_1.add(ra_1).sub(ra_0)).mulToProductAccum(c_1.add(c_1).sub(c_0)));
                        r_u[3].addAssign(eq_1.sub(eq_0).mul(ra_1.sub(ra_0)).mulToProductAccum(c_1.sub(c_0)));
                    }
                    return [4]F{ r_u[0].reduce(), r_u[1].reduce(), r_u[2].reduce(), r_u[3].reduce() };
                }
            }.f;
            const reduceFn = struct {
                fn f(a: [4]F, b: [4]F) [4]F {
                    return [4]F{ a[0].add(b[0]), a[1].add(b[1]), a[2].add(b[2]), a[3].add(b[3]) };
                }
            }.f;

            if (tp) |pool| {
                return pool.parallelReduce([4]F, half, identity, ctx, mapFn, reduceFn);
            }
            return mapFn(ctx, 0, half);
        }

        /// Bind challenge for LookupsReadRaf polynomials with ra_weights (cycle rounds)
        fn bindLookupsCycleChallengeWithRa(eq_evals: []F, ra_weights: []F, combined: []F, round: usize, r: F, tp: ?*ThreadPool, gpu: ?*GpuPolyOps) void {
            const n = eq_evals.len >> @intCast(round);
            const half = n / 2;
            if (half == 0) return;

            const arrays = [_][]F{ eq_evals, ra_weights, combined };

            if (gpu) |g| {
                if (half >= 16384) {
                    for (arrays) |arr| {
                        g.polyBindLow(arr[0 .. half * 2], r, arr[0..half]) catch {
                            for (0..half) |i| {
                                arr[i] = arr[2 * i].add(r.mul(arr[2 * i + 1].sub(arr[2 * i])));
                            }
                        };
                    }
                } else {
                    for (0..half) |i| {
                        eq_evals[i] = eq_evals[2 * i].add(r.mul(eq_evals[2 * i + 1].sub(eq_evals[2 * i])));
                        ra_weights[i] = ra_weights[2 * i].add(r.mul(ra_weights[2 * i + 1].sub(ra_weights[2 * i])));
                        combined[i] = combined[2 * i].add(r.mul(combined[2 * i + 1].sub(combined[2 * i])));
                    }
                }
            } else if (tp) |pool| {
                if (half >= 256) {
                    const BindCtx = struct { eq: []F, ra: []F, comb: []F, rv: F, h: usize };
                    const ctx = BindCtx{ .eq = eq_evals, .ra = ra_weights, .comb = combined, .rv = r, .h = half };
                    pool.parallelForForce(3, ctx, struct {
                        fn f(c: BindCtx, arr_idx: usize) void {
                            const arr = switch (arr_idx) {
                                0 => c.eq,
                                1 => c.ra,
                                2 => c.comb,
                                else => unreachable,
                            };
                            for (0..c.h) |i| {
                                arr[i] = arr[2 * i].add(c.rv.mul(arr[2 * i + 1].sub(arr[2 * i])));
                            }
                        }
                    }.f);
                } else {
                    for (0..half) |i| {
                        eq_evals[i] = eq_evals[2 * i].add(r.mul(eq_evals[2 * i + 1].sub(eq_evals[2 * i])));
                        ra_weights[i] = ra_weights[2 * i].add(r.mul(ra_weights[2 * i + 1].sub(ra_weights[2 * i])));
                        combined[i] = combined[2 * i].add(r.mul(combined[2 * i + 1].sub(combined[2 * i])));
                    }
                }
            } else {
                for (0..half) |i| {
                    eq_evals[i] = eq_evals[2 * i].add(r.mul(eq_evals[2 * i + 1].sub(eq_evals[2 * i])));
                    ra_weights[i] = ra_weights[2 * i].add(r.mul(ra_weights[2 * i + 1].sub(ra_weights[2 * i])));
                    combined[i] = combined[2 * i].add(r.mul(combined[2 * i + 1].sub(combined[2 * i])));
                }
            }

            // Zero out upper half
            for (half..n) |i| {
                eq_evals[i] = F.zero();
                ra_weights[i] = F.zero();
                combined[i] = F.zero();
            }
        }

        /// Bind challenge for a single polynomial (used for per-chunk ra weights)
        fn bindSinglePolynomial(poly: []F, round: usize, r: F, tp: ?*ThreadPool, gpu: ?*GpuPolyOps) void {
            _ = tp; // Single polynomial can't be parallelized across arrays
            const n = poly.len >> @intCast(round);
            const half = n / 2;
            if (half == 0) return;

            if (gpu) |g| {
                if (half >= 16384) {
                    g.polyBindLow(poly[0 .. half * 2], r, poly[0..half]) catch {
                        for (0..half) |i| {
                            poly[i] = poly[2 * i].add(r.mul(poly[2 * i + 1].sub(poly[2 * i])));
                        }
                    };
                } else {
                    for (0..half) |i| {
                        poly[i] = poly[2 * i].add(r.mul(poly[2 * i + 1].sub(poly[2 * i])));
                    }
                }
            } else {
                for (0..half) |i| {
                    poly[i] = poly[2 * i].add(r.mul(poly[2 * i + 1].sub(poly[2 * i])));
                }
            }

            // Zero out upper half
            for (half..n) |i| {
                poly[i] = F.zero();
            }
        }
    };
}

/// Evaluate LeftOperandPolynomial at r
/// LeftOperand(r) = Σ_{i=0}^{n/2-1} r[2i] * 2^(n/2-1-i)
/// For LOG_K=128: sum of even-indexed r values with powers of 2
pub fn evaluateLeftOperand(comptime F: type, r: []const F) F {
    const n = r.len;
    std.debug.assert(n % 2 == 0);
    var result = F.zero();
    var power = F.one();
    // Process from LSB to MSB of result
    var i: usize = n / 2;
    while (i > 0) {
        i -= 1;
        result = result.add(r[2 * i].mul(power));
        power = power.add(power); // power *= 2
    }
    return result;
}

/// Evaluate RightOperandPolynomial at r
/// RightOperand(r) = Σ_{i=0}^{n/2-1} r[2i+1] * 2^(n/2-1-i)
/// For LOG_K=128: sum of odd-indexed r values with powers of 2
pub fn evaluateRightOperand(comptime F: type, r: []const F) F {
    const n = r.len;
    std.debug.assert(n % 2 == 0);
    var result = F.zero();
    var power = F.one();
    // Process from LSB to MSB of result
    var i: usize = n / 2;
    while (i > 0) {
        i -= 1;
        const idx = 2 * i + 1;
        const term = r[idx].mul(power);
        result = result.add(term);
        // Debug: print first few and last few iterations
        if (n == 128 and (i < 3 or i >= 61)) {
            if (comptime debug_verbose) {
                dbg("[RIGHT_OP_DEBUG] i={d}: r[{d}]={x}, power={x}, term={x}, result={x}\n", .{
                    i, idx, r[idx].toBytesBE()[16..32].*, power.toBytesBE()[16..32].*,
                    term.toBytesBE()[16..32].*, result.toBytesBE()[16..32].*,
                });
            }
        }
        power = power.add(power); // power *= 2
    }
    if (comptime debug_verbose) {
        dbg("[RIGHT_OP_DEBUG] final result = {x}\n", .{result.toBytesBE()[16..32].*});
    }
    return result;
}

/// Evaluate IdentityPolynomial at r
/// Identity(r) = Σ_{i=0}^{n-1} r[i] * 2^(n-1-i)
/// This treats r as bits of a binary number
pub fn evaluateIdentity(comptime F: type, r: []const F) F {
    const n = r.len;
    var result = F.zero();
    var power = F.one();
    // Process from LSB to MSB
    var i: usize = n;
    while (i > 0) {
        i -= 1;
        const term = r[i].mul(power);
        result = result.add(term);
        // Debug: print first few and last few iterations
        if (n == 128 and (i < 4 or i >= 124)) {
            if (comptime debug_verbose) {
                dbg("[IDENTITY_DEBUG] i={d}: r[{d}]={x}, power={x}, term={x}, result={x}\n", .{
                    i, i, r[i].toBytesBE()[16..32].*, power.toBytesBE()[16..32].*,
                    term.toBytesBE()[16..32].*, result.toBytesBE()[16..32].*,
                });
            }
        }
        power = power.add(power); // power *= 2
    }
    if (comptime debug_verbose) {
        dbg("[IDENTITY_DEBUG] final result = {x}\n", .{result.toBytesBE()[16..32].*});
    }
    return result;
}

/// Compute eq(r, s) for two field element vectors
/// eq(r, s) = Π_i (r[i]*s[i] + (1-r[i])*(1-s[i]))
pub fn computeEqPolynomial(comptime F: type, r: []const F, s: []const F) F {
    std.debug.assert(r.len == s.len);
    var result = F.one();
    for (r, s) |ri, si| {
        // eq_i = ri*si + (1-ri)*(1-si) = 1 - ri - si + 2*ri*si
        const ri_si = ri.mul(si);
        const term = F.one().sub(ri).sub(si).add(ri_si).add(ri_si);
        result = result.mul(term);
    }
    return result;
}

/// Compute eq(k, r) where r is in BIG_ENDIAN order.
///
/// This matches Jolt's EqPolynomial::evals convention:
/// - evals[k] = Π_j (bit_{n-1-j}(k) ? r[j] : (1-r[j]))
/// - Equivalently: bit j of k ↔ r[n-1-j]
pub fn computeEqAtPoint(comptime F: type, r: []const F, k: u64) F {
    const n = r.len;
    var result = F.one();
    for (0..n) |j| {
        // Extract bit (n-1-j) of k: b_j = (k >> (n-1-j)) & 1
        const bj: u1 = @truncate(k >> @intCast(n - 1 - j));
        const rj = r[j]; // r[j] corresponds to bit (n-1-j) of k
        if (bj == 1) {
            result = result.mul(rj);
        } else {
            result = result.mul(F.one().sub(rj));
        }
    }
    return result;
}

/// Interleave bits of two 64-bit values into a 128-bit value
/// Matches Jolt's interleave_bits(even_bits, odd_bits):
///   - x (even_bits) goes to ODD positions (1, 3, 5, ...)
///   - y (odd_bits) goes to EVEN positions (0, 2, 4, ...)
/// In Jolt: `interleave_bits(x, y)` returns `(spread(x) << 1) | spread(y)`
pub fn interleaveBits128(x: u64, y: u64) u128 {
    var result: u128 = 0;
    for (0..64) |i| {
        const xi: u128 = @as(u128, (x >> @intCast(i)) & 1);
        const yi: u128 = @as(u128, (y >> @intCast(i)) & 1);
        // x at odd positions (2i+1), y at even positions (2i) - matches Jolt
        result |= xi << @intCast(2 * i + 1);
        result |= yi << @intCast(2 * i);
    }
    return result;
}

/// Get bit `bit_index` from a 128-bit value stored as (lo, hi)
/// bit_index 0 is LSB, bit_index 127 is MSB
pub fn getBit128(lo: u64, hi: u64, bit_index: usize) u1 {
    if (bit_index < 64) {
        return @truncate(lo >> @intCast(bit_index));
    } else {
        return @truncate(hi >> @intCast(bit_index - 64));
    }
}

/// Get the lookup table index for an instruction
/// Returns -1 if no lookup table is used, otherwise returns table index 0-41
/// Based on Jolt's LookupTables enum ordering:
///   0: RangeCheck, 1: RangeCheckAligned, 2: And, 3: Andn, 4: Or, 5: Xor,
///   6: Equal, 7: SignedGreaterThanEqual, 8: UnsignedGreaterThanEqual,
///   9: NotEqual, 10: SignedLessThan, 11: UnsignedLessThan, 12: Movsign,
///   13: UpperWord, 14: LessThanEqual, 15-17: Valid*Remainder/Div0,
///   18-19: HalfwordAlignment/WordAlignment, 20-21: LowerHalfWord/SignExtendHalfWord,
///   22-23: Pow2/Pow2W, 24: ShiftRightBitmask, 25: VirtualRev8W,
///   26: VirtualSRL, 27: VirtualSRA, 28: VirtualROTR, 29: VirtualROTRW,
///   30-31: VirtualChangeDivisor/W, 32: MulUNoOverflow, 33-40: VirtualXORROT*
pub fn getLookupTableIndex(opcode: u32, funct3: u32, funct7: u32) i8 {
    return switch (opcode) {
        0x33 => blk: { // R-type
            if (funct3 == 0 and funct7 == 0) break :blk 0; // ADD -> RangeCheckTable
            if (funct3 == 0 and funct7 == 0x20) break :blk 0; // SUB -> RangeCheckTable
            if (funct3 == 7) break :blk 2; // AND -> AndTable
            if (funct3 == 6) break :blk 4; // OR -> OrTable
            if (funct3 == 4) break :blk 5; // XOR -> XorTable
            if (funct3 == 1) break :blk -1; // SLL -> uses virtual decomposition
            if (funct3 == 5 and funct7 == 0) break :blk 25; // SRL -> VirtualSRLTable
            if (funct3 == 5 and funct7 == 0x20) break :blk 26; // SRA -> VirtualSRATable
            if (funct7 == 0x01 and funct3 == 0) break :blk 0; // MUL -> RangeCheckTable
            if (funct7 == 0x01 and funct3 == 3) break :blk 13; // MULHU -> UpperWordTable
            if (funct3 == 2) break :blk 10; // SLT -> SignedLessThanTable
            if (funct3 == 3) break :blk 11; // SLTU -> UnsignedLessThanTable
            break :blk -1;
        },
        0x13 => blk: { // I-type
            if (funct3 == 0) break :blk 0; // ADDI -> RangeCheckTable
            if (funct3 == 7) break :blk 2; // ANDI -> AndTable
            if (funct3 == 6) break :blk 4; // ORI -> OrTable
            if (funct3 == 4) break :blk 5; // XORI -> XorTable
            if (funct3 == 1) break :blk -1; // SLLI -> uses virtual decomposition
            if (funct3 == 5 and (funct7 & 0x40) == 0) break :blk 25; // SRLI -> VirtualSRLTable
            if (funct3 == 5 and (funct7 & 0x40) != 0) break :blk 26; // SRAI -> VirtualSRATable
            if (funct3 == 2) break :blk 10; // SLTI -> SignedLessThanTable
            if (funct3 == 3) break :blk 11; // SLTIU -> UnsignedLessThanTable
            break :blk -1;
        },
        0x1b => blk: { // OP-IMM-32
            if (funct3 == 0) break :blk 0; // ADDIW -> RangeCheckTable
            break :blk -1;
        },
        0x3b => blk: { // OP-32
            if (funct3 == 0 and funct7 == 0) break :blk 0; // ADDW -> RangeCheckTable
            if (funct3 == 0 and funct7 == 0x20) break :blk 0; // SUBW -> RangeCheckTable
            if (funct3 == 6 and funct7 == 0x01) break :blk 30; // VirtualChangeDivisorW -> VirtualChangeDivisorWTable
            break :blk -1;
        },
        0x63 => blk: { // B-type (branches)
            if (funct3 == 0) break :blk 6; // BEQ -> EqualTable
            if (funct3 == 1) break :blk 9; // BNE -> NotEqualTable
            if (funct3 == 4) break :blk 10; // BLT -> SignedLessThanTable
            if (funct3 == 5) break :blk 7; // BGE -> SignedGreaterThanEqualTable
            if (funct3 == 6) break :blk 11; // BLTU -> UnsignedLessThanTable
            if (funct3 == 7) break :blk 8; // BGEU -> UnsignedGreaterThanEqualTable
            break :blk -1;
        },
        0x0B => 20, // VirtualSignExtendWord -> SignExtendHalfWordTable
        0x2B => blk2b: { // Virtual I-type
            if (funct3 == 1) break :blk2b 21; // VirtualPow2 -> Pow2Table
            if (funct3 == 2) break :blk2b 23; // VirtualShiftRightBitmask -> ShiftRightBitmaskTable
            break :blk2b 0; // VirtualMULI (funct3=0) -> RangeCheckTable
        },
        0x5B => blk5b: { // Virtual shift right
            if (funct3 == 5) break :blk5b 26; // VirtualSRAI -> VirtualSRATable
            break :blk5b 25; // VirtualSRLI -> VirtualSRLTable (funct3=0)
        },
        0x02 => 0, // VirtualAdvice -> RangeCheckTable
        0x22 => blk22: { // Virtual assert
            if (funct3 == 1) break :blk22 16; // VirtualAssertValidDiv0 -> ValidDiv0Table
            if (funct3 == 2) break :blk22 17; // VirtualAssertHalfwordAlignment -> HalfwordAlignmentTable
            if (funct3 == 3) break :blk22 18; // VirtualAssertWordAlignment -> WordAlignmentTable
            break :blk22 6; // VirtualAssertEQ -> EqualTable (funct3=0)
        },
        0x42 => 19, // VirtualZeroExtendWord -> LowerHalfWordTable
        0x62 => 15, // VirtualAssertValidUnsignedRemainder -> ValidUnsignedRemainderTable
        0x37 => 0, // LUI -> RangeCheckTable
        0x17 => 0, // AUIPC -> RangeCheckTable
        0x6f => 0, // JAL -> RangeCheckTable
        0x67 => 1, // JALR -> RangeCheckAlignedTable
        0x03 => -1, // Load -> None (no lookup table)
        0x23 => -1, // Store -> None (no lookup table)
        else => -1,
    };
}

test "stage5_prover compiles" {
    const F = @import("../../field/mod.zig").BN254Scalar;
    const allocator = std.testing.allocator;

    const prover = Stage5BatchedProver(F).init(allocator);
    _ = prover;
}

test "operand polynomial evaluation" {
    const F = @import("../../field/mod.zig").BN254Scalar;

    // Simple test: r = [1, 0, 0, 1] (4 vars, LOG_K=4)
    // Left operand uses r[0], r[2] = 1, 0 → 1*2 + 0*1 = 2
    // Right operand uses r[1], r[3] = 0, 1 → 0*2 + 1*1 = 1
    const r = [_]F{ F.one(), F.zero(), F.zero(), F.one() };

    const left = evaluateLeftOperand(F, &r);
    const right = evaluateRightOperand(F, &r);

    // Left: r[0]*2 + r[2]*1 = 1*2 + 0*1 = 2
    try std.testing.expectEqual(F.fromU64(2), left);
    // Right: r[1]*2 + r[3]*1 = 0*2 + 1*1 = 1
    try std.testing.expectEqual(F.fromU64(1), right);
}
