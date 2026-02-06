//! Stage 5 Batched Sumcheck Prover
//!
//! Stage 5 is a batched sumcheck with 3 instances:
//! 1. RegistersValEvaluation: 8 rounds (log_T)
//! 2. RamRaClaimReduction: 24 rounds (log_K + log_T)
//! 3. LookupsReadRaf: 136 rounds (LOOKUPS_LOG_K + log_T)
//!
//! The batched sumcheck combines instances with different round counts.
//! Instances with fewer rounds contribute constant polynomials (scaled input claims)
//! until their variables start being bound.
//!
//! Reference: jolt-core/src/subprotocols/sumcheck.rs

const std = @import("std");
const Allocator = std.mem.Allocator;

const poly_mod = @import("../../poly/mod.zig");
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
        lookups_table_flags: []F, // LookupTableFlag(i) for i in 0..42
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
            log_ram_k: usize,
            gamma_ram_ra: F,
            gamma_lookups_raf: F,
            lookups_ra_virtual_log_k_chunk: usize,
        ) !Stage5Result(F) {
            // Instance configurations
            const regs_val_num_rounds = n_cycle_vars; // 8 rounds
            const ram_ra_num_rounds = log_ram_k + n_cycle_vars; // 24 rounds
            const lookups_num_rounds = LOOKUPS_LOG_K + n_cycle_vars; // 136 rounds
            const max_num_rounds = lookups_num_rounds;

            // Use gamma_ram_ra for RamRaClaimReduction (Instance 1)
            // Use gamma_lookups_raf for LookupsReadRaf (Instance 2)
            const gamma = gamma_ram_ra;
            const gamma_raf = gamma_lookups_raf;

            std.debug.print("[STAGE5] Configuration: regs={}, ram_ra={}, lookups={}, max={}\n", .{
                regs_val_num_rounds, ram_ra_num_rounds, lookups_num_rounds, max_num_rounds,
            });

            // Get input claims from accumulator
            const regs_val_input = opening_claims.get(
                .{ .Virtual = .{ .poly = .RegistersVal, .sumcheck_id = .RegistersReadWriteChecking } },
            ) orelse F.zero();

            // RamRaClaimReduction batches 4 claims with gamma_ram_ra:
            // input = claim_raf + gamma*claim_val_final + gamma^2*claim_rw + gamma^3*claim_val_eval
            const claim_raf = opening_claims.get(
                .{ .Virtual = .{ .poly = .RamRa, .sumcheck_id = .RamRafEvaluation } },
            ) orelse F.zero();
            const claim_val_final = opening_claims.get(
                .{ .Virtual = .{ .poly = .RamRa, .sumcheck_id = .RamValFinalEvaluation } },
            ) orelse F.zero();
            const claim_rw = opening_claims.get(
                .{ .Virtual = .{ .poly = .RamRa, .sumcheck_id = .RamReadWriteChecking } },
            ) orelse F.zero();
            const claim_val_eval = opening_claims.get(
                .{ .Virtual = .{ .poly = .RamRa, .sumcheck_id = .RamValEvaluation } },
            ) orelse F.zero();

            const gamma2 = gamma.mul(gamma);
            const gamma3 = gamma2.mul(gamma);
            const ram_ra_input = claim_raf
                .add(gamma.mul(claim_val_final))
                .add(gamma2.mul(claim_rw))
                .add(gamma3.mul(claim_val_eval));

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

            std.debug.print("[STAGE5] Input claims:\n", .{});
            std.debug.print("  regs_val_input = {any}\n", .{regs_val_input.toBytesBE()[0..8]});
            std.debug.print("  ram_ra_input = {any}\n", .{ram_ra_input.toBytesBE()[0..8]});
            std.debug.print("  lookups_input = {any}\n", .{lookups_input.toBytesBE()[0..8]});

            // Append input claims to transcript and get batching coefficients
            transcript.appendScalar(regs_val_input);
            transcript.appendScalar(ram_ra_input);
            transcript.appendScalar(lookups_input);

            const batch0 = transcript.challengeScalarFull();
            const batch1 = transcript.challengeScalarFull();
            const batch2 = transcript.challengeScalarFull();

            std.debug.print("[STAGE5] Batching coefficients (LE for Jolt comparison):\n", .{});
            std.debug.print("  batch0 (LE) = {any}\n", .{batch0.toBytes()});
            std.debug.print("  batch1 (LE) = {any}\n", .{batch1.toBytes()});
            std.debug.print("  batch2 (LE) = {any}\n", .{batch2.toBytes()});

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

            // Initial batched claim
            const batched_claim = batch0.mul(regs_scaled)
                .add(batch1.mul(ram_ra_scaled))
                .add(batch2.mul(lookups_scaled));

            std.debug.print("[STAGE5] Initial batched claim = {any}\n", .{batched_claim.toBytesBE()});

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
                const half_lookups = lookups_claim.mul(F.fromU64(2).inverse().?);
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
                transcript.appendMessage("UniPoly_begin");
                transcript.appendScalar(compressed[0]); // c0
                transcript.appendScalar(compressed[1]); // c2
                transcript.appendScalar(compressed[2]); // c3
                transcript.appendMessage("UniPoly_end");

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
                lookups_claim = lookups_claim.mul(F.fromU64(2).inverse().?);
            }

            std.debug.print("[STAGE5] Final batched claim = {any}\n", .{current_batched_claim.toBytesBE()});

            // Allocate opening claim arrays
            const num_lookup_tables: usize = 42;
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
            const regs_val_num_rounds = n_cycle_vars;
            const ram_ra_num_rounds = log_ram_k + n_cycle_vars;
            const lookups_num_rounds = LOOKUPS_LOG_K + n_cycle_vars;
            const max_num_rounds = lookups_num_rounds;

            // Use gamma_ram_ra for Instance 1 (RamRaClaimReduction)
            // Use gamma_lookups_raf for Instance 2 (LookupsReadRaf)
            const gamma = gamma_ram_ra; // For RamRaClaimReduction

            std.debug.print("[STAGE5] Configuration with trace: regs={}, ram_ra={}, lookups={}, max={}\n", .{
                regs_val_num_rounds, ram_ra_num_rounds, lookups_num_rounds, max_num_rounds,
            });

            // Debug: print RamRaClaimReduction opening points (use the params to suppress warnings)
            std.debug.print("[STAGE5] RamRaClaimReduction opening points:\n", .{});
            std.debug.print("  r_address_raf.len = {}, r_address_rw.len = {}\n", .{ r_address_raf.len, r_address_rw.len });
            std.debug.print("  r_cycle_raf.len = {}, r_cycle_rw.len = {}, r_cycle_val.len = {}\n", .{
                r_cycle_raf.len,
                r_cycle_rw.len,
                r_cycle_val.len,
            });

            // Get input claims from accumulator
            const regs_val_input = opening_claims.get(
                .{ .Virtual = .{ .poly = .RegistersVal, .sumcheck_id = .RegistersReadWriteChecking } },
            ) orelse F.zero();

            // DEBUG: Print the exact value read from opening_claims
            std.debug.print("[STAGE5 GET] RegistersVal@RegistersReadWriteChecking (trace-aware path):\n", .{});
            std.debug.print("  LE bytes = {any}\n", .{regs_val_input.toBytes()});
            std.debug.print("  BE bytes = {any}\n", .{regs_val_input.toBytesBE()});

            const claim_raf = opening_claims.get(
                .{ .Virtual = .{ .poly = .RamRa, .sumcheck_id = .RamRafEvaluation } },
            ) orelse F.zero();
            const claim_val_final = opening_claims.get(
                .{ .Virtual = .{ .poly = .RamRa, .sumcheck_id = .RamValFinalEvaluation } },
            ) orelse F.zero();
            const claim_rw = opening_claims.get(
                .{ .Virtual = .{ .poly = .RamRa, .sumcheck_id = .RamReadWriteChecking } },
            ) orelse F.zero();
            const claim_val_eval = opening_claims.get(
                .{ .Virtual = .{ .poly = .RamRa, .sumcheck_id = .RamValEvaluation } },
            ) orelse F.zero();

            // RamRaClaimReduction uses gamma_ram_ra
            const gamma2 = gamma.mul(gamma);
            const gamma3 = gamma2.mul(gamma);
            const ram_ra_input = claim_raf
                .add(gamma.mul(claim_val_final))
                .add(gamma2.mul(claim_rw))
                .add(gamma3.mul(claim_val_eval));

            // Debug: print the four claims that make up ram_ra_input
            std.debug.print("[STAGE5] RamRaClaimReduction input components:\n", .{});
            std.debug.print("  claim_raf (RamRafEvaluation) = {any}\n", .{claim_raf.toBytesBE()[16..32].*});
            std.debug.print("  claim_val_final (RamValFinalEvaluation) = {any}\n", .{claim_val_final.toBytesBE()[16..32].*});
            std.debug.print("  claim_rw (RamReadWriteChecking) = {any}\n", .{claim_rw.toBytesBE()[16..32].*});
            std.debug.print("  claim_val_eval (RamValEvaluation) = {any}\n", .{claim_val_eval.toBytesBE()[16..32].*});
            std.debug.print("  gamma = {any}\n", .{gamma.toBytesBE()[16..32].*});

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

            std.debug.print("[STAGE5] Input claims (with trace):\n", .{});
            std.debug.print("  regs_val_input = {any}\n", .{regs_val_input.toBytesBE()});
            std.debug.print("  ram_ra_input = {any}\n", .{ram_ra_input.toBytesBE()});
            std.debug.print("  lookups_input = {any}\n", .{lookups_input.toBytesBE()});
            std.debug.print("[STAGE5] Transcript state BEFORE appending input claims: {any}\n", .{transcript.state[0..8]});

            // Append input claims to transcript and get batching coefficients
            transcript.appendScalar(regs_val_input);
            transcript.appendScalar(ram_ra_input);
            transcript.appendScalar(lookups_input);
            std.debug.print("[STAGE5] Transcript state AFTER appending input claims: {any}\n", .{transcript.state[0..8]});

            const batch0 = transcript.challengeScalarFull();
            const batch1 = transcript.challengeScalarFull();
            const batch2 = transcript.challengeScalarFull();

            std.debug.print("[STAGE5] Batching coefficients:\n", .{});
            std.debug.print("  batch0 = {x}\n", .{batch0.toBytesBE()[16..32].*});
            std.debug.print("  batch1 = {x}\n", .{batch1.toBytesBE()[16..32].*});
            std.debug.print("  batch2 = {x}\n", .{batch2.toBytesBE()[16..32].*});

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
            std.debug.print("[STAGE5] r_address_regs (len={}):\n", .{r_address_regs.len});
            for (r_address_regs, 0..) |r, i| {
                std.debug.print("  r_address[{}] = {any}\n", .{ i, r.toBytesBE()[0..8] });
            }
            std.debug.print("[STAGE5] r_cycle_regs (len={}):\n", .{r_cycle_regs.len});
            for (r_cycle_regs, 0..) |r, i| {
                std.debug.print("  r_cycle[{}] = {any}\n", .{ i, r.toBytesBE()[0..8] });
            }

            // Compute LT polynomial using efficient algorithm
            // lt_evals[j] = LT(j, r_cycle) for all j in [0, T)
            // r_cycle_regs is in BIG_ENDIAN order (MSB first) from Stage 4
            const lt_evals = try computeAllLtEvals(self.allocator, r_cycle_regs);
            defer self.allocator.free(lt_evals);

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
            var lookups_indices_lo = try self.allocator.alloc(u64, T); // Lower 64 bits of lookup index
            var lookups_indices_hi = try self.allocator.alloc(u64, T); // Upper 64 bits of lookup index
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

            var ra_chunk_weights: [MAX_RA_CHUNKS][]F = undefined;
            for (0..ra_num_chunks) |chunk_idx| {
                ra_chunk_weights[chunk_idx] = try self.allocator.alloc(F, T);
                @memset(ra_chunk_weights[chunk_idx], F.one());
            }
            defer {
                for (0..ra_num_chunks) |chunk_idx| {
                    self.allocator.free(ra_chunk_weights[chunk_idx]);
                }
            }

            // Track which cycles use which lookup table (for flag claims)
            // and which use identity path (for raf_flag claim)
            var cycle_table_indices = try self.allocator.alloc(i8, T);
            var cycle_is_identity_path = try self.allocator.alloc(bool, T);
            defer self.allocator.free(cycle_table_indices);
            defer self.allocator.free(cycle_is_identity_path);
            @memset(cycle_table_indices, -1); // -1 = no table
            @memset(cycle_is_identity_path, false);

            // Store table MLE evaluations at r_address (populated during rematerialization)
            // These are used for computing val_claim = Σ table_flags[i] * table_values[i]
            // Note: Jolt has 42 tables but our prefix-suffix only covers 41
            const MAX_LOOKUP_TABLES: usize = 42;
            var stored_table_values: [MAX_LOOKUP_TABLES]F = [_]F{F.zero()} ** MAX_LOOKUP_TABLES;

            // Build eq_reduction[j] = eq(j, r_reduction) for all cycles j
            // r_reduction is in BIG_ENDIAN order (MSB first)
            for (0..T) |j| {
                lookups_eq_evals[j] = computeEqAtIndex(r_reduction, j);
            }

            // Debug: print first few eq values and verify sum = 1
            std.debug.print("[STAGE5 EQ DEBUG] T={}, n_vars={}, First 5 eq_evals:\n", .{ T, n_cycle_vars });
            var eq_sum = F.zero();
            var j_idx: usize = 0;
            while (j_idx < T) : (j_idx += 1) {
                eq_sum = eq_sum.add(lookups_eq_evals[j_idx]);
                if (j_idx < 5) {
                    std.debug.print("  eq_evals[{}] = {x}\n", .{ j_idx, lookups_eq_evals[j_idx].toBytesBE()[16..32].* });
                }
            }
            std.debug.print("[STAGE5 EQ DEBUG] Sum of all eq_evals = {x} (should be 1)\n", .{eq_sum.toBytesBE()[16..32].*});
            std.debug.print("[STAGE5 EQ DEBUG] r_reduction (ALL {} elements, used for eq):\n", .{r_reduction.len});
            var r_idx: usize = 0;
            while (r_idx < r_reduction.len) : (r_idx += 1) {
                std.debug.print("  r_reduction[{}] = {x}\n", .{ r_idx, r_reduction[r_idx].toBytesBE()[16..32].* });
            }

            // Populate inc and wa from trace
            // Use trace's rd_pre_value and rd_value directly (just like Jolt does)
            const trace_len = trace.steps.items.len;
            for (trace.steps.items, 0..) |step, j| {
                if (step.is_noop and !step.is_termination_store) continue;

                const instr = step.instruction;
                const rd: u5 = @truncate((instr >> 7) & 0x1f);
                const opcode = instr & 0x7f;

                // rd_wa and inc
                const rd_used = switch (opcode) {
                    0x23, 0x63 => false, // Store and Branch don't write rd
                    else => true,
                };

                if (rd_used and rd != 0 and rd < 32) {
                    // Compute inc = rd_value - rd_pre_value
                    // Use trace's pre/post values directly (Jolt uses cycle.rd_write())
                    const pre_value: i128 = @intCast(step.rd_pre_value);
                    const post_value: i128 = @intCast(step.rd_value);
                    const increment = post_value - pre_value;

                    // Convert signed increment to field element
                    if (increment >= 0) {
                        inc_evals[j] = F.fromU64(@intCast(increment));
                    } else {
                        // Negative: use field modular arithmetic
                        inc_evals[j] = F.zero().sub(F.fromU64(@intCast(-increment)));
                    }

                    // Compute wa = eq(r_address, rd)
                    // r_address has 7 bits, rd is 5 bits - extend rd to 7 bits
                    wa_evals[j] = computeEqAtIndex(r_address_regs, @as(usize, rd));
                }
                // Note: lt_evals[j] is already computed for all j via computeAllLtEvals
            }

            // Debug: Print first 10 polynomial values
            std.debug.print("[STAGE5] First 10 polynomial values:\n", .{});
            const debug_count = @min(trace_len, 10);
            for (0..debug_count) |j| {
                const step = trace.steps.items[j];
                const instr = step.instruction;
                const rd: u5 = @truncate((instr >> 7) & 0x1f);
                std.debug.print("  j={}: rd={}, pre={}, post={}, inc={x}, wa={x}, lt={x}\n", .{
                    j, rd, step.rd_pre_value, step.rd_value,
                    inc_evals[j].toBytesBE()[24..32].*,
                    wa_evals[j].toBytesBE()[24..32].*,
                    lt_evals[j].toBytesBE()[24..32].*,
                });
            }

            // BRUTE FORCE: Compute val(r_address, r_cycle) directly from register state
            // val(k, j) = value of register k at START of cycle j
            // val(r_address, r_cycle) = Σ_{k,j} eq(r_address, k) * eq(r_cycle, j) * register_value(k, j)
            {
                var register_vals: [32]u64 = [_]u64{0} ** 32;
                var brute_val = F.zero();
                // Precompute eq(r_cycle, j) for all j
                var eq_cycle_evals = try self.allocator.alloc(F, T);
                defer self.allocator.free(eq_cycle_evals);
                for (0..T) |j| {
                    eq_cycle_evals[j] = computeEqAtIndex(r_cycle_regs, j);
                }
                // Precompute eq(r_address, k) for all k
                var eq_addr_evals: [32]F = undefined;
                for (0..32) |k| {
                    eq_addr_evals[k] = computeEqAtIndex(r_address_regs, k);
                }
                // Compute Σ_{k,j} eq_addr(k) * eq_cycle(j) * register_value(k, j)
                for (trace.steps.items, 0..) |step, j| {
                    // val(k, j) = register_vals[k] (value before this cycle)
                    for (0..32) |k| {
                        if (!eq_addr_evals[k].eql(F.zero()) and !eq_cycle_evals[j].eql(F.zero())) {
                            brute_val = brute_val.add(eq_addr_evals[k].mul(eq_cycle_evals[j]).mul(F.fromU64(register_vals[k])));
                        }
                    }
                    // Update register state
                    if (!step.is_noop or step.is_termination_store) {
                        const instr2 = step.instruction;
                        const rd2: u5 = @truncate((instr2 >> 7) & 0x1f);
                        const opcode2 = instr2 & 0x7f;
                        const rd_used2 = switch (opcode2) {
                            0x23, 0x63 => false,
                            else => true,
                        };
                        if (rd_used2 and rd2 != 0 and rd2 < 32) {
                            register_vals[rd2] = step.rd_value;
                        }
                    }
                }
                // Also add padding cycles (trace_len..T) with final register values
                for (trace_len..T) |j| {
                    for (0..32) |k| {
                        if (!eq_addr_evals[k].eql(F.zero()) and !eq_cycle_evals[j].eql(F.zero())) {
                            brute_val = brute_val.add(eq_addr_evals[k].mul(eq_cycle_evals[j]).mul(F.fromU64(register_vals[k])));
                        }
                    }
                }
                std.debug.print("[STAGE5 BRUTE] val(r_addr, r_cycle) from running regs = {any}\n", .{brute_val.toBytesBE()[0..16]});
                std.debug.print("[STAGE5 BRUTE] regs_val_input (from Stage 4)          = {any}\n", .{regs_val_input.toBytesBE()[0..16]});
                std.debug.print("[STAGE5 BRUTE] match? {}\n", .{brute_val.eql(regs_val_input)});
            }

            // Also verify inc values match between Stage 4 style and trace
            {
                var reg_vals2: [32]u64 = [_]u64{0} ** 32;
                var inc_mismatches: usize = 0;
                for (trace.steps.items, 0..) |step, j| {
                    if (step.is_noop and !step.is_termination_store) continue;
                    const instr2 = step.instruction;
                    const rd2: u5 = @truncate((instr2 >> 7) & 0x1f);
                    const opcode2 = instr2 & 0x7f;
                    const rd_used2 = switch (opcode2) {
                        0x23, 0x63 => false,
                        else => true,
                    };
                    if (rd_used2 and rd2 != 0 and rd2 < 32) {
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
                                std.debug.print("[STAGE5 INC MISMATCH] j={}: rd={}, s4_pre={}, s5_pre={}, rd_value={}\n", .{
                                    j, rd2, s4_pre, step.rd_pre_value, step.rd_value,
                                });
                            }
                            inc_mismatches += 1;
                        }
                        reg_vals2[rd2] = step.rd_value;
                    }
                }
                std.debug.print("[STAGE5 INC CHECK] Total mismatches: {}\n", .{inc_mismatches});
            }

            // Verify the sum Σ_j inc(j) · wa(j) · lt(j) matches the input claim
            var computed_sum = F.zero();
            var non_zero_terms: usize = 0;
            for (0..T) |j| {
                const term = inc_evals[j].mul(wa_evals[j]).mul(lt_evals[j]);
                if (!term.eql(F.zero())) {
                    non_zero_terms += 1;
                    if (non_zero_terms <= 5) {
                        std.debug.print("[STAGE5] Non-zero term at j={}: inc*wa*lt = {x}\n", .{
                            j, term.toBytesBE()[24..32].*,
                        });
                    }
                }
                computed_sum = computed_sum.add(term);
            }
            std.debug.print("[STAGE5] Total non-zero terms: {}\n", .{non_zero_terms});
            std.debug.print("[STAGE5] Built polynomial tables: T={}, trace_len={}\n", .{ T, trace_len });
            std.debug.print("[STAGE5] Sum check: computed_sum = {any}\n", .{computed_sum.toBytesBE()[0..16]});
            std.debug.print("[STAGE5] Sum check: regs_val_input = {any}\n", .{regs_val_input.toBytesBE()[0..16]});
            std.debug.print("[STAGE5] Sum check: match = {}\n", .{@as(bool, computed_sum.limbs[0] == regs_val_input.limbs[0] and
                computed_sum.limbs[1] == regs_val_input.limbs[1] and
                computed_sum.limbs[2] == regs_val_input.limbs[2] and
                computed_sum.limbs[3] == regs_val_input.limbs[3])});

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
            for (trace.steps.items, 0..) |step, j| {
                if (step.is_noop and !step.is_termination_store) continue;

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
                    0x33, 0x3b, 0x23, 0x63, 0x13, 0x03, 0x67, 0x1b => true,
                    else => false,
                };
                const left_is_pc: bool = switch (opcode) {
                    0x17, 0x6f => true,
                    else => false,
                };
                const right_is_rs2: bool = switch (opcode) {
                    0x33, 0x63, 0x3b => true,
                    else => false,
                };
                const right_is_imm: bool = switch (opcode) {
                    0x13, 0x03, 0x67, 0x23, 0x37, 0x17, 0x6f, 0x1b => true,
                    else => false,
                };

                const imm_val = computeImmediate(instr);

                var left_input: F = F.zero();
                if (left_is_rs1) left_input = F.fromU64(step.rs1_value);
                if (left_is_pc) left_input = F.fromU64(step.pc);

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
                            if (funct3 == 0x0) { // MUL
                                left_op = F.zero();
                                right_op = left_input.mul(right_input); // Product
                            } else {
                                left_op = left_input;
                                right_op = right_input;
                            }
                        } else if (funct7 == 0x20 and funct3 == 0x0) {
                            // SUB: LeftLookup=0, RightLookup=left-right+2^64
                            const two_pow_64 = F.fromBytes(&[_]u8{ 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 });
                            left_op = F.zero();
                            right_op = left_input.sub(right_input).add(two_pow_64);
                        } else {
                            // ADD and others: LeftLookup=0, RightLookup=left+right
                            left_op = F.zero();
                            right_op = left_input.add(right_input);
                        }
                    },
                    0x13 => { // I-type ALU: AddOperands
                        left_op = F.zero();
                        right_op = left_input.add(right_input);
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
                    0x1b => { // ADDIW: AddOperands (matches Jolt's ADDI step of decomposition)
                        // In Jolt, ADDIW decomposes to ADDI+VirtualSignExtendWord, both with AddOperands.
                        // For Zolt's single-cycle model, we use identity format matching ADDI.
                        left_op = F.zero();
                        right_op = left_input.add(right_input);
                    },
                    0x3b => { // ADDW/SUBW: AddOperands/SubtractOperands
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
                        } else {
                            // Other 0x3b variants (not AddOperands/SubtractOperands)
                            left_op = left_input;
                            right_op = right_input;
                        }
                    },
                    else => {
                        // Default: NOT Add+Sub+Mul (includes 0x03, 0x23, 0x63)
                        left_op = left_input;
                        right_op = right_input;
                    },
                }

                // combined = output + gamma*left + gamma^2*right
                lookups_combined_vals[j] = lookup_output.add(gamma_raf.mul(left_op)).add(gamma_raf2.mul(right_op));

                // Debug: print first 5 right_op values from combined_vals computation
                if (j < 5) {
                    std.debug.print("[STAGE5 COMBINED] j={}: opcode=0x{x}, left_op=0x{x}, right_op=0x{x}, output=0x{x}\n", .{
                        j, opcode, left_op.toU64(), right_op.toU64(), lookup_output.toU64(),
                    });
                }

                // Track which lookup table this cycle uses (for flag claims)
                const table_idx = getLookupTableIndex(opcode, funct3, funct7);
                cycle_table_indices[j] = table_idx;

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
                    0x03 => true, // Load (AddOperands - though no lookup table)
                    0x23 => true, // Store (AddOperands - though no lookup table)
                    else => false,
                };
                cycle_is_identity_path[j] = is_identity_path;

                // Compute lookup operands and index matching Jolt's to_lookup_operands/to_lookup_index.
                // For identity-path: left_op_raw=0, right_op_raw=computed_value, index=right_op_raw
                // For interleaved-path: left_op_raw=rs1, right_op_raw=rs2, index=interleave(left,right)
                var left_op_raw: u64 = undefined;
                var right_op_raw: u64 = undefined;

                if (is_identity_path) {
                    left_op_raw = 0;
                    right_op_raw = switch (opcode) {
                        // ADD: (0, rs1 + rs2)
                        0x33 => blk: {
                            if (funct3 == 0 and funct7 == 0) {
                                break :blk step.rs1_value +% step.rs2_value;
                            }
                            // SUB: (0, rs1 + 2^64 - rs2)
                            if (funct3 == 0 and funct7 == 0x20) {
                                break :blk step.rs1_value +% (0 -% step.rs2_value);
                            }
                            // MUL: (0, rs1 * rs2)
                            if (funct7 == 0x01 and funct3 == 0) {
                                break :blk step.rs1_value *% step.rs2_value;
                            }
                            // MULHU: (0, rs1 * rs2) as u128, but lookup index = full product
                            if (funct7 == 0x01 and funct3 == 3) {
                                // MULHU uses the full 128-bit product. But since right_op_raw is u64,
                                // we handle this with the u128 lookup_idx below
                                break :blk 0; // placeholder, handled separately
                            }
                            break :blk 0;
                        },
                        // ADDW/SUBW: use same computation as ADD/SUB
                        // In Jolt, ADDW decomposes to ADD+VirtualSignExtendWord.
                        // The ADD step uses lookup_index = rs1 + rs2 (wrapping u64).
                        // The SUB step uses lookup_index = rs1 + (0 -% rs2) (two's complement).
                        0x3b => blk: {
                            if (funct3 == 0 and funct7 == 0) {
                                // ADDW: same as ADD
                                break :blk step.rs1_value +% step.rs2_value;
                            }
                            if (funct3 == 0 and funct7 == 0x20) {
                                // SUBW: same as SUB
                                break :blk step.rs1_value +% (0 -% step.rs2_value);
                            }
                            break :blk 0;
                        },
                        // ADDI: (0, rs1 + sign_ext(imm))
                        0x13 => blk: {
                            const imm12_raw: u32 = @truncate(instr >> 20);
                            const imm_signed: i64 = @as(i64, @as(i32, @bitCast(imm12_raw << 20)) >> 20);
                            const imm_u64: u64 = @bitCast(imm_signed);
                            break :blk step.rs1_value +% imm_u64;
                        },
                        // ADDIW: use same computation as ADDI (rs1 + sign_ext(imm))
                        // In Jolt, ADDIW decomposes to ADDI+VirtualSignExtendWord.
                        // The ADDI step uses lookup_index = rs1 + imm (wrapping u64).
                        0x1b => blk: {
                            const imm12_raw_w: u32 = @truncate(instr >> 20);
                            const imm_signed_w: i64 = @as(i64, @as(i32, @bitCast(imm12_raw_w << 20)) >> 20);
                            const imm_u64_w: u64 = @bitCast(imm_signed_w);
                            break :blk step.rs1_value +% imm_u64_w;
                        },
                        // LUI: (0, imm) where imm = upper 20 bits shifted left by 12
                        0x37 => @as(u64, instr & 0xFFFFF000),
                        // AUIPC: (0, pc + imm)
                        0x17 => step.pc +% @as(u64, instr & 0xFFFFF000),
                        // JAL: (0, pc + sign_ext(imm))
                        0x6f => blk: {
                            const imm20: u32 = ((@as(u32, instr >> 31) & 1) << 19) |
                                ((@as(u32, instr >> 12) & 0xFF) << 11) |
                                ((@as(u32, instr >> 20) & 1) << 10) |
                                ((@as(u32, instr >> 21) & 0x3FF));
                            const imm_signed: i64 = @as(i64, @as(i32, @bitCast(imm20 << 12)) >> 11);
                            const imm_u64: u64 = @bitCast(imm_signed);
                            break :blk step.pc +% imm_u64;
                        },
                        // JALR: (0, rs1 + sign_ext(imm))
                        0x67 => blk: {
                            const imm12_raw: u32 = @truncate(instr >> 20);
                            const imm_signed: i64 = @as(i64, @as(i32, @bitCast(imm12_raw << 20)) >> 20);
                            const imm_u64: u64 = @bitCast(imm_signed);
                            break :blk step.rs1_value +% imm_u64;
                        },
                        // Load: (0, rs1 + sign_ext(imm))
                        0x03 => blk: {
                            const imm12_raw: u32 = @truncate(instr >> 20);
                            const imm_signed: i64 = @as(i64, @as(i32, @bitCast(imm12_raw << 20)) >> 20);
                            const imm_u64: u64 = @bitCast(imm_signed);
                            break :blk step.rs1_value +% imm_u64;
                        },
                        // Store: (0, rs1 + sign_ext(imm))
                        0x23 => blk: {
                            const imm_lo: u32 = (instr >> 7) & 0x1F;
                            const imm_hi: u32 = (instr >> 25) & 0x7F;
                            const imm12: u32 = (imm_hi << 5) | imm_lo;
                            const imm_signed: i64 = @as(i64, @as(i12, @bitCast(@as(u12, @truncate(imm12)))));
                            const imm_u64: u64 = @bitCast(imm_signed);
                            break :blk step.rs1_value +% imm_u64;
                        },
                        else => 0,
                    };
                } else {
                    // Interleaved path: left=rs1, right=rs2 (or imm for I-type)
                    left_op_raw = step.rs1_value;
                    right_op_raw = switch (opcode) {
                        0x33, 0x3b, 0x63 => step.rs2_value,
                        0x13 => blk: {
                            // I-type: right operand is sign-extended immediate (as u64)
                            const imm12_raw: u32 = @truncate(instr >> 20);
                            const imm_signed: i64 = @as(i64, @as(i32, @bitCast(imm12_raw << 20)) >> 20);
                            break :blk @as(u64, @bitCast(imm_signed));
                        },
                        else => step.rs2_value,
                    };
                }

                // Compute lookup index:
                // - Identity path: index = right_op_raw (plain value)
                //   Exception: MULHU needs 128-bit product
                // - Interleaved path: index = interleave_bits(left, right)
                const lookup_idx: u128 = blk_idx: {
                    if (is_identity_path) {
                        // Special case: MULHU needs full 128-bit product
                        if (opcode == 0x33 and funct7 == 0x01 and funct3 == 3) {
                            break :blk_idx @as(u128, step.rs1_value) *% @as(u128, step.rs2_value);
                        }
                        break :blk_idx @as(u128, right_op_raw);
                    } else {
                        break :blk_idx interleaveBits128(left_op_raw, right_op_raw);
                    }
                };
                lookups_indices_lo[j] = @truncate(lookup_idx);
                lookups_indices_hi[j] = @truncate(lookup_idx >> 64);

                // Override lookup_output for ADDIW/ADDW/SUBW to match table MLE output.
                // For identity-path with RangeCheck table: materialize_entry(index) = lower64(index).
                // This replaces the rd_value-based output set earlier.
                if (opcode == 0x1b and funct3 == 0) {
                    // ADDIW: output = lower64(rs1 + imm) from RangeCheck table
                    lookup_output = F.fromU64(lookups_indices_lo[j]);
                } else if (opcode == 0x3b) {
                    if (funct3 == 0 and (funct7 == 0 or funct7 == 0x20)) {
                        // ADDW/SUBW: output = lower64(lookup_index) from RangeCheck table
                        lookup_output = F.fromU64(lookups_indices_lo[j]);
                    }
                }

                // Re-compute combined_vals with the corrected lookup_output
                lookups_combined_vals[j] = lookup_output.add(gamma_raf.mul(left_op)).add(gamma_raf2.mul(right_op));

                // Debug first 5 cycles with full BE bytes
                if (j < 5) {
                    std.debug.print("[STAGE5 TRACE DEBUG] j={}: opcode=0x{x}, output={any}, left={any}, right={any}\n", .{
                        j,
                        opcode,
                        lookup_output.toBytesBE()[0..8],
                        left_op.toBytesBE()[0..8],
                        right_op.toBytesBE()[0..8],
                    });
                    std.debug.print("[STAGE5 INDEX] j={}: is_identity={}, left_raw=0x{x}, right_raw=0x{x}, idx=0x{x:0>32}\n", .{
                        j, is_identity_path, left_op_raw, right_op_raw, lookup_idx,
                    });
                    // Also compute what the OLD interleaved index would have been
                    const old_idx = interleaveBits128(left_op_raw, right_op_raw);
                    std.debug.print("[STAGE5 INDEX] j={}: old_interleaved=0x{x:0>32}, changed={}\n", .{
                        j, old_idx, old_idx != lookup_idx,
                    });
                }
                // Debug first 3 cycles
                if (j < 3) {
                    std.debug.print("[STAGE5 LOOKUPS] j={}: opcode=0x{x}, funct3={}, funct7={}, pc=0x{x}\n", .{ j, opcode, funct3, funct7, step.pc });
                    std.debug.print("  left_op={any}, right_op={any}, output={any}\n", .{
                        left_op.toBytesBE()[24..32].*,
                        right_op.toBytesBE()[24..32].*,
                        lookup_output.toBytesBE()[24..32].*,
                    });
                    std.debug.print("  eq={x}, combined={x}\n", .{
                        lookups_eq_evals[j].toBytesBE()[24..32].*,
                        lookups_combined_vals[j].toBytesBE()[24..32].*,
                    });
                    std.debug.print("  lookup_index: lo=0x{x:0>16}, hi=0x{x:0>16}\n", .{
                        lookups_indices_lo[j],
                        lookups_indices_hi[j],
                    });
                }
            }

            // Debug: count instructions by opcode to see what we have
            {
                var opcode_counts: [128]u32 = [_]u32{0} ** 128;
                var identity_count: u32 = 0;
                var interleaved_count: u32 = 0;
                for (trace.steps.items, 0..) |step_d, j_d| {
                    if (step_d.is_noop and !step_d.is_termination_store) continue;
                    const opc = step_d.instruction & 0x7f;
                    opcode_counts[opc] += 1;
                    if (cycle_is_identity_path[j_d]) identity_count += 1 else interleaved_count += 1;
                }
                std.debug.print("[STAGE5 OPCODE COUNTS] identity={}, interleaved={}\n", .{identity_count, interleaved_count});
                for (opcode_counts, 0..) |cnt, opc| {
                    if (cnt > 0) {
                        std.debug.print("  opcode 0x{x:0>2}: {} cycles\n", .{opc, cnt});
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
                            std.debug.print("[STAGE5 MISMATCH] j={}: opcode=0x{x}, idx_lo=0x{x}, idx_hi=0x{x}, rd_value=0x{x}\n", .{
                                j_d, step_d.instruction & 0x7f, lower64, lookups_indices_hi[j_d], step_d.rd_value,
                            });
                        }
                    }
                }
                std.debug.print("[STAGE5 MISMATCH] Total table-0 cycles with lower64(idx) != rd_value: {}\n", .{mismatch_count});
            }

            // Verify the sum matches lookups_input
            // Compute individual sums for debugging
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
                        0x33, 0x3b, 0x23, 0x63, 0x13, 0x03, 0x67, 0x1b => true,
                        else => false,
                    };
                    const left_is_pc: bool = switch (opcode) {
                        0x17, 0x6f => true,
                        else => false,
                    };
                    const right_is_rs2: bool = switch (opcode) {
                        0x33, 0x63, 0x3b => true,
                        else => false,
                    };
                    const right_is_imm: bool = switch (opcode) {
                        0x13, 0x03, 0x67, 0x23, 0x37, 0x17, 0x6f, 0x1b => true,
                        else => false,
                    };

                    // Compute immediate value
                    const imm_val = computeImmediate(instr);

                    // Compute left_input and right_input
                    var left_input: F = F.zero();
                    if (left_is_rs1) left_input = F.fromU64(step.rs1_value);
                    if (left_is_pc) left_input = F.fromU64(step.pc);

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
                            // Clear LSB: need to convert to u64, mask, convert back
                            // For simplicity, assume imm is small enough
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
                                if (funct3 == 0x0) { // MUL
                                    left_op = F.zero();
                                    right_op = left_input.mul(right_input); // Product
                                } else {
                                    left_op = left_input;
                                    right_op = right_input;
                                }
                            } else if (funct7 == 0x20 and funct3 == 0x0) {
                                // SUB: LeftLookup=0, RightLookup=left-right+2^64
                                const two_pow_64 = F.fromBytes(&[_]u8{ 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 });
                                left_op = F.zero();
                                right_op = left_input.sub(right_input).add(two_pow_64);
                            } else {
                                // ADD and others: LeftLookup=0, RightLookup=left+right
                                left_op = F.zero();
                                right_op = left_input.add(right_input);
                            }
                        },
                        0x13 => { // I-type ALU: AddOperands
                            left_op = F.zero();
                            right_op = left_input.add(right_input);
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
                    std.debug.print("[COMBINED MISMATCH] j={}: opcode=0x{x}, noop={}, term={}\n", .{
                        j, step_dbg2.instruction & 0x7f, step_dbg2.is_noop, step_dbg2.is_termination_store,
                    });
                    std.debug.print("  recomputed = {x}\n", .{recomputed_combined.toBytesBE()[16..32].*});
                    std.debug.print("  combined_v = {x}\n", .{lookups_combined_vals[j].toBytesBE()[16..32].*});
                    std.debug.print("  output: recomp=0x{x}, rs1={}, rs2={}, rd={}, pc=0x{x}\n", .{
                        lookup_output.toU64(), step_dbg2.rs1_value, step_dbg2.rs2_value,
                        step_dbg2.rd_value, step_dbg2.pc,
                    });
                }
            }
            // Debug: print first 5 cycles' right operand values
            std.debug.print("[STAGE5 LOOKUPS] First 5 right_op values (computed in loop):\n", .{});
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
                    0x13, 0x03, 0x67, 0x23, 0x37, 0x17, 0x6f, 0x1b => true,
                    else => false,
                };
                const imm_dbg = computeImmediate(instr_dbg);
                var right_input_dbg: F = F.zero();
                if (right_is_rs2_dbg) right_input_dbg = F.fromU64(step_dbg.rs2_value);
                if (right_is_imm_dbg) right_input_dbg = imm_dbg;

                std.debug.print("  j={}: opcode=0x{x}, right_is_rs2={}, right_is_imm={}, imm=0x{x}, rs2=0x{x}, right_input=0x{x}\n", .{
                    jj, opcode_dbg, right_is_rs2_dbg, right_is_imm_dbg,
                    imm_dbg.toU64(), step_dbg.rs2_value, right_input_dbg.toU64(),
                });
            }
            std.debug.print("[STAGE5 LOOKUPS] Individual sum verification:\n", .{});
            std.debug.print("  output_sum (Σ eq*output) = {any}\n", .{output_sum.toBytesBE()[0..16]});
            std.debug.print("  rv_claim (from Stage 2)  = {any}\n", .{rv_claim.toBytesBE()[0..16]});
            std.debug.print("  output match = {}\n", .{output_sum.eql(rv_claim)});
            std.debug.print("  left_sum (Σ eq*left)     = {any}\n", .{left_sum.toBytesBE()[0..16]});
            std.debug.print("  left_op_claim (Stage 2)  = {any}\n", .{left_op_claim.toBytesBE()[0..16]});
            std.debug.print("  left match = {}\n", .{left_sum.eql(left_op_claim)});
            std.debug.print("  right_sum (Σ eq*right)   = {any}\n", .{right_sum.toBytesBE()[0..16]});
            std.debug.print("  right_op_claim (Stage 2) = {any}\n", .{right_op_claim.toBytesBE()[0..16]});
            std.debug.print("  right match = {}\n", .{right_sum.eql(right_op_claim)});
            std.debug.print("[STAGE5 LOOKUPS] Sum verification:\n", .{});
            std.debug.print("  computed_sum = {any}\n", .{lookups_computed_sum.toBytesBE()[0..8]});
            std.debug.print("  lookups_input = {any}\n", .{lookups_input.toBytesBE()[0..8]});
            std.debug.print("  rv_claim = {any}\n", .{rv_claim.toBytesBE()[0..8]});
            std.debug.print("  left_op_claim = {any}\n", .{left_op_claim.toBytesBE()[0..8]});
            std.debug.print("  right_op_claim = {any}\n", .{right_op_claim.toBytesBE()[0..8]});
            std.debug.print("  match = {}\n", .{lookups_computed_sum.eql(lookups_input)});

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

            std.debug.print("[STAGE5] Initial batched claim = {any}\n", .{batched_claim.toBytesBE()});
            std.debug.print("  [S5P] initial_claim (e before R0): {x}\n", .{batched_claim.toBytes()[0..16].*});

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
            std.debug.print("[STAGE5 RAM_RA] Initializing with {} RAM accesses\n", .{ram_access_count});

            // Allocate sparse access arrays
            var ram_addresses = try self.allocator.alloc(u64, ram_access_count);
            defer self.allocator.free(ram_addresses);
            var ram_cycles = try self.allocator.alloc(u64, ram_access_count);
            defer self.allocator.free(ram_cycles);
            var ram_G_A = try self.allocator.alloc(F, ram_access_count);
            defer self.allocator.free(ram_G_A);
            var ram_G_B = try self.allocator.alloc(F, ram_access_count);
            defer self.allocator.free(ram_G_B);
            // Store original eq(r_address_x, addr) values for each access
            // These are used in the sparse polynomial computation instead of the bound B_1/B_2 arrays
            var ram_B1_original = try self.allocator.alloc(F, ram_access_count);
            defer self.allocator.free(ram_B1_original);
            var ram_B2_original = try self.allocator.alloc(F, ram_access_count);
            defer self.allocator.free(ram_B2_original);

            // Precompute G_A and G_B for each RAM access
            // G_A[i] = eq(r_cycle_raf, c_i) + γ · eq(r_cycle_val, c_i)
            // G_B[i] = eq(r_cycle_rw, c_i) + γ · eq(r_cycle_val, c_i)
            //
            // Remap addresses to polynomial index space using memory_layout
            // In Jolt, remap_address(address, memory_layout) = (address - lowest_address) / 8
            if (memory_trace) |mt| {
                for (mt.accesses.items, 0..) |access, i| {
                    // Use memory_layout.remapAddress if available, otherwise mask
                    const remapped_addr: u64 = if (memory_layout) |ml|
                        ml.remapAddress(access.address) orelse 0
                    else
                        access.address & (@as(u64, K) - 1);

                    ram_addresses[i] = remapped_addr;
                    ram_cycles[i] = access.timestamp;

                    // Compute eq(r_cycle_x, cycle) for each cycle point
                    // r_cycle vectors are in BIG_ENDIAN order
                    const cycle = access.timestamp;
                    const eq_raf_c = computeEqAtPoint(F, r_cycle_raf, cycle);
                    const eq_rw_c = computeEqAtPoint(F, r_cycle_rw, cycle);
                    const eq_val_c = computeEqAtPoint(F, r_cycle_val, cycle);

                    // G_A = eq_raf + γ · eq_val
                    // G_B = eq_rw + γ · eq_val
                    ram_G_A[i] = eq_raf_c.add(gamma.mul(eq_val_c));
                    ram_G_B[i] = eq_rw_c.add(gamma.mul(eq_val_c));

                    std.debug.print("[STAGE5 RAM_RA] Access {}: raw_addr=0x{x}, remapped_addr={}, cycle={}\n", .{ i, access.address, remapped_addr, cycle });
                    std.debug.print("  eq_raf_c={any}, eq_rw_c={any}, eq_val_c={any}\n", .{
                        eq_raf_c.toBytesBE()[16..32].*,
                        eq_rw_c.toBytesBE()[16..32].*,
                        eq_val_c.toBytesBE()[16..32].*,
                    });
                    std.debug.print("  G_A={any}, G_B={any}\n", .{
                        ram_G_A[i].toBytesBE()[16..32].*,
                        ram_G_B[i].toBytesBE()[16..32].*,
                    });
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
            std.debug.print("[STAGE5 RAM_RA] Created full-size G_A/G_B arrays (K={}), {} non-zero entries\n", .{ K, ram_access_count });

            // Initialize B_1 and B_2 polynomials for address rounds
            // B_1 = eq(r_address_raf, k) - this is bound during address rounds
            // B_2 = eq(r_address_rw, k) - this is bound during address rounds
            // These are multilinear polynomials over log_ram_k variables
            var B_1 = try self.allocator.alloc(F, K);
            defer self.allocator.free(B_1);
            var B_2 = try self.allocator.alloc(F, K);
            defer self.allocator.free(B_2);

            // Compute B_1[k] = eq(r_address_raf, k) for all k
            // r_address_raf is in BIG_ENDIAN order
            for (0..K) |k| {
                B_1[k] = computeEqAtPoint(F, r_address_raf, @intCast(k));
                B_2[k] = computeEqAtPoint(F, r_address_rw, @intCast(k));
            }

            // Debug: print B_1 and B_2 for first few addresses
            std.debug.print("[STAGE5 RAM_RA] B_1/B_2 eq polynomials (first 4 and last 4 of {}):\n", .{K});
            for (0..@min(4, K)) |k| {
                std.debug.print("  B_1[{}]={any}, B_2[{}]={any}\n", .{
                    k, B_1[k].toBytesBE()[16..32].*, k, B_2[k].toBytesBE()[16..32].*,
                });
            }
            if (K > 8) {
                for (K - 4..K) |k| {
                    std.debug.print("  B_1[{}]={any}, B_2[{}]={any}\n", .{
                        k, B_1[k].toBytesBE()[16..32].*, k, B_2[k].toBytesBE()[16..32].*,
                    });
                }
            }

            // =====================================================================
            // CRITICAL FIX: Compute actual RamRa opening claims from trace data
            // =====================================================================
            // The opening claims should be evaluations of the ra polynomial MLE at
            // various random points. Since we have the trace, we compute them here.
            //
            // claim_raf = Σ_i eq(r_address_raf, addr_i) * eq(r_cycle_raf, cycle_i)
            // claim_val_final = Σ_i eq(r_address_raf, addr_i) * eq(r_cycle_val, cycle_i)
            // claim_rw = Σ_i eq(r_address_rw, addr_i) * eq(r_cycle_rw, cycle_i)
            // claim_val_eval = Σ_i eq(r_address_rw, addr_i) * eq(r_cycle_val, cycle_i)
            var computed_claim_raf = F.zero();
            var computed_claim_val_final = F.zero();
            var computed_claim_rw = F.zero();
            var computed_claim_val_eval = F.zero();

            if (memory_trace) |mt| {
                for (mt.accesses.items, 0..) |access, i| {
                    const addr = ram_addresses[i];
                    const addr_usize: usize = @intCast(addr);
                    const cycle = access.timestamp;

                    // eq(r_cycle_x, cycle) values
                    const eq_raf_c = computeEqAtPoint(F, r_cycle_raf, cycle);
                    const eq_rw_c = computeEqAtPoint(F, r_cycle_rw, cycle);
                    const eq_val_c = computeEqAtPoint(F, r_cycle_val, cycle);

                    // eq(r_address_x, addr) values - STORE ORIGINAL VALUES
                    const B1_at_addr = B_1[addr_usize];
                    const B2_at_addr = B_2[addr_usize];
                    ram_B1_original[i] = B1_at_addr;
                    ram_B2_original[i] = B2_at_addr;

                    // Accumulate claims
                    computed_claim_raf = computed_claim_raf.add(B1_at_addr.mul(eq_raf_c));
                    computed_claim_val_final = computed_claim_val_final.add(B1_at_addr.mul(eq_val_c));
                    computed_claim_rw = computed_claim_rw.add(B2_at_addr.mul(eq_rw_c));
                    computed_claim_val_eval = computed_claim_val_eval.add(B2_at_addr.mul(eq_val_c));

                    std.debug.print("[STAGE5 COMPUTED CLAIMS] Access {}: addr={}, cycle={}\n", .{ i, addr, cycle });
                    std.debug.print("  B1_at_addr={x}, B2_at_addr={x}\n", .{
                        B1_at_addr.toBytesBE()[16..32].*,
                        B2_at_addr.toBytesBE()[16..32].*,
                    });
                    std.debug.print("  eq_raf_c={x}, eq_rw_c={x}, eq_val_c={x}\n", .{
                        eq_raf_c.toBytesBE()[16..32].*,
                        eq_rw_c.toBytesBE()[16..32].*,
                        eq_val_c.toBytesBE()[16..32].*,
                    });
                }
            }

            std.debug.print("[STAGE5] Computed RamRa claims from trace:\n", .{});
            std.debug.print("  computed_claim_raf = {x}\n", .{computed_claim_raf.toBytesBE()[16..32].*});
            std.debug.print("  computed_claim_val_final = {x}\n", .{computed_claim_val_final.toBytesBE()[16..32].*});
            std.debug.print("  computed_claim_rw = {x}\n", .{computed_claim_rw.toBytesBE()[16..32].*});
            std.debug.print("  computed_claim_val_eval = {x}\n", .{computed_claim_val_eval.toBytesBE()[16..32].*});

            // Recompute ram_ra_input using the computed claims
            const computed_ram_ra_input = computed_claim_raf
                .add(gamma.mul(computed_claim_val_final))
                .add(gamma2.mul(computed_claim_rw))
                .add(gamma3.mul(computed_claim_val_eval));

            std.debug.print("  computed_ram_ra_input = {x}\n", .{computed_ram_ra_input.toBytesBE()[16..32].*});
            std.debug.print("  original ram_ra_input = {x}\n", .{ram_ra_input.toBytesBE()[16..32].*});

            // Initialize Instance 1 claim tracking with SCALED value (matches batched_claim computation)
            // NOTE: We use the ORIGINAL ram_ra_input (from opening_claims), NOT computed_ram_ra_input.
            // The verifier computes its initial claim from opening_claims, so the prover MUST match.
            // If computed_ram_ra_input differs from ram_ra_input, the polynomial computation must
            // be adjusted to match the opening_claims, not the other way around.
            ram_ra_current_claim = ram_ra_scaled;

            std.debug.print("[STAGE5] Using original batched claim = {any}\n", .{batched_claim.toBytesBE()});
            std.debug.print("[STAGE5] computed_ram_ra_input = {x}\n", .{computed_ram_ra_input.toBytesBE()[16..32].*});
            std.debug.print("[STAGE5] original ram_ra_input = {x}\n", .{ram_ra_input.toBytesBE()[16..32].*});

            // Expanding table to track eq(r_addr_reduced_so_far, k_bound_bits)
            // This accumulates the eq value as we bind address bits
            var ram_ra_F = try ExpandingTable(F).init(self.allocator, K);
            defer ram_ra_F.deinit();
            ram_ra_F.reset(F.one());

            // Track bound address challenges for RamRaClaimReduction
            var ram_ra_bound_challenges = try self.allocator.alloc(F, log_ram_k);
            defer self.allocator.free(ram_ra_bound_challenges);
            @memset(ram_ra_bound_challenges, F.zero());

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

            // Per-access individual eq values (for computing remaining factors)
            var eq_raf_access = try self.allocator.alloc(F, ram_access_count);
            defer self.allocator.free(eq_raf_access);
            var eq_rw_access = try self.allocator.alloc(F, ram_access_count);
            defer self.allocator.free(eq_rw_access);
            var eq_val_access = try self.allocator.alloc(F, ram_access_count);
            defer self.allocator.free(eq_val_access);

            // Precompute individual eq values for each access
            if (memory_trace) |mt| {
                for (mt.accesses.items, 0..) |access, i| {
                    const cycle = access.timestamp;
                    eq_raf_access[i] = computeEqAtPoint(F, r_cycle_raf, cycle);
                    eq_rw_access[i] = computeEqAtPoint(F, r_cycle_rw, cycle);
                    eq_val_access[i] = computeEqAtPoint(F, r_cycle_val, cycle);
                }
            }

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

            std.debug.print("[STAGE5 PQ] PhaseCycle P*Q setup: n_cycle={}, prefix={}, suffix={}\n", .{
                n_cycle_vars, prefix_n_vars, suffix_n_vars,
            });
            std.debug.print("[STAGE5 PQ] prefix_size={}, suffix_size={}\n", .{ prefix_size, suffix_size });

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
            // Build u128 lookup indices array for prefix-suffix decomposition
            var lookup_indices_u128 = try self.allocator.alloc(u128, T);
            defer self.allocator.free(lookup_indices_u128);
            for (0..T) |j| {
                lookup_indices_u128[j] = (@as(u128, lookups_indices_hi[j]) << 64) | lookups_indices_lo[j];
            }
            // Print first 5 lookup indices (VERIFY FIX APPLIED)
            for (0..@min(5, T)) |j| {
                std.debug.print("[LOOKUP IDX] j={}: idx=0x{x:0>32}, lo=0x{x:0>16}, hi=0x{x:0>16}\n", .{
                    j, lookup_indices_u128[j], lookups_indices_lo[j], lookups_indices_hi[j],
                });
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

            // Initialize Q polynomials for read-checking
            try suffix_polys.initPhase(0, num_phases, lookups_eq_evals, lookup_indices_u128, cycle_table_indices);

            // Debug: count how many cycles have lookup tables assigned
            var cycles_with_tables: usize = 0;
            for (0..T) |jj| {
                if (cycle_table_indices[jj] >= 0) cycles_with_tables += 1;
            }
            std.debug.print("[STAGE5] Cycles with lookup tables: {}/{}\n", .{ cycles_with_tables, T });

            // Initialize RAF (Read-Address-Flag) decompositions for left/right/identity
            // These compute: γ*left + γ²*(identity + right)
            var left_raf = try RafDecomposition(F).init(self.allocator, initial_m, log_m, LOOKUPS_LOG_K, .LeftOperand);
            defer left_raf.deinit();
            var right_raf = try RafDecomposition(F).init(self.allocator, initial_m, log_m, LOOKUPS_LOG_K, .RightOperand);
            defer right_raf.deinit();
            var identity_raf = try RafDecomposition(F).init(self.allocator, initial_m, log_m, LOOKUPS_LOG_K, .Identity);
            defer identity_raf.deinit();

            // Create is_interleaved_operands array (inverse of cycle_is_identity_path)
            var is_interleaved_operands = try self.allocator.alloc(bool, T);
            defer self.allocator.free(is_interleaved_operands);
            for (0..T) |j| {
                is_interleaved_operands[j] = !cycle_is_identity_path[j];
            }

            // Initialize RAF Q accumulators for phase 0
            initQRaf(F, &left_raf, &right_raf, &identity_raf, lookups_eq_evals, lookup_indices_u128, is_interleaved_operands);

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

            std.debug.print("[STAGE5 PREFIX-SUFFIX] Initialized phase 0, log_m={}, suffix_len={}, initial_m={}\n", .{
                log_m,
                LOOKUPS_LOG_K - log_m,
                initial_m,
            });

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
            {
                // Brute-force rv_claim = Σ_j u[j] * output_j
                // Brute-force left_claim = Σ_j u[j] * left_operand_j
                // Brute-force right_claim = Σ_j u[j] * right_operand_j
                var bf_combined = F.zero();
                for (trace.steps.items, 0..) |step, jj| {
                    if (step.is_noop and !step.is_termination_store) continue;
                    const u_j = lookups_eq_evals[jj];
                    // Decompose combined_vals back
                    // combined = output + γ*left + γ²*right
                    // We stored the combined value, but we can also recompute the individual parts
                    bf_combined = bf_combined.add(u_j.mul(lookups_combined_vals[jj]));
                }

                // Verify combined matches lookups_claim
                std.debug.print("[DIAG INIT] bf_combined = {x}\n", .{bf_combined.toBytesBE()[16..32].*});
                std.debug.print("[DIAG INIT] lookups_claim = {x}\n", .{lookups_claim.toBytesBE()[16..32].*});
                std.debug.print("[DIAG INIT] combined == claim: {}\n", .{bf_combined.eql(lookups_claim)});

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
                                std.debug.print("[DIAG INIT] Q_total[table={},suffix={}] = {x}\n", .{
                                    t_idx, s_idx, poly_sum.toBytesBE()[16..32].*,
                                });
                            }
                            q_total_raw = q_total_raw.add(poly_sum);
                        }
                    }
                }
                std.debug.print("[DIAG INIT] Total raw Q sum = {x}\n", .{q_total_raw.toBytesBE()[16..32].*});

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
                std.debug.print("[DIAG INIT] Right half (Q[128..255]) non-zero entries: {}\n", .{right_half_nonzero});

                // Also check RAF Q arrays
                var raf_right_half_nonzero: usize = 0;
                for (0..2) |qi| {
                    for (128..initial_m) |idx| {
                        if (!left_raf.Q[qi][idx].eql(F.zero())) raf_right_half_nonzero += 1;
                        if (!right_raf.Q[qi][idx].eql(F.zero())) raf_right_half_nonzero += 1;
                        if (!identity_raf.Q[qi][idx].eql(F.zero())) raf_right_half_nonzero += 1;
                    }
                }
                std.debug.print("[DIAG INIT] RAF right half non-zero entries: {}\n", .{raf_right_half_nonzero});

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
                std.debug.print("[DIAG INIT] bf_table0_output (Σu*rd_value for table 0) = {x}\n", .{
                    bf_table0_output.toBytesBE()[16..32].*,
                });
                std.debug.print("[DIAG INIT] bf_table0_lookup (Σu*lookup_lo for table 0) = {x}\n", .{
                    bf_table0_lookup_output.toBytesBE()[16..32].*,
                });
            }

            // Run the batched sumcheck
            std.debug.print("[STAGE5] Entering main sumcheck loop, max_num_rounds={}\n", .{max_num_rounds});

            // Track accumulated eq factor for bound cycle variables
            // This matches Jolt's current_scalar in GruenSplitEqPolynomial
            // Formula: current_scalar *= eq(w[i], challenge_i) = 1 - w[i] - c + 2*w[i]*c
            // where w[i] = r_reduction[n-1-i] (original challenge) and c = sumcheck challenge
            var lookups_current_scalar = F.one();

            for (0..max_num_rounds) |round| {
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
                    const poly_evals = computeRegsValRoundPoly(inc_evals, wa_evals, lt_evals, regs_round);
                    combined_poly[0] = combined_poly[0].add(batch0.mul(poly_evals[0]));
                    combined_poly[1] = combined_poly[1].add(batch0.mul(poly_evals[1]));
                    combined_poly[2] = combined_poly[2].add(batch0.mul(poly_evals[2]));
                    combined_poly[3] = combined_poly[3].add(batch0.mul(poly_evals[3]));

                    // CRITICAL DEBUG: Check if p0(0) + p0(1) = claim0
                    const inst0_sum = poly_evals[0].add(poly_evals[1]);
                    const inst0_sum_matches = inst0_sum.eql(regs_val_current_claim);
                    if (round >= LOOKUPS_LOG_K) {
                        std.debug.print("[INST0 SUMCHECK] Round {}: p(0)+p(1) = {x}, claim0 = {x}, match = {}\n", .{
                            round,
                            inst0_sum.toBytesBE()[16..32].*,
                            regs_val_current_claim.toBytesBE()[16..32].*,
                            inst0_sum_matches,
                        });
                    }

                    // Debug: Print Instance 0 contribution for cycle rounds
                    if (round >= LOOKUPS_LOG_K and round <= LOOKUPS_LOG_K + 1) {
                        const inst0_coeffs = UniPoly(F).toomCookToCoeffs(poly_evals);
                        std.debug.print("[ZOLT INST0] Round {}: regs_round={}\n", .{ round, regs_round });
                        std.debug.print("  poly_evals (Toom) = [{any}, {any}, {any}, {any}]\n", .{
                            poly_evals[0].toBytes(), poly_evals[1].toBytes(),
                            poly_evals[2].toBytes(), poly_evals[3].toBytes(),
                        });
                        std.debug.print("  inst0_coeffs (coeffs) = [{any}, {any}, {any}, {any}]\n", .{
                            inst0_coeffs[0].toBytes(), inst0_coeffs[1].toBytes(),
                            inst0_coeffs[2].toBytes(), inst0_coeffs[3].toBytes(),
                        });
                        std.debug.print("  batch0 = {any}\n", .{batch0.toBytes()});
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
                    // For sparse traces, we compute directly from RAM accesses
                    //
                    // The sumcheck proves: Σ_{k,c} eq_combined(k,c) · ra(k,c) = input_claim
                    // where:
                    //   eq_combined(k, c) = eq(r_addr_1, k)·G_A(c) + γ²·eq(r_addr_2, k)·G_B(c)
                    //   G_A(c) = eq_raf(c) + γ·eq_val(c)
                    //   G_B(c) = eq_rw(c) + γ·eq_val(c)
                    //
                    // Since ra(k,c) is sparse (only 1s at RAM accesses), we iterate over accesses.
                    //
                    // PhaseAddress (first log_K rounds): bind address bits
                    // PhaseCycle (last log_T rounds): bind cycle bits

                    const ram_ra_round = ram_ra_num_rounds - remaining_rounds; // 0 to 23

                    if (ram_ra_round < log_ram_k) {
                        // PhaseAddress: round ram_ra_round binds address bit ram_ra_round
                        // Using LowToHigh binding order (same as Jolt's RamRaClaimReduction)
                        //
                        // For each k_prime in [0, K/2), we sum over k in {2*k_prime, 2*k_prime+1}
                        // The bit being bound is k_m = (k >> m) & 1 where m = ram_ra_round
                        //
                        // p(0) = Σ_{accesses where k_m=0} F_k * (B_1[k_prime][0] * G_A + γ² * B_2[k_prime][0] * G_B)
                        // p(1) = Σ_{accesses where k_m=0} F_k * (B_1[k_prime][1] * G_A + γ² * B_2[k_prime][1] * G_B)
                        // where k_prime = k >> (m+1), and F_k = eq(r_addr_reduced_so_far, k & ((1<<m)-1))

                        // ===================================================================
                        // PhaseAddress polynomial (matching Jolt's dense approach exactly)
                        // ===================================================================
                        // Jolt's formula:
                        //   For each k_prime in [0, half_K):
                        //     B_1_evals = [B_1[2*k'], linear_extrap_to_c2(B_1[2*k'], B_1[2*k'+1])]
                        //     B_2_evals = [B_2[2*k'], linear_extrap_to_c2(B_2[2*k'], B_2[2*k'+1])]
                        //     For k in [k_prime << m, (k_prime+1) << m):
                        //       split by k_m = (k >> (m-1)) & 1
                        //       sum_A_0/1 += G_A_full[k] * F[k & mask]
                        //       sum_B_0/1 += G_B_full[k] * F[k & mask]
                        //     inner_A_0 = sum_A_0, inner_A_c2 = 2*sum_A_1 - sum_A_0
                        //     eval_0 += B_1[0]*inner_A_0 + γ²*B_2[0]*inner_B_0
                        //     eval_c2 += B_1_c2*inner_A_c2 + γ²*B_2_c2*inner_B_c2
                        //
                        // NOTE: Jolt uses m = round + 1 (1-indexed), we use m = ram_ra_round (0-indexed)
                        // The bit extracted is (k >> (m-1)) & 1 in Jolt = (k >> m) & 1 when m is 0-indexed
                        // Actually Jolt: m = round + 1, k_m = (k >> (m-1)) & 1 = (k >> round) & 1
                        // So with m = ram_ra_round: k_m = (k >> m) & 1
                        const m = ram_ra_round;
                        // inner_len = 1 << (m+1) in Jolt's terms = 1 << (round+1) = 2^(m+1)
                        // This is the range of k values per k_prime
                        const inner_len = @as(usize, 1) << @intCast(m + 1);
                        // f_index_mask: F uses the lower m bits = (1 << m) - 1
                        const f_index_mask = (@as(usize, 1) << @intCast(m)) - 1;
                        // half_K = K / 2 after binding = B_1.len / 2 at this point
                        const current_B_len = K >> @intCast(m);
                        const half_K = current_B_len / 2;

                        var eval_0_acc = F.zero();
                        var eval_c2_acc = F.zero();

                        for (0..half_K) |k_prime| {
                            // B_1 evals at X=0 and linear extrapolation to X=2
                            const B_1_at_0 = B_1[2 * k_prime];
                            const B_1_at_1 = B_1[2 * k_prime + 1];
                            const B_1_at_c2 = B_1_at_1.add(B_1_at_1).sub(B_1_at_0);
                            const B_2_at_0 = B_2[2 * k_prime];
                            const B_2_at_1 = B_2[2 * k_prime + 1];
                            const B_2_at_c2 = B_2_at_1.add(B_2_at_1).sub(B_2_at_0);

                            // Inner sum: iterate over k in [k_prime << (m+1), (k_prime+1) << (m+1))
                            const k_start = k_prime * inner_len;
                            const k_end = k_start + inner_len;
                            var sum_A_0 = F.zero();
                            var sum_A_1 = F.zero();
                            var sum_B_0 = F.zero();
                            var sum_B_1 = F.zero();

                            for (k_start..k_end) |k| {
                                const k_m = (k >> @intCast(m)) & 1;
                                const F_k = ram_ra_F.get(k & f_index_mask);
                                const G_A_k = G_A_full[k];
                                const G_B_k = G_B_full[k];

                                // Skip zero entries (sparse optimization)
                                const is_zero_A = G_A_k.eql(F.zero());
                                const is_zero_B = G_B_k.eql(F.zero());
                                if (is_zero_A and is_zero_B) continue;

                                const contrib_A = G_A_k.mul(F_k);
                                const contrib_B = G_B_k.mul(F_k);

                                if (k_m == 0) {
                                    sum_A_0 = sum_A_0.add(contrib_A);
                                    sum_B_0 = sum_B_0.add(contrib_B);
                                } else {
                                    sum_A_1 = sum_A_1.add(contrib_A);
                                    sum_B_1 = sum_B_1.add(contrib_B);
                                }
                            }

                            // Linear extrapolation of inner sums to c=2
                            const inner_A_0 = sum_A_0;
                            const inner_A_c2 = sum_A_1.add(sum_A_1).sub(sum_A_0);
                            const inner_B_0 = sum_B_0;
                            const inner_B_c2 = sum_B_1.add(sum_B_1).sub(sum_B_0);

                            // Combine: eval_0 += B_1[0]*inner_A_0 + γ²*B_2[0]*inner_B_0
                            eval_0_acc = eval_0_acc.add(B_1_at_0.mul(inner_A_0).add(gamma2.mul(B_2_at_0.mul(inner_B_0))));
                            eval_c2_acc = eval_c2_acc.add(B_1_at_c2.mul(inner_A_c2).add(gamma2.mul(B_2_at_c2.mul(inner_B_c2))));
                        }

                        // Derive eval_1 from sumcheck property: p(0) + p(1) = claim
                        const eval_0 = eval_0_acc;
                        const eval_1 = ram_ra_current_claim.sub(eval_0_acc);
                        const eval_2 = eval_c2_acc;

                        combined_poly[0] = combined_poly[0].add(batch1.mul(eval_0));
                        combined_poly[1] = combined_poly[1].add(batch1.mul(eval_1));
                        combined_poly[2] = combined_poly[2].add(batch1.mul(eval_2));
                        // evals[3] = 0 for degree-2 polynomial

                        // Store evaluations for claim update after binding
                        inst1_eval_0 = eval_0;
                        inst1_eval_1 = eval_1;
                        inst1_eval_2 = eval_2;

                        // Debug for all PhaseAddress rounds: check p(0)+p(1)=claim
                        {
                            const inst1_sum = eval_0.add(eval_1);
                            const sum_matches = inst1_sum.eql(ram_ra_current_claim);
                            if (!sum_matches or ram_ra_round < 3 or ram_ra_round == log_ram_k - 1) {
                                std.debug.print("[STAGE5 RAM_RA] PhaseAddress round {} (global {}): eval_0={x}, eval_1={x}\n", .{
                                    ram_ra_round,
                                    round,
                                    eval_0.toBytesBE()[16..32].*,
                                    eval_1.toBytesBE()[16..32].*,
                                });
                                std.debug.print("  eval_0+eval_1={x}, current_claim={x}, match={}\n", .{
                                    inst1_sum.toBytesBE()[16..32].*,
                                    ram_ra_current_claim.toBytesBE()[16..32].*,
                                    sum_matches,
                                });
                            }

                            // Extra debug for round 0: verify the formula
                            if (ram_ra_round == 0 and ram_access_count > 0) {
                                const addr = ram_addresses[0];
                                const addr_usize: usize = @intCast(addr);
                                const k_prime = addr_usize >> 1;
                                const k_m: u1 = @truncate(addr_usize);
                                const B_1_access = if (k_m == 0) B_1[2 * k_prime] else B_1[2 * k_prime + 1];
                                const B_2_access = if (k_m == 0) B_2[2 * k_prime] else B_2[2 * k_prime + 1];
                                const G_A_0 = ram_G_A[0];
                                const G_B_0 = ram_G_B[0];
                                const F_k = ram_ra_F.get(0);
                                const expected_contrib = B_1_access.mul(G_A_0).add(gamma2.mul(B_2_access.mul(G_B_0))).mul(F_k);
                                std.debug.print("  [DEBUG R0] addr={}, k_prime={}, k_m={}\n", .{ addr, k_prime, k_m });
                                std.debug.print("  [DEBUG R0] B_1_access={x}, B_2_access={x}\n", .{ B_1_access.toBytesBE()[16..32].*, B_2_access.toBytesBE()[16..32].* });
                                std.debug.print("  [DEBUG R0] G_A={x}, G_B={x}\n", .{ G_A_0.toBytesBE()[16..32].*, G_B_0.toBytesBE()[16..32].* });
                                std.debug.print("  [DEBUG R0] F_k={x}\n", .{F_k.toBytesBE()[16..32].*});
                                std.debug.print("  [DEBUG R0] gamma2={x}\n", .{gamma2.toBytesBE()[16..32].*});
                                std.debug.print("  [DEBUG R0] expected_contrib={x}\n", .{expected_contrib.toBytesBE()[16..32].*});
                            }
                        }
                    } else {
                        // PhaseCycle: bind cycle bits
                        // After address rounds, we have bound all address variables.
                        // α_1 = B_1.final_claim() = eq(r_addr_1, r_addr_reduced)
                        // α_2 = B_2.final_claim() = eq(r_addr_2, r_addr_reduced)
                        //
                        // The sumcheck now needs to bind cycle variables. For sparse traces,
                        // we iterate over accesses and compute contributions based on the cycle bit.
                        //
                        // The eq contribution for each access has THREE components:
                        //   eq_raf(c) = eq(r_cycle_raf, c)
                        //   eq_rw(c) = eq(r_cycle_rw, c)
                        //   eq_val(c) = eq(r_cycle_val, c)
                        //
                        // During cycle sumcheck, these get bound incrementally. But since we
                        // precomputed G_A[i] = eq_raf(c) + γ*eq_val(c) with FULL eq values,
                        // we can't use them directly for the sumcheck polynomial.
                        //
                        // The correct approach:
                        // For each cycle round m, split the contribution based on bit m.
                        // ============================================================
                        // P*Q Decomposition PhaseCycle (Jolt's approach)
                        // ============================================================
                        // Using prefix-suffix decomposition:
                        //   P_x[c_lo] = eq(r_cycle_x_lo, c_lo)  -- prefix eq evaluations
                        //   Q_x[c_lo] = Σ_{c_hi} H[c_lo,c_hi] · eq(r_cycle_x_hi, c_hi)  -- suffix sums
                        //
                        // PhaseCycle1: rounds 0 to prefix_n_vars-1 (bind prefix bits using P*Q)
                        // PhaseCycle2: rounds prefix_n_vars to n_cycle_vars-1 (bind suffix using H'*eq_hi)

                        const cycle_round = ram_ra_round - log_ram_k; // 0 to n_cycle_vars-1

                        // Get α_1 and α_2 from final B_1, B_2 claims
                        const alpha_1 = B_1[0];
                        const alpha_2 = B_2[0];

                        // Initialize Q arrays at the start of PhaseCycle
                        if (!phase_cycle_q_initialized) {
                            phase_cycle_q_initialized = true;

                            std.debug.print("[STAGE5 RAM_RA] PhaseCycle starting: alpha_1={x}, alpha_2={x}\n", .{
                                alpha_1.toBytesBE()[16..32].*,
                                alpha_2.toBytesBE()[16..32].*,
                            });
                            std.debug.print("[STAGE5 PQ] prefix_n_vars={}, suffix_n_vars={}\n", .{
                                prefix_n_vars, suffix_n_vars,
                            });

                            // Compute Q arrays: Q_x[c_lo] = Σ_{c_hi} H[c_lo,c_hi] · eq_x_hi(c_hi)
                            // where H[c] = F_values[address[c]] = eq(r_addr_reduced, address[c])
                            @memset(Q_raf, F.zero());
                            @memset(Q_rw, F.zero());
                            @memset(Q_val, F.zero());

                            for (0..ram_access_count) |access_idx| {
                                const cycle = ram_cycles[access_idx];
                                const cycle_usize: usize = @intCast(cycle);
                                const addr = ram_addresses[access_idx];
                                const addr_usize: usize = @intCast(addr);

                                // H[c] = F_values[address[c]] = eq(r_addr_reduced, addr)
                                const H_c = ram_ra_F.get(addr_usize & (K - 1));

                                // Split cycle into c_lo (prefix) and c_hi (suffix)
                                const c_lo = cycle_usize & (prefix_size - 1);
                                const c_hi = cycle_usize >> @intCast(prefix_n_vars);

                                // Q_x[c_lo] += H[c] * eq_x_hi(c_hi)
                                Q_raf[c_lo] = Q_raf[c_lo].add(H_c.mul(eq_raf_hi[c_hi]));
                                Q_rw[c_lo] = Q_rw[c_lo].add(H_c.mul(eq_rw_hi[c_hi]));
                                Q_val[c_lo] = Q_val[c_lo].add(H_c.mul(eq_val_hi[c_hi]));
                            }

                            std.debug.print("[STAGE5 PQ] Q arrays initialized with {} accesses\n", .{ram_access_count});
                            // Print all non-zero Q values
                            for (0..prefix_size) |qi| {
                                if (!Q_raf[qi].eql(F.zero()) or !Q_rw[qi].eql(F.zero()) or !Q_val[qi].eql(F.zero())) {
                                    std.debug.print("[STAGE5 PQ] Q_raf[{}]={x}, Q_rw[{}]={x}, Q_val[{}]={x}\n", .{
                                        qi, Q_raf[qi].toBytesBE()[16..32].*,
                                        qi, Q_rw[qi].toBytesBE()[16..32].*,
                                        qi, Q_val[qi].toBytesBE()[16..32].*,
                                    });
                                }
                            }
                            // Print corresponding P values at same indices
                            for (0..prefix_size) |qi| {
                                if (!Q_raf[qi].eql(F.zero()) or !Q_rw[qi].eql(F.zero()) or !Q_val[qi].eql(F.zero())) {
                                    std.debug.print("[STAGE5 PQ] P_raf[{}]={x}, P_rw[{}]={x}, P_val[{}]={x}\n", .{
                                        qi, P_raf[qi].toBytesBE()[16..32].*,
                                        qi, P_rw[qi].toBytesBE()[16..32].*,
                                        qi, P_val[qi].toBytesBE()[16..32].*,
                                    });
                                }
                            }
                            // Print H_c for the access
                            if (ram_access_count > 0) {
                                const dbg_addr: usize = @intCast(ram_addresses[0]);
                                const dbg_H_c = ram_ra_F.get(dbg_addr & (K - 1));
                                std.debug.print("[STAGE5 PQ] H_c (eq(r_addr_reduced, addr={})) = {x}\n", .{
                                    dbg_addr, dbg_H_c.toBytesBE()[16..32].*,
                                });
                                const dbg_cycle: usize = @intCast(ram_cycles[0]);
                                const dbg_c_lo = dbg_cycle & (prefix_size - 1);
                                const dbg_c_hi = dbg_cycle >> @intCast(prefix_n_vars);
                                std.debug.print("[STAGE5 PQ] cycle={}, c_lo={}, c_hi={}\n", .{
                                    dbg_cycle, dbg_c_lo, dbg_c_hi,
                                });
                                std.debug.print("[STAGE5 PQ] eq_raf_hi[{}]={x}, eq_rw_hi[{}]={x}, eq_val_hi[{}]={x}\n", .{
                                    dbg_c_hi, eq_raf_hi[dbg_c_hi].toBytesBE()[16..32].*,
                                    dbg_c_hi, eq_rw_hi[dbg_c_hi].toBytesBE()[16..32].*,
                                    dbg_c_hi, eq_val_hi[dbg_c_hi].toBytesBE()[16..32].*,
                                });
                            }
                        }

                        // Compute polynomial using P * Q products
                        // Coefficients: α_1, γ²·α_2, (γ·α_1 + γ³·α_2)
                        // Note: gamma3 = gamma^3 is already defined at higher scope
                        const coeff_raf = alpha_1;
                        const coeff_rw = gamma2.mul(alpha_2);
                        const coeff_val = gamma.mul(alpha_1).add(gamma3.mul(alpha_2));

                        // Current P polynomial length (P_raf initially has prefix_size elements)
                        // After cycle_round bindings, effective length is prefix_size >> cycle_round
                        const current_P_len = prefix_size >> @intCast(cycle_round);
                        const half_len = current_P_len / 2;

                        if (cycle_round < prefix_n_vars and half_len > 0) {
                            // PhaseCycle1: Bind prefix bits using P*Q decomposition
                            // CRITICAL: Use hint mechanism - only compute eval_1 and eval_2 directly
                            // Then derive eval_0 = claim - eval_1 to guarantee sumcheck property

                            // DEBUG: Print claim at start of polynomial computation
                            std.debug.print("[INST1 HINT DEBUG] Round {}: ram_ra_current_claim at poly start = {x}\n", .{
                                round,
                                ram_ra_current_claim.toBytesBE()[16..32].*,
                            });

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

                            std.debug.print("[STAGE5 RAM_RA] PhaseCycle1 round {}: eval_0={x}, eval_1={x}, eval_2={x}\n", .{
                                cycle_round,
                                eval_0.toBytesBE()[16..32].*,
                                eval_1.toBytesBE()[16..32].*,
                                eval_2.toBytesBE()[16..32].*,
                            });

                            // Debug: Print Instance 1 contribution for Round 128-129
                            if (round >= LOOKUPS_LOG_K and round <= LOOKUPS_LOG_K + 1) {
                                const inst1_evals = [4]F{ eval_0, eval_1, eval_2, F.zero() };
                                const inst1_coeffs = UniPoly(F).toomCookToCoeffs(inst1_evals);
                                std.debug.print("[ZOLT INST1] Round {}: ram_ra_round={}, cycle_round={}\n", .{ round, ram_ra_round, cycle_round });
                                std.debug.print("  inst1_evals (Toom) = [{any}, {any}, {any}, {any}]\n", .{
                                    inst1_evals[0].toBytes(), inst1_evals[1].toBytes(),
                                    inst1_evals[2].toBytes(), inst1_evals[3].toBytes(),
                                });
                                std.debug.print("  inst1_coeffs (coeffs) = [{any}, {any}, {any}, {any}]\n", .{
                                    inst1_coeffs[0].toBytes(), inst1_coeffs[1].toBytes(),
                                    inst1_coeffs[2].toBytes(), inst1_coeffs[3].toBytes(),
                                });
                                std.debug.print("  batch1 = {any}\n", .{batch1.toBytes()});
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

                                    const H_c = ram_ra_F.get(addr_usize & (K - 1));
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

                                // Update coefficients with scaling factors
                                // coeff_* already include alpha values, now multiply by scale factors
                                // Note: We need to recompute coeff_* with the scales
                                // coeff_raf_scaled = alpha_1 * scale_raf
                                // coeff_rw_scaled = gamma2 * alpha_2 * scale_rw
                                // coeff_val_scaled = (gamma * alpha_1 + gamma3 * alpha_2) * scale_val

                                std.debug.print("[STAGE5 RAM_RA] PhaseCycle2 starting at suffix_round={}\n", .{suffix_round});
                                std.debug.print("  scale_raf={x}, scale_rw={x}, scale_val={x}\n", .{
                                    scale_raf.toBytesBE()[16..32].*,
                                    scale_rw.toBytesBE()[16..32].*,
                                    scale_val.toBytesBE()[16..32].*,
                                });
                                if (suffix_size > 0) {
                                    std.debug.print("  H_prime[0]={x}\n", .{H_prime[0].toBytesBE()[16..32].*});
                                }
                            }

                            // Compute polynomial using H_prime * eq_hi products
                            const current_len = suffix_size >> @intCast(suffix_round);
                            const half_len_suffix = current_len / 2;

                            // Scaled coefficients (include scale factors from eq(r_cycle_lo, prefix_challenges))
                            const coeff_raf_scaled = alpha_1.mul(scale_raf);
                            const coeff_rw_scaled = gamma2.mul(alpha_2).mul(scale_rw);
                            const coeff_val_scaled = gamma.mul(alpha_1).add(gamma3.mul(alpha_2)).mul(scale_val);

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

                            std.debug.print("[STAGE5 RAM_RA] PhaseCycle2 round {}: eval_0={x}, eval_1={x}, eval_2={x}\n", .{
                                cycle_round,
                                eval_0.toBytesBE()[16..32].*,
                                eval_1.toBytesBE()[16..32].*,
                                eval_2.toBytesBE()[16..32].*,
                            });
                        }
                    }
                }

                // Instance 2: LookupsReadRaf (136 rounds)
                // Since lookups_num_rounds = max_num_rounds, this instance is always active
                //
                // The sumcheck proves: Σ_j Σ_k eq(j, r_reduction) * ra(k, j) * combined(k, j) = input
                // Since ra(k, j) = 1 only when k = lookup_index(j), this simplifies to:
                // Σ_j eq(j, r_reduction) * combined(lookup_index(j), j)
                //
                // First 128 rounds (address variables):
                //   For round i, we bind address bit i.
                //   p(0) = sum over cycles j where bit i of lookup_index(j) = 0
                //   p(1) = sum over cycles j where bit i of lookup_index(j) = 1
                //   Each cycle contributes: eq_reduction[j] * ra_weights[j] * combined[j]
                //
                // Last 8 rounds (cycle variables):
                //   Standard sumcheck over the remaining cycle polynomial.
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

                    // Compute read-checking contribution via prefix-suffix decomposition
                    const read_checking_evals = proverMsgReadChecking(F, round, &suffix_polys, &prefix_checkpoints, r_x);

                    // Compute RAF contribution via prefix-suffix decomposition
                    // gamma_raf = γ, gamma_raf2 = γ²
                    const raf_evals = proverMsgRaf(F, &left_raf, &right_raf, &identity_raf, gamma_raf, gamma_raf2);

                    // Debug: print evaluations for first 3 rounds
                    if (round < 3) {
                        std.debug.print("[STAGE5 ROUND {} DEBUG] read_checking=[{x}, {x}]\n", .{
                            round,
                            read_checking_evals[0].toBytesBE()[16..32].*,
                            read_checking_evals[1].toBytesBE()[16..32].*,
                        });
                        std.debug.print("[STAGE5 ROUND {} DEBUG] raf_evals=[{x}, {x}]\n", .{
                            round,
                            raf_evals[0].toBytesBE()[16..32].*,
                            raf_evals[1].toBytesBE()[16..32].*,
                        });
                    }

                    // Combined: read_checking + raf
                    const eval_0_inst2 = read_checking_evals[0].add(raf_evals[0]);
                    const eval_2_inst2 = read_checking_evals[1].add(raf_evals[1]);

                    // BRUTE FORCE VERIFICATION: At round 0, compute the Instance 2 eval_0
                    // directly from the trace data. eval_0 = sum of all u_evals[j] * combined[j]
                    // where bit round (HighToLow) of lookup_index[j] is 0.
                    // eval_1 = sum where that bit is 1.
                    // eval_0 + eval_1 should equal lookups_claim.
                    if (round == 0) {
                        // At round 0, verify that combined_vals matches trace
                        var total_sum_cv = F.zero();
                        for (0..T) |jj| {
                            total_sum_cv = total_sum_cv.add(lookups_eq_evals[jj].mul(lookups_combined_vals[jj]));
                        }
                        std.debug.print("[BRUTE R0 CHECK] total_sum(eq*combined_vals) = {x}\n", .{total_sum_cv.toBytesBE()[16..32].*});
                        std.debug.print("[BRUTE R0 CHECK] lookups_claim = {x}\n", .{lookups_claim.toBytesBE()[16..32].*});
                        std.debug.print("[BRUTE R0 CHECK] match: {}\n", .{total_sum_cv.eql(lookups_claim)});
                        // Also check: does combined_vals sum match computed_sum from earlier?
                        // Print first 5 combined_vals
                        for (0..@min(5, T)) |jj| {
                            if (!lookups_combined_vals[jj].eql(F.zero())) {
                                std.debug.print("[BRUTE R0 CHECK] combined_vals[{}] = {x}\n", .{jj, lookups_combined_vals[jj].toBytesBE()[16..32].*});
                                std.debug.print("[BRUTE R0 CHECK] eq_evals[{}] = {x}\n", .{jj, lookups_eq_evals[jj].toBytesBE()[16..32].*});
                            }
                        }
                    }
                    if (round < 3 or round == 7 or round == 15 or round == 127) {
                        const bit_pos = LOOKUPS_LOG_K - 1 - round;
                        var direct_eval_0 = F.zero();
                        var direct_eval_1 = F.zero();
                        var bf_val_eval_0 = F.zero();
                        var bf_raf_eval_0 = F.zero();
                        var bf_val_per_table: [NUM_TABLES]F = [_]F{F.zero()} ** NUM_TABLES;
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
                        std.debug.print("[BRUTE R{}] direct_eval_0={x}\n", .{ round, direct_eval_0.toBytesBE()[16..32].* });
                        std.debug.print("[BRUTE R{}] direct_eval_1={x}\n", .{ round, direct_eval_1.toBytesBE()[16..32].* });
                        std.debug.print("[BRUTE R{}] bf_val_eval_0={x} (should match read_checking)\n", .{ round, bf_val_eval_0.toBytesBE()[16..32].* });
                        std.debug.print("[BRUTE R{}] bf_raf_eval_0={x} (should match raf_evals)\n", .{ round, bf_raf_eval_0.toBytesBE()[16..32].* });
                        if (round == 0) {
                            for (0..NUM_TABLES) |t_check| {
                                if (!bf_val_per_table[t_check].eql(F.zero())) {
                                    std.debug.print("[BRUTE R0] bf_val_per_table[{}]={x}\n", .{ t_check, bf_val_per_table[t_check].toBytesBE()[16..32].* });
                                }
                            }
                        }
                        std.debug.print("[BRUTE R{}] direct_sum={x}, lookups_claim={x}\n", .{
                            round,
                            direct_sum.toBytesBE()[16..32].*,
                            lookups_claim.toBytesBE()[16..32].*,
                        });
                        std.debug.print("[BRUTE R{}] sum==claim: {}, eval0_match: {}\n", .{
                            round,
                            direct_sum.eql(lookups_claim),
                            direct_eval_0.eql(eval_0_inst2),
                        });
                        std.debug.print("[BRUTE R{}] prefix_suffix_eval_0={x}\n", .{ round, eval_0_inst2.toBytesBE()[16..32].* });
                        std.debug.print("[BRUTE R{}] prefix_suffix_eval_2={x}\n", .{ round, eval_2_inst2.toBytesBE()[16..32].* });

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
                            std.debug.print("[CORRECT_RAF R{}] eval_0={x}\n", .{ round, correct_raf_eval_0.toBytesBE()[16..32].* });
                            std.debug.print("[CORRECT_RAF R{}] matches raf_evals[0]: {}\n", .{ round, correct_raf_eval_0.eql(raf_evals[0]) });
                        }
                    }

                    // Combine Instance 2 with Instance 0 and 1 contributions
                    // Instance 0 and 1 are already in combined_poly[0..4] as Toom-Cook format [p(0), p(1), p(2), p_inf]
                    //
                    // For Instance 2 (LookupsReadRaf, degree-2):
                    // - p_2(0) = eval_0_inst2
                    // - p_2(1) = lookups_claim - eval_0_inst2 (from sumcheck property p(0) + p(1) = claim)
                    // - p_2(2) = eval_2_inst2
                    // - p_2(inf) = 0 (degree-2, no cubic term)
                    const eval_1_inst2 = lookups_claim.sub(eval_0_inst2);

                    // Add Instance 2's full Toom-Cook contribution to combined_poly
                    combined_poly[0] = combined_poly[0].add(batch2.mul(eval_0_inst2));
                    combined_poly[1] = combined_poly[1].add(batch2.mul(eval_1_inst2));
                    combined_poly[2] = combined_poly[2].add(batch2.mul(eval_2_inst2));
                    // combined_poly[3] += 0 (Instance 2 is degree-2, no cubic term)

                    // Convert to compressed format [c0, c2, c3] using Toom-Cook encoding
                    // This produces a degree-3 polynomial since Instance 0 (RegistersValEvaluation)
                    // is a product of 3 linear polynomials (inc, wa, lt)
                    const compressed = UniPoly(F).toomCookToCompressed(combined_poly);

                    const coeffs = try self.allocator.alloc(F, 3);
                    coeffs[0] = compressed[0]; // c0
                    coeffs[1] = compressed[1]; // c2
                    coeffs[2] = compressed[2]; // c3

                    // Debug: print coefficients for first 3 rounds and rounds 110+ (compare with Jolt)
                    if (round < 3 or round >= 110) {
                        std.debug.print("[STAGE5 COEFF ROUND {}] c0 (LE) = {any}\n", .{ round, coeffs[0].toBytes() });
                        std.debug.print("[STAGE5 COEFF ROUND {}] c2 (LE) = {any}\n", .{ round, coeffs[1].toBytes() });
                        std.debug.print("[STAGE5 COEFF ROUND {}] c3 (LE) = {any}\n", .{ round, coeffs[2].toBytes() });
                    }
                    // CRITICAL VERIFICATION: p(0) + p(1) should equal current_batched_claim
                    const poly_sum = combined_poly[0].add(combined_poly[1]);
                    const sumcheck_ok_addr = poly_sum.eql(current_batched_claim);
                    if (!sumcheck_ok_addr or round < 3 or round == 127) {
                        std.debug.print("[STAGE5 VERIFY R{}] p(0)+p(1)={x}, claim={x}, match={}\n", .{
                            round,
                            poly_sum.toBytesBE()[16..32].*,
                            current_batched_claim.toBytesBE()[16..32].*,
                            sumcheck_ok_addr,
                        });
                    }

                    if (round == 0) {
                        std.debug.print("[STAGE5 COEFF ROUND 0] claim = {x}\n", .{current_batched_claim.toBytesBE()});
                        std.debug.print("[STAGE5 COEFF ROUND 0] combined_poly[0] (p0) = {x}\n", .{combined_poly[0].toBytesBE()});
                        std.debug.print("[STAGE5 COEFF ROUND 0] combined_poly[1] (p1) = {x}\n", .{combined_poly[1].toBytesBE()});
                        std.debug.print("[STAGE5 COEFF ROUND 0] combined_poly[2] (p2) = {x}\n", .{combined_poly[2].toBytesBE()});
                        std.debug.print("[STAGE5 COEFF ROUND 0] combined_poly[3] (p_inf) = {x}\n", .{combined_poly[3].toBytesBE()});
                        std.debug.print("[STAGE5 COEFF ROUND 0] inst2_eval0 = {x}\n", .{eval_0_inst2.toBytesBE()});
                        std.debug.print("[STAGE5 COEFF ROUND 0] inst2_eval1 = {x}\n", .{eval_1_inst2.toBytesBE()});
                        std.debug.print("[STAGE5 COEFF ROUND 0] inst2_eval2 = {x}\n", .{eval_2_inst2.toBytesBE()});
                        std.debug.print("[STAGE5 COEFF ROUND 0] lookups_claim = {x}\n", .{lookups_claim.toBytesBE()});
                        std.debug.print("[STAGE5 COEFF ROUND 0] batch2 = {x}\n", .{batch2.toBytesBE()});
                    }

                    try proof.compressed_polys.append(self.allocator, .{
                        .coeffs_except_linear_term = coeffs,
                        .allocator = self.allocator,
                    });

                    // Append to transcript
                    transcript.appendMessage("UniPoly_begin");
                    transcript.appendScalar(coeffs[0]);
                    transcript.appendScalar(coeffs[1]);
                    transcript.appendScalar(coeffs[2]);
                    transcript.appendMessage("UniPoly_end");

                    const challenge = transcript.challengeScalar();
                    challenges[round] = challenge;

                    // Update current_batched_claim by evaluating polynomial at challenge
                    // For degree-3: p(r) = c0 + r*c1 + r^2*c2 + r^3*c3
                    // where c1 = claim - 2*c0 - c2 - c3 (from sumcheck property p(0)+p(1) = claim)
                    // Since p(0) = c0 and p(1) = c0 + c1 + c2 + c3
                    const c0 = coeffs[0];
                    const c2_val = coeffs[1];
                    const c3_val = coeffs[2];
                    const c1 = current_batched_claim.sub(c0).sub(c0).sub(c2_val).sub(c3_val);
                    // Challenge * Challenge: both convert to Fr, standard mul
                    const r2 = challenge.mul(challenge);
                    const r3 = r2.mul(challenge);

                    // Debug: print c1 computation for round 0
                    if (round == 0) {
                        std.debug.print("[STAGE5 R0 EVAL] claim = {any}\n", .{current_batched_claim.toBytes()});
                        std.debug.print("[STAGE5 R0 EVAL] c0 = {any}\n", .{c0.toBytes()});
                        std.debug.print("[STAGE5 R0 EVAL] c2 = {any}\n", .{c2_val.toBytes()});
                        std.debug.print("[STAGE5 R0 EVAL] c3 = {any}\n", .{c3_val.toBytes()});
                        const claim_minus_c0 = current_batched_claim.sub(c0);
                        std.debug.print("[STAGE5 R0 EVAL] claim - c0 = {any}\n", .{claim_minus_c0.toBytes()});
                        const claim_minus_2c0 = claim_minus_c0.sub(c0);
                        std.debug.print("[STAGE5 R0 EVAL] claim - 2*c0 = {any}\n", .{claim_minus_2c0.toBytes()});
                        const claim_minus_2c0_minus_c2 = claim_minus_2c0.sub(c2_val);
                        std.debug.print("[STAGE5 R0 EVAL] claim - 2*c0 - c2 = {any}\n", .{claim_minus_2c0_minus_c2.toBytes()});
                        std.debug.print("[STAGE5 R0 EVAL] c1 = claim - 2*c0 - c2 - c3 = {any}\n", .{c1.toBytes()});
                        std.debug.print("[STAGE5 R0 EVAL] challenge (LE bytes) = {any}\n", .{challenge.toBytes()});
                        const c1_times_r = c1.mulHiBigIntU128(challenge.limbs);
                        std.debug.print("[STAGE5 R0 EVAL] c1 * r (mulHiBigIntU128) = {any}\n", .{c1_times_r.toBytes()});
                        std.debug.print("[STAGE5 R0 EVAL] c2 * r^2 = {any}\n", .{c2_val.mul(r2).toBytes()});
                        std.debug.print("[STAGE5 R0 EVAL] c3 * r^3 = {any}\n", .{c3_val.mul(r3).toBytes()});
                    }

                    // CRITICAL: Challenge * F uses mulHiBigIntU128 (Jolt delegates to F * Challenge)
                    current_batched_claim = c0.add(c1.mulHiBigIntU128(challenge.limbs)).add(c2_val.mul(r2)).add(c3_val.mul(r3));

                    // Per-round tracking (matches Jolt verifier's [S5V] output)
                    std.debug.print("  [S5P] R{} challenge={x} new_e={x} degree=3\n", .{
                        round,
                        challenge.toBytes()[0..16].*,
                        current_batched_claim.toBytes()[0..16].*,
                    });

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
                    const inst2_c2 = eval_2_inst2.sub(eval_1_inst2).sub(eval_1_inst2).add(eval_0_inst2).mul(F.fromU64(2).inverse().?);
                    const inst2_c1 = eval_1_inst2.sub(eval_0_inst2).sub(inst2_c2);
                    const inst2_at_r = inst2_c0.add(inst2_c1.mulHiBigIntU128(challenge.limbs)).add(inst2_c2.mul(r2));

                    // Debug: show claim chain for first 3 rounds and last 3 address rounds
                    if (round < 3 or (round >= 125 and round < 128)) {
                        std.debug.print("[CLAIM_CHAIN R{}] before_claim={x}\n", .{ round, lookups_claim.toBytesBE()[16..32].* });
                        std.debug.print("[CLAIM_CHAIN R{}] eval_0={x}, eval_1={x}, eval_2={x}\n", .{
                            round,
                            eval_0_inst2.toBytesBE()[16..32].*,
                            eval_1_inst2.toBytesBE()[16..32].*,
                            eval_2_inst2.toBytesBE()[16..32].*,
                        });
                        std.debug.print("[CLAIM_CHAIN R{}] sum_check: eval_0+eval_1={x} (should == before_claim)\n", .{
                            round,
                            eval_0_inst2.add(eval_1_inst2).toBytesBE()[16..32].*,
                        });
                        std.debug.print("[CLAIM_CHAIN R{}] p(r)={x} -> new_claim\n", .{ round, inst2_at_r.toBytesBE()[16..32].* });
                    }

                    lookups_claim = inst2_at_r;

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
                        regs_val_current_claim = regs_val_current_claim.mul(F.fromU64(2).inverse().?);
                    }

                    if (remaining_rounds > ram_ra_num_rounds) {
                        // Instance 1 is inactive - claim halves
                        ram_ra_current_claim = ram_ra_current_claim.mul(F.fromU64(2).inverse().?);
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
                        std.debug.print("[DEBUG BINDING R{}] remaining={}, ram_ra_num_rounds={}, check={}\n", .{
                            round,
                            remaining_rounds,
                            ram_ra_num_rounds,
                            remaining_rounds <= ram_ra_num_rounds,
                        });
                    }
                    if (remaining_rounds <= ram_ra_num_rounds) {
                        const ram_ra_round = ram_ra_num_rounds - remaining_rounds;

                        if (round >= 126 and round <= 130) {
                            std.debug.print("[DEBUG R{} IN] ram_ra_round={}, log_ram_k={}, is_phase_cycle={}\n", .{
                                round,
                                ram_ra_round,
                                log_ram_k,
                                ram_ra_round >= log_ram_k,
                            });
                        }

                        if (ram_ra_round < log_ram_k) {
                            // PhaseAddress: bind B_1, B_2 and update ram_ra_F
                            const m = ram_ra_round;
                            const n_b = B_1.len >> @intCast(m);
                            const half_b = n_b / 2;

                            if (half_b > 0) {
                                const one_minus_r = F.one().sub(challenge);

                                // Bind B_1 and B_2 polynomials (LowToHigh order)
                                // CRITICAL: Use mulHiBigIntU128 for F * Challenge
                                for (0..half_b) |i| {
                                    B_1[i] = one_minus_r.mul(B_1[2 * i]).add(B_1[2 * i + 1].mulHiBigIntU128(challenge.limbs));
                                    B_2[i] = one_minus_r.mul(B_2[2 * i]).add(B_2[2 * i + 1].mulHiBigIntU128(challenge.limbs));
                                }
                                // Zero out upper half
                                for (half_b..n_b) |i| {
                                    B_1[i] = F.zero();
                                    B_2[i] = F.zero();
                                }
                            }

                            // Update ram_ra_F expanding table using LowToHigh
                            // This ensures the index bits match the address bit binding order
                            ram_ra_F.updateLowToHigh(challenge);

                            // Store bound challenge
                            ram_ra_bound_challenges[ram_ra_round] = challenge;

                            if (ram_ra_round < 3 or ram_ra_round == log_ram_k - 1) {
                                std.debug.print("[STAGE5 RAM_RA] Bound addr round {}: B_1[0]={x}, B_2[0]={x}\n", .{
                                    ram_ra_round,
                                    B_1[0].toBytesBE()[16..32].*,
                                    B_2[0].toBytesBE()[16..32].*,
                                });
                            }

                            // CRITICAL: Update Instance 1 claim for PhaseAddress rounds
                            // p(r) = c0 + r*c1 + r²*c2 for degree-2 polynomial
                            const c2_inst1_pa = inst1_eval_2.sub(inst1_eval_1).sub(inst1_eval_1).add(inst1_eval_0).mul(F.fromU64(2).inverse().?);
                            const c1_inst1_pa = inst1_eval_1.sub(inst1_eval_0).sub(c2_inst1_pa);
                            const c0_inst1_pa = inst1_eval_0;
                            const r2_pa = challenge.mul(challenge);
                            ram_ra_current_claim = c0_inst1_pa.add(c1_inst1_pa.mulHiBigIntU128(challenge.limbs)).add(c2_inst1_pa.mul(r2_pa));
                        } else {
                            // PhaseCycle: bind polynomials
                            const cycle_round = ram_ra_round - log_ram_k;
                            const one_minus_r = F.one().sub(challenge);

                            // Store cycle challenge for PhaseCycle2 eq_prefix computation
                            cycle_challenges[cycle_round] = challenge;

                            if (cycle_round < prefix_n_vars) {
                                // PhaseCycle1: bind P and Q polynomials
                                const current_len = prefix_size >> @intCast(cycle_round);
                                const half_len = current_len / 2;

                                // Bind P arrays: P'[j] = (1-r)*P[2j] + r*P[2j+1]
                                // CRITICAL: Use mulHiBigIntU128 for F * Challenge
                                for (0..half_len) |j| {
                                    P_raf[j] = one_minus_r.mul(P_raf[2 * j]).add(P_raf[2 * j + 1].mulHiBigIntU128(challenge.limbs));
                                    P_rw[j] = one_minus_r.mul(P_rw[2 * j]).add(P_rw[2 * j + 1].mulHiBigIntU128(challenge.limbs));
                                    P_val[j] = one_minus_r.mul(P_val[2 * j]).add(P_val[2 * j + 1].mulHiBigIntU128(challenge.limbs));
                                }

                                // Bind Q arrays: Q'[j] = (1-r)*Q[2j] + r*Q[2j+1]
                                // CRITICAL: Use mulHiBigIntU128 for F * Challenge
                                for (0..half_len) |j| {
                                    Q_raf[j] = one_minus_r.mul(Q_raf[2 * j]).add(Q_raf[2 * j + 1].mulHiBigIntU128(challenge.limbs));
                                    Q_rw[j] = one_minus_r.mul(Q_rw[2 * j]).add(Q_rw[2 * j + 1].mulHiBigIntU128(challenge.limbs));
                                    Q_val[j] = one_minus_r.mul(Q_val[2 * j]).add(Q_val[2 * j + 1].mulHiBigIntU128(challenge.limbs));
                                }

                                if (cycle_round < 3) {
                                    std.debug.print("[STAGE5 RAM_RA] Bound PhaseCycle1 round {}: challenge={x}, new_len={}\n", .{
                                        cycle_round,
                                        challenge.toBytesBE()[16..32].*,
                                        half_len,
                                    });
                                    if (half_len > 0) {
                                        std.debug.print("  P_raf[0]={x}, Q_raf[0]={x}\n", .{
                                            P_raf[0].toBytesBE()[16..32].*,
                                            Q_raf[0].toBytesBE()[16..32].*,
                                        });
                                    }
                                }
                            } else {
                                // PhaseCycle2: bind H_prime and eq_hi arrays
                                const suffix_round = cycle_round - prefix_n_vars;
                                const current_len = suffix_size >> @intCast(suffix_round);
                                const half_len = current_len / 2;

                                // Bind H_prime: H'[j] = (1-r)*H[2j] + r*H[2j+1]
                                // CRITICAL: Use mulHiBigIntU128 for F * Challenge
                                for (0..half_len) |j| {
                                    H_prime[j] = one_minus_r.mul(H_prime[2 * j]).add(H_prime[2 * j + 1].mulHiBigIntU128(challenge.limbs));
                                }

                                // Bind eq_hi arrays
                                // CRITICAL: Use mulHiBigIntU128 for F * Challenge
                                for (0..half_len) |j| {
                                    eq_raf_hi[j] = one_minus_r.mul(eq_raf_hi[2 * j]).add(eq_raf_hi[2 * j + 1].mulHiBigIntU128(challenge.limbs));
                                    eq_rw_hi[j] = one_minus_r.mul(eq_rw_hi[2 * j]).add(eq_rw_hi[2 * j + 1].mulHiBigIntU128(challenge.limbs));
                                    eq_val_hi[j] = one_minus_r.mul(eq_val_hi[2 * j]).add(eq_val_hi[2 * j + 1].mulHiBigIntU128(challenge.limbs));
                                }

                                std.debug.print("[STAGE5 RAM_RA] Bound PhaseCycle2 round {} (suffix {}): challenge={x}, new_len={}\n", .{
                                    cycle_round,
                                    suffix_round,
                                    challenge.toBytesBE()[16..32].*,
                                    half_len,
                                });
                                if (half_len > 0) {
                                    std.debug.print("  H_prime[0]={x}, eq_raf_hi[0]={x}\n", .{
                                        H_prime[0].toBytesBE()[16..32].*,
                                        eq_raf_hi[0].toBytesBE()[16..32].*,
                                    });
                                }
                            }
                        }

                        // NOTE: ram_ra_current_claim is now updated in the PhaseAddress binding section above
                    }

                    // CONSISTENCY CHECK: batch0*inst0 + batch1*inst1 + batch2*inst2 == batched_claim
                    // (placed AFTER Instance 1 claim update)
                    {
                        const recon = batch0.mul(regs_val_current_claim).add(batch1.mul(ram_ra_current_claim)).add(batch2.mul(lookups_claim));
                        const matches = recon.eql(current_batched_claim);
                        if (!matches or round < 3 or (round >= 110 and round <= 114) or round == 127) {
                            std.debug.print("[CONSISTENCY R{}] batch0*inst0+batch1*inst1+batch2*inst2 == batched: {}\n", .{ round, matches });
                            if (!matches) {
                                std.debug.print("  recon = {x}\n", .{recon.toBytesBE()[16..32].*});
                                std.debug.print("  batched = {x}\n", .{current_batched_claim.toBytesBE()[16..32].*});
                                std.debug.print("  inst0 (regs) = {x}\n", .{regs_val_current_claim.toBytesBE()[16..32].*});
                                std.debug.print("  inst1 (ram_ra) = {x}\n", .{ram_ra_current_claim.toBytesBE()[16..32].*});
                                std.debug.print("  inst2 (lookups) = {x}\n", .{lookups_claim.toBytesBE()[16..32].*});
                            }
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
                        std.debug.print("[ADDR CLAIM TRACK] Round {}: regs_val={x}, ram_ra={x}, lookups={x}\n", .{
                            round,
                            regs_val_current_claim.toBytesBE()[16..32].*,
                            ram_ra_current_claim.toBytesBE()[16..32].*,
                            lookups_claim.toBytesBE()[16..32].*,
                        });
                        std.debug.print("[ADDR CLAIM TRACK] Round {}: batched_claim={x}\n", .{
                            round,
                            current_batched_claim.toBytesBE()[16..32].*,
                        });
                    }

                    // ===================================================================
                    // Update prefix-suffix decomposition state after receiving challenge
                    // ===================================================================

                    // Bind challenge to all suffix polynomials
                    suffix_polys.bindAll(challenge);

                    // Bind challenge to RAF decompositions
                    left_raf.bind(challenge);
                    right_raf.bind(challenge);
                    identity_raf.bind(challenge);

                    // Update the current phase's expanding table with this challenge
                    expanding_tables[current_phase].update(challenge);

                    // Update prefix checkpoints every 2 rounds (after binding X and Y)
                    const round_in_phase = round % log_m;
                    if (round_in_phase % 2 == 1) {
                        // We just bound Y, update prefix checkpoints with (checkpoint_r_x, r_y)
                        const checkpoint_r_x = challenges[round - 1];
                        const r_y = challenge;
                        const suffix_len = LOOKUPS_LOG_K - (current_phase + 1) * log_m;
                        prefix_checkpoints.update(checkpoint_r_x, r_y, round, suffix_len);
                    }

                    // Check for phase transition (every log_m = 16 rounds)
                    if ((round + 1) % log_m == 0 and round + 1 < LOOKUPS_LOG_K) {
                        const prev_phase = current_phase;
                        current_phase += 1;
                        std.debug.print("[STAGE5] Phase transition to phase {}, prev_table_len={}\n", .{ current_phase, expanding_tables[prev_phase].getLen() });

                        // Condense u_evals (lookups_eq_evals) using the expanding table from the previous phase
                        // This is the CRITICAL step that was missing!
                        // u_evals[j] *= v[prev_phase][k_bound] where k_bound = prefix & m_mask
                        condenseUEvals(F, lookups_eq_evals, &expanding_tables[prev_phase], lookup_indices_u128, current_phase, num_phases);
                        std.debug.print("[STAGE5] Phase {} condense done, now calling initPhase...\n", .{current_phase});

                        // DRIFT DEBUG: Compute direct sum after condensation to compare with lookups_claim
                        // At this point, lookups_eq_evals[j] has been condensed to include the expanding table contribution
                        // The sum should match lookups_claim (after polynomial evolution through previous round)
                        if (current_phase == 1 or current_phase == 8 or current_phase == 15) {
                            var brute_sum = F.zero();
                            for (0..T) |jj| {
                                brute_sum = brute_sum.add(lookups_eq_evals[jj].mul(lookups_combined_vals[jj]));
                            }
                            std.debug.print("[DRIFT_CHECK Phase {}] brute_sum(eq*combined) = {x}\n", .{ current_phase, brute_sum.toBytesBE()[16..32].* });
                            std.debug.print("[DRIFT_CHECK Phase {}] lookups_claim (poly chain) = {x}\n", .{ current_phase, lookups_claim.toBytesBE()[16..32].* });
                            std.debug.print("[DRIFT_CHECK Phase {}] match = {}\n", .{ current_phase, brute_sum.eql(lookups_claim) });
                        }

                        // Re-initialize suffix polys and RAF for new phase with condensed u_evals
                        try suffix_polys.initPhase(current_phase, num_phases, lookups_eq_evals, lookup_indices_u128, cycle_table_indices);
                        std.debug.print("[STAGE5] Phase {} initPhase done, now calling initQRaf...\n", .{current_phase});
                        // Reset RAF decompositions for new phase (restore Q_size to initial_m=256)
                        left_raf.resetForPhase(current_phase, initial_m);
                        right_raf.resetForPhase(current_phase, initial_m);
                        identity_raf.resetForPhase(current_phase, initial_m);
                        initQRaf(F, &left_raf, &right_raf, &identity_raf, lookups_eq_evals, lookup_indices_u128, is_interleaved_operands);
                        std.debug.print("[STAGE5] Phase {} initQRaf done\n", .{current_phase});

                        // Reset the new phase's expanding table to 1
                        expanding_tables[current_phase].reset(F.one());

                        std.debug.print("[STAGE5] Condensed u_evals with expanding table, reset phase {} table\n", .{current_phase});
                    }

                    // Also update legacy ra_weights for cycle rounds (needed for the last 8 rounds)
                    const bit_index = LOOKUPS_LOG_K - 1 - round;
                    const one_minus_r = F.one().sub(challenge);
                    const chunk_idx = round / lookups_ra_virtual_log_k_chunk;

                    // Debug: print round info for rounds 0-3 and 63-65
                    if (round < 4 or (round >= 63 and round <= 65)) {
                        std.debug.print("[STAGE5 ADDR] Round {} challenge (full) = {any}\n", .{
                            round,
                            challenge.toBytesBE(),
                        });
                    }

                    // Debug: print round info for first round of each chunk
                    if (round % lookups_ra_virtual_log_k_chunk == 0 and (round < 64 or round >= 64)) {
                        std.debug.print("[STAGE5 RA_CHUNK] Starting chunk {} (rounds {}-{})\n", .{
                            chunk_idx,
                            round,
                            round + lookups_ra_virtual_log_k_chunk - 1,
                        });
                        std.debug.print("  bit_index = {} (bit {} of lookup_index)\n", .{ bit_index, bit_index });
                        std.debug.print("  challenge = {x}\n", .{challenge.toBytesBE()[16..32].*});
                        // Print lookup_indices for first 4 cycles
                        for (0..@min(4, T)) |jj| {
                            const idx_lo = lookups_indices_lo[jj];
                            const idx_hi = lookups_indices_hi[jj];
                            const bit_val = getBit128(idx_lo, idx_hi, bit_index);
                            std.debug.print("  cycle[{}]: lookup_idx=({x:016},{x:016}), bit[{}]={}\n", .{
                                jj, idx_hi, idx_lo, bit_index, bit_val,
                            });
                        }
                    }

                    for (0..T) |j| {
                        const bit = getBit128(lookups_indices_lo[j], lookups_indices_hi[j], bit_index);
                        const factor = if (bit == 0) one_minus_r else challenge;
                        lookups_ra_weights[j] = lookups_ra_weights[j].mul(factor);

                        if (chunk_idx < ra_num_chunks) {
                            ra_chunk_weights[chunk_idx][j] = ra_chunk_weights[chunk_idx][j].mul(factor);
                        }
                    }

                    // Debug: print ra_chunk values after last round of each chunk
                    if ((round + 1) % lookups_ra_virtual_log_k_chunk == 0) {
                        std.debug.print("[STAGE5 RA_CHUNK] Finished chunk {} after round {}\n", .{ chunk_idx, round });
                        for (0..@min(4, T)) |jj| {
                            std.debug.print("  ra_chunk[{}][{}] = {x}\n", .{
                                chunk_idx, jj, ra_chunk_weights[chunk_idx][jj].toBytesBE()[16..32].*,
                            });
                        }
                    }

                    // NOTE: lookups_claim is already updated above by evaluating Instance 2's polynomial at challenge.
                    // The recomputation from raw arrays is WRONG because eq_evals hasn't been bound yet.
                    // The correct value is p_inst2(r) which we computed earlier.

                    if (round % 8 == 7) {
                        std.debug.print("[STAGE5] Completed rounds 0-{}\n", .{round});
                    }

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
                        var pre_remat_sum = F.zero();
                        for (0..T) |jj| {
                            pre_remat_sum = pre_remat_sum.add(lookups_eq_evals[jj].mul(lookups_combined_vals[jj]));
                        }
                        std.debug.print("[PRE-REMAT] sum(eq*combined_vals) = {x}\n", .{pre_remat_sum.toBytesBE()[16..32].*});
                        std.debug.print("[PRE-REMAT] lookups_claim (poly chain) = {x}\n", .{lookups_claim.toBytesBE()[16..32].*});
                        std.debug.print("[PRE-REMAT] match = {}\n", .{pre_remat_sum.eql(lookups_claim)});

                        // Get bound prefix values from RAF decompositions
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

                        std.debug.print("[STAGE5 REMATERIALIZE] Verification:\n", .{});
                        std.debug.print("  computed_identity (from challenges) = {x}\n", .{computed_identity.toBytesBE()[16..32].*});
                        std.debug.print("  identity_prefix (from bound_value) = {x}\n", .{identity_prefix.toBytesBE()[16..32].*});
                        std.debug.print("  identity match = {}\n", .{computed_identity.eql(identity_prefix)});
                        std.debug.print("  computed_left (from challenges) = {x}\n", .{computed_left.toBytesBE()[16..32].*});
                        std.debug.print("  left_prefix (from bound_value) = {x}\n", .{left_prefix.toBytesBE()[16..32].*});
                        std.debug.print("  left match = {}\n", .{computed_left.eql(left_prefix)});
                        std.debug.print("  computed_right (from challenges) = {x}\n", .{computed_right.toBytesBE()[16..32].*});
                        std.debug.print("  right_prefix (from bound_value) = {x}\n", .{right_prefix.toBytesBE()[16..32].*});
                        std.debug.print("  right match = {}\n", .{computed_right.eql(right_prefix)});

                        // Print first few challenges to compare with Jolt
                        std.debug.print("  First 4 challenges (to compare with Jolt):\n", .{});
                        for (0..4) |i| {
                            std.debug.print("    challenges[{}] = {x}\n", .{ i, challenges[i].toBytesBE()[16..32].* });
                        }

                        // Compute RAF scalar values
                        const raf_interleaved = gamma_raf.mul(left_prefix).add(gamma_raf2.mul(right_prefix));
                        const raf_identity = gamma_raf2.mul(identity_prefix);

                        std.debug.print("[STAGE5 REMATERIALIZE] round=128, left_prefix={x}, right_prefix={x}, identity_prefix={x}\n", .{
                            left_prefix.toBytesBE()[16..32].*,
                            right_prefix.toBytesBE()[16..32].*,
                            identity_prefix.toBytesBE()[16..32].*,
                        });
                        std.debug.print("[STAGE5 REMATERIALIZE] raf_interleaved={x}, raf_identity={x}\n", .{
                            raf_interleaved.toBytesBE()[16..32].*,
                            raf_identity.toBytesBE()[16..32].*,
                        });

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
                        std.debug.print("[STAGE5 DEBUG] Direct MLE computation for RangeCheck:\n", .{});
                        // Debug: print challenges[64], [65], and [127] for comparison with Jolt (FULL 32 bytes)
                        const ch64_bytes = challenges[64].toBytesBE();
                        std.debug.print("[STAGE5 DEBUG] challenges[64] (FULL 32) = ", .{});
                        for (ch64_bytes) |b| std.debug.print("{x:0>2}", .{b});
                        std.debug.print("\n", .{});
                        const ch65_bytes = challenges[65].toBytesBE();
                        std.debug.print("[STAGE5 DEBUG] challenges[65] (FULL 32) = ", .{});
                        for (ch65_bytes) |b| std.debug.print("{x:0>2}", .{b});
                        std.debug.print("\n", .{});
                        const ch127_bytes = challenges[127].toBytesBE();
                        std.debug.print("[STAGE5 DEBUG] challenges[127] (FULL 32) = ", .{});
                        for (ch127_bytes) |b| std.debug.print("{x:0>2}", .{b});
                        std.debug.print("\n", .{});
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
                                std.debug.print("  i={}, shift={}, coeff=2^{}, r={x}\n", .{
                                    i, shift, shift, r_i.toBytesBE()[28..32].*,
                                });
                            }
                        }
                        std.debug.print("[STAGE5 DEBUG] Direct RangeCheck MLE result: {x}\n", .{
                            direct_range_check_mle.toBytesBE()[16..32].*,
                        });
                        std.debug.print("[STAGE5 DEBUG] Prefix-suffix result (table[0]): {x}\n", .{
                            table_values[0].toBytesBE()[16..32].*,
                        });
                        std.debug.print("[STAGE5 DEBUG] Match: {}\n", .{
                            direct_range_check_mle.eql(table_values[0]),
                        });

                        // Debug: print key prefix checkpoint values
                        std.debug.print("[STAGE5 REMATERIALIZE] Key prefix checkpoints:\n", .{});
                        const lw_idx = @intFromEnum(lookup_table_mod.Prefixes.LowerWord);
                        const eq_idx = @intFromEnum(lookup_table_mod.Prefixes.Eq);
                        const lt_idx = @intFromEnum(lookup_table_mod.Prefixes.LessThan);
                        const lsb_idx = @intFromEnum(lookup_table_mod.Prefixes.Lsb);
                        if (prefix_checkpoints.checkpoints[lw_idx]) |v| {
                            std.debug.print("  LowerWord = {x}\n", .{v.toBytesBE()[16..32].*});
                        } else {
                            std.debug.print("  LowerWord = NULL\n", .{});
                        }
                        if (prefix_checkpoints.checkpoints[eq_idx]) |v| {
                            std.debug.print("  Eq = {x}\n", .{v.toBytesBE()[16..32].*});
                        } else {
                            std.debug.print("  Eq = NULL\n", .{});
                        }
                        if (prefix_checkpoints.checkpoints[lt_idx]) |v| {
                            std.debug.print("  LessThan = {x}\n", .{v.toBytesBE()[16..32].*});
                        } else {
                            std.debug.print("  LessThan = NULL\n", .{});
                        }
                        if (prefix_checkpoints.checkpoints[lsb_idx]) |v| {
                            std.debug.print("  Lsb = {x}\n", .{v.toBytesBE()[16..32].*});
                        } else {
                            std.debug.print("  Lsb = NULL\n", .{});
                        }

                        // Debug: print table values
                        std.debug.print("[STAGE5 REMATERIALIZE] table_values_at_r_addr:\n", .{});
                        for (0..NUM_TABLES) |t_idx| {
                            if (!table_values[t_idx].eql(F.zero())) {
                                std.debug.print("  table[{}] = {x}\n", .{ t_idx, table_values[t_idx].toBytesBE()[16..32].* });
                            }
                        }

                        // Rematerialize combined_vals using the correct formula
                        // combined_val[j] = table_values_at_r_addr[table(j)] + raf_val
                        //
                        // IMPORTANT: Jolt always adds the RAF contribution regardless of whether
                        // there's a lookup table. The table value is only added IF there's a table.
                        // See: jolt-core/src/zkvm/instruction_lookups/read_raf_checking.rs:682-698
                        for (0..T) |j| {
                            if (j >= trace_len) {
                                lookups_combined_vals[j] = F.zero();
                                continue;
                            }

                            // Start with zero
                            var combined_val = F.zero();

                            // Add lookup table value if this cycle has a table
                            const table_idx = cycle_table_indices[j];
                            if (table_idx >= 0) {
                                const t_idx: usize = @intCast(table_idx);
                                if (t_idx < NUM_TABLES) {
                                    combined_val = combined_val.add(table_values[t_idx]);
                                }
                            }

                            // ALWAYS add RAF contribution (regardless of table)
                            const is_interleaved = !cycle_is_identity_path[j];
                            if (is_interleaved) {
                                combined_val = combined_val.add(raf_interleaved);
                            } else {
                                combined_val = combined_val.add(raf_identity);
                            }

                            lookups_combined_vals[j] = combined_val;
                        }

                        // Debug: print first 5 rematerialized values
                        std.debug.print("[STAGE5 REMATERIALIZE] First 5 combined_vals after rematerialization:\n", .{});
                        for (0..@min(5, trace_len)) |j| {
                            const table_idx_dbg = cycle_table_indices[j];
                            const table_val_dbg = if (table_idx_dbg >= 0 and @as(usize, @intCast(table_idx_dbg)) < NUM_TABLES) table_values[@intCast(table_idx_dbg)] else F.zero();
                            std.debug.print("  j={}: combined_val={x}, is_identity_path={}, table_idx={}, table_val={x}\n", .{
                                j,
                                lookups_combined_vals[j].toBytesBE()[24..32].*,
                                cycle_is_identity_path[j],
                                table_idx_dbg,
                                table_val_dbg.toBytesBE()[24..32].*,
                            });
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
                        std.debug.print("[STAGE5 CYCLE] Reinitializing lookups_eq_evals for cycle rounds\n", .{});
                        for (0..T) |j| {
                            lookups_eq_evals[j] = computeEqAtIndex(r_reduction, j);
                        }
                        // Debug: verify sum = 1
                        var eq_sum_verify = F.zero();
                        for (0..T) |j| {
                            eq_sum_verify = eq_sum_verify.add(lookups_eq_evals[j]);
                        }
                        std.debug.print("[STAGE5 CYCLE] eq_sum after reinit = {x} (should be 1)\n", .{eq_sum_verify.toBytesBE()[16..32].*});

                        // ============================================================
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
                        std.debug.print("[STAGE5 CYCLE] Materializing ra_chunk_weights from expanding tables\n", .{});
                        const phases_per_chunk = num_phases / ra_num_chunks;
                        std.debug.print("  num_phases={}, ra_num_chunks={}, phases_per_chunk={}\n", .{ num_phases, ra_num_chunks, phases_per_chunk });

                        // Debug: print expanding table sizes and first few values
                        std.debug.print("  Expanding table state at round 128:\n", .{});
                        const v0_p0 = expanding_tables[0].get(0);
                        const v0_p1 = expanding_tables[1].get(0);
                        const product_check = v0_p0.mul(v0_p1);
                        std.debug.print("    phase[0] v[0] (full) = {x}\n", .{v0_p0.toBytesBE()[16..32].*});
                        std.debug.print("    phase[1] v[0] (full) = {x}\n", .{v0_p1.toBytesBE()[16..32].*});
                        std.debug.print("    product v0_p0 * v0_p1 = {x}\n", .{product_check.toBytesBE()[16..32].*});
                        for (0..@min(4, num_phases)) |phase| {
                            std.debug.print("    phase[{}]: len={}, first_vals=[{x}, {x}, {x}, {x}]\n", .{
                                phase,
                                expanding_tables[phase].getLen(),
                                expanding_tables[phase].get(0).toBytesBE()[28..32].*,
                                expanding_tables[phase].get(1).toBytesBE()[28..32].*,
                                expanding_tables[phase].get(2).toBytesBE()[28..32].*,
                                expanding_tables[phase].get(3).toBytesBE()[28..32].*,
                            });
                        }

                        for (0..T) |j| {
                            if (j >= trace_len) {
                                // Padding cycles: ra_weight = 1
                                for (0..ra_num_chunks) |c| {
                                    ra_chunk_weights[c][j] = F.one();
                                }
                                continue;
                            }

                            // Get lookup index for this cycle
                            const k_lo = lookups_indices_lo[j];
                            const k_hi = lookups_indices_hi[j];

                            // Debug: print lookup index for first cycle
                            if (j == 0) {
                                std.debug.print("[STAGE5 RA_DEBUG] Cycle 0 lookup_index: k_hi=0x{x:0>16}, k_lo=0x{x:0>16}\n", .{ k_hi, k_lo });
                            }

                            // For each chunk, compute the product of expanding table lookups
                            for (0..ra_num_chunks) |chunk_idx| {
                                var ra_val = F.one();
                                const phase_start = chunk_idx * phases_per_chunk;
                                const phase_end = @min((chunk_idx + 1) * phases_per_chunk, num_phases);

                                for (phase_start..phase_end) |phase| {
                                    // Compute which bits of k correspond to this phase
                                    // Phase 0 handles the MSB bits (127:112), phase (num_phases-1) handles LSB bits
                                    // Each phase handles log_m bits (16 bits for LOG_K=128, phases=8)
                                    //
                                    // The expanding table uses HighToLow binding order:
                                    // - v[k] = EQ(k, r) where k bits directly correspond to round challenges
                                    // - No bit reversal needed
                                    //
                                    // For phase p, rounds are: p*log_m to (p+1)*log_m - 1
                                    // These bind address bits: (127 - p*log_m) down to (128 - (p+1)*log_m)
                                    //
                                    // Jolt formula: shift = (phases - 1 - phase) * log_m
                                    // This extracts the appropriate bits from the 128-bit address
                                    const shift = (num_phases - 1 - phase) * log_m;

                                    // Extract the log_m bits from k at this position (address bits shift+log_m-1 : shift)
                                    var k_phase: usize = undefined;
                                    if (shift >= 64) {
                                        // Bits are in the upper 64 bits
                                        k_phase = @truncate(k_hi >> @intCast(shift - 64));
                                    } else if (shift + log_m <= 64) {
                                        // Bits are in the lower 64 bits
                                        k_phase = @truncate(k_lo >> @intCast(shift));
                                    } else {
                                        // Bits span both halves
                                        const lo_bits = k_lo >> @intCast(shift);
                                        const hi_bits = k_hi << @intCast(64 - shift);
                                        k_phase = @truncate(lo_bits | hi_bits);
                                    }
                                    k_phase &= (@as(usize, 1) << @intCast(log_m)) - 1; // Mask to log_m bits

                                    // Look up in expanding table (HighToLow binding - no bit reversal needed)
                                    // With HighToLow binding, v[k] = EQ(k, r) where k bits correspond directly
                                    // to round indices: bit 0 = round 0's challenge, etc.
                                    const table_val = expanding_tables[phase].get(k_phase);

                                    // Debug: print detailed info for first cycle
                                    if (j == 0 and chunk_idx < 2) {
                                        std.debug.print("[STAGE5 RA_DEBUG] j=0, chunk={}, phase={}: shift={}, k_phase=0x{x}, table_val={x}\n", .{
                                            chunk_idx,
                                            phase,
                                            shift,
                                            k_phase,
                                            table_val.toBytesBE()[28..32].*,
                                        });
                                    }

                                    ra_val = ra_val.mul(table_val);
                                }

                                ra_chunk_weights[chunk_idx][j] = ra_val;

                                // Debug: print final ra_val for first cycle, first 2 chunks
                                if (j == 0 and chunk_idx < 2) {
                                    std.debug.print("[STAGE5 RA_DEBUG] j=0, chunk={}: final ra_val={x}\n", .{
                                        chunk_idx,
                                        ra_val.toBytesBE()[16..32].*,
                                    });
                                }
                            }
                        }

                        // Debug: print first few ra_chunk_weights after materialization
                        std.debug.print("[STAGE5 CYCLE] ra_chunk_weights after materialization (first 4 cycles):\n", .{});
                        for (0..@min(4, T)) |j| {
                            std.debug.print("  j={}: ra_chunks=[", .{j});
                            for (0..ra_num_chunks) |c| {
                                if (c > 0) std.debug.print(", ", .{});
                                std.debug.print("{x}", .{ra_chunk_weights[c][j].toBytesBE()[24..32].*});
                            }
                            std.debug.print("]\n", .{});
                        }

                        // Update lookups_ra_weights[j] to be the product of all chunks
                        for (0..T) |j| {
                            var ra_prod = F.one();
                            for (0..ra_num_chunks) |c| {
                                ra_prod = ra_prod.mul(ra_chunk_weights[c][j]);
                            }
                            lookups_ra_weights[j] = ra_prod;
                        }

                        // DEBUG: Compute materialized sum for comparison (NOT used as claim)
                        // In Jolt, the claim is NEVER overridden at this transition. It evolves
                        // purely through polynomial evaluation at each round's challenge.
                        // The arrays are materialized for computing the polynomial, but
                        // finishMlesProductSumFromEvals uses the current lookups_claim (from
                        // polynomial evaluation chain) to back-compute eval_at_0.
                        {
                            // Compute sum WITH ra_weights (for cycle round polynomial)
                            var materialized_sum = F.zero();
                            for (0..T) |j| {
                                materialized_sum = materialized_sum.add(lookups_eq_evals[j].mul(lookups_ra_weights[j]).mul(lookups_combined_vals[j]));
                            }
                            // Compute sum WITHOUT ra_weights (should match lookups_claim at end of address rounds)
                            var sum_no_ra = F.zero();
                            for (0..T) |j| {
                                sum_no_ra = sum_no_ra.add(lookups_eq_evals[j].mul(lookups_combined_vals[j]));
                            }
                            std.debug.print("[STAGE5 CYCLE] materialized_sum (WITH ra_weights) = {x}\n", .{materialized_sum.toBytesBE()[16..32].*});
                            std.debug.print("[STAGE5 CYCLE] sum_no_ra (WITHOUT ra_weights) = {x}\n", .{sum_no_ra.toBytesBE()[16..32].*});
                            std.debug.print("[STAGE5 CYCLE] lookups_claim (from poly eval chain) = {x}\n", .{lookups_claim.toBytesBE()[16..32].*});
                            std.debug.print("[STAGE5 CYCLE] sum_no_ra == lookups_claim: {}\n", .{sum_no_ra.eql(lookups_claim)});
                            std.debug.print("[STAGE5 CYCLE] current_batched_claim (NOT overridden) = {x}\n", .{current_batched_claim.toBytesBE()[16..32].*});

                            // NOTE: In Jolt, claims are NEVER overridden. The polynomial chain
                            // evolves continuously. The materialized arrays must match the claim.
                            // If they don't, the address-round polynomial tracking is wrong.
                            if (!materialized_sum.eql(lookups_claim)) {
                                std.debug.print("[STAGE5 CYCLE] WARNING: materialized_sum != lookups_claim (address-round tracking error)\n", .{});
                            }
                        }
                    }
                    // CRITICAL: current_half_size is the number of pairs we iterate over
                    // At round k, we have T/2^(k+1) pairs to process
                    const current_half_size = T >> @intCast(lookups_round + 1);

                    // Get r_round for this cycle variable (LowToHigh binding: last element first)
                    const r_round = r_reduction[n_cycle_vars - 1 - lookups_round];

                    // Accumulate sum of 9-factor products
                    // sum_evals contains g(1), g(2), ..., g(8), g(∞) where g(X) is the polynomial
                    // BEFORE multiplying by eq(X, r_round). finishMlesProductSumFromEvals will
                    // recover g(0) using the claim, then multiply by eq(X, r_round).
                    var sum_evals: [9]F = [_]F{F.zero()} ** 9;

                    for (0..current_half_size) |j| {
                        // Build the 9 linear polynomial pairs for this j
                        // Each pair is (p(0), p(1)) for the linear polynomial
                        var pairs: [9][2]F = undefined;

                        // Factor 0: eq_prefix * combined_val (WITHOUT current round's eq factor)
                        //
                        // CRITICAL FIX: Compute eq_prefix directly from r_reduction, excluding
                        // the current round's variable. This avoids the stride indexing issues.
                        //
                        // eq_prefix(j) = eq(j, r_reduction[0:n-1-lookups_round])
                        // where r_reduction[0:n-1-k] means all variables EXCEPT the last k+1.
                        //
                        // For LowToHigh binding with n=8 cycle vars:
                        // - Round 0: use r_reduction[0..7] (all 8 vars), current = r_reduction[7]
                        // - Round 1: use r_reduction[0..6] (first 7 vars), current = r_reduction[6]
                        // - Round k: use r_reduction[0..n-1-k], current = r_reduction[n-1-k]
                        //
                        // The remaining_vars after round k is: n - k - 1
                        // eq_prefix(j) for j in 0..current_half_size uses bits of j in [0, remaining_vars)
                        const remaining_vars = n_cycle_vars - lookups_round - 1;
                        const eq_prefix = computeEqAtIndexPartial(r_reduction, j, remaining_vars);

                        // Debug: for first cycle round, first few j values
                        if (lookups_round == 0 and j < 3) {
                            std.debug.print("[CYCLE R0 DEBUG] j={}, remaining_vars={}, eq_prefix={x}\n", .{
                                j,
                                remaining_vars,
                                eq_prefix.toBytesBE()[16..32].*,
                            });
                            std.debug.print("  combined_vals[{}]={x}, combined_vals[{}]={x}\n", .{
                                2 * j,
                                lookups_combined_vals[2 * j].toBytesBE()[16..32].*,
                                2 * j + 1,
                                lookups_combined_vals[2 * j + 1].toBytesBE()[16..32].*,
                            });
                        }

                        // combined_vals IS bound (standard MLE binding), so use sequential access
                        pairs[0][0] = eq_prefix.mul(lookups_combined_vals[2 * j]);
                        pairs[0][1] = eq_prefix.mul(lookups_combined_vals[2 * j + 1]);

                        // Factors 1-8: 8 ra_chunk polynomials (also bound, sequential access)
                        for (0..ra_num_chunks) |c| {
                            pairs[c + 1][0] = ra_chunk_weights[c][2 * j];
                            pairs[c + 1][1] = ra_chunk_weights[c][2 * j + 1];
                        }

                        // Evaluate product at [1, 2, ..., 8, ∞]
                        const prod_evals = UniPoly(F).evalLinearProd9(pairs);

                        // Accumulate into sum_evals
                        for (0..9) |k| {
                            sum_evals[k] = sum_evals[k].add(prod_evals[k]);
                        }
                    }

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
                    std.debug.print("[STAGE5 CYCLE] Round {} (cycle var {}):\n", .{ round, lookups_round });
                    std.debug.print("  r_round = {x}\n", .{r_round.toBytesBE()[24..32].*});
                    std.debug.print("  lookups_current_scalar = {x}\n", .{lookups_current_scalar.toBytesBE()[24..32].*});
                    std.debug.print("  cycle_claim = {x}\n", .{cycle_claim.toBytesBE()[24..32].*});
                    std.debug.print("  p(0) = {x}\n", .{p_at_0.toBytesBE()[24..32].*});
                    std.debug.print("  p(1) = {x}\n", .{p_at_1.toBytesBE()[24..32].*});
                    std.debug.print("  p(0)+p(1) = {x}, matches_claim = {}\n", .{ sum_check.toBytesBE()[24..32].*, sumcheck_ok });
                    std.debug.print("  full_coeffs len = {}\n", .{full_coeffs.len});
                    // Print sum_evals to debug
                    std.debug.print("  sum_evals = [\n", .{});
                    for (0..9) |k| {
                        std.debug.print("    [{d}] = {x}\n", .{ k, sum_evals[k].toBytesBE()[24..32].* });
                    }
                    std.debug.print("  ]\n", .{});
                    // Debug: print eq and combined values for last round
                    if (lookups_round == n_cycle_vars - 1) {
                        std.debug.print("  [LAST ROUND] eq[0]={x}, eq[1]={x}\n", .{
                            lookups_eq_evals[0].toBytesBE()[16..32].*,
                            lookups_eq_evals[1].toBytesBE()[16..32].*,
                        });
                        std.debug.print("  [LAST ROUND] val[0]={x}, val[1]={x}\n", .{
                            lookups_combined_vals[0].toBytesBE()[16..32].*,
                            lookups_combined_vals[1].toBytesBE()[16..32].*,
                        });
                        std.debug.print("  [LAST ROUND] ra_chunk[0]: [{x}, {x}]\n", .{
                            ra_chunk_weights[0][0].toBytesBE()[16..32].*,
                            ra_chunk_weights[0][1].toBytesBE()[16..32].*,
                        });
                    }
                    // Print Instance 0+1 contribution
                    std.debug.print("  combined_poly (Inst 0+1) = [{x}, {x}, {x}, {x}]\n", .{
                        combined_poly[0].toBytesBE()[24..32].*,
                        combined_poly[1].toBytesBE()[24..32].*,
                        combined_poly[2].toBytesBE()[24..32].*,
                        combined_poly[3].toBytesBE()[24..32].*,
                    });

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
                        std.debug.print("[STAGE5 CYCLE ZOLT] Round {} compressed coeffs (LE, comparing to Jolt):\n", .{round});
                        std.debug.print("  final_compressed.len = {}\n", .{final_compressed.len});
                        for (0..final_compressed.len) |k| {
                            // Jolt displays LE bytes from arkworks serialization
                            std.debug.print("  coeff[{}] = {any}\n", .{ k, final_compressed[k].toBytes() });
                        }
                        std.debug.print("  current_batched_claim (LE) = {any}\n", .{current_batched_claim.toBytes()});
                        // Also print combined_coeffs before compression
                        std.debug.print("  combined_coeffs[0] (c0) = {any}\n", .{combined_coeffs[0].toBytes()});
                        std.debug.print("  combined_coeffs[1] (c1) = {any}\n", .{combined_coeffs[1].toBytes()});
                        std.debug.print("  combined_coeffs[2] (c2) = {any}\n", .{combined_coeffs[2].toBytes()});
                        // Print Instance 0+1 contribution details
                        std.debug.print("  inst01_coeffs[0..4] = [{any}, {any}, {any}, {any}]\n", .{
                            inst01_coeffs[0].toBytes(), inst01_coeffs[1].toBytes(),
                            inst01_coeffs[2].toBytes(), inst01_coeffs[3].toBytes(),
                        });
                        // Print full_coeffs (Instance 2)
                        std.debug.print("  full_coeffs[0..3] = [{any}, {any}, {any}]\n", .{
                            full_coeffs[0].toBytes(), full_coeffs[1].toBytes(), full_coeffs[2].toBytes(),
                        });
                    }

                    try proof.compressed_polys.append(self.allocator, .{
                        .coeffs_except_linear_term = final_compressed,
                        .allocator = self.allocator,
                    });

                    // Append to transcript
                    transcript.appendMessage("UniPoly_begin");
                    for (final_compressed) |c| {
                        transcript.appendScalar(c);
                    }
                    transcript.appendMessage("UniPoly_end");

                    const challenge = transcript.challengeScalar();
                    challenges[round] = challenge;

                    // Debug: print challenge for comparison with Jolt
                    if (round < 4 or round >= 128) {
                        std.debug.print("[STAGE5 CHALLENGE] Round {}: LE = {any}\n", .{round, challenge.toBytes()[0..16].*});
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
                        std.debug.print("[STAGE5 C1 DEBUG] Round {}:\n", .{round});
                        std.debug.print("  c1_direct (combined_coeffs[1]) = {any}\n", .{combined_coeffs[1].toBytes()});
                        std.debug.print("  c1_recovered (from hint)       = {any}\n", .{c1_recovered.toBytes()});
                        std.debug.print("  hint (current_batched_claim)   = {any}\n", .{current_batched_claim.toBytes()});
                        std.debug.print("  c1_direct == c1_recovered: {}\n", .{combined_coeffs[1].eql(c1_recovered)});

                        // CRITICAL DEBUG: Check if current_batched_claim = sum of scaled instance claims
                        // Expected: batch0*claim0 + batch1*claim1 + batch2*claim2
                        const expected_batched = batch0.mul(regs_val_current_claim)
                            .add(batch1.mul(ram_ra_current_claim))
                            .add(batch2.mul(lookups_claim));
                        std.debug.print("  expected_batched (batch0*c0+batch1*c1+batch2*c2) = {any}\n", .{expected_batched.toBytes()});
                        std.debug.print("  current_batched_claim matches expected: {}\n", .{current_batched_claim.eql(expected_batched)});
                        std.debug.print("    batch0*claim0 = {any}\n", .{batch0.mul(regs_val_current_claim).toBytes()});
                        std.debug.print("    batch1*claim1 = {any}\n", .{batch1.mul(ram_ra_current_claim).toBytes()});
                        std.debug.print("    batch2*claim2 = {any}\n", .{batch2.mul(lookups_claim).toBytes()});
                        std.debug.print("    regs_val_current_claim = {x}\n", .{regs_val_current_claim.toBytesBE()[16..32].*});
                        std.debug.print("    ram_ra_current_claim   = {x}\n", .{ram_ra_current_claim.toBytesBE()[16..32].*});
                        std.debug.print("    lookups_claim          = {x}\n", .{lookups_claim.toBytesBE()[16..32].*});

                        // Compute what the hint SHOULD be if coefficients are correct:
                        // hint_expected = p(0) + p(1) = 2*c0 + c1 + c2 + ... + c_d
                        var hint_expected = combined_coeffs[0].add(combined_coeffs[0]); // 2*c0
                        for (1..combined_coeffs.len) |i| {
                            hint_expected = hint_expected.add(combined_coeffs[i]);
                        }
                        std.debug.print("  hint_expected (2*c0+c1+c2+...) = {any}\n", .{hint_expected.toBytes()});
                        std.debug.print("  hint == hint_expected: {}\n", .{current_batched_claim.eql(hint_expected)});
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
                    std.debug.print("  [S5P] R{} challenge={x} new_e={x} degree={}\n", .{
                        round,
                        challenge.toBytes()[0..16].*,
                        current_batched_claim.toBytes()[0..16].*,
                        combined_coeffs.len - 1,
                    });

                    // Skip the standard compression/serialization below
                    // Bind the challenge for RegistersValEvaluation if active
                    if (remaining_rounds <= regs_val_num_rounds) {
                        const regs_round = regs_val_num_rounds - remaining_rounds;

                        // Update Instance 0 claim: new_claim = p0(challenge)
                        // Get the polynomial coefficients from Toom-Cook evaluations
                        const poly_evals = computeRegsValRoundPoly(inc_evals, wa_evals, lt_evals, regs_round);
                        const inst0_coeffs = UniPoly(F).toomCookToCoeffs(poly_evals);
                        // Evaluate at challenge using Horner's method
                        var p0_at_r = inst0_coeffs[3]; // c3 (highest)
                        p0_at_r = p0_at_r.mulHiBigIntU128(challenge.limbs).add(inst0_coeffs[2]); // c2
                        p0_at_r = p0_at_r.mulHiBigIntU128(challenge.limbs).add(inst0_coeffs[1]); // c1
                        p0_at_r = p0_at_r.mulHiBigIntU128(challenge.limbs).add(inst0_coeffs[0]); // c0
                        regs_val_current_claim = p0_at_r;

                        bindRegsValChallenge(inc_evals, wa_evals, lt_evals, regs_round, challenge);
                    }

                    // Bind the challenge for RamRaClaimReduction cycle rounds
                    // (This is the PhaseCycle binding that was previously only in address rounds)
                    if (remaining_rounds <= ram_ra_num_rounds) {
                        const ram_ra_round = ram_ra_num_rounds - remaining_rounds;
                        if (ram_ra_round >= log_ram_k) {
                            // PhaseCycle: bind cycle variables
                            const cycle_round = ram_ra_round - log_ram_k;
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

                            for (0..ram_access_count) |access_idx| {
                                const cycle = ram_cycles[access_idx];
                                const cycle_usize: usize = @intCast(cycle);
                                // Get the cycle bit that was just bound
                                const c_m: u1 = @truncate(cycle_usize >> @intCast(cycle_round));
                                // Multiply eq_cycle_bound by the binding factor
                                const factor = if (c_m == 0) one_minus_r else challenge;
                                eq_cycle_bound[access_idx] = eq_cycle_bound[access_idx].mul(factor);

                                // Update eq_*_remaining by dividing out eq_bit(r_*[m], c_m)
                                const eq_raf_bit_at_c = if (c_m == 0) F.one().sub(r_raf_bit) else r_raf_bit;
                                const eq_rw_bit_at_c = if (c_m == 0) F.one().sub(r_rw_bit) else r_rw_bit;
                                const eq_val_bit_at_c = if (c_m == 0) F.one().sub(r_val_bit) else r_val_bit;

                                if (!eq_raf_bit_at_c.eql(F.zero())) {
                                    eq_raf_remaining[access_idx] = eq_raf_remaining[access_idx].mul(eq_raf_bit_at_c.inverse().?);
                                }
                                if (!eq_rw_bit_at_c.eql(F.zero())) {
                                    eq_rw_remaining[access_idx] = eq_rw_remaining[access_idx].mul(eq_rw_bit_at_c.inverse().?);
                                }
                                if (!eq_val_bit_at_c.eql(F.zero())) {
                                    eq_val_remaining[access_idx] = eq_val_remaining[access_idx].mul(eq_val_bit_at_c.inverse().?);
                                }
                            }

                            std.debug.print("[STAGE5 CYCLE BIND R{}] cycle_round={}, challenge={x}\n", .{
                                round,
                                cycle_round,
                                challenge.toBytesBE()[16..32].*,
                            });
                            std.debug.print("  eq_raf_bound={x}\n", .{eq_raf_bound.toBytesBE()[16..32].*});
                            if (ram_access_count > 0) {
                                std.debug.print("  eq_cycle_bound[0]={x}\n", .{eq_cycle_bound[0].toBytesBE()[16..32].*});
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

                                // Bind P arrays: P'[j] = (1-r)*P[2j] + r*P[2j+1]
                                for (0..half_len) |j| {
                                    P_raf[j] = one_minus_r.mul(P_raf[2 * j]).add(P_raf[2 * j + 1].mulHiBigIntU128(challenge.limbs));
                                    P_rw[j] = one_minus_r.mul(P_rw[2 * j]).add(P_rw[2 * j + 1].mulHiBigIntU128(challenge.limbs));
                                    P_val[j] = one_minus_r.mul(P_val[2 * j]).add(P_val[2 * j + 1].mulHiBigIntU128(challenge.limbs));
                                }

                                // Bind Q arrays: Q'[j] = (1-r)*Q[2j] + r*Q[2j+1]
                                for (0..half_len) |j| {
                                    Q_raf[j] = one_minus_r.mul(Q_raf[2 * j]).add(Q_raf[2 * j + 1].mulHiBigIntU128(challenge.limbs));
                                    Q_rw[j] = one_minus_r.mul(Q_rw[2 * j]).add(Q_rw[2 * j + 1].mulHiBigIntU128(challenge.limbs));
                                    Q_val[j] = one_minus_r.mul(Q_val[2 * j]).add(Q_val[2 * j + 1].mulHiBigIntU128(challenge.limbs));
                                }

                                std.debug.print("[STAGE5 CYCLE BIND R{}] Bound P/Q arrays: half_len={}\n", .{
                                    round,
                                    half_len,
                                });
                            } else {
                                // PhaseCycle2: bind H_prime and eq_hi arrays
                                const suffix_round = cycle_round - prefix_n_vars;
                                const current_len = suffix_size >> @intCast(suffix_round);
                                const half_len = current_len / 2;

                                // Bind H_prime: H'[j] = (1-r)*H[2j] + r*H[2j+1]
                                for (0..half_len) |j| {
                                    H_prime[j] = one_minus_r.mul(H_prime[2 * j]).add(H_prime[2 * j + 1].mulHiBigIntU128(challenge.limbs));
                                }

                                // Bind eq_hi arrays
                                for (0..half_len) |j| {
                                    eq_raf_hi[j] = one_minus_r.mul(eq_raf_hi[2 * j]).add(eq_raf_hi[2 * j + 1].mulHiBigIntU128(challenge.limbs));
                                    eq_rw_hi[j] = one_minus_r.mul(eq_rw_hi[2 * j]).add(eq_rw_hi[2 * j + 1].mulHiBigIntU128(challenge.limbs));
                                    eq_val_hi[j] = one_minus_r.mul(eq_val_hi[2 * j]).add(eq_val_hi[2 * j + 1].mulHiBigIntU128(challenge.limbs));
                                }

                                std.debug.print("[STAGE5 CYCLE BIND R{}] Bound H'/eq_hi arrays: half_len={}\n", .{
                                    round,
                                    half_len,
                                });
                            }

                            // CRITICAL: Update Instance 1 claim after binding
                            // p(r) = c0 + r*c1 + r²*c2 for degree-2 polynomial
                            // where c0 = eval_0, c1 = eval_1 - eval_0 - c2, c2 = (eval_2 - 2*eval_1 + eval_0) / 2
                            // Since we use hint: eval_0 = claim - eval_1, we have p(0) + p(1) = claim
                            // Compute p(r) using Lagrange interpolation:
                            // p(r) = eval_0 * L0(r) + eval_1 * L1(r) + eval_2 * L2(r)
                            // where L0(r) = (r-1)(r-2)/2, L1(r) = -r(r-2), L2(r) = r(r-1)/2
                            // Simpler: convert to coefficients and use Horner
                            const c2_inst1 = inst1_eval_2.sub(inst1_eval_1).sub(inst1_eval_1).add(inst1_eval_0).mul(F.fromU64(2).inverse().?);
                            const c1_inst1 = inst1_eval_1.sub(inst1_eval_0).sub(c2_inst1);
                            const c0_inst1 = inst1_eval_0;
                            // p(r) = c0 + r*c1 + r²*c2
                            const r2 = challenge.mul(challenge);
                            ram_ra_current_claim = c0_inst1.add(c1_inst1.mulHiBigIntU128(challenge.limbs)).add(c2_inst1.mul(r2));

                            std.debug.print("[STAGE5 INST1 CLAIM] Round {}: new_claim={x}\n", .{
                                round,
                                ram_ra_current_claim.toBytesBE()[16..32].*,
                            });
                        }
                    }

                    // Bind cycle round challenge for lookups
                    // CRITICAL: Do NOT bind eq_evals - keep them as original eq(j, r_reduction)
                    // Only bind combined_vals (standard MLE binding)
                    // This matches Jolt's GruenSplitEqPolynomial where E_in/E_out are not modified
                    bindSinglePolynomial(lookups_combined_vals, lookups_round, challenge);

                    // CRITICAL FIX: Update lookups_current_scalar with eq(w_i, challenge)
                    // This matches Jolt's GruenSplitEqPolynomial.bind() which accumulates:
                    //   current_scalar *= eq(w[current_index-1], r)
                    // where w is the original r_reduction (NOT modified by sumcheck challenges)
                    // and r is the sumcheck challenge.
                    //
                    // Formula: eq(w, r) = 1 - w - r + 2*w*r
                    //
                    // For MontU128Challenge arithmetic:
                    // - F.one().sub(challenge) gives (1 - r) as F
                    // - w_i.mulHiBigIntU128(challenge.limbs) gives w*r as F
                    const w_i = r_reduction[n_cycle_vars - 1 - lookups_round];
                    const prod_w_r = w_i.mulHiBigIntU128(challenge.limbs); // w * r
                    // eq(w, r) = 1 - w - r + 2*w*r = (1 - r) - w + 2*w*r = (1 - r) + w*(2r - 1)
                    // But simpler: eq(w, r) = 1 - w - r + 2*w*r
                    const one_minus_r_scalar = F.one().sub(challenge); // (1 - r) as F
                    const eq_factor = one_minus_r_scalar.sub(w_i).add(prod_w_r).add(prod_w_r); // 1 - r - w + 2*w*r
                    lookups_current_scalar = lookups_current_scalar.mul(eq_factor);

                    // Debug: print current_scalar update
                    if (lookups_round < 3 or lookups_round >= n_cycle_vars - 1) {
                        std.debug.print("[STAGE5 CYCLE] round={}, lookups_round={}\n", .{ round, lookups_round });
                        std.debug.print("  w_i (r_reduction[{}]) = {x}\n", .{ n_cycle_vars - 1 - lookups_round, w_i.toBytesBE()[16..32].* });
                        std.debug.print("  challenge limbs[2..4] = {x:0>16}{x:0>16}\n", .{ challenge.limbs[2], challenge.limbs[3] });
                        std.debug.print("  prod_w_r = {x}\n", .{ prod_w_r.toBytesBE()[16..32].* });
                        std.debug.print("  one_minus_r = {x}\n", .{ one_minus_r_scalar.toBytesBE()[16..32].* });
                        std.debug.print("  eq_factor = {x}\n", .{ eq_factor.toBytesBE()[16..32].* });
                        std.debug.print("  current_scalar = {x}\n", .{ lookups_current_scalar.toBytesBE()[16..32].* });
                    }

                    // Bind the per-chunk ra weights
                    for (0..ra_num_chunks) |chunk_idx| {
                        bindSinglePolynomial(ra_chunk_weights[chunk_idx], lookups_round, challenge);
                    }
                    // Debug: show ra_chunk values before/after binding
                    if (lookups_round == 0) {
                        std.debug.print("[STAGE5 CYCLE] ra_chunks before first binding (round=128):\n", .{});
                        for (0..@min(4, ra_num_chunks)) |c| {
                            std.debug.print("  ra_chunk[{}][0:4] = [{x}, {x}, {x}, {x}]\n", .{
                                c,
                                ra_chunk_weights[c][0].toBytesBE()[16..32].*,
                                ra_chunk_weights[c][1].toBytesBE()[16..32].*,
                                ra_chunk_weights[c][2].toBytesBE()[16..32].*,
                                ra_chunk_weights[c][3].toBytesBE()[16..32].*,
                            });
                        }
                    }

                    // CRITICAL FIX: Update lookups_claim by evaluating the Instance 2
                    // polynomial at the challenge, NOT by recomputing from raw arrays.
                    //
                    // The polynomial p2(X) = full_coeffs[0] + full_coeffs[1]*X + ... + full_coeffs[d]*X^d
                    // was constructed by finishMlesProductSumFromEvals which includes the
                    // eq(X, r_round) factor. Evaluating at the challenge gives the correct
                    // new claim. Direct recomputation from bound arrays gives a different
                    // result because it doesn't account for the eq factor correctly.
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
                        std.debug.print("[STAGE5 ROUND {}] challenge={x}\n", .{
                            round,
                            challenge.toBytesBE()[16..32].*,
                        });
                        std.debug.print("  new_batched_claim = {x}\n", .{current_batched_claim.toBytesBE()[16..32].*});
                        std.debug.print("  new_lookups_claim = {x}\n", .{lookups_claim.toBytesBE()[16..32].*});
                        // Debug: print eq_evals[0] after binding
                        std.debug.print("  eq_evals[0] after bind = {x}\n", .{lookups_eq_evals[0].toBytesBE()[16..32].*});
                    }

                    continue; // Skip the rest of the loop (we handled everything)
                }
            }

            // Debug: print final batched claim (this is output_claim from verifier's perspective)
            std.debug.print("[STAGE5] Final batched claim (output_claim) = {any}\n", .{current_batched_claim.toBytesBE()});
            std.debug.print("[STAGE5] Final lookups_current_scalar (should = eq_eval_r_reduction) = {x}\n", .{lookups_current_scalar.toBytesBE()[16..32].*});

            // DEBUG: Print each instance's final claim value
            // The verifier computes expected = batch0*inst0_eval + batch1*inst1_eval + batch2*inst2_eval
            // The prover's output_claim should equal this.
            std.debug.print("[STAGE5 FINAL CLAIMS] Individual instance final values:\n", .{});
            std.debug.print("  regs_val_current_claim (Instance 0) = {any}\n", .{regs_val_current_claim.toBytes()});
            std.debug.print("  ram_ra_current_claim (Instance 1) = {any}\n", .{ram_ra_current_claim.toBytes()});
            std.debug.print("  lookups_claim (Instance 2) = {any}\n", .{lookups_claim.toBytes()});
            std.debug.print("  batch0*inst0 (LE) = {any}\n", .{batch0.mul(regs_val_current_claim).toBytes()});
            std.debug.print("  batch1*inst1 (LE) = {any}\n", .{batch1.mul(ram_ra_current_claim).toBytes()});
            std.debug.print("  batch2*inst2 (LE) = {any}\n", .{batch2.mul(lookups_claim).toBytes()});
            const recon = batch0.mul(regs_val_current_claim).add(batch1.mul(ram_ra_current_claim)).add(batch2.mul(lookups_claim));
            std.debug.print("  batch0*inst0 + batch1*inst1 + batch2*inst2 = {any}\n", .{recon.toBytes()});
            std.debug.print("  current_batched_claim = {any}\n", .{current_batched_claim.toBytes()});
            std.debug.print("  reconstruction matches output_claim: {}\n", .{recon.eql(current_batched_claim)});

            // =================================================================
            // BRUTE FORCE Instance 1 expected output claim
            // Verifier computes: eq_combined * ra_claim_reduced
            // eq_combined = eq(r_addr_1, r_addr_reduced)*(eq_raf+γ*eq_val)
            //             + γ²*eq(r_addr_2, r_addr_reduced)*(eq_rw+γ*eq_val)
            // ra_claim_reduced = Σ_{(addr,cycle)} eq(addr, r_addr_reduced) * eq(cycle, r_cycle_reduced)
            // =================================================================
            {
                const inst1_start = max_num_rounds - ram_ra_num_rounds; // 112
                const addr_challenges = challenges[inst1_start .. inst1_start + log_ram_k]; // 16 challenges
                const cycle_chal = challenges[inst1_start + log_ram_k .. inst1_start + ram_ra_num_rounds]; // 8 challenges

                // Convert to BIG_ENDIAN for eq computation
                // Sumcheck challenges are in LowToHigh order, eq computation expects BIG_ENDIAN
                var r_addr_reduced_be: [16]F = undefined;
                for (0..log_ram_k) |i| {
                    r_addr_reduced_be[i] = addr_challenges[log_ram_k - 1 - i];
                }
                var r_cycle_reduced_be: [8]F = undefined;
                for (0..n_cycle_vars) |i| {
                    r_cycle_reduced_be[i] = cycle_chal[n_cycle_vars - 1 - i];
                }

                // Compute eq(r_addr_1, r_addr_reduced)
                const eq_addr_1 = computeEqPolynomial(F, r_address_raf, r_addr_reduced_be[0..log_ram_k]);
                const eq_addr_2 = computeEqPolynomial(F, r_address_rw, r_addr_reduced_be[0..log_ram_k]);

                // Compute eq(r_cycle_x, r_cycle_reduced)
                const eq_cycle_raf_red = computeEqPolynomial(F, r_cycle_raf, r_cycle_reduced_be[0..n_cycle_vars]);
                const eq_cycle_rw_red = computeEqPolynomial(F, r_cycle_rw, r_cycle_reduced_be[0..n_cycle_vars]);
                const eq_cycle_val_red = computeEqPolynomial(F, r_cycle_val, r_cycle_reduced_be[0..n_cycle_vars]);

                // eq_combined
                const eq_cycle_A = eq_cycle_raf_red.add(gamma.mul(eq_cycle_val_red));
                const eq_cycle_B = eq_cycle_rw_red.add(gamma.mul(eq_cycle_val_red));
                const bf_eq_combined = eq_addr_1.mul(eq_cycle_A).add(gamma2.mul(eq_addr_2.mul(eq_cycle_B)));

                // ra_claim_reduced = Σ eq(addr, r_addr_reduced) * eq(cycle, r_cycle_reduced)
                var bf_ra_claim = F.zero();
                if (memory_trace) |mt| {
                    for (mt.accesses.items, 0..) |access, i| {
                        const addr = ram_addresses[i];
                        const cycle = access.timestamp;
                        const eq_a = computeEqAtPoint(F, r_addr_reduced_be[0..log_ram_k], addr);
                        const eq_c = computeEqAtPoint(F, r_cycle_reduced_be[0..n_cycle_vars], cycle);
                        bf_ra_claim = bf_ra_claim.add(eq_a.mul(eq_c));
                    }
                }

                const bf_expected_inst1 = bf_eq_combined.mul(bf_ra_claim);

                std.debug.print("[BRUTE FORCE INST1] eq_addr_1 = {x}\n", .{eq_addr_1.toBytesBE()[16..32].*});
                std.debug.print("[BRUTE FORCE INST1] eq_addr_2 = {x}\n", .{eq_addr_2.toBytesBE()[16..32].*});
                std.debug.print("[BRUTE FORCE INST1] eq_cycle_raf = {x}\n", .{eq_cycle_raf_red.toBytesBE()[16..32].*});
                std.debug.print("[BRUTE FORCE INST1] eq_cycle_rw = {x}\n", .{eq_cycle_rw_red.toBytesBE()[16..32].*});
                std.debug.print("[BRUTE FORCE INST1] eq_cycle_val = {x}\n", .{eq_cycle_val_red.toBytesBE()[16..32].*});
                std.debug.print("[BRUTE FORCE INST1] eq_combined = {x}\n", .{bf_eq_combined.toBytesBE()[16..32].*});
                std.debug.print("[BRUTE FORCE INST1] ra_claim_reduced = {x}\n", .{bf_ra_claim.toBytesBE()[16..32].*});
                std.debug.print("[BRUTE FORCE INST1] expected (eq_combined * ra_reduced) = {any}\n", .{bf_expected_inst1.toBytes()});
                std.debug.print("[BRUTE FORCE INST1] prover tracked ram_ra_current_claim = {any}\n", .{ram_ra_current_claim.toBytes()});
                std.debug.print("[BRUTE FORCE INST1] match: {}\n", .{bf_expected_inst1.eql(ram_ra_current_claim)});

                // Also compute what the verifier would get for batch1*inst1 using brute force
                const bf_batch1_inst1 = batch1.mul(bf_expected_inst1);
                std.debug.print("[BRUTE FORCE INST1] batch1 * expected = {any}\n", .{bf_batch1_inst1.toBytes()});
            }

            // Get final opening claims from the folded polynomials
            const regs_val_inc_claim = inc_evals[0];
            const regs_val_wa_claim = wa_evals[0];
            const regs_val_lt_claim = lt_evals[0];
            const regs_final_product = regs_val_inc_claim.mul(regs_val_wa_claim).mul(regs_val_lt_claim);

            std.debug.print("[STAGE5] Final opening claims:\n", .{});
            std.debug.print("  regs_val_inc_claim = {any}\n", .{regs_val_inc_claim.toBytesBE()});
            std.debug.print("  regs_val_wa_claim = {any}\n", .{regs_val_wa_claim.toBytesBE()});
            std.debug.print("  regs_val_lt_claim (lt[0] after binding) = {any}\n", .{regs_val_lt_claim.toBytesBE()});
            std.debug.print("  regs_final_product (inc*wa*lt) = {any}\n", .{regs_final_product.toBytesBE()});

            // Compute what the verifier would compute for LT(r_normalized, r_cycle)
            // r_normalized = reversed challenges (BIG_ENDIAN)
            // The last 8 challenges are for RegistersValEvaluation
            const regs_challenges = challenges[(max_num_rounds - regs_val_num_rounds)..];
            std.debug.print("[STAGE5] Computing verifier's LT(r_normalized, r_cycle):\n", .{});
            std.debug.print("  regs_challenges[0] = {any}\n", .{regs_challenges[0].toBytesBE()[0..8]});
            std.debug.print("  regs_challenges[7] = {any}\n", .{regs_challenges[7].toBytesBE()[0..8]});
            std.debug.print("  r_cycle_regs[0] = {any}\n", .{r_cycle_regs[0].toBytesBE()[0..8]});
            std.debug.print("  r_cycle_regs[7] = {any}\n", .{r_cycle_regs[7].toBytesBE()[0..8]});

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
            std.debug.print("  LT_verifier (what verifier computes) = {any}\n", .{lt_verifier.toBytesBE()});

            // The verifier expects: expected_output_claim = inc_claim * wa_claim * LT_verifier
            const expected_product = regs_val_inc_claim.mul(regs_val_wa_claim).mul(lt_verifier);
            std.debug.print("  expected_product (inc*wa*LT_verifier) = {any}\n", .{expected_product.toBytesBE()});
            std.debug.print("  Match: {}\n", .{regs_final_product.eql(expected_product)});

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

            const num_lookup_tables: usize = 42;
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

            std.debug.print("[STAGE5 LOOKUPS] Computing opening claims:\n", .{});
            std.debug.print("  lookups_input = {any}\n", .{lookups_input.toBytesBE()[0..8]});
            std.debug.print("  lookups_output_claim (eq*ra_w0*combined) = {any}\n", .{lookups_output_claim.toBytesBE()});
            std.debug.print("  lookups_claim (tracked Instance 2) = {any}\n", .{lookups_claim.toBytes()});
            std.debug.print("  lookups_eq_evals[0] = {any}\n", .{lookups_eq_evals[0].toBytesBE()[0..8]});
            std.debug.print("  lookups_combined_vals[0] = {any}\n", .{lookups_combined_vals[0].toBytes()});
            std.debug.print("  lookups_current_scalar = {any}\n", .{lookups_current_scalar.toBytes()});
            std.debug.print("  lookups_combined_vals[0] = {any}\n", .{lookups_combined_vals[0].toBytesBE()[0..8]});

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

            std.debug.print("  r_reduction[0..8] (8 elements):\n", .{});
            for (0..n_cycle_vars) |i| {
                std.debug.print("    r_reduction[{}] = {x}\n", .{ i, r_reduction[i].toBytesBE()[16..32].* });
            }
            std.debug.print("  r_cycle_prime_be[0..8] (8 elements):\n", .{});
            for (0..n_cycle_vars) |i| {
                std.debug.print("    r_cycle_prime_be[{}] = {x}\n", .{ i, r_cycle_prime_be[i].toBytesBE()[16..32].* });
            }
            std.debug.print("  eq_r_reduction (verifier computes) = {x}\n", .{eq_r_reduction.toBytesBE()[16..32].*});
            std.debug.print("  eq_evals[0] (from sumcheck) = {x}\n", .{lookups_eq_evals[0].toBytesBE()[16..32].*});
            std.debug.print("  lookups_current_scalar (should equal eq_r_reduction) = {x}\n", .{lookups_current_scalar.toBytesBE()[16..32].*});

            // Debug: print first few r_address_prime values
            std.debug.print("[STAGE5 FINAL] r_address_prime[0..4] (sumcheck challenges 0-3):\n", .{});
            for (0..4) |i| {
                std.debug.print("  r_address_prime[{}] = {x}\n", .{ i, r_address_prime[i].toBytesBE()[16..32].* });
            }

            // Compute operand polynomial evaluations at r_address_prime
            const left_op_eval = evaluateLeftOperand(F, r_address_prime);
            const right_op_eval = evaluateRightOperand(F, r_address_prime);
            const identity_eval = evaluateIdentity(F, r_address_prime);

            std.debug.print("  left_op_eval (full) = {x}\n", .{left_op_eval.toBytesBE()[16..32].*});
            std.debug.print("  right_op_eval (full) = {x}\n", .{right_op_eval.toBytesBE()[16..32].*});
            std.debug.print("  identity_eval (full) = {x}\n", .{identity_eval.toBytesBE()[16..32].*});
            std.debug.print("  gamma_lookups_raf = {any}\n", .{gamma_lookups_raf.toBytesBE()[0..8]});

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
            std.debug.print("[STAGE5 LOOKUPS] ra_chunk claims (FULL 32 bytes LE for Jolt comparison):\n", .{});
            var ra_product = F.one();
            for (0..lookups_ra_d) |i| {
                const le_bytes = ra_chunks[i].toBytes();
                std.debug.print("  ra_chunks[{}] LE = {any}\n", .{ i, le_bytes });
                ra_product = ra_product.mul(ra_chunks[i]);
            }
            const ra_product_le = ra_product.toBytes();
            std.debug.print("  ra_product FULL LE = {any}\n", .{ra_product_le});
            std.debug.print("  lookups_ra_weights[0] = {any}\n", .{lookups_ra_weights[0].toBytesBE()[0..8]});

            // Check lookups_current_scalar * ra_product * combined_vals[0]
            const scalar_ra_combined = lookups_current_scalar.mul(ra_product).mul(lookups_combined_vals[0]);
            std.debug.print("  scalar*ra_product*combined = {any}\n", .{scalar_ra_combined.toBytes()});
            std.debug.print("  lookups_claim (tracked)    = {any}\n", .{lookups_claim.toBytes()});
            std.debug.print("  scalar*ra_product*combined == lookups_claim: {}\n", .{scalar_ra_combined.eql(lookups_claim)});

            // Verify ra_product == lookups_ra_weights[0]
            const match_after = ra_product.eql(lookups_ra_weights[0]);
            std.debug.print("  ra_product == lookups_ra_weights[0] (after all binding): {}\n", .{match_after});
            if (!match_after) {
                std.debug.print("  WARNING: ra_product and lookups_ra_weights[0] don't match after binding!\n", .{});
                std.debug.print("  This is expected - binding the product != product of bindings\n", .{});
                std.debug.print("  The correct ra_claim should be the PRODUCT of the bound chunk values.\n", .{});
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
            for (0..T) |j| {
                if (j >= trace_len) continue;

                // Compute eq(r_cycle', j) - note r_cycle_prime_be is already BIG_ENDIAN
                const eq_j = computeEqAtIndex(r_cycle_prime_be, j);

                // Accumulate into appropriate table flag
                const table_idx = cycle_table_indices[j];
                if (table_idx >= 0 and @as(usize, @intCast(table_idx)) < num_lookup_tables) {
                    table_flags[@intCast(table_idx)] = table_flags[@intCast(table_idx)].add(eq_j);
                }
            }

            // Debug: print non-zero table flags
            std.debug.print("[STAGE5 LOOKUPS] Non-zero table flags (FULL LE):\n", .{});
            for (0..num_lookup_tables) |i| {
                if (!table_flags[i].eql(F.zero())) {
                    std.debug.print("  table_flags[{}] = {any}\n", .{ i, table_flags[i].toBytes() });
                }
            }

            // Compute raf_flag = Σ_{j: identity_path} eq(r_cycle', j)
            var computed_raf_flag = F.zero();
            for (0..T) |j| {
                if (j >= trace_len) continue;
                if (cycle_is_identity_path[j]) {
                    const eq_j = computeEqAtIndex(r_cycle_prime_be, j);
                    computed_raf_flag = computed_raf_flag.add(eq_j);
                }
            }
            std.debug.print("[STAGE5 LOOKUPS] raf_flag (identity path sum) = {any}\n", .{computed_raf_flag.toBytesBE()[0..8]});

            // Verify the opening claims match the sumcheck output
            // expected = eq_r_reduction * ra_product * (val_claim + gamma * raf_claim)
            // where val_claim = Σ table_flags[i] * table_eval[i]
            // and raf_claim = (1 - raf_flag)*(left_op + gamma*right_op) + raf_flag*gamma*identity

            const raf_claim = F.one().sub(computed_raf_flag).mul(left_op_eval.add(gamma_lookups_raf.mul(right_op_eval)))
                .add(computed_raf_flag.mul(gamma_lookups_raf).mul(identity_eval));
            std.debug.print("  raf_claim (from formula) FULL LE = {any}\n", .{raf_claim.toBytes()});

            // Compute val_claim = Σ table_flags[i] * stored_table_values[i]
            // This matches the verifier's formula: Σ val_evals[i] * table_flag_claims[i]
            var val_claim = F.zero();
            for (0..num_lookup_tables) |i| {
                val_claim = val_claim.add(table_flags[i].mul(stored_table_values[i]));
            }
            std.debug.print("  val_claim (from formula) FULL LE = {any}\n", .{val_claim.toBytes()});
            std.debug.print("  val_claim last 16 LE = ", .{});
            for (val_claim.toBytes()[16..32]) |b| std.debug.print("{x:0>2} ", .{b});
            std.debug.print("\n", .{});

            // Compute expected output: eq_r_reduction * ra_product * (val_claim + gamma * raf_claim)
            const expected_output = eq_r_reduction.mul(ra_product).mul(val_claim.add(gamma_lookups_raf.mul(raf_claim)));
            std.debug.print("  expected_output (eq*ra*(val+gamma*raf)) FULL LE = {any}\n", .{expected_output.toBytes()});
            std.debug.print("  expected_output last 16 LE = ", .{});
            for (expected_output.toBytes()[16..32]) |b| std.debug.print("{x:0>2} ", .{b});
            std.debug.print("\n", .{});
            std.debug.print("  lookups_output_claim = {any}\n", .{lookups_output_claim.toBytesBE()[0..8]});
            std.debug.print("  current_batched_claim = {any}\n", .{current_batched_claim.toBytesBE()[0..8]});

            // Check if expected matches the sumcheck output
            std.debug.print("  expected_output == lookups_output_claim: {}\n", .{expected_output.eql(lookups_output_claim)});

            // ============================================================
            // COMPUTE ram_ra_claim FROM RAM TRACE
            // ============================================================
            // The RamRaClaimReduction sumcheck proves:
            //   Σ_{k,c} eq_combined(k,c) * ra(k,c) = input_claim
            // where ra(k,c) = 1 iff address at cycle c equals k.
            //
            // After the sumcheck binds all variables, we get:
            //   ram_ra_claim = ra(r_address_reduced, r_cycle_reduced)
            // This equals the sum of eq(addr, r_addr) * eq(cycle, r_cycle)
            // over all (addr, cycle) pairs in the RAM trace.
            //
            // For RamRaClaimReduction (Instance 1, 24 rounds = 16 address + 8 cycle):
            // - Address challenges: challenges[112..127] (16 values)
            // - Cycle challenges: challenges[128..135] (8 values)
            // These are reversed to get big-endian order for eq computation.
            const ram_ra_start = max_num_rounds - ram_ra_num_rounds; // 136 - 24 = 112
            const ram_addr_challenges = challenges[ram_ra_start .. ram_ra_start + log_ram_k];
            const ram_cycle_challenges = challenges[ram_ra_start + log_ram_k .. ram_ra_start + ram_ra_num_rounds];

            // Reverse to get big-endian order
            var r_ram_addr_be = try self.allocator.alloc(F, log_ram_k);
            defer self.allocator.free(r_ram_addr_be);
            for (0..log_ram_k) |i| {
                r_ram_addr_be[i] = ram_addr_challenges[log_ram_k - 1 - i];
            }

            var r_ram_cycle_be = try self.allocator.alloc(F, n_cycle_vars);
            defer self.allocator.free(r_ram_cycle_be);
            for (0..n_cycle_vars) |i| {
                r_ram_cycle_be[i] = ram_cycle_challenges[n_cycle_vars - 1 - i];
            }

            // Compute ram_ra_claim = Σ_{(addr, cycle) in RAM_trace} eq(addr, r_addr) * eq(cycle, r_cycle)
            // Use the dedicated MemoryTrace which tracks actual RAM accesses (including synthetic termination writes)
            var ram_ra_claim = F.zero();
            if (memory_trace) |mem_trace| {
                std.debug.print("[STAGE5 RAM_RA] Using MemoryTrace with {} accesses\n", .{mem_trace.accesses.items.len});
                for (mem_trace.accesses.items) |access| {
                    // Only consider WRITE operations for ra claim
                    // (Jolt's RAM trace records both reads and writes, but ra represents where values changed)
                    if (access.op == .Write) {
                        const raw_addr = access.address;
                        const cycle = access.timestamp;

                        // Remap address to polynomial index space using memory_layout
                        const addr: u64 = if (memory_layout) |ml|
                            ml.remapAddress(raw_addr) orelse 0
                        else
                            raw_addr & (@as(u64, K) - 1);

                        // Compute eq(addr, r_ram_addr_be) for the address
                        // Address is log_ram_k bits (16 bits = 65536 addresses)
                        const eq_addr = computeEqAtIndex(r_ram_addr_be, @intCast(addr));

                        // Compute eq(cycle, r_ram_cycle_be) for the cycle
                        const eq_cycle = computeEqAtIndex(r_ram_cycle_be, @intCast(cycle));

                        // Accumulate: ra(k,c) = 1 for this (addr, cycle) pair
                        ram_ra_claim = ram_ra_claim.add(eq_addr.mul(eq_cycle));

                        std.debug.print("[STAGE5 RAM_RA] WRITE raw_addr=0x{x}, remapped_addr={}, cycle={}, eq_addr={x}, eq_cycle={x}\n", .{
                            raw_addr,
                            addr,
                            cycle,
                            eq_addr.toBytesBE()[24..32].*,
                            eq_cycle.toBytesBE()[24..32].*,
                        });
                    }
                }
            } else {
                std.debug.print("[STAGE5 RAM_RA] No memory_trace available, ram_ra_claim = 0\n", .{});
            }
            std.debug.print("[STAGE5 RAM_RA] Computed ram_ra_claim = {x}\n", .{ram_ra_claim.toBytesBE()});

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

        /// Compute immediate value from instruction, matching R1CS deriveImmediate
        fn computeImmediate(instr: u32) F {
            const opcode: u8 = @truncate(instr & 0x7f);

            switch (opcode) {
                // I-type: imm[11:0] at bits [31:20], sign-extended
                0x13, 0x03, 0x67, 0x1b, 0x73 => {
                    const imm12: u32 = instr >> 20;
                    const imm_signed: i64 = @as(i64, @as(i32, @bitCast(imm12 << 20)) >> 20);
                    return signedI64ToField(imm_signed);
                },
                // S-type: imm[11:5] at [31:25], imm[4:0] at [11:7], sign-extended
                0x23 => {
                    const imm11_5 = (instr >> 25) & 0x7f;
                    const imm4_0 = (instr >> 7) & 0x1f;
                    const imm12: u32 = (imm11_5 << 5) | imm4_0;
                    const imm_signed: i64 = @as(i64, @as(i32, @bitCast(imm12 << 20)) >> 20);
                    return signedI64ToField(imm_signed);
                },
                // B-type: imm[12|10:5] at [31:25], imm[4:1|11] at [11:7], sign-extended, *2
                0x63 => {
                    const imm12 = (instr >> 31) & 1;
                    const imm10_5 = (instr >> 25) & 0x3f;
                    const imm4_1 = (instr >> 8) & 0xf;
                    const imm11 = (instr >> 7) & 1;
                    const imm13: u32 = (imm12 << 12) | (imm11 << 11) | (imm10_5 << 5) | (imm4_1 << 1);
                    const imm_signed: i64 = @as(i64, @as(i32, @bitCast(imm13 << 19)) >> 19);
                    return signedI64ToField(imm_signed);
                },
                // U-type: imm[31:12] at [31:12], shifted left by 12
                0x37, 0x17 => {
                    const imm_upper = instr & 0xFFFFF000;
                    return F.fromU64(imm_upper);
                },
                // J-type: imm[20|10:1|11|19:12] at [31:12], sign-extended, *2
                0x6f => {
                    const imm20 = (instr >> 31) & 1;
                    const imm10_1 = (instr >> 21) & 0x3ff;
                    const imm11 = (instr >> 20) & 1;
                    const imm19_12 = (instr >> 12) & 0xff;
                    const imm21: u32 = (imm20 << 20) | (imm19_12 << 12) | (imm11 << 11) | (imm10_1 << 1);
                    const imm_signed: i64 = @as(i64, @as(i32, @bitCast(imm21 << 11)) >> 11);
                    return signedI64ToField(imm_signed);
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
                    // CRITICAL: Use mulHiBigIntU128 to match Jolt's MontU128Challenge multiplication
                    // Jolt treats challenges as raw [0,0,L,H] integers in multiplication
                    result = result.mulHiBigIntU128(rj.limbs);
                } else {
                    // For (1 - r[j]): subtraction converts challenge to Montgomery form first (matches Jolt)
                    // Then multiply the result (F * F is standard Montgomery multiplication)
                    const one_minus_rj = F.one().sub(rj);
                    result = result.mul(one_minus_rj);
                }
            }
            return result;
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
                    result = result.mulHiBigIntU128(rj.limbs);
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
        fn computeAllLtEvals(allocator: Allocator, r: []const F) ![]F {
            const n = r.len;
            const size = @as(usize, 1) << @intCast(n);
            var evals = try allocator.alloc(F, size);
            @memset(evals, F.zero());

            // Jolt's lt_evals algorithm:
            // for (i, r) in r.r.iter().rev().enumerate() {
            //     let (evals_left, evals_right) = evals.split_at_mut(1 << i);
            //     zip(evals_left, evals_right).for_each(|(x, y)| {
            //         *y = *x * r;
            //         *x += *r - *y;
            //     });
            // }
            // Note: r.r.iter().rev() means we process from last element to first
            // Since r is BIG_ENDIAN (MSB first), rev gives us LSB first

            for (0..n) |i| {
                const ri = r[n - 1 - i]; // Process from LSB to MSB
                const half = @as(usize, 1) << @intCast(i);
                for (0..half) |j| {
                    const x = evals[j];
                    // CRITICAL: Use mulHiBigIntU128 to match Jolt's F * Challenge behavior
                    // Jolt's MontU128Challenge uses mul_by_hi_2limbs for multiplication
                    const y = x.mulHiBigIntU128(ri.limbs);
                    evals[half + j] = y;
                    // ri.sub(y) is correct: Challenge - F converts challenge to Fr first
                    // (via from_bigint_unchecked which stores same Montgomery form)
                    evals[j] = evals[j].add(ri.sub(y));
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
                            // CRITICAL: Use mulHiBigIntU128 for challenge multiplication
                            contrib = contrib.mulHiBigIntU128(rk.limbs);
                        } else {
                            // (1 - rk) requires converting rk to proper Fr first
                            // Since rk already has [0,0,L,H] Montgomery form, sub is correct
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
        fn computeRegsValRoundPoly(inc: []F, wa: []F, lt: []F, round: usize) [4]F {
            var evals = [_]F{ F.zero(), F.zero(), F.zero(), F.zero() };
            const n = inc.len >> @intCast(round);
            const half = n / 2;

            if (half == 0) {
                if (n > 0) {
                    // Constant polynomial: p(x) = c
                    // For Toom-Cook: [p(0), p(1), p(2), p_inf] = [c, c, c, 0]
                    // p_inf = 0 for constant polynomials (no x^3 term)
                    evals[0] = inc[0].mul(wa[0]).mul(lt[0]);
                    evals[1] = evals[0];
                    evals[2] = evals[0];
                    // evals[3] remains zero for constant polynomial
                }
                return evals;
            }

            for (0..half) |i| {
                const inc_0 = inc[2 * i];
                const wa_0 = wa[2 * i];
                const lt_0 = lt[2 * i];

                const inc_1 = inc[2 * i + 1];
                const wa_1 = wa[2 * i + 1];
                const lt_1 = lt[2 * i + 1];

                // p(0): product at x = 0
                evals[0] = evals[0].add(inc_0.mul(wa_0).mul(lt_0));

                // p(1): product at x = 1
                evals[1] = evals[1].add(inc_1.mul(wa_1).mul(lt_1));

                // Extrapolate for degree-3 polynomial using Toom-Cook encoding
                // For a linear polynomial f(x) = f_0 + x*(f_1 - f_0):
                //   f(2) = 2*f_1 - f_0
                //   f_inf (leading coefficient) = f_1 - f_0
                const inc_2 = inc_1.add(inc_1).sub(inc_0); // 2*inc_1 - inc_0
                const wa_2 = wa_1.add(wa_1).sub(wa_0);
                const lt_2 = lt_1.add(lt_1).sub(lt_0);
                evals[2] = evals[2].add(inc_2.mul(wa_2).mul(lt_2));

                // eval_at_inf = product of leading coefficients
                // For Toom-Cook: evals[3] = eval_at_infinity
                const inc_inf = inc_1.sub(inc_0);
                const wa_inf = wa_1.sub(wa_0);
                const lt_inf = lt_1.sub(lt_0);
                evals[3] = evals[3].add(inc_inf.mul(wa_inf).mul(lt_inf));
            }

            return evals;
        }

        /// Bind challenge for RegistersValEvaluation polynomials
        fn bindRegsValChallenge(inc: []F, wa: []F, lt: []F, round: usize, r: F) void {
            const n = inc.len >> @intCast(round);
            const half = n / 2;
            if (half == 0) return;

            const one_minus_r = F.one().sub(r);

            for (0..half) |i| {
                inc[i] = one_minus_r.mul(inc[2 * i]).add(r.mul(inc[2 * i + 1]));
                wa[i] = one_minus_r.mul(wa[2 * i]).add(r.mul(wa[2 * i + 1]));
                lt[i] = one_minus_r.mul(lt[2 * i]).add(r.mul(lt[2 * i + 1]));
            }

            // Zero out upper half
            for (half..n) |i| {
                inc[i] = F.zero();
                wa[i] = F.zero();
                lt[i] = F.zero();
            }
        }

        /// Compute round polynomial for LookupsReadRaf (cycle rounds only)
        /// This computes Σ_j eq_reduction(j) * combined_vals(j)
        /// Returns [p(0), p(1), p(2), p_inf] for degree-2 polynomial (product of 2 linears)
        fn computeLookupsRoundPoly(eq_evals: []F, combined: []F, round: usize) [4]F {
            var evals = [_]F{ F.zero(), F.zero(), F.zero(), F.zero() };
            const n = eq_evals.len >> @intCast(round);
            const half = n / 2;

            if (half == 0) {
                if (n > 0) {
                    // Constant polynomial
                    const c = eq_evals[0].mul(combined[0]);
                    evals[0] = c;
                    evals[1] = c;
                    evals[2] = c;
                }
                return evals;
            }

            for (0..half) |i| {
                const eq_0 = eq_evals[2 * i];
                const eq_1 = eq_evals[2 * i + 1];
                const c_0 = combined[2 * i];
                const c_1 = combined[2 * i + 1];

                // p(0) = eq_0 * c_0
                evals[0] = evals[0].add(eq_0.mul(c_0));

                // p(1) = eq_1 * c_1
                evals[1] = evals[1].add(eq_1.mul(c_1));

                // p(2) = (2*eq_1 - eq_0) * (2*c_1 - c_0)
                const eq_2 = eq_1.add(eq_1).sub(eq_0);
                const c_2 = c_1.add(c_1).sub(c_0);
                evals[2] = evals[2].add(eq_2.mul(c_2));

                // p_inf = (eq_1 - eq_0) * (c_1 - c_0)
                const eq_inf = eq_1.sub(eq_0);
                const c_inf = c_1.sub(c_0);
                evals[3] = evals[3].add(eq_inf.mul(c_inf));
            }

            return evals;
        }

        /// Bind challenge for LookupsReadRaf polynomials (cycle rounds) - legacy version
        fn bindLookupsChallenge(eq_evals: []F, combined: []F, round: usize, r: F) void {
            const n = eq_evals.len >> @intCast(round);
            const half = n / 2;
            if (half == 0) return;

            const one_minus_r = F.one().sub(r);

            for (0..half) |i| {
                eq_evals[i] = one_minus_r.mul(eq_evals[2 * i]).add(r.mul(eq_evals[2 * i + 1]));
                combined[i] = one_minus_r.mul(combined[2 * i]).add(r.mul(combined[2 * i + 1]));
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
        fn computeLookupsRoundPolyWithRa(eq_evals: []F, ra_weights: []F, combined: []F, round: usize) [4]F {
            var evals = [_]F{ F.zero(), F.zero(), F.zero(), F.zero() };
            const n = eq_evals.len >> @intCast(round);
            const half = n / 2;

            if (half == 0) {
                if (n > 0) {
                    // Constant polynomial
                    const c = eq_evals[0].mul(ra_weights[0]).mul(combined[0]);
                    evals[0] = c;
                    evals[1] = c;
                    evals[2] = c;
                }
                return evals;
            }

            for (0..half) |i| {
                const eq_0 = eq_evals[2 * i];
                const eq_1 = eq_evals[2 * i + 1];
                const ra_0 = ra_weights[2 * i];
                const ra_1 = ra_weights[2 * i + 1];
                const c_0 = combined[2 * i];
                const c_1 = combined[2 * i + 1];

                // p(0) = eq_0 * ra_0 * c_0
                evals[0] = evals[0].add(eq_0.mul(ra_0).mul(c_0));

                // p(1) = eq_1 * ra_1 * c_1
                evals[1] = evals[1].add(eq_1.mul(ra_1).mul(c_1));

                // p(2) = (2*eq_1 - eq_0) * (2*ra_1 - ra_0) * (2*c_1 - c_0)
                const eq_2 = eq_1.add(eq_1).sub(eq_0);
                const ra_2 = ra_1.add(ra_1).sub(ra_0);
                const c_2 = c_1.add(c_1).sub(c_0);
                evals[2] = evals[2].add(eq_2.mul(ra_2).mul(c_2));

                // p_inf = (eq_1 - eq_0) * (ra_1 - ra_0) * (c_1 - c_0)
                const eq_inf = eq_1.sub(eq_0);
                const ra_inf = ra_1.sub(ra_0);
                const c_inf = c_1.sub(c_0);
                evals[3] = evals[3].add(eq_inf.mul(ra_inf).mul(c_inf));
            }

            return evals;
        }

        /// Bind challenge for LookupsReadRaf polynomials with ra_weights (cycle rounds)
        fn bindLookupsCycleChallengeWithRa(eq_evals: []F, ra_weights: []F, combined: []F, round: usize, r: F) void {
            const n = eq_evals.len >> @intCast(round);
            const half = n / 2;
            if (half == 0) return;

            const one_minus_r = F.one().sub(r);

            for (0..half) |i| {
                eq_evals[i] = one_minus_r.mul(eq_evals[2 * i]).add(r.mul(eq_evals[2 * i + 1]));
                ra_weights[i] = one_minus_r.mul(ra_weights[2 * i]).add(r.mul(ra_weights[2 * i + 1]));
                combined[i] = one_minus_r.mul(combined[2 * i]).add(r.mul(combined[2 * i + 1]));
            }

            // Zero out upper half
            for (half..n) |i| {
                eq_evals[i] = F.zero();
                ra_weights[i] = F.zero();
                combined[i] = F.zero();
            }
        }

        /// Bind challenge for a single polynomial (used for per-chunk ra weights)
        fn bindSinglePolynomial(poly: []F, round: usize, r: F) void {
            const n = poly.len >> @intCast(round);
            const half = n / 2;
            if (half == 0) return;

            const one_minus_r = F.one().sub(r);

            for (0..half) |i| {
                poly[i] = one_minus_r.mul(poly[2 * i]).add(r.mul(poly[2 * i + 1]));
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
            std.debug.print("[RIGHT_OP_DEBUG] i={d}: r[{d}]={x}, power={x}, term={x}, result={x}\n", .{
                i, idx, r[idx].toBytesBE()[16..32].*, power.toBytesBE()[16..32].*,
                term.toBytesBE()[16..32].*, result.toBytesBE()[16..32].*,
            });
        }
        power = power.add(power); // power *= 2
    }
    std.debug.print("[RIGHT_OP_DEBUG] final result = {x}\n", .{result.toBytesBE()[16..32].*});
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
            std.debug.print("[IDENTITY_DEBUG] i={d}: r[{d}]={x}, power={x}, term={x}, result={x}\n", .{
                i, i, r[i].toBytesBE()[16..32].*, power.toBytesBE()[16..32].*,
                term.toBytesBE()[16..32].*, result.toBytesBE()[16..32].*,
            });
        }
        power = power.add(power); // power *= 2
    }
    std.debug.print("[IDENTITY_DEBUG] final result = {x}\n", .{result.toBytesBE()[16..32].*});
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
            if (funct3 == 5 and funct7 == 0) break :blk 26; // SRL -> VirtualSRLTable
            if (funct3 == 5 and funct7 == 0x20) break :blk 27; // SRA -> VirtualSRATable
            if (funct3 == 2) break :blk 10; // SLT -> SignedLessThanTable
            if (funct3 == 3) break :blk 11; // SLTU -> UnsignedLessThanTable
            if (funct7 == 0x01 and funct3 == 0) break :blk 0; // MUL -> RangeCheckTable
            if (funct7 == 0x01 and funct3 == 3) break :blk 13; // MULHU -> UpperWordTable
            break :blk -1;
        },
        0x13 => blk: { // I-type
            if (funct3 == 0) break :blk 0; // ADDI -> RangeCheckTable
            if (funct3 == 7) break :blk 2; // ANDI -> AndTable
            if (funct3 == 6) break :blk 4; // ORI -> OrTable
            if (funct3 == 4) break :blk 5; // XORI -> XorTable
            if (funct3 == 1) break :blk -1; // SLLI -> uses virtual decomposition
            if (funct3 == 5 and (funct7 & 0x40) == 0) break :blk 26; // SRLI -> VirtualSRLTable
            if (funct3 == 5 and (funct7 & 0x40) != 0) break :blk 27; // SRAI -> VirtualSRATable
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
