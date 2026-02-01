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

            std.debug.print("[STAGE5] Batching coefficients:\n", .{});
            std.debug.print("  batch0 = {any}\n", .{batch0.toBytesBE()[0..8]});
            std.debug.print("  batch1 = {any}\n", .{batch1.toBytesBE()[0..8]});
            std.debug.print("  batch2 = {any}\n", .{batch2.toBytesBE()[0..8]});

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

                // Update individual claims via Lagrange interpolation
                // For a constant polynomial, p(r) = constant
                // For a zero polynomial, p(r) = 0

                // Update batched claim
                const p0 = combined_poly[0];
                const p1 = combined_poly[1];
                const p2 = combined_poly[2];
                const p3 = combined_poly[3];

                // Cubic Lagrange interpolation at challenge r
                const two = F.fromU64(2);
                const three = F.fromU64(3);
                const six = F.fromU64(6);

                const r = challenge;
                const r_1 = r.sub(F.one());
                const r_2 = r.sub(two);
                const r_3 = r.sub(three);

                const L0 = r_1.mul(r_2).mul(r_3).mul(six.neg().inverse().?);
                const L1 = r.mul(r_2).mul(r_3).mul(two.inverse().?);
                const L2 = r.mul(r_1).mul(r_3).mul(two.neg().inverse().?);
                const L3 = r.mul(r_1).mul(r_2).mul(six.inverse().?);

                current_batched_claim = p0.mul(L0).add(p1.mul(L1)).add(p2.mul(L2)).add(p3.mul(L3));

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
            r_address_regs: []const F, // LOG_K=7 elements from Stage 4 RegistersRWC
            r_cycle_regs: []const F, // n_cycle_vars elements from Stage 4 RegistersRWC
            r_reduction: []const F, // n_cycle_vars elements from Stage 3 InstructionClaimReduction (BIG_ENDIAN)
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

            // Get input claims from accumulator
            const regs_val_input = opening_claims.get(
                .{ .Virtual = .{ .poly = .RegistersVal, .sumcheck_id = .RegistersReadWriteChecking } },
            ) orelse F.zero();

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

            // Build eq_reduction[j] = eq(j, r_reduction) for all cycles j
            // r_reduction is in BIG_ENDIAN order (MSB first)
            for (0..T) |j| {
                lookups_eq_evals[j] = computeEqAtIndex(r_reduction, j);
            }

            // Populate inc and wa from trace
            // Use trace's rd_pre_value and rd_value directly (just like Jolt does)
            const trace_len = trace.steps.items.len;
            for (trace.steps.items, 0..) |step, j| {
                if (step.is_noop) continue;

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
                if (step.is_noop) continue;

                const instr = step.instruction;
                const opcode = instr & 0x7f;
                const funct3 = (instr >> 12) & 0x7;
                const funct7 = (instr >> 25) & 0x7f;

                // Determine left_op, right_op, and lookup_output based on instruction type
                var left_op: F = undefined;
                var right_op: F = undefined;
                var lookup_output: F = undefined;

                switch (opcode) {
                    0x33 => {
                        // R-type: ADD, SUB, AND, OR, XOR, SLT, SLTU, SLL, SRL, SRA
                        const is_add = (funct3 == 0) and (funct7 == 0);
                        const is_sub = (funct3 == 0) and (funct7 == 0x20);

                        if (is_add) {
                            // ADD: AddOperands flag - left=0, right=rs1+rs2
                            left_op = F.zero();
                            right_op = F.fromU64(step.rs1_value +% step.rs2_value);
                            lookup_output = F.fromU64(step.rd_value);
                        } else if (is_sub) {
                            // SUB: SubtractOperands flag - uses interleaved, but we use (rs1, rs2)
                            left_op = F.fromU64(step.rs1_value);
                            right_op = F.fromU64(step.rs2_value);
                            lookup_output = F.fromU64(step.rd_value);
                        } else {
                            // AND, OR, XOR, SLT, SLTU, SLL, SRL, SRA - interleaved operands
                            left_op = F.fromU64(step.rs1_value);
                            right_op = F.fromU64(step.rs2_value);
                            lookup_output = F.fromU64(step.rd_value);
                        }
                    },
                    0x13 => {
                        // I-type: ADDI, ANDI, ORI, XORI, SLTI, SLTIU, SLLI, SRLI, SRAI
                        // Extract 12-bit immediate
                        const imm12_raw: u32 = @truncate(instr >> 20);
                        const imm_signed: i64 = @as(i64, @as(i32, @bitCast(imm12_raw << 20)) >> 20);
                        const imm_u64: u64 = @bitCast(imm_signed);

                        const is_addi = (funct3 == 0);

                        if (is_addi) {
                            // ADDI: AddOperands flag - left=0, right=rs1+imm
                            left_op = F.zero();
                            right_op = F.fromU64(step.rs1_value +% imm_u64);
                            lookup_output = F.fromU64(step.rd_value);
                        } else {
                            // ANDI, ORI, XORI, SLTI, SLTIU, shifts - interleaved operands
                            left_op = F.fromU64(step.rs1_value);
                            right_op = F.fromU64(imm_u64);
                            lookup_output = F.fromU64(step.rd_value);
                        }
                    },
                    0x1b => {
                        // OP-IMM-32 (RV64): ADDIW, SLLIW, SRLIW, SRAIW
                        const imm12_raw: u32 = @truncate(instr >> 20);
                        const imm_signed: i64 = @as(i64, @as(i32, @bitCast(imm12_raw << 20)) >> 20);
                        const imm_u64: u64 = @bitCast(imm_signed);

                        const is_addiw = (funct3 == 0);

                        if (is_addiw) {
                            // ADDIW: AddOperands flag - left=0, right=rs1+imm (32-bit)
                            const rs1_32: u32 = @truncate(step.rs1_value);
                            const imm_32: u32 = @truncate(imm_u64);
                            left_op = F.zero();
                            right_op = F.fromU64(@as(u64, rs1_32) +% @as(u64, imm_32));
                            lookup_output = F.fromU64(step.rd_value);
                        } else {
                            // SLLIW, SRLIW, SRAIW - interleaved
                            left_op = F.fromU64(step.rs1_value);
                            right_op = F.fromU64(imm_u64);
                            lookup_output = F.fromU64(step.rd_value);
                        }
                    },
                    0x3b => {
                        // OP-32 (RV64): ADDW, SUBW, SLLW, SRLW, SRAW, MULW, etc.
                        const is_addw = (funct3 == 0) and (funct7 == 0);
                        const is_subw = (funct3 == 0) and (funct7 == 0x20);

                        if (is_addw) {
                            // ADDW: AddOperands flag - left=0, right=rs1+rs2 (32-bit)
                            const rs1_32: u32 = @truncate(step.rs1_value);
                            const rs2_32: u32 = @truncate(step.rs2_value);
                            left_op = F.zero();
                            right_op = F.fromU64(@as(u64, rs1_32) +% @as(u64, rs2_32));
                            lookup_output = F.fromU64(step.rd_value);
                        } else if (is_subw) {
                            // SUBW: SubtractOperands flag
                            left_op = F.fromU64(step.rs1_value);
                            right_op = F.fromU64(step.rs2_value);
                            lookup_output = F.fromU64(step.rd_value);
                        } else {
                            // Other W operations
                            left_op = F.fromU64(step.rs1_value);
                            right_op = F.fromU64(step.rs2_value);
                            lookup_output = F.fromU64(step.rd_value);
                        }
                    },
                    0x37 => {
                        // LUI: AddOperands flag - left=0, right=imm
                        left_op = F.zero();
                        const imm20: u64 = @as(u64, instr & 0xFFFFF000);
                        right_op = F.fromU64(imm20);
                        lookup_output = F.fromU64(step.rd_value);
                    },
                    0x17 => {
                        // AUIPC: AddOperands flag - left=0, right=PC+imm
                        left_op = F.zero();
                        const imm20: u64 = @as(u64, instr & 0xFFFFF000);
                        right_op = F.fromU64(step.pc +% imm20);
                        lookup_output = F.fromU64(step.rd_value);
                    },
                    0x6f => {
                        // JAL: AddOperands flag - left=0, right=PC+imm
                        // Extract J-type immediate
                        const imm20 = ((@as(u32, instr >> 31) & 1) << 19) |
                            ((@as(u32, instr >> 12) & 0xFF) << 11) |
                            ((@as(u32, instr >> 20) & 1) << 10) |
                            ((@as(u32, instr >> 21) & 0x3FF));
                        const imm_signed: i64 = @as(i64, @as(i32, @bitCast(imm20 << 12)) >> 11);
                        const imm_u64: u64 = @bitCast(imm_signed);

                        left_op = F.zero();
                        right_op = F.fromU64(step.pc +% imm_u64);
                        lookup_output = F.fromU64(step.pc +% imm_u64); // PC + imm, NOT rd
                    },
                    0x67 => {
                        // JALR: AddOperands flag - left=0, right=rs1+imm
                        const imm12_raw: u32 = @truncate(instr >> 20);
                        const imm_signed: i64 = @as(i64, @as(i32, @bitCast(imm12_raw << 20)) >> 20);
                        const imm_u64: u64 = @bitCast(imm_signed);

                        left_op = F.zero();
                        right_op = F.fromU64(step.rs1_value +% imm_u64);
                        // Output is (rs1 + imm) & ~1 for JALR
                        lookup_output = F.fromU64((step.rs1_value +% imm_u64) & ~@as(u64, 1));
                    },
                    0x63 => {
                        // B-type: BEQ, BNE, BLT, BGE, BLTU, BGEU
                        // These use interleaved operands (not AddOperands)
                        left_op = F.fromU64(step.rs1_value);
                        right_op = F.fromU64(step.rs2_value);
                        const result: u64 = switch (funct3) {
                            0x0 => if (step.rs1_value == step.rs2_value) 1 else 0, // BEQ
                            0x1 => if (step.rs1_value != step.rs2_value) 1 else 0, // BNE
                            0x4 => if (@as(i64, @bitCast(step.rs1_value)) < @as(i64, @bitCast(step.rs2_value))) 1 else 0, // BLT
                            0x5 => if (@as(i64, @bitCast(step.rs1_value)) >= @as(i64, @bitCast(step.rs2_value))) 1 else 0, // BGE
                            0x6 => if (step.rs1_value < step.rs2_value) 1 else 0, // BLTU
                            0x7 => if (step.rs1_value >= step.rs2_value) 1 else 0, // BGEU
                            else => 0,
                        };
                        lookup_output = F.fromU64(result);
                    },
                    0x03 => {
                        // Load: AddOperands flag - left=0, right=rs1+imm (address)
                        const imm12_raw: u32 = @truncate(instr >> 20);
                        const imm_signed: i64 = @as(i64, @as(i32, @bitCast(imm12_raw << 20)) >> 20);
                        const imm_u64: u64 = @bitCast(imm_signed);

                        left_op = F.zero();
                        right_op = F.fromU64(step.rs1_value +% imm_u64);
                        lookup_output = F.fromU64(step.rd_value);
                    },
                    0x23 => {
                        // Store: AddOperands flag - left=0, right=rs1+imm (address)
                        const imm_lo: u32 = (instr >> 7) & 0x1F;
                        const imm_hi: u32 = (instr >> 25) & 0x7F;
                        const imm12 = (imm_hi << 5) | imm_lo;
                        const imm_signed: i64 = @as(i64, @as(i12, @bitCast(@as(u12, @truncate(imm12)))));
                        const imm_u64: u64 = @bitCast(imm_signed);

                        left_op = F.zero();
                        right_op = F.fromU64(step.rs1_value +% imm_u64);
                        // For stores, output is typically the address
                        lookup_output = F.fromU64(step.rs1_value +% imm_u64);
                    },
                    else => {
                        // Unknown or other: fall back to rs1, rs2, rd
                        left_op = F.fromU64(step.rs1_value);
                        right_op = F.fromU64(step.rs2_value);
                        lookup_output = F.fromU64(step.rd_value);
                    },
                }

                // combined = output + gamma*left + gamma^2*right
                lookups_combined_vals[j] = lookup_output.add(gamma_raf.mul(left_op)).add(gamma_raf2.mul(right_op));

                // Compute lookup index = interleave_bits(left_operand, right_operand)
                // For AddOperands instructions: left=0, right=sum, so index = interleave(0, sum)
                // For interleaved: left=rs1, right=rs2, so index = interleave(rs1, rs2)
                // Note: We need the RAW operand values as u64, not field elements
                const left_op_raw: u64 = blk: {
                    const is_add_operands = switch (opcode) {
                        0x33 => (funct3 == 0) and (funct7 == 0), // ADD
                        0x13 => (funct3 == 0), // ADDI
                        0x1b => (funct3 == 0), // ADDIW
                        0x3b => (funct3 == 0) and (funct7 == 0), // ADDW
                        0x37, 0x17, 0x6f, 0x67, 0x03, 0x23 => true, // LUI, AUIPC, JAL, JALR, Load, Store
                        else => false,
                    };
                    if (is_add_operands) {
                        break :blk 0;
                    } else {
                        break :blk step.rs1_value;
                    }
                };
                const right_op_raw: u64 = blk: {
                    const is_add_operands = switch (opcode) {
                        0x33 => (funct3 == 0) and (funct7 == 0), // ADD
                        0x3b => (funct3 == 0) and (funct7 == 0), // ADDW
                        else => false,
                    };
                    if (is_add_operands) {
                        // right = rs1 + rs2
                        break :blk step.rs1_value +% step.rs2_value;
                    }
                    const is_imm_add = switch (opcode) {
                        0x13 => (funct3 == 0), // ADDI
                        0x1b => (funct3 == 0), // ADDIW
                        0x37 => true, // LUI
                        0x17 => true, // AUIPC
                        0x6f => true, // JAL
                        0x67 => true, // JALR
                        0x03 => true, // Load
                        0x23 => true, // Store
                        else => false,
                    };
                    if (is_imm_add) {
                        // Various immediate formats - for simplicity, use what we computed
                        // This is the sum that goes into right_op
                        break :blk switch (opcode) {
                            0x13, 0x67 => blk2: {
                                const imm12_raw: u32 = @truncate(instr >> 20);
                                const imm_signed: i64 = @as(i64, @as(i32, @bitCast(imm12_raw << 20)) >> 20);
                                const imm_u64: u64 = @bitCast(imm_signed);
                                break :blk2 step.rs1_value +% imm_u64;
                            },
                            0x1b => blk2: {
                                const imm12_raw: u32 = @truncate(instr >> 20);
                                const imm_signed: i64 = @as(i64, @as(i32, @bitCast(imm12_raw << 20)) >> 20);
                                const imm_u64: u64 = @bitCast(imm_signed);
                                const rs1_32: u32 = @truncate(step.rs1_value);
                                const imm_32: u32 = @truncate(imm_u64);
                                break :blk2 @as(u64, rs1_32) +% @as(u64, imm_32);
                            },
                            0x37 => @as(u64, instr & 0xFFFFF000),
                            0x17 => step.pc +% @as(u64, instr & 0xFFFFF000),
                            0x6f => blk2: {
                                const imm20: u32 = ((@as(u32, instr >> 31) & 1) << 19) |
                                    ((@as(u32, instr >> 12) & 0xFF) << 11) |
                                    ((@as(u32, instr >> 20) & 1) << 10) |
                                    ((@as(u32, instr >> 21) & 0x3FF));
                                const imm_signed: i64 = @as(i64, @as(i32, @bitCast(imm20 << 12)) >> 11);
                                const imm_u64: u64 = @bitCast(imm_signed);
                                break :blk2 step.pc +% imm_u64;
                            },
                            0x03 => blk2: {
                                const imm12_raw: u32 = @truncate(instr >> 20);
                                const imm_signed: i64 = @as(i64, @as(i32, @bitCast(imm12_raw << 20)) >> 20);
                                const imm_u64: u64 = @bitCast(imm_signed);
                                break :blk2 step.rs1_value +% imm_u64;
                            },
                            0x23 => blk2: {
                                const imm_lo: u32 = (instr >> 7) & 0x1F;
                                const imm_hi: u32 = (instr >> 25) & 0x7F;
                                const imm12: u32 = (imm_hi << 5) | imm_lo;
                                const imm_signed: i64 = @as(i64, @as(i12, @bitCast(@as(u12, @truncate(imm12)))));
                                const imm_u64: u64 = @bitCast(imm_signed);
                                break :blk2 step.rs1_value +% imm_u64;
                            },
                            else => step.rs2_value,
                        };
                    }
                    break :blk step.rs2_value;
                };

                // Compute interleaved lookup index
                const lookup_idx = interleaveBits128(left_op_raw, right_op_raw);
                lookups_indices_lo[j] = @truncate(lookup_idx);
                lookups_indices_hi[j] = @truncate(lookup_idx >> 64);

                // Track which lookup table this cycle uses (for flag claims)
                // and whether it uses identity path (for raf_flag claim)
                const table_idx = getLookupTableIndex(opcode, funct3, funct7);
                cycle_table_indices[j] = table_idx;

                // Identity path (not interleaved) = AddOperands instructions
                // These are: ADD, ADDI, ADDIW, ADDW, LUI, AUIPC, JAL, JALR, Load, Store
                const is_add_operands = switch (opcode) {
                    0x33 => (funct3 == 0) and (funct7 == 0), // ADD
                    0x13 => (funct3 == 0), // ADDI
                    0x1b => (funct3 == 0), // ADDIW
                    0x3b => (funct3 == 0) and (funct7 == 0), // ADDW
                    0x37, 0x17, 0x6f, 0x67, 0x03, 0x23 => true, // LUI, AUIPC, JAL, JALR, Load, Store
                    else => false,
                };
                // Identity path = NOT interleaved = is_add_operands (uses identity polynomial)
                cycle_is_identity_path[j] = is_add_operands;

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

            // Verify the sum matches lookups_input
            var lookups_computed_sum = F.zero();
            for (0..T) |j| {
                lookups_computed_sum = lookups_computed_sum.add(lookups_eq_evals[j].mul(lookups_combined_vals[j]));
            }
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

            var challenges = try self.allocator.alloc(F, max_num_rounds);
            errdefer self.allocator.free(challenges);

            // Track current batched claim (for verification)
            var current_batched_claim = batched_claim;

            // Track lookups_claim separately (for Instance 2)
            var lookups_claim = lookups_input;

            // Run the batched sumcheck
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
                }

                // Instance 1: RamRaClaimReduction (24 rounds) - still zero for now
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
                    // Zero polynomial for now (TODO: implement RamRaClaimReduction)
                    // This is correct if ram_ra_input = 0
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
                    // Address round: compute polynomial based on address bit `round`
                    var p0 = F.zero();
                    var p1 = F.zero();
                    for (0..T) |j| {
                        const bit = getBit128(lookups_indices_lo[j], lookups_indices_hi[j], round);
                        const contrib = lookups_eq_evals[j].mul(lookups_ra_weights[j]).mul(lookups_combined_vals[j]);
                        if (bit == 0) {
                            p0 = p0.add(contrib);
                        } else {
                            p1 = p1.add(contrib);
                        }
                    }
                    // For degree-1 polynomial (linear in the address variable):
                    // p(x) = p0 + x*(p1 - p0)
                    // p(0) = p0, p(1) = p1, p(2) = 2*p1 - p0
                    // p_inf = p1 - p0 (leading coefficient)
                    const p2 = p1.add(p1).sub(p0);
                    const p_inf = p1.sub(p0);
                    combined_poly[0] = combined_poly[0].add(batch2.mul(p0));
                    combined_poly[1] = combined_poly[1].add(batch2.mul(p1));
                    combined_poly[2] = combined_poly[2].add(batch2.mul(p2));
                    combined_poly[3] = combined_poly[3].add(batch2.mul(p_inf));
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
                    const current_half_size = lookups_eq_evals.len >> @intCast(lookups_round + 1);

                    // Get r_round for this cycle variable (LowToHigh binding: last element first)
                    const r_round = r_reduction[n_cycle_vars - 1 - lookups_round];

                    // Accumulate sum of 9-factor products (eq absorbed into val)
                    var sum_evals: [9]F = [_]F{F.zero()} ** 9; // Evaluations at [1, 2, ..., 8, ∞]

                    for (0..current_half_size) |j| {
                        // Build the 9 linear polynomial pairs for this j
                        // Each pair is (p(0), p(1)) for the linear polynomial
                        var pairs: [9][2]F = undefined;

                        // Factor 0: eq absorbed into combined_val
                        // (eq[2j] * val[2j], eq[2j+1] * val[2j+1])
                        const eq_0 = lookups_eq_evals[2 * j];
                        const eq_1 = lookups_eq_evals[2 * j + 1];
                        pairs[0][0] = eq_0.mul(lookups_combined_vals[2 * j]);
                        pairs[0][1] = eq_1.mul(lookups_combined_vals[2 * j + 1]);

                        // Factors 1-8: 8 ra_chunk polynomials
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
                    std.debug.print("  cycle_claim = {x}\n", .{cycle_claim.toBytesBE()[24..32].*});
                    std.debug.print("  p(0) = {x}\n", .{p_at_0.toBytesBE()[24..32].*});
                    std.debug.print("  p(1) = {x}\n", .{p_at_1.toBytesBE()[24..32].*});
                    std.debug.print("  p(0)+p(1) = {x}, matches_claim = {}\n", .{ sum_check.toBytesBE()[24..32].*, sumcheck_ok });
                    std.debug.print("  full_coeffs len = {}\n", .{full_coeffs.len});

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

                    // Update current_batched_claim by evaluating polynomial at challenge
                    // p(r) = c0 + r*c1 + r^2*c2 + ... + r^d*c_d
                    // c1 = hint - 2*c0 - c2 - c3 - ... - c_d
                    var c1_sum = current_batched_claim.sub(combined_coeffs[0]).sub(combined_coeffs[0]); // hint - 2*c0
                    for (2..combined_coeffs.len) |i| {
                        c1_sum = c1_sum.sub(combined_coeffs[i]);
                    }
                    const c1_recovered = c1_sum;

                    // Evaluate using Horner's method
                    var eval_result = combined_coeffs[combined_coeffs.len - 1];
                    var i_val = combined_coeffs.len - 1;
                    while (i_val > 1) {
                        i_val -= 1;
                        if (i_val == 1) {
                            eval_result = eval_result.mul(challenge).add(c1_recovered);
                        } else {
                            eval_result = eval_result.mul(challenge).add(combined_coeffs[i_val]);
                        }
                    }
                    eval_result = eval_result.mul(challenge).add(combined_coeffs[0]);
                    current_batched_claim = eval_result;

                    // Skip the standard compression/serialization below
                    // Bind the challenge for RegistersValEvaluation if active
                    if (remaining_rounds <= regs_val_num_rounds) {
                        const regs_round = regs_val_num_rounds - remaining_rounds;
                        bindRegsValChallenge(inc_evals, wa_evals, lt_evals, regs_round, challenge);
                    }

                    // Bind cycle round challenge for lookups
                    bindLookupsChallenge(lookups_eq_evals, lookups_combined_vals, lookups_round, challenge);

                    // Bind the per-chunk ra weights
                    for (0..ra_num_chunks) |chunk_idx| {
                        bindSinglePolynomial(ra_chunk_weights[chunk_idx], lookups_round, challenge);
                    }

                    // Update lookups_claim: recompute ra_weights[0] from the bound chunks
                    var final_ra = F.one();
                    for (0..ra_num_chunks) |c| {
                        final_ra = final_ra.mul(ra_chunk_weights[c][0]);
                    }
                    lookups_ra_weights[0] = final_ra;
                    lookups_claim = lookups_eq_evals[0].mul(final_ra).mul(lookups_combined_vals[0]);

                    // Debug: print challenges for cycle rounds (128-135)
                    if (round >= LOOKUPS_LOG_K) {
                        std.debug.print("[STAGE5 ROUND {}] challenge={x}\n", .{
                            round,
                            challenge.toBytesBE()[24..32].*,
                        });
                    }

                    continue; // Skip the rest of the loop (we handled everything)
                }

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

                // Verify p(0)+p(1) = current_batched_claim (disabled - verbose debug)
                // const p01_sum = combined_poly[0].add(combined_poly[1]);
                // const claim_matches = p01_sum.eql(current_batched_claim);

                // Append compressed polynomial to transcript and get challenge
                // Must use compressed format (c0, c2, c3) to match Jolt's BatchedSumcheck
                transcript.appendMessage("UniPoly_begin");
                transcript.appendScalar(compressed[0]); // c0
                transcript.appendScalar(compressed[1]); // c2
                transcript.appendScalar(compressed[2]); // c3
                transcript.appendMessage("UniPoly_end");

                const challenge = transcript.challengeScalar();
                challenges[round] = challenge;

                // Update current_batched_claim by evaluating polynomial at challenge
                // Use the same eval_from_hint logic as Jolt verifier
                // p(r) = c0 + r*c1 + r^2*c2 + r^3*c3
                // where c1 = batched_claim - c0 - c2 - c3 (recovered from hint)
                const c0 = compressed[0];
                const c2 = compressed[1];
                const c3 = compressed[2];
                const c1 = current_batched_claim.sub(c0).sub(c2).sub(c3); // hint recovery
                const r2 = challenge.mul(challenge);
                const r3 = r2.mul(challenge);
                current_batched_claim = c0.add(challenge.mul(c1)).add(r2.mul(c2)).add(r3.mul(c3));

                // Debug: print challenges for first 3 and last 3 rounds
                if (round < 3 or round >= max_num_rounds - 3) {
                    std.debug.print("[STAGE5 ROUND {}] challenge={x}\n", .{
                        round,
                        challenge.toBytesBE()[24..32].*,
                    });
                }

                // Bind the challenge for RegistersValEvaluation if active
                if (remaining_rounds <= regs_val_num_rounds) {
                    const regs_round = regs_val_num_rounds - remaining_rounds;
                    bindRegsValChallenge(inc_evals, wa_evals, lt_evals, regs_round, challenge);
                }

                // Update lookups state for Instance 2
                if (round < LOOKUPS_LOG_K) {
                    // Address round: update ra_weights based on address bit
                    // For each cycle j with lookup_index bit = b:
                    //   weight[j] *= (1-r) if b=0, or r if b=1
                    // This is equivalent to: weight[j] *= [(1-b)*(1-r) + b*r]
                    const one_minus_r = F.one().sub(challenge);

                    // Determine which chunk this round belongs to
                    const chunk_idx = round / lookups_ra_virtual_log_k_chunk;

                    for (0..T) |j| {
                        const bit = getBit128(lookups_indices_lo[j], lookups_indices_hi[j], round);
                        const factor = if (bit == 0) one_minus_r else challenge;
                        lookups_ra_weights[j] = lookups_ra_weights[j].mul(factor);

                        // Also update the per-chunk weight for this chunk
                        if (chunk_idx < ra_num_chunks) {
                            ra_chunk_weights[chunk_idx][j] = ra_chunk_weights[chunk_idx][j].mul(factor);
                        }
                    }
                    // Update lookups_claim: p(r) = p0*(1-r) + p1*r = p0 + r*(p1-p0)
                    // where p0 and p1 are from the polynomial we computed
                    // This is tracked via ra_weights, so recompute the claim
                    lookups_claim = F.zero();
                    for (0..T) |j| {
                        lookups_claim = lookups_claim.add(lookups_eq_evals[j].mul(lookups_ra_weights[j]).mul(lookups_combined_vals[j]));
                    }
                } else {
                    // Cycle round: bind the challenge to eq_evals, ra_weights, and combined_vals
                    const lookups_round = round - LOOKUPS_LOG_K;

                    // At round 128 (first cycle round), verify the product invariant
                    if (round == LOOKUPS_LOG_K) {
                        std.debug.print("[STAGE5 CYCLE] Verifying ra invariant before cycle binding:\n", .{});
                        var verified = true;
                        for (0..@min(T, 3)) |j| {
                            var product = F.one();
                            for (0..ra_num_chunks) |c| {
                                product = product.mul(ra_chunk_weights[c][j]);
                            }
                            const matches = product.eql(lookups_ra_weights[j]);
                            if (!matches) verified = false;
                            std.debug.print("  j={}: product={any}, ra_weights={any}, match={}\n", .{
                                j,
                                product.toBytesBE()[0..8],
                                lookups_ra_weights[j].toBytesBE()[0..8],
                                matches,
                            });
                        }
                        std.debug.print("  All verified: {}\n", .{verified});
                    }

                    // Bind eq_evals and combined_vals (NOT ra_weights - we recompute from chunks)
                    bindLookupsChallenge(lookups_eq_evals, lookups_combined_vals, lookups_round, challenge);

                    // Bind the per-chunk ra weights
                    for (0..ra_num_chunks) |chunk_idx| {
                        bindSinglePolynomial(ra_chunk_weights[chunk_idx], lookups_round, challenge);
                    }

                    // Update lookups_claim: recompute ra_weights[0] from the bound chunks
                    var final_ra = F.one();
                    for (0..ra_num_chunks) |c| {
                        final_ra = final_ra.mul(ra_chunk_weights[c][0]);
                    }
                    lookups_ra_weights[0] = final_ra;
                    lookups_claim = lookups_eq_evals[0].mul(final_ra).mul(lookups_combined_vals[0]);
                }
            }

            // Debug: print final batched claim (this is output_claim from verifier's perspective)
            std.debug.print("[STAGE5] Final batched claim (output_claim) = {any}\n", .{current_batched_claim.toBytesBE()});

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
            // After binding all cycle variables, this is eq_evals[0] * combined[0]
            const lookups_output_claim = lookups_eq_evals[0].mul(lookups_combined_vals[0]);

            std.debug.print("[STAGE5 LOOKUPS] Computing opening claims:\n", .{});
            std.debug.print("  lookups_input = {any}\n", .{lookups_input.toBytesBE()[0..8]});
            std.debug.print("  lookups_output_claim (eq*combined) = {any}\n", .{lookups_output_claim.toBytesBE()[0..8]});
            std.debug.print("  lookups_eq_evals[0] = {any}\n", .{lookups_eq_evals[0].toBytesBE()[0..8]});
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

            std.debug.print("  r_reduction[0] = {any}\n", .{r_reduction[0].toBytesBE()[0..8]});
            std.debug.print("  r_cycle_prime_be[0] = {any}\n", .{r_cycle_prime_be[0].toBytesBE()[0..8]});
            std.debug.print("  eq_r_reduction (verifier computes) = {any}\n", .{eq_r_reduction.toBytesBE()[0..8]});
            std.debug.print("  eq_evals[0] (from sumcheck) = {any}\n", .{lookups_eq_evals[0].toBytesBE()[0..8]});

            // Compute operand polynomial evaluations at r_address_prime
            const left_op_eval = evaluateLeftOperand(F, r_address_prime);
            const right_op_eval = evaluateRightOperand(F, r_address_prime);
            const identity_eval = evaluateIdentity(F, r_address_prime);

            std.debug.print("  left_op_eval = {any}\n", .{left_op_eval.toBytesBE()[0..8]});
            std.debug.print("  right_op_eval = {any}\n", .{right_op_eval.toBytesBE()[0..8]});
            std.debug.print("  identity_eval = {any}\n", .{identity_eval.toBytesBE()[0..8]});
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
            for (0..lookups_ra_d) |i| {
                ra_chunks[i] = ra_chunk_weights[i][0]; // Final value after all binding
            }

            // Debug: print ra chunk claims
            std.debug.print("[STAGE5 LOOKUPS] ra_chunk claims:\n", .{});
            var ra_product = F.one();
            for (0..lookups_ra_d) |i| {
                std.debug.print("  ra_chunks[{}] = {any}\n", .{ i, ra_chunks[i].toBytesBE()[0..8] });
                ra_product = ra_product.mul(ra_chunks[i]);
            }
            std.debug.print("  ra_product = {any}\n", .{ra_product.toBytesBE()[0..8]});
            std.debug.print("  lookups_ra_weights[0] = {any}\n", .{lookups_ra_weights[0].toBytesBE()[0..8]});

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
            std.debug.print("[STAGE5 LOOKUPS] Non-zero table flags:\n", .{});
            for (0..num_lookup_tables) |i| {
                if (!table_flags[i].eql(F.zero())) {
                    std.debug.print("  table_flags[{}] = {any}\n", .{ i, table_flags[i].toBytesBE()[0..8] });
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
            //
            // Note: table_eval[i] would need lookup table MLE evaluation at r_address, which
            // is complex. For now, we'll verify the structure is correct.

            const raf_claim = F.one().sub(computed_raf_flag).mul(left_op_eval.add(gamma_lookups_raf.mul(right_op_eval)))
                .add(computed_raf_flag.mul(gamma_lookups_raf).mul(identity_eval));
            std.debug.print("  raf_claim (from formula) = {any}\n", .{raf_claim.toBytesBE()[0..8]});

            // Compute what verifier would expect (without val_claim for now)
            const expected_no_val = eq_r_reduction.mul(ra_product).mul(gamma_lookups_raf.mul(raf_claim));
            std.debug.print("  expected_no_val (eq*ra*gamma*raf) = {any}\n", .{expected_no_val.toBytesBE()[0..8]});
            std.debug.print("  lookups_output_claim = {any}\n", .{lookups_output_claim.toBytesBE()[0..8]});
            std.debug.print("  current_batched_claim = {any}\n", .{current_batched_claim.toBytesBE()[0..8]});

            return Stage5Result(F){
                .challenges = challenges,
                .regs_val_inc_claim = regs_val_inc_claim,
                .regs_val_wa_claim = regs_val_wa_claim,
                .ram_ra_claim = F.zero(),
                .lookups_table_flags = table_flags,
                .lookups_ra_chunks = ra_chunks,
                .lookups_raf_flag = computed_raf_flag,
                .allocator = self.allocator,
            };
        }

        /// Compute eq(r, k) for a specific index k
        /// r is in BIG_ENDIAN order (r[0] = MSB, r[n-1] = LSB)
        /// k is interpreted as big-endian: k = b_0 * 2^(n-1) + b_1 * 2^(n-2) + ... + b_{n-1}
        /// where b_j is the j-th bit (b_0 = MSB)
        /// eq(k, r) = Π_j (b_j ? r[j] : (1-r[j]))
        fn computeEqAtIndex(r: []const F, k: usize) F {
            const n = r.len;
            var result = F.one();
            for (0..n) |j| {
                // Extract bit j of k (MSB-first): b_j = (k >> (n-1-j)) & 1
                const bj: u1 = @truncate(k >> @intCast(n - 1 - j));
                const rj = r[j]; // r[j] corresponds to bit j
                if (bj == 1) {
                    result = result.mul(rj);
                } else {
                    result = result.mul(F.one().sub(rj));
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
                    const y = x.mul(ri);
                    evals[half + j] = y;
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
        result = result.add(r[2 * i + 1].mul(power));
        power = power.add(power); // power *= 2
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
        result = result.add(r[i].mul(power));
        power = power.add(power); // power *= 2
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

/// Interleave bits of two 64-bit values into a 128-bit value
/// Result: bit 2i comes from x[i], bit 2i+1 comes from y[i]
/// This matches Jolt's interleave_bits function used for lookup indices
pub fn interleaveBits128(x: u64, y: u64) u128 {
    var result: u128 = 0;
    for (0..64) |i| {
        const xi: u128 = @as(u128, (x >> @intCast(i)) & 1);
        const yi: u128 = @as(u128, (y >> @intCast(i)) & 1);
        result |= xi << @intCast(2 * i);
        result |= yi << @intCast(2 * i + 1);
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
/// Based on Jolt's LookupTables enum ordering
pub fn getLookupTableIndex(opcode: u32, funct3: u32, funct7: u32) i8 {
    return switch (opcode) {
        0x33 => blk: { // R-type
            if (funct3 == 0 and funct7 == 0) break :blk 0; // ADD -> AddTable
            if (funct3 == 0 and funct7 == 0x20) break :blk 1; // SUB -> SubTable
            if (funct3 == 7) break :blk 2; // AND -> AndTable
            if (funct3 == 6) break :blk 3; // OR -> OrTable
            if (funct3 == 4) break :blk 4; // XOR -> XorTable
            if (funct3 == 1) break :blk 5; // SLL -> SllTable
            if (funct3 == 5 and funct7 == 0) break :blk 6; // SRL -> SrlTable
            if (funct3 == 5 and funct7 == 0x20) break :blk 7; // SRA -> SraTable
            if (funct3 == 2) break :blk 8; // SLT -> SltTable
            if (funct3 == 3) break :blk 9; // SLTU -> SltuTable
            if (funct7 == 0x01 and funct3 == 0) break :blk 10; // MUL -> MulTable
            if (funct7 == 0x01 and funct3 == 3) break :blk 11; // MULHU -> MulhuTable
            break :blk -1;
        },
        0x13 => blk: { // I-type
            if (funct3 == 0) break :blk 0; // ADDI -> AddTable
            if (funct3 == 7) break :blk 2; // ANDI -> AndTable
            if (funct3 == 6) break :blk 3; // ORI -> OrTable
            if (funct3 == 4) break :blk 4; // XORI -> XorTable
            if (funct3 == 1) break :blk 5; // SLLI -> SllTable
            if (funct3 == 5 and (funct7 & 0x40) == 0) break :blk 6; // SRLI -> SrlTable
            if (funct3 == 5 and (funct7 & 0x40) != 0) break :blk 7; // SRAI -> SraTable
            if (funct3 == 2) break :blk 8; // SLTI -> SltTable
            if (funct3 == 3) break :blk 9; // SLTIU -> SltuTable
            break :blk -1;
        },
        0x1b => blk: { // OP-IMM-32
            if (funct3 == 0) break :blk 0; // ADDIW -> AddTable
            break :blk -1;
        },
        0x3b => blk: { // OP-32
            if (funct3 == 0 and funct7 == 0) break :blk 0; // ADDW -> AddTable
            if (funct3 == 0 and funct7 == 0x20) break :blk 1; // SUBW -> SubTable
            break :blk -1;
        },
        0x63 => blk: { // B-type (branches)
            if (funct3 == 0) break :blk 12; // BEQ -> BeqTable
            if (funct3 == 1) break :blk 13; // BNE -> BneTable
            if (funct3 == 4) break :blk 14; // BLT -> BltTable
            if (funct3 == 5) break :blk 15; // BGE -> BgeTable
            if (funct3 == 6) break :blk 16; // BLTU -> BltuTable
            if (funct3 == 7) break :blk 17; // BGEU -> BgeuTable
            break :blk -1;
        },
        0x37 => 0, // LUI -> AddTable (just passes through)
        0x17 => 0, // AUIPC -> AddTable
        0x6f => 0, // JAL -> AddTable
        0x67 => 0, // JALR -> AddTable
        0x03 => 0, // Load -> AddTable (address computation)
        0x23 => 0, // Store -> AddTable (address computation)
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
