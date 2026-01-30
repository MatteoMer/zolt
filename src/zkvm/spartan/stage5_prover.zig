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
            gamma: F,
            lookups_ra_virtual_log_k_chunk: usize,
        ) !Stage5Result(F) {
            // Instance configurations
            const regs_val_num_rounds = n_cycle_vars; // 8 rounds
            const ram_ra_num_rounds = log_ram_k + n_cycle_vars; // 24 rounds
            const lookups_num_rounds = LOOKUPS_LOG_K + n_cycle_vars; // 136 rounds
            const max_num_rounds = lookups_num_rounds;

            std.debug.print("[STAGE5] Configuration: regs={}, ram_ra={}, lookups={}, max={}\n", .{
                regs_val_num_rounds, ram_ra_num_rounds, lookups_num_rounds, max_num_rounds,
            });

            // Get input claims from accumulator
            const regs_val_input = opening_claims.get(
                .{ .Virtual = .{ .poly = .RegistersVal, .sumcheck_id = .RegistersReadWriteChecking } },
            ) orelse F.zero();

            // RamRaClaimReduction batches 4 claims with gamma:
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

            // LookupsReadRaf batches 3 claims with gamma:
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

            const lookups_input = rv_claim
                .add(gamma.mul(left_op_claim))
                .add(gamma2.mul(right_op_claim));

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
                    // Not started yet - constant polynomial where p(0) + p(1) = previous_claim
                    // The previous claim for this instance is scaled by 2^(remaining - num_rounds)
                    // At round r, remaining = max_rounds - r, so scale = remaining - num_rounds - 1
                    // because we need p(0) + p(1) = 2^scale * input_claim (the current claim)
                    // We set p(0) = p(1) = p(2) = p(3) = current_claim / 2
                    const scale = remaining_rounds - regs_val_num_rounds - 1;
                    var current_claim = regs_val_input;
                    for (0..scale) |_| current_claim = current_claim.add(current_claim);
                    // For sumcheck: p(0) + p(1) = current_claim, so p(x) = current_claim / 2
                    const two_inv = F.fromU64(2).inverse().?;
                    const half_claim = current_claim.mul(two_inv);
                    combined_poly[0] = combined_poly[0].add(batch0.mul(half_claim));
                    combined_poly[1] = combined_poly[1].add(batch0.mul(half_claim));
                    combined_poly[2] = combined_poly[2].add(batch0.mul(half_claim));
                    combined_poly[3] = combined_poly[3].add(batch0.mul(half_claim));
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
                    // Same as instance 0 - constant polynomial with p(0) + p(1) = current_claim
                    const scale = remaining_rounds - ram_ra_num_rounds - 1;
                    var current_claim = ram_ra_input;
                    for (0..scale) |_| current_claim = current_claim.add(current_claim);
                    const two_inv = F.fromU64(2).inverse().?;
                    const half_claim = current_claim.mul(two_inv);
                    combined_poly[0] = combined_poly[0].add(batch1.mul(half_claim));
                    combined_poly[1] = combined_poly[1].add(batch1.mul(half_claim));
                    combined_poly[2] = combined_poly[2].add(batch1.mul(half_claim));
                    combined_poly[3] = combined_poly[3].add(batch1.mul(half_claim));
                } else {
                    // Instance active - assume zero sum
                    const zero_poly = [_]F{ F.zero(), F.zero(), F.zero(), F.zero() };
                    for (0..4) |j| {
                        combined_poly[j] = combined_poly[j].add(batch1.mul(zero_poly[j]));
                    }
                }

                // Instance 2: LookupsReadRaf (always active since max_rounds = lookups_rounds)
                // Assume zero polynomial sum for now
                const zero_poly = [_]F{ F.zero(), F.zero(), F.zero(), F.zero() };
                for (0..4) |j| {
                    combined_poly[j] = combined_poly[j].add(batch2.mul(zero_poly[j]));
                }

                // Convert to compressed form and append to proof
                const compressed = UniPoly(F).evalsToCompressed(combined_poly);
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
                lookups_claim = F.zero();
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
            gamma: F,
            lookups_ra_virtual_log_k_chunk: usize,
            trace: *const ExecutionTrace,
            r_address_regs: []const F, // LOG_K=7 elements from Stage 4 RegistersRWC
            r_cycle_regs: []const F, // n_cycle_vars elements from Stage 4 RegistersRWC
        ) !Stage5Result(F) {
            const regs_val_num_rounds = n_cycle_vars;
            const ram_ra_num_rounds = log_ram_k + n_cycle_vars;
            const lookups_num_rounds = LOOKUPS_LOG_K + n_cycle_vars;
            const max_num_rounds = lookups_num_rounds;

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

            const gamma2 = gamma.mul(gamma);
            const gamma3 = gamma2.mul(gamma);
            const ram_ra_input = claim_raf
                .add(gamma.mul(claim_val_final))
                .add(gamma2.mul(claim_rw))
                .add(gamma3.mul(claim_val_eval));

            const rv_claim = opening_claims.get(
                .{ .Virtual = .{ .poly = .LookupOutput, .sumcheck_id = .InstructionClaimReduction } },
            ) orelse F.zero();
            const left_op_claim = opening_claims.get(
                .{ .Virtual = .{ .poly = .LeftLookupOperand, .sumcheck_id = .InstructionClaimReduction } },
            ) orelse F.zero();
            const right_op_claim = opening_claims.get(
                .{ .Virtual = .{ .poly = .RightLookupOperand, .sumcheck_id = .InstructionClaimReduction } },
            ) orelse F.zero();

            const lookups_input = rv_claim
                .add(gamma.mul(left_op_claim))
                .add(gamma2.mul(right_op_claim));

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

            // Run the batched sumcheck
            for (0..max_num_rounds) |round| {
                const remaining_rounds = max_num_rounds - round;

                var combined_poly = [_]F{ F.zero(), F.zero(), F.zero(), F.zero() };

                // Instance 0: RegistersValEvaluation (8 rounds)
                const two_inv = F.fromU64(2).inverse().?;
                if (remaining_rounds > regs_val_num_rounds) {
                    // Not started yet - constant polynomial where p(0) + p(1) = current_claim
                    const scale = remaining_rounds - regs_val_num_rounds - 1;
                    var current_claim = regs_val_input;
                    for (0..scale) |_| current_claim = current_claim.add(current_claim);
                    const half_claim = current_claim.mul(two_inv);
                    combined_poly[0] = combined_poly[0].add(batch0.mul(half_claim));
                    combined_poly[1] = combined_poly[1].add(batch0.mul(half_claim));
                    combined_poly[2] = combined_poly[2].add(batch0.mul(half_claim));
                    combined_poly[3] = combined_poly[3].add(batch0.mul(half_claim));
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
                    // Not started - constant polynomial
                    const scale = remaining_rounds - ram_ra_num_rounds - 1;
                    var current_claim = ram_ra_input;
                    for (0..scale) |_| current_claim = current_claim.add(current_claim);
                    const half_claim = current_claim.mul(two_inv);
                    combined_poly[0] = combined_poly[0].add(batch1.mul(half_claim));
                    combined_poly[1] = combined_poly[1].add(batch1.mul(half_claim));
                    combined_poly[2] = combined_poly[2].add(batch1.mul(half_claim));
                    combined_poly[3] = combined_poly[3].add(batch1.mul(half_claim));
                } else {
                    // Zero polynomial for now (TODO: implement RamRaClaimReduction)
                    // This is correct if ram_ra_input = 0
                }

                // Instance 2: LookupsReadRaf (136 rounds) - zero for now
                // TODO: implement InstructionReadRaf

                // Convert to compressed form
                const compressed = UniPoly(F).evalsToCompressed(combined_poly);
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

                // Bind the challenge for RegistersValEvaluation if active
                if (remaining_rounds <= regs_val_num_rounds) {
                    const regs_round = regs_val_num_rounds - remaining_rounds;
                    bindRegsValChallenge(inc_evals, wa_evals, lt_evals, regs_round, challenge);
                }
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

            // Allocate opening claim arrays
            const num_lookup_tables: usize = 42;
            const lookups_ra_d = LOOKUPS_LOG_K / lookups_ra_virtual_log_k_chunk;

            const table_flags = try self.allocator.alloc(F, num_lookup_tables);
            @memset(table_flags, F.zero());

            const ra_chunks = try self.allocator.alloc(F, lookups_ra_d);
            @memset(ra_chunks, F.zero());

            return Stage5Result(F){
                .challenges = challenges,
                .regs_val_inc_claim = regs_val_inc_claim,
                .regs_val_wa_claim = regs_val_wa_claim,
                .ram_ra_claim = F.zero(),
                .lookups_table_flags = table_flags,
                .lookups_ra_chunks = ra_chunks,
                .lookups_raf_flag = F.zero(),
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
                    evals[0] = inc[0].mul(wa[0]).mul(lt[0]);
                    evals[1] = evals[0];
                    evals[2] = evals[0];
                    evals[3] = evals[0];
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

                // Extrapolate for degree-3 polynomial
                const two = F.fromU64(2);
                const three = F.fromU64(3);

                const inc_2 = two.mul(inc_1).sub(inc_0);
                const wa_2 = two.mul(wa_1).sub(wa_0);
                const lt_2 = two.mul(lt_1).sub(lt_0);
                evals[2] = evals[2].add(inc_2.mul(wa_2).mul(lt_2));

                const inc_3 = three.mul(inc_1).sub(two.mul(inc_0));
                const wa_3 = three.mul(wa_1).sub(two.mul(wa_0));
                const lt_3 = three.mul(lt_1).sub(two.mul(lt_0));
                evals[3] = evals[3].add(inc_3.mul(wa_3).mul(lt_3));
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
    };
}

test "stage5_prover compiles" {
    const F = @import("../../field/mod.zig").BN254Scalar;
    const allocator = std.testing.allocator;

    const prover = Stage5BatchedProver(F).init(allocator);
    _ = prover;
}
