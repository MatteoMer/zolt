//! Stage 6 Verifier: 6 Batched Instances
//!
//! Instances:
//!   [0] BytecodeReadRaf:          degree bytecode_d+1,   bytecode_log_k + n_cycle_vars rounds
//!   [1] Booleanity:               degree 3,              log_k_chunk + n_cycle_vars rounds
//!   [2] HammingBooleanity:        degree 3,              n_cycle_vars rounds
//!   [3] RamRaVirtualization:      degree ram_d+1,        n_cycle_vars rounds
//!   [4] LookupsRaVirtualization:  degree instr_d+1,      n_cycle_vars rounds
//!   [5] IncClaimReduction:        degree 2,              n_cycle_vars rounds
//!
//! Pre-batching challenges (complex — many challengeScalarPowers calls):
//!   7 gamma arrays + booleanity_gamma + lookups_ra_gamma + inc_gamma

const std = @import("std");
const Allocator = std.mem.Allocator;

const zolt_arith = @import("zolt_arith");
const jolt_types = @import("../jolt_types.zig");
const opening_accumulator_mod = @import("opening_accumulator.zig");
const sumcheck_verifier = @import("sumcheck_verifier.zig");

const VirtualPolynomial = jolt_types.VirtualPolynomial;
const SumcheckId = jolt_types.SumcheckId;
const OpeningId = jolt_types.OpeningId;
const CommittedPolynomial = jolt_types.CommittedPolynomial;

pub fn Stage6Verifier(comptime F: type) type {
    return struct {
        const Self = @This();
        const Accumulator = opening_accumulator_mod.VerifierOpeningAccumulator(F);

        inc_gamma: F,
        n_cycle_vars: usize,
        log_k_chunk: usize,
        proof_opening_claims: *const jolt_types.OpeningClaims(F),

        /// Sample all pre-batching challenges from transcript.
        /// This is complex: many challengeScalarPowers + individual challenges.
        /// Must match the prover's exact sequence.
        pub fn sampleChallenges(
            transcript: anytype,
            log_k_chunk: usize,
            lookups_log_k: usize,
            n_virtual_ra_polys: usize,
            allocator: Allocator,
        ) !struct { inc_gamma: F } {
            // Stage 6 pre-batching challenge sequence:
            // 1. challengeScalarPowers(8)  → bytecode_raf_gamma_powers
            const bcraf = try transcript.challengeScalarPowers(allocator, 8);
            defer allocator.free(bcraf);

            // 2. challengeScalarPowers(16) → stage1_gammas (2 + NUM_CIRCUIT_FLAGS=14)
            const s1g = try transcript.challengeScalarPowers(allocator, 16);
            defer allocator.free(s1g);

            // 3. challengeScalarPowers(4)  → stage2_gammas
            const s2g = try transcript.challengeScalarPowers(allocator, 4);
            defer allocator.free(s2g);

            // 4. challengeScalarPowers(9)  → stage3_gammas
            const s3g = try transcript.challengeScalarPowers(allocator, 9);
            defer allocator.free(s3g);

            // 5. challengeScalarPowers(3)  → stage4_gammas
            const s4g = try transcript.challengeScalarPowers(allocator, 3);
            defer allocator.free(s4g);

            // 6. challengeScalarPowers(42) → stage5_gammas (2 + NUM_LOOKUP_TABLES=40)
            const s5g = try transcript.challengeScalarPowers(allocator, 42);
            defer allocator.free(s5g);

            // 7. If lookups_log_k < log_k_chunk: consume extra challenges
            if (lookups_log_k < log_k_chunk) {
                for (0..log_k_chunk - lookups_log_k) |_| {
                    _ = transcript.challengeScalar();
                }
            }

            // 8. booleanity_gamma
            _ = transcript.challengeScalar();

            // 9. challengeScalarPowers(n_virtual_ra_polys) → lookups_ra_gamma
            const lrg = try transcript.challengeScalarPowers(allocator, n_virtual_ra_polys);
            defer allocator.free(lrg);

            // 10. inc_gamma
            const inc_gamma = transcript.challengeScalarFull();

            return .{ .inc_gamma = inc_gamma };
        }

        pub fn computeInputClaims(self: *const Self, acc: *const Accumulator) [6]F {
            // [0] BytecodeReadRaf: complex computation from multi-stage claims
            // TODO: Full bytecode RAF input claim computation
            const bytecode_claim = F.zero();

            // [1] Booleanity: always 0
            const bool_claim = F.zero();

            // [2] HammingBooleanity: always 0
            const hamming_claim = F.zero();

            // [3] RamRaVirtualization: from RamRaClaimReduction
            const ram_ra_virt = readVirtual(acc, .RamRa, .RamRaClaimReduction);

            // [4] LookupsRaVirtualization: sum of gamma * InstructionRa[i] claims
            // TODO: Sum over n_virtual_ra_polys InstructionRa claims with gamma powers
            const lookups_ra_virt = F.zero();

            // [5] IncClaimReduction: v1 + γ*v2 + γ²*w1 + γ³*w2
            const g = self.inc_gamma;
            const g2 = g.mul(g);
            const g3 = g2.mul(g);
            const v1 = readCommitted(acc, .RamInc, .RamReadWriteChecking);
            const v2 = readCommitted(acc, .RamInc, .RamValCheck);
            const w1 = readCommitted(acc, .RdInc, .RegistersReadWriteChecking);
            const w2 = readCommitted(acc, .RdInc, .RegistersValEvaluation);
            const inc_claim = v1.add(g.mul(v2)).add(g2.mul(w1)).add(g3.mul(w2));

            return [6]F{ bytecode_claim, bool_claim, hamming_claim, ram_ra_virt, lookups_ra_virt, inc_claim };
        }

        pub fn verify(
            self: *const Self,
            proof: *const jolt_types.SumcheckInstanceProof(F),
            transcript: anytype,
            acc: *Accumulator,
            allocator: Allocator,
        ) !sumcheck_verifier.SumcheckResult(F) {
            const input_claims = self.computeInputClaims(acc);
            // Stage 6 max degree varies; use 3 as safe upper bound
            // TODO: Compute actual max_degree from bytecode_d, ram_d, instruction_d
            const max_degree: usize = 3;
            const num_rounds = proof.compressed_polys.items.len;

            for (input_claims) |claim| transcript.appendScalar("sumcheck_claim", claim);
            var batch: [6]F = undefined;
            for (&batch) |*c| c.* = transcript.challengeScalarFull();

            // For Stage 6, round counts vary significantly.
            // The initial claim scaling needs the per-instance round counts.
            // For now, assume all instances have the same rounds (simplified).
            // TODO: Use actual per-instance round counts for proper staggering
            var batched = F.zero();
            for (0..6) |i| batched = batched.add(batch[i].mul(input_claims[i]));

            return sumcheck_verifier.verifySumcheck(F, proof, batched, num_rounds, max_degree, transcript, allocator);
        }

        pub fn cacheOpenings(self: *const Self, acc: *Accumulator, challenges: []const F) void {
            _ = self;
            _ = acc;
            _ = challenges;
        }

        fn readVirtual(acc: *const Accumulator, poly: VirtualPolynomial, sc_id: SumcheckId) F {
            const entry = acc.getVirtual(poly, sc_id);
            return if (entry) |e| e.claim else F.zero();
        }

        fn readCommitted(acc: *const Accumulator, poly: CommittedPolynomial, sc_id: SumcheckId) F {
            const entry = acc.getCommitted(poly, sc_id);
            return if (entry) |e| e.claim else F.zero();
        }
    };
}
