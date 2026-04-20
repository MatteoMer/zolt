//! Stage 4 Verifier: 2 Batched Instances
//!
//! Instances:
//!   [0] RegistersReadWriteChecking:  degree 3, LOG_REGISTERS + n_cycle_vars rounds
//!   [1] RamValCheck:                 degree 3, n_cycle_vars rounds
//!
//! Pre-batching challenges:
//!   gamma_stage4        = challengeScalarFull()
//!   appendBytes("ram_val_check_gamma", &.{})
//!   ram_val_check_gamma = challengeScalarFull()

const std = @import("std");
const Allocator = std.mem.Allocator;

const zolt_arith = @import("zolt_arith");
const jolt_types = @import("../jolt_types.zig");
const opening_accumulator_mod = @import("opening_accumulator.zig");
const sumcheck_verifier = @import("sumcheck_verifier.zig");

const VirtualPolynomial = jolt_types.VirtualPolynomial;
const SumcheckId = jolt_types.SumcheckId;

const LOG_REGISTERS: usize = 7; // log2(128 registers)

pub fn Stage4Verifier(comptime F: type) type {
    return struct {
        const Self = @This();
        const Accumulator = opening_accumulator_mod.VerifierOpeningAccumulator(F);

        gamma_stage4: F,
        ram_val_check_gamma: F,
        n_cycle_vars: usize,
        proof_opening_claims: *const jolt_types.OpeningClaims(F),

        pub fn sampleChallenges(transcript: anytype) struct { gamma_stage4: F, ram_val_check_gamma: F } {
            const gamma_stage4 = transcript.challengeScalarFull();
            transcript.appendBytes("ram_val_check_gamma", &.{});
            const ram_val_check_gamma = transcript.challengeScalarFull();
            return .{ .gamma_stage4 = gamma_stage4, .ram_val_check_gamma = ram_val_check_gamma };
        }

        pub fn computeInputClaims(self: *const Self, acc: *const Accumulator) [2]F {
            // [0] RegistersReadWriteChecking: rd + gamma * rs1 + gamma² * rs2
            //     (from Stage 3 RegistersClaimReduction openings)
            const rd = readVirtual(acc, .RdWriteValue, .RegistersClaimReduction);
            const rs1 = readVirtual(acc, .Rs1Value, .RegistersClaimReduction);
            const rs2 = readVirtual(acc, .Rs2Value, .RegistersClaimReduction);
            const g2 = self.gamma_stage4.mul(self.gamma_stage4);
            const reg_claim = rd.add(self.gamma_stage4.mul(rs1)).add(g2.mul(rs2));

            // [1] RamValCheck: val_eval_claim + ram_val_check_gamma * val_final_claim
            //     (from Stage 2 RamReadWriteChecking + OutputSumcheck openings)
            const val_eval = readVirtual(acc, .RamVal, .RamReadWriteChecking);
            const val_final = readVirtual(acc, .RamValFinal, .RamOutputCheck);
            const ram_claim = val_eval.add(self.ram_val_check_gamma.mul(val_final));

            return [2]F{ reg_claim, ram_claim };
        }

        pub fn verify(
            self: *const Self,
            proof: *const jolt_types.SumcheckInstanceProof(F),
            transcript: anytype,
            acc: *Accumulator,
            allocator: Allocator,
        ) !sumcheck_verifier.SumcheckResult(F) {
            const input_claims = self.computeInputClaims(acc);
            const max_num_rounds = LOG_REGISTERS + self.n_cycle_vars;
            const max_degree: usize = 3;
            const rounds = [2]usize{ LOG_REGISTERS + self.n_cycle_vars, self.n_cycle_vars };

            for (input_claims) |claim| transcript.appendScalar("sumcheck_claim", claim);
            var batch: [2]F = undefined;
            for (&batch) |*c| c.* = transcript.challengeScalarFull();

            var batched = F.zero();
            for (0..2) |i| {
                const scale_power = max_num_rounds - rounds[i];
                var scaled = input_claims[i];
                for (0..scale_power) |_| scaled = scaled.add(scaled);
                batched = batched.add(batch[i].mul(scaled));
            }

            return sumcheck_verifier.verifySumcheck(F, proof, batched, max_num_rounds, max_degree, transcript, allocator);
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
    };
}
