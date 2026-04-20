//! Stage 5 Verifier: 3 Batched Instances
//!
//! Instances:
//!   [0] LookupsReadRaf:          degree 3, lookups_log_k + n_cycle_vars rounds
//!   [1] RamRaClaimReduction:     degree 3, n_cycle_vars rounds
//!   [2] RegistersValEvaluation:  degree 2, n_cycle_vars rounds
//!
//! Pre-batching challenges:
//!   gamma_ram_ra      = challengeScalarFull()
//!   gamma_lookups_raf = challengeScalarFull()
//!
//! NOTE: Batch claim order matches prover: [lookups, ram_ra, regs_val]

const std = @import("std");
const Allocator = std.mem.Allocator;

const zolt_arith = @import("zolt_arith");
const jolt_types = @import("../jolt_types.zig");
const opening_accumulator_mod = @import("opening_accumulator.zig");
const sumcheck_verifier = @import("sumcheck_verifier.zig");

const VirtualPolynomial = jolt_types.VirtualPolynomial;
const SumcheckId = jolt_types.SumcheckId;
const OpeningId = jolt_types.OpeningId;

pub fn Stage5Verifier(comptime F: type) type {
    return struct {
        const Self = @This();
        const Accumulator = opening_accumulator_mod.VerifierOpeningAccumulator(F);

        gamma_ram_ra: F,
        gamma_lookups_raf: F,
        n_cycle_vars: usize,
        lookups_log_k: usize, // typically 128 for RV64
        proof_opening_claims: *const jolt_types.OpeningClaims(F),

        pub fn sampleChallenges(transcript: anytype) struct { gamma_ram_ra: F, gamma_lookups_raf: F } {
            return .{
                .gamma_ram_ra = transcript.challengeScalarFull(),
                .gamma_lookups_raf = transcript.challengeScalarFull(),
            };
        }

        pub fn computeInputClaims(self: *const Self, acc: *const Accumulator) [3]F {
            const g_raf = self.gamma_lookups_raf;
            const g_ra = self.gamma_ram_ra;

            // [0] LookupsReadRaf: rv + γ*left + γ²*right
            const rv = readVirtual(acc, .LookupOutput, .InstructionClaimReduction);
            const left_op = readVirtual(acc, .LeftLookupOperand, .InstructionClaimReduction);
            const right_op = readVirtual(acc, .RightLookupOperand, .InstructionClaimReduction);
            const lookups_claim = rv.add(g_raf.mul(left_op)).add(g_raf.mul(g_raf).mul(right_op));

            // [1] RamRaClaimReduction: raf + γ*rw + γ²*val
            const raf_claim = readVirtual(acc, .RamRa, .RamRafEvaluation);
            const rw_claim = readVirtual(acc, .RamRa, .RamReadWriteChecking);
            const val_claim = readVirtual(acc, .RamRa, .RamValCheck);
            const ram_ra_claim = raf_claim.add(g_ra.mul(rw_claim)).add(g_ra.mul(g_ra).mul(val_claim));

            // [2] RegistersValEvaluation: from RegistersReadWriteChecking
            const regs_val = readVirtual(acc, .RegistersVal, .RegistersReadWriteChecking);

            return [3]F{ lookups_claim, ram_ra_claim, regs_val };
        }

        pub fn verify(
            self: *const Self,
            proof: *const jolt_types.SumcheckInstanceProof(F),
            transcript: anytype,
            acc: *Accumulator,
            allocator: Allocator,
        ) !sumcheck_verifier.SumcheckResult(F) {
            const input_claims = self.computeInputClaims(acc);
            const max_num_rounds = self.lookups_log_k + self.n_cycle_vars;
            const max_degree: usize = 3;
            const rounds = [3]usize{
                self.lookups_log_k + self.n_cycle_vars, // [0] LookupsReadRaf
                self.n_cycle_vars, // [1] RamRaClaimReduction
                self.n_cycle_vars, // [2] RegistersValEvaluation
            };

            for (input_claims) |claim| transcript.appendScalar("sumcheck_claim", claim);
            var batch: [3]F = undefined;
            for (&batch) |*c| c.* = transcript.challengeScalarFull();

            var batched = F.zero();
            for (0..3) |i| {
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
