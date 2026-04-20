//! Stage 3 Verifier: 3 Batched Instances
//!
//! Instances (all n_cycle_vars rounds):
//!   [0] ShiftSumcheck:              degree 2
//!   [1] InstructionInputSumcheck:   degree 3
//!   [2] RegistersClaimReduction:    degree 2
//!
//! Pre-batching challenges:
//!   shift_gamma_powers[5] = powers of challengeScalarFull()
//!   instr_gamma           = challengeScalarFull()
//!   reg_gamma             = challengeScalarFull()

const std = @import("std");
const Allocator = std.mem.Allocator;

const zolt_arith = @import("zolt_arith");
const jolt_types = @import("../jolt_types.zig");
const opening_accumulator_mod = @import("opening_accumulator.zig");
const sumcheck_verifier = @import("sumcheck_verifier.zig");

const VirtualPolynomial = jolt_types.VirtualPolynomial;
const SumcheckId = jolt_types.SumcheckId;
const F = zolt_arith.field.BN254Scalar;

pub fn Stage3Verifier(comptime Field: type) type {
    return struct {
        const Self = @This();
        const Accumulator = opening_accumulator_mod.VerifierOpeningAccumulator(Field);

        shift_gamma_powers: [5]Field,
        instr_gamma: Field,
        reg_gamma: Field,
        n_cycle_vars: usize,
        proof_opening_claims: *const jolt_types.OpeningClaims(Field),

        pub fn sampleChallenges(transcript: anytype) struct { shift_gamma_powers: [5]Field, instr_gamma: Field, reg_gamma: Field } {
            // deriveGammaPowersFull(5): one challengeScalarFull → powers [1, γ, γ², γ³, γ⁴]
            const shift_gamma = transcript.challengeScalarFull();
            var shift_powers: [5]Field = undefined;
            shift_powers[0] = Field.one();
            shift_powers[1] = shift_gamma;
            for (2..5) |i| shift_powers[i] = shift_powers[i - 1].mul(shift_gamma);

            return .{
                .shift_gamma_powers = shift_powers,
                .instr_gamma = transcript.challengeScalarFull(),
                .reg_gamma = transcript.challengeScalarFull(),
            };
        }

        pub fn computeInputClaims(self: *const Self, acc: *const Accumulator) [3]Field {
            const gp = self.shift_gamma_powers;

            // [0] ShiftSumcheck input:
            //   NextUnexpandedPC + gp[1]*NextPC + gp[2]*NextIsVirtual + gp[3]*NextIsFirst + gp[4]*(1-NextIsNoop)
            const next_upc = readVirtual(acc, .NextUnexpandedPC, .SpartanOuter);
            const next_pc = readVirtual(acc, .NextPC, .SpartanOuter);
            const next_virt = readVirtual(acc, .NextIsVirtual, .SpartanOuter);
            const next_first = readVirtual(acc, .NextIsFirstInSequence, .SpartanOuter);
            const next_noop = readVirtual(acc, .NextIsNoop, .SpartanProductVirtualization);

            const shift_claim = next_upc
                .add(gp[1].mul(next_pc))
                .add(gp[2].mul(next_virt))
                .add(gp[3].mul(next_first))
                .add(gp[4].mul(Field.one().sub(next_noop)));

            // [1] InstructionInputSumcheck: right + instr_gamma * left
            const left_instr = readVirtual(acc, .LeftInstructionInput, .SpartanProductVirtualization);
            const right_instr = readVirtual(acc, .RightInstructionInput, .SpartanProductVirtualization);
            const instr_claim = right_instr.add(self.instr_gamma.mul(left_instr));

            // [2] RegistersClaimReduction: rd + reg_gamma * rs1 + reg_gamma² * rs2
            const rd = readVirtual(acc, .RdWriteValue, .SpartanOuter);
            const rs1 = readVirtual(acc, .Rs1Value, .SpartanOuter);
            const rs2 = readVirtual(acc, .Rs2Value, .SpartanOuter);
            const rg2 = self.reg_gamma.mul(self.reg_gamma);
            const reg_claim = rd.add(self.reg_gamma.mul(rs1)).add(rg2.mul(rs2));

            return [3]Field{ shift_claim, instr_claim, reg_claim };
        }

        pub fn verify(
            self: *const Self,
            proof: *const jolt_types.SumcheckInstanceProof(Field),
            transcript: anytype,
            acc: *Accumulator,
            allocator: Allocator,
        ) !sumcheck_verifier.SumcheckResult(Field) {
            const input_claims = self.computeInputClaims(acc);
            const max_degree: usize = 3;

            // All 3 instances have same round count → no staggering
            for (input_claims) |claim| transcript.appendScalar("sumcheck_claim", claim);
            var batch: [3]Field = undefined;
            for (&batch) |*c| c.* = transcript.challengeScalarFull();

            var batched = Field.zero();
            for (0..3) |i| batched = batched.add(batch[i].mul(input_claims[i]));

            return sumcheck_verifier.verifySumcheck(Field, proof, batched, self.n_cycle_vars, max_degree, transcript, allocator);
        }

        pub fn cacheOpenings(self: *const Self, acc: *Accumulator, challenges: []const Field) void {
            // TODO: Register per-instance polynomial openings from proof.opening_claims
            _ = self;
            _ = acc;
            _ = challenges;
        }

        fn readVirtual(acc: *const Accumulator, poly: VirtualPolynomial, sc_id: SumcheckId) Field {
            const entry = acc.getVirtual(poly, sc_id);
            return if (entry) |e| e.claim else Field.zero();
        }
    };
}
