//! Stage 7 Verifier: 1 Instance
//!
//! Instance:
//!   [0] HammingWeightClaimReduction: degree 2, log_k_chunk rounds
//!
//! Pre-batching challenges:
//!   gamma = challengeScalarFull()
//!   gamma_powers[0..3*N-1] computed from gamma

const std = @import("std");
const Allocator = std.mem.Allocator;

const zolt_arith = @import("zolt_arith");
const jolt_types = @import("../jolt_types.zig");
const opening_accumulator_mod = @import("opening_accumulator.zig");
const sumcheck_verifier = @import("sumcheck_verifier.zig");

pub fn Stage7Verifier(comptime F: type) type {
    return struct {
        const Self = @This();
        const Accumulator = opening_accumulator_mod.VerifierOpeningAccumulator(F);

        gamma: F,
        log_k_chunk: usize,
        proof_opening_claims: *const jolt_types.OpeningClaims(F),

        pub fn sampleChallenges(transcript: anytype) struct { gamma: F } {
            return .{ .gamma = transcript.challengeScalarFull() };
        }

        pub fn computeInputClaim(self: *const Self, acc: *const Accumulator) F {
            // TODO: Compute full input claim from Stage 6 booleanity and RA virtualization claims
            // input = Σ_{i=0}^{N-1} (γ^(3i)*hw + γ^(3i+1)*bool + γ^(3i+2)*virt)
            // Requires knowing N = instruction_d + bytecode_d + ram_d and Stage 6 output claims
            _ = self;
            _ = acc;
            return F.zero();
        }

        pub fn verify(
            self: *const Self,
            proof: *const jolt_types.SumcheckInstanceProof(F),
            transcript: anytype,
            allocator: Allocator,
        ) !sumcheck_verifier.SumcheckResult(F) {
            const input_claim = self.computeInputClaim(&opening_accumulator_mod.VerifierOpeningAccumulator(F).init(allocator));
            const max_degree: usize = 2;

            // Single instance: append claim and derive single batch coeff
            transcript.appendScalar("sumcheck_claim", input_claim);
            _ = transcript.challengeScalarFull(); // single batch coeff

            return sumcheck_verifier.verifySumcheck(F, proof, input_claim, self.log_k_chunk, max_degree, transcript, allocator);
        }

        pub fn cacheOpenings(self: *const Self, acc: *Accumulator, challenges: []const F) void {
            _ = self;
            _ = acc;
            _ = challenges;
        }
    };
}
