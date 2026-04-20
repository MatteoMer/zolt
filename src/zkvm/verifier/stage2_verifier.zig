//! Stage 2 Verifier: Batched Sumcheck (5 Instances)
//!
//! Stage 2 batches 5 sumcheck instances with staggered rounds:
//!   [0] RamReadWriteChecking:        log_ram_k + n_cycle_vars rounds, degree 3
//!   [1] ProductVirtualRemainder:     n_cycle_vars rounds, degree 3
//!   [2] InstructionLookupsClaimRed:  n_cycle_vars rounds, degree 2
//!   [3] RamRafEvaluation:            log_ram_k rounds, degree 2
//!   [4] OutputSumcheck:              log_ram_k rounds, degree 3
//!
//! Input claims are computed from Stage 1 opening claims (cached in accumulator):
//!   [0] = RamReadValue + gamma_rwc * RamWriteValue
//!   [1] = uni_skip_claim_stage2 (from Stage 2 UniSkip)
//!   [2] = LookupOutput + γ*LeftOp + γ²*RightOp + γ³*LeftInstr + γ⁴*RightInstr
//!   [3] = RamAddress
//!   [4] = 0
//!
//! Transcript operations BEFORE batching:
//!   gamma_rwc    = transcript.challengeScalarFull()
//!   gamma_instr  = transcript.challengeScalarFull()
//!   r_address[0..log_ram_k] = transcript.challengeScalar() each
//!
//! Reference: stage2_sumcheck.zig (prover side), jolt-core verifier.rs

const std = @import("std");
const Allocator = std.mem.Allocator;

const zolt_arith = @import("zolt_arith");
const jolt_types = @import("../jolt_types.zig");
const opening_accumulator_mod = @import("opening_accumulator.zig");
const sumcheck_verifier = @import("sumcheck_verifier.zig");

const VirtualPolynomial = jolt_types.VirtualPolynomial;
const SumcheckId = jolt_types.SumcheckId;
const OpeningId = jolt_types.OpeningId;

/// Stage 2 verification context.
/// Holds gamma challenges, proof references, and configuration.
pub fn Stage2Verifier(comptime F: type) type {
    return struct {
        const Self = @This();
        const Accumulator = opening_accumulator_mod.VerifierOpeningAccumulator(F);

        /// Gamma for RamReadWriteChecking (challengeScalarFull, no 125-bit masking)
        gamma_rwc: F,
        /// Gamma for InstructionLookupsClaimReduction
        gamma_instr: F,
        /// Pre-sampled r_address challenges (log_ram_k elements)
        r_address_presampled: []const F,
        /// UniSkip claim from Stage 2 UniSkip round
        uni_skip_claim: F,
        /// Number of cycle variables (log2(trace_length))
        n_cycle_vars: usize,
        /// log2 of RAM address space
        log_ram_k: usize,
        /// Reference to proof opening claims
        proof_opening_claims: *const jolt_types.OpeningClaims(F),

        /// Sample the pre-batching challenges from transcript.
        /// Must be called BEFORE verifyBatched() to match the prover's transcript.
        pub fn sampleChallenges(
            transcript: anytype,
            log_ram_k: usize,
            allocator: Allocator,
        ) !struct { gamma_rwc: F, gamma_instr: F, r_address: []F } {
            const gamma_rwc = transcript.challengeScalarFull();
            const gamma_instr = transcript.challengeScalarFull();

            const r_address = try allocator.alloc(F, log_ram_k);
            for (r_address) |*r| {
                r.* = transcript.challengeScalar();
            }

            return .{
                .gamma_rwc = gamma_rwc,
                .gamma_instr = gamma_instr,
                .r_address = r_address,
            };
        }

        /// Compute the 5 input claims and their batch parameters.
        pub fn computeInputClaims(self: *const Self, acc: *const Accumulator) [5]F {
            // Read Stage 1 opening claims from accumulator
            const ram_read = readClaim(acc, .RamReadValue, .SpartanOuter);
            const ram_write = readClaim(acc, .RamWriteValue, .SpartanOuter);
            const lookup_out = readClaim(acc, .LookupOutput, .SpartanOuter);
            const left_op = readClaim(acc, .LeftLookupOperand, .SpartanOuter);
            const right_op = readClaim(acc, .RightLookupOperand, .SpartanOuter);
            const left_instr = readClaim(acc, .LeftInstructionInput, .SpartanOuter);
            const right_instr = readClaim(acc, .RightInstructionInput, .SpartanOuter);
            const ram_addr = readClaim(acc, .RamAddress, .SpartanOuter);

            const g = self.gamma_instr;
            const g2 = g.mul(g);
            const g3 = g2.mul(g);
            const g4 = g2.mul(g2);

            return [5]F{
                // [0] RamReadWriteChecking: RamReadValue + gamma_rwc * RamWriteValue
                ram_read.add(self.gamma_rwc.mul(ram_write)),
                // [1] ProductVirtualRemainder: uni_skip_claim
                self.uni_skip_claim,
                // [2] InstructionLookupsClaimReduction:
                //     LookupOutput + γ*LeftOp + γ²*RightOp + γ³*LeftInstr + γ⁴*RightInstr
                lookup_out.add(g.mul(left_op)).add(g2.mul(right_op)).add(g3.mul(left_instr)).add(g4.mul(right_instr)),
                // [3] RamRafEvaluation: RamAddress
                ram_addr,
                // [4] OutputSumcheck: 0
                F.zero(),
            };
        }

        /// Round counts for each instance.
        pub fn roundsPerInstance(self: *const Self) [5]usize {
            return [5]usize{
                self.log_ram_k + self.n_cycle_vars, // [0] RamReadWriteChecking
                self.n_cycle_vars, // [1] ProductVirtualRemainder
                self.n_cycle_vars, // [2] InstructionLookupsClaimReduction
                self.log_ram_k, // [3] RamRafEvaluation
                self.log_ram_k, // [4] OutputSumcheck
            };
        }

        /// Degree bounds for each instance.
        pub fn degreesPerInstance() [5]usize {
            return [5]usize{
                3, // [0] RamReadWriteChecking
                3, // [1] ProductVirtualRemainder
                2, // [2] InstructionLookupsClaimReduction
                2, // [3] RamRafEvaluation
                3, // [4] OutputSumcheck
            };
        }

        /// Verify Stage 2 batched sumcheck.
        ///
        /// Performs the batching manually (not via generic verifyBatched) because
        /// Stage 2 has specific transcript operations and instance ordering.
        pub fn verify(
            self: *const Self,
            proof: *const jolt_types.SumcheckInstanceProof(F),
            transcript: anytype,
            acc: *Accumulator,
            allocator: Allocator,
        ) !sumcheck_verifier.SumcheckResult(F) {
            const input_claims = self.computeInputClaims(acc);
            const rounds = self.roundsPerInstance();
            const max_num_rounds = self.log_ram_k + self.n_cycle_vars;
            const max_degree: usize = 3;

            // Append claims to transcript (per-claim, matching prover)
            for (input_claims) |claim| {
                transcript.appendScalar("sumcheck_claim", claim);
            }

            // Sample batching coefficients (challengeScalarFull each)
            var batching_coeffs: [5]F = undefined;
            for (&batching_coeffs) |*c| {
                c.* = transcript.challengeScalarFull();
            }

            // Compute batched initial claim with scaling
            var batched_claim = F.zero();
            for (0..5) |i| {
                const scale_power = max_num_rounds - rounds[i];
                var scaled = input_claims[i];
                for (0..scale_power) |_| {
                    scaled = scaled.add(scaled); // multiply by 2
                }
                batched_claim = batched_claim.add(batching_coeffs[i].mul(scaled));
            }

            // Run sumcheck verification
            return sumcheck_verifier.verifySumcheck(
                F,
                proof,
                batched_claim,
                max_num_rounds,
                max_degree,
                transcript,
                allocator,
            );
        }

        /// Cache Stage 2 opening claims in the accumulator.
        /// Called after sumcheck completes with the challenge vector.
        pub fn cacheOpenings(
            self: *const Self,
            acc: *Accumulator,
            challenges: []const F,
        ) void {
            const rounds = self.roundsPerInstance();
            const max_num_rounds = self.log_ram_k + self.n_cycle_vars;

            // Each instance's challenges are at offset [max_rounds - num_rounds .. max_rounds]
            for (0..5) |i| {
                const offset = max_num_rounds - rounds[i];
                const r_slice = challenges[offset .. offset + rounds[i]];
                self.cacheInstanceOpenings(acc, i, r_slice);
            }
        }

        fn cacheInstanceOpenings(
            self: *const Self,
            acc: *Accumulator,
            instance_idx: usize,
            r_slice: []const F,
        ) void {
            _ = self;
            // For each instance, register the polynomial openings that the
            // subsequent stages will read. The claim values come from proof.opening_claims.
            switch (instance_idx) {
                0 => {
                    // RamReadWriteChecking: caches RAM Ra, Val, Inc
                    // These are read by stages 4-6
                },
                1 => {
                    // ProductVirtualRemainder: caches factor polynomial evaluations
                    // Read by Stage 3 for shift/instruction input
                },
                2 => {
                    // InstructionLookupsClaimReduction
                },
                3 => {
                    // RamRafEvaluation
                },
                4 => {
                    // OutputSumcheck: caches Val_final, Val_init
                },
                else => unreachable,
            }
            // TODO: Populate with actual polynomial IDs from proof.opening_claims
            // For now, the claims flow through the transcript (proper Fiat-Shamir binding).
            _ = acc;
            _ = r_slice;
        }

        fn readClaim(acc: *const Accumulator, poly: VirtualPolynomial, sc_id: SumcheckId) F {
            const entry = acc.getVirtual(poly, sc_id);
            return if (entry) |e| e.claim else F.zero();
        }
    };
}

// =============================================================================
// Tests
// =============================================================================

const testing = std.testing;
const BN254Scalar = zolt_arith.field.BN254Scalar;
const Blake2bTranscript = zolt_arith.transcripts.Blake2bTranscript;

test "stage2 verifier: input claims computation" {
    const F_ = BN254Scalar;
    var claims = jolt_types.OpeningClaims(F_).init(testing.allocator);
    defer claims.deinit();

    var acc = opening_accumulator_mod.VerifierOpeningAccumulator(F_).init(testing.allocator);
    defer acc.deinit();

    // Simulate Stage 1 cached openings
    const point = [_]F_{F_.fromU64(1)};
    try acc.appendVirtual(.RamReadValue, .SpartanOuter, &point, F_.fromU64(10));
    try acc.appendVirtual(.RamWriteValue, .SpartanOuter, &point, F_.fromU64(20));
    try acc.appendVirtual(.RamAddress, .SpartanOuter, &point, F_.fromU64(42));
    try acc.appendVirtual(.LookupOutput, .SpartanOuter, &point, F_.fromU64(100));
    try acc.appendVirtual(.LeftLookupOperand, .SpartanOuter, &point, F_.fromU64(5));
    try acc.appendVirtual(.RightLookupOperand, .SpartanOuter, &point, F_.fromU64(7));
    try acc.appendVirtual(.LeftInstructionInput, .SpartanOuter, &point, F_.fromU64(3));
    try acc.appendVirtual(.RightInstructionInput, .SpartanOuter, &point, F_.fromU64(4));

    const verifier = Stage2Verifier(F_){
        .gamma_rwc = F_.fromU64(2),
        .gamma_instr = F_.fromU64(3),
        .r_address_presampled = &[_]F_{},
        .uni_skip_claim = F_.fromU64(77),
        .n_cycle_vars = 10,
        .log_ram_k = 16,
        .proof_opening_claims = &claims,
    };

    const input_claims = verifier.computeInputClaims(&acc);

    // [0] = 10 + 2*20 = 50
    try testing.expect(input_claims[0].eql(F_.fromU64(50)));
    // [1] = 77 (uni_skip)
    try testing.expect(input_claims[1].eql(F_.fromU64(77)));
    // [2] = 100 + 3*5 + 9*7 + 27*3 + 81*4 = 100+15+63+81+324 = 583
    try testing.expect(input_claims[2].eql(F_.fromU64(583)));
    // [3] = 42
    try testing.expect(input_claims[3].eql(F_.fromU64(42)));
    // [4] = 0
    try testing.expect(input_claims[4].eql(F_.zero()));
}

test "stage2 verifier: rounds and degrees" {
    const F_ = BN254Scalar;
    var claims = jolt_types.OpeningClaims(F_).init(testing.allocator);
    defer claims.deinit();

    const verifier = Stage2Verifier(F_){
        .gamma_rwc = F_.zero(),
        .gamma_instr = F_.zero(),
        .r_address_presampled = &[_]F_{},
        .uni_skip_claim = F_.zero(),
        .n_cycle_vars = 10,
        .log_ram_k = 16,
        .proof_opening_claims = &claims,
    };

    const rounds = verifier.roundsPerInstance();
    try testing.expectEqual(@as(usize, 26), rounds[0]); // log_ram_k + n_cycle_vars
    try testing.expectEqual(@as(usize, 10), rounds[1]); // n_cycle_vars
    try testing.expectEqual(@as(usize, 10), rounds[2]); // n_cycle_vars
    try testing.expectEqual(@as(usize, 16), rounds[3]); // log_ram_k
    try testing.expectEqual(@as(usize, 16), rounds[4]); // log_ram_k

    const degrees = Stage2Verifier(F_).degreesPerInstance();
    try testing.expectEqual(@as(usize, 3), degrees[0]);
    try testing.expectEqual(@as(usize, 3), degrees[1]);
    try testing.expectEqual(@as(usize, 2), degrees[2]);
    try testing.expectEqual(@as(usize, 2), degrees[3]);
    try testing.expectEqual(@as(usize, 3), degrees[4]);
}
