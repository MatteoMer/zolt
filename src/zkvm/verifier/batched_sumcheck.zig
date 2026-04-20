//! Batched Sumcheck Verifier
//!
//! Verifies multiple sumcheck instances batched together using
//! "front-loaded" staggered rounds (Jim Posen's technique).
//!
//! When instances have different numbers of variables (rounds), shorter
//! instances are padded: their claim is scaled by 2^(max_rounds - num_rounds)
//! so that they contribute a constant polynomial during the extra rounds.
//!
//! Mirrors Rust's `BatchedSumcheck::verify` from sumcheck.rs.

const std = @import("std");
const Allocator = std.mem.Allocator;

const zolt_arith = @import("zolt_arith");
const jolt_types = @import("../jolt_types.zig");
const sumcheck_verifier = @import("sumcheck_verifier.zig");
const opening_accumulator_mod = @import("opening_accumulator.zig");

/// Result of batched sumcheck verification.
pub fn BatchedSumcheckResult(comptime F: type) type {
    return struct {
        /// Random linear combination coefficients (one per instance).
        batching_coeffs: []F,
        /// All sumcheck challenges (length = max_num_rounds).
        challenges: []F,
        /// The final output claim from the sumcheck.
        output_claim: F,
        allocator: Allocator,

        pub fn deinit(self: *@This()) void {
            self.allocator.free(self.batching_coeffs);
            self.allocator.free(self.challenges);
        }
    };
}

/// Interface for a sumcheck instance participating in a batch.
///
/// Implementors provide their per-instance parameters and claim logic.
/// Use `SumcheckInstance` to wrap a struct that has these methods.
pub fn SumcheckInstance(comptime F: type) type {
    return struct {
        ptr: *const anyopaque,
        vtable: *const VTable,

        const Accumulator = opening_accumulator_mod.VerifierOpeningAccumulator(F);

        const VTable = struct {
            /// Maximum polynomial degree for this instance.
            degree: *const fn (self: *const anyopaque) usize,
            /// Number of sumcheck rounds (= number of variables).
            numRounds: *const fn (self: *const anyopaque) usize,
            /// Initial claim for this instance (may read from accumulator).
            inputClaim: *const fn (self: *const anyopaque, acc: *const Accumulator) F,
            /// Expected output claim given the sumcheck challenges.
            /// `r_slice` is the instance-local subset of challenges.
            expectedOutputClaim: *const fn (self: *const anyopaque, acc: *const Accumulator, r_slice: []const F) F,
            /// Cache opening claims in the accumulator at the sumcheck point.
            cacheOpenings: *const fn (self: *const anyopaque, acc: *Accumulator, r_slice: []const F) void,
        };

        pub fn degree(self: @This()) usize {
            return self.vtable.degree(self.ptr);
        }

        pub fn numRounds(self: @This()) usize {
            return self.vtable.numRounds(self.ptr);
        }

        pub fn inputClaim(self: @This(), acc: *const Accumulator) F {
            return self.vtable.inputClaim(self.ptr, acc);
        }

        pub fn expectedOutputClaim(self: @This(), acc: *const Accumulator, r_slice: []const F) F {
            return self.vtable.expectedOutputClaim(self.ptr, acc, r_slice);
        }

        pub fn cacheOpenings(self: @This(), acc: *Accumulator, r_slice: []const F) void {
            return self.vtable.cacheOpenings(self.ptr, acc, r_slice);
        }

        /// Create a SumcheckInstance from any struct T that implements the required methods.
        pub fn from(comptime T: type, ptr: *const T) @This() {
            const gen = struct {
                fn degree(self_ptr: *const anyopaque) usize {
                    const self: *const T = @ptrCast(@alignCast(self_ptr));
                    return self.degree();
                }
                fn numRounds(self_ptr: *const anyopaque) usize {
                    const self: *const T = @ptrCast(@alignCast(self_ptr));
                    return self.numRounds();
                }
                fn inputClaim(self_ptr: *const anyopaque, acc: *const Accumulator) F {
                    const self: *const T = @ptrCast(@alignCast(self_ptr));
                    return self.inputClaim(acc);
                }
                fn expectedOutputClaim(self_ptr: *const anyopaque, acc: *const Accumulator, r_slice: []const F) F {
                    const self: *const T = @ptrCast(@alignCast(self_ptr));
                    return self.expectedOutputClaim(acc, r_slice);
                }
                fn cacheOpenings(self_ptr: *const anyopaque, acc: *Accumulator, r_slice: []const F) void {
                    const self: *const T = @ptrCast(@alignCast(self_ptr));
                    self.cacheOpenings(acc, r_slice);
                }
            };

            return .{
                .ptr = ptr,
                .vtable = &.{
                    .degree = gen.degree,
                    .numRounds = gen.numRounds,
                    .inputClaim = gen.inputClaim,
                    .expectedOutputClaim = gen.expectedOutputClaim,
                    .cacheOpenings = gen.cacheOpenings,
                },
            };
        }
    };
}

/// Verify a batched sumcheck proof with multiple instances.
///
/// 1. Collects max_degree and max_num_rounds across all instances.
/// 2. Appends each instance's input_claim to transcript.
/// 3. Samples batching_coeffs via transcript.challengeVector().
/// 4. Computes batched initial claim (scaling shorter instances by 2^(max_rounds - num_rounds)).
/// 5. Runs core sumcheck verification.
/// 6. Checks expected output claims.
/// 7. Calls cache_openings on each instance.
///
/// Returns batching coefficients and sumcheck challenges.
pub fn verifyBatched(
    comptime F: type,
    instances: []const SumcheckInstance(F),
    proof: *const jolt_types.SumcheckInstanceProof(F),
    transcript: anytype,
    accumulator: *opening_accumulator_mod.VerifierOpeningAccumulator(F),
    allocator: Allocator,
) !BatchedSumcheckResult(F) {
    const num_instances = instances.len;
    if (num_instances == 0) return error.InvalidProof;

    // 1. Compute max_degree and max_num_rounds across all instances
    var max_degree: usize = 0;
    var max_num_rounds: usize = 0;
    for (instances) |inst| {
        max_degree = @max(max_degree, inst.degree());
        max_num_rounds = @max(max_num_rounds, inst.numRounds());
    }

    // 2. Collect input claims from each instance and append to transcript
    var input_claims = try allocator.alloc(F, num_instances);
    defer allocator.free(input_claims);

    for (instances, 0..) |inst, i| {
        input_claims[i] = inst.inputClaim(accumulator);
    }

    // Append each claim individually to transcript (must match prover's per-claim pattern)
    for (input_claims) |claim| {
        transcript.appendScalar("sumcheck_claim", claim);
    }

    // 3. Sample batching coefficients (one challengeScalarFull per instance, matching prover)
    var batching_coeffs = try allocator.alloc(F, num_instances);
    errdefer allocator.free(batching_coeffs);
    for (0..num_instances) |i| {
        batching_coeffs[i] = transcript.challengeScalarFull();
    }

    // 4. Compute batched initial claim
    // For each instance: batching_coeff[i] * input_claim[i] * 2^(max_rounds - num_rounds[i])
    var batched_claim = F.zero();
    for (instances, 0..) |inst, i| {
        var scaled_claim = input_claims[i];
        const round_diff = max_num_rounds - inst.numRounds();
        // Multiply by 2^round_diff (the padding factor for staggered rounds)
        const scale = F.fromU64(@as(u64, 1) << @intCast(round_diff));
        scaled_claim = scaled_claim.mul(scale);
        batched_claim = batched_claim.add(batching_coeffs[i].mul(scaled_claim));
    }

    // 5. Run core sumcheck verification
    const sc_result = try sumcheck_verifier.verifySumcheck(
        F,
        proof,
        batched_claim,
        max_num_rounds,
        max_degree,
        transcript,
        allocator,
    );
    errdefer allocator.free(sc_result.challenges);

    // 6. For each instance: cache openings and compute expected output claim
    var expected_output = F.zero();
    for (instances, 0..) |inst, i| {
        const round_offset = max_num_rounds - inst.numRounds();
        const r_slice = sc_result.challenges[round_offset .. round_offset + inst.numRounds()];

        // Cache polynomial opening claims
        inst.cacheOpenings(accumulator, r_slice);

        // Compute expected output claim
        const expected = inst.expectedOutputClaim(accumulator, r_slice);
        expected_output = expected_output.add(batching_coeffs[i].mul(expected));
    }

    // 7. Verify: batched expected output == sumcheck output claim
    if (!expected_output.eql(sc_result.output_claim)) {
        allocator.free(sc_result.challenges);
        allocator.free(batching_coeffs);
        return error.SumcheckFailed;
    }

    return BatchedSumcheckResult(F){
        .batching_coeffs = batching_coeffs,
        .challenges = sc_result.challenges,
        .output_claim = sc_result.output_claim,
        .allocator = allocator,
    };
}

// =============================================================================
// Tests
// =============================================================================

const testing = std.testing;
const BN254Scalar = zolt_arith.field.BN254Scalar;
const Blake2bTranscript = zolt_arith.transcripts.Blake2bTranscript;
const Interp = zolt_arith.poly.interpolation.Interpolation(BN254Scalar);

/// Test instance: a simple sumcheck with known claim and degree.
const TestInstance = struct {
    num_rounds_val: usize,
    degree_val: usize,
    input_claim_val: BN254Scalar,
    expected_output_val: BN254Scalar,

    pub fn degree(self: *const TestInstance) usize {
        return self.degree_val;
    }

    pub fn numRounds(self: *const TestInstance) usize {
        return self.num_rounds_val;
    }

    pub fn inputClaim(self: *const TestInstance, _: *const opening_accumulator_mod.VerifierOpeningAccumulator(BN254Scalar)) BN254Scalar {
        return self.input_claim_val;
    }

    pub fn expectedOutputClaim(self: *const TestInstance, _: *const opening_accumulator_mod.VerifierOpeningAccumulator(BN254Scalar), _: []const BN254Scalar) BN254Scalar {
        return self.expected_output_val;
    }

    pub fn cacheOpenings(self: *const TestInstance, _: *opening_accumulator_mod.VerifierOpeningAccumulator(BN254Scalar), _: []const BN254Scalar) void {
        _ = self;
    }
};

test "batched sumcheck: single instance matches plain sumcheck" {
    const F_ = BN254Scalar;

    // Create a degree-3, 1-round proof: p(x) = 5 + 3x + 2x^2 + x^3
    // claim = p(0) + p(1) = 5 + 11 = 16
    const full_coeffs = [_]F_{ F_.fromU64(5), F_.fromU64(3), F_.fromU64(2), F_.fromU64(1) };
    var proof = jolt_types.SumcheckInstanceProof(F_).init(testing.allocator);
    defer proof.deinit();
    try proof.addRoundPoly(&full_coeffs);

    // Simulate: run with a prover transcript to find what the expected output claim will be.
    // The batched verifier derives batching_coeffs from transcript, so we need to
    // replicate the transcript to compute the expected output.
    // For a single instance, the output claim = p(challenge).
    // We'll set expected_output_val after computing what the verifier will derive.

    // First, compute what transcript will produce:
    var sim_transcript = Blake2bTranscript(F_).init("batch_test");
    const claim_val = F_.fromU64(16);
    // Per-claim append (matching prover pattern)
    sim_transcript.appendScalar("sumcheck_claim", claim_val);
    // Per-coeff challengeScalarFull (matching prover pattern)
    const batching_coeff = sim_transcript.challengeScalarFull();

    // The batched claim = coeff * claim * 2^0 = coeff * claim
    const batched_claim = batching_coeff.mul(claim_val);

    // Now simulate sumcheck round: append compressed, derive challenge
    const compressed = [_]F_{ F_.fromU64(5), F_.fromU64(2), F_.fromU64(1) };
    sim_transcript.appendScalars("sumcheck_poly", &compressed);
    const challenge = sim_transcript.challengeScalar();
    const output_claim = Interp.evalFromHintGeneral(&compressed, batched_claim, challenge);

    // expected_output = batching_coeff * instance.expectedOutputClaim
    // So instance.expectedOutputClaim = output_claim / batching_coeff
    const expected_instance_output = output_claim.mul(batching_coeff.inverse().?);

    var instance = TestInstance{
        .num_rounds_val = 1,
        .degree_val = 3,
        .input_claim_val = claim_val,
        .expected_output_val = expected_instance_output,
    };

    var instances = [_]SumcheckInstance(F_){
        SumcheckInstance(F_).from(TestInstance, &instance),
    };

    var acc = opening_accumulator_mod.VerifierOpeningAccumulator(F_).init(testing.allocator);
    defer acc.deinit();

    var transcript = Blake2bTranscript(F_).init("batch_test");
    var result = try verifyBatched(
        F_,
        &instances,
        &proof,
        &transcript,
        &acc,
        testing.allocator,
    );
    defer result.deinit();

    try testing.expectEqual(@as(usize, 1), result.challenges.len);
    try testing.expectEqual(@as(usize, 1), result.batching_coeffs.len);
    try testing.expect(result.challenges[0].eql(challenge));
}
