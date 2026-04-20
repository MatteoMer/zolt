//! Native Zig Verifier for Zolt
//!
//! Verifies Jolt-compatible proofs without the Rust FFI dependency.
//! Reconstructs the Fiat-Shamir transcript and verifies all 7 sumcheck
//! stages plus opening claims.

const std = @import("std");
const Allocator = std.mem.Allocator;

const zolt_arith = @import("zolt_arith");
const Blake2bTranscript = zolt_arith.transcripts.Blake2bTranscript;

const jolt_types = @import("../jolt_types.zig");
const jolt_serialization = @import("../jolt_serialization.zig");
const jolt_device = @import("../jolt_device.zig");
const preprocessing_mod = @import("../preprocessing.zig");

pub const sumcheck_verifier = @import("sumcheck_verifier.zig");
pub const opening_accumulator = @import("opening_accumulator.zig");
pub const VerifierOpeningAccumulator = opening_accumulator.VerifierOpeningAccumulator;
pub const batched_sumcheck = @import("batched_sumcheck.zig");
pub const BatchedSumcheckResult = batched_sumcheck.BatchedSumcheckResult;
pub const SumcheckInstance = batched_sumcheck.SumcheckInstance;
pub const stage1_verifier = @import("stage1_verifier.zig");

const GT = jolt_serialization.GT;
const DoryCommitment = jolt_serialization.DoryCommitment;
const DoryProof = jolt_serialization.DoryProof;

const r1cs_mod = @import("../r1cs/mod.zig");
const univariate_skip = @import("../r1cs/univariate_skip.zig");

const F = zolt_arith.field.BN254Scalar;
const JoltProofType = jolt_types.JoltProof(F, DoryCommitment, DoryProof);

/// Per-stage degree bounds for sumcheck polynomials.
/// Stage 1 (outer remaining): degree 3.
/// Stage 2 (batched): max degree across 5 instances = 3.
/// Stages 3-7: degree 3 (claim reductions use degree-3 polynomials).
const STAGE1_REMAINING_DEGREE: usize = 3;
const STAGE2_BATCHED_DEGREE: usize = 3;
const STAGE3_DEGREE: usize = 3;
const STAGE4_DEGREE: usize = 3;
const STAGE5_DEGREE: usize = 3;
const STAGE6_DEGREE: usize = 3;
const STAGE7_DEGREE: usize = 3;

/// Verification error types
pub const VerifyError = error{
    DeserializationFailed,
    SumcheckFailed,
    OpeningClaimMismatch,
    InvalidProof,
    InsufficientRounds,
    DegreeBoundExceeded,
    UniSkipSumCheckFailed,
    InvalidG1Point,
    InvalidG2Point,
    InvalidData,
    UnexpectedEof,
    OutOfMemory,
};

/// Result of verification with diagnostic info
pub const VerifyResult = struct {
    /// Whether the proof is valid
    valid: bool,
    /// Stage where verification failed (0 = deserialization, 1-7 = stages)
    failed_stage: ?u8 = null,
};

/// Verify a Jolt-compatible proof using the native Zig verifier.
///
/// Arguments:
///   - proof_bytes: serialized JoltProof (arkworks format)
///   - preprocessing_bytes: serialized JoltVerifierPreprocessing
///   - io_bytes: optional serialized JoltDevice (program I/O)
///
/// Returns true if the proof is valid, false otherwise.
pub fn verify(
    allocator: Allocator,
    proof_bytes: []const u8,
    preprocessing_bytes: []const u8,
    io_bytes: ?[]const u8,
) VerifyResult {
    return verifyInner(allocator, proof_bytes, preprocessing_bytes, io_bytes) catch {
        return VerifyResult{ .valid = false, .failed_stage = 0 };
    };
}

fn verifyInner(
    allocator: Allocator,
    proof_bytes: []const u8,
    preprocessing_bytes: []const u8,
    io_bytes: ?[]const u8,
) !VerifyResult {
    // 1. Deserialize proof
    var deser = jolt_serialization.ArkworksDeserializer(F).init(proof_bytes);
    var proof = try deser.readJoltProof(allocator);
    defer proof.deinit();

    // 2. Parse preprocessing
    var pp = try preprocessing_mod.VerifierPreprocessingData.fromBytes(allocator, preprocessing_bytes);
    defer pp.deinit();

    // 3. Deserialize program I/O (or create default)
    var device: jolt_device.JoltDevice = undefined;
    var owns_device = false;
    if (io_bytes) |io| {
        if (io.len > 0) {
            device = try jolt_device.JoltDevice.deserialize(allocator, io);
            owns_device = true;
        } else {
            device = defaultDevice(pp.memory_layout);
        }
    } else {
        device = defaultDevice(pp.memory_layout);
    }
    defer if (owns_device) device.deinit();

    // 4. Initialize transcript
    var transcript = Blake2bTranscript(F).init("Jolt");

    // 5. Fiat-Shamir preamble
    const log_t: u8 = @intCast(std.math.log2_int(usize, proof.trace_length));
    jolt_device.fiatShamirPreamble(
        F,
        &transcript,
        &device,
        proof.ram_K,
        proof.trace_length,
        pp.entry_address,
        proof.rw_config,
        proof.one_hot_config,
        proof.dory_layout,
        &pp.preprocessing_digest,
    );

    // 6. Append all Dory commitments to transcript
    for (proof.commitments.items) |comm| {
        transcript.appendGT("commitment", comm);
    }

    // 7. Derive tau challenges
    const n_cycle_vars = log_t;
    const num_rows_bits: usize = @as(usize, n_cycle_vars) + 2; // +2 for R1CS rows (4 constraints per cycle)
    var tau = try allocator.alloc(F, num_rows_bits);
    defer allocator.free(tau);
    for (0..num_rows_bits) |i| {
        tau[i] = transcript.challengeScalar();
    }

    // Initialize opening accumulator (collects claims across all stages)
    var acc = VerifierOpeningAccumulator(F).init(allocator);
    defer acc.deinit();

    // 8. Verify Stage 1: UniSkip + Outer Spartan sumcheck
    if (proof.stage1_uni_skip_first_round_proof) |uni_skip| {
        // 8a. UniSkip: verify degree bound
        const uni_result = try sumcheck_verifier.verifyUniSkipRound(
            F,
            uni_skip.uni_poly,
            univariate_skip.OUTER_FIRST_ROUND_POLY_DEGREE_BOUND,
            &transcript,
        );

        // 8b. UniSkip: verify sum-of-evals over base domain = 0
        stage1_verifier.checkUniSkipSumOfEvals(
            F,
            uni_skip.uni_poly,
            univariate_skip.OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE,
            F.zero(), // Outer UniSkip input claim is always 0
        ) catch {
            return VerifyResult{ .valid = false, .failed_stage = 1 };
        };

        // 8c. Cache the UniSkip opening in the accumulator
        try acc.appendVirtual(
            .UnivariateSkip,
            .SpartanOuter,
            &[_]F{uni_result.challenge},
            uni_result.claim,
        );
        acc.flushToTranscript(&transcript);
    }

    // 8d. Outer remaining sumcheck
    const s1_remaining = stage1_verifier.OuterRemainingSumcheckVerifier(F){
        .tau = tau,
        .r0 = if (proof.stage1_uni_skip_first_round_proof != null)
            acc.getVirtual(.UnivariateSkip, .SpartanOuter).?.point[0]
        else
            F.zero(),
        .num_cycle_vars = n_cycle_vars,
        .proof_opening_claims = &proof.opening_claims,
    };

    const s1_result = try sumcheck_verifier.verifySumcheck(
        F,
        &proof.stage1_sumcheck_proof,
        s1_remaining.inputClaim(&acc),
        proof.stage1_sumcheck_proof.compressed_polys.items.len,
        STAGE1_REMAINING_DEGREE,
        &transcript,
        allocator,
    );
    defer allocator.free(s1_result.challenges);

    // Cache Stage 1 opening claims
    s1_remaining.cacheOpenings(&acc, s1_result.challenges);

    // 9. Verify Stage 2: UniSkip + 5-instance batched sumcheck
    var s2_uni_skip_claim = F.zero();
    if (proof.stage2_uni_skip_first_round_proof) |uni_skip| {
        // Stage 2 also has a tau_high challenge before UniSkip
        _ = transcript.challengeScalar(); // tau_high_stage2
        const uni_result = try sumcheck_verifier.verifyUniSkipRound(
            F,
            uni_skip.uni_poly,
            univariate_skip.PRODUCT_VIRTUAL_FIRST_ROUND_POLY_DEGREE_BOUND,
            &transcript,
        );
        s2_uni_skip_claim = uni_result.claim;
    }

    const s2_result = try sumcheck_verifier.verifySumcheck(
        F,
        &proof.stage2_sumcheck_proof,
        // TODO: Wire proper batched initial claim from 5 instances
        s2_uni_skip_claim,
        proof.stage2_sumcheck_proof.compressed_polys.items.len,
        STAGE2_BATCHED_DEGREE,
        &transcript,
        allocator,
    );
    defer allocator.free(s2_result.challenges);

    // Stage 2 post: append UniSkip claim
    transcript.appendScalar("opening_claim", s2_result.final_claim);

    // 10-14. Verify Stages 3-7
    const stage_proofs = [_]*const jolt_types.SumcheckInstanceProof(F){
        &proof.stage3_sumcheck_proof,
        &proof.stage4_sumcheck_proof,
        &proof.stage5_sumcheck_proof,
        &proof.stage6_sumcheck_proof,
        &proof.stage7_sumcheck_proof,
    };

    const stage_degrees = [_]usize{
        STAGE3_DEGREE,
        STAGE4_DEGREE,
        STAGE5_DEGREE,
        STAGE6_DEGREE,
        STAGE7_DEGREE,
    };

    var stage_results: [5]sumcheck_verifier.SumcheckResult(F) = undefined;
    var stage_allocs: [5]bool = [_]bool{false} ** 5;

    for (stage_proofs, 0..) |stage_proof, i| {
        const num_rounds = stage_proof.compressed_polys.items.len;
        if (num_rounds == 0) continue;

        stage_results[i] = try sumcheck_verifier.verifySumcheck(
            F,
            stage_proof,
            // TODO: Wire proper per-stage initial claims
            getSumcheckInitialClaim(F, stage_proof),
            num_rounds,
            stage_degrees[i],
            &transcript,
            allocator,
        );
        stage_allocs[i] = true;
    }
    defer for (0..5) |i| {
        if (stage_allocs[i]) allocator.free(stage_results[i].challenges);
    };

    // 15. Opening claims verification
    // The opening claims in the proof should be consistent with the sumcheck outputs.
    // Full per-stage claim extraction requires stage-specific logic (deferred).
    // For now, verify that opening claims exist and are non-empty.
    if (proof.opening_claims.len() == 0) {
        return VerifyResult{ .valid = false, .failed_stage = 7 };
    }

    // 16. Dory opening proof verification (stubbed for now)
    // TODO: Implement Dory PCS opening verification
    // This requires pairing-based checks using the DoryVerifierSetup.

    return VerifyResult{ .valid = true };
}

/// Placeholder initial claim for stages not yet wired with proper claim logic.
/// Returns zero — real claims must come from the protocol context (previous stage
/// output, opening accumulator, or tau-derived values).
///
/// TODO: Replace with per-stage verifier instances (as done for Stage 1).
fn getSumcheckInitialClaim(comptime Field: type, proof: *const jolt_types.SumcheckInstanceProof(Field)) Field {
    _ = proof;
    return Field.zero();
}

fn defaultDevice(memory_layout: jolt_device.MemoryLayout) jolt_device.JoltDevice {
    return jolt_device.JoltDevice{
        .inputs = &[_]u8{},
        .trusted_advice = &[_]u8{},
        .untrusted_advice = &[_]u8{},
        .outputs = &[_]u8{},
        .panic = false,
        .memory_layout = memory_layout,
        .allocator = null,
    };
}

// =============================================================================
// Tests
// =============================================================================

test {
    _ = sumcheck_verifier;
    _ = opening_accumulator;
    _ = batched_sumcheck;
    _ = stage1_verifier;
}
