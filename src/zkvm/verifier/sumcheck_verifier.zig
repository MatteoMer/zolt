//! Sumcheck Verification
//!
//! Core sumcheck round verification for the native Zig verifier.
//! Verifies compressed univariate polynomials sent by the prover,
//! checking that p(0) + p(1) = claim at each round.

const std = @import("std");
const Allocator = std.mem.Allocator;

const zolt_arith = @import("zolt_arith");
const jolt_types = @import("../jolt_types.zig");

/// Sumcheck verification result
pub fn SumcheckResult(comptime F: type) type {
    return struct {
        /// Final claim after all rounds
        final_claim: F,
        /// Challenge vector (one per round), owned by caller
        challenges: []F,
    };
}

/// Verify a sumcheck instance proof against an initial claim.
///
/// For each round i:
///   1. Read compressed polynomial [c0, c2, c3, ..., c_d] from proof
///   2. Recover linear term: c1 = claim - 2*c0 - c2 - c3 - ... - c_d
///   3. Check: p(0) + p(1) == claim
///   4. Append compressed coefficients to transcript ("sumcheck_poly")
///   5. Derive challenge r_i via transcript.challengeScalar()
///   6. Evaluate p(r_i) → new claim
///
/// Returns the final claim and all challenge values.
pub fn verifySumcheck(
    comptime F: type,
    proof: *const jolt_types.SumcheckInstanceProof(F),
    initial_claim: F,
    num_rounds: usize,
    transcript: anytype,
    allocator: Allocator,
) !SumcheckResult(F) {
    const Interp = zolt_arith.poly.interpolation.Interpolation(F);

    if (proof.compressed_polys.items.len < num_rounds) {
        return error.InsufficientRounds;
    }

    var challenges = try allocator.alloc(F, num_rounds);
    errdefer allocator.free(challenges);

    var claim = initial_claim;

    for (0..num_rounds) |round| {
        const compressed = proof.compressed_polys.items[round].coeffs_except_linear_term;

        // Verify p(0) + p(1) = claim
        // p(0) = c0
        // p(1) = c0 + c1 + c2 + ... + c_d
        //       = c0 + (claim - 2*c0 - c2 - ... - c_d) + c2 + ... + c_d
        //       = claim - c0
        // So p(0) + p(1) = c0 + (claim - c0) = claim — always true by construction.
        // The real verification happens when we check that the final claim matches
        // the polynomial evaluation at the opening point.

        // Append compressed coefficients to transcript (same as prover)
        transcript.appendScalars("sumcheck_poly", compressed);

        // Derive challenge
        const challenge = transcript.challengeScalar();
        challenges[round] = challenge;

        // Evaluate p(challenge) using hint-based recovery
        claim = Interp.evalFromHintGeneral(compressed, claim, challenge);
    }

    return SumcheckResult(F){
        .final_claim = claim,
        .challenges = challenges,
    };
}

/// Verify a UniSkip first round (full polynomial, not compressed).
///
/// The UniSkip proof contains a full univariate polynomial.
/// The verifier appends it to the transcript, derives a challenge,
/// and evaluates the polynomial at that challenge.
///
/// Returns the evaluated claim and the challenge.
pub fn verifyUniSkipRound(
    comptime F: type,
    uni_poly: []const F,
    transcript: anytype,
) struct { claim: F, challenge: F } {
    // Append the full polynomial to transcript
    transcript.appendScalars("uniskip_poly", uni_poly);

    // Derive challenge
    const challenge = transcript.challengeScalar();

    // Evaluate using Horner's method
    var result = F.zero();
    var i = uni_poly.len;
    while (i > 0) {
        i -= 1;
        result = result.mul(challenge).add(uni_poly[i]);
    }

    return .{ .claim = result, .challenge = challenge };
}

// =============================================================================
// Tests
// =============================================================================

const testing = std.testing;
const BN254Scalar = zolt_arith.field.BN254Scalar;
const Blake2bTranscript = zolt_arith.transcripts.Blake2bTranscript;

test "sumcheck verifier: single round degree-3" {
    const F = BN254Scalar;
    const Interp = zolt_arith.poly.interpolation.Interpolation(F);

    // Simulate a prover: create a degree-3 polynomial and compress it
    // p(x) = 5 + 3x + 2x^2 + x^3
    // p(0) = 5, p(1) = 5+3+2+1 = 11
    // claim = p(0) + p(1) = 16
    const claim = F.fromU64(16);

    // Compressed form: [c0, c2, c3] = [5, 2, 1] (excludes c1=3)
    var coeffs = [_]F{ F.fromU64(5), F.fromU64(2), F.fromU64(1) };

    // Create proof with one round
    var proof = jolt_types.SumcheckInstanceProof(F).init(testing.allocator);
    defer proof.deinit();

    // addRoundPoly expects full coefficients [c0, c1, c2, c3]
    // but CompressedUniPoly.init strips c1, so we pass full:
    const full_coeffs = [_]F{ F.fromU64(5), F.fromU64(3), F.fromU64(2), F.fromU64(1) };
    try proof.addRoundPoly(&full_coeffs);

    // Verify with prover transcript
    var prover_transcript = Blake2bTranscript(F).init("test");
    prover_transcript.appendScalars("sumcheck_poly", &coeffs);
    const prover_challenge = prover_transcript.challengeScalar();
    const expected_eval = Interp.evalFromHintGeneral(&coeffs, claim, prover_challenge);

    // Verify with verifier (fresh transcript, same label)
    var verifier_transcript = Blake2bTranscript(F).init("test");
    const result = try verifySumcheck(
        F,
        &proof,
        claim,
        1,
        &verifier_transcript,
        testing.allocator,
    );
    defer testing.allocator.free(result.challenges);

    // Challenge and final claim must match
    try testing.expect(result.challenges[0].eql(prover_challenge));
    try testing.expect(result.final_claim.eql(expected_eval));
}

test "sumcheck verifier: multi-round" {
    const F = BN254Scalar;

    // Two rounds with degree-2 polynomials (compressed = [c0, c2])
    // Round 1: p(x) = 10 + 4x + 2x^2
    //   p(0) = 10, p(1) = 16, claim = 26
    //   compressed = [10, 2]
    const claim1 = F.fromU64(26);
    const full1 = [_]F{ F.fromU64(10), F.fromU64(4), F.fromU64(2) };

    // Create transcript to derive challenge for round 1
    var sim_transcript = Blake2bTranscript(F).init("multi");
    sim_transcript.appendScalars("sumcheck_poly", &[_]F{ F.fromU64(10), F.fromU64(2) });
    const r1 = sim_transcript.challengeScalar();

    // Round 2: claim2 = p1(r1) (propagated from round 1)
    // We don't need to compute it explicitly — the verifier does it internally.

    // Round 2: p(x) = 3 + x + 5x^2
    // p(0) = 3, p(1) = 9
    // claim2 should equal p(0) + p(1) = 12... but claim2 comes from round 1 eval.
    // For a valid sumcheck, we need p2(0) + p2(1) = claim2.
    // So: c0_2 = 3, p2(1) = claim2 - 3 = claim2 - c0_2
    // Let's just set c0_2 = claim2/2 to make p(0)+p(1) = claim2 trivially.
    // Actually, let's construct: c0 = some, c2 = some, such that c0 + (claim2 - c0) = claim2. Always true.
    const full2 = [_]F{ F.fromU64(3), F.fromU64(1), F.fromU64(5) };

    // Build proof
    var proof = jolt_types.SumcheckInstanceProof(F).init(testing.allocator);
    defer proof.deinit();
    try proof.addRoundPoly(&full1);
    try proof.addRoundPoly(&full2);

    // Verify
    var verifier_transcript = Blake2bTranscript(F).init("multi");
    const result = try verifySumcheck(F, &proof, claim1, 2, &verifier_transcript, testing.allocator);
    defer testing.allocator.free(result.challenges);

    try testing.expectEqual(@as(usize, 2), result.challenges.len);
    // First challenge should match what we computed
    try testing.expect(result.challenges[0].eql(r1));
}

test "uni skip round verification" {
    const F = BN254Scalar;

    // Polynomial: p(x) = 1 + 2x + 3x^2
    const poly = [_]F{ F.fromU64(1), F.fromU64(2), F.fromU64(3) };

    var transcript = Blake2bTranscript(F).init("skip");
    const result = verifyUniSkipRound(F, &poly, &transcript);

    // Manually compute: append, derive challenge, evaluate
    var expected_transcript = Blake2bTranscript(F).init("skip");
    expected_transcript.appendScalars("uniskip_poly", &poly);
    const expected_challenge = expected_transcript.challengeScalar();

    // Horner: 1 + 2*r + 3*r^2
    const expected_eval = F.fromU64(1)
        .add(F.fromU64(2).mul(expected_challenge))
        .add(F.fromU64(3).mul(expected_challenge.mul(expected_challenge)));

    try testing.expect(result.challenge.eql(expected_challenge));
    try testing.expect(result.claim.eql(expected_eval));
}
