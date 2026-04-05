//! Shared helper functions for batched sumcheck provers.
//!
//! These utilities are used across stage3, stage5, and stage6 provers to
//! avoid duplicating common arithmetic patterns (gamma power derivation,
//! inactive-instance scaling, polynomial extrapolation, and Newton forward
//! difference compression).

const std = @import("std");
const Allocator = std.mem.Allocator;
const poly_mod = @import("zolt_arith").poly;

/// Re-export from stage6_helpers for cross-stage use.
pub const addEvalsAsMonomialToCoeffs = @import("stage6_helpers.zig").addEvalsAsMonomialToCoeffs;

/// Compute contribution of an inactive instance to the combined polynomial.
/// Used when remaining_rounds > instance_num_rounds.
/// Returns: input_claim * 2^(remaining_rounds - num_rounds - 1)
pub fn inactiveContribution(comptime F: type, input_claim: F, remaining_rounds: usize, num_rounds: usize) F {
    const scale = remaining_rounds - num_rounds - 1;
    var scaled = input_claim;
    for (0..scale) |_| scaled = scaled.add(scaled);
    return scaled;
}

/// Extrapolate a degree-2 polynomial evaluated at [p(0), p(1), p(2)] to get p(3).
/// Uses quadratic extrapolation: p(3) = 3*p(2) - 3*p(1) + p(0)
pub fn extrapolateDeg2(comptime F: type, evals: [3]F) F {
    return evals[2].mul(F.fromU64(3)).sub(evals[1].mul(F.fromU64(3))).add(evals[0]);
}

/// Compress 4 evaluations [p(0), p(1), p(2), p(3)] to [c0, c2, c3] via Newton forward differences.
/// The linear coefficient c1 is omitted (recoverable from the claim via hint).
pub fn finiteDifferencesCompress(comptime F: type, evals: [4]F) [3]F {
    const inv2 = comptime blk: {
        @setEvalBranchQuota(1000000);
        break :blk F.fromU64(2).inverse().?;
    };
    const inv6 = comptime blk: {
        @setEvalBranchQuota(1000000);
        break :blk F.fromU64(6).inverse().?;
    };
    const d1 = evals[1].sub(evals[0]);
    const d2 = evals[2].sub(evals[1]);
    const d3 = evals[3].sub(evals[2]);
    const dd1 = d2.sub(d1);
    const dd2 = d3.sub(d2);
    const c3 = dd2.sub(dd1).mul(inv6);
    const c2 = dd1.mul(inv2).sub(c3.mul(F.fromU64(3)));
    return .{ evals[0], c2, c3 };
}

/// Derive gamma powers: [1, gamma, gamma^2, ..., gamma^(n-1)]
pub fn deriveGammaPowers(comptime F: type, allocator: Allocator, gamma: F, n: usize) ![]F {
    const powers = try allocator.alloc(F, n);
    powers[0] = F.one();
    if (n > 1) {
        powers[1] = gamma;
        for (2..n) |i| powers[i] = powers[i - 1].mul(gamma);
    }
    return powers;
}

/// Extrapolate a multilinear polynomial from evaluations at x=0 and x=1 to x=2.
/// f(2) = 2*f(1) - f(0)
pub inline fn extrapolateLinearTo2(comptime F: type, f0: F, f1: F) F {
    return f1.add(f1).sub(f0);
}

/// Derive N batching coefficients from transcript as a comptime-sized stack array.
pub fn deriveBatchingCoeffs(comptime F: type, comptime N: usize, transcript: anytype) [N]F {
    var coeffs: [N]F = undefined;
    for (0..N) |i| coeffs[i] = transcript.challengeScalarFull();
    return coeffs;
}

/// Add an inactive instance's constant-polynomial contribution to combined monomial coefficients.
/// When remaining_rounds > instance_num_rounds, the instance contributes a constant polynomial
/// scaled by 2^(remaining - num_rounds - 1), which only affects the degree-0 coefficient.
pub fn addInactiveToMonomial(comptime F: type, combined_coeffs: []F, input_claim: F, batch_coeff: F, remaining_rounds: usize, num_rounds: usize) void {
    const scaled = inactiveContribution(F, input_claim, remaining_rounds, num_rounds);
    combined_coeffs[0] = combined_coeffs[0].add(batch_coeff.mul(scaled));
}

/// Finish a batched sumcheck round: compress monomial coefficients (strip c1),
/// append to proof and transcript, derive challenge, and evaluate combined poly.
///
/// This centralizes the correctness-critical transcript interaction that all
/// batched sumcheck stages share: the compress format, transcript label,
/// challenge derivation, and hint-based polynomial evaluation.
///
/// `combined_coeffs` must be monomial form [c0, c1, c2, ..., c_d] with len >= max_degree + 1.
/// Returns the challenge and updated batched claim.
pub fn finishSumcheckRound(
    comptime F: type,
    combined_coeffs: []const F,
    max_degree: usize,
    batched_claim: F,
    transcript: anytype,
    proof: anytype,
    allocator: Allocator,
) !struct { challenge: F, new_claim: F, compressed: []F } {
    const UniPoly = poly_mod.UniPoly(F);

    // 1. Compress: strip c1, keep [c0, c2, c3, ..., c_d]
    const compressed = try allocator.alloc(F, max_degree);
    compressed[0] = combined_coeffs[0]; // c0
    for (1..max_degree) |i| {
        compressed[i] = if (i + 1 < combined_coeffs.len) combined_coeffs[i + 1] else F.zero();
    }

    // 2. Append to proof
    try proof.compressed_polys.append(allocator, .{
        .coeffs_except_linear_term = compressed,
        .allocator = allocator,
    });

    // 3. Transcript: append compressed coefficients, derive challenge
    transcript.appendScalars("sumcheck_poly", compressed);
    const challenge = transcript.challengeScalar();

    // 4. Evaluate combined poly at challenge using hint (batched_claim = p(0) + p(1))
    const new_claim = UniPoly.evalFromHintGeneral(compressed, batched_claim, challenge);

    return .{ .challenge = challenge, .new_claim = new_claim, .compressed = compressed };
}

test "extrapolateDeg2 matches known values" {
    const F = @import("zolt_arith").field.BN254Scalar;
    // p(x) = x^2  =>  p(0)=0, p(1)=1, p(2)=4, p(3)=9
    const evals = [3]F{ F.zero(), F.one(), F.fromU64(4) };
    const p3 = extrapolateDeg2(F, evals);
    try std.testing.expectEqual(F.fromU64(9), p3);
}

test "finiteDifferencesCompress round-trip" {
    const F = @import("zolt_arith").field.BN254Scalar;
    // p(x) = 2 + 3x + 5x^2 + 7x^3
    // p(0) = 2, p(1) = 17, p(2) = 84, p(3) = 245
    const evals = [4]F{ F.fromU64(2), F.fromU64(17), F.fromU64(84), F.fromU64(245) };
    const compressed = finiteDifferencesCompress(F, evals);
    // c0 = p(0) = 2
    try std.testing.expectEqual(F.fromU64(2), compressed[0]);
    // c3 should be 7 (leading coefficient)
    try std.testing.expectEqual(F.fromU64(7), compressed[2]);
    // c2 should be 5 - 3*7 = 5 - 21 ... actually let's compute properly:
    // Newton forward differences for cubic:
    // d1 = 15, d2 = 67, d3 = 161
    // dd1 = 52, dd2 = 94
    // c3 = (94 - 52)/6 = 42/6 = 7 ✓
    // c2 = 52/2 - 7*3 = 26 - 21 = 5 ✓
    try std.testing.expectEqual(F.fromU64(5), compressed[1]);
}

test "deriveGammaPowers produces correct powers" {
    const F = @import("zolt_arith").field.BN254Scalar;
    const allocator = std.testing.allocator;
    const gamma = F.fromU64(5);
    const powers = try deriveGammaPowers(F, allocator, gamma, 4);
    defer allocator.free(powers);
    try std.testing.expectEqual(F.one(), powers[0]);
    try std.testing.expectEqual(F.fromU64(5), powers[1]);
    try std.testing.expectEqual(F.fromU64(25), powers[2]);
    try std.testing.expectEqual(F.fromU64(125), powers[3]);
}

test "extrapolateLinearTo2 computes 2*f1 - f0" {
    const F = @import("zolt_arith").field.BN254Scalar;
    // f(0) = 3, f(1) = 7 => f(2) = 2*7 - 3 = 11
    const result = extrapolateLinearTo2(F, F.fromU64(3), F.fromU64(7));
    try std.testing.expectEqual(F.fromU64(11), result);
}

test "inactiveContribution doubles correctly" {
    const F = @import("zolt_arith").field.BN254Scalar;
    // remaining_rounds=5, num_rounds=2 => scale=2, so claim * 4
    const claim = F.fromU64(7);
    const result = inactiveContribution(F, claim, 5, 2);
    try std.testing.expectEqual(F.fromU64(28), result);
}
