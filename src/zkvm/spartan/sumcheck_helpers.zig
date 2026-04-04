//! Shared helper functions for batched sumcheck provers.
//!
//! These utilities are used across stage3, stage5, and stage6 provers to
//! avoid duplicating common arithmetic patterns (gamma power derivation,
//! inactive-instance scaling, polynomial extrapolation, and Newton forward
//! difference compression).

const std = @import("std");
const Allocator = std.mem.Allocator;

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

test "inactiveContribution doubles correctly" {
    const F = @import("zolt_arith").field.BN254Scalar;
    // remaining_rounds=5, num_rounds=2 => scale=2, so claim * 4
    const claim = F.fromU64(7);
    const result = inactiveContribution(F, claim, 5, 2);
    try std.testing.expectEqual(F.fromU64(28), result);
}
