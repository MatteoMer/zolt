//! Integration Tests for Lasso Lookup Arguments
//!
//! These tests verify that the Lasso prover and verifier work correctly
//! together in an end-to-end fashion.

const std = @import("std");
const Allocator = std.mem.Allocator;

const prover = @import("prover.zig");
const verifier = @import("verifier.zig");
const expanding_table = @import("expanding_table.zig");
const split_eq = @import("split_eq.zig");
const poly = @import("../../poly/mod.zig");

const LassoProver = prover.LassoProver;
const LassoParams = prover.LassoParams;
const LassoProof = prover.LassoProof;
const LassoVerifier = verifier.LassoVerifier;
const ExpandingTable = expanding_table.ExpandingTable;
const SplitEqPolynomial = split_eq.SplitEqPolynomial;

/// Helper to run a complete Lasso protocol between prover and verifier
fn runLassoProtocol(
    comptime F: type,
    allocator: Allocator,
    lookup_indices: []const u128,
    lookup_tables: []const usize,
    params: LassoParams(F),
    initial_claim: F,
) !bool {
    var lasso_prover = try LassoProver(F).init(
        allocator,
        lookup_indices,
        lookup_tables,
        params,
    );
    defer lasso_prover.deinit();

    var lasso_verifier = try LassoVerifier(F).init(
        allocator,
        params,
        initial_claim,
    );
    defer lasso_verifier.deinit();

    // Run the protocol
    var round: usize = 0;
    while (!lasso_prover.isComplete()) : (round += 1) {
        // Prover computes round polynomial
        var round_poly = try lasso_prover.computeRoundPolynomial();
        defer round_poly.deinit();

        // Verifier checks and gets challenge
        const challenge = lasso_verifier.verifyRound(round_poly) catch |err| {
            std.debug.print("Verification failed at round {}: {}\n", .{ round, err });
            return false;
        };

        // Prover receives challenge
        try lasso_prover.receiveChallenge(challenge);
    }

    // Check final evaluation
    const final_eval = lasso_prover.getFinalEval();
    return lasso_verifier.verifyFinalEval(final_eval) catch false;
}

test "lasso end-to-end simple case" {
    const F = @import("../../field/mod.zig").BN254Scalar;
    const allocator = std.testing.allocator;

    // Simple case: 4 lookups into table of 8 entries
    const lookup_indices = [_]u128{ 0, 1, 2, 3 };
    const lookup_tables = [_]usize{ 0, 0, 0, 0 };

    // Reduction point for 2 cycle variables
    const r_reduction = [_]F{
        F.fromU64(2),
        F.fromU64(3),
    };

    const params = LassoParams(F).init(
        F.fromU64(5), // gamma
        2, // log_T (4 cycles)
        3, // log_K (8 table entries)
        &r_reduction,
    );

    // Initialize prover and compute initial claim
    var lasso_prover = try LassoProver(F).init(
        allocator,
        &lookup_indices,
        &lookup_tables,
        params,
    );
    defer lasso_prover.deinit();

    // For a simple test, compute the initial claim from the expanding table
    const initial_claim = lasso_prover.expanding_v.sum();

    // Run the protocol manually
    var lasso_verifier = try LassoVerifier(F).init(
        allocator,
        params,
        initial_claim,
    );
    defer lasso_verifier.deinit();

    // Run first round
    var round1 = try lasso_prover.computeRoundPolynomial();
    defer round1.deinit();

    // Check that g(0) + g(1) = claim
    const g_0 = round1.evaluate(F.zero());
    const g_1 = round1.evaluate(F.one());
    const sum = g_0.add(g_1);

    // The sum should match the current claim
    try std.testing.expect(sum.eql(initial_claim));
}

test "lasso expanding table accumulation" {
    const F = @import("../../field/mod.zig").BN254Scalar;
    const allocator = std.testing.allocator;

    // Test that ExpandingTable correctly accumulates EQ values
    var table = try ExpandingTable(F).init(allocator, 4);
    defer table.deinit();

    // Bind 3 challenges
    const challenges = [_]F{
        F.fromU64(2),
        F.fromU64(3),
        F.fromU64(5),
    };

    for (challenges) |c| {
        try table.bind(c);
    }

    // After 3 bindings, should have 8 entries
    try std.testing.expectEqual(@as(usize, 8), table.size());

    // Sum should always be 1 (EQ identity)
    try std.testing.expect(table.sum().eql(F.one()));

    // Manually verify some entries
    // eq(r, 000) = (1-r0)(1-r1)(1-r2)
    const one = F.one();
    const expected_000 = one.sub(challenges[0]).mul(one.sub(challenges[1])).mul(one.sub(challenges[2]));
    try std.testing.expect(table.get(0).eql(expected_000));
}

test "lasso split eq optimization" {
    const F = @import("../../field/mod.zig").BN254Scalar;
    const allocator = std.testing.allocator;

    // Test SplitEqPolynomial computes correct EQ values
    const w = [_]F{
        F.fromU64(2),
        F.fromU64(3),
        F.fromU64(5),
        F.fromU64(7),
    };

    var split = try SplitEqPolynomial(F).init(allocator, 2, 2, &w);
    defer split.deinit();

    // Create test function values
    const f = [_]F{
        F.fromU64(1),
        F.fromU64(2),
        F.fromU64(3),
        F.fromU64(4),
        F.fromU64(5),
        F.fromU64(6),
        F.fromU64(7),
        F.fromU64(8),
        F.fromU64(9),
        F.fromU64(10),
        F.fromU64(11),
        F.fromU64(12),
        F.fromU64(13),
        F.fromU64(14),
        F.fromU64(15),
        F.fromU64(16),
    };

    // Compute inner product using SplitEq
    const result = split.innerProduct(&f);

    // Verify by computing manually
    var expected = F.zero();
    for (0..16) |j| {
        const eq_val = split.getEq(j);
        expected = expected.add(eq_val.mul(f[j]));
    }

    try std.testing.expect(result.eql(expected));
}

test "lasso verifier rejects invalid proof" {
    const F = @import("../../field/mod.zig").BN254Scalar;
    const allocator = std.testing.allocator;

    const r_reduction = [_]F{
        F.fromU64(2),
        F.fromU64(3),
    };

    const params = LassoParams(F).init(
        F.fromU64(5),
        2,
        3,
        &r_reduction,
    );

    // Claim = 100
    var lasso_verifier = try LassoVerifier(F).init(allocator, params, F.fromU64(100));
    defer lasso_verifier.deinit();

    // Create invalid round polynomial where g(0) + g(1) != 100
    // g(X) = 40 + 50X => g(0) = 40, g(1) = 90, sum = 130 != 100
    const coeffs = try allocator.alloc(F, 2);
    defer allocator.free(coeffs);
    coeffs[0] = F.fromU64(40);
    coeffs[1] = F.fromU64(50);

    const round_poly = poly.UniPoly(F){
        .coeffs = coeffs,
        .allocator = allocator,
    };

    // Should fail verification
    const result = lasso_verifier.verifyRound(round_poly);
    try std.testing.expectError(error.SumcheckVerificationFailed, result);
}

test "lasso verifier accepts valid proof" {
    const F = @import("../../field/mod.zig").BN254Scalar;
    const allocator = std.testing.allocator;

    const r_reduction = [_]F{
        F.fromU64(2),
        F.fromU64(3),
    };

    const params = LassoParams(F).init(
        F.fromU64(5),
        2,
        3,
        &r_reduction,
    );

    // Claim = 100
    var lasso_verifier = try LassoVerifier(F).init(allocator, params, F.fromU64(100));
    defer lasso_verifier.deinit();

    // Create valid round polynomial where g(0) + g(1) = 100
    // g(X) = 40 + 20X => g(0) = 40, g(1) = 60, sum = 100 ✓
    const coeffs = try allocator.alloc(F, 2);
    defer allocator.free(coeffs);
    coeffs[0] = F.fromU64(40);
    coeffs[1] = F.fromU64(20);

    const round_poly = poly.UniPoly(F){
        .coeffs = coeffs,
        .allocator = allocator,
    };

    // Should pass verification
    const challenge = try lasso_verifier.verifyRound(round_poly);
    _ = challenge;

    // Verifier should advance to next round
    try std.testing.expectEqual(@as(usize, 1), lasso_verifier.round);

    // New claim should be g(challenge)
    try std.testing.expect(!lasso_verifier.claim.eql(F.fromU64(100)));
}

test "lasso multiple rounds consistent" {
    const F = @import("../../field/mod.zig").BN254Scalar;
    const allocator = std.testing.allocator;

    const lookup_indices = [_]u128{ 0, 1, 2, 3 };
    const lookup_tables = [_]usize{ 0, 0, 0, 0 };

    const r_reduction = [_]F{
        F.fromU64(2),
        F.fromU64(3),
    };

    const params = LassoParams(F).init(
        F.fromU64(5),
        2, // log_T
        3, // log_K
        &r_reduction,
    );

    var lasso_prover = try LassoProver(F).init(
        allocator,
        &lookup_indices,
        &lookup_tables,
        params,
    );
    defer lasso_prover.deinit();

    // Track claim consistency
    var prev_claim = lasso_prover.expanding_v.sum();
    var challenges_used = std.ArrayList(F).init(allocator);
    defer challenges_used.deinit();

    // Run 3 rounds
    var round: usize = 0;
    while (round < 3 and !lasso_prover.isComplete()) : (round += 1) {
        var round_poly = try lasso_prover.computeRoundPolynomial();
        defer round_poly.deinit();

        // Check consistency: g(0) + g(1) should equal previous claim
        const g_0 = round_poly.evaluate(F.zero());
        const g_1 = round_poly.evaluate(F.one());
        const sum = g_0.add(g_1);
        try std.testing.expect(sum.eql(prev_claim));

        // Generate challenge (deterministic for testing)
        const challenge = F.fromU64(@as(u64, @intCast(round + 1)) * 7);
        try challenges_used.append(challenge);

        // Update claim for next round
        prev_claim = round_poly.evaluate(challenge);

        // Prover receives challenge
        try lasso_prover.receiveChallenge(challenge);
    }

    try std.testing.expectEqual(@as(usize, 3), challenges_used.items.len);
}
