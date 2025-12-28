//! Sumcheck Protocol Example
//!
//! This example demonstrates the sumcheck protocol, a fundamental building block
//! of modern SNARK systems. The sumcheck protocol proves that:
//!
//!   ∑_{b ∈ {0,1}^n} g(b) = H
//!
//! where g is a multilinear polynomial and H is the claimed sum.
//!
//! The protocol runs in n rounds, each reducing one variable, resulting in
//! an O(n) proof size and verification time.

const std = @import("std");
const zolt = @import("zolt");

const BN254Scalar = zolt.field.BN254Scalar;
const Sumcheck = zolt.subprotocols.Sumcheck(BN254Scalar);
const DensePolynomial = zolt.poly.DensePolynomial(BN254Scalar);

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    std.debug.print("=== Zolt Sumcheck Protocol Example ===\n\n", .{});

    // ========================================
    // Step 1: Define a Polynomial
    // ========================================
    std.debug.print("--- Step 1: Define Polynomial ---\n\n", .{});

    // Create a multilinear polynomial with 4 evaluations (2 variables)
    // g(x0, x1) with evaluations at boolean hypercube
    const evals = [_]BN254Scalar{
        BN254Scalar.fromU64(3), // g(0,0) = 3
        BN254Scalar.fromU64(5), // g(1,0) = 5
        BN254Scalar.fromU64(7), // g(0,1) = 7
        BN254Scalar.fromU64(11), // g(1,1) = 11
    };

    var polynomial = try DensePolynomial.init(allocator, &evals);
    defer polynomial.deinit();

    std.debug.print("Polynomial g(x0, x1):\n", .{});
    std.debug.print("  g(0,0) = 3\n", .{});
    std.debug.print("  g(1,0) = 5\n", .{});
    std.debug.print("  g(0,1) = 7\n", .{});
    std.debug.print("  g(1,1) = 11\n\n", .{});

    // ========================================
    // Step 2: Calculate True Sum
    // ========================================
    std.debug.print("--- Step 2: Calculate Sum over Boolean Hypercube ---\n\n", .{});

    // The sum we want to prove: ∑_{b ∈ {0,1}²} g(b)
    var claimed_sum = BN254Scalar.zero();
    for (evals) |e| {
        claimed_sum = claimed_sum.add(e);
    }

    std.debug.print("Claimed sum H = sum g(b) = 3 + 5 + 7 + 11 = 26\n\n", .{});

    // ========================================
    // Step 3: Run Sumcheck Protocol (Interactive Simulation)
    // ========================================
    std.debug.print("--- Step 3: Run Sumcheck Protocol ---\n\n", .{});

    std.debug.print("Protocol: Prover claims sum g(b) = 26\n", .{});
    std.debug.print("          Verifier will check this claim.\n\n", .{});

    // Initialize prover with a copy of the polynomial
    const prover_poly = try DensePolynomial.init(allocator, &evals);
    var prover = Sumcheck.Prover.init(allocator, prover_poly);

    // Initialize verifier
    var verifier = Sumcheck.Verifier.init(allocator, claimed_sum);
    defer verifier.deinit();

    // Run the interactive protocol
    var round_num: usize = 1;
    var verification_ok = true;

    std.debug.print("Running interactive sumcheck protocol:\n\n", .{});

    while (!prover.isComplete()) {
        // Prover sends round polynomial
        const round = try prover.nextRound(allocator);

        std.debug.print("  Round {}:\n", .{round_num});
        std.debug.print("    Prover sends univariate polynomial\n", .{});

        // Verifier checks the round and generates challenge
        const challenge = verifier.verifyRound(round) catch |err| {
            std.debug.print("    Verifier REJECTS: {}\n", .{err});
            verification_ok = false;
            break;
        };

        std.debug.print("    Verifier: g(0) + g(1) = claim? YES\n", .{});
        std.debug.print("    Verifier sends challenge\n", .{});

        // Prover receives challenge and updates state
        try prover.receiveChallenge(challenge);

        round_num += 1;
    }

    std.debug.print("\nProtocol completed in {} rounds\n\n", .{round_num - 1});

    // ========================================
    // Step 4: Final Verification
    // ========================================
    std.debug.print("--- Step 4: Final Verification ---\n\n", .{});

    if (verification_ok) {
        // Get the final evaluation from prover
        const final_eval = prover.getFinalEval();

        // Verifier checks that final_eval equals their current claim
        const valid = final_eval.eql(verifier.claim);

        std.debug.print("Final evaluation matches claim: {s}\n\n", .{if (valid) "YES" else "NO"});

        if (valid) {
            std.debug.print("PROOF VERIFIED: The prover has correctly demonstrated the sum.\n", .{});
        } else {
            std.debug.print("PROOF INVALID: Final evaluation does not match claim.\n", .{});
        }
    } else {
        std.debug.print("PROOF INVALID: Verification failed during protocol.\n", .{});
    }

    std.debug.print("\n=== Example Complete ===\n", .{});
}
