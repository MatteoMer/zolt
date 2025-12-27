//! Simple Proof Example
//!
//! This example demonstrates the basic components in Zolt:
//! 1. Create a polynomial
//! 2. Evaluate it at a point
//! 3. Use the Fiat-Shamir transcript

const std = @import("std");
const zolt = @import("zolt");

const BN254Scalar = zolt.field.BN254Scalar;
const DensePolynomial = zolt.poly.DensePolynomial(BN254Scalar);
const MockCommitment = zolt.poly.commitment.MockCommitment(BN254Scalar);
const Transcript = zolt.transcripts.Transcript(BN254Scalar);

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    std.debug.print("=== Zolt Simple Proof Example ===\n\n", .{});

    // ========================================
    // Step 1: Create a Polynomial
    // ========================================
    std.debug.print("--- Step 1: Create Polynomial ---\n\n", .{});

    // Create a simple multilinear polynomial: f(x0, x1) with 4 coefficients
    // This represents evaluations at the boolean hypercube:
    //   f(0,0) = 1, f(1,0) = 2, f(0,1) = 3, f(1,1) = 4
    const coeffs = [_]BN254Scalar{
        BN254Scalar.fromU64(1), // f(0,0) = 1
        BN254Scalar.fromU64(2), // f(1,0) = 2
        BN254Scalar.fromU64(3), // f(0,1) = 3
        BN254Scalar.fromU64(4), // f(1,1) = 4
    };

    var poly = try DensePolynomial.init(allocator, &coeffs);
    defer poly.deinit();

    std.debug.print("Created polynomial with {} variables\n", .{poly.num_vars});
    std.debug.print("Number of evaluations: {}\n\n", .{poly.len()});

    // ========================================
    // Step 2: Evaluate at a Point
    // ========================================
    std.debug.print("--- Step 2: Evaluate Polynomial ---\n\n", .{});

    // Evaluate at point (5, 7)
    const point = [_]BN254Scalar{
        BN254Scalar.fromU64(5),
        BN254Scalar.fromU64(7),
    };

    const evaluation = poly.evaluate(&point);
    std.debug.print("f(5, 7) = {}\n\n", .{evaluation.toU64()});

    // Verify using manual calculation:
    // f(x0, x1) = (1-x0)(1-x1)*1 + x0*(1-x1)*2 + (1-x0)*x1*3 + x0*x1*4
    // f(5, 7) = (-4)(-6)*1 + 5*(-6)*2 + (-4)*7*3 + 5*7*4
    //         = 24 - 60 - 84 + 140 = 20
    std.debug.print("Manual verification:\n", .{});
    std.debug.print("f(x0,x1) = (1-x0)(1-x1)*1 + x0*(1-x1)*2 + (1-x0)*x1*3 + x0*x1*4\n", .{});

    // ========================================
    // Step 3: Fiat-Shamir Transcript
    // ========================================
    std.debug.print("\n--- Step 3: Fiat-Shamir Transcript ---\n\n", .{});

    var transcript = Transcript.init(allocator);
    defer transcript.deinit();

    // Append domain separator
    const domain = "example_proof";
    transcript.appendMessage("domain", domain);
    std.debug.print("Appended domain separator: \"{s}\"\n", .{domain});

    // Append the evaluation point
    transcript.appendField("point_0", point[0]);
    transcript.appendField("point_1", point[1]);
    std.debug.print("Appended evaluation point\n", .{});

    // Append the evaluation
    transcript.appendField("evaluation", evaluation);
    std.debug.print("Appended evaluation\n", .{});

    // Get a challenge
    const challenge = transcript.challengeField("challenge");
    std.debug.print("\nGenerated challenge: 0x{x}...\n", .{challenge.limbs[0]});

    // Get another challenge (demonstrates state accumulation)
    const challenge2 = transcript.challengeField("challenge2");
    std.debug.print("Generated challenge2: 0x{x}...\n", .{challenge2.limbs[0]});

    // Challenges should be different
    const different = !challenge.eql(challenge2);
    std.debug.print("Challenges are different: {}\n\n", .{different});

    // ========================================
    // Step 4: Mock Commitment
    // ========================================
    std.debug.print("--- Step 4: Mock Commitment ---\n\n", .{});

    // Setup the mock commitment scheme
    const setup = try MockCommitment.setup(allocator, poly.num_vars);

    // Commit to the polynomial
    const commitment = MockCommitment.commit(&setup, &coeffs);
    std.debug.print("Commitment hash: 0x{x}\n", .{commitment.hash});

    // Open the commitment
    const proof = MockCommitment.open(&setup, &coeffs, &point, evaluation);
    std.debug.print("Opening proof value: {}\n", .{proof.value.toU64()});

    // Verify the opening
    const verified = MockCommitment.verify(&setup, commitment, &point, evaluation, proof);
    std.debug.print("Verification result: {}\n\n", .{verified});

    if (verified) {
        std.debug.print("SUCCESS: Commitment and opening verified!\n", .{});
    } else {
        std.debug.print("FAILURE: Verification failed!\n", .{});
    }

    std.debug.print("\n=== Example Complete ===\n", .{});
}
