//! HyperKZG Commitment Example
//!
//! This example demonstrates the HyperKZG polynomial commitment scheme:
//! 1. Setup trusted reference string (SRS)
//! 2. Commit to a multilinear polynomial
//! 3. Open the commitment at a random point
//! 4. Verify the opening proof
//!
//! HyperKZG is a pairing-based polynomial commitment scheme that provides
//! O(log n) proof size and O(n log n) prover time for multilinear polynomials.

const std = @import("std");
const zolt = @import("zolt");

const BN254Scalar = zolt.field.BN254Scalar;
const HyperKZG = zolt.poly.commitment.HyperKZG(BN254Scalar);
const DensePolynomial = zolt.poly.DensePolynomial(BN254Scalar);

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    std.debug.print("=== Zolt HyperKZG Commitment Example ===\n\n", .{});

    // ========================================
    // Step 1: Setup Trusted Reference String
    // ========================================
    std.debug.print("--- Step 1: Generate SRS (Trusted Setup) ---\n\n", .{});

    // For a polynomial with n evaluations, we need an SRS of size n.
    // Here we use 8 evaluations (3 variables: 2^3 = 8)
    const poly_size: usize = 8;
    const num_vars: usize = 3;

    var srs = try HyperKZG.setup(allocator, poly_size);
    defer srs.deinit();
    // Note: In production, SRS would be loaded from a ceremony file

    std.debug.print("Generated SRS for polynomials up to {} evaluations\n", .{srs.max_degree});
    std.debug.print("Number of G1 points: {}\n", .{srs.powers_of_tau_g1.len});
    std.debug.print("G1 generator: (0x{x}...)\n\n", .{srs.g1.x.limbs[0]});

    // ========================================
    // Step 2: Create a Multilinear Polynomial
    // ========================================
    std.debug.print("--- Step 2: Create Polynomial ---\n\n", .{});

    // Create a simple multilinear polynomial with 8 evaluations
    // f(x0, x1, x2) evaluated at all boolean points {0,1}^3
    const evals = [_]BN254Scalar{
        BN254Scalar.fromU64(1), // f(0,0,0)
        BN254Scalar.fromU64(2), // f(1,0,0)
        BN254Scalar.fromU64(3), // f(0,1,0)
        BN254Scalar.fromU64(4), // f(1,1,0)
        BN254Scalar.fromU64(5), // f(0,0,1)
        BN254Scalar.fromU64(6), // f(1,0,1)
        BN254Scalar.fromU64(7), // f(0,1,1)
        BN254Scalar.fromU64(8), // f(1,1,1)
    };

    std.debug.print("Polynomial evaluations:\n", .{});
    for (0..8) |i| {
        const b0 = i & 1;
        const b1 = (i >> 1) & 1;
        const b2 = (i >> 2) & 1;
        std.debug.print("  f({},{},{}) = {}\n", .{ b0, b1, b2, i + 1 });
    }
    std.debug.print("\nPolynomial has {} variables\n\n", .{num_vars});

    // ========================================
    // Step 3: Commit to the Polynomial
    // ========================================
    std.debug.print("--- Step 3: Commit to Polynomial ---\n\n", .{});

    const commitment = HyperKZG.commit(&srs, &evals);

    std.debug.print("Commitment (G1 point):\n", .{});
    std.debug.print("  x: 0x{x}...\n", .{commitment.point.x.limbs[0]});
    std.debug.print("  y: 0x{x}...\n", .{commitment.point.y.limbs[0]});
    std.debug.print("  infinity: {}\n\n", .{commitment.point.infinity});

    // ========================================
    // Step 4: Open the Commitment at a Point
    // ========================================
    std.debug.print("--- Step 4: Open Commitment ---\n\n", .{});

    // Choose an evaluation point (not necessarily boolean)
    const point = [_]BN254Scalar{
        BN254Scalar.fromU64(2), // x0 = 2
        BN254Scalar.fromU64(3), // x1 = 3
        BN254Scalar.fromU64(5), // x2 = 5
    };

    std.debug.print("Evaluation point: (2, 3, 5)\n", .{});

    // Evaluate the polynomial at this point using multilinear extension
    var poly = try DensePolynomial.init(allocator, &evals);
    defer poly.deinit();
    const claimed_value = poly.evaluate(&point);

    // Convert from Montgomery form to show actual value
    const value_standard = claimed_value.fromMontgomery();
    std.debug.print("Claimed evaluation: f(2,3,5) = 0x{x}\n\n", .{value_standard.limbs[0]});

    // Generate opening proof
    var opening = try HyperKZG.open(&srs, &evals, &point, claimed_value, allocator);
    defer opening.deinit();

    std.debug.print("Opening proof generated\n", .{});
    std.debug.print("  Number of quotient commitments: {}\n\n", .{opening.quotient_commitments.len});

    // ========================================
    // Step 5: Verify the Opening
    // ========================================
    std.debug.print("--- Step 5: Verify Opening ---\n\n", .{});

    const valid = HyperKZG.verify(&srs, commitment, &point, claimed_value, &opening);

    std.debug.print("Verification result: {s}\n\n", .{if (valid) "VALID" else "INVALID"});

    if (valid) {
        std.debug.print("SUCCESS: The prover has demonstrated knowledge of a polynomial\n", .{});
        std.debug.print("         that commits to the given point and evaluates correctly!\n", .{});
    } else {
        std.debug.print("FAILURE: The opening proof is invalid.\n", .{});
    }

    // ========================================
    // Step 6: Demonstrate Soundness
    // ========================================
    std.debug.print("\n--- Step 6: Demonstrate Soundness ---\n\n", .{});

    // Try to verify with a wrong value - should fail
    const wrong_value = claimed_value.add(BN254Scalar.one());
    var wrong_opening = try HyperKZG.open(&srs, &evals, &point, wrong_value, allocator);
    defer wrong_opening.deinit();

    const wrong_valid = HyperKZG.verify(&srs, commitment, &point, wrong_value, &wrong_opening);

    std.debug.print("Trying to verify with incorrect claimed value...\n", .{});
    std.debug.print("Verification result: {s}\n", .{if (wrong_valid) "VALID (unexpected!)" else "INVALID (expected)"});

    std.debug.print("\n=== Example Complete ===\n", .{});
}
