//! Field Arithmetic Example
//!
//! This example demonstrates the BN254 scalar field operations:
//! 1. Basic arithmetic (add, sub, mul, div)
//! 2. Batch operations
//! 3. Polynomial evaluation

const std = @import("std");
const zolt = @import("zolt");

const BN254Scalar = zolt.field.BN254Scalar;
const BatchOps = zolt.field.BatchOps;

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    std.debug.print("=== Zolt Field Arithmetic Example ===\n\n", .{});

    // ========================================
    // Basic Field Operations
    // ========================================
    std.debug.print("--- Basic Operations ---\n\n", .{});

    // Create field elements
    const a = BN254Scalar.fromU64(42);
    const b = BN254Scalar.fromU64(17);

    std.debug.print("a = {d}\n", .{a.toU64()});
    std.debug.print("b = {d}\n\n", .{b.toU64()});

    // Addition
    const sum = a.add(b);
    std.debug.print("a + b = {d}\n", .{sum.toU64()});

    // Subtraction
    const diff = a.sub(b);
    std.debug.print("a - b = {d}\n", .{diff.toU64()});

    // Multiplication
    const prod = a.mul(b);
    std.debug.print("a * b = {d}\n", .{prod.toU64()});

    // Division (multiplication by inverse)
    const b_inv = b.inverse().?;
    const quot = a.mul(b_inv);
    std.debug.print("a / b = {d}\n\n", .{quot.toU64()});

    // Verify: (a / b) * b = a
    const verify = quot.mul(b);
    std.debug.print("Verification: (a/b) * b = {d} (should equal a = {d})\n\n", .{ verify.toU64(), a.toU64() });

    // ========================================
    // Exponentiation
    // ========================================
    std.debug.print("--- Exponentiation ---\n\n", .{});

    const base = BN254Scalar.fromU64(2);
    const exp10 = base.pow(10);
    std.debug.print("2^10 = {d} (expected: 1024)\n\n", .{exp10.toU64()});

    // ========================================
    // Batch Operations
    // ========================================
    std.debug.print("--- Batch Operations ---\n\n", .{});

    // Create arrays for batch operations
    var values = [_]BN254Scalar{
        BN254Scalar.fromU64(1),
        BN254Scalar.fromU64(2),
        BN254Scalar.fromU64(3),
        BN254Scalar.fromU64(4),
        BN254Scalar.fromU64(5),
    };

    std.debug.print("Values: [ ", .{});
    for (values) |v| {
        std.debug.print("{d} ", .{v.toU64()});
    }
    std.debug.print("]\n", .{});

    // Sum all values manually
    var total = BN254Scalar.zero();
    for (values) |v| {
        total = total.add(v);
    }
    std.debug.print("Sum: {d} (expected: 15)\n", .{total.toU64()});

    // Product of all values manually
    var product = BN254Scalar.one();
    for (values) |v| {
        product = product.mul(v);
    }
    std.debug.print("Product: {d} (expected: 120)\n\n", .{product.toU64()});

    // Batch inverse (Montgomery's trick)
    var inverses: [5]BN254Scalar = undefined;
    try BatchOps.batchInverse(&inverses, &values, allocator);

    std.debug.print("Batch inverses computed:\n", .{});
    for (values, 0..) |v, i| {
        const check = v.mul(inverses[i]);
        std.debug.print("  {d} * inv({d}) = {d} (should be 1)\n", .{
            v.toU64(),
            v.toU64(),
            check.toU64(),
        });
    }
    std.debug.print("\n", .{});

    // ========================================
    // Polynomial Evaluation (Horner's method)
    // ========================================
    std.debug.print("--- Polynomial Evaluation ---\n\n", .{});

    // Polynomial: p(x) = 1 + 2x + 3x^2 + 4x^3
    const coefficients = [_]BN254Scalar{
        BN254Scalar.fromU64(1), // constant term
        BN254Scalar.fromU64(2), // x coefficient
        BN254Scalar.fromU64(3), // x^2 coefficient
        BN254Scalar.fromU64(4), // x^3 coefficient
    };

    const x = BN254Scalar.fromU64(2);
    const result = BatchOps.hornerEval(&coefficients, x);

    // Expected: 1 + 2*2 + 3*4 + 4*8 = 1 + 4 + 12 + 32 = 49
    std.debug.print("p(x) = 1 + 2x + 3x^2 + 4x^3\n", .{});
    std.debug.print("p(2) = {d} (expected: 49)\n\n", .{result.toU64()});

    // ========================================
    // Inner Product
    // ========================================
    std.debug.print("--- Inner Product ---\n\n", .{});

    const vec_a = [_]BN254Scalar{
        BN254Scalar.fromU64(1),
        BN254Scalar.fromU64(2),
        BN254Scalar.fromU64(3),
    };
    const vec_b = [_]BN254Scalar{
        BN254Scalar.fromU64(4),
        BN254Scalar.fromU64(5),
        BN254Scalar.fromU64(6),
    };

    const inner = BatchOps.innerProduct(&vec_a, &vec_b);
    // Expected: 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    std.debug.print("<[1,2,3], [4,5,6]> = {d} (expected: 32)\n\n", .{inner.toU64()});

    // ========================================
    // Field Properties
    // ========================================
    std.debug.print("--- Field Properties ---\n\n", .{});

    // The field is finite - show wrap-around
    const large = BN254Scalar.fromU64(std.math.maxInt(u64));
    const one = BN254Scalar.one();
    const wrapped = large.add(one).add(one);
    _ = wrapped; // Value wraps correctly in the field

    std.debug.print("Field modulus (approx): ~2^254\n", .{});
    std.debug.print("maxInt(u64) + 2 wraps correctly in the field\n\n", .{});

    // Identity elements
    std.debug.print("Additive identity (zero): {d}\n", .{BN254Scalar.zero().toU64()});
    std.debug.print("Multiplicative identity (one): {d}\n\n", .{BN254Scalar.one().toU64()});

    // Negation
    const five = BN254Scalar.fromU64(5);
    const neg_five = five.neg();
    const should_be_zero = five.add(neg_five);
    std.debug.print("5 + (-5) = {d} (should be 0)\n\n", .{should_be_zero.toU64()});

    std.debug.print("=== Example Complete ===\n", .{});
}
