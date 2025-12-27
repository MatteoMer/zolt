//! Integration tests for Zolt zkVM
//!
//! These tests verify the correct interaction between components
//! and ensure end-to-end functionality.

const std = @import("std");
const testing = std.testing;

const common = @import("common/mod.zig");
const field = @import("field/mod.zig");
const poly = @import("poly/mod.zig");
const zkvm = @import("zkvm/mod.zig");
const msm = @import("msm/mod.zig");
const tracer = @import("tracer/mod.zig");
const host = @import("host/mod.zig");
const transcripts = @import("transcripts/mod.zig");

const BN254Scalar = field.BN254Scalar;

// ============================================================================
// Field Arithmetic Integration Tests
// ============================================================================

test "field operations chain" {
    // Test a chain of operations: (a + b) * (a - b) = a² - b²
    const a = BN254Scalar.fromU64(12345);
    const b = BN254Scalar.fromU64(6789);

    const sum = a.add(b);
    const diff = a.sub(b);
    const product = sum.mul(diff);

    const a_sq = a.square();
    const b_sq = b.square();
    const expected = a_sq.sub(b_sq);

    try testing.expect(product.eql(expected));
}

test "field exponentiation and inverse" {
    const a = BN254Scalar.fromU64(42);

    // a * a^(-1) = 1
    const a_inv = a.inverse().?;
    const product = a.mul(a_inv);
    try testing.expect(product.eql(BN254Scalar.one()));

    // a^2 = a * a
    const a_sq_1 = a.pow(2);
    const a_sq_2 = a.mul(a);
    try testing.expect(a_sq_1.eql(a_sq_2));
}

// ============================================================================
// Polynomial Integration Tests
// ============================================================================

test "univariate polynomial operations" {
    const allocator = testing.allocator;

    var coeffs = [_]BN254Scalar{
        BN254Scalar.fromU64(1),
        BN254Scalar.fromU64(2),
        BN254Scalar.fromU64(1),
    };

    var p = try poly.UniPoly(BN254Scalar).init(allocator, &coeffs);
    defer p.deinit();

    // Evaluate at x = 3: 1 + 2*3 + 1*9 = 16
    const result = p.evaluate(BN254Scalar.fromU64(3));
    try testing.expect(result.eql(BN254Scalar.fromU64(16)));
}

// ============================================================================
// Transcript Integration Tests
// ============================================================================

test "transcript produces consistent challenges" {
    const allocator = testing.allocator;

    var transcript1 = try transcripts.Transcript(BN254Scalar).init(allocator, "test");
    defer transcript1.deinit();

    var transcript2 = try transcripts.Transcript(BN254Scalar).init(allocator, "test");
    defer transcript2.deinit();

    // Append same messages
    try transcript1.appendMessage("msg", "hello");
    try transcript2.appendMessage("msg", "hello");

    // Should produce same challenges
    const c1 = try transcript1.challengeScalar("challenge");
    const c2 = try transcript2.challengeScalar("challenge");

    try testing.expect(c1.eql(c2));
}

test "transcript challenge changes with message" {
    const allocator = testing.allocator;

    var transcript1 = try transcripts.Transcript(BN254Scalar).init(allocator, "test");
    defer transcript1.deinit();

    var transcript2 = try transcripts.Transcript(BN254Scalar).init(allocator, "test");
    defer transcript2.deinit();

    // Append different messages
    try transcript1.appendMessage("msg", "hello");
    try transcript2.appendMessage("msg", "world");

    // Should produce different challenges
    const c1 = try transcript1.challengeScalar("challenge");
    const c2 = try transcript2.challengeScalar("challenge");

    try testing.expect(!c1.eql(c2));
}

// ============================================================================
// MSM Integration Tests
// ============================================================================

test "msm with identity points" {
    const Point = msm.AffinePoint(BN254Scalar);
    const SingleMSM = msm.MSM(BN254Scalar, BN254Scalar);

    // MSM with identity should still work
    const bases = [_]Point{
        Point.identity(),
        Point.fromCoords(BN254Scalar.fromU64(1), BN254Scalar.fromU64(2)),
        Point.identity(),
    };
    const scalars = [_]BN254Scalar{
        BN254Scalar.fromU64(5),
        BN254Scalar.fromU64(3),
        BN254Scalar.fromU64(7),
    };

    // Should compute 0 + 3*P + 0 = 3*P
    const result = SingleMSM.compute(&bases, &scalars);
    _ = result; // Just verify it doesn't crash
}

test "msm scalar multiply consistency" {
    const Point = msm.AffinePoint(BN254Scalar);
    const Proj = msm.ProjectivePoint(BN254Scalar);
    const SingleMSM = msm.MSM(BN254Scalar, BN254Scalar);

    const p = Point.fromCoords(BN254Scalar.fromU64(5), BN254Scalar.fromU64(7));

    // 2*P via scalar multiplication
    const proj_2p = SingleMSM.scalarMul(p, BN254Scalar.fromU64(2));
    const affine_2p = proj_2p.toAffine();

    // 2*P via doubling
    const proj_p = Proj.fromAffine(p);
    const doubled = proj_p.double().toAffine();

    try testing.expect(affine_2p.eql(doubled));
}

// ============================================================================
// Emulator Integration Tests
// ============================================================================

test "emulator handles memory operations" {
    const allocator = testing.allocator;
    const config = common.MemoryConfig{ .program_size = 1024 };

    var emu = tracer.Emulator.init(allocator, &config);
    defer emu.deinit();

    // Program: addi x1, x0, 42
    const program = [_]u8{
        0x93, 0x00, 0xa0, 0x02, // addi x1, x0, 42
    };

    try emu.loadProgram(&program);

    _ = try emu.step();

    const x1 = try emu.registers.read(1);
    try testing.expectEqual(@as(u64, 42), x1);
}

// ============================================================================
// Commitment Scheme Integration Tests
// ============================================================================

test "hyperkzg commit and open" {
    const allocator = testing.allocator;
    const HKZG = poly.commitment.HyperKZG(BN254Scalar);

    var params = try HKZG.setup(allocator, 16);
    defer params.deinit();

    // Simple polynomial evaluations
    const evals = [_]BN254Scalar{
        BN254Scalar.fromU64(1),
        BN254Scalar.fromU64(2),
        BN254Scalar.fromU64(3),
        BN254Scalar.fromU64(4),
    };

    // Commit
    const commitment = HKZG.commit(&params, &evals);

    // Opening point
    const point = [_]BN254Scalar{ BN254Scalar.fromU64(1), BN254Scalar.fromU64(1) };

    // Compute expected value at (1,1): should be eval[3] = 4
    const expected = BN254Scalar.fromU64(4);

    // Open
    var proof = try HKZG.open(&params, &evals, &point, expected, allocator);
    defer proof.deinit();

    // Verify
    const valid = HKZG.verify(&params, commitment, &point, expected, &proof);
    try testing.expect(valid);
}

// ============================================================================
// Compressed Instruction Integration Tests
// ============================================================================

test "compressed instruction full pipeline" {
    const allocator = testing.allocator;
    const config = common.MemoryConfig{ .program_size = 1024 };

    var emu = tracer.Emulator.init(allocator, &config);
    defer emu.deinit();

    // Test C.NOP followed by regular instruction
    const program = [_]u8{
        0x01, 0x00, // C.NOP (2 bytes)
        0x93, 0x00, 0xa0, 0x02, // addi x1, x0, 42 (4 bytes)
    };

    try emu.loadProgram(&program);

    // Execute C.NOP
    _ = try emu.step();
    try testing.expectEqual(common.constants.RAM_START_ADDRESS + 2, emu.state.pc);

    // Execute ADDI
    _ = try emu.step();
    const x1 = try emu.registers.read(1);
    try testing.expectEqual(@as(u64, 42), x1);
}

// ============================================================================
// Extension Field Integration Tests
// ============================================================================

test "fp2 arithmetic chain" {
    const Fp2 = field.pairing.Fp2;

    const a = Fp2.init(BN254Scalar.fromU64(3), BN254Scalar.fromU64(4));
    const b = Fp2.init(BN254Scalar.fromU64(1), BN254Scalar.fromU64(2));

    // (a * b) * b^(-1) = a
    const product = a.mul(b);
    const b_inv = b.inverse().?;
    const result = product.mul(b_inv);

    try testing.expect(result.eql(a));
}

test "fp12 multiplicative identity" {
    const Fp12 = field.pairing.Fp12;

    const one = Fp12.one();
    const zero = Fp12.zero();

    // 1 * 1 = 1
    try testing.expect(one.mul(one).eql(one));

    // 1 + 0 = 1
    try testing.expect(one.add(zero).eql(one));
}
