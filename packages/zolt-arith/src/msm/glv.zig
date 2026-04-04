//! GLV Endomorphism Scalar Multiplication for BN254
//!
//! G1: 2D GLV using the curve endomorphism φ(x,y) = (ω·x, y)
//!   - Decomposes 254-bit scalar into two ~128-bit sub-scalars
//!   - Uses Shamir's trick for ~2x speedup
//!
//! G2: 4D GLV using Frobenius endomorphism ψ
//!   - Decomposes 254-bit scalar into four ~66-bit sub-scalars
//!   - Uses Shamir's trick for ~4x speedup
//!
//! Ported from arkworks jolt-optimizations (a16z/jolt).

const std = @import("std");
const field = @import("../field/mod.zig");
const pairing = @import("../field/pairing.zig");
const msm_mod = @import("mod.zig");

const Fr = field.BN254Scalar;
const Fp = field.BN254BaseField;
const Fp2 = pairing.Fp2;
const G1Affine = msm_mod.AffinePoint(Fp);
const G1Projective = msm_mod.ProjectivePoint(Fp);
const G2Point = pairing.G2Point;
const G2Projective = pairing.G2Projective;

// ============================================================================
// Constants
// ============================================================================

/// BN254 scalar field modulus as u256
const MODULUS_U256: u256 = limbs4ToU256(field.BN254_MODULUS);

/// G1 GLV endomorphism coefficient ω (cube root of unity in Fq)
/// Raw limbs (non-Montgomery), converted to Montgomery on first use via comptime
const ENDO_COEFF: Fp = blk: {
    const raw = Fp{ .limbs = .{
        0xe4bd44e5607cfd48, 0xc28f069fbb966e3d,
        0x5e6dd9e7e0acccb0, 0x30644e72e131a029,
    } };
    break :blk raw.toMontgomery();
};

// 2D GLV lattice basis coefficients (absolute values)
// N11 = 147946756881789319000765030803803410728 (negative in basis)
const N11_ABS: u128 = 0x6f4d8248eeb859fc_8211bbeb7d4f1128;
// N12 = 9931322734385697763 (positive in basis)
const N12_VAL: u64 = 0x89d3256894d213e3;
// N21 = 9931322734385697763 (negative in basis)
const N21_VAL: u64 = 0x89d3256894d213e3;
// N22 = 147946756881789319010696353538189108491 (negative in basis)
const N22_ABS: u128 = 0x6f4d8248eeb859fd_0be4e1541221250b;

/// Frobenius ψ^1 coefficients for G2 (Fq2 elements, raw non-Montgomery limbs)
const PSI1_COEF2: Fp2 = fp2FromRawLimbs(
    .{ 0x99e39557176f553d, 0xb78cc310c2c3330c, 0x4c0bec3cf559b143, 0x2fb347984f7911f7 },
    .{ 0x1665d51c640fcba2, 0x32ae2a1d0b7c9dce, 0x4ba4cc8bd75a0794, 0x16c9e55061ebae20 },
);
const PSI1_COEF3: Fp2 = fp2FromRawLimbs(
    .{ 0xdc54014671a0135a, 0xdbaae0eda9c95998, 0xdc5ec698b6e2f9b9, 0x063cf305489af5dc },
    .{ 0x82d37f632623b0e3, 0x21807dc98fa25bd2, 0x0704b5a7ec796f2b, 0x07c03cbcac41049a },
);

/// Frobenius ψ^2 coefficients (Fq2 with c1=0)
const PSI2_COEF2: Fp2 = fp2FromRawLimbs(
    .{ 0xe4bd44e5607cfd48, 0xc28f069fbb966e3d, 0x5e6dd9e7e0acccb0, 0x30644e72e131a029 },
    .{ 0, 0, 0, 0 },
);
const PSI2_COEF3: Fp2 = fp2FromRawLimbs(
    .{ 0x3c208c16d87cfd46, 0x97816a916871ca8d, 0xb85045b68181585d, 0x30644e72e131a029 },
    .{ 0, 0, 0, 0 },
);

/// Frobenius ψ^3 coefficients
const PSI3_COEF2: Fp2 = fp2FromRawLimbs(
    .{ 0x7b746ee87bdcfb6d, 0x805ffd3d5d6942d3, 0xbaff1c77959f25ac, 0x0856e078b755ef0a },
    .{ 0x380cab2baaa586de, 0x0fdf31bf98ff2631, 0xa9f30e6dec26094f, 0x04f1de41b3d1766f },
);
const PSI3_COEF3: Fp2 = fp2FromRawLimbs(
    .{ 0x5fcc8ad066dce9ed, 0xbbd689a3bea870f4, 0xdbf17f1dca9e5ea3, 0x2a275b6d9896aa4c },
    .{ 0xb94d0cb3b2594c64, 0x7600ecc7d8cf6eba, 0xb14b900e9507e932, 0x28a411b634f09b8f },
);

fn fp2FromRawLimbs(c0_limbs: [4]u64, c1_limbs: [4]u64) Fp2 {
    const c0 = (Fp{ .limbs = c0_limbs }).toMontgomery();
    const c1 = (Fp{ .limbs = c1_limbs }).toMontgomery();
    return Fp2.init(c0, c1);
}

// ============================================================================
// G1: GLV Endomorphism
// ============================================================================

/// Apply GLV endomorphism to G1 affine point: φ(x,y) = (ω·x, y)
pub fn glvEndomorphismG1(p: G1Affine) G1Affine {
    if (p.infinity) return p;
    return .{ .x = p.x.mul(ENDO_COEFF), .y = p.y, .infinity = false };
}

// ============================================================================
// G2: Frobenius Endomorphism
// ============================================================================

/// Apply Frobenius endomorphism ψ^k to G2 affine point
pub fn frobeniusPsiAffine(p: G2Point, k: usize) G2Point {
    if (p.infinity) return p;

    var x = p.x;
    var y = p.y;

    // Conjugate for odd powers
    if (k & 1 == 1) {
        x = x.conjugate();
        y = y.conjugate();
    }

    // Apply coefficients based on power mod 4
    switch (k % 4) {
        0 => {},
        1 => {
            x = x.mul(PSI1_COEF2);
            y = y.mul(PSI1_COEF3);
        },
        2 => {
            x = x.mul(PSI2_COEF2);
            y = y.mul(PSI2_COEF3);
        },
        3 => {
            x = x.mul(PSI3_COEF2);
            y = y.mul(PSI3_COEF3);
        },
        else => unreachable,
    }

    return .{ .x = x, .y = y, .infinity = false };
}

/// Apply Frobenius endomorphism ψ^k to G2 projective point
pub fn frobeniusPsiProjective(p: G2Projective, k: usize) G2Projective {
    if (p.isIdentity()) return p;

    var x = p.x;
    var y = p.y;
    var z = p.z;

    if (k & 1 == 1) {
        x = x.conjugate();
        y = y.conjugate();
        z = z.conjugate();
    }

    switch (k % 4) {
        0 => {},
        1 => {
            x = x.mul(PSI1_COEF2);
            y = y.mul(PSI1_COEF3);
        },
        2 => {
            x = x.mul(PSI2_COEF2);
            y = y.mul(PSI2_COEF3);
        },
        3 => {
            x = x.mul(PSI3_COEF2);
            y = y.mul(PSI3_COEF3);
        },
        else => unreachable,
    }

    return .{ .x = x, .y = y, .z = z };
}

// ============================================================================
// 2D Scalar Decomposition (G1)
// ============================================================================

pub const Decomp2D = struct {
    k: [2][4]u64, // Two ~128-bit sub-scalars as limbs (non-Montgomery)
    signs: [2]bool, // true = positive, false = negative
};

/// Decompose scalar k into k1, k2 such that k ≡ k1 + λ·k2 (mod r)
/// where |k1|, |k2| ≈ 128 bits
pub fn decomposeScalar2D(scalar: Fr) Decomp2D {
    const normal = scalar.fromMontgomery();
    const k = limbs4ToU256(normal.limbs);

    // beta_1 = floor(k * |N22| / r) — negative conceptually since N22 < 0
    const prod1: u512 = @as(u512, k) * @as(u512, N22_ABS);
    const beta_1_abs: u128 = @truncate(prod1 / @as(u512, MODULUS_U256));

    // beta_2 = floor(k * N12 / r) — negative conceptually since we use -k*N12
    const prod2: u512 = @as(u512, k) * @as(u512, N12_VAL);
    const beta_2_abs: u128 = @truncate(prod2 / @as(u512, MODULUS_U256));

    // b1 = beta_1_abs * |N11| + beta_2_abs * |N21| (both double-negative → positive)
    const b1: u256 = @as(u256, beta_1_abs) * @as(u256, N11_ABS) + @as(u256, beta_2_abs) * @as(u256, N21_VAL);

    // b2 = -beta_1_abs * N12 + beta_2_abs * |N22|
    // = beta_2_abs * |N22| - beta_1_abs * N12
    const term_pos: u256 = @as(u256, beta_2_abs) * @as(u256, N22_ABS);
    const term_neg: u256 = @as(u256, beta_1_abs) * @as(u256, N12_VAL);

    // k1 = k - b1; k2 = -b2
    var k1_sign: bool = true; // positive
    var k1_abs: u256 = undefined;
    if (k >= b1) {
        k1_abs = k - b1;
    } else {
        k1_abs = b1 - k;
        k1_sign = false;
    }

    var k2_sign: bool = undefined;
    var k2_abs: u256 = undefined;
    // k2 = -b2 = -(term_pos - term_neg) = term_neg - term_pos
    if (term_neg >= term_pos) {
        k2_abs = term_neg - term_pos;
        k2_sign = true; // positive
    } else {
        k2_abs = term_pos - term_neg;
        k2_sign = false; // negative
    }

    return .{
        .k = .{
            u256ToLimbs4(k1_abs),
            u256ToLimbs4(k2_abs),
        },
        .signs = .{ k1_sign, k2_sign },
    };
}

// ============================================================================
// 4D Scalar Decomposition (G2)
// ============================================================================

pub const DecompEntry = struct {
    k0: [2]u64,
    k1: [2]u64,
    k2: [2]u64,
    k3: [2]u64,
    neg0: bool,
    neg1: bool,
    neg2: bool,
    neg3: bool,
};

pub const Decomp4D = struct {
    k: [4][4]u64, // Four ~66-bit sub-scalars as limbs (non-Montgomery)
    signs: [4]bool, // true = negative (subtract), false = positive (add)
    max_bits: usize,
};

/// Decompose scalar into 4D components using precomputed table
pub fn decomposeScalar4D(scalar: Fr) Decomp4D {
    const normal = scalar.fromMontgomery();

    var k0: u128 = 0;
    var k1: u128 = 0;
    var k2: u128 = 0;
    var k3: u128 = 0;

    // Iterate over set bits in the scalar
    for (0..254) |bit_pos| {
        const limb_idx = bit_pos / 64;
        const bit_idx: u6 = @intCast(bit_pos % 64);

        if (limb_idx < 4 and (normal.limbs[limb_idx] >> bit_idx) & 1 == 1) {
            const entry = POWER_OF_2_DECOMPOSITIONS[bit_pos];
            const dk0: u128 = @as(u128, entry.k0[1]) << 64 | @as(u128, entry.k0[0]);
            const dk1: u128 = @as(u128, entry.k1[1]) << 64 | @as(u128, entry.k1[0]);
            const dk2: u128 = @as(u128, entry.k2[1]) << 64 | @as(u128, entry.k2[0]);
            const dk3: u128 = @as(u128, entry.k3[1]) << 64 | @as(u128, entry.k3[0]);

            if (entry.neg0) {
                k0 = k0 -% dk0;
            } else {
                k0 = k0 +% dk0;
            }
            if (entry.neg1) {
                k1 = k1 -% dk1;
            } else {
                k1 = k1 +% dk1;
            }
            if (entry.neg2) {
                k2 = k2 -% dk2;
            } else {
                k2 = k2 +% dk2;
            }
            if (entry.neg3) {
                k3 = k3 -% dk3;
            } else {
                k3 = k3 +% dk3;
            }
        }
    }

    // Handle signs: negative when interpreted as i128
    var coeffs: [4]u128 = undefined;
    var signs: [4]bool = undefined;

    inline for (.{ k0, k1, k2, k3 }, 0..) |ki, idx| {
        if (@as(i128, @bitCast(ki)) < 0) {
            coeffs[idx] = 0 -% ki; // wrapping negate
            signs[idx] = true; // negative
        } else {
            coeffs[idx] = ki;
            signs[idx] = false; // positive
        }
    }

    // Find max bits for Shamir's trick loop bound
    var max_bits: usize = 1;
    for (coeffs) |c| {
        const bits = 128 - @as(usize, @clz(c | 1));
        if (bits > max_bits) max_bits = bits;
    }

    // Convert u128 coefficients to [4]u64 limbs
    var result_k: [4][4]u64 = undefined;
    for (coeffs, 0..) |c, idx| {
        result_k[idx] = .{
            @truncate(c),
            @truncate(c >> 64),
            0,
            0,
        };
    }

    return .{
        .k = result_k,
        .signs = signs,
        .max_bits = max_bits,
    };
}

// ============================================================================
// Shamir's Trick — G1 (2D)
// ============================================================================

/// GLV scalar multiplication for G1: ~2x speedup over naive
pub fn glvScalarMulG1(base: G1Affine, scalar: Fr) G1Projective {
    if (base.isIdentity()) return G1Projective.identity();
    if (scalar.isZero()) return G1Projective.identity();

    const decomp = decomposeScalar2D(scalar);

    // Compute bases: P and φ(P)
    var bases: [2]G1Affine = .{
        base,
        glvEndomorphismG1(base),
    };

    // Handle signs by negating base points
    if (!decomp.signs[0]) bases[0] = bases[0].neg();
    if (!decomp.signs[1]) bases[1] = bases[1].neg();

    return shamirMul2D(bases, decomp.k);
}

/// Shamir's trick for 2-point multi-scalar multiplication
fn shamirMul2D(bases: [2]G1Affine, coeffs: [2][4]u64) G1Projective {
    // Find max significant bit
    var max_bits: usize = 1;
    for (coeffs) |c| {
        for (0..4) |i| {
            const limb_idx = 3 - i;
            if (c[limb_idx] != 0) {
                const bits = limb_idx * 64 + (64 - @as(usize, @clz(c[limb_idx])));
                if (bits > max_bits) max_bits = bits;
                break;
            }
        }
    }

    var result = G1Projective.identity();

    var bit_idx: usize = max_bits;
    while (bit_idx > 0) {
        bit_idx -= 1;

        if (!result.isIdentity()) {
            result = result.double();
        }

        const limb_pos = bit_idx / 64;
        const bit_pos: u6 = @intCast(bit_idx % 64);

        for (0..2) |i| {
            if (limb_pos < 4 and (coeffs[i][limb_pos] >> bit_pos) & 1 == 1) {
                if (result.isIdentity()) {
                    result = G1Projective.fromAffine(bases[i]);
                } else {
                    result = result.addAffine(bases[i]);
                }
            }
        }
    }

    return result;
}

// ============================================================================
// Shamir's Trick — G2 (4D)
// ============================================================================

/// GLV scalar multiplication for G2: ~4x speedup over naive
pub fn glvScalarMulG2(base: G2Point, scalar: Fr) G2Projective {
    if (base.isIdentity()) return G2Projective.identity();
    if (scalar.isZero()) return G2Projective.identity();

    const decomp = decomposeScalar4D(scalar);

    // Compute Frobenius bases: P, ψ(P), ψ²(P), ψ³(P)
    var bases: [4]G2Point = .{
        base,
        frobeniusPsiAffine(base, 1),
        frobeniusPsiAffine(base, 2),
        frobeniusPsiAffine(base, 3),
    };

    // Handle signs: in 4D decomposition, signs[i]=true means negative (subtract)
    // We negate the base so we can always add
    for (0..4) |i| {
        if (decomp.signs[i]) {
            bases[i] = bases[i].neg();
        }
    }

    return shamirMul4D(bases, decomp.k, decomp.max_bits);
}

/// GLV scalar multiplication for G2 with precomputed Frobenius bases
pub fn glvScalarMulG2WithBases(bases_in: [4]G2Point, scalar: Fr) G2Projective {
    if (scalar.isZero()) return G2Projective.identity();

    const decomp = decomposeScalar4D(scalar);

    var bases = bases_in;
    for (0..4) |i| {
        if (decomp.signs[i]) {
            bases[i] = bases[i].neg();
        }
    }

    return shamirMul4D(bases, decomp.k, decomp.max_bits);
}

/// Shamir's trick for 4-point multi-scalar multiplication
fn shamirMul4D(bases: [4]G2Point, coeffs: [4][4]u64, max_bits: usize) G2Projective {
    var result = G2Projective.identity();

    var bit_idx: usize = max_bits;
    while (bit_idx > 0) {
        bit_idx -= 1;

        if (!result.isIdentity()) {
            result = result.double();
        }

        const limb_pos = bit_idx / 64;
        const bit_pos: u6 = @intCast(bit_idx % 64);

        for (0..4) |i| {
            if (limb_pos < 4 and (coeffs[i][limb_pos] >> bit_pos) & 1 == 1) {
                if (result.isIdentity()) {
                    result = G2Projective.fromAffine(bases[i]);
                } else {
                    result = result.addAffine(bases[i]);
                }
            }
        }
    }

    return result;
}

// ============================================================================
// Utility functions
// ============================================================================

fn limbs4ToU256(limbs: [4]u64) u256 {
    return @as(u256, limbs[3]) << 192 |
        @as(u256, limbs[2]) << 128 |
        @as(u256, limbs[1]) << 64 |
        @as(u256, limbs[0]);
}

fn u256ToLimbs4(val: u256) [4]u64 {
    return .{
        @truncate(val),
        @truncate(val >> 64),
        @truncate(val >> 128),
        @truncate(val >> 192),
    };
}

// Note: 2D decomposition uses u512 for intermediate products to avoid overflow

// ============================================================================
// 4D Decomposition Table
// ============================================================================

// Auto-generated from bn254_table.sage via power_of_2_decompositions.rs
// Each entry: precomputed 4D decomposition of 2^i for i in 0..253
const POWER_OF_2_DECOMPOSITIONS = @import("glv_table.zig").POWER_OF_2_DECOMPOSITIONS;

// ============================================================================
// Tests
// ============================================================================

test "glv endomorphism G1 identity" {
    const id = G1Affine.identity();
    const result = glvEndomorphismG1(id);
    try std.testing.expect(result.isIdentity());
}

test "glv endomorphism G1 preserves curve" {
    // G1 generator (1, 2)
    const gen = G1Affine.generator();
    const endo = glvEndomorphismG1(gen);
    // endo should be on the curve
    try std.testing.expect(endo.isOnCurve());
}

test "frobenius psi identity" {
    const id = G2Point.identity();
    const result = frobeniusPsiAffine(id, 1);
    try std.testing.expect(result.isIdentity());
}

test "frobenius psi k=0 is identity map" {
    const gen = G2Point.generator();
    const result = frobeniusPsiAffine(gen, 0);
    try std.testing.expect(result.eql(gen));
}

test "4d decomposition scalar 1" {
    const one = Fr.one();
    const decomp = decomposeScalar4D(one);
    // k=1 should decompose as k0=1, k1=k2=k3=0 (entry 2^0 is trivial)
    try std.testing.expectEqual(@as(u64, 1), decomp.k[0][0]);
    try std.testing.expectEqual(@as(u64, 0), decomp.k[1][0]);
    try std.testing.expectEqual(@as(u64, 0), decomp.k[2][0]);
    try std.testing.expectEqual(@as(u64, 0), decomp.k[3][0]);
    try std.testing.expectEqual(false, decomp.signs[0]);
}

test "4d decomposition roundtrip" {
    // Test that k ≡ k0 + k1*λ + k2*λ² + k3*λ³ (mod r)
    // λ_ψ (Frobenius eigenvalue) = 0x6f4d8248eeb859fbf83e9682e87cfd46
    const lambda_bytes = [_]u8{
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x6f, 0x4d, 0x82, 0x48, 0xee, 0xb8, 0x59, 0xfb,
        0xf8, 0x3e, 0x96, 0x82, 0xe8, 0x7c, 0xfd, 0x46,
    };
    const lambda = Fr.fromBytesBE(@ptrCast(&lambda_bytes));
    const lambda2 = lambda.mul(lambda);
    const lambda3 = lambda2.mul(lambda);

    // Test with a specific scalar
    const scalar = Fr.fromU64(0x123456789abcdef0);
    const decomp = decomposeScalar4D(scalar);

    // Reconstruct: k0 + k1*λ + k2*λ² + k3*λ³
    var reconstructed = Fr.zero();
    const powers = [_]Fr{ Fr.one(), lambda, lambda2, lambda3 };
    for (0..4) |i| {
        // Convert limbs to Fr
        var ki = Fr{ .limbs = decomp.k[i] };
        ki = ki.toMontgomery();
        if (decomp.signs[i]) {
            ki = ki.neg();
        }
        reconstructed = reconstructed.add(ki.mul(powers[i]));
    }

    try std.testing.expect(reconstructed.eql(scalar));
}

test "glv scalar mul G1 correctness" {
    const gen = G1Affine.generator();
    const scalar = Fr.fromU64(42);

    // GLV scalar mul
    const glv_result = glvScalarMulG1(gen, scalar).toAffine();

    // Naive scalar mul
    const naive_result = msm_mod.MSM(Fr, Fp).scalarMul(gen, scalar).toAffine();

    try std.testing.expect(glv_result.eql(naive_result));
}

test "glv scalar mul G1 larger scalar" {
    const gen = G1Affine.generator();
    // Use a scalar that will exercise both sub-scalars
    const scalar = Fr.fromU64(0xdeadbeef12345678);

    const glv_result = glvScalarMulG1(gen, scalar).toAffine();
    const naive_result = msm_mod.MSM(Fr, Fp).scalarMul(gen, scalar).toAffine();

    try std.testing.expect(glv_result.eql(naive_result));
}

test "glv scalar mul G2 correctness" {
    const gen = G2Point.generator();
    const scalar = Fr.fromU64(42);

    const glv_result = glvScalarMulG2(gen, scalar).toAffine();
    const naive_result = gen.scalarMul(scalar);

    try std.testing.expect(glv_result.eql(naive_result));
}

test "glv scalar mul G2 larger scalar" {
    const gen = G2Point.generator();
    const scalar = Fr.fromU64(0xdeadbeef12345678);

    const glv_result = glvScalarMulG2(gen, scalar).toAffine();
    const naive_result = gen.scalarMul(scalar);

    try std.testing.expect(glv_result.eql(naive_result));
}
