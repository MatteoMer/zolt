//! BN254 curve type bundle.
//!
//! Phase 1: field types only (Fr, Fp). These are instantiations of the
//! generic `MontgomeryField(N, ...)` factory with BN254's specific
//! moduli and Montgomery constants.
//!
//! Later phases will add: Fp2, Fp6, Fp12, G1Affine, G2Affine, Pairing.

const MontgomeryField = @import("../montgomery_field.zig").MontgomeryField;
const params = @import("params.zig");

/// BN254 scalar field element (4-limb, 254 bits).
pub const Fr = MontgomeryField(4, params.FR_MODULUS, params.FR_R2, params.FR_INV);

/// BN254 base field element (4-limb, 254 bits).
pub const Fp = MontgomeryField(4, params.FP_MODULUS, params.FP_R2, params.FP_INV);

// =========================================================================
// Tests — verify the generic instantiation matches the OG's constants
// =========================================================================

const testing = @import("std").testing;

test "BN254 Fr.one() matches BN254_R" {
    const o = Fr.one();
    try testing.expectEqual(params.FR_R, o.limbs);
}

test "BN254 Fp.one() matches BN254_FP_R" {
    const o = Fp.one();
    try testing.expectEqual(params.FP_R, o.limbs);
}

test "BN254 Fr: 2 * 3 = 6" {
    const two = Fr.fromRaw(.{ 2, 0, 0, 0 });
    const three = Fr.fromRaw(.{ 3, 0, 0, 0 });
    const six = Fr.fromRaw(.{ 6, 0, 0, 0 });
    try testing.expect(Fr.eql(Fr.montgomeryMul(two, three), six));
}

test "BN254 Fp: a * a^-1 = 1" {
    const a = Fp.fromRaw(.{ 0x12345678, 0, 0, 0 });
    const inv_a = Fp.inverse(a) orelse return error.SkipZigTest;
    try testing.expect(Fp.eql(Fp.montgomeryMul(a, inv_a), Fp.one()));
}

test "BN254 Fr: .limbs access works for backward compat" {
    const a = Fr.fromU64(42);
    _ = a.limbs[0];
    _ = a.limbs[3];
}

// Cross-validation: generic Fp vs OG BN254BaseField (now aliased)
const field_mod = @import("../../field/mod.zig");
const OGFp = field_mod.BN254BaseField;

test "Cross-validate: generic Fp one() matches OG BN254BaseField one()" {
    try testing.expectEqual(Fp.one().limbs, OGFp.one().limbs);
}

test "Cross-validate: generic Fp add matches OG" {
    const a = Fp.fromU64(12345);
    const b = Fp.fromU64(67890);
    const og_a = OGFp.fromU64(12345);
    const og_b = OGFp.fromU64(67890);
    try testing.expectEqual(Fp.add(a, b).limbs, OGFp.add(og_a, og_b).limbs);
}

test "Cross-validate: generic Fp mul matches OG" {
    const a = Fp.fromU64(42);
    const b = Fp.fromU64(99);
    const og_a = OGFp.fromU64(42);
    const og_b = OGFp.fromU64(99);
    try testing.expectEqual(Fp.mul(a, b).limbs, OGFp.montgomeryMul(og_a, og_b).limbs);
}

test "Cross-validate: generic Fp inverse matches OG" {
    const a = Fp.fromU64(42);
    const og_a = OGFp.fromU64(42);
    const inv = Fp.inverse(a) orelse return error.SkipZigTest;
    const og_inv = OGFp.inverse(og_a) orelse return error.SkipZigTest;
    try testing.expectEqual(inv.limbs, og_inv.limbs);
}

test "Cross-validate: generic Fp sumOfProducts matches OG" {
    const a0 = Fp.fromU64(3);
    const a1 = Fp.fromU64(7);
    const b0 = Fp.fromU64(11);
    const b1 = Fp.fromU64(13);
    const og_a0 = OGFp.fromU64(3);
    const og_a1 = OGFp.fromU64(7);
    const og_b0 = OGFp.fromU64(11);
    const og_b1 = OGFp.fromU64(13);
    const result = Fp.sumOfProducts(.{ a0, a1 }, .{ b0, b1 });
    const og_result = OGFp.sumOfProducts(.{ og_a0, og_a1 }, .{ og_b0, og_b1 });
    try testing.expectEqual(result.limbs, og_result.limbs);
}

test "Cross-validate: generic Fp double matches OG" {
    const a = Fp.fromU64(42);
    const og_a = OGFp.fromU64(42);
    try testing.expectEqual(Fp.double(a).limbs, OGFp.double(og_a).limbs);
}

test "Cross-validate: generic Fp toBytesBE matches OG" {
    const a = Fp.fromU64(0xdeadbeef);
    const og_a = OGFp.fromU64(0xdeadbeef);
    try testing.expectEqual(Fp.toBytesBE(a), OGFp.toBytesBE(og_a));
}
