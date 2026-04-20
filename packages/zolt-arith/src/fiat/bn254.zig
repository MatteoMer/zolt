//! BN254 field types backed by fiat-crypto verified arithmetic.

const FiatField = @import("fiat_field.zig").FiatField;
const params = @import("../curves/bn254/params.zig");

pub const Fr = FiatField(.fr, params.FR_MODULUS, params.FR_R2, params.FR_INV);
pub const Fp = FiatField(.fp, params.FP_MODULUS, params.FP_R2, params.FP_INV);

// =========================================================================
// Tests — cross-validate against MontgomeryField instantiations
// =========================================================================

const testing = @import("std").testing;
const MontFr = @import("../curves/bn254/mod.zig").Fr;
const MontFp = @import("../curves/bn254/mod.zig").Fp;

test "FiatFr.one() matches params.FR_R" {
    try testing.expectEqual(params.FR_R, Fr.one().limbs);
}

test "FiatFp.one() matches params.FP_R" {
    try testing.expectEqual(params.FP_R, Fp.one().limbs);
}

test "FiatFr: 2 * 3 = 6" {
    const two = Fr.fromRaw(.{ 2, 0, 0, 0 });
    const three = Fr.fromRaw(.{ 3, 0, 0, 0 });
    const six = Fr.fromRaw(.{ 6, 0, 0, 0 });
    try testing.expect(Fr.eql(Fr.mul(two, three), six));
}

test "FiatFp: 2 * 3 = 6" {
    const two = Fp.fromRaw(.{ 2, 0, 0, 0 });
    const three = Fp.fromRaw(.{ 3, 0, 0, 0 });
    const six = Fp.fromRaw(.{ 6, 0, 0, 0 });
    try testing.expect(Fp.eql(Fp.mul(two, three), six));
}

test "FiatFr: a * a^-1 = 1" {
    const a = Fr.fromRaw(.{ 0x12345678, 0, 0, 0 });
    const inv_a = Fr.inverse(a) orelse return error.SkipZigTest;
    try testing.expect(Fr.eql(Fr.mul(a, inv_a), Fr.one()));
}

test "FiatFp: a * a^-1 = 1" {
    const a = Fp.fromRaw(.{ 0x12345678, 0, 0, 0 });
    const inv_a = Fp.inverse(a) orelse return error.SkipZigTest;
    try testing.expect(Fp.eql(Fp.mul(a, inv_a), Fp.one()));
}

test "FiatFr: toRaw round-trips fromRaw" {
    const raw: [4]u64 = .{ 0x0102030405060708, 0x0a0b0c0d0e0f0001, 0x1122334455667788, 0x12345678 };
    const m = Fr.fromRaw(raw);
    try testing.expectEqual(raw, Fr.toRaw(m));
}

test "FiatFp: sub borrows correctly (0 - 1 = p - 1)" {
    const z = Fp.zero();
    const o = Fp.fromRaw(.{ 1, 0, 0, 0 });
    const result = Fp.sub(z, o);
    const expected = Fp.fromRaw(.{
        params.FP_MODULUS[0] - 1,
        params.FP_MODULUS[1],
        params.FP_MODULUS[2],
        params.FP_MODULUS[3],
    });
    try testing.expect(Fp.eql(result, expected));
}

// -- Cross-validation against MontgomeryField --

test "Cross: Fr.one() matches Mont" {
    try testing.expectEqual(Fr.one().limbs, MontFr.one().limbs);
}

test "Cross: Fp.one() matches Mont" {
    try testing.expectEqual(Fp.one().limbs, MontFp.one().limbs);
}

test "Cross: Fr add matches Mont" {
    const a = Fr.fromU64(12345);
    const b = Fr.fromU64(67890);
    const ma = MontFr.fromU64(12345);
    const mb = MontFr.fromU64(67890);
    try testing.expectEqual(Fr.add(a, b).limbs, MontFr.add(ma, mb).limbs);
}

test "Cross: Fp add matches Mont" {
    const a = Fp.fromU64(12345);
    const b = Fp.fromU64(67890);
    const ma = MontFp.fromU64(12345);
    const mb = MontFp.fromU64(67890);
    try testing.expectEqual(Fp.add(a, b).limbs, MontFp.add(ma, mb).limbs);
}

test "Cross: Fr sub matches Mont" {
    const a = Fr.fromU64(100);
    const b = Fr.fromU64(200);
    const ma = MontFr.fromU64(100);
    const mb = MontFr.fromU64(200);
    try testing.expectEqual(Fr.sub(a, b).limbs, MontFr.sub(ma, mb).limbs);
}

test "Cross: Fp mul matches Mont" {
    const a = Fp.fromU64(42);
    const b = Fp.fromU64(99);
    const ma = MontFp.fromU64(42);
    const mb = MontFp.fromU64(99);
    try testing.expectEqual(Fp.mul(a, b).limbs, MontFp.mul(ma, mb).limbs);
}

test "Cross: Fr mul matches Mont" {
    const a = Fr.fromU64(42);
    const b = Fr.fromU64(99);
    const ma = MontFr.fromU64(42);
    const mb = MontFr.fromU64(99);
    try testing.expectEqual(Fr.mul(a, b).limbs, MontFr.mul(ma, mb).limbs);
}

test "Cross: Fp square matches Mont" {
    const a = Fp.fromU64(42);
    const ma = MontFp.fromU64(42);
    try testing.expectEqual(Fp.square(a).limbs, MontFp.square(ma).limbs);
}

test "Cross: Fp inverse matches Mont" {
    const a = Fp.fromU64(42);
    const ma = MontFp.fromU64(42);
    const inv = Fp.inverse(a) orelse return error.SkipZigTest;
    const minv = MontFp.inverse(ma) orelse return error.SkipZigTest;
    try testing.expectEqual(inv.limbs, minv.limbs);
}

test "Cross: Fr inverse matches Mont" {
    const a = Fr.fromU64(42);
    const ma = MontFr.fromU64(42);
    const inv = Fr.inverse(a) orelse return error.SkipZigTest;
    const minv = MontFr.inverse(ma) orelse return error.SkipZigTest;
    try testing.expectEqual(inv.limbs, minv.limbs);
}

test "Cross: Fp neg matches Mont" {
    const a = Fp.fromU64(42);
    const ma = MontFp.fromU64(42);
    try testing.expectEqual(Fp.neg(a).limbs, MontFp.neg(ma).limbs);
}

test "Cross: Fp toBytesBE matches Mont" {
    const a = Fp.fromU64(0xdeadbeef);
    const ma = MontFp.fromU64(0xdeadbeef);
    try testing.expectEqual(Fp.toBytesBE(a), MontFp.toBytesBE(ma));
}

test "Cross: Fp fromBytesBE matches Mont" {
    const bytes = MontFp.toBytesBE(MontFp.fromU64(0xdeadbeef));
    const fiat_val = Fp.fromBytesBE(&bytes);
    const mont_val = MontFp.fromBytesBE(&bytes);
    try testing.expectEqual(fiat_val.limbs, mont_val.limbs);
}

test "Cross: Fp sumOfProducts matches Mont" {
    const a0 = Fp.fromU64(3);
    const a1 = Fp.fromU64(7);
    const b0 = Fp.fromU64(11);
    const b1 = Fp.fromU64(13);
    const ma0 = MontFp.fromU64(3);
    const ma1 = MontFp.fromU64(7);
    const mb0 = MontFp.fromU64(11);
    const mb1 = MontFp.fromU64(13);
    const result = Fp.sumOfProducts(.{ a0, a1 }, .{ b0, b1 });
    const mresult = MontFp.sumOfProducts(.{ ma0, ma1 }, .{ mb0, mb1 });
    try testing.expectEqual(result.limbs, mresult.limbs);
}
