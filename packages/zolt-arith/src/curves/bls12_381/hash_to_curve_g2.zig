//! BLS12-381 hash-to-curve for G2.
//!
//! Implements the RFC 9380 §8.8.2 pipeline for the suite
//! `BLS_SIG_BLS12381G2_XMD:SHA-256_SSWU_RO_NUL_`. The pipeline is:
//!
//!   1. `hash_to_field_fp2` (already in `hash_to_field.zig`) produces
//!      two `Fp2` field elements `u0, u1` from `(msg, DST)`.
//!   2. The simplified Shallue-van de Woestijne-Ulas (SSWU) map sends
//!      each `u_i` to a point on the isogenous curve
//!      `E': y² = x³ + 240·u·x + 1012·(1+u)`.
//!   3. The 3-degree isogeny (4 polynomials over Fp2) pushes the SSWU
//!      output from `E'` to BLS12-381 G2 (`y² = x³ + 4(1+u)`).
//!   4. The two G2 points are added together to form `Q'`.
//!   5. Cofactor clearing scales `Q'` by the G2 cofactor `h`, producing
//!      a point in the prime-order `r`-subgroup.
//!
//! All constants here come from the IETF draft and are cross-checked
//! against arkworks `ark-bls12-381` (`g2_swu_iso.rs`). The SSWU
//! algorithm follows Wahby & Boneh (2019) "Fast and simple constant-time
//! hashing to the BLS12-381 elliptic curve" §4.1, which is the
//! "avoiding inversions" optimization arkworks uses too.
//!
//! Performance is correctness-first: cofactor clearing uses naive
//! scalar multiplication by the 8-limb cofactor instead of the
//! ψ-endomorphism shortcut. A future iteration can swap that in
//! without changing the public surface.

const std = @import("std");
const bls12_381 = @import("curve.zig");
const hash_to_field = @import("hash_to_field.zig");
const bigint = @import("../../bigint.zig");

const Fp = bls12_381.Fp;
const Fp2 = bls12_381.Fp2;
const G2Affine = bls12_381.G2Affine;
const G2Projective = bls12_381.G2Projective;

// ---------------------------------------------------------------------------
// Isogenous curve E' constants.
//
//   E': y'² = x'³ + A'·x' + B'
//   A' = 240·u
//   B' = 1012 + 1012·u
//   ZETA = -(2 + u)   (the SSWU non-square parameter)
// ---------------------------------------------------------------------------

/// `A' = 240·u` in Fp2 raw form (NOT Montgomery). We turn it into
/// Montgomery form lazily so the constant block stays declarative.
const ISO_A_RAW: Fp2Raw = .{
    .c0 = .{ 0, 0, 0, 0, 0, 0 },
    .c1 = .{ 240, 0, 0, 0, 0, 0 },
};

/// `B' = 1012 + 1012·u`.
const ISO_B_RAW: Fp2Raw = .{
    .c0 = .{ 1012, 0, 0, 0, 0, 0 },
    .c1 = .{ 1012, 0, 0, 0, 0, 0 },
};

/// `ZETA = -(2 + u) = -2 - u` mod p.
const ISO_ZETA_RAW: Fp2Raw = .{
    .c0 = .{ 0xb9feffffffffaaa9, 0x1eabfffeb153ffff, 0x6730d2a0f6b0f624, 0x64774b84f38512bf, 0x4b1ba7b6434bacd7, 0x1a0111ea397fe69a },
    .c1 = .{ 0xb9feffffffffaaaa, 0x1eabfffeb153ffff, 0x6730d2a0f6b0f624, 0x64774b84f38512bf, 0x4b1ba7b6434bacd7, 0x1a0111ea397fe69a },
};

const Fp2Raw = struct {
    c0: [6]u64,
    c1: [6]u64,
};

inline fn fp2FromRaw(r: Fp2Raw) Fp2 {
    return .{
        .c0 = Fp.fromRaw(r.c0),
        .c1 = Fp.fromRaw(r.c1),
    };
}

fn isoA() Fp2 {
    return fp2FromRaw(ISO_A_RAW);
}

fn isoB() Fp2 {
    return fp2FromRaw(ISO_B_RAW);
}

fn isoZeta() Fp2 {
    return fp2FromRaw(ISO_ZETA_RAW);
}

// ---------------------------------------------------------------------------
// 3-degree isogeny constants. Each entry is a polynomial in `x'`
// (the x-coordinate of the iso curve point) with Fp2 coefficients.
// Index 0 is the constant term; index k is the x'^k coefficient.
//
// Values converted from arkworks `g2_swu_iso.rs::ISOGENY_MAP_TO_G2`
// via Python (decimal → little-endian 6-limb hex). The Rust constants
// are pinned in the IETF draft, section E.3.
// ---------------------------------------------------------------------------

const ISOGENY_X_NUM: [4]Fp2Raw = .{
    .{ .c0 = .{ 0x6238aaaaaaaa97d6, 0x5c2638e343d9c71c, 0x88b58423c50ae15d, 0x32c52d39fd3a042a, 0xbb5b7a9a47d7ed85, 0x05c759507e8e333e }, .c1 = .{ 0x6238aaaaaaaa97d6, 0x5c2638e343d9c71c, 0x88b58423c50ae15d, 0x32c52d39fd3a042a, 0xbb5b7a9a47d7ed85, 0x05c759507e8e333e } },
    .{ .c0 = .{ 0, 0, 0, 0, 0, 0 }, .c1 = .{ 0x26a9ffffffffc71a, 0x1472aaa9cb8d5555, 0x9a208c6b4f20a418, 0x984f87adf7ae0c7f, 0x32126fced787c88f, 0x11560bf17baa99bc } },
    .{ .c0 = .{ 0x26a9ffffffffc71e, 0x1472aaa9cb8d5555, 0x9a208c6b4f20a418, 0x984f87adf7ae0c7f, 0x32126fced787c88f, 0x11560bf17baa99bc }, .c1 = .{ 0x9354ffffffffe38d, 0x0a395554e5c6aaaa, 0xcd104635a790520c, 0xcc27c3d6fbd7063f, 0x190937e76bc3e447, 0x08ab05f8bdd54cde } },
    .{ .c0 = .{ 0x88e2aaaaaaaa5ed1, 0x7098e38d0f671c71, 0x22d6108f142b8575, 0xcb14b4e7f4e810aa, 0xed6dea691f5fb614, 0x171d6541fa38ccfa }, .c1 = .{ 0, 0, 0, 0, 0, 0 } },
};

const ISOGENY_X_DEN: [3]Fp2Raw = .{
    .{ .c0 = .{ 0, 0, 0, 0, 0, 0 }, .c1 = .{ 0xb9feffffffffaa63, 0x1eabfffeb153ffff, 0x6730d2a0f6b0f624, 0x64774b84f38512bf, 0x4b1ba7b6434bacd7, 0x1a0111ea397fe69a } },
    .{ .c0 = .{ 0x000000000000000c, 0, 0, 0, 0, 0 }, .c1 = .{ 0xb9feffffffffaa9f, 0x1eabfffeb153ffff, 0x6730d2a0f6b0f624, 0x64774b84f38512bf, 0x4b1ba7b6434bacd7, 0x1a0111ea397fe69a } },
    .{ .c0 = .{ 0x0000000000000001, 0, 0, 0, 0, 0 }, .c1 = .{ 0, 0, 0, 0, 0, 0 } },
};

const ISOGENY_Y_NUM: [4]Fp2Raw = .{
    .{ .c0 = .{ 0x12cfc71c71c6d706, 0xfc8c25ebf8c92f68, 0xf54439d87d27e500, 0x0f7da5d4a07f649b, 0x59a4c18b076d1193, 0x1530477c7ab4113b }, .c1 = .{ 0x12cfc71c71c6d706, 0xfc8c25ebf8c92f68, 0xf54439d87d27e500, 0x0f7da5d4a07f649b, 0x59a4c18b076d1193, 0x1530477c7ab4113b } },
    .{ .c0 = .{ 0, 0, 0, 0, 0, 0 }, .c1 = .{ 0x6238aaaaaaaa97be, 0x5c2638e343d9c71c, 0x88b58423c50ae15d, 0x32c52d39fd3a042a, 0xbb5b7a9a47d7ed85, 0x05c759507e8e333e } },
    .{ .c0 = .{ 0x26a9ffffffffc71c, 0x1472aaa9cb8d5555, 0x9a208c6b4f20a418, 0x984f87adf7ae0c7f, 0x32126fced787c88f, 0x11560bf17baa99bc }, .c1 = .{ 0x9354ffffffffe38f, 0x0a395554e5c6aaaa, 0xcd104635a790520c, 0xcc27c3d6fbd7063f, 0x190937e76bc3e447, 0x08ab05f8bdd54cde } },
    .{ .c0 = .{ 0xe1b371c71c718b10, 0x4e79097a56dc4bd9, 0xb0e977c69aa27452, 0x761b0f37a1e26286, 0xfbf7043de3811ad0, 0x124c9ad43b6cf79b }, .c1 = .{ 0, 0, 0, 0, 0, 0 } },
};

const ISOGENY_Y_DEN: [4]Fp2Raw = .{
    .{ .c0 = .{ 0xb9feffffffffa8fb, 0x1eabfffeb153ffff, 0x6730d2a0f6b0f624, 0x64774b84f38512bf, 0x4b1ba7b6434bacd7, 0x1a0111ea397fe69a }, .c1 = .{ 0xb9feffffffffa8fb, 0x1eabfffeb153ffff, 0x6730d2a0f6b0f624, 0x64774b84f38512bf, 0x4b1ba7b6434bacd7, 0x1a0111ea397fe69a } },
    .{ .c0 = .{ 0, 0, 0, 0, 0, 0 }, .c1 = .{ 0xb9feffffffffa9d3, 0x1eabfffeb153ffff, 0x6730d2a0f6b0f624, 0x64774b84f38512bf, 0x4b1ba7b6434bacd7, 0x1a0111ea397fe69a } },
    .{ .c0 = .{ 0x0000000000000012, 0, 0, 0, 0, 0 }, .c1 = .{ 0xb9feffffffffaa99, 0x1eabfffeb153ffff, 0x6730d2a0f6b0f624, 0x64774b84f38512bf, 0x4b1ba7b6434bacd7, 0x1a0111ea397fe69a } },
    .{ .c0 = .{ 0x0000000000000001, 0, 0, 0, 0, 0 }, .c1 = .{ 0, 0, 0, 0, 0, 0 } },
};

/// G2 cofactor `h` from the IETF spec. 8 limbs. NOT used for hash-to-
/// curve cofactor clearing — the IETF SSWU_RO suite mandates `h_eff`
/// instead, which produces a *different* point in the same prime-order
/// subgroup. Multiplying by `h` instead of `h_eff` would still land in
/// the r-subgroup, but the result would be a different multiple of the
/// SSWU output and would not match what `blst::min_pk` produces.
///
/// Kept around as documentation; the cofactor clearing path uses
/// `G2_H_EFF` below.
const G2_COFACTOR: [8]u64 = .{
    0xcf1c38e31c7238e5,
    0x1616ec6e786f0c70,
    0x21537e293a6691ae,
    0xa628f1cb4d9e82ef,
    0xa68a205b2e5a7ddf,
    0xcd91de4547085aba,
    0x091d50792876a202,
    0x05d543a95414e7f1,
};

/// `h_eff` for BLS12-381 G2 hash-to-curve, from RFC 9380 §8.8.2 / IETF
/// pairing-friendly-curves draft. 636 bits → 10 limbs.
///
/// The hex value (big-endian):
///   0xbc69f08f2ee75b3584c6a0ea91b352888e2a8e9145ad7689986ff031508ffe13
///     29c2f178731db956d82bf015d1212b02ec0ec69d7477c1ae954cbc06689f6a35
///     9894c0adebbf6b4e8020005aaa95551
///
/// Multiplying an SSWU output by this value lands the point in the
/// prime-order r-subgroup at exactly the multiple `blst::min_pk` (and
/// every other RFC-9380-conformant implementation) lands at. The
/// regular cofactor `h` would also land in the r-subgroup but at a
/// different multiple, so signatures would not cross-verify.
const G2_H_EFF: [10]u64 = .{
    0xe8020005aaa95551,
    0x59894c0adebbf6b4,
    0xe954cbc06689f6a3,
    0x2ec0ec69d7477c1a,
    0x6d82bf015d1212b0,
    0x329c2f178731db95,
    0x9986ff031508ffe1,
    0x88e2a8e9145ad768,
    0x584c6a0ea91b3528,
    0x0bc69f08f2ee75b3,
};

// ---------------------------------------------------------------------------
// Parity helpers (RFC 9380 §4.1 sgn0).
// ---------------------------------------------------------------------------

/// Parity of a raw Fp value: true if the integer (NOT Montgomery form)
/// is odd, false if even.
fn fpIsOddRaw(a: Fp.Element) bool {
    const raw = Fp.toRaw(a);
    return (raw[0] & 1) == 1;
}

/// Parity for an Fp2 element: take the parity of the first non-zero
/// coordinate in the order `(c0, c1)`. A pure-zero element has parity
/// 0. This matches arkworks `parity` and the RFC 9380 sgn0 rule with
/// the "first non-zero coordinate" tie-breaking.
fn fp2Parity(a: Fp2) bool {
    if (!Fp.eql(a.c0, Fp.zero())) return fpIsOddRaw(a.c0);
    if (!Fp.eql(a.c1, Fp.zero())) return fpIsOddRaw(a.c1);
    return false;
}

/// Predicate: is `a` a quadratic residue in Fp2? Implemented by trying
/// the sqrt and checking the round-trip — slow but correct.
fn fp2IsSquare(a: Fp2) bool {
    const root = bls12_381.fp2Sqrt(a) catch return false;
    return Fp2.eql(Fp2.square(root), a);
}

// ---------------------------------------------------------------------------
// Polynomial evaluation in Fp2 (Horner).
// ---------------------------------------------------------------------------

/// Evaluate the Fp2 polynomial whose coefficients are `coeffs` (constant
/// term first, then ascending degrees) at `x` using Horner's method.
fn evalPoly(comptime N: comptime_int, coeffs: [N]Fp2Raw, x: Fp2) Fp2 {
    if (N == 0) return Fp2.zero();
    var result = fp2FromRaw(coeffs[N - 1]);
    var i: usize = N - 1;
    while (i > 0) {
        i -= 1;
        result = Fp2.mul(result, x);
        result = Fp2.add(result, fp2FromRaw(coeffs[i]));
    }
    return result;
}

// ---------------------------------------------------------------------------
// Simplified SWU map (Wahby & Boneh 2019, §4.1; arkworks `swu.rs`).
//
// Given an Fp2 element `u`, produces an affine point on the iso curve
// `E': y² = x³ + 240·u·x + 1012·(1 + u)`. The result is always a valid
// curve point — no failure mode.
// ---------------------------------------------------------------------------

/// Apply the simplified SWU map to a single field element. Returns an
/// affine point on the iso curve `E'`. Caller is responsible for then
/// applying the isogeny to push the point onto BLS12-381 G2.
pub fn sswuMapToCurve(u: Fp2) struct { x: Fp2, y: Fp2 } {
    const a = isoA();
    const b = isoB();
    const zeta = isoZeta();

    // tv1 = ZETA · u²
    const u_sq = Fp2.square(u);
    const zeta_u2 = Fp2.mul(zeta, u_sq);
    // ta = (ZETA·u²)² + (ZETA·u²) = Z²u⁴ + Zu²
    const ta = Fp2.add(Fp2.square(zeta_u2), zeta_u2);
    // num_x1 = B · (ta + 1)
    const num_x1 = Fp2.mul(b, Fp2.add(ta, Fp2.one()));
    // div = if ta == 0 then A·ZETA else A·(-ta)
    const div = if (Fp2.eql(ta, Fp2.zero()))
        Fp2.mul(a, zeta)
    else
        Fp2.mul(a, Fp2.neg(ta));

    // num²_x1 = num_x1²
    const num2_x1 = Fp2.square(num_x1);
    // div² = div²
    const div2 = Fp2.square(div);
    // div³ = div² · div
    const div3 = Fp2.mul(div2, div);
    // num_gx1 = (num²_x1 + A · div²) · num_x1 + B · div³
    const num_gx1 = Fp2.add(
        Fp2.mul(Fp2.add(num2_x1, Fp2.mul(a, div2)), num_x1),
        Fp2.mul(b, div3),
    );

    // num_x2 = ZETA·u² · num_x1   (x2 = ZETA·u² · x1, same div)
    const num_x2 = Fp2.mul(zeta_u2, num_x1);

    // gx1 = num_gx1 / div³
    const div3_inv = Fp2.inv(div3);
    const gx1 = Fp2.mul(num_gx1, div3_inv);

    // Try to take sqrt(gx1). If it's a square, use (x1, sqrt(gx1)).
    // Otherwise use (x2, ZETA·u·u² · sqrt(ZETA · gx1)).
    var x_num: Fp2 = undefined;
    var y: Fp2 = undefined;
    if (fp2IsSquare(gx1)) {
        x_num = num_x1;
        y = bls12_381.fp2Sqrt(gx1) catch unreachable;
    } else {
        // ZETA · gx1 must be a square (by the structure of the SWU map).
        const zeta_gx1 = Fp2.mul(zeta, gx1);
        const y1 = bls12_381.fp2Sqrt(zeta_gx1) catch unreachable;
        // y2 = ZETA · u² · u · y1 = zeta_u2 · u · y1
        y = Fp2.mul(Fp2.mul(zeta_u2, u), y1);
        x_num = num_x2;
    }

    // x = num_x / div
    const x = Fp2.mul(x_num, Fp2.inv(div));
    // Final y-sign tweak: parity(y) must equal parity(u). RFC 9380 4.1.
    if (fp2Parity(y) != fp2Parity(u)) {
        y = Fp2.neg(y);
    }
    return .{ .x = x, .y = y };
}

// ---------------------------------------------------------------------------
// Isogeny push from E' to G2.
// ---------------------------------------------------------------------------

/// Apply the 3-degree isogeny to push an iso curve point onto BLS12-381
/// G2. Identity input maps to identity output. Identity here means
/// the iso point's denominator polynomial vanishes at the point's `x`,
/// which we treat as the point at infinity.
pub fn isogenyMap(p_x: Fp2, p_y: Fp2) G2Affine {
    const x_num_val = evalPoly(4, ISOGENY_X_NUM, p_x);
    const x_den_val = evalPoly(3, ISOGENY_X_DEN, p_x);
    const y_num_val = evalPoly(4, ISOGENY_Y_NUM, p_x);
    const y_den_val = evalPoly(4, ISOGENY_Y_DEN, p_x);

    // If either denominator vanishes the iso point lies on the kernel
    // of the isogeny — map to identity.
    if (Fp2.eql(x_den_val, Fp2.zero()) or Fp2.eql(y_den_val, Fp2.zero())) {
        return G2Affine.identity();
    }

    const new_x = Fp2.mul(x_num_val, Fp2.inv(x_den_val));
    const new_y = Fp2.mul(p_y, Fp2.mul(y_num_val, Fp2.inv(y_den_val)));
    return .{ .x = new_x, .y = new_y, .infinity = false };
}

// ---------------------------------------------------------------------------
// Cofactor clearing.
// ---------------------------------------------------------------------------

/// Multiply a G2Affine point by `h_eff` from the IETF SSWU_RO suite.
/// The result is in the prime-order r-subgroup at the same multiple
/// `blst::min_pk` lands at, so the resulting hash-to-curve output is
/// byte-for-byte cross-implementation-compatible.
///
/// Implementation: naive double-and-add scalar multiplication on
/// `G2Projective`. `h_eff` is a 636-bit scalar, so this is ~636
/// doublings + ~318 additions. The faster ψ-endomorphism shortcut from
/// the IETF draft can land later as a drop-in replacement — the public
/// surface stays the same.
pub fn clearCofactor(p: G2Affine) G2Affine {
    if (p.infinity) return p;
    const proj = G2Projective.fromAffine(p);
    return proj.mul(10, G2_H_EFF).toAffine();
}

// ---------------------------------------------------------------------------
// Top-level entry point.
// ---------------------------------------------------------------------------

/// `BLS_SIG_BLS12381G2_XMD:SHA-256_SSWU_RO_NUL_` per draft-irtf-cfrg-bls-signature-05.
pub const DST_BLS_SIG_NUL: []const u8 = "BLS_SIG_BLS12381G2_XMD:SHA-256_SSWU_RO_NUL_";

/// Hash a message to BLS12-381 G2 using the RFC 9380 `SSWU_RO`
/// (random-oracle) variant. Produces a point in the prime-order
/// r-subgroup, ready for use as a BLS signature input.
pub fn hashToG2(msg: []const u8, dst: []const u8) hash_to_field.ExpandError!G2Affine {
    var us: [2]Fp2 = undefined;
    try hash_to_field.hash_to_field_fp2(&us, msg, dst);

    // Map each of u0 and u1 through SSWU and the isogeny, then add.
    const r0 = sswuMapToCurve(us[0]);
    const q0 = isogenyMap(r0.x, r0.y);

    const r1 = sswuMapToCurve(us[1]);
    const q1 = isogenyMap(r1.x, r1.y);

    // Affine addition handles the (q0 == q1) and (q0 == -q1) edge cases
    // naturally; the result lives on G2 but generally NOT in the
    // prime-order subgroup, so cofactor clearing comes next.
    const sum = q0.add(q1);
    return clearCofactor(sum);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const testing = std.testing;

test "sswuMapToCurve: produces a point on the iso curve E'" {
    // E': y² = x³ + 240·u·x + 1012·(1+u). Pick a small Fp2 element and
    // verify the SSWU output satisfies the iso curve equation.
    const u = Fp2{
        .c0 = Fp.fromRaw(.{ 7, 0, 0, 0, 0, 0 }),
        .c1 = Fp.fromRaw(.{ 11, 0, 0, 0, 0, 0 }),
    };
    const point = sswuMapToCurve(u);
    const a = isoA();
    const b = isoB();
    const lhs = Fp2.square(point.y);
    const rhs = Fp2.add(
        Fp2.add(Fp2.mul(Fp2.square(point.x), point.x), Fp2.mul(a, point.x)),
        b,
    );
    try testing.expect(Fp2.eql(lhs, rhs));
}

test "sswuMapToCurve: deterministic" {
    const u = Fp2{
        .c0 = Fp.fromRaw(.{ 0xdeadbeef, 0, 0, 0, 0, 0 }),
        .c1 = Fp.fromRaw(.{ 0xfeedface, 0, 0, 0, 0, 0 }),
    };
    const a = sswuMapToCurve(u);
    const b = sswuMapToCurve(u);
    try testing.expect(Fp2.eql(a.x, b.x));
    try testing.expect(Fp2.eql(a.y, b.y));
}

test "sswuMapToCurve: distinct inputs produce distinct outputs" {
    const ua = Fp2{
        .c0 = Fp.fromRaw(.{ 1, 0, 0, 0, 0, 0 }),
        .c1 = Fp.fromRaw(.{ 0, 0, 0, 0, 0, 0 }),
    };
    const ub = Fp2{
        .c0 = Fp.fromRaw(.{ 2, 0, 0, 0, 0, 0 }),
        .c1 = Fp.fromRaw(.{ 0, 0, 0, 0, 0, 0 }),
    };
    const a = sswuMapToCurve(ua);
    const b = sswuMapToCurve(ub);
    try testing.expect(!Fp2.eql(a.x, b.x) or !Fp2.eql(a.y, b.y));
}

test "isogenyMap: pushes iso curve point to G2" {
    // Run SSWU then push through the isogeny. The result must lie on
    // the actual G2 curve y² = x³ + 4(1+u).
    const u = Fp2{
        .c0 = Fp.fromRaw(.{ 0x123456, 0, 0, 0, 0, 0 }),
        .c1 = Fp.fromRaw(.{ 0xabcdef, 0, 0, 0, 0, 0 }),
    };
    const iso_point = sswuMapToCurve(u);
    const g2 = isogenyMap(iso_point.x, iso_point.y);
    try testing.expect(g2.isOnCurve());
}

test "clearCofactor: result is in the prime-order subgroup" {
    // Cofactor-clearing the G2 generator must leave it in the
    // r-subgroup (which it already is), and the result should still
    // be on the curve. This is a smoke test for the 8-limb scalar
    // multiplication path.
    const g2 = bls12_381.g2Generator();
    const cleared = clearCofactor(g2);
    try testing.expect(cleared.isOnCurve());
    try testing.expect(bls12_381.isInG2Subgroup(cleared));
}

test "hashToG2: result lies on G2 in the prime-order subgroup" {
    const point = try hashToG2("hello world", DST_BLS_SIG_NUL);
    try testing.expect(point.isOnCurve());
    try testing.expect(bls12_381.isInG2Subgroup(point));
}

test "hashToG2: deterministic for same (msg, dst)" {
    const a = try hashToG2("test message", DST_BLS_SIG_NUL);
    const b = try hashToG2("test message", DST_BLS_SIG_NUL);
    try testing.expect(G2Affine.eql(a, b));
}

test "hashToG2: distinct messages produce distinct points" {
    const a = try hashToG2("message A", DST_BLS_SIG_NUL);
    const b = try hashToG2("message B", DST_BLS_SIG_NUL);
    try testing.expect(!G2Affine.eql(a, b));
}

test "hashToG2: empty message works" {
    const point = try hashToG2("", DST_BLS_SIG_NUL);
    try testing.expect(point.isOnCurve());
    try testing.expect(bls12_381.isInG2Subgroup(point));
}
