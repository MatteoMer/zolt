//! BLS12-381 instantiations of the generic field machinery.
//!
//! This file is the bridge between `zolt_arith.field` and the concrete
//! Hyli BLS surface. It pins the BLS12-381 base field (`Fp`) constants
//! and exposes a strongly-typed field instance the rest of the package
//! (and Zyli's adapter) consumes.
//!
//! BLS12-381 parameters:
//!
//!   - `p` (base field prime, 381 bits):
//!     `0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab`
//!
//!   - Curve embedding degree 12, optimal Ate pairing-friendly. The
//!     scalar field `Fr` is 255 bits and lives in a separate type.
//!
//! All Montgomery constants come from the standard `blst` reference
//! implementation. They are pinned in source so a regression in the
//! field machinery surfaces immediately rather than after a hand-typed
//! constant drifts.

const std = @import("std");
const field = @import("field.zig");
const bigint = @import("../../bigint.zig");

/// BLS12-381 base field prime, little-endian limbs.
///
///   p = 0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf
///         6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab
pub const FP_MODULUS: [6]u64 = .{
    0xb9feffffffffaaab,
    0x1eabfffeb153ffff,
    0x6730d2a0f6b0f624,
    0x64774b84f38512bf,
    0x4b1ba7b6434bacd7,
    0x1a0111ea397fe69a,
};

/// `R^2 mod p` where `R = 2^384`. Used to convert raw integers into
/// Montgomery form via `montMul(raw, R2)`. From the blst constants.
///
///   R^2 = 0x11988fe592cae3aa9a793e85b519952d67eb88a9939d83c0
///           8de5476c4c95b6d50a76e6a609d104f1f4df1f341c341746
pub const FP_R2: [6]u64 = .{
    0xf4df1f341c341746,
    0x0a76e6a609d104f1,
    0x8de5476c4c95b6d5,
    0x67eb88a9939d83c0,
    0x9a793e85b519952d,
    0x11988fe592cae3aa,
};

/// `-p^{-1} mod 2^64`. Drives the per-limb reduction in CIOS Montgomery
/// multiplication. From the blst constants.
pub const FP_N_PRIME: u64 = 0x89f3fffcfffcfffd;

/// BLS12-381 base field `Fp = ℤ / pℤ`. Elements are stored in
/// Montgomery form and indexed by 6-limb arrays.
pub const Fp = field.MontgomeryField(6, FP_MODULUS, FP_R2, FP_N_PRIME);

/// `(p + 1) / 4` derived at comptime from `FP_MODULUS`. BLS12-381's
/// base prime has `p ≡ 3 (mod 4)` (the lowest byte of p is 0xab; note
/// 0xab mod 4 = 3), which lets us compute square roots via
/// `a^((p+1)/4)` without falling back to Tonelli-Shanks.
pub const FP_P_PLUS_1_OVER_4: [6]u64 = blk: {
    @setEvalBranchQuota(10000);
    var v = FP_MODULUS;
    // p + 1: the low limb ends in 0xb9feffffffffaaab, so adding 1 gives
    // 0xb9feffffffffaaac with no carry into the next limb.
    v[0] += 1;
    // Right-shift the whole 6-limb integer by 2 bits.
    var i: usize = 0;
    while (i < 5) : (i += 1) {
        v[i] = (v[i] >> 2) | (v[i + 1] << 62);
    }
    v[5] >>= 2;
    break :blk v;
};

/// `(p - 1) / 3` derived at comptime via long division by 3. BLS12-381's
/// base prime has `p ≡ 1 (mod 3)` — that's exactly what makes `1+u`
/// a non-cube in Fp2 and lets us pick `v³ = 1+u` as the Fp6 modulus.
/// The constant is the exponent for the Frobenius coefficient
/// `γ₁ = (1+u)^((p-1)/3)`.
pub const FP_P_MINUS_1_OVER_3: [6]u64 = blk: {
    @setEvalBranchQuota(10000);
    var v = FP_MODULUS;
    v[0] -= 1; // p - 1
    // Long division by 3 from MSB to LSB.
    var rem: u128 = 0;
    var i: usize = 6;
    while (i > 0) {
        i -= 1;
        const word = (rem << 64) | @as(u128, v[i]);
        v[i] = @intCast(word / 3);
        rem = word % 3;
    }
    // Sanity: 3 must divide p - 1 exactly.
    std.debug.assert(rem == 0);
    break :blk v;
};

/// Frobenius coefficient `γ₁ = (1+u)^((p-1)/3)` for the Fp6 over Fp2
/// tower with non-residue `1+u`. Used by `fp6Frobenius` and (squared)
/// for the v² coefficient. Computed once on first call rather than
/// embedded as a hand-typed constant — if the comptime division of
/// (p-1)/3 ever drifts, this catches it via the existing tower
/// relations.
pub fn fp6FrobeniusGamma1() Fp2 {
    const one_plus_u: Fp2 = .{ .c0 = Fp.one(), .c1 = Fp.one() };
    return fp2Pow(one_plus_u, 6, FP_P_MINUS_1_OVER_3);
}

/// `(p - 1) / 6` derived at comptime by halving `FP_P_MINUS_1_OVER_3`.
/// BLS12-381's prime satisfies `p ≡ 1 (mod 6)` (CRT of `p ≡ 3 mod 4`
/// and `p ≡ 1 mod 3` gives `p ≡ 7 mod 12`, so `p mod 6 = 1`), which
/// makes the division exact.
pub const FP_P_MINUS_1_OVER_6: [6]u64 = blk: {
    @setEvalBranchQuota(10000);
    var v = FP_P_MINUS_1_OVER_3;
    // Right-shift by 1 (divide by 2). Walk LSB to MSB collecting the
    // outgoing bit from the next limb.
    var i: usize = 0;
    while (i < 5) : (i += 1) {
        v[i] = (v[i] >> 1) | (v[i + 1] << 63);
    }
    v[5] >>= 1;
    break :blk v;
};

/// Frobenius coefficient `γ_w = (1+u)^((p-1)/6)` for the Fp12 over Fp6
/// tower. Drives the action of `φ` on the `w` element, since
/// `w² = v` and `v^p = γ₁·v` give `w^(p-1) = (1+u)^((p-1)/6)`.
pub fn fp12FrobeniusGammaW() Fp2 {
    const one_plus_u: Fp2 = .{ .c0 = Fp.one(), .c1 = Fp.one() };
    return fp2Pow(one_plus_u, 6, FP_P_MINUS_1_OVER_6);
}

/// Frobenius endomorphism in Fp6: `a → a^p`. The action on the basis
/// elements `(1, v, v²)` of Fp6 over Fp2 is:
///
///   φ(1)  = 1
///   φ(v)  = γ₁ · v
///   φ(v²) = γ₁² · v²
///
/// where `γ₁ = (1+u)^((p-1)/3)` lives in Fp2. Each Fp2 coefficient
/// also gets its own Fp2 Frobenius applied.
pub fn fp6Frobenius(a: Fp6) Fp6 {
    const gamma1 = fp6FrobeniusGamma1();
    const gamma1_sq = Fp2.square(gamma1);
    return .{
        .c0 = fp2Frobenius(a.c0),
        .c1 = Fp2.mul(fp2Frobenius(a.c1), gamma1),
        .c2 = Fp2.mul(fp2Frobenius(a.c2), gamma1_sq),
    };
}

/// Multiply an Fp6 element by an Fp2 scalar (lifted as `(b, 0, 0)`).
/// Componentwise multiplication of each Fp2 coefficient by `b`.
fn fp6MulByFp2(a: Fp6, b: Fp2) Fp6 {
    return .{
        .c0 = Fp2.mul(a.c0, b),
        .c1 = Fp2.mul(a.c1, b),
        .c2 = Fp2.mul(a.c2, b),
    };
}

/// Frobenius endomorphism in Fp12: `a → a^p`. For
/// `a = c₀ + c₁·w` with `c₀, c₁ ∈ Fp6`:
///
///   φ(c₀ + c₁·w) = φ(c₀) + φ(c₁)·γ_w·w
///
/// where `γ_w = (1+u)^((p-1)/6)` lives in Fp2 (which is a subfield of
/// Fp6). The c₁ side does an Fp6 Frobenius then a scalar multiply by
/// `γ_w`.
pub fn fp12Frobenius(a: Fp12) Fp12 {
    const gamma_w = fp12FrobeniusGammaW();
    return .{
        .c0 = fp6Frobenius(a.c0),
        .c1 = fp6MulByFp2(fp6Frobenius(a.c1), gamma_w),
    };
}

/// Squared Frobenius `a → a^(p²)`. Two `fp12Frobenius` calls compose
/// into the squared form, which is what the easy part of the final
/// exponentiation needs.
pub fn fp12FrobeniusSquared(a: Fp12) Fp12 {
    return fp12Frobenius(fp12Frobenius(a));
}

/// Frobenius cubed: `a → a^(p³)`. Composes Frobenius and Frobenius².
pub fn fp12FrobeniusCubed(a: Fp12) Fp12 {
    return fp12Frobenius(fp12FrobeniusSquared(a));
}

/// "Easy" part of the BLS12-381 final exponentiation:
/// `f^((p^6 - 1)(p^2 + 1))`.
///
/// This decomposes into operations that don't need the BLS x parameter:
///
/// 1. `f^(p^6 - 1)` = `f^(p^6) · f^(-1)` = `conjugate(f) · inv(f)`.
///    (For Fp12, raising to the `p^6` power equals conjugation —
///    that's the property of the cyclotomic subgroup.)
///
/// 2. `f^(p^2 + 1)` = `f^(p^2) · f` = `frobeniusSquared(f) · f`.
///
/// After the easy part the result is in the cyclotomic subgroup, and
/// the "hard" part (`f^((p^4 - p^2 + 1) / r)`) finishes the job.
///
/// The slow `Fp12.inv` is acceptable here because the easy part is
/// only invoked once per pairing (not per Miller loop iteration).
pub fn fp12FinalExpEasy(f: Fp12) Fp12 {
    // Step 1: f1 = f^(p^6 - 1) = conjugate(f) * inv(f).
    const conj = Fp12.conjugate(f);
    const inv_f = Fp12.inv(f);
    const f1 = Fp12.mul(conj, inv_f);
    // Step 2: f2 = f1^(p^2 + 1) = frobeniusSquared(f1) * f1.
    const f1_p2 = fp12FrobeniusSquared(f1);
    return Fp12.mul(f1_p2, f1);
}

/// Square root of `a` in Fp via `a^((p+1)/4)`. The caller is responsible
/// for verifying that the returned value squares back to `a` —
/// non-residues yield a value whose square is `-a` instead.
///
/// Returns the canonical "positive" root; the caller picks the y-sign
/// when decoding compressed points.
pub fn fpSqrt(a: Fp.Element) Fp.Element {
    return Fp.pow(a, FP_P_PLUS_1_OVER_4);
}

/// Predicate: does `candidate^2 == a`? Cheap check the caller uses to
/// rule out non-residues after `fpSqrt`.
pub fn fpIsSquareRoot(a: Fp.Element, candidate: Fp.Element) bool {
    return Fp.eql(Fp.square(candidate), a);
}

/// `2⁻¹ mod p` in Montgomery form. Computed once at comptime via
/// Fermat — `inv(2)` is a constant we use repeatedly in Fp2 sqrt.
pub const FP_TWO_INV: Fp.Element = blk: {
    @setEvalBranchQuota(200000);
    const two = Fp.fromRaw(.{ 2, 0, 0, 0, 0, 0 });
    break :blk Fp.inv(two);
};

/// `2^256 mod p` as an Fp element. Used by `fpFromBytes64Be` to
/// reduce a 512-bit value modulo `p` via the identity
/// `(high·2^256 + low) mod p = ((high mod p)·(2^256 mod p) + (low mod p)) mod p`.
///
/// Since `p > 2^380 > 2^256`, both `2^256 mod p == 2^256` and any
/// 32-byte high/low chunk is already < p, so the reduction collapses
/// to a single Fp multiplication and addition.
pub const FP_2_TO_256: Fp.Element = blk: {
    @setEvalBranchQuota(50000);
    // 2^256 in raw 6-limb LE form: limb[4] = 1, everything else 0.
    break :blk Fp.fromRaw(.{ 0, 0, 0, 0, 1, 0 });
};

/// Reduce a 64-byte big-endian integer modulo `p` and return the
/// result in Montgomery form. Used by `hash_to_field` to take 64-byte
/// uniform chunks (`L = 64` for BLS12-381 with `k = 128`) and turn
/// them into Fp elements.
pub fn fpFromBytes64Be(bytes: *const [64]u8) Fp.Element {
    // Big-endian: the first 32 bytes are the high half.
    var high_buf: [48]u8 = .{0} ** 48;
    var low_buf: [48]u8 = .{0} ** 48;
    // Right-align each 32-byte chunk into a 48-byte buffer so
    // bigint.fromBytesBe interprets it as a 6-limb integer.
    @memcpy(high_buf[16..48], bytes[0..32]);
    @memcpy(low_buf[16..48], bytes[32..64]);
    const high_raw = bigint.fromBytesBe(6, &high_buf);
    const low_raw = bigint.fromBytesBe(6, &low_buf);
    // Both halves are < 2^256 < p, so they're already canonical Fp
    // values; convert into Montgomery form directly.
    const high_fp = Fp.fromRaw(high_raw);
    const low_fp = Fp.fromRaw(low_raw);
    return Fp.add(Fp.montMul(high_fp, FP_2_TO_256), low_fp);
}

/// `a / 2` in Fp.
pub fn fpHalve(a: Fp.Element) Fp.Element {
    return Fp.montMul(a, FP_TWO_INV);
}

// ---------------------------------------------------------------------------
// BLS12-381 scalar field Fr.
// ---------------------------------------------------------------------------

/// BLS12-381 scalar field prime, little-endian limbs (4 × 64 = 256
/// bits, but the actual prime is 255 bits).
///
///   r = 0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001
pub const FR_MODULUS: [4]u64 = .{
    0xffffffff00000001,
    0x53bda402fffe5bfe,
    0x3339d80809a1d805,
    0x73eda753299d7d48,
};

/// `R² mod r` where `R = 2^256`. Pinned from the standard `blst`
/// constants.
pub const FR_R2: [4]u64 = .{
    0xc999e990f3f29c6d,
    0x2b6cedcb87925c23,
    0x05d314967254398f,
    0x0748d9d99f59ff11,
};

/// `-r⁻¹ mod 2^64`.
pub const FR_N_PRIME: u64 = 0xfffffffeffffffff;

/// BLS12-381 scalar field `Fr = ℤ / rℤ`. Validators sign with scalars
/// drawn from this field, and the curve point group order is exactly
/// `r`. Stored in Montgomery form.
pub const Fr = field.MontgomeryField(4, FR_MODULUS, FR_R2, FR_N_PRIME);

/// Check that an affine G1 point lies in the prime-order r-subgroup.
///
/// The simplest correct check is `r·P == identity`. Routed through
/// `G1Projective.mul` so the 255-bit scalar multiplication doesn't
/// drag a Fermat inversion through every step. Faster checks like
/// Bowe's endomorphism trick can land later — they need the GLS /
/// GLV machinery that hasn't been written yet.
///
/// Identity is in every subgroup by definition.
pub fn isInG1Subgroup(p: G1Affine) bool {
    if (p.isIdentity()) return true;
    return G1Projective.fromAffine(p).mul(4, FR_MODULUS).isIdentity();
}

/// Same shape as `isInG1Subgroup` but for G2.
pub fn isInG2Subgroup(p: G2Affine) bool {
    if (p.isIdentity()) return true;
    return G2Projective.fromAffine(p).mul(4, FR_MODULUS).isIdentity();
}

// ---------------------------------------------------------------------------
// G1 Jacobian projective coordinates.
//
// A Jacobian point (X, Y, Z) represents the affine point (X/Z², Y/Z³).
// The identity is signalled by Z = 0. Doubling and addition are
// inversion-free; only the final affine projection needs an inverse.
//
// This is the representation the Miller loop will hold its G2
// accumulator in — affine arithmetic is too slow because every step
// runs Fermat inversion.
// ---------------------------------------------------------------------------

pub const G1Projective = struct {
    x: Fp.Element,
    y: Fp.Element,
    z: Fp.Element,

    pub fn identity() G1Projective {
        return .{ .x = Fp.zero(), .y = Fp.one(), .z = Fp.zero() };
    }

    pub fn isIdentity(self: G1Projective) bool {
        return Fp.eql(self.z, Fp.zero());
    }

    pub fn fromAffine(p: G1Affine) G1Projective {
        if (p.infinity) return identity();
        return .{ .x = p.x, .y = p.y, .z = Fp.one() };
    }

    /// Project a Jacobian point back to affine via one Fp inversion.
    pub fn toAffine(self: G1Projective) G1Affine {
        if (self.isIdentity()) return G1Affine.identity();
        const z_inv = Fp.inv(self.z);
        const z_inv_sq = Fp.square(z_inv);
        const z_inv_cubed = Fp.montMul(z_inv_sq, z_inv);
        return .{
            .x = Fp.montMul(self.x, z_inv_sq),
            .y = Fp.montMul(self.y, z_inv_cubed),
            .infinity = false,
        };
    }

    /// Doubling: standard `dbl-2009-l` formulas for `a = 0` short
    /// Weierstrass curves. ~3 squarings + 4 multiplications.
    pub fn double(self: G1Projective) G1Projective {
        if (self.isIdentity()) return self;
        // A = X²
        const A = Fp.square(self.x);
        // B = Y²
        const B = Fp.square(self.y);
        // C = B²
        const C = Fp.square(B);
        // D = 2((X + B)² - A - C)
        const x_plus_b = Fp.add(self.x, B);
        const x_plus_b_sq = Fp.square(x_plus_b);
        const D_inner = Fp.sub(Fp.sub(x_plus_b_sq, A), C);
        const D = Fp.add(D_inner, D_inner);
        // E = 3A
        const E = Fp.add(Fp.add(A, A), A);
        // F = E²
        const F = Fp.square(E);
        // X' = F - 2D
        const x3 = Fp.sub(F, Fp.add(D, D));
        // Y' = E·(D - X') - 8C
        const eight_c = blk: {
            const two_c = Fp.add(C, C);
            const four_c = Fp.add(two_c, two_c);
            break :blk Fp.add(four_c, four_c);
        };
        const y3 = Fp.sub(Fp.montMul(E, Fp.sub(D, x3)), eight_c);
        // Z' = 2 Y Z
        const z3 = Fp.add(Fp.montMul(self.y, self.z), Fp.montMul(self.y, self.z));
        return .{ .x = x3, .y = y3, .z = z3 };
    }

    /// Addition: `add-2007-bl` formulas. Falls back to `double` when
    /// the inputs are equal and to the identity when they cancel.
    /// ~12 multiplications + 4 squarings; not as fast as the
    /// specialized mixed-add but covers every case for now.
    pub fn add(p: G1Projective, q: G1Projective) G1Projective {
        if (p.isIdentity()) return q;
        if (q.isIdentity()) return p;
        // Z1Z1 = Z1²
        const Z1Z1 = Fp.square(p.z);
        // Z2Z2 = Z2²
        const Z2Z2 = Fp.square(q.z);
        // U1 = X1 · Z2Z2
        const U1 = Fp.montMul(p.x, Z2Z2);
        // U2 = X2 · Z1Z1
        const U2 = Fp.montMul(q.x, Z1Z1);
        // S1 = Y1 · Z2 · Z2Z2
        const S1 = Fp.montMul(Fp.montMul(p.y, q.z), Z2Z2);
        // S2 = Y2 · Z1 · Z1Z1
        const S2 = Fp.montMul(Fp.montMul(q.y, p.z), Z1Z1);
        if (Fp.eql(U1, U2)) {
            if (Fp.eql(S1, S2)) return p.double();
            return identity();
        }
        // H = U2 - U1
        const H = Fp.sub(U2, U1);
        // I = (2H)²
        const two_h = Fp.add(H, H);
        const I = Fp.square(two_h);
        // J = H · I
        const J = Fp.montMul(H, I);
        // r = 2(S2 - S1)
        const r = Fp.add(Fp.sub(S2, S1), Fp.sub(S2, S1));
        // V = U1 · I
        const V = Fp.montMul(U1, I);
        // X3 = r² - J - 2V
        const x3 = Fp.sub(Fp.sub(Fp.square(r), J), Fp.add(V, V));
        // Y3 = r·(V - X3) - 2·S1·J
        const two_s1_j = Fp.add(Fp.montMul(S1, J), Fp.montMul(S1, J));
        const y3 = Fp.sub(Fp.montMul(r, Fp.sub(V, x3)), two_s1_j);
        // Z3 = ((Z1 + Z2)² - Z1Z1 - Z2Z2) · H
        const z_sum_sq = Fp.square(Fp.add(p.z, q.z));
        const z3 = Fp.montMul(Fp.sub(Fp.sub(z_sum_sq, Z1Z1), Z2Z2), H);
        return .{ .x = x3, .y = y3, .z = z3 };
    }

    /// Equality check that respects the Z scaling. Two Jacobian points
    /// `(X1, Y1, Z1)` and `(X2, Y2, Z2)` represent the same affine point
    /// iff `X1·Z2² == X2·Z1²` and `Y1·Z2³ == Y2·Z1³`.
    pub fn eql(a: G1Projective, b: G1Projective) bool {
        const a_inf = a.isIdentity();
        const b_inf = b.isIdentity();
        if (a_inf and b_inf) return true;
        if (a_inf or b_inf) return false;
        const z1z1 = Fp.square(a.z);
        const z2z2 = Fp.square(b.z);
        const x1z2z2 = Fp.montMul(a.x, z2z2);
        const x2z1z1 = Fp.montMul(b.x, z1z1);
        if (!Fp.eql(x1z2z2, x2z1z1)) return false;
        const z1z1z1 = Fp.montMul(z1z1, a.z);
        const z2z2z2 = Fp.montMul(z2z2, b.z);
        const y1z2z2z2 = Fp.montMul(a.y, z2z2z2);
        const y2z1z1z1 = Fp.montMul(b.y, z1z1z1);
        return Fp.eql(y1z2z2z2, y2z1z1z1);
    }

    /// Double-and-add scalar multiplication. Generic over the scalar
    /// limb count. Dramatically faster than `G1Affine.mul` because
    /// neither doubling nor addition needs Fermat inversion — only the
    /// final `toAffine` does.
    pub fn mul(self: G1Projective, comptime ScalarLimbs: comptime_int, scalar: [ScalarLimbs]u64) G1Projective {
        const top = bigint.bitLen(ScalarLimbs, scalar);
        if (top == 0) return identity();
        var result = identity();
        var i = top;
        while (i > 0) {
            i -= 1;
            result = result.double();
            const limb = i / 64;
            const bit = @as(u6, @intCast(i % 64));
            if (((scalar[limb] >> bit) & 1) == 1) {
                result = result.add(self);
            }
        }
        return result;
    }
};

// ---------------------------------------------------------------------------
// G2 Jacobian projective coordinates. Mirrors G1Projective but with
// Fp2 coordinates. Same formulas; the only thing that changes is the
// underlying field operations.
// ---------------------------------------------------------------------------

pub const G2Projective = struct {
    x: Fp2,
    y: Fp2,
    z: Fp2,

    pub fn identity() G2Projective {
        return .{ .x = Fp2.zero(), .y = Fp2.one(), .z = Fp2.zero() };
    }

    pub fn isIdentity(self: G2Projective) bool {
        return Fp2.eql(self.z, Fp2.zero());
    }

    pub fn fromAffine(p: G2Affine) G2Projective {
        if (p.infinity) return identity();
        return .{ .x = p.x, .y = p.y, .z = Fp2.one() };
    }

    pub fn toAffine(self: G2Projective) G2Affine {
        if (self.isIdentity()) return G2Affine.identity();
        const z_inv = Fp2.inv(self.z);
        const z_inv_sq = Fp2.square(z_inv);
        const z_inv_cubed = Fp2.mul(z_inv_sq, z_inv);
        return .{
            .x = Fp2.mul(self.x, z_inv_sq),
            .y = Fp2.mul(self.y, z_inv_cubed),
            .infinity = false,
        };
    }

    pub fn double(self: G2Projective) G2Projective {
        if (self.isIdentity()) return self;
        const A = Fp2.square(self.x);
        const B = Fp2.square(self.y);
        const C = Fp2.square(B);
        const x_plus_b = Fp2.add(self.x, B);
        const x_plus_b_sq = Fp2.square(x_plus_b);
        const D_inner = Fp2.sub(Fp2.sub(x_plus_b_sq, A), C);
        const D = Fp2.add(D_inner, D_inner);
        const E = Fp2.add(Fp2.add(A, A), A);
        const F = Fp2.square(E);
        const x3 = Fp2.sub(F, Fp2.add(D, D));
        const eight_c = blk: {
            const two_c = Fp2.add(C, C);
            const four_c = Fp2.add(two_c, two_c);
            break :blk Fp2.add(four_c, four_c);
        };
        const y3 = Fp2.sub(Fp2.mul(E, Fp2.sub(D, x3)), eight_c);
        const z3 = Fp2.add(Fp2.mul(self.y, self.z), Fp2.mul(self.y, self.z));
        return .{ .x = x3, .y = y3, .z = z3 };
    }

    pub fn add(p: G2Projective, q: G2Projective) G2Projective {
        if (p.isIdentity()) return q;
        if (q.isIdentity()) return p;
        const Z1Z1 = Fp2.square(p.z);
        const Z2Z2 = Fp2.square(q.z);
        const U1 = Fp2.mul(p.x, Z2Z2);
        const U2 = Fp2.mul(q.x, Z1Z1);
        const S1 = Fp2.mul(Fp2.mul(p.y, q.z), Z2Z2);
        const S2 = Fp2.mul(Fp2.mul(q.y, p.z), Z1Z1);
        if (Fp2.eql(U1, U2)) {
            if (Fp2.eql(S1, S2)) return p.double();
            return identity();
        }
        const H = Fp2.sub(U2, U1);
        const two_h = Fp2.add(H, H);
        const I = Fp2.square(two_h);
        const J = Fp2.mul(H, I);
        const r = Fp2.add(Fp2.sub(S2, S1), Fp2.sub(S2, S1));
        const V = Fp2.mul(U1, I);
        const x3 = Fp2.sub(Fp2.sub(Fp2.square(r), J), Fp2.add(V, V));
        const two_s1_j = Fp2.add(Fp2.mul(S1, J), Fp2.mul(S1, J));
        const y3 = Fp2.sub(Fp2.mul(r, Fp2.sub(V, x3)), two_s1_j);
        const z_sum_sq = Fp2.square(Fp2.add(p.z, q.z));
        const z3 = Fp2.mul(Fp2.sub(Fp2.sub(z_sum_sq, Z1Z1), Z2Z2), H);
        return .{ .x = x3, .y = y3, .z = z3 };
    }

    pub fn eql(a: G2Projective, b: G2Projective) bool {
        const a_inf = a.isIdentity();
        const b_inf = b.isIdentity();
        if (a_inf and b_inf) return true;
        if (a_inf or b_inf) return false;
        const z1z1 = Fp2.square(a.z);
        const z2z2 = Fp2.square(b.z);
        const x1z2z2 = Fp2.mul(a.x, z2z2);
        const x2z1z1 = Fp2.mul(b.x, z1z1);
        if (!Fp2.eql(x1z2z2, x2z1z1)) return false;
        const z1z1z1 = Fp2.mul(z1z1, a.z);
        const z2z2z2 = Fp2.mul(z2z2, b.z);
        const y1z2z2z2 = Fp2.mul(a.y, z2z2z2);
        const y2z1z1z1 = Fp2.mul(b.y, z1z1z1);
        return Fp2.eql(y1z2z2z2, y2z1z1z1);
    }

    /// Double-and-add scalar multiplication in projective form. See
    /// `G1Projective.mul` for the rationale — this is the fast path
    /// for any G2 scalar mul that the affine routine handles too
    /// slowly (subgroup checks, cofactor clearing, etc.).
    pub fn mul(self: G2Projective, comptime ScalarLimbs: comptime_int, scalar: [ScalarLimbs]u64) G2Projective {
        const top = bigint.bitLen(ScalarLimbs, scalar);
        if (top == 0) return identity();
        var result = identity();
        var i = top;
        while (i > 0) {
            i -= 1;
            result = result.double();
            const limb = i / 64;
            const bit = @as(u6, @intCast(i % 64));
            if (((scalar[limb] >> bit) & 1) == 1) {
                result = result.add(self);
            }
        }
        return result;
    }
};

// ---------------------------------------------------------------------------
// BLS12-381 pairing parameters and roadmap.
// ---------------------------------------------------------------------------

/// The BLS12-381 trace parameter `|x|`. The actual `x` is negative —
/// `x = -0xd201000000010000` — but the Miller loop walks the absolute
/// value and conjugates the final result. This single 64-bit constant
/// is everything the Miller loop needs to know about the curve choice.
pub const BLS_X_ABS: u64 = 0xd201000000010000;

/// Whether `x` is negative. If true, the Miller loop result must be
/// conjugated (`(c0 + c1·w) → (c0 - c1·w)`) before final exponentiation.
pub const BLS_X_IS_NEGATIVE: bool = true;

/// Miller loop length: bit length of `|x|`. The loop walks bits
/// `BLS_X_LOOP_BITS - 2` down to `0`.
pub const BLS_X_LOOP_BITS: usize = 64;

// ---- Pairing implementation is below, after the Fp12 / point decoding ----
// See `millerLoop`, `fp12FinalExp`, and `pairing` near the end of the
// non-test section. Hash-to-curve for G2 (RFC 9380 SSWU map) is in
// hash_to_field.zig and map_to_curve.zig (still to come).

/// `Fp2 = Fp[u] / (u² + 1)`. Elements are pairs `(c0, c1)` representing
/// `c0 + c1·u`. The non-residue is `-1`, so squaring `u` produces
/// `-1 ∈ Fp` directly.
///
/// Operations:
///
///   - `(a + bu) + (c + du) = (a + c) + (b + d)u`
///   - `(a + bu) - (c + du) = (a - c) + (b - d)u`
///   - `(a + bu) · (c + du) = (ac - bd) + (ad + bc)u`
///   - `(a + bu)⁻¹ = (a - bu) / (a² + b²)`
///
/// Multiplication uses the standard Karatsuba trick: three Fp
/// multiplications instead of four. The implementation here is
/// schoolbook for clarity; the optimization can land later if benchmarks
/// justify it.
pub const Fp2 = struct {
    c0: Fp.Element,
    c1: Fp.Element,

    pub fn zero() Fp2 {
        return .{ .c0 = Fp.zero(), .c1 = Fp.zero() };
    }

    pub fn one() Fp2 {
        return .{ .c0 = Fp.one(), .c1 = Fp.zero() };
    }

    pub fn eql(a: Fp2, b: Fp2) bool {
        return Fp.eql(a.c0, b.c0) and Fp.eql(a.c1, b.c1);
    }

    pub fn add(a: Fp2, b: Fp2) Fp2 {
        return .{
            .c0 = Fp.add(a.c0, b.c0),
            .c1 = Fp.add(a.c1, b.c1),
        };
    }

    pub fn sub(a: Fp2, b: Fp2) Fp2 {
        return .{
            .c0 = Fp.sub(a.c0, b.c0),
            .c1 = Fp.sub(a.c1, b.c1),
        };
    }

    pub fn neg(a: Fp2) Fp2 {
        return .{ .c0 = Fp.neg(a.c0), .c1 = Fp.neg(a.c1) };
    }

    /// `(a + bu)·(c + du) = (ac - bd) + (ad + bc)u`.
    pub fn mul(a: Fp2, b: Fp2) Fp2 {
        const ac = Fp.montMul(a.c0, b.c0);
        const bd = Fp.montMul(a.c1, b.c1);
        const ad = Fp.montMul(a.c0, b.c1);
        const bc = Fp.montMul(a.c1, b.c0);
        return .{
            .c0 = Fp.sub(ac, bd),
            .c1 = Fp.add(ad, bc),
        };
    }

    /// `(a + bu)² = (a² - b²) + 2ab·u`. Specialized so the squaring
    /// path uses two Fp multiplications + one Fp addition instead of
    /// four Fp multiplications.
    pub fn square(a: Fp2) Fp2 {
        const a_plus_b = Fp.add(a.c0, a.c1);
        const a_minus_b = Fp.sub(a.c0, a.c1);
        const c0 = Fp.montMul(a_plus_b, a_minus_b);
        const ab = Fp.montMul(a.c0, a.c1);
        const c1 = Fp.add(ab, ab);
        return .{ .c0 = c0, .c1 = c1 };
    }

    /// `(a + bu)⁻¹ = (a - bu) · (a² + b²)⁻¹`. The denominator is the
    /// norm of the element in `Fp`, so its inversion only needs an
    /// `Fp.inv` call.
    pub fn inv(a: Fp2) Fp2 {
        if (Fp.eql(a.c0, Fp.zero()) and Fp.eql(a.c1, Fp.zero())) return zero();
        const a0_sq = Fp.square(a.c0);
        const a1_sq = Fp.square(a.c1);
        const norm = Fp.add(a0_sq, a1_sq);
        const norm_inv = Fp.inv(norm);
        return .{
            .c0 = Fp.montMul(a.c0, norm_inv),
            .c1 = Fp.montMul(Fp.neg(a.c1), norm_inv),
        };
    }
};

/// Frobenius endomorphism in Fp2: `a → a^p`. For BLS12-381's base
/// prime `p ≡ 3 (mod 4)`, raising `u` to the `p`-th power gives `-u`,
/// so `(a₀ + a₁·u)^p = a₀ - a₁·u`. Conjugation by another name.
///
/// This is the building block the higher tower extensions use to
/// implement their own Frobenius via precomputed coefficients.
pub fn fp2Frobenius(a: Fp2) Fp2 {
    return .{ .c0 = a.c0, .c1 = Fp.neg(a.c1) };
}

/// Square-and-multiply exponentiation in Fp2. The exponent is a raw
/// little-endian limb array — each bit walked from MSB to LSB. Useful
/// for computing Frobenius coefficients and similar one-shot tower
/// constants without dragging precomputed tables into source.
pub fn fp2Pow(a: Fp2, comptime ExponentLimbs: comptime_int, exponent: [ExponentLimbs]u64) Fp2 {
    const top = bigint.bitLen(ExponentLimbs, exponent);
    if (top == 0) return Fp2.one();
    var result = a;
    var i = top - 1;
    while (i > 0) {
        i -= 1;
        result = Fp2.square(result);
        const limb = i / 64;
        const bit = @as(u6, @intCast(i % 64));
        if (((exponent[limb] >> bit) & 1) == 1) {
            result = Fp2.mul(result, a);
        }
    }
    return result;
}

/// Square root of `a ∈ Fp2` when one exists, otherwise `error.NotASquare`.
///
/// Algorithm: for `a = a₀ + a₁·u` and `u² = -1`, we want
/// `(c₀ + c₁·u)² = a`. Expanding gives the system
/// `c₀² - c₁² = a₀` and `2 c₀ c₁ = a₁`. Eliminating `c₁` and solving
/// the resulting quadratic in `c₀²` yields
/// `c₀² ∈ {(a₀ + β)/2, (a₀ - β)/2}` where `β = sqrt(a₀² + a₁²)` is
/// the square root of the norm in Fp.
///
/// At least one of the two candidates is a square in Fp; we try the
/// `+` branch first and fall back to `-`. Once `c₀` is known,
/// `c₁ = a₁ / (2 c₀)`. Special-case the `c₀ == 0` path: that means
/// `a₁ = 0` (`a` is a pure `Fp` element) and the sqrt collapses to
/// `(sqrt(a₀), 0)` if `a₀` is a square or `(0, sqrt(-a₀))` otherwise.
pub fn fp2Sqrt(a: Fp2) error{NotASquare}!Fp2 {
    // Pure-Fp special case (`a₁ = 0`).
    if (Fp.eql(a.c1, Fp.zero())) {
        if (Fp.eql(a.c0, Fp.zero())) return Fp2.zero();
        const root = fpSqrt(a.c0);
        if (fpIsSquareRoot(a.c0, root)) {
            return .{ .c0 = root, .c1 = Fp.zero() };
        }
        // a₀ is not a square in Fp. Then -a₀ might be — try it.
        const neg_a0 = Fp.neg(a.c0);
        const root2 = fpSqrt(neg_a0);
        if (fpIsSquareRoot(neg_a0, root2)) {
            return .{ .c0 = Fp.zero(), .c1 = root2 };
        }
        return error.NotASquare;
    }

    // General case.
    const norm = Fp.add(Fp.square(a.c0), Fp.square(a.c1));
    const beta = fpSqrt(norm);
    if (!fpIsSquareRoot(norm, beta)) return error.NotASquare;

    // Try the `+` branch: c₀² = (a₀ + β) / 2.
    const gamma_plus = fpHalve(Fp.add(a.c0, beta));
    var c0 = fpSqrt(gamma_plus);
    if (!fpIsSquareRoot(gamma_plus, c0)) {
        // Fall back to the `-` branch: c₀² = (a₀ - β) / 2.
        const gamma_minus = fpHalve(Fp.sub(a.c0, beta));
        c0 = fpSqrt(gamma_minus);
        if (!fpIsSquareRoot(gamma_minus, c0)) return error.NotASquare;
    }
    if (Fp.eql(c0, Fp.zero())) return error.NotASquare;

    // c₁ = a₁ / (2 c₀).
    const two_c0 = Fp.add(c0, c0);
    const c1 = Fp.montMul(a.c1, Fp.inv(two_c0));

    return .{ .c0 = c0, .c1 = c1 };
}

// ---------------------------------------------------------------------------
// BLS12-381 G1 short Weierstrass curve: y² = x³ + 4 over Fp.
// ---------------------------------------------------------------------------

/// `B` curve coefficient (4 in raw form).
pub const G1_B_RAW: [6]u64 = .{ 4, 0, 0, 0, 0, 0 };

/// Affine point on G1. The point at infinity uses the `infinity = true`
/// flag rather than encoding it as `(0, 0)`, so the predicates can stay
/// straightforward and the formulas don't have to special-case `(0, 0)`
/// when computing slopes.
pub const G1Affine = struct {
    x: Fp.Element,
    y: Fp.Element,
    infinity: bool,

    /// Identity element (point at infinity).
    pub fn identity() G1Affine {
        return .{ .x = Fp.zero(), .y = Fp.zero(), .infinity = true };
    }

    /// Construct an affine point from raw little-endian limb arrays.
    /// Caller is responsible for ensuring the point is on the curve.
    pub fn fromRaw(x_raw: [6]u64, y_raw: [6]u64) G1Affine {
        return .{
            .x = Fp.fromRaw(x_raw),
            .y = Fp.fromRaw(y_raw),
            .infinity = false,
        };
    }

    pub fn isIdentity(self: G1Affine) bool {
        return self.infinity;
    }

    /// Equality. Identity points compare equal regardless of their
    /// stored coordinates.
    pub fn eql(a: G1Affine, b: G1Affine) bool {
        if (a.infinity and b.infinity) return true;
        if (a.infinity or b.infinity) return false;
        return Fp.eql(a.x, b.x) and Fp.eql(a.y, b.y);
    }

    /// Curve membership: `y² == x³ + 4`. Identity points are members
    /// by definition.
    pub fn isOnCurve(self: G1Affine) bool {
        if (self.infinity) return true;
        const y_sq = Fp.square(self.y);
        const x_sq = Fp.square(self.x);
        const x_cubed = Fp.montMul(x_sq, self.x);
        const b = Fp.fromRaw(G1_B_RAW);
        const rhs = Fp.add(x_cubed, b);
        return Fp.eql(y_sq, rhs);
    }

    /// `-P = (x, -y)`.
    pub fn neg(self: G1Affine) G1Affine {
        if (self.infinity) return self;
        return .{ .x = self.x, .y = Fp.neg(self.y), .infinity = false };
    }

    /// Affine doubling: `2P` for a non-identity point.
    /// `λ = 3x² / (2y)`, `x₃ = λ² - 2x`, `y₃ = λ(x - x₃) - y`.
    pub fn double(self: G1Affine) G1Affine {
        if (self.infinity) return self;
        // If `y == 0`, the doubled point is the identity.
        if (Fp.eql(self.y, Fp.zero())) return identity();
        const x_sq = Fp.square(self.x);
        const three_x_sq = Fp.add(Fp.add(x_sq, x_sq), x_sq);
        const two_y = Fp.add(self.y, self.y);
        const lambda = Fp.montMul(three_x_sq, Fp.inv(two_y));
        const lambda_sq = Fp.square(lambda);
        const two_x = Fp.add(self.x, self.x);
        const x3 = Fp.sub(lambda_sq, two_x);
        const y3 = Fp.sub(Fp.montMul(lambda, Fp.sub(self.x, x3)), self.y);
        return .{ .x = x3, .y = y3, .infinity = false };
    }

    /// Affine addition: `P + Q` for distinct, non-identity points.
    /// Falls back to `double` when `P == Q` and to the identity when
    /// `P == -Q`.
    pub fn add(a: G1Affine, b: G1Affine) G1Affine {
        if (a.infinity) return b;
        if (b.infinity) return a;
        if (Fp.eql(a.x, b.x)) {
            // Same x → either P+P or P + (-P).
            if (Fp.eql(a.y, b.y)) return a.double();
            return identity();
        }
        const lambda = Fp.montMul(Fp.sub(b.y, a.y), Fp.inv(Fp.sub(b.x, a.x)));
        const lambda_sq = Fp.square(lambda);
        const x3 = Fp.sub(Fp.sub(lambda_sq, a.x), b.x);
        const y3 = Fp.sub(Fp.montMul(lambda, Fp.sub(a.x, x3)), a.y);
        return .{ .x = x3, .y = y3, .infinity = false };
    }

    /// Scalar multiplication via double-and-add. The scalar is a raw
    /// little-endian limb array of arbitrary width — bits are walked
    /// from MSB to LSB.
    ///
    /// This is the simplest correct implementation. It does NOT use a
    /// constant-time ladder, sliding-window NAF, GLV decomposition, or
    /// any of the other tricks that real BLS verifiers reach for. The
    /// upcoming pairing-based verification will need scalar multiples
    /// of fixed/variable points, and we can pick a faster algorithm
    /// when benchmarks justify it.
    pub fn mul(self: G1Affine, comptime ScalarLimbs: comptime_int, scalar: [ScalarLimbs]u64) G1Affine {
        const top = bigint.bitLen(ScalarLimbs, scalar);
        if (top == 0) return identity();
        var result = identity();
        var i = top;
        while (i > 0) {
            i -= 1;
            result = result.double();
            const limb = i / 64;
            const bit = @as(u6, @intCast(i % 64));
            if (((scalar[limb] >> bit) & 1) == 1) {
                result = result.add(self);
            }
        }
        return result;
    }
};

/// BLS12-381 G1 generator point. Coordinates from RFC 9380 §8.8.1
/// (also matches the blst constants).
///
///   x = 0x17F1D3A73197D7942695638C4FA9AC0FC3688C4F9774B905
///         A14E3A3F171BAC586C55E83FF97A1AEFFB3AF00ADB22C6BB
///   y = 0x08B3F481E3AAA0F1A09E30ED741D8AE4FCF5E095D5D00AF6
///         00DB18CB2C04B3EDD03CC744A2888AE40CAA232946C5E7E1
pub const G1_GENERATOR_X: [6]u64 = .{
    0xfb3af00adb22c6bb,
    0x6c55e83ff97a1aef,
    0xa14e3a3f171bac58,
    0xc3688c4f9774b905,
    0x2695638c4fa9ac0f,
    0x17f1d3a73197d794,
};
pub const G1_GENERATOR_Y: [6]u64 = .{
    0x0caa232946c5e7e1,
    0xd03cc744a2888ae4,
    0x00db18cb2c04b3ed,
    0xfcf5e095d5d00af6,
    0xa09e30ed741d8ae4,
    0x08b3f481e3aaa0f1,
};

pub fn g1Generator() G1Affine {
    return G1Affine.fromRaw(G1_GENERATOR_X, G1_GENERATOR_Y);
}

// ---------------------------------------------------------------------------
// Fp6 = Fp2[v]/(v³ - (1+u)). Cubic extension built on top of Fp2.
// Elements are tuples (c0, c1, c2) representing c0 + c1·v + c2·v² with
// v³ = 1+u (the non-residue from Fp2).
// ---------------------------------------------------------------------------

/// Multiply an Fp2 element by the non-residue `1 + u`. Used heavily by
/// Fp6 / Fp12 reduction. The expanded form is `(c0 - c1) + (c0 + c1)·u`.
pub fn fp2MulByNonresidue(a: Fp2) Fp2 {
    return .{
        .c0 = Fp.sub(a.c0, a.c1),
        .c1 = Fp.add(a.c0, a.c1),
    };
}

pub const Fp6 = struct {
    c0: Fp2,
    c1: Fp2,
    c2: Fp2,

    pub fn zero() Fp6 {
        return .{ .c0 = Fp2.zero(), .c1 = Fp2.zero(), .c2 = Fp2.zero() };
    }

    pub fn one() Fp6 {
        return .{ .c0 = Fp2.one(), .c1 = Fp2.zero(), .c2 = Fp2.zero() };
    }

    pub fn eql(a: Fp6, b: Fp6) bool {
        return Fp2.eql(a.c0, b.c0) and Fp2.eql(a.c1, b.c1) and Fp2.eql(a.c2, b.c2);
    }

    pub fn add(a: Fp6, b: Fp6) Fp6 {
        return .{
            .c0 = Fp2.add(a.c0, b.c0),
            .c1 = Fp2.add(a.c1, b.c1),
            .c2 = Fp2.add(a.c2, b.c2),
        };
    }

    pub fn sub(a: Fp6, b: Fp6) Fp6 {
        return .{
            .c0 = Fp2.sub(a.c0, b.c0),
            .c1 = Fp2.sub(a.c1, b.c1),
            .c2 = Fp2.sub(a.c2, b.c2),
        };
    }

    pub fn neg(a: Fp6) Fp6 {
        return .{ .c0 = Fp2.neg(a.c0), .c1 = Fp2.neg(a.c1), .c2 = Fp2.neg(a.c2) };
    }

    /// Schoolbook multiplication. After collecting like-terms:
    ///
    ///   c₀ = a₀·b₀ + (a₁·b₂ + a₂·b₁) · (1+u)
    ///   c₁ = a₀·b₁ + a₁·b₀ + a₂·b₂ · (1+u)
    ///   c₂ = a₀·b₂ + a₁·b₁ + a₂·b₀
    ///
    /// 9 Fp2 multiplications + a handful of additions. Karatsuba can
    /// drop this to 6 Fp2 multiplications; left as future work.
    pub fn mul(a: Fp6, b: Fp6) Fp6 {
        const t00 = Fp2.mul(a.c0, b.c0);
        const t01 = Fp2.mul(a.c0, b.c1);
        const t02 = Fp2.mul(a.c0, b.c2);
        const t10 = Fp2.mul(a.c1, b.c0);
        const t11 = Fp2.mul(a.c1, b.c1);
        const t12 = Fp2.mul(a.c1, b.c2);
        const t20 = Fp2.mul(a.c2, b.c0);
        const t21 = Fp2.mul(a.c2, b.c1);
        const t22 = Fp2.mul(a.c2, b.c2);
        const c0 = Fp2.add(t00, fp2MulByNonresidue(Fp2.add(t12, t21)));
        const c1 = Fp2.add(Fp2.add(t01, t10), fp2MulByNonresidue(t22));
        const c2 = Fp2.add(Fp2.add(t02, t11), t20);
        return .{ .c0 = c0, .c1 = c1, .c2 = c2 };
    }

    pub fn square(a: Fp6) Fp6 {
        // Could specialize but mul(a, a) is correct and clear.
        return mul(a, a);
    }

    /// Multiply by `v` (i.e., shift coefficients up). Used by Fp12.
    /// `(c0 + c1v + c2v²) · v = c0v + c1v² + c2v³ = c2(1+u) + c0v + c1v²`.
    pub fn mulByV(a: Fp6) Fp6 {
        return .{
            .c0 = fp2MulByNonresidue(a.c2),
            .c1 = a.c0,
            .c2 = a.c1,
        };
    }

    /// Square-and-multiply exponentiation in Fp6. Generic over the
    /// exponent limb count.
    pub fn pow(a: Fp6, comptime ExponentLimbs: comptime_int, exponent: [ExponentLimbs]u64) Fp6 {
        const top = bigint.bitLen(ExponentLimbs, exponent);
        if (top == 0) return one();
        var result = a;
        var i = top - 1;
        while (i > 0) {
            i -= 1;
            result = square(result);
            const limb = i / 64;
            const bit = @as(u6, @intCast(i % 64));
            if (((exponent[limb] >> bit) & 1) == 1) {
                result = mul(result, a);
            }
        }
        return result;
    }

    /// Inversion using the standard adjugate / norm formula. For
    /// `a = a₀ + a₁v + a₂v²` in Fp6, define
    ///
    ///   A = a₀² − ξ·a₁·a₂
    ///   B = ξ·a₂² − a₀·a₁
    ///   C = a₁² − a₀·a₂
    ///
    /// Then `a · (A + Bv + Cv²) = D` where
    ///
    ///   D = a₀·A + ξ·a₂·B + ξ·a₁·C
    ///
    /// is an element of Fp2. Inverting D in Fp2 and scaling gives
    /// `a⁻¹ = (A + Bv + Cv²) / D`.
    pub fn inv(a: Fp6) Fp6 {
        const a0_sq = Fp2.square(a.c0);
        const a1_sq = Fp2.square(a.c1);
        const a2_sq = Fp2.square(a.c2);
        const a0_a1 = Fp2.mul(a.c0, a.c1);
        const a0_a2 = Fp2.mul(a.c0, a.c2);
        const a1_a2 = Fp2.mul(a.c1, a.c2);

        const A = Fp2.sub(a0_sq, fp2MulByNonresidue(a1_a2));
        const B = Fp2.sub(fp2MulByNonresidue(a2_sq), a0_a1);
        const C = Fp2.sub(a1_sq, a0_a2);

        const a0_A = Fp2.mul(a.c0, A);
        const xi_a2_B = fp2MulByNonresidue(Fp2.mul(a.c2, B));
        const xi_a1_C = fp2MulByNonresidue(Fp2.mul(a.c1, C));
        const D = Fp2.add(Fp2.add(a0_A, xi_a2_B), xi_a1_C);
        const D_inv = Fp2.inv(D);

        return .{
            .c0 = Fp2.mul(A, D_inv),
            .c1 = Fp2.mul(B, D_inv),
            .c2 = Fp2.mul(C, D_inv),
        };
    }
};

// ---------------------------------------------------------------------------
// Fp12 = Fp6[w]/(w² - v). The pairing target group lives here.
// Elements are pairs (c0, c1) representing c0 + c1·w with w² = v.
// ---------------------------------------------------------------------------

pub const Fp12 = struct {
    c0: Fp6,
    c1: Fp6,

    pub fn zero() Fp12 {
        return .{ .c0 = Fp6.zero(), .c1 = Fp6.zero() };
    }

    pub fn one() Fp12 {
        return .{ .c0 = Fp6.one(), .c1 = Fp6.zero() };
    }

    pub fn eql(a: Fp12, b: Fp12) bool {
        return Fp6.eql(a.c0, b.c0) and Fp6.eql(a.c1, b.c1);
    }

    pub fn add(a: Fp12, b: Fp12) Fp12 {
        return .{ .c0 = Fp6.add(a.c0, b.c0), .c1 = Fp6.add(a.c1, b.c1) };
    }

    pub fn sub(a: Fp12, b: Fp12) Fp12 {
        return .{ .c0 = Fp6.sub(a.c0, b.c0), .c1 = Fp6.sub(a.c1, b.c1) };
    }

    pub fn neg(a: Fp12) Fp12 {
        return .{ .c0 = Fp6.neg(a.c0), .c1 = Fp6.neg(a.c1) };
    }

    /// `(a₀ + a₁w)(b₀ + b₁w) = (a₀b₀ + a₁b₁·v) + (a₀b₁ + a₁b₀)w`.
    /// Karatsuba: `(a₀ + a₁)(b₀ + b₁) - a₀b₀ - a₁b₁` for the cross term.
    pub fn mul(a: Fp12, b: Fp12) Fp12 {
        const aa = Fp6.mul(a.c0, b.c0);
        const bb = Fp6.mul(a.c1, b.c1);
        const c0 = Fp6.add(aa, Fp6.mulByV(bb));
        const c1 = Fp6.sub(
            Fp6.sub(Fp6.mul(Fp6.add(a.c0, a.c1), Fp6.add(b.c0, b.c1)), aa),
            bb,
        );
        return .{ .c0 = c0, .c1 = c1 };
    }

    pub fn square(a: Fp12) Fp12 {
        return mul(a, a);
    }

    /// `(c₀ + c₁w)⁻¹ = (c₀ - c₁w) / (c₀² - v·c₁²)`. The denominator is
    /// in Fp6, so the cost is one Fp6 inversion plus a handful of mul.
    pub fn inv(a: Fp12) Fp12 {
        const c0_sq = Fp6.square(a.c0);
        const c1_sq = Fp6.square(a.c1);
        const norm = Fp6.sub(c0_sq, Fp6.mulByV(c1_sq));
        const norm_inv = Fp6.inv(norm);
        return .{
            .c0 = Fp6.mul(a.c0, norm_inv),
            .c1 = Fp6.neg(Fp6.mul(a.c1, norm_inv)),
        };
    }

    /// Conjugation: `(c₀ + c₁·w) → (c₀ - c₁·w)`.
    ///
    /// For elements of the cyclotomic subgroup, conjugation equals
    /// `a^(p^6)` — that's the property the easy part of the final
    /// exponentiation exploits to avoid a full p^6 powering.
    pub fn conjugate(a: Fp12) Fp12 {
        return .{ .c0 = a.c0, .c1 = Fp6.neg(a.c1) };
    }

    /// Square-and-multiply exponentiation in Fp12. Generic over the
    /// limb count of the exponent. Used by the slow-but-correct path
    /// of the final exponentiation; the optimized version replaces
    /// some powerings with Frobenius applications once the
    /// `Fp6.frobenius` constants land.
    pub fn pow(a: Fp12, comptime ExponentLimbs: comptime_int, exponent: [ExponentLimbs]u64) Fp12 {
        const top = bigint.bitLen(ExponentLimbs, exponent);
        if (top == 0) return one();
        var result = a;
        var i = top - 1;
        while (i > 0) {
            i -= 1;
            result = square(result);
            const limb = i / 64;
            const bit = @as(u6, @intCast(i % 64));
            if (((exponent[limb] >> bit) & 1) == 1) {
                result = mul(result, a);
            }
        }
        return result;
    }
};

// ---------------------------------------------------------------------------
// Compressed point encoding (BLS12-381 / RFC 9380 §3.3, IETF
// draft-irtf-cfrg-pairing-friendly-curves §C.2).
//
// A compressed G1 point is exactly 48 bytes — the big-endian encoding
// of the x coordinate, with three flag bits stuffed into the highest
// three bits of the first byte:
//
//   bit 7 (msb): compression flag — 1 = compressed, 0 = uncompressed
//   bit 6     : infinity flag
//   bit 5     : y_sign / y_lex flag (which of the two y roots to take)
//
// The actual x coordinate occupies the remaining 381 bits (the top
// three bits of the 384-bit field are masked off when reading).
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

pub const PointDecodeError = error{
    InvalidLength,
    InvalidEncoding,
    NotOnCurve,
    NotInField,
};

/// Decode a 48-byte compressed BLS12-381 G1 point. Returns the
/// resulting `G1Affine`. Validates the compression / infinity flags
/// and verifies that the recovered point lies on the curve.
///
/// Subgroup membership (cofactor clearing) is NOT checked here — that
/// is a higher-level decision the caller can layer on top.
pub fn decodeG1Compressed(bytes: []const u8) PointDecodeError!G1Affine {
    if (bytes.len != 48) return PointDecodeError.InvalidLength;

    // Extract and clear the flag bits.
    const compression_flag = (bytes[0] >> 7) & 1;
    const infinity_flag = (bytes[0] >> 6) & 1;
    const y_sign = (bytes[0] >> 5) & 1;
    if (compression_flag != 1) return PointDecodeError.InvalidEncoding;

    // Infinity: every other bit must be zero.
    if (infinity_flag == 1) {
        if (y_sign != 0) return PointDecodeError.InvalidEncoding;
        // First byte: 0xc0 (compression + infinity bits). Remaining 47
        // bytes must all be zero.
        if (bytes[0] != 0xc0) return PointDecodeError.InvalidEncoding;
        for (bytes[1..]) |b| if (b != 0) return PointDecodeError.InvalidEncoding;
        return G1Affine.identity();
    }

    // Strip the flag bits from the first byte and copy into a working
    // buffer. Then read the 48 bytes as big-endian limbs.
    var clean: [48]u8 = undefined;
    @memcpy(&clean, bytes);
    clean[0] &= 0b0001_1111;
    const x_raw = bigint.fromBytesBe(6, &clean);

    // Reject x ≥ p. The compressed encoding pretends the top three
    // bits aren't there, but a malicious sender could still set the
    // remaining 381 bits above p.
    if (bigint.cmp(6, x_raw, FP_MODULUS) != .lt) return PointDecodeError.NotInField;

    // Convert into Montgomery form and reconstruct y from the curve
    // equation: y² = x³ + 4. The square-root must round-trip — if it
    // doesn't, x is not a valid x-coordinate of any point on the curve.
    const x = Fp.fromRaw(x_raw);
    const x_sq = Fp.square(x);
    const x_cubed = Fp.montMul(x_sq, x);
    const b = Fp.fromRaw(G1_B_RAW);
    const rhs = Fp.add(x_cubed, b);
    const y_candidate = fpSqrt(rhs);
    if (!fpIsSquareRoot(rhs, y_candidate)) return PointDecodeError.NotOnCurve;

    // Pick the y root matching the sign flag. The "lexicographically
    // larger" of the two roots has its high bit set in the raw
    // representation. Compare against the negation to decide.
    const y_neg = Fp.neg(y_candidate);
    const y_raw = Fp.toRaw(y_candidate);
    const y_neg_raw = Fp.toRaw(y_neg);
    const candidate_is_larger = bigint.cmp(6, y_raw, y_neg_raw) == .gt;
    const y = if ((y_sign == 1) == candidate_is_larger) y_candidate else y_neg;

    return .{ .x = x, .y = y, .infinity = false };
}

// ---------------------------------------------------------------------------
// BLS12-381 G2 short Weierstrass curve: y² = x³ + 4(1 + u) over Fp2.
// ---------------------------------------------------------------------------

/// `B` curve coefficient for G2 in Fp2 form: `4 + 4·u`.
pub fn g2B() Fp2 {
    const four = Fp.fromRaw(.{ 4, 0, 0, 0, 0, 0 });
    return .{ .c0 = four, .c1 = four };
}

/// Affine point on G2. Mirrors `G1Affine` but with `Fp2` coordinates.
/// The point at infinity is signalled by the `infinity` flag.
pub const G2Affine = struct {
    x: Fp2,
    y: Fp2,
    infinity: bool,

    pub fn identity() G2Affine {
        return .{ .x = Fp2.zero(), .y = Fp2.zero(), .infinity = true };
    }

    pub fn isIdentity(self: G2Affine) bool {
        return self.infinity;
    }

    pub fn eql(a: G2Affine, b: G2Affine) bool {
        if (a.infinity and b.infinity) return true;
        if (a.infinity or b.infinity) return false;
        return Fp2.eql(a.x, b.x) and Fp2.eql(a.y, b.y);
    }

    /// `y² == x³ + 4(1 + u)`. Identity is on the curve by definition.
    pub fn isOnCurve(self: G2Affine) bool {
        if (self.infinity) return true;
        const y_sq = Fp2.square(self.y);
        const x_sq = Fp2.square(self.x);
        const x_cubed = Fp2.mul(x_sq, self.x);
        const rhs = Fp2.add(x_cubed, g2B());
        return Fp2.eql(y_sq, rhs);
    }

    /// `-P = (x, -y)`.
    pub fn neg(self: G2Affine) G2Affine {
        if (self.infinity) return self;
        return .{ .x = self.x, .y = Fp2.neg(self.y), .infinity = false };
    }

    /// Affine doubling. Same shape as G1; the only differences are the
    /// underlying field operations.
    pub fn double(self: G2Affine) G2Affine {
        if (self.infinity) return self;
        if (Fp2.eql(self.y, Fp2.zero())) return identity();
        const x_sq = Fp2.square(self.x);
        const three_x_sq = Fp2.add(Fp2.add(x_sq, x_sq), x_sq);
        const two_y = Fp2.add(self.y, self.y);
        const lambda = Fp2.mul(three_x_sq, Fp2.inv(two_y));
        const lambda_sq = Fp2.square(lambda);
        const two_x = Fp2.add(self.x, self.x);
        const x3 = Fp2.sub(lambda_sq, two_x);
        const y3 = Fp2.sub(Fp2.mul(lambda, Fp2.sub(self.x, x3)), self.y);
        return .{ .x = x3, .y = y3, .infinity = false };
    }

    /// Affine addition. Falls back to `double` when `P == Q` and to
    /// the identity when `P == -Q`.
    pub fn add(a: G2Affine, b: G2Affine) G2Affine {
        if (a.infinity) return b;
        if (b.infinity) return a;
        if (Fp2.eql(a.x, b.x)) {
            if (Fp2.eql(a.y, b.y)) return a.double();
            return identity();
        }
        const lambda = Fp2.mul(Fp2.sub(b.y, a.y), Fp2.inv(Fp2.sub(b.x, a.x)));
        const lambda_sq = Fp2.square(lambda);
        const x3 = Fp2.sub(Fp2.sub(lambda_sq, a.x), b.x);
        const y3 = Fp2.sub(Fp2.mul(lambda, Fp2.sub(a.x, x3)), a.y);
        return .{ .x = x3, .y = y3, .infinity = false };
    }

    /// Scalar multiplication via double-and-add. Generic over the
    /// scalar limb count.
    pub fn mul(self: G2Affine, comptime ScalarLimbs: comptime_int, scalar: [ScalarLimbs]u64) G2Affine {
        const top = bigint.bitLen(ScalarLimbs, scalar);
        if (top == 0) return identity();
        var result = identity();
        var i = top;
        while (i > 0) {
            i -= 1;
            result = result.double();
            const limb = i / 64;
            const bit = @as(u6, @intCast(i % 64));
            if (((scalar[limb] >> bit) & 1) == 1) {
                result = result.add(self);
            }
        }
        return result;
    }
};

/// BLS12-381 G2 generator coordinates. Decomposed limb-by-limb from the
/// standard hex strings (also pinned by blst):
///
///   x.c0 = 0x024aa2b2f08f0a91260805272dc51051
///            c6e47ad4fa403b02b4510b647ae3d177
///            0bac0326a805bbefd48056c8c121bdb8
///   x.c1 = 0x13e02b6052719f607dacd3a088274f65
///            596bd0d09920b61ab5da61bbdc7f5049
///            334cf11213945d57e5ac7d055d042b7e
///   y.c0 = 0x0ce5d527727d6e118cc9cdc6da2e351a
///            adfd9baa8cbdd3a76d429a695160d12c
///            923ac9cc3baca289e193548608b82801
///   y.c1 = 0x0606c4a02ea734cc32acd2b02bc28b99
///            cb3e287e85a763af267492ab572e99ab
///            3f370d275cec1da1aaa9075ff05f79be
pub const G2_GENERATOR_X_C0: [6]u64 = .{
    0xd48056c8c121bdb8,
    0x0bac0326a805bbef,
    0xb4510b647ae3d177,
    0xc6e47ad4fa403b02,
    0x260805272dc51051,
    0x024aa2b2f08f0a91,
};
pub const G2_GENERATOR_X_C1: [6]u64 = .{
    0xe5ac7d055d042b7e,
    0x334cf11213945d57,
    0xb5da61bbdc7f5049,
    0x596bd0d09920b61a,
    0x7dacd3a088274f65,
    0x13e02b6052719f60,
};
pub const G2_GENERATOR_Y_C0: [6]u64 = .{
    0xe193548608b82801,
    0x923ac9cc3baca289,
    0x6d429a695160d12c,
    0xadfd9baa8cbdd3a7,
    0x8cc9cdc6da2e351a,
    0x0ce5d527727d6e11,
};
pub const G2_GENERATOR_Y_C1: [6]u64 = .{
    0xaaa9075ff05f79be,
    0x3f370d275cec1da1,
    0x267492ab572e99ab,
    0xcb3e287e85a763af,
    0x32acd2b02bc28b99,
    0x0606c4a02ea734cc,
};

pub fn g2Generator() G2Affine {
    return .{
        .x = .{
            .c0 = Fp.fromRaw(G2_GENERATOR_X_C0),
            .c1 = Fp.fromRaw(G2_GENERATOR_X_C1),
        },
        .y = .{
            .c0 = Fp.fromRaw(G2_GENERATOR_Y_C0),
            .c1 = Fp.fromRaw(G2_GENERATOR_Y_C1),
        },
        .infinity = false,
    };
}

/// Decode a 96-byte compressed BLS12-381 G2 point. The wire format is
/// the IETF pairing-friendly-curves draft §C.2 layout: the first 48
/// bytes encode `x.c1` (the imaginary coordinate) and the next 48
/// encode `x.c0`, both as big-endian Fp elements. The same three flag
/// bits live in the high bits of byte 0:
///
///   bit 7: compression flag
///   bit 6: infinity flag
///   bit 5: y-sign / lex flag
///
/// The recovered y is reconstructed from `y² = x³ + 4(1+u)` via
/// `fp2Sqrt`, with the y-sign bit picking between the two roots based
/// on lexicographic comparison of the c1/c0 limb representation.
///
/// Subgroup membership is NOT checked here.
pub fn decodeG2Compressed(bytes: []const u8) PointDecodeError!G2Affine {
    if (bytes.len != 96) return PointDecodeError.InvalidLength;

    const compression_flag = (bytes[0] >> 7) & 1;
    const infinity_flag = (bytes[0] >> 6) & 1;
    const y_sign = (bytes[0] >> 5) & 1;
    if (compression_flag != 1) return PointDecodeError.InvalidEncoding;

    if (infinity_flag == 1) {
        if (y_sign != 0) return PointDecodeError.InvalidEncoding;
        if (bytes[0] != 0xc0) return PointDecodeError.InvalidEncoding;
        for (bytes[1..]) |b| if (b != 0) return PointDecodeError.InvalidEncoding;
        return G2Affine.identity();
    }

    // Read x.c1 (first 48 bytes) and x.c0 (next 48). Strip the flag
    // bits before parsing the c1 limb representation.
    var c1_bytes: [48]u8 = undefined;
    @memcpy(&c1_bytes, bytes[0..48]);
    c1_bytes[0] &= 0b0001_1111;
    const c0_bytes = bytes[48..96];

    const x_c1_raw = bigint.fromBytesBe(6, &c1_bytes);
    const x_c0_raw = bigint.fromBytesBe(6, c0_bytes);
    if (bigint.cmp(6, x_c1_raw, FP_MODULUS) != .lt) return PointDecodeError.NotInField;
    if (bigint.cmp(6, x_c0_raw, FP_MODULUS) != .lt) return PointDecodeError.NotInField;

    const x: Fp2 = .{
        .c0 = Fp.fromRaw(x_c0_raw),
        .c1 = Fp.fromRaw(x_c1_raw),
    };

    // y² = x³ + 4(1+u). Reconstruct y via fp2Sqrt.
    const x_sq = Fp2.square(x);
    const x_cubed = Fp2.mul(x_sq, x);
    const rhs = Fp2.add(x_cubed, g2B());
    const y_candidate = fp2Sqrt(rhs) catch return PointDecodeError.NotOnCurve;
    const y_neg = Fp2.neg(y_candidate);

    // Choose the y root matching the y_sign flag. Lexicographic order
    // on Fp2 elements compares c1 first then c0 (the natural projection
    // of the byte serialization).
    const y_c1_raw = Fp.toRaw(y_candidate.c1);
    const y_c0_raw = Fp.toRaw(y_candidate.c0);
    const y_neg_c1_raw = Fp.toRaw(y_neg.c1);
    const y_neg_c0_raw = Fp.toRaw(y_neg.c0);

    const candidate_is_larger = blk: {
        const c1_cmp = bigint.cmp(6, y_c1_raw, y_neg_c1_raw);
        if (c1_cmp != .eq) break :blk c1_cmp == .gt;
        break :blk bigint.cmp(6, y_c0_raw, y_neg_c0_raw) == .gt;
    };
    const y = if ((y_sign == 1) == candidate_is_larger) y_candidate else y_neg;

    return .{ .x = x, .y = y, .infinity = false };
}

// ---------------------------------------------------------------------------
// Compressed point encoding (inverse of `decodeG1Compressed` /
// `decodeG2Compressed`).
//
// The encoder is the simple inverse of the decoder:
//
//   1. Identity: 0xc0 || 47 zeros (G1) or 95 zeros (G2).
//   2. Otherwise: serialize the x coordinate big-endian, set the
//      compression flag (bit 7), and set the y-sign flag (bit 5) if
//      the canonical y is the lexicographically larger of the two
//      square roots.
//
// The y-sign convention matches the decoder: the bit is set when the
// affine y is greater than (-y) under the raw little-endian limb
// comparison. The infinity flag (bit 6) is mutually exclusive with the
// y-sign flag.
// ---------------------------------------------------------------------------

/// Encode an affine G1 point into 48 compressed bytes. The output is
/// the inverse of `decodeG1Compressed` — feeding the result back through
/// the decoder yields a point equal to `p`.
pub fn encodeG1Compressed(p: G1Affine) [48]u8 {
    var out: [48]u8 = .{0} ** 48;
    if (p.infinity) {
        out[0] = 0xc0; // compression flag + infinity flag
        return out;
    }
    // Serialize x as 48 big-endian bytes.
    const x_raw = Fp.toRaw(p.x);
    bigint.toBytesBe(6, x_raw, &out);

    // Decide y-sign: set bit 5 of byte 0 iff y > -y.
    const y_raw = Fp.toRaw(p.y);
    const y_neg_raw = Fp.toRaw(Fp.neg(p.y));
    const y_is_larger = bigint.cmp(6, y_raw, y_neg_raw) == .gt;

    // Set the compression flag (bit 7) and the y-sign flag if needed.
    out[0] |= 0x80;
    if (y_is_larger) out[0] |= 0x20;
    return out;
}

/// Encode an affine G2 point into 96 compressed bytes. Layout matches
/// the IETF pairing-friendly-curves draft §C.2: bytes [0..48] hold
/// `x.c1` (with the flag bits in byte 0) and bytes [48..96] hold
/// `x.c0`.
pub fn encodeG2Compressed(p: G2Affine) [96]u8 {
    var out: [96]u8 = .{0} ** 96;
    if (p.infinity) {
        out[0] = 0xc0;
        return out;
    }

    // x.c1 → first 48 bytes, x.c0 → next 48 bytes.
    const x_c1_raw = Fp.toRaw(p.x.c1);
    const x_c0_raw = Fp.toRaw(p.x.c0);
    bigint.toBytesBe(6, x_c1_raw, out[0..48]);
    bigint.toBytesBe(6, x_c0_raw, out[48..96]);

    // y-sign decision uses lex comparison on (c1, c0).
    const y_neg = Fp2.neg(p.y);
    const y_c1_raw = Fp.toRaw(p.y.c1);
    const y_c0_raw = Fp.toRaw(p.y.c0);
    const y_neg_c1_raw = Fp.toRaw(y_neg.c1);
    const y_neg_c0_raw = Fp.toRaw(y_neg.c0);

    const y_is_larger = blk: {
        const c1_cmp = bigint.cmp(6, y_c1_raw, y_neg_c1_raw);
        if (c1_cmp != .eq) break :blk c1_cmp == .gt;
        break :blk bigint.cmp(6, y_c0_raw, y_neg_c0_raw) == .gt;
    };

    out[0] |= 0x80;
    if (y_is_larger) out[0] |= 0x20;
    return out;
}

// ---------------------------------------------------------------------------
// Sparse Fp6 / Fp12 multiplication helpers used by the Miller loop.
//
// The line function evaluation in BLS12 pairings produces an Fp12 element
// with most coefficients zero. Multiplying f ∈ Fp12 by such a sparse value
// is much cheaper than a full Fp12.mul if we exploit the zero positions.
//
// Conventions match arkworks (`mul_by_014`, `mul_by_01`, `mul_by_1`):
//
//   - `Fp6.mulBy01(c0, c1)`: multiply Fp6 element by `c0 + c1·v`
//   - `Fp6.mulBy1(c1)`:      multiply Fp6 element by `c1·v`
//   - `Fp12.mulBy014(c0, c1, c4)`: multiply Fp12 element by an Fp12 whose
//     basis components at positions {0=1, 1=v, 4=vw} are c0, c1, c4 and
//     all other positions are zero. M-twist line evaluations sit there.
// ---------------------------------------------------------------------------

/// Sparse Fp6 multiplication: `a · (c0 + c1·v)`. Saves the three Fp2
/// muls that would touch the zero `v²` coefficient of `(c0, c1, 0)`.
pub fn fp6MulBy01(a: Fp6, c0: Fp2, c1: Fp2) Fp6 {
    // a_a = a.c0 · c0
    const a_a = Fp2.mul(a.c0, c0);
    // b_b = a.c1 · c1
    const b_b = Fp2.mul(a.c1, c1);

    // t1 = c1 · (a.c1 + a.c2) − b_b, then × non-residue, then + a_a
    var t1 = Fp2.mul(c1, Fp2.add(a.c1, a.c2));
    t1 = Fp2.sub(t1, b_b);
    t1 = fp2MulByNonresidue(t1);
    t1 = Fp2.add(t1, a_a);

    // t3 = c0 · (a.c0 + a.c2) − a_a + b_b
    var t3 = Fp2.mul(c0, Fp2.add(a.c0, a.c2));
    t3 = Fp2.sub(t3, a_a);
    t3 = Fp2.add(t3, b_b);

    // t2 = (c0 + c1) · (a.c0 + a.c1) − a_a − b_b
    var t2 = Fp2.mul(Fp2.add(c0, c1), Fp2.add(a.c0, a.c1));
    t2 = Fp2.sub(t2, a_a);
    t2 = Fp2.sub(t2, b_b);

    return .{ .c0 = t1, .c1 = t2, .c2 = t3 };
}

/// Sparse Fp6 multiplication: `a · (c1·v)`. Used by `mulBy014` to
/// handle the upper-half multiplication where only `vw` is non-zero.
pub fn fp6MulBy1(a: Fp6, c1: Fp2) Fp6 {
    const b_b = Fp2.mul(a.c1, c1);

    // t1 = c1 · (a.c1 + a.c2) − b_b, then × non-residue.
    var t1 = Fp2.mul(c1, Fp2.add(a.c1, a.c2));
    t1 = Fp2.sub(t1, b_b);
    t1 = fp2MulByNonresidue(t1);

    // t2 = c1 · (a.c0 + a.c1) − b_b
    var t2 = Fp2.mul(c1, Fp2.add(a.c0, a.c1));
    t2 = Fp2.sub(t2, b_b);

    return .{ .c0 = t1, .c1 = t2, .c2 = b_b };
}

/// Sparse Fp12 multiplication: `f · (c0 + c1·v + c4·vw)`.
///
/// The right operand has only three non-zero Fp2 coefficients out of
/// the six positions of Fp12 — exactly the shape of an M-twist line.
/// Implemented exactly the way arkworks does it (Karatsuba-style) so a
/// future cross-check against a known-good Rust output is straightforward.
pub fn fp12MulBy014(f: Fp12, c0: Fp2, c1: Fp2, c4: Fp2) Fp12 {
    // aa = f.c0 · (c0, c1, 0)   — three Fp2 muls saved over a full mul.
    const aa = fp6MulBy01(f.c0, c0, c1);
    // bb = f.c1 · (0, c4, 0)
    const bb = fp6MulBy1(f.c1, c4);

    // o = c1 + c4
    const o = Fp2.add(c1, c4);

    // c1' = (f.c0 + f.c1) · (c0, o, 0) − aa − bb
    const sum = Fp6.add(f.c0, f.c1);
    var c1_new = fp6MulBy01(sum, c0, o);
    c1_new = Fp6.sub(c1_new, aa);
    c1_new = Fp6.sub(c1_new, bb);

    // c0' = bb · v + aa
    var c0_new = Fp6.mulByV(bb);
    c0_new = Fp6.add(c0_new, aa);

    return .{ .c0 = c0_new, .c1 = c1_new };
}

/// Multiply an `Fp2` value by an `Fp` scalar (lifts the Fp into Fp2 as
/// `(s, 0)` and componentwise-multiplies). Used by line evaluation to
/// scale the line coefficients by `P.x` and `P.y`.
pub fn fp2MulByFp(a: Fp2, s: Fp.Element) Fp2 {
    return .{
        .c0 = Fp.montMul(a.c0, s),
        .c1 = Fp.montMul(a.c1, s),
    };
}

// ---------------------------------------------------------------------------
// G2 homogeneous projective coordinates (NOT Jacobian).
//
// A "homogeneous" projective point (X : Y : Z) represents the affine
// point (X/Z, Y/Z), unlike Jacobian which uses (X/Z², Y/Z³). The pairing
// formulas from arkworks (and Costello / Aranha-Karabina-Longa-Gebotys-
// Lopez) work in this representation. The Miller loop runs entirely on
// G2HomProjective; we never need to project back to affine until the
// loop is done.
//
// We do NOT touch the existing `G2Projective` Jacobian type — keeping
// this separate avoids a representation change that would touch every
// G2 test in the package.
// ---------------------------------------------------------------------------

pub const G2HomProjective = struct {
    x: Fp2,
    y: Fp2,
    z: Fp2,

    pub fn fromAffine(p: G2Affine) G2HomProjective {
        if (p.infinity) return .{ .x = Fp2.zero(), .y = Fp2.one(), .z = Fp2.zero() };
        return .{ .x = p.x, .y = p.y, .z = Fp2.one() };
    }

    pub fn isIdentity(self: G2HomProjective) bool {
        return Fp2.eql(self.z, Fp2.zero());
    }

    /// Doubling step that also computes the line coefficients evaluated
    /// at a G1 point. Returns `(2T, (c0, c1, c4))` where the line
    /// coefficients live in Fp2 and need to be scaled by `(P.y, P.x)`
    /// before being fed to `fp12MulBy014`.
    ///
    /// Formulas from arkworks `bls12::g2::G2HomProjective::double_in_place`,
    /// which in turn cite the Costello / "Faster Explicit Formulas"
    /// reference. Uses `b' = 4(1+u)` for BLS12-381's G2 curve constant.
    pub fn doubleStep(self: *G2HomProjective) struct { Fp2, Fp2, Fp2 } {
        // a = (X · Y) / 2
        var a = Fp2.mul(self.x, self.y);
        a = halveFp2(a);
        // b = Y²
        const b = Fp2.square(self.y);
        // c = Z²
        const c = Fp2.square(self.z);
        // e = 3 b' c
        const three_c = Fp2.add(Fp2.add(c, c), c);
        const e = Fp2.mul(g2B(), three_c);
        // f = 3 e
        const f = Fp2.add(Fp2.add(e, e), e);
        // g = (b + f) / 2
        const g = halveFp2(Fp2.add(b, f));
        // h = (Y + Z)² − (b + c)
        const y_plus_z = Fp2.add(self.y, self.z);
        const h = Fp2.sub(Fp2.square(y_plus_z), Fp2.add(b, c));
        // i = e − b
        const i = Fp2.sub(e, b);
        // j = X²
        const j = Fp2.square(self.x);
        // e_square = e²
        const e_square = Fp2.square(e);

        // X' = a · (b − f)
        self.x = Fp2.mul(a, Fp2.sub(b, f));
        // Y' = g² − 3 e²
        self.y = Fp2.sub(Fp2.square(g), Fp2.add(Fp2.add(e_square, e_square), e_square));
        // Z' = b · h
        self.z = Fp2.mul(b, h);

        // M-twist line coefficients: (i, 3j, -h)
        const three_j = Fp2.add(Fp2.add(j, j), j);
        return .{ i, three_j, Fp2.neg(h) };
    }

    /// Mixed addition step: in-place adds `q` (affine) into `self`
    /// (homogeneous projective) and returns line coefficients evaluated
    /// against a G1 point.
    pub fn addStep(self: *G2HomProjective, q: G2Affine) struct { Fp2, Fp2, Fp2 } {
        // theta  = Y - q.y · Z
        const theta = Fp2.sub(self.y, Fp2.mul(q.y, self.z));
        // lambda = X - q.x · Z
        const lambda = Fp2.sub(self.x, Fp2.mul(q.x, self.z));
        // c = theta²
        const c = Fp2.square(theta);
        // d = lambda²
        const d = Fp2.square(lambda);
        // e = lambda · d
        const e = Fp2.mul(lambda, d);
        // f = Z · c
        const f = Fp2.mul(self.z, c);
        // g = X · d
        const g = Fp2.mul(self.x, d);
        // h = e + f − 2g
        const h = Fp2.sub(Fp2.add(e, f), Fp2.add(g, g));
        // X' = lambda · h
        self.x = Fp2.mul(lambda, h);
        // Y' = theta · (g − h) − e · Y
        self.y = Fp2.sub(Fp2.mul(theta, Fp2.sub(g, h)), Fp2.mul(e, self.y));
        // Z' = Z · e
        self.z = Fp2.mul(self.z, e);
        // j = theta · q.x − lambda · q.y
        const j = Fp2.sub(Fp2.mul(theta, q.x), Fp2.mul(lambda, q.y));

        // M-twist line coefficients: (j, -theta, lambda)
        return .{ j, Fp2.neg(theta), lambda };
    }
};

/// Halve an Fp2 element. `(a₀ + a₁·u)/2 = (a₀/2) + (a₁/2)·u`.
inline fn halveFp2(a: Fp2) Fp2 {
    return .{ .c0 = fpHalve(a.c0), .c1 = fpHalve(a.c1) };
}

// ---------------------------------------------------------------------------
// Optimal Ate Miller loop for BLS12-381.
//
// Walks the bits of `|x|` from the second-most-significant down to bit 0,
// squaring the accumulator and applying a doubling line evaluation each
// step, with an extra addition line evaluation when the bit is set. The
// final result is conjugated when `x` is negative.
// ---------------------------------------------------------------------------

/// `BLS_X_ABS` as raw bytes ordered MSB → LSB so the loop can pull bits
/// out from the top down without recomputing the bit length on every
/// iteration. The constant has bit length 64 (top bit set).
const BLS_X_ABS_BITS_MSB: [BLS_X_LOOP_BITS]u1 = blk: {
    @setEvalBranchQuota(10000);
    var bits: [BLS_X_LOOP_BITS]u1 = undefined;
    var idx: usize = 0;
    while (idx < BLS_X_LOOP_BITS) : (idx += 1) {
        const i = BLS_X_LOOP_BITS - 1 - idx;
        bits[idx] = @intCast((BLS_X_ABS >> @intCast(i)) & 1);
    }
    break :blk bits;
};

/// Apply a line evaluation produced by `doubleStep`/`addStep` to a
/// running Fp12 accumulator, scaling the M-twist line coefficients by
/// `(P.x, P.y)` first.
///
/// `coeffs = (c0, c1, c4)` where `c4` already gets scaled by `P.y` and
/// `c1` already gets scaled by `P.x`. `c0` stays in Fp2.
fn ellM(
    f: Fp12,
    coeffs: struct { Fp2, Fp2, Fp2 },
    p: G1Affine,
) Fp12 {
    // For M-twist:
    //   c2.mul_assign_by_fp(p.y);
    //   c1.mul_assign_by_fp(p.x);
    //   f.mul_by_014(c0, c1, c2)
    // (where the third coefficient is at position 4 = vw).
    const c0 = coeffs[0];
    const c1_scaled = fp2MulByFp(coeffs[1], p.x);
    const c4_scaled = fp2MulByFp(coeffs[2], p.y);
    return fp12MulBy014(f, c0, c1_scaled, c4_scaled);
}

/// Optimal Ate Miller loop. Inputs are an affine G1 point `P` and an
/// affine G2 point `Q`; the result is an Fp12 element that becomes the
/// pairing value after the final exponentiation.
///
/// Identity inputs short-circuit to `Fp12.one()` (the identity-pair
/// convention; the final exponentiation maps that to the multiplicative
/// identity in the target group).
pub fn millerLoop(p: G1Affine, q: G2Affine) Fp12 {
    if (p.infinity or q.infinity) return Fp12.one();

    var f = Fp12.one();
    var t = G2HomProjective.fromAffine(q);

    // Walk bits from BLS_X_LOOP_BITS-2 down to 0 — that is, skip the
    // top bit (which is just "start with T = Q, f = 1") and do a
    // double-line for each remaining bit, plus an add-line when the
    // bit is set.
    var i: usize = 1;
    while (i < BLS_X_LOOP_BITS) : (i += 1) {
        f = Fp12.square(f);
        const dbl_coeffs = t.doubleStep();
        f = ellM(f, dbl_coeffs, p);

        if (BLS_X_ABS_BITS_MSB[i] == 1) {
            const add_coeffs = t.addStep(q);
            f = ellM(f, add_coeffs, p);
        }
    }

    // x is negative for BLS12-381 (`x = -0xd201000000010000`); the
    // resulting Miller value picks up an inversion. Conjugating an
    // Fp12 element equals raising it to `p^6`, which differs from
    // a true inverse by a factor of `(p^12 − 1)` — i.e., 1 in Fp12* —
    // so the easy part of final exponentiation absorbs the difference
    // (see arkworks `multi_miller_loop`).
    if (BLS_X_IS_NEGATIVE) {
        f = Fp12.conjugate(f);
    }

    return f;
}

// ---------------------------------------------------------------------------
// Final exponentiation.
//
// Computes `f^((p^12 - 1) / r)` in two phases:
//
//   1. easy part:  `f^((p^6 - 1)(p^2 + 1))`. After this `f` lives in
//                  the cyclotomic subgroup of order Φ_12(p) = p^4 - p^2 + 1.
//   2. hard part:  `f^((p^4 - p^2 + 1) / r)`. Computed via an addition
//                  chain over `x` plus Frobenius applications. Uses the
//                  same chain as arkworks (which itself follows the
//                  ConsenSys/Gurvy implementation; see eprint 2020/875).
// ---------------------------------------------------------------------------

/// Raise `f` to the BLS x parameter (`f^x`). Since `x` is negative for
/// BLS12-381, this conjugates the result of `f^|x|` (which equals the
/// true inverse for cyclotomic elements).
fn expByX(f: Fp12) Fp12 {
    // Use plain `Fp12.pow` against the absolute value of x. A faster
    // cyclotomic-aware exponentiation can land later — the addition
    // chain shape doesn't change, only the per-step cost.
    const x_abs_limbs: [1]u64 = .{BLS_X_ABS};
    var result = Fp12.pow(f, 1, x_abs_limbs);
    if (BLS_X_IS_NEGATIVE) {
        result = Fp12.conjugate(result);
    }
    return result;
}

/// Hard part of the BLS12-381 final exponentiation, expressed as the
/// addition chain from arkworks (eprint 2020/875). All intermediate
/// values live in the cyclotomic subgroup so conjugation is the same
/// as inversion — every `cyclotomic_inverse_in_place` in the upstream
/// implementation maps to a plain `Fp12.conjugate` here.
pub fn fp12FinalExpHard(input: Fp12) Fp12 {
    var r = input;
    var y0 = Fp12.square(r);
    var y1 = expByX(r);
    var y2 = Fp12.conjugate(r);
    y1 = Fp12.mul(y1, y2);
    y2 = expByX(y1);
    y1 = Fp12.conjugate(y1);
    y1 = Fp12.mul(y1, y2);
    y2 = expByX(y1);
    y1 = fp12Frobenius(y1);
    y1 = Fp12.mul(y1, y2);
    r = Fp12.mul(r, y0);
    y0 = expByX(y1);
    y2 = expByX(y0);
    y0 = y1;
    y0 = fp12FrobeniusSquared(y0);
    y1 = Fp12.conjugate(y1);
    y1 = Fp12.mul(y1, y2);
    y1 = Fp12.mul(y1, y0);
    r = Fp12.mul(r, y1);
    return r;
}

/// Full final exponentiation: easy part followed by hard part.
pub fn fp12FinalExp(f: Fp12) Fp12 {
    return fp12FinalExpHard(fp12FinalExpEasy(f));
}

/// Optimal Ate pairing for BLS12-381: `e(P, Q) = millerLoop(P, Q)^((p^12 - 1)/r)`.
///
/// Returns `Fp12.one()` if either input is the identity (the pairing is
/// trivially 1 in that case after final exponentiation).
pub fn pairing(p: G1Affine, q: G2Affine) Fp12 {
    if (p.infinity or q.infinity) return Fp12.one();
    return fp12FinalExp(millerLoop(p, q));
}

// ---------------------------------------------------------------------------
// Tests — these exercise the BLS12-381 Fp instance against arithmetic
// laws and a few hand-computed values. Real cross-implementation
// vectors against `blst` come once a real `blst` test harness is in
// place.
// ---------------------------------------------------------------------------

const testing = std.testing;

test "Fp.zero / Fp.one" {
    const z = Fp.zero();
    try testing.expect(bigint.isZero(6, z));
    const o = Fp.one();
    // `one` is `R mod p`, NOT raw 1, so don't test for limbs == [1, 0, ...].
    // Instead, check that toRaw(one) == 1.
    const raw = Fp.toRaw(o);
    try testing.expectEqual(@as(u64, 1), raw[0]);
    inline for (1..6) |i| try testing.expectEqual(@as(u64, 0), raw[i]);
}

test "Fp identity laws" {
    const a = Fp.fromRaw(.{ 0x0102030405060708, 0x1112131415161718, 0x2122232425262728, 0x3132333435363738, 0x4142434445464748, 0x0102030400000000 });
    try testing.expect(Fp.eql(Fp.add(a, Fp.zero()), a));
    try testing.expect(Fp.eql(Fp.add(Fp.zero(), a), a));
    try testing.expect(Fp.eql(Fp.montMul(a, Fp.one()), a));
    try testing.expect(Fp.eql(Fp.montMul(Fp.one(), a), a));
    try testing.expect(Fp.eql(Fp.add(a, Fp.neg(a)), Fp.zero()));
}

test "Fp.add wraps around the modulus" {
    const one_e = Fp.fromRaw(.{ 1, 0, 0, 0, 0, 0 });
    var p_minus_one_raw: [6]u64 = FP_MODULUS;
    p_minus_one_raw[0] -= 1;
    const p_minus_one = Fp.fromRaw(p_minus_one_raw);
    const sum = Fp.add(one_e, p_minus_one);
    try testing.expect(Fp.eql(sum, Fp.zero()));
}

test "Fp.montMul: 2 * 3 = 6" {
    const two = Fp.fromRaw(.{ 2, 0, 0, 0, 0, 0 });
    const three = Fp.fromRaw(.{ 3, 0, 0, 0, 0, 0 });
    const six = Fp.fromRaw(.{ 6, 0, 0, 0, 0, 0 });
    const product = Fp.montMul(two, three);
    try testing.expect(Fp.eql(product, six));
}

test "Fp.montMul: distributive over add" {
    const a = Fp.fromRaw(.{ 0x12345678, 0xabcdef00, 0x11111111, 0, 0, 0 });
    const b = Fp.fromRaw(.{ 0xfedcba98, 0x76543210, 0, 0x22222222, 0, 0 });
    const c = Fp.fromRaw(.{ 0x42, 0, 0, 0, 0x33333333, 0 });
    const lhs = Fp.montMul(Fp.add(a, b), c);
    const rhs = Fp.add(Fp.montMul(a, c), Fp.montMul(b, c));
    try testing.expect(Fp.eql(lhs, rhs));
}

test "Fp.montMul: associativity" {
    const a = Fp.fromRaw(.{ 7, 0, 0, 0, 0, 0 });
    const b = Fp.fromRaw(.{ 11, 0, 0, 0, 0, 0 });
    const c = Fp.fromRaw(.{ 13, 0, 0, 0, 0, 0 });
    const lhs = Fp.montMul(Fp.montMul(a, b), c);
    const rhs = Fp.montMul(a, Fp.montMul(b, c));
    try testing.expect(Fp.eql(lhs, rhs));
    // Hand check: 7 * 11 * 13 = 1001
    const raw = Fp.toRaw(lhs);
    try testing.expectEqual(@as(u64, 1001), raw[0]);
    inline for (1..6) |i| try testing.expectEqual(@as(u64, 0), raw[i]);
}

test "Fp.toRaw round-trips a near-modulus value" {
    var raw: [6]u64 = FP_MODULUS;
    raw[0] -= 7;
    const e = Fp.fromRaw(raw);
    const back = Fp.toRaw(e);
    try testing.expectEqual(raw, back);
}

test "Fp.fromBytesLeReduced rejects ≥ p" {
    var bytes: [48]u8 = .{0xff} ** 48;
    try testing.expectError(error.NotInField, Fp.fromBytesLeReduced(&bytes));

    var ok: [48]u8 = .{0} ** 48;
    ok[0] = 1;
    const e = try Fp.fromBytesLeReduced(&ok);
    try testing.expect(Fp.eql(e, Fp.one()));
}

test "Fp.toBytesLe round-trips fromBytesLeReduced" {
    var input: [48]u8 = .{0} ** 48;
    input[0] = 0x42;
    input[1] = 0x13;
    input[47] = 0x01; // High byte must respect the prime ceiling.
    const e = try Fp.fromBytesLeReduced(&input);
    var output: [48]u8 = undefined;
    Fp.toBytesLe(e, &output);
    try testing.expectEqualSlices(u8, &input, &output);
}

test "Fp: (p-1)^2 + (2p-1) ≡ 0 mod p" {
    // (p-1)^2 = p^2 - 2p + 1 ≡ 1 mod p, so (p-1)^2 + (-1) ≡ 0.
    // We test it as: (p-1)^2 + (p-1) ≡ p-1+1 = 0 ... no wait.
    // Simpler test: (p-1) + 1 = 0
    var raw: [6]u64 = FP_MODULUS;
    raw[0] -= 1;
    const p_minus_one = Fp.fromRaw(raw);
    const one = Fp.fromRaw(.{ 1, 0, 0, 0, 0, 0 });
    const sum = Fp.add(p_minus_one, one);
    try testing.expect(Fp.eql(sum, Fp.zero()));
    // And (p-1) * (p-1) = 1 mod p
    const sq = Fp.montMul(p_minus_one, p_minus_one);
    const one_mont = Fp.one();
    try testing.expect(Fp.eql(sq, one_mont));
}

test "Fp.inv: a * a^-1 = 1 (6-limb Fermat inversion)" {
    // 6-limb Fermat inversion is the most expensive operation in the
    // package — ~381 squarings + ~190 multiplies. Verify it produces
    // the multiplicative inverse for a representative non-trivial
    // value.
    const a = Fp.fromRaw(.{ 0x123456789abcdef0, 0xfedcba9876543210, 0x1111222233334444, 0x5555666677778888, 0x9999aaaabbbbcccc, 0x0123 });
    const inv_a = Fp.inv(a);
    const product = Fp.montMul(a, inv_a);
    try testing.expect(Fp.eql(product, Fp.one()));
}

test "Fp.inv: inv(1) = 1 (6 limbs)" {
    const one = Fp.one();
    try testing.expect(Fp.eql(Fp.inv(one), one));
}

test "Fp.inv: inv(zero) = zero" {
    const z = Fp.zero();
    try testing.expect(Fp.eql(Fp.inv(z), z));
}

test "Fp.square: 5^2 = 25 (6 limbs)" {
    const five = Fp.fromRaw(.{ 5, 0, 0, 0, 0, 0 });
    const sq = Fp.square(five);
    const twenty_five = Fp.fromRaw(.{ 25, 0, 0, 0, 0, 0 });
    try testing.expect(Fp.eql(sq, twenty_five));
}

test "Fp: (p+1)/4 derivation gives a sane result" {
    // Sanity check: 4 * ((p+1)/4) - 1 == p, modulo wrap.
    // We can't easily verify the constant by hand, so instead test
    // that fpSqrt(4) returns 2 (and squaring 2 gives 4 back).
    const four = Fp.fromRaw(.{ 4, 0, 0, 0, 0, 0 });
    const two = Fp.fromRaw(.{ 2, 0, 0, 0, 0, 0 });
    const root = fpSqrt(four);
    // Square root of 4 should be ±2; either branch must square to 4.
    try testing.expect(fpIsSquareRoot(four, root));
    // The "positive" root (smaller of the two) should match 2.
    if (!Fp.eql(root, two)) {
        try testing.expect(Fp.eql(root, Fp.neg(two)));
    }
}

test "Fp.sqrt: 25 -> ±5" {
    const twenty_five = Fp.fromRaw(.{ 25, 0, 0, 0, 0, 0 });
    const five = Fp.fromRaw(.{ 5, 0, 0, 0, 0, 0 });
    const root = fpSqrt(twenty_five);
    try testing.expect(fpIsSquareRoot(twenty_five, root));
    try testing.expect(Fp.eql(root, five) or Fp.eql(root, Fp.neg(five)));
}

test "Fr identity laws" {
    const a = Fr.fromRaw(.{ 0x12345678, 0xdeadbeef, 0xabad1dea, 0x0123 });
    try testing.expect(Fr.eql(Fr.add(a, Fr.zero()), a));
    try testing.expect(Fr.eql(Fr.montMul(a, Fr.one()), a));
    try testing.expect(Fr.eql(Fr.add(a, Fr.neg(a)), Fr.zero()));
}

test "Fr: 2 * 3 = 6" {
    const two = Fr.fromRaw(.{ 2, 0, 0, 0 });
    const three = Fr.fromRaw(.{ 3, 0, 0, 0 });
    const six = Fr.fromRaw(.{ 6, 0, 0, 0 });
    try testing.expect(Fr.eql(Fr.montMul(two, three), six));
}

test "Fr: distributive over add" {
    const a = Fr.fromRaw(.{ 0x12345678, 0xabcd, 0, 0 });
    const b = Fr.fromRaw(.{ 0xdeadbeef, 0, 0xfeedface, 0 });
    const c = Fr.fromRaw(.{ 0x42, 0, 0, 0xbeef });
    const lhs = Fr.montMul(Fr.add(a, b), c);
    const rhs = Fr.add(Fr.montMul(a, c), Fr.montMul(b, c));
    try testing.expect(Fr.eql(lhs, rhs));
}

test "Fr: (r-1) + 1 = 0" {
    var r_minus_one_raw: [4]u64 = FR_MODULUS;
    r_minus_one_raw[0] -= 1;
    const r_minus_one = Fr.fromRaw(r_minus_one_raw);
    const one_e = Fr.fromRaw(.{ 1, 0, 0, 0 });
    try testing.expect(Fr.eql(Fr.add(r_minus_one, one_e), Fr.zero()));
}

test "Fr.inv: a * a^-1 = 1 (4-limb scalar field)" {
    const a = Fr.fromRaw(.{ 0x123456789abcdef0, 0xfedcba9876543210, 0x1122334455667788, 0x0123456789abcdef });
    const inv_a = Fr.inv(a);
    try testing.expect(Fr.eql(Fr.montMul(a, inv_a), Fr.one()));
}

test "isInG1Subgroup: G1 generator is in subgroup" {
    try testing.expect(isInG1Subgroup(g1Generator()));
}

test "isInG1Subgroup: identity is in every subgroup" {
    try testing.expect(isInG1Subgroup(G1Affine.identity()));
}

test "isInG2Subgroup: G2 generator is in subgroup" {
    try testing.expect(isInG2Subgroup(g2Generator()));
}

test "isInG2Subgroup: identity is in every subgroup" {
    try testing.expect(isInG2Subgroup(G2Affine.identity()));
}

// ---------------------------------------------------------------------------
// Optimal Ate pairing tests
// ---------------------------------------------------------------------------

test "pairing: e(O, Q) = 1" {
    const id_g1 = G1Affine.identity();
    const g2 = g2Generator();
    const result = pairing(id_g1, g2);
    try testing.expect(Fp12.eql(result, Fp12.one()));
}

test "pairing: e(P, O) = 1" {
    const g1 = g1Generator();
    const id_g2 = G2Affine.identity();
    const result = pairing(g1, id_g2);
    try testing.expect(Fp12.eql(result, Fp12.one()));
}

test "pairing: non-degenerate (e(g1, g2) ≠ 1)" {
    const g1 = g1Generator();
    const g2 = g2Generator();
    const result = pairing(g1, g2);
    try testing.expect(!Fp12.eql(result, Fp12.one()));
}

test "pairing: deterministic" {
    const g1 = g1Generator();
    const g2 = g2Generator();
    const result1 = pairing(g1, g2);
    const result2 = pairing(g1, g2);
    try testing.expect(Fp12.eql(result1, result2));
}

test "pairing bilinearity: e(2P, Q) = e(P, Q)^2" {
    // e(2P, Q) should equal e(P, 2Q) should equal e(P, Q)².
    const g1 = g1Generator();
    const g2 = g2Generator();

    const two_g1 = g1.double();
    const two_g2 = g2.double();

    const e_2pq = pairing(two_g1, g2);
    const e_p2q = pairing(g1, two_g2);
    const e_pq = pairing(g1, g2);
    const e_pq_sq = Fp12.square(e_pq);

    try testing.expect(Fp12.eql(e_2pq, e_p2q));
    try testing.expect(Fp12.eql(e_2pq, e_pq_sq));
}

test "pairing bilinearity: e(aP, Q) = e(P, Q)^a for small a" {
    // Pick a = 5 so the test runs in a sane amount of time. The full
    // 4-limb scalar exponent path is exercised by the (aP, bQ) test.
    const g1 = g1Generator();
    const g2 = g2Generator();

    const five_g1 = g1.mul(1, .{5});
    const e_5pq = pairing(five_g1, g2);

    const e_pq = pairing(g1, g2);
    const e_pq_5 = Fp12.pow(e_pq, 1, .{5});

    try testing.expect(Fp12.eql(e_5pq, e_pq_5));
}

test "pairing bilinearity: e(aP, bQ) = e(P, Q)^(ab)" {
    const g1 = g1Generator();
    const g2 = g2Generator();

    // Pick small scalars so the affine scalar muls don't dominate
    // test runtime, but large enough that ab would catch off-by-one
    // bugs in the addition chain.
    const a: u64 = 7;
    const b: u64 = 11;
    const ab: u64 = a * b;

    const ap = g1.mul(1, .{a});
    const bq = g2.mul(1, .{b});

    const lhs = pairing(ap, bq);
    const rhs = Fp12.pow(pairing(g1, g2), 1, .{ab});

    try testing.expect(Fp12.eql(lhs, rhs));
}

test "pairing: e(P, -Q) = e(-P, Q) = e(P, Q)^(-1)" {
    // The pairing is bilinear, so swapping the sign on either input
    // should produce the inverse in the target group. After final
    // exponentiation, the inverse equals the conjugate.
    const g1 = g1Generator();
    const g2 = g2Generator();

    const e_pq = pairing(g1, g2);
    const e_neg_p_q = pairing(g1.neg(), g2);
    const e_p_neg_q = pairing(g1, g2.neg());

    // After final exponentiation we're in the cyclotomic subgroup,
    // so conjugate(x) = x^(-1).
    const inv_e_pq = Fp12.conjugate(e_pq);

    try testing.expect(Fp12.eql(e_neg_p_q, inv_e_pq));
    try testing.expect(Fp12.eql(e_p_neg_q, inv_e_pq));
    // And e(-P, Q) · e(P, Q) = 1.
    try testing.expect(Fp12.eql(Fp12.mul(e_neg_p_q, e_pq), Fp12.one()));
}

// ---------------------------------------------------------------------------
// G1Projective tests
// ---------------------------------------------------------------------------

test "G1Projective: identity round-trip via toAffine" {
    try testing.expect(G1Projective.identity().toAffine().isIdentity());
}

test "G1Projective: fromAffine -> toAffine is identity" {
    const g = g1Generator();
    const proj = G1Projective.fromAffine(g);
    const back = proj.toAffine();
    try testing.expect(G1Affine.eql(back, g));
}

test "G1Projective: double matches G1Affine.double" {
    const g = g1Generator();
    const aff_two_g = g.double();
    const proj_two_g = G1Projective.fromAffine(g).double().toAffine();
    try testing.expect(G1Affine.eql(aff_two_g, proj_two_g));
}

test "G1Projective: add matches G1Affine.add for distinct points" {
    const g = g1Generator();
    const two_g_aff = g.double();
    const three_g_aff = two_g_aff.add(g);
    // Same chain via projective.
    const g_p = G1Projective.fromAffine(g);
    const two_g_p = g_p.double();
    const three_g_p = two_g_p.add(g_p).toAffine();
    try testing.expect(G1Affine.eql(three_g_aff, three_g_p));
}

test "G1Projective: P + (-P) = identity" {
    const g = g1Generator();
    const g_p = G1Projective.fromAffine(g);
    const neg_g_p = G1Projective.fromAffine(g.neg());
    const sum = g_p.add(neg_g_p);
    try testing.expect(sum.isIdentity());
    try testing.expect(sum.toAffine().isIdentity());
}

test "G1Projective: add(P, P) falls through to double" {
    const g_p = G1Projective.fromAffine(g1Generator());
    const sum = g_p.add(g_p);
    const doubled = g_p.double();
    try testing.expect(G1Projective.eql(sum, doubled));
}

test "G1Projective: associativity through projective then back to affine" {
    const g = g1Generator();
    const a = G1Projective.fromAffine(g);
    const b = a.double();
    const c = b.add(a);
    // (a + b) + c == a + (b + c)
    const lhs = a.add(b).add(c).toAffine();
    const rhs = a.add(b.add(c)).toAffine();
    try testing.expect(G1Affine.eql(lhs, rhs));
}

// ---------------------------------------------------------------------------
// G2Projective tests — same shape as G1Projective.
// ---------------------------------------------------------------------------

test "G2Projective: identity round-trip via toAffine" {
    try testing.expect(G2Projective.identity().toAffine().isIdentity());
}

test "G2Projective: fromAffine -> toAffine is identity" {
    const g = g2Generator();
    const back = G2Projective.fromAffine(g).toAffine();
    try testing.expect(G2Affine.eql(back, g));
}

test "G2Projective: double matches G2Affine.double" {
    const g = g2Generator();
    const aff_two_g = g.double();
    const proj_two_g = G2Projective.fromAffine(g).double().toAffine();
    try testing.expect(G2Affine.eql(aff_two_g, proj_two_g));
}

test "G2Projective: add matches G2Affine.add for distinct points" {
    const g = g2Generator();
    const two_g_aff = g.double();
    const three_g_aff = two_g_aff.add(g);
    const g_p = G2Projective.fromAffine(g);
    const two_g_p = g_p.double();
    const three_g_p = two_g_p.add(g_p).toAffine();
    try testing.expect(G2Affine.eql(three_g_aff, three_g_p));
}

test "G2Projective: P + (-P) = identity" {
    const g = g2Generator();
    const g_p = G2Projective.fromAffine(g);
    const neg_g_p = G2Projective.fromAffine(g.neg());
    try testing.expect(g_p.add(neg_g_p).isIdentity());
}

test "G2Projective: add(P, P) falls through to double" {
    const g_p = G2Projective.fromAffine(g2Generator());
    try testing.expect(G2Projective.eql(g_p.add(g_p), g_p.double()));
}

test "G1Projective.mul: 0*G = identity, 1*G = G" {
    const g = G1Projective.fromAffine(g1Generator());
    try testing.expect(g.mul(1, .{0}).isIdentity());
    try testing.expect(G1Projective.eql(g.mul(1, .{1}), g));
}

test "G1Projective.mul matches affine for small scalars" {
    const g = g1Generator();
    const g_p = G1Projective.fromAffine(g);
    // 5 * G via projective and affine paths must agree.
    const five_g_proj = g_p.mul(1, .{5}).toAffine();
    const five_g_aff = g.mul(1, .{5});
    try testing.expect(G1Affine.eql(five_g_proj, five_g_aff));
}

test "G1Projective.mul: 7 * G via projective matches G + G + ... + G" {
    const g_p = G1Projective.fromAffine(g1Generator());
    const seven_g = g_p.mul(1, .{7});
    var manual = g_p;
    var i: usize = 0;
    while (i < 6) : (i += 1) manual = manual.add(g_p);
    try testing.expect(G1Projective.eql(seven_g, manual));
}

test "G2Projective.mul: 0*G = identity, 1*G = G" {
    const g = G2Projective.fromAffine(g2Generator());
    try testing.expect(g.mul(1, .{0}).isIdentity());
    try testing.expect(G2Projective.eql(g.mul(1, .{1}), g));
}

test "G2Projective.mul matches affine for small scalars" {
    const g = g2Generator();
    const g_p = G2Projective.fromAffine(g);
    const five_g_proj = g_p.mul(1, .{5}).toAffine();
    const five_g_aff = g.mul(1, .{5});
    try testing.expect(G2Affine.eql(five_g_proj, five_g_aff));
}

test "Fp.sqrt: non-residue check" {
    // We don't have an analytical "is this a residue" predicate yet,
    // but for any non-square `a`, `fpIsSquareRoot(a, fpSqrt(a))`
    // should return false. Picking 5 — quadratic residue status of
    // small primes mod the BLS12-381 base prime is not obvious by
    // hand, so we just assert that the round-trip predicate works
    // correctly: either it round-trips (residue) or it doesn't.
    const five = Fp.fromRaw(.{ 5, 0, 0, 0, 0, 0 });
    const root = fpSqrt(five);
    const round_trip = Fp.square(root);
    // round_trip is either 5 (residue) or -5 (non-residue).
    try testing.expect(Fp.eql(round_trip, five) or Fp.eql(round_trip, Fp.neg(five)));
}

// ---------------------------------------------------------------------------
// Fp2 tests
// ---------------------------------------------------------------------------

fn fpFromU64(n: u64) Fp.Element {
    return Fp.fromRaw(.{ n, 0, 0, 0, 0, 0 });
}

fn fp2FromU64Pair(c0: u64, c1: u64) Fp2 {
    return .{ .c0 = fpFromU64(c0), .c1 = fpFromU64(c1) };
}

test "Fp2 identity laws" {
    const a = fp2FromU64Pair(7, 11);
    try testing.expect(Fp2.eql(Fp2.add(a, Fp2.zero()), a));
    try testing.expect(Fp2.eql(Fp2.add(Fp2.zero(), a), a));
    try testing.expect(Fp2.eql(Fp2.mul(a, Fp2.one()), a));
    try testing.expect(Fp2.eql(Fp2.mul(Fp2.one(), a), a));
    try testing.expect(Fp2.eql(Fp2.add(a, Fp2.neg(a)), Fp2.zero()));
}

test "Fp2.mul: u² = -1" {
    // u is (0 + 1·u), and u² should equal (-1 + 0·u).
    const u = fp2FromU64Pair(0, 1);
    const u_sq = Fp2.mul(u, u);
    const minus_one_in_fp = Fp.neg(Fp.one());
    const expected: Fp2 = .{ .c0 = minus_one_in_fp, .c1 = Fp.zero() };
    try testing.expect(Fp2.eql(u_sq, expected));
}

test "Fp2.square: equivalent to mul(a, a)" {
    const a = fp2FromU64Pair(0x1234, 0xabcd);
    try testing.expect(Fp2.eql(Fp2.square(a), Fp2.mul(a, a)));
}

test "Fp2.mul: distributive over add" {
    const a = fp2FromU64Pair(3, 5);
    const b = fp2FromU64Pair(7, 11);
    const c = fp2FromU64Pair(13, 17);
    const lhs = Fp2.mul(Fp2.add(a, b), c);
    const rhs = Fp2.add(Fp2.mul(a, c), Fp2.mul(b, c));
    try testing.expect(Fp2.eql(lhs, rhs));
}

test "Fp2.mul: hand-computed value" {
    // (2 + 3u) * (5 + 7u) = (2*5 - 3*7) + (2*7 + 3*5)u
    //                     = (10 - 21)  + (14 + 15)u
    //                     = -11 + 29u
    const a = fp2FromU64Pair(2, 3);
    const b = fp2FromU64Pair(5, 7);
    const product = Fp2.mul(a, b);
    const minus_eleven = Fp.neg(fpFromU64(11));
    const expected: Fp2 = .{ .c0 = minus_eleven, .c1 = fpFromU64(29) };
    try testing.expect(Fp2.eql(product, expected));
}

test "Fp2.inv: a * a^-1 = 1" {
    const a = fp2FromU64Pair(0x12345678, 0x9abcdef0);
    const inv_a = Fp2.inv(a);
    const product = Fp2.mul(a, inv_a);
    try testing.expect(Fp2.eql(product, Fp2.one()));
}

test "Fp2.inv: inv(0) = 0" {
    try testing.expect(Fp2.eql(Fp2.inv(Fp2.zero()), Fp2.zero()));
}

test "Fp2.inv: inv(1) = 1" {
    try testing.expect(Fp2.eql(Fp2.inv(Fp2.one()), Fp2.one()));
}

test "Fp2.sqrt: round-trips a hand-built square" {
    // Build a = (3 + 5u)² and check that fp2Sqrt(a) returns ±(3 + 5u).
    const original = fp2FromU64Pair(3, 5);
    const sq = Fp2.square(original);
    const root = try fp2Sqrt(sq);
    const round_trip = Fp2.square(root);
    try testing.expect(Fp2.eql(round_trip, sq));
    try testing.expect(Fp2.eql(root, original) or Fp2.eql(root, Fp2.neg(original)));
}

test "Fp2.sqrt: round-trips a 6-limb random square" {
    const original: Fp2 = .{
        .c0 = Fp.fromRaw(.{ 0x12345678, 0x9abcdef0, 0x1111, 0, 0, 0 }),
        .c1 = Fp.fromRaw(.{ 0xfedcba98, 0x1234, 0, 0x4321, 0, 0 }),
    };
    const sq = Fp2.square(original);
    const root = try fp2Sqrt(sq);
    try testing.expect(Fp2.eql(Fp2.square(root), sq));
}

test "Fp2.sqrt: zero -> zero" {
    const root = try fp2Sqrt(Fp2.zero());
    try testing.expect(Fp2.eql(root, Fp2.zero()));
}

test "Fp2.sqrt: 1 -> ±1" {
    const root = try fp2Sqrt(Fp2.one());
    const sq = Fp2.square(root);
    try testing.expect(Fp2.eql(sq, Fp2.one()));
}

test "Fp2.sqrt: pure-Fp residue" {
    // 4 = 2² in Fp; in Fp2 the sqrt should be (2, 0) (or its negation).
    const four_in_fp2: Fp2 = .{ .c0 = Fp.fromRaw(.{ 4, 0, 0, 0, 0, 0 }), .c1 = Fp.zero() };
    const root = try fp2Sqrt(four_in_fp2);
    const sq = Fp2.square(root);
    try testing.expect(Fp2.eql(sq, four_in_fp2));
}

test "Fp2.sqrt: pure-Fp non-residue takes the (0, sqrt(-a₀)) branch" {
    // -1 in Fp2 is u² = (0, 0).c0 = -1, but actually (0, 1) since u² = -1.
    // Easier test: a = -4 has sqrt 2u, since (2u)² = -4.
    const minus_four_in_fp2: Fp2 = .{ .c0 = Fp.neg(Fp.fromRaw(.{ 4, 0, 0, 0, 0, 0 })), .c1 = Fp.zero() };
    const root = try fp2Sqrt(minus_four_in_fp2);
    const sq = Fp2.square(root);
    try testing.expect(Fp2.eql(sq, minus_four_in_fp2));
}

// ---------------------------------------------------------------------------
// Fp6 tests
// ---------------------------------------------------------------------------

fn fp6FromInts(c0_a: u64, c0_b: u64, c1_a: u64, c1_b: u64, c2_a: u64, c2_b: u64) Fp6 {
    return .{
        .c0 = fp2FromU64Pair(c0_a, c0_b),
        .c1 = fp2FromU64Pair(c1_a, c1_b),
        .c2 = fp2FromU64Pair(c2_a, c2_b),
    };
}

test "Fp6 identity laws" {
    const a = fp6FromInts(1, 2, 3, 4, 5, 6);
    try testing.expect(Fp6.eql(Fp6.add(a, Fp6.zero()), a));
    try testing.expect(Fp6.eql(Fp6.mul(a, Fp6.one()), a));
    try testing.expect(Fp6.eql(Fp6.mul(Fp6.one(), a), a));
    try testing.expect(Fp6.eql(Fp6.add(a, Fp6.neg(a)), Fp6.zero()));
}

test "Fp6.mul: v³ = 1+u" {
    // v in Fp6 = (0 + 0u, 1 + 0u, 0 + 0u) — coefficient of v is 1.
    const v: Fp6 = .{
        .c0 = Fp2.zero(),
        .c1 = Fp2.one(),
        .c2 = Fp2.zero(),
    };
    const v_sq = Fp6.mul(v, v);
    const v_cubed = Fp6.mul(v_sq, v);
    // v² should be (0, 0, 1) and v³ should be (1+u, 0, 0).
    const expected_v_sq: Fp6 = .{
        .c0 = Fp2.zero(),
        .c1 = Fp2.zero(),
        .c2 = Fp2.one(),
    };
    try testing.expect(Fp6.eql(v_sq, expected_v_sq));
    const one_plus_u: Fp2 = .{ .c0 = Fp.one(), .c1 = Fp.one() };
    const expected_v_cubed: Fp6 = .{
        .c0 = one_plus_u,
        .c1 = Fp2.zero(),
        .c2 = Fp2.zero(),
    };
    try testing.expect(Fp6.eql(v_cubed, expected_v_cubed));
}

test "Fp6.mul: distributive over add" {
    const a = fp6FromInts(1, 2, 3, 4, 5, 6);
    const b = fp6FromInts(7, 8, 9, 10, 11, 12);
    const c = fp6FromInts(13, 14, 15, 16, 17, 18);
    const lhs = Fp6.mul(Fp6.add(a, b), c);
    const rhs = Fp6.add(Fp6.mul(a, c), Fp6.mul(b, c));
    try testing.expect(Fp6.eql(lhs, rhs));
}

test "Fp6.mul: associative" {
    const a = fp6FromInts(2, 3, 5, 7, 11, 13);
    const b = fp6FromInts(17, 19, 23, 29, 31, 37);
    const c = fp6FromInts(41, 43, 47, 53, 59, 61);
    const lhs = Fp6.mul(Fp6.mul(a, b), c);
    const rhs = Fp6.mul(a, Fp6.mul(b, c));
    try testing.expect(Fp6.eql(lhs, rhs));
}

test "Fp6.square: equivalent to mul(a, a)" {
    const a = fp6FromInts(11, 13, 17, 19, 23, 29);
    try testing.expect(Fp6.eql(Fp6.square(a), Fp6.mul(a, a)));
}

test "Fp6.mulByV: equivalent to mul by (0, 1, 0)" {
    const a = fp6FromInts(11, 13, 17, 19, 23, 29);
    const v: Fp6 = .{
        .c0 = Fp2.zero(),
        .c1 = Fp2.one(),
        .c2 = Fp2.zero(),
    };
    try testing.expect(Fp6.eql(Fp6.mulByV(a), Fp6.mul(a, v)));
}

test "Fp6.inv: a * a⁻¹ = 1" {
    const a = fp6FromInts(2, 3, 5, 7, 11, 13);
    const inv_a = Fp6.inv(a);
    try testing.expect(Fp6.eql(Fp6.mul(a, inv_a), Fp6.one()));
}

test "Fp6.pow: a^0 = 1, a^1 = a, a^2 = square(a)" {
    const a = fp6FromInts(2, 3, 5, 7, 11, 13);
    try testing.expect(Fp6.eql(Fp6.pow(a, 1, .{0}), Fp6.one()));
    try testing.expect(Fp6.eql(Fp6.pow(a, 1, .{1}), a));
    try testing.expect(Fp6.eql(Fp6.pow(a, 1, .{2}), Fp6.square(a)));
}

test "fp6Frobenius: applied 6 times returns the input" {
    // The Frobenius has order dividing 6 in Fp6 (since Fp6 has degree
    // 6 over Fp). Applying it 6 times should be the identity for
    // EVERY element, even outside the cyclotomic subgroup.
    const a = fp6FromInts(11, 13, 17, 19, 23, 29);
    var result = a;
    inline for (0..6) |_| result = fp6Frobenius(result);
    try testing.expect(Fp6.eql(result, a));
}

test "fp6Frobenius: matches Fp6.pow with exponent p" {
    // φ(a) = a^p must equal Fp6.pow(a, FP_MODULUS) for any a. Slow
    // but the most direct cross-check. Limit to a small element to
    // keep the test budget reasonable.
    const a = fp6FromInts(2, 3, 5, 7, 11, 13);
    const via_frobenius = fp6Frobenius(a);
    const via_pow = Fp6.pow(a, 6, FP_MODULUS);
    try testing.expect(Fp6.eql(via_frobenius, via_pow));
}

test "fp6Frobenius: leaves Fp2 elements alone for c1, c2 = 0" {
    // For a = (a0, 0, 0), the v and v² components of fp6Frobenius
    // are zero (because they're multiplied by gamma1 / gamma1², but
    // a1 = a2 = 0). And the c0 component is fp2Frobenius(a0).
    const a: Fp6 = .{
        .c0 = fp2FromU64Pair(42, 7),
        .c1 = Fp2.zero(),
        .c2 = Fp2.zero(),
    };
    const result = fp6Frobenius(a);
    try testing.expect(Fp2.eql(result.c0, fp2Frobenius(a.c0)));
    try testing.expect(Fp2.eql(result.c1, Fp2.zero()));
    try testing.expect(Fp2.eql(result.c2, Fp2.zero()));
}

test "fp6FrobeniusGamma1: applying Frobenius 6 times to v returns v" {
    // The Frobenius coefficient γ₁ = (1+u)^((p-1)/3) drives the action
    // of φ on v ∈ Fp6. After 6 applications of Frobenius, every Fp6
    // element returns to itself; the test reaching back to v specifically
    // exercises the gamma1 / gamma1² product chain.
    const v: Fp6 = .{
        .c0 = Fp2.zero(),
        .c1 = Fp2.one(),
        .c2 = Fp2.zero(),
    };
    var v6 = v;
    inline for (0..6) |_| v6 = fp6Frobenius(v6);
    try testing.expect(Fp6.eql(v6, v));
}

// ---------------------------------------------------------------------------
// Fp12 tests
// ---------------------------------------------------------------------------

fn fp12FromInts(c0: Fp6, c1: Fp6) Fp12 {
    return .{ .c0 = c0, .c1 = c1 };
}

test "Fp12 identity laws" {
    const a = fp12FromInts(
        fp6FromInts(1, 2, 3, 4, 5, 6),
        fp6FromInts(7, 8, 9, 10, 11, 12),
    );
    try testing.expect(Fp12.eql(Fp12.add(a, Fp12.zero()), a));
    try testing.expect(Fp12.eql(Fp12.mul(a, Fp12.one()), a));
    try testing.expect(Fp12.eql(Fp12.mul(Fp12.one(), a), a));
    try testing.expect(Fp12.eql(Fp12.add(a, Fp12.neg(a)), Fp12.zero()));
}

test "Fp12.mul: w² = v" {
    // w in Fp12 = (0 + 0w + 0w² , 1 + 0w + 0w²) — i.e. c0=0, c1=1
    const w: Fp12 = .{ .c0 = Fp6.zero(), .c1 = Fp6.one() };
    const w_sq = Fp12.mul(w, w);
    // w² should be (v, 0). v = (0, 1, 0) in Fp6.
    const v_in_fp6: Fp6 = .{
        .c0 = Fp2.zero(),
        .c1 = Fp2.one(),
        .c2 = Fp2.zero(),
    };
    const expected: Fp12 = .{ .c0 = v_in_fp6, .c1 = Fp6.zero() };
    try testing.expect(Fp12.eql(w_sq, expected));
}

test "Fp12.mul: distributive over add" {
    const a = fp12FromInts(
        fp6FromInts(2, 3, 5, 7, 11, 13),
        fp6FromInts(17, 19, 23, 29, 31, 37),
    );
    const b = fp12FromInts(
        fp6FromInts(41, 43, 47, 53, 59, 61),
        fp6FromInts(67, 71, 73, 79, 83, 89),
    );
    const c = fp12FromInts(
        fp6FromInts(97, 101, 103, 107, 109, 113),
        fp6FromInts(127, 131, 137, 139, 149, 151),
    );
    const lhs = Fp12.mul(Fp12.add(a, b), c);
    const rhs = Fp12.add(Fp12.mul(a, c), Fp12.mul(b, c));
    try testing.expect(Fp12.eql(lhs, rhs));
}

test "Fp12.square: equivalent to mul(a, a)" {
    const a = fp12FromInts(
        fp6FromInts(11, 13, 17, 19, 23, 29),
        fp6FromInts(31, 37, 41, 43, 47, 53),
    );
    try testing.expect(Fp12.eql(Fp12.square(a), Fp12.mul(a, a)));
}

test "Fp12.inv: a * a⁻¹ = 1" {
    const a = fp12FromInts(
        fp6FromInts(2, 3, 5, 7, 11, 13),
        fp6FromInts(17, 19, 23, 29, 31, 37),
    );
    const inv_a = Fp12.inv(a);
    try testing.expect(Fp12.eql(Fp12.mul(a, inv_a), Fp12.one()));
}

test "Fp12.conjugate: applied twice returns the input" {
    const a = fp12FromInts(
        fp6FromInts(2, 3, 5, 7, 11, 13),
        fp6FromInts(17, 19, 23, 29, 31, 37),
    );
    try testing.expect(Fp12.eql(Fp12.conjugate(Fp12.conjugate(a)), a));
}

test "Fp12.conjugate: conjugate(real-only element) is itself" {
    const real_only: Fp12 = .{
        .c0 = fp6FromInts(2, 3, 5, 7, 11, 13),
        .c1 = Fp6.zero(),
    };
    try testing.expect(Fp12.eql(Fp12.conjugate(real_only), real_only));
}

test "Fp12.conjugate: a · conjugate(a) is in Fp6" {
    // For any a = c0 + c1·w, a · conjugate(a) = c0² - c1²·v which has
    // zero c1 component (i.e., lives in the Fp6 subfield).
    const a = fp12FromInts(
        fp6FromInts(2, 3, 5, 7, 11, 13),
        fp6FromInts(17, 19, 23, 29, 31, 37),
    );
    const product = Fp12.mul(a, Fp12.conjugate(a));
    try testing.expect(Fp6.eql(product.c1, Fp6.zero()));
}

test "Fp12.pow: a^0 = 1, a^1 = a, a^2 = square(a)" {
    const a = fp12FromInts(
        fp6FromInts(2, 3, 5, 7, 11, 13),
        fp6FromInts(17, 19, 23, 29, 31, 37),
    );
    try testing.expect(Fp12.eql(Fp12.pow(a, 1, .{0}), Fp12.one()));
    try testing.expect(Fp12.eql(Fp12.pow(a, 1, .{1}), a));
    try testing.expect(Fp12.eql(Fp12.pow(a, 1, .{2}), Fp12.square(a)));
}

test "Fp12.pow: matches manual repeated mul" {
    const a = fp12FromInts(
        fp6FromInts(11, 13, 17, 19, 23, 29),
        fp6FromInts(31, 37, 41, 43, 47, 53),
    );
    const a_to_5 = Fp12.pow(a, 1, .{5});
    var manual = a;
    inline for (0..4) |_| manual = Fp12.mul(manual, a);
    try testing.expect(Fp12.eql(a_to_5, manual));
}

test "fp12Frobenius: applied 12 times returns the input" {
    // The Fp12 Frobenius has order dividing 12.
    const a = fp12FromInts(
        fp6FromInts(2, 3, 5, 7, 11, 13),
        fp6FromInts(17, 19, 23, 29, 31, 37),
    );
    var result = a;
    inline for (0..12) |_| result = fp12Frobenius(result);
    try testing.expect(Fp12.eql(result, a));
}

test "fp12Frobenius: matches Fp12.pow with exponent p" {
    // φ(a) = a^p must equal Fp12.pow(a, FP_MODULUS) for any a.
    // Slow but the most direct cross-check. The pow path here goes
    // through ~381 Fp12 squarings + ~190 Fp12 mults; even with a small
    // input that's a few hundred ms.
    const a = fp12FromInts(
        fp6FromInts(2, 3, 0, 0, 0, 0),
        fp6FromInts(0, 0, 5, 7, 0, 0),
    );
    const via_frobenius = fp12Frobenius(a);
    const via_pow = Fp12.pow(a, 6, FP_MODULUS);
    try testing.expect(Fp12.eql(via_frobenius, via_pow));
}

test "fp12Frobenius: leaves c0=Fp6, c1=0 alone for the c1 side" {
    // For a = (c0, 0), the c1 side stays zero and the c0 side gets
    // fp6Frobenius applied directly.
    const c0 = fp6FromInts(2, 3, 5, 7, 11, 13);
    const a: Fp12 = .{ .c0 = c0, .c1 = Fp6.zero() };
    const result = fp12Frobenius(a);
    try testing.expect(Fp6.eql(result.c0, fp6Frobenius(c0)));
    try testing.expect(Fp6.eql(result.c1, Fp6.zero()));
}

test "fp12FrobeniusSquared: matches fp12Frobenius applied twice" {
    const a = fp12FromInts(
        fp6FromInts(2, 3, 5, 7, 11, 13),
        fp6FromInts(17, 19, 23, 29, 31, 37),
    );
    const direct = fp12FrobeniusSquared(a);
    const composed = fp12Frobenius(fp12Frobenius(a));
    try testing.expect(Fp12.eql(direct, composed));
}

test "fp12FinalExpEasy: result is in the cyclotomic subgroup" {
    // After the easy part, the result `g` should satisfy
    // `conjugate(g) · g = 1` (this is the defining property of the
    // cyclotomic subgroup of Fp12).
    const a = fp12FromInts(
        fp6FromInts(2, 3, 5, 7, 11, 13),
        fp6FromInts(17, 19, 23, 29, 31, 37),
    );
    const g = fp12FinalExpEasy(a);
    const product = Fp12.mul(Fp12.conjugate(g), g);
    try testing.expect(Fp12.eql(product, Fp12.one()));
}

test "fp12FinalExpEasy: a^0 stays zero, a=1 stays 1" {
    // f = 1: easy result should be 1.
    const one = Fp12.one();
    const result = fp12FinalExpEasy(one);
    try testing.expect(Fp12.eql(result, one));
}

test "fp2Frobenius: applied twice returns the input" {
    const a = fp2FromU64Pair(123, 456);
    try testing.expect(Fp2.eql(fp2Frobenius(fp2Frobenius(a)), a));
}

test "fp2Frobenius: conjugates the imaginary part" {
    const a = fp2FromU64Pair(123, 456);
    const expected: Fp2 = .{ .c0 = a.c0, .c1 = Fp.neg(a.c1) };
    try testing.expect(Fp2.eql(fp2Frobenius(a), expected));
}

test "fp2Frobenius: leaves Fp elements alone" {
    const a: Fp2 = .{ .c0 = Fp.fromRaw(.{ 0x42, 0, 0, 0, 0, 0 }), .c1 = Fp.zero() };
    try testing.expect(Fp2.eql(fp2Frobenius(a), a));
}

test "fp2Pow: a^0 = 1" {
    const a = fp2FromU64Pair(123, 456);
    const result = fp2Pow(a, 1, .{0});
    try testing.expect(Fp2.eql(result, Fp2.one()));
}

test "fp2Pow: a^1 = a" {
    const a = fp2FromU64Pair(123, 456);
    try testing.expect(Fp2.eql(fp2Pow(a, 1, .{1}), a));
}

test "fp2Pow: a^2 = square(a)" {
    const a = fp2FromU64Pair(123, 456);
    try testing.expect(Fp2.eql(fp2Pow(a, 1, .{2}), Fp2.square(a)));
}

test "fp2Pow: 3^7 = 2187 (in Fp2)" {
    const three = fp2FromU64Pair(3, 0);
    const result = fp2Pow(three, 1, .{7});
    const expected = fp2FromU64Pair(2187, 0);
    try testing.expect(Fp2.eql(result, expected));
}

test "fp2Pow matches manual repeated mul for small exponents" {
    const a = fp2FromU64Pair(5, 7);
    const a_to_5 = fp2Pow(a, 1, .{5});
    var manual = a;
    inline for (0..4) |_| manual = Fp2.mul(manual, a);
    try testing.expect(Fp2.eql(a_to_5, manual));
}

// ---------------------------------------------------------------------------
// G1 affine arithmetic tests
// ---------------------------------------------------------------------------

test "G1: generator is on the curve" {
    const g = g1Generator();
    try testing.expect(g.isOnCurve());
}

test "G1: identity is on the curve" {
    try testing.expect(G1Affine.identity().isOnCurve());
}

test "G1: identity is the additive neutral element" {
    const g = g1Generator();
    const id = G1Affine.identity();
    try testing.expect(G1Affine.eql(g.add(id), g));
    try testing.expect(G1Affine.eql(id.add(g), g));
}

test "G1: P + (-P) = identity" {
    const g = g1Generator();
    const neg_g = g.neg();
    const sum = g.add(neg_g);
    try testing.expect(sum.isIdentity());
}

test "G1: 2P via double matches P + P" {
    const g = g1Generator();
    const doubled = g.double();
    const summed = g.add(g);
    try testing.expect(G1Affine.eql(doubled, summed));
    // 2P should still be on the curve.
    try testing.expect(doubled.isOnCurve());
}

test "G1: 3P = 2P + P matches P + 2P" {
    const g = g1Generator();
    const two_g = g.double();
    const three_g_a = two_g.add(g);
    const three_g_b = g.add(two_g);
    try testing.expect(G1Affine.eql(three_g_a, three_g_b));
    try testing.expect(three_g_a.isOnCurve());
}

test "G1: 4P = 2(2P) matches 3P + P" {
    const g = g1Generator();
    const two_g = g.double();
    const four_g_a = two_g.double();
    const four_g_b = two_g.add(g).add(g);
    try testing.expect(G1Affine.eql(four_g_a, four_g_b));
    try testing.expect(four_g_a.isOnCurve());
}

test "G1: addition is commutative" {
    const g = g1Generator();
    const two_g = g.double();
    const three_g = two_g.add(g);
    const lhs = three_g.add(two_g);
    const rhs = two_g.add(three_g);
    try testing.expect(G1Affine.eql(lhs, rhs));
}

test "G1: addition is associative" {
    const g = g1Generator();
    const two_g = g.double();
    const three_g = two_g.add(g);
    // (g + 2g) + 3g == g + (2g + 3g)
    const lhs = g.add(two_g).add(three_g);
    const rhs = g.add(two_g.add(three_g));
    try testing.expect(G1Affine.eql(lhs, rhs));
}

test "G1: scalar mul matches repeated add for small scalars" {
    const g = g1Generator();
    // 5 * G == G + G + G + G + G
    const five_g = g.mul(1, .{5});
    const expected = g.add(g).add(g).add(g).add(g);
    try testing.expect(G1Affine.eql(five_g, expected));
}

test "G1: scalar mul: 0 * G = identity, 1 * G = G" {
    const g = g1Generator();
    try testing.expect(g.mul(1, .{0}).isIdentity());
    try testing.expect(G1Affine.eql(g.mul(1, .{1}), g));
}

test "G1: scalar mul: 2 * G via mul matches G.double()" {
    const g = g1Generator();
    try testing.expect(G1Affine.eql(g.mul(1, .{2}), g.double()));
}

test "G1: scalar mul distributes over scalar add (small scalars)" {
    const g = g1Generator();
    // (3 + 5) * G == 3*G + 5*G
    const lhs = g.mul(1, .{8});
    const rhs = g.mul(1, .{3}).add(g.mul(1, .{5}));
    try testing.expect(G1Affine.eql(lhs, rhs));
}

// ---------------------------------------------------------------------------
// Compressed point decoding
// ---------------------------------------------------------------------------

test "decodeG1Compressed: infinity flag round-trip" {
    var bytes: [48]u8 = .{0} ** 48;
    bytes[0] = 0xc0; // compression + infinity bits set
    const point = try decodeG1Compressed(&bytes);
    try testing.expect(point.isIdentity());
}

test "decodeG1Compressed: rejects wrong length" {
    const short: [47]u8 = .{0} ** 47;
    try testing.expectError(PointDecodeError.InvalidLength, decodeG1Compressed(&short));
}

test "decodeG1Compressed: rejects missing compression flag" {
    var bytes: [48]u8 = .{0} ** 48;
    // Top bit cleared = uncompressed encoding, which we don't support.
    try testing.expectError(PointDecodeError.InvalidEncoding, decodeG1Compressed(&bytes));
}

test "decodeG1Compressed: G1 generator round-trip" {
    // Compressed encoding of the standard G1 generator (from the
    // BLS12-381 IETF spec):
    //   0x97f1d3a7 3197d794 2695638c 4fa9ac0f
    //     c3688c4f 9774b905 a14e3a3f 171bac58
    //     6c55e83f f97a1aef fb3af00a db22c6bb
    // The high bit (0x80) is the compression flag; the next bit
    // would be infinity (cleared); the third bit is the y-sign.
    // For the standard generator the y-sign bit is 0 (lex-smaller y),
    // so the first byte is exactly 0x97 (not 0xb7).
    const compressed_hex = "97f1d3a73197d7942695638c4fa9ac0fc3688c4f9774b905a14e3a3f171bac586c55e83ff97a1aeffb3af00adb22c6bb";
    var compressed: [48]u8 = undefined;
    _ = try std.fmt.hexToBytes(&compressed, compressed_hex);
    const decoded = try decodeG1Compressed(&compressed);
    try testing.expect(decoded.isOnCurve());
    // The recovered x must equal the canonical generator x.
    try testing.expect(Fp.eql(decoded.x, Fp.fromRaw(G1_GENERATOR_X)));
    // And the recovered point must equal the canonical generator (with
    // matching y root).
    try testing.expect(G1Affine.eql(decoded, g1Generator()));
}

test "decodeG1Compressed: bit-flipped y-sign decodes to -G" {
    // Same encoding, but flip the y-sign bit (bit 5 of the first byte).
    const compressed_hex = "97f1d3a73197d7942695638c4fa9ac0fc3688c4f9774b905a14e3a3f171bac586c55e83ff97a1aeffb3af00adb22c6bb";
    var compressed: [48]u8 = undefined;
    _ = try std.fmt.hexToBytes(&compressed, compressed_hex);
    compressed[0] |= 0b0010_0000;
    const decoded = try decodeG1Compressed(&compressed);
    try testing.expect(decoded.isOnCurve());
    try testing.expect(G1Affine.eql(decoded, g1Generator().neg()));
}

test "decodeG1Compressed: rejects x ≥ p" {
    var bytes: [48]u8 = .{0xff} ** 48;
    bytes[0] = 0x80 | 0x1f; // compression flag + low 5 bits of 0xff
    try testing.expectError(PointDecodeError.NotInField, decodeG1Compressed(&bytes));
}

// ---------------------------------------------------------------------------
// G2 affine arithmetic tests. The test set mirrors G1 — the curve
// equation differs but the affine algebra is the same.
// ---------------------------------------------------------------------------

test "G2: generator is on the curve" {
    const g = g2Generator();
    try testing.expect(g.isOnCurve());
}

test "G2: identity is on the curve" {
    try testing.expect(G2Affine.identity().isOnCurve());
}

test "G2: identity is the additive neutral element" {
    const g = g2Generator();
    const id = G2Affine.identity();
    try testing.expect(G2Affine.eql(g.add(id), g));
    try testing.expect(G2Affine.eql(id.add(g), g));
}

test "G2: P + (-P) = identity" {
    const g = g2Generator();
    const sum = g.add(g.neg());
    try testing.expect(sum.isIdentity());
}

test "G2: 2P via double matches P + P" {
    const g = g2Generator();
    const doubled = g.double();
    const summed = g.add(g);
    try testing.expect(G2Affine.eql(doubled, summed));
    try testing.expect(doubled.isOnCurve());
}

test "G2: 3P consistency and on-curve" {
    const g = g2Generator();
    const two_g = g.double();
    const three_g_a = two_g.add(g);
    const three_g_b = g.add(two_g);
    try testing.expect(G2Affine.eql(three_g_a, three_g_b));
    try testing.expect(three_g_a.isOnCurve());
}

test "G2: addition is commutative" {
    const g = g2Generator();
    const two_g = g.double();
    const three_g = two_g.add(g);
    try testing.expect(G2Affine.eql(three_g.add(two_g), two_g.add(three_g)));
}

test "G2: scalar mul matches repeated add for small scalars" {
    const g = g2Generator();
    const four_g = g.mul(1, .{4});
    const expected = g.double().double();
    try testing.expect(G2Affine.eql(four_g, expected));
}

test "G2: 0 * G = identity, 1 * G = G" {
    const g = g2Generator();
    try testing.expect(g.mul(1, .{0}).isIdentity());
    try testing.expect(G2Affine.eql(g.mul(1, .{1}), g));
}

// ---------------------------------------------------------------------------
// Compressed G2 decoding
// ---------------------------------------------------------------------------

test "decodeG2Compressed: infinity flag round-trip" {
    var bytes: [96]u8 = .{0} ** 96;
    bytes[0] = 0xc0;
    const point = try decodeG2Compressed(&bytes);
    try testing.expect(point.isIdentity());
}

test "decodeG2Compressed: rejects wrong length" {
    const short: [95]u8 = .{0} ** 95;
    try testing.expectError(PointDecodeError.InvalidLength, decodeG2Compressed(&short));
}

test "decodeG2Compressed: rejects missing compression flag" {
    var bytes: [96]u8 = .{0} ** 96;
    try testing.expectError(PointDecodeError.InvalidEncoding, decodeG2Compressed(&bytes));
}

test "decodeG2Compressed: G2 generator round-trip" {
    // Compressed encoding of the standard G2 generator (from the
    // BLS12-381 IETF spec): bytes[0..48] = x.c1, bytes[48..96] = x.c0,
    // with the high three bits of byte 0 set to {1, 0, sign}.
    //
    // For the standard generator, the lex-smaller y root is taken,
    // so the y-sign bit is 0 and the first byte is 0x80 | (top byte
    // of x.c1 with high 3 bits cleared).
    //
    // x.c1 high byte = 0x13, with the compression flag set the first
    // byte becomes 0x80 | 0x13 = 0x93.
    const compressed_hex = "93e02b6052719f607dacd3a088274f65596bd0d09920b61ab5da61bbdc7f5049334cf11213945d57e5ac7d055d042b7e024aa2b2f08f0a91260805272dc51051c6e47ad4fa403b02b4510b647ae3d1770bac0326a805bbefd48056c8c121bdb8";
    var compressed: [96]u8 = undefined;
    _ = try std.fmt.hexToBytes(&compressed, compressed_hex);
    const decoded = try decodeG2Compressed(&compressed);
    try testing.expect(decoded.isOnCurve());
    // The recovered x must equal the canonical generator x.
    try testing.expect(Fp2.eql(decoded.x, g2Generator().x));
    // And the full point must equal one of {±G2}.
    const g2 = g2Generator();
    try testing.expect(G2Affine.eql(decoded, g2) or G2Affine.eql(decoded, g2.neg()));
}

// ---------------------------------------------------------------------------
// Compressed point encoder tests
// ---------------------------------------------------------------------------

test "encodeG1Compressed: identity round-trip" {
    const id = G1Affine.identity();
    const bytes = encodeG1Compressed(id);
    try testing.expectEqual(@as(u8, 0xc0), bytes[0]);
    inline for (1..48) |i| try testing.expectEqual(@as(u8, 0), bytes[i]);

    const decoded = try decodeG1Compressed(&bytes);
    try testing.expect(decoded.isIdentity());
}

test "encodeG1Compressed: generator round-trip" {
    const g = g1Generator();
    const bytes = encodeG1Compressed(g);
    // Compression flag must be set.
    try testing.expect((bytes[0] & 0x80) != 0);
    // Infinity flag must be cleared.
    try testing.expect((bytes[0] & 0x40) == 0);

    const decoded = try decodeG1Compressed(&bytes);
    try testing.expect(G1Affine.eql(decoded, g));
}

test "encodeG1Compressed: matches the canonical generator hex" {
    // The canonical compressed encoding of G1 from the IETF spec.
    const expected_hex = "97f1d3a73197d7942695638c4fa9ac0fc3688c4f9774b905a14e3a3f171bac586c55e83ff97a1aeffb3af00adb22c6bb";
    var expected: [48]u8 = undefined;
    _ = try std.fmt.hexToBytes(&expected, expected_hex);

    const bytes = encodeG1Compressed(g1Generator());
    try testing.expectEqualSlices(u8, &expected, &bytes);
}

test "encodeG1Compressed: -G round-trip with sign flag set" {
    const g = g1Generator();
    const neg_g = g.neg();
    const bytes = encodeG1Compressed(neg_g);
    // The y-sign for -G should differ from G's. Decode and verify.
    const decoded = try decodeG1Compressed(&bytes);
    try testing.expect(G1Affine.eql(decoded, neg_g));
}

test "encodeG1Compressed: random scalar multiple round-trips" {
    const g = g1Generator();
    const five_g = g.mul(1, .{5});
    const bytes = encodeG1Compressed(five_g);
    const decoded = try decodeG1Compressed(&bytes);
    try testing.expect(G1Affine.eql(decoded, five_g));
}

test "encodeG2Compressed: identity round-trip" {
    const id = G2Affine.identity();
    const bytes = encodeG2Compressed(id);
    try testing.expectEqual(@as(u8, 0xc0), bytes[0]);
    inline for (1..96) |i| try testing.expectEqual(@as(u8, 0), bytes[i]);

    const decoded = try decodeG2Compressed(&bytes);
    try testing.expect(decoded.isIdentity());
}

test "encodeG2Compressed: generator round-trip" {
    const g = g2Generator();
    const bytes = encodeG2Compressed(g);
    try testing.expect((bytes[0] & 0x80) != 0);
    try testing.expect((bytes[0] & 0x40) == 0);

    const decoded = try decodeG2Compressed(&bytes);
    try testing.expect(G2Affine.eql(decoded, g));
}

test "encodeG2Compressed: matches the canonical generator hex" {
    // The canonical compressed encoding of G2 from the IETF spec.
    const expected_hex = "93e02b6052719f607dacd3a088274f65596bd0d09920b61ab5da61bbdc7f5049334cf11213945d57e5ac7d055d042b7e024aa2b2f08f0a91260805272dc51051c6e47ad4fa403b02b4510b647ae3d1770bac0326a805bbefd48056c8c121bdb8";
    var expected: [96]u8 = undefined;
    _ = try std.fmt.hexToBytes(&expected, expected_hex);

    const bytes = encodeG2Compressed(g2Generator());
    try testing.expectEqualSlices(u8, &expected, &bytes);
}

test "encodeG2Compressed: -G round-trip with sign flag set" {
    const g = g2Generator();
    const neg_g = g.neg();
    const bytes = encodeG2Compressed(neg_g);
    const decoded = try decodeG2Compressed(&bytes);
    try testing.expect(G2Affine.eql(decoded, neg_g));
}

test "encodeG2Compressed: scalar multiple round-trips" {
    const g = g2Generator();
    const seven_g = g.mul(1, .{7});
    const bytes = encodeG2Compressed(seven_g);
    const decoded = try decodeG2Compressed(&bytes);
    try testing.expect(G2Affine.eql(decoded, seven_g));
}
