//! BN254 Extension Field Arithmetic
//!
//! This module implements the extension field tower for BN254 pairing operations:
//! - Fp2 = Fp[u] / (u^2 + 1)
//! - Fp6 = Fp2[v] / (v^3 - xi) where xi = 9 + u
//! - Fp12 = Fp6[w] / (w^2 - v)
//!
//! Also includes Frobenius coefficients and sparse multiplication helpers
//! used by the Miller loop in pairing.zig.

const std = @import("std");

const field_mod = @import("mod.zig");
const Fp = field_mod.BN254BaseField;

// ============================================================================
// Frobenius Coefficients for BN254
// ============================================================================
//
// These constants are needed for the Frobenius endomorphism on G2 and Fp12.
// For BN254 with xi = 9 + u (the non-residue), we need:
//   gamma_{1,j} = xi^{j(p-1)/6} for j = 1..5
//
// From ziskos BN254 implementation (constants.rs)
// Stored as [u64; 8] for Fp2 (two Fp elements) and [u64; 4] for Fp

/// Helper to create Fp2 from [u64; 8] limbs (raw, non-Montgomery form)
/// Converts each Fp element to Montgomery form
pub fn fp2FromLimbs(limbs: [8]u64) Fp2 {
    // Each Fp element is 4 limbs, stored in little-endian order
    // These are raw values, need to convert to Montgomery form
    const c0_raw = Fp{ .limbs = .{ limbs[0], limbs[1], limbs[2], limbs[3] } };
    const c1_raw = Fp{ .limbs = .{ limbs[4], limbs[5], limbs[6], limbs[7] } };
    // Convert to Montgomery form by multiplying by R^2 mod p (which gives a*R mod p)
    const c0 = c0_raw.toMontgomery();
    const c1 = c1_raw.toMontgomery();
    return Fp2.init(c0, c1);
}

/// Helper to create Fp from [u64; 4] limbs (raw, non-Montgomery form)
/// Converts to Montgomery form
pub fn fpFromLimbs(limbs: [4]u64) Fp {
    const raw = Fp{ .limbs = limbs };
    return raw.toMontgomery();
}

// ============================================================================
// Frobenius^1 Coefficients (require conjugation)
// ============================================================================

/// gamma_11 = xi^{(p-1)/6} - Frobenius^1 coefficient for c1 of Fp6 in c1 of Fp12
pub const FROBENIUS_GAMMA11: [8]u64 = .{
    0xD60B35DADCC9E470, 0x5C521E08292F2176, 0xE8B99FDD76E68B60, 0x1284B71C2865A7DF,
    0xCA5CF05F80F362AC, 0x747992778EEEC7E5, 0xA6327CFE12150B8E, 0x246996F3B4FAE7E6,
};

/// gamma_12 = xi^{2(p-1)/6} - Frobenius^1 coefficient for c1 of Fp6 in c0 of Fp12 (also G2 x-coord)
pub const FROBENIUS_GAMMA12: [8]u64 = .{
    0x99E39557176F553D, 0xB78CC310C2C3330C, 0x4C0BEC3CF559B143, 0x2FB347984F7911F7,
    0x1665D51C640FCBA2, 0x32AE2A1D0B7C9DCE, 0x4BA4CC8BD75A0794, 0x16C9E55061EBAE20,
};

/// gamma_13 = xi^{3(p-1)/6} - Frobenius^1 coefficient for c2 of Fp6 in c1 of Fp12 (also G2 y-coord)
pub const FROBENIUS_GAMMA13: [8]u64 = .{
    0xDC54014671A0135A, 0xDBAAE0EDA9C95998, 0xDC5EC698B6E2F9B9, 0x063CF305489AF5DC,
    0x82D37F632623B0E3, 0x21807DC98FA25BD2, 0x0704B5A7EC796F2B, 0x07C03CBCAC41049A,
};

/// gamma_14 = xi^{4(p-1)/6} - Frobenius^1 coefficient for c2 of Fp6 in c0 of Fp12
pub const FROBENIUS_GAMMA14: [8]u64 = .{
    0x848A1F55921EA762, 0xD33365F7BE94EC72, 0x80F3C0B75A181E84, 0x05B54F5E64EEA801,
    0xC13B4711CD2B8126, 0x3685D2EA1BDEC763, 0x9F3A80B03B0B1C92, 0x2C145EDBE7FD8AEE,
};

/// gamma_15 = xi^{5(p-1)/6} - Frobenius^1 coefficient for c2 of Fp6 in c1 of Fp12
pub const FROBENIUS_GAMMA15: [8]u64 = .{
    0x2EA2C810EAB7692F, 0x425C459B55AA1BD3, 0xE93A3661A4353FF4, 0x0183C1E74F798649,
    0x24C6B8EE6E0C2C4B, 0xB080CB99678E2AC0, 0xA27FB246C7729F7D, 0x12ACF2CA76FD0675,
};

// ============================================================================
// Frobenius^2 Coefficients (no conjugation needed - even power)
// ============================================================================

/// gamma_21 = xi^{(p^2-1)/6} - Fp element
pub const FROBENIUS_GAMMA21: [4]u64 = .{ 0xE4BD44E5607CFD49, 0xC28F069FBB966E3D, 0x5E6DD9E7E0ACCCB0, 0x30644E72E131A029 };

/// gamma_22 = xi^{2(p^2-1)/6} - Fp element
pub const FROBENIUS_GAMMA22: [4]u64 = .{ 0xE4BD44E5607CFD48, 0xC28F069FBB966E3D, 0x5E6DD9E7E0ACCCB0, 0x30644E72E131A029 };

/// gamma_23 = xi^{3(p^2-1)/6} - Fp element
pub const FROBENIUS_GAMMA23: [4]u64 = .{ 0x3C208C16D87CFD46, 0x97816A916871CA8D, 0xB85045B68181585D, 0x30644E72E131A029 };

/// gamma_24 = xi^{4(p^2-1)/6} - Fp element
pub const FROBENIUS_GAMMA24: [4]u64 = .{ 0x5763473177FFFFFE, 0xD4F263F1ACDB5C4F, 0x59E26BCEA0D48BAC, 0x0000000000000000 };

/// gamma_25 = xi^{5(p^2-1)/6} - Fp element
pub const FROBENIUS_GAMMA25: [4]u64 = .{ 0x5763473177FFFFFF, 0xD4F263F1ACDB5C4F, 0x59E26BCEA0D48BAC, 0x0000000000000000 };

// ============================================================================
// Frobenius^3 Coefficients (require conjugation - odd power)
// ============================================================================

/// gamma_31 = xi^{(p^3-1)/6}
pub const FROBENIUS_GAMMA31: [8]u64 = .{
    0xE86F7D391ED4A67F, 0x894CB38DBE55D24A, 0xEFE9608CD0ACAA90, 0x19DC81CFCC82E4BB,
    0x7694AA2BF4C0C101, 0x7F03A5E397D439EC, 0x06CBEEE33576139D, 0x00ABF8B60BE77D73,
};

/// gamma_32 = xi^{2(p^3-1)/6}
pub const FROBENIUS_GAMMA32: [8]u64 = .{
    0x7B746EE87BDCFB6D, 0x805FFD3D5D6942D3, 0xBAFF1C77959F25AC, 0x0856E078B755EF0A,
    0x380CAB2BAAA586DE, 0x0FDF31BF98FF2631, 0xA9F30E6DEC26094F, 0x04F1DE41B3D1766F,
};

/// gamma_33 = xi^{3(p^3-1)/6}
pub const FROBENIUS_GAMMA33: [8]u64 = .{
    0x5FCC8AD066DCE9ED, 0xBBD689A3BEA870F4, 0xDBF17F1DCA9E5EA3, 0x2A275B6D9896AA4C,
    0xB94D0CB3B2594C64, 0x7600ECC7D8CF6EBA, 0xB14B900E9507E932, 0x28A411B634F09B8F,
};

/// gamma_34 = xi^{4(p^3-1)/6}
pub const FROBENIUS_GAMMA34: [8]u64 = .{
    0x0E1A92BC3CCBF066, 0xE633094575B06BCB, 0x19BEE0F7B5B2444E, 0x0BC58C6611C08DAB,
    0x5FE3ED9D730C239F, 0xA44A9E08737F96E5, 0xFEB0F6EF0CD21D04, 0x23D5E999E1910A12,
};

/// gamma_35 = xi^{5(p^3-1)/6}
pub const FROBENIUS_GAMMA35: [8]u64 = .{
    0xEBDE847076261B43, 0x2ED68098967C84A5, 0x711699FA3B4D3F69, 0x13C49044952C0905,
    0x1F25041384282499, 0x3E2DDAEA20028021, 0x9FB1B2282A48633D, 0x16DB366A59B1DD0B,
};

/// GAMMA_12 = xi^{(p-1)/3} = xi^{2(p-1)/6}
/// Used for G2 x-coordinate in Frobenius
pub fn gamma12() Fp2 {
    return fp2FromLimbs(FROBENIUS_GAMMA12);
}

/// GAMMA_13 = xi^{(p-1)/2} = xi^{3(p-1)/6}
/// Used for G2 y-coordinate in Frobenius
pub fn gamma13() Fp2 {
    return fp2FromLimbs(FROBENIUS_GAMMA13);
}

// ============================================================================
// Fp2 scalar multiplication helper
// ============================================================================

/// Multiply Fp2 element by Fp scalar (embeds scalar as (s, 0))
pub fn fp2ScalarMul(a: Fp2, s: Fp) Fp2 {
    return Fp2.init(a.c0.mul(s), a.c1.mul(s));
}

// ============================================================================
// Extension Field Fp2 = Fp[u] / (u^2 + 1)
// ============================================================================

/// Fp2 element: a + b*u where u^2 = -1.
/// Now routed through the curve-generic Fp2 factory.
pub const Fp2 = @import("../curves/extensions.zig").Fp2(Fp);

// ============================================================================
// Extension Field Fp6 = Fp2[v] / (v^3 - xi) where xi = 9 + u
// Now routed through the curve-generic Fp6 factory.
// ============================================================================

const generic_ext = @import("../curves/extensions.zig");

/// BN254 cubic non-residue: multiply Fp2 by ξ = 9 + u (shift-add).
fn bn254MulByXi(x: Fp2) Fp2 {
    const a = x.c0;
    const a2 = a.add(a);
    const a4 = a2.add(a2);
    const a8 = a4.add(a4);
    const a9 = a8.add(a);
    const b = x.c1;
    const b2 = b.add(b);
    const b4 = b2.add(b2);
    const b8 = b4.add(b4);
    const b9 = b8.add(b);
    return Fp2.init(a9.sub(b), a.add(b9));
}

pub const Fp6 = generic_ext.Fp6(Fp2, bn254MulByXi);

/// Fp6 multiplication by v (shift operation).
/// Delegates to the generic factory's `mulByV`.
pub fn fp6MulByV(f: Fp6) Fp6 {
    return Fp6.mulByV(f);
}

// ============================================================================
// Extension Field Fp12 = Fp6[w] / (w^2 - v)
// ============================================================================

/// Fp12 element: c0 + c1*w where w^2 = v
pub const Fp12 = struct {
    c0: Fp6,
    c1: Fp6,

    pub fn zero() Fp12 {
        return .{ .c0 = Fp6.zero(), .c1 = Fp6.zero() };
    }

    pub fn one() Fp12 {
        return .{ .c0 = Fp6.one(), .c1 = Fp6.zero() };
    }

    pub fn add(self: Fp12, other: Fp12) Fp12 {
        return .{
            .c0 = self.c0.add(other.c0),
            .c1 = self.c1.add(other.c1),
        };
    }

    pub fn sub(self: Fp12, other: Fp12) Fp12 {
        return .{
            .c0 = self.c0.sub(other.c0),
            .c1 = self.c1.sub(other.c1),
        };
    }

    pub fn neg(self: Fp12) Fp12 {
        return .{
            .c0 = self.c0.neg(),
            .c1 = self.c1.neg(),
        };
    }

    /// Multiplication in Fp12 using Karatsuba (3 Fp6.mul instead of 4)
    pub fn mul(self: Fp12, other: Fp12) Fp12 {
        // (a + bw)(c + dw) = (ac + bdv) + ((a+b)(c+d) - ac - bd)w
        // where w^2 = v
        const ac = self.c0.mul(other.c0);
        const bd = self.c1.mul(other.c1);
        const ad_plus_bc = self.c0.add(self.c1).mul(other.c0.add(other.c1)).sub(ac).sub(bd);

        // Multiply bd by v (shift in the Fp6 tower)
        const bd_times_v = fp6MulByV(bd);

        return .{
            .c0 = ac.add(bd_times_v),
            .c1 = ad_plus_bc,
        };
    }

    /// Complex squaring: 2 Fp6.mul instead of 4
    /// For f = a + bw, f^2 = (a^2 + b^2*v) + 2ab*w
    pub fn square(self: Fp12) Fp12 {
        const ab = self.c0.mul(self.c1);
        const c1_new = ab.add(ab); // 2ab

        // (a+b)(a+bv) - ab - v*ab = a^2 + b^2v
        const a_plus_b = self.c0.add(self.c1);
        const bv = fp6MulByV(self.c1);
        const a_plus_bv = self.c0.add(bv);
        const abv = fp6MulByV(ab);
        const c0_new = a_plus_b.mul(a_plus_bv).sub(ab).sub(abv);

        return .{ .c0 = c0_new, .c1 = c1_new };
    }

    /// Cyclotomic squaring for unitary Fp12 elements (where x * conjugate(x) = 1).
    /// Uses Granger-Scott algorithm: 6 Fp2 squarings instead of full Fp12 square.
    /// ~40% fewer Fp2 multiplications than generic square.
    pub fn cyclotomicSquare(self: Fp12) Fp12 {
        // Extract the 6 Fp2 components using arkworks naming convention:
        // self.c0 = r0 + r4*v + r3*v^2
        // self.c1 = r2 + r1*v + r5*v^2
        const r0 = self.c0.c0;
        const r4 = self.c0.c1;
        const r3 = self.c0.c2;
        const r2 = self.c1.c0;
        const r1 = self.c1.c1;
        const r5 = self.c1.c2;

        // fp4_square helper: (a, b) -> (t_even, t_odd) where
        // (a + b*y)^2 = t_even + t_odd*y  with y^2 = xi (nonresidue)
        // Uses Karatsuba: (a+b)(xi*b+a) - a*b - xi(a*b)
        //                 and 2*a*b

        // Pair 1: (r0, r1)
        const tmp01 = r0.mul(r1);
        const t0 = r0.add(r1).mul(Fp6.mulByXi(r1).add(r0)).sub(tmp01).sub(Fp6.mulByXi(tmp01));
        const t1 = tmp01.add(tmp01);

        // Pair 2: (r2, r3)
        const tmp23 = r2.mul(r3);
        const t2 = r2.add(r3).mul(Fp6.mulByXi(r3).add(r2)).sub(tmp23).sub(Fp6.mulByXi(tmp23));
        const t3 = tmp23.add(tmp23);

        // Pair 3: (r4, r5)
        const tmp45 = r4.mul(r5);
        const t4 = r4.add(r5).mul(Fp6.mulByXi(r5).add(r4)).sub(tmp45).sub(Fp6.mulByXi(tmp45));
        const t5 = tmp45.add(tmp45);

        // Update using cyclotomic relations:
        // z0 = 3*t0 - 2*r0
        const z0 = t0.sub(r0).add(t0.sub(r0)).add(t0);
        // z1 = 3*t1 + 2*r1
        const z1 = t1.add(r1).add(t1.add(r1)).add(t1);
        // z2 = 3*xi(t5) + 2*r2
        const xi_t5 = Fp6.mulByXi(t5);
        const z2 = xi_t5.add(r2).add(xi_t5.add(r2)).add(xi_t5);
        // z3 = 3*t4 - 2*r3
        const z3 = t4.sub(r3).add(t4.sub(r3)).add(t4);
        // z4 = 3*t2 - 2*r4
        const z4 = t2.sub(r4).add(t2.sub(r4)).add(t2);
        // z5 = 3*t3 + 2*r5
        const z5 = t3.add(r5).add(t3.add(r5)).add(t3);

        return .{
            .c0 = .{ .c0 = z0, .c1 = z4, .c2 = z3 },
            .c1 = .{ .c0 = z2, .c1 = z1, .c2 = z5 },
        };
    }

    /// Conjugate for unitary elements: a + bw -> a - bw
    pub fn conjugate(self: Fp12) Fp12 {
        return .{
            .c0 = self.c0,
            .c1 = self.c1.neg(),
        };
    }

    /// Frobenius^1 endomorphism (raising to p-th power)
    /// For Fp12 = Fp6[w]/(w^2 - v) where Fp6 = Fp2[v]/(v^3 - xi)
    /// Structure: ((a11 + a12*v + a13*v^2) + (a21 + a22*v + a23*v^2)*w)^p
    ///
    /// From ziskos:
    /// c0 = conj(a11)     + conj(a12)*gamma12*v + conj(a13)*gamma14*v^2
    /// c1 = conj(a21)*gamma11 + conj(a22)*gamma13*v + conj(a23)*gamma15*v^2
    pub fn frobenius(self: Fp12) Fp12 {
        // Extract all 6 Fp2 components
        const a11 = self.c0.c0;
        const a12 = self.c0.c1;
        const a13 = self.c0.c2;
        const a21 = self.c1.c0;
        const a22 = self.c1.c1;
        const a23 = self.c1.c2;

        // Get Frobenius^1 coefficients
        const g11 = fp2FromLimbs(FROBENIUS_GAMMA11);
        const g12 = fp2FromLimbs(FROBENIUS_GAMMA12);
        const g13 = fp2FromLimbs(FROBENIUS_GAMMA13);
        const g14 = fp2FromLimbs(FROBENIUS_GAMMA14);
        const g15 = fp2FromLimbs(FROBENIUS_GAMMA15);

        // Apply conjugation (Frobenius in Fp2) and multiply by coefficients
        const c0 = Fp6{
            .c0 = a11.conjugate(),
            .c1 = a12.conjugate().mul(g12),
            .c2 = a13.conjugate().mul(g14),
        };

        const c1 = Fp6{
            .c0 = a21.conjugate().mul(g11),
            .c1 = a22.conjugate().mul(g13),
            .c2 = a23.conjugate().mul(g15),
        };

        return Fp12{ .c0 = c0, .c1 = c1 };
    }

    /// Frobenius^2 endomorphism (raising to p^2 power)
    /// No conjugation needed (even power of Frobenius)
    /// Uses scalar (Fp) multiplication since gamma_2i are in Fp
    ///
    /// c0 = a11     + a12*gamma22*v + a13*gamma24*v^2
    /// c1 = a21*gamma21 + a22*gamma23*v + a23*gamma25*v^2
    pub fn frobenius2(self: Fp12) Fp12 {
        // Extract all 6 Fp2 components
        const a11 = self.c0.c0;
        const a12 = self.c0.c1;
        const a13 = self.c0.c2;
        const a21 = self.c1.c0;
        const a22 = self.c1.c1;
        const a23 = self.c1.c2;

        // Get Frobenius^2 coefficients (Fp elements)
        const g21 = fpFromLimbs(FROBENIUS_GAMMA21);
        const g22 = fpFromLimbs(FROBENIUS_GAMMA22);
        const g23 = fpFromLimbs(FROBENIUS_GAMMA23);
        const g24 = fpFromLimbs(FROBENIUS_GAMMA24);
        const g25 = fpFromLimbs(FROBENIUS_GAMMA25);

        // c0 = a11 + a12*gamma22*v + a13*gamma24*v^2
        const c0 = Fp6{
            .c0 = a11,
            .c1 = fp2ScalarMul(a12, g22),
            .c2 = fp2ScalarMul(a13, g24),
        };

        // c1 = a21*gamma21 + a22*gamma23*v + a23*gamma25*v^2
        const c1 = Fp6{
            .c0 = fp2ScalarMul(a21, g21),
            .c1 = fp2ScalarMul(a22, g23),
            .c2 = fp2ScalarMul(a23, g25),
        };

        return Fp12{ .c0 = c0, .c1 = c1 };
    }

    /// Frobenius^3 endomorphism (raising to p^3 power)
    /// Requires conjugation (odd power of Frobenius)
    ///
    /// c0 = conj(a11)     + conj(a12)*gamma32*v + conj(a13)*gamma34*v^2
    /// c1 = conj(a21)*gamma31 + conj(a22)*gamma33*v + conj(a23)*gamma35*v^2
    pub fn frobenius3(self: Fp12) Fp12 {
        // Extract all 6 Fp2 components
        const a11 = self.c0.c0;
        const a12 = self.c0.c1;
        const a13 = self.c0.c2;
        const a21 = self.c1.c0;
        const a22 = self.c1.c1;
        const a23 = self.c1.c2;

        // Get Frobenius^3 coefficients
        const g31 = fp2FromLimbs(FROBENIUS_GAMMA31);
        const g32 = fp2FromLimbs(FROBENIUS_GAMMA32);
        const g33 = fp2FromLimbs(FROBENIUS_GAMMA33);
        const g34 = fp2FromLimbs(FROBENIUS_GAMMA34);
        const g35 = fp2FromLimbs(FROBENIUS_GAMMA35);

        // Apply conjugation and multiply by coefficients
        const c0 = Fp6{
            .c0 = a11.conjugate(),
            .c1 = a12.conjugate().mul(g32),
            .c2 = a13.conjugate().mul(g34),
        };

        const c1 = Fp6{
            .c0 = a21.conjugate().mul(g31),
            .c1 = a22.conjugate().mul(g33),
            .c2 = a23.conjugate().mul(g35),
        };

        return Fp12{ .c0 = c0, .c1 = c1 };
    }

    pub fn inverse(self: Fp12) ?Fp12 {
        // For quadratic extension: 1/(a + bw) = (a - bw)/(a^2 - b^2v)
        // Use the conjugate method
        const a_squared = self.c0.mul(self.c0);
        const b_squared = self.c1.mul(self.c1);

        // b^2v (shift in Fp6)
        const b_squared_times_v = Fp6{
            .c0 = Fp6.mulByXi(b_squared.c2),
            .c1 = b_squared.c0,
            .c2 = b_squared.c1,
        };

        const norm = a_squared.sub(b_squared_times_v);
        const norm_inv = norm.inverse() orelse return null;

        return .{
            .c0 = self.c0.mul(norm_inv),
            .c1 = self.c1.neg().mul(norm_inv),
        };
    }

    pub fn eql(self: Fp12, other: Fp12) bool {
        return self.c0.eql(other.c0) and self.c1.eql(other.c1);
    }

    pub fn isOne(self: Fp12) bool {
        return self.eql(Fp12.one());
    }

    /// Serialize Fp12 to arkworks-compatible format (384 bytes, uncompressed)
    ///
    /// Arkworks serializes Fp12 = Fp6[w]/(w^2 - v) as:
    /// - c0 (Fp6) first, then c1 (Fp6)
    /// - Each Fp6 = Fp2[v]/(v^3 - xi) is serialized as c0, c1, c2
    /// - Each Fp2 = Fp[u]/(u^2 + 1) is serialized as c0, c1
    /// - Each Fp element is 32 bytes little-endian
    ///
    /// Order: c0.c0.c0, c0.c0.c1, c0.c1.c0, c0.c1.c1, c0.c2.c0, c0.c2.c1,
    ///        c1.c0.c0, c1.c0.c1, c1.c1.c0, c1.c1.c1, c1.c2.c0, c1.c2.c1
    /// = 12 Fp elements x 32 bytes = 384 bytes
    pub fn toBytes(self: Fp12) [384]u8 {
        var result: [384]u8 = undefined;
        var offset: usize = 0;

        // Helper to serialize Fp element
        const serializeFp = struct {
            fn f(buf: []u8, fp_val: Fp) void {
                // Convert from Montgomery form to standard form
                const standard = fp_val.fromMontgomery();
                // Write as 32 bytes little-endian
                for (0..4) |i| {
                    std.mem.writeInt(u64, buf[i * 8 ..][0..8], standard.limbs[i], .little);
                }
            }
        }.f;

        // c0 (Fp6)
        serializeFp(result[offset..][0..32], self.c0.c0.c0);
        offset += 32;
        serializeFp(result[offset..][0..32], self.c0.c0.c1);
        offset += 32;
        serializeFp(result[offset..][0..32], self.c0.c1.c0);
        offset += 32;
        serializeFp(result[offset..][0..32], self.c0.c1.c1);
        offset += 32;
        serializeFp(result[offset..][0..32], self.c0.c2.c0);
        offset += 32;
        serializeFp(result[offset..][0..32], self.c0.c2.c1);
        offset += 32;

        // c1 (Fp6)
        serializeFp(result[offset..][0..32], self.c1.c0.c0);
        offset += 32;
        serializeFp(result[offset..][0..32], self.c1.c0.c1);
        offset += 32;
        serializeFp(result[offset..][0..32], self.c1.c1.c0);
        offset += 32;
        serializeFp(result[offset..][0..32], self.c1.c1.c1);
        offset += 32;
        serializeFp(result[offset..][0..32], self.c1.c2.c0);
        offset += 32;
        serializeFp(result[offset..][0..32], self.c1.c2.c1);

        return result;
    }

    /// Deserialize from arkworks-compatible format (384 bytes)
    pub fn fromBytes(bytes: *const [384]u8) Fp12 {
        var offset: usize = 0;

        // Helper to deserialize Fp element
        const deserializeFp = struct {
            fn f(buf: []const u8) Fp {
                var limbs: [4]u64 = undefined;
                for (0..4) |i| {
                    limbs[i] = std.mem.readInt(u64, buf[i * 8 ..][0..8], .little);
                }
                // Convert to Montgomery form
                const raw = Fp{ .limbs = limbs };
                return raw.toMontgomery();
            }
        }.f;

        // c0 (Fp6)
        const c0_c0_c0 = deserializeFp(bytes[offset..][0..32]);
        offset += 32;
        const c0_c0_c1 = deserializeFp(bytes[offset..][0..32]);
        offset += 32;
        const c0_c1_c0 = deserializeFp(bytes[offset..][0..32]);
        offset += 32;
        const c0_c1_c1 = deserializeFp(bytes[offset..][0..32]);
        offset += 32;
        const c0_c2_c0 = deserializeFp(bytes[offset..][0..32]);
        offset += 32;
        const c0_c2_c1 = deserializeFp(bytes[offset..][0..32]);
        offset += 32;

        // c1 (Fp6)
        const c1_c0_c0 = deserializeFp(bytes[offset..][0..32]);
        offset += 32;
        const c1_c0_c1 = deserializeFp(bytes[offset..][0..32]);
        offset += 32;
        const c1_c1_c0 = deserializeFp(bytes[offset..][0..32]);
        offset += 32;
        const c1_c1_c1 = deserializeFp(bytes[offset..][0..32]);
        offset += 32;
        const c1_c2_c0 = deserializeFp(bytes[offset..][0..32]);
        offset += 32;
        const c1_c2_c1 = deserializeFp(bytes[offset..][0..32]);

        return Fp12{
            .c0 = Fp6{
                .c0 = Fp2.init(c0_c0_c0, c0_c0_c1),
                .c1 = Fp2.init(c0_c1_c0, c0_c1_c1),
                .c2 = Fp2.init(c0_c2_c0, c0_c2_c1),
            },
            .c1 = Fp6{
                .c0 = Fp2.init(c1_c0_c0, c1_c0_c1),
                .c1 = Fp2.init(c1_c1_c0, c1_c1_c1),
                .c2 = Fp2.init(c1_c2_c0, c1_c2_c1),
            },
        };
    }
};

/// GT element (target group of the pairing)
/// This is an alias for Fp12 used in the pairing target group
pub const GT = Fp12;

// ============================================================================
// Fp6 sparse multiplication for pairing
// ============================================================================

/// Multiply Fp6 by sparse element (c0, c1, 0) - matches arkworks mul_by_01
pub fn fp6MulBy01(f: Fp6, c0: Fp2, c1: Fp2) Fp6 {
    // Following arkworks fp6_3over2.rs mul_by_01 exactly
    const a_a = f.c0.mul(c0);
    const b_b = f.c1.mul(c1);

    // t1 = c1 * (f.c1 + f.c2) - b_b, then *= xi, then += a_a
    var tmp = f.c1.add(f.c2);
    var t1 = c1.mul(tmp);
    t1 = t1.sub(b_b);
    t1 = Fp6.mulByXi(t1);
    t1 = t1.add(a_a);

    // t3 = c0 * (f.c0 + f.c2) - a_a + b_b
    tmp = f.c0.add(f.c2);
    var t3 = c0.mul(tmp);
    t3 = t3.sub(a_a);
    t3 = t3.add(b_b);

    // t2 = (c0 + c1) * (f.c0 + f.c1) - a_a - b_b
    var t2 = c0.add(c1);
    tmp = f.c0.add(f.c1);
    t2 = t2.mul(tmp);
    t2 = t2.sub(a_a);
    t2 = t2.sub(b_b);

    return Fp6{ .c0 = t1, .c1 = t2, .c2 = t3 };
}

/// Multiply Fp6 by c1 only (element at position 1)
pub fn fp6MulBy1(f: Fp6, c1: Fp2) Fp6 {
    // Sparse multiplication where only c1 is non-zero
    // (f.c0 + f.c1*v + f.c2*v^2) * (c1*v) = f.c0*c1*v + f.c1*c1*v^2 + f.c2*c1*v^3
    //                                    = f.c0*c1*v + f.c1*c1*v^2 + f.c2*c1*xi
    const b_b = f.c1.mul(c1);

    return Fp6{
        .c0 = Fp6.mulByXi(f.c2.mul(c1)),
        .c1 = f.c0.mul(c1),
        .c2 = b_b,
    };
}

// ============================================================================
// Fp12 sparse multiplication for pairing
// ============================================================================

/// Multiply Fp12 by sparse element with components at positions 0, 3, 4
/// This is for D-type twist line evaluation
/// The sparse element is: c0 + c3*w + c4*v*w where w^2=v
pub fn fp12MulBy034(f: Fp12, c0: Fp2, c3: Fp2, c4: Fp2) Fp12 {
    // Following arkworks mul_by_034 exactly
    // a = f.c0 * c0 (multiply each component of c0 Fp6 by c0 Fp2)
    const a0 = f.c0.c0.mul(c0);
    const a1 = f.c0.c1.mul(c0);
    const a2 = f.c0.c2.mul(c0);
    const a = Fp6{ .c0 = a0, .c1 = a1, .c2 = a2 };

    // b = f.c1.mul_by_01(c3, c4)
    const b = fp6MulBy01(f.c1, c3, c4);

    // Karatsuba trick
    const c0_plus_c3 = c0.add(c3);
    // e = (f.c0 + f.c1).mul_by_01(c0 + c3, c4)
    const f_sum = Fp6{ .c0 = f.c0.c0.add(f.c1.c0), .c1 = f.c0.c1.add(f.c1.c1), .c2 = f.c0.c2.add(f.c1.c2) };
    const e = fp6MulBy01(f_sum, c0_plus_c3, c4);

    // result.c1 = e - a - b
    const c1_new = Fp6{
        .c0 = e.c0.sub(a.c0).sub(b.c0),
        .c1 = e.c1.sub(a.c1).sub(b.c1),
        .c2 = e.c2.sub(a.c2).sub(b.c2),
    };

    // result.c0 = a + b * v (multiply by non-residue and add)
    const b_times_v = fp6MulByV(b);
    const c0_new = Fp6{
        .c0 = a.c0.add(b_times_v.c0),
        .c1 = a.c1.add(b_times_v.c1),
        .c2 = a.c2.add(b_times_v.c2),
    };

    return Fp12{ .c0 = c0_new, .c1 = c1_new };
}

/// Multiply two 034-sparse Fp12 elements.
/// Each has form (c0, 0, 0) + (c3, c4, 0)*w in the Fp6[w]/(w^2-v) tower.
/// Cost: 6 Fp2.mul (vs 2x13 = 26 for two sequential fp12MulBy034).
pub fn fp12Mul034By034(a0: Fp2, a3: Fp2, a4: Fp2, b0: Fp2, b3: Fp2, b4: Fp2) Fp12 {
    const x0 = a0.mul(b0);
    const x3 = a3.mul(b3);
    const x4 = a4.mul(b4);
    const x03 = a0.add(a3).mul(b0.add(b3)).sub(x0).sub(x3);
    const x04 = a0.add(a4).mul(b0.add(b4)).sub(x0).sub(x4);
    const x34 = a3.add(a4).mul(b3.add(b4)).sub(x3).sub(x4);

    return Fp12{
        .c0 = Fp6{
            .c0 = x0.add(Fp6.mulByXi(x4)),
            .c1 = x3,
            .c2 = x34,
        },
        .c1 = Fp6{
            .c0 = x03,
            .c1 = x04,
            .c2 = Fp2.zero(),
        },
    };
}

/// Multiply full Fp12 by 01234-sparse element (s.c1.c2 == 0).
/// Cost: 17 Fp2.mul (vs 18 for full Fp12.mul).
pub fn fp12MulBy01234(f: Fp12, s: Fp12) Fp12 {
    // p = f.c0 * s.c0 (full Fp6 mul: 6 Fp2.mul)
    const p = f.c0.mul(s.c0);
    // q = fp6MulBy01(f.c1, s.c1.c0, s.c1.c1) (sparse: 5 Fp2.mul)
    const q = fp6MulBy01(f.c1, s.c1.c0, s.c1.c1);
    // cross = (f.c0 + f.c1) * (s.c0 + s.c1) where s.c1.c2=0 makes s_sum dense
    const s_sum = Fp6{
        .c0 = s.c0.c0.add(s.c1.c0),
        .c1 = s.c0.c1.add(s.c1.c1),
        .c2 = s.c0.c2,
    };
    const f_sum = Fp6{
        .c0 = f.c0.c0.add(f.c1.c0),
        .c1 = f.c0.c1.add(f.c1.c1),
        .c2 = f.c0.c2.add(f.c1.c2),
    };
    // full Fp6 mul: 6 Fp2.mul
    const cross = f_sum.mul(s_sum);

    const q_v = fp6MulByV(q);
    return Fp12{
        .c0 = Fp6{
            .c0 = p.c0.add(q_v.c0),
            .c1 = p.c1.add(q_v.c1),
            .c2 = p.c2.add(q_v.c2),
        },
        .c1 = Fp6{
            .c0 = cross.c0.sub(p.c0).sub(q.c0),
            .c1 = cross.c1.sub(p.c1).sub(q.c1),
            .c2 = cross.c2.sub(p.c2).sub(q.c2),
        },
    };
}

/// Multiply Fp12 by sparse element (1, 0, 0, c3, c4, 0) where c0=1 is implicit.
/// Cost: 10 Fp2.mul (vs 13 for fp12MulBy034 with explicit c0).
pub fn fp12MulBy34(f: Fp12, c3: Fp2, c4: Fp2) Fp12 {
    // a = f.c0 * 1 = f.c0 (FREE -- saves 3 Fp2.mul)
    const a = f.c0;

    // b = fp6MulBy01(f.c1, c3, c4) -- 5 Fp2.mul
    const b = fp6MulBy01(f.c1, c3, c4);

    // cross = fp6MulBy01(f.c0 + f.c1, 1 + c3, c4) -- 5 Fp2.mul
    const f_sum = Fp6{
        .c0 = f.c0.c0.add(f.c1.c0),
        .c1 = f.c0.c1.add(f.c1.c1),
        .c2 = f.c0.c2.add(f.c1.c2),
    };
    const e = fp6MulBy01(f_sum, Fp2.one().add(c3), c4);

    const b_v = fp6MulByV(b);
    return Fp12{
        .c0 = Fp6{
            .c0 = a.c0.add(b_v.c0),
            .c1 = a.c1.add(b_v.c1),
            .c2 = a.c2.add(b_v.c2),
        },
        .c1 = Fp6{
            .c0 = e.c0.sub(a.c0).sub(b.c0),
            .c1 = e.c1.sub(a.c1).sub(b.c1),
            .c2 = e.c2.sub(a.c2).sub(b.c2),
        },
    };
}

/// Multiply two 34-sparse Fp12 elements (c0=1 implicit for both).
/// Each has form 1 + c3*w + c4*vw.
/// Cost: 3 Fp2.mul (vs 6 for fp12Mul034By034 with explicit c0).
pub fn fp12Mul34By34(a3: Fp2, a4: Fp2, b3: Fp2, b4: Fp2) Fp12 {
    // x0 = 1 (free)
    const x3 = a3.mul(b3);
    const x4 = a4.mul(b4);
    // x03 = (1+a3)(1+b3) - 1 - x3 = a3 + b3 (FREE)
    const x03 = a3.add(b3);
    // x04 = (1+a4)(1+b4) - 1 - x4 = a4 + b4 (FREE)
    const x04 = a4.add(b4);
    const x34 = a3.add(a4).mul(b3.add(b4)).sub(x3).sub(x4);

    return Fp12{
        .c0 = Fp6{
            .c0 = Fp2.one().add(Fp6.mulByXi(x4)),
            .c1 = x3,
            .c2 = x34,
        },
        .c1 = Fp6{
            .c0 = x03,
            .c1 = x04,
            .c2 = Fp2.zero(),
        },
    };
}

// ============================================================================
// Tests
// ============================================================================

test "Fp2 arithmetic" {
    const a = Fp2.init(Fp.fromU64(3), Fp.fromU64(4));
    const b = Fp2.init(Fp.fromU64(1), Fp.fromU64(2));

    // Test addition
    const sum = a.add(b);
    try std.testing.expect(sum.c0.eql(Fp.fromU64(4)));
    try std.testing.expect(sum.c1.eql(Fp.fromU64(6)));

    // Test multiplication
    const prod = a.mul(b);
    // (3 + 4i)(1 + 2i) = 3 + 6i + 4i + 8i^2 = 3 + 10i - 8 = -5 + 10i
    try std.testing.expect(prod.c0.eql(Fp.fromU64(3).sub(Fp.fromU64(8))));
    try std.testing.expect(prod.c1.eql(Fp.fromU64(10)));
}

test "Fp2 inverse" {
    const a = Fp2.init(Fp.fromU64(3), Fp.fromU64(4));
    const a_inv = a.inverse().?;
    const should_be_one = a.mul(a_inv);

    try std.testing.expect(should_be_one.c0.eql(Fp.one()));
    try std.testing.expect(should_be_one.c1.eql(Fp.zero()));
}

test "Fp12 basic operations" {
    const one = Fp12.one();
    const zero = Fp12.zero();

    // 1 + 0 = 1
    const sum = one.add(zero);
    try std.testing.expect(sum.eql(one));

    // 1 * 1 = 1
    const prod = one.mul(one);
    try std.testing.expect(prod.eql(one));
}

test "Fp6 operations" {
    const one = Fp6.one();
    const zero = Fp6.zero();

    // 1 + 0 = 1
    try std.testing.expect(one.add(zero).eql(one));

    // 1 * 1 = 1
    try std.testing.expect(one.mul(one).eql(one));

    // 1 - 1 = 0
    try std.testing.expect(one.sub(one).eql(zero));
}

test "Fp6 inverse" {
    const a = Fp6{
        .c0 = Fp2.init(Fp.fromU64(1), Fp.fromU64(2)),
        .c1 = Fp2.init(Fp.fromU64(3), Fp.fromU64(4)),
        .c2 = Fp2.init(Fp.fromU64(5), Fp.fromU64(6)),
    };

    if (a.inverse()) |a_inv| {
        const should_be_one = a.mul(a_inv);
        try std.testing.expect(should_be_one.eql(Fp6.one()));
    }
}

test "Fp6 mulByXi is correct for xi = 9 + u" {
    // Test that mulByXi(a) = a * (9 + u) for a simple Fp2 element
    const a = Fp2.init(Fp.fromU64(2), Fp.fromU64(3));

    // Compute using mulByXi
    const result = Fp6.mulByXi(a);

    // Compute manually: (2 + 3u)(9 + u) = 18 + 2u + 27u + 3u^2 = 18 + 29u - 3 = 15 + 29u
    // (since u^2 = -1)
    const expected = Fp2.init(Fp.fromU64(15), Fp.fromU64(29));

    try std.testing.expect(result.eql(expected));
}

test "Fp6 mul by v via mulByXi" {
    // Test that multiplying by v works correctly
    // v^3 = xi, so (c0 + c1*v + c2*v^2) * v = c2*xi + c0*v + c1*v^2
    const f = Fp6{
        .c0 = Fp2.init(Fp.fromU64(1), Fp.fromU64(0)),
        .c1 = Fp2.init(Fp.fromU64(2), Fp.fromU64(0)),
        .c2 = Fp2.init(Fp.fromU64(3), Fp.fromU64(0)),
    };

    // Expected: c0' = 3*(9+u) = 27 + 3u, c1' = 1, c2' = 2
    const result = fp6MulByV(f);

    try std.testing.expect(result.c0.eql(Fp2.init(Fp.fromU64(27), Fp.fromU64(3))));
    try std.testing.expect(result.c1.eql(Fp2.init(Fp.fromU64(1), Fp.fromU64(0))));
    try std.testing.expect(result.c2.eql(Fp2.init(Fp.fromU64(2), Fp.fromU64(0))));
}

test "Frobenius on Fp12" {
    // Test that Frobenius is well-defined
    const a = Fp12{
        .c0 = Fp6{
            .c0 = Fp2.init(Fp.fromU64(1), Fp.fromU64(2)),
            .c1 = Fp2.init(Fp.fromU64(3), Fp.fromU64(4)),
            .c2 = Fp2.init(Fp.fromU64(5), Fp.fromU64(6)),
        },
        .c1 = Fp6{
            .c0 = Fp2.init(Fp.fromU64(7), Fp.fromU64(8)),
            .c1 = Fp2.init(Fp.fromU64(9), Fp.fromU64(10)),
            .c2 = Fp2.init(Fp.fromU64(11), Fp.fromU64(12)),
        },
    };

    const a_frob = a.frobenius();

    // Frobenius should not return the same element (in general)
    // but applying it p times should return the original (for elements in Fp)
    // For our test, just check it's well-defined
    try std.testing.expect(!a_frob.c0.c0.c0.isZero() or !a_frob.c0.c0.c1.isZero());
}

test "Fp12 toBytes/fromBytes roundtrip" {
    // Create a non-trivial Fp12 element
    const fp12 = Fp12{
        .c0 = Fp6{
            .c0 = Fp2.init(Fp.fromU64(1), Fp.fromU64(2)),
            .c1 = Fp2.init(Fp.fromU64(3), Fp.fromU64(4)),
            .c2 = Fp2.init(Fp.fromU64(5), Fp.fromU64(6)),
        },
        .c1 = Fp6{
            .c0 = Fp2.init(Fp.fromU64(7), Fp.fromU64(8)),
            .c1 = Fp2.init(Fp.fromU64(9), Fp.fromU64(10)),
            .c2 = Fp2.init(Fp.fromU64(11), Fp.fromU64(12)),
        },
    };

    // Serialize to bytes
    const bytes = fp12.toBytes();

    // Verify it's 384 bytes
    try std.testing.expectEqual(@as(usize, 384), bytes.len);

    // Deserialize back
    const decoded = Fp12.fromBytes(&bytes);

    // Verify equality
    try std.testing.expect(fp12.eql(decoded));
}

test "Fp12 toBytes matches expected format" {
    // Test that Fp12.one() serializes correctly
    const one = Fp12.one();
    const bytes = one.toBytes();

    // Fp12.one() = (Fp6.one(), Fp6.zero())
    // Fp6.one() = (Fp2.one(), Fp2.zero(), Fp2.zero())
    // Fp2.one() = (Fp.one(), Fp.zero())
    // So first 32 bytes should be 1 in LE, rest should be 0

    // First Fp element should be 1 (c0.c0.c0 = Fp.one())
    const first_value = std.mem.readInt(u64, bytes[0..8], .little);
    try std.testing.expectEqual(@as(u64, 1), first_value);

    // Rest of first 32 bytes should be 0
    for (8..32) |i| {
        try std.testing.expectEqual(@as(u8, 0), bytes[i]);
    }

    // All remaining 352 bytes should be 0
    for (32..384) |i| {
        try std.testing.expectEqual(@as(u8, 0), bytes[i]);
    }
}

test "GT alias equals Fp12" {
    const gt_one = GT.one();
    const fp12_one = Fp12.one();
    try std.testing.expect(gt_one.eql(fp12_one));
}

test "fp12Mul034By034 matches full Fp12.mul" {
    const a0 = Fp2.init(Fp.fromU64(7), Fp.fromU64(11));
    const a3 = Fp2.init(Fp.fromU64(13), Fp.fromU64(17));
    const a4 = Fp2.init(Fp.fromU64(19), Fp.fromU64(23));
    const b0 = Fp2.init(Fp.fromU64(29), Fp.fromU64(31));
    const b3 = Fp2.init(Fp.fromU64(37), Fp.fromU64(41));
    const b4 = Fp2.init(Fp.fromU64(43), Fp.fromU64(47));

    const full_a = Fp12{
        .c0 = Fp6{ .c0 = a0, .c1 = Fp2.zero(), .c2 = Fp2.zero() },
        .c1 = Fp6{ .c0 = a3, .c1 = a4, .c2 = Fp2.zero() },
    };
    const full_b = Fp12{
        .c0 = Fp6{ .c0 = b0, .c1 = Fp2.zero(), .c2 = Fp2.zero() },
        .c1 = Fp6{ .c0 = b3, .c1 = b4, .c2 = Fp2.zero() },
    };

    const expected = full_a.mul(full_b);
    const sparse = fp12Mul034By034(a0, a3, a4, b0, b3, b4);
    try std.testing.expect(expected.eql(sparse));
}

test "fp12MulBy01234 matches full Fp12.mul" {
    const f = Fp12{
        .c0 = Fp6{
            .c0 = Fp2.init(Fp.fromU64(2), Fp.fromU64(3)),
            .c1 = Fp2.init(Fp.fromU64(5), Fp.fromU64(7)),
            .c2 = Fp2.init(Fp.fromU64(11), Fp.fromU64(13)),
        },
        .c1 = Fp6{
            .c0 = Fp2.init(Fp.fromU64(17), Fp.fromU64(19)),
            .c1 = Fp2.init(Fp.fromU64(23), Fp.fromU64(29)),
            .c2 = Fp2.init(Fp.fromU64(31), Fp.fromU64(37)),
        },
    };
    const s = Fp12{
        .c0 = Fp6{
            .c0 = Fp2.init(Fp.fromU64(41), Fp.fromU64(43)),
            .c1 = Fp2.init(Fp.fromU64(47), Fp.fromU64(53)),
            .c2 = Fp2.init(Fp.fromU64(59), Fp.fromU64(61)),
        },
        .c1 = Fp6{
            .c0 = Fp2.init(Fp.fromU64(67), Fp.fromU64(71)),
            .c1 = Fp2.init(Fp.fromU64(73), Fp.fromU64(79)),
            .c2 = Fp2.zero(),
        },
    };

    const expected = f.mul(s);
    const result = fp12MulBy01234(f, s);
    try std.testing.expect(expected.eql(result));
}

test "fp12MulBy34 matches fp12MulBy034 with c0=1" {
    const f = Fp12{
        .c0 = Fp6{
            .c0 = Fp2.init(Fp.fromU64(2), Fp.fromU64(3)),
            .c1 = Fp2.init(Fp.fromU64(5), Fp.fromU64(7)),
            .c2 = Fp2.init(Fp.fromU64(11), Fp.fromU64(13)),
        },
        .c1 = Fp6{
            .c0 = Fp2.init(Fp.fromU64(17), Fp.fromU64(19)),
            .c1 = Fp2.init(Fp.fromU64(23), Fp.fromU64(29)),
            .c2 = Fp2.init(Fp.fromU64(31), Fp.fromU64(37)),
        },
    };
    const c3 = Fp2.init(Fp.fromU64(41), Fp.fromU64(43));
    const c4 = Fp2.init(Fp.fromU64(47), Fp.fromU64(53));

    const expected = fp12MulBy034(f, Fp2.one(), c3, c4);
    const result = fp12MulBy34(f, c3, c4);
    try std.testing.expect(expected.eql(result));
}

test "fp12Mul34By34 matches fp12Mul034By034 with c0=1" {
    const a3 = Fp2.init(Fp.fromU64(7), Fp.fromU64(11));
    const a4 = Fp2.init(Fp.fromU64(13), Fp.fromU64(17));
    const b3 = Fp2.init(Fp.fromU64(23), Fp.fromU64(29));
    const b4 = Fp2.init(Fp.fromU64(31), Fp.fromU64(37));

    const expected = fp12Mul034By034(Fp2.one(), a3, a4, Fp2.one(), b3, b4);
    const result = fp12Mul34By34(a3, a4, b3, b4);
    try std.testing.expect(expected.eql(result));
}
