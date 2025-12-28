//! BN254 Pairing Operations
//!
//! This module implements elliptic curve pairing operations for the BN254 curve.
//! Pairings are bilinear maps: e(P, Q) where P ∈ G1, Q ∈ G2, and the result is in GT (Fp12).
//!
//! BN254 Curve Parameters:
//! - Base field: Fp (254-bit prime)
//! - Scalar field: Fr (used for scalars)
//! - G1: Points on y² = x³ + 3 over Fp
//! - G2: Points on y² = x³ + 3/ξ over Fp² (sextic twist)
//! - GT: Subgroup of Fp12*
//!
//! The pairing is computed via the optimal ate pairing:
//! 1. Miller loop: Compute f_{6x+2,Q}(P)
//! 2. Final exponentiation: f^((p^12-1)/r)
//!
//! Generator Points (Ethereum/EIP-196/EIP-197 convention):
//! - G1 generator: (1, 2)
//! - G2 generator (in Fp2):
//!   X = (x0, x1) where:
//!     x0 = 0x1800deef121f1e76426a00665e5c4479674322d4f75edadd46debd5cd992f6ed
//!     x1 = 0x198e9393920d483a7260bfb731fb5d25f1aa493335a9e71297e485b7aef312c2
//!   Y = (y0, y1) where:
//!     y0 = 0x12c85ea5db8c6deb4aab71808dcb408fe3d1e7690c43d37b4ce6cc0166fa7daa
//!     y1 = 0x090689d0585ff075ec9e99ad690c3395bc4b313370b38ef355acdadcd122975b
//!
//! Implementation based on:
//! - https://eprint.iacr.org/2024/640.pdf
//! - gnark-crypto BN254 implementation
//! - ziskos BN254 implementation

const std = @import("std");
const field_mod = @import("mod.zig");
const BN254Scalar = field_mod.BN254Scalar; // Scalar field Fr (for MSM scalars)
const Fp = field_mod.BN254BaseField; // Base field Fp (for pairing operations)
const msm = @import("../msm/mod.zig");

// ============================================================================
// Frobenius Coefficients for BN254
// ============================================================================
//
// These constants are needed for the Frobenius endomorphism on G2 and Fp12.
// For BN254 with ξ = 9 + u (the non-residue), we need:
//   γ_{1,j} = ξ^{j(p-1)/6} for j = 1..5
//
// From ziskos BN254 implementation (constants.rs)
// Stored as [u64; 8] for Fp2 (two Fp elements) and [u64; 4] for Fp

/// Helper to create Fp2 from [u64; 8] limbs (raw, non-Montgomery form)
/// Converts each Fp element to Montgomery form
fn fp2FromLimbs(limbs: [8]u64) Fp2 {
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
fn fpFromLimbs(limbs: [4]u64) Fp {
    const raw = Fp{ .limbs = limbs };
    return raw.toMontgomery();
}

// ============================================================================
// Frobenius^1 Coefficients (require conjugation)
// ============================================================================

/// γ₁₁ = ξ^{(p-1)/6} - Frobenius^1 coefficient for c1 of Fp6 in c1 of Fp12
const FROBENIUS_GAMMA11: [8]u64 = .{
    0xD60B35DADCC9E470, 0x5C521E08292F2176, 0xE8B99FDD76E68B60, 0x1284B71C2865A7DF,
    0xCA5CF05F80F362AC, 0x747992778EEEC7E5, 0xA6327CFE12150B8E, 0x246996F3B4FAE7E6,
};

/// γ₁₂ = ξ^{2(p-1)/6} - Frobenius^1 coefficient for c1 of Fp6 in c0 of Fp12 (also G2 x-coord)
const FROBENIUS_GAMMA12: [8]u64 = .{
    0x99E39557176F553D, 0xB78CC310C2C3330C, 0x4C0BEC3CF559B143, 0x2FB347984F7911F7,
    0x1665D51C640FCBA2, 0x32AE2A1D0B7C9DCE, 0x4BA4CC8BD75A0794, 0x16C9E55061EBAE20,
};

/// γ₁₃ = ξ^{3(p-1)/6} - Frobenius^1 coefficient for c2 of Fp6 in c1 of Fp12 (also G2 y-coord)
const FROBENIUS_GAMMA13: [8]u64 = .{
    0xDC54014671A0135A, 0xDBAAE0EDA9C95998, 0xDC5EC698B6E2F9B9, 0x063CF305489AF5DC,
    0x82D37F632623B0E3, 0x21807DC98FA25BD2, 0x0704B5A7EC796F2B, 0x07C03CBCAC41049A,
};

/// γ₁₄ = ξ^{4(p-1)/6} - Frobenius^1 coefficient for c2 of Fp6 in c0 of Fp12
const FROBENIUS_GAMMA14: [8]u64 = .{
    0x848A1F55921EA762, 0xD33365F7BE94EC72, 0x80F3C0B75A181E84, 0x05B54F5E64EEA801,
    0xC13B4711CD2B8126, 0x3685D2EA1BDEC763, 0x9F3A80B03B0B1C92, 0x2C145EDBE7FD8AEE,
};

/// γ₁₅ = ξ^{5(p-1)/6} - Frobenius^1 coefficient for c2 of Fp6 in c1 of Fp12
const FROBENIUS_GAMMA15: [8]u64 = .{
    0x2EA2C810EAB7692F, 0x425C459B55AA1BD3, 0xE93A3661A4353FF4, 0x0183C1E74F798649,
    0x24C6B8EE6E0C2C4B, 0xB080CB99678E2AC0, 0xA27FB246C7729F7D, 0x12ACF2CA76FD0675,
};

// ============================================================================
// Frobenius^2 Coefficients (no conjugation needed - even power)
// ============================================================================

/// γ₂₁ = ξ^{(p²-1)/6} - Fp element
const FROBENIUS_GAMMA21: [4]u64 = .{ 0xE4BD44E5607CFD49, 0xC28F069FBB966E3D, 0x5E6DD9E7E0ACCCB0, 0x30644E72E131A029 };

/// γ₂₂ = ξ^{2(p²-1)/6} - Fp element
const FROBENIUS_GAMMA22: [4]u64 = .{ 0xE4BD44E5607CFD48, 0xC28F069FBB966E3D, 0x5E6DD9E7E0ACCCB0, 0x30644E72E131A029 };

/// γ₂₃ = ξ^{3(p²-1)/6} - Fp element
const FROBENIUS_GAMMA23: [4]u64 = .{ 0x3C208C16D87CFD46, 0x97816A916871CA8D, 0xB85045B68181585D, 0x30644E72E131A029 };

/// γ₂₄ = ξ^{4(p²-1)/6} - Fp element
const FROBENIUS_GAMMA24: [4]u64 = .{ 0x5763473177FFFFFE, 0xD4F263F1ACDB5C4F, 0x59E26BCEA0D48BAC, 0x0000000000000000 };

/// γ₂₅ = ξ^{5(p²-1)/6} - Fp element
const FROBENIUS_GAMMA25: [4]u64 = .{ 0x5763473177FFFFFF, 0xD4F263F1ACDB5C4F, 0x59E26BCEA0D48BAC, 0x0000000000000000 };

// ============================================================================
// Frobenius^3 Coefficients (require conjugation - odd power)
// ============================================================================

/// γ₃₁ = ξ^{(p³-1)/6}
const FROBENIUS_GAMMA31: [8]u64 = .{
    0xE86F7D391ED4A67F, 0x894CB38DBE55D24A, 0xEFE9608CD0ACAA90, 0x19DC81CFCC82E4BB,
    0x7694AA2BF4C0C101, 0x7F03A5E397D439EC, 0x06CBEEE33576139D, 0x00ABF8B60BE77D73,
};

/// γ₃₂ = ξ^{2(p³-1)/6}
const FROBENIUS_GAMMA32: [8]u64 = .{
    0x7B746EE87BDCFB6D, 0x805FFD3D5D6942D3, 0xBAFF1C77959F25AC, 0x0856E078B755EF0A,
    0x380CAB2BAAA586DE, 0x0FDF31BF98FF2631, 0xA9F30E6DEC26094F, 0x04F1DE41B3D1766F,
};

/// γ₃₃ = ξ^{3(p³-1)/6}
const FROBENIUS_GAMMA33: [8]u64 = .{
    0x5FCC8AD066DCE9ED, 0xBBD689A3BEA870F4, 0xDBF17F1DCA9E5EA3, 0x2A275B6D9896AA4C,
    0xB94D0CB3B2594C64, 0x7600ECC7D8CF6EBA, 0xB14B900E9507E932, 0x28A411B634F09B8F,
};

/// γ₃₄ = ξ^{4(p³-1)/6}
const FROBENIUS_GAMMA34: [8]u64 = .{
    0x0E1A92BC3CCBF066, 0xE633094575B06BCB, 0x19BEE0F7B5B2444E, 0x0BC58C6611C08DAB,
    0x5FE3ED9D730C239F, 0xA44A9E08737F96E5, 0xFEB0F6EF0CD21D04, 0x23D5E999E1910A12,
};

/// γ₃₅ = ξ^{5(p³-1)/6}
const FROBENIUS_GAMMA35: [8]u64 = .{
    0xEBDE847076261B43, 0x2ED68098967C84A5, 0x711699FA3B4D3F69, 0x13C49044952C0905,
    0x1F25041384282499, 0x3E2DDAEA20028021, 0x9FB1B2282A48633D, 0x16DB366A59B1DD0B,
};

/// GAMMA_12 = ξ^{(p-1)/3} = ξ^{2(p-1)/6}
/// Used for G2 x-coordinate in Frobenius
fn gamma12() Fp2 {
    return fp2FromLimbs(FROBENIUS_GAMMA12);
}

/// GAMMA_13 = ξ^{(p-1)/2} = ξ^{3(p-1)/6}
/// Used for G2 y-coordinate in Frobenius
fn gamma13() Fp2 {
    return fp2FromLimbs(FROBENIUS_GAMMA13);
}

// ============================================================================
// Fp2 scalar multiplication helper
// ============================================================================

/// Multiply Fp2 element by Fp scalar (embeds scalar as (s, 0))
fn fp2ScalarMul(a: Fp2, s: Fp) Fp2 {
    return Fp2.init(a.c0.mul(s), a.c1.mul(s));
}

// ============================================================================
// Extension Field Fp2 = Fp[u] / (u² + 1)
// ============================================================================

/// Fp2 element: a + b*u where u² = -1
pub const Fp2 = struct {
    c0: Fp, // Real part
    c1: Fp, // Imaginary part

    pub fn init(c0: Fp, c1: Fp) Fp2 {
        return .{ .c0 = c0, .c1 = c1 };
    }

    pub fn zero() Fp2 {
        return .{ .c0 = Fp.zero(), .c1 = Fp.zero() };
    }

    pub fn one() Fp2 {
        return .{ .c0 = Fp.one(), .c1 = Fp.zero() };
    }

    pub fn add(self: Fp2, other: Fp2) Fp2 {
        return .{
            .c0 = self.c0.add(other.c0),
            .c1 = self.c1.add(other.c1),
        };
    }

    pub fn sub(self: Fp2, other: Fp2) Fp2 {
        return .{
            .c0 = self.c0.sub(other.c0),
            .c1 = self.c1.sub(other.c1),
        };
    }

    pub fn mul(self: Fp2, other: Fp2) Fp2 {
        // (a + bu)(c + du) = (ac - bd) + (ad + bc)u
        const ac = self.c0.mul(other.c0);
        const bd = self.c1.mul(other.c1);
        const ad = self.c0.mul(other.c1);
        const bc = self.c1.mul(other.c0);

        return .{
            .c0 = ac.sub(bd),
            .c1 = ad.add(bc),
        };
    }

    pub fn square(self: Fp2) Fp2 {
        // (a + bu)² = (a² - b²) + 2abu
        // Karatsuba optimization: (a+b)(a-b) = a² - b²
        const a_plus_b = self.c0.add(self.c1);
        const a_minus_b = self.c0.sub(self.c1);
        const a_squared_minus_b_squared = a_plus_b.mul(a_minus_b);
        const two_ab = self.c0.mul(self.c1).double();

        return .{
            .c0 = a_squared_minus_b_squared,
            .c1 = two_ab,
        };
    }

    pub fn neg(self: Fp2) Fp2 {
        return .{
            .c0 = self.c0.neg(),
            .c1 = self.c1.neg(),
        };
    }

    /// Conjugate: a + bu -> a - bu
    pub fn conjugate(self: Fp2) Fp2 {
        return .{
            .c0 = self.c0,
            .c1 = self.c1.neg(),
        };
    }

    /// Inverse using the formula: 1/(a + bu) = (a - bu)/(a² + b²)
    pub fn inverse(self: Fp2) ?Fp2 {
        const norm = self.c0.square().add(self.c1.square());
        const norm_inv = norm.inverse() orelse return null;

        return .{
            .c0 = self.c0.mul(norm_inv),
            .c1 = self.c1.neg().mul(norm_inv),
        };
    }

    pub fn eql(self: Fp2, other: Fp2) bool {
        return self.c0.eql(other.c0) and self.c1.eql(other.c1);
    }

    pub fn isZero(self: Fp2) bool {
        return self.c0.isZero() and self.c1.isZero();
    }
};

// ============================================================================
// Extension Field Fp6 = Fp2[v] / (v³ - ξ) where ξ = 9 + u
// ============================================================================

/// Fp6 element: c0 + c1*v + c2*v² where v³ = ξ
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

    pub fn add(self: Fp6, other: Fp6) Fp6 {
        return .{
            .c0 = self.c0.add(other.c0),
            .c1 = self.c1.add(other.c1),
            .c2 = self.c2.add(other.c2),
        };
    }

    pub fn sub(self: Fp6, other: Fp6) Fp6 {
        return .{
            .c0 = self.c0.sub(other.c0),
            .c1 = self.c1.sub(other.c1),
            .c2 = self.c2.sub(other.c2),
        };
    }

    pub fn neg(self: Fp6) Fp6 {
        return .{
            .c0 = self.c0.neg(),
            .c1 = self.c1.neg(),
            .c2 = self.c2.neg(),
        };
    }

    /// Multiplication by ξ = 9 + u (the non-residue for BN254)
    fn mulByXi(x: Fp2) Fp2 {
        // ξ * (a + bu) = (a + bu)(9 + u) = (9a - b) + (a + 9b)u
        const nine = Fp.fromU64(9);
        return Fp2.init(
            nine.mul(x.c0).sub(x.c1),
            x.c0.add(nine.mul(x.c1)),
        );
    }

    pub fn mul(self: Fp6, other: Fp6) Fp6 {
        // Karatsuba-like multiplication for cubic extension
        const v0 = self.c0.mul(other.c0);
        const v1 = self.c1.mul(other.c1);
        const v2 = self.c2.mul(other.c2);

        // c0 = v0 + ξ((c1 + c2)(d1 + d2) - v1 - v2)
        const c1_plus_c2 = self.c1.add(self.c2);
        const d1_plus_d2 = other.c1.add(other.c2);
        const t0 = mulByXi(c1_plus_c2.mul(d1_plus_d2).sub(v1).sub(v2));
        const new_c0 = v0.add(t0);

        // c1 = (c0 + c1)(d0 + d1) - v0 - v1 + ξ*v2
        const c0_plus_c1 = self.c0.add(self.c1);
        const d0_plus_d1 = other.c0.add(other.c1);
        const t1 = c0_plus_c1.mul(d0_plus_d1).sub(v0).sub(v1);
        const new_c1 = t1.add(mulByXi(v2));

        // c2 = (c0 + c2)(d0 + d2) - v0 - v2 + v1
        const c0_plus_c2 = self.c0.add(self.c2);
        const d0_plus_d2 = other.c0.add(other.c2);
        const t2 = c0_plus_c2.mul(d0_plus_d2).sub(v0).sub(v2);
        const new_c2 = t2.add(v1);

        return .{ .c0 = new_c0, .c1 = new_c1, .c2 = new_c2 };
    }

    pub fn square(self: Fp6) Fp6 {
        return self.mul(self);
    }

    pub fn inverse(self: Fp6) ?Fp6 {
        // Extended Euclidean algorithm for Fp6
        const c0_sq = self.c0.square();
        const c1_sq = self.c1.square();
        const c2_sq = self.c2.square();
        const c0c1 = self.c0.mul(self.c1);
        const c0c2 = self.c0.mul(self.c2);
        const c1c2 = self.c1.mul(self.c2);

        // Using the formula for inverse in cubic extension
        const a0 = c0_sq.sub(mulByXi(c1c2));
        const a1 = mulByXi(c2_sq).sub(c0c1);
        const a2 = c1_sq.sub(c0c2);

        const tmp = mulByXi(self.c1.mul(a2).add(self.c2.mul(a1)));
        const norm = self.c0.mul(a0).add(tmp);

        const norm_inv = norm.inverse() orelse return null;

        return .{
            .c0 = a0.mul(norm_inv),
            .c1 = a1.mul(norm_inv),
            .c2 = a2.mul(norm_inv),
        };
    }

    pub fn eql(self: Fp6, other: Fp6) bool {
        return self.c0.eql(other.c0) and self.c1.eql(other.c1) and self.c2.eql(other.c2);
    }

    // Note: Fp6 frobenius is not needed as a standalone method.
    // We implement it directly in Fp12 frobenius to properly handle
    // the coefficient structure across all 6 Fp2 components.
};

/// Fp6 multiplication by v (shift operation)
/// For Fp6 = Fp2[v]/(v³ - ξ), multiplying by v shifts coefficients:
/// (c0 + c1*v + c2*v²) * v = c2*ξ + c0*v + c1*v²
fn fp6MulByV(f: Fp6) Fp6 {
    return Fp6{
        .c0 = Fp6.mulByXi(f.c2),
        .c1 = f.c0,
        .c2 = f.c1,
    };
}

// ============================================================================
// Extension Field Fp12 = Fp6[w] / (w² - v)
// ============================================================================

/// Fp12 element: c0 + c1*w where w² = v
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

    /// Multiplication in Fp12
    pub fn mul(self: Fp12, other: Fp12) Fp12 {
        // (a + bw)(c + dw) = (ac + bdv) + (ad + bc)w
        // where v is the element that w² = v
        const ac = self.c0.mul(other.c0);
        const bd = self.c1.mul(other.c1);
        const ad = self.c0.mul(other.c1);
        const bc = self.c1.mul(other.c0);

        // Multiply bd by v (shift in the Fp6 tower)
        const bd_times_v = Fp6{
            .c0 = Fp6.mulByXi(bd.c2),
            .c1 = bd.c0,
            .c2 = bd.c1,
        };

        return .{
            .c0 = ac.add(bd_times_v),
            .c1 = ad.add(bc),
        };
    }

    pub fn square(self: Fp12) Fp12 {
        return self.mul(self);
    }

    /// Conjugate for unitary elements: a + bw -> a - bw
    pub fn conjugate(self: Fp12) Fp12 {
        return .{
            .c0 = self.c0,
            .c1 = self.c1.neg(),
        };
    }

    /// Frobenius^1 endomorphism (raising to p-th power)
    /// For Fp12 = Fp6[w]/(w² - v) where Fp6 = Fp2[v]/(v³ - ξ)
    /// Structure: ((a11 + a12*v + a13*v²) + (a21 + a22*v + a23*v²)*w)^p
    ///
    /// From ziskos:
    /// c0 = a̅11     + a̅12·γ12·v + a̅13·γ14·v²
    /// c1 = a̅21·γ11 + a̅22·γ13·v + a̅23·γ15·v²
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
        // c0 = a̅11 + a̅12·γ12·v + a̅13·γ14·v²
        const c0 = Fp6{
            .c0 = a11.conjugate(),
            .c1 = a12.conjugate().mul(g12),
            .c2 = a13.conjugate().mul(g14),
        };

        // c1 = a̅21·γ11 + a̅22·γ13·v + a̅23·γ15·v²
        const c1 = Fp6{
            .c0 = a21.conjugate().mul(g11),
            .c1 = a22.conjugate().mul(g13),
            .c2 = a23.conjugate().mul(g15),
        };

        return Fp12{ .c0 = c0, .c1 = c1 };
    }

    /// Frobenius^2 endomorphism (raising to p² power)
    /// No conjugation needed (even power of Frobenius)
    /// Uses scalar (Fp) multiplication since γ₂ᵢ are in Fp
    ///
    /// c0 = a11     + a12·γ22·v + a13·γ24·v²
    /// c1 = a21·γ21 + a22·γ23·v + a23·γ25·v²
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

        // c0 = a11 + a12·γ22·v + a13·γ24·v²
        const c0 = Fp6{
            .c0 = a11,
            .c1 = fp2ScalarMul(a12, g22),
            .c2 = fp2ScalarMul(a13, g24),
        };

        // c1 = a21·γ21 + a22·γ23·v + a23·γ25·v²
        const c1 = Fp6{
            .c0 = fp2ScalarMul(a21, g21),
            .c1 = fp2ScalarMul(a22, g23),
            .c2 = fp2ScalarMul(a23, g25),
        };

        return Fp12{ .c0 = c0, .c1 = c1 };
    }

    /// Frobenius^3 endomorphism (raising to p³ power)
    /// Requires conjugation (odd power of Frobenius)
    ///
    /// c0 = a̅11     + a̅12·γ32·v + a̅13·γ34·v²
    /// c1 = a̅21·γ31 + a̅22·γ33·v + a̅23·γ35·v²
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
        // c0 = a̅11 + a̅12·γ32·v + a̅13·γ34·v²
        const c0 = Fp6{
            .c0 = a11.conjugate(),
            .c1 = a12.conjugate().mul(g32),
            .c2 = a13.conjugate().mul(g34),
        };

        // c1 = a̅21·γ31 + a̅22·γ33·v + a̅23·γ35·v²
        const c1 = Fp6{
            .c0 = a21.conjugate().mul(g31),
            .c1 = a22.conjugate().mul(g33),
            .c2 = a23.conjugate().mul(g35),
        };

        return Fp12{ .c0 = c0, .c1 = c1 };
    }

    pub fn inverse(self: Fp12) ?Fp12 {
        // For quadratic extension: 1/(a + bw) = (a - bw)/(a² - b²v)
        // Use the conjugate method
        const a_squared = self.c0.mul(self.c0);
        const b_squared = self.c1.mul(self.c1);

        // b²v (shift in Fp6)
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
    /// Arkworks serializes Fp12 = Fp6[w]/(w² - v) as:
    /// - c0 (Fp6) first, then c1 (Fp6)
    /// - Each Fp6 = Fp2[v]/(v³ - ξ) is serialized as c0, c1, c2
    /// - Each Fp2 = Fp[u]/(u² + 1) is serialized as c0, c1
    /// - Each Fp element is 32 bytes little-endian
    ///
    /// Order: c0.c0.c0, c0.c0.c1, c0.c1.c0, c0.c1.c1, c0.c2.c0, c0.c2.c1,
    ///        c1.c0.c0, c1.c0.c1, c1.c1.c0, c1.c1.c1, c1.c2.c0, c1.c2.c1
    /// = 12 Fp elements × 32 bytes = 384 bytes
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
// G2 Points (on the twist curve over Fp2)
// ============================================================================

/// Point on G2 (twist curve over Fp2)
pub const G2Point = struct {
    x: Fp2,
    y: Fp2,
    infinity: bool,

    pub fn identity() G2Point {
        return .{
            .x = Fp2.zero(),
            .y = Fp2.one(),
            .infinity = true,
        };
    }

    pub fn fromCoords(x: Fp2, y: Fp2) G2Point {
        return .{
            .x = x,
            .y = y,
            .infinity = false,
        };
    }

    /// Generator point for G2 (Ethereum/EIP-197 convention)
    ///
    /// The BN254 G2 generator coordinates in Fp2:
    /// X = (x0, x1), Y = (y0, y1) where:
    ///   x0 = 0x1800deef121f1e76426a00665e5c4479674322d4f75edadd46debd5cd992f6ed
    ///   x1 = 0x198e9393920d483a7260bfb731fb5d25f1aa493335a9e71297e485b7aef312c2
    ///   y0 = 0x12c85ea5db8c6deb4aab71808dcb408fe3d1e7690c43d37b4ce6cc0166fa7daa
    ///   y1 = 0x090689d0585ff075ec9e99ad690c3395bc4b313370b38ef355acdadcd122975b
    pub fn generator() G2Point {
        // G2 generator coordinates (little-endian byte representation)
        // x0 = 0x1800deef121f1e76426a00665e5c4479674322d4f75edadd46debd5cd992f6ed
        const x0_bytes = [_]u8{
            0xed, 0xf6, 0x92, 0xd9, 0x5c, 0xbd, 0xde, 0x46,
            0xdd, 0xda, 0x5e, 0xf7, 0xd4, 0x22, 0x43, 0x67,
            0x79, 0x44, 0x5c, 0x5e, 0x66, 0x00, 0x6a, 0x42,
            0x76, 0x1e, 0x1f, 0x12, 0xef, 0xde, 0x00, 0x18,
        };
        // x1 = 0x198e9393920d483a7260bfb731fb5d25f1aa493335a9e71297e485b7aef312c2
        const x1_bytes = [_]u8{
            0xc2, 0x12, 0xf3, 0xae, 0xb7, 0x85, 0xe4, 0x97,
            0x12, 0xe7, 0xa9, 0x35, 0x33, 0x49, 0xaa, 0xf1,
            0x25, 0x5d, 0xfb, 0x31, 0xb7, 0xbf, 0x60, 0x72,
            0x3a, 0x48, 0x0d, 0x92, 0x93, 0x93, 0x8e, 0x19,
        };
        // y0 = 0x12c85ea5db8c6deb4aab71808dcb408fe3d1e7690c43d37b4ce6cc0166fa7daa
        const y0_bytes = [_]u8{
            0xaa, 0x7d, 0xfa, 0x66, 0x01, 0xcc, 0xe6, 0x4c,
            0x7b, 0xd3, 0x43, 0x0c, 0x69, 0xe7, 0xd1, 0xe3,
            0x8f, 0x40, 0xcb, 0x8d, 0x80, 0x71, 0xab, 0x4a,
            0xeb, 0x6d, 0x8c, 0xdb, 0xa5, 0x5e, 0xc8, 0x12,
        };
        // y1 = 0x090689d0585ff075ec9e99ad690c3395bc4b313370b38ef355acdadcd122975b
        const y1_bytes = [_]u8{
            0x5b, 0x97, 0x22, 0xd1, 0xdc, 0xda, 0xac, 0x55,
            0xf3, 0x8e, 0xb3, 0x70, 0x33, 0x31, 0x4b, 0xbc,
            0x95, 0x33, 0x0c, 0x69, 0xad, 0x99, 0x9e, 0xec,
            0x75, 0xf0, 0x5f, 0x58, 0xd0, 0x89, 0x06, 0x09,
        };

        const x0 = Fp.fromBytes(&x0_bytes);
        const x1 = Fp.fromBytes(&x1_bytes);
        const y0 = Fp.fromBytes(&y0_bytes);
        const y1 = Fp.fromBytes(&y1_bytes);

        return G2Point.fromCoords(
            Fp2.init(x0, x1),
            Fp2.init(y0, y1),
        );
    }

    pub fn isIdentity(self: G2Point) bool {
        return self.infinity;
    }

    pub fn eql(self: G2Point, other: G2Point) bool {
        if (self.infinity and other.infinity) return true;
        if (self.infinity or other.infinity) return false;
        return self.x.eql(other.x) and self.y.eql(other.y);
    }

    pub fn neg(self: G2Point) G2Point {
        if (self.infinity) return self;
        return .{
            .x = self.x,
            .y = self.y.neg(),
            .infinity = false,
        };
    }

    pub fn add(self: G2Point, other: G2Point) G2Point {
        if (self.infinity) return other;
        if (other.infinity) return self;

        if (self.x.eql(other.x)) {
            if (self.y.eql(other.y.neg())) {
                return G2Point.identity();
            }
            return self.double();
        }

        // Point addition formula
        const slope_num = other.y.sub(self.y);
        const slope_den = other.x.sub(self.x);
        const slope = slope_num.mul(slope_den.inverse() orelse return G2Point.identity());

        const x3 = slope.square().sub(self.x).sub(other.x);
        const y3 = slope.mul(self.x.sub(x3)).sub(self.y);

        return .{ .x = x3, .y = y3, .infinity = false };
    }

    pub fn double(self: G2Point) G2Point {
        if (self.infinity) return self;
        if (self.y.isZero()) return G2Point.identity();

        // Point doubling formula: λ = (3x²)/(2y)
        const x_sq = self.x.square();
        const three_x_sq = x_sq.add(x_sq).add(x_sq);
        const two_y = self.y.add(self.y);
        const slope = three_x_sq.mul(two_y.inverse() orelse return G2Point.identity());

        const x3 = slope.square().sub(self.x).sub(self.x);
        const y3 = slope.mul(self.x.sub(x3)).sub(self.y);

        return .{ .x = x3, .y = y3, .infinity = false };
    }

    /// Scalar multiplication using double-and-add
    /// Computes [scalar] * self
    /// Processes bits from most significant to least significant
    pub fn scalarMul(self: G2Point, scalar: BN254Scalar) G2Point {
        if (self.isIdentity()) return G2Point.identity();
        if (scalar.isZero()) return G2Point.identity();

        var result = G2Point.identity();
        var started = false;

        // Convert scalar from Montgomery form to get actual value
        const normal_scalar = scalar.fromMontgomery();

        // Process each limb of the scalar from most significant to least significant
        // limbs[3] is most significant, limbs[0] is least significant
        var limb_idx: usize = 4;
        while (limb_idx > 0) {
            limb_idx -= 1;
            const limb = normal_scalar.limbs[limb_idx];

            // Process bits from most significant to least significant
            var bit_idx: u7 = 64;
            while (bit_idx > 0) {
                bit_idx -= 1;
                // Always double (unless we haven't started yet)
                if (started) {
                    result = result.double();
                }

                const bit = (limb >> @as(u6, @intCast(bit_idx))) & 1;
                if (bit == 1) {
                    if (!started) {
                        result = self;
                        started = true;
                    } else {
                        result = result.add(self);
                    }
                }
            }
        }

        return result;
    }

    /// Scalar multiplication with a u64 scalar (convenience method)
    pub fn scalarMulU64(self: G2Point, scalar: u64) G2Point {
        return self.scalarMul(BN254Scalar.fromU64(scalar));
    }
};

// ============================================================================
// G2 Homogeneous Projective (for Miller loop)
// ============================================================================

/// G2 point in homogeneous projective coordinates (x, y, z) where x/z, y/z are affine
/// Used in Miller loop to avoid expensive field inversions
const G2HomProjective = struct {
    x: Fp2,
    y: Fp2,
    z: Fp2,

    /// Convert from affine to projective (z = 1)
    fn fromAffine(p: G2Point) G2HomProjective {
        if (p.infinity) {
            return .{ .x = Fp2.zero(), .y = Fp2.one(), .z = Fp2.zero() };
        }
        return .{ .x = p.x, .y = p.y, .z = Fp2.one() };
    }

    /// Doubling step in projective coordinates
    /// Returns new point R = 2T and line coefficients for D-type twist
    fn double_in_place(self: *G2HomProjective, two_inv: Fp) EllCoeff {
        // Formula from arkworks bn254 g2.rs
        // a = x * y / 2
        var a = self.x.mul(self.y);
        a = fp2ScalarMul(a, two_inv);

        const b = self.y.square(); // b = y²
        const c = self.z.square(); // c = z²

        // e = COEFF_B * (c + c + c) = 3*B*c where B = 3/(9+u)
        // For BN254 D-type twist: COEFF_B = 3/(ξ) where ξ = 9+u
        // We compute e = 3 * COEFF_B * c = 3 * 3/(9+u) * c = 9*c/(9+u)
        // Actually: e = COEFF_B * (c.double() + c) = COEFF_B * 3c
        const three_c = c.add(c).add(c);
        const e = mulByTwistB(three_c);

        // f = e + e + e = 3e
        const f = e.add(e).add(e);

        // g = (b + f) / 2
        var g = b.add(f);
        g = fp2ScalarMul(g, two_inv);

        // h = (y + z)² - (b + c)
        const h = self.y.add(self.z).square().sub(b.add(c));

        // i = e - b
        const i = e.sub(b);

        // j = x²
        const j = self.x.square();

        // e_square = e²
        const e_square = e.square();

        // New point:
        // x' = a * (b - f)
        self.x = a.mul(b.sub(f));
        // y' = g² - 3*e²
        self.y = g.square().sub(e_square.add(e_square).add(e_square));
        // z' = b * h
        self.z = b.mul(h);

        // Line coefficients for D-type twist: (-h, 3j, i)
        return EllCoeff{
            .c0 = h.neg(),
            .c1 = j.add(j).add(j), // 3j
            .c2 = i,
        };
    }

    /// Addition step in projective coordinates
    /// Returns new point R = T + Q and line coefficients for D-type twist
    fn add_in_place(self: *G2HomProjective, q: G2Point) EllCoeff {
        // Formula from arkworks bn254 g2.rs
        // theta = y - q.y * z
        const theta = self.y.sub(q.y.mul(self.z));
        // lambda = x - q.x * z
        const lambda = self.x.sub(q.x.mul(self.z));

        const c = theta.square();
        const d = lambda.square();
        const e = lambda.mul(d);
        const f = self.z.mul(c);
        const g = self.x.mul(d);
        const h = e.add(f).sub(g.add(g));

        // New point:
        // x' = lambda * h
        self.x = lambda.mul(h);
        // y' = theta * (g - h) - e * y
        self.y = theta.mul(g.sub(h)).sub(e.mul(self.y));
        // z' = z * e
        self.z = self.z.mul(e);

        // j = theta * q.x - lambda * q.y
        const jay = theta.mul(q.x).sub(lambda.mul(q.y));

        // Line coefficients for D-type twist: (lambda, -theta, j)
        return EllCoeff{
            .c0 = lambda,
            .c1 = theta.neg(),
            .c2 = jay,
        };
    }
};

/// Line coefficients for pairing computation (arkworks format)
/// For D-type twist: (c0, c1, c2) evaluated at P gives sparse Fp12 element
const EllCoeff = struct {
    c0: Fp2,
    c1: Fp2,
    c2: Fp2,
};

/// COEFF_B for BN254 G2 twist curve: y² = x³ + B where B = 3/(9+u)
/// This is stored as: (b0, b1) where B = b0 + b1*u
/// From arkworks: B = (19485874751759354771024239261021720505790618469301721065564631296452457478373,
///                     266929791119991161246907387137283842545076965332900288569378510910307636690)
fn twistB() Fp2 {
    // These are raw standard form values, convert to Montgomery form
    const b0_limbs: [4]u64 = .{ 0x3267e6dc24a138e5, 0xb5b4c5e559dbefa3, 0x81be18991be06ac3, 0x2b149d40ceb8aaae };
    const b1_limbs: [4]u64 = .{ 0xe4a2bd0685c315d2, 0xa74fa084e52d1852, 0xcd2cafadeed8fdf4, 0x009713b03af0fed4 };
    const b0_raw = Fp{ .limbs = b0_limbs };
    const b1_raw = Fp{ .limbs = b1_limbs };
    // Convert from standard form to Montgomery form
    return Fp2.init(b0_raw.toMontgomery(), b1_raw.toMontgomery());
}

/// Multiply Fp2 element by twist curve coefficient B
fn mulByTwistB(a: Fp2) Fp2 {
    return a.mul(twistB());
}

/// TWIST_MUL_BY_Q_X = (u+9)^((p-1)/3) for Frobenius on G2
fn twistMulByQX() Fp2 {
    // From arkworks bn254:
    // c0 = 21575463638280843010398324269430826099269044274347216827212613867836435027261
    // c1 = 10307601595873709700152284273816112264069230130616436755625194854815875713954
    const c0_limbs: [4]u64 = .{ 0x99e39557176f553d, 0xb78cc310c2c3330c, 0x4c0bec3cf559b143, 0x2fb347984f7911f7 };
    const c1_limbs: [4]u64 = .{ 0x1665d51c640fcba2, 0x32ae2a1d0b7c9dce, 0x4ba4cc8bd75a0794, 0x16c9e55061ebae20 };
    const c0 = (Fp{ .limbs = c0_limbs }).toMontgomery();
    const c1 = (Fp{ .limbs = c1_limbs }).toMontgomery();
    return Fp2.init(c0, c1);
}

/// TWIST_MUL_BY_Q_Y = (u+9)^((p-1)/2) for Frobenius on G2
fn twistMulByQY() Fp2 {
    // From arkworks bn254:
    // c0 = 2821565182194536844548159561693502659359617185244120367078079554186484126554
    // c1 = 3505843767911556378687030309984248845540243509899259641013678093033130930403
    const c0_limbs: [4]u64 = .{ 0xdc54014671a0135a, 0xdbaae0eda9c95998, 0xdc5ec698b6e2f9b9, 0x063cf305489af5dc };
    const c1_limbs: [4]u64 = .{ 0x82d37f632623b0e3, 0x21807dc98fa25bd2, 0x0704b5a7ec796f2b, 0x07c03cbcac41049a };
    const c0 = (Fp{ .limbs = c0_limbs }).toMontgomery();
    const c1 = (Fp{ .limbs = c1_limbs }).toMontgomery();
    return Fp2.init(c0, c1);
}

/// Frobenius endomorphism on G2 (multiply by char)
/// π: (x, y) → (x^p * TWIST_MUL_BY_Q_X, y^p * TWIST_MUL_BY_Q_Y)
fn mulByChar(p_pt: G2Point) G2Point {
    if (p_pt.infinity) return p_pt;

    // x^p = conjugate(x) for Fp2, then multiply by coefficient
    var x_new = p_pt.x.conjugate();
    x_new = x_new.mul(twistMulByQX());

    // y^p = conjugate(y) for Fp2, then multiply by coefficient
    var y_new = p_pt.y.conjugate();
    y_new = y_new.mul(twistMulByQY());

    return G2Point{ .x = x_new, .y = y_new, .infinity = false };
}

// ============================================================================
// Fp6 sparse multiplication for pairing
// ============================================================================

/// Multiply Fp6 by sparse element (c0, c1, 0) - matches arkworks mul_by_01
fn fp6MulBy01(f: Fp6, c0: Fp2, c1: Fp2) Fp6 {
    // Following arkworks fp6_3over2.rs mul_by_01 exactly
    const a_a = f.c0.mul(c0);
    const b_b = f.c1.mul(c1);

    // t1 = c1 * (f.c1 + f.c2) - b_b, then *= ξ, then += a_a
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
fn fp6MulBy1(f: Fp6, c1: Fp2) Fp6 {
    // Sparse multiplication where only c1 is non-zero
    // (f.c0 + f.c1*v + f.c2*v²) * (c1*v) = f.c0*c1*v + f.c1*c1*v² + f.c2*c1*v³
    //                                    = f.c0*c1*v + f.c1*c1*v² + f.c2*c1*ξ
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
/// The sparse element is: c0 + c3*w + c4*v*w where w²=v
fn fp12MulBy034(f: Fp12, c0: Fp2, c3: Fp2, c4: Fp2) Fp12 {
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

// ============================================================================
// Pairing Operations
// ============================================================================

/// Result of a pairing computation (element of GT = Fp12)
pub const PairingResult = Fp12;

/// G1 Point for pairing operations (coordinates in Fp, the base field)
pub const G1PointFp = struct {
    x: Fp,
    y: Fp,
    infinity: bool,

    pub fn identity() G1PointFp {
        return .{ .x = Fp.zero(), .y = Fp.one(), .infinity = true };
    }

    pub fn neg(self: G1PointFp) G1PointFp {
        if (self.infinity) return self;
        return .{ .x = self.x, .y = self.y.neg(), .infinity = false };
    }
};

/// G1 Point from MSM (uses scalar field, for scalar multiplication)
/// Note: For pairing operations, convert to G1PointFp
/// IMPORTANT: G1 point COORDINATES are in the BASE FIELD Fp, but scalars are in Fr
pub const G1Point = @import("../msm/mod.zig").AffinePoint(BN254Scalar);

/// G1 Point in base field (proper representation for curve points)
/// Use this for creating G1 points for pairing operations
pub const G1PointInFp = @import("../msm/mod.zig").AffinePoint(Fp);

/// Convert G1Point (scalar field coords) to G1PointFp (base field coords)
/// G1 point coordinates are conceptually raw integer values that should be
/// in Montgomery form for either field. Since G1Point uses BN254Scalar (Fr),
/// we need to:
/// 1. Convert from Fr Montgomery form to raw value
/// 2. Convert from raw value to Fp Montgomery form
fn g1ToFp(p: G1Point) G1PointFp {
    if (p.infinity) {
        return G1PointFp.identity();
    }

    // Convert x from Fr Montgomery to raw, then to Fp Montgomery
    const x_raw = p.x.fromMontgomery();
    var x_fp_tmp = Fp{ .limbs = x_raw.limbs };
    const x_fp = x_fp_tmp.toMontgomery();

    // Convert y from Fr Montgomery to raw, then to Fp Montgomery
    const y_raw = p.y.fromMontgomery();
    var y_fp_tmp = Fp{ .limbs = y_raw.limbs };
    const y_fp = y_fp_tmp.toMontgomery();

    return .{
        .x = x_fp,
        .y = y_fp,
        .infinity = false,
    };
}

/// Compute the optimal ate pairing: e(P, Q) where P ∈ G1, Q ∈ G2
///
/// The pairing consists of two parts:
/// 1. Miller loop: Compute f_{6x+2,Q}(P)
/// 2. Final exponentiation: f^((p^12-1)/r)
///
/// Uses arkworks-compatible projective Miller loop for Jolt compatibility
pub fn pairing(p: G1Point, q: G2Point) PairingResult {
    if (p.infinity or q.infinity) {
        return Fp12.one();
    }

    // Convert G1 point from Fr to Fp representation
    const p_fp = g1ToFp(p);

    // Miller loop using arkworks-style projective coordinates
    const f = millerLoopArkworks(p_fp, q);

    // Final exponentiation
    return finalExponentiation(f);
}

/// Pairing function that takes G1 point directly in base field representation
/// Use this when you have proper Fp coordinates (not Fr)
/// Uses arkworks-compatible projective Miller loop for Jolt compatibility
pub fn pairingFp(p: G1PointFp, q: G2Point) PairingResult {
    if (p.infinity or q.infinity) {
        return Fp12.one();
    }

    // Miller loop using arkworks-style projective coordinates
    const f = millerLoopArkworks(p, q);

    // Final exponentiation
    return finalExponentiation(f);
}

/// Pseudobinary representation of the loop length 6·X+2 of the optimal ate pairing over BN254.
/// From arkworks-rs/curves bn254/src/curves/mod.rs (ATE_LOOP_COUNT).
/// This is the NAF representation of 6*x+2 where x = 4965661367192848881.
/// Array has 65 elements, processed from LSB (index 0) to MSB (index 64).
const ATE_LOOP_COUNT: [65]i2 = .{
    0, 0, 0, 1, 0, 1, 0, -1, 0, 0, -1, 0, 0, 0, 1, 0,
    0, -1, 0, -1, 0, 0, 0, 1, 0, -1, 0, 0, 0, 0, -1, 0,
    0, 1, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, -1, 0, 0, -1,
    0, 1, 0, -1, 0, 0, 0, -1, 0, -1, 0, 0, 0, 1, 0, 1,
    1,
};

/// Line coefficients R0, R1 matching gnark-crypto's affine representation
/// R0 = λ (the slope)
/// R1 = λ·x_Q - y_Q (used for efficient evaluation)
const LineCoeffs = struct {
    r0: Fp2, // λ
    r1: Fp2, // λ·x_Q - y_Q
};

/// Result of doubling/addition step: new point and line coefficients
const MillerStepResult = struct {
    point: G2Point,
    coeffs: LineCoeffs,
};

/// Sparse line evaluation result matching gnark-crypto's (1, 0, 0, c3, c4, 0) format
/// where positions are: 0=1, 1=v, 2=v², 3=w, 4=vw, 5=v²w
/// This represents 1 + c3·w + c4·vw in Fp12
const SparseLineEval = struct {
    c3: Fp2, // Coefficient of w (C1.c0)
    c4: Fp2, // Coefficient of vw (C1.c1)
};

/// Evaluate line function at point P = (x_P, y_P)
/// Following gnark-crypto's affine approach:
/// - Precompute xNegOverY = -x_P/y_P and yInv = 1/y_P
/// - c3 = R0 * xNegOverY = λ * (-x_P/y_P)
/// - c4 = R1 * yInv = (λ*x_Q - y_Q) * (1/y_P)
fn evaluateLineSparse(coeffs: LineCoeffs, x_neg_over_y: Fp, y_inv: Fp) SparseLineEval {
    // c3 = R0 * xNegOverY (Fp2 * Fp -> Fp2)
    const c3 = fp2ScalarMul(coeffs.r0, x_neg_over_y);

    // c4 = R1 * yInv (Fp2 * Fp -> Fp2)
    const c4 = fp2ScalarMul(coeffs.r1, y_inv);

    return SparseLineEval{
        .c3 = c3,
        .c4 = c4,
    };
}

/// Sparse multiplication of Fp12 by sparse element (1, 0, 0, c3, c4, 0)
/// This is gnark-crypto's MulBy34 pattern: multiply by 1 + c3·w + c4·vw
///
/// In the Fp12 tower (Fp6[w]/(w² - v)):
/// The sparse element is:
/// - c0 = 1 ∈ Fp6 (just the constant 1)
/// - c1 = c3 + c4·v ∈ Fp6
///
/// For z = a·b where a ∈ Fp12, b = 1 + c3·w + c4·vw:
/// Let a = a0 + a1·w, then:
/// z = a0 + a1·w + (a0 + a1·w)(c3·w + c4·vw)
///   = a0 + a0·c3·w + a0·c4·vw + a1·w + a1·c3·w² + a1·c4·vw²
///   = a0 + a0·c3·w + a0·c4·vw + a1·w + a1·c3·v + a1·c4·v²  (using w² = v)
///   = (a0 + a1·c3·v + a1·c4·v²) + (a1 + a0·c3 + a0·c4·v)·w
fn sparseMulFp12(a: Fp12, sparse: SparseLineEval) Fp12 {
    // Construct the sparse element in Fp12 form
    // b.c0 = 1 ∈ Fp6
    // b.c1 = c3 + c4·v ∈ Fp6
    const b = Fp12{
        .c0 = Fp6.one(),
        .c1 = Fp6{
            .c0 = sparse.c3,
            .c1 = sparse.c4,
            .c2 = Fp2.zero(),
        },
    };

    return a.mul(b);
}

/// Evaluate line function at point P = (p_x, p_y)
/// Returns the Fp12 element representing the line evaluation
fn evaluateLine(coeffs: LineCoeffs, p_x: Fp, p_y: Fp) Fp12 {
    // Precompute values following gnark-crypto:
    // xNegOverY = -x_P / y_P
    // yInv = 1 / y_P
    const y_inv = p_y.inverse() orelse return Fp12.one();
    const x_neg_over_y = p_x.neg().mul(y_inv);

    const sparse = evaluateLineSparse(coeffs, x_neg_over_y, y_inv);

    // Convert sparse (1, 0, 0, c3, c4, 0) to full Fp12
    // c0 = 1 ∈ Fp6
    // c1 = c3 + c4·v ∈ Fp6
    return Fp12{
        .c0 = Fp6.one(),
        .c1 = Fp6{
            .c0 = sparse.c3,
            .c1 = sparse.c4,
            .c2 = Fp2.zero(),
        },
    };
}

/// Doubling step in Miller loop (affine coordinates)
/// Returns the new point T = 2*T and line coefficients R0, R1
/// Matching gnark-crypto's affine doubleStep:
///   R0 = λ
///   R1 = λ·x - y
fn doublingStep(t: G2Point) MillerStepResult {
    if (t.infinity or t.y.isZero()) {
        return .{
            .point = G2Point.identity(),
            .coeffs = .{
                .r0 = Fp2.zero(),
                .r1 = Fp2.zero(),
            },
        };
    }

    // Compute λ = 3x² / 2y (the slope of the tangent line)
    const x_sq = t.x.square();
    const three_x_sq = x_sq.add(x_sq).add(x_sq);
    const two_y = t.y.add(t.y);
    const lambda = three_x_sq.mul(two_y.inverse() orelse return .{
        .point = G2Point.identity(),
        .coeffs = .{ .r0 = Fp2.zero(), .r1 = Fp2.zero() },
    });

    // New point coordinates: x_r = λ² - 2x, y_r = λ(x - x_r) - y
    const x_r = lambda.square().sub(t.x).sub(t.x);
    const y_r = lambda.mul(t.x.sub(x_r)).sub(t.y);

    // Line coefficients following gnark-crypto:
    // R0 = λ
    // R1 = λ·x - y
    const r0 = lambda;
    const r1 = lambda.mul(t.x).sub(t.y);

    return .{
        .point = G2Point{ .x = x_r, .y = y_r, .infinity = false },
        .coeffs = .{ .r0 = r0, .r1 = r1 },
    };
}

/// Addition step in Miller loop (affine coordinates)
/// Returns the new point T = T + Q and line coefficients R0, R1
/// Matching gnark-crypto's affine addStep:
///   R0 = λ
///   R1 = λ·x_T - y_T
fn additionStep(t: G2Point, q: G2Point) MillerStepResult {
    if (t.infinity) {
        return .{
            .point = q,
            .coeffs = .{ .r0 = Fp2.zero(), .r1 = Fp2.zero() },
        };
    }
    if (q.infinity) {
        return .{
            .point = t,
            .coeffs = .{ .r0 = Fp2.zero(), .r1 = Fp2.zero() },
        };
    }

    // If points are equal, use doubling
    if (t.x.eql(q.x)) {
        if (t.y.eql(q.y)) {
            return doublingStep(t);
        }
        // t + (-t) = O
        return .{
            .point = G2Point.identity(),
            .coeffs = .{ .r0 = Fp2.zero(), .r1 = Fp2.zero() },
        };
    }

    // Compute λ = (q.y - t.y) / (q.x - t.x)
    const dy = q.y.sub(t.y);
    const dx = q.x.sub(t.x);
    const lambda = dy.mul(dx.inverse() orelse return .{
        .point = G2Point.identity(),
        .coeffs = .{ .r0 = Fp2.zero(), .r1 = Fp2.zero() },
    });

    // New point coordinates: x_r = λ² - x1 - x2, y_r = λ(x1 - x_r) - y1
    const x_r = lambda.square().sub(t.x).sub(q.x);
    const y_r = lambda.mul(t.x.sub(x_r)).sub(t.y);

    // Line coefficients following gnark-crypto:
    // R0 = λ
    // R1 = λ·x_T - y_T
    const r0 = lambda;
    const r1 = lambda.mul(t.x).sub(t.y);

    return .{
        .point = G2Point{ .x = x_r, .y = y_r, .infinity = false },
        .coeffs = .{ .r0 = r0, .r1 = r1 },
    };
}

/// Miller loop for optimal ate pairing on BN254
/// Computes f_{6x+2,Q}(P) where x is the BN254 curve parameter
fn millerLoop(p: G1PointFp, q: G2Point) Fp12 {
    if (p.infinity or q.infinity) {
        return Fp12.one();
    }

    // Initialize: f = 1, T = Q
    var f = Fp12.one();
    var t = q;

    // Main loop: iterate through bits of the ate loop parameter
    // Following arkworks: iterate from len-1 down to 1, check bit at i-1
    // This processes the array from MSB to LSB (high to low index in the array)
    // CRITICAL: Skip first squaring (arkworks skips when i == len-1)
    var i: usize = ATE_LOOP_COUNT.len - 1;
    while (i >= 1) : (i -= 1) {
        // Skip squaring on first iteration (matches arkworks behavior)
        if (i != ATE_LOOP_COUNT.len - 1) {
            f = f.square();
        }

        // Doubling step: T = 2T, multiply by line l_{T,T}(P)
        const dbl = doublingStep(t);
        t = dbl.point;
        const line_dbl = evaluateLine(dbl.coeffs, p.x, p.y);
        f = f.mul(line_dbl);

        // Addition step if bit at [i-1] is non-zero
        const bit = ATE_LOOP_COUNT[i - 1];
        if (bit == 1) {
            // f = f * l_{T,Q}(P), T = T + Q
            const add = additionStep(t, q);
            t = add.point;
            const line_add = evaluateLine(add.coeffs, p.x, p.y);
            f = f.mul(line_add);
        } else if (bit == -1) {
            // f = f * l_{T,-Q}(P), T = T - Q
            const neg_q = q.neg();
            const add = additionStep(t, neg_q);
            t = add.point;
            const line_add = evaluateLine(add.coeffs, p.x, p.y);
            f = f.mul(line_add);
        }
    }

    // For BN254 optimal ate, we need additional steps at the end:
    // f = f * l_{T, π(Q)}(P) * l_{T', -π²(Q)}(P)
    // where π is the Frobenius endomorphism
    // These additional lines correspond to the +2 part of 6x+2

    // Apply Frobenius to Q (π(Q) and π²(Q))
    // For BN254: π(x, y) = (x^p, y^p) which in Fp2 uses the Frobenius coefficients
    // This is simplified - full implementation needs the actual Frobenius coefficients
    const q1 = frobeniusG2(q);
    const add1 = additionStep(t, q1);
    t = add1.point;
    f = f.mul(evaluateLine(add1.coeffs, p.x, p.y));

    const q2 = frobeniusG2(frobeniusG2(q)).neg();
    const add2 = additionStep(t, q2);
    f = f.mul(evaluateLine(add2.coeffs, p.x, p.y));

    return f;
}

/// X_IS_NEGATIVE flag for BN254 - the curve parameter x is positive
const X_IS_NEGATIVE: bool = false;

/// Miller loop using arkworks-style projective coordinates
/// This is the correct implementation matching arkworks exactly
fn millerLoopArkworks(p: G1PointFp, q: G2Point) Fp12 {
    if (p.infinity or q.infinity) {
        return Fp12.one();
    }

    // Precompute two_inv = 1/2
    const two_inv = Fp.fromU64(2).inverse() orelse return Fp12.one();

    // Initialize projective point R = Q
    var r = G2HomProjective.fromAffine(q);
    const neg_q = q.neg();

    var f = Fp12.one();

    // Main loop: iterate from MSB-1 down to 0
    // arkworks iterates from (len-1) down to 1, checking bit at (i-1)
    var idx: usize = ATE_LOOP_COUNT.len - 1;
    while (idx >= 1) : (idx -= 1) {
        // Square f unless it's the first iteration
        if (idx != ATE_LOOP_COUNT.len - 1) {
            f = f.square();
        }

        // Doubling step: R = 2R, get line coefficients
        const coeffs_dbl = r.double_in_place(two_inv);
        // Evaluate line: c0 *= y_P, c1 *= x_P for D-type twist
        const c0_eval = fp2ScalarMul(coeffs_dbl.c0, p.y);
        const c1_eval = fp2ScalarMul(coeffs_dbl.c1, p.x);
        f = fp12MulBy034(f, c0_eval, c1_eval, coeffs_dbl.c2);

        // Addition step if bit is non-zero
        const bit = ATE_LOOP_COUNT[idx - 1];
        if (bit == 1) {
            const coeffs_add = r.add_in_place(q);
            const c0_add = fp2ScalarMul(coeffs_add.c0, p.y);
            const c1_add = fp2ScalarMul(coeffs_add.c1, p.x);
            f = fp12MulBy034(f, c0_add, c1_add, coeffs_add.c2);
        } else if (bit == -1) {
            const coeffs_add = r.add_in_place(neg_q);
            const c0_add = fp2ScalarMul(coeffs_add.c0, p.y);
            const c1_add = fp2ScalarMul(coeffs_add.c1, p.x);
            f = fp12MulBy034(f, c0_add, c1_add, coeffs_add.c2);
        }
    }

    // If X is negative, conjugate the result (cyclotomic inverse)
    if (X_IS_NEGATIVE) {
        f = f.conjugate();
    }

    // Final Frobenius steps: add π(Q) and -π²(Q)
    // First: R + π(Q)
    const q1 = mulByChar(q);
    const coeffs_q1 = r.add_in_place(q1);
    const c0_q1 = fp2ScalarMul(coeffs_q1.c0, p.y);
    const c1_q1 = fp2ScalarMul(coeffs_q1.c1, p.x);
    f = fp12MulBy034(f, c0_q1, c1_q1, coeffs_q1.c2);

    // Second: R + (-π²(Q))
    var q2 = mulByChar(q1);
    q2.y = q2.y.neg(); // Negate y coordinate
    const coeffs_q2 = r.add_in_place(q2);
    const c0_q2 = fp2ScalarMul(coeffs_q2.c0, p.y);
    const c1_q2 = fp2ScalarMul(coeffs_q2.c1, p.x);
    f = fp12MulBy034(f, c0_q2, c1_q2, coeffs_q2.c2);

    return f;
}

/// Apply Frobenius endomorphism to G2 point
/// π: (x, y) → (x^p · γ_{1,2}, y^p · γ_{1,3})
/// where x^p = conjugate(x), y^p = conjugate(y) in Fp2
/// and γ_{1,2} = ξ^{(p-1)/3}, γ_{1,3} = ξ^{(p-1)/2}
fn frobeniusG2(p: G2Point) G2Point {
    if (p.infinity) return p;

    // The Frobenius on Fp2 is conjugation: (a + bu) → (a - bu) = (a + bu)^p
    // Then we multiply by the twist factors (Frobenius coefficients)
    // For BN254: π(x, y) = (conjugate(x) * γ_{1,2}, conjugate(y) * γ_{1,3})

    const x_frob = p.x.conjugate().mul(gamma12());
    const y_frob = p.y.conjugate().mul(gamma13());

    return G2Point{
        .x = x_frob,
        .y = y_frob,
        .infinity = false,
    };
}

/// Final exponentiation: f^((p^12-1)/r)
/// Split into easy part and hard part
fn finalExponentiation(f: Fp12) Fp12 {
    if (f.eql(Fp12.zero())) {
        return Fp12.one();
    }

    // Easy part: f^((p^6-1)(p^2+1))
    // This can be computed using conjugation and Frobenius
    const easy = easyPartExponentiation(f);

    // Hard part: f^((p^4-p^2+1)/r)
    // This requires more complex computation
    return hardPartExponentiation(easy);
}

fn easyPartExponentiation(f: Fp12) Fp12 {
    // f^(p^6-1) = conj(f) * f^(-1) (using the fact that f^(p^6) = conj(f))
    const f_conj = f.conjugate();
    const f_inv = f.inverse() orelse return Fp12.one();
    const easy1 = f_conj.mul(f_inv);

    // easy1^(p^2+1) = easy1^(p^2) * easy1
    // Using Frobenius^2 for efficiency
    const easy1_p2 = easy1.frobenius2();
    return easy1_p2.mul(easy1);
}

/// BN254 curve parameter x = 4965661367192848881
/// Used in hard part of final exponentiation
const BN_X: u64 = 4965661367192848881;

fn hardPartExponentiation(m: Fp12) Fp12 {
    // The hard part is m^((p^4 - p^2 + 1)/r)
    // Using the optimized formula from ziskos (final_exp.rs)
    //
    // Compute:
    //   y1 = m^p · m^{p²} · m^{p³}
    //   y2 = m̄ (conjugate)
    //   y3 = (m^{x²})^{p²}
    //   y4 = conj((m^x)^p)
    //   y5 = conj(m^x · (m^{x²})^p)
    //   y6 = conj(m^{x²})
    //   y7 = conj(m^{x³} · (m^{x³})^p)
    //
    // Then compute y1·y2²·y3⁶·y4¹²·y5¹⁸·y6³⁰·y7³⁶ using an optimized addition chain

    // Compute powers of m by x
    const mx = expByX(m);
    const mxx = expByX(mx);
    const mxxx = expByX(mxx);

    // Compute Frobenius powers
    const mp = m.frobenius();
    const mpp = m.frobenius2();
    const mppp = m.frobenius3();
    const mxp = mx.frobenius();
    const mxxp = mxx.frobenius();
    const mxxxp = mxxx.frobenius();
    const mxxpp = mxx.frobenius2();

    // y1 = m^p · m^{p²} · m^{p³}
    var y1 = mp.mul(mpp);
    y1 = y1.mul(mppp);

    // y2 = m̄ (conjugate)
    const y2 = m.conjugate();

    // y3 = (m^{x²})^{p²} (already computed as mxxpp)

    // y4 = conj((m^x)^p)
    const y4 = mxp.conjugate();

    // y5 = conj(m^x · (m^{x²})^p)
    var y5 = mx.mul(mxxp);
    y5 = y5.conjugate();

    // y6 = conj(m^{x²})
    const y6 = mxx.conjugate();

    // y7 = conj(m^{x³} · (m^{x³})^p)
    var y7 = mxxx.mul(mxxxp);
    y7 = y7.conjugate();

    // Compute y1·y2²·y3⁶·y4¹²·y5¹⁸·y6³⁰·y7³⁶ using the optimized addition chain from ziskos:
    //
    // T11 = y7² · y5 · y6
    var t11 = y7.square();
    t11 = t11.mul(y5);
    t11 = t11.mul(y6);

    // T21 = T11 · y4 · y6
    var t21 = t11.mul(y4);
    t21 = t21.mul(y6);

    // T12 = T11 · y3 (y3 = mxxpp)
    const t12 = t11.mul(mxxpp);

    // T22 = T21² · T12
    var t22 = t21.square();
    t22 = t22.mul(t12);

    // T23 = T22²
    const t23 = t22.square();

    // T24 = T23 · y1
    const t24 = t23.mul(y1);

    // T13 = T23 · y2
    const t13 = t23.mul(y2);

    // T14 = T13² · T24
    var t14 = t13.square();
    t14 = t14.mul(t24);

    return t14;
}

/// Compute f^x where x is the BN254 curve parameter
fn expByX(f: Fp12) Fp12 {
    var result = Fp12.one();
    var base = f;
    var exp = BN_X;

    while (exp > 0) {
        if (exp & 1 == 1) {
            result = result.mul(base);
        }
        base = base.square();
        exp >>= 1;
    }

    return result;
}

/// Frobenius endomorphism on Fp12
/// Computes f^p using the tower structure
fn frobeniusFp12(f: Fp12) Fp12 {
    // Use the proper Frobenius implementation with coefficients
    return f.frobenius();
}

/// Pairing input pair type
pub const PairingInput = struct {
    p: G1Point,
    q: G2Point,
};

/// Multi-pairing: product of pairings e(P1,Q1) * e(P2,Q2) * ...
/// More efficient than computing individual pairings
pub fn multiPairing(pairs: []const PairingInput) PairingResult {
    var result = Fp12.one();

    for (pairs) |pair| {
        // In a real implementation, we'd use a shared Miller loop
        const single = pairing(pair.p, pair.q);
        result = result.mul(single);
    }

    return result;
}

/// Check if e(P1, Q1) == e(P2, Q2)
/// Useful for verifying KZG proofs
/// Note: Uses G1Point with Fr coordinates, converts internally
pub fn pairingCheck(p1: G1Point, q1: G2Point, p2: G1Point, q2: G2Point) bool {
    // Instead of checking e(P1,Q1) == e(P2,Q2), we check e(P1,Q1) * e(-P2,Q2) == 1
    // Create negated p2
    const p2_neg = G1Point{
        .x = p2.x,
        .y = p2.y.neg(),
        .infinity = p2.infinity,
    };
    const pairs = [_]PairingInput{
        .{ .p = p1, .q = q1 },
        .{ .p = p2_neg, .q = q2 },
    };
    const result = multiPairing(&pairs);
    return result.isOne();
}

/// Pairing input pair type with proper Fp coordinates
pub const PairingInputFp = struct {
    p: G1PointFp,
    q: G2Point,
};

/// Multi-pairing with Fp coordinates: product of pairings e(P1,Q1) * e(P2,Q2) * ...
pub fn multiPairingFp(pairs: []const PairingInputFp) PairingResult {
    var result = Fp12.one();
    for (pairs) |pair| {
        const single = pairingFp(pair.p, pair.q);
        result = result.mul(single);
    }
    return result;
}

/// Check if e(P1, Q1) == e(P2, Q2) with proper Fp coordinates
/// Useful for verifying KZG proofs
pub fn pairingCheckFp(p1: G1PointFp, q1: G2Point, p2: G1PointFp, q2: G2Point) bool {
    // Instead of checking e(P1,Q1) == e(P2,Q2), we check e(P1,Q1) * e(-P2,Q2) == 1
    const p2_neg = p2.neg();
    const pairs = [_]PairingInputFp{
        .{ .p = p1, .q = q1 },
        .{ .p = p2_neg, .q = q2 },
    };
    const result = multiPairingFp(&pairs);
    return result.isOne();
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
    // (3 + 4i)(1 + 2i) = 3 + 6i + 4i + 8i² = 3 + 10i - 8 = -5 + 10i
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

test "G2 point operations" {
    const g = G2Point.generator();
    const identity = G2Point.identity();

    // G + O = G
    const sum1 = g.add(identity);
    try std.testing.expect(sum1.eql(g));

    // G + (-G) = O
    const neg_g = g.neg();
    const sum2 = g.add(neg_g);
    try std.testing.expect(sum2.isIdentity());
}

test "G2 scalar multiplication" {
    const g = G2Point.generator();

    // [0]G = O
    const zero_times_g = g.scalarMul(BN254Scalar.zero());
    try std.testing.expect(zero_times_g.isIdentity());

    // [1]G = G
    const one_times_g = g.scalarMul(BN254Scalar.one());
    try std.testing.expect(one_times_g.eql(g));

    // [2]G = G + G = double(G)
    const two_times_g = g.scalarMul(BN254Scalar.fromU64(2));
    const g_doubled = g.double();
    try std.testing.expect(two_times_g.eql(g_doubled));

    // [3]G = G + G + G = double(G) + G
    const three_times_g = g.scalarMul(BN254Scalar.fromU64(3));
    const g_tripled = g_doubled.add(g);
    try std.testing.expect(three_times_g.eql(g_tripled));

    // Convenience method [5]G
    const five_times_g = g.scalarMulU64(5);
    const expected = g.scalarMul(BN254Scalar.fromU64(5));
    try std.testing.expect(five_times_g.eql(expected));
}

test "Miller loop doubling step" {
    // Test that doubling step produces valid line coefficients
    const q = G2Point.generator();
    const result = doublingStep(q);

    // The resulting point should not be identity (unless generator is special)
    // Line coefficients (R0, R1) should have at least one non-zero component
    try std.testing.expect(!result.coeffs.r0.isZero() or !result.coeffs.r1.isZero());
}

test "Miller loop addition step" {
    // Test addition step
    const q = G2Point.generator();
    const q2 = q.double();

    const result = additionStep(q, q2);

    // Should produce a valid point
    if (!result.point.isIdentity()) {
        try std.testing.expect(!result.coeffs.r0.isZero() or !result.coeffs.r1.isZero());
    }
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

    const a_frob = frobeniusFp12(a);

    // Frobenius should not return the same element (in general)
    // but applying it p times should return the original (for elements in Fp)
    // For our test, just check it's well-defined
    try std.testing.expect(!a_frob.c0.c0.c0.isZero() or !a_frob.c0.c0.c1.isZero());
}

test "expByX exponentiation" {
    // Test exponentiation by curve parameter
    const one = Fp12.one();
    const one_x = expByX(one);

    // 1^x = 1
    try std.testing.expect(one_x.eql(Fp12.one()));
}

test "pairing with identity" {
    // e(O, Q) = 1 and e(P, O) = 1
    // Note: G1Point uses BN254Scalar (Fr) for MSM compatibility
    // The pairing function converts to Fp internally
    const g1 = G1Point{ .x = BN254Scalar.one(), .y = BN254Scalar.fromU64(2), .infinity = false };
    const g2 = G2Point.generator();
    const g1_identity = G1Point{ .x = BN254Scalar.zero(), .y = BN254Scalar.one(), .infinity = true };
    const g2_identity = G2Point.identity();

    const result1 = pairing(g1_identity, g2);
    const result2 = pairing(g1, g2_identity);

    try std.testing.expect(result1.isOne());
    try std.testing.expect(result2.isOne());
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

test "G2 scalar mul internal consistency" {
    // Verify G2 scalar multiplication produces correct results
    const g2 = G2Point.generator();

    // [2]G2 should equal G2 + G2
    const two_g2_by_add = g2.add(g2);
    const two_g2_by_double = g2.double();
    const two_g2_by_scalar = g2.scalarMul(BN254Scalar.fromU64(2));

    // All three should be equal
    try std.testing.expect(two_g2_by_add.eql(two_g2_by_double));
    try std.testing.expect(two_g2_by_double.eql(two_g2_by_scalar));
}

// Pairing bilinearity test: verifies e([2]P, Q) = e(P, Q)^2
// Fixed iteration 15: Corrected ξ from (1 + u) to (9 + u) and use proper Fp coordinates
test "pairing bilinearity in G1" {
    // G1 generator (1, 2) in base field Fp - valid point on BN254 curve: y^2 = x^3 + 3
    const g1 = G1PointFp{ .x = Fp.one(), .y = Fp.fromU64(2), .infinity = false };
    const g2 = G2Point.generator();

    // Compute e(G1, G2)
    const e_g1_g2 = pairingFp(g1, g2);
    const e_g1_g2_squared = e_g1_g2.mul(e_g1_g2);

    // Compute [2]G1 using point doubling in Fp
    const g1_doubled = G1PointInFp.generator().double();
    const e_2g1_g2 = pairingFp(G1PointFp{
        .x = g1_doubled.x,
        .y = g1_doubled.y,
        .infinity = g1_doubled.infinity,
    }, g2);

    try std.testing.expect(e_2g1_g2.eql(e_g1_g2_squared));
}

test "pairing bilinearity in G2" {
    // Test e(P, [2]Q) = e(P, Q)^2
    const g1 = G1PointFp{ .x = Fp.one(), .y = Fp.fromU64(2), .infinity = false };
    const g2 = G2Point.generator();

    // Compute e(G1, G2)
    const e_g1_g2 = pairingFp(g1, g2);
    const e_g1_g2_squared = e_g1_g2.mul(e_g1_g2);

    // Compute [2]G2 using point doubling
    const g2_doubled = g2.double();
    const e_g1_2g2 = pairingFp(g1, g2_doubled);

    try std.testing.expect(e_g1_2g2.eql(e_g1_g2_squared));
}

test "pairing identity" {
    // Test e(P, O) = 1 and e(O, Q) = 1
    const g1 = G1PointFp{ .x = Fp.one(), .y = Fp.fromU64(2), .infinity = false };
    const g2 = G2Point.generator();

    // e(P, O) = 1
    const e_g1_o = pairingFp(g1, G2Point.identity());
    try std.testing.expect(e_g1_o.isOne());

    // e(O, Q) = 1
    const e_o_g2 = pairingFp(G1PointFp.identity(), g2);
    try std.testing.expect(e_o_g2.isOne());
}

test "pairing non-degeneracy" {
    // Test e(P, Q) != 1 for non-identity P, Q
    const g1 = G1PointFp{ .x = Fp.one(), .y = Fp.fromU64(2), .infinity = false };
    const g2 = G2Point.generator();

    const e_g1_g2 = pairingFp(g1, g2);
    try std.testing.expect(!e_g1_g2.isOne());
}

test "pairing generator comparison with jolt" {
    // Compare e(G1_gen, G2_gen) with Jolt's result
    // Jolt: e(G1_gen, G2_gen) first 16 bytes: 95 0e 87 9d 73 63 1f 5e b5 78 85 89 eb 5f 7e f8
    const g1 = G1PointFp{ .x = Fp.one(), .y = Fp.fromU64(2), .infinity = false };
    const g2 = G2Point.generator();

    // Print G2 generator for comparison
    std.debug.print("\n=== Generator Comparison ===\n", .{});

    const g2_x_c0_std = g2.x.c0.fromMontgomery();
    const g2_x_c1_std = g2.x.c1.fromMontgomery();
    const g2_y_c0_std = g2.y.c0.fromMontgomery();
    const g2_y_c1_std = g2.y.c1.fromMontgomery();

    var g2_x_c0_bytes: [32]u8 = undefined;
    var g2_x_c1_bytes: [32]u8 = undefined;
    var g2_y_c0_bytes: [32]u8 = undefined;
    var g2_y_c1_bytes: [32]u8 = undefined;
    for (0..4) |i| {
        std.mem.writeInt(u64, g2_x_c0_bytes[i * 8 ..][0..8], g2_x_c0_std.limbs[i], .little);
        std.mem.writeInt(u64, g2_x_c1_bytes[i * 8 ..][0..8], g2_x_c1_std.limbs[i], .little);
        std.mem.writeInt(u64, g2_y_c0_bytes[i * 8 ..][0..8], g2_y_c0_std.limbs[i], .little);
        std.mem.writeInt(u64, g2_y_c1_bytes[i * 8 ..][0..8], g2_y_c1_std.limbs[i], .little);
    }
    std.debug.print("Zolt G2 generator:\n", .{});
    std.debug.print("  x.c0 first 16: {x}\n", .{g2_x_c0_bytes[0..16].*});
    std.debug.print("  x.c1 first 16: {x}\n", .{g2_x_c1_bytes[0..16].*});
    std.debug.print("  y.c0 first 16: {x}\n", .{g2_y_c0_bytes[0..16].*});
    std.debug.print("  y.c1 first 16: {x}\n", .{g2_y_c1_bytes[0..16].*});

    // Jolt G2 generator:
    // x.c0: ed f6 92 d9 5c bd de 46 dd da 5e f7 d4 22 43 67
    // x.c1: c2 12 f3 ae b7 85 e4 97 12 e7 a9 35 33 49 aa f1
    // y.c0: aa 7d fa 66 01 cc e6 4c 7b d3 43 0c 69 e7 d1 e3
    // y.c1: 5b 97 22 d1 dc da ac 55 f3 8e b3 70 33 31 4b bc
    const jolt_x_c0 = [_]u8{ 0xed, 0xf6, 0x92, 0xd9, 0x5c, 0xbd, 0xde, 0x46, 0xdd, 0xda, 0x5e, 0xf7, 0xd4, 0x22, 0x43, 0x67 };
    const jolt_x_c1 = [_]u8{ 0xc2, 0x12, 0xf3, 0xae, 0xb7, 0x85, 0xe4, 0x97, 0x12, 0xe7, 0xa9, 0x35, 0x33, 0x49, 0xaa, 0xf1 };
    const jolt_y_c0 = [_]u8{ 0xaa, 0x7d, 0xfa, 0x66, 0x01, 0xcc, 0xe6, 0x4c, 0x7b, 0xd3, 0x43, 0x0c, 0x69, 0xe7, 0xd1, 0xe3 };
    const jolt_y_c1 = [_]u8{ 0x5b, 0x97, 0x22, 0xd1, 0xdc, 0xda, 0xac, 0x55, 0xf3, 0x8e, 0xb3, 0x70, 0x33, 0x31, 0x4b, 0xbc };

    var g2_match = std.mem.eql(u8, g2_x_c0_bytes[0..16], &jolt_x_c0);
    g2_match = g2_match and std.mem.eql(u8, g2_x_c1_bytes[0..16], &jolt_x_c1);
    g2_match = g2_match and std.mem.eql(u8, g2_y_c0_bytes[0..16], &jolt_y_c0);
    g2_match = g2_match and std.mem.eql(u8, g2_y_c1_bytes[0..16], &jolt_y_c1);

    if (g2_match) {
        std.debug.print("*** G2 generator MATCHES Jolt! ***\n", .{});
    } else {
        std.debug.print("*** G2 generator MISMATCH ***\n", .{});
    }

    const e_g1_g2 = pairingFp(g1, g2);
    const bytes = e_g1_g2.toBytes();

    std.debug.print("\nZolt e(G1_gen, G2_gen) first 16 bytes: {x}\n", .{bytes[0..16].*});

    const jolt_bytes = [_]u8{ 0x95, 0x0e, 0x87, 0x9d, 0x73, 0x63, 0x1f, 0x5e, 0xb5, 0x78, 0x85, 0x89, 0xeb, 0x5f, 0x7e, 0xf8 };
    if (std.mem.eql(u8, bytes[0..16], &jolt_bytes)) {
        std.debug.print("*** Generator pairing MATCHES Jolt! ***\n", .{});
    } else {
        std.debug.print("*** Generator pairing MISMATCH ***\n", .{});
        std.debug.print("Expected (Jolt): {x}\n", .{jolt_bytes});
    }
}

test "pairingCheckFp basic" {
    // Test that e(P, Q) == e(P, Q) returns true
    const g1 = G1PointFp{ .x = Fp.one(), .y = Fp.fromU64(2), .infinity = false };
    const g2 = G2Point.generator();

    // e(P, Q) == e(P, Q) should be true
    try std.testing.expect(pairingCheckFp(g1, g2, g1, g2));
}

test "pairingCheckFp bilinearity" {
    // Test that e([2]P, Q) == e(P, [2]Q)
    // This is a consequence of bilinearity
    const g1 = G1PointFp{ .x = Fp.one(), .y = Fp.fromU64(2), .infinity = false };
    const g2 = G2Point.generator();

    // [2]G1
    const g1_doubled = G1PointInFp.generator().double();
    const g1_2 = G1PointFp{ .x = g1_doubled.x, .y = g1_doubled.y, .infinity = g1_doubled.infinity };

    // [2]G2
    const g2_2 = g2.double();

    // e([2]P, Q) == e(P, [2]Q)
    try std.testing.expect(pairingCheckFp(g1_2, g2, g1, g2_2));
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

test "Fp6 mulByXi is correct for ξ = 9 + u" {
    // Test that mulByXi(a) = a * (9 + u) for a simple Fp2 element
    const a = Fp2.init(Fp.fromU64(2), Fp.fromU64(3));

    // Compute using mulByXi
    const result = Fp6.mulByXi(a);

    // Compute manually: (2 + 3u)(9 + u) = 18 + 2u + 27u + 3u² = 18 + 29u - 3 = 15 + 29u
    // (since u² = -1)
    const expected = Fp2.init(Fp.fromU64(15), Fp.fromU64(29));

    try std.testing.expect(result.eql(expected));
}

test "Fp6 mul by v via mulByXi" {
    // Test that multiplying by v works correctly
    // v^3 = ξ, so (c0 + c1*v + c2*v²) * v = c2*ξ + c0*v + c1*v²
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
