//! Extension fields (Fp2, Fp6, Fp12) instantiated on fiat-crypto verified Fp.
//!
//! Uses the generic factories from curves/extensions.zig — only the base
//! field arithmetic is fiat-crypto backed. Extension field algorithms
//! (Karatsuba, Chung-Hasan, etc.) are the same proven-correct generic code.

const ext = @import("../curves/extensions.zig");
const bn254 = @import("bn254.zig");

pub const FiatFp = bn254.Fp;

/// BN254's cubic non-residue ξ = 9 + u.
/// Multiply Fp2 by (9+u) using shift-add (same algorithm as curves/extensions.zig tests).
pub fn fiatMulByXi(x: Fp2) Fp2 {
    // 9*a using shift-add: a8 = 8*a, a9 = a8 + a
    const a = x.c0;
    const a2 = FiatFp.add(a, a);
    const a4 = FiatFp.add(a2, a2);
    const a8 = FiatFp.add(a4, a4);
    const a9 = FiatFp.add(a8, a);
    // 9*b
    const b = x.c1;
    const b2 = FiatFp.add(b, b);
    const b4 = FiatFp.add(b2, b2);
    const b8 = FiatFp.add(b4, b4);
    const b9 = FiatFp.add(b8, b);
    // (9+u)(a+bu) = (9a - b) + (a + 9b)u
    return Fp2.init(FiatFp.sub(a9, b), FiatFp.add(a, b9));
}

pub const Fp2 = ext.Fp2(FiatFp);
pub const Fp6 = ext.Fp6(Fp2, fiatMulByXi);
pub const Fp12 = ext.Fp12(Fp6, Fp2);
