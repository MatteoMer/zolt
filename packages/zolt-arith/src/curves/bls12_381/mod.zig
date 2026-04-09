//! BLS12-381 curve type bundle.
//!
//! Re-exports from the self-contained BLS12-381 implementation under
//! this directory. The public API matches the standalone `zolt-arith`
//! package that zyli consumed before consolidation.

pub const curve = @import("curve.zig");
pub const bls = @import("bls.zig");
pub const hash_to_field = @import("hash_to_field.zig");
pub const hash_to_curve_g2 = @import("hash_to_curve_g2.zig");

// Re-export the most commonly accessed types at the top level so
// callers can write `bls12_381.G1Affine` instead of
// `bls12_381.curve.G1Affine`.
pub const Fp = curve.Fp;
pub const Fr = curve.Fr;
pub const Fp2 = curve.Fp2;
pub const Fp6 = curve.Fp6;
pub const Fp12 = curve.Fp12;
pub const G1Affine = curve.G1Affine;
pub const G1Projective = curve.G1Projective;
pub const G2Affine = curve.G2Affine;
pub const G2Projective = curve.G2Projective;
pub const PointDecodeError = curve.PointDecodeError;

pub const decodeG1Compressed = curve.decodeG1Compressed;
pub const decodeG2Compressed = curve.decodeG2Compressed;
pub const encodeG1Compressed = curve.encodeG1Compressed;
pub const encodeG2Compressed = curve.encodeG2Compressed;
pub const g1Generator = curve.g1Generator;
pub const g2Generator = curve.g2Generator;
pub const isInG1Subgroup = curve.isInG1Subgroup;
pub const isInG2Subgroup = curve.isInG2Subgroup;
pub const millerLoop = curve.millerLoop;
pub const fp12FinalExp = curve.fp12FinalExp;
pub const fpFromBytes64Be = curve.fpFromBytes64Be;

test {
    _ = curve;
    _ = bls;
    _ = hash_to_field;
    _ = hash_to_curve_g2;
}
