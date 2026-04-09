//! Curve-generic substrate for pairing-friendly elliptic curves.
//!
//! This module provides the organizing abstraction that lets zolt-arith
//! support multiple curves (BN254 for zolt's prover, BLS12-381 for
//! zyli's validator signature verification) through the same generic
//! factories: `montgomery_field`, `extensions`, `weierstrass`, `pairing`.
//!
//! Each curve lives in its own submodule (`bn254/`, `bls12_381/`) and
//! exports a type bundle: `Fp, Fr, Fp2, Fp6, Fp12, G1Affine, G2Affine,
//! Pairing, HashToG2, Bls`.

pub const montgomery_field = @import("montgomery_field.zig");
pub const MontgomeryField = montgomery_field.MontgomeryField;
pub const extensions = @import("extensions.zig");
pub const Fp2 = extensions.Fp2;

/// Curve family (drives final-exponentiation and loop-count derivation).
pub const Family = enum { bn, bls };

test {
    _ = montgomery_field;
    _ = extensions;
}
