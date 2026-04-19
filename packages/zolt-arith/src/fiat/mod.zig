//! Fiat-crypto verified field arithmetic for BN254.
//!
//! All base field operations (add, sub, mul, square, etc.) are backed by
//! machine-verified Zig code generated from Coq proofs via fiat-crypto v0.1.5.
//! Extension fields and pairing are built compositionally on this verified base.

pub const bn254 = @import("bn254.zig");
pub const Fr = bn254.Fr;
pub const Fp = bn254.Fp;

pub const extensions = @import("extensions.zig");
pub const Fp2 = extensions.Fp2;
pub const Fp6 = extensions.Fp6;
pub const Fp12 = extensions.Fp12;

pub const pairing = @import("pairing.zig");
pub const G1Point = pairing.G1Point;
pub const G2Point = pairing.G2Point;
pub const pairingFp = pairing.pairingFp;
pub const millerLoop = pairing.millerLoop;
pub const finalExponentiation = pairing.finalExponentiation;
