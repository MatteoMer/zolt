//! zolt-arith: Arithmetic primitives for Zolt.
//!
//! Hosts both the BN254 stack (consumed by zolt's prover/zkVM) and the
//! BLS12-381 stack (consumed by zyli's validator signature verification).
//! The curve-generic `MontgomeryField` factory lives under `curves/`;
//! each curve has its own implementation subtree.

// --- BN254 surface (existing — used by zolt) ----------------------------
pub const field = @import("field/mod.zig");
pub const poly = @import("poly/mod.zig");
pub const msm = @import("msm/mod.zig");
pub const gpu = @import("gpu/mod.zig");
pub const transcripts = @import("transcripts/mod.zig");
pub const subprotocols = @import("subprotocols/mod.zig");

pub const JoltField = field.JoltField;
pub const BN254Scalar = field.BN254Scalar;

pub const bits = @import("bits.zig");
pub const LookupBits = bits.LookupBits;
pub const uninterleaveBits = bits.uninterleaveBits;
pub const interleaveBits = bits.interleaveBits;
pub const expanding_table = @import("expanding_table.zig");
pub const ExpandingTable = expanding_table.ExpandingTable;

// --- Curve-generic substrate + BLS12-381 surface (used by zyli) ----------
pub const curves = @import("curves/mod.zig");
pub const bigint = @import("bigint.zig");
pub const bls12_381 = @import("curves/bls12_381/mod.zig");
pub const hash_to_field = bls12_381.hash_to_field;
pub const hash_to_curve_g2 = bls12_381.hash_to_curve_g2;
pub const bls = bls12_381.bls;

test {
    _ = @import("field/mod.zig");
    _ = @import("poly/mod.zig");
    _ = @import("msm/mod.zig");
    _ = @import("transcripts/mod.zig");
    _ = @import("subprotocols/mod.zig");
    _ = @import("bits.zig");

    _ = @import("bigint.zig");
    _ = @import("curves/mod.zig");
    _ = @import("curves/montgomery_field.zig");
    _ = @import("curves/bn254/mod.zig");
    _ = @import("curves/bls12_381/mod.zig");

    // Fiat-crypto verified field arithmetic
    _ = @import("fiat/bn254.zig");
    _ = @import("fiat/diff_tests.zig");
    _ = @import("fiat/property_tests.zig");
}
