//! BN254 curve parameters for the generic curve substrate.
//!
//! These are the field moduli, Montgomery constants, and (eventually)
//! extension field / pairing parameters. They are the SAME numbers that
//! live in `field/mod.zig` today — just gathered into a self-contained
//! location so the generic factories can reference them.

// ── Scalar field Fr ──────────────────────────────────────────────────

/// BN254 scalar field modulus
/// p = 21888242871839275222246405745257275088548364400416034343698204186575808495617
pub const FR_MODULUS: [4]u64 = .{
    0x43e1f593f0000001,
    0x2833e84879b97091,
    0xb85045b68181585d,
    0x30644e72e131a029,
};

/// Montgomery R for Fr (R = 2^256 mod p)
pub const FR_R: [4]u64 = .{
    0xac96341c4ffffffb,
    0x36fc76959f60cd29,
    0x666ea36f7879462e,
    0x0e0a77c19a07df2f,
};

/// Montgomery R^2 for Fr (R^2 = 2^512 mod p)
pub const FR_R2: [4]u64 = .{
    0x1bb8e645ae216da7,
    0x53fe3ab1e35c59e3,
    0x8c49833d53bb8085,
    0x0216d0b17f4e44a5,
};

/// -p^{-1} mod 2^64
pub const FR_INV: u64 = 0xc2e1f593efffffff;

// ── Base field Fp ────────────────────────────────────────────────────

/// BN254 base field modulus
/// q = 21888242871839275222246405745257275088696311157297823662689037894645226208583
pub const FP_MODULUS: [4]u64 = .{
    0x3c208c16d87cfd47,
    0x97816a916871ca8d,
    0xb85045b68181585d,
    0x30644e72e131a029,
};

/// Montgomery R for Fp
pub const FP_R: [4]u64 = .{
    0xd35d438dc58f0d9d,
    0x0a78eb28f5c70b3d,
    0x666ea36f7879462c,
    0x0e0a77c19a07df2f,
};

/// Montgomery R^2 for Fp
pub const FP_R2: [4]u64 = .{
    0xf32cfc5b538afa89,
    0xb5e71911d44501fb,
    0x47ab1eff0a417ff6,
    0x06d89f71cab8351f,
};

/// -q^{-1} mod 2^64
pub const FP_INV: u64 = 0x87d20782e4866389;
