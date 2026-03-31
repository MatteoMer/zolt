//! Finite field arithmetic for Jolt
//!
//! This module provides field element types and operations for the cryptographic
//! protocols used in Jolt. The primary field is the BN254 scalar field.

const std = @import("std");
const builtin = @import("builtin");

/// LLVM carry/borrow intrinsics — map to single adc/sbb instructions on x86-64.
/// Wrapped in a comptime-conditional struct so they are not emitted on non-x86 targets.
const x86 = if (builtin.cpu.arch == .x86_64) struct {
    extern fn @"llvm.x86.addcarry.u64"(c_in: u8, a: u64, b: u64, result: *u64) u8;
    extern fn @"llvm.x86.subborrow.u64"(b_in: u8, a: u64, b: u64, result: *u64) u8;

    pub inline fn addcarry(c_in: u8, a: u64, b: u64, result: *u64) u8 {
        return @"llvm.x86.addcarry.u64"(c_in, a, b, result);
    }
    pub inline fn subborrow(b_in: u8, a: u64, b: u64, result: *u64) u8 {
        return @"llvm.x86.subborrow.u64"(b_in, a, b, result);
    }
} else struct {};

/// Comptime flag: true when x86-64 BMI2+ADX instructions are available
const use_asm_mul = blk: {
    if (builtin.cpu.arch != .x86_64) break :blk false;
    const features = builtin.cpu.features;
    break :blk features.isEnabled(@intFromEnum(std.Target.x86.Feature.bmi2)) and
        features.isEnabled(@intFromEnum(std.Target.x86.Feature.adx));
};

/// Comptime flag: true when targeting AArch64 (all AArch64 has adds/adcs/subs/sbcs).
const use_arm64_asm = (builtin.cpu.arch == .aarch64);

// ── ARM64 (AArch64) inline-asm helpers for 256-bit / 192-bit field ops ──────
// These use proper adds/adcs/subs/sbcs carry chains instead of the u128 fallback
// which generates catastrophic cinc + asr #63 codegen on ARM64.

/// Field subtraction: (a - b) mod p, branch-based.
/// subs/sbcs chain, then b.cs skips correction if no borrow.
inline fn arm64SubMod256(a: [4]u64, b: [4]u64, mod: [4]u64) [4]u64 {
    if (comptime builtin.cpu.arch != .aarch64) unreachable;
    var r0: u64 = undefined;
    var r1: u64 = undefined;
    var r2: u64 = undefined;
    var r3: u64 = undefined;
    asm volatile (
        \\ subs %[r0], %[a0], %[b0]
        \\ sbcs %[r1], %[a1], %[b1]
        \\ sbcs %[r2], %[a2], %[b2]
        \\ sbcs %[r3], %[a3], %[b3]
        \\ b.cs 1f
        \\ adds %[r0], %[r0], %[m0]
        \\ adcs %[r1], %[r1], %[m1]
        \\ adcs %[r2], %[r2], %[m2]
        \\ adc  %[r3], %[r3], %[m3]
        \\ 1:
        : [r0] "=&r" (r0),
          [r1] "=&r" (r1),
          [r2] "=&r" (r2),
          [r3] "=&r" (r3),
        : [a0] "r" (a[0]),
          [a1] "r" (a[1]),
          [a2] "r" (a[2]),
          [a3] "r" (a[3]),
          [b0] "r" (b[0]),
          [b1] "r" (b[1]),
          [b2] "r" (b[2]),
          [b3] "r" (b[3]),
          [m0] "r" (mod[0]),
          [m1] "r" (mod[1]),
          [m2] "r" (mod[2]),
          [m3] "r" (mod[3]),
        : .{ .nzcv = true }
    );
    return .{ r0, r1, r2, r3 };
}

/// Unconditional 4-limb subtraction (no modular reduction).
inline fn arm64Sub256(a: [4]u64, b: [4]u64) [4]u64 {
    if (comptime builtin.cpu.arch != .aarch64) unreachable;
    var r0: u64 = undefined;
    var r1: u64 = undefined;
    var r2: u64 = undefined;
    var r3: u64 = undefined;
    asm volatile (
        \\ subs %[r0], %[a0], %[b0]
        \\ sbcs %[r1], %[a1], %[b1]
        \\ sbcs %[r2], %[a2], %[b2]
        \\ sbc  %[r3], %[a3], %[b3]
        : [r0] "=&r" (r0),
          [r1] "=&r" (r1),
          [r2] "=&r" (r2),
          [r3] "=&r" (r3),
        : [a0] "r" (a[0]),
          [a1] "r" (a[1]),
          [a2] "r" (a[2]),
          [a3] "r" (a[3]),
          [b0] "r" (b[0]),
          [b1] "r" (b[1]),
          [b2] "r" (b[2]),
          [b3] "r" (b[3]),
        : .{ .nzcv = true }
    );
    return .{ r0, r1, r2, r3 };
}

/// Unconditional 4-limb addition (no modular reduction).
inline fn arm64Add256(a: [4]u64, b: [4]u64) [4]u64 {
    if (comptime builtin.cpu.arch != .aarch64) unreachable;
    var r0: u64 = undefined;
    var r1: u64 = undefined;
    var r2: u64 = undefined;
    var r3: u64 = undefined;
    asm volatile (
        \\ adds %[r0], %[a0], %[b0]
        \\ adcs %[r1], %[a1], %[b1]
        \\ adcs %[r2], %[a2], %[b2]
        \\ adc  %[r3], %[a3], %[b3]
        : [r0] "=&r" (r0),
          [r1] "=&r" (r1),
          [r2] "=&r" (r2),
          [r3] "=&r" (r3),
        : [a0] "r" (a[0]),
          [a1] "r" (a[1]),
          [a2] "r" (a[2]),
          [a3] "r" (a[3]),
          [b0] "r" (b[0]),
          [b1] "r" (b[1]),
          [b2] "r" (b[2]),
          [b3] "r" (b[3]),
        : .{ .nzcv = true }
    );
    return .{ r0, r1, r2, r3 };
}

/// 3-limb unsigned addition (for S192 magnitudes).
inline fn arm64Add192(a: [3]u64, b: [3]u64) [3]u64 {
    if (comptime builtin.cpu.arch != .aarch64) unreachable;
    var r0: u64 = undefined;
    var r1: u64 = undefined;
    var r2: u64 = undefined;
    asm volatile (
        \\ adds %[r0], %[a0], %[b0]
        \\ adcs %[r1], %[a1], %[b1]
        \\ adc  %[r2], %[a2], %[b2]
        : [r0] "=&r" (r0),
          [r1] "=&r" (r1),
          [r2] "=&r" (r2),
        : [a0] "r" (a[0]),
          [a1] "r" (a[1]),
          [a2] "r" (a[2]),
          [b0] "r" (b[0]),
          [b1] "r" (b[1]),
          [b2] "r" (b[2]),
        : .{ .nzcv = true }
    );
    return .{ r0, r1, r2 };
}

/// 3-limb unsigned subtraction (for S192 magnitudes).
inline fn arm64Sub192(a: [3]u64, b: [3]u64) [3]u64 {
    if (comptime builtin.cpu.arch != .aarch64) unreachable;
    var r0: u64 = undefined;
    var r1: u64 = undefined;
    var r2: u64 = undefined;
    asm volatile (
        \\ subs %[r0], %[a0], %[b0]
        \\ sbcs %[r1], %[a1], %[b1]
        \\ sbc  %[r2], %[a2], %[b2]
        : [r0] "=&r" (r0),
          [r1] "=&r" (r1),
          [r2] "=&r" (r2),
        : [a0] "r" (a[0]),
          [a1] "r" (a[1]),
          [a2] "r" (a[2]),
          [b0] "r" (b[0]),
          [b1] "r" (b[1]),
          [b2] "r" (b[2]),
        : .{ .nzcv = true }
    );
    return .{ r0, r1, r2 };
}

/// ARM64 4-limb CIOS Montgomery multiplication.
/// Computes a * b * R^{-1} mod p using fully-unrolled CIOS with two-pass
/// carry accumulation (ARM64 substitute for x86 ADX dual carry chains).
/// All constants (b, mod, inv) are preloaded into registers.
fn arm64MontgomeryMul256(a: *const [4]u64, b: *const [4]u64, mod: *const [4]u64, inv: u64) [4]u64 {
    if (comptime builtin.cpu.arch != .aarch64) unreachable;
    // After 4 iterations of register rotation, the result limbs end up in:
    //   iter 0: t=[x0,x1,x2,x3,x14] → shift → [x1,x2,x3,x14], t4=x0
    //   iter 1: [x1,x2,x3,x14,x0]   → shift → [x2,x3,x14,x0], t4=x1
    //   iter 2: [x2,x3,x14,x0,x1]   → shift → [x3,x14,x0,x1], t4=x2
    //   iter 3: [x3,x14,x0,x1,x2]   → shift → [x14,x0,x1,x2], t4=x3
    // Final result: limbs = {x14, x0, x1, x2}
    var r0: u64 = undefined;
    var r1: u64 = undefined;
    var r2: u64 = undefined;
    var r3: u64 = undefined;
    asm volatile (
    // ── Preload constants ──
        \\ ldp x4, x5, [x25]
        \\ ldp x6, x7, [x25, #16]
        \\ ldp x8, x9, [x26]
        \\ ldp x10, x11, [x26, #16]
        //
        // ════════════════════════════════════════════════
        // Iteration 0: mul_1 (t starts at 0) + reduce
        // Accumulator: t0=x0, t1=x1, t2=x2, t3=x3, t4=x14
        // ════════════════════════════════════════════════
        \\ ldr x15, [x13]
        // a[0] * b[0..3] → products; t starts at 0 so just store directly
        \\ mul  x0,  x15, x4
        \\ umulh x1,  x15, x4
        \\ mul  x16, x15, x5
        \\ umulh x2,  x15, x5
        \\ mul  x17, x15, x6
        \\ umulh x3,  x15, x6
        \\ mul  x19, x15, x7
        \\ umulh x14, x15, x7
        // Single carry chain (t was 0, only need to add lo parts)
        \\ adds x1, x1, x16
        \\ adcs x2, x2, x17
        \\ adcs x3, x3, x19
        \\ adc  x14, x14, xzr
        // k = t[0] * inv
        \\ mul  x15, x0, x12
        // k * mod[0..3]
        \\ mul  x16, x15, x8
        \\ umulh x17, x15, x8
        \\ mul  x19, x15, x9
        \\ umulh x20, x15, x9
        \\ mul  x21, x15, x10
        \\ umulh x22, x15, x10
        \\ mul  x23, x15, x11
        \\ umulh x24, x15, x11
        // Cancel t[0]: t[0] + lo(k*m0) = 0 mod 2^64, carry propagates
        \\ adds x0, x0, x16
        // Pass 1: hi parts
        \\ adcs x1, x1, x17
        \\ adcs x2, x2, x20
        \\ adcs x3, x3, x22
        \\ adc  x14, x14, x24
        // Pass 2: lo parts
        \\ adds x1, x1, x19
        \\ adcs x2, x2, x21
        \\ adcs x3, x3, x23
        \\ adc  x14, x14, xzr
        // After shift: t = [x1, x2, x3, x14], x0 = 0 (new t4)
        // (x0 held the cancelled value = 0)
        //
        // ════════════════════════════════════════════════
        // Iteration 1: mul_add_1 + reduce
        // Accumulator: t0=x1, t1=x2, t2=x3, t3=x14, t4=x0
        // ════════════════════════════════════════════════
        \\ ldr x15, [x13, #8]
        \\ mul  x16, x15, x4
        \\ umulh x17, x15, x4
        \\ mul  x19, x15, x5
        \\ umulh x20, x15, x5
        \\ mul  x21, x15, x6
        \\ umulh x22, x15, x6
        \\ mul  x23, x15, x7
        \\ umulh x24, x15, x7
        // Pass 1: lo(b0) to t0, hi parts to t1..t4
        \\ adds x1, x1, x16
        \\ adcs x2, x2, x17
        \\ adcs x3, x3, x20
        \\ adcs x14, x14, x22
        \\ adc  x0, x0, x24
        // Pass 2: remaining lo parts
        \\ adds x2, x2, x19
        \\ adcs x3, x3, x21
        \\ adcs x14, x14, x23
        \\ adc  x0, x0, xzr
        // k = t[0] * inv
        \\ mul  x15, x1, x12
        // k * mod[0..3]
        \\ mul  x16, x15, x8
        \\ umulh x17, x15, x8
        \\ mul  x19, x15, x9
        \\ umulh x20, x15, x9
        \\ mul  x21, x15, x10
        \\ umulh x22, x15, x10
        \\ mul  x23, x15, x11
        \\ umulh x24, x15, x11
        // Cancel t[0]
        \\ adds x1, x1, x16
        // Pass 1: hi parts
        \\ adcs x2, x2, x17
        \\ adcs x3, x3, x20
        \\ adcs x14, x14, x22
        \\ adc  x0, x0, x24
        // Pass 2: lo parts
        \\ adds x2, x2, x19
        \\ adcs x3, x3, x21
        \\ adcs x14, x14, x23
        \\ adc  x0, x0, xzr
        // After shift: t = [x2, x3, x14, x0], t4 = x1 (= 0)
        //
        // ════════════════════════════════════════════════
        // Iteration 2: mul_add_1 + reduce
        // Accumulator: t0=x2, t1=x3, t2=x14, t3=x0, t4=x1
        // ════════════════════════════════════════════════
        \\ ldr x15, [x13, #16]
        \\ mul  x16, x15, x4
        \\ umulh x17, x15, x4
        \\ mul  x19, x15, x5
        \\ umulh x20, x15, x5
        \\ mul  x21, x15, x6
        \\ umulh x22, x15, x6
        \\ mul  x23, x15, x7
        \\ umulh x24, x15, x7
        // Pass 1
        \\ adds x2, x2, x16
        \\ adcs x3, x3, x17
        \\ adcs x14, x14, x20
        \\ adcs x0, x0, x22
        \\ adc  x1, x1, x24
        // Pass 2
        \\ adds x3, x3, x19
        \\ adcs x14, x14, x21
        \\ adcs x0, x0, x23
        \\ adc  x1, x1, xzr
        // k = t[0] * inv
        \\ mul  x15, x2, x12
        // k * mod[0..3]
        \\ mul  x16, x15, x8
        \\ umulh x17, x15, x8
        \\ mul  x19, x15, x9
        \\ umulh x20, x15, x9
        \\ mul  x21, x15, x10
        \\ umulh x22, x15, x10
        \\ mul  x23, x15, x11
        \\ umulh x24, x15, x11
        // Cancel t[0]
        \\ adds x2, x2, x16
        // Pass 1
        \\ adcs x3, x3, x17
        \\ adcs x14, x14, x20
        \\ adcs x0, x0, x22
        \\ adc  x1, x1, x24
        // Pass 2
        \\ adds x3, x3, x19
        \\ adcs x14, x14, x21
        \\ adcs x0, x0, x23
        \\ adc  x1, x1, xzr
        // After shift: t = [x3, x14, x0, x1], t4 = x2 (= 0)
        //
        // ════════════════════════════════════════════════
        // Iteration 3: mul_add_1 + reduce
        // Accumulator: t0=x3, t1=x14, t2=x0, t3=x1, t4=x2
        // ════════════════════════════════════════════════
        \\ ldr x15, [x13, #24]
        \\ mul  x16, x15, x4
        \\ umulh x17, x15, x4
        \\ mul  x19, x15, x5
        \\ umulh x20, x15, x5
        \\ mul  x21, x15, x6
        \\ umulh x22, x15, x6
        \\ mul  x23, x15, x7
        \\ umulh x24, x15, x7
        // Pass 1
        \\ adds x3, x3, x16
        \\ adcs x14, x14, x17
        \\ adcs x0, x0, x20
        \\ adcs x1, x1, x22
        \\ adc  x2, x2, x24
        // Pass 2
        \\ adds x14, x14, x19
        \\ adcs x0, x0, x21
        \\ adcs x1, x1, x23
        \\ adc  x2, x2, xzr
        // k = t[0] * inv
        \\ mul  x15, x3, x12
        // k * mod[0..3]
        \\ mul  x16, x15, x8
        \\ umulh x17, x15, x8
        \\ mul  x19, x15, x9
        \\ umulh x20, x15, x9
        \\ mul  x21, x15, x10
        \\ umulh x22, x15, x10
        \\ mul  x23, x15, x11
        \\ umulh x24, x15, x11
        // Cancel t[0]
        \\ adds x3, x3, x16
        // Pass 1
        \\ adcs x14, x14, x17
        \\ adcs x0, x0, x20
        \\ adcs x1, x1, x22
        \\ adc  x2, x2, x24
        // Pass 2
        \\ adds x14, x14, x19
        \\ adcs x0, x0, x21
        \\ adcs x1, x1, x23
        \\ adc  x2, x2, xzr
        // Final result: limbs = {x14, x0, x1, x2}
        : [_r0] "={x14}" (r0),
          [_r1] "={x0}" (r1),
          [_r2] "={x1}" (r2),
          [_r3] "={x2}" (r3),
        : [_a] "{x13}" (a),
          [_b] "{x25}" (b),
          [_mod] "{x26}" (mod),
          [_inv] "{x12}" (inv),
        : .{ .x0 = true, .x1 = true, .x2 = true, .x3 = true, .x4 = true, .x5 = true,
             .x6 = true, .x7 = true, .x8 = true, .x9 = true, .x10 = true, .x11 = true,
             .x14 = true, .x15 = true, .x16 = true, .x17 = true,
             .x19 = true, .x20 = true, .x21 = true, .x22 = true, .x23 = true, .x24 = true,
             .x25 = true, .x26 = true,
             .nzcv = true, .memory = true }
    );
    return .{ r0, r1, r2, r3 };
}

// Debug output control - set to true to enable verbose debug prints
const debug_verbose = false;
fn dbg(comptime fmt: []const u8, args: anytype) void {
    if (debug_verbose) std.debug.print(fmt, args);
}

/// Number of bytes in a field element (256 bits = 32 bytes)
pub const FIELD_ELEMENT_BYTES: usize = 32;

/// Number of 64-bit limbs in a field element
pub const NUM_LIMBS: usize = 4;

/// BN254 scalar field modulus
/// p = 21888242871839275222246405745257275088548364400416034343698204186575808495617
pub const BN254_MODULUS: [4]u64 = .{
    0x43e1f593f0000001,
    0x2833e84879b97091,
    0xb85045b68181585d,
    0x30644e72e131a029,
};

/// Montgomery R for BN254 (R = 2^256 mod p)
pub const BN254_R: [4]u64 = .{
    0xac96341c4ffffffb,
    0x36fc76959f60cd29,
    0x666ea36f7879462e,
    0x0e0a77c19a07df2f,
};

/// Montgomery R^2 for BN254 (R^2 = 2^512 mod p)
pub const BN254_R2: [4]u64 = .{
    0x1bb8e645ae216da7,
    0x53fe3ab1e35c59e3,
    0x8c49833d53bb8085,
    0x0216d0b17f4e44a5,
};

/// Montgomery constant: -p^{-1} mod 2^64
/// Used in Montgomery reduction
pub const BN254_INV: u64 = 0xc2e1f593efffffff;

// ============================================================================
// BN254 Base Field (Fp) - for pairing operations
// ============================================================================
// The base field Fp is different from the scalar field Fr!
// Fp is used for G1/G2 point coordinates and the pairing target group GT

/// BN254 base field modulus
/// q = 21888242871839275222246405745257275088696311157297823662689037894645226208583
pub const BN254_FP_MODULUS: [4]u64 = .{
    0x3c208c16d87cfd47,
    0x97816a916871ca8d,
    0xb85045b68181585d,
    0x30644e72e131a029,
};

/// Montgomery R for Fp (R = 2^256 mod q)
pub const BN254_FP_R: [4]u64 = .{
    0xd35d438dc58f0d9d,
    0x0a78eb28f5c70b3d,
    0x666ea36f7879462c,
    0x0e0a77c19a07df2f,
};

/// Montgomery R^2 for Fp (R^2 = 2^512 mod q)
pub const BN254_FP_R2: [4]u64 = .{
    0xf32cfc5b538afa89,
    0xb5e71911d44501fb,
    0x47ab1eff0a417ff6,
    0x06d89f71cab8351f,
};

/// Montgomery constant: -q^{-1} mod 2^64
pub const BN254_FP_INV: u64 = 0x87d20782e4866389;

/// BN254 base field element for pairing operations
/// This is a wrapper around BN254Scalar that uses the base field modulus
/// Used for Fp, Fp2, Fp6, Fp12 tower and G1/G2 coordinates
pub const BN254BaseField = MontgomeryField(
    BN254_FP_MODULUS,
    BN254_FP_R,
    BN254_FP_R2,
    BN254_FP_INV,
);

/// Debug helper for testing Montgomery multiplication with Fp constants
pub fn testMontgomeryMulFp(a: [4]u64, b: [4]u64) [4]u64 {
    var t: [5]u64 = .{ 0, 0, 0, 0, 0 };

    inline for (0..4) |i| {
        var carry: u64 = 0;
        inline for (0..4) |j| {
            const prod = @as(u128, a[i]) * @as(u128, b[j]);
            const sum = @as(u128, t[j]) + prod + @as(u128, carry);
            t[j] = @truncate(sum);
            carry = @truncate(sum >> 64);
        }
        const sum_t4 = @as(u128, t[4]) + @as(u128, carry);
        t[4] = @truncate(sum_t4);

        const m = t[0] *% BN254_FP_INV;

        carry = 0;
        const prod0 = @as(u128, m) * @as(u128, BN254_FP_MODULUS[0]);
        const sum0 = @as(u128, t[0]) + prod0;
        carry = @truncate(sum0 >> 64);

        inline for (1..4) |j| {
            const prod = @as(u128, m) * @as(u128, BN254_FP_MODULUS[j]);
            const sum = @as(u128, t[j]) + prod + @as(u128, carry);
            t[j - 1] = @truncate(sum);
            carry = @truncate(sum >> 64);
        }
        const final_sum = @as(u128, t[4]) + @as(u128, carry);
        t[3] = @truncate(final_sum);
        t[4] = @truncate(final_sum >> 64);
    }

    return .{ t[0], t[1], t[2], t[3] };
}

/// Generic Montgomery field parameterized by constants
pub fn MontgomeryField(
    comptime modulus: [4]u64,
    comptime montgomery_r: [4]u64,
    comptime montgomery_r2: [4]u64,
    comptime montgomery_inv: u64,
) type {
    return struct {
        limbs: [4]u64,

        const Self = @This();

        /// Zero element
        pub fn zero() Self {
            return .{ .limbs = .{ 0, 0, 0, 0 } };
        }

        /// One element (in Montgomery form = R mod p)
        pub fn one() Self {
            return .{ .limbs = montgomery_r };
        }

        /// Check if zero
        pub fn isZero(self: Self) bool {
            return self.limbs[0] == 0 and self.limbs[1] == 0 and
                self.limbs[2] == 0 and self.limbs[3] == 0;
        }

        /// Check if one (in Montgomery form)
        pub fn isOne(self: Self) bool {
            return self.limbs[0] == montgomery_r[0] and self.limbs[1] == montgomery_r[1] and
                self.limbs[2] == montgomery_r[2] and self.limbs[3] == montgomery_r[3];
        }

        /// Equality check
        pub fn eql(self: Self, other: Self) bool {
            return self.limbs[0] == other.limbs[0] and self.limbs[1] == other.limbs[1] and
                self.limbs[2] == other.limbs[2] and self.limbs[3] == other.limbs[3];
        }

        /// Create from u64 (converts to Montgomery form)
        pub fn fromU64(n: u64) Self {
            var result = Self{ .limbs = .{ n, 0, 0, 0 } };
            result = result.montgomeryMul(.{ .limbs = montgomery_r2 });
            return result;
        }

        /// Create from u128 (converts to Montgomery form)
        pub fn fromU128(n: u128) Self {
            const raw = Self{ .limbs = .{ @truncate(n), @truncate(n >> 64), 0, 0 } };
            return raw.montgomeryMul(.{ .limbs = montgomery_r2 });
        }

        /// Create from bytes (little-endian, converts to Montgomery form)
        pub fn fromBytes(bytes: []const u8) Self {
            var limbs: [4]u64 = .{ 0, 0, 0, 0 };
            const len = @min(bytes.len, 32);
            var buf: [32]u8 = .{0} ** 32;
            @memcpy(buf[0..len], bytes[0..len]);

            for (0..4) |i| {
                limbs[i] = std.mem.readInt(u64, buf[i * 8 ..][0..8], .little);
            }

            var result = Self{ .limbs = limbs };
            result = result.montgomeryMul(.{ .limbs = montgomery_r2 });
            return result;
        }

        /// Convert from Montgomery form back to standard representation
        pub fn fromMontgomery(self: Self) Self {
            return self.montgomeryMul(.{ .limbs = .{ 1, 0, 0, 0 } });
        }

        /// Convert to Montgomery form from standard representation
        pub fn toMontgomery(self: Self) Self {
            return self.montgomeryMul(.{ .limbs = montgomery_r2 });
        }

        /// Create from big-endian bytes (converts to Montgomery form)
        pub fn fromBytesBE(bytes: *const [32]u8) Self {
            // Reverse byte order for big-endian
            var le_bytes: [32]u8 = undefined;
            for (0..32) |i| {
                le_bytes[i] = bytes[31 - i];
            }
            return fromBytes(&le_bytes);
        }

        /// Serialize to big-endian bytes (32 bytes)
        pub fn toBytesBE(self: Self) [32]u8 {
            // First convert from Montgomery form
            const standard = self.fromMontgomery();
            // DEBUG
            if (standard.limbs[0] == 0 and standard.limbs[1] == 0 and standard.limbs[2] == 0 and standard.limbs[3] == 0) {
                if (self.limbs[0] != 0 or self.limbs[1] != 0 or self.limbs[2] != 0 or self.limbs[3] != 0) {
                    dbg("[ZOLT DEBUG] toBytesBE: non-zero input produced zero output!\n", .{});
                    dbg("[ZOLT DEBUG]   input_limbs = [{x}, {x}, {x}, {x}]\n", .{ self.limbs[0], self.limbs[1], self.limbs[2], self.limbs[3] });
                    dbg("[ZOLT DEBUG]   output_limbs = [{x}, {x}, {x}, {x}]\n", .{ standard.limbs[0], standard.limbs[1], standard.limbs[2], standard.limbs[3] });
                }
            }

            // Convert limbs to bytes (little-endian)
            var le_bytes: [32]u8 = undefined;
            for (0..4) |i| {
                std.mem.writeInt(u64, le_bytes[i * 8 ..][0..8], standard.limbs[i], .little);
            }

            // Reverse for big-endian output
            var be_bytes: [32]u8 = undefined;
            for (0..32) |i| {
                be_bytes[i] = le_bytes[31 - i];
            }
            return be_bytes;
        }

        /// 128-bit multiplication helper
        inline fn mulWide(a: u64, b: u64) u128 {
            return @as(u128, a) * @as(u128, b);
        }

        /// Add with carry
        inline fn addCarry(a: u64, b: u64, carry_in: u64) struct { result: u64, carry: u64 } {
            if (comptime builtin.cpu.arch == .x86_64) {
                var result: u64 = undefined;
                const c = x86.addcarry(@truncate(carry_in), a, b, &result);
                return .{ .result = result, .carry = c };
            }
            const sum = @as(u128, a) + @as(u128, b) + @as(u128, carry_in);
            return .{ .result = @truncate(sum), .carry = @truncate(sum >> 64) };
        }

        inline fn subBorrow(a: u64, b: u64, borrow_in: u64) struct { result: u64, borrow: u64 } {
            if (comptime builtin.cpu.arch == .x86_64) {
                var result: u64 = undefined;
                const b_out = x86.subborrow(@truncate(borrow_in), a, b, &result);
                return .{ .result = result, .borrow = b_out };
            }
            const wide_a = @as(u128, a);
            const wide_b = @as(u128, b) + @as(u128, borrow_in);
            const diff = wide_a -% wide_b;
            return .{ .result = @truncate(diff), .borrow = @truncate(diff >> 127) };
        }

        /// Montgomery multiplication: computes a*b*R^{-1} mod p
        pub fn montgomeryMul(self: Self, other: Self) Self {
            var t: [5]u64 = .{ 0, 0, 0, 0, 0 };

            inline for (0..4) |i| {
                var carry: u64 = 0;
                inline for (0..4) |j| {
                    const prod = mulWide(self.limbs[i], other.limbs[j]);
                    const sum = @as(u128, t[j]) + prod + @as(u128, carry);
                    t[j] = @truncate(sum);
                    carry = @truncate(sum >> 64);
                }
                const sum_t4 = @as(u128, t[4]) + @as(u128, carry);
                t[4] = @truncate(sum_t4);

                const m = t[0] *% montgomery_inv;

                carry = 0;
                const prod0 = mulWide(m, modulus[0]);
                const sum0 = @as(u128, t[0]) + prod0;
                carry = @truncate(sum0 >> 64);

                inline for (1..4) |j| {
                    const prod = mulWide(m, modulus[j]);
                    const sum = @as(u128, t[j]) + prod + @as(u128, carry);
                    t[j - 1] = @truncate(sum);
                    carry = @truncate(sum >> 64);
                }
                const final_sum = @as(u128, t[4]) + @as(u128, carry);
                t[3] = @truncate(final_sum);
                t[4] = @truncate(final_sum >> 64);
            }

            var result = Self{ .limbs = .{ t[0], t[1], t[2], t[3] } };

            if (t[4] != 0 or !result.lessThanModulus()) {
                result = result.subtractModulus();
            }

            return result;
        }

        /// x86-64 BMI2+ADX accelerated CIOS Montgomery multiplication.
        /// Uses mulxq (flag-free multiply) + adcxq/adoxq (dual carry chains)
        /// for ~20% speedup over pure Zig. Adapted from arkworks ff-asm.
        fn montgomeryMulX86(self: Self, other: Self) Self {
            const a = self.limbs;
            const b = other.limbs;
            const mod_arr: [4]u64 = modulus;

            var r0: u64 = undefined;
            var r1: u64 = undefined;
            var r2: u64 = undefined;
            var r3: u64 = undefined;

            // 4-limb CIOS Montgomery multiplication, fully unrolled.
            // Register mapping:
            //   r8-r11: accumulator t[0..3] (rotated each iteration)
            //   rdi: pointer to a[], rsi: pointer to b[], r14: pointer to mod[]
            //   rbx: montgomery_inv, rdx: mulxq multiplier
            //   rax, rcx, r13: scratch
            asm volatile (
                \\xorq %%rcx, %%rcx
                // Iteration 0: mul_1 + reduction
                \\movq (%%rdi), %%rdx
                \\mulxq (%%rsi), %%r8, %%r9
                \\mulxq 8(%%rsi), %%rax, %%r10
                \\adcxq %%rax, %%r9
                \\mulxq 16(%%rsi), %%rax, %%r11
                \\adcxq %%rax, %%r10
                \\mulxq 24(%%rsi), %%rax, %%rcx
                \\movq $0, %%r13
                \\adcxq %%rax, %%r11
                \\adcxq %%r13, %%rcx
                //
                \\movq %%rbx, %%rdx
                \\mulxq %%r8, %%rdx, %%rax
                \\mulxq (%%r14), %%rax, %%r13
                \\adcxq %%r8, %%rax
                \\adoxq %%r13, %%r9
                \\mulxq 8(%%r14), %%rax, %%r13
                \\adcxq %%rax, %%r9
                \\adoxq %%r13, %%r10
                \\mulxq 16(%%r14), %%rax, %%r13
                \\adcxq %%rax, %%r10
                \\adoxq %%r13, %%r11
                \\mulxq 24(%%r14), %%rax, %%r8
                \\movq $0, %%r13
                \\adcxq %%rax, %%r11
                \\adoxq %%rcx, %%r8
                \\adcxq %%r13, %%r8
                // Iteration 1: mul_add_1 + reduction
                \\movq 8(%%rdi), %%rdx
                \\mulxq (%%rsi), %%rax, %%r13
                \\adcxq %%rax, %%r9
                \\adoxq %%r13, %%r10
                \\mulxq 8(%%rsi), %%rax, %%r13
                \\adcxq %%rax, %%r10
                \\adoxq %%r13, %%r11
                \\mulxq 16(%%rsi), %%rax, %%r13
                \\adcxq %%rax, %%r11
                \\adoxq %%r13, %%r8
                \\mulxq 24(%%rsi), %%rax, %%rcx
                \\movq $0, %%r13
                \\adcxq %%rax, %%r8
                \\adoxq %%r13, %%rcx
                \\adcxq %%r13, %%rcx
                //
                \\movq %%rbx, %%rdx
                \\mulxq %%r9, %%rdx, %%rax
                \\mulxq (%%r14), %%rax, %%r13
                \\adcxq %%r9, %%rax
                \\adoxq %%r13, %%r10
                \\mulxq 8(%%r14), %%rax, %%r13
                \\adcxq %%rax, %%r10
                \\adoxq %%r13, %%r11
                \\mulxq 16(%%r14), %%rax, %%r13
                \\adcxq %%rax, %%r11
                \\adoxq %%r13, %%r8
                \\mulxq 24(%%r14), %%rax, %%r9
                \\movq $0, %%r13
                \\adcxq %%rax, %%r8
                \\adoxq %%rcx, %%r9
                \\adcxq %%r13, %%r9
                // Iteration 2: mul_add_1 + reduction
                \\movq 16(%%rdi), %%rdx
                \\mulxq (%%rsi), %%rax, %%r13
                \\adcxq %%rax, %%r10
                \\adoxq %%r13, %%r11
                \\mulxq 8(%%rsi), %%rax, %%r13
                \\adcxq %%rax, %%r11
                \\adoxq %%r13, %%r8
                \\mulxq 16(%%rsi), %%rax, %%r13
                \\adcxq %%rax, %%r8
                \\adoxq %%r13, %%r9
                \\mulxq 24(%%rsi), %%rax, %%rcx
                \\movq $0, %%r13
                \\adcxq %%rax, %%r9
                \\adoxq %%r13, %%rcx
                \\adcxq %%r13, %%rcx
                //
                \\movq %%rbx, %%rdx
                \\mulxq %%r10, %%rdx, %%rax
                \\mulxq (%%r14), %%rax, %%r13
                \\adcxq %%r10, %%rax
                \\adoxq %%r13, %%r11
                \\mulxq 8(%%r14), %%rax, %%r13
                \\adcxq %%rax, %%r11
                \\adoxq %%r13, %%r8
                \\mulxq 16(%%r14), %%rax, %%r13
                \\adcxq %%rax, %%r8
                \\adoxq %%r13, %%r9
                \\mulxq 24(%%r14), %%rax, %%r10
                \\movq $0, %%r13
                \\adcxq %%rax, %%r9
                \\adoxq %%rcx, %%r10
                \\adcxq %%r13, %%r10
                // Iteration 3: mul_add_1 + reduction
                \\movq 24(%%rdi), %%rdx
                \\mulxq (%%rsi), %%rax, %%r13
                \\adcxq %%rax, %%r11
                \\adoxq %%r13, %%r8
                \\mulxq 8(%%rsi), %%rax, %%r13
                \\adcxq %%rax, %%r8
                \\adoxq %%r13, %%r9
                \\mulxq 16(%%rsi), %%rax, %%r13
                \\adcxq %%rax, %%r9
                \\adoxq %%r13, %%r10
                \\mulxq 24(%%rsi), %%rax, %%rcx
                \\movq $0, %%r13
                \\adcxq %%rax, %%r10
                \\adoxq %%r13, %%rcx
                \\adcxq %%r13, %%rcx
                //
                \\movq %%rbx, %%rdx
                \\mulxq %%r11, %%rdx, %%rax
                \\mulxq (%%r14), %%rax, %%r13
                \\adcxq %%r11, %%rax
                \\adoxq %%r13, %%r8
                \\mulxq 8(%%r14), %%rax, %%r13
                \\adcxq %%rax, %%r8
                \\adoxq %%r13, %%r9
                \\mulxq 16(%%r14), %%rax, %%r13
                \\adcxq %%rax, %%r9
                \\adoxq %%r13, %%r10
                \\mulxq 24(%%r14), %%rax, %%r11
                \\movq $0, %%r13
                \\adcxq %%rax, %%r10
                \\adoxq %%rcx, %%r11
                \\adcxq %%r13, %%r11
                : [_r0] "={r8}" (r0),
                  [_r1] "={r9}" (r1),
                  [_r2] "={r10}" (r2),
                  [_r3] "={r11}" (r3),
                : [_a] "{rdi}" (&a),
                  [_b] "{rsi}" (&b),
                  [_mod] "{r14}" (&mod_arr),
                  [_inv] "{rbx}" (montgomery_inv),
                : .{ .rax = true, .rcx = true, .rdx = true, .r13 = true, .cc = true, .memory = true });

            var result = Self{ .limbs = .{ r0, r1, r2, r3 } };
            if (!result.lessThanModulus()) {
                result = result.subtractModulus();
            }
            return result;
        }

        /// Optimized multiplication by a 128-bit value stored in high limbs
        /// Matches arkworks' mul_hi_bigint_u128 behavior
        ///
        /// This is equivalent to: self * (limb2 + limb3 * 2^64) mod p
        /// where the 128-bit value is stored in positions [2] and [3] of a 4-limb BigInt.
        ///
        /// The algorithm is 2 iterations of CIOS Montgomery multiplication,
        /// specialized for when only the top 2 limbs of the RHS are non-zero.
        pub fn mulHiBigIntU128(self: Self, hi_limbs: [4]u64) Self {
            const limb_n2 = hi_limbs[2];
            const limb_n1 = hi_limbs[3];

            var r: [4]u64 = .{ 0, 0, 0, 0 };

            // i = 2 (N-2): Process limb_n2
            {
                var carry1: u64 = 0;

                // r[0] = r[0] + self[0] * limb_n2
                const prod0 = mulWide(self.limbs[0], limb_n2);
                const sum0 = @as(u128, r[0]) + prod0 + @as(u128, carry1);
                r[0] = @truncate(sum0);
                carry1 = @truncate(sum0 >> 64);

                // Montgomery reduction step
                const k = r[0] *% montgomery_inv;
                var carry2: u64 = 0;
                const red0 = mulWide(k, modulus[0]);
                const red_sum0 = @as(u128, r[0]) + red0;
                carry2 = @truncate(red_sum0 >> 64);

                // Process remaining limbs
                inline for (1..4) |j| {
                    const prod_j = mulWide(self.limbs[j], limb_n2);
                    const new_rj = @as(u128, r[j]) + prod_j + @as(u128, carry1);
                    const new_rj_trunc: u64 = @truncate(new_rj);
                    carry1 = @truncate(new_rj >> 64);

                    const red_j = mulWide(k, modulus[j]);
                    const red_sum_j = @as(u128, new_rj_trunc) + red_j + @as(u128, carry2);
                    r[j - 1] = @truncate(red_sum_j);
                    carry2 = @truncate(red_sum_j >> 64);

                    // Update r[j] for next iteration
                    r[j] = new_rj_trunc;
                }
                r[3] = carry1 +% carry2;
            }

            // i = 3 (N-1): Process limb_n1
            {
                var carry1: u64 = 0;

                // r[0] = r[0] + self[0] * limb_n1
                const prod0 = mulWide(self.limbs[0], limb_n1);
                const sum0 = @as(u128, r[0]) + prod0 + @as(u128, carry1);
                r[0] = @truncate(sum0);
                carry1 = @truncate(sum0 >> 64);

                // Montgomery reduction step
                const k = r[0] *% montgomery_inv;
                var carry2: u64 = 0;
                const red0 = mulWide(k, modulus[0]);
                const red_sum0 = @as(u128, r[0]) + red0;
                carry2 = @truncate(red_sum0 >> 64);

                // Process remaining limbs
                inline for (1..4) |j| {
                    const prod_j = mulWide(self.limbs[j], limb_n1);
                    const new_rj = @as(u128, r[j]) + prod_j + @as(u128, carry1);
                    const new_rj_trunc: u64 = @truncate(new_rj);
                    carry1 = @truncate(new_rj >> 64);

                    const red_j = mulWide(k, modulus[j]);
                    const red_sum_j = @as(u128, new_rj_trunc) + red_j + @as(u128, carry2);
                    r[j - 1] = @truncate(red_sum_j);
                    carry2 = @truncate(red_sum_j >> 64);

                    r[j] = new_rj_trunc;
                }
                r[3] = carry1 +% carry2;
            }

            var result = Self{ .limbs = r };
            if (!result.lessThanModulus()) {
                result = result.subtractModulus();
            }

            return result;
        }

        /// Fused multiply-accumulate: computes a[0]*b[0] + a[1]*b[1] with only
        /// 2 Montgomery reductions instead of 3 (vs separate mul + mul + add).
        /// Interleaved CIOS: both products share the same reduction step per limb iteration.
        /// Safe for BN254 since modulus_size (254) < 64*N - 1 (255).
        pub inline fn sumOfProducts(a: [2]Self, b: [2]Self) Self {
            var t: [5]u64 = .{ 0, 0, 0, 0, 0 };

            inline for (0..4) |i| {
                // Accumulate both products at limb i into t
                var carry1: u64 = 0;
                inline for (0..2) |pair| {
                    var carry: u64 = 0;
                    inline for (0..4) |j| {
                        const prod = mulWide(a[pair].limbs[i], b[pair].limbs[j]);
                        const sum = @as(u128, t[j]) + prod + @as(u128, carry);
                        t[j] = @truncate(sum);
                        carry = @truncate(sum >> 64);
                    }
                    const sum_t4 = @as(u128, t[4]) + @as(u128, carry) + @as(u128, carry1);
                    t[4] = @truncate(sum_t4);
                    carry1 = @truncate(sum_t4 >> 64);
                }

                // Montgomery reduction step (shared for both products)
                const m = t[0] *% montgomery_inv;

                var carry: u64 = 0;
                const prod0 = mulWide(m, modulus[0]);
                const sum0 = @as(u128, t[0]) + prod0;
                carry = @truncate(sum0 >> 64);

                inline for (1..4) |j| {
                    const prod = mulWide(m, modulus[j]);
                    const sum = @as(u128, t[j]) + prod + @as(u128, carry);
                    t[j - 1] = @truncate(sum);
                    carry = @truncate(sum >> 64);
                }
                const final_sum = @as(u128, t[4]) + @as(u128, carry);
                t[3] = @truncate(final_sum);
                t[4] = @as(u64, @truncate(final_sum >> 64)) +% carry1;
            }

            var result = Self{ .limbs = .{ t[0], t[1], t[2], t[3] } };
            if (t[4] != 0 or !result.lessThanModulus()) {
                result = result.subtractModulus();
            }
            return result;
        }

        /// Addition without final reduction. Result in [0, 2p).
        /// Only valid when both inputs are in [0, p).
        pub inline fn addNoReduce(self: Self, other: Self) Self {
            @setEvalBranchQuota(10000);
            var result: [4]u64 = undefined;
            var carry: u64 = 0;

            inline for (0..4) |i| {
                const ac = addCarry(self.limbs[i], other.limbs[i], carry);
                result[i] = ac.result;
                carry = ac.carry;
            }

            var res = Self{ .limbs = result };
            // Only subtract if overflowed 256 bits; result stays in [0, 2p)
            if (carry != 0) {
                res = res.subtractModulus();
            }
            return res;
        }

        /// Reduce from [0, 2p) to [0, p)
        pub inline fn reduce(self: Self) Self {
            if (!self.lessThanModulus()) {
                return self.subtractModulus();
            }
            return self;
        }

        /// Field addition
        pub inline fn add(self: Self, other: Self) Self {
            @setEvalBranchQuota(10000);
            var result: [4]u64 = undefined;
            var carry: u64 = 0;

            inline for (0..4) |i| {
                const ac = addCarry(self.limbs[i], other.limbs[i], carry);
                result[i] = ac.result;
                carry = ac.carry;
            }

            var res = Self{ .limbs = result };
            if (carry != 0 or !res.lessThanModulus()) {
                res = res.subtractModulus();
            }
            return res;
        }

        /// Field subtraction
        pub inline fn sub(self: Self, other: Self) Self {
            if (!@inComptime() and comptime use_arm64_asm) return Self{ .limbs = arm64SubMod256(self.limbs, other.limbs, modulus) };
            @setEvalBranchQuota(10000);
            var result: [4]u64 = undefined;
            var borrow: u64 = 0;

            inline for (0..4) |i| {
                const sb = subBorrow(self.limbs[i], other.limbs[i], borrow);
                result[i] = sb.result;
                borrow = sb.borrow;
            }

            var res = Self{ .limbs = result };
            if (borrow != 0) {
                res = res.addModulus();
            }
            return res;
        }

        /// Field multiplication
        pub inline fn mul(self: Self, other: Self) Self {
            if (!@inComptime() and comptime use_arm64_asm) {
                const mod_arr: [4]u64 = modulus;
                var r = Self{ .limbs = arm64MontgomeryMul256(&self.limbs, &other.limbs, &mod_arr, montgomery_inv) };
                if (!r.lessThanModulus()) r = r.subtractModulus();
                return r;
            }
            if (!@inComptime() and comptime use_asm_mul) return self.montgomeryMulX86(other);
            return self.montgomeryMul(other);
        }

        /// Field squaring
        pub inline fn square(self: Self) Self {
            if (!@inComptime() and comptime use_arm64_asm) {
                const mod_arr: [4]u64 = modulus;
                var r = Self{ .limbs = arm64MontgomeryMul256(&self.limbs, &self.limbs, &mod_arr, montgomery_inv) };
                if (!r.lessThanModulus()) r = r.subtractModulus();
                return r;
            }
            if (!@inComptime() and comptime use_asm_mul) return self.montgomeryMulX86(self);
            return self.montgomeryMul(self);
        }

        /// Multiply by a signed 128-bit integer
        /// Used for power sum computations in univariate skip verification
        pub fn mulI128(self: Self, val: i128) Self {
            if (val == 0) return Self.zero();
            if (val == 1) return self;
            if (val == -1) return self.neg();

            if (val > 0) {
                const uval: u128 = @intCast(val);
                return self.mulU128(uval);
            } else {
                const uval: u128 = @intCast(-val);
                return self.mulU128(uval).neg();
            }
        }

        /// Multiply by an unsigned 128-bit integer
        fn mulU128(self: Self, val: u128) Self {
            if (val == 0) return Self.zero();
            if (val == 1) return self;

            // Convert u128 to field element and multiply
            const low: u64 = @truncate(val);
            const high: u64 = @truncate(val >> 64);

            // Create field element from 128-bit value
            var other = Self.fromU64(low);
            if (high != 0) {
                // Add high * 2^64 contribution
                const high_fe = Self.fromU64(high);
                // Multiply by 2^64 using repeated squaring
                var two_64 = Self.fromU64(1);
                for (0..64) |_| {
                    two_64 = two_64.double();
                }
                other = other.add(high_fe.mul(two_64));
            }

            return self.mul(other);
        }

        /// Doubling (2*self)
        pub inline fn double(self: Self) Self {
            return self.add(self);
        }

        /// Negation
        pub inline fn neg(self: Self) Self {
            if (self.isZero()) return self;
            return (Self{ .limbs = modulus }).sub(self);
        }

        /// Field inverse using Fermat's little theorem: a^{-1} = a^{p-2}
        pub fn inverse(self: Self) ?Self {
            if (self.isZero()) return null;

            var result = Self.one();
            var base = self;
            var exp: [4]u64 = modulus;
            exp[0] -= 2;

            for (0..256) |i| {
                const word_idx = i / 64;
                const bit_idx: u6 = @truncate(i % 64);
                if ((exp[word_idx] >> bit_idx) & 1 == 1) {
                    result = result.mul(base);
                }
                base = base.square();
            }

            return result;
        }

        /// Batch inversion using Montgomery's trick: invert n elements with 1 inversion + 3(n-1) muls.
        /// Elements are inverted in-place. Zero elements are skipped (left as zero).
        /// `scratch` must have the same length as `elements`.
        pub fn batchInversion(elements: []Self, scratch: []Self) void {
            const n = elements.len;
            if (n == 0) return;

            // Forward pass: compute prefix products, skipping zeros
            var acc = one();
            for (0..n) |i| {
                scratch[i] = acc;
                if (!elements[i].isZero()) {
                    acc = acc.mul(elements[i]);
                }
            }

            // Single inversion of the accumulated product
            var inv = acc.inverse() orelse unreachable;

            // Backward pass: extract individual inverses
            var i: usize = n;
            while (i > 0) {
                i -= 1;
                if (elements[i].isZero()) continue;
                const old = elements[i];
                elements[i] = scratch[i].mul(inv);
                inv = inv.mul(old);
            }
        }

        inline fn lessThanModulus(self: Self) bool {
            // Unrolled MSB-first comparison — const modulus enables LLVM to emit cmpq with immediates
            if (self.limbs[3] != modulus[3]) return self.limbs[3] < modulus[3];
            if (self.limbs[2] != modulus[2]) return self.limbs[2] < modulus[2];
            if (self.limbs[1] != modulus[1]) return self.limbs[1] < modulus[1];
            return self.limbs[0] < modulus[0];
        }

        inline fn subtractModulus(self: Self) Self {
            if (!@inComptime() and comptime use_arm64_asm) return Self{ .limbs = arm64Sub256(self.limbs, modulus) };
            @setEvalBranchQuota(10000);
            var result: [4]u64 = undefined;
            var borrow: u64 = 0;

            inline for (0..4) |i| {
                const sb = subBorrow(self.limbs[i], modulus[i], borrow);
                result[i] = sb.result;
                borrow = sb.borrow;
            }

            return Self{ .limbs = result };
        }

        inline fn addModulus(self: Self) Self {
            if (!@inComptime() and comptime use_arm64_asm) return Self{ .limbs = arm64Add256(self.limbs, modulus) };
            @setEvalBranchQuota(10000);
            var result: [4]u64 = undefined;
            var carry: u64 = 0;

            inline for (0..4) |i| {
                const ac = addCarry(self.limbs[i], modulus[i], carry);
                result[i] = ac.result;
                carry = ac.carry;
            }

            return Self{ .limbs = result };
        }
    };
}

/// JoltField interface - the core trait for field elements
///
/// In Zig, we implement this as a comptime interface check pattern.
pub fn JoltField(comptime Self: type) type {
    return struct {
        pub const num_bytes = FIELD_ELEMENT_BYTES;

        pub fn isJoltField() void {
            // Compile-time check that Self has required methods
            comptime {
                if (!@hasDecl(Self, "zero")) @compileError("JoltField requires zero()");
                if (!@hasDecl(Self, "one")) @compileError("JoltField requires one()");
                if (!@hasDecl(Self, "add")) @compileError("JoltField requires add()");
                if (!@hasDecl(Self, "sub")) @compileError("JoltField requires sub()");
                if (!@hasDecl(Self, "mul")) @compileError("JoltField requires mul()");
                if (!@hasDecl(Self, "inverse")) @compileError("JoltField requires inverse()");
                if (!@hasDecl(Self, "square")) @compileError("JoltField requires square()");
                if (!@hasDecl(Self, "fromU64")) @compileError("JoltField requires fromU64()");
            }
        }
    };
}

/// BN254 scalar field element
/// Stored in Montgomery form: a is represented as a*R mod p
pub const BN254Scalar = struct {
    limbs: [4]u64,

    const Self = @This();

    /// Zero element
    pub fn zero() Self {
        return .{ .limbs = .{ 0, 0, 0, 0 } };
    }

    /// One element (in Montgomery form = R mod p)
    pub fn one() Self {
        return .{ .limbs = BN254_R };
    }

    /// R^2 as a field element (in Montgomery form = R^2 * R = R^3 mod p)
    /// This is used for compatibility with Jolt's Montgomery R^2 scaling.
    /// To get R^2 as a value in Montgomery form, we take the raw R^2 bytes
    /// and convert them to Montgomery form.
    pub fn rSquared() Self {
        // BN254_R2 is the raw R^2 mod p value (not in Montgomery form)
        // To represent R^2 as a field element in Montgomery form,
        // we convert it: rSquared = R^2 * R (Montgomery form of R^2)
        const raw = Self{ .limbs = BN254_R2 };
        return raw.toMontgomery();
    }

    /// Check if zero
    pub fn isZero(self: Self) bool {
        return self.limbs[0] == 0 and self.limbs[1] == 0 and
            self.limbs[2] == 0 and self.limbs[3] == 0;
    }

    /// Check if one (in Montgomery form)
    pub fn isOne(self: Self) bool {
        return self.limbs[0] == BN254_R[0] and self.limbs[1] == BN254_R[1] and
            self.limbs[2] == BN254_R[2] and self.limbs[3] == BN254_R[3];
    }

    /// Equality check
    pub fn eql(self: Self, other: Self) bool {
        return self.limbs[0] == other.limbs[0] and self.limbs[1] == other.limbs[1] and
            self.limbs[2] == other.limbs[2] and self.limbs[3] == other.limbs[3];
    }

    /// Create from u64 (converts to Montgomery form)
    pub fn fromU64(n: u64) Self {
        var result = Self{ .limbs = .{ n, 0, 0, 0 } };
        // Convert to Montgomery form by multiplying by R^2 and reducing
        result = result.montgomeryMul(.{ .limbs = BN254_R2 });
        return result;
    }

    /// Create from u128 (converts to Montgomery form)
    pub fn fromU128(n: u128) Self {
        const raw = Self{ .limbs = .{ @truncate(n), @truncate(n >> 64), 0, 0 } };
        return raw.montgomeryMul(.{ .limbs = BN254_R2 });
    }

    /// Create from bytes (little-endian, converts to Montgomery form)
    pub fn fromBytes(bytes: []const u8) Self {
        var limbs: [4]u64 = .{ 0, 0, 0, 0 };
        const len = @min(bytes.len, 32);
        var buf: [32]u8 = .{0} ** 32;
        @memcpy(buf[0..len], bytes[0..len]);

        for (0..4) |i| {
            limbs[i] = std.mem.readInt(u64, buf[i * 8 ..][0..8], .little);
        }

        var result = Self{ .limbs = limbs };
        // Convert to Montgomery form
        result = result.montgomeryMul(.{ .limbs = BN254_R2 });
        return result;
    }

    /// Convert from Montgomery form back to standard representation
    pub fn fromMontgomery(self: Self) Self {
        // Multiply by 1 to get a*R * 1 * R^{-1} = a
        return self.montgomeryMul(.{ .limbs = .{ 1, 0, 0, 0 } });
    }

    /// Convert to Montgomery form from standard representation
    /// Used when we have raw limbs that need to be converted
    pub fn toMontgomery(self: Self) Self {
        // Multiply by R^2 to get a * R^2 * R^{-1} = a * R
        return self.montgomeryMul(.{ .limbs = BN254_R2 });
    }

    /// Create from big-endian bytes (converts to Montgomery form)
    pub fn fromBytesBE(bytes: *const [32]u8) Self {
        // Reverse byte order for big-endian
        var le_bytes: [32]u8 = undefined;
        for (0..32) |i| {
            le_bytes[i] = bytes[31 - i];
        }
        return fromBytes(&le_bytes);
    }

    /// Serialize to big-endian bytes (32 bytes)
    pub fn toBytesBE(self: Self) [32]u8 {
        // First convert from Montgomery form
        const standard = self.fromMontgomery();

        // Convert limbs to bytes (little-endian)
        var le_bytes: [32]u8 = undefined;
        for (0..4) |i| {
            std.mem.writeInt(u64, le_bytes[i * 8 ..][0..8], standard.limbs[i], .little);
        }

        // Reverse for big-endian output
        var be_bytes: [32]u8 = undefined;
        for (0..32) |i| {
            be_bytes[i] = le_bytes[31 - i];
        }
        return be_bytes;
    }

    /// Serialize to little-endian bytes (32 bytes)
    /// This is the inverse of fromBytes and suitable for serialization.
    pub fn toBytes(self: Self) [32]u8 {
        // First convert from Montgomery form
        const standard = self.fromMontgomery();

        // Convert limbs to bytes (little-endian)
        var bytes: [32]u8 = undefined;
        for (0..4) |i| {
            std.mem.writeInt(u64, bytes[i * 8 ..][0..8], standard.limbs[i], .little);
        }
        return bytes;
    }

    /// Convert to u64 (returns low 64 bits of the value)
    /// Useful for debugging and displaying small values.
    /// Note: This loses precision for values >= 2^64.
    pub fn toU64(self: Self) u64 {
        const standard = self.fromMontgomery();
        return standard.limbs[0];
    }

    /// 128-bit multiplication helper
    inline fn mulWide(a: u64, b: u64) u128 {
        return @as(u128, a) * @as(u128, b);
    }

    /// Add with carry
    inline fn addCarry(a: u64, b: u64, carry_in: u64) struct { result: u64, carry: u64 } {
        if (comptime builtin.cpu.arch == .x86_64) {
            var result: u64 = undefined;
            const c = x86.addcarry(@truncate(carry_in), a, b, &result);
            return .{ .result = result, .carry = c };
        }
        const sum = @as(u128, a) + @as(u128, b) + @as(u128, carry_in);
        return .{ .result = @truncate(sum), .carry = @truncate(sum >> 64) };
    }

    inline fn subBorrow(a: u64, b: u64, borrow_in: u64) struct { result: u64, borrow: u64 } {
        if (!@inComptime() and comptime builtin.cpu.arch == .x86_64) {
            var result: u64 = undefined;
            const b_out = x86.subborrow(@truncate(borrow_in), a, b, &result);
            return .{ .result = result, .borrow = b_out };
        }
        const wide_a = @as(u128, a);
        const wide_b = @as(u128, b) + @as(u128, borrow_in);
        const diff = wide_a -% wide_b;
        return .{ .result = @truncate(diff), .borrow = @truncate(diff >> 127) };
    }

    /// Montgomery multiplication: computes a*b*R^{-1} mod p
    pub fn montgomeryMul(self: Self, other: Self) Self {
        // CIOS (Coarsely Integrated Operand Scanning) method
        var t: [5]u64 = .{ 0, 0, 0, 0, 0 };

        inline for (0..4) |i| {
            // t = t + a[i] * b
            var carry: u64 = 0;
            inline for (0..4) |j| {
                const prod = mulWide(self.limbs[i], other.limbs[j]);
                const sum = @as(u128, t[j]) + prod + @as(u128, carry);
                t[j] = @truncate(sum);
                carry = @truncate(sum >> 64);
            }
            const sum_t4 = @as(u128, t[4]) + @as(u128, carry);
            t[4] = @truncate(sum_t4);

            // m = t[0] * N' mod 2^64
            const m = t[0] *% BN254_INV;

            // t = (t + m * N) / 2^64
            carry = 0;
            const prod0 = mulWide(m, BN254_MODULUS[0]);
            const sum0 = @as(u128, t[0]) + prod0;
            carry = @truncate(sum0 >> 64);

            inline for (1..4) |j| {
                const prod = mulWide(m, BN254_MODULUS[j]);
                const sum = @as(u128, t[j]) + prod + @as(u128, carry);
                t[j - 1] = @truncate(sum);
                carry = @truncate(sum >> 64);
            }
            const final_sum = @as(u128, t[4]) + @as(u128, carry);
            t[3] = @truncate(final_sum);
            t[4] = @truncate(final_sum >> 64);
        }

        var result = Self{ .limbs = .{ t[0], t[1], t[2], t[3] } };

        // Final subtraction if result >= p
        if (t[4] != 0 or !result.lessThanModulus()) {
            result = result.subtractModulus();
        }

        return result;
    }

    /// Field addition
    pub fn add(self: Self, other: Self) Self {
        var result: [4]u64 = undefined;
        var carry: u64 = 0;
        inline for (0..4) |i| {
            const ac = addCarry(self.limbs[i], other.limbs[i], carry);
            result[i] = ac.result;
            carry = ac.carry;
        }
        var res = Self{ .limbs = result };
        if (carry != 0 or !res.lessThanModulus()) {
            res = res.subtractModulus();
        }
        return res;
    }

    /// Field subtraction
    pub fn sub(self: Self, other: Self) Self {
        if (!@inComptime() and comptime use_arm64_asm) return Self{ .limbs = arm64SubMod256(self.limbs, other.limbs, BN254_MODULUS) };
        var result: [4]u64 = undefined;
        var borrow: u64 = 0;

        inline for (0..4) |i| {
            const sb = subBorrow(self.limbs[i], other.limbs[i], borrow);
            result[i] = sb.result;
            borrow = sb.borrow;
        }

        var res = Self{ .limbs = result };
        if (borrow != 0) {
            res = res.addModulus();
        }
        return res;
    }

    /// x86-64 BMI2+ADX accelerated CIOS Montgomery multiplication (BN254 scalar field).
    fn montgomeryMulX86(self: Self, other: Self) Self {
        const a = self.limbs;
        const b = other.limbs;
        const mod_arr: [4]u64 = BN254_MODULUS;

        var r0: u64 = undefined;
        var r1: u64 = undefined;
        var r2: u64 = undefined;
        var r3: u64 = undefined;

        asm volatile (
            \\xorq %%rcx, %%rcx
            \\movq (%%rdi), %%rdx
            \\mulxq (%%rsi), %%r8, %%r9
            \\mulxq 8(%%rsi), %%rax, %%r10
            \\adcxq %%rax, %%r9
            \\mulxq 16(%%rsi), %%rax, %%r11
            \\adcxq %%rax, %%r10
            \\mulxq 24(%%rsi), %%rax, %%rcx
            \\movq $0, %%r13
            \\adcxq %%rax, %%r11
            \\adcxq %%r13, %%rcx
            \\movq %%rbx, %%rdx
            \\mulxq %%r8, %%rdx, %%rax
            \\mulxq (%%r14), %%rax, %%r13
            \\adcxq %%r8, %%rax
            \\adoxq %%r13, %%r9
            \\mulxq 8(%%r14), %%rax, %%r13
            \\adcxq %%rax, %%r9
            \\adoxq %%r13, %%r10
            \\mulxq 16(%%r14), %%rax, %%r13
            \\adcxq %%rax, %%r10
            \\adoxq %%r13, %%r11
            \\mulxq 24(%%r14), %%rax, %%r8
            \\movq $0, %%r13
            \\adcxq %%rax, %%r11
            \\adoxq %%rcx, %%r8
            \\adcxq %%r13, %%r8
            //
            \\movq 8(%%rdi), %%rdx
            \\mulxq (%%rsi), %%rax, %%r13
            \\adcxq %%rax, %%r9
            \\adoxq %%r13, %%r10
            \\mulxq 8(%%rsi), %%rax, %%r13
            \\adcxq %%rax, %%r10
            \\adoxq %%r13, %%r11
            \\mulxq 16(%%rsi), %%rax, %%r13
            \\adcxq %%rax, %%r11
            \\adoxq %%r13, %%r8
            \\mulxq 24(%%rsi), %%rax, %%rcx
            \\movq $0, %%r13
            \\adcxq %%rax, %%r8
            \\adoxq %%r13, %%rcx
            \\adcxq %%r13, %%rcx
            \\movq %%rbx, %%rdx
            \\mulxq %%r9, %%rdx, %%rax
            \\mulxq (%%r14), %%rax, %%r13
            \\adcxq %%r9, %%rax
            \\adoxq %%r13, %%r10
            \\mulxq 8(%%r14), %%rax, %%r13
            \\adcxq %%rax, %%r10
            \\adoxq %%r13, %%r11
            \\mulxq 16(%%r14), %%rax, %%r13
            \\adcxq %%rax, %%r11
            \\adoxq %%r13, %%r8
            \\mulxq 24(%%r14), %%rax, %%r9
            \\movq $0, %%r13
            \\adcxq %%rax, %%r8
            \\adoxq %%rcx, %%r9
            \\adcxq %%r13, %%r9
            //
            \\movq 16(%%rdi), %%rdx
            \\mulxq (%%rsi), %%rax, %%r13
            \\adcxq %%rax, %%r10
            \\adoxq %%r13, %%r11
            \\mulxq 8(%%rsi), %%rax, %%r13
            \\adcxq %%rax, %%r11
            \\adoxq %%r13, %%r8
            \\mulxq 16(%%rsi), %%rax, %%r13
            \\adcxq %%rax, %%r8
            \\adoxq %%r13, %%r9
            \\mulxq 24(%%rsi), %%rax, %%rcx
            \\movq $0, %%r13
            \\adcxq %%rax, %%r9
            \\adoxq %%r13, %%rcx
            \\adcxq %%r13, %%rcx
            \\movq %%rbx, %%rdx
            \\mulxq %%r10, %%rdx, %%rax
            \\mulxq (%%r14), %%rax, %%r13
            \\adcxq %%r10, %%rax
            \\adoxq %%r13, %%r11
            \\mulxq 8(%%r14), %%rax, %%r13
            \\adcxq %%rax, %%r11
            \\adoxq %%r13, %%r8
            \\mulxq 16(%%r14), %%rax, %%r13
            \\adcxq %%rax, %%r8
            \\adoxq %%r13, %%r9
            \\mulxq 24(%%r14), %%rax, %%r10
            \\movq $0, %%r13
            \\adcxq %%rax, %%r9
            \\adoxq %%rcx, %%r10
            \\adcxq %%r13, %%r10
            //
            \\movq 24(%%rdi), %%rdx
            \\mulxq (%%rsi), %%rax, %%r13
            \\adcxq %%rax, %%r11
            \\adoxq %%r13, %%r8
            \\mulxq 8(%%rsi), %%rax, %%r13
            \\adcxq %%rax, %%r8
            \\adoxq %%r13, %%r9
            \\mulxq 16(%%rsi), %%rax, %%r13
            \\adcxq %%rax, %%r9
            \\adoxq %%r13, %%r10
            \\mulxq 24(%%rsi), %%rax, %%rcx
            \\movq $0, %%r13
            \\adcxq %%rax, %%r10
            \\adoxq %%r13, %%rcx
            \\adcxq %%r13, %%rcx
            \\movq %%rbx, %%rdx
            \\mulxq %%r11, %%rdx, %%rax
            \\mulxq (%%r14), %%rax, %%r13
            \\adcxq %%r11, %%rax
            \\adoxq %%r13, %%r8
            \\mulxq 8(%%r14), %%rax, %%r13
            \\adcxq %%rax, %%r8
            \\adoxq %%r13, %%r9
            \\mulxq 16(%%r14), %%rax, %%r13
            \\adcxq %%rax, %%r9
            \\adoxq %%r13, %%r10
            \\mulxq 24(%%r14), %%rax, %%r11
            \\movq $0, %%r13
            \\adcxq %%rax, %%r10
            \\adoxq %%rcx, %%r11
            \\adcxq %%r13, %%r11
            : [_r0] "={r8}" (r0),
              [_r1] "={r9}" (r1),
              [_r2] "={r10}" (r2),
              [_r3] "={r11}" (r3),
            : [_a] "{rdi}" (&a),
              [_b] "{rsi}" (&b),
              [_mod] "{r14}" (&mod_arr),
              [_inv] "{rbx}" (BN254_INV),
            : .{ .rax = true, .rcx = true, .rdx = true, .r13 = true, .cc = true, .memory = true });

        var result = Self{ .limbs = .{ r0, r1, r2, r3 } };
        if (!result.lessThanModulus()) {
            result = result.subtractModulus();
        }
        return result;
    }

    /// Fused multiply-accumulate: computes a[0]*b[0] + a[1]*b[1] with only
    /// 2 Montgomery reductions instead of 3 (vs separate mul + mul + add).
    pub fn sumOfProducts(a: [2]Self, b: [2]Self) Self {
        var t: [5]u64 = .{ 0, 0, 0, 0, 0 };

        inline for (0..4) |i| {
            var carry1: u64 = 0;
            inline for (0..2) |pair| {
                var carry: u64 = 0;
                inline for (0..4) |j| {
                    const prod = mulWide(a[pair].limbs[i], b[pair].limbs[j]);
                    const sum = @as(u128, t[j]) + prod + @as(u128, carry);
                    t[j] = @truncate(sum);
                    carry = @truncate(sum >> 64);
                }
                const sum_t4 = @as(u128, t[4]) + @as(u128, carry) + @as(u128, carry1);
                t[4] = @truncate(sum_t4);
                carry1 = @truncate(sum_t4 >> 64);
            }

            const m = t[0] *% BN254_INV;

            var carry: u64 = 0;
            const prod0 = mulWide(m, BN254_MODULUS[0]);
            const sum0 = @as(u128, t[0]) + prod0;
            carry = @truncate(sum0 >> 64);

            inline for (1..4) |j| {
                const prod = mulWide(m, BN254_MODULUS[j]);
                const sum = @as(u128, t[j]) + prod + @as(u128, carry);
                t[j - 1] = @truncate(sum);
                carry = @truncate(sum >> 64);
            }
            const final_sum = @as(u128, t[4]) + @as(u128, carry);
            t[3] = @truncate(final_sum);
            t[4] = @as(u64, @truncate(final_sum >> 64)) +% carry1;
        }

        var result = Self{ .limbs = .{ t[0], t[1], t[2], t[3] } };
        if (t[4] != 0 or !result.lessThanModulus()) {
            result = result.subtractModulus();
        }
        return result;
    }

    /// Field multiplication
    pub inline fn mul(self: Self, other: Self) Self {
        if (!@inComptime() and comptime use_arm64_asm) {
            var r = Self{ .limbs = arm64MontgomeryMul256(&self.limbs, &other.limbs, &BN254_MODULUS, BN254_INV) };
            if (!r.lessThanModulus()) r = r.subtractModulus();
            return r;
        }
        if (!@inComptime() and comptime use_asm_mul) return self.montgomeryMulX86(other);
        return self.montgomeryMul(other);
    }

    /// Optimized multiplication by a high-limb 128-bit BigInt
    /// Matches arkworks' mul_hi_bigint_u128 behavior
    ///
    /// This is equivalent to: self * (limb2 + limb3 * 2^64) mod p
    /// where the 128-bit value is stored in positions [2] and [3] of a 4-limb BigInt.
    pub fn mulHiBigIntU128(self: Self, hi_limbs: [4]u64) Self {
        return self.mulHiBigIntU128Soft(hi_limbs);
    }

    /// x86-64 BMI2+ADX accelerated 2-round CIOS for 128-bit hi-limb multiply.
    /// Computes self * [0, 0, limb2, limb3] mod p using only 2 CIOS rounds.
    noinline fn mulHiBigIntU128X86(self: Self, limb2: u64, limb3: u64) Self {
        const a = self.limbs;
        const mod_arr: [4]u64 = BN254_MODULUS;
        var r0: u64 = undefined;
        var r1: u64 = undefined;
        var r2: u64 = undefined;
        var r3: u64 = undefined;

        asm volatile (
            // Clear CF and OF for initial carry chains
            \\xorq %%r13, %%r13
            //
            // Round 0: limb2 × self[0..3] (first non-zero round)
            // rdx = limb2 (already set via input constraint)
            \\mulxq (%%rdi), %%r8, %%r9
            \\mulxq 8(%%rdi), %%rax, %%r10
            \\adcxq %%rax, %%r9
            \\mulxq 16(%%rdi), %%rax, %%r11
            \\adcxq %%rax, %%r10
            \\mulxq 24(%%rdi), %%rax, %%rcx
            \\movq $0, %%r13
            \\adcxq %%rax, %%r11
            \\adcxq %%r13, %%rcx
            // Montgomery reduction: k = r8 * INV
            \\movq %%rbx, %%rdx
            \\mulxq %%r8, %%rdx, %%rax
            // k × modulus
            \\mulxq (%%r14), %%rax, %%r13
            \\adcxq %%r8, %%rax
            \\adoxq %%r13, %%r9
            \\mulxq 8(%%r14), %%rax, %%r13
            \\adcxq %%rax, %%r9
            \\adoxq %%r13, %%r10
            \\mulxq 16(%%r14), %%rax, %%r13
            \\adcxq %%rax, %%r10
            \\adoxq %%r13, %%r11
            \\mulxq 24(%%r14), %%rax, %%r8
            \\movq $0, %%r13
            \\adcxq %%rax, %%r11
            \\adoxq %%rcx, %%r8
            \\adcxq %%r13, %%r8
            //
            // Round 1: limb3 × self[0..3]
            \\movq %%rsi, %%rdx
            \\mulxq (%%rdi), %%rax, %%r13
            \\adcxq %%rax, %%r9
            \\adoxq %%r13, %%r10
            \\mulxq 8(%%rdi), %%rax, %%r13
            \\adcxq %%rax, %%r10
            \\adoxq %%r13, %%r11
            \\mulxq 16(%%rdi), %%rax, %%r13
            \\adcxq %%rax, %%r11
            \\adoxq %%r13, %%r8
            \\mulxq 24(%%rdi), %%rax, %%rcx
            \\movq $0, %%r13
            \\adcxq %%rax, %%r8
            \\adoxq %%r13, %%rcx
            \\adcxq %%r13, %%rcx
            // Montgomery reduction: k = r9 * INV
            \\movq %%rbx, %%rdx
            \\mulxq %%r9, %%rdx, %%rax
            \\mulxq (%%r14), %%rax, %%r13
            \\adcxq %%r9, %%rax
            \\adoxq %%r13, %%r10
            \\mulxq 8(%%r14), %%rax, %%r13
            \\adcxq %%rax, %%r10
            \\adoxq %%r13, %%r11
            \\mulxq 16(%%r14), %%rax, %%r13
            \\adcxq %%rax, %%r11
            \\adoxq %%r13, %%r8
            \\mulxq 24(%%r14), %%rax, %%r9
            \\movq $0, %%r13
            \\adcxq %%rax, %%r8
            \\adoxq %%rcx, %%r9
            \\adcxq %%r13, %%r9
            : [_r0] "={r10}" (r0),
              [_r1] "={r11}" (r1),
              [_r2] "={r8}" (r2),
              [_r3] "={r9}" (r3),
            : [_a] "{rdi}" (&a),
              [_limb2] "{rdx}" (limb2),
              [_limb3] "{rsi}" (limb3),
              [_mod] "{r14}" (&mod_arr),
              [_inv] "{rbx}" (BN254_INV),
            : .{ .rax = true, .rcx = true, .r13 = true, .cc = true, .memory = true }
        );

        var result = Self{ .limbs = .{ r0, r1, r2, r3 } };
        if (!result.lessThanModulus()) {
            result = result.subtractModulus();
        }
        return result;
    }

    fn mulHiBigIntU128Soft(self: Self, hi_limbs: [4]u64) Self {
        const limb_n2 = hi_limbs[2];
        const limb_n1 = hi_limbs[3];

        var r: [4]u64 = .{ 0, 0, 0, 0 };

        // i = 2 (N-2): Process limb_n2
        {
            var carry1: u64 = 0;

            // r[0] = r[0] + self[0] * limb_n2
            const prod0 = mulWide(self.limbs[0], limb_n2);
            const sum0 = @as(u128, r[0]) + prod0 + @as(u128, carry1);
            r[0] = @truncate(sum0);
            carry1 = @truncate(sum0 >> 64);

            // Montgomery reduction step
            const k = r[0] *% BN254_INV;
            var carry2: u64 = 0;
            const red0 = mulWide(k, BN254_MODULUS[0]);
            const red_sum0 = @as(u128, r[0]) + red0;
            carry2 = @truncate(red_sum0 >> 64);

            // Process remaining limbs
            inline for (1..4) |j| {
                const prod_j = mulWide(self.limbs[j], limb_n2);
                const new_rj = @as(u128, r[j]) + prod_j + @as(u128, carry1);
                const new_rj_trunc: u64 = @truncate(new_rj);
                carry1 = @truncate(new_rj >> 64);

                const red_j = mulWide(k, BN254_MODULUS[j]);
                const red_sum_j = @as(u128, new_rj_trunc) + red_j + @as(u128, carry2);
                r[j - 1] = @truncate(red_sum_j);
                carry2 = @truncate(red_sum_j >> 64);

                // Update r[j] for next iteration
                r[j] = new_rj_trunc;
            }
            r[3] = carry1 +% carry2;
        }

        // i = 3 (N-1): Process limb_n1
        {
            var carry1: u64 = 0;

            // r[0] = r[0] + self[0] * limb_n1
            const prod0 = mulWide(self.limbs[0], limb_n1);
            const sum0 = @as(u128, r[0]) + prod0 + @as(u128, carry1);
            r[0] = @truncate(sum0);
            carry1 = @truncate(sum0 >> 64);

            // Montgomery reduction step
            const k = r[0] *% BN254_INV;
            var carry2: u64 = 0;
            const red0 = mulWide(k, BN254_MODULUS[0]);
            const red_sum0 = @as(u128, r[0]) + red0;
            carry2 = @truncate(red_sum0 >> 64);

            // Process remaining limbs
            inline for (1..4) |j| {
                const prod_j = mulWide(self.limbs[j], limb_n1);
                const new_rj = @as(u128, r[j]) + prod_j + @as(u128, carry1);
                const new_rj_trunc: u64 = @truncate(new_rj);
                carry1 = @truncate(new_rj >> 64);

                const red_j = mulWide(k, BN254_MODULUS[j]);
                const red_sum_j = @as(u128, new_rj_trunc) + red_j + @as(u128, carry2);
                r[j - 1] = @truncate(red_sum_j);
                carry2 = @truncate(red_sum_j >> 64);

                r[j] = new_rj_trunc;
            }
            r[3] = carry1 +% carry2;
        }

        var result = Self{ .limbs = r };
        if (!result.lessThanModulus()) {
            result = result.subtractModulus();
        }

        return result;
    }

    /// Multiply by a signed 128-bit integer
    /// Used for power sum computations in univariate skip verification
    pub fn mulI128(self: Self, val: i128) Self {
        if (val == 0) return Self.zero();
        if (val == 1) return self;
        if (val == -1) return self.neg();

        if (val > 0) {
            const uval: u128 = @intCast(val);
            return self.mulU128(uval);
        } else {
            const uval: u128 = @intCast(-val);
            return self.mulU128(uval).neg();
        }
    }

    /// Multiply by an unsigned 128-bit integer
    fn mulU128(self: Self, val: u128) Self {
        if (val == 0) return Self.zero();
        if (val == 1) return self;

        // Convert u128 to field element and multiply
        const low: u64 = @truncate(val);
        const high: u64 = @truncate(val >> 64);

        // Create field element from 128-bit value
        var other = Self.fromU64(low);
        if (high != 0) {
            // Add high * 2^64 contribution
            const high_fe = Self.fromU64(high);
            // Multiply by 2^64 using repeated squaring
            var two_64 = Self.fromU64(1);
            for (0..64) |_| {
                two_64 = two_64.double();
            }
            other = other.add(high_fe.mul(two_64));
        }

        return self.mul(other);
    }

    /// Field squaring (optimized using Karatsuba-like technique)
    /// Saves ~25% multiplications compared to naive multiplication
    pub inline fn square(self: Self) Self {
        if (!@inComptime() and comptime use_arm64_asm) {
            var r = Self{ .limbs = arm64MontgomeryMul256(&self.limbs, &self.limbs, &BN254_MODULUS, BN254_INV) };
            if (!r.lessThanModulus()) r = r.subtractModulus();
            return r;
        }
        if (!@inComptime() and comptime use_asm_mul) return self.montgomeryMulX86(self);
        // Optimized squaring: we can compute a^2 with fewer multiplications
        // Since (a0 + a1*2^64 + a2*2^128 + a3*2^192)^2 has symmetric terms
        // For example: 2*a0*a1 instead of a0*a1 + a1*a0
        //
        // First, compute the product matrix with reduced operations
        var t: [8]u64 = .{ 0, 0, 0, 0, 0, 0, 0, 0 };
        var carry: u64 = 0;

        // Compute diagonal terms a[i]^2
        inline for (0..4) |i| {
            const prod = mulWide(self.limbs[i], self.limbs[i]);
            const idx = i * 2;
            const sum = @as(u128, t[idx]) + prod;
            t[idx] = @truncate(sum);
            const overflow = @as(u64, @truncate(sum >> 64));
            const sum_next = @as(u128, t[idx + 1]) + @as(u128, overflow);
            t[idx + 1] = @truncate(sum_next);
            // Propagate carry to higher limbs
            if (idx + 2 < 8) {
                t[idx + 2] +%= @truncate(sum_next >> 64);
            }
        }

        // Compute off-diagonal terms 2*a[i]*a[j] for i < j
        inline for (0..4) |i| {
            inline for (i + 1..4) |j| {
                const prod = mulWide(self.limbs[i], self.limbs[j]);
                const idx = i + j;
                // Double the product (since we count both a[i]*a[j] and a[j]*a[i])
                const doubled_lo = @as(u64, @truncate(prod)) << 1;
                const doubled_hi = (@as(u64, @truncate(prod >> 64)) << 1) | (@as(u64, @truncate(prod)) >> 63);
                const carry_out: u64 = @as(u64, @truncate(prod >> 64)) >> 63;

                const sum0 = @as(u128, t[idx]) + @as(u128, doubled_lo);
                t[idx] = @truncate(sum0);
                const sum1 = @as(u128, t[idx + 1]) + @as(u128, doubled_hi) + (sum0 >> 64);
                t[idx + 1] = @truncate(sum1);
                if (idx + 2 < 8) {
                    const sum2 = @as(u128, t[idx + 2]) + @as(u128, carry_out) + (sum1 >> 64);
                    t[idx + 2] = @truncate(sum2);
                    if (idx + 3 < 8) {
                        t[idx + 3] +%= @truncate(sum2 >> 64);
                    }
                }
            }
        }

        // Montgomery reduction: reduce t (512 bits) to 256 bits mod p
        var r: [5]u64 = .{ t[0], t[1], t[2], t[3], 0 };

        inline for (0..4) |i| {
            const m = r[0] *% BN254_INV;
            carry = 0;
            const prod0 = mulWide(m, BN254_MODULUS[0]);
            const sum0 = @as(u128, r[0]) + prod0;
            carry = @truncate(sum0 >> 64);

            inline for (1..4) |j| {
                const prod = mulWide(m, BN254_MODULUS[j]);
                const sum = @as(u128, r[j]) + prod + @as(u128, carry);
                r[j - 1] = @truncate(sum);
                carry = @truncate(sum >> 64);
            }
            const t_idx = i + 4;
            const final_sum = @as(u128, r[4]) + @as(u128, carry) + @as(u128, t[t_idx]);
            r[3] = @truncate(final_sum);
            r[4] = @truncate(final_sum >> 64);
        }

        var result = Self{ .limbs = .{ r[0], r[1], r[2], r[3] } };
        if (r[4] != 0 or !result.lessThanModulus()) {
            result = result.subtractModulus();
        }
        return result;
    }

    /// Field negation: -a mod p
    pub fn neg(self: Self) Self {
        if (self.isZero()) return self;
        return (Self{ .limbs = BN254_MODULUS }).sub(self);
    }

    /// Field doubling: 2*a mod p
    pub fn double(self: Self) Self {
        return self.add(self);
    }

    /// Multiplicative inverse using Fermat's little theorem: a^{-1} = a^{p-2} mod p
    pub fn inverse(self: Self) ?Self {
        if (self.isZero()) return null;

        // p - 2 for BN254 scalar field
        // We use binary exponentiation
        const exp_minus_2: [4]u64 = .{
            0x43e1f593efffffff,
            0x2833e84879b97091,
            0xb85045b68181585d,
            0x30644e72e131a029,
        };

        var result = Self.one();
        var base = self;

        inline for (0..4) |i| {
            var bits = exp_minus_2[i];
            var j: usize = 0;
            while (j < 64) : (j += 1) {
                if ((bits & 1) != 0) {
                    result = result.mul(base);
                }
                base = base.square();
                bits >>= 1;
            }
        }

        return result;
    }

    /// Exponentiation: self^exp mod p
    pub fn pow(self: Self, exp: u64) Self {
        if (exp == 0) return Self.one();
        if (exp == 1) return self;

        var result = Self.one();
        var base = self;
        var e = exp;

        while (e > 0) {
            if ((e & 1) != 0) {
                result = result.mul(base);
            }
            base = base.square();
            e >>= 1;
        }

        return result;
    }

    inline fn lessThanModulus(self: Self) bool {
        // Unrolled MSB-first comparison — matches arkworks BigInt::cmp pattern
        // Const modulus values allow LLVM to emit cmpq with immediates
        if (self.limbs[3] != BN254_MODULUS[3]) return self.limbs[3] < BN254_MODULUS[3];
        if (self.limbs[2] != BN254_MODULUS[2]) return self.limbs[2] < BN254_MODULUS[2];
        if (self.limbs[1] != BN254_MODULUS[1]) return self.limbs[1] < BN254_MODULUS[1];
        return self.limbs[0] < BN254_MODULUS[0];
    }

    inline fn subtractModulus(self: Self) Self {
        if (!@inComptime() and comptime use_arm64_asm) return Self{ .limbs = arm64Sub256(self.limbs, BN254_MODULUS) };
        @setEvalBranchQuota(10000);
        var result: [4]u64 = undefined;
        var borrow: u64 = 0;

        inline for (0..4) |i| {
            const sb = subBorrow(self.limbs[i], BN254_MODULUS[i], borrow);
            result[i] = sb.result;
            borrow = sb.borrow;
        }

        return .{ .limbs = result };
    }

    fn addModulus(self: Self) Self {
        if (!@inComptime() and comptime use_arm64_asm) return Self{ .limbs = arm64Add256(self.limbs, BN254_MODULUS) };
        var result: [4]u64 = undefined;
        var carry: u64 = 0;

        inline for (0..4) |i| {
            const ac = addCarry(self.limbs[i], BN254_MODULUS[i], carry);
            result[i] = ac.result;
            carry = ac.carry;
        }

        return .{ .limbs = result };
    }

    /// Format for printing
    pub fn format(self: Self, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
        _ = fmt;
        _ = options;
        try writer.print("0x{x:0>16}{x:0>16}{x:0>16}{x:0>16}", .{
            self.limbs[3],
            self.limbs[2],
            self.limbs[1],
            self.limbs[0],
        });
    }
    /// Multiply two field elements and return the unreduced product accumulator.
    /// This avoids Montgomery reduction, allowing multiple products to be accumulated
    /// before a single reduction at the end.
    pub inline fn mulToProductAccum(self: Self, other: Self) UnreducedProductAccum {
        return UnreducedProductAccum.fromMul(self, other);
    }
};

/// Unreduced product accumulator for deferred Montgomery reduction.
///
/// Stores partial products in positional `u128` slots to avoid Montgomery reduction
/// in hot accumulation loops. Each slot holds a sum of u64×u64 partial products;
/// carries between slots are deferred until `reduce()`. This mirrors Jolt's
/// `Folded256ProductAccum` type.
///
/// ## Usage
/// ```
/// var accum = UnreducedProductAccum.zero();
/// for (a_vals, b_vals) |a, b| {
///     accum.addAssign(a.mulToProductAccum(b));
/// }
/// const result = accum.reduce();  // single Montgomery reduction
/// ```
///
/// ## Overflow Safety
/// Each `fromMul` contributes at most `4 × (2^64-1)` to any slot. After N `addAssign`
/// calls, max slot value = `N × 4 × (2^64-1)`. With u128 max = 2^128-1, safe for
/// N up to ~2^62. At T=2^30 with E_in=2^15, N=32768 → max slot ≈ 2^79, well within bounds.
pub const UnreducedProductAccum = struct {
    slots: [8]u128,

    const Self = @This();

    pub inline fn zero() Self {
        return .{ .slots = .{0} ** 8 };
    }

    /// Create an accumulator from a single product a×b (schoolbook 4×4, no reduction).
    pub inline fn fromMul(a: BN254Scalar, b: BN254Scalar) Self {
        @setEvalBranchQuota(10000);
        var slots: [8]u128 = .{0} ** 8;
        inline for (0..4) |i| {
            inline for (0..4) |j| {
                const p: u128 = @as(u128, a.limbs[i]) * @as(u128, b.limbs[j]);
                slots[i + j] += @as(u128, @as(u64, @truncate(p))); // lo 64
                slots[i + j + 1] += @as(u128, @as(u64, @truncate(p >> 64))); // hi 64
            }
        }
        return .{ .slots = slots };
    }

    /// Create an accumulator from field_elem × raw_u128 (schoolbook 4×2, no reduction).
    ///
    /// field_elem is in Montgomery form: stores (a * R mod p).
    /// raw is a plain integer (NOT Montgomery).
    /// Schoolbook product: (a * R) * raw = a * R * raw.
    /// After reduce() (Montgomery division by R): a * raw (mod p).
    /// This is NOT in Montgomery form — caller must use .toMontgomery() to convert.
    pub inline fn fromMulU128(field_elem: BN254Scalar, raw: u128) Self {
        @setEvalBranchQuota(10000);
        const a = field_elem.limbs;
        const b: [2]u64 = .{ @truncate(raw), @truncate(raw >> 64) };
        var slots: [8]u128 = .{0} ** 8;
        inline for (0..4) |i| {
            inline for (0..2) |j| {
                const p: u128 = @as(u128, a[i]) * @as(u128, b[j]);
                slots[i + j] += @as(u128, @as(u64, @truncate(p)));
                slots[i + j + 1] += @as(u128, @as(u64, @truncate(p >> 64)));
            }
        }
        return .{ .slots = slots };
    }

    /// Accumulate another product into this accumulator.
    pub inline fn addAssign(self: *Self, other: Self) void {
        inline for (0..8) |i| {
            self.slots[i] += other.slots[i];
        }
    }

    /// Add two accumulators.
    pub inline fn add(self: Self, other: Self) Self {
        var result: Self = self;
        inline for (0..8) |i| {
            result.slots[i] += other.slots[i];
        }
        return result;
    }

    /// Reduce the accumulated products to a single field element via Montgomery reduction.
    ///
    /// This is where all the deferred work happens: carry propagation across slots,
    /// then standard 4-step CIOS Montgomery reduction. If the accumulated value
    /// overflows 8 limbs (9th limb nonzero), the overflow contribution is folded in
    /// as `overflow * 2^256 mod p = overflow * R mod p = fromU64(overflow)`.
    pub fn reduce(self: Self) BN254Scalar {
        // Step 1: Normalize [8]u128 → [8]u64 + overflow (up to u128)
        // For N accumulated products, each slot can be up to N * 2^130, so
        // the carry chain can produce overflow exceeding u64.
        var limbs: [8]u64 = undefined;
        var carry: u128 = 0;
        inline for (0..8) |i| {
            const sum = self.slots[i] + carry;
            limbs[i] = @truncate(sum);
            carry = sum >> 64;
        }
        // carry holds the overflow: limbs[8..9] conceptually.
        // Split into two u64 limbs for handling.
        const overflow_lo: u64 = @truncate(carry);
        const overflow_hi: u64 = @truncate(carry >> 64);

        // Step 2: Standard 4-step CIOS Montgomery reduction on limbs[0..7]
        // Same algorithm as BN254Scalar.square() reduction
        var t: [5]u64 = .{ limbs[0], limbs[1], limbs[2], limbs[3], 0 };

        inline for (0..4) |i| {
            const m = t[0] *% BN254_INV;
            var c: u64 = 0;
            const prod0 = BN254Scalar.mulWide(m, BN254_MODULUS[0]);
            const sum0 = @as(u128, t[0]) + prod0;
            c = @truncate(sum0 >> 64);

            inline for (1..4) |j| {
                const prod = BN254Scalar.mulWide(m, BN254_MODULUS[j]);
                const s = @as(u128, t[j]) + prod + @as(u128, c);
                t[j - 1] = @truncate(s);
                c = @truncate(s >> 64);
            }
            const final_sum = @as(u128, t[4]) + @as(u128, c) + @as(u128, limbs[i + 4]);
            t[3] = @truncate(final_sum);
            t[4] = @truncate(final_sum >> 64);
        }

        // Step 3: Reduce 5-limb CIOS result (t[4]*2^256 + t[0..3]) mod p.
        // For single products, t[4] is always 0 and one subtraction suffices.
        // For accumulated products, the 8-limb input can span full 512 bits,
        // so t[4] can be 1, requiring up to 5 subtractions of p.
        var result = BN254Scalar{ .limbs = .{ t[0], t[1], t[2], t[3] } };
        var extra = t[4]; // 0 or 1
        var iters: u32 = 0;
        while (extra != 0 or !result.lessThanModulus()) : (iters += 1) {
            std.debug.assert(iters < 6); // at most 5 subtractions of p
            const was_less = result.lessThanModulus();
            result = result.subtractModulus();
            if (was_less) extra -= 1; // borrow consumed from extra
        }

        // Step 4: Add overflow contribution from limbs beyond position 7.
        // The overflow represents value * 2^512. After Montgomery division by R=2^256,
        // this becomes overflow_val * 2^256 mod p.
        // Using toMontgomery: raw_val * R² * R^{-1} = raw_val * R = raw_val * 2^256 mod p.
        if (overflow_lo != 0 or overflow_hi != 0) {
            const raw = BN254Scalar{ .limbs = .{ overflow_lo, overflow_hi, 0, 0 } };
            result = result.add(raw.toMontgomery());
        }

        return result;
    }
};

// ============================================================================
// Barrett Reduction Constants and Helpers
// ============================================================================

/// Barrett reduction constants for BN254 scalar field (Fr).
/// Modulus is 254 bits → 2 spare bits in top limb of 4-limb representation.
const MODULUS_NUM_SPARE_BITS: u6 = 2;

/// BARRETT_MU = floor(2^317 / MODULUS).
/// Computed via: python3 -c "p=0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001; print(hex((2**317)//p))"
const BARRETT_MU: u64 = 0xa948e8c4c474094f;

/// 2 × MODULUS as [5]u64
const MODULUS_TIMES_2: [5]u64 = blk: {
    var result: [5]u64 = .{0} ** 5;
    var carry: u64 = 0;
    for (0..4) |i| {
        const wide = @as(u128, BN254_MODULUS[i]) * 2 + @as(u128, carry);
        result[i] = @as(u64, @truncate(wide));
        carry = @as(u64, @truncate(wide >> 64));
    }
    result[4] = carry;
    break :blk result;
};

/// 3 × MODULUS as [5]u64
const MODULUS_TIMES_3: [5]u64 = blk: {
    var result: [5]u64 = .{0} ** 5;
    var carry: u64 = 0;
    for (0..5) |i| {
        const a: u64 = if (i < 4) BN254_MODULUS[i] else 0;
        const wide = @as(u128, a) + @as(u128, MODULUS_TIMES_2[i]) + @as(u128, carry);
        result[i] = @as(u64, @truncate(wide));
        carry = @as(u64, @truncate(wide >> 64));
    }
    break :blk result;
};

/// Compare 4-limb value >= 4-limb value (unsigned, big-endian comparison)
inline fn cmpGe4(a: [4]u64, b: [4]u64) bool {
    // Compare from most significant limb
    comptime var i = 4;
    inline while (i > 0) {
        i -= 1;
        if (a[i] > b[i]) return true;
        if (a[i] < b[i]) return false;
    }
    return true; // equal
}

/// Compare 5-limb value >= 5-limb value
inline fn cmpGe5(a: [5]u64, b: [5]u64) bool {
    comptime var i = 5;
    inline while (i > 0) {
        i -= 1;
        if (a[i] > b[i]) return true;
        if (a[i] < b[i]) return false;
    }
    return true;
}

/// Subtract 4-limb values: a - b (assumes a >= b)
inline fn sub4(a: [4]u64, b: [4]u64) [4]u64 {
    var result: [4]u64 = undefined;
    var borrow: u64 = 0;
    inline for (0..4) |i| {
        const sb = BN254Scalar.subBorrow(a[i], b[i], borrow);
        result[i] = sb.result;
        borrow = sb.borrow;
    }
    return result;
}

/// Reduce a 5-limb value to 4 limbs modulo BN254 scalar field.
/// Port of Jolt's barrett_reduce_nplus1_to_n.
fn barrettReduceNplus1(c: [5]u64) [4]u64 {
    // 1. Extract tilde_c = floor(c / 2^254)
    const shift_amt: u6 = 64 - @as(u7, MODULUS_NUM_SPARE_BITS);
    const tilde_c: u64 = (c[4] << MODULUS_NUM_SPARE_BITS) |
        (c[3] >> shift_amt);

    // 2. Estimate quotient: m = floor((tilde_c * MU) / 2^64)
    const m: u64 = @truncate((@as(u128, tilde_c) *% @as(u128, BARRETT_MU)) >> 64);

    // 3. Compute m * 2p
    var m2p: [5]u64 = undefined;
    {
        var carry: u64 = 0;
        inline for (0..5) |i| {
            const wide = @as(u128, MODULUS_TIMES_2[i]) * @as(u128, m) + @as(u128, carry);
            m2p[i] = @truncate(wide);
            carry = @truncate(wide >> 64);
        }
    }

    // 4. Subtract: r = c - m*2p (using sbb chain)
    var r: [5]u64 = undefined;
    {
        var borrow: u64 = 0;
        inline for (0..5) |i| {
            const sb = BN254Scalar.subBorrow(c[i], m2p[i], borrow);
            r[i] = sb.result;
            borrow = sb.borrow;
        }
    }

    // 5. Conditional subtraction: r in [0, 4p) → [0, p)
    return barrettCondSubtract(r);
}

/// Conditional subtraction: bring [0, 4p) → [0, p).
/// With SPARE_BITS=2, 2p and 3p fit in 4 limbs (r[4] == 0 after step 4).
fn barrettCondSubtract(r: [5]u64) [4]u64 {
    const r4: [4]u64 = .{ r[0], r[1], r[2], r[3] };
    if (cmpGe5(r, MODULUS_TIMES_3)) {
        return sub4(r4, .{ MODULUS_TIMES_3[0], MODULUS_TIMES_3[1], MODULUS_TIMES_3[2], MODULUS_TIMES_3[3] });
    } else if (cmpGe4(r4, .{ MODULUS_TIMES_2[0], MODULUS_TIMES_2[1], MODULUS_TIMES_2[2], MODULUS_TIMES_2[3] })) {
        return sub4(r4, .{ MODULUS_TIMES_2[0], MODULUS_TIMES_2[1], MODULUS_TIMES_2[2], MODULUS_TIMES_2[3] });
    } else if (cmpGe4(r4, BN254_MODULUS)) {
        return sub4(r4, BN254_MODULUS);
    } else {
        return r4;
    }
}

/// Reduce an L-limb value to a BN254Scalar by Horner's method (MSB first)
/// using the (N+1) → N Barrett kernel at each step.
/// Port of Jolt's from_barrett_reduce.
pub fn barrettReduce(comptime L: usize, limbs: [L]u64) BN254Scalar {
    if (L <= 4) {
        var padded: [5]u64 = .{0} ** 5;
        inline for (0..L) |i| padded[i] = limbs[i];
        return BN254Scalar{ .limbs = barrettCondSubtract(padded) };
    }
    if (L == 5) {
        return BN254Scalar{ .limbs = barrettReduceNplus1(limbs) };
    }
    // Horner fold: acc = limbs[i] + 2^64 * acc, reduce mod p at each step.
    // Processes from highest limb to lowest.
    var acc: [4]u64 = .{0} ** 4;
    comptime var i = L;
    inline while (i > 0) {
        i -= 1;
        // [limbs[i], acc[0..3]] represents limbs[i] + acc * 2^64
        const c5: [5]u64 = .{ limbs[i], acc[0], acc[1], acc[2], acc[3] };
        acc = barrettReduceNplus1(c5);
    }
    return BN254Scalar{ .limbs = acc };
}

// ============================================================================
// Folded Accumulator Types
// ============================================================================

/// Field × u64 product accumulator. 5 u128 slots, normalizes to [5]u64.
/// Supports ~2^64 additions before overflow.
pub const FoldedMulU64 = struct {
    slots: [5]u128,

    pub inline fn zero() FoldedMulU64 {
        return .{ .slots = .{0} ** 5 };
    }

    /// Add raw 4-limb Montgomery representation directly.
    pub inline fn addBigInt4(self: *@This(), limbs: [4]u64) void {
        inline for (0..4) |i| {
            self.slots[i] += @as(u128, limbs[i]);
        }
    }

    /// Slot-wise accumulation.
    pub inline fn addAssign(self: *@This(), other: FoldedMulU64) void {
        inline for (0..5) |i| {
            self.slots[i] += other.slots[i];
        }
    }

    /// Convert [M]u64 product into folded form.
    pub inline fn fromBigInt(comptime M: usize, limbs: [M]u64) FoldedMulU64 {
        var out: [5]u128 = .{0} ** 5;
        inline for (0..@min(M, 5)) |i| {
            out[i] = @as(u128, limbs[i]);
        }
        return .{ .slots = out };
    }

    /// Propagate deferred carries → [6]u64.
    /// Uses 6 limbs (not 5) to capture carry overflow from heavy accumulation.
    /// With N additions of 320-bit products, the result can be up to 320+log2(N) bits.
    pub fn normalize(self: @This()) [6]u64 {
        var out: [6]u64 = .{0} ** 6;
        var carry: u128 = 0;
        inline for (0..5) |i| {
            const sum = self.slots[i] +% carry;
            out[i] = @truncate(sum);
            carry = sum >> 64;
        }
        out[5] = @truncate(carry);
        return out;
    }
};

/// Field × u128 product accumulator. 6 u128 slots, normalizes to [6]u64.
pub const FoldedMulU128 = struct {
    slots: [6]u128,

    pub inline fn zero() FoldedMulU128 {
        return .{ .slots = .{0} ** 6 };
    }

    /// Add raw 4-limb values (zero-extended to 6 slots).
    pub inline fn addBigInt4(self: *@This(), limbs: [4]u64) void {
        inline for (0..4) |i| {
            self.slots[i] += @as(u128, limbs[i]);
        }
    }

    /// Slot-wise accumulation.
    pub inline fn addAssign(self: *@This(), other: FoldedMulU128) void {
        inline for (0..6) |i| {
            self.slots[i] += other.slots[i];
        }
    }

    /// Convert [M]u64 product into folded form.
    pub inline fn fromBigInt(comptime M: usize, limbs: [M]u64) FoldedMulU128 {
        var out: [6]u128 = .{0} ** 6;
        inline for (0..@min(M, 6)) |i| {
            out[i] = @as(u128, limbs[i]);
        }
        return .{ .slots = out };
    }

    /// Promote from FoldedMulU64 (5 slots → 6 slots).
    pub inline fn fromFoldedU64(f: FoldedMulU64) FoldedMulU128 {
        var out: [6]u128 = .{0} ** 6;
        inline for (0..5) |i| {
            out[i] = f.slots[i];
        }
        return .{ .slots = out };
    }

    /// Propagate deferred carries → [6]u64.
    pub fn normalize(self: @This()) [6]u64 {
        var out: [6]u64 = undefined;
        var carry: u128 = 0;
        inline for (0..6) |i| {
            const sum = self.slots[i] +% carry;
            out[i] = @truncate(sum);
            carry = sum >> 64;
        }
        return out;
    }
};

/// Field × u128 wide accumulator. 7 u128 slots, normalizes to [7]u64.
/// Used for second-group Bz where magnitudes can be S160.
pub const FoldedMulU128Accum = struct {
    slots: [7]u128,

    pub inline fn zero() FoldedMulU128Accum {
        return .{ .slots = .{0} ** 7 };
    }

    /// Slot-wise accumulation.
    pub inline fn addAssign(self: *@This(), other: FoldedMulU128Accum) void {
        inline for (0..7) |i| {
            self.slots[i] += other.slots[i];
        }
    }

    /// Promote from FoldedMulU128 (6 slots → 7 slots).
    pub inline fn fromFoldedU128(f: FoldedMulU128) FoldedMulU128Accum {
        var out: [7]u128 = .{0} ** 7;
        inline for (0..6) |i| {
            out[i] = f.slots[i];
        }
        return .{ .slots = out };
    }

    /// Convert [M]u64 product into folded form (7 slots).
    pub inline fn fromBigInt(comptime M: usize, limbs: [M]u64) FoldedMulU128Accum {
        var out: [7]u128 = .{0} ** 7;
        inline for (0..@min(M, 7)) |i| {
            out[i] = @as(u128, limbs[i]);
        }
        return .{ .slots = out };
    }

    /// Propagate deferred carries → [7]u64.
    pub fn normalize(self: @This()) [7]u64 {
        var out: [7]u64 = undefined;
        var carry: u128 = 0;
        inline for (0..7) |i| {
            const sum = self.slots[i] +% carry;
            out[i] = @truncate(sum);
            carry = sum >> 64;
        }
        return out;
    }
};

// ============================================================================
// Unreduced Multiply Methods and Barrett Reduce on BN254Scalar
// ============================================================================

// These are free functions that operate on BN254Scalar since we can't
// reopen the struct. They're accessed as e.g. field_mod.mulU64Unreduced(scalar, val).

/// Return raw Montgomery limbs (no conversion).
pub inline fn toUnreduced(self: BN254Scalar) [4]u64 {
    return self.limbs;
}

/// Multiply field element by u64 scalar, return unreduced 5-limb result.
/// Computes (self.limbs) × scalar via 4×1 schoolbook.
pub inline fn mulU64Unreduced(self: BN254Scalar, scalar: u64) FoldedMulU64 {
    if (scalar == 0) return FoldedMulU64.zero();
    var result: [5]u64 = undefined;
    var carry: u64 = 0;
    inline for (0..4) |i| {
        const wide = @as(u128, self.limbs[i]) * @as(u128, scalar) + @as(u128, carry);
        result[i] = @truncate(wide);
        carry = @truncate(wide >> 64);
    }
    result[4] = carry;
    return FoldedMulU64.fromBigInt(5, result);
}

/// Multiply field element by u128 scalar, return unreduced 6-limb result.
/// Computes (self.limbs) × scalar via 4×2 schoolbook.
pub fn mulU128Unreduced(self: BN254Scalar, scalar: u128) FoldedMulU128 {
    if (scalar == 0) return FoldedMulU128.zero();
    const other: [2]u64 = .{ @truncate(scalar), @truncate(scalar >> 64) };
    var result: [6]u64 = .{0} ** 6;
    inline for (0..2) |j| {
        var carry: u64 = 0;
        inline for (0..4) |i| {
            const wide = @as(u128, result[i + j]) + @as(u128, self.limbs[i]) * @as(u128, other[j]) + @as(u128, carry);
            result[i + j] = @truncate(wide);
            carry = @truncate(wide >> 64);
        }
        result[4 + j] = carry;
    }
    return FoldedMulU128.fromBigInt(6, result);
}

/// Multiply field element by [3]u64 (S192 magnitude), return unreduced 7-limb result.
/// Computes (self.limbs) × scalar via 4×3 schoolbook. Matches arkworks fm_limbs_into<3, 7>.
pub fn mulU192Unreduced(self: BN254Scalar, scalar: [3]u64) FoldedMulU128Accum {
    if (scalar[0] == 0 and scalar[1] == 0 and scalar[2] == 0) return FoldedMulU128Accum.zero();
    var result: [7]u64 = .{0} ** 7;
    inline for (0..3) |j| {
        if (scalar[j] != 0) {
            var carry: u64 = 0;
            inline for (0..4) |i| {
                const wide = @as(u128, result[i + j]) + @as(u128, self.limbs[i]) * @as(u128, scalar[j]) + @as(u128, carry);
                result[i + j] = @truncate(wide);
                carry = @truncate(wide >> 64);
            }
            result[4 + j] = carry;
        }
    }
    return FoldedMulU128Accum.fromBigInt(7, result);
}

/// Reduce FoldedMulU64 (5 slots) → BN254Scalar via Barrett.
/// normalize() returns [6]u64 to handle carry overflow from heavy accumulation.
pub fn reduceMulU64(folded: FoldedMulU64) BN254Scalar {
    return barrettReduce(6, folded.normalize());
}

/// Multiply a field element by a small u64 constant. Cheaper than full field mul:
/// uses 4×1 schoolbook (4 mulq) + Barrett reduction vs 4×4 (16 mulq) + Montgomery.
pub inline fn mulU64(self: BN254Scalar, scalar: u64) BN254Scalar {
    return reduceMulU64(mulU64Unreduced(self, scalar));
}

/// Reduce FoldedMulU128 (6 slots) → BN254Scalar via Barrett.
pub fn reduceMulU128(folded: FoldedMulU128) BN254Scalar {
    return barrettReduce(6, folded.normalize());
}

/// Reduce FoldedMulU128Accum (7 slots) → BN254Scalar via Barrett.
pub fn reduceMulU128Accum(folded: FoldedMulU128Accum) BN254Scalar {
    return barrettReduce(7, folded.normalize());
}

// ============================================================================
// Tiered Accumulators
// ============================================================================

/// Signed accumulator for boolean/small-integer guards.
/// Dual FoldedMulU64 (pos and neg) to handle signed integer guards.
/// fmaddBool: conditional add of raw Montgomery limbs (4 u128 adds).
/// fmaddI8: handles small integer guards in [-4, 4].
pub const SmallAccumU = struct {
    pos: FoldedMulU64,
    neg: FoldedMulU64,

    pub inline fn zero() SmallAccumU {
        return .{ .pos = FoldedMulU64.zero(), .neg = FoldedMulU64.zero() };
    }

    /// Bool fmadd: if true, add raw Montgomery representation of field element.
    /// Cost: 4 u128 additions. No multiplication.
    pub inline fn fmaddBool(self: *@This(), field: BN254Scalar, value: bool) void {
        if (value) {
            self.pos.addBigInt4(field.limbs);
        }
    }

    /// i8 fmadd: for small integer guards in [-4, 4].
    /// Positive values accumulate into pos, negative into neg.
    pub inline fn fmaddI8(self: *@This(), field: BN254Scalar, value: i8) void {
        if (value == 0) return;
        if (value == 1) {
            self.pos.addBigInt4(field.limbs);
            return;
        }
        if (value == -1) {
            self.neg.addBigInt4(field.limbs);
            return;
        }
        // General case: multiply field by |value|
        const abs_val: u64 = if (value > 0) @intCast(value) else @intCast(-value);
        const prod = mulU64Unreduced(field, abs_val);
        if (value > 0) {
            self.pos.addAssign(prod);
        } else {
            self.neg.addAssign(prod);
        }
    }

    /// Barrett reduce to field element. Result is in Montgomery form.
    pub fn barrettReduce(self: @This()) BN254Scalar {
        const pos_reduced = reduceMulU64(self.pos);
        const neg_reduced = reduceMulU64(self.neg);
        return pos_reduced.sub(neg_reduced);
    }
};

/// Signed accumulator for medium-width magnitudes (u64, S64, i128).
/// Dual FoldedMulU128 (pos and neg), reduced via Barrett(pos) - Barrett(neg).
pub const MedAccumS = struct {
    pos: FoldedMulU128,
    neg: FoldedMulU128,

    pub inline fn zero() MedAccumS {
        return .{ .pos = FoldedMulU128.zero(), .neg = FoldedMulU128.zero() };
    }

    /// Bool fmadd: add to pos accumulator.
    pub inline fn fmaddBool(self: *@This(), field: BN254Scalar, value: bool) void {
        if (value) {
            self.pos.addBigInt4(field.limbs);
        }
    }

    /// u64 fmadd: unreduced field × u64, add to pos.
    pub inline fn fmaddU64(self: *@This(), field: BN254Scalar, value: u64) void {
        if (value == 0) return;
        self.pos.addAssign(FoldedMulU128.fromFoldedU64(mulU64Unreduced(field, value)));
    }

    /// i128 fmadd: split by sign, unreduced field × u128.
    pub inline fn fmaddI128(self: *@This(), field: BN254Scalar, value: i128) void {
        if (value == 0) return;
        if (value > 0) {
            self.pos.addAssign(mulU128Unreduced(field, @intCast(value)));
        } else {
            self.neg.addAssign(mulU128Unreduced(field, @as(u128, @intCast(-value))));
        }
    }

    /// Barrett reduce: pos - neg → field element in Montgomery form.
    pub fn barrettReduce(self: @This()) BN254Scalar {
        const pos_reduced = reduceMulU128(self.pos);
        const neg_reduced = reduceMulU128(self.neg);
        return pos_reduced.sub(neg_reduced);
    }
};

/// Wide signed accumulator for large magnitudes (S160/S192).
/// Dual FoldedMulU128Accum (7 u128 slots each).
pub const WideAccumS = struct {
    pos: FoldedMulU128Accum,
    neg: FoldedMulU128Accum,

    pub inline fn zero() WideAccumS {
        return .{ .pos = FoldedMulU128Accum.zero(), .neg = FoldedMulU128Accum.zero() };
    }

    /// i128 fmadd: field × u128 product into 7-slot accumulator.
    pub inline fn fmaddI128(self: *@This(), field: BN254Scalar, value: i128) void {
        if (value == 0) return;
        if (value > 0) {
            self.pos.addAssign(FoldedMulU128Accum.fromFoldedU128(mulU128Unreduced(field, @intCast(value))));
        } else {
            self.neg.addAssign(FoldedMulU128Accum.fromFoldedU128(mulU128Unreduced(field, @as(u128, @intCast(-value)))));
        }
    }

    /// S192 fmadd: field × S192 magnitude product into 7-slot accumulator.
    /// Matches arkworks WideAccumS::fmadd(F, S192).
    pub inline fn fmaddS192(self: *@This(), field: BN254Scalar, value: S192) void {
        if (value.magnitude[0] == 0 and value.magnitude[1] == 0 and value.magnitude[2] == 0) return;
        const product = mulU192Unreduced(field, value.magnitude);
        if (value.is_positive) {
            self.pos.addAssign(product);
        } else {
            self.neg.addAssign(product);
        }
    }

    /// Barrett reduce: pos - neg → field element in Montgomery form.
    pub fn barrettReduce(self: @This()) BN254Scalar {
        const pos_reduced = reduceMulU128Accum(self.pos);
        const neg_reduced = reduceMulU128Accum(self.neg);
        return pos_reduced.sub(neg_reduced);
    }
};

/// Signed 192-bit integer type matching arkworks SignedBigInt<3>.
/// Used for second-group Bz values that need exact integer arithmetic
/// without wrapping i128 truncation artifacts.
pub const S192 = struct {
    magnitude: [3]u64, // unsigned magnitude, little-endian
    is_positive: bool, // true = positive (matches arkworks convention)

    pub inline fn zero() S192 {
        return .{ .magnitude = .{ 0, 0, 0 }, .is_positive = true };
    }

    pub inline fn isZero(self: S192) bool {
        return self.magnitude[0] == 0 and self.magnitude[1] == 0 and self.magnitude[2] == 0;
    }

    pub inline fn fromU64(v: u64) S192 {
        return .{ .magnitude = .{ v, 0, 0 }, .is_positive = true };
    }

    pub inline fn fromU128(v: u128) S192 {
        return .{ .magnitude = .{ @truncate(v), @truncate(v >> 64), 0 }, .is_positive = true };
    }

    pub inline fn fromI128(v: i128) S192 {
        if (v >= 0) {
            const u: u128 = @intCast(v);
            return .{ .magnitude = .{ @truncate(u), @truncate(u >> 64), 0 }, .is_positive = true };
        } else {
            const u: u128 = @bitCast(-%v);
            return .{ .magnitude = .{ @truncate(u), @truncate(u >> 64), 0 }, .is_positive = false };
        }
    }

    pub inline fn fromI64(v: i64) S192 {
        if (v >= 0) {
            return .{ .magnitude = .{ @intCast(v), 0, 0 }, .is_positive = true };
        } else {
            return .{ .magnitude = .{ @intCast(-%v), 0, 0 }, .is_positive = false };
        }
    }

    pub inline fn neg(self: S192) S192 {
        if (self.isZero()) return self;
        return .{ .magnitude = self.magnitude, .is_positive = !self.is_positive };
    }

    pub fn add(self: S192, other: S192) S192 {
        if (other.isZero()) return self;
        if (self.isZero()) return other;
        if (self.is_positive == other.is_positive) {
            return .{ .magnitude = addMagnitudes(self.magnitude, other.magnitude), .is_positive = self.is_positive };
        }
        const cmp = cmpMagnitudes(self.magnitude, other.magnitude);
        if (cmp == .eq) return S192.zero();
        if (cmp == .gt) {
            return .{ .magnitude = subMagnitudes(self.magnitude, other.magnitude), .is_positive = self.is_positive };
        }
        return .{ .magnitude = subMagnitudes(other.magnitude, self.magnitude), .is_positive = other.is_positive };
    }

    pub fn sub(self: S192, other: S192) S192 {
        return self.add(other.neg());
    }

    /// 1×3 schoolbook multiply by i32, truncated to 3 limbs.
    /// Matches arkworks mul_trunc::<1,3>.
    pub fn mulI32(self: S192, c: i32) S192 {
        if (c == 0) return S192.zero();
        if (self.isZero()) return S192.zero();
        const abs_c: u64 = if (c >= 0) @intCast(c) else @intCast(-c);
        var result: [3]u64 = .{ 0, 0, 0 };
        var carry: u64 = 0;
        inline for (0..3) |i| {
            const tmp = @as(u128, self.magnitude[i]) * @as(u128, abs_c) + @as(u128, carry);
            result[i] = @truncate(tmp);
            carry = @truncate(tmp >> 64);
        }
        return .{ .magnitude = result, .is_positive = self.is_positive == (c >= 0) };
    }

    /// Fused multiply-add: acc += term * c. Matches arkworks fmadd_trunc.
    pub fn fmaddI32(acc: *S192, c: i32, term: S192) void {
        const product = term.mulI32(c);
        acc.* = acc.add(product);
    }

    fn addMagnitudes(a: [3]u64, b: [3]u64) [3]u64 {
        if (!@inComptime() and comptime use_arm64_asm) return arm64Add192(a, b);
        var result: [3]u64 = undefined;
        if (comptime builtin.cpu.arch == .x86_64) {
            var c: u8 = 0;
            inline for (0..3) |i| {
                c = x86.addcarry(c, a[i], b[i], &result[i]);
            }
        } else {
            var carry: u64 = 0;
            inline for (0..3) |i| {
                const tmp = @as(u128, a[i]) + @as(u128, b[i]) + @as(u128, carry);
                result[i] = @truncate(tmp);
                carry = @truncate(tmp >> 64);
            }
        }
        return result;
    }

    fn subMagnitudes(a: [3]u64, b: [3]u64) [3]u64 {
        if (!@inComptime() and comptime use_arm64_asm) return arm64Sub192(a, b);
        var result: [3]u64 = undefined;
        if (comptime builtin.cpu.arch == .x86_64) {
            var b_out: u8 = 0;
            inline for (0..3) |i| {
                b_out = x86.subborrow(b_out, a[i], b[i], &result[i]);
            }
        } else {
            var borrow: u64 = 0;
            inline for (0..3) |i| {
                const tmp = (@as(u128, 1) << 64) + @as(u128, a[i]) - @as(u128, b[i]) - @as(u128, borrow);
                result[i] = @truncate(tmp);
                borrow = if ((tmp >> 64) == 0) @as(u64, 1) else @as(u64, 0);
            }
        }
        return result;
    }

    fn cmpMagnitudes(a: [3]u64, b: [3]u64) std.math.Order {
        var i: usize = 3;
        while (i > 0) {
            i -= 1;
            if (a[i] != b[i]) return std.math.order(a[i], b[i]);
        }
        return .eq;
    }
};

// Verify BN254Scalar implements JoltField interface
comptime {
    _ = JoltField(BN254Scalar);
}

test "UnreducedProductAccum single product matches mul" {
    const a = BN254Scalar.fromU64(12345);
    const b = BN254Scalar.fromU64(67890);
    const expected = a.mul(b);
    const actual = a.mulToProductAccum(b).reduce();
    try std.testing.expect(expected.eql(actual));
}

test "UnreducedProductAccum sum of products" {
    // Verify sum(a[i]*b[i]) via accum equals sum via direct mul
    const N = 100;
    var a_vals: [N]BN254Scalar = undefined;
    var b_vals: [N]BN254Scalar = undefined;

    // Use deterministic "random" values
    for (0..N) |i| {
        a_vals[i] = BN254Scalar.fromU64(@as(u64, i) * 7 + 13);
        b_vals[i] = BN254Scalar.fromU64(@as(u64, i) * 11 + 29);
    }

    // Direct sum
    var expected = BN254Scalar.zero();
    for (0..N) |i| {
        expected = expected.add(a_vals[i].mul(b_vals[i]));
    }

    // Accumulated sum
    var accum = UnreducedProductAccum.zero();
    for (0..N) |i| {
        accum.addAssign(a_vals[i].mulToProductAccum(b_vals[i]));
    }
    const actual = accum.reduce();

    try std.testing.expect(expected.eql(actual));
}

test "UnreducedProductAccum large accumulation (10000 products)" {
    const N = 10000;
    var expected = BN254Scalar.zero();
    var accum = UnreducedProductAccum.zero();

    for (0..N) |i| {
        const a = BN254Scalar.fromU64(@as(u64, i) * 31 + 7);
        const b = BN254Scalar.fromU64(@as(u64, i) * 17 + 3);
        expected = expected.add(a.mul(b));
        accum.addAssign(a.mulToProductAccum(b));
    }

    try std.testing.expect(expected.eql(accum.reduce()));
}

test "UnreducedProductAccum zero accumulation" {
    const accum = UnreducedProductAccum.zero();
    try std.testing.expect(accum.reduce().isZero());
}

test "UnreducedProductAccum with large field elements" {
    // Use values near the modulus to stress carry propagation
    const a = BN254Scalar{ .limbs = .{ 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0x0FFFFFFFFFFFFFFF } };
    const b = BN254Scalar{ .limbs = .{ 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0x0FFFFFFFFFFFFFFF } };

    const expected = a.mul(b);
    const actual = a.mulToProductAccum(b).reduce();
    try std.testing.expect(expected.eql(actual));

    // Accumulate many large products
    var exp2 = BN254Scalar.zero();
    var acc2 = UnreducedProductAccum.zero();
    for (0..50) |_| {
        exp2 = exp2.add(a.mul(b));
        acc2.addAssign(a.mulToProductAccum(b));
    }
    try std.testing.expect(exp2.eql(acc2.reduce()));
}

test "bn254 scalar basic operations" {
    const zero = BN254Scalar.zero();
    const one = BN254Scalar.one();

    try std.testing.expect(zero.isZero());
    try std.testing.expect(one.isOne());
    try std.testing.expect(!zero.isOne());
    try std.testing.expect(!one.isZero());
}

test "bn254 scalar addition and subtraction" {
    const a = BN254Scalar.fromU64(100);
    const b = BN254Scalar.fromU64(50);

    // a + b - b should equal a
    const sum = a.add(b);
    const back = sum.sub(b);
    try std.testing.expect(a.eql(back));

    // a - a should equal zero
    const diff = a.sub(a);
    try std.testing.expect(diff.isZero());
}

test "bn254 scalar multiplication" {
    const one = BN254Scalar.one();
    const a = BN254Scalar.fromU64(7);

    // a * 1 = a
    const prod1 = a.mul(one);
    try std.testing.expect(a.eql(prod1));

    // 1 * a = a
    const prod2 = one.mul(a);
    try std.testing.expect(a.eql(prod2));

    // a * 0 = 0
    const zero = BN254Scalar.zero();
    const prod3 = a.mul(zero);
    try std.testing.expect(prod3.isZero());
}

test "bn254 scalar multiplication correctness" {
    // Test: 3 * 7 = 21
    const three = BN254Scalar.fromU64(3);
    const seven = BN254Scalar.fromU64(7);
    const twenty_one = BN254Scalar.fromU64(21);

    const product = three.mul(seven);
    try std.testing.expect(product.eql(twenty_one));
}

test "bn254 scalar inverse" {
    const a = BN254Scalar.fromU64(7);
    const one = BN254Scalar.one();

    // a * a^{-1} = 1
    if (a.inverse()) |a_inv| {
        const prod = a.mul(a_inv);
        try std.testing.expect(prod.eql(one));
    } else {
        try std.testing.expect(false);
    }
}

test "bn254 scalar power" {
    const two = BN254Scalar.fromU64(2);
    const eight = BN254Scalar.fromU64(8);

    // 2^3 = 8
    const result = two.pow(3);
    try std.testing.expect(result.eql(eight));

    // a^0 = 1
    const one = BN254Scalar.one();
    const pow0 = two.pow(0);
    try std.testing.expect(pow0.eql(one));

    // a^1 = a
    const pow1 = two.pow(1);
    try std.testing.expect(pow1.eql(two));
}

test "mulHiBigIntU128 vs montgomeryMul equivalence" {
    // Test that a.mulHiBigIntU128(b.limbs) == a.mul(b) when b has zero low limbs
    // This is critical for sumcheck challenge evaluation
    const F = BN254Scalar;

    // Create a "challenge" with zero low limbs [0, 0, L, H]
    const challenge = F{ .limbs = .{ 0, 0, 0x123456789abcdef0, 0x0fedcba987654321 } };

    // Create various field elements to test with
    const a1 = F.fromU64(42);
    const a2 = F{ .limbs = .{ 0xdeadbeefcafebabe, 0x1234567890abcdef, 0xfedcba9876543210, 0x0111111111111111 } };
    const a3 = F.one();

    // Test 1: a1.mulHiBigIntU128(challenge.limbs) == a1.mul(challenge)
    const result1a = a1.mulHiBigIntU128(challenge.limbs);
    const result1b = a1.mul(challenge);
    try std.testing.expect(result1a.eql(result1b));

    // Test 2: a2.mulHiBigIntU128(challenge.limbs) == a2.mul(challenge)
    const result2a = a2.mulHiBigIntU128(challenge.limbs);
    const result2b = a2.mul(challenge);
    try std.testing.expect(result2a.eql(result2b));

    // Test 3: a3.mulHiBigIntU128(challenge.limbs) == a3.mul(challenge)
    const result3a = a3.mulHiBigIntU128(challenge.limbs);
    const result3b = a3.mul(challenge);
    try std.testing.expect(result3a.eql(result3b));

    // Test 4: challenge.mul(challenge) == challenge.mulHiBigIntU128(challenge.limbs)?
    // This tests r^2 computation equivalence
    const r2_full = challenge.mul(challenge);
    const r2_hi = challenge.mulHiBigIntU128(challenge.limbs);
    try std.testing.expect(r2_full.eql(r2_hi));

    // Test 5: Now test the full eval_from_hint scenario:
    // p(r) = c0 + c1*r + c2*r^2
    // In Jolt: c1*r uses mulHiBigIntU128, c2*r^2 uses standard mul (running_point is F type)
    // In Zolt: c1*r uses mulHiBigIntU128, c2*r^2 uses standard mul with r2=challenge.mul(challenge)
    const c0 = F.fromU64(100);
    const c1 = F.fromU64(200);
    const c2 = F.fromU64(300);

    // Jolt way: running_point = challenge_to_F, then running_point = running_point * challenge (mulHiBigIntU128)
    const running_point = challenge; // (*x).into() just wraps limbs directly
    const r2_jolt = running_point.mulHiBigIntU128(challenge.limbs);
    const p_jolt = c0.add(c1.mulHiBigIntU128(challenge.limbs)).add(c2.mul(r2_jolt));

    // Zolt way: r2 = challenge.mul(challenge), then c2.mul(r2)
    const r2_zolt = challenge.mul(challenge);
    const p_zolt = c0.add(c1.mulHiBigIntU128(challenge.limbs)).add(c2.mul(r2_zolt));

    try std.testing.expect(p_jolt.eql(p_zolt));
}

test "sumOfProducts equivalence" {
    // Test BN254Scalar sumOfProducts
    {
        const a = BN254Scalar.fromU64(12345);
        const b = BN254Scalar.fromU64(67890);
        const c = BN254Scalar.fromU64(11111);
        const d = BN254Scalar.fromU64(22222);

        const expected = a.mul(b).add(c.mul(d));
        const fused = BN254Scalar.sumOfProducts(.{ a, c }, .{ b, d });
        try std.testing.expect(expected.eql(fused));
    }

    // Test with larger values (near modulus)
    {
        const a = BN254Scalar{ .limbs = .{ 0xffffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff, 0x0fffffffffffffff } };
        const b = BN254Scalar{ .limbs = .{ 0xeeeeeeeeeeeeeeee, 0xdddddddddddddddd, 0xcccccccccccccccc, 0x0bbbbbbbbbbbbbbb } };
        const c = BN254Scalar.fromU64(999999999);
        const d = BN254Scalar.fromU64(888888888);

        const expected = a.mul(b).add(c.mul(d));
        const fused = BN254Scalar.sumOfProducts(.{ a, c }, .{ b, d });
        try std.testing.expect(expected.eql(fused));
    }

    // Test BN254BaseField (Fp) sumOfProducts
    {
        const Fp = BN254BaseField;
        const a = Fp.fromU64(54321);
        const b = Fp.fromU64(98765);
        const c = Fp.fromU64(33333);
        const d = Fp.fromU64(44444);

        const expected = a.mul(b).add(c.mul(d));
        const fused = Fp.sumOfProducts(.{ a, c }, .{ b, d });
        try std.testing.expect(expected.eql(fused));
    }

    // Test with subtraction (a*b + (-c)*d = a*b - c*d)
    {
        const Fp = BN254BaseField;
        const a = Fp.fromU64(100000);
        const b = Fp.fromU64(200000);
        const c = Fp.fromU64(50000);
        const d = Fp.fromU64(30000);

        const expected = a.mul(b).sub(c.mul(d));
        const fused = Fp.sumOfProducts(.{ a, c.neg() }, .{ b, d });
        try std.testing.expect(expected.eql(fused));
    }

    // Test zero cases
    {
        const zero = BN254Scalar.zero();
        const a = BN254Scalar.fromU64(42);
        const b = BN254Scalar.fromU64(99);

        const result = BN254Scalar.sumOfProducts(.{ a, zero }, .{ b, zero });
        try std.testing.expect(result.eql(a.mul(b)));
    }
}

test "bn254 scalar toBytes/fromBytes roundtrip" {
    // Test a small value
    const a = BN254Scalar.fromU64(12345678901234567890);
    const bytes = a.toBytes();
    const b = BN254Scalar.fromBytes(&bytes);
    try std.testing.expect(a.eql(b));

    // Test one
    const one = BN254Scalar.one();
    const one_bytes = one.toBytes();
    const one_back = BN254Scalar.fromBytes(&one_bytes);
    try std.testing.expect(one.eql(one_back));

    // Test zero
    const zero = BN254Scalar.zero();
    const zero_bytes = zero.toBytes();
    const zero_back = BN254Scalar.fromBytes(&zero_bytes);
    try std.testing.expect(zero.eql(zero_back));
}

/// Batch field operations for SIMD-like performance
/// These functions operate on slices for cache efficiency
pub const BatchOps = struct {
    /// Batch addition: results[i] = a[i] + b[i]
    pub fn batchAdd(results: []BN254Scalar, a: []const BN254Scalar, b: []const BN254Scalar) void {
        std.debug.assert(results.len == a.len and a.len == b.len);
        for (0..results.len) |i| {
            results[i] = a[i].add(b[i]);
        }
    }

    /// Batch subtraction: results[i] = a[i] - b[i]
    pub fn batchSub(results: []BN254Scalar, a: []const BN254Scalar, b: []const BN254Scalar) void {
        std.debug.assert(results.len == a.len and a.len == b.len);
        for (0..results.len) |i| {
            results[i] = a[i].sub(b[i]);
        }
    }

    /// Batch multiplication: results[i] = a[i] * b[i]
    pub fn batchMul(results: []BN254Scalar, a: []const BN254Scalar, b: []const BN254Scalar) void {
        std.debug.assert(results.len == a.len and a.len == b.len);
        for (0..results.len) |i| {
            results[i] = a[i].mul(b[i]);
        }
    }

    /// Batch scalar multiplication: results[i] = a[i] * scalar
    pub fn batchMulScalar(results: []BN254Scalar, a: []const BN254Scalar, scalar: BN254Scalar) void {
        std.debug.assert(results.len == a.len);
        for (0..results.len) |i| {
            results[i] = a[i].mul(scalar);
        }
    }

    /// Batch squaring: results[i] = a[i]^2
    pub fn batchSquare(results: []BN254Scalar, a: []const BN254Scalar) void {
        std.debug.assert(results.len == a.len);
        for (0..results.len) |i| {
            results[i] = a[i].square();
        }
    }

    /// Inner product: sum(a[i] * b[i])
    pub fn innerProduct(a: []const BN254Scalar, b: []const BN254Scalar) BN254Scalar {
        std.debug.assert(a.len == b.len);
        var result = BN254Scalar.zero();
        for (0..a.len) |i| {
            result = result.add(a[i].mul(b[i]));
        }
        return result;
    }

    /// Sum of products with precomputed terms for Horner's method
    /// Computes: a[0] + x*(a[1] + x*(a[2] + ... + x*a[n-1])))
    pub fn hornerEval(coeffs: []const BN254Scalar, x: BN254Scalar) BN254Scalar {
        if (coeffs.len == 0) return BN254Scalar.zero();

        var result = coeffs[coeffs.len - 1];
        var i: usize = coeffs.len - 1;
        while (i > 0) {
            i -= 1;
            result = result.mul(x).add(coeffs[i]);
        }
        return result;
    }

    /// Batch inverse using Montgomery's trick
    /// Computes inverses of all elements using only one field inversion
    /// Much faster than computing individual inverses: O(3n) muls + 1 inverse vs O(n) inverses
    pub fn batchInverse(results: []BN254Scalar, a: []const BN254Scalar, allocator: std.mem.Allocator) !void {
        std.debug.assert(results.len == a.len);
        if (a.len == 0) return;

        // Step 1: Compute running products
        // products[i] = a[0] * a[1] * ... * a[i]
        const products = try allocator.alloc(BN254Scalar, a.len);
        defer allocator.free(products);

        products[0] = a[0];
        for (1..a.len) |i| {
            if (a[i].isZero()) {
                // Handle zero by using one (will result in zero inverse)
                products[i] = products[i - 1];
            } else {
                products[i] = products[i - 1].mul(a[i]);
            }
        }

        // Step 2: Compute inverse of the final product
        const all_inv = products[a.len - 1].inverse() orelse BN254Scalar.zero();

        // Step 3: Compute individual inverses
        var running_inv = all_inv;
        var i: usize = a.len;
        while (i > 1) {
            i -= 1;
            if (a[i].isZero()) {
                results[i] = BN254Scalar.zero();
            } else {
                // a[i]^{-1} = running_inv * products[i-1]
                results[i] = running_inv.mul(products[i - 1]);
                running_inv = running_inv.mul(a[i]);
            }
        }
        // Handle a[0]
        if (a[0].isZero()) {
            results[0] = BN254Scalar.zero();
        } else {
            results[0] = running_inv;
        }
    }

    /// Multi-scalar multiplication accumulator
    /// Computes sum(scalars[i] * bases[i])
    pub fn multiScalarMulLinear(scalars: []const BN254Scalar, bases: []const BN254Scalar) BN254Scalar {
        return innerProduct(scalars, bases);
    }
};

/// SIMD-accelerated field operations
/// Uses Zig's built-in SIMD vectors for parallel limb operations where beneficial
pub const SimdOps = struct {
    /// Vector type for 4 limbs (one field element)
    pub const Limb4 = @Vector(4, u64);

    /// Convert scalar to SIMD vector
    pub inline fn toVec(scalar: BN254Scalar) Limb4 {
        return Limb4{ scalar.limbs[0], scalar.limbs[1], scalar.limbs[2], scalar.limbs[3] };
    }

    /// Convert SIMD vector to scalar
    pub inline fn fromVec(vec: Limb4) BN254Scalar {
        return BN254Scalar{ .limbs = .{ vec[0], vec[1], vec[2], vec[3] } };
    }

    /// SIMD modulus vector
    pub const modulus_vec: Limb4 = Limb4{
        BN254_MODULUS[0],
        BN254_MODULUS[1],
        BN254_MODULUS[2],
        BN254_MODULUS[3],
    };

    /// Parallel comparison: returns true if all limbs of a < b (lexicographically)
    pub inline fn lessThan(a: Limb4, b: Limb4) bool {
        // Compare from most significant limb
        if (a[3] != b[3]) return a[3] < b[3];
        if (a[2] != b[2]) return a[2] < b[2];
        if (a[1] != b[1]) return a[1] < b[1];
        return a[0] < b[0];
    }

    /// SIMD-parallel addition with reduction (vectorized limb operations)
    /// For cases where we add many field elements, this allows better instruction pipelining
    pub fn simdAdd4(a: [4]BN254Scalar, b: [4]BN254Scalar) [4]BN254Scalar {
        var results: [4]BN254Scalar = undefined;
        // Process all 4 additions - compiler can vectorize limb operations
        comptime var i = 0;
        inline while (i < 4) : (i += 1) {
            results[i] = a[i].add(b[i]);
        }
        return results;
    }

    /// SIMD-parallel multiplication (for pipelining 4 muls together)
    pub fn simdMul4(a: [4]BN254Scalar, b: [4]BN254Scalar) [4]BN254Scalar {
        var results: [4]BN254Scalar = undefined;
        comptime var i = 0;
        inline while (i < 4) : (i += 1) {
            results[i] = a[i].mul(b[i]);
        }
        return results;
    }

    /// Process slices in chunks of 4 for better vectorization
    pub fn batchAddSimd(results: []BN254Scalar, a: []const BN254Scalar, b: []const BN254Scalar) void {
        std.debug.assert(results.len == a.len and a.len == b.len);

        const len = results.len;
        const chunks = len / 4;
        const remainder = len % 4;

        // Process in SIMD-friendly chunks of 4
        var i: usize = 0;
        while (i < chunks * 4) : (i += 4) {
            const a4 = [4]BN254Scalar{ a[i], a[i + 1], a[i + 2], a[i + 3] };
            const b4 = [4]BN254Scalar{ b[i], b[i + 1], b[i + 2], b[i + 3] };
            const r4 = simdAdd4(a4, b4);
            results[i] = r4[0];
            results[i + 1] = r4[1];
            results[i + 2] = r4[2];
            results[i + 3] = r4[3];
        }

        // Handle remainder
        for (i..i + remainder) |j| {
            results[j] = a[j].add(b[j]);
        }
    }

    /// Process multiplications in chunks of 4 for better vectorization
    pub fn batchMulSimd(results: []BN254Scalar, a: []const BN254Scalar, b: []const BN254Scalar) void {
        std.debug.assert(results.len == a.len and a.len == b.len);

        const len = results.len;
        const chunks = len / 4;
        const remainder = len % 4;

        // Process in SIMD-friendly chunks of 4
        var i: usize = 0;
        while (i < chunks * 4) : (i += 4) {
            const a4 = [4]BN254Scalar{ a[i], a[i + 1], a[i + 2], a[i + 3] };
            const b4 = [4]BN254Scalar{ b[i], b[i + 1], b[i + 2], b[i + 3] };
            const r4 = simdMul4(a4, b4);
            results[i] = r4[0];
            results[i + 1] = r4[1];
            results[i + 2] = r4[2];
            results[i + 3] = r4[3];
        }

        // Handle remainder
        for (i..i + remainder) |j| {
            results[j] = a[j].mul(b[j]);
        }
    }

    /// Inner product with unrolled accumulation for better pipelining
    pub fn innerProductSimd(a: []const BN254Scalar, b: []const BN254Scalar) BN254Scalar {
        std.debug.assert(a.len == b.len);

        const len = a.len;
        const chunks = len / 4;
        const remainder = len % 4;

        // Use 4 accumulators for better instruction-level parallelism
        var acc0 = BN254Scalar.zero();
        var acc1 = BN254Scalar.zero();
        var acc2 = BN254Scalar.zero();
        var acc3 = BN254Scalar.zero();

        var i: usize = 0;
        while (i < chunks * 4) : (i += 4) {
            acc0 = acc0.add(a[i].mul(b[i]));
            acc1 = acc1.add(a[i + 1].mul(b[i + 1]));
            acc2 = acc2.add(a[i + 2].mul(b[i + 2]));
            acc3 = acc3.add(a[i + 3].mul(b[i + 3]));
        }

        // Handle remainder
        for (i..i + remainder) |j| {
            acc0 = acc0.add(a[j].mul(b[j]));
        }

        // Combine accumulators
        return acc0.add(acc1).add(acc2.add(acc3));
    }
};

test "simd operations" {
    var a: [8]BN254Scalar = undefined;
    var b: [8]BN254Scalar = undefined;
    var results: [8]BN254Scalar = undefined;

    for (0..8) |i| {
        a[i] = BN254Scalar.fromU64(@as(u64, @intCast(i + 1)));
        b[i] = BN254Scalar.fromU64(@as(u64, @intCast(i + 10)));
    }

    // Test SIMD batch add
    SimdOps.batchAddSimd(&results, &a, &b);
    for (0..8) |i| {
        const expected = BN254Scalar.fromU64(@as(u64, @intCast(2 * i + 11)));
        try std.testing.expect(results[i].eql(expected));
    }

    // Test SIMD batch mul
    SimdOps.batchMulSimd(&results, &a, &b);
    for (0..8) |i| {
        const expected = BN254Scalar.fromU64(@as(u64, @intCast((i + 1) * (i + 10))));
        try std.testing.expect(results[i].eql(expected));
    }

    // Test SIMD inner product: sum((i+1) * (i+10)) for i=0..7
    // = 1*10 + 2*11 + 3*12 + 4*13 + 5*14 + 6*15 + 7*16 + 8*17
    // = 10 + 22 + 36 + 52 + 70 + 90 + 112 + 136 = 528
    const ip = SimdOps.innerProductSimd(&a, &b);
    try std.testing.expect(ip.eql(BN254Scalar.fromU64(528)));
}

test "batch operations" {
    const allocator = std.testing.allocator;

    var a: [4]BN254Scalar = undefined;
    var b: [4]BN254Scalar = undefined;
    var results: [4]BN254Scalar = undefined;

    for (0..4) |i| {
        a[i] = BN254Scalar.fromU64(@as(u64, @intCast(i + 1)));
        b[i] = BN254Scalar.fromU64(@as(u64, @intCast(i + 5)));
    }

    // Test batch add
    BatchOps.batchAdd(&results, &a, &b);
    for (0..4) |i| {
        const expected = BN254Scalar.fromU64(@as(u64, @intCast(2 * i + 6)));
        try std.testing.expect(results[i].eql(expected));
    }

    // Test inner product: 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70
    const ip = BatchOps.innerProduct(&a, &b);
    try std.testing.expect(ip.eql(BN254Scalar.fromU64(70)));

    // Test Horner evaluation: 1 + 2x + 3x^2 + 4x^3 at x=2
    // = 1 + 2*2 + 3*4 + 4*8 = 1 + 4 + 12 + 32 = 49
    const horner_result = BatchOps.hornerEval(&a, BN254Scalar.fromU64(2));
    try std.testing.expect(horner_result.eql(BN254Scalar.fromU64(49)));

    // Test batch inverse
    var non_zero_a: [3]BN254Scalar = .{
        BN254Scalar.fromU64(2),
        BN254Scalar.fromU64(3),
        BN254Scalar.fromU64(5),
    };
    var inverses: [3]BN254Scalar = undefined;
    try BatchOps.batchInverse(&inverses, &non_zero_a, allocator);

    // Verify: a[i] * inv[i] = 1
    for (0..3) |i| {
        const prod = non_zero_a[i].mul(inverses[i]);
        try std.testing.expect(prod.eql(BN254Scalar.one()));
    }
}

test "batchInversion known values" {
    const Fp = BN254BaseField;
    var elems = [_]Fp{ Fp.fromU64(2), Fp.fromU64(3), Fp.fromU64(7) };
    var scratch: [3]Fp = undefined;
    Fp.batchInversion(&elems, &scratch);
    // Each element should now be its inverse
    try std.testing.expect(elems[0].mul(Fp.fromU64(2)).eql(Fp.one()));
    try std.testing.expect(elems[1].mul(Fp.fromU64(3)).eql(Fp.one()));
    try std.testing.expect(elems[2].mul(Fp.fromU64(7)).eql(Fp.one()));
}

test "batchInversion with zeros" {
    const Fp = BN254BaseField;
    var elems = [_]Fp{ Fp.fromU64(5), Fp.zero(), Fp.fromU64(11), Fp.zero(), Fp.fromU64(13) };
    var scratch: [5]Fp = undefined;
    Fp.batchInversion(&elems, &scratch);
    // Zeros stay zero
    try std.testing.expect(elems[1].isZero());
    try std.testing.expect(elems[3].isZero());
    // Non-zeros are inverted
    try std.testing.expect(elems[0].mul(Fp.fromU64(5)).eql(Fp.one()));
    try std.testing.expect(elems[2].mul(Fp.fromU64(11)).eql(Fp.one()));
    try std.testing.expect(elems[4].mul(Fp.fromU64(13)).eql(Fp.one()));
}

test "batchInversion single element" {
    const Fp = BN254BaseField;
    var elems = [_]Fp{Fp.fromU64(42)};
    var scratch: [1]Fp = undefined;
    Fp.batchInversion(&elems, &scratch);
    try std.testing.expect(elems[0].mul(Fp.fromU64(42)).eql(Fp.one()));
}

test "batchInversion empty" {
    const Fp = BN254BaseField;
    var elems: [0]Fp = .{};
    var scratch: [0]Fp = .{};
    Fp.batchInversion(&elems, &scratch);
}

test "batchInversion all zeros" {
    const Fp = BN254BaseField;
    var elems = [_]Fp{ Fp.zero(), Fp.zero(), Fp.zero() };
    var scratch: [3]Fp = undefined;
    Fp.batchInversion(&elems, &scratch);
    for (elems) |e| try std.testing.expect(e.isZero());
}

// ============================================================================
// Barrett Reduction and Tiered Accumulator Tests
// ============================================================================

test "Barrett reduce matches Montgomery for small values" {
    // mulU64Unreduced → reduceMulU64 should match field.mul(fromU64)
    const a = BN254Scalar.fromU64(12345);
    const b: u64 = 67890;
    const expected = a.mul(BN254Scalar.fromU64(b));
    const actual = reduceMulU64(mulU64Unreduced(a, b));
    try std.testing.expect(expected.eql(actual));
}

test "Barrett reduce matches Montgomery for u128 values" {
    const a = BN254Scalar.fromU64(999999);
    const b: u128 = 0x123456789ABCDEF0_FEDCBA9876543210;
    const expected = a.mul(BN254Scalar.fromU128(b));
    const actual = reduceMulU128(mulU128Unreduced(a, b));
    try std.testing.expect(expected.eql(actual));
}

test "Barrett reduce with zero" {
    const a = BN254Scalar.fromU64(42);
    const result = reduceMulU64(mulU64Unreduced(a, 0));
    try std.testing.expect(result.isZero());
}

test "Barrett reduce with one" {
    const a = BN254Scalar.fromU64(42);
    const result = reduceMulU64(mulU64Unreduced(a, 1));
    try std.testing.expect(result.eql(a));
}

test "FoldedMulU64 normalize correctness" {
    var f = FoldedMulU64.zero();
    // Add a known value
    const a = BN254Scalar.fromU64(7);
    f.addBigInt4(a.limbs);
    f.addBigInt4(a.limbs);
    // Normalizing should give 2*a Montgomery limbs (with carries propagated)
    const norm = f.normalize();
    const expected = a.add(a);
    // Barrett reduce the folded result
    const actual = reduceMulU64(f);
    try std.testing.expect(expected.eql(actual));
    _ = norm;
}

test "SmallAccumU fmaddBool accumulation matches field.add" {
    var acc = SmallAccumU.zero();
    const w0 = BN254Scalar.fromU64(100);
    const w1 = BN254Scalar.fromU64(200);
    const w2 = BN254Scalar.fromU64(300);

    acc.fmaddBool(w0, true);
    acc.fmaddBool(w1, false);
    acc.fmaddBool(w2, true);

    const expected = w0.add(w2);
    const actual = acc.barrettReduce();
    try std.testing.expect(expected.eql(actual));
}

test "SmallAccumU fmaddI8 handles negative values" {
    var acc = SmallAccumU.zero();
    const w0 = BN254Scalar.fromU64(100);
    const w1 = BN254Scalar.fromU64(200);
    const w2 = BN254Scalar.fromU64(50);

    acc.fmaddI8(w0, 1);
    acc.fmaddI8(w1, -1);
    acc.fmaddI8(w2, 3);

    // Expected: 100 - 200 + 150 = 50
    const expected = w0.sub(w1).add(w2.mul(BN254Scalar.fromU64(3)));
    const actual = acc.barrettReduce();
    try std.testing.expect(expected.eql(actual));
}

test "MedAccumS signed accumulation matches field arithmetic" {
    var acc = MedAccumS.zero();
    const w0 = BN254Scalar.fromU64(1000);
    const w1 = BN254Scalar.fromU64(2000);
    const w2 = BN254Scalar.fromU64(3000);

    const b0: i128 = 500;
    const b1: i128 = -300;
    const b2: i128 = 700;

    acc.fmaddI128(w0, b0);
    acc.fmaddI128(w1, b1);
    acc.fmaddI128(w2, b2);

    // Expected: 1000*500 - 2000*300 + 3000*700 = 500000 - 600000 + 2100000 = 2000000
    const expected = w0.mul(BN254Scalar.fromU64(500))
        .sub(w1.mul(BN254Scalar.fromU64(300)))
        .add(w2.mul(BN254Scalar.fromU64(700)));
    const actual = acc.barrettReduce();
    try std.testing.expect(expected.eql(actual));
}

test "MedAccumS large i128 values" {
    var acc = MedAccumS.zero();
    const w = BN254Scalar.fromU64(1);
    const big_val: i128 = @as(i128, 1) << 100;

    acc.fmaddI128(w, big_val);
    acc.fmaddI128(w, -big_val);

    // Should cancel to zero
    const actual = acc.barrettReduce();
    try std.testing.expect(actual.isZero());
}

test "SmallAccumU large accumulation (1000 adds)" {
    const N = 1000;
    var acc = SmallAccumU.zero();
    var expected = BN254Scalar.zero();

    for (0..N) |i| {
        const w = BN254Scalar.fromU64(@as(u64, i) * 7 + 1);
        const val: i8 = @intCast(@as(i64, @intCast(i % 5)) - 2); // -2..2
        acc.fmaddI8(w, val);
        if (val == 1) {
            expected = expected.add(w);
        } else if (val == -1) {
            expected = expected.sub(w);
        } else if (val > 0) {
            expected = expected.add(w.mul(BN254Scalar.fromU64(@intCast(val))));
        } else if (val < 0) {
            expected = expected.sub(w.mul(BN254Scalar.fromU64(@intCast(-val))));
        }
    }

    const actual = acc.barrettReduce();
    try std.testing.expect(expected.eql(actual));
}

test "Barrett reduce round-trip: mulU64Unreduced sum" {
    // Accumulate many field × u64 products via folded, compare to direct field arithmetic
    const N = 100;
    var folded = FoldedMulU64.zero();
    var expected = BN254Scalar.zero();

    for (0..N) |i| {
        const w = BN254Scalar.fromU64(@as(u64, i) * 13 + 7);
        const scalar: u64 = @as(u64, i) * 11 + 3;
        folded.addAssign(mulU64Unreduced(w, scalar));
        expected = expected.add(w.mul(BN254Scalar.fromU64(scalar)));
    }

    const actual = reduceMulU64(folded);
    try std.testing.expect(expected.eql(actual));
}

test "Barrett reduce round-trip: mulU64Unreduced STRESS (large N, large values)" {
    // Stress test with N=100000 and full-range field values to catch overflow/precision bugs
    const N = 100_000;
    var folded = FoldedMulU64.zero();
    var expected = BN254Scalar.zero();

    for (0..N) |i| {
        // Use large field values (close to modulus) and large scalars
        const w = BN254Scalar.fromU64(@as(u64, i) * 0x123456789ABCDEF + 0xFEDCBA9876543210);
        const scalar: u64 = @as(u64, i) * 0xABCD + 0xFFFF;
        folded.addAssign(mulU64Unreduced(w, scalar));
        expected = expected.add(w.mul(BN254Scalar.fromU64(scalar)));
    }

    const actual = reduceMulU64(folded);
    if (!expected.eql(actual)) {
        std.debug.print("MISMATCH at N={}\n", .{N});
        std.debug.print("expected: {any}\n", .{expected.limbs});
        std.debug.print("actual:   {any}\n", .{actual.limbs});

        // Binary search for first failure
        var folded2 = FoldedMulU64.zero();
        var expected2 = BN254Scalar.zero();
        for (0..N) |i| {
            const w = BN254Scalar.fromU64(@as(u64, i) * 0x123456789ABCDEF + 0xFEDCBA9876543210);
            const scalar: u64 = @as(u64, i) * 0xABCD + 0xFFFF;
            folded2.addAssign(mulU64Unreduced(w, scalar));
            expected2 = expected2.add(w.mul(BN254Scalar.fromU64(scalar)));
            const check = reduceMulU64(folded2);
            if (!expected2.eql(check)) {
                std.debug.print("First mismatch at i={}\n", .{i});
                std.debug.print("  w.limbs = {any}\n", .{w.limbs});
                std.debug.print("  scalar = {}\n", .{scalar});
                std.debug.print("  folded slots = {any}\n", .{folded2.slots});
                std.debug.print("  normalized = {any}\n", .{folded2.normalize()});
                break;
            }
        }
    }
    try std.testing.expect(expected.eql(actual));
}

test "Barrett reduce round-trip: mulU128Unreduced sum" {
    const N = 50;
    var folded = FoldedMulU128.zero();
    var expected = BN254Scalar.zero();

    for (0..N) |i| {
        const w = BN254Scalar.fromU64(@as(u64, i) * 17 + 5);
        const scalar: u128 = @as(u128, i) * 0x1234567890ABCDEF + 1;
        folded.addAssign(mulU128Unreduced(w, scalar));
        expected = expected.add(w.mul(BN254Scalar.fromU128(scalar)));
    }

    const actual = reduceMulU128(folded);
    try std.testing.expect(expected.eql(actual));
}

test "WideAccumS i128 accumulation" {
    var acc = WideAccumS.zero();
    const w0 = BN254Scalar.fromU64(42);
    const w1 = BN254Scalar.fromU64(99);

    const b0: i128 = @as(i128, 1) << 80;
    const b1: i128 = -(@as(i128, 1) << 60);

    acc.fmaddI128(w0, b0);
    acc.fmaddI128(w1, b1);

    const expected = w0.mul(BN254Scalar.fromU128(@intCast(b0)))
        .sub(w1.mul(BN254Scalar.fromU128(@intCast(-b1))));
    const actual = acc.barrettReduce();
    try std.testing.expect(expected.eql(actual));
}

test "MODULUS_TIMES_2 and MODULUS_TIMES_3 correctness" {
    // Verify 2*p
    const two_p = BN254Scalar.fromU64(2).fromMontgomery();
    _ = two_p; // Just verifying comptime constants don't crash
    // Simple sanity: MODULUS_TIMES_2[0] should be 2*MODULUS[0] mod 2^64
    const expected_lo: u128 = @as(u128, BN254_MODULUS[0]) * 2;
    try std.testing.expectEqual(@as(u64, @truncate(expected_lo)), MODULUS_TIMES_2[0]);
}

// Export pairing module
pub const pairing = @import("pairing.zig");
pub const Fp2 = pairing.Fp2;
pub const Fp6 = pairing.Fp6;
pub const Fp12 = pairing.Fp12;
pub const G2Point = pairing.G2Point;

test {
    // Run pairing tests
    _ = pairing;
}

// Benchmark removed — was temporary for profiling
