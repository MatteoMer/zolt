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
        : .{ .nzcv = true });
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
        : .{ .nzcv = true });
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
        : .{ .nzcv = true });
    return .{ r0, r1, r2, r3 };
}

/// 3-limb unsigned addition (for S192 magnitudes).
pub inline fn arm64Add192(a: [3]u64, b: [3]u64) [3]u64 {
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
        : .{ .nzcv = true });
    return .{ r0, r1, r2 };
}

/// 3-limb unsigned subtraction (for S192 magnitudes).
pub inline fn arm64Sub192(a: [3]u64, b: [3]u64) [3]u64 {
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
        : .{ .nzcv = true });
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
        : .{ .x0 = true, .x1 = true, .x2 = true, .x3 = true, .x4 = true, .x5 = true, .x6 = true, .x7 = true, .x8 = true, .x9 = true, .x10 = true, .x11 = true, .x14 = true, .x15 = true, .x16 = true, .x17 = true, .x19 = true, .x20 = true, .x21 = true, .x22 = true, .x23 = true, .x24 = true, .x25 = true, .x26 = true, .nzcv = true, .memory = true });
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
    pub inline fn mulWide(a: u64, b: u64) u128 {
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

    pub inline fn subBorrow(a: u64, b: u64, borrow_in: u64) struct { result: u64, borrow: u64 } {
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
            : .{ .rax = true, .rcx = true, .r13 = true, .cc = true, .memory = true });

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

    pub inline fn lessThanModulus(self: Self) bool {
        // Unrolled MSB-first comparison — matches arkworks BigInt::cmp pattern
        // Const modulus values allow LLVM to emit cmpq with immediates
        if (self.limbs[3] != BN254_MODULUS[3]) return self.limbs[3] < BN254_MODULUS[3];
        if (self.limbs[2] != BN254_MODULUS[2]) return self.limbs[2] < BN254_MODULUS[2];
        if (self.limbs[1] != BN254_MODULUS[1]) return self.limbs[1] < BN254_MODULUS[1];
        return self.limbs[0] < BN254_MODULUS[0];
    }

    pub inline fn subtractModulus(self: Self) Self {
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

// ============================================================================
// Re-exports from accumulators.zig
// ============================================================================

pub const accumulators = @import("accumulators.zig");
pub const UnreducedProductAccum = accumulators.UnreducedProductAccum;
pub const FoldedMulU64 = accumulators.FoldedMulU64;
pub const FoldedMulU128 = accumulators.FoldedMulU128;
pub const FoldedMulU128Accum = accumulators.FoldedMulU128Accum;
pub const barrettReduce = accumulators.barrettReduce;
pub const toUnreduced = accumulators.toUnreduced;
pub const mulU64Unreduced = accumulators.mulU64Unreduced;
pub const mulU128Unreduced = accumulators.mulU128Unreduced;
pub const mulU192Unreduced = accumulators.mulU192Unreduced;
pub const reduceMulU64 = accumulators.reduceMulU64;
pub const mulU64 = accumulators.mulU64;
pub const reduceMulU128 = accumulators.reduceMulU128;
pub const reduceMulU128Accum = accumulators.reduceMulU128Accum;
pub const SmallAccumU = accumulators.SmallAccumU;
pub const MedAccumS = accumulators.MedAccumS;
pub const WideAccumS = accumulators.WideAccumS;
pub const S192 = accumulators.S192;
pub const BatchOps = accumulators.BatchOps;

// ============================================================================
// Re-exports from simd_ops.zig
// ============================================================================

pub const simd_ops = @import("simd_ops.zig");
pub const SimdOps = simd_ops.SimdOps;

// Verify BN254Scalar implements JoltField interface
comptime {
    _ = JoltField(BN254Scalar);
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

test {
    // Run accumulator and simd_ops tests
    _ = accumulators;
    _ = simd_ops;
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
