//! Lookup Table Prefixes for Jolt-Compatible Sumcheck
//!
//! This module implements all 45+ prefix types from Jolt's lookup table system.
//! Each prefix provides:
//!   - prefix_mle(checkpoints, r_x, c, b, j): Evaluate the prefix MLE
//!   - update_checkpoint(checkpoints, r_x, r_y, j, suffix_len): Update checkpoint after 2 rounds
//!
//! The prefix-suffix decomposition allows efficient computation of degree-2 sumcheck
//! polynomials during address rounds of the LookupsReadRaf sumcheck.
//!
//! Reference: jolt-core/src/zkvm/lookup_table/prefixes/*.rs
const std = @import("std");

const zkvm_debug = @import("../debug.zig");
const dbg = zkvm_debug.dbg;

const Allocator = std.mem.Allocator;
/// LOG_K = 128 for RV64 (2*XLEN for interleaved operands)
pub const LOG_K: usize = 128;
/// Compute 2^exp as a field element
/// Handles large exponents (up to 128) that don't fit in u64
fn fieldPow2(comptime F: type, exp: usize) F {
    if (exp == 0) return F.one();
    // For small exponents, use direct computation
    if (exp < 64) {
        return F.fromU64(@as(u64, 1) << @intCast(exp));
    }
    // For large exponents, use repeated squaring
    // 2^exp = 2^64 * 2^(exp-64)
    const two_pow_64 = F.fromBytes(&[_]u8{
        0, 0, 0, 0, 0, 0, 0, 0, // Lower 8 bytes = 0
        1, 0, 0, 0, 0, 0, 0, 0, // 2^64 in little-endian
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
    });
    var result = two_pow_64;
    var remaining = exp - 64;
    while (remaining >= 64) {
        result = result.mul(two_pow_64);
        remaining -= 64;
    }
    if (remaining > 0) {
        result = result.mul(F.fromU64(@as(u64, 1) << @intCast(remaining)));
    }
    return result;
}
/// Safely compute suffix_len = LOG_K - j - b_len - 1
/// Returns null if the result would underflow (meaning no suffix bits remain)
fn safeSuffixLen(j: usize, b_len: usize) ?usize {
    if (j + b_len + 1 > LOG_K) {
        return null;
    }
    return LOG_K - j - b_len - 1;
}
/// XLEN = 64 for RV64
pub const XLEN: usize = 64;
/// LookupBits represents a bitvector for lookup indices
pub fn LookupBits(comptime max_bits: usize) type {
    return struct {
        const Self = @This();
        value: u128,
        len: usize,
        pub fn new(value: u128, len: usize) Self {
            std.debug.assert(len <= max_bits);
            return .{ .value = value, .len = len };
        }
        /// Pop the most significant bit
        pub fn popMsb(self: *Self) u8 {
            if (self.len == 0) return 0;
            self.len -= 1;
            const bit = @as(u8, @truncate((self.value >> @intCast(self.len)) & 1));
            self.value &= (@as(u128, 1) << @intCast(self.len)) - 1;
            return bit;
        }
        /// Get value as usize
        pub fn toUsize(self: *const Self) usize {
            return @intCast(self.value);
        }
        /// Uninterleave bits: split interleaved operands into (left_operand, right_operand)
        ///
        /// In Jolt's interleave format, operands are stored as:
        ///   interleaved = (left_operand << 1) | right_operand
        /// So left operand bits are at ODD positions (1, 3, 5, ...)
        /// and right operand bits are at EVEN positions (0, 2, 4, ...)
        ///
        /// This matches Jolt's uninterleave_bits() which returns (x, y) where:
        ///   x = bits from ODD positions = left operand
        ///   y = bits from EVEN positions = right operand
        pub fn uninterleave(self: *const Self) struct { left: u64, right: u64, left_len: usize, right_len: usize } {
            // Parallel bit extraction — O(1) using shift+mask+OR cascade.
            // Matches Jolt's uninterleave_bits (jolt-core/src/utils/mod.rs:105-124).
            // Left operand bits at odd positions (1,3,5,...), right at even (0,2,4,...).
            var x_bits: u128 = (self.value >> 1) & 0x5555_5555_5555_5555_5555_5555_5555_5555;
            var y_bits: u128 = self.value & 0x5555_5555_5555_5555_5555_5555_5555_5555;
            // Compact x bits into lower part
            x_bits = (x_bits | (x_bits >> 1)) & 0x3333_3333_3333_3333_3333_3333_3333_3333;
            x_bits = (x_bits | (x_bits >> 2)) & 0x0F0F_0F0F_0F0F_0F0F_0F0F_0F0F_0F0F_0F0F;
            x_bits = (x_bits | (x_bits >> 4)) & 0x00FF_00FF_00FF_00FF_00FF_00FF_00FF_00FF;
            x_bits = (x_bits | (x_bits >> 8)) & 0x0000_FFFF_0000_FFFF_0000_FFFF_0000_FFFF;
            x_bits = (x_bits | (x_bits >> 16)) & 0x0000_0000_FFFF_FFFF_0000_0000_FFFF_FFFF;
            x_bits = (x_bits | (x_bits >> 32)) & 0x0000_0000_0000_0000_FFFF_FFFF_FFFF_FFFF;
            // Compact y bits into lower part
            y_bits = (y_bits | (y_bits >> 1)) & 0x3333_3333_3333_3333_3333_3333_3333_3333;
            y_bits = (y_bits | (y_bits >> 2)) & 0x0F0F_0F0F_0F0F_0F0F_0F0F_0F0F_0F0F_0F0F;
            y_bits = (y_bits | (y_bits >> 4)) & 0x00FF_00FF_00FF_00FF_00FF_00FF_00FF_00FF;
            y_bits = (y_bits | (y_bits >> 8)) & 0x0000_FFFF_0000_FFFF_0000_FFFF_0000_FFFF;
            y_bits = (y_bits | (y_bits >> 16)) & 0x0000_0000_FFFF_FFFF_0000_0000_FFFF_FFFF;
            y_bits = (y_bits | (y_bits >> 32)) & 0x0000_0000_0000_0000_FFFF_FFFF_FFFF_FFFF;

            const half_len = self.len / 2;
            const left_len = half_len;
            const right_len = if (self.len % 2 == 1) half_len + 1 else half_len;
            return .{
                .left = @truncate(x_bits),
                .right = @truncate(y_bits),
                .left_len = left_len,
                .right_len = right_len,
            };
        }
        /// Split into (prefix, suffix) where suffix.len == suffix_len
        pub fn split(self: *const Self, suffix_len: usize) struct { prefix: Self, suffix: Self } {
            const suffix_bits = self.value & ((@as(u128, 1) << @intCast(suffix_len)) - 1);
            const prefix_bits = self.value >> @intCast(suffix_len);
            return .{
                .prefix = Self.new(prefix_bits, self.len - suffix_len),
                .suffix = Self.new(suffix_bits, suffix_len),
            };
        }
        /// Count trailing zeros, clamped to len
        pub fn trailingZeros(self: *const Self) u32 {
            if (self.value == 0) return @intCast(self.len);
            const tz = @ctz(self.value);
            return @min(tz, @as(u32, @intCast(self.len)));
        }
        /// Count leading ones (from MSB position self.len-1 down)
        pub fn leadingOnes(self: *const Self) u32 {
            if (self.len == 0) return 0;
            // Shift value so MSB of the value is at bit 127
            const shifted = self.value << @intCast(128 - self.len);
            return @clz(~shifted);
        }
    };
}
/// All prefix types used by Jolt's instruction lookup tables
pub const Prefixes = enum(u8) {
    LowerWord,
    LowerHalfWord,
    UpperWord,
    Eq,
    And,
    Andn,
    Or,
    Xor,
    LessThan,
    LeftOperandIsZero,
    RightOperandIsZero,
    LeftOperandMsb,
    RightOperandMsb,
    DivByZero,
    PositiveRemainderEqualsDivisor,
    PositiveRemainderLessThanDivisor,
    NegativeDivisorZeroRemainder,
    NegativeDivisorEqualsRemainder,
    NegativeDivisorGreaterThanRemainder,
    Lsb,
    Pow2,
    Pow2W,
    Rev8W,
    RightShift,
    SignExtension,
    LeftShift,
    LeftShiftHelper,
    TwoLsb,
    SignExtensionUpperHalf,
    ChangeDivisor,
    ChangeDivisorW,
    RightOperand,
    RightOperandW,
    SignExtensionRightOperand,
    RightShiftW,
    LeftShiftWHelper,
    LeftShiftW,
    OverflowBitsZero,
    XorRot16,
    XorRot24,
    XorRot32,
    XorRot63,
    XorRotW7,
    XorRotW8,
    XorRotW12,
    XorRotW16,
    pub const COUNT: usize = 46;
};
/// Wrapper for optional prefix evaluation
pub fn PrefixCheckpoint(comptime F: type) type {
    return ?F;
}
/// Array of all prefix checkpoints
pub fn PrefixCheckpoints(comptime F: type) type {
    return [Prefixes.COUNT]PrefixCheckpoint(F);
}
/// Evaluate a prefix MLE at the given parameters
///
/// Args:
///   checkpoints: Current prefix checkpoint values
///   r_x: Optional challenge for odd rounds (when j is odd)
///   c: The current variable value (0, 1, or 2)
///   b: Remaining bits to sum over
///   j: Current sumcheck round index
pub fn prefixMle(
    comptime F: type,
    prefix: Prefixes,
    checkpoints: *const PrefixCheckpoints(F),
    r_x: ?F,
    c: u32,
    b: *LookupBits(128),
    j: usize,
) F {
    return switch (prefix) {
        .Eq => eqPrefixMle(F, checkpoints, r_x, c, b, j),
        .LowerWord => lowerWordPrefixMle(F, checkpoints, r_x, c, b, j),
        .LowerHalfWord => lowerHalfWordPrefixMle(F, checkpoints, r_x, c, b, j),
        .UpperWord => upperWordPrefixMle(F, checkpoints, r_x, c, b, j),
        .And => andPrefixMle(F, checkpoints, r_x, c, b, j),
        .Andn => andnPrefixMle(F, checkpoints, r_x, c, b, j),
        .Or => orPrefixMle(F, checkpoints, r_x, c, b, j),
        .Xor => xorPrefixMle(F, checkpoints, r_x, c, b, j),
        .LessThan => lessThanPrefixMle(F, checkpoints, r_x, c, b, j),
        .LeftOperandIsZero => leftIsZeroPrefixMle(F, checkpoints, r_x, c, b, j),
        .RightOperandIsZero => rightIsZeroPrefixMle(F, checkpoints, r_x, c, b, j),
        .LeftOperandMsb => leftMsbPrefixMle(F, checkpoints, r_x, c, b, j),
        .RightOperandMsb => rightMsbPrefixMle(F, checkpoints, r_x, c, b, j),
        .DivByZero => divByZeroPrefixMle(F, checkpoints, r_x, c, b, j),
        .PositiveRemainderEqualsDivisor => positiveRemainderEqualsDivisorPrefixMle(F, checkpoints, r_x, c, b, j),
        .PositiveRemainderLessThanDivisor => positiveRemainderLessThanDivisorPrefixMle(F, checkpoints, r_x, c, b, j),
        .NegativeDivisorZeroRemainder => negativeDivisorZeroRemainderPrefixMle(F, checkpoints, r_x, c, b, j),
        .NegativeDivisorEqualsRemainder => negativeDivisorEqualsRemainderPrefixMle(F, checkpoints, r_x, c, b, j),
        .NegativeDivisorGreaterThanRemainder => negativeDivisorGreaterThanRemainderPrefixMle(F, checkpoints, r_x, c, b, j),
        .Lsb => lsbPrefixMle(F, checkpoints, r_x, c, b, j),
        .Pow2 => pow2PrefixMle(F, checkpoints, r_x, c, b, j),
        .Pow2W => pow2WPrefixMle(F, checkpoints, r_x, c, b, j),
        .Rev8W => rev8wPrefixMle(F, checkpoints, r_x, c, b, j),
        .RightShift => rightShiftPrefixMle(F, checkpoints, r_x, c, b, j),
        .SignExtension => signExtensionPrefixMle(F, checkpoints, r_x, c, b, j),
        .LeftShift => leftShiftPrefixMle(F, checkpoints, r_x, c, b, j),
        .LeftShiftHelper => leftShiftHelperPrefixMle(F, checkpoints, r_x, c, b, j),
        .TwoLsb => twoLsbPrefixMle(F, checkpoints, r_x, c, b, j),
        .SignExtensionUpperHalf => signExtensionUpperHalfPrefixMle(F, checkpoints, r_x, c, b, j),
        .ChangeDivisor => changeDivisorPrefixMle(F, checkpoints, r_x, c, b, j),
        .ChangeDivisorW => changeDivisorWPrefixMle(F, checkpoints, r_x, c, b, j),
        .RightOperand => rightOperandPrefixMle(F, checkpoints, r_x, c, b, j),
        .RightOperandW => rightOperandWPrefixMle(F, checkpoints, r_x, c, b, j),
        .SignExtensionRightOperand => signExtensionRightOperandPrefixMle(F, checkpoints, r_x, c, b, j),
        .RightShiftW => rightShiftWPrefixMle(F, checkpoints, r_x, c, b, j),
        .LeftShiftWHelper => leftShiftWHelperPrefixMle(F, checkpoints, r_x, c, b, j),
        .LeftShiftW => leftShiftWPrefixMle(F, checkpoints, r_x, c, b, j),
        .OverflowBitsZero => overflowBitsZeroPrefixMle(F, checkpoints, r_x, c, b, j),
        .XorRot16 => xorRotPrefixMle(F, 16, checkpoints, r_x, c, b, j),
        .XorRot24 => xorRotPrefixMle(F, 24, checkpoints, r_x, c, b, j),
        .XorRot32 => xorRotPrefixMle(F, 32, checkpoints, r_x, c, b, j),
        .XorRot63 => xorRotPrefixMle(F, 63, checkpoints, r_x, c, b, j),
        .XorRotW7 => xorRotWPrefixMle(F, 7, checkpoints, r_x, c, b, j),
        .XorRotW8 => xorRotWPrefixMle(F, 8, checkpoints, r_x, c, b, j),
        .XorRotW12 => xorRotWPrefixMle(F, 12, checkpoints, r_x, c, b, j),
        .XorRotW16 => xorRotWPrefixMle(F, 16, checkpoints, r_x, c, b, j),
    };
}
/// Update a prefix checkpoint after two rounds (r_x, r_y)
pub fn updatePrefixCheckpoint(
    comptime F: type,
    prefix: Prefixes,
    checkpoints: *const PrefixCheckpoints(F),
    r_x: F,
    r_y: F,
    j: usize,
    suffix_len: usize,
) PrefixCheckpoint(F) {
    return switch (prefix) {
        .Eq => eqUpdateCheckpoint(F, checkpoints, r_x, r_y, j, suffix_len),
        .LowerWord => lowerWordUpdateCheckpoint(F, checkpoints, r_x, r_y, j, suffix_len),
        .LowerHalfWord => lowerHalfWordUpdateCheckpoint(F, checkpoints, r_x, r_y, j, suffix_len),
        .UpperWord => upperWordUpdateCheckpoint(F, checkpoints, r_x, r_y, j, suffix_len),
        .And => andUpdateCheckpoint(F, checkpoints, r_x, r_y, j, suffix_len),
        .Andn => andnUpdateCheckpoint(F, checkpoints, r_x, r_y, j, suffix_len),
        .Or => orUpdateCheckpoint(F, checkpoints, r_x, r_y, j, suffix_len),
        .Xor => xorUpdateCheckpoint(F, checkpoints, r_x, r_y, j, suffix_len),
        .LessThan => lessThanUpdateCheckpoint(F, checkpoints, r_x, r_y, j, suffix_len),
        .LeftOperandIsZero => leftIsZeroUpdateCheckpoint(F, checkpoints, r_x, r_y, j, suffix_len),
        .RightOperandIsZero => rightIsZeroUpdateCheckpoint(F, checkpoints, r_x, r_y, j, suffix_len),
        .LeftOperandMsb => leftMsbUpdateCheckpoint(F, checkpoints, r_x, r_y, j, suffix_len),
        .RightOperandMsb => rightMsbUpdateCheckpoint(F, checkpoints, r_x, r_y, j, suffix_len),
        .DivByZero => divByZeroUpdateCheckpoint(F, checkpoints, r_x, r_y, j, suffix_len),
        .PositiveRemainderEqualsDivisor => positiveRemainderEqualsDivisorUpdateCheckpoint(F, checkpoints, r_x, r_y, j, suffix_len),
        .PositiveRemainderLessThanDivisor => positiveRemainderLessThanDivisorUpdateCheckpoint(F, checkpoints, r_x, r_y, j, suffix_len),
        .NegativeDivisorZeroRemainder => negativeDivisorZeroRemainderUpdateCheckpoint(F, checkpoints, r_x, r_y, j, suffix_len),
        .NegativeDivisorEqualsRemainder => negativeDivisorEqualsRemainderUpdateCheckpoint(F, checkpoints, r_x, r_y, j, suffix_len),
        .NegativeDivisorGreaterThanRemainder => negativeDivisorGreaterThanRemainderUpdateCheckpoint(F, checkpoints, r_x, r_y, j, suffix_len),
        .Lsb => lsbUpdateCheckpoint(F, checkpoints, r_x, r_y, j, suffix_len),
        .Pow2 => pow2UpdateCheckpoint(F, checkpoints, r_x, r_y, j, suffix_len),
        .Pow2W => pow2WUpdateCheckpoint(F, checkpoints, r_x, r_y, j, suffix_len),
        .Rev8W => rev8wUpdateCheckpoint(F, checkpoints, r_x, r_y, j, suffix_len),
        .RightShift => rightShiftUpdateCheckpoint(F, checkpoints, r_x, r_y, j, suffix_len),
        .SignExtension => signExtensionUpdateCheckpoint(F, checkpoints, r_x, r_y, j, suffix_len),
        .LeftShift => leftShiftUpdateCheckpoint(F, checkpoints, r_x, r_y, j, suffix_len),
        .LeftShiftHelper => leftShiftHelperUpdateCheckpoint(F, checkpoints, r_x, r_y, j, suffix_len),
        .TwoLsb => twoLsbUpdateCheckpoint(F, checkpoints, r_x, r_y, j, suffix_len),
        .SignExtensionUpperHalf => signExtensionUpperHalfUpdateCheckpoint(F, checkpoints, r_x, r_y, j, suffix_len),
        .ChangeDivisor => changeDivisorUpdateCheckpoint(F, checkpoints, r_x, r_y, j, suffix_len),
        .ChangeDivisorW => changeDivisorWUpdateCheckpoint(F, checkpoints, r_x, r_y, j, suffix_len),
        .RightOperand => rightOperandUpdateCheckpoint(F, checkpoints, r_x, r_y, j, suffix_len),
        .RightOperandW => rightOperandWUpdateCheckpoint(F, checkpoints, r_x, r_y, j, suffix_len),
        .SignExtensionRightOperand => signExtensionRightOperandUpdateCheckpoint(F, checkpoints, r_x, r_y, j, suffix_len),
        .RightShiftW => rightShiftWUpdateCheckpoint(F, checkpoints, r_x, r_y, j, suffix_len),
        .LeftShiftWHelper => leftShiftWHelperUpdateCheckpoint(F, checkpoints, r_x, r_y, j, suffix_len),
        .LeftShiftW => leftShiftWUpdateCheckpoint(F, checkpoints, r_x, r_y, j, suffix_len),
        .OverflowBitsZero => overflowBitsZeroUpdateCheckpoint(F, checkpoints, r_x, r_y, j, suffix_len),
        .XorRot16 => xorRotUpdateCheckpoint(F, 16, checkpoints, r_x, r_y, j, suffix_len),
        .XorRot24 => xorRotUpdateCheckpoint(F, 24, checkpoints, r_x, r_y, j, suffix_len),
        .XorRot32 => xorRotUpdateCheckpoint(F, 32, checkpoints, r_x, r_y, j, suffix_len),
        .XorRot63 => xorRotUpdateCheckpoint(F, 63, checkpoints, r_x, r_y, j, suffix_len),
        .XorRotW7 => xorRotWUpdateCheckpoint(F, 7, checkpoints, r_x, r_y, j, suffix_len),
        .XorRotW8 => xorRotWUpdateCheckpoint(F, 8, checkpoints, r_x, r_y, j, suffix_len),
        .XorRotW12 => xorRotWUpdateCheckpoint(F, 12, checkpoints, r_x, r_y, j, suffix_len),
        .XorRotW16 => xorRotWUpdateCheckpoint(F, 16, checkpoints, r_x, r_y, j, suffix_len),
    };
}
/// Update all prefix checkpoints
pub fn updateAllCheckpoints(
    comptime F: type,
    checkpoints: *PrefixCheckpoints(F),
    r_x: F,
    r_y: F,
    j: usize,
    suffix_len: usize,
) void {
    const prev_checkpoints = checkpoints.*;
    inline for (0..Prefixes.COUNT) |i| {
        const prefix: Prefixes = @enumFromInt(i);
        checkpoints[i] = updatePrefixCheckpoint(F, prefix, &prev_checkpoints, r_x, r_y, j, suffix_len);
    }
}
// ============================================================================
// Eq Prefix Implementation
// ============================================================================
/// EqPrefix: eq(x, y) = Π_i (x_i * y_i + (1-x_i) * (1-y_i))
fn eqPrefixMle(
    comptime F: type,
    checkpoints: *const PrefixCheckpoints(F),
    r_x: ?F,
    c: u32,
    b: *LookupBits(128),
    _: usize, // j unused
) F {
    var result = checkpoints[@intFromEnum(Prefixes.Eq)] orelse F.one();
    // EQ high-order variables of x and y
    if (r_x) |rx| {
        const y = F.fromU64(@as(u64, c));
        // eq(r_x, c) = r_x * c + (1 - r_x) * (1 - c)
        result = result.mul(rx.mul(y).add(F.one().sub(rx).mul(F.one().sub(y))));
    } else {
        const x = F.fromU64(@as(u64, c));
        const y_msb = F.fromU64(@as(u64, b.popMsb()));
        // eq(c, y_msb) = c * y_msb + (1 - c) * (1 - y_msb)
        result = result.mul(x.mul(y_msb).add(F.one().sub(x).mul(F.one().sub(y_msb))));
    }
    // EQ remaining x and y bits - if they don't match, return zero
    const uninterleaved = b.uninterleave();
    if (uninterleaved.left != uninterleaved.right) {
        return F.zero();
    }
    return result;
}
fn eqUpdateCheckpoint(
    comptime F: type,
    checkpoints: *const PrefixCheckpoints(F),
    r_x: F,
    r_y: F,
    _: usize, // j unused
    _: usize, // suffix_len unused
) PrefixCheckpoint(F) {
    // checkpoint *= r_x * r_y + (1 - r_x) * (1 - r_y)
    const prev = checkpoints[@intFromEnum(Prefixes.Eq)] orelse F.one();
    const updated = prev.mul(r_x.mul(r_y).add(F.one().sub(r_x).mul(F.one().sub(r_y))));
    return updated;
}
// ============================================================================
// LowerWord Prefix Implementation
// ============================================================================
/// LowerWord: accumulates the lower XLEN bits of the interleaved index
fn lowerWordPrefixMle(
    comptime F: type,
    checkpoints: *const PrefixCheckpoints(F),
    r_x: ?F,
    c: u32,
    b: *LookupBits(128),
    j: usize,
) F {
    // Compute suffix_len BEFORE any popMsb, matching Jolt's order
    // suffix_len = LOG_K - j - b.len - 1
    const original_b_len = b.len;
    const suffix_len_opt = safeSuffixLen(j, original_b_len);

    // Ignore high-order variables (first XLEN rounds)
    if (j < XLEN) {
        return F.zero();
    }
    var result = checkpoints[@intFromEnum(Prefixes.LowerWord)] orelse F.zero();
    if (r_x) |rx| {
        const y = F.fromU64(@as(u64, c));
        const x_shift = 2 * XLEN - j;
        const y_shift = 2 * XLEN - j - 1;
        result = result.add(F.fromU128(@as(u128, 1) << @intCast(x_shift)).mul(rx));
        result = result.add(F.fromU128(@as(u128, 1) << @intCast(y_shift)).mul(y));
    } else {
        const x = F.fromU64(@as(u64, c));
        const y_msb = b.popMsb();
        const x_shift = 2 * XLEN - j - 1;
        const y_shift = 2 * XLEN - j - 2;
        result = result.add(F.fromU128(@as(u128, 1) << @intCast(x_shift)).mul(x));
        result = result.add(F.fromU128(@as(u128, 1) << @intCast(y_shift)).mul(F.fromU64(@as(u64, y_msb))));
    }
    // Add in low-order bits from b using suffix_len computed with ORIGINAL b.len
    const suffix_len = suffix_len_opt orelse return result;
    result = result.add(F.fromU128(b.value << @intCast(suffix_len)));
    return result;
}
fn lowerWordUpdateCheckpoint(
    comptime F: type,
    checkpoints: *const PrefixCheckpoints(F),
    r_x: F,
    r_y: F,
    j: usize,
    _: usize, // suffix_len unused
) PrefixCheckpoint(F) {
    if (j < XLEN) {
        return null;
    }
    const x_shift = 2 * XLEN - j;
    const y_shift = 2 * XLEN - j - 1;
    var updated = checkpoints[@intFromEnum(Prefixes.LowerWord)] orelse F.zero();
    // Compute contributions
    const coeff_x = F.fromU128(@as(u128, 1) << @intCast(x_shift));
    const coeff_y = F.fromU128(@as(u128, 1) << @intCast(y_shift));
    const contrib_x = coeff_x.mul(r_x);
    const contrib_y = coeff_y.mul(r_y);
    // Debug: print first and last updates with full details
    if (j == 65 or j == 127) {
        dbg("[LOWERWORD UPDATE] j={}, x_shift={}, y_shift={}\n", .{ j, x_shift, y_shift });
        dbg("  r_x (limbs)      = [0x{x:0>16}, 0x{x:0>16}, 0x{x:0>16}, 0x{x:0>16}]\n", .{
            r_x.limbs[0], r_x.limbs[1], r_x.limbs[2], r_x.limbs[3],
        });
        dbg("  r_y (limbs)      = [0x{x:0>16}, 0x{x:0>16}, 0x{x:0>16}, 0x{x:0>16}]\n", .{
            r_y.limbs[0], r_y.limbs[1], r_y.limbs[2], r_y.limbs[3],
        });
        dbg("  coeff_x (limbs)  = [0x{x:0>16}, 0x{x:0>16}, 0x{x:0>16}, 0x{x:0>16}]\n", .{
            coeff_x.limbs[0], coeff_x.limbs[1], coeff_x.limbs[2], coeff_x.limbs[3],
        });
        dbg("  contrib_x (limbs)= [0x{x:0>16}, 0x{x:0>16}, 0x{x:0>16}, 0x{x:0>16}]\n", .{
            contrib_x.limbs[0], contrib_x.limbs[1], contrib_x.limbs[2], contrib_x.limbs[3],
        });
        dbg("  contrib_y (limbs)= [0x{x:0>16}, 0x{x:0>16}, 0x{x:0>16}, 0x{x:0>16}]\n", .{
            contrib_y.limbs[0], contrib_y.limbs[1], contrib_y.limbs[2], contrib_y.limbs[3],
        });
        dbg("  prev (BE bytes)  = {x}\n", .{updated.toBytesBE()[16..32].*});
    }
    updated = updated.add(contrib_x);
    updated = updated.add(contrib_y);
    if (j == 65 or j == 127) {
        dbg("  new  (BE bytes)  = {x}\n", .{updated.toBytesBE()[16..32].*});
        dbg("  new  (limbs)     = [0x{x:0>16}, 0x{x:0>16}, 0x{x:0>16}, 0x{x:0>16}]\n", .{
            updated.limbs[0], updated.limbs[1], updated.limbs[2], updated.limbs[3],
        });
    }
    return updated;
}
// ============================================================================
// UpperWord Prefix Implementation
// ============================================================================
fn upperWordPrefixMle(
    comptime F: type,
    checkpoints: *const PrefixCheckpoints(F),
    r_x: ?F,
    c: u32,
    b: *LookupBits(128),
    j: usize,
) F {
    // Compute suffix_len BEFORE any popMsb, matching Jolt's order
    const original_b_len = b.len;
    const suffix_len_opt = safeSuffixLen(j, original_b_len);

    var result = checkpoints[@intFromEnum(Prefixes.UpperWord)] orelse F.zero();

    // Ignore low-order variables (only active during upper XLEN rounds)
    if (j >= XLEN) {
        return result;
    }

    if (r_x) |rx| {
        const y = F.fromU64(@as(u64, c));
        // UpperWord uses XLEN - j (not 2*XLEN - j like LowerWord)
        const x_shift = XLEN - j;
        const y_shift = XLEN - j - 1;
        result = result.add(F.fromU64(@as(u64, 1) << @intCast(x_shift)).mul(rx));
        result = result.add(F.fromU64(@as(u64, 1) << @intCast(y_shift)).mul(y));
    } else {
        const x = F.fromU64(@as(u64, c));
        const y_msb = b.popMsb();
        // UpperWord uses XLEN - j (not 2*XLEN - j like LowerWord)
        const x_shift = XLEN - j - 1;
        const y_shift = XLEN - j - 2;
        result = result.add(F.fromU64(@as(u64, 1) << @intCast(x_shift)).mul(x));
        result = result.add(F.fromU64(@as(u64, 1) << @intCast(y_shift)).mul(F.fromU64(@as(u64, y_msb))));
    }

    // Add in bits of `b` that fall in upper word
    // This is different from LowerWord - we need to extract only the upper word bits
    const suffix_len = suffix_len_opt orelse return result;
    if (suffix_len > XLEN) {
        // All remaining bits fit in upper word
        result = result.add(F.fromU64(@truncate(b.value << @intCast(suffix_len - XLEN))));
    } else {
        // Need to extract only the high bits that fall in upper word
        // Split b into (b_high, b_low) where b_low has (XLEN - suffix_len) bits
        const split_bits = XLEN - suffix_len;
        if (split_bits < b.len) {
            const b_high = b.value >> @intCast(split_bits);
            result = result.add(F.fromU64(@truncate(b_high)));
        }
    }
    return result;
}
fn upperWordUpdateCheckpoint(
    comptime F: type,
    checkpoints: *const PrefixCheckpoints(F),
    r_x: F,
    r_y: F,
    j: usize,
    _: usize,
) PrefixCheckpoint(F) {
    if (j >= XLEN) {
        // Once we're past upper word rounds, just preserve checkpoint
        return checkpoints[@intFromEnum(Prefixes.UpperWord)];
    }
    // UpperWord uses XLEN - j (not 2*XLEN - j like LowerWord)
    const x_shift = XLEN - j;
    const y_shift = XLEN - j - 1;
    var updated = checkpoints[@intFromEnum(Prefixes.UpperWord)] orelse F.zero();
    updated = updated.add(F.fromU64(@as(u64, 1) << @intCast(x_shift)).mul(r_x));
    updated = updated.add(F.fromU64(@as(u64, 1) << @intCast(y_shift)).mul(r_y));
    return updated;
}
// ============================================================================
// Generic bitwise binary prefix scaffold
// ============================================================================
// Shared by And, Or, Xor, Andn — prefixes that accumulate
//   result += 2^shift * evalFn(x_i, y_i)
// per interleaved bit-pair, where shift = XLEN-1 - (j/2).

/// Generic MLE for bitwise binary prefixes.
/// `evalFn` computes the field-level contribution from two challenge values.
/// `suffixFn` computes the integer bitwise operation on uninterleaved suffix operands.
fn bitwiseBinaryPrefixMle(
    comptime F: type,
    comptime evalFn: fn (F, F) F,
    comptime suffixFn: fn (u64, u64) u64,
    comptime prefix_idx: Prefixes,
    checkpoints: *const PrefixCheckpoints(F),
    r_x: ?F,
    c: u32,
    b: *LookupBits(128),
    j: usize,
) F {
    // Compute suffix_len BEFORE any popMsb, matching Jolt's order
    const original_b_len = b.len;
    const suffix_len_opt = safeSuffixLen(j, original_b_len);

    var result = checkpoints[@intFromEnum(prefix_idx)] orelse F.zero();
    const x_shift = XLEN - 1 - (j / 2);
    const coeff = F.fromU128(@as(u128, 1) << @intCast(x_shift));
    if (r_x) |rx| {
        const y = F.fromU64(@as(u64, c));
        result = result.add(coeff.mul(evalFn(rx, y)));
    } else {
        const x = F.fromU64(@as(u64, c));
        const y_msb_val = F.fromU64(@as(u64, b.popMsb()));
        result = result.add(coeff.mul(evalFn(x, y_msb_val)));
    }
    // Process remaining bits using suffix_len computed with ORIGINAL b.len
    const suffix_len = suffix_len_opt orelse return result;
    const uninterleaved = b.uninterleave();
    const suffix_val = suffixFn(uninterleaved.left, uninterleaved.right);
    result = result.add(F.fromU128(@as(u128, suffix_val) << @intCast(suffix_len / 2)));
    return result;
}

/// Generic checkpoint update for bitwise binary prefixes.
fn bitwiseBinaryUpdateCheckpoint(
    comptime F: type,
    comptime evalFn: fn (F, F) F,
    comptime prefix_idx: Prefixes,
    checkpoints: *const PrefixCheckpoints(F),
    r_x: F,
    r_y: F,
    j: usize,
) PrefixCheckpoint(F) {
    const x_shift = XLEN - 1 - (j / 2);
    var updated = checkpoints[@intFromEnum(prefix_idx)] orelse F.zero();
    updated = updated.add(F.fromU128(@as(u128, 1) << @intCast(x_shift)).mul(evalFn(r_x, r_y)));
    return updated;
}

// ============================================================================
// And Prefix Implementation
// ============================================================================
fn andPrefixMle(comptime F: type, checkpoints: *const PrefixCheckpoints(F), r_x: ?F, c: u32, b: *LookupBits(128), j: usize) F {
    return bitwiseBinaryPrefixMle(F, struct {
        fn eval(a: F, b_val: F) F {
            return a.mul(b_val);
        }
    }.eval, struct {
        fn f(l: u64, r: u64) u64 {
            return l & r;
        }
    }.f, .And, checkpoints, r_x, c, b, j);
}
fn andUpdateCheckpoint(comptime F: type, checkpoints: *const PrefixCheckpoints(F), r_x: F, r_y: F, j: usize, _: usize) PrefixCheckpoint(F) {
    return bitwiseBinaryUpdateCheckpoint(F, struct {
        fn eval(a: F, b_val: F) F {
            return a.mul(b_val);
        }
    }.eval, .And, checkpoints, r_x, r_y, j);
}
// ============================================================================
// Or Prefix Implementation
// ============================================================================
fn orPrefixMle(comptime F: type, checkpoints: *const PrefixCheckpoints(F), r_x: ?F, c: u32, b: *LookupBits(128), j: usize) F {
    return bitwiseBinaryPrefixMle(F, struct {
        fn eval(a: F, b_val: F) F {
            return a.add(b_val).sub(a.mul(b_val));
        }
    }.eval, struct {
        fn f(l: u64, r: u64) u64 {
            return l | r;
        }
    }.f, .Or, checkpoints, r_x, c, b, j);
}
fn orUpdateCheckpoint(comptime F: type, checkpoints: *const PrefixCheckpoints(F), r_x: F, r_y: F, j: usize, _: usize) PrefixCheckpoint(F) {
    return bitwiseBinaryUpdateCheckpoint(F, struct {
        fn eval(a: F, b_val: F) F {
            return a.add(b_val).sub(a.mul(b_val));
        }
    }.eval, .Or, checkpoints, r_x, r_y, j);
}
// ============================================================================
// Xor Prefix Implementation
// ============================================================================
fn xorPrefixMle(comptime F: type, checkpoints: *const PrefixCheckpoints(F), r_x: ?F, c: u32, b: *LookupBits(128), j: usize) F {
    return bitwiseBinaryPrefixMle(F, struct {
        fn eval(a: F, b_val: F) F {
            const two = F.fromU64(2);
            return a.add(b_val).sub(two.mul(a.mul(b_val)));
        }
    }.eval, struct {
        fn f(l: u64, r: u64) u64 {
            return l ^ r;
        }
    }.f, .Xor, checkpoints, r_x, c, b, j);
}
fn xorUpdateCheckpoint(comptime F: type, checkpoints: *const PrefixCheckpoints(F), r_x: F, r_y: F, j: usize, _: usize) PrefixCheckpoint(F) {
    return bitwiseBinaryUpdateCheckpoint(F, struct {
        fn eval(a: F, b_val: F) F {
            const two = F.fromU64(2);
            return a.add(b_val).sub(two.mul(a.mul(b_val)));
        }
    }.eval, .Xor, checkpoints, r_x, r_y, j);
}
// ============================================================================
// LessThan Prefix Implementation
// ============================================================================
fn lessThanPrefixMle(
    comptime F: type,
    checkpoints: *const PrefixCheckpoints(F),
    r_x: ?F,
    c: u32,
    b: *LookupBits(128),
    _: usize, // j unused
) F {
    // LessThan depends on the Eq checkpoint
    // Reference: jolt-core/src/zkvm/lookup_table/prefixes/lt.rs
    var eq = checkpoints[@intFromEnum(Prefixes.Eq)] orelse F.one();
    var lt = checkpoints[@intFromEnum(Prefixes.LessThan)] orelse F.zero();
    if (r_x) |rx| {
        const y = F.fromU64(@as(u64, c));
        // LT contribution from current bit: eq_prev * (1 - r_x) * c
        lt = lt.add(eq.mul(F.one().sub(rx)).mul(y));
        // Check if remaining suffix bits have x < y
        const uninterleaved = b.uninterleave();
        if (uninterleaved.left < uninterleaved.right) {
            // Update eq for this pair, then add suffix lt contribution
            eq = eq.mul(rx.mul(y).add(F.one().sub(rx).mul(F.one().sub(y))));
            lt = lt.add(eq);
        }
    } else {
        const x = F.fromU64(@as(u64, c));
        const y_msb_val = F.fromU64(@as(u64, b.popMsb()));
        // LT contribution from current bit: eq_prev * (1 - x) * y_msb
        lt = lt.add(eq.mul(F.one().sub(x)).mul(y_msb_val));
        // Check if remaining suffix bits have x < y
        const uninterleaved = b.uninterleave();
        if (uninterleaved.left < uninterleaved.right) {
            // Update eq for this pair, then add suffix lt contribution
            eq = eq.mul(x.mul(y_msb_val).add(F.one().sub(x).mul(F.one().sub(y_msb_val))));
            lt = lt.add(eq);
        }
    }
    return lt;
}
fn lessThanUpdateCheckpoint(
    comptime F: type,
    checkpoints: *const PrefixCheckpoints(F),
    r_x: F,
    r_y: F,
    _: usize,
    _: usize,
) PrefixCheckpoint(F) {
    const eq_checkpoint = checkpoints[@intFromEnum(Prefixes.Eq)] orelse F.one();
    var updated = checkpoints[@intFromEnum(Prefixes.LessThan)] orelse F.zero();
    // LT contribution: eq_prev * (1 - r_x) * r_y
    updated = updated.add(eq_checkpoint.mul(F.one().sub(r_x)).mul(r_y));
    return updated;
}
// ============================================================================
// LeftOperandIsZero Prefix Implementation
// ============================================================================
fn leftIsZeroPrefixMle(
    comptime F: type,
    checkpoints: *const PrefixCheckpoints(F),
    r_x: ?F,
    c: u32,
    b: *LookupBits(128),
    _: usize,
) F {
    // Uninterleave to get the x (left operand) bits
    // Short-circuit: if any remaining x-bit is non-zero, the prefix is zero
    // (because LeftOperandIsZero = Π (1-x_i), and any x_i=1 makes it 0)
    const uninterleaved = b.uninterleave();
    if (uninterleaved.left != 0) {
        return F.zero();
    }

    var result = checkpoints[@intFromEnum(Prefixes.LeftOperandIsZero)] orelse F.one();
    if (r_x) |rx| {
        // On odd rounds (when r_x is present), c is the y-value, not x
        // We need to multiply by (1 - r_x) for the left operand
        result = result.mul(F.one().sub(rx));
    } else {
        // On even rounds, c is the x-value
        const x = F.fromU64(@as(u64, c));
        _ = b.popMsb(); // discard y bit
        result = result.mul(F.one().sub(x));
    }
    return result;
}
fn leftIsZeroUpdateCheckpoint(
    comptime F: type,
    checkpoints: *const PrefixCheckpoints(F),
    r_x: F,
    _: F, // r_y not used for left operand
    _: usize,
    _: usize,
) PrefixCheckpoint(F) {
    var updated = checkpoints[@intFromEnum(Prefixes.LeftOperandIsZero)] orelse F.one();
    updated = updated.mul(F.one().sub(r_x));
    return updated;
}
// ============================================================================
// RightOperandIsZero Prefix Implementation
// ============================================================================
fn rightIsZeroPrefixMle(
    comptime F: type,
    checkpoints: *const PrefixCheckpoints(F),
    r_x: ?F,
    c: u32,
    b: *LookupBits(128),
    _: usize,
) F {
    // Uninterleave to get the y (right operand) bits
    // Short-circuit: if any remaining y-bit is non-zero, the prefix is zero
    // (because RightOperandIsZero = Π (1-y_i), and any y_i=1 makes it 0)
    const uninterleaved = b.uninterleave();
    if (uninterleaved.right != 0) {
        return F.zero();
    }

    var result = checkpoints[@intFromEnum(Prefixes.RightOperandIsZero)] orelse F.one();
    if (r_x) |_| {
        // On odd rounds, c is the y-value
        const y = F.fromU64(@as(u64, c));
        result = result.mul(F.one().sub(y));
    } else {
        // On even rounds, c is the x-value, y comes from b
        // We only care about y for RightOperand, so discard x (c)
        const y_msb_val = F.fromU64(@as(u64, b.popMsb()));
        result = result.mul(F.one().sub(y_msb_val));
    }
    return result;
}
fn rightIsZeroUpdateCheckpoint(
    comptime F: type,
    checkpoints: *const PrefixCheckpoints(F),
    _: F, // r_x not used for right operand
    r_y: F,
    _: usize,
    _: usize,
) PrefixCheckpoint(F) {
    var updated = checkpoints[@intFromEnum(Prefixes.RightOperandIsZero)] orelse F.one();
    updated = updated.mul(F.one().sub(r_y));
    return updated;
}
// ============================================================================
// LeftOperandMsb Prefix Implementation
// ============================================================================
fn leftMsbPrefixMle(
    comptime F: type,
    checkpoints: *const PrefixCheckpoints(F),
    r_x: ?F,
    c: u32,
    _: *LookupBits(128),
    j: usize,
) F {
    // j == 0: even round, c is the left MSB variable being sumchecked
    if (j == 0) {
        return F.fromU64(@as(u64, c));
    }
    // j == 1: odd round, r_x is the bound challenge from round 0 (left MSB)
    if (j == 1) {
        return r_x.?;
    }
    // j >= 2: use checkpoint (set after round 1)
    return checkpoints[@intFromEnum(Prefixes.LeftOperandMsb)].?;
}
fn leftMsbUpdateCheckpoint(
    comptime F: type,
    checkpoints: *const PrefixCheckpoints(F),
    r_x: F,
    _: F,
    j: usize,
    _: usize,
) PrefixCheckpoint(F) {
    // LeftOperandMsb is the X challenge from round 0 (bit 127, the leftmost/MSB bit)
    // Updates happen at odd rounds (1, 3, 5...), so the first update is at round 1
    // At round 1, r_x is challenges[0] which is the leftMsb value we need
    if (j == 1) {
        // First update: r_x is challenges[0], the left operand MSB
        return r_x;
    }
    if (j > 1) {
        return checkpoints[@intFromEnum(Prefixes.LeftOperandMsb)];
    }
    // j == 0: would return r_x, but updates don't happen at even rounds
    return r_x;
}
// ============================================================================
// RightOperandMsb Prefix Implementation
// ============================================================================
fn rightMsbPrefixMle(
    comptime F: type,
    checkpoints: *const PrefixCheckpoints(F),
    r_x: ?F,
    c: u32,
    b: *LookupBits(128),
    j: usize,
) F {
    // Only active in round 0 (or 1 for the MSB)
    if (j > 1) {
        return checkpoints[@intFromEnum(Prefixes.RightOperandMsb)] orelse F.zero();
    }
    if (r_x) |_| {
        // On odd rounds, c is the y-value we want for right MSB
        return F.fromU64(@as(u64, c));
    } else {
        // On even rounds, y comes from the MSB of b
        return F.fromU64(@as(u64, b.popMsb()));
    }
}
fn rightMsbUpdateCheckpoint(
    comptime F: type,
    checkpoints: *const PrefixCheckpoints(F),
    _: F,
    r_y: F,
    j: usize,
    _: usize,
) PrefixCheckpoint(F) {
    if (j > 1) {
        return checkpoints[@intFromEnum(Prefixes.RightOperandMsb)];
    }
    return r_y;
}
// ============================================================================
// Andn Prefix Implementation
// ============================================================================
fn andnPrefixMle(comptime F: type, checkpoints: *const PrefixCheckpoints(F), r_x: ?F, c: u32, b: *LookupBits(128), j: usize) F {
    return bitwiseBinaryPrefixMle(F, struct {
        fn eval(a: F, b_val: F) F {
            return a.mul(F.one().sub(b_val));
        }
    }.eval, struct {
        fn f(l: u64, r: u64) u64 {
            return l & ~r;
        }
    }.f, .Andn, checkpoints, r_x, c, b, j);
}
fn andnUpdateCheckpoint(comptime F: type, checkpoints: *const PrefixCheckpoints(F), r_x: F, r_y: F, j: usize, _: usize) PrefixCheckpoint(F) {
    return bitwiseBinaryUpdateCheckpoint(F, struct {
        fn eval(a: F, b_val: F) F {
            return a.mul(F.one().sub(b_val));
        }
    }.eval, .Andn, checkpoints, r_x, r_y, j);
}
// ============================================================================
// LowerHalfWord Prefix Implementation
// ============================================================================
fn lowerHalfWordPrefixMle(
    comptime F: type,
    checkpoints: *const PrefixCheckpoints(F),
    r_x: ?F,
    c: u32,
    b: *LookupBits(128),
    j: usize,
) F {
    // Compute suffix_len BEFORE any popMsb, matching Jolt's order
    const original_b_len = b.len;
    const suffix_len_opt = safeSuffixLen(j, original_b_len);

    const half_word_size = XLEN / 2;
    // Ignore high-order variables (those above the half-word boundary)
    if (j < XLEN + half_word_size) {
        return F.zero();
    }
    var result = checkpoints[@intFromEnum(Prefixes.LowerHalfWord)] orelse F.zero();
    if (r_x) |rx| {
        const y = F.fromU64(@as(u64, c));
        const x_shift = 2 * XLEN - j;
        const y_shift = 2 * XLEN - j - 1;
        result = result.add(F.fromU128(@as(u128, 1) << @intCast(x_shift)).mul(rx));
        result = result.add(F.fromU128(@as(u128, 1) << @intCast(y_shift)).mul(y));
    } else {
        const x = F.fromU64(@as(u64, c));
        const y_msb = b.popMsb();
        const x_shift = 2 * XLEN - j - 1;
        const y_shift = 2 * XLEN - j - 2;
        result = result.add(F.fromU128(@as(u128, 1) << @intCast(x_shift)).mul(x));
        result = result.add(F.fromU128(@as(u128, 1) << @intCast(y_shift)).mul(F.fromU64(@as(u64, y_msb))));
    }
    // Add in low-order bits from b using suffix_len computed with ORIGINAL b.len
    const suffix_len = suffix_len_opt orelse return result;
    result = result.add(F.fromU128(b.value << @intCast(suffix_len)));
    return result;
}
fn lowerHalfWordUpdateCheckpoint(
    comptime F: type,
    checkpoints: *const PrefixCheckpoints(F),
    r_x: F,
    r_y: F,
    j: usize,
    _: usize,
) PrefixCheckpoint(F) {
    const half_word_size = XLEN / 2;
    if (j < XLEN + half_word_size) {
        return null;
    }
    const x_shift = 2 * XLEN - j;
    const y_shift = 2 * XLEN - j - 1;
    var updated = checkpoints[@intFromEnum(Prefixes.LowerHalfWord)] orelse F.zero();
    updated = updated.add(F.fromU128(@as(u128, 1) << @intCast(x_shift)).mul(r_x));
    updated = updated.add(F.fromU128(@as(u128, 1) << @intCast(y_shift)).mul(r_y));
    return updated;
}
// ============================================================================
// DivByZero Prefix Implementation
// ============================================================================
fn divByZeroPrefixMle(
    comptime F: type,
    checkpoints: *const PrefixCheckpoints(F),
    r_x: ?F,
    c: u32,
    b: *LookupBits(128),
    _: usize,
) F {
    const uninterleaved = b.uninterleave();
    // If low-order bits of divisor are not 0s or low-order bits of quotient are not
    // 1s, short-circuit and return 0.
    // Use right_len (quotient length) which correctly handles odd b.len
    if (uninterleaved.left != 0 or uninterleaved.right != (@as(u64, 1) << @intCast(uninterleaved.right_len)) - 1) {
        return F.zero();
    }
    var result = checkpoints[@intFromEnum(Prefixes.DivByZero)] orelse F.one();
    if (r_x) |rx| {
        const y = F.fromU64(@as(u64, c));
        result = result.mul(F.one().sub(rx)).mul(y);
    } else {
        const x = F.fromU64(@as(u64, c));
        const y = F.fromU64(@as(u64, b.popMsb()));
        result = result.mul(F.one().sub(x)).mul(y);
    }
    return result;
}
fn divByZeroUpdateCheckpoint(
    comptime F: type,
    checkpoints: *const PrefixCheckpoints(F),
    r_x: F,
    r_y: F,
    _: usize,
    _: usize,
) PrefixCheckpoint(F) {
    // checkpoint *= (1 - r_x) * r_y
    var updated = checkpoints[@intFromEnum(Prefixes.DivByZero)] orelse F.one();
    updated = updated.mul(F.one().sub(r_x)).mul(r_y);
    return updated;
}
// ============================================================================
// PositiveRemainderEqualsDivisor Prefix Implementation
// ============================================================================
fn positiveRemainderEqualsDivisorPrefixMle(
    comptime F: type,
    checkpoints: *const PrefixCheckpoints(F),
    r_x: ?F,
    c: u32,
    b: *LookupBits(128),
    j: usize,
) F {
    if (j == 0) {
        const divisor_sign = F.fromU64(@as(u64, b.popMsb()));
        const uninterleaved = b.uninterleave();
        if (uninterleaved.left != uninterleaved.right) {
            return F.zero();
        } else {
            // c is the sign bit of the remainder
            return F.one().sub(F.fromU64(@as(u64, c))).mul(F.one().sub(divisor_sign));
        }
    }
    if (j == 1) {
        const uninterleaved = b.uninterleave();
        if (uninterleaved.left != uninterleaved.right) {
            return F.zero();
        } else {
            // r_x is the sign bit of the remainder, c is sign bit of divisor
            return F.one().sub(r_x.?).mul(F.one().sub(F.fromU64(@as(u64, c))));
        }
    }
    const positive_remainder_equals_divisor = checkpoints[@intFromEnum(Prefixes.PositiveRemainderEqualsDivisor)].?;
    if (r_x) |rx| {
        const uninterleaved = b.uninterleave();
        if (uninterleaved.left != uninterleaved.right) {
            return F.zero();
        }
        const y = F.fromU64(@as(u64, c));
        return positive_remainder_equals_divisor.mul(rx.mul(y).add(F.one().sub(rx).mul(F.one().sub(y))));
    } else {
        const y = F.fromU64(@as(u64, b.popMsb()));
        const uninterleaved = b.uninterleave();
        if (uninterleaved.left != uninterleaved.right) {
            return F.zero();
        }
        const x = F.fromU64(@as(u64, c));
        return positive_remainder_equals_divisor.mul(x.mul(y).add(F.one().sub(x).mul(F.one().sub(y))));
    }
}
fn positiveRemainderEqualsDivisorUpdateCheckpoint(
    comptime F: type,
    checkpoints: *const PrefixCheckpoints(F),
    r_x: F,
    r_y: F,
    j: usize,
    _: usize,
) PrefixCheckpoint(F) {
    if (j == 1) {
        return F.one().sub(r_x).mul(F.one().sub(r_y));
    }
    var updated = checkpoints[@intFromEnum(Prefixes.PositiveRemainderEqualsDivisor)].?;
    updated = updated.mul(r_x.mul(r_y).add(F.one().sub(r_x).mul(F.one().sub(r_y))));
    return updated;
}
// ============================================================================
// PositiveRemainderLessThanDivisor Prefix Implementation
// ============================================================================
fn positiveRemainderLessThanDivisorPrefixMle(
    comptime F: type,
    checkpoints: *const PrefixCheckpoints(F),
    r_x: ?F,
    c: u32,
    b: *LookupBits(128),
    j: usize,
) F {
    if (j == 0) {
        const divisor_sign = F.fromU64(@as(u64, b.popMsb()));
        const uninterleaved = b.uninterleave();
        if (uninterleaved.left >= uninterleaved.right) {
            return F.zero();
        } else {
            return F.one().sub(F.fromU64(@as(u64, c))).mul(F.one().sub(divisor_sign));
        }
    }
    if (j == 1) {
        const uninterleaved = b.uninterleave();
        if (uninterleaved.left >= uninterleaved.right) {
            return F.zero();
        } else {
            return F.one().sub(r_x.?).mul(F.one().sub(F.fromU64(@as(u64, c))));
        }
    }
    var lt = checkpoints[@intFromEnum(Prefixes.PositiveRemainderLessThanDivisor)].?;
    var eq = checkpoints[@intFromEnum(Prefixes.PositiveRemainderEqualsDivisor)].?;
    if (j == 2) {
        const c_f = F.fromU64(@as(u64, c));
        const y_msb = F.fromU64(@as(u64, b.popMsb()));
        const uninterleaved = b.uninterleave();
        lt = lt.mul(F.one().sub(c_f).mul(y_msb));
        if (uninterleaved.left < uninterleaved.right) {
            eq = eq.mul(c_f.mul(y_msb).add(F.one().sub(c_f).mul(F.one().sub(y_msb))));
            lt = lt.add(eq);
        }
        return lt;
    }
    if (j == 3) {
        const rx = r_x.?;
        const c_f = F.fromU64(@as(u64, c));
        const uninterleaved = b.uninterleave();
        lt = lt.mul(F.one().sub(rx).mul(c_f));
        if (uninterleaved.left < uninterleaved.right) {
            eq = eq.mul(rx.mul(c_f).add(F.one().sub(rx).mul(F.one().sub(c_f))));
            lt = lt.add(eq);
        }
        return lt;
    }
    if (r_x) |rx| {
        const c_f = F.fromU64(@as(u64, c));
        lt = lt.add(eq.mul(F.one().sub(rx)).mul(c_f));
        const uninterleaved = b.uninterleave();
        if (uninterleaved.left < uninterleaved.right) {
            eq = eq.mul(rx.mul(c_f).add(F.one().sub(rx).mul(F.one().sub(c_f))));
            lt = lt.add(eq);
        }
    } else {
        const c_f = F.fromU64(@as(u64, c));
        const y_msb = F.fromU64(@as(u64, b.popMsb()));
        lt = lt.add(eq.mul(F.one().sub(c_f)).mul(y_msb));
        const uninterleaved = b.uninterleave();
        if (uninterleaved.left < uninterleaved.right) {
            eq = eq.mul(c_f.mul(y_msb).add(F.one().sub(c_f).mul(F.one().sub(y_msb))));
            lt = lt.add(eq);
        }
    }
    return lt;
}
fn positiveRemainderLessThanDivisorUpdateCheckpoint(
    comptime F: type,
    checkpoints: *const PrefixCheckpoints(F),
    r_x: F,
    r_y: F,
    j: usize,
    _: usize,
) PrefixCheckpoint(F) {
    if (j == 1) {
        return F.one().sub(r_x).mul(F.one().sub(r_y));
    }
    const lt_checkpoint = checkpoints[@intFromEnum(Prefixes.PositiveRemainderLessThanDivisor)].?;
    const eq_checkpoint = checkpoints[@intFromEnum(Prefixes.PositiveRemainderEqualsDivisor)].?;
    if (j == 3) {
        return lt_checkpoint.mul(F.one().sub(r_x)).mul(r_y);
    }
    return lt_checkpoint.add(eq_checkpoint.mul(F.one().sub(r_x)).mul(r_y));
}
// ============================================================================
// NegativeDivisorZeroRemainder Prefix Implementation
// ============================================================================
fn negativeDivisorZeroRemainderPrefixMle(
    comptime F: type,
    checkpoints: *const PrefixCheckpoints(F),
    r_x: ?F,
    c: u32,
    b: *LookupBits(128),
    j: usize,
) F {
    if (j == 0) {
        const divisor_sign = F.fromU64(@as(u64, b.popMsb()));
        const uninterleaved = b.uninterleave();
        if (uninterleaved.left != 0) {
            return F.zero();
        } else {
            return F.one().sub(F.fromU64(@as(u64, c))).mul(divisor_sign);
        }
    }
    if (j == 1) {
        const uninterleaved = b.uninterleave();
        if (uninterleaved.left != 0) {
            return F.zero();
        } else {
            return F.one().sub(r_x.?).mul(F.fromU64(@as(u64, c)));
        }
    }
    const negative_divisor_zero_remainder = checkpoints[@intFromEnum(Prefixes.NegativeDivisorZeroRemainder)].?;
    if (r_x) |rx| {
        const uninterleaved = b.uninterleave();
        if (uninterleaved.left != 0) {
            return F.zero();
        }
        return negative_divisor_zero_remainder.mul(F.one().sub(rx));
    } else {
        _ = b.popMsb();
        const uninterleaved = b.uninterleave();
        if (uninterleaved.left != 0) {
            return F.zero();
        }
        return negative_divisor_zero_remainder.mul(F.one().sub(F.fromU64(@as(u64, c))));
    }
}
fn negativeDivisorZeroRemainderUpdateCheckpoint(
    comptime F: type,
    checkpoints: *const PrefixCheckpoints(F),
    r_x: F,
    r_y: F,
    j: usize,
    _: usize,
) PrefixCheckpoint(F) {
    if (j == 1) {
        return F.one().sub(r_x).mul(r_y);
    }
    var updated = checkpoints[@intFromEnum(Prefixes.NegativeDivisorZeroRemainder)].?;
    updated = updated.mul(F.one().sub(r_x));
    return updated;
}
// ============================================================================
// NegativeDivisorEqualsRemainder Prefix Implementation
// ============================================================================
fn negativeDivisorEqualsRemainderPrefixMle(
    comptime F: type,
    checkpoints: *const PrefixCheckpoints(F),
    r_x: ?F,
    c: u32,
    b: *LookupBits(128),
    j: usize,
) F {
    if (j == 0) {
        const divisor_sign = F.fromU64(@as(u64, b.popMsb()));
        const uninterleaved = b.uninterleave();
        if (uninterleaved.left != uninterleaved.right) {
            return F.zero();
        } else {
            return F.fromU64(@as(u64, c)).mul(divisor_sign);
        }
    }
    if (j == 1) {
        const uninterleaved = b.uninterleave();
        if (uninterleaved.left != uninterleaved.right) {
            return F.zero();
        } else {
            return r_x.?.mul(F.fromU64(@as(u64, c)));
        }
    }
    const negative_divisor_equals_remainder = checkpoints[@intFromEnum(Prefixes.NegativeDivisorEqualsRemainder)].?;
    if (r_x) |rx| {
        const uninterleaved = b.uninterleave();
        if (uninterleaved.left != uninterleaved.right) {
            return F.zero();
        }
        const y = F.fromU64(@as(u64, c));
        return negative_divisor_equals_remainder.mul(rx.mul(y).add(F.one().sub(rx).mul(F.one().sub(y))));
    } else {
        const y_msb = F.fromU64(@as(u64, b.popMsb()));
        const uninterleaved = b.uninterleave();
        if (uninterleaved.left != uninterleaved.right) {
            return F.zero();
        }
        const x = F.fromU64(@as(u64, c));
        return negative_divisor_equals_remainder.mul(x.mul(y_msb).add(F.one().sub(x).mul(F.one().sub(y_msb))));
    }
}
fn negativeDivisorEqualsRemainderUpdateCheckpoint(
    comptime F: type,
    checkpoints: *const PrefixCheckpoints(F),
    r_x: F,
    r_y: F,
    j: usize,
    _: usize,
) PrefixCheckpoint(F) {
    if (j == 1) {
        return r_x.mul(r_y);
    }
    var updated = checkpoints[@intFromEnum(Prefixes.NegativeDivisorEqualsRemainder)].?;
    updated = updated.mul(r_x.mul(r_y).add(F.one().sub(r_x).mul(F.one().sub(r_y))));
    return updated;
}
// ============================================================================
// NegativeDivisorGreaterThanRemainder Prefix Implementation
// ============================================================================
fn negativeDivisorGreaterThanRemainderPrefixMle(
    comptime F: type,
    checkpoints: *const PrefixCheckpoints(F),
    r_x: ?F,
    c: u32,
    b: *LookupBits(128),
    j: usize,
) F {
    if (j == 0) {
        const divisor_sign = F.fromU64(@as(u64, b.popMsb()));
        const uninterleaved = b.uninterleave();
        if (uninterleaved.left <= uninterleaved.right) {
            return F.zero();
        } else {
            return F.fromU64(@as(u64, c)).mul(divisor_sign);
        }
    }
    if (j == 1) {
        const uninterleaved = b.uninterleave();
        if (uninterleaved.left <= uninterleaved.right) {
            return F.zero();
        } else {
            return r_x.?.mul(F.fromU64(@as(u64, c)));
        }
    }
    var gt = checkpoints[@intFromEnum(Prefixes.NegativeDivisorGreaterThanRemainder)].?;
    var eq = checkpoints[@intFromEnum(Prefixes.NegativeDivisorEqualsRemainder)].?;
    if (j == 2) {
        const c_f = F.fromU64(@as(u64, c));
        const y_msb = F.fromU64(@as(u64, b.popMsb()));
        const uninterleaved = b.uninterleave();
        gt = gt.mul(c_f.mul(F.one().sub(y_msb)));
        if (uninterleaved.left > uninterleaved.right) {
            eq = eq.mul(c_f.mul(y_msb).add(F.one().sub(c_f).mul(F.one().sub(y_msb))));
            gt = gt.add(eq);
        }
        return gt;
    }
    if (j == 3) {
        const rx = r_x.?;
        const c_f = F.fromU64(@as(u64, c));
        const uninterleaved = b.uninterleave();
        gt = gt.mul(rx.mul(F.one().sub(c_f)));
        if (uninterleaved.left > uninterleaved.right) {
            eq = eq.mul(rx.mul(c_f).add(F.one().sub(rx).mul(F.one().sub(c_f))));
            gt = gt.add(eq);
        }
        return gt;
    }
    if (r_x) |rx| {
        const c_f = F.fromU64(@as(u64, c));
        gt = gt.add(eq.mul(rx).mul(F.one().sub(c_f)));
        const uninterleaved = b.uninterleave();
        if (uninterleaved.left > uninterleaved.right) {
            eq = eq.mul(rx.mul(c_f).add(F.one().sub(rx).mul(F.one().sub(c_f))));
            gt = gt.add(eq);
        }
    } else {
        const c_f = F.fromU64(@as(u64, c));
        const y_msb = F.fromU64(@as(u64, b.popMsb()));
        gt = gt.add(eq.mul(c_f).mul(F.one().sub(y_msb)));
        const uninterleaved = b.uninterleave();
        if (uninterleaved.left > uninterleaved.right) {
            eq = eq.mul(c_f.mul(y_msb).add(F.one().sub(c_f).mul(F.one().sub(y_msb))));
            gt = gt.add(eq);
        }
    }
    return gt;
}
fn negativeDivisorGreaterThanRemainderUpdateCheckpoint(
    comptime F: type,
    checkpoints: *const PrefixCheckpoints(F),
    r_x: F,
    r_y: F,
    j: usize,
    _: usize,
) PrefixCheckpoint(F) {
    if (j == 1) {
        return r_x.mul(r_y);
    }
    const gt_checkpoint = checkpoints[@intFromEnum(Prefixes.NegativeDivisorGreaterThanRemainder)].?;
    const eq_checkpoint = checkpoints[@intFromEnum(Prefixes.NegativeDivisorEqualsRemainder)].?;
    if (j == 3) {
        return gt_checkpoint.mul(r_x).mul(F.one().sub(r_y));
    }
    return gt_checkpoint.add(eq_checkpoint.mul(r_x).mul(F.one().sub(r_y)));
}
// ============================================================================
// Lsb Prefix Implementation
// ============================================================================
fn lsbPrefixMle(
    comptime F: type,
    _: *const PrefixCheckpoints(F),
    _: ?F,
    c: u32,
    b: *LookupBits(128),
    j: usize,
) F {
    const suffix_len = safeSuffixLen(j, b.len) orelse return F.one();
    if (j == 2 * XLEN - 1) {
        // In the log(K)th round, c corresponds to the LSB
        return F.fromU64(@as(u64, c));
    } else if (suffix_len == 0) {
        // In the (log(K)-1)th round, the LSB of b is the LSB
        return F.fromU64(@as(u64, @truncate(b.value)) & 1);
    } else {
        return F.one();
    }
}
fn lsbUpdateCheckpoint(
    comptime F: type,
    _: *const PrefixCheckpoints(F),
    _: F,
    r_y: F,
    j: usize,
    _: usize,
) PrefixCheckpoint(F) {
    if (j == 2 * XLEN - 1) {
        return r_y;
    } else {
        return F.one();
    }
}
// ============================================================================
// Pow2 Prefix Implementation
// ============================================================================
fn pow2PrefixMle(
    comptime F: type,
    checkpoints: *const PrefixCheckpoints(F),
    r_x: ?F,
    c: u32,
    b: *LookupBits(128),
    j: usize,
) F {
    const suffix_len = safeSuffixLen(j, b.len) orelse return F.one();
    if (suffix_len != 0) {
        return F.one();
    }
    const log_xlen = @ctz(@as(usize, XLEN)); // log2(64) = 6
    // Shift amount is the last XLEN bits of b
    if (b.len >= log_xlen) {
        const shift_amount = @as(u6, @intCast(b.value & (XLEN - 1)));
        return F.fromU64(@as(u64, 1) << shift_amount);
    }
    const shift_amount = @as(u6, @intCast(b.value & (XLEN - 1)));
    var result = F.fromU64(@as(u64, 1) << shift_amount);
    var num_bits = b.len;
    var shift: u64 = @as(u64, 1) << @as(u6, @intCast(@as(u64, 1) << @intCast(num_bits)));
    result = result.mul(F.fromU64(1 + (shift - 1) * c));
    // Shift amount is [c, b]
    if (b.len == log_xlen - 1) {
        return result;
    }
    // Shift amount is [(r, r_x), c, b]
    num_bits += 1;
    shift = @as(u64, 1) << @as(u6, @intCast(@as(u64, 1) << @intCast(num_bits)));
    if (r_x) |rx| {
        result = result.mul(F.one().add(F.fromU64(shift - 1).mul(rx)));
    }
    result = result.mul(checkpoints[@intFromEnum(Prefixes.Pow2)] orelse F.one());
    return result;
}
fn pow2UpdateCheckpoint(
    comptime F: type,
    checkpoints: *const PrefixCheckpoints(F),
    r_x: F,
    r_y: F,
    j: usize,
    suffix_len: usize,
) PrefixCheckpoint(F) {
    if (suffix_len != 0) {
        return F.one();
    }
    const log_xlen = @ctz(@as(usize, XLEN)); // log2(64) = 6
    // r_y is the highest bit of the shift amount
    if (j == 2 * XLEN - log_xlen) {
        const shift: u64 = @as(u64, 1) << @intCast(XLEN / 2);
        return F.one().add(F.fromU64(shift - 1).mul(r_y));
    }
    // r_x and r_y are bits in the shift amount
    if (2 * XLEN - j < log_xlen) {
        var checkpoint = checkpoints[@intFromEnum(Prefixes.Pow2)].?;
        const shift1: u64 = @as(u64, 1) << @as(u6, @intCast(@as(u64, 1) << @intCast(2 * XLEN - j)));
        checkpoint = checkpoint.mul(F.one().add(F.fromU64(shift1 - 1).mul(r_x)));
        const shift2: u64 = @as(u64, 1) << @as(u6, @intCast(@as(u64, 1) << @intCast(2 * XLEN - j - 1)));
        checkpoint = checkpoint.mul(F.one().add(F.fromU64(shift2 - 1).mul(r_y)));
        return checkpoint;
    }
    return F.one();
}
// ============================================================================
// Pow2W Prefix Implementation
// ============================================================================
fn pow2WPrefixMle(
    comptime F: type,
    checkpoints: *const PrefixCheckpoints(F),
    r_x: ?F,
    c: u32,
    b: *LookupBits(128),
    j: usize,
) F {
    const suffix_len = safeSuffixLen(j, b.len) orelse return F.one();
    if (suffix_len != 0) {
        return F.one();
    }
    // Shift amount is the last 5 bits of b (for modulo 32)
    if (b.len >= 5) {
        const shift_amount = @as(u5, @intCast(b.value & 0b11111));
        return F.fromU64(@as(u64, 1) << shift_amount);
    }
    const shift_amount = @as(u5, @intCast(b.value & 0b11111));
    var result = F.fromU64(@as(u64, 1) << shift_amount);
    var num_bits = b.len;
    var shift: u64 = @as(u64, 1) << @as(u6, @intCast(@as(u64, 1) << @intCast(num_bits)));
    result = result.mul(F.fromU64(1 + (shift - 1) * c));
    // Shift amount is [c, b]
    if (b.len == 4) { // 5 - 1
        return result;
    }
    // Shift amount is [(r, r_x), c, b]
    num_bits += 1;
    shift = @as(u64, 1) << @as(u6, @intCast(@as(u64, 1) << @intCast(num_bits)));
    if (r_x) |rx| {
        result = result.mul(F.one().add(F.fromU64(shift - 1).mul(rx)));
    }
    result = result.mul(checkpoints[@intFromEnum(Prefixes.Pow2W)] orelse F.one());
    return result;
}
fn pow2WUpdateCheckpoint(
    comptime F: type,
    checkpoints: *const PrefixCheckpoints(F),
    r_x: F,
    r_y: F,
    j: usize,
    suffix_len: usize,
) PrefixCheckpoint(F) {
    if (suffix_len != 0) {
        return F.one();
    }
    // r_y is the highest bit of the shift amount
    if (j == 2 * XLEN - 5) {
        const shift: u64 = 1 << 16; // 2^(32/2)
        return F.one().add(F.fromU64(shift - 1).mul(r_y));
    }
    // r_x and r_y are bits in the shift amount
    if (2 * XLEN - j < 5) {
        var checkpoint = checkpoints[@intFromEnum(Prefixes.Pow2W)].?;
        const shift1: u64 = @as(u64, 1) << @as(u6, @intCast(@as(u64, 1) << @intCast(2 * XLEN - j)));
        checkpoint = checkpoint.mul(F.one().add(F.fromU64(shift1 - 1).mul(r_x)));
        const shift2: u64 = @as(u64, 1) << @as(u6, @intCast(@as(u64, 1) << @intCast(2 * XLEN - j - 1)));
        checkpoint = checkpoint.mul(F.one().add(F.fromU64(shift2 - 1).mul(r_y)));
        return checkpoint;
    }
    return F.one();
}
// ============================================================================
// RightShift Prefix Implementation
// ============================================================================
fn rightShiftPrefixMle(
    comptime F: type,
    checkpoints: *const PrefixCheckpoints(F),
    r_x: ?F,
    c: u32,
    b: *LookupBits(128),
    _: usize,
) F {
    var result = checkpoints[@intFromEnum(Prefixes.RightShift)] orelse F.zero();
    if (r_x) |rx| {
        result = result.mul(F.fromU64(1 + @as(u64, c)));
        result = result.add(rx.mul(F.fromU64(@as(u64, c))));
    } else {
        const y_msb = b.popMsb();
        result = result.mul(F.fromU64(1 + @as(u64, y_msb)));
        result = result.add(F.fromU64(@as(u64, @as(u8, @intCast(c)) * y_msb)));
    }
    const uninterleaved = b.uninterleave();
    const half_len = b.len / 2;
    // Create proper LookupBits for length-aware leading_ones/trailing_zeros
    const x_bits = LookupBits(128).new(uninterleaved.left, half_len);
    const y_bits = LookupBits(128).new(uninterleaved.right, half_len);
    const leading = y_bits.leadingOnes();
    const trailing = y_bits.trailingZeros();
    if (leading >= 64) {
        return result;
    }
    result = result.mul(F.fromU64(@as(u64, 1) << @intCast(leading)));
    const x_val: u64 = @truncate(x_bits.value);
    const shifted_x: u64 = if (trailing >= 64) 0 else x_val >> @intCast(trailing);
    result = result.add(F.fromU64(shifted_x));
    return result;
}
fn rightShiftUpdateCheckpoint(
    comptime F: type,
    checkpoints: *const PrefixCheckpoints(F),
    r_x: F,
    r_y: F,
    _: usize,
    _: usize,
) PrefixCheckpoint(F) {
    var updated = checkpoints[@intFromEnum(Prefixes.RightShift)] orelse F.zero();
    updated = updated.mul(F.one().add(r_y));
    updated = updated.add(r_x.mul(r_y));
    return updated;
}
// ============================================================================
// SignExtension Prefix Implementation
// ============================================================================
fn signExtensionPrefixMle(
    comptime F: type,
    checkpoints: *const PrefixCheckpoints(F),
    r_x: ?F,
    c: u32,
    b: *LookupBits(128),
    j: usize,
) F {
    if (j == 0) {
        const sign_bit = F.fromU64(@as(u64, c));
        if (sign_bit.eql(F.zero())) {
            return F.zero();
        }
        _ = b.popMsb();
        const y_val = b.uninterleave().right;
        var result = F.zero();
        var index: usize = 1;
        const y_len = b.len / 2;
        var i: usize = 0;
        while (i < y_len) : (i += 1) {
            const y_i = (y_val >> @intCast(y_len - 1 - i)) & 1;
            if (y_i == 0) {
                result = result.add(fieldPow2(F, index));
            }
            index += 1;
        }
        return result.mul(sign_bit);
    }
    if (j == 1) {
        const sign_bit = r_x.?;
        const y_val = b.uninterleave().right;
        var result = F.zero();
        var index: usize = 1;
        const y_len = b.len / 2;
        var i: usize = 0;
        while (i < y_len) : (i += 1) {
            const y_i = (y_val >> @intCast(y_len - 1 - i)) & 1;
            if (y_i == 0) {
                result = result.add(fieldPow2(F, index));
            }
            index += 1;
        }
        return result.mul(sign_bit);
    }
    const sign_bit = checkpoints[@intFromEnum(Prefixes.LeftOperandMsb)].?;
    var result = checkpoints[@intFromEnum(Prefixes.SignExtension)] orelse F.zero();
    if (r_x != null) {
        result = result.add(fieldPow2(F, j / 2).mul(F.one().sub(F.fromU64(@as(u64, c)))));
    } else {
        const y_msb = b.popMsb();
        if (y_msb == 0) {
            result = result.add(fieldPow2(F, j / 2));
        }
    }
    const y_val = b.uninterleave().right;
    var index = j / 2;
    const y_len = b.len / 2;
    var i: usize = 0;
    while (i < y_len) : (i += 1) {
        index += 1;
        const y_i = (y_val >> @intCast(y_len - 1 - i)) & 1;
        if (y_i == 0) {
            result = result.add(fieldPow2(F, index));
        }
    }
    return result.mul(sign_bit);
}
fn signExtensionUpdateCheckpoint(
    comptime F: type,
    checkpoints: *const PrefixCheckpoints(F),
    _: F,
    r_y: F,
    j: usize,
    _: usize,
) PrefixCheckpoint(F) {
    if (j == 1) {
        return null;
    }
    var updated = checkpoints[@intFromEnum(Prefixes.SignExtension)] orelse F.zero();
    updated = updated.add(fieldPow2(F, j / 2).mul(F.one().sub(r_y)));
    if (j == 2 * XLEN - 1) {
        updated = updated.mul(checkpoints[@intFromEnum(Prefixes.LeftOperandMsb)].?);
    }
    return updated;
}
// ============================================================================
// LeftShift Prefix Implementation
// ============================================================================
fn leftShiftPrefixMle(
    comptime F: type,
    checkpoints: *const PrefixCheckpoints(F),
    r_x: ?F,
    c: u32,
    b: *LookupBits(128),
    j: usize,
) F {
    var result = checkpoints[@intFromEnum(Prefixes.LeftShift)] orelse F.zero();
    var prod_one_plus_y = checkpoints[@intFromEnum(Prefixes.LeftShiftHelper)] orelse F.one();
    if (r_x) |rx| {
        result = result.add(rx.mul(F.one().sub(F.fromU64(@as(u64, c)))).mul(prod_one_plus_y).mul(F.fromU64(@as(u64, 1) << @intCast(XLEN - 1 - j / 2))));
        prod_one_plus_y = prod_one_plus_y.mul(F.fromU64(1 + @as(u64, c)));
    } else {
        const y_msb = b.popMsb();
        result = result.add(F.fromU64(@as(u64, @as(u8, @intCast(c)) * (1 - y_msb))).mul(prod_one_plus_y).mul(F.fromU64(@as(u64, 1) << @intCast(XLEN - 1 - j / 2))));
        prod_one_plus_y = prod_one_plus_y.mul(F.fromU64(1 + @as(u64, y_msb)));
    }
    const uninterleaved = b.uninterleave();
    const y_len = b.len / 2;
    const x = uninterleaved.left & ~uninterleaved.right;
    // Use length-aware leading_ones via LookupBits
    const y_bits_ls = LookupBits(128).new(uninterleaved.right, y_len);
    const y_leading_ones: usize = y_bits_ls.leadingOnes();
    // Handle potential underflow: if y_len > y_leading_ones + (XLEN - 1 - j/2), shift would be "negative"
    const total = y_leading_ones + XLEN - 1 - j / 2;
    const shifted: u64 = if (total >= y_len and total - y_len < 64)
        (x << @intCast(total - y_len))
    else
        0;
    result = result.add(F.fromU64(shifted).mul(prod_one_plus_y));
    return result;
}
fn leftShiftUpdateCheckpoint(
    comptime F: type,
    checkpoints: *const PrefixCheckpoints(F),
    r_x: F,
    r_y: F,
    j: usize,
    _: usize,
) PrefixCheckpoint(F) {
    var updated = checkpoints[@intFromEnum(Prefixes.LeftShift)] orelse F.zero();
    const prod_one_plus_y = checkpoints[@intFromEnum(Prefixes.LeftShiftHelper)] orelse F.one();
    updated = updated.add(r_x.mul(F.one().sub(r_y)).mul(prod_one_plus_y).mul(F.fromU64(@as(u64, 1) << @intCast(XLEN - 1 - j / 2))));
    return updated;
}
// ============================================================================
// LeftShiftHelper Prefix Implementation
// ============================================================================
fn leftShiftHelperPrefixMle(
    comptime F: type,
    checkpoints: *const PrefixCheckpoints(F),
    r_x: ?F,
    c: u32,
    b: *LookupBits(128),
    _: usize,
) F {
    var result = checkpoints[@intFromEnum(Prefixes.LeftShiftHelper)] orelse F.one();
    if (r_x != null) {
        result = result.mul(F.fromU64(1 + @as(u64, c)));
    } else {
        const y_msb = b.popMsb();
        result = result.mul(F.fromU64(1 + @as(u64, y_msb)));
    }
    // Use length-aware leading_ones via LookupBits
    const lsh_half_len = b.len / 2;
    const lsh_y_bits = LookupBits(128).new(b.uninterleave().right, lsh_half_len);
    const lsh_leading = lsh_y_bits.leadingOnes();
    if (lsh_leading >= 64) return result;
    result = result.mul(F.fromU64(@as(u64, 1) << @intCast(lsh_leading)));
    return result;
}
fn leftShiftHelperUpdateCheckpoint(
    comptime F: type,
    checkpoints: *const PrefixCheckpoints(F),
    _: F,
    r_y: F,
    _: usize,
    _: usize,
) PrefixCheckpoint(F) {
    var updated = checkpoints[@intFromEnum(Prefixes.LeftShiftHelper)] orelse F.one();
    updated = updated.mul(F.one().add(r_y));
    return updated;
}
// ============================================================================
// TwoLsb Prefix Implementation
// ============================================================================
fn twoLsbPrefixMle(
    comptime F: type,
    _: *const PrefixCheckpoints(F),
    r_x: ?F,
    c: u32,
    b: *LookupBits(128),
    j: usize,
) F {
    const suffix_len = safeSuffixLen(j, b.len) orelse return F.one();
    if (j == 2 * XLEN - 1) {
        // In the log(K)th round, c corresponds to bit 0 and r_x to bit 1
        return F.one().sub(F.fromU64(@as(u64, c))).mul(F.one().sub(r_x.?));
    } else if (j == 2 * XLEN - 2) {
        // In the (log(K)-1)th round, c corresponds to bit 1
        const bit0 = @as(u32, @truncate(b.value)) & 1;
        const bit1 = c;
        return F.one().sub(F.fromU64(@as(u64, bit0))).mul(F.one().sub(F.fromU64(@as(u64, bit1))));
    } else if (suffix_len == 0) {
        // In the (log(K)-2)th round, the two LSBs of b are the two LSBs
        if ((@as(u32, @truncate(b.value)) & 0b11) == 0) {
            return F.one();
        } else {
            return F.zero();
        }
    } else {
        return F.one();
    }
}
fn twoLsbUpdateCheckpoint(
    comptime F: type,
    checkpoints: *const PrefixCheckpoints(F),
    r_x: F,
    r_y: F,
    j: usize,
    _: usize,
) PrefixCheckpoint(F) {
    if (j == 2 * XLEN - 1) {
        return F.one().sub(r_x).mul(F.one().sub(r_y));
    } else {
        return checkpoints[@intFromEnum(Prefixes.TwoLsb)];
    }
}
// ============================================================================
// SignExtensionUpperHalf Prefix Implementation
// ============================================================================
fn signExtensionUpperHalfPrefixMle(
    comptime F: type,
    checkpoints: *const PrefixCheckpoints(F),
    r_x: ?F,
    c: u32,
    _: *LookupBits(128),
    j: usize,
) F {
    const suffix_len = LOG_K - j - 1; // b.len is effectively 0 when we don't use b
    const half_word_size = XLEN / 2;
    if (suffix_len >= half_word_size) {
        return F.one();
    }
    if (j == XLEN + half_word_size) {
        return F.fromU128(((@as(u128, 1) << half_word_size) - 1) << half_word_size).mul(F.fromU64(@as(u64, c)));
    } else if (j == XLEN + half_word_size + 1) {
        return F.fromU128(((@as(u128, 1) << half_word_size) - 1) << half_word_size).mul(r_x.?);
    } else if (j > XLEN + half_word_size + 1) {
        return checkpoints[@intFromEnum(Prefixes.SignExtensionUpperHalf)] orelse F.zero();
    } else {
        return F.zero(); // This case should never happen
    }
}
fn signExtensionUpperHalfUpdateCheckpoint(
    comptime F: type,
    checkpoints: *const PrefixCheckpoints(F),
    r_x: F,
    _: F,
    j: usize,
    _: usize,
) PrefixCheckpoint(F) {
    const half_word_size = XLEN / 2;
    if (j == XLEN + half_word_size + 1) {
        return F.fromU128(((@as(u128, 1) << half_word_size) - 1) << half_word_size).mul(r_x);
    } else {
        return checkpoints[@intFromEnum(Prefixes.SignExtensionUpperHalf)];
    }
}
// ============================================================================
// ChangeDivisor Prefix Implementation
// ============================================================================
fn changeDivisorPrefixMle(
    comptime F: type,
    checkpoints: *const PrefixCheckpoints(F),
    r_x: ?F,
    c: u32,
    b: *LookupBits(128),
    j: usize,
) F {
    var result = checkpoints[@intFromEnum(Prefixes.ChangeDivisor)] orelse F.fromU64(2).sub(F.fromU128(@as(u128, 1) << XLEN));
    if (j == 0) {
        const x_msb = b.popMsb();
        if (x_msb == 0) {
            return F.zero();
        }
        const uninterleaved = b.uninterleave();
        if (uninterleaved.left != 0 or uninterleaved.right != (@as(u64, 1) << @intCast(uninterleaved.right_len)) - 1) {
            return F.zero();
        }
        return result.mul(F.fromU64(@as(u64, c)));
    } else if (r_x) |rx| {
        const uninterleaved = b.uninterleave();
        if (uninterleaved.left != 0 or uninterleaved.right != (@as(u64, 1) << @intCast(uninterleaved.right_len)) - 1 or c == 0) {
            return F.zero();
        }
        if (j == 1) {
            return result.mul(rx).mul(F.fromU64(@as(u64, c)));
        } else {
            return result.mul(F.one().sub(rx)).mul(F.fromU64(@as(u64, c)));
        }
    } else {
        const uninterleaved = b.uninterleave();
        if (b.len > 0 and (uninterleaved.left != 0 or uninterleaved.right != (@as(u64, 1) << @intCast(uninterleaved.right_len)) - 1)) {
            return F.zero();
        }
        return result.mul(F.one().sub(F.fromU64(@as(u64, c))));
    }
}
fn changeDivisorUpdateCheckpoint(
    comptime F: type,
    checkpoints: *const PrefixCheckpoints(F),
    r_x: F,
    r_y: F,
    j: usize,
    _: usize,
) PrefixCheckpoint(F) {
    const base = checkpoints[@intFromEnum(Prefixes.ChangeDivisor)] orelse F.fromU64(2).sub(F.fromU128(@as(u128, 1) << XLEN));
    if (j == 1) {
        return base.mul(r_x).mul(r_y);
    } else {
        return base.mul(F.one().sub(r_x)).mul(r_y);
    }
}
// ============================================================================
// ChangeDivisorW Prefix Implementation
// ============================================================================
fn changeDivisorWPrefixMle(
    comptime F: type,
    checkpoints: *const PrefixCheckpoints(F),
    r_x: ?F,
    c: u32,
    b: *LookupBits(128),
    j: usize,
) F {
    if (j < XLEN) {
        return F.zero();
    }
    var result = if (j == XLEN or j == XLEN + 1)
        F.fromU64(2).sub(F.fromU128(@as(u128, 1) << XLEN))
    else
        checkpoints[@intFromEnum(Prefixes.ChangeDivisorW)].?;
    if (j == XLEN) {
        const x_msb = b.popMsb();
        if (x_msb == 0) {
            return F.zero();
        }
        const uninterleaved = b.uninterleave();
        if (uninterleaved.left != 0 or uninterleaved.right != (@as(u64, 1) << @intCast(uninterleaved.right_len)) - 1) {
            return F.zero();
        }
        return result.mul(F.fromU64(@as(u64, c)));
    } else if (r_x) |rx| {
        if (j > XLEN) {
            const uninterleaved = b.uninterleave();
            if (uninterleaved.left != 0 or uninterleaved.right != (@as(u64, 1) << @intCast(uninterleaved.right_len)) - 1 or c == 0) {
                return F.zero();
            }
            if (j == XLEN + 1) {
                return result.mul(rx).mul(F.fromU64(@as(u64, c)));
            } else {
                return result.mul(F.one().sub(rx)).mul(F.fromU64(@as(u64, c)));
            }
        }
    } else if (j > XLEN) {
        const uninterleaved = b.uninterleave();
        if (b.len > 0 and (uninterleaved.left != 0 or uninterleaved.right != (@as(u64, 1) << @intCast(uninterleaved.right_len)) - 1)) {
            return F.zero();
        }
        return result.mul(F.one().sub(F.fromU64(@as(u64, c))));
    }
    return result;
}
fn changeDivisorWUpdateCheckpoint(
    comptime F: type,
    checkpoints: *const PrefixCheckpoints(F),
    r_x: F,
    r_y: F,
    j: usize,
    _: usize,
) PrefixCheckpoint(F) {
    if (j < XLEN) {
        return F.zero();
    }
    if (j == XLEN + 1) {
        return F.fromU64(2).sub(F.fromU128(@as(u128, 1) << XLEN)).mul(r_x).mul(r_y);
    } else {
        return checkpoints[@intFromEnum(Prefixes.ChangeDivisorW)].?.mul(F.one().sub(r_x).mul(r_y));
    }
}
// ============================================================================
// RightOperand Prefix Implementation
// ============================================================================
fn rightOperandPrefixMle(
    comptime F: type,
    checkpoints: *const PrefixCheckpoints(F),
    _: ?F,
    c: u32,
    b: *LookupBits(128),
    j: usize,
) F {
    var result = checkpoints[@intFromEnum(Prefixes.RightOperand)] orelse F.zero();
    if (j % 2 == 1) {
        // c is of the right operand
        const shift = XLEN - 1 - j / 2;
        result = result.add(F.fromU128(@as(u128, c) << @intCast(shift)));
    }
    const y = b.uninterleave().right;
    const suffix_len = safeSuffixLen(j, b.len) orelse return result;
    result = result.add(F.fromU128(@as(u128, y) << @intCast(suffix_len / 2)));
    return result;
}
fn rightOperandUpdateCheckpoint(
    comptime F: type,
    checkpoints: *const PrefixCheckpoints(F),
    _: F,
    r_y: F,
    j: usize,
    _: usize,
) PrefixCheckpoint(F) {
    const shift = XLEN - 1 - j / 2;
    var updated = checkpoints[@intFromEnum(Prefixes.RightOperand)] orelse F.zero();
    updated = updated.add(F.fromU64(@as(u64, 1) << @intCast(shift)).mul(r_y));
    return updated;
}
// ============================================================================
// RightOperandW Prefix Implementation
// ============================================================================
fn rightOperandWPrefixMle(
    comptime F: type,
    checkpoints: *const PrefixCheckpoints(F),
    _: ?F,
    c: u32,
    b: *LookupBits(128),
    j: usize,
) F {
    var result = checkpoints[@intFromEnum(Prefixes.RightOperandW)] orelse F.zero();
    if (j % 2 == 1 and j > XLEN) {
        const shift = XLEN - 1 - j / 2;
        result = result.add(F.fromU128(@as(u128, c) << @intCast(shift)));
    }
    const suffix_len = safeSuffixLen(j, b.len) orelse return result;
    if (suffix_len < XLEN) {
        const y = b.uninterleave().right;
        result = result.add(F.fromU128(@as(u128, y) << @intCast(suffix_len / 2)));
    }
    return result;
}
fn rightOperandWUpdateCheckpoint(
    comptime F: type,
    checkpoints: *const PrefixCheckpoints(F),
    _: F,
    r_y: F,
    j: usize,
    _: usize,
) PrefixCheckpoint(F) {
    if (j > XLEN) {
        const shift = XLEN - 1 - j / 2;
        var updated = checkpoints[@intFromEnum(Prefixes.RightOperandW)] orelse F.zero();
        updated = updated.add(F.fromU64(@as(u64, 1) << @intCast(shift)).mul(r_y));
        return updated;
    } else {
        return checkpoints[@intFromEnum(Prefixes.RightOperandW)];
    }
}
// ============================================================================
// SignExtensionRightOperand Prefix Implementation
// ============================================================================
fn signExtensionRightOperandPrefixMle(
    comptime F: type,
    checkpoints: *const PrefixCheckpoints(F),
    _: ?F,
    c: u32,
    b: *LookupBits(128),
    j: usize,
) F {
    const suffix_len = safeSuffixLen(j, b.len) orelse return F.one();
    if (suffix_len >= XLEN) {
        return F.one();
    }
    if (j == XLEN) {
        const sign_bit = b.popMsb();
        return F.fromU128((@as(u128, 1) << XLEN) - (@as(u128, 1) << (XLEN / 2))).mul(F.fromU64(@as(u64, sign_bit)));
    } else if (j == XLEN + 1) {
        return F.fromU128((@as(u128, 1) << XLEN) - (@as(u128, 1) << (XLEN / 2))).mul(F.fromU64(@as(u64, c)));
    } else if (j >= XLEN + 2) {
        return checkpoints[@intFromEnum(Prefixes.SignExtensionRightOperand)] orelse F.zero();
    } else {
        return F.zero();
    }
}
fn signExtensionRightOperandUpdateCheckpoint(
    comptime F: type,
    checkpoints: *const PrefixCheckpoints(F),
    _: F,
    r_y: F,
    j: usize,
    _: usize,
) PrefixCheckpoint(F) {
    if (j == XLEN + 1) {
        return F.fromU128((@as(u128, 1) << XLEN) - (@as(u128, 1) << (XLEN / 2))).mul(r_y);
    } else {
        return checkpoints[@intFromEnum(Prefixes.SignExtensionRightOperand)];
    }
}
// ============================================================================
// RightShiftW Prefix Implementation
// ============================================================================
fn rightShiftWPrefixMle(
    comptime F: type,
    checkpoints: *const PrefixCheckpoints(F),
    r_x: ?F,
    c: u32,
    b: *LookupBits(128),
    j: usize,
) F {
    if (j < XLEN) {
        return F.zero();
    }
    var result = checkpoints[@intFromEnum(Prefixes.RightShiftW)] orelse F.zero();
    if (r_x) |rx| {
        result = result.mul(F.fromU64(1 + @as(u64, c)));
        result = result.add(rx.mul(F.fromU64(@as(u64, c))));
    } else {
        const y_msb = b.popMsb();
        result = result.mul(F.fromU64(1 + @as(u64, y_msb)));
        result = result.add(F.fromU64(@as(u64, @as(u8, @intCast(c)) * y_msb)));
    }
    const uninterleaved = b.uninterleave();
    const half_len = b.len / 2;
    // Create proper LookupBits for length-aware leading_ones/trailing_zeros
    const x_bits_w = LookupBits(128).new(uninterleaved.left, half_len);
    const y_bits_w = LookupBits(128).new(uninterleaved.right, half_len);
    const leading_w = y_bits_w.leadingOnes();
    const trailing_w = y_bits_w.trailingZeros();
    // Match Jolt's `F::from_u32(1 << y.leading_ones())` semantics: u32 wrapping shift.
    // Rust release mode masks the shift count to (BITS - 1), so `1u32 << n` for any
    // `n` becomes `1u32 << (n & 31)`. For n in [0, 31] this is `1 << n`; for n = 32
    // (y is all-ones in its `len` bits) this wraps to `1 << 0 = 1`, NOT 0.
    const factor_u32: u32 = @as(u32, 1) << @intCast(leading_w & 31);
    result = result.mul(F.fromU64(@as(u64, factor_u32)));
    // Match Jolt's `F::from_u32(u32::from(x) >> y.trailing_zeros())` semantics: u32
    // wrapping shift. For trailing_zeros = 32 (y == 0), this wraps to `x >> 0 = x`.
    const x_val_u32: u32 = @truncate(x_bits_w.value);
    const shifted_x_u32: u32 = x_val_u32 >> @intCast(trailing_w & 31);
    result = result.add(F.fromU64(@as(u64, shifted_x_u32)));
    return result;
}
fn rightShiftWUpdateCheckpoint(
    comptime F: type,
    checkpoints: *const PrefixCheckpoints(F),
    r_x: F,
    r_y: F,
    j: usize,
    _: usize,
) PrefixCheckpoint(F) {
    if (j >= XLEN) {
        var updated = checkpoints[@intFromEnum(Prefixes.RightShiftW)] orelse F.zero();
        updated = updated.mul(F.one().add(r_y));
        updated = updated.add(r_x.mul(r_y));
        return updated;
    } else {
        return F.zero();
    }
}
// ============================================================================
// LeftShiftWHelper Prefix Implementation
// ============================================================================
fn leftShiftWHelperPrefixMle(
    comptime F: type,
    checkpoints: *const PrefixCheckpoints(F),
    r_x: ?F,
    c: u32,
    b: *LookupBits(128),
    j: usize,
) F {
    if (j < XLEN) {
        return F.one();
    }
    var result = checkpoints[@intFromEnum(Prefixes.LeftShiftWHelper)] orelse F.one();
    if (r_x != null) {
        result = result.mul(F.fromU64(1 + @as(u64, c)));
    } else {
        const y_msb = b.popMsb();
        result = result.mul(F.fromU64(1 + @as(u64, y_msb)));
    }
    // Use length-aware leading_ones via LookupBits
    const lswh_half_len = b.len / 2;
    const lswh_y_bits = LookupBits(128).new(b.uninterleave().right, lswh_half_len);
    const lswh_leading = lswh_y_bits.leadingOnes();
    // Match Jolt's `F::from_u32(1 << y.leading_ones())` semantics: u32 wrapping shift.
    // For leading_ones = 32 (y is all-ones), this wraps to `1 << 0 = 1`, NOT 0.
    const factor_u32: u32 = @as(u32, 1) << @intCast(lswh_leading & 31);
    result = result.mul(F.fromU64(@as(u64, factor_u32)));
    return result;
}
fn leftShiftWHelperUpdateCheckpoint(
    comptime F: type,
    checkpoints: *const PrefixCheckpoints(F),
    _: F,
    r_y: F,
    j: usize,
    _: usize,
) PrefixCheckpoint(F) {
    if (j >= XLEN) {
        var updated = checkpoints[@intFromEnum(Prefixes.LeftShiftWHelper)] orelse F.one();
        updated = updated.mul(F.one().add(r_y));
        return updated;
    } else {
        return F.one();
    }
}
// ============================================================================
// LeftShiftW Prefix Implementation
// ============================================================================
fn leftShiftWPrefixMle(
    comptime F: type,
    checkpoints: *const PrefixCheckpoints(F),
    r_x: ?F,
    c: u32,
    b: *LookupBits(128),
    j: usize,
) F {
    if (j < XLEN) {
        return F.zero();
    }
    var result = checkpoints[@intFromEnum(Prefixes.LeftShiftW)] orelse F.zero();
    var prod_one_plus_y = checkpoints[@intFromEnum(Prefixes.LeftShiftWHelper)] orelse F.one();
    const bit_index = XLEN - 1 - j / 2;
    if (r_x) |rx| {
        const shift_val: u64 = if (bit_index < 64) (@as(u64, 1) << @intCast(bit_index)) else 0;
        result = result.add(rx.mul(F.one().sub(F.fromU64(@as(u64, c)))).mul(prod_one_plus_y).mul(F.fromU64(shift_val)));
        prod_one_plus_y = prod_one_plus_y.mul(F.fromU64(1 + @as(u64, c)));
    } else {
        const y_msb = b.popMsb();
        const shift_val: u64 = if (bit_index < 64) (@as(u64, 1) << @intCast(bit_index)) else 0;
        result = result.add(F.fromU64(@as(u64, @as(u8, @intCast(c)) * (1 - y_msb))).mul(prod_one_plus_y).mul(F.fromU64(shift_val)));
        prod_one_plus_y = prod_one_plus_y.mul(F.fromU64(1 + @as(u64, y_msb)));
    }
    const uninterleaved = b.uninterleave();
    const y_len_w = b.len / 2;
    const x = uninterleaved.left & ~uninterleaved.right;
    // Use length-aware leading_ones via LookupBits
    const y_bits_lsw = LookupBits(128).new(uninterleaved.right, y_len_w);
    const y_leading_ones: usize = y_bits_lsw.leadingOnes();
    // Handle potential underflow: if y_len > y_leading_ones + bit_index, shift is "negative" (treat as large)
    const total = y_leading_ones + bit_index;
    const shifted: u64 = if (total >= y_len_w and total - y_len_w < 64)
        (x << @intCast(total - y_len_w))
    else
        0;
    result = result.add(F.fromU64(shifted).mul(prod_one_plus_y));
    return result;
}
fn leftShiftWUpdateCheckpoint(
    comptime F: type,
    checkpoints: *const PrefixCheckpoints(F),
    r_x: F,
    r_y: F,
    j: usize,
    _: usize,
) PrefixCheckpoint(F) {
    if (j >= XLEN) {
        var updated = checkpoints[@intFromEnum(Prefixes.LeftShiftW)] orelse F.zero();
        const prod_one_plus_y = checkpoints[@intFromEnum(Prefixes.LeftShiftWHelper)] orelse F.one();
        const bit_index = XLEN - 1 - j / 2;
        const shift_val: u64 = if (bit_index < 64) (@as(u64, 1) << @intCast(bit_index)) else 0;
        updated = updated.add(r_x.mul(F.one().sub(r_y)).mul(prod_one_plus_y).mul(F.fromU64(shift_val)));
        return updated;
    } else {
        return F.zero();
    }
}
// ============================================================================
// OverflowBitsZero Prefix Implementation
// ============================================================================
fn overflowBitsZeroPrefixMle(
    comptime F: type,
    checkpoints: *const PrefixCheckpoints(F),
    r_x: ?F,
    c: u32,
    b: *LookupBits(128),
    j: usize,
) F {
    // Compute suffix_len BEFORE any popMsb, matching Jolt's order
    const original_b_len = b.len;
    const suffix_len_opt = safeSuffixLen(j, original_b_len);

    if (j >= 128 - XLEN) {
        return checkpoints[@intFromEnum(Prefixes.OverflowBitsZero)] orelse F.one();
    }
    var result = checkpoints[@intFromEnum(Prefixes.OverflowBitsZero)] orelse F.one();
    if (r_x) |rx| {
        const y = F.fromU64(@as(u64, c));
        result = result.mul(F.one().sub(rx).mul(F.one().sub(y)));
    } else {
        const x = F.fromU64(@as(u64, c));
        const y = F.fromU64(@as(u64, b.popMsb()));
        result = result.mul(F.one().sub(x).mul(F.one().sub(y)));
    }
    const rest = b.value;
    const suffix_len = suffix_len_opt orelse return result;
    const shifted = rest << @intCast(suffix_len);
    const is_zero: u64 = if ((shifted >> XLEN) == 0) 1 else 0;
    result = result.mul(F.fromU64(is_zero));
    return result;
}
fn overflowBitsZeroUpdateCheckpoint(
    comptime F: type,
    checkpoints: *const PrefixCheckpoints(F),
    r_x: F,
    r_y: F,
    j: usize,
    _: usize,
) PrefixCheckpoint(F) {
    if (j >= 128 - XLEN) {
        return checkpoints[@intFromEnum(Prefixes.OverflowBitsZero)];
    }
    var updated = checkpoints[@intFromEnum(Prefixes.OverflowBitsZero)] orelse F.one();
    updated = updated.mul(F.one().sub(r_x).mul(F.one().sub(r_y)));
    return updated;
}
// ============================================================================
// XorRot Prefix Implementation (for 16, 24, 32, 63)
// ============================================================================
fn xorRotPrefixMle(
    comptime F: type,
    comptime rotation: u32,
    checkpoints: *const PrefixCheckpoints(F),
    r_x: ?F,
    c: u32,
    b: *LookupBits(128),
    j: usize,
) F {
    // Compute suffix_len BEFORE any popMsb, matching Jolt's order
    const original_b_len = b.len;
    const suffix_len_opt = safeSuffixLen(j, original_b_len);

    const prefix_idx = switch (rotation) {
        16 => Prefixes.XorRot16,
        24 => Prefixes.XorRot24,
        32 => Prefixes.XorRot32,
        63 => Prefixes.XorRot63,
        else => unreachable,
    };
    var result = checkpoints[@intFromEnum(prefix_idx)] orelse F.zero();
    if (r_x) |rx| {
        const y = F.fromU64(@as(u64, c));
        const xor_bit = F.one().sub(rx).mul(y).add(rx.mul(F.one().sub(y)));
        const original_pos = j / 2;
        const rotated_pos = (original_pos + rotation) % XLEN;
        const shift = XLEN - 1 - rotated_pos;
        result = result.add(F.fromU64(@as(u64, 1) << @intCast(shift)).mul(xor_bit));
    } else {
        const x = F.fromU64(@as(u64, c));
        const y_msb = F.fromU64(@as(u64, b.popMsb()));
        const xor_bit = F.one().sub(x).mul(y_msb).add(x.mul(F.one().sub(y_msb)));
        const original_pos = j / 2;
        const rotated_pos = (original_pos + rotation) % XLEN;
        const shift = XLEN - 1 - rotated_pos;
        result = result.add(F.fromU64(@as(u64, 1) << @intCast(shift)).mul(xor_bit));
    }
    // Remaining x and y bits using suffix_len computed with ORIGINAL b.len
    const uninterleaved = b.uninterleave();
    const xor_result = uninterleaved.left ^ uninterleaved.right;
    const suffix_len = suffix_len_opt orelse return result;
    const shift_i: i32 = @as(i32, @intCast(suffix_len / 2)) - @as(i32, rotation);
    const shift: u6 = if (shift_i >= 0)
        @intCast(shift_i)
    else
        @intCast(@as(i32, XLEN) + shift_i);
    // Rotate left
    const rotated = std.math.rotl(u64, xor_result, shift);
    result = result.add(F.fromU64(rotated));
    return result;
}
fn xorRotUpdateCheckpoint(
    comptime F: type,
    comptime rotation: u32,
    checkpoints: *const PrefixCheckpoints(F),
    r_x: F,
    r_y: F,
    j: usize,
    _: usize,
) PrefixCheckpoint(F) {
    const prefix_idx = switch (rotation) {
        16 => Prefixes.XorRot16,
        24 => Prefixes.XorRot24,
        32 => Prefixes.XorRot32,
        63 => Prefixes.XorRot63,
        else => unreachable,
    };
    const original_pos = j / 2;
    const rotated_pos = (original_pos + rotation) % XLEN;
    const shift = XLEN - 1 - rotated_pos;
    var updated = checkpoints[@intFromEnum(prefix_idx)] orelse F.zero();
    updated = updated.add(F.fromU64(@as(u64, 1) << @intCast(shift)).mul(F.one().sub(r_x).mul(r_y).add(r_x.mul(F.one().sub(r_y)))));
    return updated;
}
// ============================================================================
// XorRotW Prefix Implementation (for 7, 8, 12, 16 - 32-bit word operations)
// ============================================================================
fn xorRotWPrefixMle(
    comptime F: type,
    comptime rotation: u32,
    checkpoints: *const PrefixCheckpoints(F),
    r_x: ?F,
    c: u32,
    b: *LookupBits(128),
    j: usize,
) F {
    // Compute suffix_len BEFORE any popMsb, matching Jolt's order
    const original_b_len = b.len;
    const suffix_len_opt = safeSuffixLen(j, original_b_len);

    if (j < XLEN) {
        return F.zero();
    }
    const prefix_idx = switch (rotation) {
        7 => Prefixes.XorRotW7,
        8 => Prefixes.XorRotW8,
        12 => Prefixes.XorRotW12,
        16 => Prefixes.XorRotW16,
        else => unreachable,
    };
    var result = checkpoints[@intFromEnum(prefix_idx)] orelse F.zero();
    if (r_x) |rx| {
        const y = F.fromU64(@as(u64, c));
        const xor_bit = F.one().sub(rx).mul(y).add(rx.mul(F.one().sub(y)));
        const position = (j - XLEN) / 2;
        var rotated_position = (position + rotation) % 32;
        rotated_position = 32 - 1 - rotated_position;
        result = result.add(F.fromU64(@as(u64, 1) << @intCast(rotated_position)).mul(xor_bit));
    } else {
        const x = F.fromU64(@as(u64, c));
        const y_msb = F.fromU64(@as(u64, b.popMsb()));
        const xor_bit = F.one().sub(x).mul(y_msb).add(x.mul(F.one().sub(y_msb)));
        const position = (j - XLEN) / 2;
        var rotated_position = (position + rotation) % 32;
        rotated_position = 32 - 1 - rotated_position;
        result = result.add(F.fromU64(@as(u64, 1) << @intCast(rotated_position)).mul(xor_bit));
    }
    // Remaining x and y bits using suffix_len computed with ORIGINAL b.len
    const uninterleaved = b.uninterleave();
    const x_32 = @as(u32, @truncate(uninterleaved.left));
    const y_32 = @as(u32, @truncate(uninterleaved.right));
    const xor_result = x_32 ^ y_32;
    const suffix_len = suffix_len_opt orelse return result;
    const shift_i: i32 = @as(i32, @intCast(suffix_len / 2)) - @as(i32, rotation);
    const shift: u5 = if (shift_i >= 0)
        @intCast(shift_i)
    else
        @intCast(@as(i32, 32) + shift_i);
    const rotated = std.math.rotl(u32, xor_result, shift);
    result = result.add(F.fromU64(@as(u64, rotated)));
    return result;
}
fn xorRotWUpdateCheckpoint(
    comptime F: type,
    comptime rotation: u32,
    checkpoints: *const PrefixCheckpoints(F),
    r_x: F,
    r_y: F,
    j: usize,
    _: usize,
) PrefixCheckpoint(F) {
    if (j >= XLEN) {
        const prefix_idx = switch (rotation) {
            7 => Prefixes.XorRotW7,
            8 => Prefixes.XorRotW8,
            12 => Prefixes.XorRotW12,
            16 => Prefixes.XorRotW16,
            else => unreachable,
        };
        const original_pos = (j - XLEN) / 2;
        const rotated_pos = (original_pos + rotation) % 32;
        const shift = 32 - 1 - rotated_pos;
        var updated = checkpoints[@intFromEnum(prefix_idx)] orelse F.zero();
        updated = updated.add(F.fromU64(@as(u64, 1) << @intCast(shift)).mul(F.one().sub(r_x).mul(r_y).add(r_x.mul(F.one().sub(r_y)))));
        return updated;
    } else {
        return F.zero();
    }
}
// ============================================================================
// Rev8W Prefix Implementation
// ============================================================================
/// Reverse bytes in each 32-bit word of a u64.
/// Matches Jolt: `(v as u32).swap_bytes() as u64 + ((((v >> 32) as u32).swap_bytes()) as u64) << 32`
fn rev8w(v: u64) u64 {
    const lo: u32 = @truncate(v);
    const hi: u32 = @truncate(v >> 32);
    return @as(u64, @byteSwap(lo)) + (@as(u64, @byteSwap(hi)) << 32);
}
fn rev8wPrefixMle(
    comptime F: type,
    checkpoints: *const PrefixCheckpoints(F),
    r_x: ?F,
    c: u32,
    b: *LookupBits(128),
    j: usize,
) F {
    const suffix_len = safeSuffixLen(j, b.len) orelse return F.zero();
    // The prefix-suffix MLE is only defined on the 64 LSBs.
    if (suffix_len >= 64) {
        return F.zero();
    }
    var eval = checkpoints[@intFromEnum(Prefixes.Rev8W)] orelse F.zero();
    // Add c contribution
    const c_bit_index = suffix_len + b.len;
    if (c_bit_index < 64) {
        const shift = @ctz(rev8w(@as(u64, 1) << @intCast(c_bit_index)));
        eval = eval.add(F.fromU128(@as(u128, c) << @intCast(shift)));
    }
    // Add r_x contribution
    const r_x_bit_index = c_bit_index + 1;
    if (r_x_bit_index < 64) {
        if (r_x) |rx| {
            const rev_pow2 = rev8w(@as(u64, 1) << @intCast(r_x_bit_index));
            eval = eval.add(rx.mul(F.fromU64(rev_pow2)));
        }
    }
    // Add b contribution
    const b_contribution = rev8w(@as(u64, @truncate(b.value)) << @intCast(suffix_len));
    eval = eval.add(F.fromU64(b_contribution));
    return eval;
}
fn rev8wUpdateCheckpoint(
    comptime F: type,
    checkpoints: *const PrefixCheckpoints(F),
    r_x: F,
    r_y: F,
    j: usize,
    _: usize,
) PrefixCheckpoint(F) {
    var res = checkpoints[@intFromEnum(Prefixes.Rev8W)] orelse F.zero();
    const r_y_bit_index = 2 * XLEN - 1 - j;
    if (r_y_bit_index < 64) {
        const rev_pow2 = rev8w(@as(u64, 1) << @intCast(r_y_bit_index));
        res = res.add(r_y.mul(F.fromU64(rev_pow2)));
    }
    const r_x_bit_index = r_y_bit_index + 1;
    if (r_x_bit_index < 64) {
        const rev_pow2 = rev8w(@as(u64, 1) << @intCast(r_x_bit_index));
        res = res.add(r_x.mul(F.fromU64(rev_pow2)));
    }
    return res;
}
// ============================================================================
// Tests
// ============================================================================
test "LookupBits uninterleave" {
    // Test Jolt-compatible uninterleave: left operand at ODD positions, right operand at EVEN positions
    // Value 0b1011 = 11 has: bit0=1, bit1=1, bit2=0, bit3=1
    // right operand (even positions 0,2): bits 1,0 -> 0b01 = 1
    // left operand (odd positions 1,3): bits 1,1 -> 0b11 = 3
    var bits = LookupBits(128).new(0b1011, 4);
    const result = bits.uninterleave();
    try std.testing.expectEqual(@as(u64, 3), result.left);
    try std.testing.expectEqual(@as(u64, 1), result.right);
}
test "EqPrefix basic" {
    const F = @import("zolt_arith").field.BN254Scalar;
    var checkpoints: PrefixCheckpoints(F) = .{null} ** Prefixes.COUNT;
    // Test eq(0, 0) should be 1, eq(0, 1) should be 0
    var b = LookupBits(128).new(0, 0);
    const eq_00 = eqPrefixMle(F, &checkpoints, null, 0, &b, 0);
    try std.testing.expect(eq_00.eql(F.one())); // (0 eq 0) with empty b
    b = LookupBits(128).new(1, 1);
    const eq_01 = eqPrefixMle(F, &checkpoints, null, 0, &b, 0);
    try std.testing.expect(eq_01.eql(F.zero())); // (0 eq 1) = 0
}
