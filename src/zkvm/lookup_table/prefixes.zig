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
const Allocator = std.mem.Allocator;

/// LOG_K = 128 for RV64 (2*XLEN for interleaved operands)
pub const LOG_K: usize = 128;
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

        /// Uninterleave bits: split interleaved (x_0, y_0, x_1, y_1, ...) into (x, y)
        /// where even indices are x bits and odd indices are y bits
        pub fn uninterleave(self: *const Self) struct { left: u64, right: u64 } {
            var left: u64 = 0;
            var right: u64 = 0;
            const half_len = self.len / 2;

            var i: usize = 0;
            while (i < half_len) : (i += 1) {
                const bit_pos = 2 * i;
                // x bits are at even positions (0, 2, 4, ...)
                const x_bit = (self.value >> @intCast(bit_pos)) & 1;
                // y bits are at odd positions (1, 3, 5, ...)
                const y_bit = (self.value >> @intCast(bit_pos + 1)) & 1;

                left |= @as(u64, @truncate(x_bit)) << @intCast(i);
                right |= @as(u64, @truncate(y_bit)) << @intCast(i);
            }

            return .{ .left = left, .right = right };
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
    const suffix_len = LOG_K - j - b.len - 1;

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

    // Add in low-order bits from b
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
    updated = updated.add(F.fromU128(@as(u128, 1) << @intCast(x_shift)).mul(r_x));
    updated = updated.add(F.fromU128(@as(u128, 1) << @intCast(y_shift)).mul(r_y));
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
    const suffix_len = LOG_K - j - b.len - 1;

    // Only active during upper XLEN variables
    if (j >= XLEN) {
        return F.zero();
    }

    var result = checkpoints[@intFromEnum(Prefixes.UpperWord)] orelse F.zero();

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

    // Add in low-order bits from b
    result = result.add(F.fromU128(b.value << @intCast(suffix_len)));

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
        return null;
    }

    const x_shift = 2 * XLEN - j;
    const y_shift = 2 * XLEN - j - 1;
    var updated = checkpoints[@intFromEnum(Prefixes.UpperWord)] orelse F.zero();
    updated = updated.add(F.fromU128(@as(u128, 1) << @intCast(x_shift)).mul(r_x));
    updated = updated.add(F.fromU128(@as(u128, 1) << @intCast(y_shift)).mul(r_y));
    return updated;
}

// ============================================================================
// And Prefix Implementation
// ============================================================================

fn andPrefixMle(
    comptime F: type,
    checkpoints: *const PrefixCheckpoints(F),
    r_x: ?F,
    c: u32,
    b: *LookupBits(128),
    j: usize,
) F {
    const suffix_len = LOG_K - j - b.len - 1;

    var result = checkpoints[@intFromEnum(Prefixes.And)] orelse F.zero();

    if (r_x) |rx| {
        const y = F.fromU64(@as(u64, c));
        const x_shift = XLEN - (j / 2);
        // AND(r_x, c) = r_x * c
        const and_contrib = F.fromU128(@as(u128, 1) << @intCast(x_shift)).mul(rx.mul(y));
        result = result.add(and_contrib);
    } else {
        const x = F.fromU64(@as(u64, c));
        const y_msb = b.popMsb();
        const x_shift = XLEN - (j / 2) - 1;
        // AND(c, y_msb) = c * y_msb
        const and_contrib = F.fromU128(@as(u128, 1) << @intCast(x_shift)).mul(x.mul(F.fromU64(@as(u64, y_msb))));
        result = result.add(and_contrib);
    }

    // Process remaining bits in b
    const uninterleaved = b.uninterleave();
    const and_suffix = uninterleaved.left & uninterleaved.right;
    result = result.add(F.fromU128(@as(u128, and_suffix) << @intCast(suffix_len / 2)));

    return result;
}

fn andUpdateCheckpoint(
    comptime F: type,
    checkpoints: *const PrefixCheckpoints(F),
    r_x: F,
    r_y: F,
    j: usize,
    _: usize,
) PrefixCheckpoint(F) {
    const x_shift = XLEN - (j / 2);
    var updated = checkpoints[@intFromEnum(Prefixes.And)] orelse F.zero();
    updated = updated.add(F.fromU128(@as(u128, 1) << @intCast(x_shift)).mul(r_x.mul(r_y)));
    return updated;
}

// ============================================================================
// Or Prefix Implementation
// ============================================================================

fn orPrefixMle(
    comptime F: type,
    checkpoints: *const PrefixCheckpoints(F),
    r_x: ?F,
    c: u32,
    b: *LookupBits(128),
    j: usize,
) F {
    const suffix_len = LOG_K - j - b.len - 1;

    var result = checkpoints[@intFromEnum(Prefixes.Or)] orelse F.zero();

    if (r_x) |rx| {
        const y = F.fromU64(@as(u64, c));
        const x_shift = XLEN - (j / 2);
        // OR(r_x, c) = r_x + c - r_x * c
        const or_contrib = rx.add(y).sub(rx.mul(y));
        result = result.add(F.fromU128(@as(u128, 1) << @intCast(x_shift)).mul(or_contrib));
    } else {
        const x = F.fromU64(@as(u64, c));
        const y_msb_val = F.fromU64(@as(u64, b.popMsb()));
        const x_shift = XLEN - (j / 2) - 1;
        // OR(c, y_msb) = c + y_msb - c * y_msb
        const or_contrib = x.add(y_msb_val).sub(x.mul(y_msb_val));
        result = result.add(F.fromU128(@as(u128, 1) << @intCast(x_shift)).mul(or_contrib));
    }

    // Process remaining bits
    const uninterleaved = b.uninterleave();
    const or_suffix = uninterleaved.left | uninterleaved.right;
    result = result.add(F.fromU128(@as(u128, or_suffix) << @intCast(suffix_len / 2)));

    return result;
}

fn orUpdateCheckpoint(
    comptime F: type,
    checkpoints: *const PrefixCheckpoints(F),
    r_x: F,
    r_y: F,
    j: usize,
    _: usize,
) PrefixCheckpoint(F) {
    const x_shift = XLEN - (j / 2);
    var updated = checkpoints[@intFromEnum(Prefixes.Or)] orelse F.zero();
    // OR(r_x, r_y) = r_x + r_y - r_x * r_y
    updated = updated.add(F.fromU128(@as(u128, 1) << @intCast(x_shift)).mul(r_x.add(r_y).sub(r_x.mul(r_y))));
    return updated;
}

// ============================================================================
// Xor Prefix Implementation
// ============================================================================

fn xorPrefixMle(
    comptime F: type,
    checkpoints: *const PrefixCheckpoints(F),
    r_x: ?F,
    c: u32,
    b: *LookupBits(128),
    j: usize,
) F {
    const suffix_len = LOG_K - j - b.len - 1;

    var result = checkpoints[@intFromEnum(Prefixes.Xor)] orelse F.zero();

    if (r_x) |rx| {
        const y = F.fromU64(@as(u64, c));
        const x_shift = XLEN - (j / 2);
        // XOR(r_x, c) = r_x + c - 2 * r_x * c
        const two = F.fromU64(2);
        const xor_contrib = rx.add(y).sub(two.mul(rx.mul(y)));
        result = result.add(F.fromU128(@as(u128, 1) << @intCast(x_shift)).mul(xor_contrib));
    } else {
        const x = F.fromU64(@as(u64, c));
        const y_msb_val = F.fromU64(@as(u64, b.popMsb()));
        const x_shift = XLEN - (j / 2) - 1;
        // XOR(c, y_msb) = c + y_msb - 2 * c * y_msb
        const two = F.fromU64(2);
        const xor_contrib = x.add(y_msb_val).sub(two.mul(x.mul(y_msb_val)));
        result = result.add(F.fromU128(@as(u128, 1) << @intCast(x_shift)).mul(xor_contrib));
    }

    // Process remaining bits
    const uninterleaved = b.uninterleave();
    const xor_suffix = uninterleaved.left ^ uninterleaved.right;
    result = result.add(F.fromU128(@as(u128, xor_suffix) << @intCast(suffix_len / 2)));

    return result;
}

fn xorUpdateCheckpoint(
    comptime F: type,
    checkpoints: *const PrefixCheckpoints(F),
    r_x: F,
    r_y: F,
    j: usize,
    _: usize,
) PrefixCheckpoint(F) {
    const x_shift = XLEN - (j / 2);
    var updated = checkpoints[@intFromEnum(Prefixes.Xor)] orelse F.zero();
    // XOR(r_x, r_y) = r_x + r_y - 2 * r_x * r_y
    const two = F.fromU64(2);
    updated = updated.add(F.fromU128(@as(u128, 1) << @intCast(x_shift)).mul(r_x.add(r_y).sub(two.mul(r_x.mul(r_y)))));
    return updated;
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
    const eq_checkpoint = checkpoints[@intFromEnum(Prefixes.Eq)] orelse F.one();
    var result = checkpoints[@intFromEnum(Prefixes.LessThan)] orelse F.zero();

    if (r_x) |rx| {
        const y = F.fromU64(@as(u64, c));
        // LT contribution: eq_prev * (1 - r_x) * y
        result = result.add(eq_checkpoint.mul(F.one().sub(rx)).mul(y));
    } else {
        const x = F.fromU64(@as(u64, c));
        const y_msb_val = F.fromU64(@as(u64, b.popMsb()));
        // LT contribution: eq_prev * (1 - x) * y_msb
        result = result.add(eq_checkpoint.mul(F.one().sub(x)).mul(y_msb_val));
    }

    return result;
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
    var result = checkpoints[@intFromEnum(Prefixes.LeftOperandIsZero)] orelse F.one();

    if (r_x) |rx| {
        // On odd rounds (when r_x is present), c is the y-value, not x
        // We need to multiply by (1 - r_x) for the left operand
        result = result.mul(F.one().sub(rx));
        // c is not used on odd rounds
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
        // c is implicitly discarded by not using it
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
    // Only active in round 0
    if (j > 0) {
        return checkpoints[@intFromEnum(Prefixes.LeftOperandMsb)] orelse F.zero();
    }

    if (r_x) |rx| {
        // On odd rounds, r_x is the x-value we want
        return rx;
    } else {
        // On even rounds, c is the x-value
        return F.fromU64(@as(u64, c));
    }
}

fn leftMsbUpdateCheckpoint(
    comptime F: type,
    checkpoints: *const PrefixCheckpoints(F),
    r_x: F,
    _: F,
    j: usize,
    _: usize,
) PrefixCheckpoint(F) {
    if (j > 0) {
        return checkpoints[@intFromEnum(Prefixes.LeftOperandMsb)];
    }
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

fn andnPrefixMle(
    comptime F: type,
    checkpoints: *const PrefixCheckpoints(F),
    r_x: ?F,
    c: u32,
    b: *LookupBits(128),
    j: usize,
) F {
    const suffix_len = LOG_K - j - b.len - 1;
    var result = checkpoints[@intFromEnum(Prefixes.Andn)] orelse F.zero();

    // ANDN high-order variables: x_i * (1 - y_i)
    if (r_x) |rx| {
        const y = F.fromU64(@as(u64, c));
        const shift = XLEN - 1 - j / 2;
        result = result.add(F.fromU64(@as(u64, 1) << @intCast(shift)).mul(rx).mul(F.one().sub(y)));
    } else {
        const y_msb = b.popMsb();
        const shift = XLEN - 1 - j / 2;
        // c * (1 - y_msb)
        result = result.add(F.fromU64(@as(u64, c) * (1 - @as(u64, y_msb))).mul(F.fromU64(@as(u64, 1) << @intCast(shift))));
    }

    // ANDN remaining x and y bits
    const uninterleaved = b.uninterleave();
    result = result.add(F.fromU128(@as(u128, uninterleaved.left & ~uninterleaved.right) << @intCast(suffix_len / 2)));

    return result;
}

fn andnUpdateCheckpoint(
    comptime F: type,
    checkpoints: *const PrefixCheckpoints(F),
    r_x: F,
    r_y: F,
    j: usize,
    _: usize,
) PrefixCheckpoint(F) {
    const shift = XLEN - 1 - j / 2;
    // checkpoint += 2^shift * r_x * (1 - r_y)
    var updated = checkpoints[@intFromEnum(Prefixes.Andn)] orelse F.zero();
    updated = updated.add(F.fromU64(@as(u64, 1) << @intCast(shift)).mul(r_x).mul(F.one().sub(r_y)));
    return updated;
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
    const suffix_len = LOG_K - j - b.len - 1;
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

    // Add in low-order bits from b
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
    const quotient_len = b.len / 2;
    if (uninterleaved.left != 0 or uninterleaved.right != (@as(u64, 1) << @intCast(quotient_len)) - 1) {
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
    const suffix_len = LOG_K - j - b.len - 1;
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
    const suffix_len = LOG_K - j - b.len - 1;
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
    const suffix_len = LOG_K - j - b.len - 1;
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
    const x_u32 = @as(u32, @truncate(uninterleaved.left));
    const y_u32 = @as(u32, @truncate(uninterleaved.right));
    result = result.mul(F.fromU64(@as(u64, 1) << @intCast(@clz(~y_u32))));
    result = result.add(F.fromU64(@as(u64, x_u32 >> @intCast(@ctz(y_u32)))));

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
            result = result.add(F.fromU64((1 - y_i) << @intCast(index)));
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
            result = result.add(F.fromU64((1 - y_i) << @intCast(index)));
            index += 1;
        }
        return result.mul(sign_bit);
    }

    const sign_bit = checkpoints[@intFromEnum(Prefixes.LeftOperandMsb)].?;
    var result = checkpoints[@intFromEnum(Prefixes.SignExtension)] orelse F.zero();

    if (r_x != null) {
        result = result.add(F.fromU64(@as(u64, 1) << @intCast(j / 2)).mul(F.one().sub(F.fromU64(@as(u64, c)))));
    } else {
        const y_msb = b.popMsb();
        if (y_msb == 0) {
            result = result.add(F.fromU64(@as(u64, 1) << @intCast(j / 2)));
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
            result = result.add(F.fromU64(@as(u64, 1) << @intCast(index)));
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
    updated = updated.add(F.fromU64(@as(u64, 1) << @intCast(j / 2)).mul(F.one().sub(r_y)));
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
    const x = uninterleaved.left & ~uninterleaved.right;
    const y_leading_ones: usize = @clz(~@as(u32, @truncate(uninterleaved.right)));
    const y_len = b.len / 2;
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

    const y = @as(u32, @truncate(b.uninterleave().right));
    result = result.mul(F.fromU64(@as(u64, 1) << @intCast(@clz(~y))));

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
    const suffix_len = LOG_K - j - b.len - 1;
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
        const y_len = b.len / 2;
        if (uninterleaved.left != 0 or uninterleaved.right != (@as(u64, 1) << @intCast(y_len)) - 1) {
            return F.zero();
        }
        return result.mul(F.fromU64(@as(u64, c)));
    } else if (r_x) |rx| {
        const uninterleaved = b.uninterleave();
        const y_len = b.len / 2;
        if (uninterleaved.left != 0 or uninterleaved.right != (@as(u64, 1) << @intCast(y_len)) - 1 or c == 0) {
            return F.zero();
        }
        if (j == 1) {
            return result.mul(rx).mul(F.fromU64(@as(u64, c)));
        } else {
            return result.mul(F.one().sub(rx)).mul(F.fromU64(@as(u64, c)));
        }
    } else {
        const uninterleaved = b.uninterleave();
        const y_len = b.len / 2;
        if (b.len > 0 and (uninterleaved.left != 0 or uninterleaved.right != (@as(u64, 1) << @intCast(y_len)) - 1)) {
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
        const y_len = b.len / 2;
        if (uninterleaved.left != 0 or uninterleaved.right != (@as(u64, 1) << @intCast(y_len)) - 1) {
            return F.zero();
        }
        return result.mul(F.fromU64(@as(u64, c)));
    } else if (r_x) |rx| {
        if (j > XLEN) {
            const uninterleaved = b.uninterleave();
            const y_len = b.len / 2;
            if (uninterleaved.left != 0 or uninterleaved.right != (@as(u64, 1) << @intCast(y_len)) - 1 or c == 0) {
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
        const y_len = b.len / 2;
        if (b.len > 0 and (uninterleaved.left != 0 or uninterleaved.right != (@as(u64, 1) << @intCast(y_len)) - 1)) {
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
    const suffix_len = LOG_K - j - b.len - 1;
    var result = checkpoints[@intFromEnum(Prefixes.RightOperand)] orelse F.zero();

    if (j % 2 == 1) {
        // c is of the right operand
        const shift = XLEN - 1 - j / 2;
        result = result.add(F.fromU128(@as(u128, c) << @intCast(shift)));
    }

    const y = b.uninterleave().right;
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
    const suffix_len = LOG_K - j - b.len - 1;
    var result = checkpoints[@intFromEnum(Prefixes.RightOperandW)] orelse F.zero();

    if (j % 2 == 1 and j > XLEN) {
        const shift = XLEN - 1 - j / 2;
        result = result.add(F.fromU128(@as(u128, c) << @intCast(shift)));
    }

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
    const suffix_len = LOG_K - j - b.len - 1;

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
    const x_u32 = @as(u32, @truncate(uninterleaved.left));
    const y_u32 = @as(u32, @truncate(uninterleaved.right));
    result = result.mul(F.fromU64(@as(u64, 1) << @intCast(@clz(~y_u32))));
    result = result.add(F.fromU64(@as(u64, x_u32 >> @intCast(@ctz(y_u32)))));

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

    const y = @as(u32, @truncate(b.uninterleave().right));
    result = result.mul(F.fromU64(@as(u64, 1) << @intCast(@clz(~y))));

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
    const x = uninterleaved.left & ~uninterleaved.right;
    const y_leading_ones: usize = @clz(~@as(u32, @truncate(uninterleaved.right)));
    const y_len = b.len / 2;
    // Handle potential underflow: if y_len > y_leading_ones + bit_index, shift is "negative" (treat as large)
    const total = y_leading_ones + bit_index;
    const shifted: u64 = if (total >= y_len and total - y_len < 64)
        (x << @intCast(total - y_len))
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
    const suffix_len = LOG_K - j - b.len - 1;
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
    const suffix_len = LOG_K - j - b.len - 1;
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

    // Remaining x and y bits
    const uninterleaved = b.uninterleave();
    const xor_result = uninterleaved.left ^ uninterleaved.right;

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
    const suffix_len = LOG_K - j - b.len - 1;
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

    // Remaining x and y bits
    const uninterleaved = b.uninterleave();
    const x_32 = @as(u32, @truncate(uninterleaved.left));
    const y_32 = @as(u32, @truncate(uninterleaved.right));
    const xor_result = x_32 ^ y_32;

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

/// Byte-reverse a 32-bit word
fn rev8w(x: u64) u64 {
    const masked = x & 0xFFFFFFFF;
    return @as(u64, @byteSwap(@as(u32, @truncate(masked))));
}

fn rev8wPrefixMle(
    comptime F: type,
    checkpoints: *const PrefixCheckpoints(F),
    r_x: ?F,
    c: u32,
    b: *LookupBits(128),
    j: usize,
) F {
    const suffix_len = LOG_K - j - b.len - 1;
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
    // Test interleaving: (x0, y0, x1, y1) = (1, 0, 1, 1) = 0b1011 = 11
    var bits = LookupBits(128).new(0b1011, 4);
    const result = bits.uninterleave();
    // x = x0 + 2*x1 = 1 + 2*1 = 3
    // y = y0 + 2*y1 = 0 + 2*1 = 2
    try std.testing.expectEqual(@as(u64, 3), result.left);
    try std.testing.expectEqual(@as(u64, 2), result.right);
}

test "EqPrefix basic" {
    const F = @import("../../field/mod.zig").BN254Scalar;
    var checkpoints: PrefixCheckpoints(F) = .{null} ** Prefixes.COUNT;

    // Test eq(0, 0) should be 1, eq(0, 1) should be 0
    var b = LookupBits(128).new(0, 0);
    const eq_00 = eqPrefixMle(F, &checkpoints, null, 0, &b, 0);
    try std.testing.expect(eq_00.eql(F.one())); // (0 eq 0) with empty b

    b = LookupBits(128).new(1, 1);
    const eq_01 = eqPrefixMle(F, &checkpoints, null, 0, &b, 0);
    try std.testing.expect(eq_01.eql(F.zero())); // (0 eq 1) = 0
}
