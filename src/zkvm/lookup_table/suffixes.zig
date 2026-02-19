//! Lookup Table Suffixes for Jolt-Compatible Sumcheck
//!
//! This module implements all suffix types from Jolt's lookup table system.
//! Each suffix provides a `suffix_mle()` function that evaluates the multilinear
//! extension of the suffix polynomial on the given suffix bits.
//!
//! The suffix MLE is used in prefix-suffix decomposition:
//!   Q[prefix] = Σ_{j: prefix_bits[j] == prefix} u_eval[j] * suffix_mle(suffix_bits[j])
//!
//! Reference: jolt-core/src/zkvm/lookup_table/suffixes/*.rs

const std = @import("std");
const prefixes_mod = @import("prefixes.zig");
const LookupBits = prefixes_mod.LookupBits;
const XLEN: usize = prefixes_mod.XLEN;

/// All suffix types used by Jolt's lookup tables
pub const Suffixes = enum(u8) {
    One = 0,
    And = 1,
    NotAnd = 2,
    Xor = 3,
    Or = 4,
    RightOperand = 5,
    RightOperandW = 6,
    ChangeDivisor = 7,
    ChangeDivisorW = 8,
    UpperWord = 9,
    LowerWord = 10,
    LowerHalfWord = 11,
    LessThan = 12,
    GreaterThan = 13,
    Eq = 14,
    LeftOperandIsZero = 15,
    RightOperandIsZero = 16,
    Lsb = 17,
    DivByZero = 18,
    Pow2 = 19,
    Pow2W = 20,
    Rev8W = 21,
    RightShiftPadding = 22,
    RightShift = 23,
    RightShiftHelper = 24,
    SignExtension = 25,
    LeftShift = 26,
    TwoLsb = 27,
    SignExtensionUpperHalf = 28,
    SignExtensionRightOperand = 29,
    RightShiftW = 30,
    RightShiftWHelper = 31,
    LeftShiftWHelper = 32,
    LeftShiftW = 33,
    OverflowBitsZero = 34,
    XorRot16 = 35,
    XorRot24 = 36,
    XorRot32 = 37,
    XorRot63 = 38,
    XorRotW16 = 39,
    XorRotW12 = 40,
    XorRotW8 = 41,
    XorRotW7 = 42,

    pub const COUNT: usize = 43;
};

/// Evaluate the suffix MLE on the given suffix bits
///
/// Args:
///   suffix: The suffix type to evaluate
///   b: The suffix bits (unbound variables)
///
/// Returns: The MLE evaluation as u64 (converted to field element by caller)
pub fn suffixMle(suffix: Suffixes, b: LookupBits(128)) u64 {
    return switch (suffix) {
        .One => oneSuffixMle(b),
        .And => andSuffixMle(b),
        .NotAnd => notAndSuffixMle(b),
        .Xor => xorSuffixMle(b),
        .Or => orSuffixMle(b),
        .RightOperand => rightOperandSuffixMle(b),
        .RightOperandW => rightOperandWSuffixMle(b),
        .UpperWord => upperWordSuffixMle(b),
        .LowerWord => lowerWordSuffixMle(b),
        .LowerHalfWord => lowerHalfWordSuffixMle(b),
        .LessThan => lessThanSuffixMle(b),
        .GreaterThan => greaterThanSuffixMle(b),
        .Eq => eqSuffixMle(b),
        .LeftOperandIsZero => leftOperandIsZeroSuffixMle(b),
        .RightOperandIsZero => rightOperandIsZeroSuffixMle(b),
        .Lsb => lsbSuffixMle(b),
        .DivByZero => divByZeroSuffixMle(b),
        .Pow2 => pow2SuffixMle(b),
        .Pow2W => pow2WSuffixMle(b),
        .SignExtension => signExtensionSuffixMle(b),
        .LeftShift => leftShiftSuffixMle(b),
        .RightShift => rightShiftSuffixMle(b),
        .TwoLsb => twoLsbSuffixMle(b),
        .ChangeDivisor => changeDivisorSuffixMle(b),
        .ChangeDivisorW => changeDivisorWSuffixMle(b),
        .Rev8W => rev8WSuffixMle(b),
        .RightShiftPadding => rightShiftPaddingSuffixMle(b),
        .RightShiftHelper => rightShiftHelperSuffixMle(b),
        .SignExtensionUpperHalf => signExtensionUpperHalfSuffixMle(b),
        .SignExtensionRightOperand => signExtensionRightOperandSuffixMle(b),
        .RightShiftW => rightShiftWSuffixMle(b),
        .RightShiftWHelper => rightShiftWHelperSuffixMle(b),
        .LeftShiftWHelper => leftShiftWHelperSuffixMle(b),
        .LeftShiftW => leftShiftWSuffixMle(b),
        .OverflowBitsZero => overflowBitsZeroSuffixMle(b),
        .XorRot16 => xorRotSuffixMle(b, 16),
        .XorRot24 => xorRotSuffixMle(b, 24),
        .XorRot32 => xorRotSuffixMle(b, 32),
        .XorRot63 => xorRotSuffixMle(b, 63),
        .XorRotW16 => xorRotWSuffixMle(b, 16),
        .XorRotW12 => xorRotWSuffixMle(b, 12),
        .XorRotW8 => xorRotWSuffixMle(b, 8),
        .XorRotW7 => xorRotWSuffixMle(b, 7),
    };
}

/// Check if a suffix is {0,1}-valued (for optimization)
pub fn is01Valued(suffix: Suffixes) bool {
    return switch (suffix) {
        .One, .LessThan, .GreaterThan, .Eq, .LeftOperandIsZero, .RightOperandIsZero, .DivByZero, .OverflowBitsZero => true,
        else => false,
    };
}

// ============================================================================
// Suffix MLE Implementations
// ============================================================================

/// One suffix: always returns 1
fn oneSuffixMle(_: LookupBits(128)) u64 {
    return 1;
}

/// AND suffix: returns x & y from uninterleaved bits
fn andSuffixMle(b: LookupBits(128)) u64 {
    const parts = b.uninterleave();
    return parts.left & parts.right;
}

/// NOT-AND suffix: returns x & (~y)
fn notAndSuffixMle(b: LookupBits(128)) u64 {
    const parts = b.uninterleave();
    return parts.left & (~parts.right);
}

/// XOR suffix: returns x ^ y from uninterleaved bits
fn xorSuffixMle(b: LookupBits(128)) u64 {
    const parts = b.uninterleave();
    return parts.left ^ parts.right;
}

/// OR suffix: returns x | y from uninterleaved bits
fn orSuffixMle(b: LookupBits(128)) u64 {
    const parts = b.uninterleave();
    return parts.left | parts.right;
}

/// RightOperand suffix: returns the right operand (y) from uninterleaved bits
fn rightOperandSuffixMle(b: LookupBits(128)) u64 {
    const parts = b.uninterleave();
    return parts.right;
}

/// RightOperandW suffix: returns the right operand masked to 32 bits
fn rightOperandWSuffixMle(b: LookupBits(128)) u64 {
    const parts = b.uninterleave();
    return parts.right & 0xFFFFFFFF;
}

/// UpperWord suffix: returns the upper XLEN bits of the raw bitvector
/// For XLEN=64, this is b >> 64
fn upperWordSuffixMle(b: LookupBits(128)) u64 {
    // Return upper 64 bits of the raw bitvector (shift right by XLEN)
    return @truncate(b.value >> 64);
}

/// LowerWord suffix: returns the lower XLEN bits of the raw bitvector
/// For XLEN=64, this is b % 2^64
fn lowerWordSuffixMle(b: LookupBits(128)) u64 {
    // Return lower 64 bits of the raw bitvector
    return @truncate(b.value);
}

/// LowerHalfWord suffix: returns the lower XLEN/2 bits of the raw bitvector
/// For XLEN=64, this is b % 2^32
fn lowerHalfWordSuffixMle(b: LookupBits(128)) u64 {
    // Return lower 32 bits of the raw bitvector (XLEN/2 = 32)
    return @truncate(b.value & 0xFFFFFFFF);
}

/// LessThan suffix: returns 1 if x < y (unsigned), 0 otherwise
fn lessThanSuffixMle(b: LookupBits(128)) u64 {
    const parts = b.uninterleave();
    return if (parts.left < parts.right) 1 else 0;
}

/// GreaterThan suffix: returns 1 if x > y (unsigned), 0 otherwise
fn greaterThanSuffixMle(b: LookupBits(128)) u64 {
    const parts = b.uninterleave();
    return if (parts.left > parts.right) 1 else 0;
}

/// Eq suffix: returns 1 if x == y, 0 otherwise
fn eqSuffixMle(b: LookupBits(128)) u64 {
    const parts = b.uninterleave();
    return if (parts.left == parts.right) 1 else 0;
}

/// LeftOperandIsZero suffix: returns 1 if x == 0, 0 otherwise
fn leftOperandIsZeroSuffixMle(b: LookupBits(128)) u64 {
    const parts = b.uninterleave();
    return if (parts.left == 0) 1 else 0;
}

/// RightOperandIsZero suffix: returns 1 if y == 0, 0 otherwise
fn rightOperandIsZeroSuffixMle(b: LookupBits(128)) u64 {
    const parts = b.uninterleave();
    return if (parts.right == 0) 1 else 0;
}

/// LSB suffix: returns the least significant bit of y
/// When the suffix has 0 bits (all bound), returns 1 to match Jolt's behavior
fn lsbSuffixMle(b: LookupBits(128)) u64 {
    if (b.len == 0) {
        // Match Jolt: when all bits are bound, return 1
        return 1;
    }
    const parts = b.uninterleave();
    return parts.right & 1;
}

/// DivByZero suffix: returns 1 if divisor (x/left) == 0 AND quotient (y/right) == all_ones
/// Used in ValidDiv0Table (table 17) where the lookup pair is (divisor, quotient).
/// This detects the division-by-zero case where the quotient must be 2^XLEN - 1 (all ones).
/// Matches Jolt: divisor_is_zero && quotient_is_all_ones
fn divByZeroSuffixMle(b: LookupBits(128)) u64 {
    const parts = b.uninterleave();
    const y_len = b.len / 2;
    if (y_len == 0) {
        // When all bits are bound, both conditions reduce to prefix-level checks
        return if (parts.left == 0) 1 else 0;
    }
    const all_ones: u64 = (@as(u64, 1) << @intCast(y_len)) - 1;
    return if (parts.left == 0 and parts.right == all_ones) 1 else 0;
}

/// Pow2 suffix: returns 2^y for y in [0, 63]
/// When the suffix has 0 bits (all bound), returns 1 to match Jolt's behavior
fn pow2SuffixMle(b: LookupBits(128)) u64 {
    if (b.len == 0) {
        return 1;
    }
    const log_xlen = 6; // log2(64) = 6
    const split_result = b.split(log_xlen);
    const shift: u64 = @truncate(split_result.suffix.value);
    return @as(u64, 1) << @intCast(shift);
}

/// Pow2W suffix: returns 2^y for y in [0, 31] (32-bit version)
/// When the suffix has 0 bits (all bound), returns 1 to match Jolt's behavior
fn pow2WSuffixMle(b: LookupBits(128)) u64 {
    if (b.len == 0) {
        return 1;
    }
    // Always extract 5 bits for modulo 32
    const split_result = b.split(5);
    const shift: u64 = @truncate(split_result.suffix.value);
    return @as(u64, 1) << @intCast(shift);
}

/// SignExtension suffix: computes the SRA sign-extension padding mask.
/// For a shift bitmask y, counts trailing zeros (= shift amount),
/// then returns a mask with that many 1-bits at the top of XLEN.
/// Formula: ((1 << XLEN) - (1 << (XLEN - padding_len)))
/// Example: y=0b...1000 (3 trailing zeros) -> top 3 bits set -> 0xE000000000000000
fn signExtensionSuffixMle(b: LookupBits(128)) u64 {
    const parts = b.uninterleave();
    // Create a LookupBits for the right operand to count trailing zeros correctly
    const y_bits = LookupBits(128).new(parts.right, parts.right_len);
    const padding_len: usize = @intCast(y_bits.trailingZeros());
    // Compute ((1 << XLEN) - (1 << (XLEN - padding_len)))
    // This produces a mask with padding_len ones at the top of XLEN bits.
    // When padding_len == 0: returns 0 (no sign extension)
    // When padding_len == XLEN: returns all ones (0xFFFFFFFFFFFFFFFF)
    if (padding_len == 0) return 0;
    if (padding_len >= XLEN) return @as(u64, 0xFFFFFFFFFFFFFFFF);
    // (1 << XLEN) - (1 << (XLEN - padding_len)) computed in u128 to avoid overflow
    const result = (@as(u128, 1) << @intCast(XLEN)) - (@as(u128, 1) << @intCast(XLEN - padding_len));
    return @truncate(result);
}

/// LeftShift suffix: left-shifts x according to bitmask y
/// Removes bits of x that have 1 in y, then shifts by leading_ones(y)
fn leftShiftSuffixMle(b: LookupBits(128)) u64 {
    const parts = b.uninterleave();
    const y_bits = LookupBits(128).new(parts.right, b.len / 2);
    // Remove bits of x that have 1 in y
    const x = parts.left & ~parts.right;
    const leading = y_bits.leadingOnes();
    if (leading >= 64) return 0;
    return x << @intCast(leading);
}

/// RightShift suffix: right-aligns the masked bits of x
/// Shifts x right by trailing_zeros(y)
fn rightShiftSuffixMle(b: LookupBits(128)) u64 {
    const parts = b.uninterleave();
    const y_bits = LookupBits(128).new(parts.right, b.len / 2);
    const tz = y_bits.trailingZeros();
    if (tz >= 64) return 0;
    return parts.left >> @intCast(tz);
}

/// TwoLsb suffix: returns 1 if the two LSBs are 0 (for alignment check), 0 otherwise
/// When the suffix has 0 bits (all bound), returns 1 to match Jolt's behavior
fn twoLsbSuffixMle(b: LookupBits(128)) u64 {
    // Returns 1 if the two least significant bits are 0
    // and 0 otherwise. Returns 1 by default when len == 0.
    if (b.len == 0 or (b.value & 0b11) == 0) {
        return 1;
    }
    return 0;
}

/// ChangeDivisor suffix: returns 1 if y is all 1s and x is 0
/// Used for detecting when to change divisor in division
fn changeDivisorSuffixMle(b: LookupBits(128)) u64 {
    const parts = b.uninterleave();
    const y_len = b.len / 2;
    if (y_len >= 64) {
        // For 64-bit values, check if y == 0xFFFFFFFFFFFFFFFF and x == 0
        return if (parts.right == 0xFFFFFFFFFFFFFFFF and parts.left == 0) 1 else 0;
    }
    const all_ones: u64 = (@as(u64, 1) << @intCast(y_len)) - 1;
    return if (parts.right == all_ones and parts.left == 0) 1 else 0;
}

/// ChangeDivisorW suffix: 32-bit version of ChangeDivisor
fn changeDivisorWSuffixMle(b: LookupBits(128)) u64 {
    const parts = b.uninterleave();
    const y_len = @min(b.len / 2, XLEN / 2);
    const x: u64 = @as(u32, @truncate(parts.left));
    const y: u64 = @as(u32, @truncate(parts.right));
    const all_ones: u64 = (@as(u64, 1) << @intCast(y_len)) - 1;
    return if (y == all_ones and x == 0) 1 else 0;
}

/// Rev8W suffix: reverses bytes in each 32-bit word
fn rev8WSuffixMle(b: LookupBits(128)) u64 {
    const val: u64 = @truncate(b.value);
    const lo: u32 = @truncate(val);
    const hi: u32 = @truncate(val >> 32);
    const lo_rev = @byteSwap(lo);
    const hi_rev = @byteSwap(hi);
    return @as(u64, lo_rev) + (@as(u64, hi_rev) << 32);
}

/// RightShiftPadding suffix: computes 2^(XLEN-1-shift) for arithmetic right shift padding
fn rightShiftPaddingSuffixMle(b: LookupBits(128)) u64 {
    if (b.len == 0) {
        // Handled by prefix
        return 1;
    }
    const log_xlen = 6; // log2(64) = 6
    const split_result = b.split(log_xlen);
    const shift: u64 = @truncate(split_result.suffix.value);
    // Subtract 1 to avoid overflow; prefix compensates with factor of 2
    if (XLEN - 1 < shift) return 1;
    return @as(u64, 1) << @intCast(XLEN - 1 - shift);
}

/// RightShiftHelper suffix: returns 2^(y.leading_ones())
fn rightShiftHelperSuffixMle(b: LookupBits(128)) u64 {
    const parts = b.uninterleave();
    const y_bits = LookupBits(128).new(parts.right, b.len / 2);
    const leading = y_bits.leadingOnes();
    if (leading >= 64) return 0; // overflow protection
    return @as(u64, 1) << @intCast(leading);
}

/// SignExtensionUpperHalf suffix: handles sign extension for upper half of word
fn signExtensionUpperHalfSuffixMle(b: LookupBits(128)) u64 {
    const half_word_size = XLEN / 2;
    if (b.len >= half_word_size) {
        // Extract bit at position (half_word_size - 1), which is the sign bit
        const bits = b.value;
        const sign_bit_position = half_word_size - 1;
        const sign_bit = (bits >> @intCast(sign_bit_position)) & 1;
        if (sign_bit == 1) {
            // Return all 1s in the upper half
            return ((@as(u64, 1) << @intCast(half_word_size)) - 1) << @intCast(half_word_size);
        } else {
            return 0;
        }
    } else {
        // Suffix too small, return 1 (prefix handles)
        return 1;
    }
}

/// SignExtensionRightOperand suffix: sign extension based on right operand
fn signExtensionRightOperandSuffixMle(b: LookupBits(128)) u64 {
    if (b.len >= XLEN) {
        const bits = b.value;
        const sign_bit_position = XLEN - 2;
        const sign_bit = (bits >> @intCast(sign_bit_position)) & 1;
        if (sign_bit == 1) {
            // Return all 1s in the upper half (for 64-bit: 0xFFFFFFFF00000000)
            // This is ((1 << XLEN) - (1 << XLEN/2)) but we can compute it as:
            // 0xFFFFFFFFFFFFFFFF ^ 0x00000000FFFFFFFF = 0xFFFFFFFF00000000
            return 0xFFFFFFFF00000000;
        } else {
            return 0;
        }
    } else {
        // Suffix too small, return 1 (prefix handles)
        return 1;
    }
}

/// RightShiftW suffix: right shift for 32-bit W variant
fn rightShiftWSuffixMle(b: LookupBits(128)) u64 {
    const parts = b.uninterleave();
    const x: u32 = @truncate(parts.left);
    const y_bits = LookupBits(128).new(parts.right, @min(b.len / 2, XLEN / 2));
    const tz = y_bits.trailingZeros();
    if (tz >= 32) return 0;
    return @as(u64, x >> @intCast(tz));
}

/// RightShiftWHelper suffix: helper for 32-bit right shift
fn rightShiftWHelperSuffixMle(b: LookupBits(128)) u64 {
    const parts = b.uninterleave();
    const y_bits = LookupBits(128).new(parts.right, @min(b.len / 2, XLEN / 2));
    const leading = y_bits.leadingOnes();
    if (leading >= 64) return 0;
    return @as(u64, 1) << @intCast(leading);
}

/// LeftShiftWHelper suffix: helper for 32-bit left shift
fn leftShiftWHelperSuffixMle(b: LookupBits(128)) u64 {
    const parts = b.uninterleave();
    const y_bits = LookupBits(128).new(parts.right, b.len / 2);
    const leading = y_bits.leadingOnes();
    // Truncate to 32 bits
    const result: u64 = @as(u64, 1) << @intCast(@min(leading, 31));
    return @as(u32, @truncate(result));
}

/// LeftShiftW suffix: left shift for 32-bit W variant
fn leftShiftWSuffixMle(b: LookupBits(128)) u64 {
    const parts = b.uninterleave();
    const y_bits = LookupBits(128).new(parts.right, @min(b.len / 2, XLEN / 2));
    const x: u32 = @truncate(parts.left);
    const y_u: u32 = @truncate(parts.right);
    // Remove bits of x that have 1 in y
    const masked_x = x & ~y_u;
    const leading = y_bits.leadingOnes();
    if (leading >= 32) return 0;
    return @as(u64, masked_x << @intCast(leading));
}

/// OverflowBitsZero suffix: returns 1 if upper bits are all zero (no overflow)
fn overflowBitsZeroSuffixMle(b: LookupBits(128)) u64 {
    const upper_bits = b.value >> @intCast(XLEN);
    return if (upper_bits == 0) 1 else 0;
}

/// XorRot suffix: XOR operands and rotate right by constant
fn xorRotSuffixMle(b: LookupBits(128), comptime rotation: u6) u64 {
    const parts = b.uninterleave();
    const xor_result = parts.left ^ parts.right;
    return std.math.rotr(u64, xor_result, rotation);
}

/// XorRotW suffix: 32-bit XOR and rotate right by constant
fn xorRotWSuffixMle(b: LookupBits(128), comptime rotation: u5) u64 {
    const parts = b.uninterleave();
    const x_32: u32 = @truncate(parts.left);
    const y_32: u32 = @truncate(parts.right);
    const xor_result = x_32 ^ y_32;
    return std.math.rotr(u32, xor_result, rotation);
}

// ============================================================================
// Lookup Table Suffix Configurations
// ============================================================================

/// Define which suffixes each lookup table uses
/// Returns an array of suffixes for the given table index
///
/// Reference: jolt-core/src/zkvm/lookup_table/*.rs - each table's suffixes() method
pub fn tableSuffixes(table_idx: usize) []const Suffixes {
    return switch (table_idx) {
        // 0: RangeCheck
        0 => &[_]Suffixes{ .One, .LowerWord },
        // 1: RangeCheckAligned
        1 => &[_]Suffixes{ .One, .LowerWord, .Lsb },
        // 2: And
        2 => &[_]Suffixes{ .One, .And },
        // 3: Andn
        3 => &[_]Suffixes{ .One, .NotAnd },
        // 4: Or
        4 => &[_]Suffixes{ .One, .Or },
        // 5: Xor
        5 => &[_]Suffixes{ .One, .Xor },
        // 6: Equal
        6 => &[_]Suffixes{.Eq},
        // 7: SignedGreaterThanEqual
        7 => &[_]Suffixes{ .One, .LessThan },
        // 8: UnsignedGreaterThanEqual
        8 => &[_]Suffixes{ .One, .LessThan },
        // 9: NotEqual
        9 => &[_]Suffixes{ .One, .Eq },
        // 10: SignedLessThan
        10 => &[_]Suffixes{ .One, .LessThan },
        // 11: UnsignedLessThan
        11 => &[_]Suffixes{ .One, .LessThan },
        // 12: Movsign
        12 => &[_]Suffixes{.One},
        // 13: UpperWord
        13 => &[_]Suffixes{ .One, .UpperWord },
        // 14: LessThanEqual (UnsignedLessThanEqual)
        14 => &[_]Suffixes{ .One, .LessThan, .Eq },
        // 15: ValidSignedRemainder
        // Jolt order: [One, LessThan, GreaterThan, LeftOperandIsZero, RightOperandIsZero]
        15 => &[_]Suffixes{ .One, .LessThan, .GreaterThan, .LeftOperandIsZero, .RightOperandIsZero },
        // 16: ValidUnsignedRemainder
        // Jolt order: [One, LessThan, RightOperandIsZero]
        16 => &[_]Suffixes{ .One, .LessThan, .RightOperandIsZero },
        // 17: ValidDiv0
        17 => &[_]Suffixes{ .One, .LeftOperandIsZero, .DivByZero },
        // 18: HalfwordAlignment
        18 => &[_]Suffixes{ .One, .Lsb },
        // 19: WordAlignment
        19 => &[_]Suffixes{.TwoLsb},
        // 20: LowerHalfWord
        20 => &[_]Suffixes{ .One, .LowerHalfWord },
        // 21: SignExtendHalfWord
        21 => &[_]Suffixes{ .One, .LowerHalfWord, .SignExtensionUpperHalf },
        // 22: Pow2
        22 => &[_]Suffixes{.Pow2},
        // 23: Pow2W
        23 => &[_]Suffixes{.Pow2W},
        // 24: ShiftRightBitmask
        24 => &[_]Suffixes{ .One, .Pow2 },
        // 25: VirtualRev8W
        25 => &[_]Suffixes{ .One, .Rev8W },
        // 26: VirtualSRL
        26 => &[_]Suffixes{ .RightShiftHelper, .RightShift },
        // 27: VirtualSRA
        27 => &[_]Suffixes{ .RightShiftHelper, .RightShift, .SignExtension, .One },
        // 28: VirtualROTR
        28 => &[_]Suffixes{ .RightShiftHelper, .RightShift, .LeftShift, .One },
        // 29: VirtualROTRW
        29 => &[_]Suffixes{ .RightShiftWHelper, .RightShiftW, .LeftShiftW, .One },
        // 30: VirtualChangeDivisor
        30 => &[_]Suffixes{ .One, .RightOperand, .ChangeDivisor },
        // 31: VirtualChangeDivisorW
        31 => &[_]Suffixes{ .One, .RightOperandW, .ChangeDivisorW, .SignExtensionRightOperand },
        // 32: MulUNoOverflow
        32 => &[_]Suffixes{.OverflowBitsZero},
        // 33: VirtualXORROT32
        33 => &[_]Suffixes{ .One, .XorRot32 },
        // 34: VirtualXORROT24
        34 => &[_]Suffixes{ .One, .XorRot24 },
        // 35: VirtualXORROT16
        35 => &[_]Suffixes{ .One, .XorRot16 },
        // 36: VirtualXORROT63
        36 => &[_]Suffixes{ .One, .XorRot63 },
        // 37: VirtualXORROTW16
        37 => &[_]Suffixes{ .One, .XorRotW16 },
        // 38: VirtualXORROTW12
        38 => &[_]Suffixes{ .One, .XorRotW12 },
        // 39: VirtualXORROTW8
        39 => &[_]Suffixes{ .One, .XorRotW8 },
        // 40: VirtualXORROTW7
        40 => &[_]Suffixes{ .One, .XorRotW7 },
        // Fallback
        else => &[_]Suffixes{.One},
    };
}

// ============================================================================
// Tests
// ============================================================================

test "suffix_mle One" {
    const b = LookupBits(128).new(0x12345678, 64);
    const result = suffixMle(.One, b);
    try std.testing.expectEqual(@as(u64, 1), result);
}

test "suffix_mle And" {
    // In Jolt's interleave format: left at ODD positions, right at EVEN positions
    // For 0xF = 0b00001111 with 8 bits (positions 0-7):
    // - left (odd positions 1,3,5,7): bits 1,3 set, 5,7 clear = 0b0011 = 3
    // - right (even positions 0,2,4,6): bits 0,2 set, 4,6 clear = 0b0011 = 3
    // AND = 3 & 3 = 3
    const b = LookupBits(128).new(0xF, 8);
    const result = suffixMle(.And, b);
    const parts = b.uninterleave();
    try std.testing.expectEqual(parts.left & parts.right, result);
}

test "suffix_mle LessThan" {
    // x = 3, y = 5 -> x < y = true = 1
    // Interleaved: x=3 (0b11), y=5 (0b101)
    // For 8 bits: x_bits=0b0011, y_bits=0b0101
    // Interleaved from LSB: x0y0 x1y1 x2y2 x3y3
    // = 11 11 00 01 = 0b01001111 = 0x4F
    const b = LookupBits(128).new(0x4F, 8);
    const parts = b.uninterleave();
    // Check the uninterleaved values
    const result = suffixMle(.LessThan, b);
    const expected: u64 = if (parts.left < parts.right) 1 else 0;
    try std.testing.expectEqual(expected, result);
}
