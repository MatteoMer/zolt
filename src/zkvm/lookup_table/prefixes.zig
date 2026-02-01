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
        .UpperWord => upperWordPrefixMle(F, checkpoints, r_x, c, b, j),
        .And => andPrefixMle(F, checkpoints, r_x, c, b, j),
        .Or => orPrefixMle(F, checkpoints, r_x, c, b, j),
        .Xor => xorPrefixMle(F, checkpoints, r_x, c, b, j),
        .LessThan => lessThanPrefixMle(F, checkpoints, r_x, c, b, j),
        .LeftOperandIsZero => leftIsZeroPrefixMle(F, checkpoints, r_x, c, b, j),
        .RightOperandIsZero => rightIsZeroPrefixMle(F, checkpoints, r_x, c, b, j),
        .LeftOperandMsb => leftMsbPrefixMle(F, checkpoints, r_x, c, b, j),
        .RightOperandMsb => rightMsbPrefixMle(F, checkpoints, r_x, c, b, j),
        // For prefixes not yet implemented, return zero
        else => F.zero(),
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
        .UpperWord => upperWordUpdateCheckpoint(F, checkpoints, r_x, r_y, j, suffix_len),
        .And => andUpdateCheckpoint(F, checkpoints, r_x, r_y, j, suffix_len),
        .Or => orUpdateCheckpoint(F, checkpoints, r_x, r_y, j, suffix_len),
        .Xor => xorUpdateCheckpoint(F, checkpoints, r_x, r_y, j, suffix_len),
        .LessThan => lessThanUpdateCheckpoint(F, checkpoints, r_x, r_y, j, suffix_len),
        .LeftOperandIsZero => leftIsZeroUpdateCheckpoint(F, checkpoints, r_x, r_y, j, suffix_len),
        .RightOperandIsZero => rightIsZeroUpdateCheckpoint(F, checkpoints, r_x, r_y, j, suffix_len),
        .LeftOperandMsb => leftMsbUpdateCheckpoint(F, checkpoints, r_x, r_y, j, suffix_len),
        .RightOperandMsb => rightMsbUpdateCheckpoint(F, checkpoints, r_x, r_y, j, suffix_len),
        // For prefixes not yet implemented, return null
        else => null,
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
