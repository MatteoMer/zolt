//! Bit manipulation utilities for lookup table indices.
//!
//! Contains interleave/uninterleave operations and the LookupBits type.
//! These are pure bit math with zero dependencies beyond std.

const std = @import("std");

/// Utility function to uninterleave bits from an interleaved index.
/// For a 2*XLEN bit index where bits are interleaved as y[0], x[0], y[1], x[1], ...,
/// returns (x, y) where each is an XLEN-bit value.
///
/// Note: This matches Jolt's convention where y bits are at even positions and x bits at odd.
pub fn uninterleaveBits(index: u128) struct { x: u64, y: u64 } {
    // Use Jolt's efficient bit manipulation algorithm
    // x comes from odd bit positions, y from even
    var x_bits: u128 = (index >> 1) & 0x5555_5555_5555_5555_5555_5555_5555_5555;
    var y_bits: u128 = index & 0x5555_5555_5555_5555_5555_5555_5555_5555;

    // Compact x bits into lower half
    x_bits = (x_bits | (x_bits >> 1)) & 0x3333_3333_3333_3333_3333_3333_3333_3333;
    x_bits = (x_bits | (x_bits >> 2)) & 0x0F0F_0F0F_0F0F_0F0F_0F0F_0F0F_0F0F_0F0F;
    x_bits = (x_bits | (x_bits >> 4)) & 0x00FF_00FF_00FF_00FF_00FF_00FF_00FF_00FF;
    x_bits = (x_bits | (x_bits >> 8)) & 0x0000_FFFF_0000_FFFF_0000_FFFF_0000_FFFF;
    x_bits = (x_bits | (x_bits >> 16)) & 0x0000_0000_FFFF_FFFF_0000_0000_FFFF_FFFF;
    x_bits = (x_bits | (x_bits >> 32)) & 0x0000_0000_0000_0000_FFFF_FFFF_FFFF_FFFF;

    // Compact y bits into lower half
    y_bits = (y_bits | (y_bits >> 1)) & 0x3333_3333_3333_3333_3333_3333_3333_3333;
    y_bits = (y_bits | (y_bits >> 2)) & 0x0F0F_0F0F_0F0F_0F0F_0F0F_0F0F_0F0F_0F0F;
    y_bits = (y_bits | (y_bits >> 4)) & 0x00FF_00FF_00FF_00FF_00FF_00FF_00FF_00FF;
    y_bits = (y_bits | (y_bits >> 8)) & 0x0000_FFFF_0000_FFFF_0000_FFFF_0000_FFFF;
    y_bits = (y_bits | (y_bits >> 16)) & 0x0000_0000_FFFF_FFFF_0000_0000_FFFF_FFFF;
    y_bits = (y_bits | (y_bits >> 32)) & 0x0000_0000_0000_0000_FFFF_FFFF_FFFF_FFFF;

    return .{ .x = @truncate(x_bits), .y = @truncate(y_bits) };
}

/// Interleave bits of x and y into a single index.
/// Produces a value where y bits are at even positions and x bits at odd positions.
/// This matches Jolt's convention: result = (spread(x) << 1) | spread(y)
pub fn interleaveBits(x: u64, y: u64) u128 {
    // Use Jolt's efficient bit spreading algorithm
    // Spread x_bits to odd positions
    var x_bits: u128 = @as(u128, x);
    x_bits = (x_bits | (x_bits << 32)) & 0x0000_0000_FFFF_FFFF_0000_0000_FFFF_FFFF;
    x_bits = (x_bits | (x_bits << 16)) & 0x0000_FFFF_0000_FFFF_0000_FFFF_0000_FFFF;
    x_bits = (x_bits | (x_bits << 8)) & 0x00FF_00FF_00FF_00FF_00FF_00FF_00FF_00FF;
    x_bits = (x_bits | (x_bits << 4)) & 0x0F0F_0F0F_0F0F_0F0F_0F0F_0F0F_0F0F_0F0F;
    x_bits = (x_bits | (x_bits << 2)) & 0x3333_3333_3333_3333_3333_3333_3333_3333;
    x_bits = (x_bits | (x_bits << 1)) & 0x5555_5555_5555_5555_5555_5555_5555_5555;

    // Spread y_bits to even positions
    var y_bits: u128 = @as(u128, y);
    y_bits = (y_bits | (y_bits << 32)) & 0x0000_0000_FFFF_FFFF_0000_0000_FFFF_FFFF;
    y_bits = (y_bits | (y_bits << 16)) & 0x0000_FFFF_0000_FFFF_0000_FFFF_0000_FFFF;
    y_bits = (y_bits | (y_bits << 8)) & 0x00FF_00FF_00FF_00FF_00FF_00FF_00FF_00FF;
    y_bits = (y_bits | (y_bits << 4)) & 0x0F0F_0F0F_0F0F_0F0F_0F0F_0F0F_0F0F_0F0F;
    y_bits = (y_bits | (y_bits << 2)) & 0x3333_3333_3333_3333_3333_3333_3333_3333;
    y_bits = (y_bits | (y_bits << 1)) & 0x5555_5555_5555_5555_5555_5555_5555_5555;

    return (x_bits << 1) | y_bits;
}

/// A bitvector type used to represent a (substring of a) lookup index.
/// Optimized representation that stores up to 128 bits with explicit length.
///
/// This is the Zig port of Jolt's `LookupBits` type.
pub const LookupBits = struct {
    const Self = @This();

    /// The bits stored as a 128-bit value (little-endian semantics)
    bits: u128,
    /// Number of valid bits in this LookupBits
    len: u8,

    /// Create a new LookupBits from a value and length
    pub fn init(bits_value: u128, bit_len: usize) Self {
        std.debug.assert(bit_len <= 128);

        // Mask to the specified length
        const masked = if (bit_len < 128)
            bits_value % (@as(u128, 1) << @intCast(bit_len))
        else
            bits_value;

        return Self{
            .bits = masked,
            .len = @intCast(bit_len),
        };
    }

    /// Create an empty LookupBits
    pub fn empty() Self {
        return Self{
            .bits = 0,
            .len = 0,
        };
    }

    /// Get the length in bits
    pub fn length(self: Self) usize {
        return @as(usize, self.len);
    }

    /// Convert to u128
    pub fn toU128(self: Self) u128 {
        return self.bits;
    }

    /// Convert to usize
    pub fn toUsize(self: Self) usize {
        return @intCast(self.bits);
    }

    /// Convert to u64
    pub fn toU64(self: Self) u64 {
        return @truncate(self.bits);
    }

    /// Uninterleave bits into (x, y) components
    pub fn uninterleave(self: Self) struct { x: Self, y: Self } {
        const result = uninterleaveBits(self.bits);
        const x_len = self.len / 2;
        const y_len = self.len - x_len;
        return .{
            .x = Self.init(@as(u128, result.x), x_len),
            .y = Self.init(@as(u128, result.y), y_len),
        };
    }

    /// Split into (prefix, suffix) where suffix has the given length
    pub fn split(self: Self, suffix_len: usize) struct { prefix: Self, suffix: Self } {
        std.debug.assert(suffix_len <= self.len);

        const suffix_bits = self.bits % (@as(u128, 1) << @intCast(suffix_len));
        const suffix = Self.init(suffix_bits, suffix_len);

        const prefix_bits = self.bits >> @intCast(suffix_len);
        const prefix_len = @as(usize, self.len) - suffix_len;
        const prefix = Self.init(prefix_bits, prefix_len);

        return .{ .prefix = prefix, .suffix = suffix };
    }

    /// Pop the most significant bit, decrementing length
    pub fn popMsb(self: *Self) u8 {
        if (self.len == 0) return 0;

        const shift: u7 = @intCast(self.len - 1);
        const msb: u8 = @intCast((self.bits >> shift) & 1);
        self.bits = self.bits % (@as(u128, 1) << shift);
        self.len -= 1;
        return msb;
    }

    /// Get the bit at a specific position (0-indexed from LSB)
    pub fn getBit(self: Self, pos: usize) u1 {
        if (pos >= self.len) return 0;
        return @intCast((self.bits >> @intCast(pos)) & 1);
    }

    /// Count trailing zeros
    pub fn trailingZeros(self: Self) u32 {
        if (self.bits == 0) return @as(u32, self.len);
        const tz = @ctz(self.bits);
        return @min(tz, @as(u32, self.len));
    }

    /// Count leading ones (in the context of the bit length)
    pub fn leadingOnes(self: Self) u32 {
        if (self.len == 0) return 0;
        const shifted = self.bits << @intCast(128 - @as(u8, self.len));
        // Count leading ones by inverting and counting leading zeros
        return @clz(~shifted);
    }

    /// Check equality
    pub fn eql(self: Self, other: Self) bool {
        return self.bits == other.bits and self.len == other.len;
    }

    /// Bitwise AND with a mask
    pub fn bitAnd(self: Self, mask: u128) u128 {
        return self.bits & mask;
    }
};

test "uninterleave and interleave bits" {
    const x: u64 = 0b1010;
    const y: u64 = 0b1100;

    const interleaved = interleaveBits(x, y);
    const result = uninterleaveBits(interleaved);

    try std.testing.expectEqual(x, result.x);
    try std.testing.expectEqual(y, result.y);
}

test "uninterleave bits pattern" {
    const index: u128 = 0b11_10_01_00;
    const result = uninterleaveBits(index);

    try std.testing.expectEqual(@as(u64, 0b1100), result.x);
    try std.testing.expectEqual(@as(u64, 0b1010), result.y);
}

test "LookupBits init" {
    const bits = LookupBits.init(0b1101, 4);
    try std.testing.expectEqual(@as(u128, 0b1101), bits.bits);
    try std.testing.expectEqual(@as(u8, 4), bits.len);
}

test "LookupBits init masks correctly" {
    const bits = LookupBits.init(0xFF, 4);
    try std.testing.expectEqual(@as(u128, 0x0F), bits.bits);
}

test "LookupBits split" {
    const bits = LookupBits.init(0b11_01_10, 6);
    const result = bits.split(2);

    try std.testing.expectEqual(@as(u128, 0b10), result.suffix.bits);
    try std.testing.expectEqual(@as(u8, 2), result.suffix.len);

    try std.testing.expectEqual(@as(u128, 0b1101), result.prefix.bits);
    try std.testing.expectEqual(@as(u8, 4), result.prefix.len);
}

test "LookupBits popMsb" {
    var bits = LookupBits.init(0b1101, 4);

    const msb1 = bits.popMsb();
    try std.testing.expectEqual(@as(u8, 1), msb1);
    try std.testing.expectEqual(@as(u128, 0b101), bits.bits);
    try std.testing.expectEqual(@as(u8, 3), bits.len);

    const msb2 = bits.popMsb();
    try std.testing.expectEqual(@as(u8, 1), msb2);
    try std.testing.expectEqual(@as(u128, 0b01), bits.bits);
    try std.testing.expectEqual(@as(u8, 2), bits.len);

    const msb3 = bits.popMsb();
    try std.testing.expectEqual(@as(u8, 0), msb3);
    try std.testing.expectEqual(@as(u128, 0b1), bits.bits);
    try std.testing.expectEqual(@as(u8, 1), bits.len);
}

test "LookupBits uninterleave" {
    const bits = LookupBits.init(0b11_10_01_00, 8);
    const result = bits.uninterleave();

    try std.testing.expectEqual(@as(u128, 0b1100), result.x.bits);
    try std.testing.expectEqual(@as(u128, 0b1010), result.y.bits);
}

test "LookupBits trailingZeros" {
    const bits1 = LookupBits.init(0b1000, 4);
    try std.testing.expectEqual(@as(u32, 3), bits1.trailingZeros());

    const bits2 = LookupBits.init(0b1001, 4);
    try std.testing.expectEqual(@as(u32, 0), bits2.trailingZeros());

    const bits3 = LookupBits.init(0, 4);
    try std.testing.expectEqual(@as(u32, 4), bits3.trailingZeros());
}

test "LookupBits getBit" {
    const bits = LookupBits.init(0b1010, 4);
    try std.testing.expectEqual(@as(u1, 0), bits.getBit(0));
    try std.testing.expectEqual(@as(u1, 1), bits.getBit(1));
    try std.testing.expectEqual(@as(u1, 0), bits.getBit(2));
    try std.testing.expectEqual(@as(u1, 1), bits.getBit(3));
    try std.testing.expectEqual(@as(u1, 0), bits.getBit(4));
}
