//! Multi-limb big integer arithmetic over fixed-size `[N]u64` arrays.
//!
//! This is the foundation `zolt-arith` builds field arithmetic on top of.
//! It is intentionally narrow and unopinionated:
//!
//!   - Limbs are little-endian: `limbs[0]` is the least-significant 64-bit
//!     word. The numeric value is sum_{i} limbs[i] * 2^(64*i).
//!   - Operations are constant-shape (no early exit on equality), but NOT
//!     constant-time on every CPU — branches around carry chains exist
//!     for portability. The constant-time story comes when the BLS12-381
//!     instantiation lands and we can pin a specific target.
//!   - There is no allocation. Every helper takes its result by `*[N]u64`
//!     or returns a `[N]u64` value.
//!
//! The 256-bit (4-limb) and 384-bit (6-limb) cases both flow through the
//! same comptime-generic code, so the BN254 / BLS12-381 split lives in
//! `field.zig` rather than here.

const std = @import("std");

/// Add `a + b` modulo 2^(64*N), returning the carry-out as `u1`.
/// Out-of-place to keep the function easy to use as a building block;
/// callers that need an in-place version can pass the destination as
/// `out` and one of the inputs as `b`.
pub fn add(comptime N: comptime_int, out: *[N]u64, a: [N]u64, b: [N]u64) u1 {
    var carry: u1 = 0;
    inline for (0..N) |i| {
        const sum_ab = @addWithOverflow(a[i], b[i]);
        const sum_c = @addWithOverflow(sum_ab[0], carry);
        out[i] = sum_c[0];
        carry = sum_ab[1] | sum_c[1];
    }
    return carry;
}

/// Subtract `a - b` modulo 2^(64*N), returning the borrow-out as `u1`.
/// `out` may alias either input.
pub fn sub(comptime N: comptime_int, out: *[N]u64, a: [N]u64, b: [N]u64) u1 {
    var borrow: u1 = 0;
    inline for (0..N) |i| {
        const diff_ab = @subWithOverflow(a[i], b[i]);
        const diff_c = @subWithOverflow(diff_ab[0], borrow);
        out[i] = diff_c[0];
        borrow = diff_ab[1] | diff_c[1];
    }
    return borrow;
}

/// Lexicographic comparison treating both operands as little-endian
/// limb arrays. Returns `.lt`, `.eq`, or `.gt` matching `std.math.Order`.
pub fn cmp(comptime N: comptime_int, a: [N]u64, b: [N]u64) std.math.Order {
    var i: usize = N;
    while (i > 0) {
        i -= 1;
        if (a[i] < b[i]) return .lt;
        if (a[i] > b[i]) return .gt;
    }
    return .eq;
}

/// `a == 0` for an N-limb integer.
pub fn isZero(comptime N: comptime_int, a: [N]u64) bool {
    inline for (0..N) |i| {
        if (a[i] != 0) return false;
    }
    return true;
}

/// `a == 1` for an N-limb integer (least-significant limb only).
pub fn isOne(comptime N: comptime_int, a: [N]u64) bool {
    if (a[0] != 1) return false;
    inline for (1..N) |i| {
        if (a[i] != 0) return false;
    }
    return true;
}

/// Bit length of an N-limb integer (1-indexed; 0 for zero).
pub fn bitLen(comptime N: comptime_int, a: [N]u64) usize {
    var i: usize = N;
    while (i > 0) {
        i -= 1;
        if (a[i] != 0) return i * 64 + (64 - @clz(a[i]));
    }
    return 0;
}

/// Read a little-endian byte slice into an N-limb integer. The slice
/// length must be at most `N * 8`; remaining high bytes are zero.
pub fn fromBytesLe(comptime N: comptime_int, bytes: []const u8) [N]u64 {
    std.debug.assert(bytes.len <= N * 8);
    var out: [N]u64 = .{0} ** N;
    var buf: [N * 8]u8 = .{0} ** (N * 8);
    @memcpy(buf[0..bytes.len], bytes);
    inline for (0..N) |i| {
        out[i] = std.mem.readInt(u64, buf[i * 8 ..][0..8], .little);
    }
    return out;
}

/// Write an N-limb integer into a little-endian byte slice. The slice
/// length must be exactly `N * 8`.
pub fn toBytesLe(comptime N: comptime_int, value: [N]u64, out: []u8) void {
    std.debug.assert(out.len == N * 8);
    inline for (0..N) |i| {
        std.mem.writeInt(u64, out[i * 8 ..][0..8], value[i], .little);
    }
}

/// Read a big-endian byte slice into an N-limb integer (the natural
/// encoding for cryptographic public keys / signatures). Slice length
/// must be at most `N * 8`.
pub fn fromBytesBe(comptime N: comptime_int, bytes: []const u8) [N]u64 {
    std.debug.assert(bytes.len <= N * 8);
    var out: [N]u64 = .{0} ** N;
    var buf: [N * 8]u8 = .{0} ** (N * 8);
    // Right-align the input so the most-significant byte lands at the
    // top of the buffer (matches the BE convention).
    const offset = (N * 8) - bytes.len;
    @memcpy(buf[offset..][0..bytes.len], bytes);
    // Walk limbs from MSB to LSB, reading 8 BE bytes each.
    inline for (0..N) |i| {
        const limb_bytes = buf[(N - 1 - i) * 8 ..][0..8];
        out[i] = std.mem.readInt(u64, limb_bytes, .big);
    }
    return out;
}

/// Write an N-limb integer as a big-endian byte slice. Slice length
/// must be exactly `N * 8`.
pub fn toBytesBe(comptime N: comptime_int, value: [N]u64, out: []u8) void {
    std.debug.assert(out.len == N * 8);
    inline for (0..N) |i| {
        std.mem.writeInt(u64, out[(N - 1 - i) * 8 ..][0..8], value[i], .big);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const testing = std.testing;

test "add: 0 + 0 = 0 with no carry" {
    var out: [4]u64 = undefined;
    const carry = add(4, &out, .{ 0, 0, 0, 0 }, .{ 0, 0, 0, 0 });
    try testing.expectEqual(@as(u1, 0), carry);
    try testing.expectEqual(@as(u64, 0), out[0]);
}

test "add: 1 + 1 = 2 with no carry (4 limbs)" {
    var out: [4]u64 = undefined;
    const carry = add(4, &out, .{ 1, 0, 0, 0 }, .{ 1, 0, 0, 0 });
    try testing.expectEqual(@as(u1, 0), carry);
    try testing.expectEqual(@as(u64, 2), out[0]);
}

test "add: u64::MAX + 1 propagates carry through all limbs" {
    var out: [4]u64 = undefined;
    const carry = add(
        4,
        &out,
        .{ std.math.maxInt(u64), std.math.maxInt(u64), std.math.maxInt(u64), std.math.maxInt(u64) },
        .{ 1, 0, 0, 0 },
    );
    try testing.expectEqual(@as(u1, 1), carry);
    try testing.expectEqual([_]u64{ 0, 0, 0, 0 }, out);
}

test "add: 6-limb (BLS12-381 width) carry chain" {
    var out: [6]u64 = undefined;
    const carry = add(
        6,
        &out,
        .{ std.math.maxInt(u64), 0, 0, 0, 0, 0 },
        .{ 1, 0, 0, 0, 0, 0 },
    );
    try testing.expectEqual(@as(u1, 0), carry);
    try testing.expectEqual([_]u64{ 0, 1, 0, 0, 0, 0 }, out);
}

test "sub: simple no-borrow case" {
    var out: [4]u64 = undefined;
    const borrow = sub(4, &out, .{ 5, 0, 0, 0 }, .{ 3, 0, 0, 0 });
    try testing.expectEqual(@as(u1, 0), borrow);
    try testing.expectEqual(@as(u64, 2), out[0]);
}

test "sub: borrow propagates through limbs" {
    var out: [4]u64 = undefined;
    const borrow = sub(4, &out, .{ 0, 1, 0, 0 }, .{ 1, 0, 0, 0 });
    try testing.expectEqual(@as(u1, 0), borrow);
    try testing.expectEqual([_]u64{ std.math.maxInt(u64), 0, 0, 0 }, out);
}

test "sub: underflow yields borrow=1" {
    var out: [4]u64 = undefined;
    const borrow = sub(4, &out, .{ 0, 0, 0, 0 }, .{ 1, 0, 0, 0 });
    try testing.expectEqual(@as(u1, 1), borrow);
    try testing.expectEqual([_]u64{ std.math.maxInt(u64), std.math.maxInt(u64), std.math.maxInt(u64), std.math.maxInt(u64) }, out);
}

test "cmp: lexicographic order across limbs" {
    try testing.expectEqual(std.math.Order.eq, cmp(4, .{ 1, 2, 3, 4 }, .{ 1, 2, 3, 4 }));
    try testing.expectEqual(std.math.Order.lt, cmp(4, .{ 1, 2, 3, 4 }, .{ 1, 2, 3, 5 }));
    try testing.expectEqual(std.math.Order.gt, cmp(4, .{ 1, 2, 3, 5 }, .{ 1, 2, 3, 4 }));
    // High limb dominates over low limbs.
    try testing.expectEqual(
        std.math.Order.gt,
        cmp(4, .{ 0, 0, 0, 1 }, .{ std.math.maxInt(u64), std.math.maxInt(u64), std.math.maxInt(u64), 0 }),
    );
}

test "isZero / isOne" {
    try testing.expect(isZero(4, .{ 0, 0, 0, 0 }));
    try testing.expect(!isZero(4, .{ 1, 0, 0, 0 }));
    try testing.expect(!isZero(4, .{ 0, 0, 0, 1 }));
    try testing.expect(isOne(4, .{ 1, 0, 0, 0 }));
    try testing.expect(!isOne(4, .{ 2, 0, 0, 0 }));
    try testing.expect(!isOne(4, .{ 1, 1, 0, 0 }));
}

test "bitLen" {
    try testing.expectEqual(@as(usize, 0), bitLen(4, .{ 0, 0, 0, 0 }));
    try testing.expectEqual(@as(usize, 1), bitLen(4, .{ 1, 0, 0, 0 }));
    try testing.expectEqual(@as(usize, 64), bitLen(4, .{ std.math.maxInt(u64), 0, 0, 0 }));
    try testing.expectEqual(@as(usize, 65), bitLen(4, .{ 0, 1, 0, 0 }));
    try testing.expectEqual(@as(usize, 256), bitLen(4, .{ 0, 0, 0, 1 << 63 }));
}

test "fromBytesLe / toBytesLe round-trip" {
    const original: [4]u64 = .{ 0x0102030405060708, 0x1112131415161718, 0x2122232425262728, 0x3132333435363738 };
    var bytes: [32]u8 = undefined;
    toBytesLe(4, original, &bytes);
    // LE within each limb: limb 0's least-significant byte (0x08) is at
    // index 0, limb 3's most-significant byte (0x31) is at index 31.
    try testing.expectEqual(@as(u8, 0x08), bytes[0]);
    try testing.expectEqual(@as(u8, 0x31), bytes[31]);
    const round_tripped = fromBytesLe(4, &bytes);
    try testing.expectEqual(original, round_tripped);
}

test "fromBytesBe / toBytesBe round-trip" {
    const original: [4]u64 = .{ 0x0102030405060708, 0x1112131415161718, 0x2122232425262728, 0x3132333435363738 };
    var bytes: [32]u8 = undefined;
    toBytesBe(4, original, &bytes);
    // BE: most-significant limb first, MSB first within the limb.
    try testing.expectEqual(@as(u8, 0x31), bytes[0]);
    try testing.expectEqual(@as(u8, 0x08), bytes[31]);
    const round_tripped = fromBytesBe(4, &bytes);
    try testing.expectEqual(original, round_tripped);
}

test "fromBytesBe handles short input by left-padding with zeroes" {
    // Public BLS12-381 G1 keys are 48 bytes (6 limbs). A short input
    // should land in the high bytes — this is the natural BE convention.
    const short = [_]u8{ 0xab, 0xcd };
    const value = fromBytesBe(6, &short);
    try testing.expectEqual(@as(u64, 0), value[5]);
    try testing.expectEqual(@as(u64, 0xabcd), value[0]);
}
