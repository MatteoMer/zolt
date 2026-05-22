const std = @import("std");
const field = @import("zolt_arith").field;
const BN254Scalar = field.BN254Scalar;
const lookup_table = @import("mod.zig");

const Table = lookup_table.LookupTable(BN254Scalar, 64);
const XLEN: usize = 64;
const NUM_LOOKUP_BITS: usize = 2 * XLEN;

const requested_tables = [_]usize{ 22, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39 };

fn bitsFromIndex(index: u128) [NUM_LOOKUP_BITS]BN254Scalar {
    var r: [NUM_LOOKUP_BITS]BN254Scalar = undefined;
    inline for (0..NUM_LOOKUP_BITS) |j| {
        const bit_pos: u7 = @intCast(NUM_LOOKUP_BITS - 1 - j);
        const bit = (index >> bit_pos) & 1;
        r[j] = if (bit == 1) BN254Scalar.one() else BN254Scalar.zero();
    }
    return r;
}

fn expectTableMatches(table_index: usize, index: u128) !void {
    var r = bitsFromIndex(index);
    const mle = Table.evaluateTableMLE(table_index, &r);
    const materialized = Table.materializeTableEntry(table_index, index);
    const expected = BN254Scalar.fromU64(materialized);
    if (!mle.eql(expected)) {
        std.debug.print("Boolean MLE mismatch: table={d} index=0x{x} materialized=0x{x}\n", .{ table_index, index, materialized });
        return error.BooleanMleMismatch;
    }
}

fn expectMaterialize(table_index: usize, index: u128, expected: u64) !void {
    try std.testing.expectEqual(expected, Table.materializeTableEntry(table_index, index));
}

fn i64Bits(value: i64) u64 {
    return @as(u64, @bitCast(value));
}

fn i32Bits(value: i32) u64 {
    return @as(u64, @as(u32, @bitCast(value)));
}

fn signExtendI32Bits(value: i32) u64 {
    return @as(u64, @bitCast(@as(i64, value)));
}

fn expectChangeDivisor(dividend: u64, divisor: u64, expected: u64) !void {
    try expectMaterialize(29, lookup_table.interleaveBits(dividend, divisor), expected);
}

fn expectChangeDivisorW(dividend: u64, divisor: u64, expected: u64) !void {
    try expectMaterialize(30, lookup_table.interleaveBits(dividend, divisor), expected);
}

fn expectXorRotMaterialize(table_index: usize, rotation: u6, x: u64, y: u64) !void {
    const index = lookup_table.interleaveBits(x, y);
    const expected = std.math.rotr(u64, x ^ y, rotation);
    try expectMaterialize(table_index, index, expected);
}

fn expectXorRotWMaterialize(table_index: usize, rotation: u5, x: u64, y: u64) !void {
    const index = lookup_table.interleaveBits(x, y);
    const x32: u32 = @truncate(x);
    const y32: u32 = @truncate(y);
    const expected: u64 = std.math.rotr(u32, x32 ^ y32, rotation);
    try expectMaterialize(table_index, index, expected);
}

fn nextRandom(state: *u64) u64 {
    state.* = state.* *% 6364136223846793005 +% 1442695040888963407;
    return state.*;
}

fn randomIndexForTable(table_index: usize, state: *u64) u128 {
    return switch (table_index) {
        22, 31 => (@as(u128, nextRandom(state)) << 64) | nextRandom(state),
        29, 30, 32, 33, 34, 35, 36, 37, 38, 39 => lookup_table.interleaveBits(nextRandom(state), nextRandom(state)),
        else => unreachable,
    };
}

fn fillNonBooleanChallenges(seed: u64) [NUM_LOOKUP_BITS]BN254Scalar {
    var state = seed;
    var r: [NUM_LOOKUP_BITS]BN254Scalar = undefined;
    for (0..NUM_LOOKUP_BITS) |i| {
        state = state *% 6364136223846793005 +% 1442695040888963407;
        const value: u64 = 2 + (state % 1_000_000);
        r[i] = BN254Scalar.fromU64(value);
    }
    return r;
}

fn cleanLine(raw: []const u8) ?[]const u8 {
    const line = std.mem.trim(u8, raw, " \t\r\n");
    if (line.len == 0 or line[0] == '#') return null;
    return line;
}

fn splitFieldsExact(comptime N: usize, line: []const u8) ![N][]const u8 {
    var fields: [N][]const u8 = undefined;
    var parts = std.mem.splitScalar(u8, line, '|');
    var i: usize = 0;
    while (parts.next()) |part| {
        if (i >= N) return error.InvalidFixtureLine;
        fields[i] = part;
        i += 1;
    }
    if (i != N) return error.InvalidFixtureLine;
    return fields;
}

fn hexNibble(c: u8) !u8 {
    return switch (c) {
        '0'...'9' => c - '0',
        'a'...'f' => c - 'a' + 10,
        'A'...'F' => c - 'A' + 10,
        else => error.InvalidHex,
    };
}

fn parseHexBytesExact(comptime N: usize, text: []const u8) ![N]u8 {
    if (text.len != N * 2) return error.InvalidHexLength;
    var out: [N]u8 = undefined;
    for (0..N) |i| {
        const hi = try hexNibble(text[2 * i]);
        const lo = try hexNibble(text[2 * i + 1]);
        out[i] = (hi << 4) | lo;
    }
    return out;
}

fn parseFrHex(text: []const u8) !BN254Scalar {
    const bytes = try parseHexBytesExact(32, text);
    return BN254Scalar.fromBytesBE(&bytes);
}

fn isRequestedTable(table_index: usize) bool {
    inline for (requested_tables) |requested| {
        if (table_index == requested) return true;
    }
    return false;
}

test "new lookup tables materialize required edge cases" {
    // Table 22: Pow2W uses the low 5 shift bits and wraps at 32.
    try expectMaterialize(22, 0, 1);
    try expectMaterialize(22, 1, 2);
    try expectMaterialize(22, 31, 0x8000_0000);
    try expectMaterialize(22, 32, 1);
    try expectMaterialize(22, (@as(u128, 1) << 100) | 7, 0x80);

    // Table 31: MulUNoOverflow is true iff the upper XLEN bits are zero.
    try expectMaterialize(31, 0, 1);
    try expectMaterialize(31, 0xffff_ffff_ffff_ffff, 1);
    try expectMaterialize(31, @as(u128, 1) << 64, 0);
    try expectMaterialize(31, std.math.maxInt(u128), 0);

    // Table 29: VirtualChangeDivisor returns the 64-bit divisor except INT_MIN / -1 -> 1.
    try expectChangeDivisor(123, 7, 7);
    try expectChangeDivisor(i64Bits(-123), i64Bits(-3), i64Bits(-3));
    try expectChangeDivisor(0, i64Bits(-1), i64Bits(-1));
    try expectChangeDivisor(i64Bits(std.math.minInt(i64)), 2, 2);
    try expectChangeDivisor(i64Bits(std.math.minInt(i64)), i64Bits(-1), 1);

    // Table 30: VirtualChangeDivisorW sign-extends the low 32-bit divisor, except i32::MIN / -1 -> 1.
    try expectChangeDivisorW(123, 7, 7);
    try expectChangeDivisorW(i32Bits(-123), i32Bits(-3), signExtendI32Bits(-3));
    try expectChangeDivisorW(0, i32Bits(-1), signExtendI32Bits(-1));
    try expectChangeDivisorW(i32Bits(std.math.minInt(i32)), 2, 2);
    try expectChangeDivisorW(i32Bits(std.math.minInt(i32)), i32Bits(-1), 1);
    try expectChangeDivisorW(0xffff_ffff_8000_0000, 0x1111_2222_ffff_ffff, 1);

    // Tables 32..35: 64-bit XOR then fixed rotate-right. Cover zero, all-ones,
    // high-bit set, and rotations that cross byte/word boundaries.
    const xor_rot_tables = [_]struct { table: usize, rotation: u6 }{
        .{ .table = 32, .rotation = 32 },
        .{ .table = 33, .rotation = 24 },
        .{ .table = 34, .rotation = 16 },
        .{ .table = 35, .rotation = 63 },
    };
    inline for (xor_rot_tables) |entry| {
        try expectXorRotMaterialize(entry.table, entry.rotation, 0, 0);
        try expectXorRotMaterialize(entry.table, entry.rotation, 0xffff_ffff_ffff_ffff, 0);
        try expectXorRotMaterialize(entry.table, entry.rotation, 0x8000_0000_0000_0000, 0);
        try expectXorRotMaterialize(entry.table, entry.rotation, 0x0000_0001_0000_0000, 0);
        try expectXorRotMaterialize(entry.table, entry.rotation, 0x0123_4567_89ab_cdef, 0xfedc_ba98_7654_3210);
    }

    // Tables 36..39: low-32-bit XOR then fixed rotate-right, zero-extended;
    // high 32 bits of both operands must be ignored.
    const xor_rotw_tables = [_]struct { table: usize, rotation: u5 }{
        .{ .table = 36, .rotation = 16 },
        .{ .table = 37, .rotation = 12 },
        .{ .table = 38, .rotation = 8 },
        .{ .table = 39, .rotation = 7 },
    };
    inline for (xor_rotw_tables) |entry| {
        try expectXorRotWMaterialize(entry.table, entry.rotation, 0, 0);
        try expectXorRotWMaterialize(entry.table, entry.rotation, 0x0000_0001, 0);
        try expectXorRotWMaterialize(entry.table, entry.rotation, 0x8000_0000, 0);
        try expectXorRotWMaterialize(entry.table, entry.rotation, 0xffff_ffff, 0);
        try expectXorRotWMaterialize(entry.table, entry.rotation, 0xffff_ffff_0000_0000, 0xffff_ffff_0000_0000);
        try expectXorRotWMaterialize(entry.table, entry.rotation, 0x1234_5678_0000_0001, 0x8765_4321_ffff_fffe);
    }
}

test "new lookup table MLEs match materialization at Boolean points" {
    const Case = struct { table: usize, index: u128 };
    const hand_cases = [_]Case{
        .{ .table = 22, .index = 0 },
        .{ .table = 22, .index = 31 },
        .{ .table = 22, .index = 32 },
        .{ .table = 22, .index = (@as(u128, 1) << 100) | 5 },

        .{ .table = 29, .index = lookup_table.interleaveBits(123, 7) },
        .{ .table = 29, .index = lookup_table.interleaveBits(i64Bits(-123), i64Bits(-3)) },
        .{ .table = 29, .index = lookup_table.interleaveBits(i64Bits(std.math.minInt(i64)), i64Bits(-1)) },

        .{ .table = 30, .index = lookup_table.interleaveBits(123, 7) },
        .{ .table = 30, .index = lookup_table.interleaveBits(i32Bits(-123), i32Bits(-3)) },
        .{ .table = 30, .index = lookup_table.interleaveBits(i32Bits(std.math.minInt(i32)), i32Bits(-1)) },

        .{ .table = 31, .index = 0 },
        .{ .table = 31, .index = 0xffff_ffff_ffff_ffff },
        .{ .table = 31, .index = @as(u128, 1) << 64 },
        .{ .table = 31, .index = std.math.maxInt(u128) },

        .{ .table = 32, .index = lookup_table.interleaveBits(0, 0) },
        .{ .table = 32, .index = lookup_table.interleaveBits(0xffff_ffff_ffff_ffff, 0) },
        .{ .table = 33, .index = lookup_table.interleaveBits(0x0123_4567_89ab_cdef, 0xfedc_ba98_7654_3210) },
        .{ .table = 34, .index = lookup_table.interleaveBits(0x8000_0000_0000_0000, 0) },
        .{ .table = 35, .index = lookup_table.interleaveBits(1, 0) },
        .{ .table = 35, .index = lookup_table.interleaveBits(0x8000_0000_0000_0000, 0) },

        .{ .table = 36, .index = lookup_table.interleaveBits(0, 0) },
        .{ .table = 36, .index = lookup_table.interleaveBits(0xffff_ffff_0000_0000, 0) },
        .{ .table = 37, .index = lookup_table.interleaveBits(0x1234_5678_0000_0001, 0x8765_4321_ffff_fffe) },
        .{ .table = 38, .index = lookup_table.interleaveBits(0xffff_ffff, 0) },
        .{ .table = 39, .index = lookup_table.interleaveBits(0x8000_0000, 0) },
    };

    for (hand_cases) |entry| {
        try expectTableMatches(entry.table, entry.index);
    }

    var state: u64 = 0x1234_5678_9abc_def0;
    inline for (requested_tables) |table_index| {
        for (0..24) |_| {
            try expectTableMatches(table_index, randomIndexForTable(table_index, &state));
        }
    }
}

test "new lookup table non-Boolean MLEs match pinned upstream Jolt fixture" {
    const allocator = std.testing.allocator;
    const fixture_text = try std.Io.Dir.cwd().readFileAlloc(
        std.Io.Threaded.global_single_threaded.io(),
        "testdata/lookup-table-mle/nonboolean_bn254.txt",
        allocator,
        .limited(1024 * 1024),
    );
    defer allocator.free(fixture_text);

    var per_table = [_]usize{0} ** 40;
    var total: usize = 0;
    var lines = std.mem.splitScalar(u8, fixture_text, '\n');
    while (lines.next()) |raw_line| {
        const line = cleanLine(raw_line) orelse continue;
        const fields = try splitFieldsExact(5, line);
        const table_index = try std.fmt.parseInt(usize, fields[0], 10);
        if (!isRequestedTable(table_index)) return error.UnexpectedFixtureTable;

        const seed = try std.fmt.parseInt(u64, fields[3], 10);
        const expected = try parseFrHex(fields[4]);
        var r = fillNonBooleanChallenges(seed);
        const actual = Table.evaluateTableMLE(table_index, &r);
        if (!actual.eql(expected)) {
            std.debug.print("non-Boolean upstream fixture mismatch: table={d} name={s} case={s} seed={d}\n", .{ table_index, fields[1], fields[2], seed });
            return error.LookupTableDifferentialMismatch;
        }

        per_table[table_index] += 1;
        total += 1;
    }

    inline for (requested_tables) |table_index| {
        try std.testing.expect(per_table[table_index] >= 4);
    }
    try std.testing.expectEqual(requested_tables.len * 4, total);
}
