const std = @import("std");
const field = @import("zolt_arith").field;
const BN254Scalar = field.BN254Scalar;
const lookup_table = @import("mod.zig");

const Table = lookup_table.LookupTable(BN254Scalar, 64);

fn bitsFromIndex(index: u128) [128]BN254Scalar {
    var r: [128]BN254Scalar = undefined;
    inline for (0..128) |j| {
        const bit_pos: u7 = @intCast(127 - j);
        const bit = (index >> bit_pos) & 1;
        r[j] = if (bit == 1) BN254Scalar.one() else BN254Scalar.zero();
    }
    return r;
}

fn expectTableMatches(comptime table_index: usize, index: u128) !void {
    var r = bitsFromIndex(index);
    const mle = Table.evaluateTableMLE(table_index, &r);
    const materialized = Table.materializeTableEntry(table_index, index);
    try std.testing.expect(mle.eql(BN254Scalar.fromU64(materialized)));
}

fn expectMaterialize(comptime table_index: usize, index: u128, expected: u64) !void {
    try std.testing.expectEqual(expected, Table.materializeTableEntry(table_index, index));
}

fn expectXorRotMaterialize(comptime table_index: usize, comptime rotation: u6, x: u64, y: u64) !void {
    const index = lookup_table.interleaveBits(x, y);
    const expected = std.math.rotr(u64, x ^ y, rotation);
    try expectMaterialize(table_index, index, expected);
}

fn expectXorRotWMaterialize(comptime table_index: usize, comptime rotation: u5, x: u64, y: u64) !void {
    const index = lookup_table.interleaveBits(x, y);
    const x32: u32 = @truncate(x);
    const y32: u32 = @truncate(y);
    const expected: u64 = std.math.rotr(u32, x32 ^ y32, rotation);
    try expectMaterialize(table_index, index, expected);
}

test "new lookup tables materialize known cases" {
    // Table 22: Pow2W uses the low 5 shift bits.
    try expectMaterialize(22, 0, 1);
    try expectMaterialize(22, 1, 2);
    try expectMaterialize(22, 31, 0x8000_0000);
    try expectMaterialize(22, 32, 1);
    try expectMaterialize(22, (@as(u128, 1) << 100) | 7, 0x80);

    // Table 31: MulUNoOverflow is true iff the upper XLEN address bits are zero.
    try expectMaterialize(31, 0, 1);
    try expectMaterialize(31, 0xffff_ffff_ffff_ffff, 1);
    try expectMaterialize(31, @as(u128, 1) << 64, 0);
    try expectMaterialize(31, std.math.maxInt(u128), 0);

    // Tables 32..35: 64-bit XOR then fixed rotate-right.
    try expectXorRotMaterialize(32, 32, 0x0000_0000_0000_0001, 0);
    try expectXorRotMaterialize(33, 24, 0x0000_0000_0000_0001, 0);
    try expectXorRotMaterialize(34, 16, 0x0000_0000_0000_0001, 0);
    try expectXorRotMaterialize(35, 63, 0x0000_0000_0000_0001, 0);
    try expectXorRotMaterialize(34, 16, 0x8000_0000_0000_0000, 0);
    try expectXorRotMaterialize(32, 32, 0xffff_ffff_ffff_ffff, 0);
    try expectXorRotMaterialize(33, 24, 0x0123_4567_89ab_cdef, 0xfedc_ba98_7654_3210);

    // Tables 36..39: low-32-bit XOR then fixed rotate-right, zero-extended.
    try expectXorRotWMaterialize(36, 16, 0x0000_0001, 0);
    try expectXorRotWMaterialize(37, 12, 0x0000_0001, 0);
    try expectXorRotWMaterialize(38, 8, 0x0000_0001, 0);
    try expectXorRotWMaterialize(39, 7, 0x0000_0001, 0);
    try expectXorRotWMaterialize(36, 16, 0x8000_0000, 0);
    try expectXorRotWMaterialize(38, 8, 0xffff_ffff, 0);
    try expectXorRotWMaterialize(39, 7, 0x1234_5678_0000_0001, 0x8765_4321_ffff_fffe);
    try expectXorRotWMaterialize(36, 16, 0x8000_0000_0000_0000, 0);
}

test "new lookup table MLEs match materialization at Boolean points" {
    const Case = struct { table: usize, index: u128 };
    const cases = [_]Case{
        // Pow2W: zero, edge shift, modulo-32 wrap, and ignored high address bits.
        .{ .table = 22, .index = 0 },
        .{ .table = 22, .index = 31 },
        .{ .table = 22, .index = 32 },
        .{ .table = 22, .index = (@as(u128, 1) << 100) | 5 },

        // MulUNoOverflow: zero, full low word, first high bit, all bits.
        .{ .table = 31, .index = 0 },
        .{ .table = 31, .index = 0xffff_ffff_ffff_ffff },
        .{ .table = 31, .index = @as(u128, 1) << 64 },
        .{ .table = 31, .index = std.math.maxInt(u128) },

        // 64-bit XORROT tables, including boundary-crossing rotations.
        .{ .table = 32, .index = lookup_table.interleaveBits(0, 0) },
        .{ .table = 32, .index = lookup_table.interleaveBits(1, 0) },
        .{ .table = 32, .index = lookup_table.interleaveBits(0xffff_ffff_ffff_ffff, 0) },
        .{ .table = 33, .index = lookup_table.interleaveBits(1, 0) },
        .{ .table = 33, .index = lookup_table.interleaveBits(0x0123_4567_89ab_cdef, 0xfedc_ba98_7654_3210) },
        .{ .table = 34, .index = lookup_table.interleaveBits(0x8000_0000_0000_0000, 0) },
        .{ .table = 34, .index = lookup_table.interleaveBits(0xaaaa_5555_ffff_0000, 0x5555_aaaa_0000_ffff) },
        .{ .table = 35, .index = lookup_table.interleaveBits(1, 0) },
        .{ .table = 35, .index = lookup_table.interleaveBits(0x8000_0000_0000_0000, 0) },

        // 32-bit XORROTW tables, including word-boundary crossings and ignored high bits.
        .{ .table = 36, .index = lookup_table.interleaveBits(0, 0) },
        .{ .table = 36, .index = lookup_table.interleaveBits(1, 0) },
        .{ .table = 36, .index = lookup_table.interleaveBits(0x8000_0000, 0) },
        .{ .table = 36, .index = lookup_table.interleaveBits(0x8000_0000_0000_0000, 0) },
        .{ .table = 37, .index = lookup_table.interleaveBits(1, 0) },
        .{ .table = 37, .index = lookup_table.interleaveBits(0x1234_5678_0000_0001, 0x8765_4321_ffff_fffe) },
        .{ .table = 38, .index = lookup_table.interleaveBits(1, 0) },
        .{ .table = 38, .index = lookup_table.interleaveBits(0xffff_ffff, 0) },
        .{ .table = 39, .index = lookup_table.interleaveBits(1, 0) },
        .{ .table = 39, .index = lookup_table.interleaveBits(0x8000_0000, 0) },
    };

    inline for (cases) |case| {
        try expectTableMatches(case.table, case.index);
    }
}
