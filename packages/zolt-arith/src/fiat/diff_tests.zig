//! Differential tests: validate fiat-crypto Fr/Fp against arkworks-generated
//! test vectors in testdata/zolt-arith-diff/field/{fr,fp}_ops.txt.

const std = @import("std");
const testing = std.testing;
const bn254_fiat = @import("bn254.zig");
const Fr = bn254_fiat.Fr;
const Fp = bn254_fiat.Fp;

fn parseHexBytes32(hex: []const u8) ![32]u8 {
    if (hex.len != 64) return error.InvalidHexLength;
    var out: [32]u8 = undefined;
    _ = try std.fmt.hexToBytes(&out, hex);
    return out;
}

fn cleanLine(raw: []const u8) ?[]const u8 {
    const trimmed = std.mem.trim(u8, raw, " \t\r");
    if (trimmed.len == 0 or trimmed[0] == '#') return null;
    return trimmed;
}

fn splitFields4(line: []const u8) ![4][]const u8 {
    var parts = std.mem.splitScalar(u8, line, '|');
    var fields: [4][]const u8 = undefined;
    for (0..4) |i| {
        fields[i] = std.mem.trim(u8, parts.next() orelse return error.NotEnoughFields, " \t\r");
    }
    return fields;
}

fn runFieldFixture(comptime F: type, fixture_text: []const u8) !void {
    var lines = std.mem.splitScalar(u8, fixture_text, '\n');
    var count: usize = 0;
    while (lines.next()) |raw_line| {
        const line = cleanLine(raw_line) orelse continue;
        const fields = try splitFields4(line);

        const a_bytes = try parseHexBytes32(fields[1]);
        const expected_bytes = try parseHexBytes32(fields[3]);
        const a = F.fromBytesBE(&a_bytes);
        const expected = F.fromBytesBE(&expected_bytes);

        if (std.mem.eql(u8, fields[0], "inv")) {
            const got = a.inverse() orelse return error.InverseOfZero;
            try testing.expect(F.eql(got, expected));
            count += 1;
            continue;
        }

        const b_bytes = try parseHexBytes32(fields[2]);
        const b = F.fromBytesBE(&b_bytes);
        if (std.mem.eql(u8, fields[0], "add")) {
            try testing.expect(F.eql(a.add(b), expected));
        } else if (std.mem.eql(u8, fields[0], "sub")) {
            try testing.expect(F.eql(a.sub(b), expected));
        } else if (std.mem.eql(u8, fields[0], "mul")) {
            try testing.expect(F.eql(a.mul(b), expected));
        } else {
            return error.UnknownFieldOperation;
        }
        count += 1;
    }
    try testing.expect(count > 0);
}

fn readFixture(relative_path: []const u8) ![]u8 {
    // Walk up from packages/zolt-arith/ to the repo root
    const dir = std.fs.cwd().openDir("../../testdata/zolt-arith-diff", .{}) catch
        return error.FixtureNotFound;
    return dir.readFileAlloc(testing.allocator, relative_path, 1024 * 1024);
}

test "fiat Fr: differential fixtures (fr_ops.txt)" {
    const text = try readFixture("field/fr_ops.txt");
    defer testing.allocator.free(text);
    try runFieldFixture(Fr, text);
}

test "fiat Fp: differential fixtures (fp_ops.txt)" {
    const text = try readFixture("field/fp_ops.txt");
    defer testing.allocator.free(text);
    try runFieldFixture(Fp, text);
}
