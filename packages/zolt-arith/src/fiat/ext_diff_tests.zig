//! Differential tests for fiat-crypto backed extension fields (Fp2, Fp6, Fp12)
//! against arkworks-generated test vectors.

const std = @import("std");
const testing = std.testing;
const fiat_ext = @import("extensions.zig");
const FiatFp = fiat_ext.FiatFp;
const Fp2 = fiat_ext.Fp2;
const Fp6 = fiat_ext.Fp6;
const Fp12 = fiat_ext.Fp12;

// -- Parsing helpers --

fn parseHexBytes(comptime N: usize, hex: []const u8) ![N]u8 {
    if (hex.len != N * 2) return error.InvalidHexLength;
    var out: [N]u8 = undefined;
    _ = try std.fmt.hexToBytes(&out, hex);
    return out;
}

fn fpFromBytesLE(bytes: *const [32]u8) FiatFp {
    var limbs: [4]u64 = undefined;
    inline for (0..4) |i| {
        limbs[i] = std.mem.readInt(u64, bytes[i * 8 ..][0..8], .little);
    }
    return (FiatFp{ .limbs = limbs }).toMontgomery();
}

fn parseFp2HexLE(text: []const u8) !Fp2 {
    const bytes = try parseHexBytes(64, text);
    return Fp2.init(fpFromBytesLE(bytes[0..32]), fpFromBytesLE(bytes[32..64]));
}

fn parseFp6HexLE(text: []const u8) !Fp6 {
    const bytes = try parseHexBytes(192, text);
    return Fp6{
        .c0 = Fp2.init(fpFromBytesLE(bytes[0..32]), fpFromBytesLE(bytes[32..64])),
        .c1 = Fp2.init(fpFromBytesLE(bytes[64..96]), fpFromBytesLE(bytes[96..128])),
        .c2 = Fp2.init(fpFromBytesLE(bytes[128..160]), fpFromBytesLE(bytes[160..192])),
    };
}

fn parseFp12HexLE(text: []const u8) !Fp12 {
    const bytes = try parseHexBytes(384, text);
    // Fp12 = (Fp6, Fp6), each Fp6 = 3 Fp2, each Fp2 = 2 Fp
    return Fp12{
        .c0 = Fp6{
            .c0 = Fp2.init(fpFromBytesLE(bytes[0..32]), fpFromBytesLE(bytes[32..64])),
            .c1 = Fp2.init(fpFromBytesLE(bytes[64..96]), fpFromBytesLE(bytes[96..128])),
            .c2 = Fp2.init(fpFromBytesLE(bytes[128..160]), fpFromBytesLE(bytes[160..192])),
        },
        .c1 = Fp6{
            .c0 = Fp2.init(fpFromBytesLE(bytes[192..224]), fpFromBytesLE(bytes[224..256])),
            .c1 = Fp2.init(fpFromBytesLE(bytes[256..288]), fpFromBytesLE(bytes[288..320])),
            .c2 = Fp2.init(fpFromBytesLE(bytes[320..352]), fpFromBytesLE(bytes[352..384])),
        },
    };
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

fn readFixture(relative_path: []const u8) ![]u8 {
    const dir = std.fs.cwd().openDir("../../testdata/zolt-arith-diff", .{}) catch
        return error.FixtureNotFound;
    return dir.readFileAlloc(testing.allocator, relative_path, 4 * 1024 * 1024);
}

// -- Fp2 tests --

test "fiat Fp2: differential fixtures" {
    const text = try readFixture("extensions/fp2_ops.txt");
    defer testing.allocator.free(text);
    var lines = std.mem.splitScalar(u8, text, '\n');
    var count: usize = 0;

    while (lines.next()) |raw_line| {
        const line = cleanLine(raw_line) orelse continue;
        const fields = try splitFields4(line);
        const op = fields[0];
        const a = try parseFp2HexLE(fields[1]);
        const expected = try parseFp2HexLE(fields[3]);

        if (std.mem.eql(u8, op, "add")) {
            try testing.expect(Fp2.eql(a.add(try parseFp2HexLE(fields[2])), expected));
        } else if (std.mem.eql(u8, op, "sub")) {
            try testing.expect(Fp2.eql(a.sub(try parseFp2HexLE(fields[2])), expected));
        } else if (std.mem.eql(u8, op, "mul")) {
            try testing.expect(Fp2.eql(a.mul(try parseFp2HexLE(fields[2])), expected));
        } else if (std.mem.eql(u8, op, "square")) {
            try testing.expect(Fp2.eql(a.square(), expected));
        } else if (std.mem.eql(u8, op, "inv")) {
            try testing.expect(Fp2.eql(a.inverse().?, expected));
        } else if (std.mem.eql(u8, op, "conjugate")) {
            try testing.expect(Fp2.eql(a.conjugate(), expected));
        } else {
            return error.UnknownOp;
        }
        count += 1;
    }
    try testing.expect(count >= 40);
}

// -- Fp6 tests --

test "fiat Fp6: differential fixtures" {
    const text = try readFixture("extensions/fp6_ops.txt");
    defer testing.allocator.free(text);
    var lines = std.mem.splitScalar(u8, text, '\n');
    var count: usize = 0;

    while (lines.next()) |raw_line| {
        const line = cleanLine(raw_line) orelse continue;
        const fields = try splitFields4(line);
        const op = fields[0];
        const a = try parseFp6HexLE(fields[1]);
        const expected = try parseFp6HexLE(fields[3]);

        if (std.mem.eql(u8, op, "add")) {
            try testing.expect(Fp6.eql(a.add(try parseFp6HexLE(fields[2])), expected));
        } else if (std.mem.eql(u8, op, "sub")) {
            try testing.expect(Fp6.eql(a.sub(try parseFp6HexLE(fields[2])), expected));
        } else if (std.mem.eql(u8, op, "mul")) {
            try testing.expect(Fp6.eql(a.mul(try parseFp6HexLE(fields[2])), expected));
        } else if (std.mem.eql(u8, op, "square")) {
            try testing.expect(Fp6.eql(a.square(), expected));
        } else if (std.mem.eql(u8, op, "inv")) {
            try testing.expect(Fp6.eql(a.inverse().?, expected));
        } else {
            return error.UnknownOp;
        }
        count += 1;
    }
    try testing.expect(count >= 30);
}

// -- Fp12 tests --

test "fiat Fp12: differential fixtures (core ops)" {
    const text = try readFixture("extensions/fp12_ops.txt");
    defer testing.allocator.free(text);
    var lines = std.mem.splitScalar(u8, text, '\n');
    var count: usize = 0;

    while (lines.next()) |raw_line| {
        const line = cleanLine(raw_line) orelse continue;
        const fields = try splitFields4(line);
        const op = fields[0];
        const a = try parseFp12HexLE(fields[1]);
        const expected = try parseFp12HexLE(fields[3]);

        if (std.mem.eql(u8, op, "add")) {
            try testing.expect(Fp12.eql(a.add(try parseFp12HexLE(fields[2])), expected));
        } else if (std.mem.eql(u8, op, "sub")) {
            try testing.expect(Fp12.eql(a.sub(try parseFp12HexLE(fields[2])), expected));
        } else if (std.mem.eql(u8, op, "mul")) {
            try testing.expect(Fp12.eql(a.mul(try parseFp12HexLE(fields[2])), expected));
        } else if (std.mem.eql(u8, op, "square")) {
            try testing.expect(Fp12.eql(a.square(), expected));
        } else if (std.mem.eql(u8, op, "inv")) {
            try testing.expect(Fp12.eql(a.inverse().?, expected));
        } else if (std.mem.eql(u8, op, "conjugate")) {
            try testing.expect(Fp12.eql(a.conjugate(), expected));
        } else if (std.mem.eql(u8, op, "cyclotomic_square")) {
            try testing.expect(Fp12.eql(a.cyclotomicSquare(), expected));
        } else if (std.mem.eql(u8, op, "frobenius") or
            std.mem.eql(u8, op, "frobenius2") or
            std.mem.eql(u8, op, "frobenius3"))
        {
            // Frobenius requires precomputed coefficients — tested in pairing step
            continue;
        } else {
            return error.UnknownOp;
        }
        count += 1;
    }
    try testing.expect(count >= 30);
}
