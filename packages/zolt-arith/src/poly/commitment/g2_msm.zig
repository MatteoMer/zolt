//! G2 Multi-Scalar Multiplication (MSM) for BN254.
//!
//! Delegates to the generic MSM(F, Fp2) which uses XYZZ bucket
//! coordinates and chunk-based parallelism.

const std = @import("std");
const pairing = @import("../../field/pairing.zig");
const field = @import("../../field/mod.zig");
const msm_mod = @import("../../msm/mod.zig");
const extensions = @import("../../field/extensions.zig");
const ThreadPool = @import("zolt_pool").ThreadPool;

const Fp = field.BN254BaseField;
const Fp2 = extensions.Fp2;
pub const G2Point = pairing.G2Point;
const AffineG2 = msm_mod.AffinePoint(Fp2);

/// Public wrapper for G2 MSM benchmarking.
pub fn msmG2Bench(comptime F: type, g2_vec: []const G2Point, scalars: []const F, tp: ?*ThreadPool) G2Point {
    return msmG2(F, g2_vec, scalars, tp);
}

/// MSM for G2 points, delegating to the generic XYZZ-based Pippenger.
pub fn msmG2(comptime F: type, g2_vec: []const G2Point, scalars: []const F, tp: ?*ThreadPool) G2Point {
    const n = @min(g2_vec.len, scalars.len);
    if (n == 0) return G2Point.identity();
    const bases = castG2Slice(g2_vec[0..n]);
    const result = msm_mod.MSM(F, Fp2).computeWithPool(bases, scalars[0..n], tp);
    return toG2Point(result);
}

// ---------------------------------------------------------------------------
// Type bridging: G2Point <-> AffinePoint(Fp2) via @ptrCast
// Both are { x: Fp2, y: Fp2, infinity: bool } — identical in-memory layout.
// ---------------------------------------------------------------------------

fn castG2Slice(g2s: []const G2Point) []const AffineG2 {
    comptime {
        if (@sizeOf(G2Point) != @sizeOf(AffineG2))
            @compileError("G2Point / AffinePoint(Fp2) size mismatch");
        if (@alignOf(G2Point) != @alignOf(AffineG2))
            @compileError("G2Point / AffinePoint(Fp2) alignment mismatch");
        if (@offsetOf(G2Point, "x") != @offsetOf(AffineG2, "x"))
            @compileError("field offset mismatch: x");
        if (@offsetOf(G2Point, "y") != @offsetOf(AffineG2, "y"))
            @compileError("field offset mismatch: y");
        if (@offsetOf(G2Point, "infinity") != @offsetOf(AffineG2, "infinity"))
            @compileError("field offset mismatch: infinity");
    }
    return @ptrCast(g2s);
}

fn toG2Point(p: AffineG2) G2Point {
    if (p.infinity) return G2Point.identity();
    return .{ .x = p.x, .y = p.y, .infinity = false };
}

fn fpToBytesLE(value: Fp) [32]u8 {
    const standard = value.fromMontgomery();
    var bytes: [32]u8 = undefined;
    inline for (0..4) |i| {
        std.mem.writeInt(u64, bytes[i * 8 ..][0..8], standard.limbs[i], .little);
    }
    return bytes;
}

test "g2 msm fr fixture vectors" {
    const Fr = field.BN254Scalar;
    const testdata = @import("../../testdata.zig");
    const fixture_text = @embedFile("../../testdata/msm/g2_fr_vectors.txt");
    var lines = std.mem.splitScalar(u8, fixture_text, '\n');
    var case_count: usize = 0;

    while (lines.next()) |raw_line| {
        const line = testdata.cleanLine(raw_line) orelse continue;
        const fields_split = try testdata.splitFieldsExact(7, line, '|');

        // Parse comma-separated BE hex scalars
        var scalar_buf: [64]Fr = undefined;
        var n: usize = 0;
        var csv = std.mem.splitScalar(u8, fields_split[1], ',');
        while (csv.next()) |token| {
            const trimmed = std.mem.trim(u8, token, " \t\r");
            if (trimmed.len == 0) continue;
            const bytes = try testdata.parseHexBytesExact(32, trimmed);
            scalar_buf[n] = Fr.fromBytesBE(&bytes);
            n += 1;
        }
        const scalars = scalar_buf[0..n];

        // Generate G2 bases: G2, 2*G2, 4*G2, ... (successive doublings)
        var base_buf: [64]G2Point = undefined;
        const gen = G2Point.generator();
        var proj = pairing.G2Projective.fromAffine(gen);
        for (0..n) |i| {
            base_buf[i] = proj.toAffine();
            proj = proj.double();
        }
        const bases = base_buf[0..n];

        const actual = msmG2Bench(Fr, bases, scalars, null);
        const expected_infinity = try testdata.parseDecimal(u8, fields_split[2]);
        try std.testing.expectEqual(expected_infinity == 1, actual.infinity);
        if (!actual.infinity) {
            const expected_x_c0 = try testdata.parseHexBytesExact(32, fields_split[3]);
            const expected_x_c1 = try testdata.parseHexBytesExact(32, fields_split[4]);
            const expected_y_c0 = try testdata.parseHexBytesExact(32, fields_split[5]);
            const expected_y_c1 = try testdata.parseHexBytesExact(32, fields_split[6]);
            try std.testing.expectEqualSlices(u8, &expected_x_c0, &fpToBytesLE(actual.x.c0));
            try std.testing.expectEqualSlices(u8, &expected_x_c1, &fpToBytesLE(actual.x.c1));
            try std.testing.expectEqualSlices(u8, &expected_y_c0, &fpToBytesLE(actual.y.c0));
            try std.testing.expectEqualSlices(u8, &expected_y_c1, &fpToBytesLE(actual.y.c1));
        }
        case_count += 1;
    }
    try std.testing.expect(case_count >= 3);
}
