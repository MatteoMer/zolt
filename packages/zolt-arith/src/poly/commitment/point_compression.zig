//! Point compression/decompression utilities for BN254 G1/G2 points.
//!
//! Arkworks-compatible encoding: x-coordinate with flags in top 2 bits of last byte.
//! Also includes transcript helpers for appending compressed points in the Dory protocol.

const std = @import("std");
const pairing = @import("../../field/pairing.zig");
const field = @import("../../field/mod.zig");

const Fp = field.BN254BaseField;
const Fr = field.BN254Scalar;
const Fp2 = pairing.Fp2;
pub const GT = pairing.GT;
const msm = @import("../../msm/mod.zig");
pub const G1Point = msm.AffinePoint(Fp);
pub const G2Point = pairing.G2Point;

// =============================================================================
// Point Compression (arkworks-compatible)
// =============================================================================

/// Compressed G1 point flags (arkworks format)
/// The flags are stored in the top 2 bits of the last byte of x-coordinate
pub const G1Flags = enum(u8) {
    YIsPositive = 0,
    PointAtInfinity = 0x40, // bit 6
    YIsNegative = 0x80, // bit 7
};

/// Append a GT element to the transcript for Dory protocol.
/// Maps to upstream JoltToDoryTranscript::append_serde which calls
/// transcript.append_bytes(b"dory_serde", &buffer) with serialized bytes (no reversal).
pub fn doryAppendGT(transcript: anytype, gt: GT) void {
    const bytes = gt.toBytes();
    transcript.appendBytes("dory_serde", &bytes);
}

/// Append a G1 point to the transcript for Dory protocol.
/// Upstream: create_evaluation_proof uses transcript.append_serde() for ALL message
/// elements (GT, G1, G2). JoltToDoryTranscript::append_serde maps to "dory_serde".
pub fn doryAppendG1(transcript: anytype, point: G1Point) void {
    const bytes = compressG1(point);
    transcript.appendBytes("dory_serde", &bytes);
}

/// Append a G2 point to the transcript for Dory protocol.
/// Upstream: create_evaluation_proof uses transcript.append_serde() for ALL message
/// elements (GT, G1, G2). JoltToDoryTranscript::append_serde maps to "dory_serde".
pub fn doryAppendG2(transcript: anytype, point: G2Point) void {
    const bytes = compressG2(point);
    transcript.appendBytes("dory_serde", &bytes);
}

/// Compress a G1 point to 32 bytes (arkworks format)
/// Format: x-coordinate with flags in top 2 bits of last byte
pub fn compressG1(point: G1Point) [32]u8 {
    var result: [32]u8 = undefined;

    if (point.infinity) {
        // Point at infinity: all zeros except flag bit
        @memset(&result, 0);
        result[31] = @intFromEnum(G1Flags.PointAtInfinity);
        return result;
    }

    // Serialize x-coordinate (convert from Montgomery form to standard form)
    const x_standard = point.x.fromMontgomery();
    for (0..4) |i| {
        std.mem.writeInt(u64, result[i * 8 ..][0..8], x_standard.limbs[i], .little);
    }

    // Determine if y is "positive" (y <= -y lexicographically)
    const neg_y = point.y.neg();
    const y_is_positive = yIsPositive(point.y, neg_y);

    // Set flag in top 2 bits of last byte
    const flag: u8 = if (y_is_positive) @intFromEnum(G1Flags.YIsPositive) else @intFromEnum(G1Flags.YIsNegative);
    result[31] = (result[31] & 0x3F) | flag;

    return result;
}

/// Decompress a G1 point from 32 bytes (arkworks format)
/// Note: Requires sqrt to be implemented for full decompression.
/// Currently only handles identity point.
pub fn decompressG1(bytes: *const [32]u8) ?G1Point {
    const flag = bytes[31] & 0xC0;

    if (flag == @intFromEnum(G1Flags.PointAtInfinity)) {
        return G1Point.identity();
    }

    // Read x-coordinate (mask off flag bits)
    var x_bytes: [32]u8 = bytes.*;
    x_bytes[31] &= 0x3F;

    var x_limbs: [4]u64 = undefined;
    for (0..4) |i| {
        x_limbs[i] = std.mem.readInt(u64, x_bytes[i * 8 ..][0..8], .little);
    }
    const x_raw = Fp{ .limbs = x_limbs };
    const x = x_raw.toMontgomery();

    // For full decompression we need sqrt implementation.
    // For now, use Tonelli-Shanks to compute y from x^3 + 3.
    const x_cubed = x.square().mul(x);
    const y_squared = x_cubed.add(Fp.fromU64(3)); // b = 3

    // Use Tonelli-Shanks algorithm to compute sqrt
    const y_option = tonelliShanks(y_squared);
    if (y_option == null) return null;

    var y = y_option.?;
    const neg_y = y.neg();

    // Check if we need to negate y based on flag
    const stored_is_positive = (flag == @intFromEnum(G1Flags.YIsPositive));
    const computed_is_positive = yIsPositive(y, neg_y);

    if (stored_is_positive != computed_is_positive) {
        y = neg_y;
    }

    return G1Point.fromCoords(x, y);
}

/// Tonelli-Shanks algorithm for computing square roots in Fp
/// BN254 base field p ≡ 3 (mod 4), so we can use simpler sqrt formula
pub fn tonelliShanks(n: Fp) ?Fp {
    if (n.isZero()) return Fp.zero();

    // For BN254 base field: p ≡ 3 (mod 4)
    // sqrt(n) = n^((p+1)/4)
    // (p+1)/4 for BN254 base field p = 21888242871839275222246405745257275088696311157297823662689037894645226208583
    // = 5472060717959818805561601436314318772174077789324455915672259473661306552146
    const exp = [4]u64{
        0x4f082305b61f3f52,
        0x65e05aa45a1c72a3,
        0x6e14116da0605617,
        0x0c19139cb84c680a,
    };

    // Compute n^((p+1)/4) using square-and-multiply
    var result = Fp.one();
    var base = n;
    for (exp) |limb| {
        var bits = limb;
        for (0..64) |_| {
            if (bits & 1 == 1) {
                result = result.mul(base);
            }
            base = base.square();
            bits >>= 1;
        }
    }

    // Verify: result^2 == n
    if (result.square().eql(n)) {
        return result;
    }

    return null;
}

/// Check if y is "positive" (y <= -y lexicographically)
/// This matches arkworks' definition where smaller = positive
pub fn yIsPositive(y: Fp, neg_y: Fp) bool {
    // Compare as integers (in standard form)
    const y_std = y.fromMontgomery();
    const neg_y_std = neg_y.fromMontgomery();

    // Compare limbs from most significant to least significant
    var i: usize = 4;
    while (i > 0) {
        i -= 1;
        if (y_std.limbs[i] < neg_y_std.limbs[i]) return true;
        if (y_std.limbs[i] > neg_y_std.limbs[i]) return false;
    }
    return true; // Equal means positive
}

/// Compress a G2 point to 64 bytes (arkworks format)
/// Format: x-coordinate (Fp2 = 64 bytes) with flags in top 2 bits of last byte
pub fn compressG2(point: G2Point) [64]u8 {
    var result: [64]u8 = undefined;

    if (point.infinity) {
        // Point at infinity: all zeros except flag bit
        @memset(&result, 0);
        result[63] = @intFromEnum(G1Flags.PointAtInfinity);
        return result;
    }

    // Serialize x.c0 (first 32 bytes)
    const x0_standard = point.x.c0.fromMontgomery();
    for (0..4) |i| {
        std.mem.writeInt(u64, result[i * 8 ..][0..8], x0_standard.limbs[i], .little);
    }

    // Serialize x.c1 (next 32 bytes)
    const x1_standard = point.x.c1.fromMontgomery();
    for (0..4) |i| {
        std.mem.writeInt(u64, result[32 + i * 8 ..][0..8], x1_standard.limbs[i], .little);
    }

    // Determine if y is "positive" (y <= -y lexicographically in Fp2)
    const neg_y = point.y.neg();
    const y_is_positive = fp2IsPositive(point.y, neg_y);

    // Set flag in top 2 bits of last byte
    const flag: u8 = if (y_is_positive) @intFromEnum(G1Flags.YIsPositive) else @intFromEnum(G1Flags.YIsNegative);
    result[63] = (result[63] & 0x3F) | flag;

    return result;
}

/// Decompress a G2 point from 64 bytes (arkworks format)
/// Note: Requires Fp2 sqrt for full decompression.
pub fn decompressG2(bytes: *const [64]u8) ?G2Point {
    const flag = bytes[63] & 0xC0;

    if (flag == @intFromEnum(G1Flags.PointAtInfinity)) {
        return G2Point.identity();
    }

    // Read x.c0 (first 32 bytes)
    var x0_limbs: [4]u64 = undefined;
    for (0..4) |i| {
        x0_limbs[i] = std.mem.readInt(u64, bytes[i * 8 ..][0..8], .little);
    }
    const x0_raw = Fp{ .limbs = x0_limbs };
    const x0 = x0_raw.toMontgomery();

    // Read x.c1 (next 32 bytes, mask off flag bits)
    var x1_bytes: [32]u8 = undefined;
    @memcpy(&x1_bytes, bytes[32..64]);
    x1_bytes[31] &= 0x3F;

    var x1_limbs: [4]u64 = undefined;
    for (0..4) |i| {
        x1_limbs[i] = std.mem.readInt(u64, x1_bytes[i * 8 ..][0..8], .little);
    }
    const x1_raw = Fp{ .limbs = x1_limbs };
    const x1 = x1_raw.toMontgomery();

    const x = Fp2.init(x0, x1);

    // Compute y from G2 curve equation: y^2 = x^3 + b/xi
    // For BN254 twist, b' = 3 / (9 + u) where u^2 = -1
    const y_squared = computeG2YSquared(x);
    const y_option = fp2Sqrt(y_squared);
    if (y_option == null) return null;

    var y = y_option.?;
    const neg_y = y.neg();

    // Check if we need to negate y based on flag
    const stored_is_positive = (flag == @intFromEnum(G1Flags.YIsPositive));
    const computed_is_positive = fp2IsPositive(y, neg_y);

    if (stored_is_positive != computed_is_positive) {
        y = neg_y;
    }

    return G2Point.fromCoords(x, y);
}

/// Compute square root in Fp2 = Fp[u]/(u^2 + 1).
///
/// Given n = a + bu, finds x = c + du such that x^2 = n.
/// Uses the identity c^2 = (a + t)/2 where t = sqrt(a^2 + b^2),
/// with fallback to c^2 = (a - t)/2. Exactly one of these is a QR
/// in Fp when b != 0 (since p ≡ 3 mod 4 for BN254).
pub fn fp2Sqrt(n: Fp2) ?Fp2 {
    if (n.isZero()) return Fp2.zero();

    // If b = 0: n = a is in Fp, so sqrt(n) = sqrt(a) or sqrt(-a)*u
    if (n.c1.isZero()) {
        if (tonelliShanks(n.c0)) |s| return Fp2.init(s, Fp.zero());
        if (tonelliShanks(n.c0.neg())) |s| return Fp2.init(Fp.zero(), s);
        return null;
    }

    // General case: t = sqrt(a^2 + b^2) in Fp
    const norm = n.c0.square().add(n.c1.square());
    const t = tonelliShanks(norm) orelse return null;
    const two_inv = Fp.fromU64(2).inverse() orelse return null;

    // Try c^2 = (a + t) / 2; if not QR, try c^2 = (a - t) / 2
    var x0 = n.c0.add(t).mul(two_inv);
    var c_opt = tonelliShanks(x0);
    if (c_opt == null) {
        x0 = n.c0.sub(t).mul(two_inv);
        c_opt = tonelliShanks(x0);
    }

    // d = b / (2c)
    const c = c_opt orelse return null;
    const d = n.c1.mul(c.add(c).inverse() orelse return null);
    return Fp2.init(c, d);
}

/// Check if Fp2 element a is "positive" compared to b
/// Arkworks compares (c1, c0) lexicographically
fn fp2IsPositive(a: Fp2, b: Fp2) bool {
    // Compare c1 first, then c0
    const a1_std = a.c1.fromMontgomery();
    const b1_std = b.c1.fromMontgomery();

    // Compare c1 limbs
    var i: usize = 4;
    while (i > 0) {
        i -= 1;
        if (a1_std.limbs[i] < b1_std.limbs[i]) return true;
        if (a1_std.limbs[i] > b1_std.limbs[i]) return false;
    }

    // c1 is equal, compare c0
    const a0_std = a.c0.fromMontgomery();
    const b0_std = b.c0.fromMontgomery();

    i = 4;
    while (i > 0) {
        i -= 1;
        if (a0_std.limbs[i] < b0_std.limbs[i]) return true;
        if (a0_std.limbs[i] > b0_std.limbs[i]) return false;
    }
    return true; // Equal means positive
}

/// Compute y^2 for G2 curve equation
/// y^2 = x^3 + b' where b' = 3 / (9 + u)
fn computeG2YSquared(x: Fp2) Fp2 {
    const x_cubed = x.mul(x).mul(x);
    // b' = 3 / (9 + u) for BN254 twist
    // This is a constant that should be precomputed
    // For now, use the actual curve equation
    const b_twist = getG2BTwist();
    return x_cubed.add(b_twist);
}

/// Get the b coefficient for G2 curve (3 / xi where xi = 9 + u)
fn getG2BTwist() Fp2 {
    // For BN254 D-type twist: b' = b / xi = 3 / (9 + u)
    // xi = 9 + u, so 1/xi = (9 - u) / (81 + 1) = (9 - u) / 82
    // b' = 3 * (9 - u) / 82 = (27 - 3u) / 82

    // Precomputed value for b' = 3 / (9 + u)
    // This matches the twist coefficient in BN254
    const nine = Fp.fromU64(9);
    const xi = Fp2.init(nine, Fp.one()); // 9 + u
    const xi_inv = xi.inverse() orelse Fp2.one();
    const three = Fp2.init(Fp.fromU64(3), Fp.zero());
    return three.mul(xi_inv);
}

// =============================================================================
// Tests
// =============================================================================

test "g1 point compression roundtrip" {
    // Test with generator point
    const g1_gen = G1Point.generator();
    const compressed = compressG1(g1_gen);
    const decompressed = decompressG1(&compressed);

    try std.testing.expect(decompressed != null);
    try std.testing.expect(decompressed.?.x.eql(g1_gen.x));
    try std.testing.expect(decompressed.?.y.eql(g1_gen.y));
}

test "g1 point compression identity" {
    const identity = G1Point.identity();
    const compressed = compressG1(identity);
    const decompressed = decompressG1(&compressed);

    try std.testing.expect(decompressed != null);
    try std.testing.expect(decompressed.?.infinity);
}

test "g2 point compression roundtrip" {
    // Identity roundtrip
    const identity = G2Point.identity();
    const compressed_id = compressG2(identity);
    const decompressed_id = decompressG2(&compressed_id);
    try std.testing.expect(decompressed_id != null);
    try std.testing.expect(decompressed_id.?.infinity);

    // Generator roundtrip
    const g2_gen = G2Point.generator();
    const compressed = compressG2(g2_gen);
    const decompressed = decompressG2(&compressed);
    try std.testing.expect(decompressed != null);
    try std.testing.expect(!decompressed.?.infinity);
    try std.testing.expect(decompressed.?.x.eql(g2_gen.x));
    try std.testing.expect(decompressed.?.y.eql(g2_gen.y));
}

test "g2 point compression identity" {
    const identity = G2Point.identity();
    const compressed = compressG2(identity);
    const decompressed = decompressG2(&compressed);

    try std.testing.expect(decompressed != null);
    try std.testing.expect(decompressed.?.infinity);
}

test "g2 compressed bytes for arkworks validation" {
    // This test outputs compressed G2 points that can be validated by arkworks
    const g2_gen = G2Point.generator();

    // Compress the generator
    const gen_compressed = compressG2(g2_gen);
    std.debug.print("\nG2_GENERATOR_COMPRESSED: ", .{});
    for (gen_compressed) |b| {
        std.debug.print("{x:0>2}", .{b});
    }
    std.debug.print("\n", .{});

    // Compress [2]G2
    const two_g2 = g2_gen.double();
    const two_compressed = compressG2(two_g2);
    std.debug.print("G2_DOUBLE_COMPRESSED: ", .{});
    for (two_compressed) |b| {
        std.debug.print("{x:0>2}", .{b});
    }
    std.debug.print("\n", .{});

    // Compress [42]G2
    const scalar_42 = Fr.fromU64(42);
    const g2_42 = g2_gen.scalarMul(scalar_42);
    const compressed_42 = compressG2(g2_42);
    std.debug.print("G2_42_COMPRESSED: ", .{});
    for (compressed_42) |b| {
        std.debug.print("{x:0>2}", .{b});
    }
    std.debug.print("\n", .{});

    // Also write to a file for the Rust test to read
    const io: std.Io = std.Io.Threaded.global_single_threaded.io();
    const file = std.Io.Dir.cwd().createFile(io, "/tmp/zolt_g2_test_points.bin", .{}) catch |err| {
        std.debug.print("Could not create file: {}\n", .{err});
        return;
    };
    defer file.close(io);

    // Write 3 compressed G2 points (64 bytes each = 192 bytes total)
    file.writeStreamingAll(io, &gen_compressed) catch return;
    file.writeStreamingAll(io, &two_compressed) catch return;
    file.writeStreamingAll(io, &compressed_42) catch return;
    std.debug.print("Wrote 3 compressed G2 points to /tmp/zolt_g2_test_points.bin\n", .{});
}

fn fpToBytesLE(value: Fp) [32]u8 {
    const standard = value.fromMontgomery();
    var bytes: [32]u8 = undefined;
    inline for (0..4) |i| {
        std.mem.writeInt(u64, bytes[i * 8 ..][0..8], standard.limbs[i], .little);
    }
    return bytes;
}

fn fpFromBytesLE(bytes: *const [32]u8) Fp {
    var limbs: [4]u64 = undefined;
    inline for (0..4) |i| {
        limbs[i] = std.mem.readInt(u64, bytes[i * 8 ..][0..8], .little);
    }
    const raw = Fp{ .limbs = limbs };
    return raw.toMontgomery();
}

test "g1 point compression fixture vectors" {
    const testdata = @import("../../testdata.zig");
    const fixture_text = @embedFile("../../testdata/point_compression/g1_vectors.txt");
    var lines = std.mem.splitScalar(u8, fixture_text, '\n');
    var case_count: usize = 0;

    while (lines.next()) |raw_line| {
        const line = testdata.cleanLine(raw_line) orelse continue;
        const fields = try testdata.splitFieldsExact(4, line, '|');

        const expected_compressed = try testdata.parseHexBytesExact(32, fields[3]);

        if (fields[1].len == 0) {
            const identity = G1Point.identity();
            const actual_compressed = compressG1(identity);
            try std.testing.expectEqualSlices(u8, &expected_compressed, &actual_compressed);
        } else {
            const x_bytes = try testdata.parseHexBytesExact(32, fields[1]);
            const y_bytes = try testdata.parseHexBytesExact(32, fields[2]);
            const point = G1Point{ .x = fpFromBytesLE(&x_bytes), .y = fpFromBytesLE(&y_bytes), .infinity = false };

            const actual_compressed = compressG1(point);
            try std.testing.expectEqualSlices(u8, &expected_compressed, &actual_compressed);

            const decompressed = decompressG1(&actual_compressed);
            try std.testing.expect(decompressed != null);
            try std.testing.expectEqualSlices(u8, &x_bytes, &fpToBytesLE(decompressed.?.x));
            try std.testing.expectEqualSlices(u8, &y_bytes, &fpToBytesLE(decompressed.?.y));
        }
        case_count += 1;
    }
    try std.testing.expect(case_count >= 6);
}

test "g2 point compression fixture vectors" {
    const testdata = @import("../../testdata.zig");
    const fixture_text = @embedFile("../../testdata/point_compression/g2_vectors.txt");
    var lines = std.mem.splitScalar(u8, fixture_text, '\n');
    var case_count: usize = 0;

    while (lines.next()) |raw_line| {
        const line = testdata.cleanLine(raw_line) orelse continue;
        const fields = try testdata.splitFieldsExact(6, line, '|');

        if (fields[1].len == 0) {
            const identity = pairing.G2Point.identity();
            const compressed = compressG2(identity);
            const decompressed = decompressG2(&compressed);
            try std.testing.expect(decompressed != null);
            try std.testing.expect(decompressed.?.infinity);
        } else {
            const x_c0 = try testdata.parseHexBytesExact(32, fields[1]);
            const x_c1 = try testdata.parseHexBytesExact(32, fields[2]);
            const y_c0 = try testdata.parseHexBytesExact(32, fields[3]);
            const y_c1 = try testdata.parseHexBytesExact(32, fields[4]);
            const point = pairing.G2Point{
                .x = Fp2.init(fpFromBytesLE(&x_c0), fpFromBytesLE(&x_c1)),
                .y = Fp2.init(fpFromBytesLE(&y_c0), fpFromBytesLE(&y_c1)),
                .infinity = false,
            };

            // Verify full compress → decompress roundtrip
            const compressed = compressG2(point);
            const decompressed = decompressG2(&compressed);
            try std.testing.expect(decompressed != null);
            try std.testing.expect(!decompressed.?.infinity);
            try std.testing.expectEqualSlices(u8, &x_c0, &fpToBytesLE(decompressed.?.x.c0));
            try std.testing.expectEqualSlices(u8, &x_c1, &fpToBytesLE(decompressed.?.x.c1));
            try std.testing.expectEqualSlices(u8, &y_c0, &fpToBytesLE(decompressed.?.y.c0));
            try std.testing.expectEqualSlices(u8, &y_c1, &fpToBytesLE(decompressed.?.y.c1));
        }
        case_count += 1;
    }
    try std.testing.expect(case_count >= 6);
}

test "fp2Sqrt correctness" {
    const g2_gen = G2Point.generator();

    // sqrt(y²) should give back ±y for the G2 generator
    const y_squared = g2_gen.y.square();
    const sqrt_result = fp2Sqrt(y_squared);
    try std.testing.expect(sqrt_result != null);
    const s = sqrt_result.?;
    try std.testing.expect(s.square().eql(y_squared));
    try std.testing.expect(s.eql(g2_gen.y) or s.eql(g2_gen.y.neg()));

    // sqrt(0) = 0
    const zero_sqrt = fp2Sqrt(Fp2.zero());
    try std.testing.expect(zero_sqrt != null);
    try std.testing.expect(zero_sqrt.?.isZero());

    // Pure real: sqrt(4) = 2
    const four = Fp2.init(Fp.fromU64(4), Fp.zero());
    const sqrt_four = fp2Sqrt(four);
    try std.testing.expect(sqrt_four != null);
    try std.testing.expect(sqrt_four.?.square().eql(four));

    // Pure real, non-QR: sqrt(-1) should give purely imaginary result
    const neg_one = Fp2.init(Fp.one().neg(), Fp.zero());
    const sqrt_neg_one = fp2Sqrt(neg_one);
    try std.testing.expect(sqrt_neg_one != null);
    try std.testing.expect(sqrt_neg_one.?.square().eql(neg_one));

    // Multiple G2 points: verify y² roundtrip for [2]G, [42]G
    const two_g = g2_gen.scalarMulU64(2);
    const ys2 = two_g.y.square();
    const s2 = fp2Sqrt(ys2);
    try std.testing.expect(s2 != null);
    try std.testing.expect(s2.?.square().eql(ys2));

    const g42 = g2_gen.scalarMulU64(42);
    const ys42 = g42.y.square();
    const s42 = fp2Sqrt(ys42);
    try std.testing.expect(s42 != null);
    try std.testing.expect(s42.?.square().eql(ys42));
}
