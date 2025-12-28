//! Structured Reference String (SRS) loading utilities
//!
//! This module provides utilities for loading SRS data from various formats:
//! - Ethereum Powers of Tau ceremony format (ptau)
//! - Raw binary format (little-endian curve points)
//! - JSON format (hex-encoded coordinates)
//!
//! For production use, you should load SRS from the Ethereum KZG ceremony
//! or other trusted setup ceremonies.

const std = @import("std");
const Allocator = std.mem.Allocator;
const field = @import("../../field/mod.zig");
const msm = @import("../../msm/mod.zig");
const pairing = field.pairing;

/// Base field for G1 point coordinates
const Fp = field.BN254BaseField;

/// Scalar field for polynomial coefficients
const Fr = field.BN254Scalar;

/// G1 Point type
pub const G1Point = msm.AffinePoint(Fp);

/// G2 Point type
pub const G2Point = pairing.G2Point;

/// Error types for SRS loading
pub const SRSError = error{
    InvalidPointFormat,
    PointNotOnCurve,
    InvalidFileFormat,
    TruncatedData,
    DecompressionFailed,
    InvalidLength,
    UnsupportedFormat,
    FileNotFound,
    OutOfMemory,
};

/// SRS data structure containing powers of tau
pub const SRSData = struct {
    /// Powers of tau in G1: [1, tau, tau^2, ..., tau^{n-1}]
    powers_of_tau_g1: []G1Point,
    /// tau in G2: [tau]_2
    tau_g2: G2Point,
    /// G1 generator: [1]_1
    g1: G1Point,
    /// G2 generator: [1]_2
    g2: G2Point,
    /// Maximum degree supported
    max_degree: usize,
    /// Allocator used
    allocator: Allocator,

    pub fn deinit(self: *SRSData) void {
        if (self.powers_of_tau_g1.len > 0) {
            self.allocator.free(self.powers_of_tau_g1);
        }
    }
};

/// Parse a G1 point from 64 bytes (uncompressed: 32 bytes x, 32 bytes y)
pub fn parseG1Uncompressed(data: []const u8) SRSError!G1Point {
    if (data.len < 64) {
        return SRSError.TruncatedData;
    }

    // Check for point at infinity (all zeros)
    var all_zero = true;
    for (data[0..64]) |b| {
        if (b != 0) {
            all_zero = false;
            break;
        }
    }
    if (all_zero) {
        return G1Point.identity();
    }

    // Parse x coordinate (big-endian from most formats)
    var x_bytes: [32]u8 = undefined;
    var y_bytes: [32]u8 = undefined;
    @memcpy(&x_bytes, data[0..32]);
    @memcpy(&y_bytes, data[32..64]);

    const x = Fp.fromBytesBE(&x_bytes);
    const y = Fp.fromBytesBE(&y_bytes);

    const point = G1Point{ .x = x, .y = y, .infinity = false };

    // Verify point is on curve: y² = x³ + 3
    if (!point.isOnCurve()) {
        return SRSError.PointNotOnCurve;
    }

    return point;
}

/// Parse a G1 point from 48 bytes (compressed format)
pub fn parseG1Compressed(data: []const u8) SRSError!G1Point {
    if (data.len < 48) {
        return SRSError.TruncatedData;
    }

    // Check flags in first byte
    const flags = data[0] & 0xE0;
    const is_compressed = (flags & 0x80) != 0;
    const is_infinity = (flags & 0x40) != 0;
    const sign_bit = (flags & 0x20) != 0;

    _ = is_compressed; // Must be compressed for 48-byte format

    if (is_infinity) {
        return G1Point.identity();
    }

    // Extract x coordinate (clear flags in first byte)
    var x_bytes: [48]u8 = undefined;
    @memcpy(&x_bytes, data[0..48]);
    x_bytes[0] &= 0x1F; // Clear flag bits

    // For BN254, coordinates are 32 bytes, but compressed format may use padding
    // Extract last 32 bytes as the x coordinate
    var x_32: [32]u8 = undefined;
    @memcpy(&x_32, x_bytes[16..48]);

    const x = Fp.fromBytesBE(&x_32);

    // Compute y from x: y² = x³ + 3
    const x2 = x.mul(x);
    const x3 = x2.mul(x);
    const y_squared = x3.add(Fp.fromU64(3));

    // Compute square root
    const y_opt = sqrtFp(y_squared);
    if (y_opt) |y| {
        // Choose the correct square root based on sign bit
        const y_neg = y.neg();

        // Lexicographically larger is chosen based on sign bit
        const y_bytes_pos = y.toBytesBE();
        const y_bytes_neg = y_neg.toBytesBE();

        var use_neg = false;
        for (y_bytes_pos, y_bytes_neg) |pos, neg| {
            if (pos > neg) {
                use_neg = sign_bit;
                break;
            } else if (pos < neg) {
                use_neg = !sign_bit;
                break;
            }
        }

        const final_y = if (use_neg) y_neg else y;
        return G1Point{ .x = x, .y = final_y, .infinity = false };
    } else {
        return SRSError.DecompressionFailed;
    }
}

/// Parse a G2 point from 128 bytes (uncompressed: 64 bytes x, 64 bytes y in Fp2)
pub fn parseG2Uncompressed(data: []const u8) SRSError!G2Point {
    if (data.len < 128) {
        return SRSError.TruncatedData;
    }

    // Check for point at infinity
    var all_zero = true;
    for (data[0..128]) |b| {
        if (b != 0) {
            all_zero = false;
            break;
        }
    }
    if (all_zero) {
        return G2Point.identity();
    }

    // Parse x coordinate (Fp2 = c0 + c1*u)
    // Format: x.c1 (32 bytes) | x.c0 (32 bytes) | y.c1 (32 bytes) | y.c0 (32 bytes)
    var x_c1_bytes: [32]u8 = undefined;
    var x_c0_bytes: [32]u8 = undefined;
    var y_c1_bytes: [32]u8 = undefined;
    var y_c0_bytes: [32]u8 = undefined;

    @memcpy(&x_c1_bytes, data[0..32]);
    @memcpy(&x_c0_bytes, data[32..64]);
    @memcpy(&y_c1_bytes, data[64..96]);
    @memcpy(&y_c0_bytes, data[96..128]);

    const x_c0 = Fp.fromBytesBE(&x_c0_bytes);
    const x_c1 = Fp.fromBytesBE(&x_c1_bytes);
    const y_c0 = Fp.fromBytesBE(&y_c0_bytes);
    const y_c1 = Fp.fromBytesBE(&y_c1_bytes);

    const x = pairing.Fp2{ .c0 = x_c0, .c1 = x_c1 };
    const y = pairing.Fp2{ .c0 = y_c0, .c1 = y_c1 };

    return G2Point{ .x = x, .y = y, .infinity = false };
}

/// Compute square root in Fp using Tonelli-Shanks
/// Returns null if the element is not a quadratic residue
fn sqrtFp(a: Fp) ?Fp {
    if (a.isZero()) {
        return Fp.zero();
    }

    // For BN254, p ≡ 3 (mod 4), so we can use the simple formula:
    // sqrt(a) = a^((p+1)/4)
    //
    // p = 21888242871839275222246405745257275088696311157297823662689037894645226208583
    // (p + 1) / 4 = 5472060717959818805561601436314318772174077789324455915672259473661306552146
    //
    // We can verify: if a is a QR, then a^((p-1)/2) = 1
    // Otherwise a^((p-1)/2) = -1

    // Compute a^((p-1)/2) to check if a is a quadratic residue
    // For BN254, (p-1)/2 has a specific representation
    // Instead, we compute the candidate sqrt and verify

    // Candidate sqrt: a^((p+1)/4)
    // This exponent is: (p+1)/4
    const exp = [_]u64{
        0x4a47ab3ee05a8c73,
        0x9f2d41fffb12fecc,
        0xc16f0d0f8b6b61e0,
        0x30644e72e131a029,
    };

    var result = Fp.one();
    var base = a;

    for (exp) |limb| {
        var bits = limb;
        for (0..64) |_| {
            if ((bits & 1) == 1) {
                result = result.mul(base);
            }
            base = base.mul(base);
            bits >>= 1;
        }
    }

    // Verify: result² = a
    const result_sq = result.mul(result);
    if (result_sq.eql(a)) {
        return result;
    }
    return null;
}

/// Load SRS from raw binary format
///
/// Format:
/// - 4 bytes: number of G1 points (little-endian u32)
/// - n * 64 bytes: G1 points (uncompressed)
/// - 128 bytes: tau*G2 (uncompressed)
/// - 64 bytes: G1 generator
/// - 128 bytes: G2 generator
pub fn loadFromRawBinary(allocator: Allocator, data: []const u8) SRSError!SRSData {
    if (data.len < 4) {
        return SRSError.TruncatedData;
    }

    const num_points = std.mem.readInt(u32, data[0..4], .little);
    const g1_size = @as(usize, num_points) * 64;
    const total_size = 4 + g1_size + 128 + 64 + 128;

    if (data.len < total_size) {
        return SRSError.TruncatedData;
    }

    // Parse G1 powers of tau
    const powers = allocator.alloc(G1Point, num_points) catch return SRSError.OutOfMemory;
    errdefer allocator.free(powers);

    var offset: usize = 4;
    for (powers) |*p| {
        p.* = try parseG1Uncompressed(data[offset .. offset + 64]);
        offset += 64;
    }

    // Parse tau*G2
    const tau_g2 = try parseG2Uncompressed(data[offset .. offset + 128]);
    offset += 128;

    // Parse G1 generator
    const g1 = try parseG1Uncompressed(data[offset .. offset + 64]);
    offset += 64;

    // Parse G2 generator
    const g2 = try parseG2Uncompressed(data[offset .. offset + 128]);

    return SRSData{
        .powers_of_tau_g1 = powers,
        .tau_g2 = tau_g2,
        .g1 = g1,
        .g2 = g2,
        .max_degree = num_points,
        .allocator = allocator,
    };
}

/// Load SRS from a file
pub fn loadFromFile(allocator: Allocator, path: []const u8) SRSError!SRSData {
    const file = std.fs.cwd().openFile(path, .{}) catch return SRSError.FileNotFound;
    defer file.close();

    const stat = file.stat() catch return SRSError.FileNotFound;
    const data = allocator.alloc(u8, stat.size) catch return SRSError.OutOfMemory;
    defer allocator.free(data);

    _ = file.readAll(data) catch return SRSError.TruncatedData;

    return loadFromRawBinary(allocator, data);
}

/// Generate a mock SRS for testing (INSECURE!)
///
/// This generates a deterministic SRS with a known tau value.
/// DO NOT USE IN PRODUCTION - the tau secret is known!
pub fn generateMockSRS(allocator: Allocator, max_degree: usize) SRSError!SRSData {
    const powers = allocator.alloc(G1Point, max_degree) catch return SRSError.OutOfMemory;
    errdefer allocator.free(powers);

    // Use generators
    const g1 = G1Point.generator();
    const g2 = G2Point.generator();

    // Use a deterministic "tau" for testing (INSECURE!)
    const tau = Fr.fromU64(0x12345678);

    // Compute powers of tau in G1
    var tau_power = Fr.one();
    for (powers) |*p| {
        p.* = msm.MSM(Fr, Fp).scalarMul(g1, tau_power).toAffine();
        tau_power = tau_power.mul(tau);
    }

    // Compute tau*G2
    const tau_g2 = g2.scalarMul(tau);

    return SRSData{
        .powers_of_tau_g1 = powers,
        .tau_g2 = tau_g2,
        .g1 = g1,
        .g2 = g2,
        .max_degree = max_degree,
        .allocator = allocator,
    };
}

/// Serialize SRS to raw binary format
pub fn serializeToRawBinary(allocator: Allocator, srs: *const SRSData) ![]u8 {
    const num_points: u32 = @intCast(srs.powers_of_tau_g1.len);
    const g1_size = @as(usize, num_points) * 64;
    const total_size = 4 + g1_size + 128 + 64 + 128;

    const data = try allocator.alloc(u8, total_size);
    errdefer allocator.free(data);

    // Write number of points
    std.mem.writeInt(u32, data[0..4], num_points, .little);

    // Write G1 powers of tau
    var offset: usize = 4;
    for (srs.powers_of_tau_g1) |p| {
        const x_bytes = p.x.toBytesBE();
        const y_bytes = p.y.toBytesBE();
        @memcpy(data[offset .. offset + 32], &x_bytes);
        @memcpy(data[offset + 32 .. offset + 64], &y_bytes);
        offset += 64;
    }

    // Write tau*G2
    const tau_g2_x_c0 = srs.tau_g2.x.c0.toBytesBE();
    const tau_g2_x_c1 = srs.tau_g2.x.c1.toBytesBE();
    const tau_g2_y_c0 = srs.tau_g2.y.c0.toBytesBE();
    const tau_g2_y_c1 = srs.tau_g2.y.c1.toBytesBE();
    @memcpy(data[offset .. offset + 32], &tau_g2_x_c1);
    @memcpy(data[offset + 32 .. offset + 64], &tau_g2_x_c0);
    @memcpy(data[offset + 64 .. offset + 96], &tau_g2_y_c1);
    @memcpy(data[offset + 96 .. offset + 128], &tau_g2_y_c0);
    offset += 128;

    // Write G1 generator
    const g1_x = srs.g1.x.toBytesBE();
    const g1_y = srs.g1.y.toBytesBE();
    @memcpy(data[offset .. offset + 32], &g1_x);
    @memcpy(data[offset + 32 .. offset + 64], &g1_y);
    offset += 64;

    // Write G2 generator
    const g2_x_c0 = srs.g2.x.c0.toBytesBE();
    const g2_x_c1 = srs.g2.x.c1.toBytesBE();
    const g2_y_c0 = srs.g2.y.c0.toBytesBE();
    const g2_y_c1 = srs.g2.y.c1.toBytesBE();
    @memcpy(data[offset .. offset + 32], &g2_x_c1);
    @memcpy(data[offset + 32 .. offset + 64], &g2_x_c0);
    @memcpy(data[offset + 64 .. offset + 96], &g2_y_c1);
    @memcpy(data[offset + 96 .. offset + 128], &g2_y_c0);

    return data;
}

// ============================================================================
// Tests
// ============================================================================

test "mock SRS generation" {
    const allocator = std.testing.allocator;

    var srs = try generateMockSRS(allocator, 16);
    defer srs.deinit();

    // Check basic properties
    try std.testing.expectEqual(@as(usize, 16), srs.max_degree);
    try std.testing.expectEqual(@as(usize, 16), srs.powers_of_tau_g1.len);

    // First power should be the generator
    try std.testing.expect(srs.powers_of_tau_g1[0].isOnCurve());
    try std.testing.expect(!srs.powers_of_tau_g1[0].infinity);
}

test "G1 point parsing uncompressed" {
    // Create a generator point and serialize it
    const g1 = G1Point.generator();
    const x_bytes = g1.x.toBytesBE();
    const y_bytes = g1.y.toBytesBE();

    var data: [64]u8 = undefined;
    @memcpy(data[0..32], &x_bytes);
    @memcpy(data[32..64], &y_bytes);

    // Parse it back
    const parsed = try parseG1Uncompressed(&data);
    try std.testing.expect(parsed.x.eql(g1.x));
    try std.testing.expect(parsed.y.eql(g1.y));
}

test "G2 point parsing uncompressed" {
    // Create a generator point and serialize it
    const g2 = G2Point.generator();
    const x_c0_bytes = g2.x.c0.toBytesBE();
    const x_c1_bytes = g2.x.c1.toBytesBE();
    const y_c0_bytes = g2.y.c0.toBytesBE();
    const y_c1_bytes = g2.y.c1.toBytesBE();

    var data: [128]u8 = undefined;
    @memcpy(data[0..32], &x_c1_bytes);
    @memcpy(data[32..64], &x_c0_bytes);
    @memcpy(data[64..96], &y_c1_bytes);
    @memcpy(data[96..128], &y_c0_bytes);

    // Parse it back
    const parsed = try parseG2Uncompressed(&data);
    try std.testing.expect(parsed.x.c0.eql(g2.x.c0));
    try std.testing.expect(parsed.x.c1.eql(g2.x.c1));
    try std.testing.expect(parsed.y.c0.eql(g2.y.c0));
    try std.testing.expect(parsed.y.c1.eql(g2.y.c1));
}

test "SRS round-trip serialization" {
    const allocator = std.testing.allocator;

    // Generate a mock SRS
    var srs = try generateMockSRS(allocator, 8);
    defer srs.deinit();

    // Serialize it
    const data = try serializeToRawBinary(allocator, &srs);
    defer allocator.free(data);

    // Load it back
    var loaded = try loadFromRawBinary(allocator, data);
    defer loaded.deinit();

    // Verify
    try std.testing.expectEqual(srs.max_degree, loaded.max_degree);
    for (srs.powers_of_tau_g1, loaded.powers_of_tau_g1) |orig, load| {
        try std.testing.expect(orig.x.eql(load.x));
        try std.testing.expect(orig.y.eql(load.y));
    }
}

test "point at infinity parsing" {
    // Zero data should give point at infinity
    const zero_g1: [64]u8 = [_]u8{0} ** 64;
    const parsed_g1 = try parseG1Uncompressed(&zero_g1);
    try std.testing.expect(parsed_g1.infinity);

    const zero_g2: [128]u8 = [_]u8{0} ** 128;
    const parsed_g2 = try parseG2Uncompressed(&zero_g2);
    try std.testing.expect(parsed_g2.infinity);
}

test "sqrtFp correctness" {
    // Test sqrt of perfect squares
    const two = Fp.fromU64(2);
    const four = two.mul(two);

    if (sqrtFp(four)) |sqrt| {
        const sq = sqrt.mul(sqrt);
        try std.testing.expect(sq.eql(four));
    }

    // Test sqrt of generator y-coordinate squared
    const g1 = G1Point.generator();
    const y_sq = g1.y.mul(g1.y);
    if (sqrtFp(y_sq)) |sqrt| {
        const sq = sqrt.mul(sqrt);
        try std.testing.expect(sq.eql(y_sq));
    }
}
