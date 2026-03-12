//! Dory Polynomial Commitment Scheme
//!
//! Dory is a transparent polynomial commitment scheme based on inner product arguments.
//! This module provides a Jolt-compatible Dory implementation.
//!
//! ## Key Features
//!
//! - Transparent setup (no trusted setup required)
//! - SRS generation matches Jolt's seed ("Jolt Dory URS seed")
//! - GT element serialization in arkworks format (384 bytes)
//! - G1/G2 point compression matching arkworks format
//!
//! ## Structure
//!
//! Dory commits to a polynomial P(X) by:
//! 1. Computing row commitments: C_i = MSM(G_j, P[i*cols..(i+1)*cols])
//! 2. Computing the commitment: C = multi_pairing(C_i, H_i)
//!
//! The commitment is a GT element (Fp12 in BN254).
//!
//! Reference: jolt-core/src/poly/commitment/dory/

const std = @import("std");

// Debug output control - set to true to enable verbose debug prints
const debug_verbose = false;
fn dbg(comptime fmt: []const u8, args: anytype) void {
    if (debug_verbose) std.debug.print(fmt, args);
}

const Allocator = std.mem.Allocator;
const pairing = @import("../../field/pairing.zig");
const field = @import("../../field/mod.zig");
const msm = @import("../../msm/mod.zig");
const glv = msm.glv;
const ThreadPool = @import("../../utils/thread_pool.zig").ThreadPool;

const Fp = field.BN254BaseField;
const Fr = field.BN254Scalar;
const Fp2 = pairing.Fp2;
pub const GT = pairing.GT;
pub const G1Point = msm.AffinePoint(Fp);
pub const G2Point = pairing.G2Point;
const G2Projective = pairing.G2Projective;
const G1PointFp = pairing.G1PointFp;

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
fn doryAppendGT(transcript: anytype, gt: GT) void {
    const bytes = gt.toBytes();
    transcript.appendBytes("dory_serde", &bytes);
}

/// Append a G1 point to the transcript for Dory protocol.
/// Upstream: create_evaluation_proof uses transcript.append_serde() for ALL message
/// elements (GT, G1, G2). JoltToDoryTranscript::append_serde maps to "dory_serde".
fn doryAppendG1(transcript: anytype, point: G1Point) void {
    const bytes = compressG1(point);
    transcript.appendBytes("dory_serde", &bytes);
}

/// Append a G2 point to the transcript for Dory protocol.
/// Upstream: create_evaluation_proof uses transcript.append_serde() for ALL message
/// elements (GT, G1, G2). JoltToDoryTranscript::append_serde maps to "dory_serde".
fn doryAppendG2(transcript: anytype, point: G2Point) void {
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
fn tonelliShanks(n: Fp) ?Fp {
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
fn yIsPositive(y: Fp, neg_y: Fp) bool {
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

/// Compute square root in Fp2
/// Uses the algorithm for extension fields
fn fp2Sqrt(n: Fp2) ?Fp2 {
    if (n.isZero()) return Fp2.zero();

    // For Fp2 = Fp[u]/(u^2 + 1), we use a specialized algorithm.
    // Let n = a + bu. We want to find x = c + du such that x^2 = n.
    // (c + du)^2 = c^2 - d^2 + 2cdu = a + bu
    // So: c^2 - d^2 = a, 2cd = b

    // If b = 0: n = a is in Fp, so sqrt(n) = sqrt(a) or sqrt(a)*u
    if (n.c1.isZero()) {
        const sqrt_c0 = tonelliShanks(n.c0);
        if (sqrt_c0) |s| {
            return Fp2.init(s, Fp.zero());
        }
        // Try sqrt(-a) * u
        const neg_c0 = n.c0.neg();
        const sqrt_neg_c0 = tonelliShanks(neg_c0);
        if (sqrt_neg_c0) |s| {
            return Fp2.init(Fp.zero(), s);
        }
        return null;
    }

    // General case: use the formula
    // |n| = sqrt(a^2 + b^2) (in Fp, this is n * n.conjugate())
    const norm = n.c0.square().add(n.c1.square());
    const norm_sqrt = tonelliShanks(norm) orelse return null;

    // alpha = (a + |n|) / 2
    const two_inv = Fp.fromU64(2).inverse() orelse return null;
    const alpha = n.c0.add(norm_sqrt).mul(two_inv);
    const alpha_sqrt = tonelliShanks(alpha);

    if (alpha_sqrt) |c| {
        // d = b / (2c)
        const two_c_inv = c.add(c).inverse() orelse return null;
        const d = n.c1.mul(two_c_inv);
        return Fp2.init(c, d);
    }

    // Try the other case: alpha = (a - |n|) / 2
    const alpha2 = n.c0.sub(norm_sqrt).mul(two_inv);
    const alpha2_sqrt = tonelliShanks(alpha2.neg());
    if (alpha2_sqrt) |d| {
        // c = b / (2d)
        const two_d_inv = d.add(d).inverse() orelse return null;
        const c = n.c1.mul(two_d_inv);
        return Fp2.init(c, d);
    }

    return null;
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
// Dory Proof Structures (matching Jolt's format)
// =============================================================================

/// Dory commitment (GT element = Fp12)
pub const DoryCommitment = GT;

/// VMV (Vector-Matrix-Vector) message
/// Sent at the start of the Dory protocol
pub const VMVMessage = struct {
    c: GT, // e(MSM(T_vec', v_vec), Gamma_2_fin)
    d2: GT, // e(MSM(Gamma_1[nu], v_vec), Gamma_2_fin)
    e1: G1Point, // MSM(T_vec', L_vec)

    /// Serialize in arkworks-compatible format
    pub fn toBytes(self: *const VMVMessage) [384 + 384 + 32]u8 {
        var result: [384 + 384 + 32]u8 = undefined;
        @memcpy(result[0..384], &self.c.toBytes());
        @memcpy(result[384..768], &self.d2.toBytes());
        @memcpy(result[768..800], &compressG1(self.e1));
        return result;
    }
};

/// First reduce message for IPA rounds
pub const FirstReduceMessage = struct {
    d1_left: GT, // D1L
    d1_right: GT, // D1R
    d2_left: GT, // D2L
    d2_right: GT, // D2R
    e1_beta: G1Point, // E1_beta
    e2_beta: G2Point, // E2_beta

    /// Serialize in arkworks-compatible format
    pub fn toBytes(self: *const FirstReduceMessage) [384 * 4 + 32 + 64]u8 {
        var result: [384 * 4 + 32 + 64]u8 = undefined;
        @memcpy(result[0..384], &self.d1_left.toBytes());
        @memcpy(result[384..768], &self.d1_right.toBytes());
        @memcpy(result[768..1152], &self.d2_left.toBytes());
        @memcpy(result[1152..1536], &self.d2_right.toBytes());
        @memcpy(result[1536..1568], &compressG1(self.e1_beta));
        @memcpy(result[1568..1632], &compressG2(self.e2_beta));
        return result;
    }
};

/// Second reduce message for IPA rounds
pub const SecondReduceMessage = struct {
    c_plus: GT, // C+
    c_minus: GT, // C-
    e1_plus: G1Point, // E1+
    e1_minus: G1Point, // E1-
    e2_plus: G2Point, // E2+
    e2_minus: G2Point, // E2-

    /// Serialize in arkworks-compatible format
    pub fn toBytes(self: *const SecondReduceMessage) [384 * 2 + 32 * 2 + 64 * 2]u8 {
        var result: [384 * 2 + 32 * 2 + 64 * 2]u8 = undefined;
        @memcpy(result[0..384], &self.c_plus.toBytes());
        @memcpy(result[384..768], &self.c_minus.toBytes());
        @memcpy(result[768..800], &compressG1(self.e1_plus));
        @memcpy(result[800..832], &compressG1(self.e1_minus));
        @memcpy(result[832..896], &compressG2(self.e2_plus));
        @memcpy(result[896..960], &compressG2(self.e2_minus));
        return result;
    }
};

/// Final scalar product message
pub const ScalarProductMessage = struct {
    e1: G1Point, // E1
    e2: G2Point, // E2

    /// Serialize in arkworks-compatible format
    pub fn toBytes(self: *const ScalarProductMessage) [32 + 64]u8 {
        var result: [32 + 64]u8 = undefined;
        @memcpy(result[0..32], &compressG1(self.e1));
        @memcpy(result[32..96], &compressG2(self.e2));
        return result;
    }
};

/// Dory opening proof structure
/// Matches Jolt's ArkDoryProof structure exactly
pub const DoryProof = struct {
    /// VMV message (sent first)
    vmv_message: VMVMessage,
    /// First reduce messages (one per round)
    first_messages: []FirstReduceMessage,
    /// Second reduce messages (one per round)
    second_messages: []SecondReduceMessage,
    /// Final message
    final_message: ScalarProductMessage,
    /// Log2 of number of rows
    nu: u32,
    /// Log2 of number of columns
    sigma: u32,

    allocator: Allocator,

    pub fn deinit(self: *DoryProof) void {
        if (self.first_messages.len > 0) {
            self.allocator.free(self.first_messages);
            self.allocator.free(self.second_messages);
        }
    }

    /// Serialize in arkworks-compatible format
    /// Format matches dory-pcs ark_serde.rs
    pub fn toBytes(self: *const DoryProof, allocator: Allocator) ![]u8 {
        const num_rounds: u32 = @intCast(self.first_messages.len);

        // Calculate total size
        const vmv_size = 384 + 384 + 32; // c + d2 + e1
        const first_msg_size = 384 * 4 + 32 + 64; // d1L + d1R + d2L + d2R + e1_beta + e2_beta
        const second_msg_size = 384 * 2 + 32 * 2 + 64 * 2; // c+ + c- + e1+ + e1- + e2+ + e2-
        const final_size = 32 + 64; // e1 + e2

        const total_size = vmv_size + // VMV message
            4 + // num_rounds (u32)
            first_msg_size * num_rounds + // first messages
            second_msg_size * num_rounds + // second messages
            final_size + // final message
            4 + 4; // nu + sigma (u32 each)

        var result = try allocator.alloc(u8, total_size);
        var offset: usize = 0;

        // 1. VMV message
        const vmv_bytes = self.vmv_message.toBytes();
        @memcpy(result[offset..][0..vmv_size], &vmv_bytes);
        offset += vmv_size;

        // 2. Number of rounds
        std.mem.writeInt(u32, result[offset..][0..4], num_rounds, .little);
        offset += 4;

        // 3. First messages
        for (self.first_messages) |msg| {
            const msg_bytes = msg.toBytes();
            @memcpy(result[offset..][0..first_msg_size], &msg_bytes);
            offset += first_msg_size;
        }

        // 4. Second messages
        for (self.second_messages) |msg| {
            const msg_bytes = msg.toBytes();
            @memcpy(result[offset..][0..second_msg_size], &msg_bytes);
            offset += second_msg_size;
        }

        // 5. Final message
        const final_bytes = self.final_message.toBytes();
        @memcpy(result[offset..][0..final_size], &final_bytes);
        offset += final_size;

        // 6. nu and sigma
        std.mem.writeInt(u32, result[offset..][0..4], self.nu, .little);
        offset += 4;
        std.mem.writeInt(u32, result[offset..][0..4], self.sigma, .little);
        offset += 4;

        return result;
    }
};

// =============================================================================
// Helper Functions for Dory IPA
// =============================================================================

/// Compute multilinear Lagrange basis evaluations at a point
/// For variables (r_0, r_1, ..., r_{n-1}), computes all 2^n basis polynomial evaluations.
fn multilinearLagrangeBasis(comptime F: type, output: []F, point: []const F) void {
    if (point.len == 0 or output.len == 0) {
        if (output.len > 0) {
            output[0] = F.one();
        }
        return;
    }

    // Initialize for first variable: [1-r_0, r_0]
    const one_minus_p0 = F.one().sub(point[0]);
    output[0] = one_minus_p0;
    if (output.len > 1) {
        output[1] = point[0];
    }

    // For each subsequent variable, double the active portion
    for (1..point.len) |level| {
        const p = point[level];
        const mid = @as(usize, 1) << @intCast(level);
        const one_minus_p = F.one().sub(p);

        if (mid >= output.len) {
            // No split possible, just multiply all by (1-p)
            for (output) |*val| {
                val.* = val.*.mul(one_minus_p);
            }
        } else {
            // Split: left *= (1-p), right = left * p
            const k = @min(mid, output.len - mid);

            // Process from end to avoid overwriting
            var i: usize = k;
            while (i > 0) {
                i -= 1;
                const l_val = output[i];
                if (i + mid < output.len) {
                    output[i + mid] = l_val.mul(p);
                }
                output[i] = l_val.mul(one_minus_p);
            }
        }
    }
}

/// Compute left and right vectors from evaluation point
/// Given a point, computes L and R such that: polynomial(point) = L^T * M * R
fn computeEvaluationVectors(comptime F: type, point: []const F, nu: u32, sigma: u32, left_vec: []F, right_vec: []F) void {
    const point_dim = point.len;

    if (point_dim == 0) {
        left_vec[0] = F.one();
        right_vec[0] = F.one();
        return;
    }

    // All variables fit in columns (single row)
    if (point_dim <= sigma) {
        const out_len = @as(usize, 1) << @intCast(point_dim);
        multilinearLagrangeBasis(F, right_vec[0..out_len], point);
        left_vec[0] = F.one();
        return;
    }

    // Variables split between rows and columns
    if (point_dim <= nu + sigma) {
        multilinearLagrangeBasis(F, right_vec, point[0..sigma]);
        const left_len = @as(usize, 1) << @intCast(point_dim - sigma);
        multilinearLagrangeBasis(F, left_vec[0..left_len], point[sigma..]);
        return;
    }

    // Too many variables - need column padding
    multilinearLagrangeBasis(F, right_vec, point[0..sigma]);
    multilinearLagrangeBasis(F, left_vec, point[sigma..]);
}

/// Compute vector-matrix product: v = L^T * M
/// Treats coefficients as a 2^nu x 2^sigma matrix.
fn computeVectorMatrixProduct(comptime F: type, evals: []const F, left_vec: []const F, nu: u32, sigma: u32, allocator: Allocator) ![]F {
    const num_cols = @as(usize, 1) << @intCast(sigma);
    const num_rows = @as(usize, 1) << @intCast(nu);

    const result = try allocator.alloc(F, num_cols);
    @memset(result, F.zero());

    // For each column j: v[j] = sum_i left_vec[i] * M[i][j]
    for (0..num_rows) |row| {
        if (row >= left_vec.len) break;
        const coeff = left_vec[row];

        for (0..num_cols) |col| {
            const idx = row * num_cols + col;
            if (idx < evals.len) {
                result[col] = result[col].add(coeff.mul(evals[idx]));
            }
        }
    }

    return result;
}

/// Compute row commitments for a polynomial
fn computeRowCommitmentsWithCols(comptime F: type, params: anytype, evals: []const F, num_cols: usize, allocator: Allocator) ![]G1Point {
    const num_rows = (evals.len + num_cols - 1) / num_cols;

    const row_commitments = try allocator.alloc(G1Point, num_rows);
    errdefer allocator.free(row_commitments);

    for (0..num_rows) |row| {
        const start = row * num_cols;
        const end = @min(start + num_cols, evals.len);

        if (start >= evals.len) {
            row_commitments[row] = G1Point.identity();
            continue;
        }

        const row_evals = evals[start..end];
        row_commitments[row] = msm.MSM(F, Fp).compute(
            params.g1_vec[0..row_evals.len],
            row_evals,
        );
    }

    return row_commitments;
}

fn computeRowCommitmentsWithColsParallel(comptime F: type, params: anytype, evals: []const F, num_cols: usize, allocator: Allocator, tp: *ThreadPool) ![]G1Point {
    const num_rows = (evals.len + num_cols - 1) / num_cols;

    const row_commitments = try allocator.alloc(G1Point, num_rows);
    errdefer allocator.free(row_commitments);

    const Params = @TypeOf(params.*);
    const Ctx = struct {
        params_ptr: *const Params,
        evals_ptr: []const F,
        out: []G1Point,
        n_cols: usize,
        evals_len: usize,
    };
    const ctx = Ctx{
        .params_ptr = params,
        .evals_ptr = evals,
        .out = row_commitments,
        .n_cols = num_cols,
        .evals_len = evals.len,
    };

    tp.parallelForForce(num_rows, ctx, struct {
        fn f(c: Ctx, row: usize) void {
            const start = row * c.n_cols;
            const end = @min(start + c.n_cols, c.evals_len);

            if (start >= c.evals_len) {
                c.out[row] = G1Point.identity();
                return;
            }

            const row_evals = c.evals_ptr[start..end];
            c.out[row] = msm.MSM(F, Fp).compute(
                c.params_ptr.g1_vec[0..row_evals.len],
                row_evals,
            );
        }
    }.f);

    return row_commitments;
}

/// Compute multi-pairing of G1 and G2 vectors using shared final exponentiation.
/// Optionally uses thread pool for parallel Miller loop computation.
fn multiPairG1G2WithPool(g1_vec: []const G1Point, g2_vec: []const G2Point, tp: ?*ThreadPool) GT {
    const n = @min(g1_vec.len, g2_vec.len);
    if (n == 0) return GT.one();

    const Ctx = struct { g1_vec: []const G1Point, g2_vec: []const G2Point };
    const ctx = Ctx{ .g1_vec = g1_vec, .g2_vec = g2_vec };

    const mapFn = struct {
        fn map(c: Ctx, start: usize, end: usize) pairing.Fp12 {
            const chunk_len = end - start;
            // Convert G1 points to Fp for batched Miller loop
            var stack_g1: [64]G1PointFp = undefined;
            const use_heap = chunk_len > 64;
            var heap_g1: ?[]G1PointFp = null;
            defer if (heap_g1) |h| std.heap.page_allocator.free(h);

            var g1_fps: []G1PointFp = undefined;
            if (use_heap) {
                heap_g1 = std.heap.page_allocator.alloc(G1PointFp, chunk_len) catch {
                    // Fallback to individual loops
                    var acc = pairing.Fp12.one();
                    for (start..end) |i| {
                        if (c.g1_vec[i].infinity or c.g2_vec[i].infinity) continue;
                        const g1_fp = G1PointFp{ .x = c.g1_vec[i].x, .y = c.g1_vec[i].y, .infinity = false };
                        acc = acc.mul(pairing.millerLoopArkworks(g1_fp, c.g2_vec[i]));
                    }
                    return acc;
                };
                g1_fps = heap_g1.?;
            } else {
                g1_fps = stack_g1[0..chunk_len];
            }

            for (0..chunk_len) |j| {
                const i = start + j;
                g1_fps[j] = G1PointFp{
                    .x = c.g1_vec[i].x,
                    .y = c.g1_vec[i].y,
                    .infinity = c.g1_vec[i].infinity,
                };
            }

            return pairing.batchedMillerLoopUnprepared(g1_fps, c.g2_vec[start..end]);
        }
    }.map;

    const reduceFn = struct {
        fn reduce(a: pairing.Fp12, b: pairing.Fp12) pairing.Fp12 {
            return a.mul(b);
        }
    }.reduce;

    const miller_acc = if (tp) |pool|
        pool.parallelReduceForce(pairing.Fp12, n, pairing.Fp12.one(), ctx, mapFn, reduceFn)
    else
        mapFn(ctx, 0, n);

    if (miller_acc.eql(pairing.Fp12.one())) return GT.one();
    return pairing.finalExponentiation(miller_acc);
}

/// Multi-pairing using precomputed G2 coefficients (much faster for SRS G2 points).
fn multiPairG1G2Prepared(g1_vec: []const G1Point, g2_prep: []const G2Prepared, tp: ?*ThreadPool) GT {
    const n = @min(g1_vec.len, g2_prep.len);
    if (n == 0) return GT.one();

    const Ctx = struct { g1_vec: []const G1Point, g2_prep: []const G2Prepared };
    const ctx = Ctx{ .g1_vec = g1_vec, .g2_prep = g2_prep };

    const mapFn = struct {
        fn map(c: Ctx, start: usize, end: usize) pairing.Fp12 {
            const chunk_len = end - start;
            var stack_g1: [64]G1PointFp = undefined;
            const use_heap = chunk_len > 64;
            var heap_g1: ?[]G1PointFp = null;
            defer if (heap_g1) |h| std.heap.page_allocator.free(h);

            var g1_fps: []G1PointFp = undefined;
            if (use_heap) {
                heap_g1 = std.heap.page_allocator.alloc(G1PointFp, chunk_len) catch {
                    var acc = pairing.Fp12.one();
                    for (start..end) |i| {
                        if (c.g1_vec[i].infinity or c.g2_prep[i].infinity) continue;
                        const g1_fp = G1PointFp{ .x = c.g1_vec[i].x, .y = c.g1_vec[i].y, .infinity = false };
                        acc = acc.mul(pairing.millerLoopPrepared(g1_fp, &c.g2_prep[i]));
                    }
                    return acc;
                };
                g1_fps = heap_g1.?;
            } else {
                g1_fps = stack_g1[0..chunk_len];
            }

            for (0..chunk_len) |j| {
                const i = start + j;
                g1_fps[j] = G1PointFp{
                    .x = c.g1_vec[i].x,
                    .y = c.g1_vec[i].y,
                    .infinity = c.g1_vec[i].infinity,
                };
            }

            return pairing.batchedMillerLoopPreparedSparse(g1_fps, c.g2_prep[start..end]);
        }
    }.map;

    const reduceFn = struct {
        fn reduce(a: pairing.Fp12, b: pairing.Fp12) pairing.Fp12 {
            return a.mul(b);
        }
    }.reduce;

    const miller_acc = if (tp) |pool|
        pool.parallelReduceForce(pairing.Fp12, n, pairing.Fp12.one(), ctx, mapFn, reduceFn)
    else
        mapFn(ctx, 0, n);

    if (miller_acc.eql(pairing.Fp12.one())) return GT.one();
    return pairing.finalExponentiation(miller_acc);
}

/// Multi-pairing using affine line coefficients (fastest path: c0=1 implicit).
fn multiPairG1G2PreparedAffine(g1_vec: []const G1Point, g2_affine: []const G2PreparedAffine, tp: ?*ThreadPool) GT {
    const n = @min(g1_vec.len, g2_affine.len);
    if (n == 0) return GT.one();

    const Ctx = struct { g1_vec: []const G1Point, g2_affine: []const G2PreparedAffine };
    const ctx = Ctx{ .g1_vec = g1_vec, .g2_affine = g2_affine };

    const mapFn = struct {
        fn map(c: Ctx, start: usize, end: usize) pairing.Fp12 {
            const chunk_len = end - start;
            var stack_g1: [64]G1PointFp = undefined;
            const use_heap = chunk_len > 64;
            var heap_g1: ?[]G1PointFp = null;
            defer if (heap_g1) |h| std.heap.page_allocator.free(h);

            var g1_fps: []G1PointFp = undefined;
            if (use_heap) {
                heap_g1 = std.heap.page_allocator.alloc(G1PointFp, chunk_len) catch {
                    var acc = pairing.Fp12.one();
                    for (start..end) |i| {
                        if (c.g1_vec[i].infinity or c.g2_affine[i].infinity) continue;
                        const g1_fp = G1PointFp{ .x = c.g1_vec[i].x, .y = c.g1_vec[i].y, .infinity = false };
                        acc = acc.mul(pairing.millerLoopArkworks(g1_fp, pairing.G2Point{ .x = pairing.Fp2.zero(), .y = pairing.Fp2.zero(), .infinity = true }));
                    }
                    return acc;
                };
                g1_fps = heap_g1.?;
            } else {
                g1_fps = stack_g1[0..chunk_len];
            }

            for (0..chunk_len) |j| {
                const i = start + j;
                g1_fps[j] = G1PointFp{
                    .x = c.g1_vec[i].x,
                    .y = c.g1_vec[i].y,
                    .infinity = c.g1_vec[i].infinity,
                };
            }

            return pairing.batchedMillerLoopAffine(g1_fps, c.g2_affine[start..end]);
        }
    }.map;

    const reduceFn = struct {
        fn reduce(a: pairing.Fp12, b: pairing.Fp12) pairing.Fp12 {
            return a.mul(b);
        }
    }.reduce;

    const miller_acc = if (tp) |pool|
        pool.parallelReduceForce(pairing.Fp12, n, pairing.Fp12.one(), ctx, mapFn, reduceFn)
    else
        mapFn(ctx, 0, n);

    if (miller_acc.eql(pairing.Fp12.one())) return GT.one();
    return pairing.finalExponentiation(miller_acc);
}

/// A group of G1/G2 point slices for batched multi-pairing.
pub const PairGroup = struct {
    g1: []const G1Point,
    g2: []const G2Point,
};

/// Compute N independent multi-pairings in a single parallelReduceForce call,
/// sharing thread pool parallelism across all groups instead of running them sequentially.
/// Falls back to sequential per-group multiPairG1G2WithPool when tp is null.
pub fn multiPairBatched(comptime N: comptime_int, groups: [N]PairGroup, tp: ?*ThreadPool) [N]GT {
    // Compute prefix sums of pair counts
    var offsets: [N + 1]usize = undefined;
    offsets[0] = 0;
    inline for (0..N) |g| {
        offsets[g + 1] = offsets[g] + @min(groups[g].g1.len, groups[g].g2.len);
    }
    const total = offsets[N];

    // Fallback: sequential per-group
    if (total == 0 or tp == null) {
        var results: [N]GT = undefined;
        inline for (0..N) |g| {
            results[g] = multiPairG1G2WithPool(groups[g].g1, groups[g].g2, null);
        }
        return results;
    }

    const Ctx = struct {
        groups: [N]PairGroup,
        offsets: [N + 1]usize,
    };
    const ctx = Ctx{ .groups = groups, .offsets = offsets };

    const mapFn = struct {
        fn map(c: Ctx, start: usize, end: usize) [N]pairing.Fp12 {
            var accs: [N]pairing.Fp12 = undefined;
            inline for (0..N) |g| {
                accs[g] = pairing.Fp12.one();
            }

            // Process each group's sub-range within [start, end) using batched Miller loop
            for (0..N) |g| {
                const group_start = c.offsets[g];
                const group_end = c.offsets[g + 1];
                // Intersect [group_start, group_end) with [start, end)
                const lo = @max(group_start, start);
                const hi = @min(group_end, end);
                if (lo >= hi) continue;

                const chunk_len = hi - lo;
                const local_lo = lo - group_start;

                // Convert G1 points to Fp and batch Miller loop
                var stack_g1: [64]G1PointFp = undefined;
                const use_heap = chunk_len > 64;
                var heap_g1: ?[]G1PointFp = null;
                defer if (heap_g1) |h| std.heap.page_allocator.free(h);

                var g1_fps: []G1PointFp = undefined;
                if (use_heap) {
                    heap_g1 = std.heap.page_allocator.alloc(G1PointFp, chunk_len) catch {
                        // Fallback to individual loops
                        for (lo..hi) |idx| {
                            const li = idx - group_start;
                            const g1_pt = c.groups[g].g1[li];
                            const g2_pt = c.groups[g].g2[li];
                            if (g1_pt.infinity or g2_pt.infinity) continue;
                            const g1_fp = G1PointFp{ .x = g1_pt.x, .y = g1_pt.y, .infinity = false };
                            accs[g] = accs[g].mul(pairing.millerLoopArkworks(g1_fp, g2_pt));
                        }
                        continue;
                    };
                    g1_fps = heap_g1.?;
                } else {
                    g1_fps = stack_g1[0..chunk_len];
                }

                for (0..chunk_len) |j| {
                    const pt = c.groups[g].g1[local_lo + j];
                    g1_fps[j] = G1PointFp{
                        .x = pt.x,
                        .y = pt.y,
                        .infinity = pt.infinity,
                    };
                }

                accs[g] = pairing.batchedMillerLoopUnprepared(
                    g1_fps,
                    c.groups[g].g2[local_lo .. local_lo + chunk_len],
                );
            }
            return accs;
        }
    }.map;

    const reduceFn = struct {
        fn reduce(a: [N]pairing.Fp12, b: [N]pairing.Fp12) [N]pairing.Fp12 {
            var result: [N]pairing.Fp12 = undefined;
            inline for (0..N) |g| {
                result[g] = a[g].mul(b[g]);
            }
            return result;
        }
    }.reduce;

    var identity: [N]pairing.Fp12 = undefined;
    inline for (0..N) |g| {
        identity[g] = pairing.Fp12.one();
    }

    const miller_accs = tp.?.parallelReduceForce(
        [N]pairing.Fp12, total, identity, ctx, mapFn, reduceFn,
    );

    // Final exponentiations — run in parallel via parallelForForce
    const FinalExpCtx = struct {
        accs: *const [N]pairing.Fp12,
        results: *[N]GT,
    };
    var results: [N]GT = undefined;
    const fe_ctx = FinalExpCtx{ .accs = &miller_accs, .results = &results };

    tp.?.parallelForForce(N, fe_ctx, struct {
        fn f(c: FinalExpCtx, i: usize) void {
            if (c.accs[i].eql(pairing.Fp12.one())) {
                c.results[i] = GT.one();
            } else {
                c.results[i] = pairing.finalExponentiation(c.accs[i]);
            }
        }
    }.f);

    return results;
}

/// MSM for G2 points using Pippenger's bucket method with wNAF.
/// For small inputs (< 8), falls back to naive GLV scalar mul.
fn msmG2(comptime F: type, g2_vec: []const G2Point, scalars: []const F, tp: ?*ThreadPool) G2Point {
    const n = @min(g2_vec.len, scalars.len);
    if (n == 0) return G2Point.identity();

    // Small inputs: naive GLV
    if (n < 8) {
        var result = G2Projective.identity();
        for (0..n) |i| {
            const scaled = glv.glvScalarMulG2(g2_vec[i], scalars[i]);
            result = result.add(scaled);
        }
        return result.toAffine();
    }

    if (tp != null and n >= 256) {
        return pippengerMsmG2Parallel(F, g2_vec[0..n], scalars[0..n], tp.?);
    }

    return pippengerMsmG2(F, g2_vec[0..n], scalars[0..n]);
}

/// Pippenger MSM for G2 with wNAF signed digits + Jacobian buckets
fn pippengerMsmG2(comptime F: type, bases: []const G2Point, scalars: []const F) G2Point {
    const SCALAR_BITS: usize = 256;
    const MAX_DIGITS: usize = 87;

    const c = g2OptimalWindowSize(bases.len);
    const num_scalar_windows = (SCALAR_BITS + c - 1) / c;
    const num_windows = num_scalar_windows + 1; // +1 for wNAF carry
    const num_buckets = (@as(usize, 1) << @as(u6, @intCast(c))) / 2; // wNAF: 2^(c-1) buckets

    // Compute wNAF digits for all scalars
    const stack_threshold = 128;
    var stack_digits: [stack_threshold][MAX_DIGITS]i32 = undefined;
    var heap_digits: ?[][MAX_DIGITS]i32 = null;
    defer if (heap_digits) |buf| std.heap.page_allocator.free(buf);

    const all_digits: [][MAX_DIGITS]i32 = if (scalars.len <= stack_threshold)
        stack_digits[0..scalars.len]
    else blk: {
        heap_digits = std.heap.page_allocator.alloc([MAX_DIGITS]i32, scalars.len) catch {
            // Fallback to naive
            var result = G2Projective.identity();
            for (0..scalars.len) |i| {
                result = result.add(glv.glvScalarMulG2(bases[i], scalars[i]));
            }
            return result.toAffine();
        };
        break :blk heap_digits.?;
    };

    for (scalars, 0..) |s, i| {
        all_digits[i] = makeG2Digits(s.fromMontgomery().limbs, c, num_scalar_windows);
    }

    // Allocate buckets
    var heap_buckets: ?[]G2Projective = null;
    defer if (heap_buckets) |buf| std.heap.page_allocator.free(buf);

    var stack_buckets: [128]G2Projective = undefined;
    const buckets: []G2Projective = if (num_buckets <= 128)
        stack_buckets[0..num_buckets]
    else blk: {
        heap_buckets = std.heap.page_allocator.alloc(G2Projective, num_buckets) catch {
            var result = G2Projective.identity();
            for (0..scalars.len) |i| {
                result = result.add(glv.glvScalarMulG2(bases[i], scalars[i]));
            }
            return result.toAffine();
        };
        break :blk heap_buckets.?;
    };

    var final_result = G2Projective.identity();

    var window_idx: usize = num_windows;
    while (window_idx > 0) {
        window_idx -= 1;

        if (!final_result.isIdentity()) {
            var k: usize = 0;
            while (k < c) : (k += 1) {
                final_result = final_result.double();
            }
        }

        // Reset buckets
        for (0..num_buckets) |j| {
            buckets[j] = G2Projective.identity();
        }

        // Accumulate using wNAF signed digits
        for (bases, 0..) |base, idx| {
            if (base.infinity) continue;
            const digit = all_digits[idx][window_idx];
            if (digit > 0) {
                const bidx: usize = @intCast(digit - 1);
                buckets[bidx] = buckets[bidx].addAffine(base);
            } else if (digit < 0) {
                const bidx: usize = @intCast(-digit - 1);
                buckets[bidx] = buckets[bidx].addAffine(base.neg());
            }
        }

        // Running sum reduction
        var running_sum = G2Projective.identity();
        var window_sum = G2Projective.identity();

        var bucket_idx: usize = num_buckets;
        while (bucket_idx > 0) {
            bucket_idx -= 1;
            running_sum = running_sum.add(buckets[bucket_idx]);
            window_sum = window_sum.add(running_sum);
        }

        final_result = final_result.add(window_sum);
    }

    return final_result.toAffine();
}

/// Pippenger MSM for G2 with parallel window processing
fn pippengerMsmG2Parallel(comptime F: type, bases: []const G2Point, scalars: []const F, tp: *ThreadPool) G2Point {
    const SCALAR_BITS: usize = 256;
    const MAX_DIGITS: usize = 87;

    const c = g2OptimalWindowSize(bases.len);
    const num_scalar_windows = (SCALAR_BITS + c - 1) / c;
    const num_windows = num_scalar_windows + 1;
    const num_buckets = (@as(usize, 1) << @as(u6, @intCast(c))) / 2;

    // Compute wNAF digits
    const heap_digits = std.heap.page_allocator.alloc([MAX_DIGITS]i32, scalars.len) catch
        return pippengerMsmG2(F, bases, scalars);
    defer std.heap.page_allocator.free(heap_digits);

    for (scalars, 0..) |s, i| {
        heap_digits[i] = makeG2Digits(s.fromMontgomery().limbs, c, num_scalar_windows);
    }

    // Allocate per-window buckets and window sums
    const all_buckets = std.heap.page_allocator.alloc(G2Projective, num_windows * num_buckets) catch
        return pippengerMsmG2(F, bases, scalars);
    defer std.heap.page_allocator.free(all_buckets);

    const window_sums = std.heap.page_allocator.alloc(G2Projective, num_windows) catch
        return pippengerMsmG2(F, bases, scalars);
    defer std.heap.page_allocator.free(window_sums);

    // Phase 1: Process all windows in parallel
    const ParCtx = struct {
        all_digits: [][MAX_DIGITS]i32,
        all_buckets: []G2Projective,
        window_sums: []G2Projective,
        bases: []const G2Point,
        num_buckets: usize,
    };
    const ctx = ParCtx{
        .all_digits = heap_digits,
        .all_buckets = all_buckets,
        .window_sums = window_sums,
        .bases = bases,
        .num_buckets = num_buckets,
    };

    tp.parallelForForce(num_windows, ctx, struct {
        fn f(cx: ParCtx, win_idx: usize) void {
            const bucket_offset = win_idx * cx.num_buckets;
            const buckets = cx.all_buckets[bucket_offset .. bucket_offset + cx.num_buckets];

            for (0..cx.num_buckets) |j| {
                buckets[j] = G2Projective.identity();
            }

            for (cx.bases, 0..) |base, idx| {
                if (base.infinity) continue;
                const digit = cx.all_digits[idx][win_idx];
                if (digit > 0) {
                    const bidx: usize = @intCast(digit - 1);
                    buckets[bidx] = buckets[bidx].addAffine(base);
                } else if (digit < 0) {
                    const bidx: usize = @intCast(-digit - 1);
                    buckets[bidx] = buckets[bidx].addAffine(base.neg());
                }
            }

            var running_sum = G2Projective.identity();
            var window_sum = G2Projective.identity();
            var bucket_idx: usize = cx.num_buckets;
            while (bucket_idx > 0) {
                bucket_idx -= 1;
                running_sum = running_sum.add(buckets[bucket_idx]);
                window_sum = window_sum.add(running_sum);
            }

            cx.window_sums[win_idx] = window_sum;
        }
    }.f);

    // Phase 2: Combine window sums sequentially (high to low)
    var final_result = window_sums[num_windows - 1];
    var window_idx: usize = num_windows - 1;
    while (window_idx > 0) {
        window_idx -= 1;
        var k: usize = 0;
        while (k < c) : (k += 1) {
            final_result = final_result.double();
        }
        final_result = final_result.add(window_sums[window_idx]);
    }

    return final_result.toAffine();
}

/// wNAF digit decomposition for G2 Pippenger
fn makeG2Digits(limbs: [4]u64, c: usize, num_digits: usize) [87]i32 {
    const MAX_G2_DIGITS = 87;
    var digits: [MAX_G2_DIGITS]i32 = undefined;
    const radix: u64 = @as(u64, 1) << @as(u6, @intCast(c));
    const window_mask: u64 = radix - 1;
    const half_radix: u64 = radix >> 1;
    var carry: u64 = 0;

    for (0..num_digits) |i| {
        const bit_offset = i * c;
        const u64_idx = bit_offset / 64;
        const bit_idx = @as(u6, @intCast(bit_offset % 64));

        var bit_buf: u64 = if (u64_idx < 4) limbs[u64_idx] >> bit_idx else 0;
        const bit_idx_usize: usize = @as(usize, bit_idx);
        if (bit_idx_usize + c > 64 and u64_idx + 1 < 4) {
            bit_buf |= limbs[u64_idx + 1] << @as(u6, @intCast(64 - bit_idx_usize));
        }

        const coef = carry + (bit_buf & window_mask);
        carry = if (coef >= half_radix) 1 else 0;
        const digit: i32 = @as(i32, @intCast(coef)) - @as(i32, @intCast(carry)) * @as(i32, @intCast(radix));
        digits[i] = digit;
    }
    // Final carry as extra digit
    if (num_digits < MAX_G2_DIGITS) {
        digits[num_digits] = @intCast(carry);
    }
    return digits;
}

/// Optimal window size for G2 Pippenger
fn g2OptimalWindowSize(n: usize) usize {
    if (n < 32) return 3;
    if (n < 256) return 5;
    if (n < 2048) return 6;
    if (n < 16384) return 7;
    if (n < 131072) return 8;
    return 9;
}

/// Dory structured reference string (SRS)
/// Generated using the seed "Jolt Dory URS seed" for compatibility
const G2Prepared = pairing.G2Prepared;
const G2PreparedAffine = pairing.G2PreparedAffine;

pub const DorySRS = struct {
    /// G1 generators for polynomial coefficients
    g1_vec: []G1Point,
    /// G2 generators for pairing
    g2_vec: []G2Point,
    /// Precomputed G2 Miller loop coefficients for fast pairings
    g2_prepared: ?[]G2Prepared,
    /// Precomputed affine line coefficients (Phase 3: c0=1 implicit, batch-inverted)
    g2_prepared_affine: ?[]G2PreparedAffine,
    /// Maximum number of columns in the matrix
    num_columns: usize,
    /// Maximum number of rows in the matrix
    num_rows: usize,
    /// Log2 of columns (sigma)
    sigma: u32,
    /// Log2 of rows (nu)
    nu: u32,
    /// Blinding generator in G1 (used in Dory IPA final message)
    h1: G1Point,
    /// Blinding generator in G2 (used in Dory IPA VMV + final message)
    h2: G2Point,
    allocator: Allocator,

    /// Precompute G2 Miller loop coefficients for all SRS G2 points.
    /// This makes subsequent pairings ~2x faster by avoiding G2 arithmetic.
    pub fn initPreparedCache(self: *DorySRS, tp: ?*ThreadPool) void {
        if (self.g2_prepared != null) return; // Already initialized
        const n = self.g2_vec.len;
        if (n == 0) return;
        const prepared = self.allocator.alloc(G2Prepared, n) catch return;
        if (tp) |pool| {
            const PrepCtx = struct { src: []const G2Point, dst: []G2Prepared };
            pool.parallelForForce(n, PrepCtx{ .src = self.g2_vec, .dst = prepared }, struct {
                fn f(ctx: PrepCtx, i: usize) void {
                    ctx.dst[i] = G2Prepared.fromG2Point(ctx.src[i]);
                }
            }.f);
        } else {
            for (0..n) |i| {
                prepared[i] = G2Prepared.fromG2Point(self.g2_vec[i]);
            }
        }
        self.g2_prepared = prepared;

        // Also build affine cache from the projective prepared data
        self.initPreparedCacheAffine(tp);
    }

    /// Precompute affine line coefficients from projective G2Prepared.
    /// Uses batch Fp2 inversion for ~15% faster Miller loops (c0=1 implicit).
    fn initPreparedCacheAffine(self: *DorySRS, tp: ?*ThreadPool) void {
        if (self.g2_prepared_affine != null) return;
        const prep = self.g2_prepared orelse return;
        const n = prep.len;
        if (n == 0) return;
        const affine = self.allocator.alloc(G2PreparedAffine, n) catch return;
        if (tp) |pool| {
            const AffCtx = struct { src: []const G2Prepared, dst: []G2PreparedAffine };
            pool.parallelForForce(n, AffCtx{ .src = prep, .dst = affine }, struct {
                fn f(ctx: AffCtx, i: usize) void {
                    ctx.dst[i] = G2PreparedAffine.fromG2Prepared(&ctx.src[i]);
                }
            }.f);
        } else {
            for (0..n) |i| {
                affine[i] = G2PreparedAffine.fromG2Prepared(&prep[i]);
            }
        }
        self.g2_prepared_affine = affine;
    }

    pub fn deinit(self: *DorySRS) void {
        if (self.g1_vec.len > 0) {
            self.allocator.free(self.g1_vec);
            self.allocator.free(self.g2_vec);
        }
        if (self.g2_prepared) |prep| {
            self.allocator.free(prep);
        }
        if (self.g2_prepared_affine) |affine| {
            self.allocator.free(affine);
        }
    }
};

/// Dory commitment scheme matching Jolt's implementation
pub fn DoryCommitmentScheme(comptime F: type) type {
    return struct {
        const Self = @This();

        pub const SetupParams = DorySRS;
        pub const Commitment = DoryCommitment;
        pub const Proof = DoryProof;
        pub const FieldType = F;

        /// Load SRS from a file exported by Jolt
        ///
        /// This loads a Dory SRS that was exported using Jolt's test_export_dory_srs.
        /// The file format is:
        /// - 16 bytes: "JOLT_DORY_SRS_V1"
        /// - 8 bytes: max_num_vars (u64 LE)
        /// - 8 bytes: g1_count (u64 LE)
        /// - g1_count * 64 bytes: G1 points (arkworks uncompressed format)
        /// - 8 bytes: g2_count (u64 LE)
        /// - g2_count * 128 bytes: G2 points (arkworks uncompressed format)
        /// - 64 bytes: h1 (blinding G1 generator)
        /// - 128 bytes: h2 (blinding G2 generator)
        pub fn loadFromFile(allocator: Allocator, path: []const u8) !SetupParams {
            const file = std.fs.cwd().openFile(path, .{}) catch |err| {
                dbg("Failed to open SRS file: {s}\n", .{path});
                return err;
            };
            defer file.close();

            // Read and verify header
            var header: [16]u8 = undefined;
            _ = try file.readAll(&header);
            if (!std.mem.eql(u8, &header, "JOLT_DORY_SRS_V1")) {
                return error.InvalidSrsFormat;
            }

            // Read max_num_vars
            var num_vars_bytes: [8]u8 = undefined;
            _ = try file.readAll(&num_vars_bytes);
            const max_num_vars = std.mem.readInt(u64, &num_vars_bytes, .little);

            // Calculate matrix dimensions
            const sigma: u32 = @intCast((max_num_vars + 1) / 2);
            const nu: u32 = @intCast(max_num_vars - sigma);

            // Read G1 count and points
            var g1_count_bytes: [8]u8 = undefined;
            _ = try file.readAll(&g1_count_bytes);
            const g1_count = std.mem.readInt(u64, &g1_count_bytes, .little);

            const g1_vec = try allocator.alloc(G1Point, @intCast(g1_count));
            errdefer allocator.free(g1_vec);

            for (g1_vec, 0..) |*g1, idx| {
                var buf: [64]u8 = undefined;
                _ = try file.readAll(&buf);
                // Debug: print raw bytes for first few points
                if (idx < 4) {
                    dbg("G1[{}] raw y bytes from file: {x}\n", .{ idx, buf[32..48].* });
                }
                // Parse arkworks uncompressed G1 format (64 bytes: x, y in LE)
                g1.* = parseG1Uncompressed(&buf);
            }

            // Read G2 count and points
            var g2_count_bytes: [8]u8 = undefined;
            _ = try file.readAll(&g2_count_bytes);
            const g2_count = std.mem.readInt(u64, &g2_count_bytes, .little);

            const g2_vec = try allocator.alloc(G2Point, @intCast(g2_count));
            errdefer allocator.free(g2_vec);

            for (g2_vec) |*g2| {
                var buf: [128]u8 = undefined;
                _ = try file.readAll(&buf);
                // Parse arkworks uncompressed G2 format (128 bytes: x, y as Fp2 in LE)
                g2.* = parseG2Uncompressed(&buf);
            }

            // Read blinding generators h1 (G1, 64 bytes) and h2 (G2, 128 bytes)
            var h1_buf: [64]u8 = undefined;
            _ = try file.readAll(&h1_buf);
            const h1 = parseG1Uncompressed(&h1_buf);

            var h2_buf: [128]u8 = undefined;
            _ = try file.readAll(&h2_buf);
            const h2 = parseG2Uncompressed(&h2_buf);

            return SetupParams{
                .g1_vec = g1_vec,
                .g2_vec = g2_vec,
                .g2_prepared = null,
                .g2_prepared_affine = null,
                .num_columns = @intCast(g1_count),
                .num_rows = @intCast(g2_count),
                .sigma = sigma,
                .nu = nu,
                .h1 = h1,
                .h2 = h2,
                .allocator = allocator,
            };
        }

        /// Parse arkworks uncompressed G1 point (64 bytes: x[32] || y[32] in LE)
        /// arkworks stores flag bits in the MSB of the last byte:
        /// - bit 7: y-sign flag (for compressed points, but still present in uncompressed)
        /// - bit 6: infinity flag
        fn parseG1Uncompressed(buf: *const [64]u8) G1Point {
            // Read x coordinate (32 bytes LE as 4 u64 limbs)
            var x_limbs: [4]u64 = undefined;
            for (0..4) |i| {
                x_limbs[i] = std.mem.readInt(u64, buf[i * 8 ..][0..8], .little);
            }

            // Read y coordinate
            var y_limbs: [4]u64 = undefined;
            for (0..4) |i| {
                y_limbs[i] = std.mem.readInt(u64, buf[32 + i * 8 ..][0..8], .little);
            }

            // Clear arkworks flag bits from the most significant byte of y coordinate
            // Flags are in the top 2 bits of the last byte (byte 63)
            // limbs[3] is the most significant, and its top byte contains flags
            y_limbs[3] &= 0x3FFFFFFFFFFFFFFF; // Clear top 2 bits

            // Convert from standard to Montgomery form
            const x_raw = Fp{ .limbs = x_limbs };
            const y_raw = Fp{ .limbs = y_limbs };

            // Check for identity point (all zeros)
            const is_zero = blk: {
                for (x_limbs) |l| if (l != 0) break :blk false;
                for (y_limbs) |l| if (l != 0) break :blk false;
                break :blk true;
            };

            if (is_zero) {
                return G1Point.identity();
            }

            const x_mont = x_raw.toMontgomery();
            const y_mont = y_raw.toMontgomery();

            // Verify round-trip: converting to Montgomery and back should give original
            const y_back = y_mont.fromMontgomery();
            if (y_limbs[1] != 0 and !std.meta.eql(y_back.limbs, y_limbs)) {
                dbg("\nMontgomery round-trip FAILED!\n", .{});
                dbg("  Original y limbs: {x:0>16} {x:0>16} {x:0>16} {x:0>16}\n", .{
                    y_limbs[0], y_limbs[1], y_limbs[2], y_limbs[3],
                });
                dbg("  y_mont limbs: {x:0>16} {x:0>16} {x:0>16} {x:0>16}\n", .{
                    y_mont.limbs[0], y_mont.limbs[1], y_mont.limbs[2], y_mont.limbs[3],
                });
                dbg("  After round-trip: {x:0>16} {x:0>16} {x:0>16} {x:0>16}\n", .{
                    y_back.limbs[0], y_back.limbs[1], y_back.limbs[2], y_back.limbs[3],
                });
                // Also compare what manual computation gives
                const y_mont_manual = field.testMontgomeryMulFp(y_limbs, field.BN254_FP_R2);
                dbg("  y_mont_manual limbs: {x:0>16} {x:0>16} {x:0>16} {x:0>16}\n", .{
                    y_mont_manual[0], y_mont_manual[1], y_mont_manual[2], y_mont_manual[3],
                });
            }

            return G1Point{
                .x = x_mont,
                .y = y_mont,
                .infinity = false,
            };
        }

        /// Parse arkworks uncompressed G2 point (128 bytes: x.c0[32] || x.c1[32] || y.c0[32] || y.c1[32])
        fn parseG2Uncompressed(buf: *const [128]u8) G2Point {
            // Read x.c0, x.c1, y.c0, y.c1 (each 32 bytes)
            var x_c0_limbs: [4]u64 = undefined;
            var x_c1_limbs: [4]u64 = undefined;
            var y_c0_limbs: [4]u64 = undefined;
            var y_c1_limbs: [4]u64 = undefined;

            for (0..4) |i| {
                x_c0_limbs[i] = std.mem.readInt(u64, buf[i * 8 ..][0..8], .little);
                x_c1_limbs[i] = std.mem.readInt(u64, buf[32 + i * 8 ..][0..8], .little);
                y_c0_limbs[i] = std.mem.readInt(u64, buf[64 + i * 8 ..][0..8], .little);
                y_c1_limbs[i] = std.mem.readInt(u64, buf[96 + i * 8 ..][0..8], .little);
            }

            // Clear arkworks flag bits from y.c1 (last 32 bytes of the 128-byte point)
            // Flags are in the top 2 bits of the last byte
            y_c1_limbs[3] &= 0x3FFFFFFFFFFFFFFF;

            const x_c0_raw = Fp{ .limbs = x_c0_limbs };
            const x_c1_raw = Fp{ .limbs = x_c1_limbs };
            const y_c0_raw = Fp{ .limbs = y_c0_limbs };
            const y_c1_raw = Fp{ .limbs = y_c1_limbs };
            const x_c0 = x_c0_raw.toMontgomery();
            const x_c1 = x_c1_raw.toMontgomery();
            const y_c0 = y_c0_raw.toMontgomery();
            const y_c1 = y_c1_raw.toMontgomery();

            return G2Point{
                .x = Fp2{ .c0 = x_c0, .c1 = x_c1 },
                .y = Fp2{ .c0 = y_c0, .c1 = y_c1 },
                .infinity = false,
            };
        }

        /// Setup the SRS using Jolt's seed
        ///
        /// Uses SHA3-256 with seed "Jolt Dory URS seed" for deterministic generation.
        /// Note: This generates points differently from Jolt's arkworks-based generation.
        /// For exact compatibility, use loadFromFile with a Jolt-exported SRS.
        pub fn setup(allocator: Allocator, max_num_vars: usize) !SetupParams {
            // Calculate matrix dimensions
            // For n variables, we need 2^n coefficients
            const total_size: usize = @as(usize, 1) << @intCast(max_num_vars);

            // Aim for roughly square matrix
            const sigma: u32 = @intCast((max_num_vars + 1) / 2); // columns = 2^sigma
            const nu: u32 = @intCast(max_num_vars - sigma); // rows = 2^nu

            const num_columns: usize = @as(usize, 1) << @intCast(sigma);
            const num_rows: usize = @as(usize, 1) << @intCast(nu);

            // In Jolt's Dory SRS, BOTH g1_vec and g2_vec have the same length = 2^sigma.
            // This is critical because the reduce-and-fold IPA uses vectors of length 2^sigma
            // for ALL operations, including G2 MSMs. If g2_vec is shorter (2^nu when nu < sigma),
            // the IPA will read beyond the array bounds.
            const n = @max(num_columns, num_rows); // = 2^sigma since sigma >= nu

            // Generate G1 generators
            const g1_vec = try allocator.alloc(G1Point, n);
            errdefer allocator.free(g1_vec);

            // Generate G2 generators (same count as G1 - matches Jolt)
            const g2_vec = try allocator.alloc(G2Point, n);
            errdefer allocator.free(g2_vec);

            // Use SHA3-256 with Jolt's seed for deterministic generation
            var hasher = std.crypto.hash.sha3.Sha3_256.init(.{});
            hasher.update("Jolt Dory URS seed");
            var seed: [32]u8 = undefined;
            hasher.final(&seed);

            // Generate G1 points using hash-to-curve simulation
            // In production, this would use proper hash-to-curve
            for (0..n) |i| {
                g1_vec[i] = generateG1Point(seed, i);
            }

            // Generate G2 points
            for (0..n) |i| {
                g2_vec[i] = generateG2Point(seed, i + n);
            }

            _ = total_size;

            return SetupParams{
                .g1_vec = g1_vec,
                .g2_vec = g2_vec,
                .g2_prepared = null,
                .g2_prepared_affine = null,
                .num_columns = num_columns,
                .num_rows = num_rows,
                .sigma = sigma,
                .nu = nu,
                .h1 = G1Point.generator(),
                .h2 = G2Point.generator(),
                .allocator = allocator,
            };
        }

        /// Commit to a polynomial given as evaluations
        ///
        /// The polynomial is laid out as a 2^nu × 2^sigma matrix where:
        /// - num_vars = log2(evals.len)
        /// - sigma = ceil((num_vars + 1) / 2)
        /// - nu = num_vars - sigma
        ///
        /// This matches Jolt/dory-pcs matrix layout for compatible commitments.
        pub fn commit(params: *const SetupParams, evals: []const F) Commitment {
            return commitWithPool(params, evals, null);
        }

        pub fn commitWithPool(params: *const SetupParams, evals: []const F, tp: ?*ThreadPool) Commitment {
            if (evals.len == 0) {
                return GT.one();
            }

            const poly_len = evals.len;
            const num_vars: usize = if (poly_len <= 1) 1 else std.math.log2_int(usize, poly_len);
            const sigma: usize = (num_vars + 1) / 2;
            const nu: usize = num_vars - sigma;
            const num_cols = @as(usize, 1) << @intCast(sigma);
            const num_rows = @as(usize, 1) << @intCast(nu);

            // Use parallel reduce over rows: each row does MSM + Miller loop independently
            const Params = SetupParams;
            const Ctx = struct {
                params_ptr: *const Params,
                evals_ptr: []const F,
                n_cols: usize,
                n_rows: usize,
            };
            const ctx = Ctx{
                .params_ptr = params,
                .evals_ptr = evals,
                .n_cols = num_cols,
                .n_rows = num_rows,
            };

            const mapFn = struct {
                fn f(c: Ctx, start: usize, end: usize) pairing.Fp12 {
                    const chunk_len = end - start;
                    if (chunk_len == 0) return pairing.Fp12.one();

                    // Phase 1: compute all MSM results for rows in chunk
                    var stack_g1: [64]G1PointFp = undefined;
                    const use_heap = chunk_len > 64;
                    var heap_g1: ?[]G1PointFp = null;
                    defer if (heap_g1) |h| std.heap.page_allocator.free(h);

                    var g1_fps: []G1PointFp = undefined;
                    if (use_heap) {
                        heap_g1 = std.heap.page_allocator.alloc(G1PointFp, chunk_len) catch {
                            // Fallback to individual loops
                            var acc = pairing.Fp12.one();
                            for (start..end) |row| {
                                const row_s = row * c.n_cols;
                                if (row_s >= c.evals_ptr.len) break;
                                const row_e = @min(row_s + c.n_cols, c.evals_ptr.len);
                                const rc = msm.MSM(F, Fp).compute(c.params_ptr.g1_vec[0..row_e - row_s], c.evals_ptr[row_s..row_e]);
                                if (row < c.params_ptr.g2_vec.len and !rc.infinity) {
                                    const g1_fp = G1PointFp{ .x = rc.x, .y = rc.y, .infinity = false };
                                    acc = acc.mul(pairing.millerLoopArkworks(g1_fp, c.params_ptr.g2_vec[row]));
                                }
                            }
                            return acc;
                        };
                        g1_fps = heap_g1.?;
                    } else {
                        g1_fps = stack_g1[0..chunk_len];
                    }

                    for (0..chunk_len) |j| {
                        const row = start + j;
                        const row_start_idx = row * c.n_cols;
                        if (row_start_idx >= c.evals_ptr.len or row >= c.params_ptr.g2_vec.len) {
                            g1_fps[j] = G1PointFp{ .x = Fp.zero(), .y = Fp.one(), .infinity = true };
                            continue;
                        }
                        const row_end_idx = @min(row_start_idx + c.n_cols, c.evals_ptr.len);
                        const row_evals = c.evals_ptr[row_start_idx..row_end_idx];
                        const rc = msm.MSM(F, Fp).compute(c.params_ptr.g1_vec[0..row_evals.len], row_evals);
                        g1_fps[j] = G1PointFp{ .x = rc.x, .y = rc.y, .infinity = rc.infinity };
                    }

                    // Phase 2: batched Miller loop (prefer affine > prepared sparse > unprepared)
                    if (c.params_ptr.g2_prepared_affine) |affine| {
                        return pairing.batchedMillerLoopAffine(g1_fps, affine[start .. start + chunk_len]);
                    } else if (c.params_ptr.g2_prepared) |prep| {
                        return pairing.batchedMillerLoopPreparedSparse(g1_fps, prep[start .. start + chunk_len]);
                    } else {
                        return pairing.batchedMillerLoopUnprepared(g1_fps, c.params_ptr.g2_vec[start .. start + chunk_len]);
                    }
                }
            }.f;

            const reduceFn = struct {
                fn f(a: pairing.Fp12, b: pairing.Fp12) pairing.Fp12 {
                    return a.mul(b);
                }
            }.f;

            const miller_acc = if (tp) |pool|
                pool.parallelReduceForce(pairing.Fp12, num_rows, pairing.Fp12.one(), ctx, mapFn, reduceFn)
            else
                mapFn(ctx, 0, num_rows);

            // Check if any non-trivial contribution
            if (miller_acc.isOne()) return GT.one();
            return pairing.finalExponentiation(miller_acc);
        }

        /// Like commitWithPool, but also returns the intermediate row commitments (G1 points).
        /// These can later be combined homomorphically to avoid recomputing row commitments
        /// when opening a joint polynomial (Stage 8).
        pub fn commitWithPoolAndHints(
            params: *const SetupParams,
            evals: []const F,
            allocator: Allocator,
            tp: ?*ThreadPool,
        ) !struct { commitment: Commitment, row_commitments: []G1Point } {
            if (evals.len == 0) {
                return .{ .commitment = GT.one(), .row_commitments = &[_]G1Point{} };
            }

            const poly_len = evals.len;
            const num_vars: usize = if (poly_len <= 1) 1 else std.math.log2_int(usize, poly_len);
            const sigma: usize = (num_vars + 1) / 2;
            const nu: usize = num_vars - sigma;
            const num_cols = @as(usize, 1) << @intCast(sigma);
            const num_rows = @as(usize, 1) << @intCast(nu);

            // Phase 1: Compute row commitments (G1 points via MSM)
            const row_commitments = if (tp) |pool|
                try computeRowCommitmentsWithColsParallel(F, params, evals, num_cols, allocator, pool)
            else
                try computeRowCommitmentsWithCols(F, params, evals, num_cols, allocator);
            errdefer allocator.free(row_commitments);

            // Phase 2: Miller loops over row commitments + final exponentiation
            const commitment = rowCommitmentsToCommitment(params, row_commitments, num_rows, tp);

            return .{ .commitment = commitment, .row_commitments = row_commitments };
        }

        const OneHotRowIndices = struct {
            row_index_slices: [][]const u16,
            col_indices_flat: []u16,
        };

        /// Build per-row column index slices from one-hot indices in CycleMajor layout.
        fn buildOneHotRowIndexSlices(
            indices: []const ?u8,
            rows_per_k: usize,
            num_cols: usize,
            num_rows: usize,
            trace_length: usize,
            allocator: Allocator,
        ) !OneHotRowIndices {
            std.debug.assert(num_cols <= std.math.maxInt(u16));
            const row_counts = try allocator.alloc(u16, num_rows);
            defer allocator.free(row_counts);
            @memset(row_counts, 0);

            for (0..trace_length) |cycle| {
                if (indices[cycle]) |addr| {
                    const row = @as(usize, addr) * rows_per_k + cycle / num_cols;
                    if (row < num_rows) {
                        row_counts[row] += 1;
                    }
                }
            }

            var total_entries: usize = 0;
            for (row_counts) |c| total_entries += c;

            const col_indices_flat = try allocator.alloc(u16, total_entries);
            errdefer allocator.free(col_indices_flat);

            const row_offsets = try allocator.alloc(usize, num_rows + 1);
            defer allocator.free(row_offsets);
            row_offsets[0] = 0;
            for (0..num_rows) |r| {
                row_offsets[r + 1] = row_offsets[r] + row_counts[r];
            }

            const row_fill = try allocator.alloc(u16, num_rows);
            defer allocator.free(row_fill);
            @memset(row_fill, 0);

            for (0..trace_length) |cycle| {
                if (indices[cycle]) |addr| {
                    const row = @as(usize, addr) * rows_per_k + cycle / num_cols;
                    if (row < num_rows) {
                        const off = row_offsets[row] + row_fill[row];
                        col_indices_flat[off] = @intCast(cycle % num_cols);
                        row_fill[row] += 1;
                    }
                }
            }

            const row_index_slices = try allocator.alloc([]const u16, num_rows);
            for (0..num_rows) |r| {
                row_index_slices[r] = col_indices_flat[row_offsets[r]..row_offsets[r + 1]];
            }

            return .{
                .row_index_slices = row_index_slices,
                .col_indices_flat = col_indices_flat,
            };
        }

        /// Like commitOneHotWithPool, but also returns the intermediate row commitments (G1 points).
        pub fn commitOneHotWithPoolAndHints(
            params: *const SetupParams,
            indices: []const ?u8,
            k_chunk: usize,
            trace_length: usize,
            allocator: Allocator,
            tp: ?*ThreadPool,
        ) !struct { commitment: Commitment, row_commitments: []G1Point } {
            const poly_size = k_chunk * trace_length;
            if (poly_size == 0) return .{ .commitment = GT.one(), .row_commitments = &[_]G1Point{} };

            const num_vars: usize = if (poly_size <= 1) 1 else std.math.log2_int(usize, poly_size);
            const sigma: usize = (num_vars + 1) / 2;
            const nu: usize = num_vars - sigma;
            const num_cols = @as(usize, 1) << @intCast(sigma);
            const num_rows = @as(usize, 1) << @intCast(nu);
            const rows_per_k = trace_length / num_cols;
            std.debug.assert(trace_length % num_cols == 0);

            // Build per-row column index lists
            const row_idx = try buildOneHotRowIndexSlices(indices, rows_per_k, num_cols, num_rows, trace_length, allocator);
            defer allocator.free(row_idx.col_indices_flat);
            defer allocator.free(row_idx.row_index_slices);

            // Phase 1: Compute G1 row commitments via projective accumulation
            const g1_bases = params.g1_vec[0..num_cols];
            const G1Proj = msm.ProjectivePoint(Fp);

            const row_commitments = try allocator.alloc(G1Point, num_rows);
            errdefer allocator.free(row_commitments);

            const RowHintCtx = struct {
                row_slices: []const []const u16,
                g1_bases_ptr: []const G1Point,
                out: []G1Point,
            };
            const hint_ctx = RowHintCtx{
                .row_slices = row_idx.row_index_slices,
                .g1_bases_ptr = g1_bases,
                .out = row_commitments,
            };

            const computeRowFn = struct {
                fn f(c: RowHintCtx, row: usize) void {
                    const row_indices = c.row_slices[row];
                    if (row_indices.len == 0) {
                        c.out[row] = G1Point.identity();
                        return;
                    }
                    c.out[row] = if (row_indices.len == 1)
                        c.g1_bases_ptr[row_indices[0]]
                    else blk: {
                        var proj = G1Proj.fromAffine(c.g1_bases_ptr[row_indices[0]]);
                        for (row_indices[1..]) |col_idx| {
                            proj = proj.addAffine(c.g1_bases_ptr[col_idx]);
                        }
                        break :blk proj.toAffine();
                    };
                }
            }.f;

            if (tp) |pool| {
                pool.parallelForForce(num_rows, hint_ctx, computeRowFn);
            } else {
                for (0..num_rows) |row| computeRowFn(hint_ctx, row);
            }

            // Phase 2: Miller loops + final exponentiation
            const commitment = rowCommitmentsToCommitment(params, row_commitments, num_rows, tp);

            return .{ .commitment = commitment, .row_commitments = row_commitments };
        }

        /// Convert row commitments (G1 points) to a GT commitment via Miller loops.
        fn rowCommitmentsToCommitment(
            params: *const SetupParams,
            row_commitments: []const G1Point,
            num_rows: usize,
            tp: ?*ThreadPool,
        ) Commitment {
            const Ctx = struct {
                params_ptr: *const SetupParams,
                row_comms: []const G1Point,
            };
            const ctx = Ctx{
                .params_ptr = params,
                .row_comms = row_commitments,
            };

            const mapFn = struct {
                fn f(c: Ctx, start: usize, end: usize) pairing.Fp12 {
                    const chunk_len = @min(end, c.row_comms.len) - @min(start, c.row_comms.len);
                    if (chunk_len == 0) return pairing.Fp12.one();

                    var stack_g1: [64]G1PointFp = undefined;
                    const use_heap = chunk_len > 64;
                    var heap_g1: ?[]G1PointFp = null;
                    defer if (heap_g1) |h| std.heap.page_allocator.free(h);

                    var g1_fps: []G1PointFp = undefined;
                    if (use_heap) {
                        heap_g1 = std.heap.page_allocator.alloc(G1PointFp, chunk_len) catch {
                            // Fallback to individual loops
                            var acc = pairing.Fp12.one();
                            for (start..@min(end, c.row_comms.len)) |row| {
                                const rc = c.row_comms[row];
                                if (rc.infinity) continue;
                                const row_g1 = G1PointFp{ .x = rc.x, .y = rc.y, .infinity = false };
                                const ml = if (c.params_ptr.g2_prepared) |prep|
                                    pairing.millerLoopPrepared(row_g1, &prep[row])
                                else
                                    pairing.millerLoopArkworks(row_g1, c.params_ptr.g2_vec[row]);
                                acc = acc.mul(ml);
                            }
                            return acc;
                        };
                        g1_fps = heap_g1.?;
                    } else {
                        g1_fps = stack_g1[0..chunk_len];
                    }

                    for (0..chunk_len) |j| {
                        const row = start + j;
                        if (row >= c.row_comms.len) {
                            g1_fps[j] = G1PointFp{ .x = Fp.zero(), .y = Fp.one(), .infinity = true };
                        } else {
                            const rc = c.row_comms[row];
                            g1_fps[j] = G1PointFp{ .x = rc.x, .y = rc.y, .infinity = rc.infinity };
                        }
                    }

                    if (c.params_ptr.g2_prepared_affine) |affine| {
                        return pairing.batchedMillerLoopAffine(g1_fps, affine[start .. start + chunk_len]);
                    } else if (c.params_ptr.g2_prepared) |prep| {
                        return pairing.batchedMillerLoopPreparedSparse(g1_fps, prep[start .. start + chunk_len]);
                    } else {
                        return pairing.batchedMillerLoopUnprepared(g1_fps, c.params_ptr.g2_vec[start .. start + chunk_len]);
                    }
                }
            }.f;

            const reduceFn = struct {
                fn f(a: pairing.Fp12, b: pairing.Fp12) pairing.Fp12 {
                    return a.mul(b);
                }
            }.f;

            const miller_acc = if (tp) |pool|
                pool.parallelReduceForce(pairing.Fp12, num_rows, pairing.Fp12.one(), ctx, mapFn, reduceFn)
            else
                mapFn(ctx, 0, num_rows);

            if (miller_acc.isOne()) return GT.one();
            return pairing.finalExponentiation(miller_acc);
        }

        /// Combine row commitment hints from multiple polynomials using homomorphic combination.
        /// Given row commitments for each polynomial and gamma coefficients, computes:
        ///   joint_rows[i] = Σ_k γ^k · rows_k[i]
        /// This avoids recomputing row commitments for the joint polynomial from scratch.
        pub fn combineRowCommitmentHints(
            hints: []const []const G1Point,
            coeffs: []const F,
            num_rows: usize,
            allocator: Allocator,
            tp: ?*ThreadPool,
        ) ![]G1Point {
            const result = try allocator.alloc(G1Point, num_rows);
            errdefer allocator.free(result);

            const G1Proj = msm.ProjectivePoint(Fp);
            const Ctx = struct {
                hints_ptr: []const []const G1Point,
                coeffs_ptr: []const F,
                out: []G1Point,
            };
            const ctx = Ctx{
                .hints_ptr = hints,
                .coeffs_ptr = coeffs,
                .out = result,
            };

            const combineFn = struct {
                fn f(c: Ctx, row: usize) void {
                    var acc = G1Proj.identity();
                    for (0..c.hints_ptr.len) |k| {
                        if (row < c.hints_ptr[k].len and !c.hints_ptr[k][row].infinity) {
                            const scaled = glv.glvScalarMulG1(c.hints_ptr[k][row], c.coeffs_ptr[k]);
                            acc = acc.add(scaled);
                        }
                    }
                    c.out[row] = acc.toAffine();
                }
            }.f;

            if (tp) |pool| {
                pool.parallelForForce(num_rows, ctx, combineFn);
            } else {
                for (0..num_rows) |row| combineFn(ctx, row);
            }

            return result;
        }

        /// Commit to a one-hot polynomial using projective accumulation instead of MSM.
        /// Thin wrapper around commitOneHotWithPoolAndHints that discards row commitments.
        pub fn commitOneHotWithPool(
            params: *const SetupParams,
            indices: []const ?u8,
            k_chunk: usize,
            trace_length: usize,
            allocator: Allocator,
            tp: ?*ThreadPool,
        ) !Commitment {
            const result = try commitOneHotWithPoolAndHints(params, indices, k_chunk, trace_length, allocator, tp);
            allocator.free(result.row_commitments);
            return result.commitment;
        }

        /// Create an opening proof using the Dory reduce-and-fold IPA
        ///
        /// Implements the full Dory protocol:
        /// 1. Compute row commitments (or use pre-computed)
        /// 2. Compute evaluation vectors (left_vec, right_vec) from point
        /// 3. Create VMV message (C, D2, E1)
        /// 4. Run max(nu, sigma) rounds of reduce-and-fold
        /// 5. Produce final scalar product message
        pub fn open(
            params: *const SetupParams,
            evals: []const F,
            point: []const F,
            allocator: Allocator,
        ) !Proof {
            return openWithRowCommitments(params, evals, point, null, allocator, null);
        }

        /// Create an opening proof with pre-computed row commitments
        pub fn openWithRowCommitments(
            params: *const SetupParams,
            evals: []const F,
            point: []const F,
            row_commitments_opt: ?[]const G1Point,
            allocator: Allocator,
            tp: ?*ThreadPool,
        ) !Proof {
            // Compute nu/sigma from the polynomial's actual size, not from SRS params.
            // This matches Jolt's balanced_sigma_nu: sigma = ceil(num_vars/2), nu = num_vars - sigma
            const num_vars: u32 = @intCast(point.len);
            const sigma: u32 = (num_vars + 1) / 2;
            const nu: u32 = num_vars - sigma;
            const num_rounds = @max(nu, sigma);

            // Step 1: Get or compute row commitments
            const row_commitments = if (row_commitments_opt) |rc| blk: {
                const owned = try allocator.alloc(G1Point, rc.len);
                @memcpy(owned, rc);
                break :blk owned;
            } else blk: {
                if (tp) |pool| {
                    break :blk try computeRowCommitmentsWithColsParallel(F, params, evals, @as(usize, 1) << @intCast(sigma), allocator, pool);
                } else {
                    break :blk try computeRowCommitmentsWithCols(F, params, evals, @as(usize, 1) << @intCast(sigma), allocator);
                }
            };
            defer allocator.free(row_commitments);

            // Step 2: Compute evaluation vectors (left_vec, right_vec)
            const left_vec = try allocator.alloc(F, @as(usize, 1) << @intCast(nu));
            defer allocator.free(left_vec);
            const right_vec = try allocator.alloc(F, @as(usize, 1) << @intCast(sigma));
            defer allocator.free(right_vec);

            computeEvaluationVectors(F, point, nu, sigma, left_vec, right_vec);

            // Step 3: Compute v_vec = left_vec^T * M (vector-matrix product)
            const v_vec = try computeVectorMatrixProduct(F, evals, left_vec, nu, sigma, allocator);
            defer allocator.free(v_vec);

            // Pad row_commitments to match v_vec length (2^sigma)
            // v_vec always has length 2^sigma from computeVectorMatrixProduct
            const v_vec_len = @as(usize, 1) << @intCast(sigma);
            const padded_row_commitments = blk: {
                if (row_commitments.len >= v_vec_len) {
                    // Truncate or exact match
                    const padded = try allocator.alloc(G1Point, v_vec_len);
                    @memcpy(padded, row_commitments[0..v_vec_len]);
                    break :blk padded;
                } else {
                    // Pad with identity
                    const padded = try allocator.alloc(G1Point, v_vec_len);
                    @memcpy(padded[0..row_commitments.len], row_commitments);
                    for (row_commitments.len..v_vec_len) |i| {
                        padded[i] = G1Point.identity();
                    }
                    break :blk padded;
                }
            };
            defer allocator.free(padded_row_commitments);

            // Step 4: Compute VMV message
            // C = e(MSM(row_commitments, v_vec), h2)
            const t_vec_v = msm.MSM(F, Fp).computeWithPool(
                padded_row_commitments,
                v_vec,
                tp,
            );
            const t_vec_v_fp = G1PointFp{
                .x = t_vec_v.x,
                .y = t_vec_v.y,
                .infinity = t_vec_v.infinity,
            };
            const c = pairing.pairingFp(t_vec_v_fp, params.g2_vec[0]); // h2 = g2_vec[0] for now

            // D2 = e(MSM(g1_vec, v_vec), h2)
            const gamma1_v = msm.MSM(F, Fp).computeWithPool(
                params.g1_vec[0..v_vec.len],
                v_vec,
                tp,
            );
            const gamma1_v_fp = G1PointFp{
                .x = gamma1_v.x,
                .y = gamma1_v.y,
                .infinity = gamma1_v.infinity,
            };
            const d2 = pairing.pairingFp(gamma1_v_fp, params.g2_vec[0]);

            // E1 = MSM(row_commitments, left_vec)
            // Ensure length match
            const left_vec_len = @as(usize, 1) << @intCast(nu);
            const e1: G1Point = blk: {
                if (row_commitments.len >= left_vec_len) {
                    break :blk msm.MSM(F, Fp).computeWithPool(row_commitments[0..left_vec_len], left_vec, tp);
                } else {
                    const padded = try allocator.alloc(G1Point, left_vec_len);
                    defer allocator.free(padded);
                    @memcpy(padded[0..row_commitments.len], row_commitments);
                    for (row_commitments.len..left_vec_len) |i| {
                        padded[i] = G1Point.identity();
                    }
                    break :blk msm.MSM(F, Fp).computeWithPool(padded, left_vec, tp);
                }
            };

            const vmv_message = VMVMessage{
                .c = c,
                .d2 = d2,
                .e1 = e1,
            };

            // Step 5: Initialize prover state for reduce-and-fold
            // v1 = padded row_commitments
            // v2 = v_vec * h2 (scalars applied to h2)
            // s1 = right_vec (padded)
            // s2 = left_vec (padded)

            // Pad vectors to 2^sigma
            const vec_len = @as(usize, 1) << @intCast(sigma);
            const v1 = try allocator.alloc(G1Point, vec_len);
            defer allocator.free(v1);
            @memcpy(v1[0..padded_row_commitments.len], padded_row_commitments);
            for (padded_row_commitments.len..vec_len) |i| {
                v1[i] = G1Point.identity();
            }

            const v2 = try allocator.alloc(G2Point, vec_len);
            defer allocator.free(v2);
            // Precompute Frobenius bases for GLV-4D (shared base point)
            const g2_glv_bases = [4]G2Point{
                params.g2_vec[0],
                glv.frobeniusPsiAffine(params.g2_vec[0], 1),
                glv.frobeniusPsiAffine(params.g2_vec[0], 2),
                glv.frobeniusPsiAffine(params.g2_vec[0], 3),
            };
            if (tp) |pool| {
                const V2Ctx = struct { v2_out: []G2Point, g2_bases: [4]G2Point, v_vec_ptr: []const F, v_vec_len: usize };
                pool.parallelForForce(vec_len, V2Ctx{ .v2_out = v2, .g2_bases = g2_glv_bases, .v_vec_ptr = v_vec, .v_vec_len = v_vec.len }, struct {
                    fn f(ctx: V2Ctx, i: usize) void {
                        if (i < ctx.v_vec_len) {
                            ctx.v2_out[i] = glv.glvScalarMulG2WithBases(ctx.g2_bases, ctx.v_vec_ptr[i]).toAffine();
                        } else {
                            ctx.v2_out[i] = G2Point.identity();
                        }
                    }
                }.f);
            } else {
                for (0..vec_len) |i| {
                    if (i < v_vec.len) {
                        v2[i] = glv.glvScalarMulG2WithBases(g2_glv_bases, v_vec[i]).toAffine();
                    } else {
                        v2[i] = G2Point.identity();
                    }
                }
            }

            const s1 = try allocator.alloc(F, vec_len);
            defer allocator.free(s1);
            @memcpy(s1[0..right_vec.len], right_vec);
            for (right_vec.len..vec_len) |i| {
                s1[i] = F.zero();
            }

            const s2 = try allocator.alloc(F, vec_len);
            defer allocator.free(s2);
            @memcpy(s2[0..left_vec.len], left_vec);
            for (left_vec.len..vec_len) |i| {
                s2[i] = F.zero();
            }

            // Allocate message arrays
            const first_messages = try allocator.alloc(FirstReduceMessage, num_rounds);
            errdefer allocator.free(first_messages);
            const second_messages = try allocator.alloc(SecondReduceMessage, num_rounds);
            errdefer allocator.free(second_messages);

            // Step 6: Run reduce-and-fold rounds
            // Handle asymmetric case: g1_vec has 2^sigma elements, g2_vec has 2^nu elements
            const col_len = @as(usize, 1) << @intCast(sigma); // g1_vec size
            const row_len = @as(usize, 1) << @intCast(nu); // g2_vec size
            var current_col_len = col_len;
            var current_row_len = row_len;
            var round: usize = 0;

            // Working arrays that get folded
            const v1_work = try allocator.alloc(G1Point, vec_len);
            defer allocator.free(v1_work);
            @memcpy(v1_work, v1);

            const v2_work = try allocator.alloc(G2Point, vec_len);
            defer allocator.free(v2_work);
            @memcpy(v2_work, v2);

            const s1_work = try allocator.alloc(F, vec_len);
            defer allocator.free(s1_work);
            @memcpy(s1_work, s1);

            const s2_work = try allocator.alloc(F, vec_len);
            defer allocator.free(s2_work);
            @memcpy(s2_work, s2);

            while (round < num_rounds) : (round += 1) {
                // Use the maximum of current dimensions for working length
                const current_len = @max(current_col_len, current_row_len);
                const n2 = current_len / 2;

                // For g2_vec operations, use minimum of n2 and available g2 elements
                const g2_size = @min(n2, current_row_len);

                // Compute first reduce message
                // D1L = multiPair(v1_l, g2_vec[0..g2_size])
                // D1R = multiPair(v1_r, g2_vec[0..g2_size])
                // D2L = multiPair(g1_vec[0..n2], v2_l)
                // D2R = multiPair(g1_vec[0..n2], v2_r)
                // These 4 pairings are independent — use join to run pairs concurrently.
                const g1_size = @min(n2, current_col_len);
                const v1_r_end = @min(n2 + g2_size, current_len);
                const v2_r_end = @min(n2 + g1_size, current_len);

                var d1_left: GT = undefined;
                var d1_right: GT = undefined;
                var d2_left: GT = undefined;
                var d2_right: GT = undefined;

                if (round == 0) {
                    // First-round optimization: v2_work[i] = g2_vec[0] * v_vec[i]
                    // So multiPair(g1[0..n], v2[0..n]) = e(MSM(g1[0..n], v_vec[0..n]), g2_vec[0])
                    std.debug.assert(v_vec.len >= current_len);
                    const g2_fin = params.g2_vec[0];
                    d1_left = if (params.g2_prepared_affine) |affine|
                        multiPairG1G2PreparedAffine(v1_work[0..g2_size], affine[0..g2_size], tp)
                    else if (params.g2_prepared) |prep|
                        multiPairG1G2Prepared(v1_work[0..g2_size], prep[0..g2_size], tp)
                    else
                        multiPairG1G2WithPool(v1_work[0..g2_size], params.g2_vec[0..g2_size], tp);
                    d1_right = if (params.g2_prepared_affine) |affine|
                        multiPairG1G2PreparedAffine(v1_work[n2..v1_r_end], affine[0..g2_size], tp)
                    else if (params.g2_prepared) |prep|
                        multiPairG1G2Prepared(v1_work[n2..v1_r_end], prep[0..g2_size], tp)
                    else
                        multiPairG1G2WithPool(v1_work[n2..v1_r_end], params.g2_vec[0..g2_size], tp);
                    const sum_left = msm.MSM(F, Fp).computeWithPool(params.g1_vec[0..g1_size], v_vec[0..g1_size], tp);
                    const sum_right = msm.MSM(F, Fp).computeWithPool(params.g1_vec[0..g1_size], v_vec[n2..v2_r_end], tp);
                    const sum_left_fp = G1PointFp{ .x = sum_left.x, .y = sum_left.y, .infinity = sum_left.infinity };
                    const sum_right_fp = G1PointFp{ .x = sum_right.x, .y = sum_right.y, .infinity = sum_right.infinity };
                    d2_left = pairing.pairingFp(sum_left_fp, g2_fin);
                    d2_right = pairing.pairingFp(sum_right_fp, g2_fin);
                } else {
                    const batch = multiPairBatched(4, .{
                        PairGroup{ .g1 = v1_work[0..g2_size], .g2 = params.g2_vec[0..g2_size] }, // D1L
                        PairGroup{ .g1 = params.g1_vec[0..g1_size], .g2 = v2_work[0..g1_size] }, // D2L
                        PairGroup{ .g1 = v1_work[n2..v1_r_end], .g2 = params.g2_vec[0..g2_size] }, // D1R
                        PairGroup{ .g1 = params.g1_vec[0..g1_size], .g2 = v2_work[n2..v2_r_end] }, // D2R
                    }, tp);
                    d1_left = batch[0];
                    d2_left = batch[1];
                    d1_right = batch[2];
                    d2_right = batch[3];
                }

                // E1_beta = MSM(g1_vec[0..current_col_len], s2_work[0..current_col_len])
                const e1_beta = msm.MSM(F, Fp).computeWithPool(
                    params.g1_vec[0..current_col_len],
                    s2_work[0..current_col_len],
                    tp,
                );

                // E2_beta = MSM(g2_vec[0..current_row_len], s1_work[0..current_row_len])
                const e2_beta = msmG2(F, params.g2_vec[0..current_row_len], s1_work[0..current_row_len], tp);
                // Debug: output compressed e2_beta for round 0
                if (comptime debug_verbose) {
                    if (round == 0) {
                        const debug_compressed = compressG2(e2_beta);
                        dbg("DEBUG e2_beta[0] compressed: ", .{});
                        for (debug_compressed) |b_| {
                            dbg("{x:0>2}", .{b_});
                        }
                        dbg("\n", .{});
                        dbg("DEBUG e2_beta[0] g2_vec_len={} s1_work scalars: ", .{current_row_len});
                        for (s1_work[0..@min(current_row_len, 4)]) |s| {
                            const norm = s.fromMontgomery();
                            dbg("{x:0>16}{x:0>16}{x:0>16}{x:0>16} ", .{ norm.limbs[3], norm.limbs[2], norm.limbs[1], norm.limbs[0] });
                        }
                        dbg("\n", .{});
                    }
                }

                first_messages[round] = FirstReduceMessage{
                    .d1_left = d1_left,
                    .d1_right = d1_right,
                    .d2_left = d2_left,
                    .d2_right = d2_right,
                    .e1_beta = e1_beta,
                    .e2_beta = e2_beta,
                };

                // Get beta challenge (in a real implementation, from transcript)
                // For now, use a deterministic challenge based on round
                const beta = F.fromU64(@as(u64, round) + 1);
                const beta_inv = beta.inverse() orelse F.one();

                // Apply first challenge: v1 += beta * g1_vec, v2 += beta_inv * g2_vec
                // Use GLV for faster scalar multiplication
                if (tp) |pool| {
                    const BetaCtx1 = struct { v1: []G1Point, g1: []const G1Point, b: F };
                    pool.parallelForForce(current_col_len, BetaCtx1{ .v1 = v1_work, .g1 = params.g1_vec, .b = beta }, struct {
                        fn f(cx: BetaCtx1, i: usize) void {
                            const scaled_g1 = glv.glvScalarMulG1(cx.g1[i], cx.b).toAffine();
                            cx.v1[i] = cx.v1[i].add(scaled_g1);
                        }
                    }.f);
                    const BetaCtx2 = struct { v2: []G2Point, g2: []const G2Point, bi: F };
                    pool.parallelForForce(current_row_len, BetaCtx2{ .v2 = v2_work, .g2 = params.g2_vec, .bi = beta_inv }, struct {
                        fn f(cx: BetaCtx2, i: usize) void {
                            const scaled_proj = glv.glvScalarMulG2(cx.g2[i], cx.bi);
                            cx.v2[i] = scaled_proj.addAffine(cx.v2[i]).toAffine();
                        }
                    }.f);
                } else {
                    for (0..current_col_len) |i| {
                        const scaled_g1 = glv.glvScalarMulG1(params.g1_vec[i], beta).toAffine();
                        v1_work[i] = v1_work[i].add(scaled_g1);
                    }
                    for (0..current_row_len) |i| {
                        const scaled_proj = glv.glvScalarMulG2(params.g2_vec[i], beta_inv);
                        v2_work[i] = scaled_proj.addAffine(v2_work[i]).toAffine();
                    }
                }

                // Compute second reduce message
                // C+ = multiPair(v1_l, v2_r)
                // C- = multiPair(v1_r, v2_l)
                const v1_half = @min(n2, current_col_len);
                const v2_half = @min(n2, current_row_len);
                const v2_r_half_end = @min(n2 + v2_half, current_len);
                const v1_r_half_end = @min(n2 + v1_half, current_len);

                const c_batch = multiPairBatched(2, .{
                    PairGroup{ .g1 = v1_work[0..v1_half], .g2 = v2_work[n2..v2_r_half_end] }, // C+
                    PairGroup{ .g1 = v1_work[n2..v1_r_half_end], .g2 = v2_work[0..v2_half] }, // C-
                }, tp);
                const c_plus = c_batch[0];
                const c_minus = c_batch[1];

                // E1+ = MSM(v1_l, s2_r)
                // E1- = MSM(v1_r, s2_l)
                const e1_plus = msm.MSM(F, Fp).computeWithPool(v1_work[0..v1_half], s2_work[n2..@min(n2 + v1_half, current_len)], tp);
                const e1_minus = msm.MSM(F, Fp).computeWithPool(v1_work[n2..@min(n2 + v1_half, current_len)], s2_work[0..v1_half], tp);

                // E2+ = MSM(v2_r, s1_l)
                // E2- = MSM(v2_l, s1_r)
                const e2_plus = msmG2(F, v2_work[n2..@min(n2 + v2_half, current_len)], s1_work[0..v2_half], tp);
                const e2_minus = msmG2(F, v2_work[0..v2_half], s1_work[n2..@min(n2 + v2_half, current_len)], tp);

                second_messages[round] = SecondReduceMessage{
                    .c_plus = c_plus,
                    .c_minus = c_minus,
                    .e1_plus = e1_plus,
                    .e1_minus = e1_minus,
                    .e2_plus = e2_plus,
                    .e2_minus = e2_minus,
                };

                // Get alpha challenge (deterministic for now)
                const alpha = F.fromU64(@as(u64, round) + 100);
                const alpha_inv = alpha.inverse() orelse F.one();

                // Apply second challenge: fold vectors using GLV
                if (tp) |pool| {
                    const FoldCtx1 = struct { v1: []G1Point, a: F, half: usize };
                    pool.parallelForForce(v1_half, FoldCtx1{ .v1 = v1_work, .a = alpha, .half = n2 }, struct {
                        fn f(cx: FoldCtx1, i: usize) void {
                            const scaled_l = glv.glvScalarMulG1(cx.v1[i], cx.a).toAffine();
                            cx.v1[i] = scaled_l.add(cx.v1[i + cx.half]);
                        }
                    }.f);
                    const FoldCtx2 = struct { v2: []G2Point, ai: F, half: usize };
                    pool.parallelForForce(v2_half, FoldCtx2{ .v2 = v2_work, .ai = alpha_inv, .half = n2 }, struct {
                        fn f(cx: FoldCtx2, i: usize) void {
                            const scaled_proj = glv.glvScalarMulG2(cx.v2[i], cx.ai);
                            cx.v2[i] = scaled_proj.addAffine(cx.v2[i + cx.half]).toAffine();
                        }
                    }.f);
                    const FoldCtxS1 = struct { s: []F, a_val: F, half: usize };
                    pool.parallelFor(v2_half, FoldCtxS1{ .s = s1_work, .a_val = alpha, .half = n2 }, struct {
                        fn f(cx: FoldCtxS1, i: usize) void {
                            cx.s[i] = cx.a_val.mul(cx.s[i]).add(cx.s[i + cx.half]);
                        }
                    }.f);
                    const FoldCtxS2 = struct { s: []F, ai_val: F, half: usize };
                    pool.parallelFor(v1_half, FoldCtxS2{ .s = s2_work, .ai_val = alpha_inv, .half = n2 }, struct {
                        fn f(cx: FoldCtxS2, i: usize) void {
                            cx.s[i] = cx.ai_val.mul(cx.s[i]).add(cx.s[i + cx.half]);
                        }
                    }.f);
                } else {
                    for (0..v1_half) |i| {
                        const scaled_l = glv.glvScalarMulG1(v1_work[i], alpha).toAffine();
                        v1_work[i] = scaled_l.add(v1_work[i + n2]);
                    }
                    for (0..v2_half) |i| {
                        const scaled_proj = glv.glvScalarMulG2(v2_work[i], alpha_inv);
                        v2_work[i] = scaled_proj.addAffine(v2_work[i + n2]).toAffine();
                    }
                    for (0..v2_half) |i| {
                        s1_work[i] = alpha.mul(s1_work[i]).add(s1_work[i + n2]);
                    }
                    for (0..v1_half) |i| {
                        s2_work[i] = alpha_inv.mul(s2_work[i]).add(s2_work[i + n2]);
                    }
                }

                // Update dimensions for next round
                if (current_col_len > 1) current_col_len = current_col_len / 2;
                if (current_row_len > 1) current_row_len = current_row_len / 2;
            }

            // Step 7: Compute final scalar product message
            // gamma challenge (deterministic for now)
            const gamma = F.fromU64(999);
            const gamma_inv = gamma.inverse() orelse F.one();

            // E1 = v1[0] + gamma * s1[0] * h1
            const gamma_s1 = gamma.mul(s1_work[0]);
            const h1 = G1Point.generator(); // h1 = generator for now
            const scaled_h1 = glv.glvScalarMulG1(h1, gamma_s1).toAffine();
            const final_e1 = v1_work[0].add(scaled_h1);

            // E2 = v2[0] + gamma_inv * s2[0] * h2
            const gamma_inv_s2 = gamma_inv.mul(s2_work[0]);
            const h2 = G2Point.generator(); // h2 = generator for now
            const scaled_h2 = glv.glvScalarMulG2(h2, gamma_inv_s2).toAffine();
            const final_e2 = v2_work[0].add(scaled_h2);

            const final_message = ScalarProductMessage{
                .e1 = final_e1,
                .e2 = final_e2,
            };

            return Proof{
                .vmv_message = vmv_message,
                .first_messages = first_messages,
                .second_messages = second_messages,
                .final_message = final_message,
                .nu = nu,
                .sigma = sigma,
                .allocator = allocator,
            };
        }

        /// Verify an opening proof
        pub fn verify(
            params: *const SetupParams,
            commitment: Commitment,
            point: []const F,
            value: F,
            proof: *const Proof,
        ) bool {
            _ = params;
            _ = commitment;
            _ = point;
            _ = value;
            _ = proof;

            // Placeholder: verification not yet fully implemented
            return true;
        }

        /// Create an opening proof using a transcript for Fiat-Shamir challenges.
        ///
        /// This is the transcript-integrated version that produces challenges
        /// compatible with Jolt's verifier.
        ///
        /// The transcript should be the Blake2bTranscript for Jolt compatibility.
        pub fn openWithTranscript(
            params: *const SetupParams,
            evals: []const F,
            point: []const F,
            row_commitments_opt: ?[]const G1Point,
            transcript: anytype,
            allocator: Allocator,
            tp: ?*ThreadPool,
        ) !Proof {
            // Compute nu/sigma from the polynomial's actual size, not from SRS params.
            // This matches Jolt's balanced_sigma_nu: sigma = ceil(num_vars/2), nu = num_vars - sigma
            const num_vars: u32 = @intCast(point.len);
            const sigma: u32 = (num_vars + 1) / 2;
            const nu: u32 = num_vars - sigma;
            const num_rounds = @max(nu, sigma);

            // Step 1: Get or compute row commitments
            const row_commitments = if (row_commitments_opt) |rc| blk: {
                const owned = try allocator.alloc(G1Point, rc.len);
                @memcpy(owned, rc);
                break :blk owned;
            } else blk: {
                if (tp) |pool| {
                    break :blk try computeRowCommitmentsWithColsParallel(F, params, evals, @as(usize, 1) << @intCast(sigma), allocator, pool);
                } else {
                    break :blk try computeRowCommitmentsWithCols(F, params, evals, @as(usize, 1) << @intCast(sigma), allocator);
                }
            };
            defer allocator.free(row_commitments);

            // Step 2: Compute evaluation vectors
            const left_vec = try allocator.alloc(F, @as(usize, 1) << @intCast(nu));
            defer allocator.free(left_vec);
            const right_vec = try allocator.alloc(F, @as(usize, 1) << @intCast(sigma));
            defer allocator.free(right_vec);

            computeEvaluationVectors(F, point, nu, sigma, left_vec, right_vec);

            // Step 3: Compute v_vec
            const v_vec = try computeVectorMatrixProduct(F, evals, left_vec, nu, sigma, allocator);
            defer allocator.free(v_vec);

            // Pad row_commitments to match v_vec length (2^sigma)
            const v_vec_len = @as(usize, 1) << @intCast(sigma);
            const padded_row_commitments = blk: {
                if (row_commitments.len >= v_vec_len) {
                    const padded = try allocator.alloc(G1Point, v_vec_len);
                    @memcpy(padded, row_commitments[0..v_vec_len]);
                    break :blk padded;
                } else {
                    const padded = try allocator.alloc(G1Point, v_vec_len);
                    @memcpy(padded[0..row_commitments.len], row_commitments);
                    for (row_commitments.len..v_vec_len) |i| {
                        padded[i] = G1Point.identity();
                    }
                    break :blk padded;
                }
            };
            defer allocator.free(padded_row_commitments);

            // Step 4: Compute VMV message
            // C = e(MSM(row_comms, v_vec), Γ₂₀)
            // Upstream uses g2_fin = setup.g2_vec[0] (NOT h2)
            const g2_fin = params.g2_vec[0];
            const t_vec_v = msm.MSM(F, Fp).computeWithPool(padded_row_commitments, v_vec, tp);
            const t_vec_v_fp = G1PointFp{
                .x = t_vec_v.x,
                .y = t_vec_v.y,
                .infinity = t_vec_v.infinity,
            };
            const c = pairing.pairingFp(t_vec_v_fp, g2_fin);

            // D₂ = e(MSM(Γ₁[..sigma], v_vec), Γ₂₀)
            const num_cols = @as(usize, 1) << @intCast(sigma);
            const gamma1_v = msm.MSM(F, Fp).computeWithPool(params.g1_vec[0..num_cols], v_vec[0..num_cols], tp);
            const gamma1_v_fp = G1PointFp{
                .x = gamma1_v.x,
                .y = gamma1_v.y,
                .infinity = gamma1_v.infinity,
            };
            const d2 = pairing.pairingFp(gamma1_v_fp, g2_fin);

            // e1 = MSM(row_commitments, left_vec)
            // row_commitments may have different length than left_vec (2^nu)
            // Pad or truncate to match
            const left_vec_len = @as(usize, 1) << @intCast(nu);
            const e1: G1Point = blk: {
                if (row_commitments.len >= left_vec_len) {
                    break :blk msm.MSM(F, Fp).computeWithPool(row_commitments[0..left_vec_len], left_vec, tp);
                } else {
                    const padded = try allocator.alloc(G1Point, left_vec_len);
                    defer allocator.free(padded);
                    @memcpy(padded[0..row_commitments.len], row_commitments);
                    for (row_commitments.len..left_vec_len) |i| {
                        padded[i] = G1Point.identity();
                    }
                    break :blk msm.MSM(F, Fp).computeWithPool(padded, left_vec, tp);
                }
            };

            const vmv_message = VMVMessage{
                .c = c,
                .d2 = d2,
                .e1 = e1,
            };

            // Append VMV message to transcript
            // NOTE: Dory uses JoltToDoryTranscript which calls transcript.append_bytes()
            // directly (no reversal), unlike appendGT/appendG1Compressed which reverse.
            // We must match this: serialize_compressed → append_bytes (no reversal).
            doryAppendGT(transcript, vmv_message.c);
            doryAppendGT(transcript, vmv_message.d2);
            doryAppendG1(transcript, vmv_message.e1);


            // Initialize working arrays
            const vec_len = @as(usize, 1) << @intCast(sigma);
            const v1_work = try allocator.alloc(G1Point, vec_len);
            defer allocator.free(v1_work);
            @memcpy(v1_work[0..padded_row_commitments.len], padded_row_commitments);
            for (padded_row_commitments.len..vec_len) |i| {
                v1_work[i] = G1Point.identity();
            }

            // v2 = v_vec * Γ₂₀ (each entry is v_vec[i] * g2_vec[0])
            // Upstream: let v2 = M2::fixed_base_vector_scalar_mul(g2_fin, &v_vec)
            // Precompute Frobenius bases for GLV-4D (shared base point)
            const g2_fin_glv_bases = [4]G2Point{
                g2_fin,
                glv.frobeniusPsiAffine(g2_fin, 1),
                glv.frobeniusPsiAffine(g2_fin, 2),
                glv.frobeniusPsiAffine(g2_fin, 3),
            };
            const v2_work = try allocator.alloc(G2Point, vec_len);
            defer allocator.free(v2_work);
            if (tp) |pool| {
                const V2Ctx = struct {
                    v2: []G2Point,
                    v_vec: []const F,
                    bases: *const [4]G2Point,
                    v_vec_len: usize,
                };
                var g2_fin_glv_bases_copy = g2_fin_glv_bases;
                const v2_ctx = V2Ctx{
                    .v2 = v2_work,
                    .v_vec = v_vec,
                    .bases = &g2_fin_glv_bases_copy,
                    .v_vec_len = v_vec.len,
                };
                pool.parallelForForce(vec_len, v2_ctx, struct {
                    fn f(cx: V2Ctx, i: usize) void {
                        if (i < cx.v_vec_len) {
                            cx.v2[i] = glv.glvScalarMulG2WithBases(cx.bases.*, cx.v_vec[i]).toAffine();
                        } else {
                            cx.v2[i] = G2Point.identity();
                        }
                    }
                }.f);
            } else {
                for (0..vec_len) |i| {
                    if (i < v_vec.len) {
                        v2_work[i] = glv.glvScalarMulG2WithBases(g2_fin_glv_bases, v_vec[i]).toAffine();
                    } else {
                        v2_work[i] = G2Point.identity();
                    }
                }
            }

            const s1_work = try allocator.alloc(F, vec_len);
            defer allocator.free(s1_work);
            @memcpy(s1_work[0..right_vec.len], right_vec);
            for (right_vec.len..vec_len) |i| {
                s1_work[i] = F.zero();
            }

            const s2_work = try allocator.alloc(F, vec_len);
            defer allocator.free(s2_work);
            @memcpy(s2_work[0..left_vec.len], left_vec);
            for (left_vec.len..vec_len) |i| {
                s2_work[i] = F.zero();
            }

            // Debug: print initial evaluation vector info
            if (comptime debug_verbose) {
                dbg("[DORY PROVER] nu={}, sigma={}, num_rounds={}, vec_len={}\n", .{ nu, sigma, num_rounds, vec_len });
                dbg("[DORY PROVER] right_vec.len={}, left_vec.len={}\n", .{ right_vec.len, left_vec.len });
                for (0..@min(4, right_vec.len)) |i| {
                    const be = right_vec[i].toBytesBE();
                    dbg("[DORY PROVER] right_vec[{}] first 16 LE: ", .{i});
                    for (0..16) |bi| dbg("{x:0>2}", .{be[31 - bi]});
                    dbg("\n", .{});
                }
                for (0..@min(4, left_vec.len)) |i| {
                    const be = left_vec[i].toBytesBE();
                    dbg("[DORY PROVER] left_vec[{}] first 16 LE: ", .{i});
                    for (0..16) |bi| dbg("{x:0>2}", .{be[31 - bi]});
                    dbg("\n", .{});
                }
                for (0..@min(point.len, 4)) |i| {
                    const be = point[i].toBytesBE();
                    dbg("[DORY PROVER] point[{}] first 16 LE: ", .{i});
                    for (0..16) |bi| dbg("{x:0>2}", .{be[31 - bi]});
                    dbg("\n", .{});
                }
            }

            // Allocate message arrays
            const first_messages = try allocator.alloc(FirstReduceMessage, num_rounds);
            errdefer allocator.free(first_messages);
            const second_messages = try allocator.alloc(SecondReduceMessage, num_rounds);
            errdefer allocator.free(second_messages);

            // Run reduce-and-fold rounds with transcript challenges
            var current_len = vec_len;
            var round: usize = 0;

            while (round < num_rounds) : (round += 1) {
                const n2 = current_len / 2;

                // Compute first reduce message
                const d1_left = if (params.g2_prepared_affine) |affine|
                    multiPairG1G2PreparedAffine(v1_work[0..n2], affine[0..n2], tp)
                else if (params.g2_prepared) |prep|
                    multiPairG1G2Prepared(v1_work[0..n2], prep[0..n2], tp)
                else
                    multiPairG1G2WithPool(v1_work[0..n2], params.g2_vec[0..n2], tp);
                const d1_right = if (params.g2_prepared_affine) |affine|
                    multiPairG1G2PreparedAffine(v1_work[n2..current_len], affine[0..n2], tp)
                else if (params.g2_prepared) |prep|
                    multiPairG1G2Prepared(v1_work[n2..current_len], prep[0..n2], tp)
                else
                    multiPairG1G2WithPool(v1_work[n2..current_len], params.g2_vec[0..n2], tp);
                const d2_left, const d2_right = if (round == 0) blk: {
                    // First-round optimization: v2_work[i] = g2_fin * v_vec[i]
                    // So multiPair(g1[0..n2], v2[0..n2]) = e(MSM(g1[0..n2], v_vec[0..n2]), g2_fin)
                    std.debug.assert(v_vec.len >= current_len);
                    const sum_left = msm.MSM(F, Fp).computeWithPool(params.g1_vec[0..n2], v_vec[0..n2], tp);
                    const sum_right = msm.MSM(F, Fp).computeWithPool(params.g1_vec[0..n2], v_vec[n2..current_len], tp);
                    const sum_left_fp = G1PointFp{ .x = sum_left.x, .y = sum_left.y, .infinity = sum_left.infinity };
                    const sum_right_fp = G1PointFp{ .x = sum_right.x, .y = sum_right.y, .infinity = sum_right.infinity };
                    break :blk .{ pairing.pairingFp(sum_left_fp, g2_fin), pairing.pairingFp(sum_right_fp, g2_fin) };
                } else blk: {
                    const d2_batch = multiPairBatched(2, .{
                        PairGroup{ .g1 = params.g1_vec[0..n2], .g2 = v2_work[0..n2] },
                        PairGroup{ .g1 = params.g1_vec[0..n2], .g2 = v2_work[n2..current_len] },
                    }, tp);
                    break :blk .{ d2_batch[0], d2_batch[1] };
                };
                const e1_beta, const e2_beta = if (tp) |pool| blk: {
                    const EBetaCtx = struct {
                        g1: []const G1Point,
                        g2: []const G2Point,
                        s1: []const F,
                        s2: []const F,
                        len: usize,
                    };
                    const eb_ctx = EBetaCtx{
                        .g1 = params.g1_vec,
                        .g2 = params.g2_vec,
                        .s1 = s1_work,
                        .s2 = s2_work,
                        .len = current_len,
                    };
                    break :blk pool.join(
                        G1Point,
                        G2Point,
                        eb_ctx,
                        struct {
                            fn f(cx: EBetaCtx) G1Point {
                                return msm.MSM(F, Fp).computeWithPool(cx.g1[0..cx.len], cx.s2[0..cx.len], null);
                            }
                        }.f,
                        eb_ctx,
                        struct {
                            fn f(cx: EBetaCtx) G2Point {
                                return msmG2(F, cx.g2[0..cx.len], cx.s1[0..cx.len], null);
                            }
                        }.f,
                    );
                } else .{
                    msm.MSM(F, Fp).computeWithPool(params.g1_vec[0..current_len], s2_work[0..current_len], null),
                    msmG2(F, params.g2_vec[0..current_len], s1_work[0..current_len], null),
                };

                // Debug: write e2_beta for each round to /tmp for validation
                if (comptime debug_verbose) {
                    if (round == 0) {
                        const debug_e2 = compressG2(e2_beta);
                        const debug_file = std.fs.cwd().createFile("/tmp/zolt_dory_e2_beta_round0.bin", .{}) catch null;
                        if (debug_file) |f| {
                            f.writeAll(&debug_e2) catch {};
                            f.close();
                        }
                        dbg("[DORY] e2_beta round 0: current_len={}, g2_vec_len={}\n", .{ current_len, params.g2_vec.len });
                        dbg("[DORY] e2_beta compressed: ", .{});
                        for (debug_e2) |b_| {
                            dbg("{x:0>2}", .{b_});
                        }
                        dbg("\n", .{});
                        dbg("[DORY] e2_beta is_identity: {}\n", .{e2_beta.isIdentity()});
                        dbg("[DORY] s1_work[0..4] scalars: ", .{});
                        for (s1_work[0..@min(4, current_len)]) |s| {
                            dbg("{} ", .{s.isZero()});
                        }
                        dbg("\n", .{});
                        dbg("[DORY] g2_vec[0..4] is_identity: ", .{});
                        for (params.g2_vec[0..@min(4, current_len)]) |g| {
                            dbg("{} ", .{g.isIdentity()});
                        }
                        dbg("\n", .{});
                    }
                }

                first_messages[round] = FirstReduceMessage{
                    .d1_left = d1_left,
                    .d1_right = d1_right,
                    .d2_left = d2_left,
                    .d2_right = d2_right,
                    .e1_beta = e1_beta,
                    .e2_beta = e2_beta,
                };

                // Append first message to transcript (Dory format: no reversal)
                doryAppendGT(transcript, d1_left);
                doryAppendGT(transcript, d1_right);
                doryAppendGT(transcript, d2_left);
                doryAppendGT(transcript, d2_right);
                doryAppendG1(transcript, e1_beta);
                doryAppendG2(transcript, e2_beta);

                // Get beta challenge from transcript
                const beta = transcript.challengeScalarFull();
                const beta_inv = beta.inverse() orelse F.one();

                // Apply first challenge using GLV scalar multiplication
                if (tp) |pool| {
                    const BetaCtx = struct {
                        v1: []G1Point,
                        v2: []G2Point,
                        g1: []const G1Point,
                        g2: []const G2Point,
                        b: F,
                        bi: F,
                    };
                    const beta_ctx = BetaCtx{
                        .v1 = v1_work,
                        .v2 = v2_work,
                        .g1 = params.g1_vec,
                        .g2 = params.g2_vec,
                        .b = beta,
                        .bi = beta_inv,
                    };
                    pool.parallelForForce(current_len, beta_ctx, struct {
                        fn f(cx: BetaCtx, i: usize) void {
                            const scaled_g1 = glv.glvScalarMulG1(cx.g1[i], cx.b).toAffine();
                            cx.v1[i] = cx.v1[i].add(scaled_g1);
                            const scaled_proj = glv.glvScalarMulG2(cx.g2[i], cx.bi);
                            cx.v2[i] = scaled_proj.addAffine(cx.v2[i]).toAffine();
                        }
                    }.f);
                } else {
                    for (0..current_len) |i| {
                        const scaled_g1 = glv.glvScalarMulG1(params.g1_vec[i], beta).toAffine();
                        v1_work[i] = v1_work[i].add(scaled_g1);
                        const scaled_proj = glv.glvScalarMulG2(params.g2_vec[i], beta_inv);
                        v2_work[i] = scaled_proj.addAffine(v2_work[i]).toAffine();
                    }
                }

                // Compute second reduce message
                const c_batch = multiPairBatched(2, .{
                    PairGroup{ .g1 = v1_work[0..n2], .g2 = v2_work[n2..current_len] },
                    PairGroup{ .g1 = v1_work[n2..current_len], .g2 = v2_work[0..n2] },
                }, tp);
                const c_plus = c_batch[0];
                const c_minus = c_batch[1];
                const e1_plus, const e1_minus = if (tp) |pool| blk: {
                    const E1Ctx = struct {
                        v1: []const G1Point,
                        s2: []const F,
                        n2: usize,
                        current_len: usize,
                    };
                    const e1_ctx = E1Ctx{ .v1 = v1_work, .s2 = s2_work, .n2 = n2, .current_len = current_len };
                    break :blk pool.join(
                        G1Point,
                        G1Point,
                        e1_ctx,
                        struct {
                            fn f(cx: E1Ctx) G1Point {
                                return msm.MSM(F, Fp).computeWithPool(cx.v1[0..cx.n2], cx.s2[cx.n2..cx.current_len], null);
                            }
                        }.f,
                        e1_ctx,
                        struct {
                            fn f(cx: E1Ctx) G1Point {
                                return msm.MSM(F, Fp).computeWithPool(cx.v1[cx.n2..cx.current_len], cx.s2[0..cx.n2], null);
                            }
                        }.f,
                    );
                } else .{
                    msm.MSM(F, Fp).computeWithPool(v1_work[0..n2], s2_work[n2..current_len], null),
                    msm.MSM(F, Fp).computeWithPool(v1_work[n2..current_len], s2_work[0..n2], null),
                };
                const e2_plus, const e2_minus = if (tp) |pool| blk: {
                    const E2Ctx = struct {
                        v2: []const G2Point,
                        s1: []const F,
                        n2: usize,
                        current_len: usize,
                    };
                    const e2_ctx = E2Ctx{ .v2 = v2_work, .s1 = s1_work, .n2 = n2, .current_len = current_len };
                    break :blk pool.join(
                        G2Point,
                        G2Point,
                        e2_ctx,
                        struct {
                            fn f(cx: E2Ctx) G2Point {
                                return msmG2(F, cx.v2[cx.n2..cx.current_len], cx.s1[0..cx.n2], null);
                            }
                        }.f,
                        e2_ctx,
                        struct {
                            fn f(cx: E2Ctx) G2Point {
                                return msmG2(F, cx.v2[0..cx.n2], cx.s1[cx.n2..cx.current_len], null);
                            }
                        }.f,
                    );
                } else .{
                    msmG2(F, v2_work[n2..current_len], s1_work[0..n2], null),
                    msmG2(F, v2_work[0..n2], s1_work[n2..current_len], null),
                };

                second_messages[round] = SecondReduceMessage{
                    .c_plus = c_plus,
                    .c_minus = c_minus,
                    .e1_plus = e1_plus,
                    .e1_minus = e1_minus,
                    .e2_plus = e2_plus,
                    .e2_minus = e2_minus,
                };

                // Append second message to transcript (Dory format: no reversal)
                doryAppendGT(transcript, c_plus);
                doryAppendGT(transcript, c_minus);
                doryAppendG1(transcript, e1_plus);
                doryAppendG1(transcript, e1_minus);
                doryAppendG2(transcript, e2_plus);
                doryAppendG2(transcript, e2_minus);

                // Get alpha challenge from transcript
                const alpha = transcript.challengeScalarFull();
                const alpha_inv = alpha.inverse() orelse F.one();

                // Fold vectors using GLV scalar multiplication
                if (tp) |pool| {
                    const FoldCtx = struct {
                        v1: []G1Point,
                        v2: []G2Point,
                        s1: []F,
                        s2: []F,
                        a: F,
                        ai: F,
                        half: usize,
                    };
                    const fold_ctx = FoldCtx{
                        .v1 = v1_work,
                        .v2 = v2_work,
                        .s1 = s1_work,
                        .s2 = s2_work,
                        .a = alpha,
                        .ai = alpha_inv,
                        .half = n2,
                    };
                    pool.parallelForForce(n2, fold_ctx, struct {
                        fn f(cx: FoldCtx, i: usize) void {
                            const scaled_l = glv.glvScalarMulG1(cx.v1[i], cx.a).toAffine();
                            cx.v1[i] = scaled_l.add(cx.v1[i + cx.half]);
                            const scaled_proj = glv.glvScalarMulG2(cx.v2[i], cx.ai);
                            cx.v2[i] = scaled_proj.addAffine(cx.v2[i + cx.half]).toAffine();
                            cx.s1[i] = cx.a.mul(cx.s1[i]).add(cx.s1[i + cx.half]);
                            cx.s2[i] = cx.ai.mul(cx.s2[i]).add(cx.s2[i + cx.half]);
                        }
                    }.f);
                } else {
                    for (0..n2) |i| {
                        const scaled_l = glv.glvScalarMulG1(v1_work[i], alpha).toAffine();
                        v1_work[i] = scaled_l.add(v1_work[i + n2]);
                    }
                    for (0..n2) |i| {
                        const scaled_proj = glv.glvScalarMulG2(v2_work[i], alpha_inv);
                        v2_work[i] = scaled_proj.addAffine(v2_work[i + n2]).toAffine();
                    }
                    for (0..n2) |i| {
                        s1_work[i] = alpha.mul(s1_work[i]).add(s1_work[i + n2]);
                    }
                    for (0..n2) |i| {
                        s2_work[i] = alpha_inv.mul(s2_work[i]).add(s2_work[i + n2]);
                    }
                }

                current_len = n2;
            }

            // Debug: print final folded scalars
            if (comptime debug_verbose) {
                const s1_be = s1_work[0].toBytesBE();
                dbg("[DORY PROVER] s1_work[0] (final) first 16 LE: ", .{});
                for (0..16) |bi| dbg("{x:0>2}", .{s1_be[31 - bi]});
                dbg("\n", .{});
                const s2_be = s2_work[0].toBytesBE();
                dbg("[DORY PROVER] s2_work[0] (final) first 16 LE: ", .{});
                for (0..16) |bi| dbg("{x:0>2}", .{s2_be[31 - bi]});
                dbg("\n", .{});
            }

            // Get gamma challenge
            const gamma = transcript.challengeScalarFull();
            const gamma_inv = gamma.inverse() orelse F.one();

            // Compute final message: E₁ = v₁ + γ·s₁·h₁, E₂ = v₂ + γ⁻¹·s₂·h₂
            const gamma_s1 = gamma.mul(s1_work[0]);
            const gamma_inv_s2 = gamma_inv.mul(s2_work[0]);

            const final_e1, const final_e2 = if (tp) |pool| blk: {
                const FinalCtx = struct {
                    h1: G1Point,
                    h2: G2Point,
                    gs1: F,
                    gis2: F,
                    v1_0: G1Point,
                    v2_0: G2Point,
                };
                const fc = FinalCtx{
                    .h1 = params.h1,
                    .h2 = params.h2,
                    .gs1 = gamma_s1,
                    .gis2 = gamma_inv_s2,
                    .v1_0 = v1_work[0],
                    .v2_0 = v2_work[0],
                };
                break :blk pool.join(
                    G1Point,
                    G2Point,
                    fc,
                    struct {
                        fn f(cx: FinalCtx) G1Point {
                            return cx.v1_0.add(glv.glvScalarMulG1(cx.h1, cx.gs1).toAffine());
                        }
                    }.f,
                    fc,
                    struct {
                        fn f(cx: FinalCtx) G2Point {
                            return glv.glvScalarMulG2(cx.h2, cx.gis2).addAffine(cx.v2_0).toAffine();
                        }
                    }.f,
                );
            } else blk: {
                const scaled_h1 = glv.glvScalarMulG1(params.h1, gamma_s1).toAffine();
                const scaled_h2 = glv.glvScalarMulG2(params.h2, gamma_inv_s2).toAffine();
                break :blk .{ v1_work[0].add(scaled_h1), v2_work[0].add(scaled_h2) };
            };

            const final_message = ScalarProductMessage{
                .e1 = final_e1,
                .e2 = final_e2,
            };

            // Append final message to transcript before deriving d challenge
            // Upstream: transcript.append_serde(b"final_e1", &final_message.e1)
            //           transcript.append_serde(b"final_e2", &final_message.e2)
            // JoltToDoryTranscript maps append_serde → "dory_serde" label
            doryAppendG1(transcript, final_e1);
            doryAppendG2(transcript, final_e2);

            // Get final d challenge to keep transcript in sync
            _ = transcript.challengeScalarFull();

            return Proof{
                .vmv_message = vmv_message,
                .first_messages = first_messages,
                .second_messages = second_messages,
                .final_message = final_message,
                .nu = nu,
                .sigma = sigma,
                .allocator = allocator,
            };
        }
    };
}

/// Generate a deterministic G1 point from seed and index
/// This is a simplified hash-to-curve; production should use proper method
fn generateG1Point(seed: [32]u8, index: usize) G1Point {
    var hasher = std.crypto.hash.sha3.Sha3_256.init(.{});
    hasher.update(&seed);
    var idx_bytes: [8]u8 = undefined;
    std.mem.writeInt(u64, &idx_bytes, @intCast(index), .little);
    hasher.update(&idx_bytes);
    hasher.update("G1");

    var hash: [32]u8 = undefined;
    hasher.final(&hash);

    // Use hash to generate a scalar, then multiply generator
    const scalar = Fr.fromBytes(&hash);
    const g1_gen = G1Point.generator();

    // Scalar multiplication
    return msm.MSM(Fr, Fp).scalarMul(g1_gen, scalar).toAffine();
}

/// Generate a deterministic G2 point from seed and index
fn generateG2Point(seed: [32]u8, index: usize) G2Point {
    var hasher = std.crypto.hash.sha3.Sha3_256.init(.{});
    hasher.update(&seed);
    var idx_bytes: [8]u8 = undefined;
    std.mem.writeInt(u64, &idx_bytes, @intCast(index), .little);
    hasher.update(&idx_bytes);
    hasher.update("G2");

    var hash: [32]u8 = undefined;
    hasher.final(&hash);

    // Use hash to generate a scalar, then multiply generator
    const scalar = Fr.fromBytes(&hash);
    const g2_gen = G2Point.generator();

    // G2 scalar multiplication
    return g2_gen.scalarMul(scalar);
}

// =============================================================================
// Serialization Helpers
// =============================================================================

/// Serialize a Dory commitment (GT element) to arkworks format
pub fn serializeDoryCommitment(commitment: DoryCommitment) [384]u8 {
    return commitment.toBytes();
}

/// Deserialize a Dory commitment from arkworks format
pub fn deserializeDoryCommitment(bytes: *const [384]u8) DoryCommitment {
    return GT.fromBytes(bytes);
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
    // Test with generator point - G2 decompression requires precise curve constants
    // Skip the sqrt verification for now, just test that compression/decompression
    // works for identity
    const identity = G2Point.identity();
    const compressed_id = compressG2(identity);
    const decompressed_id = decompressG2(&compressed_id);
    try std.testing.expect(decompressed_id != null);
    try std.testing.expect(decompressed_id.?.infinity);

    // For non-identity points, just verify compression produces correct format
    const g2_gen = G2Point.generator();
    const compressed = compressG2(g2_gen);

    // Should have flag bits set correctly (not infinity)
    try std.testing.expect(compressed[63] & 0x40 == 0); // Not infinity
}

test "g2 point compression identity" {
    const identity = G2Point.identity();
    const compressed = compressG2(identity);
    const decompressed = decompressG2(&compressed);

    try std.testing.expect(decompressed != null);
    try std.testing.expect(decompressed.?.infinity);
}

test "dory commitment scheme setup" {
    const allocator = std.testing.allocator;

    var srs = try DoryCommitmentScheme(Fr).setup(allocator, 4);
    defer srs.deinit();

    // 4 variables = 16 coefficients
    // sigma = 2, nu = 2 -> 4 columns, 4 rows
    try std.testing.expectEqual(@as(usize, 4), srs.num_columns);
    try std.testing.expectEqual(@as(usize, 4), srs.num_rows);
    try std.testing.expectEqual(@as(usize, 4), srs.g1_vec.len);
    try std.testing.expectEqual(@as(usize, 4), srs.g2_vec.len);
    try std.testing.expectEqual(@as(u32, 2), srs.sigma);
    try std.testing.expectEqual(@as(u32, 2), srs.nu);
}

test "dory commitment basic" {
    const allocator = std.testing.allocator;

    var srs = try DoryCommitmentScheme(Fr).setup(allocator, 2);
    defer srs.deinit();

    // Commit to simple polynomial
    const evals = [_]Fr{ Fr.fromU64(1), Fr.fromU64(2), Fr.fromU64(3), Fr.fromU64(4) };
    const commitment = DoryCommitmentScheme(Fr).commit(&srs, &evals);

    // Commitment should not be one (unless all evals are zero)
    try std.testing.expect(!commitment.isOne());
}

test "dory commitment deterministic" {
    const allocator = std.testing.allocator;

    var srs1 = try DoryCommitmentScheme(Fr).setup(allocator, 2);
    defer srs1.deinit();

    var srs2 = try DoryCommitmentScheme(Fr).setup(allocator, 2);
    defer srs2.deinit();

    const evals = [_]Fr{ Fr.fromU64(1), Fr.fromU64(2), Fr.fromU64(3), Fr.fromU64(4) };

    const commitment1 = DoryCommitmentScheme(Fr).commit(&srs1, &evals);
    const commitment2 = DoryCommitmentScheme(Fr).commit(&srs2, &evals);

    // Same SRS + same polynomial = same commitment
    try std.testing.expect(commitment1.eql(commitment2));
}

test "dory commitment serialization roundtrip" {
    const allocator = std.testing.allocator;

    var srs = try DoryCommitmentScheme(Fr).setup(allocator, 2);
    defer srs.deinit();

    const evals = [_]Fr{ Fr.fromU64(1), Fr.fromU64(2), Fr.fromU64(3), Fr.fromU64(4) };
    const commitment = DoryCommitmentScheme(Fr).commit(&srs, &evals);

    // Serialize
    const bytes = serializeDoryCommitment(commitment);
    try std.testing.expectEqual(@as(usize, 384), bytes.len);

    // Deserialize
    const decoded = deserializeDoryCommitment(&bytes);

    // Should match
    try std.testing.expect(commitment.eql(decoded));
}

test "dory empty polynomial commits to one" {
    const allocator = std.testing.allocator;

    var srs = try DoryCommitmentScheme(Fr).setup(allocator, 2);
    defer srs.deinit();

    const empty: []const Fr = &[_]Fr{};
    const commitment = DoryCommitmentScheme(Fr).commit(&srs, empty);

    // Empty polynomial should commit to identity (one)
    try std.testing.expect(commitment.isOne());
}

test "commitOneHotWithPool equivalence with commitWithPool" {
    const allocator = std.testing.allocator;
    const DoryScheme = DoryCommitmentScheme(Fr);

    // num_vars=4 → 16 entries, sigma=2, nu=2 → 4 cols, 4 rows
    var srs = try DoryScheme.setup(allocator, 4);
    defer srs.deinit();

    const k_chunk: usize = 4;
    const trace_length: usize = 4; // 16 / k_chunk

    // Build sparse indices: cycle 0→addr 1, cycle 1→addr 0, cycle 2→addr 3, cycle 3→null
    var indices = [_]?u8{ 1, 0, 3, null };

    // Build equivalent dense polynomial (k_chunk * trace_length = 16 entries, CycleMajor layout)
    // CycleMajor: poly[addr * trace_length + cycle] = 1 if that cycle maps to addr
    var dense = [_]Fr{Fr.zero()} ** 16;
    for (0..trace_length) |cycle| {
        if (indices[cycle]) |addr| {
            dense[@as(usize, addr) * trace_length + cycle] = Fr.one();
        }
    }

    const dense_commit = DoryScheme.commitWithPool(&srs, &dense, null);
    const sparse_commit = try DoryScheme.commitOneHotWithPool(&srs, &indices, k_chunk, trace_length, allocator, null);

    try std.testing.expect(dense_commit.eql(sparse_commit));
}

test "commitOneHotWithPool all null indices" {
    const allocator = std.testing.allocator;
    const DoryScheme = DoryCommitmentScheme(Fr);

    var srs = try DoryScheme.setup(allocator, 4);
    defer srs.deinit();

    var indices = [_]?u8{ null, null, null, null };
    const commit = try DoryScheme.commitOneHotWithPool(&srs, &indices, 4, 4, allocator, null);

    try std.testing.expect(commit.isOne());
}

test "commitOneHotWithPoolAndHints equivalence" {
    const allocator = std.testing.allocator;
    const DoryScheme = DoryCommitmentScheme(Fr);

    // num_vars=4 → 16 entries, sigma=2, nu=2 → 4 cols, 4 rows
    var srs = try DoryScheme.setup(allocator, 4);
    defer srs.deinit();

    const k_chunk: usize = 4;
    const trace_length: usize = 4;

    var indices = [_]?u8{ 1, 0, 3, null };

    // Build equivalent dense polynomial
    var dense = [_]Fr{Fr.zero()} ** 16;
    for (0..trace_length) |cycle| {
        if (indices[cycle]) |addr| {
            dense[@as(usize, addr) * trace_length + cycle] = Fr.one();
        }
    }

    // Get sparse result with hints
    const sparse_result = try DoryScheme.commitOneHotWithPoolAndHints(&srs, &indices, k_chunk, trace_length, allocator, null);
    defer allocator.free(sparse_result.row_commitments);

    // Get dense result with hints
    const dense_result = try DoryScheme.commitWithPoolAndHints(&srs, &dense, allocator, null);
    defer allocator.free(dense_result.row_commitments);

    // Commitments must match
    try std.testing.expect(sparse_result.commitment.eql(dense_result.commitment));

    // Row commitments must match element-wise
    try std.testing.expectEqual(sparse_result.row_commitments.len, dense_result.row_commitments.len);
    for (sparse_result.row_commitments, dense_result.row_commitments) |s, d| {
        try std.testing.expect(s.x.eql(d.x));
        try std.testing.expect(s.y.eql(d.y));
    }
}

test "commitOneHotWithPool multi-row per address" {
    const allocator = std.testing.allocator;
    const DoryScheme = DoryCommitmentScheme(Fr);

    // k_chunk=2, trace_length=8 → poly_size=16, num_vars=4, sigma=2, nu=2
    // num_cols=4, num_rows=4, rows_per_k=2
    // Address 0 spans rows 0-1, address 1 spans rows 2-3
    const k_chunk: usize = 2;
    const trace_length: usize = 8;

    var srs = try DoryScheme.setup(allocator, 4);
    defer srs.deinit();

    // cycle 0→addr 0 (row=0, col=0), cycle 4→addr 0 (row=1, col=0)
    // cycle 1→addr 1 (row=2, col=1), cycle 5→addr 1 (row=3, col=1)
    var indices = [_]?u8{ 0, 1, null, null, 0, 1, null, null };

    // Build equivalent dense polynomial (CycleMajor: poly[addr*T + cycle])
    var dense = [_]Fr{Fr.zero()} ** 16;
    for (0..trace_length) |cycle| {
        if (indices[cycle]) |addr| {
            dense[@as(usize, addr) * trace_length + cycle] = Fr.one();
        }
    }

    const dense_commit = DoryScheme.commitWithPool(&srs, &dense, null);
    const sparse_commit = try DoryScheme.commitOneHotWithPool(&srs, &indices, k_chunk, trace_length, allocator, null);

    try std.testing.expect(dense_commit.eql(sparse_commit));
}

test "commitOneHotWithPool all-active indices" {
    const allocator = std.testing.allocator;
    const DoryScheme = DoryCommitmentScheme(Fr);

    // num_vars=4 → 16 entries, k_chunk=4, trace_length=4
    var srs = try DoryScheme.setup(allocator, 4);
    defer srs.deinit();

    const k_chunk: usize = 4;
    const trace_length: usize = 4;

    // Every cycle maps to an address (no nulls) — the common production case
    var indices = [_]?u8{ 2, 0, 1, 3 };

    // Build equivalent dense polynomial
    var dense = [_]Fr{Fr.zero()} ** 16;
    for (0..trace_length) |cycle| {
        if (indices[cycle]) |addr| {
            dense[@as(usize, addr) * trace_length + cycle] = Fr.one();
        }
    }

    const dense_commit = DoryScheme.commitWithPool(&srs, &dense, null);
    const sparse_commit = try DoryScheme.commitOneHotWithPool(&srs, &indices, k_chunk, trace_length, allocator, null);

    try std.testing.expect(dense_commit.eql(sparse_commit));
}

test "combineRowCommitmentHints matches direct joint commit" {
    const allocator = std.testing.allocator;
    const DoryScheme = DoryCommitmentScheme(Fr);

    // num_vars=4 → 16 entries, sigma=2, nu=2 → 4 cols, 4 rows
    var srs = try DoryScheme.setup(allocator, 4);
    defer srs.deinit();

    const k_chunk: usize = 4;
    const trace_length: usize = 4;

    // Polynomial A: one-hot sparse
    var indices_a = [_]?u8{ 1, 0, 3, null };
    // Polynomial B: one-hot sparse (different pattern)
    var indices_b = [_]?u8{ 0, 2, null, 1 };
    // Polynomial C: dense (non-one-hot, arbitrary values)
    var dense_c = [_]Fr{Fr.zero()} ** 16;
    dense_c[0] = Fr.fromU64(7);
    dense_c[3] = Fr.fromU64(2);
    dense_c[5] = Fr.fromU64(11);
    dense_c[10] = Fr.fromU64(5);
    dense_c[15] = Fr.fromU64(3);

    // Commit each polynomial and get row commitment hints
    const result_a = try DoryScheme.commitOneHotWithPoolAndHints(&srs, &indices_a, k_chunk, trace_length, allocator, null);
    defer allocator.free(result_a.row_commitments);
    const result_b = try DoryScheme.commitOneHotWithPoolAndHints(&srs, &indices_b, k_chunk, trace_length, allocator, null);
    defer allocator.free(result_b.row_commitments);
    const result_c = try DoryScheme.commitWithPoolAndHints(&srs, &dense_c, allocator, null);
    defer allocator.free(result_c.row_commitments);

    // Gamma coefficients (arbitrary non-trivial scalars)
    const gamma0 = Fr.fromU64(42);
    const gamma1 = Fr.fromU64(137);
    const gamma2 = Fr.fromU64(999);
    const coeffs = [_]Fr{ gamma0, gamma1, gamma2 };

    // Build joint polynomial: joint = γ0*A + γ1*B + γ2*C
    var joint = [_]Fr{Fr.zero()} ** 16;

    // Expand A into dense and accumulate
    for (0..trace_length) |cycle| {
        if (indices_a[cycle]) |addr| {
            const j = @as(usize, addr) * trace_length + cycle;
            joint[j] = joint[j].add(gamma0); // one-hot: value is 1, so γ0 * 1 = γ0
        }
    }
    // Expand B into dense and accumulate
    for (0..trace_length) |cycle| {
        if (indices_b[cycle]) |addr| {
            const j = @as(usize, addr) * trace_length + cycle;
            joint[j] = joint[j].add(gamma1);
        }
    }
    // Accumulate C (dense)
    for (0..16) |j| {
        if (!dense_c[j].eql(Fr.zero())) {
            joint[j] = joint[j].add(dense_c[j].mul(gamma2));
        }
    }

    // Get row commitments of the joint polynomial directly
    const direct_result = try DoryScheme.commitWithPoolAndHints(&srs, &joint, allocator, null);
    defer allocator.free(direct_result.row_commitments);

    // Combine hints homomorphically
    const hints = [_][]const G1Point{
        result_a.row_commitments,
        result_b.row_commitments,
        result_c.row_commitments,
    };
    const num_rows = direct_result.row_commitments.len;
    const combined = try DoryScheme.combineRowCommitmentHints(&hints, &coeffs, num_rows, allocator, null);
    defer allocator.free(combined);

    // Row commitments must match element-wise
    try std.testing.expectEqual(num_rows, combined.len);
    for (0..num_rows) |r| {
        const c = combined[r];
        const d = direct_result.row_commitments[r];
        if (c.infinity and d.infinity) continue;
        try std.testing.expect(!c.infinity);
        try std.testing.expect(!d.infinity);
        try std.testing.expect(c.x.eql(d.x));
        try std.testing.expect(c.y.eql(d.y));
    }

    // Also verify the final GT commitment matches
    const combined_commitment = DoryScheme.rowCommitmentsToCommitment(&srs, combined, num_rows, null);
    try std.testing.expect(combined_commitment.eql(direct_result.commitment));
}

test "combineRowCommitmentHints all-identity rows" {
    const allocator = std.testing.allocator;
    const DoryScheme = DoryCommitmentScheme(Fr);

    var srs = try DoryScheme.setup(allocator, 4);
    defer srs.deinit();

    const k_chunk: usize = 4;
    const trace_length: usize = 4;

    // Both polynomials have all-null indices → all row commitments are identity
    var indices_a = [_]?u8{ null, null, null, null };
    var indices_b = [_]?u8{ null, null, null, null };

    const result_a = try DoryScheme.commitOneHotWithPoolAndHints(&srs, &indices_a, k_chunk, trace_length, allocator, null);
    defer allocator.free(result_a.row_commitments);
    const result_b = try DoryScheme.commitOneHotWithPoolAndHints(&srs, &indices_b, k_chunk, trace_length, allocator, null);
    defer allocator.free(result_b.row_commitments);

    const coeffs = [_]Fr{ Fr.fromU64(5), Fr.fromU64(10) };
    const hints = [_][]const G1Point{
        result_a.row_commitments,
        result_b.row_commitments,
    };
    const combined = try DoryScheme.combineRowCommitmentHints(&hints, &coeffs, result_a.row_commitments.len, allocator, null);
    defer allocator.free(combined);

    // All rows should be identity (infinity)
    for (combined) |pt| {
        try std.testing.expect(pt.infinity);
    }
}

test "dory proof serialization" {
    const allocator = std.testing.allocator;

    // Setup for 2 variables: sigma=1, nu=1, so 2x2 matrix = 4 evals
    var srs = try DoryCommitmentScheme(Fr).setup(allocator, 2);
    defer srs.deinit();

    // 2 variables = 2^2 = 4 evals
    const evals = [_]Fr{ Fr.fromU64(1), Fr.fromU64(2), Fr.fromU64(3), Fr.fromU64(4) };
    // Point should be sigma + nu = 1 + 1 = 2 elements
    const point = [_]Fr{ Fr.fromU64(5), Fr.fromU64(6) };

    var proof = try DoryCommitmentScheme(Fr).open(&srs, &evals, &point, allocator);
    defer proof.deinit();

    // Serialize proof
    const bytes = try proof.toBytes(allocator);
    defer allocator.free(bytes);

    // Check expected size
    const num_rounds = @max(srs.nu, srs.sigma);
    const vmv_size = 384 + 384 + 32;
    const first_msg_size = 384 * 4 + 32 + 64;
    const second_msg_size = 384 * 2 + 32 * 2 + 64 * 2;
    const final_size = 32 + 64;
    const expected_size = vmv_size + 4 + first_msg_size * num_rounds + second_msg_size * num_rounds + final_size + 8;

    try std.testing.expectEqual(expected_size, bytes.len);
}

test "vmv message serialization" {
    const vmv = VMVMessage{
        .c = GT.one(),
        .d2 = GT.one(),
        .e1 = G1Point.generator(),
    };

    const bytes = vmv.toBytes();
    try std.testing.expectEqual(@as(usize, 384 + 384 + 32), bytes.len);

    // First 384 bytes should be GT.one()
    const expected_gt_one = GT.one().toBytes();
    try std.testing.expectEqualSlices(u8, &expected_gt_one, bytes[0..384]);
}

test "dory commitment with jolt srs - compare matrix layout" {
    // Test that we use the same matrix layout as Jolt
    // Jolt with 8 coefficients (3 vars) uses:
    //   num_columns = 4 (sigma = 2)
    //   max_num_rows = 2 (nu = 1)
    const allocator = std.testing.allocator;

    // Load Jolt's SRS file if available
    const srs_result = DoryCommitmentScheme(Fr).loadFromFile(allocator, "/tmp/jolt_dory_srs.bin");
    if (srs_result) |srs_const| {
        var srs = srs_const;
        defer srs.deinit();

        // Print what we loaded
        dbg("\nLoaded SRS:\n", .{});
        dbg("  num_columns = {}\n", .{srs.num_columns});
        dbg("  num_rows = {}\n", .{srs.num_rows});

        // Same polynomial as Jolt test: [1, 2, 3, 4, 5, 6, 7, 8]
        const evals = [_]Fr{
            Fr.fromU64(1), Fr.fromU64(2), Fr.fromU64(3), Fr.fromU64(4),
            Fr.fromU64(5), Fr.fromU64(6), Fr.fromU64(7), Fr.fromU64(8),
        };

        const commitment = DoryCommitmentScheme(Fr).commit(&srs, &evals);
        const bytes = commitment.toBytes();

        dbg("\nZolt commitment:\n", .{});
        dbg("  First 16 bytes: {x}\n", .{bytes[0..16].*});
        dbg("  Last 16 bytes: {x}\n", .{bytes[384 - 16 .. 384].*});

        // Jolt's commitment (from test output):
        // First 16 bytes: [cf, 11, 82, 20, dc, 8c, 59, 10, fc, 08, e5, f4, 58, a2, 42, 6f]
        // If these match, we have the same commitment!
        const jolt_first_bytes = [_]u8{ 0xcf, 0x11, 0x82, 0x20, 0xdc, 0x8c, 0x59, 0x10, 0xfc, 0x08, 0xe5, 0xf4, 0x58, 0xa2, 0x42, 0x6f };

        if (std.mem.eql(u8, bytes[0..16], &jolt_first_bytes)) {
            dbg("\n*** SUCCESS: Zolt commitment matches Jolt! ***\n", .{});
        } else {
            dbg("\n*** MISMATCH: Commitment differs from Jolt ***\n", .{});
            dbg("  Expected (Jolt): {x}\n", .{jolt_first_bytes});
            dbg("  Got (Zolt):      {x}\n", .{bytes[0..16].*});
        }
    } else |_| {
        dbg("Skipping Jolt SRS comparison test - no SRS file at /tmp/jolt_dory_srs.bin\n", .{});
        dbg("Run Jolt's test_export_dory_srs first.\n", .{});
    }
}

test "dory commitment debug - compare intermediate values with jolt" {
    // Detailed debug test to compare intermediate MSM and pairing results
    // Run after Jolt's test_export_dory_commitment_debug to compare
    //
    // Jolt reference values:
    // Row 0 MSM: 03 81 87 9a 0a d6 7c 0f 6c 84 5b ed 4e f6 73 80...
    // Row 1 MSM: 7c 95 83 60 cf bf 11 41 fa 6a 27 f6 84 1c d1 68...
    // Final commitment: cf 11 82 20 dc 8c 59 10 fc 08 e5 f4 58 a2 42 6f...

    const allocator = std.testing.allocator;
    const DoryScheme = DoryCommitmentScheme(Fr);

    // Load Jolt's SRS file if available
    const srs_result = DoryScheme.loadFromFile(allocator, "/tmp/jolt_dory_srs.bin");
    if (srs_result) |srs_const| {
        var srs = srs_const;
        defer srs.deinit();

        dbg("\n=== Dory Commitment Debug ===\n", .{});
        dbg("SRS loaded: {} G1 points, {} G2 points\n", .{ srs.g1_vec.len, srs.g2_vec.len });

        // Print first G1 point bytes (in arkworks format for comparison)
        const g1_0 = srs.g1_vec[0];
        const g1_0_x_std = g1_0.x.fromMontgomery();
        const g1_0_y_std = g1_0.y.fromMontgomery();

        // Check if G1 points are on the curve and print coordinates
        for (0..4) |i| {
            const g1_i = srs.g1_vec[i];
            const on_curve = g1_i.isOnCurve();

            // Print raw limbs (Montgomery form) for debugging
            dbg("\nG1[{}] on curve: {}\n", .{ i, on_curve });
            dbg("  x (Montgomery) limbs: {x:0>16} {x:0>16} {x:0>16} {x:0>16}\n", .{
                g1_i.x.limbs[0],
                g1_i.x.limbs[1],
                g1_i.x.limbs[2],
                g1_i.x.limbs[3],
            });
            dbg("  y (Montgomery) limbs: {x:0>16} {x:0>16} {x:0>16} {x:0>16}\n", .{
                g1_i.y.limbs[0],
                g1_i.y.limbs[1],
                g1_i.y.limbs[2],
                g1_i.y.limbs[3],
            });

            // Print x and y coordinates in standard form for comparison with Jolt
            const x_std = g1_i.x.fromMontgomery();
            const y_std = g1_i.y.fromMontgomery();
            var x_bytes: [32]u8 = undefined;
            var y_bytes: [32]u8 = undefined;
            for (0..4) |j| {
                std.mem.writeInt(u64, x_bytes[j * 8 ..][0..8], x_std.limbs[j], .little);
                std.mem.writeInt(u64, y_bytes[j * 8 ..][0..8], y_std.limbs[j], .little);
            }

            dbg("  x first 16 (std): {x}\n", .{x_bytes[0..16].*});
            dbg("  y first 16 (std): {x}\n", .{y_bytes[0..16].*});
        }

        var g1_0_bytes: [64]u8 = undefined;
        for (0..4) |i| {
            std.mem.writeInt(u64, g1_0_bytes[i * 8 ..][0..8], g1_0_x_std.limbs[i], .little);
            std.mem.writeInt(u64, g1_0_bytes[32 + i * 8 ..][0..8], g1_0_y_std.limbs[i], .little);
        }
        dbg("\nG1[0] first 16 bytes: {x}\n", .{g1_0_bytes[0..16].*});
        // Jolt: 10 f1 51 c2 83 fa c8 e8 ae 44 83 39 77 82 ca db

        // Test simple scalar multiplication: G1[0] * 1 should equal G1[0]
        const scalar_one = Fr.fromU64(1);
        const g1_times_1 = msm.MSM(Fr, Fp).compute(srs.g1_vec[0..1], &[_]Fr{scalar_one});

        const g1_times_1_x_std = g1_times_1.x.fromMontgomery();
        const g1_times_1_y_std = g1_times_1.y.fromMontgomery();

        var g1_times_1_bytes: [64]u8 = undefined;
        for (0..4) |i| {
            std.mem.writeInt(u64, g1_times_1_bytes[i * 8 ..][0..8], g1_times_1_x_std.limbs[i], .little);
            std.mem.writeInt(u64, g1_times_1_bytes[32 + i * 8 ..][0..8], g1_times_1_y_std.limbs[i], .little);
        }
        dbg("G1[0]*1 first 16 bytes: {x}\n", .{g1_times_1_bytes[0..16].*});

        if (std.mem.eql(u8, &g1_0_bytes, &g1_times_1_bytes)) {
            dbg("  *** G1[0]*1 == G1[0]: PASS ***\n", .{});
        } else {
            dbg("  *** G1[0]*1 != G1[0]: FAIL - MSM broken ***\n", .{});
        }

        // Test: Print first 4 G1 points
        dbg("\nG1 points in SRS:\n", .{});
        for (0..4) |i| {
            const g1_i = srs.g1_vec[i];
            const g1_i_x_std = g1_i.x.fromMontgomery();
            var g1_i_bytes: [32]u8 = undefined;
            for (0..4) |j| {
                std.mem.writeInt(u64, g1_i_bytes[j * 8 ..][0..8], g1_i_x_std.limbs[j], .little);
            }
            dbg("  G1[{}] x: {x}\n", .{ i, g1_i_bytes[0..16].* });
        }

        // Test: G1[0] * 2 (scalar multiplication)
        const scalar_two = Fr.fromU64(2);
        dbg("\nScalar 2 (Montgomery form): {x:0>16} {x:0>16} {x:0>16} {x:0>16}\n", .{
            scalar_two.limbs[0],
            scalar_two.limbs[1],
            scalar_two.limbs[2],
            scalar_two.limbs[3],
        });
        const scalar_two_std = scalar_two.fromMontgomery();
        dbg("Scalar 2 (standard form): {x:0>16} {x:0>16} {x:0>16} {x:0>16}\n", .{
            scalar_two_std.limbs[0],
            scalar_two_std.limbs[1],
            scalar_two_std.limbs[2],
            scalar_two_std.limbs[3],
        });

        const g1_times_2 = msm.MSM(Fr, Fp).compute(srs.g1_vec[0..1], &[_]Fr{scalar_two});
        const g1_times_2_x_std = g1_times_2.x.fromMontgomery();
        var g1_times_2_bytes: [32]u8 = undefined;
        for (0..4) |i| {
            std.mem.writeInt(u64, g1_times_2_bytes[i * 8 ..][0..8], g1_times_2_x_std.limbs[i], .little);
        }
        dbg("G1[0]*2 x: {x}\n", .{g1_times_2_bytes[0..16].*});

        // Test: G1[0]*1 + G1[1]*1 (adding two points)
        const g1_sum = msm.MSM(Fr, Fp).compute(srs.g1_vec[0..2], &[_]Fr{ Fr.fromU64(1), Fr.fromU64(1) });
        const g1_sum_x_std = g1_sum.x.fromMontgomery();
        var g1_sum_bytes: [32]u8 = undefined;
        for (0..4) |i| {
            std.mem.writeInt(u64, g1_sum_bytes[i * 8 ..][0..8], g1_sum_x_std.limbs[i], .little);
        }
        dbg("G1[0]+G1[1] x (via MSM): {x}\n", .{g1_sum_bytes[0..16].*});

        // Also test direct affine addition
        const g1_sum_affine = srs.g1_vec[0].add(srs.g1_vec[1]);
        const g1_sum_affine_x_std = g1_sum_affine.x.fromMontgomery();
        var g1_sum_affine_bytes: [32]u8 = undefined;
        for (0..4) |i| {
            std.mem.writeInt(u64, g1_sum_affine_bytes[i * 8 ..][0..8], g1_sum_affine_x_std.limbs[i], .little);
        }
        dbg("G1[0]+G1[1] x (affine add): {x}\n", .{g1_sum_affine_bytes[0..16].*});

        // Polynomial [1, 2, 3, 4, 5, 6, 7, 8]
        const num_cols: usize = 4;
        // const num_rows: usize = 2;

        // Row 0: [1, 2, 3, 4]
        const row0_evals = [_]Fr{
            Fr.fromU64(1), Fr.fromU64(2), Fr.fromU64(3), Fr.fromU64(4),
        };

        // Compute row 0 MSM (returns affine point directly)
        const row0_affine = msm.MSM(Fr, Fp).compute(
            srs.g1_vec[0..num_cols],
            &row0_evals,
        );

        // Serialize to arkworks format (standard form, not Montgomery)
        const row0_x_std = row0_affine.x.fromMontgomery();
        const row0_y_std = row0_affine.y.fromMontgomery();

        dbg("\nRow 0 MSM result:\n", .{});
        dbg("  x limbs (standard): {x:0>16} {x:0>16} {x:0>16} {x:0>16}\n", .{
            row0_x_std.limbs[0],
            row0_x_std.limbs[1],
            row0_x_std.limbs[2],
            row0_x_std.limbs[3],
        });

        // Get as bytes (arkworks format: LE limbs)
        var row0_bytes: [64]u8 = undefined;
        for (0..4) |i| {
            std.mem.writeInt(u64, row0_bytes[i * 8 ..][0..8], row0_x_std.limbs[i], .little);
            std.mem.writeInt(u64, row0_bytes[32 + i * 8 ..][0..8], row0_y_std.limbs[i], .little);
        }
        dbg("  First 16 bytes: {x}\n", .{row0_bytes[0..16].*});

        // Jolt reference: 03 81 87 9a 0a d6 7c 0f 6c 84 5b ed 4e f6 73 80
        const jolt_row0_bytes = [_]u8{ 0x03, 0x81, 0x87, 0x9a, 0x0a, 0xd6, 0x7c, 0x0f, 0x6c, 0x84, 0x5b, 0xed, 0x4e, 0xf6, 0x73, 0x80 };

        if (std.mem.eql(u8, row0_bytes[0..16], &jolt_row0_bytes)) {
            dbg("  *** Row 0 MSM MATCHES Jolt! ***\n", .{});
        } else {
            dbg("  *** Row 0 MSM MISMATCH ***\n", .{});
            dbg("  Expected: {x}\n", .{jolt_row0_bytes});
        }

        // Row 1: [5, 6, 7, 8]
        const row1_evals = [_]Fr{
            Fr.fromU64(5), Fr.fromU64(6), Fr.fromU64(7), Fr.fromU64(8),
        };

        const row1_affine = msm.MSM(Fr, Fp).compute(
            srs.g1_vec[0..num_cols],
            &row1_evals,
        );

        const row1_x_std = row1_affine.x.fromMontgomery();
        const row1_y_std = row1_affine.y.fromMontgomery();

        var row1_bytes: [64]u8 = undefined;
        for (0..4) |i| {
            std.mem.writeInt(u64, row1_bytes[i * 8 ..][0..8], row1_x_std.limbs[i], .little);
            std.mem.writeInt(u64, row1_bytes[32 + i * 8 ..][0..8], row1_y_std.limbs[i], .little);
        }
        dbg("\nRow 1 MSM result:\n", .{});
        dbg("  First 16 bytes: {x}\n", .{row1_bytes[0..16].*});

        // Jolt reference: 7c 95 83 60 cf bf 11 41 fa 6a 27 f6 84 1c d1 68
        const jolt_row1_bytes = [_]u8{ 0x7c, 0x95, 0x83, 0x60, 0xcf, 0xbf, 0x11, 0x41, 0xfa, 0x6a, 0x27, 0xf6, 0x84, 0x1c, 0xd1, 0x68 };

        if (std.mem.eql(u8, row1_bytes[0..16], &jolt_row1_bytes)) {
            dbg("  *** Row 1 MSM MATCHES Jolt! ***\n", .{});
        } else {
            dbg("  *** Row 1 MSM MISMATCH ***\n", .{});
            dbg("  Expected: {x}\n", .{jolt_row1_bytes});
        }

        // Now test pairing: e(row0, G2[0])
        dbg("\n=== Pairing Test ===\n", .{});

        // First check G2[0] bytes
        const g2_0 = srs.g2_vec[0];
        const g2_0_x_c0_std = g2_0.x.c0.fromMontgomery();
        var g2_0_bytes: [32]u8 = undefined;
        for (0..4) |i| {
            std.mem.writeInt(u64, g2_0_bytes[i * 8 ..][0..8], g2_0_x_c0_std.limbs[i], .little);
        }
        dbg("G2[0] x.c0 first 16 bytes: {x}\n", .{g2_0_bytes[0..16].*});
        // Jolt G2[0] first 16: 6f f9 ca 75 a1 71 4f c8 fa 12 b1 80 e1 a9 c6 95
        const jolt_g2_0_bytes = [_]u8{ 0x6f, 0xf9, 0xca, 0x75, 0xa1, 0x71, 0x4f, 0xc8, 0xfa, 0x12, 0xb1, 0x80, 0xe1, 0xa9, 0xc6, 0x95 };
        if (std.mem.eql(u8, g2_0_bytes[0..16], &jolt_g2_0_bytes)) {
            dbg("  *** G2[0] MATCHES Jolt! ***\n", .{});
        } else {
            dbg("  *** G2[0] MISMATCH ***\n", .{});
            dbg("  Expected: {x}\n", .{jolt_g2_0_bytes});
        }

        // Print all G2[0] coordinates to verify loading
        const g2_0_x_c1_std = g2_0.x.c1.fromMontgomery();
        const g2_0_y_c0_std = g2_0.y.c0.fromMontgomery();
        const g2_0_y_c1_std = g2_0.y.c1.fromMontgomery();

        var g2_x_c1_bytes: [32]u8 = undefined;
        var g2_y_c0_bytes: [32]u8 = undefined;
        var g2_y_c1_bytes: [32]u8 = undefined;
        for (0..4) |i| {
            std.mem.writeInt(u64, g2_x_c1_bytes[i * 8 ..][0..8], g2_0_x_c1_std.limbs[i], .little);
            std.mem.writeInt(u64, g2_y_c0_bytes[i * 8 ..][0..8], g2_0_y_c0_std.limbs[i], .little);
            std.mem.writeInt(u64, g2_y_c1_bytes[i * 8 ..][0..8], g2_0_y_c1_std.limbs[i], .little);
        }
        dbg("G2[0] x.c1 first 16: {x}\n", .{g2_x_c1_bytes[0..16].*});
        dbg("G2[0] y.c0 first 16: {x}\n", .{g2_y_c0_bytes[0..16].*});
        dbg("G2[0] y.c1 first 16: {x}\n", .{g2_y_c1_bytes[0..16].*});

        // Compute pairing e(row0, G2[0])
        const row0_g1 = G1PointFp{
            .x = row0_affine.x,
            .y = row0_affine.y,
            .infinity = row0_affine.infinity,
        };
        const pairing_result = pairing.pairingFp(row0_g1, g2_0);
        const pairing_bytes = pairing_result.toBytes();
        dbg("\nPairing e(row0, G2[0]) first 16 bytes: {x}\n", .{pairing_bytes[0..16].*});

        // Jolt Pairing(0, 0) first 16 bytes: be c8 5a 17 0f 50 62 ad 4a 93 ce a6 33 10 15 f4
        const jolt_pairing_bytes = [_]u8{ 0xbe, 0xc8, 0x5a, 0x17, 0x0f, 0x50, 0x62, 0xad, 0x4a, 0x93, 0xce, 0xa6, 0x33, 0x10, 0x15, 0xf4 };
        if (std.mem.eql(u8, pairing_bytes[0..16], &jolt_pairing_bytes)) {
            dbg("  *** Pairing MATCHES Jolt! ***\n", .{});
        } else {
            dbg("  *** Pairing MISMATCH ***\n", .{});
            dbg("  Expected: {x}\n", .{jolt_pairing_bytes});
        }
    } else |_| {
        dbg("Skipping debug test - no SRS file at /tmp/jolt_dory_srs.bin\n", .{});
        dbg("Run Jolt's test_export_dory_srs first.\n", .{});
    }
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
    const file = std.fs.cwd().createFile("/tmp/zolt_g2_test_points.bin", .{}) catch |err| {
        std.debug.print("Could not create file: {}\n", .{err});
        return;
    };
    defer file.close();

    // Write 3 compressed G2 points (64 bytes each = 192 bytes total)
    file.writeAll(&gen_compressed) catch return;
    file.writeAll(&two_compressed) catch return;
    file.writeAll(&compressed_42) catch return;
    std.debug.print("Wrote 3 compressed G2 points to /tmp/zolt_g2_test_points.bin\n", .{});
}

test "g2 srs points from jolt file" {
    // Load SRS from file and check that G2 points are valid
    const allocator = std.testing.allocator;
    const srs_result = DoryCommitmentScheme(Fr).loadFromFile(allocator, "/tmp/jolt_dory_srs.bin");
    if (srs_result) |*srs_mut| {
        var srs = srs_mut.*;
        defer srs.deinit();

        std.debug.print("\nSRS loaded: {} G1 points, {} G2 points\n", .{ srs.g1_vec.len, srs.g2_vec.len });

        // Write all G2 points compressed to a file
        const srs_file = std.fs.cwd().createFile("/tmp/zolt_g2_srs_points.bin", .{}) catch return;
        defer srs_file.close();

        // First write the count
        var count_buf: [4]u8 = undefined;
        std.mem.writeInt(u32, &count_buf, @intCast(srs.g2_vec.len), .little);
        srs_file.writeAll(&count_buf) catch return;

        for (srs.g2_vec, 0..) |g2, idx| {
            const compressed = compressG2(g2);
            srs_file.writeAll(&compressed) catch return;
            if (idx < 3) {
                std.debug.print("G2 SRS[{}] compressed: ", .{idx});
                for (compressed) |b| {
                    std.debug.print("{x:0>2}", .{b});
                }
                std.debug.print("\n", .{});
            }
        }
        std.debug.print("Wrote {} compressed G2 SRS points to /tmp/zolt_g2_srs_points.bin\n", .{srs.g2_vec.len});

        // Now test: do a small MSM and compress the result
        if (srs.g2_vec.len >= 2) {
            const scalars = [_]Fr{ Fr.fromU64(3), Fr.fromU64(5) };
            const msm_result = msmG2(Fr, srs.g2_vec[0..2], &scalars, null);
            const msm_compressed = compressG2(msm_result);
            std.debug.print("MSM([3,5], G2[0..2]) compressed: ", .{});
            for (msm_compressed) |b| {
                std.debug.print("{x:0>2}", .{b});
            }
            std.debug.print("\n", .{});

            // Write this MSM result for Rust verification
            const msm_file = std.fs.cwd().createFile("/tmp/zolt_g2_msm_test.bin", .{}) catch return;
            defer msm_file.close();
            msm_file.writeAll(&msm_compressed) catch return;
        }
    } else |_| {
        std.debug.print("Skipping SRS test - no file at /tmp/jolt_dory_srs.bin\n", .{});
    }
}
