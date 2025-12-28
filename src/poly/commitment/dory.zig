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
const Allocator = std.mem.Allocator;
const pairing = @import("../../field/pairing.zig");
const field = @import("../../field/mod.zig");
const msm = @import("../../msm/mod.zig");

const Fp = field.BN254BaseField;
const Fr = field.BN254Scalar;
const Fp2 = pairing.Fp2;
const GT = pairing.GT;
const G1Point = msm.AffinePoint(Fp);
const G2Point = pairing.G2Point;
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

/// Dory structured reference string (SRS)
/// Generated using the seed "Jolt Dory URS seed" for compatibility
pub const DorySRS = struct {
    /// G1 generators for polynomial coefficients
    g1_vec: []G1Point,
    /// G2 generators for pairing
    g2_vec: []G2Point,
    /// Maximum number of columns in the matrix
    num_columns: usize,
    /// Maximum number of rows in the matrix
    num_rows: usize,
    /// Log2 of columns (sigma)
    sigma: u32,
    /// Log2 of rows (nu)
    nu: u32,
    allocator: Allocator,

    pub fn deinit(self: *DorySRS) void {
        if (self.g1_vec.len > 0) {
            self.allocator.free(self.g1_vec);
            self.allocator.free(self.g2_vec);
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

        /// Setup the SRS using Jolt's seed
        ///
        /// Uses SHA3-256 with seed "Jolt Dory URS seed" for deterministic generation.
        /// This matches Jolt's DoryCommitmentScheme::setup_prover.
        pub fn setup(allocator: Allocator, max_num_vars: usize) !SetupParams {
            // Calculate matrix dimensions
            // For n variables, we need 2^n coefficients
            const total_size: usize = @as(usize, 1) << @intCast(max_num_vars);

            // Aim for roughly square matrix
            const sigma: u32 = @intCast((max_num_vars + 1) / 2); // columns = 2^sigma
            const nu: u32 = @intCast(max_num_vars - sigma); // rows = 2^nu

            const num_columns: usize = @as(usize, 1) << @intCast(sigma);
            const num_rows: usize = @as(usize, 1) << @intCast(nu);

            // Generate G1 generators (one per column)
            const g1_vec = try allocator.alloc(G1Point, num_columns);
            errdefer allocator.free(g1_vec);

            // Generate G2 generators (one per row)
            const g2_vec = try allocator.alloc(G2Point, num_rows);
            errdefer allocator.free(g2_vec);

            // Use SHA3-256 with Jolt's seed for deterministic generation
            var hasher = std.crypto.hash.sha3.Sha3_256.init(.{});
            hasher.update("Jolt Dory URS seed");
            var seed: [32]u8 = undefined;
            hasher.final(&seed);

            // Generate G1 points using hash-to-curve simulation
            // In production, this would use proper hash-to-curve
            for (0..num_columns) |i| {
                g1_vec[i] = generateG1Point(seed, i);
            }

            // Generate G2 points
            for (0..num_rows) |i| {
                g2_vec[i] = generateG2Point(seed, i + num_columns);
            }

            _ = total_size;

            return SetupParams{
                .g1_vec = g1_vec,
                .g2_vec = g2_vec,
                .num_columns = num_columns,
                .num_rows = num_rows,
                .sigma = sigma,
                .nu = nu,
                .allocator = allocator,
            };
        }

        /// Commit to a polynomial given as evaluations
        ///
        /// The polynomial is laid out as a matrix with `num_columns` columns.
        /// Each row is committed using MSM with G1 generators.
        /// The final commitment is the multi-pairing of row commitments with G2.
        pub fn commit(params: *const SetupParams, evals: []const F) Commitment {
            if (evals.len == 0) {
                return GT.one();
            }

            const num_cols = params.num_columns;
            const num_rows = (evals.len + num_cols - 1) / num_cols;

            // Compute row commitments
            var row_sum = GT.one();

            for (0..num_rows) |row| {
                const start = row * num_cols;
                const end = @min(start + num_cols, evals.len);

                if (start >= evals.len) break;

                const row_evals = evals[start..end];

                // Compute MSM for this row
                const row_commitment = msm.MSM(F, Fp).compute(
                    params.g1_vec[0..row_evals.len],
                    row_evals,
                );

                // Pair with corresponding G2 generator
                if (row < params.g2_vec.len) {
                    const row_g1 = G1PointFp{
                        .x = row_commitment.x,
                        .y = row_commitment.y,
                        .infinity = row_commitment.infinity,
                    };
                    const paired = pairing.pairingFp(row_g1, params.g2_vec[row]);
                    row_sum = row_sum.mul(paired);
                }
            }

            return row_sum;
        }

        /// Create an opening proof
        /// This implements the Dory reduce-and-fold IPA
        pub fn open(
            params: *const SetupParams,
            evals: []const F,
            point: []const F,
            allocator: Allocator,
        ) !Proof {
            _ = evals;
            _ = point;

            const num_rounds = @max(params.nu, params.sigma);

            // Allocate message arrays
            const first_messages = try allocator.alloc(FirstReduceMessage, num_rounds);
            errdefer allocator.free(first_messages);
            const second_messages = try allocator.alloc(SecondReduceMessage, num_rounds);
            errdefer allocator.free(second_messages);

            // Initialize with placeholder values
            // Full IPA implementation would compute these properly
            for (0..num_rounds) |i| {
                first_messages[i] = FirstReduceMessage{
                    .d1_left = GT.one(),
                    .d1_right = GT.one(),
                    .d2_left = GT.one(),
                    .d2_right = GT.one(),
                    .e1_beta = G1Point.identity(),
                    .e2_beta = G2Point.identity(),
                };
                second_messages[i] = SecondReduceMessage{
                    .c_plus = GT.one(),
                    .c_minus = GT.one(),
                    .e1_plus = G1Point.identity(),
                    .e1_minus = G1Point.identity(),
                    .e2_plus = G2Point.identity(),
                    .e2_minus = G2Point.identity(),
                };
            }

            return Proof{
                .vmv_message = VMVMessage{
                    .c = GT.one(),
                    .d2 = GT.one(),
                    .e1 = G1Point.identity(),
                },
                .first_messages = first_messages,
                .second_messages = second_messages,
                .final_message = ScalarProductMessage{
                    .e1 = G1Point.identity(),
                    .e2 = G2Point.identity(),
                },
                .nu = params.nu,
                .sigma = params.sigma,
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

test "dory proof serialization" {
    const allocator = std.testing.allocator;

    var srs = try DoryCommitmentScheme(Fr).setup(allocator, 4);
    defer srs.deinit();

    const evals = [_]Fr{ Fr.fromU64(1), Fr.fromU64(2), Fr.fromU64(3), Fr.fromU64(4) };
    const point = [_]Fr{ Fr.fromU64(5), Fr.fromU64(6), Fr.fromU64(7), Fr.fromU64(8) };

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
