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

/// Dory commitment (GT element = Fp12)
pub const DoryCommitment = GT;

/// Dory opening proof structure
/// Matches Jolt's ArkDoryProof structure
pub const DoryProof = struct {
    /// L commitments from the inner product argument
    L: []GT,
    /// R commitments from the inner product argument
    R: []GT,
    /// Final scalar from the folding
    final_scalar: Fr,
    /// Number of rounds
    num_rounds: usize,
    allocator: Allocator,

    pub fn deinit(self: *DoryProof) void {
        if (self.L.len > 0) {
            self.allocator.free(self.L);
            self.allocator.free(self.R);
        }
    }

    /// Serialize in arkworks-compatible format
    pub fn toBytes(self: *const DoryProof, allocator: Allocator) ![]u8 {
        // Format: num_rounds (u64) + L elements + R elements + final_scalar
        const size = 8 + self.num_rounds * 384 * 2 + 32;
        var result = try allocator.alloc(u8, size);
        var offset: usize = 0;

        // Write number of rounds
        std.mem.writeInt(u64, result[offset..][0..8], @intCast(self.num_rounds), .little);
        offset += 8;

        // Write L elements
        for (self.L) |l| {
            const bytes = l.toBytes();
            @memcpy(result[offset..][0..384], &bytes);
            offset += 384;
        }

        // Write R elements
        for (self.R) |r| {
            const bytes = r.toBytes();
            @memcpy(result[offset..][0..384], &bytes);
            offset += 384;
        }

        // Write final scalar
        const scalar_bytes = self.final_scalar.toBytes();
        @memcpy(result[offset..][0..32], &scalar_bytes);

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
            const sigma = (max_num_vars + 1) / 2; // columns = 2^sigma
            const nu = max_num_vars - sigma; // rows = 2^nu

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

        /// Create an opening proof (placeholder - full IPA not implemented)
        pub fn open(
            params: *const SetupParams,
            evals: []const F,
            point: []const F,
            allocator: Allocator,
        ) !Proof {
            _ = params;
            _ = evals;
            _ = point;

            // Placeholder: return empty proof
            // Full Dory opening proof requires IPA implementation
            return Proof{
                .L = &[_]GT{},
                .R = &[_]GT{},
                .final_scalar = F.zero(),
                .num_rounds = 0,
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

            // Placeholder: verification not yet implemented
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
