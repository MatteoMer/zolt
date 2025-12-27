//! Batch Polynomial Commitment Verification
//!
//! This module provides batch verification for polynomial commitment openings.
//! Batching allows verifying multiple opening claims with fewer pairing operations.
//!
//! ## Overview
//!
//! When the verifier has accumulated multiple polynomial opening claims from
//! different sumcheck stages, they can be batched together:
//!
//! 1. Combine all claims with random linear combination coefficients
//! 2. Compute a single combined commitment and evaluation
//! 3. Verify with a single pairing check
//!
//! ## Usage
//!
//! ```zig
//! var accumulator = OpeningAccumulator(F).init(allocator);
//! defer accumulator.deinit();
//!
//! // Collect claims from each sumcheck stage
//! accumulator.addClaim(commitment1, point1, value1);
//! accumulator.addClaim(commitment2, point2, value2);
//!
//! // Verify all claims at once
//! const valid = accumulator.verifyBatch(&srs, &transcript);
//! ```
//!
//! Reference: jolt-core/src/poly/commitment/hyperkzg/batched.rs

const std = @import("std");
const Allocator = std.mem.Allocator;

const msm = @import("../../msm/mod.zig");
const field = @import("../../field/mod.zig");
const pairing = field.pairing;
const commitment = @import("mod.zig");

/// Opening claim for batch verification
pub fn OpeningClaim(comptime F: type) type {
    const Point = msm.AffinePoint(F);

    return struct {
        const Self = @This();

        /// The committed polynomial's commitment
        commitment: Point,
        /// The evaluation point (r1, r2, ..., rn)
        point: []const F,
        /// The claimed evaluation value
        value: F,
        /// Quotient commitments (for HyperKZG)
        quotients: ?[]const Point,

        pub fn init(
            comm: Point,
            point: []const F,
            value: F,
            quotients: ?[]const Point,
        ) Self {
            return Self{
                .commitment = comm,
                .point = point,
                .value = value,
                .quotients = quotients,
            };
        }
    };
}

/// Batch opening accumulator
pub fn BatchOpeningAccumulator(comptime F: type) type {
    const Point = msm.AffinePoint(F);
    const G2Point = pairing.G2Point;
    const Claim = OpeningClaim(F);

    return struct {
        const Self = @This();

        /// Accumulated claims
        claims: std.ArrayListUnmanaged(Claim),
        /// Allocator
        allocator: Allocator,

        pub fn init(allocator: Allocator) Self {
            return Self{
                .claims = .{},
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            self.claims.deinit(self.allocator);
        }

        /// Add an opening claim
        pub fn addClaim(
            self: *Self,
            commitment_point: Point,
            point: []const F,
            value: F,
            quotients: ?[]const Point,
        ) !void {
            try self.claims.append(self.allocator, Claim.init(
                commitment_point,
                point,
                value,
                quotients,
            ));
        }

        /// Verify all claims with a single batched pairing check
        ///
        /// The batched verification uses random linear combination:
        /// 1. Sample random coefficients gamma_i from the transcript
        /// 2. Compute combined commitment: C' = sum_i gamma_i * C_i
        /// 3. Compute combined evaluation: v' = sum_i gamma_i * v_i
        /// 4. Compute combined witness: W' = sum_i gamma_i * W_i
        /// 5. Verify: e(C' - v'*G1, G2) == e(W', [tau]_2)
        pub fn verifyBatch(
            self: *const Self,
            g1: Point,
            g2: G2Point,
            tau_g2: G2Point,
            transcript: anytype,
        ) bool {
            if (self.claims.items.len == 0) {
                return true;
            }

            // Get batching coefficients from transcript
            var gamma = F.one();
            var gamma_power = F.one();

            // Accumulate combined values
            var combined_commitment = Point.identity();
            var combined_value = F.zero();
            var combined_witness = Point.identity();

            for (self.claims.items) |claim| {
                // Update gamma power for this claim
                gamma_power = gamma_power.mul(gamma);

                // Add gamma^i * commitment
                const scaled_comm = msm.MSM(F, F).scalarMul(
                    claim.commitment.toProjective(),
                    gamma_power,
                ).toAffine();
                combined_commitment = combined_commitment.add(scaled_comm);

                // Add gamma^i * value
                combined_value = combined_value.add(gamma_power.mul(claim.value));

                // Add gamma^i * witness (from quotients)
                if (claim.quotients) |quotients| {
                    for (quotients) |q| {
                        const scaled_q = msm.MSM(F, F).scalarMul(
                            q.toProjective(),
                            gamma_power,
                        ).toAffine();
                        combined_witness = combined_witness.add(scaled_q);
                    }
                }

                // Get next gamma from transcript
                gamma = transcript.challengeScalar("batch_gamma");
            }

            // Compute v' * G1
            const v_g1 = msm.MSM(F, F).scalarMul(g1.toProjective(), combined_value).toAffine();

            // Compute C' - v' * G1
            const lhs = combined_commitment.add(v_g1.neg());

            // Perform pairing check: e(C' - v'*G1, G2) == e(W', [tau]_2)
            return pairing.pairingCheck(lhs, g2, combined_witness, tau_g2);
        }

        /// Verify all claims individually (for debugging/testing)
        pub fn verifyIndividual(
            self: *const Self,
            g1: Point,
            g2: G2Point,
            tau_g2: G2Point,
        ) bool {
            for (self.claims.items) |claim| {
                // Compute v * G1
                const v_g1 = msm.MSM(F, F).scalarMul(g1.toProjective(), claim.value).toAffine();

                // Compute C - v * G1
                const lhs = claim.commitment.add(v_g1.neg());

                // Combine quotient commitments
                var witness = Point.identity();
                if (claim.quotients) |quotients| {
                    for (quotients) |q| {
                        witness = witness.add(q);
                    }
                }

                // Pairing check for this claim
                if (!pairing.pairingCheck(lhs, g2, witness, tau_g2)) {
                    return false;
                }
            }

            return true;
        }

        /// Get number of accumulated claims
        pub fn numClaims(self: *const Self) usize {
            return self.claims.items.len;
        }
    };
}

/// Helper to convert stage proof opening claims to batch verification format
pub fn OpeningClaimConverter(comptime F: type) type {
    const Point = msm.AffinePoint(F);

    return struct {
        const Self = @This();

        /// Convert verifier's accumulated opening claims to batch format
        pub fn fromVerifierAccumulator(
            accumulator: anytype,
            commitments: []const Point,
            allocator: Allocator,
        ) !BatchOpeningAccumulator(F) {
            var batch = BatchOpeningAccumulator(F).init(allocator);

            // The accumulator should have pairs of (point, claimed_eval)
            // Match them with the corresponding commitments
            const points = accumulator.opening_points.items;
            const evals = accumulator.claimed_evals.items;

            const num_claims = @min(points.len, @min(evals.len, commitments.len));

            for (0..num_claims) |i| {
                try batch.addClaim(
                    commitments[i],
                    points[i],
                    evals[i],
                    null, // Quotients would come from the proof
                );
            }

            return batch;
        }
    };
}

// ============================================================================
// Tests
// ============================================================================

test "batch opening accumulator init" {
    const allocator = std.testing.allocator;
    const F = field.BN254Scalar;

    var acc = BatchOpeningAccumulator(F).init(allocator);
    defer acc.deinit();

    try std.testing.expectEqual(@as(usize, 0), acc.numClaims());
}

test "batch opening accumulator add claim" {
    const allocator = std.testing.allocator;
    const F = field.BN254Scalar;
    const Point = msm.AffinePoint(F);

    var acc = BatchOpeningAccumulator(F).init(allocator);
    defer acc.deinit();

    const comm = Point.generator();
    const point = [_]F{ F.one(), F.fromU64(2) };
    const value = F.fromU64(42);

    try acc.addClaim(comm, &point, value, null);

    try std.testing.expectEqual(@as(usize, 1), acc.numClaims());
}
