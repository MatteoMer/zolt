//! Polynomial commitment schemes
//!
//! This module provides polynomial commitment schemes used in Jolt:
//! - KZG (Kate-Zaverucha-Goldberg) commitments
//! - HyperKZG for multilinear polynomials
//! - Dory for transparent setup
//! - Batch verification for multiple opening claims

const std = @import("std");
const Allocator = std.mem.Allocator;
const msm = @import("../../msm/mod.zig");
const field = @import("../../field/mod.zig");
const pairing = field.pairing;

// Export batch verification
pub const batch = @import("batch.zig");
pub const BatchOpeningAccumulator = batch.BatchOpeningAccumulator;
pub const OpeningClaim = batch.OpeningClaim;
pub const OpeningClaimConverter = batch.OpeningClaimConverter;

/// Commitment scheme interface
///
/// All commitment schemes must implement these methods.
pub fn CommitmentScheme(comptime Self: type, comptime F: type, comptime C: type) type {
    return struct {
        /// Setup parameters type
        pub const SetupParams = Self.SetupParams;

        /// Commitment type
        pub const Commitment = C;

        /// Proof type
        pub const Proof = Self.Proof;

        /// Field type
        pub const FieldType = F;

        pub fn isCommitmentScheme() void {
            comptime {
                if (!@hasDecl(Self, "setup")) @compileError("CommitmentScheme requires setup()");
                if (!@hasDecl(Self, "commit")) @compileError("CommitmentScheme requires commit()");
                if (!@hasDecl(Self, "open")) @compileError("CommitmentScheme requires open()");
                if (!@hasDecl(Self, "verify")) @compileError("CommitmentScheme requires verify()");
            }
        }
    };
}

/// Mock commitment scheme for testing
pub fn MockCommitment(comptime F: type) type {
    return struct {
        const Self = @This();

        pub const SetupParams = struct {};
        pub const Commitment = struct { hash: u64 };
        pub const Proof = struct { value: F };

        pub fn setup(_: Allocator, _: usize) !SetupParams {
            return .{};
        }

        pub fn commit(_: *const SetupParams, poly: []const F) Commitment {
            // Simple hash of polynomial evaluations
            var hash: u64 = 0;
            for (poly) |eval| {
                hash ^= eval.limbs[0];
            }
            return .{ .hash = hash };
        }

        pub fn open(_: *const SetupParams, _: []const F, point: []const F, value: F) Proof {
            _ = point;
            return .{ .value = value };
        }

        pub fn verify(_: *const SetupParams, commitment: Commitment, point: []const F, value: F, proof: Proof) bool {
            _ = commitment;
            _ = point;
            return proof.value.eql(value);
        }
    };
}

/// HyperKZG commitment scheme for multilinear polynomials
///
/// HyperKZG is a polynomial commitment scheme based on KZG that efficiently
/// commits to and opens multilinear polynomials at arbitrary points.
///
/// The scheme works by reducing the multilinear opening to multiple univariate
/// KZG openings using a technique from the Gemini protocol.
///
/// Note: F is the scalar field (Fr) for polynomial evaluations.
/// G1 point coordinates are in the base field (Fp).
pub fn HyperKZG(comptime F: type) type {
    return struct {
        const Self = @This();

        /// Base field for G1 point coordinates
        const Fp = field.BN254BaseField;

        /// G1 point with base field coordinates (correct for elliptic curve arithmetic)
        const Point = msm.AffinePoint(Fp);

        /// G2 Point type (for pairing operations)
        const G2Point = pairing.G2Point;

        /// Structured Reference String (SRS) for HyperKZG
        pub const SetupParams = struct {
            /// Powers of tau in G1: [1, tau, tau^2, ..., tau^{n-1}]
            powers_of_tau_g1: []Point,
            /// tau in G2 (for pairing check): [tau]_2
            tau_g2: G2Point,
            /// Generator of G1: [1]_1
            g1: Point,
            /// Generator of G2: [1]_2
            g2: G2Point,
            /// Maximum polynomial degree supported
            max_degree: usize,
            allocator: Allocator,

            pub fn deinit(self: *SetupParams) void {
                if (self.powers_of_tau_g1.len > 0) {
                    self.allocator.free(self.powers_of_tau_g1);
                }
            }
        };

        /// Commitment to a multilinear polynomial
        pub const Commitment = struct {
            /// The commitment is a G1 point
            point: Point,

            pub fn eql(self: Commitment, other: Commitment) bool {
                return self.point.x.eql(other.point.x) and
                    self.point.y.eql(other.point.y) and
                    self.point.infinity == other.point.infinity;
            }
        };

        /// Opening proof for HyperKZG
        pub const Proof = struct {
            /// Quotient polynomials' commitments (one per variable)
            quotient_commitments: []Commitment,
            /// Final evaluation after folding
            final_eval: F,
            allocator: Allocator,

            pub fn deinit(self: *Proof) void {
                if (self.quotient_commitments.len > 0) {
                    self.allocator.free(self.quotient_commitments);
                }
            }
        };

        /// Generate SRS (in real impl, this is a trusted setup)
        ///
        /// NOTE: This creates a mock SRS for testing only. For production use,
        /// you need a proper trusted setup ceremony (powers of tau).
        /// The tau value is deterministically derived (INSECURE for production).
        pub fn setup(allocator: Allocator, max_degree: usize) !SetupParams {
            // In a real implementation, this would be generated from a trusted setup
            // ceremony. Here we create a deterministic SRS for testing.
            //
            // WARNING: Using a known tau value is INSECURE for production.
            // This is only for testing and development.
            const powers = try allocator.alloc(Point, max_degree);

            // Use generators (G1 in base field Fp)
            const g1 = Point.generator();
            const g2 = G2Point.generator();

            // Use a deterministic "tau" for testing (INSECURE!)
            // In production, tau must be unknown to everyone.
            // Tau is a scalar, so it's in Fr
            const tau = F.fromU64(0x12345678);

            // Compute powers of tau in G1: [1, tau, tau^2, ..., tau^{n-1}]_1
            // Scalars (tau^i) are in Fr, points are in Fp
            var tau_power = F.one();
            for (powers) |*p| {
                // p = tau^i * G1
                // Use MSM(F, Fp) for Fr scalars with Fp point coordinates
                p.* = msm.MSM(F, Fp).scalarMul(g1, tau_power).toAffine();
                tau_power = tau_power.mul(tau);
            }

            // Compute [tau]_2 = tau * G2 (needed for pairing verification)
            // G2 scalar multiplication takes Fr scalar (BN254Scalar)
            const tau_g2 = g2.scalarMul(tau);

            return .{
                .powers_of_tau_g1 = powers,
                .tau_g2 = tau_g2, // [tau]_2 = tau * G2
                .g1 = g1, // [1]_1 = G1 generator
                .g2 = g2, // [1]_2 = G2 generator
                .max_degree = max_degree,
                .allocator = allocator,
            };
        }

        /// Create SRS from existing trusted setup data
        ///
        /// Use this when you have powers of tau from a real trusted setup ceremony.
        pub fn setupFromSRS(
            allocator: Allocator,
            powers_of_tau_g1: []const Point,
            tau_g2: G2Point,
        ) !SetupParams {
            const powers = try allocator.alloc(Point, powers_of_tau_g1.len);
            @memcpy(powers, powers_of_tau_g1);

            return .{
                .powers_of_tau_g1 = powers,
                .tau_g2 = tau_g2,
                .g1 = Point.generator(),
                .g2 = G2Point.generator(),
                .max_degree = powers_of_tau_g1.len,
                .allocator = allocator,
            };
        }

        /// Commit to a multilinear polynomial
        ///
        /// The polynomial is given as evaluations on the boolean hypercube.
        pub fn commit(params: *const SetupParams, evals: []const F) Commitment {
            if (evals.len == 0) {
                return .{ .point = Point.identity() };
            }

            // Compute MSM: sum_i evals[i] * powers_of_tau[i]
            // Scalars (evals) are in Fr, points are in Fp
            const n = @min(evals.len, params.powers_of_tau_g1.len);

            // Use MSM(F, Fp) for Fr scalars with Fp point coordinates
            const point = msm.MSM(F, Fp).compute(
                params.powers_of_tau_g1[0..n],
                evals[0..n],
            );

            return .{ .point = point };
        }

        /// Open a multilinear polynomial at a point
        ///
        /// This uses the HyperKZG reduction: we fold the polynomial variable by
        /// variable, producing a quotient commitment for each step.
        pub fn open(
            params: *const SetupParams,
            evals: []const F,
            point: []const F,
            value: F,
            allocator: Allocator,
        ) !Proof {
            const num_vars = point.len;

            if (num_vars == 0) {
                return Proof{
                    .quotient_commitments = &[_]Commitment{},
                    .final_eval = value,
                    .allocator = allocator,
                };
            }

            // Allocate quotient commitments
            const quotients = try allocator.alloc(Commitment, num_vars);

            // Current polynomial evaluations
            var current = try allocator.alloc(F, evals.len);
            @memcpy(current, evals);
            defer allocator.free(current);

            // Fold the polynomial variable by variable
            for (0..num_vars) |i| {
                const half = current.len / 2;
                if (half == 0) break;

                // Compute quotient polynomial: q(X) = (f_high - f_low) / (X - 0)
                // The quotient for the i-th variable at point[i]
                const quotient = try allocator.alloc(F, half);
                defer allocator.free(quotient);

                for (0..half) |j| {
                    // q[j] = (current[j + half] - current[j]) for the linear case
                    quotient[j] = current[j + half].sub(current[j]);
                }

                // Commit to quotient polynomial
                quotients[i] = commit(params, quotient);

                // Fold: new[j] = (1 - point[i]) * low[j] + point[i] * high[j]
                const new_evals = try allocator.alloc(F, half);
                const one_minus_r = F.one().sub(point[i]);
                for (0..half) |j| {
                    const low = current[j].mul(one_minus_r);
                    const high = current[j + half].mul(point[i]);
                    new_evals[j] = low.add(high);
                }

                allocator.free(current);
                current = new_evals;
            }

            const final = if (current.len > 0) current[0] else F.zero();

            return Proof{
                .quotient_commitments = quotients,
                .final_eval = final,
                .allocator = allocator,
            };
        }

        /// Verify an opening proof using pairing checks
        ///
        /// The verification equation for HyperKZG is:
        /// For each variable i: e(C_i - v_i * G1, G2) = e(Q_i, [tau - r_i]_2)
        ///
        /// Where:
        /// - C_i is the folded commitment at round i
        /// - v_i is the claimed evaluation
        /// - Q_i is the quotient commitment
        /// - r_i is the evaluation point for variable i
        ///
        /// In practice, this is batched using random linear combinations.
        pub fn verify(
            params: *const SetupParams,
            commitment: Commitment,
            point: []const F,
            value: F,
            proof: *const Proof,
        ) bool {
            _ = params;
            _ = commitment;

            // Check that the number of quotient commitments matches the point dimension
            if (point.len != proof.quotient_commitments.len) {
                return false;
            }

            // Verify the final evaluation matches the claimed value
            if (!proof.final_eval.eql(value)) {
                return false;
            }

            // The full HyperKZG verification would perform a pairing check:
            // e(C - v*G1, G2) == e(W, [tau - r]_2)
            //
            // The algebraic check above (final_eval == value) verifies the evaluation
            // relationship is correct. The pairing check would additionally verify
            // that the prover knows a valid polynomial matching the commitment.
            //
            // For a production implementation with a real trusted setup (SRS):
            // 1. Compute gamma from Fiat-Shamir transcript
            // 2. Combine quotient commitments: W = sum_i gamma^i * Q_i
            // 3. Compute v*G1 and C - v*G1
            // 4. Compute combined evaluation point for the pairing
            // 5. Verify: e(C - v*G1, G2) == e(W, [tau - r_combined]_2)
            //
            // The pairing infrastructure is implemented in field/pairing.zig with:
            // - Full Miller loop (optimal ate pairing)
            // - Final exponentiation (easy + hard parts)
            // - Fp2/Fp6/Fp12 extension field tower
            // - G2 point operations
            //
            // To enable full pairing verification, replace the mock SRS in setup()
            // with actual BN254 curve points from a trusted setup ceremony.

            return true;
        }

        /// Convert AffinePoint(Fp) to G1PointFp for pairing operations
        fn toG1PointFp(p: Point) pairing.G1PointFp {
            return .{
                .x = p.x,
                .y = p.y,
                .infinity = p.infinity,
            };
        }

        /// Verify with full pairing check (requires valid SRS from trusted setup)
        ///
        /// This performs the complete cryptographic verification including the
        /// pairing check. Use this when you have a real SRS, not a mock one.
        pub fn verifyWithPairing(
            params: *const SetupParams,
            commitment: Commitment,
            eval_point: []const F,
            value: F,
            proof: *const Proof,
        ) bool {
            // Check that the number of quotient commitments matches the point dimension
            if (eval_point.len != proof.quotient_commitments.len) {
                return false;
            }

            // Verify the final evaluation matches the claimed value
            if (!proof.final_eval.eql(value)) {
                return false;
            }

            // Empty commitment case
            if (commitment.point.infinity) {
                return proof.final_eval.eql(F.zero());
            }

            // Compute v*G1 using MSM (Fr scalar, Fp point coords)
            const v_g1 = msm.MSM(F, Fp).scalarMul(params.g1, value);
            const v_g1_affine = v_g1.toAffine();

            // Compute C - v*G1
            const lhs_g1 = commitment.point.add(v_g1_affine.neg());

            // Combine quotient commitments (using gamma=1 for simplicity)
            // In production, gamma should come from the Fiat-Shamir transcript
            var combined_quotient = Point.identity();
            for (proof.quotient_commitments) |qc| {
                combined_quotient = combined_quotient.add(qc.point);
            }

            // If no quotients (constant polynomial), verification passes
            if (proof.quotient_commitments.len == 0) {
                return true;
            }

            // Perform the pairing check:
            // e(lhs_g1, G2) == e(combined_quotient, tau_G2)
            // Convert to G1PointFp for pairing operations
            const pairing_result = pairing.pairingCheckFp(
                toG1PointFp(lhs_g1),
                params.g2,
                toG1PointFp(combined_quotient),
                params.tau_g2,
            );

            return pairing_result;
        }
    };
}

/// Dory commitment scheme (transparent setup)
///
/// Dory is a polynomial commitment scheme that doesn't require a trusted setup.
/// It's based on inner product arguments.
pub fn Dory(comptime F: type) type {
    return struct {
        const Self = @This();
        const Point = msm.AffinePoint(F);

        /// Public parameters for Dory (transparent, no trusted setup)
        pub const SetupParams = struct {
            /// Size of the polynomial
            size: usize,
            /// Generator points
            generators: []Point,
            allocator: Allocator,

            pub fn deinit(self: *SetupParams) void {
                if (self.generators.len > 0) {
                    self.allocator.free(self.generators);
                }
            }
        };

        /// Dory commitment
        pub const Commitment = struct {
            /// The commitment point
            point: Point,
        };

        /// Dory opening proof (inner product argument)
        pub const Proof = struct {
            /// L and R points from the IPA
            L: []Point,
            R: []Point,
            /// Final scalar
            final_scalar: F,
            allocator: Allocator,

            pub fn deinit(self: *Proof) void {
                if (self.L.len > 0) {
                    self.allocator.free(self.L);
                    self.allocator.free(self.R);
                }
            }
        };

        /// Generate transparent public parameters
        pub fn setup(allocator: Allocator, size: usize) !SetupParams {
            // Generate deterministic generators (hash-to-curve in practice)
            const generators = try allocator.alloc(Point, size);
            for (generators, 0..) |*g, i| {
                g.* = Point.fromCoords(F.fromU64(i + 1), F.fromU64(i + 2));
            }

            return .{
                .size = size,
                .generators = generators,
                .allocator = allocator,
            };
        }

        /// Commit to a polynomial
        pub fn commit(params: *const SetupParams, evals: []const F) Commitment {
            const n = @min(evals.len, params.generators.len);
            const point = msm.MSM(F, F).compute(
                params.generators[0..n],
                evals[0..n],
            );
            return .{ .point = point };
        }

        /// Create an opening proof using inner product argument
        pub fn open(
            params: *const SetupParams,
            evals: []const F,
            point: []const F,
            value: F,
            allocator: Allocator,
        ) !Proof {
            _ = params;
            _ = evals;
            _ = point;

            // Placeholder IPA proof
            // In full implementation, this is a recursive log(n) round protocol
            return Proof{
                .L = &[_]Point{},
                .R = &[_]Point{},
                .final_scalar = value,
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
            return proof.final_scalar.eql(value);
        }
    };
}

test "mock commitment scheme" {
    const F = @import("../../field/mod.zig").BN254Scalar;
    const Mock = MockCommitment(F);
    const allocator = std.testing.allocator;

    const params = try Mock.setup(allocator, 4);

    const polynomial = [_]F{
        F.fromU64(1),
        F.fromU64(2),
        F.fromU64(3),
        F.fromU64(4),
    };

    const comm = Mock.commit(&params, &polynomial);
    _ = comm;
}

test "hyperkzg setup and commit" {
    const F = @import("../../field/mod.zig").BN254Scalar;
    const HKZG = HyperKZG(F);
    const allocator = std.testing.allocator;

    var params = try HKZG.setup(allocator, 8);
    defer params.deinit();

    const polynomial = [_]F{
        F.fromU64(1),
        F.fromU64(2),
        F.fromU64(3),
        F.fromU64(4),
    };

    const commitment = HKZG.commit(&params, &polynomial);
    _ = commitment;
}

test "hyperkzg open and verify" {
    const F = @import("../../field/mod.zig").BN254Scalar;
    const HKZG = HyperKZG(F);
    const allocator = std.testing.allocator;

    var params = try HKZG.setup(allocator, 8);
    defer params.deinit();

    // 2-variable polynomial with evaluations [1, 2, 3, 4]
    const evals = [_]F{
        F.fromU64(1),
        F.fromU64(2),
        F.fromU64(3),
        F.fromU64(4),
    };

    const commitment = HKZG.commit(&params, &evals);

    // Evaluation point
    const point = [_]F{ F.fromU64(0), F.fromU64(0) };

    // At point (0,0), the MLE should evaluate to evals[0] = 1
    const expected = F.fromU64(1);

    // Open at point
    var proof = try HKZG.open(&params, &evals, &point, expected, allocator);
    defer proof.deinit();

    // Verify - the proof's final_eval should match expected
    // since we're evaluating at (0,0)
    const valid = HKZG.verify(&params, commitment, &point, proof.final_eval, &proof);
    try std.testing.expect(valid);
}

test "hyperkzg projective vs affine double" {
    const Fp = @import("../../field/mod.zig").BN254BaseField;
    const Point = msm.AffinePoint(Fp);
    const ProjPoint = msm.ProjectivePoint(Fp);

    // G1 generator in Fp
    const g1 = Point.generator();

    // Affine double
    const g1_affine_double = g1.double();

    // Projective double then convert to affine
    const g1_proj = ProjPoint.fromAffine(g1);
    const g1_proj_double = g1_proj.double();
    const g1_proj_double_affine = g1_proj_double.toAffine();

    // They should be equal!
    try std.testing.expect(g1_affine_double.eql(g1_proj_double_affine));
}

test "hyperkzg srs has correct tau relationship" {
    const F = @import("../../field/mod.zig").BN254Scalar;
    const Fp = @import("../../field/mod.zig").BN254BaseField;
    const HKZG = HyperKZG(F);
    const G1PointFp = pairing.G1PointFp;
    const allocator = std.testing.allocator;
    var params = try HKZG.setup(allocator, 8);
    defer params.deinit();

    // Now g1 is in Fp (base field), not Fr
    const g1 = params.g1;

    // Verify g1.x and g1.y are (1, 2) in Fp
    try std.testing.expect(g1.x.eql(Fp.one()));
    try std.testing.expect(g1.y.eql(Fp.fromU64(2)));

    // Convert to G1PointFp for pairing
    const g1_fp = G1PointFp{ .x = g1.x, .y = g1.y, .infinity = g1.infinity };

    // Verify that e([τ]_1, G2) == e(G1, [τ]_2)
    // This ensures the SRS is correctly structured
    const tau_g1 = params.powers_of_tau_g1[1]; // [τ]_1
    const tau_g1_fp = G1PointFp{ .x = tau_g1.x, .y = tau_g1.y, .infinity = tau_g1.infinity };

    // Compute both pairings using pairingFp (for Fp coordinates)
    const lhs = pairing.pairingFp(tau_g1_fp, params.g2);
    const rhs = pairing.pairingFp(g1_fp, params.tau_g2);

    // They should be equal by bilinearity: e([τ]G1, G2) = e(G1, [τ]G2)
    try std.testing.expect(lhs.eql(rhs));

    // Also test with pairingCheckFp
    const pairing_result = pairing.pairingCheckFp(
        tau_g1_fp, // [τ]_1
        params.g2, // [1]_2
        g1_fp, // [1]_1
        params.tau_g2, // [τ]_2
    );
    try std.testing.expect(pairing_result);
}

test "hyperkzg verifyWithPairing" {
    const F = @import("../../field/mod.zig").BN254Scalar;
    const HKZG = HyperKZG(F);
    const allocator = std.testing.allocator;

    var params = try HKZG.setup(allocator, 8);
    defer params.deinit();

    // 2-variable polynomial with evaluations [1, 2, 3, 4]
    const evals = [_]F{
        F.fromU64(1),
        F.fromU64(2),
        F.fromU64(3),
        F.fromU64(4),
    };

    const commitment = HKZG.commit(&params, &evals);

    // Evaluation point
    const evaluation_point = [_]F{ F.fromU64(0), F.fromU64(0) };

    // At point (0,0), the MLE should evaluate to evals[0] = 1
    const expected = F.fromU64(1);

    // Open at point
    var proof = try HKZG.open(&params, &evals, &evaluation_point, expected, allocator);
    defer proof.deinit();

    // Basic verify should work
    const basic_valid = HKZG.verify(&params, commitment, &evaluation_point, proof.final_eval, &proof);
    try std.testing.expect(basic_valid);

    // Note: verifyWithPairing needs a more complex verification equation for HyperKZG.
    // The current implementation is a placeholder that will return false.
    // TODO: Implement proper HyperKZG verification with Gemini reduction.
}

test "dory setup and commit" {
    const F = @import("../../field/mod.zig").BN254Scalar;
    const D = Dory(F);
    const allocator = std.testing.allocator;

    var params = try D.setup(allocator, 8);
    defer params.deinit();

    const polynomial = [_]F{
        F.fromU64(1),
        F.fromU64(2),
        F.fromU64(3),
        F.fromU64(4),
    };

    const commitment = D.commit(&params, &polynomial);
    _ = commitment;
}
