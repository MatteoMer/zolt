//! Polynomial commitment schemes
//!
//! This module provides polynomial commitment schemes used in Jolt:
//! - KZG (Kate-Zaverucha-Goldberg) commitments
//! - HyperKZG for multilinear polynomials
//! - Dory for transparent setup

const std = @import("std");
const Allocator = std.mem.Allocator;
const msm = @import("../../msm/mod.zig");
const field = @import("../../field/mod.zig");
const pairing = field.pairing;

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
pub fn HyperKZG(comptime F: type) type {
    return struct {
        const Self = @This();
        const Point = msm.AffinePoint(F);

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
        pub fn setup(allocator: Allocator, max_degree: usize) !SetupParams {
            // In a real implementation, this would be generated from a trusted setup
            // ceremony. Here we create a mock SRS for testing.
            const powers = try allocator.alloc(Point, max_degree);

            // Initialize with mock values (in practice, these are [tau^i]_1)
            for (powers, 0..) |*p, i| {
                p.* = Point.fromCoords(F.fromU64(i + 1), F.fromU64(i + 2));
            }

            return .{
                .powers_of_tau_g1 = powers,
                .tau_g2 = G2Point.generator(), // [tau]_2
                .g1 = Point.fromCoords(F.one(), F.fromU64(2)), // [1]_1
                .g2 = G2Point.generator(), // [1]_2
                .max_degree = max_degree,
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
            const n = @min(evals.len, params.powers_of_tau_g1.len);

            // Use MSM to compute the commitment
            const point = msm.MSM(F, F).compute(
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
            _ = commitment;
            _ = point;

            // For a complete implementation, we would verify:
            // 1. Reconstruct the folded commitment from quotients
            // 2. Check that the final evaluation matches the claimed value
            // 3. Verify the pairing equation:
            //    e(C - v*G1, G2) == e(W, [tau]_2 - r*G2)
            //
            // The pairing check ensures that the prover knows a polynomial
            // that matches the commitment and evaluates correctly at the point.

            // For now, check the algebraic relationship
            if (!proof.final_eval.eql(value)) {
                return false;
            }

            // In production, verify the pairing equation:
            // pairingCheck(commitment.point.sub(params.g1.scalarMul(value)),
            //              params.g2,
            //              quotient_combined,
            //              tau_minus_r_g2)

            // Placeholder: pairing check would go here
            // const lhs = pairing.pairing(left_g1, params.g2);
            // const rhs = pairing.pairing(right_g1, tau_g2);
            // return lhs.eql(rhs);

            _ = params;
            return true;
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
    const point = [_]F{ F.fromU64(5), F.fromU64(7) };

    // Compute expected value using multilinear extension
    // f(x0, x1) = (1-x0)(1-x1)*1 + x0(1-x1)*2 + (1-x0)x1*3 + x0*x1*4
    const x0 = point[0];
    const x1 = point[1];
    const one_minus_x0 = F.one().sub(x0);
    const one_minus_x1 = F.one().sub(x1);

    const v00 = one_minus_x0.mul(one_minus_x1).mul(F.fromU64(1));
    const v10 = x0.mul(one_minus_x1).mul(F.fromU64(2));
    const v01 = one_minus_x0.mul(x1).mul(F.fromU64(3));
    const v11 = x0.mul(x1).mul(F.fromU64(4));
    const expected = v00.add(v10).add(v01).add(v11);

    // Open at point
    var proof = try HKZG.open(&params, &evals, &point, expected, allocator);
    defer proof.deinit();

    // Verify
    const valid = HKZG.verify(&params, commitment, &point, expected, &proof);
    try std.testing.expect(valid);
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
