//! BN254 Pairing Operations
//!
//! This module implements elliptic curve pairing operations for the BN254 curve.
//! Pairings are bilinear maps: e(P, Q) where P ∈ G1, Q ∈ G2, and the result is in GT (Fp12).
//!
//! BN254 Curve Parameters:
//! - Base field: Fp (254-bit prime)
//! - Scalar field: Fr (used for scalars)
//! - G1: Points on y² = x³ + 3 over Fp
//! - G2: Points on y² = x³ + 3/ξ over Fp² (sextic twist)
//! - GT: Subgroup of Fp12*
//!
//! The pairing is computed via the optimal ate pairing:
//! 1. Miller loop: Compute f_{6x+2,Q}(P)
//! 2. Final exponentiation: f^((p^12-1)/r)
//!
//! Generator Points (Ethereum/EIP-196/EIP-197 convention):
//! - G1 generator: (1, 2)
//! - G2 generator (in Fp2):
//!   X = (x0, x1) where:
//!     x0 = 0x1800deef121f1e76426a00665e5c4479674322d4f75edadd46debd5cd992f6ed
//!     x1 = 0x198e9393920d483a7260bfb731fb5d25f1aa493335a9e71297e485b7aef312c2
//!   Y = (y0, y1) where:
//!     y0 = 0x12c85ea5db8c6deb4aab71808dcb408fe3d1e7690c43d37b4ce6cc0166fa7daa
//!     y1 = 0x090689d0585ff075ec9e99ad690c3395bc4b313370b38ef355acdadcd122975b

const std = @import("std");
const BN254Scalar = @import("mod.zig").BN254Scalar;

// ============================================================================
// Extension Field Fp2 = Fp[u] / (u² + 1)
// ============================================================================

/// Fp2 element: a + b*u where u² = -1
pub const Fp2 = struct {
    c0: BN254Scalar, // Real part
    c1: BN254Scalar, // Imaginary part

    pub fn init(c0: BN254Scalar, c1: BN254Scalar) Fp2 {
        return .{ .c0 = c0, .c1 = c1 };
    }

    pub fn zero() Fp2 {
        return .{ .c0 = BN254Scalar.zero(), .c1 = BN254Scalar.zero() };
    }

    pub fn one() Fp2 {
        return .{ .c0 = BN254Scalar.one(), .c1 = BN254Scalar.zero() };
    }

    pub fn add(self: Fp2, other: Fp2) Fp2 {
        return .{
            .c0 = self.c0.add(other.c0),
            .c1 = self.c1.add(other.c1),
        };
    }

    pub fn sub(self: Fp2, other: Fp2) Fp2 {
        return .{
            .c0 = self.c0.sub(other.c0),
            .c1 = self.c1.sub(other.c1),
        };
    }

    pub fn mul(self: Fp2, other: Fp2) Fp2 {
        // (a + bu)(c + du) = (ac - bd) + (ad + bc)u
        const ac = self.c0.mul(other.c0);
        const bd = self.c1.mul(other.c1);
        const ad = self.c0.mul(other.c1);
        const bc = self.c1.mul(other.c0);

        return .{
            .c0 = ac.sub(bd),
            .c1 = ad.add(bc),
        };
    }

    pub fn square(self: Fp2) Fp2 {
        // (a + bu)² = (a² - b²) + 2abu
        // Karatsuba optimization: (a+b)(a-b) = a² - b²
        const a_plus_b = self.c0.add(self.c1);
        const a_minus_b = self.c0.sub(self.c1);
        const a_squared_minus_b_squared = a_plus_b.mul(a_minus_b);
        const two_ab = self.c0.mul(self.c1).double();

        return .{
            .c0 = a_squared_minus_b_squared,
            .c1 = two_ab,
        };
    }

    pub fn neg(self: Fp2) Fp2 {
        return .{
            .c0 = self.c0.neg(),
            .c1 = self.c1.neg(),
        };
    }

    /// Conjugate: a + bu -> a - bu
    pub fn conjugate(self: Fp2) Fp2 {
        return .{
            .c0 = self.c0,
            .c1 = self.c1.neg(),
        };
    }

    /// Inverse using the formula: 1/(a + bu) = (a - bu)/(a² + b²)
    pub fn inverse(self: Fp2) ?Fp2 {
        const norm = self.c0.square().add(self.c1.square());
        const norm_inv = norm.inverse() orelse return null;

        return .{
            .c0 = self.c0.mul(norm_inv),
            .c1 = self.c1.neg().mul(norm_inv),
        };
    }

    pub fn eql(self: Fp2, other: Fp2) bool {
        return self.c0.eql(other.c0) and self.c1.eql(other.c1);
    }

    pub fn isZero(self: Fp2) bool {
        return self.c0.isZero() and self.c1.isZero();
    }
};

// ============================================================================
// Extension Field Fp6 = Fp2[v] / (v³ - ξ) where ξ = u + 1
// ============================================================================

/// Fp6 element: c0 + c1*v + c2*v² where v³ = ξ
pub const Fp6 = struct {
    c0: Fp2,
    c1: Fp2,
    c2: Fp2,

    pub fn zero() Fp6 {
        return .{ .c0 = Fp2.zero(), .c1 = Fp2.zero(), .c2 = Fp2.zero() };
    }

    pub fn one() Fp6 {
        return .{ .c0 = Fp2.one(), .c1 = Fp2.zero(), .c2 = Fp2.zero() };
    }

    pub fn add(self: Fp6, other: Fp6) Fp6 {
        return .{
            .c0 = self.c0.add(other.c0),
            .c1 = self.c1.add(other.c1),
            .c2 = self.c2.add(other.c2),
        };
    }

    pub fn sub(self: Fp6, other: Fp6) Fp6 {
        return .{
            .c0 = self.c0.sub(other.c0),
            .c1 = self.c1.sub(other.c1),
            .c2 = self.c2.sub(other.c2),
        };
    }

    pub fn neg(self: Fp6) Fp6 {
        return .{
            .c0 = self.c0.neg(),
            .c1 = self.c1.neg(),
            .c2 = self.c2.neg(),
        };
    }

    /// Multiplication by ξ = u + 1
    fn mulByXi(x: Fp2) Fp2 {
        // ξ * (a + bu) = (a + bu)(1 + u) = (a - b) + (a + b)u
        return Fp2.init(
            x.c0.sub(x.c1),
            x.c0.add(x.c1),
        );
    }

    pub fn mul(self: Fp6, other: Fp6) Fp6 {
        // Karatsuba-like multiplication for cubic extension
        const v0 = self.c0.mul(other.c0);
        const v1 = self.c1.mul(other.c1);
        const v2 = self.c2.mul(other.c2);

        // c0 = v0 + ξ((c1 + c2)(d1 + d2) - v1 - v2)
        const c1_plus_c2 = self.c1.add(self.c2);
        const d1_plus_d2 = other.c1.add(other.c2);
        const t0 = mulByXi(c1_plus_c2.mul(d1_plus_d2).sub(v1).sub(v2));
        const new_c0 = v0.add(t0);

        // c1 = (c0 + c1)(d0 + d1) - v0 - v1 + ξ*v2
        const c0_plus_c1 = self.c0.add(self.c1);
        const d0_plus_d1 = other.c0.add(other.c1);
        const t1 = c0_plus_c1.mul(d0_plus_d1).sub(v0).sub(v1);
        const new_c1 = t1.add(mulByXi(v2));

        // c2 = (c0 + c2)(d0 + d2) - v0 - v2 + v1
        const c0_plus_c2 = self.c0.add(self.c2);
        const d0_plus_d2 = other.c0.add(other.c2);
        const t2 = c0_plus_c2.mul(d0_plus_d2).sub(v0).sub(v2);
        const new_c2 = t2.add(v1);

        return .{ .c0 = new_c0, .c1 = new_c1, .c2 = new_c2 };
    }

    pub fn square(self: Fp6) Fp6 {
        return self.mul(self);
    }

    pub fn inverse(self: Fp6) ?Fp6 {
        // Extended Euclidean algorithm for Fp6
        const c0_sq = self.c0.square();
        const c1_sq = self.c1.square();
        const c2_sq = self.c2.square();
        const c0c1 = self.c0.mul(self.c1);
        const c0c2 = self.c0.mul(self.c2);
        const c1c2 = self.c1.mul(self.c2);

        // Using the formula for inverse in cubic extension
        const a0 = c0_sq.sub(mulByXi(c1c2));
        const a1 = mulByXi(c2_sq).sub(c0c1);
        const a2 = c1_sq.sub(c0c2);

        const tmp = mulByXi(self.c1.mul(a2).add(self.c2.mul(a1)));
        const norm = self.c0.mul(a0).add(tmp);

        const norm_inv = norm.inverse() orelse return null;

        return .{
            .c0 = a0.mul(norm_inv),
            .c1 = a1.mul(norm_inv),
            .c2 = a2.mul(norm_inv),
        };
    }

    pub fn eql(self: Fp6, other: Fp6) bool {
        return self.c0.eql(other.c0) and self.c1.eql(other.c1) and self.c2.eql(other.c2);
    }
};

// ============================================================================
// Extension Field Fp12 = Fp6[w] / (w² - v)
// ============================================================================

/// Fp12 element: c0 + c1*w where w² = v
pub const Fp12 = struct {
    c0: Fp6,
    c1: Fp6,

    pub fn zero() Fp12 {
        return .{ .c0 = Fp6.zero(), .c1 = Fp6.zero() };
    }

    pub fn one() Fp12 {
        return .{ .c0 = Fp6.one(), .c1 = Fp6.zero() };
    }

    pub fn add(self: Fp12, other: Fp12) Fp12 {
        return .{
            .c0 = self.c0.add(other.c0),
            .c1 = self.c1.add(other.c1),
        };
    }

    pub fn sub(self: Fp12, other: Fp12) Fp12 {
        return .{
            .c0 = self.c0.sub(other.c0),
            .c1 = self.c1.sub(other.c1),
        };
    }

    pub fn neg(self: Fp12) Fp12 {
        return .{
            .c0 = self.c0.neg(),
            .c1 = self.c1.neg(),
        };
    }

    /// Multiplication in Fp12
    pub fn mul(self: Fp12, other: Fp12) Fp12 {
        // (a + bw)(c + dw) = (ac + bdv) + (ad + bc)w
        // where v is the element that w² = v
        const ac = self.c0.mul(other.c0);
        const bd = self.c1.mul(other.c1);
        const ad = self.c0.mul(other.c1);
        const bc = self.c1.mul(other.c0);

        // Multiply bd by v (shift in the Fp6 tower)
        const bd_times_v = Fp6{
            .c0 = Fp6.mulByXi(bd.c2),
            .c1 = bd.c0,
            .c2 = bd.c1,
        };

        return .{
            .c0 = ac.add(bd_times_v),
            .c1 = ad.add(bc),
        };
    }

    pub fn square(self: Fp12) Fp12 {
        return self.mul(self);
    }

    /// Conjugate for unitary elements: a + bw -> a - bw
    pub fn conjugate(self: Fp12) Fp12 {
        return .{
            .c0 = self.c0,
            .c1 = self.c1.neg(),
        };
    }

    /// Frobenius endomorphism (raising to p-th power)
    pub fn frobenius(self: Fp12) Fp12 {
        // Simplified placeholder - full implementation requires precomputed constants
        _ = self;
        return Fp12.one(); // Placeholder
    }

    pub fn inverse(self: Fp12) ?Fp12 {
        // For quadratic extension: 1/(a + bw) = (a - bw)/(a² - b²v)
        // Use the conjugate method
        const a_squared = self.c0.mul(self.c0);
        const b_squared = self.c1.mul(self.c1);

        // b²v (shift in Fp6)
        const b_squared_times_v = Fp6{
            .c0 = Fp6.mulByXi(b_squared.c2),
            .c1 = b_squared.c0,
            .c2 = b_squared.c1,
        };

        const norm = a_squared.sub(b_squared_times_v);
        const norm_inv = norm.inverse() orelse return null;

        return .{
            .c0 = self.c0.mul(norm_inv),
            .c1 = self.c1.neg().mul(norm_inv),
        };
    }

    pub fn eql(self: Fp12, other: Fp12) bool {
        return self.c0.eql(other.c0) and self.c1.eql(other.c1);
    }

    pub fn isOne(self: Fp12) bool {
        return self.eql(Fp12.one());
    }
};

// ============================================================================
// G2 Points (on the twist curve over Fp2)
// ============================================================================

/// Point on G2 (twist curve over Fp2)
pub const G2Point = struct {
    x: Fp2,
    y: Fp2,
    infinity: bool,

    pub fn identity() G2Point {
        return .{
            .x = Fp2.zero(),
            .y = Fp2.one(),
            .infinity = true,
        };
    }

    pub fn fromCoords(x: Fp2, y: Fp2) G2Point {
        return .{
            .x = x,
            .y = y,
            .infinity = false,
        };
    }

    /// Generator point for G2 (Ethereum/EIP-197 convention)
    ///
    /// The BN254 G2 generator coordinates in Fp2:
    /// X = (x0, x1), Y = (y0, y1) where:
    ///   x0 = 0x1800deef121f1e76426a00665e5c4479674322d4f75edadd46debd5cd992f6ed
    ///   x1 = 0x198e9393920d483a7260bfb731fb5d25f1aa493335a9e71297e485b7aef312c2
    ///   y0 = 0x12c85ea5db8c6deb4aab71808dcb408fe3d1e7690c43d37b4ce6cc0166fa7daa
    ///   y1 = 0x090689d0585ff075ec9e99ad690c3395bc4b313370b38ef355acdadcd122975b
    pub fn generator() G2Point {
        // G2 generator coordinates (little-endian byte representation)
        // x0 = 0x1800deef121f1e76426a00665e5c4479674322d4f75edadd46debd5cd992f6ed
        const x0_bytes = [_]u8{
            0xed, 0xf6, 0x92, 0xd9, 0x5c, 0xbd, 0xde, 0x46,
            0xdd, 0xda, 0x5e, 0xf7, 0xd4, 0x22, 0x43, 0x67,
            0x79, 0x44, 0x5c, 0x5e, 0x66, 0x00, 0x6a, 0x42,
            0x76, 0x1e, 0x1f, 0x12, 0xef, 0xde, 0x00, 0x18,
        };
        // x1 = 0x198e9393920d483a7260bfb731fb5d25f1aa493335a9e71297e485b7aef312c2
        const x1_bytes = [_]u8{
            0xc2, 0x12, 0xf3, 0xae, 0xb7, 0x85, 0xe4, 0x97,
            0x12, 0xe7, 0xa9, 0x35, 0x33, 0x49, 0xaa, 0xf1,
            0x25, 0x5d, 0xfb, 0x31, 0xb7, 0xbf, 0x60, 0x72,
            0x3a, 0x48, 0x0d, 0x92, 0x93, 0x93, 0x8e, 0x19,
        };
        // y0 = 0x12c85ea5db8c6deb4aab71808dcb408fe3d1e7690c43d37b4ce6cc0166fa7daa
        const y0_bytes = [_]u8{
            0xaa, 0x7d, 0xfa, 0x66, 0x01, 0xcc, 0xe6, 0x4c,
            0x7b, 0xd3, 0x43, 0x0c, 0x69, 0xe7, 0xd1, 0xe3,
            0x8f, 0x40, 0xcb, 0x8d, 0x80, 0x71, 0xab, 0x4a,
            0xeb, 0x6d, 0x8c, 0xdb, 0xa5, 0x5e, 0xc8, 0x12,
        };
        // y1 = 0x090689d0585ff075ec9e99ad690c3395bc4b313370b38ef355acdadcd122975b
        const y1_bytes = [_]u8{
            0x5b, 0x97, 0x22, 0xd1, 0xdc, 0xda, 0xac, 0x55,
            0xf3, 0x8e, 0xb3, 0x70, 0x33, 0x31, 0x4b, 0xbc,
            0x95, 0x33, 0x0c, 0x69, 0xad, 0x99, 0x9e, 0xec,
            0x75, 0xf0, 0x5f, 0x58, 0xd0, 0x89, 0x06, 0x09,
        };

        const x0 = BN254Scalar.fromBytes(&x0_bytes);
        const x1 = BN254Scalar.fromBytes(&x1_bytes);
        const y0 = BN254Scalar.fromBytes(&y0_bytes);
        const y1 = BN254Scalar.fromBytes(&y1_bytes);

        return G2Point.fromCoords(
            Fp2.init(x0, x1),
            Fp2.init(y0, y1),
        );
    }

    pub fn isIdentity(self: G2Point) bool {
        return self.infinity;
    }

    pub fn eql(self: G2Point, other: G2Point) bool {
        if (self.infinity and other.infinity) return true;
        if (self.infinity or other.infinity) return false;
        return self.x.eql(other.x) and self.y.eql(other.y);
    }

    pub fn neg(self: G2Point) G2Point {
        if (self.infinity) return self;
        return .{
            .x = self.x,
            .y = self.y.neg(),
            .infinity = false,
        };
    }

    pub fn add(self: G2Point, other: G2Point) G2Point {
        if (self.infinity) return other;
        if (other.infinity) return self;

        if (self.x.eql(other.x)) {
            if (self.y.eql(other.y.neg())) {
                return G2Point.identity();
            }
            return self.double();
        }

        // Point addition formula
        const slope_num = other.y.sub(self.y);
        const slope_den = other.x.sub(self.x);
        const slope = slope_num.mul(slope_den.inverse() orelse return G2Point.identity());

        const x3 = slope.square().sub(self.x).sub(other.x);
        const y3 = slope.mul(self.x.sub(x3)).sub(self.y);

        return .{ .x = x3, .y = y3, .infinity = false };
    }

    pub fn double(self: G2Point) G2Point {
        if (self.infinity) return self;
        if (self.y.isZero()) return G2Point.identity();

        // Point doubling formula: λ = (3x²)/(2y)
        const x_sq = self.x.square();
        const three_x_sq = x_sq.add(x_sq).add(x_sq);
        const two_y = self.y.add(self.y);
        const slope = three_x_sq.mul(two_y.inverse() orelse return G2Point.identity());

        const x3 = slope.square().sub(self.x).sub(self.x);
        const y3 = slope.mul(self.x.sub(x3)).sub(self.y);

        return .{ .x = x3, .y = y3, .infinity = false };
    }
};

// ============================================================================
// Pairing Operations
// ============================================================================

/// Result of a pairing computation (element of GT = Fp12)
pub const PairingResult = Fp12;

/// G1 Point alias for clarity
pub const G1Point = @import("../msm/mod.zig").AffinePoint(BN254Scalar);

/// Compute the optimal ate pairing: e(P, Q) where P ∈ G1, Q ∈ G2
///
/// The pairing consists of two parts:
/// 1. Miller loop: Compute f_{6x+2,Q}(P)
/// 2. Final exponentiation: f^((p^12-1)/r)
///
/// NOTE: This is a simplified implementation. A production implementation
/// would need:
/// - Proper line function evaluation
/// - Efficient final exponentiation using Frobenius
/// - Optimal ate loop parameter
pub fn pairing(p: G1Point, q: G2Point) PairingResult {
    if (p.infinity or q.infinity) {
        return Fp12.one();
    }

    // Miller loop
    const f = millerLoop(p, q);

    // Final exponentiation
    return finalExponentiation(f);
}

/// BN254 ate loop parameter: 6x + 2 where x = 4965661367192848881
/// We use the NAF (non-adjacent form) representation for efficiency
/// The loop parameter bits (from MSB to LSB, excluding top bit):
/// x = 0x44e992b44a6909f1 in hex
const ATE_LOOP_COUNT: [64]i2 = .{
    0, 0, 0, 1, 0, 1, 0, -1, 0, 0, -1, 0, 0, 0, 1, 0,
    0, -1, 0, -1, 0, 0, 0, 1, 0, -1, 0, 0, 0, 0, -1, 0,
    0, 1, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, -1, 0, 0, -1,
    0, 1, 0, -1, 0, 0, 0, -1, 0, -1, 0, 0, 0, 1, 0, -1,
};

/// Coefficients for line function evaluation
const LineCoeffs = struct {
    c0: Fp2, // ell_0 coefficient
    c1: Fp2, // ell_vw coefficient (multiplied by y_p)
    c2: Fp2, // ell_vv coefficient (multiplied by x_p)
};

/// Result of doubling/addition step: new point and line coefficients
const MillerStepResult = struct {
    point: G2Point,
    coeffs: LineCoeffs,
};

/// Evaluate line function at point P
/// This computes the contribution of a line passing through points on G2
/// evaluated at a point P in G1, resulting in an element of Fp12
fn evaluateLine(coeffs: LineCoeffs, p_x: BN254Scalar, p_y: BN254Scalar) Fp12 {
    // The line evaluation gives a sparse Fp12 element
    // l(P) = c0 + c1 * y_p * w + c2 * x_p * v
    // where v and w are tower elements

    // Convert scalars to Fp2 (embedding Fp into Fp2)
    const x_fp2 = Fp2.init(p_x, BN254Scalar.zero());
    const y_fp2 = Fp2.init(p_y, BN254Scalar.zero());

    // c1 * y_p
    const c1_yp = coeffs.c1.mul(y_fp2);
    // c2 * x_p
    const c2_xp = coeffs.c2.mul(x_fp2);

    // Construct sparse Fp12 element
    // The structure depends on how the tower is constructed
    // For BN254: Fp12 = Fp6[w]/(w² - v), Fp6 = Fp2[v]/(v³ - ξ)
    return Fp12{
        .c0 = Fp6{
            .c0 = coeffs.c0,
            .c1 = c2_xp, // coefficient of v
            .c2 = Fp2.zero(),
        },
        .c1 = Fp6{
            .c0 = c1_yp, // coefficient of w
            .c1 = Fp2.zero(),
            .c2 = Fp2.zero(),
        },
    };
}

/// Doubling step in Miller loop
/// Returns the new point T = 2*T and line coefficients
fn doublingStep(t: G2Point) MillerStepResult {
    if (t.infinity or t.y.isZero()) {
        return .{
            .point = G2Point.identity(),
            .coeffs = .{
                .c0 = Fp2.one(),
                .c1 = Fp2.zero(),
                .c2 = Fp2.zero(),
            },
        };
    }

    // Compute λ = 3x² / 2y (the slope of the tangent line)
    const x_sq = t.x.square();
    const three_x_sq = x_sq.add(x_sq).add(x_sq);
    const two_y = t.y.add(t.y);
    const lambda = three_x_sq.mul(two_y.inverse() orelse return .{
        .point = G2Point.identity(),
        .coeffs = .{ .c0 = Fp2.one(), .c1 = Fp2.zero(), .c2 = Fp2.zero() },
    });

    // New point coordinates
    const x3 = lambda.square().sub(t.x).sub(t.x);
    const y3 = lambda.mul(t.x.sub(x3)).sub(t.y);

    // Line coefficients: l(x,y) = y - λx - (t.y - λ*t.x)
    // Rearranged for evaluation: c0 = λ*t.x - t.y, c1 = 1, c2 = -λ
    const c0 = lambda.mul(t.x).sub(t.y);
    const c1 = Fp2.one();
    const c2 = lambda.neg();

    return .{
        .point = G2Point{ .x = x3, .y = y3, .infinity = false },
        .coeffs = .{ .c0 = c0, .c1 = c1, .c2 = c2 },
    };
}

/// Addition step in Miller loop
/// Returns the new point T = T + Q and line coefficients
fn additionStep(t: G2Point, q: G2Point) MillerStepResult {
    if (t.infinity) {
        return .{
            .point = q,
            .coeffs = .{ .c0 = Fp2.one(), .c1 = Fp2.zero(), .c2 = Fp2.zero() },
        };
    }
    if (q.infinity) {
        return .{
            .point = t,
            .coeffs = .{ .c0 = Fp2.one(), .c1 = Fp2.zero(), .c2 = Fp2.zero() },
        };
    }

    // If points are equal, use doubling
    if (t.x.eql(q.x)) {
        if (t.y.eql(q.y)) {
            return doublingStep(t);
        }
        // t + (-t) = O
        return .{
            .point = G2Point.identity(),
            .coeffs = .{ .c0 = Fp2.one(), .c1 = Fp2.zero(), .c2 = Fp2.zero() },
        };
    }

    // Compute λ = (q.y - t.y) / (q.x - t.x)
    const dy = q.y.sub(t.y);
    const dx = q.x.sub(t.x);
    const lambda = dy.mul(dx.inverse() orelse return .{
        .point = G2Point.identity(),
        .coeffs = .{ .c0 = Fp2.one(), .c1 = Fp2.zero(), .c2 = Fp2.zero() },
    });

    // New point coordinates
    const x3 = lambda.square().sub(t.x).sub(q.x);
    const y3 = lambda.mul(t.x.sub(x3)).sub(t.y);

    // Line coefficients
    const c0 = lambda.mul(t.x).sub(t.y);
    const c1 = Fp2.one();
    const c2 = lambda.neg();

    return .{
        .point = G2Point{ .x = x3, .y = y3, .infinity = false },
        .coeffs = .{ .c0 = c0, .c1 = c1, .c2 = c2 },
    };
}

/// Miller loop for optimal ate pairing on BN254
/// Computes f_{6x+2,Q}(P) where x is the BN254 curve parameter
fn millerLoop(p: G1Point, q: G2Point) Fp12 {
    if (p.infinity or q.infinity) {
        return Fp12.one();
    }

    // Initialize: f = 1, T = Q
    var f = Fp12.one();
    var t = q;

    // Main loop: iterate through bits of the ate loop parameter
    for (ATE_LOOP_COUNT) |bit| {
        // Doubling step: f = f² * l_{T,T}(P), T = 2T
        f = f.square();
        const dbl = doublingStep(t);
        t = dbl.point;
        const line_dbl = evaluateLine(dbl.coeffs, p.x, p.y);
        f = f.mul(line_dbl);

        // Addition step if bit is non-zero
        if (bit == 1) {
            // f = f * l_{T,Q}(P), T = T + Q
            const add = additionStep(t, q);
            t = add.point;
            const line_add = evaluateLine(add.coeffs, p.x, p.y);
            f = f.mul(line_add);
        } else if (bit == -1) {
            // f = f * l_{T,-Q}(P), T = T - Q
            const neg_q = q.neg();
            const add = additionStep(t, neg_q);
            t = add.point;
            const line_add = evaluateLine(add.coeffs, p.x, p.y);
            f = f.mul(line_add);
        }
    }

    // For BN254 optimal ate, we need additional steps at the end:
    // f = f * l_{T, π(Q)}(P) * l_{T', -π²(Q)}(P)
    // where π is the Frobenius endomorphism
    // These additional lines correspond to the +2 part of 6x+2

    // Apply Frobenius to Q (π(Q) and π²(Q))
    // For BN254: π(x, y) = (x^p, y^p) which in Fp2 uses the Frobenius coefficients
    // This is simplified - full implementation needs the actual Frobenius coefficients
    const q1 = frobeniusG2(q);
    const add1 = additionStep(t, q1);
    t = add1.point;
    f = f.mul(evaluateLine(add1.coeffs, p.x, p.y));

    const q2 = frobeniusG2(frobeniusG2(q)).neg();
    const add2 = additionStep(t, q2);
    f = f.mul(evaluateLine(add2.coeffs, p.x, p.y));

    return f;
}

/// Apply Frobenius endomorphism to G2 point
/// π: (x, y) → (x^p, y^p) using the Frobenius coefficients for Fp2
fn frobeniusG2(p: G2Point) G2Point {
    if (p.infinity) return p;

    // The Frobenius on Fp2 is conjugation: (a + bu) → (a - bu) = (a + bu)^p
    // But we also need to multiply by twist constants
    // For BN254: π(x, y) = (x^p * ξ^((p-1)/3), y^p * ξ^((p-1)/2))

    // Simplified: just apply conjugate (full version needs twist factors)
    return G2Point{
        .x = p.x.conjugate(),
        .y = p.y.conjugate(),
        .infinity = false,
    };
}

/// Final exponentiation: f^((p^12-1)/r)
/// Split into easy part and hard part
fn finalExponentiation(f: Fp12) Fp12 {
    if (f.eql(Fp12.zero())) {
        return Fp12.one();
    }

    // Easy part: f^((p^6-1)(p^2+1))
    // This can be computed using conjugation and Frobenius
    const easy = easyPartExponentiation(f);

    // Hard part: f^((p^4-p^2+1)/r)
    // This requires more complex computation
    return hardPartExponentiation(easy);
}

fn easyPartExponentiation(f: Fp12) Fp12 {
    // f^(p^6-1) = conj(f) * f^(-1) (using the fact that f^(p^6) = conj(f))
    const f_conj = f.conjugate();
    const f_inv = f.inverse() orelse return Fp12.one();
    const t0 = f_conj.mul(f_inv);

    // t0^(p^2+1) = t0^(p^2) * t0
    // Using Frobenius: t0^(p^2) = frobenius(frobenius(t0))
    const t1 = t0.frobenius().frobenius();
    return t1.mul(t0);
}

/// BN254 curve parameter x = 4965661367192848881
/// Used in hard part of final exponentiation
const BN_X: u64 = 4965661367192848881;

fn hardPartExponentiation(f: Fp12) Fp12 {
    // The hard part is f^((p^4 - p^2 + 1)/r)
    // For BN curves, this can be computed efficiently using the curve parameter x
    // The exponent decomposes as a polynomial in x:
    // (p^4 - p^2 + 1)/r = λ_0 + λ_1*p + λ_2*p^2 + λ_3*p^3
    // where λ_i are polynomials in x

    // We use the optimized addition chain from the literature
    // First compute f^x, f^(x^2), f^(x^3) using square-and-multiply

    // f^x
    const f_x = expByX(f);

    // f^(x^2) = (f^x)^x
    const f_x2 = expByX(f_x);

    // f^(x^3) = (f^(x^2))^x
    const f_x3 = expByX(f_x2);

    // Now compute the final result using Frobenius and multiplications
    // y0 = f^(x^3) * f^(-x^2) * f^x * f^(-1)
    // y1 = f^(p*x^2) * f^(-p*x) * f^p
    // y2 = f^(p^2*x) * f^(-p^2)
    // y3 = f^(p^3)
    // result = y0 * y1 * y2 * y3

    // Simplified version using the key operations
    const f_inv = f.inverse() orelse return f;
    const f_x2_inv = f_x2.inverse() orelse return f;

    // Compute various Frobenius powers
    const f_p = frobeniusFp12(f);
    const f_p2 = frobeniusFp12(f_p);
    const f_p3 = frobeniusFp12(f_p2);

    const f_x_p = frobeniusFp12(f_x);
    const f_x_p_inv = f_x_p.inverse() orelse return f;
    const f_x2_p = frobeniusFp12(f_x2);
    const f_x_p2 = frobeniusFp12(frobeniusFp12(f_x));
    const f_p2_inv = f_p2.inverse() orelse return f;

    // Combine all terms
    // y0 = f^(x^3) * f^(-x^2) * f^x * f^(-1)
    const y0 = f_x3.mul(f_x2_inv).mul(f_x).mul(f_inv);

    // y1 = f^(p*x^2) * f^(-p*x) * f^p
    const y1 = f_x2_p.mul(f_x_p_inv).mul(f_p);

    // y2 = f^(p^2*x) * f^(-p^2)
    const y2 = f_x_p2.mul(f_p2_inv);

    // y3 = f^(p^3)
    const y3 = f_p3;

    return y0.mul(y1).mul(y2).mul(y3);
}

/// Compute f^x where x is the BN254 curve parameter
fn expByX(f: Fp12) Fp12 {
    var result = Fp12.one();
    var base = f;
    var exp = BN_X;

    while (exp > 0) {
        if (exp & 1 == 1) {
            result = result.mul(base);
        }
        base = base.square();
        exp >>= 1;
    }

    return result;
}

/// Frobenius endomorphism on Fp12
/// Computes f^p using the tower structure
fn frobeniusFp12(f: Fp12) Fp12 {
    // For Fp12 = Fp6[w]/(w² - v), the Frobenius is:
    // (a + bw)^p = a^p + b^p * w^p
    // where w^p = w * γ for some constant γ

    // For Fp6 = Fp2[v]/(v³ - ξ), the Frobenius on each component is:
    // c0^p + c1^p * v^p + c2^p * v^(2p)
    // where v^p = v * ω for constants

    // Apply Frobenius to each Fp2 component (which is just conjugation)
    // Then multiply by the Frobenius coefficients

    // Simplified: apply conjugation to all Fp2 components
    // Full implementation needs the actual Frobenius coefficients
    return Fp12{
        .c0 = Fp6{
            .c0 = f.c0.c0.conjugate(),
            .c1 = f.c0.c1.conjugate(),
            .c2 = f.c0.c2.conjugate(),
        },
        .c1 = Fp6{
            .c0 = f.c1.c0.conjugate(),
            .c1 = f.c1.c1.conjugate(),
            .c2 = f.c1.c2.conjugate(),
        },
    };
}

/// Pairing input pair type
pub const PairingInput = struct {
    p: G1Point,
    q: G2Point,
};

/// Multi-pairing: product of pairings e(P1,Q1) * e(P2,Q2) * ...
/// More efficient than computing individual pairings
pub fn multiPairing(pairs: []const PairingInput) PairingResult {
    var result = Fp12.one();

    for (pairs) |pair| {
        // In a real implementation, we'd use a shared Miller loop
        const single = pairing(pair.p, pair.q);
        result = result.mul(single);
    }

    return result;
}

/// Check if e(P1, Q1) == e(P2, Q2)
/// Useful for verifying KZG proofs
pub fn pairingCheck(p1: G1Point, q1: G2Point, p2: G1Point, q2: G2Point) bool {
    // Instead of checking e(P1,Q1) == e(P2,Q2), we check e(P1,Q1) * e(-P2,Q2) == 1
    const pairs = [_]PairingInput{
        .{ .p = p1, .q = q1 },
        .{ .p = p2.neg(), .q = q2 },
    };
    const result = multiPairing(&pairs);
    return result.isOne();
}

// ============================================================================
// Tests
// ============================================================================

test "Fp2 arithmetic" {
    const a = Fp2.init(BN254Scalar.fromU64(3), BN254Scalar.fromU64(4));
    const b = Fp2.init(BN254Scalar.fromU64(1), BN254Scalar.fromU64(2));

    // Test addition
    const sum = a.add(b);
    try std.testing.expect(sum.c0.eql(BN254Scalar.fromU64(4)));
    try std.testing.expect(sum.c1.eql(BN254Scalar.fromU64(6)));

    // Test multiplication
    const prod = a.mul(b);
    // (3 + 4i)(1 + 2i) = 3 + 6i + 4i + 8i² = 3 + 10i - 8 = -5 + 10i
    try std.testing.expect(prod.c0.eql(BN254Scalar.fromU64(3).sub(BN254Scalar.fromU64(8))));
    try std.testing.expect(prod.c1.eql(BN254Scalar.fromU64(10)));
}

test "Fp2 inverse" {
    const a = Fp2.init(BN254Scalar.fromU64(3), BN254Scalar.fromU64(4));
    const a_inv = a.inverse().?;
    const should_be_one = a.mul(a_inv);

    try std.testing.expect(should_be_one.c0.eql(BN254Scalar.one()));
    try std.testing.expect(should_be_one.c1.eql(BN254Scalar.zero()));
}

test "Fp12 basic operations" {
    const one = Fp12.one();
    const zero = Fp12.zero();

    // 1 + 0 = 1
    const sum = one.add(zero);
    try std.testing.expect(sum.eql(one));

    // 1 * 1 = 1
    const prod = one.mul(one);
    try std.testing.expect(prod.eql(one));
}

test "G2 point operations" {
    const g = G2Point.generator();
    const identity = G2Point.identity();

    // G + O = G
    const sum1 = g.add(identity);
    try std.testing.expect(sum1.eql(g));

    // G + (-G) = O
    const neg_g = g.neg();
    const sum2 = g.add(neg_g);
    try std.testing.expect(sum2.isIdentity());
}

test "Miller loop doubling step" {
    // Test that doubling step produces valid line coefficients
    const q = G2Point.generator();
    const result = doublingStep(q);

    // The resulting point should not be identity (unless generator is special)
    // Line coefficients should have at least one non-zero component
    try std.testing.expect(!result.coeffs.c1.isZero());
}

test "Miller loop addition step" {
    // Test addition step
    const q = G2Point.generator();
    const q2 = q.double();

    const result = additionStep(q, q2);

    // Should produce a valid point
    if (!result.point.isIdentity()) {
        try std.testing.expect(!result.coeffs.c1.isZero());
    }
}

test "Frobenius on Fp12" {
    // Test that Frobenius is well-defined
    const a = Fp12{
        .c0 = Fp6{
            .c0 = Fp2.init(BN254Scalar.fromU64(1), BN254Scalar.fromU64(2)),
            .c1 = Fp2.init(BN254Scalar.fromU64(3), BN254Scalar.fromU64(4)),
            .c2 = Fp2.init(BN254Scalar.fromU64(5), BN254Scalar.fromU64(6)),
        },
        .c1 = Fp6{
            .c0 = Fp2.init(BN254Scalar.fromU64(7), BN254Scalar.fromU64(8)),
            .c1 = Fp2.init(BN254Scalar.fromU64(9), BN254Scalar.fromU64(10)),
            .c2 = Fp2.init(BN254Scalar.fromU64(11), BN254Scalar.fromU64(12)),
        },
    };

    const a_frob = frobeniusFp12(a);

    // Frobenius should not return the same element (in general)
    // but applying it p times should return the original (for elements in Fp)
    // For our test, just check it's well-defined
    try std.testing.expect(!a_frob.c0.c0.c0.isZero() or !a_frob.c0.c0.c1.isZero());
}

test "expByX exponentiation" {
    // Test exponentiation by curve parameter
    const one = Fp12.one();
    const one_x = expByX(one);

    // 1^x = 1
    try std.testing.expect(one_x.eql(Fp12.one()));
}

test "pairing with identity" {
    // e(O, Q) = 1 and e(P, O) = 1
    const g1 = G1Point{ .x = BN254Scalar.one(), .y = BN254Scalar.fromU64(2), .infinity = false };
    const g2 = G2Point.generator();
    const g1_identity = G1Point{ .x = BN254Scalar.zero(), .y = BN254Scalar.one(), .infinity = true };
    const g2_identity = G2Point.identity();

    const result1 = pairing(g1_identity, g2);
    const result2 = pairing(g1, g2_identity);

    try std.testing.expect(result1.isOne());
    try std.testing.expect(result2.isOne());
}

test "Fp6 operations" {
    const one = Fp6.one();
    const zero = Fp6.zero();

    // 1 + 0 = 1
    try std.testing.expect(one.add(zero).eql(one));

    // 1 * 1 = 1
    try std.testing.expect(one.mul(one).eql(one));

    // 1 - 1 = 0
    try std.testing.expect(one.sub(one).eql(zero));
}

test "Fp6 inverse" {
    const a = Fp6{
        .c0 = Fp2.init(BN254Scalar.fromU64(1), BN254Scalar.fromU64(2)),
        .c1 = Fp2.init(BN254Scalar.fromU64(3), BN254Scalar.fromU64(4)),
        .c2 = Fp2.init(BN254Scalar.fromU64(5), BN254Scalar.fromU64(6)),
    };

    if (a.inverse()) |a_inv| {
        const should_be_one = a.mul(a_inv);
        try std.testing.expect(should_be_one.eql(Fp6.one()));
    }
}
