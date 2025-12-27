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

    /// Generator point for G2
    /// Note: These are placeholder values - real implementation needs the actual generator
    pub fn generator() G2Point {
        // Placeholder - actual generator coordinates are much more complex
        return G2Point.fromCoords(
            Fp2.init(BN254Scalar.one(), BN254Scalar.zero()),
            Fp2.init(BN254Scalar.fromU64(2), BN254Scalar.zero()),
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

/// Miller loop for optimal ate pairing
fn millerLoop(p: G1Point, q: G2Point) Fp12 {
    _ = p;
    _ = q;
    // Simplified placeholder
    // Full implementation requires:
    // 1. Line function evaluation at each step
    // 2. Doubling and addition steps following the ate loop
    // 3. Efficient sparse multiplication in Fp12
    return Fp12.one();
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

fn hardPartExponentiation(f: Fp12) Fp12 {
    // Simplified placeholder
    // Full implementation uses the BN254 curve parameter x to compute
    // f^((p^4-p^2+1)/r) efficiently using a addition chain
    return f;
}

/// Multi-pairing: product of pairings e(P1,Q1) * e(P2,Q2) * ...
/// More efficient than computing individual pairings
pub fn multiPairing(pairs: []const struct { p: G1Point, q: G2Point }) PairingResult {
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
    const pairs = [_]struct { p: G1Point, q: G2Point }{
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
