//! BN254 G2 Curve Points and Prepared Structures
//!
//! This module implements G2 point types for the BN254 twisted curve over Fp2:
//! - G2Point (affine representation)
//! - G2Projective (Jacobian projective for scalar multiplication)
//! - G2HomProjective (homogeneous projective for Miller loop)
//! - G2Prepared (precomputed line coefficients for fast Miller loop)
//! - G2PreparedAffine (affine line coefficients for fastest Miller loop)
//!
//! The G2 twist curve is y^2 = x^3 + B where B = 3/xi, xi = 9 + u.

const std = @import("std");

const field_mod = @import("mod.zig");
const BN254Scalar = field_mod.BN254Scalar;
const Fp = field_mod.BN254BaseField;

const extensions = @import("extensions.zig");
const Fp2 = extensions.Fp2;
const Fp6 = extensions.Fp6;
const Fp12 = extensions.Fp12;
const fp2ScalarMul = extensions.fp2ScalarMul;
const fp6MulBy01 = extensions.fp6MulBy01;
const fp6MulByV = extensions.fp6MulByV;
const gamma12 = extensions.gamma12;
const gamma13 = extensions.gamma13;

// ============================================================================
// G2 Twist Curve Helpers
// ============================================================================

/// COEFF_B for BN254 G2 twist curve: y^2 = x^3 + B where B = 3/(9+u)
/// This is stored as: (b0, b1) where B = b0 + b1*u
/// From arkworks: B = (19485874751759354771024239261021720505790618469301721065564631296452457478373,
///                     266929791119991161246907387137283842545076965332900288569378510910307636690)
pub fn twistB() Fp2 {
    // These are raw standard form values, convert to Montgomery form
    const b0_limbs: [4]u64 = .{ 0x3267e6dc24a138e5, 0xb5b4c5e559dbefa3, 0x81be18991be06ac3, 0x2b149d40ceb8aaae };
    const b1_limbs: [4]u64 = .{ 0xe4a2bd0685c315d2, 0xa74fa084e52d1852, 0xcd2cafadeed8fdf4, 0x009713b03af0fed4 };
    const b0_raw = Fp{ .limbs = b0_limbs };
    const b1_raw = Fp{ .limbs = b1_limbs };
    // Convert from standard form to Montgomery form
    return Fp2.init(b0_raw.toMontgomery(), b1_raw.toMontgomery());
}

/// Multiply Fp2 element by twist curve coefficient B
pub fn mulByTwistB(a: Fp2) Fp2 {
    return a.mul(twistB());
}

/// TWIST_MUL_BY_Q_X = (u+9)^((p-1)/3) for Frobenius on G2
pub fn twistMulByQX() Fp2 {
    // From arkworks bn254:
    // c0 = 21575463638280843010398324269430826099269044274347216827212613867836435027261
    // c1 = 10307601595873709700152284273816112264069230130616436755625194854815875713954
    const c0_limbs: [4]u64 = .{ 0x99e39557176f553d, 0xb78cc310c2c3330c, 0x4c0bec3cf559b143, 0x2fb347984f7911f7 };
    const c1_limbs: [4]u64 = .{ 0x1665d51c640fcba2, 0x32ae2a1d0b7c9dce, 0x4ba4cc8bd75a0794, 0x16c9e55061ebae20 };
    const c0 = (Fp{ .limbs = c0_limbs }).toMontgomery();
    const c1 = (Fp{ .limbs = c1_limbs }).toMontgomery();
    return Fp2.init(c0, c1);
}

/// TWIST_MUL_BY_Q_Y = (u+9)^((p-1)/2) for Frobenius on G2
pub fn twistMulByQY() Fp2 {
    // From arkworks bn254:
    // c0 = 2821565182194536844548159561693502659359617185244120367078079554186484126554
    // c1 = 3505843767911556378687030309984248845540243509899259641013678093033130930403
    const c0_limbs: [4]u64 = .{ 0xdc54014671a0135a, 0xdbaae0eda9c95998, 0xdc5ec698b6e2f9b9, 0x063cf305489af5dc };
    const c1_limbs: [4]u64 = .{ 0x82d37f632623b0e3, 0x21807dc98fa25bd2, 0x0704b5a7ec796f2b, 0x07c03cbcac41049a };
    const c0 = (Fp{ .limbs = c0_limbs }).toMontgomery();
    const c1 = (Fp{ .limbs = c1_limbs }).toMontgomery();
    return Fp2.init(c0, c1);
}

/// Frobenius endomorphism on G2 (multiply by char)
/// pi: (x, y) -> (x^p * TWIST_MUL_BY_Q_X, y^p * TWIST_MUL_BY_Q_Y)
pub fn mulByChar(p_pt: G2Point) G2Point {
    if (p_pt.infinity) return p_pt;

    // x^p = conjugate(x) for Fp2, then multiply by coefficient
    var x_new = p_pt.x.conjugate();
    x_new = x_new.mul(twistMulByQX());

    // y^p = conjugate(y) for Fp2, then multiply by coefficient
    var y_new = p_pt.y.conjugate();
    y_new = y_new.mul(twistMulByQY());

    return G2Point{ .x = x_new, .y = y_new, .infinity = false };
}

/// Apply Frobenius endomorphism to G2 point
/// pi: (x, y) -> (x^p * gamma_{1,2}, y^p * gamma_{1,3})
/// where x^p = conjugate(x), y^p = conjugate(y) in Fp2
/// and gamma_{1,2} = xi^{(p-1)/3}, gamma_{1,3} = xi^{(p-1)/2}
pub fn frobeniusG2(p: G2Point) G2Point {
    if (p.infinity) return p;

    // The Frobenius on Fp2 is conjugation: (a + bu) -> (a - bu) = (a + bu)^p
    // Then we multiply by the twist factors (Frobenius coefficients)
    // For BN254: pi(x, y) = (conjugate(x) * gamma_{1,2}, conjugate(y) * gamma_{1,3})

    const x_frob = p.x.conjugate().mul(gamma12());
    const y_frob = p.y.conjugate().mul(gamma13());

    return G2Point{
        .x = x_frob,
        .y = y_frob,
        .infinity = false,
    };
}

// ============================================================================
// Pseudobinary representation of the loop length 6*X+2
// ============================================================================

/// Pseudobinary representation of the loop length 6*X+2 of the optimal ate pairing over BN254.
/// From arkworks-rs/curves bn254/src/curves/mod.rs (ATE_LOOP_COUNT).
/// This is the NAF representation of 6*x+2 where x = 4965661367192848881.
/// Array has 65 elements, processed from LSB (index 0) to MSB (index 64).
pub const ATE_LOOP_COUNT: [65]i2 = .{
    0, 0, 0, 1, 0, 1, 0, -1, 0, 0, -1, 0, 0, 0, 1, 0,
    0, -1, 0, -1, 0, 0, 0, 1, 0, -1, 0, 0, 0, 0, -1, 0,
    0, 1, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, -1, 0, 0, -1,
    0, 1, 0, -1, 0, 0, 0, -1, 0, -1, 0, 0, 0, 1, 0, 1,
    1,
};

/// Number of EllCoeff entries in a prepared G2 point.
/// 64 doubling steps + up to 21 addition steps + 2 final Frobenius additions = 87.
pub const PREPARED_COEFFS_LEN: usize = 87;

/// X_IS_NEGATIVE flag for BN254 - the curve parameter x is positive
pub const X_IS_NEGATIVE: bool = false;

// ============================================================================
// Line Coefficient Types
// ============================================================================

/// Line coefficients for pairing computation (arkworks format)
/// For D-type twist: (c0, c1, c2) evaluated at P gives sparse Fp12 element
pub const EllCoeff = struct {
    c0: Fp2,
    c1: Fp2,
    c2: Fp2,
};

/// Line coefficients R0, R1 matching gnark-crypto's affine representation
/// R0 = lambda (the slope)
/// R1 = lambda*x_Q - y_Q (used for efficient evaluation)
pub const LineCoeffs = struct {
    r0: Fp2, // lambda
    r1: Fp2, // lambda*x_Q - y_Q
};

/// Sparse line evaluation result matching gnark-crypto's (1, 0, 0, c3, c4, 0) format
/// where positions are: 0=1, 1=v, 2=v^2, 3=w, 4=vw, 5=v^2w
/// This represents 1 + c3*w + c4*vw in Fp12
pub const SparseLineEval = struct {
    c3: Fp2, // Coefficient of w (C1.c0)
    c4: Fp2, // Coefficient of vw (C1.c1)
};

/// Evaluate line function at point P = (x_P, y_P)
/// Following gnark-crypto's affine approach:
/// - Precompute xNegOverY = -x_P/y_P and yInv = 1/y_P
/// - c3 = R0 * xNegOverY = lambda * (-x_P/y_P)
/// - c4 = R1 * yInv = (lambda*x_Q - y_Q) * (1/y_P)
pub fn evaluateLineSparse(coeffs: LineCoeffs, x_neg_over_y: Fp, y_inv: Fp) SparseLineEval {
    // c3 = R0 * xNegOverY (Fp2 * Fp -> Fp2)
    const c3 = fp2ScalarMul(coeffs.r0, x_neg_over_y);

    // c4 = R1 * yInv (Fp2 * Fp -> Fp2)
    const c4 = fp2ScalarMul(coeffs.r1, y_inv);

    return SparseLineEval{
        .c3 = c3,
        .c4 = c4,
    };
}

// ============================================================================
// Batch Inversion
// ============================================================================

/// Batch inversion of Fp2 elements using Montgomery's trick.
/// Inverts elements in-place. Zero elements are skipped.
/// `scratch` must have the same length as `elements`.
/// Cost: 2(n-1) Fp2.mul + 1 Fp2.inverse (vs n individual Fp2.inverse).
pub fn batchInverseFp2(elements: []Fp2, scratch: []Fp2) void {
    const n_elems = elements.len;
    if (n_elems == 0) return;

    // Forward pass: prefix products
    var acc = Fp2.one();
    for (0..n_elems) |i| {
        scratch[i] = acc;
        if (!elements[i].isZero()) {
            acc = acc.mul(elements[i]);
        }
    }

    // Single inversion
    var inv = acc.inverse() orelse unreachable;

    // Backward pass: extract individual inverses
    var i: usize = n_elems;
    while (i > 0) {
        i -= 1;
        if (elements[i].isZero()) continue;
        const old = elements[i];
        elements[i] = scratch[i].mul(inv);
        inv = inv.mul(old);
    }
}

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

        const x0 = Fp.fromBytes(&x0_bytes);
        const x1 = Fp.fromBytes(&x1_bytes);
        const y0 = Fp.fromBytes(&y0_bytes);
        const y1 = Fp.fromBytes(&y1_bytes);

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

        // Point doubling formula: lambda = (3x^2)/(2y)
        const x_sq = self.x.square();
        const three_x_sq = x_sq.add(x_sq).add(x_sq);
        const two_y = self.y.add(self.y);
        const slope = three_x_sq.mul(two_y.inverse() orelse return G2Point.identity());

        const x3 = slope.square().sub(self.x).sub(self.x);
        const y3 = slope.mul(self.x.sub(x3)).sub(self.y);

        return .{ .x = x3, .y = y3, .infinity = false };
    }

    /// Scalar multiplication using double-and-add in Jacobian projective coordinates.
    /// Only one Fp2 inversion at the final toAffine().
    pub fn scalarMul(self: G2Point, scalar: BN254Scalar) G2Point {
        if (self.isIdentity()) return G2Point.identity();
        if (scalar.isZero()) return G2Point.identity();

        const normal_scalar = scalar.fromMontgomery();
        return self.scalarMulWithLimbs(normal_scalar.limbs);
    }

    /// Scalar multiplication returning Jacobian projective result (no final inversion).
    /// Use when accumulating multiple results in projective before a single toAffine().
    pub fn scalarMulWithLimbsProjective(self: G2Point, normal_limbs: [4]u64) G2Projective {
        if (self.isIdentity()) return G2Projective.identity();

        var result = G2Projective.identity();
        var started = false;

        var limb_idx: usize = 4;
        while (limb_idx > 0) {
            limb_idx -= 1;
            const limb = normal_limbs[limb_idx];

            var bit_idx: u7 = 64;
            while (bit_idx > 0) {
                bit_idx -= 1;
                if (started) {
                    result = result.double();
                }

                const bit = (limb >> @as(u6, @intCast(bit_idx))) & 1;
                if (bit == 1) {
                    if (!started) {
                        result = G2Projective.fromAffine(self);
                        started = true;
                    } else {
                        result = result.addAffine(self);
                    }
                }
            }
        }

        return result;
    }

    /// Scalar multiplication using pre-converted (non-Montgomery) limbs
    pub fn scalarMulWithLimbs(self: G2Point, normal_limbs: [4]u64) G2Point {
        return self.scalarMulWithLimbsProjective(normal_limbs).toAffine();
    }

    /// Scalar multiplication with a u64 scalar (convenience method)
    pub fn scalarMulU64(self: G2Point, scalar: u64) G2Point {
        return self.scalarMul(BN254Scalar.fromU64(scalar));
    }
};

// ============================================================================
// G2 Jacobian Projective (for scalar multiplication)
// ============================================================================

/// G2 point in Jacobian projective coordinates (X, Y, Z) where affine = (X/Z^2, Y/Z^3)
/// Eliminates intermediate Fp2 inversions during scalar multiplication.
pub const G2Projective = struct {
    x: Fp2,
    y: Fp2,
    z: Fp2,

    pub fn identity() G2Projective {
        return .{
            .x = Fp2.one(),
            .y = Fp2.one(),
            .z = Fp2.zero(),
        };
    }

    pub fn fromAffine(p: G2Point) G2Projective {
        if (p.infinity) return identity();
        return .{ .x = p.x, .y = p.y, .z = Fp2.one() };
    }

    pub fn isIdentity(self: G2Projective) bool {
        return self.z.isZero();
    }

    pub fn toAffine(self: G2Projective) G2Point {
        if (self.z.isZero()) return G2Point.identity();

        const z_inv = self.z.inverse() orelse return G2Point.identity();
        const z_inv_sq = z_inv.square();
        const z_inv_cube = z_inv_sq.mul(z_inv);

        return .{
            .x = self.x.mul(z_inv_sq),
            .y = self.y.mul(z_inv_cube),
            .infinity = false,
        };
    }

    /// Jacobian doubling for y^2 = x^3 + b/xi (a = 0, BN254 twist curve)
    /// Using dbl-2009-l formulas from https://hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html
    pub fn double(self: G2Projective) G2Projective {
        if (self.z.isZero()) return self;

        const A = self.x.square();
        const B = self.y.square();
        const C = B.square();
        const xpb = self.x.add(B);
        const half_D = xpb.square().sub(A).sub(C);
        const D = half_D.add(half_D);
        const E = A.add(A).add(A);
        const FF = E.square();
        const two_D = D.add(D);
        const X3 = FF.sub(two_D);
        const two_C = C.add(C);
        const four_C = two_C.add(two_C);
        const eight_C = four_C.add(four_C);
        const Y3 = E.mul(D.sub(X3)).sub(eight_C);
        const yz = self.y.mul(self.z);
        const Z3 = yz.add(yz);

        return .{ .x = X3, .y = Y3, .z = Z3 };
    }

    /// Mixed addition: Jacobian + affine (saves one Fp2 mul vs full Jacobian add)
    pub fn addAffine(self: G2Projective, other: G2Point) G2Projective {
        if (other.infinity) return self;
        if (self.z.isZero()) return fromAffine(other);

        const z1z1 = self.z.square();
        const U2 = other.x.mul(z1z1);
        const S2 = other.y.mul(self.z).mul(z1z1);
        const H = U2.sub(self.x);
        const HH = H.square();
        const I = HH.add(HH).add(HH).add(HH);
        const J = H.mul(I);
        const r = S2.sub(self.y).add(S2.sub(self.y));
        const V = self.x.mul(I);
        const X3 = r.square().sub(J).sub(V).sub(V);
        const Y3 = r.mul(V.sub(X3)).sub(self.y.mul(J)).sub(self.y.mul(J));
        const Z3 = self.z.add(H).square().sub(z1z1).sub(HH);

        if (H.isZero()) {
            if (r.isZero()) {
                return self.double();
            } else {
                return identity();
            }
        }

        return .{ .x = X3, .y = Y3, .z = Z3 };
    }

    /// Full Jacobian addition
    pub fn add(self: G2Projective, other: G2Projective) G2Projective {
        if (self.z.isZero()) return other;
        if (other.z.isZero()) return self;

        const z1z1 = self.z.square();
        const z2z2 = other.z.square();
        const U1 = self.x.mul(z2z2);
        const U2 = other.x.mul(z1z1);
        const S1 = self.y.mul(other.z).mul(z2z2);
        const S2 = other.y.mul(self.z).mul(z1z1);
        const H = U2.sub(U1);
        const r = S2.sub(S1);

        if (H.isZero()) {
            if (r.isZero()) {
                return self.double();
            } else {
                return identity();
            }
        }

        const HH = H.square();
        const I = HH.add(HH).add(HH).add(HH);
        const J = H.mul(I);
        const rr = r.add(r);
        const V = U1.mul(I);
        const X3 = rr.square().sub(J).sub(V).sub(V);
        const Y3 = rr.mul(V.sub(X3)).sub(S1.mul(J).add(S1.mul(J)));
        const Z3 = self.z.add(other.z).square().sub(z1z1).sub(z2z2).mul(H);

        return .{ .x = X3, .y = Y3, .z = Z3 };
    }

    /// Batch normalize G2 projective points to affine using Montgomery's trick.
    /// Single Fp2 inversion + ~6n Fp2 multiplications instead of n inversions.
    pub fn batchNormalize(points: []const G2Projective, out: []G2Point) void {
        std.debug.assert(out.len >= points.len);
        const n = points.len;
        if (n == 0) return;

        var heap_products: ?[]Fp2 = null;
        defer if (heap_products) |h| std.heap.page_allocator.free(h);

        var stack_products: [1024]Fp2 = undefined;
        const products: []Fp2 = if (n <= 1024)
            stack_products[0..n]
        else blk: {
            heap_products = std.heap.page_allocator.alloc(Fp2, n) catch {
                for (points, 0..) |p, i| out[i] = p.toAffine();
                return;
            };
            break :blk heap_products.?;
        };

        // Forward pass: accumulate Z products
        var acc = Fp2.one();
        for (points, 0..) |p, i| {
            if (p.isIdentity()) {
                products[i] = acc;
            } else {
                products[i] = acc;
                acc = acc.mul(p.z);
            }
        }

        // Invert the accumulated product
        var inv = acc.inverse() orelse Fp2.one();

        // Backward pass: extract individual Z inverses
        var i: usize = n;
        while (i > 0) {
            i -= 1;
            if (points[i].isIdentity()) {
                out[i] = G2Point.identity();
            } else {
                const z_inv = products[i].mul(inv);
                inv = inv.mul(points[i].z);
                const z_inv_sq = z_inv.square();
                const z_inv_cube = z_inv_sq.mul(z_inv);
                out[i] = .{
                    .x = points[i].x.mul(z_inv_sq),
                    .y = points[i].y.mul(z_inv_cube),
                    .infinity = false,
                };
            }
        }
    }
};

// ============================================================================
// G2 Homogeneous Projective (for Miller loop)
// ============================================================================

/// G2 point in homogeneous projective coordinates (x, y, z) where x/z, y/z are affine
/// Used in Miller loop to avoid expensive field inversions
pub const G2HomProjective = struct {
    x: Fp2,
    y: Fp2,
    z: Fp2,

    /// Convert from affine to projective (z = 1)
    pub fn fromAffine(p: G2Point) G2HomProjective {
        if (p.infinity) {
            return .{ .x = Fp2.zero(), .y = Fp2.one(), .z = Fp2.zero() };
        }
        return .{ .x = p.x, .y = p.y, .z = Fp2.one() };
    }

    /// Doubling step in projective coordinates
    /// Returns new point R = 2T and line coefficients for D-type twist
    pub fn double_in_place(self: *G2HomProjective, two_inv: Fp) EllCoeff {
        // Formula from arkworks bn254 g2.rs
        // a = x * y / 2
        var a = self.x.mul(self.y);
        a = fp2ScalarMul(a, two_inv);

        const b = self.y.square(); // b = y^2
        const c = self.z.square(); // c = z^2

        // e = COEFF_B * (c + c + c) = 3*B*c where B = 3/(9+u)
        // For BN254 D-type twist: COEFF_B = 3/(xi) where xi = 9+u
        // We compute e = 3 * COEFF_B * c = 3 * 3/(9+u) * c = 9*c/(9+u)
        // Actually: e = COEFF_B * (c.double() + c) = COEFF_B * 3c
        const three_c = c.add(c).add(c);
        const e = mulByTwistB(three_c);

        // f = e + e + e = 3e
        const f = e.add(e).add(e);

        // g = (b + f) / 2
        var g = b.add(f);
        g = fp2ScalarMul(g, two_inv);

        // h = (y + z)^2 - (b + c)
        const h = self.y.add(self.z).square().sub(b.add(c));

        // i = e - b
        const i = e.sub(b);

        // j = x^2
        const j = self.x.square();

        // e_square = e^2
        const e_square = e.square();

        // New point:
        // x' = a * (b - f)
        self.x = a.mul(b.sub(f));
        // y' = g^2 - 3*e^2
        self.y = g.square().sub(e_square.add(e_square).add(e_square));
        // z' = b * h
        self.z = b.mul(h);

        // Line coefficients for D-type twist: (-h, 3j, i)
        return EllCoeff{
            .c0 = h.neg(),
            .c1 = j.add(j).add(j), // 3j
            .c2 = i,
        };
    }

    /// Addition step in projective coordinates
    /// Returns new point R = T + Q and line coefficients for D-type twist
    pub fn add_in_place(self: *G2HomProjective, q: G2Point) EllCoeff {
        // Formula from arkworks bn254 g2.rs
        // theta = y - q.y * z
        const theta = self.y.sub(q.y.mul(self.z));
        // lambda = x - q.x * z
        const lambda = self.x.sub(q.x.mul(self.z));

        const c = theta.square();
        const d = lambda.square();
        const e = lambda.mul(d);
        const f = self.z.mul(c);
        const g = self.x.mul(d);
        const h = e.add(f).sub(g.add(g));

        // New point:
        // x' = lambda * h
        self.x = lambda.mul(h);
        // y' = theta * (g - h) - e * y
        self.y = theta.mul(g.sub(h)).sub(e.mul(self.y));
        // z' = z * e
        self.z = self.z.mul(e);

        // j = theta * q.x - lambda * q.y
        const jay = theta.mul(q.x).sub(lambda.mul(q.y));

        // Line coefficients for D-type twist: (lambda, -theta, j)
        return EllCoeff{
            .c0 = lambda,
            .c1 = theta.neg(),
            .c2 = jay,
        };
    }
};

// ============================================================================
// Precomputed G2 Structures
// ============================================================================

/// Precomputed G2 point for fast Miller loop evaluation.
/// Stores all line function coefficients so the Miller loop
/// only needs Fp12 arithmetic (no G2 projective operations).
pub const G2Prepared = struct {
    coeffs: [PREPARED_COEFFS_LEN]EllCoeff,
    infinity: bool,

    pub fn fromG2Point(q: G2Point) G2Prepared {
        if (q.infinity) {
            return G2Prepared{
                .coeffs = [_]EllCoeff{.{ .c0 = Fp2.zero(), .c1 = Fp2.zero(), .c2 = Fp2.zero() }} ** PREPARED_COEFFS_LEN,
                .infinity = true,
            };
        }

        const two_inv = Fp.fromU64(2).inverse().?;
        var r = G2HomProjective.fromAffine(q);
        const neg_q = q.neg();

        var result: G2Prepared = undefined;
        result.infinity = false;
        var coeff_idx: usize = 0;

        // Main loop
        var idx: usize = ATE_LOOP_COUNT.len - 1;
        while (idx >= 1) : (idx -= 1) {
            // Doubling step
            result.coeffs[coeff_idx] = r.double_in_place(two_inv);
            coeff_idx += 1;

            // Addition step if bit is non-zero
            const bit = ATE_LOOP_COUNT[idx - 1];
            if (bit == 1) {
                result.coeffs[coeff_idx] = r.add_in_place(q);
                coeff_idx += 1;
            } else if (bit == -1) {
                result.coeffs[coeff_idx] = r.add_in_place(neg_q);
                coeff_idx += 1;
            }
        }

        // Final Frobenius steps
        const q1 = mulByChar(q);
        result.coeffs[coeff_idx] = r.add_in_place(q1);
        coeff_idx += 1;

        var q2 = mulByChar(q1);
        q2.y = q2.y.neg();
        result.coeffs[coeff_idx] = r.add_in_place(q2);
        coeff_idx += 1;

        std.debug.assert(coeff_idx == PREPARED_COEFFS_LEN);
        return result;
    }
};

/// Precomputed G2 point with affine line coefficients.
/// Line evaluation at P gives (1, 0, 0, c3, c4, 0) -- c0=1 implicit.
/// This enables fp12MulBy34 (10 Fp2.mul) instead of fp12MulBy034 (13 Fp2.mul).
pub const G2PreparedAffine = struct {
    coeffs: [PREPARED_COEFFS_LEN]LineCoeffs,
    infinity: bool,

    /// Convert from projective G2Prepared to affine line coefficients.
    /// Uses batch Fp2 inversion to convert all 87 coefficients at once.
    pub fn fromG2Prepared(prep: *const G2Prepared) G2PreparedAffine {
        if (prep.infinity) {
            var result: G2PreparedAffine = undefined;
            result.infinity = true;
            for (0..PREPARED_COEFFS_LEN) |i| {
                result.coeffs[i] = .{ .r0 = Fp2.zero(), .r1 = Fp2.zero() };
            }
            return result;
        }

        // Extract c0 values and batch-invert them
        var c0_values: [PREPARED_COEFFS_LEN]Fp2 = undefined;
        var c0_scratch: [PREPARED_COEFFS_LEN]Fp2 = undefined;
        for (0..PREPARED_COEFFS_LEN) |i| {
            c0_values[i] = prep.coeffs[i].c0;
        }
        batchInverseFp2(&c0_values, &c0_scratch);

        // Convert: r0 = -c1 * inv_c0, r1 = c2 * inv_c0
        var result: G2PreparedAffine = undefined;
        result.infinity = false;
        for (0..PREPARED_COEFFS_LEN) |i| {
            const inv_c0 = c0_values[i];
            result.coeffs[i] = .{
                .r0 = prep.coeffs[i].c1.neg().mul(inv_c0),
                .r1 = prep.coeffs[i].c2.mul(inv_c0),
            };
        }
        return result;
    }

    /// Create directly from a G2 point.
    pub fn fromG2Point(q: G2Point) G2PreparedAffine {
        const prep = G2Prepared.fromG2Point(q);
        return fromG2Prepared(&prep);
    }
};

// ============================================================================
// Tests
// ============================================================================

test "G2 point operations" {
    const g = G2Point.generator();
    const identity_pt = G2Point.identity();

    // G + O = G
    const sum1 = g.add(identity_pt);
    try std.testing.expect(sum1.eql(g));

    // G + (-G) = O
    const neg_g = g.neg();
    const sum2 = g.add(neg_g);
    try std.testing.expect(sum2.isIdentity());
}

test "G2 scalar multiplication" {
    const g = G2Point.generator();

    // [0]G = O
    const zero_times_g = g.scalarMul(BN254Scalar.zero());
    try std.testing.expect(zero_times_g.isIdentity());

    // [1]G = G
    const one_times_g = g.scalarMul(BN254Scalar.one());
    try std.testing.expect(one_times_g.eql(g));

    // [2]G = G + G = double(G)
    const two_times_g = g.scalarMul(BN254Scalar.fromU64(2));
    const g_doubled = g.double();
    try std.testing.expect(two_times_g.eql(g_doubled));

    // [3]G = G + G + G = double(G) + G
    const three_times_g = g.scalarMul(BN254Scalar.fromU64(3));
    const g_tripled = g_doubled.add(g);
    try std.testing.expect(three_times_g.eql(g_tripled));

    // Convenience method [5]G
    const five_times_g = g.scalarMulU64(5);
    const expected = g.scalarMul(BN254Scalar.fromU64(5));
    try std.testing.expect(five_times_g.eql(expected));
}

test "G2 scalar mul internal consistency" {
    // Verify G2 scalar multiplication produces correct results
    const g2 = G2Point.generator();

    // [2]G2 should equal G2 + G2
    const two_g2_by_add = g2.add(g2);
    const two_g2_by_double = g2.double();
    const two_g2_by_scalar = g2.scalarMul(BN254Scalar.fromU64(2));

    // All three should be equal
    try std.testing.expect(two_g2_by_add.eql(two_g2_by_double));
    try std.testing.expect(two_g2_by_double.eql(two_g2_by_scalar));
}

test "batchInverseFp2 correctness" {
    var elements: [4]Fp2 = .{
        Fp2.init(Fp.fromU64(7), Fp.fromU64(11)),
        Fp2.init(Fp.fromU64(13), Fp.fromU64(17)),
        Fp2.init(Fp.fromU64(23), Fp.fromU64(29)),
        Fp2.init(Fp.fromU64(31), Fp.fromU64(37)),
    };
    const originals: [4]Fp2 = elements;
    var scratch: [4]Fp2 = undefined;

    batchInverseFp2(&elements, &scratch);

    for (0..4) |i| {
        const product = originals[i].mul(elements[i]);
        try std.testing.expect(product.eql(Fp2.one()));
    }
}
