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
//!
//! Implementation based on:
//! - https://eprint.iacr.org/2024/640.pdf
//! - gnark-crypto BN254 implementation
//! - ziskos BN254 implementation

const std = @import("std");

// Debug output control - set to true to enable verbose debug prints
const debug_verbose = false;
fn dbg(comptime fmt: []const u8, args: anytype) void {
    if (debug_verbose) std.debug.print(fmt, args);
}

const field_mod = @import("mod.zig");
const BN254Scalar = field_mod.BN254Scalar; // Scalar field Fr (for MSM scalars)
const Fp = field_mod.BN254BaseField; // Base field Fp (for pairing operations)

// ============================================================================
// Re-exports from extensions.zig (extension field types and helpers)
// ============================================================================

pub const extensions = @import("extensions.zig");
pub const Fp2 = extensions.Fp2;
pub const Fp6 = extensions.Fp6;
pub const Fp12 = extensions.Fp12;
pub const GT = extensions.GT;
pub const fp2ScalarMul = extensions.fp2ScalarMul;
pub const fp2FromLimbs = extensions.fp2FromLimbs;
pub const fpFromLimbs = extensions.fpFromLimbs;
pub const fp6MulByV = extensions.fp6MulByV;
pub const fp6MulBy01 = extensions.fp6MulBy01;
pub const fp6MulBy1 = extensions.fp6MulBy1;
pub const fp12MulBy034 = extensions.fp12MulBy034;
pub const fp12Mul034By034 = extensions.fp12Mul034By034;
pub const fp12MulBy01234 = extensions.fp12MulBy01234;
pub const fp12MulBy34 = extensions.fp12MulBy34;
pub const fp12Mul34By34 = extensions.fp12Mul34By34;

// ============================================================================
// Re-exports from g2.zig (G2 point types and helpers)
// ============================================================================

pub const g2 = @import("g2.zig");
pub const G2Point = g2.G2Point;
pub const G2Projective = g2.G2Projective;
pub const G2Prepared = g2.G2Prepared;
pub const G2PreparedAffine = g2.G2PreparedAffine;
pub const G2HomProjective = g2.G2HomProjective;
pub const EllCoeff = g2.EllCoeff;
pub const LineCoeffs = g2.LineCoeffs;
pub const SparseLineEval = g2.SparseLineEval;
pub const evaluateLineSparse = g2.evaluateLineSparse;
pub const batchInverseFp2 = g2.batchInverseFp2;
pub const mulByChar = g2.mulByChar;
pub const frobeniusG2 = g2.frobeniusG2;

const ATE_LOOP_COUNT = g2.ATE_LOOP_COUNT;
const PREPARED_COEFFS_LEN = g2.PREPARED_COEFFS_LEN;
const X_IS_NEGATIVE = g2.X_IS_NEGATIVE;

// ============================================================================
// Pairing Operations
// ============================================================================

/// Result of a pairing computation (element of GT = Fp12)
pub const PairingResult = Fp12;

/// G1 Point for pairing operations (coordinates in Fp, the base field)
pub const G1PointFp = struct {
    x: Fp,
    y: Fp,
    infinity: bool,

    pub fn identity() G1PointFp {
        return .{ .x = Fp.zero(), .y = Fp.one(), .infinity = true };
    }

    pub fn neg(self: G1PointFp) G1PointFp {
        if (self.infinity) return self;
        return .{ .x = self.x, .y = self.y.neg(), .infinity = false };
    }
};

/// G1 Point from MSM (uses scalar field, for scalar multiplication)
/// Note: For pairing operations, convert to G1PointFp
/// IMPORTANT: G1 point COORDINATES are in the BASE FIELD Fp, but scalars are in Fr
pub const G1Point = @import("../msm/mod.zig").AffinePoint(BN254Scalar);

/// G1 Point in base field (proper representation for curve points)
/// Use this for creating G1 points for pairing operations
pub const G1PointInFp = @import("../msm/mod.zig").AffinePoint(Fp);

/// Convert G1Point (scalar field coords) to G1PointFp (base field coords)
/// G1 point coordinates are conceptually raw integer values that should be
/// in Montgomery form for either field. Since G1Point uses BN254Scalar (Fr),
/// we need to:
/// 1. Convert from Fr Montgomery form to raw value
/// 2. Convert from raw value to Fp Montgomery form
fn g1ToFp(p: G1Point) G1PointFp {
    if (p.infinity) {
        return G1PointFp.identity();
    }

    // Convert x from Fr Montgomery to raw, then to Fp Montgomery
    const x_raw = p.x.fromMontgomery();
    var x_fp_tmp = Fp{ .limbs = x_raw.limbs };
    const x_fp = x_fp_tmp.toMontgomery();

    // Convert y from Fr Montgomery to raw, then to Fp Montgomery
    const y_raw = p.y.fromMontgomery();
    var y_fp_tmp = Fp{ .limbs = y_raw.limbs };
    const y_fp = y_fp_tmp.toMontgomery();

    return .{
        .x = x_fp,
        .y = y_fp,
        .infinity = false,
    };
}

/// Compute the optimal ate pairing: e(P, Q) where P ∈ G1, Q ∈ G2
///
/// The pairing consists of two parts:
/// 1. Miller loop: Compute f_{6x+2,Q}(P)
/// 2. Final exponentiation: f^((p^12-1)/r)
///
/// Uses arkworks-compatible projective Miller loop for Jolt compatibility
pub fn pairing(p: G1Point, q: G2Point) PairingResult {
    if (p.infinity or q.infinity) {
        return Fp12.one();
    }

    // Convert G1 point from Fr to Fp representation
    const p_fp = g1ToFp(p);

    // Miller loop using arkworks-style projective coordinates
    const f = millerLoopArkworks(p_fp, q);

    // Final exponentiation
    return finalExponentiation(f);
}

/// Pairing function that takes G1 point directly in base field representation
/// Use this when you have proper Fp coordinates (not Fr)
/// Uses arkworks-compatible projective Miller loop for Jolt compatibility
pub fn pairingFp(p: G1PointFp, q: G2Point) PairingResult {
    if (p.infinity or q.infinity) {
        return Fp12.one();
    }

    // Miller loop using arkworks-style projective coordinates
    const f = millerLoopArkworks(p, q);

    // Final exponentiation
    return finalExponentiation(f);
}

/// Miller loop using arkworks-style projective coordinates
/// This is the correct implementation matching arkworks exactly
pub fn millerLoopArkworks(p: G1PointFp, q: G2Point) Fp12 {
    if (p.infinity or q.infinity) {
        return Fp12.one();
    }

    // Precompute two_inv = 1/2
    const two_inv = Fp.fromU64(2).inverse() orelse return Fp12.one();

    // Initialize projective point R = Q
    var r = G2HomProjective.fromAffine(q);
    const neg_q = q.neg();

    var f = Fp12.one();

    // Main loop: iterate from MSB-1 down to 0
    // arkworks iterates from (len-1) down to 1, checking bit at (i-1)
    var idx: usize = ATE_LOOP_COUNT.len - 1;
    while (idx >= 1) : (idx -= 1) {
        // Square f unless it's the first iteration
        if (idx != ATE_LOOP_COUNT.len - 1) {
            f = f.square();
        }

        // Doubling step: R = 2R, get line coefficients
        const coeffs_dbl = r.double_in_place(two_inv);
        // Evaluate line: c0 *= y_P, c1 *= x_P for D-type twist
        const c0_eval = fp2ScalarMul(coeffs_dbl.c0, p.y);
        const c1_eval = fp2ScalarMul(coeffs_dbl.c1, p.x);
        f = fp12MulBy034(f, c0_eval, c1_eval, coeffs_dbl.c2);

        // Addition step if bit is non-zero
        const bit = ATE_LOOP_COUNT[idx - 1];
        if (bit == 1) {
            const coeffs_add = r.add_in_place(q);
            const c0_add = fp2ScalarMul(coeffs_add.c0, p.y);
            const c1_add = fp2ScalarMul(coeffs_add.c1, p.x);
            f = fp12MulBy034(f, c0_add, c1_add, coeffs_add.c2);
        } else if (bit == -1) {
            const coeffs_add = r.add_in_place(neg_q);
            const c0_add = fp2ScalarMul(coeffs_add.c0, p.y);
            const c1_add = fp2ScalarMul(coeffs_add.c1, p.x);
            f = fp12MulBy034(f, c0_add, c1_add, coeffs_add.c2);
        }
    }

    // If X is negative, conjugate the result (cyclotomic inverse)
    if (X_IS_NEGATIVE) {
        f = f.conjugate();
    }

    // Final Frobenius steps: add π(Q) and -π²(Q)
    // First: R + π(Q)
    const q1 = mulByChar(q);
    const coeffs_q1 = r.add_in_place(q1);
    const c0_q1 = fp2ScalarMul(coeffs_q1.c0, p.y);
    const c1_q1 = fp2ScalarMul(coeffs_q1.c1, p.x);
    f = fp12MulBy034(f, c0_q1, c1_q1, coeffs_q1.c2);

    // Second: R + (-π²(Q))
    var q2 = mulByChar(q1);
    q2.y = q2.y.neg(); // Negate y coordinate
    const coeffs_q2 = r.add_in_place(q2);
    const c0_q2 = fp2ScalarMul(coeffs_q2.c0, p.y);
    const c1_q2 = fp2ScalarMul(coeffs_q2.c1, p.x);
    f = fp12MulBy034(f, c0_q2, c1_q2, coeffs_q2.c2);

    return f;
}

/// Miller loop using precomputed G2 coefficients.
/// Avoids all G2 projective arithmetic — only Fp12 multiplications.
pub fn millerLoopPrepared(p: G1PointFp, q_prep: *const G2Prepared) Fp12 {
    if (p.infinity or q_prep.infinity) {
        return Fp12.one();
    }

    var f = Fp12.one();
    var coeff_idx: usize = 0;

    var idx: usize = ATE_LOOP_COUNT.len - 1;
    while (idx >= 1) : (idx -= 1) {
        if (idx != ATE_LOOP_COUNT.len - 1) {
            f = f.square();
        }

        // Doubling coefficients
        const coeffs_dbl = q_prep.coeffs[coeff_idx];
        coeff_idx += 1;
        const c0_eval = fp2ScalarMul(coeffs_dbl.c0, p.y);
        const c1_eval = fp2ScalarMul(coeffs_dbl.c1, p.x);
        f = fp12MulBy034(f, c0_eval, c1_eval, coeffs_dbl.c2);

        // Addition coefficients if bit is non-zero
        const bit = ATE_LOOP_COUNT[idx - 1];
        if (bit == 1 or bit == -1) {
            const coeffs_add = q_prep.coeffs[coeff_idx];
            coeff_idx += 1;
            const c0_add = fp2ScalarMul(coeffs_add.c0, p.y);
            const c1_add = fp2ScalarMul(coeffs_add.c1, p.x);
            f = fp12MulBy034(f, c0_add, c1_add, coeffs_add.c2);
        }
    }

    if (X_IS_NEGATIVE) {
        f = f.conjugate();
    }

    // Final Frobenius steps
    const coeffs_q1 = q_prep.coeffs[coeff_idx];
    coeff_idx += 1;
    const c0_q1 = fp2ScalarMul(coeffs_q1.c0, p.y);
    const c1_q1 = fp2ScalarMul(coeffs_q1.c1, p.x);
    f = fp12MulBy034(f, c0_q1, c1_q1, coeffs_q1.c2);

    const coeffs_q2 = q_prep.coeffs[coeff_idx];
    coeff_idx += 1;
    const c0_q2 = fp2ScalarMul(coeffs_q2.c0, p.y);
    const c1_q2 = fp2ScalarMul(coeffs_q2.c1, p.x);
    f = fp12MulBy034(f, c0_q2, c1_q2, coeffs_q2.c2);

    std.debug.assert(coeff_idx == PREPARED_COEFFS_LEN);
    return f;
}

/// Final exponentiation: f^((p^12-1)/r)
/// Using arkworks algorithm from bn/mod.rs
pub fn finalExponentiation(f: Fp12) Fp12 {
    if (f.eql(Fp12.zero())) {
        return Fp12.one();
    }

    // Easy part: f^((p^6-1)(p^2+1))
    // f1 = f^(p^6) = conjugate(f) for cyclotomic elements
    var f1 = f.conjugate();

    // f2 = f^(-1)
    const f2_opt = f.inverse();
    if (f2_opt == null) return Fp12.one();
    var f2 = f2_opt.?;

    // r = f^(p^6 - 1) = f1 * f2 = conj(f) * f^(-1)
    var r = f1.mul(f2);

    // f2 = f^(p^6 - 1) (save for later)
    f2 = r;

    // r = f^((p^6 - 1)(p^2)) = r.frobenius_map(2)
    r = r.frobenius2();

    // r = f^((p^6 - 1)(p^2 + 1))
    r = r.mul(f2);

    // Hard part using arkworks "Faster hashing to G2" algorithm
    return hardPartExponentiationArkworks(r);
}

fn easyPartExponentiation(f: Fp12) Fp12 {
    // f^(p^6-1) = conj(f) * f^(-1) (using the fact that f^(p^6) = conj(f))
    const f_conj = f.conjugate();
    const f_inv = f.inverse() orelse return Fp12.one();
    const easy1 = f_conj.mul(f_inv);

    // easy1^(p^2+1) = easy1^(p^2) * easy1
    // Using Frobenius^2 for efficiency
    const easy1_p2 = easy1.frobenius2();
    return easy1_p2.mul(easy1);
}

/// BN254 curve parameter x = 4965661367192848881
/// Used in hard part of final exponentiation
const BN_X: u64 = 4965661367192848881;

fn hardPartExponentiation(m: Fp12) Fp12 {
    // The hard part is m^((p^4 - p^2 + 1)/r)
    // Using the optimized formula from ziskos (final_exp.rs)
    //
    // Compute:
    //   y1 = m^p · m^{p²} · m^{p³}
    //   y2 = m̄ (conjugate)
    //   y3 = (m^{x²})^{p²}
    //   y4 = conj((m^x)^p)
    //   y5 = conj(m^x · (m^{x²})^p)
    //   y6 = conj(m^{x²})
    //   y7 = conj(m^{x³} · (m^{x³})^p)
    //
    // Then compute y1·y2²·y3⁶·y4¹²·y5¹⁸·y6³⁰·y7³⁶ using an optimized addition chain

    // Compute powers of m by x
    const mx = expByX(m);
    const mxx = expByX(mx);
    const mxxx = expByX(mxx);

    // Compute Frobenius powers
    const mp = m.frobenius();
    const mpp = m.frobenius2();
    const mppp = m.frobenius3();
    const mxp = mx.frobenius();
    const mxxp = mxx.frobenius();
    const mxxxp = mxxx.frobenius();
    const mxxpp = mxx.frobenius2();

    // y1 = m^p · m^{p²} · m^{p³}
    var y1 = mp.mul(mpp);
    y1 = y1.mul(mppp);

    // y2 = m̄ (conjugate)
    const y2 = m.conjugate();

    // y3 = (m^{x²})^{p²} (already computed as mxxpp)

    // y4 = conj((m^x)^p)
    const y4 = mxp.conjugate();

    // y5 = conj(m^x · (m^{x²})^p)
    var y5 = mx.mul(mxxp);
    y5 = y5.conjugate();

    // y6 = conj(m^{x²})
    const y6 = mxx.conjugate();

    // y7 = conj(m^{x³} · (m^{x³})^p)
    var y7 = mxxx.mul(mxxxp);
    y7 = y7.conjugate();

    // Compute y1·y2²·y3⁶·y4¹²·y5¹⁸·y6³⁰·y7³⁶ using the optimized addition chain from ziskos:
    //
    // T11 = y7² · y5 · y6
    var t11 = y7.cyclotomicSquare();
    t11 = t11.mul(y5);
    t11 = t11.mul(y6);

    // T21 = T11 · y4 · y6
    var t21 = t11.mul(y4);
    t21 = t21.mul(y6);

    // T12 = T11 · y3 (y3 = mxxpp)
    const t12 = t11.mul(mxxpp);

    // T22 = T21² · T12
    var t22 = t21.cyclotomicSquare();
    t22 = t22.mul(t12);

    // T23 = T22²
    const t23 = t22.cyclotomicSquare();

    // T24 = T23 · y1
    const t24 = t23.mul(y1);

    // T13 = T23 · y2
    const t13 = t23.mul(y2);

    // T14 = T13² · T24
    var t14 = t13.cyclotomicSquare();
    t14 = t14.mul(t24);

    return t14;
}

/// Compute f^x where x is the BN254 curve parameter
/// Uses cyclotomic squaring since f is a unitary element in the cyclotomic subgroup.
fn expByX(f: Fp12) Fp12 {
    var result = Fp12.one();
    var base = f;
    var exp = BN_X;

    while (exp > 0) {
        if (exp & 1 == 1) {
            result = result.mul(base);
        }
        base = base.cyclotomicSquare();
        exp >>= 1;
    }

    return result;
}

/// Compute f^(-x) for arkworks algorithm
/// Since x is positive for BN254, this returns conjugate(f^x)
fn expByNegX(f: Fp12) Fp12 {
    const fx = expByX(f);
    // x is positive, so exp_by_neg_x returns conjugate
    return fx.conjugate();
}

/// Hard part of final exponentiation using arkworks algorithm
/// From "Faster hashing to G2" by Laura Fuentes-Castaneda et al.
fn hardPartExponentiationArkworks(r: Fp12) Fp12 {
    // y0 = exp_by_neg_x(r)
    const y0 = expByNegX(r);

    // y1 = y0^2 (cyclotomic square)
    const y1 = y0.cyclotomicSquare();

    // y2 = y1^2
    const y2 = y1.cyclotomicSquare();

    // y3 = y2 * y1
    var y3 = y2.mul(y1);

    // y4 = exp_by_neg_x(y3)
    const y4 = expByNegX(y3);

    // y5 = y4^2
    const y5 = y4.cyclotomicSquare();

    // y6 = exp_by_neg_x(y5)
    var y6 = expByNegX(y5);

    // y3 = conjugate(y3)
    y3 = y3.conjugate();

    // y6 = conjugate(y6)
    y6 = y6.conjugate();

    // y7 = y6 * y4
    const y7 = y6.mul(y4);

    // y8 = y7 * y3
    var y8 = y7.mul(y3);

    // y9 = y8 * y1
    const y9 = y8.mul(y1);

    // y10 = y8 * y4
    const y10 = y8.mul(y4);

    // y11 = y10 * r
    const y11 = y10.mul(r);

    // y12 = y9.frobenius_map(1)
    var y12 = y9.frobenius();

    // y13 = y12 * y11
    const y13 = y12.mul(y11);

    // y8 = y8.frobenius_map(2)
    y8 = y8.frobenius2();

    // y14 = y8 * y13
    const y14 = y8.mul(y13);

    // r_inv = conjugate(r) (cyclotomic inverse)
    const r_inv = r.conjugate();

    // y15 = r_inv * y9
    var y15 = r_inv.mul(y9);

    // y15 = y15.frobenius_map(3)
    y15 = y15.frobenius3();

    // y16 = y15 * y14
    const y16 = y15.mul(y14);

    return y16;
}

/// Pairing input pair type with proper Fp coordinates
pub const PairingInputFp = struct {
    p: G1PointFp,
    q: G2Point,
};

/// Multi-pairing with Fp coordinates: product of pairings e(P1,Q1) * e(P2,Q2) * ...
/// Uses batch Miller loop with shared final exponentiation for efficiency.
pub fn multiPairingFp(pairs: []const PairingInputFp) PairingResult {
    if (pairs.len == 0) return Fp12.one();

    var miller_acc = Fp12.one();
    for (pairs) |pair| {
        if (pair.p.infinity or pair.q.infinity) continue;
        const ml = millerLoopArkworks(pair.p, pair.q);
        miller_acc = miller_acc.mul(ml);
    }

    if (miller_acc.eql(Fp12.one())) return Fp12.one();
    return finalExponentiation(miller_acc);
}

/// Check if e(P1, Q1) == e(P2, Q2) with proper Fp coordinates
/// Useful for verifying KZG proofs
pub fn pairingCheckFp(p1: G1PointFp, q1: G2Point, p2: G1PointFp, q2: G2Point) bool {
    // Instead of checking e(P1,Q1) == e(P2,Q2), we check e(P1,Q1) * e(-P2,Q2) == 1
    const p2_neg = p2.neg();
    const pairs = [_]PairingInputFp{
        .{ .p = p1, .q = q1 },
        .{ .p = p2_neg, .q = q2 },
    };
    const result = multiPairingFp(&pairs);
    return result.isOne();
}

// ============================================================================
// Batched Miller Loop (shared squaring across pairs)
// ============================================================================

/// Maximum pairs per sub-batch for unprepared Miller loop.
/// Keeps per-pair G2HomProjective accumulators (~320 bytes/pair) in L1 cache.
const MAX_UNPREPARED_BATCH: usize = 8;

/// Maximum pairs per sub-batch for prepared Miller loop.
/// Only reads EllCoeff per step (~48 bytes/pair), so larger batches fit L1.
const MAX_PREPARED_BATCH: usize = 64;

/// Batched Miller loop using precomputed G2 coefficients.
/// Shares a single Fp12.square() per ATE iteration across all pairs,
/// saving (n-1) × 64 Fp12 squarings compared to n independent Miller loops.
pub fn batchedMillerLoopPrepared(
    g1_points: []const G1PointFp,
    g2_preps: []const G2Prepared,
) Fp12 {
    const n = g1_points.len;
    if (n == 0) return Fp12.one();
    if (n == 1) return millerLoopPrepared(g1_points[0], &g2_preps[0]);

    // Sub-batch to keep per-pair read data in L1 cache
    if (n > MAX_PREPARED_BATCH) {
        var acc = Fp12.one();
        var offset: usize = 0;
        while (offset < n) {
            const batch_end = @min(offset + MAX_PREPARED_BATCH, n);
            const ml = batchedMillerLoopPrepared(g1_points[offset..batch_end], g2_preps[offset..batch_end]);
            acc = acc.mul(ml);
            offset = batch_end;
        }
        return acc;
    }

    var f = Fp12.one();
    var coeff_idx: usize = 0;

    var idx: usize = ATE_LOOP_COUNT.len - 1;
    while (idx >= 1) : (idx -= 1) {
        // ONE shared square per iteration
        if (idx != ATE_LOOP_COUNT.len - 1) {
            f = f.square();
        }

        // Doubling coefficients for all pairs
        for (0..n) |k| {
            if (g1_points[k].infinity or g2_preps[k].infinity) continue;
            const coeffs = g2_preps[k].coeffs[coeff_idx];
            const c0_eval = fp2ScalarMul(coeffs.c0, g1_points[k].y);
            const c1_eval = fp2ScalarMul(coeffs.c1, g1_points[k].x);
            f = fp12MulBy034(f, c0_eval, c1_eval, coeffs.c2);
        }
        coeff_idx += 1;

        // Addition coefficients if bit is non-zero
        const bit = ATE_LOOP_COUNT[idx - 1];
        if (bit == 1 or bit == -1) {
            for (0..n) |k| {
                if (g1_points[k].infinity or g2_preps[k].infinity) continue;
                const coeffs = g2_preps[k].coeffs[coeff_idx];
                const c0_eval = fp2ScalarMul(coeffs.c0, g1_points[k].y);
                const c1_eval = fp2ScalarMul(coeffs.c1, g1_points[k].x);
                f = fp12MulBy034(f, c0_eval, c1_eval, coeffs.c2);
            }
            coeff_idx += 1;
        }
    }

    if (X_IS_NEGATIVE) {
        f = f.conjugate();
    }

    // Final Frobenius steps
    for (0..2) |_| {
        for (0..n) |k| {
            if (g1_points[k].infinity or g2_preps[k].infinity) continue;
            const coeffs = g2_preps[k].coeffs[coeff_idx];
            const c0_eval = fp2ScalarMul(coeffs.c0, g1_points[k].y);
            const c1_eval = fp2ScalarMul(coeffs.c1, g1_points[k].x);
            f = fp12MulBy034(f, c0_eval, c1_eval, coeffs.c2);
        }
        coeff_idx += 1;
    }

    std.debug.assert(coeff_idx == PREPARED_COEFFS_LEN);
    return f;
}

/// Batched Miller loop without precomputed coefficients.
/// Maintains per-pair G2HomProjective accumulators while sharing Fp12.square().
/// For n > MAX_UNPREPARED_BATCH, processes in sub-batches and multiplies results.
pub fn batchedMillerLoopUnprepared(
    g1_points: []const G1PointFp,
    g2_points: []const G2Point,
) Fp12 {
    const n = g1_points.len;
    if (n == 0) return Fp12.one();
    if (n == 1) return millerLoopArkworks(g1_points[0], g2_points[0]);

    // Sub-batch to keep per-pair state in L1 cache
    if (n > MAX_UNPREPARED_BATCH) {
        var acc = Fp12.one();
        var offset: usize = 0;
        while (offset < n) {
            const batch_end = @min(offset + MAX_UNPREPARED_BATCH, n);
            const ml = batchedMillerLoopUnprepared(g1_points[offset..batch_end], g2_points[offset..batch_end]);
            acc = acc.mul(ml);
            offset = batch_end;
        }
        return acc;
    }

    const two_inv = Fp.fromU64(2).inverse() orelse return Fp12.one();

    // Per-pair projective accumulators and negated Q (n <= MAX_UNPREPARED_BATCH, stack is fine)
    var rs: [MAX_UNPREPARED_BATCH]G2HomProjective = undefined;
    var nqs: [MAX_UNPREPARED_BATCH]G2Point = undefined;

    for (0..n) |k| {
        rs[k] = G2HomProjective.fromAffine(g2_points[k]);
        nqs[k] = g2_points[k].neg();
    }

    var f = Fp12.one();

    var idx: usize = ATE_LOOP_COUNT.len - 1;
    while (idx >= 1) : (idx -= 1) {
        if (idx != ATE_LOOP_COUNT.len - 1) {
            f = f.square();
        }

        const bit = ATE_LOOP_COUNT[idx - 1];

        if (bit == 1 or bit == -1) {
            // Non-zero bit: combine doubling + addition lines via sparse-sparse
            for (0..n) |k| {
                if (g1_points[k].infinity or g2_points[k].infinity) continue;
                // Doubling line coefficients
                const coeffs_dbl = rs[k].double_in_place(two_inv);
                const dbl_c0 = fp2ScalarMul(coeffs_dbl.c0, g1_points[k].y);
                const dbl_c1 = fp2ScalarMul(coeffs_dbl.c1, g1_points[k].x);
                // Addition line coefficients
                const q = if (bit == 1) g2_points[k] else nqs[k];
                const coeffs_add = rs[k].add_in_place(q);
                const add_c0 = fp2ScalarMul(coeffs_add.c0, g1_points[k].y);
                const add_c1 = fp2ScalarMul(coeffs_add.c1, g1_points[k].x);
                // Sparse × sparse combination (6 Fp2.mul)
                const combined = fp12Mul034By034(dbl_c0, dbl_c1, coeffs_dbl.c2, add_c0, add_c1, coeffs_add.c2);
                // 01234-sparse × full (17 Fp2.mul)
                f = fp12MulBy01234(f, combined);
            }
        } else {
            // Zero bit: only doubling line
            for (0..n) |k| {
                if (g1_points[k].infinity or g2_points[k].infinity) continue;
                const coeffs_dbl = rs[k].double_in_place(two_inv);
                const c0_eval = fp2ScalarMul(coeffs_dbl.c0, g1_points[k].y);
                const c1_eval = fp2ScalarMul(coeffs_dbl.c1, g1_points[k].x);
                f = fp12MulBy034(f, c0_eval, c1_eval, coeffs_dbl.c2);
            }
        }
    }

    if (X_IS_NEGATIVE) {
        f = f.conjugate();
    }

    // Final Frobenius steps
    for (0..n) |k| {
        if (g1_points[k].infinity or g2_points[k].infinity) continue;
        const q1 = mulByChar(g2_points[k]);
        const coeffs_q1 = rs[k].add_in_place(q1);
        const c0_q1 = fp2ScalarMul(coeffs_q1.c0, g1_points[k].y);
        const c1_q1 = fp2ScalarMul(coeffs_q1.c1, g1_points[k].x);
        f = fp12MulBy034(f, c0_q1, c1_q1, coeffs_q1.c2);
    }

    for (0..n) |k| {
        if (g1_points[k].infinity or g2_points[k].infinity) continue;
        var q2 = mulByChar(mulByChar(g2_points[k]));
        q2.y = q2.y.neg();
        const coeffs_q2 = rs[k].add_in_place(q2);
        const c0_q2 = fp2ScalarMul(coeffs_q2.c0, g1_points[k].y);
        const c1_q2 = fp2ScalarMul(coeffs_q2.c1, g1_points[k].x);
        f = fp12MulBy034(f, c0_q2, c1_q2, coeffs_q2.c2);
    }

    return f;
}

/// Batched Miller loop with sparse-sparse line combination (Phase 2).
/// At non-zero ATE bits, combines the doubling and addition lines via
/// fp12Mul034By034 + fp12MulBy01234 (23 Fp2.mul vs 26 for two fp12MulBy034).
pub fn batchedMillerLoopPreparedSparse(
    g1_points: []const G1PointFp,
    g2_preps: []const G2Prepared,
) Fp12 {
    const n = g1_points.len;
    if (n == 0) return Fp12.one();
    if (n == 1) return millerLoopPrepared(g1_points[0], &g2_preps[0]);

    if (n > MAX_PREPARED_BATCH) {
        var acc = Fp12.one();
        var offset: usize = 0;
        while (offset < n) {
            const batch_end = @min(offset + MAX_PREPARED_BATCH, n);
            const ml = batchedMillerLoopPreparedSparse(g1_points[offset..batch_end], g2_preps[offset..batch_end]);
            acc = acc.mul(ml);
            offset = batch_end;
        }
        return acc;
    }

    var f = Fp12.one();
    var coeff_idx: usize = 0;

    var idx: usize = ATE_LOOP_COUNT.len - 1;
    while (idx >= 1) : (idx -= 1) {
        if (idx != ATE_LOOP_COUNT.len - 1) {
            f = f.square();
        }

        const bit = ATE_LOOP_COUNT[idx - 1];

        if (bit == 1 or bit == -1) {
            // Non-zero bit: combine doubling + addition lines via sparse-sparse
            for (0..n) |k| {
                if (g1_points[k].infinity or g2_preps[k].infinity) continue;
                // Doubling line coefficients
                const dbl = g2_preps[k].coeffs[coeff_idx];
                const dbl_c0 = fp2ScalarMul(dbl.c0, g1_points[k].y);
                const dbl_c1 = fp2ScalarMul(dbl.c1, g1_points[k].x);
                // Addition line coefficients
                const add_coeff = g2_preps[k].coeffs[coeff_idx + 1];
                const add_c0 = fp2ScalarMul(add_coeff.c0, g1_points[k].y);
                const add_c1 = fp2ScalarMul(add_coeff.c1, g1_points[k].x);
                // Sparse × sparse combination (6 Fp2.mul)
                const combined = fp12Mul034By034(dbl_c0, dbl_c1, dbl.c2, add_c0, add_c1, add_coeff.c2);
                // 01234-sparse × full (17 Fp2.mul)
                f = fp12MulBy01234(f, combined);
            }
            coeff_idx += 2;
        } else {
            // Zero bit: only doubling line
            for (0..n) |k| {
                if (g1_points[k].infinity or g2_preps[k].infinity) continue;
                const coeffs = g2_preps[k].coeffs[coeff_idx];
                const c0_eval = fp2ScalarMul(coeffs.c0, g1_points[k].y);
                const c1_eval = fp2ScalarMul(coeffs.c1, g1_points[k].x);
                f = fp12MulBy034(f, c0_eval, c1_eval, coeffs.c2);
            }
            coeff_idx += 1;
        }
    }

    if (X_IS_NEGATIVE) {
        f = f.conjugate();
    }

    // Final Frobenius steps — two addition-only lines, combine with sparse-sparse
    for (0..n) |k| {
        if (g1_points[k].infinity or g2_preps[k].infinity) continue;
        const frob1 = g2_preps[k].coeffs[coeff_idx];
        const f1_c0 = fp2ScalarMul(frob1.c0, g1_points[k].y);
        const f1_c1 = fp2ScalarMul(frob1.c1, g1_points[k].x);
        const frob2 = g2_preps[k].coeffs[coeff_idx + 1];
        const f2_c0 = fp2ScalarMul(frob2.c0, g1_points[k].y);
        const f2_c1 = fp2ScalarMul(frob2.c1, g1_points[k].x);
        const combined = fp12Mul034By034(f1_c0, f1_c1, frob1.c2, f2_c0, f2_c1, frob2.c2);
        f = fp12MulBy01234(f, combined);
    }
    coeff_idx += 2;

    std.debug.assert(coeff_idx == PREPARED_COEFFS_LEN);
    return f;
}

/// Batched Miller loop with affine line precomputation (Phase 3).
/// Uses fp12MulBy34 (10 Fp2.mul) for zero-bit steps and
/// fp12Mul34By34 + fp12MulBy01234 (20 Fp2.mul) for non-zero-bit steps.
pub fn batchedMillerLoopAffine(
    g1_points: []const G1PointFp,
    g2_lines: []const G2PreparedAffine,
) Fp12 {
    const n = g1_points.len;
    if (n == 0) return Fp12.one();

    if (n > MAX_PREPARED_BATCH) {
        var acc = Fp12.one();
        var offset: usize = 0;
        while (offset < n) {
            const batch_end = @min(offset + MAX_PREPARED_BATCH, n);
            const ml = batchedMillerLoopAffine(g1_points[offset..batch_end], g2_lines[offset..batch_end]);
            acc = acc.mul(ml);
            offset = batch_end;
        }
        return acc;
    }

    // Precompute x_neg_over_y and y_inv for each G1 point via batch Fp inversion
    var y_vals: [MAX_PREPARED_BATCH]Fp = undefined;
    var y_scratch: [MAX_PREPARED_BATCH]Fp = undefined;
    var xnoy_vals: [MAX_PREPARED_BATCH]Fp = undefined;

    for (0..n) |k| {
        y_vals[k] = if (g1_points[k].infinity) Fp.zero() else g1_points[k].y;
    }
    Fp.batchInversion(y_vals[0..n], y_scratch[0..n]);

    for (0..n) |k| {
        xnoy_vals[k] = if (g1_points[k].infinity) Fp.zero() else g1_points[k].x.neg().mul(y_vals[k]);
    }

    var f = Fp12.one();
    var coeff_idx: usize = 0;

    var idx: usize = ATE_LOOP_COUNT.len - 1;
    while (idx >= 1) : (idx -= 1) {
        if (idx != ATE_LOOP_COUNT.len - 1) {
            f = f.square();
        }

        const bit = ATE_LOOP_COUNT[idx - 1];

        if (bit == 1 or bit == -1) {
            // Non-zero bit: combine doubling + addition via 34×34
            for (0..n) |k| {
                if (g1_points[k].infinity or g2_lines[k].infinity) continue;
                const dbl_line = evaluateLineSparse(g2_lines[k].coeffs[coeff_idx], xnoy_vals[k], y_vals[k]);
                const add_line = evaluateLineSparse(g2_lines[k].coeffs[coeff_idx + 1], xnoy_vals[k], y_vals[k]);
                const combined = fp12Mul34By34(dbl_line.c3, dbl_line.c4, add_line.c3, add_line.c4);
                f = fp12MulBy01234(f, combined);
            }
            coeff_idx += 2;
        } else {
            // Zero bit: only doubling, use fp12MulBy34
            for (0..n) |k| {
                if (g1_points[k].infinity or g2_lines[k].infinity) continue;
                const line = evaluateLineSparse(g2_lines[k].coeffs[coeff_idx], xnoy_vals[k], y_vals[k]);
                f = fp12MulBy34(f, line.c3, line.c4);
            }
            coeff_idx += 1;
        }
    }

    if (X_IS_NEGATIVE) {
        f = f.conjugate();
    }

    // Final Frobenius steps — combine both lines
    for (0..n) |k| {
        if (g1_points[k].infinity or g2_lines[k].infinity) continue;
        const f1_line = evaluateLineSparse(g2_lines[k].coeffs[coeff_idx], xnoy_vals[k], y_vals[k]);
        const f2_line = evaluateLineSparse(g2_lines[k].coeffs[coeff_idx + 1], xnoy_vals[k], y_vals[k]);
        const combined = fp12Mul34By34(f1_line.c3, f1_line.c4, f2_line.c3, f2_line.c4);
        f = fp12MulBy01234(f, combined);
    }
    coeff_idx += 2;

    std.debug.assert(coeff_idx == PREPARED_COEFFS_LEN);
    return f;
}

// ============================================================================
// Tests
// ============================================================================

test "expByX exponentiation" {
    // Test exponentiation by curve parameter
    const one = Fp12.one();
    const one_x = expByX(one);

    // 1^x = 1
    try std.testing.expect(one_x.eql(Fp12.one()));
}

test "pairing with identity" {
    // e(O, Q) = 1 and e(P, O) = 1
    // Note: G1Point uses BN254Scalar (Fr) for MSM compatibility
    // The pairing function converts to Fp internally
    const g1 = G1Point{ .x = BN254Scalar.one(), .y = BN254Scalar.fromU64(2), .infinity = false };
    const g2_gen = G2Point.generator();
    const g1_identity = G1Point{ .x = BN254Scalar.zero(), .y = BN254Scalar.one(), .infinity = true };
    const g2_identity = G2Point.identity();

    const result1 = pairing(g1_identity, g2_gen);
    const result2 = pairing(g1, g2_identity);

    try std.testing.expect(result1.isOne());
    try std.testing.expect(result2.isOne());
}

// Pairing bilinearity test: verifies e([2]P, Q) = e(P, Q)^2
// Fixed iteration 15: Corrected ξ from (1 + u) to (9 + u) and use proper Fp coordinates
test "pairing bilinearity in G1" {
    // G1 generator (1, 2) in base field Fp - valid point on BN254 curve: y^2 = x^3 + 3
    const g1 = G1PointFp{ .x = Fp.one(), .y = Fp.fromU64(2), .infinity = false };
    const g2_gen = G2Point.generator();

    // Compute e(G1, G2)
    const e_g1_g2 = pairingFp(g1, g2_gen);
    const e_g1_g2_squared = e_g1_g2.mul(e_g1_g2);

    // Compute [2]G1 using point doubling in Fp
    const g1_doubled = G1PointInFp.generator().double();
    const e_2g1_g2 = pairingFp(G1PointFp{
        .x = g1_doubled.x,
        .y = g1_doubled.y,
        .infinity = g1_doubled.infinity,
    }, g2_gen);

    try std.testing.expect(e_2g1_g2.eql(e_g1_g2_squared));
}

test "pairing bilinearity in G2" {
    // Test e(P, [2]Q) = e(P, Q)^2
    const g1 = G1PointFp{ .x = Fp.one(), .y = Fp.fromU64(2), .infinity = false };
    const g2_gen = G2Point.generator();

    // Compute e(G1, G2)
    const e_g1_g2 = pairingFp(g1, g2_gen);
    const e_g1_g2_squared = e_g1_g2.mul(e_g1_g2);

    // Compute [2]G2 using point doubling
    const g2_doubled = g2_gen.double();
    const e_g1_2g2 = pairingFp(g1, g2_doubled);

    try std.testing.expect(e_g1_2g2.eql(e_g1_g2_squared));
}

test "pairing identity" {
    // Test e(P, O) = 1 and e(O, Q) = 1
    const g1 = G1PointFp{ .x = Fp.one(), .y = Fp.fromU64(2), .infinity = false };
    const g2_gen = G2Point.generator();

    // e(P, O) = 1
    const e_g1_o = pairingFp(g1, G2Point.identity());
    try std.testing.expect(e_g1_o.isOne());

    // e(O, Q) = 1
    const e_o_g2 = pairingFp(G1PointFp.identity(), g2_gen);
    try std.testing.expect(e_o_g2.isOne());
}

test "pairing non-degeneracy" {
    // Test e(P, Q) != 1 for non-identity P, Q
    const g1 = G1PointFp{ .x = Fp.one(), .y = Fp.fromU64(2), .infinity = false };
    const g2_gen = G2Point.generator();

    const e_g1_g2 = pairingFp(g1, g2_gen);
    try std.testing.expect(!e_g1_g2.isOne());
}

test "pairing generator comparison with jolt" {
    // Compare e(G1_gen, G2_gen) with Jolt's result
    // Jolt: e(G1_gen, G2_gen) first 16 bytes: 95 0e 87 9d 73 63 1f 5e b5 78 85 89 eb 5f 7e f8
    const g1 = G1PointFp{ .x = Fp.one(), .y = Fp.fromU64(2), .infinity = false };
    const g2_gen = G2Point.generator();

    // Print G2 generator for comparison
    dbg("\n=== Generator Comparison ===\n", .{});

    const g2_x_c0_std = g2_gen.x.c0.fromMontgomery();
    const g2_x_c1_std = g2_gen.x.c1.fromMontgomery();
    const g2_y_c0_std = g2_gen.y.c0.fromMontgomery();
    const g2_y_c1_std = g2_gen.y.c1.fromMontgomery();

    var g2_x_c0_bytes: [32]u8 = undefined;
    var g2_x_c1_bytes: [32]u8 = undefined;
    var g2_y_c0_bytes: [32]u8 = undefined;
    var g2_y_c1_bytes: [32]u8 = undefined;
    for (0..4) |i| {
        std.mem.writeInt(u64, g2_x_c0_bytes[i * 8 ..][0..8], g2_x_c0_std.limbs[i], .little);
        std.mem.writeInt(u64, g2_x_c1_bytes[i * 8 ..][0..8], g2_x_c1_std.limbs[i], .little);
        std.mem.writeInt(u64, g2_y_c0_bytes[i * 8 ..][0..8], g2_y_c0_std.limbs[i], .little);
        std.mem.writeInt(u64, g2_y_c1_bytes[i * 8 ..][0..8], g2_y_c1_std.limbs[i], .little);
    }
    dbg("Zolt G2 generator:\n", .{});
    dbg("  x.c0 first 16: {x}\n", .{g2_x_c0_bytes[0..16].*});
    dbg("  x.c1 first 16: {x}\n", .{g2_x_c1_bytes[0..16].*});
    dbg("  y.c0 first 16: {x}\n", .{g2_y_c0_bytes[0..16].*});
    dbg("  y.c1 first 16: {x}\n", .{g2_y_c1_bytes[0..16].*});

    // Jolt G2 generator:
    // x.c0: ed f6 92 d9 5c bd de 46 dd da 5e f7 d4 22 43 67
    // x.c1: c2 12 f3 ae b7 85 e4 97 12 e7 a9 35 33 49 aa f1
    // y.c0: aa 7d fa 66 01 cc e6 4c 7b d3 43 0c 69 e7 d1 e3
    // y.c1: 5b 97 22 d1 dc da ac 55 f3 8e b3 70 33 31 4b bc
    const jolt_x_c0 = [_]u8{ 0xed, 0xf6, 0x92, 0xd9, 0x5c, 0xbd, 0xde, 0x46, 0xdd, 0xda, 0x5e, 0xf7, 0xd4, 0x22, 0x43, 0x67 };
    const jolt_x_c1 = [_]u8{ 0xc2, 0x12, 0xf3, 0xae, 0xb7, 0x85, 0xe4, 0x97, 0x12, 0xe7, 0xa9, 0x35, 0x33, 0x49, 0xaa, 0xf1 };
    const jolt_y_c0 = [_]u8{ 0xaa, 0x7d, 0xfa, 0x66, 0x01, 0xcc, 0xe6, 0x4c, 0x7b, 0xd3, 0x43, 0x0c, 0x69, 0xe7, 0xd1, 0xe3 };
    const jolt_y_c1 = [_]u8{ 0x5b, 0x97, 0x22, 0xd1, 0xdc, 0xda, 0xac, 0x55, 0xf3, 0x8e, 0xb3, 0x70, 0x33, 0x31, 0x4b, 0xbc };

    var g2_match = std.mem.eql(u8, g2_x_c0_bytes[0..16], &jolt_x_c0);
    g2_match = g2_match and std.mem.eql(u8, g2_x_c1_bytes[0..16], &jolt_x_c1);
    g2_match = g2_match and std.mem.eql(u8, g2_y_c0_bytes[0..16], &jolt_y_c0);
    g2_match = g2_match and std.mem.eql(u8, g2_y_c1_bytes[0..16], &jolt_y_c1);

    if (g2_match) {
        dbg("*** G2 generator MATCHES Jolt! ***\n", .{});
    } else {
        dbg("*** G2 generator MISMATCH ***\n", .{});
    }

    const e_g1_g2 = pairingFp(g1, g2_gen);
    const bytes = e_g1_g2.toBytes();

    dbg("\nZolt e(G1_gen, G2_gen) first 16 bytes: {x}\n", .{bytes[0..16].*});

    const jolt_bytes = [_]u8{ 0x95, 0x0e, 0x87, 0x9d, 0x73, 0x63, 0x1f, 0x5e, 0xb5, 0x78, 0x85, 0x89, 0xeb, 0x5f, 0x7e, 0xf8 };
    if (std.mem.eql(u8, bytes[0..16], &jolt_bytes)) {
        dbg("*** Generator pairing MATCHES Jolt! ***\n", .{});
    } else {
        dbg("*** Generator pairing MISMATCH ***\n", .{});
        dbg("Expected (Jolt): {x}\n", .{jolt_bytes});
    }
}

test "pairingCheckFp basic" {
    // Test that e(P, Q) == e(P, Q) returns true
    const g1 = G1PointFp{ .x = Fp.one(), .y = Fp.fromU64(2), .infinity = false };
    const g2_gen = G2Point.generator();

    // e(P, Q) == e(P, Q) should be true
    try std.testing.expect(pairingCheckFp(g1, g2_gen, g1, g2_gen));
}

test "pairingCheckFp bilinearity" {
    // Test that e([2]P, Q) == e(P, [2]Q)
    // This is a consequence of bilinearity
    const g1 = G1PointFp{ .x = Fp.one(), .y = Fp.fromU64(2), .infinity = false };
    const g2_gen = G2Point.generator();

    // [2]G1
    const g1_doubled = G1PointInFp.generator().double();
    const g1_2 = G1PointFp{ .x = g1_doubled.x, .y = g1_doubled.y, .infinity = g1_doubled.infinity };

    // [2]G2
    const g2_2 = g2_gen.double();

    // e([2]P, Q) == e(P, [2]Q)
    try std.testing.expect(pairingCheckFp(g1_2, g2_gen, g1, g2_2));
}

test "batchedMillerLoopPrepared matches individual" {
    const g1 = G1PointFp{ .x = Fp.one(), .y = Fp.fromU64(2), .infinity = false };
    const g2_gen = G2Point.generator();
    const prep = G2Prepared.fromG2Point(g2_gen);

    // n=1: should match single millerLoopPrepared
    const single = millerLoopPrepared(g1, &prep);
    const batch1 = batchedMillerLoopPrepared(&.{g1}, &.{prep});
    try std.testing.expect(single.eql(batch1));

    // n=2: should match product of two individual Miller loops
    const g1b = G1PointFp{ .x = Fp.fromU64(3), .y = Fp.fromU64(5), .infinity = false };
    const g2b = G2Point.generator();
    const prep_b = G2Prepared.fromG2Point(g2b);

    const ml_a = millerLoopPrepared(g1, &prep);
    const ml_b = millerLoopPrepared(g1b, &prep_b);
    const product = ml_a.mul(ml_b);

    const batch2 = batchedMillerLoopPrepared(&.{ g1, g1b }, &.{ prep, prep_b });
    try std.testing.expect(product.eql(batch2));
}

test "batchedMillerLoopUnprepared matches prepared" {
    const g1 = G1PointFp{ .x = Fp.one(), .y = Fp.fromU64(2), .infinity = false };
    const g2_gen = G2Point.generator();
    const prep = G2Prepared.fromG2Point(g2_gen);

    const prepared_result = batchedMillerLoopPrepared(&.{g1}, &.{prep});
    const unprepared_result = batchedMillerLoopUnprepared(&.{g1}, &.{g2_gen});
    try std.testing.expect(prepared_result.eql(unprepared_result));
}

test "batchedMillerLoopPreparedSparse matches non-sparse" {
    const g1 = G1PointFp{ .x = Fp.one(), .y = Fp.fromU64(2), .infinity = false };
    const g2_gen = G2Point.generator();
    const prep = G2Prepared.fromG2Point(g2_gen);

    const non_sparse = batchedMillerLoopPrepared(&.{g1}, &.{prep});
    const sparse = batchedMillerLoopPreparedSparse(&.{g1}, &.{prep});
    try std.testing.expect(non_sparse.eql(sparse));
}

test "batchedMillerLoopAffine matches prepared" {
    const g1 = G1PointFp{ .x = Fp.one(), .y = Fp.fromU64(2), .infinity = false };
    const g2_gen = G2Point.generator();
    const prep = G2Prepared.fromG2Point(g2_gen);
    const affine_prep = G2PreparedAffine.fromG2Prepared(&prep);

    const prepared_result = millerLoopPrepared(g1, &prep);
    const affine_result = batchedMillerLoopAffine(&.{g1}, &.{affine_prep});

    // Affine path differs from projective by a factor killed by final exponentiation
    const fe_prep = finalExponentiation(prepared_result);
    const fe_affine = finalExponentiation(affine_result);
    try std.testing.expect(fe_prep.eql(fe_affine));
}

// ============================================================================
// Fixture-backed vector tests (arkworks-validated)
// ============================================================================

const testdata = @import("../testdata.zig");
const msm_mod = @import("../msm/mod.zig");
const G1MSM = msm_mod.MSM(BN254Scalar, Fp);

test "pairing fixture vectors" {
    const fixture_text = @embedFile("../testdata/pairing/generator_vectors.txt");
    var lines = std.mem.splitScalar(u8, fixture_text, '\n');

    var case_count: usize = 0;
    while (lines.next()) |raw_line| {
        const line = testdata.cleanLine(raw_line) orelse continue;
        const fields = try testdata.splitFieldsExact(4, line, '|');

        const g1_scalar = BN254Scalar.fromU64(try testdata.parseDecimal(u64, fields[1]));
        const g2_scalar = BN254Scalar.fromU64(try testdata.parseDecimal(u64, fields[2]));

        const g1_affine = G1MSM.scalarMul(G1PointInFp.generator(), g1_scalar).toAffine();
        const g1 = G1PointFp{
            .x = g1_affine.x,
            .y = g1_affine.y,
            .infinity = g1_affine.infinity,
        };
        const g2_pt = G2Point.generator().scalarMul(g2_scalar);

        const expected = try testdata.parseHexBytesExact(384, fields[3]);
        const actual = pairingFp(g1, g2_pt).toBytes();
        try std.testing.expectEqualSlices(u8, &expected, &actual);
        case_count += 1;
    }
    try std.testing.expect(case_count >= 5);
}

test {
    // Run tests from sub-modules
    _ = extensions;
    _ = g2;
}
