//! Product-tree helpers for multipoint evaluation (ported from Jolt's mles_product_sum.rs).
//!
//! Contains divide-and-conquer product-tree routines for evaluating products
//! of linear polynomials, plus `finishMlesProductSumFromEvals` and `toCompressed`.

const std = @import("std");
const Allocator = std.mem.Allocator;
const field_mod = @import("../field/mod.zig");
const interp_mod = @import("interpolation.zig");

/// Product-tree helpers parameterised on a field type `F`.
pub fn ProductTree(comptime F: type) type {
    const Interp = interp_mod.Interpolation(F);

    return struct {
        const Self = @This();

        // =============================================================
        // Private extrapolation helpers
        // =============================================================

        /// Extrapolate degree-2 polynomial to next integer point.
        /// Given f[0]=P(k), f[1]=P(k+1), f_inf=leading coeff, returns P(k+2).
        /// Cost: 0 field multiplications.
        inline fn ex2(f: [2]F, f_inf: F) F {
            return f[1].add(f_inf).double().sub(f[0]);
        }

        /// Extrapolate degree-4 polynomial to next integer point.
        /// f[0..4]=P(k)..P(k+3), f_inf6=6*leading_coeff. Returns P(k+4).
        /// Cost: 0 field multiplications.
        inline fn ex4(f: [4]F, f_inf6: F) F {
            var t = f_inf6;
            t = t.add(f[3]);
            t = t.sub(f[2]);
            t = t.add(f[1]);
            t = t.double();
            t = t.sub(f[2]);
            t = t.double();
            t = t.sub(f[0]);
            return t;
        }

        /// Joint extrapolation of two consecutive degree-4 polynomial points.
        /// Returns (P(k+4), P(k+5)). Cost: 0 field multiplications.
        inline fn ex4_2(f: [4]F, f_inf6: F) [2]F {
            const f3m2 = f[3].sub(f[2]);
            var f4 = f_inf6;
            f4 = f4.add(f3m2);
            f4 = f4.add(f[1]);
            f4 = f4.double();
            f4 = f4.sub(f[2]);
            f4 = f4.double();
            f4 = f4.sub(f[0]);

            var f5 = f4.sub(f3m2).add(f_inf6);
            f5 = f5.double();
            f5 = f5.sub(f[3]);
            f5 = f5.double();
            f5 = f5.sub(f[1]);

            return .{ f4, f5 };
        }

        // =============================================================
        // Private product-tree internals
        // =============================================================

        /// Product of 2 linear polynomials at {1, 2, inf}. Cost: 3 field muls.
        inline fn evalLinearProd2Internal(p: [2]F, q: [2]F) [3]F {
            const p_inf = p[1].sub(p[0]);
            const p2 = p_inf.add(p[1]);
            const q_inf = q[1].sub(q[0]);
            const q2 = q_inf.add(q[1]);
            return .{
                p[1].mul(q[1]), // P(1)
                p2.mul(q2), // P(2)
                p_inf.mul(q_inf), // P(inf)
            };
        }

        /// Product of 4 linear polynomials at {1, 2, 3, 4, inf}. Cost: 11 field muls.
        inline fn evalLinearProd4Internal(p: [4][2]F) [5]F {
            const ab = evalLinearProd2Internal(p[0], p[1]);
            const a3 = ex2(.{ ab[0], ab[1] }, ab[2]);
            const a4 = ex2(.{ ab[1], a3 }, ab[2]);

            const cd = evalLinearProd2Internal(p[2], p[3]);
            const b3 = ex2(.{ cd[0], cd[1] }, cd[2]);
            const b4 = ex2(.{ cd[1], b3 }, cd[2]);

            return .{
                ab[0].mul(cd[0]), // P(1)
                ab[1].mul(cd[1]), // P(2)
                a3.mul(b3), // P(3)
                a4.mul(b4), // P(4)
                ab[2].mul(cd[2]), // P(inf)
            };
        }

        /// Product of 8 linear polynomials at {1..8, inf}. Cost: 31 muls + 2 mulU64.
        inline fn evalLinearProd8Internal(p: [8][2]F) [9]F {
            // Left half
            const a = evalLinearProd4Internal(p[0..4].*);
            const a_inf6 = field_mod.mulU64(a[4], 6);
            const a_56 = ex4_2(.{ a[0], a[1], a[2], a[3] }, a_inf6);
            const a_78 = ex4_2(.{ a[2], a[3], a_56[0], a_56[1] }, a_inf6);

            // Right half
            const b = evalLinearProd4Internal(p[4..8].*);
            const b_inf6 = field_mod.mulU64(b[4], 6);
            const b_56 = ex4_2(.{ b[0], b[1], b[2], b[3] }, b_inf6);
            const b_78 = ex4_2(.{ b[2], b[3], b_56[0], b_56[1] }, b_inf6);

            // 9 pointwise cross-products
            return .{
                a[0].mul(b[0]), // P(1)
                a[1].mul(b[1]), // P(2)
                a[2].mul(b[2]), // P(3)
                a[3].mul(b[3]), // P(4)
                a_56[0].mul(b_56[0]), // P(5)
                a_56[1].mul(b_56[1]), // P(6)
                a_78[0].mul(b_78[0]), // P(7)
                a_78[1].mul(b_78[1]), // P(8)
                a[4].mul(b[4]), // P(inf)
            };
        }

        // =============================================================
        // Public API
        // =============================================================

        /// Evaluate the product of 9 linear polynomials at points [1, 2, ..., 8, ∞]
        ///
        /// Uses divide-and-conquer product tree: 40 field muls total
        /// (31 from 8-internal + 9 cross-products with 9th factor).
        pub inline fn evalLinearProd9(pairs: [9][2]F) [9]F {
            const tree8 = evalLinearProd8Internal(pairs[0..8].*);

            // 9th factor: slide linearly from p(1) to p(8)
            const delta = pairs[8][1].sub(pairs[8][0]);
            var cur = pairs[8][1]; // p_9(1)
            var result: [9]F = undefined;

            result[0] = tree8[0].mul(cur);
            inline for (1..8) |i| {
                cur = cur.add(delta);
                result[i] = tree8[i].mul(cur);
            }
            result[8] = tree8[8].mul(delta); // P(inf)

            return result;
        }

        /// Evaluate product of 9 linear polynomials and accumulate into unreduced
        /// product accumulators. Defers Montgomery reduction for better throughput.
        /// Cost: 31 muls (8-internal) + 2 mulU64 + 9 widening muls.
        pub inline fn evalProd9Accumulate(
            pairs: [9][2]F,
            accum: *[9]field_mod.UnreducedProductAccum,
        ) void {
            const tree8 = evalLinearProd8Internal(pairs[0..8].*);

            const delta = pairs[8][1].sub(pairs[8][0]);
            var cur = pairs[8][1];

            accum[0].addAssign(tree8[0].mulToProductAccum(cur));
            inline for (1..8) |i| {
                cur = cur.add(delta);
                accum[i].addAssign(tree8[i].mulToProductAccum(cur));
            }
            accum[8].addAssign(tree8[8].mulToProductAccum(delta));
        }

        /// Evaluate product of 10 linear polynomials at points [1, 2, ..., 9, ∞]
        ///
        /// Uses product tree for first 8, slides 9th and 10th factors.
        /// Cost: ~58 muls (was 90 naive).
        pub inline fn evalLinearProd10(pairs: [10][2]F) [10]F {
            const tree8 = evalLinearProd8Internal(pairs[0..8].*);

            const delta9 = pairs[8][1].sub(pairs[8][0]);
            const delta10 = pairs[9][1].sub(pairs[9][0]);
            var cur9 = pairs[8][1];
            var cur10 = pairs[9][1];
            var result: [10]F = undefined;

            // Points 1..8: tree8[i] * p9(i+1) * p10(i+1)
            result[0] = tree8[0].mul(cur9).mul(cur10);
            inline for (1..8) |i| {
                cur9 = cur9.add(delta9);
                cur10 = cur10.add(delta10);
                result[i] = tree8[i].mul(cur9).mul(cur10);
            }

            // Point 9: compute tree8 at point 9 naively (product of 8 linear polys at x=9)
            cur9 = cur9.add(delta9);
            cur10 = cur10.add(delta10);
            var tree8_at_9 = pairs[0][0].add(pairs[0][1].sub(pairs[0][0]).mul(F.fromU64(9)));
            inline for (1..8) |i| {
                const pi_9 = pairs[i][0].add(pairs[i][1].sub(pairs[i][0]).mul(F.fromU64(9)));
                tree8_at_9 = tree8_at_9.mul(pi_9);
            }
            result[8] = tree8_at_9.mul(cur9).mul(cur10);

            // Point inf
            result[9] = tree8[8].mul(delta9).mul(delta10);

            return result;
        }

        /// Finish MLE product sum computation from evaluations
        ///
        /// Given evaluations of q(X) / eq(X, r[round]) at [1, 2, ..., d-1, ∞],
        /// recovers the full polynomial q(X) such that q(0) + q(1) = claim.
        ///
        /// This matches Jolt's finish_mles_product_sum_from_evals():
        /// 1. Compute eval_at_0 from the claim and sum_evals[0]
        /// 2. Interpolate the quotient polynomial from [eval_at_0, eval_at_1, ..., eval_at_{d-1}, eval_at_∞]
        /// 3. Multiply by eq(X, r[round]) to get the final polynomial
        ///
        /// Returns coefficients [c0, c1, ..., c_{d+1}] where d = sum_evals.len
        pub fn finishMlesProductSumFromEvals(allocator: Allocator, sum_evals: []const F, claim: F, r_round: F) ![]F {
            // eq(X, r) = (1 - r) + (2r - 1)X = eq_at_0 + eq_slope * X
            const eq_at_0 = F.one().sub(r_round);
            const eq_at_1 = r_round;

            // claim = q(0) + q(1) = eq_at_0 * quot(0) + eq_at_1 * quot(1)
            // where quot(1) = sum_evals[0]
            // So: eq_at_0 * quot(0) = claim - eq_at_1 * sum_evals[0]
            //     quot(0) = (claim - eq_at_1 * sum_evals[0]) / eq_at_0
            const eval_at_1 = sum_evals[0];
            const eval_at_0 = if (eq_at_0.eql(F.zero()))
                F.zero()
            else
                claim.sub(eq_at_1.mul(eval_at_1)).mul(eq_at_0.inverse().?);

            // Build full toom evaluations: [eval_at_0, sum_evals[0..]]
            const toom_evals = try allocator.alloc(F, sum_evals.len + 1);
            defer allocator.free(toom_evals);
            toom_evals[0] = eval_at_0;
            @memcpy(toom_evals[1..], sum_evals);

            // Interpolate quotient polynomial
            const tmp_coeffs = try Interp.fromEvalsToom(allocator, toom_evals);
            defer allocator.free(tmp_coeffs);

            // Multiply by eq(X, r[round]) = (1 - r) + (2r - 1)X
            // eq_c0 = 1 - r, eq_c1 = 2r - 1
            const constant_coeff = F.one().sub(r_round);
            const x_coeff = r_round.add(r_round).sub(F.one());

            // Result is degree (d-1) * 1 + 1 = d where d = sum_evals.len (quotient degree d-1)
            // So final degree = tmp_coeffs.len
            const coeffs = try allocator.alloc(F, tmp_coeffs.len + 1);
            @memset(coeffs, F.zero());

            for (tmp_coeffs, 0..) |coeff, i| {
                coeffs[i] = coeffs[i].add(coeff.mul(constant_coeff));
                coeffs[i + 1] = coeffs[i + 1].add(coeff.mul(x_coeff));
            }

            return coeffs;
        }

        /// Convert polynomial coefficients to compressed format [c0, c2, c3, ..., c_d]
        /// This skips c1 which is recovered from the hint (claim).
        pub fn toCompressed(allocator: Allocator, coeffs: []const F) ![]F {
            if (coeffs.len <= 1) {
                const result = try allocator.alloc(F, coeffs.len);
                if (coeffs.len > 0) result[0] = coeffs[0];
                return result;
            }

            // compressed = [c0, c2, c3, ..., c_d] (skip c1)
            const compressed = try allocator.alloc(F, coeffs.len - 1);
            compressed[0] = coeffs[0]; // c0
            for (2..coeffs.len) |i| {
                compressed[i - 1] = coeffs[i]; // c2, c3, ..., c_d
            }
            return compressed;
        }
    };
}
