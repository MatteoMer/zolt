//! Interpolation and evaluation utilities for univariate polynomials.
//!
//! Contains Vandermonde interpolation, Toom-Cook interpolation,
//! compressed-format conversions, and Newton forward-difference evaluation.
//! These are used by `UniPoly` in mod.zig but live here to keep file sizes manageable.

const std = @import("std");
const Allocator = std.mem.Allocator;

/// Interpolation helpers parameterised on a field type `F`.
pub fn Interpolation(comptime F: type) type {
    return struct {
        const Self = @This();

        /// Precomputed field constants to avoid redundant inverse() calls per round.
        /// Each inverse costs ~256 Montgomery muls via Fermat's little theorem.
        pub const INV2: F = blk: {
            @setEvalBranchQuota(1000000);
            break :blk F.fromU64(2).inverse().?;
        };
        pub const INV6: F = blk: {
            @setEvalBranchQuota(1000000);
            break :blk F.fromU64(6).inverse().?;
        };

        /// Interpolate a degree-3 polynomial from evaluations at 0, 1, 2, 3
        ///
        /// Given p(0), p(1), p(2), p(3), returns coefficients [c0, c1, c2, c3]
        /// where p(X) = c0 + c1*X + c2*X² + c3*X³
        ///
        /// Uses the explicit inverse of the Vandermonde matrix.
        pub fn interpolateDegree3(evals: [4]F) [4]F {
            const p0 = evals[0];
            const p1 = evals[1];
            const p2 = evals[2];
            const p3 = evals[3];

            // c0 = p(0)
            const c0 = p0;

            // For the other coefficients, we solve the Vandermonde system
            // The inverse of the 4x4 Vandermonde matrix at points 0,1,2,3 gives:
            //
            // c1 = (-11*p0 + 18*p1 - 9*p2 + 2*p3) / 6
            // c2 = (2*p0 - 5*p1 + 4*p2 - p3) / 2
            // c3 = (-p0 + 3*p1 - 3*p2 + p3) / 6
            //
            // We compute these using field arithmetic

            // Compute 1/6 and 1/2 as field inverses
            const inv6 = INV6;
            const inv2 = INV2;

            // c1 = (-11*p0 + 18*p1 - 9*p2 + 2*p3) / 6
            const c1_num = F.zero()
                .sub(F.fromU64(11).mul(p0))
                .add(F.fromU64(18).mul(p1))
                .sub(F.fromU64(9).mul(p2))
                .add(F.fromU64(2).mul(p3));
            const c1 = c1_num.mul(inv6);

            // c2 = (2*p0 - 5*p1 + 4*p2 - p3) / 2
            const c2_num = F.fromU64(2).mul(p0)
                .sub(F.fromU64(5).mul(p1))
                .add(F.fromU64(4).mul(p2))
                .sub(p3);
            const c2 = c2_num.mul(inv2);

            // c3 = (-p0 + 3*p1 - 3*p2 + p3) / 6
            const c3_num = F.zero().sub(p0)
                .add(F.fromU64(3).mul(p1))
                .sub(F.fromU64(3).mul(p2))
                .add(p3);
            const c3 = c3_num.mul(inv6);

            return [4]F{ c0, c1, c2, c3 };
        }

        /// Convert evaluations at 0,1,2,3 to Jolt's compressed format [c0, c2, c3]
        ///
        /// Jolt stores coefficients except the linear term, which is recovered from the hint.
        pub fn evalsToCompressed(evals: [4]F) [3]F {
            const coeffs = interpolateDegree3(evals);
            return [3]F{ coeffs[0], coeffs[2], coeffs[3] };
        }

        /// Convert Toom-Cook style evaluations [p(0), p(1), p(2), p_inf] to full coefficients [c0, c1, c2, c3]
        ///
        /// For a cubic polynomial p(x) = c0 + c1*x + c2*x^2 + c3*x^3:
        /// - p(0) = c0
        /// - p(1) = c0 + c1 + c2 + c3
        /// - p(2) = c0 + 2*c1 + 4*c2 + 8*c3
        /// - p_inf = c3 (leading coefficient)
        pub fn toomCookToCoeffs(evals: [4]F) [4]F {
            const p0 = evals[0];
            const p1 = evals[1];
            const p2 = evals[2];
            const p_inf = evals[3];

            // c0 = p(0)
            const c0 = p0;

            // c3 = p_inf
            const c3 = p_inf;

            // c2 = (p(2) - 2*p(1) + p(0) - 6*p_inf) / 2
            const two = F.fromU64(2);
            const six = F.fromU64(6);
            const inv2 = INV2;

            const c2_num = p2.sub(two.mul(p1)).add(p0).sub(six.mul(p_inf));
            const c2 = c2_num.mul(inv2);

            // c1 = p(1) - c0 - c2 - c3
            const c1 = p1.sub(c0).sub(c2).sub(c3);

            return [4]F{ c0, c1, c2, c3 };
        }

        /// Convert Toom-Cook style evaluations [p(0), p(1), p(2), p_inf] to Jolt's compressed format [c0, c2, c3]
        ///
        /// For a cubic polynomial p(x) = c0 + c1*x + c2*x^2 + c3*x^3:
        /// - p(0) = c0
        /// - p(1) = c0 + c1 + c2 + c3
        /// - p(2) = c0 + 2*c1 + 4*c2 + 8*c3
        /// - p_inf = c3 (leading coefficient)
        ///
        /// Solving for c2:
        /// c2 = (p(2) - 2*p(1) + p(0) - 6*p_inf) / 2
        pub fn toomCookToCompressed(evals: [4]F) [3]F {
            const p0 = evals[0];
            const p1 = evals[1];
            const p2 = evals[2];
            const p_inf = evals[3];

            // c0 = p(0)
            const c0 = p0;

            // c3 = p_inf
            const c3 = p_inf;

            // c2 = (p(2) - 2*p(1) + p(0) - 6*p_inf) / 2
            const two = F.fromU64(2);
            const six = F.fromU64(6);
            const inv2 = INV2;

            const c2_num = p2.sub(two.mul(p1)).add(p0).sub(six.mul(p_inf));
            const c2 = c2_num.mul(inv2);

            // Compressed format: [c0, c2, c3] - Jolt omits c1 and recovers from hint
            return [3]F{ c0, c2, c3 };
        }

        /// Evaluate a cubic polynomial at a point given Toom-Cook style evaluations [p(0), p(1), p(2), p_inf]
        ///
        /// First converts to coefficients [c0, c1, c2, c3], then evaluates at x using Horner's method.
        /// This matches how Jolt's prover evaluates round polynomials.
        pub fn evaluateToomCookAt(evals: [4]F, x: F) F {
            // Convert to coefficients
            const coeffs = toomCookToCoeffs(evals);

            // Evaluate using Horner's method: c0 + x*(c1 + x*(c2 + x*c3))
            var result = coeffs[3]; // c3
            result = result.mul(x).add(coeffs[2]); // c3*x + c2
            result = result.mul(x).add(coeffs[1]); // (c3*x + c2)*x + c1
            result = result.mul(x).add(coeffs[0]); // ((c3*x + c2)*x + c1)*x + c0
            return result;
        }

        /// Create compressed coefficients from previous claim and degree-2 evaluations
        ///
        /// For a degree-2 polynomial p(x) = c0 + c1*x + c2*x^2:
        /// - p(0) = c0 = eval_0
        /// - p(1) = c0 + c1 + c2 = previous_claim - eval_0 (from sumcheck property p(0)+p(1)=claim)
        /// - p(2) = c0 + 2*c1 + 4*c2 = eval_2
        ///
        /// Solving:
        /// - c0 = eval_0
        /// - c1 = previous_claim - eval_0 - c2 (recovered from hint)
        /// - c2 = (eval_2 - 2*p(1) + eval_0) / 2
        ///      = (eval_2 - 2*(previous_claim - eval_0) + eval_0) / 2
        ///      = (eval_2 - 2*previous_claim + 3*eval_0) / 2
        ///
        /// Returns [c0, c2, 0] since degree is 2 (no cubic term)
        pub fn fromEvalsAndHint(previous_claim: F, eval_0: F, eval_2: F) struct { coeffs: [3]F } {
            const p1 = previous_claim.sub(eval_0); // p(1) from sumcheck property

            // c2 = (eval_2 - 2*p(1) + eval_0) / 2
            const two = F.fromU64(2);
            const c2_num = eval_2.sub(two.mul(p1)).add(eval_0);
            const c2 = c2_num.mul(INV2);

            // Compressed format: [c0, c2, c3] where c3=0 for degree-2
            return .{ .coeffs = [3]F{ eval_0, c2, F.zero() } };
        }

        /// Interpolate a polynomial from evaluations at [0, 1, ..., d-1, ∞]
        ///
        /// Given evaluations [p(0), p(1), ..., p(d-1), p(∞)], where p(∞) is the leading coefficient,
        /// returns all coefficients [c0, c1, ..., c_d].
        ///
        /// This matches Jolt's UniPoly::from_evals_toom() which uses Gaussian elimination
        /// on an augmented Vandermonde matrix.
        pub fn fromEvalsToom(allocator: Allocator, evals: []const F) ![]F {
            const n = evals.len;
            if (n == 0) {
                return try allocator.alloc(F, 0);
            }

            // Build augmented Vandermonde matrix and solve via Gaussian elimination
            // Matrix is n x (n+1): [Vandermonde | evals]
            var matrix = try allocator.alloc([]F, n);
            errdefer {
                for (matrix) |row| {
                    allocator.free(row);
                }
                allocator.free(matrix);
            }

            // Rows for finite x values (0, 1, ..., n-2)
            for (0..n - 1) |i| {
                matrix[i] = try allocator.alloc(F, n + 1);
                matrix[i][0] = F.one();
                const x = F.fromU64(@intCast(i));
                for (1..n) |j| {
                    matrix[i][j] = matrix[i][j - 1].mul(x);
                }
                matrix[i][n] = evals[i]; // RHS
            }

            // Row for x=infinity: coefficients are [0, 0, ..., 0, 1] and RHS is evals[n-1]
            matrix[n - 1] = try allocator.alloc(F, n + 1);
            for (0..n - 1) |j| {
                matrix[n - 1][j] = F.zero();
            }
            matrix[n - 1][n - 1] = F.one(); // Leading coefficient position
            matrix[n - 1][n] = evals[n - 1]; // p(∞) = leading coefficient

            // Gaussian elimination with partial pivoting
            for (0..n) |col| {
                // Find pivot
                var max_row = col;
                for ((col + 1)..n) |row| {
                    // Simple comparison - just use first non-zero as pivot
                    if (!matrix[row][col].eql(F.zero()) and matrix[max_row][col].eql(F.zero())) {
                        max_row = row;
                    }
                }

                // Swap rows if needed
                if (max_row != col) {
                    const tmp = matrix[col];
                    matrix[col] = matrix[max_row];
                    matrix[max_row] = tmp;
                }

                // Check for zero pivot
                if (matrix[col][col].eql(F.zero())) {
                    // Skip if zero pivot (singular matrix case)
                    continue;
                }

                // Eliminate column
                const pivot_inv = matrix[col][col].inverse().?;
                for ((col + 1)..n) |row| {
                    if (!matrix[row][col].eql(F.zero())) {
                        const factor = matrix[row][col].mul(pivot_inv);
                        for (col..n + 1) |j| {
                            matrix[row][j] = matrix[row][j].sub(factor.mul(matrix[col][j]));
                        }
                    }
                }
            }

            // Back substitution
            const coeffs = try allocator.alloc(F, n);
            var i_plus_1 = n;
            while (i_plus_1 > 0) {
                const i = i_plus_1 - 1;
                i_plus_1 -= 1;

                var sum = matrix[i][n]; // RHS
                for ((i + 1)..n) |j| {
                    sum = sum.sub(matrix[i][j].mul(coeffs[j]));
                }
                if (!matrix[i][i].eql(F.zero())) {
                    coeffs[i] = sum.mul(matrix[i][i].inverse().?);
                } else {
                    coeffs[i] = F.zero();
                }
            }

            // Free matrix
            for (matrix) |row| {
                allocator.free(row);
            }
            allocator.free(matrix);

            return coeffs;
        }

        /// Convert Toom-Cook evaluations [p(0), p(1), ..., p(d-1), p_inf] to compressed format [c0, c2, c3, ..., c_d]
        ///
        /// For a degree-d polynomial, given d+1 evaluations (d finite points + point at infinity),
        /// this computes all coefficients via Gaussian elimination, then returns the compressed
        /// format: all coefficients except c1 (linear term).
        ///
        /// The compressed format has d elements: [c0, c2, c3, ..., c_d]
        /// The verifier recovers c1 = hint - 2*c0 - c2 - c3 - ... - c_d
        pub fn toomCookToCompressedGeneral(allocator: Allocator, evals: []const F) ![]F {
            const n = evals.len; // n = d + 1 (number of evaluation points)
            if (n == 0) return try allocator.alloc(F, 0);

            // Get all coefficients via interpolation
            const coeffs = try fromEvalsToom(allocator, evals);
            defer allocator.free(coeffs);

            // Compressed format: [c0, c2, c3, ..., c_d] (skip c1)
            // Size = n - 1 (all coefficients except linear term)
            const compressed = try allocator.alloc(F, n - 1);
            compressed[0] = coeffs[0]; // c0
            for (1..n - 1) |i| {
                compressed[i] = coeffs[i + 1]; // c2, c3, ..., c_d
            }
            return compressed;
        }

        /// Evaluate a general-degree polynomial at a point given Toom-Cook evaluations
        ///
        /// Given [p(0), p(1), ..., p(d-1), p_inf], interpolates coefficients and evaluates at x.
        pub fn evaluateToomCookGeneralAt(allocator: Allocator, evals: []const F, x: F) !F {
            const coeffs = try fromEvalsToom(allocator, evals);
            defer allocator.free(coeffs);

            // Evaluate using Horner's method: c0 + x*(c1 + x*(c2 + ... + x*c_d))
            var result = coeffs[coeffs.len - 1];
            var i = coeffs.len - 1;
            while (i > 0) {
                i -= 1;
                result = result.mul(x).add(coeffs[i]);
            }
            return result;
        }

        /// Vandermonde interpolation: given evaluations at [0, 1, ..., n-1], compute
        /// polynomial coefficients [c0, c1, ..., c_{n-1}] using Gaussian elimination.
        ///
        /// This matches Jolt's UniPoly::vandermonde_interpolation / from_evals.
        pub fn fromEvalsVandermonde(allocator: Allocator, evals: []const F) ![]F {
            const n = evals.len;
            if (n == 0) return try allocator.alloc(F, 0);

            // Build augmented Vandermonde matrix [V | evals]
            // Matches Jolt's gaussian_elimination exactly
            var matrix = try allocator.alloc([]F, n);
            errdefer {
                for (matrix) |row| allocator.free(row);
                allocator.free(matrix);
            }

            for (0..n) |i| {
                matrix[i] = try allocator.alloc(F, n + 1);
                matrix[i][0] = F.one();
                const x = F.fromU64(@intCast(i));
                for (1..n) |j| {
                    matrix[i][j] = matrix[i][j - 1].mul(x);
                }
                matrix[i][n] = evals[i]; // RHS
            }

            // Forward elimination (echelon) - matches Jolt's echelon function
            for (0..n - 1) |i| {
                for (i..n - 1) |j| {
                    // echelon(matrix, i, j): eliminate matrix[j+1][i]
                    if (!matrix[i][i].eql(F.zero())) {
                        const factor = matrix[j + 1][i].mul(matrix[i][i].inverse().?);
                        for (i..n + 1) |k| {
                            const tmp = matrix[i][k];
                            matrix[j + 1][k] = matrix[j + 1][k].sub(factor.mul(tmp));
                        }
                    }
                }
            }

            // Backward elimination - matches Jolt's eliminate function
            {
                var i_rev = n;
                while (i_rev > 1) {
                    i_rev -= 1;
                    // eliminate(matrix, i_rev)
                    if (!matrix[i_rev][i_rev].eql(F.zero())) {
                        var j = i_rev;
                        while (j > 0) {
                            j -= 1;
                            const factor = matrix[j][i_rev].mul(matrix[i_rev][i_rev].inverse().?);
                            var k = n;
                            while (k > 0) {
                                const tmp = matrix[i_rev][k];
                                matrix[j][k] = matrix[j][k].sub(factor.mul(tmp));
                                if (k == 0) break;
                                k -= 1;
                            }
                        }
                    }
                }
            }

            // Extract result: result[i] = matrix[i][n] / matrix[i][i]
            const coeffs = try allocator.alloc(F, n);
            for (0..n) |i| {
                if (!matrix[i][i].eql(F.zero())) {
                    coeffs[i] = matrix[i][n].mul(matrix[i][i].inverse().?);
                } else {
                    coeffs[i] = F.zero();
                }
            }

            for (matrix) |row| allocator.free(row);
            allocator.free(matrix);

            return coeffs;
        }

        /// Convert Vandermonde evaluations [p(0), p(1), ..., p(d)] to compressed format [c0, c2, c3, ..., c_d]
        ///
        /// For a degree-d polynomial, given d+1 evaluations at consecutive integer points,
        /// this computes all coefficients via Vandermonde interpolation, then returns the
        /// compressed format: all coefficients except c1 (linear term).
        ///
        /// The compressed format has d elements: [c0, c2, c3, ..., c_d]
        /// The verifier recovers c1 = hint - 2*c0 - c2 - c3 - ... - c_d
        pub fn vandermondeToCompressed(allocator: Allocator, evals: []const F) ![]F {
            const n = evals.len; // n = d + 1 (number of evaluation points)
            if (n == 0) return try allocator.alloc(F, 0);

            // Fast path for small n using closed-form finite differences (no Gaussian elimination)
            if (n <= 4) {
                const inv2 = INV2;
                const compressed = try allocator.alloc(F, n - 1);
                compressed[0] = evals[0]; // c0 = p(0)
                if (n == 2) {
                    // degree 1: c0 = p(0), skip c1
                    return compressed;
                } else if (n == 3) {
                    // degree 2: c2 = (p(2) - 2p(1) + p(0)) / 2
                    compressed[1] = evals[2].sub(evals[1]).sub(evals[1]).add(evals[0]).mul(inv2);
                    return compressed;
                } else {
                    // degree 3: finite differences
                    const inv6 = INV6;
                    const d1 = evals[1].sub(evals[0]);
                    const d2 = evals[2].sub(evals[1]);
                    const d3 = evals[3].sub(evals[2]);
                    const dd1 = d2.sub(d1);
                    const dd2 = d3.sub(d2);
                    const c3 = dd2.sub(dd1).mul(inv6);
                    const c2 = dd1.mul(inv2).sub(c3.mul(F.fromU64(3)));
                    compressed[1] = c2;
                    compressed[2] = c3;
                    return compressed;
                }
            }

            const coeffs = try fromEvalsVandermonde(allocator, evals);
            defer allocator.free(coeffs);

            // Compressed format: [c0, c2, c3, ..., c_d] (skip c1)
            const compressed = try allocator.alloc(F, n - 1);
            compressed[0] = coeffs[0]; // c0
            for (1..n - 1) |i| {
                compressed[i] = coeffs[i + 1]; // c2, c3, ..., c_d
            }
            return compressed;
        }

        /// Evaluate a general-degree polynomial at a point given Vandermonde evaluations
        ///
        /// Given [p(0), p(1), ..., p(d)], interpolates coefficients and evaluates at x.
        pub fn evaluateVandermondeAt(allocator: Allocator, evals: []const F, x: F) !F {
            const coeffs = try fromEvalsVandermonde(allocator, evals);
            defer allocator.free(coeffs);

            // Evaluate using Horner's method: c0 + x*(c1 + x*(c2 + ... + x*c_d))
            var result = coeffs[coeffs.len - 1];
            var i = coeffs.len - 1;
            while (i > 0) {
                i -= 1;
                result = result.mul(x).add(coeffs[i]);
            }
            return result;
        }

        /// Evaluate cubic from compressed form [c0, c2, c3] and hint = p(0)+p(1).
        /// Recovers c1 = hint - 2*c0 - c2 - c3, then Horner evaluation.
        /// 4 subs + 3 muls. No allocation.
        pub fn evalFromHint(compressed: [3]F, hint: F, x: F) F {
            const c0 = compressed[0];
            const c2 = compressed[1];
            const c3 = compressed[2];
            // c1 = hint - 2*c0 - c2 - c3
            const c1 = hint.sub(c0).sub(c0).sub(c2).sub(c3);
            // Horner: c0 + x*(c1 + x*(c2 + x*c3))
            return c0.add(x.mul(c1.add(x.mul(c2.add(x.mul(c3))))));
        }

        /// Evaluate from compressed form [c0, c2, c3, ..., c_d] and hint = p(0)+p(1).
        /// Recovers c1 = hint - 2*c0 - Σ compressed[1..], then Horner. No allocation.
        pub fn evalFromHintGeneral(compressed: []const F, hint: F, x: F) F {
            // Match Jolt's eval_from_hint exactly:
            // linear_term = hint - 2*c0 - c2 - c3 - ... - c_d
            // result = c0 + linear_term*x + c2*x^2 + c3*x^3 + ... + c_d*x^d
            const c0 = compressed[0];
            var linear_term = hint.sub(c0).sub(c0);
            for (compressed[1..]) |ci| {
                linear_term = linear_term.sub(ci);
            }

            var running_point = x;
            var running_sum = c0.add(x.mul(linear_term));
            for (compressed[1..]) |ci| {
                running_point = running_point.mul(x);
                running_sum = running_sum.add(ci.mul(running_point));
            }
            return running_sum;
        }

        /// Evaluate degree-2 poly at x from Vandermonde evals [p(0), p(1), p(2)].
        /// Uses Newton forward differences: 6 muls, no allocation.
        pub fn evalFromEvalsDeg2(evals: [3]F, x: F) F {
            const inv2 = INV2;
            // Newton forward differences for points 0, 1, 2:
            // p(x) = p(0) + Δ₁·x + Δ₂·x·(x-1)/2
            // where Δ₁ = p(1)-p(0), Δ₂ = p(2)-2p(1)+p(0)
            const d1 = evals[1].sub(evals[0]);
            const dd = evals[2].sub(evals[1]).sub(d1); // second difference
            // p(x) = p(0) + x·(Δ₁ + (x-1)·Δ₂/2)
            return evals[0].add(x.mul(d1.add(x.sub(F.one()).mul(dd).mul(inv2))));
        }

        /// Evaluate degree-3 poly at x from Vandermonde evals [p(0), p(1), p(2), p(3)].
        /// Uses Newton forward differences: ~8 muls, no allocation.
        pub fn evalFromEvalsDeg3(evals: [4]F, x: F) F {
            const inv2 = INV2;
            const inv6 = INV6;
            // Newton forward differences for points 0, 1, 2, 3:
            const d1 = evals[1].sub(evals[0]);
            const d2 = evals[2].sub(evals[1]);
            const d3 = evals[3].sub(evals[2]);
            const dd1 = d2.sub(d1);
            const dd2 = d3.sub(d2);
            const ddd = dd2.sub(dd1);
            // p(x) = p(0) + x·Δ₁ + x(x-1)/2·Δ₂ + x(x-1)(x-2)/6·Δ₃
            const xm1 = x.sub(F.one());
            const xm2 = x.sub(F.fromU64(2));
            return evals[0].add(x.mul(d1.add(xm1.mul(dd1.mul(inv2).add(xm2.mul(ddd).mul(inv6))))));
        }

        /// Evaluate general-degree poly at x from Vandermonde evals [p(0), ..., p(d)].
        /// Uses Newton forward differences. No allocation for d <= 15.
        pub fn evalFromEvalsGeneral(evals: []const F, x: F) F {
            const n = evals.len;
            if (n == 0) return F.zero();
            if (n == 1) return evals[0];
            if (n == 3) return evalFromEvalsDeg2(.{ evals[0], evals[1], evals[2] }, x);
            if (n == 4) return evalFromEvalsDeg3(.{ evals[0], evals[1], evals[2], evals[3] }, x);

            // General Newton forward differences with static buffer
            var dd: [16]F = undefined; // supports up to degree 15
            std.debug.assert(n <= 16);
            for (0..n) |i| dd[i] = evals[i];

            // Build forward difference table in-place
            var order: usize = 1;
            while (order < n) : (order += 1) {
                var i = n - 1;
                while (i >= order) : (i -= 1) {
                    dd[i] = dd[i].sub(dd[i - 1]);
                    if (i == order) break;
                }
            }
            // dd[k] now holds Δ^k[0] (k-th forward difference at 0)
            // Evaluate: p(x) = Σ_k C(x,k) · Δ^k[0]
            // where C(x,k) = x(x-1)...(x-k+1) / k!
            var result = dd[0];
            var falling_factorial = F.one();
            var k_factorial: u64 = 1;
            for (1..n) |k| {
                falling_factorial = falling_factorial.mul(x.sub(F.fromU64(@intCast(k - 1))));
                k_factorial *= @as(u64, @intCast(k));
                result = result.add(falling_factorial.mul(F.fromU64(k_factorial).inverse().?).mul(dd[k]));
            }
            return result;
        }
    };
}

// =============================================================================
// Tests
// =============================================================================

test "unipoly interpolate degree 3" {
    const field = @import("../field/mod.zig");
    const F = field.BN254Scalar;
    const Interp = Interpolation(F);

    // Test: p(X) = 1 + 2X + 3X² + 4X³
    // p(0) = 1
    // p(1) = 1 + 2 + 3 + 4 = 10
    // p(2) = 1 + 4 + 12 + 32 = 49
    // p(3) = 1 + 6 + 27 + 108 = 142
    const evals = [4]F{
        F.fromU64(1),
        F.fromU64(10),
        F.fromU64(49),
        F.fromU64(142),
    };

    const coeffs = Interp.interpolateDegree3(evals);

    try std.testing.expect(coeffs[0].eql(F.fromU64(1)));
    try std.testing.expect(coeffs[1].eql(F.fromU64(2)));
    try std.testing.expect(coeffs[2].eql(F.fromU64(3)));
    try std.testing.expect(coeffs[3].eql(F.fromU64(4)));
}

test "unipoly compressed format" {
    const field = @import("../field/mod.zig");
    const F = field.BN254Scalar;
    const Interp = Interpolation(F);

    // Same polynomial: p(X) = 1 + 2X + 3X² + 4X³
    const evals = [4]F{
        F.fromU64(1),
        F.fromU64(10),
        F.fromU64(49),
        F.fromU64(142),
    };

    const compressed = Interp.evalsToCompressed(evals);

    // Should be [c0, c2, c3] = [1, 3, 4]
    try std.testing.expect(compressed[0].eql(F.fromU64(1)));
    try std.testing.expect(compressed[1].eql(F.fromU64(3)));
    try std.testing.expect(compressed[2].eql(F.fromU64(4)));
}
