//! Streaming Outer Sumcheck Prover for Jolt Compatibility
//!
//! This module implements the full Spartan outer sumcheck prover matching
//! Jolt's implementation. The key innovation is using multiquadratic
//! polynomial representation for memory-efficient streaming evaluation.
//!
//! ## Protocol Overview
//!
//! The outer sumcheck proves:
//!   Σ_{x ∈ {0,1}^n} L(τ_high, x_uniskip) * eq(τ, x) * Az(x) * Bz(x) = 0
//!
//! Where:
//! - L(τ_high, x_uniskip) is the Lagrange polynomial for univariate skip (first round)
//! - eq(τ, x) is factored as eq(τ_out, x_out) * eq(τ_in, x_in)
//! - Az(x), Bz(x) are R1CS matrix products
//!
//! ## Rounds
//!
//! - Round 0: Univariate skip (degree 27 for domain size 10)
//! - Rounds 1..n: Streaming sumcheck (degree 3)
//!
//! ## Reference
//!
//! jolt-core/src/zkvm/spartan/outer.rs

const std = @import("std");
const Allocator = std.mem.Allocator;

const constraints = @import("../r1cs/constraints.zig");
const univariate_skip = @import("../r1cs/univariate_skip.zig");
const jolt_types = @import("../jolt_types.zig");
const poly_mod = @import("../../poly/mod.zig");
const GruenSplitEqPolynomial = poly_mod.GruenSplitEqPolynomial;
const MultiquadraticPolynomial = poly_mod.MultiquadraticPolynomial;
const utils = @import("../../utils/mod.zig");
const ExpandingTable = utils.ExpandingTable;

/// Streaming outer sumcheck prover for Jolt compatibility
pub fn StreamingOuterProver(comptime F: type) type {
    return struct {
        const Self = @This();

        /// Configuration constants from Jolt
        pub const NUM_CONSTRAINTS: usize = univariate_skip.NUM_R1CS_CONSTRAINTS;
        pub const FIRST_GROUP_SIZE: usize = univariate_skip.OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE;
        pub const SECOND_GROUP_SIZE: usize = NUM_CONSTRAINTS - FIRST_GROUP_SIZE;
        pub const UNISKIP_DEGREE: usize = univariate_skip.OUTER_UNIVARIATE_SKIP_DEGREE;
        pub const FIRST_ROUND_NUM_COEFFS: usize = univariate_skip.OUTER_FIRST_ROUND_POLY_NUM_COEFFS;
        pub const REMAINING_DEGREE: usize = 3;

        /// Per-cycle R1CS witnesses
        cycle_witnesses: []const constraints.R1CSCycleInputs(F),
        /// Number of cycle variables (log2 of trace length)
        num_cycle_vars: usize,
        /// Padded trace length (power of 2)
        padded_trace_len: usize,

        /// Split eq polynomial for efficient factored evaluation
        split_eq: GruenSplitEqPolynomial(F),

        /// Current sumcheck claim
        current_claim: F,
        /// Collected challenges
        challenges: std.ArrayListUnmanaged(F),
        /// Current round number
        current_round: usize,

        /// Precomputed Lagrange basis evaluations at first-round challenge r0
        /// Used for remaining rounds
        lagrange_evals_r0: [FIRST_GROUP_SIZE]F,

        /// Bound r_stream value (set after streaming round)
        /// Used to combine constraint groups in subsequent rounds
        r_stream: ?F,

        /// Expanding table for bound challenge weights
        /// eq(r_1, b_1) * eq(r_2, b_2) * ... for each cycle's binary representation
        /// Matches Jolt's r_grid in OuterSharedState
        r_grid: ExpandingTable(F),

        /// Allocator
        allocator: Allocator,

        /// Initialize the streaming outer prover (without scaling)
        ///
        /// tau_low: Challenge vector of length (num_cycle_vars + 1)
        ///          - tau_low[0..num_cycle_vars]: cycle variable challenges
        ///          - tau_low[num_cycle_vars]: streaming round challenge
        ///
        /// Note: This is tau_low = tau[..tau.len()-1], excluding tau_high.
        /// tau_high is handled by the UniSkip Lagrange kernel.
        pub fn init(
            allocator: Allocator,
            cycle_witnesses: []const constraints.R1CSCycleInputs(F),
            tau_low: []const F,
        ) !Self {
            return initWithScaling(allocator, cycle_witnesses, tau_low, null);
        }

        /// Initialize the streaming outer prover with Lagrange kernel scaling
        ///
        /// tau_low: Challenge vector of length (num_cycle_vars + 1)
        ///          - tau_low[0..num_cycle_vars]: cycle variable challenges
        ///          - tau_low[num_cycle_vars]: streaming round challenge
        ///
        /// lagrange_tau_r0: The Lagrange kernel L(r0, tau_high) from UniSkip
        ///                  This is multiplied into all eq evaluations.
        ///
        /// In Jolt, the full eq factorization is:
        ///   eq(tau, r) = L(tau_high, r0) * eq(tau_low, r_tail)
        ///
        /// where r = (r0, r_tail) and tau = (tau_low, tau_high).
        pub fn initWithScaling(
            allocator: Allocator,
            cycle_witnesses: []const constraints.R1CSCycleInputs(F),
            tau_low: []const F,
            lagrange_tau_r0: ?F,
        ) !Self {
            const num_cycles = cycle_witnesses.len;
            if (num_cycles == 0) {
                return error.EmptyTrace;
            }

            // Pad to next power of 2
            const padded_len = nextPowerOfTwo(num_cycles);
            const num_cycle_vars = std.math.log2_int(usize, padded_len);

            // tau_low should have length = num_cycle_vars + 1
            // - First num_cycle_vars elements are cycle variable challenges
            // - Last element is the streaming round challenge
            const num_x_in = 1; // One bit for constraint group selector (streaming round)
            const split_eq = try GruenSplitEqPolynomial(F).initWithScaling(allocator, tau_low, num_x_in, lagrange_tau_r0);

            // Initialize r_grid for tracking bound challenge weights
            // Capacity = padded_len (maximum number of cycles)
            var r_grid = try ExpandingTable(F).init(allocator, padded_len, .LowToHigh);
            r_grid.reset(F.one());

            return Self{
                .cycle_witnesses = cycle_witnesses,
                .num_cycle_vars = num_cycle_vars,
                .padded_trace_len = padded_len,
                .split_eq = split_eq,
                .current_claim = F.zero(),
                .challenges = .{},
                .current_round = 0,
                .lagrange_evals_r0 = [_]F{F.zero()} ** FIRST_GROUP_SIZE,
                .r_stream = null,
                .r_grid = r_grid,
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            self.split_eq.deinit();
            self.challenges.deinit(self.allocator);
            self.r_grid.deinit();
        }

        /// Total number of rounds for the remaining sumcheck (after UniSkip)
        /// = 1 (streaming/constraint group) + num_cycle_vars (cycle bits)
        ///
        /// Note: This does NOT include the UniSkip round, which is handled separately.
        /// The streaming round binds the constraint group selector variable.
        /// The cycle rounds bind the cycle index bits.
        pub fn numRounds(self: *const Self) usize {
            return 1 + self.num_cycle_vars;
        }

        /// Compute the first-round univariate skip polynomial
        ///
        /// This is a degree-27 polynomial over the extended domain.
        /// It computes:
        ///   s₁(Y) = L(τ_high, Y) * Σ_{x_out, x_in} eq(τ, x) * Az(x, Y) * Bz(x, Y)
        ///
        /// Returns coefficients for the degree-27 polynomial
        pub fn computeFirstRoundPoly(self: *Self) ![FIRST_ROUND_NUM_COEFFS]F {
            // For each point in the extended domain, compute the sum over all cycles
            var extended_evals: [univariate_skip.OUTER_UNIVARIATE_SKIP_EXTENDED_DOMAIN_SIZE]F = undefined;

            // Get eq evaluations for cycles
            const eq_table = try self.split_eq.getFullEqTable(self.allocator);
            defer self.allocator.free(eq_table);

            // Evaluate at each domain point
            for (0..univariate_skip.OUTER_UNIVARIATE_SKIP_EXTENDED_DOMAIN_SIZE) |domain_idx| {
                var sum = F.zero();

                // Sum over all cycles
                for (0..@min(self.cycle_witnesses.len, self.padded_trace_len)) |cycle| {
                    const eq_val = if (cycle < eq_table.len) eq_table[cycle] else F.zero();

                    // Get Az and Bz for this cycle
                    if (cycle < self.cycle_witnesses.len) {
                        const witness = &self.cycle_witnesses[cycle];

                        // Evaluate Az * Bz at this domain point
                        // Domain point maps to constraint index
                        const az_bz = self.evaluateAzBzAtDomainPoint(witness, domain_idx);
                        sum = sum.add(eq_val.mul(az_bz));
                    }
                }

                extended_evals[domain_idx] = sum;
            }

            // Multiply by Lagrange kernel and interpolate to get coefficients
            return self.interpolateFirstRoundPoly(&extended_evals);
        }

        /// Evaluate Az * Bz for a single cycle at a specific domain point Y
        ///
        /// The domain points are in the extended symmetric window:
        /// - Index 0: Y = -DEGREE (= -9)
        /// - Index 9: Y = 0
        /// - Index 18: Y = DEGREE (= 9)
        ///
        /// For Y in the base window {-4, -3, ..., 4, 5}, we can directly evaluate
        /// the constraint at that index.
        ///
        /// For Y outside the base window, we use Lagrange extrapolation:
        /// - Evaluate Az and Bz at base window points
        /// - Extrapolate to Y using precomputed Lagrange coefficients
        /// - Multiply extrapolated values
        fn evaluateAzBzAtDomainPoint(
            self: *const Self,
            witness: *const constraints.R1CSCycleInputs(F),
            domain_idx: usize,
        ) F {
            _ = self;

            // Convert domain_idx to the actual Y coordinate
            // domain_idx 0 -> Y = -DEGREE = -9
            // domain_idx DEGREE -> Y = 0
            // domain_idx 2*DEGREE -> Y = DEGREE = 9
            const DEGREE = univariate_skip.OUTER_UNIVARIATE_SKIP_DEGREE;
            const y_coord: i64 = @as(i64, @intCast(domain_idx)) - @as(i64, DEGREE);

            // Base window is {-4, -3, -2, -1, 0, 1, 2, 3, 4, 5}
            // which corresponds to base_left = -4, base_right = 5
            const base_left: i64 = -@as(i64, (FIRST_GROUP_SIZE - 1) / 2);
            const base_right: i64 = base_left + @as(i64, FIRST_GROUP_SIZE) - 1;

            // Check if Y is in the base window
            if (y_coord >= base_left and y_coord <= base_right) {
                // Y is in the base window - evaluate constraint directly
                // Map Y to constraint index: Y = base_left + i => i = Y - base_left
                const constraint_pos: usize = @intCast(y_coord - base_left);

                // For a valid R1CS, Az*Bz should be ZERO at base window points
                // because the constraints are satisfied
                const constraint_idx = constraints.FIRST_GROUP_INDICES[constraint_pos];
                const constraint = constraints.UNIFORM_CONSTRAINTS[constraint_idx];
                const az = constraint.condition.evaluate(F, witness.asSlice());
                const bz = constraint.left.evaluate(F, witness.asSlice())
                    .sub(constraint.right.evaluate(F, witness.asSlice()));
                return az.mul(bz);
            }

            // Y is outside the base window - use Lagrange extrapolation
            // Compute Az and Bz at all base window points
            var az_base: [FIRST_GROUP_SIZE]F = undefined;
            var bz_base: [FIRST_GROUP_SIZE]F = undefined;

            for (0..FIRST_GROUP_SIZE) |i| {
                const constraint_idx = constraints.FIRST_GROUP_INDICES[i];
                const constraint = constraints.UNIFORM_CONSTRAINTS[constraint_idx];
                az_base[i] = constraint.condition.evaluate(F, witness.asSlice());
                bz_base[i] = constraint.left.evaluate(F, witness.asSlice())
                    .sub(constraint.right.evaluate(F, witness.asSlice()));
            }

            // Use precomputed Lagrange coefficients to extrapolate to Y
            // Find the target index j corresponding to Y
            const targets = univariate_skip.UNISKIP_TARGETS;
            var target_j: ?usize = null;
            for (targets, 0..) |t, j| {
                if (t == y_coord) {
                    target_j = j;
                    break;
                }
            }

            if (target_j) |j| {
                // Use COEFFS_PER_J[j] for extrapolation
                const coeffs = univariate_skip.COEFFS_PER_J[j];

                // Extrapolate Az(Y) = Σ_i coeffs[i] * az_base[i]
                var az_y = F.zero();
                for (0..FIRST_GROUP_SIZE) |i| {
                    const c = coeffs[i];
                    if (c != 0) {
                        const c_field = if (c > 0)
                            F.fromU64(@intCast(c))
                        else
                            F.zero().sub(F.fromU64(@intCast(-c)));
                        az_y = az_y.add(az_base[i].mul(c_field));
                    }
                }

                // Extrapolate Bz(Y) = Σ_i coeffs[i] * bz_base[i]
                var bz_y = F.zero();
                for (0..FIRST_GROUP_SIZE) |i| {
                    const c = coeffs[i];
                    if (c != 0) {
                        const c_field = if (c > 0)
                            F.fromU64(@intCast(c))
                        else
                            F.zero().sub(F.fromU64(@intCast(-c)));
                        bz_y = bz_y.add(bz_base[i].mul(c_field));
                    }
                }

                return az_y.mul(bz_y);
            }

            // Should not reach here for valid domain indices
            return F.zero();
        }

        /// Interpolate extended evaluations to polynomial coefficients
        ///
        /// This implements Jolt's `build_uniskip_first_round_poly`:
        /// 1. Rebuild t1 on the full extended symmetric window
        /// 2. Interpolate t1 to get coefficients
        /// 3. Compute Lagrange kernel L(τ_high, Y) coefficients
        /// 4. Multiply polynomials to get s1(Y) = L(τ_high, Y) * t1(Y)
        fn interpolateFirstRoundPoly(
            self: *const Self,
            extended_evals: *const [univariate_skip.OUTER_UNIVARIATE_SKIP_EXTENDED_DOMAIN_SIZE]F,
        ) [FIRST_ROUND_NUM_COEFFS]F {
            const DOMAIN_SIZE = FIRST_GROUP_SIZE;
            const DEGREE = univariate_skip.OUTER_UNIVARIATE_SKIP_DEGREE;
            const EXTENDED_SIZE = univariate_skip.OUTER_UNIVARIATE_SKIP_EXTENDED_DOMAIN_SIZE;

            // The extended_evals are evaluations of t1(Y) on the extended symmetric window
            // {-DEGREE, ..., DEGREE} = {-9, -8, ..., 8, 9}
            // These 19 values are provided in order from extended_evals[0] = t1(-9) to extended_evals[18] = t1(9)

            // Step 1: Interpolate t1(Y) from extended evaluations
            // Domain points: {-DEGREE, -DEGREE+1, ..., DEGREE-1, DEGREE}
            var t1_coeffs: [EXTENDED_SIZE]F = [_]F{F.zero()} ** EXTENDED_SIZE;

            // Use Lagrange interpolation: p(Y) = Σ_i y_i * L_i(Y)
            // where L_i(Y) = Π_{j≠i} (Y - x_j) / (x_i - x_j)
            for (0..EXTENDED_SIZE) |i| {
                // Evaluation y_i at domain point x_i = -DEGREE + i
                const y_i = extended_evals[i];

                if (y_i.eql(F.zero())) continue;

                // Compute denominator (scalar) Π_{j≠i} (x_i - x_j)
                // Since x_i = -DEGREE + i and x_j = -DEGREE + j, we have x_i - x_j = i - j
                var den = F.one();
                for (0..EXTENDED_SIZE) |j| {
                    if (i == j) continue;
                    // x_i - x_j = (i - j)
                    const diff: i64 = @as(i64, @intCast(i)) - @as(i64, @intCast(j));
                    const diff_field = if (diff >= 0)
                        F.fromU64(@intCast(diff))
                    else
                        F.zero().sub(F.fromU64(@intCast(-diff)));
                    den = den.mul(diff_field);
                }

                const scale = y_i.mul(den.inverse().?);

                // Build numerator polynomial Π_{j≠i} (Y - x_j)
                // Start with constant 1, multiply by (Y - x_j) for each j ≠ i
                var basis: [EXTENDED_SIZE]F = [_]F{F.zero()} ** EXTENDED_SIZE;
                basis[0] = F.one();
                var deg: usize = 0;

                for (0..EXTENDED_SIZE) |j| {
                    if (i == j) continue;
                    const x_j: i64 = -@as(i64, DEGREE) + @as(i64, @intCast(j));
                    const neg_x_j = if (x_j >= 0)
                        F.zero().sub(F.fromU64(@intCast(x_j)))
                    else
                        F.fromU64(@intCast(-x_j));

                    // Multiply basis by (Y - x_j)
                    // New polynomial: basis[k+1] += basis[k] and basis[k] *= neg_x_j
                    var k: usize = deg + 1;
                    while (k > 0) {
                        k -= 1;
                        const old = basis[k];
                        if (k + 1 <= EXTENDED_SIZE - 1) {
                            basis[k + 1] = basis[k + 1].add(old);
                        }
                        basis[k] = old.mul(neg_x_j);
                    }
                    deg += 1;
                }

                // Add scaled basis to t1_coeffs
                for (0..EXTENDED_SIZE) |k| {
                    t1_coeffs[k] = t1_coeffs[k].add(basis[k].mul(scale));
                }
            }

            // Step 2: Compute Lagrange kernel L(τ_high, Y) coefficients
            // τ_high is the last element of tau, which is the high bit challenge
            const tau_high = self.split_eq.getTauHigh();

            // L(τ_high, Y) evaluations at base domain {-4, -3, ..., 4, 5}
            var lagrange_evals: [DOMAIN_SIZE]F = undefined;
            const base_left: i64 = -@as(i64, (DOMAIN_SIZE - 1) / 2);

            for (0..DOMAIN_SIZE) |i| {
                const x_i: i64 = base_left + @as(i64, @intCast(i));
                var num = F.one();
                var den = F.one();

                for (0..DOMAIN_SIZE) |j| {
                    if (i == j) continue;
                    const x_j: i64 = base_left + @as(i64, @intCast(j));
                    const x_j_field = if (x_j >= 0) F.fromU64(@intCast(x_j)) else F.zero().sub(F.fromU64(@intCast(-x_j)));
                    num = num.mul(tau_high.sub(x_j_field));

                    const diff: i64 = x_i - x_j;
                    const diff_field = if (diff >= 0) F.fromU64(@intCast(diff)) else F.zero().sub(F.fromU64(@intCast(-diff)));
                    den = den.mul(diff_field);
                }

                lagrange_evals[i] = num.mul(den.inverse().?);
            }

            // Interpolate Lagrange kernel to coefficients (degree DOMAIN_SIZE-1 = 9)
            var lagrange_coeffs: [DOMAIN_SIZE]F = [_]F{F.zero()} ** DOMAIN_SIZE;

            for (0..DOMAIN_SIZE) |i| {
                const y_i = lagrange_evals[i];
                if (y_i.eql(F.zero())) continue;

                var den = F.one();
                for (0..DOMAIN_SIZE) |j| {
                    if (i == j) continue;
                    const diff: i64 = @as(i64, @intCast(i)) - @as(i64, @intCast(j));
                    const diff_field = if (diff >= 0) F.fromU64(@intCast(diff)) else F.zero().sub(F.fromU64(@intCast(-diff)));
                    den = den.mul(diff_field);
                }
                const scale = y_i.mul(den.inverse().?);

                var basis: [DOMAIN_SIZE]F = [_]F{F.zero()} ** DOMAIN_SIZE;
                basis[0] = F.one();
                var deg: usize = 0;

                for (0..DOMAIN_SIZE) |j| {
                    if (i == j) continue;
                    const x_j: i64 = base_left + @as(i64, @intCast(j));
                    const neg_x_j = if (x_j >= 0) F.zero().sub(F.fromU64(@intCast(x_j))) else F.fromU64(@intCast(-x_j));

                    var k: usize = deg + 1;
                    while (k > 0) {
                        k -= 1;
                        const old = basis[k];
                        if (k + 1 < DOMAIN_SIZE) {
                            basis[k + 1] = basis[k + 1].add(old);
                        }
                        basis[k] = old.mul(neg_x_j);
                    }
                    deg += 1;
                }

                for (0..DOMAIN_SIZE) |k| {
                    lagrange_coeffs[k] = lagrange_coeffs[k].add(basis[k].mul(scale));
                }
            }

            // Step 3: Multiply polynomials s1 = L * t1
            // deg(L) = DOMAIN_SIZE - 1 = 9
            // deg(t1) = EXTENDED_SIZE - 1 = 18
            // deg(s1) = 9 + 18 = 27
            var s1_coeffs: [FIRST_ROUND_NUM_COEFFS]F = [_]F{F.zero()} ** FIRST_ROUND_NUM_COEFFS;

            for (0..DOMAIN_SIZE) |i| {
                for (0..EXTENDED_SIZE) |j| {
                    if (i + j < FIRST_ROUND_NUM_COEFFS) {
                        s1_coeffs[i + j] = s1_coeffs[i + j].add(lagrange_coeffs[i].mul(t1_coeffs[j]));
                    }
                }
            }

            return s1_coeffs;
        }

        /// Bind the first-round challenge and set up for remaining rounds
        ///
        /// The uni_skip_claim parameter is uni_poly(r0), the evaluation of the
        /// univariate skip polynomial at the first-round challenge.
        ///
        /// IMPORTANT: r0 is NOT bound in split_eq! In Jolt, r0's contribution is:
        /// 1. Pre-multiplied into current_scalar via L(tau_high, r0) at initialization
        /// 2. Used to compute Lagrange weights for Az/Bz evaluation
        /// The split_eq only binds the streaming round and cycle round challenges.
        pub fn bindFirstRoundChallenge(self: *Self, r0: F, uni_skip_claim: F) !void {
            // Note: r0 is added to challenges for transcript consistency, but NOT bound in split_eq
            try self.challenges.append(self.allocator, r0);
            self.current_round = 1;
            self.current_claim = uni_skip_claim;

            // Compute Lagrange basis evaluations at r0 for use in remaining rounds
            self.computeLagrangeEvalsAtR0(r0);

            // DO NOT bind r0 in split_eq! The Lagrange kernel scaling was already
            // applied during initialization. The streaming round will bind the first
            // actual sumcheck challenge.
        }

        /// Compute Lagrange basis evaluations at r0
        ///
        /// IMPORTANT: The domain is the symmetric window {-4, -3, -2, -1, 0, 1, 2, 3, 4, 5}
        /// matching Jolt's LagrangePolynomial::start_i64 which computes -(N-1)/2 = -4 for N=10.
        fn computeLagrangeEvalsAtR0(self: *Self, r0: F) void {
            // L_i(r0) for i in 0..FIRST_GROUP_SIZE
            // Domain is {start, start+1, ..., start+FIRST_GROUP_SIZE-1}
            // where start = -((FIRST_GROUP_SIZE - 1) / 2) = -4

            const start: i64 = -@as(i64, (FIRST_GROUP_SIZE - 1) / 2); // = -4

            for (0..FIRST_GROUP_SIZE) |i| {
                _ = start + @as(i64, @intCast(i)); // actual domain point (unused but shows semantics)
                var numer = F.one();
                var denom = F.one();

                for (0..FIRST_GROUP_SIZE) |j| {
                    if (i != j) {
                        const x_j: i64 = start + @as(i64, @intCast(j));

                        // numer *= (r0 - x_j)
                        const x_j_field = if (x_j >= 0)
                            F.fromU64(@intCast(x_j))
                        else
                            F.zero().sub(F.fromU64(@intCast(-x_j)));
                        numer = numer.mul(r0.sub(x_j_field));

                        // denom *= (x_i - x_j) = (i - j) since x_k = start + k
                        const diff: i64 = @as(i64, @intCast(i)) - @as(i64, @intCast(j));
                        if (diff > 0) {
                            denom = denom.mul(F.fromU64(@intCast(diff)));
                        } else {
                            denom = denom.mul(F.zero().sub(F.fromU64(@intCast(-diff))));
                        }
                    }
                }

                // L_i(r0) = numer / denom
                self.lagrange_evals_r0[i] = if (!denom.eql(F.zero()))
                    numer.mul(denom.inverse().?)
                else
                    F.zero();
            }
        }

        /// Compute a remaining round polynomial (degree 3)
        ///
        /// There are two types of rounds:
        /// 1. Streaming round (current_round == 1): Sums over constraint groups
        /// 2. Cycle rounds (current_round > 1): Sums over cycle halves using combined Az*Bz
        ///
        /// IMPORTANT: The eq weights for cycles use a FACTORIZED representation:
        ///   eq_val[i] = E_out[i >> head_in_bits] * E_in[i & ((1 << head_in_bits) - 1)]
        /// This allows us to handle 1024 cycles with only 32+32=64 precomputed values.
        pub fn computeRemainingRoundPoly(self: *Self) ![4]F {
            // Get eq tables for this window - uses Jolt's factorized split eq approach
            // The first parameter is ignored - split_eq uses current_index directly
            const eq_tables = self.split_eq.getWindowEqTables(0, 1);

            const E_out = eq_tables.E_out;
            const E_in = eq_tables.E_in;
            const head_in_bits = eq_tables.head_in_bits;
            const e_in_mask = (@as(usize, 1) << @intCast(head_in_bits)) - 1;

            // Gruen's multiquadratic method computes:
            // - t'(0) = Σ eq * Az(0) * Bz(0)
            // - t'(∞) = Σ eq * Az(∞) * Bz(∞) = Σ eq * (Az(1) - Az(0)) * (Bz(1) - Bz(0))
            //
            // Note: t'(∞) is the product of SLOPES, NOT the slope of the product!
            // This is crucial for the cubic polynomial construction.

            var t_zero = F.zero();
            var t_infinity = F.zero();

            if (self.current_round == 1) {
                // STREAMING ROUND: The streaming variable selects constraint group (0 or 1)
                // - Position 0: first constraint group (Az_g0, Bz_g0)
                // - Position 1: second constraint group (Az_g1, Bz_g1)
                // - Position ∞: slope = (value at 1) - (value at 0)
                //
                // t'(0) = Σ_cycles eq * Az_g0 * Bz_g0
                // t'(∞) = Σ_cycles eq * (Az_g1 - Az_g0) * (Bz_g1 - Bz_g0)  [product of slopes]
                //
                // The eq weights are factorized: eq[i] = E_out[i >> head_in_bits] * E_in[i & mask]
                // NOTE: Do NOT include current_scalar here - it will be applied in computeCubicRoundPoly.

                for (0..@min(self.padded_trace_len, self.cycle_witnesses.len)) |i| {
                    // Factorized eq weight using proper bit shifting
                    const out_idx = i >> @intCast(head_in_bits);
                    const in_idx = i & e_in_mask;
                    const e_out_val = if (out_idx < E_out.len) E_out[out_idx] else F.zero();
                    const e_in_val = if (in_idx < E_in.len) E_in[in_idx] else F.zero();
                    // No current_scalar - it will be applied when computing l(X) in computeCubicRoundPoly
                    const eq_val = e_out_val.mul(e_in_val);

                    // Compute Az and Bz separately for both groups to get slopes
                    const az_bz = self.computeCycleAzBzForMultiquadratic(&self.cycle_witnesses[i]);
                    t_zero = t_zero.add(eq_val.mul(az_bz.prod_0)); // Az_g0 * Bz_g0
                    t_infinity = t_infinity.add(eq_val.mul(az_bz.prod_inf)); // slope_a * slope_b
                }
            } else {
                // CYCLE ROUND: Match Jolt's linear phase structure exactly
                //
                // In Jolt, the full index space is:
                //   full_idx = x_out << (x_in_bits + window + r_bits) | x_in << (window + r_bits) | x_val << r_bits | r_idx
                //   step_idx = full_idx >> 1
                //   selector = full_idx & 1
                //
                // The key insight is that r_idx contributes to step_idx via (r_idx >> 1),
                // so different r_idx values map to the same or adjacent cycles.
                //
                // For window_size = 1:
                //   - x_val ∈ {0, 1} is the current cycle bit
                //   - r_idx ∈ {0, ..., r_grid.len-1} expands the index space
                //   - The LSB of full_idx selects the constraint group

                const r_stream = self.r_stream orelse F.zero();
                const r_grid = &self.r_grid;
                const r_grid_len = r_grid.length();
                const num_r_bits: u6 = if (r_grid_len > 1) @intCast(std.math.log2_int(usize, r_grid_len)) else 0;

                // window_size is always 1 for cycle rounds in linear phase
                const window_bits: u6 = 1;

                // Iterate over the factorized index space
                var x_out_idx: usize = 0;
                while (x_out_idx < E_out.len) : (x_out_idx += 1) {
                    const e_out_val = E_out[x_out_idx];

                    var x_in_idx: usize = 0;
                    while (x_in_idx < E_in.len) : (x_in_idx += 1) {
                        const e_in_val = E_in[x_in_idx];
                        const eq_base = e_out_val.mul(e_in_val);

                        // Accumulate Az/Bz for the multiquadratic grid
                        var az_grid = [2]F{ F.zero(), F.zero() };
                        var bz_grid = [2]F{ F.zero(), F.zero() };

                        // Compute base_idx following Jolt's structure
                        const base_idx: usize = (x_out_idx << @intCast(head_in_bits + window_bits + num_r_bits)) |
                            (x_in_idx << @intCast(window_bits + num_r_bits));

                        // Iterate over x_val (current window variable) and r_idx
                        var x_val: usize = 0;
                        while (x_val < 2) : (x_val += 1) {
                            const x_val_shifted = x_val << num_r_bits;

                            var r_idx: usize = 0;
                            while (r_idx < r_grid_len) : (r_idx += 1) {
                                const r_weight = r_grid.get(r_idx);

                                // Compute full_idx and derive step_idx
                                const full_idx = base_idx | x_val_shifted | r_idx;
                                const step_idx = full_idx >> 1;

                                // Get Az/Bz for this cycle (using r_stream-combined values)
                                if (step_idx < self.cycle_witnesses.len) {
                                    const result = self.computeCycleAzBzSeparate(&self.cycle_witnesses[step_idx], r_stream);
                                    // Weight by r_grid and accumulate to appropriate x_val slot
                                    az_grid[x_val] = az_grid[x_val].add(r_weight.mul(result.az));
                                    bz_grid[x_val] = bz_grid[x_val].add(r_weight.mul(result.bz));
                                }
                            }
                        }

                        // Multiquadratic expansion: t'(0) and t'(∞)
                        const prod_0 = az_grid[0].mul(bz_grid[0]);
                        const slope_az = az_grid[1].sub(az_grid[0]);
                        const slope_bz = bz_grid[1].sub(bz_grid[0]);
                        const prod_inf = slope_az.mul(slope_bz);

                        t_zero = t_zero.add(eq_base.mul(prod_0));
                        t_infinity = t_infinity.add(eq_base.mul(prod_inf));
                    }
                }
            }

            // Use Gruen's method to compute the cubic round polynomial
            const previous_claim = self.current_claim;
            const round_poly = self.split_eq.computeCubicRoundPoly(
                t_zero,
                t_infinity,
                previous_claim,
            );

            return round_poly;
        }

        /// Compute Az * Bz product for a single cycle (for given constraint group)
        ///
        /// Computation:
        /// Az = Σ_i L_i(r0) * condition_i(witness)
        /// Bz = Σ_i L_i(r0) * (left_i - right_i)(witness)
        /// Return Az * Bz
        ///
        /// This computes Az*Bz for a single constraint group. The streaming round
        /// uses both groups combined with r_stream.
        fn computeCycleAzBzProductForGroup(
            self: *const Self,
            witness: *const constraints.R1CSCycleInputs(F),
            group: usize, // 0 = first group, 1 = second group
        ) F {
            var az_sum = F.zero();
            var bz_sum = F.zero();

            const group_size = if (group == 0) FIRST_GROUP_SIZE else @min(SECOND_GROUP_SIZE, FIRST_GROUP_SIZE);
            const group_indices = if (group == 0) &constraints.FIRST_GROUP_INDICES else &constraints.SECOND_GROUP_INDICES;

            // Sum over group constraints weighted by Lagrange basis
            for (0..group_size) |i| {
                const constraint_idx = group_indices[i];
                const constraint = constraints.UNIFORM_CONSTRAINTS[constraint_idx];
                const condition = constraint.condition.evaluate(F, witness.asSlice());
                const left = constraint.left.evaluate(F, witness.asSlice());
                const right = constraint.right.evaluate(F, witness.asSlice());
                const magnitude = left.sub(right);

                // Weighted sum for Az (conditions)
                az_sum = az_sum.add(self.lagrange_evals_r0[i].mul(condition));

                // Weighted sum for Bz (magnitudes)
                bz_sum = bz_sum.add(self.lagrange_evals_r0[i].mul(magnitude));
            }

            // Return the PRODUCT of the sums
            return az_sum.mul(bz_sum);
        }

        /// Compute combined Az * Bz for a single cycle using bound r_stream value
        ///
        /// The formula is:
        /// Az_final = (1 - r_stream) * Az_g0 + r_stream * Az_g1
        /// Bz_final = (1 - r_stream) * Bz_g0 + r_stream * Bz_g1
        /// Return Az_final * Bz_final
        fn computeCycleAzBzProductCombined(
            self: *const Self,
            witness: *const constraints.R1CSCycleInputs(F),
            r_stream: F,
        ) F {
            // Compute Az and Bz for both groups
            var az_g0 = F.zero();
            var bz_g0 = F.zero();
            var az_g1 = F.zero();
            var bz_g1 = F.zero();

            // Group 0
            for (0..FIRST_GROUP_SIZE) |i| {
                const constraint_idx = constraints.FIRST_GROUP_INDICES[i];
                const constraint = constraints.UNIFORM_CONSTRAINTS[constraint_idx];
                const condition = constraint.condition.evaluate(F, witness.asSlice());
                const left = constraint.left.evaluate(F, witness.asSlice());
                const right = constraint.right.evaluate(F, witness.asSlice());
                const magnitude = left.sub(right);
                az_g0 = az_g0.add(self.lagrange_evals_r0[i].mul(condition));
                bz_g0 = bz_g0.add(self.lagrange_evals_r0[i].mul(magnitude));
            }

            // Group 1
            const g2_size = @min(SECOND_GROUP_SIZE, FIRST_GROUP_SIZE);
            for (0..g2_size) |i| {
                const constraint_idx = constraints.SECOND_GROUP_INDICES[i];
                const constraint = constraints.UNIFORM_CONSTRAINTS[constraint_idx];
                const condition = constraint.condition.evaluate(F, witness.asSlice());
                const left = constraint.left.evaluate(F, witness.asSlice());
                const right = constraint.right.evaluate(F, witness.asSlice());
                const magnitude = left.sub(right);
                az_g1 = az_g1.add(self.lagrange_evals_r0[i].mul(condition));
                bz_g1 = bz_g1.add(self.lagrange_evals_r0[i].mul(magnitude));
            }

            // Combine using r_stream: final = g0 + r_stream * (g1 - g0)
            const az_final = az_g0.add(r_stream.mul(az_g1.sub(az_g0)));
            const bz_final = bz_g0.add(r_stream.mul(bz_g1.sub(bz_g0)));

            return az_final.mul(bz_final);
        }

        /// Compute Az * Bz product for a single cycle (legacy, uses only group 0)
        /// This is kept for compatibility but should not be used for correct proofs.
        fn computeCycleAzBzProduct(self: *const Self, witness: *const constraints.R1CSCycleInputs(F)) F {
            return self.computeCycleAzBzProductForGroup(witness, 0);
        }

        /// Compute Az*Bz for multiquadratic expansion (streaming round)
        ///
        /// Returns values for the multiquadratic grid:
        /// - prod_0 = Az_g0 * Bz_g0 (product at position 0)
        /// - prod_inf = (Az_g1 - Az_g0) * (Bz_g1 - Bz_g0) (product of slopes)
        ///
        /// This is used in the streaming round where we select between constraint groups.
        fn computeCycleAzBzForMultiquadratic(
            self: *const Self,
            witness: *const constraints.R1CSCycleInputs(F),
        ) struct { prod_0: F, prod_inf: F } {
            // Compute Az and Bz for both groups
            var az_g0 = F.zero();
            var bz_g0 = F.zero();
            var az_g1 = F.zero();
            var bz_g1 = F.zero();

            // Group 0
            for (0..FIRST_GROUP_SIZE) |i| {
                const constraint_idx = constraints.FIRST_GROUP_INDICES[i];
                const constraint = constraints.UNIFORM_CONSTRAINTS[constraint_idx];
                const condition = constraint.condition.evaluate(F, witness.asSlice());
                const left = constraint.left.evaluate(F, witness.asSlice());
                const right = constraint.right.evaluate(F, witness.asSlice());
                const magnitude = left.sub(right);
                az_g0 = az_g0.add(self.lagrange_evals_r0[i].mul(condition));
                bz_g0 = bz_g0.add(self.lagrange_evals_r0[i].mul(magnitude));
            }

            // Group 1
            const g2_size = @min(SECOND_GROUP_SIZE, FIRST_GROUP_SIZE);
            for (0..g2_size) |i| {
                const constraint_idx = constraints.SECOND_GROUP_INDICES[i];
                const constraint = constraints.UNIFORM_CONSTRAINTS[constraint_idx];
                const condition = constraint.condition.evaluate(F, witness.asSlice());
                const left = constraint.left.evaluate(F, witness.asSlice());
                const right = constraint.right.evaluate(F, witness.asSlice());
                const magnitude = left.sub(right);
                az_g1 = az_g1.add(self.lagrange_evals_r0[i].mul(condition));
                bz_g1 = bz_g1.add(self.lagrange_evals_r0[i].mul(magnitude));
            }

            // Multiquadratic values:
            // prod_0 = Az_g0 * Bz_g0 (at position 0)
            // slope_az = Az_g1 - Az_g0
            // slope_bz = Bz_g1 - Bz_g0
            // prod_inf = slope_az * slope_bz (product of slopes)
            const prod_0 = az_g0.mul(bz_g0);
            const slope_az = az_g1.sub(az_g0);
            const slope_bz = bz_g1.sub(bz_g0);
            const prod_inf = slope_az.mul(slope_bz);

            return .{ .prod_0 = prod_0, .prod_inf = prod_inf };
        }

        /// Compute Az and Bz separately for a single cycle (combined groups)
        ///
        /// Returns (Az_final, Bz_final) where:
        /// Az_final = (1 - r_stream) * Az_g0 + r_stream * Az_g1
        /// Bz_final = (1 - r_stream) * Bz_g0 + r_stream * Bz_g1
        fn computeCycleAzBzSeparate(
            self: *const Self,
            witness: *const constraints.R1CSCycleInputs(F),
            r_stream: F,
        ) struct { az: F, bz: F } {
            // Compute Az and Bz for both groups
            var az_g0 = F.zero();
            var bz_g0 = F.zero();
            var az_g1 = F.zero();
            var bz_g1 = F.zero();

            // Group 0
            for (0..FIRST_GROUP_SIZE) |i| {
                const constraint_idx = constraints.FIRST_GROUP_INDICES[i];
                const constraint = constraints.UNIFORM_CONSTRAINTS[constraint_idx];
                const condition = constraint.condition.evaluate(F, witness.asSlice());
                const left = constraint.left.evaluate(F, witness.asSlice());
                const right = constraint.right.evaluate(F, witness.asSlice());
                const magnitude = left.sub(right);
                az_g0 = az_g0.add(self.lagrange_evals_r0[i].mul(condition));
                bz_g0 = bz_g0.add(self.lagrange_evals_r0[i].mul(magnitude));
            }

            // Group 1
            const g2_size = @min(SECOND_GROUP_SIZE, FIRST_GROUP_SIZE);
            for (0..g2_size) |i| {
                const constraint_idx = constraints.SECOND_GROUP_INDICES[i];
                const constraint = constraints.UNIFORM_CONSTRAINTS[constraint_idx];
                const condition = constraint.condition.evaluate(F, witness.asSlice());
                const left = constraint.left.evaluate(F, witness.asSlice());
                const right = constraint.right.evaluate(F, witness.asSlice());
                const magnitude = left.sub(right);
                az_g1 = az_g1.add(self.lagrange_evals_r0[i].mul(condition));
                bz_g1 = bz_g1.add(self.lagrange_evals_r0[i].mul(magnitude));
            }

            // Combine using r_stream: final = g0 + r_stream * (g1 - g0)
            const az_final = az_g0.add(r_stream.mul(az_g1.sub(az_g0)));
            const bz_final = bz_g0.add(r_stream.mul(bz_g1.sub(bz_g0)));

            return .{ .az = az_final, .bz = bz_final };
        }

        /// Compute remaining round polynomial using multiquadratic expansion
        ///
        /// This is the correct approach:
        /// 1. Compute Az and Bz grids separately for each cycle
        /// 2. Expand each to multiquadratic (f(∞) = f(1) - f(0))
        /// 3. Multiply pointwise to get Az*Bz on multiquadratic grid
        /// 4. Sum with eq weights to get t'(0) and t'(∞)
        ///
        /// IMPORTANT: The eq weights for cycles use a FACTORIZED representation:
        ///   eq_val[i] = E_out[i / E_in.len] * E_in[i % E_in.len]
        ///
        /// The r_grid contains eq(r_bound, cycle_bits) for already-bound challenges.
        /// This is used to weight each cycle according to how it matches the bound challenges.
        ///
        /// For the streaming round (current_round == 1):
        /// - We sum over all cycles and constraint groups
        ///
        /// For subsequent cycle rounds:
        /// - We sum over cycle halves (based on current variable)
        /// - Each cycle is weighted by E_out * E_in * r_grid[k]
        pub fn computeRemainingRoundPolyMultiquadratic(self: *Self) ![4]F {
            // Match Jolt's linear phase index structure exactly:
            //   full_idx = x_out << (x_in_bits + window + r_bits) | x_in << (window + r_bits) | x_val << r_bits | r_idx
            //   step_idx = full_idx >> 1
            //   selector = full_idx & 1
            //
            // The constraint group selector is always the LSB.
            // For cycle rounds, x_val ∈ {0, 1} is the current cycle bit.
            // r_idx indexes into r_grid for the bound streaming challenges.

            const r_stream = self.r_stream orelse F.zero();

            // Get eq tables for current window
            const eq_tables = self.split_eq.getWindowEqTables(0, 1);
            const E_out = eq_tables.E_out;
            const E_in = eq_tables.E_in;
            const head_in_bits: u6 = @intCast(eq_tables.head_in_bits);

            // r_grid parameters
            const r_grid = &self.r_grid;
            const r_grid_len = r_grid.length();
            const num_r_bits: u6 = if (r_grid_len > 1) @intCast(std.math.log2_int(usize, r_grid_len)) else 0;

            // window_size is always 1 for linear phase cycle rounds
            const window_bits: u6 = 1;

            // Accumulators for multiquadratic polynomial
            var t_00 = F.zero(); // t'(0)
            var t_inf = F.zero(); // t'(∞)

            // Iterate over the factorized index space: (x_out, x_in) × (x_val) × (r_idx)
            var x_out_idx: usize = 0;
            while (x_out_idx < E_out.len) : (x_out_idx += 1) {
                const e_out_val = E_out[x_out_idx];

                var x_in_idx: usize = 0;
                while (x_in_idx < E_in.len) : (x_in_idx += 1) {
                    const e_in_val = E_in[x_in_idx];
                    const eq_base = e_out_val.mul(e_in_val);

                    // Accumulate Az/Bz for x_val = 0 and x_val = 1
                    var az_grid = [2]F{ F.zero(), F.zero() };
                    var bz_grid = [2]F{ F.zero(), F.zero() };

                    // Compute base_idx following Jolt's structure
                    const base_idx: usize = (x_out_idx << @intCast(head_in_bits + window_bits + num_r_bits)) |
                        (x_in_idx << @intCast(window_bits + num_r_bits));

                    // Iterate over x_val (window variable) and r_idx
                    var x_val: usize = 0;
                    while (x_val < 2) : (x_val += 1) {
                        const x_val_shifted = x_val << num_r_bits;

                        var r_idx: usize = 0;
                        while (r_idx < r_grid_len) : (r_idx += 1) {
                            const r_weight = r_grid.get(r_idx);

                            // Compute full_idx and derive step_idx
                            const full_idx = base_idx | x_val_shifted | r_idx;
                            const step_idx = full_idx >> 1;

                            // Get Az/Bz for this cycle
                            if (step_idx < self.cycle_witnesses.len) {
                                const result = self.computeCycleAzBzSeparate(&self.cycle_witnesses[step_idx], r_stream);
                                // Weight by r_grid and accumulate
                                az_grid[x_val] = az_grid[x_val].add(r_weight.mul(result.az));
                                bz_grid[x_val] = bz_grid[x_val].add(r_weight.mul(result.bz));
                            }
                        }
                    }

                    // Multiquadratic: t'(0) and t'(∞)
                    const prod_0 = az_grid[0].mul(bz_grid[0]);
                    const slope_az = az_grid[1].sub(az_grid[0]);
                    const slope_bz = bz_grid[1].sub(bz_grid[0]);
                    const prod_inf = slope_az.mul(slope_bz);

                    t_00 = t_00.add(eq_base.mul(prod_0));
                    t_inf = t_inf.add(eq_base.mul(prod_inf));
                }
            }

            // Use Gruen's method
            const previous_claim = self.current_claim;
            const round_poly = self.split_eq.computeCubicRoundPoly(
                t_00,
                t_inf,
                previous_claim,
            );

            return round_poly;
        }

        /// Bind a remaining round challenge
        ///
        /// Matches Jolt's HalfSplitSchedule:
        /// - Streaming phase (first half): update r_grid with each challenge
        /// - Linear phase (second half): don't update r_grid, just bind split_eq
        ///
        /// For the remaining sumcheck with num_rounds = 1 + num_cycle_vars:
        /// - Switch-over is at round (num_rounds + 1) / 2
        /// - Streaming rounds: 1 to switch_over (inclusive)
        /// - Linear rounds: switch_over+1 to num_rounds (exclusive)
        pub fn bindRemainingRoundChallenge(self: *Self, r: F) !void {
            // If this is the streaming round (current_round == 1), save r_stream
            if (self.current_round == 1) {
                self.r_stream = r;
            }

            // Jolt's HalfSplitSchedule: streaming phase is first half of rounds
            // For num_rounds = 1 + num_cycle_vars, switch-over is at (num_rounds + 1) / 2
            // This matches the outer remaining sumcheck structure in Jolt
            const num_remaining_rounds = self.numRounds() - 1; // Exclude UniSkip round 0
            const switch_over = (num_remaining_rounds + 1) / 2;

            // Streaming phase: rounds 1 to switch_over (inclusive) - update r_grid
            // Linear phase: rounds > switch_over - don't update r_grid
            if (self.current_round <= switch_over) {
                // Streaming window phase: update r_grid
                self.r_grid.update(r);
            }
            // Linear phase: don't update r_grid

            try self.challenges.append(self.allocator, r);
            self.split_eq.bind(r);

            self.current_round += 1;
        }

        /// Update the current claim after a round
        ///
        /// round_poly contains evaluations [s(0), s(1), s(2), s(3)], NOT coefficients.
        /// We need to first convert to coefficients, then evaluate at challenge.
        pub fn updateClaim(self: *Self, round_poly: [4]F, challenge: F) void {
            // Convert evaluations to coefficients via Lagrange interpolation
            const coeffs = poly_mod.UniPoly(F).interpolateDegree3(round_poly);

            // Now evaluate the polynomial at challenge using Horner's method
            // s(r) = c0 + r * (c1 + r * (c2 + r * c3))
            self.current_claim = coeffs[0]
                .add(challenge.mul(
                coeffs[1]
                    .add(challenge.mul(
                    coeffs[2]
                        .add(challenge.mul(coeffs[3])),
                )),
            ));
        }

        /// Get the final evaluation after all rounds
        pub fn getFinalEval(self: *const Self) F {
            return self.current_claim;
        }

        /// Generate the full sumcheck proof
        pub fn generateProof(
            self: *Self,
            transcript: anytype,
        ) !struct {
            uniskip_proof: jolt_types.UniSkipFirstRoundProof(F),
            sumcheck_proof: jolt_types.SumcheckInstanceProof(F),
            final_eval: F,
        } {
            // Round 0: Univariate skip
            const first_round_coeffs = try self.computeFirstRoundPoly();

            // Get challenge from transcript
            transcript.appendSlice(&first_round_coeffs);
            const r0 = transcript.challengeScalar();

            // Compute the claim from first round polynomial
            const uni_skip_claim = self.evaluatePolyAtChallenge(&first_round_coeffs, r0);
            try self.bindFirstRoundChallenge(r0, uni_skip_claim);

            // Create proofs
            const uniskip_proof = try jolt_types.UniSkipFirstRoundProof(F).init(
                self.allocator,
                &first_round_coeffs,
            );

            var sumcheck_proof = jolt_types.SumcheckInstanceProof(F).init(self.allocator);

            // Remaining rounds
            while (self.current_round < self.numRounds()) {
                const round_poly = try self.computeRemainingRoundPoly();

                // Add to proof (compressed format - omit linear term)
                const coeffs = [_]F{ round_poly[0], round_poly[2], round_poly[3] };
                try sumcheck_proof.addRoundPoly(&coeffs);

                // Get challenge
                transcript.appendSlice(&round_poly);
                const r = transcript.challengeScalar();

                // Update state
                self.updateClaim(round_poly, r);
                try self.bindRemainingRoundChallenge(r);
            }

            return .{
                .uniskip_proof = uniskip_proof,
                .sumcheck_proof = sumcheck_proof,
                .final_eval = self.getFinalEval(),
            };
        }

        /// Evaluate polynomial at a point using Horner's method
        fn evaluatePolyAtChallenge(self: *const Self, coeffs: []const F, x: F) F {
            _ = self;
            if (coeffs.len == 0) return F.zero();

            var result = coeffs[coeffs.len - 1];
            var i = coeffs.len - 1;
            while (i > 0) {
                i -= 1;
                result = result.mul(x).add(coeffs[i]);
            }
            return result;
        }
    };
}

/// Round up to next power of two
fn nextPowerOfTwo(n: usize) usize {
    if (n == 0) return 1;
    var v = n - 1;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v |= v >> 32;
    return v + 1;
}

// ============================================================================
// Tests
// ============================================================================

const testing = std.testing;
const BN254Scalar = @import("../../field/mod.zig").BN254Scalar;

test "StreamingOuterProver: initialization" {
    const F = BN254Scalar;

    // Create trivial witnesses
    const witnesses = [_]constraints.R1CSCycleInputs(F){
        .{ .values = [_]F{F.zero()} ** 36 },
        .{ .values = [_]F{F.zero()} ** 36 },
        .{ .values = [_]F{F.zero()} ** 36 },
        .{ .values = [_]F{F.zero()} ** 36 },
    };

    const tau = [_]F{ F.fromU64(1), F.fromU64(2), F.fromU64(3) };

    var prover = try StreamingOuterProver(F).init(testing.allocator, &witnesses, &tau);
    defer prover.deinit();

    try testing.expectEqual(@as(usize, 2), prover.num_cycle_vars); // log2(4) = 2
    try testing.expectEqual(@as(usize, 4), prover.padded_trace_len);
    try testing.expectEqual(@as(usize, 3), prover.numRounds()); // 1 + 2 (streaming + 2 cycle vars)
}

test "StreamingOuterProver: first round poly" {
    const F = BN254Scalar;

    const witnesses = [_]constraints.R1CSCycleInputs(F){
        .{ .values = [_]F{F.zero()} ** 36 },
        .{ .values = [_]F{F.zero()} ** 36 },
    };

    const tau = [_]F{ F.fromU64(1), F.fromU64(2) };

    var prover = try StreamingOuterProver(F).init(testing.allocator, &witnesses, &tau);
    defer prover.deinit();

    const first_round = try prover.computeFirstRoundPoly();

    // With zero witnesses, all coefficients should be zero
    for (first_round) |coeff| {
        try testing.expect(coeff.eql(F.zero()));
    }
}

test "StreamingOuterProver: Lagrange basis at r0" {
    const F = BN254Scalar;

    const witnesses = [_]constraints.R1CSCycleInputs(F){
        .{ .values = [_]F{F.zero()} ** 36 },
    };

    const tau = [_]F{F.fromU64(1)};

    var prover = try StreamingOuterProver(F).init(testing.allocator, &witnesses, &tau);
    defer prover.deinit();

    // Bind first round with r0 = 0
    // Domain is {-4, -3, -2, -1, 0, 1, 2, 3, 4, 5}
    // So domain point 0 is at index 4, meaning L_4(0) = 1
    try prover.bindFirstRoundChallenge(F.zero(), F.zero());

    try testing.expect(prover.lagrange_evals_r0[4].eql(F.one()));
}
