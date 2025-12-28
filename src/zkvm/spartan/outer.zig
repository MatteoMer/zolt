//! Spartan Outer Sumcheck Prover
//!
//! This module implements Stage 1 of Jolt's sumcheck protocol - the "outer" Spartan
//! sumcheck for R1CS constraints. It uses the univariate skip optimization for the
//! first round to produce degree-27 polynomials matching Jolt's format.
//!
//! ## Univariate Skip Optimization
//!
//! For the first round, instead of the standard degree-2 sumcheck polynomial, we
//! produce a degree-27 polynomial that encodes all 19 R1CS constraints:
//!
//! s1(Y) = L(tau_high, Y) * t1(Y)
//!
//! where:
//! - t1(Y) = Σ_{x} eq(tau, x) * Az(x,Y) * Bz(x,Y)
//! - L(tau_high, Y) is the Lagrange kernel polynomial
//!
//! ## Integration with Evaluators
//!
//! The Az and Bz values are computed using the constraint evaluators from
//! `r1cs/evaluators.zig`, which match Jolt's first-group and second-group
//! constraint structure exactly.
//!
//! Reference: jolt-core/src/zkvm/spartan/outer.rs

const std = @import("std");
const Allocator = std.mem.Allocator;

const r1cs_mod = @import("../r1cs/mod.zig");
const univariate_skip = r1cs_mod.univariate_skip;
const constraints = r1cs_mod.constraints;
const evaluators = r1cs_mod.evaluators;
const jolt_types = @import("../jolt_types.zig");

/// Spartan outer prover with univariate skip optimization
pub fn SpartanOuterProver(comptime F: type) type {
    return struct {
        const Self = @This();

        /// Number of R1CS constraints
        pub const NUM_CONSTRAINTS: usize = univariate_skip.NUM_R1CS_CONSTRAINTS;
        /// Domain size for first group
        pub const DOMAIN_SIZE: usize = univariate_skip.OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE;
        /// Degree of univariate skip
        pub const DEGREE: usize = univariate_skip.OUTER_UNIVARIATE_SKIP_DEGREE;
        /// Extended domain size
        pub const EXTENDED_SIZE: usize = univariate_skip.OUTER_UNIVARIATE_SKIP_EXTENDED_DOMAIN_SIZE;
        /// Number of coefficients in first-round polynomial
        pub const NUM_COEFFS: usize = univariate_skip.OUTER_FIRST_ROUND_POLY_NUM_COEFFS;

        /// Az evaluations (condition side of constraints)
        Az: []F,
        /// Bz evaluations (left - right side of constraints)
        Bz: []F,
        /// EQ polynomial evaluations at tau
        eq_evals: []F,
        /// Number of execution cycles
        num_cycles: usize,
        /// tau challenge (split into tau_high, tau_mid, tau_low)
        tau: []const F,
        /// Current working values for sumcheck folding
        working_vals: []F,
        /// Current length
        current_len: usize,
        /// Round challenges collected
        challenges: std.ArrayListUnmanaged(F),
        allocator: Allocator,
        /// Base window evaluations (computed from constraint evaluators)
        base_window_evals: ?[DOMAIN_SIZE]F,
        /// Whether we own the Az/Bz arrays (and need to free them)
        owns_az_bz: bool,

        /// Initialize the prover from R1CS witness data
        pub fn init(
            allocator: Allocator,
            Az: []F,
            Bz: []F,
            eq_evals: []F,
            tau: []const F,
            num_cycles: usize,
        ) !Self {
            // Compute initial working values: eq(tau,x) * Az(x) * Bz(x)
            const size = Az.len;
            const working_vals = try allocator.alloc(F, size);

            for (0..size) |i| {
                const eq_val = if (i < eq_evals.len) eq_evals[i] else F.zero();
                const az_val = if (i < Az.len) Az[i] else F.zero();
                const bz_val = if (i < Bz.len) Bz[i] else F.zero();
                working_vals[i] = eq_val.mul(az_val.mul(bz_val));
            }

            return Self{
                .Az = Az,
                .Bz = Bz,
                .eq_evals = eq_evals,
                .num_cycles = num_cycles,
                .tau = tau,
                .working_vals = working_vals,
                .current_len = size,
                .challenges = .{},
                .allocator = allocator,
                .base_window_evals = null,
                .owns_az_bz = false,
            };
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.working_vals);
            self.challenges.deinit(self.allocator);
            // Free Az/Bz if we own them
            if (self.owns_az_bz) {
                self.allocator.free(self.Az);
                self.allocator.free(self.Bz);
                self.allocator.free(self.eq_evals);
            }
        }

        /// Initialize from per-cycle witness data using the constraint evaluators
        ///
        /// This is the recommended initialization method as it uses the exact
        /// constraint structure that matches Jolt's first-group evaluation.
        pub fn initFromWitnesses(
            allocator: Allocator,
            cycle_witnesses: []const constraints.R1CSCycleInputs(F),
            eq_evals: []const F,
            tau: []const F,
        ) !Self {
            const num_cycles = cycle_witnesses.len;

            // Create the univariate skip evaluator which computes Az*Bz products
            // using the proper constraint structure
            var skip_eval = try evaluators.UnivariateSkipEvaluator(F).init(allocator, cycle_witnesses);
            defer skip_eval.deinit();

            // Compute base window evaluations using the evaluator
            const base_evals = skip_eval.computeBaseWindowEvals(eq_evals);

            // Convert to full Az and Bz arrays for the standard prover interface
            // Size = num_cycles * NUM_CONSTRAINTS (padded)
            const size = num_cycles * NUM_CONSTRAINTS;
            const Az = try allocator.alloc(F, size);
            errdefer allocator.free(Az);
            const Bz = try allocator.alloc(F, size);
            errdefer allocator.free(Bz);

            // Fill in Az and Bz from the per-cycle witnesses
            for (0..num_cycles) |cycle| {
                const witness = cycle_witnesses[cycle].asSlice();
                const az_first = evaluators.AzFirstGroup(F).fromWitness(witness);
                const bz_first = evaluators.BzFirstGroup(F).fromWitness(witness);

                // Store the 10 first-group values
                for (0..DOMAIN_SIZE) |i| {
                    const idx = cycle * NUM_CONSTRAINTS + i;
                    if (idx < size) {
                        Az[idx] = az_first.values[i];
                        Bz[idx] = bz_first.values[i];
                    }
                }

                // Store the 9 second-group values
                const az_second = evaluators.AzSecondGroup(F).fromWitness(witness);
                const bz_second = evaluators.BzSecondGroup(F).fromWitness(witness);

                for (0..evaluators.SECOND_GROUP_SIZE) |i| {
                    const idx = cycle * NUM_CONSTRAINTS + DOMAIN_SIZE + i;
                    if (idx < size) {
                        Az[idx] = az_second.values[i];
                        Bz[idx] = bz_second.values[i];
                    }
                }
            }

            // Copy eq_evals
            const eq_copy = try allocator.alloc(F, eq_evals.len);
            @memcpy(eq_copy, eq_evals);

            // Initialize using the standard constructor
            var self = try Self.init(allocator, Az, Bz, eq_copy, tau, num_cycles);

            // Store the base window evals for use in univariate skip
            self.base_window_evals = base_evals;
            self.owns_az_bz = true;

            return self;
        }

        /// Compute the univariate skip first-round polynomial
        ///
        /// This produces a degree-27 polynomial that encodes the sumcheck for
        /// the first variable, spanning all 19 R1CS constraints.
        ///
        /// If initialized via `initFromWitnesses`, uses the pre-computed base window
        /// evaluations from the constraint evaluators.
        pub fn computeUniskipFirstRoundPoly(self: *Self) !univariate_skip.UniPoly(F) {
            // For univariate skip, we need to compute t1(y) at the extended evaluation points
            // t1(y) = Σ_{x} eq(tau, x) * Az(x, y) * Bz(x, y)
            //
            // The base window covers y ∈ {-4, -3, -2, -1, 0, 1, 2, 3, 4, 5} (10 points)
            // The extended points are y ∈ {-9, -8, ..., -5} ∪ {6, 7, 8, 9} (9 points)

            // Get tau_high (last element of tau)
            const tau_high = if (self.tau.len > 0) self.tau[self.tau.len - 1] else F.zero();

            // Compute base window evaluations: Σ_x eq(τ,x) * Az(x,y) * Bz(x,y)
            // For satisfied constraints, Az*Bz = 0 so base_evals will be all zeros
            var base_evals: [DOMAIN_SIZE]F = undefined;

            if (self.base_window_evals) |precomputed| {
                // Use the precomputed evaluations from the constraint evaluators
                base_evals = precomputed;
            } else {
                // Compute from stored Az/Bz values
                @memset(&base_evals, F.zero());
                for (0..self.num_cycles) |cycle| {
                    const eq_val = self.getEqValue(cycle);
                    for (0..DOMAIN_SIZE) |constraint_idx| {
                        const idx = cycle * NUM_CONSTRAINTS + constraint_idx;
                        if (idx < self.Az.len and idx < self.Bz.len) {
                            const az_bz = self.Az[idx].mul(self.Bz[idx]);
                            base_evals[constraint_idx] = base_evals[constraint_idx].add(eq_val.mul(az_bz));
                        }
                    }
                }
            }

            // Compute extended evaluations using precomputed Lagrange coefficients.
            // This is the CORRECT approach: we evaluate Az(y_j) and Bz(y_j) separately
            // at each extended point using COEFFS_PER_J, then multiply them.
            // This gives non-zero results even when base_evals are all zeros.
            var extended_evals: [DEGREE]F = undefined;
            @memset(&extended_evals, F.zero());

            for (0..self.num_cycles) |cycle| {
                const eq_val = self.getEqValue(cycle);

                // Get Az and Bz values for this cycle (10 constraints each)
                var az_vals: [DOMAIN_SIZE]F = undefined;
                var bz_vals: [DOMAIN_SIZE]F = undefined;
                for (0..DOMAIN_SIZE) |i| {
                    const idx = cycle * NUM_CONSTRAINTS + i;
                    az_vals[i] = if (idx < self.Az.len) self.Az[idx] else F.zero();
                    bz_vals[i] = if (idx < self.Bz.len) self.Bz[idx] else F.zero();
                }

                // For each extended target point j
                for (0..DEGREE) |j| {
                    // Get the precomputed Lagrange coefficients for target j
                    const coeffs = univariate_skip.COEFFS_PER_J[j];

                    // Compute Az(y_j) = sum_i coeffs[i] * az_vals[i]
                    var az_at_yj = F.zero();
                    for (0..DOMAIN_SIZE) |i| {
                        const coeff_i = coeffs[i];
                        if (coeff_i >= 0) {
                            az_at_yj = az_at_yj.add(az_vals[i].mul(F.fromU64(@intCast(coeff_i))));
                        } else {
                            az_at_yj = az_at_yj.sub(az_vals[i].mul(F.fromU64(@intCast(-coeff_i))));
                        }
                    }

                    // Compute Bz(y_j) = sum_i coeffs[i] * bz_vals[i]
                    var bz_at_yj = F.zero();
                    for (0..DOMAIN_SIZE) |i| {
                        const coeff_i = coeffs[i];
                        if (coeff_i >= 0) {
                            bz_at_yj = bz_at_yj.add(bz_vals[i].mul(F.fromU64(@intCast(coeff_i))));
                        } else {
                            bz_at_yj = bz_at_yj.sub(bz_vals[i].mul(F.fromU64(@intCast(-coeff_i))));
                        }
                    }

                    // Add eq-weighted Az*Bz product to this extended point
                    extended_evals[j] = extended_evals[j].add(eq_val.mul(az_at_yj.mul(bz_at_yj)));
                }
            }

            // Build the first-round polynomial
            return try univariate_skip.buildUniskipFirstRoundPoly(
                F,
                DOMAIN_SIZE,
                DEGREE,
                EXTENDED_SIZE,
                NUM_COEFFS,
                &base_evals,
                &extended_evals,
                tau_high,
                self.allocator,
            );
        }

        /// Extrapolate from base window evaluations to an extended point using Lagrange interpolation
        fn extrapolateFromBaseWindow(self: *const Self, base_evals: *const [DOMAIN_SIZE]F, y_i64: i64) F {
            _ = self;
            const BASE_LEFT: i64 = -4; // -((DOMAIN_SIZE - 1) / 2)

            // Lagrange interpolation: sum_i L_i(y) * base_evals[i]
            // L_i(y) = prod_{j != i} (y - x_j) / (x_i - x_j)
            var result = F.zero();
            const y = fieldFromI64(y_i64);

            for (0..DOMAIN_SIZE) |i| {
                const x_i = fieldFromI64(BASE_LEFT + @as(i64, @intCast(i)));

                var numerator = F.one();
                var denominator = F.one();

                for (0..DOMAIN_SIZE) |j| {
                    if (i == j) continue;
                    const x_j = fieldFromI64(BASE_LEFT + @as(i64, @intCast(j)));
                    numerator = numerator.mul(y.sub(x_j));
                    denominator = denominator.mul(x_i.sub(x_j));
                }

                const lagrange_coeff = numerator.mul(denominator.inverse().?);
                result = result.add(lagrange_coeff.mul(base_evals[i]));
            }

            return result;
        }

        /// Convert i64 to field element (handling negatives)
        fn fieldFromI64(val: i64) F {
            if (val >= 0) {
                return F.fromU64(@intCast(val));
            } else {
                return F.zero().sub(F.fromU64(@intCast(-val)));
            }
        }

        /// Get eq polynomial value for a given cycle
        fn getEqValue(self: *const Self, cycle: usize) F {
            if (cycle < self.eq_evals.len) {
                return self.eq_evals[cycle];
            }
            return F.zero();
        }

        /// Create UniSkipFirstRoundProof from the computed polynomial
        pub fn proveUniskipFirstRound(self: *Self) !jolt_types.UniSkipFirstRoundProof(F) {
            var poly = try self.computeUniskipFirstRoundPoly();
            defer poly.deinit();

            // Extract challenge from polynomial (would come from transcript)
            // For now, just return the proof structure
            return try jolt_types.UniSkipFirstRoundProof(F).init(self.allocator, poly.coeffs);
        }

        /// Standard sumcheck round (for rounds after the first)
        pub fn computeStandardRoundPoly(self: *Self) ![3]F {
            if (self.current_len <= 1) {
                return [3]F{
                    if (self.current_len == 1) self.working_vals[0] else F.zero(),
                    F.zero(),
                    F.zero(),
                };
            }

            const half = self.current_len / 2;

            // p(0) = sum of first half
            // p(1) = sum of second half
            var p0 = F.zero();
            var p1 = F.zero();

            for (0..half) |i| {
                p0 = p0.add(self.working_vals[i]);
                p1 = p1.add(self.working_vals[i + half]);
            }

            // p(2) = linear extrapolation
            const p2 = p1.add(p1).sub(p0);

            return [3]F{ p0, p1, p2 };
        }

        /// Bind challenge for a round
        pub fn bindChallenge(self: *Self, challenge: F) !void {
            try self.challenges.append(self.allocator, challenge);

            if (self.current_len <= 1) return;

            const half = self.current_len / 2;
            const one_minus_r = F.one().sub(challenge);

            // Fold: new[i] = (1-r) * old[i] + r * old[i + half]
            for (0..half) |i| {
                self.working_vals[i] = one_minus_r.mul(self.working_vals[i])
                    .add(challenge.mul(self.working_vals[i + half]));
            }

            self.current_len = half;
        }
    };
}

// ============================================================================
// Tests
// ============================================================================

test "spartan outer prover constants" {
    try std.testing.expectEqual(@as(usize, 19), SpartanOuterProver(u64).NUM_CONSTRAINTS);
    try std.testing.expectEqual(@as(usize, 10), SpartanOuterProver(u64).DOMAIN_SIZE);
    try std.testing.expectEqual(@as(usize, 9), SpartanOuterProver(u64).DEGREE);
    try std.testing.expectEqual(@as(usize, 28), SpartanOuterProver(u64).NUM_COEFFS);
}

test "spartan outer prover initialization" {
    const field = @import("../../field/mod.zig");
    const F = field.BN254Scalar;
    const allocator = std.testing.allocator;

    // Create minimal test data
    const Az = try allocator.alloc(F, 19);
    defer allocator.free(Az);
    const Bz = try allocator.alloc(F, 19);
    defer allocator.free(Bz);
    const eq_evals = try allocator.alloc(F, 19);
    defer allocator.free(eq_evals);

    for (0..19) |i| {
        Az[i] = F.fromU64(@intCast(i + 1));
        Bz[i] = F.fromU64(@intCast(i + 2));
        eq_evals[i] = F.one();
    }

    const tau = [_]F{ F.fromU64(1), F.fromU64(2), F.fromU64(3) };

    var prover = try SpartanOuterProver(F).init(allocator, Az, Bz, eq_evals, &tau, 1);
    defer prover.deinit();

    try std.testing.expectEqual(@as(usize, 19), prover.current_len);
}

test "spartan outer uniskip first round poly" {
    const field = @import("../../field/mod.zig");
    const F = field.BN254Scalar;
    const allocator = std.testing.allocator;

    // Create minimal test data
    const Az = try allocator.alloc(F, 19);
    defer allocator.free(Az);
    const Bz = try allocator.alloc(F, 19);
    defer allocator.free(Bz);
    const eq_evals = try allocator.alloc(F, 19);
    defer allocator.free(eq_evals);

    for (0..19) |i| {
        Az[i] = F.fromU64(@intCast(i + 1));
        Bz[i] = F.fromU64(@intCast(i + 2));
        eq_evals[i] = F.one();
    }

    const tau = [_]F{ F.fromU64(1), F.fromU64(2), F.fromU64(3) };

    var prover = try SpartanOuterProver(F).init(allocator, Az, Bz, eq_evals, &tau, 1);
    defer prover.deinit();

    // Compute the univariate skip polynomial
    var poly = try prover.computeUniskipFirstRoundPoly();
    defer poly.deinit();

    // Should have 28 coefficients (degree 27)
    try std.testing.expectEqual(@as(usize, 28), poly.coeffs.len);

    // Check if any coefficients are non-zero (they should be for non-trivial input)
    var has_nonzero = false;
    for (poly.coeffs) |c| {
        if (!c.eql(F.zero())) {
            has_nonzero = true;
            break;
        }
    }
    try std.testing.expect(has_nonzero);
}
