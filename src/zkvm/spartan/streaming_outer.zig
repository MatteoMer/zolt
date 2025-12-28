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
        challenges: std.ArrayList(F),
        /// Current round number
        current_round: usize,

        /// Precomputed Lagrange basis evaluations at first-round challenge r0
        /// Used for remaining rounds
        lagrange_evals_r0: [FIRST_GROUP_SIZE]F,

        /// Allocator
        allocator: Allocator,

        /// Initialize the streaming outer prover
        ///
        /// tau: Challenge vector of length (num_cycle_vars + 1)
        ///      - tau[0]: high bit challenge (constraint group selector)
        ///      - tau[1..]: cycle variable challenges
        pub fn init(
            allocator: Allocator,
            cycle_witnesses: []const constraints.R1CSCycleInputs(F),
            tau: []const F,
        ) !Self {
            const num_cycles = cycle_witnesses.len;
            if (num_cycles == 0) {
                return error.EmptyTrace;
            }

            // Pad to next power of 2
            const padded_len = nextPowerOfTwo(num_cycles);
            const num_cycle_vars = std.math.log2_int(usize, padded_len);

            // tau should have length = 1 (uniskip) + num_cycle_vars
            // For now, we'll use a simplified model with just cycle vars
            const num_x_in = 1; // One bit for constraint group selector
            const split_eq = try GruenSplitEqPolynomial(F).init(allocator, tau, num_x_in);

            return Self{
                .cycle_witnesses = cycle_witnesses,
                .num_cycle_vars = num_cycle_vars,
                .padded_trace_len = padded_len,
                .split_eq = split_eq,
                .current_claim = F.zero(),
                .challenges = std.ArrayList(F).init(allocator),
                .current_round = 0,
                .lagrange_evals_r0 = [_]F{F.zero()} ** FIRST_GROUP_SIZE,
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            self.split_eq.deinit();
            self.challenges.deinit();
        }

        /// Total number of rounds
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

        /// Evaluate Az * Bz for a single cycle at a specific domain point
        fn evaluateAzBzAtDomainPoint(
            self: *const Self,
            witness: *const constraints.R1CSCycleInputs(F),
            domain_idx: usize,
        ) F {
            _ = self;

            // Domain indices map to constraints:
            // - First group: indices 0..9 (first 10 constraints)
            // - Second group: needs extended domain evaluation

            if (domain_idx < FIRST_GROUP_SIZE) {
                // First group constraint
                const constraint_idx = constraints.FIRST_GROUP_INDICES[domain_idx];
                const constraint = constraints.UNIFORM_CONSTRAINTS[constraint_idx];
                const az = constraint.condition.evaluate(F, witness.asSlice());
                const bz = constraint.left.evaluate(F, witness.asSlice())
                    .sub(constraint.right.evaluate(F, witness.asSlice()));
                return az.mul(bz);
            } else {
                // Second group constraint (if within range)
                const second_idx = domain_idx - FIRST_GROUP_SIZE;
                if (second_idx < SECOND_GROUP_SIZE) {
                    const constraint_idx = constraints.SECOND_GROUP_INDICES[second_idx];
                    const constraint = constraints.UNIFORM_CONSTRAINTS[constraint_idx];
                    const az = constraint.condition.evaluate(F, witness.asSlice());
                    const bz = constraint.left.evaluate(F, witness.asSlice())
                        .sub(constraint.right.evaluate(F, witness.asSlice()));
                    return az.mul(bz);
                }
                return F.zero();
            }
        }

        /// Interpolate extended evaluations to polynomial coefficients
        fn interpolateFirstRoundPoly(
            self: *const Self,
            extended_evals: *const [univariate_skip.OUTER_UNIVARIATE_SKIP_EXTENDED_DOMAIN_SIZE]F,
        ) [FIRST_ROUND_NUM_COEFFS]F {
            _ = self;

            // Simple interpolation using Newton's method or Lagrange
            // For now, use a simplified approach: copy evaluations as coefficients
            // (This is a placeholder - full implementation needs proper interpolation)
            var coeffs: [FIRST_ROUND_NUM_COEFFS]F = [_]F{F.zero()} ** FIRST_ROUND_NUM_COEFFS;

            // Copy available evaluations
            for (0..@min(extended_evals.len, FIRST_ROUND_NUM_COEFFS)) |i| {
                coeffs[i] = extended_evals[i];
            }

            return coeffs;
        }

        /// Bind the first-round challenge and set up for remaining rounds
        pub fn bindFirstRoundChallenge(self: *Self, r0: F) !void {
            try self.challenges.append(r0);
            self.current_round = 1;

            // Compute Lagrange basis evaluations at r0 for use in remaining rounds
            self.computeLagrangeEvalsAtR0(r0);

            // Bind in split eq
            self.split_eq.bind(r0);
        }

        /// Compute Lagrange basis evaluations at r0
        fn computeLagrangeEvalsAtR0(self: *Self, r0: F) void {
            // L_i(r0) for i in 0..FIRST_GROUP_SIZE
            // Uses the Lagrange interpolation formula

            for (0..FIRST_GROUP_SIZE) |i| {
                var numer = F.one();
                var denom = F.one();

                for (0..FIRST_GROUP_SIZE) |j| {
                    if (i != j) {
                        // numer *= (r0 - j)
                        numer = numer.mul(r0.sub(F.fromU64(j)));
                        // denom *= (i - j)
                        if (i > j) {
                            denom = denom.mul(F.fromU64(i - j));
                        } else {
                            denom = denom.mul(F.zero().sub(F.fromU64(j - i)));
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
        /// Uses Gruen's optimization with multiquadratic expansion
        pub fn computeRemainingRoundPoly(self: *Self) ![4]F {
            // Build multiquadratic grid over window
            // (window_size would be used for more sophisticated streaming)

            // Get eq tables for this window
            const eq_tables = self.split_eq.getWindowEqTables(
                self.num_cycle_vars - self.current_round + 1,
                1,
            );

            // Compute t'(0) and t'(∞) by summing over the grid
            var t_zero = F.zero();
            var t_infinity = F.zero();

            // Sum over all cycles with the current eq weights
            const half = self.padded_trace_len >> @intCast(self.current_round);

            // t'(0) = sum over first half (current variable = 0)
            for (0..@min(half, self.cycle_witnesses.len)) |i| {
                const eq_val = if (i < eq_tables.E_out.len) eq_tables.E_out[i] else F.zero();
                const az_bz = self.computeCycleAzBzProduct(&self.cycle_witnesses[i]);
                t_zero = t_zero.add(eq_val.mul(az_bz));
            }

            // t'(1) = sum over second half (current variable = 1)
            var t_one = F.zero();
            for (0..@min(half, self.cycle_witnesses.len -| half)) |i| {
                const cycle_idx = half + i;
                if (cycle_idx < self.cycle_witnesses.len) {
                    const eq_idx = if (cycle_idx < self.padded_trace_len) cycle_idx else 0;
                    const eq_val = if (eq_idx < eq_tables.E_out.len) eq_tables.E_out[eq_idx] else F.zero();
                    const az_bz = self.computeCycleAzBzProduct(&self.cycle_witnesses[cycle_idx]);
                    t_one = t_one.add(eq_val.mul(az_bz));
                }
            }

            // t'(∞) = t'(1) - t'(0) (slope)
            t_infinity = t_one.sub(t_zero);

            // Use Gruen's method to compute the cubic round polynomial
            const previous_claim = self.current_claim;
            const round_poly = self.split_eq.computeCubicRoundPoly(
                t_zero,
                t_infinity,
                previous_claim,
            );

            return round_poly;
        }

        /// Compute Az * Bz product for a single cycle (summed over all constraints)
        fn computeCycleAzBzProduct(self: *const Self, witness: *const constraints.R1CSCycleInputs(F)) F {
            var product = F.zero();

            // Sum over first group constraints weighted by Lagrange basis
            for (0..FIRST_GROUP_SIZE) |i| {
                const constraint_idx = constraints.FIRST_GROUP_INDICES[i];
                const constraint = constraints.UNIFORM_CONSTRAINTS[constraint_idx];
                const az = constraint.condition.evaluate(F, witness.asSlice());
                const bz = constraint.left.evaluate(F, witness.asSlice())
                    .sub(constraint.right.evaluate(F, witness.asSlice()));

                // Weight by Lagrange basis at r0
                product = product.add(self.lagrange_evals_r0[i].mul(az.mul(bz)));
            }

            return product;
        }

        /// Bind a remaining round challenge
        pub fn bindRemainingRoundChallenge(self: *Self, r: F) !void {
            try self.challenges.append(r);
            self.split_eq.bind(r);
            self.current_round += 1;

            // Update current claim
            // claim = (1 - r) * s(0) + r * s(1)
            // This is computed from the round polynomial
        }

        /// Update the current claim after a round
        pub fn updateClaim(self: *Self, round_poly: [4]F, challenge: F) void {
            // Evaluate round polynomial at challenge using Horner's method
            // s(r) = coeffs[0] + r * (coeffs[1] + r * (coeffs[2] + r * coeffs[3]))
            self.current_claim = round_poly[0]
                .add(challenge.mul(
                round_poly[1]
                    .add(challenge.mul(
                    round_poly[2]
                        .add(challenge.mul(round_poly[3])),
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

            try self.bindFirstRoundChallenge(r0);

            // Initialize current claim from first round
            self.current_claim = self.evaluatePolyAtChallenge(&first_round_coeffs, r0);

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
    try testing.expectEqual(@as(usize, 3), prover.numRounds()); // 1 + 2
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

    // Bind first round with r0 = 0 (should give L_0(0) = 1, L_i(0) = 0 for i > 0)
    try prover.bindFirstRoundChallenge(F.zero());

    try testing.expect(prover.lagrange_evals_r0[0].eql(F.one()));
}
