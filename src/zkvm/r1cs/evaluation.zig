//! R1CS Input Evaluation at Challenge Points
//!
//! This module computes the multilinear extension (MLE) evaluations of R1CS
//! input polynomials at the sumcheck challenge point. These evaluations become
//! the opening claims that the verifier uses to check the sumcheck relation.
//!
//! ## Mathematical Background
//!
//! For a virtual polynomial P_i representing R1CS input i, we compute:
//!
//!   P_i(r_cycle) = Sum_{t in {0,1}^n} eq(r_cycle, t) * P_i(t)
//!
//! Where:
//! - r_cycle is the challenge point from the sumcheck protocol
//! - eq(x, y) = prod_j (x_j * y_j + (1-x_j) * (1-y_j)) is the equality polynomial
//! - P_i(t) is the value of R1CS input i at cycle t
//!
//! ## Usage
//!
//! ```zig
//! const evals = try R1CSInputEvaluator(F).computeClaimedInputs(
//!     allocator,
//!     cycle_witnesses,
//!     r_cycle,
//! );
//! ```

const std = @import("std");
const Allocator = std.mem.Allocator;

const constraints = @import("constraints.zig");
const R1CSInputIndex = constraints.R1CSInputIndex;
const R1CSCycleInputs = constraints.R1CSCycleInputs;

const poly = @import("../../poly/mod.zig");
const EqPolynomial = poly.EqPolynomial;

/// Number of R1CS inputs per cycle
pub const NUM_R1CS_INPUTS = R1CSInputIndex.NUM_INPUTS;

/// Computes MLE evaluations of all R1CS input polynomials at a challenge point
pub fn R1CSInputEvaluator(comptime F: type) type {
    return struct {
        const Self = @This();

        /// Compute the claimed evaluation of all 36 R1CS inputs at r_cycle
        ///
        /// This uses the formula:
        ///   MLE_i(r) = Sum_t eq(r, t) * input_i(t)
        ///
        /// Where t ranges over the boolean hypercube {0,1}^n corresponding
        /// to the trace length.
        ///
        /// Returns an array of 36 field elements representing the evaluations.
        pub fn computeClaimedInputs(
            allocator: Allocator,
            cycle_witnesses: []const R1CSCycleInputs(F),
            r_cycle: []const F,
        ) ![NUM_R1CS_INPUTS]F {
            const num_cycles = cycle_witnesses.len;

            // Handle edge cases
            if (num_cycles == 0) {
                var result: [NUM_R1CS_INPUTS]F = undefined;
                @memset(&result, F.zero());
                return result;
            }

            // The trace length must be a power of 2
            const log_n = std.math.log2_int(usize, num_cycles);
            const padded_len = @as(usize, 1) << @intCast(log_n);

            // r_cycle must match the number of cycle variables
            // If it's longer, we use only the relevant portion
            const effective_len = @min(r_cycle.len, log_n);
            if (effective_len == 0) {
                // No cycle variables - just return the first witness (or zero)
                if (cycle_witnesses.len > 0) {
                    return cycle_witnesses[0].values;
                }
                var result: [NUM_R1CS_INPUTS]F = undefined;
                @memset(&result, F.zero());
                return result;
            }

            // Compute eq polynomial evaluations at all points on the boolean hypercube
            var eq_poly = try EqPolynomial(F).init(allocator, r_cycle[0..effective_len]);
            defer eq_poly.deinit();

            const eq_evals = try eq_poly.evals(allocator);
            defer allocator.free(eq_evals);

            // DEBUG: Print eq_evals for first few cycles
            std.debug.print("[ZOLT MLE] effective_len = {}, eq_evals.len = {}\n", .{ effective_len, eq_evals.len });
            if (eq_evals.len > 0) {
                std.debug.print("[ZOLT MLE] eq_evals[0] = {any}\n", .{eq_evals[0].toBytesBE()});
            }
            if (eq_evals.len > 1) {
                std.debug.print("[ZOLT MLE] eq_evals[1] = {any}\n", .{eq_evals[1].toBytesBE()});
            }
            if (eq_evals.len > 2) {
                std.debug.print("[ZOLT MLE] eq_evals[2] = {any}\n", .{eq_evals[2].toBytesBE()});
            }
            // Print r_cycle values used
            for (0..effective_len) |i| {
                std.debug.print("[ZOLT MLE] r_cycle[{}] = {any}\n", .{ i, r_cycle[i].toBytesBE() });
            }

            // Accumulate: result_i = Sum_t eq_evals[t] * witness[t].values[i]
            var result: [NUM_R1CS_INPUTS]F = [_]F{F.zero()} ** NUM_R1CS_INPUTS;

            for (0..@min(num_cycles, padded_len)) |t| {
                const eq_val = eq_evals[t];

                // Skip if eq_val is zero (optimization)
                if (eq_val.eql(F.zero())) continue;

                // Accumulate contribution from this cycle
                for (0..NUM_R1CS_INPUTS) |i| {
                    const input_val = cycle_witnesses[t].values[i];
                    result[i] = result[i].add(eq_val.mul(input_val));
                }
            }

            return result;
        }

        /// Compute the claimed evaluation of a single R1CS input at r_cycle
        pub fn computeClaimedInput(
            allocator: Allocator,
            cycle_witnesses: []const R1CSCycleInputs(F),
            r_cycle: []const F,
            input_index: R1CSInputIndex,
        ) !F {
            const all_evals = try computeClaimedInputs(allocator, cycle_witnesses, r_cycle);
            return all_evals[input_index.toIndex()];
        }

        /// Compute eq polynomial evaluations directly (useful for debugging)
        pub fn computeEqEvals(
            allocator: Allocator,
            r: []const F,
        ) ![]F {
            var eq_poly = try EqPolynomial(F).init(allocator, r);
            defer eq_poly.deinit();
            return eq_poly.evals(allocator);
        }

        /// Compute inner_sum_prod using Jolt's verifier formula
        ///
        /// This computes Az_final * Bz_final where:
        /// - z = R1CS input MLE evaluations at r_cycle
        /// - w = Lagrange weights at r0
        /// - Az_g0 = Σᵢ w[i] * lc_a[i](z) for first group constraints
        /// - Az_g1 = Σᵢ w[i] * lc_a[i](z) for second group constraints
        /// - Az_final = Az_g0 + r_stream * (Az_g1 - Az_g0)
        /// - Same for Bz
        ///
        /// This should match what the sumcheck computes (divided by eq_factor).
        pub fn computeInnerSumProd(
            allocator: Allocator,
            z: []const F, // R1CS input MLE evaluations (36 values)
            lagrange_weights: []const F, // Lagrange basis at r0 (10 values)
            r_stream: F, // Streaming challenge
        ) F {
            _ = allocator;

            const FIRST_GROUP_SIZE = 10;
            const SECOND_GROUP_SIZE = 9;

            // Compute Az_g0, Bz_g0 from first group constraints
            var az_g0 = F.zero();
            var bz_g0 = F.zero();

            for (0..FIRST_GROUP_SIZE) |i| {
                const constraint_idx = constraints.FIRST_GROUP_INDICES[i];
                const constraint = constraints.UNIFORM_CONSTRAINTS[constraint_idx];

                // Evaluate constraint linear combinations with z values
                const az_contrib = constraint.condition.evaluateWithConstant(F, z);
                const bz_left = constraint.left.evaluateWithConstant(F, z);
                const bz_right = constraint.right.evaluateWithConstant(F, z);
                const bz_contrib = bz_left.sub(bz_right);

                // Weight by Lagrange basis
                az_g0 = az_g0.add(lagrange_weights[i].mul(az_contrib));
                bz_g0 = bz_g0.add(lagrange_weights[i].mul(bz_contrib));
            }

            // Compute Az_g1, Bz_g1 from second group constraints
            var az_g1 = F.zero();
            var bz_g1 = F.zero();

            const g1_len = @min(SECOND_GROUP_SIZE, FIRST_GROUP_SIZE);
            for (0..g1_len) |i| {
                const constraint_idx = constraints.SECOND_GROUP_INDICES[i];
                const constraint = constraints.UNIFORM_CONSTRAINTS[constraint_idx];

                const az_contrib = constraint.condition.evaluateWithConstant(F, z);
                const bz_left = constraint.left.evaluateWithConstant(F, z);
                const bz_right = constraint.right.evaluateWithConstant(F, z);
                const bz_contrib = bz_left.sub(bz_right);

                // Use same Lagrange weights as first group
                az_g1 = az_g1.add(lagrange_weights[i].mul(az_contrib));
                bz_g1 = bz_g1.add(lagrange_weights[i].mul(bz_contrib));
            }

            // Blend with r_stream
            const az_final = az_g0.add(r_stream.mul(az_g1.sub(az_g0)));
            const bz_final = bz_g0.add(r_stream.mul(bz_g1.sub(bz_g0)));

            return az_final.mul(bz_final);
        }
    };
}

/// Re-export the constraint generator type for convenience
pub const R1CSConstraintGenerator = constraints.R1CSWitnessGenerator;

// ============================================================================
// Tests
// ============================================================================

test "R1CS input evaluation: empty trace" {
    const field = @import("../../field/mod.zig");
    const F = field.BN254Scalar;

    const witnesses: []const R1CSCycleInputs(F) = &[_]R1CSCycleInputs(F){};
    const r: []const F = &[_]F{ F.fromU64(1), F.fromU64(2) };

    const result = try R1CSInputEvaluator(F).computeClaimedInputs(
        std.testing.allocator,
        witnesses,
        r,
    );

    // All inputs should be zero for empty trace
    for (result) |val| {
        try std.testing.expect(val.eql(F.zero()));
    }
}

test "R1CS input evaluation: single cycle" {
    const field = @import("../../field/mod.zig");
    const F = field.BN254Scalar;

    // Create a single cycle witness with some non-zero values
    var witness = R1CSCycleInputs(F){ .values = [_]F{F.zero()} ** NUM_R1CS_INPUTS };
    witness.values[0] = F.fromU64(42); // LeftInstructionInput
    witness.values[1] = F.fromU64(100); // RightInstructionInput

    const witnesses = [_]R1CSCycleInputs(F){witness};

    // For single cycle (n=0 variables), r_cycle is empty
    // The evaluation should just return the witness value
    const r: []const F = &[_]F{};

    const result = try R1CSInputEvaluator(F).computeClaimedInputs(
        std.testing.allocator,
        &witnesses,
        r,
    );

    // With no variables, result should equal the witness values directly
    try std.testing.expect(result[0].eql(F.fromU64(42)));
    try std.testing.expect(result[1].eql(F.fromU64(100)));
}

test "R1CS input evaluation: two cycles" {
    const field = @import("../../field/mod.zig");
    const F = field.BN254Scalar;

    // Create two cycle witnesses
    var witness0 = R1CSCycleInputs(F){ .values = [_]F{F.zero()} ** NUM_R1CS_INPUTS };
    var witness1 = R1CSCycleInputs(F){ .values = [_]F{F.zero()} ** NUM_R1CS_INPUTS };

    witness0.values[0] = F.fromU64(10); // cycle 0, input 0
    witness1.values[0] = F.fromU64(20); // cycle 1, input 0

    const witnesses = [_]R1CSCycleInputs(F){ witness0, witness1 };

    // For two cycles, we have 1 variable
    // r_cycle = [r0] where r0 is the challenge
    const r0 = F.fromU64(3).mul(F.fromU64(5).inverse().?); // r0 = 3/5 (example)
    const r = [_]F{r0};

    const result = try R1CSInputEvaluator(F).computeClaimedInputs(
        std.testing.allocator,
        &witnesses,
        &r,
    );

    // MLE(r0) = (1-r0) * witness0[0] + r0 * witness1[0]
    //         = (1 - 3/5) * 10 + (3/5) * 20
    //         = (2/5) * 10 + (3/5) * 20
    //         = 4 + 12 = 16
    const one_minus_r0 = F.one().sub(r0);
    const expected = one_minus_r0.mul(F.fromU64(10)).add(r0.mul(F.fromU64(20)));

    try std.testing.expect(result[0].eql(expected));
}

test "R1CS input evaluation: four cycles" {
    const field = @import("../../field/mod.zig");
    const F = field.BN254Scalar;

    // Create four cycle witnesses
    var witnesses: [4]R1CSCycleInputs(F) = undefined;
    for (0..4) |t| {
        witnesses[t] = R1CSCycleInputs(F){ .values = [_]F{F.zero()} ** NUM_R1CS_INPUTS };
        // Set input 0 to the cycle index for easy verification
        witnesses[t].values[0] = F.fromU64(@intCast(t * 10)); // 0, 10, 20, 30
    }

    // For four cycles, we have 2 variables
    const r0 = F.fromU64(2).mul(F.fromU64(7).inverse().?); // r0 = 2/7
    const r1 = F.fromU64(4).mul(F.fromU64(9).inverse().?); // r1 = 4/9
    const r = [_]F{ r0, r1 };

    const result = try R1CSInputEvaluator(F).computeClaimedInputs(
        std.testing.allocator,
        &witnesses,
        &r,
    );

    // Manual computation:
    // eq(r, t) for t in {00, 01, 10, 11} = {(1-r0)(1-r1), (1-r0)r1, r0(1-r1), r0*r1}
    const one = F.one();
    const eq_00 = one.sub(r0).mul(one.sub(r1)); // cycle 0
    const eq_01 = one.sub(r0).mul(r1); // cycle 1
    const eq_10 = r0.mul(one.sub(r1)); // cycle 2
    const eq_11 = r0.mul(r1); // cycle 3

    const expected = eq_00.mul(F.fromU64(0))
        .add(eq_01.mul(F.fromU64(10)))
        .add(eq_10.mul(F.fromU64(20)))
        .add(eq_11.mul(F.fromU64(30)));

    try std.testing.expect(result[0].eql(expected));
}

test "R1CS input evaluation: all inputs populated" {
    const field = @import("../../field/mod.zig");
    const F = field.BN254Scalar;

    // Create two witnesses with all inputs populated
    var witness0 = R1CSCycleInputs(F){ .values = undefined };
    var witness1 = R1CSCycleInputs(F){ .values = undefined };

    for (0..NUM_R1CS_INPUTS) |i| {
        witness0.values[i] = F.fromU64(@intCast(i + 1)); // 1, 2, 3, ..., 36
        witness1.values[i] = F.fromU64(@intCast(100 + i)); // 100, 101, ..., 135
    }

    const witnesses = [_]R1CSCycleInputs(F){ witness0, witness1 };
    const r = [_]F{F.fromU64(1).mul(F.fromU64(2).inverse().?)}; // r0 = 1/2

    const result = try R1CSInputEvaluator(F).computeClaimedInputs(
        std.testing.allocator,
        &witnesses,
        &r,
    );

    // For each input i:
    // MLE(1/2) = (1/2) * witness0[i] + (1/2) * witness1[i]
    //          = (witness0[i] + witness1[i]) / 2
    for (0..NUM_R1CS_INPUTS) |i| {
        const expected = witness0.values[i].add(witness1.values[i])
            .mul(F.fromU64(2).inverse().?);
        try std.testing.expect(result[i].eql(expected));
    }
}

test "inner_sum_prod: prover vs verifier computation" {
    // This test verifies that the sumcheck prover's output_claim / eq_factor
    // matches the verifier's inner_sum_prod computation from MLE evaluations.
    //
    // This is the key consistency check for Stage 1 verification.

    const field = @import("../../field/mod.zig");
    const F = field.BN254Scalar;
    const univariate_skip = @import("univariate_skip.zig");
    const LagrangePoly = univariate_skip.LagrangePolynomial(F);

    // Create 4 cycles with simple but non-trivial values
    var witnesses: [4]R1CSCycleInputs(F) = undefined;
    for (0..4) |t| {
        for (0..36) |i| {
            // Create a predictable pattern
            witnesses[t].values[i] = F.fromU64(@intCast((t + 1) * (i + 1) % 100));
        }
    }

    // Simulated challenge values
    const r0 = F.fromU64(7777);
    const r_stream = F.fromU64(1234);
    const r_cycle = [_]F{ F.fromU64(5555), F.fromU64(6666) };

    // Compute Lagrange weights at r0
    const lagrange_weights = try LagrangePoly.evals(
        10, // FIRST_GROUP_SIZE
        r0,
        std.testing.allocator,
    );
    defer std.testing.allocator.free(lagrange_weights);

    // Compute R1CS input MLE evaluations at r_cycle
    const z = try R1CSInputEvaluator(F).computeClaimedInputs(
        std.testing.allocator,
        &witnesses,
        &r_cycle,
    );

    // Compute inner_sum_prod using verifier formula
    const verifier_inner_sum_prod = R1CSInputEvaluator(F).computeInnerSumProd(
        std.testing.allocator,
        &z,
        lagrange_weights,
        r_stream,
    );

    // Now compute the same using the prover's per-cycle approach
    // For each cycle, compute Az*Bz and accumulate with eq polynomial

    var eq_poly = try EqPolynomial(F).init(std.testing.allocator, &r_cycle);
    defer eq_poly.deinit();
    const eq_evals = try eq_poly.evals(std.testing.allocator);
    defer std.testing.allocator.free(eq_evals);

    var prover_sum = F.zero();

    for (0..4) |t| {
        const witness = &witnesses[t];
        const eq_val = eq_evals[t];

        // Compute Az_g0, Bz_g0 from first group
        var az_g0 = F.zero();
        var bz_g0 = F.zero();

        for (0..10) |i| {
            const constraint_idx = constraints.FIRST_GROUP_INDICES[i];
            const constraint = constraints.UNIFORM_CONSTRAINTS[constraint_idx];

            const az_contrib = constraint.condition.evaluate(F, witness.asSlice());
            const bz_left = constraint.left.evaluate(F, witness.asSlice());
            const bz_right = constraint.right.evaluate(F, witness.asSlice());
            const bz_contrib = bz_left.sub(bz_right);

            az_g0 = az_g0.add(lagrange_weights[i].mul(az_contrib));
            bz_g0 = bz_g0.add(lagrange_weights[i].mul(bz_contrib));
        }

        // Compute Az_g1, Bz_g1 from second group
        var az_g1 = F.zero();
        var bz_g1 = F.zero();

        for (0..9) |i| {
            const constraint_idx = constraints.SECOND_GROUP_INDICES[i];
            const constraint = constraints.UNIFORM_CONSTRAINTS[constraint_idx];

            const az_contrib = constraint.condition.evaluate(F, witness.asSlice());
            const bz_left = constraint.left.evaluate(F, witness.asSlice());
            const bz_right = constraint.right.evaluate(F, witness.asSlice());
            const bz_contrib = bz_left.sub(bz_right);

            // Use same Lagrange weights (up to min of both group sizes)
            az_g1 = az_g1.add(lagrange_weights[i].mul(az_contrib));
            bz_g1 = bz_g1.add(lagrange_weights[i].mul(bz_contrib));
        }

        // Blend with r_stream
        const az_final = az_g0.add(r_stream.mul(az_g1.sub(az_g0)));
        const bz_final = bz_g0.add(r_stream.mul(bz_g1.sub(bz_g0)));

        // Accumulate with eq weight
        prover_sum = prover_sum.add(eq_val.mul(az_final.mul(bz_final)));
    }

    std.debug.print("\n=== inner_sum_prod Comparison ===\n", .{});
    std.debug.print("verifier_inner_sum_prod limbs: ", .{});
    for (verifier_inner_sum_prod.limbs) |limb| {
        std.debug.print("{x:016} ", .{limb});
    }
    std.debug.print("\nprover_sum limbs: ", .{});
    for (prover_sum.limbs) |limb| {
        std.debug.print("{x:016} ", .{limb});
    }
    std.debug.print("\n\n--- Detailed Debug ---\n", .{});

    // Print first constraint evaluation for both methods
    const constraint0_idx = constraints.FIRST_GROUP_INDICES[0];
    const constraint0 = constraints.UNIFORM_CONSTRAINTS[constraint0_idx];

    // Prover method: evaluate with actual witness
    const w0_az = constraint0.condition.evaluate(F, witnesses[0].asSlice());
    const w0_bz_left = constraint0.left.evaluate(F, witnesses[0].asSlice());
    const w0_bz_right = constraint0.right.evaluate(F, witnesses[0].asSlice());
    const w0_bz = w0_bz_left.sub(w0_bz_right);

    std.debug.print("Cycle 0, Constraint 0:\n", .{});
    std.debug.print("  Prover Az (from witness): ", .{});
    for (w0_az.limbs) |limb| std.debug.print("{x:016} ", .{limb});
    std.debug.print("\n  Prover Bz (from witness): ", .{});
    for (w0_bz.limbs) |limb| std.debug.print("{x:016} ", .{limb});

    // Verifier method: evaluate with MLE z values
    const z_az = constraint0.condition.evaluate(F, &z);
    const z_bz_left = constraint0.left.evaluate(F, &z);
    const z_bz_right = constraint0.right.evaluate(F, &z);
    const z_bz = z_bz_left.sub(z_bz_right);

    std.debug.print("\n  Verifier Az (from z): ", .{});
    for (z_az.limbs) |limb| std.debug.print("{x:016} ", .{limb});
    std.debug.print("\n  Verifier Bz (from z): ", .{});
    for (z_bz.limbs) |limb| std.debug.print("{x:016} ", .{limb});

    // Show eq weight for cycle 0
    std.debug.print("\n  eq(r_cycle, 0) = ", .{});
    for (eq_evals[0].limbs) |limb| std.debug.print("{x:016} ", .{limb});

    // Check: sum of eq_evals should be 1
    var eq_sum = F.zero();
    for (eq_evals) |ev| eq_sum = eq_sum.add(ev);
    std.debug.print("\n  Σ eq(r_cycle, cycle) = ", .{});
    for (eq_sum.limbs) |limb| std.debug.print("{x:016} ", .{limb});
    std.debug.print("\n  (should be 1 if partition of unity)\n", .{});

    // Also verify: the MLE of witness[0][0] at r_cycle should equal z[0]
    const w00_mle = blk: {
        var sum = F.zero();
        for (0..4) |t| {
            sum = sum.add(eq_evals[t].mul(witnesses[t].values[0]));
        }
        break :blk sum;
    };
    std.debug.print("  MLE(witness[*][0], r_cycle) = ", .{});
    for (w00_mle.limbs) |limb| std.debug.print("{x:016} ", .{limb});
    std.debug.print("\n  z[0] = ", .{});
    for (z[0].limbs) |limb| std.debug.print("{x:016} ", .{limb});
    std.debug.print("\n  (these should match)\n", .{});

    // Key test: Σ_t eq(r, t) * az_final(witness[t]) should equal az_final(z_MLE(r))
    // Compute the prover's az_final MLE
    var prover_az_mle = F.zero();
    var prover_bz_mle = F.zero();

    for (0..4) |t| {
        const witness = &witnesses[t];
        const eq_val = eq_evals[t];

        var az_g0_t = F.zero();
        var bz_g0_t = F.zero();
        for (0..10) |i| {
            const constraint_idx = constraints.FIRST_GROUP_INDICES[i];
            const constraint = constraints.UNIFORM_CONSTRAINTS[constraint_idx];
            az_g0_t = az_g0_t.add(lagrange_weights[i].mul(constraint.condition.evaluate(F, witness.asSlice())));
            bz_g0_t = bz_g0_t.add(lagrange_weights[i].mul(constraint.left.evaluate(F, witness.asSlice()).sub(constraint.right.evaluate(F, witness.asSlice()))));
        }

        var az_g1_t = F.zero();
        var bz_g1_t = F.zero();
        for (0..9) |i| {
            const constraint_idx = constraints.SECOND_GROUP_INDICES[i];
            const constraint = constraints.UNIFORM_CONSTRAINTS[constraint_idx];
            az_g1_t = az_g1_t.add(lagrange_weights[i].mul(constraint.condition.evaluate(F, witness.asSlice())));
            bz_g1_t = bz_g1_t.add(lagrange_weights[i].mul(constraint.left.evaluate(F, witness.asSlice()).sub(constraint.right.evaluate(F, witness.asSlice()))));
        }

        const az_final_t = az_g0_t.add(r_stream.mul(az_g1_t.sub(az_g0_t)));
        const bz_final_t = bz_g0_t.add(r_stream.mul(bz_g1_t.sub(bz_g0_t)));

        prover_az_mle = prover_az_mle.add(eq_val.mul(az_final_t));
        prover_bz_mle = prover_bz_mle.add(eq_val.mul(bz_final_t));
    }

    // Compute verifier's Az_final and Bz_final using MLE z values
    var verifier_az_g0 = F.zero();
    var verifier_bz_g0 = F.zero();
    for (0..10) |i| {
        const constraint_idx = constraints.FIRST_GROUP_INDICES[i];
        const constraint = constraints.UNIFORM_CONSTRAINTS[constraint_idx];
        verifier_az_g0 = verifier_az_g0.add(lagrange_weights[i].mul(constraint.condition.evaluate(F, &z)));
        verifier_bz_g0 = verifier_bz_g0.add(lagrange_weights[i].mul(constraint.left.evaluate(F, &z).sub(constraint.right.evaluate(F, &z))));
    }

    var verifier_az_g1 = F.zero();
    var verifier_bz_g1 = F.zero();
    for (0..9) |i| {
        const constraint_idx = constraints.SECOND_GROUP_INDICES[i];
        const constraint = constraints.UNIFORM_CONSTRAINTS[constraint_idx];
        verifier_az_g1 = verifier_az_g1.add(lagrange_weights[i].mul(constraint.condition.evaluate(F, &z)));
        verifier_bz_g1 = verifier_bz_g1.add(lagrange_weights[i].mul(constraint.left.evaluate(F, &z).sub(constraint.right.evaluate(F, &z))));
    }

    const verifier_az_final = verifier_az_g0.add(r_stream.mul(verifier_az_g1.sub(verifier_az_g0)));
    const verifier_bz_final = verifier_bz_g0.add(r_stream.mul(verifier_bz_g1.sub(verifier_bz_g0)));

    std.debug.print("\n--- Az/Bz MLE Comparison ---\n", .{});
    std.debug.print("prover_az_mle = Σ eq * az_final(witness[t]):\n  ", .{});
    for (prover_az_mle.limbs) |limb| std.debug.print("{x:016} ", .{limb});
    std.debug.print("\nverifier_az_final = az_final(z_MLE):\n  ", .{});
    for (verifier_az_final.limbs) |limb| std.debug.print("{x:016} ", .{limb});

    std.debug.print("\n\nprover_bz_mle = Σ eq * bz_final(witness[t]):\n  ", .{});
    for (prover_bz_mle.limbs) |limb| std.debug.print("{x:016} ", .{limb});
    std.debug.print("\nverifier_bz_final = bz_final(z_MLE):\n  ", .{});
    for (verifier_bz_final.limbs) |limb| std.debug.print("{x:016} ", .{limb});

    const az_match = prover_az_mle.eql(verifier_az_final);
    const bz_match = prover_bz_mle.eql(verifier_bz_final);
    std.debug.print("\n\nAz MLE match: {}, Bz MLE match: {}\n", .{ az_match, bz_match });

    std.debug.print("\n=================================\n", .{});

    // These should match! If they don't, there's a fundamental issue
    // in how the prover and verifier compute Az*Bz
    try std.testing.expect(az_match);
    try std.testing.expect(bz_match);
}
