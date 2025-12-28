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
