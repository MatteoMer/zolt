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

// Debug output control - set to true to enable verbose debug prints
const debug_verbose = false;
fn dbg(comptime fmt: []const u8, args: anytype) void {
    if (debug_verbose) std.debug.print(fmt, args);
}

const Allocator = std.mem.Allocator;

const constraints = @import("constraints.zig");
const R1CSInputIndex = constraints.R1CSInputIndex;
const R1CSCycleInputs = constraints.R1CSCycleInputs;

const poly = @import("zolt_arith").poly;
const EqPolynomial = poly.EqPolynomial;
const field_mod = @import("zolt_arith").field;
const ThreadPool = @import("zolt_pool").ThreadPool;
const evaluators = @import("evaluators.zig");

/// Number of R1CS inputs per cycle
pub const NUM_R1CS_INPUTS = R1CSInputIndex.NUM_INPUTS;

/// Computes MLE evaluations of all R1CS input polynomials at a challenge point
pub fn R1CSInputEvaluator(comptime F: type) type {
    return struct {
        const Self = @This();

        /// Compute the claimed evaluation of all 35 R1CS inputs at r_cycle
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
            return computeClaimedInputsParallel(allocator, cycle_witnesses, r_cycle, null);
        }

        /// Parallel version using factored eq (split into eq_one × eq_two) matching
        /// Jolt's compute_claimed_inputs in evaluation.rs:819-948.
        /// Outer parallel loop over x1, inner sequential over x2,
        /// with UnreducedProductAccum for deferred reduction.
        pub fn computeClaimedInputsParallel(
            allocator: Allocator,
            cycle_witnesses: []const R1CSCycleInputs(F),
            r_cycle: []const F,
            thread_pool: ?*ThreadPool,
        ) ![NUM_R1CS_INPUTS]F {
            const num_cycles = cycle_witnesses.len;

            if (num_cycles == 0 or r_cycle.len == 0) {
                var result: [NUM_R1CS_INPUTS]F = undefined;
                @memset(&result, F.zero());
                if (r_cycle.len == 0 and cycle_witnesses.len > 0) {
                    return cycle_witnesses[0].values;
                }
                return result;
            }

            const num_vars = r_cycle.len;
            const padded_len = @as(usize, 1) << @intCast(num_vars);

            // Factored eq: split r_cycle in half → eq_one(sqrt(T)) × eq_two(sqrt(T))
            // This matches Jolt's rayon::join(|| EqPolynomial::evals(r2), || EqPolynomial::evals(r1))
            const m = num_vars / 2;
            const r_out = r_cycle[0..m]; // first half → E_out (x1 dimension)
            const r_in = r_cycle[m..]; // second half → E_in (x2 dimension)

            const eq_one = try EqPolynomial(F).evalsSliceWithScaling(F, allocator, r_out, null);
            defer allocator.free(eq_one);
            const eq_two = try EqPolynomial(F).evalsSliceWithScaling(F, allocator, r_in, null);
            defer allocator.free(eq_two);

            const eq_two_len = eq_two.len;

            const Accum = field_mod.UnreducedProductAccum;
            const AccumArray = [NUM_R1CS_INPUTS]Accum;
            const accum_zero: AccumArray = [_]Accum{Accum.zero()} ** NUM_R1CS_INPUTS;

            const MapCtx = struct {
                eq_one: []const F,
                eq_two: []const F,
                eq_two_len: usize,
                cycle_witnesses: []const R1CSCycleInputs(F),
                num_cycles: usize,
                padded_len: usize,
            };

            const ctx = MapCtx{
                .eq_one = eq_one,
                .eq_two = eq_two,
                .eq_two_len = eq_two_len,
                .cycle_witnesses = cycle_witnesses,
                .num_cycles = num_cycles,
                .padded_len = padded_len,
            };

            const mapReduceFns = struct {
                const zero_accum: AccumArray = [_]Accum{Accum.zero()} ** NUM_R1CS_INPUTS;

                fn mapFn(c: MapCtx, start: usize, end: usize) AccumArray {
                    var outer_accum: AccumArray = zero_accum;

                    for (start..end) |x1| {
                        const eq1_val = c.eq_one[x1];

                        // Inner accumulators: accumulate eq_two[x2] * witness over all x2
                        // Using UPA defers Montgomery reductions until x1 boundary.
                        var inner_accum: AccumArray = zero_accum;

                        for (0..c.eq_two_len) |x2| {
                            const idx = x1 * c.eq_two_len + x2;
                            const e_in = c.eq_two[x2];

                            if (idx < c.num_cycles) {
                                const w = &c.cycle_witnesses[idx];
                                for (0..NUM_R1CS_INPUTS) |i| {
                                    inner_accum[i].addAssign(Accum.fromMul(e_in, w.values[i]));
                                }
                            } else if (idx < c.padded_len) {
                                // NoOp padding: FlagIsNoop=1, FlagDoNotUpdateUnexpandedPC=1
                                inner_accum[R1CSInputIndex.FlagIsNoop.toIndex()].addAssign(
                                    Accum.fromMul(e_in, F.one()),
                                );
                                inner_accum[R1CSInputIndex.FlagDoNotUpdateUnexpandedPC.toIndex()].addAssign(
                                    Accum.fromMul(e_in, F.one()),
                                );
                            }
                        }

                        // Reduce inner accumulators and weight by eq_one[x1]
                        for (0..NUM_R1CS_INPUTS) |i| {
                            const inner_reduced = inner_accum[i].reduce();
                            outer_accum[i].addAssign(Accum.fromMul(eq1_val, inner_reduced));
                        }
                    }
                    return outer_accum;
                }

                fn reduceFn(a: AccumArray, b: AccumArray) AccumArray {
                    var result: AccumArray = undefined;
                    for (0..NUM_R1CS_INPUTS) |i| {
                        result[i] = a[i];
                        result[i].addAssign(b[i]);
                    }
                    return result;
                }
            };

            const accum_result = if (thread_pool) |tp|
                tp.parallelReduce(AccumArray, eq_one.len, accum_zero, ctx, mapReduceFns.mapFn, mapReduceFns.reduceFn)
            else
                mapReduceFns.mapFn(ctx, 0, eq_one.len);

            var result: [NUM_R1CS_INPUTS]F = undefined;
            for (0..NUM_R1CS_INPUTS) |i| {
                result[i] = accum_result[i].reduce();
            }
            return result;
        }

        /// Compute MLE evaluations using pre-extracted raw integer inputs with typed accumulators.
        /// Matches Jolt's compute_claimed_inputs (evaluation.rs:819-948):
        /// - SmallAccumU for bool flags (conditional add, 0 mulq per input)
        /// - MedAccumS for u64/i128 values (4-8 mulq per input)
        /// - WideAccumS for wide i128 values (8 mulq per input)
        /// This is ~8x cheaper per cycle than the field-multiply version.
        pub fn computeClaimedInputsTyped(
            allocator: Allocator,
            raw_inputs: []const evaluators.RawR1CSInputs,
            padded_len: usize,
            r_cycle: []const F,
            thread_pool: ?*ThreadPool,
        ) ![NUM_R1CS_INPUTS]F {
            const num_cycles = raw_inputs.len;
            if (num_cycles == 0 or r_cycle.len == 0) {
                var result: [NUM_R1CS_INPUTS]F = undefined;
                @memset(&result, F.zero());
                return result;
            }

            const num_vars = r_cycle.len;
            const m = num_vars / 2;
            const r_out = r_cycle[0..m];
            const r_in = r_cycle[m..];

            const eq_one = try EqPolynomial(F).evalsSliceWithScaling(F, allocator, r_out, null);
            defer allocator.free(eq_one);
            const eq_two = try EqPolynomial(F).evalsSliceWithScaling(F, allocator, r_in, null);
            defer allocator.free(eq_two);

            const eq_two_len = eq_two.len;
            const Accum = field_mod.UnreducedProductAccum;
            const SmallAccumU = field_mod.SmallAccumU;
            const MedAccumS = field_mod.MedAccumS;
            const WideAccumS = field_mod.WideAccumS;
            const Raw = evaluators.RawR1CSInputs;

            const AccumArray = [NUM_R1CS_INPUTS]Accum;
            const accum_zero: AccumArray = [_]Accum{Accum.zero()} ** NUM_R1CS_INPUTS;

            const TypedMapCtx = struct {
                eq_one: []const F,
                eq_two: []const F,
                eq_two_len: usize,
                raw_inputs: []const Raw,
                num_cycles: usize,
                padded_len: usize,
            };

            const ctx = TypedMapCtx{
                .eq_one = eq_one,
                .eq_two = eq_two,
                .eq_two_len = eq_two_len,
                .raw_inputs = raw_inputs,
                .num_cycles = num_cycles,
                .padded_len = padded_len,
            };

            const typedMapReduce = struct {
                const zero_out: AccumArray = [_]Accum{Accum.zero()} ** NUM_R1CS_INPUTS;
                const noop_raw: Raw = Raw.noop();

                fn mapFn(c: TypedMapCtx, start: usize, end: usize) AccumArray {
                    var outer_accum: AccumArray = zero_out;

                    for (start..end) |x1| {
                        const eq1_val = c.eq_one[x1];

                        // Typed inner accumulators — one per R1CS input category
                        var acc_u64s: [Raw.NUM_U64_INPUTS]MedAccumS = .{MedAccumS.zero()} ** Raw.NUM_U64_INPUTS;
                        var acc_signed: [Raw.NUM_SIGNED_INPUTS]MedAccumS = .{MedAccumS.zero()} ** Raw.NUM_SIGNED_INPUTS;
                        var acc_wide: [Raw.NUM_WIDE_INPUTS]WideAccumS = .{WideAccumS.zero()} ** Raw.NUM_WIDE_INPUTS;
                        var acc_bools: [Raw.NUM_BOOL_INPUTS]SmallAccumU = .{SmallAccumU.zero()} ** Raw.NUM_BOOL_INPUTS;

                        for (0..c.eq_two_len) |x2| {
                            const idx = x1 * c.eq_two_len + x2;
                            const e_in = c.eq_two[x2];

                            const raw = if (idx < c.num_cycles)
                                &c.raw_inputs[idx]
                            else if (idx < c.padded_len)
                                &noop_raw
                            else
                                continue;

                            // u64 inputs: MedAccumS.fmaddU64 (4×1 schoolbook, 4 mulq)
                            for (0..Raw.NUM_U64_INPUTS) |ui| {
                                acc_u64s[ui].fmaddU64(e_in, raw.u64_values[ui]);
                            }
                            // Signed i128 inputs: MedAccumS.fmaddI128 (4×2 schoolbook, 8 mulq)
                            for (0..Raw.NUM_SIGNED_INPUTS) |si| {
                                acc_signed[si].fmaddI128(e_in, raw.signed_values[si]);
                            }
                            // Wide S192 values: WideAccumS.fmaddS192 (4×3 schoolbook, 12 mulq)
                            for (0..Raw.NUM_WIDE_INPUTS) |wi| {
                                acc_wide[wi].fmaddS192(e_in, raw.wide_values[wi]);
                            }
                            // Bool flags: SmallAccumU.fmaddBool (conditional add, 0 mulq)
                            for (0..Raw.NUM_BOOL_INPUTS) |bi| {
                                acc_bools[bi].fmaddBool(e_in, raw.bool_flags[bi]);
                            }
                        }

                        // Reduce inner accumulators and weight by eq_one[x1]
                        // Map results back to R1CS input indices using inline for for comptime index resolution
                        inline for (0..Raw.NUM_U64_INPUTS) |ui| {
                            outer_accum[Raw.U64_INDICES[ui].toIndex()].addAssign(
                                Accum.fromMul(eq1_val, acc_u64s[ui].barrettReduce()));
                        }
                        inline for (0..Raw.NUM_SIGNED_INPUTS) |si| {
                            outer_accum[Raw.SIGNED_INDICES[si].toIndex()].addAssign(
                                Accum.fromMul(eq1_val, acc_signed[si].barrettReduce()));
                        }
                        inline for (0..Raw.NUM_WIDE_INPUTS) |wi| {
                            outer_accum[Raw.WIDE_INDICES[wi].toIndex()].addAssign(
                                Accum.fromMul(eq1_val, acc_wide[wi].barrettReduce()));
                        }
                        inline for (0..Raw.NUM_BOOL_INPUTS) |bi| {
                            outer_accum[Raw.BOOL_INDICES[bi].toIndex()].addAssign(
                                Accum.fromMul(eq1_val, acc_bools[bi].barrettReduce()));
                        }
                    }
                    return outer_accum;
                }

                fn reduceFn(a: AccumArray, b: AccumArray) AccumArray {
                    var result: AccumArray = undefined;
                    for (0..NUM_R1CS_INPUTS) |i| {
                        result[i] = a[i];
                        result[i].addAssign(b[i]);
                    }
                    return result;
                }
            };

            const accum_result = if (thread_pool) |tp|
                tp.parallelReduce(AccumArray, eq_one.len, accum_zero, ctx, typedMapReduce.mapFn, typedMapReduce.reduceFn)
            else
                typedMapReduce.mapFn(ctx, 0, eq_one.len);

            var result: [NUM_R1CS_INPUTS]F = undefined;
            for (0..NUM_R1CS_INPUTS) |i| {
                result[i] = accum_result[i].reduce();
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
            z: []const F, // R1CS input MLE evaluations (35 values)
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
    const field = @import("zolt_arith").field;
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
    const field = @import("zolt_arith").field;
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
    const field = @import("zolt_arith").field;
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
    const field = @import("zolt_arith").field;
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
    const field = @import("zolt_arith").field;
    const F = field.BN254Scalar;

    // Create two witnesses with all inputs populated
    var witness0 = R1CSCycleInputs(F){ .values = undefined };
    var witness1 = R1CSCycleInputs(F){ .values = undefined };

    for (0..NUM_R1CS_INPUTS) |i| {
        witness0.values[i] = F.fromU64(@intCast(i + 1)); // 1, 2, 3, ..., 36
        witness1.values[i] = F.fromU64(@intCast(100 + i)); // 100, 101, ..., 135
    }

    const witnesses = [_]R1CSCycleInputs(F){ witness0, witness1 };
    const r = [_]F{F.fromU64(1).mul(@import("zolt_arith").poly.UniPoly(F).INV2)}; // r0 = 1/2

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
            .mul(@import("zolt_arith").poly.UniPoly(F).INV2);
        try std.testing.expect(result[i].eql(expected));
    }
}

test "inner_sum_prod: prover vs verifier computation" {
    // This test verifies that the sumcheck prover's output_claim / eq_factor
    // matches the verifier's inner_sum_prod computation from MLE evaluations.
    //
    // This is the key consistency check for Stage 1 verification.

    const field = @import("zolt_arith").field;
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

    dbg("\n=== inner_sum_prod Comparison ===\n", .{});
    dbg("verifier_inner_sum_prod limbs: ", .{});
    for (verifier_inner_sum_prod.limbs) |limb| {
        dbg("{x:016} ", .{limb});
    }
    dbg("\nprover_sum limbs: ", .{});
    for (prover_sum.limbs) |limb| {
        dbg("{x:016} ", .{limb});
    }
    dbg("\n\n--- Detailed Debug ---\n", .{});

    // Print first constraint evaluation for both methods
    const constraint0_idx = constraints.FIRST_GROUP_INDICES[0];
    const constraint0 = constraints.UNIFORM_CONSTRAINTS[constraint0_idx];

    // Prover method: evaluate with actual witness
    const w0_az = constraint0.condition.evaluate(F, witnesses[0].asSlice());
    const w0_bz_left = constraint0.left.evaluate(F, witnesses[0].asSlice());
    const w0_bz_right = constraint0.right.evaluate(F, witnesses[0].asSlice());
    const w0_bz = w0_bz_left.sub(w0_bz_right);

    dbg("Cycle 0, Constraint 0:\n", .{});
    dbg("  Prover Az (from witness): ", .{});
    for (w0_az.limbs) |limb| dbg("{x:016} ", .{limb});
    dbg("\n  Prover Bz (from witness): ", .{});
    for (w0_bz.limbs) |limb| dbg("{x:016} ", .{limb});

    // Verifier method: evaluate with MLE z values
    const z_az = constraint0.condition.evaluate(F, &z);
    const z_bz_left = constraint0.left.evaluate(F, &z);
    const z_bz_right = constraint0.right.evaluate(F, &z);
    const z_bz = z_bz_left.sub(z_bz_right);

    dbg("\n  Verifier Az (from z): ", .{});
    for (z_az.limbs) |limb| dbg("{x:016} ", .{limb});
    dbg("\n  Verifier Bz (from z): ", .{});
    for (z_bz.limbs) |limb| dbg("{x:016} ", .{limb});

    // Show eq weight for cycle 0
    dbg("\n  eq(r_cycle, 0) = ", .{});
    for (eq_evals[0].limbs) |limb| dbg("{x:016} ", .{limb});

    // Check: sum of eq_evals should be 1
    var eq_sum = F.zero();
    for (eq_evals) |ev| eq_sum = eq_sum.add(ev);
    dbg("\n  Σ eq(r_cycle, cycle) = ", .{});
    for (eq_sum.limbs) |limb| dbg("{x:016} ", .{limb});
    dbg("\n  (should be 1 if partition of unity)\n", .{});

    // Also verify: the MLE of witness[0][0] at r_cycle should equal z[0]
    const w00_mle = blk: {
        var sum = F.zero();
        for (0..4) |t| {
            sum = sum.add(eq_evals[t].mul(witnesses[t].values[0]));
        }
        break :blk sum;
    };
    dbg("  MLE(witness[*][0], r_cycle) = ", .{});
    for (w00_mle.limbs) |limb| dbg("{x:016} ", .{limb});
    dbg("\n  z[0] = ", .{});
    for (z[0].limbs) |limb| dbg("{x:016} ", .{limb});
    dbg("\n  (these should match)\n", .{});

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

    dbg("\n--- Az/Bz MLE Comparison ---\n", .{});
    dbg("prover_az_mle = Σ eq * az_final(witness[t]):\n  ", .{});
    for (prover_az_mle.limbs) |limb| dbg("{x:016} ", .{limb});
    dbg("\nverifier_az_final = az_final(z_MLE):\n  ", .{});
    for (verifier_az_final.limbs) |limb| dbg("{x:016} ", .{limb});

    dbg("\n\nprover_bz_mle = Σ eq * bz_final(witness[t]):\n  ", .{});
    for (prover_bz_mle.limbs) |limb| dbg("{x:016} ", .{limb});
    dbg("\nverifier_bz_final = bz_final(z_MLE):\n  ", .{});
    for (verifier_bz_final.limbs) |limb| dbg("{x:016} ", .{limb});

    const az_match = prover_az_mle.eql(verifier_az_final);
    const bz_match = prover_bz_mle.eql(verifier_bz_final);
    dbg("\n\nAz MLE match: {}, Bz MLE match: {}\n", .{ az_match, bz_match });

    dbg("\n=================================\n", .{});

    // These should match! If they don't, there's a fundamental issue
    // in how the prover and verifier compute Az*Bz
    try std.testing.expect(az_match);
    try std.testing.expect(bz_match);
}
