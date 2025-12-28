//! R1CS Constraint Evaluators for Jolt Compatibility
//!
//! This module provides structured evaluators for Az and Bz that match Jolt's
//! first-group and second-group constraint organization. These evaluators are
//! used for the univariate skip optimization in stages 1 and 2.
//!
//! ## Constraint Groups
//!
//! - **First Group (10 constraints)**: Boolean Az guards, Bz fits in ~64 bits
//!   Maps to univariate skip domain {-4, -3, -2, -1, 0, 1, 2, 3, 4, 5}
//!
//! - **Second Group (9 constraints)**: Mixed Az types, Bz can be ~128-160 bits
//!   Evaluated separately (not part of first-round univariate skip)
//!
//! Reference: jolt-core/src/zkvm/r1cs/evaluation.rs

const std = @import("std");
const Allocator = std.mem.Allocator;

const constraints = @import("constraints.zig");
const R1CSInputIndex = constraints.R1CSInputIndex;
const UNIFORM_CONSTRAINTS = constraints.UNIFORM_CONSTRAINTS;
const FIRST_GROUP_INDICES = constraints.FIRST_GROUP_INDICES;
const SECOND_GROUP_INDICES = constraints.SECOND_GROUP_INDICES;

/// Number of first group constraints (univariate skip domain size)
pub const FIRST_GROUP_SIZE: usize = 10;

/// Number of second group constraints
pub const SECOND_GROUP_SIZE: usize = 9;

/// Base domain left index for univariate skip
pub const BASE_LEFT: i64 = -4;

/// Az evaluation result for first group (10 boolean guards)
pub fn AzFirstGroup(comptime F: type) type {
    return struct {
        const Self = @This();

        /// The 10 Az guard values
        values: [FIRST_GROUP_SIZE]F,

        /// Initialize from per-cycle witness values
        pub fn fromWitness(witness: []const F) Self {
            var result = Self{ .values = undefined };

            // Evaluate each first-group constraint's condition (Az)
            inline for (FIRST_GROUP_INDICES, 0..) |constraint_idx, i| {
                const constraint = UNIFORM_CONSTRAINTS[constraint_idx];
                result.values[i] = constraint.condition.evaluate(F, witness);
            }

            return result;
        }

        /// Get Az value for first-group constraint at domain point y
        /// Domain point y maps to constraint index: i = y - BASE_LEFT
        pub fn atDomainPoint(self: *const Self, y_i64: i64) F {
            const idx = y_i64 - BASE_LEFT;
            if (idx >= 0 and idx < FIRST_GROUP_SIZE) {
                return self.values[@intCast(idx)];
            }
            return F.zero();
        }
    };
}

/// Bz evaluation result for first group (10 magnitude values)
pub fn BzFirstGroup(comptime F: type) type {
    return struct {
        const Self = @This();

        /// The 10 Bz magnitude values (left - right)
        values: [FIRST_GROUP_SIZE]F,

        /// Initialize from per-cycle witness values
        pub fn fromWitness(witness: []const F) Self {
            var result = Self{ .values = undefined };

            // Evaluate each first-group constraint's magnitude (Bz = left - right)
            inline for (FIRST_GROUP_INDICES, 0..) |constraint_idx, i| {
                const constraint = UNIFORM_CONSTRAINTS[constraint_idx];
                const left = constraint.left.evaluate(F, witness);
                const right = constraint.right.evaluate(F, witness);
                result.values[i] = left.sub(right);
            }

            return result;
        }

        /// Get Bz value for first-group constraint at domain point y
        pub fn atDomainPoint(self: *const Self, y_i64: i64) F {
            const idx = y_i64 - BASE_LEFT;
            if (idx >= 0 and idx < FIRST_GROUP_SIZE) {
                return self.values[@intCast(idx)];
            }
            return F.zero();
        }
    };
}

/// Az evaluation result for second group (9 guards)
pub fn AzSecondGroup(comptime F: type) type {
    return struct {
        const Self = @This();

        /// The 9 Az guard values
        values: [SECOND_GROUP_SIZE]F,

        /// Initialize from per-cycle witness values
        pub fn fromWitness(witness: []const F) Self {
            var result = Self{ .values = undefined };

            inline for (SECOND_GROUP_INDICES, 0..) |constraint_idx, i| {
                const constraint = UNIFORM_CONSTRAINTS[constraint_idx];
                result.values[i] = constraint.condition.evaluate(F, witness);
            }

            return result;
        }
    };
}

/// Bz evaluation result for second group (9 magnitude values)
pub fn BzSecondGroup(comptime F: type) type {
    return struct {
        const Self = @This();

        /// The 9 Bz magnitude values
        values: [SECOND_GROUP_SIZE]F,

        /// Initialize from per-cycle witness values
        pub fn fromWitness(witness: []const F) Self {
            var result = Self{ .values = undefined };

            inline for (SECOND_GROUP_INDICES, 0..) |constraint_idx, i| {
                const constraint = UNIFORM_CONSTRAINTS[constraint_idx];
                const left = constraint.left.evaluate(F, witness);
                const right = constraint.right.evaluate(F, witness);
                result.values[i] = left.sub(right);
            }

            return result;
        }
    };
}

/// Combined Az*Bz evaluator for univariate skip first round
///
/// This computes the product Az(cycle, y) * Bz(cycle, y) for each domain point y,
/// summed over all execution cycles.
pub fn UnivariateSkipEvaluator(comptime F: type) type {
    return struct {
        const Self = @This();

        /// Per-cycle Az evaluations for first group
        az_first_group: []AzFirstGroup(F),
        /// Per-cycle Bz evaluations for first group
        bz_first_group: []BzFirstGroup(F),
        /// Number of cycles
        num_cycles: usize,
        allocator: Allocator,

        /// Initialize from execution trace witnesses
        pub fn init(
            allocator: Allocator,
            cycle_witnesses: []const constraints.R1CSCycleInputs(F),
        ) !Self {
            const num_cycles = cycle_witnesses.len;

            const az_first = try allocator.alloc(AzFirstGroup(F), num_cycles);
            errdefer allocator.free(az_first);
            const bz_first = try allocator.alloc(BzFirstGroup(F), num_cycles);

            for (0..num_cycles) |i| {
                const witness = cycle_witnesses[i].asSlice();
                az_first[i] = AzFirstGroup(F).fromWitness(witness);
                bz_first[i] = BzFirstGroup(F).fromWitness(witness);
            }

            return Self{
                .az_first_group = az_first,
                .bz_first_group = bz_first,
                .num_cycles = num_cycles,
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.az_first_group);
            self.allocator.free(self.bz_first_group);
        }

        /// Compute Az*Bz product at domain point y, summed over all cycles
        ///
        /// This is t1(y) = Σ_{x} eq(tau, x) * Az(x, y) * Bz(x, y)
        /// where x iterates over cycles and y is the constraint index.
        ///
        /// For the first-round univariate skip, we compute this at each of the
        /// 10 base domain points {-4, -3, ..., 5}.
        pub fn computeAzBzSum(self: *const Self, y_i64: i64, eq_evals: []const F) F {
            var sum = F.zero();

            for (0..self.num_cycles) |cycle| {
                if (cycle >= eq_evals.len) break;

                const eq_val = eq_evals[cycle];
                const az = self.az_first_group[cycle].atDomainPoint(y_i64);
                const bz = self.bz_first_group[cycle].atDomainPoint(y_i64);

                sum = sum.add(eq_val.mul(az.mul(bz)));
            }

            return sum;
        }

        /// Compute base window evaluations for univariate skip
        ///
        /// Returns t1(y) for y in {BASE_LEFT, BASE_LEFT+1, ..., BASE_LEFT+9}
        pub fn computeBaseWindowEvals(self: *const Self, eq_evals: []const F) [FIRST_GROUP_SIZE]F {
            var result: [FIRST_GROUP_SIZE]F = undefined;

            for (0..FIRST_GROUP_SIZE) |i| {
                const y = BASE_LEFT + @as(i64, @intCast(i));
                result[i] = self.computeAzBzSum(y, eq_evals);
            }

            return result;
        }

        /// Compute extended evaluations for univariate skip
        ///
        /// For the univariate skip, we need evaluations at extended domain points
        /// outside the base window. These are computed by evaluating the constraint
        /// polynomials at extended Y values.
        ///
        /// For the first group (10 constraints), the extended domain extends to
        /// {-9, -8, ..., -5} ∪ {6, 7, 8, 9} (9 additional points).
        pub fn computeExtendedEvals(
            self: *const Self,
            eq_evals: []const F,
            comptime NUM_EXTENDED: usize,
            targets: [NUM_EXTENDED]i64,
        ) [NUM_EXTENDED]F {
            var result: [NUM_EXTENDED]F = undefined;

            for (0..NUM_EXTENDED) |i| {
                const y = targets[i];
                // For extended points, we evaluate the constraint polynomial at y
                // Since constraints are Lagrange polynomials over the base domain,
                // we need to extrapolate using the Lagrange basis
                result[i] = self.evaluateExtendedPoint(y, eq_evals);
            }

            return result;
        }

        /// Compute extended evaluations using precomputed Lagrange coefficients.
        ///
        /// This is the correct approach: we evaluate Az(y_j) and Bz(y_j) separately
        /// using the COEFFS_PER_J weights, then multiply them. This gives non-zero
        /// results even when base evaluations of Az*Bz are all zero.
        pub fn computeExtendedEvalsWithCoeffs(
            self: *const Self,
            eq_evals: []const F,
            comptime NUM_EXTENDED: usize,
        ) [NUM_EXTENDED]F {
            const univariate_skip = @import("univariate_skip.zig");

            var result: [NUM_EXTENDED]F = undefined;
            @memset(&result, F.zero());

            // For each cycle, compute the contribution to extended evaluations
            for (0..self.num_cycles) |cycle| {
                if (cycle >= eq_evals.len) break;

                const eq_val = eq_evals[cycle];
                const az_vals = self.az_first_group[cycle].values;
                const bz_vals = self.bz_first_group[cycle].values;

                // For each extended target point j
                for (0..NUM_EXTENDED) |j| {
                    // Get the precomputed Lagrange coefficients for target j
                    const coeffs = univariate_skip.COEFFS_PER_J[j];

                    // Compute Az(y_j) = sum_i coeffs[i] * az_vals[i]
                    var az_at_yj = F.zero();
                    for (0..FIRST_GROUP_SIZE) |i| {
                        const coeff_i = coeffs[i];
                        if (coeff_i >= 0) {
                            az_at_yj = az_at_yj.add(az_vals[i].mul(F.fromU64(@intCast(coeff_i))));
                        } else {
                            az_at_yj = az_at_yj.sub(az_vals[i].mul(F.fromU64(@intCast(-coeff_i))));
                        }
                    }

                    // Compute Bz(y_j) = sum_i coeffs[i] * bz_vals[i]
                    var bz_at_yj = F.zero();
                    for (0..FIRST_GROUP_SIZE) |i| {
                        const coeff_i = coeffs[i];
                        if (coeff_i >= 0) {
                            bz_at_yj = bz_at_yj.add(bz_vals[i].mul(F.fromU64(@intCast(coeff_i))));
                        } else {
                            bz_at_yj = bz_at_yj.sub(bz_vals[i].mul(F.fromU64(@intCast(-coeff_i))));
                        }
                    }

                    // Add eq-weighted Az*Bz product to this extended point
                    result[j] = result[j].add(eq_val.mul(az_at_yj.mul(bz_at_yj)));
                }
            }

            return result;
        }

        /// Evaluate Az*Bz at an extended domain point using Lagrange extrapolation
        /// DEPRECATED: This doesn't work for satisfied constraints (base evals are zero)
        fn evaluateExtendedPoint(self: *const Self, y_i64: i64, eq_evals: []const F) F {
            // For points outside the base domain, we use Lagrange extrapolation
            // from the 10 base domain evaluations.
            //
            // L_i(y) = Π_{j≠i} (y - x_j) / (x_i - x_j)
            //
            // where x_j = BASE_LEFT + j for j in {0, ..., 9}

            // First, compute the base evaluations
            const base_evals = self.computeBaseWindowEvals(eq_evals);

            // Now extrapolate to y using Lagrange interpolation
            var result = F.zero();
            const y = fieldFromI64(F, y_i64);

            for (0..FIRST_GROUP_SIZE) |i| {
                const x_i = fieldFromI64(F, BASE_LEFT + @as(i64, @intCast(i)));

                // Compute L_i(y) = Π_{j≠i} (y - x_j) / (x_i - x_j)
                var numerator = F.one();
                var denominator = F.one();

                for (0..FIRST_GROUP_SIZE) |j| {
                    if (i == j) continue;
                    const x_j = fieldFromI64(F, BASE_LEFT + @as(i64, @intCast(j)));
                    numerator = numerator.mul(y.sub(x_j));
                    denominator = denominator.mul(x_i.sub(x_j));
                }

                const lagrange_coeff = numerator.mul(denominator.inv());
                result = result.add(lagrange_coeff.mul(base_evals[i]));
            }

            return result;
        }
    };
}

/// Convert i64 to field element (handling negatives)
fn fieldFromI64(comptime F: type, val: i64) F {
    if (val >= 0) {
        return F.fromU64(@intCast(val));
    } else {
        return F.zero().sub(F.fromU64(@intCast(-val)));
    }
}

// ============================================================================
// Tests
// ============================================================================

test "az first group from witness" {
    const field = @import("../../field/mod.zig");
    const F = field.BN254Scalar;

    // Create a witness for a LOAD instruction
    var witness: [R1CSInputIndex.NUM_INPUTS]F = [_]F{F.zero()} ** R1CSInputIndex.NUM_INPUTS;
    witness[R1CSInputIndex.FlagLoad.toIndex()] = F.one();

    const az = AzFirstGroup(F).fromWitness(&witness);

    // Constraint 1 (first group index 0): if { 1 - Load - Store } ...
    // With Load=1, Store=0, condition = 1 - 1 - 0 = 0
    try std.testing.expect(az.values[0].eql(F.zero()));

    // Constraint 2 (first group index 1): if { Load } ...
    // With Load=1, condition = 1
    try std.testing.expect(az.values[1].eql(F.one()));

    // Constraint 3 (first group index 2): if { Load } ...
    try std.testing.expect(az.values[2].eql(F.one()));

    // Constraint 4 (first group index 3): if { Store } ...
    // With Store=0, condition = 0
    try std.testing.expect(az.values[3].eql(F.zero()));
}

test "bz first group from witness" {
    const field = @import("../../field/mod.zig");
    const F = field.BN254Scalar;

    // Create a witness for a LOAD instruction with matching values
    var witness: [R1CSInputIndex.NUM_INPUTS]F = [_]F{F.zero()} ** R1CSInputIndex.NUM_INPUTS;
    witness[R1CSInputIndex.FlagLoad.toIndex()] = F.one();
    witness[R1CSInputIndex.RamReadValue.toIndex()] = F.fromU64(42);
    witness[R1CSInputIndex.RamWriteValue.toIndex()] = F.fromU64(42);
    witness[R1CSInputIndex.RdWriteValue.toIndex()] = F.fromU64(42);

    const bz = BzFirstGroup(F).fromWitness(&witness);

    // Constraint 2 (first group index 1): RamReadValue == RamWriteValue
    // With equal values, Bz = 42 - 42 = 0
    try std.testing.expect(bz.values[1].eql(F.zero()));

    // Constraint 3 (first group index 2): RamReadValue == RdWriteValue
    try std.testing.expect(bz.values[2].eql(F.zero()));
}

test "az*bz product for satisfied constraint is zero" {
    const field = @import("../../field/mod.zig");
    const F = field.BN254Scalar;

    // Create a witness for a LOAD instruction with matching values
    var witness: [R1CSInputIndex.NUM_INPUTS]F = [_]F{F.zero()} ** R1CSInputIndex.NUM_INPUTS;
    witness[R1CSInputIndex.FlagLoad.toIndex()] = F.one();
    witness[R1CSInputIndex.RamReadValue.toIndex()] = F.fromU64(42);
    witness[R1CSInputIndex.RamWriteValue.toIndex()] = F.fromU64(42);

    const az = AzFirstGroup(F).fromWitness(&witness);
    const bz = BzFirstGroup(F).fromWitness(&witness);

    // For constraint 2 (first group index 1): if Load => RamReadValue == RamWriteValue
    // Az = 1, Bz = 0, so Az * Bz = 0
    const product = az.values[1].mul(bz.values[1]);
    try std.testing.expect(product.eql(F.zero()));
}

test "domain point mapping" {
    const field = @import("../../field/mod.zig");
    const F = field.BN254Scalar;

    var witness: [R1CSInputIndex.NUM_INPUTS]F = [_]F{F.zero()} ** R1CSInputIndex.NUM_INPUTS;
    witness[R1CSInputIndex.FlagLoad.toIndex()] = F.one();

    const az = AzFirstGroup(F).fromWitness(&witness);

    // Domain point -4 should map to first constraint (index 0)
    try std.testing.expect(az.atDomainPoint(-4).eql(az.values[0]));

    // Domain point 0 should map to constraint index 4
    try std.testing.expect(az.atDomainPoint(0).eql(az.values[4]));

    // Domain point 5 should map to constraint index 9
    try std.testing.expect(az.atDomainPoint(5).eql(az.values[9]));

    // Out of range should return zero
    try std.testing.expect(az.atDomainPoint(-5).eql(F.zero()));
    try std.testing.expect(az.atDomainPoint(6).eql(F.zero()));
}

test "field from i64" {
    const field = @import("../../field/mod.zig");
    const F = field.BN254Scalar;

    try std.testing.expect(fieldFromI64(F, 0).eql(F.zero()));
    try std.testing.expect(fieldFromI64(F, 1).eql(F.one()));
    try std.testing.expect(fieldFromI64(F, 5).eql(F.fromU64(5)));

    // Test negative: -3 should equal 0 - 3
    const neg3 = fieldFromI64(F, -3);
    const expected = F.zero().sub(F.fromU64(3));
    try std.testing.expect(neg3.eql(expected));
}
