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

const field_mod = @import("zolt_arith").field;
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

/// Convert i32 to field element (handling negatives)
pub fn fieldFromI32(comptime F: type, val: i32) F {
    if (val >= 0) {
        return F.fromU64(@intCast(val));
    } else if (val == -1) {
        return F.zero().sub(F.one());
    } else {
        return F.zero().sub(F.fromU64(@intCast(-val)));
    }
}

/// Convert i128 to field element. Handles the full i128 range including values
/// produced by wrapping arithmetic (e.g., from @bitCast of large u128 values).
pub fn fieldFromI128(comptime F: type, val: i128) F {
    if (val >= 0) {
        return F.fromU128(@intCast(val));
    } else {
        // For negative values, compute F.zero() - F.fromU128(|val|).
        // Use wrapping negate + bitcast to avoid overflow when val == i128.min.
        const abs: u128 = @bitCast(-%val);
        return F.zero().sub(F.fromU128(abs));
    }
}

// ============================================================================
// Fast integer-based Az evaluation
// ============================================================================

/// Read a boolean witness flag as i8 (0 or 1)
/// Uses Montgomery-form comparison (cheap 4-limb equality check)
fn witnessBool(comptime F: type, witness: []const F, comptime idx: R1CSInputIndex) i8 {
    return if (witness[comptime idx.toIndex()].eql(F.zero())) @as(i8, 0) else @as(i8, 1);
}

/// Compute first-group Az guard values as small integers.
/// All values are guaranteed to be in [-3, 3].
///
/// This avoids all field arithmetic by reading boolean flags directly.
/// Each guard matches the corresponding constraint's condition LC.
pub fn computeAzFirstGroupInt(comptime F: type, witness: []const F) [FIRST_GROUP_SIZE]i8 {
    const load = witnessBool(F, witness, .FlagLoad);
    const store = witnessBool(F, witness, .FlagStore);
    const add = witnessBool(F, witness, .FlagAddOperands);
    const sub_op = witnessBool(F, witness, .FlagSubtractOperands);
    const mul = witnessBool(F, witness, .FlagMultiplyOperands);
    const assert_flag = witnessBool(F, witness, .FlagAssert);
    const should_jump = witnessBool(F, witness, .ShouldJump);
    const vi = witnessBool(F, witness, .FlagVirtualInstruction);
    const is_last = witnessBool(F, witness, .FlagIsLastInSequence);
    const next_is_virtual = witnessBool(F, witness, .NextIsVirtual);
    const next_is_first = witnessBool(F, witness, .NextIsFirstInSequence);

    return .{
        1 - load - store, // FG0: Constraint 1 = 1 - Load - Store
        load, // FG1: Constraint 2 = Load
        load, // FG2: Constraint 3 = Load
        store, // FG3: Constraint 4 = Store
        add + sub_op + mul, // FG4: Constraint 5 = Add + Sub + Mul
        1 - add - sub_op - mul, // FG5: Constraint 6 = 1 - Add - Sub - Mul
        assert_flag, // FG6: Constraint 11 = Assert
        should_jump, // FG7: Constraint 14 = ShouldJump
        vi - is_last, // FG8: Constraint 17 = VI - IsLast
        next_is_virtual - next_is_first, // FG9: Constraint 18 = NextIsVirtual - NextIsFirst
    };
}

/// Compute second-group Az guard values as small integers.
/// All values are guaranteed to be in [-4, 3].
pub fn computeAzSecondGroupInt(comptime F: type, witness: []const F) [SECOND_GROUP_SIZE]i8 {
    const load = witnessBool(F, witness, .FlagLoad);
    const store = witnessBool(F, witness, .FlagStore);
    const add = witnessBool(F, witness, .FlagAddOperands);
    const sub_op = witnessBool(F, witness, .FlagSubtractOperands);
    const mul = witnessBool(F, witness, .FlagMultiplyOperands);
    const advice = witnessBool(F, witness, .FlagAdvice);
    const write_lookup = witnessBool(F, witness, .FlagWriteLookupOutputToRD);
    const jump = witnessBool(F, witness, .FlagJump);
    const should_branch = witnessBool(F, witness, .ShouldBranch);

    return .{
        load + store, // SG0: Constraint 0 = Load + Store
        add, // SG1: Constraint 7 = AddOperands
        sub_op, // SG2: Constraint 8 = SubtractOperands
        mul, // SG3: Constraint 9 = MultiplyOperands
        1 - add - sub_op - mul - advice, // SG4: Constraint 10 = 1 - Add - Sub - Mul - Advice
        write_lookup, // SG5: Constraint 12 = WriteLookupOutputToRD
        jump, // SG6: Constraint 13 = Jump
        should_branch, // SG7: Constraint 15 = ShouldBranch
        1 - should_branch - jump, // SG8: Constraint 16 = 1 - ShouldBranch - Jump
    };
}

/// Compute first-group Bz (left - right) values directly from witness fields.
/// Avoids LC.evaluate overhead (i128ToField conversions, etc.).
///
/// Each entry is `constraint.left - constraint.right` computed directly.
pub fn computeBzFirstGroupDirect(comptime F: type, witness: []const F) [FIRST_GROUP_SIZE]F {
    const I = R1CSInputIndex;
    return .{
        // FG0: Constraint 1: left=RamAddress, right=0
        witness[comptime I.RamAddress.toIndex()],
        // FG1: Constraint 2: left=RamReadValue, right=RamWriteValue
        witness[comptime I.RamReadValue.toIndex()].sub(witness[comptime I.RamWriteValue.toIndex()]),
        // FG2: Constraint 3: left=RamReadValue, right=RdWriteValue
        witness[comptime I.RamReadValue.toIndex()].sub(witness[comptime I.RdWriteValue.toIndex()]),
        // FG3: Constraint 4: left=Rs2Value, right=RamWriteValue
        witness[comptime I.Rs2Value.toIndex()].sub(witness[comptime I.RamWriteValue.toIndex()]),
        // FG4: Constraint 5: left=LeftLookupOperand, right=0
        witness[comptime I.LeftLookupOperand.toIndex()],
        // FG5: Constraint 6: left=LeftLookupOperand, right=LeftInstructionInput
        witness[comptime I.LeftLookupOperand.toIndex()].sub(witness[comptime I.LeftInstructionInput.toIndex()]),
        // FG6: Constraint 11: left=LookupOutput, right=1
        witness[comptime I.LookupOutput.toIndex()].sub(F.one()),
        // FG7: Constraint 14: left=NextUnexpandedPC, right=LookupOutput
        witness[comptime I.NextUnexpandedPC.toIndex()].sub(witness[comptime I.LookupOutput.toIndex()]),
        // FG8: Constraint 17: left=NextPC, right=PC+1
        witness[comptime I.NextPC.toIndex()].sub(witness[comptime I.PC.toIndex()]).sub(F.one()),
        // FG9: Constraint 18: left=1, right=DoNotUpdateUnexpandedPC
        F.one().sub(witness[comptime I.FlagDoNotUpdateUnexpandedPC.toIndex()]),
    };
}

/// Compute second-group Bz (left - right) values directly from witness fields.
/// The `two_pow_64` parameter must be F representing 2^64 (precomputed by caller).
pub fn computeBzSecondGroupDirect(comptime F: type, witness: []const F, two_pow_64: F) [SECOND_GROUP_SIZE]F {
    const I = R1CSInputIndex;
    const two = F.one().add(F.one());
    const four = two.add(two);

    return .{
        // SG0: Constraint 0: left=RamAddress, right=Rs1Value + Imm
        witness[comptime I.RamAddress.toIndex()].sub(witness[comptime I.Rs1Value.toIndex()]).sub(witness[comptime I.Imm.toIndex()]),
        // SG1: Constraint 7: left=RightLookupOperand, right=LeftInput + RightInput
        witness[comptime I.RightLookupOperand.toIndex()].sub(witness[comptime I.LeftInstructionInput.toIndex()]).sub(witness[comptime I.RightInstructionInput.toIndex()]),
        // SG2: Constraint 8: left=RightLookupOperand, right=LeftInput - RightInput + 2^64
        witness[comptime I.RightLookupOperand.toIndex()].sub(witness[comptime I.LeftInstructionInput.toIndex()]).add(witness[comptime I.RightInstructionInput.toIndex()]).sub(two_pow_64),
        // SG3: Constraint 9: left=RightLookupOperand, right=Product
        witness[comptime I.RightLookupOperand.toIndex()].sub(witness[comptime I.Product.toIndex()]),
        // SG4: Constraint 10: left=RightLookupOperand, right=RightInstructionInput
        witness[comptime I.RightLookupOperand.toIndex()].sub(witness[comptime I.RightInstructionInput.toIndex()]),
        // SG5: Constraint 12: left=RdWriteValue, right=LookupOutput
        witness[comptime I.RdWriteValue.toIndex()].sub(witness[comptime I.LookupOutput.toIndex()]),
        // SG6: Constraint 13: left=RdWriteValue, right=UnexpandedPC + 4 - 2*IsCompressed
        witness[comptime I.RdWriteValue.toIndex()].sub(witness[comptime I.UnexpandedPC.toIndex()]).sub(four).add(two.mul(witness[comptime I.FlagIsCompressed.toIndex()])),
        // SG7: Constraint 15: left=NextUnexpandedPC, right=UnexpandedPC + Imm
        witness[comptime I.NextUnexpandedPC.toIndex()].sub(witness[comptime I.UnexpandedPC.toIndex()]).sub(witness[comptime I.Imm.toIndex()]),
        // SG8: Constraint 16: left=NextUnexpandedPC, right=UnexpandedPC + 4 - 4*DoNotUpdate - 2*IsCompressed
        witness[comptime I.NextUnexpandedPC.toIndex()].sub(witness[comptime I.UnexpandedPC.toIndex()]).sub(four).add(four.mul(witness[comptime I.FlagDoNotUpdateUnexpandedPC.toIndex()])).add(two.mul(witness[comptime I.FlagIsCompressed.toIndex()])),
    };
}

/// Interpolate Az*Bz at a target point from base group values.
///
/// Given integer Az guards and field Bz magnitudes at the 10 base domain points,
/// uses COEFFS_PER_J to interpolate to target point j, then multiplies.
///
/// Returns Az(Y_j) * Bz(Y_j) as a field element.
pub fn interpolateAzBzProduct(
    comptime F: type,
    az_int: []const i8,
    bz_field: []const F,
    coeffs: []const i32,
    group_size: usize,
) F {
    // Compute Az(Y_j) as integer: Σ coeffs[i] * az_int[i]
    var az_j: i32 = 0;
    for (0..group_size) |i| {
        az_j += coeffs[i] * @as(i32, az_int[i]);
    }

    // Early exit if guard is zero
    if (az_j == 0) return F.zero();

    // Convert Az to field
    const az_f = fieldFromI32(F, az_j);

    // Compute Bz(Y_j) as field: Σ coeffs[i] * bz_field[i]
    var bz_f = F.zero();
    for (0..group_size) |i| {
        const c = coeffs[i];
        if (c == 0) continue;
        if (c == 1) {
            bz_f = bz_f.add(bz_field[i]);
        } else if (c == -1) {
            bz_f = bz_f.sub(bz_field[i]);
        } else if (c > 0) {
            bz_f = bz_f.add(F.fromU64(@intCast(c)).mul(bz_field[i]));
        } else {
            bz_f = bz_f.sub(F.fromU64(@intCast(-c)).mul(bz_field[i]));
        }
    }

    return az_f.mul(bz_f);
}

// ============================================================================
// Compact Integer Witness for Fast Outer Spartan Evaluation
// ============================================================================

/// Compact integer witness storing precomputed Az/Bz values as raw integers.
/// Size: ~256 bytes per cycle vs 1344 bytes for full field witness.
/// This enables L3-cache-resident evaluation (16MB vs 84MB for T=65536).
pub const CompactWitness = struct {
    /// First-group Az guards as small integers (10 values, each in [-3, 3])
    az_first: [FIRST_GROUP_SIZE]i8,
    /// Second-group Az guards as small integers (9 values, each in [-4, 3])
    az_second: [SECOND_GROUP_SIZE]i8,
    _pad: [5]u8 = .{0} ** 5, // align to 24 bytes
    /// First-group Bz magnitudes as i128 (left - right for each constraint)
    /// Must be i128 (not i64) because witness values can be > i64.max
    /// (e.g., sign-extended 32-bit values stored as large u64)
    bz_first: [FIRST_GROUP_SIZE]i128,
    /// Second-group Bz as S192 (signed 192-bit integer, exact arithmetic without wrapping)
    bz_second: [SECOND_GROUP_SIZE]field_mod.S192,

    /// NoOp padding witness: Az reflects constraint conditions on zero flags
    /// (not_load_store=1, not_add_sub_mul=1, etc.), Bz all zero.
    pub fn noop() CompactWitness {
        return .{
            .az_first = .{ 1, 0, 0, 0, 0, 1, 0, 0, 0, 0 },
            .az_second = .{ 0, 0, 0, 0, 1, 0, 0, 0, 1 },
            ._pad = .{0} ** 5,
            .bz_first = .{0} ** FIRST_GROUP_SIZE,
            .bz_second = .{field_mod.S192.zero()} ** SECOND_GROUP_SIZE,
        };
    }
};

/// Raw integer R1CS inputs for typed-accumulator claims computation.
/// Stores each of the 42 R1CS inputs in its natural integer type, avoiding
/// Montgomery encode/decode in the MLE evaluation inner loop.
/// Matches Jolt's R1CSCycleInputs (raw u64/S64/S128/bool).
pub const RawR1CSInputs = struct {
    /// u64-typed inputs (indices: 0,4,5,7,8,9,10,11,12,13,15,16,19)
    /// Order: LeftInput, PC, UnexpPC, RamAddr, Rs1, Rs2, Rd, RamRead, RamWrite, LeftLookup, NextUnexpPC, NextPC, LookupOutput
    u64_values: [NUM_U64_INPUTS]u64,
    /// Signed i128 inputs (indices: 1,6)
    /// Order: RightInput, Imm
    signed_values: [NUM_SIGNED_INPUTS]i128,
    /// Wide values needing >128-bit support (indices: 2,14)
    /// Product (u64*u64 can reach 2^128) and RightLookupOperand (u128)
    /// Stored as S192 for full range. Order: Product, RightLookupOperand
    wide_values: [NUM_WIDE_INPUTS]field_mod.S192,
    /// Bool inputs (indices: 3,17,18,20, 21-41)
    /// Order: ShouldBranch, NextIsVirtual, NextIsFirstInSequence, ShouldJump, then 21 flags (21-41)
    bool_flags: [NUM_BOOL_INPUTS]bool,

    pub const NUM_U64_INPUTS = 13;
    pub const NUM_SIGNED_INPUTS = 2; // RightInput, Imm
    pub const NUM_WIDE_INPUTS = 2; // Product, RightLookupOperand
    pub const NUM_BOOL_INPUTS = 25;

    /// Map from u64_values index to R1CSInputIndex
    pub const U64_INDICES = [NUM_U64_INPUTS]R1CSInputIndex{
        .LeftInstructionInput, .PC,                .UnexpandedPC,     .RamAddress,
        .Rs1Value,             .Rs2Value,          .RdWriteValue,     .RamReadValue,
        .RamWriteValue,        .LeftLookupOperand, .NextUnexpandedPC, .NextPC,
        .LookupOutput,
    };
    /// Map from signed_values index to R1CSInputIndex
    pub const SIGNED_INDICES = [NUM_SIGNED_INPUTS]R1CSInputIndex{
        .RightInstructionInput, .Imm,
    };
    /// Map from wide_values index to R1CSInputIndex
    pub const WIDE_INDICES = [NUM_WIDE_INPUTS]R1CSInputIndex{
        .Product, .RightLookupOperand,
    };
    /// Map from bool_flags index to R1CSInputIndex
    pub const BOOL_INDICES = [NUM_BOOL_INPUTS]R1CSInputIndex{
        .ShouldBranch,          .NextIsVirtual,               .NextIsFirstInSequence,     .ShouldJump,
        .FlagAddOperands,       .FlagSubtractOperands,        .FlagMultiplyOperands,      .FlagLoad,
        .FlagStore,             .FlagJump,                    .FlagWriteLookupOutputToRD, .FlagVirtualInstruction,
        .FlagAssert,            .FlagDoNotUpdateUnexpandedPC, .FlagAdvice,                .FlagIsCompressed,
        .FlagIsFirstInSequence, .FlagIsLastInSequence,        .FlagIsRdNotZero,           .FlagBranch,
        .FlagIsNoop,            .FlagLeftOperandIsRs1,        .FlagLeftOperandIsPC,       .FlagRightOperandIsRs2,
        .FlagRightOperandIsImm,
    };

    /// Convert a single R1CS input from its native integer type to a Montgomery field element.
    /// This enables on-the-fly encoding without pre-computing all 42 field elements per cycle.
    pub fn toFieldValue(self: *const RawR1CSInputs, comptime F: type, comptime idx: R1CSInputIndex) F {
        // Map R1CSInputIndex to the correct typed storage and convert
        switch (idx) {
            // u64 values
            .LeftInstructionInput => return F.fromU64(self.u64_values[0]),
            .PC => return F.fromU64(self.u64_values[1]),
            .UnexpandedPC => return F.fromU64(self.u64_values[2]),
            .RamAddress => return F.fromU64(self.u64_values[3]),
            .Rs1Value => return F.fromU64(self.u64_values[4]),
            .Rs2Value => return F.fromU64(self.u64_values[5]),
            .RdWriteValue => return F.fromU64(self.u64_values[6]),
            .RamReadValue => return F.fromU64(self.u64_values[7]),
            .RamWriteValue => return F.fromU64(self.u64_values[8]),
            .LeftLookupOperand => return F.fromU64(self.u64_values[9]),
            .NextUnexpandedPC => return F.fromU64(self.u64_values[10]),
            .NextPC => return F.fromU64(self.u64_values[11]),
            .LookupOutput => return F.fromU64(self.u64_values[12]),
            // Signed i128 values
            .RightInstructionInput, .Imm => {
                const slot: usize = if (idx == .RightInstructionInput) 0 else 1;
                const val = self.signed_values[slot];
                if (val >= 0) {
                    const v: u128 = @intCast(val);
                    if (v <= 0xFFFFFFFFFFFFFFFF) {
                        return F.fromU64(@intCast(v));
                    } else {
                        var bytes: [16]u8 = undefined;
                        std.mem.writeInt(u128, &bytes, v, .little);
                        return F.fromBytes(&bytes);
                    }
                } else {
                    const neg_v: u128 = @intCast(-val);
                    if (neg_v <= 0xFFFFFFFFFFFFFFFF) {
                        return F.zero().sub(F.fromU64(@intCast(neg_v)));
                    } else {
                        var bytes: [16]u8 = undefined;
                        std.mem.writeInt(u128, &bytes, neg_v, .little);
                        return F.zero().sub(F.fromBytes(&bytes));
                    }
                }
            },
            // Wide S192 values
            .Product, .RightLookupOperand => {
                const slot: usize = if (idx == .Product) 0 else 1;
                const s = self.wide_values[slot];
                // Convert magnitude to field element
                var bytes: [24]u8 = undefined;
                std.mem.writeInt(u64, bytes[0..8], s.magnitude[0], .little);
                std.mem.writeInt(u64, bytes[8..16], s.magnitude[1], .little);
                std.mem.writeInt(u64, bytes[16..24], s.magnitude[2], .little);
                const mag = F.fromBytes(&bytes);
                return if (s.is_positive) mag else F.zero().sub(mag);
            },
            // Bool flags
            else => {
                // All remaining indices are boolean flags
                const bool_idx = switch (idx) {
                    .ShouldBranch => 0,
                    .NextIsVirtual => 1,
                    .NextIsFirstInSequence => 2,
                    .ShouldJump => 3,
                    .FlagAddOperands => 4,
                    .FlagSubtractOperands => 5,
                    .FlagMultiplyOperands => 6,
                    .FlagLoad => 7,
                    .FlagStore => 8,
                    .FlagJump => 9,
                    .FlagWriteLookupOutputToRD => 10,
                    .FlagVirtualInstruction => 11,
                    .FlagAssert => 12,
                    .FlagDoNotUpdateUnexpandedPC => 13,
                    .FlagAdvice => 14,
                    .FlagIsCompressed => 15,
                    .FlagIsFirstInSequence => 16,
                    .FlagIsLastInSequence => 17,
                    .FlagIsRdNotZero => 18,
                    .FlagBranch => 19,
                    .FlagIsNoop => 20,
                    .FlagLeftOperandIsRs1 => 21,
                    .FlagLeftOperandIsPC => 22,
                    .FlagRightOperandIsRs2 => 23,
                    .FlagRightOperandIsImm => 24,
                    else => unreachable,
                };
                return if (self.bool_flags[bool_idx]) F.one() else F.zero();
            },
        }
    }

    /// NoOp witness: FlagIsNoop=true, FlagDoNotUpdateUnexpandedPC=true, all else zero
    pub fn noop() RawR1CSInputs {
        var raw: RawR1CSInputs = undefined;
        raw.u64_values = .{0} ** NUM_U64_INPUTS;
        raw.signed_values = .{0} ** NUM_SIGNED_INPUTS;
        raw.wide_values = .{field_mod.S192.zero()} ** NUM_WIDE_INPUTS;
        raw.bool_flags = .{false} ** NUM_BOOL_INPUTS;
        // FlagIsNoop is at bool index 20, FlagDoNotUpdateUnexpandedPC at bool index 13
        raw.bool_flags[20] = true; // FlagIsNoop
        raw.bool_flags[13] = true; // FlagDoNotUpdateUnexpandedPC
        return raw;
    }
};

const ThreadPool = @import("zolt_pool").ThreadPool;
const pool_helpers = @import("zolt_pool").helpers;

/// Build compact witness array from field-form cycle witnesses.
/// This performs Montgomery de-encoding once per value, then all subsequent
/// evaluation uses pure integer arithmetic.
pub fn buildCompactWitnesses(
    comptime F: type,
    cycle_witnesses: []const constraints.R1CSCycleInputs(F),
    allocator: Allocator,
    thread_pool: ?*ThreadPool,
) ![]CompactWitness {
    const n = cycle_witnesses.len;
    const result = try allocator.alloc(CompactWitness, n);

    const Ctx = struct {
        cycle_witnesses: []const constraints.R1CSCycleInputs(F),
        result: []CompactWitness,
    };

    const ctx = Ctx{
        .cycle_witnesses = cycle_witnesses,
        .result = result,
    };

    const mapFn = struct {
        fn f(c: Ctx, i: usize) void {
            const ws = c.cycle_witnesses[i].asSlice();
            c.result[i] = compactFromFieldWitness(F, ws);
        }
    }.f;

    pool_helpers.parallelForOptional(thread_pool, n, ctx, mapFn);

    return result;
}

/// Build both CompactWitness AND RawR1CSInputs in one parallel pass.
/// Extracts all raw integers from field witnesses once, producing both
/// outputs. This avoids redundant Montgomery de-encoding.
pub fn buildCompactAndRawWitnesses(
    comptime F: type,
    cycle_witnesses: []const constraints.R1CSCycleInputs(F),
    allocator: Allocator,
    thread_pool: ?*ThreadPool,
) !struct { compact: []CompactWitness, raw: []RawR1CSInputs } {
    const n = cycle_witnesses.len;
    const compact = try allocator.alloc(CompactWitness, n);
    errdefer allocator.free(compact);
    const raw = try allocator.alloc(RawR1CSInputs, n);
    errdefer allocator.free(raw);

    const Ctx = struct {
        cycle_witnesses: []const constraints.R1CSCycleInputs(F),
        compact: []CompactWitness,
        raw: []RawR1CSInputs,
    };

    const ctx = Ctx{
        .cycle_witnesses = cycle_witnesses,
        .compact = compact,
        .raw = raw,
    };

    const mapFn = struct {
        fn f(c: Ctx, i: usize) void {
            const ws = c.cycle_witnesses[i].asSlice();
            c.compact[i] = compactFromFieldWitness(F, ws);
            c.raw[i] = rawFromFieldWitness(F, ws);
        }
    }.f;

    pool_helpers.parallelForOptional(thread_pool, n, ctx, mapFn);

    return .{ .compact = compact, .raw = raw };
}

/// Convert a single cycle's field witness to compact integer form.
pub fn compactFromFieldWitnessPublic(comptime F: type, witness: []const F) CompactWitness {
    return compactFromFieldWitness(F, witness);
}
fn compactFromFieldWitness(comptime F: type, witness: []const F) CompactWitness {
    var cw: CompactWitness = undefined;
    cw._pad = .{0} ** 5;

    // Az values (same as existing computeAzFirstGroupInt / computeAzSecondGroupInt)
    cw.az_first = computeAzFirstGroupInt(F, witness);
    cw.az_second = computeAzSecondGroupInt(F, witness);

    // Extract raw integer values via Montgomery de-encoding
    const I = R1CSInputIndex;
    const ram_addr = witness[comptime I.RamAddress.toIndex()].toU64();
    const ram_read = witness[comptime I.RamReadValue.toIndex()].toU64();
    const ram_write = witness[comptime I.RamWriteValue.toIndex()].toU64();
    const rs1 = witness[comptime I.Rs1Value.toIndex()].toU64();
    const rs2 = witness[comptime I.Rs2Value.toIndex()].toU64();
    const rd_write = witness[comptime I.RdWriteValue.toIndex()].toU64();
    const left_lookup = witness[comptime I.LeftLookupOperand.toIndex()].toU64();
    const left_input = witness[comptime I.LeftInstructionInput.toIndex()].toU64();
    // Extract right_input as i128 to handle field-negative values (e.g., ANDI with -8)
    const right_input_f = witness[comptime I.RightInstructionInput.toIndex()];
    const right_input_std = right_input_f.fromMontgomery();
    const right_input_i128: i128 = blk_ri: {
        if (right_input_std.limbs[1] == 0 and right_input_std.limbs[2] == 0 and right_input_std.limbs[3] == 0) {
            break :blk_ri @as(i128, right_input_std.limbs[0]);
        }
        const neg_ri = F.zero().sub(right_input_f);
        const neg_ri_std = neg_ri.fromMontgomery();
        if (neg_ri_std.limbs[1] == 0 and neg_ri_std.limbs[2] == 0 and neg_ri_std.limbs[3] == 0) {
            break :blk_ri -@as(i128, neg_ri_std.limbs[0]);
        }
        break :blk_ri @as(i128, right_input_std.limbs[0]) | (@as(i128, right_input_std.limbs[1]) << 64);
    };
    const lookup_out = witness[comptime I.LookupOutput.toIndex()].toU64();
    const next_upc = witness[comptime I.NextUnexpandedPC.toIndex()].toU64();
    const pc = witness[comptime I.PC.toIndex()].toU64();
    const next_pc = witness[comptime I.NextPC.toIndex()].toU64();
    const upc = witness[comptime I.UnexpandedPC.toIndex()].toU64();
    const dont_update: u64 = if (witness[comptime I.FlagDoNotUpdateUnexpandedPC.toIndex()].eql(F.zero())) 0 else 1;
    const is_compressed: u64 = if (witness[comptime I.FlagIsCompressed.toIndex()].eql(F.zero())) 0 else 1;

    // For second group: values that may be > u64
    // RightLookupOperand and Product can be u128/i128
    // Extract RightLookupOperand as S192, handling field-negative values.
    // Field-negative values (p-k) arise when the operand is a negative immediate (e.g., ANDI -8).
    // We detect sign by checking if the standard form fits in 128 bits.
    const right_lookup_f = witness[comptime I.RightLookupOperand.toIndex()];
    const right_lookup_std = right_lookup_f.fromMontgomery();
    const S192 = field_mod.S192;
    const right_lookup_s192: S192 = blk_rl: {
        if (right_lookup_std.limbs[2] == 0 and right_lookup_std.limbs[3] == 0) {
            // Positive value fits in 128 bits
            const val = @as(u128, right_lookup_std.limbs[0]) | (@as(u128, right_lookup_std.limbs[1]) << 64);
            break :blk_rl S192.fromU128(val);
        }
        // Field-negative: negate to get magnitude, then represent as negative S192
        const neg = F.zero().sub(right_lookup_f);
        const neg_std = neg.fromMontgomery();
        if (neg_std.limbs[2] == 0 and neg_std.limbs[3] == 0) {
            const neg_val = @as(u128, neg_std.limbs[0]) | (@as(u128, neg_std.limbs[1]) << 64);
            break :blk_rl S192.fromU128(neg_val).neg();
        }
        // Fallback: use lower 128 bits (shouldn't happen for valid R1CS values)
        const val = @as(u128, right_lookup_std.limbs[0]) | (@as(u128, right_lookup_std.limbs[1]) << 64);
        break :blk_rl S192.fromU128(val);
    };

    // Extract Product as S192 with same sign-detection
    const product_f = witness[comptime I.Product.toIndex()];
    const product_std = product_f.fromMontgomery();
    const product_s192: S192 = blk_pr: {
        if (product_std.limbs[2] == 0 and product_std.limbs[3] == 0) {
            const val = @as(u128, product_std.limbs[0]) | (@as(u128, product_std.limbs[1]) << 64);
            break :blk_pr S192.fromU128(val);
        }
        const neg = F.zero().sub(product_f);
        const neg_std = neg.fromMontgomery();
        if (neg_std.limbs[2] == 0 and neg_std.limbs[3] == 0) {
            const neg_val = @as(u128, neg_std.limbs[0]) | (@as(u128, neg_std.limbs[1]) << 64);
            break :blk_pr S192.fromU128(neg_val).neg();
        }
        const val = @as(u128, product_std.limbs[0]) | (@as(u128, product_std.limbs[1]) << 64);
        break :blk_pr S192.fromU128(val);
    };

    // Imm has two possible representations depending on instruction type:
    // 1. F.fromU64(sign_extended_u64) for ADDI/JALR: limbs = [large_u64, 0, 0, 0]
    // 2. F.zero().sub(F.fromU64(k)) for BRANCH/LOAD/STORE: limbs = [p-k], all 4 nonzero
    // Extract as i128 to handle both cases correctly.
    const imm_f = witness[comptime I.Imm.toIndex()];
    const imm_std = imm_f.fromMontgomery();
    const imm: i128 = blk: {
        if (imm_std.limbs[1] == 0 and imm_std.limbs[2] == 0 and imm_std.limbs[3] == 0) {
            // Case 1: positive u64 value (possibly large, e.g. 2^64-1 for sign-extended -1)
            break :blk @as(i128, imm_std.limbs[0]);
        }
        // Case 2: field-negative (p - k), extract as -k
        const neg = F.zero().sub(imm_f);
        const neg_std = neg.fromMontgomery();
        if (neg_std.limbs[1] == 0 and neg_std.limbs[2] == 0 and neg_std.limbs[3] == 0) {
            break :blk -@as(i128, neg_std.limbs[0]);
        }
        // Fallback: use lower 128 bits (shouldn't happen for valid RISC-V immediates)
        break :blk @as(i128, imm_std.limbs[0]) | (@as(i128, imm_std.limbs[1]) << 64);
    };

    // First group Bz as integers (i128 to safely hold u64 differences)
    cw.bz_first = .{
        @as(i128, ram_addr), // FG0: RamAddress
        @as(i128, ram_read) - @as(i128, ram_write), // FG1: RamRead - RamWrite
        @as(i128, ram_read) - @as(i128, rd_write), // FG2: RamRead - RdWrite
        @as(i128, rs2) - @as(i128, ram_write), // FG3: Rs2 - RamWrite
        @as(i128, left_lookup), // FG4: LeftLookup
        @as(i128, left_lookup) - @as(i128, left_input), // FG5: LeftLookup - LeftInput
        @as(i128, lookup_out) - 1, // FG6: LookupOutput - 1
        @as(i128, next_upc) - @as(i128, lookup_out), // FG7: NextUPC - LookupOutput
        @as(i128, next_pc) - @as(i128, pc) - 1, // FG8: NextPC - PC - 1
        1 - @as(i128, dont_update), // FG9: 1 - DoNotUpdate
    };

    // Second group Bz as S192 (signed 192-bit integers).
    // Uses exact S192 arithmetic instead of wrapping i128 to avoid sign truncation
    // for values where u128 > 2^127 (e.g., RightLookupOperand, Product).
    const rl = right_lookup_s192;
    cw.bz_second = .{
        // SG0: RamAddress - Rs1 - Imm
        S192.fromU64(ram_addr).sub(S192.fromU64(rs1)).sub(S192.fromI128(imm)),
        // SG1: RightLookup - LeftInput - RightInput
        rl.sub(S192.fromU64(left_input)).sub(S192.fromI128(right_input_i128)),
        // SG2: RightLookup - LeftInput + RightInput - 2^64
        rl.sub(S192.fromU64(left_input)).add(S192.fromI128(right_input_i128)).sub(S192.fromU128(@as(u128, 1) << 64)),
        // SG3: RightLookup - Product
        rl.sub(product_s192),
        // SG4: RightLookup - RightInput
        rl.sub(S192.fromI128(right_input_i128)),
        // SG5: RdWrite - LookupOutput
        S192.fromU64(rd_write).sub(S192.fromU64(lookup_out)),
        // SG6: RdWrite - UPC - 4 + 2*IsCompressed
        S192.fromU64(rd_write).sub(S192.fromU64(upc)).sub(S192.fromU64(4)).add(S192.fromU64(2 * is_compressed)),
        // SG7: NextUPC - UPC - Imm
        S192.fromU64(next_upc).sub(S192.fromU64(upc)).sub(S192.fromI128(imm)),
        // SG8: NextUPC - UPC - 4 + 4*DoNotUpdate + 2*IsCompressed
        S192.fromU64(next_upc).sub(S192.fromU64(upc)).sub(S192.fromU64(4)).add(S192.fromU64(4 * dont_update)).add(S192.fromU64(2 * is_compressed)),
    };

    return cw;
}

/// Extract raw integer R1CS inputs from field-form witness (public wrapper).
pub fn rawFromFieldWitnessPublic(comptime F: type, witness: []const F) RawR1CSInputs {
    return rawFromFieldWitness(F, witness);
}
/// Extract raw integer R1CS inputs from field-form witness.
fn rawFromFieldWitness(comptime F: type, witness: []const F) RawR1CSInputs {
    const I = R1CSInputIndex;
    var raw: RawR1CSInputs = undefined;

    // u64 values: direct .toU64() extraction
    raw.u64_values = .{
        witness[comptime I.LeftInstructionInput.toIndex()].toU64(),
        witness[comptime I.PC.toIndex()].toU64(),
        witness[comptime I.UnexpandedPC.toIndex()].toU64(),
        witness[comptime I.RamAddress.toIndex()].toU64(),
        witness[comptime I.Rs1Value.toIndex()].toU64(),
        witness[comptime I.Rs2Value.toIndex()].toU64(),
        witness[comptime I.RdWriteValue.toIndex()].toU64(),
        witness[comptime I.RamReadValue.toIndex()].toU64(),
        witness[comptime I.RamWriteValue.toIndex()].toU64(),
        witness[comptime I.LeftLookupOperand.toIndex()].toU64(),
        witness[comptime I.NextUnexpandedPC.toIndex()].toU64(),
        witness[comptime I.NextPC.toIndex()].toU64(),
        witness[comptime I.LookupOutput.toIndex()].toU64(),
    };

    // Signed i128 values: sign-detection extraction
    raw.signed_values[0] = extractI128(F, witness[comptime I.RightInstructionInput.toIndex()]);
    raw.signed_values[1] = extractI128(F, witness[comptime I.Imm.toIndex()]);

    // Wide values as S192: Product (u64*u64 can reach 2^128) and RightLookupOperand
    raw.wide_values[0] = extractS192(F, witness[comptime I.Product.toIndex()]);
    raw.wide_values[1] = extractS192(F, witness[comptime I.RightLookupOperand.toIndex()]);

    // Bool flags: check if field value equals zero
    inline for (RawR1CSInputs.BOOL_INDICES, 0..) |idx, i| {
        raw.bool_flags[i] = !witness[comptime idx.toIndex()].eql(F.zero());
    }

    return raw;
}

/// Extract a potentially-signed i128 from a field element.
/// Handles both positive (small standard form) and negative (p - k) representations.
pub fn extractI128(comptime F: type, val: F) i128 {
    const std_form = val.fromMontgomery();
    // Check if positive: fits in lower 128 bits
    if (std_form.limbs[2] == 0 and std_form.limbs[3] == 0) {
        return @as(i128, std_form.limbs[0]) | (@as(i128, std_form.limbs[1]) << 64);
    }
    // Check if negative: p - k where k fits in 128 bits
    const neg = F.zero().sub(val);
    const neg_std = neg.fromMontgomery();
    if (neg_std.limbs[2] == 0 and neg_std.limbs[3] == 0) {
        const mag = @as(i128, neg_std.limbs[0]) | (@as(i128, neg_std.limbs[1]) << 64);
        return -mag;
    }
    // Fallback: use lower 128 bits
    return @as(i128, std_form.limbs[0]) | (@as(i128, std_form.limbs[1]) << 64);
}

/// Extract a potentially-signed value as S192 from a field element.
/// Handles values up to 192 bits (including u64*u64 products that can reach 2^128).
fn extractS192(comptime F: type, val: F) field_mod.S192 {
    const S192 = field_mod.S192;
    const std_form = val.fromMontgomery();
    // Check if positive: fits in lower 192 bits (limbs[3] == 0)
    if (std_form.limbs[3] == 0) {
        return S192{ .magnitude = .{ std_form.limbs[0], std_form.limbs[1], std_form.limbs[2] }, .is_positive = true };
    }
    // Check if negative: p - k where k fits in 192 bits
    const neg = F.zero().sub(val);
    const neg_std = neg.fromMontgomery();
    if (neg_std.limbs[3] == 0) {
        return S192{ .magnitude = .{ neg_std.limbs[0], neg_std.limbs[1], neg_std.limbs[2] }, .is_positive = false };
    }
    // Fallback: use lower 192 bits
    return S192{ .magnitude = .{ std_form.limbs[0], std_form.limbs[1], std_form.limbs[2] }, .is_positive = true };
}

/// Integer-based Az*Bz interpolation for the first group.
/// Returns the product as i128 (guaranteed to fit for BN254 coefficient magnitudes).
///
/// Exploits constraint satisfaction: when az_i != 0, bz_i must be 0.
/// So Az(j) only gets contributions from "active" guards, and
/// Bz(j) only gets contributions from "inactive" guards.
pub fn interpolateAzBzProductInt(
    az_int: *const [FIRST_GROUP_SIZE]i8,
    bz_int: *const [FIRST_GROUP_SIZE]i128,
    coeffs: *const [FIRST_GROUP_SIZE]i32,
) i128 {
    // Compute Az(j) = Σ coeffs[i] * az_int[i] (integer, fits in i32)
    var az_j: i32 = 0;
    // Compute Bz(j) = Σ coeffs[i] * bz_int[i] (integer, fits in i128)
    // Only accumulate terms where az_int[i] == 0 (exploiting constraint satisfaction)
    var bz_j: i128 = 0;

    inline for (0..FIRST_GROUP_SIZE) |i| {
        const c = coeffs[i];
        const a = az_int[i];
        if (a != 0) {
            std.debug.assert(bz_int[i] == 0); // constraint satisfaction: az!=0 implies bz==0
            az_j += c * @as(i32, a);
        } else {
            bz_j += @as(i128, c) * @as(i128, bz_int[i]);
        }
    }

    // Early exit if guard sum is zero
    if (az_j == 0) return 0;

    // Product fits in i128: max |az_j| ≈ 10*140140 ≈ 1.4M, max |bz_j| ≈ 10*140140*2^64 ≈ 2^81
    // Product ≈ 2^21 * 2^81 = 2^102, well within i128 range
    return @as(i128, az_j) * bz_j;
}

/// Integer-based Az*Bz interpolation for the second group using S192.
/// Returns the product as S192 (avoids wrapping i128 artifacts).
///
/// Same constraint-satisfaction trick as interpolateAzBzProductInt:
/// when az_i != 0, bz_i must be 0, so Az and Bz accumulate disjointly.
pub fn interpolateAzBzProductSecondGroupInt(
    az_int: *const [SECOND_GROUP_SIZE]i8,
    bz_s192: *const [SECOND_GROUP_SIZE]field_mod.S192,
    coeffs: *const [SECOND_GROUP_SIZE]i32,
) field_mod.S192 {
    const S192 = field_mod.S192;
    var az_j: i32 = 0;
    var bz_j: S192 = S192.zero();

    inline for (0..SECOND_GROUP_SIZE) |i| {
        const c = coeffs[i];
        const a = az_int[i];
        if (a != 0) {
            az_j += c * @as(i32, a);
        } else {
            S192.fmaddI32(&bz_j, c, bz_s192[i]);
        }
    }

    if (az_j == 0) return S192.zero();
    return bz_j.mulI32(az_j);
}

// ============================================================================
// Tests
// ============================================================================

test "az first group from witness" {
    const field = @import("zolt_arith").field;
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
    const field = @import("zolt_arith").field;
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
    const field = @import("zolt_arith").field;
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
    const field = @import("zolt_arith").field;
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
    const field = @import("zolt_arith").field;
    const F = field.BN254Scalar;

    try std.testing.expect(fieldFromI64(F, 0).eql(F.zero()));
    try std.testing.expect(fieldFromI64(F, 1).eql(F.one()));
    try std.testing.expect(fieldFromI64(F, 5).eql(F.fromU64(5)));

    // Test negative: -3 should equal 0 - 3
    const neg3 = fieldFromI64(F, -3);
    const expected = F.zero().sub(F.fromU64(3));
    try std.testing.expect(neg3.eql(expected));
}

test "fast Az int matches field Az for LOAD instruction" {
    const field = @import("zolt_arith").field;
    const F = field.BN254Scalar;

    var witness: [R1CSInputIndex.NUM_INPUTS]F = [_]F{F.zero()} ** R1CSInputIndex.NUM_INPUTS;
    witness[R1CSInputIndex.FlagLoad.toIndex()] = F.one();

    // Compare field-based Az with integer-based Az
    const az_field = AzFirstGroup(F).fromWitness(&witness);
    const az_int = computeAzFirstGroupInt(F, &witness);

    for (0..FIRST_GROUP_SIZE) |i| {
        const expected_field = fieldFromI64(F, @intCast(az_int[i]));
        try std.testing.expect(az_field.values[i].eql(expected_field));
    }
}

test "fast Az int matches field Az for ADD instruction" {
    const field = @import("zolt_arith").field;
    const F = field.BN254Scalar;

    var witness: [R1CSInputIndex.NUM_INPUTS]F = [_]F{F.zero()} ** R1CSInputIndex.NUM_INPUTS;
    witness[R1CSInputIndex.FlagAddOperands.toIndex()] = F.one();

    const az_field = AzFirstGroup(F).fromWitness(&witness);
    const az_int = computeAzFirstGroupInt(F, &witness);

    for (0..FIRST_GROUP_SIZE) |i| {
        const expected_field = fieldFromI64(F, @intCast(az_int[i]));
        try std.testing.expect(az_field.values[i].eql(expected_field));
    }
}

test "fast Bz direct matches field Bz for first group" {
    const field = @import("zolt_arith").field;
    const F = field.BN254Scalar;

    var witness: [R1CSInputIndex.NUM_INPUTS]F = [_]F{F.zero()} ** R1CSInputIndex.NUM_INPUTS;
    witness[R1CSInputIndex.FlagLoad.toIndex()] = F.one();
    witness[R1CSInputIndex.RamAddress.toIndex()] = F.fromU64(1000);
    witness[R1CSInputIndex.RamReadValue.toIndex()] = F.fromU64(42);
    witness[R1CSInputIndex.RamWriteValue.toIndex()] = F.fromU64(42);
    witness[R1CSInputIndex.RdWriteValue.toIndex()] = F.fromU64(42);
    witness[R1CSInputIndex.LeftLookupOperand.toIndex()] = F.fromU64(7);
    witness[R1CSInputIndex.LeftInstructionInput.toIndex()] = F.fromU64(5);
    witness[R1CSInputIndex.LookupOutput.toIndex()] = F.fromU64(1);
    witness[R1CSInputIndex.NextUnexpandedPC.toIndex()] = F.fromU64(104);
    witness[R1CSInputIndex.NextPC.toIndex()] = F.fromU64(11);
    witness[R1CSInputIndex.PC.toIndex()] = F.fromU64(10);

    // Compare field-based Bz with direct Bz
    const bz_field = BzFirstGroup(F).fromWitness(&witness);
    const bz_direct = computeBzFirstGroupDirect(F, &witness);

    for (0..FIRST_GROUP_SIZE) |i| {
        try std.testing.expect(bz_field.values[i].eql(bz_direct[i]));
    }
}

test "fast second-group Az int matches field Az" {
    const field = @import("zolt_arith").field;
    const F = field.BN254Scalar;

    var witness: [R1CSInputIndex.NUM_INPUTS]F = [_]F{F.zero()} ** R1CSInputIndex.NUM_INPUTS;
    witness[R1CSInputIndex.FlagLoad.toIndex()] = F.one();
    witness[R1CSInputIndex.FlagAddOperands.toIndex()] = F.zero();
    witness[R1CSInputIndex.ShouldBranch.toIndex()] = F.zero();

    const az_field = AzSecondGroup(F).fromWitness(&witness);
    const az_int = computeAzSecondGroupInt(F, &witness);

    for (0..SECOND_GROUP_SIZE) |i| {
        const expected_field = fieldFromI64(F, @intCast(az_int[i]));
        try std.testing.expect(az_field.values[i].eql(expected_field));
    }
}

test "fast second-group Bz direct matches field Bz" {
    const field = @import("zolt_arith").field;
    const F = field.BN254Scalar;

    // Build 2^64 field element for the test
    var bytes: [16]u8 = undefined;
    std.mem.writeInt(u128, &bytes, 0x10000000000000000, .little);
    const two_pow_64 = F.fromBytes(&bytes);

    var witness: [R1CSInputIndex.NUM_INPUTS]F = [_]F{F.zero()} ** R1CSInputIndex.NUM_INPUTS;
    witness[R1CSInputIndex.RamAddress.toIndex()] = F.fromU64(1000);
    witness[R1CSInputIndex.Rs1Value.toIndex()] = F.fromU64(900);
    witness[R1CSInputIndex.Imm.toIndex()] = F.fromU64(100);
    witness[R1CSInputIndex.LeftInstructionInput.toIndex()] = F.fromU64(50);
    witness[R1CSInputIndex.RightInstructionInput.toIndex()] = F.fromU64(30);
    witness[R1CSInputIndex.RightLookupOperand.toIndex()] = F.fromU64(80);
    witness[R1CSInputIndex.Product.toIndex()] = F.fromU64(1500);
    witness[R1CSInputIndex.RdWriteValue.toIndex()] = F.fromU64(200);
    witness[R1CSInputIndex.LookupOutput.toIndex()] = F.fromU64(200);
    witness[R1CSInputIndex.UnexpandedPC.toIndex()] = F.fromU64(100);
    witness[R1CSInputIndex.NextUnexpandedPC.toIndex()] = F.fromU64(104);

    const bz_field = BzSecondGroup(F).fromWitness(&witness);
    const bz_direct = computeBzSecondGroupDirect(F, &witness, two_pow_64);

    for (0..SECOND_GROUP_SIZE) |i| {
        try std.testing.expect(bz_field.values[i].eql(bz_direct[i]));
    }
}

test "interpolateAzBzProductInt matches brute force for satisfied constraints" {
    const univariate_skip = @import("univariate_skip.zig");

    // Simulate a satisfied constraint: when az[i] != 0, bz[i] must be 0.
    // This is the invariant that interpolateAzBzProductInt exploits.
    const az = [FIRST_GROUP_SIZE]i8{ 1, 0, 0, -1, 0, 0, 0, 2, 0, 0 };
    const bz = [FIRST_GROUP_SIZE]i128{
        0, // az[0]=1, so bz must be 0 (satisfied)
        42, // az[1]=0, so bz can be non-zero
        -100, // az[2]=0
        0, // az[3]=-1, so bz must be 0 (satisfied)
        999, // az[4]=0
        -@as(i128, 0x7FFFFFFF_FFFFFFFF), // large negative
        @as(i128, 0xFFFFFFFF_FFFFFFFE), // large positive
        0, // az[7]=2, so bz must be 0 (satisfied)
        0, // az[8]=0
        12345, // az[9]=0
    };

    for (0..univariate_skip.OUTER_UNIVARIATE_SKIP_DEGREE) |j| {
        const coeffs = &univariate_skip.COEFFS_PER_J[j];

        // Brute-force: az_j = Σ c[i]*az[i], bz_j = Σ c[i]*bz[i], product = az_j * bz_j
        var az_j_bf: i64 = 0;
        var bz_j_bf: i128 = 0;
        for (0..FIRST_GROUP_SIZE) |i| {
            az_j_bf += @as(i64, coeffs[i]) * @as(i64, az[i]);
            bz_j_bf += @as(i128, coeffs[i]) * bz[i];
        }
        const bf_product = @as(i128, az_j_bf) * bz_j_bf;

        // Optimized path
        const opt_product = interpolateAzBzProductInt(&az, &bz, coeffs);

        // Must match because az[i]!=0 implies bz[i]==0
        try std.testing.expectEqual(bf_product, opt_product);
    }
}

test "interpolateAzBzProductInt matches field path for satisfied constraints" {
    const field = @import("zolt_arith").field;
    const F = field.BN254Scalar;
    const univariate_skip = @import("univariate_skip.zig");

    // Same satisfied-constraint data, but compare against field-arithmetic path
    const az = [FIRST_GROUP_SIZE]i8{ 0, 1, 0, 0, -1, 0, 0, 0, 3, 0 };
    const bz_int = [FIRST_GROUP_SIZE]i128{
        @as(i128, 0xFFFFFFFF_FFFFFFFE), // large value (az=0)
        0, // az=1, bz must be 0
        -55555, // az=0
        @as(i128, 1) << 64, // 2^64 (az=0)
        0, // az=-1, bz must be 0
        42, // az=0
        -1, // az=0
        0, // az=0
        0, // az=3, bz must be 0
        999999, // az=0
    };

    // Convert bz_int to field values for the reference path
    var bz_field: [FIRST_GROUP_SIZE]F = undefined;
    for (0..FIRST_GROUP_SIZE) |i| {
        bz_field[i] = if (bz_int[i] >= 0)
            F.fromU128(@intCast(bz_int[i]))
        else
            F.fromU128(@intCast(-bz_int[i])).neg();
    }

    for (0..univariate_skip.OUTER_UNIVARIATE_SKIP_DEGREE) |j| {
        const coeffs = &univariate_skip.COEFFS_PER_J[j];

        // Field-based reference
        const field_product = interpolateAzBzProduct(F, &az, &bz_field, coeffs, FIRST_GROUP_SIZE);

        // Integer-based optimized path
        const int_product = interpolateAzBzProductInt(&az, &bz_int, coeffs);
        const int_as_field = if (int_product >= 0)
            F.fromU128(@intCast(int_product))
        else
            F.fromU128(@intCast(-int_product)).neg();

        try std.testing.expect(field_product.eql(int_as_field));
    }
}

test "compact witness az/bz match field witnesses" {
    const field = @import("zolt_arith").field;
    const F = field.BN254Scalar;

    var witness: [R1CSInputIndex.NUM_INPUTS]F = [_]F{F.zero()} ** R1CSInputIndex.NUM_INPUTS;
    witness[R1CSInputIndex.FlagLoad.toIndex()] = F.one();
    witness[R1CSInputIndex.RamAddress.toIndex()] = F.fromU64(500);
    witness[R1CSInputIndex.RamReadValue.toIndex()] = F.fromU64(42);
    witness[R1CSInputIndex.RamWriteValue.toIndex()] = F.fromU64(42);

    const cw = compactFromFieldWitness(F, &witness);

    // Verify az_first matches computeAzFirstGroupInt
    const az_ref = computeAzFirstGroupInt(F, &witness);
    for (0..FIRST_GROUP_SIZE) |i| {
        try std.testing.expectEqual(az_ref[i], cw.az_first[i]);
    }

    // Verify az_second matches computeAzSecondGroupInt
    const az2_ref = computeAzSecondGroupInt(F, &witness);
    for (0..SECOND_GROUP_SIZE) |i| {
        try std.testing.expectEqual(az2_ref[i], cw.az_second[i]);
    }

    // Verify bz_first integer values match field Bz when converted back
    const bz_ref = computeBzFirstGroupDirect(F, &witness);
    for (0..FIRST_GROUP_SIZE) |i| {
        const bz_i = cw.bz_first[i];
        const bz_f = if (bz_i >= 0)
            F.fromU128(@as(u128, @intCast(bz_i)))
        else
            F.fromU128(@as(u128, @intCast(-bz_i))).neg();
        try std.testing.expect(bz_ref[i].eql(bz_f));
    }
}

test "compact witness bz_second with large u128 values" {
    const field = @import("zolt_arith").field;
    const F = field.BN254Scalar;

    // Create witness with RightLookupOperand > 2^127 to exercise @bitCast path
    var witness: [R1CSInputIndex.NUM_INPUTS]F = [_]F{F.zero()} ** R1CSInputIndex.NUM_INPUTS;

    // Set a large RightLookupOperand (> 2^127, would overflow @intCast)
    const large_val: u128 = (1 << 127) + 42;
    const large_f = F.fromU128(@truncate(large_val));
    witness[R1CSInputIndex.RightLookupOperand.toIndex()] = large_f;
    witness[R1CSInputIndex.LeftInstructionInput.toIndex()] = F.fromU64(100);
    witness[R1CSInputIndex.RightInstructionInput.toIndex()] = F.fromU64(200);

    // This should not panic (the bug was @intCast trapping for values >= 2^127)
    const cw = compactFromFieldWitness(F, &witness);

    // Verify bz_second[1] = RightLookup - LeftInput - RightInput (wrapping)
    const large_as_i128: i128 = @bitCast(large_val);
    const expected_i128: i128 = large_as_i128 -% 100 -% 200;
    const expected_s192 = field_mod.S192.fromI128(expected_i128);
    try std.testing.expectEqual(expected_s192, cw.bz_second[1]);

    // Verify round-trip through fieldFromI128 produces correct field element
    const field_val = fieldFromI128(F, expected_i128);
    // The field value should equal F(right_lookup) - F(100) - F(200)
    const expected_field = large_f.sub(F.fromU64(100)).sub(F.fromU64(200));
    try std.testing.expect(field_val.eql(expected_field));
}
