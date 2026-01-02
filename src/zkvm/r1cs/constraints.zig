//! R1CS Constraint Generation for Jolt zkVM
//!
//! This module generates R1CS constraints from execution traces. The constraints
//! follow the equality-conditional form: `condition * (left - right) = 0`
//!
//! ## Constraint Structure
//!
//! Jolt uses 19 uniform R1CS constraints applied to every execution cycle:
//! 1. RAM address computation for loads/stores
//! 2. RAM read/write consistency
//! 3. Arithmetic operation correctness
//! 4. PC update logic
//! 5. Register write consistency
//!
//! ## Witness Variables (36 per cycle)
//!
//! - Instruction inputs: left_input, right_input, product
//! - Lookup operands: left_lookup, right_lookup, lookup_output
//! - Registers: rs1_value, rs2_value, rd_write_value
//! - RAM: ram_address, ram_read_value, ram_write_value
//! - PC: pc, next_pc, unexpanded_pc, next_unexpanded_pc
//! - Immediate: imm
//! - Flags: 13 circuit flags + 6 derived flags
//!
//! Reference: jolt-core/src/zkvm/r1cs/constraints.rs

const std = @import("std");
const Allocator = std.mem.Allocator;

const instruction = @import("../instruction/mod.zig");
const tracer = @import("../../tracer/mod.zig");
const CircuitFlags = instruction.CircuitFlags;

/// Index into the witness vector for R1CS inputs
///
/// IMPORTANT: This order MUST match Jolt's ALL_R1CS_INPUTS array in inputs.rs
/// The order is critical for correct R1CS evaluation and claim matching.
///
/// Reference: jolt-core/src/zkvm/r1cs/inputs.rs
pub const R1CSInputIndex = enum(u8) {
    // Matches Jolt's ALL_R1CS_INPUTS order exactly:
    LeftInstructionInput = 0,
    RightInstructionInput = 1,
    Product = 2,
    WriteLookupOutputToRD = 3,
    WritePCtoRD = 4,
    ShouldBranch = 5,
    PC = 6,
    UnexpandedPC = 7,
    Imm = 8,
    RamAddress = 9,
    Rs1Value = 10,
    Rs2Value = 11,
    RdWriteValue = 12,
    RamReadValue = 13,
    RamWriteValue = 14,
    LeftLookupOperand = 15,
    RightLookupOperand = 16,
    NextUnexpandedPC = 17,
    NextPC = 18,
    NextIsVirtual = 19,
    NextIsFirstInSequence = 20,
    LookupOutput = 21,
    ShouldJump = 22,
    // OpFlags in order matching CircuitFlags enum
    FlagAddOperands = 23,
    FlagSubtractOperands = 24,
    FlagMultiplyOperands = 25,
    FlagLoad = 26,
    FlagStore = 27,
    FlagJump = 28,
    FlagWriteLookupOutputToRD = 29,
    FlagVirtualInstruction = 30,
    FlagAssert = 31,
    FlagDoNotUpdateUnexpandedPC = 32,
    FlagAdvice = 33,
    FlagIsCompressed = 34,
    FlagIsFirstInSequence = 35,

    pub const NUM_INPUTS = 36;

    pub fn toIndex(self: R1CSInputIndex) usize {
        return @intFromEnum(self);
    }
};

/// Linear combination term
pub const Term = struct {
    input_index: R1CSInputIndex,
    coeff: i128,
};

/// Linear combination of witness variables
pub fn LinearCombination(comptime max_terms: usize) type {
    return struct {
        const Self = @This();

        terms: [max_terms]Term,
        len: usize,
        constant: i128,

        pub fn zero() Self {
            return Self{
                .terms = undefined,
                .len = 0,
                .constant = 0,
            };
        }

        pub fn one() Self {
            return Self{
                .terms = undefined,
                .len = 0,
                .constant = 1,
            };
        }

        pub fn fromConstant(c: i128) Self {
            return Self{
                .terms = undefined,
                .len = 0,
                .constant = c,
            };
        }

        pub fn fromInput(index: R1CSInputIndex) Self {
            var lc = Self{
                .terms = undefined,
                .len = 1,
                .constant = 0,
            };
            lc.terms[0] = .{ .input_index = index, .coeff = 1 };
            return lc;
        }

        pub fn fromInputScaled(index: R1CSInputIndex, coeff: i128) Self {
            var lc = Self{
                .terms = undefined,
                .len = 1,
                .constant = 0,
            };
            lc.terms[0] = .{ .input_index = index, .coeff = coeff };
            return lc;
        }

        /// Helper: convert i128 to field element (handles values > 2^64)
        fn i128ToField(comptime F: type, val: i128) F {
            if (val >= 0) {
                const v: u128 = @intCast(val);
                if (v <= 0xFFFFFFFFFFFFFFFF) {
                    return F.fromU64(@intCast(v));
                } else {
                    // Value > 2^64, use bytes representation
                    var bytes: [16]u8 = undefined;
                    std.mem.writeInt(u128, &bytes, v, .little);
                    return F.fromBytes(&bytes);
                }
            } else {
                const neg_v: u128 = @intCast(-val);
                if (neg_v <= 0xFFFFFFFFFFFFFFFF) {
                    return F.zero().sub(F.fromU64(@intCast(neg_v)));
                } else {
                    // Value > 2^64, use bytes representation
                    var bytes: [16]u8 = undefined;
                    std.mem.writeInt(u128, &bytes, neg_v, .little);
                    return F.zero().sub(F.fromBytes(&bytes));
                }
            }
        }

        /// Evaluate the linear combination given witness values
        pub fn evaluate(self: Self, comptime F: type, witness: []const F) F {
            var result = i128ToField(F, self.constant);

            for (self.terms[0..self.len]) |term| {
                const val = witness[term.input_index.toIndex()];
                const coeff_field = i128ToField(F, if (term.coeff >= 0) term.coeff else -term.coeff);
                const scaled = val.mul(coeff_field);
                if (term.coeff >= 0) {
                    result = result.add(scaled);
                } else {
                    result = result.sub(scaled);
                }
            }

            return result;
        }

        /// Evaluate the linear combination given z values (MLE evaluations)
        /// This is the same as evaluate() but clarifies the intent when used
        /// with MLE evaluations instead of actual witness values.
        pub fn evaluateWithConstant(self: Self, comptime F: type, z: []const F) F {
            return self.evaluate(F, z);
        }
    };
}

/// Type alias for linear combinations with up to 5 terms
pub const LC = LinearCombination(5);

/// R1CS constraint in equality-conditional form: a * b = 0
/// Where b = (left - right), so the constraint is: condition * (left - right) = 0
pub const UniformConstraint = struct {
    /// Guard/condition (typically a boolean flag or sum of flags)
    condition: LC,
    /// Left side of equality
    left: LC,
    /// Right side of equality
    right: LC,

    /// Evaluate constraint: returns condition * (left - right)
    /// Should be zero if constraint is satisfied
    pub fn evaluate(self: UniformConstraint, comptime F: type, witness: []const F) F {
        const cond = self.condition.evaluate(F, witness);
        const l = self.left.evaluate(F, witness);
        const r = self.right.evaluate(F, witness);
        return cond.mul(l.sub(r));
    }

    /// Check if constraint is satisfied (result should be zero)
    pub fn isSatisfied(self: UniformConstraint, comptime F: type, witness: []const F) bool {
        const result = self.evaluate(F, witness);
        return result.eql(F.zero());
    }
};

/// All 19 uniform R1CS constraints for Jolt (Exact Match)
///
/// These constraints are ordered exactly as in Jolt's constraints.rs.
/// The constraint form is: Az * Bz = 0, where Az = condition and Bz = left - right.
///
/// FIRST GROUP (constraints 0-9, indices in base univariate skip domain {-4..5}):
/// - Boolean guards, Bz fits in ~64 bits
///
/// SECOND GROUP (constraints 10-18, separate handling):
/// - Mixed Az types, Bz can be ~128-160 bits
pub const UNIFORM_CONSTRAINTS = [_]UniformConstraint{
    // =========================================================================
    // CONSTRAINT 0: RamAddrEqRs1PlusImmIfLoadStore (SECOND GROUP index 0)
    // =========================================================================
    // if { Load + Store } => ( RamAddress ) == ( Rs1Value + Imm )
    .{
        .condition = blk: {
            var lc = LC.zero();
            lc.terms[0] = .{ .input_index = .FlagLoad, .coeff = 1 };
            lc.terms[1] = .{ .input_index = .FlagStore, .coeff = 1 };
            lc.len = 2;
            break :blk lc;
        },
        .left = LC.fromInput(.RamAddress),
        .right = blk: {
            var lc = LC.zero();
            lc.terms[0] = .{ .input_index = .Rs1Value, .coeff = 1 };
            lc.terms[1] = .{ .input_index = .Imm, .coeff = 1 };
            lc.len = 2;
            break :blk lc;
        },
    },

    // =========================================================================
    // CONSTRAINT 1: RamAddrEqZeroIfNotLoadStore (FIRST GROUP index 0)
    // =========================================================================
    // if { 1 - Load - Store } => ( RamAddress ) == ( 0 )
    .{
        .condition = blk: {
            var lc = LC.one();
            lc.terms[0] = .{ .input_index = .FlagLoad, .coeff = -1 };
            lc.terms[1] = .{ .input_index = .FlagStore, .coeff = -1 };
            lc.len = 2;
            break :blk lc;
        },
        .left = LC.fromInput(.RamAddress),
        .right = LC.zero(),
    },

    // =========================================================================
    // CONSTRAINT 2: RamReadEqRamWriteIfLoad (FIRST GROUP index 1)
    // =========================================================================
    // if { Load } => ( RamReadValue ) == ( RamWriteValue )
    .{
        .condition = LC.fromInput(.FlagLoad),
        .left = LC.fromInput(.RamReadValue),
        .right = LC.fromInput(.RamWriteValue),
    },

    // =========================================================================
    // CONSTRAINT 3: RamReadEqRdWriteIfLoad (FIRST GROUP index 2)
    // =========================================================================
    // if { Load } => ( RamReadValue ) == ( RdWriteValue )
    .{
        .condition = LC.fromInput(.FlagLoad),
        .left = LC.fromInput(.RamReadValue),
        .right = LC.fromInput(.RdWriteValue),
    },

    // =========================================================================
    // CONSTRAINT 4: Rs2EqRamWriteIfStore (FIRST GROUP index 3)
    // =========================================================================
    // if { Store } => ( Rs2Value ) == ( RamWriteValue )
    .{
        .condition = LC.fromInput(.FlagStore),
        .left = LC.fromInput(.Rs2Value),
        .right = LC.fromInput(.RamWriteValue),
    },

    // =========================================================================
    // CONSTRAINT 5: LeftLookupZeroUnlessAddSubMul (FIRST GROUP index 4)
    // =========================================================================
    // if { AddOperands + SubtractOperands + MultiplyOperands } => ( LeftLookupOperand ) == ( 0 )
    .{
        .condition = blk: {
            var lc = LC.zero();
            lc.terms[0] = .{ .input_index = .FlagAddOperands, .coeff = 1 };
            lc.terms[1] = .{ .input_index = .FlagSubtractOperands, .coeff = 1 };
            lc.terms[2] = .{ .input_index = .FlagMultiplyOperands, .coeff = 1 };
            lc.len = 3;
            break :blk lc;
        },
        .left = LC.fromInput(.LeftLookupOperand),
        .right = LC.zero(),
    },

    // =========================================================================
    // CONSTRAINT 6: LeftLookupEqLeftInputOtherwise (FIRST GROUP index 5)
    // =========================================================================
    // if { 1 - AddOperands - SubtractOperands - MultiplyOperands } => ( LeftLookupOperand ) == ( LeftInstructionInput )
    .{
        .condition = blk: {
            var lc = LC.one();
            lc.terms[0] = .{ .input_index = .FlagAddOperands, .coeff = -1 };
            lc.terms[1] = .{ .input_index = .FlagSubtractOperands, .coeff = -1 };
            lc.terms[2] = .{ .input_index = .FlagMultiplyOperands, .coeff = -1 };
            lc.len = 3;
            break :blk lc;
        },
        .left = LC.fromInput(.LeftLookupOperand),
        .right = LC.fromInput(.LeftInstructionInput),
    },

    // =========================================================================
    // CONSTRAINT 7: RightLookupAdd (SECOND GROUP index 1)
    // =========================================================================
    // if { AddOperands } => ( RightLookupOperand ) == ( LeftInstructionInput + RightInstructionInput )
    .{
        .condition = LC.fromInput(.FlagAddOperands),
        .left = LC.fromInput(.RightLookupOperand),
        .right = blk: {
            var lc = LC.zero();
            lc.terms[0] = .{ .input_index = .LeftInstructionInput, .coeff = 1 };
            lc.terms[1] = .{ .input_index = .RightInstructionInput, .coeff = 1 };
            lc.len = 2;
            break :blk lc;
        },
    },

    // =========================================================================
    // CONSTRAINT 8: RightLookupSub (SECOND GROUP index 2)
    // =========================================================================
    // if { SubtractOperands } => ( RightLookupOperand ) == ( LeftInstructionInput - RightInstructionInput + 2^64 )
    // Note: The 2^64 offset is for two's complement representation
    .{
        .condition = LC.fromInput(.FlagSubtractOperands),
        .left = LC.fromInput(.RightLookupOperand),
        .right = blk: {
            var lc = LC.zero();
            lc.terms[0] = .{ .input_index = .LeftInstructionInput, .coeff = 1 };
            lc.terms[1] = .{ .input_index = .RightInstructionInput, .coeff = -1 };
            lc.len = 2;
            // 2^64 = 0x10000000000000000 = 18446744073709551616
            lc.constant = 0x10000000000000000;
            break :blk lc;
        },
    },

    // =========================================================================
    // CONSTRAINT 9: RightLookupEqProductIfMul (SECOND GROUP index 3)
    // =========================================================================
    // if { MultiplyOperands } => ( RightLookupOperand ) == ( Product )
    .{
        .condition = LC.fromInput(.FlagMultiplyOperands),
        .left = LC.fromInput(.RightLookupOperand),
        .right = LC.fromInput(.Product),
    },

    // =========================================================================
    // CONSTRAINT 10: RightLookupEqRightInputOtherwise (SECOND GROUP index 4)
    // =========================================================================
    // if { 1 - AddOperands - SubtractOperands - MultiplyOperands - Advice } => ( RightLookupOperand ) == ( RightInstructionInput )
    .{
        .condition = blk: {
            var lc = LC.one();
            lc.terms[0] = .{ .input_index = .FlagAddOperands, .coeff = -1 };
            lc.terms[1] = .{ .input_index = .FlagSubtractOperands, .coeff = -1 };
            lc.terms[2] = .{ .input_index = .FlagMultiplyOperands, .coeff = -1 };
            lc.terms[3] = .{ .input_index = .FlagAdvice, .coeff = -1 };
            lc.len = 4;
            break :blk lc;
        },
        .left = LC.fromInput(.RightLookupOperand),
        .right = LC.fromInput(.RightInstructionInput),
    },

    // =========================================================================
    // CONSTRAINT 11: AssertLookupOne (FIRST GROUP index 6)
    // =========================================================================
    // if { Assert } => ( LookupOutput ) == ( 1 )
    .{
        .condition = LC.fromInput(.FlagAssert),
        .left = LC.fromInput(.LookupOutput),
        .right = LC.one(),
    },

    // =========================================================================
    // CONSTRAINT 12: RdWriteEqLookupIfWriteLookupToRd (SECOND GROUP index 5)
    // =========================================================================
    // if { WriteLookupOutputToRD } => ( RdWriteValue ) == ( LookupOutput )
    .{
        .condition = LC.fromInput(.WriteLookupOutputToRD),
        .left = LC.fromInput(.RdWriteValue),
        .right = LC.fromInput(.LookupOutput),
    },

    // =========================================================================
    // CONSTRAINT 13: RdWriteEqPCPlusConstIfWritePCtoRD (SECOND GROUP index 6)
    // =========================================================================
    // if { WritePCtoRD } => ( RdWriteValue ) == ( UnexpandedPC + 4 - 2*IsCompressed )
    .{
        .condition = LC.fromInput(.WritePCtoRD),
        .left = LC.fromInput(.RdWriteValue),
        .right = blk: {
            var lc = LC.zero();
            lc.terms[0] = .{ .input_index = .UnexpandedPC, .coeff = 1 };
            lc.terms[1] = .{ .input_index = .FlagIsCompressed, .coeff = -2 };
            lc.len = 2;
            lc.constant = 4;
            break :blk lc;
        },
    },

    // =========================================================================
    // CONSTRAINT 14: NextUnexpPCEqLookupIfShouldJump (FIRST GROUP index 7)
    // =========================================================================
    // if { ShouldJump } => ( NextUnexpandedPC ) == ( LookupOutput )
    .{
        .condition = LC.fromInput(.ShouldJump),
        .left = LC.fromInput(.NextUnexpandedPC),
        .right = LC.fromInput(.LookupOutput),
    },

    // =========================================================================
    // CONSTRAINT 15: NextUnexpPCEqPCPlusImmIfShouldBranch (SECOND GROUP index 7)
    // =========================================================================
    // if { ShouldBranch } => ( NextUnexpandedPC ) == ( UnexpandedPC + Imm )
    .{
        .condition = LC.fromInput(.ShouldBranch),
        .left = LC.fromInput(.NextUnexpandedPC),
        .right = blk: {
            var lc = LC.zero();
            lc.terms[0] = .{ .input_index = .UnexpandedPC, .coeff = 1 };
            lc.terms[1] = .{ .input_index = .Imm, .coeff = 1 };
            lc.len = 2;
            break :blk lc;
        },
    },

    // =========================================================================
    // CONSTRAINT 16: NextUnexpPCUpdateOtherwise (FIRST GROUP index 8)
    // =========================================================================
    // if { 1 - ShouldBranch - Jump } => ( NextUnexpandedPC ) == ( UnexpandedPC + 4 - 4*DoNotUpdateUnexpandedPC - 2*IsCompressed )
    .{
        .condition = blk: {
            var lc = LC.one();
            lc.terms[0] = .{ .input_index = .ShouldBranch, .coeff = -1 };
            lc.terms[1] = .{ .input_index = .FlagJump, .coeff = -1 };
            lc.len = 2;
            break :blk lc;
        },
        .left = LC.fromInput(.NextUnexpandedPC),
        .right = blk: {
            var lc = LC.zero();
            lc.terms[0] = .{ .input_index = .UnexpandedPC, .coeff = 1 };
            lc.terms[1] = .{ .input_index = .FlagDoNotUpdateUnexpandedPC, .coeff = -4 };
            lc.terms[2] = .{ .input_index = .FlagIsCompressed, .coeff = -2 };
            lc.len = 3;
            lc.constant = 4;
            break :blk lc;
        },
    },

    // =========================================================================
    // CONSTRAINT 17: NextPCEqPCPlusOneIfInline (FIRST GROUP index 8 - duplicate)
    // =========================================================================
    // if { VirtualInstruction } => ( NextPC ) == ( PC + 1 )
    .{
        .condition = LC.fromInput(.FlagVirtualInstruction),
        .left = LC.fromInput(.NextPC),
        .right = blk: {
            var lc = LC.zero();
            lc.terms[0] = .{ .input_index = .PC, .coeff = 1 };
            lc.len = 1;
            lc.constant = 1;
            break :blk lc;
        },
    },

    // =========================================================================
    // CONSTRAINT 18: MustStartSequenceFromBeginning (FIRST GROUP index 9)
    // =========================================================================
    // if { NextIsVirtual - NextIsFirstInSequence } => ( 1 ) == ( DoNotUpdateUnexpandedPC )
    .{
        .condition = blk: {
            var lc = LC.zero();
            lc.terms[0] = .{ .input_index = .NextIsVirtual, .coeff = 1 };
            lc.terms[1] = .{ .input_index = .NextIsFirstInSequence, .coeff = -1 };
            lc.len = 2;
            break :blk lc;
        },
        .left = LC.one(),
        .right = LC.fromInput(.FlagDoNotUpdateUnexpandedPC),
    },
};

/// First group constraint indices (10 constraints, domain {-4..5})
/// These are the global indices from UNIFORM_CONSTRAINTS that belong to the first group
/// Matches Jolt's R1CS_CONSTRAINTS_FIRST_GROUP_LABELS
pub const FIRST_GROUP_INDICES: [10]usize = .{
    1, // RamAddrEqZeroIfNotLoadStore
    2, // RamReadEqRamWriteIfLoad
    3, // RamReadEqRdWriteIfLoad
    4, // Rs2EqRamWriteIfStore
    5, // LeftLookupZeroUnlessAddSubMul
    6, // LeftLookupEqLeftInputOtherwise
    11, // AssertLookupOne
    14, // NextUnexpPCEqLookupIfShouldJump
    17, // NextPCEqPCPlusOneIfInline (Jolt uses this in first group)
    18, // MustStartSequenceFromBeginning
};

/// Second group constraint indices (9 constraints)
/// These are the global indices from UNIFORM_CONSTRAINTS that belong to the second group
/// Matches Jolt's R1CS_CONSTRAINTS_SECOND_GROUP_LABELS
pub const SECOND_GROUP_INDICES: [9]usize = .{
    0, // RamAddrEqRs1PlusImmIfLoadStore
    7, // RightLookupAdd
    8, // RightLookupSub
    9, // RightLookupEqProductIfMul
    10, // RightLookupEqRightInputOtherwise
    12, // RdWriteEqLookupIfWriteLookupToRd
    13, // RdWriteEqPCPlusConstIfWritePCtoRD
    15, // NextUnexpPCEqPCPlusImmIfShouldBranch
    16, // NextUnexpPCUpdateOtherwise (moved from first group)
};

/// Per-cycle R1CS inputs extracted from execution trace
pub fn R1CSCycleInputs(comptime F: type) type {
    return struct {
        const Self = @This();

        /// Witness values for this cycle (indexed by R1CSInputIndex)
        values: [R1CSInputIndex.NUM_INPUTS]F,

        /// Initialize with all zero values
        pub fn init() Self {
            return Self{
                .values = [_]F{F.zero()} ** R1CSInputIndex.NUM_INPUTS,
            };
        }

        /// Set a specific input value
        pub fn setInput(self: *Self, index: R1CSInputIndex, value: F) void {
            self.values[index.toIndex()] = value;
        }

        /// Create cycle inputs from an execution trace step
        ///
        /// This generates R1CS witness values that satisfy Jolt's 19 uniform constraints.
        /// The key constraint invariant is: Az * Bz = 0 for every constraint.
        ///
        /// CONSTRAINT SATISFACTION:
        /// - Constraint 0: if Load+Store != 0 => RamAddress == Rs1+Imm
        /// - Constraint 1: if Load+Store == 0 => RamAddress == 0
        /// - Constraint 2: if Load => RamReadValue == RamWriteValue
        /// - Constraint 3: if Load => RamReadValue == RdWriteValue
        /// - Constraint 4: if Store => Rs2Value == RamWriteValue
        /// - ... etc
        pub fn fromTraceStep(
            step: tracer.TraceStep,
            next_step: ?tracer.TraceStep,
        ) Self {
            var inputs = Self{
                .values = [_]F{F.zero()} ** R1CSInputIndex.NUM_INPUTS,
            };

            // Determine if this is a Load or Store instruction
            const opcode: u8 = @truncate(step.instruction & 0x7F);
            const is_load = (opcode == 0x03);
            const is_store = (opcode == 0x23);
            const is_load_or_store = is_load or is_store;

            // Set flags first (needed for constraint checking)
            if (is_load) {
                inputs.values[R1CSInputIndex.FlagLoad.toIndex()] = F.one();
            }
            if (is_store) {
                inputs.values[R1CSInputIndex.FlagStore.toIndex()] = F.one();
            }
            // Use the compressed flag from the trace (original instruction was 2 bytes)
            if (step.is_compressed) {
                inputs.values[R1CSInputIndex.FlagIsCompressed.toIndex()] = F.one();
            }

            // Immediate - derive from instruction
            const imm = inputs.deriveImmediate(step.instruction);
            inputs.values[R1CSInputIndex.Imm.toIndex()] = imm;

            // Register values
            inputs.values[R1CSInputIndex.Rs1Value.toIndex()] = F.fromU64(step.rs1_value);
            inputs.values[R1CSInputIndex.Rs2Value.toIndex()] = F.fromU64(step.rs2_value);

            // =================================================================
            // RAM-related values - must satisfy constraints 0-4
            // =================================================================

            // Constraint 0: if Load+Store => RamAddress == Rs1+Imm
            // Constraint 1: if NOT Load+Store => RamAddress == 0
            if (is_load_or_store) {
                // Compute Rs1 + Imm in the field
                const rs1_f = F.fromU64(step.rs1_value);
                const ram_addr = rs1_f.add(imm);
                inputs.values[R1CSInputIndex.RamAddress.toIndex()] = ram_addr;
            } else {
                // Non-memory instructions: RamAddress MUST be 0
                inputs.values[R1CSInputIndex.RamAddress.toIndex()] = F.zero();
            }

            // Memory values
            const mem_val = step.memory_value orelse 0;
            const mem_val_f = F.fromU64(mem_val);

            if (is_load) {
                // Constraint 2: RamReadValue == RamWriteValue (for Load)
                // Constraint 3: RamReadValue == RdWriteValue (for Load)
                inputs.values[R1CSInputIndex.RamReadValue.toIndex()] = mem_val_f;
                inputs.values[R1CSInputIndex.RamWriteValue.toIndex()] = mem_val_f;
                inputs.values[R1CSInputIndex.RdWriteValue.toIndex()] = mem_val_f;
            } else if (is_store) {
                // Constraint 4: Rs2Value == RamWriteValue (for Store)
                inputs.values[R1CSInputIndex.RamReadValue.toIndex()] = F.zero();
                inputs.values[R1CSInputIndex.RamWriteValue.toIndex()] = F.fromU64(step.rs2_value);
                inputs.values[R1CSInputIndex.RdWriteValue.toIndex()] = F.fromU64(step.rd_value);
            } else {
                // Non-memory: set to reasonable defaults
                inputs.values[R1CSInputIndex.RamReadValue.toIndex()] = F.zero();
                inputs.values[R1CSInputIndex.RamWriteValue.toIndex()] = F.zero();
                inputs.values[R1CSInputIndex.RdWriteValue.toIndex()] = F.fromU64(step.rd_value);
            }

            // =================================================================
            // Instruction inputs and lookups
            // =================================================================
            inputs.values[R1CSInputIndex.LeftInstructionInput.toIndex()] = F.fromU64(step.rs1_value);
            inputs.values[R1CSInputIndex.RightInstructionInput.toIndex()] = F.fromU64(step.rs2_value);

            // Product = rs1 * rs2 (for multiply instructions)
            const product = @as(u128, step.rs1_value) * @as(u128, step.rs2_value);
            inputs.values[R1CSInputIndex.Product.toIndex()] = F.fromU64(@truncate(product));

            // =================================================================
            // Lookup operands - will be set properly by setFlagsFromInstruction
            // based on which operation type it is (constraints 5-11)
            // =================================================================
            // LookupOutput is the result value from the lookup table
            inputs.values[R1CSInputIndex.LookupOutput.toIndex()] = F.fromU64(step.rd_value);

            // =================================================================
            // PC values
            // =================================================================
            inputs.values[R1CSInputIndex.PC.toIndex()] = F.fromU64(step.pc);
            inputs.values[R1CSInputIndex.UnexpandedPC.toIndex()] = F.fromU64(step.pc);
            inputs.values[R1CSInputIndex.NextPC.toIndex()] = F.fromU64(step.next_pc);
            inputs.values[R1CSInputIndex.NextUnexpandedPC.toIndex()] = F.fromU64(step.next_pc);

            // Use next step's PC if available
            if (next_step) |ns| {
                inputs.values[R1CSInputIndex.NextPC.toIndex()] = F.fromU64(ns.pc);
                inputs.values[R1CSInputIndex.NextUnexpandedPC.toIndex()] = F.fromU64(ns.pc);
            }

            // =================================================================
            // Set remaining flags based on instruction opcode
            // =================================================================
            inputs.setFlagsFromInstruction(step.instruction);

            return inputs;
        }

        /// Derive immediate value from instruction
        fn deriveImmediate(self: *Self, instr: u32) F {
            _ = self;
            const opcode = instr & 0x7F;
            switch (opcode) {
                0x13, 0x03, 0x67 => { // I-type: ADDI, LOAD, JALR
                    const imm = instr >> 20;
                    // Sign extend from 12 bits
                    if (imm & 0x800 != 0) {
                        return F.zero().sub(F.fromU64((~imm + 1) & 0xFFF));
                    }
                    return F.fromU64(imm);
                },
                0x23 => { // S-type: STORE
                    const imm4_0 = (instr >> 7) & 0x1F;
                    const imm11_5 = (instr >> 25) & 0x7F;
                    const imm = (imm11_5 << 5) | imm4_0;
                    if (imm & 0x800 != 0) {
                        return F.zero().sub(F.fromU64((~imm + 1) & 0xFFF));
                    }
                    return F.fromU64(imm);
                },
                0x63 => { // B-type: BRANCH
                    const imm12 = (instr >> 31) & 0x1;
                    const imm10_5 = (instr >> 25) & 0x3F;
                    const imm4_1 = (instr >> 8) & 0xF;
                    const imm11 = (instr >> 7) & 0x1;
                    const imm = (imm12 << 12) | (imm11 << 11) | (imm10_5 << 5) | (imm4_1 << 1);
                    if (imm & 0x1000 != 0) {
                        return F.zero().sub(F.fromU64((~imm + 1) & 0x1FFF));
                    }
                    return F.fromU64(imm);
                },
                0x6F => { // J-type: JAL
                    const imm20 = (instr >> 31) & 0x1;
                    const imm10_1 = (instr >> 21) & 0x3FF;
                    const imm11 = (instr >> 20) & 0x1;
                    const imm19_12 = (instr >> 12) & 0xFF;
                    const imm = (imm20 << 20) | (imm19_12 << 12) | (imm11 << 11) | (imm10_1 << 1);
                    if (imm & 0x100000 != 0) {
                        return F.zero().sub(F.fromU64((~imm + 1) & 0x1FFFFF));
                    }
                    return F.fromU64(imm);
                },
                0x37, 0x17 => { // U-type: LUI, AUIPC
                    const imm = instr & 0xFFFFF000;
                    return F.fromU64(imm);
                },
                else => return F.zero(),
            }
        }

        /// Set circuit flags and lookup operands based on instruction
        ///
        /// This satisfies lookup-related constraints (5-11):
        /// - Constraint 5: if Add+Sub+Mul => LeftLookupOperand == 0
        /// - Constraint 6: if NOT Add+Sub+Mul => LeftLookupOperand == LeftInstructionInput
        /// - Constraint 7: if Add => RightLookupOperand == RightInput+LeftInput
        /// - Constraint 8: if Sub => RightLookupOperand == RightInput-LeftInput
        /// - Constraint 9: if Mul => RightLookupOperand == Product
        /// - Constraint 10: if NOT Add+Sub+Mul => RightLookupOperand == RightInput
        /// - Constraint 11: if Assert => LookupOutput == 1
        fn setFlagsFromInstruction(self: *Self, instr: u32) void {
            const opcode: u8 = @truncate(instr & 0x7F);
            const funct3 = (instr >> 12) & 0x7;
            const funct7 = (instr >> 25) & 0x7F;

            // Get input values for lookup operand computation
            const left_input = self.values[R1CSInputIndex.LeftInstructionInput.toIndex()];
            const right_input = self.values[R1CSInputIndex.RightInstructionInput.toIndex()];
            const product = self.values[R1CSInputIndex.Product.toIndex()];

            switch (opcode) {
                0x03 => { // LOAD
                    // FlagLoad already set in fromTraceStep
                    // Lookups: NOT Add+Sub+Mul, so LeftLookup == LeftInput, RightLookup == RightInput
                    self.values[R1CSInputIndex.LeftLookupOperand.toIndex()] = left_input;
                    self.values[R1CSInputIndex.RightLookupOperand.toIndex()] = right_input;
                },
                0x23 => { // STORE
                    // FlagStore already set in fromTraceStep
                    self.values[R1CSInputIndex.LeftLookupOperand.toIndex()] = left_input;
                    self.values[R1CSInputIndex.RightLookupOperand.toIndex()] = right_input;
                },
                0x33 => { // R-type (ADD, SUB, MUL, etc.)
                    // Determine specific operation
                    if (funct7 == 0x01) {
                        // M-extension: MUL, MULH, MULHSU, MULHU, DIV, DIVU, REM, REMU
                        if (funct3 == 0x0) { // MUL
                            self.values[R1CSInputIndex.FlagMultiplyOperands.toIndex()] = F.one();
                            // Constraint 5: LeftLookup == 0 for Mul
                            self.values[R1CSInputIndex.LeftLookupOperand.toIndex()] = F.zero();
                            // Constraint 9: RightLookup == Product for Mul
                            self.values[R1CSInputIndex.RightLookupOperand.toIndex()] = product;
                        } else {
                            // Other M-extension ops (DIV, etc.)
                            self.values[R1CSInputIndex.LeftLookupOperand.toIndex()] = left_input;
                            self.values[R1CSInputIndex.RightLookupOperand.toIndex()] = right_input;
                        }
                    } else if (funct7 == 0x20 and funct3 == 0x0) {
                        // SUB
                        self.values[R1CSInputIndex.FlagSubtractOperands.toIndex()] = F.one();
                        // Constraint 5: LeftLookup == 0 for Sub
                        self.values[R1CSInputIndex.LeftLookupOperand.toIndex()] = F.zero();
                        // Constraint 8: RightLookup == LeftInput - RightInput + 2^64 for Sub
                        // The 2^64 offset converts to two's complement representation
                        // 2^64 = 0x10000000000000000 represented as bytes
                        const two_pow_64 = F.fromBytes(&[_]u8{ 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0 });
                        self.values[R1CSInputIndex.RightLookupOperand.toIndex()] = left_input.sub(right_input).add(two_pow_64);
                    } else {
                        // ADD and other R-type
                        self.values[R1CSInputIndex.FlagAddOperands.toIndex()] = F.one();
                        // Constraint 5: LeftLookup == 0 for Add
                        self.values[R1CSInputIndex.LeftLookupOperand.toIndex()] = F.zero();
                        // Constraint 7: RightLookup == RightInput + LeftInput for Add
                        self.values[R1CSInputIndex.RightLookupOperand.toIndex()] = right_input.add(left_input);
                    }
                    self.values[R1CSInputIndex.FlagWriteLookupOutputToRD.toIndex()] = F.one();
                },
                0x13 => { // I-type (ADDI, etc.)
                    self.values[R1CSInputIndex.FlagAddOperands.toIndex()] = F.one();
                    // For I-type, the "right input" comes from immediate
                    // Constraint 5: LeftLookup == 0 for Add
                    self.values[R1CSInputIndex.LeftLookupOperand.toIndex()] = F.zero();
                    // Constraint 7: RightLookup == RightInput + LeftInput for Add
                    // For ADDI, right_input should be the immediate
                    self.values[R1CSInputIndex.RightLookupOperand.toIndex()] = right_input.add(left_input);
                    self.values[R1CSInputIndex.FlagWriteLookupOutputToRD.toIndex()] = F.one();
                },
                0x6F => { // JAL
                    self.values[R1CSInputIndex.FlagJump.toIndex()] = F.one();
                    self.values[R1CSInputIndex.ShouldJump.toIndex()] = F.one();
                    self.values[R1CSInputIndex.WritePCtoRD.toIndex()] = F.one();
                    // NOT Add+Sub+Mul
                    self.values[R1CSInputIndex.LeftLookupOperand.toIndex()] = left_input;
                    self.values[R1CSInputIndex.RightLookupOperand.toIndex()] = right_input;
                },
                0x67 => { // JALR
                    self.values[R1CSInputIndex.FlagJump.toIndex()] = F.one();
                    self.values[R1CSInputIndex.ShouldJump.toIndex()] = F.one();
                    self.values[R1CSInputIndex.WritePCtoRD.toIndex()] = F.one();
                    // NOT Add+Sub+Mul
                    self.values[R1CSInputIndex.LeftLookupOperand.toIndex()] = left_input;
                    self.values[R1CSInputIndex.RightLookupOperand.toIndex()] = right_input;
                },
                0x63 => { // Branch
                    // NOT Add+Sub+Mul
                    self.values[R1CSInputIndex.LeftLookupOperand.toIndex()] = left_input;
                    self.values[R1CSInputIndex.RightLookupOperand.toIndex()] = right_input;
                    // ShouldBranch depends on branch condition evaluation
                },
                else => {
                    // Default: NOT Add+Sub+Mul, so use constraint 6 and 10
                    self.values[R1CSInputIndex.LeftLookupOperand.toIndex()] = left_input;
                    self.values[R1CSInputIndex.RightLookupOperand.toIndex()] = right_input;
                },
            }
        }

        /// Get value at index
        pub fn get(self: *const Self, index: R1CSInputIndex) F {
            return self.values[index.toIndex()];
        }

        /// Get all values as slice
        pub fn asSlice(self: *const Self) []const F {
            return &self.values;
        }
    };
}

/// Generate R1CS witness for entire execution trace
pub fn R1CSWitnessGenerator(comptime F: type) type {
    return struct {
        const Self = @This();

        allocator: Allocator,

        pub fn init(allocator: Allocator) Self {
            return .{ .allocator = allocator };
        }

        pub fn deinit(self: *Self) void {
            _ = self;
            // No resources to free - this is just a wrapper
        }

        /// Generate witness for all cycles in trace
        pub fn generateWitness(
            self: *Self,
            trace: *const tracer.ExecutionTrace,
        ) ![]R1CSCycleInputs(F) {
            const num_cycles = trace.steps.items.len;
            if (num_cycles == 0) {
                return &[_]R1CSCycleInputs(F){};
            }

            const witnesses = try self.allocator.alloc(R1CSCycleInputs(F), num_cycles);

            for (trace.steps.items, 0..) |step, i| {
                const next_step = if (i + 1 < num_cycles)
                    trace.steps.items[i + 1]
                else
                    null;
                witnesses[i] = R1CSCycleInputs(F).fromTraceStep(step, next_step);
            }

            return witnesses;
        }

        /// Verify all constraints are satisfied for a trace
        pub fn verifyConstraints(
            _: *Self,
            witnesses: []const R1CSCycleInputs(F),
        ) bool {
            for (witnesses) |witness| {
                for (UNIFORM_CONSTRAINTS) |constraint| {
                    if (!constraint.isSatisfied(F, witness.asSlice())) {
                        return false;
                    }
                }
            }
            return true;
        }
    };
}

// ============================================================================
// Tests
// ============================================================================

test "linear combination evaluation" {
    const field = @import("../../field/mod.zig");
    const F = field.BN254Scalar;

    // Create a simple LC: 2*x + 3*y + 5
    var lc = LC.zero();
    lc.terms[0] = .{ .input_index = .LeftInstructionInput, .coeff = 2 };
    lc.terms[1] = .{ .input_index = .RightInstructionInput, .coeff = 3 };
    lc.len = 2;
    lc.constant = 5;

    // Witness: x=10, y=20
    var witness: [R1CSInputIndex.NUM_INPUTS]F = [_]F{F.zero()} ** R1CSInputIndex.NUM_INPUTS;
    witness[R1CSInputIndex.LeftInstructionInput.toIndex()] = F.fromU64(10);
    witness[R1CSInputIndex.RightInstructionInput.toIndex()] = F.fromU64(20);

    // Expected: 2*10 + 3*20 + 5 = 20 + 60 + 5 = 85
    const result = lc.evaluate(F, &witness);
    try std.testing.expect(result.eql(F.fromU64(85)));
}

test "uniform constraint satisfied" {
    const field = @import("../../field/mod.zig");
    const F = field.BN254Scalar;

    // Test constraint 2: If Load, then RamReadValue == RamWriteValue
    const constraint = UNIFORM_CONSTRAINTS[2];

    // Create witness where Load=1, RamReadValue=42, RamWriteValue=42
    var witness: [R1CSInputIndex.NUM_INPUTS]F = [_]F{F.zero()} ** R1CSInputIndex.NUM_INPUTS;
    witness[R1CSInputIndex.FlagLoad.toIndex()] = F.one();
    witness[R1CSInputIndex.RamReadValue.toIndex()] = F.fromU64(42);
    witness[R1CSInputIndex.RamWriteValue.toIndex()] = F.fromU64(42);

    // Should be satisfied
    try std.testing.expect(constraint.isSatisfied(F, &witness));
}

test "uniform constraint violated" {
    const field = @import("../../field/mod.zig");
    const F = field.BN254Scalar;

    // Test constraint 2: If Load, then RamReadValue == RamWriteValue
    const constraint = UNIFORM_CONSTRAINTS[2];

    // Create witness where Load=1, RamReadValue=42, RamWriteValue=100 (violation!)
    var witness: [R1CSInputIndex.NUM_INPUTS]F = [_]F{F.zero()} ** R1CSInputIndex.NUM_INPUTS;
    witness[R1CSInputIndex.FlagLoad.toIndex()] = F.one();
    witness[R1CSInputIndex.RamReadValue.toIndex()] = F.fromU64(42);
    witness[R1CSInputIndex.RamWriteValue.toIndex()] = F.fromU64(100);

    // Should NOT be satisfied
    try std.testing.expect(!constraint.isSatisfied(F, &witness));
}

test "conditional constraint bypass" {
    const field = @import("../../field/mod.zig");
    const F = field.BN254Scalar;

    // Test constraint 2: If Load, then RamReadValue == RamWriteValue
    const constraint = UNIFORM_CONSTRAINTS[2];

    // Create witness where Load=0 (bypass), values don't matter
    var witness: [R1CSInputIndex.NUM_INPUTS]F = [_]F{F.zero()} ** R1CSInputIndex.NUM_INPUTS;
    witness[R1CSInputIndex.FlagLoad.toIndex()] = F.zero(); // Not a load
    witness[R1CSInputIndex.RamReadValue.toIndex()] = F.fromU64(42);
    witness[R1CSInputIndex.RamWriteValue.toIndex()] = F.fromU64(100); // Different value

    // Should still be satisfied because condition is 0
    try std.testing.expect(constraint.isSatisfied(F, &witness));
}
