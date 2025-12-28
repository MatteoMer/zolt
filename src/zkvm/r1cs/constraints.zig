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
pub const R1CSInputIndex = enum(u8) {
    // Instruction inputs (0-2)
    LeftInstructionInput = 0,
    RightInstructionInput = 1,
    Product = 2,

    // Lookup operands (3-5)
    LeftLookupOperand = 3,
    RightLookupOperand = 4,
    LookupOutput = 5,

    // Register values (6-8)
    Rs1Value = 6,
    Rs2Value = 7,
    RdWriteValue = 8,

    // RAM access (9-11)
    RamAddress = 9,
    RamReadValue = 10,
    RamWriteValue = 11,

    // PC values (12-15)
    PC = 12,
    NextPC = 13,
    UnexpandedPC = 14,
    NextUnexpandedPC = 15,

    // Immediate (16)
    Imm = 16,

    // Derived flags (17-22)
    WriteLookupOutputToRD = 17,
    WritePCtoRD = 18,
    ShouldBranch = 19,
    ShouldJump = 20,
    NextIsVirtual = 21,
    NextIsFirstInSequence = 22,

    // Circuit flags (23-35)
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

        /// Evaluate the linear combination given witness values
        pub fn evaluate(self: Self, comptime F: type, witness: []const F) F {
            var result = if (self.constant >= 0)
                F.fromU64(@intCast(@as(u128, @intCast(self.constant))))
            else
                F.zero().sub(F.fromU64(@intCast(@as(u128, @intCast(-self.constant)))));

            for (self.terms[0..self.len]) |term| {
                const val = witness[term.input_index.toIndex()];
                const scaled = if (term.coeff >= 0)
                    val.mul(F.fromU64(@intCast(@as(u128, @intCast(term.coeff)))))
                else
                    val.mul(F.fromU64(@intCast(@as(u128, @intCast(-term.coeff))))).neg();
                result = result.add(scaled);
            }

            return result;
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
            // 2^64 = 18446744073709551616 (too large for i128, handled separately)
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
pub const FIRST_GROUP_INDICES: [10]usize = .{
    1, // RamAddrEqZeroIfNotLoadStore
    2, // RamReadEqRamWriteIfLoad
    3, // RamReadEqRdWriteIfLoad
    4, // Rs2EqRamWriteIfStore
    5, // LeftLookupZeroUnlessAddSubMul
    6, // LeftLookupEqLeftInputOtherwise
    11, // AssertLookupOne
    14, // NextUnexpPCEqLookupIfShouldJump
    16, // NextUnexpPCUpdateOtherwise (or 17 - NextPCEqPCPlusOneIfInline)
    18, // MustStartSequenceFromBeginning
};

/// Second group constraint indices (9 constraints)
/// These are the global indices from UNIFORM_CONSTRAINTS that belong to the second group
pub const SECOND_GROUP_INDICES: [9]usize = .{
    0, // RamAddrEqRs1PlusImmIfLoadStore
    7, // RightLookupAdd
    8, // RightLookupSub
    9, // RightLookupEqProductIfMul
    10, // RightLookupEqRightInputOtherwise
    12, // RdWriteEqLookupIfWriteLookupToRd
    13, // RdWriteEqPCPlusConstIfWritePCtoRD
    15, // NextUnexpPCEqPCPlusImmIfShouldBranch
    17, // NextPCEqPCPlusOneIfInline (if 16 is for first group)
};

/// Per-cycle R1CS inputs extracted from execution trace
pub fn R1CSCycleInputs(comptime F: type) type {
    return struct {
        const Self = @This();

        /// Witness values for this cycle (indexed by R1CSInputIndex)
        values: [R1CSInputIndex.NUM_INPUTS]F,

        /// Create cycle inputs from an execution trace step
        pub fn fromTraceStep(
            step: tracer.TraceStep,
            next_step: ?tracer.TraceStep,
        ) Self {
            var inputs = Self{
                .values = [_]F{F.zero()} ** R1CSInputIndex.NUM_INPUTS,
            };

            // Instruction inputs
            inputs.values[R1CSInputIndex.LeftInstructionInput.toIndex()] = F.fromU64(step.rs1_value);
            inputs.values[R1CSInputIndex.RightInstructionInput.toIndex()] = F.fromU64(step.rs2_value);
            // Product = rs1 * rs2 (for multiply instructions)
            const product = @as(u128, step.rs1_value) * @as(u128, step.rs2_value);
            inputs.values[R1CSInputIndex.Product.toIndex()] = F.fromU64(@truncate(product));

            // Lookup operands (simplified - would come from lookup trace)
            inputs.values[R1CSInputIndex.LeftLookupOperand.toIndex()] = F.fromU64(step.rs1_value);
            inputs.values[R1CSInputIndex.RightLookupOperand.toIndex()] = F.fromU64(step.rs2_value);
            inputs.values[R1CSInputIndex.LookupOutput.toIndex()] = F.fromU64(step.rd_value);

            // Register values
            inputs.values[R1CSInputIndex.Rs1Value.toIndex()] = F.fromU64(step.rs1_value);
            inputs.values[R1CSInputIndex.Rs2Value.toIndex()] = F.fromU64(step.rs2_value);
            inputs.values[R1CSInputIndex.RdWriteValue.toIndex()] = F.fromU64(step.rd_value);

            // RAM access (simplified) - use memory_addr if present
            const ram_addr = step.memory_addr orelse step.rs1_value;
            inputs.values[R1CSInputIndex.RamAddress.toIndex()] = F.fromU64(ram_addr);
            inputs.values[R1CSInputIndex.RamReadValue.toIndex()] = F.fromU64(step.memory_value orelse 0);
            inputs.values[R1CSInputIndex.RamWriteValue.toIndex()] = if (step.is_memory_write)
                F.fromU64(step.memory_value orelse 0)
            else
                F.zero();

            // PC values
            inputs.values[R1CSInputIndex.PC.toIndex()] = F.fromU64(step.pc);
            inputs.values[R1CSInputIndex.UnexpandedPC.toIndex()] = F.fromU64(step.pc);
            inputs.values[R1CSInputIndex.NextPC.toIndex()] = F.fromU64(step.next_pc);
            inputs.values[R1CSInputIndex.NextUnexpandedPC.toIndex()] = F.fromU64(step.next_pc);

            // Use next step's PC if available for verification
            if (next_step) |ns| {
                inputs.values[R1CSInputIndex.NextPC.toIndex()] = F.fromU64(ns.pc);
            }

            // Immediate - derive from instruction
            const imm = inputs.deriveImmediate(step.instruction);
            inputs.values[R1CSInputIndex.Imm.toIndex()] = imm;

            // Circuit flags - set based on instruction opcode
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

        /// Set circuit flags based on instruction
        fn setFlagsFromInstruction(self: *Self, instr: u32) void {
            const opcode: u8 = @truncate(instr & 0x7F);
            // Map opcode to circuit flags
            switch (opcode) {
                0x03 => { // LOAD
                    self.values[R1CSInputIndex.FlagLoad.toIndex()] = F.one();
                },
                0x23 => { // STORE
                    self.values[R1CSInputIndex.FlagStore.toIndex()] = F.one();
                },
                0x33 => { // R-type (ADD, SUB, etc.)
                    self.values[R1CSInputIndex.FlagAddOperands.toIndex()] = F.one();
                    self.values[R1CSInputIndex.FlagWriteLookupOutputToRD.toIndex()] = F.one();
                },
                0x13 => { // I-type (ADDI, etc.)
                    self.values[R1CSInputIndex.FlagAddOperands.toIndex()] = F.one();
                    self.values[R1CSInputIndex.FlagWriteLookupOutputToRD.toIndex()] = F.one();
                },
                0x6F => { // JAL
                    self.values[R1CSInputIndex.FlagJump.toIndex()] = F.one();
                    self.values[R1CSInputIndex.ShouldJump.toIndex()] = F.one();
                    self.values[R1CSInputIndex.WritePCtoRD.toIndex()] = F.one();
                },
                0x67 => { // JALR
                    self.values[R1CSInputIndex.FlagJump.toIndex()] = F.one();
                    self.values[R1CSInputIndex.ShouldJump.toIndex()] = F.one();
                    self.values[R1CSInputIndex.WritePCtoRD.toIndex()] = F.one();
                },
                0x63 => { // Branch
                    // ShouldBranch is set based on lookup output
                    // Simplified here
                },
                else => {},
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
