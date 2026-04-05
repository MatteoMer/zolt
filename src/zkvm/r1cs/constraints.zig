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

// Re-export witness types from dedicated module
const witness_types = @import("witness_types.zig");
pub const R1CSInputIndex = witness_types.R1CSInputIndex;
pub const Term = witness_types.Term;
pub const LinearCombination = witness_types.LinearCombination;
pub const LC = witness_types.LC;
pub const UniformConstraint = witness_types.UniformConstraint;

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
    // if { OpFlags(WriteLookupOutputToRD) } => ( RdWriteValue ) == ( LookupOutput )
    .{
        .condition = LC.fromInput(.FlagWriteLookupOutputToRD),
        .left = LC.fromInput(.RdWriteValue),
        .right = LC.fromInput(.LookupOutput),
    },

    // =========================================================================
    // CONSTRAINT 13: RdWriteEqPCPlusConstIfWritePCtoRD (SECOND GROUP index 6)
    // =========================================================================
    // if { OpFlags(Jump) } => ( RdWriteValue ) == ( UnexpandedPC + 4 - 2*IsCompressed )
    .{
        .condition = LC.fromInput(.FlagJump),
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
    // CONSTRAINT 17: NextPCEqPCPlusOneIfInline (FIRST GROUP index 8)
    // =========================================================================
    // if { VirtualInstruction - IsLastInSequence } => ( NextPC ) == ( PC + 1 )
    // Guard = VI - IsLast. For valid boolean inputs where IsLast=1 implies VI=1,
    // this equals VI && !IsLast. Skips constraint when JALR terminates a virtual sequence.
    .{
        .condition = blk: {
            var lc = LC.zero();
            lc.terms[0] = .{ .input_index = .FlagVirtualInstruction, .coeff = 1 };
            lc.terms[1] = .{ .input_index = .FlagIsLastInSequence, .coeff = -1 };
            lc.len = 2;
            break :blk lc;
        },
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

/// Check if a trace step is a NOOP instruction
/// In RISC-V, NOOP is typically encoded as ADDI x0, x0, 0
/// This matches Jolt's InstructionFlags::IsNoop semantics
fn isNoopInstruction(step_opt: ?tracer.TraceStep) bool {
    const step = step_opt orelse return false; // No next step means not a noop

    // Check if this is a NoOp padding cycle (marked explicitly)
    if (step.is_noop) return true;

    // Check if it's ADDI x0, x0, 0 (the canonical NOP instruction)
    const instr = step.instruction;
    const opcode: u8 = @truncate(instr & 0x7F);

    if (opcode == 0x13) {
        const rd: u8 = @truncate((instr >> 7) & 0x1F);
        const rs1: u8 = @truncate((instr >> 15) & 0x1F);
        const funct3: u8 = @truncate((instr >> 12) & 0x7);

        // ADDI x0, x0, 0: funct3 = 0, rd = 0, rs1 = 0, imm = 0
        if (rd == 0 and rs1 == 0 and funct3 == 0) {
            const imm: i32 = @bitCast(@as(u32, @truncate(instr >> 20)));
            if (imm == 0) {
                return true;
            }
        }
    }

    return false;
}

/// Compute the LookupOutput value for a trace step
/// LookupOutput is the result of the instruction's lookup table operation.
/// For JAL: jump target = PC + imm
/// For JALR: jump target = (rs1 + imm) & ~1
/// For other instructions: rd_value (the result written to rd)
pub fn computeLookupOutput(comptime FieldType: type, step: tracer.TraceStep) FieldType {
    const opcode: u8 = @truncate(step.instruction & 0x7F);
    const funct3: u3 = @truncate((step.instruction >> 12) & 0x7);
    const funct7: u7 = @truncate(step.instruction >> 25);

    // CRITICAL: Instructions without a lookup table return 0 for lookup_output.
    // In Jolt, these instructions have to_lookup_output() = 0.
    // This includes: Load, Store, SLL, SLLI, and any instruction where
    // getLookupTableIndex() returns -1.
    //
    // Check if this instruction has a lookup table. If not, return 0.
    const has_table = hasLookupTable(opcode, funct3, funct7);
    if (!has_table) {
        return FieldType.zero();
    }

    switch (opcode) {
        0x6F => { // JAL
            // LookupOutput = unexpanded_pc + imm (the jump target)
            // Must use unexpanded_pc (ELF address), not pc (bytecode index),
            // to match computeU128LookupOperand and Stage 5's combined_vals.
            const imm = decodeJTypeImmediate(step.instruction);
            const pc_i64: i64 = @intCast(step.unexpanded_pc);
            const target = @as(u64, @bitCast(pc_i64 +% imm));
            return FieldType.fromU64(target);
        },
        0x67 => { // JALR
            // LookupOutput = (rs1 + imm) & ~1 (the jump target with LSB cleared)
            const imm = decodeITypeImmediate(step.instruction);
            const rs1_i64: i64 = @intCast(step.rs1_value);
            const target = @as(u64, @bitCast(rs1_i64 +% imm)) & ~@as(u64, 1);
            return FieldType.fromU64(target);
        },
        0x63 => { // Branch - LookupOutput = branch condition result (0 or 1)
            const rs1 = step.rs1_value;
            const rs2 = step.rs2_value;
            const taken: bool = switch (funct3) {
                0x0 => rs1 == rs2, // BEQ
                0x1 => rs1 != rs2, // BNE
                0x4 => @as(i64, @bitCast(rs1)) < @as(i64, @bitCast(rs2)), // BLT (signed)
                0x5 => @as(i64, @bitCast(rs1)) >= @as(i64, @bitCast(rs2)), // BGE (signed)
                0x6 => rs1 < rs2, // BLTU (unsigned)
                0x7 => rs1 >= rs2, // BGEU (unsigned)
                else => false,
            };
            return if (taken) FieldType.one() else FieldType.zero();
        },
        0x22, 0x62 => {
            // VirtualAssertEQ and VirtualAssertValidUnsignedRemainder are Assert instructions.
            // For Assert instructions, the lookup output is always 1 (assertion passed).
            // This satisfies Constraint 11: if { Assert } => ( LookupOutput ) == ( 1 )
            return FieldType.one();
        },
        else => {
            // For other instructions with a lookup table, use rd_value
            return FieldType.fromU64(step.rd_value);
        },
    }
}

/// Check if an instruction has a lookup table assignment.
/// Returns false for Load, Store, SLL, SLLI, and other instructions
/// that don't use lookup tables (matching Jolt's lookup_table() = None).
pub fn hasLookupTable(opcode: u8, funct3: u3, funct7: u7) bool {
    return switch (opcode) {
        0x33 => blk: { // R-type
            if (funct3 == 0 and funct7 == 0) break :blk true; // ADD
            if (funct3 == 0 and funct7 == 0x20) break :blk true; // SUB
            if (funct3 == 7) break :blk true; // AND
            if (funct3 == 6) break :blk true; // OR
            if (funct3 == 4) break :blk true; // XOR
            if (funct3 == 1) break :blk false; // SLL - no table
            if (funct3 == 5 and funct7 == 0) break :blk true; // SRL
            if (funct3 == 5 and funct7 == 0x20) break :blk true; // SRA
            if (funct3 == 2) break :blk true; // SLT
            if (funct3 == 3) break :blk true; // SLTU
            if (funct7 == 0x01 and funct3 == 0) break :blk true; // MUL
            if (funct7 == 0x01 and funct3 == 3) break :blk true; // MULHU
            break :blk false;
        },
        0x13 => blk: { // I-type
            if (funct3 == 0) break :blk true; // ADDI
            if (funct3 == 7) break :blk true; // ANDI
            if (funct3 == 6) break :blk true; // ORI
            if (funct3 == 4) break :blk true; // XORI
            if (funct3 == 1) break :blk false; // SLLI - no table
            if (funct3 == 5) break :blk true; // SRLI/SRAI
            if (funct3 == 2) break :blk true; // SLTI
            if (funct3 == 3) break :blk true; // SLTIU
            break :blk false;
        },
        0x1b => (funct3 == 0), // ADDIW
        0x3b => blk: { // OP-32
            if (funct3 == 0 and funct7 == 0) break :blk true; // ADDW
            if (funct3 == 0 and funct7 == 0x20) break :blk true; // SUBW
            if (funct3 == 6 and funct7 == 0x01) break :blk true; // VirtualChangeDivisorW
            break :blk false;
        },
        0x63 => true, // All branches have tables
        0x37 => true, // LUI
        0x17 => true, // AUIPC
        0x6f => true, // JAL
        0x67 => true, // JALR
        0x0B => true, // VirtualSignExtendWord - uses SignExtendHalfWord table
        0x2B => true, // VirtualMULI - uses RangeCheck table
        0x5B => true, // VirtualSRLI - uses VirtualSRL table
        0x02 => true, // VirtualAdvice - uses RangeCheck table (Advice)
        0x22 => true, // VirtualAssertEQ - uses Equal table (Assert)
        0x42 => true, // VirtualZeroExtendWord - uses LowerHalfWord table (AddOperands)
        0x62 => true, // VirtualAssertValidUnsignedRemainder - uses ValidUnsignedRemainder table (Assert)
        0x03 => false, // Load - no table
        0x23 => false, // Store - no table
        else => false,
    };
}

/// Decode J-type immediate (for JAL)
pub fn decodeJTypeImmediate(instr: u32) i64 {
    // J-type immediate: imm[20|10:1|11|19:12] - bits 31, 30:21, 20, 19:12
    const imm20: u32 = (instr >> 31) & 0x1;
    const imm10_1: u32 = (instr >> 21) & 0x3FF;
    const imm11: u32 = (instr >> 20) & 0x1;
    const imm19_12: u32 = (instr >> 12) & 0xFF;
    const unsigned_imm: u32 = (imm20 << 20) | (imm19_12 << 12) | (imm11 << 11) | (imm10_1 << 1);
    // Sign extend from bit 20
    if (imm20 != 0) {
        return @as(i64, @bitCast(@as(u64, unsigned_imm) | 0xFFFFFFFFFFE00000));
    }
    return @as(i64, unsigned_imm);
}

/// Decode I-type immediate (for JALR)
pub fn decodeITypeImmediate(instr: u32) i64 {
    // I-type immediate: imm[11:0] in bits 31:20
    const unsigned_imm: u32 = instr >> 20;
    // Sign extend from bit 11
    if ((unsigned_imm & 0x800) != 0) {
        return @as(i64, @bitCast(@as(u64, unsigned_imm) | 0xFFFFFFFFFFFFF000));
    }
    return @as(i64, unsigned_imm);
}

/// Result of computing instruction inputs (left_input, right_input)
fn InstructionInputs(comptime F: type) type {
    return struct {
        left: F,
        right: F,
        /// Whether the right input is signed (for Product computation)
        right_is_signed: bool,
        /// The raw signed right value as i128 (needed for correct Product computation)
        right_i128: i128,
    };
}

/// Compute instruction inputs matching Jolt's LookupQuery::to_instruction_inputs semantics.
///
/// Different instruction types use different operands:
/// - ADD, SUB, AND, OR, XOR, etc.: left=rs1, right=rs2
/// - ADDI, SLTI, XORI, ORI, ANDI: left=rs1, right=imm
/// - JAL, AUIPC: left=PC, right=imm
/// - JALR: left=rs1, right=imm
/// - LUI: left=0, right=imm
/// - LOAD/STORE: left=rs1, right=imm
/// - Branches: left=rs1, right=rs2
fn computeInstructionInputs(comptime F: type, step: tracer.TraceStep) InstructionInputs(F) {
    const opcode: u8 = @truncate(step.instruction & 0x7F);
    const funct3 = (step.instruction >> 12) & 0x7;
    _ = funct3;

    switch (opcode) {
        // R-type: ADD, SUB, AND, OR, XOR, SLT, SLTU, SLL, SRL, SRA, MUL, etc.
        0x33 => {
            // left = rs1, right = rs2
            return .{
                .left = F.fromU64(step.rs1_value),
                .right = F.fromU64(step.rs2_value),
                .right_is_signed = false,
                .right_i128 = @as(i128, step.rs2_value),
            };
        },
        // I-type arithmetic: ADDI, SLTI, SLTIU, XORI, ORI, ANDI, SLLI, SRLI, SRAI
        0x13 => {
            // left = rs1, right = imm (sign-extended)
            const imm = decodeITypeImmediate(step.instruction);
            return .{
                .left = F.fromU64(step.rs1_value),
                .right = signedI64ToField(F, imm),
                .right_is_signed = true,
                .right_i128 = @as(i128, imm),
            };
        },
        // LOAD: LB, LH, LW, LD, LBU, LHU, LWU
        0x03 => {
            // left = rs1, right = imm
            const imm = decodeITypeImmediate(step.instruction);
            return .{
                .left = F.fromU64(step.rs1_value),
                .right = signedI64ToField(F, imm),
                .right_is_signed = true,
                .right_i128 = @as(i128, imm),
            };
        },
        // STORE: SB, SH, SW, SD
        0x23 => {
            // left = rs1, right = imm (S-type encoding)
            const imm = decodeSTypeImmediate(step.instruction);
            return .{
                .left = F.fromU64(step.rs1_value),
                .right = signedI64ToField(F, imm),
                .right_is_signed = true,
                .right_i128 = @as(i128, imm),
            };
        },
        // JAL
        0x6F => {
            // left = PC, right = imm
            const imm = decodeJTypeImmediate(step.instruction);
            return .{
                .left = F.fromU64(step.pc),
                .right = signedI64ToField(F, imm),
                .right_is_signed = true,
                .right_i128 = @as(i128, imm),
            };
        },
        // JALR
        0x67 => {
            // left = rs1, right = imm
            const imm = decodeITypeImmediate(step.instruction);
            return .{
                .left = F.fromU64(step.rs1_value),
                .right = signedI64ToField(F, imm),
                .right_is_signed = true,
                .right_i128 = @as(i128, imm),
            };
        },
        // Branches: BEQ, BNE, BLT, BGE, BLTU, BGEU
        0x63 => {
            // left = rs1, right = rs2
            return .{
                .left = F.fromU64(step.rs1_value),
                .right = F.fromU64(step.rs2_value),
                .right_is_signed = false,
                .right_i128 = @as(i128, step.rs2_value),
            };
        },
        // LUI: rd = imm
        0x37 => {
            // left = 0, right = imm (U-type, sign-extended to 64 bits, treated as UNSIGNED u64)
            // Jolt: FormatU.parse sign-extends via `as i32 as i64 as u64`, then
            // S64::from_u64_with_sign(imm, true) treats it as positive u64.
            // So the field element is F.fromU64(sign_extended_bits), NOT signedI64ToField.
            const imm = decodeUTypeImmediate(step.instruction);
            const imm_u64: u64 = @bitCast(imm);
            return .{
                .left = F.zero(),
                .right = F.fromU64(imm_u64),
                .right_is_signed = false,
                .right_i128 = @as(i128, imm_u64),
            };
        },
        // AUIPC: rd = PC + imm
        0x17 => {
            // left = PC, right = imm (U-type, sign-extended to 64 bits, treated as UNSIGNED u64)
            const imm = decodeUTypeImmediate(step.instruction);
            const imm_u64: u64 = @bitCast(imm);
            return .{
                .left = F.fromU64(step.pc),
                .right = F.fromU64(imm_u64),
                .right_is_signed = false,
                .right_i128 = @as(i128, imm_u64),
            };
        },
        // OP-IMM-32 (RV64I word operations): ADDIW, SLLIW, SRLIW, SRAIW
        0x1B => {
            const imm = decodeITypeImmediate(step.instruction);
            return .{
                .left = F.fromU64(step.rs1_value),
                .right = signedI64ToField(F, imm),
                .right_is_signed = true,
                .right_i128 = @as(i128, imm),
            };
        },
        // OP-32 (RV64I/M word operations): ADDW, SUBW, MULW, etc.
        0x3B => {
            return .{
                .left = F.fromU64(step.rs1_value),
                .right = F.fromU64(step.rs2_value),
                .right_is_signed = false,
                .right_i128 = @as(i128, step.rs2_value),
            };
        },
        // Virtual instructions on opcode 0x2B, dispatched by funct3:
        //   funct3=0: VirtualMULI (SLLI decomposition): left=rs1, right=multiplier (1 << shamt)
        //   funct3=1: VirtualPow2: left=rs1, right=0, imm=0
        //   funct3=2: VirtualShiftRightBitmask: left=rs1, right=0, imm=0
        0x2B => {
            const funct3_2b: u3 = @truncate((step.instruction >> 12) & 0x7);
            switch (funct3_2b) {
                0 => {
                    // VirtualMULI: The instruction encoding stores the shift amount in the I-type imm field.
                    // We compute the multiplier (1 << shamt) from it.
                    const shamt_raw: u32 = step.instruction >> 20;
                    const shamt: u6 = @truncate(shamt_raw & 0x3F);
                    const multiplier: u64 = @as(u64, 1) << shamt;
                    return .{
                        .left = F.fromU64(step.rs1_value),
                        .right = F.fromU64(multiplier),
                        .right_is_signed = false,
                        .right_i128 = @as(i128, multiplier),
                    };
                },
                1, 2 => {
                    // VirtualPow2 / VirtualShiftRightBitmask: AddOperands with imm=0
                    // to_instruction_inputs = (rs1, 0)
                    return .{
                        .left = F.fromU64(step.rs1_value),
                        .right = F.zero(),
                        .right_is_signed = false,
                        .right_i128 = 0,
                    };
                },
                else => {
                    return .{
                        .left = F.zero(),
                        .right = F.zero(),
                        .right_is_signed = false,
                        .right_i128 = 0,
                    };
                },
            }
        },
        // Virtual instructions on opcode 0x5B, dispatched by funct3 and rs2_read:
        //   I-type (rs2_read=false): VirtualSRLI (funct3=0) or VirtualSRAI (funct3=5)
        //     left=rs1, right=bitmask (computed from total shift in imm field)
        //   R-type (rs2_read=true): VirtualSRL (funct3=0) or VirtualSRA (funct3=5)
        //     left=rs1, right=rs2 (bitmask from virtual register)
        0x5B => {
            if (step.rs2_read) {
                // R-type VirtualSRL/VirtualSRA: left=rs1, right=rs2
                return .{
                    .left = F.fromU64(step.rs1_value),
                    .right = F.fromU64(step.rs2_value),
                    .right_is_signed = false,
                    .right_i128 = @as(i128, step.rs2_value),
                };
            } else {
                // I-type VirtualSRLI/VirtualSRAI: left=rs1, right=bitmask from shift amount
                const total_shift_raw: u32 = step.instruction >> 20;
                const total_shift: u7 = @truncate(total_shift_raw & 0x3F);
                const ones: u128 = (@as(u128, 1) << @intCast(64 - @as(u8, total_shift))) - 1;
                const bitmask: u64 = @truncate(ones << total_shift);
                return .{
                    .left = F.fromU64(step.rs1_value),
                    .right = F.fromU64(bitmask),
                    .right_is_signed = false,
                    .right_i128 = @as(i128, bitmask),
                };
            }
        },
        // SYSTEM: ECALL, EBREAK (opcode 0x73)
        0x73 => {
            // No instruction inputs for system calls
            return .{
                .left = F.zero(),
                .right = F.zero(),
                .right_is_signed = false,
                .right_i128 = 0,
            };
        },
        // MISC-MEM: FENCE (opcode 0x0F)
        0x0F => {
            // No instruction inputs for fence
            return .{
                .left = F.zero(),
                .right = F.zero(),
                .right_is_signed = false,
                .right_i128 = 0,
            };
        },
        // Default: treat as no inputs (matching Jolt's default behavior for unknown instructions)
        // This ensures right_is_rs2*rs2 + right_is_imm*imm = RightInstructionInput = 0
        else => {
            return .{
                .left = F.zero(),
                .right = F.zero(),
                .right_is_signed = false,
                .right_i128 = 0,
            };
        },
    }
}

/// Decode S-type immediate (for STORE instructions)
fn decodeSTypeImmediate(instr: u32) i64 {
    const imm4_0: u32 = (instr >> 7) & 0x1F;
    const imm11_5: u32 = (instr >> 25) & 0x7F;
    const unsigned_imm: u32 = (imm11_5 << 5) | imm4_0;
    // Sign extend from bit 11
    if ((unsigned_imm & 0x800) != 0) {
        return @as(i64, @bitCast(@as(u64, unsigned_imm) | 0xFFFFFFFFFFFFF000));
    }
    return @as(i64, unsigned_imm);
}

/// Decode U-type immediate (for LUI, AUIPC)
/// The immediate occupies bits [31:12] and is sign-extended.
/// Matches Jolt's FormatU::parse which does sign extension.
fn decodeUTypeImmediate(instr: u32) i64 {
    // Extract bits [31:12] with sign extension
    // Jolt does: (word & 0xfffff000) as i32 as i64 as u64
    // which sign-extends from bit 31
    const unsigned_imm: u32 = instr & 0xFFFFF000;
    // Sign extend from bit 31 (treat as i32, then extend to i64)
    return @as(i64, @as(i32, @bitCast(unsigned_imm)));
}

/// Convert i64 to field element (handles negative values)
fn signedI64ToField(comptime F: type, val: i64) F {
    if (val >= 0) {
        return F.fromU64(@intCast(val));
    } else {
        // For negative values, compute field_modulus - |val|
        return F.zero().sub(F.fromU64(@intCast(-val)));
    }
}

/// Compute Product = left * right matching Jolt's S64 * S128 semantics.
/// Jolt uses signed multiplication with truncation for the Product witness.
fn computeProduct(comptime F: type, inputs: InstructionInputs(F)) F {
    // Extract left as u64 (it's always treated as unsigned in Jolt's semantics)
    // For simplicity, we'll do 128-bit multiplication and truncate

    // Convert left field element to u64
    // Note: This is a simplification; we just compute left * right directly in the field
    // For most cases, this is sufficient since the product is used in R1CS verification

    // Actually, Jolt computes: left_s64.mul_trunc::<2, 2>(&right_s128)
    // This is signed multiplication where the result is truncated to lower 128 bits
    // then converted to field element

    // For now, multiply in field (this may need adjustment for signed semantics)
    return inputs.left.mul(inputs.right);
}

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
            return fromTraceStepWithPCMap(step, next_step, null);
        }

        /// Create cycle inputs from an execution trace step with optional PC mapper
        /// When pc_map is provided, PC values are converted from ELF addresses to bytecode indices
        pub fn fromTraceStepWithPCMap(
            step: tracer.TraceStep,
            next_step: ?tracer.TraceStep,
            pc_map: ?*const @import("../preprocessing.zig").BytecodePCMapper,
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
            // For identity-path AddOperands instructions (ADDI, ADDIW, JAL, JALR),
            // store Imm as the UNSIGNED u64 representation of the sign-extended immediate.
            // This ensures consistency between:
            //   - R1CS constraint 7: RightLookup == LeftInput + RightInput (field arithmetic)
            //   - Stage 5 RAF: identity(k) where k = x + y_unsigned (u128)
            //
            // For Load/Store/Branch, Imm stays as the signed field value.
            // This is safe because constraints 0 and 15 (which use Imm) only fire
            // for Load/Store and Branch respectively, never for ADDI/ADDIW/JAL/JALR.
            const imm = blk_imm: {
                // Virtual instructions on opcode 0x2B, dispatched by funct3:
                //   funct3=0: VirtualMULI: IMM = multiplier = 1 << shamt
                //   funct3=1: VirtualPow2: IMM = 0 (instruction_inputs right = 0)
                //   funct3=2: VirtualShiftRightBitmask: IMM = 0 (instruction_inputs right = 0)
                if (opcode == 0x2B) {
                    const funct3_2b: u3 = @truncate((step.instruction >> 12) & 0x7);
                    if (funct3_2b == 0) {
                        // VirtualMULI: The instruction encoding stores the shift amount in the I-type imm field.
                        const shamt_raw: u32 = step.instruction >> 20;
                        const shamt: u6 = @truncate(shamt_raw & 0x3F);
                        const multiplier: u64 = @as(u64, 1) << shamt;
                        break :blk_imm F.fromU64(multiplier);
                    } else {
                        // VirtualPow2 / VirtualShiftRightBitmask: imm = 0
                        break :blk_imm F.zero();
                    }
                }
                // Virtual instructions on opcode 0x5B:
                //   I-type (rs2_read=false): VirtualSRLI/VirtualSRAI: IMM = bitmask from shift
                //   R-type (rs2_read=true): VirtualSRL/VirtualSRA: IMM = 0 (uses rs2 not imm)
                if (opcode == 0x5B) {
                    if (step.rs2_read) {
                        // R-type: right operand is rs2, not imm
                        break :blk_imm F.zero();
                    } else {
                        // I-type: bitmask computed from total shift amount in imm field
                        const total_shift_raw: u32 = step.instruction >> 20;
                        const total_shift: u7 = @truncate(total_shift_raw & 0x3F);
                        const ones: u128 = (@as(u128, 1) << @intCast(64 - @as(u8, total_shift))) - 1;
                        const bitmask: u64 = @truncate(ones << total_shift);
                        break :blk_imm F.fromU64(bitmask);
                    }
                }
                // VirtualAssertHalfwordAlignment (0x22, funct3=2) and
                // VirtualAssertWordAlignment (0x22, funct3=3):
                // These are AddOperands identity-path instructions with I-type encoding.
                // IMM = unsigned u64 representation of sign-extended immediate.
                if (opcode == 0x22) {
                    const funct3_22: u3 = @truncate((step.instruction >> 12) & 0x7);
                    if (funct3_22 == 2 or funct3_22 == 3) {
                        // Signed encoding matching Jolt verifier's F::from_i128(i64 as i128)
                        const imm12_raw: u32 = @truncate(step.instruction >> 20);
                        const imm_signed: i64 = @as(i64, @as(i32, @bitCast(imm12_raw << 20)) >> 20);
                        if (imm_signed < 0) {
                            break :blk_imm F.fromU64(@intCast(-imm_signed)).neg();
                        } else {
                            break :blk_imm F.fromU64(@intCast(imm_signed));
                        }
                    }
                    // funct3=0,1 (VirtualAssertEQ, VirtualAssertValidDiv0): no imm used
                    break :blk_imm F.zero();
                }
                const is_identity_add = switch (opcode) {
                    0x13 => (step.instruction >> 12) & 0x7 == 0, // ADDI
                    0x1b => (step.instruction >> 12) & 0x7 == 0, // ADDIW
                    0x6f => true, // JAL
                    0x67 => true, // JALR
                    else => false,
                };
                if (is_identity_add) {
                    // Compute unsigned u64 representation of sign-extended immediate
                    const unsigned_imm = computeUnsignedImmediate(step.instruction);
                    break :blk_imm F.fromU64(unsigned_imm);
                }
                break :blk_imm inputs.deriveImmediate(step.instruction);
            };
            inputs.values[R1CSInputIndex.Imm.toIndex()] = imm;

            // Register values - only set for instructions that actually read the registers
            // This matches Jolt's cycle.rs1_read().unwrap_or_default().1 behavior
            //
            // Instructions that read rs1:
            //   I-type: 0x13 (ALU-I), 0x03 (LOAD), 0x67 (JALR), 0x1b (ALU-I-32)
            //   R-type: 0x33 (ALU-R), 0x3b (ALU-R-32)
            //   S-type: 0x23 (STORE)
            //   B-type: 0x63 (BRANCH)
            //
            // Instructions that DON'T read rs1 (Rs1Value = 0):
            //   U-type: 0x37 (LUI), 0x17 (AUIPC)
            //   J-type: 0x6f (JAL)
            const reads_rs1 = switch (opcode) {
                0x13, 0x03, 0x67, 0x1b, 0x33, 0x3b, 0x23, 0x63, 0x0B, 0x2B, 0x5B => true,
                0x22 => true, // VirtualAssertEQ: left = rs1 (LeftOperandIsRs1Value)
                0x42 => true, // VirtualZeroExtendWord: left = rs1 (LeftOperandIsRs1Value)
                0x62 => true, // VirtualAssertValidUnsignedRemainder: left = rs1 (LeftOperandIsRs1Value)
                // 0x02 (VirtualAdvice): does NOT read rs1, instruction_inputs = (0, 0)
                else => false,
            };
            if (reads_rs1) {
                inputs.values[R1CSInputIndex.Rs1Value.toIndex()] = F.fromU64(step.rs1_value);
            }
            // else Rs1Value stays 0 (initialized in init())

            // Instructions that read rs2:
            //   R-type: 0x33 (ALU-R), 0x3b (ALU-R-32)
            //   S-type: 0x23 (STORE)
            //   B-type: 0x63 (BRANCH)
            //
            // Instructions that DON'T read rs2 (Rs2Value = 0):
            //   All I-type, U-type, J-type
            const reads_rs2 = switch (opcode) {
                0x33, 0x3b, 0x23, 0x63 => true,
                0x22 => blk_22_rs2: {
                    // funct3=0,1 (VirtualAssertEQ, VirtualAssertValidDiv0): right = rs2
                    // funct3=2,3 (alignment assertions): right = imm, NOT rs2
                    const f3_22: u3 = @truncate((step.instruction >> 12) & 0x7);
                    break :blk_22_rs2 (f3_22 == 0 or f3_22 == 1);
                },
                0x5B => step.rs2_read, // R-type VirtualSRL/VirtualSRA read rs2
                0x62 => true, // VirtualAssertValidUnsignedRemainder: right = rs2 (RightOperandIsRs2Value)
                else => false,
            };
            if (reads_rs2) {
                inputs.values[R1CSInputIndex.Rs2Value.toIndex()] = F.fromU64(step.rs2_value);
            }
            // else Rs2Value stays 0 (initialized in init())

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

            // Determine if instruction writes to rd using TraceStep fields.
            // CRITICAL: Must use step.rd_written and step.rd_index (u8) instead of
            // extracting from instruction word. Virtual instructions write to virtual
            // registers (32+) whose indices don't fit in RISC-V's 5-bit rd field.
            // Using @truncate((instruction >> 7) & 0x1f) would map virtual register 32
            // to physical register 0, incorrectly making writes_to_rd = false.
            const is_branch = (opcode == 0x63);
            const writes_to_rd = step.rd_written and step.rd_index != 0;

            if (is_load) {
                // Constraint 2: RamReadValue == RamWriteValue (for Load)
                // Constraint 3: RamReadValue == RdWriteValue (for Load)
                // For loads: both read and write value are the value read from memory
                inputs.values[R1CSInputIndex.RamReadValue.toIndex()] = mem_val_f;
                inputs.values[R1CSInputIndex.RamWriteValue.toIndex()] = mem_val_f;
                inputs.values[R1CSInputIndex.RdWriteValue.toIndex()] = mem_val_f;
            } else if (is_store) {
                // For stores:
                // - RamReadValue = pre-value (value before write) = step.memory_pre_value
                // - RamWriteValue = post-value (value being written) = step.rs2_value
                // CRITICAL: memory_value for Store is the POST-value (written value).
                // We must use memory_pre_value for the RamReadValue (like Jolt's w.pre_value).
                // Constraint 4: Rs2Value == RamWriteValue (for Store)
                // CRITICAL: Stores don't write to rd, so RdWriteValue = 0
                const pre_val = step.memory_pre_value orelse 0;
                inputs.values[R1CSInputIndex.RamReadValue.toIndex()] = F.fromU64(pre_val); // pre-value (before write)
                inputs.values[R1CSInputIndex.RamWriteValue.toIndex()] = F.fromU64(step.rs2_value); // post-value (written value)
                inputs.values[R1CSInputIndex.RdWriteValue.toIndex()] = F.zero();
            } else {
                // Non-memory instructions
                inputs.values[R1CSInputIndex.RamReadValue.toIndex()] = F.zero();
                inputs.values[R1CSInputIndex.RamWriteValue.toIndex()] = F.zero();
                // Set RdWriteValue based on instruction type:
                // - JAL/JALR (FlagJump=1): MUST be UnexpandedPC + 4 - 2*IsCompressed
                //   regardless of rd_index (constraint 13 checks FlagJump, not rd!=0)
                // - Other instructions: rd_value if writes_to_rd, else 0
                const is_jump = (opcode == 0x6F or opcode == 0x67);
                if (is_jump) {
                    // Constraint 13: if FlagJump => RdWriteValue == UnexpandedPC + 4 - 2*IsCompressed
                    // Jolt always records the link address as rd_write_value for JAL/JALR
                    // Use step values directly since R1CS inputs may not be set yet
                    const upc_f = F.fromU64(step.unexpanded_pc);
                    const compressed_offset: u64 = if (step.is_compressed) 2 else 0;
                    const link_addr = upc_f.add(F.fromU64(4)).sub(F.fromU64(compressed_offset));
                    inputs.values[R1CSInputIndex.RdWriteValue.toIndex()] = link_addr;
                } else if (writes_to_rd) {
                    inputs.values[R1CSInputIndex.RdWriteValue.toIndex()] = F.fromU64(step.rd_value);
                } else {
                    inputs.values[R1CSInputIndex.RdWriteValue.toIndex()] = F.zero();
                }
            }

            // =================================================================
            // Instruction inputs computed using the flag formula
            // CRITICAL: Must match the sumcheck formula used in Stage 3:
            //   left = left_is_rs1 * rs1_value + left_is_pc * unexpanded_pc
            //   right = right_is_rs2 * rs2_value + right_is_imm * imm
            // =================================================================

            // First, compute the operand flags based on opcode
            const instr_opcode = opcode; // Already computed above
            const funct3_bits: u3 = @truncate((step.instruction >> 12) & 0x7);
            const funct7_bits: u7 = @truncate(step.instruction >> 25);
            const has_lookup_table = hasLookupTable(instr_opcode, funct3_bits, funct7_bits);

            // CRITICAL: Instructions without a lookup table (Load, Store, SLL, SLLI, etc.)
            // must have ALL instruction input flags = 0. In Jolt, these instructions decompose
            // into virtual sequences and never appear as raw cycles, so their R1CS witness has
            // LeftInstructionInput = 0, RightInstructionInput = 0, LookupOutput = 0.
            // This ensures the Stage 3/4 claims (left_op_claim, right_op_claim) are consistent
            // with Stage 5's combined_val = 0 for these cycles.
            const left_is_rs1: F = if (!has_lookup_table) F.zero() else switch (instr_opcode) {
                0x33 => F.one(), // R-type: left = rs1
                0x13, 0x67 => F.one(), // I-type: left = rs1
                0x63 => F.one(), // B-type: left = rs1
                0x37 => F.zero(), // LUI: no operand
                0x17 => F.zero(), // AUIPC: left = PC
                0x6F => F.zero(), // JAL: left = PC
                0x1B => F.one(), // OP-IMM-32: left = rs1
                0x3B => F.one(), // OP-32: left = rs1
                0x0B => F.one(), // VirtualSignExtendWord: left = rs1 (LeftOperandIsRs1Value)
                0x2B => F.one(), // VirtualMULI: left = rs1 (LeftOperandIsRs1Value)
                0x5B => F.one(), // VirtualSRLI: left = rs1 (LeftOperandIsRs1Value)
                0x02 => F.zero(), // VirtualAdvice: no operands (instruction_inputs = (0,0))
                0x22 => F.one(), // VirtualAssertEQ: left = rs1 (LeftOperandIsRs1Value)
                0x42 => F.one(), // VirtualZeroExtendWord: left = rs1 (LeftOperandIsRs1Value)
                0x62 => F.one(), // VirtualAssertValidUnsignedRemainder: left = rs1 (LeftOperandIsRs1Value)
                0x73 => F.zero(), // SYSTEM (ECALL/EBREAK): no operand
                0x0F => F.zero(), // FENCE: no operand
                else => F.zero(),
            };
            const left_is_pc: F = if (!has_lookup_table) F.zero() else switch (instr_opcode) {
                0x17 => F.one(), // AUIPC: left = PC
                0x6F => F.one(), // JAL: left = PC
                else => F.zero(),
            };
            const right_is_rs2: F = if (!has_lookup_table) F.zero() else switch (instr_opcode) {
                0x33 => F.one(), // R-type: right = rs2
                0x63 => F.one(), // B-type: right = rs2 (for comparison)
                0x3B => F.one(), // OP-32: right = rs2 (ADDW, SUBW, etc.)
                0x22 => blk_22_ris2: {
                    // funct3=0,1 (VirtualAssertEQ, VirtualAssertValidDiv0): right = rs2
                    // funct3=2,3 (alignment assertions): right = imm, NOT rs2
                    const f3_22r: u3 = @truncate((step.instruction >> 12) & 0x7);
                    break :blk_22_ris2 if (f3_22r == 0 or f3_22r == 1) F.one() else F.zero();
                },
                0x5B => if (step.rs2_read) F.one() else F.zero(), // R-type VirtualSRL/VirtualSRA: right = rs2
                0x62 => F.one(), // VirtualAssertValidUnsignedRemainder: right = rs2 (RightOperandIsRs2Value)
                else => F.zero(),
            };
            const right_is_imm: F = if (!has_lookup_table) F.zero() else switch (instr_opcode) {
                0x13, 0x67 => F.one(), // I-type: right = imm
                0x37 => F.one(), // LUI: right = imm (upper bits)
                0x17 => F.one(), // AUIPC: right = imm
                0x6F => F.one(), // JAL: right = imm (offset)
                0x1B => F.one(), // OP-IMM-32: right = imm (ADDIW, etc.)
                0x2B => F.one(), // VirtualMULI/VirtualPow2/VirtualShiftRightBitmask: all use RightOperandIsImm
                0x5B => if (step.rs2_read) F.zero() else F.one(), // I-type uses imm, R-type uses rs2
                0x22 => blk_22_rim: {
                    // funct3=2,3 (alignment assertions): right = imm (RightOperandIsImm)
                    // funct3=0,1 (VirtualAssertEQ, VirtualAssertValidDiv0): right = rs2
                    const f3_22i: u3 = @truncate((step.instruction >> 12) & 0x7);
                    break :blk_22_rim if (f3_22i == 2 or f3_22i == 3) F.one() else F.zero();
                },
                else => F.zero(),
            };

            // Store the flags (will be used in Stage 3 InstructionInput sumcheck)
            inputs.values[R1CSInputIndex.FlagLeftOperandIsRs1.toIndex()] = left_is_rs1;
            inputs.values[R1CSInputIndex.FlagLeftOperandIsPC.toIndex()] = left_is_pc;
            inputs.values[R1CSInputIndex.FlagRightOperandIsRs2.toIndex()] = right_is_rs2;
            inputs.values[R1CSInputIndex.FlagRightOperandIsImm.toIndex()] = right_is_imm;

            // Now compute instruction inputs using the EXACT same formula as the sumcheck:
            // This ensures consistency between witness and sumcheck computation
            const rs1_val = inputs.values[R1CSInputIndex.Rs1Value.toIndex()];
            const rs2_val = inputs.values[R1CSInputIndex.Rs2Value.toIndex()];
            const pc_val = F.fromU64(step.unexpanded_pc); // Use unexpanded_pc, not expanded PC
            const imm_val = imm; // Already computed above via deriveImmediate

            const left_instr_input = left_is_rs1.mul(rs1_val).add(left_is_pc.mul(pc_val));
            const right_instr_input = right_is_rs2.mul(rs2_val).add(right_is_imm.mul(imm_val));

            inputs.values[R1CSInputIndex.LeftInstructionInput.toIndex()] = left_instr_input;
            inputs.values[R1CSInputIndex.RightInstructionInput.toIndex()] = right_instr_input;

            // Product = left_input * right_input (for multiply instructions)
            const product = left_instr_input.mul(right_instr_input);
            inputs.values[R1CSInputIndex.Product.toIndex()] = product;

            // =================================================================
            // Lookup operands - will be set properly by setFlagsFromInstruction
            // based on which operation type it is (constraints 5-11)
            // =================================================================
            // LookupOutput is the result value from the lookup table
            // For most instructions, this is the rd_value.
            // For JAL/JALR, this is the jump target (PC + imm for JAL, (rs1 + imm) & ~1 for JALR)
            const lookup_output = computeLookupOutput(F, step);
            inputs.values[R1CSInputIndex.LookupOutput.toIndex()] = lookup_output;

            // =================================================================
            // PC values
            // PC = bytecode array index (converted from ELF address via BytecodePCMapper)
            // UnexpandedPC = raw RISC-V address (ELF address)
            //
            // In Jolt, PC is the bytecode array index (0 = NoOp, 1+ = real instructions),
            // while UnexpandedPC is the actual ELF address (0x80000000+).
            // =================================================================
            const pc_as_bytecode_idx: u64 = if (pc_map) |pm|
                @intCast(pm.getPCForStep(step))
            else
                step.pc; // Fallback: use ELF address if no pc_map
            inputs.values[R1CSInputIndex.PC.toIndex()] = F.fromU64(pc_as_bytecode_idx);
            inputs.values[R1CSInputIndex.UnexpandedPC.toIndex()] = F.fromU64(step.unexpanded_pc);

            // Use next step's values if available
            // Match Jolt's behavior: When next cycle is NoOp, all Next* values are 0.
            // This is required for the Stage 3 shift sumcheck which verifies:
            //   NextUPC[j] = UPC[j+1] for all j
            // Since NoOp cycles have UPC = 0, we need NextUPC = 0 at the boundary.
            //
            // For Stage 1 constraint 16 (NextUnexpPCUpdateOtherwise), this works because:
            // - Jump/branch instructions: condition = 0 (disabled)
            // - RISC-V programs typically end with ECALL or jump at termination
            // - The tracer stops at infinite loop detection (before the loop instruction)
            if (next_step) |ns| {
                if (ns.is_noop and !ns.is_termination_store) {
                    // Next is NoOp padding: set all Next* values to 0 (matching Jolt)
                    // In Jolt: NoOp.normalize().address = 0 and get_pc(NoOp) = 0
                    inputs.values[R1CSInputIndex.NextPC.toIndex()] = F.zero();
                    inputs.values[R1CSInputIndex.NextUnexpandedPC.toIndex()] = F.zero();
                } else {
                    // Next is a real step (or termination store dummy noop):
                    // use its PC values (bytecode index for NextPC)
                    const next_pc_idx: u64 = if (pc_map) |pm|
                        @intCast(pm.getPCForStep(ns))
                    else
                        ns.pc;
                    inputs.values[R1CSInputIndex.NextPC.toIndex()] = F.fromU64(next_pc_idx);
                    inputs.values[R1CSInputIndex.NextUnexpandedPC.toIndex()] = F.fromU64(ns.unexpanded_pc);
                }

                // NextIsVirtual: 1 if the next step has FlagVirtualInstruction=1.
                // This is required for the shift sumcheck identity:
                //   VirtualInstr[j+1] must equal NextIsVirtual[j]
                // A step is virtual if:
                //   - It has virtual_sequence_remaining > 0 (first in a W-ext decomposition, or termination store with vsr>0)
                //   - It is a VirtualSignExtendWord instruction (opcode 0x0B, which has vsr=Some(0) but is still virtual)
                //   - It is a VirtualMULI instruction (opcode 0x2B, which has vsr=Some(0) for standalone SLLI)
                const next_opcode: u8 = @truncate(ns.instruction & 0x7F);
                const next_is_virtual = (!ns.is_noop and ns.virtual_sequence_remaining > 0) or
                    (!ns.is_noop and ns.is_last_in_sequence) or // last step of virtual sequence (vsr=0 but still virtual)
                    (next_opcode == 0x0B) or // VirtualSignExtendWord is always virtual (vsr=Some(0))
                    (next_opcode == 0x2B) or // VirtualMULI is always virtual (standalone SLLI has vsr=Some(0))
                    (next_opcode == 0x5B) or // VirtualSRLI is always virtual (standalone SRLI has vsr=Some(0))
                    (next_opcode == 0x02) or // VirtualAdvice is always virtual
                    (next_opcode == 0x22) or // VirtualAssertEQ is always virtual
                    (next_opcode == 0x42) or // VirtualZeroExtendWord is always virtual
                    (next_opcode == 0x62); // VirtualAssertValidUnsignedRemainder is always virtual
                inputs.values[R1CSInputIndex.NextIsVirtual.toIndex()] = if (next_is_virtual) F.one() else F.zero();

                // NextIsFirstInSequence must match FlagIsFirstInSequence of the NEXT cycle's witness.
                // This includes both the trace step field AND setFlagsFromInstruction overrides.
                // We compute it the same way FlagIsFirstInSequence is set for the next step:
                // The trace step's is_first_in_sequence is authoritative for both
                // standalone virtual instructions and decomposed-sequence steps.
                const next_is_first = !ns.is_noop and ns.is_first_in_sequence;
                inputs.values[R1CSInputIndex.NextIsFirstInSequence.toIndex()] = if (next_is_first) F.one() else F.zero();
            } else {
                // No next step: all Next* values are 0 (matching Jolt)
                inputs.values[R1CSInputIndex.NextPC.toIndex()] = F.zero();
                inputs.values[R1CSInputIndex.NextUnexpandedPC.toIndex()] = F.zero();
                inputs.values[R1CSInputIndex.NextIsVirtual.toIndex()] = F.zero();
                inputs.values[R1CSInputIndex.NextIsFirstInSequence.toIndex()] = F.zero();
            }

            // =================================================================
            // Set remaining flags based on instruction opcode
            inputs.setFlagsFromInstruction(step.instruction, step);

            // =================================================================
            // Virtual sequence flags from trace step
            // In Jolt, virtual_sequence_remaining on an instruction determines:
            //   VirtualInstruction = (vsr != null), i.e., the instruction is part of a virtual sequence
            //   DoNotUpdateUnexpandedPC = (vsr > 0), i.e., PC should not advance for this step
            //   IsFirstInSequence = true for the first instruction in the sequence
            // This applies to both the base instruction (e.g., ADDI with vsr=1)
            // and the virtual instruction (e.g., VirtualSignExtendWord with vsr=0).
            // =================================================================
            if (step.virtual_sequence_remaining > 0 and !step.is_termination_store) {
                // This instruction is part of a virtual sequence with vsr>0
                inputs.values[R1CSInputIndex.FlagVirtualInstruction.toIndex()] = F.one();
                inputs.values[R1CSInputIndex.FlagDoNotUpdateUnexpandedPC.toIndex()] = F.one();
            } else if (step.is_last_in_sequence and !step.is_termination_store) {
                // Last step of a virtual sequence (vsr=0, but is_last_in_sequence=true)
                inputs.values[R1CSInputIndex.FlagVirtualInstruction.toIndex()] = F.one();
            }
            if (step.is_first_in_sequence) {
                inputs.values[R1CSInputIndex.FlagIsFirstInSequence.toIndex()] = F.one();
            }
            // IsLastInSequence: true ONLY for JALR (opcode 0x67) with virtual_sequence_remaining == 0.
            // Upstream Jolt only sets this in JALR's circuit_flags() implementation.
            {
                const opcode_7bit = step.instruction & 0x7F;
                const is_jalr = (opcode_7bit == 0x67);
                if (is_jalr and step.is_last_in_sequence) {
                    inputs.values[R1CSInputIndex.FlagIsLastInSequence.toIndex()] = F.one();
                }
            }

            // =================================================================
            // ShouldJump = FlagJump * (1 - NextIsNoop)
            // This must be computed AFTER setFlagsFromInstruction sets FlagJump
            // =================================================================
            const flag_jump = inputs.values[R1CSInputIndex.FlagJump.toIndex()];
            const next_is_noop_f = if (isNoopInstruction(next_step)) F.one() else F.zero();
            const one_minus_noop = F.one().sub(next_is_noop_f);
            inputs.values[R1CSInputIndex.ShouldJump.toIndex()] = flag_jump.mul(one_minus_noop);

            // =================================================================
            // Compute Product Virtualization outputs (field products)
            // These feed into Stage 2's base_evals for product sumcheck
            // =================================================================

            // IsRdNotZero: check if destination register != x0
            // Uses step.rd_index (u8) to handle virtual registers (32+) correctly.
            // CRITICAL: Store (0x23) and Branch (0x63) instructions don't write to rd,
            // so IsRdNotZero must be false regardless of the raw rd bits (which encode
            // immediate values for these formats). This matches Jolt where branch/store
            // instructions have operands.rd = None and IsRdNotZero is never set.
            const is_rd_not_zero = if (step.rd_index != 0 and !is_store and !is_branch) F.one() else F.zero();

            // BranchFlag: 1 if this is a branch instruction (opcode 0x63)
            // Note: instr_opcode is already defined above for instruction inputs
            const branch_flag_f = if (instr_opcode == 0x63) F.one() else F.zero();

            // ShouldBranch = LookupOutput * BranchFlag
            // LookupOutput contains the branch condition result (0 or 1) for branches
            const lookup_out = inputs.values[R1CSInputIndex.LookupOutput.toIndex()];
            inputs.values[R1CSInputIndex.ShouldBranch.toIndex()] = lookup_out.mul(branch_flag_f);

            // Store the raw flags for product virtualization factor evaluation
            inputs.values[R1CSInputIndex.FlagIsRdNotZero.toIndex()] = is_rd_not_zero;
            inputs.values[R1CSInputIndex.FlagBranch.toIndex()] = branch_flag_f;

            // IsNoop: In Jolt, IsNoop is only true for the synthetic Cycle::NoOp padding cycles
            // Real instructions (even ADDI x0, x0, 0) have IsNoop = false
            // We only mark this as noop for padding cycles (which are handled separately)
            // Real trace cycles always have IsNoop = false
            inputs.values[R1CSInputIndex.FlagIsNoop.toIndex()] = F.zero();

            // Instruction operand flags already set earlier (before computing instruction inputs)
            // This ensures consistency between flags and LeftInstructionInput/RightInstructionInput

            return inputs;
        }

        /// Compute signed immediate as unsigned u64 (two's complement representation).
        /// This matches Jolt's `y as u64` in to_lookup_operands() where y is the
        /// signed i128 immediate from to_instruction_inputs().
        /// Derive immediate value from instruction
        /// Compute the sign-extended immediate as an UNSIGNED u64 (two's complement).
        /// For example, imm=-1 → 0xFFFFFFFFFFFFFFFF, imm=-4 → 0xFFFFFFFFFFFFFFFC.
        /// This is used for identity-path AddOperands instructions where the u128
        /// lookup index is computed as: x as u128 + y as u64 as u128.
        fn computeUnsignedImmediate(instr: u32) u64 {
            const opcode = instr & 0x7F;
            switch (opcode) {
                0x13, 0x03, 0x67, 0x1b, 0x22 => { // I-type (including VirtualAssert alignment)
                    const imm12: u32 = instr >> 20;
                    const imm_signed: i64 = @as(i64, @as(i32, @bitCast(imm12 << 20)) >> 20);
                    return @bitCast(imm_signed);
                },
                0x6F => { // J-type (JAL)
                    const imm20 = (instr >> 31) & 0x1;
                    const imm10_1 = (instr >> 21) & 0x3FF;
                    const imm11 = (instr >> 20) & 0x1;
                    const imm19_12 = (instr >> 12) & 0xFF;
                    const raw = (imm20 << 20) | (imm19_12 << 12) | (imm11 << 11) | (imm10_1 << 1);
                    const imm_signed: i64 = @as(i64, @as(i32, @bitCast(raw << 11)) >> 11);
                    return @bitCast(imm_signed);
                },
                else => return 0,
            }
        }

        /// Derive immediate value from instruction (as signed field element)
        pub fn deriveImmediate(self: *Self, instr: u32) F {
            _ = self;
            const opcode = instr & 0x7F;
            switch (opcode) {
                0x13, 0x67, 0x1b => { // I-type (FormatI: u64 imm): ADDI, JALR, ADDIW
                    // FormatI stores imm as u64. NormalizedOperands.imm = u64 as i128 (zero-ext → positive)
                    // F::from_i128(positive) = F.fromU64(u64). Use unsigned encoding.
                    const imm12: u32 = instr >> 20;
                    const imm_signed: i64 = @as(i64, @as(i32, @bitCast(imm12 << 20)) >> 20);
                    return F.fromU64(@as(u64, @bitCast(imm_signed)));
                },
                0x03 => { // LOAD (FormatLoad: i64 imm)
                    // FormatLoad stores imm as i64. NormalizedOperands.imm = i64 as i128 (sign-ext → can be negative)
                    // F::from_i128(negative) = F(p - |imm|). Use signed encoding.
                    const imm12: u32 = instr >> 20;
                    const imm_signed: i64 = @as(i64, @as(i32, @bitCast(imm12 << 20)) >> 20);
                    if (imm_signed < 0) {
                        return F.fromU64(@intCast(-imm_signed)).neg();
                    }
                    return F.fromU64(@intCast(imm_signed));
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
                    // Sign-extend 32-bit immediate to 64-bit, treat as unsigned u64
                    // Matches Jolt's FormatU.parse: `as i32 as i64 as u64`
                    const imm_u32: u32 = instr & 0xFFFFF000;
                    const imm_sext: u64 = @bitCast(@as(i64, @as(i32, @bitCast(imm_u32))));
                    return F.fromU64(imm_sext);
                },
                else => return F.zero(),
            }
        }

        /// Compute the u128 lookup index for identity-path instructions.
        /// This matches Jolt's to_lookup_operands() which returns u128 results.
        /// For AddOperands: x as u128 + y as u64 as u128 (where y is the signed imm reinterpreted as u64)
        /// For SubtractOperands: x as u128 + 2^64 - y as u128
        /// For MultiplyOperands: x as u128 * y as u128
        pub fn computeU128LookupOperand(instr: u32, step: tracer.TraceStep) F {
            const opcode: u8 = @truncate(instr & 0x7F);
            const funct3 = (instr >> 12) & 0x7;
            const funct7 = (instr >> 25) & 0x7F;

            switch (opcode) {
                0x33 => { // R-type: only identity-path instructions use u128 lookup operand
                    if (funct7 == 0x01) {
                        if (funct3 == 0x0) {
                            // MUL: x * y as u128
                            const result = @as(u128, step.rs1_value) * @as(u128, step.rs2_value);
                            return F.fromU128(result);
                        }
                        if (funct3 == 3) {
                            // MULHU: x * y as u128
                            const result = @as(u128, step.rs1_value) * @as(u128, step.rs2_value);
                            return F.fromU128(result);
                        }
                        // Other M-extension (DIVU, REMU, etc.): interleaved, not identity
                        return F.zero();
                    }
                    if (funct7 == 0x20 and funct3 == 0x0) {
                        // SUB: x + 2^64 - y
                        const result = @as(u128, step.rs1_value) + (@as(u128, 1) << 64) - @as(u128, step.rs2_value);
                        return F.fromU128(result);
                    }
                    if (funct3 == 0x0 and funct7 == 0x0) {
                        // ADD: x + y as u128
                        const result = @as(u128, step.rs1_value) + @as(u128, step.rs2_value);
                        return F.fromU128(result);
                    }
                    // XOR, AND, OR, SLT, SLTU, SRL, SRA: interleaved, u128 operand not used
                    return F.zero();
                },
                0x13 => { // ADDI
                    if (funct3 == 0) {
                        const imm12_raw: u32 = @truncate(instr >> 20);
                        const imm_signed: i64 = @as(i64, @as(i32, @bitCast(imm12_raw << 20)) >> 20);
                        const imm_u64: u64 = @bitCast(imm_signed);
                        return F.fromU128(@as(u128, step.rs1_value) + @as(u128, imm_u64));
                    }
                    return F.zero();
                },
                0x1b => { // ADDIW
                    if (funct3 == 0) {
                        const imm12_raw: u32 = @truncate(instr >> 20);
                        const imm_signed: i64 = @as(i64, @as(i32, @bitCast(imm12_raw << 20)) >> 20);
                        const imm_u64: u64 = @bitCast(imm_signed);
                        return F.fromU128(@as(u128, step.rs1_value) + @as(u128, imm_u64));
                    }
                    return F.zero();
                },
                0x37 => { // LUI: (0, imm) sign-extended to 64 bits
                    const imm_u32: u32 = instr & 0xFFFFF000;
                    const imm_sext: u64 = @bitCast(@as(i64, @as(i32, @bitCast(imm_u32))));
                    return F.fromU128(@as(u128, imm_sext));
                },
                0x17 => { // AUIPC: (0, PC + imm) sign-extended to 64 bits
                    const imm_u32: u32 = instr & 0xFFFFF000;
                    const imm_sext: u64 = @bitCast(@as(i64, @as(i32, @bitCast(imm_u32))));
                    return F.fromU128(@as(u128, step.unexpanded_pc) + @as(u128, imm_sext));
                },
                0x6f => { // JAL: (0, PC + imm)
                    const imm20: u32 = ((@as(u32, instr >> 31) & 1) << 19) |
                        ((@as(u32, instr >> 12) & 0xFF) << 11) |
                        ((@as(u32, instr >> 20) & 1) << 10) |
                        ((@as(u32, instr >> 21) & 0x3FF));
                    const imm_signed: i64 = @as(i64, @as(i32, @bitCast(imm20 << 12)) >> 11);
                    const imm_u64: u64 = @bitCast(imm_signed);
                    return F.fromU128(@as(u128, step.unexpanded_pc) + @as(u128, imm_u64));
                },
                0x67 => { // JALR: (0, rs1 + imm)
                    const imm12_raw: u32 = @truncate(instr >> 20);
                    const imm_signed: i64 = @as(i64, @as(i32, @bitCast(imm12_raw << 20)) >> 20);
                    const imm_u64: u64 = @bitCast(imm_signed);
                    return F.fromU128(@as(u128, step.rs1_value) + @as(u128, imm_u64));
                },
                0x0B => { // VirtualSignExtendWord: lookup index = rs1_val
                    // to_lookup_operands returns (0, rs1 + 0) = (0, rs1)
                    return F.fromU128(@as(u128, step.rs1_value));
                },
                0x2B => { // Virtual instructions on opcode 0x2B, dispatched by funct3
                    if (funct3 == 0) {
                        // VirtualMULI: to_lookup_operands returns (0, rs1 * imm)
                        const shamt_raw: u32 = instr >> 20;
                        const shamt: u6 = @truncate(shamt_raw & 0x3F);
                        const multiplier: u64 = @as(u64, 1) << shamt;
                        return F.fromU128(@as(u128, step.rs1_value) * @as(u128, multiplier));
                    } else {
                        // funct3=1 (VirtualPow2), funct3=2 (VirtualShiftRightBitmask):
                        // AddOperands: to_lookup_operands returns (0, rs1 + 0) = (0, rs1)
                        return F.fromU128(@as(u128, step.rs1_value));
                    }
                },
                0x3b => { // ADDW/SUBW - should no longer appear in traces after W-ext decomposition
                    if (funct3 == 0 and funct7 == 0) {
                        // ADDW: x + y
                        return F.fromU128(@as(u128, step.rs1_value) + @as(u128, step.rs2_value));
                    }
                    if (funct3 == 0 and funct7 == 0x20) {
                        // SUBW: x + 2^64 - y
                        return F.fromU128(@as(u128, step.rs1_value) + (@as(u128, 1) << 64) - @as(u128, step.rs2_value));
                    }
                    return F.zero();
                },
                0x02 => { // VirtualAdvice: lookup operands = (0, advice_value)
                    // In Jolt, advice is the oracle-provided value (quotient or remainder)
                    // to_lookup_operands returns (0, self.instruction.advice as u128)
                    // The advice value is stored in step.rd_value (written to rd register)
                    return F.fromU128(@as(u128, step.rd_value));
                },
                0x22 => { // VirtualAssert* instructions on opcode 0x22
                    if (funct3 == 2 or funct3 == 3) {
                        // Wrapping u64 addition matching tracer's lookup index
                        const imm12_raw: u32 = @truncate(instr >> 20);
                        const imm_signed: i64 = @as(i64, @as(i32, @bitCast(imm12_raw << 20)) >> 20);
                        const addr: u64 = step.rs1_value +% @as(u64, @bitCast(imm_signed));
                        return F.fromU128(@as(u128, addr));
                    }
                    // funct3=0,1 (VirtualAssertEQ, VirtualAssertValidDiv0): interleaved, not identity
                    return F.zero();
                },
                0x42 => { // VirtualZeroExtendWord: lookup operands = (0, rs1 + 0) = (0, rs1)
                    // AddOperands: RightLookup = u128(left + right) = u128(rs1 + 0) = rs1
                    return F.fromU128(@as(u128, step.rs1_value));
                },
                else => return F.zero(),
            }
        }

        /// Set circuit flags and lookup operands based on instruction
        ///
        /// RightLookupOperand uses u128 arithmetic (matching Jolt's to_lookup_operands):
        /// - AddOperands: RightLookup = F(x as u128 + y as u64 as u128)
        /// - SubtractOperands: RightLookup = F(x as u128 + 2^64 - y as u128)
        /// - MultiplyOperands: RightLookup = F(x as u128 * y as u128)
        /// - Others: RightLookup = RightInstructionInput
        ///
        /// Note: This means R1CS constraint 7 (RightLookup == LeftInput + RightInput)
        /// may not hold when the u128 result differs from field arithmetic. This matches
        /// Jolt's behavior where to_lookup_operands() returns u128 values.
        fn setFlagsFromInstruction(self: *Self, instr: u32, step: tracer.TraceStep) void {
            const opcode: u8 = @truncate(instr & 0x7F);
            const funct3 = (instr >> 12) & 0x7;
            const funct7 = (instr >> 25) & 0x7F;

            // CRITICAL: Check if this instruction has a lookup table.
            const funct3_u3: u3 = @truncate(funct3);
            const funct7_u7: u7 = @truncate(funct7);
            if (!hasLookupTable(opcode, funct3_u3, funct7_u7)) {
                self.values[R1CSInputIndex.LeftLookupOperand.toIndex()] = F.zero();
                self.values[R1CSInputIndex.RightLookupOperand.toIndex()] = F.zero();
                return;
            }

            // Get input values for lookup operand computation (used for non-identity-path)
            const left_input = self.values[R1CSInputIndex.LeftInstructionInput.toIndex()];
            const right_input = self.values[R1CSInputIndex.RightInstructionInput.toIndex()];

            // Compute u128 lookup operand for identity-path instructions
            // This matches Jolt's to_lookup_operands() which returns u128 results
            const u128_right_lookup = computeU128LookupOperand(instr, step);

            switch (opcode) {
                0x33 => { // R-type (ADD, SUB, MUL, XOR, AND, OR, SLT, SLTU, SRL, SRA, etc.)
                    if (funct7 == 0x01) {
                        // M-extension
                        if (funct3 == 0x0) { // MUL
                            self.values[R1CSInputIndex.FlagMultiplyOperands.toIndex()] = F.one();
                            self.values[R1CSInputIndex.LeftLookupOperand.toIndex()] = F.zero();
                            self.values[R1CSInputIndex.RightLookupOperand.toIndex()] = u128_right_lookup;
                        } else if (funct3 == 0x3) { // MULHU
                            self.values[R1CSInputIndex.FlagMultiplyOperands.toIndex()] = F.one();
                            self.values[R1CSInputIndex.LeftLookupOperand.toIndex()] = F.zero();
                            self.values[R1CSInputIndex.RightLookupOperand.toIndex()] = u128_right_lookup;
                        } else {
                            // DIVU, REMU, MULHSU, etc. - interleaved operands
                            self.values[R1CSInputIndex.LeftLookupOperand.toIndex()] = left_input;
                            self.values[R1CSInputIndex.RightLookupOperand.toIndex()] = right_input;
                        }
                    } else if (funct3 == 0x0 and funct7 == 0x20) {
                        // SUB
                        self.values[R1CSInputIndex.FlagSubtractOperands.toIndex()] = F.one();
                        self.values[R1CSInputIndex.LeftLookupOperand.toIndex()] = F.zero();
                        self.values[R1CSInputIndex.RightLookupOperand.toIndex()] = u128_right_lookup;
                    } else if (funct3 == 0x0 and funct7 == 0x0) {
                        // ADD: AddOperands, identity path
                        self.values[R1CSInputIndex.FlagAddOperands.toIndex()] = F.one();
                        self.values[R1CSInputIndex.LeftLookupOperand.toIndex()] = F.zero();
                        self.values[R1CSInputIndex.RightLookupOperand.toIndex()] = u128_right_lookup;
                    } else {
                        // XOR, AND, OR, SLT, SLTU, SRL, SRA: interleaved operands
                        // These do NOT set AddOperands/SubtractOperands/MultiplyOperands
                        // Jolt: is_interleaved_operands = true for these
                        self.values[R1CSInputIndex.LeftLookupOperand.toIndex()] = left_input;
                        self.values[R1CSInputIndex.RightLookupOperand.toIndex()] = right_input;
                    }
                    self.values[R1CSInputIndex.FlagWriteLookupOutputToRD.toIndex()] = F.one();
                },
                0x13 => { // I-type ALU
                    const funct3_13: u3 = @truncate((instr >> 12) & 0x7);
                    if (funct3_13 == 0) {
                        // ADDI: AddOperands, u128 right lookup
                        self.values[R1CSInputIndex.FlagAddOperands.toIndex()] = F.one();
                        self.values[R1CSInputIndex.LeftLookupOperand.toIndex()] = F.zero();
                        self.values[R1CSInputIndex.RightLookupOperand.toIndex()] = u128_right_lookup;
                    } else {
                        // SLTI, SLTIU, XORI, SRLI, SRAI, ORI, ANDI: interleaved
                        self.values[R1CSInputIndex.LeftLookupOperand.toIndex()] = left_input;
                        self.values[R1CSInputIndex.RightLookupOperand.toIndex()] = right_input;
                    }
                    self.values[R1CSInputIndex.FlagWriteLookupOutputToRD.toIndex()] = F.one();
                },
                0x6F => { // JAL: AddOperands
                    self.values[R1CSInputIndex.FlagJump.toIndex()] = F.one();
                    self.values[R1CSInputIndex.FlagAddOperands.toIndex()] = F.one();
                    self.values[R1CSInputIndex.LeftLookupOperand.toIndex()] = F.zero();
                    self.values[R1CSInputIndex.RightLookupOperand.toIndex()] = u128_right_lookup;
                },
                0x67 => { // JALR: AddOperands
                    self.values[R1CSInputIndex.FlagJump.toIndex()] = F.one();
                    self.values[R1CSInputIndex.FlagAddOperands.toIndex()] = F.one();
                    self.values[R1CSInputIndex.LeftLookupOperand.toIndex()] = F.zero();
                    self.values[R1CSInputIndex.RightLookupOperand.toIndex()] = u128_right_lookup;
                },
                0x63 => { // Branch: NOT Add+Sub+Mul
                    self.values[R1CSInputIndex.LeftLookupOperand.toIndex()] = left_input;
                    self.values[R1CSInputIndex.RightLookupOperand.toIndex()] = right_input;
                },
                0x37 => { // LUI: AddOperands
                    self.values[R1CSInputIndex.FlagAddOperands.toIndex()] = F.one();
                    self.values[R1CSInputIndex.FlagWriteLookupOutputToRD.toIndex()] = F.one();
                    self.values[R1CSInputIndex.LeftLookupOperand.toIndex()] = F.zero();
                    self.values[R1CSInputIndex.RightLookupOperand.toIndex()] = u128_right_lookup;
                },
                0x17 => { // AUIPC: AddOperands
                    self.values[R1CSInputIndex.FlagAddOperands.toIndex()] = F.one();
                    self.values[R1CSInputIndex.FlagWriteLookupOutputToRD.toIndex()] = F.one();
                    self.values[R1CSInputIndex.LeftLookupOperand.toIndex()] = F.zero();
                    self.values[R1CSInputIndex.RightLookupOperand.toIndex()] = u128_right_lookup;
                },
                0x0B => { // VirtualSignExtendWord: AddOperands, WriteLookupOutputToRD
                    // VirtualSignExtendWord uses AddOperands with u128 lookup
                    // Lookup operands: (0, rs1_val) - the value to sign-extend
                    self.values[R1CSInputIndex.FlagAddOperands.toIndex()] = F.one();
                    self.values[R1CSInputIndex.FlagWriteLookupOutputToRD.toIndex()] = F.one();
                    self.values[R1CSInputIndex.LeftLookupOperand.toIndex()] = F.zero();
                    self.values[R1CSInputIndex.RightLookupOperand.toIndex()] = u128_right_lookup;
                    // VirtualInstruction flag: always true for VirtualSignExtendWord
                    // (virtual_sequence_remaining is Some(0), which means is_some() = true)
                    self.values[R1CSInputIndex.FlagVirtualInstruction.toIndex()] = F.one();
                    // DoNotUpdateUnexpandedPC: false for VirtualSignExtendWord (vsr=0, 0!=0 is false)
                    // IsFirstInSequence: false for VirtualSignExtendWord (it's the second in sequence)
                },
                0x2B => { // Virtual instructions on opcode 0x2B, dispatched by funct3
                    const funct3_2b: u3 = @truncate(funct3);
                    switch (funct3_2b) {
                        0 => {
                            // VirtualMULI: MultiplyOperands, WriteLookupOutputToRD
                            // Lookup operands: (0, rs1 * imm)
                            self.values[R1CSInputIndex.FlagMultiplyOperands.toIndex()] = F.one();
                            self.values[R1CSInputIndex.FlagWriteLookupOutputToRD.toIndex()] = F.one();
                            self.values[R1CSInputIndex.LeftLookupOperand.toIndex()] = F.zero();
                            self.values[R1CSInputIndex.RightLookupOperand.toIndex()] = u128_right_lookup;
                        },
                        1 => {
                            // VirtualPow2: AddOperands, WriteLookupOutputToRD
                            // Lookup operands: (0, rs1 + 0) = (0, rs1)
                            self.values[R1CSInputIndex.FlagAddOperands.toIndex()] = F.one();
                            self.values[R1CSInputIndex.FlagWriteLookupOutputToRD.toIndex()] = F.one();
                            self.values[R1CSInputIndex.LeftLookupOperand.toIndex()] = F.zero();
                            self.values[R1CSInputIndex.RightLookupOperand.toIndex()] = u128_right_lookup;
                        },
                        2 => {
                            // VirtualShiftRightBitmask: AddOperands, WriteLookupOutputToRD
                            // Lookup operands: (0, rs1 + 0) = (0, rs1)
                            self.values[R1CSInputIndex.FlagAddOperands.toIndex()] = F.one();
                            self.values[R1CSInputIndex.FlagWriteLookupOutputToRD.toIndex()] = F.one();
                            self.values[R1CSInputIndex.LeftLookupOperand.toIndex()] = F.zero();
                            self.values[R1CSInputIndex.RightLookupOperand.toIndex()] = u128_right_lookup;
                        },
                        else => {
                            self.values[R1CSInputIndex.LeftLookupOperand.toIndex()] = left_input;
                            self.values[R1CSInputIndex.RightLookupOperand.toIndex()] = right_input;
                        },
                    }
                    // VirtualInstruction: ALWAYS true for all 0x2B instructions.
                    // In Jolt, opcode 0x2B always has virtual_sequence_remaining=Some(...),
                    // so VirtualInstruction = is_some() = true unconditionally.
                    // Cases:
                    //   Standalone SLLI: vsr=Some(0), VirtInstr=true, IsFirst=true, DoNotUpdateUPC=false
                    //   SLLIW first step: vsr=Some(1), VirtInstr=true, IsFirst=true, DoNotUpdateUPC=true
                    //   SLL step 1 (VirtualPow2): vsr>0, VirtInstr=true, IsFirst=true, DoNotUpdateUPC=true
                    //   SRL/SRA step 1 (VirtualShiftRightBitmask): vsr>0, VirtInstr=true, IsFirst=true, DoNotUpdateUPC=true
                    self.values[R1CSInputIndex.FlagVirtualInstruction.toIndex()] = F.one();
                    if (step.virtual_sequence_remaining == 0) {
                        // Standalone virtual instruction (from SLLI): first and only step in sequence.
                        // IsFirstInSequence = true, DoNotUpdateUnexpandedPC = false (default).
                        self.values[R1CSInputIndex.FlagIsFirstInSequence.toIndex()] = F.one();
                    } else {
                        // Multi-step sequence first step (vsr>0): DoNotUpdateUnexpandedPC = true.
                        // VirtualInstruction and IsFirstInSequence already set by vsr>0 block above.
                        self.values[R1CSInputIndex.FlagDoNotUpdateUnexpandedPC.toIndex()] = F.one();
                    }
                },
                0x5B => { // Virtual instructions on opcode 0x5B (I-type and R-type)
                    // Both I-type (VirtualSRLI/VirtualSRAI) and R-type (VirtualSRL/VirtualSRA)
                    // use WriteLookupOutputToRD with interleaved operands (NO Add/Sub/Mul)
                    self.values[R1CSInputIndex.FlagWriteLookupOutputToRD.toIndex()] = F.one();
                    self.values[R1CSInputIndex.LeftLookupOperand.toIndex()] = left_input;
                    self.values[R1CSInputIndex.RightLookupOperand.toIndex()] = right_input;
                    // VirtualInstruction: ALWAYS true for all 0x5B instructions.
                    // Cases:
                    //   Standalone SRLI/SRAI: vsr=Some(0), VirtInstr=true, IsFirst=true, DoNotUpdateUPC=false
                    //   SRLIW/SRAIW middle step: vsr=Some(1), VirtInstr=true, IsFirst=false, DoNotUpdateUPC=true
                    //   R-type VirtualSRL/VirtualSRA (SRL/SRA step 2): vsr=Some(0), VirtInstr=true, IsLast=true
                    self.values[R1CSInputIndex.FlagVirtualInstruction.toIndex()] = F.one();
                    if (step.virtual_sequence_remaining == 0) {
                        // vsr=0: standalone SRLI/SRAI or last step of decomposed load (LBU/LHU/LWU/LB/LH).
                        // IsFirstInSequence is set from step.is_first_in_sequence (line 1535),
                        // not from instruction type, because the same I-type 0x5B can appear
                        // in both standalone (IsFirst=true) and decomposed-last-step (IsFirst=false) contexts.
                        // R-type with vsr=0: IsFirstInSequence stays false (it's step 2 of SRL/SRA).
                    } else {
                        // Middle step (vsr>0): DoNotUpdateUnexpandedPC = true.
                        self.values[R1CSInputIndex.FlagDoNotUpdateUnexpandedPC.toIndex()] = F.one();
                    }
                },
                0x02 => { // VirtualAdvice: Advice, WriteLookupOutputToRD
                    // VirtualAdvice injects oracle value; uses RangeCheck table (identity)
                    // Lookup operands: (0, advice)
                    self.values[R1CSInputIndex.FlagAdvice.toIndex()] = F.one();
                    self.values[R1CSInputIndex.FlagWriteLookupOutputToRD.toIndex()] = F.one();
                    self.values[R1CSInputIndex.LeftLookupOperand.toIndex()] = F.zero();
                    self.values[R1CSInputIndex.RightLookupOperand.toIndex()] = u128_right_lookup;
                    // Always part of a virtual sequence
                    self.values[R1CSInputIndex.FlagVirtualInstruction.toIndex()] = F.one();
                    if (step.virtual_sequence_remaining > 0) {
                        self.values[R1CSInputIndex.FlagDoNotUpdateUnexpandedPC.toIndex()] = F.one();
                    }
                },
                0x22 => { // VirtualAssert* instructions on opcode 0x22, dispatched by funct3
                    const funct3_22: u3 = @truncate(funct3);
                    switch (funct3_22) {
                        0, 1 => {
                            // funct3=0: VirtualAssertEQ: Assert (interleaved operands)
                            // funct3=1: VirtualAssertValidDiv0: Assert (interleaved operands)
                            // Lookup operands: (rs1, rs2) interleaved
                            self.values[R1CSInputIndex.FlagAssert.toIndex()] = F.one();
                            self.values[R1CSInputIndex.LeftLookupOperand.toIndex()] = left_input;
                            self.values[R1CSInputIndex.RightLookupOperand.toIndex()] = right_input;
                        },
                        2, 3 => {
                            // funct3=2: VirtualAssertHalfwordAlignment: Assert + AddOperands
                            // funct3=3: VirtualAssertWordAlignment: Assert + AddOperands
                            // Lookup operands: (0, rs1 + imm) via AddOperands identity path
                            // NOTE: no WriteLookupOutputToRD (alignment assertions have no rd)
                            self.values[R1CSInputIndex.FlagAssert.toIndex()] = F.one();
                            self.values[R1CSInputIndex.FlagAddOperands.toIndex()] = F.one();
                            self.values[R1CSInputIndex.LeftLookupOperand.toIndex()] = F.zero();
                            self.values[R1CSInputIndex.RightLookupOperand.toIndex()] = u128_right_lookup;
                        },
                        else => {
                            self.values[R1CSInputIndex.LeftLookupOperand.toIndex()] = left_input;
                            self.values[R1CSInputIndex.RightLookupOperand.toIndex()] = right_input;
                        },
                    }
                    // Always part of a virtual sequence
                    self.values[R1CSInputIndex.FlagVirtualInstruction.toIndex()] = F.one();
                    if (step.virtual_sequence_remaining > 0) {
                        self.values[R1CSInputIndex.FlagDoNotUpdateUnexpandedPC.toIndex()] = F.one();
                    }
                },
                0x42 => { // VirtualZeroExtendWord: AddOperands, WriteLookupOutputToRD
                    // VirtualZeroExtendWord zeros upper bits; uses LowerHalfWord table
                    // Lookup operands: (0, rs1) via AddOperands
                    self.values[R1CSInputIndex.FlagAddOperands.toIndex()] = F.one();
                    self.values[R1CSInputIndex.FlagWriteLookupOutputToRD.toIndex()] = F.one();
                    self.values[R1CSInputIndex.LeftLookupOperand.toIndex()] = F.zero();
                    self.values[R1CSInputIndex.RightLookupOperand.toIndex()] = u128_right_lookup;
                    // Always part of a virtual sequence
                    self.values[R1CSInputIndex.FlagVirtualInstruction.toIndex()] = F.one();
                    if (step.virtual_sequence_remaining > 0) {
                        self.values[R1CSInputIndex.FlagDoNotUpdateUnexpandedPC.toIndex()] = F.one();
                    }
                },
                0x3B => {
                    // OP-32 (ADDW, SUBW, VirtualChangeDivisorW, etc.)
                    const funct3_3b: u3 = @truncate((instr >> 12) & 0x7);
                    const funct7_3b: u7 = @truncate(instr >> 25);
                    if (funct3_3b == 0 and funct7_3b == 0) {
                        // ADDW: AddOperands, WriteLookupOutputToRD
                        self.values[R1CSInputIndex.FlagAddOperands.toIndex()] = F.one();
                        self.values[R1CSInputIndex.FlagWriteLookupOutputToRD.toIndex()] = F.one();
                        self.values[R1CSInputIndex.LeftLookupOperand.toIndex()] = F.zero();
                        self.values[R1CSInputIndex.RightLookupOperand.toIndex()] = u128_right_lookup;
                    } else if (funct3_3b == 0 and funct7_3b == 0x20) {
                        // SUBW: SubtractOperands, WriteLookupOutputToRD
                        self.values[R1CSInputIndex.FlagSubtractOperands.toIndex()] = F.one();
                        self.values[R1CSInputIndex.FlagWriteLookupOutputToRD.toIndex()] = F.one();
                        self.values[R1CSInputIndex.LeftLookupOperand.toIndex()] = F.zero();
                        self.values[R1CSInputIndex.RightLookupOperand.toIndex()] = u128_right_lookup;
                    } else if (funct3_3b == 6 and funct7_3b == 0x01) {
                        // VirtualChangeDivisorW: interleaved, WriteLookupOutputToRD
                        // Jolt's to_instruction_inputs: (rs1 as u32 as u64, rs2 as i128)
                        // to_lookup_operands: (rs1 as u32 as u64, rs2 as u64)
                        // Left operand is rs1 TRUNCATED to 32 bits (zero-extended)
                        const rs1_lower32: u64 = step.rs1_value & 0xFFFFFFFF;
                        self.values[R1CSInputIndex.FlagWriteLookupOutputToRD.toIndex()] = F.one();
                        self.values[R1CSInputIndex.LeftLookupOperand.toIndex()] = F.fromU64(rs1_lower32);
                        self.values[R1CSInputIndex.RightLookupOperand.toIndex()] = F.fromU64(step.rs2_value);
                        // Always part of a virtual sequence
                        self.values[R1CSInputIndex.FlagVirtualInstruction.toIndex()] = F.one();
                        if (step.virtual_sequence_remaining > 0) {
                            self.values[R1CSInputIndex.FlagDoNotUpdateUnexpandedPC.toIndex()] = F.one();
                        }
                    } else {
                        // Other OP-32: interleaved operands
                        self.values[R1CSInputIndex.LeftLookupOperand.toIndex()] = left_input;
                        self.values[R1CSInputIndex.RightLookupOperand.toIndex()] = right_input;
                    }
                },
                0x62 => { // VirtualAssertValidUnsignedRemainder: Assert (interleaved operands)
                    // VirtualAssertValidUnsignedRemainder checks remainder < divisor
                    // Uses ValidUnsignedRemainder table
                    // Lookup operands: (rs1, rs2) interleaved
                    self.values[R1CSInputIndex.FlagAssert.toIndex()] = F.one();
                    self.values[R1CSInputIndex.LeftLookupOperand.toIndex()] = left_input;
                    self.values[R1CSInputIndex.RightLookupOperand.toIndex()] = right_input;
                    // Always part of a virtual sequence
                    self.values[R1CSInputIndex.FlagVirtualInstruction.toIndex()] = F.one();
                    if (step.virtual_sequence_remaining > 0) {
                        self.values[R1CSInputIndex.FlagDoNotUpdateUnexpandedPC.toIndex()] = F.one();
                    }
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

        /// Create witness values for a NoOp padding cycle.
        ///
        /// In Jolt, the trace is padded with Cycle::NoOp after the last real cycle.
        /// NoOp cycles have all values = 0 except for two flags:
        /// - FlagDoNotUpdateUnexpandedPC = 1
        /// - FlagIsNoop = 1
        ///
        /// Reference: jolt-core/src/zkvm/instruction/mod.rs (Instruction::NoOp flags)
        /// Create witness values for the synthetic termination Store instruction.
        ///
        /// This is used for the termination steps that write 1 to the termination address.
        /// The sequence is: NoOp(dummy) → LUI(vsr=2) → ADDI(vsr=1) → SB(vsr=0).
        ///
        /// For the NoOp dummy and LUI/ADDI (vsr>0): DNUPC=true, VI=true.
        /// For the SB anchor (vsr=0): VI=true, DNUPC=false.
        ///   - Constraint 17 (if VI then NextPC=PC+1) holds: next is JAL at PC=tbpc+3.
        ///   - Constraint 16 (NextUPC=UPC+4): NextUPC=0+4=4, JAL has UPC=4.
        ///   - This matches vanilla Jolt's circuit_flags for SD with vsr=Some(0).
        pub fn createTerminationStoreWitness(
            step: tracer.TraceStep,
            next_step: ?tracer.TraceStep,
            pc_map: ?*const @import("../preprocessing.zig").BytecodePCMapper,
        ) Self {
            // Build the base witness using the normal instruction path
            var inputs = fromTraceStepWithPCMap(step, next_step, pc_map);

            if (step.is_noop) {
                // Dummy noop termination step: maps to bytecode k=0 (NoOp entry).
                // Bytecode entry at k=0 has:
                //   circuit_flags[DoNotUpdateUnexpandedPC] = true
                //   instruction_flags[IsNoop] = true
                // R1CS witness must match, so set FlagIsNoop=1 and DNUPC=1.
                // This also ensures product virtualization picks up NextIsNoop=1
                // for the preceding JAL cycle (JAL's ShouldJump = Jump*(1-NextIsNoop) = 0).
                inputs.values[R1CSInputIndex.FlagDoNotUpdateUnexpandedPC.toIndex()] = F.one();
                inputs.values[R1CSInputIndex.FlagIsNoop.toIndex()] = F.one();
            } else if (step.virtual_sequence_remaining > 0) {
                // Non-anchor termination instruction (LUI vsr=2, ADDI vsr=1):
                // Bytecode entry has VirtualInstruction=true, DoNotUpdateUnexpandedPC=true.
                // R1CS constraint 17 (if VirtualInstruction then NextPC==PC+1) holds
                // because LUI→ADDI and ADDI→SB have consecutive PCs.
                inputs.values[R1CSInputIndex.FlagDoNotUpdateUnexpandedPC.toIndex()] = F.one();
                inputs.values[R1CSInputIndex.FlagVirtualInstruction.toIndex()] = F.one();
                // Override NextIsVirtual=1: the next termination step also has VI=true.
                // For ADDI (vsr=1) → SB (vsr=0): SB has VI=true (vsr=Some(0) in vanilla Jolt),
                // but fromTraceStepWithPCMap computes NextIsVirtual=0 because SB's vsr=0.
                // The shift sumcheck requires VirtualInstr[j+1] = NextIsVirtual[j].
                inputs.values[R1CSInputIndex.NextIsVirtual.toIndex()] = F.one();
            } else {
                // SB anchor (vsr=0): VirtualInstruction=true, DoNotUpdatePC=false.
                // This matches vanilla Jolt's circuit_flags for SD with vsr=Some(0):
                //   VI = vsr.is_some() = true
                //   DNUPC = vsr.map_or(false, |v| v > 0) = false
                // Constraint 17 (if VI then NextPC=PC+1): NextPC=tbpc+3 (JAL) ✓
                // Constraint 16 (NextUPC=UPC+4-4*DNUPC): NextUPC=0+4=4, JAL has UPC=4 ✓
                inputs.values[R1CSInputIndex.FlagVirtualInstruction.toIndex()] = F.one();
                // DNUPC stays 0 (default from fromTraceStepWithPCMap)
            }

            return inputs;
        }

        pub fn createNoopWitness() Self {
            var inputs = Self{
                .values = [_]F{F.zero()} ** R1CSInputIndex.NUM_INPUTS,
            };

            // Only two flags are true for NoOp cycles
            inputs.values[R1CSInputIndex.FlagDoNotUpdateUnexpandedPC.toIndex()] = F.one();
            inputs.values[R1CSInputIndex.FlagIsNoop.toIndex()] = F.one();

            // All other values remain zero:
            // - PC = 0, NextPC = 0, UnexpandedPC = 0, NextUnexpandedPC = 0
            // - All register values = 0
            // - All instruction inputs/outputs = 0
            // - All other flags = 0
            // - ShouldJump = 0 (Jump is 0, so Jump * (1 - NextIsNoop) = 0)

            return inputs;
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

        /// Generate witness for all cycles in trace (without PC mapping).
        ///
        /// The trace must be pre-padded with NoOp cycles (via padWithNoop).
        /// - NoOp padding cycles use createNoopWitness() (all zeros with IsNoop=1)
        /// - Real cycles use fromTraceStep() with the next step (real or NoOp)
        ///
        /// Reference: jolt-core/src/zkvm/prover.rs (trace.resize with Cycle::NoOp)
        pub fn generateWitness(
            self: *Self,
            trace: *const tracer.ExecutionTrace,
        ) ![]R1CSCycleInputs(F) {
            return self.generateWitnessWithPCMap(trace, null);
        }

        /// Generate witness for all cycles in trace with optional PC mapping.
        ///
        /// When pc_map is provided, PC values are converted from ELF addresses
        /// to bytecode array indices (matching Jolt's convention where PC = bytecode index).
        pub fn generateWitnessWithPCMap(
            self: *Self,
            trace: *const tracer.ExecutionTrace,
            pc_map: ?*const @import("../preprocessing.zig").BytecodePCMapper,
        ) ![]R1CSCycleInputs(F) {
            const num_cycles = trace.steps.items.len;
            if (num_cycles == 0) {
                return &[_]R1CSCycleInputs(F){};
            }

            const witnesses = try self.allocator.alloc(R1CSCycleInputs(F), num_cycles);

            for (0..num_cycles) |i| {
                const step = trace.steps.items[i];

                if (step.is_termination_jal) {
                    // Termination JAL-to-self: normal JAL witness via fromTraceStepWithPCMap.
                    // Jump=1 disables constraint 16 (condition=1-0-1=0).
                    // ShouldJump = Jump*(1-NextIsNoop) = 0 disables constraint 14.
                    // No special overrides needed — the normal witness path handles it.
                    const next_step = trace.steps.items[i + 1];
                    witnesses[i] = R1CSCycleInputs(F).fromTraceStepWithPCMap(step, next_step, pc_map);
                } else if (step.is_termination_store) {
                    // Termination Store: uses createTerminationStoreWitness for flag overrides.
                    // MUST be checked before is_noop because termination_store has is_noop=true
                    // (for the PREVIOUS step's NextIsNoop check) but uses Store witness.
                    const next_step = trace.steps.items[i + 1];
                    witnesses[i] = R1CSCycleInputs(F).createTerminationStoreWitness(step, next_step, pc_map);
                } else if (step.is_noop) {
                    // NoOp padding cycle: all zeros with IsNoop=1
                    witnesses[i] = R1CSCycleInputs(F).createNoopWitness();
                } else {
                    // Real cycle: next step always exists after padding
                    const next_step = trace.steps.items[i + 1];
                    witnesses[i] = R1CSCycleInputs(F).fromTraceStepWithPCMap(step, next_step, pc_map);
                }
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
    const field = @import("zolt_arith").field;
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
    const field = @import("zolt_arith").field;
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
    const field = @import("zolt_arith").field;
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
    const field = @import("zolt_arith").field;
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
