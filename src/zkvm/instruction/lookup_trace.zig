//! Lookup Trace Collector
//!
//! This module provides a lookup trace collector that connects the RISC-V
//! execution tracer to the Shout lookup argument infrastructure.
//!
//! During execution, each instruction generates one or more lookup queries
//! that are recorded in the lookup trace. This trace is then used by the
//! Shout prover to generate the lookup argument proof.
//!
//! Reference: jolt-core/src/zkvm/instruction_lookups/

const std = @import("std");
const Allocator = std.mem.Allocator;

const mod = @import("mod.zig");
const lookups = @import("lookups.zig");
const lookup_table = @import("../lookup_table/mod.zig");

const CircuitFlags = mod.CircuitFlags;
const CircuitFlagSet = mod.CircuitFlagSet;
const InstructionFlags = mod.InstructionFlags;
const InstructionFlagSet = mod.InstructionFlagSet;
const LookupTables = mod.LookupTables;
const DecodedInstruction = mod.DecodedInstruction;
const Opcode = mod.Opcode;
const OpFunct3 = mod.OpFunct3;
const OpImmFunct3 = mod.OpImmFunct3;
const BranchFunct3 = mod.BranchFunct3;

/// A single lookup entry in the trace
pub fn LookupEntry(comptime XLEN: comptime_int) type {
    return struct {
        const Self = @This();

        /// CPU cycle at which this lookup occurred
        cycle: usize,
        /// Program counter at this cycle
        pc: u64,
        /// The lookup table being queried
        table: LookupTables(XLEN),
        /// The lookup index (interleaved operands for binary ops)
        index: u128,
        /// The lookup result/output
        result: u64,
        /// Left operand value
        left_operand: u64,
        /// Right operand value
        right_operand: u64,
        /// Circuit flags for constraint generation
        circuit_flags: CircuitFlagSet,
        /// Instruction metadata flags
        instruction_flags: InstructionFlagSet,
        /// Raw RISC-V instruction
        instruction: u32,

        /// Generic factory for two-operand (rs1, rs2) lookups.
        /// Replaces individual fromAdd, fromSub, fromAnd, etc.
        pub fn fromBinaryLookup(comptime LookupType: type, cycle: usize, pc: u64, instruction: u32, rs1: u64, rs2: u64) Self {
            const inst = LookupType.init(rs1, rs2);
            return Self{
                .cycle = cycle,
                .pc = pc,
                .table = LookupType.lookupTable(),
                .index = inst.toLookupIndex(),
                .result = inst.computeResult(),
                .left_operand = rs1,
                .right_operand = rs2,
                .circuit_flags = LookupType.circuitFlags(),
                .instruction_flags = LookupType.instructionFlags(),
                .instruction = instruction,
            };
        }

        /// Create entry for an ADD instruction within a virtual sequence
        /// Like fromAdd but with VirtualInstruction and DoNotUpdateUnexpandedPC flags
        pub fn fromAddVirtual(
            cycle: usize,
            pc: u64,
            instruction: u32,
            rs1: u64,
            rs2: u64,
            do_not_update_pc: bool,
            is_first_in_sequence: bool,
            is_compressed: bool,
        ) Self {
            const AddLookup = lookups.AddLookup(XLEN);
            const add = AddLookup.init(rs1, rs2);
            var circuit_flags = AddLookup.circuitFlags();
            circuit_flags.set(.VirtualInstruction);
            if (do_not_update_pc) circuit_flags.set(.DoNotUpdateUnexpandedPC);
            if (is_first_in_sequence) circuit_flags.set(.IsFirstInSequence);
            if (is_compressed) circuit_flags.set(.IsCompressed);
            return Self{
                .cycle = cycle,
                .pc = pc,
                .table = AddLookup.lookupTable(),
                .index = add.toLookupIndex(),
                .result = add.computeResult(),
                .left_operand = rs1,
                .right_operand = rs2,
                .circuit_flags = circuit_flags,
                .instruction_flags = AddLookup.instructionFlags(),
                .instruction = instruction,
            };
        }

        /// Create entry for LUI (load upper immediate)
        pub fn fromLui(cycle: usize, pc: u64, instruction: u32, imm: i32) Self {
            const LuiLookup = lookups.LuiLookup(XLEN);
            const lui = LuiLookup.init(imm);
            return Self{
                .cycle = cycle,
                .pc = pc,
                .table = LuiLookup.lookupTable(),
                .index = lui.toLookupIndex(),
                .result = lui.computeResult(),
                .left_operand = 0, // No rs1 for LUI
                .right_operand = @as(u64, @bitCast(@as(i64, imm))),
                .circuit_flags = LuiLookup.circuitFlags(),
                .instruction_flags = LuiLookup.instructionFlags(),
                .instruction = instruction,
            };
        }

        /// Create entry for AUIPC (add upper immediate to PC)
        pub fn fromAuipc(cycle: usize, pc: u64, instruction: u32, imm: i32) Self {
            const AuipcLookup = lookups.AuipcLookup(XLEN);
            const auipc = AuipcLookup.init(pc, imm);
            return Self{
                .cycle = cycle,
                .pc = pc,
                .table = AuipcLookup.lookupTable(),
                .index = auipc.toLookupIndex(),
                .result = auipc.computeResult(),
                .left_operand = pc,
                .right_operand = @as(u64, @bitCast(@as(i64, imm))),
                .circuit_flags = AuipcLookup.circuitFlags(),
                .instruction_flags = AuipcLookup.instructionFlags(),
                .instruction = instruction,
            };
        }

        /// Create entry for JAL (jump and link)
        pub fn fromJal(cycle: usize, pc: u64, instruction: u32, imm: i32, is_compressed: bool) Self {
            const JalLookup = lookups.JalLookup(XLEN);
            const jal = JalLookup.init(pc, imm, is_compressed);
            return Self{
                .cycle = cycle,
                .pc = pc,
                .table = JalLookup.lookupTable(),
                .index = jal.toLookupIndex(),
                .result = jal.computeResult(),
                .left_operand = pc,
                .right_operand = @as(u64, @bitCast(@as(i64, imm))),
                .circuit_flags = JalLookup.circuitFlags(),
                .instruction_flags = JalLookup.instructionFlags(),
                .instruction = instruction,
            };
        }

        /// Create entry for JALR (jump and link register)
        pub fn fromJalr(cycle: usize, pc: u64, instruction: u32, rs1: u64, imm: i32, is_compressed: bool) Self {
            const JalrLookup = lookups.JalrLookup(XLEN);
            const jalr = JalrLookup.init(pc, rs1, imm, is_compressed);
            return Self{
                .cycle = cycle,
                .pc = pc,
                .table = JalrLookup.lookupTable(),
                .index = jalr.toLookupIndex(),
                .result = jalr.computeResult(),
                .left_operand = rs1,
                .right_operand = @as(u64, @bitCast(@as(i64, imm))),
                .circuit_flags = JalrLookup.circuitFlags(),
                .instruction_flags = JalrLookup.instructionFlags(),
                .instruction = instruction,
            };
        }

        /// Create entry for a MUL instruction within a virtual sequence
        /// Like fromMul but with VirtualInstruction and DoNotUpdateUnexpandedPC flags
        pub fn fromMulVirtual(
            cycle: usize,
            pc: u64,
            instruction: u32,
            rs1: u64,
            rs2: u64,
            do_not_update_pc: bool,
            is_first_in_sequence: bool,
            is_compressed: bool,
        ) Self {
            const MulLookup = lookups.MulLookup(XLEN);
            const mul = MulLookup.init(rs1, rs2);
            var circuit_flags = MulLookup.circuitFlags();
            circuit_flags.set(.VirtualInstruction);
            if (do_not_update_pc) circuit_flags.set(.DoNotUpdateUnexpandedPC);
            if (is_first_in_sequence) circuit_flags.set(.IsFirstInSequence);
            if (is_compressed) circuit_flags.set(.IsCompressed);
            return Self{
                .cycle = cycle,
                .pc = pc,
                .table = MulLookup.lookupTable(),
                .index = mul.toLookupIndex(),
                .result = mul.computeResult(),
                .left_operand = rs1,
                .right_operand = rs2,
                .circuit_flags = circuit_flags,
                .instruction_flags = MulLookup.instructionFlags(),
                .instruction = instruction,
            };
        }

        // ========================================================================
        // Virtual Instructions
        // ========================================================================

        /// Create entry for VirtualSignExtendWord
        /// Sign-extends lower XLEN/2 bits to full XLEN bits.
        /// Used as the second step of W-extension instruction decomposition.
        ///
        /// In Jolt:
        /// - Lookup table: SignExtendHalfWord (table index 21)
        /// - Operands: (0, rs1_val) where rs1_val is the full 64-bit result from the base instruction
        /// - Result: sign-extended lower 32 bits
        /// - Circuit flags: WriteLookupOutputToRD, AddOperands, VirtualInstruction (when vsr>0),
        ///   DoNotUpdateUnexpandedPC (when vsr!=0), IsFirstInSequence, IsCompressed
        /// - Instruction flags: LeftOperandIsRs1Value, IsRdNotZero
        ///
        /// Reference: jolt-core/src/zkvm/instruction/virtual_sign_extend_word.rs
        pub fn fromVirtualSignExtendWord(
            cycle: usize,
            pc: u64,
            instruction: u32,
            rs1_val: u64,
            is_virtual: bool,
            do_not_update_pc: bool,
            is_first_in_sequence: bool,
            is_compressed: bool,
            is_rd_not_zero: bool,
        ) Self {
            const VsewLookup = lookups.VirtualSignExtendWordLookup(XLEN);
            const vsew = VsewLookup.init(rs1_val, is_virtual, do_not_update_pc, is_first_in_sequence, is_compressed, is_rd_not_zero);
            return Self{
                .cycle = cycle,
                .pc = pc,
                .table = VsewLookup.lookupTable(),
                .index = vsew.toLookupIndex(),
                .result = vsew.computeResult(),
                .left_operand = rs1_val,
                .right_operand = 0,
                .circuit_flags = vsew.circuitFlags(),
                .instruction_flags = vsew.instructionFlags(),
                .instruction = instruction,
            };
        }

        /// Create lookup entry for VirtualSRLI instruction.
        ///
        /// VirtualSRLI uses VirtualSRL table (table index 26) with interleaved operands.
        /// Reference: jolt-core/src/zkvm/instruction/virtual_srli.rs
        pub fn fromVirtualSRLI(
            cycle: usize,
            pc: u64,
            instruction: u32,
            rs1_val: u64,
            bitmask: u64,
            is_virtual: bool,
            do_not_update_pc: bool,
            is_first_in_sequence: bool,
            is_compressed: bool,
            is_rd_not_zero: bool,
        ) Self {
            const VsrliLookup = lookups.VirtualSRLILookup(XLEN);
            const vsrli = VsrliLookup.init(rs1_val, bitmask, is_virtual, do_not_update_pc, is_first_in_sequence, is_compressed, is_rd_not_zero);
            return Self{
                .cycle = cycle,
                .pc = pc,
                .table = VsrliLookup.lookupTable(),
                .index = vsrli.toLookupIndex(),
                .result = vsrli.computeResult(),
                .left_operand = rs1_val,
                .right_operand = bitmask,
                .circuit_flags = vsrli.circuitFlags(),
                .instruction_flags = vsrli.instructionFlags(),
                .instruction = instruction,
            };
        }

        /// Create entry for VirtualAdvice
        /// Injects an oracle-provided value into the register file.
        /// Used as the first steps of division/remainder inline sequences.
        ///
        /// In Jolt:
        /// - Lookup table: RangeCheck (identity)
        /// - Operands: (0, advice)
        /// - Circuit flags: Advice, WriteLookupOutputToRD, VirtualInstruction, etc.
        /// - Instruction flags: IsRdNotZero
        pub fn fromVirtualAdvice(
            cycle: usize,
            pc: u64,
            instruction: u32,
            advice: u64,
            is_virtual: bool,
            do_not_update_pc: bool,
            is_first_in_sequence: bool,
            is_compressed: bool,
            is_rd_not_zero: bool,
        ) Self {
            const VadvLookup = lookups.VirtualAdviceLookup(XLEN);
            const vadv = VadvLookup.init(advice, is_virtual, do_not_update_pc, is_first_in_sequence, is_compressed, is_rd_not_zero);
            return Self{
                .cycle = cycle,
                .pc = pc,
                .table = VadvLookup.lookupTable(),
                .index = vadv.toLookupIndex(),
                .result = vadv.computeResult(),
                .left_operand = 0, // No input operand for advice
                .right_operand = advice,
                .circuit_flags = vadv.circuitFlags(),
                .instruction_flags = vadv.instructionFlags(),
                .instruction = instruction,
            };
        }

        /// Create entry for VirtualAssertEQ
        /// Asserts that two register values are equal.
        ///
        /// In Jolt:
        /// - Lookup table: Equal (table index 6)
        /// - Operands: (rs1, rs2) interleaved
        /// - Circuit flags: Assert, VirtualInstruction, etc.
        /// - Instruction flags: LeftOperandIsRs1Value, RightOperandIsRs2Value
        pub fn fromVirtualAssertEQ(
            cycle: usize,
            pc: u64,
            instruction: u32,
            rs1_val: u64,
            rs2_val: u64,
            is_virtual: bool,
            do_not_update_pc: bool,
            is_first_in_sequence: bool,
            is_compressed: bool,
        ) Self {
            const VaeqLookup = lookups.VirtualAssertEQLookup(XLEN);
            const vaeq = VaeqLookup.init(rs1_val, rs2_val, is_virtual, do_not_update_pc, is_first_in_sequence, is_compressed);
            return Self{
                .cycle = cycle,
                .pc = pc,
                .table = VaeqLookup.lookupTable(),
                .index = vaeq.toLookupIndex(),
                .result = vaeq.computeResult(),
                .left_operand = rs1_val,
                .right_operand = rs2_val,
                .circuit_flags = vaeq.circuitFlags(),
                .instruction_flags = vaeq.instructionFlags(),
                .instruction = instruction,
            };
        }

        /// Create entry for VirtualZeroExtendWord
        /// Zero-extends lower XLEN/2 bits.
        ///
        /// In Jolt:
        /// - Lookup table: LowerHalfWord (table index 20)
        /// - Operands: (0, rs1) via AddOperands mode
        /// - Circuit flags: WriteLookupOutputToRD, AddOperands, VirtualInstruction, etc.
        /// - Instruction flags: LeftOperandIsRs1Value, IsRdNotZero
        pub fn fromVirtualZeroExtendWord(
            cycle: usize,
            pc: u64,
            instruction: u32,
            rs1_val: u64,
            is_virtual: bool,
            do_not_update_pc: bool,
            is_first_in_sequence: bool,
            is_compressed: bool,
            is_rd_not_zero: bool,
        ) Self {
            const VzewLookup = lookups.VirtualZeroExtendWordLookup(XLEN);
            const vzew = VzewLookup.init(rs1_val, is_virtual, do_not_update_pc, is_first_in_sequence, is_compressed, is_rd_not_zero);
            return Self{
                .cycle = cycle,
                .pc = pc,
                .table = VzewLookup.lookupTable(),
                .index = vzew.toLookupIndex(),
                .result = vzew.computeResult(),
                .left_operand = rs1_val,
                .right_operand = 0,
                .circuit_flags = vzew.circuitFlags(),
                .instruction_flags = vzew.instructionFlags(),
                .instruction = instruction,
            };
        }

        /// Create entry for VirtualAssertValidUnsignedRemainder
        /// Asserts that remainder < divisor (or divisor == 0).
        ///
        /// In Jolt:
        /// - Lookup table: ValidUnsignedRemainder (table index 16)
        /// - Operands: (rs1, rs2) interleaved
        /// - Circuit flags: Assert, VirtualInstruction, etc.
        /// - Instruction flags: LeftOperandIsRs1Value, RightOperandIsRs2Value
        pub fn fromVirtualAssertValidUnsignedRemainder(
            cycle: usize,
            pc: u64,
            instruction: u32,
            rs1_val: u64,
            rs2_val: u64,
            is_virtual: bool,
            do_not_update_pc: bool,
            is_first_in_sequence: bool,
            is_compressed: bool,
        ) Self {
            const VavurLookup = lookups.VirtualAssertValidUnsignedRemainderLookup(XLEN);
            const vavur = VavurLookup.init(rs1_val, rs2_val, is_virtual, do_not_update_pc, is_first_in_sequence, is_compressed);
            return Self{
                .cycle = cycle,
                .pc = pc,
                .table = VavurLookup.lookupTable(),
                .index = vavur.toLookupIndex(),
                .result = vavur.computeResult(),
                .left_operand = rs1_val,
                .right_operand = rs2_val,
                .circuit_flags = vavur.circuitFlags(),
                .instruction_flags = vavur.instructionFlags(),
                .instruction = instruction,
            };
        }

        /// Create lookup entry for VirtualMULI instruction.
        ///
        /// VirtualMULI is used for SLLI decomposition (SLLI → VirtualMULI) and as
        /// the base step of SLLIW decomposition (SLLIW → VirtualMULI + VirtualSignExtendWord).
        ///
        /// Key properties:
        /// - Lookup table: RangeCheck (identity)
        /// - Lookup operands: (0, rs1 * imm)
        /// - Circuit flags: MultiplyOperands, WriteLookupOutputToRD, VirtualInstruction (when vsr.is_some()),
        ///   DoNotUpdateUnexpandedPC (when vsr!=0), IsFirstInSequence, IsCompressed
        /// - Instruction flags: LeftOperandIsRs1Value, RightOperandIsImm, IsRdNotZero
        ///
        /// Reference: jolt-core/src/zkvm/instruction/virtual_muli.rs
        pub fn fromVirtualMULI(
            cycle: usize,
            pc: u64,
            instruction: u32,
            rs1_val: u64,
            imm_val: u64,
            is_virtual: bool,
            do_not_update_pc: bool,
            is_first_in_sequence: bool,
            is_compressed: bool,
            is_rd_not_zero: bool,
        ) Self {
            const VmuliLookup = lookups.VirtualMULILookup(XLEN);
            const vmuli = VmuliLookup.init(rs1_val, imm_val, is_virtual, do_not_update_pc, is_first_in_sequence, is_compressed, is_rd_not_zero);
            return Self{
                .cycle = cycle,
                .pc = pc,
                .table = VmuliLookup.lookupTable(),
                .index = vmuli.toLookupIndex(),
                .result = vmuli.computeResult(),
                .left_operand = rs1_val,
                .right_operand = imm_val,
                .circuit_flags = vmuli.circuitFlags(),
                .instruction_flags = vmuli.instructionFlags(),
                .instruction = instruction,
            };
        }
    };
}

/// Lookup trace collector that records all lookup operations during execution
pub fn LookupTraceCollector(comptime XLEN: comptime_int) type {
    return struct {
        const Self = @This();
        const Entry = LookupEntry(XLEN);

        /// All lookup entries
        entries: std.ArrayListUnmanaged(Entry),
        /// Allocator for dynamic memory
        allocator: Allocator,
        /// Whether to collect lookups (can be disabled for faster emulation)
        enabled: bool,

        pub fn init(allocator: Allocator) Self {
            return Self{
                .entries = .empty,
                .allocator = allocator,
                .enabled = true,
            };
        }

        pub fn deinit(self: *Self) void {
            self.entries.deinit(self.allocator);
        }

        /// Clear all entries (for reuse)
        pub fn clear(self: *Self) void {
            self.entries.clearRetainingCapacity();
        }

        /// Enable or disable lookup collection
        pub fn setEnabled(self: *Self, enabled: bool) void {
            self.enabled = enabled;
        }

        /// Record a lookup entry
        pub fn record(self: *Self, entry: Entry) !void {
            if (!self.enabled) return;
            try self.entries.append(self.allocator, entry);
        }

        /// Record lookup for an instruction based on decoded opcode and function
        /// This is the main entry point called by the emulator
        pub fn recordInstruction(
            self: *Self,
            cycle: usize,
            pc: u64,
            instruction: u32,
            decoded: DecodedInstruction,
            rs1_val: u64,
            rs2_val: u64,
        ) !void {
            if (!self.enabled) return;

            switch (decoded.opcode) {
                .OP => {
                    // Check for M extension first
                    if (decoded.funct7 == 0b0000001) {
                        // M extension: MUL, MULH, MULHSU, MULHU, DIV, DIVU, REM, REMU
                        const entry: Entry = switch (decoded.funct3) {
                            0b000 => Entry.fromBinaryLookup(lookups.MulLookup(XLEN), cycle, pc, instruction, rs1_val, rs2_val), // MUL
                            0b001 => Entry.fromBinaryLookup(lookups.MulhLookup(XLEN), cycle, pc, instruction, rs1_val, rs2_val), // MULH
                            0b010 => Entry.fromBinaryLookup(lookups.MulhsuLookup(XLEN), cycle, pc, instruction, rs1_val, rs2_val), // MULHSU
                            0b011 => Entry.fromBinaryLookup(lookups.MulhuLookup(XLEN), cycle, pc, instruction, rs1_val, rs2_val), // MULHU
                            0b100 => Entry.fromBinaryLookup(lookups.DivLookup(XLEN), cycle, pc, instruction, rs1_val, rs2_val), // DIV
                            0b101 => Entry.fromBinaryLookup(lookups.DivuLookup(XLEN), cycle, pc, instruction, rs1_val, rs2_val), // DIVU
                            0b110 => Entry.fromBinaryLookup(lookups.RemLookup(XLEN), cycle, pc, instruction, rs1_val, rs2_val), // REM
                            0b111 => Entry.fromBinaryLookup(lookups.RemuLookup(XLEN), cycle, pc, instruction, rs1_val, rs2_val), // REMU
                        };
                        try self.entries.append(self.allocator, entry);
                        return;
                    }

                    // Standard ALU operations
                    const funct3 = @as(OpFunct3, @enumFromInt(decoded.funct3));
                    const entry: ?Entry = switch (funct3) {
                        .ADD_SUB => blk: {
                            if ((decoded.funct7 & 0x20) != 0) {
                                // SUB
                                break :blk Entry.fromBinaryLookup(lookups.SubLookup(XLEN), cycle, pc, instruction, rs1_val, rs2_val);
                            } else {
                                // ADD
                                break :blk Entry.fromBinaryLookup(lookups.AddLookup(XLEN), cycle, pc, instruction, rs1_val, rs2_val);
                            }
                        },
                        .AND => Entry.fromBinaryLookup(lookups.AndLookup(XLEN), cycle, pc, instruction, rs1_val, rs2_val),
                        .OR => Entry.fromBinaryLookup(lookups.OrLookup(XLEN), cycle, pc, instruction, rs1_val, rs2_val),
                        .XOR => Entry.fromBinaryLookup(lookups.XorLookup(XLEN), cycle, pc, instruction, rs1_val, rs2_val),
                        .SLT => Entry.fromBinaryLookup(lookups.SltLookup(XLEN), cycle, pc, instruction, rs1_val, rs2_val),
                        .SLTU => Entry.fromBinaryLookup(lookups.SltuLookup(XLEN), cycle, pc, instruction, rs1_val, rs2_val),
                        .SLL => Entry.fromBinaryLookup(lookups.SllLookup(XLEN), cycle, pc, instruction, rs1_val, rs2_val),
                        .SRL_SRA => blk: {
                            if ((decoded.funct7 & 0x20) != 0) {
                                // SRA (arithmetic)
                                break :blk Entry.fromBinaryLookup(lookups.SraLookup(XLEN), cycle, pc, instruction, rs1_val, rs2_val);
                            } else {
                                // SRL (logical)
                                break :blk Entry.fromBinaryLookup(lookups.SrlLookup(XLEN), cycle, pc, instruction, rs1_val, rs2_val);
                            }
                        },
                    };
                    if (entry) |e| {
                        try self.entries.append(self.allocator, e);
                    }
                },
                .OP_IMM => {
                    // Immediate ALU operations use immediate as second operand
                    const imm_val: u64 = @bitCast(@as(i64, decoded.imm));
                    const funct3 = @as(OpImmFunct3, @enumFromInt(decoded.funct3));
                    const entry: ?Entry = switch (funct3) {
                        .ADDI => Entry.fromBinaryLookup(lookups.AddLookup(XLEN), cycle, pc, instruction, rs1_val, imm_val),
                        .ANDI => Entry.fromBinaryLookup(lookups.AndLookup(XLEN), cycle, pc, instruction, rs1_val, imm_val),
                        .ORI => Entry.fromBinaryLookup(lookups.OrLookup(XLEN), cycle, pc, instruction, rs1_val, imm_val),
                        .XORI => Entry.fromBinaryLookup(lookups.XorLookup(XLEN), cycle, pc, instruction, rs1_val, imm_val),
                        .SLTI => Entry.fromBinaryLookup(lookups.SltLookup(XLEN), cycle, pc, instruction, rs1_val, imm_val),
                        .SLTIU => Entry.fromBinaryLookup(lookups.SltuLookup(XLEN), cycle, pc, instruction, rs1_val, imm_val),
                        .SLLI => blk: {
                            // Shift amount is in the lower bits of imm
                            const imm_u32: u32 = @bitCast(@as(i32, @truncate(decoded.imm)));
                            const shamt: u64 = @as(u64, imm_u32 & 0x3F);
                            break :blk Entry.fromBinaryLookup(lookups.SlliLookup(XLEN), cycle, pc, instruction, rs1_val, shamt);
                        },
                        .SRLI_SRAI => blk: {
                            // Shift amount is in the lower bits of imm
                            const imm_u32: u32 = @bitCast(@as(i32, @truncate(decoded.imm)));
                            const shamt: u64 = @as(u64, imm_u32 & 0x3F);
                            if ((decoded.funct7 & 0x20) != 0) {
                                // SRAI (arithmetic)
                                break :blk Entry.fromBinaryLookup(lookups.SraiLookup(XLEN), cycle, pc, instruction, rs1_val, shamt);
                            } else {
                                // SRLI (logical)
                                break :blk Entry.fromBinaryLookup(lookups.SrliLookup(XLEN), cycle, pc, instruction, rs1_val, shamt);
                            }
                        },
                    };
                    if (entry) |e| {
                        try self.entries.append(self.allocator, e);
                    }
                },
                .BRANCH => {
                    // Branch operations - use dedicated branch lookups
                    const funct3 = @as(BranchFunct3, @enumFromInt(decoded.funct3));
                    const entry: ?Entry = switch (funct3) {
                        .BEQ => Entry.fromBinaryLookup(lookups.BeqLookup(XLEN), cycle, pc, instruction, rs1_val, rs2_val),
                        .BNE => Entry.fromBinaryLookup(lookups.BneLookup(XLEN), cycle, pc, instruction, rs1_val, rs2_val),
                        .BLT => Entry.fromBinaryLookup(lookups.BltLookup(XLEN), cycle, pc, instruction, rs1_val, rs2_val),
                        .BGE => Entry.fromBinaryLookup(lookups.BgeLookup(XLEN), cycle, pc, instruction, rs1_val, rs2_val),
                        .BLTU => Entry.fromBinaryLookup(lookups.BltuLookup(XLEN), cycle, pc, instruction, rs1_val, rs2_val),
                        .BGEU => Entry.fromBinaryLookup(lookups.BgeuLookup(XLEN), cycle, pc, instruction, rs1_val, rs2_val),
                        _ => null,
                    };
                    if (entry) |e| {
                        try self.entries.append(self.allocator, e);
                    }
                },
                .OP_32 => {
                    // 32-bit integer register-register operations (RV64 only)
                    // Check for M extension first
                    if (decoded.funct7 == 0b0000001) {
                        // RV64M word operations: MULW, DIVW, DIVUW, REMW, REMUW
                        const entry: Entry = switch (decoded.funct3) {
                            0b000 => Entry.fromBinaryLookup(lookups.MulwLookup(XLEN), cycle, pc, instruction, rs1_val, rs2_val), // MULW
                            0b100 => Entry.fromBinaryLookup(lookups.DivwLookup(XLEN), cycle, pc, instruction, rs1_val, rs2_val), // DIVW
                            0b101 => Entry.fromBinaryLookup(lookups.DivuwLookup(XLEN), cycle, pc, instruction, rs1_val, rs2_val), // DIVUW
                            0b110 => Entry.fromBinaryLookup(lookups.RemwLookup(XLEN), cycle, pc, instruction, rs1_val, rs2_val), // REMW
                            0b111 => Entry.fromBinaryLookup(lookups.RemuwLookup(XLEN), cycle, pc, instruction, rs1_val, rs2_val), // REMUW
                            else => Entry.fromBinaryLookup(lookups.AddwLookup(XLEN), cycle, pc, instruction, rs1_val, rs2_val), // fallback
                        };
                        try self.entries.append(self.allocator, entry);
                        return;
                    }

                    // Standard RV64I word operations: ADDW, SUBW, SLLW, SRLW, SRAW
                    const entry: Entry = switch (decoded.funct3) {
                        0b000 => blk: {
                            if ((decoded.funct7 & 0x20) != 0) {
                                break :blk Entry.fromBinaryLookup(lookups.SubwLookup(XLEN), cycle, pc, instruction, rs1_val, rs2_val); // SUBW
                            } else {
                                break :blk Entry.fromBinaryLookup(lookups.AddwLookup(XLEN), cycle, pc, instruction, rs1_val, rs2_val); // ADDW
                            }
                        },
                        0b001 => Entry.fromBinaryLookup(lookups.SllwLookup(XLEN), cycle, pc, instruction, rs1_val, rs2_val), // SLLW
                        0b101 => blk: {
                            if ((decoded.funct7 & 0x20) != 0) {
                                break :blk Entry.fromBinaryLookup(lookups.SrawLookup(XLEN), cycle, pc, instruction, rs1_val, rs2_val); // SRAW
                            } else {
                                break :blk Entry.fromBinaryLookup(lookups.SrlwLookup(XLEN), cycle, pc, instruction, rs1_val, rs2_val); // SRLW
                            }
                        },
                        else => Entry.fromBinaryLookup(lookups.AddwLookup(XLEN), cycle, pc, instruction, rs1_val, rs2_val), // fallback
                    };
                    try self.entries.append(self.allocator, entry);
                },
                .LUI => {
                    // Load upper immediate
                    const entry = Entry.fromLui(cycle, pc, instruction, decoded.imm);
                    try self.entries.append(self.allocator, entry);
                },
                .AUIPC => {
                    // Add upper immediate to PC
                    const entry = Entry.fromAuipc(cycle, pc, instruction, decoded.imm);
                    try self.entries.append(self.allocator, entry);
                },
                .JAL => {
                    // Jump and link
                    const is_compressed = false; // Standard instructions are not compressed
                    const entry = Entry.fromJal(cycle, pc, instruction, decoded.imm, is_compressed);
                    try self.entries.append(self.allocator, entry);
                },
                .JALR => {
                    // Jump and link register
                    const is_compressed = false;
                    const entry = Entry.fromJalr(cycle, pc, instruction, rs1_val, decoded.imm, is_compressed);
                    try self.entries.append(self.allocator, entry);
                },
                else => {
                    // LOAD, STORE - memory operations handled separately
                },
            }
        }

        /// Record lookup for a VirtualSignExtendWord instruction
        /// This is the second step in W-extension decomposition (ADDIW, ADDW, SUBW, MULW)
        pub fn recordVirtualSignExtendWord(
            self: *Self,
            cycle: usize,
            pc: u64,
            instruction: u32,
            rs1_val: u64,
            sign_extended_result: u64,
        ) !void {
            if (!self.enabled) return;

            // Determine rd from the synthetic instruction encoding
            const rd: u8 = @truncate((instruction >> 7) & 0x1f);

            const entry = Entry.fromVirtualSignExtendWord(
                cycle,
                pc,
                instruction,
                rs1_val,
                true, // is_virtual: VirtualSignExtendWord is always virtual when vsr > 0
                false, // do_not_update_pc: vsr=0 for VirtualSignExtendWord (last in sequence)
                false, // is_first_in_sequence: false for VirtualSignExtendWord
                false, // is_compressed: inherited, but VirtualSignExtendWord is synthetic
                rd != 0, // is_rd_not_zero
            );
            _ = sign_extended_result; // Result is computed by the lookup
            try self.entries.append(self.allocator, entry);
        }

        /// Record lookup for a VirtualSRLI instruction
        /// Used for SRLI decomposition (standalone) and SRLIW middle step (virtual sequence)
        pub fn recordVirtualSRLI(
            self: *Self,
            cycle: usize,
            pc: u64,
            instruction: u32,
            rs1_val: u64,
            bitmask: u64,
            result_val: u64,
            is_virtual: bool,
            do_not_update_pc: bool,
            is_first_in_sequence: bool,
            is_compressed: bool,
        ) !void {
            if (!self.enabled) return;

            const rd: u8 = @truncate((instruction >> 7) & 0x1f);

            const entry = Entry.fromVirtualSRLI(
                cycle,
                pc,
                instruction,
                rs1_val,
                bitmask,
                is_virtual,
                do_not_update_pc,
                is_first_in_sequence,
                is_compressed,
                rd != 0,
            );
            _ = result_val; // Result is computed by the lookup
            try self.entries.append(self.allocator, entry);
        }

        /// Record lookup for a VirtualMULI instruction
        /// Used for SLLI decomposition (standalone) and SLLIW base step (virtual sequence)
        pub fn recordVirtualMULI(
            self: *Self,
            cycle: usize,
            pc: u64,
            instruction: u32,
            rs1_val: u64,
            imm_val: u64,
            is_virtual: bool,
            do_not_update_pc: bool,
            is_first_in_sequence: bool,
            is_compressed: bool,
        ) !void {
            if (!self.enabled) return;

            const rd: u8 = @truncate((instruction >> 7) & 0x1f);

            const entry = Entry.fromVirtualMULI(
                cycle,
                pc,
                instruction,
                rs1_val,
                imm_val,
                is_virtual,
                do_not_update_pc,
                is_first_in_sequence,
                is_compressed,
                rd != 0, // is_rd_not_zero
            );
            try self.entries.append(self.allocator, entry);
        }

        /// Record lookup for a MUL instruction within a virtual sequence
        pub fn recordMulVirtual(
            self: *Self,
            cycle: usize,
            pc: u64,
            instruction: u32,
            rs1_val: u64,
            rs2_val: u64,
            do_not_update_pc: bool,
            is_first_in_sequence: bool,
            is_compressed: bool,
        ) !void {
            if (!self.enabled) return;
            const entry = Entry.fromMulVirtual(
                cycle,
                pc,
                instruction,
                rs1_val,
                rs2_val,
                do_not_update_pc,
                is_first_in_sequence,
                is_compressed,
            );
            try self.entries.append(self.allocator, entry);
        }

        /// Record lookup for an ADD instruction within a virtual sequence
        pub fn recordAddVirtual(
            self: *Self,
            cycle: usize,
            pc: u64,
            instruction: u32,
            rs1_val: u64,
            rs2_val: u64,
            do_not_update_pc: bool,
            is_first_in_sequence: bool,
            is_compressed: bool,
        ) !void {
            if (!self.enabled) return;
            const entry = Entry.fromAddVirtual(
                cycle,
                pc,
                instruction,
                rs1_val,
                rs2_val,
                do_not_update_pc,
                is_first_in_sequence,
                is_compressed,
            );
            try self.entries.append(self.allocator, entry);
        }

        /// Record lookup for a VirtualAdvice instruction
        /// Used for oracle-provided values in division/remainder inline sequences
        pub fn recordVirtualAdvice(
            self: *Self,
            cycle: usize,
            pc: u64,
            instruction: u32,
            advice: u64,
            is_virtual: bool,
            do_not_update_pc: bool,
            is_first_in_sequence: bool,
            is_compressed: bool,
        ) !void {
            if (!self.enabled) return;

            const rd: u8 = @truncate((instruction >> 7) & 0x1f);

            const entry = Entry.fromVirtualAdvice(
                cycle,
                pc,
                instruction,
                advice,
                is_virtual,
                do_not_update_pc,
                is_first_in_sequence,
                is_compressed,
                rd != 0,
            );
            try self.entries.append(self.allocator, entry);
        }

        /// Record lookup for a VirtualAssertEQ instruction
        /// Used to assert equality of two registers in inline sequences
        pub fn recordVirtualAssertEQ(
            self: *Self,
            cycle: usize,
            pc: u64,
            instruction: u32,
            rs1_val: u64,
            rs2_val: u64,
            is_virtual: bool,
            do_not_update_pc: bool,
            is_first_in_sequence: bool,
            is_compressed: bool,
        ) !void {
            if (!self.enabled) return;

            const entry = Entry.fromVirtualAssertEQ(
                cycle,
                pc,
                instruction,
                rs1_val,
                rs2_val,
                is_virtual,
                do_not_update_pc,
                is_first_in_sequence,
                is_compressed,
            );
            try self.entries.append(self.allocator, entry);
        }

        /// Record lookup for a VirtualZeroExtendWord instruction
        /// Used to zero-extend lower 32 bits in inline sequences
        pub fn recordVirtualZeroExtendWord(
            self: *Self,
            cycle: usize,
            pc: u64,
            instruction: u32,
            rs1_val: u64,
            is_virtual: bool,
            do_not_update_pc: bool,
            is_first_in_sequence: bool,
            is_compressed: bool,
        ) !void {
            if (!self.enabled) return;

            const rd: u8 = @truncate((instruction >> 7) & 0x1f);

            const entry = Entry.fromVirtualZeroExtendWord(
                cycle,
                pc,
                instruction,
                rs1_val,
                is_virtual,
                do_not_update_pc,
                is_first_in_sequence,
                is_compressed,
                rd != 0,
            );
            try self.entries.append(self.allocator, entry);
        }

        /// Record lookup for a VirtualAssertValidUnsignedRemainder instruction
        /// Used to validate remainder < divisor in unsigned division inline sequences
        pub fn recordVirtualAssertValidUnsignedRemainder(
            self: *Self,
            cycle: usize,
            pc: u64,
            instruction: u32,
            rs1_val: u64,
            rs2_val: u64,
            is_virtual: bool,
            do_not_update_pc: bool,
            is_first_in_sequence: bool,
            is_compressed: bool,
        ) !void {
            if (!self.enabled) return;

            const entry = Entry.fromVirtualAssertValidUnsignedRemainder(
                cycle,
                pc,
                instruction,
                rs1_val,
                rs2_val,
                is_virtual,
                do_not_update_pc,
                is_first_in_sequence,
                is_compressed,
            );
            try self.entries.append(self.allocator, entry);
        }

        /// Record lookup for a VirtualSRAI instruction within a virtual sequence.
        /// Similar to VirtualSRLI but uses VirtualSRA table (index 27).
        pub fn recordVirtualSRAI(
            self: *Self,
            cycle: usize,
            pc: u64,
            instruction: u32,
            rs1_val: u64,
            is_virtual: bool,
            do_not_update_pc: bool,
            is_first_in_sequence: bool,
            is_compressed: bool,
        ) !void {
            if (!self.enabled) return;
            // VirtualSRAI uses same pattern as VirtualSRLI but with VirtualSRA table
            const rd: u8 = @truncate((instruction >> 7) & 0x1f);
            const shift_raw: u32 = instruction >> 20;
            const shift: u7 = @truncate(shift_raw & 0x3F);
            const ones: u128 = (@as(u128, 1) << @intCast(64 - @as(u8, shift))) - 1;
            const bitmask: u64 = @truncate(ones << shift);
            const index = lookup_table.interleaveBits(rs1_val, bitmask);
            // Compute result: arithmetic right shift
            const result: u64 = @bitCast(@as(i64, @bitCast(rs1_val)) >> @intCast(shift));

            var cf = CircuitFlagSet.init();
            cf.set(.WriteLookupOutputToRD);
            if (is_virtual) cf.set(.VirtualInstruction);
            if (do_not_update_pc) cf.set(.DoNotUpdateUnexpandedPC);
            if (is_first_in_sequence) cf.set(.IsFirstInSequence);
            if (is_compressed) cf.set(.IsCompressed);

            var inf = InstructionFlagSet.init();
            inf.set(.LeftOperandIsRs1Value);
            inf.set(.RightOperandIsImm);
            if (rd != 0) inf.set(.IsRdNotZero);

            const entry = Entry{
                .cycle = cycle,
                .pc = pc,
                .table = .VirtualSRA,
                .index = index,
                .result = result,
                .left_operand = rs1_val,
                .right_operand = bitmask,
                .circuit_flags = cf,
                .instruction_flags = inf,
                .instruction = instruction,
            };
            try self.entries.append(self.allocator, entry);
        }

        /// Record lookup for a VirtualChangeDivisorW instruction within a virtual sequence.
        /// Uses lookup table VirtualChangeDivisorW (index 31).
        pub fn recordVirtualChangeDivisorW(
            self: *Self,
            cycle: usize,
            pc: u64,
            instruction: u32,
            rs1_val: u64,
            rs2_val: u64,
            is_virtual: bool,
            do_not_update_pc: bool,
            is_first_in_sequence: bool,
            is_compressed: bool,
        ) !void {
            if (!self.enabled) return;
            const rd: u8 = @truncate((instruction >> 7) & 0x1f);
            const index = lookup_table.interleaveBits(rs1_val, rs2_val);
            // Result: if dividend == INT32_MIN and divisor == -1, return 1; else return divisor
            const dv_i32: i32 = @truncate(@as(i64, @bitCast(rs1_val)));
            const ds_i32: i32 = @truncate(@as(i64, @bitCast(rs2_val)));
            const result: u64 = if (dv_i32 == std.math.minInt(i32) and ds_i32 == -1)
                1
            else
                @bitCast(@as(i64, ds_i32));

            var cf = CircuitFlagSet.init();
            cf.set(.WriteLookupOutputToRD);
            if (is_virtual) cf.set(.VirtualInstruction);
            if (do_not_update_pc) cf.set(.DoNotUpdateUnexpandedPC);
            if (is_first_in_sequence) cf.set(.IsFirstInSequence);
            if (is_compressed) cf.set(.IsCompressed);

            var inf = InstructionFlagSet.init();
            inf.set(.LeftOperandIsRs1Value);
            inf.set(.RightOperandIsRs2Value);
            if (rd != 0) inf.set(.IsRdNotZero);

            const entry = Entry{
                .cycle = cycle,
                .pc = pc,
                .table = .VirtualChangeDivisorW,
                .index = index,
                .result = result,
                .left_operand = rs1_val,
                .right_operand = rs2_val,
                .circuit_flags = cf,
                .instruction_flags = inf,
                .instruction = instruction,
            };
            try self.entries.append(self.allocator, entry);
        }

        /// Record lookup for an ANDN instruction (rs1 & ~rs2)
        /// Uses Andn lookup table (interleaved)
        /// Record lookup for VirtualRev8W (opcode 0x5B funct3=0): byte-swap each 32-bit half.
        /// Single-operand: index = rs1_val (no interleaving). Lookup table = VirtualRev8W (24).
        pub fn recordVirtualRev8W(
            self: *Self,
            cycle: usize,
            pc: u64,
            instruction: u32,
            rs1_val: u64,
            is_virtual: bool,
            do_not_update_pc: bool,
            is_first_in_sequence: bool,
            is_compressed: bool,
        ) !void {
            if (!self.enabled) return;
            const rd: u8 = @truncate((instruction >> 7) & 0x1f);
            const Rev8WLkp = lookups.VirtualRev8WLookup(XLEN);
            const lkp = Rev8WLkp.init(rs1_val, is_virtual, do_not_update_pc, is_first_in_sequence, is_compressed, rd != 0);
            const entry = Entry{
                .cycle = cycle,
                .pc = pc,
                .table = Rev8WLkp.lookupTable(),
                .index = lkp.toLookupIndex(),
                .result = lkp.computeResult(),
                .left_operand = 0,
                .right_operand = rs1_val,
                .circuit_flags = lkp.circuitFlags(),
                .instruction_flags = lkp.instructionFlags(),
                .instruction = instruction,
            };
            try self.entries.append(self.allocator, entry);
        }

        pub fn recordANDN(
            self: *Self,
            cycle: usize,
            pc: u64,
            instruction: u32,
            rs1_val: u64,
            rs2_val: u64,
            is_virtual: bool,
            do_not_update_pc: bool,
            is_first_in_sequence: bool,
            is_compressed: bool,
        ) !void {
            if (!self.enabled) return;
            const rd: u8 = @truncate((instruction >> 7) & 0x1f);
            const AndnLkp = lookups.AndnLookup(XLEN);
            const lkp = AndnLkp.init(rs1_val, rs2_val, is_virtual, do_not_update_pc, is_first_in_sequence, is_compressed, rd != 0);
            const entry = Entry{
                .cycle = cycle,
                .pc = pc,
                .table = AndnLkp.lookupTable(),
                .index = lkp.toLookupIndex(),
                .result = lkp.computeResult(),
                .left_operand = rs1_val,
                .right_operand = rs2_val,
                .circuit_flags = lkp.circuitFlags(),
                .instruction_flags = lkp.instructionFlags(),
                .instruction = instruction,
            };
            try self.entries.append(self.allocator, entry);
        }

        /// Record lookup for a VirtualROTRI instruction (64-bit rotate right by bitmask)
        /// Uses VirtualROTR lookup table (interleaved)
        pub fn recordVirtualROTRI(
            self: *Self,
            cycle: usize,
            pc: u64,
            instruction: u32,
            rs1_val: u64,
            bitmask: u64,
            is_virtual: bool,
            do_not_update_pc: bool,
            is_first_in_sequence: bool,
            is_compressed: bool,
        ) !void {
            if (!self.enabled) return;
            const rd: u8 = @truncate((instruction >> 7) & 0x1f);
            const RotrLkp = lookups.VirtualROTRILookup(XLEN);
            const lkp = RotrLkp.init(rs1_val, bitmask, is_virtual, do_not_update_pc, is_first_in_sequence, is_compressed, rd != 0);
            const entry = Entry{
                .cycle = cycle,
                .pc = pc,
                .table = RotrLkp.lookupTable(),
                .index = lkp.toLookupIndex(),
                .result = lkp.computeResult(),
                .left_operand = rs1_val,
                .right_operand = bitmask,
                .circuit_flags = lkp.circuitFlags(),
                .instruction_flags = lkp.instructionFlags(),
                .instruction = instruction,
            };
            try self.entries.append(self.allocator, entry);
        }

        /// Record lookup for a VirtualROTRIW instruction (32-bit rotate right by bitmask)
        /// Uses VirtualROTRW lookup table (interleaved)
        pub fn recordVirtualROTRIW(
            self: *Self,
            cycle: usize,
            pc: u64,
            instruction: u32,
            rs1_val: u64,
            bitmask: u64,
            is_virtual: bool,
            do_not_update_pc: bool,
            is_first_in_sequence: bool,
            is_compressed: bool,
        ) !void {
            if (!self.enabled) return;
            const rd: u8 = @truncate((instruction >> 7) & 0x1f);
            const RotrwLkp = lookups.VirtualROTRIWLookup(XLEN);
            const lkp = RotrwLkp.init(rs1_val, bitmask, is_virtual, do_not_update_pc, is_first_in_sequence, is_compressed, rd != 0);
            const entry = Entry{
                .cycle = cycle,
                .pc = pc,
                .table = RotrwLkp.lookupTable(),
                .index = lkp.toLookupIndex(),
                .result = lkp.computeResult(),
                .left_operand = rs1_val,
                .right_operand = bitmask,
                .circuit_flags = lkp.circuitFlags(),
                .instruction_flags = lkp.instructionFlags(),
                .instruction = instruction,
            };
            try self.entries.append(self.allocator, entry);
        }

        /// Record lookup for a SUB instruction within a virtual sequence.
        pub fn recordSubVirtual(
            self: *Self,
            cycle: usize,
            pc: u64,
            instruction: u32,
            rs1_val: u64,
            rs2_val: u64,
            do_not_update_pc: bool,
            is_first_in_sequence: bool,
            is_compressed: bool,
        ) !void {
            if (!self.enabled) return;
            const SubLookup = lookups.SubLookup(XLEN);
            const sub = SubLookup.init(rs1_val, rs2_val);
            var circuit_flags = SubLookup.circuitFlags();
            circuit_flags.set(.VirtualInstruction);
            if (do_not_update_pc) circuit_flags.set(.DoNotUpdateUnexpandedPC);
            if (is_first_in_sequence) circuit_flags.set(.IsFirstInSequence);
            if (is_compressed) circuit_flags.set(.IsCompressed);
            const entry = Entry{
                .cycle = cycle,
                .pc = pc,
                .table = SubLookup.lookupTable(),
                .index = sub.toLookupIndex(),
                .result = sub.computeResult(),
                .left_operand = rs1_val,
                .right_operand = rs2_val,
                .circuit_flags = circuit_flags,
                .instruction_flags = SubLookup.instructionFlags(),
                .instruction = instruction,
            };
            try self.entries.append(self.allocator, entry);
        }

        /// Record lookup for an XOR instruction within a virtual sequence.
        pub fn recordXorVirtual(
            self: *Self,
            cycle: usize,
            pc: u64,
            instruction: u32,
            rs1_val: u64,
            rs2_val: u64,
            do_not_update_pc: bool,
            is_first_in_sequence: bool,
            is_compressed: bool,
        ) !void {
            if (!self.enabled) return;
            const XorLookup = lookups.XorLookup(XLEN);
            const xor_op = XorLookup.init(rs1_val, rs2_val);
            var circuit_flags = XorLookup.circuitFlags();
            circuit_flags.set(.VirtualInstruction);
            if (do_not_update_pc) circuit_flags.set(.DoNotUpdateUnexpandedPC);
            if (is_first_in_sequence) circuit_flags.set(.IsFirstInSequence);
            if (is_compressed) circuit_flags.set(.IsCompressed);
            const entry = Entry{
                .cycle = cycle,
                .pc = pc,
                .table = XorLookup.lookupTable(),
                .index = xor_op.toLookupIndex(),
                .result = xor_op.computeResult(),
                .left_operand = rs1_val,
                .right_operand = rs2_val,
                .circuit_flags = circuit_flags,
                .instruction_flags = XorLookup.instructionFlags(),
                .instruction = instruction,
            };
            try self.entries.append(self.allocator, entry);
        }

        /// Record lookup for a VirtualPow2 instruction within a virtual sequence.
        /// VirtualPow2 computes 1 << (rs1_val % 64). Stub — full implementation pending.
        pub fn recordVirtualPow2(
            self: *Self,
            cycle: usize,
            pc: u64,
            instruction: u32,
            rs1_val: u64,
            is_virtual: bool,
            do_not_update_pc: bool,
            is_first_in_sequence: bool,
            is_compressed: bool,
        ) !void {
            if (!self.enabled) return;
            const rd: u8 = @truncate((instruction >> 7) & 0x1f);
            const shift: u6 = @truncate(rs1_val & 0x3F);
            const result: u64 = @as(u64, 1) << shift;
            // AddOperands: lookup index = rs1 + imm = rs1 + 0 = rs1
            const index: u128 = @as(u128, rs1_val);

            var cf = CircuitFlagSet.init();
            cf.set(.AddOperands);
            cf.set(.WriteLookupOutputToRD);
            if (is_virtual) cf.set(.VirtualInstruction);
            if (do_not_update_pc) cf.set(.DoNotUpdateUnexpandedPC);
            if (is_first_in_sequence) cf.set(.IsFirstInSequence);
            if (is_compressed) cf.set(.IsCompressed);

            var inf = InstructionFlagSet.init();
            inf.set(.LeftOperandIsRs1Value);
            inf.set(.RightOperandIsImm);
            if (rd != 0) inf.set(.IsRdNotZero);

            const entry = Entry{
                .cycle = cycle,
                .pc = pc,
                .table = .VirtualPow2,
                .index = index,
                .result = result,
                .left_operand = rs1_val,
                .right_operand = 0,
                .circuit_flags = cf,
                .instruction_flags = inf,
                .instruction = instruction,
            };
            try self.entries.append(self.allocator, entry);
        }

        /// Record lookup for a VirtualShiftRightBitmask instruction within a virtual sequence.
        /// VirtualShiftRightBitmask computes the bitmask for a right shift. Stub — full implementation pending.
        pub fn recordVirtualShiftRightBitmask(
            self: *Self,
            cycle: usize,
            pc: u64,
            instruction: u32,
            rs1_val: u64,
            is_virtual: bool,
            do_not_update_pc: bool,
            is_first_in_sequence: bool,
            is_compressed: bool,
        ) !void {
            if (!self.enabled) return;
            const rd: u8 = @truncate((instruction >> 7) & 0x1f);
            const shift: u6 = @truncate(rs1_val & 0x3F);
            const ones: u128 = (@as(u128, 1) << @intCast(64 - @as(u8, shift))) - 1;
            const bitmask: u64 = @truncate(ones << shift);
            // AddOperands: lookup index = rs1 + imm = rs1 + 0 = rs1
            const index: u128 = @as(u128, rs1_val);

            var cf = CircuitFlagSet.init();
            cf.set(.AddOperands);
            cf.set(.WriteLookupOutputToRD);
            if (is_virtual) cf.set(.VirtualInstruction);
            if (do_not_update_pc) cf.set(.DoNotUpdateUnexpandedPC);
            if (is_first_in_sequence) cf.set(.IsFirstInSequence);
            if (is_compressed) cf.set(.IsCompressed);

            var inf = InstructionFlagSet.init();
            inf.set(.LeftOperandIsRs1Value);
            inf.set(.RightOperandIsImm);
            if (rd != 0) inf.set(.IsRdNotZero);

            const entry = Entry{
                .cycle = cycle,
                .pc = pc,
                .table = .VirtualShiftRightBitmask,
                .index = index,
                .result = bitmask,
                .left_operand = rs1_val,
                .right_operand = 0,
                .circuit_flags = cf,
                .instruction_flags = inf,
                .instruction = instruction,
            };
            try self.entries.append(self.allocator, entry);
        }

        /// Record lookup for a VirtualSRL R-type instruction within a virtual sequence.
        /// VirtualSRL(rd, rs1, rs2) performs logical right shift using bitmask in rs2.
        pub fn recordVirtualSRL_R(
            self: *Self,
            cycle: usize,
            pc: u64,
            instruction: u32,
            rs1_val: u64,
            rs2_val: u64,
            is_virtual: bool,
            do_not_update_pc: bool,
            is_first_in_sequence: bool,
            is_compressed: bool,
        ) !void {
            if (!self.enabled) return;
            const rd: u8 = @truncate((instruction >> 7) & 0x1f);
            const index = lookup_table.interleaveBits(rs1_val, rs2_val);
            // Result: logical right shift by trailing zeros of bitmask
            const shift: u6 = @truncate(@ctz(rs2_val));
            const result: u64 = rs1_val >> shift;

            var cf = CircuitFlagSet.init();
            cf.set(.WriteLookupOutputToRD);
            if (is_virtual) cf.set(.VirtualInstruction);
            if (do_not_update_pc) cf.set(.DoNotUpdateUnexpandedPC);
            if (is_first_in_sequence) cf.set(.IsFirstInSequence);
            if (is_compressed) cf.set(.IsCompressed);

            var inf = InstructionFlagSet.init();
            inf.set(.LeftOperandIsRs1Value);
            inf.set(.RightOperandIsRs2Value);
            if (rd != 0) inf.set(.IsRdNotZero);

            const entry = Entry{
                .cycle = cycle,
                .pc = pc,
                .table = .VirtualSRL,
                .index = index,
                .result = result,
                .left_operand = rs1_val,
                .right_operand = rs2_val,
                .circuit_flags = cf,
                .instruction_flags = inf,
                .instruction = instruction,
            };
            try self.entries.append(self.allocator, entry);
        }

        /// Record lookup for a VirtualSRA R-type instruction within a virtual sequence.
        /// VirtualSRA(rd, rs1, rs2) performs arithmetic right shift using bitmask in rs2.
        pub fn recordVirtualSRA_R(
            self: *Self,
            cycle: usize,
            pc: u64,
            instruction: u32,
            rs1_val: u64,
            rs2_val: u64,
            is_virtual: bool,
            do_not_update_pc: bool,
            is_first_in_sequence: bool,
            is_compressed: bool,
        ) !void {
            if (!self.enabled) return;
            const rd: u8 = @truncate((instruction >> 7) & 0x1f);
            const index = lookup_table.interleaveBits(rs1_val, rs2_val);
            // Result: arithmetic right shift by trailing zeros of bitmask
            const shift: u6 = @truncate(@ctz(rs2_val));
            const result: u64 = @bitCast(@as(i64, @bitCast(rs1_val)) >> shift);

            var cf = CircuitFlagSet.init();
            cf.set(.WriteLookupOutputToRD);
            if (is_virtual) cf.set(.VirtualInstruction);
            if (do_not_update_pc) cf.set(.DoNotUpdateUnexpandedPC);
            if (is_first_in_sequence) cf.set(.IsFirstInSequence);
            if (is_compressed) cf.set(.IsCompressed);

            var inf = InstructionFlagSet.init();
            inf.set(.LeftOperandIsRs1Value);
            inf.set(.RightOperandIsRs2Value);
            if (rd != 0) inf.set(.IsRdNotZero);

            const entry = Entry{
                .cycle = cycle,
                .pc = pc,
                .table = .VirtualSRA,
                .index = index,
                .result = result,
                .left_operand = rs1_val,
                .right_operand = rs2_val,
                .circuit_flags = cf,
                .instruction_flags = inf,
                .instruction = instruction,
            };
            try self.entries.append(self.allocator, entry);
        }

        /// Record lookup for a VirtualAssertHalfwordAlignment instruction.
        /// Asserts that (rs1 + imm) is halfword-aligned.
        pub fn recordVirtualAssertHalfwordAlignment(
            self: *Self,
            cycle: usize,
            pc: u64,
            instruction: u32,
            rs1_val: u64,
            imm_val: u64,
            is_virtual: bool,
            do_not_update_pc: bool,
            is_first_in_sequence: bool,
            is_compressed: bool,
        ) !void {
            if (!self.enabled) return;
            const addr: u64 = rs1_val +% imm_val;
            // AddOperands: lookup index = rs1 + imm
            const index: u128 = @as(u128, addr);

            var cf = CircuitFlagSet.init();
            cf.set(.Assert);
            cf.set(.AddOperands);
            if (is_virtual) cf.set(.VirtualInstruction);
            if (do_not_update_pc) cf.set(.DoNotUpdateUnexpandedPC);
            if (is_first_in_sequence) cf.set(.IsFirstInSequence);
            if (is_compressed) cf.set(.IsCompressed);

            var inf = InstructionFlagSet.init();
            inf.set(.LeftOperandIsRs1Value);
            inf.set(.RightOperandIsImm);

            const entry = Entry{
                .cycle = cycle,
                .pc = pc,
                .table = .VirtualAssertHalfwordAlignment,
                .index = index,
                .result = if (addr & 1 == 0) 1 else 0, // 1 if aligned (assertion passed)
                .left_operand = rs1_val,
                .right_operand = imm_val,
                .circuit_flags = cf,
                .instruction_flags = inf,
                .instruction = instruction,
            };
            try self.entries.append(self.allocator, entry);
        }

        /// Record lookup for a VirtualAssertWordAlignment instruction.
        /// Asserts that (rs1 + imm) is word-aligned.
        pub fn recordVirtualAssertWordAlignment(
            self: *Self,
            cycle: usize,
            pc: u64,
            instruction: u32,
            rs1_val: u64,
            imm_val: u64,
            is_virtual: bool,
            do_not_update_pc: bool,
            is_first_in_sequence: bool,
            is_compressed: bool,
        ) !void {
            if (!self.enabled) return;
            const addr: u64 = rs1_val +% imm_val;
            // AddOperands: lookup index = rs1 + imm
            const index: u128 = @as(u128, addr);

            var cf = CircuitFlagSet.init();
            cf.set(.Assert);
            cf.set(.AddOperands);
            if (is_virtual) cf.set(.VirtualInstruction);
            if (do_not_update_pc) cf.set(.DoNotUpdateUnexpandedPC);
            if (is_first_in_sequence) cf.set(.IsFirstInSequence);
            if (is_compressed) cf.set(.IsCompressed);

            var inf = InstructionFlagSet.init();
            inf.set(.LeftOperandIsRs1Value);
            inf.set(.RightOperandIsImm);

            const entry = Entry{
                .cycle = cycle,
                .pc = pc,
                .table = .VirtualAssertWordAlignment,
                .index = index,
                .result = if (addr & 3 == 0) 1 else 0, // 1 if aligned (assertion passed)
                .left_operand = rs1_val,
                .right_operand = imm_val,
                .circuit_flags = cf,
                .instruction_flags = inf,
                .instruction = instruction,
            };
            try self.entries.append(self.allocator, entry);
        }

        /// Get the number of lookup entries
        pub fn len(self: *const Self) usize {
            return self.entries.items.len;
        }

        /// Get entry at index
        pub fn get(self: *const Self, index: usize) ?Entry {
            if (index >= self.entries.items.len) return null;
            return self.entries.items[index];
        }

        /// Get all entries as a slice
        pub fn getEntries(self: *const Self) []const Entry {
            return self.entries.items;
        }

        /// Count lookups by table type
        pub fn countByTable(self: *const Self, table: LookupTables(XLEN)) usize {
            var count: usize = 0;
            for (self.entries.items) |entry| {
                if (entry.table == table) {
                    count += 1;
                }
            }
            return count;
        }

        /// Get statistics about the lookup trace
        pub const Stats = struct {
            total_lookups: usize,
            and_lookups: usize,
            or_lookups: usize,
            xor_lookups: usize,
            sub_lookups: usize,
            range_check_lookups: usize,
            signed_lt_lookups: usize,
            unsigned_lt_lookups: usize,
            equal_lookups: usize,
            not_equal_lookups: usize,
        };

        pub fn getStats(self: *const Self) Stats {
            return Stats{
                .total_lookups = self.entries.items.len,
                .and_lookups = self.countByTable(.And),
                .or_lookups = self.countByTable(.Or),
                .xor_lookups = self.countByTable(.Xor),
                .sub_lookups = self.countByTable(.Sub),
                .range_check_lookups = self.countByTable(.RangeCheck),
                .signed_lt_lookups = self.countByTable(.SignedLessThan),
                .unsigned_lt_lookups = self.countByTable(.UnsignedLessThan),
                .equal_lookups = self.countByTable(.Equal),
                .not_equal_lookups = self.countByTable(.NotEqual),
            };
        }
    };
}

// ============================================================================
// Tests
// ============================================================================

test "lookup entry creation" {
    const Entry = LookupEntry(64);

    // Test ADD entry
    const add_entry = Entry.fromBinaryLookup(lookups.AddLookup(64), 0, 0x1000, 0x00208033, 10, 20);
    try std.testing.expectEqual(@as(usize, 0), add_entry.cycle);
    try std.testing.expectEqual(@as(u64, 0x1000), add_entry.pc);
    try std.testing.expectEqual(@as(u64, 30), add_entry.result);
    try std.testing.expectEqual(LookupTables(64).RangeCheck, add_entry.table);

    // Test SUB entry
    const sub_entry = Entry.fromBinaryLookup(lookups.SubLookup(64), 1, 0x1004, 0x40208033, 30, 10);
    try std.testing.expectEqual(@as(u64, 20), sub_entry.result);
    try std.testing.expectEqual(LookupTables(64).Sub, sub_entry.table);

    // Test AND entry
    const and_entry = Entry.fromBinaryLookup(lookups.AndLookup(64), 2, 0x1008, 0x00207033, 0xFF, 0x0F);
    try std.testing.expectEqual(@as(u64, 0x0F), and_entry.result);
    try std.testing.expectEqual(LookupTables(64).And, and_entry.table);
}

test "lookup trace collector basic" {
    const allocator = std.testing.allocator;
    const Collector = LookupTraceCollector(64);
    const Entry = LookupEntry(64);

    var collector = Collector.init(allocator);
    defer collector.deinit();

    // Record some entries
    try collector.record(Entry.fromBinaryLookup(lookups.AddLookup(64), 0, 0x1000, 0x00208033, 5, 3));
    try collector.record(Entry.fromBinaryLookup(lookups.SubLookup(64), 1, 0x1004, 0x40208033, 10, 4));
    try collector.record(Entry.fromBinaryLookup(lookups.AndLookup(64), 2, 0x1008, 0x00207033, 0xFF, 0x0F));

    try std.testing.expectEqual(@as(usize, 3), collector.len());

    // Check stats
    const stats = collector.getStats();
    try std.testing.expectEqual(@as(usize, 3), stats.total_lookups);
    try std.testing.expectEqual(@as(usize, 1), stats.range_check_lookups); // ADD
    try std.testing.expectEqual(@as(usize, 1), stats.sub_lookups);
    try std.testing.expectEqual(@as(usize, 1), stats.and_lookups);
}

test "lookup trace collector record instruction" {
    const allocator = std.testing.allocator;
    const Collector = LookupTraceCollector(64);

    var collector = Collector.init(allocator);
    defer collector.deinit();

    // Test ADD instruction: add x1, x2, x3
    // opcode=0x33 (OP), rd=1, rs1=2, rs2=3, funct3=0, funct7=0
    const add_instr: u32 = 0x003100b3;
    const add_decoded = DecodedInstruction.decode(add_instr);
    try collector.recordInstruction(0, 0x1000, add_instr, add_decoded, 10, 20);

    try std.testing.expectEqual(@as(usize, 1), collector.len());
    const entry = collector.get(0).?;
    try std.testing.expectEqual(@as(u64, 30), entry.result);
    try std.testing.expectEqual(LookupTables(64).RangeCheck, entry.table);

    // Test SUB instruction: sub x1, x2, x3
    // opcode=0x33 (OP), rd=1, rs1=2, rs2=3, funct3=0, funct7=0x20
    const sub_instr: u32 = 0x403100b3;
    const sub_decoded = DecodedInstruction.decode(sub_instr);
    try collector.recordInstruction(1, 0x1004, sub_instr, sub_decoded, 30, 10);

    try std.testing.expectEqual(@as(usize, 2), collector.len());
    const sub_entry = collector.get(1).?;
    try std.testing.expectEqual(@as(u64, 20), sub_entry.result);
    try std.testing.expectEqual(LookupTables(64).Sub, sub_entry.table);
}

test "lookup trace collector immediate instructions" {
    const allocator = std.testing.allocator;
    const Collector = LookupTraceCollector(64);

    var collector = Collector.init(allocator);
    defer collector.deinit();

    // Test ADDI instruction: addi x1, x2, 42
    // opcode=0x13 (OP_IMM), rd=1, rs1=2, imm=42, funct3=0
    const addi_instr: u32 = 0x02a10093;
    const addi_decoded = DecodedInstruction.decode(addi_instr);
    try collector.recordInstruction(0, 0x1000, addi_instr, addi_decoded, 10, 0);

    try std.testing.expectEqual(@as(usize, 1), collector.len());
    const entry = collector.get(0).?;
    try std.testing.expectEqual(@as(u64, 52), entry.result); // 10 + 42
}

test "lookup trace collector branch instructions" {
    const allocator = std.testing.allocator;
    const Collector = LookupTraceCollector(64);

    var collector = Collector.init(allocator);
    defer collector.deinit();

    // Test BEQ instruction
    // opcode=0x63 (BRANCH), funct3=0 (BEQ)
    const beq_instr: u32 = 0x00208063;
    const beq_decoded = DecodedInstruction.decode(beq_instr);
    try collector.recordInstruction(0, 0x1000, beq_instr, beq_decoded, 42, 42);

    try std.testing.expectEqual(@as(usize, 1), collector.len());
    const entry = collector.get(0).?;
    try std.testing.expectEqual(@as(u64, 1), entry.result); // Equal
    try std.testing.expectEqual(LookupTables(64).Equal, entry.table);
    try std.testing.expect(entry.instruction_flags.get(.Branch));
}

test "lookup trace collector disabled" {
    const allocator = std.testing.allocator;
    const Collector = LookupTraceCollector(64);
    const Entry = LookupEntry(64);

    var collector = Collector.init(allocator);
    defer collector.deinit();

    // Disable collection
    collector.setEnabled(false);

    // Record should be no-op
    try collector.record(Entry.fromBinaryLookup(lookups.AddLookup(64), 0, 0x1000, 0x00208033, 5, 3));
    try std.testing.expectEqual(@as(usize, 0), collector.len());

    // Re-enable
    collector.setEnabled(true);
    try collector.record(Entry.fromBinaryLookup(lookups.AddLookup(64), 0, 0x1000, 0x00208033, 5, 3));
    try std.testing.expectEqual(@as(usize, 1), collector.len());
}
