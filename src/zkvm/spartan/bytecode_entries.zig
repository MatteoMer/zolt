//! Bytecode entry construction for the Jolt R1CS bytecode-checking protocol.
//!
//! This module contains the BytecodeEntry struct and all functions for populating
//! bytecode entries from raw ELF instructions, Jolt preprocessing, and virtual
//! instruction sequences. Extracted from stage6_prover.zig for modularity.

const std = @import("std");
const Allocator = std.mem.Allocator;

const instruction_mod = @import("../instruction/mod.zig");
const CircuitFlags = instruction_mod.CircuitFlags;
const InstructionFlags = instruction_mod.InstructionFlags;
const preprocessing = @import("../preprocessing.zig");
const BytecodePCMapper = preprocessing.BytecodePCMapper;
const tracer = @import("../../tracer/mod.zig");

const zkvm_debug = @import("../debug.zig");
const dbg = zkvm_debug.dbg;

/// Bytecode entry properties needed for BytecodeReadRaf Val polynomial computation.
/// One entry per bytecode address k. Indexed by the expanded PC (bytecode array index).
pub const BytecodeEntry = struct {
    /// ELF address of this instruction (unexpanded_pc)
    address: u64,
    /// Immediate value (field element representation)
    imm: i64,
    /// Register indices (5-bit)
    rd: u8,
    rs1: u8,
    rs2: u8,
    /// Circuit flags (14 flags, matches CircuitFlags enum)
    circuit_flags: [14]bool,
    /// Instruction flags (7 flags, matches InstructionFlags enum)
    instruction_flags: [7]bool,
    /// Lookup table index (0..39, or 255 for no lookup table)
    lookup_table_index: u8,
    /// Whether operands are interleaved (not combined arithmetically)
    is_interleaved: bool,
    /// Virtual sequence remaining count
    virtual_sequence_remaining: ?u16,
    /// Whether this is the first in a virtual sequence
    is_first_in_sequence: bool,
    /// Raw opcode (7-bit) for Imm encoding discrimination.
    /// Needed because the R1CS witness uses different Imm encodings:
    ///   - ADDI/ADDIW/JAL/JALR: unsigned u64 bitcast of sign-extended i64
    ///   - LUI/AUIPC: sign-extended u64 (instr & 0xFFFFF000 as i32 as i64 as u64)
    ///   - Load/Store/Branch: signed field value (p - |imm| for negative)
    opcode: u8,
    /// funct3 field (3-bit) for ADDI vs other I-type discrimination
    funct3: u3,
};

/// Populate a BytecodeEntry for a VirtualMULI instruction (SLLI decomposition).
/// VirtualMULI has opcode 0x2B with: MultiplyOperands, WriteLookupOutputToRD,
/// LeftOperandIsRs1Value, RightOperandIsImm, lookup table = RangeCheck (0).
fn populateVirtualMULIEntry(
    entry: *BytecodeEntry,
    rd: u8,
    rs1: u8,
    elf_address: u64,
    shamt: u6,
    /// virtual_sequence_remaining: null if standalone, Some(N) if part of a sequence.
    /// For standalone SLLI: 0 (it's the only instruction, so vsr=0).
    /// For SLLIW first entry: 1 (one instruction remaining after this).
    virtual_sequence_remaining: ?u16,
    is_first_in_sequence: bool,
) void {
    entry.address = elf_address;
    // The immediate in the bytecode entry is the multiplier (1 << shamt),
    // matching what preprocessing.zig stores in the FormatI operands.
    const multiplier: u64 = @as(u64, 1) << shamt;
    entry.imm = @intCast(multiplier);

    entry.rd = rd; // rd=0 contributes eq_r_register[0] in Jolt (Some(0), not None)
    entry.rs1 = rs1;
    entry.rs2 = 255; // VirtualMULI is I-type: no rs2

    entry.opcode = 0x2B;
    entry.funct3 = 0;

    entry.circuit_flags = [_]bool{false} ** 14;
    entry.instruction_flags = [_]bool{false} ** 7;

    var cf = &entry.circuit_flags;
    cf[@intFromEnum(CircuitFlags.WriteLookupOutputToRD)] = true;
    cf[@intFromEnum(CircuitFlags.MultiplyOperands)] = true;
    // VirtualInstruction = true when virtual_sequence_remaining.is_some()
    // This applies to ALL virtual instructions, even standalone SLLI (which has vsr=Some(0))
    if (virtual_sequence_remaining != null) {
        cf[@intFromEnum(CircuitFlags.VirtualInstruction)] = true;
    }
    // DoNotUpdateUnexpandedPC = true when vsr > 0 (not the last in sequence)
    if (virtual_sequence_remaining) |vsr| {
        if (vsr != 0) {
            cf[@intFromEnum(CircuitFlags.DoNotUpdateUnexpandedPC)] = true;
        }
    }
    // IsFirstInSequence flag in circuit_flags must match the entry field
    if (is_first_in_sequence) {
        cf[@intFromEnum(CircuitFlags.IsFirstInSequence)] = true;
    }

    var inf = &entry.instruction_flags;
    inf[@intFromEnum(InstructionFlags.LeftOperandIsRs1Value)] = true;
    inf[@intFromEnum(InstructionFlags.RightOperandIsImm)] = true;
    if (rd != 0) {
        inf[@intFromEnum(InstructionFlags.IsRdNotZero)] = true;
    }

    entry.lookup_table_index = 0; // RangeCheck
    // MultiplyOperands is set, so is_interleaved = false (not interleaved)
    entry.is_interleaved = false;
    entry.virtual_sequence_remaining = virtual_sequence_remaining;
    entry.is_first_in_sequence = is_first_in_sequence;
}

/// Populate a BytecodeEntry for a VirtualSignExtendWord instruction.
/// VirtualSignExtendWord has opcode 0x0B with: AddOperands, WriteLookupOutputToRD,
/// LeftOperandIsRs1Value, RightOperandIsImm, lookup table = SignExtendHalfWord (20).
fn populateVirtualSignExtendWordEntry(
    entry: *BytecodeEntry,
    rd: u8,
    elf_address: u64,
    /// is_compressed: only set on the LAST instruction in a virtual sequence per Jolt's finalize()
    is_compressed: bool,
) void {
    entry.address = elf_address;
    entry.imm = 0; // VirtualSignExtendWord always has imm=0

    entry.rd = rd; // rd=0 contributes eq_r_register[0] in Jolt (Some(0), not None)
    entry.rs1 = rd; // Sign-extend reads from rd (rs1 is always set, even for rd=0)
    entry.rs2 = 255; // I-type: no rs2

    entry.opcode = 0x0B;
    entry.funct3 = 0;

    entry.circuit_flags = [_]bool{false} ** 14;
    entry.instruction_flags = [_]bool{false} ** 7;

    var cf = &entry.circuit_flags;
    cf[@intFromEnum(CircuitFlags.WriteLookupOutputToRD)] = true;
    cf[@intFromEnum(CircuitFlags.AddOperands)] = true;
    // VirtualSignExtendWord always has virtual_sequence_remaining=Some(0),
    // so VirtualInstruction=true, DoNotUpdateUnexpandedPC=false
    cf[@intFromEnum(CircuitFlags.VirtualInstruction)] = true;
    // is_compressed is inherited from the original instruction (only on last entry)
    if (is_compressed) {
        cf[@intFromEnum(CircuitFlags.IsCompressed)] = true;
    }

    var inf = &entry.instruction_flags;
    inf[@intFromEnum(InstructionFlags.LeftOperandIsRs1Value)] = true;
    // NOTE: VirtualSignExtendWord does NOT set RightOperandIsImm!
    // This is confirmed in Jolt's virtual_sign_extend_word.rs instruction_flags()
    if (rd != 0) {
        inf[@intFromEnum(InstructionFlags.IsRdNotZero)] = true;
    }

    entry.lookup_table_index = 20; // SignExtendHalfWord
    // AddOperands is set, so is_interleaved = false
    entry.is_interleaved = false;
    // VirtualSignExtendWord is always the last in a sequence: vsr=Some(0)
    entry.virtual_sequence_remaining = 0;
    // Never the first in a sequence (always the 2nd entry)
    entry.is_first_in_sequence = false;
}

/// Populate a BytecodeEntry for VirtualSignExtendWord within a multi-step virtual sequence.
/// Unlike populateVirtualSignExtendWordEntry, this allows specifying rs1 and vsr explicitly
/// (used in REMW/DIVW 21-step sequences where VirtualSignExtendWord appears at non-terminal positions).
fn populateVirtualSignExtendWordEntryWithParams(
    entry: *BytecodeEntry,
    rd: u8,
    rs1: u8,
    elf_address: u64,
    virtual_sequence_remaining: u16,
    is_first_in_sequence: bool,
) void {
    entry.address = elf_address;
    entry.imm = 0;

    entry.rd = rd; // rd=0 contributes eq_r_register[0] in Jolt (Some(0), not None)
    entry.rs1 = rs1;
    entry.rs2 = 255;

    entry.opcode = 0x0B;
    entry.funct3 = 0;

    entry.circuit_flags = [_]bool{false} ** 14;
    entry.instruction_flags = [_]bool{false} ** 7;

    var cf = &entry.circuit_flags;
    cf[@intFromEnum(CircuitFlags.WriteLookupOutputToRD)] = true;
    cf[@intFromEnum(CircuitFlags.AddOperands)] = true;
    cf[@intFromEnum(CircuitFlags.VirtualInstruction)] = true;
    if (virtual_sequence_remaining != 0) {
        cf[@intFromEnum(CircuitFlags.DoNotUpdateUnexpandedPC)] = true;
    }
    if (is_first_in_sequence) {
        cf[@intFromEnum(CircuitFlags.IsFirstInSequence)] = true;
    }

    var inf = &entry.instruction_flags;
    inf[@intFromEnum(InstructionFlags.LeftOperandIsRs1Value)] = true;
    if (rd != 0) {
        inf[@intFromEnum(InstructionFlags.IsRdNotZero)] = true;
    }

    entry.lookup_table_index = 20; // SignExtendHalfWord
    entry.is_interleaved = false;
    entry.virtual_sequence_remaining = virtual_sequence_remaining;
    entry.is_first_in_sequence = is_first_in_sequence;
}

/// Populate a BytecodeEntry for a VirtualSRLI instruction.
/// VirtualSRLI has opcode 0x5B with: WriteLookupOutputToRD (NO AddOperands, NO MultiplyOperands),
/// LeftOperandIsRs1Value, RightOperandIsImm, lookup table = VirtualSRL (25).
/// The immediate is a BITMASK reconstructed from the total shift amount.
fn populateVirtualSRLIEntry(
    entry: *BytecodeEntry,
    rd: u8,
    rs1: u8,
    elf_address: u64,
    /// bitmask: the 64-bit bitmask for VirtualSRLI (NOT the shift amount).
    /// Computed as: ones = (1 << (64 - total_shift)) - 1; bitmask = ones << total_shift.
    bitmask: u64,
    /// virtual_sequence_remaining: null if standalone, Some(N) if part of a sequence.
    virtual_sequence_remaining: ?u16,
    is_first_in_sequence: bool,
) void {
    entry.address = elf_address;
    // The immediate in the bytecode entry is the bitmask,
    // matching what preprocessing.zig stores in the FormatI operands.
    entry.imm = @bitCast(bitmask);

    entry.rd = rd; // rd=0 contributes eq_r_register[0] in Jolt (Some(0), not None)
    entry.rs1 = rs1;
    entry.rs2 = 255; // VirtualSRLI is I-type: no rs2

    entry.opcode = 0x5B;
    entry.funct3 = 0;

    entry.circuit_flags = [_]bool{false} ** 14;
    entry.instruction_flags = [_]bool{false} ** 7;

    var cf = &entry.circuit_flags;
    cf[@intFromEnum(CircuitFlags.WriteLookupOutputToRD)] = true;
    // VirtualSRLI does NOT set AddOperands, SubtractOperands, or MultiplyOperands.
    // It uses interleaved operands with VirtualSRL table (table index 25).
    if (virtual_sequence_remaining != null) {
        cf[@intFromEnum(CircuitFlags.VirtualInstruction)] = true;
    }
    if (virtual_sequence_remaining) |vsr| {
        if (vsr != 0) {
            cf[@intFromEnum(CircuitFlags.DoNotUpdateUnexpandedPC)] = true;
        }
    }
    if (is_first_in_sequence) {
        cf[@intFromEnum(CircuitFlags.IsFirstInSequence)] = true;
    }

    var inf = &entry.instruction_flags;
    inf[@intFromEnum(InstructionFlags.LeftOperandIsRs1Value)] = true;
    inf[@intFromEnum(InstructionFlags.RightOperandIsImm)] = true;
    if (rd != 0) {
        inf[@intFromEnum(InstructionFlags.IsRdNotZero)] = true;
    }

    entry.lookup_table_index = 25; // VirtualSRL
    // VirtualSRLI uses interleaved operands (NOT identity path)
    entry.is_interleaved = true;
    entry.virtual_sequence_remaining = virtual_sequence_remaining;
    entry.is_first_in_sequence = is_first_in_sequence;
}

/// Populate a BytecodeEntry for a VirtualAdvice instruction (opcode 0x02).
/// VirtualAdvice: Advice flag set, WriteLookupOutputToRD, no operand flags.
/// Jolt's instruction_inputs = (0, 0), lookup table = RangeCheck (0), identity-path.
fn populateVirtualAdviceEntry(
    entry: *BytecodeEntry,
    rd: u8,
    elf_address: u64,
    virtual_sequence_remaining: ?u16,
    is_first_in_sequence: bool,
) void {
    entry.address = elf_address;
    entry.imm = 0;

    entry.rd = rd; // rd=0 contributes eq_r_register[0] in Jolt (Some(0), not None)
    entry.rs1 = 255; // VirtualAdvice has no rs1
    entry.rs2 = 255; // VirtualAdvice has no rs2

    entry.opcode = 0x02;
    entry.funct3 = 0;

    entry.circuit_flags = [_]bool{false} ** 14;
    entry.instruction_flags = [_]bool{false} ** 7;

    var cf = &entry.circuit_flags;
    cf[@intFromEnum(CircuitFlags.WriteLookupOutputToRD)] = true;
    cf[@intFromEnum(CircuitFlags.Advice)] = true;
    if (virtual_sequence_remaining != null) {
        cf[@intFromEnum(CircuitFlags.VirtualInstruction)] = true;
    }
    if (virtual_sequence_remaining) |vsr| {
        if (vsr != 0) {
            cf[@intFromEnum(CircuitFlags.DoNotUpdateUnexpandedPC)] = true;
        }
    }
    if (is_first_in_sequence) {
        cf[@intFromEnum(CircuitFlags.IsFirstInSequence)] = true;
    }

    // VirtualAdvice: NO operand flags (no LeftOperandIsRs1Value, etc.)
    // Only IsRdNotZero if rd != 0
    var inf = &entry.instruction_flags;
    if (rd != 0) {
        inf[@intFromEnum(InstructionFlags.IsRdNotZero)] = true;
    }

    entry.lookup_table_index = 0; // RangeCheck
    // Advice flag set → identity-path (not interleaved)
    entry.is_interleaved = false;
    entry.virtual_sequence_remaining = virtual_sequence_remaining;
    entry.is_first_in_sequence = is_first_in_sequence;
}

/// Populate a BytecodeEntry for a VirtualAssertEQ instruction (opcode 0x22).
/// VirtualAssertEQ: Assert flag set, LeftOperandIsRs1Value, RightOperandIsRs2Value.
/// Lookup table = Equal (6), interleaved-path.
fn populateVirtualAssertEQEntry(
    entry: *BytecodeEntry,
    rs1: u8,
    rs2: u8,
    elf_address: u64,
    virtual_sequence_remaining: ?u16,
    is_first_in_sequence: bool,
) void {
    entry.address = elf_address;
    entry.imm = 0;

    entry.rd = 255; // Assert instructions don't write to rd
    entry.rs1 = rs1;
    entry.rs2 = rs2;

    entry.opcode = 0x22;
    entry.funct3 = 0;

    entry.circuit_flags = [_]bool{false} ** 14;
    entry.instruction_flags = [_]bool{false} ** 7;

    var cf = &entry.circuit_flags;
    cf[@intFromEnum(CircuitFlags.Assert)] = true;
    if (virtual_sequence_remaining != null) {
        cf[@intFromEnum(CircuitFlags.VirtualInstruction)] = true;
    }
    if (virtual_sequence_remaining) |vsr| {
        if (vsr != 0) {
            cf[@intFromEnum(CircuitFlags.DoNotUpdateUnexpandedPC)] = true;
        }
    }
    if (is_first_in_sequence) {
        cf[@intFromEnum(CircuitFlags.IsFirstInSequence)] = true;
    }

    var inf = &entry.instruction_flags;
    inf[@intFromEnum(InstructionFlags.LeftOperandIsRs1Value)] = true;
    inf[@intFromEnum(InstructionFlags.RightOperandIsRs2Value)] = true;
    // Assert instructions: IsRdNotZero is false (no rd write)

    entry.lookup_table_index = 6; // Equal
    // No AddOperands/SubtractOperands/MultiplyOperands/Advice → interleaved
    entry.is_interleaved = true;
    entry.virtual_sequence_remaining = virtual_sequence_remaining;
    entry.is_first_in_sequence = is_first_in_sequence;
}

/// Populate a BytecodeEntry for a VirtualZeroExtendWord instruction (opcode 0x42).
/// VirtualZeroExtendWord: AddOperands flag set, WriteLookupOutputToRD, LeftOperandIsRs1Value.
/// Lookup table = LowerHalfWord (20), identity-path.
fn populateVirtualZeroExtendWordEntry(
    entry: *BytecodeEntry,
    rd: u8,
    rs1: u8,
    elf_address: u64,
    virtual_sequence_remaining: ?u16,
    is_first_in_sequence: bool,
) void {
    entry.address = elf_address;
    entry.imm = 0;

    entry.rd = rd; // rd=0 contributes eq_r_register[0] in Jolt (Some(0), not None)
    entry.rs1 = rs1;
    entry.rs2 = 255; // No rs2

    entry.opcode = 0x42;
    entry.funct3 = 0;

    entry.circuit_flags = [_]bool{false} ** 14;
    entry.instruction_flags = [_]bool{false} ** 7;

    var cf = &entry.circuit_flags;
    cf[@intFromEnum(CircuitFlags.WriteLookupOutputToRD)] = true;
    cf[@intFromEnum(CircuitFlags.AddOperands)] = true;
    if (virtual_sequence_remaining != null) {
        cf[@intFromEnum(CircuitFlags.VirtualInstruction)] = true;
    }
    if (virtual_sequence_remaining) |vsr| {
        if (vsr != 0) {
            cf[@intFromEnum(CircuitFlags.DoNotUpdateUnexpandedPC)] = true;
        }
    }
    if (is_first_in_sequence) {
        cf[@intFromEnum(CircuitFlags.IsFirstInSequence)] = true;
    }

    var inf = &entry.instruction_flags;
    inf[@intFromEnum(InstructionFlags.LeftOperandIsRs1Value)] = true;
    if (rd != 0) {
        inf[@intFromEnum(InstructionFlags.IsRdNotZero)] = true;
    }

    entry.lookup_table_index = 19; // LowerHalfWord
    // AddOperands set → identity-path (not interleaved)
    entry.is_interleaved = false;
    entry.virtual_sequence_remaining = virtual_sequence_remaining;
    entry.is_first_in_sequence = is_first_in_sequence;
}

/// Populate a BytecodeEntry for a VirtualAssertValidUnsignedRemainder instruction (opcode 0x62).
/// VirtualAssertValidUnsignedRemainder: Assert flag set, LeftOperandIsRs1Value, RightOperandIsRs2Value.
/// Lookup table = ValidUnsignedRemainder (15), interleaved-path.
fn populateVirtualAssertValidUnsignedRemainderEntry(
    entry: *BytecodeEntry,
    rs1: u8,
    rs2: u8,
    elf_address: u64,
    virtual_sequence_remaining: ?u16,
    is_first_in_sequence: bool,
) void {
    entry.address = elf_address;
    entry.imm = 0;

    entry.rd = 255; // Assert instructions don't write to rd
    entry.rs1 = rs1;
    entry.rs2 = rs2;

    entry.opcode = 0x62;
    entry.funct3 = 0;

    entry.circuit_flags = [_]bool{false} ** 14;
    entry.instruction_flags = [_]bool{false} ** 7;

    var cf = &entry.circuit_flags;
    cf[@intFromEnum(CircuitFlags.Assert)] = true;
    if (virtual_sequence_remaining != null) {
        cf[@intFromEnum(CircuitFlags.VirtualInstruction)] = true;
    }
    if (virtual_sequence_remaining) |vsr| {
        if (vsr != 0) {
            cf[@intFromEnum(CircuitFlags.DoNotUpdateUnexpandedPC)] = true;
        }
    }
    if (is_first_in_sequence) {
        cf[@intFromEnum(CircuitFlags.IsFirstInSequence)] = true;
    }

    var inf = &entry.instruction_flags;
    inf[@intFromEnum(InstructionFlags.LeftOperandIsRs1Value)] = true;
    inf[@intFromEnum(InstructionFlags.RightOperandIsRs2Value)] = true;

    entry.lookup_table_index = 15; // ValidUnsignedRemainder
    // No AddOperands/SubtractOperands/MultiplyOperands/Advice → interleaved
    entry.is_interleaved = true;
    entry.virtual_sequence_remaining = virtual_sequence_remaining;
    entry.is_first_in_sequence = is_first_in_sequence;
}

/// Populate a BytecodeEntry for a VirtualAssertValidDiv0 instruction.
/// Assert + VirtualInstruction flags. Lookup table = ValidDiv0 (16).
fn populateVirtualAssertValidDiv0Entry(
    entry: *BytecodeEntry,
    rs1: u8,
    rs2: u8,
    elf_address: u64,
    virtual_sequence_remaining: ?u16,
    is_first_in_sequence: bool,
) void {
    entry.address = elf_address;
    entry.imm = 0;
    entry.rd = 255;
    entry.rs1 = rs1;
    entry.rs2 = rs2;
    entry.opcode = 0x22;
    entry.funct3 = 1; // funct3=1 distinguishes VirtualAssertValidDiv0 from VirtualAssertEQ (funct3=0)
    entry.circuit_flags = [_]bool{false} ** 14;
    entry.instruction_flags = [_]bool{false} ** 7;
    var cf = &entry.circuit_flags;
    cf[@intFromEnum(CircuitFlags.Assert)] = true;
    if (virtual_sequence_remaining != null) cf[@intFromEnum(CircuitFlags.VirtualInstruction)] = true;
    if (virtual_sequence_remaining) |vsr| {
        if (vsr != 0) cf[@intFromEnum(CircuitFlags.DoNotUpdateUnexpandedPC)] = true;
    }
    if (is_first_in_sequence) cf[@intFromEnum(CircuitFlags.IsFirstInSequence)] = true;
    var inf = &entry.instruction_flags;
    inf[@intFromEnum(InstructionFlags.LeftOperandIsRs1Value)] = true;
    inf[@intFromEnum(InstructionFlags.RightOperandIsRs2Value)] = true;
    entry.lookup_table_index = 16; // ValidDiv0
    entry.is_interleaved = true;
    entry.virtual_sequence_remaining = virtual_sequence_remaining;
    entry.is_first_in_sequence = is_first_in_sequence;
}

/// Populate a BytecodeEntry for a VirtualChangeDivisorW instruction (R-format).
/// WriteLookupOutputToRD + VirtualInstruction flags. Lookup table = VirtualChangeDivisorW (30).
fn populateVirtualChangeDivisorWEntry(
    entry: *BytecodeEntry,
    rd: u8,
    rs1: u8,
    rs2: u8,
    elf_address: u64,
    virtual_sequence_remaining: ?u16,
    is_first_in_sequence: bool,
) void {
    entry.address = elf_address;
    entry.imm = 0;
    entry.rd = rd; // rd=0 contributes eq_r_register[0] in Jolt (Some(0), not None)
    entry.rs1 = rs1;
    entry.rs2 = rs2;
    entry.opcode = 0x3b;
    entry.funct3 = 6; // funct3=6 distinguishes VirtualChangeDivisorW from ADDW/SUBW etc
    entry.circuit_flags = [_]bool{false} ** 14;
    entry.instruction_flags = [_]bool{false} ** 7;
    var cf = &entry.circuit_flags;
    cf[@intFromEnum(CircuitFlags.WriteLookupOutputToRD)] = true;
    if (virtual_sequence_remaining != null) cf[@intFromEnum(CircuitFlags.VirtualInstruction)] = true;
    if (virtual_sequence_remaining) |vsr| {
        if (vsr != 0) cf[@intFromEnum(CircuitFlags.DoNotUpdateUnexpandedPC)] = true;
    }
    if (is_first_in_sequence) cf[@intFromEnum(CircuitFlags.IsFirstInSequence)] = true;
    var inf = &entry.instruction_flags;
    inf[@intFromEnum(InstructionFlags.LeftOperandIsRs1Value)] = true;
    inf[@intFromEnum(InstructionFlags.RightOperandIsRs2Value)] = true;
    if (rd != 0) inf[@intFromEnum(InstructionFlags.IsRdNotZero)] = true;
    entry.lookup_table_index = 30; // VirtualChangeDivisorW
    entry.is_interleaved = true; // No Add/Sub/Mul/Advice flags
    entry.virtual_sequence_remaining = virtual_sequence_remaining;
    entry.is_first_in_sequence = is_first_in_sequence;
}

/// Populate a virtual R-type entry within a virtual sequence.
/// Used for XOR, SUB, MUL, ADD within REMW/DIVW sequences.
/// The entry gets its flags from the instruction word after mapping the opcode.
fn populateVirtualRTypeEntry(
    entry: *BytecodeEntry,
    rd: u8,
    rs1: u8,
    rs2: u8,
    elf_address: u64,
    virtual_sequence_remaining: u16,
    is_first_in_sequence: bool,
    opcode: u8,
    funct3: u3,
    funct7: u7,
) void {
    // Build a synthetic instruction word for the R-type operation
    const instr: u32 = (@as(u32, funct7) << 25) |
        (@as(u32, rs2 & 0x1F) << 20) |
        (@as(u32, rs1 & 0x1F) << 15) |
        (@as(u32, funct3) << 12) |
        (@as(u32, if (rd == 0) @as(u8, 0) else (rd & 0x1F)) << 7) |
        @as(u32, opcode);
    populateEntryFromInstruction(entry, instr, elf_address);
    // Override register indices with full virtual register values
    entry.rd = rd; // rd=0 contributes eq_r_register[0] in Jolt (Some(0), not None)
    entry.rs1 = rs1;
    entry.rs2 = rs2;
    // Set virtual sequence flags
    entry.circuit_flags[@intFromEnum(CircuitFlags.VirtualInstruction)] = true;
    if (virtual_sequence_remaining != 0)
        entry.circuit_flags[@intFromEnum(CircuitFlags.DoNotUpdateUnexpandedPC)] = true;
    if (is_first_in_sequence)
        entry.circuit_flags[@intFromEnum(CircuitFlags.IsFirstInSequence)] = true;
    entry.virtual_sequence_remaining = virtual_sequence_remaining;
    entry.is_first_in_sequence = is_first_in_sequence;
}

/// Populate a BytecodeEntry for a VirtualSRAI instruction within a virtual sequence.
/// VirtualSRAI has: WriteLookupOutputToRD, LeftOperandIsRs1Value, RightOperandIsImm,
/// lookup table = VirtualSRA (26), is_interleaved = true.
/// The immediate is a BITMASK (not a shift amount): bitmask = ((1 << (64 - shift)) - 1) << shift.
fn populateVirtualSRAIEntry(
    entry: *BytecodeEntry,
    rd: u8,
    rs1: u8,
    elf_address: u64,
    bitmask: u64,
    virtual_sequence_remaining: u16,
    is_first_in_sequence: bool,
) void {
    entry.address = elf_address;
    entry.imm = @bitCast(@as(i64, @bitCast(bitmask)));
    entry.rd = rd; // rd=0 contributes eq_r_register[0] in Jolt (Some(0), not None)
    entry.rs1 = rs1;
    entry.rs2 = 255;
    entry.opcode = 0x5B; // Virtual instruction opcode space (same as VirtualSRLI)
    entry.funct3 = 5; // funct3=5 distinguishes VirtualSRAI from VirtualSRLI (funct3=0)
    entry.circuit_flags = [_]bool{false} ** 14;
    entry.instruction_flags = [_]bool{false} ** 7;
    var cf = &entry.circuit_flags;
    cf[@intFromEnum(CircuitFlags.WriteLookupOutputToRD)] = true;
    cf[@intFromEnum(CircuitFlags.VirtualInstruction)] = true;
    if (virtual_sequence_remaining != 0)
        cf[@intFromEnum(CircuitFlags.DoNotUpdateUnexpandedPC)] = true;
    if (is_first_in_sequence)
        cf[@intFromEnum(CircuitFlags.IsFirstInSequence)] = true;
    var inf = &entry.instruction_flags;
    inf[@intFromEnum(InstructionFlags.LeftOperandIsRs1Value)] = true;
    inf[@intFromEnum(InstructionFlags.RightOperandIsImm)] = true;
    if (rd != 0) inf[@intFromEnum(InstructionFlags.IsRdNotZero)] = true;
    entry.lookup_table_index = 26; // VirtualSRA
    entry.is_interleaved = true; // No Add/Sub/Mul/Advice flags
    entry.virtual_sequence_remaining = virtual_sequence_remaining;
    entry.is_first_in_sequence = is_first_in_sequence;
}

/// Populate a BytecodeEntry for a VirtualPow2 instruction.
/// VirtualPow2 computes 2^(rs1 % 64). It uses AddOperands + WriteLookupOutputToRD,
/// lookup table = Pow2 (21), NOT interleaved (AddOperands set).
fn populateVirtualPow2Entry(
    entry: *BytecodeEntry,
    rd: u8,
    rs1: u8,
    elf_address: u64,
    virtual_sequence_remaining: u16,
    is_first_in_sequence: bool,
) void {
    entry.address = elf_address;
    entry.imm = 0;
    entry.rd = rd;
    entry.rs1 = rs1;
    entry.rs2 = 255;
    entry.opcode = 0x2B; // Virtual instruction opcode (same space as VirtualMULI)
    entry.funct3 = 1; // funct3=1 distinguishes VirtualPow2 from VirtualMULI (funct3=0)
    entry.circuit_flags = [_]bool{false} ** 14;
    entry.instruction_flags = [_]bool{false} ** 7;
    var cf = &entry.circuit_flags;
    cf[@intFromEnum(CircuitFlags.AddOperands)] = true;
    cf[@intFromEnum(CircuitFlags.WriteLookupOutputToRD)] = true;
    cf[@intFromEnum(CircuitFlags.VirtualInstruction)] = true;
    if (virtual_sequence_remaining != 0)
        cf[@intFromEnum(CircuitFlags.DoNotUpdateUnexpandedPC)] = true;
    if (is_first_in_sequence)
        cf[@intFromEnum(CircuitFlags.IsFirstInSequence)] = true;
    var inf = &entry.instruction_flags;
    inf[@intFromEnum(InstructionFlags.LeftOperandIsRs1Value)] = true;
    inf[@intFromEnum(InstructionFlags.RightOperandIsImm)] = true;
    if (rd != 0) inf[@intFromEnum(InstructionFlags.IsRdNotZero)] = true;
    entry.lookup_table_index = 21; // Pow2
    entry.is_interleaved = false; // AddOperands set → NOT interleaved
    entry.virtual_sequence_remaining = virtual_sequence_remaining;
    entry.is_first_in_sequence = is_first_in_sequence;
}

/// Populate a BytecodeEntry for a VirtualShiftRightBitmask instruction.
/// Computes bitmask for right shift from register value.
/// Uses AddOperands + WriteLookupOutputToRD, lookup table = ShiftRightBitmask (23).
fn populateVirtualShiftRightBitmaskEntry(
    entry: *BytecodeEntry,
    rd: u8,
    rs1: u8,
    elf_address: u64,
    virtual_sequence_remaining: u16,
    is_first_in_sequence: bool,
) void {
    entry.address = elf_address;
    entry.imm = 0;
    entry.rd = rd;
    entry.rs1 = rs1;
    entry.rs2 = 255;
    entry.opcode = 0x2B;
    entry.funct3 = 2; // funct3=2 for VirtualShiftRightBitmask
    entry.circuit_flags = [_]bool{false} ** 14;
    entry.instruction_flags = [_]bool{false} ** 7;
    var cf = &entry.circuit_flags;
    cf[@intFromEnum(CircuitFlags.AddOperands)] = true;
    cf[@intFromEnum(CircuitFlags.WriteLookupOutputToRD)] = true;
    cf[@intFromEnum(CircuitFlags.VirtualInstruction)] = true;
    if (virtual_sequence_remaining != 0)
        cf[@intFromEnum(CircuitFlags.DoNotUpdateUnexpandedPC)] = true;
    if (is_first_in_sequence)
        cf[@intFromEnum(CircuitFlags.IsFirstInSequence)] = true;
    var inf = &entry.instruction_flags;
    inf[@intFromEnum(InstructionFlags.LeftOperandIsRs1Value)] = true;
    inf[@intFromEnum(InstructionFlags.RightOperandIsImm)] = true;
    if (rd != 0) inf[@intFromEnum(InstructionFlags.IsRdNotZero)] = true;
    entry.lookup_table_index = 23; // ShiftRightBitmask
    entry.is_interleaved = false; // AddOperands set
    entry.virtual_sequence_remaining = virtual_sequence_remaining;
    entry.is_first_in_sequence = is_first_in_sequence;
}

/// Populate a BytecodeEntry for VirtualAssertHalfwordAlignment.
/// Assert that (rs1 + imm) is halfword-aligned (divisible by 2).
/// Uses Assert + AddOperands, lookup table = HalfwordAlignment (17).
fn populateVirtualAssertHalfwordAlignmentEntry(
    entry: *BytecodeEntry,
    rs1: u8,
    imm: i64,
    elf_address: u64,
    virtual_sequence_remaining: u16,
    is_first_in_sequence: bool,
) void {
    entry.address = elf_address;
    entry.imm = imm;
    entry.rd = 255; // Assert: no destination register
    entry.rs1 = rs1;
    entry.rs2 = 255;
    entry.opcode = 0x22; // Virtual assert opcode space
    entry.funct3 = 2; // funct3=2 for HalfwordAlignment
    entry.circuit_flags = [_]bool{false} ** 14;
    entry.instruction_flags = [_]bool{false} ** 7;
    var cf = &entry.circuit_flags;
    cf[@intFromEnum(CircuitFlags.Assert)] = true;
    cf[@intFromEnum(CircuitFlags.AddOperands)] = true;
    cf[@intFromEnum(CircuitFlags.VirtualInstruction)] = true;
    if (virtual_sequence_remaining != 0)
        cf[@intFromEnum(CircuitFlags.DoNotUpdateUnexpandedPC)] = true;
    if (is_first_in_sequence)
        cf[@intFromEnum(CircuitFlags.IsFirstInSequence)] = true;
    var inf = &entry.instruction_flags;
    inf[@intFromEnum(InstructionFlags.LeftOperandIsRs1Value)] = true;
    inf[@intFromEnum(InstructionFlags.RightOperandIsImm)] = true;
    // Assert: no rd, so IsRdNotZero = false
    entry.lookup_table_index = 17; // HalfwordAlignment
    entry.is_interleaved = false; // AddOperands set
    entry.virtual_sequence_remaining = virtual_sequence_remaining;
    entry.is_first_in_sequence = is_first_in_sequence;
}

/// Populate a BytecodeEntry for VirtualAssertWordAlignment.
/// Assert that (rs1 + imm) is word-aligned (divisible by 4).
/// Uses Assert + AddOperands, lookup table = WordAlignment (18).
fn populateVirtualAssertWordAlignmentEntry(
    entry: *BytecodeEntry,
    rs1: u8,
    imm: i64,
    elf_address: u64,
    virtual_sequence_remaining: u16,
    is_first_in_sequence: bool,
) void {
    entry.address = elf_address;
    entry.imm = imm;
    entry.rd = 255;
    entry.rs1 = rs1;
    entry.rs2 = 255;
    entry.opcode = 0x22;
    entry.funct3 = 3; // funct3=3 for WordAlignment
    entry.circuit_flags = [_]bool{false} ** 14;
    entry.instruction_flags = [_]bool{false} ** 7;
    var cf = &entry.circuit_flags;
    cf[@intFromEnum(CircuitFlags.Assert)] = true;
    cf[@intFromEnum(CircuitFlags.AddOperands)] = true;
    cf[@intFromEnum(CircuitFlags.VirtualInstruction)] = true;
    if (virtual_sequence_remaining != 0)
        cf[@intFromEnum(CircuitFlags.DoNotUpdateUnexpandedPC)] = true;
    if (is_first_in_sequence)
        cf[@intFromEnum(CircuitFlags.IsFirstInSequence)] = true;
    var inf = &entry.instruction_flags;
    inf[@intFromEnum(InstructionFlags.LeftOperandIsRs1Value)] = true;
    inf[@intFromEnum(InstructionFlags.RightOperandIsImm)] = true;
    entry.lookup_table_index = 18; // WordAlignment
    entry.is_interleaved = false; // AddOperands set
    entry.virtual_sequence_remaining = virtual_sequence_remaining;
    entry.is_first_in_sequence = is_first_in_sequence;
}

/// Populate a BytecodeEntry for VirtualSRL (R-type, register-based logical right shift).
/// Uses WriteLookupOutputToRD ONLY (NO AddOperands), lookup table = VirtualSRL (25).
/// Interleaved (no arithmetic combination of operands).
fn populateVirtualSRLEntry(
    entry: *BytecodeEntry,
    rd: u8,
    rs1: u8,
    rs2: u8,
    elf_address: u64,
    virtual_sequence_remaining: u16,
    is_first_in_sequence: bool,
) void {
    entry.address = elf_address;
    entry.imm = 0;
    entry.rd = rd;
    entry.rs1 = rs1;
    entry.rs2 = rs2;
    entry.opcode = 0x5B;
    entry.funct3 = 0; // Same as VirtualSRLI
    entry.circuit_flags = [_]bool{false} ** 14;
    entry.instruction_flags = [_]bool{false} ** 7;
    var cf = &entry.circuit_flags;
    cf[@intFromEnum(CircuitFlags.WriteLookupOutputToRD)] = true;
    cf[@intFromEnum(CircuitFlags.VirtualInstruction)] = true;
    if (virtual_sequence_remaining != 0)
        cf[@intFromEnum(CircuitFlags.DoNotUpdateUnexpandedPC)] = true;
    if (is_first_in_sequence)
        cf[@intFromEnum(CircuitFlags.IsFirstInSequence)] = true;
    var inf = &entry.instruction_flags;
    inf[@intFromEnum(InstructionFlags.LeftOperandIsRs1Value)] = true;
    inf[@intFromEnum(InstructionFlags.RightOperandIsRs2Value)] = true;
    if (rd != 0) inf[@intFromEnum(InstructionFlags.IsRdNotZero)] = true;
    entry.lookup_table_index = 25; // VirtualSRL
    entry.is_interleaved = true; // No Add/Sub/Mul flags
    entry.virtual_sequence_remaining = virtual_sequence_remaining;
    entry.is_first_in_sequence = is_first_in_sequence;
}

/// Populate a BytecodeEntry for VirtualSRA (R-type, register-based arithmetic right shift).
/// Uses WriteLookupOutputToRD ONLY (NO AddOperands), lookup table = VirtualSRA (26).
fn populateVirtualSRAEntry(
    entry: *BytecodeEntry,
    rd: u8,
    rs1: u8,
    rs2: u8,
    elf_address: u64,
    virtual_sequence_remaining: u16,
    is_first_in_sequence: bool,
) void {
    entry.address = elf_address;
    entry.imm = 0;
    entry.rd = rd;
    entry.rs1 = rs1;
    entry.rs2 = rs2;
    entry.opcode = 0x5B;
    entry.funct3 = 5; // Same as VirtualSRAI
    entry.circuit_flags = [_]bool{false} ** 14;
    entry.instruction_flags = [_]bool{false} ** 7;
    var cf = &entry.circuit_flags;
    cf[@intFromEnum(CircuitFlags.WriteLookupOutputToRD)] = true;
    cf[@intFromEnum(CircuitFlags.VirtualInstruction)] = true;
    if (virtual_sequence_remaining != 0)
        cf[@intFromEnum(CircuitFlags.DoNotUpdateUnexpandedPC)] = true;
    if (is_first_in_sequence)
        cf[@intFromEnum(CircuitFlags.IsFirstInSequence)] = true;
    var inf = &entry.instruction_flags;
    inf[@intFromEnum(InstructionFlags.LeftOperandIsRs1Value)] = true;
    inf[@intFromEnum(InstructionFlags.RightOperandIsRs2Value)] = true;
    if (rd != 0) inf[@intFromEnum(InstructionFlags.IsRdNotZero)] = true;
    entry.lookup_table_index = 26; // VirtualSRA
    entry.is_interleaved = true;
    entry.virtual_sequence_remaining = virtual_sequence_remaining;
    entry.is_first_in_sequence = is_first_in_sequence;
}

/// Populate a BytecodeEntry from a raw 32-bit instruction word and ELF address.
/// This sets all static properties (flags, registers, immediates, lookup table)
/// from the instruction encoding alone, without any trace-specific data.
fn populateEntryFromInstruction(entry: *BytecodeEntry, instr: u32, elf_address: u64) void {
    const opcode: u8 = @truncate(instr & 0x7F);

    // Normalize funct7 for R-type (0x33) and OP-32 (0x3b) instructions.
    // ELF binaries may have non-canonical funct7 values (e.g. from assembler bugs
    // or future extensions). Jolt only recognizes canonical funct7 values, so
    // unknown funct7 → 0 makes the instruction decode as the base variant (ADD/ADDW).
    const funct3_raw: u3 = @truncate((instr >> 12) & 0x7);
    const funct7_raw: u7 = @truncate(instr >> 25);
    var norm_instr = instr;
    if (opcode == 0x33 or opcode == 0x3b) {
        const canonical_funct7: u7 = switch (funct3_raw) {
            0 => if (funct7_raw == 0x20) @as(u7, 0x20) else if (funct7_raw == 0x01) @as(u7, 0x01) else 0,
            5 => if (funct7_raw == 0x20) @as(u7, 0x20) else 0,
            7 => if (funct7_raw == 0x20) @as(u7, 0x20) // ANDN (Zbb)
            else if (funct7_raw == 0x01) @as(u7, 0x01) // REMU
            else 0,
            1, 2, 3, 4, 6 => if (funct7_raw == 0x01) @as(u7, 0x01) else 0,
        };
        norm_instr = (instr & ~(@as(u32, 0x7F) << 25)) | (@as(u32, canonical_funct7) << 25);
    }

    const decoded = instruction_mod.DecodedInstruction.decode(norm_instr);

    entry.address = elf_address;
    // FENCE: imm=0 (Jolt's FENCE uses None operands, no imm contribution)
    entry.imm = if (opcode == 0x0F) 0 else @intCast(decoded.imm);

    // Set rd, rs1, rs2 matching Jolt's NormalizedOperands behavior.
    // We use 255 as sentinel for "not present" so that `entry.X < REGISTER_COUNT`
    // yields false, giving zero contribution in val poly.
    //
    // rd:  sentinel ONLY for S-format (0x23) and B-format (0x63), which have rd=None
    //      in Jolt's NormalizedOperands. For ALL other formats (including rd=0),
    //      Jolt stores rd=Some(rd_value), so rd=0 contributes eq_r_register[0]
    //      (non-zero) to Stages 4 and 5 val polynomials. We must NOT sentinel rd=0.
    // rs1: sentinel for U-type (LUI 0x37, AUIPC 0x17) and J-type (JAL 0x6f)
    // rs2: sentinel for I-type (0x13, 0x03, 0x67, 0x1b), U-type (0x37, 0x17), J-type (0x6f)
    // FENCE (0x0F): no operands in Jolt (None format), so sentinel all registers
    if (opcode == 0x0F) {
        entry.rd = 255;
        entry.rs1 = 255;
        entry.rs2 = 255;
    } else {
        entry.rd = if (opcode == 0x23 or opcode == 0x63) 255 else decoded.rd;
        entry.rs1 = switch (opcode) {
            0x37, 0x17, 0x6f => 255, // U-type, J-type: no rs1
            else => decoded.rs1,
        };
        entry.rs2 = switch (opcode) {
            0x13, 0x03, 0x67, 0x1b, 0x37, 0x17, 0x6f, 0x0B, 0x2B, 0x5B, 0x6B => 255, // I-type, U-type, J-type, Virtual: no rs2
            else => decoded.rs2,
        };
    }
    const funct3: u3 = @truncate((norm_instr >> 12) & 0x7);
    const funct7: u7 = @truncate(norm_instr >> 25);

    // UNIMPL detection: if the opcode/funct3/funct7 combination is not a known
    // RISC-V instruction recognized by Jolt, treat it as UNIMPL (Jolt's Default).
    // Return early with address=0, all flags false, matching Jolt's UNIMPL behavior.
    if (!isKnownInstruction(opcode, funct3, funct7)) {
        // UNIMPL: address = 0 (Jolt's UNIMPL normalizes to Default which has address=0).
        // All flags false, no registers, no lookup table.
        entry.address = 0;
        entry.imm = 0;
        entry.rd = 255;
        entry.rs1 = 255;
        entry.rs2 = 255;
        entry.circuit_flags = [_]bool{false} ** 14;
        entry.instruction_flags = [_]bool{false} ** 7;
        entry.lookup_table_index = 255;
        entry.is_interleaved = true;
        entry.virtual_sequence_remaining = null;
        entry.is_first_in_sequence = false;
        entry.opcode = 0;
        entry.funct3 = 0;
        return;
    }

    // VirtualHostIO (opcode 0x5B, funct3 != 0/5): NOP-like with real address, no flags.
    // In Jolt, VirtualHostIO has circuit_flags=[false; 14] and instruction_flags=[false; 7].
    // It preserves the address but has no lookup, no register writes, no operand flags.
    if (opcode == 0x5B and funct3 != 0 and funct3 != 5) {
        entry.address = elf_address;
        entry.imm = 0;
        entry.rd = 0; // rd=0 from instruction encoding
        entry.rs1 = 0; // rs1=0 from instruction encoding
        entry.rs2 = 255; // no rs2
        entry.opcode = opcode;
        entry.funct3 = funct3;
        entry.circuit_flags = [_]bool{false} ** 14;
        entry.instruction_flags = [_]bool{false} ** 7;
        entry.lookup_table_index = 255;
        entry.is_interleaved = true;
        entry.virtual_sequence_remaining = null;
        entry.is_first_in_sequence = false;
        return;
    }

    // Store opcode and funct3 for Imm encoding discrimination in val poly computation
    entry.opcode = opcode;
    entry.funct3 = funct3;

    // Reset all flags before setting instruction-specific ones.
    // This is critical because entries are initialized with NoOp flags
    // (DoNotUpdateUnexpandedPC=true, IsNoop=true) and real instructions
    // must clear those defaults.
    entry.circuit_flags = [_]bool{false} ** 14;
    entry.instruction_flags = [_]bool{false} ** 7;

    // Circuit flags
    var cf = &entry.circuit_flags;

    // Load/Store
    if (opcode == 0x03) cf[@intFromEnum(CircuitFlags.Load)] = true;
    if (opcode == 0x23) cf[@intFromEnum(CircuitFlags.Store)] = true;

    // Jump
    if (opcode == 0x6F or opcode == 0x67) cf[@intFromEnum(CircuitFlags.Jump)] = true;

    // WriteLookupOutputToRD
    // Must match R1CS witness: instructions that write lookup output to rd.
    // JAL (0x6F) and JALR (0x67) do NOT set this flag - they write PC+4 to rd
    // via the WritePCtoRD mechanism, not the lookup output.
    // BRANCH (0x63) also doesn't set this flag.
    const has_lookup = hasLookupTable(opcode, funct3, funct7);
    if (has_lookup) {
        if (opcode != 0x63 and opcode != 0x6F and opcode != 0x67) {
            cf[@intFromEnum(CircuitFlags.WriteLookupOutputToRD)] = true;
        }
    }

    // AddOperands, SubtractOperands, MultiplyOperands
    // Must match what the R1CS witness sets for FlagAddOperands etc.
    // Note: LUI (0x37), AUIPC (0x17), and JAL (0x6F) all need AddOperands=true
    // even though they use identity-path lookups (no interleaving).
    if (has_lookup) {
        switch (opcode) {
            0x33 => { // R-type
                if (funct3 == 0 and funct7 == 0) cf[@intFromEnum(CircuitFlags.AddOperands)] = true; // ADD
                if (funct3 == 0 and funct7 == 0x20) cf[@intFromEnum(CircuitFlags.SubtractOperands)] = true; // SUB
                if (funct7 == 0x01 and funct3 == 0) cf[@intFromEnum(CircuitFlags.MultiplyOperands)] = true; // MUL
                if (funct7 == 0x01 and funct3 == 3) cf[@intFromEnum(CircuitFlags.MultiplyOperands)] = true; // MULHU
            },
            0x13 => { // I-type
                if (funct3 == 0) cf[@intFromEnum(CircuitFlags.AddOperands)] = true; // ADDI
            },
            0x67 => { // JALR
                cf[@intFromEnum(CircuitFlags.AddOperands)] = true;
            },
            0x37 => { // LUI - identity-path AddOperands
                cf[@intFromEnum(CircuitFlags.AddOperands)] = true;
            },
            0x17 => { // AUIPC - identity-path AddOperands
                cf[@intFromEnum(CircuitFlags.AddOperands)] = true;
            },
            0x6F => { // JAL - identity-path AddOperands
                cf[@intFromEnum(CircuitFlags.AddOperands)] = true;
            },
            0x1b => { // OP-IMM-32
                if (funct3 == 0) cf[@intFromEnum(CircuitFlags.AddOperands)] = true; // ADDIW
            },
            0x3b => { // OP-32
                if (funct3 == 0 and funct7 == 0) cf[@intFromEnum(CircuitFlags.AddOperands)] = true; // ADDW
                if (funct3 == 0 and funct7 == 0x20) cf[@intFromEnum(CircuitFlags.SubtractOperands)] = true; // SUBW
            },
            0x0B => { // VirtualSignExtendWord
                cf[@intFromEnum(CircuitFlags.AddOperands)] = true;
            },
            0x2B => { // VirtualMULI
                cf[@intFromEnum(CircuitFlags.MultiplyOperands)] = true;
            },
            0x73, 0x0F => {
                // ECALL and FENCE: no additional circuit flags beyond defaults
                // (Jolt's ECALL only sets IsFirstInSequence and IsCompressed,
                //  Jolt's FENCE only sets IsCompressed — both from instance fields)
            },
            else => {},
        }
    }

    // Instruction flags
    var inf = &entry.instruction_flags;

    // ECALL/FENCE: instruction_flags are all false in Jolt
    // (NOT IsNoop — that's only for the NoOp instruction variant)

    // LeftOperandIsPC
    if (has_lookup and (opcode == 0x17 or opcode == 0x6F)) {
        inf[@intFromEnum(InstructionFlags.LeftOperandIsPC)] = true;
    }

    // LeftOperandIsRs1Value
    if (has_lookup) {
        switch (opcode) {
            0x33, 0x13, 0x67, 0x63, 0x1B, 0x3B, 0x0B, 0x2B, 0x5B, 0x6B => {
                inf[@intFromEnum(InstructionFlags.LeftOperandIsRs1Value)] = true;
            },
            else => {},
        }
    }

    // RightOperandIsImm
    if (has_lookup) {
        switch (opcode) {
            0x13, 0x67, 0x37, 0x17, 0x6F, 0x1B, 0x0B, 0x2B, 0x5B, 0x6B => {
                inf[@intFromEnum(InstructionFlags.RightOperandIsImm)] = true;
            },
            else => {},
        }
    }

    // RightOperandIsRs2Value
    if (has_lookup) {
        switch (opcode) {
            0x33, 0x63, 0x3B => {
                inf[@intFromEnum(InstructionFlags.RightOperandIsRs2Value)] = true;
            },
            else => {},
        }
    }

    // Branch
    if (opcode == 0x63) {
        inf[@intFromEnum(InstructionFlags.Branch)] = true;
    }

    // IsRdNotZero
    if (decoded.rd != 0 and opcode != 0x23 and opcode != 0x63) {
        inf[@intFromEnum(InstructionFlags.IsRdNotZero)] = true;
    }

    // FENCE and ECALL: In Jolt, these have ALL flags false (circuit and instruction).
    // Jolt's Flags impl for FENCE/ECALL only sets IsFirstInSequence and IsCompressed,
    // which are always false for standalone instructions. NO DoNotUpdateUnexpandedPC, NO IsNoop.

    // Lookup table index and interleaving
    entry.lookup_table_index = getLookupTableIndex(opcode, funct3, funct7);
    entry.is_interleaved = !cf[@intFromEnum(CircuitFlags.AddOperands)] and
        !cf[@intFromEnum(CircuitFlags.SubtractOperands)] and
        !cf[@intFromEnum(CircuitFlags.MultiplyOperands)] and
        !cf[@intFromEnum(CircuitFlags.Advice)];
}

/// Populate a BytecodeEntry from a JoltInstruction (from preprocessing.zig).
/// This is the preferred path when preprocessing bytecode is available,
/// as it ensures the prover's bytecode entries match exactly what the
/// verifier computes from the serialized preprocessing.
///
/// Maps each InstructionVariant to the correct circuit_flags, instruction_flags,
/// lookup_table_index, registers, immediate, opcode, and funct3 fields.
fn populateEntryFromJoltInstruction(entry: *BytecodeEntry, instr: preprocessing.JoltInstruction) void {
    // Extract registers and immediate from the operand format
    var rd: u8 = 255;
    var rs1: u8 = 255;
    var rs2: u8 = 255;
    var imm: i64 = 0;

    switch (instr.operands) {
        .FormatR => |r| {
            rd = r.rd;
            rs1 = r.rs1;
            rs2 = r.rs2;
        },
        .FormatI => |i_op| {
            rd = i_op.rd;
            rs1 = i_op.rs1;
            imm = @bitCast(i_op.imm);
        },
        .FormatLoad => |l| {
            rd = l.rd;
            rs1 = l.rs1;
            imm = l.imm;
        },
        .FormatS => |s| {
            rs1 = s.rs1;
            rs2 = s.rs2;
            imm = s.imm;
        },
        .FormatB => |b| {
            rs1 = b.rs1;
            rs2 = b.rs2;
            imm = @intCast(b.imm);
        },
        .FormatU => |u_op| {
            rd = u_op.rd;
            imm = @bitCast(u_op.imm);
        },
        .FormatJ => |j| {
            rd = j.rd;
            imm = @bitCast(j.imm);
        },
        .FormatAssert => |a| {
            rs1 = a.rs1;
            imm = a.imm;
        },
        .FormatInline => |il| {
            rs1 = il.rs1;
            rs2 = il.rs2;
        },
        .FormatVirtualRightShiftI => |vrs| {
            rd = vrs.rd;
            rs1 = vrs.rs1;
            imm = @bitCast(vrs.imm);
        },
        .None => {},
    }

    const vsr = instr.virtual_sequence_remaining;
    const is_first = instr.is_first_in_sequence;
    const is_compressed = instr.is_compressed;

    // Dispatch to existing populate helpers for virtual instructions
    switch (instr.variant) {
        // =====================================================================
        // Virtual instructions — delegate to dedicated populate functions
        // =====================================================================
        .VirtualMULI => {
            // VirtualMULI has FormatI operands with imm = multiplier (1 << shamt)
            // We need to pass the shamt, but for populateVirtualMULIEntry the imm
            // is set directly from the multiplier. Use the direct population path.
            entry.address = instr.address;
            entry.imm = imm;
            entry.rd = rd;
            entry.rs1 = rs1;
            entry.rs2 = 255;
            entry.opcode = 0x2B;
            entry.funct3 = 0;
            entry.circuit_flags = [_]bool{false} ** 14;
            entry.instruction_flags = [_]bool{false} ** 7;
            var cf = &entry.circuit_flags;
            cf[@intFromEnum(CircuitFlags.WriteLookupOutputToRD)] = true;
            cf[@intFromEnum(CircuitFlags.MultiplyOperands)] = true;
            setVirtualSequenceFlags(cf, vsr, is_first, is_compressed);
            var inf = &entry.instruction_flags;
            inf[@intFromEnum(InstructionFlags.LeftOperandIsRs1Value)] = true;
            inf[@intFromEnum(InstructionFlags.RightOperandIsImm)] = true;
            if (rd != 0 and rd != 255) inf[@intFromEnum(InstructionFlags.IsRdNotZero)] = true;
            entry.lookup_table_index = 0; // RangeCheck
            entry.is_interleaved = false;
            entry.virtual_sequence_remaining = vsr;
            entry.is_first_in_sequence = is_first;
        },
        .VirtualSignExtendWord => {
            entry.address = instr.address;
            entry.imm = 0;
            entry.rd = rd;
            entry.rs1 = rs1;
            entry.rs2 = 255;
            entry.opcode = 0x0B;
            entry.funct3 = 0;
            entry.circuit_flags = [_]bool{false} ** 14;
            entry.instruction_flags = [_]bool{false} ** 7;
            var cf = &entry.circuit_flags;
            cf[@intFromEnum(CircuitFlags.WriteLookupOutputToRD)] = true;
            cf[@intFromEnum(CircuitFlags.AddOperands)] = true;
            setVirtualSequenceFlags(cf, vsr, is_first, is_compressed);
            var inf = &entry.instruction_flags;
            inf[@intFromEnum(InstructionFlags.LeftOperandIsRs1Value)] = true;
            // VirtualSignExtendWord does NOT set RightOperandIsImm
            if (rd != 0 and rd != 255) inf[@intFromEnum(InstructionFlags.IsRdNotZero)] = true;
            entry.lookup_table_index = 20; // SignExtendHalfWord
            entry.is_interleaved = false;
            entry.virtual_sequence_remaining = vsr;
            entry.is_first_in_sequence = is_first;
        },
        .VirtualZeroExtendWord => {
            entry.address = instr.address;
            entry.imm = 0;
            entry.rd = rd;
            entry.rs1 = rs1;
            entry.rs2 = 255;
            entry.opcode = 0x42;
            entry.funct3 = 0;
            entry.circuit_flags = [_]bool{false} ** 14;
            entry.instruction_flags = [_]bool{false} ** 7;
            var cf = &entry.circuit_flags;
            cf[@intFromEnum(CircuitFlags.WriteLookupOutputToRD)] = true;
            cf[@intFromEnum(CircuitFlags.AddOperands)] = true;
            setVirtualSequenceFlags(cf, vsr, is_first, is_compressed);
            var inf = &entry.instruction_flags;
            inf[@intFromEnum(InstructionFlags.LeftOperandIsRs1Value)] = true;
            if (rd != 0 and rd != 255) inf[@intFromEnum(InstructionFlags.IsRdNotZero)] = true;
            entry.lookup_table_index = 19; // LowerHalfWord
            entry.is_interleaved = false;
            entry.virtual_sequence_remaining = vsr;
            entry.is_first_in_sequence = is_first;
        },
        .VirtualSRLI => {
            entry.address = instr.address;
            entry.imm = imm; // bitmask stored as i64
            entry.rd = rd;
            entry.rs1 = rs1;
            entry.rs2 = 255;
            entry.opcode = 0x5B;
            entry.funct3 = 0;
            entry.circuit_flags = [_]bool{false} ** 14;
            entry.instruction_flags = [_]bool{false} ** 7;
            var cf = &entry.circuit_flags;
            cf[@intFromEnum(CircuitFlags.WriteLookupOutputToRD)] = true;
            setVirtualSequenceFlags(cf, vsr, is_first, is_compressed);
            var inf = &entry.instruction_flags;
            inf[@intFromEnum(InstructionFlags.LeftOperandIsRs1Value)] = true;
            inf[@intFromEnum(InstructionFlags.RightOperandIsImm)] = true;
            if (rd != 0 and rd != 255) inf[@intFromEnum(InstructionFlags.IsRdNotZero)] = true;
            entry.lookup_table_index = 25; // VirtualSRL
            entry.is_interleaved = true;
            entry.virtual_sequence_remaining = vsr;
            entry.is_first_in_sequence = is_first;
        },
        .VirtualSRAI => {
            entry.address = instr.address;
            entry.imm = imm; // bitmask stored as i64
            entry.rd = rd;
            entry.rs1 = rs1;
            entry.rs2 = 255;
            entry.opcode = 0x5B;
            entry.funct3 = 5;
            entry.circuit_flags = [_]bool{false} ** 14;
            entry.instruction_flags = [_]bool{false} ** 7;
            var cf = &entry.circuit_flags;
            cf[@intFromEnum(CircuitFlags.WriteLookupOutputToRD)] = true;
            setVirtualSequenceFlags(cf, vsr, is_first, is_compressed);
            var inf = &entry.instruction_flags;
            inf[@intFromEnum(InstructionFlags.LeftOperandIsRs1Value)] = true;
            inf[@intFromEnum(InstructionFlags.RightOperandIsImm)] = true;
            if (rd != 0 and rd != 255) inf[@intFromEnum(InstructionFlags.IsRdNotZero)] = true;
            entry.lookup_table_index = 26; // VirtualSRA
            entry.is_interleaved = true;
            entry.virtual_sequence_remaining = vsr;
            entry.is_first_in_sequence = is_first;
        },
        .VirtualSRL => {
            entry.address = instr.address;
            entry.imm = 0;
            entry.rd = rd;
            entry.rs1 = rs1;
            entry.rs2 = rs2;
            entry.opcode = 0x5B;
            entry.funct3 = 0;
            entry.circuit_flags = [_]bool{false} ** 14;
            entry.instruction_flags = [_]bool{false} ** 7;
            var cf = &entry.circuit_flags;
            cf[@intFromEnum(CircuitFlags.WriteLookupOutputToRD)] = true;
            setVirtualSequenceFlags(cf, vsr, is_first, is_compressed);
            var inf = &entry.instruction_flags;
            inf[@intFromEnum(InstructionFlags.LeftOperandIsRs1Value)] = true;
            inf[@intFromEnum(InstructionFlags.RightOperandIsRs2Value)] = true;
            if (rd != 0 and rd != 255) inf[@intFromEnum(InstructionFlags.IsRdNotZero)] = true;
            entry.lookup_table_index = 25; // VirtualSRL
            entry.is_interleaved = true;
            entry.virtual_sequence_remaining = vsr;
            entry.is_first_in_sequence = is_first;
        },
        .VirtualSRA => {
            entry.address = instr.address;
            entry.imm = 0;
            entry.rd = rd;
            entry.rs1 = rs1;
            entry.rs2 = rs2;
            entry.opcode = 0x5B;
            entry.funct3 = 5;
            entry.circuit_flags = [_]bool{false} ** 14;
            entry.instruction_flags = [_]bool{false} ** 7;
            var cf = &entry.circuit_flags;
            cf[@intFromEnum(CircuitFlags.WriteLookupOutputToRD)] = true;
            setVirtualSequenceFlags(cf, vsr, is_first, is_compressed);
            var inf = &entry.instruction_flags;
            inf[@intFromEnum(InstructionFlags.LeftOperandIsRs1Value)] = true;
            inf[@intFromEnum(InstructionFlags.RightOperandIsRs2Value)] = true;
            if (rd != 0 and rd != 255) inf[@intFromEnum(InstructionFlags.IsRdNotZero)] = true;
            entry.lookup_table_index = 26; // VirtualSRA
            entry.is_interleaved = true;
            entry.virtual_sequence_remaining = vsr;
            entry.is_first_in_sequence = is_first;
        },
        .VirtualAdvice => {
            entry.address = instr.address;
            entry.imm = 0;
            entry.rd = rd;
            entry.rs1 = 255;
            entry.rs2 = 255;
            entry.opcode = 0x02;
            entry.funct3 = 0;
            entry.circuit_flags = [_]bool{false} ** 14;
            entry.instruction_flags = [_]bool{false} ** 7;
            var cf = &entry.circuit_flags;
            cf[@intFromEnum(CircuitFlags.WriteLookupOutputToRD)] = true;
            cf[@intFromEnum(CircuitFlags.Advice)] = true;
            setVirtualSequenceFlags(cf, vsr, is_first, is_compressed);
            var inf = &entry.instruction_flags;
            if (rd != 0 and rd != 255) inf[@intFromEnum(InstructionFlags.IsRdNotZero)] = true;
            entry.lookup_table_index = 0; // RangeCheck
            entry.is_interleaved = false;
            entry.virtual_sequence_remaining = vsr;
            entry.is_first_in_sequence = is_first;
        },
        .VirtualAssertEQ => {
            entry.address = instr.address;
            entry.imm = 0;
            entry.rd = 255;
            entry.rs1 = rs1;
            entry.rs2 = rs2;
            entry.opcode = 0x22;
            entry.funct3 = 0;
            entry.circuit_flags = [_]bool{false} ** 14;
            entry.instruction_flags = [_]bool{false} ** 7;
            var cf = &entry.circuit_flags;
            cf[@intFromEnum(CircuitFlags.Assert)] = true;
            setVirtualSequenceFlags(cf, vsr, is_first, is_compressed);
            var inf = &entry.instruction_flags;
            inf[@intFromEnum(InstructionFlags.LeftOperandIsRs1Value)] = true;
            inf[@intFromEnum(InstructionFlags.RightOperandIsRs2Value)] = true;
            entry.lookup_table_index = 6; // Equal
            entry.is_interleaved = true;
            entry.virtual_sequence_remaining = vsr;
            entry.is_first_in_sequence = is_first;
        },
        .VirtualAssertLTE => {
            entry.address = instr.address;
            entry.imm = 0;
            entry.rd = 255;
            entry.rs1 = rs1;
            entry.rs2 = rs2;
            entry.opcode = 0x22;
            entry.funct3 = 4; // funct3=4 for LTE
            entry.circuit_flags = [_]bool{false} ** 14;
            entry.instruction_flags = [_]bool{false} ** 7;
            var cf = &entry.circuit_flags;
            cf[@intFromEnum(CircuitFlags.Assert)] = true;
            setVirtualSequenceFlags(cf, vsr, is_first, is_compressed);
            var inf = &entry.instruction_flags;
            inf[@intFromEnum(InstructionFlags.LeftOperandIsRs1Value)] = true;
            inf[@intFromEnum(InstructionFlags.RightOperandIsRs2Value)] = true;
            entry.lookup_table_index = 14; // UnsignedLTE
            entry.is_interleaved = true;
            entry.virtual_sequence_remaining = vsr;
            entry.is_first_in_sequence = is_first;
        },
        .VirtualAssertValidDiv0 => {
            entry.address = instr.address;
            entry.imm = 0;
            entry.rd = 255;
            entry.rs1 = rs1;
            entry.rs2 = rs2;
            entry.opcode = 0x22;
            entry.funct3 = 1;
            entry.circuit_flags = [_]bool{false} ** 14;
            entry.instruction_flags = [_]bool{false} ** 7;
            var cf = &entry.circuit_flags;
            cf[@intFromEnum(CircuitFlags.Assert)] = true;
            setVirtualSequenceFlags(cf, vsr, is_first, is_compressed);
            var inf = &entry.instruction_flags;
            inf[@intFromEnum(InstructionFlags.LeftOperandIsRs1Value)] = true;
            inf[@intFromEnum(InstructionFlags.RightOperandIsRs2Value)] = true;
            entry.lookup_table_index = 16; // ValidDiv0
            entry.is_interleaved = true;
            entry.virtual_sequence_remaining = vsr;
            entry.is_first_in_sequence = is_first;
        },
        .VirtualAssertValidUnsignedRemainder => {
            entry.address = instr.address;
            entry.imm = 0;
            entry.rd = 255;
            entry.rs1 = rs1;
            entry.rs2 = rs2;
            entry.opcode = 0x62;
            entry.funct3 = 0;
            entry.circuit_flags = [_]bool{false} ** 14;
            entry.instruction_flags = [_]bool{false} ** 7;
            var cf = &entry.circuit_flags;
            cf[@intFromEnum(CircuitFlags.Assert)] = true;
            setVirtualSequenceFlags(cf, vsr, is_first, is_compressed);
            var inf = &entry.instruction_flags;
            inf[@intFromEnum(InstructionFlags.LeftOperandIsRs1Value)] = true;
            inf[@intFromEnum(InstructionFlags.RightOperandIsRs2Value)] = true;
            entry.lookup_table_index = 15; // ValidUnsignedRemainder
            entry.is_interleaved = true;
            entry.virtual_sequence_remaining = vsr;
            entry.is_first_in_sequence = is_first;
        },
        .VirtualAssertHalfwordAlignment => {
            entry.address = instr.address;
            entry.imm = imm;
            entry.rd = 255;
            entry.rs1 = rs1;
            entry.rs2 = 255;
            entry.opcode = 0x22;
            entry.funct3 = 2;
            entry.circuit_flags = [_]bool{false} ** 14;
            entry.instruction_flags = [_]bool{false} ** 7;
            var cf = &entry.circuit_flags;
            cf[@intFromEnum(CircuitFlags.Assert)] = true;
            cf[@intFromEnum(CircuitFlags.AddOperands)] = true;
            setVirtualSequenceFlags(cf, vsr, is_first, is_compressed);
            var inf = &entry.instruction_flags;
            inf[@intFromEnum(InstructionFlags.LeftOperandIsRs1Value)] = true;
            inf[@intFromEnum(InstructionFlags.RightOperandIsImm)] = true;
            entry.lookup_table_index = 17; // HalfwordAlignment
            entry.is_interleaved = false;
            entry.virtual_sequence_remaining = vsr;
            entry.is_first_in_sequence = is_first;
        },
        .VirtualAssertWordAlignment => {
            entry.address = instr.address;
            entry.imm = imm;
            entry.rd = 255;
            entry.rs1 = rs1;
            entry.rs2 = 255;
            entry.opcode = 0x22;
            entry.funct3 = 3;
            entry.circuit_flags = [_]bool{false} ** 14;
            entry.instruction_flags = [_]bool{false} ** 7;
            var cf = &entry.circuit_flags;
            cf[@intFromEnum(CircuitFlags.Assert)] = true;
            cf[@intFromEnum(CircuitFlags.AddOperands)] = true;
            setVirtualSequenceFlags(cf, vsr, is_first, is_compressed);
            var inf = &entry.instruction_flags;
            inf[@intFromEnum(InstructionFlags.LeftOperandIsRs1Value)] = true;
            inf[@intFromEnum(InstructionFlags.RightOperandIsImm)] = true;
            entry.lookup_table_index = 18; // WordAlignment
            entry.is_interleaved = false;
            entry.virtual_sequence_remaining = vsr;
            entry.is_first_in_sequence = is_first;
        },
        .VirtualChangeDivisorW => {
            entry.address = instr.address;
            entry.imm = 0;
            entry.rd = rd;
            entry.rs1 = rs1;
            entry.rs2 = rs2;
            entry.opcode = 0x3b;
            entry.funct3 = 6;
            entry.circuit_flags = [_]bool{false} ** 14;
            entry.instruction_flags = [_]bool{false} ** 7;
            var cf = &entry.circuit_flags;
            cf[@intFromEnum(CircuitFlags.WriteLookupOutputToRD)] = true;
            setVirtualSequenceFlags(cf, vsr, is_first, is_compressed);
            var inf = &entry.instruction_flags;
            inf[@intFromEnum(InstructionFlags.LeftOperandIsRs1Value)] = true;
            inf[@intFromEnum(InstructionFlags.RightOperandIsRs2Value)] = true;
            if (rd != 0 and rd != 255) inf[@intFromEnum(InstructionFlags.IsRdNotZero)] = true;
            entry.lookup_table_index = 30; // VirtualChangeDivisorW
            entry.is_interleaved = true;
            entry.virtual_sequence_remaining = vsr;
            entry.is_first_in_sequence = is_first;
        },
        .VirtualPow2 => {
            entry.address = instr.address;
            entry.imm = 0;
            entry.rd = rd;
            entry.rs1 = rs1;
            entry.rs2 = 255;
            entry.opcode = 0x2B;
            entry.funct3 = 1;
            entry.circuit_flags = [_]bool{false} ** 14;
            entry.instruction_flags = [_]bool{false} ** 7;
            var cf = &entry.circuit_flags;
            cf[@intFromEnum(CircuitFlags.AddOperands)] = true;
            cf[@intFromEnum(CircuitFlags.WriteLookupOutputToRD)] = true;
            setVirtualSequenceFlags(cf, vsr, is_first, is_compressed);
            var inf = &entry.instruction_flags;
            inf[@intFromEnum(InstructionFlags.LeftOperandIsRs1Value)] = true;
            inf[@intFromEnum(InstructionFlags.RightOperandIsImm)] = true;
            if (rd != 0 and rd != 255) inf[@intFromEnum(InstructionFlags.IsRdNotZero)] = true;
            entry.lookup_table_index = 21; // Pow2
            entry.is_interleaved = false;
            entry.virtual_sequence_remaining = vsr;
            entry.is_first_in_sequence = is_first;
        },
        .VirtualShiftRightBitmask => {
            entry.address = instr.address;
            entry.imm = 0;
            entry.rd = rd;
            entry.rs1 = rs1;
            entry.rs2 = 255;
            entry.opcode = 0x2B;
            entry.funct3 = 2;
            entry.circuit_flags = [_]bool{false} ** 14;
            entry.instruction_flags = [_]bool{false} ** 7;
            var cf = &entry.circuit_flags;
            cf[@intFromEnum(CircuitFlags.AddOperands)] = true;
            cf[@intFromEnum(CircuitFlags.WriteLookupOutputToRD)] = true;
            setVirtualSequenceFlags(cf, vsr, is_first, is_compressed);
            var inf = &entry.instruction_flags;
            inf[@intFromEnum(InstructionFlags.LeftOperandIsRs1Value)] = true;
            inf[@intFromEnum(InstructionFlags.RightOperandIsImm)] = true;
            if (rd != 0 and rd != 255) inf[@intFromEnum(InstructionFlags.IsRdNotZero)] = true;
            entry.lookup_table_index = 23; // ShiftRightBitmask
            entry.is_interleaved = false;
            entry.virtual_sequence_remaining = vsr;
            entry.is_first_in_sequence = is_first;
        },
        .VirtualROTRI => {
            // VirtualROTRI: opcode 0x6B, funct3=0, interleaved(rs1, bitmask)
            // The imm from FormatVirtualRightShiftI IS already the bitmask
            entry.address = instr.address;
            entry.imm = imm; // imm is already the bitmask from FormatVirtualRightShiftI
            entry.rd = rd;
            entry.rs1 = rs1;
            entry.rs2 = 255;
            entry.opcode = 0x6B;
            entry.funct3 = 0;
            entry.circuit_flags = [_]bool{false} ** 14;
            entry.instruction_flags = [_]bool{false} ** 7;
            var cf = &entry.circuit_flags;
            cf[@intFromEnum(CircuitFlags.WriteLookupOutputToRD)] = true;
            setVirtualSequenceFlags(cf, vsr, is_first, is_compressed);
            var inf = &entry.instruction_flags;
            inf[@intFromEnum(InstructionFlags.LeftOperandIsRs1Value)] = true;
            inf[@intFromEnum(InstructionFlags.RightOperandIsImm)] = true;
            if (rd != 0 and rd != 255) inf[@intFromEnum(InstructionFlags.IsRdNotZero)] = true;
            entry.lookup_table_index = 27; // VirtualROTR
            entry.is_interleaved = true;
            entry.virtual_sequence_remaining = vsr;
            entry.is_first_in_sequence = is_first;
        },
        .VirtualROTRIW => {
            // VirtualROTRIW: opcode 0x6B, funct3=1, interleaved(rs1, bitmask)
            // The imm from FormatVirtualRightShiftI IS already the bitmask
            entry.address = instr.address;
            entry.imm = imm; // imm is already the bitmask from FormatVirtualRightShiftI
            entry.rd = rd;
            entry.rs1 = rs1;
            entry.rs2 = 255;
            entry.opcode = 0x6B;
            entry.funct3 = 1;
            entry.circuit_flags = [_]bool{false} ** 14;
            entry.instruction_flags = [_]bool{false} ** 7;
            var cf_w = &entry.circuit_flags;
            cf_w[@intFromEnum(CircuitFlags.WriteLookupOutputToRD)] = true;
            setVirtualSequenceFlags(cf_w, vsr, is_first, is_compressed);
            var inf_w = &entry.instruction_flags;
            inf_w[@intFromEnum(InstructionFlags.LeftOperandIsRs1Value)] = true;
            inf_w[@intFromEnum(InstructionFlags.RightOperandIsImm)] = true;
            if (rd != 0 and rd != 255) inf_w[@intFromEnum(InstructionFlags.IsRdNotZero)] = true;
            entry.lookup_table_index = 28; // VirtualROTRW
            entry.is_interleaved = true;
            entry.virtual_sequence_remaining = vsr;
            entry.is_first_in_sequence = is_first;
        },

        // =====================================================================
        // Standard RISC-V instructions — build a 32-bit word and delegate
        // =====================================================================

        // R-type (opcode 0x33): ADD, SUB, SLL, SLT, SLTU, XOR, SRL, SRA, OR, AND,
        //                       MUL, MULH, MULHSU, MULHU, DIV, DIVU, REM, REMU
        .ADD, .SUB, .SLL, .SLT, .SLTU, .XOR, .SRL, .SRA, .OR, .AND, .ANDN, .MUL, .MULH, .MULHSU, .MULHU, .DIV, .DIVU, .REM, .REMU => {
            const info = getRTypeEncoding(instr.variant);
            const word = buildRType(info.funct7, rs2, rs1, info.funct3, rd, 0x33);
            populateEntryFromInstruction(entry, word, instr.address);
            applyVirtualAndCompressedFlags(entry, rd, rs1, rs2, vsr, is_first, is_compressed);
        },

        // OP-32 R-type (opcode 0x3b): ADDW, SUBW, SLLW, SRLW, SRAW, MULW, DIVW, DIVUW, REMW, REMUW
        .ADDW, .SUBW, .SLLW, .SRLW, .SRAW, .MULW, .DIVW, .DIVUW, .REMW, .REMUW => {
            const info = getOp32RTypeEncoding(instr.variant);
            const word = buildRType(info.funct7, rs2, rs1, info.funct3, rd, 0x3b);
            populateEntryFromInstruction(entry, word, instr.address);
            applyVirtualAndCompressedFlags(entry, rd, rs1, rs2, vsr, is_first, is_compressed);
        },

        // I-type ALU (opcode 0x13): ADDI, SLTI, SLTIU, XORI, ORI, ANDI, SLLI, SRLI, SRAI
        .ADDI, .SLTI, .SLTIU, .XORI, .ORI, .ANDI, .SLLI, .SRLI, .SRAI => {
            const info = getITypeEncoding(instr.variant);
            const imm_u64: u64 = @bitCast(imm);
            const word = buildIType(imm_u64, rs1, info.funct3, rd, 0x13);
            populateEntryFromInstruction(entry, word, instr.address);
            // populateEntryFromInstruction decodes entry.imm from the encoded 12-bit
            // field, which truncates wide inline immediates (e.g. SHA-256 K[i]).
            // Restore the full u64 to keep bytecode consistent with the trace witness.
            entry.imm = imm;
            applyVirtualAndCompressedFlags(entry, rd, rs1, 255, vsr, is_first, is_compressed);
        },

        // OP-IMM-32 (opcode 0x1b): ADDIW, SLLIW, SRLIW, SRAIW
        .ADDIW, .SLLIW, .SRLIW, .SRAIW => {
            const info = getOpImm32Encoding(instr.variant);
            const imm_u64: u64 = @bitCast(imm);
            const word = buildIType(imm_u64, rs1, info.funct3, rd, 0x1b);
            populateEntryFromInstruction(entry, word, instr.address);
            entry.imm = imm;
            applyVirtualAndCompressedFlags(entry, rd, rs1, 255, vsr, is_first, is_compressed);
        },

        // Load (opcode 0x03): LB, LBU, LH, LHU, LW, LWU, LD
        .LB, .LBU, .LH, .LHU, .LW, .LWU, .LD => {
            const f3 = getLoadFunct3(instr.variant);
            const imm_u64: u64 = @bitCast(imm);
            const word = buildIType(imm_u64, rs1, f3, rd, 0x03);
            populateEntryFromInstruction(entry, word, instr.address);
            applyVirtualAndCompressedFlags(entry, rd, rs1, 255, vsr, is_first, is_compressed);
        },

        // Store (opcode 0x23): SB, SH, SW, SD
        .SB, .SH, .SW, .SD => {
            const f3 = getStoreFunct3(instr.variant);
            const word = buildSType(imm, rs2, rs1, f3, 0x23);
            populateEntryFromInstruction(entry, word, instr.address);
            applyVirtualAndCompressedFlags(entry, 255, rs1, rs2, vsr, is_first, is_compressed);
        },

        // Branch (opcode 0x63): BEQ, BNE, BLT, BGE, BLTU, BGEU
        .BEQ, .BNE, .BLT, .BGE, .BLTU, .BGEU => {
            const f3 = getBranchFunct3(instr.variant);
            const word = buildBType(imm, rs2, rs1, f3, 0x63);
            populateEntryFromInstruction(entry, word, instr.address);
            applyVirtualAndCompressedFlags(entry, 255, rs1, rs2, vsr, is_first, is_compressed);
        },

        // U-type: LUI (0x37), AUIPC (0x17)
        .LUI => {
            const imm_u64: u64 = @bitCast(imm);
            const word = buildUType(imm_u64, rd, 0x37);
            populateEntryFromInstruction(entry, word, instr.address);
            applyVirtualAndCompressedFlags(entry, rd, 255, 255, vsr, is_first, is_compressed);
        },
        .AUIPC => {
            const imm_u64: u64 = @bitCast(imm);
            const word = buildUType(imm_u64, rd, 0x17);
            populateEntryFromInstruction(entry, word, instr.address);
            applyVirtualAndCompressedFlags(entry, rd, 255, 255, vsr, is_first, is_compressed);
        },

        // J-type: JAL (0x6F)
        .JAL => {
            // Preprocessing already remapped rd=0 → rd=40 for JAL x0
            const raw_rd: u8 = if (rd == 40) 0 else rd; // undo remapping for instruction word
            const imm_u64: u64 = @bitCast(imm);
            const word = buildJType(imm_u64, raw_rd, 0x6F);
            populateEntryFromInstruction(entry, word, instr.address);
            // JAL x0 → vr40 remapping
            if (rd == 0 or rd == 40) {
                entry.rd = 40;
                entry.instruction_flags[@intFromEnum(InstructionFlags.IsRdNotZero)] = true;
            }
            applyVirtualAndCompressedFlags(entry, entry.rd, 255, 255, vsr, is_first, is_compressed);
        },

        // JALR (opcode 0x67)
        .JALR => {
            // Preprocessing already remapped rd=0 → rd=40 for JALR x0
            const raw_rd: u8 = if (rd == 40) 0 else rd; // undo remapping for instruction word
            const imm_u64: u64 = @bitCast(imm);
            const word = buildIType(imm_u64, rs1, 0, raw_rd, 0x67);
            populateEntryFromInstruction(entry, word, instr.address);
            // JALR x0 → vr40 remapping
            if (rd == 0 or rd == 40) {
                entry.rd = 40;
                entry.instruction_flags[@intFromEnum(InstructionFlags.IsRdNotZero)] = true;
            }
            applyVirtualAndCompressedFlags(entry, entry.rd, rs1, 255, vsr, is_first, is_compressed);
        },

        // System: FENCE, ECALL → NoOp-like (no lookup, IsNoop flag)
        .FENCE => {
            entry.address = instr.address;
            entry.imm = 0;
            entry.rd = 255;
            entry.rs1 = 255;
            entry.rs2 = 255;
            entry.opcode = 0x0F;
            entry.funct3 = 0;
            // Jolt's FENCE circuit_flags: only IsFirstInSequence and IsCompressed (both false for standalone)
            // Jolt's FENCE instruction_flags: all false (NO IsNoop, NO DoNotUpdateUnexpandedPC!)
            entry.circuit_flags = [_]bool{false} ** 14;
            entry.instruction_flags = [_]bool{false} ** 7;
            entry.lookup_table_index = 255;
            entry.is_interleaved = true;
            entry.virtual_sequence_remaining = vsr;
            entry.is_first_in_sequence = is_first;
            if (is_compressed) entry.circuit_flags[@intFromEnum(CircuitFlags.IsCompressed)] = true;
        },
        .ECALL => {
            entry.address = instr.address;
            entry.imm = 0;
            // Jolt's ECALL FormatI: rd=Some(rd), rs1=Some(rs1), rs2=None
            entry.rd = rd;
            entry.rs1 = rs1;
            entry.rs2 = 255;
            entry.opcode = 0x73;
            entry.funct3 = 0;
            entry.circuit_flags = [_]bool{false} ** 14;
            entry.instruction_flags = [_]bool{false} ** 7;
            entry.lookup_table_index = 255;
            entry.is_interleaved = true;
            entry.virtual_sequence_remaining = vsr;
            entry.is_first_in_sequence = is_first;
            if (is_compressed) entry.circuit_flags[@intFromEnum(CircuitFlags.IsCompressed)] = true;
        },

        // VirtualRev8W (0x5B funct3=0 in ELF, 0x7B internally): byte-swap each 32-bit half.
        // Jolt: AddOperands=true, WriteLookupOutputToRD=true, LeftOperandIsRs1Value=true,
        // lookup_table = VirtualRev8WTable (index 24 in pinned Jolt 997c1543).
        // We use synthetic opcode 0x7B internally to distinguish from VirtualSRLI which
        // uses 0x5B funct3=0 as its synthetic trace encoding.
        .VirtualRev8W => {
            entry.address = instr.address;
            entry.imm = 0;
            entry.rd = rd;
            entry.rs1 = rs1;
            entry.rs2 = 255;
            entry.opcode = 0x7B;
            entry.funct3 = 0;
            entry.circuit_flags = [_]bool{false} ** 14;
            entry.instruction_flags = [_]bool{false} ** 7;
            var cf = &entry.circuit_flags;
            cf[@intFromEnum(CircuitFlags.AddOperands)] = true;
            cf[@intFromEnum(CircuitFlags.WriteLookupOutputToRD)] = true;
            setVirtualSequenceFlags(cf, vsr, is_first, is_compressed);
            var inf = &entry.instruction_flags;
            inf[@intFromEnum(InstructionFlags.LeftOperandIsRs1Value)] = true;
            if (rd != 0 and rd != 255) inf[@intFromEnum(InstructionFlags.IsRdNotZero)] = true;
            entry.lookup_table_index = 24; // VirtualRev8W table
            entry.is_interleaved = false; // AddOperands set
            entry.virtual_sequence_remaining = vsr;
            entry.is_first_in_sequence = is_first;
        },
        // SDK NoOp-like (no lookup): VirtualHostIO, AdviceLB/H/W/D, VirtualAdviceLoad, VirtualAdviceLen
        .VirtualHostIO, .AdviceLB, .AdviceLH, .AdviceLW, .AdviceLD, .VirtualAdviceLoad, .VirtualAdviceLen => {
            // Jolt SDK instructions: NOP-like with real address, no flags, no lookup.
            // Matches Jolt's VirtualHostIO circuit_flags=[false; 14], instruction_flags=[false; 7].
            // NOTE: AdviceLB/H/W/D should be expanded to VirtualAdviceLoad+SLLI+SRAI in
            // preprocessing for full correctness; this is a fallback path.
            entry.address = instr.address;
            entry.imm = 0;
            entry.rd = rd;
            entry.rs1 = rs1;
            entry.rs2 = 255;
            entry.opcode = 0x5B;
            entry.funct3 = switch (instr.variant) {
                .VirtualHostIO => 2,
                .AdviceLB => 3,
                .AdviceLH => 4,
                .AdviceLW => 5,
                .AdviceLD => 6,
                .VirtualAdviceLen => 7,
                .VirtualAdviceLoad => 3,
                else => 2,
            };
            entry.circuit_flags = [_]bool{false} ** 14;
            entry.instruction_flags = [_]bool{false} ** 7;
            entry.lookup_table_index = 255;
            entry.is_interleaved = true;
            entry.virtual_sequence_remaining = null;
            entry.is_first_in_sequence = false;
        },
        .CSRRW, .CSRRS, .MRET => {
            entry.address = instr.address;
            entry.imm = 0;
            entry.rd = 255;
            entry.rs1 = 255;
            entry.rs2 = 255;
            entry.opcode = 0x73;
            entry.funct3 = 0;
            entry.circuit_flags = [_]bool{false} ** 14;
            entry.instruction_flags = [_]bool{false} ** 7;
            entry.instruction_flags[@intFromEnum(InstructionFlags.IsNoop)] = true;
            entry.circuit_flags[@intFromEnum(CircuitFlags.DoNotUpdateUnexpandedPC)] = true;
            entry.lookup_table_index = 255;
            entry.is_interleaved = true;
            entry.virtual_sequence_remaining = vsr;
            entry.is_first_in_sequence = is_first;
            if (is_compressed) entry.circuit_flags[@intFromEnum(CircuitFlags.IsCompressed)] = true;
        },

        // NoOp — should not reach here (filtered at call site), but handle defensively
        .NoOp => {
            entry.address = 0;
            entry.imm = 0;
            entry.rd = 255;
            entry.rs1 = 255;
            entry.rs2 = 255;
            entry.opcode = 0;
            entry.funct3 = 0;
            entry.circuit_flags = [_]bool{false} ** 14;
            entry.instruction_flags = [_]bool{false} ** 7;
            entry.circuit_flags[@intFromEnum(CircuitFlags.DoNotUpdateUnexpandedPC)] = true;
            entry.instruction_flags[@intFromEnum(InstructionFlags.IsNoop)] = true;
            entry.lookup_table_index = 255;
            entry.is_interleaved = true;
            entry.virtual_sequence_remaining = null;
            entry.is_first_in_sequence = false;
        },

        // UNIMPL — all flags false, no lookup
        .UNIMPL => {
            entry.address = 0; // UNIMPL normalizes to Default in Jolt, which has address=0
            entry.imm = 0;
            entry.rd = 255;
            entry.rs1 = 255;
            entry.rs2 = 255;
            entry.opcode = 0;
            entry.funct3 = 0;
            entry.circuit_flags = [_]bool{false} ** 14;
            entry.instruction_flags = [_]bool{false} ** 7;
            entry.lookup_table_index = 255;
            entry.is_interleaved = true;
            entry.virtual_sequence_remaining = null;
            entry.is_first_in_sequence = false;
        },
    }
}

/// Helper: set VirtualInstruction, DoNotUpdateUnexpandedPC, IsFirstInSequence, IsCompressed flags.
fn setVirtualSequenceFlags(cf: *[14]bool, vsr: ?u16, is_first: bool, is_compressed: bool) void {
    if (vsr != null) {
        cf[@intFromEnum(CircuitFlags.VirtualInstruction)] = true;
    }
    if (vsr) |v| {
        if (v != 0) {
            cf[@intFromEnum(CircuitFlags.DoNotUpdateUnexpandedPC)] = true;
        }
    }
    if (is_first) {
        cf[@intFromEnum(CircuitFlags.IsFirstInSequence)] = true;
    }
    if (is_compressed) {
        cf[@intFromEnum(CircuitFlags.IsCompressed)] = true;
    }
}

/// Helper: apply virtual sequence flags and is_compressed on top of an entry
/// already populated by populateEntryFromInstruction. Also fixes register
/// indices for virtual register values > 31 (which can't be encoded in a 5-bit
/// instruction word field).
fn applyVirtualAndCompressedFlags(
    entry: *BytecodeEntry,
    rd_full: u8,
    rs1_full: u8,
    rs2_full: u8,
    vsr: ?u16,
    is_first: bool,
    is_compressed: bool,
) void {
    // Fix register indices: populateEntryFromInstruction only sees the low 5 bits
    // from the instruction word. Virtual registers (32+) need the full value.
    if (rd_full != 255) {
        entry.rd = rd_full;
        // Also fix IsRdNotZero: the instruction word had rd&0x1F which truncates VRs to 0
        entry.instruction_flags[@intFromEnum(InstructionFlags.IsRdNotZero)] = (rd_full != 0);
    }
    if (rs1_full != 255) entry.rs1 = rs1_full;
    if (rs2_full != 255) entry.rs2 = rs2_full;

    // Set virtual sequence metadata
    entry.virtual_sequence_remaining = vsr;
    entry.is_first_in_sequence = is_first;
    if (vsr != null) {
        entry.circuit_flags[@intFromEnum(CircuitFlags.VirtualInstruction)] = true;
    }
    if (vsr) |v| {
        if (v != 0) {
            entry.circuit_flags[@intFromEnum(CircuitFlags.DoNotUpdateUnexpandedPC)] = true;
        }
    }
    if (is_first) {
        entry.circuit_flags[@intFromEnum(CircuitFlags.IsFirstInSequence)] = true;
    }
    if (is_compressed) {
        entry.circuit_flags[@intFromEnum(CircuitFlags.IsCompressed)] = true;
    }
}

// =========================================================================
// Instruction word builders — construct 32-bit RISC-V instruction words
// from fields. Used by populateEntryFromJoltInstruction.
// =========================================================================

fn buildRType(funct7: u7, rs2_val: u8, rs1_val: u8, funct3: u3, rd_val: u8, opcode: u7) u32 {
    return (@as(u32, funct7) << 25) |
        (@as(u32, rs2_val & 0x1F) << 20) |
        (@as(u32, rs1_val & 0x1F) << 15) |
        (@as(u32, funct3) << 12) |
        (@as(u32, rd_val & 0x1F) << 7) |
        @as(u32, opcode);
}

fn buildIType(imm_u64: u64, rs1_val: u8, funct3: u3, rd_val: u8, opcode: u7) u32 {
    const imm12: u32 = @truncate(imm_u64 & 0xFFF);
    return (imm12 << 20) |
        (@as(u32, rs1_val & 0x1F) << 15) |
        (@as(u32, funct3) << 12) |
        (@as(u32, rd_val & 0x1F) << 7) |
        @as(u32, opcode);
}

fn buildSType(imm_val: i64, rs2_val: u8, rs1_val: u8, funct3: u3, opcode: u7) u32 {
    const imm_bits: u32 = @bitCast(@as(i32, @truncate(imm_val)));
    return ((imm_bits >> 5) & 0x7F) << 25 |
        (@as(u32, rs2_val & 0x1F) << 20) |
        (@as(u32, rs1_val & 0x1F) << 15) |
        (@as(u32, funct3) << 12) |
        ((imm_bits & 0x1F) << 7) |
        @as(u32, opcode);
}

fn buildBType(imm_val: i64, rs2_val: u8, rs1_val: u8, funct3: u3, opcode: u7) u32 {
    // B-type immediate encoding: imm[12|10:5] in bits[31:25], imm[4:1|11] in bits[11:7]
    const imm_bits: u32 = @bitCast(@as(i32, @truncate(imm_val)));
    const bit12 = (imm_bits >> 12) & 1;
    const bits10_5 = (imm_bits >> 5) & 0x3F;
    const bits4_1 = (imm_bits >> 1) & 0xF;
    const bit11 = (imm_bits >> 11) & 1;
    return (bit12 << 31) |
        (bits10_5 << 25) |
        (@as(u32, rs2_val & 0x1F) << 20) |
        (@as(u32, rs1_val & 0x1F) << 15) |
        (@as(u32, funct3) << 12) |
        (bits4_1 << 8) |
        (bit11 << 7) |
        @as(u32, opcode);
}

fn buildUType(imm_u64: u64, rd_val: u8, opcode: u7) u32 {
    // U-type: imm[31:12] in bits[31:12]
    const imm_bits: u32 = @truncate(imm_u64);
    return (imm_bits & 0xFFFFF000) |
        (@as(u32, rd_val & 0x1F) << 7) |
        @as(u32, opcode);
}

fn buildJType(imm_u64: u64, rd_val: u8, opcode: u7) u32 {
    // J-type immediate encoding: imm[20|10:1|11|19:12]
    const imm_bits: u32 = @truncate(imm_u64);
    const bit20 = (imm_bits >> 20) & 1;
    const bits10_1 = (imm_bits >> 1) & 0x3FF;
    const bit11 = (imm_bits >> 11) & 1;
    const bits19_12 = (imm_bits >> 12) & 0xFF;
    return (bit20 << 31) |
        (bits10_1 << 21) |
        (bit11 << 20) |
        (bits19_12 << 12) |
        (@as(u32, rd_val & 0x1F) << 7) |
        @as(u32, opcode);
}

// =========================================================================
// Variant-to-encoding mapping tables
// =========================================================================

const RTypeInfo = struct { funct3: u3, funct7: u7 };

fn getRTypeEncoding(variant: preprocessing.JoltInstruction.InstructionVariant) RTypeInfo {
    return switch (variant) {
        .ADD => .{ .funct3 = 0, .funct7 = 0x00 },
        .SUB => .{ .funct3 = 0, .funct7 = 0x20 },
        .SLL => .{ .funct3 = 1, .funct7 = 0x00 },
        .SLT => .{ .funct3 = 2, .funct7 = 0x00 },
        .SLTU => .{ .funct3 = 3, .funct7 = 0x00 },
        .XOR => .{ .funct3 = 4, .funct7 = 0x00 },
        .SRL => .{ .funct3 = 5, .funct7 = 0x00 },
        .SRA => .{ .funct3 = 5, .funct7 = 0x20 },
        .OR => .{ .funct3 = 6, .funct7 = 0x00 },
        .AND => .{ .funct3 = 7, .funct7 = 0x00 },
        .ANDN => .{ .funct3 = 7, .funct7 = 0x20 },
        .MUL => .{ .funct3 = 0, .funct7 = 0x01 },
        .MULH => .{ .funct3 = 1, .funct7 = 0x01 },
        .MULHSU => .{ .funct3 = 2, .funct7 = 0x01 },
        .MULHU => .{ .funct3 = 3, .funct7 = 0x01 },
        .DIV => .{ .funct3 = 4, .funct7 = 0x01 },
        .DIVU => .{ .funct3 = 5, .funct7 = 0x01 },
        .REM => .{ .funct3 = 6, .funct7 = 0x01 },
        .REMU => .{ .funct3 = 7, .funct7 = 0x01 },
        else => unreachable,
    };
}

fn getOp32RTypeEncoding(variant: preprocessing.JoltInstruction.InstructionVariant) RTypeInfo {
    return switch (variant) {
        .ADDW => .{ .funct3 = 0, .funct7 = 0x00 },
        .SUBW => .{ .funct3 = 0, .funct7 = 0x20 },
        .SLLW => .{ .funct3 = 1, .funct7 = 0x00 },
        .SRLW => .{ .funct3 = 5, .funct7 = 0x00 },
        .SRAW => .{ .funct3 = 5, .funct7 = 0x20 },
        .MULW => .{ .funct3 = 0, .funct7 = 0x01 },
        .DIVW => .{ .funct3 = 4, .funct7 = 0x01 },
        .DIVUW => .{ .funct3 = 5, .funct7 = 0x01 },
        .REMW => .{ .funct3 = 6, .funct7 = 0x01 },
        .REMUW => .{ .funct3 = 7, .funct7 = 0x01 },
        else => unreachable,
    };
}

fn getITypeEncoding(variant: preprocessing.JoltInstruction.InstructionVariant) struct { funct3: u3 } {
    return .{
        .funct3 = switch (variant) {
            .ADDI => 0,
            .SLTI => 2,
            .SLTIU => 3,
            .XORI => 4,
            .ORI => 6,
            .ANDI => 7,
            .SLLI => 1,
            .SRLI => 5, // funct7 encoded in imm[11:5]
            .SRAI => 5, // funct7 encoded in imm[11:5] (bit 30 set)
            else => unreachable,
        },
    };
}

fn getOpImm32Encoding(variant: preprocessing.JoltInstruction.InstructionVariant) struct { funct3: u3 } {
    return .{ .funct3 = switch (variant) {
        .ADDIW => 0,
        .SLLIW => 1,
        .SRLIW => 5,
        .SRAIW => 5,
        else => unreachable,
    } };
}

fn getLoadFunct3(variant: preprocessing.JoltInstruction.InstructionVariant) u3 {
    return switch (variant) {
        .LB => 0,
        .LH => 1,
        .LW => 2,
        .LD => 3,
        .LBU => 4,
        .LHU => 5,
        .LWU => 6,
        else => unreachable,
    };
}

fn getStoreFunct3(variant: preprocessing.JoltInstruction.InstructionVariant) u3 {
    return switch (variant) {
        .SB => 0,
        .SH => 1,
        .SW => 2,
        .SD => 3,
        else => unreachable,
    };
}

fn getBranchFunct3(variant: preprocessing.JoltInstruction.InstructionVariant) u3 {
    return switch (variant) {
        .BEQ => 0,
        .BNE => 1,
        .BLT => 4,
        .BGE => 5,
        .BLTU => 6,
        .BGEU => 7,
        else => unreachable,
    };
}

/// Build bytecode entry table from static ELF bytecode + execution trace overlay.
///
/// Phase 1: Populate ALL entries from the static ELF code bytes. This ensures
/// every instruction in the program (including unexecuted ones) has correct
/// properties, matching what the Jolt verifier computes from its bytecode array.
///
/// Phase 2: Populate termination store entries (LUI+ADDI+SB) at their own
/// bytecode indices (termination_base_pc, +1, +2). Each gets a separate entry
/// with per-instruction flags, matching Jolt's termination_entry_virtual/anchor.
///
/// pc_map converts ELF addresses to bytecode array indices.
pub fn buildBytecodeEntries(
    allocator: Allocator,
    trace: *const tracer.ExecutionTrace,
    bytecode_K: usize,
    pc_map: *const BytecodePCMapper,
    program_code_bytes: ?[]const u8,
    code_base_address: u64,
    termination_address: u64,
    text_size: usize,
    bytecode_preprocessing: ?*const @import("../preprocessing.zig").BytecodePreprocessing,
) ![]BytecodeEntry {
    _ = trace;
    const entries = try allocator.alloc(BytecodeEntry, bytecode_K);

    // Initialize all entries as NoOps matching Jolt's Instruction::NoOp flags:
    //   circuit_flags[DoNotUpdateUnexpandedPC] = true
    //   instruction_flags[IsNoop] = true
    // This is critical for BytecodeReadRaf correctness: NoOp cycles in the R1CS
    // witness have FlagDoNotUpdateUnexpandedPC=1 and FlagIsNoop=1, and these must
    // match the bytecode entry flags at k=0 (and any unused padding entries).
    for (0..bytecode_K) |k| {
        var cf = [_]bool{false} ** 14;
        cf[@intFromEnum(CircuitFlags.DoNotUpdateUnexpandedPC)] = true;
        var inf = [_]bool{false} ** 7;
        inf[@intFromEnum(InstructionFlags.IsNoop)] = true;
        entries[k] = BytecodeEntry{
            .address = 0,
            .imm = 0,
            // Use sentinel 255 for rd/rs1/rs2 so that noop entries contribute ZERO
            // to Stages 4 and 5 val polynomials (which use eq(entry.rd, r_register)).
            // With rd=255, the check `entry.rd < REGISTER_COUNT` (128) fails → zero.
            // This matches Jolt's original behavior where Instruction::NoOp has
            // operands.rd = None → map_or(F::zero(), ...) = zero.
            .rd = 255,
            .rs1 = 255,
            .rs2 = 255,
            .circuit_flags = cf,
            .instruction_flags = inf,
            .lookup_table_index = 255,
            // NoOp has no AddOperands/SubtractOperands/MultiplyOperands/Advice flags,
            // so is_interleaved_operands() = true in Jolt. This means !is_interleaved = false,
            // so noops do NOT contribute to the identity-path (InstructionRafFlag) sum.
            // This matches Stage 5's trace-based computation where opcode=0x00 → not identity path.
            .is_interleaved = true,
            .virtual_sequence_remaining = null,
            .is_first_in_sequence = false,
            .opcode = 0, // NoOp has no real opcode
            .funct3 = 0,
        };
    }

    // ================================================================
    // Phase 1: Use preprocessing bytecode when available
    // ================================================================
    if (bytecode_preprocessing) |prep| {
        for (0..@min(bytecode_K, prep.bytecode.items.len)) |k| {
            const prep_instr = prep.bytecode.items[k];
            if (prep_instr.variant == .NoOp) continue;
            populateEntryFromJoltInstruction(&entries[k], prep_instr);
        }
    } else if (program_code_bytes) |code_bytes| {
        // Using raw ELF path
        const decode_limit = @min(text_size, code_bytes.len);
        var offset: usize = 0;
        while (offset < decode_limit) {
            const addr = code_base_address + offset;

            // Check if compressed (RVC)
            if (offset + 2 > decode_limit) break;
            const first_halfword: u16 = std.mem.readInt(u16, code_bytes[offset..][0..2], .little);
            const is_compressed = (first_halfword & 0x3) != 0x3;

            var instr_word: u32 = undefined;
            var instr_size: usize = undefined;

            if (first_halfword == 0) {
                // Zero halfword in a code gap — skip 2 bytes, leave entry as NoOp
                offset += 2;
                continue;
            }
            if (is_compressed) {
                // 16-bit compressed instruction - expand it
                instr_word = instruction_mod.uncompressInstruction(@as(u32, first_halfword), .Bit64);
                instr_size = 2;
            } else {
                // 32-bit instruction
                if (offset + 4 > code_bytes.len) break;
                instr_word = std.mem.readInt(u32, code_bytes[offset..][0..4], .little);
                instr_size = 4;
            }

            // Map ELF address to bytecode array index
            const k = pc_map.getPC(addr, 0);
            if (k > 0 and k < bytecode_K) {
                // Detect SLLI/SLLIW instructions and decompose them to virtual instruction entries,
                // matching the preprocessing decomposition in preprocessing.zig.
                // Without this, the bytecode entries would have SLLI flags (no lookup table)
                // while the execution trace uses VirtualMULI (lookup table = RangeCheck).
                const raw_opcode: u8 = @truncate(instr_word & 0x7F);
                const raw_funct3: u3 = @truncate((instr_word >> 12) & 0x7);
                const raw_rd: u8 = @truncate((instr_word >> 7) & 0x1F);
                const raw_rs1: u8 = @truncate((instr_word >> 15) & 0x1F);

                if (raw_opcode == 0x13 and raw_funct3 == 1) {
                    // SLLI → VirtualMULI (single entry, standalone virtual sequence)
                    // In Jolt, standalone SLLI becomes a 1-instruction virtual sequence:
                    //   VirtualMULI with vsr=Some(0), is_first_in_sequence=true
                    // VirtualInstruction=true (vsr.is_some()), DoNotUpdateUnexpandedPC=false (vsr=0)
                    const shamt: u6 = @truncate((instr_word >> 20) & 0x3F);
                    populateVirtualMULIEntry(&entries[k], raw_rd, raw_rs1, addr, shamt, 0, true);
                    // is_compressed: only set on last (=only) instruction in sequence
                    if (is_compressed) {
                        entries[k].circuit_flags[@intFromEnum(CircuitFlags.IsCompressed)] = true;
                    }
                } else if (raw_opcode == 0x1b and raw_funct3 == 1) {
                    // SLLIW → VirtualMULI + VirtualSignExtendWord (2 entries)
                    // pc_map for 2-entry sequences has max_inline_seq=1.
                    // getPC(addr, 0) returns base_pc+1 (the VirtualSignExtendWord entry).
                    // So k = base_pc+1. The VirtualMULI entry is at k-1.
                    const shamt: u6 = @truncate((instr_word >> 20) & 0x1F); // SLLIW uses 5-bit shamt
                    // Populate VirtualSignExtendWord at k (= base_pc+1, last in sequence)
                    // is_compressed: only on last instruction in sequence (per Jolt finalize)
                    populateVirtualSignExtendWordEntry(&entries[k], raw_rd, addr, is_compressed);
                    // Populate VirtualMULI at k-1 (= base_pc, first in sequence)
                    // vsr=Some(1): 1 instruction remaining after this
                    // is_first_in_sequence=true: it's the first entry
                    // is_compressed: only on last entry (k), not propagated to k-1
                    if (k >= 1) {
                        populateVirtualMULIEntry(&entries[k - 1], raw_rd, raw_rs1, addr, shamt, 1, true);
                    }
                } else if (raw_opcode == 0x13 and raw_funct3 == 5 and (instr_word >> 30) & 1 == 0) {
                    // SRLI → VirtualSRLI (single entry, standalone virtual sequence)
                    // Like SLLI → VirtualMULI, standalone SRLI becomes a 1-instruction virtual sequence:
                    //   VirtualSRLI with vsr=null (standalone), is_first_in_sequence=false
                    // Wait - actually, looking at Jolt's SRLI inline_sequence:
                    //   It returns [VirtualSRLI(rd, rs1, bitmask)] with finalize() setting vsr=Some(0), first=true
                    const shamt: u7 = @truncate((instr_word >> 20) & 0x3F);
                    const ones: u128 = (@as(u128, 1) << @intCast(64 - @as(u8, shamt))) - 1;
                    const bitmask: u64 = @truncate(ones << shamt);
                    populateVirtualSRLIEntry(&entries[k], raw_rd, raw_rs1, addr, bitmask, 0, true);
                    if (is_compressed) {
                        entries[k].circuit_flags[@intFromEnum(CircuitFlags.IsCompressed)] = true;
                    }
                } else if (raw_opcode == 0x1b and raw_funct3 == 5 and (instr_word >> 30) & 1 == 0) {
                    // SRLIW → VirtualMULI + VirtualSRLI + VirtualSignExtendWord (3 entries)
                    // pc_map for 3-entry sequences has max_inline_seq=2.
                    // getPC(addr, 0) returns base_pc+2 (the VirtualSignExtendWord entry).
                    // So k = base_pc+2. VirtualSRLI is at k-1, VirtualMULI is at k-2.
                    const shamt: u7 = @truncate((instr_word >> 20) & 0x1F); // SRLIW uses 5-bit shamt
                    const total_shift: u7 = shamt + 32;
                    const ones: u128 = (@as(u128, 1) << @intCast(64 - @as(u8, total_shift))) - 1;
                    const bitmask: u64 = @truncate(ones << total_shift);
                    // Entry at k: VirtualSignExtendWord (last in sequence, vsr=0)
                    populateVirtualSignExtendWordEntry(&entries[k], raw_rd, addr, is_compressed);
                    // Entry at k-1: VirtualSRLI (middle, vsr=1)
                    if (k >= 1) {
                        populateVirtualSRLIEntry(&entries[k - 1], raw_rd, 32, addr, bitmask, 1, false);
                    }
                    // Entry at k-2: VirtualMULI (first in sequence, vsr=2)
                    if (k >= 2) {
                        // SLLI by 32: shamt=32 stored as u6 (fits since 32 < 64)
                        populateVirtualMULIEntry(&entries[k - 2], 32, raw_rs1, addr, 32, 2, true);
                    }
                } else if (raw_opcode == 0x3b and raw_funct3 == 7 and (instr_word >> 25) == 0x01) {
                    // REMUW → 12-instruction inline sequence (matching Jolt's decomposition)
                    // pc_map for 12-entry sequences has max_inline_seq=11.
                    // getPC(addr, 0) returns base_pc+11 (the VirtualSignExtendWord entry).
                    // So k = base_pc+11. Entries go at k-11 through k.
                    const raw_rs2: u8 = @truncate((instr_word >> 20) & 0x1F);
                    // Virtual registers matching preprocessing.zig
                    const a2: u8 = 32;
                    const a3: u8 = 33;
                    const t0: u8 = 34;
                    const t1: u8 = 35;
                    const t2: u8 = 36;
                    const t3: u8 = 37;
                    const t4: u8 = 38;

                    // Step 1 (k-11): VirtualAdvice(a2) → quotient (vsr=11, first)
                    if (k >= 11) {
                        populateVirtualAdviceEntry(&entries[k - 11], a2, addr, 11, true);
                    }
                    // Step 2 (k-10): VirtualAdvice(a3) → remainder (vsr=10)
                    if (k >= 10) {
                        populateVirtualAdviceEntry(&entries[k - 10], a3, addr, 10, false);
                    }
                    // Step 3 (k-9): VirtualZeroExtendWord(t3, a2) → zero-extend quotient (vsr=9)
                    if (k >= 9) {
                        populateVirtualZeroExtendWordEntry(&entries[k - 9], t3, a2, addr, 9, false);
                    }
                    // Step 4 (k-8): VirtualZeroExtendWord(t1, rs1) → zero-extend dividend (vsr=8)
                    if (k >= 8) {
                        populateVirtualZeroExtendWordEntry(&entries[k - 8], t1, raw_rs1, addr, 8, false);
                    }
                    // Step 5 (k-7): VirtualZeroExtendWord(t2, rs2) → zero-extend divisor (vsr=7)
                    if (k >= 7) {
                        populateVirtualZeroExtendWordEntry(&entries[k - 7], t2, raw_rs2, addr, 7, false);
                    }
                    // Step 6 (k-6): MUL(t0, t3, t2) → quotient * divisor (vsr=6)
                    // MUL is a regular R-type with opcode=0x33, funct3=0, funct7=1
                    if (k >= 6) {
                        const mul_instr: u32 = (0x01 << 25) | (@as(u32, t2 & 0x1F) << 20) | (@as(u32, t3 & 0x1F) << 15) | (0 << 12) | (@as(u32, if (t0 == 0) @as(u8, 0) else (t0 & 0x1F)) << 7) | 0x33;
                        populateEntryFromInstruction(&entries[k - 6], mul_instr, addr);
                        // Override register indices with full virtual register values
                        // (populateEntryFromInstruction truncates to 5 bits via instruction word encoding)
                        entries[k - 6].rd = t0;
                        entries[k - 6].rs1 = t3;
                        entries[k - 6].rs2 = t2;
                        entries[k - 6].circuit_flags[@intFromEnum(CircuitFlags.VirtualInstruction)] = true;
                        entries[k - 6].circuit_flags[@intFromEnum(CircuitFlags.DoNotUpdateUnexpandedPC)] = true;
                        entries[k - 6].virtual_sequence_remaining = 6;
                        entries[k - 6].is_first_in_sequence = false;
                    }
                    // Step 7 (k-5): VirtualZeroExtendWord(t4, t0) → mask to 32 bits (vsr=5)
                    if (k >= 5) {
                        populateVirtualZeroExtendWordEntry(&entries[k - 5], t4, t0, addr, 5, false);
                    }
                    // Step 8 (k-4): VirtualAssertEQ(t4, t0) → assert no overflow (vsr=4)
                    if (k >= 4) {
                        populateVirtualAssertEQEntry(&entries[k - 4], t4, t0, addr, 4, false);
                    }
                    // Step 9 (k-3): ADD(t0, t0, a3) → add remainder (vsr=3)
                    // ADD is a regular R-type with opcode=0x33, funct3=0, funct7=0
                    if (k >= 3) {
                        const add_instr: u32 = (@as(u32, a3 & 0x1F) << 20) | (@as(u32, t0 & 0x1F) << 15) | (0 << 12) | (@as(u32, if (t0 == 0) @as(u8, 0) else (t0 & 0x1F)) << 7) | 0x33;
                        populateEntryFromInstruction(&entries[k - 3], add_instr, addr);
                        // Override register indices with full virtual register values
                        // (populateEntryFromInstruction truncates to 5 bits via instruction word encoding)
                        entries[k - 3].rd = t0;
                        entries[k - 3].rs1 = t0;
                        entries[k - 3].rs2 = a3;
                        entries[k - 3].circuit_flags[@intFromEnum(CircuitFlags.VirtualInstruction)] = true;
                        entries[k - 3].circuit_flags[@intFromEnum(CircuitFlags.DoNotUpdateUnexpandedPC)] = true;
                        entries[k - 3].virtual_sequence_remaining = 3;
                        entries[k - 3].is_first_in_sequence = false;
                    }
                    // Step 10 (k-2): VirtualAssertEQ(t0, t1) → assert dividend = q*d + r (vsr=2)
                    if (k >= 2) {
                        populateVirtualAssertEQEntry(&entries[k - 2], t0, t1, addr, 2, false);
                    }
                    // Step 11 (k-1): VirtualAssertValidUnsignedRemainder(a3, t2) → r < d (vsr=1)
                    if (k >= 1) {
                        populateVirtualAssertValidUnsignedRemainderEntry(&entries[k - 1], a3, t2, addr, 1, false);
                    }
                    // Step 12 (k): VirtualSignExtendWord(rd, a3) → sign-extend result (vsr=0, last)
                    populateVirtualSignExtendWordEntry(&entries[k], raw_rd, addr, is_compressed);
                    // Fix rs1: VirtualSignExtendWord reads from a3
                    entries[k].rs1 = a3;
                } else if (raw_opcode == 0x3b and (raw_funct3 == 6 or raw_funct3 == 4) and (instr_word >> 25) == 0x01) {
                    // REMW (funct3=6) or DIVW (funct3=4) → 21-instruction inline sequence
                    // pc_map for 21-entry sequences has max_inline_seq=20.
                    // getPC(addr, 0) returns base_pc+20 (the VirtualSignExtendWord entry).
                    // So k = base_pc+20. Entries go at k-20 through k.
                    const raw_rs2: u8 = @truncate((instr_word >> 20) & 0x1F);
                    // Virtual registers matching preprocessing.zig
                    const a2: u8 = 32; // quotient
                    const a3: u8 = 33; // |remainder|
                    const t0: u8 = 34; // adjusted divisor
                    const t1: u8 = 35; // temporary
                    const t2: u8 = 36; // temporary
                    const t3: u8 = 37; // signed remainder
                    const t4: u8 = 38; // sign-extended dividend

                    // Step 1 (k-20): VirtualAdvice(a2) → quotient (vsr=20, first)
                    if (k >= 20) {
                        populateVirtualAdviceEntry(&entries[k - 20], a2, addr, 20, true);
                    }
                    // Step 2 (k-19): VirtualAdvice(a3) → |remainder| (vsr=19)
                    if (k >= 19) {
                        populateVirtualAdviceEntry(&entries[k - 19], a3, addr, 19, false);
                    }
                    // Step 3 (k-18): VirtualSignExtendWord(t4, rs1) → sign-extend dividend (vsr=18)
                    if (k >= 18) {
                        populateVirtualSignExtendWordEntryWithParams(&entries[k - 18], t4, raw_rs1, addr, 18, false);
                    }
                    // Step 4 (k-17): VirtualSignExtendWord(t3, rs2) → sign-extend divisor (vsr=17)
                    if (k >= 17) {
                        populateVirtualSignExtendWordEntryWithParams(&entries[k - 17], t3, raw_rs2, addr, 17, false);
                    }
                    // Step 5 (k-16): VirtualAssertValidDiv0(t3, a2) → handle div-by-zero (vsr=16)
                    if (k >= 16) {
                        populateVirtualAssertValidDiv0Entry(&entries[k - 16], t3, a2, addr, 16, false);
                    }
                    // Step 6 (k-15): VirtualChangeDivisorW(t0, t4, t3) → handle overflow (vsr=15)
                    if (k >= 15) {
                        populateVirtualChangeDivisorWEntry(&entries[k - 15], t0, t4, t3, addr, 15, false);
                    }
                    // Step 7 (k-14): VirtualSignExtendWord(t1, a2) → sign-extend quotient (vsr=14)
                    if (k >= 14) {
                        populateVirtualSignExtendWordEntryWithParams(&entries[k - 14], t1, a2, addr, 14, false);
                    }
                    // Step 8 (k-13): VirtualAssertEQ(t1, a2) → assert quotient fits 32 bits (vsr=13)
                    if (k >= 13) {
                        populateVirtualAssertEQEntry(&entries[k - 13], t1, a2, addr, 13, false);
                    }
                    // VirtualSRAI bitmask for shift=31: ((1<<33)-1) << 31 = 0xFFFFFFFF80000000
                    const srai_bitmask: u64 = blk: {
                        const shift_amt: u7 = 31;
                        const ones: u128 = (@as(u128, 1) << @intCast(64 - @as(u8, shift_amt))) - 1;
                        break :blk @truncate(ones << shift_amt);
                    };
                    // Step 9 (k-12): VirtualSRAI(t2, a3, bitmask) → sign bit of |remainder| (vsr=12)
                    if (k >= 12) {
                        populateVirtualSRAIEntry(&entries[k - 12], t2, a3, addr, srai_bitmask, 12, false);
                    }
                    // Step 10 (k-11): VirtualAssertEQ(t2, 0) → assert non-negative (vsr=11)
                    if (k >= 11) {
                        populateVirtualAssertEQEntry(&entries[k - 11], t2, 0, addr, 11, false);
                    }
                    // Step 11 (k-10): VirtualSRAI(t2, t4, bitmask) → sign bit of dividend (vsr=10)
                    if (k >= 10) {
                        populateVirtualSRAIEntry(&entries[k - 10], t2, t4, addr, srai_bitmask, 10, false);
                    }
                    // Step 12 (k-9): XOR(t3, a3, t2) → XOR |remainder| with sign mask (vsr=9)
                    if (k >= 9) {
                        populateVirtualRTypeEntry(&entries[k - 9], t3, a3, t2, addr, 9, false, 0x33, 4, 0);
                    }
                    // Step 13 (k-8): SUB(t3, t3, t2) → sign-corrected remainder (vsr=8)
                    if (k >= 8) {
                        populateVirtualRTypeEntry(&entries[k - 8], t3, t3, t2, addr, 8, false, 0x33, 0, 0x20);
                    }
                    // Step 14 (k-7): MUL(t1, a2, t0) → quotient × adjusted_divisor (vsr=7)
                    if (k >= 7) {
                        populateVirtualRTypeEntry(&entries[k - 7], t1, a2, t0, addr, 7, false, 0x33, 0, 0x01);
                    }
                    // Step 15 (k-6): ADD(t1, t1, t3) → + remainder (vsr=6)
                    if (k >= 6) {
                        populateVirtualRTypeEntry(&entries[k - 6], t1, t1, t3, addr, 6, false, 0x33, 0, 0);
                    }
                    // Step 16 (k-5): VirtualAssertEQ(t1, t4) → assert dividend = q*d + r (vsr=5)
                    if (k >= 5) {
                        populateVirtualAssertEQEntry(&entries[k - 5], t1, t4, addr, 5, false);
                    }
                    // Step 17 (k-4): VirtualSRAI(t2, t0, bitmask) → sign bit of adjusted divisor (vsr=4)
                    if (k >= 4) {
                        populateVirtualSRAIEntry(&entries[k - 4], t2, t0, addr, srai_bitmask, 4, false);
                    }
                    // Step 18 (k-3): XOR(t1, t0, t2) → (vsr=3)
                    if (k >= 3) {
                        populateVirtualRTypeEntry(&entries[k - 3], t1, t0, t2, addr, 3, false, 0x33, 4, 0);
                    }
                    // Step 19 (k-2): SUB(t1, t1, t2) → abs(divisor) (vsr=2)
                    if (k >= 2) {
                        populateVirtualRTypeEntry(&entries[k - 2], t1, t1, t2, addr, 2, false, 0x33, 0, 0x20);
                    }
                    // Step 20 (k-1): VirtualAssertValidUnsignedRemainder(a3, t1) → |r| < |d| (vsr=1)
                    if (k >= 1) {
                        populateVirtualAssertValidUnsignedRemainderEntry(&entries[k - 1], a3, t1, addr, 1, false);
                    }
                    // Step 21 (k): VirtualSignExtendWord(rd, output) → sign-extend result (vsr=0, last)
                    // REMW: output = t3 (signed remainder), DIVW: output = a2 (quotient)
                    const output_reg = if (raw_funct3 == 6) t3 else a2;
                    populateVirtualSignExtendWordEntry(&entries[k], raw_rd, addr, is_compressed);
                    entries[k].rs1 = output_reg;
                } else if (raw_opcode == 0x13 and raw_funct3 == 5 and (instr_word >> 30) & 1 == 1) {
                    // SRAI → VirtualSRAI (single entry, standalone virtual sequence)
                    const shamt: u7 = @truncate((instr_word >> 20) & 0x3F);
                    const ones: u128 = (@as(u128, 1) << @intCast(64 - @as(u8, shamt))) - 1;
                    const bitmask: u64 = @truncate(ones << shamt);
                    populateVirtualSRAIEntry(&entries[k], raw_rd, raw_rs1, addr, bitmask, 0, true);
                    if (is_compressed) {
                        entries[k].circuit_flags[@intFromEnum(CircuitFlags.IsCompressed)] = true;
                    }
                } else if (raw_opcode == 0x33 and raw_funct3 == 1 and (instr_word >> 25) == 0) {
                    // SLL → VirtualPow2 + MUL (2 entries)
                    // k = base_pc + 1: MUL at k (last, vsr=0), VirtualPow2 at k-1 (first, vsr=1)
                    const raw_rs2: u8 = @truncate((instr_word >> 20) & 0x1F);
                    const v0: u8 = 40;
                    // k: MUL(rd, rs1, v0) — last step (vsr=0)
                    populateVirtualRTypeEntry(&entries[k], raw_rd, raw_rs1, v0, addr, 0, false, 0x33, 0, 0x01);
                    entries[k].circuit_flags[@intFromEnum(CircuitFlags.VirtualInstruction)] = true;
                    if (is_compressed) entries[k].circuit_flags[@intFromEnum(CircuitFlags.IsCompressed)] = true;
                    // k-1: VirtualPow2(v0, rs2, 0) — first step (vsr=1)
                    if (k >= 1) {
                        populateVirtualPow2Entry(&entries[k - 1], v0, raw_rs2, addr, 1, true);
                    }
                } else if (raw_opcode == 0x33 and raw_funct3 == 5 and (instr_word >> 25) == 0) {
                    // SRL → VirtualShiftRightBitmask + VirtualSRL (2 entries)
                    const raw_rs2: u8 = @truncate((instr_word >> 20) & 0x1F);
                    const v0: u8 = 40;
                    // k: VirtualSRL(rd, rs1, v0) — last (vsr=0)
                    populateVirtualSRLEntry(&entries[k], raw_rd, raw_rs1, v0, addr, 0, false);
                    entries[k].circuit_flags[@intFromEnum(CircuitFlags.VirtualInstruction)] = true;
                    if (is_compressed) entries[k].circuit_flags[@intFromEnum(CircuitFlags.IsCompressed)] = true;
                    // k-1: VirtualShiftRightBitmask(v0, rs2, 0) — first (vsr=1)
                    if (k >= 1) {
                        populateVirtualShiftRightBitmaskEntry(&entries[k - 1], v0, raw_rs2, addr, 1, true);
                    }
                } else if (raw_opcode == 0x33 and raw_funct3 == 5 and (instr_word >> 25) == 0x20) {
                    // SRA → VirtualShiftRightBitmask + VirtualSRA (2 entries)
                    const raw_rs2: u8 = @truncate((instr_word >> 20) & 0x1F);
                    const v0: u8 = 40;
                    // k: VirtualSRA(rd, rs1, v0) — last (vsr=0)
                    populateVirtualSRAEntry(&entries[k], raw_rd, raw_rs1, v0, addr, 0, false);
                    entries[k].circuit_flags[@intFromEnum(CircuitFlags.VirtualInstruction)] = true;
                    if (is_compressed) entries[k].circuit_flags[@intFromEnum(CircuitFlags.IsCompressed)] = true;
                    // k-1: VirtualShiftRightBitmask(v0, rs2, 0) — first (vsr=1)
                    if (k >= 1) {
                        populateVirtualShiftRightBitmaskEntry(&entries[k - 1], v0, raw_rs2, addr, 1, true);
                    }
                } else if (raw_opcode == 0x03 and raw_funct3 != 3) {
                    // Sub-word loads: LB(f3=0), LH(f3=1), LW(f3=2), LBU(f3=4), LHU(f3=5), LWU(f3=6)
                    // LD (f3=3) is NOT expanded
                    const raw_rs2_unused: u8 = @truncate((instr_word >> 20) & 0x1F);
                    _ = raw_rs2_unused;
                    const raw_imm: i64 = @as(i64, @as(i32, @bitCast(instr_word)) >> 20);
                    const v0: u8 = 40;
                    const v1: u8 = 41;
                    const v2: u8 = 42;
                    switch (raw_funct3) {
                        0, 4 => {
                            // LB (f3=0) / LBU (f3=4) → 8 entries
                            // k = base_pc + 7
                            const shift_56: u7 = 56;
                            const ones_56: u128 = (@as(u128, 1) << @intCast(64 - @as(u8, shift_56))) - 1;
                            const bitmask_56: u64 = @truncate(ones_56 << shift_56);
                            // k: VirtualSRAI/VirtualSRLI (last, vsr=0)
                            if (raw_funct3 == 0) {
                                populateVirtualSRAIEntry(&entries[k], raw_rd, v1, addr, bitmask_56, 0, false);
                            } else {
                                populateVirtualSRLIEntry(&entries[k], raw_rd, v1, addr, bitmask_56, 0, false);
                            }
                            if (is_compressed) entries[k].circuit_flags[@intFromEnum(CircuitFlags.IsCompressed)] = true;
                            // k-1: MUL(v1, v1, v2) (vsr=1)
                            if (k >= 1) {
                                populateVirtualRTypeEntry(&entries[k - 1], v1, v1, v2, addr, 1, false, 0x33, 0, 0x01);
                            }
                            // k-2: VirtualPow2(v2, v0, 0) (vsr=2)
                            if (k >= 2) {
                                populateVirtualPow2Entry(&entries[k - 2], v2, v0, addr, 2, false);
                            }
                            // k-3: VirtualMULI(v0, v0, 8) (vsr=3)
                            if (k >= 3) {
                                populateVirtualMULIEntry(&entries[k - 3], v0, v0, addr, 3, 3, false);
                            }
                            // k-4: XORI(v0, v0, 7) (vsr=4)
                            if (k >= 4) {
                                const xori_instr: u32 = (7 << 20) | (@as(u32, v0 & 0x1F) << 15) | (4 << 12) | (@as(u32, v0 & 0x1F) << 7) | 0x13;
                                populateEntryFromInstruction(&entries[k - 4], xori_instr, addr);
                                entries[k - 4].rd = v0;
                                entries[k - 4].rs1 = v0;
                                entries[k - 4].circuit_flags[@intFromEnum(CircuitFlags.VirtualInstruction)] = true;
                                entries[k - 4].circuit_flags[@intFromEnum(CircuitFlags.DoNotUpdateUnexpandedPC)] = true;
                                entries[k - 4].virtual_sequence_remaining = 4;
                                entries[k - 4].is_first_in_sequence = false;
                            }
                            // k-5: LD(v1, v1, 0) (vsr=5)
                            if (k >= 5) {
                                const ld_instr: u32 = (3 << 12) | (@as(u32, v1 & 0x1F) << 15) | (@as(u32, v1 & 0x1F) << 7) | 0x03;
                                populateEntryFromInstruction(&entries[k - 5], ld_instr, addr);
                                entries[k - 5].rd = v1;
                                entries[k - 5].rs1 = v1;
                                entries[k - 5].circuit_flags[@intFromEnum(CircuitFlags.VirtualInstruction)] = true;
                                entries[k - 5].circuit_flags[@intFromEnum(CircuitFlags.DoNotUpdateUnexpandedPC)] = true;
                                entries[k - 5].virtual_sequence_remaining = 5;
                                entries[k - 5].is_first_in_sequence = false;
                            }
                            // k-6: ANDI(v1, v0, -8) (vsr=6)
                            if (k >= 6) {
                                const andi_instr: u32 = (@as(u32, @bitCast(@as(i32, -8))) & 0xFFF) << 20 | (@as(u32, v0 & 0x1F) << 15) | (7 << 12) | (@as(u32, v1 & 0x1F) << 7) | 0x13;
                                populateEntryFromInstruction(&entries[k - 6], andi_instr, addr);
                                entries[k - 6].rd = v1;
                                entries[k - 6].rs1 = v0;
                                entries[k - 6].circuit_flags[@intFromEnum(CircuitFlags.VirtualInstruction)] = true;
                                entries[k - 6].circuit_flags[@intFromEnum(CircuitFlags.DoNotUpdateUnexpandedPC)] = true;
                                entries[k - 6].virtual_sequence_remaining = 6;
                                entries[k - 6].is_first_in_sequence = false;
                            }
                            // k-7: ADDI(v0, rs1, imm) (vsr=7, first)
                            if (k >= 7) {
                                const imm_bits: u32 = @bitCast(@as(i32, @truncate(raw_imm)));
                                const addi_instr: u32 = (imm_bits & 0xFFF) << 20 | (@as(u32, raw_rs1 & 0x1F) << 15) | (0 << 12) | (@as(u32, v0 & 0x1F) << 7) | 0x13;
                                populateEntryFromInstruction(&entries[k - 7], addi_instr, addr);
                                entries[k - 7].rd = v0;
                                entries[k - 7].rs1 = raw_rs1;
                                entries[k - 7].circuit_flags[@intFromEnum(CircuitFlags.VirtualInstruction)] = true;
                                entries[k - 7].circuit_flags[@intFromEnum(CircuitFlags.DoNotUpdateUnexpandedPC)] = true;
                                entries[k - 7].circuit_flags[@intFromEnum(CircuitFlags.IsFirstInSequence)] = true;
                                entries[k - 7].virtual_sequence_remaining = 7;
                                entries[k - 7].is_first_in_sequence = true;
                            }
                        },
                        1, 5 => {
                            // LH (f3=1) / LHU (f3=5) → 9 entries
                            // k = base_pc + 8
                            const shift_48: u7 = 48;
                            const ones_48: u128 = (@as(u128, 1) << @intCast(64 - @as(u8, shift_48))) - 1;
                            const bitmask_48: u64 = @truncate(ones_48 << shift_48);
                            // k: VirtualSRAI/VirtualSRLI (last, vsr=0)
                            if (raw_funct3 == 1) {
                                populateVirtualSRAIEntry(&entries[k], raw_rd, v1, addr, bitmask_48, 0, false);
                            } else {
                                populateVirtualSRLIEntry(&entries[k], raw_rd, v1, addr, bitmask_48, 0, false);
                            }
                            if (is_compressed) entries[k].circuit_flags[@intFromEnum(CircuitFlags.IsCompressed)] = true;
                            // k-1: MUL(v1, v1, v2) (vsr=1)
                            if (k >= 1) {
                                populateVirtualRTypeEntry(&entries[k - 1], v1, v1, v2, addr, 1, false, 0x33, 0, 0x01);
                            }
                            // k-2: VirtualPow2(v2, v0, 0) (vsr=2)
                            if (k >= 2) {
                                populateVirtualPow2Entry(&entries[k - 2], v2, v0, addr, 2, false);
                            }
                            // k-3: VirtualMULI(v0, v0, 8) (vsr=3)
                            if (k >= 3) {
                                populateVirtualMULIEntry(&entries[k - 3], v0, v0, addr, 3, 3, false);
                            }
                            // k-4: XORI(v0, v0, 6) (vsr=4)
                            if (k >= 4) {
                                const xori_instr: u32 = (6 << 20) | (@as(u32, v0 & 0x1F) << 15) | (4 << 12) | (@as(u32, v0 & 0x1F) << 7) | 0x13;
                                populateEntryFromInstruction(&entries[k - 4], xori_instr, addr);
                                entries[k - 4].rd = v0;
                                entries[k - 4].rs1 = v0;
                                entries[k - 4].circuit_flags[@intFromEnum(CircuitFlags.VirtualInstruction)] = true;
                                entries[k - 4].circuit_flags[@intFromEnum(CircuitFlags.DoNotUpdateUnexpandedPC)] = true;
                                entries[k - 4].virtual_sequence_remaining = 4;
                                entries[k - 4].is_first_in_sequence = false;
                            }
                            // k-5: LD(v1, v1, 0) (vsr=5)
                            if (k >= 5) {
                                const ld_instr: u32 = (3 << 12) | (@as(u32, v1 & 0x1F) << 15) | (@as(u32, v1 & 0x1F) << 7) | 0x03;
                                populateEntryFromInstruction(&entries[k - 5], ld_instr, addr);
                                entries[k - 5].rd = v1;
                                entries[k - 5].rs1 = v1;
                                entries[k - 5].circuit_flags[@intFromEnum(CircuitFlags.VirtualInstruction)] = true;
                                entries[k - 5].circuit_flags[@intFromEnum(CircuitFlags.DoNotUpdateUnexpandedPC)] = true;
                                entries[k - 5].virtual_sequence_remaining = 5;
                                entries[k - 5].is_first_in_sequence = false;
                            }
                            // k-6: ANDI(v1, v0, -8) (vsr=6)
                            if (k >= 6) {
                                const andi_instr: u32 = (@as(u32, @bitCast(@as(i32, -8))) & 0xFFF) << 20 | (@as(u32, v0 & 0x1F) << 15) | (7 << 12) | (@as(u32, v1 & 0x1F) << 7) | 0x13;
                                populateEntryFromInstruction(&entries[k - 6], andi_instr, addr);
                                entries[k - 6].rd = v1;
                                entries[k - 6].rs1 = v0;
                                entries[k - 6].circuit_flags[@intFromEnum(CircuitFlags.VirtualInstruction)] = true;
                                entries[k - 6].circuit_flags[@intFromEnum(CircuitFlags.DoNotUpdateUnexpandedPC)] = true;
                                entries[k - 6].virtual_sequence_remaining = 6;
                                entries[k - 6].is_first_in_sequence = false;
                            }
                            // k-7: ADDI(v0, rs1, imm) (vsr=7)
                            if (k >= 7) {
                                const imm_bits: u32 = @bitCast(@as(i32, @truncate(raw_imm)));
                                const addi_instr: u32 = (imm_bits & 0xFFF) << 20 | (@as(u32, raw_rs1 & 0x1F) << 15) | (0 << 12) | (@as(u32, v0 & 0x1F) << 7) | 0x13;
                                populateEntryFromInstruction(&entries[k - 7], addi_instr, addr);
                                entries[k - 7].rd = v0;
                                entries[k - 7].rs1 = raw_rs1;
                                entries[k - 7].circuit_flags[@intFromEnum(CircuitFlags.VirtualInstruction)] = true;
                                entries[k - 7].circuit_flags[@intFromEnum(CircuitFlags.DoNotUpdateUnexpandedPC)] = true;
                                entries[k - 7].virtual_sequence_remaining = 7;
                                entries[k - 7].is_first_in_sequence = false;
                            }
                            // k-8: VirtualAssertHalfwordAlignment(rs1, imm) (vsr=8, first)
                            if (k >= 8) {
                                populateVirtualAssertHalfwordAlignmentEntry(&entries[k - 8], raw_rs1, raw_imm, addr, 8, true);
                            }
                        },
                        2 => {
                            // LW → 8 entries (with SRL, not SLL)
                            // k = base_pc + 7
                            // k: VirtualSignExtendWord(rd, v1, 0) (vsr=0)
                            populateVirtualSignExtendWordEntry(&entries[k], raw_rd, addr, is_compressed);
                            entries[k].rs1 = v1;
                            // k-1: VirtualSRL(v1, v1, v2) (vsr=1)
                            if (k >= 1) {
                                populateVirtualSRLEntry(&entries[k - 1], v1, v1, v2, addr, 1, false);
                            }
                            // k-2: VirtualShiftRightBitmask(v2, v0, 0) (vsr=2)
                            if (k >= 2) {
                                populateVirtualShiftRightBitmaskEntry(&entries[k - 2], v2, v0, addr, 2, false);
                            }
                            // k-3: VirtualMULI(v0, v0, 8) (vsr=3)
                            if (k >= 3) {
                                populateVirtualMULIEntry(&entries[k - 3], v0, v0, addr, 3, 3, false);
                            }
                            // k-4: LD(v1, v1, 0) (vsr=4)
                            if (k >= 4) {
                                const ld_instr: u32 = (3 << 12) | (@as(u32, v1 & 0x1F) << 15) | (@as(u32, v1 & 0x1F) << 7) | 0x03;
                                populateEntryFromInstruction(&entries[k - 4], ld_instr, addr);
                                entries[k - 4].rd = v1;
                                entries[k - 4].rs1 = v1;
                                entries[k - 4].circuit_flags[@intFromEnum(CircuitFlags.VirtualInstruction)] = true;
                                entries[k - 4].circuit_flags[@intFromEnum(CircuitFlags.DoNotUpdateUnexpandedPC)] = true;
                                entries[k - 4].virtual_sequence_remaining = 4;
                                entries[k - 4].is_first_in_sequence = false;
                            }
                            // k-5: ANDI(v1, v0, -8) (vsr=5)
                            if (k >= 5) {
                                const andi_instr: u32 = (@as(u32, @bitCast(@as(i32, -8))) & 0xFFF) << 20 | (@as(u32, v0 & 0x1F) << 15) | (7 << 12) | (@as(u32, v1 & 0x1F) << 7) | 0x13;
                                populateEntryFromInstruction(&entries[k - 5], andi_instr, addr);
                                entries[k - 5].rd = v1;
                                entries[k - 5].rs1 = v0;
                                entries[k - 5].circuit_flags[@intFromEnum(CircuitFlags.VirtualInstruction)] = true;
                                entries[k - 5].circuit_flags[@intFromEnum(CircuitFlags.DoNotUpdateUnexpandedPC)] = true;
                                entries[k - 5].virtual_sequence_remaining = 5;
                                entries[k - 5].is_first_in_sequence = false;
                            }
                            // k-6: ADDI(v0, rs1, imm) (vsr=6)
                            if (k >= 6) {
                                const imm_bits: u32 = @bitCast(@as(i32, @truncate(raw_imm)));
                                const addi_instr: u32 = (imm_bits & 0xFFF) << 20 | (@as(u32, raw_rs1 & 0x1F) << 15) | (0 << 12) | (@as(u32, v0 & 0x1F) << 7) | 0x13;
                                populateEntryFromInstruction(&entries[k - 6], addi_instr, addr);
                                entries[k - 6].rd = v0;
                                entries[k - 6].rs1 = raw_rs1;
                                entries[k - 6].circuit_flags[@intFromEnum(CircuitFlags.VirtualInstruction)] = true;
                                entries[k - 6].circuit_flags[@intFromEnum(CircuitFlags.DoNotUpdateUnexpandedPC)] = true;
                                entries[k - 6].virtual_sequence_remaining = 6;
                                entries[k - 6].is_first_in_sequence = false;
                            }
                            // k-7: VirtualAssertWordAlignment(rs1, imm) (vsr=7, first)
                            if (k >= 7) {
                                populateVirtualAssertWordAlignmentEntry(&entries[k - 7], raw_rs1, raw_imm, addr, 7, true);
                            }
                        },
                        6 => {
                            // LWU → 9 entries (with SLL then SRLI)
                            // k = base_pc + 8
                            const shift_32: u7 = 32;
                            const ones_32: u128 = (@as(u128, 1) << @intCast(64 - @as(u8, shift_32))) - 1;
                            const bitmask_32: u64 = @truncate(ones_32 << shift_32);
                            // k: VirtualSRLI(rd, v1, bitmask_32) (vsr=0)
                            populateVirtualSRLIEntry(&entries[k], raw_rd, v1, addr, bitmask_32, 0, false);
                            if (is_compressed) entries[k].circuit_flags[@intFromEnum(CircuitFlags.IsCompressed)] = true;
                            // k-1: MUL(v1, v1, v2) (vsr=1)
                            if (k >= 1) {
                                populateVirtualRTypeEntry(&entries[k - 1], v1, v1, v2, addr, 1, false, 0x33, 0, 0x01);
                            }
                            // k-2: VirtualPow2(v2, v0, 0) (vsr=2)
                            if (k >= 2) {
                                populateVirtualPow2Entry(&entries[k - 2], v2, v0, addr, 2, false);
                            }
                            // k-3: VirtualMULI(v0, v0, 8) (vsr=3)
                            if (k >= 3) {
                                populateVirtualMULIEntry(&entries[k - 3], v0, v0, addr, 3, 3, false);
                            }
                            // k-4: XORI(v0, v0, 4) (vsr=4)
                            if (k >= 4) {
                                const xori_instr: u32 = (4 << 20) | (@as(u32, v0 & 0x1F) << 15) | (4 << 12) | (@as(u32, v0 & 0x1F) << 7) | 0x13;
                                populateEntryFromInstruction(&entries[k - 4], xori_instr, addr);
                                entries[k - 4].rd = v0;
                                entries[k - 4].rs1 = v0;
                                entries[k - 4].circuit_flags[@intFromEnum(CircuitFlags.VirtualInstruction)] = true;
                                entries[k - 4].circuit_flags[@intFromEnum(CircuitFlags.DoNotUpdateUnexpandedPC)] = true;
                                entries[k - 4].virtual_sequence_remaining = 4;
                                entries[k - 4].is_first_in_sequence = false;
                            }
                            // k-5: LD(v1, v1, 0) (vsr=5)
                            if (k >= 5) {
                                const ld_instr: u32 = (3 << 12) | (@as(u32, v1 & 0x1F) << 15) | (@as(u32, v1 & 0x1F) << 7) | 0x03;
                                populateEntryFromInstruction(&entries[k - 5], ld_instr, addr);
                                entries[k - 5].rd = v1;
                                entries[k - 5].rs1 = v1;
                                entries[k - 5].circuit_flags[@intFromEnum(CircuitFlags.VirtualInstruction)] = true;
                                entries[k - 5].circuit_flags[@intFromEnum(CircuitFlags.DoNotUpdateUnexpandedPC)] = true;
                                entries[k - 5].virtual_sequence_remaining = 5;
                                entries[k - 5].is_first_in_sequence = false;
                            }
                            // k-6: ANDI(v1, v0, -8) (vsr=6)
                            if (k >= 6) {
                                const andi_instr: u32 = (@as(u32, @bitCast(@as(i32, -8))) & 0xFFF) << 20 | (@as(u32, v0 & 0x1F) << 15) | (7 << 12) | (@as(u32, v1 & 0x1F) << 7) | 0x13;
                                populateEntryFromInstruction(&entries[k - 6], andi_instr, addr);
                                entries[k - 6].rd = v1;
                                entries[k - 6].rs1 = v0;
                                entries[k - 6].circuit_flags[@intFromEnum(CircuitFlags.VirtualInstruction)] = true;
                                entries[k - 6].circuit_flags[@intFromEnum(CircuitFlags.DoNotUpdateUnexpandedPC)] = true;
                                entries[k - 6].virtual_sequence_remaining = 6;
                                entries[k - 6].is_first_in_sequence = false;
                            }
                            // k-7: ADDI(v0, rs1, imm) (vsr=7)
                            if (k >= 7) {
                                const imm_bits: u32 = @bitCast(@as(i32, @truncate(raw_imm)));
                                const addi_instr: u32 = (imm_bits & 0xFFF) << 20 | (@as(u32, raw_rs1 & 0x1F) << 15) | (0 << 12) | (@as(u32, v0 & 0x1F) << 7) | 0x13;
                                populateEntryFromInstruction(&entries[k - 7], addi_instr, addr);
                                entries[k - 7].rd = v0;
                                entries[k - 7].rs1 = raw_rs1;
                                entries[k - 7].circuit_flags[@intFromEnum(CircuitFlags.VirtualInstruction)] = true;
                                entries[k - 7].circuit_flags[@intFromEnum(CircuitFlags.DoNotUpdateUnexpandedPC)] = true;
                                entries[k - 7].virtual_sequence_remaining = 7;
                                entries[k - 7].is_first_in_sequence = false;
                            }
                            // k-8: VirtualAssertWordAlignment(rs1, imm) (vsr=8, first)
                            if (k >= 8) {
                                populateVirtualAssertWordAlignmentEntry(&entries[k - 8], raw_rs1, raw_imm, addr, 8, true);
                            }
                        },
                        else => {
                            // LD or unknown — pass through
                            populateEntryFromInstruction(&entries[k], instr_word, addr);
                            if (is_compressed) entries[k].circuit_flags[@intFromEnum(CircuitFlags.IsCompressed)] = true;
                        },
                    }
                } else if (raw_opcode == 0x23 and raw_funct3 != 3) {
                    // Sub-word stores: SB(f3=0), SH(f3=1), SW(f3=2)
                    // SD (f3=3) is NOT expanded
                    const raw_rs2: u8 = @truncate((instr_word >> 20) & 0x1F);
                    // S-type immediate: imm[11:5] = instr[31:25], imm[4:0] = instr[11:7]
                    const raw_imm: i64 = @as(i64, @as(i32, @bitCast(instr_word)) >> 20) & ~@as(i64, 0x1F) | @as(i64, (instr_word >> 7) & 0x1F);
                    const v0: u8 = 40;
                    const v1: u8 = 41;
                    const v2: u8 = 42;
                    const v3: u8 = 43;
                    const v4: u8 = 44;
                    const v5: u8 = 45;
                    switch (raw_funct3) {
                        0 => {
                            // SB → 13 entries. k = base_pc + 12
                            // k: SD(v1, v2, 0) (vsr=0)
                            {
                                const sd_instr: u32 = (@as(u32, v2 & 0x1F) << 20) | (@as(u32, v1 & 0x1F) << 15) | (3 << 12) | 0x23;
                                populateEntryFromInstruction(&entries[k], sd_instr, addr);
                                entries[k].rs1 = v1;
                                entries[k].rs2 = v2;
                                entries[k].circuit_flags[@intFromEnum(CircuitFlags.VirtualInstruction)] = true;
                                entries[k].virtual_sequence_remaining = 0;
                                if (is_compressed) entries[k].circuit_flags[@intFromEnum(CircuitFlags.IsCompressed)] = true;
                            }
                            // k-1: XOR(v2, v2, v3) (vsr=1)
                            if (k >= 1) {
                                populateVirtualRTypeEntry(&entries[k - 1], v2, v2, v3, addr, 1, false, 0x33, 4, 0);
                            }
                            // k-2: AND(v3, v3, v0) (vsr=2)
                            if (k >= 2) {
                                populateVirtualRTypeEntry(&entries[k - 2], v3, v3, v0, addr, 2, false, 0x33, 7, 0);
                            }
                            // k-3: XOR(v3, v2, v3) (vsr=3)
                            if (k >= 3) {
                                populateVirtualRTypeEntry(&entries[k - 3], v3, v2, v3, addr, 3, false, 0x33, 4, 0);
                            }
                            // k-4: MUL(v3, rs2, v5) (vsr=4) — from SLL(v3, rs2, v3) step 2
                            if (k >= 4) {
                                populateVirtualRTypeEntry(&entries[k - 4], v3, raw_rs2, v5, addr, 4, false, 0x33, 0, 0x01);
                            }
                            // k-5: VirtualPow2(v5, v3, 0) (vsr=5) — from SLL step 1
                            if (k >= 5) {
                                populateVirtualPow2Entry(&entries[k - 5], v5, v3, addr, 5, false);
                            }
                            // k-6: MUL(v0, v0, v4) (vsr=6) — from SLL(v0, v0, v3) step 2
                            if (k >= 6) {
                                populateVirtualRTypeEntry(&entries[k - 6], v0, v0, v4, addr, 6, false, 0x33, 0, 0x01);
                            }
                            // k-7: VirtualPow2(v4, v3, 0) (vsr=7) — from SLL step 1
                            if (k >= 7) {
                                populateVirtualPow2Entry(&entries[k - 7], v4, v3, addr, 7, false);
                            }
                            // k-8: LUI(v0, 0xff) (vsr=8)
                            if (k >= 8) {
                                const lui_instr: u32 = (0xff << 12) | (@as(u32, v0 & 0x1F) << 7) | 0x37;
                                populateEntryFromInstruction(&entries[k - 8], lui_instr, addr);
                                entries[k - 8].rd = v0;
                                entries[k - 8].circuit_flags[@intFromEnum(CircuitFlags.VirtualInstruction)] = true;
                                entries[k - 8].circuit_flags[@intFromEnum(CircuitFlags.DoNotUpdateUnexpandedPC)] = true;
                                entries[k - 8].virtual_sequence_remaining = 8;
                                entries[k - 8].is_first_in_sequence = false;
                            }
                            // k-9: VirtualMULI(v3, v0, 8) (vsr=9)
                            if (k >= 9) {
                                populateVirtualMULIEntry(&entries[k - 9], v3, v0, addr, 3, 9, false);
                            }
                            // k-10: LD(v2, v1, 0) (vsr=10)
                            if (k >= 10) {
                                const ld_instr: u32 = (3 << 12) | (@as(u32, v1 & 0x1F) << 15) | (@as(u32, v2 & 0x1F) << 7) | 0x03;
                                populateEntryFromInstruction(&entries[k - 10], ld_instr, addr);
                                entries[k - 10].rd = v2;
                                entries[k - 10].rs1 = v1;
                                entries[k - 10].circuit_flags[@intFromEnum(CircuitFlags.VirtualInstruction)] = true;
                                entries[k - 10].circuit_flags[@intFromEnum(CircuitFlags.DoNotUpdateUnexpandedPC)] = true;
                                entries[k - 10].virtual_sequence_remaining = 10;
                                entries[k - 10].is_first_in_sequence = false;
                            }
                            // k-11: ANDI(v1, v0, -8) (vsr=11)
                            if (k >= 11) {
                                const andi_instr: u32 = (@as(u32, @bitCast(@as(i32, -8))) & 0xFFF) << 20 | (@as(u32, v0 & 0x1F) << 15) | (7 << 12) | (@as(u32, v1 & 0x1F) << 7) | 0x13;
                                populateEntryFromInstruction(&entries[k - 11], andi_instr, addr);
                                entries[k - 11].rd = v1;
                                entries[k - 11].rs1 = v0;
                                entries[k - 11].circuit_flags[@intFromEnum(CircuitFlags.VirtualInstruction)] = true;
                                entries[k - 11].circuit_flags[@intFromEnum(CircuitFlags.DoNotUpdateUnexpandedPC)] = true;
                                entries[k - 11].virtual_sequence_remaining = 11;
                                entries[k - 11].is_first_in_sequence = false;
                            }
                            // k-12: ADDI(v0, rs1, imm) (vsr=12, first)
                            if (k >= 12) {
                                const imm_bits: u32 = @bitCast(@as(i32, @truncate(raw_imm)));
                                const addi_instr: u32 = (imm_bits & 0xFFF) << 20 | (@as(u32, raw_rs1 & 0x1F) << 15) | (0 << 12) | (@as(u32, v0 & 0x1F) << 7) | 0x13;
                                populateEntryFromInstruction(&entries[k - 12], addi_instr, addr);
                                entries[k - 12].rd = v0;
                                entries[k - 12].rs1 = raw_rs1;
                                entries[k - 12].circuit_flags[@intFromEnum(CircuitFlags.VirtualInstruction)] = true;
                                entries[k - 12].circuit_flags[@intFromEnum(CircuitFlags.DoNotUpdateUnexpandedPC)] = true;
                                entries[k - 12].circuit_flags[@intFromEnum(CircuitFlags.IsFirstInSequence)] = true;
                                entries[k - 12].virtual_sequence_remaining = 12;
                                entries[k - 12].is_first_in_sequence = true;
                            }
                        },
                        1 => {
                            // SH → 14 entries. k = base_pc + 13
                            // Same as SB but with alignment assert + 0xffff mask
                            // k: SD(v1, v2, 0) (vsr=0)
                            {
                                const sd_instr: u32 = (@as(u32, v2 & 0x1F) << 20) | (@as(u32, v1 & 0x1F) << 15) | (3 << 12) | 0x23;
                                populateEntryFromInstruction(&entries[k], sd_instr, addr);
                                entries[k].rs1 = v1;
                                entries[k].rs2 = v2;
                                entries[k].circuit_flags[@intFromEnum(CircuitFlags.VirtualInstruction)] = true;
                                entries[k].virtual_sequence_remaining = 0;
                                if (is_compressed) entries[k].circuit_flags[@intFromEnum(CircuitFlags.IsCompressed)] = true;
                            }
                            if (k >= 1) {
                                populateVirtualRTypeEntry(&entries[k - 1], v2, v2, v3, addr, 1, false, 0x33, 4, 0);
                            }
                            if (k >= 2) {
                                populateVirtualRTypeEntry(&entries[k - 2], v3, v3, v0, addr, 2, false, 0x33, 7, 0);
                            }
                            if (k >= 3) {
                                populateVirtualRTypeEntry(&entries[k - 3], v3, v2, v3, addr, 3, false, 0x33, 4, 0);
                            }
                            if (k >= 4) {
                                populateVirtualRTypeEntry(&entries[k - 4], v3, raw_rs2, v5, addr, 4, false, 0x33, 0, 0x01);
                            }
                            if (k >= 5) {
                                populateVirtualPow2Entry(&entries[k - 5], v5, v3, addr, 5, false);
                            }
                            if (k >= 6) {
                                populateVirtualRTypeEntry(&entries[k - 6], v0, v0, v4, addr, 6, false, 0x33, 0, 0x01);
                            }
                            if (k >= 7) {
                                populateVirtualPow2Entry(&entries[k - 7], v4, v3, addr, 7, false);
                            }
                            // k-8: LUI(v0, 0xffff) (vsr=8)
                            if (k >= 8) {
                                const lui_instr: u32 = (0xffff << 12) | (@as(u32, v0 & 0x1F) << 7) | 0x37;
                                populateEntryFromInstruction(&entries[k - 8], lui_instr, addr);
                                entries[k - 8].rd = v0;
                                entries[k - 8].circuit_flags[@intFromEnum(CircuitFlags.VirtualInstruction)] = true;
                                entries[k - 8].circuit_flags[@intFromEnum(CircuitFlags.DoNotUpdateUnexpandedPC)] = true;
                                entries[k - 8].virtual_sequence_remaining = 8;
                                entries[k - 8].is_first_in_sequence = false;
                            }
                            if (k >= 9) {
                                populateVirtualMULIEntry(&entries[k - 9], v3, v0, addr, 3, 9, false);
                            }
                            // k-10: LD(v2, v1, 0) (vsr=10)
                            if (k >= 10) {
                                const ld_instr: u32 = (3 << 12) | (@as(u32, v1 & 0x1F) << 15) | (@as(u32, v2 & 0x1F) << 7) | 0x03;
                                populateEntryFromInstruction(&entries[k - 10], ld_instr, addr);
                                entries[k - 10].rd = v2;
                                entries[k - 10].rs1 = v1;
                                entries[k - 10].circuit_flags[@intFromEnum(CircuitFlags.VirtualInstruction)] = true;
                                entries[k - 10].circuit_flags[@intFromEnum(CircuitFlags.DoNotUpdateUnexpandedPC)] = true;
                                entries[k - 10].virtual_sequence_remaining = 10;
                                entries[k - 10].is_first_in_sequence = false;
                            }
                            if (k >= 11) {
                                const andi_instr: u32 = (@as(u32, @bitCast(@as(i32, -8))) & 0xFFF) << 20 | (@as(u32, v0 & 0x1F) << 15) | (7 << 12) | (@as(u32, v1 & 0x1F) << 7) | 0x13;
                                populateEntryFromInstruction(&entries[k - 11], andi_instr, addr);
                                entries[k - 11].rd = v1;
                                entries[k - 11].rs1 = v0;
                                entries[k - 11].circuit_flags[@intFromEnum(CircuitFlags.VirtualInstruction)] = true;
                                entries[k - 11].circuit_flags[@intFromEnum(CircuitFlags.DoNotUpdateUnexpandedPC)] = true;
                                entries[k - 11].virtual_sequence_remaining = 11;
                                entries[k - 11].is_first_in_sequence = false;
                            }
                            if (k >= 12) {
                                const imm_bits: u32 = @bitCast(@as(i32, @truncate(raw_imm)));
                                const addi_instr: u32 = (imm_bits & 0xFFF) << 20 | (@as(u32, raw_rs1 & 0x1F) << 15) | (0 << 12) | (@as(u32, v0 & 0x1F) << 7) | 0x13;
                                populateEntryFromInstruction(&entries[k - 12], addi_instr, addr);
                                entries[k - 12].rd = v0;
                                entries[k - 12].rs1 = raw_rs1;
                                entries[k - 12].circuit_flags[@intFromEnum(CircuitFlags.VirtualInstruction)] = true;
                                entries[k - 12].circuit_flags[@intFromEnum(CircuitFlags.DoNotUpdateUnexpandedPC)] = true;
                                entries[k - 12].virtual_sequence_remaining = 12;
                                entries[k - 12].is_first_in_sequence = false;
                            }
                            // k-13: VirtualAssertHalfwordAlignment(rs1, imm) (vsr=13, first)
                            if (k >= 13) {
                                populateVirtualAssertHalfwordAlignmentEntry(&entries[k - 13], raw_rs1, raw_imm, addr, 13, true);
                            }
                        },
                        2 => {
                            // SW → 15 entries. k = base_pc + 14
                            const shift_32: u7 = 32;
                            const ones_32: u128 = (@as(u128, 1) << @intCast(64 - @as(u8, shift_32))) - 1;
                            const bitmask_32: u64 = @truncate(ones_32 << shift_32);
                            // k: SD(v1, v2, 0) (vsr=0)
                            {
                                const sd_instr: u32 = (@as(u32, v2 & 0x1F) << 20) | (@as(u32, v1 & 0x1F) << 15) | (3 << 12) | 0x23;
                                populateEntryFromInstruction(&entries[k], sd_instr, addr);
                                entries[k].rs1 = v1;
                                entries[k].rs2 = v2;
                                entries[k].circuit_flags[@intFromEnum(CircuitFlags.VirtualInstruction)] = true;
                                entries[k].virtual_sequence_remaining = 0;
                                if (is_compressed) entries[k].circuit_flags[@intFromEnum(CircuitFlags.IsCompressed)] = true;
                            }
                            if (k >= 1) {
                                populateVirtualRTypeEntry(&entries[k - 1], v2, v2, v0, addr, 1, false, 0x33, 4, 0);
                            }
                            if (k >= 2) {
                                populateVirtualRTypeEntry(&entries[k - 2], v0, v0, v3, addr, 2, false, 0x33, 7, 0);
                            }
                            if (k >= 3) {
                                populateVirtualRTypeEntry(&entries[k - 3], v0, v2, v0, addr, 3, false, 0x33, 4, 0);
                            }
                            if (k >= 4) {
                                populateVirtualRTypeEntry(&entries[k - 4], v0, raw_rs2, v5, addr, 4, false, 0x33, 0, 0x01);
                            }
                            if (k >= 5) {
                                populateVirtualPow2Entry(&entries[k - 5], v5, v0, addr, 5, false);
                            }
                            if (k >= 6) {
                                populateVirtualRTypeEntry(&entries[k - 6], v3, v3, v4, addr, 6, false, 0x33, 0, 0x01);
                            }
                            if (k >= 7) {
                                populateVirtualPow2Entry(&entries[k - 7], v4, v0, addr, 7, false);
                            }
                            // k-8: VirtualSRLI(v3, v3, bitmask_32) (vsr=8)
                            if (k >= 8) {
                                populateVirtualSRLIEntry(&entries[k - 8], v3, v3, addr, bitmask_32, 8, false);
                            }
                            // k-9: ORI(v3, x0, -1) (vsr=9)
                            if (k >= 9) {
                                const ori_instr: u32 = (@as(u32, @bitCast(@as(i32, -1))) & 0xFFF) << 20 | (0 << 15) | (6 << 12) | (@as(u32, v3 & 0x1F) << 7) | 0x13;
                                populateEntryFromInstruction(&entries[k - 9], ori_instr, addr);
                                entries[k - 9].rd = v3;
                                entries[k - 9].rs1 = 0;
                                entries[k - 9].circuit_flags[@intFromEnum(CircuitFlags.VirtualInstruction)] = true;
                                entries[k - 9].circuit_flags[@intFromEnum(CircuitFlags.DoNotUpdateUnexpandedPC)] = true;
                                entries[k - 9].virtual_sequence_remaining = 9;
                                entries[k - 9].is_first_in_sequence = false;
                            }
                            // k-10: VirtualMULI(v0, v0, 8) (vsr=10)
                            if (k >= 10) {
                                populateVirtualMULIEntry(&entries[k - 10], v0, v0, addr, 3, 10, false);
                            }
                            // k-11: LD(v2, v1, 0) (vsr=11)
                            if (k >= 11) {
                                const ld_instr: u32 = (3 << 12) | (@as(u32, v1 & 0x1F) << 15) | (@as(u32, v2 & 0x1F) << 7) | 0x03;
                                populateEntryFromInstruction(&entries[k - 11], ld_instr, addr);
                                entries[k - 11].rd = v2;
                                entries[k - 11].rs1 = v1;
                                entries[k - 11].circuit_flags[@intFromEnum(CircuitFlags.VirtualInstruction)] = true;
                                entries[k - 11].circuit_flags[@intFromEnum(CircuitFlags.DoNotUpdateUnexpandedPC)] = true;
                                entries[k - 11].virtual_sequence_remaining = 11;
                                entries[k - 11].is_first_in_sequence = false;
                            }
                            if (k >= 12) {
                                const andi_instr: u32 = (@as(u32, @bitCast(@as(i32, -8))) & 0xFFF) << 20 | (@as(u32, v0 & 0x1F) << 15) | (7 << 12) | (@as(u32, v1 & 0x1F) << 7) | 0x13;
                                populateEntryFromInstruction(&entries[k - 12], andi_instr, addr);
                                entries[k - 12].rd = v1;
                                entries[k - 12].rs1 = v0;
                                entries[k - 12].circuit_flags[@intFromEnum(CircuitFlags.VirtualInstruction)] = true;
                                entries[k - 12].circuit_flags[@intFromEnum(CircuitFlags.DoNotUpdateUnexpandedPC)] = true;
                                entries[k - 12].virtual_sequence_remaining = 12;
                                entries[k - 12].is_first_in_sequence = false;
                            }
                            if (k >= 13) {
                                const imm_bits: u32 = @bitCast(@as(i32, @truncate(raw_imm)));
                                const addi_instr: u32 = (imm_bits & 0xFFF) << 20 | (@as(u32, raw_rs1 & 0x1F) << 15) | (0 << 12) | (@as(u32, v0 & 0x1F) << 7) | 0x13;
                                populateEntryFromInstruction(&entries[k - 13], addi_instr, addr);
                                entries[k - 13].rd = v0;
                                entries[k - 13].rs1 = raw_rs1;
                                entries[k - 13].circuit_flags[@intFromEnum(CircuitFlags.VirtualInstruction)] = true;
                                entries[k - 13].circuit_flags[@intFromEnum(CircuitFlags.DoNotUpdateUnexpandedPC)] = true;
                                entries[k - 13].virtual_sequence_remaining = 13;
                                entries[k - 13].is_first_in_sequence = false;
                            }
                            // k-14: VirtualAssertWordAlignment(rs1, imm) (vsr=14, first)
                            if (k >= 14) {
                                populateVirtualAssertWordAlignmentEntry(&entries[k - 14], raw_rs1, raw_imm, addr, 14, true);
                            }
                        },
                        else => {
                            // SD or unknown — pass through
                            populateEntryFromInstruction(&entries[k], instr_word, addr);
                            if (is_compressed) entries[k].circuit_flags[@intFromEnum(CircuitFlags.IsCompressed)] = true;
                        },
                    }
                } else if (isWExtensionWith2EntryDecomposition(raw_opcode, raw_funct3, @truncate(instr_word >> 25))) {
                    // W-extension instructions that decompose to base + VirtualSignExtendWord:
                    // ADDIW (0x1b/f3=0), ADDW (0x3b/f3=0/f7=0), SUBW (0x3b/f3=0/f7=0x20),
                    // MULW (0x3b/f3=0/f7=1), SLLW (0x3b/f3=1/f7=0), etc.
                    // pc_map has max_inline_seq=1, so k = base_pc+1.
                    // k is the VirtualSignExtendWord position; k-1 is the base instruction.
                    //
                    // Populate VirtualSignExtendWord at k (= base_pc+1, last in sequence)
                    // is_compressed: only on last instruction per Jolt finalize
                    populateVirtualSignExtendWordEntry(&entries[k], raw_rd, addr, is_compressed);
                    // Populate the base instruction at k-1 (= base_pc, first in sequence)
                    // The base instruction is the W-extension's non-W equivalent:
                    // ADDIW → ADDI, ADDW → ADD, SUBW → SUB, MULW → MUL
                    if (k >= 1) {
                        // Build the base instruction word by mapping the opcode:
                        // 0x1b (OP-IMM-32) → 0x13 (OP-IMM), 0x3b (OP-32) → 0x33 (OP)
                        const base_opcode: u8 = if (raw_opcode == 0x1b) 0x13 else 0x33;
                        var base_instr = (instr_word & ~@as(u32, 0x7F)) | @as(u32, base_opcode);
                        // Normalize funct7 for OP-32 → OP mapping (same logic as populateEntryFromInstruction)
                        if (raw_opcode == 0x3b) {
                            const raw_f7: u7 = @truncate(instr_word >> 25);
                            const canon_f7: u7 = switch (raw_funct3) {
                                0 => if (raw_f7 == 0x20) @as(u7, 0x20) else if (raw_f7 == 0x01) @as(u7, 0x01) else 0,
                                5 => if (raw_f7 == 0x20) @as(u7, 0x20) else 0,
                                1, 2, 3, 4, 6, 7 => if (raw_f7 == 0x01) @as(u7, 0x01) else 0,
                            };
                            base_instr = (base_instr & ~(@as(u32, 0x7F) << 25)) | (@as(u32, canon_f7) << 25);
                        }
                        populateEntryFromInstruction(&entries[k - 1], base_instr, addr);
                        // Set virtual sequence flags for the base (first) instruction:
                        // vsr=Some(1): 1 instruction remaining
                        // VirtualInstruction=true (vsr.is_some())
                        // DoNotUpdateUnexpandedPC=true (vsr=1 != 0)
                        // IsFirstInSequence=true (first in sequence)
                        entries[k - 1].circuit_flags[@intFromEnum(CircuitFlags.VirtualInstruction)] = true;
                        entries[k - 1].circuit_flags[@intFromEnum(CircuitFlags.DoNotUpdateUnexpandedPC)] = true;
                        entries[k - 1].circuit_flags[@intFromEnum(CircuitFlags.IsFirstInSequence)] = true;
                        entries[k - 1].virtual_sequence_remaining = 1;
                        entries[k - 1].is_first_in_sequence = true;
                    }
                } else if (raw_opcode == 0x0B) {
                    // Jolt inline instruction (0x0B) — already expanded by Phase 1 preprocessing.
                    // Skip to avoid overwriting the correct entries with a single wrong one.
                } else if (raw_opcode == 0x73 and (raw_funct3 == 1 or raw_funct3 == 2 or
                    (raw_funct3 == 0 and ((instr_word >> 20) & 0xFFF) == 0x302)))
                {
                    // CSR instructions (CSRRW funct3=1, CSRRS funct3=2) and MRET (funct3=0, funct12=0x302)
                    // — already expanded by preprocessing into ADDI/OR/JALR virtual sequences.
                    // Skip to avoid overwriting with a single ECALL-like entry.
                } else if (raw_opcode == 0x5B and raw_funct3 != 0 and raw_funct3 != 5) {
                    // Jolt SDK VirtualHostIO instructions (opcode 0x5B, funct3 != 0 and != 5).
                    // funct3=0 is VirtualSRLI and funct3=5 is VirtualSRAI (Zolt's own virtual instructions).
                    // Other funct3 values (1=AssertEQ, 2=HostIO, 3-7=advice loads) come from the
                    // Jolt SDK in the ELF. These are NOP-like (no lookup, no register writes).
                    // Phase 1 preprocessing handles them as UNIMPL with correct flags.
                    // Skip to avoid misprocessing them as VirtualSRLI/VirtualSRAI.
                } else {
                    populateEntryFromInstruction(&entries[k], instr_word, addr);

                    // Upstream Jolt remaps JAL/JALR with rd=x0 to virtual register 40.
                    // The bytecode entry must match the trace step's rd_index.
                    const opcode_byte: u8 = @truncate(instr_word & 0x7F);
                    const rd_from_instr: u8 = @truncate((instr_word >> 7) & 0x1F);
                    if ((opcode_byte == 0x6F or opcode_byte == 0x67) and rd_from_instr == 0) {
                        entries[k].rd = 40; // FIRST_VIRTUAL_ALLOC_REG
                        entries[k].instruction_flags[@intFromEnum(InstructionFlags.IsRdNotZero)] = true;
                    }

                    // Mark compressed instructions
                    if (is_compressed) {
                        entries[k].circuit_flags[@intFromEnum(CircuitFlags.IsCompressed)] = true;
                    }
                }
            }

            offset += instr_size;
        }
    }

    // ================================================================
    // Phase 2: Populate termination sequence bytecode entries
    // ================================================================
    // Each termination instruction (LUI, ADDI, SB, JAL) gets its own bytecode entry
    // at indices termination_base_pc, +1, +2, +3.
    // LUI/ADDI: VirtualInstruction=true, DoNotUpdateUnexpandedPC=true (vsr>0)
    // SB anchor: VirtualInstruction=true, DoNotUpdateUnexpandedPC=false (vsr=Some(0))
    //   — matches vanilla Jolt's circuit_flags for SD with vsr=Some(0)
    // JAL: normal instruction (Jump=1 disables NextUPC constraints for JAL→NoOp)
    {
        const tbpc = pc_map.termination_base_pc;
        if (tbpc > 0 and tbpc + 3 < bytecode_K) {
            const upper20: u32 = @truncate((termination_address >> 12) & 0xFFFFF);
            const lower12: u32 = @truncate(termination_address & 0xFFF);
            const imm_upper7: u32 = (lower12 >> 5) & 0x7F;
            const imm_lower5: u32 = lower12 & 0x1F;

            // LUI x31, upper20(addr)
            const lui_word: u32 = (upper20 << 12) | (31 << 7) | 0x37;
            // ADDI x30, x0, 1
            const addi_word: u32 = (1 << 20) | (0 << 15) | (0 << 12) | (30 << 7) | 0x13;
            // SB x30, lower12(addr)(x31)
            const sb_word: u32 = (imm_upper7 << 25) | (30 << 20) | (31 << 15) | (0 << 12) | (imm_lower5 << 7) | 0x23;
            // JAL x0, 0 (j . = infinite loop)
            const jal_word: u32 = 0x0000006F;

            // Entry at tbpc: LUI x31 (virtual, vsr=2)
            // from_raw_word(lui_word, 0) + VirtualInstruction + DoNotUpdateUnexpandedPC
            populateEntryFromInstruction(&entries[tbpc], lui_word, 0);
            entries[tbpc].circuit_flags[@intFromEnum(CircuitFlags.VirtualInstruction)] = true;
            entries[tbpc].circuit_flags[@intFromEnum(CircuitFlags.DoNotUpdateUnexpandedPC)] = true;
            entries[tbpc].virtual_sequence_remaining = 2;
            entries[tbpc].is_first_in_sequence = false;

            // Entry at tbpc+1: ADDI x30 (virtual, vsr=1)
            populateEntryFromInstruction(&entries[tbpc + 1], addi_word, 0);
            entries[tbpc + 1].circuit_flags[@intFromEnum(CircuitFlags.VirtualInstruction)] = true;
            entries[tbpc + 1].circuit_flags[@intFromEnum(CircuitFlags.DoNotUpdateUnexpandedPC)] = true;
            entries[tbpc + 1].virtual_sequence_remaining = 1;
            entries[tbpc + 1].is_first_in_sequence = false;

            // Entry at tbpc+2: SB x30, lower12(x31) (anchor, vsr=Some(0))
            // Matches vanilla Jolt's circuit_flags for SD with vsr=Some(0):
            //   VirtualInstruction = vsr.is_some() = true
            //   DoNotUpdateUnexpandedPC = vsr.map_or(false, |v| v > 0) = false
            // R1CS witness also sets VI=true, DNUPC=false for this step.
            // Constraint 17: NextPC = tbpc+2+1 = tbpc+3 (JAL entry) ✓
            // Constraint 16: NextUPC = 0+4 = 4 (JAL has UPC=4) ✓
            populateEntryFromInstruction(&entries[tbpc + 2], sb_word, 0);
            entries[tbpc + 2].circuit_flags[@intFromEnum(CircuitFlags.VirtualInstruction)] = true;
            // DoNotUpdateUnexpandedPC stays false (default)
            entries[tbpc + 2].virtual_sequence_remaining = 0;
            entries[tbpc + 2].is_first_in_sequence = false;

            // Entry at tbpc+3: JAL x0, 0 (j . = infinite loop)
            // Normal instruction entry: Jump=1 from populateEntryFromInstruction.
            // address=4 (synthetic) so UPC=4 satisfies SB's constraint 16.
            // No VirtualInstruction, no DoNotUpdateUnexpandedPC.
            // Jump=1 disables constraint 16 for JAL→NoOp (condition=1-0-1=0).
            // ShouldJump=Jump*(1-NextIsNoop)=0 disables constraint 14.
            populateEntryFromInstruction(&entries[tbpc + 3], jal_word, 4);
            entries[tbpc + 3].virtual_sequence_remaining = null;
            entries[tbpc + 3].is_first_in_sequence = false;
            // JAL x0 remapped to vr40 (upstream inline_sequence)
            entries[tbpc + 3].rd = 40;
            entries[tbpc + 3].instruction_flags[@intFromEnum(InstructionFlags.IsRdNotZero)] = true;

            dbg("[PHASE2] Termination entries at tbpc={d}: LUI=0x{x:0>8} ADDI=0x{x:0>8} SB=0x{x:0>8} JAL=0x{x:0>8}\n", .{ tbpc, lui_word, addi_word, sb_word, jal_word });
        }
    }

    // Post-processing: set IsLastInSequence for JALR entries with virtual_sequence_remaining == 0
    // Upstream Jolt only sets this flag on JALR instructions (opcode 0x67), not on all instructions.
    for (0..bytecode_K) |k| {
        if (entries[k].virtual_sequence_remaining) |vsr| {
            if (vsr == 0 and entries[k].opcode == 0x67) {
                entries[k].circuit_flags[@intFromEnum(CircuitFlags.IsLastInSequence)] = true;
            }
        }
    }

    // Debug output removed

    // ================================================================
    // Phase 3: Sync with preprocessing to ensure bytecode entries match
    // ================================================================
    // The preprocessing (serialized to the verifier) may have different entries
    // at indices corresponding to data bytes (.rodata) that both decoders interpret
    // as UNIMPL but with different instruction alignment. Sync all entries where
    // the preprocessing and raw-byte path disagree on address or variant.
    if (bytecode_preprocessing) |prep| {
        var sync_count: usize = 0;
        for (0..@min(bytecode_K, prep.bytecode.items.len)) |k| {
            const prep_instr = prep.bytecode.items[k];
            const prep_addr = prep_instr.address;
            const is_prep_noop = (prep_instr.variant == .NoOp);
            const is_prep_unimpl = (prep_instr.variant == .UNIMPL);
            const entry_addr = entries[k].address;

            // If addresses match and preprocessing is not UNIMPL, entries are consistent
            if (prep_addr == entry_addr and !is_prep_unimpl and !is_prep_noop) continue;
            // If both are NoOp/UNIMPL with same address, skip
            if (prep_addr == entry_addr and entry_addr == 0) continue;

            sync_count += 1;

            // Addresses differ — sync to match preprocessing
            if (is_prep_noop) {
                // Preprocessing has NoOp but we have something else → make it NoOp
                var cf = [_]bool{false} ** 14;
                cf[@intFromEnum(CircuitFlags.DoNotUpdateUnexpandedPC)] = true;
                var inf = [_]bool{false} ** 7;
                inf[@intFromEnum(InstructionFlags.IsNoop)] = true;
                entries[k] = BytecodeEntry{
                    .address = 0,
                    .imm = 0,
                    .rd = 255,
                    .rs1 = 255,
                    .rs2 = 255,
                    .circuit_flags = cf,
                    .instruction_flags = inf,
                    .lookup_table_index = 255,
                    .is_interleaved = true,
                    .virtual_sequence_remaining = null,
                    .is_first_in_sequence = false,
                    .opcode = 0,
                    .funct3 = 0,
                };
            } else if (is_prep_unimpl) {
                // Preprocessing has UNIMPL — set entry to UNIMPL (address=0, matching Jolt's Default)
                entries[k].address = 0;
                entries[k].imm = 0;
                entries[k].rd = 255;
                entries[k].rs1 = 255;
                entries[k].rs2 = 255;
                entries[k].circuit_flags = [_]bool{false} ** 14;
                entries[k].instruction_flags = [_]bool{false} ** 7;
                entries[k].lookup_table_index = 255;
                entries[k].is_interleaved = true;
                entries[k].virtual_sequence_remaining = null;
                entries[k].is_first_in_sequence = false;
                entries[k].opcode = 0;
                entries[k].funct3 = 0;
            } else {
                // Real instruction in preprocessing but prover has different/NoOp.
                // Build a synthetic instruction word from the preprocessing variant
                // and use populateEntryFromInstruction to ensure consistent flag computation.
                const synth = buildSyntheticWordFromPrep(prep_instr);
                if (synth.word != 0) {
                    populateEntryFromInstruction(&entries[k], synth.word, prep_addr);
                    // Apply virtual sequence metadata
                    applyVirtualAndCompressedFlags(&entries[k], synth.rd_full, synth.rs1_full, synth.rs2_full, prep_instr.virtual_sequence_remaining, prep_instr.is_first_in_sequence, prep_instr.is_compressed);
                } else {
                    // Unknown variant — use populateEntryFromJoltInstruction as fallback
                    populateEntryFromJoltInstruction(&entries[k], prep_instr);
                }
            }
        }
    }

    // Debug comparison removed

    return entries;
}

/// Build a synthetic 32-bit instruction word from a preprocessing JoltInstruction.
/// Returns .word=0 if the variant is not supported.
fn buildSyntheticWordFromPrep(instr: preprocessing.JoltInstruction) struct { word: u32, rd_full: u8, rs1_full: u8, rs2_full: u8 } {
    var rd: u8 = 0;
    var rs1: u8 = 0;
    var rs2: u8 = 0;
    var imm: i64 = 0;
    switch (instr.operands) {
        .FormatR => |r| { rd = r.rd; rs1 = r.rs1; rs2 = r.rs2; },
        .FormatI => |i_op| { rd = i_op.rd; rs1 = i_op.rs1; imm = @bitCast(i_op.imm); },
        .FormatLoad => |l| { rd = l.rd; rs1 = l.rs1; imm = l.imm; },
        .FormatS => |s| { rs1 = s.rs1; rs2 = s.rs2; imm = s.imm; },
        .FormatJ => |j| { rd = j.rd; imm = @bitCast(j.imm); },
        .FormatU => |u_op| { rd = u_op.rd; imm = @bitCast(u_op.imm); },
        else => {},
    }
    const rd_full = rd;
    const rs1_full = rs1;
    const rs2_full = rs2;

    const word: u32 = switch (instr.variant) {
        // I-type: ADDI, XORI, ORI, ANDI, SLTI, SLTIU
        .ADDI => buildIType(@bitCast(imm), rs1, 0, rd, 0x13),
        .XORI => buildIType(@bitCast(imm), rs1, 4, rd, 0x13),
        .ORI => buildIType(@bitCast(imm), rs1, 6, rd, 0x13),
        .ANDI => buildIType(@bitCast(imm), rs1, 7, rd, 0x13),
        // R-type: ADD, SUB, XOR, OR, AND, etc.
        .ADD => buildRType(0, rs2, rs1, 0, rd, 0x33),
        .SUB => buildRType(0x20, rs2, rs1, 0, rd, 0x33),
        .XOR => buildRType(0, rs2, rs1, 4, rd, 0x33),
        .OR => buildRType(0, rs2, rs1, 6, rd, 0x33),
        .AND => buildRType(0, rs2, rs1, 7, rd, 0x33),
        .MUL => buildRType(1, rs2, rs1, 0, rd, 0x33),
        // Jump: JALR, JAL
        .JALR => blk_jalr: {
            const raw_rd = if (rd == 40) @as(u8, 0) else rd;
            break :blk_jalr buildIType(@bitCast(imm), rs1, 0, raw_rd, 0x67);
        },
        .JAL => blk_jal: {
            const raw_rd = if (rd == 40) @as(u8, 0) else rd;
            break :blk_jal buildJType(@bitCast(imm), raw_rd, 0x6F);
        },
        // Load/Store
        .LD => buildIType(@as(u64, @bitCast(imm)), rs1, 3, rd, 0x03),
        .SD => buildSType(imm, rs2, rs1, 3, 0x23),
        // Virtual
        .VirtualSignExtendWord => buildIType(0, rs1, 0, rd, 0x0B),
        .VirtualZeroExtendWord => buildIType(0, rs1, 0, rd, 0x42),
        .VirtualMULI => buildIType(@bitCast(imm), rs1, 0, rd, 0x2B),
        .VirtualSRLI => blk_srli: {
            const bitmask: u64 = @bitCast(imm);
            const total_shift: u7 = if (bitmask == 0) 0 else @intCast(@ctz(bitmask));
            break :blk_srli buildIType(@as(u64, total_shift), rs1, 0, rd, 0x5B);
        },
        // SDK instructions
        .VirtualHostIO => (@as(u32, 2) << 12) | (@as(u32, rs1 & 0x1F) << 15) | (@as(u32, rd & 0x1F) << 7) | 0x5B,
        else => 0, // Unsupported
    };
    return .{ .word = word, .rd_full = rd_full, .rs1_full = rs1_full, .rs2_full = rs2_full };
}

/// Check if a raw instruction is a W-extension that decomposes into 2 entries:
/// base_instruction + VirtualSignExtendWord. This includes ADDIW, ADDW, SUBW, MULW.
/// Excludes SLLIW (handled separately with VirtualMULI decomposition).
fn isWExtensionWith2EntryDecomposition(opcode: u8, funct3: u3, funct7: u7) bool {
    _ = funct7;
    return switch (opcode) {
        0x1b => funct3 == 0, // ADDIW (funct3=0); SLLIW (funct3=1) excluded
        0x3b => funct3 == 0,
        else => false,
    };
}

/// Check if an opcode/funct3/funct7 combination is a known RISC-V instruction
/// recognized by Jolt. Unknown combinations are treated as UNIMPL (Default).
fn isKnownInstruction(opcode: u8, funct3: u3, funct7: u7) bool {
    switch (opcode) {
        0x33 => return switch (funct3) { // R-type
            0 => (funct7 == 0 or funct7 == 0x20 or funct7 == 0x01), // ADD, SUB, MUL
            1, 2, 3, 4, 6 => (funct7 == 0 or funct7 == 0x01), // SLL, SLT, SLTU, XOR, OR + M-ext
            5 => (funct7 == 0 or funct7 == 0x20 or funct7 == 0x01), // SRL, SRA, DIVU
            7 => (funct7 == 0 or funct7 == 0x20 or funct7 == 0x01), // AND, ANDN, REMU
        },
        0x3b => return switch (funct3) { // OP-32
            0 => (funct7 == 0 or funct7 == 0x20 or funct7 == 0x01), // ADDW, SUBW, MULW
            1 => (funct7 == 0), // SLLW
            4, 5, 6, 7 => (funct7 == 0x01), // DIVW, DIVUW, REMW, REMUW
            2, 3 => false,
        },
        0x13 => return switch (funct3) { // OP-IMM
            0, 1, 2, 3, 4, 5, 6, 7 => true, // ADDI, SLLI, SLTI, SLTIU, XORI, SRLI/SRAI, ORI, ANDI
        },
        0x1b => return switch (funct3) { // OP-IMM-32
            0 => true, // ADDIW
            1 => true, // SLLIW
            5 => true, // SRLIW/SRAIW
            else => false,
        },
        0x03,
        0x23,
        0x63,
        0x37,
        0x17,
        0x6F,
        0x67, // Standard opcodes
        0x73,
        0x0F, // ECALL, FENCE (treated as NoOp in Jolt)
        => return true,
        // Jolt SDK / virtual instructions at opcode 0x5B
        //   funct3=0 → VirtualRev8W (real instruction in jolt-inlines/sha2 ELFs)
        //   funct3=2 → VirtualHostIO
        //   funct3=3,4,5,6 → AdviceLB/LH/LW/LD
        //   funct3=7 → VirtualAdviceLen
        // funct3=1 is unused in pinned Jolt 997c1543 (newer revs use it for VirtualAssertEQ).
        0x5B => return funct3 != 1,
        // 0x7B is our internal synthetic opcode for VirtualRev8W trace cycles
        0x7B => return true,
        // Other virtual opcodes only appear in virtual sequence entries, not ELF bytes.
        else => return false,
    }
}

/// Map a RISC-V instruction to its lookup table index (0..40).
/// Returns 255 if no lookup table is used.
/// Must match Jolt's LookupTables enum discriminant ordering:
///   0=RangeCheck, 1=RangeCheckAligned, 2=And, 3=Andn, 4=Or, 5=Xor,
///   6=Equal, 7=SignedGTE, 8=UnsignedGTE, 9=NotEqual, 10=SignedLT,
///   11=UnsignedLT, 12=Movsign, 13=UpperWord, 14=UnsignedLTE,
///   15=ValidUnsignedRemainder, 16=ValidDiv0,
///   17=HalfwordAlignment, 18=WordAlignment, 19=LowerHalfWord,
///   20=SignExtendHalfWord, 21=Pow2, 22=Pow2W, 23=ShiftRightBitmask,
///   24=VirtualRev8W, 25=VirtualSRL, 26=VirtualSRA, 27=VirtualROTR,
///   28=VirtualROTRW, 29=VirtualChangeDivisor, 30=VirtualChangeDivisorW,
///   31=MulUNoOverflow, 32-39=VirtualXORROT variants
pub fn getLookupTableIndex(opcode: u8, funct3: u3, funct7: u7) u8 {
    return switch (opcode) {
        0x33 => switch (funct3) { // R-type
            0 => if (funct7 == 0) @as(u8, 0) // ADD → RangeCheck
            else if (funct7 == 0x20) 0 // SUB → RangeCheck
            else if (funct7 == 0x01) 0 // MUL → RangeCheck
            else 255,
            7 => if (funct7 == 0) @as(u8, 2) // AND → And
            else if (funct7 == 0x20) 3 // ANDN → Andn
            else if (funct7 == 0x01) 13 // REMU → (was UpperWord, should be ValidUnsignedRemainder but decomposed)
            else 255,
            6 => if (funct7 == 0) @as(u8, 4) // OR → Or
            else 255,
            4 => if (funct7 == 0) @as(u8, 5) // XOR → Xor
            else 255,
            1 => 255, // SLL - decomposed to virtual instructions
            5 => 255, // SRL/SRA - decomposed to virtual instructions
            2 => 10, // SLT → SignedLessThan
            3 => if (funct7 == 0x01) @as(u8, 13) // MULHU → UpperWord
            else 11, // SLTU → UnsignedLessThan
        },
        0x13 => switch (funct3) { // I-type ALU
            0 => 0, // ADDI → RangeCheck
            7 => 2, // ANDI → And
            6 => 4, // ORI → Or
            4 => 5, // XORI → Xor
            1 => 255, // SLLI - decomposed to virtual instructions
            5 => 255, // SRLI/SRAI - decomposed to virtual instructions
            2 => 10, // SLTI → SignedLessThan
            3 => 11, // SLTIU → UnsignedLessThan
        },
        0x63 => switch (funct3) { // Branches
            0 => 6, // BEQ → Equal
            1 => 9, // BNE → NotEqual
            4 => 10, // BLT → SignedLessThan
            5 => 7, // BGE → SignedGreaterThanEqual
            6 => 11, // BLTU → UnsignedLessThan
            7 => 8, // BGEU → UnsignedGreaterThanEqual
            2, 3 => 255, // unused branch funct3 values
        },
        0x37 => 0, // LUI → RangeCheck
        0x17 => 0, // AUIPC → RangeCheck
        0x6F => 0, // JAL → RangeCheck
        0x67 => 1, // JALR → RangeCheckAligned
        0x1b => if (funct3 == 0) @as(u8, 0) else 255, // ADDIW → RangeCheck
        0x3b => switch (funct3) { // OP-32
            0 => if (funct7 == 0) @as(u8, 0) // ADDW → RangeCheck
            else if (funct7 == 0x20) 0 // SUBW → RangeCheck
            else 255,
            6 => 30, // VirtualChangeDivisorW → VirtualChangeDivisorW table
            else => 255,
        },
        0x0B => 20, // VirtualSignExtendWord → SignExtendHalfWord
        0x2B => switch (funct3) { // Virtual I-type
            1 => 21, // VirtualPow2 → Pow2
            2 => 23, // VirtualShiftRightBitmask → ShiftRightBitmask
            else => 0, // VirtualMULI (funct3=0) → RangeCheck
        },
        0x5B => switch (funct3) {
            // NOTE: opcode 0x5B funct3=0/5 is shared between two distinct cases:
            //   - Internal: VirtualSRLI/VirtualSRL (table 25), VirtualSRAI/VirtualSRA (table 26)
            //   - External (Jolt 997c1543): VirtualRev8W (funct3=0, table 24), AdviceLW (funct3=5, no table)
            // The raw-decode path here matches our internal usage (VirtualSRL family). External
            // VirtualRev8W from a real ELF is handled by the prep-first path via the
            // .VirtualRev8W variant in populateEntryFromJoltInstruction (which sets table 24).
            0 => @as(u8, 25), // VirtualSRLI/VirtualSRL → VirtualSRL
            5 => @as(u8, 26), // VirtualSRAI/VirtualSRA → VirtualSRA
            else => 255, // VirtualHostIO/Advice* — no lookup table at raw decode
        },
        0x02 => 0, // VirtualAdvice → RangeCheck
        0x22 => switch (funct3) { // Virtual assert
            1 => 16, // VirtualAssertValidDiv0 → ValidDiv0
            2 => 17, // VirtualAssertHalfwordAlignment → HalfwordAlignment
            3 => 18, // VirtualAssertWordAlignment → WordAlignment
            else => 6, // VirtualAssertEQ → Equal
        },
        0x42 => 19, // VirtualZeroExtendWord → LowerHalfWord
        0x62 => 15, // VirtualAssertValidUnsignedRemainder → ValidUnsignedRemainder
        0x6B => if (funct3 == 0) @as(u8, 27) // VirtualROTRI → VirtualROTR
        else if (funct3 == 1) 28 // VirtualROTRIW → VirtualROTRW
        else 255,
        0x7B => 24, // VirtualRev8W (internal synthetic) → VirtualRev8W table
        else => 255, // Load, Store, ECALL, FENCE - no lookup table
    };
}

/// Check if an instruction has a lookup table assignment
pub fn hasLookupTable(opcode: u8, funct3: u3, funct7: u7) bool {
    return getLookupTableIndex(opcode, funct3, funct7) != 255;
}
