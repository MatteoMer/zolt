//! Stage 6 Batched Sumcheck Prover
//!
//! Stage 6 is a batched sumcheck with 6 instances:
//! 0. BytecodeReadRaf: bytecode_log_k + n_cycle_vars rounds, degree bytecode_d + 1
//! 1. Booleanity: log_k_chunk + n_cycle_vars rounds, degree 3 (input_claim = 0)
//! 2. HammingBooleanity: n_cycle_vars rounds, degree 3 (input_claim = 0)
//! 3. RamRaVirtual: n_cycle_vars rounds, degree ram_d + 1
//! 4. LookupsRaVirtual: n_cycle_vars rounds, degree n_committed_per_virtual + 1
//! 5. IncClaimReduction: n_cycle_vars rounds, degree 2
//!
//! ALL instances use real sumcheck provers with actual polynomial materialization
//! from execution trace data. No shortcuts, no placeholders.

const std = @import("std");

// Debug output control - set to true to enable verbose debug prints
const debug_verbose = false;

// Maximum evaluation points for parallelReduce accumulator.
// Covers all sub-provers: LookupsRa (M+2 ≤ 10), RamRa (d+2 ≤ 6), BytecodeReadRaf (d+2 ≤ 4).
const MAX_RA_EVALS = 16;
fn dbg(comptime fmt: []const u8, args: anytype) void {
    if (debug_verbose) std.debug.print(fmt, args);
}

const Allocator = std.mem.Allocator;
const ThreadPool = @import("../../utils/thread_pool.zig").ThreadPool;

const poly_mod = @import("../../poly/mod.zig");
const UniPoly = poly_mod.UniPoly;
const transcripts = @import("../../transcripts/mod.zig");
const Blake2bTranscript = transcripts.Blake2bTranscript;
const jolt_types = @import("../jolt_types.zig");
const SumcheckInstanceProof = jolt_types.SumcheckInstanceProof;
const OpeningClaims = jolt_types.OpeningClaims;
const OpeningId = jolt_types.OpeningId;
const tracer = @import("../../tracer/mod.zig");
const ExecutionTrace = tracer.ExecutionTrace;
const ram = @import("../ram/mod.zig");
const jolt_device = @import("../jolt_device.zig");
const instruction_mod = @import("../instruction/mod.zig");
const CircuitFlags = instruction_mod.CircuitFlags;
const InstructionFlags = instruction_mod.InstructionFlags;
const preprocessing = @import("../preprocessing.zig");
const BytecodePCMapper = preprocessing.BytecodePCMapper;

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
    /// Lookup table index (0..41, or 255 for no lookup table)
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
/// LeftOperandIsRs1Value, RightOperandIsImm, lookup table = SignExtendHalfWord (21).
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

    entry.lookup_table_index = 21; // SignExtendHalfWord
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

    entry.lookup_table_index = 21; // SignExtendHalfWord
    entry.is_interleaved = false;
    entry.virtual_sequence_remaining = virtual_sequence_remaining;
    entry.is_first_in_sequence = is_first_in_sequence;
}

/// Populate a BytecodeEntry for a VirtualSRLI instruction.
/// VirtualSRLI has opcode 0x5B with: WriteLookupOutputToRD (NO AddOperands, NO MultiplyOperands),
/// LeftOperandIsRs1Value, RightOperandIsImm, lookup table = VirtualSRL (26).
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
    // It uses interleaved operands with VirtualSRL table (table index 26).
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

    entry.lookup_table_index = 26; // VirtualSRL
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

    entry.lookup_table_index = 20; // LowerHalfWord
    // AddOperands set → identity-path (not interleaved)
    entry.is_interleaved = false;
    entry.virtual_sequence_remaining = virtual_sequence_remaining;
    entry.is_first_in_sequence = is_first_in_sequence;
}

/// Populate a BytecodeEntry for a VirtualAssertValidUnsignedRemainder instruction (opcode 0x62).
/// VirtualAssertValidUnsignedRemainder: Assert flag set, LeftOperandIsRs1Value, RightOperandIsRs2Value.
/// Lookup table = ValidUnsignedRemainder (16), interleaved-path.
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

    entry.lookup_table_index = 16; // ValidUnsignedRemainder
    // No AddOperands/SubtractOperands/MultiplyOperands/Advice → interleaved
    entry.is_interleaved = true;
    entry.virtual_sequence_remaining = virtual_sequence_remaining;
    entry.is_first_in_sequence = is_first_in_sequence;
}

/// Populate a BytecodeEntry for a VirtualAssertValidDiv0 instruction.
/// Assert + VirtualInstruction flags. Lookup table = ValidDiv0 (17).
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
    if (virtual_sequence_remaining) |vsr| { if (vsr != 0) cf[@intFromEnum(CircuitFlags.DoNotUpdateUnexpandedPC)] = true; }
    if (is_first_in_sequence) cf[@intFromEnum(CircuitFlags.IsFirstInSequence)] = true;
    var inf = &entry.instruction_flags;
    inf[@intFromEnum(InstructionFlags.LeftOperandIsRs1Value)] = true;
    inf[@intFromEnum(InstructionFlags.RightOperandIsRs2Value)] = true;
    entry.lookup_table_index = 17; // ValidDiv0
    entry.is_interleaved = true;
    entry.virtual_sequence_remaining = virtual_sequence_remaining;
    entry.is_first_in_sequence = is_first_in_sequence;
}

/// Populate a BytecodeEntry for a VirtualChangeDivisorW instruction (R-format).
/// WriteLookupOutputToRD + VirtualInstruction flags. Lookup table = VirtualChangeDivisorW (31).
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
    if (virtual_sequence_remaining) |vsr| { if (vsr != 0) cf[@intFromEnum(CircuitFlags.DoNotUpdateUnexpandedPC)] = true; }
    if (is_first_in_sequence) cf[@intFromEnum(CircuitFlags.IsFirstInSequence)] = true;
    var inf = &entry.instruction_flags;
    inf[@intFromEnum(InstructionFlags.LeftOperandIsRs1Value)] = true;
    inf[@intFromEnum(InstructionFlags.RightOperandIsRs2Value)] = true;
    if (rd != 0) inf[@intFromEnum(InstructionFlags.IsRdNotZero)] = true;
    entry.lookup_table_index = 31; // VirtualChangeDivisorW
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
/// lookup table = VirtualSRA (27), is_interleaved = true.
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
    entry.lookup_table_index = 27; // VirtualSRA
    entry.is_interleaved = true; // No Add/Sub/Mul/Advice flags
    entry.virtual_sequence_remaining = virtual_sequence_remaining;
    entry.is_first_in_sequence = is_first_in_sequence;
}

/// Populate a BytecodeEntry from a raw 32-bit instruction word and ELF address.
/// This sets all static properties (flags, registers, immediates, lookup table)
/// from the instruction encoding alone, without any trace-specific data.
fn populateEntryFromInstruction(entry: *BytecodeEntry, instr: u32, elf_address: u64) void {
    const decoded = instruction_mod.DecodedInstruction.decode(instr);

    entry.address = elf_address;
    entry.imm = @intCast(decoded.imm);

    const opcode: u8 = @truncate(instr & 0x7F);

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
    entry.rd = if (opcode == 0x23 or opcode == 0x63) 255 else decoded.rd;
    entry.rs1 = switch (opcode) {
        0x37, 0x17, 0x6f => 255, // U-type, J-type: no rs1
        else => decoded.rs1,
    };
    entry.rs2 = switch (opcode) {
        0x13, 0x03, 0x67, 0x1b, 0x37, 0x17, 0x6f, 0x0B, 0x2B, 0x5B => 255, // I-type, U-type, J-type, Virtual: no rs2
        else => decoded.rs2,
    };
    const funct3: u3 = @truncate((instr >> 12) & 0x7);
    const funct7: u7 = @truncate(instr >> 25);

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
            else => {},
        }
    }

    // Instruction flags
    var inf = &entry.instruction_flags;

    // LeftOperandIsPC
    if (has_lookup and (opcode == 0x17 or opcode == 0x6F)) {
        inf[@intFromEnum(InstructionFlags.LeftOperandIsPC)] = true;
    }

    // LeftOperandIsRs1Value
    if (has_lookup) {
        switch (opcode) {
            0x33, 0x13, 0x67, 0x63, 0x1B, 0x3B, 0x0B, 0x2B, 0x5B => {
                inf[@intFromEnum(InstructionFlags.LeftOperandIsRs1Value)] = true;
            },
            else => {},
        }
    }

    // RightOperandIsImm
    if (has_lookup) {
        switch (opcode) {
            0x13, 0x67, 0x37, 0x17, 0x6F, 0x1B, 0x0B, 0x2B, 0x5B => {
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

    // Lookup table index and interleaving
    entry.lookup_table_index = getLookupTableIndex(opcode, funct3, funct7);
    entry.is_interleaved = !cf[@intFromEnum(CircuitFlags.AddOperands)] and
        !cf[@intFromEnum(CircuitFlags.SubtractOperands)] and
        !cf[@intFromEnum(CircuitFlags.MultiplyOperands)] and
        !cf[@intFromEnum(CircuitFlags.Advice)];
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
    // Phase 1: Populate from static ELF code bytes
    // ================================================================
    // This fills in ALL instructions from the program, including those
    // that are never executed in a particular trace. The bytecode array
    // is: [NoOp at k=0] [instruction at addr0] [instruction at addr1] ...
    // Index 0 is always the NoOp/padding entry.
    if (program_code_bytes) |code_bytes| {
        var offset: usize = 0;
        while (offset < code_bytes.len) {
            const addr = code_base_address + offset;

            // Check if compressed (RVC)
            if (offset + 2 > code_bytes.len) break;
            const first_halfword: u16 = std.mem.readInt(u16, code_bytes[offset..][0..2], .little);
            const is_compressed = (first_halfword & 0x3) != 0x3;

            var instr_word: u32 = undefined;
            var instr_size: usize = undefined;

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
                    // is_compressed: NOT set on non-last instructions (Jolt only sets on last)
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
                    if (k >= 11) populateVirtualAdviceEntry(&entries[k - 11], a2, addr, 11, true);
                    // Step 2 (k-10): VirtualAdvice(a3) → remainder (vsr=10)
                    if (k >= 10) populateVirtualAdviceEntry(&entries[k - 10], a3, addr, 10, false);
                    // Step 3 (k-9): VirtualZeroExtendWord(t3, a2) → zero-extend quotient (vsr=9)
                    if (k >= 9) populateVirtualZeroExtendWordEntry(&entries[k - 9], t3, a2, addr, 9, false);
                    // Step 4 (k-8): VirtualZeroExtendWord(t1, rs1) → zero-extend dividend (vsr=8)
                    if (k >= 8) populateVirtualZeroExtendWordEntry(&entries[k - 8], t1, raw_rs1, addr, 8, false);
                    // Step 5 (k-7): VirtualZeroExtendWord(t2, rs2) → zero-extend divisor (vsr=7)
                    if (k >= 7) populateVirtualZeroExtendWordEntry(&entries[k - 7], t2, raw_rs2, addr, 7, false);
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
                    if (k >= 5) populateVirtualZeroExtendWordEntry(&entries[k - 5], t4, t0, addr, 5, false);
                    // Step 8 (k-4): VirtualAssertEQ(t4, t0) → assert no overflow (vsr=4)
                    if (k >= 4) populateVirtualAssertEQEntry(&entries[k - 4], t4, t0, addr, 4, false);
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
                    if (k >= 2) populateVirtualAssertEQEntry(&entries[k - 2], t0, t1, addr, 2, false);
                    // Step 11 (k-1): VirtualAssertValidUnsignedRemainder(a3, t2) → r < d (vsr=1)
                    if (k >= 1) populateVirtualAssertValidUnsignedRemainderEntry(&entries[k - 1], a3, t2, addr, 1, false);
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
                    if (k >= 20) populateVirtualAdviceEntry(&entries[k - 20], a2, addr, 20, true);
                    // Step 2 (k-19): VirtualAdvice(a3) → |remainder| (vsr=19)
                    if (k >= 19) populateVirtualAdviceEntry(&entries[k - 19], a3, addr, 19, false);
                    // Step 3 (k-18): VirtualSignExtendWord(t4, rs1) → sign-extend dividend (vsr=18)
                    if (k >= 18) {
                        populateVirtualSignExtendWordEntryWithParams(&entries[k - 18], t4, raw_rs1, addr, 18, false);
                    }
                    // Step 4 (k-17): VirtualSignExtendWord(t3, rs2) → sign-extend divisor (vsr=17)
                    if (k >= 17) {
                        populateVirtualSignExtendWordEntryWithParams(&entries[k - 17], t3, raw_rs2, addr, 17, false);
                    }
                    // Step 5 (k-16): VirtualAssertValidDiv0(t3, a2) → handle div-by-zero (vsr=16)
                    if (k >= 16) populateVirtualAssertValidDiv0Entry(&entries[k - 16], t3, a2, addr, 16, false);
                    // Step 6 (k-15): VirtualChangeDivisorW(t0, t4, t3) → handle overflow (vsr=15)
                    if (k >= 15) populateVirtualChangeDivisorWEntry(&entries[k - 15], t0, t4, t3, addr, 15, false);
                    // Step 7 (k-14): VirtualSignExtendWord(t1, a2) → sign-extend quotient (vsr=14)
                    if (k >= 14) {
                        populateVirtualSignExtendWordEntryWithParams(&entries[k - 14], t1, a2, addr, 14, false);
                    }
                    // Step 8 (k-13): VirtualAssertEQ(t1, a2) → assert quotient fits 32 bits (vsr=13)
                    if (k >= 13) populateVirtualAssertEQEntry(&entries[k - 13], t1, a2, addr, 13, false);
                    // VirtualSRAI bitmask for shift=31: ((1<<33)-1) << 31 = 0xFFFFFFFF80000000
                    const srai_bitmask: u64 = blk: {
                        const shift_amt: u7 = 31;
                        const ones: u128 = (@as(u128, 1) << @intCast(64 - @as(u8, shift_amt))) - 1;
                        break :blk @truncate(ones << shift_amt);
                    };
                    // Step 9 (k-12): VirtualSRAI(t2, a3, bitmask) → sign bit of |remainder| (vsr=12)
                    if (k >= 12) populateVirtualSRAIEntry(&entries[k - 12], t2, a3, addr, srai_bitmask, 12, false);
                    // Step 10 (k-11): VirtualAssertEQ(t2, 0) → assert non-negative (vsr=11)
                    if (k >= 11) populateVirtualAssertEQEntry(&entries[k - 11], t2, 0, addr, 11, false);
                    // Step 11 (k-10): VirtualSRAI(t2, t4, bitmask) → sign bit of dividend (vsr=10)
                    if (k >= 10) populateVirtualSRAIEntry(&entries[k - 10], t2, t4, addr, srai_bitmask, 10, false);
                    // Step 12 (k-9): XOR(t3, a3, t2) → XOR |remainder| with sign mask (vsr=9)
                    if (k >= 9) populateVirtualRTypeEntry(&entries[k - 9], t3, a3, t2, addr, 9, false, 0x33, 4, 0);
                    // Step 13 (k-8): SUB(t3, t3, t2) → sign-corrected remainder (vsr=8)
                    if (k >= 8) populateVirtualRTypeEntry(&entries[k - 8], t3, t3, t2, addr, 8, false, 0x33, 0, 0x20);
                    // Step 14 (k-7): MUL(t1, a2, t0) → quotient × adjusted_divisor (vsr=7)
                    if (k >= 7) populateVirtualRTypeEntry(&entries[k - 7], t1, a2, t0, addr, 7, false, 0x33, 0, 0x01);
                    // Step 15 (k-6): ADD(t1, t1, t3) → + remainder (vsr=6)
                    if (k >= 6) populateVirtualRTypeEntry(&entries[k - 6], t1, t1, t3, addr, 6, false, 0x33, 0, 0);
                    // Step 16 (k-5): VirtualAssertEQ(t1, t4) → assert dividend = q*d + r (vsr=5)
                    if (k >= 5) populateVirtualAssertEQEntry(&entries[k - 5], t1, t4, addr, 5, false);
                    // Step 17 (k-4): VirtualSRAI(t2, t0, bitmask) → sign bit of adjusted divisor (vsr=4)
                    if (k >= 4) populateVirtualSRAIEntry(&entries[k - 4], t2, t0, addr, srai_bitmask, 4, false);
                    // Step 18 (k-3): XOR(t1, t0, t2) → (vsr=3)
                    if (k >= 3) populateVirtualRTypeEntry(&entries[k - 3], t1, t0, t2, addr, 3, false, 0x33, 4, 0);
                    // Step 19 (k-2): SUB(t1, t1, t2) → abs(divisor) (vsr=2)
                    if (k >= 2) populateVirtualRTypeEntry(&entries[k - 2], t1, t1, t2, addr, 2, false, 0x33, 0, 0x20);
                    // Step 20 (k-1): VirtualAssertValidUnsignedRemainder(a3, t1) → |r| < |d| (vsr=1)
                    if (k >= 1) populateVirtualAssertValidUnsignedRemainderEntry(&entries[k - 1], a3, t1, addr, 1, false);
                    // Step 21 (k): VirtualSignExtendWord(rd, output) → sign-extend result (vsr=0, last)
                    // REMW: output = t3 (signed remainder), DIVW: output = a2 (quotient)
                    const output_reg = if (raw_funct3 == 6) t3 else a2;
                    populateVirtualSignExtendWordEntry(&entries[k], raw_rd, addr, is_compressed);
                    entries[k].rs1 = output_reg;
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
                        const base_instr = (instr_word & ~@as(u32, 0x7F)) | @as(u32, base_opcode);
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
                        // is_compressed: NOT set on non-last instructions (Jolt only sets on last)
                    }
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

    // Debug: dump first entries
    dbg("\n[ZOLT BYTECODE ENTRIES] bytecode_K={}\n", .{bytecode_K});
    for (0..@min(bytecode_K, 32)) |k| {
        const e = entries[k];
        // Compute a compact representation: cf_bits, if_bits
        var cf_bits: u16 = 0;
        for (0..14) |i| {
            if (e.circuit_flags[i]) cf_bits |= @as(u16, 1) << @intCast(i);
        }
        var if_bits: u8 = 0;
        for (0..7) |i| {
            if (e.instruction_flags[i]) if_bits |= @as(u8, 1) << @intCast(i);
        }
        dbg("  entry[{d:2}]: addr=0x{x:0>8} rd={d:2} rs1={d:2} rs2={d:2} imm={d:6} cf=0x{x:04} if=0x{x:02} lt={d:3} interl={}\n", .{
            k, e.address, e.rd, e.rs1, e.rs2, e.imm, cf_bits, if_bits, e.lookup_table_index, @intFromBool(e.is_interleaved),
        });
    }
    dbg("\n", .{});

    return entries;
}

/// Check if a raw instruction is a W-extension that decomposes into 2 entries:
/// base_instruction + VirtualSignExtendWord. This includes ADDIW, ADDW, SUBW, MULW.
/// Excludes SLLIW (handled separately with VirtualMULI decomposition).
fn isWExtensionWith2EntryDecomposition(opcode: u8, funct3: u3, funct7: u7) bool {
    return switch (opcode) {
        0x1b => funct3 == 0, // ADDIW (funct3=0); SLLIW (funct3=1) excluded
        0x3b => switch (funct3) {
            0 => (funct7 == 0 or funct7 == 0x20 or funct7 == 0x01), // ADDW, SUBW, MULW
            else => false,
        },
        else => false,
    };
}

/// Map a RISC-V instruction to its lookup table index (0..40).
/// Returns 255 if no lookup table is used.
/// Must match Jolt's LookupTables enum discriminant ordering:
///   0=RangeCheck, 1=RangeCheckAligned, 2=And, 3=Andn, 4=Or, 5=Xor,
///   6=Equal, 7=SignedGTE, 8=UnsignedGTE, 9=NotEqual, 10=SignedLT,
///   11=UnsignedLT, 12=Movsign, 13=UpperWord, 14=UnsignedLTE,
///   15=ValidSignedRemainder, 16=ValidUnsignedRemainder, 17=ValidDiv0,
///   18=HalfwordAlignment, 19=WordAlignment, 20=LowerHalfWord,
///   21=SignExtendHalfWord, 22=Pow2, 23=Pow2W, 24=ShiftRightBitmask,
///   25=VirtualRev8W, 26=VirtualSRL, 27=VirtualSRA, 28=VirtualROTR,
///   29=VirtualROTRW, 30=VirtualChangeDivisor, 31=VirtualChangeDivisorW,
///   32=MulUNoOverflow, 33-40=VirtualXORROT variants
fn getLookupTableIndex(opcode: u8, funct3: u3, funct7: u7) u8 {
    return switch (opcode) {
        0x33 => switch (funct3) { // R-type
            0 => if (funct7 == 0) @as(u8, 0) // ADD → RangeCheck
            else if (funct7 == 0x20) 0 // SUB → RangeCheck
            else if (funct7 == 0x01) 0 // MUL → RangeCheck
            else 255,
            7 => if (funct7 == 0) @as(u8, 2) // AND → And
            else if (funct7 == 0x01) 13 // MULHU → UpperWord
            else 255,
            6 => if (funct7 == 0) @as(u8, 4) // OR → Or
            else 255,
            4 => if (funct7 == 0) @as(u8, 5) // XOR → Xor
            else 255,
            1 => 255, // SLL - decomposed to virtual instructions
            5 => 255, // SRL/SRA - decomposed to virtual instructions
            2 => 10, // SLT → SignedLessThan
            3 => 11, // SLTU → UnsignedLessThan
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
            6 => 31, // VirtualChangeDivisorW → VirtualChangeDivisorW table
            else => 255,
        },
        0x0B => 21, // VirtualSignExtendWord → SignExtendHalfWord
        0x2B => 0, // VirtualMULI → RangeCheck
        0x5B => if (funct3 == 5) @as(u8, 27) else 26, // VirtualSRAI → VirtualSRA, VirtualSRLI → VirtualSRL
        0x02 => 0, // VirtualAdvice → RangeCheck
        0x22 => if (funct3 == 1) @as(u8, 17) else 6, // VirtualAssertValidDiv0 → ValidDiv0, VirtualAssertEQ → Equal
        0x42 => 20, // VirtualZeroExtendWord → LowerHalfWord
        0x62 => 16, // VirtualAssertValidUnsignedRemainder → ValidUnsignedRemainder
        else => 255, // Load, Store, ECALL, FENCE - no lookup table
    };
}

/// Check if an instruction has a lookup table assignment
fn hasLookupTable(opcode: u8, funct3: u3, funct7: u7) bool {
    return getLookupTableIndex(opcode, funct3, funct7) != 255;
}

/// Result of Stage 6 sumcheck
pub fn Stage6Result(comptime F: type) type {
    return struct {
        const Self = @This();

        /// All sumcheck challenges (stage6_max_rounds elements)
        challenges: []F,

        /// BytecodeReadRaf opening claims: BytecodeRa(i) for i in 0..bytecode_d
        bytecode_ra_claims: []F,

        /// HammingBooleanity opening claim: RamHammingWeight
        hamming_weight_claim: F,

        /// Booleanity opening claims: all RA polys [InstructionRa(0..inst_d), BytecodeRa(0..bc_d), RamRa(0..ram_d)]
        booleanity_ra_claims: []F,

        /// RamRaVirtualization opening claims: RamRa(i) for i in 0..ram_d
        ram_ra_virtual_claims: []F,

        /// InstructionRaVirtualization opening claims: InstructionRa(i) for i in 0..instruction_d
        instruction_ra_virtual_claims: []F,

        /// IncClaimReduction opening claims: [RamInc, RdInc]
        ram_inc_claim: F,
        rd_inc_claim: F,

        /// Stage 6 configuration for Stage 7 opening point extraction
        bytecode_log_k: usize,
        log_k_chunk: usize,
        n_cycle_vars: usize,
        bytecode_d: usize,
        ram_d: usize,
        instruction_d: usize,

        allocator: Allocator,

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.challenges);
            self.allocator.free(self.bytecode_ra_claims);
            self.allocator.free(self.booleanity_ra_claims);
            self.allocator.free(self.ram_ra_virtual_claims);
            self.allocator.free(self.instruction_ra_virtual_claims);
        }
    };
}

// =============================================================================
// IncClaimReduction Sumcheck Instance (Instance 5)
// =============================================================================
// Proves: Sigma_j [RamInc(j) * eq_ram_combined(j) + gamma^2 * RdInc(j) * eq_rd_combined(j)] = input_claim
// where eq_ram_combined = eq(r_stage2, j) + gamma * eq(r_stage4, j)
//       eq_rd_combined  = eq(s_stage4, j) + gamma * eq(s_stage5, j)
// Degree 2: product of two linear polys (Inc x eq)
fn IncClaimReductionProver(comptime F: type) type {
    return struct {
        const Self = @This();

        ram_inc: []F,
        rd_inc: []F,
        eq_ram: []F,
        eq_rd: []F,
        gamma_sqr: F,
        current_len: usize,
        allocator: Allocator,
        pool: ?*ThreadPool = null,

        pub fn init(
            allocator: Allocator,
            trace: *const ExecutionTrace,
            gamma: F,
            r_cycle_stage2: []const F,
            r_cycle_stage4: []const F,
            s_cycle_stage4: []const F,
            s_cycle_stage5: []const F,
            pool: ?*ThreadPool,
        ) !Self {
            const T = trace.steps.items.len;
            const n_vars = std.math.log2_int(usize, T);

            var ram_inc_arr = try allocator.alloc(F, T);
            var rd_inc_arr = try allocator.alloc(F, T);

            // Uses TraceStep pre-recorded rd_pre_value — no sequential register tracking needed.
            // Must match Stage 4 gruen prover's inc_poly computation exactly.
            for (0..T) |j| {
                const step = trace.steps.items[j];

                // RdInc: skip rd=0 (x0 hardwired to 0, inc always 0)
                if (!step.is_noop and step.rd_written and step.rd_index != 0) {
                    rd_inc_arr[j] = F.fromU64(step.rd_value).sub(F.fromU64(step.rd_pre_value));
                } else {
                    rd_inc_arr[j] = F.zero();
                }

                // RamInc = memory_value - memory_pre_value (only for writes)
                if (step.is_memory_write) {
                    const mem_post: i128 = @intCast(step.memory_value orelse 0);
                    const mem_pre: i128 = @intCast(step.memory_pre_value orelse 0);
                    ram_inc_arr[j] = fieldFromI128(F, mem_post - mem_pre);
                } else {
                    ram_inc_arr[j] = F.zero();
                }
            }

            // All r_cycle inputs are in BE order; reverse for LE computeEqTable
            var rev_buf = try allocator.alloc(F, n_vars);
            defer allocator.free(rev_buf);

            for (0..n_vars) |i| rev_buf[i] = r_cycle_stage2[n_vars - 1 - i];
            const eq_stage2 = try computeEqTableParallel(F, allocator, rev_buf, n_vars, pool);
            defer allocator.free(eq_stage2);

            for (0..n_vars) |i| rev_buf[i] = r_cycle_stage4[n_vars - 1 - i];
            const eq_stage4 = try computeEqTableParallel(F, allocator, rev_buf, n_vars, pool);
            defer allocator.free(eq_stage4);

            for (0..n_vars) |i| rev_buf[i] = s_cycle_stage4[n_vars - 1 - i];
            const eq_s4 = try computeEqTableParallel(F, allocator, rev_buf, n_vars, pool);
            defer allocator.free(eq_s4);

            for (0..n_vars) |i| rev_buf[i] = s_cycle_stage5[n_vars - 1 - i];
            const eq_s5 = try computeEqTableParallel(F, allocator, rev_buf, n_vars, pool);
            defer allocator.free(eq_s5);

            var eq_ram_arr = try allocator.alloc(F, T);
            var eq_rd_arr = try allocator.alloc(F, T);

            for (0..T) |j| {
                eq_ram_arr[j] = eq_stage2[j].add(gamma.mul(eq_stage4[j]));
                eq_rd_arr[j] = eq_s4[j].add(gamma.mul(eq_s5[j]));
            }

            return Self{
                .ram_inc = ram_inc_arr,
                .rd_inc = rd_inc_arr,
                .eq_ram = eq_ram_arr,
                .eq_rd = eq_rd_arr,
                .gamma_sqr = gamma.mul(gamma),
                .current_len = T,
                .allocator = allocator,
                .pool = pool,
            };
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.ram_inc);
            self.allocator.free(self.rd_inc);
            self.allocator.free(self.eq_ram);
            self.allocator.free(self.eq_rd);
        }

        /// Compute round polynomial evaluations at [0, 2, inf]
        pub fn computeRoundPoly(self: *Self) [3]F {
            const half = self.current_len / 2;

            const Ctx = struct {
                ram_inc: []const F,
                rd_inc: []const F,
                eq_ram: []const F,
                eq_rd: []const F,
                gamma_sqr: F,
            };
            const ctx = Ctx{
                .ram_inc = self.ram_inc,
                .rd_inc = self.rd_inc,
                .eq_ram = self.eq_ram,
                .eq_rd = self.eq_rd,
                .gamma_sqr = self.gamma_sqr,
            };

            const mapFn = struct {
                fn f(c: Ctx, start: usize, end: usize) [3]F {
                    var e0 = F.zero();
                    var e1 = F.zero();
                    var e2 = F.zero();
                    for (start..end) |j| {
                        const ram0 = c.ram_inc[2 * j];
                        const ram1 = c.ram_inc[2 * j + 1];
                        const ram_delta = ram1.sub(ram0);
                        const eq_r0 = c.eq_ram[2 * j];
                        const eq_r1 = c.eq_ram[2 * j + 1];
                        const eq_r_delta = eq_r1.sub(eq_r0);

                        const rd0 = c.rd_inc[2 * j];
                        const rd1 = c.rd_inc[2 * j + 1];
                        const rd_delta = rd1.sub(rd0);
                        const eq_d0 = c.eq_rd[2 * j];
                        const eq_d1 = c.eq_rd[2 * j + 1];
                        const eq_d_delta = eq_d1.sub(eq_d0);

                        e0 = e0.add(ram0.mul(eq_r0).add(c.gamma_sqr.mul(rd0.mul(eq_d0))));
                        e1 = e1.add(ram1.mul(eq_r1).add(c.gamma_sqr.mul(rd1.mul(eq_d1))));

                        const two = F.fromU64(2);
                        const ram2 = ram0.add(two.mul(ram_delta));
                        const eq_r2 = eq_r0.add(two.mul(eq_r_delta));
                        const rd2 = rd0.add(two.mul(rd_delta));
                        const eq_d2 = eq_d0.add(two.mul(eq_d_delta));
                        e2 = e2.add(ram2.mul(eq_r2).add(c.gamma_sqr.mul(rd2.mul(eq_d2))));
                    }
                    return [3]F{ e0, e1, e2 };
                }
            }.f;

            const reduceFn = struct {
                fn f(a: [3]F, b: [3]F) [3]F {
                    return [3]F{ a[0].add(b[0]), a[1].add(b[1]), a[2].add(b[2]) };
                }
            }.f;

            if (self.pool) |pool| {
                return pool.parallelReduce([3]F, half, [3]F{ F.zero(), F.zero(), F.zero() }, ctx, mapFn, reduceFn);
            }

            // Fallback: sequential
            return mapFn(ctx, 0, half);
        }

        pub fn bindChallenge(self: *Self, r: F) void {
            const half = self.current_len / 2;

            const bindOne = struct {
                fn f(arr: []F, h: usize, challenge: F) void {
                    for (0..h) |j| {
                        arr[j] = arr[2 * j].add(challenge.mul(arr[2 * j + 1].sub(arr[2 * j])));
                    }
                }
            }.f;

            if (self.pool) |pool| {
                const arrays = [4][]F{ self.ram_inc, self.rd_inc, self.eq_ram, self.eq_rd };
                const Ctx = struct { arrs: [4][]F, half: usize, r: F };
                const ctx = Ctx{ .arrs = arrays, .half = half, .r = r };
                pool.parallelForForce(4, ctx, struct {
                    fn f(c: Ctx, idx: usize) void {
                        bindOne(c.arrs[idx], c.half, c.r);
                    }
                }.f);
            } else {
                bindOne(self.ram_inc, half, r);
                bindOne(self.rd_inc, half, r);
                bindOne(self.eq_ram, half, r);
                bindOne(self.eq_rd, half, r);
            }

            self.current_len = half;
        }

        pub fn openingClaims(self: *const Self) struct { ram_inc: F, rd_inc: F } {
            return .{ .ram_inc = self.ram_inc[0], .rd_inc = self.rd_inc[0] };
        }
    };
}

// =============================================================================
// HammingBooleanity Sumcheck Instance (Instance 1)
// =============================================================================
// Proves: Sigma_j eq(r_cycle, j) * (H(j)^2 - H(j)) = 0
// Degree 3: eq is linear * (H^2 - H is quadratic)
fn HammingBooleanityProver(comptime F: type) type {
    return struct {
        const Self = @This();

        H: []F,
        eq: []F,
        current_len: usize,
        allocator: Allocator,
        pool: ?*ThreadPool = null,

        pub fn init(
            allocator: Allocator,
            trace: *const ExecutionTrace,
            r_cycle: []const F,
            pool: ?*ThreadPool,
        ) !Self {
            const T = trace.steps.items.len;
            const n_vars = std.math.log2_int(usize, T);

            var H_arr = try allocator.alloc(F, T);
            for (0..T) |j| {
                const step = trace.steps.items[j];
                if (step.memory_addr) |addr| {
                    H_arr[j] = if (addr != 0) F.one() else F.zero();
                } else {
                    H_arr[j] = F.zero();
                }
            }

            // r_cycle is in BE order; reverse for LE computeEqTable
            var r_cycle_rev = try allocator.alloc(F, n_vars);
            defer allocator.free(r_cycle_rev);
            for (0..n_vars) |i| r_cycle_rev[i] = r_cycle[n_vars - 1 - i];
            const eq_arr = try computeEqTableParallel(F, allocator, r_cycle_rev, n_vars, pool);

            return Self{
                .H = H_arr,
                .eq = eq_arr,
                .current_len = T,
                .allocator = allocator,
                .pool = pool,
            };
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.H);
            self.allocator.free(self.eq);
        }

        /// Compute round polynomial at [0, 1, 2, inf]
        pub fn computeRoundPoly(self: *Self) [4]F {
            const half = self.current_len / 2;

            const Ctx = struct {
                H: []const F,
                eq: []const F,
            };
            const ctx = Ctx{ .H = self.H, .eq = self.eq };

            const mapFn = struct {
                fn f(c: Ctx, start: usize, end: usize) [4]F {
                    var e0 = F.zero();
                    var e1 = F.zero();
                    var e2 = F.zero();
                    var e3 = F.zero();
                    for (start..end) |j| {
                        const h0 = c.H[2 * j];
                        const h1 = c.H[2 * j + 1];
                        const h_delta = h1.sub(h0);

                        const eq0 = c.eq[2 * j];
                        const eq1 = c.eq[2 * j + 1];
                        const e_delta = eq1.sub(eq0);

                        e0 = e0.add(eq0.mul(h0.mul(h0).sub(h0)));
                        e1 = e1.add(eq1.mul(h1.mul(h1).sub(h1)));

                        const two = F.fromU64(2);
                        const h_at_2 = h0.add(two.mul(h_delta));
                        const e_at_2 = eq0.add(two.mul(e_delta));
                        e2 = e2.add(e_at_2.mul(h_at_2.mul(h_at_2).sub(h_at_2)));

                        const three = F.fromU64(3);
                        const h_at_3 = h0.add(three.mul(h_delta));
                        const e_at_3 = eq0.add(three.mul(e_delta));
                        e3 = e3.add(e_at_3.mul(h_at_3.mul(h_at_3).sub(h_at_3)));
                    }
                    return [4]F{ e0, e1, e2, e3 };
                }
            }.f;

            const reduceFn = struct {
                fn f(a: [4]F, b: [4]F) [4]F {
                    return [4]F{ a[0].add(b[0]), a[1].add(b[1]), a[2].add(b[2]), a[3].add(b[3]) };
                }
            }.f;

            if (self.pool) |pool| {
                return pool.parallelReduce([4]F, half, [4]F{ F.zero(), F.zero(), F.zero(), F.zero() }, ctx, mapFn, reduceFn);
            }

            return mapFn(ctx, 0, half);
        }

        pub fn bindChallenge(self: *Self, r: F) void {
            const half = self.current_len / 2;

            const bindOne = struct {
                fn f(arr: []F, h: usize, challenge: F) void {
                    for (0..h) |j| {
                        arr[j] = arr[2 * j].add(challenge.mul(arr[2 * j + 1].sub(arr[2 * j])));
                    }
                }
            }.f;

            if (self.pool) |pool| {
                const arrays = [2][]F{ self.H, self.eq };
                const Ctx = struct { arrs: [2][]F, half: usize, r: F };
                const ctx = Ctx{ .arrs = arrays, .half = half, .r = r };
                pool.parallelForForce(2, ctx, struct {
                    fn f(c: Ctx, idx: usize) void {
                        bindOne(c.arrs[idx], c.half, c.r);
                    }
                }.f);
            } else {
                bindOne(self.H, half, r);
                bindOne(self.eq, half, r);
            }

            self.current_len = half;
        }

        pub fn openingClaim(self: *const Self) F {
            return self.H[0];
        }
    };
}

// =============================================================================
// RamRaVirtual Sumcheck Instance (Instance 3)
// =============================================================================
// Proves: Sigma_c eq(r_cycle_reduced, c) * Prod_{i=0}^{d-1} ra_i(r_addr_chunk_i, c) = claim
// Variables: n_cycle_vars
// Degree: d+1 (product of d linear ra_i * one linear eq)
fn RamRaVirtualProver(comptime F: type) type {
    return struct {
        const Self = @This();

        /// ra_bound[i][j] = eq(r_addr_chunk_i, addr_chunk_i(j))
        ra_bound: [][]F,
        /// eq(r_cycle_reduced, .) evaluations
        eq: []F,
        d: usize,
        current_len: usize,
        allocator: Allocator,
        pool: ?*ThreadPool = null,

        pub fn init(
            allocator: Allocator,
            trace: *const ExecutionTrace,
            r_cycle: []const F, // BIG_ENDIAN
            r_addr_chunks: []const []const F, // r_addr_chunks[i] is BIG_ENDIAN
            d: usize,
            memory_layout: *const jolt_device.MemoryLayout,
            log_k_chunk: usize,
            init_pool: ?*ThreadPool,
        ) !Self {
            const T = trace.steps.items.len;
            const n_vars = std.math.log2_int(usize, T);
            const k_chunk: usize = @as(usize, 1) << @intCast(log_k_chunk);

            var ra_bound_arr = try allocator.alloc([]F, d);
            errdefer {
                for (ra_bound_arr[0..d]) |arr| allocator.free(arr);
                allocator.free(ra_bound_arr);
            }

            for (0..d) |i| {
                ra_bound_arr[i] = try allocator.alloc(F, T);

                // r_addr_chunks[i] is in BE order; reverse for LE computeEqTable
                // Small table (chunk-sized), no parallelism needed
                var r_chunk_rev = try allocator.alloc(F, log_k_chunk);
                defer allocator.free(r_chunk_rev);
                for (0..log_k_chunk) |ci| r_chunk_rev[ci] = r_addr_chunks[i][log_k_chunk - 1 - ci];
                const eq_table = try computeEqTable(F, allocator, r_chunk_rev, log_k_chunk);
                defer allocator.free(eq_table);

                for (0..T) |j| {
                    const step = trace.steps.items[j];
                    if (step.memory_addr) |addr| {
                        if (addr == 0) {
                            // No memory access - remap_address returns None for addr=0
                            ra_bound_arr[i][j] = F.zero();
                        } else {
                            const remapped = memory_layout.remapAddress(addr);
                            if (remapped) |raddr| {
                                // MSB-first chunk extraction: chunk 0 = MSB
                                const chunk_val = extractChunkMSB(raddr, i, d, log_k_chunk);
                                if (chunk_val < k_chunk) {
                                    ra_bound_arr[i][j] = eq_table[chunk_val];
                                } else {
                                    ra_bound_arr[i][j] = F.zero();
                                }
                            } else {
                                ra_bound_arr[i][j] = F.zero();
                            }
                        }
                    } else {
                        // No memory access at this cycle
                        ra_bound_arr[i][j] = F.zero();
                    }
                }
            }

            // r_cycle is in BE order; reverse for LE computeEqTable
            var r_cycle_rev = try allocator.alloc(F, n_vars);
            defer allocator.free(r_cycle_rev);
            for (0..n_vars) |i| r_cycle_rev[i] = r_cycle[n_vars - 1 - i];
            const eq_arr = try computeEqTableParallel(F, allocator, r_cycle_rev, n_vars, init_pool);

            return Self{
                .ra_bound = ra_bound_arr,
                .eq = eq_arr,
                .d = d,
                .current_len = T,
                .allocator = allocator,
                .pool = init_pool,
            };
        }

        pub fn deinit(self: *Self) void {
            for (self.ra_bound) |arr| self.allocator.free(arr);
            self.allocator.free(self.ra_bound);
            self.allocator.free(self.eq);
        }

        /// Compute round polynomial evaluations
        /// f(x) = eq(x) * Prod_i ra_i(x), degree = d + 1
        /// Need d+2 evaluation points: [0, 1, 2, ..., d, inf]
        pub fn computeRoundPoly(self: *Self, allocator: Allocator) ![]F {
            const half = self.current_len / 2;
            const n_evals = self.d + 2;

            // Precompute x_vals
            var x_vals: [MAX_RA_EVALS]F = undefined;
            for (0..n_evals) |i| {
                x_vals[i] = F.fromU64(@intCast(i));
            }

            const Ctx = struct {
                ra_bound: [][]F,
                eq: []F,
                d: usize,
                n_evals: usize,
                x_vals: [MAX_RA_EVALS]F,
            };
            const ctx = Ctx{
                .ra_bound = self.ra_bound,
                .eq = self.eq,
                .d = self.d,
                .n_evals = n_evals,
                .x_vals = x_vals,
            };

            const mapFn = struct {
                fn f(c: Ctx, start: usize, end: usize) [MAX_RA_EVALS]F {
                    var acc: [MAX_RA_EVALS]F = .{F.zero()} ** MAX_RA_EVALS;
                    for (start..end) |j| {
                        const eq0 = c.eq[2 * j];
                        const eq1 = c.eq[2 * j + 1];
                        const eq_delta = eq1.sub(eq0);

                        for (0..c.n_evals) |pt_idx| {
                            const x = c.x_vals[pt_idx];
                            var product = F.one();

                            for (0..c.d) |i| {
                                const v0 = c.ra_bound[i][2 * j];
                                const v1 = c.ra_bound[i][2 * j + 1];
                                product = product.mul(v0.add(x.mul(v1.sub(v0))));
                            }
                            product = product.mul(eq0.add(x.mul(eq_delta)));

                            acc[pt_idx] = acc[pt_idx].add(product);
                        }
                    }
                    return acc;
                }
            }.f;

            const reduceFn = struct {
                fn f(a: [MAX_RA_EVALS]F, b: [MAX_RA_EVALS]F) [MAX_RA_EVALS]F {
                    var r: [MAX_RA_EVALS]F = undefined;
                    for (0..MAX_RA_EVALS) |i| {
                        r[i] = a[i].add(b[i]);
                    }
                    return r;
                }
            }.f;

            const result = if (self.pool) |pool|
                pool.parallelReduce([MAX_RA_EVALS]F, half, .{F.zero()} ** MAX_RA_EVALS, ctx, mapFn, reduceFn)
            else
                mapFn(ctx, 0, half);

            var evals = try allocator.alloc(F, n_evals);
            for (0..n_evals) |i| {
                evals[i] = result[i];
            }
            return evals;
        }

        pub fn bindChallenge(self: *Self, r: F) void {
            const half = self.current_len / 2;

            const bindOne = struct {
                fn f(arr: []F, h: usize, challenge: F) void {
                    for (0..h) |j| {
                        arr[j] = arr[2 * j].add(challenge.mul(arr[2 * j + 1].sub(arr[2 * j])));
                    }
                }
            }.f;

            if (self.pool) |pool| {
                // d+1 independent arrays: d ra_bound arrays + 1 eq array
                const total = self.d + 1;
                const Ctx = struct { ra: [][]F, eq: []F, d: usize, half: usize, r: F };
                const ctx = Ctx{ .ra = self.ra_bound, .eq = self.eq, .d = self.d, .half = half, .r = r };
                pool.parallelForForce(total, ctx, struct {
                    fn f(c: Ctx, idx: usize) void {
                        if (idx < c.d) {
                            bindOne(c.ra[idx], c.half, c.r);
                        } else {
                            bindOne(c.eq, c.half, c.r);
                        }
                    }
                }.f);
            } else {
                for (0..self.d) |i| {
                    bindOne(self.ra_bound[i], half, r);
                }
                bindOne(self.eq, half, r);
            }

            self.current_len = half;
        }

        pub fn getOpeningClaims(self: *const Self, allocator: Allocator) ![]F {
            var claims = try allocator.alloc(F, self.d);
            for (0..self.d) |i| {
                claims[i] = self.ra_bound[i][0];
            }
            return claims;
        }
    };
}

// =============================================================================
// Booleanity Sumcheck Instance (Instance 2) - REAL prover
// =============================================================================
// Proves: 0 = Σ_{k,j} eq(r_addr, k) · eq(r_cycle, j) · Σ_i γ^{2i} · (ra_i(k,j)² - ra_i(k,j))
//
// Phase 1: log_k_chunk address rounds (degree 3)
//   Uses G tables (full size K, never halved), expanding table F, and split-eq B.
//   At round m: G stays full, F has size 2^m, B tracks eq(r_addr_fixed, ...).
//   p(X_m) = l(X) * q(X) where l = eq linear part, q = Σ γ^{2i} * G*F*(G*F-1)
//
// Phase 2: n_cycle_vars cycle rounds (degree 3)
//   Uses H tables (initialized from F at transition, halved each round) and eq_cycle D.
//   H[i][j] = eq(r_addr_bound, chunk_i(j)), scaled by eq_r_r.
//
// r_addr and r_cycle are FIXED reference points from Stage 5 InstructionReadRaf.
// G_i[k] = Σ_j eq(r_cycle_fixed, j) * [chunk_i(j) == k]  (pushforward)
fn BooleanityProver(comptime F: type) type {
    return struct {
        const Self = @This();

        /// G_i tables (pushforward): G_i[k] = Σ_j eq(r_cycle, j) * [chunk_i(j) == k]
        /// Stays at FULL size K throughout Phase 1 (never halved).
        G: [][]F,
        /// Expanding table F: F[k] = eq(r_bound_so_far, k). Starts size 1, doubles each round.
        F_table: []F,
        /// Current size of F_table
        F_size: usize,
        /// r_address (LE, LowToHigh order) - fixed reference point
        r_address_le: []F,
        /// B_scalar: accumulated eq(r_addr_fixed[bound_vars], r_challenges[bound_vars])
        B_scalar: F,
        /// eq(r_cycle, j) table for Phase 2 - halves each round
        eq_cycle: []F,
        /// γ^{2i} powers for batching
        gamma_powers_sq: []F,
        /// Number of RA polynomials
        N: usize,
        /// K = 2^log_k_chunk (address table size)
        K: usize,
        /// log_k_chunk (address rounds)
        log_k_chunk: usize,
        /// n_cycle_vars (cycle rounds)
        n_cycle_vars: usize,
        /// Current round number (0-indexed)
        round: usize,
        /// eq(r_addr_fixed, r_addr_bound) - set at Phase 1→2 transition
        eq_r_r: F,
        /// H tables for Phase 2: H[i][j] = eq(r_addr_bound, chunk_i(j))
        /// Initialized at Phase 1→2 transition. Halved each Phase 2 round.
        H: ?[][]F,
        /// Current table length for Phase 2 (T, then T/2, etc.)
        phase2_len: usize,
        /// Trace reference for building H tables at transition
        trace: *const ExecutionTrace,
        /// Parameters needed for H table construction
        instruction_d: usize,
        bytecode_d: usize,
        ram_d: usize,
        memory_layout: *const jolt_device.MemoryLayout,
        pc_map: *const BytecodePCMapper,
        allocator: std.mem.Allocator,
        pool: ?*ThreadPool = null,

        pub fn init(
            allocator: std.mem.Allocator,
            G_tables: [][]F,
            r_addr_le: []F,
            eq_cycle_table: []F,
            gamma_sq: []F,
            N_val: usize,
            log_k: usize,
            n_cycle: usize,
            trace: *const ExecutionTrace,
            instr_d: usize,
            bc_d: usize,
            ram_d_val: usize,
            mem_layout: *const jolt_device.MemoryLayout,
            pc_mapper: *const BytecodePCMapper,
        ) !Self {
            const K_val = @as(usize, 1) << @intCast(log_k);
            // Initialize expanding table F with F[0] = 1
            const f_table = try allocator.alloc(F, K_val);
            @memset(f_table, F.zero());
            f_table[0] = F.one();

            // Reverse r_addr_le to match Jolt's binding order (MSB first).
            // Jolt's GruenSplitEqPolynomial binds variables from high index to low:
            //   round 0 binds w[n-1] (MSB), round 1 binds w[n-2], ..., round n-1 binds w[0] (LSB).
            // The G table inner loop uses bit m of k at round m, paired with the
            // eq factor from w[n-1-m]. By reversing, r_addr[m] = w[n-1-m].
            // Debug: print BEFORE reversal
            {
                for (0..log_k) |dbg_i| {
                    const dbg_b = r_addr_le[dbg_i].toBytesBE();
                    dbg("[BOOL_INIT] r_addr_BEFORE[{}] LE=[{x:0>2}{x:0>2}{x:0>2}{x:0>2}]\n", .{
                        dbg_i, dbg_b[31], dbg_b[30], dbg_b[29], dbg_b[28],
                    });
                }
            }

            std.mem.reverse(F, r_addr_le);

            // Debug: print AFTER reversal
            {
                for (0..log_k) |dbg_i| {
                    const dbg_b = r_addr_le[dbg_i].toBytesBE();
                    dbg("[BOOL_INIT] r_addr_AFTER[{}] LE=[{x:0>2}{x:0>2}{x:0>2}{x:0>2}]\n", .{
                        dbg_i, dbg_b[31], dbg_b[30], dbg_b[29], dbg_b[28],
                    });
                }
            }

            return Self{
                .G = G_tables,
                .F_table = f_table,
                .F_size = 1,
                .r_address_le = r_addr_le,
                .B_scalar = F.one(),
                .eq_cycle = eq_cycle_table,
                .gamma_powers_sq = gamma_sq,
                .N = N_val,
                .K = K_val,
                .log_k_chunk = log_k,
                .n_cycle_vars = n_cycle,
                .round = 0,
                .eq_r_r = F.zero(),
                .H = null,
                .phase2_len = 0,
                .trace = trace,
                .instruction_d = instr_d,
                .bytecode_d = bc_d,
                .ram_d = ram_d_val,
                .memory_layout = mem_layout,
                .pc_map = pc_mapper,
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            for (self.G) |g| self.allocator.free(g);
            self.allocator.free(self.G);
            self.allocator.free(self.F_table);
            self.allocator.free(self.r_address_le);
            self.allocator.free(self.eq_cycle);
            self.allocator.free(self.gamma_powers_sq);
            if (self.H) |ht| {
                for (ht) |h| self.allocator.free(h);
                self.allocator.free(ht);
            }
        }

        /// Get the opening claims from the final H state after all sumcheck rounds.
        /// H[i][0] gives ra_i(ρ_addr, ρ_cycle) after all bindings.
        pub fn getBooleanityClaims(self: *const Self, allocator: std.mem.Allocator) ![]F {
            const claims = try allocator.alloc(F, self.N);
            dbg("[BOOL_CLAIMS] phase2_len={}, round={}, N={}\n", .{ self.phase2_len, self.round, self.N });
            if (self.H) |ht| {
                var all_same_claims = true;
                for (0..self.N) |i| {
                    claims[i] = ht[i][0];
                    if (i < 5 or i >= self.N - 5 or (i >= 28 and i < 34)) {
                        const hbe = ht[i][0].toBytesBE();
                        dbg("[BOOL_CLAIMS] H[{}][0]_LE=[{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}]\n", .{
                            i, hbe[31], hbe[30], hbe[29], hbe[28], hbe[27], hbe[26], hbe[25], hbe[24],
                        });
                    }
                    if (i > 0 and !claims[i].eql(claims[0])) all_same_claims = false;
                }
                dbg("[BOOL_CLAIMS] all_same={}\n", .{@intFromBool(all_same_claims)});
            } else {
                @memset(claims, F.zero());
            }
            return claims;
        }

        /// Compute round polynomial evaluations: [s(0), s(1), s(2), s(3)]
        /// Returns 4 evaluation points (NOT [s(0), s(1), s(2), p_inf]).
        /// Phase 1 uses gruen_poly_deg_3 approach (derive Q(1) from previous_claim).
        pub fn computeRoundPoly(self: *Self, allocator: std.mem.Allocator, claim: F) ![]F {
            const evals = try allocator.alloc(F, 4);
            @memset(evals, F.zero());

            if (self.round < self.log_k_chunk) {
                self.computePhase1Poly(evals, claim);
            } else {
                self.computePhase2Poly(evals, claim);
            }

            return evals;
        }

        fn computePhase1Poly(self: *Self, evals: []F, previous_claim: F) void {
            // Gruen poly deg 3 approach (matching Jolt's gruen_poly_deg_3):
            //
            // Compute c = Q(0) (constant of quadratic Q) and e (X² coeff of Q).
            // Derive Q(1) from previous_claim to guarantee s(0)+s(1) = claim.
            // Extrapolate Q(2), compute s(2) and p_inf.
            // Return [s(0), s(1), s(2), p_inf] (Toom-Cook format).

            const m = self.round;
            const f_mask = if (m == 0) 0 else (@as(usize, 1) << @intCast(m)) - 1;
            const upper_bits = self.log_k_chunk - m - 1;

            // Build eq_upper table for the head (unbound, non-summed) address variables.
            // In Jolt's LowToHigh convention, the unbound variables at round m are
            // w[0..n-m-1] in LE order. The head = w[0..n-m-2], Gruen = w[n-m-1].
            // After reversal, r_address_le[i] = w[n-1-i], so:
            //   w[0] = r_address_le[n-1], w[1] = r_address_le[n-2], ..., w[n-m-2] = r_address_le[m+1]
            // We process from w[0] (MSB of eq_upper index) to w[n-m-2] (LSB),
            // i.e., from r_address_le[n-1] down to r_address_le[m+1].
            var eq_upper: [16]F = undefined;
            if (upper_bits == 0) {
                eq_upper[0] = F.one();
            } else {
                eq_upper[0] = F.one();
                var eq_upper_len: usize = 1;
                // Process in DESCENDING order: r_address_le[log_k-1] down to r_address_le[m+1]
                var bit: usize = self.log_k_chunk - 1;
                while (bit >= m + 1) : (bit -= 1) {
                    const w = self.r_address_le[bit];
                    const one_minus_w = F.one().sub(w);
                    var idx: usize = eq_upper_len;
                    while (idx > 0) {
                        idx -= 1;
                        eq_upper[2 * idx + 1] = eq_upper[idx].mul(w);
                        eq_upper[2 * idx] = eq_upper[idx].mul(one_minus_w);
                    }
                    eq_upper_len *= 2;
                    if (bit == 0) break;  // prevent underflow on usize
                }
            }

            // Inner loop: compute c (=Q(0), constant of Q) and e (X² coeff of Q)
            // c = Σ_{k:k_m=0} eu * Σ_i γ^{2i} * G*F*(F-1)
            // e = Σ_{all k} eu * Σ_i γ^{2i} * G*F²
            var c = F.zero();
            var e = F.zero();

            for (0..self.K) |k| {
                const k_m = (k >> @intCast(m)) & 1;
                const k_bound = k & f_mask;
                const k_upper = k >> @intCast(m + 1);
                const f_k = if (m == 0) F.one() else self.F_table[k_bound];
                const eu = eq_upper[k_upper];
                const f_sq = f_k.mul(f_k);

                var gamma_G_sum = F.zero();
                for (0..self.N) |i| {
                    gamma_G_sum = gamma_G_sum.add(self.gamma_powers_sq[i].mul(self.G[i][k]));
                }

                // e contribution (all k): eu * Σ_i γ^{2i} * G * F²
                e = e.add(eu.mul(gamma_G_sum).mul(f_sq));

                // c contribution (k_m=0 only): eu * Σ_i γ^{2i} * G*F*(F-1)
                if (k_m == 0) {
                    const G_times_F = gamma_G_sum.mul(f_k);
                    c = c.add(eu.mul(G_times_F.mul(f_k).sub(G_times_F)));
                }
            }

            // Linear eq evaluations: l(X) = eq_0 + b*X where b = eq_slope
            const w_m = self.r_address_le[m];
            const eq_eval_1 = self.B_scalar.mul(w_m);
            const eq_eval_0 = self.B_scalar.sub(eq_eval_1);
            const eq_slope = eq_eval_1.sub(eq_eval_0);
            const eq_eval_2 = eq_eval_1.add(eq_slope);

            // Derive Q(1) from previous_claim (Jolt's gruen_poly_deg_3 approach):
            // s(0) = eq_eval_0 * Q(0) = eq_eval_0 * c
            // s(1) = previous_claim - s(0)
            // Q(1) = s(1) / eq_eval_1
            const s0 = eq_eval_0.mul(c);
            const s1 = previous_claim.sub(s0);
            const q1 = if (eq_eval_1.eql(F.zero())) F.zero() else s1.mul(eq_eval_1.inverse().?);

            // Extrapolate: Q(2) = 2*Q(1) - Q(0) + 2*e
            const e_times_2 = e.add(e);
            const q2 = q1.add(q1).sub(c).add(e_times_2);

            // Extrapolate: Q(3) = 3*Q(1) - 2*Q(0) + 6*e
            const three = F.fromU64(3);
            const six = F.fromU64(6);
            const q3 = three.mul(q1).sub(c.add(c)).add(six.mul(e));

            // l(3) = eq_eval_0 + 3 * eq_slope
            const eq_eval_3 = eq_eval_0.add(three.mul(eq_slope));

            // Return Vandermonde format [s(0), s(1), s(2), s(3)]
            evals[0] = s0;
            evals[1] = s1;
            evals[2] = eq_eval_2.mul(q2);
            evals[3] = eq_eval_3.mul(q3);

            // Debug: brute force check
            if (true) {
                var bf_Q0 = F.zero();
                var bf_Q1 = F.zero();
                for (0..self.K) |k| {
                    const k_m_bf = (k >> @intCast(m)) & 1;
                    const k_bound_bf = k & f_mask;
                    const k_upper_bf = k >> @intCast(m + 1);
                    const f_k_bf = if (m == 0) F.one() else self.F_table[k_bound_bf];
                    const eu_bf = eq_upper[k_upper_bf];
                    var gsum_bf = F.zero();
                    for (0..self.N) |i| {
                        gsum_bf = gsum_bf.add(self.gamma_powers_sq[i].mul(self.G[i][k]));
                    }
                    const qval_bf = eu_bf.mul(gsum_bf.mul(f_k_bf)).mul(f_k_bf.sub(F.one()));
                    if (k_m_bf == 0) { bf_Q0 = bf_Q0.add(qval_bf); }
                    else { bf_Q1 = bf_Q1.add(qval_bf); }
                }
                const bf_s0 = eq_eval_0.mul(bf_Q0);
                const bf_s1 = eq_eval_1.mul(bf_Q1);
                const bf_sum = bf_s0.add(bf_s1);
                const c_ok: u8 = if (c.eql(bf_Q0)) 1 else 0;
                const sum_ok: u8 = if (bf_sum.eql(previous_claim)) 1 else 0;
                // Also compute e and the expected sum from the polynomial theory
                var bf_e2 = F.zero();
                for (0..self.K) |k| {
                    const k_bound_bf = k & f_mask;
                    const k_upper_bf = k >> @intCast(m + 1);
                    const f_k_bf = if (m == 0) F.one() else self.F_table[k_bound_bf];
                    const eu_bf = eq_upper[k_upper_bf];
                    var gsum_bf = F.zero();
                    for (0..self.N) |i| {
                        gsum_bf = gsum_bf.add(self.gamma_powers_sq[i].mul(self.G[i][k]));
                    }
                    bf_e2 = bf_e2.add(eu_bf.mul(gsum_bf).mul(f_k_bf.mul(f_k_bf)));
                }
                const e_ok: u8 = if (e.eql(bf_e2)) 1 else 0;
                dbg("  [BOOL_PH1] m={} c_ok={} e_ok={} sum_ok={}\n", .{ m, c_ok, e_ok, sum_ok });
                if (sum_ok == 0) {
                    // Print what previous_claim should be
                    // The actual claim should be s(r) from the PREVIOUS round
                    // which equals eq(w_m, r) * Q(r) where Q is the quadratic
                    // But we don't have easy access to the challenge here.
                    // Instead, just print the values for inspection.
                    const pc = previous_claim.toBytesBE();
                    const bs = bf_sum.toBytesBE();
                    dbg("  [BOOL_PH1]   prev_claim_BE={x:0>2}{x:0>2}{x:0>2}{x:0>2}\n", .{ pc[0], pc[1], pc[2], pc[3] });
                    dbg("  [BOOL_PH1]   bf_sum_BE    ={x:0>2}{x:0>2}{x:0>2}{x:0>2}\n", .{ bs[0], bs[1], bs[2], bs[3] });
                    // Diagnose: compute from the polynomial at round m=0
                    // If m=1, the claim comes from round 0's polynomial at the challenge.
                    // Round 0's polynomial: s_0(X) = eq(w_0,X) * e_prev * X * (X-1)
                    // where e_prev = e at round 0.
                    // But we don't store e from previous rounds.
                    // Instead, let's check a simpler invariant:
                    // The claim at round m should equal B_scalar * [something].
                    // Print B_scalar for inspection.
                    const bsb = self.B_scalar.toBytesBE();
                    dbg("  [BOOL_PH1]   B_scalar_BE  ={x:0>2}{x:0>2}{x:0>2}{x:0>2}\n", .{ bsb[0], bsb[1], bsb[2], bsb[3] });
                }
            }
        }

        fn computePhase2Poly(self: *Self, evals: []F, previous_claim: F) void {
            // Phase 2: Gruen poly deg 3 approach (matching Jolt's compute_phase2_message)
            //
            // The polynomial is:
            //   p(X) = eq_r_r * Σ_j d_j(X) * Q_j(X)
            // where d_j(X) is the linear eq_cycle factor, Q_j(X) = Σ_i γ^{2i} * h_i(X)*(h_i(X)-1)
            //
            // The D polynomial (eq_cycle) plays the role of the Gruen split-eq.
            // We compute c = Q_weighted(0) and e (X² coeff of Q_weighted), then use gruen approach.

            const ht = self.H orelse return;
            const half = self.phase2_len / 2;

            // Compute c (constant of quadratic, weighted by eq_cycle) and e (X² coeff)
            // c = Σ_j d0_j * Σ_i γ²ⁱ * h0_i*(h0_i-1)
            // e = Σ_j Σ_i γ²ⁱ * (h1_i-h0_i)²  (X² coefficient, NOT weighted by eq_cycle slope)
            //
            // Wait - the D.gruen_poly_deg_3 in Jolt uses par_fold_out_in_unreduced which weights
            // both c and e by the E_out * E_in tables. So both c and e ARE weighted by eq_cycle.
            //
            // Actually no: Jolt uses the split-eq D for the cycle direction. The per_g_values
            // closure returns [c_per_j_prime, e_per_j_prime] and then fold_out_in multiplies
            // by e_out * e_in. So c and e are both weighted by the outer eq factors.
            // The gruen_poly_deg_3 then uses the current linear eq factor (innermost) to build
            // the cubic s(X) = l(X) * Q(X).
            //
            // But in our simplified approach (no Gruen split-eq, just halving eq_cycle),
            // we can compute c and e using the full eq_cycle halving:
            // The eq_cycle has entries for j = 0..phase2_len-1, paired as (d0, d1).
            // The linear factor is l_j(X) = d0_j + (d1_j - d0_j)*X for each pair j.
            //
            // The sumcheck variable is the FIRST bit, so:
            //   For pair j: j_pair_index = j, d0 = eq_cycle[2j], d1 = eq_cycle[2j+1]
            //   Q_j(X) = Σ_i γ²ⁱ * h_i_j(X) * (h_i_j(X) - 1)
            //   h_i_j(X) = ht[i][2j] + (ht[i][2j+1] - ht[i][2j]) * X
            //
            // Total: s(X) = eq_r_r * Σ_j l_j(X) * Q_j(X)
            //
            // For the gruen approach, we need to express this as s(X) = eq_r_r * [sum of l*Q].
            // But each j pair has a different l_j. This is NOT a simple l*Q factorization.
            //
            // In Jolt, the D split-eq handles this by having E_out and E_in weight the j_prime
            // groups, and the current variable's eq factor is handled by gruen_poly_deg_3.
            //
            // For our approach (direct halving), we compute s(0), s(1), s(2) directly and
            // also compute p_inf (X³ coefficient). Then s(0)+s(1) should equal claim.
            // Let's keep the direct computation approach but verify against claim.

            const BoolP2Ctx = struct {
                ht: [][]F,
                eq_cycle: []const F,
                gamma_powers_sq: []const F,
                N: usize,
            };
            const ctx = BoolP2Ctx{
                .ht = ht,
                .eq_cycle = self.eq_cycle,
                .gamma_powers_sq = self.gamma_powers_sq,
                .N = self.N,
            };

            const mapFn = struct {
                fn f(c: BoolP2Ctx, start: usize, end: usize) [4]F {
                    var ev = [4]F{ F.zero(), F.zero(), F.zero(), F.zero() };
                    for (start..end) |j| {
                        const d0 = c.eq_cycle[2 * j];
                        const d1 = c.eq_cycle[2 * j + 1];

                        // Evaluate at points 0, 1, 2, 3
                        var q0 = F.zero();
                        var q1 = F.zero();
                        var q2 = F.zero();
                        var q3 = F.zero();
                        for (0..c.N) |i| {
                            const h0 = c.ht[i][2 * j];
                            const h1 = c.ht[i][2 * j + 1];
                            const h_delta = h1.sub(h0);
                            const gp = c.gamma_powers_sq[i];

                            q0 = q0.add(gp.mul(h0.mul(h0).sub(h0)));
                            q1 = q1.add(gp.mul(h1.mul(h1).sub(h1)));

                            const h2 = h0.add(F.fromU64(2).mul(h_delta));
                            q2 = q2.add(gp.mul(h2.mul(h2).sub(h2)));

                            const h3 = h0.add(F.fromU64(3).mul(h_delta));
                            q3 = q3.add(gp.mul(h3.mul(h3).sub(h3)));
                        }
                        const d_delta = d1.sub(d0);
                        ev[0] = ev[0].add(d0.mul(q0));
                        ev[1] = ev[1].add(d1.mul(q1));
                        ev[2] = ev[2].add(d0.add(F.fromU64(2).mul(d_delta)).mul(q2));
                        ev[3] = ev[3].add(d0.add(F.fromU64(3).mul(d_delta)).mul(q3));
                    }
                    return ev;
                }
            }.f;

            const reduceFn = struct {
                fn f(a: [4]F, b: [4]F) [4]F {
                    return [4]F{ a[0].add(b[0]), a[1].add(b[1]), a[2].add(b[2]), a[3].add(b[3]) };
                }
            }.f;

            const result = if (self.pool) |pool|
                pool.parallelReduce([4]F, half, [4]F{ F.zero(), F.zero(), F.zero(), F.zero() }, ctx, mapFn, reduceFn)
            else
                mapFn(ctx, 0, half);

            for (0..4) |k| {
                evals[k] = result[k];
            }

            // Scale by eq_r_r
            for (0..4) |k| {
                evals[k] = evals[k].mul(self.eq_r_r);
            }

            // Debug: check s(0)+s(1) vs previous_claim
            if (true) {
                const sum = evals[0].add(evals[1]);
                const ok: u8 = if (sum.eql(previous_claim)) 1 else 0;
                dbg("  [BOOL_PH2] p(0)+p(1)=claim? {} phase2_round={}\n", .{ ok, self.round - self.log_k_chunk });
            }
        }

        pub fn bindChallenge(self: *Self, r: F) !void {
            if (self.round < self.log_k_chunk) {
                // Phase 1: update B_scalar and F_table
                const w_m = self.r_address_le[self.round];
                // B_scalar *= eq(w_m, r) = w_m*r + (1-w_m)*(1-r) = 1 - w_m - r + 2*w_m*r
                const prod = w_m.mul(r);
                self.B_scalar = self.B_scalar.mul(F.one().sub(w_m).sub(r).add(prod.add(prod)));

                // Update F: double size from 2^m to 2^(m+1)
                // Match Jolt's LowToHigh ExpandingTable: new entries go in the UPPER half.
                // This ensures bit j of the F index corresponds to sumcheck challenge r_j.
                // Jolt: F[i+len] = F[i]*r, F[i] = F[i]*(1-r)  (for i in 0..len)
                for (0..self.F_size) |idx| {
                    self.F_table[idx + self.F_size] = self.F_table[idx].mul(r);
                    self.F_table[idx] = self.F_table[idx].sub(self.F_table[idx + self.F_size]);
                }
                self.F_size *= 2;

                // Phase 1→2 transition after last address round
                if (self.round == self.log_k_chunk - 1) {
                    self.eq_r_r = self.B_scalar; // eq(r_addr_fixed, r_addr_bound)
                    try self.transitionToPhase2();
                }
            } else {
                // Phase 2: bind cycle variable, halve H tables and eq_cycle
                const half = self.phase2_len / 2;

                const bindOne = struct {
                    fn f(arr: []F, h: usize, challenge: F) void {
                        for (0..h) |j| {
                            arr[j] = arr[2 * j].add(challenge.mul(arr[2 * j + 1].sub(arr[2 * j])));
                        }
                    }
                }.f;

                if (self.H) |ht| {
                    if (self.pool) |pool| {
                        // N+1 independent arrays: N H tables + 1 eq_cycle
                        const total = self.N + 1;
                        const Ctx2 = struct { ht: [][]F, eq_cycle: []F, n: usize, half: usize, challenge: F };
                        const ctx2 = Ctx2{ .ht = ht, .eq_cycle = self.eq_cycle, .n = self.N, .half = half, .challenge = r };
                        pool.parallelForForce(total, ctx2, struct {
                            fn f2(c: Ctx2, idx: usize) void {
                                if (idx < c.n) {
                                    bindOne(c.ht[idx], c.half, c.challenge);
                                } else {
                                    bindOne(c.eq_cycle, c.half, c.challenge);
                                }
                            }
                        }.f2);
                    } else {
                        for (0..self.N) |i| {
                            bindOne(ht[i], half, r);
                        }
                        bindOne(self.eq_cycle, half, r);
                    }
                } else {
                    bindOne(self.eq_cycle, half, r);
                }
                self.phase2_len = half;
            }
            self.round += 1;
        }

        fn transitionToPhase2(self: *Self) !void {
            // F_table now has K entries: F[k] = eq(r_challenges, k) for k ∈ [0, K)
            // Build H tables: for each cycle j, H[i][j] = F[chunk_i(j)] = eq(r_addr_bound, chunk_i(j))
            const T_val = @as(usize, 1) << @intCast(self.n_cycle_vars);
            const trace = self.trace;
            const instr_d = self.instruction_d;
            const bc_d = self.bytecode_d;
            const ram_d_val = self.ram_d;
            const K = self.K;

            var ht = try self.allocator.alloc([]F, self.N);
            for (0..self.N) |i| {
                ht[i] = try self.allocator.alloc(F, T_val);
                @memset(ht[i], F.zero());
            }

            const dbg_nonzero_chunks: usize = 0;
            for (0..T_val) |j| {
                const step = trace.steps.items[j];

                // InstructionRa chunks
                // Use the centralized computeLookupIndex to ensure consistency
                // across all sumcheck instances (Stage 6 virtualization, booleanity, Stage 7).
                {
                    const lookup_idx = computeLookupIndex(step);
                    for (0..instr_d) |i| {
                        const shift = self.log_k_chunk * (instr_d - 1 - i);
                        const mask: u128 = (@as(u128, 1) << @intCast(self.log_k_chunk)) - 1;
                        const chunk_val: usize = @intCast((lookup_idx >> @intCast(shift)) & mask);
                        if (chunk_val < K) {
                            ht[i][j] = self.F_table[chunk_val];
                        }
                    }
                }

                // BytecodeRa chunks
                {
                    const pc_idx: u64 = @intCast(self.pc_map.getPCForStep(step));
                    for (0..bc_d) |i| {
                        const chunk_val = extractChunkMSB(pc_idx, i, bc_d, self.log_k_chunk);
                        if (chunk_val < K) {
                            ht[instr_d + i][j] = self.F_table[chunk_val];
                        }
                    }
                }

                // RamRa chunks
                {
                    if (step.memory_addr) |addr| {
                        if (addr != 0) {
                            if (self.memory_layout.remapAddress(addr)) |raddr| {
                                for (0..ram_d_val) |i| {
                                    const chunk_val = extractChunkMSB(raddr, i, ram_d_val, self.log_k_chunk);
                                    if (chunk_val < K) {
                                        ht[instr_d + bc_d + i][j] = self.F_table[chunk_val];
                                    }
                                }
                            }
                        }
                    }
                }
            }

            self.H = ht;
            self.phase2_len = T_val;

            // Debug: check if all H tables are identical
            {
                var all_same = true;
                for (1..self.N) |i| {
                    if (!std.mem.eql(u8, &ht[0][0].toBytesBE(), &ht[i][0].toBytesBE())) {
                        all_same = false;
                        break;
                    }
                }
                dbg("[BOOL_H_INIT] T={}, all_H[i][0]_same={}, dbg_nonzero_chunks={}\n", .{ T_val, @intFromBool(all_same), dbg_nonzero_chunks });
                // Print first few H entries for different i at interesting cycles
                // Check that H tables differ across polynomials at non-noop cycles
                {
                    var first_nontrivial_j: usize = 0;
                    for (0..T_val) |jj| {
                        if (!ht[0][jj].eql(ht[0][0])) {
                            first_nontrivial_j = jj;
                            break;
                        }
                    }
                    dbg("[BOOL_H_INIT] first non-trivial j={}\n", .{first_nontrivial_j});
                    // Print H[i][j] at this cycle for first few polynomials
                    for (0..@min(4, self.N)) |i| {
                        const hj = ht[i][first_nontrivial_j].toBytesBE();
                        const h0 = ht[i][0].toBytesBE();
                        dbg("[BOOL_H_INIT] H[{}][0]_LE=[{x:0>2}{x:0>2}{x:0>2}{x:0>2}] H[{}][{}]_LE=[{x:0>2}{x:0>2}{x:0>2}{x:0>2}]\n", .{
                            i, h0[31], h0[30], h0[29], h0[28],
                            i, first_nontrivial_j, hj[31], hj[30], hj[29], hj[28],
                        });
                    }
                    // Count how many cycles have non-F[0] values for each poly
                    // Check MSB chunks (0..4) AND LSB chunks (instr_d-4..instr_d) AND bytecode/ram
                    const check_indices = [_]usize{ 0, 1, 2, 3, instr_d - 4, instr_d - 3, instr_d - 2, instr_d - 1, instr_d, instr_d + 1, instr_d + bc_d, instr_d + bc_d + 1 };
                    for (check_indices) |i| {
                        if (i >= self.N) continue;
                        var nontrivial: usize = 0;
                        var nonzero: usize = 0;
                        for (0..T_val) |jj| {
                            if (!ht[i][jj].eql(F.zero())) nonzero += 1;
                            if (!ht[i][jj].eql(F.zero()) and !ht[i][jj].eql(self.F_table[0])) {
                                nontrivial += 1;
                            }
                        }
                        dbg("[BOOL_H_INIT] poly {} nontrivial_cycles={} nonzero={}\n", .{ i, nontrivial, nonzero });
                    }
                    // Also show distinct values for poly instr_d-1 (LSB chunk)
                    {
                        const lsb_i = instr_d - 1;
                        if (lsb_i < self.N) {
                            dbg("[BOOL_H_INIT] LSB poly {} first 8 values:", .{lsb_i});
                            for (0..@min(8, T_val)) |jj| {
                                const hv = ht[lsb_i][jj].toBytesBE();
                                dbg(" [{x:0>2}{x:0>2}{x:0>2}{x:0>2}]", .{ hv[31], hv[30], hv[29], hv[28] });
                            }
                            dbg("\n", .{});
                        }
                    }
                }
                // Check F_table state
                dbg("[BOOL_H_INIT] F_size={}\n", .{ self.F_size });
                for (0..@min(self.F_size, 8)) |fi| {
                    const fb = self.F_table[fi].toBytesBE();
                    dbg("[BOOL_H_INIT] F[{}]_LE=[{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}]\n", .{
                        fi, fb[31], fb[30], fb[29], fb[28], fb[27], fb[26], fb[25], fb[24],
                    });
                }
            }

            // Debug: compute Phase 2 full sum and compare with what the claim should be
            {
                var phase2_sum = F.zero();
                for (0..T_val) |jj| {
                    var q_j = F.zero();
                    for (0..self.N) |i| {
                        const h_val = ht[i][jj];
                        q_j = q_j.add(self.gamma_powers_sq[i].mul(h_val.mul(h_val).sub(h_val)));
                    }
                    phase2_sum = phase2_sum.add(self.eq_cycle[jj].mul(q_j));
                }
                phase2_sum = phase2_sum.mul(self.eq_r_r);
                const ps_be = phase2_sum.toBytesBE();
                dbg("[BOOL_TRANSITION] phase2_full_sum LE=[{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}]\n", .{
                    ps_be[31], ps_be[30], ps_be[29], ps_be[28], ps_be[27], ps_be[26], ps_be[25], ps_be[24],
                });

                // Also compute Phase 1 full sum (using F_table directly)
                var phase1_sum = F.zero();
                for (0..T_val) |jj| {
                    var q_j_ph1 = F.zero();
                    for (0..self.N) |i| {
                        // H[i][jj] = F_table[chunk_i(jj)] which should equal the raw value
                        const h_val_ph1 = ht[i][jj]; // Same as F_table[chunk_i(j)]
                        q_j_ph1 = q_j_ph1.add(self.gamma_powers_sq[i].mul(h_val_ph1.mul(h_val_ph1).sub(h_val_ph1)));
                    }
                    phase1_sum = phase1_sum.add(self.eq_cycle[jj].mul(q_j_ph1));
                }
                phase1_sum = phase1_sum.mul(self.eq_r_r);
                // This should be the same as phase2_sum
            }

            dbg("[BOOL_PROVER] Phase 1→2 transition: eq_r_r=", .{});
            const err_be = self.eq_r_r.toBytesBE();
            for (0..8) |bi| dbg("{x:0>2}", .{err_be[31 - bi]});
            dbg(", H[0][0..3]=", .{});
            for (0..@min(3, T_val)) |jj| {
                const hv = ht[0][jj].toBytesBE();
                dbg("[{x:0>2}{x:0>2}{x:0>2}{x:0>2}]", .{ hv[31], hv[30], hv[29], hv[28] });
            }
            dbg("\n", .{});
        }

        /// Get opening claims: ra_i(r_addr_bound, r_cycle_bound)
        /// After all rounds, H[i][0] = eq(r_addr_bound, chunk_i(r_cycle_bound)) = ra_i(r_addr_bound, r_cycle_bound).
        pub fn getOpeningClaims(self: *const Self, allocator: std.mem.Allocator) ![]F {
            const claims = try allocator.alloc(F, self.N);
            if (self.H) |ht| {
                for (0..self.N) |i| {
                    claims[i] = ht[i][0];
                }
            } else {
                @memset(claims, F.zero());
            }
            return claims;
        }
    };
}

// =============================================================================
// LookupsRaVirtual Sumcheck Instance (Instance 4)
// =============================================================================
// Proves: Sigma_c eq(r_cycle, c) * Sum_{v=0}^{N-1} gamma^v * Prod_{j=0}^{M-1} ra_{v*M+j}(c)
// Variables: n_cycle_vars
// Degree: M+1 (product of M linear ra polys * one linear eq)
fn LookupsRaVirtualProver(comptime F: type) type {
    return struct {
        const Self = @This();

        /// ra_bound[i][j] - pre-bound to address chunks
        /// First poly in each virtual batch pre-scaled by gamma^batch
        ra_bound: [][]F,
        /// eq(r_cycle, .) evaluations
        eq: []F,
        M: usize,
        N: usize,
        total_committed: usize,
        current_len: usize,
        allocator: Allocator,
        pool: ?*ThreadPool = null,

        pub fn init(
            allocator: Allocator,
            trace: *const ExecutionTrace,
            r_cycle: []const F, // BIG_ENDIAN
            r_addr_chunks: []const []const F, // r_addr_chunks[i] for each committed poly
            gamma_powers: []const F, // gamma^v for v in 0..N
            M: usize,
            N: usize,
            log_k_chunk: usize,
            instruction_d: usize,
            init_pool: ?*ThreadPool,
        ) !Self {
            const T = trace.steps.items.len;
            const n_vars = std.math.log2_int(usize, T);
            const total_committed = M * N;
            const k_chunk: usize = @as(usize, 1) << @intCast(log_k_chunk);

            var ra_bound_arr = try allocator.alloc([]F, total_committed);
            errdefer {
                for (ra_bound_arr[0..total_committed]) |arr| allocator.free(arr);
                allocator.free(ra_bound_arr);
            }

            for (0..total_committed) |i| {
                ra_bound_arr[i] = try allocator.alloc(F, T);

                // r_addr_chunks[i] must be reversed for LE computeEqTable to match
                // Jolt's BE EqPolynomial::evals convention
                var r_chunk_rev = try allocator.alloc(F, log_k_chunk);
                defer allocator.free(r_chunk_rev);
                for (0..log_k_chunk) |ci| r_chunk_rev[ci] = r_addr_chunks[i][log_k_chunk - 1 - ci];
                const eq_table = try computeEqTable(F, allocator, r_chunk_rev, log_k_chunk);
                defer allocator.free(eq_table);

                // Determine gamma scaling for first poly in each virtual batch
                const virtual_batch = i / M;
                const is_first_in_batch = (i % M == 0);
                const scale = if (is_first_in_batch) gamma_powers[virtual_batch] else F.one();

                // Debug: print eq_table values
                {
                    const et0_le = eq_table[0].toBytes();
                    if (i < 16) {
                        dbg("[EQ_TABLE_DBG] chunk[{}] eq_table[0]_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}] scale={}\n", .{
                            i, et0_le[0], et0_le[1], et0_le[2], et0_le[3], et0_le[4], et0_le[5], et0_le[6], et0_le[7],
                            @intFromBool(is_first_in_batch),
                        });
                    }
                }

                for (0..T) |j| {
                    const step = trace.steps.items[j];
                    // Get lookup index chunk - uses interleaved bits and MSB-first ordering
                    const chunk_val = getLookupChunkInterleaved(step, i, log_k_chunk, instruction_d);
                    if (chunk_val < k_chunk) {
                        ra_bound_arr[i][j] = eq_table[chunk_val].mul(scale);
                    } else {
                        ra_bound_arr[i][j] = F.zero();
                    }
                }
            }

            // Debug: find any cycle where batch 3 chunks are non-zero
            {
                var nonzero_count: usize = 0;
                for (0..T) |j| {
                    const step = trace.steps.items[j];
                    var any_nonzero = false;
                    for (12..16) |ci| {
                        const cv = getLookupChunkInterleaved(step, ci, log_k_chunk, instruction_d);
                        if (cv != 0) {
                            any_nonzero = true;
                            break;
                        }
                    }
                    if (any_nonzero) {
                        const li = computeLookupIndex(step);
                        dbg("[BATCH3_NONZERO] cycle {} lookup_index=0x{x:0>32} instr=0x{x:0>8} noop={}\n", .{ j, li, step.instruction, @intFromBool(step.is_noop) });
                        nonzero_count += 1;
                        if (nonzero_count >= 5) break;
                    }
                }
                dbg("[BATCH3_CHECK] {} cycles have non-zero batch 3 chunks\n", .{nonzero_count});
            }

            // Debug: compute and print product of ra_bound[v*M..v*M+M-1][0] for each virtual batch
            // For all-zero chunks, this is just product of eq_table[0] values
            dbg("[LOOKUPS_RA_BATCH_PRODUCT] M={}, N={}\n", .{ M, N });
            for (0..N) |v| {
                var prod = F.one();
                for (0..M) |m| {
                    prod = prod.mul(ra_bound_arr[v * M + m][0]);
                }
                // Remove gamma scaling: first poly in batch v was scaled by gamma^v
                const unscaled_prod = if (v == 0) prod else prod.mul(gamma_powers[v].inverse().?);
                const p_le = unscaled_prod.toBytes();
                dbg("  batch[{}] product_at_cycle0 (unscaled)_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                    v, p_le[0], p_le[1], p_le[2], p_le[3], p_le[4], p_le[5], p_le[6], p_le[7],
                });
            }

            // r_cycle is in BE order; reverse for LE computeEqTable
            var r_cycle_rev = try allocator.alloc(F, n_vars);
            defer allocator.free(r_cycle_rev);
            for (0..n_vars) |i| r_cycle_rev[i] = r_cycle[n_vars - 1 - i];
            const eq_arr = try computeEqTableParallel(F, allocator, r_cycle_rev, n_vars, init_pool);

            // Verify: sum over all cycles should equal the input claim
            {
                var total = F.zero();
                for (0..T) |c| {
                    var virtual_sum = F.zero();
                    for (0..N) |v| {
                        var product = F.one();
                        for (0..M) |m| {
                            product = product.mul(ra_bound_arr[v * M + m][c]);
                        }
                        virtual_sum = virtual_sum.add(product);
                    }
                    if (c < 4) {
                        // Debug: print per-virtual-batch products for first 4 cycles
                        dbg("[LOOKUPS_RA_INIT] cycle {} eq={x}\n", .{ c, eq_arr[c].toBytesBE()[24..32].* });
                        for (0..@min(N, 2)) |v| {
                            var product_dbg = F.one();
                            for (0..M) |m| {
                                const idx = v * M + m;
                                const rv = ra_bound_arr[idx][c];
                                dbg("  ra_bound[{}][{}]={x}", .{ idx, c, rv.toBytesBE()[24..32].* });
                                product_dbg = product_dbg.mul(rv);
                            }
                            dbg("  prod={x}\n", .{product_dbg.toBytesBE()[24..32].*});
                        }
                        const step = trace.steps.items[c];
                        const lookup_idx = computeLookupIndex(step);
                        dbg("  lookup_index=0x{x:0>32} rs1=0x{x:0>16} rs2=0x{x:0>16} instr=0x{x:0>8}\n", .{ lookup_idx, step.rs1_value, step.rs2_value, step.instruction });
                        for (0..@min(M, 4)) |m| {
                            const cv = getLookupChunkInterleaved(step, m, log_k_chunk, instruction_d);
                            dbg("  chunk[{}]={}\n", .{ m, cv });
                        }
                    }
                    total = total.add(eq_arr[c].mul(virtual_sum));
                }
                const t_le = total.toBytes();
                dbg("[LOOKUPS_RA_INIT] computed_claim_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                    t_le[0], t_le[1], t_le[2], t_le[3], t_le[4], t_le[5], t_le[6], t_le[7],
                });
                // Also print the per-virtual-batch sum (with eq)
                dbg("[LOOKUPS_RA_INIT] N={} M={} T={}\n", .{ N, M, T });
                // Per-batch claim: Σ_c eq(r_cycle, c) * Π_{j=0}^{M-1} ra_bound[v*M+j][c]
                // Also compute ra_claim_v (without gamma) by dividing batch sum by gamma^v
                var verif_total = F.zero();
                for (0..N) |v| {
                    var batch_sum = F.zero();
                    for (0..T) |c| {
                        var prod = F.one();
                        for (0..M) |m| {
                            prod = prod.mul(ra_bound_arr[v * M + m][c]);
                        }
                        batch_sum = batch_sum.add(eq_arr[c].mul(prod));
                    }
                    verif_total = verif_total.add(batch_sum);
                    {
                        const bs_le = batch_sum.toBytes();
                        // Divide by gamma^v to get the unscaled batch sum (should match ra_chunks[v])
                        const unscaled = if (v == 0) batch_sum else batch_sum.mul(gamma_powers[v].inverse().?);
                        const us_le = unscaled.toBytes();
                        dbg("[LOOKUPS_RA_INIT] batch[{}] sum_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}] unscaled_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                            v,
                            bs_le[0], bs_le[1], bs_le[2], bs_le[3], bs_le[4], bs_le[5], bs_le[6], bs_le[7],
                            us_le[0], us_le[1], us_le[2], us_le[3], us_le[4], us_le[5], us_le[6], us_le[7],
                        });
                    }
                }
                const vt_le = verif_total.toBytes();
                dbg("[LOOKUPS_RA_INIT] verif_total(Σbatch[v])_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                    vt_le[0], vt_le[1], vt_le[2], vt_le[3], vt_le[4], vt_le[5], vt_le[6], vt_le[7],
                });
                // Check against total computed earlier
                const total_ok: u8 = if (std.mem.eql(u8, &verif_total.toBytesBE(), &total.toBytesBE())) 1 else 0;
                dbg("[LOOKUPS_RA_INIT] verif_total == total? {}\n", .{total_ok});
            }

            return Self{
                .ra_bound = ra_bound_arr,
                .eq = eq_arr,
                .M = M,
                .N = N,
                .total_committed = total_committed,
                .current_len = T,
                .allocator = allocator,
                .pool = init_pool,
            };
        }

        pub fn deinit(self: *Self) void {
            for (self.ra_bound) |arr| self.allocator.free(arr);
            self.allocator.free(self.ra_bound);
            self.allocator.free(self.eq);
        }

        /// f(x) = eq(x) * Sum_v Prod_{j=0}^{M-1} ra_{v*M+j}(x)
        /// Degree = M + 1
        pub fn computeRoundPoly(self: *Self, allocator: Allocator) ![]F {
            const half = self.current_len / 2;
            const n_evals = self.M + 2;

            // Precompute x_vals
            var x_vals: [MAX_RA_EVALS]F = undefined;
            for (0..n_evals) |i| {
                x_vals[i] = F.fromU64(@intCast(i));
            }

            const Ctx = struct {
                ra_bound: [][]F,
                eq: []F,
                M: usize,
                N: usize,
                n_evals: usize,
                x_vals: [MAX_RA_EVALS]F,
            };
            const ctx = Ctx{
                .ra_bound = self.ra_bound,
                .eq = self.eq,
                .M = self.M,
                .N = self.N,
                .n_evals = n_evals,
                .x_vals = x_vals,
            };

            const mapFn = struct {
                fn f(c: Ctx, start: usize, end: usize) [MAX_RA_EVALS]F {
                    var acc: [MAX_RA_EVALS]F = .{F.zero()} ** MAX_RA_EVALS;
                    for (start..end) |j| {
                        const eq0 = c.eq[2 * j];
                        const eq1 = c.eq[2 * j + 1];
                        const eq_delta = eq1.sub(eq0);

                        for (0..c.n_evals) |pt_idx| {
                            const x = c.x_vals[pt_idx];
                            var virtual_sum = F.zero();

                            for (0..c.N) |v| {
                                var product = F.one();

                                for (0..c.M) |m| {
                                    const idx = v * c.M + m;
                                    const v0 = c.ra_bound[idx][2 * j];
                                    const v1 = c.ra_bound[idx][2 * j + 1];
                                    product = product.mul(v0.add(x.mul(v1.sub(v0))));
                                }

                                virtual_sum = virtual_sum.add(product);
                            }

                            acc[pt_idx] = acc[pt_idx].add(eq0.add(x.mul(eq_delta)).mul(virtual_sum));
                        }
                    }
                    return acc;
                }
            }.f;

            const reduceFn = struct {
                fn f(a: [MAX_RA_EVALS]F, b: [MAX_RA_EVALS]F) [MAX_RA_EVALS]F {
                    var r: [MAX_RA_EVALS]F = undefined;
                    for (0..MAX_RA_EVALS) |i| {
                        r[i] = a[i].add(b[i]);
                    }
                    return r;
                }
            }.f;

            const result = if (self.pool) |pool|
                pool.parallelReduce([MAX_RA_EVALS]F, half, .{F.zero()} ** MAX_RA_EVALS, ctx, mapFn, reduceFn)
            else
                mapFn(ctx, 0, half);

            var evals = try allocator.alloc(F, n_evals);
            for (0..n_evals) |i| {
                evals[i] = result[i];
            }
            return evals;
        }

        pub fn bindChallenge(self: *Self, r: F) void {
            const half = self.current_len / 2;

            const bindOne = struct {
                fn f(arr: []F, h: usize, challenge: F) void {
                    for (0..h) |j| {
                        arr[j] = arr[2 * j].add(challenge.mul(arr[2 * j + 1].sub(arr[2 * j])));
                    }
                }
            }.f;

            if (self.pool) |pool| {
                // total_committed+1 independent arrays: total_committed ra_bound + 1 eq
                const total = self.total_committed + 1;
                const Ctx = struct { ra: [][]F, eq: []F, tc: usize, half: usize, r: F };
                const ctx = Ctx{ .ra = self.ra_bound, .eq = self.eq, .tc = self.total_committed, .half = half, .r = r };
                pool.parallelForForce(total, ctx, struct {
                    fn f(c: Ctx, idx: usize) void {
                        if (idx < c.tc) {
                            bindOne(c.ra[idx], c.half, c.r);
                        } else {
                            bindOne(c.eq, c.half, c.r);
                        }
                    }
                }.f);
            } else {
                for (0..self.total_committed) |i| {
                    bindOne(self.ra_bound[i], half, r);
                }
                bindOne(self.eq, half, r);
            }

            self.current_len = half;
        }

        pub fn getOpeningClaims(self: *const Self, allocator: Allocator, gamma_powers: []const F) ![]F {
            // Return individual committed RA poly evaluations with gamma scaling undone
            var claims = try allocator.alloc(F, self.total_committed);
            for (0..self.total_committed) |i| {
                var claim = self.ra_bound[i][0];
                // Undo gamma pre-scaling for first poly in each batch
                const is_first_in_batch = (i % self.M == 0);
                if (is_first_in_batch) {
                    const virtual_batch = i / self.M;
                    claim = claim.mul(gamma_powers[virtual_batch].inverse().?);
                }
                claims[i] = claim;
            }
            return claims;
        }
    };
}

// =============================================================================
// BytecodeReadRaf Sumcheck Instance (Instance 0)
// =============================================================================
// Most complex instance. Two phases:
// Phase 1: Address binding (bytecode_log_k rounds)
//   Polynomial: H(k) = Sum_s gamma^s * F_s[k] * (Val_s(k) + RAF_s(k))
//   where F_s[k] = Sum_c eq(r_cycle_s, c) * delta(PC(c)=k)
//   Both F_s and Val are linear in the bound address variable, so the product
//   gives a DEGREE 2 round polynomial.
//
// Phase 2: Cycle binding (n_cycle_vars rounds)
//   After binding address to r_addr, polynomial becomes:
//   f(c) = [Prod_i ra_chunk_i(c)] * [Sum_s gamma^s * bound_val_s * eq_s(c)]
//   Degree = bytecode_d + 1
fn BytecodeReadRafProver(comptime F: type) type {
    return struct {
        const Self = @This();

        /// Phase 1: Separate F_s and val_with_raf arrays per stage
        /// F_s_arrs[s][k] = Sum_c:PC(c)=k eq(r_cycle_s, c)
        F_s_arrs: [5][]F,
        /// val_with_raf[s][k] = Val_s(k) + RAF_s(k)
        val_with_raf: [5][]F,
        /// Per-stage running claims for Phase 1
        stage_claims: [5]F,

        /// Phase 2: combined scalar polynomial over cycle vars
        /// combined[c] = Sum_s bound_val_s * eq_s(c)
        combined: ?[]F,

        /// Phase 2: RA chunk polynomials ra_chunks[i][c]
        ra_chunks: ?[][]F,

        /// Phase tracking
        phase: u8,
        bytecode_log_k: usize,
        n_cycle_vars: usize,
        bytecode_d: usize,
        log_k_chunk: usize,
        current_len: usize,
        addr_rounds_done: usize,

        /// Stored from Phase 1→2 transition for diagnostics
        bound_vals_stored: [5]F,

        /// Data needed for phase transition
        trace: *const ExecutionTrace,
        pc_map: *const BytecodePCMapper,
        stage_r_cycles: [5][]const F,
        gamma_powers: [7]F,
        /// Val polynomials per stage: val_polys[s][k]
        val_polys: [5][]F,
        /// Identity polynomial: int_poly[k] = k as field element
        int_poly: []F,

        allocator: Allocator,
        pool: ?*ThreadPool = null,

        pub fn init(
            allocator: Allocator,
            trace: *const ExecutionTrace,
            pc_map: *const BytecodePCMapper,
            val_polys: [5][]F, // Val_s(k) for each stage, length bytecode_K each
            bytecode_log_k: usize,
            n_cycle_vars: usize,
            bytecode_d: usize,
            log_k_chunk: usize,
            gamma_powers: [7]F,
            stage_r_cycles: [5][]const F,
            int_poly: []F,
            external_stage_claims: [5]F, // From opening claims: claim_per_stage[s]
            init_pool: ?*ThreadPool,
        ) !Self {
            const bytecode_K: usize = @as(usize, 1) << @intCast(bytecode_log_k);
            const T: usize = @as(usize, 1) << @intCast(n_cycle_vars);

            // Phase 1: Build separate F_s and val_with_raf arrays per stage
            var F_s_arrs: [5][]F = undefined;
            var val_with_raf_arrs: [5][]F = undefined;
            var stage_claims_init: [5]F = undefined;

            for (0..5) |s| {
                // stage_r_cycles[s] is in BE order (r[0]=MSB).
                // computeEqTable uses LE indexing (bit 0 of index → r[0]).
                // To match Jolt's EqPolynomial::evals (BE indexing: bit 0 → r[n-1]),
                // we reverse so that bit 0 of cycle index c maps to the LSB variable.
                var r_cycle_rev = try allocator.alloc(F, n_cycle_vars);
                defer allocator.free(r_cycle_rev);
                for (0..n_cycle_vars) |i| {
                    r_cycle_rev[i] = stage_r_cycles[s][n_cycle_vars - 1 - i];
                }
                const eq_table = try computeEqTableParallel(F, allocator, r_cycle_rev, n_cycle_vars, init_pool);
                defer allocator.free(eq_table);

                // F_s[k] = Sum_{c: PC(c)=k} eq(r_cycle_s, c)
                F_s_arrs[s] = try allocator.alloc(F, bytecode_K);
                @memset(F_s_arrs[s], F.zero());

                for (0..T) |c| {
                    const step = trace.steps.items[c];
                    const pc_idx = pc_map.getPCForStep(step);
                    if (pc_idx < bytecode_K) {
                        F_s_arrs[s][pc_idx] = F_s_arrs[s][pc_idx].add(eq_table[c]);
                    }
                    // Debug: print first cycles' PC mappings and trace step info for stage 0
                    if (s == 0 and c < 256) {
                        if (c < 8 or (!step.is_noop and c < 64)) {
                            const el = eq_table[c].toBytes();
                            const instr = step.instruction;
                            const opc: u8 = @truncate(instr & 0x7F);
                            const rd_raw: u8 = @truncate((instr >> 7) & 0x1F);
                            dbg("[BCRAF_PC] c={} noop={} pc=0x{x:0>8} pc_idx={} vsr={} opc=0x{x:0>2} rd_raw={} tstore={} eq_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                                c, @intFromBool(step.is_noop), step.pc, pc_idx, step.virtual_sequence_remaining,
                                opc, rd_raw, @intFromBool(step.is_termination_store),
                                el[0], el[1], el[2], el[3], el[4], el[5], el[6], el[7],
                            });
                        }
                    }
                }

                // val_with_raf[s][k] = Val_s(k) + RAF_s(k)
                val_with_raf_arrs[s] = try allocator.alloc(F, bytecode_K);
                for (0..bytecode_K) |k| {
                    var val_plus_raf = if (val_polys[s].len > k) val_polys[s][k] else F.zero();
                    // RAF terms
                    if (s == 0) {
                        val_plus_raf = val_plus_raf.add(gamma_powers[5].mul(int_poly[k]));
                    } else if (s == 2) {
                        val_plus_raf = val_plus_raf.add(gamma_powers[4].mul(int_poly[k]));
                    }
                    val_with_raf_arrs[s][k] = val_plus_raf;
                }

                // Compute claim from val_polys and F_s
                var recomputed_claim = F.zero();
                var val_only_claim = F.zero();
                var raf_only_claim = F.zero();
                for (0..bytecode_K) |k| {
                    recomputed_claim = recomputed_claim.add(F_s_arrs[s][k].mul(val_with_raf_arrs[s][k]));
                    val_only_claim = val_only_claim.add(F_s_arrs[s][k].mul(if (val_polys[s].len > k) val_polys[s][k] else F.zero()));
                    if (s == 0) {
                        raf_only_claim = raf_only_claim.add(F_s_arrs[s][k].mul(gamma_powers[5].mul(int_poly[k])));
                    } else if (s == 2) {
                        raf_only_claim = raf_only_claim.add(F_s_arrs[s][k].mul(gamma_powers[4].mul(int_poly[k])));
                    }
                }
                if (s == 0 or s == 2) {
                    const vocl = val_only_claim.toBytes();
                    const rocl = raf_only_claim.toBytes();
                    const rv_ext = external_stage_claims[s];
                    // Decompose external: ext = rv + raf_ext
                    // For s=0: raf_ext = gamma^5 * raf_claim (from opening claims)
                    // For s=2: raf_ext = gamma^4 * raf_shift_claim (from opening claims)
                    const raf_ext = rv_ext.sub(val_only_claim);
                    const rexl = raf_ext.toBytes();
                    dbg("[BCRAF_DECOMP_CLAIM] s={}: val_only_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}] raf_only_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}] raf_from_ext_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                        s,
                        vocl[0], vocl[1], vocl[2], vocl[3], vocl[4], vocl[5], vocl[6], vocl[7],
                        rocl[0], rocl[1], rocl[2], rocl[3], rocl[4], rocl[5], rocl[6], rocl[7],
                        rexl[0], rexl[1], rexl[2], rexl[3], rexl[4], rexl[5], rexl[6], rexl[7],
                    });
                    dbg("[BCRAF_DECOMP_CLAIM] s={}: raf_match={}\n", .{
                        s, @as(u8, if (raf_only_claim.eql(raf_ext)) 1 else 0),
                    });
                    // Also check: does val_only_claim match rv_claims[s] from external (without RAF)?
                    // external_stage_claims[s] = rv + raf, so rv = ext - raf_ext (using prover's RAF)
                    // But we can also check directly: does val_only match ext - raf_from_arrays?
                    const ext_minus_raf = rv_ext.sub(raf_only_claim);
                    dbg("[BCRAF_DECOMP_CLAIM] s={}: val_only==ext-raf_arrays? {}\n", .{
                        s, @as(u8, if (val_only_claim.eql(ext_minus_raf)) 1 else 0),
                    });
                }

                // Also compute claim directly from cycle iteration (for debugging)
                var direct_claim = F.zero();
                for (0..T) |c| {
                    const step2 = trace.steps.items[c];
                    const pc_idx2 = pc_map.getPCForStep(step2);
                    if (pc_idx2 < bytecode_K) {
                        direct_claim = direct_claim.add(eq_table[c].mul(val_with_raf_arrs[s][pc_idx2]));
                    }
                }

                // Compare with external claim (from opening claims)
                const rc_le = recomputed_claim.toBytes();
                const ec_le = external_stage_claims[s].toBytes();
                const dc_le = direct_claim.toBytes();
                dbg("[BCRAF_INIT] Stage {} recomputed_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}] external_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}] match={}\n", .{
                    s,
                    rc_le[0], rc_le[1], rc_le[2], rc_le[3], rc_le[4], rc_le[5], rc_le[6], rc_le[7],
                    ec_le[0], ec_le[1], ec_le[2], ec_le[3], ec_le[4], ec_le[5], ec_le[6], ec_le[7],
                    @as(u8, if (recomputed_claim.eql(external_stage_claims[s])) 1 else 0),
                });
                dbg("[BCRAF_INIT] Stage {} direct_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}] direct==recomp={}\n", .{
                    s,
                    dc_le[0], dc_le[1], dc_le[2], dc_le[3], dc_le[4], dc_le[5], dc_le[6], dc_le[7],
                    @as(u8, if (direct_claim.eql(recomputed_claim)) 1 else 0),
                });
                // Print F_s sums and eq_table sum for this stage
                {
                    var fs_sum = F.zero();
                    for (0..bytecode_K) |k| {
                        fs_sum = fs_sum.add(F_s_arrs[s][k]);
                    }
                    var eq_sum = F.zero();
                    for (0..T) |c| {
                        eq_sum = eq_sum.add(eq_table[c]);
                    }
                    const fsl = fs_sum.toBytes();
                    const eql = eq_sum.toBytes();
                    dbg("[BCRAF_INIT] Stage {} F_s_sum_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}] eq_sum_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}] match={}\n", .{
                        s,
                        fsl[0], fsl[1], fsl[2], fsl[3], fsl[4], fsl[5], fsl[6], fsl[7],
                        eql[0], eql[1], eql[2], eql[3], eql[4], eql[5], eql[6], eql[7],
                        @as(u8, if (fs_sum.eql(eq_sum)) 1 else 0),
                    });
                    // Print first 5 F_s values
                    for (0..@min(bytecode_K, 5)) |k| {
                        const fkl = F_s_arrs[s][k].toBytes();
                        dbg("[BCRAF_INIT] Stage {} F_s[{}]_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                            s, k, fkl[0], fkl[1], fkl[2], fkl[3], fkl[4], fkl[5], fkl[6], fkl[7],
                        });
                    }
                }

                // Debug: for stage 0, decompose by UnexpandedPC field alone
                // Compute Σ_k F_s[0][k] * bytecode_address[k] and compare with
                // Σ_t eq(r,t) * witness[t].UnexpandedPC (i.e., the opening claim)
                if (s == 0) {
                    // Compute Σeq*imm from trace using TWO methods:
                    // 1. val-poly encoding (same as bytecode entry imm_field)
                    // 2. R1CS witness encoding (same as computeClaimedInputs)
                    var imm_valpoly = F.zero();
                    var imm_r1cs = F.zero();
                    var addr_trace = F.zero();
                    var diff_count: usize = 0;
                    for (0..T) |c| {
                        const step3 = trace.steps.items[c];
                        const eq_val = eq_table[c];
                        if (eq_val.eql(F.zero())) continue;

                        addr_trace = addr_trace.add(eq_val.mul(F.fromU64(step3.unexpanded_pc)));

                        const inst = step3.instruction;
                        const opc: u8 = @truncate(inst & 0x7F);

                        // Method 1: val poly encoding
                        const decoded_imm = instruction_mod.DecodedInstruction.decode(inst).imm;
                        const vp_imm: F = if (step3.is_noop)
                            F.zero()
                        else if ((opc == 0x63) or (opc == 0x23) or (opc == 0x03))
                            fieldFromI128(F, @as(i128, @as(i64, decoded_imm)))
                        else
                            F.fromU64(@as(u64, @bitCast(@as(i64, decoded_imm))));
                        imm_valpoly = imm_valpoly.add(eq_val.mul(vp_imm));

                        // Method 2: R1CS witness encoding (same as fromTraceStepWithPCMap)
                        const r1cs_imm: F = blk: {
                            if (step3.is_noop) break :blk F.zero();
                            // VirtualMULI special case
                            if (opc == 0x2B) {
                                const shamt_raw: u32 = inst >> 20;
                                const shamt: u6 = @truncate(shamt_raw & 0x3F);
                                const multiplier: u64 = @as(u64, 1) << shamt;
                                break :blk F.fromU64(multiplier);
                            }
                            const is_identity_add = switch (opc) {
                                0x13 => ((inst >> 12) & 0x7) == 0, // ADDI
                                0x1b => ((inst >> 12) & 0x7) == 0, // ADDIW
                                0x6f => true, // JAL
                                0x67 => true, // JALR
                                else => false,
                            };
                            if (is_identity_add) {
                                // Use computeUnsignedImmediate logic
                                switch (opc) {
                                    0x13, 0x03, 0x67, 0x1b => {
                                        const imm12: u32 = inst >> 20;
                                        const imm_signed: i64 = @as(i64, @as(i32, @bitCast(imm12 << 20)) >> 20);
                                        break :blk F.fromU64(@as(u64, @bitCast(imm_signed)));
                                    },
                                    0x6F => {
                                        const imm20 = (inst >> 31) & 0x1;
                                        const imm10_1 = (inst >> 21) & 0x3FF;
                                        const imm11 = (inst >> 20) & 0x1;
                                        const imm19_12 = (inst >> 12) & 0xFF;
                                        const raw2 = (imm20 << 20) | (imm19_12 << 12) | (imm11 << 11) | (imm10_1 << 1);
                                        const imm_signed: i64 = @as(i64, @as(i32, @bitCast(raw2 << 11)) >> 11);
                                        break :blk F.fromU64(@as(u64, @bitCast(imm_signed)));
                                    },
                                    else => break :blk F.zero(),
                                }
                            }
                            // deriveImmediate logic
                            switch (opc) {
                                0x13, 0x03, 0x67, 0x1b => {
                                    const imm12 = inst >> 20;
                                    if (imm12 & 0x800 != 0) {
                                        break :blk F.zero().sub(F.fromU64((~imm12 + 1) & 0xFFF));
                                    }
                                    break :blk F.fromU64(imm12);
                                },
                                0x23 => {
                                    const imm4_0 = (inst >> 7) & 0x1F;
                                    const imm11_5 = (inst >> 25) & 0x7F;
                                    const ival = (imm11_5 << 5) | imm4_0;
                                    if (ival & 0x800 != 0) {
                                        break :blk F.zero().sub(F.fromU64((~ival + 1) & 0xFFF));
                                    }
                                    break :blk F.fromU64(ival);
                                },
                                0x63 => {
                                    const bimm12 = (inst >> 31) & 0x1;
                                    const bimm10_5 = (inst >> 25) & 0x3F;
                                    const bimm4_1 = (inst >> 8) & 0xF;
                                    const bimm11 = (inst >> 7) & 0x1;
                                    const bval = (bimm12 << 12) | (bimm11 << 11) | (bimm10_5 << 5) | (bimm4_1 << 1);
                                    if (bval & 0x1000 != 0) {
                                        break :blk F.zero().sub(F.fromU64((~bval + 1) & 0x1FFF));
                                    }
                                    break :blk F.fromU64(bval);
                                },
                                0x37, 0x17 => {
                                    // Sign-extend U-type immediate to 64 bits
                                    const imm_u32_6: u32 = inst & 0xFFFFF000;
                                    const imm_sext_6: u64 = @bitCast(@as(i64, @as(i32, @bitCast(imm_u32_6))));
                                    break :blk F.fromU64(imm_sext_6);
                                },
                                else => break :blk F.zero(),
                            }
                        };
                        imm_r1cs = imm_r1cs.add(eq_val.mul(r1cs_imm));

                        // Print per-cycle difference if non-zero
                        if (!vp_imm.eql(r1cs_imm) and c < 256) {
                            diff_count += 1;
                            if (diff_count <= 10) {
                                const vpl = vp_imm.toBytes();
                                const r1l = r1cs_imm.toBytes();
                                dbg("[BCRAF_IMM_DIFF] c={} opc=0x{x:0>2} noop={} vp_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}] r1cs_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                                    c, opc, @intFromBool(step3.is_noop),
                                    vpl[0], vpl[1], vpl[2], vpl[3], vpl[4], vpl[5], vpl[6], vpl[7],
                                    r1l[0], r1l[1], r1l[2], r1l[3], r1l[4], r1l[5], r1l[6], r1l[7],
                                });
                            }
                        }
                    }
                    // Method 3: ACTUAL bytecode entry imm via val_polys[0][k] contribution
                    // val_polys[0][k] = unexpanded_pc + γ₁¹·imm + Σγ₁^(2+i)·cf[i]
                    // We compute Σ_k F_s[k] * val_polys[0][k] directly
                    var vp_sum = F.zero();
                    for (0..bytecode_K) |k2| {
                        vp_sum = vp_sum.add(F_s_arrs[0][k2].mul(val_polys[0][k2]));
                    }
                    const vpsl = vp_sum.toBytes();
                    dbg("[BCRAF_DECOMP] Σ F_s*Val_0 (from vp arrays)_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                        vpsl[0], vpsl[1], vpsl[2], vpsl[3], vpsl[4], vpsl[5], vpsl[6], vpsl[7],
                    });
                    const addr_le = addr_trace.toBytes();
                    const imm_vp_le = imm_valpoly.toBytes();
                    const imm_r1_le = imm_r1cs.toBytes();
                    dbg("[BCRAF_DECOMP] Σeq*addr_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                        addr_le[0], addr_le[1], addr_le[2], addr_le[3], addr_le[4], addr_le[5], addr_le[6], addr_le[7],
                    });
                    dbg("[BCRAF_DECOMP] Σeq*imm(decode)_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                        imm_vp_le[0], imm_vp_le[1], imm_vp_le[2], imm_vp_le[3], imm_vp_le[4], imm_vp_le[5], imm_vp_le[6], imm_vp_le[7],
                    });
                    dbg("[BCRAF_DECOMP] Σeq*imm(r1cs)_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}] diff_count={}\n", .{
                        imm_r1_le[0], imm_r1_le[1], imm_r1_le[2], imm_r1_le[3], imm_r1_le[4], imm_r1_le[5], imm_r1_le[6], imm_r1_le[7],
                        diff_count,
                    });
                }

                // Use val_poly-derived claims for sumcheck consistency
                // The sumcheck polynomial must sum to the claimed value,
                // and the polynomial is built from val_polys and F_s.
                // If we use external claims that differ from the actual polynomial sum,
                // the sumcheck will be inconsistent.
                stage_claims_init[s] = recomputed_claim;

                // Debug: Check if recomputed matches external
                if (comptime debug_verbose) {
                    const match_ext = @as(u8, if (recomputed_claim.eql(external_stage_claims[s])) 1 else 0);
                    if (match_ext == 0) {
                        const rc_full = recomputed_claim.toBytesBE();
                        const ec_full = external_stage_claims[s].toBytesBE();
                        dbg("[BCRAF_MISMATCH] Stage {d}: recomputed != external!\n", .{s});
                        dbg("  recomputed_LE=[", .{});
                        for (0..32) |bi| dbg("{x:0>2}", .{rc_full[31 - bi]});
                        dbg("]\n  external_LE=[", .{});
                        for (0..32) |bi| dbg("{x:0>2}", .{ec_full[31 - bi]});
                        dbg("]\n", .{});
                    }
                }
            }

            // Debug: print per-stage claims with full aggregation detail
            {
                var total = F.zero();
                for (0..5) |s| {
                    const term = gamma_powers[s].mul(stage_claims_init[s]);
                    total = total.add(term);
                    const sc_le = stage_claims_init[s].toBytes();
                    const gp_le = gamma_powers[s].toBytes();
                    const tm_le = term.toBytes();
                    dbg("[BCRAF_AGG_PR] s={} sc==ext={}", .{
                        s, @as(u8, if (stage_claims_init[s].eql(external_stage_claims[s])) 1 else 0),
                    });
                    dbg(" gp=[{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}]", .{
                        gp_le[0], gp_le[1], gp_le[2], gp_le[3], gp_le[4], gp_le[5], gp_le[6], gp_le[7],
                    });
                    dbg(" sc=[{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}]", .{
                        sc_le[0], sc_le[1], sc_le[2], sc_le[3], sc_le[4], sc_le[5], sc_le[6], sc_le[7],
                    });
                    dbg(" term=[{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}]", .{
                        tm_le[0], tm_le[1], tm_le[2], tm_le[3], tm_le[4], tm_le[5], tm_le[6], tm_le[7],
                    });
                    dbg("\n", .{});
                }
                const tl = total.toBytes();
                dbg("[BCRAF_AGG_PR] total_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{tl[0], tl[1], tl[2], tl[3], tl[4], tl[5], tl[6], tl[7]});
            }

            return Self{
                .F_s_arrs = F_s_arrs,
                .val_with_raf = val_with_raf_arrs,
                .stage_claims = stage_claims_init,
                .combined = null,
                .ra_chunks = null,
                .phase = 0,
                .bytecode_log_k = bytecode_log_k,
                .n_cycle_vars = n_cycle_vars,
                .bytecode_d = bytecode_d,
                .log_k_chunk = log_k_chunk,
                .current_len = bytecode_K,
                .addr_rounds_done = 0,
                .bound_vals_stored = [_]F{F.zero()} ** 5,
                .trace = trace,
                .pc_map = pc_map,
                .stage_r_cycles = stage_r_cycles,
                .gamma_powers = gamma_powers,
                .val_polys = val_polys,
                .int_poly = int_poly,
                .allocator = allocator,
                .pool = init_pool,
            };
        }

        pub fn deinit(self: *Self) void {
            // Phase 1 arrays (freed during transition if we got that far)
            for (0..5) |s| {
                if (self.F_s_arrs[s].len > 0) self.allocator.free(self.F_s_arrs[s]);
                if (self.val_with_raf[s].len > 0) self.allocator.free(self.val_with_raf[s]);
            }
            // Phase 2 arrays
            if (self.combined) |cc| self.allocator.free(cc);
            if (self.ra_chunks) |chunks| {
                for (chunks) |arr| self.allocator.free(arr);
                self.allocator.free(chunks);
            }
            for (&self.val_polys) |vp| {
                if (vp.len > 0) self.allocator.free(vp);
            }
            self.allocator.free(self.int_poly);
        }

        /// Phase 1: degree-2 round poly over address vars
        /// Returns .{ agg=[p(0), p(2)], per_stage=[5][eval_0, eval_2] }
        /// Matches Jolt's approach: product of F_s (linear) and val_with_raf (linear) = degree 2
        pub fn computeRoundPolyPhase1(self: *Self) struct { agg: [2]F, per_stage: [5][2]F } {
            const half = self.current_len / 2;
            var per_stage: [5][2]F = undefined;

            if (self.pool) |pool| {
                // Compute 5 stages in parallel, each accumulating [2]F
                const Ctx = struct {
                    F_s_arrs: *const [5][]F,
                    val_with_raf: *const [5][]F,
                    half: usize,
                    results: *[5][2]F,
                };
                const ctx = Ctx{
                    .F_s_arrs = &self.F_s_arrs,
                    .val_with_raf = &self.val_with_raf,
                    .half = half,
                    .results = &per_stage,
                };
                pool.parallelForForce(5, ctx, struct {
                    fn f(c: Ctx, s: usize) void {
                        var eval_at_0 = F.zero();
                        var eval_at_2 = F.zero();
                        for (0..c.half) |k| {
                            const f_lo = c.F_s_arrs[s][2 * k];
                            const f_hi = c.F_s_arrs[s][2 * k + 1];
                            const v_lo = c.val_with_raf[s][2 * k];
                            const v_hi = c.val_with_raf[s][2 * k + 1];

                            eval_at_0 = eval_at_0.add(f_lo.mul(v_lo));

                            const f_at_2 = f_hi.add(f_hi).sub(f_lo);
                            const v_at_2 = v_hi.add(v_hi).sub(v_lo);
                            eval_at_2 = eval_at_2.add(f_at_2.mul(v_at_2));
                        }
                        c.results[s] = [2]F{ eval_at_0, eval_at_2 };
                    }
                }.f);
            } else {
                for (0..5) |s| {
                    var eval_at_0 = F.zero();
                    var eval_at_2 = F.zero();

                    for (0..half) |k| {
                        const f_lo = self.F_s_arrs[s][2 * k];
                        const f_hi = self.F_s_arrs[s][2 * k + 1];
                        const v_lo = self.val_with_raf[s][2 * k];
                        const v_hi = self.val_with_raf[s][2 * k + 1];

                        eval_at_0 = eval_at_0.add(f_lo.mul(v_lo));

                        const f_at_2 = f_hi.add(f_hi).sub(f_lo);
                        const v_at_2 = v_hi.add(v_hi).sub(v_lo);
                        eval_at_2 = eval_at_2.add(f_at_2.mul(v_at_2));
                    }

                    per_stage[s] = [2]F{ eval_at_0, eval_at_2 };
                }
            }

            var agg_eval_0 = F.zero();
            var agg_eval_2 = F.zero();
            for (0..5) |s| {
                agg_eval_0 = agg_eval_0.add(self.gamma_powers[s].mul(per_stage[s][0]));
                agg_eval_2 = agg_eval_2.add(self.gamma_powers[s].mul(per_stage[s][1]));
            }

            return .{ .agg = [2]F{ agg_eval_0, agg_eval_2 }, .per_stage = per_stage };
        }

        /// Bind challenge and update per-stage claims from polynomial evaluation
        /// per_stage_evals: [5][eval_0, eval_2] from computeRoundPolyPhase1
        pub fn bindChallengePhase1(self: *Self, r: F, per_stage_evals: [5][2]F) void {
            const half = self.current_len / 2;
            const two = F.fromU64(2);
            const two_inv = two.inverse().?;

            const bindStage = struct {
                fn f(
                    F_s: []F,
                    vwr: []F,
                    stage_claim: *F,
                    h: usize,
                    challenge: F,
                    pse: [2]F,
                    t_inv: F,
                ) void {
                    // Bind F_s and val_with_raf arrays
                    for (0..h) |k| {
                        F_s[k] = F_s[2 * k].add(challenge.mul(F_s[2 * k + 1].sub(F_s[2 * k])));
                        vwr[k] = vwr[2 * k].add(challenge.mul(vwr[2 * k + 1].sub(vwr[2 * k])));
                    }

                    // Update per-stage claim
                    const p0 = pse[0];
                    const p2 = pse[1];
                    const p1 = stage_claim.*.sub(p0);
                    const a0 = p0;
                    const a2 = p2.sub(p1.add(p1)).add(p0).mul(t_inv);
                    const a1 = p1.sub(p0).sub(a2);
                    stage_claim.* = a0.add(challenge.mul(a1.add(challenge.mul(a2))));
                }
            }.f;

            if (self.pool) |pool| {
                const Ctx = struct {
                    F_s_arrs: *[5][]F,
                    val_with_raf: *[5][]F,
                    stage_claims: *[5]F,
                    half: usize,
                    r: F,
                    per_stage_evals: [5][2]F,
                    two_inv: F,
                };
                const ctx = Ctx{
                    .F_s_arrs = &self.F_s_arrs,
                    .val_with_raf = &self.val_with_raf,
                    .stage_claims = &self.stage_claims,
                    .half = half,
                    .r = r,
                    .per_stage_evals = per_stage_evals,
                    .two_inv = two_inv,
                };
                pool.parallelForForce(5, ctx, struct {
                    fn f(c: Ctx, s: usize) void {
                        bindStage(
                            c.F_s_arrs[s],
                            c.val_with_raf[s],
                            &c.stage_claims[s],
                            c.half,
                            c.r,
                            c.per_stage_evals[s],
                            c.two_inv,
                        );
                    }
                }.f);
            } else {
                for (0..5) |s| {
                    bindStage(
                        self.F_s_arrs[s],
                        self.val_with_raf[s],
                        &self.stage_claims[s],
                        half,
                        r,
                        per_stage_evals[s],
                        two_inv,
                    );
                }
            }

            self.current_len = half;
            self.addr_rounds_done += 1;
        }

        /// Transition from Phase 1 to Phase 2 after binding all address vars
        /// r_address_challenges are the challenges from Phase 1 in binding order (low-to-high)
        pub fn transitionToPhase2(
            self: *Self,
            r_address_challenges: []const F, // Low-to-high binding order from the sumcheck
        ) !void {
            const T: usize = @as(usize, 1) << @intCast(self.n_cycle_vars);
            const bytecode_K: usize = @as(usize, 1) << @intCast(self.bytecode_log_k);

            // The Phase 1 sumcheck binds variables in LowToHigh order:
            // r_address_challenges[0] = r_0 (bound to LSB of index), ..., [n-1] = r_{n-1} (MSB).
            //
            // For computing val_eval = Σ_k val[k] * eq(k, r), we need:
            // eq[k] = Π_j (r_j if bit j of k is 1, else 1-r_j).
            // Our computeEqTable with LE indexing gives exactly this when passed
            // the challenges in LH order.
            //
            // For RA chunk computation, Jolt uses r_address_BE = [r_{n-1},...,r_0]
            // and chunks sequentially: chunk 0 = MSB vars, chunk d-1 = LSB vars.
            // We keep r_address_be for RA chunk slicing (same convention as before).

            // Print address challenges (ALWAYS ON for debugging)
            {
                dbg("[BCRAF_TRANS] r_address_challenges (len={}, LH order):\n", .{self.bytecode_log_k});
                for (0..self.bytecode_log_k) |i| {
                    const ch_be = r_address_challenges[i].toBytesBE();
                    dbg("  ch[{d}]_LE=[", .{i});
                    for (0..32) |bi| dbg("{x:0>2}", .{ch_be[31 - bi]});
                    dbg("]\n", .{});
                }
            }

            // Compute r_address_be for RA chunk slicing (same as before)
            var r_address_be = try self.allocator.alloc(F, self.bytecode_log_k);
            defer self.allocator.free(r_address_be);
            for (0..self.bytecode_log_k) |i| {
                r_address_be[i] = r_address_challenges[self.bytecode_log_k - 1 - i];
            }

            // Compute bound_vals[s] = Val_s(r_address) + RAF_s(r_address)
            // The sumcheck binds variables MSB-first: r_address_challenges[0] = MSB.
            // But val_poly coefficients are indexed with bit 0 = LSB.
            // Jolt's verifier reverses challenges (normalize_opening_point) before evaluate,
            // so r[0] = LSB challenge maps to bit 0 of coefficient index.
            // We must do the same: use r_address_be (reversed) for the eq table.
            const eq_addr = try computeEqTableParallel(F, self.allocator, r_address_be, self.bytecode_log_k, self.pool);
            defer self.allocator.free(eq_addr);

            // Debug: eq_addr entries (ALWAYS ON, full 32 bytes)
            {
                for (0..bytecode_K) |ek| {
                    const eab = eq_addr[ek].toBytesBE();
                    dbg("[ZOLT_EQ_ADDR] eq[{d}]_LE=[", .{ek});
                    for (0..32) |bi| dbg("{x:0>2}", .{eab[31 - bi]});
                    dbg("]\n", .{});
                }
            }

            // Debug: val_polys entries (ALWAYS ON for debugging)
            {
                for (0..5) |vs| {
                    for (0..bytecode_K) |kk| {
                        const vpk = self.val_polys[vs][kk].toBytesBE();
                        dbg("[ZOLT_VP] Val[{d}][{d}]_LE=[", .{ vs, kk });
                        for (0..8) |bi| dbg("{x:0>2}", .{vpk[31 - bi]});
                        dbg("]\n", .{});
                    }
                }
            }

            var bound_vals: [5]F = undefined;
            for (0..5) |s| {
                var val_eval = F.zero();
                const max_k = @min(self.val_polys[s].len, bytecode_K);
                for (0..max_k) |k| {
                    const term = self.val_polys[s][k].mul(eq_addr[k]);
                    val_eval = val_eval.add(term);
                    if (s == 0) {
                        const t_be = term.toBytesBE();
                        const ps_be = val_eval.toBytesBE();
                        dbg("[DOTPROD] s=0 k={d} term_LE=[", .{k});
                        for (0..8) |bi| dbg("{x:0>2}", .{t_be[31 - bi]});
                        dbg("] partial_sum_LE=[", .{});
                        for (0..8) |bi| dbg("{x:0>2}", .{ps_be[31 - bi]});
                        dbg("]\n", .{});
                    }
                }

                // Add RAF terms (identity polynomial contribution)
                // Stage 0: RAF = gamma^5 * identity_eval
                // Stage 2: RAF = gamma^4 * identity_eval
                if (s == 0) {
                    var identity_eval = F.zero();
                    for (0..bytecode_K) |k| {
                        identity_eval = identity_eval.add(self.int_poly[k].mul(eq_addr[k]));
                    }
                    const raf_contrib = self.gamma_powers[5].mul(identity_eval);
                    val_eval = val_eval.add(raf_contrib);
                    // Print identity_eval, gamma[5], RAF contribution, val_before_raf
                    const ie_be = identity_eval.toBytesBE();
                    const g5_be = self.gamma_powers[5].toBytesBE();
                    const rc_be = raf_contrib.toBytesBE();
                    dbg("[TRANS_RAF] s=0: identity_eval_LE=[", .{});
                    for (0..32) |bi| dbg("{x:0>2}", .{ie_be[31 - bi]});
                    dbg("] gamma5_LE=[", .{});
                    for (0..32) |bi| dbg("{x:0>2}", .{g5_be[31 - bi]});
                    dbg("] raf_LE=[", .{});
                    for (0..32) |bi| dbg("{x:0>2}", .{rc_be[31 - bi]});
                    dbg("]\n", .{});
                } else if (s == 2) {
                    var identity_eval = F.zero();
                    for (0..bytecode_K) |k| {
                        identity_eval = identity_eval.add(self.int_poly[k].mul(eq_addr[k]));
                    }
                    const raf_contrib = self.gamma_powers[4].mul(identity_eval);
                    val_eval = val_eval.add(raf_contrib);
                    const ie_be = identity_eval.toBytesBE();
                    const g4_be = self.gamma_powers[4].toBytesBE();
                    const rc_be = raf_contrib.toBytesBE();
                    dbg("[TRANS_RAF] s=2: identity_eval_LE=[", .{});
                    for (0..32) |bi| dbg("{x:0>2}", .{ie_be[31 - bi]});
                    dbg("] gamma4_LE=[", .{});
                    for (0..32) |bi| dbg("{x:0>2}", .{g4_be[31 - bi]});
                    dbg("] raf_LE=[", .{});
                    for (0..32) |bi| dbg("{x:0>2}", .{rc_be[31 - bi]});
                    dbg("]\n", .{});
                }

                // bound_vals[s] = gamma^s * val_with_raf[s][0]
                // Use the Phase 1 bound value directly (like Jolt's poly.final_sumcheck_claim()),
                // NOT the recomputed val_eval from the eq table.
                bound_vals[s] = self.gamma_powers[s].mul(self.val_with_raf[s][0]);
                self.bound_vals_stored[s] = bound_vals[s];

                // DIAGNOSTIC: compare re-computed val_eval with Phase 1 bound val_with_raf[s][0]
                {
                    const phase1_bound = self.val_with_raf[s][0];
                    const match_p1 = val_eval.eql(phase1_bound);
                    const p1b = phase1_bound.toBytesBE();
                    const ve_b = val_eval.toBytesBE();
                    dbg("[TRANS_CHECK] stage[{}]: val_eval_recomp_LE=[", .{s});
                    for (0..32) |bi| dbg("{x:0>2}", .{ve_b[31 - bi]});
                    dbg("] phase1_bound_LE=[", .{});
                    for (0..32) |bi| dbg("{x:0>2}", .{p1b[31 - bi]});
                    dbg("] match={}\n", .{@as(u8, if (match_p1) 1 else 0)});

                    // Also print F_s[0] for this stage (the eq contribution after Phase 1 binding)
                    const fs0 = self.F_s_arrs[s][0];
                    const fs0b = fs0.toBytesBE();
                    dbg("[TRANS_CHECK] stage[{}]: F_s[0]_LE=[", .{s});
                    for (0..32) |bi| dbg("{x:0>2}", .{fs0b[31 - bi]});
                    dbg("]\n", .{});

                    // Print stage_claims[s] = F_s[0] * val_with_raf[s][0] (should match)
                    const sc = self.stage_claims[s];
                    const sc_recomp = fs0.mul(phase1_bound);
                    const scb = sc.toBytesBE();
                    dbg("[TRANS_CHECK] stage[{}]: stage_claim_LE=[", .{s});
                    for (0..32) |bi| dbg("{x:0>2}", .{scb[31 - bi]});
                    dbg("] F_s*val_bound=[", .{});
                    const src = sc_recomp.toBytesBE();
                    for (0..32) |bi| dbg("{x:0>2}", .{src[31 - bi]});
                    dbg("] match={}\n", .{@as(u8, if (sc.eql(sc_recomp)) 1 else 0)});
                }

                // Debug: Print val_eval and bound_val for comparison with Jolt verifier
                if (comptime debug_verbose) {
                    const ve_be = val_eval.toBytesBE();
                    const bv_be = bound_vals[s].toBytesBE();
                    dbg("[BCRAF_TRANS] stage[{}]: val_eval_LE=[", .{s});
                    for (0..32) |bi| dbg("{x:0>2}", .{ve_be[31 - bi]});
                    dbg("] bound_val_LE=[", .{});
                    for (0..32) |bi| dbg("{x:0>2}", .{bv_be[31 - bi]});
                    dbg("]\n", .{});
                }
            }

            // Build RA chunk polynomials for cycle binding
            // ra_chunks[i][c] = eq(r_addr_chunk_i, PC_chunk_i(c))
            //
            // Like Jolt's compute_r_address_chunks: pad r_address_be with zeros at MSB
            // to make length a multiple of log_k_chunk, then split into d chunks of
            // exactly log_k_chunk variables each.
            const padded_len = self.bytecode_d * self.log_k_chunk;
            const pad_count = padded_len - self.bytecode_log_k;
            var r_address_be_padded = try self.allocator.alloc(F, padded_len);
            defer self.allocator.free(r_address_be_padded);
            // Pad MSB end with zeros (Jolt prepends zeros to r_address which is BE)
            for (0..pad_count) |i| {
                r_address_be_padded[i] = F.zero();
            }
            for (0..self.bytecode_log_k) |i| {
                r_address_be_padded[pad_count + i] = r_address_be[i];
            }

            self.ra_chunks = try self.allocator.alloc([]F, self.bytecode_d);
            const chunk_K: usize = @as(usize, 1) << @intCast(self.log_k_chunk);

            for (0..self.bytecode_d) |i| {
                self.ra_chunks.?[i] = try self.allocator.alloc(F, T);

                const chunk_start = i * self.log_k_chunk;
                const chunk_end = chunk_start + self.log_k_chunk;

                // Jolt uses EqPolynomial::evals (BE indexing) over r_address_BE chunks.
                // Our computeEqTable uses LE indexing. To match, we reverse the chunk.
                const r_chunk_be = r_address_be_padded[chunk_start..chunk_end];
                var r_chunk_rev = try self.allocator.alloc(F, self.log_k_chunk);
                defer self.allocator.free(r_chunk_rev);
                for (0..self.log_k_chunk) |ci| {
                    r_chunk_rev[ci] = r_chunk_be[self.log_k_chunk - 1 - ci];
                }
                const eq_table = try computeEqTable(F, self.allocator, r_chunk_rev, self.log_k_chunk);
                defer self.allocator.free(eq_table);

                for (0..T) |c| {
                    const step = self.trace.steps.items[c];
                    // Convert ELF address to bytecode array index
                    const pc = self.pc_map.getPCForStep(step);
                    if (pc < bytecode_K) {
                        // Extract chunk using MSB-first ordering with log_k_chunk bits
                        const chunk_val = extractChunkMSB(pc, i, self.bytecode_d, self.log_k_chunk);
                        if (chunk_val < chunk_K) {
                            self.ra_chunks.?[i][c] = eq_table[chunk_val];
                        } else {
                            self.ra_chunks.?[i][c] = F.zero();
                        }
                    } else {
                        self.ra_chunks.?[i][c] = F.zero();
                    }
                }
            }

            // Build eq tables per stage for cycle binding
            // stage_r_cycles[s] is in BE order; reverse for LE computeEqTable
            var eq_per_stage: [5][]F = undefined;
            for (0..5) |s| {
                var r_cycle_rev = try self.allocator.alloc(F, self.n_cycle_vars);
                defer self.allocator.free(r_cycle_rev);
                for (0..self.n_cycle_vars) |i| {
                    r_cycle_rev[i] = self.stage_r_cycles[s][self.n_cycle_vars - 1 - i];
                }
                eq_per_stage[s] = try computeEqTableParallel(F, self.allocator, r_cycle_rev, self.n_cycle_vars, self.pool);
            }

            // Compute combined[c] = Sum_s bound_vals[s] * eq_s(c)
            self.combined = try self.allocator.alloc(F, T);
            for (0..T) |c| {
                var val = F.zero();
                for (0..5) |s| {
                    val = val.add(bound_vals[s].mul(eq_per_stage[s][c]));
                }
                self.combined.?[c] = val;
            }

            // Debug: verify Π_i ra_chunk_i(c) = eq_addr[PC(c)] for each cycle
            // ALWAYS ON: check ALL cycles to find mismatches
            {
                dbg("[BCRAF_RA] bytecode_d={} log_k_chunk={} bytecode_log_k={} T={}\n", .{
                    self.bytecode_d, self.log_k_chunk, self.bytecode_log_k, T,
                });
                var mismatch_count: usize = 0;
                for (0..T) |c| {
                    var ra_prod = F.one();
                    for (0..self.bytecode_d) |i| {
                        ra_prod = ra_prod.mul(self.ra_chunks.?[i][c]);
                    }
                    const step = self.trace.steps.items[c];
                    const pc = self.pc_map.getPCForStep(step);
                    const full_eq = if (pc < bytecode_K) eq_addr[pc] else F.zero();
                    const match_c = if (ra_prod.eql(full_eq)) @as(u8, 1) else @as(u8, 0);
                    if (match_c == 0) mismatch_count += 1;
                    if (match_c == 0 and mismatch_count <= 5) {
                        // Print mismatch details (limited to first 5)
                        const rp_be = ra_prod.toBytesBE();
                        const fe_be = full_eq.toBytesBE();
                        dbg("[BCRAF_RA_MISMATCH] c={} pc={} ra_prod_LE=[", .{ c, pc });
                        for (0..8) |bi| dbg("{x:0>2}", .{rp_be[31 - bi]});
                        dbg("] eq_addr[pc]_LE=[", .{});
                        for (0..8) |bi| dbg("{x:0>2}", .{fe_be[31 - bi]});
                        dbg("]\n", .{});
                        // Print per-chunk values
                        for (0..self.bytecode_d) |i| {
                            const cv = extractChunkMSB(@intCast(pc), i, self.bytecode_d, self.log_k_chunk);
                            const rv = self.ra_chunks.?[i][c];
                            const rv_be = rv.toBytesBE();
                            dbg("[BCRAF_RA_MISMATCH]   chunk[{}] chunk_val={} ra_LE=[", .{ i, cv });
                            for (0..8) |bi| dbg("{x:0>2}", .{rv_be[31 - bi]});
                            dbg("]\n", .{});
                        }
                    }
                }
                dbg("[BCRAF_RA] total mismatches: {}/{}\n", .{ mismatch_count, T });

                // Also check Σ_c eq_s(c) * eq_addr[PC(c)] vs F_s_bound for stage 0
                var direct_sum = F.zero();
                for (0..T) |c| {
                    const step = self.trace.steps.items[c];
                    const pc = self.pc_map.getPCForStep(step);
                    if (pc < bytecode_K) {
                        direct_sum = direct_sum.add(eq_per_stage[0][c].mul(eq_addr[pc]));
                    }
                }
                const ds_be = direct_sum.toBytesBE();
                const fb_be = self.F_s_arrs[0][0].toBytesBE();
                dbg("[BCRAF_RA] stage0: Σeq_s*eq_addr[PC]_LE=[", .{});
                for (0..8) |bi| dbg("{x:0>2}", .{ds_be[31 - bi]});
                dbg("] F_s_bound_LE=[", .{});
                for (0..8) |bi| dbg("{x:0>2}", .{fb_be[31 - bi]});
                dbg("] match={}\n", .{if (direct_sum.eql(self.F_s_arrs[0][0])) @as(u8, 1) else @as(u8, 0)});
            }

            // Debug: verify the claim after transition - PER STAGE
            if (comptime debug_verbose) {
                var total_claim = F.zero();
                for (0..5) |s| {
                    var stage_sum = F.zero();
                    for (0..T) |c| {
                        var ra_prod = F.one();
                        for (0..self.bytecode_d) |i| {
                            ra_prod = ra_prod.mul(self.ra_chunks.?[i][c]);
                        }
                        stage_sum = stage_sum.add(eq_per_stage[s][c].mul(ra_prod));
                    }
                    const f_s_bound = self.F_s_arrs[s][0];
                    const match_f = if (stage_sum.eql(f_s_bound)) @as(u8, 1) else @as(u8, 0);
                    const ss_be = stage_sum.toBytesBE();
                    const fb_be = f_s_bound.toBytesBE();
                    dbg("[BCRAF_P2CHK] stage[{}]: Σeq*ra_LE=[", .{s});
                    for (0..8) |bi| dbg("{x:0>2}", .{ss_be[31 - bi]});
                    dbg("] F_s_bound_LE=[", .{});
                    for (0..8) |bi| dbg("{x:0>2}", .{fb_be[31 - bi]});
                    dbg("] match={}\n", .{match_f});

                    const p2_claim = bound_vals[s].mul(stage_sum);
                    const p1_gamma_claim = self.gamma_powers[s].mul(self.stage_claims[s]);
                    const match_s = if (p2_claim.eql(p1_gamma_claim)) @as(u8, 1) else @as(u8, 0);
                    const p2_be = p2_claim.toBytesBE();
                    const p1_be = p1_gamma_claim.toBytesBE();
                    dbg("[BCRAF_P2CHK] stage[{}]: P2_claim_LE=[", .{s});
                    for (0..8) |bi| dbg("{x:0>2}", .{p2_be[31 - bi]});
                    dbg("] P1_gamma_claim_LE=[", .{});
                    for (0..8) |bi| dbg("{x:0>2}", .{p1_be[31 - bi]});
                    dbg("] match={}\n", .{match_s});
                    total_claim = total_claim.add(p2_claim);
                }
                const tc_le = total_claim.toBytes();
                dbg("[BCRAF_P2CHK] Phase2 total LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                    tc_le[0], tc_le[1], tc_le[2], tc_le[3], tc_le[4], tc_le[5], tc_le[6], tc_le[7],
                });

                var p1_total = F.zero();
                for (0..5) |s| {
                    p1_total = p1_total.add(self.gamma_powers[s].mul(self.stage_claims[s]));
                }
                const p1t_le = p1_total.toBytes();
                dbg("[BCRAF_TRANS] Phase1 total LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                    p1t_le[0], p1t_le[1], p1t_le[2], p1t_le[3], p1t_le[4], p1t_le[5], p1t_le[6], p1t_le[7],
                });
            }

            // Free eq tables
            for (0..5) |s| {
                self.allocator.free(eq_per_stage[s]);
            }

            // Free Phase 1 arrays (no longer needed)
            // Replace with zero-length allocations so deinit doesn't double-free
            for (0..5) |s| {
                self.allocator.free(self.F_s_arrs[s]);
                self.F_s_arrs[s] = try self.allocator.alloc(F, 0);
                self.allocator.free(self.val_with_raf[s]);
                self.val_with_raf[s] = try self.allocator.alloc(F, 0);
            }

            self.current_len = T;
            self.phase = 1;
        }

        /// Phase 2: degree bytecode_d+1 round poly
        /// Returns evals in Toom-Cook format: [p(0), p(1), ..., p(d), p_inf]
        pub fn computeRoundPolyPhase2(self: *Self, allocator: Allocator) ![]F {
            const half = self.current_len / 2;
            const combined = self.combined.?;
            const ra_chunks = self.ra_chunks.?;
            const n_evals = self.bytecode_d + 2;

            // Precompute x_vals
            var x_vals: [MAX_RA_EVALS]F = undefined;
            for (0..n_evals) |i| {
                x_vals[i] = F.fromU64(@intCast(i));
            }

            const Ctx = struct {
                ra_chunks: [][]F,
                combined: []F,
                bytecode_d: usize,
                n_evals: usize,
                x_vals: [MAX_RA_EVALS]F,
            };
            const ctx = Ctx{
                .ra_chunks = ra_chunks,
                .combined = combined,
                .bytecode_d = self.bytecode_d,
                .n_evals = n_evals,
                .x_vals = x_vals,
            };

            const mapFn = struct {
                fn f(c: Ctx, start: usize, end: usize) [MAX_RA_EVALS]F {
                    var acc: [MAX_RA_EVALS]F = .{F.zero()} ** MAX_RA_EVALS;
                    for (start..end) |j| {
                        const val0 = c.combined[2 * j];
                        const val1 = c.combined[2 * j + 1];
                        const val_delta = val1.sub(val0);

                        for (0..c.n_evals) |pt_idx| {
                            const x = c.x_vals[pt_idx];
                            var ra_product = F.one();

                            for (0..c.bytecode_d) |i| {
                                const r0 = c.ra_chunks[i][2 * j];
                                const r1 = c.ra_chunks[i][2 * j + 1];
                                ra_product = ra_product.mul(r0.add(x.mul(r1.sub(r0))));
                            }
                            ra_product = ra_product.mul(val0.add(x.mul(val_delta)));

                            acc[pt_idx] = acc[pt_idx].add(ra_product);
                        }
                    }
                    return acc;
                }
            }.f;

            const reduceFn = struct {
                fn f(a: [MAX_RA_EVALS]F, b: [MAX_RA_EVALS]F) [MAX_RA_EVALS]F {
                    var r: [MAX_RA_EVALS]F = undefined;
                    for (0..MAX_RA_EVALS) |i| {
                        r[i] = a[i].add(b[i]);
                    }
                    return r;
                }
            }.f;

            const result = if (self.pool) |pool|
                pool.parallelReduce([MAX_RA_EVALS]F, half, .{F.zero()} ** MAX_RA_EVALS, ctx, mapFn, reduceFn)
            else
                mapFn(ctx, 0, half);

            var evals = try allocator.alloc(F, n_evals);
            for (0..n_evals) |i| {
                evals[i] = result[i];
            }
            return evals;
        }

        pub fn bindChallengePhase2(self: *Self, r: F) void {
            const half = self.current_len / 2;
            const combined = self.combined.?;
            const ra_chunks = self.ra_chunks.?;

            const bindOne = struct {
                fn f(arr: []F, h: usize, challenge: F) void {
                    for (0..h) |j| {
                        arr[j] = arr[2 * j].add(challenge.mul(arr[2 * j + 1].sub(arr[2 * j])));
                    }
                }
            }.f;

            if (self.pool) |pool| {
                // bytecode_d+1 independent arrays: bytecode_d ra_chunks + 1 combined
                const total = self.bytecode_d + 1;
                const Ctx = struct { ra: [][]F, combined: []F, d: usize, half: usize, r: F };
                const ctx = Ctx{ .ra = ra_chunks, .combined = combined, .d = self.bytecode_d, .half = half, .r = r };
                pool.parallelForForce(total, ctx, struct {
                    fn f(c: Ctx, idx: usize) void {
                        if (idx < c.d) {
                            bindOne(c.ra[idx], c.half, c.r);
                        } else {
                            bindOne(c.combined, c.half, c.r);
                        }
                    }
                }.f);
            } else {
                bindOne(combined, half, r);
                for (0..self.bytecode_d) |i| {
                    bindOne(ra_chunks[i], half, r);
                }
            }

            self.current_len = half;
        }

        pub fn getOpeningClaims(self: *const Self, allocator: Allocator) ![]F {
            var claims = try allocator.alloc(F, self.bytecode_d);
            for (0..self.bytecode_d) |i| {
                claims[i] = self.ra_chunks.?[i][0];
            }
            return claims;
        }
    };
}

// =============================================================================
// Stage 6 Batched Sumcheck Prover (Main)
// =============================================================================
pub fn Stage6BatchedProver(comptime F: type) type {
    return struct {
        const Self = @This();

        allocator: Allocator,
        thread_pool: ?*ThreadPool = null,

        pub fn init(allocator: Allocator) Self {
            return .{ .allocator = allocator };
        }

        /// Generate Stage 6 batched sumcheck proof with real polynomial evaluation
        pub fn generateStage6Proof(
            self: *Self,
            proof: *SumcheckInstanceProof(F),
            transcript: *Blake2bTranscript(F),
            opening_claims: *OpeningClaims(F),
            // Parameters
            n_cycle_vars: usize,
            bytecode_log_k: usize,
            log_k_chunk: usize,
            bytecode_d: usize,
            ram_d: usize,
            instruction_d: usize,
            lookups_ra_virtual_log_k_chunk: usize,
            // Execution trace
            trace: *const ExecutionTrace,
            // Opening points for BytecodeReadRaf (all BIG_ENDIAN)
            r_cycle_bc1_spartan_outer: []const F,
            r_cycle_bc2_product_virt: []const F,
            r_cycle_bc3_spartan_shift: []const F,
            r_cycle_bc4_regs_rwc: []const F,
            r_cycle_bc5_regs_val: []const F,
            // Opening points for IncClaimReduction (all BIG_ENDIAN)
            r_cycle_inc_ram_rwc: []const F, // RamReadWriteChecking
            r_cycle_inc_ram_val: []const F, // RamValEvaluation
            // Stage 5 challenges for deriving LookupsRaVirtual and RamRaVirtual points
            stage5_challenges: []const F,
            // RAM r_address from Stage 2 (BIG_ENDIAN) — the aligned address used by RamRaClaimReduction
            ram_r_address_stage2_be: []const F,
            // Memory layout for address remapping
            memory_layout: *const jolt_device.MemoryLayout,
            // Bytecode entry table for Val polynomial computation
            bytecode_entries: []const BytecodeEntry,
            // Register address opening points for Stages 4 and 5 (BIG_ENDIAN)
            r_register_4: []const F, // From RegistersReadWriteChecking (address portion)
            r_register_5: []const F, // From RegistersValEvaluation (address portion)
            // BytecodePCMapper for converting ELF addresses to bytecode array indices
            pc_map: *const BytecodePCMapper,
            // Stage 4 inc_poly copy for diagnostic comparison (pass null slice to skip)
            stage4_inc_poly_copy: []const F,
        ) !Stage6Result(F) {
            // Instance round counts
            const bytecodeReadRaf_rounds = bytecode_log_k + n_cycle_vars;
            const hammingBooleanity_rounds = n_cycle_vars;
            const booleanity_rounds = log_k_chunk + n_cycle_vars;
            const ramRaVirtual_rounds = n_cycle_vars;
            const lookupsRaVirtual_rounds = n_cycle_vars;
            const incClaimReduction_rounds = n_cycle_vars;

            const max_num_rounds = bytecodeReadRaf_rounds;

            // Instance degrees
            const bytecodeReadRaf_degree = bytecode_d + 1;
            const hammingBooleanity_degree: usize = 3;
            const booleanity_degree: usize = 3;
            const ramRaVirtual_degree = ram_d + 1;
            const n_committed_per_virtual = lookups_ra_virtual_log_k_chunk / log_k_chunk;
            const n_virtual_ra_polys = 128 / lookups_ra_virtual_log_k_chunk;
            const lookupsRaVirtual_degree = n_committed_per_virtual + 1;
            const incClaimReduction_degree: usize = 2;

            const max_degree = @max(
                @max(@max(bytecodeReadRaf_degree, hammingBooleanity_degree), @max(booleanity_degree, ramRaVirtual_degree)),
                @max(lookupsRaVirtual_degree, incClaimReduction_degree),
            );

            dbg("[STAGE6] Configuration:\n", .{});
            dbg("  bytecodeReadRaf: {} rounds (addr={}, cycle={}), degree {}\n", .{ bytecodeReadRaf_rounds, bytecode_log_k, n_cycle_vars, bytecodeReadRaf_degree });
            dbg("  hammingBooleanity: {} rounds, degree {}\n", .{ hammingBooleanity_rounds, hammingBooleanity_degree });
            dbg("  booleanity: {} rounds, degree {}\n", .{ booleanity_rounds, booleanity_degree });
            dbg("  ramRaVirtual: {} rounds, degree {}\n", .{ ramRaVirtual_rounds, ramRaVirtual_degree });
            dbg("  lookupsRaVirtual: {} rounds, degree {}\n", .{ lookupsRaVirtual_rounds, lookupsRaVirtual_degree });
            dbg("  incClaimReduction: {} rounds, degree {}\n", .{ incClaimReduction_rounds, incClaimReduction_degree });
            dbg("  max_num_rounds: {}, max_degree: {}\n", .{ max_num_rounds, max_degree });

            // ====================================================================
            // Sample gammas (must match Jolt verifier)
            // ====================================================================

            // Debug: dump transcript state at Stage 6 entry
            if (comptime debug_verbose) {
                dbg("[STAGE6] Transcript state at entry: {{ ", .{});
                for (transcript.state) |b| dbg("{x:0>2} ", .{b});
                dbg("}}, round={}\n", .{transcript.n_rounds});
            }

            dbg("[STAGE6] Transcript at entry: round={}\n", .{transcript.n_rounds});
            const bytecode_raf_gamma_powers = try transcript.challengeScalarPowers(self.allocator, 7);
            defer self.allocator.free(bytecode_raf_gamma_powers);

            // Debug: print first gamma to verify transcript sync
            {
                const g0_be = bytecode_raf_gamma_powers[1].toBytesBE(); // [1] is gamma itself
                dbg("[STAGE6] bytecodeRaf_gamma = [{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                    g0_be[31], g0_be[30], g0_be[29], g0_be[28], g0_be[27], g0_be[26], g0_be[25], g0_be[24],
                });
            }

            const NUM_CIRCUIT_FLAGS: usize = 14;
            const stage1_gammas = try transcript.challengeScalarPowers(self.allocator, 2 + NUM_CIRCUIT_FLAGS);
            defer self.allocator.free(stage1_gammas);

            const stage2_gammas = try transcript.challengeScalarPowers(self.allocator, 4);
            defer self.allocator.free(stage2_gammas);

            const stage3_gammas = try transcript.challengeScalarPowers(self.allocator, 9);
            defer self.allocator.free(stage3_gammas);

            const stage4_gammas = try transcript.challengeScalarPowers(self.allocator, 3);
            defer self.allocator.free(stage4_gammas);

            const NUM_LOOKUP_TABLES: usize = 41;
            const stage5_gammas = try transcript.challengeScalarPowers(self.allocator, 2 + NUM_LOOKUP_TABLES);
            defer self.allocator.free(stage5_gammas);

            dbg("[STAGE6] Sampled BytecodeReadRaf gammas\n", .{});

            // BooleanitySumcheckParams::new() - conditional extra challenges
            // When Stage 5 address variables < log_k_chunk, Jolt samples extra challenges
            // to pad r_address to log_k_chunk length. This happens when LOOKUPS_LOG_K is
            // smaller than log_k_chunk, which doesn't happen in practice (128 > 4).
            if (lookups_ra_virtual_log_k_chunk < log_k_chunk) {
                const extra_count = log_k_chunk - lookups_ra_virtual_log_k_chunk;
                for (0..extra_count) |_| {
                    _ = transcript.challengeScalar();
                }
            }
            // Jolt samples 1 gamma via challenge_scalar_optimized() and derives powers:
            //   gamma_powers_square[i] = γ^(2i) for i = 0..total_d
            // The prover uses gamma_powers[i] = γ^i internally for polynomial scaling,
            // and the verifier uses gamma_powers_square[i] = γ^(2i) for expected_output_claim.
            const total_d = instruction_d + bytecode_d + ram_d;
            const booleanity_gamma = transcript.challengeScalar();
            // Handle degenerate gamma=0 case (same as Jolt: replace with 1)
            const booleanity_gamma_f: F = if (booleanity_gamma.isZero()) F.one() else booleanity_gamma;
            const booleanity_gamma_sq = booleanity_gamma_f.mul(booleanity_gamma_f);
            const booleanity_gammas = try self.allocator.alloc(F, total_d);
            booleanity_gammas[0] = F.one(); // γ^0 = 1
            for (1..total_d) |i| {
                booleanity_gammas[i] = booleanity_gammas[i - 1].mul(booleanity_gamma_sq); // γ^(2i)
            }

            // LookupsRa::new() - gamma powers for virtual RA batching
            const lookups_ra_gamma_powers = try transcript.challengeScalarPowers(self.allocator, n_virtual_ra_polys);
            defer self.allocator.free(lookups_ra_gamma_powers);
            {
                dbg("[STAGE6] lookups_ra_gamma_powers:\n", .{});
                for (0..@min(n_virtual_ra_polys, 4)) |i| {
                    const gp_le = lookups_ra_gamma_powers[i].toBytes();
                    dbg("  gamma_powers[{}]_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                        i, gp_le[0], gp_le[1], gp_le[2], gp_le[3], gp_le[4], gp_le[5], gp_le[6], gp_le[7],
                    });
                }
            }

            // IncClaimReduction::new() - gamma
            // Jolt uses challenge_scalar() (FULL 128-bit) for inc gamma, not optimized
            const inc_gamma = transcript.challengeScalarFull();

            // ====================================================================
            // Compute input claims
            // ====================================================================

            const bcraf_result = self.computeBytecodeReadRafInputClaim(
                opening_claims,
                bytecode_raf_gamma_powers,
                stage1_gammas,
                stage2_gammas,
                stage3_gammas,
                stage4_gammas,
                stage5_gammas,
            );
            const bytecodeReadRaf_input = bcraf_result.total;
            const bcraf_per_stage_claims = bcraf_result.per_stage;

            const hammingBooleanity_input = F.zero();
            const booleanity_input = F.zero();

            const ramRaVirtual_input = opening_claims.get(
                .{ .Virtual = .{ .poly = .RamRa, .sumcheck_id = .RamRaClaimReduction } },
            ) orelse F.zero();

            var lookupsRaVirtual_input = F.zero();
            for (0..n_virtual_ra_polys) |i| {
                const ra_claim = opening_claims.get(
                    .{ .Virtual = .{ .poly = .{ .InstructionRa = i }, .sumcheck_id = .InstructionReadRaf } },
                ) orelse F.zero();
                lookupsRaVirtual_input = lookupsRaVirtual_input.add(lookups_ra_gamma_powers[i].mul(ra_claim));
            }

            const inc_gamma2 = inc_gamma.mul(inc_gamma);
            const inc_gamma3 = inc_gamma2.mul(inc_gamma);

            const v1_claim = opening_claims.get(
                .{ .Committed = .{ .poly = .RamInc, .sumcheck_id = .RamReadWriteChecking } },
            ) orelse F.zero();
            const v2_claim = opening_claims.get(
                .{ .Committed = .{ .poly = .RamInc, .sumcheck_id = .RamValCheck } },
            ) orelse F.zero();
            const w1_claim = opening_claims.get(
                .{ .Committed = .{ .poly = .RdInc, .sumcheck_id = .RegistersReadWriteChecking } },
            ) orelse F.zero();
            const w2_claim = opening_claims.get(
                .{ .Committed = .{ .poly = .RdInc, .sumcheck_id = .RegistersValEvaluation } },
            ) orelse F.zero();

            // Debug: dump inc_gamma and individual claims
            {
                const ig_be = inc_gamma.toBytesBE();
                const v1_be = v1_claim.toBytesBE();
                const v2_be = v2_claim.toBytesBE();
                const w1_be = w1_claim.toBytesBE();
                const w2_be = w2_claim.toBytesBE();
                dbg("[STAGE6] inc_gamma = [{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                    ig_be[31], ig_be[30], ig_be[29], ig_be[28], ig_be[27], ig_be[26], ig_be[25], ig_be[24],
                });
                dbg("[STAGE6] IncClaim v1(RamInc@RamRWC) = [{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                    v1_be[31], v1_be[30], v1_be[29], v1_be[28], v1_be[27], v1_be[26], v1_be[25], v1_be[24],
                });
                dbg("[STAGE6] IncClaim v2(RamInc@RamVal) = [{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                    v2_be[31], v2_be[30], v2_be[29], v2_be[28], v2_be[27], v2_be[26], v2_be[25], v2_be[24],
                });
                dbg("[STAGE6] IncClaim w1(RdInc@RegsRWC) = [{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                    w1_be[31], w1_be[30], w1_be[29], w1_be[28], w1_be[27], w1_be[26], w1_be[25], w1_be[24],
                });
                dbg("[STAGE6] IncClaim w2(RdInc@RegsVal) = [{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                    w2_be[31], w2_be[30], w2_be[29], w2_be[28], w2_be[27], w2_be[26], w2_be[25], w2_be[24],
                });
            }

            const incClaimReduction_input = v1_claim
                .add(inc_gamma.mul(v2_claim))
                .add(inc_gamma2.mul(w1_claim))
                .add(inc_gamma3.mul(w2_claim));

            dbg("[STAGE6] Input claims (LE first 8):\n", .{});
            // Print components for IncClaimReduction
            {
                const v1_be = v1_claim.toBytesBE();
                const v2_be = v2_claim.toBytesBE();
                const w1_be = w1_claim.toBytesBE();
                const w2_be = w2_claim.toBytesBE();
                dbg("  IncClaim components: v1=[{x:0>2},{x:0>2},...] v2=[{x:0>2},{x:0>2},...] w1=[{x:0>2},{x:0>2},...] w2=[{x:0>2},{x:0>2},...]\n", .{
                    v1_be[31], v1_be[30], v2_be[31], v2_be[30], w1_be[31], w1_be[30], w2_be[31], w2_be[30],
                });
            }
            // Print LookupsRa claims
            for (0..@min(n_virtual_ra_polys, 4)) |i| {
                const ra_c = opening_claims.get(
                    .{ .Virtual = .{ .poly = .{ .InstructionRa = i }, .sumcheck_id = .InstructionReadRaf } },
                ) orelse F.zero();
                const ra_be = ra_c.toBytesBE();
                dbg("  InstructionRa[{}] = [{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                    i, ra_be[31], ra_be[30], ra_be[29], ra_be[28], ra_be[27], ra_be[26], ra_be[25], ra_be[24],
                });
            }
            // Print BytecodeReadRaf components
            {
                const bc_be = bytecodeReadRaf_input.toBytesBE();
                dbg("  bytecodeReadRaf_input = [{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                    bc_be[31], bc_be[30], bc_be[29], bc_be[28], bc_be[27], bc_be[26], bc_be[25], bc_be[24],
                });
            }
            {
                const ram_be = ramRaVirtual_input.toBytesBE();
                dbg("  ramRaVirtual_input = [{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                    ram_be[31], ram_be[30], ram_be[29], ram_be[28], ram_be[27], ram_be[26], ram_be[25], ram_be[24],
                });
            }
            {
                const look_be = lookupsRaVirtual_input.toBytesBE();
                dbg("  lookupsRaVirtual_input = [{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                    look_be[31], look_be[30], look_be[29], look_be[28], look_be[27], look_be[26], look_be[25], look_be[24],
                });
            }
            {
                const inc_be = incClaimReduction_input.toBytesBE();
                dbg("  incClaimReduction_input = [{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                    inc_be[31], inc_be[30], inc_be[29], inc_be[28], inc_be[27], inc_be[26], inc_be[25], inc_be[24],
                });
            }

            // ====================================================================
            // Derive opening points for RamRaVirtual and LookupsRaVirtual from Stage 5
            // ====================================================================

            const LOOKUPS_LOG_K: usize = 128;
            const ram_log_k: usize = ram_r_address_stage2_be.len;

            // RamRaVirtual: r_cycle from Stage 5 RamRaClaimReduction, r_address from Stage 2
            // RamRaClaimReduction is cycle-only (log_T rounds), NOT address+cycle.
            // The r_address comes from Stage 2's aligned RAM address, stored in ram_r_address_stage2_be.
            const stage5_max_rounds = LOOKUPS_LOG_K + n_cycle_vars;
            // RamRaClaimReduction has n_cycle_vars rounds (cycle-only), offset = stage5_max - n_cycle_vars
            const ram_ra_offset = stage5_max_rounds - n_cycle_vars;
            dbg("[STAGE6] RamRa challenge offset: stage5_max={}, ram_ra_rounds={}, offset={}\n", .{
                stage5_max_rounds, n_cycle_vars, ram_ra_offset,
            });
            var ram_ra_r_cycle = try self.allocator.alloc(F, n_cycle_vars);
            defer self.allocator.free(ram_ra_r_cycle);
            for (0..n_cycle_vars) |i| {
                // Reverse cycle part: challenges[offset..offset+n_cycle_vars] reversed (BE)
                ram_ra_r_cycle[i] = stage5_challenges[ram_ra_offset + n_cycle_vars - 1 - i];
            }

            // r_address for RamRa: from Stage 2 aligned RAM address (already BIG_ENDIAN)
            // Pad with leading zeros to make length a multiple of log_k_chunk (matching Jolt's compute_r_address_chunks)
            const padded_ram_len = ((ram_log_k + log_k_chunk - 1) / log_k_chunk) * log_k_chunk;
            var ram_ra_r_address_be: []F = undefined;
            var ram_ra_r_address_allocated = false;
            if (padded_ram_len != ram_log_k) {
                ram_ra_r_address_be = try self.allocator.alloc(F, padded_ram_len);
                ram_ra_r_address_allocated = true;
                const pad_count = padded_ram_len - ram_log_k;
                @memset(ram_ra_r_address_be[0..pad_count], F.zero());
                @memcpy(ram_ra_r_address_be[pad_count..], ram_r_address_stage2_be);
            } else {
                ram_ra_r_address_be = @constCast(ram_r_address_stage2_be);
            }
            defer if (ram_ra_r_address_allocated) self.allocator.free(ram_ra_r_address_be);

            // Split r_address into chunks (BIG_ENDIAN, chunk[0] = MSB)
            var ram_ra_addr_chunks = try self.allocator.alloc([]const F, ram_d);
            defer self.allocator.free(ram_ra_addr_chunks);
            for (0..ram_d) |i| {
                const chunk_start = i * log_k_chunk;
                const chunk_end = chunk_start + log_k_chunk;
                ram_ra_addr_chunks[i] = ram_ra_r_address_be[chunk_start..chunk_end];
            }

            // LookupsRaVirtual: r_cycle and r_addr_chunks from InstructionReadRaf (Stage 5 Instance 1)
            // InstructionReadRaf has LOOKUPS_LOG_K + n_cycle_vars = 136 rounds
            // normalize_opening_point: address NOT reversed, cycle IS reversed
            var lookups_ra_r_cycle = try self.allocator.alloc(F, n_cycle_vars);
            defer self.allocator.free(lookups_ra_r_cycle);
            for (0..n_cycle_vars) |i| {
                // Reverse cycle part: challenges[128..136] reversed
                lookups_ra_r_cycle[i] = stage5_challenges[LOOKUPS_LOG_K + n_cycle_vars - 1 - i];
            }
            // Debug: print lookups_ra_r_cycle to compare with Jolt's r_cycle
            for (0..n_cycle_vars) |dbg_i| {
                const dbg_b = lookups_ra_r_cycle[dbg_i].toBytesBE();
                dbg("[S6_RCYCLE] lookups_ra_r_cycle[{}] LE=[", .{dbg_i});
                for (0..8) |bi| dbg("{x:0>2}", .{dbg_b[31 - bi]});
                dbg("]\n", .{});
            }

            // r_address for Lookups: challenges[0..128] NOT reversed (stays LITTLE_ENDIAN)
            // Then compute_r_address_chunks splits into log_k_chunk-sized pieces
            var lookups_ra_addr_chunks = try self.allocator.alloc([]const F, instruction_d);
            defer self.allocator.free(lookups_ra_addr_chunks);
            for (0..instruction_d) |i| {
                const chunk_start = i * log_k_chunk;
                const chunk_end = @min(chunk_start + log_k_chunk, LOOKUPS_LOG_K);
                lookups_ra_addr_chunks[i] = stage5_challenges[chunk_start..chunk_end];
            }

            // ====================================================================
            // Initialize ALL sumcheck instances
            // ====================================================================

            // Instance 5: IncClaimReduction (degree 2)
            // IncClaimReduction uses RAM r_cycles (not BytecodeReadRaf r_cycles)
            var inc_prover = try IncClaimReductionProver(F).init(
                self.allocator, trace, inc_gamma,
                r_cycle_inc_ram_rwc, r_cycle_inc_ram_val,
                r_cycle_bc4_regs_rwc, r_cycle_bc5_regs_val,
                self.thread_pool,
            );
            defer inc_prover.deinit();

            // Direct comparison: Stage 6 rd_inc vs Stage 4 inc_poly
            if (comptime debug_verbose) if (stage4_inc_poly_copy.len > 0) {
                var inc_diff_count: usize = 0;
                const cmp_len = @min(inc_prover.rd_inc.len, stage4_inc_poly_copy.len);
                for (0..cmp_len) |j| {
                    if (!inc_prover.rd_inc[j].eql(stage4_inc_poly_copy[j])) {
                        if (inc_diff_count < 8) {
                            const a = inc_prover.rd_inc[j].toBytes();
                            const b = stage4_inc_poly_copy[j].toBytes();
                            const step_j = trace.steps.items[j];
                            std.debug.print("[S6 vs S4 INC] j={} rd={} noop={} wr={} s6_LE={x:0>16} s4_LE={x:0>16}\n", .{
                                j, step_j.rd_index,
                                @as(u8, if (step_j.is_noop) 1 else 0),
                                @as(u8, if (step_j.rd_written) 1 else 0),
                                @as(u64, @bitCast(a[0..8].*)),
                                @as(u64, @bitCast(b[0..8].*)),
                            });
                        }
                        inc_diff_count += 1;
                    }
                }
                std.debug.print("[S6 vs S4 INC] total differences: {}\n", .{inc_diff_count});
            };

            // Diagnostic: verify IncClaimReduction individual component sums
            if (comptime debug_verbose) {
                const T_inc = inc_prover.current_len;
                // Recompute individual eq tables for diagnosis
                var rev_buf2 = try self.allocator.alloc(F, n_cycle_vars);
                defer self.allocator.free(rev_buf2);

                for (0..n_cycle_vars) |i| rev_buf2[i] = r_cycle_inc_ram_rwc[n_cycle_vars - 1 - i];
                const eq_r2_diag = try computeEqTable(F, self.allocator, rev_buf2, n_cycle_vars);
                defer self.allocator.free(eq_r2_diag);

                for (0..n_cycle_vars) |i| rev_buf2[i] = r_cycle_inc_ram_val[n_cycle_vars - 1 - i];
                const eq_r4_diag = try computeEqTable(F, self.allocator, rev_buf2, n_cycle_vars);
                defer self.allocator.free(eq_r4_diag);

                for (0..n_cycle_vars) |i| rev_buf2[i] = r_cycle_bc4_regs_rwc[n_cycle_vars - 1 - i];
                const eq_s4_diag = try computeEqTable(F, self.allocator, rev_buf2, n_cycle_vars);
                defer self.allocator.free(eq_s4_diag);

                for (0..n_cycle_vars) |i| rev_buf2[i] = r_cycle_bc5_regs_val[n_cycle_vars - 1 - i];
                const eq_s5_diag = try computeEqTable(F, self.allocator, rev_buf2, n_cycle_vars);
                defer self.allocator.free(eq_s5_diag);

                var sv1 = F.zero();
                var sv2 = F.zero();
                var sw1 = F.zero();
                var sw2 = F.zero();
                for (0..T_inc) |j| {
                    sv1 = sv1.add(inc_prover.ram_inc[j].mul(eq_r2_diag[j]));
                    sv2 = sv2.add(inc_prover.ram_inc[j].mul(eq_r4_diag[j]));
                    sw1 = sw1.add(inc_prover.rd_inc[j].mul(eq_s4_diag[j]));
                    sw2 = sw2.add(inc_prover.rd_inc[j].mul(eq_s5_diag[j]));
                }
                const v1_ok: u8 = if (std.mem.eql(u8, &sv1.toBytesBE(), &v1_claim.toBytesBE())) 1 else 0;
                const v2_ok: u8 = if (std.mem.eql(u8, &sv2.toBytesBE(), &v2_claim.toBytesBE())) 1 else 0;
                const w1_ok: u8 = if (std.mem.eql(u8, &sw1.toBytesBE(), &w1_claim.toBytesBE())) 1 else 0;
                const w2_ok: u8 = if (std.mem.eql(u8, &sw2.toBytesBE(), &w2_claim.toBytesBE())) 1 else 0;
                std.debug.print("[INC_DIAG] v1_match={} v2_match={} w1_match={} w2_match={}\n", .{ v1_ok, v2_ok, w1_ok, w2_ok });
                if (v1_ok == 0) {
                    const a = sv1.toBytesBE();
                    const b = v1_claim.toBytesBE();
                    std.debug.print("[INC_DIAG] v1: sum_LE={x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2} claim_LE={x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}\n", .{
                        a[31], a[30], a[29], a[28], a[27], a[26], a[25], a[24],
                        b[31], b[30], b[29], b[28], b[27], b[26], b[25], b[24],
                    });
                }
                if (v2_ok == 0) {
                    const a = sv2.toBytesBE();
                    const b = v2_claim.toBytesBE();
                    std.debug.print("[INC_DIAG] v2: sum_LE={x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2} claim_LE={x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}\n", .{
                        a[31], a[30], a[29], a[28], a[27], a[26], a[25], a[24],
                        b[31], b[30], b[29], b[28], b[27], b[26], b[25], b[24],
                    });
                }
                if (w1_ok == 0) {
                    const a = sw1.toBytesBE();
                    const b = w1_claim.toBytesBE();
                    std.debug.print("[INC_DIAG] w1: sum_LE={x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2} claim_LE={x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}\n", .{
                        a[31], a[30], a[29], a[28], a[27], a[26], a[25], a[24],
                        b[31], b[30], b[29], b[28], b[27], b[26], b[25], b[24],
                    });
                }
                if (w2_ok == 0) {
                    const a = sw2.toBytesBE();
                    const b = w2_claim.toBytesBE();
                    std.debug.print("[INC_DIAG] w2: sum_LE={x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2} claim_LE={x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}\n", .{
                        a[31], a[30], a[29], a[28], a[27], a[26], a[25], a[24],
                        b[31], b[30], b[29], b[28], b[27], b[26], b[25], b[24],
                    });
                }
            }

            // Instance 1: HammingBooleanity (degree 3)
            var hamming_prover = try HammingBooleanityProver(F).init(
                self.allocator, trace, r_cycle_bc1_spartan_outer,
                self.thread_pool,
            );
            defer hamming_prover.deinit();

            // Instance 3: RamRaVirtual (degree ram_d+1)
            var ram_ra_prover = try RamRaVirtualProver(F).init(
                self.allocator, trace, ram_ra_r_cycle,
                ram_ra_addr_chunks, ram_d, memory_layout, log_k_chunk,
                self.thread_pool,
            );
            defer ram_ra_prover.deinit();

            // Instance 4: LookupsRaVirtual (degree n_committed_per_virtual+1)
            var lookups_ra_prover = try LookupsRaVirtualProver(F).init(
                self.allocator, trace, lookups_ra_r_cycle,
                lookups_ra_addr_chunks, lookups_ra_gamma_powers,
                n_committed_per_virtual, n_virtual_ra_polys,
                log_k_chunk, instruction_d,
                self.thread_pool,
            );
            defer lookups_ra_prover.deinit();

            // Instance 2: Booleanity (degree 3, two-phase)
            // Build BooleanityProver with G tables and eq tables from Stage 5 opening point.
            //
            // In Jolt, r_address for booleanity = last log_k_chunk elements of Stage 5
            // InstructionReadRaf address (reversed to LE). Stage 5 address uses HighToLow
            // binding, so stage5_challenges[0]=MSB. After reverse to LE: [ch[127],...,ch[0]].
            // Last log_k_chunk elements = [ch[3],ch[2],ch[1],ch[0]] = MSB bits in LE.
            //
            // r_cycle for booleanity = same as InstructionReadRaf cycle (LE) = lookups_ra_r_cycle
            //
            // Binding order: LowToHigh for both Phase 1 (address) and Phase 2 (cycle)
            var booleanity_prover = blk_bool: {
                const total_bool_polys = instruction_d + bytecode_d + ram_d;

                // r_address_bool: last log_k_chunk of Stage 5 address in LE
                // Stage 5 address in BE: stage5_challenges[0..128] (MSB first since HighToLow binding)
                // Reverse to LE: [ch[127], ch[126], ..., ch[0]]
                // Last log_k_chunk: [ch[log_k_chunk-1], ..., ch[0]] = MSB bits in LE
                var r_address_bool_le = try self.allocator.alloc(F, log_k_chunk);
                // No defer free - BooleanityProver takes ownership of r_address_bool_le
                for (0..log_k_chunk) |i| {
                    // In LE, element i corresponds to Stage5 address challenge (LOOKUPS_LOG_K - 1 - (LOOKUPS_LOG_K - log_k_chunk + i))
                    // = log_k_chunk - 1 - i
                    r_address_bool_le[i] = stage5_challenges[log_k_chunk - 1 - i];
                }

                // r_cycle_bool_le: same as lookups_ra_r_cycle (already LE)
                // lookups_ra_r_cycle[i] = stage5_challenges[LOOKUPS_LOG_K + n_cycle_vars - 1 - i]

                // Build eq_addr table for Phase 1 (LowToHigh binding)
                // computeEqTable expects BE input (MSB-first) for its internal convention.
                // Since r_address_bool_le is LE and we want LowToHigh binding,
                // the eq table should be indexed such that eq_addr[k] = eq(r_addr_le, k)
                // where bit 0 of k is the LSB, bound first.
                // Jolt's LowToHigh EqPolynomial: eq(r, k) = Π_i (r[i]*k_i + (1-r[i])*(1-k_i))
                // where r[0] corresponds to the LSB of k.
                // For computeEqTable: it expects r in "BE" (MSB first), so reverse LE to BE.
                var r_addr_bool_be_for_eq = try self.allocator.alloc(F, log_k_chunk);
                defer self.allocator.free(r_addr_bool_be_for_eq);
                for (0..log_k_chunk) |i| {
                    r_addr_bool_be_for_eq[i] = r_address_bool_le[log_k_chunk - 1 - i];
                }
                const eq_addr_bool_phase1 = try computeEqTable(F, self.allocator, r_addr_bool_be_for_eq, log_k_chunk);
                defer self.allocator.free(eq_addr_bool_phase1); // Only used for debug verification below

                // Build a SINGLE eq_cycle table used for BOTH G construction AND Phase 2 halving.
                //
                // The table ordering must match Jolt's evals_parallel which iterates .rev():
                //   bit 0 of index j → r_cycle[n-1] (MSB)
                // For our computeEqTable (forward iteration), input[0] must be MSB = lookups[0].
                // So input = lookups_ra_r_cycle directly (BE, MSB first).
                //
                // Using the SAME table for G construction and Phase 2 ensures consistency:
                // Phase 1 reduces address variables with G tables weighted by eq_cycle[j],
                // and Phase 2 halves the same eq_cycle[j] table. The running claim from Phase 1
                // equals the initial Phase 2 polynomial sum, satisfying the transition.
                //
                // After Phase 2 halving with LowToHigh binding, the final eq value equals
                // eq(challenges, r_cycle_BE) = eq(challenges, rev(r_cycle_LE)), matching
                // Jolt's verifier which computes combined_r_cycle = rev(r_cycle_LE).
                const eq_cycle_bool_phase2 = try computeEqTableParallel(F, self.allocator, lookups_ra_r_cycle, n_cycle_vars, self.thread_pool);
                // eq_cycle_bool_phase2 is NOT deferred - shared with BooleanityProver

                // Build G tables: G_i[k] = Σ_j eq(r_cycle_fixed, j) * [chunk_i(j) == k]
                const T_val: usize = @as(usize, 1) << @intCast(n_cycle_vars);
                const K_val: usize = @as(usize, 1) << @intCast(log_k_chunk);
                var G_tables = try self.allocator.alloc([]F, total_bool_polys);
                for (0..total_bool_polys) |i| {
                    G_tables[i] = try self.allocator.alloc(F, K_val);
                    @memset(G_tables[i], F.zero());
                }

                for (0..T_val) |j| {
                    const eq_j = eq_cycle_bool_phase2[j]; // eq(r_cycle_fixed, j) - same table as Phase 2
                    if (eq_j.eql(F.zero())) continue;

                    const step = trace.steps.items[j];

                    // InstructionRa chunks - use centralized computeLookupIndex
                    // to ensure consistency with transitionToPhase2, LookupsRaVirtual,
                    // and Stage 7 G table builder.
                    {
                        const lookup_idx = computeLookupIndex(step);
                        for (0..instruction_d) |i| {
                            const shift = log_k_chunk * (instruction_d - 1 - i);
                            const mask: u128 = (@as(u128, 1) << @intCast(log_k_chunk)) - 1;
                            const chunk_val: usize = @intCast((lookup_idx >> @intCast(shift)) & mask);
                            if (chunk_val < K_val) {
                                G_tables[i][chunk_val] = G_tables[i][chunk_val].add(eq_j);
                            }
                        }
                    }

                    // BytecodeRa chunks
                    {
                        const pc_idx: u64 = @intCast(pc_map.getPCForStep(step));
                        for (0..bytecode_d) |i| {
                            const chunk_val = extractChunkMSB(pc_idx, i, bytecode_d, log_k_chunk);
                            if (chunk_val < K_val) {
                                G_tables[instruction_d + i][chunk_val] = G_tables[instruction_d + i][chunk_val].add(eq_j);
                            }
                        }
                    }

                    // RamRa chunks
                    {
                        if (step.memory_addr) |addr| {
                            if (addr != 0) {
                                if (memory_layout.remapAddress(addr)) |raddr| {
                                    for (0..ram_d) |i| {
                                        const chunk_val = extractChunkMSB(raddr, i, ram_d, log_k_chunk);
                                        if (chunk_val < K_val) {
                                            G_tables[instruction_d + bytecode_d + i][chunk_val] = G_tables[instruction_d + bytecode_d + i][chunk_val].add(eq_j);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                // Use the independently sampled gammas directly (matching Jolt's challenge_vector_optimized)
                // Jolt formula: Σ_i γ_i * (ra_i² - ra_i), where γ_i are independent challenges
                // booleanity_gammas ownership transfers to BooleanityProver (freed by deinit)
                const gamma_sq = booleanity_gammas;

                // Verify G tables: Σ_k G_i[k] should equal Σ_j eq(r_cycle, j) = 1
                // Actually Σ_k G_i[k] = Σ_j eq(r_cycle, j) * Σ_k [chunk_i(j)==k]
                //                     = Σ_j eq(r_cycle, j) * 1 = 1 (since chunk_i(j) always hits exactly one k)
                // Wait no: only if all j have valid chunks. Noop steps may have chunk_val=0 added.
                // Let's just print the first few G tables for debug.
                dbg("[BOOL_PROVER] G tables built: N={}, K={}, T={}\n", .{ total_bool_polys, K_val, T_val });
                for (0..@min(3, total_bool_polys)) |i| {
                    var g_sum = F.zero();
                    for (0..K_val) |k| g_sum = g_sum.add(G_tables[i][k]);
                    const gs_be = g_sum.toBytesBE();
                    dbg("  G[{}] sum_LE=[{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}]\n", .{
                        i, gs_be[31], gs_be[30], gs_be[29], gs_be[28], gs_be[27], gs_be[26], gs_be[25], gs_be[24],
                    });
                }

                // Initial claim verification: Σ_k eq_addr[k] * Σ_i γ^{2i} * (G_i[k]^2 - G_i[k])
                // This should be zero since ra_i(k,j) is binary.
                // Actually that's the FULL sum; at random r it won't be zero for individual terms.
                // But the initial claim IS zero.
                {
                    var init_sum = F.zero();
                    for (0..K_val) |k| {
                        var q_val = F.zero();
                        for (0..total_bool_polys) |i| {
                            const g_k = G_tables[i][k];
                            q_val = q_val.add(gamma_sq[i].mul(g_k.mul(g_k).sub(g_k)));
                        }
                        init_sum = init_sum.add(eq_addr_bool_phase1[k].mul(q_val));
                    }
                    const is_be = init_sum.toBytesBE();
                    dbg("[BOOL_PROVER] Initial sum (should be ~0) LE=[{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}]\n", .{
                        is_be[31], is_be[30], is_be[29], is_be[28], is_be[27], is_be[26], is_be[25], is_be[24],
                    });
                }

                break :blk_bool try BooleanityProver(F).init(
                    self.allocator,
                    G_tables,
                    r_address_bool_le,
                    eq_cycle_bool_phase2,
                    gamma_sq,
                    total_bool_polys,
                    log_k_chunk,
                    n_cycle_vars,
                    trace,
                    instruction_d,
                    bytecode_d,
                    ram_d,
                    memory_layout,
                    pc_map,
                );
            };
            booleanity_prover.pool = self.thread_pool;
            defer booleanity_prover.deinit();

            // Instance 0: BytecodeReadRaf (degree bytecode_d+1)
            // Compute Val polynomials from bytecode entries and stage gammas
            const bytecode_K: usize = @as(usize, 1) << @intCast(bytecode_log_k);
            var bytecode_val_polys: [5][]F = undefined;

            // Precompute eq tables for Stages 4 and 5 register addresses
            // r_register_4 and r_register_5 are the address portions from
            // RegistersReadWriteChecking and RegistersValEvaluation opening points
            const REGISTER_COUNT_LOG2: usize = 7; // log2(128 registers: 32 RISC-V + 96 virtual)
            dbg("[STAGE6] r_register_4 (len={}):\n", .{r_register_4.len});
            for (r_register_4, 0..) |rv, i| {
                dbg("  r_register_4[{}] mont_limbs=[0x{x:0>16}, 0x{x:0>16}, 0x{x:0>16}, 0x{x:0>16}]\n", .{i, rv.limbs[0], rv.limbs[1], rv.limbs[2], rv.limbs[3]});
            }
            dbg("[STAGE6] r_register_5 (len={}):\n", .{r_register_5.len});
            for (r_register_5, 0..) |rv, i| {
                dbg("  r_register_5[{}] mont_limbs=[0x{x:0>16}, 0x{x:0>16}, 0x{x:0>16}, 0x{x:0>16}]\n", .{i, rv.limbs[0], rv.limbs[1], rv.limbs[2], rv.limbs[3]});
            }
            // Jolt's EqPolynomial::evals uses BIG-ENDIAN bit indexing:
            // r[0] maps to MSB of index, r[n-1] maps to LSB.
            // Our computeEqTable uses LITTLE-ENDIAN: r[0] maps to LSB.
            // Fix: reverse the input array so our LE computation produces BE-indexed results.
            var r_register_4_rev = try self.allocator.alloc(F, r_register_4.len);
            defer self.allocator.free(r_register_4_rev);
            for (0..r_register_4.len) |i| {
                r_register_4_rev[i] = r_register_4[r_register_4.len - 1 - i];
            }
            var r_register_5_rev = try self.allocator.alloc(F, r_register_5.len);
            defer self.allocator.free(r_register_5_rev);
            for (0..r_register_5.len) |i| {
                r_register_5_rev[i] = r_register_5[r_register_5.len - 1 - i];
            }
            const eq_table_4 = try computeEqTable(F, self.allocator, r_register_4_rev, REGISTER_COUNT_LOG2);
            defer self.allocator.free(eq_table_4);
            const eq_table_5 = try computeEqTable(F, self.allocator, r_register_5_rev, REGISTER_COUNT_LOG2);
            defer self.allocator.free(eq_table_5);
            // Print eq_table_4 entries in LE hex for comparison with Jolt
            dbg("[STAGE6] eq_table_4 (len={}):\n", .{eq_table_4.len});
            for ([_]usize{0, 1, 2, 8, 10, 15, 31, 127}) |idx| {
                if (idx < eq_table_4.len) {
                    const vbe = eq_table_4[idx].toBytesBE();
                    dbg("  eq4[{}]_LE=[", .{idx});
                    for (0..32) |bi| dbg("{x:0>2}", .{vbe[31 - bi]});
                    dbg("]\n", .{});
                }
            }
            // Print stage4_gammas in LE hex
            dbg("[STAGE6] stage4_gammas:\n", .{});
            for (0..3) |i| {
                const gbe = stage4_gammas[i].toBytesBE();
                dbg("  gamma4[{}]_LE=[", .{i});
                for (0..32) |bi| dbg("{x:0>2}", .{gbe[31 - bi]});
                dbg("]\n", .{});
            }

            for (0..5) |s| {
                bytecode_val_polys[s] = try self.allocator.alloc(F, bytecode_K);
                @memset(bytecode_val_polys[s], F.zero());
            }

            for (0..bytecode_K) |k| {
                if (k >= bytecode_entries.len) break;
                const entry = bytecode_entries[k];

                // Stage 1: unexpanded_pc + γ₁¹·imm + Σ γ₁^(2+i)·circuit_flag_i
                // CRITICAL: The Imm encoding must match Jolt's vanilla verifier exactly.
                // Jolt's NormalizedOperands.imm is i128, but how it gets there depends
                // on the instruction FORMAT type:
                //   FormatI (I-type): u64 as i128 → zero-extended (always positive)
                //   FormatU (U-type): u64 as i128 → zero-extended (always positive)
                //   FormatJ (J-type): u64 as i128 → zero-extended (always positive)
                //   FormatB (B-type): i128 directly → signed
                //   FormatS (S-type): i64 as i128 → sign-extended (signed)
                //   Virtual (0x0B, 0x2B): u64 as i128 (from emit_i helper)
                // Then Jolt calls from_i128(operands.imm) to get the field element.
                const imm_field: F = blk: {
                    const opcode_for_imm = entry.opcode;
                    // Jolt stores imm as i128 in NormalizedOperands, then uses from_i128().
                    // The i128 value depends on the instruction format's source type:
                    //   FormatI (u64): u64 as i128 → zero-extended (always positive)
                    //   FormatU (u64): u64 as i128 → zero-extended (always positive)
                    //   FormatJ (u64): u64 as i128 → zero-extended (always positive)
                    //   FormatB (i128): direct → can be negative
                    //   FormatS (i64): i64 as i128 → sign-extended (can be negative)
                    //   FormatLoad (i64): i64 as i128 → sign-extended (can be negative)
                    // We must match: signed formats use fieldFromI128, unsigned use fromU64.
                    const is_signed_format = (opcode_for_imm == 0x63) or // B-type (branches)
                        (opcode_for_imm == 0x23) or // S-type (stores)
                        (opcode_for_imm == 0x03); // Load (FormatLoad uses i64, sign-extends to i128)
                    if (is_signed_format) {
                        break :blk fieldFromI128(F, @as(i128, entry.imm));
                    } else {
                        // I-type, U-type, J-type, Virtual: u64 zero-extended to i128.
                        // from_i128(u64 as i128) = from_u64(u64), so fromU64(@bitCast) matches.
                        break :blk F.fromU64(@as(u64, @bitCast(entry.imm)));
                    }
                };
                var val1 = F.fromU64(entry.address); // No gamma[0] - Jolt formula: unexpanded_pc + γ¹·imm + Σγ^(2+i)·cf[i]
                val1 = val1.add(stage1_gammas[1].mul(imm_field));
                for (0..14) |i| {
                    if (entry.circuit_flags[i]) {
                        val1 = val1.add(stage1_gammas[2 + i]);
                    }
                }
                bytecode_val_polys[0][k] = val1;

                // Debug: print details for mismatching entries
                if (k == 3 or k == 4 or k == 10 or k == 16 or k == 18 or k == 27 or k == 29 or k == 35) {
                    const addr_be = F.fromU64(entry.address).toBytesBE();
                    const imm_be = imm_field.toBytesBE();
                    dbg("[ZOLT_BC_ENTRY] k={}: addr=0x{x:0>8} imm_LE=[", .{k, entry.address});
                    for (0..8) |bi| dbg("{x:0>2}", .{imm_be[31 - bi]});
                    dbg("] opcode=0x{x:0>2} raw_imm={} cf=[", .{entry.opcode, entry.imm});
                    for (0..14) |ci| {
                        if (entry.circuit_flags[ci]) dbg("1", .{}) else dbg("0", .{});
                    }
                    dbg("]\n", .{});
                    _ = addr_be;
                }

                // Stage 2: γ₂⁰·jump + γ₂¹·branch + γ₂²·write_lookup_to_rd + γ₂³·virtual_instruction
                // Matches upstream a16z/jolt (no IsRdNotZero — that was fork-only)
                var val2 = F.zero();
                if (entry.circuit_flags[@intFromEnum(CircuitFlags.Jump)]) {
                    val2 = val2.add(stage2_gammas[0]);
                }
                if (entry.instruction_flags[@intFromEnum(InstructionFlags.Branch)]) {
                    val2 = val2.add(stage2_gammas[1]);
                }
                if (entry.circuit_flags[@intFromEnum(CircuitFlags.WriteLookupOutputToRD)]) {
                    val2 = val2.add(stage2_gammas[2]);
                }
                if (entry.circuit_flags[@intFromEnum(CircuitFlags.VirtualInstruction)]) {
                    val2 = val2.add(stage2_gammas[3]);
                }
                bytecode_val_polys[1][k] = val2;

                // Stage 3: γ₃⁰·imm + γ₃¹·unexpanded_pc + γ₃²·L_is_rs1 + γ₃³·L_is_pc
                //         + γ₃⁴·R_is_rs2 + γ₃⁵·R_is_imm + γ₃⁶·is_noop
                //         + γ₃⁷·virtual_instruction + γ₃⁸·is_first_in_sequence
                // Uses same signed Imm encoding as Stage 1 (see comment above)
                var val3 = imm_field; // No gamma[0] - Jolt formula: imm + γ¹·unexpanded_pc + Σγ^(2+i)·flags[i]
                val3 = val3.add(stage3_gammas[1].mul(F.fromU64(entry.address)));
                if (entry.instruction_flags[@intFromEnum(InstructionFlags.LeftOperandIsRs1Value)]) {
                    val3 = val3.add(stage3_gammas[2]);
                }
                if (entry.instruction_flags[@intFromEnum(InstructionFlags.LeftOperandIsPC)]) {
                    val3 = val3.add(stage3_gammas[3]);
                }
                if (entry.instruction_flags[@intFromEnum(InstructionFlags.RightOperandIsRs2Value)]) {
                    val3 = val3.add(stage3_gammas[4]);
                }
                if (entry.instruction_flags[@intFromEnum(InstructionFlags.RightOperandIsImm)]) {
                    val3 = val3.add(stage3_gammas[5]);
                }
                if (entry.instruction_flags[@intFromEnum(InstructionFlags.IsNoop)]) {
                    val3 = val3.add(stage3_gammas[6]);
                }
                if (entry.circuit_flags[@intFromEnum(CircuitFlags.VirtualInstruction)]) {
                    val3 = val3.add(stage3_gammas[7]);
                }
                if (entry.is_first_in_sequence) {
                    val3 = val3.add(stage3_gammas[8]);
                }
                bytecode_val_polys[2][k] = val3;

                // Stage 4: γ₄⁰·eq(rd, r_reg4) + γ₄¹·eq(rs1, r_reg4) + γ₄²·eq(rs2, r_reg4)
                const REGISTER_COUNT: usize = 128; // 32 RISC-V + 96 virtual
                var val4 = F.zero();
                if (entry.rd < REGISTER_COUNT) {
                    val4 = val4.add(stage4_gammas[0].mul(eq_table_4[entry.rd]));
                }
                if (entry.rs1 < REGISTER_COUNT) {
                    val4 = val4.add(stage4_gammas[1].mul(eq_table_4[entry.rs1]));
                }
                if (entry.rs2 < REGISTER_COUNT) {
                    val4 = val4.add(stage4_gammas[2].mul(eq_table_4[entry.rs2]));
                }
                bytecode_val_polys[3][k] = val4;

                // Stage 5: eq(rd, r_reg5) + γ₅¹·!is_interleaved + Σ γ₅^(2+i)·table_flag_i
                var val5 = F.zero();
                if (entry.rd < REGISTER_COUNT) {
                    val5 = val5.add(eq_table_5[entry.rd]);
                }
                if (!entry.is_interleaved) {
                    val5 = val5.add(stage5_gammas[1]);
                }
                if (entry.lookup_table_index < 41) {
                    val5 = val5.add(stage5_gammas[2 + @as(usize, entry.lookup_table_index)]);
                }
                bytecode_val_polys[4][k] = val5;
            }

            // Debug: Print Stage 3 Val poly for comparison with Jolt verifier
            if (comptime debug_verbose) {
                dbg("[STAGE6] Val[3] (Stage 4/RegistersRWC) entries:\n", .{});
                for (0..bytecode_K) |k| {
                    const vbe = bytecode_val_polys[3][k].toBytesBE();
                    dbg("  Val[3][{}]_LE=[", .{k});
                    for (0..32) |bi| dbg("{x:0>2}", .{vbe[31 - bi]});
                    dbg("]\n", .{});
                }
            }
            if (debug_verbose) {
                for ([_]usize{0, 1, 2, 4}) |s| {
                    for (0..bytecode_K) |k| {
                        const vbe = bytecode_val_polys[s][k].toBytesBE();
                        dbg("  Val[{}][{}]_LE=[", .{s, k});
                        for (0..32) |bi| dbg("{x:0>2}", .{vbe[31 - bi]});
                        dbg("]\n", .{});
                    }
                }
            }

            // Debug: Dump bytecode entries
            if (comptime debug_verbose) {
                dbg("[STAGE6] Bytecode entries (ALL k=0..{}):\n", .{bytecode_K});
                for (0..@min(bytecode_K, 64)) |k| {
                    if (k >= bytecode_entries.len) break;
                    const entry = bytecode_entries[k];
                    dbg("[STAGE6] entry[{}]: addr=0x{x:0>8} rd={} rs1={} rs2={} imm={} cf=[", .{ k, entry.address, entry.rd, entry.rs1, entry.rs2, entry.imm });
                    for (0..14) |i| {
                        if (i > 0) dbg(",", .{});
                        if (entry.circuit_flags[i]) dbg("1", .{}) else dbg("0", .{});
                    }
                    dbg("] if=[", .{});
                    for (0..7) |i| {
                        if (i > 0) dbg(",", .{});
                        if (entry.instruction_flags[i]) dbg("1", .{}) else dbg("0", .{});
                    }
                    dbg("] lt={} interleaved={}\n", .{ entry.lookup_table_index, @intFromBool(entry.is_interleaved) });
                }
            }

            // Build identity polynomial
            var bytecode_int_poly = try self.allocator.alloc(F, bytecode_K);
            for (0..bytecode_K) |k| {
                bytecode_int_poly[k] = F.fromU64(@intCast(k));
            }

            // DEBUG: Per-field comparison for Stage 1 (SpartanOuter)
            if (comptime debug_verbose) {
                // Compute eq table for Stage 1's r_cycle
                const n_vars = n_cycle_vars;
                const T = @as(usize, 1) << @intCast(n_vars);
                var r_cycle_rev = try self.allocator.alloc(F, n_vars);
                defer self.allocator.free(r_cycle_rev);
                for (0..n_vars) |i| r_cycle_rev[i] = r_cycle_bc1_spartan_outer[n_vars - 1 - i];
                const eq_table_s1 = try computeEqTableParallel(F, self.allocator, r_cycle_rev, n_vars, self.thread_pool);
                defer self.allocator.free(eq_table_s1);

                // Compute F_s[k] = Σ_{c:PC(c)=k} eq(r_cycle, c) for Stage 1
                var F_s_s1 = try self.allocator.alloc(F, bytecode_K);
                defer self.allocator.free(F_s_s1);
                @memset(F_s_s1, F.zero());
                for (0..T) |c| {
                    const step = trace.steps.items[c];
                    const pc_idx = pc_map.getPCForStep(step);
                    if (pc_idx < bytecode_K) {
                        F_s_s1[pc_idx] = F_s_s1[pc_idx].add(eq_table_s1[c]);
                    }
                }

                // Compute per-field bytecode-weighted sums for Stage 1:
                // Stage 1 = γ₁⁰·address + γ₁¹·imm + Σ_i γ₁^(2+i)·cf[i]
                var bc_addr_sum = F.zero();
                var bc_imm_sum = F.zero();
                var bc_cf_sums: [14]F = [_]F{F.zero()} ** 14;

                for (0..bytecode_K) |k| {
                    if (k >= bytecode_entries.len) break;
                    const entry = bytecode_entries[k];
                    bc_addr_sum = bc_addr_sum.add(F_s_s1[k].mul(F.fromU64(entry.address)));
                    const debug_imm_field: F = if (entry.opcode == 0x63 or entry.opcode == 0x23)
                        fieldFromI128(F, @as(i128, entry.imm))
                    else
                        F.fromU64(@as(u64, @bitCast(entry.imm)));
                    bc_imm_sum = bc_imm_sum.add(F_s_s1[k].mul(debug_imm_field));
                    for (0..14) |fi| {
                        if (entry.circuit_flags[fi]) {
                            bc_cf_sums[fi] = bc_cf_sums[fi].add(F_s_s1[k]);
                        }
                    }
                }

                // Get corresponding opening claims for SpartanOuter
                const getClaim = struct {
                    fn get(oc: *OpeningClaims(F), key: OpeningId) F {
                        return oc.get(key) orelse F.zero();
                    }
                }.get;
                const oc_addr = getClaim(opening_claims, .{ .Virtual = .{ .poly = .UnexpandedPC, .sumcheck_id = .SpartanOuter } });
                const oc_imm = getClaim(opening_claims, .{ .Virtual = .{ .poly = .Imm, .sumcheck_id = .SpartanOuter } });

                // Compare and print mismatches
                const addr_match = bc_addr_sum.eql(oc_addr);
                const imm_match = bc_imm_sum.eql(oc_imm);
                dbg("\n[BCRAF_FIELD_CMP] Stage 1 field-by-field comparison:\n", .{});
                dbg("  address: match={}\n", .{@as(u8, if (addr_match) 1 else 0)});
                if (!addr_match) {
                    const a1 = bc_addr_sum.toBytes();
                    const a2 = oc_addr.toBytes();
                    dbg("    bc_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{a1[0],a1[1],a1[2],a1[3],a1[4],a1[5],a1[6],a1[7]});
                    dbg("    oc_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{a2[0],a2[1],a2[2],a2[3],a2[4],a2[5],a2[6],a2[7]});
                }
                dbg("  imm: match={}\n", .{@as(u8, if (imm_match) 1 else 0)});
                if (!imm_match) {
                    const ib1 = bc_imm_sum.toBytes();
                    const ib2 = oc_imm.toBytes();
                    dbg("    bc_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{ib1[0],ib1[1],ib1[2],ib1[3],ib1[4],ib1[5],ib1[6],ib1[7]});
                    dbg("    oc_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{ib2[0],ib2[1],ib2[2],ib2[3],ib2[4],ib2[5],ib2[6],ib2[7]});
                }
                const cf_names = [14][]const u8{ "AddOp", "SubOp", "MulOp", "Load", "Store", "Jump", "WrLookup", "VirtInstr", "Assert", "NoUpdateUPC", "Advice", "IsCompr", "IsFirst", "IsLast" };
                for (0..14) |fi| {
                    const oc_cf = getClaim(opening_claims, .{ .Virtual = .{ .poly = .{ .OpFlags = @intCast(fi) }, .sumcheck_id = .SpartanOuter } });
                    const cf_match = bc_cf_sums[fi].eql(oc_cf);
                    if (!cf_match) {
                        dbg("  cf[{}] ({s}): MISMATCH\n", .{fi, cf_names[fi]});
                        const c1 = bc_cf_sums[fi].toBytes();
                        const c2 = oc_cf.toBytes();
                        dbg("    bc_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{c1[0],c1[1],c1[2],c1[3],c1[4],c1[5],c1[6],c1[7]});
                        dbg("    oc_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{c2[0],c2[1],c2[2],c2[3],c2[4],c2[5],c2[6],c2[7]});
                    }
                }
                // Also check non-RAF rv_claim_1 directly
                var rv1_recomp = F.zero();
                rv1_recomp = rv1_recomp.add(bc_addr_sum); // No gamma[0] - matches Jolt formula
                rv1_recomp = rv1_recomp.add(stage1_gammas[1].mul(bc_imm_sum));
                for (0..14) |fi| {
                    rv1_recomp = rv1_recomp.add(stage1_gammas[2 + fi].mul(bc_cf_sums[fi]));
                }
                const rv1_ext = getClaim(opening_claims, .{ .Virtual = .{ .poly = .UnexpandedPC, .sumcheck_id = .SpartanOuter } });
                _ = rv1_ext;
                // Compare rv1_recomp with rv_claim_1 from computeBytecodeReadRafInputClaim
                // rv1_recomp = Σ_k F_s[k] * val_1_no_raf(k) (the non-RAF part of recomputed)
                // rv1_opening = Σ_i gamma_i * opening_claim_i (from opening_claims)
                var rv1_opening = F.zero();
                rv1_opening = rv1_opening.add(oc_addr); // No gamma[0] - matches Jolt formula
                rv1_opening = rv1_opening.add(stage1_gammas[1].mul(oc_imm));
                for (0..14) |fi| {
                    const oc_cf_fi = getClaim(opening_claims, .{ .Virtual = .{ .poly = .{ .OpFlags = @intCast(fi) }, .sumcheck_id = .SpartanOuter } });
                    rv1_opening = rv1_opening.add(stage1_gammas[2 + fi].mul(oc_cf_fi));
                }
                const rv1_match = rv1_recomp.eql(rv1_opening);
                dbg("  rv1 non-RAF match: {}\n", .{@as(u8, if (rv1_match) 1 else 0)});

                // Check RAF contribution
                const raf_oc = getClaim(opening_claims, .{ .Virtual = .{ .poly = .PC, .sumcheck_id = .SpartanOuter } });
                var bc_pc_sum = F.zero();
                for (0..bytecode_K) |k| {
                    bc_pc_sum = bc_pc_sum.add(F_s_s1[k].mul(F.fromU64(@intCast(k))));
                }
                const raf_match = bc_pc_sum.eql(raf_oc);
                dbg("  PC/RAF match: {}\n", .{@as(u8, if (raf_match) 1 else 0)});
                if (!raf_match) {
                    const r1 = bc_pc_sum.toBytes();
                    const r2 = raf_oc.toBytes();
                    dbg("    bc_pc_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{r1[0],r1[1],r1[2],r1[3],r1[4],r1[5],r1[6],r1[7]});
                    dbg("    oc_pc_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{r2[0],r2[1],r2[2],r2[3],r2[4],r2[5],r2[6],r2[7]});
                }
                // Total claim check
                const total_recomp = rv1_recomp.add(bytecode_raf_gamma_powers[5].mul(bc_pc_sum));
                const total_ext = rv1_opening.add(bytecode_raf_gamma_powers[5].mul(raf_oc));
                dbg("  total_stage1_recomp match total_ext: {}\n", .{@as(u8, if (total_recomp.eql(total_ext)) 1 else 0)});
                dbg("  total_stage1_recomp match bcraf_per_stage_claims[0]: {}\n", .{@as(u8, if (total_recomp.eql(bcraf_per_stage_claims[0])) 1 else 0)});

                dbg("[BCRAF_FIELD_CMP] Done\n\n", .{});
            }

            // DEBUG: Per-field comparison for Stage 2 (SpartanProductVirtualization)
            if (comptime debug_verbose) {
                const n_vars = n_cycle_vars;
                const T = @as(usize, 1) << @intCast(n_vars);
                var r_cycle_rev2 = try self.allocator.alloc(F, n_vars);
                defer self.allocator.free(r_cycle_rev2);
                for (0..n_vars) |i| r_cycle_rev2[i] = r_cycle_bc2_product_virt[n_vars - 1 - i];
                const eq_table_s2 = try computeEqTableParallel(F, self.allocator, r_cycle_rev2, n_vars, self.thread_pool);
                defer self.allocator.free(eq_table_s2);

                // Compute per-field sums: Σ_c eq(r_cycle_2, c) * witness_field[c]
                // Stage 2 witnesses: JumpFlag, BranchFlag, IsRdNotZero, WriteLookupToRD
                var cycle_jump_sum = F.zero();
                var cycle_branch_sum = F.zero();
                var cycle_isrdnz_sum = F.zero();
                var cycle_wrlookup_sum = F.zero();

                for (0..T) |c| {
                    const step = trace.steps.items[c];
                    const pc_idx = pc_map.getPCForStep(step);
                    if (pc_idx < bytecode_K and pc_idx < bytecode_entries.len) {
                        const entry = bytecode_entries[pc_idx];
                        if (entry.circuit_flags[@intFromEnum(CircuitFlags.Jump)]) {
                            cycle_jump_sum = cycle_jump_sum.add(eq_table_s2[c]);
                        }
                        if (entry.instruction_flags[@intFromEnum(InstructionFlags.Branch)]) {
                            cycle_branch_sum = cycle_branch_sum.add(eq_table_s2[c]);
                        }
                        if (entry.instruction_flags[@intFromEnum(InstructionFlags.IsRdNotZero)]) {
                            cycle_isrdnz_sum = cycle_isrdnz_sum.add(eq_table_s2[c]);
                        }
                        if (entry.circuit_flags[@intFromEnum(CircuitFlags.WriteLookupOutputToRD)]) {
                            cycle_wrlookup_sum = cycle_wrlookup_sum.add(eq_table_s2[c]);
                        }
                    }
                }

                const getClaim2 = struct {
                    fn get(oc: *OpeningClaims(F), key: OpeningId) F {
                        return oc.get(key) orelse F.zero();
                    }
                }.get;

                const oc_jump = getClaim2(opening_claims, .{ .Virtual = .{ .poly = .{ .OpFlags = 5 }, .sumcheck_id = .SpartanProductVirtualization } });
                const oc_branch = getClaim2(opening_claims, .{ .Virtual = .{ .poly = .{ .InstructionFlags = 4 }, .sumcheck_id = .SpartanProductVirtualization } });
                const oc_isrdnz = getClaim2(opening_claims, .{ .Virtual = .{ .poly = .{ .InstructionFlags = 6 }, .sumcheck_id = .SpartanProductVirtualization } });
                const oc_wrlookup = getClaim2(opening_claims, .{ .Virtual = .{ .poly = .{ .OpFlags = 6 }, .sumcheck_id = .SpartanProductVirtualization } });

                dbg("\n[BCRAF_FIELD_CMP2] Stage 2 (SpartanProductVirt) field comparison:\n", .{});
                const fields2 = [4]struct { name: []const u8, bc: F, oc: F }{
                    .{ .name = "Jump(OpFlags=5)", .bc = cycle_jump_sum, .oc = oc_jump },
                    .{ .name = "Branch(InstrFlags=4)", .bc = cycle_branch_sum, .oc = oc_branch },
                    .{ .name = "IsRdNotZero(InstrFlags=6)", .bc = cycle_isrdnz_sum, .oc = oc_isrdnz },
                    .{ .name = "WriteLookupToRD(OpFlags=6)", .bc = cycle_wrlookup_sum, .oc = oc_wrlookup },
                };
                for (fields2) |f| {
                    const match2 = f.bc.eql(f.oc);
                    const b1 = f.bc.toBytes();
                    const b2 = f.oc.toBytes();
                    dbg("  {s}: {s}\n", .{f.name, if (match2) "MATCH" else "MISMATCH"});
                    dbg("    bc_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{b1[0],b1[1],b1[2],b1[3],b1[4],b1[5],b1[6],b1[7]});
                    dbg("    oc_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{b2[0],b2[1],b2[2],b2[3],b2[4],b2[5],b2[6],b2[7]});
                }

                // Compute rv2 from recomputed per-field values vs rv2 from opening claims
                var rv2_recomp = F.zero();
                rv2_recomp = rv2_recomp.add(stage2_gammas[0].mul(cycle_jump_sum));
                rv2_recomp = rv2_recomp.add(stage2_gammas[1].mul(cycle_branch_sum));
                rv2_recomp = rv2_recomp.add(stage2_gammas[2].mul(cycle_isrdnz_sum));
                rv2_recomp = rv2_recomp.add(stage2_gammas[3].mul(cycle_wrlookup_sum));

                var rv2_ext = F.zero();
                rv2_ext = rv2_ext.add(stage2_gammas[0].mul(oc_jump));
                rv2_ext = rv2_ext.add(stage2_gammas[1].mul(oc_branch));
                rv2_ext = rv2_ext.add(stage2_gammas[2].mul(oc_isrdnz));
                rv2_ext = rv2_ext.add(stage2_gammas[3].mul(oc_wrlookup));

                const rv2r = rv2_recomp.toBytes();
                const rv2e = rv2_ext.toBytes();
                dbg("  rv2_recomp_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{rv2r[0],rv2r[1],rv2r[2],rv2r[3],rv2r[4],rv2r[5],rv2r[6],rv2r[7]});
                dbg("  rv2_ext_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{rv2e[0],rv2e[1],rv2e[2],rv2e[3],rv2e[4],rv2e[5],rv2e[6],rv2e[7]});
                dbg("  rv2_match: {}\n", .{@as(u8, if (rv2_recomp.eql(rv2_ext)) 1 else 0)});

                dbg("[BCRAF_FIELD_CMP2] Done\n\n", .{});
            }

            // DEBUG: Per-field comparison for Stage 3 (RegistersReadWriteChecking)
            if (comptime debug_verbose) {
                const n_vars = n_cycle_vars;
                const T = @as(usize, 1) << @intCast(n_vars);
                var r_cycle_rev4 = try self.allocator.alloc(F, n_vars);
                defer self.allocator.free(r_cycle_rev4);
                for (0..n_vars) |i| r_cycle_rev4[i] = r_cycle_bc4_regs_rwc[n_vars - 1 - i];
                const eq_table_s4 = try computeEqTableParallel(F, self.allocator, r_cycle_rev4, n_vars, self.thread_pool);
                defer self.allocator.free(eq_table_s4);

                // For each field (rd, rs1, rs2), compute Σ_k F_s[k] * eq(entry[k].reg, r_register_4)
                // F_s[k] = Σ_c:PC(c)=k eq(r_cycle_4, c)
                // First compute F_s[k] for all k
                var F_s = try self.allocator.alloc(F, bytecode_K);
                defer self.allocator.free(F_s);
                @memset(F_s, F.zero());
                for (0..T) |c| {
                    const step = trace.steps.items[c];
                    const pc_idx = pc_map.getPCForStep(step);
                    if (pc_idx < bytecode_K) {
                        F_s[pc_idx] = F_s[pc_idx].add(eq_table_s4[c]);
                    }
                }

                var bc_rd_sum = F.zero();
                var bc_rs1_sum = F.zero();
                var bc_rs2_sum = F.zero();
                const REG_COUNT: usize = 128;
                for (0..bytecode_K) |k| {
                    if (k >= bytecode_entries.len) break;
                    const entry = bytecode_entries[k];
                    if (entry.rd < REG_COUNT) {
                        bc_rd_sum = bc_rd_sum.add(F_s[k].mul(eq_table_4[entry.rd]));
                    }
                    if (entry.rs1 < REG_COUNT) {
                        bc_rs1_sum = bc_rs1_sum.add(F_s[k].mul(eq_table_4[entry.rs1]));
                    }
                    if (entry.rs2 < REG_COUNT) {
                        bc_rs2_sum = bc_rs2_sum.add(F_s[k].mul(eq_table_4[entry.rs2]));
                    }
                }

                const getClaim3 = struct {
                    fn get(oc: *OpeningClaims(F), key: OpeningId) F {
                        return oc.get(key) orelse F.zero();
                    }
                }.get;

                const oc_rd = getClaim3(opening_claims, .{ .Virtual = .{ .poly = .RdWa, .sumcheck_id = .RegistersReadWriteChecking } });
                const oc_rs1 = getClaim3(opening_claims, .{ .Virtual = .{ .poly = .Rs1Ra, .sumcheck_id = .RegistersReadWriteChecking } });
                const oc_rs2 = getClaim3(opening_claims, .{ .Virtual = .{ .poly = .Rs2Ra, .sumcheck_id = .RegistersReadWriteChecking } });

                dbg("\n[BCRAF_FIELD_CMP3] Stage 3 (RegistersRWC) field comparison:\n", .{});
                const fields3 = [3]struct { name: []const u8, bc: F, oc: F }{
                    .{ .name = "RdWa", .bc = bc_rd_sum, .oc = oc_rd },
                    .{ .name = "Rs1Ra", .bc = bc_rs1_sum, .oc = oc_rs1 },
                    .{ .name = "Rs2Ra", .bc = bc_rs2_sum, .oc = oc_rs2 },
                };
                for (fields3) |f| {
                    const match3 = f.bc.eql(f.oc);
                    const b1 = f.bc.toBytesBE();
                    const b2 = f.oc.toBytesBE();
                    dbg("  {s}: {s}\n", .{ f.name, if (match3) "MATCH" else "MISMATCH" });
                    dbg("    bc_LE=[", .{});
                    for (0..8) |bi| dbg("{x:0>2}", .{b1[31 - bi]});
                    dbg("]\n", .{});
                    dbg("    oc_LE=[", .{});
                    for (0..8) |bi| dbg("{x:0>2}", .{b2[31 - bi]});
                    dbg("]\n", .{});
                }

                // Also compute and show combined claim
                var rv4_bc = F.zero();
                rv4_bc = rv4_bc.add(stage4_gammas[0].mul(bc_rd_sum));
                rv4_bc = rv4_bc.add(stage4_gammas[1].mul(bc_rs1_sum));
                rv4_bc = rv4_bc.add(stage4_gammas[2].mul(bc_rs2_sum));
                var rv4_oc = F.zero();
                rv4_oc = rv4_oc.add(stage4_gammas[0].mul(oc_rd));
                rv4_oc = rv4_oc.add(stage4_gammas[1].mul(oc_rs1));
                rv4_oc = rv4_oc.add(stage4_gammas[2].mul(oc_rs2));
                dbg("  rv4_bc match rv4_oc: {}\n", .{@as(u8, if (rv4_bc.eql(rv4_oc)) 1 else 0)});
                dbg("  rv4_bc match bcraf_per_stage[3]: {}\n", .{@as(u8, if (rv4_bc.eql(bcraf_per_stage_claims[3])) 1 else 0)});

                // Compute trace-based rd using val polys (should match bc-based)
                var trace_rd_sum = F.zero();
                var trace_rs1_sum = F.zero();
                var trace_rs2_sum = F.zero();
                var trace_rd_valpoly = F.zero(); // Using bytecode val poly like bc-based
                var trace_rs1_valpoly = F.zero();
                var trace_rs2_valpoly = F.zero();
                var n_mismatch: usize = 0;
                for (0..T) |c| {
                    const step = trace.steps.items[c];
                    const pc_idx = pc_map.getPCForStep(step);

                    // Val-poly-based (should match bc-based Σ_k F_s[k] * eq4[rd_k])
                    if (pc_idx < bytecode_K and pc_idx < bytecode_entries.len) {
                        const ent = bytecode_entries[pc_idx];
                        if (ent.rd < REG_COUNT) {
                            trace_rd_valpoly = trace_rd_valpoly.add(eq_table_s4[c].mul(eq_table_4[ent.rd]));
                        }
                        if (ent.rs1 < REG_COUNT) {
                            trace_rs1_valpoly = trace_rs1_valpoly.add(eq_table_s4[c].mul(eq_table_4[ent.rs1]));
                        }
                        if (ent.rs2 < REG_COUNT) {
                            trace_rs2_valpoly = trace_rs2_valpoly.add(eq_table_s4[c].mul(eq_table_4[ent.rs2]));
                        }
                    }

                    // Opening-claim-based (from trace raw instruction)
                    if (step.is_noop and !step.is_termination_store) continue;
                    const instr = step.instruction;
                    const opcode = instr & 0x7f;
                    const rd_raw: u8 = @truncate((instr >> 7) & 0x1f);
                    const rs1_raw: u8 = @truncate((instr >> 15) & 0x1f);
                    const rs2_raw: u8 = @truncate((instr >> 20) & 0x1f);

                    const writes_rd = switch (opcode) {
                        0x23, 0x63 => false,
                        else => true,
                    };
                    if (writes_rd and rd_raw != 0) {
                        trace_rd_sum = trace_rd_sum.add(eq_table_s4[c].mul(eq_table_4[rd_raw]));
                    }
                    const reads_rs1 = switch (opcode) {
                        0x13, 0x03, 0x67, 0x1b, 0x33, 0x3b, 0x23, 0x63 => true,
                        else => false,
                    };
                    if (reads_rs1) {
                        trace_rs1_sum = trace_rs1_sum.add(eq_table_s4[c].mul(eq_table_4[rs1_raw]));
                    }
                    const reads_rs2 = switch (opcode) {
                        0x33, 0x3b, 0x23, 0x63 => true,
                        else => false,
                    };
                    if (reads_rs2) {
                        trace_rs2_sum = trace_rs2_sum.add(eq_table_s4[c].mul(eq_table_4[rs2_raw]));
                    }
                    // Check for per-cycle rd contribution divergence
                    if (pc_idx < bytecode_K and pc_idx < bytecode_entries.len) {
                        const ent2 = bytecode_entries[pc_idx];
                        // Compute val-poly rd contribution for this cycle
                        const vp_rd_contrib = if (ent2.rd < REG_COUNT) eq_table_4[ent2.rd] else F.zero();
                        // Compute trace-based rd contribution for this cycle
                        const tr_rd_contrib = if (writes_rd and rd_raw != 0 and rd_raw < REG_COUNT)
                            eq_table_4[rd_raw]
                        else
                            F.zero();
                        if (!vp_rd_contrib.eql(tr_rd_contrib) and n_mismatch < 15) {
                            dbg("  [RD_DIVERGE] c={} k={} pc=0x{x} opc=0x{x:0>2} bc_rd={} raw_rd={} writes={} noop={} term={}\n", .{
                                c, pc_idx, step.pc, opcode, ent2.rd, rd_raw, @intFromBool(writes_rd),
                                @intFromBool(step.is_noop), @intFromBool(step.is_termination_store),
                            });
                            n_mismatch += 1;
                        }
                    }
                }
                dbg("  valpoly_rd match bc_rd: {}\n", .{@as(u8, if (trace_rd_valpoly.eql(bc_rd_sum)) 1 else 0)});
                dbg("  valpoly_rs1 match bc_rs1: {}\n", .{@as(u8, if (trace_rs1_valpoly.eql(bc_rs1_sum)) 1 else 0)});
                dbg("  valpoly_rs2 match bc_rs2: {}\n", .{@as(u8, if (trace_rs2_valpoly.eql(bc_rs2_sum)) 1 else 0)});
                dbg("  trace_rd match oc_rd: {}\n", .{@as(u8, if (trace_rd_sum.eql(oc_rd)) 1 else 0)});
                dbg("  valpoly_rd match oc_rd: {}\n", .{@as(u8, if (trace_rd_valpoly.eql(oc_rd)) 1 else 0)});
                // Critical: Does bc_rs1 match oc_rs1? This is the actual BCRAF check.
                dbg("  [RS1_MATCH] bc_rs1 == oc_rs1: {}\n", .{@as(u8, if (bc_rs1_sum.eql(oc_rs1)) 1 else 0)});
                dbg("  [RS1_MATCH] valpoly_rs1 == oc_rs1: {}\n", .{@as(u8, if (trace_rs1_valpoly.eql(oc_rs1)) 1 else 0)});
                // Per-cycle rs1 divergence: compare bytecode entry rs1 vs trace step rs1_index
                {
                    var rs1_div: usize = 0;
                    for (0..T) |c2| {
                        const step_c = trace.steps.items[c2];
                        if (step_c.is_noop and !step_c.is_termination_store) continue;
                        const pc_c = pc_map.getPCForStep(step_c);
                        if (pc_c >= bytecode_K or pc_c >= bytecode_entries.len) continue;
                        const bc_ent = bytecode_entries[pc_c];
                        // bc_ent.rs1 = bytecode entry rs1 (used in BCRAF)
                        // step_c.rs1_index = trace step rs1 (used in opening claim)
                        // step_c.rs1_read = whether rs1 is actually read
                        if (step_c.rs1_read) {
                            // Bytecode says rs1=bc_ent.rs1, trace says rs1=step_c.rs1_index
                            if (bc_ent.rs1 != step_c.rs1_index and rs1_div < 20) {
                                dbg("  [RS1_DIVERGE] c={} k={} pc=0x{x:0>8} bc_rs1={} trace_rs1={} opc=0x{x:0>2}\n", .{
                                    c2, pc_c, step_c.pc, bc_ent.rs1, step_c.rs1_index,
                                    step_c.instruction & 0x7f,
                                });
                                rs1_div += 1;
                            }
                        }
                    }
                    dbg("  [RS1_DIVERGE] total divergences: {}\n", .{rs1_div});
                    // Check for cycles where rs1_read=false but bytecode entry has rs1 < 128
                    var phantom_count: usize = 0;
                    var phantom_contrib = F.zero();
                    for (0..T) |c3| {
                        const step_d = trace.steps.items[c3];
                        if (step_d.is_noop and !step_d.is_termination_store) continue;
                        if (!step_d.rs1_read) {
                            const pc_d = pc_map.getPCForStep(step_d);
                            if (pc_d < bytecode_K and pc_d < bytecode_entries.len) {
                                const bc_d = bytecode_entries[pc_d];
                                if (bc_d.rs1 < REG_COUNT) {
                                    const contrib = eq_table_s4[c3].mul(eq_table_4[bc_d.rs1]);
                                    phantom_contrib = phantom_contrib.add(contrib);
                                    if (phantom_count < 10) {
                                        dbg("  [RS1_PHANTOM] c={} k={} opc=0x{x:0>2} bc_rs1={} rs1_read=false\n", .{
                                            c3, pc_d, step_d.instruction & 0x7f, bc_d.rs1,
                                        });
                                    }
                                    phantom_count += 1;
                                }
                            }
                        }
                    }
                    dbg("  [RS1_PHANTOM] count={}, nonzero={}\n", .{phantom_count, @as(u8, if (!phantom_contrib.eql(F.zero())) 1 else 0)});
                    // If bc_rs1 - phantom_contrib == oc_rs1, then the phantom entries explain the mismatch
                    const adjusted = bc_rs1_sum.sub(phantom_contrib);
                    dbg("  [RS1_PHANTOM] bc_rs1 - phantom == oc_rs1: {}\n", .{@as(u8, if (adjusted.eql(oc_rs1)) 1 else 0)});
                }
                const t_rd = trace_rd_sum.toBytesBE();
                const t_rs1 = trace_rs1_sum.toBytesBE();
                const t_rs2 = trace_rs2_sum.toBytesBE();
                dbg("  trace_rd_LE=[", .{});
                for (0..8) |bi| dbg("{x:0>2}", .{t_rd[31 - bi]});
                dbg("] match_oc={}\n", .{@as(u8, if (trace_rd_sum.eql(oc_rd)) 1 else 0)});
                dbg("  trace_rs1_LE=[", .{});
                for (0..8) |bi| dbg("{x:0>2}", .{t_rs1[31 - bi]});
                dbg("] match_oc={}\n", .{@as(u8, if (trace_rs1_sum.eql(oc_rs1)) 1 else 0)});
                dbg("  trace_rs2_LE=[", .{});
                for (0..8) |bi| dbg("{x:0>2}", .{t_rs2[31 - bi]});
                dbg("] match_oc={}\n", .{@as(u8, if (trace_rs2_sum.eql(oc_rs2)) 1 else 0)});
                // CRITICAL: Compute RdWa claim using EXACT same logic as Stage 4 prover
                // Stage 4 sets rd_wa_poly[rd * T + cycle] = 1 when step.rd_written (including rd=0)
                // After sumcheck: rd_wa_claim = Σ_c eq(r_cycle, c) * eq(rd_index(c), r_addr) * 1{rd_written(c)}
                {
                    var direct_rd_claim = F.zero();
                    var rd_written_0_count: usize = 0;
                    var rd_not_written_but_bc_has_rd: usize = 0;
                    for (0..T) |c4| {
                        const step_e = trace.steps.items[c4];
                        if (step_e.is_noop) {
                            // Stage 4 prover skips noop cycles
                            continue;
                        }
                        if (step_e.rd_written) {
                            const rd_idx = @as(usize, step_e.rd_index);
                            if (rd_idx < REG_COUNT) {
                                direct_rd_claim = direct_rd_claim.add(eq_table_s4[c4].mul(eq_table_4[rd_idx]));
                            }
                            if (rd_idx == 0) rd_written_0_count += 1;
                        } else {
                            // Check if bytecode entry has rd < 128 for this cycle
                            const pc_e = pc_map.getPCForStep(step_e);
                            if (pc_e < bytecode_K and pc_e < bytecode_entries.len) {
                                if (bytecode_entries[pc_e].rd < REG_COUNT) {
                                    rd_not_written_but_bc_has_rd += 1;
                                    if (rd_not_written_but_bc_has_rd <= 5) {
                                        dbg("  [RD_GHOST] c={} k={} pc=0x{x:0>8} opc=0x{x:0>2} bc_rd={} step.rd_idx={} rd_written=0\n", .{
                                            c4, pc_e, step_e.pc, step_e.instruction & 0x7f,
                                            bytecode_entries[pc_e].rd, step_e.rd_index,
                                        });
                                    }
                                }
                            }
                        }
                    }
                    const drcl = direct_rd_claim.toBytesBE();
                    dbg("  [DIRECT_RD] claim_LE=[", .{});
                    for (0..8) |bi| dbg("{x:0>2}", .{drcl[31 - bi]});
                    dbg("] match_oc={} match_bc={}\n", .{
                        @as(u8, if (direct_rd_claim.eql(oc_rd)) 1 else 0),
                        @as(u8, if (direct_rd_claim.eql(bc_rd_sum)) 1 else 0),
                    });
                    dbg("  [DIRECT_RD] rd_written_0_count={} rd_not_written_but_bc_has_rd={}\n", .{
                        rd_written_0_count, rd_not_written_but_bc_has_rd,
                    });
                    // Compute difference
                    const diff = bc_rd_sum.sub(direct_rd_claim);
                    const diff_le = diff.toBytesBE();
                    dbg("  [DIRECT_RD] bc_rd - direct = [", .{});
                    for (0..8) |bi| dbg("{x:0>2}", .{diff_le[31 - bi]});
                    dbg("]\n", .{});
                    // Check: does direct_rd match oc_rd? If not, Stage 4 prover has a bug
                    const diff2 = direct_rd_claim.sub(oc_rd);
                    const diff2_le = diff2.toBytesBE();
                    dbg("  [DIRECT_RD] direct - oc_rd = [", .{});
                    for (0..8) |bi| dbg("{x:0>2}", .{diff2_le[31 - bi]});
                    dbg("]\n", .{});
                }
                dbg("[BCRAF_FIELD_CMP3] Done\n\n", .{});
            }

            // DEBUG: Per-field comparison for Stage 4 (RegistersValEval + InstructionReadRaf)
            if (comptime debug_verbose) {
                const n_vars = n_cycle_vars;
                const T = @as(usize, 1) << @intCast(n_vars);
                var r_cycle_rev5 = try self.allocator.alloc(F, n_vars);
                defer self.allocator.free(r_cycle_rev5);
                for (0..n_vars) |i| r_cycle_rev5[i] = r_cycle_bc5_regs_val[n_vars - 1 - i];
                const eq_table_s5 = try computeEqTableParallel(F, self.allocator, r_cycle_rev5, n_vars, self.thread_pool);
                defer self.allocator.free(eq_table_s5);

                var F_s5 = try self.allocator.alloc(F, bytecode_K);
                defer self.allocator.free(F_s5);
                @memset(F_s5, F.zero());
                for (0..T) |c| {
                    const step = trace.steps.items[c];
                    const pc_idx = pc_map.getPCForStep(step);
                    if (pc_idx < bytecode_K) {
                        F_s5[pc_idx] = F_s5[pc_idx].add(eq_table_s5[c]);
                    }
                }

                const REG_COUNT5: usize = 128;
                var bc_rd5_sum = F.zero();
                var bc_iraf_sum = F.zero();
                var bc_table_sums: [41]F = undefined;
                for (0..41) |t| bc_table_sums[t] = F.zero();
                for (0..bytecode_K) |k| {
                    if (k >= bytecode_entries.len) break;
                    const entry = bytecode_entries[k];
                    if (entry.rd < REG_COUNT5) {
                        bc_rd5_sum = bc_rd5_sum.add(F_s5[k].mul(eq_table_5[entry.rd]));
                    }
                    if (!entry.is_interleaved) {
                        bc_iraf_sum = bc_iraf_sum.add(F_s5[k]);
                    }
                    if (entry.lookup_table_index < 41) {
                        bc_table_sums[entry.lookup_table_index] = bc_table_sums[entry.lookup_table_index].add(F_s5[k]);
                    }
                }

                const getClaim5 = struct {
                    fn get(oc: *OpeningClaims(F), key: OpeningId) F {
                        return oc.get(key) orelse F.zero();
                    }
                }.get;

                const oc_rd5 = getClaim5(opening_claims, .{ .Virtual = .{ .poly = .RdWa, .sumcheck_id = .RegistersValEvaluation } });
                const oc_iraf = getClaim5(opening_claims, .{ .Virtual = .{ .poly = .InstructionRafFlag, .sumcheck_id = .InstructionReadRaf } });

                dbg("\n[BCRAF_FIELD_CMP4] Stage 4 (RegistersValEval+InstrReadRaf) field comparison:\n", .{});
                const rd5_match = bc_rd5_sum.eql(oc_rd5);
                const iraf_match = bc_iraf_sum.eql(oc_iraf);
                const b1r = bc_rd5_sum.toBytesBE();
                const b2r = oc_rd5.toBytesBE();
                dbg("  RdWa: {s}\n", .{if (rd5_match) "MATCH" else "MISMATCH"});
                dbg("    bc_LE=[", .{});
                for (0..8) |bi| dbg("{x:0>2}", .{b1r[31 - bi]});
                dbg("]\n", .{});
                dbg("    oc_LE=[", .{});
                for (0..8) |bi| dbg("{x:0>2}", .{b2r[31 - bi]});
                dbg("]\n", .{});
                const b1i = bc_iraf_sum.toBytesBE();
                const b2i = oc_iraf.toBytesBE();
                dbg("  InstructionRafFlag: {s}\n", .{if (iraf_match) "MATCH" else "MISMATCH"});
                dbg("    bc_LE=[", .{});
                for (0..8) |bi| dbg("{x:0>2}", .{b1i[31 - bi]});
                dbg("]\n", .{});
                dbg("    oc_LE=[", .{});
                for (0..8) |bi| dbg("{x:0>2}", .{b2i[31 - bi]});
                dbg("]\n", .{});

                // Check first few table flags
                var table_mismatches: usize = 0;
                for (0..41) |t| {
                    const oc_tf = getClaim5(opening_claims, .{ .Virtual = .{ .poly = .{ .LookupTableFlag = t }, .sumcheck_id = .InstructionReadRaf } });
                    if (!bc_table_sums[t].eql(oc_tf)) {
                        table_mismatches += 1;
                        if (table_mismatches <= 5) {
                            const bt1 = bc_table_sums[t].toBytesBE();
                            const bt2 = oc_tf.toBytesBE();
                            dbg("  LookupTableFlag[{}]: MISMATCH\n", .{t});
                            dbg("    bc_LE=[", .{});
                            for (0..8) |bi| dbg("{x:0>2}", .{bt1[31 - bi]});
                            dbg("]\n", .{});
                            dbg("    oc_LE=[", .{});
                            for (0..8) |bi| dbg("{x:0>2}", .{bt2[31 - bi]});
                            dbg("]\n", .{});
                        }
                    }
                }
                dbg("  Total LookupTableFlag mismatches: {}\n", .{table_mismatches});

                // Compute per-cycle iraf sum by iterating trace and checking opcode-based identity path
                // This mirrors Stage 5's cycle_is_identity_path logic
                var trace_iraf_sum = F.zero();
                var bc_vs_trace_mismatches: usize = 0;
                for (0..T) |c| {
                    const step = trace.steps.items[c];
                    const pc_idx = pc_map.getPCForStep(step);

                    // Compute identity path from instruction opcode (same as Stage 5)
                    const instr = step.instruction;
                    const opcode_7: u8 = @truncate(instr & 0x7F);
                    const funct3_3: u3 = @truncate((instr >> 12) & 0x7);
                    const funct7_7: u7 = @truncate(instr >> 25);
                    const trace_is_identity = switch (opcode_7) {
                        0x33 => (funct3_3 == 0 and funct7_7 == 0) or // ADD
                            (funct3_3 == 0 and funct7_7 == 0x20) or // SUB
                            (funct7_7 == 0x01 and funct3_3 == 0) or // MUL
                            (funct7_7 == 0x01 and funct3_3 == 3), // MULHU
                        0x13 => (funct3_3 == 0), // ADDI
                        0x1b => (funct3_3 == 0), // ADDIW
                        0x3b => (funct3_3 == 0 and funct7_7 == 0) or // ADDW
                            (funct3_3 == 0 and funct7_7 == 0x20), // SUBW
                        0x37 => true, // LUI
                        0x17 => true, // AUIPC
                        0x6f => true, // JAL
                        0x67 => true, // JALR
                        0x02 => true, // VirtualAdvice (Advice → identity path)
                        0x42 => true, // VirtualZeroExtendWord (AddOperands → identity path)
                        0x0B => true, // VirtualSignExtendWord (AddOperands → identity path)
                        0x2B => true, // VirtualMULI (MultiplyOperands → identity path)
                        else => false,
                    };

                    // bytecode path
                    const bc_raf: bool = if (pc_idx < bytecode_entries.len) !bytecode_entries[pc_idx].is_interleaved else false;

                    if (trace_is_identity) {
                        trace_iraf_sum = trace_iraf_sum.add(eq_table_s5[c]);
                    }

                    if (trace_is_identity != bc_raf and bc_vs_trace_mismatches < 10) {
                        dbg("  [IRAF_MISMATCH] c={} pc_idx={} noop={} trace_ident={} bc_raf={} opcode=0x{x:0>2}\n", .{
                            c, pc_idx, @intFromBool(step.is_noop), @intFromBool(trace_is_identity), @intFromBool(bc_raf), opcode_7,
                        });
                        if (pc_idx < bytecode_entries.len) {
                            dbg("    bc_cf=[", .{});
                            for (0..14) |fi| {
                                if (fi > 0) dbg(",", .{});
                                dbg("{}", .{@intFromBool(bytecode_entries[pc_idx].circuit_flags[fi])});
                            }
                            dbg("] bc_is_interleaved={}\n", .{@intFromBool(bytecode_entries[pc_idx].is_interleaved)});
                        }
                        bc_vs_trace_mismatches += 1;
                    }
                }
                const ti_le = trace_iraf_sum.toBytesBE();
                dbg("  trace_iraf_sum_LE=[", .{});
                for (0..8) |bi| dbg("{x:0>2}", .{ti_le[31 - bi]});
                dbg("] match_oc={} match_bc={}\n", .{
                    @intFromBool(trace_iraf_sum.eql(oc_iraf)),
                    @intFromBool(trace_iraf_sum.eql(bc_iraf_sum)),
                });
                dbg("  bc_vs_trace mismatches: {}\n", .{bc_vs_trace_mismatches});

                dbg("[BCRAF_FIELD_CMP4] Done\n\n", .{});
            }

            var bytecode_gamma_arr: [7]F = undefined;
            for (0..7) |i| {
                bytecode_gamma_arr[i] = bytecode_raf_gamma_powers[i];
            }
            var bytecode_prover = try BytecodeReadRafProver(F).init(
                self.allocator, trace, pc_map, bytecode_val_polys,
                bytecode_log_k, n_cycle_vars, bytecode_d, log_k_chunk,
                bytecode_gamma_arr,
                [5][]const F{
                    r_cycle_bc1_spartan_outer,
                    r_cycle_bc2_product_virt,
                    r_cycle_bc3_spartan_shift,
                    r_cycle_bc4_regs_rwc,
                    r_cycle_bc5_regs_val,
                },
                bytecode_int_poly,
                bcraf_per_stage_claims,
                self.thread_pool,
            );
            defer bytecode_prover.deinit();

            // Debug: Compare prover's initial BytecodeReadRaf claim with opening-claims-derived claim
            if (comptime debug_verbose) {
                var prover_initial = F.zero();
                for (0..5) |s| {
                    prover_initial = prover_initial.add(bytecode_prover.gamma_powers[s].mul(bytecode_prover.stage_claims[s]));
                }
                const pi_be = prover_initial.toBytesBE();
                const oc_be = bytecodeReadRaf_input.toBytesBE();
                dbg("\n[S6P_BCRAF_COMPARE] prover_initial_LE=[", .{});
                for (0..32) |bi| dbg("{x:0>2}", .{pi_be[31 - bi]});
                dbg("]\n[S6P_BCRAF_COMPARE] opening_claims_LE=[", .{});
                for (0..32) |bi| dbg("{x:0>2}", .{oc_be[31 - bi]});
                dbg("]\n[S6P_BCRAF_COMPARE] match={}\n", .{@as(u8, if (prover_initial.eql(bytecodeReadRaf_input)) 1 else 0)});

                for (0..5) |s| {
                    const ps_be = bytecode_prover.stage_claims[s].toBytesBE();
                    const os_be = bcraf_per_stage_claims[s].toBytesBE();
                    const sm = @as(u8, if (bytecode_prover.stage_claims[s].eql(bcraf_per_stage_claims[s])) 1 else 0);
                    if (sm == 0) {
                        dbg("[S6P_BCRAF_COMPARE] stage[{}] MISMATCH! prover_LE=[", .{s});
                        for (0..32) |bi| dbg("{x:0>2}", .{ps_be[31 - bi]});
                        dbg("] opening_LE=[", .{});
                        for (0..32) |bi| dbg("{x:0>2}", .{os_be[31 - bi]});
                        dbg("]\n", .{});
                    }
                }
            }

            // Debug: print r_cycle values for comparison with Jolt
            {
                const r_cycles = [5][]const F{
                    r_cycle_bc1_spartan_outer,
                    r_cycle_bc2_product_virt,
                    r_cycle_bc3_spartan_shift,
                    r_cycle_bc4_regs_rwc,
                    r_cycle_bc5_regs_val,
                };
                for (0..5) |s| {
                    dbg("[ZOLT_BCRAF] r_cycle[{}] (len={}):", .{ s, r_cycles[s].len });
                    for (0..@min(r_cycles[s].len, 4)) |i| {
                        const v_le = r_cycles[s][i].toBytes();
                        dbg(" [{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}]", .{
                            v_le[0], v_le[1], v_le[2], v_le[3], v_le[4], v_le[5], v_le[6], v_le[7],
                        });
                    }
                    if (r_cycles[s].len > 4) dbg("...", .{});
                    dbg("\n", .{});
                }
            }

            // ====================================================================
            // Append input claims and get batching coefficients
            // ====================================================================

            dbg("[STAGE6] Transcript before input_claims: round={}\n", .{transcript.n_rounds});

            transcript.appendScalar("sumcheck_claim", bytecodeReadRaf_input);
            transcript.appendScalar("sumcheck_claim", booleanity_input);
            transcript.appendScalar("sumcheck_claim", hammingBooleanity_input);
            transcript.appendScalar("sumcheck_claim", ramRaVirtual_input);
            transcript.appendScalar("sumcheck_claim", lookupsRaVirtual_input);
            transcript.appendScalar("sumcheck_claim", incClaimReduction_input);

            const batch = try self.allocator.alloc(F, 6);
            defer self.allocator.free(batch);
            for (0..6) |i| {
                batch[i] = transcript.challengeScalarFull();
            }

            const input_claims = [6]F{
                bytecodeReadRaf_input,
                booleanity_input,
                hammingBooleanity_input,
                ramRaVirtual_input,
                lookupsRaVirtual_input,
                incClaimReduction_input,
            };
            const num_rounds_arr = [6]usize{
                bytecodeReadRaf_rounds,
                booleanity_rounds,
                hammingBooleanity_rounds,
                ramRaVirtual_rounds,
                lookupsRaVirtual_rounds,
                incClaimReduction_rounds,
            };

            var batched_claim = F.zero();
            for (0..6) |i| {
                const scale = max_num_rounds - num_rounds_arr[i];
                var scaled = input_claims[i];
                for (0..scale) |_| scaled = scaled.add(scaled);
                batched_claim = batched_claim.add(batch[i].mul(scaled));
            }

            // Debug: print the initial batched claim and all batch coefficients
            {
                const bc_be = batched_claim.toBytesBE();
                dbg("[S6P_BATCHED] initial_batched_claim_LE=[", .{});
                for (0..32) |bi| dbg("{x:0>2}", .{bc_be[31 - bi]});
                dbg("]\n", .{});
                for (0..6) |i| {
                    const b_be = batch[i].toBytesBE();
                    const ic_be = input_claims[i].toBytesBE();
                    dbg("[S6P_BATCHED] batch[{}]_LE=[", .{i});
                    for (0..32) |bi| dbg("{x:0>2}", .{b_be[31 - bi]});
                    dbg("] input_claim_LE=[", .{});
                    for (0..32) |bi| dbg("{x:0>2}", .{ic_be[31 - bi]});
                    dbg("] rounds={}\n", .{num_rounds_arr[i]});
                }
            }

            // ====================================================================
            // Run batched sumcheck
            // ====================================================================

            var challenges = try self.allocator.alloc(F, max_num_rounds);
            errdefer self.allocator.free(challenges);

            var instance_claims: [6]F = input_claims;
            var current_batched_claim = batched_claim;

            const num_evals = max_degree + 1;
            const num_compressed = max_degree;

            // Track Phase 1 address challenges for BytecodeReadRaf
            var bytecode_addr_challenges = try self.allocator.alloc(F, bytecode_log_k);
            defer self.allocator.free(bytecode_addr_challenges);

            for (0..max_num_rounds) |round| {
                const remaining_rounds = max_num_rounds - round;

                var combined_evals = try self.allocator.alloc(F, num_evals);
                defer self.allocator.free(combined_evals);
                @memset(combined_evals, F.zero());

                // Per-instance cached round poly evals for claim tracking
                // We cache each instance's round poly so we don't recompute after challenge
                // Phase 1: degree-2 coefficients [a0, a1, a2] for p(x) = a0 + a1*x + a2*x^2
                var cached_bc_phase1_coeffs: [3]F = undefined;
                var cached_bc_phase1_per_stage: [5][2]F = undefined;
                var cached_bc_phase2: ?[]F = null;
                var cached_hamming: [4]F = undefined;
                var cached_ram_ra: ?[]F = null;
                var cached_lookups_ra: ?[]F = null;
                var cached_inc: [3]F = undefined; // Vandermonde: [p(0), p(1), p(2)]
                var cached_inc_p1: F = F.zero(); // recovered p(1)

                // Track which instances are active this round
                var inst_active: [6]bool = .{ false, false, false, false, false, false };
                const debug_r5 = (round == 5 or round == 6);
                // Debug: per-instance contribution to combined_evals[0] and [1]
                var dbg_inst_p0: [6]F = .{F.zero()} ** 6;
                var dbg_inst_p1: [6]F = .{F.zero()} ** 6;

                // Instance 0: BytecodeReadRaf - REAL prover
                {
                    const inst = 0;
                    if (remaining_rounds > num_rounds_arr[inst]) {
                        // Not started yet - constant polynomial
                        const scale = remaining_rounds - num_rounds_arr[inst] - 1;
                        var scaled = input_claims[inst];
                        for (0..scale) |_| scaled = scaled.add(scaled);
                        const contrib = batch[inst].mul(scaled);
                        for (0..num_evals) |j_eval| {
                            combined_evals[j_eval] = combined_evals[j_eval].add(contrib);
                        }
                    } else {
                        inst_active[inst] = true;
                        if (bytecode_prover.phase == 0) {
                            // Phase 1: address binding (degree-2 poly)
                            // computeRoundPolyPhase1 returns aggregated [p(0), p(2)] and per-stage evals
                            const phase1_result = bytecode_prover.computeRoundPolyPhase1();
                            cached_bc_phase1_per_stage = phase1_result.per_stage;
                            const p0 = phase1_result.agg[0];
                            const p2 = phase1_result.agg[1];
                            // Recover p(1) from sumcheck constraint: p(0) + p(1) = claim
                            const p1 = instance_claims[inst].sub(p0);

                            if (round < 2) {
                                const bc_sum = p0.add(p1);
                                dbg("  [S6P] R{} BC_Phase1 p(0)={any} p(1)={any} p(2)={any} sum={any} claim={any}\n", .{
                                    round,
                                    p0.toBytesBE()[0..8],
                                    p1.toBytesBE()[0..8],
                                    p2.toBytesBE()[0..8],
                                    bc_sum.toBytesBE()[0..8],
                                    instance_claims[0].toBytesBE()[0..8],
                                });
                            }

                            // Interpolate degree-2 coefficients from evals at {0, 1, 2}
                            // p(x) = a0 + a1*x + a2*x^2
                            // a0 = p(0)
                            // a2 = (p(2) - 2*p(1) + p(0)) / 2
                            // a1 = p(1) - p(0) - a2
                            const two = F.fromU64(2);
                            const two_inv = two.inverse().?;
                            const a0 = p0;
                            const a2 = p2.sub(p1.add(p1)).add(p0).mul(two_inv);
                            const a1 = p1.sub(p0).sub(a2);
                            cached_bc_phase1_coeffs = [3]F{ a0, a1, a2 };

                            // Evaluate degree-2 poly at all finite points for combined_evals
                            // combined_evals format (Vandermonde): [p(0), p(1), ..., p(max_degree)]
                            for (0..num_evals) |k| {
                                const x = F.fromU64(@intCast(k));
                                const pk = a0.add(x.mul(a1.add(x.mul(a2))));
                                combined_evals[k] = combined_evals[k].add(batch[inst].mul(pk));
                            }
                        } else {
                            // Phase 2: cycle binding (degree bytecode_d+1)
                            const polys = try bytecode_prover.computeRoundPolyPhase2(self.allocator);
                            cached_bc_phase2 = polys;
                            if (debug_r5) {
                                const p01 = polys[0].add(polys[1]);
                                const p01_ok: u8 = if (std.mem.eql(u8, &p01.toBytesBE(), &instance_claims[inst].toBytesBE())) 1 else 0;
                                dbg("  [R5_DBG] inst0_phase2 polys_len={} p(0)+p(1)=claim? {}\n", .{ polys.len, p01_ok });
                            }
                            addInstanceEvalsToCombibed(F, combined_evals, polys, batch[inst], num_evals);
                        }
                    }
                }

                dbg_inst_p0[0] = combined_evals[0];
                dbg_inst_p1[0] = combined_evals[1];

                if (debug_r5) {
                    const e0 = combined_evals[0].toBytes();
                    const e1 = combined_evals[1].toBytes();
                    dbg("  [R5_DBG] after inst0: e[0]_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}] e[1]_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                        e0[0], e0[1], e0[2], e0[3], e0[4], e0[5], e0[6], e0[7],
                        e1[0], e1[1], e1[2], e1[3], e1[4], e1[5], e1[6], e1[7],
                    });
                }

                // Instance 1: Booleanity - REAL prover (degree 3)
                var cached_booleanity: ?[]F = null;
                {
                    const inst = 1;
                    if (remaining_rounds > num_rounds_arr[inst]) {
                        const scale = remaining_rounds - num_rounds_arr[inst] - 1;
                        var scaled = input_claims[inst];
                        for (0..scale) |_| scaled = scaled.add(scaled);
                        const contrib = batch[inst].mul(scaled);
                        for (0..num_evals) |j_eval| {
                            combined_evals[j_eval] = combined_evals[j_eval].add(contrib);
                        }
                    } else {
                        inst_active[inst] = true;
                        const polys = try booleanity_prover.computeRoundPoly(self.allocator, instance_claims[inst]);
                        cached_booleanity = polys;
                        {
                            const p01 = polys[0].add(polys[1]);
                            const p01_ok: u8 = if (std.mem.eql(u8, &p01.toBytesBE(), &instance_claims[inst].toBytesBE())) 1 else 0;
                            const p0b = polys[0].toBytesBE();
                            const p1b = polys[1].toBytesBE();
                            dbg("  [S6P] R{} Bool p(0)+p(1)=claim? {} phase={} p0=[{x:0>2}{x:0>2}{x:0>2}{x:0>2}] p1=[{x:0>2}{x:0>2}{x:0>2}{x:0>2}]\n", .{
                                round, p01_ok, if (booleanity_prover.round < booleanity_prover.log_k_chunk) @as(u8, 1) else 2,
                                p0b[31], p0b[30], p0b[29], p0b[28],
                                p1b[31], p1b[30], p1b[29], p1b[28],
                            });
                        }
                        addFixedEvalsToCombibed(F, combined_evals, polys, 4, batch[inst], num_evals);
                    }
                }
                dbg_inst_p0[1] = combined_evals[0];
                dbg_inst_p1[1] = combined_evals[1];

                if (debug_r5) {
                    const e0 = combined_evals[0].toBytes();
                    const e1 = combined_evals[1].toBytes();
                    dbg("  [R5_DBG] after inst1: e[0]_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}] e[1]_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                        e0[0], e0[1], e0[2], e0[3], e0[4], e0[5], e0[6], e0[7],
                        e1[0], e1[1], e1[2], e1[3], e1[4], e1[5], e1[6], e1[7],
                    });
                }

                // Instance 2: HammingBooleanity - REAL prover
                {
                    const inst = 2;
                    if (remaining_rounds > num_rounds_arr[inst]) {
                        const scale = remaining_rounds - num_rounds_arr[inst] - 1;
                        var scaled = input_claims[inst];
                        for (0..scale) |_| scaled = scaled.add(scaled);
                        const contrib = batch[inst].mul(scaled);
                        for (0..num_evals) |j_eval| {
                            combined_evals[j_eval] = combined_evals[j_eval].add(contrib);
                        }
                    } else {
                        inst_active[inst] = true;
                        const polys = hamming_prover.computeRoundPoly();
                        cached_hamming = polys;
                        addFixedEvalsToCombibed(F, combined_evals, &polys, 4, batch[inst], num_evals);
                    }
                }
                dbg_inst_p0[2] = combined_evals[0];
                dbg_inst_p1[2] = combined_evals[1];

                if (debug_r5) {
                    const e0 = combined_evals[0].toBytes();
                    const e1 = combined_evals[1].toBytes();
                    dbg("  [R5_DBG] after inst2: e[0]_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}] e[1]_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                        e0[0], e0[1], e0[2], e0[3], e0[4], e0[5], e0[6], e0[7],
                        e1[0], e1[1], e1[2], e1[3], e1[4], e1[5], e1[6], e1[7],
                    });
                }

                // Instance 3: RamRaVirtual - REAL prover
                {
                    const inst = 3;
                    if (remaining_rounds > num_rounds_arr[inst]) {
                        const scale = remaining_rounds - num_rounds_arr[inst] - 1;
                        var scaled = input_claims[inst];
                        for (0..scale) |_| scaled = scaled.add(scaled);
                        const contrib = batch[inst].mul(scaled);
                        for (0..num_evals) |j_eval| {
                            combined_evals[j_eval] = combined_evals[j_eval].add(contrib);
                        }
                    } else {
                        inst_active[inst] = true;
                        const polys = try ram_ra_prover.computeRoundPoly(self.allocator);
                        cached_ram_ra = polys;
                        if (debug_r5) {
                            // Check p(0)+p(1)=claim for RamRaVirtual
                            const p01 = polys[0].add(polys[1]);
                            const p01_ok: u8 = if (std.mem.eql(u8, &p01.toBytesBE(), &instance_claims[inst].toBytesBE())) 1 else 0;
                            dbg("  [R5_DBG] inst3 polys_len={} p(0)+p(1)=claim? {}\n", .{ polys.len, p01_ok });
                            const p0_le = polys[0].toBytes();
                            const p1_le = polys[1].toBytes();
                            const claim_le = instance_claims[inst].toBytes();
                            dbg("  [R5_DBG] inst3 p(0)_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                                p0_le[0], p0_le[1], p0_le[2], p0_le[3], p0_le[4], p0_le[5], p0_le[6], p0_le[7],
                            });
                            dbg("  [R5_DBG] inst3 p(1)_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                                p1_le[0], p1_le[1], p1_le[2], p1_le[3], p1_le[4], p1_le[5], p1_le[6], p1_le[7],
                            });
                            dbg("  [R5_DBG] inst3 claim_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                                claim_le[0], claim_le[1], claim_le[2], claim_le[3], claim_le[4], claim_le[5], claim_le[6], claim_le[7],
                            });
                        }
                        addInstanceEvalsToCombibed(F, combined_evals, polys, batch[inst], num_evals);
                    }
                }
                dbg_inst_p0[3] = combined_evals[0];
                dbg_inst_p1[3] = combined_evals[1];

                if (debug_r5) {
                    const e0 = combined_evals[0].toBytes();
                    const e1 = combined_evals[1].toBytes();
                    dbg("  [R5_DBG] after inst3: e[0]_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}] e[1]_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                        e0[0], e0[1], e0[2], e0[3], e0[4], e0[5], e0[6], e0[7],
                        e1[0], e1[1], e1[2], e1[3], e1[4], e1[5], e1[6], e1[7],
                    });
                }

                // Instance 4: LookupsRaVirtual - REAL prover
                {
                    const inst = 4;
                    if (remaining_rounds > num_rounds_arr[inst]) {
                        const scale = remaining_rounds - num_rounds_arr[inst] - 1;
                        var scaled = input_claims[inst];
                        for (0..scale) |_| scaled = scaled.add(scaled);
                        const contrib = batch[inst].mul(scaled);
                        for (0..num_evals) |j_eval| {
                            combined_evals[j_eval] = combined_evals[j_eval].add(contrib);
                        }
                    } else {
                        inst_active[inst] = true;
                        const polys = try lookups_ra_prover.computeRoundPoly(self.allocator);
                        cached_lookups_ra = polys;
                        if (debug_r5) {
                            const p01 = polys[0].add(polys[1]);
                            const p01_ok: u8 = if (std.mem.eql(u8, &p01.toBytesBE(), &instance_claims[inst].toBytesBE())) 1 else 0;
                            dbg("  [R5_DBG] inst4 polys_len={} p(0)+p(1)=claim? {}\n", .{ polys.len, p01_ok });
                            const p0_le = polys[0].toBytes();
                            const p1_le = polys[1].toBytes();
                            const cl_le = instance_claims[inst].toBytes();
                            dbg("  [R5_DBG] inst4 p(0)_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                                p0_le[0], p0_le[1], p0_le[2], p0_le[3], p0_le[4], p0_le[5], p0_le[6], p0_le[7],
                            });
                            dbg("  [R5_DBG] inst4 p(1)_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                                p1_le[0], p1_le[1], p1_le[2], p1_le[3], p1_le[4], p1_le[5], p1_le[6], p1_le[7],
                            });
                            dbg("  [R5_DBG] inst4 claim_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                                cl_le[0], cl_le[1], cl_le[2], cl_le[3], cl_le[4], cl_le[5], cl_le[6], cl_le[7],
                            });
                        }
                        addInstanceEvalsToCombibed(F, combined_evals, polys, batch[inst], num_evals);
                    }
                }
                dbg_inst_p0[4] = combined_evals[0];
                dbg_inst_p1[4] = combined_evals[1];

                if (debug_r5) {
                    const e0 = combined_evals[0].toBytes();
                    const e1 = combined_evals[1].toBytes();
                    dbg("  [R5_DBG] after inst4: e[0]_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}] e[1]_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                        e0[0], e0[1], e0[2], e0[3], e0[4], e0[5], e0[6], e0[7],
                        e1[0], e1[1], e1[2], e1[3], e1[4], e1[5], e1[6], e1[7],
                    });
                }

                // Instance 5: IncClaimReduction - REAL prover
                {
                    const inst = 5;
                    if (remaining_rounds > num_rounds_arr[inst]) {
                        const scale = remaining_rounds - num_rounds_arr[inst] - 1;
                        var scaled = input_claims[inst];
                        for (0..scale) |_| scaled = scaled.add(scaled);
                        const contrib = batch[inst].mul(scaled);
                        for (0..num_evals) |j_eval| {
                            combined_evals[j_eval] = combined_evals[j_eval].add(contrib);
                        }
                    } else {
                        inst_active[inst] = true;
                        const polys = inc_prover.computeRoundPoly();
                        cached_inc = polys;
                        // polys = [p(0), p(1), p(2)] in Vandermonde format for degree 2
                        const p0 = polys[0];
                        const p1 = polys[1];
                        cached_inc_p1 = p1;
                        if (debug_r5) {
                            const p01_ok: u8 = if (std.mem.eql(u8, &p0.add(p1).toBytesBE(), &instance_claims[inst].toBytesBE())) 1 else 0;
                            dbg("  [R5_DBG] inst5 p(0)+p(1)=claim? {} p(0)_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}] p(1)_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                                p01_ok,
                                p0.toBytes()[0], p0.toBytes()[1], p0.toBytes()[2], p0.toBytes()[3], p0.toBytes()[4], p0.toBytes()[5], p0.toBytes()[6], p0.toBytes()[7],
                                p1.toBytes()[0], p1.toBytes()[1], p1.toBytes()[2], p1.toBytes()[3], p1.toBytes()[4], p1.toBytes()[5], p1.toBytes()[6], p1.toBytes()[7],
                            });
                        }

                        // IncClaimReduction is degree 2 in Vandermonde format [p(0), p(1), p(2)].
                        // Interpolate coefficients: a0 + a1*x + a2*x^2
                        const a0 = p0;
                        const two = F.fromU64(2);
                        const two_inv = two.inverse().?;
                        const a2_coeff = polys[2].sub(p1.add(p1)).add(p0).mul(two_inv);
                        const a1 = p1.sub(a0).sub(a2_coeff);

                        // Add evaluations at all finite points [0, 1, ..., num_evals-1]
                        // p(k) = a0 + a1*k + a2*k^2
                        for (0..num_evals) |k| {
                            const x = F.fromU64(@intCast(k));
                            const px = a0.add(x.mul(a1.add(x.mul(a2_coeff))));
                            combined_evals[k] = combined_evals[k].add(batch[inst].mul(px));
                        }
                    }
                }
                dbg_inst_p0[5] = combined_evals[0];
                dbg_inst_p1[5] = combined_evals[1];

                if (debug_r5) {
                    const e0 = combined_evals[0].toBytes();
                    const e1 = combined_evals[1].toBytes();
                    dbg("  [R5_DBG] after inst5: e[0]_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}] e[1]_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                        e0[0], e0[1], e0[2], e0[3], e0[4], e0[5], e0[6], e0[7],
                        e1[0], e1[1], e1[2], e1[3], e1[4], e1[5], e1[6], e1[7],
                    });
                    const sum = combined_evals[0].add(combined_evals[1]);
                    const sum_le = sum.toBytes();
                    const claim_le = current_batched_claim.toBytes();
                    dbg("  [R5_DBG] sum=e[0]+e[1]_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                        sum_le[0], sum_le[1], sum_le[2], sum_le[3], sum_le[4], sum_le[5], sum_le[6], sum_le[7],
                    });
                    dbg("  [R5_DBG] claim_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                        claim_le[0], claim_le[1], claim_le[2], claim_le[3], claim_le[4], claim_le[5], claim_le[6], claim_le[7],
                    });
                    // Also check each instance's expected contribution to sum
                    for (0..6) |ii| {
                        const ic_le = instance_claims[ii].toBytes();
                        const ba_le = batch[ii].toBytes();
                        dbg("  [R5_DBG] inst[{}] claim_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}] batch_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}] rounds={}\n", .{
                            ii,
                            ic_le[0], ic_le[1], ic_le[2], ic_le[3], ic_le[4], ic_le[5], ic_le[6], ic_le[7],
                            ba_le[0], ba_le[1], ba_le[2], ba_le[3], ba_le[4], ba_le[5], ba_le[6], ba_le[7],
                            num_rounds_arr[ii],
                        });
                    }
                    // Recompute expected batched claim for round 5
                    // At round 5, remaining_rounds = 13-5 = 8
                    // inst 0 (13 rounds): active, scale = 0
                    // inst 1 (8 rounds): remaining 8 > 8? no, so active, scale = 0
                    // inst 2 (8 rounds): active, scale = 0
                    // inst 3 (8 rounds): active, scale = 0
                    // inst 4 (8 rounds): active, scale = 0
                    // inst 5 (8 rounds): active, scale = 0
                    // All active! Batched claim = Σ batch[i] * instance_claims[i]
                    var expected_sum = F.zero();
                    for (0..6) |ii| {
                        expected_sum = expected_sum.add(batch[ii].mul(instance_claims[ii]));
                    }
                    const exp_le = expected_sum.toBytes();
                    dbg("  [R5_DBG] expected_batched_Σ(b*c)_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                        exp_le[0], exp_le[1], exp_le[2], exp_le[3], exp_le[4], exp_le[5], exp_le[6], exp_le[7],
                    });
                }

                // Debug: check sumcheck invariant p(0)+p(1)=claim for ALL rounds
                {
                    const p01_sum = combined_evals[0].add(combined_evals[1]);
                    const p01_match = p01_sum.eql(current_batched_claim);
                    if (!p01_match) {
                        dbg("  [S6P] R{} *** SUMCHECK INVARIANT VIOLATED *** p(0)+p(1) != claim\n", .{round});
                        const ps = p01_sum.toBytes();
                        const cb = current_batched_claim.toBytes();
                        dbg("    p(0)+p(1)_LE=[", .{});
                        for (0..32) |bi| dbg("{x:0>2}", .{ps[bi]});
                        dbg("]\n    claim_LE=[", .{});
                        for (0..32) |bi| dbg("{x:0>2}", .{cb[bi]});
                        dbg("]\n", .{});
                        // Print each instance's contribution and per-instance p(0)+p(1) check
                        for (0..6) |di| {
                            const di_claim = instance_claims[di].toBytes();
                            dbg("    inst[{}] claim_LE=[", .{di});
                            for (0..32) |bi| dbg("{x:0>2}", .{di_claim[bi]});
                            dbg("] active={} rounds={}\n", .{@as(u8, if (inst_active[di]) 1 else 0), num_rounds_arr[di]});
                        }
                        // Recompute expected batched claim from per-instance claims
                        var recomp = F.zero();
                        for (0..6) |di| {
                            if (inst_active[di]) {
                                recomp = recomp.add(batch[di].mul(instance_claims[di]));
                            } else {
                                const scale = remaining_rounds - num_rounds_arr[di] - 1;
                                var scaled = input_claims[di];
                                for (0..scale) |_| scaled = scaled.add(scaled);
                                recomp = recomp.add(batch[di].mul(scaled).add(batch[di].mul(scaled)));
                            }
                        }
                        const rc_le = recomp.toBytes();
                        dbg("    recomputed_LE=[", .{});
                        for (0..32) |bi| dbg("{x:0>2}", .{rc_le[bi]});
                        dbg("] match_claim={}\n", .{@as(u8, if (recomp.eql(current_batched_claim)) 1 else 0)});
                        // Per-instance p(0)+p(1) vs batch*claim check using cumulative deltas
                        var prev_p0 = F.zero();
                        var prev_p1 = F.zero();
                        for (0..6) |di| {
                            const inst_p0 = dbg_inst_p0[di].sub(prev_p0);
                            const inst_p1 = dbg_inst_p1[di].sub(prev_p1);
                            const inst_sum = inst_p0.add(inst_p1);
                            const expected_contrib = batch[di].mul(instance_claims[di]);
                            if (!inst_sum.eql(expected_contrib)) {
                                const is_le = inst_sum.toBytes();
                                const ex_le = expected_contrib.toBytes();
                                dbg("    *** MISMATCH inst[{}]: batch*(p0+p1)_LE=[", .{di});
                                for (0..32) |bi| dbg("{x:0>2}", .{is_le[bi]});
                                dbg("] batch*claim_LE=[", .{});
                                for (0..32) |bi| dbg("{x:0>2}", .{ex_le[bi]});
                                dbg("]\n", .{});
                            } else {
                                dbg("    inst[{}] p(0)+p(1)=claim OK\n", .{di});
                            }
                            prev_p0 = dbg_inst_p0[di];
                            prev_p1 = dbg_inst_p1[di];
                        }
                    }
                }

                // Debug: print Vandermonde evaluations for round 7
                if (round == 7) {
                    dbg("  [S6P] R7 Vandermonde evals:\n", .{});
                    for (0..num_evals) |ev_idx| {
                        const ev_le = combined_evals[ev_idx].toBytes();
                        dbg("    p({})=[", .{ev_idx});
                        for (0..32) |bi| dbg("{x:0>2}", .{ev_le[bi]});
                        dbg("]\n", .{});
                    }
                    // Verify p(0)+p(1) = current_batched_claim (hint)
                    const sum01 = combined_evals[0].add(combined_evals[1]);
                    const sum_le = sum01.toBytes();
                    const hint_le = current_batched_claim.toBytes();
                    dbg("    p(0)+p(1)=[", .{});
                    for (0..32) |bi| dbg("{x:0>2}", .{sum_le[bi]});
                    dbg("]\n    hint    =[", .{});
                    for (0..32) |bi| dbg("{x:0>2}", .{hint_le[bi]});
                    dbg("]\n    match={}\n", .{sum01.eql(current_batched_claim)});
                }

                // Compress and append to transcript (Vandermonde format)
                const compressed = try UniPoly(F).vandermondeToCompressed(self.allocator, combined_evals);
                defer self.allocator.free(compressed);

                // Debug: print compressed coefficients LE for ALL rounds
                {
                    var c_idx: usize = 0;
                    while (c_idx < compressed.len) : (c_idx += 1) {
                        const le = compressed[c_idx].toBytes();
                        dbg("  [S6P] R{} coeff[{}]=[", .{ round, c_idx });
                        for (0..32) |bi| dbg("{x:0>2}", .{le[bi]});
                        dbg("]\n", .{});
                    }
                }

                const coeffs = try self.allocator.alloc(F, num_compressed);
                for (0..num_compressed) |j| {
                    coeffs[j] = if (j < compressed.len) compressed[j] else F.zero();
                }

                try proof.compressed_polys.append(self.allocator, .{
                    .coeffs_except_linear_term = coeffs,
                    .allocator = self.allocator,
                });

                // Write diagnostic data to file for R0 - BEFORE appending to transcript
                if (round == 0) {
                    const diag_file = std.fs.cwd().createFile("/tmp/s6p_diag.bin", .{}) catch null;
                    if (diag_file) |f| {
                        defer f.close();
                        // Write: transcript state BEFORE append (32 bytes), then 5 compressed coefficients (5*32=160 bytes)
                        f.writeAll(&transcript.state) catch {};
                        for (0..num_compressed) |j| {
                            const le = coeffs[j].toBytes();
                            f.writeAll(&le) catch {};
                        }
                    }
                }

                transcript.appendScalars("sumcheck_poly", coeffs[0..num_compressed]);

                // Dump transcript state AFTER appending R0 polynomial
                if (round == 0) {
                    const diag_after = std.fs.cwd().createFile("/tmp/s6p_state_after_r0.bin", .{}) catch null;
                    if (diag_after) |fa| {
                        defer fa.close();
                        fa.writeAll(&transcript.state) catch {};
                        // Also write n_rounds as u32 LE
                        var nr_buf: [4]u8 = undefined;
                        std.mem.writeInt(u32, &nr_buf, transcript.n_rounds, .little);
                        fa.writeAll(&nr_buf) catch {};
                    }
                }

                const challenge = transcript.challengeScalar();
                challenges[round] = challenge;

                // Write R0 challenge to diagnostic file
                if (round == 0) {
                    const diag2 = std.fs.cwd().createFile("/tmp/s6p_r0_challenge.bin", .{}) catch null;
                    if (diag2) |f2| {
                        defer f2.close();
                        const ch_le = challenge.toBytes();
                        f2.writeAll(&ch_le) catch {};
                    }
                }

                // Evaluate combined polynomial at challenge (Vandermonde format)
                current_batched_claim = try UniPoly(F).evaluateVandermondeAt(self.allocator, combined_evals, challenge);

                // VERIFY: eval_from_hint should match evaluateVandermondeAt for ALL rounds
                {
                    // Simulate verifier's eval_from_hint using stored compressed coefficients
                    // hint = p(0) + p(1) = combined_evals[0] + combined_evals[1]
                    const hint_val = combined_evals[0].add(combined_evals[1]);
                    // Use the STORED coeffs (which may be padded to num_compressed=5)
                    var c1_efh = hint_val.sub(coeffs[0]).sub(coeffs[0]); // hint - 2*c0
                    for (1..num_compressed) |ci| {
                        c1_efh = c1_efh.sub(coeffs[ci]);
                    }
                    // Evaluate: c0 + c1*x + c2*x^2 + ...
                    var running_point_efh = challenge; // x
                    var running_sum_efh = coeffs[0].add(challenge.mul(c1_efh)); // c0 + x*c1
                    for (1..num_compressed) |ci| {
                        running_point_efh = running_point_efh.mul(challenge); // x^(ci+1)
                        running_sum_efh = running_sum_efh.add(coeffs[ci].mul(running_point_efh));
                    }
                    const efh_match = running_sum_efh.eql(current_batched_claim);
                    if (!efh_match) {
                        const efh_le = running_sum_efh.toBytes();
                        const vdm_le = current_batched_claim.toBytes();
                        dbg("  [S6P] R{} EVAL_MISMATCH! eval_from_hint=[", .{round});
                        for (0..32) |bi| dbg("{x:0>2}", .{efh_le[bi]});
                        dbg("]\n  [S6P] R{} EVAL_MISMATCH! vandermonde  =[", .{round});
                        for (0..32) |bi| dbg("{x:0>2}", .{vdm_le[bi]});
                        dbg("]\n", .{});
                        // Print hint, coeffs, challenge for diagnosing
                        const h_le = hint_val.toBytes();
                        dbg("  [S6P] R{} hint=[", .{round});
                        for (0..32) |bi| dbg("{x:0>2}", .{h_le[bi]});
                        dbg("]\n  [S6P] R{} c1_recovered=[", .{round});
                        const c1_le = c1_efh.toBytes();
                        for (0..32) |bi| dbg("{x:0>2}", .{c1_le[bi]});
                        dbg("]\n", .{});
                        // Also print stored coefficients count vs compressed count
                        dbg("  [S6P] R{} num_compressed={}, compressed.len={}\n", .{ round, num_compressed, compressed.len });
                    }
                    dbg("  [S6P] R{} efh_match={}\n", .{ round, @intFromBool(efh_match) });
                }

                {
                    const ch_le = challenge.toBytes();
                    const cl_le = current_batched_claim.toBytes();
                    dbg("  [S6P] R{} challenge_LE=[", .{round});
                    for (0..32) |bi| dbg("{x:0>2}", .{ch_le[bi]});
                    dbg("]\n", .{});
                    dbg("  [S6P] R{} new_claim_LE=[", .{round});
                    for (0..32) |bi| dbg("{x:0>2}", .{cl_le[bi]});
                    dbg("]\n", .{});
                }

                // Update per-instance claims from CACHED round polys and bind challenge
                // Instance 0: BytecodeReadRaf
                if (inst_active[0]) {
                    if (bytecode_prover.phase == 0) {
                        // Phase 1: degree-2 poly, p(r) = a0 + a1*r + a2*r^2
                        const bc_a0 = cached_bc_phase1_coeffs[0];
                        const bc_a1 = cached_bc_phase1_coeffs[1];
                        const bc_a2 = cached_bc_phase1_coeffs[2];
                        instance_claims[0] = bc_a0.add(challenge.mul(bc_a1.add(challenge.mul(bc_a2))));
                        {
                            const ic_le = instance_claims[0].toBytes();
                            dbg("  [S6P] R{} inst0_from_poly_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                                round, ic_le[0], ic_le[1], ic_le[2], ic_le[3], ic_le[4], ic_le[5], ic_le[6], ic_le[7],
                            });
                        }
                        bytecode_addr_challenges[bytecode_prover.addr_rounds_done] = challenge;
                        bytecode_prover.bindChallengePhase1(challenge, cached_bc_phase1_per_stage);
                        // Check invariant: instance_claims[0] == Σ gamma^s * stage_claims[s]
                        {
                            var agg_check = F.zero();
                            for (0..5) |si| {
                                agg_check = agg_check.add(bytecode_prover.gamma_powers[si].mul(bytecode_prover.stage_claims[si]));
                            }
                            const ac_le = agg_check.toBytes();
                            const ic_le2 = instance_claims[0].toBytes();
                            // Also print per-stage stage_claims after bind
                            for (0..5) |si| {
                                const scl = bytecode_prover.stage_claims[si].toBytes();
                                dbg("[INVARIANT_CHECK] R{} stage[{}]_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                                    round, si,
                                    scl[0], scl[1], scl[2], scl[3], scl[4], scl[5], scl[6], scl[7],
                                });
                            }
                            dbg("[INVARIANT_CHECK] R{} agg_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}] inst0_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}] match={}\n", .{
                                round,
                                ac_le[0], ac_le[1], ac_le[2], ac_le[3], ac_le[4], ac_le[5], ac_le[6], ac_le[7],
                                ic_le2[0], ic_le2[1], ic_le2[2], ic_le2[3], ic_le2[4], ic_le2[5], ic_le2[6], ic_le2[7],
                                @as(u8, if (agg_check.eql(instance_claims[0])) 1 else 0),
                            });
                            // Verify by computing Σ gamma^s * per_stage_p(r)_s directly
                            // where p(r)_s = old_stage_claims[s] evaluated at r
                            // We don't have old claims here, but let's re-aggregate from coefficients:
                            // Note: cached_bc_phase1_coeffs is the AGGREGATED a0,a1,a2
                            // Let's evaluate it manually:
                            const manual_eval = bc_a0.add(challenge.mul(bc_a1.add(challenge.mul(bc_a2))));
                            const me_le = manual_eval.toBytes();
                            dbg("[INVARIANT_CHECK] R{} manual_eval_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}] match_inst={}\n", .{
                                round,
                                me_le[0], me_le[1], me_le[2], me_le[3], me_le[4], me_le[5], me_le[6], me_le[7],
                                @as(u8, if (manual_eval.eql(instance_claims[0])) 1 else 0),
                            });
                        }
                        if (bytecode_prover.addr_rounds_done == bytecode_log_k) {
                            // BEFORE transition: check Σ_s gamma^s * stage_claims[s] vs instance_claims[0]
                            {
                                var agg_from_stages = F.zero();
                                for (0..5) |si| {
                                    agg_from_stages = agg_from_stages.add(bytecode_prover.gamma_powers[si].mul(bytecode_prover.stage_claims[si]));
                                }
                                const afs_le = agg_from_stages.toBytes();
                                const ic0_le = instance_claims[0].toBytes();
                                dbg("[PHASE_TRANSITION_PRE] agg_stages_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}] inst0_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}] match={}\n", .{
                                    afs_le[0], afs_le[1], afs_le[2], afs_le[3], afs_le[4], afs_le[5], afs_le[6], afs_le[7],
                                    ic0_le[0], ic0_le[1], ic0_le[2], ic0_le[3], ic0_le[4], ic0_le[5], ic0_le[6], ic0_le[7],
                                    @as(u8, if (agg_from_stages.eql(instance_claims[0])) 1 else 0),
                                });
                                // Print per-stage claims
                                for (0..5) |si| {
                                    const sc_le2 = bytecode_prover.stage_claims[si].toBytes();
                                    dbg("[PHASE_TRANSITION_PRE] stage[{}]_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                                        si, sc_le2[0], sc_le2[1], sc_le2[2], sc_le2[3], sc_le2[4], sc_le2[5], sc_le2[6], sc_le2[7],
                                    });
                                }
                            }
                            try bytecode_prover.transitionToPhase2(bytecode_addr_challenges);
                            // After transition, check Phase 2 polynomial sum
                            // Phase 2 sum = Σ_c combined[c] * Π_i ra_chunks[i][c]
                            const bc_combined = bytecode_prover.combined.?;
                            const bc_ra_chunks = bytecode_prover.ra_chunks.?;
                            const bc_T = bytecode_prover.current_len;
                            var phase2_sum = F.zero();
                            for (0..bc_T) |c| {
                                var ra_prod = F.one();
                                for (0..bytecode_prover.bytecode_d) |di| {
                                    ra_prod = ra_prod.mul(bc_ra_chunks[di][c]);
                                }
                                phase2_sum = phase2_sum.add(bc_combined[c].mul(ra_prod));
                            }
                            const ic_old_le = instance_claims[0].toBytes();
                            const p2_le = phase2_sum.toBytes();
                            dbg("[PHASE_TRANSITION] inst0 claim_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}] phase2_sum_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}] match={}\n", .{
                                ic_old_le[0], ic_old_le[1], ic_old_le[2], ic_old_le[3], ic_old_le[4], ic_old_le[5], ic_old_le[6], ic_old_le[7],
                                p2_le[0], p2_le[1], p2_le[2], p2_le[3], p2_le[4], p2_le[5], p2_le[6], p2_le[7],
                                @as(u8, if (instance_claims[0].eql(phase2_sum)) 1 else 0),
                            });
                            // DO NOT overwrite instance_claims[0] - keep the value from Phase 1 evaluation
                            // The phase2_sum is just the sum over Phase 2 arrays, which SHOULD match but
                            // if it doesn't, the Phase 1 claim is what's in the transcript.
                            // instance_claims[0] = phase2_sum;
                        }
                    } else {
                        // Phase 2: evaluate from cached evals using Lagrange interpolation
                        instance_claims[0] = evaluatePolyFromEvals(F, cached_bc_phase2.?, challenge);
                        self.allocator.free(cached_bc_phase2.?);
                        cached_bc_phase2 = null;
                        bytecode_prover.bindChallengePhase2(challenge);

                        // BRUTE FORCE CHECK: recompute Σ_c combined[c] * Π_i ra[i][c] after bind
                        {
                            const bc_combined_dbg = bytecode_prover.combined.?;
                            const bc_ra_dbg = bytecode_prover.ra_chunks.?;
                            const bc_T_dbg = bytecode_prover.current_len;
                            var bf_sum = F.zero();
                            for (0..bc_T_dbg) |c_dbg| {
                                var ra_prod_dbg = F.one();
                                for (0..bytecode_prover.bytecode_d) |di_dbg| {
                                    ra_prod_dbg = ra_prod_dbg.mul(bc_ra_dbg[di_dbg][c_dbg]);
                                }
                                bf_sum = bf_sum.add(bc_combined_dbg[c_dbg].mul(ra_prod_dbg));
                            }
                            const bf_be = bf_sum.toBytesBE();
                            const ic_be2 = instance_claims[0].toBytesBE();
                            dbg("[BF_CHECK] R{} Phase2 T={} inst0_LE=[", .{ round, bc_T_dbg });
                            for (0..8) |bi| dbg("{x:0>2}", .{ic_be2[31 - bi]});
                            dbg("] bf_sum_LE=[", .{});
                            for (0..8) |bi| dbg("{x:0>2}", .{bf_be[31 - bi]});
                            dbg("] match={}\n", .{@as(u8, if (bf_sum.eql(instance_claims[0])) 1 else 0)});
                        }
                    }
                }

                // Instance 1: Booleanity (real prover)
                if (inst_active[1]) {
                    if (cached_booleanity) |polys| {
                        // Evaluate degree-3 poly at challenge from Vandermonde [p(0), p(1), p(2), p(3)]
                        const evals_arr = [4]F{ polys[0], polys[1], polys[2], polys[3] };
                        instance_claims[1] = evaluateDeg3FromEvals(F, evals_arr, challenge);
                        self.allocator.free(polys);
                        cached_booleanity = null;
                    }
                    try booleanity_prover.bindChallenge(challenge);
                    // Debug: print claim after Phase 1→2 transition
                    if (booleanity_prover.round == booleanity_prover.log_k_chunk) {
                        const ic1_be = instance_claims[1].toBytesBE();
                        dbg("[BOOL_TRANSITION] inst_claim[1] after Ph1 LE=[{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}]\n", .{
                            ic1_be[31], ic1_be[30], ic1_be[29], ic1_be[28], ic1_be[27], ic1_be[26], ic1_be[25], ic1_be[24],
                        });
                    }
                }

                // Instance 2: HammingBooleanity
                if (inst_active[2]) {
                    instance_claims[2] = evaluateDeg3FromEvals(F, cached_hamming, challenge);
                    hamming_prover.bindChallenge(challenge);
                }

                // Instance 3: RamRaVirtual
                if (inst_active[3]) {
                    instance_claims[3] = evaluatePolyFromEvals(F, cached_ram_ra.?, challenge);
                    self.allocator.free(cached_ram_ra.?);
                    cached_ram_ra = null;
                    ram_ra_prover.bindChallenge(challenge);
                }

                // Instance 4: LookupsRaVirtual
                if (inst_active[4]) {
                    instance_claims[4] = evaluatePolyFromEvals(F, cached_lookups_ra.?, challenge);
                    self.allocator.free(cached_lookups_ra.?);
                    cached_lookups_ra = null;
                    lookups_ra_prover.bindChallenge(challenge);
                }

                // Instance 5: IncClaimReduction
                if (inst_active[5]) {
                    const p0 = cached_inc[0];
                    const p1_val = cached_inc[1]; // Vandermonde format: polys[1] = p(1) directly
                    const p2_val = cached_inc[2]; // Vandermonde format: polys[2] = p(2) directly
                    // Interpolate coefficients: p(x) = a0 + a1*x + a2*x^2
                    const a0 = p0;
                    const inc_two = F.fromU64(2);
                    const inc_two_inv = inc_two.inverse().?;
                    const a2 = p2_val.sub(p1_val.add(p1_val)).add(p0).mul(inc_two_inv);
                    const a1 = p1_val.sub(a0).sub(a2);
                    instance_claims[5] = a0.add(challenge.mul(a1.add(challenge.mul(a2))));
                    // BRUTE FORCE CHECK BEFORE BIND: recompute p(0), p(1) from arrays
                    if (round == 5) {
                        var bf_p0 = F.zero();
                        var bf_p1 = F.zero();
                        const half = inc_prover.current_len / 2;
                        for (0..half) |bj| {
                            bf_p0 = bf_p0.add(inc_prover.ram_inc[2 * bj].mul(inc_prover.eq_ram[2 * bj]));
                            bf_p0 = bf_p0.add(inc_prover.gamma_sqr.mul(inc_prover.rd_inc[2 * bj].mul(inc_prover.eq_rd[2 * bj])));
                            bf_p1 = bf_p1.add(inc_prover.ram_inc[2 * bj + 1].mul(inc_prover.eq_ram[2 * bj + 1]));
                            bf_p1 = bf_p1.add(inc_prover.gamma_sqr.mul(inc_prover.rd_inc[2 * bj + 1].mul(inc_prover.eq_rd[2 * bj + 1])));
                        }
                        const bf_claim = bf_p0.add(bf_p1);
                        dbg("[INC_R5_CHECK] bf_p0 match cached_inc[0]? {}\n", .{
                            @as(u8, if (std.mem.eql(u8, &bf_p0.toBytesBE(), &cached_inc[0].toBytesBE())) 1 else 0),
                        });
                        dbg("[INC_R5_CHECK] bf_claim match instance_claims[5]? {}\n", .{
                            @as(u8, if (std.mem.eql(u8, &bf_claim.toBytesBE(), &instance_claims[5].toBytesBE())) 1 else 0),
                        });
                        // Print all values
                        const p0_be = cached_inc[0].toBytesBE();
                        const p2_be = cached_inc[1].toBytesBE();
                        const pinf_be = cached_inc[2].toBytesBE();
                        const p1v_be = cached_inc_p1.toBytesBE();
                        const bfp0_be = bf_p0.toBytesBE();
                        const bfp1_be = bf_p1.toBytesBE();
                        const bfc_be = bf_claim.toBytesBE();
                        const ic5_be = instance_claims[5].toBytesBE();
                        dbg("  cached p(0)_LE=[", .{});
                        for (0..32) |bi| dbg("{x:0>2}", .{p0_be[31 - bi]});
                        dbg("]\n  cached p(2)_LE=[", .{});
                        for (0..32) |bi| dbg("{x:0>2}", .{p2_be[31 - bi]});
                        dbg("]\n  cached p(inf)_LE=[", .{});
                        for (0..32) |bi| dbg("{x:0>2}", .{pinf_be[31 - bi]});
                        dbg("]\n  cached p(1)_LE=[", .{});
                        for (0..32) |bi| dbg("{x:0>2}", .{p1v_be[31 - bi]});
                        dbg("]\n  bf p(0)_LE=[", .{});
                        for (0..32) |bi| dbg("{x:0>2}", .{bfp0_be[31 - bi]});
                        dbg("]\n  bf p(1)_LE=[", .{});
                        for (0..32) |bi| dbg("{x:0>2}", .{bfp1_be[31 - bi]});
                        dbg("]\n  bf claim_LE=[", .{});
                        for (0..32) |bi| dbg("{x:0>2}", .{bfc_be[31 - bi]});
                        dbg("]\n  instance[5]_LE=[", .{});
                        for (0..32) |bi| dbg("{x:0>2}", .{ic5_be[31 - bi]});
                        dbg("]\n", .{});
                    }

                    inc_prover.bindChallenge(challenge);

                    // BRUTE FORCE CHECK: recompute from arrays and compare
                    {
                        var brute_sum = F.zero();
                        for (0..inc_prover.current_len) |bj| {
                            brute_sum = brute_sum.add(inc_prover.ram_inc[bj].mul(inc_prover.eq_ram[bj]));
                            brute_sum = brute_sum.add(inc_prover.gamma_sqr.mul(inc_prover.rd_inc[bj].mul(inc_prover.eq_rd[bj])));
                        }
                        const brute_match = std.mem.eql(u8, &brute_sum.toBytesBE(), &instance_claims[5].toBytesBE());
                        if (!brute_match) {
                            dbg("[INC_BRUTE] R{} MISMATCH! instance[5] != brute_sum\n", .{round});
                            const bs_be = brute_sum.toBytesBE();
                            const ic_be = instance_claims[5].toBytesBE();
                            dbg("  brute_LE=[", .{});
                            for (0..32) |bi| dbg("{x:0>2}", .{bs_be[31 - bi]});
                            dbg("]\n  inst5_LE=[", .{});
                            for (0..32) |bi| dbg("{x:0>2}", .{ic_be[31 - bi]});
                            dbg("]\n", .{});
                        }
                    }
                }

                // NOTE: Instance claims for inactive instances are NOT halved here.
                // In Zolt, instance_claims starts at the UNSCALED input_claims (not 2^offset-scaled),
                // and the inactive round contributions are computed directly from input_claims with
                // the correct power-of-2 scaling. When an instance first becomes active,
                // instance_claims[i] = input_claims[i] = the correct unscaled claim.
            }

            // Debug: print final instance claims after sumcheck (ALWAYS ON)
            {
                dbg("\n[S6P] Final instance claims after sumcheck:\n", .{});
                for (0..6) |i| {
                    const be = instance_claims[i].toBytesBE();
                    dbg("  instance[{d}] final_claim_LE=[", .{i});
                    for (0..32) |bi| dbg("{x:0>2}", .{be[31 - bi]});
                    dbg("]\n", .{});
                }
                const bb = current_batched_claim.toBytesBE();
                dbg("  batched_output_claim_LE=[", .{});
                for (0..32) |bi| dbg("{x:0>2}", .{bb[31 - bi]});
                dbg("]\n", .{});
                dbg("[S6P] All challenges (LE first 8 bytes):\n", .{});
                for (0..max_num_rounds) |i| {
                    const be = challenges[i].toBytesBE();
                    dbg("  ch[{d}] = [{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                        i, be[31], be[30], be[29], be[28], be[27], be[26], be[25], be[24],
                    });
                }
            }

            // ====================================================================
            // Extract opening claims from all real provers
            // ====================================================================

            const inc_opening = inc_prover.openingClaims();
            const ram_inc_claim = inc_opening.ram_inc;
            const rd_inc_claim = inc_opening.rd_inc;
            // Debug: verify inc prover internal consistency
            {
                const eq_r = inc_prover.eq_ram[0];
                const eq_d = inc_prover.eq_rd[0];
                const recomp = ram_inc_claim.mul(eq_r).add(inc_gamma2.mul(rd_inc_claim.mul(eq_d)));
                const er_be = eq_r.toBytesBE();
                const ed_be = eq_d.toBytesBE();
                const rc_be = recomp.toBytesBE();
                dbg("[INC_DEBUG] eq_ram[0]_LE=[", .{});
                for (0..32) |bi| dbg("{x:0>2}", .{er_be[31 - bi]});
                dbg("]\n  eq_rd[0]_LE=[", .{});
                for (0..32) |bi| dbg("{x:0>2}", .{ed_be[31 - bi]});
                dbg("]\n  recomp_LE=[", .{});
                for (0..32) |bi| dbg("{x:0>2}", .{rc_be[31 - bi]});
                dbg("]\n  instance[5]_LE=[", .{});
                const i5_be = instance_claims[5].toBytesBE();
                for (0..32) |bi| dbg("{x:0>2}", .{i5_be[31 - bi]});
                dbg("]\n", .{});
            }

            const hamming_weight_claim = hamming_prover.openingClaim();

            const bytecode_ra_claims = try bytecode_prover.getOpeningClaims(self.allocator);
            // Debug: bytecode RA claims (ALWAYS ON for debugging)
            {
                dbg("[S6P] Bytecode RA claims (d={d}):\n", .{bytecode_d});
                for (0..bytecode_d) |i| {
                    const be = bytecode_ra_claims[i].toBytesBE();
                    dbg("  ra[{d}]_LE=[", .{i});
                    for (0..32) |bi| dbg("{x:0>2}", .{be[31 - bi]});
                    dbg("]\n", .{});
                }
                // Print combined[0] (the "val" part after all Phase 2 bindings)
                const comb0 = bytecode_prover.combined.?[0];
                const comb0_be = comb0.toBytesBE();
                dbg("  combined[0]_LE=[", .{});
                for (0..32) |bi| dbg("{x:0>2}", .{comb0_be[31 - bi]});
                dbg("]\n", .{});
                // Compute val_from_prover = combined[0] * Π ra[i]
                var val_ra_prod = comb0;
                for (0..bytecode_d) |i| {
                    val_ra_prod = val_ra_prod.mul(bytecode_ra_claims[i]);
                }
                const vrp_be = val_ra_prod.toBytesBE();
                dbg("  combined[0]*Π_ra_LE=[", .{});
                for (0..32) |bi| dbg("{x:0>2}", .{vrp_be[31 - bi]});
                dbg("]\n", .{});
                // Compare with instance_claims[0]
                const ic0_be = instance_claims[0].toBytesBE();
                dbg("  instance_claims[0]_LE=[", .{});
                for (0..32) |bi| dbg("{x:0>2}", .{ic0_be[31 - bi]});
                dbg("] match_val_ra={}\n", .{@as(u8, if (val_ra_prod.eql(instance_claims[0])) 1 else 0)});

                // === PER-STAGE DECOMPOSITION ===
                // Recompute combined[0] = Σ_s bound_vals[s] * eq_mle(r_cycle_s, r_cycle_prime)
                // r_cycle_prime = reversed Phase 2 challenges (matching Jolt's normalize_opening_point)
                const cycle_start = bytecode_log_k;
                var r_cycle_prime = try self.allocator.alloc(F, n_cycle_vars);
                defer self.allocator.free(r_cycle_prime);
                for (0..n_cycle_vars) |ci| {
                    r_cycle_prime[ci] = challenges[cycle_start + n_cycle_vars - 1 - ci];
                }
                // Print r_cycle_prime
                dbg("[DECOMP] r_cycle_prime (reversed cycle challenges, BE):\n", .{});
                for (0..@min(4, n_cycle_vars)) |ci| {
                    const rcp_be = r_cycle_prime[ci].toBytesBE();
                    dbg("  r_cycle_prime[{}]_LE=[", .{ci});
                    for (0..8) |bi| dbg("{x:0>2}", .{rcp_be[31 - bi]});
                    dbg("]\n", .{});
                }

                var decomp_sum = F.zero();
                for (0..5) |s| {
                    // Compute eq_mle(r_cycle_s, r_cycle_prime) = Π_i (r_s[i]*r_p[i] + (1-r_s[i])(1-r_p[i]))
                    // Both r_cycle_s and r_cycle_prime are in BE order
                    var eq_mle = F.one();
                    const r_s = bytecode_prover.stage_r_cycles[s];
                    for (0..n_cycle_vars) |ci| {
                        const a = r_s[ci];
                        const b = r_cycle_prime[ci];
                        // eq term: a*b + (1-a)*(1-b) = 1 - a - b + 2*a*b
                        const ab = a.mul(b);
                        const term = F.one().sub(a).sub(b).add(ab).add(ab);
                        eq_mle = eq_mle.mul(term);
                    }

                    const bv = bytecode_prover.bound_vals_stored[s];
                    const stage_contrib = bv.mul(eq_mle);
                    decomp_sum = decomp_sum.add(stage_contrib);

                    const bv_be = bv.toBytesBE();
                    const eq_be = eq_mle.toBytesBE();
                    const sc_be = stage_contrib.toBytesBE();
                    dbg("[DECOMP] stage[{}]: bound_val_LE=[", .{s});
                    for (0..8) |bi| dbg("{x:0>2}", .{bv_be[31 - bi]});
                    dbg("] eq_mle_LE=[", .{});
                    for (0..8) |bi| dbg("{x:0>2}", .{eq_be[31 - bi]});
                    dbg("] contrib_LE=[", .{});
                    for (0..8) |bi| dbg("{x:0>2}", .{sc_be[31 - bi]});
                    dbg("]\n", .{});
                }
                const ds_be = decomp_sum.toBytesBE();
                dbg("[DECOMP] val_sum_LE=[", .{});
                for (0..8) |bi| dbg("{x:0>2}", .{ds_be[31 - bi]});
                dbg("] match_combined={}\n", .{@as(u8, if (decomp_sum.eql(comb0)) 1 else 0)});

                // Also print val_with_raf bound values (without gamma)
                for (0..5) |s| {
                    const vwr = bytecode_prover.bound_vals_stored[s];
                    const gp = bytecode_prover.gamma_powers[s];
                    // val_with_raf[s][0] = bound_vals[s] / gamma[s]
                    // Print bound_val directly (it already includes gamma)
                    const vwr_be = vwr.toBytesBE();
                    const gp_be = gp.toBytesBE();
                    dbg("[DECOMP] stage[{}]: gamma_LE=[", .{s});
                    for (0..8) |bi| dbg("{x:0>2}", .{gp_be[31 - bi]});
                    dbg("] gamma*val_LE=[", .{});
                    for (0..8) |bi| dbg("{x:0>2}", .{vwr_be[31 - bi]});
                    dbg("]\n", .{});
                }
            }

            const ram_ra_virtual_claims = try ram_ra_prover.getOpeningClaims(self.allocator);

            const instruction_ra_virtual_claims = try lookups_ra_prover.getOpeningClaims(self.allocator, lookups_ra_gamma_powers);

            // Get booleanity claims directly from the prover's final H state.
            // After all Phase 2 rounds, H[i][0] = ra_i(ρ_addr, ρ_cycle).
            const booleanity_ra_claims = try booleanity_prover.getBooleanityClaims(self.allocator);
            {
                const total_booleanity_polys = instruction_d + bytecode_d + ram_d;
                dbg("[STAGE6] Booleanity claims from H final state:\n", .{});
                for (0..@min(5, total_booleanity_polys)) |i| {
                    const brc_be = booleanity_ra_claims[i].toBytesBE();
                    dbg("  bool_claim[{}]_LE=[", .{i});
                    for (0..8) |bi| dbg("{x:0>2}", .{brc_be[31 - bi]});
                    dbg("]\n", .{});
                }
            }

            // Debug: compute what the verifier would compute for Instance 1 (Booleanity)
            // expected = eq(challenges, combined_r) * Σ gamma^{2i} * (ra_i^2 - ra_i)
            // combined_r = r_address.reversed ++ r_cycle.reversed
            // In Jolt: r_address reversed means the original r_address (from params) reversed.
            // The booleanity params store r_address in LE format. "reversed" in Jolt means
            // going from LE to reversed-LE. But actually Jolt stores r_address and r_cycle in a
            // specific order from BooleanitySumcheckParams::new, and then reverses them.
            {
                const total_booleanity_polys = instruction_d + bytecode_d + ram_d;
                // Jolt's BooleanitySumcheckParams stores r_address and r_cycle from Stage 5.
                // r_address = last log_k_chunk challenges from the InstructionReadRaf address.
                // r_cycle = cycle challenges from InstructionReadRaf.
                // The verifier reverses both: combined_r = rev(r_address) ++ rev(r_cycle).
                //
                // In our code:
                // r_address_bool_le = [ch[log_k-1], ch[log_k-2], ..., ch[0]] (from stage5 MSB-first)
                // But the Jolt params store them in a specific order based on Stage 5's binding.
                // Jolt's BooleanitySumcheckParams::new extracts r_address from the accumulator
                // which stores them in the binding order from Stage 5 InstructionReadRaf.
                //
                // For now, let me compute the expected claim using the data I have:
                // The sumcheck challenges for booleanity rounds are challenges[0..log_k+n_cycle].
                // Booleanity uses rounds 0..log_k for address, log_k..log_k+n_cycle for cycle.
                //
                // The actual output_claim from the sumcheck should be:
                //   eq_r_r * eq_cycle_final * Σ gamma^{2i} * (H[i][0]^2 - H[i][0])
                // where eq_cycle_final is what eq_cycle[0] becomes after all Phase 2 bindings.
                //
                // Let me just compute Σ gamma^{2i} * (ra_i^2 - ra_i) and the eq parts.
                var sum_gamma_ra = F.zero();
                for (0..total_booleanity_polys) |i| {
                    const ra = booleanity_ra_claims[i];
                    sum_gamma_ra = sum_gamma_ra.add(booleanity_prover.gamma_powers_sq[i].mul(ra.mul(ra).sub(ra)));
                }
                // Also, get the actual eq values from the prover
                const bp_eq_r_r = booleanity_prover.eq_r_r;
                const bp_eq_cycle_final = booleanity_prover.eq_cycle[0];
                const actual_output = bp_eq_r_r.mul(bp_eq_cycle_final).mul(sum_gamma_ra);

                const sg_be = sum_gamma_ra.toBytesBE();
                const err_be = bp_eq_r_r.toBytesBE();
                const ecf_be = bp_eq_cycle_final.toBytesBE();
                const ao_be = actual_output.toBytesBE();
                dbg("[BOOL_VERIFY] sum_gamma_ra_LE=[", .{});
                for (0..8) |bi| dbg("{x:0>2}", .{sg_be[31 - bi]});
                dbg("]\n", .{});
                dbg("[BOOL_VERIFY] eq_r_r_LE=[", .{});
                for (0..8) |bi| dbg("{x:0>2}", .{err_be[31 - bi]});
                dbg("]\n", .{});
                dbg("[BOOL_VERIFY] eq_cycle_final_LE=[", .{});
                for (0..8) |bi| dbg("{x:0>2}", .{ecf_be[31 - bi]});
                dbg("]\n", .{});
                dbg("[BOOL_VERIFY] actual_output_LE=[", .{});
                for (0..8) |bi| dbg("{x:0>2}", .{ao_be[31 - bi]});
                dbg("]\n", .{});

                // Compare with instance_claims[1] (the sumcheck output claim for booleanity)
                const ic1_be = instance_claims[1].toBytesBE();
                dbg("[BOOL_VERIFY] instance_claims[1]_LE=[", .{});
                for (0..8) |bi| dbg("{x:0>2}", .{ic1_be[31 - bi]});
                dbg("]\n", .{});
                dbg("[BOOL_VERIFY] match={}\n", .{@intFromBool(actual_output.eql(instance_claims[1]))});

                // Now compute eq(challenges, combined_r) directly, the way the verifier does.
                // combined_r = rev(r_address_LE) ++ rev(r_cycle_LE)
                // r_address_LE (in Jolt) = last log_k_chunk elements of Stage5 addr reversed to LE
                // In our code: the ORIGINAL r_address_bool_le (before reversal in init) is the LE version.
                // After init() reversed it, booleanity_prover.r_address_le[m] = MSB at m=0.
                // To get Jolt's LE r_address, we need to reverse it back.
                // Then rev(r_address_LE) = booleanity_prover.r_address_le (as-is, since it was reversed to BE)
                //
                // combined_r_addr[m] = r_address_LE[log_k-1-m] = booleanity_prover.r_address_le[m]
                // combined_r_cycle[m] = r_cycle_LE[n_cycle-1-m]
                //
                // r_cycle_LE = lookups_ra_r_cycle (the original, before computeEqTable)
                // combined_r_cycle[m] = lookups_ra_r_cycle[n_cycle-1-m]
                //
                // eq(ch[m], combined_r[m]) for m < log_k:
                //   = eq(ch[m], booleanity_prover.r_address_le[m])
                // eq(ch[log_k+m], combined_r[log_k+m]) for m < n_cycle:
                //   = eq(ch[log_k+m], lookups_ra_r_cycle[n_cycle-1-m])
                {
                    const bool_start_round = max_num_rounds - num_rounds_arr[1];
                    dbg("[BOOL_VERIFY] bool_start_round={}, log_k={}, n_cycle={}\n", .{
                        bool_start_round, log_k_chunk, n_cycle_vars,
                    });

                    // Print ALL eq factors matching Jolt's format
                    // Jolt: combined_r = rev(r_address_LE) ++ rev(r_cycle_LE)
                    // Zolt: r_address_le[m] = MSB at 0 (reversed in init) = rev(r_address_LE)[m]
                    // Zolt: combined_r_cycle[m] = r_cycle_LE[n_cycle-1-m] = lookups_ra_r_cycle[n_cycle-1-m]
                    var eq_direct = F.one();
                    for (0..log_k_chunk) |m| {
                        const ch_val = challenges[bool_start_round + m];
                        const w_val = booleanity_prover.r_address_le[m];
                        const prod = ch_val.mul(w_val);
                        const eq_factor = F.one().sub(ch_val).sub(w_val).add(prod.add(prod));
                        eq_direct = eq_direct.mul(eq_factor);

                        const ch_be = ch_val.toBytesBE();
                        const w_be = w_val.toBytesBE();
                        const ef_be = eq_factor.toBytesBE();
                        dbg("[BOOL_EQ_ZOLT] idx={} sc=[", .{m});
                        for (0..8) |bi| dbg("{x:0>2}", .{ch_be[31 - bi]});
                        dbg("] cr=[", .{});
                        for (0..8) |bi| dbg("{x:0>2}", .{w_be[31 - bi]});
                        dbg("] eq_i=[", .{});
                        for (0..8) |bi| dbg("{x:0>2}", .{ef_be[31 - bi]});
                        dbg("]\n", .{});
                    }
                    for (0..n_cycle_vars) |m| {
                        const ch_val = challenges[bool_start_round + log_k_chunk + m];
                        // Jolt: combined_r_cycle[m] = rev(r_cycle_LE)[m] = r_cycle_LE[n-1-m]
                        // Since lookups_ra_r_cycle is BE (MSB at 0), and Jolt r_cycle_LE[n-1-m] = lookups[m]
                        const w_val = lookups_ra_r_cycle[m]; // direct index, no reversal
                        const prod = ch_val.mul(w_val);
                        const eq_factor = F.one().sub(ch_val).sub(w_val).add(prod.add(prod));
                        eq_direct = eq_direct.mul(eq_factor);

                        const ch_be = ch_val.toBytesBE();
                        const w_be = w_val.toBytesBE();
                        const ef_be = eq_factor.toBytesBE();
                        dbg("[BOOL_EQ_ZOLT] idx={} sc=[", .{log_k_chunk + m});
                        for (0..8) |bi| dbg("{x:0>2}", .{ch_be[31 - bi]});
                        dbg("] cr=[", .{});
                        for (0..8) |bi| dbg("{x:0>2}", .{w_be[31 - bi]});
                        dbg("] eq_i=[", .{});
                        for (0..8) |bi| dbg("{x:0>2}", .{ef_be[31 - bi]});
                        dbg("]\n", .{});
                    }

                    const eq_from_prover = bp_eq_r_r.mul(bp_eq_cycle_final);
                    const ed_be = eq_direct.toBytesBE();
                    const ep_be = eq_from_prover.toBytesBE();
                    dbg("[BOOL_VERIFY] eq_direct_LE=[", .{});
                    for (0..8) |bi| dbg("{x:0>2}", .{ed_be[31 - bi]});
                    dbg("]\n", .{});
                    dbg("[BOOL_VERIFY] eq_from_prover_LE=[", .{});
                    for (0..8) |bi| dbg("{x:0>2}", .{ep_be[31 - bi]});
                    dbg("]\n", .{});
                    dbg("[BOOL_VERIFY] eq_match={}\n", .{@intFromBool(eq_direct.eql(eq_from_prover))});
                }
            }

            dbg("[STAGE6] Opening claims (full LE hex):\n", .{});
            {
                const be = ram_inc_claim.toBytesBE();
                dbg("  ram_inc_LE=[", .{});
                for (0..32) |bi| dbg("{x:0>2}", .{be[31 - bi]});
                dbg("]\n", .{});
            }
            {
                const be = rd_inc_claim.toBytesBE();
                dbg("  rd_inc_LE=[", .{});
                for (0..32) |bi| dbg("{x:0>2}", .{be[31 - bi]});
                dbg("]\n", .{});
            }
            {
                const be = hamming_weight_claim.toBytesBE();
                dbg("  hamming_weight_LE=[", .{});
                for (0..32) |bi| dbg("{x:0>2}", .{be[31 - bi]});
                dbg("]\n", .{});
            }
            for (0..bytecode_d) |i| {
                const be = bytecode_ra_claims[i].toBytesBE();
                dbg("  bytecode_ra[{d}]_LE=[", .{i});
                for (0..32) |bi| dbg("{x:0>2}", .{be[31 - bi]});
                dbg("]\n", .{});
            }
            {
                const be = ram_ra_virtual_claims[0].toBytesBE();
                dbg("  ram_ra_virtual[0]_LE=[", .{});
                for (0..32) |bi| dbg("{x:0>2}", .{be[31 - bi]});
                dbg("]\n", .{});
            }
            {
                const be = instruction_ra_virtual_claims[0].toBytesBE();
                dbg("  instruction_ra_virtual[0]_LE=[", .{});
                for (0..32) |bi| dbg("{x:0>2}", .{be[31 - bi]});
                dbg("]\n", .{});
            }
            for (0..3) |i| {
                const be = booleanity_ra_claims[i].toBytesBE();
                dbg("  booleanity_ra[{d}]_LE=[", .{i});
                for (0..32) |bi| dbg("{x:0>2}", .{be[31 - bi]});
                dbg("]\n", .{});
            }

            // Consistency check: instance_claims[0] should equal val * Π ra[i]
            // where val = bytecode_prover.combined.?[0] (after all binding)
            {
                const bc_combined_val = bytecode_prover.combined.?[0];
                var bc_ra_prod = F.one();
                for (bytecode_ra_claims) |c| bc_ra_prod = bc_ra_prod.mul(c);
                const bc_recomputed = bc_combined_val.mul(bc_ra_prod);
                dbg("[STAGE6] Consistency check Instance 0:\n", .{});
                // Print combined[0] as LE hex for comparison with Jolt's "val (sum)"
                const cval_be = bc_combined_val.toBytesBE();
                dbg("  combined[0]_LE=[", .{});
                for (0..32) |bi| dbg("{x:0>2}", .{cval_be[31 - bi]});
                dbg("]\n", .{});
                // Print ra claims
                for (0..bytecode_d) |i| {
                    const ra_be = bytecode_ra_claims[i].toBytesBE();
                    dbg("  ra[{}]_LE=[", .{i});
                    for (0..32) |bi| dbg("{x:0>2}", .{ra_be[31 - bi]});
                    dbg("]\n", .{});
                }
                dbg("  recomputed_LE=[", .{});
                const rc_be = bc_recomputed.toBytesBE();
                for (0..32) |bi| dbg("{x:0>2}", .{rc_be[31 - bi]});
                dbg("]\n", .{});
                dbg("  instance[0]_LE=[", .{});
                const ic_be = instance_claims[0].toBytesBE();
                for (0..32) |bi| dbg("{x:0>2}", .{ic_be[31 - bi]});
                dbg("]\n", .{});
                dbg("  match = {}\n", .{@as(u8, if (std.mem.eql(u8, &bc_recomputed.toBytesBE(), &instance_claims[0].toBytesBE())) 1 else 0)});
            }

            // Consistency check Instance 5 (IncClaimReduction):
            // expected = ram_inc * eq_ram_combined(rho) + gamma^2 * rd_inc * eq_rd_combined(rho)
            // where rho = reversed sumcheck challenges (opening point in BE)
            {
                // Build opening point: reverse challenges for LE->BE
                var opening_point = try self.allocator.alloc(F, n_cycle_vars);
                defer self.allocator.free(opening_point);
                // Instance 5 has n_cycle_vars rounds; offset = max_num_rounds - n_cycle_vars
                const inc_offset = max_num_rounds - n_cycle_vars;
                for (0..n_cycle_vars) |i| {
                    opening_point[n_cycle_vars - 1 - i] = challenges[inc_offset + i];
                }

                // Compute eq evaluations at opening_point vs each r_cycle
                // eq(a, b) = prod_i (a[i]*b[i] + (1-a[i])*(1-b[i]))
                const computeEqEval = struct {
                    fn eval(a: []const F, b: []const F) F {
                        var result = F.one();
                        for (0..a.len) |i| {
                            const prod = a[i].mul(b[i]);
                            const sum = a[i].add(b[i]);
                            result = result.mul(prod.add(prod).add(F.one()).sub(sum));
                        }
                        return result;
                    }
                }.eval;

                const eq_r2 = computeEqEval(opening_point, r_cycle_inc_ram_rwc);
                const eq_r4 = computeEqEval(opening_point, r_cycle_inc_ram_val);
                const eq_s4 = computeEqEval(opening_point, r_cycle_bc4_regs_rwc);
                const eq_s5 = computeEqEval(opening_point, r_cycle_bc5_regs_val);

                const eq_ram_combined = eq_r2.add(inc_gamma.mul(eq_r4));
                const eq_rd_combined = eq_s4.add(inc_gamma.mul(eq_s5));

                const expected_inc = ram_inc_claim.mul(eq_ram_combined).add(inc_gamma2.mul(rd_inc_claim.mul(eq_rd_combined)));

                dbg("[STAGE6] Inc consistency check:\n", .{});
                dbg("  ram_inc_claim_LE=[", .{});
                const ric_be = ram_inc_claim.toBytesBE();
                for (0..32) |bi| dbg("{x:0>2}", .{ric_be[31 - bi]});
                dbg("]\n", .{});
                dbg("  rd_inc_claim_LE=[", .{});
                const rdc_be = rd_inc_claim.toBytesBE();
                for (0..32) |bi| dbg("{x:0>2}", .{rdc_be[31 - bi]});
                dbg("]\n", .{});
                dbg("  eq_r2_LE=[", .{});
                const er2 = eq_r2.toBytesBE();
                for (0..32) |bi| dbg("{x:0>2}", .{er2[31 - bi]});
                dbg("]\n", .{});
                dbg("  eq_r4_LE=[", .{});
                const er4 = eq_r4.toBytesBE();
                for (0..32) |bi| dbg("{x:0>2}", .{er4[31 - bi]});
                dbg("]\n", .{});
                dbg("  eq_s4_LE=[", .{});
                const es4 = eq_s4.toBytesBE();
                for (0..32) |bi| dbg("{x:0>2}", .{es4[31 - bi]});
                dbg("]\n", .{});
                dbg("  eq_s5_LE=[", .{});
                const es5 = eq_s5.toBytesBE();
                for (0..32) |bi| dbg("{x:0>2}", .{es5[31 - bi]});
                dbg("]\n", .{});
                dbg("  expected_inc_LE=[", .{});
                const eibc = expected_inc.toBytesBE();
                for (0..32) |bi| dbg("{x:0>2}", .{eibc[31 - bi]});
                dbg("]\n", .{});
                dbg("  instance[5]_LE=[", .{});
                const i5_be = instance_claims[5].toBytesBE();
                for (0..32) |bi| dbg("{x:0>2}", .{i5_be[31 - bi]});
                dbg("]\n", .{});
                dbg("  match = {}\n", .{@as(u8, if (std.mem.eql(u8, &expected_inc.toBytesBE(), &instance_claims[5].toBytesBE())) 1 else 0)});

                // Also print the r_cycle values themselves
                dbg("  r_cycle_inc_ram_rwc[0]_LE=[", .{});
                const rr0 = r_cycle_inc_ram_rwc[0].toBytesBE();
                for (0..32) |bi| dbg("{x:0>2}", .{rr0[31 - bi]});
                dbg("]\n", .{});
                dbg("  r_cycle_inc_ram_val[0]_LE=[", .{});
                const rv0 = r_cycle_inc_ram_val[0].toBytesBE();
                for (0..32) |bi| dbg("{x:0>2}", .{rv0[31 - bi]});
                dbg("]\n", .{});
                dbg("  r_cycle_bc4_regs_rwc[0]_LE=[", .{});
                const rc0 = r_cycle_bc4_regs_rwc[0].toBytesBE();
                for (0..32) |bi| dbg("{x:0>2}", .{rc0[31 - bi]});
                dbg("]\n", .{});
                dbg("  r_cycle_bc5_regs_val[0]_LE=[", .{});
                const rv5 = r_cycle_bc5_regs_val[0].toBytesBE();
                for (0..32) |bi| dbg("{x:0>2}", .{rv5[31 - bi]});
                dbg("]\n", .{});
            }

            // ====================================================================
            // Cache openings to transcript
            // ====================================================================

            dbg("[STAGE6] Transcript before cache_openings: round={}\n", .{transcript.n_rounds});

            // Instance 0: BytecodeReadRaf
            for (bytecode_ra_claims) |claim| {
                transcript.appendScalar("opening_claim", claim);
            }
            dbg("[STAGE6] After BytecodeReadRaf openings ({}): round={}\n", .{bytecode_ra_claims.len, transcript.n_rounds});

            // Instance 1: Booleanity
            // Upstream aliasing: when bytecode_log_k is a multiple of log_k_chunk,
            // BytecodeRa(0)/Booleanity has the same opening point as BytecodeRa(0)/BytecodeReadRaf
            // (no zero-padding in compute_r_address_chunks), so the verifier aliases it
            // and does NOT flush it to transcript.
            const bytecode_ra0_aliases = (bytecode_log_k % log_k_chunk == 0);
            const bool_skip_index = instruction_ra_virtual_claims.len; // BytecodeRa(0) is at index instruction_d in Booleanity's polynomial_types
            for (booleanity_ra_claims, 0..) |claim, i| {
                if (bytecode_ra0_aliases and i == bool_skip_index) continue;
                transcript.appendScalar("opening_claim", claim);
            }

            // Instance 2: HammingBooleanity
            transcript.appendScalar("opening_claim", hamming_weight_claim);

            // Instance 3: RamRaVirtual
            for (ram_ra_virtual_claims) |claim| {
                transcript.appendScalar("opening_claim", claim);
            }

            // Instance 4: LookupsRaVirtual
            for (instruction_ra_virtual_claims) |claim| {
                transcript.appendScalar("opening_claim", claim);
            }

            dbg("[STAGE6] After LookupsRaVirtual openings ({}): round={}\n", .{instruction_ra_virtual_claims.len, transcript.n_rounds});

            // Instance 5: IncClaimReduction
            transcript.appendScalar("opening_claim", ram_inc_claim);
            transcript.appendScalar("opening_claim", rd_inc_claim);
            dbg("[STAGE6] After ALL cache_openings: round={}\n", .{transcript.n_rounds});

            return Stage6Result(F){
                .challenges = challenges,
                .bytecode_ra_claims = bytecode_ra_claims,
                .hamming_weight_claim = hamming_weight_claim,
                .booleanity_ra_claims = booleanity_ra_claims,
                .ram_ra_virtual_claims = ram_ra_virtual_claims,
                .instruction_ra_virtual_claims = instruction_ra_virtual_claims,
                .ram_inc_claim = ram_inc_claim,
                .rd_inc_claim = rd_inc_claim,
                .bytecode_log_k = bytecode_log_k,
                .log_k_chunk = log_k_chunk,
                .n_cycle_vars = n_cycle_vars,
                .bytecode_d = bytecode_d,
                .ram_d = ram_d,
                .instruction_d = instruction_d,
                .allocator = self.allocator,
            };
        }

        /// Compute BytecodeReadRaf input claim and per-stage claims
        /// Returns .{ total_claim, [5]per_stage_claims }
        fn computeBytecodeReadRafInputClaim(
            self: *Self,
            opening_claims: *OpeningClaims(F),
            gamma_powers: []const F,
            stage1_gammas: []const F,
            stage2_gammas: []const F,
            stage3_gammas: []const F,
            stage4_gammas: []const F,
            stage5_gammas: []const F,
        ) struct { total: F, per_stage: [5]F } {
            _ = self;

            const getClaim = struct {
                fn get(oc: *OpeningClaims(F), key: OpeningId) F {
                    return oc.get(key) orelse F.zero();
                }
            }.get;

            // rv_claim_1 (Stage 1 / SpartanOuter)
            var rv1 = F.zero();
            const oc_upc = getClaim(opening_claims, .{ .Virtual = .{ .poly = .UnexpandedPC, .sumcheck_id = .SpartanOuter } });
            rv1 = rv1.add(oc_upc); // No gamma[0] - Jolt formula: unexpanded_pc + γ¹·imm + Σγ^(2+i)·cf[i]
            const oc_imm = getClaim(opening_claims, .{ .Virtual = .{ .poly = .Imm, .sumcheck_id = .SpartanOuter } });
            rv1 = rv1.add(stage1_gammas[1].mul(oc_imm));
            var oc_flags: [14]F = undefined;
            for (0..14) |i| {
                oc_flags[i] = getClaim(opening_claims, .{ .Virtual = .{ .poly = .{ .OpFlags = @intCast(i) }, .sumcheck_id = .SpartanOuter } });
                rv1 = rv1.add(stage1_gammas[2 + i].mul(oc_flags[i]));
            }
            // Debug: print each opening claim component for rv1
            {
                const upc_le = oc_upc.toBytes();
                const imm_le = oc_imm.toBytes();
                dbg("[BCRAF_RV1_DETAIL] oc_UnexpandedPC_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                    upc_le[0], upc_le[1], upc_le[2], upc_le[3], upc_le[4], upc_le[5], upc_le[6], upc_le[7],
                });
                dbg("[BCRAF_RV1_DETAIL] oc_Imm_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                    imm_le[0], imm_le[1], imm_le[2], imm_le[3], imm_le[4], imm_le[5], imm_le[6], imm_le[7],
                });
                for (0..14) |i| {
                    const fl = oc_flags[i].toBytes();
                    dbg("[BCRAF_RV1_DETAIL] oc_OpFlag[{}]_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                        i, fl[0], fl[1], fl[2], fl[3], fl[4], fl[5], fl[6], fl[7],
                    });
                }
                // Also print the oc_PC claim (used for RAF) and FlagIsNoop
                const oc_pc = getClaim(opening_claims, .{ .Virtual = .{ .poly = .PC, .sumcheck_id = .SpartanOuter } });
                const pc_le = oc_pc.toBytes();
                dbg("[BCRAF_RV1_DETAIL] oc_PC_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                    pc_le[0], pc_le[1], pc_le[2], pc_le[3], pc_le[4], pc_le[5], pc_le[6], pc_le[7],
                });
            }

            // rv_claim_2 (Stage 2 / SpartanProductVirtualization)
            var rv2 = F.zero();
            // Upstream: Jump + γ·Branch + γ²·WriteLookupOutputToRD + γ³·VirtualInstruction
            rv2 = rv2.add(stage2_gammas[0].mul(getClaim(opening_claims,
                .{ .Virtual = .{ .poly = .{ .OpFlags = 5 }, .sumcheck_id = .SpartanProductVirtualization } })));
            rv2 = rv2.add(stage2_gammas[1].mul(getClaim(opening_claims,
                .{ .Virtual = .{ .poly = .{ .InstructionFlags = 4 }, .sumcheck_id = .SpartanProductVirtualization } })));
            rv2 = rv2.add(stage2_gammas[2].mul(getClaim(opening_claims,
                .{ .Virtual = .{ .poly = .{ .OpFlags = 6 }, .sumcheck_id = .SpartanProductVirtualization } })));
            rv2 = rv2.add(stage2_gammas[3].mul(getClaim(opening_claims,
                .{ .Virtual = .{ .poly = .{ .OpFlags = 7 }, .sumcheck_id = .SpartanProductVirtualization } })));

            // rv_claim_3 (Stage 3)
            var rv3 = F.zero();
            rv3 = rv3.add(getClaim(opening_claims,
                .{ .Virtual = .{ .poly = .Imm, .sumcheck_id = .InstructionInputVirtualization } })); // No gamma[0] - Jolt formula: imm + γ¹·unexpanded_pc + ...
            rv3 = rv3.add(stage3_gammas[1].mul(getClaim(opening_claims,
                .{ .Virtual = .{ .poly = .UnexpandedPC, .sumcheck_id = .SpartanShift } })));
            rv3 = rv3.add(stage3_gammas[2].mul(getClaim(opening_claims,
                .{ .Virtual = .{ .poly = .{ .InstructionFlags = 2 }, .sumcheck_id = .InstructionInputVirtualization } })));
            rv3 = rv3.add(stage3_gammas[3].mul(getClaim(opening_claims,
                .{ .Virtual = .{ .poly = .{ .InstructionFlags = 0 }, .sumcheck_id = .InstructionInputVirtualization } })));
            rv3 = rv3.add(stage3_gammas[4].mul(getClaim(opening_claims,
                .{ .Virtual = .{ .poly = .{ .InstructionFlags = 3 }, .sumcheck_id = .InstructionInputVirtualization } })));
            rv3 = rv3.add(stage3_gammas[5].mul(getClaim(opening_claims,
                .{ .Virtual = .{ .poly = .{ .InstructionFlags = 1 }, .sumcheck_id = .InstructionInputVirtualization } })));
            rv3 = rv3.add(stage3_gammas[6].mul(getClaim(opening_claims,
                .{ .Virtual = .{ .poly = .{ .InstructionFlags = 5 }, .sumcheck_id = .SpartanShift } })));
            rv3 = rv3.add(stage3_gammas[7].mul(getClaim(opening_claims,
                .{ .Virtual = .{ .poly = .{ .OpFlags = 7 }, .sumcheck_id = .SpartanShift } })));
            rv3 = rv3.add(stage3_gammas[8].mul(getClaim(opening_claims,
                .{ .Virtual = .{ .poly = .{ .OpFlags = 12 }, .sumcheck_id = .SpartanShift } })));

            // rv_claim_4 (Stage 4)
            var rv4 = F.zero();
            rv4 = rv4.add(stage4_gammas[0].mul(getClaim(opening_claims,
                .{ .Virtual = .{ .poly = .RdWa, .sumcheck_id = .RegistersReadWriteChecking } })));
            rv4 = rv4.add(stage4_gammas[1].mul(getClaim(opening_claims,
                .{ .Virtual = .{ .poly = .Rs1Ra, .sumcheck_id = .RegistersReadWriteChecking } })));
            rv4 = rv4.add(stage4_gammas[2].mul(getClaim(opening_claims,
                .{ .Virtual = .{ .poly = .Rs2Ra, .sumcheck_id = .RegistersReadWriteChecking } })));

            // rv_claim_5 (Stage 5)
            const NUM_LOOKUP_TABLES: usize = 41;
            var rv5 = F.zero();
            const rv5_rdwa = getClaim(opening_claims,
                .{ .Virtual = .{ .poly = .RdWa, .sumcheck_id = .RegistersValEvaluation } });
            rv5 = rv5.add(rv5_rdwa); // No gamma[0] - Jolt formula: eq(rd,r) + γ¹·!interleaved + ...
            const rv5_raf_flag = getClaim(opening_claims,
                .{ .Virtual = .{ .poly = .InstructionRafFlag, .sumcheck_id = .InstructionReadRaf } });
            rv5 = rv5.add(stage5_gammas[1].mul(rv5_raf_flag));
            for (0..NUM_LOOKUP_TABLES) |i| {
                const lt_claim = getClaim(opening_claims,
                    .{ .Virtual = .{ .poly = .{ .LookupTableFlag = i }, .sumcheck_id = .InstructionReadRaf } });
                rv5 = rv5.add(stage5_gammas[2 + i].mul(lt_claim));
                if (!lt_claim.eql(F.zero())) {
                    const ltb = lt_claim.toBytes();
                    dbg("[BCRAF_RV5] LookupTableFlag({})_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                        i, ltb[0], ltb[1], ltb[2], ltb[3], ltb[4], ltb[5], ltb[6], ltb[7],
                    });
                }
            }
            {
                const rdwa_le = rv5_rdwa.toBytes();
                const rff_le = rv5_raf_flag.toBytes();
                const rv5_le = rv5.toBytes();
                dbg("[BCRAF_RV5] RdWa_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                    rdwa_le[0], rdwa_le[1], rdwa_le[2], rdwa_le[3], rdwa_le[4], rdwa_le[5], rdwa_le[6], rdwa_le[7],
                });
                dbg("[BCRAF_RV5] InstructionRafFlag_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                    rff_le[0], rff_le[1], rff_le[2], rff_le[3], rff_le[4], rff_le[5], rff_le[6], rff_le[7],
                });
                dbg("[BCRAF_RV5] total_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                    rv5_le[0], rv5_le[1], rv5_le[2], rv5_le[3], rv5_le[4], rv5_le[5], rv5_le[6], rv5_le[7],
                });
            }

            // RAF claims
            const raf_claim = getClaim(opening_claims,
                .{ .Virtual = .{ .poly = .PC, .sumcheck_id = .SpartanOuter } });
            const raf_shift_claim = getClaim(opening_claims,
                .{ .Virtual = .{ .poly = .PC, .sumcheck_id = .SpartanShift } });

            // Debug: print per-stage rv_claims and raf_claims
            {
                const rv_arr = [5]F{ rv1, rv2, rv3, rv4, rv5 };
                for (0..5) |s| {
                    const rvl = rv_arr[s].toBytes();
                    dbg("[BCRAF_INPUT] rv_claim[{}]_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                        s, rvl[0], rvl[1], rvl[2], rvl[3], rvl[4], rvl[5], rvl[6], rvl[7],
                    });
                }
                const raf_le = raf_claim.toBytes();
                const rafs_le = raf_shift_claim.toBytes();
                dbg("[BCRAF_INPUT] raf_claim_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                    raf_le[0], raf_le[1], raf_le[2], raf_le[3], raf_le[4], raf_le[5], raf_le[6], raf_le[7],
                });
                dbg("[BCRAF_INPUT] raf_shift_claim_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                    rafs_le[0], rafs_le[1], rafs_le[2], rafs_le[3], rafs_le[4], rafs_le[5], rafs_le[6], rafs_le[7],
                });
                // Also print per-stage claims with RAF folded in (like Jolt's claim_per_stage)
                const cps0 = rv1.add(gamma_powers[5].mul(raf_claim));
                const cps2 = rv3.add(gamma_powers[4].mul(raf_shift_claim));
                const cps0l = cps0.toBytes();
                const cps2l = cps2.toBytes();
                dbg("[BCRAF_INPUT] claim_per_stage[0]_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                    cps0l[0], cps0l[1], cps0l[2], cps0l[3], cps0l[4], cps0l[5], cps0l[6], cps0l[7],
                });
                dbg("[BCRAF_INPUT] claim_per_stage[2]_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                    cps2l[0], cps2l[1], cps2l[2], cps2l[3], cps2l[4], cps2l[5], cps2l[6], cps2l[7],
                });
            }

            // Per-stage claims (like Jolt's claim_per_stage)
            // claim_per_stage[s] = rv_claim[s] + RAF_s contribution
            const per_stage = [5]F{
                rv1.add(gamma_powers[5].mul(raf_claim)), // Stage 0: rv1 + gamma^5 * raf
                rv2, // Stage 1: rv2
                rv3.add(gamma_powers[4].mul(raf_shift_claim)), // Stage 2: rv3 + gamma^4 * raf_shift
                rv4, // Stage 3: rv4
                rv5, // Stage 4: rv5
            };

            // Combine: total = Σ_s gamma^s * per_stage[s]
            var result = F.zero();
            for (0..5) |s| {
                const term = gamma_powers[s].mul(per_stage[s]);
                result = result.add(term);
                const ps_le = per_stage[s].toBytes();
                const gp_le = gamma_powers[s].toBytes();
                const tm_le = term.toBytes();
                dbg("[BCRAF_AGG_OC] s={} gp_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}] ps_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}] term_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                    s,
                    gp_le[0], gp_le[1], gp_le[2], gp_le[3], gp_le[4], gp_le[5], gp_le[6], gp_le[7],
                    ps_le[0], ps_le[1], ps_le[2], ps_le[3], ps_le[4], ps_le[5], ps_le[6], ps_le[7],
                    tm_le[0], tm_le[1], tm_le[2], tm_le[3], tm_le[4], tm_le[5], tm_le[6], tm_le[7],
                });
            }
            const res_le = result.toBytes();
            dbg("[BCRAF_AGG_OC] total_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                res_le[0], res_le[1], res_le[2], res_le[3], res_le[4], res_le[5], res_le[6], res_le[7],
            });

            return .{ .total = result, .per_stage = per_stage };
        }
    };
}

// =============================================================================
// Helper: Add variable-length instance evals to combined_evals with interpolation
// =============================================================================
// All evaluation arrays use Vandermonde format: [p(0), p(1), ..., p(d)]
// (evaluations at consecutive integer points, no p_inf)
fn addInstanceEvalsToCombibed(comptime F: type, combined_evals: []F, polys: []const F, batch_coeff: F, num_evals: usize) void {
    const inst_n_evals = polys.len;

    if (inst_n_evals >= num_evals) {
        // Instance has enough eval points - just add the first num_evals
        for (0..num_evals) |k| {
            combined_evals[k] = combined_evals[k].add(batch_coeff.mul(polys[k]));
        }
    } else {
        // Instance has fewer eval points - need Lagrange interpolation for missing points
        // polys format (Vandermonde): [p(0), p(1), ..., p(inst_n_evals-1)]
        // Need to interpolate p(inst_n_evals), ..., p(num_evals-1)

        // Add known evaluation points
        for (0..inst_n_evals) |k| {
            combined_evals[k] = combined_evals[k].add(batch_coeff.mul(polys[k]));
        }

        // Lagrange interpolation for missing points
        for (inst_n_evals..num_evals) |k| {
            const x = F.fromU64(@intCast(k));
            var lagrange_val = F.zero();
            for (0..inst_n_evals) |m| {
                var basis = F.one();
                const xm = F.fromU64(@intCast(m));
                for (0..inst_n_evals) |n| {
                    if (n != m) {
                        const xn = F.fromU64(@intCast(n));
                        basis = basis.mul(x.sub(xn)).mul(xm.sub(xn).inverse().?);
                    }
                }
                lagrange_val = lagrange_val.add(basis.mul(polys[m]));
            }
            combined_evals[k] = combined_evals[k].add(batch_coeff.mul(lagrange_val));
        }
    }
}

/// Add fixed-size instance evaluations to combined (for degree-3 instances like Hamming)
// All evaluation arrays use Vandermonde format: [p(0), p(1), ..., p(d)]
fn addFixedEvalsToCombibed(comptime F: type, combined_evals: []F, polys: []const F, n_polys: usize, batch_coeff: F, num_evals: usize) void {
    if (n_polys >= num_evals) {
        // Instance has enough eval points - add the first num_evals
        for (0..num_evals) |k| {
            combined_evals[k] = combined_evals[k].add(batch_coeff.mul(polys[k]));
        }
    } else {
        // Instance has fewer eval points - need Lagrange interpolation for missing points
        for (0..n_polys) |k| {
            combined_evals[k] = combined_evals[k].add(batch_coeff.mul(polys[k]));
        }

        // Lagrange interpolation for missing points
        for (n_polys..num_evals) |k| {
            const x = F.fromU64(@intCast(k));
            var lagrange_val = F.zero();
            for (0..n_polys) |m| {
                var basis = F.one();
                const xm = F.fromU64(@intCast(m));
                for (0..n_polys) |n| {
                    if (n != m) {
                        const xn = F.fromU64(@intCast(n));
                        basis = basis.mul(x.sub(xn)).mul(xm.sub(xn).inverse().?);
                    }
                }
                lagrange_val = lagrange_val.add(basis.mul(polys[m]));
            }
            combined_evals[k] = combined_evals[k].add(batch_coeff.mul(lagrange_val));
        }
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Compute eq polynomial table: eq(r, j) for all j in [0, 2^n_vars)
/// r is in BIG_ENDIAN order (r[0] is the most significant variable)
pub fn computeEqTable(comptime F: type, allocator: Allocator, r: []const F, n_vars: usize) ![]F {
    return computeEqTableParallel(F, allocator, r, n_vars, null);
}

/// Compute eq polynomial table with optional parallel inner loops.
/// Same as computeEqTable but parallelizes large levels via ThreadPool.
pub fn computeEqTableParallel(comptime F: type, allocator: Allocator, r: []const F, n_vars: usize, pool: ?*ThreadPool) ![]F {
    const size: usize = @as(usize, 1) << @intCast(n_vars);
    var table = try allocator.alloc(F, size);

    table[0] = F.one();

    for (0..n_vars) |i| {
        const r_i = r[i];
        const cur_size: usize = @as(usize, 1) << @intCast(i);

        if (pool != null and cur_size >= 256) {
            // Parallel: forward iteration, writes to disjoint halves [0..cur_size) and [cur_size..2*cur_size)
            const Ctx = struct {
                tbl: []F,
                ri: F,
                cs: usize,
            };
            const ctx = Ctx{ .tbl = table, .ri = r_i, .cs = cur_size };
            pool.?.parallelForForce(cur_size, ctx, struct {
                fn f(c: Ctx, j: usize) void {
                    const x = c.tbl[j];
                    const y = x.mul(c.ri);
                    c.tbl[j + c.cs] = y;
                    c.tbl[j] = x.sub(y);
                }
            }.f);
        } else {
            // Sequential: backward iteration (original)
            var j: usize = cur_size;
            while (j > 0) {
                j -= 1;
                const x = table[j];
                const y = x.mul(r_i);
                table[j + cur_size] = y;
                table[j] = x.sub(y);
            }
        }
    }

    return table;
}

/// Convert signed i128 to field element
fn fieldFromI128(comptime F: type, val: i128) F {
    if (val >= 0) {
        return F.fromU128(@intCast(val));
    } else {
        return F.fromU128(@intCast(-val)).neg();
    }
}

/// Extract chunk from address value using MSB-first ordering (matching Jolt)
/// chunk_idx=0 is the most significant chunk
pub fn extractChunkMSB(addr: u64, chunk_idx: usize, total_chunks: usize, log_k_chunk: usize) usize {
    // Jolt: shift = log_k_chunk * (d - 1 - chunk_idx)
    const shift_amount = log_k_chunk * (total_chunks - 1 - chunk_idx);
    if (shift_amount >= 64) return 0;
    const shift: u6 = @intCast(shift_amount);
    const mask: u64 = (@as(u64, 1) << @intCast(log_k_chunk)) - 1;
    return @intCast((addr >> shift) & mask);
}

/// Interleave bits of two 64-bit values to form a 128-bit lookup index
/// Matches Jolt's interleave_bits(even_bits, odd_bits): result = (even << 1) | odd
/// So even_bits (rs1) go to odd bit positions (1,3,5,...,127)
/// and odd_bits (rs2) go to even bit positions (0,2,4,...,126)
pub fn interleaveBits(rs1: u64, rs2: u64) u128 {
    // Spread rs1 bits to odd positions
    var x: u128 = @intCast(rs1);
    x = (x | (x << 32)) & 0x0000_0000_FFFF_FFFF_0000_0000_FFFF_FFFF;
    x = (x | (x << 16)) & 0x0000_FFFF_0000_FFFF_0000_FFFF_0000_FFFF;
    x = (x | (x << 8)) & 0x00FF_00FF_00FF_00FF_00FF_00FF_00FF_00FF;
    x = (x | (x << 4)) & 0x0F0F_0F0F_0F0F_0F0F_0F0F_0F0F_0F0F_0F0F;
    x = (x | (x << 2)) & 0x3333_3333_3333_3333_3333_3333_3333_3333;
    x = (x | (x << 1)) & 0x5555_5555_5555_5555_5555_5555_5555_5555;

    // Spread rs2 bits to even positions
    var y: u128 = @intCast(rs2);
    y = (y | (y << 32)) & 0x0000_0000_FFFF_FFFF_0000_0000_FFFF_FFFF;
    y = (y | (y << 16)) & 0x0000_FFFF_0000_FFFF_0000_FFFF_0000_FFFF;
    y = (y | (y << 8)) & 0x00FF_00FF_00FF_00FF_00FF_00FF_00FF_00FF;
    y = (y | (y << 4)) & 0x0F0F_0F0F_0F0F_0F0F_0F0F_0F0F_0F0F_0F0F;
    y = (y | (y << 2)) & 0x3333_3333_3333_3333_3333_3333_3333_3333;
    y = (y | (y << 1)) & 0x5555_5555_5555_5555_5555_5555_5555_5555;

    return (x << 1) | y;
}

/// Decode sign-extended immediate from RISC-V instruction encoding, returned as u64 (two's complement).
/// This matches Jolt's `to_instruction_inputs()` which sign-extends the immediate value.
fn decodeImmediateU64(instr: u32) u64 {
    const opcode: u8 = @truncate(instr & 0x7f);
    switch (opcode) {
        // I-type: imm[11:0] at bits [31:20], sign-extended
        0x13, 0x03, 0x67, 0x1b, 0x73 => {
            const imm12: u32 = instr >> 20;
            const imm_signed: i64 = @as(i64, @as(i32, @bitCast(imm12 << 20)) >> 20);
            return @bitCast(imm_signed);
        },
        // S-type: imm[11:5] at [31:25], imm[4:0] at [11:7], sign-extended
        0x23 => {
            const imm11_5 = (instr >> 25) & 0x7f;
            const imm4_0 = (instr >> 7) & 0x1f;
            const imm12: u32 = (imm11_5 << 5) | imm4_0;
            const imm_signed: i64 = @as(i64, @as(i32, @bitCast(imm12 << 20)) >> 20);
            return @bitCast(imm_signed);
        },
        // B-type: imm[12|10:5] at [31:25], imm[4:1|11] at [11:7], sign-extended, *2
        0x63 => {
            const imm12 = (instr >> 31) & 1;
            const imm10_5 = (instr >> 25) & 0x3f;
            const imm4_1 = (instr >> 8) & 0xf;
            const imm11 = (instr >> 7) & 1;
            const imm13: u32 = (imm12 << 12) | (imm11 << 11) | (imm10_5 << 5) | (imm4_1 << 1);
            const imm_signed: i64 = @as(i64, @as(i32, @bitCast(imm13 << 19)) >> 19);
            return @bitCast(imm_signed);
        },
        // U-type: imm[31:12] at [31:12], shifted left by 12, SIGN-EXTENDED to 64 bits
        // Matches Jolt's FormatU.parse: `as i32 as i64 as u64`
        0x37, 0x17 => {
            const imm_upper: u32 = instr & 0xFFFFF000;
            return @bitCast(@as(i64, @as(i32, @bitCast(imm_upper))));
        },
        // J-type: imm[20|10:1|11|19:12] at [31:12], sign-extended, *2
        0x6f => {
            const imm20 = (instr >> 31) & 1;
            const imm10_1 = (instr >> 21) & 0x3ff;
            const imm11 = (instr >> 20) & 1;
            const imm19_12 = (instr >> 12) & 0xff;
            const imm21: u32 = (imm20 << 20) | (imm19_12 << 12) | (imm11 << 11) | (imm10_1 << 1);
            const imm_signed: i64 = @as(i64, @as(i32, @bitCast(imm21 << 11)) >> 11);
            return @bitCast(imm_signed);
        },
        else => return 0,
    }
}

/// Compute the 128-bit lookup index for a trace step.
///
/// This matches Jolt's per-instruction `to_lookup_index()` method:
/// - AddOperands instructions (ADD, ADDI, etc.): returns raw sum as u128 (NO interleaving)
/// - SubtractOperands instructions (SUB, SUBW): returns raw shifted difference as u128
/// - MultiplyOperands instructions (MUL, MULHU): returns raw product as u128
/// - Standard instructions (XOR, AND, OR, SLT, branches): returns interleave_bits(x, y)
/// - No-lookup instructions (Load, Store, SLL, SRL): returns 0
/// - NoOp cycles: returns 0
pub fn computeLookupIndex(step: tracer.TraceStep) u128 {
    if (step.is_noop and !step.is_termination_store) return 0;

    const instr = step.instruction;
    const opcode: u8 = @truncate(instr & 0x7f);
    const funct3: u3 = @truncate((instr >> 12) & 0x7);
    const funct7: u7 = @truncate(instr >> 25);

    // Check if instruction has a lookup table at all
    if (!hasLookupTable(opcode, funct3, funct7)) return 0;

    // Virtual opcodes: handle specially since they don't follow standard RISC-V encoding
    if (opcode == 0x0B) {
        // VirtualSignExtendWord: AddOperands → rs1 + 0 = rs1
        // Jolt's to_lookup_index() returns rs1 directly (no interleaving)
        return @as(u128, step.rs1_value);
    }
    if (opcode == 0x2B) {
        // VirtualMULI: MultiplyOperands → rs1 * (1 << shamt)
        // The instruction encodes shamt in I-type imm field (bits [31:20])
        // Jolt's to_lookup_index() returns rs1 * imm where imm = 1 << shamt
        const shamt_raw: u32 = instr >> 20;
        const shamt: u6 = @truncate(shamt_raw & 0x3F);
        const multiplier: u128 = @as(u128, 1) << shamt;
        return @as(u128, step.rs1_value) * multiplier;
    }
    if (opcode == 0x5B) {
        // VirtualSRLI: interleaved(rs1_value, bitmask)
        // The instruction encodes total_shift in I-type imm field (bits [31:20])
        // The 64-bit bitmask is reconstructed: ones = (1 << (64-shift)) - 1; bitmask = ones << shift
        const total_shift_raw: u32 = instr >> 20;
        const total_shift: u7 = @truncate(total_shift_raw & 0x3F);
        const ones: u128 = (@as(u128, 1) << @intCast(64 - @as(u8, total_shift))) - 1;
        const bitmask: u64 = @truncate(ones << total_shift);
        return interleaveBits(step.rs1_value, bitmask);
    }
    if (opcode == 0x02) {
        // VirtualAdvice: the lookup index is the advice value (rd_value)
        // Jolt's to_lookup_index() returns the second operand which is the advice value
        return @as(u128, step.rd_value);
    }
    if (opcode == 0x22) {
        // VirtualAssertEQ: interleaved(rs1_value, rs2_value)
        // LeftOperandIsRs1Value, RightOperandIsRs2Value → interleave
        return interleaveBits(step.rs1_value, step.rs2_value);
    }
    if (opcode == 0x42) {
        // VirtualZeroExtendWord: AddOperands → rs1 + 0 = rs1
        // Jolt's to_lookup_index() returns rs1 directly (like SignExtendWord)
        return @as(u128, step.rs1_value);
    }
    if (opcode == 0x62) {
        // VirtualAssertValidUnsignedRemainder: interleaved(rs1_value, rs2_value)
        // LeftOperandIsRs1Value, RightOperandIsRs2Value → interleave
        return interleaveBits(step.rs1_value, step.rs2_value);
    }

    // Determine left_input and right_input (matching Jolt's to_instruction_inputs)
    const left_is_rs1: bool = switch (opcode) {
        0x33, 0x3b, 0x23, 0x63, 0x13, 0x03, 0x67, 0x1b => true,
        else => false,
    };
    const left_is_pc: bool = switch (opcode) {
        0x17, 0x6f => true,
        else => false,
    };
    const right_is_rs2: bool = switch (opcode) {
        0x33, 0x63, 0x3b => true,
        else => false,
    };
    const right_is_imm: bool = switch (opcode) {
        0x13, 0x03, 0x67, 0x23, 0x37, 0x17, 0x6f, 0x1b => true,
        else => false,
    };

    var left_input: u64 = 0;
    if (left_is_rs1) left_input = step.rs1_value;
    if (left_is_pc) left_input = step.unexpanded_pc;

    var right_input: u64 = 0;
    if (right_is_rs2) right_input = step.rs2_value;
    if (right_is_imm) right_input = decodeImmediateU64(instr);

    // Now compute the lookup index based on the instruction's operand mode
    switch (opcode) {
        0x33 => { // R-type
            if (funct7 == 0x01) {
                // M-extension
                if (funct3 == 0x0) {
                    // MUL: MultiplyOperands → raw product
                    return @as(u128, left_input) * @as(u128, right_input);
                } else if (funct3 == 0x3) {
                    // MULHU: MultiplyOperands → raw product
                    return @as(u128, left_input) * @as(u128, right_input);
                } else {
                    // Other M-ext: interleaved
                    return interleaveBits(left_input, right_input);
                }
            } else if (funct7 == 0x20 and funct3 == 0x0) {
                // SUB: SubtractOperands → x + (2^64 - y)
                return @as(u128, left_input) + (@as(u128, 1) << 64) - @as(u128, right_input);
            } else if (funct7 == 0 and funct3 == 0x0) {
                // ADD: AddOperands → raw sum
                return @as(u128, left_input) + @as(u128, right_input);
            } else {
                // Other R-type (AND, OR, XOR, SLT, SLTU): interleaved
                return interleaveBits(left_input, right_input);
            }
        },
        0x13 => { // I-type ALU
            if (funct3 == 0) {
                // ADDI: AddOperands → raw sum
                return @as(u128, left_input) + @as(u128, right_input);
            } else {
                // SLLI, SLTI, SLTIU, XORI, SRLI, SRAI, ORI, ANDI: interleaved
                return interleaveBits(left_input, right_input);
            }
        },
        0x37 => { // LUI: AddOperands → immediate directly (left=0)
            return @as(u128, left_input) + @as(u128, right_input);
        },
        0x17 => { // AUIPC: AddOperands → PC + imm
            return @as(u128, left_input) + @as(u128, right_input);
        },
        0x6f => { // JAL: AddOperands → PC + imm
            return @as(u128, left_input) + @as(u128, right_input);
        },
        0x67 => { // JALR: AddOperands → rs1 + imm
            return @as(u128, left_input) + @as(u128, right_input);
        },
        0x1b => { // I-type word ALU
            if (funct3 == 0) {
                // ADDIW: AddOperands → raw sum
                return @as(u128, left_input) + @as(u128, right_input);
            } else {
                // SLLIW, SRLIW, SRAIW: interleaved
                return interleaveBits(left_input, right_input);
            }
        },
        0x3b => { // OP-32
            if (funct3 == 0 and funct7 == 0) {
                // ADDW: AddOperands → raw sum
                return @as(u128, left_input) + @as(u128, right_input);
            } else if (funct3 == 0 and funct7 == 0x20) {
                // SUBW: SubtractOperands → x + (2^64 - y)
                return @as(u128, left_input) + (@as(u128, 1) << 64) - @as(u128, right_input);
            } else {
                // Other 0x3b: interleaved
                return interleaveBits(left_input, right_input);
            }
        },
        0x63 => { // Branch: interleaved
            return interleaveBits(left_input, right_input);
        },
        else => {
            // Default: interleaved
            return interleaveBits(left_input, right_input);
        },
    }
}

/// Get lookup index chunk from trace step.
/// This matches Jolt's lookup_index_chunk with instruction_shifts (MSB-first ordering).
/// Uses the instruction-type-aware computeLookupIndex to correctly handle
/// AddOperands, SubtractOperands, and MultiplyOperands instructions.
fn getLookupChunkInterleaved(step: tracer.TraceStep, chunk_idx: usize, log_k_chunk: usize, instruction_d: usize) usize {
    // Build the correct 128-bit lookup index based on instruction type
    const lookup_index = computeLookupIndex(step);

    // MSB-first: shift = log_k_chunk * (instruction_d - 1 - chunk_idx)
    const shift_amount = log_k_chunk * (instruction_d - 1 - chunk_idx);
    if (shift_amount >= 128) return 0;
    const shift: u7 = @intCast(shift_amount);
    const mask: u128 = (@as(u128, 1) << @intCast(log_k_chunk)) - 1;
    return @intCast((lookup_index >> shift) & mask);
}

/// Evaluate a polynomial at a point given its Toom-Cook evals format:
/// evals = [p(0), p(1), ..., p(d-1), p(inf)]
/// where p(inf) is the leading coefficient (coefficient of x^d).
/// The polynomial has degree d where d = evals.len - 1.
/// Uses Lagrange interpolation on the d finite points {0, 1, ..., d-1}
/// plus the leading coefficient correction.
/// Evaluate polynomial at challenge given Vandermonde evals [p(0), p(1), ..., p(d)]
/// Uses Lagrange interpolation through all n_evals points at consecutive integers.
fn evaluatePolyFromEvals(comptime F: type, evals: []const F, challenge: F) F {
    const n_evals = evals.len;

    // Lagrange interpolation through (0, p(0)), (1, p(1)), ..., (n_evals-1, p(n_evals-1))
    var result = F.zero();
    for (0..n_evals) |m| {
        var basis = F.one();
        const xm = F.fromU64(@intCast(m));
        for (0..n_evals) |n| {
            if (n != m) {
                const xn = F.fromU64(@intCast(n));
                basis = basis.mul(challenge.sub(xn)).mul(xm.sub(xn).inverse().?);
            }
        }
        result = result.add(basis.mul(evals[m]));
    }

    return result;
}

/// Evaluate degree-3 polynomial at challenge given Vandermonde evals [p(0), p(1), p(2), p(3)]
fn evaluateDeg3FromEvals(comptime F: type, evals: [4]F, challenge: F) F {
    const p0 = evals[0];
    const p1 = evals[1];
    const p2 = evals[2];
    const p3 = evals[3];

    // Lagrange interpolation through (0, p0), (1, p1), (2, p2), (3, p3)
    // L_0(x) = (x-1)(x-2)(x-3)/((0-1)(0-2)(0-3)) = (x-1)(x-2)(x-3)/(-6)
    // L_1(x) = (x-0)(x-2)(x-3)/((1-0)(1-2)(1-3)) = x(x-2)(x-3)/(2)
    // L_2(x) = (x-0)(x-1)(x-3)/((2-0)(2-1)(2-3)) = x(x-1)(x-3)/(-2)
    // L_3(x) = (x-0)(x-1)(x-2)/((3-0)(3-1)(3-2)) = x(x-1)(x-2)/(6)
    const x = challenge;
    const xm1 = x.sub(F.one());
    const xm2 = x.sub(F.fromU64(2));
    const xm3 = x.sub(F.fromU64(3));
    const six_inv = F.fromU64(6).inverse().?;
    const two_inv = F.fromU64(2).inverse().?;

    const l0 = xm1.mul(xm2).mul(xm3).mul(six_inv).neg();
    const l1 = x.mul(xm2).mul(xm3).mul(two_inv);
    const l2 = x.mul(xm1).mul(xm3).mul(two_inv).neg();
    const l3 = x.mul(xm1).mul(xm2).mul(six_inv);

    return l0.mul(p0).add(l1.mul(p1)).add(l2.mul(p2)).add(l3.mul(p3));
}
