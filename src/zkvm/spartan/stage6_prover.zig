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
// Stage 6 fine-grained bench timing - set to true for per-instance timing
const s6_bench_timing = false;

// Maximum evaluation points for parallelReduce accumulator.
// Covers all sub-provers: LookupsRa (M+2 ≤ 10), RamRa (d+2 ≤ 6), BytecodeReadRaf (d+2 ≤ 4).
const MAX_RA_EVALS = 16;
fn dbg(comptime fmt: []const u8, args: anytype) void {
    if (debug_verbose) std.debug.print(fmt, args);
}

const Allocator = std.mem.Allocator;
const ThreadPool = @import("../../utils/thread_pool.zig").ThreadPool;
const GpuPolyOps = @import("../../gpu/mod.zig").GpuPolyOps;

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
const ra_poly_mod = @import("ra_poly.zig");
const UnreducedProductAccum = @import("../../field/mod.zig").UnreducedProductAccum;

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
    if (virtual_sequence_remaining) |vsr| { if (vsr != 0) cf[@intFromEnum(CircuitFlags.DoNotUpdateUnexpandedPC)] = true; }
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
    if (virtual_sequence_remaining) |vsr| { if (vsr != 0) cf[@intFromEnum(CircuitFlags.DoNotUpdateUnexpandedPC)] = true; }
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
            1, 2, 3, 4, 6, 7 => if (funct7_raw == 0x01) @as(u7, 0x01) else 0,
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
            0x13, 0x03, 0x67, 0x1b, 0x37, 0x17, 0x6f, 0x0B, 0x2B, 0x5B => 255, // I-type, U-type, J-type, Virtual: no rs2
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

        // =====================================================================
        // Standard RISC-V instructions — build a 32-bit word and delegate
        // =====================================================================

        // R-type (opcode 0x33): ADD, SUB, SLL, SLT, SLTU, XOR, SRL, SRA, OR, AND,
        //                       MUL, MULH, MULHSU, MULHU, DIV, DIVU, REM, REMU
        .ADD, .SUB, .SLL, .SLT, .SLTU, .XOR, .SRL, .SRA, .OR, .AND, .MUL, .MULH, .MULHSU, .MULHU, .DIV, .DIVU, .REM, .REMU => {
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
            applyVirtualAndCompressedFlags(entry, rd, rs1, 255, vsr, is_first, is_compressed);
        },

        // OP-IMM-32 (opcode 0x1b): ADDIW, SLLIW, SRLIW, SRAIW
        .ADDIW, .SLLIW, .SRLIW, .SRAIW => {
            const info = getOpImm32Encoding(instr.variant);
            const imm_u64: u64 = @bitCast(imm);
            const word = buildIType(imm_u64, rs1, info.funct3, rd, 0x1b);
            populateEntryFromInstruction(entry, word, instr.address);
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
            const imm_u64: u64 = @bitCast(imm);
            const word = buildJType(imm_u64, rd, 0x6F);
            populateEntryFromInstruction(entry, word, instr.address);
            // JAL x0 → vr40 remapping
            if (rd == 0) {
                entry.rd = 40;
                entry.instruction_flags[@intFromEnum(InstructionFlags.IsRdNotZero)] = true;
            }
            applyVirtualAndCompressedFlags(entry, entry.rd, 255, 255, vsr, is_first, is_compressed);
        },

        // JALR (opcode 0x67)
        .JALR => {
            const imm_u64: u64 = @bitCast(imm);
            const word = buildIType(imm_u64, rs1, 0, rd, 0x67);
            populateEntryFromInstruction(entry, word, instr.address);
            // JALR x0 → vr40 remapping
            if (rd == 0) {
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
        .ECALL => {
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
            entry.address = instr.address;
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
    if (rd_full != 255) entry.rd = rd_full;
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
    return .{ .funct3 = switch (variant) {
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
    } };
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
    // Phase 1: Populate from preprocessing bytecode (preferred) or raw ELF bytes
    // ================================================================
    // Using preprocessing bytecode ensures the prover's bytecode entries match
    // exactly what the verifier will compute from the serialized preprocessing.
    // This is critical for programs with .rodata sections (like SHA256) where
    // data bytes in the code section would be decoded differently by two
    // independent decoders.
    if (program_code_bytes) |code_bytes| {
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
                    .address = 0, .imm = 0, .rd = 255, .rs1 = 255, .rs2 = 255,
                    .circuit_flags = cf, .instruction_flags = inf,
                    .lookup_table_index = 255, .is_interleaved = true,
                    .virtual_sequence_remaining = null, .is_first_in_sequence = false,
                    .opcode = 0, .funct3 = 0,
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
            }
            else {
                // Real instruction in preprocessing but prover has different/UNIMPL.
                // Re-decode from raw bytes at the preprocessing's address.
                if (program_code_bytes) |code_bytes| {
                    const byte_offset = prep_addr - code_base_address;
                    if (byte_offset + 2 <= code_bytes.len) {
                        const hw: u16 = std.mem.readInt(u16, code_bytes[byte_offset..][0..2], .little);
                        const is_comp = (hw & 0x3) != 0x3;
                        var instr_word: u32 = undefined;
                        if (is_comp) {
                            instr_word = instruction_mod.uncompressInstruction(@as(u32, hw), .Bit64);
                        } else if (byte_offset + 4 <= code_bytes.len) {
                            instr_word = std.mem.readInt(u32, code_bytes[byte_offset..][0..4], .little);
                        } else {
                            instr_word = 0;
                        }
                        populateEntryFromInstruction(&entries[k], instr_word, prep_addr);
                        if (is_comp) entries[k].circuit_flags[@intFromEnum(CircuitFlags.IsCompressed)] = true;
                    }
                }
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
            7 => (funct7 == 0 or funct7 == 0x01), // AND, REMU
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
        0x03, 0x23, 0x63, 0x37, 0x17, 0x6F, 0x67, // Standard opcodes
        0x73, 0x0F, // ECALL, FENCE (treated as NoOp in Jolt)
        => return true,
        // Virtual opcodes (0x0B, 0x2B, 0x5B, 0x02, 0x22) are NOT recognized here
        // since they only appear in virtual sequence entries created by populate functions,
        // never in raw ELF bytes.
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
fn getLookupTableIndex(opcode: u8, funct3: u3, funct7: u7) u8 {
    return switch (opcode) {
        0x33 => switch (funct3) { // R-type
            0 => if (funct7 == 0) @as(u8, 0) // ADD → RangeCheck
            else if (funct7 == 0x20) 0 // SUB → RangeCheck
            else if (funct7 == 0x01) 0 // MUL → RangeCheck
            else 255,
            7 => if (funct7 == 0) @as(u8, 2) // AND → And
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
        0x5B => if (funct3 == 5) @as(u8, 26) else 25, // VirtualSRAI/VirtualSRA → VirtualSRA, VirtualSRLI/VirtualSRL → VirtualSRL
        0x02 => 0, // VirtualAdvice → RangeCheck
        0x22 => switch (funct3) { // Virtual assert
            1 => 16, // VirtualAssertValidDiv0 → ValidDiv0
            2 => 17, // VirtualAssertHalfwordAlignment → HalfwordAlignment
            3 => 18, // VirtualAssertWordAlignment → WordAlignment
            else => 6, // VirtualAssertEQ → Equal
        },
        0x42 => 19, // VirtualZeroExtendWord → LowerHalfWord
        0x62 => 15, // VirtualAssertValidUnsignedRemainder → ValidUnsignedRemainder
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
//
// Two-phase P/Q prefix-suffix split (matches upstream Jolt):
// Phase 1: Operates on sqrt(T)-sized P (prefix eq) and Q (suffix-folded inc) arrays
//   p(t) = Σ_j [P_r2·Q_r2 + γ·P_r4·Q_r4 + γ²·P_s4·Q_s4 + γ³·P_s5·Q_s5]
//   Runs for prefix_n_vars rounds.
// Phase 2: Materializes suffix-sized ram_inc, rd_inc, eq_ram, eq_rd arrays
//   p(t) = Σ_j [ram_inc·eq_ram + γ²·rd_inc·eq_rd]
//   Runs for suffix_n_vars rounds.
fn IncClaimReductionProver(comptime F: type) type {
    return struct {
        const Self = @This();
        const Phase = enum { phase1, phase2 };

        phase: Phase,
        // Phase 1 state: prefix eq tables (P) and suffix-folded polys (Q)
        // Indices: [0]=r_stage2, [1]=r_stage4, [2]=s_stage4, [3]=s_stage5
        P: [4][]F, // prefix eq tables, prefix_len each
        Q: [4][]F, // suffix-folded inc polys, prefix_len each
        eq_hi: [4][]F, // suffix eq tables, suffix_len each (kept for Phase 2 transition)
        p1_current_len: usize, // Phase 1 current len (prefix_len → prefix_len/2 → ...)
        challenges: []F, // pre-allocated for prefix_n_vars challenges
        num_challenges: usize,

        // Phase 2 state: suffix-sized dense arrays
        ram_inc: []F,
        rd_inc: []F,
        eq_ram: []F,
        eq_rd: []F,
        p2_current_len: usize,

        // Shared
        gamma: F,
        gamma_sqr: F,
        gamma_cub: F,
        prefix_n_vars: usize,
        suffix_n_vars: usize,
        n_vars: usize,
        /// Caller-owned trace; must outlive Phase 1→2 transition (not accessed after).
        trace: *const ExecutionTrace,
        /// Original BE opening points (caller-owned); must outlive Phase 1→2 transition.
        points_be: [4][]const F,
        allocator: Allocator,
        pool: ?*ThreadPool = null,
        gpu: ?*GpuPolyOps = null,

        /// Compute scalar MLE: eq(a, b) = Π_i (a_i·b_i + (1-a_i)·(1-b_i))
        fn computeMle(a: []const F, b: []const F) F {
            var result = F.one();
            for (0..a.len) |i| {
                const prod = a[i].mul(b[i]);
                const sum = a[i].add(b[i]);
                // a·b + (1-a)·(1-b) = 2·a·b + 1 - a - b
                result = result.mul(prod.add(prod).add(F.one()).sub(sum));
            }
            return result;
        }

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
            const prefix_n_vars = n_vars / 2;
            const suffix_n_vars = n_vars - prefix_n_vars;
            const prefix_len: usize = @as(usize, 1) << @intCast(prefix_n_vars);
            const suffix_len: usize = @as(usize, 1) << @intCast(suffix_n_vars);

            const points_be = [4][]const F{
                r_cycle_stage2, r_cycle_stage4, s_cycle_stage4, s_cycle_stage5,
            };

            // For each point, split into lo (prefix) and hi (suffix) BE halves,
            // reverse each to LE, then compute eq tables.
            // BE: [0..suffix_n_vars] = hi (MSB), [suffix_n_vars..n_vars] = lo (LSB)
            // LE lo = reverse(be_lo), LE hi = reverse(be_hi)
            var P: [4][]F = undefined;
            var eq_hi: [4][]F = undefined;

            var rev_lo = try allocator.alloc(F, prefix_n_vars);
            defer allocator.free(rev_lo);
            var rev_hi = try allocator.alloc(F, suffix_n_vars);
            defer allocator.free(rev_hi);

            for (0..4) |i| {
                // LE lo: reverse of BE[suffix_n_vars..n_vars]
                for (0..prefix_n_vars) |k| {
                    rev_lo[k] = points_be[i][n_vars - 1 - k];
                }
                P[i] = try computeEqTableParallel(F, allocator, rev_lo, prefix_n_vars, pool);

                // LE hi: reverse of BE[0..suffix_n_vars]
                for (0..suffix_n_vars) |k| {
                    rev_hi[k] = points_be[i][suffix_n_vars - 1 - k];
                }
                eq_hi[i] = try computeEqTableParallel(F, allocator, rev_hi, suffix_n_vars, pool);
            }

            // Q[i][x_lo] = Σ_{x_hi} Inc(x_lo + x_hi << prefix_n_vars) * eq_hi[i][x_hi]
            // Q[0], Q[1] for RamInc at points 0,1
            // Q[2], Q[3] for RdInc at points 2,3
            var Q: [4][]F = undefined;
            for (0..4) |i| {
                Q[i] = try allocator.alloc(F, prefix_len);
            }

            const QCtx = struct {
                steps: []const tracer.TraceStep,
                eq_hi: [4][]const F,
                Q: [4][]F,
                prefix_n_vars: u6,
                suffix_len: usize,
            };
            const q_ctx = QCtx{
                .steps = trace.steps.items,
                .eq_hi = .{ eq_hi[0], eq_hi[1], eq_hi[2], eq_hi[3] },
                .Q = Q,
                .prefix_n_vars = @intCast(prefix_n_vars),
                .suffix_len = suffix_len,
            };
            const qFn = struct {
                fn f(c: QCtx, x_lo: usize) void {
                    var acc: [4]F = .{ F.zero(), F.zero(), F.zero(), F.zero() };
                    for (0..c.suffix_len) |x_hi| {
                        const x = x_lo + (x_hi << c.prefix_n_vars);
                        const step = c.steps[x];

                        // RamInc
                        var ram_inc = F.zero();
                        if (step.is_memory_write) {
                            const mem_post: i128 = @intCast(step.memory_value orelse 0);
                            const mem_pre: i128 = @intCast(step.memory_pre_value orelse 0);
                            ram_inc = fieldFromI128(F, mem_post - mem_pre);
                        }
                        // RdInc
                        var rd_inc_val = F.zero();
                        if (!step.is_noop and step.rd_written and step.rd_index != 0) {
                            rd_inc_val = F.fromU64(step.rd_value).sub(F.fromU64(step.rd_pre_value));
                        }

                        acc[0] = acc[0].add(c.eq_hi[0][x_hi].mul(ram_inc));
                        acc[1] = acc[1].add(c.eq_hi[1][x_hi].mul(ram_inc));
                        acc[2] = acc[2].add(c.eq_hi[2][x_hi].mul(rd_inc_val));
                        acc[3] = acc[3].add(c.eq_hi[3][x_hi].mul(rd_inc_val));
                    }
                    c.Q[0][x_lo] = acc[0];
                    c.Q[1][x_lo] = acc[1];
                    c.Q[2][x_lo] = acc[2];
                    c.Q[3][x_lo] = acc[3];
                }
            }.f;

            if (pool) |p| {
                p.parallelFor(prefix_len, q_ctx, qFn);
            } else {
                for (0..prefix_len) |x_lo| qFn(q_ctx, x_lo);
            }

            const challenges_buf = try allocator.alloc(F, prefix_n_vars);
            @memset(challenges_buf, F.zero());

            return Self{
                .phase = .phase1,
                .P = P,
                .Q = Q,
                .eq_hi = eq_hi,
                .p1_current_len = prefix_len,
                .challenges = challenges_buf,
                .num_challenges = 0,
                .ram_inc = &[_]F{},
                .rd_inc = &[_]F{},
                .eq_ram = &[_]F{},
                .eq_rd = &[_]F{},
                .p2_current_len = 0,
                .gamma = gamma,
                .gamma_sqr = gamma.mul(gamma),
                .gamma_cub = gamma.mul(gamma).mul(gamma),
                .prefix_n_vars = prefix_n_vars,
                .suffix_n_vars = suffix_n_vars,
                .n_vars = n_vars,
                .trace = trace,
                .points_be = points_be,
                .allocator = allocator,
                .pool = pool,
            };
        }

        pub fn deinit(self: *Self) void {
            switch (self.phase) {
                .phase1 => {
                    for (0..4) |i| {
                        self.allocator.free(self.P[i]);
                        self.allocator.free(self.Q[i]);
                        self.allocator.free(self.eq_hi[i]);
                    }
                    self.allocator.free(self.challenges);
                },
                .phase2 => {
                    self.allocator.free(self.ram_inc);
                    self.allocator.free(self.rd_inc);
                    self.allocator.free(self.eq_ram);
                    self.allocator.free(self.eq_rd);
                },
            }
        }

        /// Phase 1 round polynomial: P·Q products with gamma weighting
        fn computeRoundPolyPhase1(self: *Self) [3]F {
            const half = self.p1_current_len / 2;

            const Ctx = struct {
                P: [4][]const F,
                Q: [4][]const F,
                gamma: F,
                gamma_sqr: F,
                gamma_cub: F,
            };
            const ctx = Ctx{
                .P = .{ self.P[0], self.P[1], self.P[2], self.P[3] },
                .Q = .{ self.Q[0], self.Q[1], self.Q[2], self.Q[3] },
                .gamma = self.gamma,
                .gamma_sqr = self.gamma_sqr,
                .gamma_cub = self.gamma_cub,
            };

            const mapFn = struct {
                fn f(c: Ctx, start: usize, end: usize) [3]F {
                    var e0 = F.zero();
                    var e1 = F.zero();
                    var e2 = F.zero();
                    const weights = [4]F{ F.one(), c.gamma, c.gamma_sqr, c.gamma_cub };

                    for (start..end) |j| {
                        inline for (0..4) |k| {
                            const p0 = c.P[k][2 * j];
                            const p1 = c.P[k][2 * j + 1];
                            const q0 = c.Q[k][2 * j];
                            const q1 = c.Q[k][2 * j + 1];

                            e0 = e0.add(weights[k].mul(p0.mul(q0)));
                            e1 = e1.add(weights[k].mul(p1.mul(q1)));

                            const p2 = p1.add(p1).sub(p0);
                            const q2 = q1.add(q1).sub(q0);
                            e2 = e2.add(weights[k].mul(p2.mul(q2)));
                        }
                    }
                    return [3]F{ e0, e1, e2 };
                }
            }.f;

            const reduceFn = struct {
                fn f(a: [3]F, b: [3]F) [3]F {
                    return [3]F{ a[0].add(b[0]), a[1].add(b[1]), a[2].add(b[2]) };
                }
            }.f;

            if (self.pool) |p| {
                return p.parallelReduce([3]F, half, [3]F{ F.zero(), F.zero(), F.zero() }, ctx, mapFn, reduceFn);
            }
            return mapFn(ctx, 0, half);
        }

        /// Phase 2 round polynomial: ram_inc·eq_ram + γ²·rd_inc·eq_rd
        fn computeRoundPolyPhase2(self: *Self) [3]F {
            const half = self.p2_current_len / 2;

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
                        const eq_r0 = c.eq_ram[2 * j];
                        const eq_r1 = c.eq_ram[2 * j + 1];
                        const rd0 = c.rd_inc[2 * j];
                        const rd1 = c.rd_inc[2 * j + 1];
                        const eq_d0 = c.eq_rd[2 * j];
                        const eq_d1 = c.eq_rd[2 * j + 1];

                        e0 = e0.add(ram0.mul(eq_r0).add(c.gamma_sqr.mul(rd0.mul(eq_d0))));
                        e1 = e1.add(ram1.mul(eq_r1).add(c.gamma_sqr.mul(rd1.mul(eq_d1))));

                        const ram2 = ram1.add(ram1).sub(ram0);
                        const eq_r2 = eq_r1.add(eq_r1).sub(eq_r0);
                        const rd2 = rd1.add(rd1).sub(rd0);
                        const eq_d2 = eq_d1.add(eq_d1).sub(eq_d0);
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

            if (self.pool) |p| {
                return p.parallelReduce([3]F, half, [3]F{ F.zero(), F.zero(), F.zero() }, ctx, mapFn, reduceFn);
            }
            return mapFn(ctx, 0, half);
        }

        pub fn computeRoundPoly(self: *Self) [3]F {
            return switch (self.phase) {
                .phase1 => self.computeRoundPolyPhase1(),
                .phase2 => self.computeRoundPolyPhase2(),
            };
        }

        /// Transition from Phase 1 to Phase 2
        fn transitionToPhase2(self: *Self, last_challenge: F) !void {
            // Store the final challenge
            self.challenges[self.num_challenges] = last_challenge;
            self.num_challenges += 1;

            const prefix_n_vars = self.prefix_n_vars;
            const suffix_n_vars = self.suffix_n_vars;
            const n_vars = self.n_vars;
            const suffix_len: usize = @as(usize, 1) << @intCast(suffix_n_vars);
            const prefix_len: usize = @as(usize, 1) << @intCast(prefix_n_vars);

            // Compute eq_prefix table: eq(challenges, x_lo) for each x_lo
            const eq_prefix = try computeEqTableParallel(F, self.allocator, self.challenges[0..prefix_n_vars], prefix_n_vars, self.pool);
            defer self.allocator.free(eq_prefix);

            // Compute scalar MLE: eq(challenges_LE → BE, point_lo_BE) for each point
            // challenges are in LE order; point_lo_LE = reverse(point_be[suffix_n_vars..])
            var point_lo_le = try self.allocator.alloc(F, prefix_n_vars);
            defer self.allocator.free(point_lo_le);

            var eq_prefix_scalars: [4]F = undefined;
            for (0..4) |i| {
                for (0..prefix_n_vars) |k| {
                    point_lo_le[k] = self.points_be[i][n_vars - 1 - k];
                }
                eq_prefix_scalars[i] = computeMle(self.challenges[0..prefix_n_vars], point_lo_le);
            }

            // Build combined eq arrays: eq_ram[x_hi] = scalar_r2·eq_hi_r2[x_hi] + γ·scalar_r4·eq_hi_r4[x_hi]
            const eq_ram_arr = try self.allocator.alloc(F, suffix_len);
            errdefer self.allocator.free(eq_ram_arr);
            const eq_rd_arr = try self.allocator.alloc(F, suffix_len);
            errdefer self.allocator.free(eq_rd_arr);

            const scale_r2 = eq_prefix_scalars[0];
            const scale_r4 = eq_prefix_scalars[1];
            const scale_s4 = eq_prefix_scalars[2];
            const scale_s5 = eq_prefix_scalars[3];

            const EqP2Ctx = struct {
                eq_hi_0: []const F,
                eq_hi_1: []const F,
                eq_hi_2: []const F,
                eq_hi_3: []const F,
                eq_ram_out: []F,
                eq_rd_out: []F,
                scale_r2: F,
                scale_r4: F,
                scale_s4: F,
                scale_s5: F,
                gamma: F,
            };
            const eq_ctx = EqP2Ctx{
                .eq_hi_0 = self.eq_hi[0],
                .eq_hi_1 = self.eq_hi[1],
                .eq_hi_2 = self.eq_hi[2],
                .eq_hi_3 = self.eq_hi[3],
                .eq_ram_out = eq_ram_arr,
                .eq_rd_out = eq_rd_arr,
                .scale_r2 = scale_r2,
                .scale_r4 = scale_r4,
                .scale_s4 = scale_s4,
                .scale_s5 = scale_s5,
                .gamma = self.gamma,
            };
            const eqP2Fn = struct {
                fn f(c: EqP2Ctx, x_hi: usize) void {
                    c.eq_ram_out[x_hi] = c.scale_r2.mul(c.eq_hi_0[x_hi]).add(c.gamma.mul(c.scale_r4.mul(c.eq_hi_1[x_hi])));
                    c.eq_rd_out[x_hi] = c.scale_s4.mul(c.eq_hi_2[x_hi]).add(c.gamma.mul(c.scale_s5.mul(c.eq_hi_3[x_hi])));
                }
            }.f;
            if (self.pool) |p| {
                p.parallelFor(suffix_len, eq_ctx, eqP2Fn);
            } else {
                for (0..suffix_len) |x_hi| eqP2Fn(eq_ctx, x_hi);
            }

            // Materialize ram_inc and rd_inc by folding trace over prefix dimension
            const ram_inc_arr = try self.allocator.alloc(F, suffix_len);
            errdefer self.allocator.free(ram_inc_arr);
            const rd_inc_arr = try self.allocator.alloc(F, suffix_len);
            errdefer self.allocator.free(rd_inc_arr);

            const IncP2Ctx = struct {
                steps: []const tracer.TraceStep,
                eq_prefix: []const F,
                ram_inc_out: []F,
                rd_inc_out: []F,
                prefix_len: usize,
                prefix_n_vars: u6,
            };
            const inc_ctx = IncP2Ctx{
                .steps = self.trace.steps.items,
                .eq_prefix = eq_prefix,
                .ram_inc_out = ram_inc_arr,
                .rd_inc_out = rd_inc_arr,
                .prefix_len = prefix_len,
                .prefix_n_vars = @intCast(prefix_n_vars),
            };
            const incP2Fn = struct {
                fn f(c: IncP2Ctx, x_hi: usize) void {
                    var acc_ram = F.zero();
                    var acc_rd = F.zero();
                    for (0..c.prefix_len) |x_lo| {
                        const x = x_lo + (x_hi << c.prefix_n_vars);
                        const step = c.steps[x];
                        const eq_val = c.eq_prefix[x_lo];

                        if (step.is_memory_write) {
                            const mem_post: i128 = @intCast(step.memory_value orelse 0);
                            const mem_pre: i128 = @intCast(step.memory_pre_value orelse 0);
                            acc_ram = acc_ram.add(eq_val.mul(fieldFromI128(F, mem_post - mem_pre)));
                        }
                        if (!step.is_noop and step.rd_written and step.rd_index != 0) {
                            acc_rd = acc_rd.add(eq_val.mul(F.fromU64(step.rd_value).sub(F.fromU64(step.rd_pre_value))));
                        }
                    }
                    c.ram_inc_out[x_hi] = acc_ram;
                    c.rd_inc_out[x_hi] = acc_rd;
                }
            }.f;
            if (self.pool) |p| {
                p.parallelFor(suffix_len, inc_ctx, incP2Fn);
            } else {
                for (0..suffix_len) |x_hi| incP2Fn(inc_ctx, x_hi);
            }

            // Free Phase 1 arrays
            for (0..4) |i| {
                self.allocator.free(self.P[i]);
                self.allocator.free(self.Q[i]);
                self.allocator.free(self.eq_hi[i]);
            }
            self.allocator.free(self.challenges);

            // Set Phase 2 state
            self.ram_inc = ram_inc_arr;
            self.rd_inc = rd_inc_arr;
            self.eq_ram = eq_ram_arr;
            self.eq_rd = eq_rd_arr;
            self.p2_current_len = suffix_len;
            self.phase = .phase2;
        }

        pub fn bindChallenge(self: *Self, r: F) !void {
            const bindOne = struct {
                fn f(arr: []F, h: usize, challenge: F) void {
                    for (0..h) |j| {
                        arr[j] = arr[2 * j].add(challenge.mul(arr[2 * j + 1].sub(arr[2 * j])));
                    }
                }
            }.f;

            switch (self.phase) {
                .phase1 => {
                    // Check if we should transition (P has length 2 → last Phase 1 round)
                    if (self.p1_current_len == 2) {
                        try self.transitionToPhase2(r);
                        return;
                    }

                    // Normal Phase 1 bind: bind all 8 P/Q arrays
                    const half = self.p1_current_len / 2;
                    self.challenges[self.num_challenges] = r;
                    self.num_challenges += 1;

                    if (self.gpu) |gpu| {
                        if (half >= 16384) {
                            inline for (0..4) |i| {
                                gpu.polyBindLow(self.P[i][0 .. half * 2], r, self.P[i][0..half]) catch bindOne(self.P[i], half, r);
                                gpu.polyBindLow(self.Q[i][0 .. half * 2], r, self.Q[i][0..half]) catch bindOne(self.Q[i], half, r);
                            }
                        } else {
                            inline for (0..4) |i| {
                                bindOne(self.P[i], half, r);
                                bindOne(self.Q[i], half, r);
                            }
                        }
                    } else if (self.pool) |pool| {
                        const arrays = [8][]F{ self.P[0], self.P[1], self.P[2], self.P[3], self.Q[0], self.Q[1], self.Q[2], self.Q[3] };
                        const Ctx = struct { arrs: [8][]F, half: usize, r: F };
                        const ctx = Ctx{ .arrs = arrays, .half = half, .r = r };
                        pool.parallelForForce(8, ctx, struct {
                            fn f(c: Ctx, idx: usize) void {
                                bindOne(c.arrs[idx], c.half, c.r);
                            }
                        }.f);
                    } else {
                        inline for (0..4) |i| {
                            bindOne(self.P[i], half, r);
                            bindOne(self.Q[i], half, r);
                        }
                    }
                    self.p1_current_len = half;
                },
                .phase2 => {
                    const half = self.p2_current_len / 2;
                    if (self.gpu) |gpu| {
                        if (half >= 16384) {
                            gpu.polyBindLow(self.ram_inc[0 .. half * 2], r, self.ram_inc[0..half]) catch bindOne(self.ram_inc, half, r);
                            gpu.polyBindLow(self.rd_inc[0 .. half * 2], r, self.rd_inc[0..half]) catch bindOne(self.rd_inc, half, r);
                            gpu.polyBindLow(self.eq_ram[0 .. half * 2], r, self.eq_ram[0..half]) catch bindOne(self.eq_ram, half, r);
                            gpu.polyBindLow(self.eq_rd[0 .. half * 2], r, self.eq_rd[0..half]) catch bindOne(self.eq_rd, half, r);
                        } else {
                            bindOne(self.ram_inc, half, r);
                            bindOne(self.rd_inc, half, r);
                            bindOne(self.eq_ram, half, r);
                            bindOne(self.eq_rd, half, r);
                        }
                    } else if (self.pool) |pool| {
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
                    self.p2_current_len = half;
                },
            }
        }

        pub fn openingClaims(self: *const Self) struct { ram_inc: F, rd_inc: F } {
            std.debug.assert(self.phase == .phase2);
            return .{
                .ram_inc = self.ram_inc[0],
                .rd_inc = self.rd_inc[0],
            };
        }
    };
}

// =============================================================================
// HammingBooleanity Sumcheck Instance (Instance 1)
// =============================================================================
// Proves: Sigma_j eq(r_cycle, j) * (H(j)^2 - H(j)) = 0
// Degree 3: eq is linear * (H^2 - H is quadratic)
//
// Split-eq optimization: replaces T-sized eq table with sqrt(T)-sized E_lo/E_hi.
// Phase 1: eq factored as eq_lo(x_lo) * eq_hi(x_hi), bind eq_lo + H for prefix_n_vars rounds.
// Phase 2: merge eq_lo scalar into eq_hi, bind merged eq + H for suffix_n_vars rounds.
fn HammingBooleanityProver(comptime F: type) type {
    return struct {
        const Self = @This();
        const Phase = enum { phase1, phase2 };

        H: []F,
        phase: Phase,
        // Phase 1: split eq tables
        eq_lo: []F, // prefix eq, prefix_len → prefix_len/2 → ... → 1
        eq_hi: []F, // suffix eq, constant during Phase 1 (freed at transition)
        // Phase 2: merged eq (reuses eq_hi allocation scaled by eq_lo[0])
        eq: []F,
        current_len: usize, // H length
        prefix_current_len: usize, // eq_lo length (>0 in Phase 1)
        suffix_len: usize,
        allocator: Allocator,
        pool: ?*ThreadPool = null,
        gpu: ?*GpuPolyOps = null,

        pub fn init(
            allocator: Allocator,
            trace: *const ExecutionTrace,
            r_cycle: []const F,
            pool: ?*ThreadPool,
        ) !Self {
            const T = trace.steps.items.len;
            const n_vars = std.math.log2_int(usize, T);
            const prefix_n_vars = n_vars / 2;
            const suffix_n_vars = n_vars - prefix_n_vars;
            const prefix_len: usize = @as(usize, 1) << @intCast(prefix_n_vars);
            const suffix_len: usize = @as(usize, 1) << @intCast(suffix_n_vars);

            const H_arr = try allocator.alloc(F, T);
            const HInitCtx = struct {
                steps: []const tracer.TraceStep,
                H_out: []F,
            };
            const h_init_ctx = HInitCtx{ .steps = trace.steps.items, .H_out = H_arr };
            const hInitFn = struct {
                fn f(c: HInitCtx, j: usize) void {
                    const step = c.steps[j];
                    if (step.memory_addr) |addr| {
                        c.H_out[j] = if (addr != 0) F.one() else F.zero();
                    } else {
                        c.H_out[j] = F.zero();
                    }
                }
            }.f;
            if (pool) |p| {
                p.parallelFor(T, h_init_ctx, hInitFn);
            } else {
                for (0..T) |j| hInitFn(h_init_ctx, j);
            }

            // r_cycle is in BE order; reverse for LE
            var r_cycle_rev = try allocator.alloc(F, n_vars);
            defer allocator.free(r_cycle_rev);
            for (0..n_vars) |i| r_cycle_rev[i] = r_cycle[n_vars - 1 - i];

            // Split eq: E_lo over first prefix_n_vars LE vars, E_hi over remaining
            const eq_lo = try computeEqTableParallel(F, allocator, r_cycle_rev[0..prefix_n_vars], prefix_n_vars, pool);
            const eq_hi = try computeEqTableParallel(F, allocator, r_cycle_rev[prefix_n_vars..n_vars], suffix_n_vars, pool);

            return Self{
                .H = H_arr,
                .phase = .phase1,
                .eq_lo = eq_lo,
                .eq_hi = eq_hi,
                .eq = &[_]F{},
                .current_len = T,
                .prefix_current_len = prefix_len,
                .suffix_len = suffix_len,
                .allocator = allocator,
                .pool = pool,
            };
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.H);
            switch (self.phase) {
                .phase1 => {
                    self.allocator.free(self.eq_lo);
                    self.allocator.free(self.eq_hi);
                },
                .phase2 => {
                    self.allocator.free(self.eq);
                },
            }
        }

        /// Phase 1: double loop with factored eq = eq_lo(x_lo) * eq_hi(x_hi)
        fn computeRoundPolyPhase1(self: *Self) [4]F {
            const half_lo = self.prefix_current_len / 2;
            const suffix_len = self.suffix_len;

            const Ctx = struct {
                H: []const F,
                eq_lo: []const F,
                eq_hi: []const F,
                half_lo: usize,
                suffix_len: usize,
            };
            const ctx = Ctx{
                .H = self.H,
                .eq_lo = self.eq_lo,
                .eq_hi = self.eq_hi,
                .half_lo = half_lo,
                .suffix_len = suffix_len,
            };

            // Parallelize over suffix_len (outer loop over j_outer).
            // Index layout: H is a flat T-sized array with LE indexing. The flat index
            // x = x_lo + x_hi * prefix_len, where x_lo ∈ [0, prefix_len) is the
            // fastest-varying (prefix) dimension. We pair indices (2j, 2j+1) which
            // toggle bit 0 (the first sumcheck variable). Decomposing j = j_inner +
            // j_outer * half_lo gives: 2j = 2*j_inner + j_outer*prefix_len, keeping
            // the j_outer (suffix) block constant within each pair — so eq_hi[j_outer]
            // factors out correctly.
            //
            // Cache locality note: H accesses stride by prefix_len between j_outer blocks
            // (non-contiguous). Swapping loop order (parallel over prefix, sequential over
            // suffix) would improve locality but would prevent eq_hi factorization. The
            // current order is chosen because the eq_hi factorization saves a multiply per
            // pair, which dominates the cache cost for typical sizes.
            const mapFn = struct {
                fn f(c: Ctx, start: usize, end: usize) [4]F {
                    var e0 = F.zero();
                    var e1 = F.zero();
                    var e2 = F.zero();
                    var e3 = F.zero();

                    for (start..end) |j_outer| {
                        const eq_hi_val = c.eq_hi[j_outer];
                        for (0..c.half_lo) |j_inner| {
                            const j = j_inner + j_outer * c.half_lo;
                            const h0 = c.H[2 * j];
                            const h1 = c.H[2 * j + 1];
                            const h_delta = h1.sub(h0);

                            const eq_lo_0 = c.eq_lo[2 * j_inner];
                            const eq_lo_1 = c.eq_lo[2 * j_inner + 1];
                            const eq0 = eq_lo_0.mul(eq_hi_val);
                            const eq1 = eq_lo_1.mul(eq_hi_val);
                            const e_delta = eq1.sub(eq0);

                            e0 = e0.add(eq0.mul(h0.mul(h0).sub(h0)));
                            e1 = e1.add(eq1.mul(h1.mul(h1).sub(h1)));

                            const h_at_2 = h1.add(h_delta);
                            const e_at_2 = eq1.add(e_delta);
                            e2 = e2.add(e_at_2.mul(h_at_2.mul(h_at_2).sub(h_at_2)));

                            const h_at_3 = h_at_2.add(h_delta);
                            const e_at_3 = e_at_2.add(e_delta);
                            e3 = e3.add(e_at_3.mul(h_at_3.mul(h_at_3).sub(h_at_3)));
                        }
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
                return pool.parallelReduce([4]F, suffix_len, [4]F{ F.zero(), F.zero(), F.zero(), F.zero() }, ctx, mapFn, reduceFn);
            }
            return mapFn(ctx, 0, suffix_len);
        }

        /// Phase 2: standard flat loop with merged eq
        fn computeRoundPolyPhase2(self: *Self) [4]F {
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

                        const h_at_2 = h1.add(h_delta);
                        const e_at_2 = eq1.add(e_delta);
                        e2 = e2.add(e_at_2.mul(h_at_2.mul(h_at_2).sub(h_at_2)));

                        const h_at_3 = h_at_2.add(h_delta);
                        const e_at_3 = e_at_2.add(e_delta);
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

        pub fn computeRoundPoly(self: *Self) [4]F {
            return switch (self.phase) {
                .phase1 => self.computeRoundPolyPhase1(),
                .phase2 => self.computeRoundPolyPhase2(),
            };
        }

        pub fn bindChallenge(self: *Self, r: F) void {
            const bindOne = struct {
                fn f(arr: []F, h: usize, challenge: F) void {
                    for (0..h) |j| {
                        arr[j] = arr[2 * j].add(challenge.mul(arr[2 * j + 1].sub(arr[2 * j])));
                    }
                }
            }.f;

            switch (self.phase) {
                .phase1 => {
                    const half = self.current_len / 2;
                    const half_lo = self.prefix_current_len / 2;

                    // eq_lo is tiny (sqrt(T)), bind inline. H is large — use GPU if available.
                    bindOne(self.eq_lo, half_lo, r);
                    if (self.gpu) |gpu| {
                        if (half >= 16384) {
                            gpu.polyBindLow(self.H[0 .. half * 2], r, self.H[0..half]) catch bindOne(self.H, half, r);
                        } else {
                            bindOne(self.H, half, r);
                        }
                    } else {
                        bindOne(self.H, half, r);
                    }
                    self.current_len = half;
                    self.prefix_current_len = half_lo;

                    // Transition to Phase 2 when eq_lo reaches length 1
                    if (half_lo == 1) {
                        const eq_lo_scalar = self.eq_lo[0];
                        self.allocator.free(self.eq_lo);

                        // Merge: eq[j_hi] = eq_lo_scalar * eq_hi[j_hi]
                        const eq_merged = self.eq_hi;
                        const ScaleCtx = struct { eq: []F, scalar: F };
                        const scale_ctx = ScaleCtx{ .eq = eq_merged, .scalar = eq_lo_scalar };
                        const scaleFn = struct {
                            fn f(c: ScaleCtx, j: usize) void {
                                c.eq[j] = c.scalar.mul(c.eq[j]);
                            }
                        }.f;
                        if (self.pool) |pool| {
                            pool.parallelFor(self.suffix_len, scale_ctx, scaleFn);
                        } else {
                            for (0..self.suffix_len) |j| scaleFn(scale_ctx, j);
                        }
                        self.eq = eq_merged;
                        self.phase = .phase2;
                    }
                },
                .phase2 => {
                    const half = self.current_len / 2;
                    if (self.gpu) |gpu| {
                        if (half >= 16384) {
                            gpu.polyBindLow(self.H[0 .. half * 2], r, self.H[0..half]) catch bindOne(self.H, half, r);
                            gpu.polyBindLow(self.eq[0 .. half * 2], r, self.eq[0..half]) catch bindOne(self.eq, half, r);
                        } else {
                            bindOne(self.H, half, r);
                            bindOne(self.eq, half, r);
                        }
                    } else if (self.pool) |pool| {
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
                },
            }
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
    const RaPoly = ra_poly_mod.RaPolynomial(F);

    return struct {
        const Self = @This();

        /// In-place MLE bind: arr[j] = arr[2j] + challenge*(arr[2j+1] - arr[2j]) for j < h.
        /// Sequential only (write[j] aliases future read[2j], cannot parallelize within one array).
        fn bindSlice(arr: []F, h: usize, challenge: F) void {
            for (0..h) |j| {
                arr[j] = arr[2 * j].add(challenge.mul(arr[2 * j + 1].sub(arr[2 * j])));
            }
        }

        /// Compressed ra polynomials (u8 indices in round 1, dense after bind)
        ra_polys: []RaPoly,
        /// GruenSplitEq for eq(r_cycle, .) — O(1) bind
        gruen_eq: poly_mod.GruenSplitEqPolynomial(F),
        d: usize,
        current_len: usize,
        allocator: Allocator,
        pool: ?*ThreadPool = null,
        gpu: ?*GpuPolyOps = null,

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

            // u8 indices can represent chunk values up to 255
            std.debug.assert(log_k_chunk <= ra_poly_mod.MAX_LOG_K_CHUNK);

            var ra_polys = try allocator.alloc(RaPoly, d);
            // Track how many ra_polys have been assembled for safe errdefer cleanup
            var ra_polys_assembled: usize = 0;
            errdefer {
                for (ra_polys[0..ra_polys_assembled]) |*rp| rp.deinit(allocator);
                allocator.free(ra_polys);
            }

            // Pre-allocate all d index arrays and eq_tables
            // eq_tables are owned by RaPolynomial (freed on bind/deinit)
            var indices_arr = try allocator.alloc([]?u8, d);
            defer allocator.free(indices_arr);
            var eq_tables = try allocator.alloc([]F, d);
            defer allocator.free(eq_tables); // only frees the pointer array, not contents

            // Track allocation progress for errdefer cleanup (before assembly into RaPolys)
            var indices_allocated: usize = 0;
            var eq_tables_allocated: usize = 0;
            errdefer {
                for (0..eq_tables_allocated) |i| allocator.free(eq_tables[i]);
                for (0..indices_allocated) |i| allocator.free(indices_arr[i]);
            }

            for (0..d) |i| {
                indices_arr[i] = try allocator.alloc(?u8, T);
                indices_allocated += 1;
                var r_chunk_rev = try allocator.alloc(F, log_k_chunk);
                defer allocator.free(r_chunk_rev);
                for (0..log_k_chunk) |ci| r_chunk_rev[ci] = r_addr_chunks[i][log_k_chunk - 1 - ci];
                eq_tables[i] = try computeEqTable(F, allocator, r_chunk_rev, log_k_chunk);
                eq_tables_allocated += 1;
            }

            // Parallel fill: each chunk i is independent
            const RamRaInitCtx = struct {
                steps: []const tracer.TraceStep,
                indices_arr: [][]?u8,
                memory_layout: *const jolt_device.MemoryLayout,
                d: usize,
                log_k_chunk: usize,
                k_chunk: usize,
            };
            const ram_ra_ctx = RamRaInitCtx{
                .steps = trace.steps.items,
                .indices_arr = indices_arr,
                .memory_layout = memory_layout,
                .d = d,
                .log_k_chunk = log_k_chunk,
                .k_chunk = k_chunk,
            };
            const ramRaInitFn = struct {
                fn f(c: RamRaInitCtx, i: usize) void {
                    for (0..c.steps.len) |j| {
                        const step = c.steps[j];
                        if (step.memory_addr) |addr| {
                            if (addr == 0) {
                                c.indices_arr[i][j] = null;
                            } else {
                                const remapped = c.memory_layout.remapAddress(addr);
                                if (remapped) |raddr| {
                                    const chunk_val = extractChunkMSB(raddr, i, c.d, c.log_k_chunk);
                                    if (chunk_val < c.k_chunk) {
                                        c.indices_arr[i][j] = @intCast(chunk_val);
                                    } else {
                                        c.indices_arr[i][j] = null;
                                    }
                                } else {
                                    c.indices_arr[i][j] = null;
                                }
                            }
                        } else {
                            c.indices_arr[i][j] = null;
                        }
                    }
                }
            }.f;
            if (init_pool) |p| {
                p.parallelForForce(d, ram_ra_ctx, ramRaInitFn);
            } else {
                for (0..d) |i| ramRaInitFn(ram_ra_ctx, i);
            }

            // Assemble RaPolynomials (prescales eq_table by scale=1, validates invariants).
            // Ownership of indices_arr[i] and eq_tables[i] transfers to the RaPoly;
            // clear allocation counters so the pre-assembly errdefer won't double-free.
            for (0..d) |i| {
                ra_polys[i] = RaPoly.initRound1(indices_arr[i], eq_tables[i], F.one());
                ra_polys_assembled += 1;
            }
            indices_allocated = 0;
            eq_tables_allocated = 0;

            // r_cycle is in BE order; pass directly to GruenSplitEq
            const gruen_eq = try poly_mod.GruenSplitEqPolynomial(F).init(allocator, r_cycle[0..n_vars]);

            return Self{
                .ra_polys = ra_polys,
                .gruen_eq = gruen_eq,
                .d = d,
                .current_len = T,
                .allocator = allocator,
                .pool = init_pool,
            };
        }

        pub fn deinit(self: *Self) void {
            for (self.ra_polys) |*rp| rp.deinit(self.allocator);
            self.allocator.free(self.ra_polys);
            self.gruen_eq.deinit();
        }

        /// f(x) = eq(x,r) * Prod_i ra_i(x), degree = d + 1
        /// Uses quotient polynomial approach: factor out eq(x,r), compute quotient
        /// q(x) = Prod_i ra_i(x) at Toom points {1, 2, ..., d-1, ∞}, then
        /// reconstruct f(x) via finishMlesProductSumFromEvals.
        /// Returns monomial coefficients.
        pub fn computeRoundPoly(self: *Self, allocator: Allocator, claim: F) ![]F {
            const half = self.current_len / 2;
            const n_toom_evals = self.d;

            // Get factored eq tables from GruenSplitEq
            const eq_tables = self.gruen_eq.getWindowEqTables(self.gruen_eq.current_index, 1);
            const E_out = eq_tables.E_out;
            const E_in = eq_tables.E_in;
            const head_in_bits = eq_tables.head_in_bits;
            const in_mask = (@as(usize, 1) << @intCast(head_in_bits)) -| 1;

            const Ctx = struct {
                ra_polys: []RaPoly,
                E_out: []const F,
                E_in: []const F,
                in_mask: usize,
                head_in_bits: usize,
                d: usize,
                n_toom_evals: usize,
            };
            const ctx = Ctx{
                .ra_polys = self.ra_polys,
                .E_out = E_out,
                .E_in = E_in,
                .in_mask = in_mask,
                .head_in_bits = head_in_bits,
                .d = self.d,
                .n_toom_evals = n_toom_evals,
            };

            const UPA = UnreducedProductAccum;
            const mapFn = struct {
                fn f(c: Ctx, start: usize, end: usize) [MAX_RA_EVALS]F {
                    const MAX_D = 8;
                    var upa_acc: [MAX_RA_EVALS]UPA = .{UPA.zero()} ** MAX_RA_EVALS;
                    for (start..end) |j| {
                        const x_out = j >> @intCast(c.head_in_bits);
                        const x_in = j & c.in_mask;
                        const eq_prefix = (if (x_out < c.E_out.len) c.E_out[x_out] else F.one())
                            .mul(if (x_in < c.E_in.len) c.E_in[x_in] else F.one());

                        var lo: [MAX_D]F = undefined;
                        var delta: [MAX_D]F = undefined;
                        for (0..c.d) |i| {
                            lo[i] = c.ra_polys[i].getBoundCoeff(2 * j);
                            delta[i] = c.ra_polys[i].getBoundCoeff(2 * j + 1).sub(lo[i]);
                        }

                        var cur: [MAX_D]F = undefined;
                        for (0..c.d) |i| cur[i] = lo[i].add(delta[i]);

                        for (0..c.n_toom_evals -| 1) |k| {
                            if (k > 0) {
                                for (0..c.d) |i| cur[i] = cur[i].add(delta[i]);
                            }
                            var product = cur[0];
                            for (1..c.d) |i| product = product.mul(cur[i]);
                            upa_acc[k].addAssign(eq_prefix.mulToProductAccum(product));
                        }

                        var product = delta[0];
                        for (1..c.d) |i| product = product.mul(delta[i]);
                        upa_acc[c.n_toom_evals - 1].addAssign(eq_prefix.mulToProductAccum(product));
                    }
                    var acc: [MAX_RA_EVALS]F = undefined;
                    for (0..MAX_RA_EVALS) |i| acc[i] = upa_acc[i].reduce();
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

            // Scale by current_scalar (accumulated eq from previously bound variables)
            const scalar = self.gruen_eq.current_scalar;
            var toom_evals = try allocator.alloc(F, n_toom_evals);
            defer allocator.free(toom_evals);
            for (0..n_toom_evals) |i| toom_evals[i] = result[i].mul(scalar);

            // Extract r_round and reconstruct full polynomial
            const r_round = self.gruen_eq.tau[self.gruen_eq.current_index - 1];
            return poly_mod.UniPoly(F).finishMlesProductSumFromEvals(allocator, toom_evals, claim, r_round);
        }

        pub fn bindChallenge(self: *Self, r: F) !void {
            const half = self.current_len / 2;

            // After the first bind(), all ra_polys transition from round1→dense simultaneously
            // (they all have the same length T). Check index 0 as representative.
            const all_dense = self.ra_polys.len > 0 and self.ra_polys[0] == .dense;
            if (std.debug.runtime_safety and all_dense) {
                for (self.ra_polys) |rp| std.debug.assert(rp == .dense);
            }

            if (all_dense and self.gpu != null and half >= 16384) {
                // GPU bind: d ra_poly dense arrays + e_out
                const gpu = self.gpu.?;
                for (self.ra_polys) |*rp| {
                    const dense = &rp.dense;
                    const h = dense.current_len / 2;
                    gpu.polyBindLow(dense.coeffs[0 .. h * 2], r, dense.coeffs[0..h]) catch bindSlice(dense.coeffs[0 .. h * 2], h, r);
                    dense.current_len = h;
                }
            } else if (all_dense and self.pool != null) {
                // Parallel bind: d ra_poly dense arrays (eq is O(√T), done separately below)
                const RaBindCtx = struct { ra: []RaPoly, d: usize, half: usize, r: F };
                const ctx = RaBindCtx{ .ra = self.ra_polys, .d = self.d, .half = half, .r = r };
                self.pool.?.parallelForForce(self.d, ctx, struct {
                    fn f(c: RaBindCtx, idx: usize) void {
                        std.debug.assert(c.ra[idx] == .dense);
                        bindSlice(c.ra[idx].dense.coeffs[0 .. c.half * 2], c.half, c.r);
                        c.ra[idx].dense.current_len = c.half;
                    }
                }.f);
            } else {
                // First round (round1→dense transition) or no pool: sequential.
                for (self.ra_polys) |*rp| {
                    try rp.bind(r, self.allocator);
                }
            }

            // GruenSplitEq bind — O(1) instead of O(T/2^round)
            self.gruen_eq.bind(r);

            self.current_len = half;
        }

        pub fn getOpeningClaims(self: *const Self, allocator: Allocator) ![]F {
            var claims = try allocator.alloc(F, self.d);
            for (0..self.d) |i| {
                claims[i] = self.ra_polys[i].finalClaim();
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
        /// γ^i powers for pre-scaling (used in Phase 2 Gruen optimization)
        gamma_powers: []F,
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
        /// null during lazy rounds (first 3 Phase 2 rounds).
        H: ?[][]F,
        /// Current table length for Phase 2 (T, then T/2, etc.)
        phase2_len: usize,
        /// Chunk indices for lazy H evaluation: chunk_indices[i][j] = chunk index for poly i, cycle j
        /// Only allocated during lazy rounds (first 3 Phase 2 rounds).
        chunk_indices: ?[][]u8,
        /// Lookup tables for lazy H evaluation. In round1: tables[0][k] = F_table[k].
        /// In round2: tables[0][k]=(1-r)*F[k], tables[1][k]=r*F[k].
        /// In round3: tables[0..4][k] = (1-r1)(1-r0)F[k], (1-r1)(r0)F[k], etc.
        lazy_tables: [4][]F,
        /// Number of valid tables in lazy_tables (1=round1, 2=round2, 4=round3, 0=dense)
        lazy_num_tables: u8,
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
        gpu: ?*GpuPolyOps = null,

        pub fn init(
            allocator: std.mem.Allocator,
            G_tables: [][]F,
            r_addr_le: []F,
            eq_cycle_table: []F,
            gamma_sq: []F,
            gamma_unsq: []F,
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
                .gamma_powers = gamma_unsq,
                .N = N_val,
                .K = K_val,
                .log_k_chunk = log_k,
                .n_cycle_vars = n_cycle,
                .round = 0,
                .eq_r_r = F.zero(),
                .H = null,
                .phase2_len = 0,
                .chunk_indices = null,
                .lazy_tables = .{ &.{}, &.{}, &.{}, &.{} },
                .lazy_num_tables = 0,
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
            self.allocator.free(self.gamma_powers);
            if (self.H) |ht| {
                for (ht) |h| self.allocator.free(h);
                self.allocator.free(ht);
            }
            if (self.chunk_indices) |ci| {
                for (ci) |c| self.allocator.free(c);
                self.allocator.free(ci);
            }
            for (0..@as(usize, self.lazy_num_tables)) |i| {
                if (self.lazy_tables[i].len > 0) self.allocator.free(self.lazy_tables[i]);
            }
        }

        /// Get the opening claims from the final H state after all sumcheck rounds.
        /// H[i][0] gives ra_i(ρ_addr, ρ_cycle) after all bindings.
        /// H tables are pre-scaled by γ^i, so we unscale via γ^{-i} = (γ^{-1})^i.
        /// This uses 1 inversion + (N-1) muls instead of N inversions.
        pub fn getBooleanityClaims(self: *const Self, allocator: std.mem.Allocator) ![]F {
            const claims = try allocator.alloc(F, self.N);
            dbg("[BOOL_CLAIMS] phase2_len={}, round={}, N={}\n", .{ self.phase2_len, self.round, self.N });
            if (self.H) |ht| {
                // gamma_powers[1] = γ, so γ^{-1} is a single inversion
                const gamma_inv = self.gamma_powers[1].inverse().?;
                var gamma_inv_i = F.one(); // γ^{-0} = 1

                var all_same_claims = true;
                for (0..self.N) |i| {
                    claims[i] = ht[i][0].mul(gamma_inv_i);
                    gamma_inv_i = gamma_inv_i.mul(gamma_inv);
                    if (i < 5 or i >= self.N - 5 or (i >= 28 and i < 34)) {
                        const hbe = claims[i].toBytesBE();
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
        }

        /// Look up the bound coefficient for poly i at position pos in lazy state.
        /// In round1 (1 table): h = tables[0][chunk_indices[i][pos]]
        /// In round2 (2 tables): h = tables[0][ci[i][2*pos]] + tables[1][ci[i][2*pos+1]]
        /// In round3 (4 tables): h = sum of tables[t][ci[i][4*pos+t]] for t=0..3
        inline fn lazyGetCoeff(
            ci: []const []const u8,
            tables: [4][]const F,
            num_tables: u8,
            i: usize,
            pos: usize,
        ) F {
            switch (num_tables) {
                1 => return tables[0][ci[i][pos]],
                2 => return tables[0][ci[i][2 * pos]].add(tables[1][ci[i][2 * pos + 1]]),
                4 => return tables[0][ci[i][4 * pos]].add(tables[1][ci[i][4 * pos + 1]])
                    .add(tables[2][ci[i][4 * pos + 2]]).add(tables[3][ci[i][4 * pos + 3]]),
                else => unreachable,
            }
        }

        fn computePhase2Poly(self: *Self, evals: []F, previous_claim: F) void {
            const half = self.phase2_len / 2;

            if (self.chunk_indices != null) {
                // Lazy evaluation: use chunk_indices + lookup tables
                self.computePhase2PolyLazy(evals, half, previous_claim);
            } else {
                // Dense evaluation: use materialized H arrays (Gruen c/e optimization)
                self.computePhase2PolyDense(evals, half, previous_claim);
            }
        }

        fn computePhase2PolyLazy(self: *Self, evals: []F, half: usize, previous_claim: F) void {
            const ci = self.chunk_indices.?;
            const num_tables = self.lazy_num_tables;
            const tables = [4][]const F{
                self.lazy_tables[0],
                if (num_tables >= 2) self.lazy_tables[1] else &.{},
                if (num_tables >= 4) self.lazy_tables[2] else &.{},
                if (num_tables >= 4) self.lazy_tables[3] else &.{},
            };

            // Use flat eq table for lazy compute, GruenSplitEq for bind only
            const LazyCtx = struct {
                ci: []const []const u8,
                tables: [4][]const F,
                num_tables: u8,
                eq_cycle: []const F,
                gamma_powers: []const F,
                N: usize,
            };
            const ctx = LazyCtx{
                .ci = ci,
                .tables = tables,
                .num_tables = num_tables,
                .eq_cycle = self.eq_cycle,
                .gamma_powers = self.gamma_powers,
                .N = self.N,
            };

            const mapFn = struct {
                fn f(c: LazyCtx, start: usize, end: usize) [4]F {
                    const UPA = UnreducedProductAccum;
                    var c_weighted = F.zero();
                    var e_weighted = F.zero();
                    var eq_sum_0 = F.zero();
                    var eq_sum_1 = F.zero();
                    for (start..end) |j| {
                        const d0 = c.eq_cycle[2 * j];
                        const d1 = c.eq_cycle[2 * j + 1];
                        var acc_c = UPA.zero();
                        var acc_e = UPA.zero();
                        for (0..c.N) |i| {
                            const h0_raw = lazyGetCoeff(c.ci, c.tables, c.num_tables, i, 2 * j);
                            const h1_raw = lazyGetCoeff(c.ci, c.tables, c.num_tables, i, 2 * j + 1);
                            const rho = c.gamma_powers[i];
                            const h0 = rho.mul(h0_raw);
                            const h1 = rho.mul(h1_raw);
                            const b = h1.sub(h0);
                            acc_c.addAssign(h0.mulToProductAccum(h0.sub(rho)));
                            acc_e.addAssign(b.mulToProductAccum(b));
                        }
                        const q_c = acc_c.reduce();
                        const q_e = acc_e.reduce();
                        c_weighted = c_weighted.add(d0.mul(q_c));
                        e_weighted = e_weighted.add(d0.mul(q_e));
                        eq_sum_0 = eq_sum_0.add(d0);
                        eq_sum_1 = eq_sum_1.add(d1);
                    }
                    return [4]F{ c_weighted, e_weighted, eq_sum_0, eq_sum_1 };
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

            const c_weighted = result[0];
            const e_weighted = result[1];
            const eq_eval_0 = result[2];
            const eq_eval_1 = result[3];
            const adjusted_claim = previous_claim.mul(self.eq_r_r.inverse().?);
            const s0_inner = c_weighted;
            const s1_inner = adjusted_claim.sub(c_weighted);
            const eq0_inv = eq_eval_0.inverse().?;
            const eq1_inv = eq_eval_1.inverse().?;
            const q_total_0 = c_weighted.mul(eq0_inv);
            const q_total_1 = s1_inner.mul(eq1_inv);
            const q_total_e = e_weighted.mul(eq0_inv);
            const e_times_2 = q_total_e.add(q_total_e);
            const q_total_2 = q_total_1.add(q_total_1).sub(q_total_0).add(e_times_2);
            const q_total_3 = q_total_2.add(q_total_1).sub(q_total_0).add(e_times_2.add(e_times_2));
            const eq_slope = eq_eval_1.sub(eq_eval_0);
            const eq_eval_2 = eq_eval_1.add(eq_slope);
            const eq_eval_3 = eq_eval_2.add(eq_slope);
            evals[0] = s0_inner.mul(self.eq_r_r);
            evals[1] = s1_inner.mul(self.eq_r_r);
            evals[2] = eq_eval_2.mul(q_total_2).mul(self.eq_r_r);
            evals[3] = eq_eval_3.mul(q_total_3).mul(self.eq_r_r);
        }

        fn computePhase2PolyDense(self: *Self, evals: []F, half: usize, previous_claim: F) void {
            const ht = self.H orelse return;

            // Use flat eq table (BE, from getFullEqTable) for compute,
            // GruenSplitEq for bind only
            const BoolP2Ctx = struct {
                ht: [][]F,
                eq_cycle: []const F,
                gamma_powers: []const F,
                N: usize,
            };
            const ctx = BoolP2Ctx{
                .ht = ht,
                .eq_cycle = self.eq_cycle,
                .gamma_powers = self.gamma_powers,
                .N = self.N,
            };

            const mapFn = struct {
                fn f(c: BoolP2Ctx, start: usize, end: usize) [4]F {
                    const UPA = UnreducedProductAccum;
                    var c_weighted = F.zero();
                    var e_weighted = F.zero();
                    var eq_sum_0 = F.zero();
                    var eq_sum_1 = F.zero();
                    for (start..end) |j| {
                        const d0 = c.eq_cycle[2 * j];
                        const d1 = c.eq_cycle[2 * j + 1];
                        var acc_c = UPA.zero();
                        var acc_e = UPA.zero();
                        for (0..c.N) |i| {
                            const h0 = c.ht[i][2 * j];
                            const h1 = c.ht[i][2 * j + 1];
                            const b = h1.sub(h0);
                            const rho = c.gamma_powers[i];
                            acc_c.addAssign(h0.mulToProductAccum(h0.sub(rho)));
                            acc_e.addAssign(b.mulToProductAccum(b));
                        }
                        const q_c = acc_c.reduce();
                        const q_e = acc_e.reduce();
                        c_weighted = c_weighted.add(d0.mul(q_c));
                        e_weighted = e_weighted.add(d0.mul(q_e));
                        eq_sum_0 = eq_sum_0.add(d0);
                        eq_sum_1 = eq_sum_1.add(d1);
                    }
                    return [4]F{ c_weighted, e_weighted, eq_sum_0, eq_sum_1 };
                }
            }.f;

            const reduceFn = struct {
                fn f(a: [4]F, b: [4]F) [4]F {
                    return .{ a[0].add(b[0]), a[1].add(b[1]), a[2].add(b[2]), a[3].add(b[3]) };
                }
            }.f;

            const result = if (self.pool) |pool|
                pool.parallelReduce([4]F, half, [4]F{ F.zero(), F.zero(), F.zero(), F.zero() }, ctx, mapFn, reduceFn)
            else
                mapFn(ctx, 0, half);

            const c_weighted = result[0];
            const e_weighted = result[1];
            const eq_eval_0 = result[2];
            const eq_eval_1 = result[3];
            const adjusted_claim = previous_claim.mul(self.eq_r_r.inverse().?);
            const s0_inner = c_weighted;
            const s1_inner = adjusted_claim.sub(c_weighted);
            const eq0_inv = eq_eval_0.inverse().?;
            const eq1_inv = eq_eval_1.inverse().?;
            const c = c_weighted.mul(eq0_inv);
            const e = e_weighted.mul(eq0_inv);
            const q_1 = s1_inner.mul(eq1_inv);
            const e_times_2 = e.add(e);
            const q_2 = q_1.add(q_1).sub(c).add(e_times_2);
            const q_3 = q_2.add(q_1).sub(c).add(e_times_2.add(e_times_2));
            const eq_slope = eq_eval_1.sub(eq_eval_0);
            const eq_eval_2 = eq_eval_1.add(eq_slope);
            const eq_eval_3 = eq_eval_2.add(eq_slope);
            evals[0] = s0_inner.mul(self.eq_r_r);
            evals[1] = s1_inner.mul(self.eq_r_r);
            evals[2] = eq_eval_2.mul(q_2).mul(self.eq_r_r);
            evals[3] = eq_eval_3.mul(q_3).mul(self.eq_r_r);
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
                // Phase 2: bind cycle variable
                const half = self.phase2_len / 2;

                const bindOne = struct {
                    fn f(arr: []F, h: usize, challenge: F) void {
                        for (0..h) |j| {
                            arr[j] = arr[2 * j].add(challenge.mul(arr[2 * j + 1].sub(arr[2 * j])));
                        }
                    }
                }.f;

                if (self.chunk_indices != null) {
                    // Lazy state: split tables, don't bind dense arrays
                    const K = self.K;

                    if (self.lazy_num_tables == 1) {
                        // Round1 → Round2: split into tables_0 = (1-r)*table, tables_1 = r*table
                        const old_table = self.lazy_tables[0];
                        const tbl_len = K + 1; // includes sentinel entry
                        const t0 = try self.allocator.alloc(F, tbl_len);
                        const t1 = try self.allocator.alloc(F, tbl_len);
                        for (0..K) |k| {
                            t1[k] = r.mul(old_table[k]);
                            t0[k] = old_table[k].sub(t1[k]);
                        }
                        t0[K] = F.zero(); // sentinel stays zero
                        t1[K] = F.zero();
                        self.allocator.free(old_table);
                        self.lazy_tables[0] = t0;
                        self.lazy_tables[1] = t1;
                        self.lazy_num_tables = 2;
                    } else if (self.lazy_num_tables == 2) {
                        // Round2 → Round3: split each of 2 tables into 2
                        // After Round1→Round2: tables[0]=(1-r0)*F, tables[1]=r0*F
                        // Binding with r1, the position offset within a group of 4 encodes:
                        //   bit 0 = r0 selector, bit 1 = r1 selector
                        // So tables must be ordered by [bit1, bit0] matching position offsets:
                        //   [0]=pos0=(1-r1)(1-r0), [1]=pos1=(1-r1)*r0,
                        //   [2]=pos2=r1*(1-r0),    [3]=pos3=r1*r0
                        const old_t0 = self.lazy_tables[0]; // (1-r0)*F
                        const old_t1 = self.lazy_tables[1]; // r0*F
                        const tbl_len = K + 1;
                        const t_pos0 = try self.allocator.alloc(F, tbl_len); // (1-r1)(1-r0)
                        const t_pos1 = try self.allocator.alloc(F, tbl_len); // (1-r1)*r0
                        const t_pos2 = try self.allocator.alloc(F, tbl_len); // r1*(1-r0)
                        const t_pos3 = try self.allocator.alloc(F, tbl_len); // r1*r0
                        for (0..K) |k| {
                            t_pos2[k] = r.mul(old_t0[k]);           // r1*(1-r0)*F
                            t_pos3[k] = r.mul(old_t1[k]);           // r1*r0*F
                            t_pos0[k] = old_t0[k].sub(t_pos2[k]);  // (1-r1)(1-r0)*F
                            t_pos1[k] = old_t1[k].sub(t_pos3[k]);  // (1-r1)*r0*F
                        }
                        t_pos0[K] = F.zero();
                        t_pos1[K] = F.zero();
                        t_pos2[K] = F.zero();
                        t_pos3[K] = F.zero();
                        self.allocator.free(old_t0);
                        self.allocator.free(old_t1);
                        self.lazy_tables[0] = t_pos0;
                        self.lazy_tables[1] = t_pos1;
                        self.lazy_tables[2] = t_pos2;
                        self.lazy_tables[3] = t_pos3;
                        self.lazy_num_tables = 4;
                    } else {
                        // Round3 → Dense: materialize H[N][T/8] and free indices/tables
                        try self.materializeDense(r);
                    }

                    // Bind eq_cycle only (no H arrays to bind in lazy state)
                    bindOne(self.eq_cycle, half, r);
                } else if (self.H) |ht| {
                    // Dense state: bind H arrays and eq_cycle
                    if (self.gpu) |gpu| {
                        if (half >= 16384) {
                            for (0..self.N) |i| {
                                gpu.polyBindLow(ht[i][0 .. half * 2], r, ht[i][0..half]) catch bindOne(ht[i], half, r);
                            }
                            gpu.polyBindLow(self.eq_cycle[0 .. half * 2], r, self.eq_cycle[0..half]) catch bindOne(self.eq_cycle, half, r);
                        } else {
                            for (0..self.N) |i| {
                                bindOne(ht[i], half, r);
                            }
                            bindOne(self.eq_cycle, half, r);
                        }
                    } else if (self.pool) |pool| {
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

        /// Materialize dense H[N][dense_len] from Round3 (4 tables) + chunk_indices,
        /// binding with challenge r in the process. After this, chunk_indices and
        /// lazy_tables are freed, and self.H is set.
        fn materializeDense(self: *Self, r: F) !void {
            const ci = self.chunk_indices.?;
            const K = self.K;
            const T_orig = ci[0].len;
            // After 3 lazy rounds, the "current length" is T/4 (phase2_len was already
            // halved 2 times; this is the 3rd bind). The dense materialization performs
            // the bind as part of the materialization, producing T/8 entries.
            const dense_len = T_orig / 8;

            // Build 8 combined tables for the 8 original positions within each group.
            // Position offset g within group of 8 has bits [b2, b1, b0]:
            //   b0 = r0 selector, b1 = r1 selector, b2 = r2 selector (current bind)
            // lazy_tables[0..4] are already ordered by position offset (matching bit pattern):
            //   [0]=(1-r1)(1-r0), [1]=(1-r1)*r0, [2]=r1*(1-r0), [3]=r1*r0
            // The current bind with r2 adds the b2 dimension:
            //   combined[g] = ((g & 4) ? r2 : (1-r2)) * lazy_tables[g & 3]
            const tbl_len = K + 1; // includes sentinel
            var combined_tables: [8][]F = undefined;
            for (0..8) |g| {
                combined_tables[g] = try self.allocator.alloc(F, tbl_len);
            }
            errdefer for (combined_tables) |ct| self.allocator.free(ct);

            for (0..K) |k| {
                inline for (0..4) |g| {
                    combined_tables[g + 4][k] = r.mul(self.lazy_tables[g][k]);
                    combined_tables[g][k] = self.lazy_tables[g][k].sub(combined_tables[g + 4][k]);
                }
            }
            // Sentinel entries stay zero
            inline for (0..8) |g| {
                combined_tables[g][K] = F.zero();
            }

            // Materialize dense H arrays, pre-scaled by γ^i for Gruen optimization
            var ht = try self.allocator.alloc([]F, self.N);
            for (0..self.N) |i| {
                ht[i] = try self.allocator.alloc(F, dense_len);
                const idx = ci[i];
                const rho = self.gamma_powers[i]; // γ^i
                for (0..dense_len) |j| {
                    const base = j * 8;
                    var val = F.zero();
                    inline for (0..8) |g| {
                        val = val.add(combined_tables[g][idx[base + g]]);
                    }
                    ht[i][j] = val.mul(rho); // Pre-scale by γ^i
                }
            }

            // Free chunk indices and lazy tables
            for (ci) |c| self.allocator.free(c);
            self.allocator.free(ci);
            self.chunk_indices = null;
            for (0..4) |i| {
                self.allocator.free(self.lazy_tables[i]);
                self.lazy_tables[i] = &.{};
            }
            self.lazy_num_tables = 0;
            for (combined_tables) |ct| self.allocator.free(ct);

            self.H = ht;
        }

        fn transitionToPhase2(self: *Self) !void {
            // F_table now has K entries: F[k] = eq(r_challenges, k) for k ∈ [0, K)
            // Instead of materializing full H[N][T] dense arrays (76MB for N=38, T=65536),
            // store u8 chunk indices (2.5MB) and look up F_table values lazily.
            // This reduces working set from 76MB to ~2.5MB, fitting in L2 cache.
            // After 3 Phase 2 rounds, materialize dense arrays of size T/8 (9.4MB).
            const T_val = @as(usize, 1) << @intCast(self.n_cycle_vars);
            const trace = self.trace;
            const instr_d = self.instruction_d;
            const bc_d = self.bytecode_d;
            const ram_d_val = self.ram_d;
            const K = self.K;

            // Allocate chunk index arrays: N arrays of T u8 entries.
            // Use K as sentinel for "no value" (F.zero()), with tables extended by 1 entry.
            std.debug.assert(K <= 255); // K+1 must fit in u8
            const sentinel: u8 = @intCast(K);
            var ci = try self.allocator.alloc([]u8, self.N);
            errdefer {
                for (ci[0..self.N]) |c| self.allocator.free(c);
                self.allocator.free(ci);
            }
            for (0..self.N) |i| {
                ci[i] = try self.allocator.alloc(u8, T_val);
                @memset(ci[i], sentinel); // default = sentinel (zero value)
            }

            // Parallel chunk_indices build — each j writes to unique ci[*][j] positions
            const CiCtx = struct {
                steps: []const tracer.TraceStep,
                ci_arr: [][]u8,
                pc_map_ptr: *const BytecodePCMapper,
                mem_layout: *const jolt_device.MemoryLayout,
                log_kc: usize,
                instr_d_v: usize,
                bc_d_v: usize,
                ram_d_v: usize,
                K_v: usize,
                sentinel_v: u8,
            };
            const ci_ctx = CiCtx{
                .steps = trace.steps.items,
                .ci_arr = ci,
                .pc_map_ptr = self.pc_map,
                .mem_layout = self.memory_layout,
                .log_kc = self.log_k_chunk,
                .instr_d_v = instr_d,
                .bc_d_v = bc_d,
                .ram_d_v = ram_d_val,
                .K_v = K,
                .sentinel_v = sentinel,
            };
            const ciFn = struct {
                fn f(c: CiCtx, j: usize) void {
                    const step = c.steps[j];
                    // InstructionRa chunks
                    {
                        const lookup_idx = computeLookupIndex(step);
                        for (0..c.instr_d_v) |i| {
                            const shift = c.log_kc * (c.instr_d_v - 1 - i);
                            const mask: u128 = (@as(u128, 1) << @intCast(c.log_kc)) - 1;
                            const chunk_val: usize = @intCast((lookup_idx >> @intCast(shift)) & mask);
                            c.ci_arr[i][j] = if (chunk_val < c.K_v) @intCast(chunk_val) else c.sentinel_v;
                        }
                    }
                    // BytecodeRa chunks
                    {
                        const pc_idx: u64 = @intCast(c.pc_map_ptr.getPCForStep(step));
                        for (0..c.bc_d_v) |i| {
                            const chunk_val = extractChunkMSB(pc_idx, i, c.bc_d_v, c.log_kc);
                            c.ci_arr[c.instr_d_v + i][j] = if (chunk_val < c.K_v) @intCast(chunk_val) else c.sentinel_v;
                        }
                    }
                    // RamRa chunks
                    {
                        if (step.memory_addr) |addr| {
                            if (addr != 0) {
                                if (c.mem_layout.remapAddress(addr)) |raddr| {
                                    for (0..c.ram_d_v) |i| {
                                        const chunk_val = extractChunkMSB(raddr, i, c.ram_d_v, c.log_kc);
                                        c.ci_arr[c.instr_d_v + c.bc_d_v + i][j] = if (chunk_val < c.K_v) @intCast(chunk_val) else c.sentinel_v;
                                    }
                                }
                            }
                        }
                    }
                }
            }.f;
            if (self.pool) |pool| {
                pool.parallelFor(T_val, ci_ctx, ciFn);
            } else {
                for (0..T_val) |j| ciFn(ci_ctx, j);
            }

            self.chunk_indices = ci;
            // Copy F_table as the initial lazy lookup table (round1 state)
            // Extra entry at index K = F.zero() for sentinel "no value" positions
            const lt = try self.allocator.alloc(F, K + 1);
            @memcpy(lt[0..K], self.F_table[0..K]);
            lt[K] = F.zero();
            self.lazy_tables[0] = lt;
            self.lazy_num_tables = 1;

            self.H = null;
            self.phase2_len = T_val;

            // Debug: print F_table values at transition
            if (comptime debug_verbose) {
                dbg("[BOOL_H_INIT] T={}, using lazy chunk indices\n", .{T_val});
                dbg("[BOOL_H_INIT] F_size={}\n", .{self.F_size});
                for (0..@min(self.F_size, 8)) |fi| {
                    const fb = self.F_table[fi].toBytesBE();
                    dbg("[BOOL_H_INIT] F[{}]_LE=[{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}]\n", .{
                        fi, fb[31], fb[30], fb[29], fb[28], fb[27], fb[26], fb[25], fb[24],
                    });
                }
            }

            dbg("[BOOL_PROVER] Phase 1→2 transition: eq_r_r=", .{});
            const err_be = self.eq_r_r.toBytesBE();
            for (0..8) |bi| dbg("{x:0>2}", .{err_be[31 - bi]});
            dbg(", lazy_num_tables={}\n", .{self.lazy_num_tables});
        }

    };
}

// =============================================================================
// LookupsRaVirtual Sumcheck Instance (Instance 4)
// =============================================================================
// Proves: Sigma_c eq(r_cycle, c) * Sum_{v=0}^{N-1} gamma^v * Prod_{j=0}^{M-1} ra_{v*M+j}(c)
// Variables: n_cycle_vars
// Degree: M+1 (product of M linear ra polys * one linear eq)
//
// NOTE: Unlike RamRaVirtualProver, this uses dense []F arrays rather than RaPolynomial
// compression. The gamma scale is baked into the first poly of each virtual batch at
// init time (ra_bound[v*M] *= gamma^v), so the compressed representation would need
// per-index scaling, not a single shared eq_table scale. Future optimization could
// store separate gamma-scaled and unscaled eq_tables per batch.
/// Evaluate the product of 4 linear polynomials at the grid {1, 2, 3, ∞}.
/// Each pair[i] = { p_i(0), p_i(1) }. Returns { P(1), P(2), P(3), P(∞) }
/// where P(x) = p_0(x) * p_1(x) * p_2(x) * p_3(x).
/// Uses Toom-Cook factoring: 10 field muls total.
/// Ported from Jolt's eval_prod_4_assign (mles_product_sum.rs:453-462).
fn evalLinearProd4(comptime F: type, pairs: [4][2]F) [4]F {
    // eval_linear_prod_2_internal on first pair (p[0], p[1]):
    // For linear poly p(x) = p0 + (p1-p0)*x: p(1)=p1, p(∞)=p1-p0, p(2)=p1+(p1-p0)
    const p0_inf = pairs[0][1].sub(pairs[0][0]); // slope of p0
    const p1_inf = pairs[1][1].sub(pairs[1][0]); // slope of p1
    const a1 = pairs[0][1].mul(pairs[1][1]); // A(1) = p0(1)*p1(1)
    const a2 = p0_inf.add(pairs[0][1]).mul(p1_inf.add(pairs[1][1])); // A(2) = p0(2)*p1(2)
    const a_inf = p0_inf.mul(p1_inf); // A(∞) = leading coeff

    // ex2 extrapolation: A(3) = 2*(A(2) + A(∞)) - A(1)  (0 muls, pure adds)
    const a3 = a2.add(a_inf).add(a2.add(a_inf)).sub(a1);

    // eval_linear_prod_2_internal on second pair (p[2], p[3]):
    const p2_inf = pairs[2][1].sub(pairs[2][0]);
    const p3_inf = pairs[3][1].sub(pairs[3][0]);
    const b1 = pairs[2][1].mul(pairs[3][1]);
    const b2 = p2_inf.add(pairs[2][1]).mul(p3_inf.add(pairs[3][1]));
    const b_inf = p2_inf.mul(p3_inf);
    const b3 = b2.add(b_inf).add(b2.add(b_inf)).sub(b1);

    // Pointwise multiply: 4 muls
    return .{
        a1.mul(b1), // P(1)
        a2.mul(b2), // P(2)
        a3.mul(b3), // P(3)
        a_inf.mul(b_inf), // P(∞)
    };
}

/// Reconstruct the full round polynomial from quotient evaluations.
/// Implements Jolt's finish_mles_product_sum_from_evals (mles_product_sum.rs:235-269).
///
/// sum_evals: quotient g(x)/eq(x, r_round) evaluated at [1, 2, 3, ∞]
/// claim: the sumcheck claim p(0) + p(1) for this round
/// gruen_eq: the split-eq polynomial (provides r_round = tau[current_index - 1])
///
/// Returns: monomial coefficients of the full round polynomial g(x) (degree M+1=5, 6 coefficients).
/// Caller owns returned slice.
fn finishMlesProductSum(
    comptime F: type,
    allocator: std.mem.Allocator,
    sum_evals: [4]F,
    claim: F,
    gruen_eq: *const poly_mod.GruenSplitEqPolynomial(F),
) ![]F {
    const UniPolyT = poly_mod.UniPoly(F);

    // 1. Get r_round from the split-eq's current tau value
    const r_round = gruen_eq.tau[gruen_eq.current_index - 1];
    const eq_at_0 = F.one().sub(r_round); // eq(0, r) = 1 - r
    const eq_at_1 = r_round; // eq(1, r) = r

    // 2. Recover quotient(0) from claim:
    //    claim = eq(0,r)*q(0) + eq(1,r)*q(1)
    //    q(0) = (claim - r * q(1)) / (1 - r)
    const q_at_0 = claim.sub(eq_at_1.mul(sum_evals[0])).mul(eq_at_0.inverse().?);

    // 3. Interpolate quotient poly from [q(0), q(1), q(2), q(3), q(∞)]
    //    fromEvalsToom handles grid [0, 1, ..., n-2, ∞] where last eval = leading coeff
    var toom_evals = [5]F{ q_at_0, sum_evals[0], sum_evals[1], sum_evals[2], sum_evals[3] };
    const quotient_coeffs = try UniPolyT.fromEvalsToom(allocator, &toom_evals);
    defer allocator.free(quotient_coeffs);

    // 4. Multiply back by eq(x, r_round) = (1-r) + (2r-1)*x
    //    This produces the full g(x) of degree d = len(quotient_coeffs)
    const constant_coeff = eq_at_0; // 1 - r
    const x_coeff = r_round.add(r_round).sub(F.one()); // 2r - 1
    var final_coeffs = try allocator.alloc(F, quotient_coeffs.len + 1);
    @memset(final_coeffs, F.zero());
    for (0..quotient_coeffs.len) |i| {
        final_coeffs[i] = final_coeffs[i].add(quotient_coeffs[i].mul(constant_coeff));
        final_coeffs[i + 1] = final_coeffs[i + 1].add(quotient_coeffs[i].mul(x_coeff));
    }

    return final_coeffs;
}

fn LookupsRaVirtualProver(comptime F: type) type {
    const RaPoly = ra_poly_mod.RaPolynomial(F);

    return struct {
        const Self = @This();

        /// In-place MLE bind (same as RamRaVirtualProver.bindSlice).
        fn bindSlice(arr: []F, h: usize, challenge: F) void {
            if (challenge.limbs[0] == 0 and challenge.limbs[1] == 0) {
                for (0..h) |j| {
                    arr[j] = arr[2 * j].add(arr[2 * j + 1].sub(arr[2 * j]).mulHiBigIntU128(challenge.limbs));
                }
            } else {
                for (0..h) |j| {
                    arr[j] = arr[2 * j].add(challenge.mul(arr[2 * j + 1].sub(arr[2 * j])));
                }
            }
        }

        /// Compressed RA polynomials (lazy materialization through round1→round2→round3→dense)
        ra_polys: []RaPoly,
        /// GruenSplitEq for eq(r_cycle, .) — O(1) bind
        gruen_eq: poly_mod.GruenSplitEqPolynomial(F),
        M: usize,
        N: usize,
        total_committed: usize,
        current_len: usize,
        allocator: Allocator,
        pool: ?*ThreadPool = null,
        gpu: ?*GpuPolyOps = null,

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
            std.debug.assert(log_k_chunk <= ra_poly_mod.MAX_LOG_K_CHUNK);
            const T = trace.steps.items.len;
            const total_committed = M * N;
            const k_chunk: usize = @as(usize, 1) << @intCast(log_k_chunk);

            // Build RaPolynomials with compressed u8 indices + small eq tables
            var ra_polys_arr = try allocator.alloc(RaPoly, total_committed);
            errdefer allocator.free(ra_polys_arr);

            // Pre-allocate index arrays for all committed polys
            var indices_arr = try allocator.alloc([]?u8, total_committed);
            defer allocator.free(indices_arr);

            for (0..total_committed) |i| {
                indices_arr[i] = try allocator.alloc(?u8, T);
            }

            // Parallel fill: compute index arrays
            const LkRaInitCtx = struct {
                steps: []const tracer.TraceStep,
                indices: [][]?u8,
                log_k_chunk: usize,
                k_chunk: usize,
                instruction_d: usize,
            };
            const lk_ra_ctx = LkRaInitCtx{
                .steps = trace.steps.items,
                .indices = indices_arr,
                .log_k_chunk = log_k_chunk,
                .k_chunk = k_chunk,
                .instruction_d = instruction_d,
            };
            const lkRaInitFn = struct {
                fn f(c: LkRaInitCtx, i: usize) void {
                    for (0..c.steps.len) |j| {
                        const step = c.steps[j];
                        const chunk_val = getLookupChunkInterleaved(step, i, c.log_k_chunk, c.instruction_d);
                        c.indices[i][j] = if (chunk_val < c.k_chunk) @intCast(chunk_val) else null;
                    }
                }
            }.f;
            if (init_pool) |p| {
                p.parallelFor(total_committed, lk_ra_ctx, lkRaInitFn);
            } else {
                for (0..total_committed) |i| lkRaInitFn(lk_ra_ctx, i);
            }

            // Build eq tables and create RaPolynomials
            for (0..total_committed) |i| {
                var r_chunk_rev = try allocator.alloc(F, log_k_chunk);
                defer allocator.free(r_chunk_rev);
                for (0..log_k_chunk) |ci| r_chunk_rev[ci] = r_addr_chunks[i][log_k_chunk - 1 - ci];
                const eq_table = try computeEqTable(F, allocator, r_chunk_rev, log_k_chunk);

                const virtual_batch = i / M;
                const is_first_in_batch = (i % M == 0);
                const gamma_scale = if (is_first_in_batch) gamma_powers[virtual_batch] else F.one();

                // initRound1 takes ownership of indices and eq_table, prescales by gamma_scale
                ra_polys_arr[i] = RaPoly.initRound1(indices_arr[i], eq_table, gamma_scale);
            }

            // r_cycle is in BE order; pass directly to GruenSplitEq (same as Stage 3)
            const n_vars = std.math.log2_int(usize, T);
            const gruen_eq = try poly_mod.GruenSplitEqPolynomial(F).init(allocator, r_cycle[0..n_vars]);

            return Self{
                .ra_polys = ra_polys_arr,
                .gruen_eq = gruen_eq,
                .M = M,
                .N = N,
                .total_committed = total_committed,
                .current_len = T,
                .allocator = allocator,
                .pool = init_pool,
            };
        }

        pub fn deinit(self: *Self) void {
            for (self.ra_polys) |*p| p.deinit(self.allocator);
            self.allocator.free(self.ra_polys);
            self.gruen_eq.deinit();
        }

        /// f(x) = eq(x,r) * Sum_v Prod_{j=0}^{M-1} ra_{v*M+j}(x)
        /// Uses quotient polynomial approach: compute q(x) = f(x)/eq(x,r) at Toom points {1,2,3,∞}
        /// via evalLinearProd4, then reconstruct f(x) via finishMlesProductSumFromEvals.
        /// Returns monomial coefficients (degree M+1, i.e. 6 coefficients for M=4).
        pub fn computeRoundPoly(self: *Self, allocator: Allocator, claim: F) ![]F {
            const half = self.current_len / 2;

            // Get factored eq tables from GruenSplitEq (same pattern as Stage 3)
            const eq_tables = self.gruen_eq.getWindowEqTables(self.gruen_eq.current_index, 1);
            const E_out = eq_tables.E_out;
            const E_in = eq_tables.E_in;
            const head_in_bits = eq_tables.head_in_bits;
            const in_mask = (@as(usize, 1) << @intCast(head_in_bits)) -| 1;



            const Ctx = struct {
                ra_polys: []RaPoly,
                E_out: []const F,
                E_in: []const F,
                in_mask: usize,
                head_in_bits: usize,
                M: usize,
                N: usize,
            };
            const ctx = Ctx{
                .ra_polys = self.ra_polys,
                .E_out = E_out,
                .E_in = E_in,
                .in_mask = in_mask,
                .head_in_bits = head_in_bits,
                .M = self.M,
                .N = self.N,
            };

            // Compute quotient q(x) = Σ_j E_prefix(j) * Σ_v Π_k ra_{v*M+k}(x)
            // at 4 Toom points {1, 2, 3, ∞}
            const UPA = UnreducedProductAccum;
            const mapFn = struct {
                fn f(c: Ctx, start: usize, end: usize) [4]F {
                    var upa_acc: [4]UPA = .{UPA.zero()} ** 4;
                    for (start..end) |j| {
                        // Factored eq prefix: E_out[x_out] * E_in[x_in]
                        const x_out = j >> @intCast(c.head_in_bits);
                        const x_in = j & c.in_mask;
                        const eq_prefix = (if (x_out < c.E_out.len) c.E_out[x_out] else F.one())
                            .mul(if (x_in < c.E_in.len) c.E_in[x_in] else F.one());

                        // Accumulate sum of products across all virtual batches
                        var virtual_sum = [4]F{ F.zero(), F.zero(), F.zero(), F.zero() };
                        for (0..c.N) |v| {
                            var pairs: [4][2]F = undefined;
                            for (0..c.M) |m_idx| {
                                const idx = v * c.M + m_idx;
                                pairs[m_idx] = .{
                                    c.ra_polys[idx].getBoundCoeff(2 * j),
                                    c.ra_polys[idx].getBoundCoeff(2 * j + 1),
                                };
                            }
                            const prod_evals = evalLinearProd4(F, pairs);
                            for (0..4) |k| virtual_sum[k] = virtual_sum[k].add(prod_evals[k]);
                        }

                        for (0..4) |k| {
                            upa_acc[k].addAssign(eq_prefix.mulToProductAccum(virtual_sum[k]));
                        }
                    }
                    return .{ upa_acc[0].reduce(), upa_acc[1].reduce(), upa_acc[2].reduce(), upa_acc[3].reduce() };
                }
            }.f;

            const reduceFn = struct {
                fn f(a: [4]F, b: [4]F) [4]F {
                    return .{ a[0].add(b[0]), a[1].add(b[1]), a[2].add(b[2]), a[3].add(b[3]) };
                }
            }.f;

            const sum_evals = if (self.pool) |pool|
                pool.parallelReduce([4]F, half, .{F.zero()} ** 4, ctx, mapFn, reduceFn)
            else
                mapFn(ctx, 0, half);

            // Scale by current_scalar (accumulated eq from all previously bound variables)
            // The inner loop computed quotient' = Σ E_out*E_in*product (without current_scalar).
            // The actual quotient = current_scalar * quotient'.
            const scalar = self.gruen_eq.current_scalar;
            var scaled_evals = [4]F{
                sum_evals[0].mul(scalar),
                sum_evals[1].mul(scalar),
                sum_evals[2].mul(scalar),
                sum_evals[3].mul(scalar),
            };

            // Extract r_round from gruen_eq and reconstruct full polynomial
            const r_round = self.gruen_eq.tau[self.gruen_eq.current_index - 1];
            return poly_mod.UniPoly(F).finishMlesProductSumFromEvals(allocator, &scaled_evals, claim, r_round);
        }

        pub fn bindChallenge(self: *Self, r: F) !void {
            const half = self.current_len / 2;

            // Bind RA polynomials — O(K) for compressed states, O(T/2^round) for dense
            const all_dense = self.ra_polys[0].isDense();
            if (all_dense and self.gpu != null and half >= 16384) {
                // GPU bind: total_committed ra_poly dense arrays
                const gpu = self.gpu.?;
                for (self.ra_polys) |*rp| {
                    const dense = &rp.dense;
                    const h = dense.current_len / 2;
                    gpu.polyBindLow(dense.coeffs[0 .. h * 2], r, dense.coeffs[0..h]) catch {
                        for (0..h) |jj| {
                            dense.coeffs[jj] = dense.coeffs[2 * jj].add(r.mul(dense.coeffs[2 * jj + 1].sub(dense.coeffs[2 * jj])));
                        }
                    };
                    dense.current_len = h;
                }
            } else if (all_dense) {
                if (self.pool) |pool| {
                    const BindCtx = struct { ra: []RaPoly, tc: usize, half: usize, r: F };
                    const ctx = BindCtx{ .ra = self.ra_polys, .tc = self.total_committed, .half = half, .r = r };
                    pool.parallelForForce(self.total_committed, ctx, struct {
                        fn f(c: BindCtx, idx: usize) void {
                            const dense = &c.ra[idx].dense;
                            const h = dense.current_len / 2;
                            for (0..h) |jj| {
                                dense.coeffs[jj] = dense.coeffs[2 * jj].add(c.r.mul(dense.coeffs[2 * jj + 1].sub(dense.coeffs[2 * jj])));
                            }
                            dense.current_len = h;
                        }
                    }.f);
                } else {
                    for (self.ra_polys) |*p| try p.bind(r, self.allocator);
                }
            } else {
                for (self.ra_polys) |*p| try p.bind(r, self.allocator);
            }

            // GruenSplitEq bind — O(1) instead of O(T/2^round)
            self.gruen_eq.bind(r);

            self.current_len = half;
        }

        pub fn getOpeningClaims(self: *const Self, allocator: Allocator, gamma_powers: []const F) ![]F {
            var claims = try allocator.alloc(F, self.total_committed);
            for (0..self.total_committed) |i| {
                var claim = self.ra_polys[i].finalClaim();
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
        /// F_s[0] values saved before freeing F_s_arrs (for Phase 2 consistency check)
        f_s_bound_saved: [5]F,

        /// Data needed for phase transition
        trace: *const ExecutionTrace,
        pc_map: *const BytecodePCMapper,
        stage_r_cycles: [5][]const F,
        gamma_powers: [8]F,
        /// Val polynomials per stage: val_polys[s][k]
        val_polys: [5][]F,
        /// Identity polynomial: int_poly[k] = k as field element
        int_poly: []F,

        entry_gamma: F,
        entry_val: F,
        entry_ri: usize,
        bound_f_entry: F,
        eq_zero_scalar: F,

        allocator: Allocator,
        pool: ?*ThreadPool = null,
        gpu: ?*GpuPolyOps = null,

        pub fn init(
            allocator: Allocator,
            trace: *const ExecutionTrace,
            pc_map: *const BytecodePCMapper,
            val_polys: [5][]F, // Val_s(k) for each stage, length bytecode_K each
            bytecode_log_k: usize,
            n_cycle_vars: usize,
            bytecode_d: usize,
            log_k_chunk: usize,
            gamma_powers: [8]F,
            stage_r_cycles: [5][]const F,
            int_poly: []F,
            external_stage_claims: [5]F, // From opening claims: claim_per_stage[s]
            entry_bytecode_index: usize,
            init_pool: ?*ThreadPool,
        ) !Self {
            const bytecode_K: usize = @as(usize, 1) << @intCast(bytecode_log_k);

            // Phase 1: Build separate F_s and val_with_raf arrays per stage
            var F_s_arrs: [5][]F = undefined;
            var val_with_raf_arrs: [5][]F = undefined;
            var stage_claims_init: [5]F = undefined;

            // Split-eq F_s computation: replaces T-sized eq tables with sqrt(T)-sized E_lo/E_hi
            // F_s[s][k] = Σ_c eq(r_cycle_s, c) * δ(PC(c)=k)
            //           = Σ_{c_hi} E_hi[c_hi] * (Σ_{c_lo: PC(c)=k} E_lo[c_lo])
            // Inner loop over c_lo is additions only; one mul per touched PC per c_hi block.
            const lo_bits = n_cycle_vars / 2;
            const hi_bits = n_cycle_vars - lo_bits;
            const in_len: usize = @as(usize, 1) << @intCast(lo_bits);
            const out_len: usize = @as(usize, 1) << @intCast(hi_bits);

            // Compute all 5 stages' E_lo and E_hi tables, then run all 5 double-loops.
            // Each stage has its own buffers to enable parallel execution.
            var E_lo_arr: [5][]F = undefined;
            var E_hi_arr: [5][]F = undefined;

            for (0..5) |s| {
                var r_cycle_rev_s = try allocator.alloc(F, n_cycle_vars);
                defer allocator.free(r_cycle_rev_s);
                for (0..n_cycle_vars) |i| {
                    r_cycle_rev_s[i] = stage_r_cycles[s][n_cycle_vars - 1 - i];
                }
                E_lo_arr[s] = try computeEqTableParallel(F, allocator, r_cycle_rev_s[0..lo_bits], lo_bits, init_pool);
                E_hi_arr[s] = try computeEqTableParallel(F, allocator, r_cycle_rev_s[lo_bits..n_cycle_vars], hi_bits, init_pool);
            }
            defer for (0..5) |s| {
                allocator.free(E_lo_arr[s]);
                allocator.free(E_hi_arr[s]);
            };

            // Allocate F_s output and per-stage temp buffers
            for (0..5) |s| {
                F_s_arrs[s] = try allocator.alloc(F, bytecode_K);
                @memset(F_s_arrs[s], F.zero());
            }

            // Run all 5 stages' double-loops in parallel (each stage independent)
            // Pre-allocate per-stage heap buffers (bytecode_K can be >256 for large programs)
            var per_stage_inner: [5][]F = undefined;
            var per_stage_touched: [5][]usize = undefined;
            var per_stage_tset: [5][]bool = undefined;
            for (0..5) |s| {
                per_stage_inner[s] = try allocator.alloc(F, bytecode_K);
                @memset(per_stage_inner[s], F.zero());
                per_stage_touched[s] = try allocator.alloc(usize, bytecode_K);
                per_stage_tset[s] = try allocator.alloc(bool, bytecode_K);
                @memset(per_stage_tset[s], false);
            }
            defer for (0..5) |s| {
                allocator.free(per_stage_inner[s]);
                allocator.free(per_stage_touched[s]);
                allocator.free(per_stage_tset[s]);
            };

            if (init_pool) |pool| {
                const FsCtx = struct {
                    F_s_out: *[5][]F,
                    E_lo_a: *[5][]F,
                    E_hi_a: *[5][]F,
                    steps: []const tracer.TraceStep,
                    pc_map_ptr: *const BytecodePCMapper,
                    in_len: usize,
                    out_len: usize,
                    lo_bits: usize,
                    bK: usize,
                    inner_bufs: *[5][]F,
                    touched_bufs: *[5][]usize,
                    tset_bufs: *[5][]bool,
                };
                const fs_ctx = FsCtx{
                    .F_s_out = &F_s_arrs,
                    .E_lo_a = &E_lo_arr,
                    .E_hi_a = &E_hi_arr,
                    .steps = trace.steps.items,
                    .pc_map_ptr = pc_map,
                    .in_len = in_len,
                    .out_len = out_len,
                    .lo_bits = lo_bits,
                    .bK = bytecode_K,
                    .inner_bufs = &per_stage_inner,
                    .touched_bufs = &per_stage_touched,
                    .tset_bufs = &per_stage_tset,
                };
                pool.parallelForForce(5, fs_ctx, struct {
                    fn f(c: FsCtx, s: usize) void {
                        const E_lo = c.E_lo_a[s];
                        const E_hi = c.E_hi_a[s];
                        const F_s = c.F_s_out[s];
                        const inner_buf = c.inner_bufs[s];
                        const touched_buf = c.touched_bufs[s];
                        const touched_set = c.tset_bufs[s];

                        for (0..c.out_len) |c_hi| {
                            var touched_count: usize = 0;
                            for (0..c.in_len) |c_lo| {
                                const idx = c_lo + (c_hi << @intCast(c.lo_bits));
                                const step = c.steps[idx];
                                const pc_idx = c.pc_map_ptr.getPCForStep(step);
                                if (pc_idx < c.bK) {
                                    if (!touched_set[pc_idx]) {
                                        touched_set[pc_idx] = true;
                                        touched_buf[touched_count] = pc_idx;
                                        touched_count += 1;
                                    }
                                    inner_buf[pc_idx] = inner_buf[pc_idx].add(E_lo[c_lo]);
                                }
                            }
                            const e_hi_val = E_hi[c_hi];
                            for (0..touched_count) |ti| {
                                const pc = touched_buf[ti];
                                F_s[pc] = F_s[pc].add(e_hi_val.mul(inner_buf[pc]));
                                inner_buf[pc] = F.zero();
                                touched_set[pc] = false;
                            }
                        }
                    }
                }.f);
            } else {
                // Sequential fallback
                var inner_buf = try allocator.alloc(F, bytecode_K);
                defer allocator.free(inner_buf);
                var touched_buf = try allocator.alloc(usize, bytecode_K);
                defer allocator.free(touched_buf);
                var touched_set = try allocator.alloc(bool, bytecode_K);
                defer allocator.free(touched_set);

                for (0..5) |s| {
                    @memset(inner_buf, F.zero());
                    @memset(touched_set, false);

                    for (0..out_len) |c_hi| {
                        var touched_count: usize = 0;
                        for (0..in_len) |c_lo| {
                            const c = c_lo + (c_hi << @intCast(lo_bits));
                            const step = trace.steps.items[c];
                            const pc_idx = pc_map.getPCForStep(step);
                            if (pc_idx < bytecode_K) {
                                if (!touched_set[pc_idx]) {
                                    touched_set[pc_idx] = true;
                                    touched_buf[touched_count] = pc_idx;
                                    touched_count += 1;
                                }
                                inner_buf[pc_idx] = inner_buf[pc_idx].add(E_lo_arr[s][c_lo]);
                            }
                        }
                        const e_hi_val = E_hi_arr[s][c_hi];
                        for (0..touched_count) |ti| {
                            const pc = touched_buf[ti];
                            F_s_arrs[s][pc] = F_s_arrs[s][pc].add(e_hi_val.mul(inner_buf[pc]));
                            inner_buf[pc] = F.zero();
                            touched_set[pc] = false;
                        }
                    }
                }
            }

            // Build val_with_raf and compute claims for each stage
            for (0..5) |s| {
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
                for (0..bytecode_K) |k| {
                    recomputed_claim = recomputed_claim.add(F_s_arrs[s][k].mul(val_with_raf_arrs[s][k]));
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
            if (comptime debug_verbose) {
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
                .f_s_bound_saved = [_]F{F.zero()} ** 5,
                .trace = trace,
                .pc_map = pc_map,
                .stage_r_cycles = stage_r_cycles,
                .gamma_powers = gamma_powers,
                .val_polys = val_polys,
                .int_poly = int_poly,
                .entry_gamma = gamma_powers[7],
                .entry_val = F.one(),
                .entry_ri = entry_bytecode_index,
                .bound_f_entry = F.zero(),
                .eq_zero_scalar = F.one(),
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

            const entry_bit = self.entry_ri & 1;
            const ev_sq = self.entry_val.mul(self.entry_val);
            const eg_ev = self.entry_gamma.mul(ev_sq);
            if (entry_bit == 0) {
                agg_eval_0 = agg_eval_0.add(eg_ev);
                agg_eval_2 = agg_eval_2.add(eg_ev);
            } else {
                agg_eval_2 = agg_eval_2.add(eg_ev.add(eg_ev).add(eg_ev).add(eg_ev));
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

            const bindOneArr = struct {
                fn f(arr: []F, h: usize, challenge: F) void {
                    for (0..h) |k| {
                        arr[k] = arr[2 * k].add(challenge.mul(arr[2 * k + 1].sub(arr[2 * k])));
                    }
                }
            }.f;

            const updateClaim = struct {
                fn f(stage_claim: *F, pse: [2]F, challenge: F, t_inv: F) void {
                    const p0 = pse[0];
                    const p2 = pse[1];
                    const p1 = stage_claim.*.sub(p0);
                    const a0 = p0;
                    const a2 = p2.sub(p1.add(p1)).add(p0).mul(t_inv);
                    const a1 = p1.sub(p0).sub(a2);
                    stage_claim.* = a0.add(challenge.mul(a1.add(challenge.mul(a2))));
                }
            }.f;

            if (self.gpu) |gpu| {
                if (half >= 16384) {
                    // GPU bind: 5 stages x 2 arrays each, then update claims on CPU
                    for (0..5) |s| {
                        gpu.polyBindLow(self.F_s_arrs[s][0 .. half * 2], r, self.F_s_arrs[s][0..half]) catch bindOneArr(self.F_s_arrs[s], half, r);
                        gpu.polyBindLow(self.val_with_raf[s][0 .. half * 2], r, self.val_with_raf[s][0..half]) catch bindOneArr(self.val_with_raf[s], half, r);
                        updateClaim(&self.stage_claims[s], per_stage_evals[s], r, two_inv);
                    }
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
            } else if (self.pool) |pool| {
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


            const entry_bit = self.entry_ri & 1;
            if (entry_bit == 0) {
                self.entry_val = self.entry_val.mul(F.one().sub(r));
            } else {
                self.entry_val = self.entry_val.mul(r);
            }
            self.entry_ri >>= 1;

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
            // Use r_address_be (reversed) for the eq table, matching Jolt's normalize_opening_point.
            const eq_addr = try computeEqTableParallel(F, self.allocator, r_address_be, self.bytecode_log_k, self.pool);
            defer self.allocator.free(eq_addr);

            // Debug: eq_addr entries
            if (comptime debug_verbose) {
                for (0..bytecode_K) |ek| {
                    const eab = eq_addr[ek].toBytesBE();
                    dbg("[ZOLT_EQ_ADDR] eq[{d}]_LE=[", .{ek});
                    for (0..32) |bi| dbg("{x:0>2}", .{eab[31 - bi]});
                    dbg("]\n", .{});
                }
            }

            // Debug: val_polys entries
            if (comptime debug_verbose) {
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
                    val_eval = val_eval.add(self.val_polys[s][k].mul(eq_addr[k]));
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
                // After Phase 1 binding of all address variables, val_with_raf[s][0] is the
                // MLE evaluation at the binding point. stage_claims[s] = F_s[0]*val_with_raf[s][0]
                // since both are reduced to single elements.
                bound_vals[s] = self.gamma_powers[s].mul(self.val_with_raf[s][0]);
                self.bound_vals_stored[s] = bound_vals[s];

                // DIAGNOSTIC: compare re-computed val_eval with Phase 1 bound val_with_raf[s][0]
                if (comptime debug_verbose) {
                    const phase1_bound = self.val_with_raf[s][0];
                    dbg("[TRANS_CHECK] stage[{}]: match={}\n", .{ s, @as(u8, if (val_eval.eql(phase1_bound)) 1 else 0) });
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
            // Debug: verify RA product vs eq_binding (behind comptime debug_verbose)
            if (comptime debug_verbose) {
                // Check RA product vs eq_binding (using LH challenges directly)
                const eq_binding_check = try computeEqTableParallel(F, self.allocator, r_address_challenges, self.bytecode_log_k, self.pool);
                defer self.allocator.free(eq_binding_check);
                var mismatch_count: usize = 0;
                for (0..T) |c| {
                    var ra_prod = F.one();
                    for (0..self.bytecode_d) |i| {
                        ra_prod = ra_prod.mul(self.ra_chunks.?[i][c]);
                    }
                    const step = self.trace.steps.items[c];
                    const pc = self.pc_map.getPCForStep(step);
                    const full_eq = if (pc < bytecode_K) eq_binding_check[pc] else F.zero();
                    if (!ra_prod.eql(full_eq)) mismatch_count += 1;
                }
                dbg("[BCRAF_RA] total mismatches: {}/{}\n", .{ mismatch_count, T });
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

            // Save F_s[0] before freeing
            for (0..5) |s| {
                self.f_s_bound_saved[s] = if (self.F_s_arrs[s].len > 0) self.F_s_arrs[s][0] else F.zero();
            }

            // Free Phase 1 arrays (no longer needed)
            // Replace with zero-length allocations so deinit doesn't double-free
            for (0..5) |s| {
                self.allocator.free(self.F_s_arrs[s]);
                self.F_s_arrs[s] = try self.allocator.alloc(F, 0);
                self.allocator.free(self.val_with_raf[s]);
                self.val_with_raf[s] = try self.allocator.alloc(F, 0);
            }


            self.bound_f_entry = self.entry_val;
            self.combined.?[0] = self.combined.?[0].add(self.entry_gamma.mul(self.bound_f_entry));

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
                            // Start product from first ra_chunk (avoid F.one() mul)
                            const r0_first = c.ra_chunks[0][2 * j];
                            const r1_first = c.ra_chunks[0][2 * j + 1];
                            var ra_product = r0_first.add(x.mul(r1_first.sub(r0_first)));

                            for (1..c.bytecode_d) |i| {
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

            if (self.gpu) |gpu| {
                if (half >= 16384) {
                    // GPU bind: bytecode_d ra_chunks + 1 combined
                    for (0..self.bytecode_d) |i| {
                        gpu.polyBindLow(ra_chunks[i][0 .. half * 2], r, ra_chunks[i][0..half]) catch bindOne(ra_chunks[i], half, r);
                    }
                    gpu.polyBindLow(combined[0 .. half * 2], r, combined[0..half]) catch bindOne(combined, half, r);
                } else {
                    bindOne(combined, half, r);
                    for (0..self.bytecode_d) |i| {
                        bindOne(ra_chunks[i], half, r);
                    }
                }
            } else if (self.pool) |pool| {
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
        gpu_ops: ?*GpuPolyOps = null,

        pub fn init(allocator: Allocator) Self {
            return .{ .allocator = allocator };
        }

        /// GPU-accelerated bindLow: arr[j] = arr[2j] + r*(arr[2j+1] - arr[2j])
        /// Falls back to CPU when GPU unavailable or array too small.
        fn gpuBindLow(arr: []F, half: usize, r: F, gpu_ops: ?*GpuPolyOps) void {
            if (gpu_ops) |gpu| {
                if (half >= 16384) {
                    gpu.polyBindLow(arr[0 .. half * 2], r, arr[0..half]) catch {
                        cpuBindLow(arr, half, r);
                        return;
                    };
                    return;
                }
            }
            cpuBindLow(arr, half, r);
        }

        fn cpuBindLow(arr: []F, half: usize, r: F) void {
            for (0..half) |j| {
                arr[j] = arr[2 * j].add(r.montgomeryMul(arr[2 * j + 1].sub(arr[2 * j])));
            }
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
            entry_address: u64,
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
            const bytecode_raf_gamma_powers = try transcript.challengeScalarPowers(self.allocator, 8);
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

            const NUM_LOOKUP_TABLES: usize = 40;
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
            // Also compute γ^i powers for Phase 2 pre-scaling optimization
            const booleanity_gamma_unsq = try self.allocator.alloc(F, total_d);
            booleanity_gamma_unsq[0] = F.one(); // γ^0 = 1
            for (1..total_d) |i| {
                booleanity_gamma_unsq[i] = booleanity_gamma_unsq[i - 1].mul(booleanity_gamma_f); // γ^i
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
            var bytecodeReadRaf_input = bcraf_result.total.add(bytecode_raf_gamma_powers[7]);
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
            var s6_init_timer: if (s6_bench_timing) std.time.Timer else void = if (comptime s6_bench_timing) std.time.Timer.start() catch unreachable else {};

            // Instance 5: IncClaimReduction (degree 2)
            // IncClaimReduction uses RAM r_cycles (not BytecodeReadRaf r_cycles)
            var inc_prover = try IncClaimReductionProver(F).init(
                self.allocator, trace, inc_gamma,
                r_cycle_inc_ram_rwc, r_cycle_inc_ram_val,
                r_cycle_bc4_regs_rwc, r_cycle_bc5_regs_val,
                self.thread_pool,
            );
            inc_prover.gpu = self.gpu_ops;
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
            hamming_prover.gpu = self.gpu_ops;
            defer hamming_prover.deinit();

            // Instance 3: RamRaVirtual (degree ram_d+1)
            var ram_ra_prover = try RamRaVirtualProver(F).init(
                self.allocator, trace, ram_ra_r_cycle,
                ram_ra_addr_chunks, ram_d, memory_layout, log_k_chunk,
                self.thread_pool,
            );
            ram_ra_prover.gpu = self.gpu_ops;
            defer ram_ra_prover.deinit();

            // Instance 4: LookupsRaVirtual (degree n_committed_per_virtual+1)
            var lookups_ra_prover = try LookupsRaVirtualProver(F).init(
                self.allocator, trace, lookups_ra_r_cycle,
                lookups_ra_addr_chunks, lookups_ra_gamma_powers,
                n_committed_per_virtual, n_virtual_ra_polys,
                log_k_chunk, instruction_d,
                self.thread_pool,
            );
            lookups_ra_prover.gpu = self.gpu_ops;
            defer lookups_ra_prover.deinit();

            // Verify: eq table partition of unity (Σ eq[j] = 1)
            if (comptime debug_verbose) {
                var eq_sum = F.zero();
                for (0..lookups_ra_prover.current_len) |j| {
                    eq_sum = eq_sum.add(lookups_ra_prover.e_out[j]);
                }
                dbg("[LR_EQ] Σeq==1? {} T={}\n", .{eq_sum.eql(F.one()), lookups_ra_prover.current_len});
            }

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
                // Build GruenSplitEq for Booleanity Phase 2 (O(1) bind)
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

                // OPTIMIZATION: Pre-compute chunk indices for all T steps in ONE parallel pass.
                // This avoids calling computeLookupIndex 38 times per step (once per poly).
                // Each step produces: instruction chunks [0..instr_d], bytecode chunks [0..bc_d], ram chunks [0..ram_d]
                // Stored as u8 per chunk (K < 256).
                const MAX_BOOL_POLYS = 48; // instruction_d(32) + bytecode_d(~3-5) + ram_d(~2-3)
                std.debug.assert(total_bool_polys <= MAX_BOOL_POLYS);

                // Allocate per-step chunk index arrays: chunk_idx[j][poly_i] = chunk value (or K_val for invalid)
                const chunk_idx = try self.allocator.alloc([MAX_BOOL_POLYS]u8, T_val);
                defer self.allocator.free(chunk_idx);

                // Phase 1: Single-pass pre-compute all chunk indices (parallel over T)
                {
                    const ChunkPreCtx = struct {
                        steps: []const tracer.TraceStep,
                        pc_map_ptr: *const BytecodePCMapper,
                        mem_layout: *const jolt_device.MemoryLayout,
                        instr_d: usize,
                        bc_d: usize,
                        rm_d: usize,
                        lkc: usize,
                        K: usize,
                        total_polys: usize,
                        chunk_idx: [][MAX_BOOL_POLYS]u8,
                    };
                    const pre_ctx = ChunkPreCtx{
                        .steps = trace.steps.items,
                        .pc_map_ptr = pc_map,
                        .mem_layout = memory_layout,
                        .instr_d = instruction_d,
                        .bc_d = bytecode_d,
                        .rm_d = ram_d,
                        .lkc = log_k_chunk,
                        .K = K_val,
                        .total_polys = total_bool_polys,
                        .chunk_idx = chunk_idx,
                    };
                    const precomputeFn = struct {
                        fn f(c: ChunkPreCtx, j: usize) void {
                            const step = c.steps[j];
                            const sentinel: u8 = @intCast(c.K); // K < 256, use K as "invalid" sentinel

                            // InstructionRa: compute lookup_idx ONCE, extract all chunks
                            const lookup_idx = computeLookupIndex(step);
                            const mask: u128 = (@as(u128, 1) << @intCast(c.lkc)) - 1;
                            for (0..c.instr_d) |i| {
                                const shift = c.lkc * (c.instr_d - 1 - i);
                                const chunk_val: usize = @intCast((lookup_idx >> @intCast(shift)) & mask);
                                c.chunk_idx[j][i] = if (chunk_val < c.K) @intCast(chunk_val) else sentinel;
                            }

                            // BytecodeRa: compute PC ONCE, extract all chunks
                            const pc_idx: u64 = @intCast(c.pc_map_ptr.getPCForStep(step));
                            for (0..c.bc_d) |i| {
                                const chunk_val = extractChunkMSB(pc_idx, i, c.bc_d, c.lkc);
                                c.chunk_idx[j][c.instr_d + i] = if (chunk_val < c.K) @intCast(chunk_val) else sentinel;
                            }

                            // RamRa: compute address ONCE, extract all chunks
                            if (step.memory_addr) |addr| {
                                if (addr != 0) {
                                    if (c.mem_layout.remapAddress(addr)) |raddr| {
                                        for (0..c.rm_d) |i| {
                                            const chunk_val = extractChunkMSB(raddr, i, c.rm_d, c.lkc);
                                            c.chunk_idx[j][c.instr_d + c.bc_d + i] = if (chunk_val < c.K) @intCast(chunk_val) else sentinel;
                                        }
                                    } else {
                                        for (0..c.rm_d) |i| c.chunk_idx[j][c.instr_d + c.bc_d + i] = sentinel;
                                    }
                                } else {
                                    for (0..c.rm_d) |i| c.chunk_idx[j][c.instr_d + c.bc_d + i] = sentinel;
                                }
                            } else {
                                for (0..c.rm_d) |i| c.chunk_idx[j][c.instr_d + c.bc_d + i] = sentinel;
                            }
                        }
                    }.f;
                    if (self.thread_pool) |pool| {
                        pool.parallelForForce(T_val, pre_ctx, precomputeFn);
                    } else {
                        for (0..T_val) |j| precomputeFn(pre_ctx, j);
                    }
                }

                // Phase 2: Build G tables using pre-computed indices (parallel over polys)
                // Each poly's inner loop is now a simple scatter-add with O(1) index lookup.
                if (self.thread_pool) |pool| {
                    const GBuildCtx = struct {
                        eq_cycle: []const F,
                        chunk_idx: [][MAX_BOOL_POLYS]u8,
                        K: usize,
                        T: usize,
                        G_out: [][]F,
                    };
                    const g_ctx = GBuildCtx{
                        .eq_cycle = eq_cycle_bool_phase2,
                        .chunk_idx = chunk_idx,
                        .K = K_val,
                        .T = T_val,
                        .G_out = G_tables,
                    };
                    pool.parallelForForce(total_bool_polys, g_ctx, struct {
                        fn f(c: GBuildCtx, poly_i: usize) void {
                            const G_i = c.G_out[poly_i];
                            const sentinel: u8 = @intCast(c.K);
                            for (0..c.T) |j| {
                                const cv = c.chunk_idx[j][poly_i];
                                if (cv != sentinel) {
                                    const eq_j = c.eq_cycle[j];
                                    G_i[cv] = G_i[cv].add(eq_j);
                                }
                            }
                        }
                    }.f);
                } else {
                    // Sequential: single pass over T, scatter to all polys per step
                    for (0..T_val) |j| {
                        const eq_j = eq_cycle_bool_phase2[j];
                        if (eq_j.eql(F.zero())) continue;
                        const sentinel: u8 = @intCast(K_val);
                        for (0..total_bool_polys) |i| {
                            const cv = chunk_idx[j][i];
                            if (cv != sentinel) {
                                G_tables[i][cv] = G_tables[i][cv].add(eq_j);
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
                    booleanity_gamma_unsq,
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
            booleanity_prover.gpu = self.gpu_ops;
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
                    // Signed encoding: must match R1CS witness and Jolt verifier.
                    const is_signed_format = (opcode_for_imm == 0x63) or // B-type (branches: FormatB i128)
                        (opcode_for_imm == 0x23) or // S-type (stores: FormatS i64)
                        (opcode_for_imm == 0x03) or // Load (FormatLoad: i64 sign-extended to i128)
                        (opcode_for_imm == 0x22); // VirtualAssert (FormatAssert: signed i64)
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
                if (entry.lookup_table_index < 40) {
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
                var bc_table_sums: [40]F = undefined;
                for (0..40) |t| bc_table_sums[t] = F.zero();
                for (0..bytecode_K) |k| {
                    if (k >= bytecode_entries.len) break;
                    const entry = bytecode_entries[k];
                    if (entry.rd < REG_COUNT5) {
                        bc_rd5_sum = bc_rd5_sum.add(F_s5[k].mul(eq_table_5[entry.rd]));
                    }
                    if (!entry.is_interleaved) {
                        bc_iraf_sum = bc_iraf_sum.add(F_s5[k]);
                    }
                    if (entry.lookup_table_index < 40) {
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
                for (0..40) |t| {
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

            var bytecode_gamma_arr: [8]F = undefined;
            for (0..8) |i| {
                bytecode_gamma_arr[i] = bytecode_raf_gamma_powers[i];
            }
            const entry_bytecode_index = pc_map.getPC(entry_address, 0);
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
                entry_bytecode_index,
                self.thread_pool,
            );
            bytecode_prover.gpu = self.gpu_ops;
            defer bytecode_prover.deinit();

            // pc_maps now consistent — no override needed

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

            const num_compressed = max_degree;

            // Track Phase 1 address challenges for BytecodeReadRaf
            var bytecode_addr_challenges = try self.allocator.alloc(F, bytecode_log_k);
            defer self.allocator.free(bytecode_addr_challenges);

            // Stage 6 fine-grained timing (gated by s6_bench_timing)
            if (comptime s6_bench_timing) {
                std.debug.print("    [STAGE6-BENCH] Init: {d:7.1}ms\n", .{
                    @as(f64, @floatFromInt(s6_init_timer.read())) / 1_000_000.0,
                });
            }
            var s6_t_compute: if (s6_bench_timing) [6]u64 else void = if (comptime s6_bench_timing) [6]u64{ 0, 0, 0, 0, 0, 0 } else {};
            var s6_t_bind: if (s6_bench_timing) [6]u64 else void = if (comptime s6_bench_timing) [6]u64{ 0, 0, 0, 0, 0, 0 } else {};
            var s6_t_transcript: if (s6_bench_timing) u64 else void = if (comptime s6_bench_timing) @as(u64, 0) else {};
            var s6_timer: if (s6_bench_timing) std.time.Timer else void = if (comptime s6_bench_timing) std.time.Timer.start() catch unreachable else {};

            for (0..max_num_rounds) |round| {
                const remaining_rounds = max_num_rounds - round;

                // Monomial-form batched polynomial: combined_coeffs[i] = coefficient of x^i
                // This matches Jolt's approach: each instance returns a UniPoly in monomial form,
                // and the batched poly is Σ batch[i] * poly_i in coefficient space.
                var combined_coeffs = try self.allocator.alloc(F, max_degree + 1);
                defer self.allocator.free(combined_coeffs);
                @memset(combined_coeffs, F.zero());

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
                // Debug: per-instance contribution to combined_coeffs[0] and [1]
                var dbg_inst_p0: [6]F = .{F.zero()} ** 6;
                var dbg_inst_p1: [6]F = .{F.zero()} ** 6;

                // Instance 0: BytecodeReadRaf - REAL prover
                if (comptime s6_bench_timing) s6_timer.reset();
                {
                    const inst = 0;
                    if (remaining_rounds > num_rounds_arr[inst]) {
                        // Not started yet - constant polynomial (degree 0)
                        // Jolt: c0 = previous_claim / 2. In Zolt terms: c0 = input_claims[inst] * 2^scale
                        // where scale = remaining_rounds - num_rounds - 1, which equals Jolt's individual_claims[inst] / 2.
                        const scale = remaining_rounds - num_rounds_arr[inst] - 1;
                        var scaled = input_claims[inst];
                        for (0..scale) |_| scaled = scaled.add(scaled);
                        combined_coeffs[0] = combined_coeffs[0].add(batch[inst].mul(scaled));
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

                            // Add degree-2 monomial coefficients [a0, a1, a2] to combined_coeffs
                            combined_coeffs[0] = combined_coeffs[0].add(batch[inst].mul(a0));
                            combined_coeffs[1] = combined_coeffs[1].add(batch[inst].mul(a1));
                            combined_coeffs[2] = combined_coeffs[2].add(batch[inst].mul(a2));
                        } else {
                            // Phase 2: cycle binding (degree bytecode_d+1)
                            const polys = try bytecode_prover.computeRoundPolyPhase2(self.allocator);
                            cached_bc_phase2 = polys;
                            if (debug_r5) {
                                const p01 = polys[0].add(polys[1]);
                                const p01_ok: u8 = if (std.mem.eql(u8, &p01.toBytesBE(), &instance_claims[inst].toBytesBE())) 1 else 0;
                                dbg("  [R5_DBG] inst0_phase2 polys_len={} p(0)+p(1)=claim? {}\n", .{ polys.len, p01_ok });
                            }
                            // Convert evaluations to monomial coefficients and add to combined_coeffs
                            const mono = try UniPoly(F).fromEvalsVandermonde(self.allocator, polys);
                            defer self.allocator.free(mono);
                            for (0..mono.len) |ci| {
                                combined_coeffs[ci] = combined_coeffs[ci].add(batch[inst].mul(mono[ci]));
                            }
                        }
                    }
                }

                dbg_inst_p0[0] = combined_coeffs[0];
                dbg_inst_p1[0] = combined_coeffs[1];

                if (debug_r5) {
                    const e0 = combined_coeffs[0].toBytes();
                    const e1 = combined_coeffs[1].toBytes();
                    dbg("  [R5_DBG] after inst0: c[0]_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}] c[1]_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                        e0[0], e0[1], e0[2], e0[3], e0[4], e0[5], e0[6], e0[7],
                        e1[0], e1[1], e1[2], e1[3], e1[4], e1[5], e1[6], e1[7],
                    });
                }

                if (comptime s6_bench_timing) s6_t_compute[0] += s6_timer.read();
                // Instance 1: Booleanity - REAL prover (degree 3)
                if (comptime s6_bench_timing) s6_timer.reset();
                var cached_booleanity: ?[]F = null;
                {
                    const inst = 1;
                    if (remaining_rounds > num_rounds_arr[inst]) {
                        const scale = remaining_rounds - num_rounds_arr[inst] - 1;
                        var scaled = input_claims[inst];
                        for (0..scale) |_| scaled = scaled.add(scaled);
                        combined_coeffs[0] = combined_coeffs[0].add(batch[inst].mul(scaled));
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
                        // Convert degree-3 evals [p(0), p(1), p(2), p(3)] to monomial coefficients
                        // using finite differences, then add batch[inst] * coeffs to combined_coeffs
                        addEvalsAsMonomialToCoeffs(F, combined_coeffs, polys, 4, batch[inst]);
                    }
                }
                dbg_inst_p0[1] = combined_coeffs[0];
                dbg_inst_p1[1] = combined_coeffs[1];

                if (debug_r5) {
                    const e0 = combined_coeffs[0].toBytes();
                    const e1 = combined_coeffs[1].toBytes();
                    dbg("  [R5_DBG] after inst1: c[0]_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}] c[1]_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                        e0[0], e0[1], e0[2], e0[3], e0[4], e0[5], e0[6], e0[7],
                        e1[0], e1[1], e1[2], e1[3], e1[4], e1[5], e1[6], e1[7],
                    });
                }

                if (comptime s6_bench_timing) s6_t_compute[1] += s6_timer.read();
                // Instance 2: HammingBooleanity - REAL prover
                if (comptime s6_bench_timing) s6_timer.reset();
                {
                    const inst = 2;
                    if (remaining_rounds > num_rounds_arr[inst]) {
                        const scale = remaining_rounds - num_rounds_arr[inst] - 1;
                        var scaled = input_claims[inst];
                        for (0..scale) |_| scaled = scaled.add(scaled);
                        combined_coeffs[0] = combined_coeffs[0].add(batch[inst].mul(scaled));
                    } else {
                        inst_active[inst] = true;
                        const polys = hamming_prover.computeRoundPoly();
                        cached_hamming = polys;
                        addEvalsAsMonomialToCoeffs(F, combined_coeffs, &polys, 4, batch[inst]);
                    }
                }
                dbg_inst_p0[2] = combined_coeffs[0];
                dbg_inst_p1[2] = combined_coeffs[1];

                if (debug_r5) {
                    const e0 = combined_coeffs[0].toBytes();
                    const e1 = combined_coeffs[1].toBytes();
                    dbg("  [R5_DBG] after inst2: c[0]_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}] c[1]_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                        e0[0], e0[1], e0[2], e0[3], e0[4], e0[5], e0[6], e0[7],
                        e1[0], e1[1], e1[2], e1[3], e1[4], e1[5], e1[6], e1[7],
                    });
                }

                if (comptime s6_bench_timing) s6_t_compute[2] += s6_timer.read();
                // Instance 3: RamRaVirtual - REAL prover
                if (comptime s6_bench_timing) s6_timer.reset();
                {
                    const inst = 3;
                    if (remaining_rounds > num_rounds_arr[inst]) {
                        const scale = remaining_rounds - num_rounds_arr[inst] - 1;
                        var scaled = input_claims[inst];
                        for (0..scale) |_| scaled = scaled.add(scaled);
                        combined_coeffs[0] = combined_coeffs[0].add(batch[inst].mul(scaled));
                    } else {
                        inst_active[inst] = true;
                        // computeRoundPoly now returns monomial coefficients directly (Toom-Cook quotient approach)
                        const mono = try ram_ra_prover.computeRoundPoly(self.allocator, instance_claims[inst]);
                        cached_ram_ra = mono;
                        if (debug_r5) {
                            // Check p(0)+p(1)=claim for RamRaVirtual (mono format: eval via Horner)
                            var p0 = mono[mono.len - 1];
                            var ci_dbg: usize = mono.len - 1;
                            while (ci_dbg > 0) { ci_dbg -= 1; p0 = p0.mul(F.zero()).add(mono[ci_dbg]); }
                            var p1 = mono[mono.len - 1];
                            ci_dbg = mono.len - 1;
                            while (ci_dbg > 0) { ci_dbg -= 1; p1 = p1.mul(F.one()).add(mono[ci_dbg]); }
                            const p01 = p0.add(p1);
                            const p01_ok: u8 = if (std.mem.eql(u8, &p01.toBytesBE(), &instance_claims[inst].toBytesBE())) 1 else 0;
                            dbg("  [R5_DBG] inst3 polys_len={} p(0)+p(1)=claim? {}\n", .{ mono.len, p01_ok });
                        }
                        for (0..mono.len) |ci| {
                            combined_coeffs[ci] = combined_coeffs[ci].add(batch[inst].mul(mono[ci]));
                        }
                    }
                }
                dbg_inst_p0[3] = combined_coeffs[0];
                dbg_inst_p1[3] = combined_coeffs[1];

                if (debug_r5) {
                    const e0 = combined_coeffs[0].toBytes();
                    const e1 = combined_coeffs[1].toBytes();
                    dbg("  [R5_DBG] after inst3: c[0]_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}] c[1]_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                        e0[0], e0[1], e0[2], e0[3], e0[4], e0[5], e0[6], e0[7],
                        e1[0], e1[1], e1[2], e1[3], e1[4], e1[5], e1[6], e1[7],
                    });
                }

                if (comptime s6_bench_timing) s6_t_compute[3] += s6_timer.read();
                // Instance 4: LookupsRaVirtual - REAL prover
                // Overlap with previous instances via join when both are active
                if (comptime s6_bench_timing) s6_timer.reset();
                {
                    const inst = 4;
                    if (remaining_rounds > num_rounds_arr[inst]) {
                        const scale = remaining_rounds - num_rounds_arr[inst] - 1;
                        var scaled = input_claims[inst];
                        for (0..scale) |_| scaled = scaled.add(scaled);
                        combined_coeffs[0] = combined_coeffs[0].add(batch[inst].mul(scaled));
                    } else {
                        inst_active[inst] = true;
                        // computeRoundPoly now returns monomial coefficients directly (Toom-Cook quotient approach)
                        const mono = try lookups_ra_prover.computeRoundPoly(self.allocator, instance_claims[inst]);
                        cached_lookups_ra = mono;
                        for (0..mono.len) |ci| {
                            combined_coeffs[ci] = combined_coeffs[ci].add(batch[inst].mul(mono[ci]));
                        }
                    }
                }
                dbg_inst_p0[4] = combined_coeffs[0];
                dbg_inst_p1[4] = combined_coeffs[1];

                if (debug_r5) {
                    const e0 = combined_coeffs[0].toBytes();
                    const e1 = combined_coeffs[1].toBytes();
                    dbg("  [R5_DBG] after inst4: c[0]_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}] c[1]_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                        e0[0], e0[1], e0[2], e0[3], e0[4], e0[5], e0[6], e0[7],
                        e1[0], e1[1], e1[2], e1[3], e1[4], e1[5], e1[6], e1[7],
                    });
                }

                if (comptime s6_bench_timing) s6_t_compute[4] += s6_timer.read();
                // Instance 5: IncClaimReduction - REAL prover
                if (comptime s6_bench_timing) s6_timer.reset();
                {
                    const inst = 5;
                    if (remaining_rounds > num_rounds_arr[inst]) {
                        const scale = remaining_rounds - num_rounds_arr[inst] - 1;
                        var scaled = input_claims[inst];
                        for (0..scale) |_| scaled = scaled.add(scaled);
                        combined_coeffs[0] = combined_coeffs[0].add(batch[inst].mul(scaled));
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
                        // Interpolate monomial coefficients: a0 + a1*x + a2*x^2
                        const a0 = p0;
                        const two = F.fromU64(2);
                        const two_inv = two.inverse().?;
                        const a2_coeff = polys[2].sub(p1.add(p1)).add(p0).mul(two_inv);
                        const a1 = p1.sub(a0).sub(a2_coeff);

                        // Add monomial coefficients to combined_coeffs
                        combined_coeffs[0] = combined_coeffs[0].add(batch[inst].mul(a0));
                        combined_coeffs[1] = combined_coeffs[1].add(batch[inst].mul(a1));
                        combined_coeffs[2] = combined_coeffs[2].add(batch[inst].mul(a2_coeff));
                    }
                }
                dbg_inst_p0[5] = combined_coeffs[0];
                dbg_inst_p1[5] = combined_coeffs[1];

                if (debug_r5) {
                    const e0 = combined_coeffs[0].toBytes();
                    const e1 = combined_coeffs[1].toBytes();
                    dbg("  [R5_DBG] after inst5: c[0]_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}] c[1]_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                        e0[0], e0[1], e0[2], e0[3], e0[4], e0[5], e0[6], e0[7],
                        e1[0], e1[1], e1[2], e1[3], e1[4], e1[5], e1[6], e1[7],
                    });
                    // In monomial form, p(0)+p(1) = 2*c0 + c1 + c2 + ... + cd
                    var sum = combined_coeffs[0].add(combined_coeffs[0]); // 2*c0
                    for (1..max_degree + 1) |ci| sum = sum.add(combined_coeffs[ci]); // + c1 + c2 + ... + cd
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
                // In monomial form: p(0)+p(1) = 2*c0 + c1 + c2 + ... + cd
                if (comptime debug_verbose) {
                    var p01_sum = combined_coeffs[0].add(combined_coeffs[0]); // 2*c0
                    for (1..max_degree + 1) |cii| p01_sum = p01_sum.add(combined_coeffs[cii]);
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

                // Debug: print monomial coefficients for round 7
                if (comptime debug_verbose) {
                    if (round == 7) {
                        dbg("  [S6P] R7 monomial coeffs:\n", .{});
                        for (0..max_degree + 1) |ci_idx| {
                            const ci_le = combined_coeffs[ci_idx].toBytes();
                            dbg("    c[{}]=[", .{ci_idx});
                            for (0..32) |bi| dbg("{x:0>2}", .{ci_le[bi]});
                            dbg("]\n", .{});
                        }
                        // p(0)+p(1) = 2*c0 + c1 + c2 + ... + cd
                        var sum01 = combined_coeffs[0].add(combined_coeffs[0]);
                        for (1..max_degree + 1) |ci_idx| sum01 = sum01.add(combined_coeffs[ci_idx]);
                        const sum_le = sum01.toBytes();
                        const hint_le = current_batched_claim.toBytes();
                        dbg("    p(0)+p(1)=[", .{});
                        for (0..32) |bi| dbg("{x:0>2}", .{sum_le[bi]});
                        dbg("]\n    hint    =[", .{});
                        for (0..32) |bi| dbg("{x:0>2}", .{hint_le[bi]});
                        dbg("]\n    match={}\n", .{sum01.eql(current_batched_claim)});
                    }
                }

                if (comptime s6_bench_timing) s6_t_compute[5] += s6_timer.read();
                if (comptime s6_bench_timing) s6_timer.reset();
                // Compress: strip c1 (linear term) from monomial coefficients
                // compressed = [c0, c2, c3, ..., c_d] (same as Jolt's UniPoly::compress)
                const compressed = try self.allocator.alloc(F, max_degree);
                defer self.allocator.free(compressed);
                compressed[0] = combined_coeffs[0]; // c0
                for (1..max_degree) |ci_idx| {
                    compressed[ci_idx] = combined_coeffs[ci_idx + 1]; // c2, c3, ..., c_d
                }

                // Debug: print compressed coefficients LE for ALL rounds
                if (comptime debug_verbose) {
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
                if (comptime debug_verbose) {
                    if (round == 0) {
                        const diag_file = std.fs.cwd().createFile("/tmp/s6p_diag.bin", .{}) catch null;
                        if (diag_file) |f| {
                            defer f.close();
                            f.writeAll(&transcript.state) catch {};
                            for (0..num_compressed) |j| {
                                const le = coeffs[j].toBytes();
                                f.writeAll(&le) catch {};
                            }
                        }
                    }
                }

                transcript.appendScalars("sumcheck_poly", coeffs[0..num_compressed]);

                // Dump transcript state AFTER appending R0 polynomial
                if (comptime debug_verbose) {
                    if (round == 0) {
                        const diag_after = std.fs.cwd().createFile("/tmp/s6p_state_after_r0.bin", .{}) catch null;
                        if (diag_after) |fa| {
                            defer fa.close();
                            fa.writeAll(&transcript.state) catch {};
                            var nr_buf: [4]u8 = undefined;
                            std.mem.writeInt(u32, &nr_buf, transcript.n_rounds, .little);
                            fa.writeAll(&nr_buf) catch {};
                        }
                    }
                }

                const challenge = transcript.challengeScalar();
                challenges[round] = challenge;

                // Write R0 challenge to diagnostic file
                if (comptime debug_verbose) {
                    if (round == 0) {
                        const diag2 = std.fs.cwd().createFile("/tmp/s6p_r0_challenge.bin", .{}) catch null;
                        if (diag2) |f2| {
                            defer f2.close();
                            const ch_le = challenge.toBytes();
                            f2.writeAll(&ch_le) catch {};
                        }
                    }
                }

                // Evaluate combined polynomial at challenge using evalFromHintGeneral
                current_batched_claim = UniPoly(F).evalFromHintGeneral(coeffs[0..num_compressed], current_batched_claim, challenge);

                if (comptime debug_verbose) {
                    // Verify: directly evaluate combined_coeffs at challenge via Horner
                    var direct_eval = combined_coeffs[max_degree];
                    {
                        var ci_rev = max_degree;
                        while (ci_rev > 0) {
                            ci_rev -= 1;
                            direct_eval = direct_eval.mul(challenge).add(combined_coeffs[ci_rev]);
                        }
                    }
                    const efh_match = direct_eval.eql(current_batched_claim);
                    if (!efh_match) {
                        const efh_le = direct_eval.toBytes();
                        const vdm_le = current_batched_claim.toBytes();
                        dbg("  [S6P] R{} EVAL_MISMATCH! direct_eval=[", .{round});
                        for (0..32) |bi| dbg("{x:0>2}", .{efh_le[bi]});
                        dbg("]\n  [S6P] R{} EVAL_MISMATCH! evalFromHint=[", .{round});
                        for (0..32) |bi| dbg("{x:0>2}", .{vdm_le[bi]});
                        dbg("]\n", .{});
                        dbg("  [S6P] R{} num_compressed={}, compressed.len={}\n", .{ round, num_compressed, compressed.len });
                    }
                    dbg("  [S6P] R{} efh_match={}\n", .{ round, @intFromBool(efh_match) });
                }

                if (comptime debug_verbose) {
                    const ch_le = challenge.toBytes();
                    const cl_le = current_batched_claim.toBytes();
                    dbg("  [S6P] R{} challenge_LE=[", .{round});
                    for (0..32) |bi| dbg("{x:0>2}", .{ch_le[bi]});
                    dbg("]\n", .{});
                    dbg("  [S6P] R{} new_claim_LE=[", .{round});
                    for (0..32) |bi| dbg("{x:0>2}", .{cl_le[bi]});
                    dbg("]\n", .{});
                }

                if (comptime s6_bench_timing) s6_t_transcript += s6_timer.read();
                // Update per-instance claims from CACHED round polys and bind challenge
                // Instance 0: BytecodeReadRaf
                if (comptime s6_bench_timing) s6_timer.reset();
                if (inst_active[0]) {
                    if (bytecode_prover.phase == 0) {
                        // Phase 1: degree-2 poly, p(r) = a0 + a1*r + a2*r^2
                        const bc_a0 = cached_bc_phase1_coeffs[0];
                        const bc_a1 = cached_bc_phase1_coeffs[1];
                        const bc_a2 = cached_bc_phase1_coeffs[2];
                        instance_claims[0] = bc_a0.add(challenge.mul(bc_a1.add(challenge.mul(bc_a2))));
                        if (comptime debug_verbose) {
                            const ic_le = instance_claims[0].toBytes();
                            dbg("  [S6P] R{} inst0_from_poly_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                                round, ic_le[0], ic_le[1], ic_le[2], ic_le[3], ic_le[4], ic_le[5], ic_le[6], ic_le[7],
                            });
                        }
                        bytecode_addr_challenges[bytecode_prover.addr_rounds_done] = challenge;
                        bytecode_prover.bindChallengePhase1(challenge, cached_bc_phase1_per_stage);
                        if (comptime debug_verbose) {
                            // Check invariant: instance_claims[0] == Σ gamma^s * stage_claims[s]
                            var agg_check = F.zero();
                            for (0..5) |si| {
                                agg_check = agg_check.add(bytecode_prover.gamma_powers[si].mul(bytecode_prover.stage_claims[si]));
                            }
                            const ac_le = agg_check.toBytes();
                            const ic_le2 = instance_claims[0].toBytes();
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
                            const bc_a0_ = cached_bc_phase1_coeffs[0];
                            const bc_a1_ = cached_bc_phase1_coeffs[1];
                            const bc_a2_ = cached_bc_phase1_coeffs[2];
                            const manual_eval = bc_a0_.add(challenge.mul(bc_a1_.add(challenge.mul(bc_a2_))));
                            const me_le = manual_eval.toBytes();
                            dbg("[INVARIANT_CHECK] R{} manual_eval_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}] match_inst={}\n", .{
                                round,
                                me_le[0], me_le[1], me_le[2], me_le[3], me_le[4], me_le[5], me_le[6], me_le[7],
                                @as(u8, if (manual_eval.eql(instance_claims[0])) 1 else 0),
                            });
                        }
                        if (bytecode_prover.addr_rounds_done == bytecode_log_k) {
                            if (comptime debug_verbose) {
                                // BEFORE transition: check Σ_s gamma^s * stage_claims[s] vs instance_claims[0]
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
                                for (0..5) |si| {
                                    const sc_le2 = bytecode_prover.stage_claims[si].toBytes();
                                    dbg("[PHASE_TRANSITION_PRE] stage[{}]_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                                        si, sc_le2[0], sc_le2[1], sc_le2[2], sc_le2[3], sc_le2[4], sc_le2[5], sc_le2[6], sc_le2[7],
                                    });
                                }
                            }
                            try bytecode_prover.transitionToPhase2(bytecode_addr_challenges);
                            if (comptime debug_verbose) {
                                // After transition, check Phase 2 polynomial sum
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
                                dbg("[PHASE_TRANSITION] inst0 claim match={}\n", .{
                                    @as(u8, if (instance_claims[0].eql(phase2_sum)) 1 else 0),
                                });
                                _ = ic_old_le;
                                _ = p2_le;
                            }
                        }
                    } else {
                        // Phase 2: evaluate from cached evals using Lagrange interpolation
                        instance_claims[0] = UniPoly(F).evalFromEvalsGeneral(cached_bc_phase2.?, challenge);
                        self.allocator.free(cached_bc_phase2.?);
                        cached_bc_phase2 = null;
                        bytecode_prover.bindChallengePhase2(challenge);

                    }
                }

                if (comptime s6_bench_timing) s6_t_bind[0] += s6_timer.read();
                // Instance 1: Booleanity (real prover)
                if (comptime s6_bench_timing) s6_timer.reset();
                if (inst_active[1]) {
                    if (cached_booleanity) |polys| {
                        // Evaluate degree-3 poly at challenge from Vandermonde [p(0), p(1), p(2), p(3)]
                        const evals_arr = [4]F{ polys[0], polys[1], polys[2], polys[3] };
                        instance_claims[1] = UniPoly(F).evalFromEvalsDeg3(evals_arr, challenge);
                        self.allocator.free(polys);
                        cached_booleanity = null;
                    }
                    try booleanity_prover.bindChallenge(challenge);
                    if (comptime debug_verbose) {
                        if (booleanity_prover.round == booleanity_prover.log_k_chunk) {
                            const ic1_be = instance_claims[1].toBytesBE();
                            dbg("[BOOL_TRANSITION] inst_claim[1] after Ph1 LE=[{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}]\n", .{
                                ic1_be[31], ic1_be[30], ic1_be[29], ic1_be[28], ic1_be[27], ic1_be[26], ic1_be[25], ic1_be[24],
                            });
                        }
                    }
                }

                if (comptime s6_bench_timing) s6_t_bind[1] += s6_timer.read();
                // Instance 2: HammingBooleanity
                if (comptime s6_bench_timing) s6_timer.reset();
                if (inst_active[2]) {
                    instance_claims[2] = UniPoly(F).evalFromEvalsDeg3(cached_hamming, challenge);
                    hamming_prover.bindChallenge(challenge);
                }

                if (comptime s6_bench_timing) s6_t_bind[2] += s6_timer.read();
                // Instance 3: RamRaVirtual
                if (comptime s6_bench_timing) s6_timer.reset();
                if (inst_active[3]) {
                    // Monomial coefficients — evaluate via Horner's method
                    const ram_mono = cached_ram_ra.?;
                    var ram_val = ram_mono[ram_mono.len - 1];
                    var ram_ci: usize = ram_mono.len - 1;
                    while (ram_ci > 0) {
                        ram_ci -= 1;
                        ram_val = ram_val.mul(challenge).add(ram_mono[ram_ci]);
                    }
                    instance_claims[3] = ram_val;
                    self.allocator.free(ram_mono);
                    cached_ram_ra = null;
                    try ram_ra_prover.bindChallenge(challenge);
                }

                if (comptime s6_bench_timing) s6_t_bind[3] += s6_timer.read();
                // Instance 4: LookupsRaVirtual
                if (comptime s6_bench_timing) s6_timer.reset();
                if (inst_active[4]) {
                    // Monomial coefficients — evaluate via Horner's method
                    const mono = cached_lookups_ra.?;
                    var val = mono[mono.len - 1];
                    var ci: usize = mono.len - 1;
                    while (ci > 0) {
                        ci -= 1;
                        val = val.mul(challenge).add(mono[ci]);
                    }
                    instance_claims[4] = val;
                    self.allocator.free(mono);
                    cached_lookups_ra = null;
                    try lookups_ra_prover.bindChallenge(challenge);
                }

                if (comptime s6_bench_timing) s6_t_bind[4] += s6_timer.read();
                // Instance 5: IncClaimReduction
                if (comptime s6_bench_timing) s6_timer.reset();
                if (inst_active[5]) {
                    instance_claims[5] = UniPoly(F).evalFromEvalsDeg2(cached_inc, challenge);

                    try inc_prover.bindChallenge(challenge);
                }
                if (comptime s6_bench_timing) s6_t_bind[5] += s6_timer.read();

                // NOTE: Instance claims for inactive instances are NOT halved here.
                // In Zolt, instance_claims starts at the UNSCALED input_claims (not 2^offset-scaled),
                // and the inactive round contributions are computed directly from input_claims with
                // the correct power-of-2 scaling. When an instance first becomes active,
                // instance_claims[i] = input_claims[i] = the correct unscaled claim.
            }


            if (comptime s6_bench_timing) {
                const names = [6][]const u8{ "BcRaf", "Bool ", "Hamm ", "RamRa", "LkRa ", "Inc  " };
                var total_compute: u64 = 0;
                var total_bind: u64 = 0;
                for (0..6) |i| {
                    total_compute += s6_t_compute[i];
                    total_bind += s6_t_bind[i];
                }
                std.debug.print("\n    [STAGE6-BENCH] Per-instance timing (compute + bind):\n", .{});
                for (0..6) |i| {
                    std.debug.print("    [STAGE6-BENCH]   {s}: compute={d:7.1}ms  bind={d:7.1}ms  total={d:7.1}ms\n", .{
                        names[i],
                        @as(f64, @floatFromInt(s6_t_compute[i])) / 1_000_000.0,
                        @as(f64, @floatFromInt(s6_t_bind[i])) / 1_000_000.0,
                        @as(f64, @floatFromInt(s6_t_compute[i] + s6_t_bind[i])) / 1_000_000.0,
                    });
                }
                std.debug.print("    [STAGE6-BENCH]   transcript+compress: {d:7.1}ms\n", .{
                    @as(f64, @floatFromInt(s6_t_transcript)) / 1_000_000.0,
                });
                std.debug.print("    [STAGE6-BENCH]   TOTAL: compute={d:7.1}ms  bind={d:7.1}ms  other={d:7.1}ms\n", .{
                    @as(f64, @floatFromInt(total_compute)) / 1_000_000.0,
                    @as(f64, @floatFromInt(total_bind)) / 1_000_000.0,
                    @as(f64, @floatFromInt(s6_t_transcript)) / 1_000_000.0,
                });
            }

            // ====================================================================
            // Extract opening claims from all real provers
            // ====================================================================

            const inc_opening = inc_prover.openingClaims();
            const ram_inc_claim = inc_opening.ram_inc;
            const rd_inc_claim = inc_opening.rd_inc;
            if (comptime debug_verbose) {
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
            if (comptime debug_verbose) {
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
            if (comptime debug_verbose) {
                const total_booleanity_polys = instruction_d + bytecode_d + ram_d;
                dbg("[STAGE6] Booleanity claims from H final state:\n", .{});
                for (0..@min(5, total_booleanity_polys)) |i| {
                    const brc_be = booleanity_ra_claims[i].toBytesBE();
                    dbg("  bool_claim[{}]_LE=[", .{i});
                    for (0..8) |bi| dbg("{x:0>2}", .{brc_be[31 - bi]});
                    dbg("]\n", .{});
                }
            }

            if (comptime debug_verbose) {
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
            } // end if (comptime debug_verbose) for BOOL_VERIFY

            if (comptime debug_verbose) {
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
            } // end if (comptime debug_verbose)

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
            const NUM_LOOKUP_TABLES: usize = 40;
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
// Helper: Convert evaluations to monomial coefficients and add batch*coeffs to combined_coeffs
// =============================================================================
// Converts [p(0), p(1), ..., p(d)] (Vandermonde evals) to monomial [c0, c1, ..., cd]
// using finite differences for small degrees (d <= 3), then adds batch * c_i to combined_coeffs[i].
fn addEvalsAsMonomialToCoeffs(comptime F: type, combined_coeffs: []F, polys: []const F, n_evals: usize, batch_coeff: F) void {
    if (n_evals == 1) {
        // Degree 0: c0 = p(0)
        combined_coeffs[0] = combined_coeffs[0].add(batch_coeff.mul(polys[0]));
    } else if (n_evals == 2) {
        // Degree 1: c0 = p(0), c1 = p(1) - p(0)
        const c0 = polys[0];
        const c1 = polys[1].sub(polys[0]);
        combined_coeffs[0] = combined_coeffs[0].add(batch_coeff.mul(c0));
        combined_coeffs[1] = combined_coeffs[1].add(batch_coeff.mul(c1));
    } else if (n_evals == 3) {
        // Degree 2: c0 = p(0), c2 = (p(2) - 2p(1) + p(0)) / 2, c1 = p(1) - p(0) - c2
        const inv2 = F.fromU64(2).inverse().?;
        const c0 = polys[0];
        const c2 = polys[2].sub(polys[1]).sub(polys[1]).add(polys[0]).mul(inv2);
        const c1 = polys[1].sub(polys[0]).sub(c2);
        combined_coeffs[0] = combined_coeffs[0].add(batch_coeff.mul(c0));
        combined_coeffs[1] = combined_coeffs[1].add(batch_coeff.mul(c1));
        combined_coeffs[2] = combined_coeffs[2].add(batch_coeff.mul(c2));
    } else if (n_evals == 4) {
        // Degree 3: finite differences
        const inv2 = F.fromU64(2).inverse().?;
        const inv6 = F.fromU64(6).inverse().?;
        const c0 = polys[0];
        const d1 = polys[1].sub(polys[0]);
        const d2 = polys[2].sub(polys[1]);
        const d3 = polys[3].sub(polys[2]);
        const dd1 = d2.sub(d1);
        const dd2 = d3.sub(d2);
        const c3 = dd2.sub(dd1).mul(inv6);
        const c2 = dd1.mul(inv2).sub(c3.mul(F.fromU64(3)));
        const c1 = d1.sub(c2).sub(c3);
        combined_coeffs[0] = combined_coeffs[0].add(batch_coeff.mul(c0));
        combined_coeffs[1] = combined_coeffs[1].add(batch_coeff.mul(c1));
        combined_coeffs[2] = combined_coeffs[2].add(batch_coeff.mul(c2));
        combined_coeffs[3] = combined_coeffs[3].add(batch_coeff.mul(c3));
    } else {
        // General case: use Newton forward differences with static buffer
        // Supports up to degree 15 (16 eval points)
        std.debug.assert(n_evals <= 16);
        var dd: [16]F = undefined;
        for (0..n_evals) |i| dd[i] = polys[i];

        // Build forward difference table: dd[k] = k-th order forward difference at 0
        // After processing, dd[k] = Δ^k p(0)
        var coeffs_buf: [16]F = undefined;
        coeffs_buf[0] = dd[0]; // Δ^0 = p(0)

        var order: usize = 1;
        while (order < n_evals) : (order += 1) {
            // Compute order-th forward differences in-place
            var i = n_evals - 1;
            while (i >= order) : (i -= 1) {
                dd[i] = dd[i].sub(dd[i - 1]);
                if (i == order) break;
            }
            coeffs_buf[order] = dd[order]; // Δ^order p(0)
        }

        // Convert Newton forward differences to monomial coefficients
        // Newton form: p(x) = Σ_k Δ^k p(0) * C(x, k)
        // where C(x, k) = x(x-1)...(x-k+1) / k!
        // We need to convert to monomial c0 + c1*x + c2*x^2 + ...
        // Use the fact that Δ^k p(0) / k! is the leading coefficient contribution
        // Actually, the simplest approach for general n: use the Vandermonde solver result
        // which is already available via fromEvalsVandermonde. But since this is a non-allocating
        // path, we use Sterling numbers of the first kind.
        //
        // Actually for the general case, let's just compute monomial coefficients directly
        // from the forward differences using the Stirling number relationship.
        // c_j = Σ_{k=j}^{d} S1(k, j) * Δ^k p(0) / k!
        // This is complex. For now, fall back to evaluating the Newton form at integer points
        // and using the same approach as vandermondeToCompressed for n > 4.
        //
        // Simpler: we have forward differences. Convert via the standard formula:
        // The Newton forward difference interpolation gives:
        // c_k = Σ_{j=0}^{k} (-1)^{k-j} C(k,j) * Δ^j p(0) / ... no, this is circular.
        //
        // Let's just directly use finite-difference-to-monomial conversion:
        // Start with Newton basis coefficients dd[0..n] = [Δ^0 p(0)/0!, Δ^1 p(0)/1!, ...]
        // and convert to monomial via the standard algorithm.

        // Divide by factorials to get Newton basis coefficients
        var fact = F.one();
        for (1..n_evals) |k| {
            fact = fact.mul(F.fromU64(@intCast(k)));
            coeffs_buf[k] = coeffs_buf[k].mul(fact.inverse().?);
        }

        // Convert Newton basis to monomial: c(x) = Σ a_k * x*(x-1)*...*(x-k+1)
        // Process from highest to lowest degree, expanding x*(x-1)*...*(x-k+1) into monomials.
        // Use the recurrence: multiply running polynomial by (x - k) at each step.
        var mono: [16]F = .{F.zero()} ** 16;
        mono[0] = coeffs_buf[0];

        for (1..n_evals) |k| {
            // We need to add coeffs_buf[k] * x*(x-1)*...*(x-k+1) to mono
            // Build the falling factorial x*(x-1)*...*(x-k+1) incrementally
            // ff[k] = ff[k-1] * (x - (k-1))
            // We maintain ff_mono[0..k] = monomial coefficients of x*(x-1)*...*(x-k+1)
            // Start: ff_mono = [0, 1] for x
            // Multiply by (x - j) for j = 1, 2, ..., k-1
            var ff: [16]F = .{F.zero()} ** 16;
            ff[1] = F.one(); // x
            for (1..k) |j| {
                // Multiply ff by (x - j): new[i] = ff[i-1] - j*ff[i]
                const neg_j = F.zero().sub(F.fromU64(@intCast(j)));
                var i_rev = j + 1;
                while (i_rev > 0) {
                    i_rev -= 1;
                    const prev = if (i_rev > 0) ff[i_rev - 1] else F.zero();
                    ff[i_rev] = prev.add(neg_j.mul(ff[i_rev]));
                }
            }
            // Add coeffs_buf[k] * ff to mono
            for (0..k + 1) |i| {
                mono[i] = mono[i].add(coeffs_buf[k].mul(ff[i]));
            }
        }


        // Add batch * mono to combined_coeffs
        for (0..n_evals) |i| {
            combined_coeffs[i] = combined_coeffs[i].add(batch_coeff.mul(mono[i]));
        }
    }
}

// =============================================================================
// Helper: Add variable-length instance evals to combined_evals with interpolation (LEGACY)
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
        if (funct3 == 0) {
            // VirtualMULI: MultiplyOperands → rs1 * (1 << shamt)
            const shamt_raw: u32 = instr >> 20;
            const shamt: u6 = @truncate(shamt_raw & 0x3F);
            const multiplier: u128 = @as(u128, 1) << shamt;
            return @as(u128, step.rs1_value) * multiplier;
        } else {
            // VirtualPow2 (funct3=1), VirtualShiftRightBitmask (funct3=2): AddOperands → rs1 + 0 = rs1
            return @as(u128, step.rs1_value);
        }
    }
    if (opcode == 0x5B) {
        if (step.rs2_read) {
            // VirtualSRL/VirtualSRA R-type: interleaved(rs1_value, rs2_value)
            return interleaveBits(step.rs1_value, step.rs2_value);
        } else {
            // VirtualSRLI/VirtualSRAI I-type: interleaved(rs1_value, bitmask)
            const total_shift_raw: u32 = instr >> 20;
            const total_shift: u7 = @truncate(total_shift_raw & 0x3F);
            const ones: u128 = (@as(u128, 1) << @intCast(64 - @as(u8, total_shift))) - 1;
            const bitmask: u64 = @truncate(ones << total_shift);
            return interleaveBits(step.rs1_value, bitmask);
        }
    }
    if (opcode == 0x02) {
        // VirtualAdvice: the lookup index is the advice value (rd_value)
        // Jolt's to_lookup_index() returns the second operand which is the advice value
        return @as(u128, step.rd_value);
    }
    if (opcode == 0x22) {
        if (funct3 == 2 or funct3 == 3) {
            // VirtualAssertHalfwordAlignment/WordAlignment: AddOperands → rs1 + imm
            const imm_raw: u32 = instr >> 20;
            const imm_signed: i64 = @as(i64, @as(i32, @bitCast(imm_raw << 20)) >> 20);
            return @as(u128, step.rs1_value +% @as(u64, @bitCast(imm_signed)));
        } else {
            // VirtualAssertEQ (funct3=0) / VirtualAssertValidDiv0 (funct3=1): interleaved
            return interleaveBits(step.rs1_value, step.rs2_value);
        }
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

// =============================================================================
// Tests
// =============================================================================

const testing = std.testing;
const BN254Scalar = @import("../../field/mod.zig").BN254Scalar;

/// Helper: compute eq(r, x) directly for a boolean vector x and field vector r.
/// Both in LE order (r[0] = LSB, matching computeEqTable's output convention).
fn eqEvalDirect(r: []const BN254Scalar, x: usize) BN254Scalar {
    var result = BN254Scalar.one();
    for (0..r.len) |i| {
        const bit: u1 = @truncate(x >> @intCast(i));
        if (bit == 1) {
            result = result.mul(r[i]);
        } else {
            result = result.mul(BN254Scalar.one().sub(r[i]));
        }
    }
    return result;
}

test "split-eq factorization: eq_lo * eq_hi = eq_full" {
    // Verify the core split-eq identity:
    //   eq(r, x) = eq(r_lo, x_lo) * eq(r_hi, x_hi)
    // where x = x_lo + x_hi << prefix_n_vars
    //
    // computeEqTable takes BE input r[0..n], output table[j] has bit i → r[i].
    // For x = x_lo | (x_hi << prefix_n_vars):
    //   bits 0..prefix_n_vars-1 (x_lo) → r_be[0..prefix_n_vars]
    //   bits prefix_n_vars..n_vars-1 (x_hi) → r_be[prefix_n_vars..n_vars]
    const allocator = testing.allocator;
    const F = BN254Scalar;

    const n_vars = 4;
    const prefix_n_vars = 2;
    const suffix_n_vars = 2;
    const T: usize = 1 << n_vars;
    const prefix_len: usize = 1 << prefix_n_vars;
    const suffix_len: usize = 1 << suffix_n_vars;

    // Full BE challenge
    var r_be = [4]F{ F.fromU64(17), F.fromU64(31), F.fromU64(7), F.fromU64(53) };
    const eq_full = try computeEqTable(F, allocator, &r_be, n_vars);
    defer allocator.free(eq_full);

    // Split: prefix (x_lo bits) uses r_be[0..prefix_n_vars]
    var r_lo_be = [2]F{ r_be[0], r_be[1] };
    const eq_lo = try computeEqTable(F, allocator, &r_lo_be, prefix_n_vars);
    defer allocator.free(eq_lo);

    // Suffix (x_hi bits) uses r_be[prefix_n_vars..n_vars]
    var r_hi_be = [2]F{ r_be[2], r_be[3] };
    const eq_hi = try computeEqTable(F, allocator, &r_hi_be, suffix_n_vars);
    defer allocator.free(eq_hi);

    // Verify: eq_full[x] == eq_lo[x_lo] * eq_hi[x_hi] for all x
    for (0..T) |x| {
        const x_lo = x & (prefix_len - 1);
        const x_hi = x >> prefix_n_vars;
        const product = eq_lo[x_lo].mul(eq_hi[x_hi]);
        try testing.expect(eq_full[x].eql(product));
    }

    // Also verify: Σ_{x_hi} f(x_lo, x_hi) * eq_hi[x_hi] correctly folds suffix dimension
    var folded = [_]F{F.zero()} ** prefix_len;
    for (0..prefix_len) |x_lo| {
        for (0..suffix_len) |x_hi| {
            const x = x_lo + (x_hi << prefix_n_vars);
            folded[x_lo] = folded[x_lo].add(eq_hi[x_hi].mul(F.fromU64(@intCast(x))));
        }
    }
    // Verify: Σ_x_lo P[x_lo] * folded[x_lo] == Σ_x eq_full[x] * f(x)
    var sum_pq = F.zero();
    for (0..prefix_len) |x_lo| {
        sum_pq = sum_pq.add(eq_lo[x_lo].mul(folded[x_lo]));
    }
    var sum_direct = F.zero();
    for (0..T) |x| {
        sum_direct = sum_direct.add(eq_full[x].mul(F.fromU64(@intCast(x))));
    }
    try testing.expect(sum_pq.eql(sum_direct));
}

test "split-eq bind Phase 1 then Phase 2 matches flat eq bind" {
    // Verify that binding a split eq (Phase 1 prefix, then Phase 2 suffix)
    // produces the same result as binding the flat eq table.
    const allocator = testing.allocator;
    const F = BN254Scalar;

    const n_vars = 4;
    const prefix_n_vars = 2;
    const suffix_n_vars = 2;
    const prefix_len: usize = 1 << prefix_n_vars;
    const suffix_len: usize = 1 << suffix_n_vars;

    var r_be = [4]F{ F.fromU64(5), F.fromU64(13), F.fromU64(3), F.fromU64(19) };
    const challenges = [4]F{ F.fromU64(7), F.fromU64(11), F.fromU64(2), F.fromU64(17) };

    // Build flat eq table and bind sequentially
    var eq_flat = try computeEqTable(F, allocator, &r_be, n_vars);
    defer allocator.free(eq_flat);

    var flat_len: usize = 1 << n_vars;
    for (challenges) |ch| {
        const half = flat_len / 2;
        for (0..half) |j| {
            eq_flat[j] = eq_flat[2 * j].add(ch.mul(eq_flat[2 * j + 1].sub(eq_flat[2 * j])));
        }
        flat_len = half;
    }
    const flat_final = eq_flat[0];

    // Split: prefix uses r_be[0..prefix_n_vars], suffix uses r_be[prefix_n_vars..]
    var r_lo_be = [2]F{ r_be[0], r_be[1] };
    var eq_lo = try computeEqTable(F, allocator, &r_lo_be, prefix_n_vars);
    defer allocator.free(eq_lo);

    var r_hi_be = [2]F{ r_be[2], r_be[3] };
    var eq_hi = try computeEqTable(F, allocator, &r_hi_be, suffix_n_vars);
    defer allocator.free(eq_hi);

    // Phase 1: bind prefix rounds on eq_lo
    var lo_len = prefix_len;
    for (0..prefix_n_vars) |round| {
        const half = lo_len / 2;
        for (0..half) |j| {
            eq_lo[j] = eq_lo[2 * j].add(challenges[round].mul(eq_lo[2 * j + 1].sub(eq_lo[2 * j])));
        }
        lo_len = half;
    }
    const eq_lo_scalar = eq_lo[0];

    // Phase 2: scale eq_hi by eq_lo scalar and bind suffix rounds
    for (0..suffix_len) |j| {
        eq_hi[j] = eq_hi[j].mul(eq_lo_scalar);
    }
    var hi_len = suffix_len;
    for (0..suffix_n_vars) |round| {
        const half = hi_len / 2;
        for (0..half) |j| {
            eq_hi[j] = eq_hi[2 * j].add(challenges[prefix_n_vars + round].mul(eq_hi[2 * j + 1].sub(eq_hi[2 * j])));
        }
        hi_len = half;
    }
    const split_final = eq_hi[0];

    try testing.expect(flat_final.eql(split_final));
}

test "P*Q sum matches flat polynomial sum" {
    // Verify that Σ P[x_lo] * Q[x_lo] == Σ_x eq(r, x) * f(x)
    // where Q[x_lo] = Σ_{x_hi} eq_hi(r_hi, x_hi) * f(x_lo, x_hi)
    // This is the IncClaimReduction Phase 1 correctness property.
    const allocator = testing.allocator;
    const F = BN254Scalar;

    const n_vars = 6;
    const prefix_n_vars = 3;
    const suffix_n_vars = 3;
    const T: usize = 1 << n_vars;
    const prefix_len: usize = 1 << prefix_n_vars;
    const suffix_len: usize = 1 << suffix_n_vars;

    var r_be = [6]F{
        F.fromU64(3), F.fromU64(7), F.fromU64(11),
        F.fromU64(17), F.fromU64(23), F.fromU64(29),
    };

    const eq_full = try computeEqTable(F, allocator, &r_be, n_vars);
    defer allocator.free(eq_full);

    // Prefix uses r_be[0..prefix_n_vars], suffix uses r_be[prefix_n_vars..]
    var r_lo_be = [3]F{ r_be[0], r_be[1], r_be[2] };
    const eq_lo = try computeEqTable(F, allocator, &r_lo_be, prefix_n_vars);
    defer allocator.free(eq_lo);

    var r_hi_be = [3]F{ r_be[3], r_be[4], r_be[5] };
    const eq_hi = try computeEqTable(F, allocator, &r_hi_be, suffix_n_vars);
    defer allocator.free(eq_hi);

    // f(x) = x^2 + 3x + 1 (arbitrary polynomial for testing)
    var f_vals = try allocator.alloc(F, T);
    defer allocator.free(f_vals);
    for (0..T) |x| {
        const xf = F.fromU64(@intCast(x));
        f_vals[x] = xf.mul(xf).add(F.fromU64(3).mul(xf)).add(F.one());
    }

    // Q[x_lo] = Σ_{x_hi} eq_hi[x_hi] * f(x_lo + x_hi << prefix_n_vars)
    var Q = try allocator.alloc(F, prefix_len);
    defer allocator.free(Q);
    for (0..prefix_len) |x_lo| {
        Q[x_lo] = F.zero();
        for (0..suffix_len) |x_hi| {
            const x = x_lo + (x_hi << prefix_n_vars);
            Q[x_lo] = Q[x_lo].add(eq_hi[x_hi].mul(f_vals[x]));
        }
    }

    // Σ P[x_lo] * Q[x_lo]
    var sum_pq = F.zero();
    for (0..prefix_len) |x_lo| {
        sum_pq = sum_pq.add(eq_lo[x_lo].mul(Q[x_lo]));
    }

    // Σ eq_full[x] * f(x)
    var sum_direct = F.zero();
    for (0..T) |x| {
        sum_direct = sum_direct.add(eq_full[x].mul(f_vals[x]));
    }

    try testing.expect(sum_pq.eql(sum_direct));
}

test "P*Q Phase 1 sumcheck round polynomial matches flat" {
    // Verify that the Phase 1 round polynomial from the P*Q factorization
    // produces the same evaluations as computing from the flat polynomial.
    const allocator = testing.allocator;
    const F = BN254Scalar;

    const n_vars = 4;
    const prefix_n_vars = 2;
    const suffix_n_vars = 2;
    const T: usize = 1 << n_vars;
    const prefix_len: usize = 1 << prefix_n_vars;
    const suffix_len: usize = 1 << suffix_n_vars;

    var r_be = [4]F{ F.fromU64(5), F.fromU64(13), F.fromU64(3), F.fromU64(19) };

    // Build flat polynomial: poly[x] = eq(r, x) * f(x)
    const eq_full = try computeEqTable(F, allocator, &r_be, n_vars);
    defer allocator.free(eq_full);

    // f(x) = x + 1
    var poly = try allocator.alloc(F, T);
    defer allocator.free(poly);
    for (0..T) |x| {
        poly[x] = eq_full[x].mul(F.fromU64(@intCast(x + 1)));
    }

    // Flat round 1: p(0) = Σ poly[2j], p(1) = Σ poly[2j+1]
    var flat_p0 = F.zero();
    var flat_p1 = F.zero();
    for (0..T / 2) |j| {
        flat_p0 = flat_p0.add(poly[2 * j]);
        flat_p1 = flat_p1.add(poly[2 * j + 1]);
    }

    // Split: P * Q version (prefix = r_be[0..2], suffix = r_be[2..4])
    var r_lo_be = [2]F{ r_be[0], r_be[1] };
    const P = try computeEqTable(F, allocator, &r_lo_be, prefix_n_vars);
    defer allocator.free(P);

    var r_hi_be = [2]F{ r_be[2], r_be[3] };
    const eq_hi = try computeEqTable(F, allocator, &r_hi_be, suffix_n_vars);
    defer allocator.free(eq_hi);

    var Q = try allocator.alloc(F, prefix_len);
    defer allocator.free(Q);
    for (0..prefix_len) |x_lo| {
        Q[x_lo] = F.zero();
        for (0..suffix_len) |x_hi| {
            const x = x_lo + (x_hi << prefix_n_vars);
            Q[x_lo] = Q[x_lo].add(eq_hi[x_hi].mul(F.fromU64(@intCast(x + 1))));
        }
    }

    // Phase 1 round 1: p(t) = Σ_{x_lo} P(x_lo, t) * Q(x_lo, t)
    // P(x_lo, 0) = P[2*x_lo], P(x_lo, 1) = P[2*x_lo+1] (standard MLE bind)
    // Q same structure
    var split_p0 = F.zero();
    var split_p1 = F.zero();
    const half = prefix_len / 2;
    for (0..half) |j| {
        split_p0 = split_p0.add(P[2 * j].mul(Q[2 * j]));
        split_p1 = split_p1.add(P[2 * j + 1].mul(Q[2 * j + 1]));
    }

    try testing.expect(flat_p0.eql(split_p0));
    try testing.expect(flat_p1.eql(split_p1));
}

test "HammingBooleanity split-eq: Phase 1 sum matches flat" {
    // HammingBooleanity computes Σ_x eq(r, x) * H(x) * (H(x) - 1)
    // Verify split-eq Phase 1 round poly matches flat computation.
    const allocator = testing.allocator;
    const F = BN254Scalar;

    const n_vars = 4;
    const prefix_n_vars = 2;
    const suffix_n_vars = 2;
    const T: usize = 1 << n_vars;
    const prefix_len: usize = 1 << prefix_n_vars;
    const suffix_len: usize = 1 << suffix_n_vars;

    var r_be = [4]F{ F.fromU64(11), F.fromU64(23), F.fromU64(7), F.fromU64(41) };

    // Build flat eq
    const eq_full = try computeEqTable(F, allocator, &r_be, n_vars);
    defer allocator.free(eq_full);

    // H(x) = some test values (simulating Hamming weight or similar)
    var H = [16]F{
        F.fromU64(0), F.fromU64(1), F.fromU64(1), F.fromU64(2),
        F.fromU64(1), F.fromU64(2), F.fromU64(2), F.fromU64(3),
        F.fromU64(1), F.fromU64(2), F.fromU64(2), F.fromU64(3),
        F.fromU64(2), F.fromU64(3), F.fromU64(3), F.fromU64(4),
    };

    // Flat sum: Σ eq(r,x) * H(x) * (H(x) - 1) for degree 3 sumcheck
    // Round 1: p(t) at t=0 and t=1
    var flat_p0 = F.zero();
    var flat_p1 = F.zero();
    for (0..T / 2) |j| {
        flat_p0 = flat_p0.add(eq_full[2 * j].mul(H[2 * j]).mul(H[2 * j].sub(F.one())));
        flat_p1 = flat_p1.add(eq_full[2 * j + 1].mul(H[2 * j + 1]).mul(H[2 * j + 1].sub(F.one())));
    }

    // Split-eq: prefix = r_be[0..2], suffix = r_be[2..4]
    var r_lo_be = [2]F{ r_be[0], r_be[1] };
    const eq_lo = try computeEqTable(F, allocator, &r_lo_be, prefix_n_vars);
    defer allocator.free(eq_lo);

    var r_hi_be = [2]F{ r_be[2], r_be[3] };
    const eq_hi = try computeEqTable(F, allocator, &r_hi_be, suffix_n_vars);
    defer allocator.free(eq_hi);

    // Split round 1 (prefix dimension, bit 0):
    // p(t) = Σ_{x_lo_rest, x_hi} eq_lo(x_lo_rest, t) * eq_hi(x_hi) * H * (H-1)
    // At t=0: sum over even x_lo indices; at t=1: sum over odd x_lo indices
    var split_p0 = F.zero();
    var split_p1 = F.zero();
    const half_lo = prefix_len / 2;
    for (0..half_lo) |j_lo| {
        for (0..suffix_len) |j_hi| {
            const x0 = 2 * j_lo + (j_hi << prefix_n_vars);
            const x1 = 2 * j_lo + 1 + (j_hi << prefix_n_vars);
            const eq_term = eq_lo[2 * j_lo].mul(eq_hi[j_hi]);
            const eq_term1 = eq_lo[2 * j_lo + 1].mul(eq_hi[j_hi]);
            split_p0 = split_p0.add(eq_term.mul(H[x0]).mul(H[x0].sub(F.one())));
            split_p1 = split_p1.add(eq_term1.mul(H[x1]).mul(H[x1].sub(F.one())));
        }
    }

    try testing.expect(flat_p0.eql(split_p0));
    try testing.expect(flat_p1.eql(split_p1));
}

test "IncClaimReduction Phase 1→2 transition: folded suffix matches flat" {
    // Verify that the Phase 1→2 transition math produces the same result as flat computation.
    // All eq tables use LE convention (matching the actual prover which reverses BE→LE first).
    const allocator = testing.allocator;
    const F = BN254Scalar;

    const n_vars = 4;
    const prefix_n_vars = 2;
    const suffix_n_vars = 2;
    const T: usize = 1 << n_vars;
    const prefix_len: usize = 1 << prefix_n_vars;
    const suffix_len: usize = 1 << suffix_n_vars;

    const gamma = F.fromU64(13);
    const challenges = [2]F{ F.fromU64(7), F.fromU64(11) }; // prefix sumcheck challenges

    // 4 opening points in LE order (simulates the prover's reversed BE→LE points).
    // In the prover: r_cycle_rev[i] = r_cycle_be[n_vars - 1 - i].
    // Here we just define them directly in LE.
    var points_le: [4][4]F = undefined;
    points_le[0] = .{ F.fromU64(23), F.fromU64(5), F.fromU64(17), F.fromU64(3) };
    points_le[1] = .{ F.fromU64(19), F.fromU64(2), F.fromU64(11), F.fromU64(7) };
    points_le[2] = .{ F.fromU64(37), F.fromU64(31), F.fromU64(29), F.fromU64(13) };
    points_le[3] = .{ F.fromU64(53), F.fromU64(47), F.fromU64(43), F.fromU64(41) };

    // Build full eq tables for each point (LE input to computeEqTable)
    var eq_full: [4][]F = undefined;
    for (0..4) |i| {
        eq_full[i] = try computeEqTable(F, allocator, &points_le[i], n_vars);
    }
    defer for (0..4) |i| allocator.free(eq_full[i]);

    // Flat approach: eq_ram[x] = eq_0[x] + gamma*eq_1[x], eq_rd[x] = eq_2[x] + gamma*eq_3[x]
    // Then bind prefix variables with challenges to get suffix-sized arrays.
    var flat_eq_ram = try allocator.alloc(F, T);
    defer allocator.free(flat_eq_ram);
    var flat_eq_rd = try allocator.alloc(F, T);
    defer allocator.free(flat_eq_rd);
    for (0..T) |x| {
        flat_eq_ram[x] = eq_full[0][x].add(gamma.mul(eq_full[1][x]));
        flat_eq_rd[x] = eq_full[2][x].add(gamma.mul(eq_full[3][x]));
    }

    // Bind prefix_n_vars rounds (round 0 binds bit 0, round 1 binds bit 1)
    var flat_len: usize = T;
    for (challenges) |ch| {
        const half = flat_len / 2;
        for (0..half) |j| {
            flat_eq_ram[j] = flat_eq_ram[2 * j].add(ch.mul(flat_eq_ram[2 * j + 1].sub(flat_eq_ram[2 * j])));
            flat_eq_rd[j] = flat_eq_rd[2 * j].add(ch.mul(flat_eq_rd[2 * j + 1].sub(flat_eq_rd[2 * j])));
        }
        flat_len = half;
    }

    // Split approach: eq_lo from first prefix_n_vars LE vars, eq_hi from the rest.
    // This mirrors the prover's init which does:
    //   P[i] = computeEqTable(rev_lo, prefix_n_vars) where rev_lo[k] = points_be[n-1-k]
    //   eq_hi[i] = computeEqTable(rev_hi, suffix_n_vars) where rev_hi[k] = points_be[suffix-1-k]
    // In LE terms: lo = points_le[0..prefix_n_vars], hi = points_le[prefix_n_vars..n_vars]
    var eq_hi: [4][]F = undefined;
    for (0..4) |i| {
        var r_hi: [2]F = undefined;
        for (0..suffix_n_vars) |k| r_hi[k] = points_le[i][prefix_n_vars + k];
        eq_hi[i] = try computeEqTable(F, allocator, &r_hi, suffix_n_vars);
    }
    defer for (0..4) |i| allocator.free(eq_hi[i]);

    // Prefix scalars: eq(challenges, point_lo_i) where point_lo = points_le[0..prefix_n_vars]
    var eq_prefix_scalars: [4]F = undefined;
    for (0..4) |i| {
        var result = F.one();
        for (0..prefix_n_vars) |k| {
            const a = challenges[k];
            const b = points_le[i][k];
            const prod = a.mul(b);
            result = result.mul(prod.add(prod).add(F.one()).sub(a.add(b)));
        }
        eq_prefix_scalars[i] = result;
    }

    // Build split eq arrays and compare
    for (0..suffix_len) |x_hi| {
        const split_ram = eq_prefix_scalars[0].mul(eq_hi[0][x_hi]).add(gamma.mul(eq_prefix_scalars[1].mul(eq_hi[1][x_hi])));
        const split_rd = eq_prefix_scalars[2].mul(eq_hi[2][x_hi]).add(gamma.mul(eq_prefix_scalars[3].mul(eq_hi[3][x_hi])));
        try testing.expect(flat_eq_ram[x_hi].eql(split_ram));
        try testing.expect(flat_eq_rd[x_hi].eql(split_rd));
    }

    // Also verify the inc folding: Σ_{x_lo} eq_prefix[x_lo] * f(x_lo, x_hi) matches
    // flat bind of f(x) over prefix variables.
    const eq_prefix_table = try computeEqTable(F, allocator, &challenges, prefix_n_vars);
    defer allocator.free(eq_prefix_table);

    // f(x) = x + 1 (synthetic)
    var f_vals = try allocator.alloc(F, T);
    defer allocator.free(f_vals);
    for (0..T) |x| f_vals[x] = F.fromU64(@intCast(x + 1));

    // Flat bind of f over prefix
    var f_flat = try allocator.alloc(F, T);
    defer allocator.free(f_flat);
    @memcpy(f_flat, f_vals);
    var f_len: usize = T;
    for (challenges) |ch| {
        const half = f_len / 2;
        for (0..half) |j| {
            f_flat[j] = f_flat[2 * j].add(ch.mul(f_flat[2 * j + 1].sub(f_flat[2 * j])));
        }
        f_len = half;
    }

    // Split fold: Σ_{x_lo} eq_prefix[x_lo] * f(x_lo + x_hi << prefix_n_vars)
    for (0..suffix_len) |x_hi| {
        var acc = F.zero();
        for (0..prefix_len) |x_lo| {
            const x = x_lo + (x_hi << prefix_n_vars);
            acc = acc.add(eq_prefix_table[x_lo].mul(f_vals[x]));
        }
        try testing.expect(f_flat[x_hi].eql(acc));
    }
}

test "BytecodeReadRaf split-eq F_s: inner*outer matches flat eq pushforward" {
    // Verify F_s[pc] = Σ_c eq(r_cycle, c) * δ(PC(c)=pc) is the same whether computed
    // via a flat T-sized eq table or via the split-eq double loop with touched-PC tracking.
    const allocator = testing.allocator;
    const F = BN254Scalar;

    const n_vars = 4;
    const T: usize = 1 << n_vars;
    const lo_bits = n_vars / 2;
    const hi_bits = n_vars - lo_bits;
    const in_len: usize = 1 << lo_bits;
    const out_len: usize = 1 << hi_bits;
    const bytecode_K: usize = 8;

    // PC map: cycle c → pc_idx (some synthetic mapping)
    var pc_map_arr: [T]usize = undefined;
    for (0..T) |c| {
        pc_map_arr[c] = (c * 3 + 1) % bytecode_K;
    }

    // r_cycle in LE order (r[0]→LSB, as used by computeEqTable)
    var r_le = [4]F{ F.fromU64(5), F.fromU64(17), F.fromU64(31), F.fromU64(43) };

    // Method 1: Flat computation with full T-sized eq table
    const eq_flat = try computeEqTable(F, allocator, &r_le, n_vars);
    defer allocator.free(eq_flat);

    var F_s_flat: [bytecode_K]F = .{F.zero()} ** bytecode_K;
    for (0..T) |c| {
        F_s_flat[pc_map_arr[c]] = F_s_flat[pc_map_arr[c]].add(eq_flat[c]);
    }

    // Method 2: Split-eq double loop (same algorithm as BytecodeReadRafProver.init)
    // Split LE points into lo and hi halves

    var r_lo_arr = [2]F{ r_le[0], r_le[1] };
    const E_lo = try computeEqTable(F, allocator, &r_lo_arr, lo_bits);
    defer allocator.free(E_lo);

    var r_hi_arr = [2]F{ r_le[2], r_le[3] };
    const E_hi = try computeEqTable(F, allocator, &r_hi_arr, hi_bits);
    defer allocator.free(E_hi);

    var F_s_split: [bytecode_K]F = .{F.zero()} ** bytecode_K;
    var inner_buf: [bytecode_K]F = .{F.zero()} ** bytecode_K;
    var touched_buf: [bytecode_K]usize = undefined;
    var touched_set: [bytecode_K]bool = .{false} ** bytecode_K;

    for (0..out_len) |c_hi| {
        var touched_count: usize = 0;

        for (0..in_len) |c_lo| {
            const c = c_lo + (c_hi << @intCast(lo_bits));
            const pc = pc_map_arr[c];
            if (!touched_set[pc]) {
                touched_set[pc] = true;
                touched_buf[touched_count] = pc;
                touched_count += 1;
            }
            inner_buf[pc] = inner_buf[pc].add(E_lo[c_lo]);
        }

        const e_hi_val = E_hi[c_hi];
        for (0..touched_count) |ti| {
            const pc = touched_buf[ti];
            F_s_split[pc] = F_s_split[pc].add(e_hi_val.mul(inner_buf[pc]));
            inner_buf[pc] = F.zero();
            touched_set[pc] = false;
        }
    }

    for (0..bytecode_K) |k| {
        try testing.expect(F_s_flat[k].eql(F_s_split[k]));
    }
}

test "IncClaimReduction full multi-round: split P/Q matches flat across phase transition" {
    // Full multi-round sumcheck simulation for IncClaimReduction:
    // Phase 1 (prefix rounds on P/Q) → transition → Phase 2 (suffix rounds on dense arrays).
    // The sumcheck is degree 2 (product of two linear factors: eq × inc).
    // We keep the factors separate in the flat reference to properly evaluate the degree-2
    // round polynomial at 3 points [s(0), s(1), s(2)].
    const allocator = testing.allocator;
    const F = BN254Scalar;

    const n_vars = 6;
    const prefix_n_vars = 3;
    const suffix_n_vars = 3;
    const T: usize = 1 << n_vars;
    const prefix_len: usize = 1 << prefix_n_vars;
    const suffix_len: usize = 1 << suffix_n_vars;

    const gamma = F.fromU64(13);
    const gamma_sqr = gamma.mul(gamma);

    // 4 opening points in LE order
    const points_le = [4][6]F{
        .{ F.fromU64(3), F.fromU64(7), F.fromU64(11), F.fromU64(17), F.fromU64(23), F.fromU64(29) },
        .{ F.fromU64(5), F.fromU64(13), F.fromU64(19), F.fromU64(31), F.fromU64(37), F.fromU64(41) },
        .{ F.fromU64(2), F.fromU64(43), F.fromU64(47), F.fromU64(53), F.fromU64(59), F.fromU64(61) },
        .{ F.fromU64(67), F.fromU64(71), F.fromU64(73), F.fromU64(79), F.fromU64(83), F.fromU64(89) },
    };

    // Synthetic inc values
    var ram_inc_vals: [T]F = undefined;
    var rd_inc_vals: [T]F = undefined;
    for (0..T) |x| {
        ram_inc_vals[x] = F.fromU64(@intCast(x + 1));
        rd_inc_vals[x] = F.fromU64(@intCast(2 * x + 3));
    }

    // Build flat eq tables
    var eq_full: [4][]F = undefined;
    for (0..4) |i| {
        eq_full[i] = try computeEqTable(F, allocator, @constCast(&points_le[i]), n_vars);
    }
    defer for (0..4) |i| allocator.free(eq_full[i]);

    // Flat: keep eq and inc separate (4 eq arrays, 2 inc arrays) for degree-2 round poly
    var flat_ram_inc = try allocator.alloc(F, T);
    defer allocator.free(flat_ram_inc);
    var flat_rd_inc = try allocator.alloc(F, T);
    defer allocator.free(flat_rd_inc);
    @memcpy(flat_ram_inc, &ram_inc_vals);
    @memcpy(flat_rd_inc, &rd_inc_vals);

    // --- Split approach: build P, Q arrays ---
    var P: [4][]F = undefined;
    var eq_hi: [4][]F = undefined;
    for (0..4) |i| {
        var r_lo: [3]F = undefined;
        for (0..prefix_n_vars) |k| r_lo[k] = points_le[i][k];
        P[i] = try computeEqTable(F, allocator, &r_lo, prefix_n_vars);

        var r_hi: [3]F = undefined;
        for (0..suffix_n_vars) |k| r_hi[k] = points_le[i][prefix_n_vars + k];
        eq_hi[i] = try computeEqTable(F, allocator, &r_hi, suffix_n_vars);
    }
    defer for (0..4) |i| {
        allocator.free(P[i]);
        allocator.free(eq_hi[i]);
    };

    var Q: [4][]F = undefined;
    for (0..4) |i| {
        Q[i] = try allocator.alloc(F, prefix_len);
        for (0..prefix_len) |x_lo| {
            var acc = F.zero();
            for (0..suffix_len) |x_hi| {
                const x = x_lo + (x_hi << prefix_n_vars);
                const inc_val = if (i < 2) ram_inc_vals[x] else rd_inc_vals[x];
                acc = acc.add(eq_hi[i][x_hi].mul(inc_val));
            }
            Q[i][x_lo] = acc;
        }
    }
    defer for (0..4) |i| allocator.free(Q[i]);

    const gamma_cub = gamma_sqr.mul(gamma);
    const weights = [4]F{ F.one(), gamma, gamma_sqr, gamma_cub };

    var flat_len: usize = T;
    var p_len: usize = prefix_len;
    var challenges: [6]F = undefined;
    var in_phase2 = false;

    var p2_ram_inc: ?[]F = null;
    defer if (p2_ram_inc) |a| allocator.free(a);
    var p2_rd_inc: ?[]F = null;
    defer if (p2_rd_inc) |a| allocator.free(a);
    var p2_eq_ram: ?[]F = null;
    defer if (p2_eq_ram) |a| allocator.free(a);
    var p2_eq_rd: ?[]F = null;
    defer if (p2_eq_rd) |a| allocator.free(a);
    var p2_len: usize = 0;

    for (0..n_vars) |round| {
        const r = F.fromU64(@intCast(round * 7 + 3));
        challenges[round] = r;

        const flat_half = flat_len / 2;

        // --- Flat round poly (degree 2): 3 evaluation points ---
        // s(t) = Σ_j [ (eq_0(t) + γ·eq_1(t))·ram_inc(t) + γ²·(eq_2(t) + γ·eq_3(t))·rd_inc(t) ]
        var flat_evals: [3]F = .{ F.zero(), F.zero(), F.zero() };
        for (0..flat_half) |j| {
            // Values at t=0, t=1, t=2
            var eq_ram_at: [3]F = undefined;
            var eq_rd_at: [3]F = undefined;
            var ram_at: [3]F = undefined;
            var rd_at: [3]F = undefined;
            for (0..3) |t| {
                const tf = F.fromU64(@intCast(t));
                inline for (0..4) |k| {
                    const v0 = eq_full[k][2 * j];
                    const v1 = eq_full[k][2 * j + 1];
                    const interp = v0.add(tf.mul(v1.sub(v0)));
                    if (k == 0) eq_ram_at[t] = interp;
                    if (k == 1) eq_ram_at[t] = eq_ram_at[t].add(gamma.mul(interp));
                    if (k == 2) eq_rd_at[t] = interp;
                    if (k == 3) eq_rd_at[t] = eq_rd_at[t].add(gamma.mul(interp));
                }
                const r0 = flat_ram_inc[2 * j];
                const r1 = flat_ram_inc[2 * j + 1];
                ram_at[t] = r0.add(tf.mul(r1.sub(r0)));
                const d0 = flat_rd_inc[2 * j];
                const d1 = flat_rd_inc[2 * j + 1];
                rd_at[t] = d0.add(tf.mul(d1.sub(d0)));
            }
            for (0..3) |t| {
                flat_evals[t] = flat_evals[t].add(
                    ram_at[t].mul(eq_ram_at[t]).add(gamma_sqr.mul(rd_at[t].mul(eq_rd_at[t]))),
                );
            }
        }

        // --- Split round poly ---
        var split_evals: [3]F = .{ F.zero(), F.zero(), F.zero() };

        if (!in_phase2) {
            const half = p_len / 2;
            for (0..half) |j| {
                for (0..3) |t| {
                    const tf = F.fromU64(@intCast(t));
                    var term = F.zero();
                    for (0..4) |k| {
                        const p0 = P[k][2 * j];
                        const p1 = P[k][2 * j + 1];
                        const q0 = Q[k][2 * j];
                        const q1 = Q[k][2 * j + 1];
                        const p_t = p0.add(tf.mul(p1.sub(p0)));
                        const q_t = q0.add(tf.mul(q1.sub(q0)));
                        term = term.add(weights[k].mul(p_t.mul(q_t)));
                    }
                    split_evals[t] = split_evals[t].add(term);
                }
            }
        } else {
            const half = p2_len / 2;
            for (0..half) |j| {
                for (0..3) |t| {
                    const tf = F.fromU64(@intCast(t));
                    const ram_t = p2_ram_inc.?[2 * j].add(tf.mul(p2_ram_inc.?[2 * j + 1].sub(p2_ram_inc.?[2 * j])));
                    const eq_r_t = p2_eq_ram.?[2 * j].add(tf.mul(p2_eq_ram.?[2 * j + 1].sub(p2_eq_ram.?[2 * j])));
                    const rd_t = p2_rd_inc.?[2 * j].add(tf.mul(p2_rd_inc.?[2 * j + 1].sub(p2_rd_inc.?[2 * j])));
                    const eq_d_t = p2_eq_rd.?[2 * j].add(tf.mul(p2_eq_rd.?[2 * j + 1].sub(p2_eq_rd.?[2 * j])));
                    split_evals[t] = split_evals[t].add(
                        ram_t.mul(eq_r_t).add(gamma_sqr.mul(rd_t.mul(eq_d_t))),
                    );
                }
            }
        }

        for (0..3) |t| {
            try testing.expect(flat_evals[t].eql(split_evals[t]));
        }

        // --- Bind all arrays ---
        // Flat: bind 4 eq arrays + 2 inc arrays
        for (0..flat_half) |j| {
            for (0..4) |k| {
                eq_full[k][j] = eq_full[k][2 * j].add(r.mul(eq_full[k][2 * j + 1].sub(eq_full[k][2 * j])));
            }
            flat_ram_inc[j] = flat_ram_inc[2 * j].add(r.mul(flat_ram_inc[2 * j + 1].sub(flat_ram_inc[2 * j])));
            flat_rd_inc[j] = flat_rd_inc[2 * j].add(r.mul(flat_rd_inc[2 * j + 1].sub(flat_rd_inc[2 * j])));
        }
        flat_len = flat_half;

        if (!in_phase2) {
            if (p_len == 2) {
                // Transition to Phase 2
                const eq_prefix = try computeEqTable(F, allocator, challenges[0 .. round + 1], prefix_n_vars);
                defer allocator.free(eq_prefix);

                var eq_prefix_scalars: [4]F = undefined;
                for (0..4) |i| {
                    var result = F.one();
                    for (0..prefix_n_vars) |k| {
                        const a = challenges[k];
                        const b = points_le[i][k];
                        const prod = a.mul(b);
                        result = result.mul(prod.add(prod).add(F.one()).sub(a.add(b)));
                    }
                    eq_prefix_scalars[i] = result;
                }

                p2_eq_ram = try allocator.alloc(F, suffix_len);
                p2_eq_rd = try allocator.alloc(F, suffix_len);
                for (0..suffix_len) |x_hi| {
                    p2_eq_ram.?[x_hi] = eq_prefix_scalars[0].mul(eq_hi[0][x_hi]).add(
                        gamma.mul(eq_prefix_scalars[1].mul(eq_hi[1][x_hi])),
                    );
                    p2_eq_rd.?[x_hi] = eq_prefix_scalars[2].mul(eq_hi[2][x_hi]).add(
                        gamma.mul(eq_prefix_scalars[3].mul(eq_hi[3][x_hi])),
                    );
                }

                p2_ram_inc = try allocator.alloc(F, suffix_len);
                p2_rd_inc = try allocator.alloc(F, suffix_len);
                for (0..suffix_len) |x_hi| {
                    var acc_ram = F.zero();
                    var acc_rd = F.zero();
                    for (0..prefix_len) |x_lo| {
                        const x = x_lo + (x_hi << prefix_n_vars);
                        acc_ram = acc_ram.add(eq_prefix[x_lo].mul(ram_inc_vals[x]));
                        acc_rd = acc_rd.add(eq_prefix[x_lo].mul(rd_inc_vals[x]));
                    }
                    p2_ram_inc.?[x_hi] = acc_ram;
                    p2_rd_inc.?[x_hi] = acc_rd;
                }
                p2_len = suffix_len;
                in_phase2 = true;
            } else {
                const half = p_len / 2;
                for (0..4) |k| {
                    for (0..half) |j| {
                        P[k][j] = P[k][2 * j].add(r.mul(P[k][2 * j + 1].sub(P[k][2 * j])));
                        Q[k][j] = Q[k][2 * j].add(r.mul(Q[k][2 * j + 1].sub(Q[k][2 * j])));
                    }
                }
                p_len = half;
            }
        } else {
            const half = p2_len / 2;
            for (0..half) |j| {
                p2_ram_inc.?[j] = p2_ram_inc.?[2 * j].add(r.mul(p2_ram_inc.?[2 * j + 1].sub(p2_ram_inc.?[2 * j])));
                p2_rd_inc.?[j] = p2_rd_inc.?[2 * j].add(r.mul(p2_rd_inc.?[2 * j + 1].sub(p2_rd_inc.?[2 * j])));
                p2_eq_ram.?[j] = p2_eq_ram.?[2 * j].add(r.mul(p2_eq_ram.?[2 * j + 1].sub(p2_eq_ram.?[2 * j])));
                p2_eq_rd.?[j] = p2_eq_rd.?[2 * j].add(r.mul(p2_eq_rd.?[2 * j + 1].sub(p2_eq_rd.?[2 * j])));
            }
            p2_len = half;
        }
    }

    // Final scalar: split must match flat
    const flat_final = flat_ram_inc[0].mul(
        eq_full[0][0].add(gamma.mul(eq_full[1][0])),
    ).add(gamma_sqr.mul(flat_rd_inc[0].mul(
        eq_full[2][0].add(gamma.mul(eq_full[3][0])),
    )));
    const split_final = p2_ram_inc.?[0].mul(p2_eq_ram.?[0]).add(
        gamma_sqr.mul(p2_rd_inc.?[0].mul(p2_eq_rd.?[0])),
    );
    try testing.expect(flat_final.eql(split_final));
}

test "HammingBooleanity full multi-round: split-eq matches flat across phase transition" {
    // Full multi-round sumcheck simulation for HammingBooleanity:
    // Phase 1 (prefix rounds with factored eq_lo·eq_hi) → transition → Phase 2 (merged eq).
    // Verifies every round polynomial matches the flat (unsplit) computation.
    const allocator = testing.allocator;
    const F = BN254Scalar;

    const n_vars = 6;
    const prefix_n_vars = 3;
    const suffix_n_vars = 3;
    const T: usize = 1 << n_vars;
    const prefix_len: usize = 1 << prefix_n_vars;
    const suffix_len: usize = 1 << suffix_n_vars;

    // r_cycle in LE order
    var r_le = [6]F{
        F.fromU64(5), F.fromU64(13), F.fromU64(3),
        F.fromU64(19), F.fromU64(7), F.fromU64(11),
    };

    // H values: simulate Hamming weight (binary values for booleanity test)
    var H_flat: [T]F = undefined;
    var H_split: [T]F = undefined;
    for (0..T) |x| {
        // Mix of 0 and 1 with some non-boolean values to make test interesting
        const v: u64 = if (x % 5 == 0) 0 else if (x % 3 == 0) 1 else @intCast(x % 4);
        H_flat[x] = F.fromU64(v);
        H_split[x] = F.fromU64(v);
    }

    // Flat eq table
    var eq_flat = try computeEqTable(F, allocator, &r_le, n_vars);
    defer allocator.free(eq_flat);

    // Split eq tables
    var r_lo: [3]F = undefined;
    for (0..prefix_n_vars) |k| r_lo[k] = r_le[k];
    var eq_lo = try computeEqTable(F, allocator, &r_lo, prefix_n_vars);
    defer allocator.free(eq_lo);

    var r_hi: [3]F = undefined;
    for (0..suffix_n_vars) |k| r_hi[k] = r_le[prefix_n_vars + k];
    const eq_hi = try computeEqTable(F, allocator, &r_hi, suffix_n_vars);
    defer allocator.free(eq_hi);

    var flat_len: usize = T;
    var split_h_len: usize = T;
    var lo_len: usize = prefix_len;
    var in_phase2 = false;

    // Phase 2 state
    var eq_merged: ?[]F = null;
    defer if (eq_merged) |a| allocator.free(a);
    var merged_len: usize = 0;

    for (0..n_vars) |round| {
        const r = F.fromU64(@intCast(round * 11 + 2));
        const two = F.fromU64(2);
        const three = F.fromU64(3);

        // --- Flat round poly: [s(0), s(1), s(2), s(3)] ---
        const flat_half = flat_len / 2;
        var flat_evals: [4]F = .{ F.zero(), F.zero(), F.zero(), F.zero() };
        for (0..flat_half) |j| {
            const h0 = H_flat[2 * j];
            const h1 = H_flat[2 * j + 1];
            const h_delta = h1.sub(h0);
            const e0 = eq_flat[2 * j];
            const e1 = eq_flat[2 * j + 1];
            const e_delta = e1.sub(e0);

            flat_evals[0] = flat_evals[0].add(e0.mul(h0.mul(h0).sub(h0)));
            flat_evals[1] = flat_evals[1].add(e1.mul(h1.mul(h1).sub(h1)));

            const h_at_2 = h0.add(two.mul(h_delta));
            const e_at_2 = e0.add(two.mul(e_delta));
            flat_evals[2] = flat_evals[2].add(e_at_2.mul(h_at_2.mul(h_at_2).sub(h_at_2)));

            const h_at_3 = h0.add(three.mul(h_delta));
            const e_at_3 = e0.add(three.mul(e_delta));
            flat_evals[3] = flat_evals[3].add(e_at_3.mul(h_at_3.mul(h_at_3).sub(h_at_3)));
        }

        // --- Split round poly ---
        var split_evals: [4]F = .{ F.zero(), F.zero(), F.zero(), F.zero() };

        if (!in_phase2) {
            // Phase 1: double loop with factored eq = eq_lo(x_lo) * eq_hi(x_hi)
            const half_lo = lo_len / 2;
            for (0..suffix_len) |j_outer| {
                const eq_hi_val = eq_hi[j_outer];
                for (0..half_lo) |j_inner| {
                    const j = j_inner + j_outer * half_lo;
                    const h0 = H_split[2 * j];
                    const h1 = H_split[2 * j + 1];
                    const h_delta = h1.sub(h0);

                    const eq_lo_0 = eq_lo[2 * j_inner];
                    const eq_lo_1 = eq_lo[2 * j_inner + 1];
                    const e0 = eq_lo_0.mul(eq_hi_val);
                    const e1 = eq_lo_1.mul(eq_hi_val);
                    const e_delta = e1.sub(e0);

                    split_evals[0] = split_evals[0].add(e0.mul(h0.mul(h0).sub(h0)));
                    split_evals[1] = split_evals[1].add(e1.mul(h1.mul(h1).sub(h1)));

                    const h_at_2 = h0.add(two.mul(h_delta));
                    const e_at_2 = e0.add(two.mul(e_delta));
                    split_evals[2] = split_evals[2].add(e_at_2.mul(h_at_2.mul(h_at_2).sub(h_at_2)));

                    const h_at_3 = h0.add(three.mul(h_delta));
                    const e_at_3 = e0.add(three.mul(e_delta));
                    split_evals[3] = split_evals[3].add(e_at_3.mul(h_at_3.mul(h_at_3).sub(h_at_3)));
                }
            }
        } else {
            // Phase 2: flat loop with merged eq
            const half = split_h_len / 2;
            for (0..half) |j| {
                const h0 = H_split[2 * j];
                const h1 = H_split[2 * j + 1];
                const h_delta = h1.sub(h0);
                const e0 = eq_merged.?[2 * j];
                const e1 = eq_merged.?[2 * j + 1];
                const e_delta = e1.sub(e0);

                split_evals[0] = split_evals[0].add(e0.mul(h0.mul(h0).sub(h0)));
                split_evals[1] = split_evals[1].add(e1.mul(h1.mul(h1).sub(h1)));

                const h_at_2 = h0.add(two.mul(h_delta));
                const e_at_2 = e0.add(two.mul(e_delta));
                split_evals[2] = split_evals[2].add(e_at_2.mul(h_at_2.mul(h_at_2).sub(h_at_2)));

                const h_at_3 = h0.add(three.mul(h_delta));
                const e_at_3 = e0.add(three.mul(e_delta));
                split_evals[3] = split_evals[3].add(e_at_3.mul(h_at_3.mul(h_at_3).sub(h_at_3)));
            }
        }

        // All 4 evaluation points must match
        for (0..4) |k| {
            try testing.expect(flat_evals[k].eql(split_evals[k]));
        }

        // --- Bind ---
        // Flat: bind eq and H
        for (0..flat_half) |j| {
            eq_flat[j] = eq_flat[2 * j].add(r.mul(eq_flat[2 * j + 1].sub(eq_flat[2 * j])));
            H_flat[j] = H_flat[2 * j].add(r.mul(H_flat[2 * j + 1].sub(H_flat[2 * j])));
        }
        flat_len = flat_half;

        // Split: bind H always, plus eq_lo or merged eq
        const split_half = split_h_len / 2;
        for (0..split_half) |j| {
            H_split[j] = H_split[2 * j].add(r.mul(H_split[2 * j + 1].sub(H_split[2 * j])));
        }
        split_h_len = split_half;

        if (!in_phase2) {
            const half_lo = lo_len / 2;
            for (0..half_lo) |j| {
                eq_lo[j] = eq_lo[2 * j].add(r.mul(eq_lo[2 * j + 1].sub(eq_lo[2 * j])));
            }
            lo_len = half_lo;

            // Transition when eq_lo reaches length 1
            if (half_lo == 1) {
                const eq_lo_scalar = eq_lo[0];
                // Merge: eq_merged[j_hi] = eq_lo_scalar * eq_hi[j_hi]
                eq_merged = try allocator.alloc(F, suffix_len);
                for (0..suffix_len) |j| {
                    eq_merged.?[j] = eq_lo_scalar.mul(eq_hi[j]);
                }
                merged_len = suffix_len;
                in_phase2 = true;
            }
        } else {
            // Phase 2: bind merged eq
            const half = merged_len / 2;
            for (0..half) |j| {
                eq_merged.?[j] = eq_merged.?[2 * j].add(r.mul(eq_merged.?[2 * j + 1].sub(eq_merged.?[2 * j])));
            }
            merged_len = half;
        }
    }

    // Final scalars must match
    try testing.expect(H_flat[0].eql(H_split[0]));
    try testing.expect(eq_flat[0].eql(eq_merged.?[0]));
}
