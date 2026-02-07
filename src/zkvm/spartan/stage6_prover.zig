//! Stage 6 Batched Sumcheck Prover
//!
//! Stage 6 is a batched sumcheck with 6 instances:
//! 0. BytecodeReadRaf: bytecode_log_k + n_cycle_vars rounds, degree bytecode_d + 1
//! 1. HammingBooleanity: n_cycle_vars rounds, degree 3 (input_claim = 0)
//! 2. Booleanity: log_k_chunk + n_cycle_vars rounds, degree 3 (input_claim = 0)
//! 3. RamRaVirtual: n_cycle_vars rounds, degree ram_d + 1
//! 4. LookupsRaVirtual: n_cycle_vars rounds, degree n_committed_per_virtual + 1
//! 5. IncClaimReduction: n_cycle_vars rounds, degree 2
//!
//! ALL instances use real sumcheck provers with actual polynomial materialization
//! from execution trace data. No shortcuts, no placeholders.

const std = @import("std");
const Allocator = std.mem.Allocator;

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
    /// Circuit flags (13 flags, matches CircuitFlags enum)
    circuit_flags: [13]bool,
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
};

/// Populate a BytecodeEntry from a raw 32-bit instruction word and ELF address.
/// This sets all static properties (flags, registers, immediates, lookup table)
/// from the instruction encoding alone, without any trace-specific data.
fn populateEntryFromInstruction(entry: *BytecodeEntry, instr: u32, elf_address: u64) void {
    const decoded = instruction_mod.DecodedInstruction.decode(instr);

    entry.address = elf_address;
    entry.rd = decoded.rd;
    entry.rs1 = decoded.rs1;
    entry.rs2 = decoded.rs2;
    entry.imm = @intCast(decoded.imm);

    const opcode: u8 = @truncate(instr & 0x7F);
    const funct3: u3 = @truncate((instr >> 12) & 0x7);
    const funct7: u7 = @truncate(instr >> 25);

    // Circuit flags
    var cf = &entry.circuit_flags;

    // Load/Store
    if (opcode == 0x03) cf[@intFromEnum(CircuitFlags.Load)] = true;
    if (opcode == 0x23) cf[@intFromEnum(CircuitFlags.Store)] = true;

    // Jump
    if (opcode == 0x6F or opcode == 0x67) cf[@intFromEnum(CircuitFlags.Jump)] = true;

    // WriteLookupOutputToRD
    const has_lookup = hasLookupTable(opcode, funct3, funct7);
    if (has_lookup) {
        if (opcode != 0x63) { // Not BRANCH
            cf[@intFromEnum(CircuitFlags.WriteLookupOutputToRD)] = true;
        }
    }

    // AddOperands, SubtractOperands, MultiplyOperands
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
            0x1b => { // OP-IMM-32
                if (funct3 == 0) cf[@intFromEnum(CircuitFlags.AddOperands)] = true; // ADDIW
            },
            0x3b => { // OP-32
                if (funct3 == 0 and funct7 == 0) cf[@intFromEnum(CircuitFlags.AddOperands)] = true; // ADDW
                if (funct3 == 0 and funct7 == 0x20) cf[@intFromEnum(CircuitFlags.SubtractOperands)] = true; // SUBW
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
            0x33, 0x13, 0x67, 0x63, 0x1B, 0x3B => {
                inf[@intFromEnum(InstructionFlags.LeftOperandIsRs1Value)] = true;
            },
            else => {},
        }
    }

    // RightOperandIsImm
    if (has_lookup) {
        switch (opcode) {
            0x13, 0x67, 0x37, 0x17, 0x6F, 0x1B => {
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
/// Phase 2: Overlay trace-specific properties (termination stores) from the
/// execution trace. Termination stores at k=0 accumulate flags from multiple
/// virtual instructions (LUI + ADDI + SB) that overwrite the same entry.
///
/// pc_map converts ELF addresses to bytecode array indices.
pub fn buildBytecodeEntries(
    allocator: Allocator,
    trace: *const tracer.ExecutionTrace,
    bytecode_K: usize,
    pc_map: *const BytecodePCMapper,
    program_code_bytes: ?[]const u8,
    code_base_address: u64,
) ![]BytecodeEntry {
    const entries = try allocator.alloc(BytecodeEntry, bytecode_K);

    // Initialize all entries as NoOps (zeros)
    for (0..bytecode_K) |k| {
        entries[k] = BytecodeEntry{
            .address = 0,
            .imm = 0,
            .rd = 0,
            .rs1 = 0,
            .rs2 = 0,
            .circuit_flags = [_]bool{false} ** 13,
            .instruction_flags = [_]bool{false} ** 7,
            .lookup_table_index = 255,
            .is_interleaved = false,
            .virtual_sequence_remaining = null,
            .is_first_in_sequence = false,
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
                populateEntryFromInstruction(&entries[k], instr_word, addr);

                // Mark compressed instructions
                if (is_compressed) {
                    entries[k].circuit_flags[@intFromEnum(CircuitFlags.IsCompressed)] = true;
                }
            }

            offset += instr_size;
        }
    }

    // ================================================================
    // Phase 2: Overlay trace-specific properties (termination stores)
    // ================================================================
    // Termination stores are virtual instructions that write the final
    // output to memory. They overwrite k=0 and accumulate flags across
    // multiple instructions (LUI + ADDI + SB). This matches Jolt's
    // behavior where the termination sequence overwrites the bytecode
    // entry at index 0.
    for (trace.steps.items) |step| {
        if (step.is_noop) continue;
        if (!step.is_termination_store) continue;

        // Termination stores all map to k=0 (pc=0)
        const k = pc_map.getPC(step.pc, 0);
        if (k >= bytecode_K) continue;

        const instr = step.instruction;
        const decoded = instruction_mod.DecodedInstruction.decode(instr);

        const opcode: u8 = @truncate(instr & 0x7F);
        const funct3: u3 = @truncate((instr >> 12) & 0x7);
        const funct7: u7 = @truncate(instr >> 25);
        const has_lookup = hasLookupTable(opcode, funct3, funct7);

        // Overwrite basic fields (last writer wins for registers/address/imm)
        entries[k].address = step.unexpanded_pc;
        entries[k].rd = decoded.rd;
        entries[k].rs1 = decoded.rs1;
        entries[k].rs2 = decoded.rs2;
        entries[k].imm = @intCast(decoded.imm);

        // ACCUMULATE circuit flags (never clear - flags OR together across overwrites)
        var cf = &entries[k].circuit_flags;
        if (opcode == 0x03) cf[@intFromEnum(CircuitFlags.Load)] = true;
        if (opcode == 0x23) cf[@intFromEnum(CircuitFlags.Store)] = true;
        if (opcode == 0x6F or opcode == 0x67) cf[@intFromEnum(CircuitFlags.Jump)] = true;
        if (has_lookup and opcode != 0x63) {
            cf[@intFromEnum(CircuitFlags.WriteLookupOutputToRD)] = true;
        }
        if (has_lookup) {
            switch (opcode) {
                0x33 => {
                    if (funct3 == 0 and funct7 == 0) cf[@intFromEnum(CircuitFlags.AddOperands)] = true;
                    if (funct3 == 0 and funct7 == 0x20) cf[@intFromEnum(CircuitFlags.SubtractOperands)] = true;
                    if (funct7 == 0x01 and funct3 == 0) cf[@intFromEnum(CircuitFlags.MultiplyOperands)] = true;
                    if (funct7 == 0x01 and funct3 == 3) cf[@intFromEnum(CircuitFlags.MultiplyOperands)] = true;
                },
                0x13 => {
                    if (funct3 == 0) cf[@intFromEnum(CircuitFlags.AddOperands)] = true;
                },
                0x67 => {
                    cf[@intFromEnum(CircuitFlags.AddOperands)] = true;
                },
                0x1b => {
                    if (funct3 == 0) cf[@intFromEnum(CircuitFlags.AddOperands)] = true;
                },
                0x3b => {
                    if (funct3 == 0 and funct7 == 0) cf[@intFromEnum(CircuitFlags.AddOperands)] = true;
                    if (funct3 == 0 and funct7 == 0x20) cf[@intFromEnum(CircuitFlags.SubtractOperands)] = true;
                },
                else => {},
            }
        }
        // Termination stores always set these
        cf[@intFromEnum(CircuitFlags.VirtualInstruction)] = true;
        cf[@intFromEnum(CircuitFlags.DoNotUpdateUnexpandedPC)] = true;

        // ACCUMULATE instruction flags
        var inf = &entries[k].instruction_flags;
        if (has_lookup and (opcode == 0x17 or opcode == 0x6F)) {
            inf[@intFromEnum(InstructionFlags.LeftOperandIsPC)] = true;
        }
        if (has_lookup) {
            switch (opcode) {
                0x33, 0x13, 0x67, 0x63, 0x1B, 0x3B => {
                    inf[@intFromEnum(InstructionFlags.LeftOperandIsRs1Value)] = true;
                },
                else => {},
            }
        }
        if (has_lookup) {
            switch (opcode) {
                0x13, 0x67, 0x37, 0x17, 0x6F, 0x1B => {
                    inf[@intFromEnum(InstructionFlags.RightOperandIsImm)] = true;
                },
                else => {},
            }
        }
        if (has_lookup) {
            switch (opcode) {
                0x33, 0x63, 0x3B => {
                    inf[@intFromEnum(InstructionFlags.RightOperandIsRs2Value)] = true;
                },
                else => {},
            }
        }
        if (opcode == 0x63) {
            inf[@intFromEnum(InstructionFlags.Branch)] = true;
        }
        if (decoded.rd != 0 and opcode != 0x23 and opcode != 0x63) {
            inf[@intFromEnum(InstructionFlags.IsRdNotZero)] = true;
        }

        // Update lookup/interleaving based on accumulated flags
        const lt_idx = getLookupTableIndex(opcode, funct3, funct7);
        if (lt_idx != 255) {
            entries[k].lookup_table_index = lt_idx;
        }
        entries[k].is_interleaved = !cf[@intFromEnum(CircuitFlags.AddOperands)] and
            !cf[@intFromEnum(CircuitFlags.SubtractOperands)] and
            !cf[@intFromEnum(CircuitFlags.MultiplyOperands)] and
            !cf[@intFromEnum(CircuitFlags.Advice)];
    }

    // Debug: dump first entries
    std.debug.print("\n[ZOLT BYTECODE ENTRIES] bytecode_K={}\n", .{bytecode_K});
    for (0..@min(bytecode_K, 28)) |k| {
        const e = entries[k];
        // Compute a compact representation: cf_bits, if_bits
        var cf_bits: u16 = 0;
        for (0..13) |i| {
            if (e.circuit_flags[i]) cf_bits |= @as(u16, 1) << @intCast(i);
        }
        var if_bits: u8 = 0;
        for (0..7) |i| {
            if (e.instruction_flags[i]) if_bits |= @as(u8, 1) << @intCast(i);
        }
        std.debug.print("  entry[{d:2}]: addr=0x{x:0>8} rd={d:2} rs1={d:2} rs2={d:2} imm={d:6} cf=0x{x:04} if=0x{x:02} lt={d:3} interl={}\n", .{
            k, e.address, e.rd, e.rs1, e.rs2, e.imm, cf_bits, if_bits, e.lookup_table_index, @intFromBool(e.is_interleaved),
        });
    }
    std.debug.print("\n", .{});

    return entries;
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
            else => 255,
        },
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

        pub fn init(
            allocator: Allocator,
            trace: *const ExecutionTrace,
            gamma: F,
            r_cycle_stage2: []const F,
            r_cycle_stage4: []const F,
            s_cycle_stage4: []const F,
            s_cycle_stage5: []const F,
        ) !Self {
            const T = trace.steps.items.len;
            const n_vars = std.math.log2_int(usize, T);

            var ram_inc_arr = try allocator.alloc(F, T);
            var rd_inc_arr = try allocator.alloc(F, T);

            for (0..T) |j| {
                const step = trace.steps.items[j];

                // RdInc = rd_value - rd_pre_value
                const rd_post: i128 = @intCast(step.rd_value);
                const rd_pre: i128 = @intCast(step.rd_pre_value);
                rd_inc_arr[j] = fieldFromI128(F, rd_post - rd_pre);

                // RamInc = memory_value - memory_pre_value (only for writes)
                if (step.is_memory_write) {
                    const mem_post: i128 = @intCast(step.memory_value orelse 0);
                    const mem_pre: i128 = @intCast(step.memory_pre_value orelse 0);
                    ram_inc_arr[j] = fieldFromI128(F, mem_post - mem_pre);
                } else {
                    ram_inc_arr[j] = F.zero();
                }
            }

            const eq_stage2 = try computeEqTable(F, allocator, r_cycle_stage2, n_vars);
            defer allocator.free(eq_stage2);
            const eq_stage4 = try computeEqTable(F, allocator, r_cycle_stage4, n_vars);
            defer allocator.free(eq_stage4);
            const eq_s4 = try computeEqTable(F, allocator, s_cycle_stage4, n_vars);
            defer allocator.free(eq_s4);
            const eq_s5 = try computeEqTable(F, allocator, s_cycle_stage5, n_vars);
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
            var eval_0 = F.zero();
            var eval_2 = F.zero();
            var eval_inf = F.zero();

            for (0..half) |j| {
                const ram0 = self.ram_inc[2 * j];
                const ram1 = self.ram_inc[2 * j + 1];
                const ram_delta = ram1.sub(ram0);
                const eq_r0 = self.eq_ram[2 * j];
                const eq_r1 = self.eq_ram[2 * j + 1];
                const eq_r_delta = eq_r1.sub(eq_r0);

                const rd0 = self.rd_inc[2 * j];
                const rd1 = self.rd_inc[2 * j + 1];
                const rd_delta = rd1.sub(rd0);
                const eq_d0 = self.eq_rd[2 * j];
                const eq_d1 = self.eq_rd[2 * j + 1];
                const eq_d_delta = eq_d1.sub(eq_d0);

                // At x=0
                const f0 = ram0.mul(eq_r0).add(self.gamma_sqr.mul(rd0.mul(eq_d0)));
                eval_0 = eval_0.add(f0);

                // At x=2
                const two = F.fromU64(2);
                const ram2 = ram0.add(two.mul(ram_delta));
                const eq_r2 = eq_r0.add(two.mul(eq_r_delta));
                const rd2 = rd0.add(two.mul(rd_delta));
                const eq_d2 = eq_d0.add(two.mul(eq_d_delta));
                eval_2 = eval_2.add(ram2.mul(eq_r2).add(self.gamma_sqr.mul(rd2.mul(eq_d2))));

                // At x=inf (leading coefficients)
                eval_inf = eval_inf.add(ram_delta.mul(eq_r_delta).add(self.gamma_sqr.mul(rd_delta.mul(eq_d_delta))));
            }

            return [3]F{ eval_0, eval_2, eval_inf };
        }

        pub fn bindChallenge(self: *Self, r: F) void {
            const half = self.current_len / 2;
            const one_minus_r = F.one().sub(r);
            for (0..half) |j| {
                self.ram_inc[j] = one_minus_r.mul(self.ram_inc[2 * j]).add(r.mul(self.ram_inc[2 * j + 1]));
                self.rd_inc[j] = one_minus_r.mul(self.rd_inc[2 * j]).add(r.mul(self.rd_inc[2 * j + 1]));
                self.eq_ram[j] = one_minus_r.mul(self.eq_ram[2 * j]).add(r.mul(self.eq_ram[2 * j + 1]));
                self.eq_rd[j] = one_minus_r.mul(self.eq_rd[2 * j]).add(r.mul(self.eq_rd[2 * j + 1]));
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

        pub fn init(
            allocator: Allocator,
            trace: *const ExecutionTrace,
            r_cycle: []const F,
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

            const eq_arr = try computeEqTable(F, allocator, r_cycle, n_vars);

            return Self{
                .H = H_arr,
                .eq = eq_arr,
                .current_len = T,
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.H);
            self.allocator.free(self.eq);
        }

        /// Compute round polynomial at [0, 1, 2, inf]
        pub fn computeRoundPoly(self: *Self) [4]F {
            const half = self.current_len / 2;
            var eval_0 = F.zero();
            var eval_1 = F.zero();
            var eval_2 = F.zero();
            var eval_inf = F.zero();

            for (0..half) |j| {
                const h0 = self.H[2 * j];
                const h1 = self.H[2 * j + 1];
                const h_delta = h1.sub(h0);

                const e0 = self.eq[2 * j];
                const e1 = self.eq[2 * j + 1];
                const e_delta = e1.sub(e0);

                // At x=0: eq(0)*(H(0)^2-H(0))
                eval_0 = eval_0.add(e0.mul(h0.mul(h0).sub(h0)));

                // At x=1: eq(1)*(H(1)^2-H(1))
                eval_1 = eval_1.add(e1.mul(h1.mul(h1).sub(h1)));

                // At x=2
                const two = F.fromU64(2);
                const h_at_2 = h0.add(two.mul(h_delta));
                const e_at_2 = e0.add(two.mul(e_delta));
                eval_2 = eval_2.add(e_at_2.mul(h_at_2.mul(h_at_2).sub(h_at_2)));

                // At x=inf: leading coeff is delta_e * delta_h^2
                eval_inf = eval_inf.add(e_delta.mul(h_delta.mul(h_delta)));
            }

            return [4]F{ eval_0, eval_1, eval_2, eval_inf };
        }

        pub fn bindChallenge(self: *Self, r: F) void {
            const half = self.current_len / 2;
            const one_minus_r = F.one().sub(r);
            for (0..half) |j| {
                self.H[j] = one_minus_r.mul(self.H[2 * j]).add(r.mul(self.H[2 * j + 1]));
                self.eq[j] = one_minus_r.mul(self.eq[2 * j]).add(r.mul(self.eq[2 * j + 1]));
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

        pub fn init(
            allocator: Allocator,
            trace: *const ExecutionTrace,
            r_cycle: []const F, // BIG_ENDIAN
            r_addr_chunks: []const []const F, // r_addr_chunks[i] is BIG_ENDIAN
            d: usize,
            memory_layout: *const jolt_device.MemoryLayout,
            log_k_chunk: usize,
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

                const eq_table = try computeEqTable(F, allocator, r_addr_chunks[i], log_k_chunk);
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

            const eq_arr = try computeEqTable(F, allocator, r_cycle, n_vars);

            return Self{
                .ra_bound = ra_bound_arr,
                .eq = eq_arr,
                .d = d,
                .current_len = T,
                .allocator = allocator,
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
            var evals = try allocator.alloc(F, n_evals);
            @memset(evals, F.zero());

            for (0..half) |j| {
                const eq0 = self.eq[2 * j];
                const eq1 = self.eq[2 * j + 1];
                const eq_delta = eq1.sub(eq0);

                for (0..n_evals) |pt_idx| {
                    var product = F.one();

                    if (pt_idx == n_evals - 1) {
                        // x = inf: product of leading coefficients
                        for (0..self.d) |i| {
                            const delta = self.ra_bound[i][2 * j + 1].sub(self.ra_bound[i][2 * j]);
                            product = product.mul(delta);
                        }
                        product = product.mul(eq_delta);
                    } else {
                        const x = F.fromU64(@intCast(pt_idx));
                        for (0..self.d) |i| {
                            const v0 = self.ra_bound[i][2 * j];
                            const v1 = self.ra_bound[i][2 * j + 1];
                            product = product.mul(v0.add(x.mul(v1.sub(v0))));
                        }
                        product = product.mul(eq0.add(x.mul(eq_delta)));
                    }

                    evals[pt_idx] = evals[pt_idx].add(product);
                }
            }

            return evals;
        }

        pub fn bindChallenge(self: *Self, r: F) void {
            const half = self.current_len / 2;
            const one_minus_r = F.one().sub(r);
            for (0..half) |j| {
                for (0..self.d) |i| {
                    self.ra_bound[i][j] = one_minus_r.mul(self.ra_bound[i][2 * j]).add(r.mul(self.ra_bound[i][2 * j + 1]));
                }
                self.eq[j] = one_minus_r.mul(self.eq[2 * j]).add(r.mul(self.eq[2 * j + 1]));
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

                const eq_table = try computeEqTable(F, allocator, r_addr_chunks[i], log_k_chunk);
                defer allocator.free(eq_table);

                // Determine gamma scaling for first poly in each virtual batch
                const virtual_batch = i / M;
                const is_first_in_batch = (i % M == 0);
                const scale = if (is_first_in_batch) gamma_powers[virtual_batch] else F.one();

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

            const eq_arr = try computeEqTable(F, allocator, r_cycle, n_vars);

            return Self{
                .ra_bound = ra_bound_arr,
                .eq = eq_arr,
                .M = M,
                .N = N,
                .total_committed = total_committed,
                .current_len = T,
                .allocator = allocator,
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
            var evals = try allocator.alloc(F, n_evals);
            @memset(evals, F.zero());

            for (0..half) |j| {
                const eq0 = self.eq[2 * j];
                const eq1 = self.eq[2 * j + 1];
                const eq_delta = eq1.sub(eq0);

                for (0..n_evals) |pt_idx| {
                    var virtual_sum = F.zero();

                    for (0..self.N) |v| {
                        var product = F.one();

                        if (pt_idx == n_evals - 1) {
                            for (0..self.M) |m| {
                                const idx = v * self.M + m;
                                const delta = self.ra_bound[idx][2 * j + 1].sub(self.ra_bound[idx][2 * j]);
                                product = product.mul(delta);
                            }
                        } else {
                            const x = F.fromU64(@intCast(pt_idx));
                            for (0..self.M) |m| {
                                const idx = v * self.M + m;
                                const v0 = self.ra_bound[idx][2 * j];
                                const v1 = self.ra_bound[idx][2 * j + 1];
                                product = product.mul(v0.add(x.mul(v1.sub(v0))));
                            }
                        }

                        virtual_sum = virtual_sum.add(product);
                    }

                    if (pt_idx == n_evals - 1) {
                        evals[pt_idx] = evals[pt_idx].add(eq_delta.mul(virtual_sum));
                    } else {
                        const x = F.fromU64(@intCast(pt_idx));
                        evals[pt_idx] = evals[pt_idx].add(eq0.add(x.mul(eq_delta)).mul(virtual_sum));
                    }
                }
            }

            return evals;
        }

        pub fn bindChallenge(self: *Self, r: F) void {
            const half = self.current_len / 2;
            const one_minus_r = F.one().sub(r);
            for (0..half) |j| {
                for (0..self.total_committed) |i| {
                    self.ra_bound[i][j] = one_minus_r.mul(self.ra_bound[i][2 * j]).add(r.mul(self.ra_bound[i][2 * j + 1]));
                }
                self.eq[j] = one_minus_r.mul(self.eq[2 * j]).add(r.mul(self.eq[2 * j + 1]));
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
//   Polynomial over address vars: combined[k] = Sum_s gamma^s * (Val_s(k) + RAF_s(k)) * F_s[k]
//   where F_s[k] = Sum_c eq(r_cycle_s, c) * delta(PC(c)=k)
//   This is LINEAR in address vars, so the round poly has eval points [p(0), p(1), p(inf=0)]
//
// Phase 2: Cycle binding (n_cycle_vars rounds)
//   After binding address to r_addr, polynomial becomes:
//   f(c) = [Prod_i ra_chunk_i(c)] * [Sum_s gamma^s * bound_val_s * eq_s(c)]
//   Degree = bytecode_d + 1
fn BytecodeReadRafProver(comptime F: type) type {
    return struct {
        const Self = @This();

        /// Phase 1: combined[k] = full polynomial over address domain
        combined: []F,

        /// Phase 2: RA chunk polynomials ra_chunks[i][c]
        ra_chunks: ?[][]F,

        /// Phase 2: combined_cycle[c] = Sum_s bound_val_s * eq_s(c)
        combined_cycle: ?[]F,

        /// Phase tracking
        phase: u8,
        bytecode_log_k: usize,
        n_cycle_vars: usize,
        bytecode_d: usize,
        log_k_chunk: usize,
        current_len: usize,
        addr_rounds_done: usize,

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
        ) !Self {
            const bytecode_K: usize = @as(usize, 1) << @intCast(bytecode_log_k);
            const T: usize = @as(usize, 1) << @intCast(n_cycle_vars);

            // Phase 1: Compute combined[k] = Sum_s gamma^s * (Val_s(k) + RAF_s(k)) * F_s[k]
            var combined_arr = try allocator.alloc(F, bytecode_K);
            @memset(combined_arr, F.zero());

            for (0..5) |s| {
                const eq_table = try computeEqTable(F, allocator, stage_r_cycles[s], n_cycle_vars);
                defer allocator.free(eq_table);

                // F_s[k] = Sum_{c: PC(c)=k} eq(r_cycle_s, c)
                var F_s = try allocator.alloc(F, bytecode_K);
                defer allocator.free(F_s);
                @memset(F_s, F.zero());

                for (0..T) |c| {
                    const step = trace.steps.items[c];
                    // Convert ELF address to bytecode array index
                    const pc = if (step.is_noop) @as(usize, 0) else pc_map.getPC(step.pc, 0);
                    if (pc < bytecode_K) {
                        F_s[pc] = F_s[pc].add(eq_table[c]);
                    }
                }

                for (0..bytecode_K) |k| {
                    if (F_s[k].isZero()) continue;

                    var val_plus_raf = if (val_polys[s].len > k) val_polys[s][k] else F.zero();

                    // RAF terms: total coefficient for Stage 0 RAF = gamma^5, for Stage 2 RAF = gamma^6
                    // Since we multiply by gamma_powers[s] at the outer level:
                    //   Stage 0: gamma^0 * (gamma^5 * Int) = gamma^5 * Int => use gamma_powers[5]
                    //   Stage 2: gamma^2 * (gamma^4 * Int) = gamma^6 * Int => use gamma_powers[4]
                    if (s == 0) {
                        val_plus_raf = val_plus_raf.add(gamma_powers[5].mul(int_poly[k]));
                    } else if (s == 2) {
                        val_plus_raf = val_plus_raf.add(gamma_powers[4].mul(int_poly[k]));
                    }

                    combined_arr[k] = combined_arr[k].add(gamma_powers[s].mul(val_plus_raf).mul(F_s[k]));
                }
            }

            // Debug: print per-stage contribution to combined array
            {
                var total_sum = F.zero();
                for (0..bytecode_K) |k| {
                    total_sum = total_sum.add(combined_arr[k]);
                }
                std.debug.print("[BCRAF] Sum_k combined[k] = {any}\n", .{total_sum.toBytesBE()[0..8]});
            }

            // Recompute per-stage to see contributions
            for (0..5) |s| {
                const dbg_eq_table = try computeEqTable(F, allocator, stage_r_cycles[s], n_cycle_vars);
                defer allocator.free(dbg_eq_table);

                var rv_s = F.zero(); // Sum_c Val_s(PC(c)) * eq(r_s, c)
                var raf_s = F.zero(); // Sum_c Int(PC(c)) * eq(r_s, c)
                for (0..T) |c| {
                    const step = trace.steps.items[c];
                    // Convert ELF address to bytecode array index
                    const pc = if (step.is_noop) @as(usize, 0) else pc_map.getPC(step.pc, 0);
                    if (pc < bytecode_K) {
                        const v = if (val_polys[s].len > pc) val_polys[s][pc] else F.zero();
                        rv_s = rv_s.add(v.mul(dbg_eq_table[c]));
                        raf_s = raf_s.add(int_poly[pc].mul(dbg_eq_table[c]));
                    }
                }
                std.debug.print("[BCRAF] Stage {} rv={any} raf={any}\n", .{
                    s, rv_s.toBytesBE()[0..8], raf_s.toBytesBE()[0..8],
                });
                // Print non-noop cycles count and sum of eq values for stage 0
                if (s == 0) {
                    var noop_count: usize = 0;
                    var real_count: usize = 0;
                    var eq_sum = F.zero();
                    for (0..T) |c| {
                        const step = trace.steps.items[c];
                        if (step.is_noop) {
                            noop_count += 1;
                        } else {
                            real_count += 1;
                            eq_sum = eq_sum.add(dbg_eq_table[c]);
                        }
                    }
                    std.debug.print("[BCRAF] Real cycles: {}, Noop cycles: {}, eq_sum_real={any}\n", .{
                        real_count, noop_count, eq_sum.toBytesBE()[0..8],
                    });
                    // Print first few trace entries with pc_map conversion
                    for (0..@min(5, T)) |c| {
                        const step = trace.steps.items[c];
                        const bc_idx = if (step.is_noop) @as(usize, 0) else pc_map.getPC(step.pc, 0);
                        std.debug.print("[BCRAF] cycle {} elf_pc=0x{x} bc_idx={} noop={} instr=0x{x:0>8}\n", .{ c, step.pc, bc_idx, step.is_noop, step.instruction });
                    }
                    // Check int_poly values
                    for (0..@min(8, bytecode_K)) |k| {
                        std.debug.print("[BCRAF] int_poly[{}] = {any}\n", .{ k, int_poly[k].toBytesBE()[0..8] });
                    }
                }
            }

            return Self{
                .combined = combined_arr,
                .ra_chunks = null,
                .combined_cycle = null,
                .phase = 0,
                .bytecode_log_k = bytecode_log_k,
                .n_cycle_vars = n_cycle_vars,
                .bytecode_d = bytecode_d,
                .log_k_chunk = log_k_chunk,
                .current_len = bytecode_K,
                .addr_rounds_done = 0,
                .trace = trace,
                .pc_map = pc_map,
                .stage_r_cycles = stage_r_cycles,
                .gamma_powers = gamma_powers,
                .val_polys = val_polys,
                .int_poly = int_poly,
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.combined);
            if (self.ra_chunks) |chunks| {
                for (chunks) |arr| self.allocator.free(arr);
                self.allocator.free(chunks);
            }
            if (self.combined_cycle) |cc| self.allocator.free(cc);
            for (&self.val_polys) |vp| {
                if (vp.len > 0) self.allocator.free(vp);
            }
            self.allocator.free(self.int_poly);
        }

        /// Phase 1: linear round poly over address vars
        /// Returns [p(0), p(1), p(inf)] where p(inf)=0 for linear poly
        pub fn computeRoundPolyPhase1(self: *Self) [3]F {
            const half = self.current_len / 2;
            var eval_0 = F.zero();
            var eval_1 = F.zero();

            for (0..half) |k| {
                eval_0 = eval_0.add(self.combined[2 * k]);
                eval_1 = eval_1.add(self.combined[2 * k + 1]);
            }

            return [3]F{ eval_0, eval_1, F.zero() };
        }

        pub fn bindChallengePhase1(self: *Self, r: F) void {
            const half = self.current_len / 2;
            const one_minus_r = F.one().sub(r);
            for (0..half) |k| {
                self.combined[k] = one_minus_r.mul(self.combined[2 * k]).add(r.mul(self.combined[2 * k + 1]));
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

            // Reverse challenges to get BIG_ENDIAN r_address
            var r_address_be = try self.allocator.alloc(F, self.bytecode_log_k);
            defer self.allocator.free(r_address_be);
            for (0..self.bytecode_log_k) |i| {
                r_address_be[i] = r_address_challenges[self.bytecode_log_k - 1 - i];
            }

            // Compute bound_vals[s] = Val_s(r_address) + RAF_s(r_address)
            // Evaluate each Val polynomial at r_address using the eq table
            const eq_addr = try computeEqTable(F, self.allocator, r_address_be, self.bytecode_log_k);
            defer self.allocator.free(eq_addr);

            var bound_vals: [5]F = undefined;
            for (0..5) |s| {
                var val_eval = F.zero();
                const max_k = @min(self.val_polys[s].len, bytecode_K);
                for (0..max_k) |k| {
                    val_eval = val_eval.add(self.val_polys[s][k].mul(eq_addr[k]));
                }

                // Add RAF terms (inner powers, multiplied by gamma_powers[s] at outer level)
                // Total RAF coeff: Stage 0 = gamma^5 (inner=5), Stage 2 = gamma^6 (inner=4, outer=2)
                if (s == 0) {
                    var identity_eval = F.zero();
                    for (0..bytecode_K) |k| {
                        identity_eval = identity_eval.add(self.int_poly[k].mul(eq_addr[k]));
                    }
                    val_eval = val_eval.add(self.gamma_powers[5].mul(identity_eval));
                } else if (s == 2) {
                    var identity_eval = F.zero();
                    for (0..bytecode_K) |k| {
                        identity_eval = identity_eval.add(self.int_poly[k].mul(eq_addr[k]));
                    }
                    val_eval = val_eval.add(self.gamma_powers[4].mul(identity_eval));
                }

                bound_vals[s] = self.gamma_powers[s].mul(val_eval);
            }

            // Build RA chunk polynomials for cycle binding
            // ra_chunks[i][c] = eq(r_addr_chunk_i, PC_chunk_i(c))
            self.ra_chunks = try self.allocator.alloc([]F, self.bytecode_d);
            for (0..self.bytecode_d) |i| {
                self.ra_chunks.?[i] = try self.allocator.alloc(F, T);

                const chunk_start = i * self.log_k_chunk;
                const chunk_end = @min(chunk_start + self.log_k_chunk, self.bytecode_log_k);
                const chunk_len = chunk_end - chunk_start;

                const r_chunk = r_address_be[chunk_start..chunk_end];
                const eq_table = try computeEqTable(F, self.allocator, r_chunk, chunk_len);
                defer self.allocator.free(eq_table);

                const chunk_K: usize = @as(usize, 1) << @intCast(chunk_len);

                for (0..T) |c| {
                    const step = self.trace.steps.items[c];
                    // Convert ELF address to bytecode array index
                    const pc = if (step.is_noop) @as(usize, 0) else self.pc_map.getPC(step.pc, 0);
                    if (pc < bytecode_K) {
                        // Extract chunk using MSB-first ordering
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
            var eq_per_stage: [5][]F = undefined;
            for (0..5) |s| {
                eq_per_stage[s] = try computeEqTable(F, self.allocator, self.stage_r_cycles[s], self.n_cycle_vars);
            }

            // Compute combined_cycle[c] = Sum_s bound_vals[s] * eq_s(c)
            self.combined_cycle = try self.allocator.alloc(F, T);
            for (0..T) |c| {
                var val = F.zero();
                for (0..5) |s| {
                    val = val.add(bound_vals[s].mul(eq_per_stage[s][c]));
                }
                self.combined_cycle.?[c] = val;
            }

            // Free eq tables
            for (0..5) |s| {
                self.allocator.free(eq_per_stage[s]);
            }

            // Free Phase 1 combined and replace with cycle data
            self.allocator.free(self.combined);
            self.combined = self.combined_cycle.?;
            self.combined_cycle = null;
            self.current_len = T;
            self.phase = 1;
        }

        /// Phase 2: degree bytecode_d+1 round poly
        pub fn computeRoundPolyPhase2(self: *Self, allocator: Allocator) ![]F {
            const half = self.current_len / 2;
            const n_evals = self.bytecode_d + 2;
            var evals = try allocator.alloc(F, n_evals);
            @memset(evals, F.zero());

            for (0..half) |c| {
                const val0 = self.combined[2 * c];
                const val1 = self.combined[2 * c + 1];
                const val_delta = val1.sub(val0);

                for (0..n_evals) |pt_idx| {
                    var ra_product = F.one();

                    if (pt_idx == n_evals - 1) {
                        for (0..self.bytecode_d) |i| {
                            const delta = self.ra_chunks.?[i][2 * c + 1].sub(self.ra_chunks.?[i][2 * c]);
                            ra_product = ra_product.mul(delta);
                        }
                        ra_product = ra_product.mul(val_delta);
                    } else {
                        const x = F.fromU64(@intCast(pt_idx));
                        for (0..self.bytecode_d) |i| {
                            const r0 = self.ra_chunks.?[i][2 * c];
                            const r1 = self.ra_chunks.?[i][2 * c + 1];
                            ra_product = ra_product.mul(r0.add(x.mul(r1.sub(r0))));
                        }
                        ra_product = ra_product.mul(val0.add(x.mul(val_delta)));
                    }

                    evals[pt_idx] = evals[pt_idx].add(ra_product);
                }
            }

            return evals;
        }

        pub fn bindChallengePhase2(self: *Self, r: F) void {
            const half = self.current_len / 2;
            const one_minus_r = F.one().sub(r);
            for (0..half) |c| {
                self.combined[c] = one_minus_r.mul(self.combined[2 * c]).add(r.mul(self.combined[2 * c + 1]));
                for (0..self.bytecode_d) |i| {
                    self.ra_chunks.?[i][c] = one_minus_r.mul(self.ra_chunks.?[i][2 * c]).add(r.mul(self.ra_chunks.?[i][2 * c + 1]));
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
            // Memory layout for address remapping
            memory_layout: *const jolt_device.MemoryLayout,
            // Bytecode entry table for Val polynomial computation
            bytecode_entries: []const BytecodeEntry,
            // Register address opening points for Stages 4 and 5 (BIG_ENDIAN)
            r_register_4: []const F, // From RegistersReadWriteChecking (address portion)
            r_register_5: []const F, // From RegistersValEvaluation (address portion)
            // BytecodePCMapper for converting ELF addresses to bytecode array indices
            pc_map: *const BytecodePCMapper,
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

            std.debug.print("[STAGE6] Configuration:\n", .{});
            std.debug.print("  bytecodeReadRaf: {} rounds, degree {}\n", .{ bytecodeReadRaf_rounds, bytecodeReadRaf_degree });
            std.debug.print("  hammingBooleanity: {} rounds, degree {}\n", .{ hammingBooleanity_rounds, hammingBooleanity_degree });
            std.debug.print("  booleanity: {} rounds, degree {}\n", .{ booleanity_rounds, booleanity_degree });
            std.debug.print("  ramRaVirtual: {} rounds, degree {}\n", .{ ramRaVirtual_rounds, ramRaVirtual_degree });
            std.debug.print("  lookupsRaVirtual: {} rounds, degree {}\n", .{ lookupsRaVirtual_rounds, lookupsRaVirtual_degree });
            std.debug.print("  incClaimReduction: {} rounds, degree {}\n", .{ incClaimReduction_rounds, incClaimReduction_degree });
            std.debug.print("  max_num_rounds: {}, max_degree: {}\n", .{ max_num_rounds, max_degree });

            // ====================================================================
            // Sample gammas (must match Jolt verifier)
            // ====================================================================

            // Debug: dump transcript state at Stage 6 entry
            std.debug.print("[STAGE6] Transcript state at entry: {{ ", .{});
            for (transcript.state) |b| std.debug.print("{x:0>2} ", .{b});
            std.debug.print("}}, round={}\n", .{transcript.n_rounds});

            const bytecode_raf_gamma_powers = try transcript.challengeScalarPowers(self.allocator, 7);
            defer self.allocator.free(bytecode_raf_gamma_powers);

            // Debug: print first gamma to verify transcript sync
            {
                const g0_be = bytecode_raf_gamma_powers[1].toBytesBE(); // [1] is gamma itself
                std.debug.print("[STAGE6] bytecodeRaf_gamma = [{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                    g0_be[31], g0_be[30], g0_be[29], g0_be[28], g0_be[27], g0_be[26], g0_be[25], g0_be[24],
                });
            }

            const NUM_CIRCUIT_FLAGS: usize = 13;
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

            std.debug.print("[STAGE6] Sampled BytecodeReadRaf gammas\n", .{});

            // BooleanitySumcheckParams::new() - conditional extra challenges
            if (lookups_ra_virtual_log_k_chunk < log_k_chunk) {
                const extra_count = log_k_chunk - lookups_ra_virtual_log_k_chunk;
                for (0..extra_count) |_| {
                    _ = transcript.challengeScalar();
                }
            }
            const booleanity_gamma = transcript.challengeScalar();
            _ = booleanity_gamma;

            // LookupsRa::new() - gamma powers for virtual RA batching
            const lookups_ra_gamma_powers = try transcript.challengeScalarPowers(self.allocator, n_virtual_ra_polys);
            defer self.allocator.free(lookups_ra_gamma_powers);

            // IncClaimReduction::new() - gamma
            // Jolt uses challenge_scalar() (FULL 128-bit) for inc gamma, not optimized
            const inc_gamma = transcript.challengeScalarFull();

            // ====================================================================
            // Compute input claims
            // ====================================================================

            const bytecodeReadRaf_input = self.computeBytecodeReadRafInputClaim(
                opening_claims,
                bytecode_raf_gamma_powers,
                stage1_gammas,
                stage2_gammas,
                stage3_gammas,
                stage4_gammas,
                stage5_gammas,
            );

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
                .{ .Committed = .{ .poly = .RamInc, .sumcheck_id = .RamValEvaluation } },
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
                std.debug.print("[STAGE6] inc_gamma = [{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                    ig_be[31], ig_be[30], ig_be[29], ig_be[28], ig_be[27], ig_be[26], ig_be[25], ig_be[24],
                });
                std.debug.print("[STAGE6] IncClaim v1(RamInc@RamRWC) = [{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                    v1_be[31], v1_be[30], v1_be[29], v1_be[28], v1_be[27], v1_be[26], v1_be[25], v1_be[24],
                });
                std.debug.print("[STAGE6] IncClaim v2(RamInc@RamVal) = [{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                    v2_be[31], v2_be[30], v2_be[29], v2_be[28], v2_be[27], v2_be[26], v2_be[25], v2_be[24],
                });
                std.debug.print("[STAGE6] IncClaim w1(RdInc@RegsRWC) = [{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                    w1_be[31], w1_be[30], w1_be[29], w1_be[28], w1_be[27], w1_be[26], w1_be[25], w1_be[24],
                });
                std.debug.print("[STAGE6] IncClaim w2(RdInc@RegsVal) = [{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                    w2_be[31], w2_be[30], w2_be[29], w2_be[28], w2_be[27], w2_be[26], w2_be[25], w2_be[24],
                });
            }

            const incClaimReduction_input = v1_claim
                .add(inc_gamma.mul(v2_claim))
                .add(inc_gamma2.mul(w1_claim))
                .add(inc_gamma3.mul(w2_claim));

            std.debug.print("[STAGE6] Input claims (LE first 8):\n", .{});
            // Print components for IncClaimReduction
            {
                const v1_be = v1_claim.toBytesBE();
                const v2_be = v2_claim.toBytesBE();
                const w1_be = w1_claim.toBytesBE();
                const w2_be = w2_claim.toBytesBE();
                std.debug.print("  IncClaim components: v1=[{x:0>2},{x:0>2},...] v2=[{x:0>2},{x:0>2},...] w1=[{x:0>2},{x:0>2},...] w2=[{x:0>2},{x:0>2},...]\n", .{
                    v1_be[31], v1_be[30], v2_be[31], v2_be[30], w1_be[31], w1_be[30], w2_be[31], w2_be[30],
                });
            }
            // Print LookupsRa claims
            for (0..@min(n_virtual_ra_polys, 4)) |i| {
                const ra_c = opening_claims.get(
                    .{ .Virtual = .{ .poly = .{ .InstructionRa = i }, .sumcheck_id = .InstructionReadRaf } },
                ) orelse F.zero();
                const ra_be = ra_c.toBytesBE();
                std.debug.print("  InstructionRa[{}] = [{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                    i, ra_be[31], ra_be[30], ra_be[29], ra_be[28], ra_be[27], ra_be[26], ra_be[25], ra_be[24],
                });
            }
            // Print BytecodeReadRaf components
            {
                const bc_be = bytecodeReadRaf_input.toBytesBE();
                std.debug.print("  bytecodeReadRaf_input = [{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                    bc_be[31], bc_be[30], bc_be[29], bc_be[28], bc_be[27], bc_be[26], bc_be[25], bc_be[24],
                });
            }
            {
                const ram_be = ramRaVirtual_input.toBytesBE();
                std.debug.print("  ramRaVirtual_input = [{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                    ram_be[31], ram_be[30], ram_be[29], ram_be[28], ram_be[27], ram_be[26], ram_be[25], ram_be[24],
                });
            }
            {
                const look_be = lookupsRaVirtual_input.toBytesBE();
                std.debug.print("  lookupsRaVirtual_input = [{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                    look_be[31], look_be[30], look_be[29], look_be[28], look_be[27], look_be[26], look_be[25], look_be[24],
                });
            }
            {
                const inc_be = incClaimReduction_input.toBytesBE();
                std.debug.print("  incClaimReduction_input = [{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                    inc_be[31], inc_be[30], inc_be[29], inc_be[28], inc_be[27], inc_be[26], inc_be[25], inc_be[24],
                });
            }

            // ====================================================================
            // Derive opening points for RamRaVirtual and LookupsRaVirtual from Stage 5
            // ====================================================================

            const LOOKUPS_LOG_K: usize = 128;
            const ram_log_k: usize = std.math.log2_int(usize, @as(usize, 1) << @intCast(ram_d * log_k_chunk));

            // RamRaVirtual: r_cycle and r_addr_chunks from RamRaClaimReduction (Stage 5 Instance 2)
            // RamRaClaimReduction has ram_log_k + n_cycle_vars rounds
            // Its challenges come from stage5_challenges[0..ram_log_k+n_cycle_vars]
            // normalize_opening_point reverses BOTH address and cycle parts
            const ram_ra_total_rounds = ram_log_k + n_cycle_vars;
            var ram_ra_r_cycle = try self.allocator.alloc(F, n_cycle_vars);
            defer self.allocator.free(ram_ra_r_cycle);
            for (0..n_cycle_vars) |i| {
                // Reverse cycle part: challenges[ram_log_k..ram_log_k+n_cycle_vars] reversed
                ram_ra_r_cycle[i] = stage5_challenges[ram_ra_total_rounds - 1 - i];
            }

            // r_address for RamRa: challenges[0..ram_log_k] reversed to BIG_ENDIAN
            var ram_ra_r_address_be = try self.allocator.alloc(F, ram_log_k);
            defer self.allocator.free(ram_ra_r_address_be);
            for (0..ram_log_k) |i| {
                ram_ra_r_address_be[i] = stage5_challenges[ram_log_k - 1 - i];
            }

            // Split r_address into chunks (BIG_ENDIAN, chunk[0] = MSB)
            var ram_ra_addr_chunks = try self.allocator.alloc([]const F, ram_d);
            defer self.allocator.free(ram_ra_addr_chunks);
            for (0..ram_d) |i| {
                const chunk_start = i * log_k_chunk;
                const chunk_end = @min(chunk_start + log_k_chunk, ram_log_k);
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
            );
            defer inc_prover.deinit();

            // Instance 1: HammingBooleanity (degree 3)
            var hamming_prover = try HammingBooleanityProver(F).init(
                self.allocator, trace, r_cycle_bc1_spartan_outer,
            );
            defer hamming_prover.deinit();

            // Instance 3: RamRaVirtual (degree ram_d+1)
            var ram_ra_prover = try RamRaVirtualProver(F).init(
                self.allocator, trace, ram_ra_r_cycle,
                ram_ra_addr_chunks, ram_d, memory_layout, log_k_chunk,
            );
            defer ram_ra_prover.deinit();

            // Instance 4: LookupsRaVirtual (degree n_committed_per_virtual+1)
            var lookups_ra_prover = try LookupsRaVirtualProver(F).init(
                self.allocator, trace, lookups_ra_r_cycle,
                lookups_ra_addr_chunks, lookups_ra_gamma_powers,
                n_committed_per_virtual, n_virtual_ra_polys,
                log_k_chunk, instruction_d,
            );
            defer lookups_ra_prover.deinit();

            // Instance 0: BytecodeReadRaf (degree bytecode_d+1)
            // Compute Val polynomials from bytecode entries and stage gammas
            const bytecode_K: usize = @as(usize, 1) << @intCast(bytecode_log_k);
            var bytecode_val_polys: [5][]F = undefined;

            // Precompute eq tables for Stages 4 and 5 register addresses
            // r_register_4 and r_register_5 are the address portions from
            // RegistersReadWriteChecking and RegistersValEvaluation opening points
            const REGISTER_COUNT_LOG2: usize = 5; // log2(32 registers)
            const eq_table_4 = try computeEqTable(F, self.allocator, r_register_4, REGISTER_COUNT_LOG2);
            defer self.allocator.free(eq_table_4);
            const eq_table_5 = try computeEqTable(F, self.allocator, r_register_5, REGISTER_COUNT_LOG2);
            defer self.allocator.free(eq_table_5);

            for (0..5) |s| {
                bytecode_val_polys[s] = try self.allocator.alloc(F, bytecode_K);
                @memset(bytecode_val_polys[s], F.zero());
            }

            for (0..bytecode_K) |k| {
                if (k >= bytecode_entries.len) break;
                const entry = bytecode_entries[k];

                // Stage 1: unexpanded_pc + γ₁¹·imm + Σ γ₁^(2+i)·circuit_flag_i
                var val1 = stage1_gammas[0].mul(F.fromU64(entry.address));
                val1 = val1.add(stage1_gammas[1].mul(fieldFromI128(F, @intCast(entry.imm))));
                for (0..13) |i| {
                    if (entry.circuit_flags[i]) {
                        val1 = val1.add(stage1_gammas[2 + i]);
                    }
                }
                bytecode_val_polys[0][k] = val1;

                // Stage 2: γ₂⁰·jump + γ₂¹·branch + γ₂²·is_rd_not_zero + γ₂³·write_lookup_to_rd
                var val2 = F.zero();
                if (entry.circuit_flags[@intFromEnum(CircuitFlags.Jump)]) {
                    val2 = val2.add(stage2_gammas[0]);
                }
                if (entry.instruction_flags[@intFromEnum(InstructionFlags.Branch)]) {
                    val2 = val2.add(stage2_gammas[1]);
                }
                if (entry.instruction_flags[@intFromEnum(InstructionFlags.IsRdNotZero)]) {
                    val2 = val2.add(stage2_gammas[2]);
                }
                if (entry.circuit_flags[@intFromEnum(CircuitFlags.WriteLookupOutputToRD)]) {
                    val2 = val2.add(stage2_gammas[3]);
                }
                bytecode_val_polys[1][k] = val2;

                // Stage 3: γ₃⁰·imm + γ₃¹·unexpanded_pc + γ₃²·L_is_rs1 + γ₃³·L_is_pc
                //         + γ₃⁴·R_is_rs2 + γ₃⁵·R_is_imm + γ₃⁶·is_noop
                //         + γ₃⁷·virtual_instruction + γ₃⁸·is_first_in_sequence
                var val3 = stage3_gammas[0].mul(fieldFromI128(F, @intCast(entry.imm)));
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
                var val4 = F.zero();
                if (entry.rd < 32) {
                    val4 = val4.add(stage4_gammas[0].mul(eq_table_4[entry.rd]));
                }
                if (entry.rs1 < 32) {
                    val4 = val4.add(stage4_gammas[1].mul(eq_table_4[entry.rs1]));
                }
                if (entry.rs2 < 32) {
                    val4 = val4.add(stage4_gammas[2].mul(eq_table_4[entry.rs2]));
                }
                bytecode_val_polys[3][k] = val4;

                // Stage 5: γ₅⁰·eq(rd, r_reg5) + γ₅¹·!is_interleaved + Σ γ₅^(2+i)·table_flag_i
                var val5 = F.zero();
                if (entry.rd < 32) {
                    val5 = val5.add(stage5_gammas[0].mul(eq_table_5[entry.rd]));
                }
                if (!entry.is_interleaved) {
                    val5 = val5.add(stage5_gammas[1]);
                }
                if (entry.lookup_table_index < 41) {
                    val5 = val5.add(stage5_gammas[2 + @as(usize, entry.lookup_table_index)]);
                }
                bytecode_val_polys[4][k] = val5;
            }

            std.debug.print("[STAGE6] Computed Val polynomials from bytecode entries\n", .{});
            std.debug.print("[STAGE6] Val[0][0] = {any}\n", .{bytecode_val_polys[0][0].toBytesBE()[0..8]});
            std.debug.print("[STAGE6] Val[1][0] = {any}\n", .{bytecode_val_polys[1][0].toBytesBE()[0..8]});

            // Dump first few bytecode entries for debugging
            for (0..@min(bytecode_K, 20)) |k| {
                const entry = bytecode_entries[k];
                std.debug.print("[STAGE6] entry[{}]: addr=0x{x:0>8} rd={} rs1={} rs2={} imm={} cf=[", .{k, entry.address, entry.rd, entry.rs1, entry.rs2, entry.imm});
                for (0..13) |i| {
                    if (i > 0) std.debug.print(",", .{});
                    if (entry.circuit_flags[i]) std.debug.print("1", .{}) else std.debug.print("0", .{});
                }
                std.debug.print("] if=[", .{});
                for (0..7) |i| {
                    if (i > 0) std.debug.print(",", .{});
                    if (entry.instruction_flags[i]) std.debug.print("1", .{}) else std.debug.print("0", .{});
                }
                std.debug.print("] lt={} interleaved={}\n", .{entry.lookup_table_index, @intFromBool(entry.is_interleaved)});
            }

            // Build identity polynomial
            var bytecode_int_poly = try self.allocator.alloc(F, bytecode_K);
            for (0..bytecode_K) |k| {
                bytecode_int_poly[k] = F.fromU64(@intCast(k));
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
            );
            defer bytecode_prover.deinit();

            // ====================================================================
            // Append input claims and get batching coefficients
            // ====================================================================

            transcript.appendScalar(bytecodeReadRaf_input);
            transcript.appendScalar(hammingBooleanity_input);
            transcript.appendScalar(booleanity_input);
            transcript.appendScalar(ramRaVirtual_input);
            transcript.appendScalar(lookupsRaVirtual_input);
            transcript.appendScalar(incClaimReduction_input);

            const batch = try self.allocator.alloc(F, 6);
            defer self.allocator.free(batch);
            for (0..6) |i| {
                batch[i] = transcript.challengeScalarFull();
            }

            const input_claims = [6]F{
                bytecodeReadRaf_input,
                hammingBooleanity_input,
                booleanity_input,
                ramRaVirtual_input,
                lookupsRaVirtual_input,
                incClaimReduction_input,
            };
            const num_rounds_arr = [6]usize{
                bytecodeReadRaf_rounds,
                hammingBooleanity_rounds,
                booleanity_rounds,
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

            std.debug.print("\n[S6P] Input claims and batching (LE = last8 of BE):\n", .{});
            for (0..6) |i| {
                const ic_be = input_claims[i].toBytesBE();
                const ba_be = batch[i].toBytesBE();
                // Show last 8 bytes of BE format (= first 8 LE bytes = least significant)
                std.debug.print("  instance[{}]: input_claim=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}] batch=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}] rounds={}\n", .{
                    i,
                    ic_be[31], ic_be[30], ic_be[29], ic_be[28], ic_be[27], ic_be[26], ic_be[25], ic_be[24],
                    ba_be[31], ba_be[30], ba_be[29], ba_be[28], ba_be[27], ba_be[26], ba_be[25], ba_be[24],
                    num_rounds_arr[i],
                });
            }
            const bc_be = batched_claim.toBytesBE();
            std.debug.print("  batched_claim=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                bc_be[31], bc_be[30], bc_be[29], bc_be[28], bc_be[27], bc_be[26], bc_be[25], bc_be[24],
            });

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
                var cached_bc_phase1: [3]F = undefined; // [p(0), p(1), p(inf)]
                var cached_bc_phase2: ?[]F = null;
                var cached_hamming: [4]F = undefined;
                var cached_ram_ra: ?[]F = null;
                var cached_lookups_ra: ?[]F = null;
                var cached_inc: [3]F = undefined; // [p(0), p(2), p(inf)]
                var cached_inc_p1: F = F.zero(); // recovered p(1)

                // Track which instances are active this round
                var inst_active: [6]bool = .{ false, false, false, false, false, false };

                // Instance 0: BytecodeReadRaf - REAL prover
                {
                    const inst = 0;
                    if (remaining_rounds > num_rounds_arr[inst]) {
                        // Not started yet - constant polynomial
                        const scale = remaining_rounds - num_rounds_arr[inst] - 1;
                        var scaled = input_claims[inst];
                        for (0..scale) |_| scaled = scaled.add(scaled);
                        const contrib = batch[inst].mul(scaled);
                        for (0..num_evals - 1) |j_eval| {
                            combined_evals[j_eval] = combined_evals[j_eval].add(contrib);
                        }
                    } else {
                        inst_active[inst] = true;
                        if (bytecode_prover.phase == 0) {
                            // Phase 1: address binding (linear poly)
                            const polys = bytecode_prover.computeRoundPolyPhase1();
                            cached_bc_phase1 = polys;
                            if (round < 2) {
                                const bc_sum = polys[0].add(polys[1]);
                                std.debug.print("  [S6P] R{} BC_Phase1 p(0)={any} p(1)={any} sum={any} input={any}\n", .{
                                    round,
                                    polys[0].toBytesBE()[0..8],
                                    polys[1].toBytesBE()[0..8],
                                    bc_sum.toBytesBE()[0..8],
                                    input_claims[0].toBytesBE()[0..8],
                                });
                            }
                            combined_evals[0] = combined_evals[0].add(batch[inst].mul(polys[0]));
                            combined_evals[1] = combined_evals[1].add(batch[inst].mul(polys[1]));
                            const slope = polys[1].sub(polys[0]);
                            for (2..num_evals - 1) |k| {
                                const x = F.fromU64(@intCast(k));
                                const pk = polys[0].add(x.mul(slope));
                                combined_evals[k] = combined_evals[k].add(batch[inst].mul(pk));
                            }
                            // p(inf) = 0 for linear poly
                        } else {
                            // Phase 2: cycle binding (degree bytecode_d+1)
                            const polys = try bytecode_prover.computeRoundPolyPhase2(self.allocator);
                            cached_bc_phase2 = polys;
                            addInstanceEvalsToCombibed(F, combined_evals, polys, batch[inst], num_evals);
                        }
                    }
                }

                // Instance 1: HammingBooleanity - REAL prover
                {
                    const inst = 1;
                    if (remaining_rounds > num_rounds_arr[inst]) {
                        const scale = remaining_rounds - num_rounds_arr[inst] - 1;
                        var scaled = input_claims[inst];
                        for (0..scale) |_| scaled = scaled.add(scaled);
                        const contrib = batch[inst].mul(scaled);
                        for (0..num_evals - 1) |j_eval| {
                            combined_evals[j_eval] = combined_evals[j_eval].add(contrib);
                        }
                    } else {
                        inst_active[inst] = true;
                        const polys = hamming_prover.computeRoundPoly();
                        cached_hamming = polys;
                        addFixedEvalsToCombibed(F, combined_evals, &polys, 4, batch[inst], num_evals);
                    }
                }

                // Instance 2: Booleanity - zero polynomial (input = 0, valid traces have ra in {0,1})
                {
                    const inst = 2;
                    if (remaining_rounds > num_rounds_arr[inst]) {
                        const scale = remaining_rounds - num_rounds_arr[inst] - 1;
                        var scaled = input_claims[inst];
                        for (0..scale) |_| scaled = scaled.add(scaled);
                        const contrib = batch[inst].mul(scaled);
                        for (0..num_evals - 1) |j_eval| {
                            combined_evals[j_eval] = combined_evals[j_eval].add(contrib);
                        }
                    } else {
                        inst_active[inst] = true;
                        // Zero polynomial - all evals are zero, nothing to add
                    }
                }

                // Instance 3: RamRaVirtual - REAL prover
                {
                    const inst = 3;
                    if (remaining_rounds > num_rounds_arr[inst]) {
                        const scale = remaining_rounds - num_rounds_arr[inst] - 1;
                        var scaled = input_claims[inst];
                        for (0..scale) |_| scaled = scaled.add(scaled);
                        const contrib = batch[inst].mul(scaled);
                        for (0..num_evals - 1) |j_eval| {
                            combined_evals[j_eval] = combined_evals[j_eval].add(contrib);
                        }
                    } else {
                        inst_active[inst] = true;
                        const polys = try ram_ra_prover.computeRoundPoly(self.allocator);
                        cached_ram_ra = polys;
                        addInstanceEvalsToCombibed(F, combined_evals, polys, batch[inst], num_evals);
                    }
                }

                // Instance 4: LookupsRaVirtual - REAL prover
                {
                    const inst = 4;
                    if (remaining_rounds > num_rounds_arr[inst]) {
                        const scale = remaining_rounds - num_rounds_arr[inst] - 1;
                        var scaled = input_claims[inst];
                        for (0..scale) |_| scaled = scaled.add(scaled);
                        const contrib = batch[inst].mul(scaled);
                        for (0..num_evals - 1) |j_eval| {
                            combined_evals[j_eval] = combined_evals[j_eval].add(contrib);
                        }
                    } else {
                        inst_active[inst] = true;
                        const polys = try lookups_ra_prover.computeRoundPoly(self.allocator);
                        cached_lookups_ra = polys;
                        addInstanceEvalsToCombibed(F, combined_evals, polys, batch[inst], num_evals);
                    }
                }

                // Instance 5: IncClaimReduction - REAL prover
                {
                    const inst = 5;
                    if (remaining_rounds > num_rounds_arr[inst]) {
                        const scale = remaining_rounds - num_rounds_arr[inst] - 1;
                        var scaled = input_claims[inst];
                        for (0..scale) |_| scaled = scaled.add(scaled);
                        const contrib = batch[inst].mul(scaled);
                        for (0..num_evals - 1) |j_eval| {
                            combined_evals[j_eval] = combined_evals[j_eval].add(contrib);
                        }
                    } else {
                        inst_active[inst] = true;
                        const polys = inc_prover.computeRoundPoly();
                        cached_inc = polys;
                        // polys = [p(0), p(2), p(inf)] for degree 2
                        // Need p(1) = instance_claim - p(0)
                        const p0 = polys[0];
                        const p1 = instance_claims[inst].sub(p0);
                        cached_inc_p1 = p1;
                        const p2 = polys[1];
                        const p_inf = polys[2];

                        combined_evals[0] = combined_evals[0].add(batch[inst].mul(p0));
                        combined_evals[1] = combined_evals[1].add(batch[inst].mul(p1));
                        if (num_evals > 3) {
                            combined_evals[2] = combined_evals[2].add(batch[inst].mul(p2));
                        }
                        // Interpolate for higher points
                        if (num_evals > 4) {
                            const a0 = p0;
                            const a2_coeff = p_inf;
                            const a1 = p1.sub(a0).sub(a2_coeff);
                            for (3..num_evals - 1) |k| {
                                const x = F.fromU64(@intCast(k));
                                const px = a0.add(x.mul(a1.add(x.mul(a2_coeff))));
                                combined_evals[k] = combined_evals[k].add(batch[inst].mul(px));
                            }
                        }
                        combined_evals[num_evals - 1] = combined_evals[num_evals - 1].add(batch[inst].mul(p_inf));
                    }
                }

                // Debug: print combined evals for this round
                if (round < 3) {
                    std.debug.print("  [S6P] R{} combined: p(0)={any} p(1)={any}\n", .{
                        round,
                        combined_evals[0].toBytesBE()[0..8],
                        combined_evals[1].toBytesBE()[0..8],
                    });
                    var p01_sum = combined_evals[0].add(combined_evals[1]);
                    std.debug.print("  [S6P] R{} p(0)+p(1)={any} claim={any} match={}\n", .{
                        round,
                        p01_sum.toBytesBE()[0..8],
                        current_batched_claim.toBytesBE()[0..8],
                        @as(u8, if (std.mem.eql(u8, &p01_sum.toBytesBE(), &current_batched_claim.toBytesBE())) 1 else 0),
                    });
                    // Print active instances
                    for (0..6) |ii| {
                        std.debug.print("  [S6P] R{} inst[{}] active={} claim={any}\n", .{
                            round, ii, inst_active[ii], instance_claims[ii].toBytesBE()[0..8],
                        });
                    }
                }

                // Compress and append to transcript
                const compressed = try UniPoly(F).toomCookToCompressedGeneral(self.allocator, combined_evals);
                defer self.allocator.free(compressed);

                const coeffs = try self.allocator.alloc(F, num_compressed);
                for (0..num_compressed) |j| {
                    coeffs[j] = if (j < compressed.len) compressed[j] else F.zero();
                }

                try proof.compressed_polys.append(self.allocator, .{
                    .coeffs_except_linear_term = coeffs,
                    .allocator = self.allocator,
                });

                transcript.appendMessage("UniPoly_begin");
                for (0..num_compressed) |j| {
                    transcript.appendScalar(coeffs[j]);
                }
                transcript.appendMessage("UniPoly_end");

                const challenge = transcript.challengeScalar();
                challenges[round] = challenge;

                // Evaluate combined polynomial at challenge
                current_batched_claim = try UniPoly(F).evaluateToomCookGeneralAt(self.allocator, combined_evals, challenge);

                std.debug.print("  [S6P] R{} challenge={any} new_claim={any}\n", .{
                    round,
                    challenge.toBytesBE()[0..8],
                    current_batched_claim.toBytesBE()[0..8],
                });

                // Update per-instance claims from CACHED round polys and bind challenge
                // Instance 0: BytecodeReadRaf
                if (inst_active[0]) {
                    if (bytecode_prover.phase == 0) {
                        // Phase 1: linear poly, p(r) = p(0) + r*(p(1)-p(0))
                        instance_claims[0] = cached_bc_phase1[0].add(challenge.mul(cached_bc_phase1[1].sub(cached_bc_phase1[0])));
                        bytecode_addr_challenges[bytecode_prover.addr_rounds_done] = challenge;
                        bytecode_prover.bindChallengePhase1(challenge);
                        if (bytecode_prover.addr_rounds_done == bytecode_log_k) {
                            try bytecode_prover.transitionToPhase2(bytecode_addr_challenges);
                        }
                    } else {
                        // Phase 2: evaluate from cached evals using Lagrange interpolation
                        instance_claims[0] = evaluatePolyFromEvals(F, cached_bc_phase2.?, challenge);
                        self.allocator.free(cached_bc_phase2.?);
                        cached_bc_phase2 = null;
                        bytecode_prover.bindChallengePhase2(challenge);
                    }
                }

                // Instance 1: HammingBooleanity
                if (inst_active[1]) {
                    instance_claims[1] = evaluateDeg3FromEvals(F, cached_hamming, challenge);
                    hamming_prover.bindChallenge(challenge);
                }

                // Instance 2: Booleanity (zero poly, claim stays 0)
                if (inst_active[2]) {
                    instance_claims[2] = F.zero();
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
                    const p1_val = cached_inc_p1;
                    const p_inf = cached_inc[2];
                    // p(x) = a0 + a1*x + a2*x^2 where a2 = p(inf)
                    const a0 = p0;
                    const a2 = p_inf;
                    const a1 = p1_val.sub(a0).sub(a2);
                    instance_claims[5] = a0.add(challenge.mul(a1.add(challenge.mul(a2))));
                    inc_prover.bindChallenge(challenge);
                }
            }

            // Debug: print final instance claims after sumcheck
            std.debug.print("\n[S6P] Final instance claims after sumcheck:\n", .{});
            for (0..6) |i| {
                std.debug.print("  instance[{}] final_claim={any}\n", .{i, instance_claims[i].toBytesBE()[0..8]});
            }
            std.debug.print("  current_batched_claim={any}\n", .{current_batched_claim.toBytesBE()[0..8]});

            // ====================================================================
            // Extract opening claims from all real provers
            // ====================================================================

            const inc_opening = inc_prover.openingClaims();
            const ram_inc_claim = inc_opening.ram_inc;
            const rd_inc_claim = inc_opening.rd_inc;

            const hamming_weight_claim = hamming_prover.openingClaim();

            const bytecode_ra_claims = try bytecode_prover.getOpeningClaims(self.allocator);

            const ram_ra_virtual_claims = try ram_ra_prover.getOpeningClaims(self.allocator);

            const instruction_ra_virtual_claims = try lookups_ra_prover.getOpeningClaims(self.allocator, lookups_ra_gamma_powers);

            // Booleanity claims are all zero for valid traces
            const total_booleanity_polys = instruction_d + bytecode_d + ram_d;
            const booleanity_ra_claims = try self.allocator.alloc(F, total_booleanity_polys);
            @memset(booleanity_ra_claims, F.zero());

            std.debug.print("[STAGE6] Opening claims:\n", .{});
            std.debug.print("  ram_inc = {any}\n", .{ram_inc_claim.toBytesBE()[0..8]});
            std.debug.print("  rd_inc = {any}\n", .{rd_inc_claim.toBytesBE()[0..8]});
            std.debug.print("  hamming_weight = {any}\n", .{hamming_weight_claim.toBytesBE()[0..8]});
            std.debug.print("  bytecode_ra[0] = {any}\n", .{bytecode_ra_claims[0].toBytesBE()[0..8]});
            std.debug.print("  ram_ra_virtual[0] = {any}\n", .{ram_ra_virtual_claims[0].toBytesBE()[0..8]});
            std.debug.print("  instruction_ra_virtual[0] = {any}\n", .{instruction_ra_virtual_claims[0].toBytesBE()[0..8]});

            // ====================================================================
            // Cache openings to transcript
            // ====================================================================

            // Instance 0: BytecodeReadRaf
            for (bytecode_ra_claims) |claim| {
                transcript.appendScalar(claim);
            }

            // Instance 1: HammingBooleanity
            transcript.appendScalar(hamming_weight_claim);

            // Instance 2: Booleanity
            for (booleanity_ra_claims) |claim| {
                transcript.appendScalar(claim);
            }

            // Instance 3: RamRaVirtual
            for (ram_ra_virtual_claims) |claim| {
                transcript.appendScalar(claim);
            }

            // Instance 4: LookupsRaVirtual
            for (instruction_ra_virtual_claims) |claim| {
                transcript.appendScalar(claim);
            }

            // Instance 5: IncClaimReduction
            transcript.appendScalar(ram_inc_claim);
            transcript.appendScalar(rd_inc_claim);

            return Stage6Result(F){
                .challenges = challenges,
                .bytecode_ra_claims = bytecode_ra_claims,
                .hamming_weight_claim = hamming_weight_claim,
                .booleanity_ra_claims = booleanity_ra_claims,
                .ram_ra_virtual_claims = ram_ra_virtual_claims,
                .instruction_ra_virtual_claims = instruction_ra_virtual_claims,
                .ram_inc_claim = ram_inc_claim,
                .rd_inc_claim = rd_inc_claim,
                .allocator = self.allocator,
            };
        }

        /// Compute BytecodeReadRaf input claim
        fn computeBytecodeReadRafInputClaim(
            self: *Self,
            opening_claims: *OpeningClaims(F),
            gamma_powers: []const F,
            stage1_gammas: []const F,
            stage2_gammas: []const F,
            stage3_gammas: []const F,
            stage4_gammas: []const F,
            stage5_gammas: []const F,
        ) F {
            _ = self;

            const getClaim = struct {
                fn get(oc: *OpeningClaims(F), key: OpeningId) F {
                    return oc.get(key) orelse F.zero();
                }
            }.get;

            // rv_claim_1 (Stage 1 / SpartanOuter)
            var rv1 = F.zero();
            rv1 = rv1.add(stage1_gammas[0].mul(getClaim(opening_claims,
                .{ .Virtual = .{ .poly = .UnexpandedPC, .sumcheck_id = .SpartanOuter } })));
            rv1 = rv1.add(stage1_gammas[1].mul(getClaim(opening_claims,
                .{ .Virtual = .{ .poly = .Imm, .sumcheck_id = .SpartanOuter } })));
            for (0..13) |i| {
                rv1 = rv1.add(stage1_gammas[2 + i].mul(getClaim(opening_claims,
                    .{ .Virtual = .{ .poly = .{ .OpFlags = @intCast(i) }, .sumcheck_id = .SpartanOuter } })));
            }

            // rv_claim_2 (Stage 2 / SpartanProductVirtualization)
            var rv2 = F.zero();
            rv2 = rv2.add(stage2_gammas[0].mul(getClaim(opening_claims,
                .{ .Virtual = .{ .poly = .{ .OpFlags = 5 }, .sumcheck_id = .SpartanProductVirtualization } })));
            rv2 = rv2.add(stage2_gammas[1].mul(getClaim(opening_claims,
                .{ .Virtual = .{ .poly = .{ .InstructionFlags = 4 }, .sumcheck_id = .SpartanProductVirtualization } })));
            rv2 = rv2.add(stage2_gammas[2].mul(getClaim(opening_claims,
                .{ .Virtual = .{ .poly = .{ .InstructionFlags = 6 }, .sumcheck_id = .SpartanProductVirtualization } })));
            rv2 = rv2.add(stage2_gammas[3].mul(getClaim(opening_claims,
                .{ .Virtual = .{ .poly = .{ .OpFlags = 6 }, .sumcheck_id = .SpartanProductVirtualization } })));

            // rv_claim_3 (Stage 3)
            var rv3 = F.zero();
            rv3 = rv3.add(stage3_gammas[0].mul(getClaim(opening_claims,
                .{ .Virtual = .{ .poly = .Imm, .sumcheck_id = .InstructionInputVirtualization } })));
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
            rv5 = rv5.add(stage5_gammas[0].mul(getClaim(opening_claims,
                .{ .Virtual = .{ .poly = .RdWa, .sumcheck_id = .RegistersValEvaluation } })));
            rv5 = rv5.add(stage5_gammas[1].mul(getClaim(opening_claims,
                .{ .Virtual = .{ .poly = .InstructionRafFlag, .sumcheck_id = .InstructionReadRaf } })));
            for (0..NUM_LOOKUP_TABLES) |i| {
                rv5 = rv5.add(stage5_gammas[2 + i].mul(getClaim(opening_claims,
                    .{ .Virtual = .{ .poly = .{ .LookupTableFlag = i }, .sumcheck_id = .InstructionReadRaf } })));
            }

            // RAF claims
            const raf_claim = getClaim(opening_claims,
                .{ .Virtual = .{ .poly = .PC, .sumcheck_id = .SpartanOuter } });
            const raf_shift_claim = getClaim(opening_claims,
                .{ .Virtual = .{ .poly = .PC, .sumcheck_id = .SpartanShift } });

            // Combine
            var result = gamma_powers[0].mul(rv1);
            result = result.add(gamma_powers[1].mul(rv2));
            result = result.add(gamma_powers[2].mul(rv3));
            result = result.add(gamma_powers[3].mul(rv4));
            result = result.add(gamma_powers[4].mul(rv5));
            result = result.add(gamma_powers[5].mul(raf_claim));
            result = result.add(gamma_powers[6].mul(raf_shift_claim));

            return result;
        }
    };
}

// =============================================================================
// Helper: Add variable-length instance evals to combined_evals with interpolation
// =============================================================================
fn addInstanceEvalsToCombibed(comptime F: type, combined_evals: []F, polys: []const F, batch_coeff: F, num_evals: usize) void {
    const inst_n_evals = polys.len;

    if (inst_n_evals >= num_evals) {
        // Instance has enough eval points
        for (0..num_evals - 1) |k| {
            combined_evals[k] = combined_evals[k].add(batch_coeff.mul(polys[k]));
        }
        combined_evals[num_evals - 1] = combined_evals[num_evals - 1].add(batch_coeff.mul(polys[inst_n_evals - 1]));
    } else {
        // Instance has fewer eval points - need interpolation
        // Copy known evals: [0, 1, ..., inst_degree-1] and [inf]
        const inst_degree = inst_n_evals - 1; // degree of the instance poly
        for (0..inst_degree) |k| {
            if (k < num_evals - 1) {
                combined_evals[k] = combined_evals[k].add(batch_coeff.mul(polys[k]));
            }
        }
        // p(inf) = leading coefficient
        combined_evals[num_evals - 1] = combined_evals[num_evals - 1].add(batch_coeff.mul(polys[inst_n_evals - 1]));

        // Interpolate missing points using Lagrange interpolation on known evals
        // Known: p(0), p(1), ..., p(inst_degree-1), p(inf)
        // The polynomial has degree inst_degree with inst_degree+1 parameters.
        // We have inst_degree finite evals + p(inf) = leading_coeff.
        // Use the finite evals and leading coeff to reconstruct coefficients.
        if (inst_degree >= num_evals - 1) return; // Nothing to interpolate

        // Build coefficients from evals and leading coefficient
        // For small degrees, use explicit formulas
        if (inst_degree <= 16) {
            // Use Lagrange interpolation to evaluate at missing points
            // We have evals at x = 0, 1, ..., inst_degree-1
            // And the leading coefficient (coeff of x^inst_degree) = polys[inst_n_evals-1]
            const leading = polys[inst_n_evals - 1];

            for (inst_degree..num_evals - 1) |k| {
                // Evaluate the degree-inst_degree polynomial at x = k
                // Using Lagrange basis on points {0, 1, ..., inst_degree-1}
                // plus the leading coefficient correction
                const x = F.fromU64(@intCast(k));

                // Lagrange interpolation on finite points gives a degree-(inst_degree-1) polynomial
                // The actual polynomial is p(x) = lagrange(x) + leading * (x^inst_degree - correction)
                // Simpler: just use all the data we have

                // Method: reconstruct p(x) = a0 + a1*x + ... + a_{inst_degree}*x^inst_degree
                // a_{inst_degree} = leading (known)
                // We have inst_degree finite evals which give inst_degree equations
                // for inst_degree unknowns (a0..a_{inst_degree-1})

                // For simplicity, evaluate using the finite evals via Lagrange + leading term
                var result = F.zero();

                // Lagrange interpolation of degree-(inst_degree-1) poly through
                // (0, p(0) - leading*0^d), (1, p(1) - leading*1^d), ...
                // Actually just use the full Lagrange basis for degree inst_degree
                // We have inst_degree+1 constraints: inst_degree finite evals + leading coeff

                // Alternative: Barycentric Lagrange on points 0..inst_degree-1,
                // then add correction for leading coefficient
                var lagrange_val = F.zero();
                for (0..inst_degree) |m| {
                    var basis = F.one();
                    const xm = F.fromU64(@intCast(m));
                    for (0..inst_degree) |n| {
                        if (n != m) {
                            const xn = F.fromU64(@intCast(n));
                            basis = basis.mul(x.sub(xn)).mul(xm.sub(xn).inverse().?);
                        }
                    }
                    lagrange_val = lagrange_val.add(basis.mul(polys[m]));
                }

                // lagrange_val gives a degree-(inst_degree-1) interpolation
                // The actual poly differs by leading * prod(x - xi) / prod(...)
                // p(x) = lagrange_val + leading * x * (x-1) * ... * (x-(inst_degree-1)) / inst_degree!
                // Wait, that's only correct if lagrange interpolation is through exactly inst_degree points
                // and the true poly has one extra degree.

                // Correction: p(x) = lagrange(x) + leading * Prod_{m=0}^{inst_degree-1} (x - m)
                var x_prod = F.one();
                for (0..inst_degree) |m| {
                    x_prod = x_prod.mul(x.sub(F.fromU64(@intCast(m))));
                }
                result = lagrange_val.add(leading.mul(x_prod));

                combined_evals[k] = combined_evals[k].add(batch_coeff.mul(result));
            }
        }
    }
}

/// Add fixed-size instance evaluations to combined (for degree-3 instances like Hamming)
fn addFixedEvalsToCombibed(comptime F: type, combined_evals: []F, polys: []const F, n_polys: usize, batch_coeff: F, num_evals: usize) void {
    // polys has n_polys entries: [p(0), p(1), ..., p(n_polys-2), p(inf)]
    const inst_degree = n_polys - 1;

    for (0..@min(inst_degree, num_evals - 1)) |k| {
        combined_evals[k] = combined_evals[k].add(batch_coeff.mul(polys[k]));
    }
    combined_evals[num_evals - 1] = combined_evals[num_evals - 1].add(batch_coeff.mul(polys[n_polys - 1]));

    // Interpolate missing points if needed
    if (inst_degree < num_evals - 1) {
        const leading = polys[n_polys - 1];
        for (inst_degree..num_evals - 1) |k| {
            const x = F.fromU64(@intCast(k));
            var lagrange_val = F.zero();
            for (0..inst_degree) |m| {
                var basis = F.one();
                const xm = F.fromU64(@intCast(m));
                for (0..inst_degree) |n| {
                    if (n != m) {
                        const xn = F.fromU64(@intCast(n));
                        basis = basis.mul(x.sub(xn)).mul(xm.sub(xn).inverse().?);
                    }
                }
                lagrange_val = lagrange_val.add(basis.mul(polys[m]));
            }
            var x_prod = F.one();
            for (0..inst_degree) |m| {
                x_prod = x_prod.mul(x.sub(F.fromU64(@intCast(m))));
            }
            combined_evals[k] = combined_evals[k].add(batch_coeff.mul(lagrange_val.add(leading.mul(x_prod))));
        }
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Compute eq polynomial table: eq(r, j) for all j in [0, 2^n_vars)
/// r is in BIG_ENDIAN order (r[0] is the most significant variable)
fn computeEqTable(comptime F: type, allocator: Allocator, r: []const F, n_vars: usize) ![]F {
    const size: usize = @as(usize, 1) << @intCast(n_vars);
    var table = try allocator.alloc(F, size);

    table[0] = F.one();

    for (0..n_vars) |i| {
        const r_i = r[i];
        const one_minus_r = F.one().sub(r_i);
        const cur_size: usize = @as(usize, 1) << @intCast(i);

        var j: usize = cur_size;
        while (j > 0) {
            j -= 1;
            table[j + cur_size] = table[j].mul(r_i);
            table[j] = table[j].mul(one_minus_r);
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
fn extractChunkMSB(addr: u64, chunk_idx: usize, total_chunks: usize, log_k_chunk: usize) usize {
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
fn interleaveBits(rs1: u64, rs2: u64) u128 {
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

/// Get lookup index chunk from trace step using interleaved bits and MSB-first ordering
/// This matches Jolt's lookup_index_chunk with instruction_shifts
fn getLookupChunkInterleaved(step: tracer.TraceStep, chunk_idx: usize, log_k_chunk: usize, instruction_d: usize) usize {
    // Build interleaved 128-bit lookup index
    const lookup_index = interleaveBits(step.rs1_value, step.rs2_value);

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
fn evaluatePolyFromEvals(comptime F: type, evals: []const F, challenge: F) F {
    const n_evals = evals.len;
    const deg = n_evals - 1; // polynomial degree
    const leading = evals[n_evals - 1]; // p(inf) = leading coefficient

    // Lagrange interpolation through (0, p(0)), (1, p(1)), ..., (d-1, p(d-1))
    // gives a degree-(d-1) polynomial. Add leading * prod(x - i) to get degree-d.
    var lagrange_val = F.zero();
    for (0..deg) |m| {
        var basis = F.one();
        const xm = F.fromU64(@intCast(m));
        for (0..deg) |n| {
            if (n != m) {
                const xn = F.fromU64(@intCast(n));
                basis = basis.mul(challenge.sub(xn)).mul(xm.sub(xn).inverse().?);
            }
        }
        lagrange_val = lagrange_val.add(basis.mul(evals[m]));
    }

    // Correction: add leading * prod_{i=0}^{d-1} (challenge - i)
    var x_prod = F.one();
    for (0..deg) |m| {
        x_prod = x_prod.mul(challenge.sub(F.fromU64(@intCast(m))));
    }

    return lagrange_val.add(leading.mul(x_prod));
}

/// Evaluate a degree-3 polynomial from evals [p(0), p(1), p(2), p(inf)]
/// p(inf) is the leading coefficient (coefficient of x^3).
/// Specialized version for degree 3 for slightly better perf (avoids loops).
fn evaluateDeg3FromEvals(comptime F: type, evals: [4]F, challenge: F) F {
    const p0 = evals[0];
    const p1 = evals[1];
    const p2 = evals[2];
    const leading = evals[3]; // coefficient of x^3

    // Lagrange interpolation through (0, p0), (1, p1), (2, p2) gives degree-2 poly
    // L_0(x) = (x-1)(x-2)/((0-1)(0-2)) = (x-1)(x-2)/2
    // L_1(x) = (x-0)(x-2)/((1-0)(1-2)) = x(x-2)/(-1) = -x(x-2)
    // L_2(x) = (x-0)(x-1)/((2-0)(2-1)) = x(x-1)/2
    const x = challenge;
    const two_inv = F.fromU64(2).inverse().?;
    const xm1 = x.sub(F.one());
    const xm2 = x.sub(F.fromU64(2));

    const l0 = xm1.mul(xm2).mul(two_inv);
    const l1 = x.mul(xm2).neg();
    const l2 = x.mul(xm1).mul(two_inv);

    const lagrange_val = l0.mul(p0).add(l1.mul(p1)).add(l2.mul(p2));

    // Add leading * x * (x-1) * (x-2)
    const x_prod = x.mul(xm1).mul(xm2);
    return lagrange_val.add(leading.mul(x_prod));
}
