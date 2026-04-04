//! Jolt-compatible preprocessing serialization
//!
//! This module provides the ability to export Zolt preprocessing in a format
//! that can be loaded and used by Jolt's verifier. This enables cross-verification
//! where Zolt generates both the proof and the preprocessing for the same program.
//!
//! The preprocessing contains:
//! - BytecodePreprocessing: The program's bytecode in Jolt Instruction format
//! - RAMPreprocessing: Initial memory state
//! - MemoryLayout: Memory region addresses
//! - VerifierSetup: Commitment scheme generators (Dory)
//!
//! Reference: jolt-core/src/zkvm/verifier.rs

const std = @import("std");

const zkvm_debug = @import("debug.zig");
const dbg = zkvm_debug.dbg;

const Allocator = std.mem.Allocator;
const jolt_device = @import("jolt_device.zig");
const MemoryLayout = jolt_device.MemoryLayout;
const common = @import("../common/mod.zig");

// Re-exported submodules
pub const instruction_decoder = @import("instruction_decoder.zig");
pub const decodeToJoltInstruction = instruction_decoder.decodeToJoltInstruction;

pub const bytecode_pc_mapper = @import("bytecode_pc_mapper.zig");
pub const BytecodePCMapper = bytecode_pc_mapper.BytecodePCMapper;

/// A RISC-V instruction in Jolt's format
/// This is serialized as JSON for arkworks CanonicalSerialize compatibility
pub const JoltInstruction = struct {
    /// Instruction variant name (e.g., "ADD", "ADDI", "LW", etc.)
    variant: InstructionVariant,
    /// Memory address where the instruction is located
    address: u64,
    /// Decoded operands
    operands: Operands,
    /// For virtual instruction sequences, the remaining count
    virtual_sequence_remaining: ?u16,
    /// Whether this is the first instruction in a virtual sequence
    is_first_in_sequence: bool,
    /// Whether this was a compressed (RVC) instruction
    is_compressed: bool,

    pub const InstructionVariant = enum {
        NoOp,
        UNIMPL,
        // RV32I Base
        ADD,
        ADDI,
        AND,
        ANDI,
        AUIPC,
        BEQ,
        BGE,
        BGEU,
        BLT,
        BLTU,
        BNE,
        JAL,
        JALR,
        LB,
        LBU,
        LD,
        LH,
        LHU,
        LUI,
        LW,
        LWU,
        OR,
        ORI,
        SB,
        SD,
        SH,
        SLL,
        SLLI,
        SLT,
        SLTI,
        SLTIU,
        SLTU,
        SRA,
        SRAI,
        SRL,
        SRLI,
        SUB,
        SW,
        XOR,
        XORI,
        // RV64I
        ADDIW,
        ADDW,
        SLLIW,
        SLLW,
        SRAIW,
        SRAW,
        SRLIW,
        SRLW,
        SUBW,
        // M Extension
        DIV,
        DIVU,
        DIVUW,
        DIVW,
        MUL,
        MULH,
        MULHSU,
        MULHU,
        MULW,
        REM,
        REMU,
        REMUW,
        REMW,
        // System
        ECALL,
        FENCE,
        // Atomics (placeholder)
        // Virtual instructions (names must match Jolt's Rust enum exactly for JSON serialization)
        VirtualAdvice,
        VirtualAssertEQ,
        VirtualAssertLTE,
        VirtualAssertValidDiv0,
        VirtualAssertValidUnsignedRemainder,
        VirtualChangeDivisorW,
        VirtualSignExtendWord,
        VirtualZeroExtendWord,
        VirtualMULI,
        VirtualPow2,
        VirtualSRAI,
        VirtualSRLI,
        VirtualShiftRightBitmask,
        VirtualAssertHalfwordAlignment,
        VirtualAssertWordAlignment,
        VirtualSRL,
        VirtualSRA,
    };

    /// Instruction operands - different formats store different fields
    /// Types match Jolt's Rust struct definitions:
    /// - FormatR: rd/rs1/rs2 as u8
    /// - FormatI: imm as u64 (sign-extended from i32)
    /// - FormatS: imm as i64 (signed)
    /// - FormatB: imm as i128 (signed)
    /// - FormatU: imm as u64 (sign-extended from i32)
    /// - FormatJ: imm as u64 (sign-extended from i32)
    pub const Operands = union(enum) {
        /// R-type: rd, rs1, rs2
        FormatR: struct { rd: u8, rs1: u8, rs2: u8 },
        /// I-type: rd, rs1, imm (Jolt uses u64)
        FormatI: struct { rd: u8, rs1: u8, imm: u64 },
        /// Load-type: rd, rs1, imm (Jolt uses i64 — distinct from FormatI)
        FormatLoad: struct { rd: u8, rs1: u8, imm: i64 },
        /// S-type: rs1, rs2, imm (Jolt uses i64)
        FormatS: struct { rs1: u8, rs2: u8, imm: i64 },
        /// B-type: rs1, rs2, imm (Jolt uses i128)
        FormatB: struct { rs1: u8, rs2: u8, imm: i128 },
        /// U-type: rd, imm (Jolt uses u64)
        FormatU: struct { rd: u8, imm: u64 },
        /// J-type: rd, imm (Jolt uses u64)
        FormatJ: struct { rd: u8, imm: u64 },
        /// Assert format: rs1, imm (no rd — used by alignment assertions)
        /// imm is u64 (unsigned), matching FormatI encoding so that the Jolt verifier's
        /// NormalizedOperands.imm = u64 as i128 (zero-extension) produces the same
        /// field element as the R1CS witness's F.fromU64(@bitCast(imm_signed)).
        FormatAssert: struct { rs1: u8, imm: i64 },
        /// No operands (NoOp, FENCE, ECALL)
        None: void,
    };

    /// Serialize this instruction to JSON bytes (for arkworks compatibility)
    /// NoOp and UNIMPL are unit variants in Jolt, so they serialize as just "NoOp" or "UNIMPL"
    /// Other instructions serialize as {"VARIANT":{...fields...}}
    pub fn toJson(self: JoltInstruction, allocator: Allocator) ![]u8 {
        var list = std.ArrayListUnmanaged(u8){};
        errdefer list.deinit(allocator);
        const writer = list.writer(allocator);

        // NoOp and UNIMPL are unit variants in Jolt's Instruction enum
        // They serialize as just "NoOp" or "UNIMPL" (a JSON string)
        if (self.variant == .NoOp) {
            try writer.writeAll("\"NoOp\"");
            return list.toOwnedSlice(allocator);
        }
        if (self.variant == .UNIMPL) {
            try writer.writeAll("\"UNIMPL\"");
            return list.toOwnedSlice(allocator);
        }

        // Other instructions: {"VARIANT":{"address":123,"operands":{...},...}}
        try writer.writeAll("{\"");
        try writer.writeAll(@tagName(self.variant));
        try writer.writeAll("\":{\"address\":");
        try std.fmt.format(writer, "{}", .{self.address});

        // Operands
        try writer.writeAll(",\"operands\":");
        switch (self.operands) {
            .FormatR => |r| {
                try std.fmt.format(writer, "{{\"rd\":{},\"rs1\":{},\"rs2\":{}}}", .{ r.rd, r.rs1, r.rs2 });
            },
            .FormatI => |i| {
                try std.fmt.format(writer, "{{\"rd\":{},\"rs1\":{},\"imm\":{}}}", .{ i.rd, i.rs1, i.imm });
            },
            .FormatLoad => |l| {
                try std.fmt.format(writer, "{{\"rd\":{},\"rs1\":{},\"imm\":{}}}", .{ l.rd, l.rs1, l.imm });
            },
            .FormatS => |s| {
                try std.fmt.format(writer, "{{\"rs1\":{},\"rs2\":{},\"imm\":{}}}", .{ s.rs1, s.rs2, s.imm });
            },
            .FormatB => |b| {
                try std.fmt.format(writer, "{{\"rs1\":{},\"rs2\":{},\"imm\":{}}}", .{ b.rs1, b.rs2, b.imm });
            },
            .FormatU => |u_op| {
                try std.fmt.format(writer, "{{\"rd\":{},\"imm\":{}}}", .{ u_op.rd, u_op.imm });
            },
            .FormatJ => |j| {
                try std.fmt.format(writer, "{{\"rd\":{},\"imm\":{}}}", .{ j.rd, j.imm });
            },
            .FormatAssert => |a| {
                try std.fmt.format(writer, "{{\"rs1\":{},\"imm\":{}}}", .{ a.rs1, a.imm });
            },
            .None => {
                try writer.writeAll("{}");
            },
        }

        // virtual_sequence_remaining
        try writer.writeAll(",\"virtual_sequence_remaining\":");
        if (self.virtual_sequence_remaining) |vsr| {
            try std.fmt.format(writer, "{}", .{vsr});
        } else {
            try writer.writeAll("null");
        }

        // is_first_in_sequence
        try writer.writeAll(",\"is_first_in_sequence\":");
        try writer.writeAll(if (self.is_first_in_sequence) "true" else "false");

        // VirtualAdvice has an extra 'advice' field (u64) that other instructions don't have.
        // In preprocessing, advice is always 0 (actual values are filled at runtime).
        if (self.variant == .VirtualAdvice) {
            try writer.writeAll(",\"advice\":0");
        }

        // is_compressed
        try writer.writeAll(",\"is_compressed\":");
        try writer.writeAll(if (self.is_compressed) "true" else "false");

        try writer.writeAll("}}");

        return list.toOwnedSlice(allocator);
    }
};

/// BytecodePreprocessing - matches Jolt's BytecodePreprocessing
pub const BytecodePreprocessing = struct {
    /// Power-of-2 padded code size
    code_size: usize,
    /// Vector of instructions (serialized as JSON)
    bytecode: std.ArrayListUnmanaged(JoltInstruction),
    /// PC mapper
    pc_map: BytecodePCMapper,
    /// Raw 32-bit instruction words (one per bytecode entry, including NoOp=0 at index 0
    /// and virtual instruction words). Used by Jolt verifier for Zolt-compatible flag computation.
    raw_words: std.ArrayListUnmanaged(u32),
    /// ELF entry point address (e_entry). Serialized after pc_map to match upstream.
    entry_address: u64,

    allocator: Allocator,

    pub fn init(allocator: Allocator) BytecodePreprocessing {
        return .{
            .code_size = 0,
            .bytecode = .{},
            .pc_map = BytecodePCMapper.init(allocator),
            .raw_words = .{},
            .entry_address = 0,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *BytecodePreprocessing) void {
        self.bytecode.deinit(self.allocator);
        self.raw_words.deinit(self.allocator);
        self.pc_map.deinit();
    }

    /// Preprocess bytecode from raw bytes.
    /// `termination_address_opt` is the memory-mapped I/O address where termination is signaled
    /// (from MemoryLayout.termination). This is used to generate the synthetic LUI+ADDI+SD
    /// termination sequence that matches the prover's bytecode table.
    /// If null, uses the default MemoryLayout termination address (0x7FFFC008).
    pub fn preprocess(allocator: Allocator, code_bytes: []const u8, base_address: u64, termination_address_opt: ?u64) !BytecodePreprocessing {
        return preprocessWithTextSize(allocator, code_bytes, base_address, termination_address_opt, code_bytes.len);
    }

    /// Preprocess bytecode, only decoding instructions within the first `text_size` bytes.
    /// Bytes beyond text_size are treated as data (.rodata) and skipped.
    pub fn preprocessWithTextSize(allocator: Allocator, code_bytes: []const u8, base_address: u64, termination_address_opt: ?u64, text_size: usize) !BytecodePreprocessing {
        const termination_address = termination_address_opt orelse 0x7FFFC008; // Default from MemoryLayout with standard 4KB sizes
        var self = BytecodePreprocessing.init(allocator);
        self.entry_address = base_address;
        errdefer self.deinit();

        // Prepend a single NoOp instruction (as Jolt does)
        try self.bytecode.append(allocator, .{
            .variant = .NoOp,
            .address = 0,
            .operands = .{ .None = {} },
            .virtual_sequence_remaining = null,
            .is_first_in_sequence = false,
            .is_compressed = false,
        });
        try self.raw_words.append(allocator, 0); // NoOp = raw word 0

        // Decode instructions within the .text section only
        const decode_limit = @min(text_size, code_bytes.len);
        var offset: usize = 0;
        while (offset < decode_limit) {
            const addr = base_address + offset;

            // Check if compressed (RVC)
            const first_halfword: u16 = std.mem.readInt(u16, code_bytes[offset..][0..2], .little);
            const is_compressed = (first_halfword & 0x3) != 0x3;

            var instruction: u32 = undefined;
            var instr_size: usize = undefined;

            if (first_halfword == 0) {
                // Zero halfword in a code gap — skip 2 bytes, leave as NoOp padding
                offset += 2;
                continue;
            } else if (is_compressed) {
                // 16-bit compressed instruction - expand it
                const zkvm_instruction = @import("instruction/mod.zig");
                instruction = zkvm_instruction.uncompressInstruction(first_halfword, .Bit64);
                instr_size = 2;
            } else {
                // 32-bit instruction
                if (offset + 4 > code_bytes.len) break;
                instruction = std.mem.readInt(u32, code_bytes[offset..][0..4], .little);
                instr_size = 4;
            }

            // Decode and decompose W-extension instructions into virtual sequences
            const jolt_instr = try instruction_decoder.decodeToJoltInstruction(instruction, addr, is_compressed);
            const bytecode_len_before = self.bytecode.items.len;

            // Check if this is a W-extension instruction that needs decomposition
            switch (jolt_instr.variant) {
                .ADDIW => {
                    // ADDIW → ADDI + VirtualSignExtendWord (2-instruction sequence)
                    // Step 1: ADDI(rd, rs1, imm) with virtual_sequence_remaining=1, is_first_in_sequence=true
                    try self.bytecode.append(allocator, .{
                        .variant = .ADDI,
                        .address = addr,
                        .operands = jolt_instr.operands, // Same FormatI operands
                        .virtual_sequence_remaining = 1,
                        .is_first_in_sequence = true,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 2: VirtualSignExtendWord(rd, rd, 0) with virtual_sequence_remaining=0
                    const rd = switch (jolt_instr.operands) {
                        .FormatI => |i| i.rd,
                        else => 0,
                    };
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualSignExtendWord,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = rd, .rs1 = rd, .imm = 0 } },
                        .virtual_sequence_remaining = 0,
                        .is_first_in_sequence = false,
                        .is_compressed = is_compressed,
                    });
                },
                .ADDW => {
                    // ADDW → ADD + VirtualSignExtendWord (2-instruction sequence)
                    // Step 1: ADD(rd, rs1, rs2)
                    try self.bytecode.append(allocator, .{
                        .variant = .ADD,
                        .address = addr,
                        .operands = jolt_instr.operands, // Same FormatR operands
                        .virtual_sequence_remaining = 1,
                        .is_first_in_sequence = true,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 2: VirtualSignExtendWord(rd, rd, 0)
                    const rd = switch (jolt_instr.operands) {
                        .FormatR => |r| r.rd,
                        else => 0,
                    };
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualSignExtendWord,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = rd, .rs1 = rd, .imm = 0 } },
                        .virtual_sequence_remaining = 0,
                        .is_first_in_sequence = false,
                        .is_compressed = is_compressed,
                    });
                },
                .SUBW => {
                    // SUBW → SUB + VirtualSignExtendWord (2-instruction sequence)
                    try self.bytecode.append(allocator, .{
                        .variant = .SUB,
                        .address = addr,
                        .operands = jolt_instr.operands,
                        .virtual_sequence_remaining = 1,
                        .is_first_in_sequence = true,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    const rd = switch (jolt_instr.operands) {
                        .FormatR => |r| r.rd,
                        else => 0,
                    };
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualSignExtendWord,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = rd, .rs1 = rd, .imm = 0 } },
                        .virtual_sequence_remaining = 0,
                        .is_first_in_sequence = false,
                        .is_compressed = is_compressed,
                    });
                },
                .MULW => {
                    // MULW → MUL + VirtualSignExtendWord (2-instruction sequence)
                    try self.bytecode.append(allocator, .{
                        .variant = .MUL,
                        .address = addr,
                        .operands = jolt_instr.operands,
                        .virtual_sequence_remaining = 1,
                        .is_first_in_sequence = true,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    const rd = switch (jolt_instr.operands) {
                        .FormatR => |r| r.rd,
                        else => 0,
                    };
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualSignExtendWord,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = rd, .rs1 = rd, .imm = 0 } },
                        .virtual_sequence_remaining = 0,
                        .is_first_in_sequence = false,
                        .is_compressed = is_compressed,
                    });
                },
                .SLLI => {
                    // SLLI rd, rs1, imm → VirtualMULI rd, rs1, (1 << imm)
                    // Single virtual instruction - standalone 1-entry virtual sequence.
                    // Jolt's finalize() sets vsr=Some(0) and is_first_in_sequence=true.
                    const shift_amount = switch (jolt_instr.operands) {
                        .FormatI => |i| i.imm,
                        else => 0,
                    };
                    const rd = switch (jolt_instr.operands) {
                        .FormatI => |i| i.rd,
                        else => 0,
                    };
                    const rs1 = switch (jolt_instr.operands) {
                        .FormatI => |i| i.rs1,
                        else => 0,
                    };
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualMULI,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = rd, .rs1 = rs1, .imm = @as(u64, 1) << @intCast(shift_amount) } },
                        .virtual_sequence_remaining = 0,
                        .is_first_in_sequence = true,
                        .is_compressed = is_compressed,
                    });
                },
                .SLLIW => {
                    // SLLIW rd, rs1, imm → VirtualMULI rd, rs1, (1 << imm) + VirtualSignExtendWord(rd, rd, 0)
                    const shift_amount = switch (jolt_instr.operands) {
                        .FormatI => |i| i.imm,
                        else => 0,
                    };
                    const rd = switch (jolt_instr.operands) {
                        .FormatI => |i| i.rd,
                        else => 0,
                    };
                    const rs1 = switch (jolt_instr.operands) {
                        .FormatI => |i| i.rs1,
                        else => 0,
                    };
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualMULI,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = rd, .rs1 = rs1, .imm = @as(u64, 1) << @intCast(shift_amount) } },
                        .virtual_sequence_remaining = 1,
                        .is_first_in_sequence = true,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualSignExtendWord,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = rd, .rs1 = rd, .imm = 0 } },
                        .virtual_sequence_remaining = 0,
                        .is_first_in_sequence = false,
                        .is_compressed = is_compressed,
                    });
                },
                .SRLI => {
                    // SRLI rd, rs1, shamt → VirtualSRLI(rd, rs1, bitmask)
                    // Single virtual instruction - standalone 1-entry virtual sequence.
                    // Jolt's finalize() sets vsr=Some(0) and is_first_in_sequence=true.
                    const raw_imm = switch (jolt_instr.operands) {
                        .FormatI => |i| i.imm,
                        else => 0,
                    };
                    const rd = switch (jolt_instr.operands) {
                        .FormatI => |i| i.rd,
                        else => 0,
                    };
                    const rs1_val = switch (jolt_instr.operands) {
                        .FormatI => |i| i.rs1,
                        else => 0,
                    };
                    const shift: u7 = @intCast(raw_imm & 0x3f);
                    const ones: u128 = (@as(u128, 1) << @intCast(64 - @as(u8, shift))) - 1;
                    const bitmask: u64 = @truncate(ones << shift);
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualSRLI,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = rd, .rs1 = rs1_val, .imm = bitmask } },
                        .virtual_sequence_remaining = 0,
                        .is_first_in_sequence = true,
                        .is_compressed = is_compressed,
                    });
                },
                .SRLIW => {
                    // SRLIW rd, rs1, shamt → 3-step sequence:
                    //   Step 1: SLLI(v_rs1, rs1, 32) → VirtualMULI(v_rs1, rs1, 2^32)
                    //   Step 2: VirtualSRLI(rd, v_rs1, bitmask) where shift = shamt + 32
                    //   Step 3: VirtualSignExtendWord(rd, rd, 0)
                    const raw_imm = switch (jolt_instr.operands) {
                        .FormatI => |i| i.imm,
                        else => 0,
                    };
                    const rd = switch (jolt_instr.operands) {
                        .FormatI => |i| i.rd,
                        else => 0,
                    };
                    const rs1_val = switch (jolt_instr.operands) {
                        .FormatI => |i| i.rs1,
                        else => 0,
                    };
                    // Virtual register for intermediate result (register 32 = first virtual register)
                    const v_rs1: u8 = 32;
                    // Compute bitmask: shift = (shamt & 0x1f) + 32
                    const shamt: u7 = @intCast(raw_imm & 0x1f);
                    const total_shift: u7 = shamt + 32;
                    const ones: u128 = (@as(u128, 1) << @intCast(64 - @as(u8, total_shift))) - 1;
                    const bitmask: u64 = @truncate(ones << total_shift);
                    // Step 1: VirtualMULI(v_rs1, rs1, 2^32) - shift left by 32
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualMULI,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = v_rs1, .rs1 = rs1_val, .imm = @as(u64, 1) << 32 } },
                        .virtual_sequence_remaining = 2,
                        .is_first_in_sequence = true,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 2: VirtualSRLI(rd, v_rs1, bitmask)
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualSRLI,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = rd, .rs1 = v_rs1, .imm = bitmask } },
                        .virtual_sequence_remaining = 1,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 3: VirtualSignExtendWord(rd, rd, 0)
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualSignExtendWord,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = rd, .rs1 = rd, .imm = 0 } },
                        .virtual_sequence_remaining = 0,
                        .is_first_in_sequence = false,
                        .is_compressed = is_compressed,
                    });
                },
                .REMUW => {
                    // REMUW → 12-instruction inline sequence (matching Jolt's decomposition)
                    // Virtual registers: a2=32, a3=33, t0=34, t1=35, t2=36, t3=37, t4=38
                    const rs1 = switch (jolt_instr.operands) {
                        .FormatR => |r| r.rs1,
                        else => 0,
                    };
                    const rs2 = switch (jolt_instr.operands) {
                        .FormatR => |r| r.rs2,
                        else => 0,
                    };
                    const rd = switch (jolt_instr.operands) {
                        .FormatR => |r| r.rd,
                        else => 0,
                    };
                    const a2: u8 = 32;
                    const a3: u8 = 33;
                    const t0: u8 = 34;
                    const t1: u8 = 35;
                    const t2: u8 = 36;
                    const t3: u8 = 37;
                    const t4: u8 = 38;

                    // Step 1: VirtualAdvice(a2) → quotient (vsr=11, first)
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualAdvice,
                        .address = addr,
                        .operands = .{ .FormatJ = .{ .rd = a2, .imm = 0 } },
                        .virtual_sequence_remaining = 11,
                        .is_first_in_sequence = true,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 2: VirtualAdvice(a3) → remainder (vsr=10)
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualAdvice,
                        .address = addr,
                        .operands = .{ .FormatJ = .{ .rd = a3, .imm = 0 } },
                        .virtual_sequence_remaining = 10,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 3: VirtualZeroExtendWord(t3, a2) → zero-extend quotient (vsr=9)
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualZeroExtendWord,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = t3, .rs1 = a2, .imm = 0 } },
                        .virtual_sequence_remaining = 9,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 4: VirtualZeroExtendWord(t1, rs1) → zero-extend dividend (vsr=8)
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualZeroExtendWord,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = t1, .rs1 = rs1, .imm = 0 } },
                        .virtual_sequence_remaining = 8,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 5: VirtualZeroExtendWord(t2, rs2) → zero-extend divisor (vsr=7)
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualZeroExtendWord,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = t2, .rs1 = rs2, .imm = 0 } },
                        .virtual_sequence_remaining = 7,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 6: MUL(t0, t3, t2) → quotient * divisor (vsr=6)
                    try self.bytecode.append(allocator, .{
                        .variant = .MUL,
                        .address = addr,
                        .operands = .{ .FormatR = .{ .rd = t0, .rs1 = t3, .rs2 = t2 } },
                        .virtual_sequence_remaining = 6,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 7: VirtualZeroExtendWord(t4, t0) → mask to 32 bits (vsr=5)
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualZeroExtendWord,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = t4, .rs1 = t0, .imm = 0 } },
                        .virtual_sequence_remaining = 5,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 8: VirtualAssertEQ(t4, t0) → assert no overflow (vsr=4)
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualAssertEQ,
                        .address = addr,
                        .operands = .{ .FormatB = .{ .rs1 = t4, .rs2 = t0, .imm = 0 } },
                        .virtual_sequence_remaining = 4,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 9: ADD(t0, t0, a3) → add remainder (vsr=3)
                    try self.bytecode.append(allocator, .{
                        .variant = .ADD,
                        .address = addr,
                        .operands = .{ .FormatR = .{ .rd = t0, .rs1 = t0, .rs2 = a3 } },
                        .virtual_sequence_remaining = 3,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 10: VirtualAssertEQ(t0, t1) → assert dividend = q*d + r (vsr=2)
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualAssertEQ,
                        .address = addr,
                        .operands = .{ .FormatB = .{ .rs1 = t0, .rs2 = t1, .imm = 0 } },
                        .virtual_sequence_remaining = 2,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 11: VirtualAssertValidUnsignedRemainder(a3, t2) → r < d (vsr=1)
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualAssertValidUnsignedRemainder,
                        .address = addr,
                        .operands = .{ .FormatB = .{ .rs1 = a3, .rs2 = t2, .imm = 0 } },
                        .virtual_sequence_remaining = 1,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 12: VirtualSignExtendWord(rd, a3) → sign-extend result (vsr=0, last)
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualSignExtendWord,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = rd, .rs1 = a3, .imm = 0 } },
                        .virtual_sequence_remaining = 0,
                        .is_first_in_sequence = false,
                        .is_compressed = is_compressed,
                    });
                },
                .REMW, .DIVW => {
                    // REMW/DIVW → 21-instruction inline sequence (matching Jolt's decomposition)
                    // Signed division/remainder verification with overflow handling
                    // Virtual registers: a2=32, a3=33, t0=34, t1=35, t2=36, t3=37, t4=38
                    const rs1 = switch (jolt_instr.operands) {
                        .FormatR => |r| r.rs1,
                        else => 0,
                    };
                    const rs2 = switch (jolt_instr.operands) {
                        .FormatR => |r| r.rs2,
                        else => 0,
                    };
                    const rd = switch (jolt_instr.operands) {
                        .FormatR => |r| r.rd,
                        else => 0,
                    };
                    const a2: u8 = 32; // quotient
                    const a3: u8 = 33; // |remainder|
                    const t0: u8 = 34; // adjusted divisor
                    const t1: u8 = 35; // temporary
                    const t2: u8 = 36; // temporary
                    const t3: u8 = 37; // signed remainder
                    const t4: u8 = 38; // sign-extended dividend

                    // Step 1: VirtualAdvice(a2) → quotient (vsr=20, first)
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualAdvice,
                        .address = addr,
                        .operands = .{ .FormatJ = .{ .rd = a2, .imm = 0 } },
                        .virtual_sequence_remaining = 20,
                        .is_first_in_sequence = true,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 2: VirtualAdvice(a3) → |remainder| (vsr=19)
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualAdvice,
                        .address = addr,
                        .operands = .{ .FormatJ = .{ .rd = a3, .imm = 0 } },
                        .virtual_sequence_remaining = 19,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 3: VirtualSignExtendWord(t4, rs1) → sign-extend dividend (vsr=18)
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualSignExtendWord,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = t4, .rs1 = rs1, .imm = 0 } },
                        .virtual_sequence_remaining = 18,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 4: VirtualSignExtendWord(t3, rs2) → sign-extend divisor (vsr=17)
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualSignExtendWord,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = t3, .rs1 = rs2, .imm = 0 } },
                        .virtual_sequence_remaining = 17,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 5: VirtualAssertValidDiv0(t3, a2) → handle div-by-zero (vsr=16)
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualAssertValidDiv0,
                        .address = addr,
                        .operands = .{ .FormatB = .{ .rs1 = t3, .rs2 = a2, .imm = 0 } },
                        .virtual_sequence_remaining = 16,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 6: VirtualChangeDivisorW(t0, t4, t3) → handle overflow (vsr=15)
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualChangeDivisorW,
                        .address = addr,
                        .operands = .{ .FormatR = .{ .rd = t0, .rs1 = t4, .rs2 = t3 } },
                        .virtual_sequence_remaining = 15,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 7: VirtualSignExtendWord(t1, a2) → sign-extend quotient (vsr=14)
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualSignExtendWord,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = t1, .rs1 = a2, .imm = 0 } },
                        .virtual_sequence_remaining = 14,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 8: VirtualAssertEQ(t1, a2) → assert quotient fits 32 bits (vsr=13)
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualAssertEQ,
                        .address = addr,
                        .operands = .{ .FormatB = .{ .rs1 = t1, .rs2 = a2, .imm = 0 } },
                        .virtual_sequence_remaining = 13,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 9: VirtualSRAI(t2, a3, bitmask_31) → sign bit of |remainder| (vsr=12)
                    // SRAI is expanded to VirtualSRAI with bitmask: shift=31, bitmask = ((1<<33)-1) << 31
                    const srai_bitmask: u64 = blk: {
                        const shift_amt: u7 = 31;
                        const ones: u128 = (@as(u128, 1) << @intCast(64 - @as(u8, shift_amt))) - 1;
                        break :blk @truncate(ones << shift_amt);
                    };
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualSRAI,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = t2, .rs1 = a3, .imm = srai_bitmask } },
                        .virtual_sequence_remaining = 12,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 10: VirtualAssertEQ(t2, 0) → assert |remainder| is non-negative (vsr=11)
                    // Note: rs2=0 means comparing against register x0 (always 0)
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualAssertEQ,
                        .address = addr,
                        .operands = .{ .FormatB = .{ .rs1 = t2, .rs2 = 0, .imm = 0 } },
                        .virtual_sequence_remaining = 11,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 11: VirtualSRAI(t2, t4, bitmask_31) → sign bit of dividend (vsr=10)
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualSRAI,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = t2, .rs1 = t4, .imm = srai_bitmask } },
                        .virtual_sequence_remaining = 10,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 12: XOR(t3, a3, t2) → XOR |remainder| with sign mask (vsr=9)
                    try self.bytecode.append(allocator, .{
                        .variant = .XOR,
                        .address = addr,
                        .operands = .{ .FormatR = .{ .rd = t3, .rs1 = a3, .rs2 = t2 } },
                        .virtual_sequence_remaining = 9,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 13: SUB(t3, t3, t2) → t3 = sign-corrected remainder (vsr=8)
                    try self.bytecode.append(allocator, .{
                        .variant = .SUB,
                        .address = addr,
                        .operands = .{ .FormatR = .{ .rd = t3, .rs1 = t3, .rs2 = t2 } },
                        .virtual_sequence_remaining = 8,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 14: MUL(t1, a2, t0) → quotient × adjusted_divisor (vsr=7)
                    try self.bytecode.append(allocator, .{
                        .variant = .MUL,
                        .address = addr,
                        .operands = .{ .FormatR = .{ .rd = t1, .rs1 = a2, .rs2 = t0 } },
                        .virtual_sequence_remaining = 7,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 15: ADD(t1, t1, t3) → + remainder (vsr=6)
                    try self.bytecode.append(allocator, .{
                        .variant = .ADD,
                        .address = addr,
                        .operands = .{ .FormatR = .{ .rd = t1, .rs1 = t1, .rs2 = t3 } },
                        .virtual_sequence_remaining = 6,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 16: VirtualAssertEQ(t1, t4) → assert dividend = q*d + r (vsr=5)
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualAssertEQ,
                        .address = addr,
                        .operands = .{ .FormatB = .{ .rs1 = t1, .rs2 = t4, .imm = 0 } },
                        .virtual_sequence_remaining = 5,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 17: VirtualSRAI(t2, t0, bitmask_31) → sign bit of adjusted divisor (vsr=4)
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualSRAI,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = t2, .rs1 = t0, .imm = srai_bitmask } },
                        .virtual_sequence_remaining = 4,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 18: XOR(t1, t0, t2) → (vsr=3)
                    try self.bytecode.append(allocator, .{
                        .variant = .XOR,
                        .address = addr,
                        .operands = .{ .FormatR = .{ .rd = t1, .rs1 = t0, .rs2 = t2 } },
                        .virtual_sequence_remaining = 3,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 19: SUB(t1, t1, t2) → t1 = abs(divisor) (vsr=2)
                    try self.bytecode.append(allocator, .{
                        .variant = .SUB,
                        .address = addr,
                        .operands = .{ .FormatR = .{ .rd = t1, .rs1 = t1, .rs2 = t2 } },
                        .virtual_sequence_remaining = 2,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 20: VirtualAssertValidUnsignedRemainder(a3, t1) → |r| < |d| (vsr=1)
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualAssertValidUnsignedRemainder,
                        .address = addr,
                        .operands = .{ .FormatB = .{ .rs1 = a3, .rs2 = t1, .imm = 0 } },
                        .virtual_sequence_remaining = 1,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 21: VirtualSignExtendWord(rd, output) → sign-extend result (vsr=0, last)
                    // REMW: output = t3 (signed remainder), DIVW: output = a2 (quotient)
                    const output_reg = if (jolt_instr.variant == .REMW) t3 else a2;
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualSignExtendWord,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = rd, .rs1 = output_reg, .imm = 0 } },
                        .virtual_sequence_remaining = 0,
                        .is_first_in_sequence = false,
                        .is_compressed = is_compressed,
                    });
                },
                .SLL => {
                    // SLL rd, rs1, rs2 → 2-step: VirtualPow2(v0, rs2, 0) + MUL(rd, rs1, v0)
                    const rd = switch (jolt_instr.operands) {
                        .FormatR => |r| r.rd,
                        else => 0,
                    };
                    const rs1 = switch (jolt_instr.operands) {
                        .FormatR => |r| r.rs1,
                        else => 0,
                    };
                    const rs2 = switch (jolt_instr.operands) {
                        .FormatR => |r| r.rs2,
                        else => 0,
                    };
                    const v0: u8 = 40; // first virtual alloc register
                    // Step 1: VirtualPow2(v0, rs2, 0) — compute 2^(rs2 % 64)
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualPow2,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = v0, .rs1 = rs2, .imm = 0 } },
                        .virtual_sequence_remaining = 1,
                        .is_first_in_sequence = true,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 2: MUL(rd, rs1, v0)
                    try self.bytecode.append(allocator, .{
                        .variant = .MUL,
                        .address = addr,
                        .operands = .{ .FormatR = .{ .rd = rd, .rs1 = rs1, .rs2 = v0 } },
                        .virtual_sequence_remaining = 0,
                        .is_first_in_sequence = false,
                        .is_compressed = is_compressed,
                    });
                },
                .SRL => {
                    // SRL rd, rs1, rs2 → 2-step: VirtualShiftRightBitmask(v0, rs2, 0) + VirtualSRL(rd, rs1, v0)
                    const rd = switch (jolt_instr.operands) {
                        .FormatR => |r| r.rd,
                        else => 0,
                    };
                    const rs1 = switch (jolt_instr.operands) {
                        .FormatR => |r| r.rs1,
                        else => 0,
                    };
                    const rs2 = switch (jolt_instr.operands) {
                        .FormatR => |r| r.rs2,
                        else => 0,
                    };
                    const v0: u8 = 40;
                    // Step 1: VirtualShiftRightBitmask(v0, rs2, 0)
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualShiftRightBitmask,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = v0, .rs1 = rs2, .imm = 0 } },
                        .virtual_sequence_remaining = 1,
                        .is_first_in_sequence = true,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 2: VirtualSRL(rd, rs1, v0)
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualSRL,
                        .address = addr,
                        .operands = .{ .FormatR = .{ .rd = rd, .rs1 = rs1, .rs2 = v0 } },
                        .virtual_sequence_remaining = 0,
                        .is_first_in_sequence = false,
                        .is_compressed = is_compressed,
                    });
                },
                .SRA => {
                    // SRA rd, rs1, rs2 → 2-step: VirtualShiftRightBitmask(v0, rs2, 0) + VirtualSRA(rd, rs1, v0)
                    const rd = switch (jolt_instr.operands) {
                        .FormatR => |r| r.rd,
                        else => 0,
                    };
                    const rs1 = switch (jolt_instr.operands) {
                        .FormatR => |r| r.rs1,
                        else => 0,
                    };
                    const rs2 = switch (jolt_instr.operands) {
                        .FormatR => |r| r.rs2,
                        else => 0,
                    };
                    const v0: u8 = 40;
                    // Step 1: VirtualShiftRightBitmask(v0, rs2, 0)
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualShiftRightBitmask,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = v0, .rs1 = rs2, .imm = 0 } },
                        .virtual_sequence_remaining = 1,
                        .is_first_in_sequence = true,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 2: VirtualSRA(rd, rs1, v0)
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualSRA,
                        .address = addr,
                        .operands = .{ .FormatR = .{ .rd = rd, .rs1 = rs1, .rs2 = v0 } },
                        .virtual_sequence_remaining = 0,
                        .is_first_in_sequence = false,
                        .is_compressed = is_compressed,
                    });
                },
                .SRAI => {
                    // SRAI rd, rs1, shamt → VirtualSRAI(rd, rs1, bitmask)
                    // Single virtual instruction (same pattern as SRLI → VirtualSRLI)
                    const raw_imm = switch (jolt_instr.operands) {
                        .FormatI => |i| i.imm,
                        else => 0,
                    };
                    const rd = switch (jolt_instr.operands) {
                        .FormatI => |i| i.rd,
                        else => 0,
                    };
                    const rs1_val = switch (jolt_instr.operands) {
                        .FormatI => |i| i.rs1,
                        else => 0,
                    };
                    const shift: u7 = @intCast(raw_imm & 0x3f);
                    const ones: u128 = (@as(u128, 1) << @intCast(64 - @as(u8, shift))) - 1;
                    const bitmask: u64 = @truncate(ones << shift);
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualSRAI,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = rd, .rs1 = rs1_val, .imm = bitmask } },
                        .virtual_sequence_remaining = 0,
                        .is_first_in_sequence = true,
                        .is_compressed = is_compressed,
                    });
                },
                .LB, .LBU => {
                    // LB/LBU rd, rs1, imm → 8 flat steps
                    const rd = switch (jolt_instr.operands) {
                        .FormatLoad => |l| l.rd,
                        else => 0,
                    };
                    const rs1 = switch (jolt_instr.operands) {
                        .FormatLoad => |l| l.rs1,
                        else => 0,
                    };
                    const imm = switch (jolt_instr.operands) {
                        .FormatLoad => |l| l.imm,
                        else => 0,
                    };
                    const v0: u8 = 40;
                    const v1: u8 = 41;
                    const v2: u8 = 42; // allocated by inner SLL
                    const total_steps: u16 = 7; // vsr counts from 7 down to 0
                    // Step 1: ADDI(v0, rs1, imm)
                    try self.bytecode.append(allocator, .{
                        .variant = .ADDI,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = v0, .rs1 = rs1, .imm = @bitCast(@as(i64, imm)) } },
                        .virtual_sequence_remaining = total_steps,
                        .is_first_in_sequence = true,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 2: ANDI(v1, v0, -8)
                    try self.bytecode.append(allocator, .{
                        .variant = .ANDI,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = v1, .rs1 = v0, .imm = @bitCast(@as(i64, -8)) } },
                        .virtual_sequence_remaining = total_steps - 1,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 3: LD(v1, v1, 0) — MEMORY READ
                    try self.bytecode.append(allocator, .{
                        .variant = .LD,
                        .address = addr,
                        .operands = .{ .FormatLoad = .{ .rd = v1, .rs1 = v1, .imm = 0 } },
                        .virtual_sequence_remaining = total_steps - 2,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 4: XORI(v0, v0, 7)
                    try self.bytecode.append(allocator, .{
                        .variant = .XORI,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = v0, .rs1 = v0, .imm = 7 } },
                        .virtual_sequence_remaining = total_steps - 3,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 5: VirtualMULI(v0, v0, 8) — from SLLI v0, v0, 3
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualMULI,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = v0, .rs1 = v0, .imm = 8 } },
                        .virtual_sequence_remaining = total_steps - 4,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 6: VirtualPow2(v2, v0, 0) — from SLL expansion step 1
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualPow2,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = v2, .rs1 = v0, .imm = 0 } },
                        .virtual_sequence_remaining = total_steps - 5,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 7: MUL(v1, v1, v2) — from SLL expansion step 2
                    try self.bytecode.append(allocator, .{
                        .variant = .MUL,
                        .address = addr,
                        .operands = .{ .FormatR = .{ .rd = v1, .rs1 = v1, .rs2 = v2 } },
                        .virtual_sequence_remaining = total_steps - 6,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 8: VirtualSRAI/VirtualSRLI(rd, v1, bitmask_56)
                    const shift_56: u7 = 56;
                    const ones_56: u128 = (@as(u128, 1) << @intCast(64 - @as(u8, shift_56))) - 1;
                    const bitmask_56: u64 = @truncate(ones_56 << shift_56);
                    if (jolt_instr.variant == .LB) {
                        try self.bytecode.append(allocator, .{
                            .variant = .VirtualSRAI,
                            .address = addr,
                            .operands = .{ .FormatI = .{ .rd = rd, .rs1 = v1, .imm = bitmask_56 } },
                            .virtual_sequence_remaining = 0,
                            .is_first_in_sequence = false,
                            .is_compressed = is_compressed,
                        });
                    } else {
                        // LBU: logical right shift (zero-extend)
                        try self.bytecode.append(allocator, .{
                            .variant = .VirtualSRLI,
                            .address = addr,
                            .operands = .{ .FormatI = .{ .rd = rd, .rs1 = v1, .imm = bitmask_56 } },
                            .virtual_sequence_remaining = 0,
                            .is_first_in_sequence = false,
                            .is_compressed = is_compressed,
                        });
                    }
                },
                .LH, .LHU => {
                    // LH/LHU rd, rs1, imm → 9 flat steps (includes alignment assertion)
                    const rd = switch (jolt_instr.operands) {
                        .FormatLoad => |l| l.rd,
                        else => 0,
                    };
                    const rs1 = switch (jolt_instr.operands) {
                        .FormatLoad => |l| l.rs1,
                        else => 0,
                    };
                    const imm = switch (jolt_instr.operands) {
                        .FormatLoad => |l| l.imm,
                        else => 0,
                    };
                    const v0: u8 = 40;
                    const v1: u8 = 41;
                    const v2: u8 = 42;
                    const total_steps: u16 = 8; // vsr counts from 8 down to 0
                    // Step 1: VirtualAssertHalfwordAlignment(rs1, imm)
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualAssertHalfwordAlignment,
                        .address = addr,
                        .operands = .{ .FormatAssert = .{ .rs1 = rs1, .imm = @as(i64, imm) } },
                        .virtual_sequence_remaining = total_steps,
                        .is_first_in_sequence = true,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 2: ADDI(v0, rs1, imm)
                    try self.bytecode.append(allocator, .{
                        .variant = .ADDI,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = v0, .rs1 = rs1, .imm = @bitCast(@as(i64, imm)) } },
                        .virtual_sequence_remaining = total_steps - 1,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 3: ANDI(v1, v0, -8)
                    try self.bytecode.append(allocator, .{
                        .variant = .ANDI,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = v1, .rs1 = v0, .imm = @bitCast(@as(i64, -8)) } },
                        .virtual_sequence_remaining = total_steps - 2,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 4: LD(v1, v1, 0) — MEMORY READ
                    try self.bytecode.append(allocator, .{
                        .variant = .LD,
                        .address = addr,
                        .operands = .{ .FormatLoad = .{ .rd = v1, .rs1 = v1, .imm = 0 } },
                        .virtual_sequence_remaining = total_steps - 3,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 5: XORI(v0, v0, 6)
                    try self.bytecode.append(allocator, .{
                        .variant = .XORI,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = v0, .rs1 = v0, .imm = 6 } },
                        .virtual_sequence_remaining = total_steps - 4,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 6: VirtualMULI(v0, v0, 8) — from SLLI v0, v0, 3
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualMULI,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = v0, .rs1 = v0, .imm = 8 } },
                        .virtual_sequence_remaining = total_steps - 5,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 7: VirtualPow2(v2, v0, 0) — from SLL step 1
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualPow2,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = v2, .rs1 = v0, .imm = 0 } },
                        .virtual_sequence_remaining = total_steps - 6,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 8: MUL(v1, v1, v2) — from SLL step 2
                    try self.bytecode.append(allocator, .{
                        .variant = .MUL,
                        .address = addr,
                        .operands = .{ .FormatR = .{ .rd = v1, .rs1 = v1, .rs2 = v2 } },
                        .virtual_sequence_remaining = total_steps - 7,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 9: VirtualSRAI/VirtualSRLI(rd, v1, bitmask_48)
                    const shift_48: u7 = 48;
                    const ones_48: u128 = (@as(u128, 1) << @intCast(64 - @as(u8, shift_48))) - 1;
                    const bitmask_48: u64 = @truncate(ones_48 << shift_48);
                    if (jolt_instr.variant == .LH) {
                        try self.bytecode.append(allocator, .{
                            .variant = .VirtualSRAI,
                            .address = addr,
                            .operands = .{ .FormatI = .{ .rd = rd, .rs1 = v1, .imm = bitmask_48 } },
                            .virtual_sequence_remaining = 0,
                            .is_first_in_sequence = false,
                            .is_compressed = is_compressed,
                        });
                    } else {
                        try self.bytecode.append(allocator, .{
                            .variant = .VirtualSRLI,
                            .address = addr,
                            .operands = .{ .FormatI = .{ .rd = rd, .rs1 = v1, .imm = bitmask_48 } },
                            .virtual_sequence_remaining = 0,
                            .is_first_in_sequence = false,
                            .is_compressed = is_compressed,
                        });
                    }
                },
                .LW => {
                    // LW rd, rs1, imm → 8 flat steps (RV64: with alignment assert, SRL, sign-extend)
                    const rd = switch (jolt_instr.operands) {
                        .FormatLoad => |l| l.rd,
                        else => 0,
                    };
                    const rs1 = switch (jolt_instr.operands) {
                        .FormatLoad => |l| l.rs1,
                        else => 0,
                    };
                    const imm = switch (jolt_instr.operands) {
                        .FormatLoad => |l| l.imm,
                        else => 0,
                    };
                    const v0: u8 = 40;
                    const v1: u8 = 41;
                    const v2: u8 = 42; // allocated by inner SRL
                    const total_steps: u16 = 7;
                    // Step 1: VirtualAssertWordAlignment(rs1, imm)
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualAssertWordAlignment,
                        .address = addr,
                        .operands = .{ .FormatAssert = .{ .rs1 = rs1, .imm = @as(i64, imm) } },
                        .virtual_sequence_remaining = total_steps,
                        .is_first_in_sequence = true,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 2: ADDI(v0, rs1, imm)
                    try self.bytecode.append(allocator, .{
                        .variant = .ADDI,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = v0, .rs1 = rs1, .imm = @bitCast(@as(i64, imm)) } },
                        .virtual_sequence_remaining = total_steps - 1,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 3: ANDI(v1, v0, -8)
                    try self.bytecode.append(allocator, .{
                        .variant = .ANDI,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = v1, .rs1 = v0, .imm = @bitCast(@as(i64, -8)) } },
                        .virtual_sequence_remaining = total_steps - 2,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 4: LD(v1, v1, 0)
                    try self.bytecode.append(allocator, .{
                        .variant = .LD,
                        .address = addr,
                        .operands = .{ .FormatLoad = .{ .rd = v1, .rs1 = v1, .imm = 0 } },
                        .virtual_sequence_remaining = total_steps - 3,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 5: VirtualMULI(v0, v0, 8) — from SLLI v0, v0, 3 (NO XORI for LW)
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualMULI,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = v0, .rs1 = v0, .imm = 8 } },
                        .virtual_sequence_remaining = total_steps - 4,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 6: VirtualShiftRightBitmask(v2, v0, 0) — from SRL step 1
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualShiftRightBitmask,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = v2, .rs1 = v0, .imm = 0 } },
                        .virtual_sequence_remaining = total_steps - 5,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 7: VirtualSRL(v1, v1, v2) — from SRL step 2
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualSRL,
                        .address = addr,
                        .operands = .{ .FormatR = .{ .rd = v1, .rs1 = v1, .rs2 = v2 } },
                        .virtual_sequence_remaining = total_steps - 6,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 8: VirtualSignExtendWord(rd, v1, 0)
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualSignExtendWord,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = rd, .rs1 = v1, .imm = 0 } },
                        .virtual_sequence_remaining = 0,
                        .is_first_in_sequence = false,
                        .is_compressed = is_compressed,
                    });
                },
                .LWU => {
                    // LWU rd, rs1, imm → 9 flat steps (with XORI, SLL, SRLI)
                    const rd = switch (jolt_instr.operands) {
                        .FormatLoad => |l| l.rd,
                        else => 0,
                    };
                    const rs1 = switch (jolt_instr.operands) {
                        .FormatLoad => |l| l.rs1,
                        else => 0,
                    };
                    const imm = switch (jolt_instr.operands) {
                        .FormatLoad => |l| l.imm,
                        else => 0,
                    };
                    const v0: u8 = 40;
                    const v1: u8 = 41;
                    const v2: u8 = 42; // from inner SLL
                    const total_steps: u16 = 8;
                    // Step 1: VirtualAssertWordAlignment(rs1, imm)
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualAssertWordAlignment,
                        .address = addr,
                        .operands = .{ .FormatAssert = .{ .rs1 = rs1, .imm = @as(i64, imm) } },
                        .virtual_sequence_remaining = total_steps,
                        .is_first_in_sequence = true,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 2: ADDI(v0, rs1, imm)
                    try self.bytecode.append(allocator, .{
                        .variant = .ADDI,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = v0, .rs1 = rs1, .imm = @bitCast(@as(i64, imm)) } },
                        .virtual_sequence_remaining = total_steps - 1,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 3: ANDI(v1, v0, -8)
                    try self.bytecode.append(allocator, .{
                        .variant = .ANDI,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = v1, .rs1 = v0, .imm = @bitCast(@as(i64, -8)) } },
                        .virtual_sequence_remaining = total_steps - 2,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 4: LD(v1, v1, 0)
                    try self.bytecode.append(allocator, .{
                        .variant = .LD,
                        .address = addr,
                        .operands = .{ .FormatLoad = .{ .rd = v1, .rs1 = v1, .imm = 0 } },
                        .virtual_sequence_remaining = total_steps - 3,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 5: XORI(v0, v0, 4)
                    try self.bytecode.append(allocator, .{
                        .variant = .XORI,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = v0, .rs1 = v0, .imm = 4 } },
                        .virtual_sequence_remaining = total_steps - 4,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 6: VirtualMULI(v0, v0, 8) — from SLLI
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualMULI,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = v0, .rs1 = v0, .imm = 8 } },
                        .virtual_sequence_remaining = total_steps - 5,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 7: VirtualPow2(v2, v0, 0) — from SLL step 1
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualPow2,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = v2, .rs1 = v0, .imm = 0 } },
                        .virtual_sequence_remaining = total_steps - 6,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 8: MUL(v1, v1, v2) — from SLL step 2
                    try self.bytecode.append(allocator, .{
                        .variant = .MUL,
                        .address = addr,
                        .operands = .{ .FormatR = .{ .rd = v1, .rs1 = v1, .rs2 = v2 } },
                        .virtual_sequence_remaining = total_steps - 7,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 9: VirtualSRLI(rd, v1, bitmask_32)
                    const shift_32: u7 = 32;
                    const ones_32: u128 = (@as(u128, 1) << @intCast(64 - @as(u8, shift_32))) - 1;
                    const bitmask_32: u64 = @truncate(ones_32 << shift_32);
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualSRLI,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = rd, .rs1 = v1, .imm = bitmask_32 } },
                        .virtual_sequence_remaining = 0,
                        .is_first_in_sequence = false,
                        .is_compressed = is_compressed,
                    });
                },
                .SB => {
                    // SB rs2, rs1, imm → 13 flat steps
                    const rs1 = switch (jolt_instr.operands) {
                        .FormatS => |s| s.rs1,
                        else => 0,
                    };
                    const rs2 = switch (jolt_instr.operands) {
                        .FormatS => |s| s.rs2,
                        else => 0,
                    };
                    const imm = switch (jolt_instr.operands) {
                        .FormatS => |s| s.imm,
                        else => 0,
                    };
                    const v0: u8 = 40;
                    const v1: u8 = 41;
                    const v2: u8 = 42;
                    const v3: u8 = 43;
                    const v4: u8 = 44; // from inner SLL #1
                    const v5: u8 = 45; // from inner SLL #2
                    const total_steps: u16 = 12;
                    // Step 1: ADDI(v0, rs1, imm)
                    try self.bytecode.append(allocator, .{
                        .variant = .ADDI,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = v0, .rs1 = rs1, .imm = @bitCast(@as(i64, imm)) } },
                        .virtual_sequence_remaining = total_steps,
                        .is_first_in_sequence = true,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 2: ANDI(v1, v0, -8)
                    try self.bytecode.append(allocator, .{
                        .variant = .ANDI,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = v1, .rs1 = v0, .imm = @bitCast(@as(i64, -8)) } },
                        .virtual_sequence_remaining = total_steps - 1,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 3: LD(v2, v1, 0) — MEMORY READ
                    try self.bytecode.append(allocator, .{
                        .variant = .LD,
                        .address = addr,
                        .operands = .{ .FormatLoad = .{ .rd = v2, .rs1 = v1, .imm = 0 } },
                        .virtual_sequence_remaining = total_steps - 2,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 4: VirtualMULI(v3, v0, 8) — from SLLI v3, v0, 3
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualMULI,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = v3, .rs1 = v0, .imm = 8 } },
                        .virtual_sequence_remaining = total_steps - 3,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 5: LUI(v0, 0xff)
                    try self.bytecode.append(allocator, .{
                        .variant = .LUI,
                        .address = addr,
                        .operands = .{ .FormatU = .{ .rd = v0, .imm = 0xff << 12 } },
                        .virtual_sequence_remaining = total_steps - 4,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 6: VirtualPow2(v4, v3, 0) — from SLL(v0, v0, v3) step 1
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualPow2,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = v4, .rs1 = v3, .imm = 0 } },
                        .virtual_sequence_remaining = total_steps - 5,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 7: MUL(v0, v0, v4) — from SLL step 2
                    try self.bytecode.append(allocator, .{
                        .variant = .MUL,
                        .address = addr,
                        .operands = .{ .FormatR = .{ .rd = v0, .rs1 = v0, .rs2 = v4 } },
                        .virtual_sequence_remaining = total_steps - 6,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 8: VirtualPow2(v5, v3, 0) — from SLL(v3, rs2, v3) step 1
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualPow2,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = v5, .rs1 = v3, .imm = 0 } },
                        .virtual_sequence_remaining = total_steps - 7,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 9: MUL(v3, rs2, v5) — from SLL step 2
                    try self.bytecode.append(allocator, .{
                        .variant = .MUL,
                        .address = addr,
                        .operands = .{ .FormatR = .{ .rd = v3, .rs1 = rs2, .rs2 = v5 } },
                        .virtual_sequence_remaining = total_steps - 8,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 10: XOR(v3, v2, v3)
                    try self.bytecode.append(allocator, .{
                        .variant = .XOR,
                        .address = addr,
                        .operands = .{ .FormatR = .{ .rd = v3, .rs1 = v2, .rs2 = v3 } },
                        .virtual_sequence_remaining = total_steps - 9,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 11: AND(v3, v3, v0)
                    try self.bytecode.append(allocator, .{
                        .variant = .AND,
                        .address = addr,
                        .operands = .{ .FormatR = .{ .rd = v3, .rs1 = v3, .rs2 = v0 } },
                        .virtual_sequence_remaining = total_steps - 10,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 12: XOR(v2, v2, v3)
                    try self.bytecode.append(allocator, .{
                        .variant = .XOR,
                        .address = addr,
                        .operands = .{ .FormatR = .{ .rd = v2, .rs1 = v2, .rs2 = v3 } },
                        .virtual_sequence_remaining = total_steps - 11,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 13: SD(v1, v2, 0) — MEMORY WRITE
                    try self.bytecode.append(allocator, .{
                        .variant = .SD,
                        .address = addr,
                        .operands = .{ .FormatS = .{ .rs1 = v1, .rs2 = v2, .imm = 0 } },
                        .virtual_sequence_remaining = 0,
                        .is_first_in_sequence = false,
                        .is_compressed = is_compressed,
                    });
                },
                .SH => {
                    // SH rs2, rs1, imm → 14 flat steps (SB + alignment assertion + 0xffff mask)
                    const rs1 = switch (jolt_instr.operands) {
                        .FormatS => |s| s.rs1,
                        else => 0,
                    };
                    const rs2 = switch (jolt_instr.operands) {
                        .FormatS => |s| s.rs2,
                        else => 0,
                    };
                    const imm = switch (jolt_instr.operands) {
                        .FormatS => |s| s.imm,
                        else => 0,
                    };
                    const v0: u8 = 40;
                    const v1: u8 = 41;
                    const v2: u8 = 42;
                    const v3: u8 = 43;
                    const v4: u8 = 44;
                    const v5: u8 = 45;
                    const total_steps: u16 = 13;
                    // Step 1: VirtualAssertHalfwordAlignment(rs1, imm)
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualAssertHalfwordAlignment,
                        .address = addr,
                        .operands = .{ .FormatAssert = .{ .rs1 = rs1, .imm = @as(i64, imm) } },
                        .virtual_sequence_remaining = total_steps,
                        .is_first_in_sequence = true,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 2: ADDI(v0, rs1, imm)
                    try self.bytecode.append(allocator, .{
                        .variant = .ADDI,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = v0, .rs1 = rs1, .imm = @bitCast(@as(i64, imm)) } },
                        .virtual_sequence_remaining = total_steps - 1,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 3: ANDI(v1, v0, -8)
                    try self.bytecode.append(allocator, .{
                        .variant = .ANDI,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = v1, .rs1 = v0, .imm = @bitCast(@as(i64, -8)) } },
                        .virtual_sequence_remaining = total_steps - 2,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 4: LD(v2, v1, 0) — MEMORY READ
                    try self.bytecode.append(allocator, .{
                        .variant = .LD,
                        .address = addr,
                        .operands = .{ .FormatLoad = .{ .rd = v2, .rs1 = v1, .imm = 0 } },
                        .virtual_sequence_remaining = total_steps - 3,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 5: VirtualMULI(v3, v0, 8) — from SLLI
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualMULI,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = v3, .rs1 = v0, .imm = 8 } },
                        .virtual_sequence_remaining = total_steps - 4,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 6: LUI(v0, 0xffff)
                    try self.bytecode.append(allocator, .{
                        .variant = .LUI,
                        .address = addr,
                        .operands = .{ .FormatU = .{ .rd = v0, .imm = 0xffff << 12 } },
                        .virtual_sequence_remaining = total_steps - 5,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 7: VirtualPow2(v4, v3, 0) — from SLL(v0, v0, v3)
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualPow2,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = v4, .rs1 = v3, .imm = 0 } },
                        .virtual_sequence_remaining = total_steps - 6,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 8: MUL(v0, v0, v4)
                    try self.bytecode.append(allocator, .{
                        .variant = .MUL,
                        .address = addr,
                        .operands = .{ .FormatR = .{ .rd = v0, .rs1 = v0, .rs2 = v4 } },
                        .virtual_sequence_remaining = total_steps - 7,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 9: VirtualPow2(v5, v3, 0) — from SLL(v3, rs2, v3)
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualPow2,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = v5, .rs1 = v3, .imm = 0 } },
                        .virtual_sequence_remaining = total_steps - 8,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 10: MUL(v3, rs2, v5)
                    try self.bytecode.append(allocator, .{
                        .variant = .MUL,
                        .address = addr,
                        .operands = .{ .FormatR = .{ .rd = v3, .rs1 = rs2, .rs2 = v5 } },
                        .virtual_sequence_remaining = total_steps - 9,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 11: XOR(v3, v2, v3)
                    try self.bytecode.append(allocator, .{
                        .variant = .XOR,
                        .address = addr,
                        .operands = .{ .FormatR = .{ .rd = v3, .rs1 = v2, .rs2 = v3 } },
                        .virtual_sequence_remaining = total_steps - 10,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 12: AND(v3, v3, v0)
                    try self.bytecode.append(allocator, .{
                        .variant = .AND,
                        .address = addr,
                        .operands = .{ .FormatR = .{ .rd = v3, .rs1 = v3, .rs2 = v0 } },
                        .virtual_sequence_remaining = total_steps - 11,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 13: XOR(v2, v2, v3)
                    try self.bytecode.append(allocator, .{
                        .variant = .XOR,
                        .address = addr,
                        .operands = .{ .FormatR = .{ .rd = v2, .rs1 = v2, .rs2 = v3 } },
                        .virtual_sequence_remaining = total_steps - 12,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 14: SD(v1, v2, 0) — MEMORY WRITE
                    try self.bytecode.append(allocator, .{
                        .variant = .SD,
                        .address = addr,
                        .operands = .{ .FormatS = .{ .rs1 = v1, .rs2 = v2, .imm = 0 } },
                        .virtual_sequence_remaining = 0,
                        .is_first_in_sequence = false,
                        .is_compressed = is_compressed,
                    });
                },
                .SW => {
                    // SW rs2, rs1, imm → 15 flat steps (RV64)
                    const rs1 = switch (jolt_instr.operands) {
                        .FormatS => |s| s.rs1,
                        else => 0,
                    };
                    const rs2 = switch (jolt_instr.operands) {
                        .FormatS => |s| s.rs2,
                        else => 0,
                    };
                    const imm = switch (jolt_instr.operands) {
                        .FormatS => |s| s.imm,
                        else => 0,
                    };
                    const v0: u8 = 40;
                    const v1: u8 = 41;
                    const v2: u8 = 42;
                    const v3: u8 = 43;
                    const v4: u8 = 44; // from inner SLL #1
                    const v5: u8 = 45; // from inner SLL #2
                    const total_steps: u16 = 14;
                    // Step 1: VirtualAssertWordAlignment(rs1, imm)
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualAssertWordAlignment,
                        .address = addr,
                        .operands = .{ .FormatAssert = .{ .rs1 = rs1, .imm = @as(i64, imm) } },
                        .virtual_sequence_remaining = total_steps,
                        .is_first_in_sequence = true,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 2: ADDI(v0, rs1, imm)
                    try self.bytecode.append(allocator, .{
                        .variant = .ADDI,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = v0, .rs1 = rs1, .imm = @bitCast(@as(i64, imm)) } },
                        .virtual_sequence_remaining = total_steps - 1,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 3: ANDI(v1, v0, -8)
                    try self.bytecode.append(allocator, .{
                        .variant = .ANDI,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = v1, .rs1 = v0, .imm = @bitCast(@as(i64, -8)) } },
                        .virtual_sequence_remaining = total_steps - 2,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 4: LD(v2, v1, 0) — MEMORY READ
                    try self.bytecode.append(allocator, .{
                        .variant = .LD,
                        .address = addr,
                        .operands = .{ .FormatLoad = .{ .rd = v2, .rs1 = v1, .imm = 0 } },
                        .virtual_sequence_remaining = total_steps - 3,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 5: VirtualMULI(v0, v0, 8) — from SLLI v0, v0, 3
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualMULI,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = v0, .rs1 = v0, .imm = 8 } },
                        .virtual_sequence_remaining = total_steps - 4,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 6: ORI(v3, x0, -1) — all 1s
                    try self.bytecode.append(allocator, .{
                        .variant = .ORI,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = v3, .rs1 = 0, .imm = @bitCast(@as(i64, -1)) } },
                        .virtual_sequence_remaining = total_steps - 5,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 7: VirtualSRLI(v3, v3, bitmask_32) — 32-bit mask
                    const shift_32: u7 = 32;
                    const ones_32: u128 = (@as(u128, 1) << @intCast(64 - @as(u8, shift_32))) - 1;
                    const bitmask_32: u64 = @truncate(ones_32 << shift_32);
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualSRLI,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = v3, .rs1 = v3, .imm = bitmask_32 } },
                        .virtual_sequence_remaining = total_steps - 6,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 8: VirtualPow2(v4, v0, 0) — from SLL(v3, v3, v0) step 1
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualPow2,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = v4, .rs1 = v0, .imm = 0 } },
                        .virtual_sequence_remaining = total_steps - 7,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 9: MUL(v3, v3, v4) — shifted 32-bit mask
                    try self.bytecode.append(allocator, .{
                        .variant = .MUL,
                        .address = addr,
                        .operands = .{ .FormatR = .{ .rd = v3, .rs1 = v3, .rs2 = v4 } },
                        .virtual_sequence_remaining = total_steps - 8,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 10: VirtualPow2(v5, v0, 0) — from SLL(v0, rs2, v0) step 1
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualPow2,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = v5, .rs1 = v0, .imm = 0 } },
                        .virtual_sequence_remaining = total_steps - 9,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 11: MUL(v0, rs2, v5) — shifted value
                    try self.bytecode.append(allocator, .{
                        .variant = .MUL,
                        .address = addr,
                        .operands = .{ .FormatR = .{ .rd = v0, .rs1 = rs2, .rs2 = v5 } },
                        .virtual_sequence_remaining = total_steps - 10,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 12: XOR(v0, v2, v0)
                    try self.bytecode.append(allocator, .{
                        .variant = .XOR,
                        .address = addr,
                        .operands = .{ .FormatR = .{ .rd = v0, .rs1 = v2, .rs2 = v0 } },
                        .virtual_sequence_remaining = total_steps - 11,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 13: AND(v0, v0, v3)
                    try self.bytecode.append(allocator, .{
                        .variant = .AND,
                        .address = addr,
                        .operands = .{ .FormatR = .{ .rd = v0, .rs1 = v0, .rs2 = v3 } },
                        .virtual_sequence_remaining = total_steps - 12,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 14: XOR(v2, v2, v0)
                    try self.bytecode.append(allocator, .{
                        .variant = .XOR,
                        .address = addr,
                        .operands = .{ .FormatR = .{ .rd = v2, .rs1 = v2, .rs2 = v0 } },
                        .virtual_sequence_remaining = total_steps - 13,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 15: SD(v1, v2, 0) — MEMORY WRITE
                    try self.bytecode.append(allocator, .{
                        .variant = .SD,
                        .address = addr,
                        .operands = .{ .FormatS = .{ .rs1 = v1, .rs2 = v2, .imm = 0 } },
                        .virtual_sequence_remaining = 0,
                        .is_first_in_sequence = false,
                        .is_compressed = is_compressed,
                    });
                },
                else => {
                    // Non-decomposed instructions: append as-is
                    try self.bytecode.append(allocator, jolt_instr);
                },
            }

            // Track raw 32-bit instruction word for each bytecode entry added.
            // All entries from a single ELF instruction share the same raw word.
            const entries_added = self.bytecode.items.len - bytecode_len_before;
            for (0..entries_added) |_| {
                try self.raw_words.append(allocator, instruction);
            }

            offset += instr_size;
        }

        // Build the PC map BEFORE adding termination entries.
        // This ensures termination_base_pc = last_real_instruction_pc + 1,
        // which matches the array index where termination entries will be appended.
        // If we build AFTER, the JAL (address=4) would be counted as a real
        // instruction, making termination_base_pc off by 1.
        try self.pc_map.build(self.bytecode.items);

        // Add termination sequence (LUI + ADDI + SB + JAL) = 4 entries.
        // These must be in the bytecode array BEFORE power-of-2 padding so that
        // code_size accounts for them. This matches computeBytecodeCodeSize which
        // also adds +4 for termination.
        //
        // The termination stores write a sentinel value to the termination address
        // to signal program completion, followed by a JAL-to-self that matches
        // vanilla Jolt's `j .` in _start.
        //
        // LUI x31, upper20(term_addr) - load upper bits of termination address
        // ADDI x30, x0, 1 - load value 1
        // SB x30, lower12(term_addr)(x31) - store value 1 to termination address
        // JAL x0, 0 - infinite loop (j .)
        {
            // Use address=0 for all termination entries (they are virtual, not real ELF instructions)
            // Use the memory layout's termination I/O address (NOT base_address + code_size)
            // to match the prover's bytecode table.
            const term_addr = termination_address;
            const upper20: u32 = @truncate((term_addr >> 12) & 0xFFFFF);
            const lower12: u32 = @truncate(term_addr & 0xFFF);
            const imm_upper7: u32 = (lower12 >> 5) & 0x7F;
            const imm_lower5: u32 = lower12 & 0x1F;

            // Compute instruction words (for raw_words export)
            const lui_word: u32 = (upper20 << 12) | (31 << 7) | 0x37;
            const addi_word: u32 = (1 << 20) | (0 << 15) | (0 << 12) | (30 << 7) | 0x13;
            const sb_word: u32 = (imm_upper7 << 25) | (30 << 20) | (31 << 15) | (0 << 12) | (imm_lower5 << 7) | 0x23;

            // LUI x31 (virtual, vsr=2)
            try self.bytecode.append(allocator, .{
                .variant = .LUI,
                .address = 0,
                .operands = .{ .FormatU = .{ .rd = 31, .imm = @as(u64, upper20) << 12 } },
                .virtual_sequence_remaining = 2,
                .is_first_in_sequence = false,
                .is_compressed = false,
            });
            try self.raw_words.append(allocator, lui_word);

            // ADDI x30, x0, 1 (virtual, vsr=1)
            try self.bytecode.append(allocator, .{
                .variant = .ADDI,
                .address = 0,
                .operands = .{ .FormatI = .{ .rd = 30, .rs1 = 0, .imm = 1 } },
                .virtual_sequence_remaining = 1,
                .is_first_in_sequence = false,
                .is_compressed = false,
            });
            try self.raw_words.append(allocator, addi_word);

            // SD x30, lower12(x31) (anchor, vsr=Some(0))
            // NOTE: We use SD instead of SB because SB is not in Jolt's
            // define_rv32im_trait_impls! macro (circuit_flags panics on SB).
            // SD has the Store flag and is a valid Jolt instruction.
            // The raw_word still encodes the original SB for raw word matching.
            // vsr=Some(0) matches Jolt: VirtualInstruction=true, DoNotUpdatePC=false
            try self.bytecode.append(allocator, .{
                .variant = .SD,
                .address = 0,
                .operands = .{ .FormatS = .{ .rs1 = 31, .rs2 = 30, .imm = @as(i64, @intCast(lower12)) } },
                .virtual_sequence_remaining = 0, // Some(0): last in virtual sequence
                .is_first_in_sequence = false,
                .is_compressed = false,
            });
            try self.raw_words.append(allocator, sb_word);

            // JAL x0, 0 (j . = infinite loop, vsr=None)
            // Matches vanilla Jolt's `j .` in _start after main returns.
            // address=4 (synthetic) provides UPC=4 for SB's constraint 16.
            // vsr=null means VirtualInstruction=false, DoNotUpdatePC=false.
            // Jump=1 disables constraint 16 for JAL→NoOp transition.
            // rd remapped to vr40 (upstream Jolt remaps JAL x0 to virtual register).
            const jal_word: u32 = 0x0000006F;
            try self.bytecode.append(allocator, .{
                .variant = .JAL,
                .address = 4, // Synthetic address: UPC=4 satisfies SB's NextUPC constraint
                .operands = .{ .FormatJ = .{ .rd = 40, .imm = 0 } },
                .virtual_sequence_remaining = null, // Not a virtual sequence
                .is_first_in_sequence = false,
                .is_compressed = false,
            });
            try self.raw_words.append(allocator, jal_word);
        }

        // Pad to next power of 2
        var size = self.bytecode.items.len;
        if (size < 2) size = 2;
        size = std.math.ceilPowerOfTwo(usize, size) catch size;
        self.code_size = size;

        // Pad with NoOps
        while (self.bytecode.items.len < size) {
            try self.bytecode.append(allocator, .{
                .variant = .NoOp,
                .address = 0,
                .operands = .{ .None = {} },
                .virtual_sequence_remaining = null,
                .is_first_in_sequence = false,
                .is_compressed = false,
            });
            try self.raw_words.append(allocator, 0); // NoOp padding = raw word 0
        }

        return self;
    }

    /// Serialize to arkworks format
    pub fn serialize(self: *const BytecodePreprocessing, allocator: Allocator, writer: anytype) !void {
        // code_size as usize (u64)
        try writer.writeInt(u64, @intCast(self.code_size), .little);

        // bytecode: Vec<Instruction>
        // Each instruction is serialized as: u64 length + JSON bytes
        try writer.writeInt(u64, @intCast(self.bytecode.items.len), .little);

        for (self.bytecode.items) |instr| {
            const json = try instr.toJson(allocator);
            defer allocator.free(json);

            try writer.writeInt(u64, @intCast(json.len), .little);
            try writer.writeAll(json);
        }

        // pc_map
        try self.pc_map.serialize(writer);

        // entry_address (u64, added in upstream PR #1335)
        try writer.writeInt(u64, self.entry_address, .little);
    }
};

/// RAMPreprocessing - initial memory state
pub const RAMPreprocessing = struct {
    /// Minimum bytecode address
    min_bytecode_address: u64,
    /// Memory words (8-byte aligned)
    bytecode_words: std.ArrayListUnmanaged(u64),
    allocator: Allocator,

    pub fn init(allocator: Allocator) RAMPreprocessing {
        return .{
            .min_bytecode_address = 0,
            .bytecode_words = .{},
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *RAMPreprocessing) void {
        self.bytecode_words.deinit(self.allocator);
    }

    /// Create from memory initialization data
    pub fn preprocess(allocator: Allocator, memory_init: []const struct { u64, u8 }) !RAMPreprocessing {
        var self = RAMPreprocessing.init(allocator);
        errdefer self.deinit();

        if (memory_init.len == 0) {
            return self;
        }

        // Find min/max addresses
        var min_addr: u64 = memory_init[0][0];
        var max_addr: u64 = memory_init[0][0];
        for (memory_init) |entry| {
            min_addr = @min(min_addr, entry[0]);
            max_addr = @max(max_addr, entry[0]);
        }

        // Account for instruction bytes (4 bytes per instruction)
        max_addr += 3;

        // Calculate word range
        const min_word = min_addr / 8;
        const max_word = (max_addr + 7) / 8;
        const num_words = max_word - min_word + 1;

        self.min_bytecode_address = min_word * 8;

        // Allocate and zero words
        try self.bytecode_words.resize(allocator, num_words);
        @memset(self.bytecode_words.items, 0);

        // Fill in bytes
        for (memory_init) |entry| {
            const addr = entry[0];
            const byte = entry[1];
            const word_idx = (addr / 8) - min_word;
            const byte_offset: u6 = @intCast(addr % 8);
            self.bytecode_words.items[word_idx] |= @as(u64, byte) << (byte_offset * 8);
        }

        return self;
    }

    /// Serialize to arkworks format
    pub fn serialize(self: *const RAMPreprocessing, writer: anytype) !void {
        // min_bytecode_address
        try writer.writeInt(u64, self.min_bytecode_address, .little);

        // bytecode_words: Vec<u64>
        try writer.writeInt(u64, @intCast(self.bytecode_words.items.len), .little);
        for (self.bytecode_words.items) |word| {
            try writer.writeInt(u64, word, .little);
        }
    }
};

/// JoltSharedPreprocessing - shared between prover and verifier
pub const JoltSharedPreprocessing = struct {
    bytecode: BytecodePreprocessing,
    ram: RAMPreprocessing,
    memory_layout: MemoryLayout,
    max_padded_trace_length: usize,

    pub fn deinit(self: *JoltSharedPreprocessing) void {
        self.bytecode.deinit();
        self.ram.deinit();
    }

    /// Serialize to arkworks format
    pub fn serialize(self: *const JoltSharedPreprocessing, allocator: Allocator, writer: anytype) !void {
        try self.bytecode.serialize(allocator, writer);
        try self.ram.serialize(writer);
        try serializeMemoryLayout(&self.memory_layout, writer);
        // max_padded_trace_length: usize (as u64)
        try writer.writeInt(u64, @intCast(self.max_padded_trace_length), .little);
    }
};

/// Serialize MemoryLayout to arkworks format
pub fn serializeMemoryLayout(layout: *const MemoryLayout, writer: anytype) !void {
    try writer.writeInt(u64, layout.program_size, .little);
    try writer.writeInt(u64, layout.max_trusted_advice_size, .little);
    try writer.writeInt(u64, layout.trusted_advice_start, .little);
    try writer.writeInt(u64, layout.trusted_advice_end, .little);
    try writer.writeInt(u64, layout.max_untrusted_advice_size, .little);
    try writer.writeInt(u64, layout.untrusted_advice_start, .little);
    try writer.writeInt(u64, layout.untrusted_advice_end, .little);
    try writer.writeInt(u64, layout.max_input_size, .little);
    try writer.writeInt(u64, layout.max_output_size, .little);
    try writer.writeInt(u64, layout.input_start, .little);
    try writer.writeInt(u64, layout.input_end, .little);
    try writer.writeInt(u64, layout.output_start, .little);
    try writer.writeInt(u64, layout.output_end, .little);
    try writer.writeInt(u64, layout.stack_size, .little);
    try writer.writeInt(u64, layout.stack_end, .little);
    try writer.writeInt(u64, layout.heap_size, .little);
    try writer.writeInt(u64, layout.heap_end, .little);
    try writer.writeInt(u64, layout.panic, .little);
    try writer.writeInt(u64, layout.termination, .little);
    try writer.writeInt(u64, layout.io_end, .little);
}

// ============================================================================
// Tests
// ============================================================================

test "bytecode preprocessing" {
    const allocator = std.testing.allocator;

    // Simple program: ADDI x1, x0, 42; ADD x2, x1, x1
    const code = [_]u8{
        0x93, 0x00, 0xa0, 0x02, // ADDI x1, x0, 42
        0x33, 0x81, 0x10, 0x00, // ADD x2, x1, x1
    };

    var preprocessing = try BytecodePreprocessing.preprocess(allocator, &code, 0x80000000, null);
    defer preprocessing.deinit();

    // Should have NoOp + 2 instructions, padded to power of 2
    try std.testing.expect(preprocessing.bytecode.items.len >= 3);
    try std.testing.expect(preprocessing.code_size >= 4); // Next power of 2

    // First should be NoOp
    try std.testing.expectEqual(JoltInstruction.InstructionVariant.NoOp, preprocessing.bytecode.items[0].variant);

    // Second should be ADDI
    try std.testing.expectEqual(JoltInstruction.InstructionVariant.ADDI, preprocessing.bytecode.items[1].variant);

    // Third should be ADD
    try std.testing.expectEqual(JoltInstruction.InstructionVariant.ADD, preprocessing.bytecode.items[2].variant);
}

// ============================================================================
// Dory Verifier Setup (re-exported from dory_verifier_setup.zig)
// ============================================================================

pub const dory_verifier_setup = @import("dory_verifier_setup.zig");
pub const DoryVerifierSetup = dory_verifier_setup.DoryVerifierSetup;
pub const serializeGT = dory_verifier_setup.serializeGT;
pub const serializeFp6 = dory_verifier_setup.serializeFp6;
pub const serializeFp2 = dory_verifier_setup.serializeFp2;
pub const serializeFp = dory_verifier_setup.serializeFp;
pub const serializeG1 = dory_verifier_setup.serializeG1;
pub const serializeG2 = dory_verifier_setup.serializeG2;
pub const lexicographicallyLess = dory_verifier_setup.lexicographicallyLess;
pub const lexicographicallyLessFp2 = dory_verifier_setup.lexicographicallyLessFp2;
pub const GT = dory_verifier_setup.GT;
pub const G1Point = dory_verifier_setup.G1Point;
pub const G2Point = dory_verifier_setup.G2Point;
pub const DorySRS = dory_verifier_setup.DorySRS;

/// JoltVerifierPreprocessing - full preprocessing for verification
pub const JoltVerifierPreprocessing = struct {
    generators: DoryVerifierSetup,
    shared: JoltSharedPreprocessing,

    pub fn deinit(self: *JoltVerifierPreprocessing) void {
        self.generators.deinit();
        self.shared.deinit();
    }

    /// Serialize to arkworks format
    pub fn serialize(self: *const JoltVerifierPreprocessing, allocator: Allocator, writer: anytype) !void {
        // First serialize generators (VerifierSetup)
        try self.generators.serialize(writer);
        // Then serialize shared preprocessing
        try self.shared.serialize(allocator, writer);
    }
};
