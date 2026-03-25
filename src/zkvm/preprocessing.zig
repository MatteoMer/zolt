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

// Debug output control - set to true to enable verbose debug prints
const debug_verbose = false;
fn dbg(comptime fmt: []const u8, args: anytype) void {
    if (debug_verbose) std.debug.print(fmt, args);
}

const Allocator = std.mem.Allocator;
const jolt_device = @import("jolt_device.zig");
const MemoryLayout = jolt_device.MemoryLayout;
const common = @import("../common/mod.zig");

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

/// BytecodePCMapper - maps instruction addresses to program counter indices
/// Matches Jolt's BytecodePCMapper: maps (address - RAM_START) / ALIGNMENT to (base_pc, max_inline_seq)
/// RAM_START_ADDRESS = 0x80000000, ALIGNMENT_FACTOR_BYTECODE = 2
pub const BytecodePCMapper = struct {
    const RAM_START_ADDRESS: u64 = 0x80000000;
    const ALIGNMENT_FACTOR: u64 = 2;

    /// Maps (address - base) / alignment to (base_pc, max_inline_seq)
    indices: std.ArrayListUnmanaged(?struct { usize, u16 }),
    allocator: Allocator,
    /// Base bytecode index for termination sequence (LUI, ADDI, SB, JAL).
    /// Set to last_pc + 1 after processing all real instructions.
    /// LUI→termination_base_pc, ADDI→+1, SB→+2, JAL→+3
    termination_base_pc: usize = 0,

    pub fn init(allocator: Allocator) BytecodePCMapper {
        return .{
            .indices = .{},
            .allocator = allocator,
            .termination_base_pc = 0,
        };
    }

    pub fn deinit(self: *BytecodePCMapper) void {
        self.indices.deinit(self.allocator);
    }

    /// Build the PC map from a bytecode array (already has NoOp prepended at index 0)
    pub fn build(self: *BytecodePCMapper, bytecode: []const JoltInstruction) !void {
        if (bytecode.len <= 1) {
            // Only the NoOp, nothing to map
            try self.indices.append(self.allocator, .{ 0, 0 }); // index 0 for NoOp
            return;
        }

        // Find the maximum address to size the indices array
        var max_addr: u64 = 0;
        for (bytecode) |instr| {
            if (instr.variant != .NoOp and instr.variant != .UNIMPL and instr.address > 0) {
                max_addr = @max(max_addr, instr.address);
            }
        }

        if (max_addr == 0) {
            try self.indices.append(self.allocator, .{ 0, 0 });
            return;
        }

        // Size the array: getIndex(max_addr) + 1
        const max_index = getIndex(max_addr) + 1;
        try self.indices.resize(self.allocator, max_index);
        @memset(self.indices.items, null);

        // Index 0 maps to NoOp (pc=0)
        self.indices.items[0] = .{ 0, 0 };

        // Walk through bytecode array (skip index 0 which is NoOp)
        var last_pc: usize = 0;
        for (bytecode[1..]) |instr| {
            if (instr.address == 0) {
                // Padding NoOp or UNIMPL - skip
                continue;
            }
            last_pc += 1;
            const idx = getIndex(instr.address);
            if (idx < self.indices.items.len) {
                if (self.indices.items[idx] == null) {
                    self.indices.items[idx] = .{
                        last_pc,
                        instr.virtual_sequence_remaining orelse 0,
                    };
                }
            }
        }

        // Reserve 4 bytecode entries for the termination sequence (LUI+ADDI+SB+JAL)
        // These come after all real instructions: k=last_pc+1 through k=last_pc+4
        self.termination_base_pc = last_pc + 1;
    }

    /// Convert an ELF address to array index: (address - RAM_START) / ALIGNMENT + 1
    pub fn getIndex(address: u64) usize {
        return @intCast((address - RAM_START_ADDRESS) / ALIGNMENT_FACTOR + 1);
    }

    /// Get the bytecode array index (PC) for a given ELF address and virtual_sequence_remaining.
    /// For NoOp (address=0), returns 0 (NOP entry).
    /// For regular instructions (no virtual sequences), virtual_sequence_remaining = 0.
    pub fn getPC(self: *const BytecodePCMapper, address: u64, virtual_sequence_remaining: u16) usize {
        if (address == 0) return 0;
        const idx = getIndex(address);
        if (idx >= self.indices.items.len) return 0;
        if (self.indices.items[idx]) |entry| {
            const base_pc = entry[0];
            const max_inline_seq = entry[1];
            return base_pc + @as(usize, max_inline_seq - virtual_sequence_remaining);
        }
        return 0;
    }

    /// Get the bytecode index for a termination store virtual instruction.
    /// LUI (vsr=2) → termination_base_pc, ADDI (vsr=1) → +1, SD (vsr=0) → +2
    pub fn getTerminationPC(self: *const BytecodePCMapper, virtual_sequence_remaining: u16) usize {
        return self.termination_base_pc + @as(usize, 2 - virtual_sequence_remaining);
    }

    /// Get the bytecode index for a trace step (convenience function).
    /// Handles NoOp, termination store, termination JAL, and regular instruction cases.
    pub fn getPCForStep(self: *const BytecodePCMapper, step: anytype) usize {
        if (step.is_termination_jal) return self.termination_base_pc + 3; // JAL-to-self
        if (step.is_noop and !step.is_termination_store) return 0; // NOP padding
        if (step.is_termination_store and !step.is_noop) {
            // Real termination store instruction (LUI, ADDI, or SB)
            return self.getTerminationPC(step.virtual_sequence_remaining);
        }
        if (step.is_noop and step.is_termination_store) return 0; // Dummy termination noop → NOP
        return self.getPC(step.pc, step.virtual_sequence_remaining);
    }

    /// Serialize to arkworks format
    pub fn serialize(self: *const BytecodePCMapper, writer: anytype) !void {
        // Vec<Option<(usize, u16)>>
        // Length as u64
        try writer.writeInt(u64, @intCast(self.indices.items.len), .little);

        for (self.indices.items) |maybe_entry| {
            if (maybe_entry) |entry| {
                // Some variant: 1 byte flag + (usize as u64) + u16
                try writer.writeByte(1);
                try writer.writeInt(u64, @intCast(entry[0]), .little);
                try writer.writeInt(u16, entry[1], .little);
            } else {
                // None variant: 1 byte flag
                try writer.writeByte(0);
            }
        }
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
            const jolt_instr = try decodeToJoltInstruction(instruction, addr, is_compressed);
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

/// Decode a 32-bit instruction to JoltInstruction format
fn decodeToJoltInstruction(instruction: u32, address: u64, is_compressed: bool) !JoltInstruction {
    const opcode = instruction & 0x7f;
    const rd: u8 = @truncate((instruction >> 7) & 0x1f);
    const funct3: u3 = @truncate((instruction >> 12) & 0x7);
    const rs1: u8 = @truncate((instruction >> 15) & 0x1f);
    const rs2: u8 = @truncate((instruction >> 20) & 0x1f);
    const funct7: u7 = @truncate((instruction >> 25) & 0x7f);

    var variant: JoltInstruction.InstructionVariant = .UNIMPL;
    var operands: JoltInstruction.Operands = .{ .None = {} };

    switch (opcode) {
        0b0110111 => { // LUI
            variant = .LUI;
            const imm = decodeUImmediate(instruction);
            operands = .{ .FormatU = .{ .rd = rd, .imm = imm } };
        },
        0b0010111 => { // AUIPC
            variant = .AUIPC;
            const imm = decodeUImmediate(instruction);
            operands = .{ .FormatU = .{ .rd = rd, .imm = imm } };
        },
        0b1101111 => { // JAL
            variant = .JAL;
            const imm = decodeJImmediate(instruction);
            // Upstream Jolt remaps JAL with rd=x0 to virtual register 40
            const effective_rd: u8 = if (rd == 0) 40 else rd;
            operands = .{ .FormatJ = .{ .rd = effective_rd, .imm = imm } };
        },
        0b1100111 => { // JALR
            variant = .JALR;
            const imm = decodeIImmediate(instruction);
            // Upstream Jolt remaps JALR with rd=x0 to virtual register 40
            const effective_rd: u8 = if (rd == 0) 40 else rd;
            operands = .{ .FormatI = .{ .rd = effective_rd, .rs1 = rs1, .imm = imm } };
        },
        0b1100011 => { // Branch
            const imm = decodeBImmediate(instruction);
            operands = .{ .FormatB = .{ .rs1 = rs1, .rs2 = rs2, .imm = imm } };
            variant = switch (funct3) {
                0b000 => .BEQ,
                0b001 => .BNE,
                0b100 => .BLT,
                0b101 => .BGE,
                0b110 => .BLTU,
                0b111 => .BGEU,
                else => .UNIMPL,
            };
        },
        0b0000011 => { // Load
            const imm = decodeLoadImmediate(instruction);
            operands = .{ .FormatLoad = .{ .rd = rd, .rs1 = rs1, .imm = imm } };
            variant = switch (funct3) {
                0b000 => .LB,
                0b001 => .LH,
                0b010 => .LW,
                0b011 => .LD,
                0b100 => .LBU,
                0b101 => .LHU,
                0b110 => .LWU,
                else => .UNIMPL,
            };
        },
        0b0100011 => { // Store
            const imm = decodeSImmediate(instruction);
            operands = .{ .FormatS = .{ .rs1 = rs1, .rs2 = rs2, .imm = imm } };
            variant = switch (funct3) {
                0b000 => .SB,
                0b001 => .SH,
                0b010 => .SW,
                0b011 => .SD,
                else => .UNIMPL,
            };
        },
        0b0010011 => { // OP-IMM
            const imm = decodeIImmediate(instruction);
            operands = .{ .FormatI = .{ .rd = rd, .rs1 = rs1, .imm = imm } };
            switch (funct3) {
                0b000 => variant = .ADDI,
                0b010 => variant = .SLTI,
                0b011 => variant = .SLTIU,
                0b100 => variant = .XORI,
                0b110 => variant = .ORI,
                0b111 => variant = .ANDI,
                0b001 => {
                    variant = .SLLI;
                    // RV64: shift amount is 6 bits (bits 25:20), not just 5 (rs2 field is only 24:20)
                    operands = .{ .FormatI = .{ .rd = rd, .rs1 = rs1, .imm = @as(u64, (rs2 & 0x1f) | (@as(u8, @intCast(@as(u8, funct7) & 1)) << 5)) } };
                },
                0b101 => {
                    if (funct7 & 0x20 != 0) {
                        variant = .SRAI;
                    } else {
                        variant = .SRLI;
                    }
                    // RV64: shift amount is 6 bits (bits 25:20), not just 5 (rs2 field)
                    operands = .{ .FormatI = .{ .rd = rd, .rs1 = rs1, .imm = @as(u64, (rs2 & 0x1f) | (@as(u8, @intCast(@as(u8, funct7) & 1)) << 5)) } };
                },
            }
        },
        0b0110011 => { // OP
            operands = .{ .FormatR = .{ .rd = rd, .rs1 = rs1, .rs2 = rs2 } };
            if (funct7 == 0b0000001) {
                // M extension
                variant = switch (funct3) {
                    0b000 => .MUL,
                    0b001 => .MULH,
                    0b010 => .MULHSU,
                    0b011 => .MULHU,
                    0b100 => .DIV,
                    0b101 => .DIVU,
                    0b110 => .REM,
                    0b111 => .REMU,
                };
            } else {
                variant = switch (funct3) {
                    0b000 => if (funct7 == 0x20) .SUB else .ADD,
                    0b001 => .SLL,
                    0b010 => .SLT,
                    0b011 => .SLTU,
                    0b100 => .XOR,
                    0b101 => if (funct7 == 0x20) .SRA else .SRL,
                    0b110 => .OR,
                    0b111 => .AND,
                };
            }
        },
        0b0011011 => { // OP-IMM-32 (RV64I)
            const imm = decodeIImmediate(instruction);
            operands = .{ .FormatI = .{ .rd = rd, .rs1 = rs1, .imm = imm } };
            switch (funct3) {
                0b000 => variant = .ADDIW,
                0b001 => {
                    variant = .SLLIW;
                    operands = .{ .FormatI = .{ .rd = rd, .rs1 = rs1, .imm = @as(u64, rs2 & 0x1f) } };
                },
                0b101 => {
                    if (funct7 & 0x20 != 0) {
                        variant = .SRAIW;
                    } else {
                        variant = .SRLIW;
                    }
                    operands = .{ .FormatI = .{ .rd = rd, .rs1 = rs1, .imm = @as(u64, rs2 & 0x1f) } };
                },
                else => variant = .UNIMPL,
            }
        },
        0b0111011 => { // OP-32 (RV64I)
            operands = .{ .FormatR = .{ .rd = rd, .rs1 = rs1, .rs2 = rs2 } };
            if (funct7 == 0b0000001) {
                // M extension 32-bit
                variant = switch (funct3) {
                    0b000 => .MULW,
                    0b100 => .DIVW,
                    0b101 => .DIVUW,
                    0b110 => .REMW,
                    0b111 => .REMUW,
                    else => .UNIMPL,
                };
            } else {
                variant = switch (funct3) {
                    0b000 => if (funct7 == 0x20) .SUBW else .ADDW,
                    0b001 => .SLLW,
                    0b101 => if (funct7 == 0x20) .SRAW else .SRLW,
                    else => .UNIMPL,
                };
            }
        },
        0b0001111 => { // FENCE - uses FormatI in Jolt
            variant = .FENCE;
            const imm = decodeIImmediate(instruction);
            operands = .{ .FormatI = .{ .rd = rd, .rs1 = rs1, .imm = imm } };
        },
        0b1110011 => { // SYSTEM - ECALL uses FormatI in Jolt
            variant = .ECALL;
            const imm = decodeIImmediate(instruction);
            operands = .{ .FormatI = .{ .rd = rd, .rs1 = rs1, .imm = imm } };
        },
        else => {
            variant = .UNIMPL;
        },
    }

    return .{
        .variant = variant,
        .address = address,
        .operands = operands,
        .virtual_sequence_remaining = null,
        .is_first_in_sequence = false,
        .is_compressed = is_compressed,
    };
}

/// Decode Load-format immediate to i64 (sign-extended from 12-bit signed)
/// Jolt uses i64 for FormatLoad.imm (distinct from FormatI which uses u64)
fn decodeLoadImmediate(instruction: u32) i64 {
    const imm: u32 = instruction >> 20;
    const signed: i32 = if (imm & 0x800 != 0)
        @bitCast(imm | 0xFFFFF000)
    else
        @bitCast(imm);
    return @as(i64, signed);
}

/// Decode I-format immediate to u64 (sign-extended from 12-bit signed)
/// Jolt uses u64 for FormatI.imm
fn decodeIImmediate(instruction: u32) u64 {
    const imm: u32 = instruction >> 20;
    // Sign extend from 12 bits to i32, then to i64, then cast to u64
    const signed: i32 = if (imm & 0x800 != 0)
        @bitCast(imm | 0xFFFFF000)
    else
        @bitCast(imm);
    return @bitCast(@as(i64, signed));
}

/// Decode S-format immediate to i64 (signed)
/// Jolt uses i64 for FormatS.imm
fn decodeSImmediate(instruction: u32) i64 {
    const imm11_5 = (instruction >> 25) & 0x7F;
    const imm4_0 = (instruction >> 7) & 0x1F;
    const imm = (imm11_5 << 5) | imm4_0;
    // Sign extend from 12 bits to i32, then to i64
    const signed: i32 = if (imm & 0x800 != 0)
        @bitCast(imm | 0xFFFFF000)
    else
        @bitCast(imm);
    return @as(i64, signed);
}

/// Decode B-format immediate to i128 (signed)
/// Jolt uses i128 for FormatB.imm
fn decodeBImmediate(instruction: u32) i128 {
    const imm12 = (instruction >> 31) & 1;
    const imm10_5 = (instruction >> 25) & 0x3F;
    const imm4_1 = (instruction >> 8) & 0xF;
    const imm11 = (instruction >> 7) & 1;
    const imm = (imm12 << 12) | (imm11 << 11) | (imm10_5 << 5) | (imm4_1 << 1);
    // Sign extend from 13 bits to i32, then to i128
    const signed: i32 = if (imm & 0x1000 != 0)
        @bitCast(imm | 0xFFFFE000)
    else
        @bitCast(imm);
    return @as(i128, signed);
}

/// Decode J-format immediate to u64 (sign-extended from 21-bit signed)
/// Jolt uses u64 for FormatJ.imm
fn decodeJImmediate(instruction: u32) u64 {
    const imm20 = (instruction >> 31) & 1;
    const imm10_1 = (instruction >> 21) & 0x3FF;
    const imm11 = (instruction >> 20) & 1;
    const imm19_12 = (instruction >> 12) & 0xFF;
    const imm = (imm20 << 20) | (imm19_12 << 12) | (imm11 << 11) | (imm10_1 << 1);
    // Sign extend from 21 bits to i32, then to i64, then cast to u64
    const signed: i32 = if (imm & 0x100000 != 0)
        @bitCast(imm | 0xFFE00000)
    else
        @bitCast(imm);
    return @bitCast(@as(i64, signed));
}

/// Decode U-format immediate to u64 (sign-extended from 32-bit signed)
/// Jolt uses u64 for FormatU.imm
fn decodeUImmediate(instruction: u32) u64 {
    // Upper 20 bits of instruction, in upper 20 bits of result
    const imm: i32 = @bitCast(instruction & 0xFFFFF000);
    // Sign extend to 64 bits then cast to u64
    return @bitCast(@as(i64, imm));
}

// ============================================================================
// Tests
// ============================================================================

test "decode ADD instruction to JSON" {
    const allocator = std.testing.allocator;

    // ADD x1, x2, x3 -> 0x003100b3
    const instr = try decodeToJoltInstruction(0x003100b3, 0x80000000, false);
    try std.testing.expectEqual(JoltInstruction.InstructionVariant.ADD, instr.variant);

    const json = try instr.toJson(allocator);
    defer allocator.free(json);

    // Should contain the variant name and operands
    try std.testing.expect(std.mem.indexOf(u8, json, "\"ADD\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"rd\":1") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"rs1\":2") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"rs2\":3") != null);
}

test "decode ADDI instruction to JSON" {
    const allocator = std.testing.allocator;

    // ADDI x1, x2, 100 -> 0x06410093
    const instr = try decodeToJoltInstruction(0x06410093, 0x80000004, false);
    try std.testing.expectEqual(JoltInstruction.InstructionVariant.ADDI, instr.variant);

    const json = try instr.toJson(allocator);
    defer allocator.free(json);

    try std.testing.expect(std.mem.indexOf(u8, json, "\"ADDI\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"imm\":100") != null);
}

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
// Dory Verifier Setup Serialization
// ============================================================================

const dory = @import("../poly/commitment/dory.zig");
const pairing = @import("../field/pairing.zig");
const field_mod = @import("../field/mod.zig");
const Fp = field_mod.BN254BaseField;
const ThreadPool = @import("../utils/thread_pool.zig").ThreadPool;

pub const GT = dory.GT;
pub const G1Point = dory.G1Point;
pub const G2Point = dory.G2Point;
pub const DorySRS = dory.DorySRS;

/// Convert G1Point x-coordinate to Fp
/// G1Point stores x,y in BN254Scalar (Fr) Montgomery form
/// We need to convert to BN254BaseField (Fp) for pairing
fn g1PointXToFp(p: G1Point) Fp {
    // The x coordinate is stored in Montgomery form for Fr
    // Both Fr and Fp have the same limb structure, just different moduli
    // Since the G1 generator coords are valid in both fields (they're small),
    // we can interpret the limbs directly
    return Fp{ .limbs = p.x.limbs };
}

fn g1PointYToFp(p: G1Point) Fp {
    return Fp{ .limbs = p.y.limbs };
}

/// Multi-pairing of G1 and G2 vectors using batch Miller loop + shared final exponentiation.
/// Optionally parallelizes Miller loop computation across threads.
fn multiPair(g1_vec: []const G1Point, g2_vec: []const G2Point, tp: ?*ThreadPool) GT {
    const n = @min(g1_vec.len, g2_vec.len);
    if (n == 0) return GT.one();

    const Ctx = struct { g1: []const G1Point, g2: []const G2Point };
    const ctx = Ctx{ .g1 = g1_vec, .g2 = g2_vec };

    const mapFn = struct {
        fn map(c: Ctx, start: usize, end: usize) pairing.Fp12 {
            var acc = pairing.Fp12.one();
            for (start..end) |i| {
                if (c.g1[i].infinity or c.g2[i].infinity) continue;
                const g1_fp = pairing.G1PointFp{
                    .x = g1PointXToFp(c.g1[i]),
                    .y = g1PointYToFp(c.g1[i]),
                    .infinity = false,
                };
                const ml = pairing.millerLoopArkworks(g1_fp, c.g2[i]);
                acc = acc.mul(ml);
            }
            return acc;
        }
    }.map;

    const reduceFn = struct {
        fn reduce(a: pairing.Fp12, b: pairing.Fp12) pairing.Fp12 {
            return a.mul(b);
        }
    }.reduce;

    const miller_acc = if (tp) |pool|
        pool.parallelReduceForce(pairing.Fp12, n, pairing.Fp12.one(), ctx, mapFn, reduceFn)
    else
        mapFn(ctx, 0, n);

    if (miller_acc.eql(pairing.Fp12.one())) return GT.one();
    return pairing.finalExponentiation(miller_acc);
}

/// DoryVerifierSetup - precomputed pairing values for verification
/// Matches Jolt's VerifierSetup<BN254> structure
pub const DoryVerifierSetup = struct {
    /// Δ₁L[k] = e(Γ₁[..2^(k-1)], Γ₂[..2^(k-1)])
    delta_1l: std.ArrayListUnmanaged(GT),
    /// Δ₁R[k] = e(Γ₁[2^(k-1)..2^k], Γ₂[..2^(k-1)])
    delta_1r: std.ArrayListUnmanaged(GT),
    /// Δ₂L[k] = same as Δ₁L[k]
    delta_2l: std.ArrayListUnmanaged(GT),
    /// Δ₂R[k] = e(Γ₁[..2^(k-1)], Γ₂[2^(k-1)..2^k])
    delta_2r: std.ArrayListUnmanaged(GT),
    /// χ[k] = e(Γ₁[..2^k], Γ₂[..2^k])
    chi: std.ArrayListUnmanaged(GT),
    /// First G1 generator
    g1_0: G1Point,
    /// First G2 generator
    g2_0: G2Point,
    /// Blinding generator in G1
    h1: G1Point,
    /// Blinding generator in G2
    h2: G2Point,
    /// h_t = e(h₁, h₂)
    ht: GT,
    /// Maximum log₂ of polynomial size supported
    max_log_n: usize,
    /// Allocator used
    allocator: Allocator,

    pub fn deinit(self: *DoryVerifierSetup) void {
        self.delta_1l.deinit(self.allocator);
        self.delta_1r.deinit(self.allocator);
        self.delta_2l.deinit(self.allocator);
        self.delta_2r.deinit(self.allocator);
        self.chi.deinit(self.allocator);
    }

    /// Create verifier setup from prover setup (SRS)
    pub fn fromSRS(allocator: Allocator, srs: *const DorySRS, tp: ?*ThreadPool) !DoryVerifierSetup {
        const max_num_rounds = std.math.log2_int(usize, srs.g1_vec.len);

        var delta_1l = std.ArrayListUnmanaged(GT){};
        var delta_1r = std.ArrayListUnmanaged(GT){};
        var delta_2r = std.ArrayListUnmanaged(GT){};
        var chi = std.ArrayListUnmanaged(GT){};

        try delta_1l.ensureTotalCapacity(allocator, max_num_rounds + 1);
        try delta_1r.ensureTotalCapacity(allocator, max_num_rounds + 1);
        try delta_2r.ensureTotalCapacity(allocator, max_num_rounds + 1);
        try chi.ensureTotalCapacity(allocator, max_num_rounds + 1);

        for (0..(max_num_rounds + 1)) |k| {
            if (k == 0) {
                // Base case: identities for deltas, single pairing for chi
                try delta_1l.append(allocator, GT.one());
                try delta_1r.append(allocator, GT.one());
                try delta_2r.append(allocator, GT.one());
                // chi[0] = e(g1_vec[0], g2_vec[0])
                const g1_0_fp = pairing.G1PointFp{
                    .x = g1PointXToFp(srs.g1_vec[0]),
                    .y = g1PointYToFp(srs.g1_vec[0]),
                    .infinity = srs.g1_vec[0].infinity,
                };
                const chi_0 = pairing.pairingFp(g1_0_fp, srs.g2_vec[0]);
                try chi.append(allocator, chi_0);
            } else {
                const half_len = @as(usize, 1) << @intCast(k - 1);
                const full_len = @as(usize, 1) << @intCast(k);

                const g1_first_half = srs.g1_vec[0..half_len];
                const g1_second_half = srs.g1_vec[half_len..full_len];
                const g2_first_half = srs.g2_vec[0..half_len];
                const g2_second_half = srs.g2_vec[half_len..full_len];

                // Δ₁L[k] = χ[k-1] (reuse previous chi)
                try delta_1l.append(allocator, chi.items[k - 1]);

                // Compute 3 independent multi-pairings:
                //   Δ₁R[k] = e(Γ₁[2^(k-1)..2^k], Γ₂[..2^(k-1)])
                //   Δ₂R[k] = e(Γ₁[..2^(k-1)], Γ₂[2^(k-1)..2^k])
                //   cross  = e(Γ₁[2^(k-1)..2^k], Γ₂[2^(k-1)..2^k])
                // Batch all 3 independent multi-pairings into one parallelReduceForce call.
                const batch = dory.multiPairBatched(3, .{
                    dory.PairGroup{ .g1 = g1_second_half, .g2 = g2_first_half }, // delta_1r
                    dory.PairGroup{ .g1 = g1_first_half, .g2 = g2_second_half }, // delta_2r
                    dory.PairGroup{ .g1 = g1_second_half, .g2 = g2_second_half }, // cross
                }, tp);
                try delta_1r.append(allocator, batch[0]);
                try delta_2r.append(allocator, batch[1]);

                // χ[k] = χ[k-1] * cross
                const chi_k = chi.items[k - 1].mul(batch[2]);
                try chi.append(allocator, chi_k);
            }
        }

        // delta_2l == delta_1l (clone)
        var delta_2l = std.ArrayListUnmanaged(GT){};
        try delta_2l.ensureTotalCapacity(allocator, delta_1l.items.len);
        for (delta_1l.items) |item| {
            try delta_2l.append(allocator, item);
        }

        // Use h1, h2 from the SRS (these are separate blinding generators)
        const h1 = srs.h1;
        const h2 = srs.h2;
        const h1_fp = pairing.G1PointFp{
            .x = g1PointXToFp(h1),
            .y = g1PointYToFp(h1),
            .infinity = h1.infinity,
        };
        const ht = pairing.pairingFp(h1_fp, h2);

        return DoryVerifierSetup{
            .delta_1l = delta_1l,
            .delta_1r = delta_1r,
            .delta_2l = delta_2l,
            .delta_2r = delta_2r,
            .chi = chi,
            .g1_0 = srs.g1_vec[0],
            .g2_0 = srs.g2_vec[0],
            .h1 = h1,
            .h2 = h2,
            .ht = ht,
            .max_log_n = max_num_rounds * 2, // Since square matrices
            .allocator = allocator,
        };
    }

    /// Serialize to arkworks-compatible format
    /// Matches the CanonicalSerialize impl for VerifierSetup<BN254>
    pub fn serialize(self: *const DoryVerifierSetup, writer: anytype) !void {
        // Serialize delta_1l: Vec<GT>
        try writer.writeInt(u64, @intCast(self.delta_1l.items.len), .little);
        for (self.delta_1l.items) |gt| {
            try serializeGT(gt, writer);
        }

        // Serialize delta_1r: Vec<GT>
        try writer.writeInt(u64, @intCast(self.delta_1r.items.len), .little);
        for (self.delta_1r.items) |gt| {
            try serializeGT(gt, writer);
        }

        // Serialize delta_2l: Vec<GT>
        try writer.writeInt(u64, @intCast(self.delta_2l.items.len), .little);
        for (self.delta_2l.items) |gt| {
            try serializeGT(gt, writer);
        }

        // Serialize delta_2r: Vec<GT>
        try writer.writeInt(u64, @intCast(self.delta_2r.items.len), .little);
        for (self.delta_2r.items) |gt| {
            try serializeGT(gt, writer);
        }

        // Serialize chi: Vec<GT>
        try writer.writeInt(u64, @intCast(self.chi.items.len), .little);
        for (self.chi.items) |gt| {
            try serializeGT(gt, writer);
        }

        // Serialize g1_0: G1
        try serializeG1(self.g1_0, writer);

        // Serialize g2_0: G2
        try serializeG2(self.g2_0, writer);

        // Serialize h1: G1
        try serializeG1(self.h1, writer);

        // Serialize h2: G2
        try serializeG2(self.h2, writer);

        // Serialize ht: GT
        try serializeGT(self.ht, writer);

        // Serialize max_log_n: usize (as u64)
        try writer.writeInt(u64, @intCast(self.max_log_n), .little);
    }
};

/// Serialize GT element in arkworks format (uncompressed Fq12)
fn serializeGT(gt: GT, writer: anytype) !void {
    // GT is Fp12 = (Fp6, Fp6) where Fp6 = (Fp2, Fp2, Fp2)
    // arkworks serializes as c0 first, then c1
    // Each Fp2 is (c0, c1) where each Fp is 32 bytes LE

    // Serialize gt.c0 (Fp6)
    try serializeFp6(&gt.c0, writer);
    // Serialize gt.c1 (Fp6)
    try serializeFp6(&gt.c1, writer);
}

fn serializeFp6(fp6: *const pairing.Fp6, writer: anytype) !void {
    // Fp6 = (Fp2, Fp2, Fp2)
    try serializeFp2(&fp6.c0, writer);
    try serializeFp2(&fp6.c1, writer);
    try serializeFp2(&fp6.c2, writer);
}

fn serializeFp2(fp2: *const pairing.Fp2, writer: anytype) !void {
    // Fp2 = (c0, c1) where each is Fp
    try serializeFp(&fp2.c0, writer);
    try serializeFp(&fp2.c1, writer);
}

fn serializeFp(fp: *const @import("../field/mod.zig").BN254BaseField, writer: anytype) !void {
    // Fp is 32 bytes in little-endian (standard form)
    const std_form = fp.fromMontgomery();
    for (0..4) |i| {
        try writer.writeInt(u64, std_form.limbs[i], .little);
    }
}

fn serializeG1(point: G1Point, writer: anytype) !void {
    // G1 compressed serialization: 32 bytes (x coordinate + flags)
    // arkworks compressed format: x with flags in MSB of last limb
    if (point.infinity) {
        // Point at infinity: write 0s with infinity flag (bit 62)
        for (0..3) |_| {
            try writer.writeInt(u64, 0, .little);
        }
        // Set infinity flag (bit 62 = 0x4000_0000_0000_0000)
        try writer.writeInt(u64, 0x4000000000000000, .little);
        return;
    }

    // Write x coordinate (32 bytes LE, standard form)
    const x_std = point.x.fromMontgomery();
    for (0..3) |i| {
        try writer.writeInt(u64, x_std.limbs[i], .little);
    }

    // Set flag bits in top byte of last limb of x:
    // - bit 63: y sign (positive_y_over_neg_y flag)
    // - bit 62: infinity (already handled above)
    const neg_y = point.y.neg();
    const y_is_positive = lexicographicallyLess(point.y, neg_y);
    var last_limb = x_std.limbs[3];
    if (!y_is_positive) {
        last_limb |= 0x8000000000000000; // Set bit 63 if y is "negative"
    }
    try writer.writeInt(u64, last_limb, .little);
}

fn serializeG2(point: G2Point, writer: anytype) !void {
    // G2 compressed serialization: 64 bytes (x as Fp2 + flags)
    // arkworks compressed format: x.c0, x.c1 with flags in MSB of x.c1's last limb
    if (point.infinity) {
        // Point at infinity: write 0s with infinity flag (bit 62 of x.c1)
        for (0..7) |_| {
            try writer.writeInt(u64, 0, .little);
        }
        // Set infinity flag (bit 62 = 0x4000_0000_0000_0000)
        try writer.writeInt(u64, 0x4000000000000000, .little);
        return;
    }

    // Write x.c0 (32 bytes)
    const x_c0_std = point.x.c0.fromMontgomery();
    for (0..4) |i| {
        try writer.writeInt(u64, x_c0_std.limbs[i], .little);
    }

    // Write x.c1 (32 bytes) with flags in MSB
    const x_c1_std = point.x.c1.fromMontgomery();
    for (0..3) |i| {
        try writer.writeInt(u64, x_c1_std.limbs[i], .little);
    }

    // Set flag bits in top byte of last limb of x.c1:
    // - bit 63: y sign (for Fp2, compare lexicographically)
    // - bit 62: infinity (already handled above)
    const neg_y = point.y.neg();
    const y_is_positive = lexicographicallyLessFp2(point.y, neg_y);
    var last_limb = x_c1_std.limbs[3];
    if (!y_is_positive) {
        last_limb |= 0x8000000000000000; // Set bit 63 if y is "negative"
    }
    try writer.writeInt(u64, last_limb, .little);
}

fn lexicographicallyLess(a: @import("../field/mod.zig").BN254BaseField, b: @import("../field/mod.zig").BN254BaseField) bool {
    const a_std = a.fromMontgomery();
    const b_std = b.fromMontgomery();
    var i: usize = 4;
    while (i > 0) {
        i -= 1;
        if (a_std.limbs[i] < b_std.limbs[i]) return true;
        if (a_std.limbs[i] > b_std.limbs[i]) return false;
    }
    return false; // Equal
}

fn lexicographicallyLessFp2(a: pairing.Fp2, b: pairing.Fp2) bool {
    // For Fp2 = (c0, c1), compare c1 first (more significant), then c0
    // This matches arkworks' lexicographic ordering for Fp2
    const a_c1_std = a.c1.fromMontgomery();
    const b_c1_std = b.c1.fromMontgomery();

    // Compare c1 (more significant)
    var i: usize = 4;
    while (i > 0) {
        i -= 1;
        if (a_c1_std.limbs[i] < b_c1_std.limbs[i]) return true;
        if (a_c1_std.limbs[i] > b_c1_std.limbs[i]) return false;
    }

    // c1 is equal, compare c0
    const a_c0_std = a.c0.fromMontgomery();
    const b_c0_std = b.c0.fromMontgomery();

    i = 4;
    while (i > 0) {
        i -= 1;
        if (a_c0_std.limbs[i] < b_c0_std.limbs[i]) return true;
        if (a_c0_std.limbs[i] > b_c0_std.limbs[i]) return false;
    }
    return false; // Equal
}

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

test "DoryVerifierSetup serialization" {
    const allocator = std.testing.allocator;
    const DoryCommitmentScheme = dory.DoryCommitmentScheme(@import("../field/mod.zig").BN254Scalar);

    // Create a small SRS for testing
    var srs = try DoryCommitmentScheme.setup(allocator, 4);
    defer srs.deinit();

    // Create verifier setup from SRS
    var verifier_setup = try DoryVerifierSetup.fromSRS(allocator, &srs, null);
    defer verifier_setup.deinit();

    // Check that delta/chi arrays have correct sizes
    // For 4 variables (16 coeffs), we get sigma=2, nu=2
    // g1_vec.len = 4, g2_vec.len = 4
    // max_num_rounds = log2(4) = 2
    // So we have 3 entries (k=0,1,2)
    try std.testing.expect(verifier_setup.delta_1l.items.len == 3);
    try std.testing.expect(verifier_setup.chi.items.len == 3);

    // Test serialization
    var buf = std.ArrayListUnmanaged(u8){};
    defer buf.deinit(allocator);

    try verifier_setup.serialize(buf.writer(allocator));

    // Should produce non-empty output
    try std.testing.expect(buf.items.len > 0);
    dbg("Verifier setup serialized to {} bytes\n", .{buf.items.len});
}
