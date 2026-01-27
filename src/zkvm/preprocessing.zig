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
        // Virtual instructions
        VirtualASSERT_EQ,
        VirtualASSERT_LTE,
        VirtualADVICE,
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
        /// S-type: rs1, rs2, imm (Jolt uses i64)
        FormatS: struct { rs1: u8, rs2: u8, imm: i64 },
        /// B-type: rs1, rs2, imm (Jolt uses i128)
        FormatB: struct { rs1: u8, rs2: u8, imm: i128 },
        /// U-type: rd, imm (Jolt uses u64)
        FormatU: struct { rd: u8, imm: u64 },
        /// J-type: rd, imm (Jolt uses u64)
        FormatJ: struct { rd: u8, imm: u64 },
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

        // is_compressed
        try writer.writeAll(",\"is_compressed\":");
        try writer.writeAll(if (self.is_compressed) "true" else "false");

        try writer.writeAll("}}");

        return list.toOwnedSlice(allocator);
    }
};

/// BytecodePCMapper - maps instruction addresses to program counter indices
pub const BytecodePCMapper = struct {
    /// Maps (address - base) / alignment to (base_pc, max_inline_seq)
    indices: std.ArrayListUnmanaged(?struct { usize, u16 }),
    allocator: Allocator,

    pub fn init(allocator: Allocator) BytecodePCMapper {
        return .{
            .indices = .{},
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *BytecodePCMapper) void {
        self.indices.deinit(self.allocator);
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

    allocator: Allocator,

    pub fn init(allocator: Allocator) BytecodePreprocessing {
        return .{
            .code_size = 0,
            .bytecode = .{},
            .pc_map = BytecodePCMapper.init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *BytecodePreprocessing) void {
        self.bytecode.deinit(self.allocator);
        self.pc_map.deinit();
    }

    /// Preprocess bytecode from raw bytes
    pub fn preprocess(allocator: Allocator, code_bytes: []const u8, base_address: u64) !BytecodePreprocessing {
        var self = BytecodePreprocessing.init(allocator);
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

        // Decode all instructions
        var offset: usize = 0;
        while (offset < code_bytes.len) {
            const addr = base_address + offset;

            // Check if compressed (RVC)
            const first_halfword: u16 = std.mem.readInt(u16, code_bytes[offset..][0..2], .little);
            const is_compressed = (first_halfword & 0x3) != 0x3;

            var instruction: u32 = undefined;
            var instr_size: usize = undefined;

            if (is_compressed) {
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

            // Decode and add
            const jolt_instr = try decodeToJoltInstruction(instruction, addr, is_compressed);
            try self.bytecode.append(allocator, jolt_instr);

            offset += instr_size;
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
fn serializeMemoryLayout(layout: *const MemoryLayout, writer: anytype) !void {
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
    try writer.writeInt(u64, layout.memory_size, .little);
    try writer.writeInt(u64, layout.memory_end, .little);
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
            operands = .{ .FormatJ = .{ .rd = rd, .imm = imm } };
        },
        0b1100111 => { // JALR
            variant = .JALR;
            const imm = decodeIImmediate(instruction);
            operands = .{ .FormatI = .{ .rd = rd, .rs1 = rs1, .imm = imm } };
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
            const imm = decodeIImmediate(instruction);
            operands = .{ .FormatI = .{ .rd = rd, .rs1 = rs1, .imm = imm } };
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
                    // Shift amount is in lower 6 bits of imm for RV64
                    operands = .{ .FormatI = .{ .rd = rd, .rs1 = rs1, .imm = @as(u64, rs2) } };
                },
                0b101 => {
                    if (funct7 & 0x20 != 0) {
                        variant = .SRAI;
                    } else {
                        variant = .SRLI;
                    }
                    operands = .{ .FormatI = .{ .rd = rd, .rs1 = rs1, .imm = @as(u64, rs2) } };
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

    var preprocessing = try BytecodePreprocessing.preprocess(allocator, &code, 0x80000000);
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

/// Multi-pairing of G1 and G2 vectors
fn multiPair(g1_vec: []const G1Point, g2_vec: []const G2Point) GT {
    const n = @min(g1_vec.len, g2_vec.len);
    if (n == 0) return GT.one();

    var result = GT.one();
    for (0..n) |i| {
        if (g1_vec[i].infinity or g2_vec[i].infinity) continue;
        // Convert G1Point to Fp coordinates for pairing
        const g1_fp = pairing.G1PointFp{
            .x = g1PointXToFp(g1_vec[i]),
            .y = g1PointYToFp(g1_vec[i]),
            .infinity = g1_vec[i].infinity,
        };
        const paired = pairing.pairingFp(g1_fp, g2_vec[i]);
        result = result.mul(paired);
    }
    return result;
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
    pub fn fromSRS(allocator: Allocator, srs: *const DorySRS) !DoryVerifierSetup {
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

                // Δ₁R[k] = e(Γ₁[2^(k-1)..2^k], Γ₂[..2^(k-1)])
                try delta_1r.append(allocator, multiPair(g1_second_half, g2_first_half));

                // Δ₂R[k] = e(Γ₁[..2^(k-1)], Γ₂[2^(k-1)..2^k])
                try delta_2r.append(allocator, multiPair(g1_first_half, g2_second_half));

                // χ[k] = χ[k-1] + e(Γ₁[2^(k-1)..2^k], Γ₂[2^(k-1)..2^k])
                const chi_k = chi.items[k - 1].mul(multiPair(g1_second_half, g2_second_half));
                try chi.append(allocator, chi_k);
            }
        }

        // delta_2l == delta_1l (clone)
        var delta_2l = std.ArrayListUnmanaged(GT){};
        try delta_2l.ensureTotalCapacity(allocator, delta_1l.items.len);
        for (delta_1l.items) |item| {
            try delta_2l.append(allocator, item);
        }

        // Use first generators, and compute h1, h2, ht
        // Note: Jolt generates h1, h2 from the RNG too, but we'll use first generators for simplicity
        // In production, these should match Jolt's exact values
        const h1 = srs.g1_vec[0];
        const h2 = srs.g2_vec[0];
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
    var verifier_setup = try DoryVerifierSetup.fromSRS(allocator, &srs);
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
    std.debug.print("Verifier setup serialized to {} bytes\n", .{buf.items.len});
}
