//! RISC-V instruction type in Jolt's format
//!
//! This is the canonical definition of JoltInstruction, used by preprocessing,
//! instruction decoder, and serialization.

const std = @import("std");
const Allocator = std.mem.Allocator;

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
        // CSR instructions (decomposed into virtual sequences of ADDI/OR/JALR)
        CSRRW,
        CSRRS,
        MRET,
        // Jolt SDK instructions (opcode 0x5B) — names must match Jolt's Instruction enum
        VirtualHostIO, // 0x5B funct3=2 (host I/O, cycle tracking, print)
        VirtualAdviceLoad, // 0x5B funct3=3-6 (advice tape byte/halfword/word/doubleword loads)
        VirtualAdviceLen, // 0x5B funct3=7 (get advice tape remaining length)
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
        // Jolt-inline instructions
        ANDN,
        VirtualROTRI,
        VirtualROTRIW,
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
        /// Inline format: rs1, rs2, rs3 (memory pointers for jolt-inline 0x0B/0x2B instructions)
        FormatInline: struct { rs1: u8, rs2: u8, rs3: u8, funct3: u3, funct7: u7 },
        /// Virtual right-shift-I format: rd, rs1, imm (u64 bitmask for VirtualROTRI/W)
        FormatVirtualRightShiftI: struct { rd: u8, rs1: u8, imm: u64 },
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
            .FormatInline => |il| {
                try std.fmt.format(writer, "{{\"rs1\":{},\"rs2\":{},\"rs3\":{}}}", .{ il.rs1, il.rs2, il.rs3 });
            },
            .FormatVirtualRightShiftI => |vrs| {
                try std.fmt.format(writer, "{{\"rd\":{},\"rs1\":{},\"imm\":{}}}", .{ vrs.rd, vrs.rs1, vrs.imm });
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

