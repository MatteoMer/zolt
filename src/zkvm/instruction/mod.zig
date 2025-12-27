//! RISC-V instruction definitions for Jolt
//!
//! This module defines the RISC-V instruction set supported by Jolt.

const std = @import("std");

/// RISC-V instruction opcodes (lower 7 bits)
pub const Opcode = enum(u7) {
    // RV32I Base Integer Instructions
    LUI = 0b0110111, // Load Upper Immediate
    AUIPC = 0b0010111, // Add Upper Immediate to PC
    JAL = 0b1101111, // Jump and Link
    JALR = 0b1100111, // Jump and Link Register
    BRANCH = 0b1100011, // Branch
    LOAD = 0b0000011, // Load
    STORE = 0b0100011, // Store
    OP_IMM = 0b0010011, // Integer Register-Immediate
    OP = 0b0110011, // Integer Register-Register
    FENCE = 0b0001111, // Memory Fence
    SYSTEM = 0b1110011, // System (ECALL, EBREAK)

    // RV64I Extensions (additional load/store widths)
    OP_IMM_32 = 0b0011011, // 32-bit Integer Register-Immediate (RV64)
    OP_32 = 0b0111011, // 32-bit Integer Register-Register (RV64)

    // M Extension (Multiply/Divide)
    // Uses OP and OP_32 opcodes with funct7 = 0b0000001

    _,

    pub fn fromInstruction(instruction: u32) Opcode {
        return @enumFromInt(@as(u7, @truncate(instruction)));
    }
};

/// Branch function codes (funct3 for BRANCH opcode)
pub const BranchFunct3 = enum(u3) {
    BEQ = 0b000, // Branch if Equal
    BNE = 0b001, // Branch if Not Equal
    BLT = 0b100, // Branch if Less Than
    BGE = 0b101, // Branch if Greater or Equal
    BLTU = 0b110, // Branch if Less Than Unsigned
    BGEU = 0b111, // Branch if Greater or Equal Unsigned
    _,
};

/// Load function codes (funct3 for LOAD opcode)
pub const LoadFunct3 = enum(u3) {
    LB = 0b000, // Load Byte
    LH = 0b001, // Load Halfword
    LW = 0b010, // Load Word
    LD = 0b011, // Load Doubleword (RV64)
    LBU = 0b100, // Load Byte Unsigned
    LHU = 0b101, // Load Halfword Unsigned
    LWU = 0b110, // Load Word Unsigned (RV64)
    _,
};

/// Store function codes (funct3 for STORE opcode)
pub const StoreFunct3 = enum(u3) {
    SB = 0b000, // Store Byte
    SH = 0b001, // Store Halfword
    SW = 0b010, // Store Word
    SD = 0b011, // Store Doubleword (RV64)
    _,
};

/// Integer register-immediate function codes (funct3 for OP_IMM)
pub const OpImmFunct3 = enum(u3) {
    ADDI = 0b000, // Add Immediate
    SLTI = 0b010, // Set Less Than Immediate
    SLTIU = 0b011, // Set Less Than Immediate Unsigned
    XORI = 0b100, // XOR Immediate
    ORI = 0b110, // OR Immediate
    ANDI = 0b111, // AND Immediate
    SLLI = 0b001, // Shift Left Logical Immediate
    SRLI_SRAI = 0b101, // Shift Right Logical/Arithmetic Immediate
    _,
};

/// Integer register-register function codes (funct3 for OP)
pub const OpFunct3 = enum(u3) {
    ADD_SUB = 0b000, // Add/Subtract (funct7 distinguishes)
    SLL = 0b001, // Shift Left Logical
    SLT = 0b010, // Set Less Than
    SLTU = 0b011, // Set Less Than Unsigned
    XOR = 0b100, // XOR
    SRL_SRA = 0b101, // Shift Right Logical/Arithmetic
    OR = 0b110, // OR
    AND = 0b111, // AND
    _,
};

/// M Extension function codes (funct3 for OP with funct7 = 0b0000001)
/// These are multiplcation and division operations
pub const MulDivFunct3 = enum(u3) {
    MUL = 0b000, // Multiply (lower bits)
    MULH = 0b001, // Multiply High (signed * signed)
    MULHSU = 0b010, // Multiply High (signed * unsigned)
    MULHU = 0b011, // Multiply High (unsigned * unsigned)
    DIV = 0b100, // Divide (signed)
    DIVU = 0b101, // Divide (unsigned)
    REM = 0b110, // Remainder (signed)
    REMU = 0b111, // Remainder (unsigned)
};

/// M Extension function codes for 32-bit operations (RV64M with OP_32)
pub const MulDivW_Funct3 = enum(u3) {
    MULW = 0b000, // Multiply Word (lower 32 bits)
    DIVW = 0b100, // Divide Word (signed)
    DIVUW = 0b101, // Divide Word (unsigned)
    REMW = 0b110, // Remainder Word (signed)
    REMUW = 0b111, // Remainder Word (unsigned)
    _,
};

/// Decoded RISC-V instruction
pub const DecodedInstruction = struct {
    /// Original 32-bit instruction
    raw: u32,
    /// Instruction opcode
    opcode: Opcode,
    /// Destination register
    rd: u5,
    /// First source register
    rs1: u5,
    /// Second source register
    rs2: u5,
    /// Function code 3
    funct3: u3,
    /// Function code 7
    funct7: u7,
    /// Immediate value (sign-extended)
    imm: i32,
    /// Instruction format
    format: Format,

    pub const Format = enum {
        R, // Register-Register
        I, // Immediate
        S, // Store
        B, // Branch
        U, // Upper Immediate
        J, // Jump
    };

    /// Decode a 32-bit instruction
    pub fn decode(instruction: u32) DecodedInstruction {
        const opcode = Opcode.fromInstruction(instruction);
        const rd: u5 = @truncate((instruction >> 7) & 0x1F);
        const funct3: u3 = @truncate((instruction >> 12) & 0x7);
        const rs1: u5 = @truncate((instruction >> 15) & 0x1F);
        const rs2: u5 = @truncate((instruction >> 20) & 0x1F);
        const funct7: u7 = @truncate((instruction >> 25) & 0x7F);

        var format: Format = .R;
        var imm: i32 = 0;

        switch (opcode) {
            .LUI, .AUIPC => {
                format = .U;
                imm = @bitCast(instruction & 0xFFFFF000);
            },
            .JAL => {
                format = .J;
                // J-type immediate: imm[20|10:1|11|19:12]
                const imm20 = (instruction >> 31) & 1;
                const imm10_1 = (instruction >> 21) & 0x3FF;
                const imm11 = (instruction >> 20) & 1;
                const imm19_12 = (instruction >> 12) & 0xFF;
                const raw_imm = (imm20 << 20) | (imm19_12 << 12) | (imm11 << 11) | (imm10_1 << 1);
                imm = @bitCast(signExtend(u21, raw_imm));
            },
            .JALR, .LOAD, .OP_IMM, .OP_IMM_32 => {
                format = .I;
                imm = @bitCast(signExtend(u12, @truncate(instruction >> 20)));
            },
            .BRANCH => {
                format = .B;
                // B-type immediate: imm[12|10:5|4:1|11]
                const imm12 = (instruction >> 31) & 1;
                const imm10_5 = (instruction >> 25) & 0x3F;
                const imm4_1 = (instruction >> 8) & 0xF;
                const imm11 = (instruction >> 7) & 1;
                const raw_imm = (imm12 << 12) | (imm11 << 11) | (imm10_5 << 5) | (imm4_1 << 1);
                imm = @bitCast(signExtend(u13, raw_imm));
            },
            .STORE => {
                format = .S;
                // S-type immediate: imm[11:5] | imm[4:0]
                const imm11_5 = (instruction >> 25) & 0x7F;
                const imm4_0 = (instruction >> 7) & 0x1F;
                const raw_imm = (imm11_5 << 5) | imm4_0;
                imm = @bitCast(signExtend(u12, raw_imm));
            },
            .OP, .OP_32 => {
                format = .R;
            },
            else => {},
        }

        return .{
            .raw = instruction,
            .opcode = opcode,
            .rd = rd,
            .rs1 = rs1,
            .rs2 = rs2,
            .funct3 = funct3,
            .funct7 = funct7,
            .imm = imm,
            .format = format,
        };
    }

    /// Check if this is a branch instruction
    pub fn isBranch(self: DecodedInstruction) bool {
        return self.opcode == .BRANCH;
    }

    /// Check if this is a jump instruction
    pub fn isJump(self: DecodedInstruction) bool {
        return self.opcode == .JAL or self.opcode == .JALR;
    }

    /// Check if this is a memory load
    pub fn isLoad(self: DecodedInstruction) bool {
        return self.opcode == .LOAD;
    }

    /// Check if this is a memory store
    pub fn isStore(self: DecodedInstruction) bool {
        return self.opcode == .STORE;
    }

    /// Check if this is an M extension instruction (multiply/divide)
    pub fn isMulDiv(self: DecodedInstruction) bool {
        return (self.opcode == .OP or self.opcode == .OP_32) and self.funct7 == 0b0000001;
    }

    /// Check if this is a multiplication instruction
    pub fn isMul(self: DecodedInstruction) bool {
        if (!self.isMulDiv()) return false;
        const funct3: MulDivFunct3 = @enumFromInt(self.funct3);
        return funct3 == .MUL or funct3 == .MULH or funct3 == .MULHSU or funct3 == .MULHU;
    }

    /// Check if this is a division instruction
    pub fn isDiv(self: DecodedInstruction) bool {
        if (!self.isMulDiv()) return false;
        const funct3: MulDivFunct3 = @enumFromInt(self.funct3);
        return funct3 == .DIV or funct3 == .DIVU or funct3 == .REM or funct3 == .REMU;
    }
};

/// Sign extend a smaller integer to i32
fn signExtend(comptime T: type, value: u32) u32 {
    const bits = @typeInfo(T).int.bits;
    const sign_bit = (value >> (bits - 1)) & 1;
    if (sign_bit == 1) {
        return value | (~@as(u32, 0) << bits);
    }
    return value;
}

test "decode NOP" {
    // NOP is encoded as ADDI x0, x0, 0
    const nop: u32 = 0x00000013;
    const decoded = DecodedInstruction.decode(nop);

    try std.testing.expectEqual(Opcode.OP_IMM, decoded.opcode);
    try std.testing.expectEqual(@as(u5, 0), decoded.rd);
    try std.testing.expectEqual(@as(u5, 0), decoded.rs1);
    try std.testing.expectEqual(@as(i32, 0), decoded.imm);
}

test "decode ADDI" {
    // addi x1, x2, 100
    const addi: u32 = 0x06410093;
    const decoded = DecodedInstruction.decode(addi);

    try std.testing.expectEqual(Opcode.OP_IMM, decoded.opcode);
    try std.testing.expectEqual(@as(u5, 1), decoded.rd);
    try std.testing.expectEqual(@as(u5, 2), decoded.rs1);
    try std.testing.expectEqual(@as(i32, 100), decoded.imm);
}

test "decode LUI" {
    // lui x5, 0x12345
    const lui: u32 = 0x123452b7;
    const decoded = DecodedInstruction.decode(lui);

    try std.testing.expectEqual(Opcode.LUI, decoded.opcode);
    try std.testing.expectEqual(@as(u5, 5), decoded.rd);
    try std.testing.expectEqual(@as(i32, 0x12345000), decoded.imm);
}
