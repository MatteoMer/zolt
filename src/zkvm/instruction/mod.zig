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

// ============================================================================
// RISC-V C Extension (Compressed Instructions) Support
// ============================================================================

/// XLEN enumeration for 32-bit or 64-bit mode
pub const Xlen = enum {
    Bit32,
    Bit64,
};

/// Check if an instruction is compressed (16-bit)
/// Compressed instructions have bits [1:0] != 0b11
pub fn isCompressed(instruction: u32) bool {
    return (instruction & 0x3) != 0x3;
}

/// Uncompress a 16-bit compressed instruction to its 32-bit equivalent
/// Returns 0xffffffff for invalid/reserved instructions
pub fn uncompressInstruction(halfword: u32, xlen: Xlen) u32 {
    const op = halfword & 0x3; // [1:0]
    const funct3 = (halfword >> 13) & 0x7; // [15:13]

    switch (op) {
        0 => {
            // Quadrant 0 instructions
            return uncompressQ0(halfword, funct3);
        },
        1 => {
            // Quadrant 1 instructions
            return uncompressQ1(halfword, funct3, xlen);
        },
        2 => {
            // Quadrant 2 instructions
            return uncompressQ2(halfword, funct3);
        },
        else => return 0xffffffff, // op == 3 means 32-bit instruction
    }
}

/// Uncompress Quadrant 0 instructions
fn uncompressQ0(halfword: u32, funct3: u32) u32 {
    switch (funct3) {
        0 => {
            // C.ADDI4SPN: addi rd+8, x2, nzuimm
            const rd = (halfword >> 2) & 0x7; // [4:2]
            const nzuimm = ((halfword >> 7) & 0x30) | // nzuimm[5:4] <= [12:11]
                ((halfword >> 1) & 0x3c0) | // nzuimm[9:6] <= [10:7]
                ((halfword >> 4) & 0x4) | // nzuimm[2] <= [6]
                ((halfword >> 2) & 0x8); // nzuimm[3] <= [5]
            if (nzuimm != 0) {
                return (nzuimm << 20) | (2 << 15) | ((rd + 8) << 7) | 0x13;
            }
            return 0xffffffff; // Reserved
        },
        1 => {
            // C.FLD: fld rd+8, offset(rs1+8)
            const rd = (halfword >> 2) & 0x7;
            const rs1 = (halfword >> 7) & 0x7;
            const offset = ((halfword >> 7) & 0x38) | // offset[5:3] <= [12:10]
                ((halfword << 1) & 0xc0); // offset[7:6] <= [6:5]
            return (offset << 20) | ((rs1 + 8) << 15) | (3 << 12) | ((rd + 8) << 7) | 0x7;
        },
        2 => {
            // C.LW: lw rd+8, offset(rs1+8)
            const rs1 = (halfword >> 7) & 0x7;
            const rd = (halfword >> 2) & 0x7;
            const offset = ((halfword >> 7) & 0x38) | // offset[5:3] <= [12:10]
                ((halfword >> 4) & 0x4) | // offset[2] <= [6]
                ((halfword << 1) & 0x40); // offset[6] <= [5]
            return (offset << 20) | ((rs1 + 8) << 15) | (2 << 12) | ((rd + 8) << 7) | 0x3;
        },
        3 => {
            // C.LD: ld rd+8, offset(rs1+8) (RV64C)
            const rs1 = (halfword >> 7) & 0x7;
            const rd = (halfword >> 2) & 0x7;
            const offset = ((halfword >> 7) & 0x38) | // offset[5:3] <= [12:10]
                ((halfword << 1) & 0xc0); // offset[7:6] <= [6:5]
            return (offset << 20) | ((rs1 + 8) << 15) | (3 << 12) | ((rd + 8) << 7) | 0x3;
        },
        4 => {
            // Reserved
            return 0xffffffff;
        },
        5 => {
            // C.FSD: fsd rs2+8, offset(rs1+8)
            const rs1 = (halfword >> 7) & 0x7;
            const rs2 = (halfword >> 2) & 0x7;
            const offset = ((halfword >> 7) & 0x38) | // uimm[5:3] <= [12:10]
                ((halfword << 1) & 0xc0); // uimm[7:6] <= [6:5]
            const imm11_5 = (offset >> 5) & 0x7f;
            const imm4_0 = offset & 0x1f;
            return (imm11_5 << 25) | ((rs2 + 8) << 20) | ((rs1 + 8) << 15) | (3 << 12) | (imm4_0 << 7) | 0x27;
        },
        6 => {
            // C.SW: sw rs2+8, offset(rs1+8)
            const rs1 = (halfword >> 7) & 0x7;
            const rs2 = (halfword >> 2) & 0x7;
            const offset = ((halfword >> 7) & 0x38) | // offset[5:3] <= [12:10]
                ((halfword << 1) & 0x40) | // offset[6] <= [5]
                ((halfword >> 4) & 0x4); // offset[2] <= [6]
            const imm11_5 = (offset >> 5) & 0x7f;
            const imm4_0 = offset & 0x1f;
            return (imm11_5 << 25) | ((rs2 + 8) << 20) | ((rs1 + 8) << 15) | (2 << 12) | (imm4_0 << 7) | 0x23;
        },
        7 => {
            // C.SD: sd rs2+8, offset(rs1+8) (RV64C)
            const rs1 = (halfword >> 7) & 0x7;
            const rs2 = (halfword >> 2) & 0x7;
            const offset = ((halfword >> 7) & 0x38) | // uimm[5:3] <= [12:10]
                ((halfword << 1) & 0xc0); // uimm[7:6] <= [6:5]
            const imm11_5 = (offset >> 5) & 0x7f;
            const imm4_0 = offset & 0x1f;
            return (imm11_5 << 25) | ((rs2 + 8) << 20) | ((rs1 + 8) << 15) | (3 << 12) | (imm4_0 << 7) | 0x23;
        },
        else => return 0xffffffff,
    }
}

/// Uncompress Quadrant 1 instructions
fn uncompressQ1(halfword: u32, funct3: u32, xlen: Xlen) u32 {
    switch (funct3) {
        0 => {
            // C.ADDI / C.NOP
            const r = (halfword >> 7) & 0x1f; // [11:7]
            const imm = computeSignedImm6(halfword);

            if (r == 0 or imm == 0) {
                // NOP or HINT
                return 0x13;
            }
            return (imm << 20) | (r << 15) | (r << 7) | 0x13;
        },
        1 => {
            switch (xlen) {
                .Bit32 => {
                    // C.JAL (RV32C only): jal x1, offset
                    const offset = computeJumpOffset(halfword);
                    const imm = ((offset >> 1) & 0x80000) | // imm[19] <= offset[20]
                        ((offset << 8) & 0x7fe00) | // imm[18:9] <= offset[10:1]
                        ((offset >> 3) & 0x100) | // imm[8] <= offset[11]
                        ((offset >> 12) & 0xff); // imm[7:0] <= offset[19:12]
                    return (imm << 12) | (1 << 7) | 0x6f;
                },
                .Bit64 => {
                    // C.ADDIW (RV64C only)
                    const r = (halfword >> 7) & 0x1f;
                    const imm = computeSignedImm6(halfword);
                    if (r == 0) {
                        return 0xffffffff; // Reserved
                    } else if (imm == 0) {
                        // sext.w rd
                        return (r << 15) | (r << 7) | 0x1b;
                    } else {
                        // addiw r, r, imm
                        return (imm << 20) | (r << 15) | (r << 7) | 0x1b;
                    }
                },
            }
        },
        2 => {
            // C.LI: addi rd, x0, imm
            const r = (halfword >> 7) & 0x1f;
            const imm = computeSignedImm6(halfword);
            if (r != 0) {
                return (imm << 20) | (r << 7) | 0x13;
            }
            return 0x13; // HINT
        },
        3 => {
            const r = (halfword >> 7) & 0x1f; // [11:7]
            if (r == 2) {
                // C.ADDI16SP: addi sp, sp, nzimm
                const imm = computeAddi16spImm(halfword);
                if (imm != 0) {
                    return (imm << 20) | (r << 15) | (r << 7) | 0x13;
                }
                return 0xffffffff; // Reserved
            }
            if (r != 0) {
                // C.LUI: lui r, nzimm
                const nzimm = computeLuiImm(halfword);
                if (nzimm != 0) {
                    return nzimm | (r << 7) | 0x37;
                }
                return 0xffffffff; // Reserved
            }
            return 0x13; // NOP
        },
        4 => {
            return uncompressQ1Funct4(halfword);
        },
        5 => {
            // C.J: jal x0, imm
            const offset = computeJumpOffset(halfword);
            const imm = ((offset >> 1) & 0x80000) | // imm[19] <= offset[20]
                ((offset << 8) & 0x7fe00) | // imm[18:9] <= offset[10:1]
                ((offset >> 3) & 0x100) | // imm[8] <= offset[11]
                ((offset >> 12) & 0xff); // imm[7:0] <= offset[19:12]
            return (imm << 12) | 0x6f;
        },
        6 => {
            // C.BEQZ: beq r+8, x0, offset
            const r = (halfword >> 7) & 0x7;
            const offset = computeBranchOffset(halfword);
            const imm2 = ((offset >> 6) & 0x40) | ((offset >> 5) & 0x3f);
            const imm1 = (offset & 0x1e) | ((offset >> 11) & 0x1);
            return (imm2 << 25) | ((r + 8) << 20) | (imm1 << 7) | 0x63;
        },
        7 => {
            // C.BNEZ: bne r+8, x0, offset
            const r = (halfword >> 7) & 0x7;
            const offset = computeBranchOffset(halfword);
            const imm2 = ((offset >> 6) & 0x40) | ((offset >> 5) & 0x3f);
            const imm1 = (offset & 0x1e) | ((offset >> 11) & 0x1);
            return (imm2 << 25) | ((r + 8) << 20) | (1 << 12) | (imm1 << 7) | 0x63;
        },
        else => return 0xffffffff,
    }
}

/// Uncompress Q1 funct3=4 instructions (ALU operations)
fn uncompressQ1Funct4(halfword: u32) u32 {
    const funct2 = (halfword >> 10) & 0x3; // [11:10]
    switch (funct2) {
        0 => {
            // C.SRLI: srli rs1+8, rs1+8, shamt
            const shamt = ((halfword >> 7) & 0x20) | ((halfword >> 2) & 0x1f);
            const rs1 = (halfword >> 7) & 0x7;
            return (shamt << 20) | ((rs1 + 8) << 15) | (5 << 12) | ((rs1 + 8) << 7) | 0x13;
        },
        1 => {
            // C.SRAI: srai rs1+8, rs1+8, shamt
            const shamt = ((halfword >> 7) & 0x20) | ((halfword >> 2) & 0x1f);
            const rs1 = (halfword >> 7) & 0x7;
            return (0x20 << 25) | (shamt << 20) | ((rs1 + 8) << 15) | (5 << 12) | ((rs1 + 8) << 7) | 0x13;
        },
        2 => {
            // C.ANDI: andi r+8, r+8, imm
            const r = (halfword >> 7) & 0x7;
            const imm = computeSignedImm6(halfword);
            return (imm << 20) | ((r + 8) << 15) | (7 << 12) | ((r + 8) << 7) | 0x13;
        },
        3 => {
            const funct1 = (halfword >> 12) & 1; // [12]
            const funct2_2 = (halfword >> 5) & 0x3; // [6:5]
            const rs1 = (halfword >> 7) & 0x7;
            const rs2 = (halfword >> 2) & 0x7;
            switch (funct1) {
                0 => {
                    switch (funct2_2) {
                        0 => {
                            // C.SUB: sub rs1+8, rs1+8, rs2+8
                            return (0x20 << 25) | ((rs2 + 8) << 20) | ((rs1 + 8) << 15) | ((rs1 + 8) << 7) | 0x33;
                        },
                        1 => {
                            // C.XOR: xor rs1+8, rs1+8, rs2+8
                            return ((rs2 + 8) << 20) | ((rs1 + 8) << 15) | (4 << 12) | ((rs1 + 8) << 7) | 0x33;
                        },
                        2 => {
                            // C.OR: or rs1+8, rs1+8, rs2+8
                            return ((rs2 + 8) << 20) | ((rs1 + 8) << 15) | (6 << 12) | ((rs1 + 8) << 7) | 0x33;
                        },
                        3 => {
                            // C.AND: and rs1+8, rs1+8, rs2+8
                            return ((rs2 + 8) << 20) | ((rs1 + 8) << 15) | (7 << 12) | ((rs1 + 8) << 7) | 0x33;
                        },
                    }
                },
                1 => {
                    switch (funct2_2) {
                        0 => {
                            // C.SUBW: subw r1+8, r1+8, r2+8
                            return (0x20 << 25) | ((rs2 + 8) << 20) | ((rs1 + 8) << 15) | ((rs1 + 8) << 7) | 0x3b;
                        },
                        1 => {
                            // C.ADDW: addw r1+8, r1+8, r2+8
                            return ((rs2 + 8) << 20) | ((rs1 + 8) << 15) | ((rs1 + 8) << 7) | 0x3b;
                        },
                        else => return 0xffffffff, // Reserved
                    }
                },
            }
        },
    }
}

/// Uncompress Quadrant 2 instructions
fn uncompressQ2(halfword: u32, funct3: u32) u32 {
    switch (funct3) {
        0 => {
            // C.SLLI: slli r, r, shamt
            const r = (halfword >> 7) & 0x1f;
            const shamt = ((halfword >> 7) & 0x20) | ((halfword >> 2) & 0x1f);
            if (r != 0) {
                return (shamt << 20) | (r << 15) | (1 << 12) | (r << 7) | 0x13;
            }
            return 0xffffffff; // Reserved
        },
        1 => {
            // C.FLDSP: fld rd, offset(x2)
            const rd = (halfword >> 7) & 0x1f;
            const offset = ((halfword >> 7) & 0x20) | // offset[5] <= [12]
                ((halfword >> 2) & 0x18) | // offset[4:3] <= [6:5]
                ((halfword << 4) & 0x1c0); // offset[8:6] <= [4:2]
            if (rd != 0) {
                return (offset << 20) | (2 << 15) | (3 << 12) | (rd << 7) | 0x7;
            }
            return 0xffffffff; // Reserved
        },
        2 => {
            // C.LWSP: lw r, offset(x2)
            const r = (halfword >> 7) & 0x1f;
            const offset = ((halfword >> 7) & 0x20) | // offset[5] <= [12]
                ((halfword >> 2) & 0x1c) | // offset[4:2] <= [6:4]
                ((halfword << 4) & 0xc0); // offset[7:6] <= [3:2]
            if (r != 0) {
                return (offset << 20) | (2 << 15) | (2 << 12) | (r << 7) | 0x3;
            }
            return 0xffffffff; // Reserved
        },
        3 => {
            // C.LDSP: ld rd, offset(x2) (RV64C)
            const rd = (halfword >> 7) & 0x1f;
            const offset = ((halfword >> 7) & 0x20) | // offset[5] <= [12]
                ((halfword >> 2) & 0x18) | // offset[4:3] <= [6:5]
                ((halfword << 4) & 0x1c0); // offset[8:6] <= [4:2]
            if (rd != 0) {
                return (offset << 20) | (2 << 15) | (3 << 12) | (rd << 7) | 0x3;
            }
            return 0xffffffff; // Reserved
        },
        4 => {
            const funct1 = (halfword >> 12) & 1; // [12]
            const rs1 = (halfword >> 7) & 0x1f; // [11:7]
            const rs2 = (halfword >> 2) & 0x1f; // [6:2]
            switch (funct1) {
                0 => {
                    if (rs1 == 0 and rs2 == 0) {
                        return 0xffffffff; // Reserved
                    } else if (rs2 == 0) {
                        // C.JR: jalr x0, 0(rs1)
                        return (rs1 << 15) | 0x67;
                    } else if (rs1 == 0) {
                        return 0x13; // HINT
                    } else {
                        // C.MV: add rd, x0, rs2
                        return (rs2 << 20) | (rs1 << 7) | 0x33;
                    }
                },
                1 => {
                    if (rs1 == 0 and rs2 == 0) {
                        // C.EBREAK
                        return 0x00100073;
                    } else if (rs2 == 0) {
                        // C.JALR: jalr x1, 0(rs1)
                        return (rs1 << 15) | (1 << 7) | 0x67;
                    } else if (rs1 == 0) {
                        return 0x13; // HINT
                    } else {
                        // C.ADD: add rs1, rs1, rs2
                        return (rs2 << 20) | (rs1 << 15) | (rs1 << 7) | 0x33;
                    }
                },
            }
        },
        5 => {
            // C.FSDSP: fsd rs2, offset(x2)
            const rs2 = (halfword >> 2) & 0x1f;
            const offset = ((halfword >> 7) & 0x38) | // offset[5:3] <= [12:10]
                ((halfword >> 1) & 0x1c0); // offset[8:6] <= [9:7]
            const imm11_5 = (offset >> 5) & 0x3f;
            const imm4_0 = offset & 0x1f;
            return (imm11_5 << 25) | (rs2 << 20) | (2 << 15) | (3 << 12) | (imm4_0 << 7) | 0x27;
        },
        6 => {
            // C.SWSP: sw rs2, offset(x2)
            const rs2 = (halfword >> 2) & 0x1f;
            const offset = ((halfword >> 7) & 0x3c) | // offset[5:2] <= [12:9]
                ((halfword >> 1) & 0xc0); // offset[7:6] <= [8:7]
            const imm11_5 = (offset >> 5) & 0x3f;
            const imm4_0 = offset & 0x1f;
            return (imm11_5 << 25) | (rs2 << 20) | (2 << 15) | (2 << 12) | (imm4_0 << 7) | 0x23;
        },
        7 => {
            // C.SDSP: sd rs, offset(x2) (RV64C)
            const rs2 = (halfword >> 2) & 0x1f;
            const offset = ((halfword >> 7) & 0x38) | // offset[5:3] <= [12:10]
                ((halfword >> 1) & 0x1c0); // offset[8:6] <= [9:7]
            const imm11_5 = (offset >> 5) & 0x3f;
            const imm4_0 = offset & 0x1f;
            return (imm11_5 << 25) | (rs2 << 20) | (2 << 15) | (3 << 12) | (imm4_0 << 7) | 0x23;
        },
        else => return 0xffffffff,
    }
}

// Helper functions for immediate computation

fn computeSignedImm6(halfword: u32) u32 {
    const base: u32 = if ((halfword & 0x1000) != 0) 0xffffffc0 else 0;
    return base |
        ((halfword >> 7) & 0x20) | // imm[5] <= [12]
        ((halfword >> 2) & 0x1f); // imm[4:0] <= [6:2]
}

fn computeJumpOffset(halfword: u32) u32 {
    const base: u32 = if ((halfword & 0x1000) != 0) 0xfffff000 else 0;
    return base |
        ((halfword >> 1) & 0x800) | // offset[11] <= [12]
        ((halfword >> 7) & 0x10) | // offset[4] <= [11]
        ((halfword >> 1) & 0x300) | // offset[9:8] <= [10:9]
        ((halfword << 2) & 0x400) | // offset[10] <= [8]
        ((halfword >> 1) & 0x40) | // offset[6] <= [7]
        ((halfword << 1) & 0x80) | // offset[7] <= [6]
        ((halfword >> 2) & 0xe) | // offset[3:1] <= [5:3]
        ((halfword << 3) & 0x20); // offset[5] <= [2]
}

fn computeBranchOffset(halfword: u32) u32 {
    const base: u32 = if ((halfword & 0x1000) != 0) 0xfffffe00 else 0;
    return base |
        ((halfword >> 4) & 0x100) | // offset[8] <= [12]
        ((halfword >> 7) & 0x18) | // offset[4:3] <= [11:10]
        ((halfword << 1) & 0xc0) | // offset[7:6] <= [6:5]
        ((halfword >> 2) & 0x6) | // offset[2:1] <= [4:3]
        ((halfword << 3) & 0x20); // offset[5] <= [2]
}

fn computeAddi16spImm(halfword: u32) u32 {
    const base: u32 = if ((halfword & 0x1000) != 0) 0xfffffc00 else 0;
    return base |
        ((halfword >> 3) & 0x200) | // imm[9] <= [12]
        ((halfword >> 2) & 0x10) | // imm[4] <= [6]
        ((halfword << 1) & 0x40) | // imm[6] <= [5]
        ((halfword << 4) & 0x180) | // imm[8:7] <= [4:3]
        ((halfword << 3) & 0x20); // imm[5] <= [2]
}

fn computeLuiImm(halfword: u32) u32 {
    const base: u32 = if ((halfword & 0x1000) != 0) 0xfffc0000 else 0;
    return base |
        ((halfword << 5) & 0x20000) | // nzimm[17] <= [12]
        ((halfword << 10) & 0x1f000); // nzimm[16:12] <= [6:2]
}

// ============================================================================
// Compressed Instruction Tests
// ============================================================================

test "isCompressed" {
    // Non-compressed instruction (bits [1:0] == 0b11)
    try std.testing.expect(!isCompressed(0x00000013)); // NOP
    try std.testing.expect(!isCompressed(0x123452b7)); // LUI

    // Compressed instructions (bits [1:0] != 0b11)
    try std.testing.expect(isCompressed(0x0020)); // C.ADDI4SPN like
    try std.testing.expect(isCompressed(0x0001)); // op = 01
    try std.testing.expect(isCompressed(0x0002)); // op = 10
}

test "uncompress C.ADDI" {
    // C.ADDI x8, 8 -> addi x8, x8, 8
    // Encoding: 0b0000_0100_0010_0001 = 0x0421
    // Actually test a known working encoding
    const compressed: u32 = 0x0421; // C.ADDI
    const expanded = uncompressInstruction(compressed, .Bit64);
    // Should expand to an ADDI instruction
    try std.testing.expect((expanded & 0x7f) == 0x13); // ADDI opcode
}

test "uncompress C.LW" {
    // C.LW rd', offset(rs1') expands to lw rd', offset(rs1')
    // Test basic structure
    const compressed: u32 = 0x4000; // C.LW with rd=x8, rs1=x8, offset=0
    const expanded = uncompressInstruction(compressed, .Bit64);
    // Should be a load word instruction
    try std.testing.expect((expanded & 0x7f) == 0x03); // LOAD opcode
    try std.testing.expect(((expanded >> 12) & 0x7) == 2); // funct3 = LW
}

test "uncompress C.J" {
    // C.J offset -> jal x0, offset
    // funct3 = 5, op = 1
    const compressed: u32 = 0xa001; // C.J with small offset
    const expanded = uncompressInstruction(compressed, .Bit64);
    // Should expand to JAL x0
    try std.testing.expect((expanded & 0x7f) == 0x6f); // JAL opcode
    try std.testing.expect(((expanded >> 7) & 0x1f) == 0); // rd = x0
}

test "uncompress C.LWSP" {
    // C.LWSP rd, offset(sp) -> lw rd, offset(x2)
    // funct3 = 2, op = 2
    const compressed: u32 = 0x4082; // C.LWSP x1, 0(sp)
    const expanded = uncompressInstruction(compressed, .Bit64);
    // Should be a load word instruction from sp
    try std.testing.expect((expanded & 0x7f) == 0x03); // LOAD opcode
    try std.testing.expect(((expanded >> 15) & 0x1f) == 2); // rs1 = x2 (sp)
}

test "uncompress C.NOP" {
    // C.NOP -> addi x0, x0, 0 (NOP)
    const compressed: u32 = 0x0001; // C.NOP
    const expanded = uncompressInstruction(compressed, .Bit64);
    try std.testing.expectEqual(@as(u32, 0x13), expanded); // NOP encoding
}
