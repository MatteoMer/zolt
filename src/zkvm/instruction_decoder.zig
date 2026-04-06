//! RISC-V instruction decoder for Jolt-compatible format
//!
//! Decodes 32-bit RISC-V instructions into JoltInstruction format with
//! proper immediate encoding for each instruction format (R/I/S/B/U/J).

const std = @import("std");
const jolt_instruction = @import("jolt_instruction.zig");
const JoltInstruction = jolt_instruction.JoltInstruction;
const InstructionVariant = JoltInstruction.InstructionVariant;
const Operands = JoltInstruction.Operands;

/// Decode a 32-bit instruction to JoltInstruction format
pub fn decodeToJoltInstruction(instruction: u32, address: u64, is_compressed: bool) !JoltInstruction {
    const opcode = instruction & 0x7f;
    const rd: u8 = @truncate((instruction >> 7) & 0x1f);
    const funct3: u3 = @truncate((instruction >> 12) & 0x7);
    const rs1: u8 = @truncate((instruction >> 15) & 0x1f);
    const rs2: u8 = @truncate((instruction >> 20) & 0x1f);
    const funct7: u7 = @truncate((instruction >> 25) & 0x7f);

    var variant: InstructionVariant = .UNIMPL;
    var operands: Operands = .{ .None = {} };

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
        0b1110011 => { // SYSTEM
            const imm = decodeIImmediate(instruction);
            switch (funct3) {
                0b001 => { variant = .CSRRW; operands = .{ .FormatI = .{ .rd = rd, .rs1 = rs1, .imm = imm } }; },
                0b010 => { variant = .CSRRS; operands = .{ .FormatI = .{ .rd = rd, .rs1 = rs1, .imm = imm } }; },
                0b000 => {
                    const funct12: u12 = @truncate((instruction >> 20) & 0xFFF);
                    if (funct12 == 0x302) { variant = .MRET; } else { variant = .ECALL; }
                    operands = .{ .FormatI = .{ .rd = rd, .rs1 = rs1, .imm = imm } };
                },
                else => { variant = .ECALL; operands = .{ .FormatI = .{ .rd = rd, .rs1 = rs1, .imm = imm } }; },
            }
        },
        0b0001011 => { // Custom-0 (0x0B): Jolt inline instructions
            // The emulator dispatches on funct3/funct7 to determine the actual inline type.
            // For the decoder, we just parse the FormatInline operands.
            variant = .UNIMPL;
            operands = .{ .FormatInline = .{
                .rs1 = rs1,
                .rs2 = rs2,
                .rs3 = rd, // rd field maps to rs3 in FormatInline
                .funct3 = funct3,
                .funct7 = funct7,
            } };
        },
        0b1011011 => { // Custom-2 (0x5B): Jolt SDK instructions
            const imm = decodeIImmediate(instruction);
            operands = .{ .FormatI = .{ .rd = rd, .rs1 = rs1, .imm = imm } };
            variant = switch (funct3) {
                0b010 => .VirtualHostIO,
                0b011, 0b100, 0b101, 0b110 => .VirtualAdviceLoad,
                0b111 => .VirtualAdviceLen,
                else => .UNIMPL,
            };
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
pub fn decodeLoadImmediate(instruction: u32) i64 {
    const imm: u32 = instruction >> 20;
    const signed: i32 = if (imm & 0x800 != 0)
        @bitCast(imm | 0xFFFFF000)
    else
        @bitCast(imm);
    return @as(i64, signed);
}

/// Decode I-format immediate to u64 (sign-extended from 12-bit signed)
/// Jolt uses u64 for FormatI.imm
pub fn decodeIImmediate(instruction: u32) u64 {
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
pub fn decodeSImmediate(instruction: u32) i64 {
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
pub fn decodeBImmediate(instruction: u32) i128 {
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
pub fn decodeJImmediate(instruction: u32) u64 {
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
pub fn decodeUImmediate(instruction: u32) u64 {
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
    try std.testing.expectEqual(InstructionVariant.ADD, instr.variant);

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
    try std.testing.expectEqual(InstructionVariant.ADDI, instr.variant);

    const json = try instr.toJson(allocator);
    defer allocator.free(json);

    try std.testing.expect(std.mem.indexOf(u8, json, "\"ADDI\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"imm\":100") != null);
}
