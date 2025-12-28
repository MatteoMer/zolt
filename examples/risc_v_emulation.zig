//! RISC-V Emulation Example
//!
//! This example demonstrates the RISC-V instruction decoder:
//! 1. Decode various RISC-V instructions
//! 2. Show the instruction fields

const std = @import("std");
const zolt = @import("zolt");

const DecodedInstruction = zolt.zkvm.instruction.DecodedInstruction;
const Opcode = zolt.zkvm.instruction.Opcode;
const isCompressed = zolt.zkvm.instruction.isCompressed;

pub fn main() !void {
    std.debug.print("=== Zolt RISC-V Instruction Decoder Example ===\n\n", .{});

    // Example instructions to decode
    const instructions = [_]struct { inst: u32, desc: []const u8 }{
        // addi x10, x0, 42  (li a0, 42)
        .{ .inst = 0x02a00513, .desc = "ADDI x10, x0, 42 (load immediate 42)" },
        // addi x11, x0, 10  (li a1, 10)
        .{ .inst = 0x00a00593, .desc = "ADDI x11, x0, 10 (load immediate 10)" },
        // add x12, x10, x11  (add a2, a0, a1)
        .{ .inst = 0x00b50633, .desc = "ADD x12, x10, x11 (add registers)" },
        // lui x5, 0x12345 (load upper immediate)
        .{ .inst = 0x12345297, .desc = "LUI x5, 0x12345 (load upper immediate)" },
        // jal x1, offset (jump and link)
        .{ .inst = 0x000000ef, .desc = "JAL x1, 0 (jump and link)" },
        // beq x1, x2, offset (branch if equal)
        .{ .inst = 0x00208463, .desc = "BEQ x1, x2, 8 (branch if equal)" },
        // lw x3, 0(x4) (load word)
        .{ .inst = 0x00022183, .desc = "LW x3, 0(x4) (load word)" },
        // sw x5, 4(x6) (store word)
        .{ .inst = 0x00532223, .desc = "SW x5, 4(x6) (store word)" },
        // mul x7, x8, x9 (multiply - M extension)
        .{ .inst = 0x029403b3, .desc = "MUL x7, x8, x9 (multiply)" },
        // ecall (environment call)
        .{ .inst = 0x00000073, .desc = "ECALL (environment call)" },
    };

    std.debug.print("Decoding {d} instructions:\n\n", .{instructions.len});

    for (instructions, 0..) |item, i| {
        const decoded = DecodedInstruction.decode(item.inst);

        std.debug.print("[{d}] 0x{x:0>8}: {s}\n", .{ i, item.inst, item.desc });
        std.debug.print("    Opcode: {s}\n", .{@tagName(decoded.opcode)});
        std.debug.print("    rd={d}, rs1={d}, rs2={d}\n", .{
            decoded.rd,
            decoded.rs1,
            decoded.rs2,
        });
        std.debug.print("    funct3={d}, funct7={d}, imm={d}\n\n", .{
            decoded.funct3,
            decoded.funct7,
            decoded.imm,
        });
    }

    // ========================================
    // Compressed Instruction Handling
    // ========================================
    std.debug.print("--- Compressed Instructions (C Extension) ---\n\n", .{});

    const compressed = [_]struct { inst: u16, desc: []const u8 }{
        .{ .inst = 0x0001, .desc = "C.NOP (no operation)" },
        .{ .inst = 0x4501, .desc = "C.LI rd, imm (load immediate)" },
        .{ .inst = 0x0002, .desc = "C.SLLI (shift left logical immediate)" },
    };

    for (compressed) |item| {
        const is_comp = isCompressed(@as(u32, item.inst));
        std.debug.print("0x{x:0>4}: {s}\n", .{ item.inst, item.desc });
        std.debug.print("    Is compressed: {}\n\n", .{is_comp});
    }

    // ========================================
    // Instruction Classification
    // ========================================
    std.debug.print("--- Instruction Categories ---\n\n", .{});

    const categories = [_]struct { opcode: Opcode, desc: []const u8 }{
        .{ .opcode = .OP_IMM, .desc = "Immediate ALU operations (ADDI, SLTI, etc.)" },
        .{ .opcode = .OP, .desc = "Register ALU operations (ADD, SUB, etc.)" },
        .{ .opcode = .LOAD, .desc = "Memory loads (LB, LH, LW, LD)" },
        .{ .opcode = .STORE, .desc = "Memory stores (SB, SH, SW, SD)" },
        .{ .opcode = .BRANCH, .desc = "Conditional branches (BEQ, BNE, etc.)" },
        .{ .opcode = .JAL, .desc = "Jump and link" },
        .{ .opcode = .JALR, .desc = "Jump and link register" },
        .{ .opcode = .LUI, .desc = "Load upper immediate" },
        .{ .opcode = .AUIPC, .desc = "Add upper immediate to PC" },
        .{ .opcode = .SYSTEM, .desc = "System calls (ECALL, EBREAK)" },
    };

    for (categories) |cat| {
        std.debug.print("  {s}: {s}\n", .{ @tagName(cat.opcode), cat.desc });
    }

    std.debug.print("\n=== Example Complete ===\n", .{});
}
