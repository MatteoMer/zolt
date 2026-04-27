//! Run command: Execute a RISC-V ELF binary in the emulator.

const std = @import("std");
const zolt = @import("../root.zig");

pub fn runEmulator(allocator: std.mem.Allocator, io: std.Io, elf_path: []const u8, show_regs: bool, show_trace: bool, max_trace_steps: ?usize, input_bytes: ?[]const u8) !void {
    std.debug.print("Loading ELF: {s}\n", .{elf_path});

    // Load the ELF file
    var loader = zolt.host.ELFLoader.init(allocator);
    const program = loader.loadFile(io, elf_path) catch |err| {
        return err;
    };
    defer {
        var prog = program;
        prog.deinit();
    }

    std.debug.print("Entry point: 0x{x:0>8}\n", .{program.entry_point});
    std.debug.print("Code size: {} bytes\n", .{program.bytecode.len});
    std.debug.print("Base address: 0x{x:0>8}\n", .{program.base_address});

    var config = zolt.common.MemoryConfig{
        .program_size = program.bytecode.len,
        .heap_size = 32768,
    };

    var emulator = zolt.tracer.Emulator.init(allocator, &config);
    defer emulator.deinit();

    try emulator.loadProgramAt(program.bytecode, program.base_address);

    if (input_bytes) |inputs| {
        try emulator.setInputs(inputs);
        std.debug.print("Input bytes: {} bytes\n", .{inputs.len});
        std.debug.print("Input region: 0x{x:0>16} - 0x{x:0>16}\n", .{
            emulator.device.memory_layout.input_start,
            emulator.device.memory_layout.input_end,
        });
    }

    emulator.state.pc = program.entry_point;

    if (show_trace) {
        // Trace mode: run and display execution trace
        std.debug.print("\n", .{});
        var running = true;
        while (running) {
            running = emulator.step() catch |err| {
                std.debug.print("Emulator error during trace: {}\n", .{err});
                break;
            };
        }

        const max_steps = max_trace_steps orelse 100;
        const total_steps = emulator.trace.len();
        const display_steps = @min(total_steps, max_steps);

        std.debug.print("=== Execution Trace ({} of {} steps) ===\n\n", .{ display_steps, total_steps });
        std.debug.print("{s:>6} | {s:>10} | {s:>10} | {s:>12} | {s}\n", .{ "Cycle", "PC", "Instr", "RD Value", "Disasm" });
        std.debug.print("{s:-<6}-+-{s:-<10}-+-{s:-<10}-+-{s:-<12}-+-{s:-<30}\n", .{ "", "", "", "", "" });

        for (0..display_steps) |i| {
            if (emulator.trace.get(i)) |step| {
                const decoded = zolt.zkvm.instruction.DecodedInstruction.decode(step.instruction);
                const mnemonic = blk: {
                    switch (decoded.opcode) {
                        .LUI => break :blk "LUI",
                        .AUIPC => break :blk "AUIPC",
                        .JAL => break :blk "JAL",
                        .JALR => break :blk "JALR",
                        .BRANCH => break :blk "BRANCH",
                        .LOAD => break :blk "LOAD",
                        .STORE => break :blk "STORE",
                        .OP_IMM => break :blk "OP_IMM",
                        .OP => break :blk "OP",
                        .FENCE => break :blk "FENCE",
                        .SYSTEM => break :blk "SYSTEM",
                        .OP_IMM_32 => break :blk "OP_IMM_32",
                        .OP_32 => break :blk "OP_32",
                        _ => break :blk "???",
                    }
                };

                var rd_buf: [16]u8 = undefined;
                const rd_str = if (step.rd_value != 0)
                    std.fmt.bufPrint(&rd_buf, "0x{x:0>8}", .{step.rd_value}) catch "?"
                else
                    std.fmt.bufPrint(&rd_buf, "-", .{}) catch "?";

                var mem_buf: [40]u8 = undefined;
                const mem_str = if (step.memory_addr) |addr| mem_blk: {
                    const mem_val = step.memory_value orelse 0;
                    if (step.is_memory_write) {
                        break :mem_blk std.fmt.bufPrint(&mem_buf, "[0x{x}] <- 0x{x}", .{ addr, mem_val }) catch "";
                    } else {
                        break :mem_blk std.fmt.bufPrint(&mem_buf, "[0x{x}] -> 0x{x}", .{ addr, mem_val }) catch "";
                    }
                } else "";

                std.debug.print("{:>6} | 0x{x:0>8} | 0x{x:0>8} | {s:>12} | {s}", .{
                    step.cycle,
                    step.pc,
                    step.instruction,
                    rd_str,
                    mnemonic,
                });

                switch (decoded.format) {
                    .R => std.debug.print(" x{}, x{}, x{}", .{ decoded.rd, decoded.rs1, decoded.rs2 }),
                    .I => {
                        if (decoded.opcode == .LOAD or decoded.opcode == .JALR) {
                            std.debug.print(" x{}, {}(x{})", .{ decoded.rd, decoded.imm, decoded.rs1 });
                        } else {
                            std.debug.print(" x{}, x{}, {}", .{ decoded.rd, decoded.rs1, decoded.imm });
                        }
                    },
                    .S => std.debug.print(" x{}, {}(x{})", .{ decoded.rs2, decoded.imm, decoded.rs1 }),
                    .B => std.debug.print(" x{}, x{}, {}", .{ decoded.rs1, decoded.rs2, decoded.imm }),
                    .U, .J => std.debug.print(" x{}, 0x{x}", .{ decoded.rd, @as(u32, @bitCast(decoded.imm)) }),
                }

                if (mem_str.len > 0) {
                    std.debug.print("  ; {s}", .{mem_str});
                }

                std.debug.print("\n", .{});
            }
        }

        if (total_steps > max_steps) {
            std.debug.print("\n... {} more steps (use --max N to show more)\n", .{total_steps - max_steps});
        }

        std.debug.print("\n====================\n", .{});
        std.debug.print("Total cycles: {}\n", .{emulator.state.cycle});
        std.debug.print("Final PC: 0x{x:0>8}\n", .{emulator.state.pc});
    } else {
        // Normal run mode
        std.debug.print("\nStarting execution...\n", .{});

        emulator.run() catch |err| {
            std.debug.print("Execution stopped: {}\n", .{err});
        };

        std.debug.print("\nExecution complete!\n", .{});
        std.debug.print("Cycles executed: {}\n", .{emulator.state.cycle});
        std.debug.print("Final PC: 0x{x:0>8}\n", .{emulator.state.pc});
        std.debug.print("Trace entries: {}\n", .{emulator.trace.len()});

        if (show_regs) {
            std.debug.print("\nFinal Register State:\n", .{});
            var reg_i: u8 = 0;
            while (reg_i < 32) : (reg_i += 1) {
                const val = emulator.registers.read(reg_i) catch |err| {
                    std.debug.print("  x{d:0>2}: error reading register: {}\n", .{ reg_i, err });
                    continue;
                };
                if (val != 0) {
                    const reg_name = switch (reg_i) {
                        0 => "zero",
                        1 => "ra  ",
                        2 => "sp  ",
                        3 => "gp  ",
                        4 => "tp  ",
                        5 => "t0  ",
                        6 => "t1  ",
                        7 => "t2  ",
                        8 => "s0  ",
                        9 => "s1  ",
                        10 => "a0  ",
                        11 => "a1  ",
                        12 => "a2  ",
                        13 => "a3  ",
                        14 => "a4  ",
                        15 => "a5  ",
                        16 => "a6  ",
                        17 => "a7  ",
                        18 => "s2  ",
                        19 => "s3  ",
                        20 => "s4  ",
                        21 => "s5  ",
                        22 => "s6  ",
                        23 => "s7  ",
                        24 => "s8  ",
                        25 => "s9  ",
                        26 => "s10 ",
                        27 => "s11 ",
                        28 => "t3  ",
                        29 => "t4  ",
                        30 => "t5  ",
                        31 => "t6  ",
                        else => "x?? ",
                    };
                    std.debug.print("  x{d:0>2} ({s}): 0x{x:0>16} ({d})\n", .{ reg_i, reg_name, val, val });
                }
            }
        }
    }
}
