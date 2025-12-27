//! RISC-V execution tracer for Jolt
//!
//! This module traces the execution of RISC-V programs to generate
//! the execution trace needed for proving.

const std = @import("std");
const Allocator = std.mem.Allocator;
const common = @import("../common/mod.zig");
const zkvm = @import("../zkvm/mod.zig");

/// A single step in the execution trace
pub const TraceStep = struct {
    /// Cycle number
    cycle: u64,
    /// Program counter before execution
    pc: u64,
    /// Instruction executed
    instruction: u32,
    /// Source register 1 value
    rs1_value: u64,
    /// Source register 2 value
    rs2_value: u64,
    /// Destination register value (after execution)
    rd_value: u64,
    /// Memory address accessed (if any)
    memory_addr: ?u64,
    /// Memory value (if any)
    memory_value: ?u64,
    /// Whether this is a memory write
    is_memory_write: bool,
    /// Next PC
    next_pc: u64,
};

/// Full execution trace
pub const ExecutionTrace = struct {
    steps: std.ArrayList(TraceStep),
    allocator: Allocator,

    pub fn init(allocator: Allocator) ExecutionTrace {
        return .{
            .steps = std.ArrayList(TraceStep).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *ExecutionTrace) void {
        self.steps.deinit();
    }

    /// Add a step to the trace
    pub fn addStep(self: *ExecutionTrace, step: TraceStep) !void {
        try self.steps.append(step);
    }

    /// Get the number of steps
    pub fn len(self: *const ExecutionTrace) usize {
        return self.steps.items.len;
    }

    /// Get step at index
    pub fn get(self: *const ExecutionTrace, index: usize) ?TraceStep {
        if (index >= self.steps.items.len) return null;
        return self.steps.items[index];
    }
};

/// RISC-V emulator for tracing
pub const Emulator = struct {
    /// Current VM state
    state: zkvm.VMState,
    /// Memory
    ram: zkvm.ram.RAMState,
    /// Register file with tracing
    registers: zkvm.registers.RegisterFile,
    /// I/O device
    device: common.JoltDevice,
    /// Execution trace
    trace: ExecutionTrace,
    /// Maximum cycles to execute
    max_cycles: u64,
    allocator: Allocator,

    pub fn init(allocator: Allocator, config: *const common.MemoryConfig) Emulator {
        return .{
            .state = zkvm.VMState.init(common.constants.RAM_START_ADDRESS),
            .ram = zkvm.ram.RAMState.init(allocator),
            .registers = zkvm.registers.RegisterFile.init(allocator),
            .device = common.JoltDevice.init(allocator, config),
            .trace = ExecutionTrace.init(allocator),
            .max_cycles = common.constants.DEFAULT_MAX_TRACE_LENGTH,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Emulator) void {
        self.ram.deinit();
        self.registers.deinit();
        self.device.deinit();
        self.trace.deinit();
    }

    /// Load a program into memory
    pub fn loadProgram(self: *Emulator, bytecode: []const u8) !void {
        var addr: u64 = common.constants.RAM_START_ADDRESS;
        for (bytecode) |byte| {
            try self.ram.writeByte(addr, byte, 0);
            addr += 1;
        }
    }

    /// Set input data
    pub fn setInputs(self: *Emulator, inputs: []const u8) !void {
        try self.device.inputs.appendSlice(inputs);
    }

    /// Execute a single instruction
    pub fn step(self: *Emulator) !bool {
        if (self.state.cycle >= self.max_cycles) {
            return false; // Max cycles reached
        }

        // Fetch instruction
        const instruction = try self.fetchInstruction();
        self.state.instruction = instruction;

        // Decode
        const decoded = zkvm.instruction.DecodedInstruction.decode(instruction);

        // Record pre-execution state
        const rs1_value = try self.registers.read(decoded.rs1);
        const rs2_value = try self.registers.read(decoded.rs2);

        // Execute
        const result = try self.execute(decoded, rs1_value, rs2_value);

        // Record trace step
        try self.trace.addStep(.{
            .cycle = self.state.cycle,
            .pc = self.state.pc,
            .instruction = instruction,
            .rs1_value = rs1_value,
            .rs2_value = rs2_value,
            .rd_value = result.rd_value,
            .memory_addr = result.memory_addr,
            .memory_value = result.memory_value,
            .is_memory_write = result.is_memory_write,
            .next_pc = result.next_pc,
        });

        // Update state
        self.state.pc = result.next_pc;
        self.state.cycle += 1;
        self.registers.tick();

        return true;
    }

    /// Run until completion or max cycles
    pub fn run(self: *Emulator) !void {
        while (try self.step()) {}
    }

    /// Fetch instruction from memory
    fn fetchInstruction(self: *Emulator) !u32 {
        var instruction: u32 = 0;
        inline for (0..4) |i| {
            const byte = try self.ram.readByte(self.state.pc + i, self.state.cycle);
            instruction |= @as(u32, byte) << (@as(u5, @intCast(i)) * 8);
        }
        return instruction;
    }

    const ExecutionResult = struct {
        rd_value: u64,
        memory_addr: ?u64,
        memory_value: ?u64,
        is_memory_write: bool,
        next_pc: u64,
    };

    /// Execute a decoded instruction
    fn execute(
        self: *Emulator,
        decoded: zkvm.instruction.DecodedInstruction,
        rs1: u64,
        rs2: u64,
    ) !ExecutionResult {
        var result = ExecutionResult{
            .rd_value = 0,
            .memory_addr = null,
            .memory_value = null,
            .is_memory_write = false,
            .next_pc = self.state.pc + 4,
        };

        switch (decoded.opcode) {
            .LUI => {
                result.rd_value = @bitCast(@as(i64, decoded.imm));
                try self.registers.write(decoded.rd, result.rd_value);
            },
            .AUIPC => {
                result.rd_value = @bitCast(@as(i64, @as(i32, @intCast(self.state.pc))) + decoded.imm);
                try self.registers.write(decoded.rd, result.rd_value);
            },
            .JAL => {
                result.rd_value = self.state.pc + 4;
                result.next_pc = @bitCast(@as(i64, @as(i32, @intCast(self.state.pc))) + decoded.imm);
                try self.registers.write(decoded.rd, result.rd_value);
            },
            .JALR => {
                result.rd_value = self.state.pc + 4;
                const target = (@as(i64, @bitCast(rs1)) + decoded.imm) & ~@as(i64, 1);
                result.next_pc = @bitCast(target);
                try self.registers.write(decoded.rd, result.rd_value);
            },
            .BRANCH => {
                const taken = switch (@as(zkvm.instruction.BranchFunct3, @enumFromInt(decoded.funct3))) {
                    .BEQ => rs1 == rs2,
                    .BNE => rs1 != rs2,
                    .BLT => @as(i64, @bitCast(rs1)) < @as(i64, @bitCast(rs2)),
                    .BGE => @as(i64, @bitCast(rs1)) >= @as(i64, @bitCast(rs2)),
                    .BLTU => rs1 < rs2,
                    .BGEU => rs1 >= rs2,
                    _ => false,
                };
                if (taken) {
                    result.next_pc = @bitCast(@as(i64, @as(i32, @intCast(self.state.pc))) + decoded.imm);
                }
            },
            .LOAD => {
                const addr: u64 = @bitCast(@as(i64, @bitCast(rs1)) + decoded.imm);
                result.memory_addr = addr;

                const value = switch (@as(zkvm.instruction.LoadFunct3, @enumFromInt(decoded.funct3))) {
                    .LB => blk: {
                        const byte = try self.ram.readByte(addr, self.state.cycle);
                        break :blk @as(u64, @bitCast(@as(i64, @as(i8, @bitCast(byte)))));
                    },
                    .LBU => try self.ram.readByte(addr, self.state.cycle),
                    .LH => blk: {
                        const low = try self.ram.readByte(addr, self.state.cycle);
                        const high = try self.ram.readByte(addr + 1, self.state.cycle);
                        const halfword: i16 = @bitCast((@as(u16, high) << 8) | low);
                        break :blk @as(u64, @bitCast(@as(i64, halfword)));
                    },
                    .LW => blk: {
                        const word = try self.ram.read(addr & ~@as(u64, 3), self.state.cycle);
                        const signed: i32 = @truncate(@as(i64, @bitCast(word)));
                        break :blk @as(u64, @bitCast(@as(i64, signed)));
                    },
                    .LD => try self.ram.read(addr & ~@as(u64, 7), self.state.cycle),
                    else => 0,
                };

                result.rd_value = value;
                result.memory_value = value;
                try self.registers.write(decoded.rd, result.rd_value);
            },
            .STORE => {
                const addr: u64 = @bitCast(@as(i64, @bitCast(rs1)) + decoded.imm);
                result.memory_addr = addr;
                result.is_memory_write = true;

                switch (@as(zkvm.instruction.StoreFunct3, @enumFromInt(decoded.funct3))) {
                    .SB => {
                        try self.ram.writeByte(addr, @truncate(rs2), self.state.cycle);
                        result.memory_value = rs2 & 0xFF;
                    },
                    .SH => {
                        try self.ram.writeByte(addr, @truncate(rs2), self.state.cycle);
                        try self.ram.writeByte(addr + 1, @truncate(rs2 >> 8), self.state.cycle);
                        result.memory_value = rs2 & 0xFFFF;
                    },
                    .SW => {
                        try self.ram.write(addr & ~@as(u64, 3), rs2 & 0xFFFFFFFF, self.state.cycle);
                        result.memory_value = rs2 & 0xFFFFFFFF;
                    },
                    .SD => {
                        try self.ram.write(addr & ~@as(u64, 7), rs2, self.state.cycle);
                        result.memory_value = rs2;
                    },
                    _ => {},
                }
            },
            .OP_IMM => {
                const imm_u64: u64 = @bitCast(@as(i64, decoded.imm));
                result.rd_value = switch (@as(zkvm.instruction.OpImmFunct3, @enumFromInt(decoded.funct3))) {
                    .ADDI => rs1 +% imm_u64,
                    .SLTI => if (@as(i64, @bitCast(rs1)) < decoded.imm) 1 else 0,
                    .SLTIU => if (rs1 < imm_u64) 1 else 0,
                    .XORI => rs1 ^ imm_u64,
                    .ORI => rs1 | imm_u64,
                    .ANDI => rs1 & imm_u64,
                    .SLLI => rs1 << @truncate(decoded.imm & 0x3F),
                    .SRLI_SRAI => blk: {
                        const shamt: u6 = @truncate(decoded.imm & 0x3F);
                        if ((decoded.funct7 & 0x20) != 0) {
                            // SRAI
                            break :blk @bitCast(@as(i64, @bitCast(rs1)) >> shamt);
                        } else {
                            // SRLI
                            break :blk rs1 >> shamt;
                        }
                    },
                    _ => 0,
                };
                try self.registers.write(decoded.rd, result.rd_value);
            },
            .OP => {
                result.rd_value = switch (@as(zkvm.instruction.OpFunct3, @enumFromInt(decoded.funct3))) {
                    .ADD_SUB => blk: {
                        if ((decoded.funct7 & 0x20) != 0) {
                            break :blk rs1 -% rs2; // SUB
                        } else if ((decoded.funct7 & 0x01) != 0) {
                            break :blk rs1 *% rs2; // MUL
                        } else {
                            break :blk rs1 +% rs2; // ADD
                        }
                    },
                    .SLL => rs1 << @truncate(rs2 & 0x3F),
                    .SLT => if (@as(i64, @bitCast(rs1)) < @as(i64, @bitCast(rs2))) 1 else 0,
                    .SLTU => if (rs1 < rs2) 1 else 0,
                    .XOR => rs1 ^ rs2,
                    .SRL_SRA => blk: {
                        const shamt: u6 = @truncate(rs2 & 0x3F);
                        if ((decoded.funct7 & 0x20) != 0) {
                            break :blk @bitCast(@as(i64, @bitCast(rs1)) >> shamt);
                        } else {
                            break :blk rs1 >> shamt;
                        }
                    },
                    .OR => rs1 | rs2,
                    .AND => rs1 & rs2,
                    _ => 0,
                };
                try self.registers.write(decoded.rd, result.rd_value);
            },
            else => {
                // Unsupported instruction - treat as NOP
            },
        }

        return result;
    }

    /// Get the outputs
    pub fn getOutputs(self: *const Emulator) []const u8 {
        return self.device.outputs.items;
    }
};

test "emulator initialization" {
    const allocator = std.testing.allocator;
    const config = common.MemoryConfig{
        .program_size = 1024,
    };
    var emu = Emulator.init(allocator, &config);
    defer emu.deinit();

    try std.testing.expectEqual(@as(u64, 0), emu.state.cycle);
}

test "simple instruction execution" {
    const allocator = std.testing.allocator;
    const config = common.MemoryConfig{
        .program_size = 1024,
    };
    var emu = Emulator.init(allocator, &config);
    defer emu.deinit();

    // Load a simple program: addi x1, x0, 42
    const program = [_]u8{
        0x93, 0x00, 0xa0, 0x02, // addi x1, x0, 42
    };
    try emu.loadProgram(&program);

    // Execute one step
    const continued = try emu.step();
    try std.testing.expect(continued);

    // Check result
    const x1 = try emu.registers.read(1);
    try std.testing.expectEqual(@as(u64, 42), x1);
}
