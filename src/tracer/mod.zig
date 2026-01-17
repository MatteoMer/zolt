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
    /// Program counter before execution (expanded PC - bytecode array index)
    pc: u64,
    /// Unexpanded PC (raw RISC-V instruction address from ELF)
    /// For programs without virtual sequences, this equals pc.
    /// When virtual sequences are used (e.g., MUL expansion), pc != unexpanded_pc.
    unexpanded_pc: u64,
    /// Instruction executed (expanded to 32-bit if compressed)
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
    /// Whether the original instruction was compressed (RVC - 2 bytes)
    is_compressed: bool,
    /// Whether this is a NoOp padding cycle (not a real execution step)
    is_noop: bool = false,
};

/// Full execution trace
pub const ExecutionTrace = struct {
    steps: std.ArrayListUnmanaged(TraceStep),
    allocator: Allocator,

    pub fn init(allocator: Allocator) ExecutionTrace {
        return .{
            .steps = .{},
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *ExecutionTrace) void {
        self.steps.deinit(self.allocator);
    }

    /// Add a step to the trace
    pub fn addStep(self: *ExecutionTrace, step: TraceStep) !void {
        try self.steps.append(self.allocator, step);
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

    /// Pad trace with NoOp cycles to next power of 2 (matching Jolt's behavior).
    /// Minimum padded length is 256, otherwise (len + 1).next_power_of_two().
    /// This ensures the last real cycle has a NoOp as its next cycle.
    pub fn padWithNoop(self: *ExecutionTrace) !void {
        const unpadded_len = self.steps.items.len;

        // Skip if already padded (check if last step is NoOp)
        if (unpadded_len > 0 and self.steps.items[unpadded_len - 1].is_noop) {
            return; // Already padded
        }

        const padded_len = if (unpadded_len < 256)
            256
        else
            std.math.ceilPowerOfTwo(usize, unpadded_len + 1) catch unreachable;

        std.debug.print("[PADDING] Padding trace from {d} to {d} cycles\n", .{ unpadded_len, padded_len });

        // Create NoOp step template - all zeros with is_noop = true
        const noop_step = TraceStep{
            .cycle = 0,
            .pc = 0,
            .unexpanded_pc = 0,
            .instruction = 0,
            .rs1_value = 0,
            .rs2_value = 0,
            .rd_value = 0,
            .memory_addr = null,
            .memory_value = null,
            .is_memory_write = false,
            .next_pc = 0,
            .is_compressed = false,
            .is_noop = true,
        };

        // Pad with NoOp cycles
        try self.steps.ensureTotalCapacity(self.allocator, padded_len);
        while (self.steps.items.len < padded_len) {
            self.steps.appendAssumeCapacity(noop_step);
        }
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
    /// Lookup trace for Lasso proofs
    lookup_trace: zkvm.instruction.LookupTraceCollector(64),
    /// Whether the current instruction is compressed
    is_compressed: bool,
    /// Previous PC for infinite loop detection (matching Jolt's termination heuristic)
    prev_pc: u64,
    allocator: Allocator,

    pub fn init(allocator: Allocator, config: *const common.MemoryConfig) Emulator {
        return .{
            .state = zkvm.VMState.init(common.constants.RAM_START_ADDRESS),
            .ram = zkvm.ram.RAMState.init(allocator),
            .registers = zkvm.registers.RegisterFile.init(allocator),
            .device = common.JoltDevice.init(allocator, config),
            .trace = ExecutionTrace.init(allocator),
            .lookup_trace = zkvm.instruction.LookupTraceCollector(64).init(allocator),
            .is_compressed = false,
            .prev_pc = 0, // Will be set to initial PC on first step
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Emulator) void {
        self.ram.deinit();
        self.registers.deinit();
        self.device.deinit();
        self.trace.deinit();
        self.lookup_trace.deinit();
    }

    /// Load a program into memory at a specific base address
    /// NOTE: This does NOT record to the trace - program loading is part of initial state, not execution.
    /// Jolt treats program loading as initial memory state, not as execution trace entries.
    pub fn loadProgramAt(self: *Emulator, bytecode: []const u8, base_address: u64) !void {
        var addr: u64 = base_address;
        for (bytecode) |byte| {
            // Use untraced write - program loading is initial state, not execution
            try self.ram.writeByteUntraced(addr, byte);
            addr += 1;
        }
    }

    /// Load a program into memory at default RAM_START_ADDRESS
    pub fn loadProgram(self: *Emulator, bytecode: []const u8) !void {
        return self.loadProgramAt(bytecode, common.constants.RAM_START_ADDRESS);
    }

    /// Set input data
    pub fn setInputs(self: *Emulator, inputs: []const u8) !void {
        try self.device.inputs.appendSlice(self.allocator, inputs);
    }

    /// Check if address is in the I/O region (input, output, advice, etc.)
    fn isIOAddress(self: *const Emulator, address: u64) bool {
        return self.device.isInput(address) or
            self.device.isOutput(address) or
            self.device.isTrustedAdvice(address) or
            self.device.isUntrustedAdvice(address) or
            self.device.isPanic(address) or
            self.device.isTermination(address);
    }

    /// Read a byte, checking I/O region first
    fn readByteWithIO(self: *Emulator, address: u64) !u8 {
        if (self.isIOAddress(address)) {
            return self.device.load(address);
        }
        return self.ram.readByte(address, self.state.cycle);
    }

    /// Read a 64-bit word, checking I/O region first
    fn readWordWithIO(self: *Emulator, address: u64) !u64 {
        const aligned_addr = address & ~@as(u64, 7);
        if (self.isIOAddress(aligned_addr)) {
            // Read 8 bytes from I/O
            var result: u64 = 0;
            for (0..8) |i| {
                const byte = self.device.load(aligned_addr + i);
                result |= @as(u64, byte) << (@as(u6, @intCast(i)) * 8);
            }
            return result;
        }
        return self.ram.read(aligned_addr, self.state.cycle);
    }

    /// Write a byte, checking I/O region first
    fn writeByteWithIO(self: *Emulator, address: u64, value: u8) !void {
        if (self.isIOAddress(address)) {
            return self.device.store(address, value);
        }
        return self.ram.writeByte(address, value, self.state.cycle);
    }

    /// Write a 64-bit word, checking I/O region first
    fn writeWordWithIO(self: *Emulator, address: u64, value: u64) !void {
        const aligned_addr = address & ~@as(u64, 7);
        if (self.isIOAddress(aligned_addr)) {
            // Write 8 bytes to I/O
            for (0..8) |i| {
                try self.device.store(aligned_addr + i, @truncate(value >> (@as(u6, @intCast(i)) * 8)));
            }
            return;
        }
        return self.ram.write(aligned_addr, value, self.state.cycle);
    }

    /// Execute a single instruction
    pub fn step(self: *Emulator) !bool {
        // Infinite loop detection (matching Jolt's termination heuristic)
        // If PC hasn't changed since last step, the program has terminated
        // via an infinite loop (e.g., JAL x0, 0). This is standard for bare-metal
        // RISC-V programs without an OS to return to.
        // Skip this check on first step (prev_pc == 0 and state.pc != 0)
        if (self.prev_pc != 0 and self.prev_pc == self.state.pc) {
            std.debug.print("[TRACE] Detected infinite loop at PC 0x{x:0>8}, cycle {d}\n", .{ self.state.pc, self.state.cycle });
            return false; // Program terminated via infinite loop
        }

        // Fetch instruction
        const instruction = try self.fetchInstruction();
        self.state.instruction = instruction;

        // Debug removed for now - too noisy

        // Decode
        const decoded = zkvm.instruction.DecodedInstruction.decode(instruction);

        // Record pre-execution state
        const rs1_value = try self.registers.read(decoded.rs1);
        const rs2_value = try self.registers.read(decoded.rs2);

        // Record lookup trace for Lasso proofs (before execution)
        try self.lookup_trace.recordInstruction(
            @intCast(self.state.cycle),
            self.state.pc,
            instruction,
            decoded,
            rs1_value,
            rs2_value,
        );

        // Execute (may return error for ECALL)
        const result = self.execute(decoded, rs1_value, rs2_value) catch |err| {
            // Still count this cycle even if we're stopping on ECALL
            self.state.cycle += 1;
            return err;
        };

        // Record trace step
        try self.trace.addStep(.{
            .cycle = self.state.cycle,
            .pc = self.state.pc,
            // For now without virtual sequences, unexpanded_pc = pc
            // When virtual sequences are implemented, this would be the raw RISC-V address
            .unexpanded_pc = self.state.pc,
            .instruction = instruction,
            .rs1_value = rs1_value,
            .rs2_value = rs2_value,
            .rd_value = result.rd_value,
            .memory_addr = result.memory_addr,
            .memory_value = result.memory_value,
            .is_memory_write = result.is_memory_write,
            .next_pc = result.next_pc,
            .is_compressed = self.is_compressed,
        });

        // Update prev_pc for infinite loop detection (set to current PC before we change it)
        self.prev_pc = self.state.pc;

        // Update state
        self.state.pc = result.next_pc;
        self.state.cycle += 1;
        self.registers.tick();

        return true;
    }

    /// Run until completion
    /// Stops on ECALL (normal program termination) or infinite loop detection
    pub fn run(self: *Emulator) !void {
        while (true) {
            const running = self.step() catch |err| switch (err) {
                error.Ecall => {
                    std.debug.print("[TRACE] Terminated via ECALL at cycle {d}\n", .{self.state.cycle});
                    // NOTE: In Jolt, the termination bit is set directly in val_final during
                    // OutputSumcheck, NOT in the execution/memory trace. The RWC sumcheck
                    // only includes actual LOAD/STORE instructions from the trace.
                    // We do NOT record a synthetic termination write here.
                    return; // Normal termination
                },
                else => return err,
            };
            if (!running) {
                // Program terminated via infinite loop detection
                std.debug.print("[TRACE] Terminated via infinite loop at PC 0x{x}, cycle {d}\n", .{ self.state.pc, self.state.cycle });
                // Print last instruction to verify it's a jump
                if (self.trace.steps.items.len > 0) {
                    const last_step = self.trace.steps.items[self.trace.steps.items.len - 1];
                    std.debug.print("[TRACE] Last instruction: 0x{x:0>8} at PC 0x{x}\n", .{ last_step.instruction, last_step.pc });
                }
                // NOTE: In Jolt, the termination bit is set directly in val_final during
                // OutputSumcheck, NOT in the execution/memory trace. The RWC sumcheck
                // only includes actual LOAD/STORE instructions from the trace.
                // We do NOT record a synthetic termination write here.
                return;
            }
        }
    }

    // NOTE: recordTerminationWrite was removed. In Jolt, the termination bit
    // is set directly in val_final during OutputSumcheck. It is NOT part of
    // the execution/memory trace, so RWC sumcheck doesn't include it.

    /// Fetch instruction from memory, handling compressed instructions
    /// Returns the 32-bit instruction (expanded if compressed) and updates PC accordingly
    ///
    /// NOTE: Instruction fetches do NOT record to the RAM trace.
    /// In Jolt, instruction fetches are proven via bytecode commitment, not the RAM trace.
    /// Only explicit data memory operations (LW, SW) are recorded in the RAM trace.
    fn fetchInstruction(self: *Emulator) !u32 {
        // First fetch the lower 16 bits - use untraced reads
        var halfword: u32 = 0;
        inline for (0..2) |i| {
            const byte = self.ram.readByteUntraced(self.state.pc + i);
            halfword |= @as(u32, byte) << (@as(u5, @intCast(i)) * 8);
        }

        // Check if this is a compressed instruction
        if (zkvm.instruction.isCompressed(halfword)) {
            // 16-bit compressed instruction - expand it
            const expanded = zkvm.instruction.uncompressInstruction(halfword, .Bit64);
            // Advance PC by 2 (will be done in step())
            self.is_compressed = true;
            return expanded;
        } else {
            // 32-bit instruction - fetch the remaining 16 bits - use untraced reads
            var instruction = halfword;
            inline for (2..4) |i| {
                const byte = self.ram.readByteUntraced(self.state.pc + i);
                instruction |= @as(u32, byte) << (@as(u5, @intCast(i)) * 8);
            }
            self.is_compressed = false;
            return instruction;
        }
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
        // PC increment: 2 for compressed, 4 for regular instructions
        const pc_increment: u64 = if (self.is_compressed) 2 else 4;
        var result = ExecutionResult{
            .rd_value = 0,
            .memory_addr = null,
            .memory_value = null,
            .is_memory_write = false,
            .next_pc = self.state.pc + pc_increment,
        };

        switch (decoded.opcode) {
            .LUI => {
                result.rd_value = @bitCast(@as(i64, decoded.imm));
                try self.registers.write(decoded.rd, result.rd_value);
            },
            .AUIPC => {
                // PC + sign-extended immediate - use wrapping arithmetic for high addresses
                const pc_as_signed: i64 = @bitCast(self.state.pc);
                result.rd_value = @bitCast(pc_as_signed +% decoded.imm);
                try self.registers.write(decoded.rd, result.rd_value);
            },
            .JAL => {
                // Return address: PC + 2 for compressed, PC + 4 for regular
                result.rd_value = self.state.pc + pc_increment;
                // PC + sign-extended immediate - use wrapping arithmetic for high addresses
                const pc_as_signed: i64 = @bitCast(self.state.pc);
                result.next_pc = @bitCast(pc_as_signed +% decoded.imm);
                try self.registers.write(decoded.rd, result.rd_value);
            },
            .JALR => {
                // Return address: PC + 2 for compressed, PC + 4 for regular
                result.rd_value = self.state.pc + pc_increment;
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
                    // PC is u64, immediate is i32 - add correctly handling sign extension
                    result.next_pc = @bitCast(@as(i64, @bitCast(self.state.pc)) +% decoded.imm);
                }
            },
            .LOAD => {
                const addr: u64 = @bitCast(@as(i64, @bitCast(rs1)) + decoded.imm);
                result.memory_addr = addr;

                const value = switch (@as(zkvm.instruction.LoadFunct3, @enumFromInt(decoded.funct3))) {
                    .LB => blk: {
                        const byte = try self.readByteWithIO(addr);
                        break :blk @as(u64, @bitCast(@as(i64, @as(i8, @bitCast(byte)))));
                    },
                    .LBU => try self.readByteWithIO(addr),
                    .LH => blk: {
                        const low = try self.readByteWithIO(addr);
                        const high = try self.readByteWithIO(addr + 1);
                        const halfword: i16 = @bitCast((@as(u16, high) << 8) | low);
                        break :blk @as(u64, @bitCast(@as(i64, halfword)));
                    },
                    .LW => blk: {
                        // Read 4 bytes
                        var word: u32 = 0;
                        for (0..4) |i| {
                            const byte = try self.readByteWithIO(addr + i);
                            word |= @as(u32, byte) << (@as(u5, @intCast(i)) * 8);
                        }
                        const signed: i32 = @bitCast(word);
                        break :blk @as(u64, @bitCast(@as(i64, signed)));
                    },
                    .LWU => blk: {
                        // Load word unsigned (RV64)
                        var word: u32 = 0;
                        for (0..4) |i| {
                            const byte = try self.readByteWithIO(addr + i);
                            word |= @as(u32, byte) << (@as(u5, @intCast(i)) * 8);
                        }
                        break :blk @as(u64, word);
                    },
                    .LD => blk: {
                        // Read 8 bytes
                        var dword: u64 = 0;
                        for (0..8) |i| {
                            const byte = try self.readByteWithIO(addr + i);
                            dword |= @as(u64, byte) << (@as(u6, @intCast(i)) * 8);
                        }
                        break :blk dword;
                    },
                    .LHU => blk: {
                        // Load halfword unsigned
                        const low = try self.readByteWithIO(addr);
                        const high = try self.readByteWithIO(addr + 1);
                        break :blk @as(u64, (@as(u16, high) << 8) | low);
                    },
                    _ => 0,
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
                        try self.writeByteWithIO(addr, @truncate(rs2));
                        result.memory_value = rs2 & 0xFF;
                    },
                    .SH => {
                        try self.writeByteWithIO(addr, @truncate(rs2));
                        try self.writeByteWithIO(addr + 1, @truncate(rs2 >> 8));
                        result.memory_value = rs2 & 0xFFFF;
                    },
                    .SW => {
                        // Write 4 bytes
                        for (0..4) |i| {
                            try self.writeByteWithIO(addr + i, @truncate(rs2 >> (@as(u6, @intCast(i)) * 8)));
                        }
                        result.memory_value = rs2 & 0xFFFFFFFF;
                    },
                    .SD => {
                        // Write 8 bytes
                        for (0..8) |i| {
                            try self.writeByteWithIO(addr + i, @truncate(rs2 >> (@as(u6, @intCast(i)) * 8)));
                        }
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
                    .SLLI => rs1 << @as(u6, @intCast(@as(u32, @bitCast(decoded.imm)) & 0x3F)),
                    .SRLI_SRAI => blk: {
                        const shamt: u6 = @intCast(@as(u32, @bitCast(decoded.imm)) & 0x3F);
                        if ((decoded.funct7 & 0x20) != 0) {
                            // SRAI
                            break :blk @bitCast(@as(i64, @bitCast(rs1)) >> shamt);
                        } else {
                            // SRLI
                            break :blk rs1 >> shamt;
                        }
                    },
                };
                try self.registers.write(decoded.rd, result.rd_value);
            },
            .OP => {
                // Check for M extension (funct7 = 0b0000001)
                if (decoded.funct7 == 0b0000001) {
                    // M Extension: multiply/divide operations
                    result.rd_value = switch (@as(zkvm.instruction.MulDivFunct3, @enumFromInt(decoded.funct3))) {
                        .MUL => rs1 *% rs2, // Lower 64 bits of product
                        .MULH => blk: {
                            // Signed * Signed -> upper 64 bits
                            const a: i128 = @as(i64, @bitCast(rs1));
                            const b: i128 = @as(i64, @bitCast(rs2));
                            const prod = a * b;
                            break :blk @bitCast(@as(i64, @truncate(prod >> 64)));
                        },
                        .MULHSU => blk: {
                            // Signed * Unsigned -> upper 64 bits
                            const a: i128 = @as(i64, @bitCast(rs1));
                            const b: i128 = @as(i128, rs2);
                            const prod = a * b;
                            break :blk @bitCast(@as(i64, @truncate(prod >> 64)));
                        },
                        .MULHU => blk: {
                            // Unsigned * Unsigned -> upper 64 bits
                            const a: u128 = rs1;
                            const b: u128 = rs2;
                            const prod = a * b;
                            break :blk @truncate(prod >> 64);
                        },
                        .DIV => blk: {
                            // Signed division
                            if (rs2 == 0) {
                                break :blk @as(u64, @bitCast(@as(i64, -1)));
                            }
                            const a: i64 = @bitCast(rs1);
                            const b: i64 = @bitCast(rs2);
                            // Handle overflow case
                            if (a == std.math.minInt(i64) and b == -1) {
                                break :blk rs1; // Overflow returns dividend
                            }
                            break :blk @bitCast(@divTrunc(a, b));
                        },
                        .DIVU => blk: {
                            // Unsigned division
                            if (rs2 == 0) {
                                break :blk std.math.maxInt(u64);
                            }
                            break :blk rs1 / rs2;
                        },
                        .REM => blk: {
                            // Signed remainder
                            if (rs2 == 0) {
                                break :blk rs1;
                            }
                            const a: i64 = @bitCast(rs1);
                            const b: i64 = @bitCast(rs2);
                            // Handle overflow case
                            if (a == std.math.minInt(i64) and b == -1) {
                                break :blk 0;
                            }
                            break :blk @bitCast(@rem(a, b));
                        },
                        .REMU => blk: {
                            // Unsigned remainder
                            if (rs2 == 0) {
                                break :blk rs1;
                            }
                            break :blk rs1 % rs2;
                        },
                    };
                } else {
                    result.rd_value = switch (@as(zkvm.instruction.OpFunct3, @enumFromInt(decoded.funct3))) {
                        .ADD_SUB => blk: {
                            if ((decoded.funct7 & 0x20) != 0) {
                                break :blk rs1 -% rs2; // SUB
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
                    };
                }
                try self.registers.write(decoded.rd, result.rd_value);
            },
            .OP_IMM_32 => {
                // RV64I word operations (32-bit, sign-extended to 64)
                const imm_lower: u32 = @truncate(@as(u64, @bitCast(@as(i64, decoded.imm))));
                const rs1_lower: u32 = @truncate(rs1);
                const result_32: i32 = switch (decoded.funct3) {
                    0b000 => blk: {
                        // ADDIW: rd = sext((rs1 + imm)[31:0])
                        const imm_i32: i32 = @bitCast(imm_lower);
                        const rs1_i32: i32 = @bitCast(rs1_lower);
                        break :blk rs1_i32 +% imm_i32;
                    },
                    0b001 => blk: {
                        // SLLIW: rd = sext((rs1 << shamt)[31:0])
                        const shamt: u5 = @truncate(imm_lower & 0x1F);
                        break :blk @bitCast(rs1_lower << shamt);
                    },
                    0b101 => blk: {
                        // SRLIW or SRAIW
                        const shamt: u5 = @truncate(imm_lower & 0x1F);
                        if ((decoded.funct7 & 0x20) != 0) {
                            // SRAIW: arithmetic right shift
                            const rs1_i32: i32 = @bitCast(rs1_lower);
                            break :blk rs1_i32 >> shamt;
                        } else {
                            // SRLIW: logical right shift
                            break :blk @bitCast(rs1_lower >> shamt);
                        }
                    },
                    else => 0,
                };
                result.rd_value = @bitCast(@as(i64, result_32));
                try self.registers.write(decoded.rd, result.rd_value);
            },
            .OP_32 => {
                // RV64I/M word operations (32-bit register-register, sign-extended to 64)
                const rs1_lower: u32 = @truncate(rs1);
                const rs2_lower: u32 = @truncate(rs2);
                const result_32: i32 = if (decoded.funct7 == 0b0000001) blk: {
                    // RV64M word operations
                    break :blk switch (decoded.funct3) {
                        0b000 => blk2: {
                            // MULW: multiply word
                            const a: i32 = @bitCast(rs1_lower);
                            const b: i32 = @bitCast(rs2_lower);
                            break :blk2 a *% b;
                        },
                        0b100 => blk2: {
                            // DIVW: signed divide word
                            const a: i32 = @bitCast(rs1_lower);
                            const b: i32 = @bitCast(rs2_lower);
                            if (b == 0) {
                                break :blk2 -1;
                            }
                            if (a == std.math.minInt(i32) and b == -1) {
                                break :blk2 a; // Overflow
                            }
                            break :blk2 @divTrunc(a, b);
                        },
                        0b101 => blk2: {
                            // DIVUW: unsigned divide word
                            if (rs2_lower == 0) {
                                break :blk2 @bitCast(@as(u32, std.math.maxInt(u32)));
                            }
                            break :blk2 @bitCast(rs1_lower / rs2_lower);
                        },
                        0b110 => blk2: {
                            // REMW: signed remainder word
                            const a: i32 = @bitCast(rs1_lower);
                            const b: i32 = @bitCast(rs2_lower);
                            if (b == 0) {
                                break :blk2 a;
                            }
                            if (a == std.math.minInt(i32) and b == -1) {
                                break :blk2 0; // Overflow
                            }
                            break :blk2 @rem(a, b);
                        },
                        0b111 => blk2: {
                            // REMUW: unsigned remainder word
                            if (rs2_lower == 0) {
                                break :blk2 @bitCast(rs1_lower);
                            }
                            break :blk2 @bitCast(rs1_lower % rs2_lower);
                        },
                        else => 0,
                    };
                } else blk: {
                    // RV64I word operations
                    break :blk switch (decoded.funct3) {
                        0b000 => blk2: {
                            // ADDW or SUBW
                            const a: i32 = @bitCast(rs1_lower);
                            const b: i32 = @bitCast(rs2_lower);
                            if ((decoded.funct7 & 0x20) != 0) {
                                break :blk2 a -% b; // SUBW
                            } else {
                                break :blk2 a +% b; // ADDW
                            }
                        },
                        0b001 => blk2: {
                            // SLLW: shift left logical word
                            const shamt: u5 = @truncate(rs2_lower & 0x1F);
                            break :blk2 @bitCast(rs1_lower << shamt);
                        },
                        0b101 => blk2: {
                            // SRLW or SRAW
                            const shamt: u5 = @truncate(rs2_lower & 0x1F);
                            if ((decoded.funct7 & 0x20) != 0) {
                                // SRAW: arithmetic right shift word
                                const a: i32 = @bitCast(rs1_lower);
                                break :blk2 a >> shamt;
                            } else {
                                // SRLW: logical right shift word
                                break :blk2 @bitCast(rs1_lower >> shamt);
                            }
                        },
                        else => 0,
                    };
                };
                result.rd_value = @bitCast(@as(i64, result_32));
                try self.registers.write(decoded.rd, result.rd_value);
            },
            .SYSTEM => {
                // ECALL and EBREAK
                // funct12 field: ECALL = 0, EBREAK = 1
                const funct12: u12 = @truncate((@as(u32, @bitCast(decoded.imm)) >> 0) & 0xFFF);
                if (funct12 == 0) {
                    // ECALL - check if it's a Jolt SDK call or termination
                    const a0 = try self.registers.read(10); // syscall number in a0
                    const a7 = try self.registers.read(17); // Also check a7 (standard syscall convention)

                    // Jolt SDK ECALL numbers
                    const JOLT_CYCLE_TRACK_ECALL_NUM: u64 = 0xC7C1E; // "C Y C L E"
                    const JOLT_PRINT_ECALL_NUM: u64 = 0x5072696E; // "P r i n"

                    if (a0 == JOLT_CYCLE_TRACK_ECALL_NUM or a7 == JOLT_CYCLE_TRACK_ECALL_NUM) {
                        // Cycle tracking - ignore and continue execution
                        // In Jolt this is used for profiling, we can skip it
                    } else if (a0 == JOLT_PRINT_ECALL_NUM or a7 == JOLT_PRINT_ECALL_NUM) {
                        // Print/output - we could implement this but for now just continue
                    } else if (self.device.isTermination(@truncate(a0))) {
                        // Check if it's a termination request via device address
                        return error.Ecall;
                    } else {
                        // For standard bare-metal programs (like fibonacci.c), ECALL is
                        // used as the exit syscall. Terminate execution.
                        // Note: Jolt SDK programs use infinite loop for termination,
                        // but simple C programs use ECALL.
                        return error.Ecall;
                    }
                }
                // EBREAK and other system instructions - treat as NOP
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

test "M extension multiply" {
    const allocator = std.testing.allocator;
    const config = common.MemoryConfig{
        .program_size = 1024,
    };
    var emu = Emulator.init(allocator, &config);
    defer emu.deinit();

    // Set up registers for multiplication
    try emu.registers.write(1, 7); // x1 = 7
    try emu.registers.write(2, 6); // x2 = 6

    // mul x3, x1, x2  (0x02208133)
    // opcode=0x33 (OP), rd=3, rs1=1, rs2=2, funct3=0, funct7=1
    const mul_instr: u32 = 0x022080b3; // mul x1, x1, x2
    const program = std.mem.asBytes(&mul_instr);
    try emu.loadProgram(program);

    _ = try emu.step();

    const x1 = try emu.registers.read(1);
    try std.testing.expectEqual(@as(u64, 42), x1); // 7 * 6 = 42
}

test "M extension division" {
    const allocator = std.testing.allocator;
    const config = common.MemoryConfig{
        .program_size = 1024,
    };
    var emu = Emulator.init(allocator, &config);
    defer emu.deinit();

    // Set up registers
    try emu.registers.write(1, 42); // x1 = 42
    try emu.registers.write(2, 6); // x2 = 6

    // divu x1, x1, x2 (opcode=0x33, funct3=5, funct7=1)
    // 0000001 | rs2 | rs1 | 101 | rd | 0110011
    // 0000001 00010 00001 101 00001 0110011
    const divu_instr: u32 = 0x0220d0b3;
    const program = std.mem.asBytes(&divu_instr);
    try emu.loadProgram(program);

    _ = try emu.step();

    const x1 = try emu.registers.read(1);
    try std.testing.expectEqual(@as(u64, 7), x1); // 42 / 6 = 7
}

test "M extension division by zero" {
    const allocator = std.testing.allocator;
    const config = common.MemoryConfig{
        .program_size = 1024,
    };
    var emu = Emulator.init(allocator, &config);
    defer emu.deinit();

    // Set up registers
    try emu.registers.write(1, 42); // x1 = 42
    try emu.registers.write(2, 0); // x2 = 0

    // divu x1, x1, x2
    const divu_instr: u32 = 0x0220d0b3;
    const program = std.mem.asBytes(&divu_instr);
    try emu.loadProgram(program);

    _ = try emu.step();

    const x1 = try emu.registers.read(1);
    // RISC-V spec: unsigned division by zero returns max value
    try std.testing.expectEqual(std.math.maxInt(u64), x1);
}

test "compressed instruction C.NOP" {
    const allocator = std.testing.allocator;
    const config = common.MemoryConfig{
        .program_size = 1024,
    };
    var emu = Emulator.init(allocator, &config);
    defer emu.deinit();

    // C.NOP is 0x0001 (2 bytes)
    // C.NOP expands to addi x0, x0, 0
    const program = [_]u8{
        0x01, 0x00, // C.NOP
        0x01, 0x00, // C.NOP
    };
    try emu.loadProgram(&program);

    // Execute first C.NOP
    const continued = try emu.step();
    try std.testing.expect(continued);

    // PC should advance by 2 (compressed instruction)
    try std.testing.expectEqual(common.constants.RAM_START_ADDRESS + 2, emu.state.pc);
}

test "compressed instruction C.LI" {
    const allocator = std.testing.allocator;
    const config = common.MemoryConfig{
        .program_size = 1024,
    };
    var emu = Emulator.init(allocator, &config);
    defer emu.deinit();

    // C.LI x10, 5 expands to addi x10, x0, 5
    // Format: funct3=010, imm[5]=0, rd=01010, imm[4:0]=00101, op=01
    // Binary: 010 0 01010 00101 01 = 0x4505
    const program = [_]u8{
        0x15, 0x45, // C.LI x10, 5 (little endian: 0x4515)
    };
    try emu.loadProgram(&program);

    _ = try emu.step();

    // Check that x10 = 5
    const x10 = try emu.registers.read(10);
    try std.testing.expectEqual(@as(u64, 5), x10);
}

/// Witness generation from execution trace
///
/// Converts an execution trace into witnesses for R1CS constraints.
/// Each trace step generates witness values for the circuit.
pub fn WitnessGenerator(comptime F: type) type {
    return struct {
        const Self = @This();

        allocator: Allocator,

        pub fn init(allocator: Allocator) Self {
            return .{ .allocator = allocator };
        }

        /// Witness layout for a single CPU step
        /// This matches Jolt's circuit structure
        pub const StepWitness = struct {
            /// Program counter
            pc: F,
            /// Instruction (as field element)
            instruction: F,
            /// Source register 1 value
            rs1: F,
            /// Source register 2 value
            rs2: F,
            /// Destination register value
            rd: F,
            /// Immediate value
            imm: F,
            /// Memory address (0 if no memory access)
            memory_addr: F,
            /// Memory value
            memory_value: F,
            /// Next PC
            next_pc: F,
            /// Opcode breakdown bits (for instruction decoding constraints)
            opcode_bits: [7]F,
            /// Register index bits for rs1
            rs1_idx_bits: [5]F,
            /// Register index bits for rs2
            rs2_idx_bits: [5]F,
            /// Register index bits for rd
            rd_idx_bits: [5]F,
        };

        /// Generate witness for a single trace step
        pub fn generateStepWitness(self: *Self, step: TraceStep) StepWitness {
            _ = self;
            const decoded = zkvm.instruction.DecodedInstruction.decode(step.instruction);

            // Convert values to field elements
            const pc = F.fromU64(step.pc);
            const instruction_f = F.fromU64(step.instruction);
            const rs1 = F.fromU64(step.rs1_value);
            const rs2 = F.fromU64(step.rs2_value);
            const rd = F.fromU64(step.rd_value);
            const imm: F = if (decoded.imm >= 0)
                F.fromU64(@intCast(decoded.imm))
            else
                F.zero().sub(F.fromU64(@intCast(-decoded.imm)));
            const memory_addr = F.fromU64(step.memory_addr orelse 0);
            const memory_value = F.fromU64(step.memory_value orelse 0);
            const next_pc = F.fromU64(step.next_pc);

            // Extract opcode bits
            var opcode_bits: [7]F = undefined;
            const opcode_val = step.instruction & 0x7F;
            for (0..7) |i| {
                opcode_bits[i] = if ((opcode_val >> @intCast(i)) & 1 == 1) F.one() else F.zero();
            }

            // Extract register index bits
            var rs1_idx_bits: [5]F = undefined;
            var rs2_idx_bits: [5]F = undefined;
            var rd_idx_bits: [5]F = undefined;

            for (0..5) |i| {
                rs1_idx_bits[i] = if ((decoded.rs1 >> @intCast(i)) & 1 == 1) F.one() else F.zero();
                rs2_idx_bits[i] = if ((decoded.rs2 >> @intCast(i)) & 1 == 1) F.one() else F.zero();
                rd_idx_bits[i] = if ((decoded.rd >> @intCast(i)) & 1 == 1) F.one() else F.zero();
            }

            return .{
                .pc = pc,
                .instruction = instruction_f,
                .rs1 = rs1,
                .rs2 = rs2,
                .rd = rd,
                .imm = imm,
                .memory_addr = memory_addr,
                .memory_value = memory_value,
                .next_pc = next_pc,
                .opcode_bits = opcode_bits,
                .rs1_idx_bits = rs1_idx_bits,
                .rs2_idx_bits = rs2_idx_bits,
                .rd_idx_bits = rd_idx_bits,
            };
        }

        /// Generate full witness vector from trace
        ///
        /// Returns a flat array of field elements suitable for R1CS verification
        pub fn generateWitness(self: *Self, trace: *const ExecutionTrace) ![]F {
            const steps = trace.steps.items;
            if (steps.len == 0) {
                return self.allocator.alloc(F, 0);
            }

            // Witness layout:
            // [1 (constant), public_inputs..., step_witnesses...]
            //
            // Each step has the following witness elements:
            // pc, instruction, rs1, rs2, rd, imm, memory_addr, memory_value, next_pc,
            // 7 opcode bits, 5+5+5 register bits = 31 elements per step

            const elements_per_step: usize = 9 + 7 + 15; // 31 elements

            // Public inputs: initial PC, final PC, cycle count
            const public_inputs_count: usize = 3;

            const total_elements = 1 + public_inputs_count + (steps.len * elements_per_step);
            const witness = try self.allocator.alloc(F, total_elements);

            var idx: usize = 0;

            // Constant 1
            witness[idx] = F.one();
            idx += 1;

            // Public inputs
            witness[idx] = F.fromU64(steps[0].pc); // Initial PC
            idx += 1;
            witness[idx] = F.fromU64(steps[steps.len - 1].next_pc); // Final PC
            idx += 1;
            witness[idx] = F.fromU64(steps.len); // Cycle count
            idx += 1;

            // Step witnesses
            for (steps) |step| {
                const sw = self.generateStepWitness(step);

                witness[idx] = sw.pc;
                idx += 1;
                witness[idx] = sw.instruction;
                idx += 1;
                witness[idx] = sw.rs1;
                idx += 1;
                witness[idx] = sw.rs2;
                idx += 1;
                witness[idx] = sw.rd;
                idx += 1;
                witness[idx] = sw.imm;
                idx += 1;
                witness[idx] = sw.memory_addr;
                idx += 1;
                witness[idx] = sw.memory_value;
                idx += 1;
                witness[idx] = sw.next_pc;
                idx += 1;

                // Opcode bits
                for (sw.opcode_bits) |bit| {
                    witness[idx] = bit;
                    idx += 1;
                }

                // Register index bits
                for (sw.rs1_idx_bits) |bit| {
                    witness[idx] = bit;
                    idx += 1;
                }
                for (sw.rs2_idx_bits) |bit| {
                    witness[idx] = bit;
                    idx += 1;
                }
                for (sw.rd_idx_bits) |bit| {
                    witness[idx] = bit;
                    idx += 1;
                }
            }

            return witness;
        }

        /// Generate memory checking witness
        ///
        /// Creates offline memory checking tuples for all memory accesses
        pub fn generateMemoryWitness(self: *Self, trace: *const ExecutionTrace) !MemoryWitness(F) {
            const steps = trace.steps.items;

            // Use unmanaged ArrayList for Zig 0.15
            var reads: std.ArrayListUnmanaged(MemoryTuple(F)) = .{};
            var writes: std.ArrayListUnmanaged(MemoryTuple(F)) = .{};

            for (steps, 0..) |step, i| {
                if (step.memory_addr) |addr| {
                    const tuple = MemoryTuple(F){
                        .address = F.fromU64(addr),
                        .value = F.fromU64(step.memory_value orelse 0),
                        .timestamp = F.fromU64(@intCast(i)),
                    };

                    if (step.is_memory_write) {
                        try writes.append(self.allocator, tuple);
                    } else {
                        try reads.append(self.allocator, tuple);
                    }
                }
            }

            return MemoryWitness(F){
                .reads = try reads.toOwnedSlice(self.allocator),
                .writes = try writes.toOwnedSlice(self.allocator),
                .allocator = self.allocator,
            };
        }
    };
}

/// Memory access tuple for offline memory checking
pub fn MemoryTuple(comptime F: type) type {
    return struct {
        address: F,
        value: F,
        timestamp: F,
    };
}

/// Full memory witness for offline memory checking
pub fn MemoryWitness(comptime F: type) type {
    return struct {
        const Self = @This();

        reads: []MemoryTuple(F),
        writes: []MemoryTuple(F),
        allocator: Allocator,

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.reads);
            self.allocator.free(self.writes);
        }

        /// Total number of memory operations
        pub fn totalOps(self: *const Self) usize {
            return self.reads.len + self.writes.len;
        }
    };
}

test "witness generation from trace" {
    const field = @import("../field/mod.zig");
    const F = field.BN254Scalar;
    const allocator = std.testing.allocator;

    var config = common.MemoryConfig{ .program_size = 1024 };
    var emu = Emulator.init(allocator, &config);
    defer emu.deinit();

    // Simple program: addi x1, x0, 42; addi x2, x1, 8
    const program = [_]u8{
        0x93, 0x00, 0xa0, 0x02, // addi x1, x0, 42
        0x13, 0x81, 0x80, 0x00, // addi x2, x1, 8
    };
    try emu.loadProgram(&program);

    // Execute both instructions
    _ = try emu.step();
    _ = try emu.step();

    // Generate witness
    var wg = WitnessGenerator(F).init(allocator);
    const witness = try wg.generateWitness(&emu.trace);
    defer allocator.free(witness);

    // Verify witness has expected size
    // 1 + 3 public inputs + 2 steps * 31 elements = 66
    try std.testing.expectEqual(@as(usize, 66), witness.len);

    // Verify constant 1
    try std.testing.expect(witness[0].eql(F.one()));

    // Verify cycle count = 2
    try std.testing.expect(witness[3].eql(F.fromU64(2)));
}

test "memory witness generation" {
    const field = @import("../field/mod.zig");
    const F = field.BN254Scalar;
    const allocator = std.testing.allocator;

    var config = common.MemoryConfig{ .program_size = 1024 };
    var emu = Emulator.init(allocator, &config);
    defer emu.deinit();

    // Set up for store instruction
    try emu.registers.write(1, 0x1000); // x1 = address
    try emu.registers.write(2, 42); // x2 = value to store

    // sw x2, 0(x1)
    // Encoding: imm[11:5]=0, rs2=2, rs1=1, funct3=010, imm[4:0]=0, opcode=0100011
    const sw_instr: u32 = 0x0020a023; // sw x2, 0(x1)
    const program = std.mem.asBytes(&sw_instr);
    try emu.loadProgram(program);

    _ = try emu.step();

    // Generate memory witness
    var wg = WitnessGenerator(F).init(allocator);
    var mem_wit = try wg.generateMemoryWitness(&emu.trace);
    defer mem_wit.deinit();

    // Should have 1 write
    try std.testing.expectEqual(@as(usize, 1), mem_wit.writes.len);
    try std.testing.expectEqual(@as(usize, 0), mem_wit.reads.len);

    // Verify write tuple
    try std.testing.expect(mem_wit.writes[0].address.eql(F.fromU64(0x1000)));
}

test "lookup trace integration" {
    const allocator = std.testing.allocator;
    const config = common.MemoryConfig{ .program_size = 1024 };
    var emu = Emulator.init(allocator, &config);
    defer emu.deinit();

    // Program that exercises different instruction types:
    // addi x1, x0, 10   ; x1 = 10 (generates ADD lookup)
    // addi x2, x0, 5    ; x2 = 5  (generates ADD lookup)
    // add  x3, x1, x2   ; x3 = 15 (generates ADD lookup)
    // and  x4, x1, x2   ; x4 = 0  (generates AND lookup)
    // or   x5, x1, x2   ; x5 = 15 (generates OR lookup)
    const program = [_]u8{
        0x93, 0x00, 0xa0, 0x00, // addi x1, x0, 10
        0x13, 0x01, 0x50, 0x00, // addi x2, x0, 5
        0xb3, 0x81, 0x20, 0x00, // add x3, x1, x2
        0x33, 0xf2, 0x20, 0x00, // and x4, x1, x2
        0xb3, 0xe2, 0x20, 0x00, // or x5, x1, x2
    };
    try emu.loadProgram(&program);

    // Execute all instructions
    _ = try emu.step(); // addi x1, x0, 10
    _ = try emu.step(); // addi x2, x0, 5
    _ = try emu.step(); // add x3, x1, x2
    _ = try emu.step(); // and x4, x1, x2
    _ = try emu.step(); // or x5, x1, x2

    // Check register results
    try std.testing.expectEqual(@as(u64, 10), try emu.registers.read(1));
    try std.testing.expectEqual(@as(u64, 5), try emu.registers.read(2));
    try std.testing.expectEqual(@as(u64, 15), try emu.registers.read(3));
    try std.testing.expectEqual(@as(u64, 0), try emu.registers.read(4)); // 10 & 5 = 0 (binary: 1010 & 0101 = 0)
    try std.testing.expectEqual(@as(u64, 15), try emu.registers.read(5)); // 10 | 5 = 15

    // Check lookup trace was recorded
    const stats = emu.lookup_trace.getStats();
    try std.testing.expect(stats.total_lookups >= 5);

    // Verify specific lookup types were recorded
    try std.testing.expect(stats.range_check_lookups >= 3); // 3 ADD operations
    try std.testing.expect(stats.and_lookups >= 1);
    try std.testing.expect(stats.or_lookups >= 1);
}

test "I/O region read from inputs" {
    const allocator = std.testing.allocator;
    const config = common.MemoryConfig{
        .program_size = 1024,
    };
    var emu = Emulator.init(allocator, &config);
    defer emu.deinit();

    // Set up input data - simulating postcard-serialized u32 value 50
    // postcard serializes small u32 as a single varint byte
    try emu.setInputs(&[_]u8{ 50 });

    // Get the input address
    const input_addr = emu.device.memory_layout.input_start;

    // Verify we can read the input via the I/O region helpers
    try std.testing.expect(emu.isIOAddress(input_addr));

    const byte = try emu.readByteWithIO(input_addr);
    try std.testing.expectEqual(@as(u8, 50), byte);
}

test "I/O region write to outputs" {
    const allocator = std.testing.allocator;
    const config = common.MemoryConfig{
        .program_size = 1024,
    };
    var emu = Emulator.init(allocator, &config);
    defer emu.deinit();

    // Get the output address
    const output_addr = emu.device.memory_layout.output_start;

    // Write to output region
    try emu.writeByteWithIO(output_addr, 42);
    try emu.writeByteWithIO(output_addr + 1, 43);

    // Verify outputs were recorded
    try std.testing.expectEqual(@as(u8, 42), emu.device.outputs.items[0]);
    try std.testing.expectEqual(@as(u8, 43), emu.device.outputs.items[1]);
}
