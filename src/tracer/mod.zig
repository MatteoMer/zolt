//! RISC-V execution tracer for Jolt
//!
//! This module traces the execution of RISC-V programs to generate
//! the execution trace needed for proving.

const std = @import("std");

const zkvm_debug = @import("../zkvm/debug.zig");
const dbg = zkvm_debug.dbg;

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
    /// Destination register value BEFORE execution (pre-value for increment computation)
    rd_pre_value: u64,
    /// Destination register value AFTER execution (post-value)
    rd_value: u64,
    /// Register indices (0-127, supporting virtual registers 32+)
    /// These are used by Stage 4 (registers read/write checking) instead of
    /// extracting from the instruction word, which can only hold 5-bit indices.
    rd_index: u8 = 0,
    rs1_index: u8 = 0,
    rs2_index: u8 = 0,
    /// Whether rd is actually written this cycle (false for branches, stores, noops)
    rd_written: bool = false,
    /// Whether rs1 is actually read this cycle
    rs1_read: bool = false,
    /// Whether rs2 is actually read this cycle
    rs2_read: bool = false,
    /// Memory address accessed (if any)
    memory_addr: ?u64,
    /// Memory value BEFORE write (pre-value for increment computation, only meaningful for writes)
    memory_pre_value: ?u64,
    /// Memory value AFTER write (post-value, only meaningful for writes)
    memory_value: ?u64,
    /// Whether this is a memory write
    is_memory_write: bool,
    /// Next PC
    next_pc: u64,
    /// Whether the original instruction was compressed (RVC - 2 bytes)
    is_compressed: bool,
    /// Whether this is a NoOp padding cycle (not a real execution step)
    is_noop: bool = false,
    /// Whether this is a synthetic termination Store instruction.
    /// When true, the R1CS witness uses createTerminationStoreWitness() instead of
    /// fromTraceStep(), which sets FlagStore=1 AND FlagDoNotUpdateUnexpandedPC=1.
    is_termination_store: bool = false,
    /// Whether this is the synthetic JAL-to-self instruction after the termination store.
    /// This instruction has Jump=1 which disables all NextUPC constraints for the
    /// JAL→NoOp transition, allowing the SB anchor to use VI=true, DNUPC=false
    /// (matching vanilla Jolt's circuit_flags for SD with vsr=Some(0)).
    is_termination_jal: bool = false,
    /// For virtual instruction sequences (e.g., termination stores: LUI+ADDI+SB),
    /// counts down from (sequence_length - 1) to 0. Used by BytecodePCMapper to
    /// assign each virtual instruction its own bytecode index k.
    /// Default 0 means this is the last (or only) instruction in its sequence.
    virtual_sequence_remaining: u16 = 0,
    /// Whether this is the first instruction in a virtual sequence.
    /// In Jolt, this maps to the `is_first_in_sequence` field on instructions.
    /// For 2-step W-extension: the base instruction is first.
    /// For 12-step REMUW: the first VirtualAdvice is first.
    is_first_in_sequence: bool = false,
    /// Whether this is the last instruction in a virtual sequence (vsr == Some(0)).
    /// In upstream Jolt, this is CircuitFlags::IsLastInSequence.
    is_last_in_sequence: bool = false,
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

        // Skip if already padded (check if last step is a real NoOp padding cycle,
        // not a termination Store which also has is_noop=true)
        if (unpadded_len > 0) {
            const last = self.steps.items[unpadded_len - 1];
            if (last.is_noop and !last.is_termination_store) {
                return; // Already padded
            }
        }

        const padded_len = if (unpadded_len < 256)
            256
        else
            std.math.ceilPowerOfTwo(usize, unpadded_len + 1) catch unreachable;

        dbg("[PADDING] Padding trace from {d} to {d} cycles\n", .{ unpadded_len, padded_len });

        // Create NoOp step template - all zeros with is_noop = true
        const noop_step = TraceStep{
            .cycle = 0,
            .pc = 0,
            .unexpanded_pc = 0,
            .instruction = 0,
            .rs1_value = 0,
            .rs2_value = 0,
            .rd_pre_value = 0,
            .rd_value = 0,
            .memory_addr = null,
            .memory_pre_value = null,
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
    /// Lookup trace for Shout proofs
    lookup_trace: zkvm.instruction.LookupTraceCollector(64),
    /// Whether the current instruction is compressed
    is_compressed: bool,
    /// Previous PC for infinite loop detection (matching Jolt's termination heuristic)
    prev_pc: u64,
    /// Advice tape read position (consumed by AdviceLB/H/W/D).
    /// Returns 0 when tape is exhausted (valid externally-supplied witness).
    advice_pos: usize,
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
            .advice_pos = 0,
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

    /// Read `num_bytes` (1, 2, 4, or 8) from the untrusted advice tape.
    /// Advances `advice_pos`. Returns 0 when the tape is exhausted (valid
    /// witness — advice is externally supplied and only range-checked).
    fn adviceTapeRead(self: *Emulator, num_bytes: usize) u64 {
        std.debug.assert(num_bytes == 1 or num_bytes == 2 or num_bytes == 4 or num_bytes == 8);
        var value: u64 = 0;
        const tape = self.device.untrusted_advice.items;
        for (0..num_bytes) |i| {
            const byte: u8 = if (self.advice_pos + i < tape.len) tape[self.advice_pos + i] else 0;
            value |= @as(u64, byte) << @intCast(i * 8);
        }
        self.advice_pos += num_bytes;
        return value;
    }

    /// Execute AdviceLB/LH/LW/LD as the inline sequence expansion
    /// (VirtualAdviceLoad [+ VirtualMULI (SLLI) + VirtualSRAI (SRAI)]).
    /// For num_bytes=8 (AdviceLD) it's just one VirtualAdviceLoad cycle.
    /// For num_bytes=1/2/4 it's 3 cycles: VirtualAdviceLoad, VirtualMULI (SLLI × shift),
    /// VirtualSRAI (SRAI × shift) where shift = 64 - num_bytes*8 to sign-extend.
    fn stepAdviceLoadSignExt(
        self: *Emulator,
        decoded: zkvm.instruction.DecodedInstruction,
        num_bytes: usize,
    ) !bool {
        const pc_increment: u64 = if (self.is_compressed) 2 else 4;
        const base_pc = self.state.pc;
        const rd: u8 = decoded.rd;
        const shift_amount: u6 = @intCast(64 - num_bytes * 8);

        // ── Step 1 (always): VirtualAdvice(rd) reads `num_bytes` from advice tape ──
        const advice_value: u64 = self.adviceTapeRead(num_bytes);
        const rd_pre = try self.registers.read(rd);
        if (rd != 0) try self.registers.write(rd, advice_value);

        const advice_load_instr: u32 = buildVirtualAdviceInstr(rd);
        const is_single_cycle = (num_bytes == 8);

        try self.lookup_trace.recordVirtualAdvice(
            @intCast(self.state.cycle),
            base_pc,
            advice_load_instr,
            advice_value,
            true, // is_virtual
            !is_single_cycle, // do_not_update_pc (unless it's the only cycle)
            true, // is_first_in_sequence
            if (is_single_cycle) self.is_compressed else false,
        );

        try self.trace.addStep(.{
            .cycle = self.state.cycle,
            .pc = base_pc,
            .unexpanded_pc = base_pc,
            .instruction = advice_load_instr,
            .rs1_value = 0,
            .rs2_value = 0,
            .rd_pre_value = rd_pre,
            .rd_value = advice_value,
            .rd_index = rd,
            .rs1_index = 0,
            .rs2_index = 0,
            .rd_written = true,
            .rs1_read = false,
            .rs2_read = false,
            .memory_addr = null,
            .memory_pre_value = null,
            .memory_value = null,
            .is_memory_write = false,
            .next_pc = if (is_single_cycle) base_pc + pc_increment else base_pc,
            .is_compressed = if (is_single_cycle) self.is_compressed else false,
            .virtual_sequence_remaining = if (is_single_cycle) 0 else 2,
            .is_first_in_sequence = true,
            .is_last_in_sequence = is_single_cycle,
        });
        self.state.cycle += 1;
        self.registers.tick();

        if (is_single_cycle) {
            self.prev_pc = base_pc;
            self.state.pc = base_pc + pc_increment;
            return true;
        }

        // ── Step 2: VirtualMULI(rd, rd, 1 << shift_amount) — matches SLLI decomposition ──
        const multiplier: u64 = @as(u64, 1) << shift_amount;
        const shifted_left: u64 = advice_value *% multiplier;
        const rd_pre_step2 = try self.registers.read(rd);
        if (rd != 0) try self.registers.write(rd, shifted_left);

        const vmuli_instr: u32 = buildVirtualMULIInstr(rd, rd, @intCast(shift_amount));
        try self.lookup_trace.recordVirtualMULI(
            @intCast(self.state.cycle),
            base_pc,
            vmuli_instr,
            advice_value,
            multiplier,
            true, // is_virtual
            true, // do_not_update_pc: middle of sequence (vsr=1)
            false, // is_first_in_sequence
            false, // is_compressed (not last)
        );

        try self.trace.addStep(.{
            .cycle = self.state.cycle,
            .pc = base_pc,
            .unexpanded_pc = base_pc,
            .instruction = vmuli_instr,
            .rs1_value = advice_value,
            .rs2_value = 0,
            .rd_pre_value = rd_pre_step2,
            .rd_value = shifted_left,
            .rd_index = rd,
            .rs1_index = rd,
            .rs2_index = 0,
            .rd_written = true,
            .rs1_read = true,
            .rs2_read = false,
            .memory_addr = null,
            .memory_pre_value = null,
            .memory_value = null,
            .is_memory_write = false,
            .next_pc = base_pc,
            .is_compressed = false,
            .virtual_sequence_remaining = 1,
            .is_first_in_sequence = false,
        });
        self.state.cycle += 1;
        self.registers.tick();

        // ── Step 3: VirtualSRAI(rd, rd, bitmask) — matches SRAI decomposition ──
        const signed_left: i64 = @bitCast(shifted_left);
        const result_signed: i64 = signed_left >> @intCast(shift_amount);
        const result: u64 = @bitCast(result_signed);
        const rd_pre_step3 = try self.registers.read(rd);
        if (rd != 0) try self.registers.write(rd, result);

        const vsrai_instr: u32 = buildVirtualSRAIInstr(rd, rd, @intCast(shift_amount));
        try self.lookup_trace.recordVirtualSRAI(
            @intCast(self.state.cycle),
            base_pc,
            vsrai_instr,
            shifted_left,
            true, // is_virtual
            false, // do_not_update_pc: last in sequence (vsr=0)
            false, // is_first_in_sequence
            self.is_compressed, // is_compressed propagated to last step
        );

        try self.trace.addStep(.{
            .cycle = self.state.cycle,
            .pc = base_pc,
            .unexpanded_pc = base_pc,
            .instruction = vsrai_instr,
            .rs1_value = shifted_left,
            .rs2_value = 0,
            .rd_pre_value = rd_pre_step3,
            .rd_value = result,
            .rd_index = rd,
            .rs1_index = rd,
            .rs2_index = 0,
            .rd_written = true,
            .rs1_read = true,
            .rs2_read = false,
            .memory_addr = null,
            .memory_pre_value = null,
            .memory_value = null,
            .is_memory_write = false,
            .next_pc = base_pc + pc_increment,
            .is_compressed = self.is_compressed,
            .virtual_sequence_remaining = 0,
            .is_first_in_sequence = false,
            .is_last_in_sequence = true,
        });
        self.state.cycle += 1;
        self.registers.tick();

        self.prev_pc = base_pc;
        self.state.pc = base_pc + pc_increment;
        return true;
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
    /// Synthetic instruction encoding for VirtualSignExtendWord.
    /// Uses RISC-V custom-0 opcode (0x0B) with funct3=0.
    /// Encoding: I-type: imm[11:0] | rs1[19:15] | funct3[14:12] | rd[11:7] | opcode[6:0]
    pub const VIRTUAL_SIGN_EXTEND_WORD_OPCODE: u7 = 0x0B;

    /// Synthetic instruction encoding for VirtualMULI.
    /// Uses RISC-V custom-1 opcode (0x2B) with funct3=0.
    /// Encoding: I-type: imm[11:0] | rs1[19:15] | funct3[14:12] | rd[11:7] | opcode[6:0]
    /// The imm field stores the SHIFT AMOUNT (0-63), NOT the multiplier (1 << shamt).
    /// This ensures the value always fits in 12 bits. The R1CS witness and lookup
    /// compute the multiplier as (1 << shamt) from this field.
    pub const VIRTUAL_MULI_OPCODE: u7 = 0x2B;

    /// Synthetic instruction encoding for VirtualSRLI.
    /// Uses RISC-V custom-2 opcode (0x5B) with funct3=0.
    /// Encoding: I-type: shift_amount[11:0] | rs1[19:15] | funct3[14:12] | rd[11:7] | opcode[6:0]
    /// The imm field stores the TOTAL SHIFT AMOUNT (for bitmask reconstruction).
    /// The 64-bit bitmask is computed as: ones = (1 << (64 - shift)) - 1; bitmask = ones << shift
    pub const VIRTUAL_SRLI_OPCODE: u7 = 0x5B;

    /// Synthetic opcodes for inline sequence virtual instructions.
    /// These don't correspond to real RISC-V opcodes - they're only used to encode
    /// register indices in the trace. They use RISC-V custom opcode space (0x02, 0x22, 0x42, 0x62).
    /// VirtualAdvice: J-type format (rd only) - uses custom-0b
    pub const VIRTUAL_ADVICE_OPCODE: u7 = 0x02;
    /// VirtualAssertEQ: B-type format (rs1, rs2) - uses custom-0c
    pub const VIRTUAL_ASSERT_EQ_OPCODE: u7 = 0x22;
    /// VirtualZeroExtendWord: I-type format (rd, rs1) - uses custom-0d
    pub const VIRTUAL_ZERO_EXTEND_WORD_OPCODE: u7 = 0x42;
    /// VirtualAssertValidUnsignedRemainder: B-type format (rs1, rs2) - uses custom-0e
    pub const VIRTUAL_ASSERT_VALID_UNSIGNED_REMAINDER_OPCODE: u7 = 0x62;
    /// VirtualSRAI: I-type format (rd, rs1, shift_amount) - same opcode as VirtualSRLI but funct3=5
    pub const VIRTUAL_SRAI_OPCODE: u7 = 0x5B;
    pub const VIRTUAL_SRAI_FUNCT3: u3 = 5;
    /// VirtualAssertValidDiv0: B-type format (rs1, rs2) - same opcode as VirtualAssertEQ but funct3=1
    pub const VIRTUAL_ASSERT_VALID_DIV0_OPCODE: u7 = 0x22;
    pub const VIRTUAL_ASSERT_VALID_DIV0_FUNCT3: u3 = 1;
    /// VirtualChangeDivisorW: R-type format (rd, rs1, rs2) - uses OP-32 with M-ext funct7 and REMW funct3
    pub const VIRTUAL_CHANGE_DIVISOR_W_OPCODE: u7 = 0x3b;
    pub const VIRTUAL_CHANGE_DIVISOR_W_FUNCT3: u3 = 6;
    pub const VIRTUAL_CHANGE_DIVISOR_W_FUNCT7: u7 = 0x01;

    /// VirtualROTRI/VirtualROTRIW: I-type format (rd, rs1, bitmask_imm) - uses custom opcode 0x6B
    /// The imm field stores the ROTATION AMOUNT (0-63). The 64-bit bitmask is
    /// reconstructed as: ones = (1 << (width - rotation)) - 1; bitmask = ones << rotation
    pub const VIRTUAL_ROTRI_OPCODE: u7 = 0x6B;
    pub const VIRTUAL_ROTRI_FUNCT3: u3 = 0;
    pub const VIRTUAL_ROTRIW_FUNCT3: u3 = 1;

    /// ANDN: R-type format (rd, rs1, rs2) - standard OP opcode 0x33 with funct7=0x20, funct3=7
    /// Computes rd = rs1 & ~rs2 (Zbb bit manipulation extension)
    pub const ANDN_FUNCT7: u7 = 0x20;
    pub const ANDN_FUNCT3: u3 = 7;

    /// Build a synthetic ANDN instruction word (R-type: opcode=0x33, funct3=7, funct7=0x20)
    fn buildANDNInstr(rd: u8, rs1: u8, rs2: u8) u32 {
        return (@as(u32, ANDN_FUNCT7) << 25) |
            (@as(u32, rs2 & 0x1F) << 20) |
            (@as(u32, rs1 & 0x1F) << 15) |
            (@as(u32, ANDN_FUNCT3) << 12) |
            (@as(u32, rd & 0x1F) << 7) |
            0x33; // OP opcode
    }

    /// Build a synthetic VirtualROTRI instruction word (I-type: opcode=0x6B, funct3=0)
    /// rotation: the rotation amount (0-63), stored in I-type imm field.
    fn buildVirtualROTRIInstr(rd: u8, rs1: u8, rotation: u6) u32 {
        return (@as(u32, rotation) << 20) |
            (@as(u32, rs1 & 0x1F) << 15) |
            (@as(u32, VIRTUAL_ROTRI_FUNCT3) << 12) |
            (@as(u32, rd & 0x1F) << 7) |
            @as(u32, VIRTUAL_ROTRI_OPCODE);
    }

    /// Build a synthetic VirtualROTRIW instruction word (I-type: opcode=0x6B, funct3=1)
    /// rotation: the rotation amount (0-31), stored in I-type imm field.
    fn buildVirtualROTRIWInstr(rd: u8, rs1: u8, rotation: u5) u32 {
        return (@as(u32, rotation) << 20) |
            (@as(u32, rs1 & 0x1F) << 15) |
            (@as(u32, VIRTUAL_ROTRIW_FUNCT3) << 12) |
            (@as(u32, rd & 0x1F) << 7) |
            @as(u32, VIRTUAL_ROTRI_OPCODE);
    }

    /// Build a synthetic VirtualMULI instruction word.
    /// shamt: shift amount (0-63) stored in the I-type immediate field.
    /// The actual multiplier (1 << shamt) is computed by R1CS witness/lookup code.
    fn buildVirtualMULIInstr(rd: u8, rs1: u8, shamt: u6) u32 {
        // Mask register indices to 5 bits to prevent overflow into adjacent fields
        return (@as(u32, shamt) << 20) |
            (@as(u32, rs1 & 0x1F) << 15) |
            (0 << 12) | // funct3 = 0
            (@as(u32, rd & 0x1F) << 7) |
            @as(u32, VIRTUAL_MULI_OPCODE);
    }

    /// Build a synthetic VirtualSignExtendWord instruction word
    fn buildVirtualSignExtendWordInstr(rd: u8, rs1: u8) u32 {
        // Mask register indices to 5 bits to prevent overflow into adjacent fields
        return @as(u32, 0) | // imm = 0
            (@as(u32, rs1 & 0x1F) << 15) |
            (0 << 12) | // funct3 = 0
            (@as(u32, rd & 0x1F) << 7) |
            @as(u32, VIRTUAL_SIGN_EXTEND_WORD_OPCODE);
    }

    /// Build a synthetic VirtualSRLI instruction word.
    /// total_shift: the total shift amount (0-63), stored in I-type imm field.
    /// The 64-bit bitmask is: ones = (1 << (64 - shift)) - 1; bitmask = ones << shift
    fn buildVirtualSRLIInstr(rd: u8, rs1: u8, total_shift: u7) u32 {
        // Mask register indices to 5 bits to prevent overflow into adjacent fields
        return (@as(u32, total_shift) << 20) |
            (@as(u32, rs1 & 0x1F) << 15) |
            (0 << 12) | // funct3 = 0
            (@as(u32, rd & 0x1F) << 7) |
            @as(u32, VIRTUAL_SRLI_OPCODE);
    }

    /// Build a synthetic VirtualAdvice instruction word (J-type: rd only)
    fn buildVirtualAdviceInstr(rd: u8) u32 {
        // Mask register index to 5 bits to prevent overflow into adjacent fields
        return (@as(u32, rd & 0x1F) << 7) |
            @as(u32, VIRTUAL_ADVICE_OPCODE);
    }

    /// Build a synthetic VirtualAssertEQ instruction word (B-type: rs1, rs2)
    fn buildVirtualAssertEQInstr(rs1: u8, rs2: u8) u32 {
        // Mask register indices to 5 bits to prevent overflow into adjacent fields
        return (@as(u32, rs2 & 0x1F) << 20) |
            (@as(u32, rs1 & 0x1F) << 15) |
            @as(u32, VIRTUAL_ASSERT_EQ_OPCODE);
    }

    /// Build a synthetic VirtualZeroExtendWord instruction word (I-type: rd, rs1)
    fn buildVirtualZeroExtendWordInstr(rd: u8, rs1: u8) u32 {
        // Mask register indices to 5 bits to prevent overflow into adjacent fields
        return (@as(u32, rs1 & 0x1F) << 15) |
            (@as(u32, rd & 0x1F) << 7) |
            @as(u32, VIRTUAL_ZERO_EXTEND_WORD_OPCODE);
    }

    /// Build a synthetic VirtualAssertValidUnsignedRemainder instruction word (B-type: rs1, rs2)
    fn buildVirtualAssertValidUnsignedRemainderInstr(rs1: u8, rs2: u8) u32 {
        // Mask register indices to 5 bits to prevent overflow into adjacent fields
        return (@as(u32, rs2 & 0x1F) << 20) |
            (@as(u32, rs1 & 0x1F) << 15) |
            @as(u32, VIRTUAL_ASSERT_VALID_UNSIGNED_REMAINDER_OPCODE);
    }

    /// Build a standard R-type MUL instruction: MUL rd, rs1, rs2
    /// opcode=0x33, funct7=0x01, funct3=0x00
    fn buildMULInstr(rd: u8, rs1: u8, rs2: u8) u32 {
        // IMPORTANT: Mask register indices to 5 bits to prevent overflow into
        // funct3/funct7 fields. Virtual registers (32+) have bit 5 set which
        // would corrupt adjacent instruction fields if not masked.
        return (0x01 << 25) | // funct7 = 0x01 (M extension)
            (@as(u32, rs2 & 0x1F) << 20) |
            (@as(u32, rs1 & 0x1F) << 15) |
            (0x00 << 12) | // funct3 = 0 (MUL)
            (@as(u32, rd & 0x1F) << 7) |
            0x33; // opcode = OP
    }

    /// Build a standard R-type ADD instruction: ADD rd, rs1, rs2
    /// opcode=0x33, funct7=0x00, funct3=0x00
    fn buildADDInstr(rd: u8, rs1: u8, rs2: u8) u32 {
        // IMPORTANT: Mask register indices to 5 bits to prevent overflow.
        return (0x00 << 25) | // funct7 = 0x00 (ADD)
            (@as(u32, rs2 & 0x1F) << 20) |
            (@as(u32, rs1 & 0x1F) << 15) |
            (0x00 << 12) | // funct3 = 0 (ADD)
            (@as(u32, rd & 0x1F) << 7) |
            0x33; // opcode = OP
    }

    /// Build a standard R-type SUB instruction: SUB rd, rs1, rs2
    /// opcode=0x33, funct7=0x20, funct3=0x00
    fn buildSUBInstr(rd: u8, rs1: u8, rs2: u8) u32 {
        return (0x20 << 25) | // funct7 = 0x20 (SUB)
            (@as(u32, rs2 & 0x1F) << 20) |
            (@as(u32, rs1 & 0x1F) << 15) |
            (0x00 << 12) | // funct3 = 0
            (@as(u32, rd & 0x1F) << 7) |
            0x33; // opcode = OP
    }

    /// Build a standard R-type XOR instruction: XOR rd, rs1, rs2
    /// opcode=0x33, funct7=0x00, funct3=0x04
    fn buildXORInstr(rd: u8, rs1: u8, rs2: u8) u32 {
        return (0x00 << 25) | // funct7 = 0x00
            (@as(u32, rs2 & 0x1F) << 20) |
            (@as(u32, rs1 & 0x1F) << 15) |
            (0x04 << 12) | // funct3 = 4 (XOR)
            (@as(u32, rd & 0x1F) << 7) |
            0x33; // opcode = OP
    }

    /// Build a synthetic VirtualSRAI instruction word.
    /// shift: shift amount (0-63), stored in I-type imm field.
    /// Uses opcode=0x5B, funct3=5 to distinguish from VirtualSRLI (funct3=0).
    fn buildVirtualSRAIInstr(rd: u8, rs1: u8, shift: u7) u32 {
        return (@as(u32, shift) << 20) |
            (@as(u32, rs1 & 0x1F) << 15) |
            (@as(u32, VIRTUAL_SRAI_FUNCT3) << 12) |
            (@as(u32, rd & 0x1F) << 7) |
            @as(u32, VIRTUAL_SRAI_OPCODE);
    }

    /// Build a synthetic VirtualAssertValidDiv0 instruction word (B-type: rs1, rs2)
    /// Uses opcode=0x22, funct3=1 to distinguish from VirtualAssertEQ (funct3=0).
    fn buildVirtualAssertValidDiv0Instr(rs1: u8, rs2: u8) u32 {
        return (@as(u32, rs2 & 0x1F) << 20) |
            (@as(u32, rs1 & 0x1F) << 15) |
            (@as(u32, VIRTUAL_ASSERT_VALID_DIV0_FUNCT3) << 12) |
            @as(u32, VIRTUAL_ASSERT_VALID_DIV0_OPCODE);
    }

    /// Build a synthetic VirtualChangeDivisorW instruction word (R-type: rd, rs1, rs2)
    /// Uses opcode=0x3b, funct3=6, funct7=0x01.
    fn buildVirtualChangeDivisorWInstr(rd: u8, rs1: u8, rs2: u8) u32 {
        return (@as(u32, VIRTUAL_CHANGE_DIVISOR_W_FUNCT7) << 25) |
            (@as(u32, rs2 & 0x1F) << 20) |
            (@as(u32, rs1 & 0x1F) << 15) |
            (@as(u32, VIRTUAL_CHANGE_DIVISOR_W_FUNCT3) << 12) |
            (@as(u32, rd & 0x1F) << 7) |
            @as(u32, VIRTUAL_CHANGE_DIVISOR_W_OPCODE);
    }

    const WExtType = enum { ADDIW, ADDW, SUBW, MULW, SLLIW, SRLIW };

    /// Check if an instruction is a W-extension instruction that needs decomposition
    fn isWExtension(decoded: zkvm.instruction.DecodedInstruction) ?WExtType {
        switch (decoded.opcode) {
            .OP_IMM_32 => {
                if (decoded.funct3 == 0) return .ADDIW;
                if (decoded.funct3 == 1) return .SLLIW;
                if (decoded.funct3 == 5 and (decoded.funct7 & 0x20) == 0) return .SRLIW;
                return null;
            },
            .OP_32 => {
                if (decoded.funct7 == 0x01) {
                    if (decoded.funct3 == 0) return .MULW;
                    return null; // DIVW, REMW etc. have complex decompositions
                }
                if (decoded.funct3 == 0) {
                    if ((decoded.funct7 & 0x20) != 0) return .SUBW;
                    return .ADDW;
                }
                return null; // SLLW, SRLW, SRAW have complex decompositions
            },
            else => return null,
        }
    }

    /// Build a synthetic base instruction for W-extension decomposition.
    /// ADDIW → ADDI (same operands): opcode=0x13, same rd/rs1/imm
    /// ADDW → ADD (same operands): opcode=0x33, funct7=0, same rd/rs1/rs2
    /// SUBW → SUB (same operands): opcode=0x33, funct7=0x20, same rd/rs1/rs2
    /// MULW → MUL (same operands): opcode=0x33, funct7=0x01, same rd/rs1/rs2
    fn buildBaseInstruction(original: u32, w_type: WExtType) u32 {
        const rd: u32 = (original >> 7) & 0x1f;
        const rs1: u32 = (original >> 15) & 0x1f;
        const rs2: u32 = (original >> 20) & 0x1f;
        const imm_bits: u32 = original & 0xFFF00000; // upper 12 bits for I-type

        return switch (w_type) {
            .ADDIW => imm_bits | (rs1 << 15) | (0b000 << 12) | (rd << 7) | 0x13, // ADDI
            .ADDW => (0x00 << 25) | (rs2 << 20) | (rs1 << 15) | (0b000 << 12) | (rd << 7) | 0x33, // ADD
            .SUBW => (0x20 << 25) | (rs2 << 20) | (rs1 << 15) | (0b000 << 12) | (rd << 7) | 0x33, // SUB
            .MULW => (0x01 << 25) | (rs2 << 20) | (rs1 << 15) | (0b000 << 12) | (rd << 7) | 0x33, // MUL
            .SLLIW => blk: {
                // SLLIW → VirtualMULI(rd, rs1, shamt)
                // Store shift amount in imm field; multiplier = 1 << shamt
                const shamt: u5 = @truncate(rs2); // shamt in bits [24:20]
                break :blk buildVirtualMULIInstr(@truncate(rd), @truncate(rs1), @as(u6, shamt));
            },
            .SRLIW => unreachable, // Handled separately in stepSRLIW
        };
    }

    /// Check if instruction is REMUW (needs 12-step inline sequence decomposition)
    /// REMUW: opcode=OP_32 (0x3B), funct7=0x01 (M extension), funct3=0b111
    fn isREMUW(decoded: zkvm.instruction.DecodedInstruction) bool {
        return decoded.opcode == .OP_32 and decoded.funct7 == 0x01 and decoded.funct3 == 0b111;
    }

    /// Check if instruction is DIVW (needs 21-step inline sequence decomposition)
    /// DIVW: opcode=OP_32 (0x3B), funct7=0x01 (M extension), funct3=0b100
    fn isDIVW(decoded: zkvm.instruction.DecodedInstruction) bool {
        return decoded.opcode == .OP_32 and decoded.funct7 == 0x01 and decoded.funct3 == 0b100;
    }

    /// Check if instruction is REMW (needs 21-step inline sequence decomposition)
    /// REMW: opcode=OP_32 (0x3B), funct7=0x01 (M extension), funct3=0b110
    fn isREMW(decoded: zkvm.instruction.DecodedInstruction) bool {
        return decoded.opcode == .OP_32 and decoded.funct7 == 0x01 and decoded.funct3 == 0b110;
    }

    /// Check if instruction is SLLI (which decomposes to a single VirtualMULI step)
    fn isSLLI(decoded: zkvm.instruction.DecodedInstruction) bool {
        return decoded.opcode == .OP_IMM and decoded.funct3 == 0b001;
    }

    /// Check if instruction is SRLI (which decomposes to a single VirtualSRLI step)
    /// Note: SRAI has funct7 bit 5 set, SRLI does not
    fn isSRLI(decoded: zkvm.instruction.DecodedInstruction) bool {
        return decoded.opcode == .OP_IMM and decoded.funct3 == 0b101 and (decoded.funct7 & 0x20) == 0;
    }

    fn isSRAI(decoded: zkvm.instruction.DecodedInstruction) bool {
        return decoded.opcode == .OP_IMM and decoded.funct3 == 0b101 and (decoded.funct7 & 0x20) != 0;
    }

    fn isSLL(decoded: zkvm.instruction.DecodedInstruction) bool {
        return decoded.opcode == .OP and decoded.funct3 == 0b001 and decoded.funct7 == 0;
    }

    fn isSRL(decoded: zkvm.instruction.DecodedInstruction) bool {
        return decoded.opcode == .OP and decoded.funct3 == 0b101 and decoded.funct7 == 0;
    }

    fn isSRA(decoded: zkvm.instruction.DecodedInstruction) bool {
        return decoded.opcode == .OP and decoded.funct3 == 0b101 and decoded.funct7 == 0x20;
    }

    fn isSubWordLoad(decoded: zkvm.instruction.DecodedInstruction) bool {
        return decoded.opcode == .LOAD and decoded.funct3 != 0b011; // Not LD
    }

    fn isSubWordStore(decoded: zkvm.instruction.DecodedInstruction) bool {
        return decoded.opcode == .STORE and decoded.funct3 != 0b011; // Not SD
    }

    /// Check if instruction is a jolt-inline instruction (opcode 0x0B = custom-0)
    fn isInlineInstruction(instruction: u32) bool {
        return (instruction & 0x7f) == 0x0B;
    }

    /// Check if instruction is CSRRW (opcode=0x73, funct3=1)
    fn isCSRRW(decoded: zkvm.instruction.DecodedInstruction) bool {
        return decoded.opcode == .SYSTEM and decoded.funct3 == 1;
    }

    /// Check if instruction is CSRRS (opcode=0x73, funct3=2)
    fn isCSRRS(decoded: zkvm.instruction.DecodedInstruction) bool {
        return decoded.opcode == .SYSTEM and decoded.funct3 == 2;
    }

    /// Check if instruction is MRET (opcode=0x73, funct3=0, funct12=0x302)
    fn isMRET(instruction: u32) bool {
        return (instruction & 0x7f) == 0x73 and
            ((instruction >> 12) & 0x7) == 0 and
            ((instruction >> 20) & 0xFFF) == 0x302;
    }

    pub fn step(self: *Emulator) !bool {
        // Infinite loop detection (matching Jolt's termination heuristic)
        // If PC hasn't changed since last step, the program has terminated
        // via an infinite loop (e.g., JAL x0, 0). This is standard for bare-metal
        // RISC-V programs without an OS to return to.
        // Skip this check on first step (prev_pc == 0 and state.pc != 0)
        if (self.prev_pc != 0 and self.prev_pc == self.state.pc) {
            dbg("[TRACE] Detected infinite loop at PC 0x{x:0>8}, cycle {d}\n", .{ self.state.pc, self.state.cycle });
            return false; // Program terminated via infinite loop
        }

        // Fetch instruction
        const instruction = try self.fetchInstruction();
        self.state.instruction = instruction;

        // Decode
        const decoded = zkvm.instruction.DecodedInstruction.decode(instruction);

        // Check if this is a W-extension instruction that needs decomposition
        const w_ext = isWExtension(decoded);

        if (w_ext) |w_type| {
            if (w_type == .SRLIW) {
                // SRLIW has a 3-step decomposition (unlike the 2-step ones)
                return try self.stepSRLIW(instruction, decoded);
            }
            // Other W-extension decomposition: generate 2 trace steps
            return try self.stepWExtension(instruction, decoded, w_type);
        }

        // Check if this is SLLI (decomposes to single VirtualMULI)
        if (isSLLI(decoded)) {
            return try self.stepSLLI(instruction, decoded);
        }

        // Check if this is SRLI (decomposes to single VirtualSRLI)
        if (isSRLI(decoded)) {
            return try self.stepSRLI(instruction, decoded);
        }

        // Check if this is SRAI (decomposes to single VirtualSRAI)
        if (isSRAI(decoded)) {
            return try self.stepSRAI(instruction, decoded);
        }

        // Check if this is SLL (2-step: VirtualPow2 + MUL)
        if (isSLL(decoded)) {
            return try self.stepSLL(instruction, decoded);
        }

        // Check if this is SRL (2-step: VirtualShiftRightBitmask + VirtualSRL)
        if (isSRL(decoded)) {
            return try self.stepSRL(instruction, decoded);
        }

        // Check if this is SRA (2-step: VirtualShiftRightBitmask + VirtualSRA)
        if (isSRA(decoded)) {
            return try self.stepSRA(instruction, decoded);
        }

        // Check if this is a sub-word load (LB/LBU/LH/LHU/LW/LWU)
        if (isSubWordLoad(decoded)) {
            return try self.stepSubWordLoad(instruction, decoded);
        }

        // Check if this is a sub-word store (SB/SH/SW)
        if (isSubWordStore(decoded)) {
            return try self.stepSubWordStore(instruction, decoded);
        }

        // Check if this is REMUW (12-step inline sequence decomposition)
        if (isREMUW(decoded)) {
            return try self.stepREMUW(instruction, decoded);
        }

        // Check if this is REMW or DIVW (21-step inline sequence decomposition)
        if (isREMW(decoded) or isDIVW(decoded)) {
            return try self.stepREMWDIVW(instruction, decoded);
        }

        // Check if this is a jolt-inline instruction (opcode 0x0B)
        if (isInlineInstruction(instruction)) {
            return try self.stepInline(instruction);
        }

        if (isCSRRW(decoded)) return try self.stepCSRRW(instruction, decoded);
        if (isCSRRS(decoded)) return try self.stepCSRRS(instruction, decoded);
        if (isMRET(instruction)) return try self.stepMRET(instruction, decoded);

        // VirtualRev8W (opcode 0x5B funct3=0): byte-swap each 32-bit half
        const op_raw: u8 = @truncate(instruction & 0x7F);
        const f3_raw: u3 = @truncate((instruction >> 12) & 0x7);
        if (op_raw == 0x5B) {
            switch (f3_raw) {
                0 => return try self.stepVirtualRev8W(instruction, decoded),
                3 => return try self.stepAdviceLoadSignExt(decoded, 1), // AdviceLB
                4 => return try self.stepAdviceLoadSignExt(decoded, 2), // AdviceLH
                5 => return try self.stepAdviceLoadSignExt(decoded, 4), // AdviceLW
                6 => return try self.stepAdviceLoadSignExt(decoded, 8), // AdviceLD (single cycle)
                // funct3 = 1, 2, 7: fall through to stepNormal (VirtualAssertEQ,
                // VirtualHostIO, VirtualAdviceLen — NoOp passthroughs).
                else => {},
            }
        }

        // Standard (non-W-extension) instruction execution
        return try self.stepNormal(instruction, decoded);
    }

    /// Execute VirtualRev8W: rd = rev8w(rs1) where rev8w byte-swaps each 32-bit half.
    /// Single-cycle trace step. Format: I-type (rd, rs1).
    fn stepVirtualRev8W(
        self: *Emulator,
        instruction: u32,
        decoded: zkvm.instruction.DecodedInstruction,
    ) !bool {
        const pc_increment: u64 = if (self.is_compressed) 2 else 4;
        const rs1_value = try self.registers.read(decoded.rs1);
        const rd_pre = try self.registers.read(decoded.rd);

        // rev8w: swap bytes in each 32-bit half independently
        const lo: u32 = @byteSwap(@as(u32, @truncate(rs1_value)));
        const hi: u32 = @byteSwap(@as(u32, @truncate(rs1_value >> 32)));
        const result: u64 = @as(u64, lo) | (@as(u64, hi) << 32);

        if (decoded.rd != 0) {
            try self.registers.write(decoded.rd, result);
        }

        // Build a synthetic instruction word with opcode 0x7B funct3=0 to distinguish
        // VirtualRev8W from our internal VirtualSRLI (which uses 0x5B funct3=0).
        // This mapping is only used by the R1CS witness and Stage 5 lookup-index
        // computation inside Zolt; the serialized bytecode entry still uses variant
        // .VirtualRev8W so Jolt's verifier interprets it correctly.
        const synth_instr: u32 = (@as(u32, decoded.rs1 & 0x1F) << 15) |
            (@as(u32, decoded.rd & 0x1F) << 7) |
            @as(u32, 0x7B);

        try self.lookup_trace.recordVirtualRev8W(
            @intCast(self.state.cycle),
            self.state.pc,
            synth_instr,
            rs1_value,
            false, // is_virtual: standalone instruction (not in inline sequence)
            false, // do_not_update_pc
            false, // is_first_in_sequence
            self.is_compressed,
        );

        try self.trace.addStep(.{
            .cycle = self.state.cycle,
            .pc = self.state.pc,
            .unexpanded_pc = self.state.pc,
            .instruction = synth_instr,
            .rs1_value = rs1_value,
            .rs2_value = 0,
            .rd_pre_value = if (decoded.rd == 0) 0 else rd_pre,
            .rd_value = if (decoded.rd == 0) 0 else result,
            .rd_index = decoded.rd,
            .rs1_index = decoded.rs1,
            .rs2_index = 0,
            .rd_written = decoded.rd != 0,
            .rs1_read = true,
            .rs2_read = false,
            .memory_addr = null,
            .memory_pre_value = null,
            .memory_value = null,
            .is_memory_write = false,
            .next_pc = self.state.pc + pc_increment,
            .is_compressed = self.is_compressed,
        });
        _ = instruction; // original ELF word, not used in the synthetic trace step

        self.prev_pc = self.state.pc;
        self.state.pc += pc_increment;
        self.state.cycle += 1;
        self.registers.tick();
        return true;
    }

    /// Execute a standard (non-W-extension) instruction as a single trace step
    fn stepNormal(self: *Emulator, instruction: u32, decoded: zkvm.instruction.DecodedInstruction) !bool {
        // Upstream Jolt remaps JAL/JALR with rd=x0 to virtual register 40.
        // Compute effective_rd early so pre-value read uses the correct register.
        const FIRST_VIRTUAL_ALLOC_REG: u8 = 40; // 32 (RISCV) + 8 (reserved)
        const is_jump_rd0 = (decoded.opcode == .JAL or decoded.opcode == .JALR) and decoded.rd == 0;
        const effective_rd: u8 = if (is_jump_rd0) FIRST_VIRTUAL_ALLOC_REG else decoded.rd;

        // Record pre-execution state
        const rs1_value = try self.registers.read(decoded.rs1);
        const rs2_value = try self.registers.read(decoded.rs2);
        // Capture rd pre-value from effective register (vr40 for JAL/JALR x0)
        const rd_pre_value = try self.registers.read(effective_rd);

        // Record lookup trace for Shout proofs (before execution)
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

        // For memory writes, get pre-value from the RAM trace (it was just recorded during write)
        // The RAM trace records both pre and post values for each write operation
        const memory_pre_value: ?u64 = if (result.is_memory_write) blk: {
            // The most recent write in RAM trace corresponds to this cycle's write
            const ram_accesses = self.ram.trace.accesses.items;
            if (ram_accesses.len > 0) {
                const last_access = ram_accesses[ram_accesses.len - 1];
                if (last_access.op == .Write and last_access.timestamp == self.state.cycle) {
                    break :blk last_access.pre_value;
                }
            }
            break :blk null;
        } else null;

        // Determine which registers are read/written based on opcode
        const opcode = decoded.opcode;
        const opcode_raw: u8 = @truncate(@as(u32, instruction) & 0x7F);
        const funct3_raw: u3 = @truncate((instruction >> 12) & 0x7);
        // Jolt SDK custom opcodes (0x5B with funct3 != 0/5) — VirtualHostIO and
        // VirtualAdviceLoad/Len — use FormatI with rs1 in the bytecode entry.
        // The trace must record the rs1 read so the Stage 4 Rs1Ra polynomial
        // matches the bytecode val_poly. Without this, BCRAF Stage 4 sumcheck
        // mismatches the opening claim and Stage 6 verification fails.
        // funct3 == 0/5 are VirtualSRL/SRA which are handled by their own
        // step handlers (stepSRL/stepSRA/stepSRLI) that set rs1_read/rs2_read
        // explicitly, so we don't override here.
        const is_sdk_custom_2 = opcode_raw == 0x5B and funct3_raw != 0 and funct3_raw != 5;
        const reads_rs1 = is_sdk_custom_2 or switch (opcode) {
            .OP_IMM, .LOAD, .JALR, .OP_IMM_32, .OP, .OP_32, .STORE, .BRANCH => true,
            else => false,
        };
        const reads_rs2 = switch (opcode) {
            .OP, .OP_32, .STORE, .BRANCH => true,
            else => false,
        };
        // Jolt includes rd=0 writes in the RdWa polynomial - the trace captures
        // cpu.x[0] before and after execution (both 0), creating a (0→0) transition.
        // We must match this: instructions with an rd field always have rd_written=true,
        // even when rd=0. Only STORE and BRANCH have no rd field (rd=None in Jolt).
        const writes_rd = switch (opcode) {
            .STORE, .BRANCH => false,
            else => true, // All other instructions have rd field (even rd=0)
        };

        // For JAL/JALR with rd=x0 remapped to vr40, write the link address there
        if (is_jump_rd0) {
            try self.registers.write(FIRST_VIRTUAL_ALLOC_REG, result.rd_value);
        }

        const actual_rd_value = if (effective_rd == 0) @as(u64, 0) else result.rd_value;
        const actual_rd_pre_value = if (effective_rd == 0) @as(u64, 0) else rd_pre_value;

        // Record trace step
        try self.trace.addStep(.{
            .cycle = self.state.cycle,
            .pc = self.state.pc,
            .unexpanded_pc = self.state.pc,
            .instruction = instruction,
            .rs1_value = rs1_value,
            .rs2_value = rs2_value,
            .rd_pre_value = actual_rd_pre_value,
            .rd_value = actual_rd_value,
            .rd_index = effective_rd,
            .rs1_index = decoded.rs1,
            .rs2_index = decoded.rs2,
            .rd_written = writes_rd,
            .rs1_read = reads_rs1,
            .rs2_read = reads_rs2,
            .memory_addr = result.memory_addr,
            .memory_pre_value = memory_pre_value,
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

    /// Execute SLLI as a single VirtualMULI trace step.
    /// SLLI rd, rs1, shamt → VirtualMULI rd, rs1, (1 << shamt)
    /// This is NOT a virtual sequence - it's a standalone single-step decomposition.
    fn stepSLLI(
        self: *Emulator,
        _: u32, // original SLLI instruction (not used - we build VirtualMULI encoding)
        decoded: zkvm.instruction.DecodedInstruction,
    ) !bool {
        const rs1_value = try self.registers.read(decoded.rs1);
        const rs2_value = try self.registers.read(decoded.rs2);
        const rd_pre_value = try self.registers.read(decoded.rd);
        const pc_increment: u64 = if (self.is_compressed) 2 else 4;

        // Compute shift amount and multiplier
        const imm_u32: u32 = @bitCast(@as(i32, @truncate(decoded.imm)));
        const shamt: u6 = @intCast(imm_u32 & 0x3F);
        const multiplier: u64 = @as(u64, 1) << shamt;

        // Compute result: rs1 << shamt = rs1 * (1 << shamt)
        const result_val: u64 = rs1_value *% multiplier;

        // Build synthetic VirtualMULI instruction encoding
        // The imm field stores the shift amount (not the multiplier).
        // The R1CS witness computes multiplier = 1 << shamt from this field.
        const rd_u8: u8 = decoded.rd;
        const rs1_u8: u8 = decoded.rs1;
        const vmuli_instr = buildVirtualMULIInstr(rd_u8, rs1_u8, shamt);

        // Record lookup trace for VirtualMULI
        // SLLI → single VirtualMULI, NOT part of a virtual sequence
        // virtual_sequence_remaining = None in Jolt (not Some(0))
        try self.lookup_trace.recordVirtualMULI(
            @intCast(self.state.cycle),
            self.state.pc,
            vmuli_instr,
            rs1_value,
            multiplier,
            false, // is_virtual: false (standalone, vsr = None)
            false, // do_not_update_pc: false (not in a sequence)
            false, // is_first_in_sequence: false
            self.is_compressed, // is_compressed
        );

        // Write result to register
        try self.registers.write(decoded.rd, result_val);

        // Record trace step (as VirtualMULI, NOT SLLI)
        try self.trace.addStep(.{
            .cycle = self.state.cycle,
            .pc = self.state.pc,
            .unexpanded_pc = self.state.pc,
            .instruction = vmuli_instr,
            .rs1_value = rs1_value,
            .rs2_value = rs2_value, // rs2 field value (not used by VirtualMULI but must be consistent)
            .rd_pre_value = rd_pre_value,
            .rd_value = result_val,
            .rd_index = rd_u8,
            .rs1_index = rs1_u8,
            .rs2_index = 0,
            .rd_written = true, // Jolt includes rd=0 writes in RdWa polynomial
            .rs1_read = true,
            .rs2_read = false,
            .memory_addr = null,
            .memory_pre_value = null,
            .memory_value = null,
            .is_memory_write = false,
            .next_pc = self.state.pc + pc_increment,
            .is_compressed = self.is_compressed,
            .is_first_in_sequence = true, // Standalone VirtualMULI is first (and only) in sequence
            .is_last_in_sequence = true, // Also last (standalone)
        });

        // Update prev_pc for infinite loop detection
        self.prev_pc = self.state.pc;

        // Update state
        self.state.pc = self.state.pc + pc_increment;
        self.state.cycle += 1;
        self.registers.tick();

        return true;
    }

    /// Execute SRLI as a single VirtualSRLI trace step.
    /// SRLI rd, rs1, shamt → VirtualSRLI(rd, rs1, bitmask)
    fn stepSRLI(
        self: *Emulator,
        _: u32, // original SRLI instruction
        decoded: zkvm.instruction.DecodedInstruction,
    ) !bool {
        const rs1_value = try self.registers.read(decoded.rs1);
        const rs2_value = try self.registers.read(decoded.rs2);
        const rd_pre_value = try self.registers.read(decoded.rd);
        const pc_increment: u64 = if (self.is_compressed) 2 else 4;

        // Compute shift amount and bitmask
        const imm_u32: u32 = @bitCast(@as(i32, @truncate(decoded.imm)));
        const shamt: u7 = @intCast(imm_u32 & 0x3F);

        // Compute result: logical right shift
        const result_val: u64 = rs1_value >> @intCast(shamt);

        // Compute bitmask: ones in positions that contain data after shift
        const ones: u128 = (@as(u128, 1) << @intCast(64 - @as(u8, shamt))) - 1;
        const bitmask: u64 = @truncate(ones << shamt);

        // Build synthetic VirtualSRLI instruction encoding
        const rd_u8: u8 = decoded.rd;
        const rs1_u8: u8 = decoded.rs1;
        const vsrli_instr = buildVirtualSRLIInstr(rd_u8, rs1_u8, shamt);

        // Record lookup trace for VirtualSRLI
        try self.lookup_trace.recordVirtualSRLI(
            @intCast(self.state.cycle),
            self.state.pc,
            vsrli_instr,
            rs1_value,
            bitmask,
            result_val,
            false, // is_virtual: standalone (vsr = None)
            false, // do_not_update_pc
            false, // is_first_in_sequence
            self.is_compressed,
        );

        // Write result to register
        try self.registers.write(decoded.rd, result_val);

        // Record trace step (as VirtualSRLI, NOT SRLI)
        try self.trace.addStep(.{
            .cycle = self.state.cycle,
            .pc = self.state.pc,
            .unexpanded_pc = self.state.pc,
            .instruction = vsrli_instr,
            .rs1_value = rs1_value,
            .rs2_value = rs2_value,
            .rd_pre_value = rd_pre_value,
            .rd_value = result_val,
            .rd_index = rd_u8,
            .rs1_index = rs1_u8,
            .rs2_index = 0,
            .rd_written = true, // Jolt includes rd=0 writes in RdWa polynomial
            .rs1_read = true,
            .rs2_read = false,
            .memory_addr = null,
            .memory_pre_value = null,
            .memory_value = null,
            .is_memory_write = false,
            .next_pc = self.state.pc + pc_increment,
            .is_compressed = self.is_compressed,
            .is_first_in_sequence = true, // Standalone VirtualSRLI is first (and only) in sequence
            .is_last_in_sequence = true, // Also last (standalone)
        });

        self.prev_pc = self.state.pc;
        self.state.pc = self.state.pc + pc_increment;
        self.state.cycle += 1;
        self.registers.tick();
        return true;
    }

    /// Execute SRLIW as a 3-step virtual sequence:
    /// Step 1: VirtualMULI(v_rs1, rs1, 2^32) - shift left by 32
    /// Step 2: VirtualSRLI(rd, v_rs1, bitmask) - logical right shift
    /// Step 3: VirtualSignExtendWord(rd, rd, 0) - sign-extend 32→64
    fn stepSRLIW(
        self: *Emulator,
        _: u32, // original SRLIW instruction
        decoded: zkvm.instruction.DecodedInstruction,
    ) !bool {
        const rs1_value = try self.registers.read(decoded.rs1);
        _ = try self.registers.read(decoded.rs2); // rs2 not used by SRLIW
        _ = try self.registers.read(decoded.rd); // rd pre-value tracked per-step below
        const pc_increment: u64 = if (self.is_compressed) 2 else 4;

        const imm_u32: u32 = @bitCast(@as(i32, @truncate(decoded.imm)));
        const shamt: u7 = @intCast(imm_u32 & 0x1F); // 5-bit shift for W

        // Virtual register 32 (first virtual register)
        const v_rs1: u8 = 32;
        const rd_u8: u8 = decoded.rd;
        const rs1_u8: u8 = decoded.rs1;

        // === Step 1: VirtualMULI(v_rs1, rs1, 2^32) ===
        // This shifts rs1 left by 32, placing the lower 32 bits in the upper half
        const step1_multiplier: u64 = @as(u64, 1) << 32;
        const step1_result: u64 = rs1_value *% step1_multiplier;
        const step1_shamt: u6 = 32;
        const vmuli_instr = buildVirtualMULIInstr(v_rs1, rs1_u8, step1_shamt);

        // Read virtual register pre-value (0 initially, may be non-zero if SRLIW executed before)
        const v_rs1_pre_value = try self.registers.read(v_rs1);

        try self.lookup_trace.recordVirtualMULI(
            @intCast(self.state.cycle),
            self.state.pc,
            vmuli_instr,
            rs1_value,
            step1_multiplier,
            true, // is_virtual: part of SRLIW virtual sequence (vsr=Some(2))
            true, // do_not_update_pc: vsr=2, 2!=0
            true, // is_first_in_sequence
            self.is_compressed,
        );

        // Write step1 result to virtual register
        try self.registers.write(v_rs1, step1_result);

        // Record trace step 1: VirtualMULI(v_rs1=32, rs1, 2^32)
        try self.trace.addStep(.{
            .cycle = self.state.cycle,
            .pc = self.state.pc,
            .unexpanded_pc = self.state.pc,
            .instruction = vmuli_instr,
            .rs1_value = rs1_value,
            .rs2_value = 0,
            .rd_pre_value = v_rs1_pre_value, // Virtual register's current value
            .rd_value = step1_result,
            .rd_index = v_rs1, // virtual register 32
            .rs1_index = rs1_u8,
            .rs2_index = 0,
            .rd_written = true, // writes to virtual register
            .rs1_read = true,
            .rs2_read = false,
            .memory_addr = null,
            .memory_pre_value = null,
            .memory_value = null,
            .is_memory_write = false,
            .next_pc = self.state.pc + pc_increment,
            .is_compressed = false, // Only LAST step gets is_compressed
            .virtual_sequence_remaining = 2, // 3-step sequence: step 1
            .is_first_in_sequence = true, // First step in SRLIW virtual sequence
        });

        self.state.cycle += 1;
        self.registers.tick();

        // === Step 2: VirtualSRLI(rd, v_rs1, bitmask) ===
        // Total shift is (shamt + 32) since we shifted left by 32 in step 1
        const total_shift: u7 = shamt + 32;
        const ones: u128 = (@as(u128, 1) << @intCast(64 - @as(u8, total_shift))) - 1;
        const bitmask: u64 = @truncate(ones << total_shift);

        // Compute result: logical right shift of step1_result
        const step2_result: u64 = step1_result >> @intCast(total_shift);

        const vsrli_instr = buildVirtualSRLIInstr(rd_u8, v_rs1, total_shift);

        // Read rd pre-value for step 2 (the register we're writing to)
        const rd_pre_value_step2 = try self.registers.read(decoded.rd);

        try self.lookup_trace.recordVirtualSRLI(
            @intCast(self.state.cycle),
            self.state.pc,
            vsrli_instr,
            step1_result, // rs1_value is the virtual register value
            bitmask,
            step2_result,
            true, // is_virtual: part of sequence (vsr=Some(1))
            true, // do_not_update_pc: vsr=1, 1!=0
            false, // is_first_in_sequence: false (step 2)
            self.is_compressed,
        );

        // Write step2 result to rd
        try self.registers.write(decoded.rd, step2_result);

        // Record trace step 2: VirtualSRLI(rd, v_rs1=32, bitmask)
        try self.trace.addStep(.{
            .cycle = self.state.cycle,
            .pc = self.state.pc,
            .unexpanded_pc = self.state.pc,
            .instruction = vsrli_instr,
            .rs1_value = step1_result, // v_rs1 value
            .rs2_value = 0,
            .rd_pre_value = rd_pre_value_step2,
            .rd_value = step2_result,
            .rd_index = rd_u8,
            .rs1_index = v_rs1, // reads from virtual register 32
            .rs2_index = 0,
            .rd_written = true, // Jolt includes rd=0 writes in RdWa polynomial
            .rs1_read = true,
            .rs2_read = false,
            .memory_addr = null,
            .memory_pre_value = null,
            .memory_value = null,
            .is_memory_write = false,
            .next_pc = self.state.pc + pc_increment,
            .is_compressed = false, // Only LAST step gets is_compressed
            .virtual_sequence_remaining = 1, // 3-step sequence: step 2
        });

        self.state.cycle += 1;
        self.registers.tick();

        // NOTE: Do NOT reset virtual register here. Jolt does NOT reset virtual
        // registers (32-38) between instruction sequences. The value persists and
        // affects rd_pre_value/inc_poly in Stage 4 polynomial construction.

        // === Step 3: VirtualSignExtendWord(rd, rd, 0) ===
        const sign_extended: u64 = @bitCast(@as(i64, @as(i32, @truncate(@as(i64, @bitCast(step2_result))))));
        const vsew_instr = buildVirtualSignExtendWordInstr(rd_u8, rd_u8);
        const rd_pre_value_step3 = step2_result;

        try self.lookup_trace.recordVirtualSignExtendWord(
            @intCast(self.state.cycle),
            self.state.pc,
            vsew_instr,
            step2_result,
            sign_extended,
        );

        try self.registers.write(decoded.rd, sign_extended);

        // Record trace step 3: VirtualSignExtendWord(rd, rd, 0)
        try self.trace.addStep(.{
            .cycle = self.state.cycle,
            .pc = self.state.pc,
            .unexpanded_pc = self.state.pc,
            .instruction = vsew_instr,
            .rs1_value = step2_result,
            .rs2_value = 0,
            .rd_pre_value = rd_pre_value_step3,
            .rd_value = sign_extended,
            .rd_index = rd_u8,
            .rs1_index = rd_u8, // VirtualSignExtendWord reads rd
            .rs2_index = 0,
            .rd_written = true, // Jolt includes rd=0 writes in RdWa polynomial
            .rs1_read = true,
            .rs2_read = false,
            .memory_addr = null,
            .memory_pre_value = null,
            .memory_value = null,
            .is_memory_write = false,
            .next_pc = self.state.pc + pc_increment,
            .is_compressed = self.is_compressed,
            .virtual_sequence_remaining = 0, // Last in 3-step sequence
            .is_last_in_sequence = true,
        });

        self.prev_pc = self.state.pc;
        self.state.pc = self.state.pc + pc_increment;
        self.state.cycle += 1;
        self.registers.tick();
        return true;
    }

    /// Execute a jolt-inline instruction (opcode 0x0B) by expanding it into a
    /// virtual instruction sequence. Currently supports SHA256 (funct7=0, funct3=0/1).
    fn stepInline(self: *Emulator, instruction: u32) !bool {
        const sha256_inline = @import("sha256_inline.zig");

        const funct3: u3 = @truncate((instruction >> 12) & 0x7);
        const funct7: u7 = @truncate((instruction >> 25) & 0x7f);
        const rs1_reg: u8 = @truncate((instruction >> 15) & 0x1f);
        const rs2_reg: u8 = @truncate((instruction >> 20) & 0x1f);
        const rs3_reg: u8 = @truncate((instruction >> 7) & 0x1f); // rd field = rs3 in FormatInline

        // Determine inline type
        const initial = (funct7 == 0x00 and funct3 == 0x01); // SHA256INIT
        const is_sha256 = (funct7 == 0x00 and (funct3 == 0x00 or funct3 == 0x01));
        if (!is_sha256) {
            // Unsupported inline - treat as NOP (fallback)
            self.prev_pc = self.state.pc;
            self.state.pc += 4;
            self.state.cycle += 1;
            self.registers.tick();
            return true;
        }

        // If rs3 (rd) is x0, remap to virtual register 40 (matching Jolt)
        const effective_rs3: u8 = if (rs3_reg == 0) 40 else rs3_reg;
        _ = effective_rs3; // rs3 not directly used in SHA256 inline (state is at memory[rs1])

        // Read memory pointers from registers
        const state_ptr = try self.registers.read(rs1_reg);
        const input_ptr = try self.registers.read(rs2_reg);
        _ = state_ptr;
        _ = input_ptr;

        // Build the SHA256 virtual instruction sequence
        var sequence = try sha256_inline.buildSha256Sequence(
            self.allocator,
            rs1_reg,
            rs2_reg,
            initial,
        );
        defer sequence.deinit(self.allocator);

        const seq_len = sequence.items.len;
        const base_pc = self.state.pc;
        const pc_increment: u64 = if (self.is_compressed) 2 else 4;

        // Execute each virtual instruction and emit a trace step
        for (sequence.items, 0..) |instr, idx| {
            const vsr: u16 = @intCast(seq_len - 1 - idx);
            const is_first = (idx == 0);
            const is_last = (idx == seq_len - 1);

            // Read operand values
            const rs1_val = try self.registers.read(instr.rs1);
            const rs2_val = if (instr.rs2 != 0 or instr.kind == .ANDN or instr.kind == .AND or
                instr.kind == .XOR or instr.kind == .OR or instr.kind == .SD)
                try self.registers.read(instr.rs2)
            else
                0;

            // Compute result based on instruction kind
            const rd_val: u64 = switch (instr.kind) {
                .ADD => rs1_val +% rs2_val,
                .ADDI => rs1_val +% instr.imm,
                .XOR => rs1_val ^ rs2_val,
                .XORI => rs1_val ^ instr.imm,
                .AND => rs1_val & rs2_val,
                .ANDI => rs1_val & instr.imm,
                .OR => rs1_val | rs2_val,
                .ANDN => rs1_val & ~rs2_val,
                .VirtualMULI => blk: {
                    // VirtualMULI: rd = rs1 * (1 << imm) — imm is the shift amount
                    const shamt: u6 = @intCast(instr.imm & 0x3F);
                    break :blk rs1_val *% (@as(u64, 1) << shamt);
                },
                .VirtualSRLI => blk: {
                    // VirtualSRLI: rd = rs1 & bitmask — imm IS the bitmask
                    // The result is rs1 logical-right-shifted by trailing_zeros(bitmask)
                    const bitmask = instr.imm;
                    const shift: u6 = if (bitmask == 0) 0 else @intCast(@ctz(bitmask));
                    break :blk rs1_val >> shift;
                },
                .VirtualSignExtendWord => blk: {
                    // Sign-extend lower 32 bits to 64 bits
                    const lower32: u32 = @truncate(rs1_val);
                    break :blk @bitCast(@as(i64, @as(i32, @bitCast(lower32))));
                },
                .VirtualZeroExtendWord => rs1_val & 0xFFFFFFFF,
                .VirtualROTRIW => blk: {
                    const val32: u32 = @truncate(rs1_val);
                    // instr.imm is the BITMASK, extract rotation via trailing zeros
                    const bm32: u32 = @truncate(instr.imm);
                    const rotation: u5 = if (bm32 == 0) 0 else @intCast(@ctz(bm32));
                    break :blk @as(u64, std.math.rotr(u32, val32, rotation));
                },
                .LD => blk: {
                    const addr: u64 = @bitCast(@as(i64, @bitCast(rs1_val)) +% @as(i64, @bitCast(instr.imm)));
                    break :blk try self.ram.read(addr, @intCast(self.state.cycle));
                },
                .SD => blk: {
                    // For stores, rs2_val is what gets written
                    const addr: u64 = @bitCast(@as(i64, @bitCast(rs1_val)) +% @as(i64, @bitCast(instr.imm)));
                    try self.ram.write(addr, rs2_val, @intCast(self.state.cycle));
                    break :blk 0;
                },
            };

            // Write result to destination register (except for stores and x0)
            const rd_pre_value = try self.registers.read(instr.rd);
            const is_store = (instr.kind == .SD);
            if (!is_store and instr.rd != 0) {
                try self.registers.write(instr.rd, rd_val);
            }

            // Build synthetic instruction word for the trace
            const synth_word: u32 = switch (instr.kind) {
                .ADD => blk: {
                    break :blk (@as(u32, 0) << 25) | (@as(u32, instr.rs2 & 0x1F) << 20) | (@as(u32, instr.rs1 & 0x1F) << 15) | (0 << 12) | (@as(u32, instr.rd & 0x1F) << 7) | 0x33;
                },
                .ADDI => blk: {
                    const imm12: u32 = @truncate(instr.imm & 0xFFF);
                    break :blk (imm12 << 20) | (@as(u32, instr.rs1 & 0x1F) << 15) | (0 << 12) | (@as(u32, instr.rd & 0x1F) << 7) | 0x13;
                },
                .XOR => blk: {
                    break :blk (@as(u32, 0) << 25) | (@as(u32, instr.rs2 & 0x1F) << 20) | (@as(u32, instr.rs1 & 0x1F) << 15) | (4 << 12) | (@as(u32, instr.rd & 0x1F) << 7) | 0x33;
                },
                .XORI => blk: {
                    const imm12: u32 = @truncate(instr.imm & 0xFFF);
                    break :blk (imm12 << 20) | (@as(u32, instr.rs1 & 0x1F) << 15) | (4 << 12) | (@as(u32, instr.rd & 0x1F) << 7) | 0x13;
                },
                .AND => blk: {
                    break :blk (@as(u32, 0) << 25) | (@as(u32, instr.rs2 & 0x1F) << 20) | (@as(u32, instr.rs1 & 0x1F) << 15) | (7 << 12) | (@as(u32, instr.rd & 0x1F) << 7) | 0x33;
                },
                .ANDI => blk: {
                    const imm12: u32 = @truncate(instr.imm & 0xFFF);
                    break :blk (imm12 << 20) | (@as(u32, instr.rs1 & 0x1F) << 15) | (7 << 12) | (@as(u32, instr.rd & 0x1F) << 7) | 0x13;
                },
                .OR => blk: {
                    break :blk (@as(u32, 0) << 25) | (@as(u32, instr.rs2 & 0x1F) << 20) | (@as(u32, instr.rs1 & 0x1F) << 15) | (6 << 12) | (@as(u32, instr.rd & 0x1F) << 7) | 0x33;
                },
                .ANDN => buildANDNInstr(instr.rd, instr.rs1, instr.rs2),
                .VirtualMULI => buildVirtualMULIInstr(instr.rd, instr.rs1, @intCast(instr.imm & 0x3F)),
                .VirtualSRLI => blk: {
                    // VirtualSRLI uses opcode 0x5B, stores total_shift in imm field
                    const bitmask = instr.imm;
                    const total_shift: u7 = if (bitmask == 0) 0 else @intCast(@ctz(bitmask));
                    break :blk buildVirtualSRLIInstr(instr.rd, instr.rs1, total_shift);
                },
                .VirtualSignExtendWord => buildVirtualSignExtendWordInstr(instr.rd, instr.rs1),
                .VirtualZeroExtendWord => buildVirtualZeroExtendWordInstr(instr.rd, instr.rs1),
                .VirtualROTRIW => blk_rotriw: {
                    // instr.imm is the BITMASK, extract rotation amount via trailing zeros
                    const bm32: u32 = @truncate(instr.imm);
                    const rot: u5 = if (bm32 == 0) 0 else @intCast(@ctz(bm32));
                    break :blk_rotriw buildVirtualROTRIWInstr(instr.rd, instr.rs1, rot);
                },
                .LD => blk: {
                    const imm12: u32 = @truncate(@as(u64, @bitCast(@as(i64, @bitCast(instr.imm)))) & 0xFFF);
                    break :blk (imm12 << 20) | (@as(u32, instr.rs1 & 0x1F) << 15) | (3 << 12) | (@as(u32, instr.rd & 0x1F) << 7) | 0x03;
                },
                .SD => blk: {
                    const imm_val: i64 = @bitCast(instr.imm);
                    const imm_u: u32 = @truncate(@as(u64, @bitCast(imm_val)) & 0xFFF);
                    break :blk ((imm_u >> 5) << 25) | (@as(u32, instr.rs2 & 0x1F) << 20) | (@as(u32, instr.rs1 & 0x1F) << 15) | (3 << 12) | ((imm_u & 0x1F) << 7) | 0x23;
                },
            };

            // Record lookup trace entry (for instructions that have lookup tables)
            const do_not_update_pc = (vsr != 0);
            switch (instr.kind) {
                .ADD, .ADDI => {
                    // RangeCheck table (identity)
                    try self.lookup_trace.recordInstruction(
                        @intCast(self.state.cycle),
                        base_pc,
                        synth_word,
                        zkvm.instruction.DecodedInstruction.decode(synth_word),
                        rs1_val,
                        rs2_val,
                    );
                },
                .XOR, .XORI, .AND, .ANDI, .OR => {
                    try self.lookup_trace.recordInstruction(
                        @intCast(self.state.cycle),
                        base_pc,
                        synth_word,
                        zkvm.instruction.DecodedInstruction.decode(synth_word),
                        rs1_val,
                        rs2_val,
                    );
                },
                .ANDN => {
                    try self.lookup_trace.recordANDN(
                        @intCast(self.state.cycle),
                        base_pc,
                        synth_word,
                        rs1_val,
                        rs2_val,
                        true, // is_virtual
                        do_not_update_pc,
                        is_first,
                        is_last and self.is_compressed,
                    );
                },
                .VirtualROTRIW => {
                    // Reconstruct the bitmask from rotation amount
                    const rotation = instr.imm & 0x1F;
                    const bitmask: u64 = if (rotation == 0) 0xFFFFFFFF else (((@as(u64, 1) << @intCast(32 - rotation)) - 1) << @intCast(rotation));
                    try self.lookup_trace.recordVirtualROTRIW(
                        @intCast(self.state.cycle),
                        base_pc,
                        synth_word,
                        rs1_val,
                        bitmask,
                        true, // is_virtual
                        do_not_update_pc,
                        is_first,
                        is_last and self.is_compressed,
                    );
                },
                .VirtualZeroExtendWord => {
                    try self.lookup_trace.recordVirtualZeroExtendWord(
                        @intCast(self.state.cycle),
                        base_pc,
                        synth_word,
                        rs1_val,
                        true, // is_virtual
                        do_not_update_pc,
                        is_first,
                        is_last and self.is_compressed,
                    );
                },
                .VirtualMULI => {
                    // imm_val for VirtualMULI is the multiplier (1 << shamt)
                    const muli_shamt: u6 = @intCast(instr.imm & 0x3F);
                    const muli_multiplier: u64 = @as(u64, 1) << muli_shamt;
                    try self.lookup_trace.recordVirtualMULI(
                        @intCast(self.state.cycle),
                        base_pc,
                        synth_word,
                        rs1_val,
                        muli_multiplier,
                        true, // is_virtual
                        do_not_update_pc,
                        is_first,
                        is_last and self.is_compressed,
                    );
                },
                .VirtualSRLI => {
                    const srli_bitmask = instr.imm;
                    try self.lookup_trace.recordVirtualSRLI(
                        @intCast(self.state.cycle),
                        base_pc,
                        synth_word,
                        rs1_val,
                        srli_bitmask,
                        rd_val, // result
                        true, // is_virtual
                        do_not_update_pc,
                        is_first,
                        is_last and self.is_compressed,
                    );
                },
                .VirtualSignExtendWord => {
                    try self.lookup_trace.recordVirtualSignExtendWord(
                        @intCast(self.state.cycle),
                        base_pc,
                        synth_word,
                        rs1_val,
                        rd_val, // sign_extended_result
                    );
                },
                .LD, .SD => {
                    // Load/store instructions don't use lookup tables
                },
            }

            // Determine memory access for trace step
            var memory_addr: ?u64 = null;
            var memory_value: ?u64 = null;
            var memory_pre_value: ?u64 = null;
            var is_memory_write: bool = false;
            if (instr.kind == .LD) {
                memory_addr = @bitCast(@as(i64, @bitCast(rs1_val)) +% @as(i64, @bitCast(instr.imm)));
                memory_value = rd_val;
            } else if (instr.kind == .SD) {
                memory_addr = @bitCast(@as(i64, @bitCast(rs1_val)) +% @as(i64, @bitCast(instr.imm)));
                memory_value = rs2_val;
                memory_pre_value = if (self.ram.memory.get(memory_addr.?)) |v| v else 0;
                is_memory_write = true;
            }

            // Emit trace step
            try self.trace.steps.append(self.allocator, .{
                .cycle = self.state.cycle,
                .pc = base_pc, // raw PC (same as other step functions)
                .unexpanded_pc = base_pc,
                .instruction = synth_word,
                .rs1_value = rs1_val,
                .rs2_value = if (instr.kind == .SD) rs2_val else if (instr.kind == .ANDN or instr.kind == .AND or instr.kind == .XOR or instr.kind == .OR or instr.kind == .ADD) rs2_val else 0,
                .rd_pre_value = rd_pre_value,
                .rd_value = if (is_store) 0 else rd_val,
                .rd_index = instr.rd,
                .rs1_index = instr.rs1,
                .rs2_index = instr.rs2,
                .rd_written = !is_store and instr.rd != 0,
                .rs1_read = true,
                .rs2_read = (instr.kind == .ANDN or instr.kind == .AND or instr.kind == .XOR or instr.kind == .OR or instr.kind == .ADD or instr.kind == .SD),
                .memory_addr = memory_addr,
                .memory_pre_value = memory_pre_value,
                .memory_value = memory_value,
                .is_memory_write = is_memory_write,
                .next_pc = if (is_last) base_pc + pc_increment else base_pc,
                .is_compressed = if (is_last) self.is_compressed else false,
                .virtual_sequence_remaining = vsr,
                .is_first_in_sequence = is_first,
                .is_last_in_sequence = is_last,
            });

            self.state.cycle += 1;
            self.registers.tick();
        }

        // Advance PC
        self.prev_pc = self.state.pc;
        self.state.pc = base_pc + pc_increment;

        return true;
    }

    /// Execute a W-extension instruction as 2 decomposed trace steps:
    /// Step 1: Base instruction (ADDI/ADD/SUB/MUL/VirtualMULI) with full 64-bit result
    /// Step 2: VirtualSignExtendWord - sign-extend lower 32 bits to 64 bits
    fn stepWExtension(
        self: *Emulator,
        instruction: u32,
        decoded: zkvm.instruction.DecodedInstruction,
        w_type: WExtType,
    ) !bool {
        const rs1_value = try self.registers.read(decoded.rs1);
        const rs2_value = try self.registers.read(decoded.rs2);
        const rd_pre_value = try self.registers.read(decoded.rd);
        const pc_increment: u64 = if (self.is_compressed) 2 else 4;

        // --- Step 1: Base instruction with full 64-bit semantics ---
        // Compute the FULL 64-bit result (not truncated to 32 bits)
        const base_result: u64 = switch (w_type) {
            .ADDIW => rs1_value +% @as(u64, @bitCast(@as(i64, decoded.imm))),
            .ADDW => rs1_value +% rs2_value,
            .SUBW => rs1_value -% rs2_value,
            .MULW => rs1_value *% rs2_value,
            .SLLIW => blk: {
                const imm_u32: u32 = @bitCast(@as(i32, @truncate(decoded.imm)));
                const shamt: u6 = @intCast(imm_u32 & 0x3F);
                const multiplier: u64 = @as(u64, 1) << shamt;
                break :blk rs1_value *% multiplier;
            },
            .SRLIW => unreachable, // Handled separately in stepSRLIW
        };

        // Build the synthetic base instruction (ADDI/ADD/SUB/MUL/VirtualMULI encoding)
        const base_instr = buildBaseInstruction(instruction, w_type);

        if (w_type == .SLLIW) {
            // SLLIW base step: VirtualMULI (uses recordVirtualMULI, not recordInstruction)
            const imm_u32: u32 = @bitCast(@as(i32, @truncate(decoded.imm)));
            const shamt: u6 = @intCast(imm_u32 & 0x3F);
            const multiplier: u64 = @as(u64, 1) << shamt;

            try self.lookup_trace.recordVirtualMULI(
                @intCast(self.state.cycle),
                self.state.pc,
                base_instr,
                rs1_value,
                multiplier,
                true, // is_virtual: part of SLLIW virtual sequence (vsr=Some(1))
                true, // do_not_update_pc: vsr=1, 1!=0 so true
                true, // is_first_in_sequence: first in SLLIW decomposition
                self.is_compressed, // is_compressed
            );
        } else {
            const base_decoded = zkvm.instruction.DecodedInstruction.decode(base_instr);

            // Determine the right operand value for the base instruction
            const base_rs2_value: u64 = switch (w_type) {
                .ADDIW => @as(u64, @bitCast(@as(i64, decoded.imm))), // imm as rs2 for ADDI
                .ADDW, .SUBW, .MULW => rs2_value,
                .SLLIW, .SRLIW => unreachable, // Handled above/separately
            };

            // Record lookup trace for base instruction
            try self.lookup_trace.recordInstruction(
                @intCast(self.state.cycle),
                self.state.pc,
                base_instr,
                base_decoded,
                rs1_value,
                base_rs2_value,
            );
        }

        // Write base result to register
        try self.registers.write(decoded.rd, base_result);

        // Determine rs2 read for base instruction
        const base_reads_rs2 = switch (w_type) {
            .ADDW, .SUBW, .MULW => true,
            .ADDIW, .SLLIW, .SRLIW => false,
        };

        // Record trace step 1 (base instruction)
        try self.trace.addStep(.{
            .cycle = self.state.cycle,
            .pc = self.state.pc,
            .unexpanded_pc = self.state.pc,
            .instruction = base_instr,
            .rs1_value = rs1_value,
            .rs2_value = switch (w_type) {
                .ADDIW, .SLLIW, .SRLIW => 0, // I-type: rs2 field is not used (imm is in instruction)
                .ADDW, .SUBW, .MULW => rs2_value,
            },
            .rd_pre_value = rd_pre_value,
            .rd_value = base_result,
            .rd_index = decoded.rd,
            .rs1_index = decoded.rs1,
            .rs2_index = decoded.rs2,
            .rd_written = true, // Jolt includes rd=0 writes in RdWa polynomial
            .rs1_read = true,
            .rs2_read = base_reads_rs2,
            .memory_addr = null,
            .memory_pre_value = null,
            .memory_value = null,
            .is_memory_write = false,
            .next_pc = self.state.pc + pc_increment, // Next PC is the same address for virtual seq
            .is_compressed = false, // Only the LAST step of a virtual sequence gets is_compressed
            .virtual_sequence_remaining = 1, // 2-instruction sequence: base(1), VirtualSignExtendWord(0)
            .is_first_in_sequence = true, // First step in W-extension virtual sequence
        });

        self.state.cycle += 1;
        self.registers.tick();

        // --- Step 2: VirtualSignExtendWord ---
        // Sign-extend lower 32 bits of the base result
        const sign_extended: u64 = @bitCast(@as(i64, @as(i32, @truncate(@as(i64, @bitCast(base_result))))));

        // Build synthetic VirtualSignExtendWord instruction
        const vsew_instr = buildVirtualSignExtendWordInstr(decoded.rd, decoded.rd);

        // The rd pre-value for step 2 is the base_result from step 1
        const rd_pre_value_step2 = base_result;

        // Record lookup trace for VirtualSignExtendWord
        try self.lookup_trace.recordVirtualSignExtendWord(
            @intCast(self.state.cycle),
            self.state.pc,
            vsew_instr,
            base_result,
            sign_extended,
        );

        // Write sign-extended result to register
        try self.registers.write(decoded.rd, sign_extended);

        // Record trace step 2 (VirtualSignExtendWord)
        try self.trace.addStep(.{
            .cycle = self.state.cycle,
            .pc = self.state.pc,
            .unexpanded_pc = self.state.pc,
            .instruction = vsew_instr,
            .rs1_value = base_result, // VirtualSignExtendWord reads rd (which has base_result)
            .rs2_value = 0,
            .rd_pre_value = rd_pre_value_step2,
            .rd_value = sign_extended,
            .rd_index = decoded.rd,
            .rs1_index = decoded.rd, // VirtualSignExtendWord reads from rd
            .rs2_index = 0,
            .rd_written = true, // Jolt includes rd=0 writes in RdWa polynomial
            .rs1_read = true,
            .rs2_read = false,
            .memory_addr = null,
            .memory_pre_value = null,
            .memory_value = null,
            .is_memory_write = false,
            .next_pc = self.state.pc + pc_increment,
            .is_compressed = self.is_compressed,
            .virtual_sequence_remaining = 0, // Last in 2-instruction sequence
            .is_last_in_sequence = true,
        });

        // Update prev_pc for infinite loop detection
        self.prev_pc = self.state.pc;

        // Update state - PC advances by instruction size (NOT doubled)
        self.state.pc = self.state.pc + pc_increment;
        self.state.cycle += 1;
        self.registers.tick();

        return true;
    }

    /// Execute REMUW as a 12-step inline sequence decomposition.
    /// Matches Jolt's REMUW decomposition exactly:
    ///   Step 1:  VirtualAdvice(a2) → quotient
    ///   Step 2:  VirtualAdvice(a3) → remainder
    ///   Step 3:  VirtualZeroExtendWord(t3, a2) → zero-extend quotient
    ///   Step 4:  VirtualZeroExtendWord(t1, rs1) → zero-extend dividend
    ///   Step 5:  VirtualZeroExtendWord(t2, rs2) → zero-extend divisor
    ///   Step 6:  MUL(t0, t3, t2) → quotient * divisor
    ///   Step 7:  VirtualZeroExtendWord(t4, t0) → mask to 32 bits
    ///   Step 8:  VirtualAssertEQ(t4, t0) → assert no overflow
    ///   Step 9:  ADD(t0, t0, a3) → add remainder
    ///   Step 10: VirtualAssertEQ(t0, t1) → assert dividend = q*d + r
    ///   Step 11: VirtualAssertValidUnsignedRemainder(a3, t2) → r < d
    ///   Step 12: VirtualSignExtendWord(rd, a3) → sign-extend result
    fn stepREMUW(
        self: *Emulator,
        instruction: u32,
        decoded: zkvm.instruction.DecodedInstruction,
    ) !bool {
        // Read operands
        const rs1_value = try self.registers.read(decoded.rs1);
        const rs2_value = try self.registers.read(decoded.rs2);
        const rd_pre_value = try self.registers.read(decoded.rd);
        const pc_increment: u64 = if (self.is_compressed) 2 else 4;
        _ = instruction;

        // Virtual register indices (matching Jolt's allocation)
        const a2: u8 = 32;
        const a3: u8 = 33;
        const t0: u8 = 34;
        const t1: u8 = 35;
        const t2: u8 = 36;
        const t3: u8 = 37;
        const t4: u8 = 38;

        // Compute the actual REMUW result (for VirtualAdvice oracle values)
        const rs1_lower: u32 = @truncate(rs1_value);
        const rs2_lower: u32 = @truncate(rs2_value);
        const quotient_u32: u32 = if (rs2_lower == 0) 0 else rs1_lower / rs2_lower;
        const remainder_u32: u32 = if (rs2_lower == 0) rs1_lower else rs1_lower % rs2_lower;
        // Sign-extend to 64 bits for register storage
        const quotient: u64 = @bitCast(@as(i64, @as(i32, @bitCast(quotient_u32))));
        const remainder: u64 = @bitCast(@as(i64, @as(i32, @bitCast(remainder_u32))));

        // Initialize virtual register pre-values (zero before first use)
        const a2_pre: u64 = try self.registers.read(a2);
        const a3_pre: u64 = try self.registers.read(a3);
        const t0_pre: u64 = try self.registers.read(t0);
        const t1_pre: u64 = try self.registers.read(t1);
        const t2_pre: u64 = try self.registers.read(t2);
        const t3_pre: u64 = try self.registers.read(t3);
        const t4_pre: u64 = try self.registers.read(t4);

        // ==================== Step 1: VirtualAdvice(a2) → quotient ====================
        const step1_instr = buildVirtualAdviceInstr(a2);
        try self.lookup_trace.recordVirtualAdvice(
            @intCast(self.state.cycle),
            self.state.pc,
            step1_instr,
            quotient,
            true,
            true,
            true,
            self.is_compressed, // is_virtual, do_not_update_pc, is_first_in_sequence
        );
        try self.registers.write(a2, quotient);
        try self.trace.addStep(.{
            .cycle = self.state.cycle,
            .pc = self.state.pc,
            .unexpanded_pc = self.state.pc,
            .instruction = step1_instr,
            .rs1_value = 0,
            .rs2_value = 0,
            .rd_pre_value = a2_pre,
            .rd_value = quotient,
            .rd_index = a2,
            .rs1_index = 0,
            .rs2_index = 0,
            .rd_written = true,
            .rs1_read = false,
            .rs2_read = false,
            .memory_addr = null,
            .memory_pre_value = null,
            .memory_value = null,
            .is_memory_write = false,
            .next_pc = self.state.pc + pc_increment,
            .is_compressed = false,
            .virtual_sequence_remaining = 11,
            .is_first_in_sequence = true, // First step in REMUW 12-step virtual sequence
        });
        self.state.cycle += 1;
        self.registers.tick();

        // ==================== Step 2: VirtualAdvice(a3) → remainder ====================
        const step2_instr = buildVirtualAdviceInstr(a3);
        try self.lookup_trace.recordVirtualAdvice(
            @intCast(self.state.cycle),
            self.state.pc,
            step2_instr,
            remainder,
            true,
            true,
            false,
            self.is_compressed,
        );
        try self.registers.write(a3, remainder);
        try self.trace.addStep(.{
            .cycle = self.state.cycle,
            .pc = self.state.pc,
            .unexpanded_pc = self.state.pc,
            .instruction = step2_instr,
            .rs1_value = 0,
            .rs2_value = 0,
            .rd_pre_value = a3_pre,
            .rd_value = remainder,
            .rd_index = a3,
            .rs1_index = 0,
            .rs2_index = 0,
            .rd_written = true,
            .rs1_read = false,
            .rs2_read = false,
            .memory_addr = null,
            .memory_pre_value = null,
            .memory_value = null,
            .is_memory_write = false,
            .next_pc = self.state.pc + pc_increment,
            .is_compressed = false,
            .virtual_sequence_remaining = 10,
        });
        self.state.cycle += 1;
        self.registers.tick();

        // ==================== Step 3: VirtualZeroExtendWord(t3, a2) → zero-extend quotient ====================
        const a2_val = try self.registers.read(a2);
        const t3_zero_ext: u64 = a2_val & 0xFFFFFFFF;
        const step3_instr = buildVirtualZeroExtendWordInstr(t3, a2);
        try self.lookup_trace.recordVirtualZeroExtendWord(
            @intCast(self.state.cycle),
            self.state.pc,
            step3_instr,
            a2_val,
            true,
            true,
            false,
            self.is_compressed,
        );
        try self.registers.write(t3, t3_zero_ext);
        try self.trace.addStep(.{
            .cycle = self.state.cycle,
            .pc = self.state.pc,
            .unexpanded_pc = self.state.pc,
            .instruction = step3_instr,
            .rs1_value = a2_val,
            .rs2_value = 0,
            .rd_pre_value = t3_pre,
            .rd_value = t3_zero_ext,
            .rd_index = t3,
            .rs1_index = a2,
            .rs2_index = 0,
            .rd_written = true,
            .rs1_read = true,
            .rs2_read = false,
            .memory_addr = null,
            .memory_pre_value = null,
            .memory_value = null,
            .is_memory_write = false,
            .next_pc = self.state.pc + pc_increment,
            .is_compressed = false,
            .virtual_sequence_remaining = 9,
        });
        self.state.cycle += 1;
        self.registers.tick();

        // ==================== Step 4: VirtualZeroExtendWord(t1, rs1) → zero-extend dividend ====================
        const t1_zero_ext: u64 = rs1_value & 0xFFFFFFFF;
        const step4_instr = buildVirtualZeroExtendWordInstr(t1, decoded.rs1);
        try self.lookup_trace.recordVirtualZeroExtendWord(
            @intCast(self.state.cycle),
            self.state.pc,
            step4_instr,
            rs1_value,
            true,
            true,
            false,
            self.is_compressed,
        );
        try self.registers.write(t1, t1_zero_ext);
        try self.trace.addStep(.{
            .cycle = self.state.cycle,
            .pc = self.state.pc,
            .unexpanded_pc = self.state.pc,
            .instruction = step4_instr,
            .rs1_value = rs1_value,
            .rs2_value = 0,
            .rd_pre_value = t1_pre,
            .rd_value = t1_zero_ext,
            .rd_index = t1,
            .rs1_index = decoded.rs1,
            .rs2_index = 0,
            .rd_written = true,
            .rs1_read = true,
            .rs2_read = false,
            .memory_addr = null,
            .memory_pre_value = null,
            .memory_value = null,
            .is_memory_write = false,
            .next_pc = self.state.pc + pc_increment,
            .is_compressed = false,
            .virtual_sequence_remaining = 8,
        });
        self.state.cycle += 1;
        self.registers.tick();

        // ==================== Step 5: VirtualZeroExtendWord(t2, rs2) → zero-extend divisor ====================
        const t2_zero_ext: u64 = rs2_value & 0xFFFFFFFF;
        const step5_instr = buildVirtualZeroExtendWordInstr(t2, decoded.rs2);
        try self.lookup_trace.recordVirtualZeroExtendWord(
            @intCast(self.state.cycle),
            self.state.pc,
            step5_instr,
            rs2_value,
            true,
            true,
            false,
            self.is_compressed,
        );
        try self.registers.write(t2, t2_zero_ext);
        try self.trace.addStep(.{
            .cycle = self.state.cycle,
            .pc = self.state.pc,
            .unexpanded_pc = self.state.pc,
            .instruction = step5_instr,
            .rs1_value = rs2_value,
            .rs2_value = 0,
            .rd_pre_value = t2_pre,
            .rd_value = t2_zero_ext,
            .rd_index = t2,
            .rs1_index = decoded.rs2,
            .rs2_index = 0,
            .rd_written = true,
            .rs1_read = true,
            .rs2_read = false,
            .memory_addr = null,
            .memory_pre_value = null,
            .memory_value = null,
            .is_memory_write = false,
            .next_pc = self.state.pc + pc_increment,
            .is_compressed = false,
            .virtual_sequence_remaining = 7,
        });
        self.state.cycle += 1;
        self.registers.tick();

        // ==================== Step 6: MUL(t0, t3, t2) → quotient * divisor ====================
        const t3_val = try self.registers.read(t3);
        const t2_val = try self.registers.read(t2);
        const mul_result: u64 = t3_val *% t2_val;
        const step6_instr = buildMULInstr(t0, t3, t2);
        try self.lookup_trace.recordMulVirtual(
            @intCast(self.state.cycle),
            self.state.pc,
            step6_instr,
            t3_val,
            t2_val,
            true,
            false,
            self.is_compressed, // do_not_update_pc, is_first_in_sequence
        );
        try self.registers.write(t0, mul_result);
        try self.trace.addStep(.{
            .cycle = self.state.cycle,
            .pc = self.state.pc,
            .unexpanded_pc = self.state.pc,
            .instruction = step6_instr,
            .rs1_value = t3_val,
            .rs2_value = t2_val,
            .rd_pre_value = t0_pre,
            .rd_value = mul_result,
            .rd_index = t0,
            .rs1_index = t3,
            .rs2_index = t2,
            .rd_written = true,
            .rs1_read = true,
            .rs2_read = true,
            .memory_addr = null,
            .memory_pre_value = null,
            .memory_value = null,
            .is_memory_write = false,
            .next_pc = self.state.pc + pc_increment,
            .is_compressed = false,
            .virtual_sequence_remaining = 6,
        });
        self.state.cycle += 1;
        self.registers.tick();

        // ==================== Step 7: VirtualZeroExtendWord(t4, t0) → mask to 32 bits ====================
        const t0_val = try self.registers.read(t0);
        const t4_zero_ext: u64 = t0_val & 0xFFFFFFFF;
        const step7_instr = buildVirtualZeroExtendWordInstr(t4, t0);
        try self.lookup_trace.recordVirtualZeroExtendWord(
            @intCast(self.state.cycle),
            self.state.pc,
            step7_instr,
            t0_val,
            true,
            true,
            false,
            self.is_compressed,
        );
        try self.registers.write(t4, t4_zero_ext);
        try self.trace.addStep(.{
            .cycle = self.state.cycle,
            .pc = self.state.pc,
            .unexpanded_pc = self.state.pc,
            .instruction = step7_instr,
            .rs1_value = t0_val,
            .rs2_value = 0,
            .rd_pre_value = t4_pre,
            .rd_value = t4_zero_ext,
            .rd_index = t4,
            .rs1_index = t0,
            .rs2_index = 0,
            .rd_written = true,
            .rs1_read = true,
            .rs2_read = false,
            .memory_addr = null,
            .memory_pre_value = null,
            .memory_value = null,
            .is_memory_write = false,
            .next_pc = self.state.pc + pc_increment,
            .is_compressed = false,
            .virtual_sequence_remaining = 5,
        });
        self.state.cycle += 1;
        self.registers.tick();

        // ==================== Step 8: VirtualAssertEQ(t4, t0) → assert no overflow ====================
        const t4_val = try self.registers.read(t4);
        const t0_val2 = try self.registers.read(t0);
        const step8_instr = buildVirtualAssertEQInstr(t4, t0);
        try self.lookup_trace.recordVirtualAssertEQ(
            @intCast(self.state.cycle),
            self.state.pc,
            step8_instr,
            t4_val,
            t0_val2,
            true,
            true,
            false,
            self.is_compressed,
        );
        // Assert instructions don't write to rd
        try self.trace.addStep(.{
            .cycle = self.state.cycle,
            .pc = self.state.pc,
            .unexpanded_pc = self.state.pc,
            .instruction = step8_instr,
            .rs1_value = t4_val,
            .rs2_value = t0_val2,
            .rd_pre_value = 0,
            .rd_value = 0,
            .rd_index = 0,
            .rs1_index = t4,
            .rs2_index = t0,
            .rd_written = false,
            .rs1_read = true,
            .rs2_read = true,
            .memory_addr = null,
            .memory_pre_value = null,
            .memory_value = null,
            .is_memory_write = false,
            .next_pc = self.state.pc + pc_increment,
            .is_compressed = false,
            .virtual_sequence_remaining = 4,
        });
        self.state.cycle += 1;
        self.registers.tick();

        // ==================== Step 9: ADD(t0, t0, a3) → add remainder ====================
        const t0_val3 = try self.registers.read(t0);
        const a3_val = try self.registers.read(a3);
        const add_result: u64 = t0_val3 +% a3_val;
        const step9_instr = buildADDInstr(t0, t0, a3);
        const t0_pre_step9 = t0_val3; // t0's current value before this step
        try self.lookup_trace.recordAddVirtual(
            @intCast(self.state.cycle),
            self.state.pc,
            step9_instr,
            t0_val3,
            a3_val,
            true,
            false,
            self.is_compressed, // do_not_update_pc, is_first_in_sequence
        );
        try self.registers.write(t0, add_result);
        try self.trace.addStep(.{
            .cycle = self.state.cycle,
            .pc = self.state.pc,
            .unexpanded_pc = self.state.pc,
            .instruction = step9_instr,
            .rs1_value = t0_val3,
            .rs2_value = a3_val,
            .rd_pre_value = t0_pre_step9,
            .rd_value = add_result,
            .rd_index = t0,
            .rs1_index = t0,
            .rs2_index = a3,
            .rd_written = true,
            .rs1_read = true,
            .rs2_read = true,
            .memory_addr = null,
            .memory_pre_value = null,
            .memory_value = null,
            .is_memory_write = false,
            .next_pc = self.state.pc + pc_increment,
            .is_compressed = false,
            .virtual_sequence_remaining = 3,
        });
        self.state.cycle += 1;
        self.registers.tick();

        // ==================== Step 10: VirtualAssertEQ(t0, t1) → assert dividend = q*d + r ====================
        const t0_val4 = try self.registers.read(t0);
        const t1_val = try self.registers.read(t1);
        const step10_instr = buildVirtualAssertEQInstr(t0, t1);
        try self.lookup_trace.recordVirtualAssertEQ(
            @intCast(self.state.cycle),
            self.state.pc,
            step10_instr,
            t0_val4,
            t1_val,
            true,
            true,
            false,
            self.is_compressed,
        );
        try self.trace.addStep(.{
            .cycle = self.state.cycle,
            .pc = self.state.pc,
            .unexpanded_pc = self.state.pc,
            .instruction = step10_instr,
            .rs1_value = t0_val4,
            .rs2_value = t1_val,
            .rd_pre_value = 0,
            .rd_value = 0,
            .rd_index = 0,
            .rs1_index = t0,
            .rs2_index = t1,
            .rd_written = false,
            .rs1_read = true,
            .rs2_read = true,
            .memory_addr = null,
            .memory_pre_value = null,
            .memory_value = null,
            .is_memory_write = false,
            .next_pc = self.state.pc + pc_increment,
            .is_compressed = false,
            .virtual_sequence_remaining = 2,
        });
        self.state.cycle += 1;
        self.registers.tick();

        // ==================== Step 11: VirtualAssertValidUnsignedRemainder(a3, t2) ====================
        const a3_val2 = try self.registers.read(a3);
        const t2_val2 = try self.registers.read(t2);
        const step11_instr = buildVirtualAssertValidUnsignedRemainderInstr(a3, t2);
        try self.lookup_trace.recordVirtualAssertValidUnsignedRemainder(
            @intCast(self.state.cycle),
            self.state.pc,
            step11_instr,
            a3_val2,
            t2_val2,
            true,
            true,
            false,
            self.is_compressed,
        );
        try self.trace.addStep(.{
            .cycle = self.state.cycle,
            .pc = self.state.pc,
            .unexpanded_pc = self.state.pc,
            .instruction = step11_instr,
            .rs1_value = a3_val2,
            .rs2_value = t2_val2,
            .rd_pre_value = 0,
            .rd_value = 0,
            .rd_index = 0,
            .rs1_index = a3,
            .rs2_index = t2,
            .rd_written = false,
            .rs1_read = true,
            .rs2_read = true,
            .memory_addr = null,
            .memory_pre_value = null,
            .memory_value = null,
            .is_memory_write = false,
            .next_pc = self.state.pc + pc_increment,
            .is_compressed = false,
            .virtual_sequence_remaining = 1,
        });
        self.state.cycle += 1;
        self.registers.tick();

        // ==================== Step 12: VirtualSignExtendWord(rd, a3) → sign-extend result ====================
        const a3_val3 = try self.registers.read(a3);
        const final_result: u64 = @bitCast(@as(i64, @as(i32, @truncate(@as(i64, @bitCast(a3_val3))))));
        const step12_instr = buildVirtualSignExtendWordInstr(decoded.rd, a3);
        try self.lookup_trace.recordVirtualSignExtendWord(
            @intCast(self.state.cycle),
            self.state.pc,
            step12_instr,
            a3_val3,
            final_result,
        );
        try self.registers.write(decoded.rd, final_result);
        try self.trace.addStep(.{
            .cycle = self.state.cycle,
            .pc = self.state.pc,
            .unexpanded_pc = self.state.pc,
            .instruction = step12_instr,
            .rs1_value = a3_val3,
            .rs2_value = 0,
            .rd_pre_value = rd_pre_value,
            .rd_value = final_result,
            .rd_index = decoded.rd,
            .rs1_index = a3,
            .rs2_index = 0,
            .rd_written = true, // Jolt includes rd=0 writes in RdWa polynomial
            .rs1_read = true,
            .rs2_read = false,
            .memory_addr = null,
            .memory_pre_value = null,
            .memory_value = null,
            .is_memory_write = false,
            .next_pc = self.state.pc + pc_increment,
            .is_compressed = self.is_compressed,
            .virtual_sequence_remaining = 0,
            .is_last_in_sequence = true,
        });

        // Update prev_pc for infinite loop detection
        self.prev_pc = self.state.pc;

        // Update state - PC advances by instruction size (NOT doubled)
        self.state.pc = self.state.pc + pc_increment;
        self.state.cycle += 1;
        self.registers.tick();

        return true;
    }

    /// Execute REMW or DIVW as a 21-step inline sequence (matching Jolt's decomposition).
    /// REMW: signed word remainder, DIVW: signed word division.
    /// Both use the same 21-step verification sequence; only the final output register differs.
    fn stepREMWDIVW(
        self: *Emulator,
        instruction: u32,
        decoded: zkvm.instruction.DecodedInstruction,
    ) !bool {
        const rs1_value = try self.registers.read(decoded.rs1);
        const rs2_value = try self.registers.read(decoded.rs2);
        const rd_pre_value = try self.registers.read(decoded.rd);
        const pc_increment: u64 = if (self.is_compressed) 2 else 4;
        _ = instruction;

        const is_remw = decoded.funct3 == 0b110;

        // Virtual register indices
        const a2: u8 = 32; // quotient
        const a3: u8 = 33; // |remainder|
        const t0: u8 = 34; // adjusted divisor
        const t1: u8 = 35; // temporary
        const t2: u8 = 36; // sign mask temporary
        const t3: u8 = 37; // signed remainder
        const t4: u8 = 38; // sign-extended dividend

        // Compute the actual REMW/DIVW results (oracle values)
        const rs1_lower: u32 = @truncate(rs1_value);
        const rs2_lower: u32 = @truncate(rs2_value);
        const dividend_i32: i32 = @bitCast(rs1_lower);
        const divisor_i32: i32 = @bitCast(rs2_lower);

        var quotient_u64: u64 = undefined;
        var abs_remainder_u64: u64 = undefined;

        if (divisor_i32 == 0) {
            // Division by zero: quotient = -1 (all 1s), remainder = dividend
            quotient_u64 = @bitCast(@as(i64, -1));
            abs_remainder_u64 = @bitCast(@as(i64, @as(i32, @bitCast(rs1_lower))));
            // Actually |remainder| is abs(dividend) for the oracle
            // But wait - in div-by-zero, the remainder IS the dividend per RISC-V spec
            // And the VirtualAdvice provides |remainder|, which is abs(dividend)
            const abs_div: u64 = if (dividend_i32 < 0)
                @bitCast(@as(i64, -@as(i64, dividend_i32)))
            else
                @bitCast(@as(i64, dividend_i32));
            abs_remainder_u64 = abs_div;
        } else if (dividend_i32 == std.math.minInt(i32) and divisor_i32 == -1) {
            // Overflow case: quotient = INT32_MIN, remainder = 0
            quotient_u64 = @bitCast(@as(i64, @as(i32, std.math.minInt(i32))));
            abs_remainder_u64 = 0;
        } else {
            const q_i32 = @divTrunc(dividend_i32, divisor_i32);
            const r_i32 = @rem(dividend_i32, divisor_i32);
            quotient_u64 = @bitCast(@as(i64, q_i32));
            abs_remainder_u64 = if (r_i32 < 0)
                @bitCast(@as(i64, -@as(i64, r_i32)))
            else
                @bitCast(@as(i64, r_i32));
        }

        // Read virtual register pre-values
        const a2_pre: u64 = try self.registers.read(a2);
        const a3_pre: u64 = try self.registers.read(a3);
        const t0_pre: u64 = try self.registers.read(t0);
        const t1_pre: u64 = try self.registers.read(t1);
        const t2_pre: u64 = try self.registers.read(t2);
        const t3_pre: u64 = try self.registers.read(t3);
        const t4_pre: u64 = try self.registers.read(t4);

        // ==================== Step 1: VirtualAdvice(a2) → quotient (vsr=20, first) ====================
        const step1_instr = buildVirtualAdviceInstr(a2);
        try self.lookup_trace.recordVirtualAdvice(
            @intCast(self.state.cycle),
            self.state.pc,
            step1_instr,
            quotient_u64,
            true,
            true,
            true,
            self.is_compressed,
        );
        try self.registers.write(a2, quotient_u64);
        try self.trace.addStep(.{
            .cycle = self.state.cycle,
            .pc = self.state.pc,
            .unexpanded_pc = self.state.pc,
            .instruction = step1_instr,
            .rs1_value = 0,
            .rs2_value = 0,
            .rd_pre_value = a2_pre,
            .rd_value = quotient_u64,
            .rd_index = a2,
            .rs1_index = 0,
            .rs2_index = 0,
            .rd_written = true,
            .rs1_read = false,
            .rs2_read = false,
            .memory_addr = null,
            .memory_pre_value = null,
            .memory_value = null,
            .is_memory_write = false,
            .next_pc = self.state.pc + pc_increment,
            .is_compressed = false,
            .virtual_sequence_remaining = 20,
            .is_first_in_sequence = true,
        });
        self.state.cycle += 1;
        self.registers.tick();

        // ==================== Step 2: VirtualAdvice(a3) → |remainder| (vsr=19) ====================
        const step2_instr = buildVirtualAdviceInstr(a3);
        try self.lookup_trace.recordVirtualAdvice(
            @intCast(self.state.cycle),
            self.state.pc,
            step2_instr,
            abs_remainder_u64,
            true,
            true,
            false,
            self.is_compressed,
        );
        try self.registers.write(a3, abs_remainder_u64);
        try self.trace.addStep(.{
            .cycle = self.state.cycle,
            .pc = self.state.pc,
            .unexpanded_pc = self.state.pc,
            .instruction = step2_instr,
            .rs1_value = 0,
            .rs2_value = 0,
            .rd_pre_value = a3_pre,
            .rd_value = abs_remainder_u64,
            .rd_index = a3,
            .rs1_index = 0,
            .rs2_index = 0,
            .rd_written = true,
            .rs1_read = false,
            .rs2_read = false,
            .memory_addr = null,
            .memory_pre_value = null,
            .memory_value = null,
            .is_memory_write = false,
            .next_pc = self.state.pc + pc_increment,
            .is_compressed = false,
            .virtual_sequence_remaining = 19,
        });
        self.state.cycle += 1;
        self.registers.tick();

        // ==================== Step 3: VirtualSignExtendWord(t4, rs1) → sign-ext dividend (vsr=18) ====================
        const se_dividend: u64 = @bitCast(@as(i64, @as(i32, @bitCast(rs1_lower))));
        const step3_instr = buildVirtualSignExtendWordInstr(t4, decoded.rs1);
        try self.lookup_trace.recordVirtualSignExtendWord(
            @intCast(self.state.cycle),
            self.state.pc,
            step3_instr,
            rs1_value,
            se_dividend,
        );
        try self.registers.write(t4, se_dividend);
        try self.trace.addStep(.{
            .cycle = self.state.cycle,
            .pc = self.state.pc,
            .unexpanded_pc = self.state.pc,
            .instruction = step3_instr,
            .rs1_value = rs1_value,
            .rs2_value = 0,
            .rd_pre_value = t4_pre,
            .rd_value = se_dividend,
            .rd_index = t4,
            .rs1_index = decoded.rs1,
            .rs2_index = 0,
            .rd_written = true,
            .rs1_read = true,
            .rs2_read = false,
            .memory_addr = null,
            .memory_pre_value = null,
            .memory_value = null,
            .is_memory_write = false,
            .next_pc = self.state.pc + pc_increment,
            .is_compressed = false,
            .virtual_sequence_remaining = 18,
        });
        self.state.cycle += 1;
        self.registers.tick();

        // ==================== Step 4: VirtualSignExtendWord(t3, rs2) → sign-ext divisor (vsr=17) ====================
        const se_divisor: u64 = @bitCast(@as(i64, divisor_i32));
        const step4_instr = buildVirtualSignExtendWordInstr(t3, decoded.rs2);
        try self.lookup_trace.recordVirtualSignExtendWord(
            @intCast(self.state.cycle),
            self.state.pc,
            step4_instr,
            rs2_value,
            se_divisor,
        );
        try self.registers.write(t3, se_divisor);
        try self.trace.addStep(.{
            .cycle = self.state.cycle,
            .pc = self.state.pc,
            .unexpanded_pc = self.state.pc,
            .instruction = step4_instr,
            .rs1_value = rs2_value,
            .rs2_value = 0,
            .rd_pre_value = t3_pre,
            .rd_value = se_divisor,
            .rd_index = t3,
            .rs1_index = decoded.rs2,
            .rs2_index = 0,
            .rd_written = true,
            .rs1_read = true,
            .rs2_read = false,
            .memory_addr = null,
            .memory_pre_value = null,
            .memory_value = null,
            .is_memory_write = false,
            .next_pc = self.state.pc + pc_increment,
            .is_compressed = false,
            .virtual_sequence_remaining = 17,
        });
        self.state.cycle += 1;
        self.registers.tick();

        // ==================== Step 5: VirtualAssertValidDiv0(t3, a2) → assert div0 handling (vsr=16) ====================
        const t3_val_s5 = try self.registers.read(t3);
        const a2_val_s5 = try self.registers.read(a2);
        const step5_instr = buildVirtualAssertValidDiv0Instr(t3, a2);
        try self.lookup_trace.recordVirtualAssertEQ(
            @intCast(self.state.cycle),
            self.state.pc,
            step5_instr,
            t3_val_s5,
            a2_val_s5,
            true,
            true,
            false,
            self.is_compressed,
        );
        try self.trace.addStep(.{
            .cycle = self.state.cycle,
            .pc = self.state.pc,
            .unexpanded_pc = self.state.pc,
            .instruction = step5_instr,
            .rs1_value = t3_val_s5,
            .rs2_value = a2_val_s5,
            .rd_pre_value = 0,
            .rd_value = 0,
            .rd_index = 0,
            .rs1_index = t3,
            .rs2_index = a2,
            .rd_written = false,
            .rs1_read = true,
            .rs2_read = true,
            .memory_addr = null,
            .memory_pre_value = null,
            .memory_value = null,
            .is_memory_write = false,
            .next_pc = self.state.pc + pc_increment,
            .is_compressed = false,
            .virtual_sequence_remaining = 16,
        });
        self.state.cycle += 1;
        self.registers.tick();

        // ==================== Step 6: VirtualChangeDivisorW(t0, t4, t3) → adjusted divisor (vsr=15) ====================
        const t4_val_s6 = try self.registers.read(t4);
        const t3_val_s6 = try self.registers.read(t3);
        const adjusted_divisor: u64 = blk: {
            const dv_i32: i32 = @truncate(@as(i64, @bitCast(t4_val_s6)));
            const ds_i32: i32 = @truncate(@as(i64, @bitCast(t3_val_s6)));
            if (dv_i32 == std.math.minInt(i32) and ds_i32 == -1) {
                break :blk 1; // Replace divisor with 1 to prevent overflow
            } else {
                break :blk @bitCast(@as(i64, ds_i32)); // Keep original divisor
            }
        };
        const step6_instr = buildVirtualChangeDivisorWInstr(t0, t4, t3);
        try self.lookup_trace.recordVirtualChangeDivisorW(
            @intCast(self.state.cycle),
            self.state.pc,
            step6_instr,
            t4_val_s6,
            t3_val_s6,
            true,
            true,
            false,
            self.is_compressed,
        );
        try self.registers.write(t0, adjusted_divisor);
        try self.trace.addStep(.{
            .cycle = self.state.cycle,
            .pc = self.state.pc,
            .unexpanded_pc = self.state.pc,
            .instruction = step6_instr,
            .rs1_value = t4_val_s6,
            .rs2_value = t3_val_s6,
            .rd_pre_value = t0_pre,
            .rd_value = adjusted_divisor,
            .rd_index = t0,
            .rs1_index = t4,
            .rs2_index = t3,
            .rd_written = true,
            .rs1_read = true,
            .rs2_read = true,
            .memory_addr = null,
            .memory_pre_value = null,
            .memory_value = null,
            .is_memory_write = false,
            .next_pc = self.state.pc + pc_increment,
            .is_compressed = false,
            .virtual_sequence_remaining = 15,
        });
        self.state.cycle += 1;
        self.registers.tick();

        // ==================== Step 7: VirtualSignExtendWord(t1, a2) → sign-ext quotient (vsr=14) ====================
        const a2_val_s7 = try self.registers.read(a2);
        const se_quotient: u64 = @bitCast(@as(i64, @as(i32, @truncate(@as(i64, @bitCast(a2_val_s7))))));
        const step7_instr = buildVirtualSignExtendWordInstr(t1, a2);
        try self.lookup_trace.recordVirtualSignExtendWord(
            @intCast(self.state.cycle),
            self.state.pc,
            step7_instr,
            a2_val_s7,
            se_quotient,
        );
        try self.registers.write(t1, se_quotient);
        try self.trace.addStep(.{
            .cycle = self.state.cycle,
            .pc = self.state.pc,
            .unexpanded_pc = self.state.pc,
            .instruction = step7_instr,
            .rs1_value = a2_val_s7,
            .rs2_value = 0,
            .rd_pre_value = t1_pre,
            .rd_value = se_quotient,
            .rd_index = t1,
            .rs1_index = a2,
            .rs2_index = 0,
            .rd_written = true,
            .rs1_read = true,
            .rs2_read = false,
            .memory_addr = null,
            .memory_pre_value = null,
            .memory_value = null,
            .is_memory_write = false,
            .next_pc = self.state.pc + pc_increment,
            .is_compressed = false,
            .virtual_sequence_remaining = 14,
        });
        self.state.cycle += 1;
        self.registers.tick();

        // ==================== Step 8: VirtualAssertEQ(t1, a2) → quotient fits 32 bits (vsr=13) ====================
        const t1_val_s8 = try self.registers.read(t1);
        const a2_val_s8 = try self.registers.read(a2);
        const step8_instr = buildVirtualAssertEQInstr(t1, a2);
        try self.lookup_trace.recordVirtualAssertEQ(
            @intCast(self.state.cycle),
            self.state.pc,
            step8_instr,
            t1_val_s8,
            a2_val_s8,
            true,
            true,
            false,
            self.is_compressed,
        );
        try self.trace.addStep(.{
            .cycle = self.state.cycle,
            .pc = self.state.pc,
            .unexpanded_pc = self.state.pc,
            .instruction = step8_instr,
            .rs1_value = t1_val_s8,
            .rs2_value = a2_val_s8,
            .rd_pre_value = 0,
            .rd_value = 0,
            .rd_index = 0,
            .rs1_index = t1,
            .rs2_index = a2,
            .rd_written = false,
            .rs1_read = true,
            .rs2_read = true,
            .memory_addr = null,
            .memory_pre_value = null,
            .memory_value = null,
            .is_memory_write = false,
            .next_pc = self.state.pc + pc_increment,
            .is_compressed = false,
            .virtual_sequence_remaining = 13,
        });
        self.state.cycle += 1;
        self.registers.tick();

        // ==================== Step 9: VirtualSRAI(t2, a3, 31) → sign bit of |remainder| (vsr=12) ====================
        const a3_val_s9 = try self.registers.read(a3);
        const srai_result_s9: u64 = @bitCast(@as(i64, @bitCast(a3_val_s9)) >> 31);
        const step9_instr = buildVirtualSRAIInstr(t2, a3, 31);
        try self.lookup_trace.recordVirtualSRAI(
            @intCast(self.state.cycle),
            self.state.pc,
            step9_instr,
            a3_val_s9,
            true,
            true,
            false,
            self.is_compressed,
        );
        try self.registers.write(t2, srai_result_s9);
        try self.trace.addStep(.{
            .cycle = self.state.cycle,
            .pc = self.state.pc,
            .unexpanded_pc = self.state.pc,
            .instruction = step9_instr,
            .rs1_value = a3_val_s9,
            .rs2_value = 0,
            .rd_pre_value = t2_pre,
            .rd_value = srai_result_s9,
            .rd_index = t2,
            .rs1_index = a3,
            .rs2_index = 0,
            .rd_written = true,
            .rs1_read = true,
            .rs2_read = false,
            .memory_addr = null,
            .memory_pre_value = null,
            .memory_value = null,
            .is_memory_write = false,
            .next_pc = self.state.pc + pc_increment,
            .is_compressed = false,
            .virtual_sequence_remaining = 12,
        });
        self.state.cycle += 1;
        self.registers.tick();

        // ==================== Step 10: VirtualAssertEQ(t2, 0) → |remainder| is non-negative (vsr=11) ====================
        const t2_val_s10 = try self.registers.read(t2);
        const step10_instr = buildVirtualAssertEQInstr(t2, 0);
        try self.lookup_trace.recordVirtualAssertEQ(
            @intCast(self.state.cycle),
            self.state.pc,
            step10_instr,
            t2_val_s10,
            0,
            true,
            true,
            false,
            self.is_compressed,
        );
        try self.trace.addStep(.{
            .cycle = self.state.cycle,
            .pc = self.state.pc,
            .unexpanded_pc = self.state.pc,
            .instruction = step10_instr,
            .rs1_value = t2_val_s10,
            .rs2_value = 0,
            .rd_pre_value = 0,
            .rd_value = 0,
            .rd_index = 0,
            .rs1_index = t2,
            .rs2_index = 0,
            .rd_written = false,
            .rs1_read = true,
            .rs2_read = true,
            .memory_addr = null,
            .memory_pre_value = null,
            .memory_value = null,
            .is_memory_write = false,
            .next_pc = self.state.pc + pc_increment,
            .is_compressed = false,
            .virtual_sequence_remaining = 11,
        });
        self.state.cycle += 1;
        self.registers.tick();

        // ==================== Step 11: VirtualSRAI(t2, t4, 31) → sign mask of dividend (vsr=10) ====================
        const t4_val_s11 = try self.registers.read(t4);
        const srai_result_s11: u64 = @bitCast(@as(i64, @bitCast(t4_val_s11)) >> 31);
        const step11_instr = buildVirtualSRAIInstr(t2, t4, 31);
        try self.lookup_trace.recordVirtualSRAI(
            @intCast(self.state.cycle),
            self.state.pc,
            step11_instr,
            t4_val_s11,
            true,
            true,
            false,
            self.is_compressed,
        );
        try self.registers.write(t2, srai_result_s11);
        const t2_pre_s11 = try self.registers.read(t2);
        _ = t2_pre_s11;
        try self.trace.addStep(.{
            .cycle = self.state.cycle,
            .pc = self.state.pc,
            .unexpanded_pc = self.state.pc,
            .instruction = step11_instr,
            .rs1_value = t4_val_s11,
            .rs2_value = 0,
            .rd_pre_value = srai_result_s9,
            .rd_value = srai_result_s11,
            .rd_index = t2,
            .rs1_index = t4,
            .rs2_index = 0,
            .rd_written = true,
            .rs1_read = true,
            .rs2_read = false,
            .memory_addr = null,
            .memory_pre_value = null,
            .memory_value = null,
            .is_memory_write = false,
            .next_pc = self.state.pc + pc_increment,
            .is_compressed = false,
            .virtual_sequence_remaining = 10,
        });
        self.state.cycle += 1;
        self.registers.tick();

        // ==================== Step 12: XOR(t3, a3, t2) → |remainder| XOR sign_mask (vsr=9) ====================
        const a3_val_s12 = try self.registers.read(a3);
        const t2_val_s12 = try self.registers.read(t2);
        const xor_result_s12: u64 = a3_val_s12 ^ t2_val_s12;
        const step12_instr = buildXORInstr(t3, a3, t2);
        const t3_pre_s12 = try self.registers.read(t3);
        try self.lookup_trace.recordXorVirtual(
            @intCast(self.state.cycle),
            self.state.pc,
            step12_instr,
            a3_val_s12,
            t2_val_s12,
            true,
            false,
            self.is_compressed,
        );
        try self.registers.write(t3, xor_result_s12);
        try self.trace.addStep(.{
            .cycle = self.state.cycle,
            .pc = self.state.pc,
            .unexpanded_pc = self.state.pc,
            .instruction = step12_instr,
            .rs1_value = a3_val_s12,
            .rs2_value = t2_val_s12,
            .rd_pre_value = t3_pre_s12,
            .rd_value = xor_result_s12,
            .rd_index = t3,
            .rs1_index = a3,
            .rs2_index = t2,
            .rd_written = true,
            .rs1_read = true,
            .rs2_read = true,
            .memory_addr = null,
            .memory_pre_value = null,
            .memory_value = null,
            .is_memory_write = false,
            .next_pc = self.state.pc + pc_increment,
            .is_compressed = false,
            .virtual_sequence_remaining = 9,
        });
        self.state.cycle += 1;
        self.registers.tick();

        // ==================== Step 13: SUB(t3, t3, t2) → signed remainder (vsr=8) ====================
        const t3_val_s13 = try self.registers.read(t3);
        const t2_val_s13 = try self.registers.read(t2);
        const sub_result_s13: u64 = t3_val_s13 -% t2_val_s13;
        const step13_instr = buildSUBInstr(t3, t3, t2);
        try self.lookup_trace.recordSubVirtual(
            @intCast(self.state.cycle),
            self.state.pc,
            step13_instr,
            t3_val_s13,
            t2_val_s13,
            true,
            false,
            self.is_compressed,
        );
        try self.registers.write(t3, sub_result_s13);
        try self.trace.addStep(.{
            .cycle = self.state.cycle,
            .pc = self.state.pc,
            .unexpanded_pc = self.state.pc,
            .instruction = step13_instr,
            .rs1_value = t3_val_s13,
            .rs2_value = t2_val_s13,
            .rd_pre_value = t3_val_s13,
            .rd_value = sub_result_s13,
            .rd_index = t3,
            .rs1_index = t3,
            .rs2_index = t2,
            .rd_written = true,
            .rs1_read = true,
            .rs2_read = true,
            .memory_addr = null,
            .memory_pre_value = null,
            .memory_value = null,
            .is_memory_write = false,
            .next_pc = self.state.pc + pc_increment,
            .is_compressed = false,
            .virtual_sequence_remaining = 8,
        });
        self.state.cycle += 1;
        self.registers.tick();

        // ==================== Step 14: MUL(t1, a2, t0) → quotient × adjusted_divisor (vsr=7) ====================
        const a2_val_s14 = try self.registers.read(a2);
        const t0_val_s14 = try self.registers.read(t0);
        const mul_result_s14: u64 = a2_val_s14 *% t0_val_s14;
        const step14_instr = buildMULInstr(t1, a2, t0);
        const t1_pre_s14 = try self.registers.read(t1);
        _ = t1_pre_s14;
        try self.lookup_trace.recordMulVirtual(
            @intCast(self.state.cycle),
            self.state.pc,
            step14_instr,
            a2_val_s14,
            t0_val_s14,
            true,
            false,
            self.is_compressed,
        );
        try self.registers.write(t1, mul_result_s14);
        try self.trace.addStep(.{
            .cycle = self.state.cycle,
            .pc = self.state.pc,
            .unexpanded_pc = self.state.pc,
            .instruction = step14_instr,
            .rs1_value = a2_val_s14,
            .rs2_value = t0_val_s14,
            .rd_pre_value = se_quotient,
            .rd_value = mul_result_s14,
            .rd_index = t1,
            .rs1_index = a2,
            .rs2_index = t0,
            .rd_written = true,
            .rs1_read = true,
            .rs2_read = true,
            .memory_addr = null,
            .memory_pre_value = null,
            .memory_value = null,
            .is_memory_write = false,
            .next_pc = self.state.pc + pc_increment,
            .is_compressed = false,
            .virtual_sequence_remaining = 7,
        });
        self.state.cycle += 1;
        self.registers.tick();

        // ==================== Step 15: ADD(t1, t1, t3) → + remainder (vsr=6) ====================
        const t1_val_s15 = try self.registers.read(t1);
        const t3_val_s15 = try self.registers.read(t3);
        const add_result_s15: u64 = t1_val_s15 +% t3_val_s15;
        const step15_instr = buildADDInstr(t1, t1, t3);
        try self.lookup_trace.recordAddVirtual(
            @intCast(self.state.cycle),
            self.state.pc,
            step15_instr,
            t1_val_s15,
            t3_val_s15,
            true,
            false,
            self.is_compressed,
        );
        try self.registers.write(t1, add_result_s15);
        try self.trace.addStep(.{
            .cycle = self.state.cycle,
            .pc = self.state.pc,
            .unexpanded_pc = self.state.pc,
            .instruction = step15_instr,
            .rs1_value = t1_val_s15,
            .rs2_value = t3_val_s15,
            .rd_pre_value = t1_val_s15,
            .rd_value = add_result_s15,
            .rd_index = t1,
            .rs1_index = t1,
            .rs2_index = t3,
            .rd_written = true,
            .rs1_read = true,
            .rs2_read = true,
            .memory_addr = null,
            .memory_pre_value = null,
            .memory_value = null,
            .is_memory_write = false,
            .next_pc = self.state.pc + pc_increment,
            .is_compressed = false,
            .virtual_sequence_remaining = 6,
        });
        self.state.cycle += 1;
        self.registers.tick();

        // ==================== Step 16: VirtualAssertEQ(t1, t4) → dividend = q*d + r (vsr=5) ====================
        const t1_val_s16 = try self.registers.read(t1);
        const t4_val_s16 = try self.registers.read(t4);
        const step16_instr = buildVirtualAssertEQInstr(t1, t4);
        try self.lookup_trace.recordVirtualAssertEQ(
            @intCast(self.state.cycle),
            self.state.pc,
            step16_instr,
            t1_val_s16,
            t4_val_s16,
            true,
            true,
            false,
            self.is_compressed,
        );
        try self.trace.addStep(.{
            .cycle = self.state.cycle,
            .pc = self.state.pc,
            .unexpanded_pc = self.state.pc,
            .instruction = step16_instr,
            .rs1_value = t1_val_s16,
            .rs2_value = t4_val_s16,
            .rd_pre_value = 0,
            .rd_value = 0,
            .rd_index = 0,
            .rs1_index = t1,
            .rs2_index = t4,
            .rd_written = false,
            .rs1_read = true,
            .rs2_read = true,
            .memory_addr = null,
            .memory_pre_value = null,
            .memory_value = null,
            .is_memory_write = false,
            .next_pc = self.state.pc + pc_increment,
            .is_compressed = false,
            .virtual_sequence_remaining = 5,
        });
        self.state.cycle += 1;
        self.registers.tick();

        // ==================== Step 17: VirtualSRAI(t2, t0, 31) → sign mask of adj divisor (vsr=4) ====================
        const t0_val_s17 = try self.registers.read(t0);
        const srai_result_s17: u64 = @bitCast(@as(i64, @bitCast(t0_val_s17)) >> 31);
        const step17_instr = buildVirtualSRAIInstr(t2, t0, 31);
        try self.lookup_trace.recordVirtualSRAI(
            @intCast(self.state.cycle),
            self.state.pc,
            step17_instr,
            t0_val_s17,
            true,
            true,
            false,
            self.is_compressed,
        );
        const t2_pre_s17 = try self.registers.read(t2);
        try self.registers.write(t2, srai_result_s17);
        try self.trace.addStep(.{
            .cycle = self.state.cycle,
            .pc = self.state.pc,
            .unexpanded_pc = self.state.pc,
            .instruction = step17_instr,
            .rs1_value = t0_val_s17,
            .rs2_value = 0,
            .rd_pre_value = t2_pre_s17,
            .rd_value = srai_result_s17,
            .rd_index = t2,
            .rs1_index = t0,
            .rs2_index = 0,
            .rd_written = true,
            .rs1_read = true,
            .rs2_read = false,
            .memory_addr = null,
            .memory_pre_value = null,
            .memory_value = null,
            .is_memory_write = false,
            .next_pc = self.state.pc + pc_increment,
            .is_compressed = false,
            .virtual_sequence_remaining = 4,
        });
        self.state.cycle += 1;
        self.registers.tick();

        // ==================== Step 18: XOR(t1, t0, t2) → adj_divisor XOR sign_mask (vsr=3) ====================
        const t0_val_s18 = try self.registers.read(t0);
        const t2_val_s18 = try self.registers.read(t2);
        const xor_result_s18: u64 = t0_val_s18 ^ t2_val_s18;
        const step18_instr = buildXORInstr(t1, t0, t2);
        const t1_pre_s18 = try self.registers.read(t1);
        try self.lookup_trace.recordXorVirtual(
            @intCast(self.state.cycle),
            self.state.pc,
            step18_instr,
            t0_val_s18,
            t2_val_s18,
            true,
            false,
            self.is_compressed,
        );
        try self.registers.write(t1, xor_result_s18);
        try self.trace.addStep(.{
            .cycle = self.state.cycle,
            .pc = self.state.pc,
            .unexpanded_pc = self.state.pc,
            .instruction = step18_instr,
            .rs1_value = t0_val_s18,
            .rs2_value = t2_val_s18,
            .rd_pre_value = t1_pre_s18,
            .rd_value = xor_result_s18,
            .rd_index = t1,
            .rs1_index = t0,
            .rs2_index = t2,
            .rd_written = true,
            .rs1_read = true,
            .rs2_read = true,
            .memory_addr = null,
            .memory_pre_value = null,
            .memory_value = null,
            .is_memory_write = false,
            .next_pc = self.state.pc + pc_increment,
            .is_compressed = false,
            .virtual_sequence_remaining = 3,
        });
        self.state.cycle += 1;
        self.registers.tick();

        // ==================== Step 19: SUB(t1, t1, t2) → |adj_divisor| (vsr=2) ====================
        const t1_val_s19 = try self.registers.read(t1);
        const t2_val_s19 = try self.registers.read(t2);
        const sub_result_s19: u64 = t1_val_s19 -% t2_val_s19;
        const step19_instr = buildSUBInstr(t1, t1, t2);
        try self.lookup_trace.recordSubVirtual(
            @intCast(self.state.cycle),
            self.state.pc,
            step19_instr,
            t1_val_s19,
            t2_val_s19,
            true,
            false,
            self.is_compressed,
        );
        try self.registers.write(t1, sub_result_s19);
        try self.trace.addStep(.{
            .cycle = self.state.cycle,
            .pc = self.state.pc,
            .unexpanded_pc = self.state.pc,
            .instruction = step19_instr,
            .rs1_value = t1_val_s19,
            .rs2_value = t2_val_s19,
            .rd_pre_value = t1_val_s19,
            .rd_value = sub_result_s19,
            .rd_index = t1,
            .rs1_index = t1,
            .rs2_index = t2,
            .rd_written = true,
            .rs1_read = true,
            .rs2_read = true,
            .memory_addr = null,
            .memory_pre_value = null,
            .memory_value = null,
            .is_memory_write = false,
            .next_pc = self.state.pc + pc_increment,
            .is_compressed = false,
            .virtual_sequence_remaining = 2,
        });
        self.state.cycle += 1;
        self.registers.tick();

        // ==================== Step 20: VirtualAssertValidUnsignedRemainder(a3, t1) → |r| < |d| (vsr=1) ====================
        const a3_val_s20 = try self.registers.read(a3);
        const t1_val_s20 = try self.registers.read(t1);
        const step20_instr = buildVirtualAssertValidUnsignedRemainderInstr(a3, t1);
        try self.lookup_trace.recordVirtualAssertValidUnsignedRemainder(
            @intCast(self.state.cycle),
            self.state.pc,
            step20_instr,
            a3_val_s20,
            t1_val_s20,
            true,
            true,
            false,
            self.is_compressed,
        );
        try self.trace.addStep(.{
            .cycle = self.state.cycle,
            .pc = self.state.pc,
            .unexpanded_pc = self.state.pc,
            .instruction = step20_instr,
            .rs1_value = a3_val_s20,
            .rs2_value = t1_val_s20,
            .rd_pre_value = 0,
            .rd_value = 0,
            .rd_index = 0,
            .rs1_index = a3,
            .rs2_index = t1,
            .rd_written = false,
            .rs1_read = true,
            .rs2_read = true,
            .memory_addr = null,
            .memory_pre_value = null,
            .memory_value = null,
            .is_memory_write = false,
            .next_pc = self.state.pc + pc_increment,
            .is_compressed = false,
            .virtual_sequence_remaining = 1,
        });
        self.state.cycle += 1;
        self.registers.tick();

        // ==================== Step 21: VirtualSignExtendWord(rd, output) → final result (vsr=0) ====================
        // REMW: output = t3 (signed remainder), DIVW: output = a2 (quotient)
        const output_reg: u8 = if (is_remw) t3 else a2;
        const output_val = try self.registers.read(output_reg);
        const final_result: u64 = @bitCast(@as(i64, @as(i32, @truncate(@as(i64, @bitCast(output_val))))));
        const step21_instr = buildVirtualSignExtendWordInstr(decoded.rd, output_reg);
        try self.lookup_trace.recordVirtualSignExtendWord(
            @intCast(self.state.cycle),
            self.state.pc,
            step21_instr,
            output_val,
            final_result,
        );
        try self.registers.write(decoded.rd, final_result);
        try self.trace.addStep(.{
            .cycle = self.state.cycle,
            .pc = self.state.pc,
            .unexpanded_pc = self.state.pc,
            .instruction = step21_instr,
            .rs1_value = output_val,
            .rs2_value = 0,
            .rd_pre_value = rd_pre_value,
            .rd_value = final_result,
            .rd_index = decoded.rd,
            .rs1_index = output_reg,
            .rs2_index = 0,
            .rd_written = true,
            .rs1_read = true,
            .rs2_read = false,
            .memory_addr = null,
            .memory_pre_value = null,
            .memory_value = null,
            .is_memory_write = false,
            .next_pc = self.state.pc + pc_increment,
            .is_compressed = self.is_compressed,
            .virtual_sequence_remaining = 0,
            .is_last_in_sequence = true,
        });

        // Update state
        self.prev_pc = self.state.pc;
        self.state.pc = self.state.pc + pc_increment;
        self.state.cycle += 1;
        self.registers.tick();

        return true;
    }

    // ======================== Instruction builder functions ========================

    /// Build VirtualPow2 instruction. opcode=0x2B, funct3=1
    fn buildVirtualPow2Instr(rd: u8, rs1: u8) u32 {
        return (@as(u32, rs1 & 0x1F) << 15) |
            (1 << 12) | // funct3 = 1 (Pow2)
            (@as(u32, rd & 0x1F) << 7) |
            0x2B;
    }

    /// Build VirtualShiftRightBitmask instruction. opcode=0x2B, funct3=2
    fn buildVirtualShiftRightBitmaskInstr(rd: u8, rs1: u8) u32 {
        return (@as(u32, rs1 & 0x1F) << 15) |
            (2 << 12) | // funct3 = 2
            (@as(u32, rd & 0x1F) << 7) |
            0x2B;
    }

    /// Build VirtualAssertHalfwordAlignment instruction. opcode=0x22, funct3=2
    fn buildVirtualAssertHalfwordAlignmentInstr(rs1: u8, imm_12: u12) u32 {
        return (@as(u32, imm_12) << 20) |
            (@as(u32, rs1 & 0x1F) << 15) |
            (2 << 12) |
            0x22;
    }

    /// Build VirtualAssertWordAlignment instruction. opcode=0x22, funct3=3
    fn buildVirtualAssertWordAlignmentInstr(rs1: u8, imm_12: u12) u32 {
        return (@as(u32, imm_12) << 20) |
            (@as(u32, rs1 & 0x1F) << 15) |
            (3 << 12) |
            0x22;
    }

    /// Build I-type ADDI instruction
    fn buildADDIInstr(rd: u8, rs1: u8, imm_12: u12) u32 {
        return (@as(u32, imm_12) << 20) |
            (@as(u32, rs1 & 0x1F) << 15) |
            (0 << 12) |
            (@as(u32, rd & 0x1F) << 7) |
            0x13;
    }

    /// Build I-type ANDI instruction
    fn buildANDIInstr(rd: u8, rs1: u8, imm_12: u12) u32 {
        return (@as(u32, imm_12) << 20) |
            (@as(u32, rs1 & 0x1F) << 15) |
            (7 << 12) |
            (@as(u32, rd & 0x1F) << 7) |
            0x13;
    }

    /// Build I-type XORI instruction
    fn buildXORIInstr(rd: u8, rs1: u8, imm_12: u12) u32 {
        return (@as(u32, imm_12) << 20) |
            (@as(u32, rs1 & 0x1F) << 15) |
            (4 << 12) |
            (@as(u32, rd & 0x1F) << 7) |
            0x13;
    }

    /// Build I-type ORI instruction
    fn buildORIInstr(rd: u8, rs1: u8, imm_12: u12) u32 {
        return (@as(u32, imm_12) << 20) |
            (@as(u32, rs1 & 0x1F) << 15) |
            (6 << 12) |
            (@as(u32, rd & 0x1F) << 7) |
            0x13;
    }

    /// Build I-type LD instruction
    fn buildLDInstr(rd: u8, rs1: u8) u32 {
        return (@as(u32, rs1 & 0x1F) << 15) |
            (3 << 12) | // funct3 = 3 (LD)
            (@as(u32, rd & 0x1F) << 7) |
            0x03;
    }

    /// Build U-type LUI instruction
    fn buildLUIInstr(rd: u8, imm_20: u20) u32 {
        return (@as(u32, imm_20) << 12) |
            (@as(u32, rd & 0x1F) << 7) |
            0x37;
    }

    /// Build S-type SD instruction
    fn buildSDInstr(rs1: u8, rs2: u8) u32 {
        return (@as(u32, rs2 & 0x1F) << 20) |
            (@as(u32, rs1 & 0x1F) << 15) |
            (3 << 12) | // funct3 = 3 (SD)
            0x23;
    }

    /// Build R-type AND instruction
    fn buildANDInstr(rd: u8, rs1: u8, rs2: u8) u32 {
        return (@as(u32, rs2 & 0x1F) << 20) |
            (@as(u32, rs1 & 0x1F) << 15) |
            (7 << 12) |
            (@as(u32, rd & 0x1F) << 7) |
            0x33;
    }

    /// Build VirtualSRL R-type instruction. opcode=0x5B, funct3=0
    fn buildVirtualSRLInstr(rd: u8, rs1: u8, rs2: u8) u32 {
        return (@as(u32, rs2 & 0x1F) << 20) |
            (@as(u32, rs1 & 0x1F) << 15) |
            (0 << 12) |
            (@as(u32, rd & 0x1F) << 7) |
            0x5B;
    }

    /// Build VirtualSRA R-type instruction. opcode=0x5B, funct3=5
    fn buildVirtualSRAInstr(rd: u8, rs1: u8, rs2: u8) u32 {
        return (@as(u32, rs2 & 0x1F) << 20) |
            (@as(u32, rs1 & 0x1F) << 15) |
            (5 << 12) |
            (@as(u32, rd & 0x1F) << 7) |
            0x5B;
    }

    /// Build R-type SLL instruction: SLL rd, rs1, rs2
    /// opcode=0x33, funct7=0x00, funct3=0x01
    fn buildSLLInstr(rd: u8, rs1: u8, rs2: u8) u32 {
        return (0x00 << 25) | // funct7 = 0x00
            (@as(u32, rs2 & 0x1F) << 20) |
            (@as(u32, rs1 & 0x1F) << 15) |
            (0x01 << 12) | // funct3 = 1 (SLL)
            (@as(u32, rd & 0x1F) << 7) |
            0x33; // opcode = OP
    }

    // ======================== Step functions for shifts ========================

    /// Execute SRAI as a single VirtualSRAI trace step.
    /// SRAI rd, rs1, shamt → VirtualSRAI(rd, rs1, bitmask)
    /// This is a standalone single-step decomposition (like SRLI).
    fn stepSRAI(
        self: *Emulator,
        _: u32,
        decoded: zkvm.instruction.DecodedInstruction,
    ) !bool {
        const rs1_value = try self.registers.read(decoded.rs1);
        const rs2_value = try self.registers.read(decoded.rs2);
        const rd_pre_value = try self.registers.read(decoded.rd);
        const pc_increment: u64 = if (self.is_compressed) 2 else 4;

        // Compute shift amount
        const imm_u32: u32 = @bitCast(@as(i32, @truncate(decoded.imm)));
        const shamt: u7 = @intCast(imm_u32 & 0x3F);

        // Compute result: arithmetic right shift
        const result_val: u64 = @bitCast(@as(i64, @bitCast(rs1_value)) >> @intCast(shamt));

        // Build synthetic VirtualSRAI instruction encoding
        const rd_u8: u8 = decoded.rd;
        const rs1_u8: u8 = decoded.rs1;
        const vsrai_instr = buildVirtualSRAIInstr(rd_u8, rs1_u8, shamt);

        // Record lookup trace for VirtualSRAI
        try self.lookup_trace.recordVirtualSRAI(
            @intCast(self.state.cycle),
            self.state.pc,
            vsrai_instr,
            rs1_value,
            false, // is_virtual: standalone (vsr = None)
            false, // do_not_update_pc
            false, // is_first_in_sequence
            self.is_compressed,
        );

        // Write result to register
        try self.registers.write(decoded.rd, result_val);

        // Record trace step (as VirtualSRAI, NOT SRAI)
        try self.trace.addStep(.{
            .cycle = self.state.cycle,
            .pc = self.state.pc,
            .unexpanded_pc = self.state.pc,
            .instruction = vsrai_instr,
            .rs1_value = rs1_value,
            .rs2_value = rs2_value,
            .rd_pre_value = rd_pre_value,
            .rd_value = result_val,
            .rd_index = rd_u8,
            .rs1_index = rs1_u8,
            .rs2_index = 0,
            .rd_written = true,
            .rs1_read = true,
            .rs2_read = false,
            .memory_addr = null,
            .memory_pre_value = null,
            .memory_value = null,
            .is_memory_write = false,
            .next_pc = self.state.pc + pc_increment,
            .is_compressed = self.is_compressed,
            .is_first_in_sequence = true,
            .is_last_in_sequence = true,
        });

        self.prev_pc = self.state.pc;
        self.state.pc = self.state.pc + pc_increment;
        self.state.cycle += 1;
        self.registers.tick();
        return true;
    }

    /// Execute SLL as a 2-step virtual sequence:
    /// Step 1: VirtualPow2(v0=40, rs2, 0) → pow2 = 1 << (rs2_value % 64)
    /// Step 2: MUL(rd, rs1, v0=40) → result = rs1_value * pow2
    fn stepSLL(
        self: *Emulator,
        _: u32,
        decoded: zkvm.instruction.DecodedInstruction,
    ) !bool {
        const rs1_value = try self.registers.read(decoded.rs1);
        const rs2_value = try self.registers.read(decoded.rs2);
        _ = try self.registers.read(decoded.rd); // rd pre-value tracked per-step
        const pc_increment: u64 = if (self.is_compressed) 2 else 4;

        const v0: u8 = 40; // First virtual alloc register
        const rd_u8: u8 = decoded.rd;
        const rs1_u8: u8 = decoded.rs1;
        const rs2_u8: u8 = decoded.rs2;

        // Compute pow2 = 1 << (rs2_value % 64)
        const shift_amount: u6 = @truncate(rs2_value & 0x3F);
        const pow2: u64 = @as(u64, 1) << shift_amount;

        // === Step 1: VirtualPow2(v0, rs2, 0) ===
        const step1_instr = buildVirtualPow2Instr(v0, rs2_u8);
        const v0_pre = try self.registers.read(v0);

        try self.lookup_trace.recordVirtualPow2(
            @intCast(self.state.cycle),
            self.state.pc,
            step1_instr,
            rs2_value,
            true, // is_virtual
            true, // do_not_update_pc
            true, // is_first_in_sequence
            self.is_compressed,
        );

        try self.registers.write(v0, pow2);

        try self.trace.addStep(.{
            .cycle = self.state.cycle,
            .pc = self.state.pc,
            .unexpanded_pc = self.state.pc,
            .instruction = step1_instr,
            .rs1_value = rs2_value,
            .rs2_value = 0,
            .rd_pre_value = v0_pre,
            .rd_value = pow2,
            .rd_index = v0,
            .rs1_index = rs2_u8,
            .rs2_index = 0,
            .rd_written = true,
            .rs1_read = true,
            .rs2_read = false,
            .memory_addr = null,
            .memory_pre_value = null,
            .memory_value = null,
            .is_memory_write = false,
            .next_pc = self.state.pc + pc_increment,
            .is_compressed = false,
            .virtual_sequence_remaining = 1,
            .is_first_in_sequence = true,
        });

        self.state.cycle += 1;
        self.registers.tick();

        // === Step 2: MUL(rd, rs1, v0) ===
        const v0_val = try self.registers.read(v0);
        const mul_result: u64 = rs1_value *% v0_val;
        const step2_instr = buildMULInstr(rd_u8, rs1_u8, v0);
        const rd_pre_value_step2 = try self.registers.read(decoded.rd);

        try self.lookup_trace.recordMulVirtual(
            @intCast(self.state.cycle),
            self.state.pc,
            step2_instr,
            rs1_value,
            v0_val,
            false, // do_not_update_pc: last step (vsr=0)
            false, // is_first_in_sequence
            self.is_compressed,
        );

        try self.registers.write(decoded.rd, mul_result);

        try self.trace.addStep(.{
            .cycle = self.state.cycle,
            .pc = self.state.pc,
            .unexpanded_pc = self.state.pc,
            .instruction = step2_instr,
            .rs1_value = rs1_value,
            .rs2_value = v0_val,
            .rd_pre_value = rd_pre_value_step2,
            .rd_value = mul_result,
            .rd_index = rd_u8,
            .rs1_index = rs1_u8,
            .rs2_index = v0,
            .rd_written = true,
            .rs1_read = true,
            .rs2_read = true,
            .memory_addr = null,
            .memory_pre_value = null,
            .memory_value = null,
            .is_memory_write = false,
            .next_pc = self.state.pc + pc_increment,
            .is_compressed = self.is_compressed,
            .virtual_sequence_remaining = 0,
            .is_last_in_sequence = true,
        });

        self.prev_pc = self.state.pc;
        self.state.pc = self.state.pc + pc_increment;
        self.state.cycle += 1;
        self.registers.tick();
        return true;
    }

    /// Execute SRL as a 2-step virtual sequence:
    /// Step 1: VirtualShiftRightBitmask(v0=40, rs2, 0) → bitmask
    /// Step 2: VirtualSRL(rd, rs1, v0=40) → result = rs1_value >> trailing_zeros(bitmask)
    fn stepSRL(
        self: *Emulator,
        _: u32,
        decoded: zkvm.instruction.DecodedInstruction,
    ) !bool {
        const rs1_value = try self.registers.read(decoded.rs1);
        const rs2_value = try self.registers.read(decoded.rs2);
        _ = try self.registers.read(decoded.rd);
        const pc_increment: u64 = if (self.is_compressed) 2 else 4;

        const v0: u8 = 40;
        const rd_u8: u8 = decoded.rd;
        const rs1_u8: u8 = decoded.rs1;
        const rs2_u8: u8 = decoded.rs2;

        // Compute bitmask from rs2_value
        const shift_amount: u6 = @truncate(rs2_value & 0x3F);
        const ones: u128 = (@as(u128, 1) << @intCast(64 - @as(u8, shift_amount))) - 1;
        const bitmask: u64 = @truncate(ones << shift_amount);

        // === Step 1: VirtualShiftRightBitmask(v0, rs2, 0) ===
        const step1_instr = buildVirtualShiftRightBitmaskInstr(v0, rs2_u8);
        const v0_pre = try self.registers.read(v0);

        try self.lookup_trace.recordVirtualShiftRightBitmask(
            @intCast(self.state.cycle),
            self.state.pc,
            step1_instr,
            rs2_value,
            true, // is_virtual
            true, // do_not_update_pc
            true, // is_first_in_sequence
            self.is_compressed,
        );

        try self.registers.write(v0, bitmask);

        try self.trace.addStep(.{
            .cycle = self.state.cycle,
            .pc = self.state.pc,
            .unexpanded_pc = self.state.pc,
            .instruction = step1_instr,
            .rs1_value = rs2_value,
            .rs2_value = 0,
            .rd_pre_value = v0_pre,
            .rd_value = bitmask,
            .rd_index = v0,
            .rs1_index = rs2_u8,
            .rs2_index = 0,
            .rd_written = true,
            .rs1_read = true,
            .rs2_read = false,
            .memory_addr = null,
            .memory_pre_value = null,
            .memory_value = null,
            .is_memory_write = false,
            .next_pc = self.state.pc + pc_increment,
            .is_compressed = false,
            .virtual_sequence_remaining = 1,
            .is_first_in_sequence = true,
        });

        self.state.cycle += 1;
        self.registers.tick();

        // === Step 2: VirtualSRL(rd, rs1, v0) ===
        const v0_val = try self.registers.read(v0);
        const srl_result: u64 = rs1_value >> @intCast(shift_amount);
        const step2_instr = buildVirtualSRLInstr(rd_u8, rs1_u8, v0);
        const rd_pre_value_step2 = try self.registers.read(decoded.rd);

        try self.lookup_trace.recordVirtualSRL_R(
            @intCast(self.state.cycle),
            self.state.pc,
            step2_instr,
            rs1_value,
            v0_val,
            true, // is_virtual
            false, // do_not_update_pc: last step
            false, // is_first_in_sequence
            self.is_compressed,
        );

        try self.registers.write(decoded.rd, srl_result);

        try self.trace.addStep(.{
            .cycle = self.state.cycle,
            .pc = self.state.pc,
            .unexpanded_pc = self.state.pc,
            .instruction = step2_instr,
            .rs1_value = rs1_value,
            .rs2_value = v0_val,
            .rd_pre_value = rd_pre_value_step2,
            .rd_value = srl_result,
            .rd_index = rd_u8,
            .rs1_index = rs1_u8,
            .rs2_index = v0,
            .rd_written = true,
            .rs1_read = true,
            .rs2_read = true,
            .memory_addr = null,
            .memory_pre_value = null,
            .memory_value = null,
            .is_memory_write = false,
            .next_pc = self.state.pc + pc_increment,
            .is_compressed = self.is_compressed,
            .virtual_sequence_remaining = 0,
            .is_last_in_sequence = true,
        });

        self.prev_pc = self.state.pc;
        self.state.pc = self.state.pc + pc_increment;
        self.state.cycle += 1;
        self.registers.tick();
        return true;
    }

    /// Execute SRA as a 2-step virtual sequence:
    /// Step 1: VirtualShiftRightBitmask(v0=40, rs2, 0) → bitmask
    /// Step 2: VirtualSRA(rd, rs1, v0=40) → result = arithmetic right shift
    fn stepSRA(
        self: *Emulator,
        _: u32,
        decoded: zkvm.instruction.DecodedInstruction,
    ) !bool {
        const rs1_value = try self.registers.read(decoded.rs1);
        const rs2_value = try self.registers.read(decoded.rs2);
        _ = try self.registers.read(decoded.rd);
        const pc_increment: u64 = if (self.is_compressed) 2 else 4;

        const v0: u8 = 40;
        const rd_u8: u8 = decoded.rd;
        const rs1_u8: u8 = decoded.rs1;
        const rs2_u8: u8 = decoded.rs2;

        // Compute bitmask from rs2_value
        const shift_amount: u6 = @truncate(rs2_value & 0x3F);
        const ones: u128 = (@as(u128, 1) << @intCast(64 - @as(u8, shift_amount))) - 1;
        const bitmask: u64 = @truncate(ones << shift_amount);

        // === Step 1: VirtualShiftRightBitmask(v0, rs2, 0) ===
        const step1_instr = buildVirtualShiftRightBitmaskInstr(v0, rs2_u8);
        const v0_pre = try self.registers.read(v0);

        try self.lookup_trace.recordVirtualShiftRightBitmask(
            @intCast(self.state.cycle),
            self.state.pc,
            step1_instr,
            rs2_value,
            true, // is_virtual
            true, // do_not_update_pc
            true, // is_first_in_sequence
            self.is_compressed,
        );

        try self.registers.write(v0, bitmask);

        try self.trace.addStep(.{
            .cycle = self.state.cycle,
            .pc = self.state.pc,
            .unexpanded_pc = self.state.pc,
            .instruction = step1_instr,
            .rs1_value = rs2_value,
            .rs2_value = 0,
            .rd_pre_value = v0_pre,
            .rd_value = bitmask,
            .rd_index = v0,
            .rs1_index = rs2_u8,
            .rs2_index = 0,
            .rd_written = true,
            .rs1_read = true,
            .rs2_read = false,
            .memory_addr = null,
            .memory_pre_value = null,
            .memory_value = null,
            .is_memory_write = false,
            .next_pc = self.state.pc + pc_increment,
            .is_compressed = false,
            .virtual_sequence_remaining = 1,
            .is_first_in_sequence = true,
        });

        self.state.cycle += 1;
        self.registers.tick();

        // === Step 2: VirtualSRA(rd, rs1, v0) ===
        const v0_val = try self.registers.read(v0);
        const sra_result: u64 = @bitCast(@as(i64, @bitCast(rs1_value)) >> @intCast(shift_amount));
        const step2_instr = buildVirtualSRAInstr(rd_u8, rs1_u8, v0);
        const rd_pre_value_step2 = try self.registers.read(decoded.rd);

        try self.lookup_trace.recordVirtualSRA_R(
            @intCast(self.state.cycle),
            self.state.pc,
            step2_instr,
            rs1_value,
            v0_val,
            true, // is_virtual
            false, // do_not_update_pc: last step
            false, // is_first_in_sequence
            self.is_compressed,
        );

        try self.registers.write(decoded.rd, sra_result);

        try self.trace.addStep(.{
            .cycle = self.state.cycle,
            .pc = self.state.pc,
            .unexpanded_pc = self.state.pc,
            .instruction = step2_instr,
            .rs1_value = rs1_value,
            .rs2_value = v0_val,
            .rd_pre_value = rd_pre_value_step2,
            .rd_value = sra_result,
            .rd_index = rd_u8,
            .rs1_index = rs1_u8,
            .rs2_index = v0,
            .rd_written = true,
            .rs1_read = true,
            .rs2_read = true,
            .memory_addr = null,
            .memory_pre_value = null,
            .memory_value = null,
            .is_memory_write = false,
            .next_pc = self.state.pc + pc_increment,
            .is_compressed = self.is_compressed,
            .virtual_sequence_remaining = 0,
            .is_last_in_sequence = true,
        });

        self.prev_pc = self.state.pc;
        self.state.pc = self.state.pc + pc_increment;
        self.state.cycle += 1;
        self.registers.tick();
        return true;
    }

    // ======================== Sub-word load step function ========================

    /// Execute a sub-word load as a multi-step virtual sequence (RV64 only).
    /// Handles: LB (funct3=0), LH (funct3=1), LW (funct3=2), LBU (funct3=4), LHU (funct3=5), LWU (funct3=6)
    ///
    /// LB/LBU (8 steps):  ADDI, ANDI, LD, XORI, VirtualMULI, VirtualPow2, MUL, VirtualSRAI/VirtualSRLI
    /// LH/LHU (9 steps):  VirtualAssertHalfwordAlignment, ADDI, ANDI, LD, XORI, VirtualMULI, VirtualPow2, MUL, VirtualSRAI/VirtualSRLI
    /// LW  (8 steps):     VirtualAssertWordAlignment, ADDI, ANDI, LD, VirtualMULI, VirtualShiftRightBitmask, VirtualSRL, VirtualSignExtendWord
    /// LWU (9 steps):     VirtualAssertWordAlignment, ADDI, ANDI, LD, XORI, VirtualMULI, VirtualPow2, MUL, VirtualSRLI
    fn stepSubWordLoad(
        self: *Emulator,
        _: u32,
        decoded: zkvm.instruction.DecodedInstruction,
    ) !bool {
        const rs1_value = try self.registers.read(decoded.rs1);
        _ = try self.registers.read(decoded.rs2);
        _ = try self.registers.read(decoded.rd);
        const pc_increment: u64 = if (self.is_compressed) 2 else 4;

        // Virtual registers for the sequence
        const v0: u8 = 40;
        const v1: u8 = 41;
        const rd_u8: u8 = decoded.rd;
        const rs1_u8: u8 = decoded.rs1;

        // Compute effective address
        const effective_addr: u64 = @bitCast(@as(i64, @bitCast(rs1_value)) +% decoded.imm);
        // Align to 8-byte boundary for doubleword access
        const aligned_addr: u64 = effective_addr & ~@as(u64, 7);
        // Immediate as 12-bit signed value for ADDI encoding
        const imm_12: u12 = @truncate(@as(u32, @bitCast(@as(i32, @truncate(decoded.imm)))));

        // Determine sequence parameters based on funct3
        const funct3 = decoded.funct3;
        const is_lb = funct3 == 0b000; // LB
        const is_lh = funct3 == 0b001; // LH
        const is_lw = funct3 == 0b010; // LW
        const is_lbu = funct3 == 0b100; // LBU
        const is_lhu = funct3 == 0b101; // LHU
        const is_lwu = funct3 == 0b110; // LWU

        // Total steps: LB/LBU=8, LH/LHU=9, LW=8, LWU=9
        const total_steps: u16 = if (is_lb or is_lbu) 8 else if (is_lh or is_lhu) 9 else if (is_lw) 8 else 9;
        var step_idx: u16 = 0;

        // === Optional alignment assert (LH/LHU/LW/LWU only) ===
        if (is_lh or is_lhu) {
            const align_instr = buildVirtualAssertHalfwordAlignmentInstr(rs1_u8, imm_12);
            try self.lookup_trace.recordVirtualAssertHalfwordAlignment(
                @intCast(self.state.cycle),
                self.state.pc,
                align_instr,
                rs1_value,
                @as(u64, @bitCast(@as(i64, decoded.imm))),
                true,
                true,
                step_idx == 0,
                self.is_compressed,
            );
            try self.trace.addStep(.{
                .cycle = self.state.cycle,
                .pc = self.state.pc,
                .unexpanded_pc = self.state.pc,
                .instruction = align_instr,
                .rs1_value = rs1_value,
                .rs2_value = 0,
                .rd_pre_value = 0,
                .rd_value = 0,
                .rd_index = 0,
                .rs1_index = rs1_u8,
                .rs2_index = 0,
                .rd_written = false,
                .rs1_read = true,
                .rs2_read = false,
                .memory_addr = null,
                .memory_pre_value = null,
                .memory_value = null,
                .is_memory_write = false,
                .next_pc = self.state.pc + pc_increment,
                .is_compressed = false,
                .virtual_sequence_remaining = total_steps - 1 - step_idx,
                .is_first_in_sequence = step_idx == 0,
            });
            self.state.cycle += 1;
            self.registers.tick();
            step_idx += 1;
        } else if (is_lw or is_lwu) {
            const align_instr = buildVirtualAssertWordAlignmentInstr(rs1_u8, imm_12);
            try self.lookup_trace.recordVirtualAssertWordAlignment(
                @intCast(self.state.cycle),
                self.state.pc,
                align_instr,
                rs1_value,
                @as(u64, @bitCast(@as(i64, decoded.imm))),
                true,
                true,
                step_idx == 0,
                self.is_compressed,
            );
            try self.trace.addStep(.{
                .cycle = self.state.cycle,
                .pc = self.state.pc,
                .unexpanded_pc = self.state.pc,
                .instruction = align_instr,
                .rs1_value = rs1_value,
                .rs2_value = 0,
                .rd_pre_value = 0,
                .rd_value = 0,
                .rd_index = 0,
                .rs1_index = rs1_u8,
                .rs2_index = 0,
                .rd_written = false,
                .rs1_read = true,
                .rs2_read = false,
                .memory_addr = null,
                .memory_pre_value = null,
                .memory_value = null,
                .is_memory_write = false,
                .next_pc = self.state.pc + pc_increment,
                .is_compressed = false,
                .virtual_sequence_remaining = total_steps - 1 - step_idx,
                .is_first_in_sequence = step_idx == 0,
            });
            self.state.cycle += 1;
            self.registers.tick();
            step_idx += 1;
        }

        // === ADDI(v0, rs1, imm) → effective address ===
        {
            const v0_pre = try self.registers.read(v0);
            const addi_instr = buildADDIInstr(v0, rs1_u8, imm_12);
            const addi_decoded = zkvm.instruction.DecodedInstruction.decode(addi_instr);
            const imm_u64: u64 = @bitCast(@as(i64, decoded.imm));
            try self.lookup_trace.recordInstruction(
                @intCast(self.state.cycle),
                self.state.pc,
                addi_instr,
                addi_decoded,
                rs1_value,
                imm_u64,
            );
            try self.registers.write(v0, effective_addr);
            try self.trace.addStep(.{
                .cycle = self.state.cycle,
                .pc = self.state.pc,
                .unexpanded_pc = self.state.pc,
                .instruction = addi_instr,
                .rs1_value = rs1_value,
                .rs2_value = 0,
                .rd_pre_value = v0_pre,
                .rd_value = effective_addr,
                .rd_index = v0,
                .rs1_index = rs1_u8,
                .rs2_index = 0,
                .rd_written = true,
                .rs1_read = true,
                .rs2_read = false,
                .memory_addr = null,
                .memory_pre_value = null,
                .memory_value = null,
                .is_memory_write = false,
                .next_pc = self.state.pc + pc_increment,
                .is_compressed = false,
                .virtual_sequence_remaining = total_steps - 1 - step_idx,
                .is_first_in_sequence = step_idx == 0,
            });
            self.state.cycle += 1;
            self.registers.tick();
            step_idx += 1;
        }

        // === ANDI(v1, v0, -8) → aligned address ===
        {
            const v1_pre = try self.registers.read(v1);
            const v0_val = try self.registers.read(v0);
            const mask_imm: u12 = @truncate(@as(u32, @bitCast(@as(i32, -8))));
            const andi_instr = buildANDIInstr(v1, v0, mask_imm);
            const andi_decoded = zkvm.instruction.DecodedInstruction.decode(andi_instr);
            const mask_u64: u64 = @bitCast(@as(i64, -8));
            try self.lookup_trace.recordInstruction(
                @intCast(self.state.cycle),
                self.state.pc,
                andi_instr,
                andi_decoded,
                v0_val,
                mask_u64,
            );
            try self.registers.write(v1, aligned_addr);
            try self.trace.addStep(.{
                .cycle = self.state.cycle,
                .pc = self.state.pc,
                .unexpanded_pc = self.state.pc,
                .instruction = andi_instr,
                .rs1_value = v0_val,
                .rs2_value = 0,
                .rd_pre_value = v1_pre,
                .rd_value = aligned_addr,
                .rd_index = v1,
                .rs1_index = v0,
                .rs2_index = 0,
                .rd_written = true,
                .rs1_read = true,
                .rs2_read = false,
                .memory_addr = null,
                .memory_pre_value = null,
                .memory_value = null,
                .is_memory_write = false,
                .next_pc = self.state.pc + pc_increment,
                .is_compressed = false,
                .virtual_sequence_remaining = total_steps - 1 - step_idx,
                .is_first_in_sequence = step_idx == 0,
            });
            self.state.cycle += 1;
            self.registers.tick();
            step_idx += 1;
        }

        // === LD(v1, v1, 0) → load doubleword from aligned address ===
        _ = blk: {
            const v1_val = try self.registers.read(v1);
            const dw_pre = v1_val; // v1's pre-value for LD step is the aligned address (it gets overwritten)
            const dw = try self.ram.read(aligned_addr, self.state.cycle);
            const ld_instr = buildLDInstr(v1, v1);
            const ld_decoded = zkvm.instruction.DecodedInstruction.decode(ld_instr);
            try self.lookup_trace.recordInstruction(
                @intCast(self.state.cycle),
                self.state.pc,
                ld_instr,
                ld_decoded,
                v1_val,
                0,
            );
            try self.registers.write(v1, dw);
            try self.trace.addStep(.{
                .cycle = self.state.cycle,
                .pc = self.state.pc,
                .unexpanded_pc = self.state.pc,
                .instruction = ld_instr,
                .rs1_value = v1_val,
                .rs2_value = 0,
                .rd_pre_value = dw_pre,
                .rd_value = dw,
                .rd_index = v1,
                .rs1_index = v1,
                .rs2_index = 0,
                .rd_written = true,
                .rs1_read = true,
                .rs2_read = false,
                .memory_addr = aligned_addr,
                .memory_pre_value = null,
                .memory_value = dw,
                .is_memory_write = false,
                .next_pc = self.state.pc + pc_increment,
                .is_compressed = false,
                .virtual_sequence_remaining = total_steps - 1 - step_idx,
                .is_first_in_sequence = step_idx == 0,
            });
            self.state.cycle += 1;
            self.registers.tick();
            step_idx += 1;
            break :blk dw;
        };

        // LW uses a different final sequence: SLLI, SRL, VirtualSignExtendWord
        if (is_lw) {
            // === LW path: VirtualMULI(v0, v0, shamt=v0*8), VirtualShiftRightBitmask, VirtualSRL, VirtualSignExtendWord ===

            // Step: VirtualMULI(v0, v0, shamt) — where shamt = (byte_offset * 8)
            // In upstream: emit_i::<SLLI>(*v0, *v0, 3) which becomes VirtualMULI(v0, v0, 3)
            {
                const v0_val = try self.registers.read(v0);
                const v0_pre = v0_val;
                const shamt_slli: u6 = 3;
                const multiplier: u64 = @as(u64, 1) << shamt_slli;
                const slli_result: u64 = v0_val *% multiplier;
                const vmuli_instr = buildVirtualMULIInstr(v0, v0, shamt_slli);

                try self.lookup_trace.recordVirtualMULI(
                    @intCast(self.state.cycle),
                    self.state.pc,
                    vmuli_instr,
                    v0_val,
                    multiplier,
                    true,
                    true,
                    false,
                    self.is_compressed,
                );

                try self.registers.write(v0, slli_result);
                try self.trace.addStep(.{
                    .cycle = self.state.cycle,
                    .pc = self.state.pc,
                    .unexpanded_pc = self.state.pc,
                    .instruction = vmuli_instr,
                    .rs1_value = v0_val,
                    .rs2_value = 0,
                    .rd_pre_value = v0_pre,
                    .rd_value = slli_result,
                    .rd_index = v0,
                    .rs1_index = v0,
                    .rs2_index = 0,
                    .rd_written = true,
                    .rs1_read = true,
                    .rs2_read = false,
                    .memory_addr = null,
                    .memory_pre_value = null,
                    .memory_value = null,
                    .is_memory_write = false,
                    .next_pc = self.state.pc + pc_increment,
                    .is_compressed = false,
                    .virtual_sequence_remaining = total_steps - 1 - step_idx,
                });
                self.state.cycle += 1;
                self.registers.tick();
                step_idx += 1;
            }

            // Step: SRL(v1, v1, v0) → this becomes VirtualShiftRightBitmask + VirtualSRL (2 sub-steps)
            // Actually, upstream emits emit_r::<SRL>(*v1, *v1, *v0) which recursively expands.
            // SRL inline_sequence = VirtualShiftRightBitmask(v_bitmask, rs2, 0) + VirtualSRL(rd, rs1, v_bitmask)
            // In the recursive expansion, SRL allocates its own virtual register. But in Zolt,
            // the allocator is not recursive. Looking at the virtual register allocation:
            // SRL's inline_sequence allocates v_bitmask. In upstream, the allocator starts from
            // where we left off. After v0=40, v1=41, the next allocation would be v2=42.
            const v2: u8 = 42;
            {
                // VirtualShiftRightBitmask(v2, v0, 0) → bitmask from v0's value (which is byte_offset * 8)
                const v0_val = try self.registers.read(v0);
                const v2_pre = try self.registers.read(v2);
                const srl_shift: u6 = @truncate(v0_val & 0x3F);
                const srl_ones: u128 = (@as(u128, 1) << @intCast(64 - @as(u8, srl_shift))) - 1;
                const srl_bitmask: u64 = @truncate(srl_ones << srl_shift);
                const srbm_instr = buildVirtualShiftRightBitmaskInstr(v2, v0);

                try self.lookup_trace.recordVirtualShiftRightBitmask(
                    @intCast(self.state.cycle),
                    self.state.pc,
                    srbm_instr,
                    v0_val,
                    true,
                    true,
                    false,
                    self.is_compressed,
                );

                try self.registers.write(v2, srl_bitmask);
                try self.trace.addStep(.{
                    .cycle = self.state.cycle,
                    .pc = self.state.pc,
                    .unexpanded_pc = self.state.pc,
                    .instruction = srbm_instr,
                    .rs1_value = v0_val,
                    .rs2_value = 0,
                    .rd_pre_value = v2_pre,
                    .rd_value = srl_bitmask,
                    .rd_index = v2,
                    .rs1_index = v0,
                    .rs2_index = 0,
                    .rd_written = true,
                    .rs1_read = true,
                    .rs2_read = false,
                    .memory_addr = null,
                    .memory_pre_value = null,
                    .memory_value = null,
                    .is_memory_write = false,
                    .next_pc = self.state.pc + pc_increment,
                    .is_compressed = false,
                    .virtual_sequence_remaining = total_steps - 1 - step_idx,
                });
                self.state.cycle += 1;
                self.registers.tick();
                step_idx += 1;

                // VirtualSRL(v1, v1, v2)
                const v1_val = try self.registers.read(v1);
                const v2_val = try self.registers.read(v2);
                const v1_pre_srl = v1_val;
                const srl_result: u64 = v1_val >> @intCast(srl_shift);
                const vsrl_instr = buildVirtualSRLInstr(v1, v1, v2);

                try self.lookup_trace.recordVirtualSRL_R(
                    @intCast(self.state.cycle),
                    self.state.pc,
                    vsrl_instr,
                    v1_val,
                    v2_val,
                    true,
                    true,
                    false,
                    self.is_compressed,
                );

                try self.registers.write(v1, srl_result);
                try self.trace.addStep(.{
                    .cycle = self.state.cycle,
                    .pc = self.state.pc,
                    .unexpanded_pc = self.state.pc,
                    .instruction = vsrl_instr,
                    .rs1_value = v1_val,
                    .rs2_value = v2_val,
                    .rd_pre_value = v1_pre_srl,
                    .rd_value = srl_result,
                    .rd_index = v1,
                    .rs1_index = v1,
                    .rs2_index = v2,
                    .rd_written = true,
                    .rs1_read = true,
                    .rs2_read = true,
                    .memory_addr = null,
                    .memory_pre_value = null,
                    .memory_value = null,
                    .is_memory_write = false,
                    .next_pc = self.state.pc + pc_increment,
                    .is_compressed = false,
                    .virtual_sequence_remaining = total_steps - 1 - step_idx,
                });
                self.state.cycle += 1;
                self.registers.tick();
                step_idx += 1;
            }

            // VirtualSignExtendWord(rd, v1, 0)
            {
                const v1_val = try self.registers.read(v1);
                const sign_extended: u64 = @bitCast(@as(i64, @as(i32, @truncate(@as(i64, @bitCast(v1_val))))));
                const vsew_instr = buildVirtualSignExtendWordInstr(rd_u8, v1);
                const rd_pre_final = try self.registers.read(decoded.rd);

                try self.lookup_trace.recordVirtualSignExtendWord(
                    @intCast(self.state.cycle),
                    self.state.pc,
                    vsew_instr,
                    v1_val,
                    sign_extended,
                );

                try self.registers.write(decoded.rd, sign_extended);
                try self.trace.addStep(.{
                    .cycle = self.state.cycle,
                    .pc = self.state.pc,
                    .unexpanded_pc = self.state.pc,
                    .instruction = vsew_instr,
                    .rs1_value = v1_val,
                    .rs2_value = 0,
                    .rd_pre_value = rd_pre_final,
                    .rd_value = sign_extended,
                    .rd_index = rd_u8,
                    .rs1_index = v1,
                    .rs2_index = 0,
                    .rd_written = true,
                    .rs1_read = true,
                    .rs2_read = false,
                    .memory_addr = null,
                    .memory_pre_value = null,
                    .memory_value = null,
                    .is_memory_write = false,
                    .next_pc = self.state.pc + pc_increment,
                    .is_compressed = self.is_compressed,
                    .virtual_sequence_remaining = 0,
                    .is_last_in_sequence = true,
                });
            }
        } else {
            // === LB/LBU/LH/LHU/LWU path: XORI, VirtualMULI(SLLI), VirtualPow2+MUL(SLL), VirtualSRAI/VirtualSRLI ===

            // XOR immediate value for byte/halfword/word extraction:
            // LB/LBU: XOR with 7 (byte position in doubleword)
            // LH/LHU: XOR with 6 (halfword position in doubleword)
            // LWU: XOR with 4 (word position in doubleword)
            const xor_val: u64 = if (is_lb or is_lbu) 7 else if (is_lh or is_lhu) 6 else 4;

            // === XORI(v0, v0, xor_val) ===
            {
                const v0_val = try self.registers.read(v0);
                const v0_pre = v0_val;
                const xori_result: u64 = v0_val ^ xor_val;
                const xori_instr = buildXORIInstr(v0, v0, @truncate(xor_val));
                const xori_decoded = zkvm.instruction.DecodedInstruction.decode(xori_instr);
                try self.lookup_trace.recordInstruction(
                    @intCast(self.state.cycle),
                    self.state.pc,
                    xori_instr,
                    xori_decoded,
                    v0_val,
                    xor_val,
                );
                try self.registers.write(v0, xori_result);
                try self.trace.addStep(.{
                    .cycle = self.state.cycle,
                    .pc = self.state.pc,
                    .unexpanded_pc = self.state.pc,
                    .instruction = xori_instr,
                    .rs1_value = v0_val,
                    .rs2_value = 0,
                    .rd_pre_value = v0_pre,
                    .rd_value = xori_result,
                    .rd_index = v0,
                    .rs1_index = v0,
                    .rs2_index = 0,
                    .rd_written = true,
                    .rs1_read = true,
                    .rs2_read = false,
                    .memory_addr = null,
                    .memory_pre_value = null,
                    .memory_value = null,
                    .is_memory_write = false,
                    .next_pc = self.state.pc + pc_increment,
                    .is_compressed = false,
                    .virtual_sequence_remaining = total_steps - 1 - step_idx,
                    .is_first_in_sequence = step_idx == 0,
                });
                self.state.cycle += 1;
                self.registers.tick();
                step_idx += 1;
            }

            // === VirtualMULI(v0, v0, shamt=3) — from SLLI(v0, v0, 3) ===
            {
                const v0_val = try self.registers.read(v0);
                const v0_pre = v0_val;
                const slli_shamt: u6 = 3;
                const multiplier: u64 = @as(u64, 1) << slli_shamt;
                const slli_result: u64 = v0_val *% multiplier;
                const vmuli_instr = buildVirtualMULIInstr(v0, v0, slli_shamt);

                try self.lookup_trace.recordVirtualMULI(
                    @intCast(self.state.cycle),
                    self.state.pc,
                    vmuli_instr,
                    v0_val,
                    multiplier,
                    true,
                    true,
                    false,
                    self.is_compressed,
                );

                try self.registers.write(v0, slli_result);
                try self.trace.addStep(.{
                    .cycle = self.state.cycle,
                    .pc = self.state.pc,
                    .unexpanded_pc = self.state.pc,
                    .instruction = vmuli_instr,
                    .rs1_value = v0_val,
                    .rs2_value = 0,
                    .rd_pre_value = v0_pre,
                    .rd_value = slli_result,
                    .rd_index = v0,
                    .rs1_index = v0,
                    .rs2_index = 0,
                    .rd_written = true,
                    .rs1_read = true,
                    .rs2_read = false,
                    .memory_addr = null,
                    .memory_pre_value = null,
                    .memory_value = null,
                    .is_memory_write = false,
                    .next_pc = self.state.pc + pc_increment,
                    .is_compressed = false,
                    .virtual_sequence_remaining = total_steps - 1 - step_idx,
                });
                self.state.cycle += 1;
                self.registers.tick();
                step_idx += 1;
            }

            // === SLL(v1, v1, v0) → VirtualPow2 + MUL (2 sub-steps from SLL decomposition) ===
            // SLL's inline_sequence: VirtualPow2(v_pow2, rs2, 0) + MUL(rd, rs1, v_pow2)
            // v_pow2 is the next allocated virtual register. After v0=40, v1=41, next is v2=42.
            const v2: u8 = 42;
            {
                // VirtualPow2(v2, v0, 0)
                const v0_val = try self.registers.read(v0);
                const v2_pre = try self.registers.read(v2);
                const pow2_shift: u6 = @truncate(v0_val & 0x3F);
                const pow2_val: u64 = @as(u64, 1) << pow2_shift;
                const vpow2_instr = buildVirtualPow2Instr(v2, v0);

                try self.lookup_trace.recordVirtualPow2(
                    @intCast(self.state.cycle),
                    self.state.pc,
                    vpow2_instr,
                    v0_val,
                    true,
                    true,
                    false,
                    self.is_compressed,
                );

                try self.registers.write(v2, pow2_val);
                try self.trace.addStep(.{
                    .cycle = self.state.cycle,
                    .pc = self.state.pc,
                    .unexpanded_pc = self.state.pc,
                    .instruction = vpow2_instr,
                    .rs1_value = v0_val,
                    .rs2_value = 0,
                    .rd_pre_value = v2_pre,
                    .rd_value = pow2_val,
                    .rd_index = v2,
                    .rs1_index = v0,
                    .rs2_index = 0,
                    .rd_written = true,
                    .rs1_read = true,
                    .rs2_read = false,
                    .memory_addr = null,
                    .memory_pre_value = null,
                    .memory_value = null,
                    .is_memory_write = false,
                    .next_pc = self.state.pc + pc_increment,
                    .is_compressed = false,
                    .virtual_sequence_remaining = total_steps - 1 - step_idx,
                });
                self.state.cycle += 1;
                self.registers.tick();
                step_idx += 1;

                // MUL(v1, v1, v2)
                const v1_val = try self.registers.read(v1);
                const v2_val = try self.registers.read(v2);
                const v1_pre_mul = v1_val;
                const mul_result: u64 = v1_val *% v2_val;
                const mul_instr = buildMULInstr(v1, v1, v2);

                try self.lookup_trace.recordMulVirtual(
                    @intCast(self.state.cycle),
                    self.state.pc,
                    mul_instr,
                    v1_val,
                    v2_val,
                    true,
                    false,
                    self.is_compressed,
                );

                try self.registers.write(v1, mul_result);
                try self.trace.addStep(.{
                    .cycle = self.state.cycle,
                    .pc = self.state.pc,
                    .unexpanded_pc = self.state.pc,
                    .instruction = mul_instr,
                    .rs1_value = v1_val,
                    .rs2_value = v2_val,
                    .rd_pre_value = v1_pre_mul,
                    .rd_value = mul_result,
                    .rd_index = v1,
                    .rs1_index = v1,
                    .rs2_index = v2,
                    .rd_written = true,
                    .rs1_read = true,
                    .rs2_read = true,
                    .memory_addr = null,
                    .memory_pre_value = null,
                    .memory_value = null,
                    .is_memory_write = false,
                    .next_pc = self.state.pc + pc_increment,
                    .is_compressed = false,
                    .virtual_sequence_remaining = total_steps - 1 - step_idx,
                });
                self.state.cycle += 1;
                self.registers.tick();
                step_idx += 1;
            }

            // === Final shift: VirtualSRAI (signed) or VirtualSRLI (unsigned) ===
            // LB:  SRAI by 56 (shift byte from MSB position, sign-extend)
            // LBU: SRLI by 56 (shift byte from MSB position, zero-extend)
            // LH:  SRAI by 48 (shift halfword from MSB position, sign-extend)
            // LHU: SRLI by 48 (shift halfword from MSB position, zero-extend)
            // LWU: SRLI by 32 (shift word from MSB position, zero-extend)
            const final_shift: u7 = if (is_lb or is_lbu) 56 else if (is_lh or is_lhu) 48 else 32;
            const is_signed = is_lb or is_lh;

            {
                const v1_val = try self.registers.read(v1);
                const rd_pre_final = try self.registers.read(decoded.rd);
                const final_result: u64 = if (is_signed)
                    @bitCast(@as(i64, @bitCast(v1_val)) >> @intCast(final_shift))
                else
                    v1_val >> @intCast(final_shift);

                if (is_signed) {
                    // VirtualSRAI(rd, v1, shift)
                    const vsrai_instr = buildVirtualSRAIInstr(rd_u8, v1, final_shift);
                    try self.lookup_trace.recordVirtualSRAI(
                        @intCast(self.state.cycle),
                        self.state.pc,
                        vsrai_instr,
                        v1_val,
                        true,
                        false,
                        false,
                        self.is_compressed,
                    );
                    try self.registers.write(decoded.rd, final_result);
                    try self.trace.addStep(.{
                        .cycle = self.state.cycle,
                        .pc = self.state.pc,
                        .unexpanded_pc = self.state.pc,
                        .instruction = vsrai_instr,
                        .rs1_value = v1_val,
                        .rs2_value = 0,
                        .rd_pre_value = rd_pre_final,
                        .rd_value = final_result,
                        .rd_index = rd_u8,
                        .rs1_index = v1,
                        .rs2_index = 0,
                        .rd_written = true,
                        .rs1_read = true,
                        .rs2_read = false,
                        .memory_addr = null,
                        .memory_pre_value = null,
                        .memory_value = null,
                        .is_memory_write = false,
                        .next_pc = self.state.pc + pc_increment,
                        .is_compressed = self.is_compressed,
                        .virtual_sequence_remaining = 0,
                        .is_last_in_sequence = true,
                    });
                } else {
                    // VirtualSRLI(rd, v1, total_shift)
                    const vsrli_instr = buildVirtualSRLIInstr(rd_u8, v1, final_shift);
                    const srli_ones: u128 = (@as(u128, 1) << @intCast(64 - @as(u8, final_shift))) - 1;
                    const srli_bitmask: u64 = @truncate(srli_ones << final_shift);
                    try self.lookup_trace.recordVirtualSRLI(
                        @intCast(self.state.cycle),
                        self.state.pc,
                        vsrli_instr,
                        v1_val,
                        srli_bitmask,
                        final_result,
                        true,
                        false,
                        false,
                        self.is_compressed,
                    );
                    try self.registers.write(decoded.rd, final_result);
                    try self.trace.addStep(.{
                        .cycle = self.state.cycle,
                        .pc = self.state.pc,
                        .unexpanded_pc = self.state.pc,
                        .instruction = vsrli_instr,
                        .rs1_value = v1_val,
                        .rs2_value = 0,
                        .rd_pre_value = rd_pre_final,
                        .rd_value = final_result,
                        .rd_index = rd_u8,
                        .rs1_index = v1,
                        .rs2_index = 0,
                        .rd_written = true,
                        .rs1_read = true,
                        .rs2_read = false,
                        .memory_addr = null,
                        .memory_pre_value = null,
                        .memory_value = null,
                        .is_memory_write = false,
                        .next_pc = self.state.pc + pc_increment,
                        .is_compressed = self.is_compressed,
                        .virtual_sequence_remaining = 0,
                        .is_last_in_sequence = true,
                    });
                }
            }
        }

        // Also execute the actual load on the emulator state for correctness
        // (The actual RISC-V result must be written to the destination register)
        // The register file already has the correct value from the virtual sequence steps.
        // However, we need to perform the actual sub-word memory operations through the
        // byte-level I/O path for device I/O compatibility.
        // For proving, the virtual sequence above is what matters.

        self.prev_pc = self.state.pc;
        self.state.pc = self.state.pc + pc_increment;
        self.state.cycle += 1;
        self.registers.tick();
        return true;
    }

    // ======================== Sub-word store step function ========================

    /// Execute a sub-word store as a multi-step virtual sequence (RV64 only).
    /// Handles: SB (funct3=0), SH (funct3=1), SW (funct3=2)
    ///
    /// SB (13 steps):  ADDI, ANDI, LD, VirtualMULI(SLLI), LUI, VirtualPow2+MUL(SLL), VirtualPow2+MUL(SLL), XOR, AND, XOR, SD
    /// SH (14 steps):  VirtualAssertHalfwordAlignment, ADDI, ANDI, LD, VirtualMULI(SLLI), LUI, VirtualPow2+MUL(SLL), VirtualPow2+MUL(SLL), XOR, AND, XOR, SD
    /// SW (15 steps):  VirtualAssertWordAlignment, ADDI, ANDI, LD, VirtualMULI(SLLI), ORI, VirtualSRLI(SRLI), VirtualPow2+MUL(SLL), VirtualPow2+MUL(SLL), XOR, AND, XOR, SD
    fn stepSubWordStore(
        self: *Emulator,
        _: u32,
        decoded: zkvm.instruction.DecodedInstruction,
    ) !bool {
        const rs1_value = try self.registers.read(decoded.rs1);
        const rs2_value = try self.registers.read(decoded.rs2);
        const pc_increment: u64 = if (self.is_compressed) 2 else 4;

        // Virtual registers
        const v0: u8 = 40;
        const v1: u8 = 41;
        const v2: u8 = 42;
        const v3: u8 = 43;
        const rs1_u8: u8 = decoded.rs1;
        const rs2_u8: u8 = decoded.rs2;

        // Compute effective address
        const effective_addr: u64 = @bitCast(@as(i64, @bitCast(rs1_value)) +% decoded.imm);
        const aligned_addr: u64 = effective_addr & ~@as(u64, 7);
        const imm_12: u12 = @truncate(@as(u32, @bitCast(@as(i32, @truncate(decoded.imm)))));

        const funct3 = decoded.funct3;
        const is_sb = funct3 == 0b000;
        const is_sh = funct3 == 0b001;
        const is_sw = funct3 == 0b010;

        // Total steps
        const total_steps: u16 = if (is_sb) 13 else if (is_sh) 14 else 15;
        var step_idx: u16 = 0;

        // === Optional alignment assert ===
        if (is_sh) {
            const align_instr = buildVirtualAssertHalfwordAlignmentInstr(rs1_u8, imm_12);
            try self.lookup_trace.recordVirtualAssertHalfwordAlignment(
                @intCast(self.state.cycle),
                self.state.pc,
                align_instr,
                rs1_value,
                @as(u64, @bitCast(@as(i64, decoded.imm))),
                true,
                true,
                true,
                self.is_compressed,
            );
            try self.trace.addStep(.{
                .cycle = self.state.cycle,
                .pc = self.state.pc,
                .unexpanded_pc = self.state.pc,
                .instruction = align_instr,
                .rs1_value = rs1_value,
                .rs2_value = 0,
                .rd_pre_value = 0,
                .rd_value = 0,
                .rd_index = 0,
                .rs1_index = rs1_u8,
                .rs2_index = 0,
                .rd_written = false,
                .rs1_read = true,
                .rs2_read = false,
                .memory_addr = null,
                .memory_pre_value = null,
                .memory_value = null,
                .is_memory_write = false,
                .next_pc = self.state.pc + pc_increment,
                .is_compressed = false,
                .virtual_sequence_remaining = total_steps - 1 - step_idx,
                .is_first_in_sequence = true,
            });
            self.state.cycle += 1;
            self.registers.tick();
            step_idx += 1;
        } else if (is_sw) {
            const align_instr = buildVirtualAssertWordAlignmentInstr(rs1_u8, imm_12);
            try self.lookup_trace.recordVirtualAssertWordAlignment(
                @intCast(self.state.cycle),
                self.state.pc,
                align_instr,
                rs1_value,
                @as(u64, @bitCast(@as(i64, decoded.imm))),
                true,
                true,
                true,
                self.is_compressed,
            );
            try self.trace.addStep(.{
                .cycle = self.state.cycle,
                .pc = self.state.pc,
                .unexpanded_pc = self.state.pc,
                .instruction = align_instr,
                .rs1_value = rs1_value,
                .rs2_value = 0,
                .rd_pre_value = 0,
                .rd_value = 0,
                .rd_index = 0,
                .rs1_index = rs1_u8,
                .rs2_index = 0,
                .rd_written = false,
                .rs1_read = true,
                .rs2_read = false,
                .memory_addr = null,
                .memory_pre_value = null,
                .memory_value = null,
                .is_memory_write = false,
                .next_pc = self.state.pc + pc_increment,
                .is_compressed = false,
                .virtual_sequence_remaining = total_steps - 1 - step_idx,
                .is_first_in_sequence = true,
            });
            self.state.cycle += 1;
            self.registers.tick();
            step_idx += 1;
        }

        // === ADDI(v0, rs1, imm) → effective address ===
        {
            const v0_pre = try self.registers.read(v0);
            const addi_instr = buildADDIInstr(v0, rs1_u8, imm_12);
            const addi_decoded = zkvm.instruction.DecodedInstruction.decode(addi_instr);
            const imm_u64: u64 = @bitCast(@as(i64, decoded.imm));
            try self.lookup_trace.recordInstruction(
                @intCast(self.state.cycle),
                self.state.pc,
                addi_instr,
                addi_decoded,
                rs1_value,
                imm_u64,
            );
            try self.registers.write(v0, effective_addr);
            try self.trace.addStep(.{
                .cycle = self.state.cycle,
                .pc = self.state.pc,
                .unexpanded_pc = self.state.pc,
                .instruction = addi_instr,
                .rs1_value = rs1_value,
                .rs2_value = 0,
                .rd_pre_value = v0_pre,
                .rd_value = effective_addr,
                .rd_index = v0,
                .rs1_index = rs1_u8,
                .rs2_index = 0,
                .rd_written = true,
                .rs1_read = true,
                .rs2_read = false,
                .memory_addr = null,
                .memory_pre_value = null,
                .memory_value = null,
                .is_memory_write = false,
                .next_pc = self.state.pc + pc_increment,
                .is_compressed = false,
                .virtual_sequence_remaining = total_steps - 1 - step_idx,
                .is_first_in_sequence = step_idx == 0,
            });
            self.state.cycle += 1;
            self.registers.tick();
            step_idx += 1;
        }

        // === ANDI(v1, v0, -8) → aligned address ===
        {
            const v1_pre = try self.registers.read(v1);
            const v0_val = try self.registers.read(v0);
            const mask_imm: u12 = @truncate(@as(u32, @bitCast(@as(i32, -8))));
            const andi_instr = buildANDIInstr(v1, v0, mask_imm);
            const andi_decoded = zkvm.instruction.DecodedInstruction.decode(andi_instr);
            const mask_u64: u64 = @bitCast(@as(i64, -8));
            try self.lookup_trace.recordInstruction(
                @intCast(self.state.cycle),
                self.state.pc,
                andi_instr,
                andi_decoded,
                v0_val,
                mask_u64,
            );
            try self.registers.write(v1, aligned_addr);
            try self.trace.addStep(.{
                .cycle = self.state.cycle,
                .pc = self.state.pc,
                .unexpanded_pc = self.state.pc,
                .instruction = andi_instr,
                .rs1_value = v0_val,
                .rs2_value = 0,
                .rd_pre_value = v1_pre,
                .rd_value = aligned_addr,
                .rd_index = v1,
                .rs1_index = v0,
                .rs2_index = 0,
                .rd_written = true,
                .rs1_read = true,
                .rs2_read = false,
                .memory_addr = null,
                .memory_pre_value = null,
                .memory_value = null,
                .is_memory_write = false,
                .next_pc = self.state.pc + pc_increment,
                .is_compressed = false,
                .virtual_sequence_remaining = total_steps - 1 - step_idx,
            });
            self.state.cycle += 1;
            self.registers.tick();
            step_idx += 1;
        }

        // === LD(v2, v1, 0) → load old doubleword ===
        const old_doubleword: u64 = blk: {
            const v1_val = try self.registers.read(v1);
            const v2_pre = try self.registers.read(v2);
            const dw = try self.ram.read(aligned_addr, self.state.cycle);
            const ld_instr = buildLDInstr(v2, v1);
            const ld_decoded = zkvm.instruction.DecodedInstruction.decode(ld_instr);
            try self.lookup_trace.recordInstruction(
                @intCast(self.state.cycle),
                self.state.pc,
                ld_instr,
                ld_decoded,
                v1_val,
                0,
            );
            try self.registers.write(v2, dw);
            try self.trace.addStep(.{
                .cycle = self.state.cycle,
                .pc = self.state.pc,
                .unexpanded_pc = self.state.pc,
                .instruction = ld_instr,
                .rs1_value = v1_val,
                .rs2_value = 0,
                .rd_pre_value = v2_pre,
                .rd_value = dw,
                .rd_index = v2,
                .rs1_index = v1,
                .rs2_index = 0,
                .rd_written = true,
                .rs1_read = true,
                .rs2_read = false,
                .memory_addr = aligned_addr,
                .memory_pre_value = null,
                .memory_value = dw,
                .is_memory_write = false,
                .next_pc = self.state.pc + pc_increment,
                .is_compressed = false,
                .virtual_sequence_remaining = total_steps - 1 - step_idx,
            });
            self.state.cycle += 1;
            self.registers.tick();
            step_idx += 1;
            break :blk dw;
        };

        if (is_sw) {
            // === SW path (15 steps total) ===
            // SW upstream: SLLI(v0, v0, 3), ORI(v3, 0, -1), SRLI(v3, v3, 32), SLL(v3, v3, v0),
            //              SLL(v0, rs2, v0), XOR(v0, v2, v0), AND(v0, v0, v3), XOR(v2, v2, v0), SD(v1, v2, 0)

            // VirtualMULI(v0, v0, shamt=3) — SLLI(v0, v0, 3)
            {
                const v0_val = try self.registers.read(v0);
                const v0_pre = v0_val;
                const slli_shamt: u6 = 3;
                const multiplier: u64 = @as(u64, 1) << slli_shamt;
                const slli_result: u64 = v0_val *% multiplier;
                const vmuli_instr = buildVirtualMULIInstr(v0, v0, slli_shamt);
                try self.lookup_trace.recordVirtualMULI(
                    @intCast(self.state.cycle),
                    self.state.pc,
                    vmuli_instr,
                    v0_val,
                    multiplier,
                    true,
                    true,
                    false,
                    self.is_compressed,
                );
                try self.registers.write(v0, slli_result);
                try self.trace.addStep(.{
                    .cycle = self.state.cycle,
                    .pc = self.state.pc,
                    .unexpanded_pc = self.state.pc,
                    .instruction = vmuli_instr,
                    .rs1_value = v0_val,
                    .rs2_value = 0,
                    .rd_pre_value = v0_pre,
                    .rd_value = slli_result,
                    .rd_index = v0,
                    .rs1_index = v0,
                    .rs2_index = 0,
                    .rd_written = true,
                    .rs1_read = true,
                    .rs2_read = false,
                    .memory_addr = null,
                    .memory_pre_value = null,
                    .memory_value = null,
                    .is_memory_write = false,
                    .next_pc = self.state.pc + pc_increment,
                    .is_compressed = false,
                    .virtual_sequence_remaining = total_steps - 1 - step_idx,
                });
                self.state.cycle += 1;
                self.registers.tick();
                step_idx += 1;
            }

            // ORI(v3, 0, -1) → v3 = 0xFFFFFFFFFFFFFFFF
            {
                const v3_pre = try self.registers.read(v3);
                const x0_val: u64 = 0;
                const ori_imm: u12 = @truncate(@as(u32, @bitCast(@as(i32, -1))));
                const ori_instr = buildORIInstr(v3, 0, ori_imm);
                const ori_decoded = zkvm.instruction.DecodedInstruction.decode(ori_instr);
                const all_ones: u64 = @bitCast(@as(i64, -1));
                try self.lookup_trace.recordInstruction(
                    @intCast(self.state.cycle),
                    self.state.pc,
                    ori_instr,
                    ori_decoded,
                    x0_val,
                    all_ones,
                );
                try self.registers.write(v3, all_ones);
                try self.trace.addStep(.{
                    .cycle = self.state.cycle,
                    .pc = self.state.pc,
                    .unexpanded_pc = self.state.pc,
                    .instruction = ori_instr,
                    .rs1_value = x0_val,
                    .rs2_value = 0,
                    .rd_pre_value = v3_pre,
                    .rd_value = all_ones,
                    .rd_index = v3,
                    .rs1_index = 0,
                    .rs2_index = 0,
                    .rd_written = true,
                    .rs1_read = true,
                    .rs2_read = false,
                    .memory_addr = null,
                    .memory_pre_value = null,
                    .memory_value = null,
                    .is_memory_write = false,
                    .next_pc = self.state.pc + pc_increment,
                    .is_compressed = false,
                    .virtual_sequence_remaining = total_steps - 1 - step_idx,
                });
                self.state.cycle += 1;
                self.registers.tick();
                step_idx += 1;
            }

            // VirtualSRLI(v3, v3, 32) — SRLI(v3, v3, 32)
            {
                const v3_val = try self.registers.read(v3);
                const v3_pre = v3_val;
                const srli_shift: u7 = 32;
                const srli_result: u64 = v3_val >> @intCast(srli_shift);
                const srli_ones: u128 = (@as(u128, 1) << @intCast(64 - @as(u8, srli_shift))) - 1;
                const srli_bitmask: u64 = @truncate(srli_ones << srli_shift);
                const vsrli_instr = buildVirtualSRLIInstr(v3, v3, srli_shift);
                try self.lookup_trace.recordVirtualSRLI(
                    @intCast(self.state.cycle),
                    self.state.pc,
                    vsrli_instr,
                    v3_val,
                    srli_bitmask,
                    srli_result,
                    true,
                    true,
                    false,
                    self.is_compressed,
                );
                try self.registers.write(v3, srli_result);
                try self.trace.addStep(.{
                    .cycle = self.state.cycle,
                    .pc = self.state.pc,
                    .unexpanded_pc = self.state.pc,
                    .instruction = vsrli_instr,
                    .rs1_value = v3_val,
                    .rs2_value = 0,
                    .rd_pre_value = v3_pre,
                    .rd_value = srli_result,
                    .rd_index = v3,
                    .rs1_index = v3,
                    .rs2_index = 0,
                    .rd_written = true,
                    .rs1_read = true,
                    .rs2_read = false,
                    .memory_addr = null,
                    .memory_pre_value = null,
                    .memory_value = null,
                    .is_memory_write = false,
                    .next_pc = self.state.pc + pc_increment,
                    .is_compressed = false,
                    .virtual_sequence_remaining = total_steps - 1 - step_idx,
                });
                self.state.cycle += 1;
                self.registers.tick();
                step_idx += 1;
            }

            // SLL(v3, v3, v0) → VirtualPow2(v4, v0) + MUL(v3, v3, v4)
            const v4: u8 = 44;
            {
                const v0_val = try self.registers.read(v0);
                const v4_pre = try self.registers.read(v4);
                const pow2_shift: u6 = @truncate(v0_val & 0x3F);
                const pow2_val: u64 = @as(u64, 1) << pow2_shift;
                const vpow2_instr = buildVirtualPow2Instr(v4, v0);
                try self.lookup_trace.recordVirtualPow2(
                    @intCast(self.state.cycle),
                    self.state.pc,
                    vpow2_instr,
                    v0_val,
                    true,
                    true,
                    false,
                    self.is_compressed,
                );
                try self.registers.write(v4, pow2_val);
                try self.trace.addStep(.{
                    .cycle = self.state.cycle,
                    .pc = self.state.pc,
                    .unexpanded_pc = self.state.pc,
                    .instruction = vpow2_instr,
                    .rs1_value = v0_val,
                    .rs2_value = 0,
                    .rd_pre_value = v4_pre,
                    .rd_value = pow2_val,
                    .rd_index = v4,
                    .rs1_index = v0,
                    .rs2_index = 0,
                    .rd_written = true,
                    .rs1_read = true,
                    .rs2_read = false,
                    .memory_addr = null,
                    .memory_pre_value = null,
                    .memory_value = null,
                    .is_memory_write = false,
                    .next_pc = self.state.pc + pc_increment,
                    .is_compressed = false,
                    .virtual_sequence_remaining = total_steps - 1 - step_idx,
                });
                self.state.cycle += 1;
                self.registers.tick();
                step_idx += 1;

                const v3_val = try self.registers.read(v3);
                const v4_val = try self.registers.read(v4);
                const v3_pre_mul = v3_val;
                const mul_result: u64 = v3_val *% v4_val;
                const mul_instr = buildMULInstr(v3, v3, v4);
                try self.lookup_trace.recordMulVirtual(
                    @intCast(self.state.cycle),
                    self.state.pc,
                    mul_instr,
                    v3_val,
                    v4_val,
                    true,
                    false,
                    self.is_compressed,
                );
                try self.registers.write(v3, mul_result);
                try self.trace.addStep(.{
                    .cycle = self.state.cycle,
                    .pc = self.state.pc,
                    .unexpanded_pc = self.state.pc,
                    .instruction = mul_instr,
                    .rs1_value = v3_val,
                    .rs2_value = v4_val,
                    .rd_pre_value = v3_pre_mul,
                    .rd_value = mul_result,
                    .rd_index = v3,
                    .rs1_index = v3,
                    .rs2_index = v4,
                    .rd_written = true,
                    .rs1_read = true,
                    .rs2_read = true,
                    .memory_addr = null,
                    .memory_pre_value = null,
                    .memory_value = null,
                    .is_memory_write = false,
                    .next_pc = self.state.pc + pc_increment,
                    .is_compressed = false,
                    .virtual_sequence_remaining = total_steps - 1 - step_idx,
                });
                self.state.cycle += 1;
                self.registers.tick();
                step_idx += 1;
            }

            // SLL(v0, rs2, v0) → VirtualPow2(v5, v0_shift_amount) + MUL(v0, rs2, v5)
            // Wait - v0 has the shift amount already. We need to re-read it.
            // Actually in upstream: emit_r::<SLL>(*v0, self.operands.rs2, *v0)
            // SLL decomposes to VirtualPow2(v_new, v0) + MUL(v0, rs2, v_new)
            const v5: u8 = 45;
            {
                const v0_val = try self.registers.read(v0);
                const v5_pre = try self.registers.read(v5);
                const pow2_shift2: u6 = @truncate(v0_val & 0x3F);
                const pow2_val2: u64 = @as(u64, 1) << pow2_shift2;
                const vpow2_instr2 = buildVirtualPow2Instr(v5, v0);
                try self.lookup_trace.recordVirtualPow2(
                    @intCast(self.state.cycle),
                    self.state.pc,
                    vpow2_instr2,
                    v0_val,
                    true,
                    true,
                    false,
                    self.is_compressed,
                );
                try self.registers.write(v5, pow2_val2);
                try self.trace.addStep(.{
                    .cycle = self.state.cycle,
                    .pc = self.state.pc,
                    .unexpanded_pc = self.state.pc,
                    .instruction = vpow2_instr2,
                    .rs1_value = v0_val,
                    .rs2_value = 0,
                    .rd_pre_value = v5_pre,
                    .rd_value = pow2_val2,
                    .rd_index = v5,
                    .rs1_index = v0,
                    .rs2_index = 0,
                    .rd_written = true,
                    .rs1_read = true,
                    .rs2_read = false,
                    .memory_addr = null,
                    .memory_pre_value = null,
                    .memory_value = null,
                    .is_memory_write = false,
                    .next_pc = self.state.pc + pc_increment,
                    .is_compressed = false,
                    .virtual_sequence_remaining = total_steps - 1 - step_idx,
                });
                self.state.cycle += 1;
                self.registers.tick();
                step_idx += 1;

                const v0_pre = try self.registers.read(v0);
                const v5_val = try self.registers.read(v5);
                const mul_result2: u64 = rs2_value *% v5_val;
                const mul_instr2 = buildMULInstr(v0, rs2_u8, v5);
                try self.lookup_trace.recordMulVirtual(
                    @intCast(self.state.cycle),
                    self.state.pc,
                    mul_instr2,
                    rs2_value,
                    v5_val,
                    true,
                    false,
                    self.is_compressed,
                );
                try self.registers.write(v0, mul_result2);
                try self.trace.addStep(.{
                    .cycle = self.state.cycle,
                    .pc = self.state.pc,
                    .unexpanded_pc = self.state.pc,
                    .instruction = mul_instr2,
                    .rs1_value = rs2_value,
                    .rs2_value = v5_val,
                    .rd_pre_value = v0_pre,
                    .rd_value = mul_result2,
                    .rd_index = v0,
                    .rs1_index = rs2_u8,
                    .rs2_index = v5,
                    .rd_written = true,
                    .rs1_read = true,
                    .rs2_read = true,
                    .memory_addr = null,
                    .memory_pre_value = null,
                    .memory_value = null,
                    .is_memory_write = false,
                    .next_pc = self.state.pc + pc_increment,
                    .is_compressed = false,
                    .virtual_sequence_remaining = total_steps - 1 - step_idx,
                });
                self.state.cycle += 1;
                self.registers.tick();
                step_idx += 1;
            }

            // XOR(v0, v2, v0), AND(v0, v0, v3), XOR(v2, v2, v0), SD(v1, v2, 0)
            // These are the final merge steps
            {
                // XOR(v0, v2, v0)
                const v2_val = try self.registers.read(v2);
                const v0_val = try self.registers.read(v0);
                const v0_pre = v0_val;
                const xor_result: u64 = v2_val ^ v0_val;
                const xor_instr = buildXORInstr(v0, v2, v0);
                try self.lookup_trace.recordXorVirtual(
                    @intCast(self.state.cycle),
                    self.state.pc,
                    xor_instr,
                    v2_val,
                    v0_val,
                    true,
                    false,
                    self.is_compressed,
                );
                try self.registers.write(v0, xor_result);
                try self.trace.addStep(.{
                    .cycle = self.state.cycle,
                    .pc = self.state.pc,
                    .unexpanded_pc = self.state.pc,
                    .instruction = xor_instr,
                    .rs1_value = v2_val,
                    .rs2_value = v0_val,
                    .rd_pre_value = v0_pre,
                    .rd_value = xor_result,
                    .rd_index = v0,
                    .rs1_index = v2,
                    .rs2_index = v0,
                    .rd_written = true,
                    .rs1_read = true,
                    .rs2_read = true,
                    .memory_addr = null,
                    .memory_pre_value = null,
                    .memory_value = null,
                    .is_memory_write = false,
                    .next_pc = self.state.pc + pc_increment,
                    .is_compressed = false,
                    .virtual_sequence_remaining = total_steps - 1 - step_idx,
                });
                self.state.cycle += 1;
                self.registers.tick();
                step_idx += 1;
            }

            {
                // AND(v0, v0, v3)
                const v0_val = try self.registers.read(v0);
                const v3_val = try self.registers.read(v3);
                const v0_pre = v0_val;
                const and_result: u64 = v0_val & v3_val;
                const and_instr = buildANDInstr(v0, v0, v3);
                const and_decoded = zkvm.instruction.DecodedInstruction.decode(and_instr);
                try self.lookup_trace.recordInstruction(
                    @intCast(self.state.cycle),
                    self.state.pc,
                    and_instr,
                    and_decoded,
                    v0_val,
                    v3_val,
                );
                try self.registers.write(v0, and_result);
                try self.trace.addStep(.{
                    .cycle = self.state.cycle,
                    .pc = self.state.pc,
                    .unexpanded_pc = self.state.pc,
                    .instruction = and_instr,
                    .rs1_value = v0_val,
                    .rs2_value = v3_val,
                    .rd_pre_value = v0_pre,
                    .rd_value = and_result,
                    .rd_index = v0,
                    .rs1_index = v0,
                    .rs2_index = v3,
                    .rd_written = true,
                    .rs1_read = true,
                    .rs2_read = true,
                    .memory_addr = null,
                    .memory_pre_value = null,
                    .memory_value = null,
                    .is_memory_write = false,
                    .next_pc = self.state.pc + pc_increment,
                    .is_compressed = false,
                    .virtual_sequence_remaining = total_steps - 1 - step_idx,
                });
                self.state.cycle += 1;
                self.registers.tick();
                step_idx += 1;
            }

            {
                // XOR(v2, v2, v0)
                const v2_val = try self.registers.read(v2);
                const v0_val = try self.registers.read(v0);
                const v2_pre = v2_val;
                const new_dw: u64 = v2_val ^ v0_val;
                const xor_instr2 = buildXORInstr(v2, v2, v0);
                try self.lookup_trace.recordXorVirtual(
                    @intCast(self.state.cycle),
                    self.state.pc,
                    xor_instr2,
                    v2_val,
                    v0_val,
                    true,
                    false,
                    self.is_compressed,
                );
                try self.registers.write(v2, new_dw);
                try self.trace.addStep(.{
                    .cycle = self.state.cycle,
                    .pc = self.state.pc,
                    .unexpanded_pc = self.state.pc,
                    .instruction = xor_instr2,
                    .rs1_value = v2_val,
                    .rs2_value = v0_val,
                    .rd_pre_value = v2_pre,
                    .rd_value = new_dw,
                    .rd_index = v2,
                    .rs1_index = v2,
                    .rs2_index = v0,
                    .rd_written = true,
                    .rs1_read = true,
                    .rs2_read = true,
                    .memory_addr = null,
                    .memory_pre_value = null,
                    .memory_value = null,
                    .is_memory_write = false,
                    .next_pc = self.state.pc + pc_increment,
                    .is_compressed = false,
                    .virtual_sequence_remaining = total_steps - 1 - step_idx,
                });
                self.state.cycle += 1;
                self.registers.tick();
                step_idx += 1;
            }

            // SD(v1, v2, 0) → store new doubleword
            {
                const v1_val = try self.registers.read(v1);
                const v2_val = try self.registers.read(v2);
                const sd_instr = buildSDInstr(v1, v2);
                try self.ram.write(aligned_addr, v2_val, self.state.cycle);
                const sd_decoded = zkvm.instruction.DecodedInstruction.decode(sd_instr);
                try self.lookup_trace.recordInstruction(
                    @intCast(self.state.cycle),
                    self.state.pc,
                    sd_instr,
                    sd_decoded,
                    v1_val,
                    v2_val,
                );
                try self.trace.addStep(.{
                    .cycle = self.state.cycle,
                    .pc = self.state.pc,
                    .unexpanded_pc = self.state.pc,
                    .instruction = sd_instr,
                    .rs1_value = v1_val,
                    .rs2_value = v2_val,
                    .rd_pre_value = 0,
                    .rd_value = 0,
                    .rd_index = 0,
                    .rs1_index = v1,
                    .rs2_index = v2,
                    .rd_written = false,
                    .rs1_read = true,
                    .rs2_read = true,
                    .memory_addr = aligned_addr,
                    .memory_pre_value = old_doubleword,
                    .memory_value = v2_val,
                    .is_memory_write = true,
                    .next_pc = self.state.pc + pc_increment,
                    .is_compressed = self.is_compressed,
                    .virtual_sequence_remaining = 0,
                    .is_last_in_sequence = true,
                });
            }
        } else {
            // === SB/SH path ===
            // SB: SLLI(v3, v0, 3), LUI(v0, mask), SLL(v0, v0, v3), SLL(v3, rs2, v3),
            //     XOR(v3, v2, v3), AND(v3, v3, v0), XOR(v2, v2, v3), SD(v1, v2, 0)
            // SH: same but LUI mask is 0xffff instead of 0xff

            const lui_mask: u20 = if (is_sb) 0xff else 0xffff;

            // VirtualMULI(v3, v0, shamt=3) — SLLI(v3, v0, 3)
            {
                const v0_val = try self.registers.read(v0);
                const v3_pre = try self.registers.read(v3);
                const slli_shamt: u6 = 3;
                const multiplier: u64 = @as(u64, 1) << slli_shamt;
                const slli_result: u64 = v0_val *% multiplier;
                const vmuli_instr = buildVirtualMULIInstr(v3, v0, slli_shamt);
                try self.lookup_trace.recordVirtualMULI(
                    @intCast(self.state.cycle),
                    self.state.pc,
                    vmuli_instr,
                    v0_val,
                    multiplier,
                    true,
                    true,
                    false,
                    self.is_compressed,
                );
                try self.registers.write(v3, slli_result);
                try self.trace.addStep(.{
                    .cycle = self.state.cycle,
                    .pc = self.state.pc,
                    .unexpanded_pc = self.state.pc,
                    .instruction = vmuli_instr,
                    .rs1_value = v0_val,
                    .rs2_value = 0,
                    .rd_pre_value = v3_pre,
                    .rd_value = slli_result,
                    .rd_index = v3,
                    .rs1_index = v0,
                    .rs2_index = 0,
                    .rd_written = true,
                    .rs1_read = true,
                    .rs2_read = false,
                    .memory_addr = null,
                    .memory_pre_value = null,
                    .memory_value = null,
                    .is_memory_write = false,
                    .next_pc = self.state.pc + pc_increment,
                    .is_compressed = false,
                    .virtual_sequence_remaining = total_steps - 1 - step_idx,
                });
                self.state.cycle += 1;
                self.registers.tick();
                step_idx += 1;
            }

            // LUI(v0, mask)
            {
                const v0_pre = try self.registers.read(v0);
                const lui_instr = buildLUIInstr(v0, lui_mask);
                const lui_result: u64 = @bitCast(@as(i64, @as(i32, @bitCast(@as(u32, lui_mask) << 12))));
                const lui_decoded = zkvm.instruction.DecodedInstruction.decode(lui_instr);
                try self.lookup_trace.recordInstruction(
                    @intCast(self.state.cycle),
                    self.state.pc,
                    lui_instr,
                    lui_decoded,
                    0,
                    0,
                );
                try self.registers.write(v0, lui_result);
                try self.trace.addStep(.{
                    .cycle = self.state.cycle,
                    .pc = self.state.pc,
                    .unexpanded_pc = self.state.pc,
                    .instruction = lui_instr,
                    .rs1_value = 0,
                    .rs2_value = 0,
                    .rd_pre_value = v0_pre,
                    .rd_value = lui_result,
                    .rd_index = v0,
                    .rs1_index = 0,
                    .rs2_index = 0,
                    .rd_written = true,
                    .rs1_read = false,
                    .rs2_read = false,
                    .memory_addr = null,
                    .memory_pre_value = null,
                    .memory_value = null,
                    .is_memory_write = false,
                    .next_pc = self.state.pc + pc_increment,
                    .is_compressed = false,
                    .virtual_sequence_remaining = total_steps - 1 - step_idx,
                });
                self.state.cycle += 1;
                self.registers.tick();
                step_idx += 1;
            }

            // SLL(v0, v0, v3) → VirtualPow2(v4, v3) + MUL(v0, v0, v4)
            const v4: u8 = 44;
            {
                const v3_val = try self.registers.read(v3);
                const v4_pre = try self.registers.read(v4);
                const pow2_shift: u6 = @truncate(v3_val & 0x3F);
                const pow2_val: u64 = @as(u64, 1) << pow2_shift;
                const vpow2_instr = buildVirtualPow2Instr(v4, v3);
                try self.lookup_trace.recordVirtualPow2(
                    @intCast(self.state.cycle),
                    self.state.pc,
                    vpow2_instr,
                    v3_val,
                    true,
                    true,
                    false,
                    self.is_compressed,
                );
                try self.registers.write(v4, pow2_val);
                try self.trace.addStep(.{
                    .cycle = self.state.cycle,
                    .pc = self.state.pc,
                    .unexpanded_pc = self.state.pc,
                    .instruction = vpow2_instr,
                    .rs1_value = v3_val,
                    .rs2_value = 0,
                    .rd_pre_value = v4_pre,
                    .rd_value = pow2_val,
                    .rd_index = v4,
                    .rs1_index = v3,
                    .rs2_index = 0,
                    .rd_written = true,
                    .rs1_read = true,
                    .rs2_read = false,
                    .memory_addr = null,
                    .memory_pre_value = null,
                    .memory_value = null,
                    .is_memory_write = false,
                    .next_pc = self.state.pc + pc_increment,
                    .is_compressed = false,
                    .virtual_sequence_remaining = total_steps - 1 - step_idx,
                });
                self.state.cycle += 1;
                self.registers.tick();
                step_idx += 1;

                const v0_val = try self.registers.read(v0);
                const v4_val = try self.registers.read(v4);
                const v0_pre = v0_val;
                const mul_result: u64 = v0_val *% v4_val;
                const mul_instr = buildMULInstr(v0, v0, v4);
                try self.lookup_trace.recordMulVirtual(
                    @intCast(self.state.cycle),
                    self.state.pc,
                    mul_instr,
                    v0_val,
                    v4_val,
                    true,
                    false,
                    self.is_compressed,
                );
                try self.registers.write(v0, mul_result);
                try self.trace.addStep(.{
                    .cycle = self.state.cycle,
                    .pc = self.state.pc,
                    .unexpanded_pc = self.state.pc,
                    .instruction = mul_instr,
                    .rs1_value = v0_val,
                    .rs2_value = v4_val,
                    .rd_pre_value = v0_pre,
                    .rd_value = mul_result,
                    .rd_index = v0,
                    .rs1_index = v0,
                    .rs2_index = v4,
                    .rd_written = true,
                    .rs1_read = true,
                    .rs2_read = true,
                    .memory_addr = null,
                    .memory_pre_value = null,
                    .memory_value = null,
                    .is_memory_write = false,
                    .next_pc = self.state.pc + pc_increment,
                    .is_compressed = false,
                    .virtual_sequence_remaining = total_steps - 1 - step_idx,
                });
                self.state.cycle += 1;
                self.registers.tick();
                step_idx += 1;
            }

            // SLL(v3, rs2, v3) → VirtualPow2(v5, v3) + MUL(v3, rs2, v5)
            const v5: u8 = 45;
            {
                const v3_val = try self.registers.read(v3);
                const v5_pre = try self.registers.read(v5);
                const pow2_shift2: u6 = @truncate(v3_val & 0x3F);
                const pow2_val2: u64 = @as(u64, 1) << pow2_shift2;
                const vpow2_instr2 = buildVirtualPow2Instr(v5, v3);
                try self.lookup_trace.recordVirtualPow2(
                    @intCast(self.state.cycle),
                    self.state.pc,
                    vpow2_instr2,
                    v3_val,
                    true,
                    true,
                    false,
                    self.is_compressed,
                );
                try self.registers.write(v5, pow2_val2);
                try self.trace.addStep(.{
                    .cycle = self.state.cycle,
                    .pc = self.state.pc,
                    .unexpanded_pc = self.state.pc,
                    .instruction = vpow2_instr2,
                    .rs1_value = v3_val,
                    .rs2_value = 0,
                    .rd_pre_value = v5_pre,
                    .rd_value = pow2_val2,
                    .rd_index = v5,
                    .rs1_index = v3,
                    .rs2_index = 0,
                    .rd_written = true,
                    .rs1_read = true,
                    .rs2_read = false,
                    .memory_addr = null,
                    .memory_pre_value = null,
                    .memory_value = null,
                    .is_memory_write = false,
                    .next_pc = self.state.pc + pc_increment,
                    .is_compressed = false,
                    .virtual_sequence_remaining = total_steps - 1 - step_idx,
                });
                self.state.cycle += 1;
                self.registers.tick();
                step_idx += 1;

                const v3_pre = try self.registers.read(v3);
                const v5_val = try self.registers.read(v5);
                const mul_result2: u64 = rs2_value *% v5_val;
                const mul_instr2 = buildMULInstr(v3, rs2_u8, v5);
                try self.lookup_trace.recordMulVirtual(
                    @intCast(self.state.cycle),
                    self.state.pc,
                    mul_instr2,
                    rs2_value,
                    v5_val,
                    true,
                    false,
                    self.is_compressed,
                );
                try self.registers.write(v3, mul_result2);
                try self.trace.addStep(.{
                    .cycle = self.state.cycle,
                    .pc = self.state.pc,
                    .unexpanded_pc = self.state.pc,
                    .instruction = mul_instr2,
                    .rs1_value = rs2_value,
                    .rs2_value = v5_val,
                    .rd_pre_value = v3_pre,
                    .rd_value = mul_result2,
                    .rd_index = v3,
                    .rs1_index = rs2_u8,
                    .rs2_index = v5,
                    .rd_written = true,
                    .rs1_read = true,
                    .rs2_read = true,
                    .memory_addr = null,
                    .memory_pre_value = null,
                    .memory_value = null,
                    .is_memory_write = false,
                    .next_pc = self.state.pc + pc_increment,
                    .is_compressed = false,
                    .virtual_sequence_remaining = total_steps - 1 - step_idx,
                });
                self.state.cycle += 1;
                self.registers.tick();
                step_idx += 1;
            }

            // XOR(v3, v2, v3)
            {
                const v2_val = try self.registers.read(v2);
                const v3_val = try self.registers.read(v3);
                const v3_pre = v3_val;
                const xor_result: u64 = v2_val ^ v3_val;
                const xor_instr = buildXORInstr(v3, v2, v3);
                try self.lookup_trace.recordXorVirtual(
                    @intCast(self.state.cycle),
                    self.state.pc,
                    xor_instr,
                    v2_val,
                    v3_val,
                    true,
                    false,
                    self.is_compressed,
                );
                try self.registers.write(v3, xor_result);
                try self.trace.addStep(.{
                    .cycle = self.state.cycle,
                    .pc = self.state.pc,
                    .unexpanded_pc = self.state.pc,
                    .instruction = xor_instr,
                    .rs1_value = v2_val,
                    .rs2_value = v3_val,
                    .rd_pre_value = v3_pre,
                    .rd_value = xor_result,
                    .rd_index = v3,
                    .rs1_index = v2,
                    .rs2_index = v3,
                    .rd_written = true,
                    .rs1_read = true,
                    .rs2_read = true,
                    .memory_addr = null,
                    .memory_pre_value = null,
                    .memory_value = null,
                    .is_memory_write = false,
                    .next_pc = self.state.pc + pc_increment,
                    .is_compressed = false,
                    .virtual_sequence_remaining = total_steps - 1 - step_idx,
                });
                self.state.cycle += 1;
                self.registers.tick();
                step_idx += 1;
            }

            // AND(v3, v3, v0)
            {
                const v3_val = try self.registers.read(v3);
                const v0_val = try self.registers.read(v0);
                const v3_pre = v3_val;
                const and_result: u64 = v3_val & v0_val;
                const and_instr = buildANDInstr(v3, v3, v0);
                const and_decoded = zkvm.instruction.DecodedInstruction.decode(and_instr);
                try self.lookup_trace.recordInstruction(
                    @intCast(self.state.cycle),
                    self.state.pc,
                    and_instr,
                    and_decoded,
                    v3_val,
                    v0_val,
                );
                try self.registers.write(v3, and_result);
                try self.trace.addStep(.{
                    .cycle = self.state.cycle,
                    .pc = self.state.pc,
                    .unexpanded_pc = self.state.pc,
                    .instruction = and_instr,
                    .rs1_value = v3_val,
                    .rs2_value = v0_val,
                    .rd_pre_value = v3_pre,
                    .rd_value = and_result,
                    .rd_index = v3,
                    .rs1_index = v3,
                    .rs2_index = v0,
                    .rd_written = true,
                    .rs1_read = true,
                    .rs2_read = true,
                    .memory_addr = null,
                    .memory_pre_value = null,
                    .memory_value = null,
                    .is_memory_write = false,
                    .next_pc = self.state.pc + pc_increment,
                    .is_compressed = false,
                    .virtual_sequence_remaining = total_steps - 1 - step_idx,
                });
                self.state.cycle += 1;
                self.registers.tick();
                step_idx += 1;
            }

            // XOR(v2, v2, v3)
            {
                const v2_val = try self.registers.read(v2);
                const v3_val = try self.registers.read(v3);
                const v2_pre = v2_val;
                const new_dw: u64 = v2_val ^ v3_val;
                const xor_instr2 = buildXORInstr(v2, v2, v3);
                try self.lookup_trace.recordXorVirtual(
                    @intCast(self.state.cycle),
                    self.state.pc,
                    xor_instr2,
                    v2_val,
                    v3_val,
                    true,
                    false,
                    self.is_compressed,
                );
                try self.registers.write(v2, new_dw);
                try self.trace.addStep(.{
                    .cycle = self.state.cycle,
                    .pc = self.state.pc,
                    .unexpanded_pc = self.state.pc,
                    .instruction = xor_instr2,
                    .rs1_value = v2_val,
                    .rs2_value = v3_val,
                    .rd_pre_value = v2_pre,
                    .rd_value = new_dw,
                    .rd_index = v2,
                    .rs1_index = v2,
                    .rs2_index = v3,
                    .rd_written = true,
                    .rs1_read = true,
                    .rs2_read = true,
                    .memory_addr = null,
                    .memory_pre_value = null,
                    .memory_value = null,
                    .is_memory_write = false,
                    .next_pc = self.state.pc + pc_increment,
                    .is_compressed = false,
                    .virtual_sequence_remaining = total_steps - 1 - step_idx,
                });
                self.state.cycle += 1;
                self.registers.tick();
                step_idx += 1;
            }

            // SD(v1, v2, 0) → store new doubleword
            {
                const v1_val = try self.registers.read(v1);
                const v2_val = try self.registers.read(v2);
                const sd_instr = buildSDInstr(v1, v2);
                try self.ram.write(aligned_addr, v2_val, self.state.cycle);
                const sd_decoded = zkvm.instruction.DecodedInstruction.decode(sd_instr);
                try self.lookup_trace.recordInstruction(
                    @intCast(self.state.cycle),
                    self.state.pc,
                    sd_instr,
                    sd_decoded,
                    v1_val,
                    v2_val,
                );
                try self.trace.addStep(.{
                    .cycle = self.state.cycle,
                    .pc = self.state.pc,
                    .unexpanded_pc = self.state.pc,
                    .instruction = sd_instr,
                    .rs1_value = v1_val,
                    .rs2_value = v2_val,
                    .rd_pre_value = 0,
                    .rd_value = 0,
                    .rd_index = 0,
                    .rs1_index = v1,
                    .rs2_index = v2,
                    .rd_written = false,
                    .rs1_read = true,
                    .rs2_read = true,
                    .memory_addr = aligned_addr,
                    .memory_pre_value = old_doubleword,
                    .memory_value = v2_val,
                    .is_memory_write = true,
                    .next_pc = self.state.pc + pc_increment,
                    .is_compressed = self.is_compressed,
                    .virtual_sequence_remaining = 0,
                    .is_last_in_sequence = true,
                });
            }
        }

        // Also perform the actual byte/halfword/word store through the emulator's
        // memory for byte-level I/O correctness.
        // For proving, the doubleword SD in the virtual sequence is what matters.

        self.prev_pc = self.state.pc;
        self.state.pc = self.state.pc + pc_increment;
        self.state.cycle += 1;
        self.registers.tick();
        return true;
    }

    /// Build a synthetic OR instruction word: OR rd, rs1, rs2
    /// opcode=0x33, funct7=0, funct3=6
    fn buildORInstr(rd: u8, rs1: u8, rs2: u8) u32 {
        return (0x00 << 25) | // funct7 = 0
            (@as(u32, rs2 & 0x1F) << 20) |
            (@as(u32, rs1 & 0x1F) << 15) |
            (0x06 << 12) | // funct3 = 6 (OR)
            (@as(u32, rd & 0x1F) << 7) |
            0x33; // opcode = OP
    }

    /// Build a synthetic JALR instruction word: JALR rd, rs1, 0
    /// opcode=0x67, funct3=0, imm=0
    fn buildJALRInstr(rd: u8, rs1: u8) u32 {
        return (0 << 20) | // imm = 0
            (@as(u32, rs1 & 0x1F) << 15) |
            (0 << 12) | // funct3 = 0
            (@as(u32, rd & 0x1F) << 7) |
            0x67; // opcode = JALR
    }

    /// CSR address to virtual register mapping (imported from preprocessing)
    fn csrToVirtualReg(csr_addr: u12) u8 {
        const bytecode_preproc = @import("../zkvm/bytecode_preprocessing.zig");
        return bytecode_preproc.csrToVirtualReg(csr_addr);
    }

    /// Execute CSRRW as a virtual sequence of ADDI instructions.
    /// CSRRW rd, csr, rs1:
    ///   If rd==0: ADDI virtual_reg, rs1, 0 (1 step)
    ///   If rd==rs1: ADDI temp, rs1, 0; ADDI rd, virtual_reg, 0; ADDI virtual_reg, temp, 0 (3 steps)
    ///   Else: ADDI rd, virtual_reg, 0; ADDI virtual_reg, rs1, 0 (2 steps)
    fn stepCSRRW(
        self: *Emulator,
        instruction: u32,
        decoded: zkvm.instruction.DecodedInstruction,
    ) !bool {
        _ = instruction;
        const pc_increment: u64 = if (self.is_compressed) 2 else 4;
        const csr_addr: u12 = @truncate((@as(u32, @bitCast(decoded.imm)) >> 0) & 0xFFF);
        const virtual_reg = csrToVirtualReg(csr_addr);
        const temp_reg: u8 = 40;
        const rd = decoded.rd;
        const rs1 = decoded.rs1;

        // Read source values
        const rs1_value = try self.registers.read(rs1);
        const vr_value = try self.registers.read(virtual_reg);

        if (rd == 0) {
            // csrw pseudo: 1 step - ADDI virtual_reg, rs1, 0
            const rd_pre = try self.registers.read(virtual_reg);
            try self.registers.write(virtual_reg, rs1_value);
            const synth_instr = buildADDIInstr(virtual_reg, rs1, 0);
            try self.trace.addStep(.{
                .cycle = self.state.cycle,
                .pc = self.state.pc,
                .unexpanded_pc = self.state.pc,
                .instruction = synth_instr,
                .rs1_value = rs1_value,
                .rs2_value = 0,
                .rd_pre_value = rd_pre,
                .rd_value = rs1_value,
                .rd_index = virtual_reg,
                .rs1_index = rs1,
                .rs2_index = 0,
                .rd_written = true,
                .rs1_read = true,
                .rs2_read = false,
                .memory_addr = null,
                .memory_pre_value = null,
                .memory_value = null,
                .is_memory_write = false,
                .next_pc = self.state.pc + pc_increment,
                .is_compressed = self.is_compressed,
                .virtual_sequence_remaining = 0,
                .is_first_in_sequence = true,
                .is_last_in_sequence = true,
            });
            self.prev_pc = self.state.pc;
            self.state.pc += pc_increment;
            self.state.cycle += 1;
            self.registers.tick();
        } else if (rd == rs1) {
            // rd == rs1: 3 steps with temp
            // Step 1: ADDI temp, rs1, 0
            const temp_pre = try self.registers.read(temp_reg);
            try self.registers.write(temp_reg, rs1_value);
            try self.trace.addStep(.{
                .cycle = self.state.cycle,
                .pc = self.state.pc,
                .unexpanded_pc = self.state.pc,
                .instruction = buildADDIInstr(temp_reg, rs1, 0),
                .rs1_value = rs1_value,
                .rs2_value = 0,
                .rd_pre_value = temp_pre,
                .rd_value = rs1_value,
                .rd_index = temp_reg,
                .rs1_index = rs1,
                .rs2_index = 0,
                .rd_written = true,
                .rs1_read = true,
                .rs2_read = false,
                .memory_addr = null,
                .memory_pre_value = null,
                .memory_value = null,
                .is_memory_write = false,
                .next_pc = self.state.pc + pc_increment,
                .is_compressed = false,
                .virtual_sequence_remaining = 2,
                .is_first_in_sequence = true,
            });
            self.state.cycle += 1;
            self.registers.tick();

            // Step 2: ADDI rd, virtual_reg, 0
            const rd_pre = try self.registers.read(rd);
            try self.registers.write(rd, vr_value);
            try self.trace.addStep(.{
                .cycle = self.state.cycle,
                .pc = self.state.pc,
                .unexpanded_pc = self.state.pc,
                .instruction = buildADDIInstr(rd, virtual_reg, 0),
                .rs1_value = vr_value,
                .rs2_value = 0,
                .rd_pre_value = rd_pre,
                .rd_value = vr_value,
                .rd_index = rd,
                .rs1_index = virtual_reg,
                .rs2_index = 0,
                .rd_written = true,
                .rs1_read = true,
                .rs2_read = false,
                .memory_addr = null,
                .memory_pre_value = null,
                .memory_value = null,
                .is_memory_write = false,
                .next_pc = self.state.pc + pc_increment,
                .is_compressed = false,
                .virtual_sequence_remaining = 1,
                .is_first_in_sequence = false,
            });
            self.state.cycle += 1;
            self.registers.tick();

            // Step 3: ADDI virtual_reg, temp, 0
            const temp_val = try self.registers.read(temp_reg);
            const vr_pre = try self.registers.read(virtual_reg);
            try self.registers.write(virtual_reg, temp_val);
            try self.trace.addStep(.{
                .cycle = self.state.cycle,
                .pc = self.state.pc,
                .unexpanded_pc = self.state.pc,
                .instruction = buildADDIInstr(virtual_reg, temp_reg, 0),
                .rs1_value = temp_val,
                .rs2_value = 0,
                .rd_pre_value = vr_pre,
                .rd_value = temp_val,
                .rd_index = virtual_reg,
                .rs1_index = temp_reg,
                .rs2_index = 0,
                .rd_written = true,
                .rs1_read = true,
                .rs2_read = false,
                .memory_addr = null,
                .memory_pre_value = null,
                .memory_value = null,
                .is_memory_write = false,
                .next_pc = self.state.pc + pc_increment,
                .is_compressed = self.is_compressed,
                .virtual_sequence_remaining = 0,
                .is_first_in_sequence = false,
                .is_last_in_sequence = true,
            });
            self.prev_pc = self.state.pc;
            self.state.pc += pc_increment;
            self.state.cycle += 1;
            self.registers.tick();
        } else {
            // rd != rs1, rd != 0: 2 steps
            // Step 1: ADDI rd, virtual_reg, 0
            const rd_pre = try self.registers.read(rd);
            try self.registers.write(rd, vr_value);
            try self.trace.addStep(.{
                .cycle = self.state.cycle,
                .pc = self.state.pc,
                .unexpanded_pc = self.state.pc,
                .instruction = buildADDIInstr(rd, virtual_reg, 0),
                .rs1_value = vr_value,
                .rs2_value = 0,
                .rd_pre_value = rd_pre,
                .rd_value = vr_value,
                .rd_index = rd,
                .rs1_index = virtual_reg,
                .rs2_index = 0,
                .rd_written = true,
                .rs1_read = true,
                .rs2_read = false,
                .memory_addr = null,
                .memory_pre_value = null,
                .memory_value = null,
                .is_memory_write = false,
                .next_pc = self.state.pc + pc_increment,
                .is_compressed = false,
                .virtual_sequence_remaining = 1,
                .is_first_in_sequence = true,
            });
            self.state.cycle += 1;
            self.registers.tick();

            // Step 2: ADDI virtual_reg, rs1, 0
            const vr_pre = try self.registers.read(virtual_reg);
            try self.registers.write(virtual_reg, rs1_value);
            try self.trace.addStep(.{
                .cycle = self.state.cycle,
                .pc = self.state.pc,
                .unexpanded_pc = self.state.pc,
                .instruction = buildADDIInstr(virtual_reg, rs1, 0),
                .rs1_value = rs1_value,
                .rs2_value = 0,
                .rd_pre_value = vr_pre,
                .rd_value = rs1_value,
                .rd_index = virtual_reg,
                .rs1_index = rs1,
                .rs2_index = 0,
                .rd_written = true,
                .rs1_read = true,
                .rs2_read = false,
                .memory_addr = null,
                .memory_pre_value = null,
                .memory_value = null,
                .is_memory_write = false,
                .next_pc = self.state.pc + pc_increment,
                .is_compressed = self.is_compressed,
                .virtual_sequence_remaining = 0,
                .is_first_in_sequence = false,
                .is_last_in_sequence = true,
            });
            self.prev_pc = self.state.pc;
            self.state.pc += pc_increment;
            self.state.cycle += 1;
            self.registers.tick();
        }

        return true;
    }

    /// Execute CSRRS as a virtual sequence of ADDI/OR instructions.
    /// CSRRS rd, csr, rs1:
    ///   If rs1==0: ADDI rd, virtual_reg, 0 (1 step)
    ///   If rd==0: OR virtual_reg, virtual_reg, rs1 (1 step)
    ///   If rd==rs1: ADDI temp, rs1, 0; ADDI rd, virtual_reg, 0; OR virtual_reg, virtual_reg, temp (3 steps)
    ///   Else: ADDI rd, virtual_reg, 0; OR virtual_reg, virtual_reg, rs1 (2 steps)
    fn stepCSRRS(
        self: *Emulator,
        instruction: u32,
        decoded: zkvm.instruction.DecodedInstruction,
    ) !bool {
        _ = instruction;
        const pc_increment: u64 = if (self.is_compressed) 2 else 4;
        const csr_addr: u12 = @truncate((@as(u32, @bitCast(decoded.imm)) >> 0) & 0xFFF);
        const virtual_reg = csrToVirtualReg(csr_addr);
        const temp_reg: u8 = 40;
        const rd = decoded.rd;
        const rs1 = decoded.rs1;

        const rs1_value = try self.registers.read(rs1);
        const vr_value = try self.registers.read(virtual_reg);

        if (rs1 == 0) {
            // csrr pseudo: 1 step - ADDI rd, virtual_reg, 0
            const rd_pre = try self.registers.read(rd);
            const result_val = vr_value;
            if (rd != 0) try self.registers.write(rd, result_val);
            try self.trace.addStep(.{
                .cycle = self.state.cycle,
                .pc = self.state.pc,
                .unexpanded_pc = self.state.pc,
                .instruction = buildADDIInstr(rd, virtual_reg, 0),
                .rs1_value = vr_value,
                .rs2_value = 0,
                .rd_pre_value = rd_pre,
                .rd_value = if (rd == 0) 0 else result_val,
                .rd_index = rd,
                .rs1_index = virtual_reg,
                .rs2_index = 0,
                .rd_written = true,
                .rs1_read = true,
                .rs2_read = false,
                .memory_addr = null,
                .memory_pre_value = null,
                .memory_value = null,
                .is_memory_write = false,
                .next_pc = self.state.pc + pc_increment,
                .is_compressed = self.is_compressed,
                .virtual_sequence_remaining = 0,
                .is_first_in_sequence = true,
                .is_last_in_sequence = true,
            });
            self.prev_pc = self.state.pc;
            self.state.pc += pc_increment;
            self.state.cycle += 1;
            self.registers.tick();
        } else if (rd == 0) {
            // csrs pseudo: 1 step - OR virtual_reg, virtual_reg, rs1
            const vr_pre = vr_value;
            const result_val = vr_value | rs1_value;
            try self.registers.write(virtual_reg, result_val);
            try self.trace.addStep(.{
                .cycle = self.state.cycle,
                .pc = self.state.pc,
                .unexpanded_pc = self.state.pc,
                .instruction = buildORInstr(virtual_reg, virtual_reg, rs1),
                .rs1_value = vr_value,
                .rs2_value = rs1_value,
                .rd_pre_value = vr_pre,
                .rd_value = result_val,
                .rd_index = virtual_reg,
                .rs1_index = virtual_reg,
                .rs2_index = rs1,
                .rd_written = true,
                .rs1_read = true,
                .rs2_read = true,
                .memory_addr = null,
                .memory_pre_value = null,
                .memory_value = null,
                .is_memory_write = false,
                .next_pc = self.state.pc + pc_increment,
                .is_compressed = self.is_compressed,
                .virtual_sequence_remaining = 0,
                .is_first_in_sequence = true,
                .is_last_in_sequence = true,
            });
            self.prev_pc = self.state.pc;
            self.state.pc += pc_increment;
            self.state.cycle += 1;
            self.registers.tick();
        } else if (rd == rs1) {
            // rd == rs1: 3 steps with temp
            // Step 1: ADDI temp, rs1, 0
            const temp_pre = try self.registers.read(temp_reg);
            try self.registers.write(temp_reg, rs1_value);
            try self.trace.addStep(.{
                .cycle = self.state.cycle,
                .pc = self.state.pc,
                .unexpanded_pc = self.state.pc,
                .instruction = buildADDIInstr(temp_reg, rs1, 0),
                .rs1_value = rs1_value,
                .rs2_value = 0,
                .rd_pre_value = temp_pre,
                .rd_value = rs1_value,
                .rd_index = temp_reg,
                .rs1_index = rs1,
                .rs2_index = 0,
                .rd_written = true,
                .rs1_read = true,
                .rs2_read = false,
                .memory_addr = null,
                .memory_pre_value = null,
                .memory_value = null,
                .is_memory_write = false,
                .next_pc = self.state.pc + pc_increment,
                .is_compressed = false,
                .virtual_sequence_remaining = 2,
                .is_first_in_sequence = true,
            });
            self.state.cycle += 1;
            self.registers.tick();

            // Step 2: ADDI rd, virtual_reg, 0
            const rd_pre = try self.registers.read(rd);
            try self.registers.write(rd, vr_value);
            try self.trace.addStep(.{
                .cycle = self.state.cycle,
                .pc = self.state.pc,
                .unexpanded_pc = self.state.pc,
                .instruction = buildADDIInstr(rd, virtual_reg, 0),
                .rs1_value = vr_value,
                .rs2_value = 0,
                .rd_pre_value = rd_pre,
                .rd_value = vr_value,
                .rd_index = rd,
                .rs1_index = virtual_reg,
                .rs2_index = 0,
                .rd_written = true,
                .rs1_read = true,
                .rs2_read = false,
                .memory_addr = null,
                .memory_pre_value = null,
                .memory_value = null,
                .is_memory_write = false,
                .next_pc = self.state.pc + pc_increment,
                .is_compressed = false,
                .virtual_sequence_remaining = 1,
                .is_first_in_sequence = false,
            });
            self.state.cycle += 1;
            self.registers.tick();

            // Step 3: OR virtual_reg, virtual_reg, temp
            const temp_val = try self.registers.read(temp_reg);
            const vr_pre = try self.registers.read(virtual_reg);
            const or_result = vr_pre | temp_val;
            try self.registers.write(virtual_reg, or_result);
            try self.trace.addStep(.{
                .cycle = self.state.cycle,
                .pc = self.state.pc,
                .unexpanded_pc = self.state.pc,
                .instruction = buildORInstr(virtual_reg, virtual_reg, temp_reg),
                .rs1_value = vr_pre,
                .rs2_value = temp_val,
                .rd_pre_value = vr_pre,
                .rd_value = or_result,
                .rd_index = virtual_reg,
                .rs1_index = virtual_reg,
                .rs2_index = temp_reg,
                .rd_written = true,
                .rs1_read = true,
                .rs2_read = true,
                .memory_addr = null,
                .memory_pre_value = null,
                .memory_value = null,
                .is_memory_write = false,
                .next_pc = self.state.pc + pc_increment,
                .is_compressed = self.is_compressed,
                .virtual_sequence_remaining = 0,
                .is_first_in_sequence = false,
                .is_last_in_sequence = true,
            });
            self.prev_pc = self.state.pc;
            self.state.pc += pc_increment;
            self.state.cycle += 1;
            self.registers.tick();
        } else {
            // rd != rs1, both nonzero: 2 steps
            // Step 1: ADDI rd, virtual_reg, 0
            const rd_pre = try self.registers.read(rd);
            try self.registers.write(rd, vr_value);
            try self.trace.addStep(.{
                .cycle = self.state.cycle,
                .pc = self.state.pc,
                .unexpanded_pc = self.state.pc,
                .instruction = buildADDIInstr(rd, virtual_reg, 0),
                .rs1_value = vr_value,
                .rs2_value = 0,
                .rd_pre_value = rd_pre,
                .rd_value = vr_value,
                .rd_index = rd,
                .rs1_index = virtual_reg,
                .rs2_index = 0,
                .rd_written = true,
                .rs1_read = true,
                .rs2_read = false,
                .memory_addr = null,
                .memory_pre_value = null,
                .memory_value = null,
                .is_memory_write = false,
                .next_pc = self.state.pc + pc_increment,
                .is_compressed = false,
                .virtual_sequence_remaining = 1,
                .is_first_in_sequence = true,
            });
            self.state.cycle += 1;
            self.registers.tick();

            // Step 2: OR virtual_reg, virtual_reg, rs1
            const vr_pre = try self.registers.read(virtual_reg);
            const or_result = vr_pre | rs1_value;
            try self.registers.write(virtual_reg, or_result);
            try self.trace.addStep(.{
                .cycle = self.state.cycle,
                .pc = self.state.pc,
                .unexpanded_pc = self.state.pc,
                .instruction = buildORInstr(virtual_reg, virtual_reg, rs1),
                .rs1_value = vr_pre,
                .rs2_value = rs1_value,
                .rd_pre_value = vr_pre,
                .rd_value = or_result,
                .rd_index = virtual_reg,
                .rs1_index = virtual_reg,
                .rs2_index = rs1,
                .rd_written = true,
                .rs1_read = true,
                .rs2_read = true,
                .memory_addr = null,
                .memory_pre_value = null,
                .memory_value = null,
                .is_memory_write = false,
                .next_pc = self.state.pc + pc_increment,
                .is_compressed = self.is_compressed,
                .virtual_sequence_remaining = 0,
                .is_first_in_sequence = false,
                .is_last_in_sequence = true,
            });
            self.prev_pc = self.state.pc;
            self.state.pc += pc_increment;
            self.state.cycle += 1;
            self.registers.tick();
        }

        return true;
    }

    /// Execute MRET as JALR v40, mepc(vr36), 0
    /// Jumps to the address stored in mepc virtual register.
    fn stepMRET(
        self: *Emulator,
        instruction: u32,
        decoded: zkvm.instruction.DecodedInstruction,
    ) !bool {
        _ = instruction;
        _ = decoded;
        const pc_increment: u64 = if (self.is_compressed) 2 else 4;
        const mepc_reg: u8 = 36;
        const temp_reg: u8 = 40; // v40 for return address (like JAL/JALR with rd=0)

        const mepc_value = try self.registers.read(mepc_reg);
        const temp_pre = try self.registers.read(temp_reg);

        // JALR semantics: rd = PC + 4, PC = (rs1 + imm) & ~1
        const link_addr = self.state.pc + pc_increment;
        const target = mepc_value & ~@as(u64, 1);

        try self.registers.write(temp_reg, link_addr);

        const synth_instr = buildJALRInstr(temp_reg, mepc_reg);
        try self.trace.addStep(.{
            .cycle = self.state.cycle,
            .pc = self.state.pc,
            .unexpanded_pc = self.state.pc,
            .instruction = synth_instr,
            .rs1_value = mepc_value,
            .rs2_value = 0,
            .rd_pre_value = temp_pre,
            .rd_value = link_addr,
            .rd_index = temp_reg,
            .rs1_index = mepc_reg,
            .rs2_index = 0,
            .rd_written = true,
            .rs1_read = true,
            .rs2_read = false,
            .memory_addr = null,
            .memory_pre_value = null,
            .memory_value = null,
            .is_memory_write = false,
            .next_pc = target,
            .is_compressed = self.is_compressed,
            .virtual_sequence_remaining = 0,
            .is_first_in_sequence = true,
            .is_last_in_sequence = true,
        });

        self.prev_pc = self.state.pc;
        self.state.pc = target;
        self.state.cycle += 1;
        self.registers.tick();

        return true;
    }

    /// Run until completion.
    /// Stops on ECALL (normal program termination) or infinite loop detection (PC stall).
    /// No cycle limit — trace length is validated at proving time against max_trace_length.
    pub fn run(self: *Emulator) !void {
        while (true) {
            const running = self.step() catch |err| switch (err) {
                error.Ecall => {
                    dbg("[TRACE] Terminated via ECALL at cycle {d}\n", .{self.state.cycle});
                    self.recordTerminationWrite() catch |term_err| {
                        dbg("[TRACE] Warning: failed to record termination write: {any}\n", .{term_err});
                    };
                    return;
                },
                else => return err,
            };
            if (!running) {
                dbg("[TRACE] Terminated via infinite loop at PC 0x{x}, cycle {d}\n", .{ self.state.pc, self.state.cycle });
                self.recordTerminationWrite() catch |term_err| {
                    dbg("[TRACE] Warning: failed to record termination write: {any}\n", .{term_err});
                };
                return;
            }
        }
    }

    /// Record a synthetic termination write to the termination address.
    ///
    /// This injects a sequence of 4 synthetic trace steps to properly write
    /// the termination bit (value 1) to the termination address:
    ///
    ///   Cycle N:   NoOp (satisfies previous j.'s NextIsNoop=1)
    ///   Cycle N+1: LUI x31, upper20(addr)    — sets x31 = addr & ~0xFFF
    ///   Cycle N+2: ADDI x30, x0, 1           — sets x30 = 1
    ///   Cycle N+3: SB x30, lower12(addr)(x31) — stores 1 at termination_addr
    ///
    /// Using REAL register values (x31 and x30) ensures consistency across ALL
    /// proof stages. Previously we used SB x0, 0(x0) with overridden values and
    /// virtual register slots k=32,33, but this broke the ValEvaluation identity
    /// (Stage 5) because val_poly entries at virtual slots had no corresponding
    /// write increments (no rd_wa, no inc).
    ///
    /// All synthetic steps have is_termination_store=true, which causes the R1CS
    /// witness generator to use createTerminationStoreWitness() adding:
    ///   FlagDoNotUpdateUnexpandedPC=1 (satisfies constraint 16: NextUPC = 0+4-4 = 0)
    ///   FlagIsNoop=1 (for Stage 2 product virtualization NextIsNoop factor)
    ///
    /// The LUI and ADDI steps have is_noop=false so Stage 4 processes their register
    /// writes properly (rd_wa and inc entries), ensuring the register polynomials
    /// are consistent with the R1CS witness values.
    fn recordTerminationWrite(self: *Emulator) !void {
        const termination_addr = self.device.memory_layout.termination;
        var cycle = self.state.cycle;

        // Get the current value at termination address (should be 0 for initial state)
        const pre_value = self.ram.memory.get(termination_addr) orelse 0;
        const post_value: u64 = 1;

        // Decompose termination_addr for instruction encoding
        // LUI loads upper 20 bits, SB immediate provides lower 12 bits
        const upper20: u32 = @truncate(termination_addr >> 12);
        const lower12: u32 = @truncate(termination_addr & 0xFFF);

        // Get current register values for x30 and x31 (pre-values for inc computation)
        const x31_pre = try self.registers.read(31);
        const x30_pre = try self.registers.read(30);

        // LUI result: sign-extend the 32-bit value on RV64
        // LUI places imm[31:12] in the destination, zeros bits [11:0]
        const lui_result: u64 = @bitCast(@as(i64, @as(i32, @bitCast(upper20 << 12))));

        // --- Step 1: NoOp (satisfies previous j.'s NextIsNoop=1) ---
        try self.trace.steps.append(self.allocator, TraceStep{
            .cycle = cycle,
            .pc = 0,
            .unexpanded_pc = 0,
            .instruction = 0,
            .rs1_value = 0,
            .rs2_value = 0,
            .rd_pre_value = 0,
            .rd_value = 0,
            .rd_index = 0,
            .rs1_index = 0,
            .rs2_index = 0,
            .rd_written = false, // NoOp writes nothing
            .rs1_read = false,
            .rs2_read = false,
            .memory_addr = null,
            .memory_pre_value = null,
            .memory_value = null,
            .is_memory_write = false,
            .next_pc = 0,
            .is_compressed = false,
            .is_noop = true,
            .is_termination_store = true, // DoNotUpdateUPC=1 in R1CS
        });
        cycle += 1;

        // --- Step 2: LUI x31, upper20(termination_addr) ---
        // Encoding: imm[31:12] | rd[11:7] | opcode[6:0]
        const lui_instr: u32 = (upper20 << 12) | (31 << 7) | 0x37;
        try self.trace.steps.append(self.allocator, TraceStep{
            .cycle = cycle,
            .pc = 0,
            .unexpanded_pc = 0,
            .instruction = lui_instr,
            .rs1_value = 0, // LUI doesn't read rs1
            .rs2_value = 0, // LUI doesn't read rs2
            .rd_pre_value = x31_pre,
            .rd_value = lui_result,
            .rd_index = 31, // LUI writes to x31
            .rs1_index = 0,
            .rs2_index = 0,
            .rd_written = true, // LUI writes rd
            .rs1_read = false, // LUI doesn't read rs1
            .rs2_read = false, // LUI doesn't read rs2
            .memory_addr = null,
            .memory_pre_value = null,
            .memory_value = null,
            .is_memory_write = false,
            .next_pc = 0,
            .is_compressed = false,
            .is_noop = false, // NOT noop — Stage 4 must process rd write
            .is_termination_store = true, // DoNotUpdateUPC=1, FlagIsNoop=1 in R1CS
            .virtual_sequence_remaining = 2, // 3-instruction sequence: LUI(2), ADDI(1), SB(0)
        });
        // Update register file so subsequent reads see the correct value
        try self.registers.write(31, lui_result);
        cycle += 1;

        // --- Step 3: ADDI x30, x0, 1 ---
        // Encoding: imm[11:0] | rs1[19:15] | funct3[14:12] | rd[11:7] | opcode[6:0]
        const addi_instr: u32 = (1 << 20) | (0 << 15) | (0 << 12) | (30 << 7) | 0x13;
        try self.trace.steps.append(self.allocator, TraceStep{
            .cycle = cycle,
            .pc = 0,
            .unexpanded_pc = 0,
            .instruction = addi_instr,
            .rs1_value = 0, // Reading x0 = 0
            .rs2_value = 0, // ADDI doesn't read rs2
            .rd_pre_value = x30_pre,
            .rd_value = 1, // x30 = 0 + 1 = 1
            .rd_index = 30, // ADDI writes to x30
            .rs1_index = 0, // ADDI reads x0
            .rs2_index = 0,
            .rd_written = true, // ADDI writes rd
            .rs1_read = true, // ADDI reads rs1 (x0)
            .rs2_read = false, // ADDI doesn't read rs2
            .memory_addr = null,
            .memory_pre_value = null,
            .memory_value = null,
            .is_memory_write = false,
            .next_pc = 0,
            .is_compressed = false,
            .is_noop = false, // NOT noop — Stage 4 must process rd write
            .is_termination_store = true, // DoNotUpdateUPC=1, FlagIsNoop=1 in R1CS
            .virtual_sequence_remaining = 1, // 3-instruction sequence: LUI(2), ADDI(1), SB(0)
        });
        try self.registers.write(30, 1);
        cycle += 1;

        // --- Step 4: SB x30, lower12(x31) ---
        // S-type encoding: imm[11:5] | rs2[24:20] | rs1[19:15] | funct3[14:12] | imm[4:0] | opcode[6:0]
        const imm_upper7: u32 = (lower12 >> 5) & 0x7F;
        const imm_lower5: u32 = lower12 & 0x1F;
        const sb_instr: u32 = (imm_upper7 << 25) | (30 << 20) | (31 << 15) | (0 << 12) | (imm_lower5 << 7) | 0x23;
        try self.trace.steps.append(self.allocator, TraceStep{
            .cycle = cycle,
            .pc = 0,
            .unexpanded_pc = 0,
            .instruction = sb_instr,
            .rs1_value = lui_result, // x31 = addr base
            .rs2_value = 1, // x30 = value to store
            .rd_pre_value = 0, // Stores don't write to rd
            .rd_value = 0,
            .rd_index = 0, // SB doesn't write to rd
            .rs1_index = 31, // SB reads x31 (addr base)
            .rs2_index = 30, // SB reads x30 (value to store)
            .rd_written = false, // SB doesn't write rd
            .rs1_read = true, // SB reads rs1
            .rs2_read = true, // SB reads rs2
            .memory_addr = termination_addr,
            .memory_pre_value = pre_value,
            .memory_value = post_value,
            .is_memory_write = true,
            .next_pc = 0,
            .is_compressed = false,
            .is_noop = false, // NOT noop — Stage 4 must process register reads
            .is_termination_store = true, // DoNotUpdateUPC=1, FlagIsNoop=1 in R1CS
            .virtual_sequence_remaining = 0, // 3-instruction sequence: LUI(2), ADDI(1), SB(0)
            .is_last_in_sequence = true,
        });

        // Record the write in the RAM trace at the SB cycle
        try self.ram.trace.recordWrite(termination_addr, pre_value, post_value, cycle);

        // Update memory state so final_ram includes the termination bit
        try self.ram.memory.put(self.allocator, termination_addr, post_value);

        cycle += 1;

        // --- Step 5: JAL x0, 0 (j . = infinite loop) ---
        // This matches vanilla Jolt's model where _start has `j .` after main returns.
        // The JAL has Jump=1 which disables constraint 16's condition (1-ShouldBranch-Jump=0),
        // and ShouldJump = Jump * (1-NextIsNoop) = 0 disables constraint 14.
        // This allows the SB anchor to use VI=true, DNUPC=false (matching vanilla Jolt).
        // UPC=4 satisfies the SB's constraint 16: NextUPC = 0+4-0 = 4.
        const jal_instr: u32 = 0x0000006F; // JAL x0, 0
        const FIRST_VIRTUAL_ALLOC_REG: u8 = 40; // 32 (RISCV) + 8 (reserved)
        try self.trace.steps.append(self.allocator, TraceStep{
            .cycle = cycle,
            .pc = 0,
            .unexpanded_pc = 4, // Synthetic UPC=4 to satisfy SB's constraint 16
            .instruction = jal_instr,
            .rs1_value = 0,
            .rs2_value = 0,
            .rd_pre_value = self.registers.registers[FIRST_VIRTUAL_ALLOC_REG],
            .rd_value = 8, // JAL x0 at UPC=4: link_address = 4+4 = 8 (must match R1CS constraint 13)
            .rd_index = FIRST_VIRTUAL_ALLOC_REG, // JAL x0 remapped to vr40 (upstream inline_sequence)
            .rs1_index = 0,
            .rs2_index = 0,
            .rd_written = true, // Jolt includes rd=0 writes in RdWa polynomial (0→0 transition)
            .rs1_read = false, // JAL doesn't read rs1
            .rs2_read = false,
            .memory_addr = null,
            .memory_pre_value = null,
            .memory_value = null,
            .is_memory_write = false,
            .next_pc = 0,
            .is_compressed = false,
            .is_noop = false,
            .is_termination_jal = true,
        });
        cycle += 1;

        self.state.cycle = cycle;

        dbg("[TRACE] Recorded termination sequence: 5 steps (noop + LUI x31 + ADDI x30 + SB + JAL), addr=0x{x:0>16}, cycles {}-{}\n", .{
            termination_addr,
            cycle - 5,
            cycle - 1,
        });
    }

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
                        // Read aligned 8-byte doubleword (1 trace entry, matching Jolt)
                        break :blk try self.readWordWithIO(addr);
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
                        // Write aligned 8-byte doubleword (1 trace entry, matching Jolt)
                        try self.writeWordWithIO(addr, rs2);
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

// Witness generation types - re-exported from witness.zig
pub const witness = @import("witness.zig");
pub const WitnessGenerator = witness.WitnessGenerator;
pub const MemoryTuple = witness.MemoryTuple;
pub const MemoryWitness = witness.MemoryWitness;

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
    try emu.setInputs(&[_]u8{50});

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
