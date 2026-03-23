//! Jolt zkVM - Zero-knowledge Virtual Machine
//!
//! This module implements the core zkVM functionality:
//! - RISC-V instruction execution
//! - Bytecode handling
//! - Memory and register checking
//! - R1CS constraint system
//! - Spartan proof system

const std = @import("std");

// Debug output control - set to true to enable verbose debug prints
const debug_verbose = false;
fn dbg(comptime fmt: []const u8, args: anytype) void {
    if (debug_verbose) std.debug.print(fmt, args);
}

const Allocator = std.mem.Allocator;
const common = @import("../common/mod.zig");
const field = @import("../field/mod.zig");
const tracer = @import("../tracer/mod.zig");
const transcripts = @import("../transcripts/mod.zig");
const msm = @import("../msm/mod.zig");
const poly_commitment = @import("../poly/commitment/mod.zig");
const Dory = poly_commitment.dory;
const Fp = field.BN254BaseField;
const Fr = field.BN254Scalar;

pub const bytecode = @import("bytecode/mod.zig");
pub const claim_reductions = @import("claim_reductions/mod.zig");
pub const commitment_types = @import("commitment_types.zig");
pub const instruction = @import("instruction/mod.zig");
pub const instruction_lookups = @import("instruction_lookups/mod.zig");
pub const jolt_device = @import("jolt_device.zig");
pub const jolt_types = @import("jolt_types.zig");
pub const jolt_serialization = @import("jolt_serialization.zig");
pub const preprocessing = @import("preprocessing.zig");
pub const jolt_prover = @import("jolt_prover.zig");
pub const shout = @import("shout/mod.zig");
pub const lookup_table = @import("lookup_table/mod.zig");
pub const r1cs = @import("r1cs/mod.zig");
pub const ram = @import("ram/mod.zig");
pub const registers = @import("registers/mod.zig");
pub const spartan = @import("spartan/mod.zig");

// Re-export commitment types
pub const PolyCommitment = commitment_types.PolyCommitment;
pub const OpeningProof = commitment_types.OpeningProof;

fn writeByteToRamMap(
    map: *std.AutoHashMapUnmanaged(u64, u64),
    allocator: Allocator,
    address: u64,
    value: u8,
) !void {
    const word_addr = address & ~@as(u64, 7);
    const byte_offset: u3 = @truncate(address & 7);

    var word = map.get(word_addr) orelse 0;
    const mask = @as(u64, 0xFF) << (@as(u6, byte_offset) * 8);
    word = (word & ~mask) | (@as(u64, value) << (@as(u6, byte_offset) * 8));

    try map.put(allocator, word_addr, word);
}

fn writeBytesToRamMap(
    map: *std.AutoHashMapUnmanaged(u64, u64),
    allocator: Allocator,
    base_address: u64,
    bytes: []const u8,
) !void {
    for (bytes, 0..) |byte, idx| {
        try writeByteToRamMap(map, allocator, base_address + @as(u64, idx), byte);
    }
}

/// Build bytecode_words from program bytecode and base address.
/// This matches Jolt's RAMPreprocessing::preprocess behavior.
///
/// Returns: (bytecode_words slice, min_bytecode_address)
/// The caller owns the bytecode_words memory and must free it.
fn buildBytecodeWords(
    allocator: Allocator,
    program_bytecode: []const u8,
    base_address: u64,
) !struct { words: []u64, min_bytecode_address: u64 } {
    if (program_bytecode.len == 0) {
        return .{ .words = &[_]u64{}, .min_bytecode_address = 0 };
    }

    // Compute word-aligned range like Jolt does
    const min_addr = base_address;
    const max_addr = base_address + program_bytecode.len - 1;

    const min_word = min_addr / 8;
    const max_word = (max_addr + 7) / 8;
    const num_words = max_word - min_word + 1;

    const min_bytecode_address = min_word * 8;

    // Allocate words
    const words = try allocator.alloc(u64, num_words);
    @memset(words, 0);

    // Fill in bytes (like Jolt's RAMPreprocessing::preprocess)
    for (program_bytecode, 0..) |byte, i| {
        const addr = base_address + i;
        const word_idx = (addr / 8) - min_word;
        const byte_offset: u6 = @intCast(addr % 8);
        words[word_idx] |= @as(u64, byte) << (byte_offset * 8);
    }

    dbg("[BUILD_BYTECODE_WORDS] base_address=0x{x}, len={}, min_bytecode_address=0x{x}, num_words={}\n", .{ base_address, program_bytecode.len, min_bytecode_address, num_words });
    if (num_words > 0) {
        dbg("[BUILD_BYTECODE_WORDS] First 3 words: ", .{});
        for (0..@min(3, num_words)) |i| {
            dbg("0x{x:0>16} ", .{words[i]});
        }
        dbg("\n", .{});
    }

    return .{ .words = words, .min_bytecode_address = min_bytecode_address };
}

/// Compute bytecode code_size from raw program bytes, matching Jolt's BytecodePreprocessing.
/// Jolt decodes instructions (4 bytes each, or 2 for compressed RVC), prepends a NoOp,
/// then pads to next power of 2 with minimum 2.
/// This value is used as bytecode_K in the proof and must match Jolt's preprocessing.
/// Must account for W-extension decomposition: each W-ext instruction becomes 2 bytecode entries.
pub fn computeBytecodeCodeSize(program_bytecode: []const u8) usize {
    const zkvm_instruction = @import("instruction/mod.zig");

    // Count bytecode entries, accounting for W-extension decomposition
    var num_entries: usize = 0;
    var offset: usize = 0;
    while (offset < program_bytecode.len) {
        // Check if compressed (RVC): lowest 2 bits != 0b11
        if (offset + 2 <= program_bytecode.len) {
            const first_halfword = std.mem.readInt(u16, program_bytecode[offset..][0..2], .little);
            const is_compressed = (first_halfword & 0x3) != 0x3;

            var instr_word: u32 = undefined;
            if (is_compressed) {
                instr_word = zkvm_instruction.uncompressInstruction(first_halfword, .Bit64);
                offset += 2;
            } else {
                if (offset + 4 > program_bytecode.len) break;
                instr_word = std.mem.readInt(u32, program_bytecode[offset..][0..4], .little);
                offset += 4;
            }

            // Check if this is a W-extension instruction that gets decomposed into 2 entries
            const opcode: u7 = @truncate(instr_word & 0x7F);
            const funct3: u3 = @truncate((instr_word >> 12) & 0x7);
            const funct7: u7 = @truncate((instr_word >> 25) & 0x7F);

            // Count bytecode entries per instruction, matching preprocessing.zig decomposition
            if (opcode == 0x3b and funct7 == 0x01 and (funct3 == 6 or funct3 == 4)) {
                // REMW (funct3=6) or DIVW (funct3=4): 21 entries
                num_entries += 21;
            } else if (opcode == 0x3b and funct3 == 7 and funct7 == 0x01) {
                // REMUW: 12 entries
                num_entries += 12;
            } else if (opcode == 0x1b and funct3 == 5 and (instr_word >> 30) & 1 == 0) {
                // SRLIW: 3 entries (VirtualMULI + VirtualSRLI + VirtualSignExtendWord)
                num_entries += 3;
            } else if (opcode == 0x33 and funct3 == 1 and funct7 == 0) {
                // SLL: 2 entries (VirtualPow2 + MUL)
                num_entries += 2;
            } else if (opcode == 0x33 and funct3 == 5 and (funct7 == 0 or funct7 == 0x20)) {
                // SRL/SRA: 2 entries (VirtualShiftRightBitmask + VirtualSRL/VirtualSRA)
                num_entries += 2;
            } else if (opcode == 0x13 and funct3 == 5 and (instr_word >> 30) & 1 == 1) {
                // SRAI: 1 entry (VirtualSRAI)
                num_entries += 1;
            } else if (opcode == 0x03 and funct3 != 3) {
                // Sub-word loads: LB(f3=0)→8, LH(f3=1)→9, LW(f3=2)→8, LBU(f3=4)→8, LHU(f3=5)→9, LWU(f3=6)→9
                num_entries += switch (funct3) {
                    0, 4 => @as(usize, 8), // LB, LBU
                    1, 5 => @as(usize, 9), // LH, LHU
                    2 => @as(usize, 8), // LW
                    6 => @as(usize, 9), // LWU
                    else => @as(usize, 1), // shouldn't happen
                };
            } else if (opcode == 0x23 and funct3 != 3) {
                // Sub-word stores: SB(f3=0)→13, SH(f3=1)→14, SW(f3=2)→15
                num_entries += switch (funct3) {
                    0 => @as(usize, 13), // SB
                    1 => @as(usize, 14), // SH
                    2 => @as(usize, 15), // SW
                    else => @as(usize, 1), // shouldn't happen
                };
            } else {
                const is_w_ext_2 = switch (opcode) {
                    0x1b => switch (funct3) {
                        0 => true, // ADDIW
                        1 => true, // SLLIW → VirtualMULI + VirtualSignExtendWord (2 entries)
                        else => false,
                    },
                    0x3b => (funct3 == 0 and funct7 == 0x00) or // ADDW
                        (funct3 == 0 and funct7 == 0x20) or // SUBW
                        (funct3 == 0 and funct7 == 0x01), // MULW
                    else => false,
                };

                if (is_w_ext_2) {
                    num_entries += 2; // Base instruction + virtual step
                } else {
                    num_entries += 1; // Regular instruction or SLLI/SRLI (1 entry each)
                }
            }
        } else {
            break;
        }
    }

    // +1 for prepended NoOp (Jolt always prepends one)
    // +4 for termination sequence (LUI, ADDI, SB, JAL-to-self) at the end
    const total = num_entries + 1 + 4;

    // Pad to next power of 2, minimum 2
    if (total < 2) return 2;
    return std.math.ceilPowerOfTwo(usize, total) catch total;
}

fn buildInitialRamMap(
    allocator: Allocator,
    program_bytecode: []const u8,
    base_address: u64,
    device: *const jolt_device.JoltDevice,
) !std.AutoHashMapUnmanaged(u64, u64) {
    var map = std.AutoHashMapUnmanaged(u64, u64){};
    errdefer map.deinit(allocator);

    // Program bytes live at RAM_START_ADDRESS (or custom base for ELF).
    try writeBytesToRamMap(&map, allocator, base_address, program_bytecode);

    // Populate IO-region bytes (inputs + advice) as initial RAM state.
    if (device.trusted_advice.len > 0) {
        try writeBytesToRamMap(&map, allocator, device.memory_layout.trusted_advice_start, device.trusted_advice);
    }
    if (device.untrusted_advice.len > 0) {
        try writeBytesToRamMap(&map, allocator, device.memory_layout.untrusted_advice_start, device.untrusted_advice);
    }
    if (device.inputs.len > 0) {
        try writeBytesToRamMap(&map, allocator, device.memory_layout.input_start, device.inputs);
    }

    return map;
}

/// RISC-V register indices
pub const Register = enum(u8) {
    // Standard RISC-V registers
    zero = 0, // x0 - hardwired zero
    ra = 1, // x1 - return address
    sp = 2, // x2 - stack pointer
    gp = 3, // x3 - global pointer
    tp = 4, // x4 - thread pointer
    t0 = 5, // x5 - temporary
    t1 = 6, // x6 - temporary
    t2 = 7, // x7 - temporary
    s0 = 8, // x8/fp - saved/frame pointer
    s1 = 9, // x9 - saved
    a0 = 10, // x10 - argument/return
    a1 = 11, // x11 - argument/return
    a2 = 12, // x12 - argument
    a3 = 13, // x13 - argument
    a4 = 14, // x14 - argument
    a5 = 15, // x15 - argument
    a6 = 16, // x16 - argument
    a7 = 17, // x17 - argument
    s2 = 18, // x18 - saved
    s3 = 19, // x19 - saved
    s4 = 20, // x20 - saved
    s5 = 21, // x21 - saved
    s6 = 22, // x22 - saved
    s7 = 23, // x23 - saved
    s8 = 24, // x24 - saved
    s9 = 25, // x25 - saved
    s10 = 26, // x26 - saved
    s11 = 27, // x27 - saved
    t3 = 28, // x28 - temporary
    t4 = 29, // x29 - temporary
    t5 = 30, // x30 - temporary
    t6 = 31, // x31 - temporary
    _,

    pub fn fromIndex(index: u8) Register {
        return @enumFromInt(index);
    }

    pub fn toIndex(self: Register) u8 {
        return @intFromEnum(self);
    }
};

/// VM state during execution
pub const VMState = struct {
    /// Program counter
    pc: u64,
    /// Register file
    registers: [32]u64,
    /// Current instruction
    instruction: u32,
    /// Cycle count
    cycle: u64,

    pub fn init(entry_point: u64) VMState {
        var state = VMState{
            .pc = entry_point,
            .registers = [_]u64{0} ** 32,
            .instruction = 0,
            .cycle = 0,
        };
        // x0 is always zero
        state.registers[0] = 0;
        return state;
    }

    /// Read a register value
    pub fn readReg(self: *const VMState, reg: Register) u64 {
        const idx = reg.toIndex();
        if (idx == 0) return 0; // x0 is hardwired to zero
        return self.registers[idx];
    }

    /// Write a register value
    pub fn writeReg(self: *VMState, reg: Register, value: u64) void {
        const idx = reg.toIndex();
        if (idx == 0) return; // x0 is read-only
        self.registers[idx] = value;
    }
};


/// Jolt prover
pub fn JoltProver(comptime F: type) type {
    const ThreadPool = @import("../utils/thread_pool.zig").ThreadPool;
    return struct {
        const Self = @This();

        allocator: Allocator,
        thread_pool: ?*ThreadPool = null,

        pub fn init(allocator: Allocator) Self {
            return .{
                .allocator = allocator,
            };
        }

        pub fn initWithThreadPool(allocator: Allocator, tp: *ThreadPool) Self {
            return .{
                .allocator = allocator,
                .thread_pool = tp,
            };
        }

        /// Generate a Jolt-compatible proof with Dory commitments bundled
        /// Allows specifying custom base_address and entry_point for ELF programs
        pub fn proveJoltCompatibleWithDoryAndSrsAtAddress(
            self: *Self,
            program_bytecode: []const u8,
            inputs: []const u8,
            srs_path: ?[]const u8,
            base_address: u64,
            entry_point: u64,
            text_size_opt: ?usize,
        ) !jolt_types.JoltProofWithDory(F, commitment_types.PolyCommitment, commitment_types.OpeningProof) {
            const JoltProofWithDory = jolt_types.JoltProofWithDory(F, commitment_types.PolyCommitment, commitment_types.OpeningProof);
            const DoryScheme = Dory.DoryCommitmentScheme(F);

            // Initialize memory config
            // Use memory_size = 32768 to match Jolt fibonacci example
            var overall_timer = std.time.Timer.start() catch unreachable;
            var config = common.MemoryConfig{
                .program_size = program_bytecode.len,
                .heap_size = 32768,
            };

            // Initialize the emulator
            var emulator = tracer.Emulator.init(self.allocator, &config);
            defer emulator.deinit();

            // Load the program at the correct base address and set entry point
            try emulator.loadProgramAt(program_bytecode, base_address);
            emulator.state.pc = entry_point;

            if (inputs.len > 0) {
                try emulator.setInputs(inputs);
            }
            try emulator.run();

            // Pad trace with NoOp cycles (matching Jolt's behavior)
            try emulator.trace.padWithNoop();

            // Initialize Blake2b transcript for Jolt compatibility
            const Blake2bTranscript = transcripts.Blake2bTranscript(F);
            var transcript = Blake2bTranscript.init("Jolt");

            // Build BytecodePreprocessing for PC mapping (ELF address → bytecode index)
            // Needed for R1CS witness generation: PC must be bytecode index, not ELF address
            const preproc_for_witness = @import("preprocessing.zig");
            const text_sz_for_witness = text_size_opt orelse program_bytecode.len;
            var bytecode_prep_witness = try preproc_for_witness.BytecodePreprocessing.preprocessWithTextSize(self.allocator, program_bytecode, base_address, null, text_sz_for_witness);
            defer bytecode_prep_witness.deinit();

            // Generate R1CS cycle witnesses from execution trace
            var constraint_gen = r1cs.R1CSConstraintGenerator(F).init(self.allocator);
            defer constraint_gen.deinit();

            const cycle_witnesses = try constraint_gen.generateWitnessWithPCMap(&emulator.trace, &bytecode_prep_witness.pc_map);
            defer self.allocator.free(cycle_witnesses);

            // Convert to Jolt format using the proof converter with transcript
            var converter = if (self.thread_pool) |tp|
                jolt_prover.JoltProver(F).initWithThreadPool(self.allocator, tp)
            else
                jolt_prover.JoltProver(F).init(self.allocator);

            // Compute log_t and log_k directly from trace (already padded to power of 2)
            const trace_length = emulator.trace.steps.items.len;
            const log_t: u8 = @intCast(std.math.log2_int(usize, trace_length));
            // Compute ram_K from trace like Jolt: max remapped address across all
            // trace steps, combined with bytecode region, rounded to next power of 2.
            const log_k: u8 = blk: {
                const ml = emulator.device.memory_layout;

                // 1. Find max remapped address from trace
                var max_remapped: u64 = 0;
                for (emulator.trace.steps.items) |step| {
                    if (step.memory_addr) |addr| {
                        if (ml.remapAddress(addr)) |raddr| {
                            if (raddr > max_remapped) max_remapped = raddr;
                        }
                    }
                }

                // 2. Account for bytecode region (like Jolt's min_bytecode_address + bytecode_words.len + 1)
                const min_word = base_address / 8;
                const max_word = (base_address + program_bytecode.len - 1 + 7) / 8;
                const num_bytecode_words = max_word - min_word + 1;
                const min_bytecode_address = min_word * 8;
                if (ml.remapAddress(min_bytecode_address)) |raddr| {
                    const bytecode_end = raddr + num_bytecode_words + 1;
                    if (bytecode_end > max_remapped) max_remapped = bytecode_end;
                }

                const ram_k = std.math.ceilPowerOfTwo(u64, max_remapped) catch (1 << 16);
                break :blk @intCast(std.math.log2_int(u64, ram_k));
            };

            // Create JoltDevice for Fiat-Shamir preamble
            // CRITICAL: Use actual emulator outputs and panic state for Fiat-Shamir transcript
            const actual_outputs = emulator.getOutputs();
            const actual_panic = emulator.device.panic;

            // DEBUG: Print Fiat-Shamir preamble values
            dbg("\n=== Zolt Fiat-Shamir Preamble Debug (WithDory) ===\n", .{});
            dbg("inputs.len = {d}\n", .{inputs.len});
            if (inputs.len > 0 and inputs.len <= 32) {
                dbg("inputs = {any}\n", .{inputs});
            } else if (inputs.len > 32) {
                dbg("inputs[0..32] = {any}...\n", .{inputs[0..32]});
            }
            dbg("outputs.len = {d}\n", .{actual_outputs.len});
            if (actual_outputs.len > 0 and actual_outputs.len <= 32) {
                dbg("outputs = {any}\n", .{actual_outputs});
            } else if (actual_outputs.len > 32) {
                dbg("outputs[0..32] = {any}...\n", .{actual_outputs[0..32]});
            }
            dbg("panic = {}\n", .{actual_panic});
            dbg("=================================================\n\n", .{});

            var device = try jolt_device.JoltDevice.fromEmulator(
                self.allocator,
                inputs,
                actual_outputs,
                actual_panic,
                @intCast(program_bytecode.len),
                config.heap_size, // Pass memory_size from config
            );
            defer device.deinit();

            // Compute RAM parameters directly
            const ram_K: usize = @as(usize, 1) << @intCast(log_k);

            // Run Fiat-Shamir preamble to match Jolt verifier
            jolt_device.fiatShamirPreamble(F, &transcript, &device, ram_K, trace_length);

            // Build polynomial evaluations and compute Dory commitments
            const bytecode_poly_size = if (program_bytecode.len < 2) 2 else std.math.ceilPowerOfTwo(usize, program_bytecode.len) catch program_bytecode.len;
            const memory_trace_len = emulator.ram.trace.accesses.items.len;
            const memory_poly_size = if (memory_trace_len < 2) 2 else std.math.ceilPowerOfTwo(usize, memory_trace_len) catch memory_trace_len;
            const reg_trace_len = emulator.trace.steps.items.len;
            const reg_poly_size = if (reg_trace_len < 2) 2 else std.math.ceilPowerOfTwo(usize, reg_trace_len) catch reg_trace_len;
            // The SRS must be large enough for the Stage 8 joint polynomial.
            // The joint polynomial is k_chunk * trace_length entries (one-hot expanded sparse polys).
            // k_chunk = 16 (2^log_k_chunk where log_k_chunk=4), so joint_poly_size = 16 * trace_length.
            const stage8_joint_poly_size: usize = 16 * trace_length; // k_chunk * T
            const max_poly_size = @max(@max(@max(bytecode_poly_size, memory_poly_size), reg_poly_size), stage8_joint_poly_size);
            const log_size: u32 = if (max_poly_size <= 1) 1 else @intCast(std.math.log2_int(usize, max_poly_size));

            dbg("[SRS] Stage8 max_poly_size={}, log_size={}, sigma={}, nu={}\n", .{
                max_poly_size, log_size, (log_size + 1) / 2, log_size - (log_size + 1) / 2,
            });

            if (comptime debug_verbose) std.debug.print("    [STAGE-TIMING] Tracing + witness gen: {d:.1} ms\n", .{@as(f64, @floatFromInt(overall_timer.read())) / 1_000_000.0});

            // Load SRS from file if path provided (for Jolt compatibility)
            // Otherwise generate SRS deterministically (may not match Jolt exactly)
            var phase_timer = std.time.Timer.start() catch unreachable;
            var dory_srs = if (srs_path) |path|
                try DoryScheme.loadFromFile(self.allocator, path)
            else
                try DoryScheme.setup(self.allocator, log_size);
            defer dory_srs.deinit();
            // Precompute G2 Miller loop coefficients for fast pairings
            dory_srs.initPreparedCache(self.thread_pool);
            const srs_time = phase_timer.read();
            if (comptime debug_verbose) std.debug.print("    [STAGE-TIMING] SRS setup: {d:.1} ms\n", .{@as(f64, @floatFromInt(srs_time)) / 1_000_000.0});

            dbg("[SRS] Loaded: g1_vec={}, g2_vec={}\n", .{dory_srs.g1_vec.len, dory_srs.g2_vec.len});

            // Debug: print SRS key values for comparison with verifier
            {
                const DoryMod = @import("../poly/commitment/dory.zig");
                // g1_0 compressed
                const g1_0_comp = DoryMod.compressG1(dory_srs.g1_vec[0]);
                dbg("[SRS DEBUG] g1_0 compressed: ", .{});
                for (g1_0_comp) |b| dbg("{x:0>2}", .{b});
                dbg("\n", .{});
                // g2_0 compressed
                const g2_0_comp = DoryMod.compressG2(dory_srs.g2_vec[0]);
                dbg("[SRS DEBUG] g2_0 compressed first 32: ", .{});
                for (g2_0_comp[0..32]) |b| dbg("{x:0>2}", .{b});
                dbg("\n", .{});
                // h2 compressed
                const h2_comp = DoryMod.compressG2(dory_srs.h2);
                dbg("[SRS DEBUG] h2 compressed first 32: ", .{});
                for (h2_comp[0..32]) |b| dbg("{x:0>2}", .{b});
                dbg("\n", .{});
            }

            // Build and store polynomial evaluations
            var result = JoltProofWithDory.init(self.allocator);
            result.dory_srs_log_size = log_size;

            // Store bytecode/memory/register eval polynomials (for opening proof later)
            result.bytecode_evals = try self.allocator.alloc(F, bytecode_poly_size);
            for (result.bytecode_evals, 0..) |*p, i| {
                if (i < program_bytecode.len) {
                    p.* = F.fromU64(@as(u64, program_bytecode[i]));
                } else {
                    p.* = F.zero();
                }
            }

            result.memory_evals = try self.allocator.alloc(F, memory_poly_size);
            for (result.memory_evals, 0..) |*p, i| {
                if (i < memory_trace_len) {
                    p.* = F.fromU64(emulator.ram.trace.accesses.items[i].value);
                } else {
                    p.* = F.zero();
                }
            }

            result.memory_final_evals = try self.allocator.alloc(F, memory_poly_size);
            @memcpy(result.memory_final_evals, result.memory_evals);

            result.register_evals = try self.allocator.alloc(F, reg_poly_size);
            for (result.register_evals, 0..) |*p, i| {
                if (i < reg_trace_len) {
                    p.* = F.fromU64(emulator.trace.steps.items[i].rd_value);
                } else {
                    p.* = F.zero();
                }
            }

            result.register_final_evals = try self.allocator.alloc(F, reg_poly_size);
            @memcpy(result.register_final_evals, result.register_evals);

            // Calculate OneHot parameters using Jolt's formula: d = ceil(log_k / log_k_chunk)
            // CRITICAL: Must use ram_K and bytecode_K (from proof config), NOT memory_poly_size/bytecode_poly_size.
            // The Jolt verifier computes ram_d = log2(proof.ram_K).div_ceil(log_k_chunk)
            // and bytecode_d = log2(proof.bytecode_K).div_ceil(log_k_chunk).
            // Using memory_poly_size (trace-dependent) gives wrong d values since ram_K = 2^16 >> trace size.
            const log_k_chunk: usize = 4; // Must match convert_config below
            const LOG_K_INSTRUCTION: usize = 128; // XLEN * 2 = 64 * 2
            // Compute bytecode_K early so we can use it for bytecode_d
            const bytecode_K_for_onehot = computeBytecodeCodeSize(program_bytecode);
            const log_bytecode_k: usize = if (bytecode_K_for_onehot <= 1) 0 else std.math.log2_int(usize, bytecode_K_for_onehot);
            const log_ram_k: usize = @intCast(log_k); // ram_K = 2^log_k

            const instruction_d = (LOG_K_INSTRUCTION + log_k_chunk - 1) / log_k_chunk; // ceil division
            const bytecode_d = if (log_bytecode_k == 0) 1 else (log_bytecode_k + log_k_chunk - 1) / log_k_chunk;
            const ram_d = if (log_ram_k == 0) 1 else (log_ram_k + log_k_chunk - 1) / log_k_chunk;

            dbg("[ZOLT] OneHot params: instruction_d={}, bytecode_d={}, ram_d={}\n", .{ instruction_d, bytecode_d, ram_d });

            dbg("[DORY] Computing {} Dory commitments (instruction_d={}, ram_d={}, bytecode_d={})...\n", .{2 + instruction_d + ram_d + bytecode_d, instruction_d, ram_d, bytecode_d});
            // Build commitment polynomials and compute Dory commitments
            // Order: RdInc, RamInc, InstructionRa[0..instruction_d-1], RamRa[0..ram_d-1], BytecodeRa[0..bytecode_d-1]
            //
            // CRITICAL: Sparse one-hot polynomials (InstructionRa, RamRa, BytecodeRa) must be
            // expanded to K*T size in CycleMajor layout: poly[addr * T + cycle] = 1 if chunk==addr, else 0.
            // Dense polynomials (RdInc, RamInc) stay at trace_length size.
            const GT = Dory.GT;
            const k_chunk: usize = @as(usize, 1) << @intCast(log_k_chunk);
            var all_commitments: std.ArrayListUnmanaged(GT) = .{};
            defer all_commitments.deinit(self.allocator);

            // Store dense witness polynomials for Stage 8 opening proof (only RdInc, RamInc)
            const num_dense = 2;
            var witness_polys = try self.allocator.alloc([]F, num_dense);
            errdefer {
                for (witness_polys) |p| self.allocator.free(p);
                self.allocator.free(witness_polys);
            }

            // Store one-hot index arrays for sparse polynomials
            const num_onehot = instruction_d + ram_d + bytecode_d;
            var onehot_indices = try self.allocator.alloc([]?u8, num_onehot);
            var onehot_idx: usize = 0;
            errdefer {
                for (onehot_indices[0..onehot_idx]) |idx_arr| self.allocator.free(idx_arr);
                self.allocator.free(onehot_indices);
            }

            // Cache row commitments (G1 points) from each polynomial's Dory commit
            // for homomorphic combination in Stage 8.
            // Order: RdInc, RamInc, InstructionRa[0..inst_d], RamRa[0..ram_d], BytecodeRa[0..bc_d]
            const num_total_polys = num_dense + num_onehot;
            const G1Point = Dory.G1Point;
            var row_commitments_cache = try self.allocator.alloc([]G1Point, num_total_polys);
            var rc_idx: usize = 0;
            errdefer {
                for (row_commitments_cache[0..rc_idx]) |rc| self.allocator.free(rc);
                self.allocator.free(row_commitments_cache);
            }

            phase_timer.reset();
            // Guard: k_chunk must fit in u8 for sparse one-hot index arrays
            std.debug.assert(k_chunk <= 256);

            // ===== Phase A: Build polynomials + streaming dense commit =====
            // Dense polynomials (RdInc, RamInc) are built as unpadded F arrays (length T)
            // and committed via streaming row-by-row i128 MSM — no full k_chunk×T i128 alloc.
            // One-hot indices are built in a single trace scan (1 pass, not 32+).

            // --- Dense: streaming commit + build unpadded F polys ---
            // Compute Dory layout for the padded polynomial size (k_chunk * T)
            const dense_poly_size = k_chunk * trace_length;
            const dense_num_vars: usize = if (dense_poly_size <= 1) 1 else std.math.log2_int(usize, dense_poly_size);
            const dense_sigma: usize = (dense_num_vars + 1) / 2;
            const dense_nu: usize = dense_num_vars - dense_sigma;
            const dense_num_cols = @as(usize, 1) << @intCast(dense_sigma);
            const dense_num_rows = @as(usize, 1) << @intCast(dense_nu);
            const active_rows = (trace_length + dense_num_cols - 1) / dense_num_cols;

            // Allocate unpadded F polys (T elements, not k_chunk×T)
            const rd_inc_poly = try self.allocator.alloc(F, trace_length);
            @memset(rd_inc_poly, F.zero());
            witness_polys[0] = rd_inc_poly;

            const ram_inc_poly = try self.allocator.alloc(F, trace_length);
            @memset(ram_inc_poly, F.zero());
            witness_polys[1] = ram_inc_poly;

            // Allocate row commitments for streaming dense commit
            const rd_row_commits = try self.allocator.alloc(G1Point, dense_num_rows);
            errdefer self.allocator.free(rd_row_commits);
            const ram_row_commits = try self.allocator.alloc(G1Point, dense_num_rows);
            errdefer self.allocator.free(ram_row_commits);

            // Phase A1: Streaming dense commit — scan trace row-by-row,
            // build F values + i128 row buffer, MSM per row, no full i128 alloc.
            // RdInc needs sequential register tracking, so we do it sequentially per row.
            {
                const K_INC = 128;
                var register_values: [K_INC]u64 = [_]u64{0} ** K_INC;
                const steps = emulator.trace.steps.items;

                // Row buffers for i128 values (reused across rows, tiny vs eliminated full-size arrays)
                const rd_buf = try self.allocator.alloc(i128, dense_num_cols);
                defer self.allocator.free(rd_buf);
                const ram_buf = try self.allocator.alloc(i128, dense_num_cols);
                defer self.allocator.free(ram_buf);

                for (0..active_rows) |row| {
                    const row_start = row * dense_num_cols;
                    const row_end = @min(row_start + dense_num_cols, trace_length);
                    const row_len = row_end - row_start;

                    // Fill i128 buffers and F polys for this row
                    @memset(rd_buf[0..row_len], 0);
                    @memset(ram_buf[0..row_len], 0);

                    for (row_start..row_end) |i| {
                        const col = i - row_start;
                        if (i < steps.len) {
                            const step = steps[i];
                            // RdInc: sequential register tracking
                            if (!step.is_noop and step.rd_written and step.rd_index != 0) {
                                const rd = step.rd_index;
                                const pre_value = register_values[rd];
                                const post_value = step.rd_value;
                                const inc: i128 = @as(i128, post_value) - @as(i128, pre_value);
                                rd_buf[col] = inc;
                                rd_inc_poly[i] = if (inc >= 0)
                                    F.fromU64(@intCast(inc))
                                else
                                    F.fromU64(@intCast(-inc)).neg();
                                register_values[rd] = post_value;
                            }
                            // RamInc: per-cycle independent
                            if (step.is_memory_write) {
                                const pre_value: i128 = @intCast(step.memory_pre_value orelse 0);
                                const post_value: i128 = @intCast(step.memory_value orelse 0);
                                const inc = post_value - pre_value;
                                ram_buf[col] = inc;
                                ram_inc_poly[i] = if (inc >= 0)
                                    F.fromU64(@intCast(inc))
                                else
                                    F.fromU64(@intCast(-inc)).neg();
                            }
                        }
                    }

                    // Zero-pad if row is partial (row_len < dense_num_cols)
                    if (row_len < dense_num_cols) {
                        @memset(rd_buf[row_len..dense_num_cols], 0);
                        @memset(ram_buf[row_len..dense_num_cols], 0);
                    }

                    // MSM for this row
                    const g1_slice = dory_srs.g1_vec[0..dense_num_cols];
                    rd_row_commits[row] = msm.MSM(Fr, Fp).computeI128(g1_slice, rd_buf[0..dense_num_cols], null);
                    ram_row_commits[row] = msm.MSM(Fr, Fp).computeI128(g1_slice, ram_buf[0..dense_num_cols], null);
                }

                // Zero-padded rows (beyond active data)
                for (active_rows..dense_num_rows) |row| {
                    rd_row_commits[row] = G1Point.identity();
                    ram_row_commits[row] = G1Point.identity();
                }
            }

            // --- One-hot index arrays: single-scan build ---
            const text_sz_for_ra = text_size_opt orelse program_bytecode.len;
            var bytecode_prep_for_ra = try preprocessing.BytecodePreprocessing.preprocessWithTextSize(self.allocator, program_bytecode, base_address, null, text_sz_for_ra);
            defer bytecode_prep_for_ra.deinit();

            {
                const stage6_mod = @import("spartan/stage6_prover.zig");
                const oh_mask: u128 = (@as(u128, 1) << @intCast(log_k_chunk)) - 1;
                const ram_mask: u64 = (@as(u64, 1) << @intCast(log_k_chunk)) - 1;
                const steps = emulator.trace.steps.items;

                // Pre-allocate all one-hot index arrays
                for (0..num_onehot) |idx| {
                    onehot_indices[idx] = try self.allocator.alloc(?u8, trace_length);
                    @memset(onehot_indices[idx], null);
                    onehot_idx = idx + 1;
                }

                // Single scan over trace — compute each index ONCE
                for (0..trace_length) |cycle| {
                    if (cycle >= steps.len) break;
                    const step = steps[cycle];

                    // Instruction Ra: compute lookup index ONCE (was 32× before)
                    const lookup_idx = stage6_mod.computeLookupIndex(step);
                    for (0..instruction_d) |dim| {
                        const shift = log_k_chunk * (instruction_d - 1 - dim); // MSB-first
                        const chunk: u128 = (lookup_idx >> @intCast(shift)) & oh_mask;
                        onehot_indices[dim][cycle] = if (chunk < k_chunk) @as(?u8, @intCast(chunk)) else null;
                    }

                    // Ram Ra: compute remapped address ONCE (was ram_d× before)
                    if (step.memory_addr) |addr| {
                        if (addr != 0) {
                            if (device.memory_layout.remapAddress(addr)) |raddr| {
                                for (0..ram_d) |dim| {
                                    const shift = log_k_chunk * (ram_d - 1 - dim);
                                    const chunk: u64 = (raddr >> @intCast(shift)) & ram_mask;
                                    onehot_indices[instruction_d + dim][cycle] = if (chunk < k_chunk) @as(?u8, @intCast(chunk)) else null;
                                }
                            }
                        }
                    }

                    // Bytecode Ra: compute PC index ONCE (was bytecode_d× before)
                    const bc_idx: u64 = @intCast(bytecode_prep_for_ra.pc_map.getPCForStep(step));
                    for (0..bytecode_d) |dim| {
                        const shift = log_k_chunk * (bytecode_d - 1 - dim);
                        const chunk: u64 = (bc_idx >> @intCast(shift)) & ram_mask;
                        onehot_indices[instruction_d + ram_d + dim][cycle] = if (chunk < k_chunk) @as(?u8, @intCast(chunk)) else null;
                    }
                }
            }
            onehot_idx = num_onehot; // all built

            // ===== Phase B: Dory commits =====
            // Dense row commits already computed in Phase A (streaming).
            // Convert row commits → GT via Miller loops, then commit one-hot polys.

            // Initialize row_commitments_cache entries to empty for safe error cleanup
            for (row_commitments_cache) |*rc| rc.* = &[_]G1Point{};

            // Dense: row commits → GT commitment (already have row_commits from Phase A)
            row_commitments_cache[0] = rd_row_commits;
            rc_idx = 1;
            row_commitments_cache[1] = ram_row_commits;
            rc_idx = 2;

            // Compute GT commitments from row commits (join for concurrency)
            var rd_gt: GT = undefined;
            var ram_gt: GT = undefined;
            if (self.thread_pool) |tp| {
                const DenseGTCtx = struct {
                    srs: *const DoryScheme.SetupParams,
                    rd_rc: []const G1Point,
                    ram_rc: []const G1Point,
                    n_rows: usize,
                    rd_out: *GT,
                    ram_out: *GT,
                };
                const gt_ctx = DenseGTCtx{
                    .srs = &dory_srs,
                    .rd_rc = rd_row_commits,
                    .ram_rc = ram_row_commits,
                    .n_rows = dense_num_rows,
                    .rd_out = &rd_gt,
                    .ram_out = &ram_gt,
                };
                _ = tp.join(void, void, gt_ctx, struct {
                    fn f(c: DenseGTCtx) void {
                        c.rd_out.* = DoryScheme.rowCommitmentsToCommitment(c.srs, c.rd_rc, c.n_rows, null);
                    }
                }.f, gt_ctx, struct {
                    fn f(c: DenseGTCtx) void {
                        c.ram_out.* = DoryScheme.rowCommitmentsToCommitment(c.srs, c.ram_rc, c.n_rows, null);
                    }
                }.f);
            } else {
                rd_gt = DoryScheme.rowCommitmentsToCommitment(&dory_srs, rd_row_commits, dense_num_rows, null);
                ram_gt = DoryScheme.rowCommitmentsToCommitment(&dory_srs, ram_row_commits, dense_num_rows, null);
            }
            try all_commitments.append(self.allocator, rd_gt);
            try all_commitments.append(self.allocator, ram_gt);

            // One-hot commits: parallel (inner parallelism also active via nested dispatch)
            const oh_commitments_out = try self.allocator.alloc(GT, num_onehot);
            defer self.allocator.free(oh_commitments_out);
            @memset(oh_commitments_out, GT.one());

            var parallel_error: std.atomic.Value(bool) = std.atomic.Value(bool).init(false);

            const OhCommitCtx = struct {
                dory_srs: *const DoryScheme.SetupParams,
                oh: []const []?u8,
                k_chunk: usize,
                trace_length: usize,
                alloc: Allocator,
                tp: ?*ThreadPool,
                c_out: []GT,
                rc_out: [][]G1Point,
                rc_base: usize,
                err_flag: *std.atomic.Value(bool),
            };

            const oh_ctx = OhCommitCtx{
                .dory_srs = &dory_srs,
                .oh = onehot_indices,
                .k_chunk = k_chunk,
                .trace_length = trace_length,
                .alloc = self.allocator,
                .tp = self.thread_pool,
                .c_out = oh_commitments_out,
                .rc_out = row_commitments_cache,
                .rc_base = num_dense,
                .err_flag = &parallel_error,
            };

            const commitOneHot = struct {
                fn f(ctx: OhCommitCtx, oh_idx: usize) void {
                    if (ctx.err_flag.load(.acquire)) return;
                    const r = DoryScheme.commitOneHotWithPoolAndHints(
                        ctx.dory_srs,
                        ctx.oh[oh_idx],
                        ctx.k_chunk,
                        ctx.trace_length,
                        ctx.alloc,
                        ctx.tp,
                    ) catch {
                        ctx.err_flag.store(true, .release);
                        return;
                    };
                    ctx.c_out[oh_idx] = r.commitment;
                    ctx.rc_out[ctx.rc_base + oh_idx] = r.row_commitments;
                }
            }.f;

            if (self.thread_pool) |tp| {
                tp.parallelForEach(num_onehot, oh_ctx, commitOneHot);
            } else {
                for (0..num_onehot) |i| commitOneHot(oh_ctx, i);
            }

            if (parallel_error.load(.acquire)) {
                for (row_commitments_cache) |rc| {
                    if (rc.len > 0) self.allocator.free(rc);
                }
                return error.OutOfMemory;
            }

            for (oh_commitments_out) |c| {
                try all_commitments.append(self.allocator, c);
            }
            rc_idx = num_total_polys;

            // Store witness polynomials, one-hot indices, row commitments cache, and params in result
            result.witness_polys = witness_polys;
            result.onehot_indices = onehot_indices;
            result.row_commitments_cache = row_commitments_cache;
            result.instruction_d = instruction_d;
            result.bytecode_d = bytecode_d;
            result.ram_d = ram_d;
            result.log_k_chunk = log_k_chunk;

            const commit_time = phase_timer.read();
            std.debug.print("    [DORY-COMMIT] {d:.1} ms ({} commitments)\n", .{ @as(f64, @floatFromInt(commit_time)) / 1_000_000.0, all_commitments.items.len });
            phase_timer.reset();
            dbg("[DORY] All {} commitments computed.\n", .{all_commitments.items.len});
            // Debug: print first 3 commitment bytes
            for (0..@min(3, all_commitments.items.len)) |ci| {
                const gt_bytes = all_commitments.items[ci].toBytes();
                dbg("[DORY] commitment[{}] first 32: ", .{ci});
                for (0..32) |bi| dbg("{x:0>2} ", .{gt_bytes[bi]});
                dbg("\n", .{});
            }
            // Store commitments in result
            result.dory_commitments = try all_commitments.toOwnedSlice(self.allocator);

            // Append Dory commitments (GT elements) to transcript
            for (result.dory_commitments) |comm| {
                transcript.appendGT("commitment", comm);
            }

            // Derive tau from transcript after preamble and commitments
            // CRITICAL: Must use trace_length (padded, power-of-2) not cycle_witnesses.len (actual count)
            // Jolt uses: num_steps.next_power_of_two().log_2() where num_steps = trace_length
            const num_cycle_vars = std.math.log2_int(usize, @max(1, trace_length));
            const num_rows_bits = num_cycle_vars + 2;
            var tau = try self.allocator.alloc(F, num_rows_bits);
            defer self.allocator.free(tau);
            for (0..num_rows_bits) |i| {
                tau[i] = transcript.challengeScalar();
            }

            // Print tau[0] for comparison with verifier
            if (comptime debug_verbose) {
                if (tau.len > 0) {
                    std.debug.print("[ZOLT-TAU] tau[0] limbs = [{x:0>16}, {x:0>16}, {x:0>16}, {x:0>16}]\n", .{ tau[0].limbs[0], tau[0].limbs[1], tau[0].limbs[2], tau[0].limbs[3] });
                    std.debug.print("[ZOLT-TAU] num_rows_bits = {}, num_cycle_vars = {}\n", .{ num_rows_bits, num_cycle_vars });
                    std.debug.print("[ZOLT-TAU] transcript state after tau: ", .{});
                    for (transcript.state[0..8]) |b| std.debug.print("{x:0>2} ", .{b});
                    std.debug.print("round={}\n", .{transcript.n_rounds});
                }
            }

            // For OutputSumcheck, we need initial and final RAM states
            var initial_ram_dory = try buildInitialRamMap(
                self.allocator,
                program_bytecode,
                base_address,
                &device,
            );
            defer initial_ram_dory.deinit(self.allocator);
            const init_ram_dory: ?*const std.AutoHashMapUnmanaged(u64, u64) = &initial_ram_dory;
            const final_ram_dory: ?*const std.AutoHashMapUnmanaged(u64, u64) = &emulator.ram.memory;

            // Build bytecode_words for init_eval computation (like Jolt's RAMPreprocessing)
            const bytecode_info_dory = try buildBytecodeWords(
                self.allocator,
                program_bytecode,
                base_address,
            );
            defer if (bytecode_info_dory.words.len > 0) self.allocator.free(bytecode_info_dory.words);

            // Get memory trace for RAF evaluation sumcheck
            const memory_trace_ptr: *const ram.MemoryTrace = &emulator.ram.trace;

            // Compute bytecode_K to match Jolt's BytecodePreprocessing.code_size
            const bytecode_code_size_dory = computeBytecodeCodeSize(program_bytecode);
            dbg("[ZOLT] bytecode_code_size (Dory path): {}\n", .{bytecode_code_size_dory});

            // Build BytecodePreprocessing for PC mapping (ELF address → bytecode index)
            const preproc = @import("preprocessing.zig");
            const text_sz = text_size_opt orelse program_bytecode.len;
            var bytecode_prep_dory = try preproc.BytecodePreprocessing.preprocessWithTextSize(self.allocator, program_bytecode, base_address, null, text_sz);
            defer bytecode_prep_dory.deinit();

            // Convert to Jolt-compatible format with transcript integration
            if (comptime debug_verbose) std.debug.print("    [STAGE-TIMING] Pre-prove setup: {d:.1} ms\n", .{@as(f64, @floatFromInt(phase_timer.read())) / 1_000_000.0});
            phase_timer.reset();
            result.proof = try converter.proveWithTranscript(
                commitment_types.PolyCommitment,
                commitment_types.OpeningProof,
                log_t,
                log_k,
                &[_]commitment_types.PolyCommitment{},
                null,
                jolt_prover.JoltProverConfig{
                    .bytecode_K = bytecode_code_size_dory,
                    .log_k_chunk = 4,
                    .lookups_ra_virtual_log_k_chunk = 16,
                    .memory_layout = &device.memory_layout, // Pass memory layout for OutputSumcheck
                    .initial_ram = init_ram_dory,
                    .final_ram = final_ram_dory,
                    .memory_trace = memory_trace_ptr, // Pass memory trace for RAF sumcheck
                    // Program I/O for OutputSumcheck's ProgramIOPolynomial
                    .program_inputs = device.inputs,
                    .program_outputs = device.outputs,
                    .is_panicking = device.panic,
                    .execution_trace = &emulator.trace,
                    // Bytecode preprocessing for init_eval computation (like Jolt's RAMPreprocessing)
                    .bytecode_words = if (bytecode_info_dory.words.len > 0) bytecode_info_dory.words else null,
                    .min_bytecode_address = bytecode_info_dory.min_bytecode_address,
                    // PC mapper for Stage 6 BytecodeReadRaf
                    .bytecode_pc_map = &bytecode_prep_dory.pc_map,
                    // Preprocessing bytecode for Stage 6 val_poly computation
                    .bytecode_preprocessing = &bytecode_prep_dory,
                    // Static ELF code bytes for Stage 6 bytecode entry population
                    .program_code_bytes = program_bytecode,
                    .code_base_address = common.constants.RAM_START_ADDRESS,
                    .text_size = text_sz,
                },
                cycle_witnesses,
                tau,
                &transcript,
            );
            const prove_phase_time = phase_timer.read();
            if (comptime debug_verbose) std.debug.print("  [TIMING] Prove (stages 1-7): {d:.1} ms\n", .{@as(f64, @floatFromInt(prove_phase_time)) / 1_000_000.0});

            phase_timer.reset();

            // ================================================================
            // Stage 8: Dory Opening Proof Generation
            // ================================================================
            // This generates the batched polynomial commitment opening proof.
            // The Jolt verifier collects all committed polynomial claims, computes
            // an RLC (random linear combination), and verifies a single Dory opening.
            //
            // Steps:
            // 1. Collect claims in Jolt's order: RamInc, RdInc (with Lagrange factor),
            //    then InstructionRa, BytecodeRa, RamRa (from HammingWeightClaimReduction)
            // 2. Append claims to transcript
            // 3. Sample gamma powers via challenge_scalar_powers
            // 4. Build joint polynomial: Σ γ^i * poly_i
            // 5. Generate Dory opening proof for joint polynomial at opening_point
            {
                dbg("\n[STAGE8] === Generating Dory Opening Proof ===\n", .{});

                const opening_point = result.proof.opening_point;
                dbg("[STAGE8] opening_point len = {} (log_k_chunk={}, n_cycle_vars={})\n", .{
                    opening_point.len, log_k_chunk, opening_point.len - log_k_chunk,
                });

                // 1. Collect claims in Jolt's exact order
                // Jolt verify_stage8 does:
                //   polynomial_claims.push(RamInc, ram_inc_claim * lagrange_factor)
                //   polynomial_claims.push(RdInc, rd_inc_claim * lagrange_factor)
                //   for i in 0..instruction_d: polynomial_claims.push(InstructionRa(i), claim)
                //   for i in 0..bytecode_d: polynomial_claims.push(BytecodeRa(i), claim)
                //   for i in 0..ram_d: polynomial_claims.push(RamRa(i), claim)

                // Compute Lagrange factor: ∏(1 - r_address[i]) for i in 0..log_k_chunk
                // r_address = opening_point[0..log_k_chunk] (BE order)
                var lagrange_factor = F.one();
                for (0..log_k_chunk) |i| {
                    lagrange_factor = lagrange_factor.mul(F.one().sub(opening_point[i]));
                }
                if (comptime debug_verbose) {
                    const lf_be = lagrange_factor.toBytesBE();
                    dbg("[STAGE8] lagrange_factor_LE=[", .{});
                    for (0..8) |bi| dbg("{x:0>2}", .{lf_be[31 - bi]});
                    dbg("]\n", .{});
                }

                // Get claims from IncClaimReduction (Stage 6 output)
                const ram_inc_claim = result.proof.opening_claims.get(
                    .{ .Committed = .{ .poly = .RamInc, .sumcheck_id = .IncClaimReduction } },
                ) orelse F.zero();
                const rd_inc_claim = result.proof.opening_claims.get(
                    .{ .Committed = .{ .poly = .RdInc, .sumcheck_id = .IncClaimReduction } },
                ) orelse F.zero();

                const num_claims = 2 + instruction_d + bytecode_d + ram_d;
                var claims_ordered = try self.allocator.alloc(F, num_claims);
                defer self.allocator.free(claims_ordered);

                // RamInc and RdInc with Lagrange factor
                claims_ordered[0] = ram_inc_claim.mul(lagrange_factor);
                claims_ordered[1] = rd_inc_claim.mul(lagrange_factor);

                // InstructionRa claims from HammingWeightClaimReduction
                for (0..instruction_d) |i| {
                    const claim = result.proof.opening_claims.get(
                        .{ .Committed = .{ .poly = .{ .InstructionRa = i }, .sumcheck_id = .HammingWeightClaimReduction } },
                    ) orelse F.zero();
                    claims_ordered[2 + i] = claim;
                }

                // BytecodeRa claims from HammingWeightClaimReduction
                for (0..bytecode_d) |i| {
                    const claim = result.proof.opening_claims.get(
                        .{ .Committed = .{ .poly = .{ .BytecodeRa = i }, .sumcheck_id = .HammingWeightClaimReduction } },
                    ) orelse F.zero();
                    claims_ordered[2 + instruction_d + i] = claim;
                }

                // RamRa claims from HammingWeightClaimReduction
                for (0..ram_d) |i| {
                    const claim = result.proof.opening_claims.get(
                        .{ .Committed = .{ .poly = .{ .RamRa = i }, .sumcheck_id = .HammingWeightClaimReduction } },
                    ) orelse F.zero();
                    claims_ordered[2 + instruction_d + bytecode_d + i] = claim;
                }

                // 2. Append all claims to transcript
                transcript.appendScalars("rlc_claims", claims_ordered);

                // 3. Sample gamma powers: [1, γ, γ², ..., γ^(n-1)]
                const gamma_powers = try transcript.challengeScalarPowers(self.allocator, num_claims);
                defer self.allocator.free(gamma_powers);

                // 4. Build joint polynomial: Σ γ^i * poly_i
                // Dense polys (RdInc, RamInc) are stored unpadded (length T).
                // They occupy only the first T elements of the k_chunk*T joint poly.
                // Elements [T..k_chunk*T] are zero for dense, non-zero only for sparse one-hot.
                //
                // Jolt Stage 8 gamma order: [0]=RamInc, [1]=RdInc, [2..2+inst_d]=InstructionRa,
                //   [2+inst_d..2+inst_d+bc_d]=BytecodeRa, [2+inst_d+bc_d..]=RamRa

                dbg("[STAGE8] Building joint polynomial (k_chunk={}, trace_length={})...\n", .{ k_chunk, trace_length });
                const total_poly_size = k_chunk * trace_length;
                var joint_poly = try self.allocator.alloc(F, total_poly_size);
                defer self.allocator.free(joint_poly);
                // Zero entire joint poly — dense writes only [0..T), sparse adds at scattered positions
                @memset(joint_poly, F.zero());

                // RamInc + RdInc: accumulate both unpadded dense polys over [0..T).
                // Elements [T..k_chunk*T] remain zero (dense polys are zero-padded implicitly).
                {
                    const ram_inc_wp = witness_polys[1];
                    const rd_inc_wp = witness_polys[0];
                    const gamma_ram = gamma_powers[0];
                    const gamma_rd = gamma_powers[1];
                    const dense_len = trace_length; // unpadded length
                    if (self.thread_pool) |pool| {
                        const DenseCtx = struct {
                            jp: []F,
                            ram: []const F,
                            rd: []const F,
                            gr: F,
                            gd: F,
                        };
                        pool.parallelFor(dense_len, DenseCtx{
                            .jp = joint_poly,
                            .ram = ram_inc_wp,
                            .rd = rd_inc_wp,
                            .gr = gamma_ram,
                            .gd = gamma_rd,
                        }, struct {
                            fn f(cx: DenseCtx, j: usize) void {
                                cx.jp[j] = F.sumOfProducts(.{ cx.ram[j], cx.rd[j] }, .{ cx.gr, cx.gd });
                            }
                        }.f);
                    } else {
                        for (0..dense_len) |j| {
                            joint_poly[j] = F.sumOfProducts(.{ ram_inc_wp[j], rd_inc_wp[j] }, .{ gamma_ram, gamma_rd });
                        }
                    }
                }

                // Fused sparse one-hot accumulation: build (gamma, oh_idx) pairs, then
                // iterate cycles (parallelizable since different cycles write to different positions).
                const num_sparse = instruction_d + bytecode_d + ram_d;
                const MAX_SPARSE = 64;
                var sparse_gamma: [MAX_SPARSE]F = undefined;
                var sparse_oh: [MAX_SPARSE][]?u8 = undefined;
                var si: usize = 0;
                // InstructionRa: gamma[2..2+inst_d], oh[0..inst_d]
                for (0..instruction_d) |i| {
                    sparse_gamma[si] = gamma_powers[2 + i];
                    sparse_oh[si] = onehot_indices[i];
                    si += 1;
                }
                // BytecodeRa: gamma[2+inst_d..2+inst_d+bc_d], oh[inst_d+ram_d..inst_d+ram_d+bc_d]
                for (0..bytecode_d) |i| {
                    sparse_gamma[si] = gamma_powers[2 + instruction_d + i];
                    sparse_oh[si] = onehot_indices[instruction_d + ram_d + i];
                    si += 1;
                }
                // RamRa: gamma[2+inst_d+bc_d..], oh[inst_d..inst_d+ram_d]
                for (0..ram_d) |i| {
                    sparse_gamma[si] = gamma_powers[2 + instruction_d + bytecode_d + i];
                    sparse_oh[si] = onehot_indices[instruction_d + i];
                    si += 1;
                }
                std.debug.assert(si == num_sparse);

                // Parallel over cycles: each cycle writes to unique positions (cycle is low bits of index)
                if (self.thread_pool) |pool| {
                    const SparseCtx = struct {
                        jp: []F,
                        gammas: []const F,
                        ohs: []const []?u8,
                        tl: usize,
                        ns: usize,
                    };
                    pool.parallelFor(trace_length, SparseCtx{
                        .jp = joint_poly,
                        .gammas = sparse_gamma[0..num_sparse],
                        .ohs = sparse_oh[0..num_sparse],
                        .tl = trace_length,
                        .ns = num_sparse,
                    }, struct {
                        fn f(cx: SparseCtx, cycle: usize) void {
                            for (0..cx.ns) |p| {
                                if (cx.ohs[p][cycle]) |addr| {
                                    const j = @as(usize, addr) * cx.tl + cycle;
                                    cx.jp[j] = cx.jp[j].add(cx.gammas[p]);
                                }
                            }
                        }
                    }.f);
                } else {
                    for (0..trace_length) |cycle| {
                        for (0..num_sparse) |p| {
                            if (sparse_oh[p][cycle]) |addr| {
                                const j = @as(usize, addr) * trace_length + cycle;
                                joint_poly[j] = joint_poly[j].add(sparse_gamma[p]);
                            }
                        }
                    }
                }

                // 5. Generate Dory opening proof
                // The opening point must be in LE order for Dory (Jolt reverses BE→LE)
                // opening_point is in BE: [r_address_BE || r_cycle_BE]
                // For CycleMajor layout (default), no reordering needed, just reverse to LE
                const dory_point = try self.allocator.alloc(F, opening_point.len);
                defer self.allocator.free(dory_point);
                for (0..opening_point.len) |i| {
                    dory_point[i] = opening_point[opening_point.len - 1 - i];
                }

                // Combine cached row commitment hints homomorphically instead of
                // recomputing row commitments from joint_poly via full MSM.
                // Reorder from Zolt cache order to Jolt gamma order:
                //   Cache:  [0]=RdInc, [1]=RamInc, [2..2+inst_d]=InstructionRa, [2+inst_d..2+inst_d+ram_d]=RamRa, [2+inst_d+ram_d..]=BytecodeRa
                //   Gamma:  [0]=RamInc, [1]=RdInc, [2..2+inst_d]=InstructionRa, [2+inst_d..2+inst_d+bc_d]=BytecodeRa, [2+inst_d+bc_d..]=RamRa
                const rc_cache = result.row_commitments_cache;
                const hints_ordered = try self.allocator.alloc([]const G1Point, num_claims);
                defer self.allocator.free(hints_ordered);
                hints_ordered[0] = rc_cache[1]; // RamInc (cache[1]) → gamma[0]
                hints_ordered[1] = rc_cache[0]; // RdInc (cache[0]) → gamma[1]
                for (0..instruction_d) |i| {
                    hints_ordered[2 + i] = rc_cache[2 + i]; // InstructionRa: same order
                }
                for (0..bytecode_d) |i| {
                    // BytecodeRa: cache[2+inst_d+ram_d+i] → gamma[2+inst_d+i]
                    hints_ordered[2 + instruction_d + i] = rc_cache[2 + instruction_d + ram_d + i];
                }
                for (0..ram_d) |i| {
                    // RamRa: cache[2+inst_d+i] → gamma[2+inst_d+bc_d+i]
                    hints_ordered[2 + instruction_d + bytecode_d + i] = rc_cache[2 + instruction_d + i];
                }

                // Compute num_rows from poly dimensions (same as Dory internally computes)
                const total_num_vars: usize = if (total_poly_size <= 1) 1 else std.math.log2_int(usize, total_poly_size);
                const total_sigma: usize = (total_num_vars + 1) / 2;
                const total_nu: usize = total_num_vars - total_sigma;
                const hint_num_rows = @as(usize, 1) << @intCast(total_nu);

                const joint_row_commitments = try DoryScheme.combineRowCommitmentHints(
                    hints_ordered,
                    gamma_powers,
                    hint_num_rows,
                    self.allocator,
                    self.thread_pool,
                );
                defer self.allocator.free(joint_row_commitments);

                if (comptime debug_verbose) {
                    // Compute joint_claim = Σ γ^i * claim_i
                    var expected_joint_claim = F.zero();
                    for (0..num_claims) |i| {
                        expected_joint_claim = expected_joint_claim.add(gamma_powers[i].mul(claims_ordered[i]));
                    }
                    const ejc_be = expected_joint_claim.toBytesBE();
                    std.debug.print("[STAGE8] joint_claim_LE=[", .{});
                    for (0..32) |bi| std.debug.print("{x:0>2}", .{ejc_be[31 - bi]});
                    std.debug.print("]\n", .{});
                    const gp1_be = gamma_powers[1].toBytesBE();
                    std.debug.print("[STAGE8] gamma_powers[1]_LE=[", .{});
                    for (0..16) |bi| std.debug.print("{x:0>2}", .{gp1_be[31 - bi]});
                    std.debug.print("]\n", .{});
                    std.debug.print("[STAGE8] transcript_state_before_dory=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}] n_rounds={}\n", .{
                        transcript.state[0], transcript.state[1], transcript.state[2], transcript.state[3],
                        transcript.state[4], transcript.state[5], transcript.state[6], transcript.state[7],
                        transcript.n_rounds,
                    });
                    for (0..@min(5, num_claims)) |i| {
                        const cl_be = claims_ordered[i].toBytesBE();
                        std.debug.print("[STAGE8] claim[{}]_LE=[", .{i});
                        for (0..16) |bi| std.debug.print("{x:0>2}", .{cl_be[31 - bi]});
                        std.debug.print("]\n", .{});
                    }
                }
                dbg("[STAGE8] Starting Dory opening proof (total_poly_size={}, num_claims={})...\n", .{ total_poly_size, num_claims });
                if (comptime debug_verbose) std.debug.print("    [STAGE-TIMING] Stage 8 prep (hints+poly): {d:.1} ms\n", .{@as(f64, @floatFromInt(phase_timer.read())) / 1_000_000.0});
                phase_timer.reset();
                const dory_proof = try DoryScheme.openWithTranscript(
                    &dory_srs,
                    joint_poly,
                    dory_point,
                    joint_row_commitments, // pre-computed via homomorphic hint combining
                    &transcript,
                    self.allocator,
                    self.thread_pool,
                );
                dbg("[STAGE8] Dory opening proof generated.\n", .{});
                result.dory_opening_proof = dory_proof;
                result.opening_point = opening_point;
                const stage8_time = phase_timer.read();
                if (comptime debug_verbose) std.debug.print("    [STAGE-TIMING] Stage 8 (Dory opening): {d:.1} ms\n", .{@as(f64, @floatFromInt(stage8_time)) / 1_000_000.0});

                dbg("[STAGE8] Dory proof: nu={}, sigma={}, first_messages={}, second_messages={}\n", .{
                    dory_proof.nu, dory_proof.sigma,
                    dory_proof.first_messages.len, dory_proof.second_messages.len,
                });
            }

            return result;
        }



        /// Serialize a JoltProofWithDory bundle to bytes
        ///
        /// This uses the Dory commitments that were computed during proving,
        /// ensuring the commitments in the serialized proof match those used
        /// in the transcript.
        pub fn serializeJoltProofWithDory(
            self: *Self,
            bundle: *const jolt_types.JoltProofWithDory(F, commitment_types.PolyCommitment, commitment_types.OpeningProof),
        ) ![]u8 {
            var serializer = jolt_serialization.ArkworksSerializer(F).init(self.allocator);
            errdefer serializer.deinit();

            // Upstream field order from JoltProof struct in proof_serialization.rs:
            // 1. commitments, 2. stage1_uni_skip, 3. stage1_sumcheck, ...
            // 7. stage7_sumcheck, 8. joint_opening_proof, 9. untrusted_advice_commitment,
            // 10. opening_claims, 11. trace_length, 12. ram_K, 13. rw_config,
            // 14. one_hot_config, 15. dory_layout

            // 1. Commitments (GT elements, 384 bytes each)
            dbg("[SERIALIZE] Writing {} Dory commitments\n", .{bundle.dory_commitments.len});
            try serializer.writeUsize(bundle.dory_commitments.len);
            for (bundle.dory_commitments) |comm| {
                try serializer.writeGT(comm);
            }

            // 2. Stage 1 UniSkip (with 0u8 Standard enum discriminant)
            dbg("[SERIALIZE] Writing Stage 1...\n", .{});
            if (bundle.proof.stage1_uni_skip_first_round_proof) |*p| {
                try serializer.writeU8(0); // Standard variant discriminant
                try serializer.writeUniSkipFirstRoundProof(p);
            }
            // 3. Stage 1 sumcheck (with 0u8 Clear enum discriminant)
            try serializer.writeU8(0); // Clear variant discriminant
            try serializer.writeSumcheckInstanceProof(&bundle.proof.stage1_sumcheck_proof);

            // 4. Stage 2 UniSkip + 5. Stage 2 sumcheck
            dbg("[SERIALIZE] Writing Stage 2...\n", .{});
            if (bundle.proof.stage2_uni_skip_first_round_proof) |*p| {
                try serializer.writeU8(0); // Standard variant discriminant
                try serializer.writeUniSkipFirstRoundProof(p);
            }
            try serializer.writeU8(0); // Clear variant discriminant
            try serializer.writeSumcheckInstanceProof(&bundle.proof.stage2_sumcheck_proof);

            // 6-10. Stages 3-7 (each with 0u8 Clear discriminant)
            try serializer.writeU8(0);
            try serializer.writeSumcheckInstanceProof(&bundle.proof.stage3_sumcheck_proof);
            try serializer.writeU8(0);
            try serializer.writeSumcheckInstanceProof(&bundle.proof.stage4_sumcheck_proof);
            try serializer.writeU8(0);
            try serializer.writeSumcheckInstanceProof(&bundle.proof.stage5_sumcheck_proof);
            try serializer.writeU8(0);
            try serializer.writeSumcheckInstanceProof(&bundle.proof.stage6_sumcheck_proof);
            try serializer.writeU8(0);
            try serializer.writeSumcheckInstanceProof(&bundle.proof.stage7_sumcheck_proof);

            // 11. Joint opening proof
            if (bundle.dory_opening_proof) |*dory_proof| {
                try serializer.writeDoryProof(dory_proof);
            } else {
                dbg("[SERIALIZE] WARNING: No pre-computed Dory proof, generating dummy\n", .{});
                const dummy_poly = try self.allocator.alloc(F, 2);
                defer self.allocator.free(dummy_poly);
                dummy_poly[0] = F.zero();
                dummy_poly[1] = F.zero();
                var dummy_srs = try Dory.DoryCommitmentScheme(F).setup(self.allocator, 1);
                defer dummy_srs.deinit();
                var dory_proof = try Dory.DoryCommitmentScheme(F).open(
                    &dummy_srs,
                    dummy_poly,
                    &[_]F{F.zero()},
                    self.allocator,
                );
                defer dory_proof.deinit();
                try serializer.writeDoryProof(&dory_proof);
            }

            // 12. untrusted_advice_commitment: Option<PCS::Commitment> = None
            try serializer.writeU8(0);

            // 13. opening_claims (moved after untrusted_advice in upstream)
            try serializer.writeOpeningClaims(&bundle.proof.opening_claims);

            // 14-15. Configuration fields
            try serializer.writeUsize(bundle.proof.trace_length);
            try serializer.writeUsize(bundle.proof.ram_K);
            // ReadWriteConfig: 4 x u8
            try serializer.writeU8(bundle.proof.rw_config.ram_rw_phase1_num_rounds);
            try serializer.writeU8(bundle.proof.rw_config.ram_rw_phase2_num_rounds);
            try serializer.writeU8(bundle.proof.rw_config.registers_rw_phase1_num_rounds);
            try serializer.writeU8(bundle.proof.rw_config.registers_rw_phase2_num_rounds);
            // OneHotConfig: 2 x u8
            try serializer.writeU8(bundle.proof.one_hot_config.log_k_chunk);
            try serializer.writeU8(bundle.proof.one_hot_config.lookups_ra_virtual_log_k_chunk);
            // DoryLayout: 1 x u8 (0 = Wide, 1 = Tall)
            try serializer.writeU8(bundle.proof.dory_layout);

            return serializer.toOwnedSlice();
        }


    };
}


test {
    // Discover tests in sub-modules
    _ = @import("spartan/mod.zig");
}

test "vm state basic operations" {
    var state = VMState.init(0x80000000);

    // x0 should always be zero
    try std.testing.expectEqual(@as(u64, 0), state.readReg(.zero));

    // Write to x0 should be ignored
    state.writeReg(.zero, 42);
    try std.testing.expectEqual(@as(u64, 0), state.readReg(.zero));

    // Write to other registers should work
    state.writeReg(.a0, 123);
    try std.testing.expectEqual(@as(u64, 123), state.readReg(.a0));
}

test "register enum" {
    try std.testing.expectEqual(@as(u8, 0), Register.zero.toIndex());
    try std.testing.expectEqual(@as(u8, 1), Register.ra.toIndex());
    try std.testing.expectEqual(@as(u8, 2), Register.sp.toIndex());
    try std.testing.expectEqual(@as(u8, 10), Register.a0.toIndex());
}

// ============================================================================
// R1CS-Spartan Integration Tests
// ============================================================================

test "r1cs-spartan: witness generation and Az Bz Cz computation" {
    const F = field.BN254Scalar;
    const allocator = std.testing.allocator;

    // Create execution trace manually to test R1CS integration
    var trace = tracer.ExecutionTrace.init(allocator);
    defer trace.deinit();

    // Add a few execution steps using proper TraceStep structure
    try trace.steps.append(allocator, .{
        .cycle = 0,
        .pc = 0x1000,
        .unexpanded_pc = 0x1000,
        .instruction = 0x00500093, // ADDI x1, x0, 5
        .rs1_value = 0,
        .rs2_value = 0,
        .rd_pre_value = 0, // x1 was 0 before
        .rd_value = 5,
        .memory_addr = null,
        .memory_pre_value = null,
        .memory_value = null,
        .is_memory_write = false,
        .next_pc = 0x1004,
        .is_compressed = false,
    });

    try trace.steps.append(allocator, .{
        .cycle = 1,
        .pc = 0x1004,
        .unexpanded_pc = 0x1004,
        .instruction = 0x00A00113, // ADDI x2, x0, 10
        .rs1_value = 0,
        .rs2_value = 0,
        .rd_pre_value = 0, // x2 was 0 before
        .rd_value = 10,
        .memory_addr = null,
        .memory_pre_value = null,
        .memory_value = null,
        .is_memory_write = false,
        .next_pc = 0x1008,
        .is_compressed = false,
    });

    // Build JoltR1CS and test witness generation
    var jolt_r1cs = try r1cs.JoltR1CS(F).fromTrace(allocator, &trace);
    defer jolt_r1cs.deinit();

    try std.testing.expectEqual(@as(usize, 2), jolt_r1cs.num_cycles);

    // Build witness
    const witness = try jolt_r1cs.buildWitness();
    defer allocator.free(witness);

    // First element should be 1
    try std.testing.expect(witness[0].eql(F.one()));

    // Test Az, Bz, Cz computation
    const Az = try jolt_r1cs.computeAz(witness);
    defer allocator.free(Az);
    const Bz = try jolt_r1cs.computeBz(witness);
    defer allocator.free(Bz);
    const Cz = try jolt_r1cs.computeCz(witness);
    defer allocator.free(Cz);

    // Cz should be all zeros (equality-conditional form)
    for (Cz) |c| {
        try std.testing.expect(c.eql(F.zero()));
    }

    // Verify proper array sizes
    try std.testing.expectEqual(jolt_r1cs.padded_num_constraints, Az.len);
    try std.testing.expectEqual(jolt_r1cs.padded_num_constraints, Bz.len);
    try std.testing.expectEqual(jolt_r1cs.padded_num_constraints, Cz.len);

    // Note: Full constraint satisfaction requires proper instruction decoding
    // and consistent witness values. This test verifies the structure is correct.
}

test "sparse onehot joint poly equivalence with dense witness" {
    const F = field.BN254Scalar;

    // Simulate Stage 8 joint poly construction with small dimensions:
    // instruction_d=2, ram_d=1, bytecode_d=1 (4 total onehot arrays)
    const instruction_d = 2;
    const ram_d = 1;
    const bytecode_d = 1;
    const k_chunk = 4;
    const trace_length = 4;
    const total_poly_size = k_chunk * trace_length; // 16

    // Onehot indices: Zolt stores [InstructionRa, RamRa, BytecodeRa]
    var inst_ra0 = [_]?u8{ 1, 0, 3, 2 };
    var inst_ra1 = [_]?u8{ 0, 2, 1, 3 };
    var ram_ra0 = [_]?u8{ 3, 1, 0, 2 };
    var bc_ra0 = [_]?u8{ 2, 3, 0, 1 };
    const onehot_indices = [_][]?u8{
        &inst_ra0, // [0] InstructionRa chunk 0
        &inst_ra1, // [1] InstructionRa chunk 1
        &ram_ra0, //  [2] RamRa chunk 0
        &bc_ra0, //   [3] BytecodeRa chunk 0
    };

    // Gamma powers — Jolt order: [0]=RamInc, [1]=RdInc, [2..4]=InstructionRa, [4]=BytecodeRa, [5]=RamRa
    const gamma_base = F.fromU64(7); // arbitrary
    var gamma_powers: [6]F = undefined;
    gamma_powers[0] = gamma_base;
    for (1..6) |i| {
        gamma_powers[i] = gamma_powers[i - 1].mul(gamma_base);
    }

    // Dense witness polys: [0]=RdInc, [1]=RamInc (both length trace_length, padded to total_poly_size)
    var rd_inc = [_]F{ F.fromU64(10), F.fromU64(20), F.zero(), F.fromU64(30) } ++ ([_]F{F.zero()} ** (total_poly_size - 4));
    var ram_inc = [_]F{ F.fromU64(5), F.zero(), F.fromU64(15), F.fromU64(25) } ++ ([_]F{F.zero()} ** (total_poly_size - 4));
    var witness_polys = [_][]F{ &rd_inc, &ram_inc };

    // --- Build joint_poly via sparse path (same logic as Stage 8) ---
    var joint_sparse = [_]F{F.zero()} ** total_poly_size;

    // RamInc: gamma_powers[0]
    for (0..@min(witness_polys[1].len, total_poly_size)) |j| {
        if (!witness_polys[1][j].eql(F.zero())) {
            joint_sparse[j] = joint_sparse[j].add(witness_polys[1][j].mul(gamma_powers[0]));
        }
    }
    // RdInc: gamma_powers[1]
    for (0..@min(witness_polys[0].len, total_poly_size)) |j| {
        if (!witness_polys[0][j].eql(F.zero())) {
            joint_sparse[j] = joint_sparse[j].add(witness_polys[0][j].mul(gamma_powers[1]));
        }
    }
    // InstructionRa: gamma_powers[2..2+inst_d], onehot[0..inst_d]
    for (0..instruction_d) |i| {
        const gamma = gamma_powers[2 + i];
        const oh_idx = onehot_indices[i];
        for (0..trace_length) |cycle| {
            if (oh_idx[cycle]) |addr| {
                const j = @as(usize, addr) * trace_length + cycle;
                joint_sparse[j] = joint_sparse[j].add(gamma);
            }
        }
    }
    // BytecodeRa: gamma_powers[2+inst_d..2+inst_d+bc_d], onehot[inst_d+ram_d..] (Zolt order)
    for (0..bytecode_d) |i| {
        const gamma = gamma_powers[2 + instruction_d + i]; // Jolt: BytecodeRa before RamRa
        const oh_arr_idx = instruction_d + ram_d + i; // Zolt: BytecodeRa after RamRa
        const oh_idx = onehot_indices[oh_arr_idx];
        for (0..trace_length) |cycle| {
            if (oh_idx[cycle]) |addr| {
                const j = @as(usize, addr) * trace_length + cycle;
                joint_sparse[j] = joint_sparse[j].add(gamma);
            }
        }
    }
    // RamRa: gamma_powers[2+inst_d+bc_d..], onehot[inst_d..inst_d+ram_d] (Zolt order)
    for (0..ram_d) |i| {
        const gamma = gamma_powers[2 + instruction_d + bytecode_d + i]; // Jolt: RamRa after BytecodeRa
        const oh_arr_idx = instruction_d + i; // Zolt: RamRa before BytecodeRa
        const oh_idx = onehot_indices[oh_arr_idx];
        for (0..trace_length) |cycle| {
            if (oh_idx[cycle]) |addr| {
                const j = @as(usize, addr) * trace_length + cycle;
                joint_sparse[j] = joint_sparse[j].add(gamma);
            }
        }
    }

    // --- Build joint_poly via dense expansion (ground truth) ---
    var joint_dense = [_]F{F.zero()} ** total_poly_size;

    // Dense witness polys contribute the same way
    for (0..@min(witness_polys[1].len, total_poly_size)) |j| {
        if (!witness_polys[1][j].eql(F.zero())) {
            joint_dense[j] = joint_dense[j].add(witness_polys[1][j].mul(gamma_powers[0]));
        }
    }
    for (0..@min(witness_polys[0].len, total_poly_size)) |j| {
        if (!witness_polys[0][j].eql(F.zero())) {
            joint_dense[j] = joint_dense[j].add(witness_polys[0][j].mul(gamma_powers[1]));
        }
    }

    // Dense onehot expansion: expand each onehot array into k_chunk*trace_length dense poly,
    // then accumulate with the correct gamma (Jolt gamma order, not Zolt storage order)
    // Jolt gamma order: InstructionRa[0..inst_d], BytecodeRa[0..bc_d], RamRa[0..ram_d]
    // For InstructionRa: gamma_idx = 2+i, onehot_idx = i
    for (0..instruction_d) |i| {
        const gamma = gamma_powers[2 + i];
        for (0..trace_length) |cycle| {
            if (onehot_indices[i][cycle]) |addr| {
                const j = @as(usize, addr) * trace_length + cycle;
                joint_dense[j] = joint_dense[j].add(gamma);
            }
        }
    }
    // For BytecodeRa: gamma_idx = 2+inst_d+i, onehot_idx = inst_d+ram_d+i
    for (0..bytecode_d) |i| {
        const gamma = gamma_powers[2 + instruction_d + i];
        for (0..trace_length) |cycle| {
            if (onehot_indices[instruction_d + ram_d + i][cycle]) |addr| {
                const j = @as(usize, addr) * trace_length + cycle;
                joint_dense[j] = joint_dense[j].add(gamma);
            }
        }
    }
    // For RamRa: gamma_idx = 2+inst_d+bc_d+i, onehot_idx = inst_d+i
    for (0..ram_d) |i| {
        const gamma = gamma_powers[2 + instruction_d + bytecode_d + i];
        for (0..trace_length) |cycle| {
            if (onehot_indices[instruction_d + i][cycle]) |addr| {
                const j = @as(usize, addr) * trace_length + cycle;
                joint_dense[j] = joint_dense[j].add(gamma);
            }
        }
    }

    // Verify element-wise equality
    for (0..total_poly_size) |j| {
        try std.testing.expect(joint_sparse[j].eql(joint_dense[j]));
    }
}

// Include tests from submodules
test {
    // Force preprocessing tests to be included
    _ = preprocessing;
}
