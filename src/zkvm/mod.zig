//! Jolt zkVM - Zero-knowledge Virtual Machine
//!
//! This module implements the core zkVM functionality:
//! - RISC-V instruction execution
//! - Bytecode handling
//! - Memory and register checking
//! - R1CS constraint system
//! - Spartan proof system

const std = @import("std");
const Allocator = std.mem.Allocator;
const common = @import("../common/mod.zig");
const field = @import("../field/mod.zig");
const tracer = @import("../tracer/mod.zig");
const transcripts = @import("../transcripts/mod.zig");

pub const bytecode = @import("bytecode/mod.zig");
pub const instruction = @import("instruction/mod.zig");
pub const lasso = @import("lasso/mod.zig");
pub const lookup_table = @import("lookup_table/mod.zig");
pub const prover = @import("prover.zig");
pub const r1cs = @import("r1cs/mod.zig");
pub const ram = @import("ram/mod.zig");
pub const registers = @import("registers/mod.zig");
pub const spartan = @import("spartan/mod.zig");

// Re-export multi-stage prover types
pub const MultiStageProver = prover.MultiStageProver;
pub const BatchedSumcheckProver = prover.BatchedSumcheckProver;
pub const StageProof = prover.StageProof;
pub const JoltStageProofs = prover.JoltStageProofs;
pub const OpeningAccumulator = prover.OpeningAccumulator;
pub const SumcheckInstance = prover.SumcheckInstance;

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

/// Jolt proof structure
pub fn JoltProof(comptime F: type) type {
    return struct {
        const Self = @This();

        /// Bytecode proof
        bytecode_proof: bytecode.BytecodeProof(F),
        /// Read-write memory proof
        memory_proof: ram.MemoryProof(F),
        /// Register proof
        register_proof: registers.RegisterProof(F),
        /// R1CS/Spartan proof
        r1cs_proof: spartan.R1CSProof(F),

        pub fn deinit(self: *Self, allocator: Allocator) void {
            self.bytecode_proof.deinit(allocator);
            self.memory_proof.deinit(allocator);
            self.register_proof.deinit(allocator);
            self.r1cs_proof.deinit(allocator);
        }
    };
}

/// Jolt prover
pub fn JoltProver(comptime F: type) type {
    return struct {
        const Self = @This();

        allocator: Allocator,
        max_cycles: u64,

        pub fn init(allocator: Allocator) Self {
            return .{
                .allocator = allocator,
                .max_cycles = common.constants.DEFAULT_MAX_TRACE_LENGTH,
            };
        }

        /// Generate a proof for program execution
        ///
        /// This function:
        /// 1. Executes the program in the emulator to generate an execution trace
        /// 2. Runs the multi-stage sumcheck protocol
        /// 3. Generates commitment proofs for polynomials
        /// 4. Returns the complete Jolt proof
        pub fn prove(
            self: *Self,
            program_bytecode: []const u8,
            inputs: []const u8,
        ) !JoltProof(F) {
            // Initialize memory config
            var config = common.MemoryConfig{
                .program_size = program_bytecode.len,
            };

            // Initialize the emulator
            var emulator = tracer.Emulator.init(self.allocator, &config);
            defer emulator.deinit();

            // Set max cycles
            emulator.max_cycles = self.max_cycles;

            // Load the program into memory
            try emulator.loadProgram(program_bytecode);

            // Set input data
            if (inputs.len > 0) {
                try emulator.setInputs(inputs);
            }

            // Execute the program
            try emulator.run();

            // Calculate parameters for multi-stage prover
            // log_k: log2 of address space size (2^16 addresses)
            // log_t: log2 of trace length (calculated internally by MultiStageProver)
            const log_k: usize = 16;

            // Initialize multi-stage prover
            var multi_stage = prover.MultiStageProver(F).init(
                self.allocator,
                &emulator.trace,
                &emulator.ram.trace,
                &emulator.lookup_trace,
                log_k,
                common.constants.RAM_START_ADDRESS,
            );
            defer multi_stage.deinit();

            // Initialize transcript for Fiat-Shamir
            var transcript = transcripts.Transcript.init(self.allocator);
            defer transcript.deinit();

            // Run multi-stage proving
            // Note: This currently only runs the skeleton of each stage
            // Full implementation would generate real proofs
            _ = try multi_stage.prove(&transcript);

            // For now, return placeholder proof components
            // A full implementation would generate real commitments and proofs
            return JoltProof(F){
                .bytecode_proof = bytecode.BytecodeProof(F){
                    .commitment = F.zero(),
                    .read_ts_commitment = F.zero(),
                    .write_ts_commitment = F.zero(),
                },
                .memory_proof = ram.MemoryProof(F){
                    .commitment = F.zero(),
                    .read_ts_commitment = F.zero(),
                    .write_ts_commitment = F.zero(),
                },
                .register_proof = registers.RegisterProof(F){
                    .commitment = F.zero(),
                    .read_ts_commitment = F.zero(),
                    .write_ts_commitment = F.zero(),
                },
                .r1cs_proof = spartan.R1CSProof(F){
                    .sumcheck_proof = F.zero(),
                    .opening_proof = F.zero(),
                },
            };
        }

        /// Set the maximum number of cycles to execute
        pub fn setMaxCycles(self: *Self, max_cycles: u64) void {
            self.max_cycles = max_cycles;
        }
    };
}

/// Jolt verifier
pub fn JoltVerifier(comptime F: type) type {
    return struct {
        const Self = @This();

        allocator: Allocator,

        pub fn init(allocator: Allocator) Self {
            return .{
                .allocator = allocator,
            };
        }

        /// Verify a proof
        pub fn verify(
            self: *Self,
            _: *const JoltProof(F),
            _: []const u8, // public inputs
        ) !bool {
            _ = self;
            @panic("JoltVerifier.verify not yet implemented");
        }
    };
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
