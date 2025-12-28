//! Host interface for Jolt zkVM
//!
//! This module provides the high-level interface for:
//! - Loading and compiling RISC-V programs
//! - Executing programs
//! - Generating proofs
//! - Verifying proofs

const std = @import("std");
const Allocator = std.mem.Allocator;
const common = @import("../common/mod.zig");
const zkvm = @import("../zkvm/mod.zig");
const field = @import("../field/mod.zig");
const tracer = @import("../tracer/mod.zig");
pub const elf = @import("elf.zig");

/// ELF program loader
pub const ELFLoader = struct {
    allocator: Allocator,

    pub fn init(allocator: Allocator) ELFLoader {
        return .{
            .allocator = allocator,
        };
    }

    /// Load a RISC-V ELF binary from bytes
    pub fn load(self: *ELFLoader, data: []const u8) !Program {
        // Parse the ELF file
        var parsed = try elf.parse(self.allocator, data);
        defer parsed.deinit();

        // Validate it's a RISC-V binary
        if (!parsed.isRiscV()) {
            return error.NotRiscV;
        }

        // Calculate total bytecode size needed
        var max_addr: u64 = 0;
        var min_addr: u64 = std.math.maxInt(u64);
        for (parsed.segments) |segment| {
            if (segment.vaddr < min_addr) {
                min_addr = segment.vaddr;
            }
            const end_addr = segment.vaddr + segment.memsz;
            if (end_addr > max_addr) {
                max_addr = end_addr;
            }
        }

        // Use RAM_START_ADDRESS as base if no segments
        if (min_addr == std.math.maxInt(u64)) {
            min_addr = common.constants.RAM_START_ADDRESS;
            max_addr = min_addr;
        }

        // Allocate bytecode buffer (relative to base address)
        const base_addr = min_addr;
        const bytecode_size = max_addr - base_addr;
        const bytecode = try self.allocator.alloc(u8, bytecode_size);
        errdefer self.allocator.free(bytecode);

        // Initialize to zero (for .bss sections)
        @memset(bytecode, 0);

        // Copy segment data into bytecode buffer
        for (parsed.segments) |segment| {
            const offset = segment.vaddr - base_addr;
            const dest = bytecode[@as(usize, @intCast(offset))..];
            @memcpy(dest[0..segment.data.len], segment.data);
        }

        // Create memory config for layout
        var config = common.MemoryConfig{
            .program_size = bytecode_size,
        };

        return Program{
            .bytecode = bytecode,
            .entry_point = parsed.header.entry,
            .base_address = base_addr,
            .memory_layout = common.MemoryLayout.init(&config),
            .allocator = self.allocator,
        };
    }

    /// Load from file path
    pub fn loadFile(self: *ELFLoader, path: []const u8) !Program {
        const file = try std.fs.cwd().openFile(path, .{});
        defer file.close();

        const stat = try file.stat();
        const contents = try self.allocator.alloc(u8, stat.size);
        defer self.allocator.free(contents);

        _ = try file.readAll(contents);

        return self.load(contents);
    }
};

/// A loaded RISC-V program
pub const Program = struct {
    /// Program bytecode
    bytecode: []const u8,
    /// Entry point address
    entry_point: u64,
    /// Base address for the program in memory
    base_address: u64,
    /// Memory layout
    memory_layout: common.MemoryLayout,
    allocator: Allocator,

    pub fn deinit(self: *Program) void {
        self.allocator.free(self.bytecode);
    }

    /// Get the program size
    pub fn size(self: *const Program) usize {
        return self.bytecode.len;
    }

    /// Get byte at a given address
    pub fn getByte(self: *const Program, addr: u64) ?u8 {
        if (addr < self.base_address) return null;
        const offset = addr - self.base_address;
        if (offset >= self.bytecode.len) return null;
        return self.bytecode[@as(usize, @intCast(offset))];
    }

    /// Get a slice of bytes at a given address
    pub fn getBytes(self: *const Program, addr: u64, len: usize) ?[]const u8 {
        if (addr < self.base_address) return null;
        const offset = @as(usize, @intCast(addr - self.base_address));
        if (offset + len > self.bytecode.len) return null;
        return self.bytecode[offset .. offset + len];
    }
};

/// Execution trace from running a program
pub const ExecutionTrace = struct {
    /// Number of cycles executed
    num_cycles: u64,
    /// Register trace
    register_trace: zkvm.registers.RegisterTrace,
    /// Memory trace
    memory_trace: zkvm.ram.MemoryTrace,
    /// Final VM state
    final_state: zkvm.VMState,
    allocator: Allocator,

    pub fn deinit(self: *ExecutionTrace) void {
        self.register_trace.deinit();
        self.memory_trace.deinit();
    }
};

/// Program execution options
pub const ExecutionOptions = struct {
    /// Maximum number of cycles to execute
    max_cycles: u64 = common.constants.DEFAULT_MAX_TRACE_LENGTH,
    /// Whether to record trace for proving
    record_trace: bool = true,
};

/// Execute a program and generate a trace
///
/// This function runs the RISC-V program in the emulator and returns
/// an execution trace that can be used for proof generation.
pub fn execute(
    allocator: Allocator,
    program: *const Program,
    inputs: []const u8,
    options: ExecutionOptions,
) !ExecutionTrace {
    // Create memory config for the emulator
    var config = common.MemoryConfig{
        .program_size = program.bytecode.len,
    };

    // Initialize the emulator
    var emulator = tracer.Emulator.init(allocator, &config);
    defer emulator.deinit();

    // Set max cycles
    emulator.max_cycles = options.max_cycles;

    // Load the program into memory
    try emulator.loadProgram(program.bytecode);

    // Set the entry point
    emulator.state.pc = program.entry_point;

    // Set input data
    if (inputs.len > 0) {
        try emulator.setInputs(inputs);
    }

    // Run the program
    try emulator.run();

    // Extract trace data
    // For now, we return a simplified trace. The full implementation would
    // convert the emulator's trace to the format needed for proving.
    return ExecutionTrace{
        .num_cycles = emulator.state.cycle,
        .register_trace = try emulator.registers.toTrace(allocator),
        .memory_trace = try emulator.ram.toTrace(allocator),
        .final_state = emulator.state,
        .allocator = allocator,
    };
}

/// Jolt instance for proving and verifying
pub fn Jolt(comptime F: type) type {
    return struct {
        const Self = @This();

        allocator: Allocator,
        prover: zkvm.JoltProver(F),
        verifier: zkvm.JoltVerifier(F),

        pub fn init(allocator: Allocator) Self {
            return .{
                .allocator = allocator,
                .prover = zkvm.JoltProver(F).init(allocator),
                .verifier = zkvm.JoltVerifier(F).init(allocator),
            };
        }

        /// Prove program execution
        pub fn prove(
            self: *Self,
            program: *const Program,
            inputs: []const u8,
        ) !zkvm.JoltProof(F) {
            return self.prover.prove(program.bytecode, inputs);
        }

        /// Verify a proof
        pub fn verify(
            self: *Self,
            proof: *const zkvm.JoltProof(F),
            public_inputs: []const u8,
        ) !bool {
            return self.verifier.verify(proof, public_inputs);
        }
    };
}

/// Shared preprocessing data (used by both prover and verifier)
pub fn SharedPreprocessing(comptime F: type) type {
    return struct {
        const Self = @This();
        const FieldType = F;

        /// Bytecode preprocessing
        bytecode_size: usize,
        /// Padded bytecode size (power of 2)
        padded_bytecode_size: usize,
        /// Memory layout
        memory_layout: common.MemoryLayout,
        /// Initial memory state hash (for public verification)
        init_memory_hash: F,
        allocator: Allocator,

        pub fn init(allocator: Allocator, program: *const Program) !Self {
            // Round bytecode size to power of 2
            var size = program.bytecode.len;
            if (size == 0) size = 1;
            var padded: usize = 1;
            while (padded < size) padded <<= 1;
            if (padded < 2) padded = 2;

            // Compute a simple hash of initial bytecode
            // In full implementation, this would be a polynomial commitment
            var hash = F.zero();
            for (program.bytecode, 0..) |byte, i| {
                const byte_field = F.fromU64(@as(u64, byte));
                const idx_field = F.fromU64(@as(u64, i + 1));
                hash = hash.add(byte_field.mul(idx_field));
            }

            return .{
                .bytecode_size = program.bytecode.len,
                .padded_bytecode_size = padded,
                .memory_layout = program.memory_layout,
                .init_memory_hash = hash,
                .allocator = allocator,
            };
        }

        pub fn deinit(_: *const Self) void {
            // Nothing to deallocate for now
        }
    };
}

/// Preprocessing for Jolt (generates proving/verifying keys)
pub fn Preprocessing(comptime F: type) type {
    return struct {
        const Self = @This();
        const FieldType = F;
        const poly = @import("../poly/mod.zig");
        const HyperKZG = poly.commitment.HyperKZG(F);

        /// Proving key - contains prover-specific data
        pub const ProvingKey = struct {
            /// Shared preprocessing data
            shared: SharedPreprocessing(F),
            /// Polynomial commitment SRS (prover side)
            srs: HyperKZG.SetupParams,
            /// Maximum trace length supported
            max_trace_length: usize,
            allocator: Allocator,

            pub fn deinit(self: *ProvingKey) void {
                self.shared.deinit();
                self.srs.deinit();
            }
        };

        /// G1 point type (for MSM) - uses base field Fp for coordinates
        const msm = @import("../msm/mod.zig");
        const Fp = @import("../field/mod.zig").BN254BaseField;
        const G1Point = msm.AffinePoint(Fp);
        const G2Point = @import("../field/mod.zig").pairing.G2Point;

        /// Verifying key - contains verifier-specific data
        pub const VerifyingKey = struct {
            /// Shared preprocessing data
            shared: SharedPreprocessing(F),
            /// G1 generator for verification
            g1: G1Point,
            /// G2 generator for verification
            g2: G2Point,
            /// [tau]_2 for pairing checks
            tau_g2: G2Point,
            allocator: Allocator,

            pub fn deinit(self: *VerifyingKey) void {
                self.shared.deinit();
            }
        };

        allocator: Allocator,
        /// Maximum trace length to support
        max_trace_length: usize,

        pub fn init(allocator: Allocator) Self {
            return .{
                .allocator = allocator,
                .max_trace_length = common.constants.DEFAULT_MAX_TRACE_LENGTH,
            };
        }

        /// Set the maximum trace length to support
        pub fn setMaxTraceLength(self: *Self, max_trace_length: usize) void {
            self.max_trace_length = max_trace_length;
        }

        /// Preprocess a program to generate proving and verifying keys
        ///
        /// This performs the following steps:
        /// 1. Create shared preprocessing data (bytecode, memory layout)
        /// 2. Generate polynomial commitment SRS
        /// 3. Create proving key with full SRS
        /// 4. Create verifying key with verification parameters
        pub fn preprocess(
            self: *Self,
            program: *const Program,
        ) !struct { pk: ProvingKey, vk: VerifyingKey } {
            // Create shared preprocessing for prover
            const shared_pk = try SharedPreprocessing(F).init(self.allocator, program);
            errdefer shared_pk.deinit();

            // Create shared preprocessing for verifier (copy)
            const shared_vk = try SharedPreprocessing(F).init(self.allocator, program);
            errdefer shared_vk.deinit();

            // Compute SRS size: need enough powers of tau for the trace
            // log_chunk typically 8, so we need 2^(log_chunk + log_T) powers
            const padded_trace_length = blk: {
                var len: usize = 1;
                while (len < self.max_trace_length) len <<= 1;
                break :blk len;
            };
            const log_chunk: usize = 8;
            const srs_size = (@as(usize, 1) << log_chunk) + padded_trace_length;

            // Generate SRS
            const srs = try HyperKZG.setup(self.allocator, srs_size);

            // Create proving key
            const pk = ProvingKey{
                .shared = shared_pk,
                .srs = srs,
                .max_trace_length = self.max_trace_length,
                .allocator = self.allocator,
            };

            // Create verifying key (extract verification parameters from SRS)
            const vk = VerifyingKey{
                .shared = shared_vk,
                .g1 = srs.g1,
                .g2 = srs.g2,
                .tau_g2 = srs.tau_g2,
                .allocator = self.allocator,
            };

            return .{ .pk = pk, .vk = vk };
        }

        /// Preprocess with an external SRS (e.g., loaded from a PTAU file)
        ///
        /// This is useful when you want to use a production SRS from a ceremony
        /// instead of a mock SRS generated for testing.
        pub fn preprocessWithSRS(
            self: *Self,
            program: *const Program,
            srs: HyperKZG.SetupParams,
        ) !struct { pk: ProvingKey, vk: VerifyingKey } {
            // Create shared preprocessing for prover
            const shared_pk = try SharedPreprocessing(F).init(self.allocator, program);
            errdefer shared_pk.deinit();

            // Create shared preprocessing for verifier (copy)
            const shared_vk = try SharedPreprocessing(F).init(self.allocator, program);
            errdefer shared_vk.deinit();

            // Validate SRS is large enough for the trace
            const padded_trace_length = blk: {
                var len: usize = 1;
                while (len < self.max_trace_length) len <<= 1;
                break :blk len;
            };
            const log_chunk: usize = 8;
            const required_size = (@as(usize, 1) << log_chunk) + padded_trace_length;

            if (srs.max_degree < required_size) {
                return error.SRSTooSmall;
            }

            // Create proving key with the provided SRS
            const pk = ProvingKey{
                .shared = shared_pk,
                .srs = srs,
                .max_trace_length = self.max_trace_length,
                .allocator = self.allocator,
            };

            // Create verifying key (extract verification parameters from SRS)
            const vk = VerifyingKey{
                .shared = shared_vk,
                .g1 = srs.g1,
                .g2 = srs.g2,
                .tau_g2 = srs.tau_g2,
                .allocator = self.allocator,
            };

            return .{ .pk = pk, .vk = vk };
        }
    };
}

test "host types compile" {
    const F = field.BN254Scalar;

    // Verify types compile
    _ = ELFLoader;
    _ = Program;
    _ = ExecutionTrace;
    _ = Jolt(F);
    _ = Preprocessing(F);
}

test "preprocessing generates keys" {
    const allocator = std.testing.allocator;
    const F = field.BN254Scalar;

    // Create a simple mock program
    const bytecode = [_]u8{
        0x93, 0x00, 0xa0, 0x02, // addi x1, x0, 42
        0x01, 0x00, // c.nop
    };

    var config = common.MemoryConfig{
        .program_size = bytecode.len,
    };

    const program = Program{
        .bytecode = &bytecode,
        .entry_point = common.constants.RAM_START_ADDRESS,
        .base_address = common.constants.RAM_START_ADDRESS,
        .memory_layout = common.MemoryLayout.init(&config),
        .allocator = allocator,
    };

    // Run preprocessing with a small max trace length for testing
    var preprocessor = Preprocessing(F).init(allocator);
    preprocessor.setMaxTraceLength(256);

    var result = try preprocessor.preprocess(&program);
    defer result.pk.deinit();
    defer result.vk.deinit();

    // Verify proving key
    try std.testing.expectEqual(@as(usize, bytecode.len), result.pk.shared.bytecode_size);
    try std.testing.expect(result.pk.shared.padded_bytecode_size >= bytecode.len);
    try std.testing.expectEqual(@as(usize, 256), result.pk.max_trace_length);
    try std.testing.expect(result.pk.srs.max_degree > 0);

    // Verify verifying key
    try std.testing.expectEqual(@as(usize, bytecode.len), result.vk.shared.bytecode_size);
    try std.testing.expect(!result.vk.g1.infinity);
    try std.testing.expect(!result.vk.g2.infinity);
}

test "shared preprocessing hash consistency" {
    const allocator = std.testing.allocator;
    const F = field.BN254Scalar;

    // Same bytecode should produce same hash
    const bytecode = [_]u8{ 0x13, 0x00, 0x00, 0x00 };
    var config = common.MemoryConfig{
        .program_size = bytecode.len,
    };

    const program = Program{
        .bytecode = &bytecode,
        .entry_point = common.constants.RAM_START_ADDRESS,
        .base_address = common.constants.RAM_START_ADDRESS,
        .memory_layout = common.MemoryLayout.init(&config),
        .allocator = allocator,
    };

    var shared1 = try SharedPreprocessing(F).init(allocator, &program);
    defer shared1.deinit();

    var shared2 = try SharedPreprocessing(F).init(allocator, &program);
    defer shared2.deinit();

    try std.testing.expect(shared1.init_memory_hash.eql(shared2.init_memory_hash));
}

test "execute runs simple program" {
    const allocator = std.testing.allocator;

    // Simple program: c.nop (compressed NOP instruction)
    const bytecode = [_]u8{ 0x01, 0x00 };
    var config = common.MemoryConfig{
        .program_size = bytecode.len,
    };

    const program = Program{
        .bytecode = &bytecode,
        .entry_point = common.constants.RAM_START_ADDRESS,
        .base_address = common.constants.RAM_START_ADDRESS,
        .memory_layout = common.MemoryLayout.init(&config),
        .allocator = allocator,
    };

    // Execute with small cycle limit
    var trace = try execute(allocator, &program, &[_]u8{}, .{
        .max_cycles = 10,
    });
    defer trace.deinit();

    // Should have executed at least 1 cycle
    try std.testing.expect(trace.num_cycles >= 1);
}

test "execute with longer program" {
    const allocator = std.testing.allocator;

    // Program: addi x1, x0, 42; addi x2, x1, 1; c.nop
    const bytecode = [_]u8{
        0x93, 0x00, 0xa0, 0x02, // addi x1, x0, 42
        0x13, 0x01, 0x10, 0x00, // addi x2, x1, 1
        0x01, 0x00, // c.nop
    };
    var config = common.MemoryConfig{
        .program_size = bytecode.len,
    };

    const program = Program{
        .bytecode = &bytecode,
        .entry_point = common.constants.RAM_START_ADDRESS,
        .base_address = common.constants.RAM_START_ADDRESS,
        .memory_layout = common.MemoryLayout.init(&config),
        .allocator = allocator,
    };

    // Execute
    var trace = try execute(allocator, &program, &[_]u8{}, .{
        .max_cycles = 50,
    });
    defer trace.deinit();

    // Should have executed at least 3 cycles for the 3 instructions
    try std.testing.expect(trace.num_cycles >= 3);
}
