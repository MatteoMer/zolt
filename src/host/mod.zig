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

/// Preprocessing for Jolt (generates proving/verifying keys)
pub fn Preprocessing(comptime F: type) type {
    return struct {
        const Self = @This();
        const FieldType = F;

        /// Proving key
        pub const ProvingKey = struct {
            _marker: ?*const FieldType = null,
        };

        /// Verifying key
        pub const VerifyingKey = struct {
            _marker: ?*const FieldType = null,
        };

        allocator: Allocator,

        pub fn init(allocator: Allocator) Self {
            return .{
                .allocator = allocator,
            };
        }

        /// Preprocess a program
        pub fn preprocess(
            self: *Self,
            _: *const Program,
        ) !struct { pk: ProvingKey, vk: VerifyingKey } {
            _ = self;
            @panic("Preprocessing.preprocess not yet implemented");
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
