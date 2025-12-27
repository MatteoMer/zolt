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

/// ELF program loader
pub const ELFLoader = struct {
    allocator: Allocator,

    pub fn init(allocator: Allocator) ELFLoader {
        return .{
            .allocator = allocator,
        };
    }

    /// Load a RISC-V ELF binary
    pub fn load(self: *ELFLoader, _: []const u8) !Program {
        _ = self;
        @panic("ELFLoader.load not yet implemented");
    }

    /// Load from file path
    pub fn loadFile(self: *ELFLoader, path: []const u8) !Program {
        const file = try std.fs.cwd().openFile(path, .{});
        defer file.close();

        const contents = try file.readToEndAlloc(self.allocator, 10 * 1024 * 1024);
        defer self.allocator.free(contents);

        return self.load(contents);
    }
};

/// A loaded RISC-V program
pub const Program = struct {
    /// Program bytecode
    bytecode: []const u8,
    /// Entry point address
    entry_point: u64,
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
pub fn execute(
    allocator: Allocator,
    _: *const Program,
    _: []const u8, // inputs
    _: ExecutionOptions,
) !ExecutionTrace {
    _ = allocator;
    @panic("execute not yet implemented");
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
