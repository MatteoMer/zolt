//! Jolt-compatible preprocessing serialization
//!
//! This module provides the ability to export Zolt preprocessing in a format
//! that can be loaded and used by Jolt's verifier. This enables cross-verification
//! where Zolt generates both the proof and the preprocessing for the same program.
//!
//! The preprocessing contains:
//! - BytecodePreprocessing: The program's bytecode in Jolt Instruction format
//! - RAMPreprocessing: Initial memory state
//! - MemoryLayout: Memory region addresses
//! - VerifierSetup: Commitment scheme generators (Dory)
//!
//! Reference: jolt-core/src/zkvm/verifier.rs

const std = @import("std");

const zkvm_debug = @import("debug.zig");
const dbg = zkvm_debug.dbg;

const Allocator = std.mem.Allocator;
const jolt_device = @import("jolt_device.zig");
const MemoryLayout = jolt_device.MemoryLayout;
const common = @import("../common/mod.zig");

// Re-exported submodules
pub const instruction_decoder = @import("instruction_decoder.zig");
pub const decodeToJoltInstruction = instruction_decoder.decodeToJoltInstruction;

pub const bytecode_pc_mapper = @import("bytecode_pc_mapper.zig");
pub const BytecodePCMapper = bytecode_pc_mapper.BytecodePCMapper;

// Re-exported from extracted files
pub const jolt_instruction = @import("jolt_instruction.zig");
pub const JoltInstruction = jolt_instruction.JoltInstruction;

pub const bytecode_preprocessing = @import("bytecode_preprocessing.zig");
pub const BytecodePreprocessing = bytecode_preprocessing.BytecodePreprocessing;

/// RAMPreprocessing - initial memory state
pub const RAMPreprocessing = struct {
    /// Minimum bytecode address
    min_bytecode_address: u64,
    /// Memory words (8-byte aligned)
    bytecode_words: std.ArrayListUnmanaged(u64),
    allocator: Allocator,

    pub fn init(allocator: Allocator) RAMPreprocessing {
        return .{
            .min_bytecode_address = 0,
            .bytecode_words = .{},
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *RAMPreprocessing) void {
        self.bytecode_words.deinit(self.allocator);
    }

    /// Create from memory initialization data
    pub fn preprocess(allocator: Allocator, memory_init: []const struct { u64, u8 }) !RAMPreprocessing {
        var self = RAMPreprocessing.init(allocator);
        errdefer self.deinit();

        if (memory_init.len == 0) {
            return self;
        }

        // Find min/max addresses
        var min_addr: u64 = memory_init[0][0];
        var max_addr: u64 = memory_init[0][0];
        for (memory_init) |entry| {
            min_addr = @min(min_addr, entry[0]);
            max_addr = @max(max_addr, entry[0]);
        }

        // Account for instruction bytes (4 bytes per instruction)
        max_addr += 3;

        // Calculate word range
        const min_word = min_addr / 8;
        const max_word = (max_addr + 7) / 8;
        const num_words = max_word - min_word + 1;

        self.min_bytecode_address = min_word * 8;

        // Allocate and zero words
        try self.bytecode_words.resize(allocator, num_words);
        @memset(self.bytecode_words.items, 0);

        // Fill in bytes
        for (memory_init) |entry| {
            const addr = entry[0];
            const byte = entry[1];
            const word_idx = (addr / 8) - min_word;
            const byte_offset: u6 = @intCast(addr % 8);
            self.bytecode_words.items[word_idx] |= @as(u64, byte) << (byte_offset * 8);
        }

        return self;
    }

    /// Serialize to arkworks format
    pub fn serialize(self: *const RAMPreprocessing, writer: anytype) !void {
        // min_bytecode_address
        try writer.writeInt(u64, self.min_bytecode_address, .little);

        // bytecode_words: Vec<u64>
        try writer.writeInt(u64, @intCast(self.bytecode_words.items.len), .little);
        for (self.bytecode_words.items) |word| {
            try writer.writeInt(u64, word, .little);
        }
    }
};

/// JoltSharedPreprocessing - shared between prover and verifier
pub const JoltSharedPreprocessing = struct {
    bytecode: BytecodePreprocessing,
    ram: RAMPreprocessing,
    memory_layout: MemoryLayout,
    max_padded_trace_length: usize,

    pub fn deinit(self: *JoltSharedPreprocessing) void {
        self.bytecode.deinit();
        self.ram.deinit();
    }

    /// Serialize to arkworks format
    pub fn serialize(self: *const JoltSharedPreprocessing, allocator: Allocator, writer: anytype) !void {
        try self.bytecode.serialize(allocator, writer);
        try self.ram.serialize(writer);
        try self.memory_layout.serialize(writer);
        // max_padded_trace_length: usize (as u64)
        try writer.writeInt(u64, @intCast(self.max_padded_trace_length), .little);
    }

    /// Compute a Blake2b-256 digest of the serialized preprocessing.
    /// Used to bind preprocessing to the Fiat-Shamir transcript (PR #1408).
    pub fn digest(self: *const JoltSharedPreprocessing, allocator: Allocator) ![32]u8 {
        const Blake2b256 = std.crypto.hash.blake2.Blake2b256;

        // Serialize to an in-memory buffer
        var buf = std.ArrayListUnmanaged(u8){};
        defer buf.deinit(allocator);
        try self.serialize(allocator, buf.writer(allocator));

        // Hash with Blake2b-256
        var h = Blake2b256.init(.{});
        h.update(buf.items);
        var out: [32]u8 = undefined;
        h.final(&out);
        return out;
    }
};

// serializeMemoryLayout has been moved to MemoryLayout.serialize() in common/jolt_device.zig

// ============================================================================
// Tests
// ============================================================================

test "bytecode preprocessing" {
    const allocator = std.testing.allocator;

    // Simple program: ADDI x1, x0, 42; ADD x2, x1, x1
    const code = [_]u8{
        0x93, 0x00, 0xa0, 0x02, // ADDI x1, x0, 42
        0x33, 0x81, 0x10, 0x00, // ADD x2, x1, x1
    };

    var preprocessing = try BytecodePreprocessing.preprocess(allocator, &code, 0x80000000, null);
    defer preprocessing.deinit();

    // Should have NoOp + 2 instructions, padded to power of 2
    try std.testing.expect(preprocessing.bytecode.items.len >= 3);
    try std.testing.expect(preprocessing.code_size >= 4); // Next power of 2

    // First should be NoOp
    try std.testing.expectEqual(JoltInstruction.InstructionVariant.NoOp, preprocessing.bytecode.items[0].variant);

    // Second should be ADDI
    try std.testing.expectEqual(JoltInstruction.InstructionVariant.ADDI, preprocessing.bytecode.items[1].variant);

    // Third should be ADD
    try std.testing.expectEqual(JoltInstruction.InstructionVariant.ADD, preprocessing.bytecode.items[2].variant);
}

// ============================================================================
// Dory Verifier Setup (re-exported from dory_verifier_setup.zig)
// ============================================================================

pub const dory_verifier_setup = @import("dory_verifier_setup.zig");
pub const DoryVerifierSetup = dory_verifier_setup.DoryVerifierSetup;
pub const serializeGT = dory_verifier_setup.serializeGT;
pub const serializeFp6 = dory_verifier_setup.serializeFp6;
pub const serializeFp2 = dory_verifier_setup.serializeFp2;
pub const serializeFp = dory_verifier_setup.serializeFp;
pub const serializeG1 = dory_verifier_setup.serializeG1;
pub const serializeG2 = dory_verifier_setup.serializeG2;
pub const lexicographicallyLess = dory_verifier_setup.lexicographicallyLess;
pub const lexicographicallyLessFp2 = dory_verifier_setup.lexicographicallyLessFp2;
pub const GT = dory_verifier_setup.GT;
pub const G1Point = dory_verifier_setup.G1Point;
pub const G2Point = dory_verifier_setup.G2Point;
pub const DorySRS = dory_verifier_setup.DorySRS;

/// JoltVerifierPreprocessing - full preprocessing for verification
pub const JoltVerifierPreprocessing = struct {
    generators: DoryVerifierSetup,
    shared: JoltSharedPreprocessing,

    pub fn deinit(self: *JoltVerifierPreprocessing) void {
        self.generators.deinit();
        self.shared.deinit();
    }

    /// Serialize to arkworks format
    pub fn serialize(self: *const JoltVerifierPreprocessing, allocator: Allocator, writer: anytype) !void {
        // First serialize generators (VerifierSetup)
        try self.generators.serialize(writer);
        // Then serialize shared preprocessing
        try self.shared.serialize(allocator, writer);
    }
};
