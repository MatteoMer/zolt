//! JoltDevice for Fiat-Shamir Preamble and Proof Serialization
//!
//! This type matches the Jolt definition in `common/src/jolt_device.rs`
//! and is used for Fiat-Shamir transcript preamble to ensure compatible
//! challenge derivation between Zolt prover and Jolt verifier.
//!
//! MemoryLayout is imported from `src/common/jolt_device.zig` — the single
//! canonical definition shared by both the emulator and the proof system.
//!
//! Reference: jolt/common/src/jolt_device.rs

const std = @import("std");

const zkvm_debug = @import("debug.zig");
const dbg = zkvm_debug.dbg;
const jolt_types = @import("jolt_types.zig");

const Allocator = std.mem.Allocator;
const common_jolt = @import("../common/jolt_device.zig");
const constants = @import("../common/constants.zig");

/// Re-export canonical MemoryLayout from common
pub const MemoryLayout = common_jolt.MemoryLayout;

/// JoltDevice matching Jolt's JoltDevice struct
pub const JoltDevice = struct {
    inputs: []const u8,
    trusted_advice: []const u8,
    untrusted_advice: []const u8,
    outputs: []const u8,
    panic: bool,
    memory_layout: MemoryLayout,

    allocator: ?Allocator,

    const Self = @This();

    /// Create a JoltDevice from Zolt emulator state
    ///
    /// heap_size: The memory size to use. If null, uses 128 MB default.
    ///              For Jolt fibonacci example, use 32768 (32 KB).
    ///
    /// CRITICAL: Memory layout matches Jolt's MemoryLayout::new from common/src/jolt_device.rs
    /// The I/O region is placed BEFORE RAM_START_ADDRESS, not after!
    pub fn fromEmulator(
        allocator: Allocator,
        inputs: []const u8,
        outputs: []const u8,
        panic_flag: bool,
        program_size: u64,
        heap_size_opt: ?u64,
    ) !Self {
        // Copy inputs/outputs
        const inputs_copy = try allocator.alloc(u8, inputs.len);
        @memcpy(inputs_copy, inputs);
        const outputs_copy = try allocator.alloc(u8, outputs.len);
        @memcpy(outputs_copy, outputs);

        const config = common_jolt.MemoryConfig{
            .program_size = program_size,
            .heap_size = heap_size_opt orelse constants.DEFAULT_MEMORY_SIZE,
        };

        return Self{
            .inputs = inputs_copy,
            .trusted_advice = &[_]u8{},
            .untrusted_advice = &[_]u8{},
            .outputs = outputs_copy,
            .panic = panic_flag,
            .memory_layout = MemoryLayout.init(&config),
            .allocator = allocator,
        };
    }

    /// Deserialize from arkworks canonical format (from Jolt-generated file)
    pub fn deserialize(allocator: Allocator, data: []const u8) !Self {
        var stream = std.io.fixedBufferStream(data);
        var reader = stream.reader();

        // Read inputs Vec<u8>: length (u64) + bytes
        const inputs_len = try reader.readInt(u64, .little);
        const inputs = try allocator.alloc(u8, inputs_len);
        errdefer allocator.free(inputs);
        if (inputs_len > 0) {
            const read_len = try reader.readAll(inputs);
            if (read_len != inputs_len) return error.UnexpectedEof;
        }

        // Read trusted_advice Vec<u8>
        const trusted_len = try reader.readInt(u64, .little);
        const trusted_advice = try allocator.alloc(u8, trusted_len);
        errdefer allocator.free(trusted_advice);
        if (trusted_len > 0) {
            const read_len = try reader.readAll(trusted_advice);
            if (read_len != trusted_len) return error.UnexpectedEof;
        }

        // Read untrusted_advice Vec<u8>
        const untrusted_len = try reader.readInt(u64, .little);
        const untrusted_advice = try allocator.alloc(u8, untrusted_len);
        errdefer allocator.free(untrusted_advice);
        if (untrusted_len > 0) {
            const read_len = try reader.readAll(untrusted_advice);
            if (read_len != untrusted_len) return error.UnexpectedEof;
        }

        // Read outputs Vec<u8>
        const outputs_len = try reader.readInt(u64, .little);
        const outputs = try allocator.alloc(u8, outputs_len);
        errdefer allocator.free(outputs);
        if (outputs_len > 0) {
            const read_len = try reader.readAll(outputs);
            if (read_len != outputs_len) return error.UnexpectedEof;
        }

        // Read panic bool (1 byte)
        const panic_byte = try reader.readByte();
        const panic_val = panic_byte != 0;

        // Read MemoryLayout
        const memory_layout = try MemoryLayout.deserialize(reader);

        return Self{
            .inputs = inputs,
            .trusted_advice = trusted_advice,
            .untrusted_advice = untrusted_advice,
            .outputs = outputs,
            .panic = panic_val,
            .memory_layout = memory_layout,
            .allocator = allocator,
        };
    }

    /// Deserialize from a file
    pub fn deserializeFromFile(allocator: Allocator, path: []const u8) !Self {
        const file = try std.fs.openFileAbsolute(path, .{ .mode = .read_only });
        defer file.close();

        const stat = try file.stat();
        const data = try allocator.alloc(u8, stat.size);
        defer allocator.free(data);

        const bytes_read = try file.readAll(data);
        if (bytes_read != stat.size) return error.UnexpectedEof;

        return try deserialize(allocator, data);
    }

    pub fn deinit(self: *Self) void {
        if (self.allocator) |alloc| {
            if (self.inputs.len > 0) alloc.free(self.inputs);
            if (self.trusted_advice.len > 0) alloc.free(self.trusted_advice);
            if (self.untrusted_advice.len > 0) alloc.free(self.untrusted_advice);
            if (self.outputs.len > 0) alloc.free(self.outputs);
        }
    }
};

/// Fiat-Shamir preamble matching Jolt's fiat_shamir_preamble
///
/// This appends the same data to the transcript as Jolt does,
/// ensuring identical challenge derivation.
///
/// Reference: jolt-core/src/zkvm/mod.rs (PRs #1358, #1408)
pub fn fiatShamirPreamble(
    comptime F: type,
    transcript: anytype,
    device: *const JoltDevice,
    ram_K: usize,
    trace_length: usize,
    entry_address: u64,
    rw_config: jolt_types.ReadWriteConfig,
    one_hot_config: jolt_types.OneHotConfig,
    dory_layout: u8,
    preprocessing_digest: *const [32]u8,
) void {
    _ = F;

    // Preprocessing digest is appended FIRST (PR #1408)
    transcript.appendBytes("preprocessing_digest", preprocessing_digest);

    transcript.appendU64("max_input_size", device.memory_layout.max_input_size);
    transcript.appendU64("max_output_size", device.memory_layout.max_output_size);
    transcript.appendU64("heap_size", device.memory_layout.heap_size);
    transcript.appendBytes("inputs", device.inputs);
    transcript.appendBytes("outputs", device.outputs);
    transcript.appendU64("panic", if (device.panic) 1 else 0);
    transcript.appendU64("ram_K", @intCast(ram_K));
    transcript.appendU64("trace_length", @intCast(trace_length));
    transcript.appendU64("entry_address", entry_address);

    // Config parameters bound to transcript (PR #1358)
    transcript.appendU64("ram_rw_phase1_num_rounds", @intCast(rw_config.ram_rw_phase1_num_rounds));
    transcript.appendU64("ram_rw_phase2_num_rounds", @intCast(rw_config.ram_rw_phase2_num_rounds));
    transcript.appendU64("registers_rw_phase1_num_rounds", @intCast(rw_config.registers_rw_phase1_num_rounds));
    transcript.appendU64("registers_rw_phase2_num_rounds", @intCast(rw_config.registers_rw_phase2_num_rounds));
    transcript.appendU64("log_k_chunk", @intCast(one_hot_config.log_k_chunk));
    transcript.appendU64("lookups_ra_virtual_log_k_chunk", @intCast(one_hot_config.lookups_ra_virtual_log_k_chunk));
    transcript.appendU64("dory_layout", @intCast(dory_layout));
}

// ============================================================================
// Tests
// ============================================================================

test "memory layout size" {
    // Each field is u64 (8 bytes), and there are 20 fields
    try std.testing.expectEqual(@as(usize, 20 * 8), @sizeOf(MemoryLayout));
}

test "jolt device from emulator" {
    const allocator = std.testing.allocator;

    var device = try JoltDevice.fromEmulator(
        allocator,
        &[_]u8{ 1, 2, 3 },
        &[_]u8{ 4, 5 },
        false,
        4192,
        null, // use default heap_size
    );
    defer device.deinit();

    try std.testing.expectEqual(@as(usize, 3), device.inputs.len);
    try std.testing.expectEqual(@as(usize, 2), device.outputs.len);
    try std.testing.expect(!device.panic);
    try std.testing.expectEqual(@as(u64, 4192), device.memory_layout.program_size);
    try std.testing.expectEqual(@as(u64, 4096), device.memory_layout.max_input_size);
}
