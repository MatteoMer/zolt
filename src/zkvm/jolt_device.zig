//! JoltDevice and MemoryLayout Types for Fiat-Shamir Preamble
//!
//! These types match the Jolt definitions in `common/src/jolt_device.rs`
//! and are used for Fiat-Shamir transcript preamble to ensure compatible
//! challenge derivation between Zolt prover and Jolt verifier.
//!
//! Reference: jolt/common/src/jolt_device.rs

const std = @import("std");
const Allocator = std.mem.Allocator;

/// Memory layout configuration matching Jolt's MemoryLayout
pub const MemoryLayout = struct {
    program_size: u64,
    max_trusted_advice_size: u64,
    trusted_advice_start: u64,
    trusted_advice_end: u64,
    max_untrusted_advice_size: u64,
    untrusted_advice_start: u64,
    untrusted_advice_end: u64,
    max_input_size: u64,
    max_output_size: u64,
    input_start: u64,
    input_end: u64,
    output_start: u64,
    output_end: u64,
    stack_size: u64,
    stack_end: u64,
    memory_size: u64,
    memory_end: u64,
    panic: u64,
    termination: u64,
    io_end: u64,

    /// Deserialize from arkworks canonical format
    pub fn deserialize(reader: anytype) !MemoryLayout {
        return MemoryLayout{
            .program_size = try reader.readInt(u64, .little),
            .max_trusted_advice_size = try reader.readInt(u64, .little),
            .trusted_advice_start = try reader.readInt(u64, .little),
            .trusted_advice_end = try reader.readInt(u64, .little),
            .max_untrusted_advice_size = try reader.readInt(u64, .little),
            .untrusted_advice_start = try reader.readInt(u64, .little),
            .untrusted_advice_end = try reader.readInt(u64, .little),
            .max_input_size = try reader.readInt(u64, .little),
            .max_output_size = try reader.readInt(u64, .little),
            .input_start = try reader.readInt(u64, .little),
            .input_end = try reader.readInt(u64, .little),
            .output_start = try reader.readInt(u64, .little),
            .output_end = try reader.readInt(u64, .little),
            .stack_size = try reader.readInt(u64, .little),
            .stack_end = try reader.readInt(u64, .little),
            .memory_size = try reader.readInt(u64, .little),
            .memory_end = try reader.readInt(u64, .little),
            .panic = try reader.readInt(u64, .little),
            .termination = try reader.readInt(u64, .little),
            .io_end = try reader.readInt(u64, .little),
        };
    }
};

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
    pub fn fromEmulator(
        allocator: Allocator,
        inputs: []const u8,
        outputs: []const u8,
        panic: bool,
        program_size: u64,
    ) !Self {
        // Use Jolt's default constants (from common/src/constants.rs)
        const DEFAULT_MAX_INPUT_SIZE: u64 = 4096;
        const DEFAULT_MAX_OUTPUT_SIZE: u64 = 4096;
        const DEFAULT_MEMORY_SIZE: u64 = 1024 * 1024 * 128; // 128 MB (matches Jolt's EMULATOR_MEMORY_CAPACITY)
        const DEFAULT_STACK_SIZE: u64 = 4096; // 4 KB (matches Jolt's DEFAULT_STACK_SIZE)
        const RAM_START_ADDRESS: u64 = 0x80000000;

        // Copy inputs/outputs
        const inputs_copy = try allocator.alloc(u8, inputs.len);
        @memcpy(inputs_copy, inputs);
        const outputs_copy = try allocator.alloc(u8, outputs.len);
        @memcpy(outputs_copy, outputs);

        // Compute memory layout matching Jolt's MemoryLayout::new
        const program_size_aligned = alignUp(program_size, 8);
        const stack_end = RAM_START_ADDRESS + program_size_aligned + DEFAULT_STACK_SIZE;
        const memory_end = stack_end + DEFAULT_MEMORY_SIZE;

        // IO region starts after memory
        const input_start = memory_end;
        const input_end = input_start + DEFAULT_MAX_INPUT_SIZE;
        const output_start = input_end;
        const output_end = output_start + DEFAULT_MAX_OUTPUT_SIZE;
        const panic_addr = output_end;
        const termination = panic_addr + 1;
        const io_end = termination + 1;

        return Self{
            .inputs = inputs_copy,
            .trusted_advice = &[_]u8{},
            .untrusted_advice = &[_]u8{},
            .outputs = outputs_copy,
            .panic = panic,
            .memory_layout = MemoryLayout{
                .program_size = program_size,
                .max_trusted_advice_size = 0,
                .trusted_advice_start = 0,
                .trusted_advice_end = 0,
                .max_untrusted_advice_size = 0,
                .untrusted_advice_start = 0,
                .untrusted_advice_end = 0,
                .max_input_size = DEFAULT_MAX_INPUT_SIZE,
                .max_output_size = DEFAULT_MAX_OUTPUT_SIZE,
                .input_start = input_start,
                .input_end = input_end,
                .output_start = output_start,
                .output_end = output_end,
                .stack_size = DEFAULT_STACK_SIZE,
                .stack_end = stack_end,
                .memory_size = DEFAULT_MEMORY_SIZE,
                .memory_end = memory_end,
                .panic = panic_addr,
                .termination = termination,
                .io_end = io_end,
            },
            .allocator = allocator,
        };
    }

    fn alignUp(val: u64, alignment: u64) u64 {
        if (alignment == 0) return val;
        return ((val + alignment - 1) / alignment) * alignment;
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
        const panic = panic_byte != 0;

        // Read MemoryLayout
        const memory_layout = try MemoryLayout.deserialize(reader);

        return Self{
            .inputs = inputs,
            .trusted_advice = trusted_advice,
            .untrusted_advice = untrusted_advice,
            .outputs = outputs,
            .panic = panic,
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
pub fn fiatShamirPreamble(
    comptime F: type,
    transcript: anytype,
    device: *const JoltDevice,
    ram_K: usize,
    trace_length: usize,
) void {
    const Blake2bTranscript = @import("../transcripts/blake2b.zig").Blake2bTranscript;

    // Append memory layout values
    transcript.appendU64(device.memory_layout.max_input_size);
    transcript.appendU64(device.memory_layout.max_output_size);
    transcript.appendU64(device.memory_layout.memory_size);

    // Append program inputs
    transcript.appendBytes(device.inputs);

    // Append program outputs
    transcript.appendBytes(device.outputs);

    // Append panic flag
    transcript.appendU64(if (device.panic) 1 else 0);

    // Append RAM and trace parameters
    transcript.appendU64(@intCast(ram_K));
    transcript.appendU64(@intCast(trace_length));

    _ = F;
    _ = Blake2bTranscript;
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
    );
    defer device.deinit();

    try std.testing.expectEqual(@as(usize, 3), device.inputs.len);
    try std.testing.expectEqual(@as(usize, 2), device.outputs.len);
    try std.testing.expect(!device.panic);
    try std.testing.expectEqual(@as(u64, 4192), device.memory_layout.program_size);
    try std.testing.expectEqual(@as(u64, 4096), device.memory_layout.max_input_size);
}
