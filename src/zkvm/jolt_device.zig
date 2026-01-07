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

    /// Returns the start address of memory (lowest of trusted/untrusted advice)
    /// Matches Jolt's MemoryLayout::get_lowest_address()
    pub fn getLowestAddress(self: *const MemoryLayout) u64 {
        return @min(self.trusted_advice_start, self.untrusted_advice_start);
    }

    /// Remap a physical address to a polynomial index
    /// Returns null if address is 0 (no read/write)
    /// Matches Jolt's remap_address function in zkvm/ram/mod.rs
    pub fn remapAddress(self: *const MemoryLayout, address: u64) ?u64 {
        if (address == 0) {
            return null;
        }

        const lowest_address = self.getLowestAddress();
        if (address >= lowest_address) {
            return (address - lowest_address) / 8;
        } else {
            std.debug.panic("Unexpected address 0x{X}", .{address});
        }
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
    ///
    /// memory_size: The memory size to use. If null, uses 128 MB default.
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
        memory_size_opt: ?u64,
    ) !Self {
        // Use Jolt's default constants (from common/src/constants.rs)
        // These MUST match the values used when Jolt generates the preprocessing!
        const DEFAULT_MAX_INPUT_SIZE: u64 = 4096;
        const DEFAULT_MAX_OUTPUT_SIZE: u64 = 4096;
        const DEFAULT_MAX_TRUSTED_ADVICE_SIZE: u64 = 4096; // Must match Jolt's default
        const DEFAULT_MAX_UNTRUSTED_ADVICE_SIZE: u64 = 4096; // Must match Jolt's default
        const DEFAULT_MEMORY_SIZE: u64 = 1024 * 1024 * 128; // 128 MB
        const DEFAULT_STACK_SIZE: u64 = 4096; // 4 KB
        const RAM_START_ADDRESS: u64 = 0x80000000;

        // Use provided memory_size or default
        const memory_size = memory_size_opt orelse DEFAULT_MEMORY_SIZE;

        // Copy inputs/outputs
        const inputs_copy = try allocator.alloc(u8, inputs.len);
        @memcpy(inputs_copy, inputs);
        const outputs_copy = try allocator.alloc(u8, outputs.len);
        @memcpy(outputs_copy, outputs);

        // Align all sizes to 8 bytes (matching Jolt)
        const max_trusted_advice_size = alignUp(DEFAULT_MAX_TRUSTED_ADVICE_SIZE, 8);
        const max_untrusted_advice_size = alignUp(DEFAULT_MAX_UNTRUSTED_ADVICE_SIZE, 8);
        const max_input_size = alignUp(DEFAULT_MAX_INPUT_SIZE, 8);
        const max_output_size = alignUp(DEFAULT_MAX_OUTPUT_SIZE, 8);
        const stack_size = alignUp(DEFAULT_STACK_SIZE, 8);

        // Calculate I/O region size (includes panic and termination = 16 bytes)
        const io_region_bytes = max_input_size + max_trusted_advice_size +
            max_untrusted_advice_size + max_output_size + 16;

        // Round up to next power of two in words, then convert back to bytes
        const io_region_words = std.math.ceilPowerOfTwo(u64, io_region_bytes / 8) catch @panic("overflow");
        const io_bytes = io_region_words * 8;

        // I/O region is placed BEFORE RAM_START_ADDRESS
        // Place larger advice region first (at lower address)
        var trusted_advice_start: u64 = undefined;
        var trusted_advice_end: u64 = undefined;
        var untrusted_advice_start: u64 = undefined;
        var untrusted_advice_end: u64 = undefined;

        const first_advice_start = RAM_START_ADDRESS - io_bytes;

        if (max_trusted_advice_size >= max_untrusted_advice_size) {
            // Trusted advice goes first (at lower address)
            trusted_advice_start = first_advice_start;
            trusted_advice_end = trusted_advice_start + max_trusted_advice_size;
            untrusted_advice_start = trusted_advice_end;
            untrusted_advice_end = untrusted_advice_start + max_untrusted_advice_size;
        } else {
            // Untrusted advice goes first
            untrusted_advice_start = first_advice_start;
            untrusted_advice_end = untrusted_advice_start + max_untrusted_advice_size;
            trusted_advice_start = untrusted_advice_end;
            trusted_advice_end = trusted_advice_start + max_trusted_advice_size;
        }

        // Input/output region follows the advice regions
        const input_start = @max(trusted_advice_end, untrusted_advice_end);
        const input_end = input_start + max_input_size;
        const output_start = input_end;
        const output_end = output_start + max_output_size;
        const panic_addr = output_end;
        const termination = panic_addr + 8; // 8-byte word for panic
        const io_end = termination + 8; // 8-byte word for termination

        // Stack and memory are placed AFTER RAM_START_ADDRESS
        const program_size_aligned = alignUp(program_size, 8);
        const stack_end = RAM_START_ADDRESS + program_size_aligned;
        const memory_end = stack_end + stack_size + memory_size;

        return Self{
            .inputs = inputs_copy,
            .trusted_advice = &[_]u8{},
            .untrusted_advice = &[_]u8{},
            .outputs = outputs_copy,
            .panic = panic_flag,
            .memory_layout = MemoryLayout{
                .program_size = program_size,
                .max_trusted_advice_size = max_trusted_advice_size,
                .trusted_advice_start = trusted_advice_start,
                .trusted_advice_end = trusted_advice_end,
                .max_untrusted_advice_size = max_untrusted_advice_size,
                .untrusted_advice_start = untrusted_advice_start,
                .untrusted_advice_end = untrusted_advice_end,
                .max_input_size = max_input_size,
                .max_output_size = max_output_size,
                .input_start = input_start,
                .input_end = input_end,
                .output_start = output_start,
                .output_end = output_end,
                .stack_size = stack_size,
                .stack_end = stack_end,
                .memory_size = memory_size,
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

    std.debug.print("\n[ZOLT PREAMBLE] === Fiat-Shamir Preamble Start ===\n", .{});

    // Append memory layout values
    std.debug.print("[ZOLT PREAMBLE] appendU64: max_input_size={d}\n", .{device.memory_layout.max_input_size});
    transcript.appendU64(device.memory_layout.max_input_size);

    std.debug.print("[ZOLT PREAMBLE] appendU64: max_output_size={d}\n", .{device.memory_layout.max_output_size});
    transcript.appendU64(device.memory_layout.max_output_size);

    std.debug.print("[ZOLT PREAMBLE] appendU64: memory_size={d}\n", .{device.memory_layout.memory_size});
    transcript.appendU64(device.memory_layout.memory_size);

    // Append program inputs
    std.debug.print("[ZOLT PREAMBLE] appendBytes: inputs.len={d}\n", .{device.inputs.len});
    if (device.inputs.len > 0 and device.inputs.len <= 32) {
        std.debug.print("[ZOLT PREAMBLE]   inputs={{ ", .{});
        for (device.inputs) |b| std.debug.print("{x:0>2} ", .{b});
        std.debug.print("}}\n", .{});
    }
    transcript.appendBytes(device.inputs);

    // Append program outputs
    std.debug.print("[ZOLT PREAMBLE] appendBytes: outputs.len={d}\n", .{device.outputs.len});
    if (device.outputs.len > 0 and device.outputs.len <= 32) {
        std.debug.print("[ZOLT PREAMBLE]   outputs={{ ", .{});
        for (device.outputs) |b| std.debug.print("{x:0>2} ", .{b});
        std.debug.print("}}\n", .{});
    }
    transcript.appendBytes(device.outputs);

    // Append panic flag
    std.debug.print("[ZOLT PREAMBLE] appendU64: panic={d}\n", .{if (device.panic) @as(u64, 1) else @as(u64, 0)});
    transcript.appendU64(if (device.panic) 1 else 0);

    // Append RAM and trace parameters
    std.debug.print("[ZOLT PREAMBLE] appendU64: ram_K={d}\n", .{ram_K});
    transcript.appendU64(@intCast(ram_K));

    std.debug.print("[ZOLT PREAMBLE] appendU64: trace_length={d}\n", .{trace_length});
    transcript.appendU64(@intCast(trace_length));

    std.debug.print("[ZOLT PREAMBLE] === Fiat-Shamir Preamble End ===\n\n", .{});

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
        null, // use default memory_size
    );
    defer device.deinit();

    try std.testing.expectEqual(@as(usize, 3), device.inputs.len);
    try std.testing.expectEqual(@as(usize, 2), device.outputs.len);
    try std.testing.expect(!device.panic);
    try std.testing.expectEqual(@as(u64, 4192), device.memory_layout.program_size);
    try std.testing.expectEqual(@as(u64, 4096), device.memory_layout.max_input_size);
}
