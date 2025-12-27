//! JoltDevice - Peripheral device for RISC-V emulator I/O
//!
//! This captures all reads from the reserved memory address space for program inputs
//! and all writes to the reserved memory address space for program outputs.
//! The inputs and outputs are part of the public inputs to the proof.

const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const constants = @import("constants.zig");

/// Configuration for memory layout
pub const MemoryConfig = struct {
    max_input_size: u64 = constants.DEFAULT_MAX_INPUT_SIZE,
    max_trusted_advice_size: u64 = constants.DEFAULT_MAX_TRUSTED_ADVICE_SIZE,
    max_untrusted_advice_size: u64 = constants.DEFAULT_MAX_UNTRUSTED_ADVICE_SIZE,
    max_output_size: u64 = constants.DEFAULT_MAX_OUTPUT_SIZE,
    stack_size: u64 = constants.DEFAULT_STACK_SIZE,
    memory_size: u64 = constants.DEFAULT_MEMORY_SIZE,
    program_size: ?u64 = null,

    pub fn default() MemoryConfig {
        return .{};
    }
};

/// Memory layout for the Jolt VM
pub const MemoryLayout = struct {
    /// The total size of the ELF's sections (.text, .data, .rodata, .bss)
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
    /// Stack starts from (RAM_START_ADDRESS + program_size + stack_size) and grows down
    stack_end: u64,
    memory_size: u64,
    /// Heap starts just after the start of the stack
    memory_end: u64,
    panic_addr: u64,
    termination: u64,
    /// End of the memory region containing inputs, outputs, panic bit, and termination bit
    io_end: u64,

    /// Create a new memory layout from configuration
    pub fn init(config: *const MemoryConfig) MemoryLayout {
        const program_size = config.program_size orelse @panic("MemoryLayout requires bytecode size to be set");

        // Must be 8-byte aligned
        const max_trusted_advice_size = alignUp(config.max_trusted_advice_size, 8);
        const max_untrusted_advice_size = alignUp(config.max_untrusted_advice_size, 8);
        const max_input_size = alignUp(config.max_input_size, 8);
        const max_output_size = alignUp(config.max_output_size, 8);
        const stack_size = alignUp(config.stack_size, 8);
        const memory_size = alignUp(config.memory_size, 8);

        // Critical for ValEvaluation and ValFinal sumchecks in RAM
        std.debug.assert(std.math.isPowerOfTwo(max_trusted_advice_size) or max_trusted_advice_size == 0);
        std.debug.assert(std.math.isPowerOfTwo(max_untrusted_advice_size) or max_untrusted_advice_size == 0);

        // Adds 16 to account for panic bit and termination bit
        const io_region_bytes = max_input_size + max_trusted_advice_size + max_untrusted_advice_size + max_output_size + 16;

        // Padded so that the witness index corresponding to input_start has form 0b11...100...0
        const io_region_words = std.math.ceilPowerOfTwo(u64, io_region_bytes / 8) catch @panic("overflow");
        const io_bytes = io_region_words * 8;

        // Place the larger advice region first in memory (at lower address)
        var trusted_advice_start: u64 = undefined;
        var trusted_advice_end: u64 = undefined;
        var untrusted_advice_start: u64 = undefined;
        var untrusted_advice_end: u64 = undefined;

        if (max_trusted_advice_size >= max_untrusted_advice_size) {
            // Trusted advice goes first
            trusted_advice_start = constants.RAM_START_ADDRESS - io_bytes;
            trusted_advice_end = trusted_advice_start + max_trusted_advice_size;
            untrusted_advice_start = trusted_advice_end;
            untrusted_advice_end = untrusted_advice_start + max_untrusted_advice_size;
        } else {
            // Untrusted advice goes first
            untrusted_advice_start = constants.RAM_START_ADDRESS - io_bytes;
            untrusted_advice_end = untrusted_advice_start + max_untrusted_advice_size;
            trusted_advice_start = untrusted_advice_end;
            trusted_advice_end = trusted_advice_start + max_trusted_advice_size;
        }

        const input_start = @max(untrusted_advice_end, trusted_advice_end);
        const input_end = input_start + max_input_size;
        const output_start = input_end;
        const output_end = output_start + max_output_size;
        const panic_addr = output_end;
        const termination = panic_addr + 8;
        const io_end = termination + 8;

        // Stack grows downwards from bytecode_end + stack_size to bytecode_end
        const stack_end = constants.RAM_START_ADDRESS + program_size;
        const stack_start = stack_end + stack_size;

        // Heap grows up from the top of the stack
        const memory_end = stack_start + memory_size;

        return .{
            .program_size = program_size,
            .max_trusted_advice_size = max_trusted_advice_size,
            .max_untrusted_advice_size = max_untrusted_advice_size,
            .max_input_size = max_input_size,
            .max_output_size = max_output_size,
            .trusted_advice_start = trusted_advice_start,
            .trusted_advice_end = trusted_advice_end,
            .untrusted_advice_start = untrusted_advice_start,
            .untrusted_advice_end = untrusted_advice_end,
            .input_start = input_start,
            .input_end = input_end,
            .output_start = output_start,
            .output_end = output_end,
            .stack_size = stack_size,
            .stack_end = stack_end,
            .memory_size = memory_size,
            .memory_end = memory_end,
            .panic_addr = panic_addr,
            .termination = termination,
            .io_end = io_end,
        };
    }

    /// Returns the lowest address in memory
    pub fn getLowestAddress(self: *const MemoryLayout) u64 {
        return @min(self.trusted_advice_start, self.untrusted_advice_start);
    }

    /// Returns the total emulator memory size
    pub fn getTotalMemorySize(self: *const MemoryLayout) u64 {
        return self.memory_size + self.stack_size + constants.STACK_CANARY_SIZE;
    }

    pub fn format(self: *const MemoryLayout, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
        _ = fmt;
        _ = options;
        try writer.print("MemoryLayout{{\n", .{});
        try writer.print("  program_size: {d}\n", .{self.program_size});
        try writer.print("  trusted_advice: 0x{X:0>16} - 0x{X:0>16}\n", .{ self.trusted_advice_start, self.trusted_advice_end });
        try writer.print("  untrusted_advice: 0x{X:0>16} - 0x{X:0>16}\n", .{ self.untrusted_advice_start, self.untrusted_advice_end });
        try writer.print("  input: 0x{X:0>16} - 0x{X:0>16}\n", .{ self.input_start, self.input_end });
        try writer.print("  output: 0x{X:0>16} - 0x{X:0>16}\n", .{ self.output_start, self.output_end });
        try writer.print("  panic: 0x{X:0>16}\n", .{self.panic_addr});
        try writer.print("  termination: 0x{X:0>16}\n", .{self.termination});
        try writer.print("  io_end: 0x{X:0>16}\n", .{self.io_end});
        try writer.print("  stack_end: 0x{X:0>16}\n", .{self.stack_end});
        try writer.print("  memory_end: 0x{X:0>16}\n", .{self.memory_end});
        try writer.print("}}", .{});
    }
};

/// JoltDevice - Peripheral device for program I/O
pub const JoltDevice = struct {
    inputs: std.ArrayListUnmanaged(u8),
    trusted_advice: std.ArrayListUnmanaged(u8),
    untrusted_advice: std.ArrayListUnmanaged(u8),
    outputs: std.ArrayListUnmanaged(u8),
    panic: bool,
    memory_layout: MemoryLayout,
    allocator: Allocator,

    /// Create a new JoltDevice with the given memory configuration
    pub fn init(allocator: Allocator, config: *const MemoryConfig) JoltDevice {
        return .{
            .inputs = .{},
            .trusted_advice = .{},
            .untrusted_advice = .{},
            .outputs = .{},
            .panic = false,
            .memory_layout = MemoryLayout.init(config),
            .allocator = allocator,
        };
    }

    /// Free all allocated memory
    pub fn deinit(self: *JoltDevice) void {
        self.inputs.deinit(self.allocator);
        self.trusted_advice.deinit(self.allocator);
        self.untrusted_advice.deinit(self.allocator);
        self.outputs.deinit(self.allocator);
    }

    /// Load a byte from the given address
    pub fn load(self: *const JoltDevice, address: u64) u8 {
        if (self.isPanic(address)) {
            return if (self.panic) 1 else 0;
        } else if (self.isTermination(address)) {
            return 0; // Termination bit should never be loaded after it is set
        } else if (self.isInput(address)) {
            const internal_address = self.convertReadAddress(address);
            if (self.inputs.items.len <= internal_address) {
                return 0;
            }
            return self.inputs.items[internal_address];
        } else if (self.isTrustedAdvice(address)) {
            const internal_address = self.convertTrustedAdviceReadAddress(address);
            if (self.trusted_advice.items.len <= internal_address) {
                return 0;
            }
            return self.trusted_advice.items[internal_address];
        } else if (self.isUntrustedAdvice(address)) {
            const internal_address = self.convertUntrustedAdviceReadAddress(address);
            if (self.untrusted_advice.items.len <= internal_address) {
                return 0;
            }
            return self.untrusted_advice.items[internal_address];
        } else if (self.isOutput(address)) {
            const internal_address = self.convertWriteAddress(address);
            if (self.outputs.items.len <= internal_address) {
                return 0;
            }
            return self.outputs.items[internal_address];
        } else {
            std.debug.assert(address <= constants.RAM_START_ADDRESS - 8);
            return 0; // zero-padding
        }
    }

    /// Store a byte at the given address
    pub fn store(self: *JoltDevice, address: u64, value: u8) !void {
        if (address == self.memory_layout.panic_addr) {
            self.panic = true;
            return;
        } else if (self.isPanic(address) or self.isTermination(address)) {
            return;
        }

        const internal_address = self.convertWriteAddress(address);
        if (self.outputs.items.len <= internal_address) {
            try self.outputs.resize(self.allocator, internal_address + 1);
        }
        self.outputs.items[internal_address] = value;
    }

    /// Get total size of inputs and outputs
    pub fn size(self: *const JoltDevice) usize {
        return self.inputs.items.len + self.outputs.items.len;
    }

    /// Check if address is in input range
    pub fn isInput(self: *const JoltDevice, address: u64) bool {
        return address >= self.memory_layout.input_start and address < self.memory_layout.input_end;
    }

    /// Check if address is in trusted advice range
    pub fn isTrustedAdvice(self: *const JoltDevice, address: u64) bool {
        return address >= self.memory_layout.trusted_advice_start and address < self.memory_layout.trusted_advice_end;
    }

    /// Check if address is in untrusted advice range
    pub fn isUntrustedAdvice(self: *const JoltDevice, address: u64) bool {
        return address >= self.memory_layout.untrusted_advice_start and address < self.memory_layout.untrusted_advice_end;
    }

    /// Check if address is in output range
    pub fn isOutput(self: *const JoltDevice, address: u64) bool {
        return address >= self.memory_layout.output_start and address < self.memory_layout.termination;
    }

    /// Check if address is the panic address
    pub fn isPanic(self: *const JoltDevice, address: u64) bool {
        return address >= self.memory_layout.panic_addr and address < self.memory_layout.termination;
    }

    /// Check if address is the termination address
    pub fn isTermination(self: *const JoltDevice, address: u64) bool {
        return address >= self.memory_layout.termination and address < self.memory_layout.io_end;
    }

    fn convertReadAddress(self: *const JoltDevice, address: u64) usize {
        return @intCast(address - self.memory_layout.input_start);
    }

    fn convertTrustedAdviceReadAddress(self: *const JoltDevice, address: u64) usize {
        return @intCast(address - self.memory_layout.trusted_advice_start);
    }

    fn convertUntrustedAdviceReadAddress(self: *const JoltDevice, address: u64) usize {
        return @intCast(address - self.memory_layout.untrusted_advice_start);
    }

    fn convertWriteAddress(self: *const JoltDevice, address: u64) usize {
        return @intCast(address - self.memory_layout.output_start);
    }
};

/// Align value up to the given alignment
fn alignUp(val: u64, alignment: u64) u64 {
    if (alignment == 0) return val;
    const rem = val % alignment;
    if (rem == 0) return val;
    return val + (alignment - rem);
}

test "memory layout creation" {
    const config = MemoryConfig{
        .program_size = 1024,
    };
    const layout = MemoryLayout.init(&config);

    // Basic sanity checks
    try std.testing.expect(layout.input_start < layout.input_end);
    try std.testing.expect(layout.output_start < layout.output_end);
    try std.testing.expect(layout.panic_addr < layout.termination);
    try std.testing.expect(layout.termination < layout.io_end);
}

test "jolt device basic operations" {
    const allocator = std.testing.allocator;
    const config = MemoryConfig{
        .program_size = 1024,
    };
    var device = JoltDevice.init(allocator, &config);
    defer device.deinit();

    // Set some input data
    try device.inputs.appendSlice(allocator, &[_]u8{ 1, 2, 3, 4 });

    // Read from input range
    const input_addr = device.memory_layout.input_start;
    try std.testing.expectEqual(@as(u8, 1), device.load(input_addr));
    try std.testing.expectEqual(@as(u8, 2), device.load(input_addr + 1));

    // Read beyond input data should return 0
    try std.testing.expectEqual(@as(u8, 0), device.load(input_addr + 100));
}
