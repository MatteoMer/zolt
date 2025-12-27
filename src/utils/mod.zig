//! Utility functions and types for Jolt
//!
//! This module provides common utilities used throughout the codebase.

const std = @import("std");
const Allocator = std.mem.Allocator;

/// Error types for Jolt
pub const JoltError = error{
    /// Invalid proof
    InvalidProof,
    /// Sumcheck verification failed
    SumcheckVerificationFailed,
    /// Commitment verification failed
    CommitmentVerificationFailed,
    /// Invalid witness
    InvalidWitness,
    /// Memory access out of bounds
    MemoryOutOfBounds,
    /// Invalid instruction
    InvalidInstruction,
    /// Allocation failed
    OutOfMemory,
    /// Invalid input
    InvalidInput,
    /// Serialization error
    SerializationError,
};

/// Compute the ceiling of log2
pub fn log2Ceil(n: usize) usize {
    if (n == 0) return 0;
    if (n == 1) return 0;

    var result: usize = 0;
    var val = n - 1;
    while (val > 0) {
        result += 1;
        val >>= 1;
    }
    return result;
}

/// Check if n is a power of 2
pub fn isPowerOfTwo(n: usize) bool {
    return n != 0 and (n & (n - 1)) == 0;
}

/// Round up to the next power of 2
pub fn nextPowerOfTwo(n: usize) usize {
    if (n == 0) return 1;
    return std.math.ceilPowerOfTwo(usize, n) catch std.math.maxInt(usize);
}

/// Parallel iterator helper
pub fn parallelFor(comptime T: type, slice: []T, context: anytype, comptime func: fn (*T, @TypeOf(context)) void) void {
    // Simple sequential implementation for now
    // TODO: Use std.Thread for parallelism
    for (slice) |*item| {
        func(item, context);
    }
}

/// Timer for profiling
pub const Timer = struct {
    start_time: i128,
    name: []const u8,

    pub fn start(name: []const u8) Timer {
        return .{
            .start_time = std.time.nanoTimestamp(),
            .name = name,
        };
    }

    pub fn stop(self: Timer) i128 {
        const end_time = std.time.nanoTimestamp();
        const elapsed = end_time - self.start_time;
        return elapsed;
    }

    pub fn stopAndPrint(self: Timer) void {
        const elapsed_ns = self.stop();
        const elapsed_ms = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000.0;
        std.debug.print("{s}: {d:.2}ms\n", .{ self.name, elapsed_ms });
    }
};

/// Simple thread pool for parallel operations
pub const ThreadPool = struct {
    threads: []std.Thread,
    allocator: Allocator,

    pub fn init(allocator: Allocator, num_threads: usize) !ThreadPool {
        const threads = try allocator.alloc(std.Thread, num_threads);
        return .{
            .threads = threads,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *ThreadPool) void {
        self.allocator.free(self.threads);
    }

    pub fn numThreads() usize {
        return std.Thread.getCpuCount() catch 1;
    }
};

/// Bit manipulation utilities
pub const BitUtils = struct {
    /// Get bit at position (pos must be < 64)
    pub fn getBit(value: usize, pos: usize) u1 {
        const shift: u6 = @intCast(pos & 63);
        return @intCast((value >> shift) & 1);
    }

    /// Set bit at position (pos must be < 64)
    pub fn setBit(value: usize, pos: usize) usize {
        const shift: u6 = @intCast(pos & 63);
        return value | (@as(usize, 1) << shift);
    }

    /// Clear bit at position (pos must be < 64)
    pub fn clearBit(value: usize, pos: usize) usize {
        const shift: u6 = @intCast(pos & 63);
        return value & ~(@as(usize, 1) << shift);
    }

    /// Count leading zeros
    pub fn clz(value: usize) usize {
        return @clz(value);
    }

    /// Count trailing zeros
    pub fn ctz(value: usize) usize {
        return @ctz(value);
    }

    /// Population count (number of 1 bits)
    pub fn popCount(value: usize) usize {
        return @popCount(value);
    }
};

/// Serialization helpers
pub const Serialize = struct {
    /// Write a u64 in little-endian format
    pub fn writeU64(writer: anytype, value: u64) !void {
        var buf: [8]u8 = undefined;
        std.mem.writeInt(u64, &buf, value, .little);
        try writer.writeAll(&buf);
    }

    /// Read a u64 in little-endian format
    pub fn readU64(reader: anytype) !u64 {
        var buf: [8]u8 = undefined;
        try reader.readNoEof(&buf);
        return std.mem.readInt(u64, &buf, .little);
    }

    /// Write a slice with length prefix
    pub fn writeSlice(writer: anytype, data: []const u8) !void {
        try writeU64(writer, data.len);
        try writer.writeAll(data);
    }

    /// Read a slice with length prefix
    pub fn readSlice(reader: anytype, allocator: Allocator) ![]u8 {
        const len = try readU64(reader);
        const data = try allocator.alloc(u8, @intCast(len));
        try reader.readNoEof(data);
        return data;
    }
};

test "log2Ceil" {
    try std.testing.expectEqual(@as(usize, 0), log2Ceil(1));
    try std.testing.expectEqual(@as(usize, 1), log2Ceil(2));
    try std.testing.expectEqual(@as(usize, 2), log2Ceil(3));
    try std.testing.expectEqual(@as(usize, 2), log2Ceil(4));
    try std.testing.expectEqual(@as(usize, 3), log2Ceil(5));
    try std.testing.expectEqual(@as(usize, 4), log2Ceil(16));
}

test "isPowerOfTwo" {
    try std.testing.expect(isPowerOfTwo(1));
    try std.testing.expect(isPowerOfTwo(2));
    try std.testing.expect(isPowerOfTwo(4));
    try std.testing.expect(isPowerOfTwo(256));
    try std.testing.expect(!isPowerOfTwo(0));
    try std.testing.expect(!isPowerOfTwo(3));
    try std.testing.expect(!isPowerOfTwo(5));
}

test "bit utils" {
    try std.testing.expectEqual(@as(u1, 1), BitUtils.getBit(0b1010, 1));
    try std.testing.expectEqual(@as(u1, 0), BitUtils.getBit(0b1010, 0));
    try std.testing.expectEqual(@as(usize, 0b1011), BitUtils.setBit(0b1010, 0));
    try std.testing.expectEqual(@as(usize, 0b1000), BitUtils.clearBit(0b1010, 1));
}
