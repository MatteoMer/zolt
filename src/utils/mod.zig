//! Utility functions and types for Jolt
//!
//! This module provides common utilities used throughout the codebase.

const std = @import("std");

const zkvm_debug = @import("../zkvm/debug.zig");
const dbg = zkvm_debug.dbg;

const Allocator = std.mem.Allocator;

/// ExpandingTable for incremental EQ polynomial computation
pub const ExpandingTable = @import("zolt_arith").ExpandingTable;

/// Bit interleaving utilities and LookupBits type (from zolt_arith)
const bits = @import("zolt_arith").bits;
pub const LookupBits = bits.LookupBits;
pub const uninterleaveBits = bits.uninterleaveBits;
pub const interleaveBits = bits.interleaveBits;

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

pub const thread_pool = @import("zolt_pool").thread_pool;
pub const ThreadPool = @import("zolt_pool").ThreadPool;
pub const MonotonicTimer = @import("zolt_pool").MonotonicTimer;

/// Timer for profiling (uses debug.nanoTimestamp with default Io)
pub const Timer = struct {
    start_time: i128,
    name: []const u8,

    pub fn start(name: []const u8) Timer {
        return .{
            .start_time = zkvm_debug.nanoTimestamp(zkvm_debug.defaultIo()),
            .name = name,
        };
    }

    pub fn stop(self: Timer) i128 {
        const end_time = zkvm_debug.nanoTimestamp(zkvm_debug.defaultIo());
        const elapsed = end_time - self.start_time;
        return elapsed;
    }

    pub fn stopAndPrint(self: Timer) void {
        const elapsed_ns = self.stop();
        const elapsed_ms = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000.0;
        dbg("{s}: {d:.2}ms\n", .{ self.name, elapsed_ms });
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

/// Serialization helpers for primitive types using std.Io.Writer / std.Io.Reader.
pub const Serialize = struct {
    pub fn writeU64(writer: *std.Io.Writer, value: u64) !void {
        try writer.writeInt(u64, value, .little);
    }

    pub fn readU64(reader: *std.Io.Reader) !u64 {
        return try reader.takeInt(u64, .little);
    }

    pub fn writeU32(writer: *std.Io.Writer, value: u32) !void {
        try writer.writeInt(u32, value, .little);
    }

    pub fn readU32(reader: *std.Io.Reader) !u32 {
        return try reader.takeInt(u32, .little);
    }

    pub fn writeSlice(writer: *std.Io.Writer, data: []const u8) !void {
        try writer.writeInt(u64, data.len, .little);
        try writer.writeAll(data);
    }

    pub fn readSlice(reader: *std.Io.Reader, allocator: Allocator) ![]u8 {
        const len = try reader.takeInt(u64, .little);
        const data = try allocator.alloc(u8, @intCast(len));
        try reader.readSliceAll(data);
        return data;
    }
};

/// Proof serialization for Jolt proofs (extracted to proof_serializer.zig)
pub const ProofSerializer = @import("proof_serializer.zig").ProofSerializer;

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

test {
    // Run tests from extracted modules
    _ = @import("zolt_arith").bits;
    _ = @import("proof_serializer.zig");
}
