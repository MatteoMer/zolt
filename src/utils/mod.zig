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

fn monotonicNs() i128 {
    const io: std.Io = std.Io.Threaded.global_single_threaded.io();
    return std.Io.Timestamp.now(io, .boot).nanoseconds;
}

/// Timer for profiling
pub const Timer = struct {
    start_time: i128,
    name: []const u8,

    pub fn start(name: []const u8) Timer {
        return .{
            .start_time = monotonicNs(),
            .name = name,
        };
    }

    pub fn stop(self: Timer) i128 {
        const end_time = monotonicNs();
        const elapsed = end_time - self.start_time;
        return elapsed;
    }

    pub fn stopAndPrint(self: Timer) void {
        const elapsed_ns = self.stop();
        const elapsed_ms = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000.0;
        dbg("{s}: {d:.2}ms\n", .{ self.name, elapsed_ms });
    }
};

// ThreadPool re-exported from thread_pool.zig above

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

/// Simple slice-backed reader providing readInt/readByte/readAll/readSliceAll
/// for use with `anytype` reader parameters in serialization code.
/// Replaces the removed std.io.fixedBufferStream reader.
pub const SliceReader = struct {
    data: []const u8,
    pos: usize = 0,

    pub fn readInt(self: *SliceReader, comptime T: type, endian: std.builtin.Endian) !T {
        const n = @divExact(@typeInfo(T).int.bits, 8);
        if (self.pos + n > self.data.len) return error.EndOfStream;
        const val = std.mem.readInt(T, self.data[self.pos..][0..n], endian);
        self.pos += n;
        return val;
    }

    pub fn readByte(self: *SliceReader) !u8 {
        if (self.pos >= self.data.len) return error.EndOfStream;
        const b = self.data[self.pos];
        self.pos += 1;
        return b;
    }

    pub fn readAll(self: *SliceReader, buf: []u8) !usize {
        const available = @min(buf.len, self.data.len - self.pos);
        @memcpy(buf[0..available], self.data[self.pos..][0..available]);
        self.pos += available;
        return available;
    }

    pub fn readSliceAll(self: *SliceReader, buf: []u8) !void {
        if (self.pos + buf.len > self.data.len) return error.EndOfStream;
        @memcpy(buf, self.data[self.pos..][0..buf.len]);
        self.pos += buf.len;
    }

    pub fn writeAll(self: *SliceReader, buf: []const u8) !void {
        // Compatibility shim: some code passes reader to writeAll (for readNoEof replacement)
        return self.readSliceAll(@constCast(buf));
    }
};

/// Serialization helpers for primitive types
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
        try reader.readSliceAll(&buf);
        return std.mem.readInt(u64, &buf, .little);
    }

    /// Write a u32 in little-endian format
    pub fn writeU32(writer: anytype, value: u32) !void {
        var buf: [4]u8 = undefined;
        std.mem.writeInt(u32, &buf, value, .little);
        try writer.writeAll(&buf);
    }

    /// Read a u32 in little-endian format
    pub fn readU32(reader: anytype) !u32 {
        var buf: [4]u8 = undefined;
        try reader.readSliceAll(&buf);
        return std.mem.readInt(u32, &buf, .little);
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
