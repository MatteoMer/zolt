//! RAM (Random Access Memory) checking for Jolt
//!
//! This module implements memory checking using offline memory checking techniques.
//! It ensures that all memory reads return the most recently written value.

const std = @import("std");
const Allocator = std.mem.Allocator;
const common = @import("../../common/mod.zig");

/// Memory operation type
pub const MemoryOp = enum {
    Read,
    Write,
};

/// A single memory access record
pub const MemoryAccess = struct {
    /// Memory address
    address: u64,
    /// Value read or written
    value: u64,
    /// Operation type
    op: MemoryOp,
    /// Timestamp (cycle number)
    timestamp: u64,
};

/// Memory trace for proving
pub const MemoryTrace = struct {
    accesses: std.ArrayListUnmanaged(MemoryAccess),
    allocator: Allocator,

    pub fn init(allocator: Allocator) MemoryTrace {
        return .{
            .accesses = .{},
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *MemoryTrace) void {
        self.accesses.deinit(self.allocator);
    }

    /// Record a read operation
    pub fn recordRead(self: *MemoryTrace, address: u64, value: u64, timestamp: u64) !void {
        try self.accesses.append(self.allocator, .{
            .address = address,
            .value = value,
            .op = .Read,
            .timestamp = timestamp,
        });
    }

    /// Record a write operation
    pub fn recordWrite(self: *MemoryTrace, address: u64, value: u64, timestamp: u64) !void {
        try self.accesses.append(self.allocator, .{
            .address = address,
            .value = value,
            .op = .Write,
            .timestamp = timestamp,
        });
    }

    /// Get the number of accesses
    pub fn len(self: *const MemoryTrace) usize {
        return self.accesses.items.len;
    }
};

/// RAM state during emulation
pub const RAMState = struct {
    /// Memory contents (sparse representation)
    memory: std.AutoHashMapUnmanaged(u64, u64),
    /// Memory trace for proving
    trace: MemoryTrace,
    allocator: Allocator,

    pub fn init(allocator: Allocator) RAMState {
        return .{
            .memory = .{},
            .trace = MemoryTrace.init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *RAMState) void {
        self.memory.deinit(self.allocator);
        self.trace.deinit();
    }

    /// Read a word from memory
    pub fn read(self: *RAMState, address: u64, timestamp: u64) !u64 {
        const value = self.memory.get(address) orelse 0;
        try self.trace.recordRead(address, value, timestamp);
        return value;
    }

    /// Write a word to memory
    pub fn write(self: *RAMState, address: u64, value: u64, timestamp: u64) !void {
        try self.memory.put(self.allocator, address, value);
        try self.trace.recordWrite(address, value, timestamp);
    }

    /// Read a byte from memory
    pub fn readByte(self: *RAMState, address: u64, timestamp: u64) !u8 {
        const word_addr = address & ~@as(u64, 7);
        const byte_offset = @as(u3, @truncate(address & 7));
        const word = try self.read(word_addr, timestamp);
        return @truncate(word >> (@as(u6, byte_offset) * 8));
    }

    /// Write a byte to memory
    pub fn writeByte(self: *RAMState, address: u64, value: u8, timestamp: u64) !void {
        const word_addr = address & ~@as(u64, 7);
        const byte_offset = @as(u3, @truncate(address & 7));

        var word = self.memory.get(word_addr) orelse 0;
        const mask = @as(u64, 0xFF) << (@as(u6, byte_offset) * 8);
        word = (word & ~mask) | (@as(u64, value) << (@as(u6, byte_offset) * 8));

        try self.memory.put(self.allocator, word_addr, word);
        try self.trace.recordWrite(word_addr, word, timestamp);
    }
};

/// Memory proof for zkVM
pub fn MemoryProof(comptime F: type) type {
    return struct {
        const Self = @This();

        /// Commitment to memory polynomial
        commitment: F,
        /// Read timestamp polynomial commitment
        read_ts_commitment: F,
        /// Write timestamp polynomial commitment
        write_ts_commitment: F,

        pub fn deinit(_: *Self, _: Allocator) void {
            // Nothing to free for now
        }
    };
}

test "ram state basic operations" {
    const allocator = std.testing.allocator;
    var ram = RAMState.init(allocator);
    defer ram.deinit();

    // Write and read
    try ram.write(0x1000, 0xDEADBEEF, 0);
    const value = try ram.read(0x1000, 1);
    try std.testing.expectEqual(@as(u64, 0xDEADBEEF), value);

    // Read uninitialized memory should return 0
    const zero_value = try ram.read(0x2000, 2);
    try std.testing.expectEqual(@as(u64, 0), zero_value);

    // Check trace
    try std.testing.expectEqual(@as(usize, 3), ram.trace.len());
}

test "ram byte operations" {
    const allocator = std.testing.allocator;
    var ram = RAMState.init(allocator);
    defer ram.deinit();

    // Write bytes
    try ram.writeByte(0x1000, 0xAB, 0);
    try ram.writeByte(0x1001, 0xCD, 1);

    // Read bytes back
    const byte0 = try ram.readByte(0x1000, 2);
    const byte1 = try ram.readByte(0x1001, 3);

    try std.testing.expectEqual(@as(u8, 0xAB), byte0);
    try std.testing.expectEqual(@as(u8, 0xCD), byte1);
}
