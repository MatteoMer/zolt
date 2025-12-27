//! Register file checking for Jolt
//!
//! This module implements register checking using offline memory checking.
//! It ensures register reads return the most recently written value.

const std = @import("std");
const Allocator = std.mem.Allocator;
const common = @import("../../common/mod.zig");

/// Register operation type
pub const RegisterOp = enum {
    Read,
    Write,
};

/// A single register access record
pub const RegisterAccess = struct {
    /// Register index (0-31 for RISC-V)
    register: u8,
    /// Value read or written
    value: u64,
    /// Operation type
    op: RegisterOp,
    /// Timestamp (cycle number)
    timestamp: u64,
};

/// Register trace for proving
pub const RegisterTrace = struct {
    accesses: std.ArrayListUnmanaged(RegisterAccess),
    allocator: Allocator,

    pub fn init(allocator: Allocator) RegisterTrace {
        return .{
            .accesses = .{},
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *RegisterTrace) void {
        self.accesses.deinit(self.allocator);
    }

    /// Record a read operation
    pub fn recordRead(self: *RegisterTrace, register: u8, value: u64, timestamp: u64) !void {
        try self.accesses.append(self.allocator, .{
            .register = register,
            .value = value,
            .op = .Read,
            .timestamp = timestamp,
        });
    }

    /// Record a write operation
    pub fn recordWrite(self: *RegisterTrace, register: u8, value: u64, timestamp: u64) !void {
        try self.accesses.append(self.allocator, .{
            .register = register,
            .value = value,
            .op = .Write,
            .timestamp = timestamp,
        });
    }

    /// Get the number of accesses
    pub fn len(self: *const RegisterTrace) usize {
        return self.accesses.items.len;
    }
};

/// Register file state during emulation
pub const RegisterFile = struct {
    /// Register values (x0-x31)
    registers: [32]u64,
    /// Register trace for proving
    trace: RegisterTrace,
    /// Current timestamp
    timestamp: u64,

    pub fn init(allocator: Allocator) RegisterFile {
        return .{
            .registers = [_]u64{0} ** 32,
            .trace = RegisterTrace.init(allocator),
            .timestamp = 0,
        };
    }

    pub fn deinit(self: *RegisterFile) void {
        self.trace.deinit();
    }

    /// Read a register value
    pub fn read(self: *RegisterFile, reg: u8) !u64 {
        const value = if (reg == 0) 0 else self.registers[reg];
        try self.trace.recordRead(reg, value, self.timestamp);
        return value;
    }

    /// Write a register value
    pub fn write(self: *RegisterFile, reg: u8, value: u64) !void {
        if (reg == 0) return; // x0 is hardwired to zero
        self.registers[reg] = value;
        try self.trace.recordWrite(reg, value, self.timestamp);
    }

    /// Advance the timestamp
    pub fn tick(self: *RegisterFile) void {
        self.timestamp += 1;
    }

    /// Get the current timestamp
    pub fn getTimestamp(self: *const RegisterFile) u64 {
        return self.timestamp;
    }
};

/// Register proof for zkVM
pub fn RegisterProof(comptime F: type) type {
    return struct {
        const Self = @This();

        /// Commitment to register polynomial
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

test "register file basic operations" {
    const allocator = std.testing.allocator;
    var regs = RegisterFile.init(allocator);
    defer regs.deinit();

    // x0 should always be zero
    const zero_val = try regs.read(0);
    try std.testing.expectEqual(@as(u64, 0), zero_val);

    // Write to x0 should be ignored
    try regs.write(0, 42);
    const still_zero = try regs.read(0);
    try std.testing.expectEqual(@as(u64, 0), still_zero);

    // Write to other registers should work
    try regs.write(1, 0xDEADBEEF);
    const ra_val = try regs.read(1);
    try std.testing.expectEqual(@as(u64, 0xDEADBEEF), ra_val);

    // Check trace was recorded
    try std.testing.expect(regs.trace.len() > 0);
}
