//! Guest interface for Jolt zkVM
//!
//! This module provides the interface for programs running inside the zkVM.
//! It includes I/O functions and utilities for interacting with the host.

const std = @import("std");
const common = @import("../common/mod.zig");

/// Read input data from the host
pub fn readInput(buffer: []u8) usize {
    // In actual guest code, this would read from memory-mapped I/O
    // For now, this is a placeholder
    _ = buffer;
    return 0;
}

/// Write output data to the host
pub fn writeOutput(data: []const u8) void {
    // In actual guest code, this would write to memory-mapped I/O
    _ = data;
}

/// Halt the VM with a panic
pub fn panic(message: []const u8) noreturn {
    // Write panic message
    _ = message;
    // Trigger halt
    @panic("guest panic");
}

/// Halt the VM successfully
pub fn halt() noreturn {
    @panic("guest halt");
}

/// Get the current cycle count (for gas metering)
pub fn cycleCount() u64 {
    // Would read from CSR in actual implementation
    return 0;
}

/// Print a debug message (only in debug builds)
pub fn debugPrint(message: []const u8) void {
    if (@import("builtin").mode == .Debug) {
        _ = message;
        // Would write to debug output
    }
}

/// Read trusted advice from the host
pub fn readTrustedAdvice(buffer: []u8) usize {
    _ = buffer;
    return 0;
}

/// Read untrusted advice from the host
pub fn readUntrustedAdvice(buffer: []u8) usize {
    _ = buffer;
    return 0;
}

/// Commit to a value (public output)
pub fn commit(value: u64) void {
    _ = value;
}

/// Assert a condition (will cause verification failure if false)
pub fn assert(condition: bool) void {
    if (!condition) {
        panic("assertion failed");
    }
}

/// Hint structure for providing non-deterministic advice
pub const Hint = struct {
    data: []const u8,

    pub fn readU64(self: *Hint) !u64 {
        if (self.data.len < 8) return error.HintExhausted;
        const value = std.mem.readInt(u64, self.data[0..8], .little);
        self.data = self.data[8..];
        return value;
    }

    pub fn readU32(self: *Hint) !u32 {
        if (self.data.len < 4) return error.HintExhausted;
        const value = std.mem.readInt(u32, self.data[0..4], .little);
        self.data = self.data[4..];
        return value;
    }

    pub fn readBytes(self: *Hint, len: usize) ![]const u8 {
        if (self.data.len < len) return error.HintExhausted;
        const result = self.data[0..len];
        self.data = self.data[len..];
        return result;
    }
};

test "guest module compiles" {
    // Just verify the module compiles
    _ = readInput;
    _ = writeOutput;
    _ = cycleCount;
}
