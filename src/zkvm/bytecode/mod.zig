//! Bytecode handling for Jolt zkVM
//!
//! This module handles the representation and verification of RISC-V bytecode.

const std = @import("std");
const Allocator = std.mem.Allocator;
const common = @import("../../common/mod.zig");

/// A single bytecode entry
pub const BytecodeEntry = struct {
    /// Program counter address
    address: u64,
    /// Encoded instruction
    instruction: u32,
    /// Decoded opcode
    opcode: u8,
    /// Source register 1
    rs1: u8,
    /// Source register 2
    rs2: u8,
    /// Destination register
    rd: u8,
    /// Immediate value
    imm: i32,
};

/// Bytecode table for the program
pub const BytecodeTable = struct {
    entries: std.ArrayList(BytecodeEntry),
    allocator: Allocator,

    pub fn init(allocator: Allocator) BytecodeTable {
        return .{
            .entries = std.ArrayList(BytecodeEntry).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *BytecodeTable) void {
        self.entries.deinit();
    }

    /// Add a bytecode entry
    pub fn addEntry(self: *BytecodeTable, entry: BytecodeEntry) !void {
        try self.entries.append(entry);
    }

    /// Look up an entry by address
    pub fn lookup(self: *const BytecodeTable, address: u64) ?BytecodeEntry {
        for (self.entries.items) |entry| {
            if (entry.address == address) {
                return entry;
            }
        }
        return null;
    }

    /// Get the number of entries
    pub fn len(self: *const BytecodeTable) usize {
        return self.entries.items.len;
    }
};

/// Bytecode proof for memory checking
pub fn BytecodeProof(comptime F: type) type {
    return struct {
        const Self = @This();

        /// Commitment to bytecode polynomial
        commitment: F,
        /// Opening proof
        opening: F,

        pub fn deinit(_: *Self, _: Allocator) void {
            // Nothing to free for now
        }
    };
}

test "bytecode table" {
    const allocator = std.testing.allocator;
    var table = BytecodeTable.init(allocator);
    defer table.deinit();

    try table.addEntry(.{
        .address = 0x80000000,
        .instruction = 0x00000013, // nop (addi x0, x0, 0)
        .opcode = 0x13,
        .rs1 = 0,
        .rs2 = 0,
        .rd = 0,
        .imm = 0,
    });

    try std.testing.expectEqual(@as(usize, 1), table.len());

    const entry = table.lookup(0x80000000);
    try std.testing.expect(entry != null);
    try std.testing.expectEqual(@as(u32, 0x00000013), entry.?.instruction);
}
