//! BytecodePCMapper - maps instruction addresses to program counter indices
//!
//! Maps ELF addresses to bytecode array indices for the Jolt verifier.
//! Matches Jolt's BytecodePCMapper: maps (address - RAM_START) / ALIGNMENT
//! to (base_pc, max_inline_seq).

const std = @import("std");
const Allocator = std.mem.Allocator;
const preprocessing = @import("preprocessing.zig");
const JoltInstruction = preprocessing.JoltInstruction;

/// BytecodePCMapper - maps instruction addresses to program counter indices
/// Matches Jolt's BytecodePCMapper: maps (address - RAM_START) / ALIGNMENT to (base_pc, max_inline_seq)
/// RAM_START_ADDRESS = 0x80000000, ALIGNMENT_FACTOR_BYTECODE = 2
pub const BytecodePCMapper = struct {
    const RAM_START_ADDRESS: u64 = 0x80000000;
    const ALIGNMENT_FACTOR: u64 = 2;

    /// Maps (address - base) / alignment to (base_pc, max_inline_seq)
    indices: std.ArrayListUnmanaged(?struct { usize, u16 }),
    allocator: Allocator,
    /// Base bytecode index for termination sequence (LUI, ADDI, SB, JAL).
    /// Set to last_pc + 1 after processing all real instructions.
    /// LUI→termination_base_pc, ADDI→+1, SB→+2, JAL→+3
    termination_base_pc: usize = 0,

    pub fn init(allocator: Allocator) BytecodePCMapper {
        return .{
            .indices = .{},
            .allocator = allocator,
            .termination_base_pc = 0,
        };
    }

    pub fn deinit(self: *BytecodePCMapper) void {
        self.indices.deinit(self.allocator);
    }

    /// Build the PC map from a bytecode array (already has NoOp prepended at index 0)
    pub fn build(self: *BytecodePCMapper, bytecode: []const JoltInstruction) !void {
        if (bytecode.len <= 1) {
            // Only the NoOp, nothing to map
            try self.indices.append(self.allocator, .{ 0, 0 }); // index 0 for NoOp
            return;
        }

        // Find the maximum address to size the indices array
        var max_addr: u64 = 0;
        for (bytecode) |instr| {
            if (instr.variant != .NoOp and instr.variant != .UNIMPL and instr.address > 0) {
                max_addr = @max(max_addr, instr.address);
            }
        }

        if (max_addr == 0) {
            try self.indices.append(self.allocator, .{ 0, 0 });
            return;
        }

        // Size the array: getIndex(max_addr) + 1
        const max_index = getIndex(max_addr) + 1;
        try self.indices.resize(self.allocator, max_index);
        @memset(self.indices.items, null);

        // Index 0 maps to NoOp (pc=0)
        self.indices.items[0] = .{ 0, 0 };

        // Walk through bytecode array (skip index 0 which is NoOp)
        var last_pc: usize = 0;
        for (bytecode[1..]) |instr| {
            if (instr.address == 0) {
                // Padding NoOp or UNIMPL - skip
                continue;
            }
            last_pc += 1;
            const idx = getIndex(instr.address);
            if (idx < self.indices.items.len) {
                if (self.indices.items[idx] == null) {
                    self.indices.items[idx] = .{
                        last_pc,
                        instr.virtual_sequence_remaining orelse 0,
                    };
                }
            }
        }

        // Reserve 4 bytecode entries for the termination sequence (LUI+ADDI+SB+JAL)
        // These come after all real instructions: k=last_pc+1 through k=last_pc+4
        self.termination_base_pc = last_pc + 1;
    }

    /// Convert an ELF address to array index: (address - RAM_START) / ALIGNMENT + 1
    pub fn getIndex(address: u64) usize {
        return @intCast((address - RAM_START_ADDRESS) / ALIGNMENT_FACTOR + 1);
    }

    /// Get the bytecode array index (PC) for a given ELF address and virtual_sequence_remaining.
    /// For NoOp (address=0), returns 0 (NOP entry).
    /// For regular instructions (no virtual sequences), virtual_sequence_remaining = 0.
    pub fn getPC(self: *const BytecodePCMapper, address: u64, virtual_sequence_remaining: u16) usize {
        if (address == 0) return 0;
        const idx = getIndex(address);
        if (idx >= self.indices.items.len) return 0;
        if (self.indices.items[idx]) |entry| {
            const base_pc = entry[0];
            const max_inline_seq = entry[1];
            return base_pc + @as(usize, max_inline_seq - virtual_sequence_remaining);
        }
        return 0;
    }

    /// Get the bytecode index for a termination store virtual instruction.
    /// LUI (vsr=2) → termination_base_pc, ADDI (vsr=1) → +1, SD (vsr=0) → +2
    pub fn getTerminationPC(self: *const BytecodePCMapper, virtual_sequence_remaining: u16) usize {
        return self.termination_base_pc + @as(usize, 2 - virtual_sequence_remaining);
    }

    /// Get the bytecode index for a trace step (convenience function).
    /// Handles NoOp, termination store, termination JAL, and regular instruction cases.
    pub fn getPCForStep(self: *const BytecodePCMapper, step: anytype) usize {
        if (step.is_termination_jal) return self.termination_base_pc + 3; // JAL-to-self
        if (step.is_noop and !step.is_termination_store) return 0; // NOP padding
        if (step.is_termination_store and !step.is_noop) {
            // Real termination store instruction (LUI, ADDI, or SB)
            return self.getTerminationPC(step.virtual_sequence_remaining);
        }
        if (step.is_noop and step.is_termination_store) return 0; // Dummy termination noop → NOP
        return self.getPC(step.pc, step.virtual_sequence_remaining);
    }

    /// Serialize to arkworks format
    pub fn serialize(self: *const BytecodePCMapper, writer: anytype) !void {
        // Vec<Option<(usize, u16)>>
        // Length as u64
        try writer.writeInt(u64, @intCast(self.indices.items.len), .little);

        for (self.indices.items) |maybe_entry| {
            if (maybe_entry) |entry| {
                // Some variant: 1 byte flag + (usize as u64) + u16
                try writer.writeByte(1);
                try writer.writeInt(u64, @intCast(entry[0]), .little);
                try writer.writeInt(u16, entry[1], .little);
            } else {
                // None variant: 1 byte flag
                try writer.writeByte(0);
            }
        }
    }
};
