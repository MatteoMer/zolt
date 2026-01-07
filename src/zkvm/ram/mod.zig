//! RAM (Random Access Memory) checking for Jolt
//!
//! This module implements memory checking using offline memory checking techniques.
//! It ensures that all memory reads return the most recently written value.

const std = @import("std");
const Allocator = std.mem.Allocator;
const common = @import("../../common/mod.zig");
const commitment_types = @import("../commitment_types.zig");
const PolyCommitment = commitment_types.PolyCommitment;

// RAF (Read-After-Final) checking
pub const raf_checking = @import("raf_checking.zig");
pub const RafEvaluationParams = raf_checking.RafEvaluationParams;
pub const RaPolynomial = raf_checking.RaPolynomial;
pub const UnmapPolynomial = raf_checking.UnmapPolynomial;
pub const RafEvaluationProver = raf_checking.RafEvaluationProver;
pub const RafEvaluationVerifier = raf_checking.RafEvaluationVerifier;

// Value evaluation checking
pub const val_evaluation = @import("val_evaluation.zig");
pub const ValEvaluationParams = val_evaluation.ValEvaluationParams;
pub const IncPolynomial = val_evaluation.IncPolynomial;
pub const WaPolynomial = val_evaluation.WaPolynomial;
pub const LtPolynomial = val_evaluation.LtPolynomial;
pub const ValEvaluationProver = val_evaluation.ValEvaluationProver;
pub const ValEvaluationVerifier = val_evaluation.ValEvaluationVerifier;

// Output checking (IO region verification)
pub const output_check = @import("output_check.zig");
pub const OutputSumcheckParams = output_check.OutputSumcheckParams;
pub const OutputSumcheckProver = output_check.OutputSumcheckProver;

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

    /// Clone the trace for external use
    /// The caller owns the returned trace and must call deinit() on it.
    pub fn toTrace(self: *const RAMState, allocator: Allocator) !MemoryTrace {
        var new_trace = MemoryTrace.init(allocator);
        errdefer new_trace.deinit();

        // Copy trace entries
        try new_trace.accesses.ensureTotalCapacity(new_trace.allocator, self.trace.accesses.items.len);
        for (self.trace.accesses.items) |entry| {
            try new_trace.accesses.append(new_trace.allocator, entry);
        }

        return new_trace;
    }
};

/// Memory proof for zkVM
///
/// Contains polynomial commitments for memory verification.
/// Uses offline memory checking to ensure read-write consistency.
pub fn MemoryProof(comptime F: type) type {
    return struct {
        const Self = @This();

        /// Commitment to memory value polynomial (address -> value mapping)
        commitment: PolyCommitment,
        /// Read timestamp polynomial commitment
        read_ts_commitment: PolyCommitment,
        /// Write timestamp polynomial commitment
        write_ts_commitment: PolyCommitment,
        /// Commitment to final memory state (for RAF checking)
        final_state_commitment: PolyCommitment,
        /// Opening proof for batch verification (optional)
        opening_proof: ?*commitment_types.OpeningProof,

        /// Legacy field element (for backward compatibility)
        _legacy_commitment: F,

        /// Create a default proof with identity commitments
        pub fn init() Self {
            return .{
                .commitment = PolyCommitment.zero(),
                .read_ts_commitment = PolyCommitment.zero(),
                .write_ts_commitment = PolyCommitment.zero(),
                .final_state_commitment = PolyCommitment.zero(),
                .opening_proof = null,
                ._legacy_commitment = F.zero(),
            };
        }

        /// Create proof with specific commitments
        pub fn withCommitments(
            commitment: PolyCommitment,
            read_ts: PolyCommitment,
            write_ts: PolyCommitment,
            final_state: PolyCommitment,
        ) Self {
            return .{
                .commitment = commitment,
                .read_ts_commitment = read_ts,
                .write_ts_commitment = write_ts,
                .final_state_commitment = final_state,
                .opening_proof = null,
                ._legacy_commitment = F.zero(),
            };
        }

        pub fn deinit(self: *Self, allocator: Allocator) void {
            if (self.opening_proof) |proof| {
                proof.deinit();
                allocator.destroy(proof);
                self.opening_proof = null;
            }
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
