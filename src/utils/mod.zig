//! Utility functions and types for Jolt
//!
//! This module provides common utilities used throughout the codebase.

const std = @import("std");
const Allocator = std.mem.Allocator;
const lookup_table = @import("../zkvm/lookup_table/mod.zig");

/// A bitvector type used to represent a (substring of a) lookup index.
/// Optimized representation that stores up to 128 bits with explicit length.
///
/// This is the Zig port of Jolt's `LookupBits` type.
pub const LookupBits = struct {
    const Self = @This();

    /// The bits stored as a 128-bit value (little-endian semantics)
    bits: u128,
    /// Number of valid bits in this LookupBits
    len: u8,

    /// Create a new LookupBits from a value and length
    pub fn init(bits_value: u128, bit_len: usize) Self {
        std.debug.assert(bit_len <= 128);

        // Mask to the specified length
        const masked = if (bit_len < 128)
            bits_value % (@as(u128, 1) << @intCast(bit_len))
        else
            bits_value;

        return Self{
            .bits = masked,
            .len = @intCast(bit_len),
        };
    }

    /// Create an empty LookupBits
    pub fn empty() Self {
        return Self{
            .bits = 0,
            .len = 0,
        };
    }

    /// Get the length in bits
    pub fn length(self: Self) usize {
        return @as(usize, self.len);
    }

    /// Convert to u128
    pub fn toU128(self: Self) u128 {
        return self.bits;
    }

    /// Convert to usize
    pub fn toUsize(self: Self) usize {
        return @intCast(self.bits);
    }

    /// Convert to u64
    pub fn toU64(self: Self) u64 {
        return @truncate(self.bits);
    }

    /// Uninterleave bits into (x, y) components
    pub fn uninterleave(self: Self) struct { x: Self, y: Self } {
        const result = lookup_table.uninterleaveBits(self.bits);
        const x_len = self.len / 2;
        const y_len = self.len - x_len;
        return .{
            .x = Self.init(@as(u128, result.x), x_len),
            .y = Self.init(@as(u128, result.y), y_len),
        };
    }

    /// Split into (prefix, suffix) where suffix has the given length
    pub fn split(self: Self, suffix_len: usize) struct { prefix: Self, suffix: Self } {
        std.debug.assert(suffix_len <= self.len);

        const suffix_bits = self.bits % (@as(u128, 1) << @intCast(suffix_len));
        const suffix = Self.init(suffix_bits, suffix_len);

        const prefix_bits = self.bits >> @intCast(suffix_len);
        const prefix_len = @as(usize, self.len) - suffix_len;
        const prefix = Self.init(prefix_bits, prefix_len);

        return .{ .prefix = prefix, .suffix = suffix };
    }

    /// Pop the most significant bit, decrementing length
    pub fn popMsb(self: *Self) u8 {
        if (self.len == 0) return 0;

        const shift: u7 = @intCast(self.len - 1);
        const msb: u8 = @intCast((self.bits >> shift) & 1);
        self.bits = self.bits % (@as(u128, 1) << shift);
        self.len -= 1;
        return msb;
    }

    /// Get the bit at a specific position (0-indexed from LSB)
    pub fn getBit(self: Self, pos: usize) u1 {
        if (pos >= self.len) return 0;
        return @intCast((self.bits >> @intCast(pos)) & 1);
    }

    /// Count trailing zeros
    pub fn trailingZeros(self: Self) u32 {
        if (self.bits == 0) return @as(u32, self.len);
        const tz = @ctz(self.bits);
        return @min(tz, @as(u32, self.len));
    }

    /// Count leading ones (in the context of the bit length)
    pub fn leadingOnes(self: Self) u32 {
        if (self.len == 0) return 0;
        const shifted = self.bits << @intCast(128 - @as(u8, self.len));
        // Count leading ones by inverting and counting leading zeros
        return @clz(~shifted);
    }

    /// Check equality
    pub fn eql(self: Self, other: Self) bool {
        return self.bits == other.bits and self.len == other.len;
    }

    /// Bitwise AND with a mask
    pub fn bitAnd(self: Self, mask: u128) u128 {
        return self.bits & mask;
    }
};

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
///
/// This implementation processes items sequentially for simplicity and determinism.
/// For parallel execution, use ThreadPool or std.Thread directly with explicit
/// work partitioning for your specific use case.
pub fn parallelFor(comptime T: type, slice: []T, context: anytype, comptime func: fn (*T, @TypeOf(context)) void) void {
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
        try reader.readNoEof(&buf);
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
        try reader.readNoEof(&buf);
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
        try reader.readNoEof(data);
        return data;
    }
};

/// Proof serialization for Jolt proofs
/// Binary format for efficient storage and transmission
pub fn ProofSerializer(comptime F: type) type {
    return struct {
        const Self = @This();

        /// Magic number to identify Zolt proof files
        pub const MAGIC: [4]u8 = .{ 'Z', 'O', 'L', 'T' };
        /// Version number
        pub const VERSION: u32 = 1;

        /// Serialize a field element to bytes (little-endian limbs)
        pub fn serializeField(writer: anytype, value: F) !void {
            for (value.limbs) |limb| {
                try Serialize.writeU64(writer, limb);
            }
        }

        /// Deserialize a field element from bytes
        pub fn deserializeField(reader: anytype) !F {
            var limbs: [4]u64 = undefined;
            for (&limbs) |*limb| {
                limb.* = try Serialize.readU64(reader);
            }
            return F{ .limbs = limbs };
        }

        /// Serialize an array of field elements
        pub fn serializeFieldArray(writer: anytype, values: []const F) !void {
            try Serialize.writeU64(writer, values.len);
            for (values) |value| {
                try serializeField(writer, value);
            }
        }

        /// Deserialize an array of field elements
        pub fn deserializeFieldArray(reader: anytype, allocator: Allocator) ![]F {
            const len = try Serialize.readU64(reader);
            const values = try allocator.alloc(F, @intCast(len));
            errdefer allocator.free(values);

            for (values) |*value| {
                value.* = try deserializeField(reader);
            }
            return values;
        }

        /// Write proof header with magic and version
        pub fn writeHeader(writer: anytype) !void {
            try writer.writeAll(&MAGIC);
            try Serialize.writeU32(writer, VERSION);
        }

        /// Read and verify proof header
        pub fn readHeader(reader: anytype) !void {
            var magic: [4]u8 = undefined;
            try reader.readNoEof(&magic);
            if (!std.mem.eql(u8, &magic, &MAGIC)) {
                return error.InvalidProofFormat;
            }

            const version = try Serialize.readU32(reader);
            if (version != VERSION) {
                return error.UnsupportedProofVersion;
            }
        }

        /// Serialize a sumcheck proof
        pub fn serializeSumcheckProof(writer: anytype, proof: anytype) !void {
            // Write claim
            try serializeField(writer, proof.claim);

            // Write number of rounds
            try Serialize.writeU64(writer, proof.rounds.len);

            // Write each round's polynomial coefficients
            for (proof.rounds) |round| {
                try serializeFieldArray(writer, round.poly.coeffs);
            }

            // Write final point
            try serializeFieldArray(writer, proof.final_point);

            // Write final evaluation
            try serializeField(writer, proof.final_eval);
        }

        /// Deserialize a sumcheck proof
        pub fn deserializeSumcheckProof(reader: anytype, allocator: Allocator) !@import("../subprotocols/mod.zig").Sumcheck(F).Proof {
            const poly_mod = @import("../poly/mod.zig");
            const subprotocols = @import("../subprotocols/mod.zig");

            // Read claim
            const claim = try deserializeField(reader);

            // Read number of rounds
            const num_rounds = try Serialize.readU64(reader);
            const rounds = try allocator.alloc(subprotocols.Sumcheck(F).Round, @intCast(num_rounds));
            errdefer allocator.free(rounds);

            // Read each round
            for (rounds) |*round| {
                const coeffs = try deserializeFieldArray(reader, allocator);
                round.* = .{
                    .poly = poly_mod.UniPoly(F){
                        .coeffs = coeffs,
                        .allocator = allocator,
                    },
                };
            }

            // Read final point
            const final_point = try deserializeFieldArray(reader, allocator);

            // Read final evaluation
            const final_eval = try deserializeField(reader);

            return .{
                .claim = claim,
                .rounds = rounds,
                .final_point = final_point,
                .final_eval = final_eval,
                .allocator = allocator,
            };
        }

        /// Serialize an R1CS/Spartan proof
        pub fn serializeR1CSProof(writer: anytype, proof: anytype) !void {
            // Write tau
            try serializeFieldArray(writer, proof.tau);

            // Write sumcheck proof
            try serializeSumcheckProof(writer, proof.sumcheck_proof);

            // Write eval claims
            for (proof.eval_claims) |claim| {
                try serializeField(writer, claim);
            }

            // Write eval point
            try serializeFieldArray(writer, proof.eval_point);
        }

        /// Deserialize an R1CS/Spartan proof
        pub fn deserializeR1CSProof(reader: anytype, allocator: Allocator) !@import("../zkvm/spartan/mod.zig").R1CSProof(F) {
            const spartan = @import("../zkvm/spartan/mod.zig");

            // Read tau
            const tau = try deserializeFieldArray(reader, allocator);
            errdefer allocator.free(tau);

            // Read sumcheck proof
            const sumcheck_proof = try deserializeSumcheckProof(reader, allocator);

            // Read eval claims
            var eval_claims: [3]F = undefined;
            for (&eval_claims) |*claim| {
                claim.* = try deserializeField(reader);
            }

            // Read eval point
            const eval_point = try deserializeFieldArray(reader, allocator);

            return spartan.R1CSProof(F){
                .tau = tau,
                .sumcheck_proof = sumcheck_proof,
                .eval_claims = eval_claims,
                .eval_point = eval_point,
                .allocator = allocator,
            };
        }

        /// Convenience function to serialize a proof to a byte buffer
        pub fn toBytes(allocator: Allocator, proof: anytype) ![]u8 {
            var list: std.ArrayListUnmanaged(u8) = .{};
            errdefer list.deinit(allocator);

            const writer = list.writer();

            // Write header
            try writeHeader(writer);

            // Determine proof type and serialize
            const ProofType = @TypeOf(proof);
            const type_info = @typeInfo(ProofType);

            if (type_info == .@"struct") {
                if (@hasField(ProofType, "sumcheck_proof") and @hasField(ProofType, "tau")) {
                    // R1CS proof
                    try writer.writeByte(1); // Type tag for R1CS
                    try serializeR1CSProof(writer, proof);
                } else if (@hasField(ProofType, "rounds") and @hasField(ProofType, "claim")) {
                    // Sumcheck proof
                    try writer.writeByte(0); // Type tag for Sumcheck
                    try serializeSumcheckProof(writer, proof);
                }
            }

            return list.toOwnedSlice();
        }

        /// Compute a simple hash of proof bytes for integrity verification
        pub fn computeHash(data: []const u8) [32]u8 {
            // Simple Keccak-like mixing (for demonstration)
            // In production, use proper cryptographic hash
            var hash: [32]u8 = .{0} ** 32;

            for (data, 0..) |byte, i| {
                const idx = i % 32;
                hash[idx] ^= byte;
                hash[(idx + 1) % 32] ^= byte >> 4;
            }

            // Mix rounds
            for (0..24) |_| {
                var temp: [32]u8 = undefined;
                for (0..32) |i| {
                    temp[i] = hash[i] ^ hash[(i + 13) % 32] ^ hash[(i + 23) % 32];
                }
                hash = temp;
            }

            return hash;
        }
    };
}

test "proof serialization" {
    const field = @import("../field/mod.zig");
    const F = field.BN254Scalar;

    const PS = ProofSerializer(F);

    // Test field serialization roundtrip
    const original = F.fromU64(12345678);

    var buffer: [256]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buffer);

    try PS.serializeField(fbs.writer(), original);

    fbs.reset();
    const deserialized = try PS.deserializeField(fbs.reader());

    try std.testing.expect(original.eql(deserialized));
}

test "field array serialization" {
    const field = @import("../field/mod.zig");
    const F = field.BN254Scalar;
    const allocator = std.testing.allocator;

    const PS = ProofSerializer(F);

    const original = [_]F{
        F.fromU64(1),
        F.fromU64(2),
        F.fromU64(3),
    };

    var buffer: [512]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buffer);

    try PS.serializeFieldArray(fbs.writer(), &original);

    fbs.reset();
    const deserialized = try PS.deserializeFieldArray(fbs.reader(), allocator);
    defer allocator.free(deserialized);

    try std.testing.expectEqual(original.len, deserialized.len);
    for (0..original.len) |i| {
        try std.testing.expect(original[i].eql(deserialized[i]));
    }
}

test "proof hash computation" {
    const F = @import("../field/mod.zig").BN254Scalar;
    const PS = ProofSerializer(F);

    const data1 = "Hello, Zolt!";
    const data2 = "Hello, Zolt?";

    const hash1 = PS.computeHash(data1);
    const hash2 = PS.computeHash(data2);

    // Different inputs should produce different hashes
    try std.testing.expect(!std.mem.eql(u8, &hash1, &hash2));

    // Same input should produce same hash
    const hash1_again = PS.computeHash(data1);
    try std.testing.expect(std.mem.eql(u8, &hash1, &hash1_again));
}

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

test "LookupBits init" {
    const bits = LookupBits.init(0b1101, 4);
    try std.testing.expectEqual(@as(u128, 0b1101), bits.bits);
    try std.testing.expectEqual(@as(u8, 4), bits.len);
}

test "LookupBits init masks correctly" {
    // Value 0xFF with length 4 should be masked to 0x0F
    const bits = LookupBits.init(0xFF, 4);
    try std.testing.expectEqual(@as(u128, 0x0F), bits.bits);
}

test "LookupBits split" {
    const bits = LookupBits.init(0b11_01_10, 6);
    const result = bits.split(2);

    // suffix should be last 2 bits: 0b10
    try std.testing.expectEqual(@as(u128, 0b10), result.suffix.bits);
    try std.testing.expectEqual(@as(u8, 2), result.suffix.len);

    // prefix should be first 4 bits: 0b1101
    try std.testing.expectEqual(@as(u128, 0b1101), result.prefix.bits);
    try std.testing.expectEqual(@as(u8, 4), result.prefix.len);
}

test "LookupBits popMsb" {
    var bits = LookupBits.init(0b1101, 4);

    // MSB is 1
    const msb1 = bits.popMsb();
    try std.testing.expectEqual(@as(u8, 1), msb1);
    try std.testing.expectEqual(@as(u128, 0b101), bits.bits);
    try std.testing.expectEqual(@as(u8, 3), bits.len);

    // Next MSB is 1
    const msb2 = bits.popMsb();
    try std.testing.expectEqual(@as(u8, 1), msb2);
    try std.testing.expectEqual(@as(u128, 0b01), bits.bits);
    try std.testing.expectEqual(@as(u8, 2), bits.len);

    // Next MSB is 0
    const msb3 = bits.popMsb();
    try std.testing.expectEqual(@as(u8, 0), msb3);
    try std.testing.expectEqual(@as(u128, 0b1), bits.bits);
    try std.testing.expectEqual(@as(u8, 1), bits.len);
}

test "LookupBits uninterleave" {
    // Jolt format: y bits at even positions, x bits at odd positions
    // interleaved: y[0], x[0], y[1], x[1], ... = 0b11_10_01_00
    // y = 0b1010 = 10, x = 0b1100 = 12
    const bits = LookupBits.init(0b11_10_01_00, 8);
    const result = bits.uninterleave();

    try std.testing.expectEqual(@as(u128, 0b1100), result.x.bits);
    try std.testing.expectEqual(@as(u128, 0b1010), result.y.bits);
}

test "LookupBits trailingZeros" {
    const bits1 = LookupBits.init(0b1000, 4);
    try std.testing.expectEqual(@as(u32, 3), bits1.trailingZeros());

    const bits2 = LookupBits.init(0b1001, 4);
    try std.testing.expectEqual(@as(u32, 0), bits2.trailingZeros());

    const bits3 = LookupBits.init(0, 4);
    try std.testing.expectEqual(@as(u32, 4), bits3.trailingZeros());
}

test "LookupBits getBit" {
    const bits = LookupBits.init(0b1010, 4);
    try std.testing.expectEqual(@as(u1, 0), bits.getBit(0));
    try std.testing.expectEqual(@as(u1, 1), bits.getBit(1));
    try std.testing.expectEqual(@as(u1, 0), bits.getBit(2));
    try std.testing.expectEqual(@as(u1, 1), bits.getBit(3));
    try std.testing.expectEqual(@as(u1, 0), bits.getBit(4)); // Out of bounds
}
