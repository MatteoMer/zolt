//! Proof Serialization and Deserialization
//!
//! This module provides binary serialization for Jolt proofs, enabling:
//! - Saving proofs to disk for later verification
//! - Transmitting proofs over the network
//! - Compact binary representation
//!
//! ## Format
//!
//! The serialization format is designed to be:
//! - Deterministic: Same proof always produces same bytes
//! - Compact: No redundant information
//! - Self-describing: Includes length prefixes for variable-length data
//!
//! ## Binary Layout
//!
//! ```
//! JoltProof:
//!   [4 bytes] magic: "ZOLT"
//!   [4 bytes] version: u32 (currently 1)
//!   [BytecodeProof] bytecode_proof
//!   [MemoryProof] memory_proof
//!   [RegisterProof] register_proof
//!   [R1CSProof] r1cs_proof
//!   [1 byte] has_stage_proofs: bool
//!   [JoltStageProofs?] stage_proofs (if has_stage_proofs)
//!
//! JoltStageProofs:
//!   [8 bytes] log_t: u64
//!   [8 bytes] log_k: u64
//!   [6 x StageProof] stage_proofs
//!
//! StageProof:
//!   [8 bytes] num_round_polys: u64
//!   For each round poly:
//!     [8 bytes] poly_len: u64
//!     [poly_len x 32 bytes] coefficients
//!   [8 bytes] num_challenges: u64
//!   [num_challenges x 32 bytes] challenges
//!   [8 bytes] num_claims: u64
//!   [num_claims x 32 bytes] claims
//! ```

const std = @import("std");
const Allocator = std.mem.Allocator;

const prover = @import("prover.zig");
const bytecode = @import("bytecode/mod.zig");
const ram = @import("ram/mod.zig");
const registers = @import("registers/mod.zig");
const spartan = @import("spartan/mod.zig");
const commitment_types = @import("commitment_types.zig");
const field = @import("../field/mod.zig");

/// Magic bytes for Zolt proof files
pub const MAGIC: [4]u8 = .{ 'Z', 'O', 'L', 'T' };

/// Current serialization format version
pub const VERSION: u32 = 1;

/// Serialization errors
pub const SerializationError = error{
    InvalidMagic,
    UnsupportedVersion,
    UnexpectedEof,
    InvalidData,
    OutOfMemory,
};

/// Writer for serializing proofs
pub fn ProofWriter(comptime F: type) type {
    return struct {
        const Self = @This();

        buffer: std.ArrayListUnmanaged(u8),
        allocator: Allocator,

        pub fn init(allocator: Allocator) Self {
            return .{
                .buffer = .{},
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            self.buffer.deinit(self.allocator);
        }

        /// Get the serialized bytes
        pub fn toOwnedSlice(self: *Self) ![]u8 {
            return self.buffer.toOwnedSlice(self.allocator);
        }

        /// Get the current buffer as a slice (borrowed)
        pub fn bytes(self: *const Self) []const u8 {
            return self.buffer.items;
        }

        /// Write bytes directly
        pub fn writeBytes(self: *Self, data: []const u8) !void {
            try self.buffer.appendSlice(self.allocator, data);
        }

        /// Write a u8
        pub fn writeU8(self: *Self, value: u8) !void {
            try self.buffer.append(self.allocator, value);
        }

        /// Write a u32 in little-endian
        pub fn writeU32(self: *Self, value: u32) !void {
            const bytes_arr = std.mem.toBytes(std.mem.nativeToLittle(u32, value));
            try self.buffer.appendSlice(self.allocator, &bytes_arr);
        }

        /// Write a u64 in little-endian
        pub fn writeU64(self: *Self, value: u64) !void {
            const bytes_arr = std.mem.toBytes(std.mem.nativeToLittle(u64, value));
            try self.buffer.appendSlice(self.allocator, &bytes_arr);
        }

        /// Write a field element (32 bytes, little-endian limbs)
        pub fn writeFieldElement(self: *Self, elem: F) !void {
            const bytes_arr = elem.toBytes();
            try self.buffer.appendSlice(self.allocator, &bytes_arr);
        }

        /// Write a polynomial commitment (G1 point)
        pub fn writeCommitment(self: *Self, c: commitment_types.PolyCommitment) !void {
            const bytes_arr = c.toBytes();
            try self.buffer.appendSlice(self.allocator, &bytes_arr);
        }

        /// Write a stage proof
        pub fn writeStageProof(self: *Self, stage: *const prover.StageProof(F)) !void {
            // Write round polynomials
            try self.writeU64(stage.round_polys.items.len);
            for (stage.round_polys.items) |poly| {
                try self.writeU64(poly.len);
                for (poly) |coeff| {
                    try self.writeFieldElement(coeff);
                }
            }

            // Write challenges
            try self.writeU64(stage.challenges.items.len);
            for (stage.challenges.items) |challenge| {
                try self.writeFieldElement(challenge);
            }

            // Write final claims
            try self.writeU64(stage.final_claims.items.len);
            for (stage.final_claims.items) |claim| {
                try self.writeFieldElement(claim);
            }
        }

        /// Write JoltStageProofs
        pub fn writeJoltStageProofs(self: *Self, proofs: *const prover.JoltStageProofs(F)) !void {
            try self.writeU64(proofs.log_t);
            try self.writeU64(proofs.log_k);

            for (&proofs.stage_proofs) |*stage| {
                try self.writeStageProof(stage);
            }
        }
    };
}

/// Reader for deserializing proofs
pub fn ProofReader(comptime F: type) type {
    return struct {
        const Self = @This();

        data: []const u8,
        pos: usize,
        allocator: Allocator,

        pub fn init(allocator: Allocator, data: []const u8) Self {
            return .{
                .data = data,
                .pos = 0,
                .allocator = allocator,
            };
        }

        /// Read exactly n bytes
        pub fn readBytes(self: *Self, n: usize) SerializationError![]const u8 {
            if (self.pos + n > self.data.len) {
                return SerializationError.UnexpectedEof;
            }
            const result = self.data[self.pos .. self.pos + n];
            self.pos += n;
            return result;
        }

        /// Read a u8
        pub fn readU8(self: *Self) SerializationError!u8 {
            const bytes_slice = try self.readBytes(1);
            return bytes_slice[0];
        }

        /// Read a u32 in little-endian
        pub fn readU32(self: *Self) SerializationError!u32 {
            const bytes_slice = try self.readBytes(4);
            return std.mem.littleToNative(u32, std.mem.bytesToValue(u32, bytes_slice[0..4]));
        }

        /// Read a u64 in little-endian
        pub fn readU64(self: *Self) SerializationError!u64 {
            const bytes_slice = try self.readBytes(8);
            return std.mem.littleToNative(u64, std.mem.bytesToValue(u64, bytes_slice[0..8]));
        }

        /// Read a field element (32 bytes)
        pub fn readFieldElement(self: *Self) SerializationError!F {
            const bytes_slice = try self.readBytes(32);
            return F.fromBytes(bytes_slice);
        }

        /// Read a polynomial commitment
        pub fn readCommitment(self: *Self) SerializationError!commitment_types.PolyCommitment {
            const bytes_slice = try self.readBytes(64);
            return commitment_types.PolyCommitment.fromBytes(bytes_slice[0..64].*);
        }

        /// Read a stage proof
        pub fn readStageProof(self: *Self) (SerializationError || Allocator.Error)!prover.StageProof(F) {
            var stage = prover.StageProof(F).init(self.allocator);
            errdefer stage.deinit();

            // Read round polynomials
            const num_polys = try self.readU64();
            for (0..num_polys) |_| {
                const poly_len = try self.readU64();
                const poly = try self.allocator.alloc(F, poly_len);
                errdefer self.allocator.free(poly);

                for (poly) |*coeff| {
                    coeff.* = try self.readFieldElement();
                }
                try stage.round_polys.append(self.allocator, poly);
            }

            // Read challenges
            const num_challenges = try self.readU64();
            for (0..num_challenges) |_| {
                const challenge = try self.readFieldElement();
                try stage.challenges.append(self.allocator, challenge);
            }

            // Read final claims
            const num_claims = try self.readU64();
            for (0..num_claims) |_| {
                const claim = try self.readFieldElement();
                try stage.final_claims.append(self.allocator, claim);
            }

            return stage;
        }

        /// Read JoltStageProofs
        pub fn readJoltStageProofs(self: *Self) (SerializationError || Allocator.Error)!prover.JoltStageProofs(F) {
            const log_t = try self.readU64();
            const log_k = try self.readU64();

            var proofs = prover.JoltStageProofs(F).init(self.allocator);
            errdefer proofs.deinit();

            proofs.log_t = log_t;
            proofs.log_k = log_k;

            for (&proofs.stage_proofs) |*stage| {
                stage.deinit(); // Free the initialized empty stage
                stage.* = try self.readStageProof();
            }

            return proofs;
        }
    };
}

/// Serialize a complete JoltProof to bytes
pub fn serializeProof(comptime F: type, allocator: Allocator, proof: anytype) ![]u8 {
    var writer = ProofWriter(F).init(allocator);
    defer writer.deinit();

    // Write header
    try writer.writeBytes(&MAGIC);
    try writer.writeU32(VERSION);

    // Write bytecode proof
    try writer.writeCommitment(proof.bytecode_proof.commitment);
    try writer.writeCommitment(proof.bytecode_proof.read_ts_commitment);
    try writer.writeCommitment(proof.bytecode_proof.write_ts_commitment);
    try writer.writeFieldElement(proof.bytecode_proof._legacy_commitment);

    // Write memory proof
    try writer.writeCommitment(proof.memory_proof.commitment);
    try writer.writeCommitment(proof.memory_proof.final_state_commitment);
    try writer.writeCommitment(proof.memory_proof.read_ts_commitment);
    try writer.writeCommitment(proof.memory_proof.write_ts_commitment);

    // Write register proof
    try writer.writeCommitment(proof.register_proof.commitment);
    try writer.writeCommitment(proof.register_proof.final_state_commitment);
    try writer.writeCommitment(proof.register_proof.read_ts_commitment);
    try writer.writeCommitment(proof.register_proof.write_ts_commitment);

    // Write R1CS proof
    // tau
    try writer.writeU64(proof.r1cs_proof.tau.len);
    for (proof.r1cs_proof.tau) |e| {
        try writer.writeFieldElement(e);
    }
    // eval_claims [3]
    for (proof.r1cs_proof.eval_claims) |e| {
        try writer.writeFieldElement(e);
    }
    // eval_point
    try writer.writeU64(proof.r1cs_proof.eval_point.len);
    for (proof.r1cs_proof.eval_point) |e| {
        try writer.writeFieldElement(e);
    }
    // sumcheck_proof
    try writer.writeFieldElement(proof.r1cs_proof.sumcheck_proof.claim);
    try writer.writeFieldElement(proof.r1cs_proof.sumcheck_proof.final_eval);
    try writer.writeU64(proof.r1cs_proof.sumcheck_proof.rounds.len);
    // Note: sumcheck rounds serialization is simplified for now
    try writer.writeU64(proof.r1cs_proof.sumcheck_proof.final_point.len);
    for (proof.r1cs_proof.sumcheck_proof.final_point) |e| {
        try writer.writeFieldElement(e);
    }

    // Write stage proofs (optional)
    if (proof.stage_proofs) |stage_proofs| {
        try writer.writeU8(1); // has_stage_proofs = true
        try writer.writeJoltStageProofs(&stage_proofs);
    } else {
        try writer.writeU8(0); // has_stage_proofs = false
    }

    return writer.toOwnedSlice();
}

/// Deserialize a JoltProof from bytes
pub fn deserializeProof(comptime F: type, allocator: Allocator, data: []const u8) !@import("mod.zig").JoltProof(F) {
    const zkvm = @import("mod.zig");
    var reader = ProofReader(F).init(allocator, data);

    // Read and verify header
    const magic = try reader.readBytes(4);
    if (!std.mem.eql(u8, magic, &MAGIC)) {
        return SerializationError.InvalidMagic;
    }

    const version = try reader.readU32();
    if (version != VERSION) {
        return SerializationError.UnsupportedVersion;
    }

    // Read bytecode proof
    var bc_proof = bytecode.BytecodeProof(F).init();
    bc_proof.commitment = try reader.readCommitment();
    bc_proof.read_ts_commitment = try reader.readCommitment();
    bc_proof.write_ts_commitment = try reader.readCommitment();
    bc_proof._legacy_commitment = try reader.readFieldElement();

    // Read memory proof
    var mem_proof = ram.MemoryProof(F).init();
    mem_proof.commitment = try reader.readCommitment();
    mem_proof.final_state_commitment = try reader.readCommitment();
    mem_proof.read_ts_commitment = try reader.readCommitment();
    mem_proof.write_ts_commitment = try reader.readCommitment();

    // Read register proof
    var reg_proof = registers.RegisterProof(F).init();
    reg_proof.commitment = try reader.readCommitment();
    reg_proof.final_state_commitment = try reader.readCommitment();
    reg_proof.read_ts_commitment = try reader.readCommitment();
    reg_proof.write_ts_commitment = try reader.readCommitment();

    // Read R1CS proof
    // tau
    const tau_len = try reader.readU64();
    const tau = try allocator.alloc(F, tau_len);
    errdefer allocator.free(tau);
    for (tau) |*e| {
        e.* = try reader.readFieldElement();
    }
    // eval_claims [3]
    var eval_claims: [3]F = undefined;
    for (&eval_claims) |*e| {
        e.* = try reader.readFieldElement();
    }
    // eval_point
    const eval_point_len = try reader.readU64();
    const eval_point = try allocator.alloc(F, eval_point_len);
    errdefer allocator.free(eval_point);
    for (eval_point) |*e| {
        e.* = try reader.readFieldElement();
    }
    // sumcheck_proof
    const sc_claim = try reader.readFieldElement();
    const sc_final_eval = try reader.readFieldElement();
    const sc_rounds_len = try reader.readU64();
    _ = sc_rounds_len; // Rounds are empty for now
    const sc_final_point_len = try reader.readU64();
    const sc_final_point = try allocator.alloc(F, sc_final_point_len);
    errdefer allocator.free(sc_final_point);
    for (sc_final_point) |*e| {
        e.* = try reader.readFieldElement();
    }

    // Create empty rounds array
    const sc_rounds = try allocator.alloc(@import("../subprotocols/mod.zig").Sumcheck(F).Round, 0);

    const r1cs_proof = spartan.R1CSProof(F){
        .tau = tau,
        .sumcheck_proof = .{
            .claim = sc_claim,
            .rounds = sc_rounds,
            .final_point = sc_final_point,
            .final_eval = sc_final_eval,
            .allocator = allocator,
        },
        .eval_claims = eval_claims,
        .eval_point = eval_point,
        .allocator = allocator,
    };

    // Read stage proofs
    const has_stage_proofs = try reader.readU8();
    var stage_proofs: ?prover.JoltStageProofs(F) = null;
    if (has_stage_proofs != 0) {
        stage_proofs = try reader.readJoltStageProofs();
    }

    return zkvm.JoltProof(F){
        .bytecode_proof = bc_proof,
        .memory_proof = mem_proof,
        .register_proof = reg_proof,
        .r1cs_proof = r1cs_proof,
        .stage_proofs = stage_proofs,
        .allocator = allocator,
    };
}

/// Write a proof to a file
pub fn writeProofToFile(comptime F: type, allocator: Allocator, proof: anytype, path: []const u8) !void {
    const bytes = try serializeProof(F, allocator, proof);
    defer allocator.free(bytes);

    const file = try std.fs.cwd().createFile(path, .{});
    defer file.close();

    try file.writeAll(bytes);
}

/// Read a proof from a file
pub fn readProofFromFile(comptime F: type, allocator: Allocator, path: []const u8) !@import("mod.zig").JoltProof(F) {
    const file = try std.fs.cwd().openFile(path, .{});
    defer file.close();

    const stat = try file.stat();
    const data = try allocator.alloc(u8, stat.size);
    defer allocator.free(data);

    const bytes_read = try file.readAll(data);
    if (bytes_read != stat.size) {
        return SerializationError.UnexpectedEof;
    }

    return deserializeProof(F, allocator, data);
}

// ============================================================================
// Tests
// ============================================================================

test "serialize and deserialize empty stage proof" {
    const F = field.BN254Scalar;
    const allocator = std.testing.allocator;

    // Create an empty stage proof
    var stage = prover.StageProof(F).init(allocator);
    defer stage.deinit();

    // Serialize
    var writer = ProofWriter(F).init(allocator);
    defer writer.deinit();
    try writer.writeStageProof(&stage);

    // Deserialize
    var reader = ProofReader(F).init(allocator, writer.bytes());
    var deserialized = try reader.readStageProof();
    defer deserialized.deinit();

    // Verify
    try std.testing.expectEqual(@as(usize, 0), deserialized.round_polys.items.len);
    try std.testing.expectEqual(@as(usize, 0), deserialized.challenges.items.len);
    try std.testing.expectEqual(@as(usize, 0), deserialized.final_claims.items.len);
}

test "serialize and deserialize stage proof with data" {
    const F = field.BN254Scalar;
    const allocator = std.testing.allocator;

    // Create a stage proof with data
    var stage = prover.StageProof(F).init(allocator);
    defer stage.deinit();

    // Add round polynomials
    const poly1 = try allocator.alloc(F, 3);
    poly1[0] = F.fromU64(1);
    poly1[1] = F.fromU64(2);
    poly1[2] = F.fromU64(3);
    try stage.round_polys.append(allocator, poly1);

    const poly2 = try allocator.alloc(F, 2);
    poly2[0] = F.fromU64(10);
    poly2[1] = F.fromU64(20);
    try stage.round_polys.append(allocator, poly2);

    // Add challenges
    try stage.challenges.append(allocator, F.fromU64(100));
    try stage.challenges.append(allocator, F.fromU64(200));

    // Add claims
    try stage.final_claims.append(allocator, F.fromU64(999));

    // Serialize
    var writer = ProofWriter(F).init(allocator);
    defer writer.deinit();
    try writer.writeStageProof(&stage);

    // Deserialize
    var reader = ProofReader(F).init(allocator, writer.bytes());
    var deserialized = try reader.readStageProof();
    defer deserialized.deinit();

    // Verify round polynomials
    try std.testing.expectEqual(@as(usize, 2), deserialized.round_polys.items.len);
    try std.testing.expectEqual(@as(usize, 3), deserialized.round_polys.items[0].len);
    try std.testing.expect(deserialized.round_polys.items[0][0].eql(F.fromU64(1)));
    try std.testing.expect(deserialized.round_polys.items[0][1].eql(F.fromU64(2)));
    try std.testing.expect(deserialized.round_polys.items[0][2].eql(F.fromU64(3)));
    try std.testing.expectEqual(@as(usize, 2), deserialized.round_polys.items[1].len);
    try std.testing.expect(deserialized.round_polys.items[1][0].eql(F.fromU64(10)));

    // Verify challenges
    try std.testing.expectEqual(@as(usize, 2), deserialized.challenges.items.len);
    try std.testing.expect(deserialized.challenges.items[0].eql(F.fromU64(100)));
    try std.testing.expect(deserialized.challenges.items[1].eql(F.fromU64(200)));

    // Verify claims
    try std.testing.expectEqual(@as(usize, 1), deserialized.final_claims.items.len);
    try std.testing.expect(deserialized.final_claims.items[0].eql(F.fromU64(999)));
}

test "serialize and deserialize JoltStageProofs" {
    const F = field.BN254Scalar;
    const allocator = std.testing.allocator;

    // Create stage proofs
    var proofs = prover.JoltStageProofs(F).init(allocator);
    defer proofs.deinit();

    proofs.log_t = 10;
    proofs.log_k = 16;

    // Add some data to stage 0
    const poly = try allocator.alloc(F, 2);
    poly[0] = F.fromU64(42);
    poly[1] = F.fromU64(43);
    try proofs.stage_proofs[0].round_polys.append(allocator, poly);
    try proofs.stage_proofs[0].challenges.append(allocator, F.fromU64(123));

    // Serialize
    var writer = ProofWriter(F).init(allocator);
    defer writer.deinit();
    try writer.writeJoltStageProofs(&proofs);

    // Deserialize
    var reader = ProofReader(F).init(allocator, writer.bytes());
    var deserialized = try reader.readJoltStageProofs();
    defer deserialized.deinit();

    // Verify
    try std.testing.expectEqual(@as(usize, 10), deserialized.log_t);
    try std.testing.expectEqual(@as(usize, 16), deserialized.log_k);
    try std.testing.expectEqual(@as(usize, 1), deserialized.stage_proofs[0].round_polys.items.len);
    try std.testing.expect(deserialized.stage_proofs[0].round_polys.items[0][0].eql(F.fromU64(42)));
    try std.testing.expect(deserialized.stage_proofs[0].challenges.items[0].eql(F.fromU64(123)));
}

test "magic and version validation" {
    const F = field.BN254Scalar;
    const allocator = std.testing.allocator;

    // Invalid magic
    {
        const bad_data = [_]u8{ 'B', 'A', 'D', '!' } ++ [_]u8{0} ** 100;
        const result = deserializeProof(F, allocator, &bad_data);
        try std.testing.expectError(SerializationError.InvalidMagic, result);
    }

    // Invalid version
    {
        var bad_data: [108]u8 = undefined;
        @memcpy(bad_data[0..4], &MAGIC);
        const version_bytes = std.mem.toBytes(std.mem.nativeToLittle(u32, 99));
        @memcpy(bad_data[4..8], &version_bytes);
        const result = deserializeProof(F, allocator, &bad_data);
        try std.testing.expectError(SerializationError.UnsupportedVersion, result);
    }
}

test "field element roundtrip" {
    const F = field.BN254Scalar;
    const allocator = std.testing.allocator;

    var writer = ProofWriter(F).init(allocator);
    defer writer.deinit();

    // Write some field elements
    const elem1 = F.fromU64(12345678901234567890);
    const elem2 = F.one();
    const elem3 = F.zero();

    try writer.writeFieldElement(elem1);
    try writer.writeFieldElement(elem2);
    try writer.writeFieldElement(elem3);

    // Read them back
    var reader = ProofReader(F).init(allocator, writer.bytes());

    const read1 = try reader.readFieldElement();
    const read2 = try reader.readFieldElement();
    const read3 = try reader.readFieldElement();

    try std.testing.expect(read1.eql(elem1));
    try std.testing.expect(read2.eql(elem2));
    try std.testing.expect(read3.eql(elem3));
}

test "serialize and deserialize full JoltProof" {
    const F = field.BN254Scalar;
    const allocator = std.testing.allocator;
    const zkvm = @import("mod.zig");

    // Create a minimal bytecode program (li a0, 42; li a1, 10; add a0, a0, a1; ebreak)
    const program = [_]u8{
        0x13, 0x05, 0xa0, 0x02, // li a0, 42
        0x93, 0x05, 0xa0, 0x00, // li a1, 10
        0x33, 0x05, 0xb5, 0x00, // add a0, a0, a1
        0x73, 0x00, 0x10, 0x00, // ebreak
    };

    // Create a prover
    var prover_inst = zkvm.JoltProver(F).init(allocator);
    prover_inst.max_cycles = 64;

    // Generate a proof
    var proof = try prover_inst.prove(&program, &[_]u8{});
    defer proof.deinit();

    // Verify the proof was generated
    try std.testing.expect(proof.stage_proofs != null);

    // Serialize the proof
    const serialized = try serializeProof(F, allocator, proof);
    defer allocator.free(serialized);

    // Check the serialized data starts with magic
    try std.testing.expect(std.mem.eql(u8, serialized[0..4], &MAGIC));

    // Deserialize the proof
    var deserialized = try deserializeProof(F, allocator, serialized);
    defer deserialized.deinit();

    // Verify the deserialized proof matches the original
    try std.testing.expect(deserialized.stage_proofs != null);

    // Compare stage proofs
    const orig_stages = proof.stage_proofs.?;
    const deser_stages = deserialized.stage_proofs.?;

    try std.testing.expectEqual(orig_stages.log_t, deser_stages.log_t);
    try std.testing.expectEqual(orig_stages.log_k, deser_stages.log_k);

    // Compare each stage's round polys
    for (orig_stages.stage_proofs, deser_stages.stage_proofs) |orig, deser| {
        try std.testing.expectEqual(orig.round_polys.items.len, deser.round_polys.items.len);
        for (orig.round_polys.items, deser.round_polys.items) |orig_poly, deser_poly| {
            try std.testing.expectEqual(orig_poly.len, deser_poly.len);
            for (orig_poly, deser_poly) |orig_coeff, deser_coeff| {
                try std.testing.expect(orig_coeff.eql(deser_coeff));
            }
        }
    }

    // Verify the deserialized proof works
    var verifier = zkvm.JoltVerifier(F).init(allocator);
    verifier.setVerifyingKey(zkvm.VerifyingKey.init());
    const result = try verifier.verify(&deserialized, &[_]u8{});
    try std.testing.expect(result);
}

// ============================================================================
// JSON Serialization
// ============================================================================

/// JSON serialization magic identifier
pub const JSON_MAGIC: []const u8 = "ZOLT-JSON";

/// Convert a field element to a hex string
pub fn fieldToHex(comptime F: type, elem: F) [64]u8 {
    const bytes = elem.toBytes();
    var hex: [64]u8 = undefined;
    const hex_chars = "0123456789abcdef";
    for (bytes, 0..) |byte, i| {
        hex[i * 2] = hex_chars[byte >> 4];
        hex[i * 2 + 1] = hex_chars[byte & 0xf];
    }
    return hex;
}

/// Convert a hex string to a field element
pub fn hexToField(comptime F: type, hex: []const u8) SerializationError!F {
    if (hex.len != 64) return SerializationError.InvalidData;
    var bytes: [32]u8 = undefined;
    for (0..32) |i| {
        const hi = hexCharToNibble(hex[i * 2]) orelse return SerializationError.InvalidData;
        const lo = hexCharToNibble(hex[i * 2 + 1]) orelse return SerializationError.InvalidData;
        bytes[i] = (hi << 4) | lo;
    }
    return F.fromBytes(&bytes);
}

fn hexCharToNibble(c: u8) ?u4 {
    return switch (c) {
        '0'...'9' => @intCast(c - '0'),
        'a'...'f' => @intCast(c - 'a' + 10),
        'A'...'F' => @intCast(c - 'A' + 10),
        else => null,
    };
}

/// JSON writer for proofs
pub fn JsonProofWriter(comptime F: type) type {
    return struct {
        const Self = @This();

        buffer: std.ArrayListUnmanaged(u8),
        allocator: Allocator,
        indent: usize,

        pub fn init(allocator: Allocator) Self {
            return .{
                .buffer = .{},
                .allocator = allocator,
                .indent = 0,
            };
        }

        pub fn deinit(self: *Self) void {
            self.buffer.deinit(self.allocator);
        }

        pub fn toOwnedSlice(self: *Self) ![]u8 {
            return self.buffer.toOwnedSlice(self.allocator);
        }

        fn writeIndent(self: *Self) !void {
            for (0..self.indent * 2) |_| {
                try self.buffer.append(self.allocator, ' ');
            }
        }

        fn write(self: *Self, data: []const u8) !void {
            try self.buffer.appendSlice(self.allocator, data);
        }

        fn writeLine(self: *Self, data: []const u8) !void {
            try self.writeIndent();
            try self.write(data);
            try self.write("\n");
        }

        fn writeString(self: *Self, key: []const u8, value: []const u8) !void {
            try self.writeIndent();
            try self.write("\"");
            try self.write(key);
            try self.write("\": \"");
            try self.write(value);
            try self.write("\"");
        }

        fn writeNumber(self: *Self, key: []const u8, value: u64) !void {
            try self.writeIndent();
            try self.write("\"");
            try self.write(key);
            try self.write("\": ");
            var buf: [20]u8 = undefined;
            const len = std.fmt.formatIntBuf(&buf, value, 10, .lower, .{});
            try self.write(buf[0..len]);
        }

        fn writeFieldElement(self: *Self, key: []const u8, elem: F) !void {
            const hex = fieldToHex(F, elem);
            try self.writeString(key, &hex);
        }

        /// Write a stage proof to JSON
        pub fn writeStageProof(self: *Self, name: []const u8, stage: *const prover.StageProof(F)) !void {
            try self.writeIndent();
            try self.write("\"");
            try self.write(name);
            try self.write("\": {\n");
            self.indent += 1;

            // Round polynomials
            try self.writeNumber("num_rounds", stage.round_polys.items.len);
            try self.write(",\n");

            try self.writeLine("\"round_polys\": [");
            self.indent += 1;
            for (stage.round_polys.items, 0..) |poly, i| {
                try self.writeIndent();
                try self.write("[");
                for (poly, 0..) |coeff, j| {
                    const hex = fieldToHex(F, coeff);
                    try self.write("\"");
                    try self.write(&hex);
                    try self.write("\"");
                    if (j < poly.len - 1) try self.write(", ");
                }
                try self.write("]");
                if (i < stage.round_polys.items.len - 1) try self.write(",");
                try self.write("\n");
            }
            self.indent -= 1;
            try self.writeLine("],");

            // Challenges
            try self.writeLine("\"challenges\": [");
            self.indent += 1;
            for (stage.challenges.items, 0..) |challenge, i| {
                try self.writeIndent();
                const hex = fieldToHex(F, challenge);
                try self.write("\"");
                try self.write(&hex);
                try self.write("\"");
                if (i < stage.challenges.items.len - 1) try self.write(",");
                try self.write("\n");
            }
            self.indent -= 1;
            try self.writeLine("],");

            // Final claims
            try self.writeLine("\"final_claims\": [");
            self.indent += 1;
            for (stage.final_claims.items, 0..) |claim, i| {
                try self.writeIndent();
                const hex = fieldToHex(F, claim);
                try self.write("\"");
                try self.write(&hex);
                try self.write("\"");
                if (i < stage.final_claims.items.len - 1) try self.write(",");
                try self.write("\n");
            }
            self.indent -= 1;
            try self.writeLine("]");

            self.indent -= 1;
            try self.writeIndent();
            try self.write("}");
        }

        /// Write JoltStageProofs to JSON
        pub fn writeJoltStageProofs(self: *Self, proofs: *const prover.JoltStageProofs(F)) !void {
            try self.writeLine("\"stage_proofs\": {");
            self.indent += 1;

            try self.writeNumber("log_t", proofs.log_t);
            try self.write(",\n");
            try self.writeNumber("log_k", proofs.log_k);
            try self.write(",\n");

            const stage_names = [_][]const u8{
                "spartan",
                "raf",
                "lasso",
                "val",
                "register",
                "booleanity",
            };

            for (&proofs.stage_proofs, stage_names, 0..) |*stage, name, i| {
                try self.writeStageProof(name, stage);
                if (i < 5) try self.write(",");
                try self.write("\n");
            }

            self.indent -= 1;
            try self.writeLine("}");
        }
    };
}

/// Serialize a JoltProof to JSON format
pub fn serializeProofToJson(comptime F: type, allocator: Allocator, proof: anytype) ![]u8 {
    var writer = JsonProofWriter(F).init(allocator);
    defer writer.deinit();

    try writer.writeLine("{");
    writer.indent += 1;

    // Header
    try writer.writeString("format", JSON_MAGIC);
    try writer.write(",\n");
    try writer.writeNumber("version", VERSION);
    try writer.write(",\n");

    // Bytecode proof
    try writer.writeLine("\"bytecode_proof\": {");
    writer.indent += 1;
    try writer.writeFieldElement("commitment_x", proof.bytecode_proof.commitment.x);
    try writer.write(",\n");
    try writer.writeFieldElement("commitment_y", proof.bytecode_proof.commitment.y);
    try writer.write("\n");
    writer.indent -= 1;
    try writer.writeLine("},");

    // Memory proof
    try writer.writeLine("\"memory_proof\": {");
    writer.indent += 1;
    try writer.writeFieldElement("commitment_x", proof.memory_proof.commitment.x);
    try writer.write(",\n");
    try writer.writeFieldElement("commitment_y", proof.memory_proof.commitment.y);
    try writer.write("\n");
    writer.indent -= 1;
    try writer.writeLine("},");

    // Register proof
    try writer.writeLine("\"register_proof\": {");
    writer.indent += 1;
    try writer.writeFieldElement("commitment_x", proof.register_proof.commitment.x);
    try writer.write(",\n");
    try writer.writeFieldElement("commitment_y", proof.register_proof.commitment.y);
    try writer.write("\n");
    writer.indent -= 1;
    try writer.writeLine("},");

    // R1CS proof summary
    try writer.writeLine("\"r1cs_proof\": {");
    writer.indent += 1;
    try writer.writeNumber("tau_len", proof.r1cs_proof.tau.len);
    try writer.write(",\n");
    try writer.writeFieldElement("sumcheck_claim", proof.r1cs_proof.sumcheck_proof.claim);
    try writer.write(",\n");
    try writer.writeFieldElement("sumcheck_final_eval", proof.r1cs_proof.sumcheck_proof.final_eval);
    try writer.write(",\n");
    try writer.writeNumber("sumcheck_rounds", proof.r1cs_proof.sumcheck_proof.rounds.len);
    try writer.write(",\n");
    try writer.writeNumber("eval_point_len", proof.r1cs_proof.eval_point.len);
    try writer.write("\n");
    writer.indent -= 1;
    try writer.writeLine("},");

    // Stage proofs
    if (proof.stage_proofs) |stage_proofs| {
        try writer.writeJoltStageProofs(&stage_proofs);
    } else {
        try writer.writeLine("\"stage_proofs\": null");
    }

    writer.indent -= 1;
    try writer.writeLine("}");

    return writer.toOwnedSlice();
}

/// Write proof to JSON file
pub fn writeProofToJsonFile(comptime F: type, allocator: Allocator, proof: anytype, path: []const u8) !void {
    const json = try serializeProofToJson(F, allocator, proof);
    defer allocator.free(json);

    const file = try std.fs.cwd().createFile(path, .{});
    defer file.close();

    try file.writeAll(json);
}

test "field to hex roundtrip" {
    const F = field.BN254Scalar;

    const test_values = [_]F{
        F.zero(),
        F.one(),
        F.fromU64(12345678901234567890),
        F.fromU64(0xdeadbeefcafebabe),
    };

    for (test_values) |val| {
        const hex = fieldToHex(F, val);
        const restored = try hexToField(F, &hex);
        try std.testing.expect(val.eql(restored));
    }
}

test "json stage proof serialization" {
    const F = field.BN254Scalar;
    const allocator = std.testing.allocator;

    // Create a stage proof with data
    var stage = prover.StageProof(F).init(allocator);
    defer stage.deinit();

    // Add round polynomial
    const poly = try allocator.alloc(F, 2);
    poly[0] = F.fromU64(42);
    poly[1] = F.fromU64(43);
    try stage.round_polys.append(allocator, poly);

    // Add challenge
    try stage.challenges.append(allocator, F.fromU64(100));

    // Add claim
    try stage.final_claims.append(allocator, F.fromU64(999));

    // Serialize to JSON
    var writer = JsonProofWriter(F).init(allocator);
    defer writer.deinit();

    try writer.writeLine("{");
    writer.indent += 1;
    try writer.writeStageProof("test_stage", &stage);
    try writer.write("\n");
    writer.indent -= 1;
    try writer.writeLine("}");

    const json = try writer.toOwnedSlice();
    defer allocator.free(json);

    // Verify it's valid JSON structure (basic check)
    try std.testing.expect(json.len > 0);
    try std.testing.expect(json[0] == '{');
    try std.testing.expect(json[json.len - 2] == '}'); // -2 because of trailing newline

    // Check it contains expected keys
    try std.testing.expect(std.mem.indexOf(u8, json, "\"test_stage\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"round_polys\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"challenges\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"final_claims\"") != null);
}

test "full proof JSON serialization" {
    const F = field.BN254Scalar;
    const allocator = std.testing.allocator;
    const zkvm = @import("mod.zig");

    // Create a minimal bytecode program
    const program = [_]u8{
        0x13, 0x05, 0xa0, 0x02, // li a0, 42
        0x93, 0x05, 0xa0, 0x00, // li a1, 10
        0x33, 0x05, 0xb5, 0x00, // add a0, a0, a1
        0x73, 0x00, 0x10, 0x00, // ebreak
    };

    // Create prover and generate proof
    var prover_inst = zkvm.JoltProver(F).init(allocator);
    prover_inst.max_cycles = 64;

    var proof = try prover_inst.prove(&program, &[_]u8{});
    defer proof.deinit();

    // Serialize to JSON
    const json = try serializeProofToJson(F, allocator, proof);
    defer allocator.free(json);

    // Verify it's valid JSON structure
    try std.testing.expect(json.len > 0);
    try std.testing.expect(json[0] == '{');

    // Check it contains expected sections
    try std.testing.expect(std.mem.indexOf(u8, json, "\"format\": \"ZOLT-JSON\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"version\": 1") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"bytecode_proof\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"memory_proof\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"register_proof\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"r1cs_proof\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"stage_proofs\"") != null);
}

// ============================================================================
// JSON Deserialization
// ============================================================================

/// JSON deserialization error types
pub const JsonDeserializationError = error{
    InvalidFormat,
    MissingField,
    InvalidFieldValue,
    UnsupportedVersion,
    OutOfMemory,
};

/// JSON proof reader for deserializing proofs
pub fn JsonProofReader(comptime F: type) type {
    return struct {
        const Self = @This();
        const json = std.json;

        allocator: Allocator,

        pub fn init(allocator: Allocator) Self {
            return .{
                .allocator = allocator,
            };
        }

        /// Extract a string field from a JSON object
        fn getString(obj: json.Value, key: []const u8) JsonDeserializationError![]const u8 {
            const val = obj.object.get(key) orelse return JsonDeserializationError.MissingField;
            return switch (val) {
                .string => |s| s,
                else => JsonDeserializationError.InvalidFieldValue,
            };
        }

        /// Extract an integer field from a JSON object
        fn getInt(obj: json.Value, key: []const u8) JsonDeserializationError!u64 {
            const val = obj.object.get(key) orelse return JsonDeserializationError.MissingField;
            return switch (val) {
                .integer => |i| @intCast(i),
                else => JsonDeserializationError.InvalidFieldValue,
            };
        }

        /// Extract an object field from a JSON object
        fn getObject(obj: json.Value, key: []const u8) JsonDeserializationError!json.Value {
            const val = obj.object.get(key) orelse return JsonDeserializationError.MissingField;
            return switch (val) {
                .object => val,
                else => JsonDeserializationError.InvalidFieldValue,
            };
        }

        /// Extract an array field from a JSON object
        fn getArray(obj: json.Value, key: []const u8) JsonDeserializationError!json.Array {
            const val = obj.object.get(key) orelse return JsonDeserializationError.MissingField;
            return switch (val) {
                .array => |arr| arr,
                else => JsonDeserializationError.InvalidFieldValue,
            };
        }

        /// Parse a field element from a hex string
        fn parseFieldElement(hex_str: []const u8) (JsonDeserializationError || SerializationError)!F {
            return hexToField(F, hex_str) catch |err| switch (err) {
                SerializationError.InvalidData => JsonDeserializationError.InvalidFieldValue,
                else => err,
            };
        }

        /// Parse a commitment from JSON object (with x and y fields)
        fn parseCommitment(self: *Self, obj: json.Value) (JsonDeserializationError || SerializationError)!commitment_types.PolyCommitment {
            _ = self;
            const x_hex = try getString(obj, "commitment_x");
            const y_hex = try getString(obj, "commitment_y");

            const x = try parseFieldElement(x_hex);
            const y = try parseFieldElement(y_hex);

            return commitment_types.PolyCommitment{
                .x = x,
                .y = y,
            };
        }

        /// Parse a stage proof from JSON object
        fn parseStageProof(self: *Self, obj: json.Value) !prover.StageProof(F) {
            var stage = prover.StageProof(F).init(self.allocator);
            errdefer stage.deinit();

            // Parse round polynomials
            const round_polys_arr = try getArray(obj, "round_polys");
            for (round_polys_arr.items) |poly_val| {
                const poly_arr = switch (poly_val) {
                    .array => |arr| arr,
                    else => return JsonDeserializationError.InvalidFieldValue,
                };

                const poly = try self.allocator.alloc(F, poly_arr.items.len);
                errdefer self.allocator.free(poly);

                for (poly_arr.items, 0..) |coeff_val, i| {
                    const hex = switch (coeff_val) {
                        .string => |s| s,
                        else => return JsonDeserializationError.InvalidFieldValue,
                    };
                    poly[i] = try parseFieldElement(hex);
                }

                try stage.round_polys.append(self.allocator, poly);
            }

            // Parse challenges
            const challenges_arr = try getArray(obj, "challenges");
            for (challenges_arr.items) |challenge_val| {
                const hex = switch (challenge_val) {
                    .string => |s| s,
                    else => return JsonDeserializationError.InvalidFieldValue,
                };
                const challenge = try parseFieldElement(hex);
                try stage.challenges.append(self.allocator, challenge);
            }

            // Parse final claims
            const claims_arr = try getArray(obj, "final_claims");
            for (claims_arr.items) |claim_val| {
                const hex = switch (claim_val) {
                    .string => |s| s,
                    else => return JsonDeserializationError.InvalidFieldValue,
                };
                const claim = try parseFieldElement(hex);
                try stage.final_claims.append(self.allocator, claim);
            }

            return stage;
        }

        /// Parse JoltStageProofs from JSON object
        fn parseJoltStageProofs(self: *Self, obj: json.Value) !prover.JoltStageProofs(F) {
            const log_t = try getInt(obj, "log_t");
            const log_k = try getInt(obj, "log_k");

            var proofs = prover.JoltStageProofs(F).init(self.allocator);
            errdefer proofs.deinit();

            proofs.log_t = log_t;
            proofs.log_k = log_k;

            const stage_names = [_][]const u8{
                "spartan",
                "raf",
                "lasso",
                "val",
                "register",
                "booleanity",
            };

            for (stage_names, 0..) |name, i| {
                const stage_obj = try getObject(obj, name);
                proofs.stage_proofs[i].deinit(); // Free the initialized empty stage
                proofs.stage_proofs[i] = try self.parseStageProof(stage_obj);
            }

            return proofs;
        }
    };
}

/// Check if a file is a JSON proof (based on content, not extension)
pub fn isJsonProof(data: []const u8) bool {
    // Skip leading whitespace
    var i: usize = 0;
    while (i < data.len and (data[i] == ' ' or data[i] == '\t' or data[i] == '\n' or data[i] == '\r')) {
        i += 1;
    }

    // Check if it starts with '{'
    if (i >= data.len or data[i] != '{') {
        return false;
    }

    // Check for ZOLT-JSON magic string
    return std.mem.indexOf(u8, data, "\"ZOLT-JSON\"") != null;
}

/// Deserialize a JoltProof from JSON bytes
pub fn deserializeProofFromJson(comptime F: type, allocator: Allocator, data: []const u8) !@import("mod.zig").JoltProof(F) {
    const zkvm = @import("mod.zig");
    const json = std.json;

    // Parse JSON
    var parsed = json.parseFromSlice(json.Value, allocator, data, .{}) catch {
        return JsonDeserializationError.InvalidFormat;
    };
    defer parsed.deinit();

    const root = parsed.value;

    // Verify format magic
    const Reader = JsonProofReader(F);
    const format = try Reader.getString(root, "format");
    if (!std.mem.eql(u8, format, JSON_MAGIC)) {
        return JsonDeserializationError.InvalidFormat;
    }

    // Verify version
    const version = try Reader.getInt(root, "version");
    if (version != VERSION) {
        return JsonDeserializationError.UnsupportedVersion;
    }

    var reader = Reader.init(allocator);

    // Parse bytecode proof
    const bc_obj = try Reader.getObject(root, "bytecode_proof");
    var bc_proof = bytecode.BytecodeProof(F).init();
    bc_proof.commitment = try reader.parseCommitment(bc_obj);

    // Parse memory proof
    const mem_obj = try Reader.getObject(root, "memory_proof");
    var mem_proof = ram.MemoryProof(F).init();
    mem_proof.commitment = try reader.parseCommitment(mem_obj);

    // Parse register proof
    const reg_obj = try Reader.getObject(root, "register_proof");
    var reg_proof = registers.RegisterProof(F).init();
    reg_proof.commitment = try reader.parseCommitment(reg_obj);

    // Parse R1CS proof (minimal - we need to restore enough for verification)
    const r1cs_obj = try Reader.getObject(root, "r1cs_proof");
    const tau_len = try Reader.getInt(r1cs_obj, "tau_len");

    // Allocate tau with zeros (JSON doesn't include full tau for space reasons)
    const tau = try allocator.alloc(F, tau_len);
    errdefer allocator.free(tau);
    for (tau) |*t| {
        t.* = F.zero();
    }

    // Parse sumcheck values
    const sc_claim_hex = try Reader.getString(r1cs_obj, "sumcheck_claim");
    const sc_claim = try Reader.parseFieldElement(sc_claim_hex);

    const sc_final_eval_hex = try Reader.getString(r1cs_obj, "sumcheck_final_eval");
    const sc_final_eval = try Reader.parseFieldElement(sc_final_eval_hex);

    const eval_point_len = try Reader.getInt(r1cs_obj, "eval_point_len");
    const eval_point = try allocator.alloc(F, eval_point_len);
    errdefer allocator.free(eval_point);
    for (eval_point) |*e| {
        e.* = F.zero();
    }

    // Create empty rounds array
    const sc_rounds = try allocator.alloc(@import("../subprotocols/mod.zig").Sumcheck(F).Round, 0);

    // Create R1CS proof with default eval_claims
    const r1cs_proof = spartan.R1CSProof(F){
        .tau = tau,
        .sumcheck_proof = .{
            .claim = sc_claim,
            .rounds = sc_rounds,
            .final_point = eval_point,
            .final_eval = sc_final_eval,
            .allocator = allocator,
        },
        .eval_claims = .{ F.zero(), F.zero(), F.zero() },
        .eval_point = eval_point,
        .allocator = allocator,
    };

    // Parse stage proofs (if present and not null)
    var stage_proofs: ?prover.JoltStageProofs(F) = null;
    const stage_proofs_val = root.object.get("stage_proofs");
    if (stage_proofs_val) |spv| {
        switch (spv) {
            .null => {},
            .object => {
                stage_proofs = try reader.parseJoltStageProofs(spv);
            },
            else => return JsonDeserializationError.InvalidFieldValue,
        }
    }

    return zkvm.JoltProof(F){
        .bytecode_proof = bc_proof,
        .memory_proof = mem_proof,
        .register_proof = reg_proof,
        .r1cs_proof = r1cs_proof,
        .stage_proofs = stage_proofs,
        .allocator = allocator,
    };
}

/// Read a proof from a JSON file
pub fn readProofFromJsonFile(comptime F: type, allocator: Allocator, path: []const u8) !@import("mod.zig").JoltProof(F) {
    const file = try std.fs.cwd().openFile(path, .{});
    defer file.close();

    const stat = try file.stat();
    const data = try allocator.alloc(u8, stat.size);
    defer allocator.free(data);

    const bytes_read = try file.readAll(data);
    if (bytes_read != stat.size) {
        return SerializationError.UnexpectedEof;
    }

    return deserializeProofFromJson(F, allocator, data);
}

/// Auto-detect format and read proof from file
pub fn readProofAutoDetect(comptime F: type, allocator: Allocator, path: []const u8) !@import("mod.zig").JoltProof(F) {
    const file = try std.fs.cwd().openFile(path, .{});
    defer file.close();

    const stat = try file.stat();
    const data = try allocator.alloc(u8, stat.size);
    defer allocator.free(data);

    const bytes_read = try file.readAll(data);
    if (bytes_read != stat.size) {
        return SerializationError.UnexpectedEof;
    }

    // Check if it's JSON or binary
    if (isJsonProof(data)) {
        return deserializeProofFromJson(F, allocator, data);
    } else {
        return deserializeProof(F, allocator, data);
    }
}

test "JSON deserialization basic" {
    const F = field.BN254Scalar;
    const allocator = std.testing.allocator;

    // Create a simple JSON proof manually
    const json_proof =
        \\{
        \\  "format": "ZOLT-JSON",
        \\  "version": 1,
        \\  "bytecode_proof": {
        \\    "commitment_x": "0000000000000000000000000000000000000000000000000000000000000001",
        \\    "commitment_y": "0000000000000000000000000000000000000000000000000000000000000002"
        \\  },
        \\  "memory_proof": {
        \\    "commitment_x": "0000000000000000000000000000000000000000000000000000000000000001",
        \\    "commitment_y": "0000000000000000000000000000000000000000000000000000000000000002"
        \\  },
        \\  "register_proof": {
        \\    "commitment_x": "0000000000000000000000000000000000000000000000000000000000000001",
        \\    "commitment_y": "0000000000000000000000000000000000000000000000000000000000000002"
        \\  },
        \\  "r1cs_proof": {
        \\    "tau_len": 4,
        \\    "sumcheck_claim": "0000000000000000000000000000000000000000000000000000000000000000",
        \\    "sumcheck_final_eval": "0000000000000000000000000000000000000000000000000000000000000000",
        \\    "sumcheck_rounds": 0,
        \\    "eval_point_len": 4
        \\  },
        \\  "stage_proofs": null
        \\}
    ;

    var proof = try deserializeProofFromJson(F, allocator, json_proof);
    defer proof.deinit();

    // Verify basic structure
    try std.testing.expect(proof.stage_proofs == null);
}

test "JSON roundtrip with stage proofs" {
    const F = field.BN254Scalar;
    const allocator = std.testing.allocator;
    const zkvm = @import("mod.zig");

    // Create a minimal bytecode program
    const program = [_]u8{
        0x13, 0x05, 0xa0, 0x02, // li a0, 42
        0x93, 0x05, 0xa0, 0x00, // li a1, 10
        0x33, 0x05, 0xb5, 0x00, // add a0, a0, a1
        0x73, 0x00, 0x10, 0x00, // ebreak
    };

    // Create prover and generate proof
    var prover_inst = zkvm.JoltProver(F).init(allocator);
    prover_inst.max_cycles = 64;

    var proof = try prover_inst.prove(&program, &[_]u8{});
    defer proof.deinit();

    // Serialize to JSON
    const json_data = try serializeProofToJson(F, allocator, proof);
    defer allocator.free(json_data);

    // Deserialize from JSON
    var deserialized = try deserializeProofFromJson(F, allocator, json_data);
    defer deserialized.deinit();

    // Verify structure matches
    try std.testing.expect(deserialized.stage_proofs != null);

    const orig_stages = proof.stage_proofs.?;
    const deser_stages = deserialized.stage_proofs.?;

    try std.testing.expectEqual(orig_stages.log_t, deser_stages.log_t);
    try std.testing.expectEqual(orig_stages.log_k, deser_stages.log_k);

    // Compare stage proof sizes
    for (orig_stages.stage_proofs, deser_stages.stage_proofs) |orig, deser| {
        try std.testing.expectEqual(orig.round_polys.items.len, deser.round_polys.items.len);
        try std.testing.expectEqual(orig.challenges.items.len, deser.challenges.items.len);
        try std.testing.expectEqual(orig.final_claims.items.len, deser.final_claims.items.len);
    }
}

test "isJsonProof detection" {
    // JSON proof
    try std.testing.expect(isJsonProof("{\"format\": \"ZOLT-JSON\"}"));
    try std.testing.expect(isJsonProof("  \n{\"format\": \"ZOLT-JSON\"}"));

    // Binary proof (starts with ZOLT magic)
    try std.testing.expect(!isJsonProof("ZOLT\x01\x00\x00\x00"));

    // Invalid
    try std.testing.expect(!isJsonProof("not json"));
    try std.testing.expect(!isJsonProof("{}"));
}
