//! Jolt-Compatible Proof Serialization
//!
//! This module provides serialization that is byte-compatible with Jolt's
//! arkworks-based serialization format.
//!
//! ## Key Differences from Zolt Native Format
//!
//! 1. **No Magic Header**: Jolt proofs have no "ZOLT" prefix
//! 2. **usize as u64**: All lengths are serialized as 8-byte little-endian
//! 3. **Field Elements**: 32 bytes little-endian (Montgomery form limbs)
//! 4. **GT Elements**: 384 bytes (12 Fp elements for Fp12/Dory commitments)
//! 5. **BTreeMap**: Length prefix + sorted key-value pairs
//!
//! Reference: jolt-core/src/zkvm/proof_serialization.rs

const std = @import("std");
const Allocator = std.mem.Allocator;
const jolt_types = @import("jolt_types.zig");
const commitment_mod = @import("../poly/commitment/mod.zig");
const pairing = @import("../field/pairing.zig");
const dory_mod = @import("../poly/commitment/dory.zig");

/// GT element (Dory commitment) type
pub const GT = pairing.GT;

/// Dory commitment type
pub const DoryCommitment = commitment_mod.DoryCommitment;

/// Dory proof type
pub const DoryProof = dory_mod.DoryProof;

/// Arkworks-compatible serializer
pub fn ArkworksSerializer(comptime F: type) type {
    return struct {
        const Self = @This();

        buffer: std.ArrayListUnmanaged(u8),
        allocator: Allocator,

        pub fn init(allocator: Allocator) Self {
            return Self{
                .buffer = .{},
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            self.buffer.deinit(self.allocator);
        }

        /// Get the serialized bytes (borrowed)
        pub fn bytes(self: *const Self) []const u8 {
            return self.buffer.items;
        }

        /// Get the serialized bytes (owned)
        pub fn toOwnedSlice(self: *Self) ![]u8 {
            return self.buffer.toOwnedSlice(self.allocator);
        }

        /// Write a usize as u64 little-endian (arkworks format)
        pub fn writeUsize(self: *Self, value: usize) !void {
            var buf: [8]u8 = undefined;
            std.mem.writeInt(u64, &buf, @intCast(value), .little);
            try self.buffer.appendSlice(self.allocator, &buf);
        }

        /// Write a u8
        pub fn writeU8(self: *Self, value: u8) !void {
            try self.buffer.append(self.allocator, value);
        }

        /// Alias for writeU8, compatible with writer interface
        pub fn writeByte(self: *Self, value: u8) !void {
            try self.buffer.append(self.allocator, value);
        }

        /// Write bytes directly
        pub fn writeBytes(self: *Self, data: []const u8) !void {
            try self.buffer.appendSlice(self.allocator, data);
        }

        /// Write a field element in arkworks format (32 bytes LE)
        /// This matches `serialize_uncompressed` for BN254 Fr
        ///
        /// IMPORTANT: arkworks serialize_uncompressed converts FROM Montgomery form
        /// to standard representation. We must do the same using toBytes().
        pub fn writeFieldElement(self: *Self, scalar: F) !void {
            // toBytes() converts from Montgomery form to standard form
            const buf = scalar.toBytes();
            try self.buffer.appendSlice(self.allocator, &buf);
        }

        /// Write a slice of field elements
        pub fn writeFieldElements(self: *Self, scalars: []const F) !void {
            for (scalars) |scalar| {
                try self.writeFieldElement(scalar);
            }
        }

        /// Write a Vec of field elements with length prefix
        pub fn writeVecFieldElements(self: *Self, scalars: []const F) !void {
            try self.writeUsize(scalars.len);
            try self.writeFieldElements(scalars);
        }

        /// Write a GT element (Dory commitment) in arkworks format (384 bytes)
        /// GT = Fp12, serialized as 12 Fp elements in LE format
        pub fn writeGT(self: *Self, gt: GT) !void {
            const gt_bytes = gt.toBytes();
            try self.buffer.appendSlice(self.allocator, &gt_bytes);
        }

        /// Write a Dory commitment (alias for writeGT)
        pub fn writeDoryCommitment(self: *Self, comm: DoryCommitment) !void {
            try self.writeGT(comm);
        }

        /// Write a Vec of Dory commitments with length prefix
        pub fn writeVecDoryCommitments(self: *Self, comms: []const DoryCommitment) !void {
            try self.writeUsize(comms.len);
            for (comms) |c| {
                try self.writeDoryCommitment(c);
            }
        }

        /// Write a G1 point in arkworks compressed format (32 bytes)
        pub fn writeG1Compressed(self: *Self, point: dory_mod.G1Point) !void {
            const compressed = dory_mod.compressG1(point);
            try self.buffer.appendSlice(self.allocator, &compressed);
        }

        /// Write a G2 point in arkworks compressed format (64 bytes)
        pub fn writeG2Compressed(self: *Self, point: dory_mod.G2Point) !void {
            const compressed = dory_mod.compressG2(point);
            try self.buffer.appendSlice(self.allocator, &compressed);
        }

        /// Write a u32 in little-endian format
        pub fn writeU32(self: *Self, value: u32) !void {
            var buf: [4]u8 = undefined;
            std.mem.writeInt(u32, &buf, value, .little);
            try self.buffer.appendSlice(self.allocator, &buf);
        }

        /// Write a DoryProof in arkworks format
        /// Format matches dory-pcs ark_serde.rs CanonicalSerialize
        pub fn writeDoryProof(self: *Self, proof: *const DoryProof) !void {
            // 1. VMV message: c (GT), d2 (GT), e1 (G1)
            try self.writeGT(proof.vmv_message.c);
            try self.writeGT(proof.vmv_message.d2);
            try self.writeG1Compressed(proof.vmv_message.e1);

            // 2. Number of rounds (u32)
            const num_rounds: u32 = @intCast(proof.first_messages.len);
            try self.writeU32(num_rounds);

            // 3. First messages
            for (proof.first_messages) |msg| {
                try self.writeGT(msg.d1_left);
                try self.writeGT(msg.d1_right);
                try self.writeGT(msg.d2_left);
                try self.writeGT(msg.d2_right);
                try self.writeG1Compressed(msg.e1_beta);
                try self.writeG2Compressed(msg.e2_beta);
            }

            // 4. Second messages
            for (proof.second_messages) |msg| {
                try self.writeGT(msg.c_plus);
                try self.writeGT(msg.c_minus);
                try self.writeG1Compressed(msg.e1_plus);
                try self.writeG1Compressed(msg.e1_minus);
                try self.writeG2Compressed(msg.e2_plus);
                try self.writeG2Compressed(msg.e2_minus);
            }

            // 5. Final message: e1 (G1), e2 (G2)
            try self.writeG1Compressed(proof.final_message.e1);
            try self.writeG2Compressed(proof.final_message.e2);

            // 6. nu and sigma (u32 each)
            try self.writeU32(proof.nu);
            try self.writeU32(proof.sigma);
        }

        /// Write an optional field
        pub fn writeOption(self: *Self, comptime T: type, opt: ?T, writeFn: *const fn (*Self, T) anyerror!void) !void {
            if (opt) |value| {
                try self.writeU8(1); // Some
                try writeFn(self, value);
            } else {
                try self.writeU8(0); // None
            }
        }

        /// Write a CompressedUniPoly
        pub fn writeCompressedUniPoly(self: *Self, poly: *const jolt_types.CompressedUniPoly(F)) !void {
            try self.writeUsize(poly.coeffs_except_linear_term.len);
            try self.writeFieldElements(poly.coeffs_except_linear_term);
        }

        /// Write a SumcheckInstanceProof
        pub fn writeSumcheckInstanceProof(self: *Self, proof: *const jolt_types.SumcheckInstanceProof(F)) !void {
            try self.writeUsize(proof.compressed_polys.items.len);
            for (proof.compressed_polys.items) |*poly| {
                try self.writeCompressedUniPoly(poly);
            }
        }

        /// Write a UniSkipFirstRoundProof
        pub fn writeUniSkipFirstRoundProof(self: *Self, proof: *const jolt_types.UniSkipFirstRoundProof(F)) !void {
            try self.writeUsize(proof.uni_poly.len);
            try self.writeFieldElements(proof.uni_poly);
        }

        /// Write an OpeningId
        pub fn writeOpeningId(self: *Self, id: jolt_types.OpeningId) !void {
            switch (id) {
                .UntrustedAdvice => |sumcheck_id| {
                    try self.writeU8(jolt_types.OpeningId.UNTRUSTED_ADVICE_BASE + @intFromEnum(sumcheck_id));
                },
                .TrustedAdvice => |sumcheck_id| {
                    try self.writeU8(jolt_types.OpeningId.TRUSTED_ADVICE_BASE + @intFromEnum(sumcheck_id));
                },
                .Committed => |c| {
                    try self.writeU8(jolt_types.OpeningId.COMMITTED_BASE + @intFromEnum(c.sumcheck_id));
                    switch (c.poly) {
                        .RdInc => try self.writeU8(0),
                        .RamInc => try self.writeU8(1),
                        .InstructionRa => |i| {
                            try self.writeU8(2);
                            try self.writeU8(@truncate(i));
                        },
                        .BytecodeRa => |i| {
                            try self.writeU8(3);
                            try self.writeU8(@truncate(i));
                        },
                        .RamRa => |i| {
                            try self.writeU8(4);
                            try self.writeU8(@truncate(i));
                        },
                    }
                },
                .Virtual => |v| {
                    try self.writeU8(jolt_types.OpeningId.VIRTUAL_BASE + @intFromEnum(v.sumcheck_id));
                    try v.poly.serialize(self);
                },
            }
        }

        /// Write OpeningClaims (BTreeMap format)
        pub fn writeOpeningClaims(self: *Self, claims: *const jolt_types.OpeningClaims(F)) !void {
            try self.writeUsize(claims.entries.items.len);
            for (claims.entries.items) |entry| {
                try self.writeOpeningId(entry.id);
                try self.writeFieldElement(entry.claim);
            }
        }

        /// Write a complete JoltProof
        pub fn writeJoltProof(
            self: *Self,
            comptime Commitment: type,
            comptime Proof: type,
            proof: *const jolt_types.JoltProof(F, Commitment, Proof),
            writeCommitment: *const fn (*Self, Commitment) anyerror!void,
            writeProof: *const fn (*Self, Proof) anyerror!void,
        ) !void {
            // 1. Opening claims
            try self.writeOpeningClaims(&proof.opening_claims);

            // 2. Commitments
            try self.writeUsize(proof.commitments.items.len);
            for (proof.commitments.items) |comm| {
                try writeCommitment(self, comm);
            }

            // 3. Stage 1
            if (proof.stage1_uni_skip_first_round_proof) |*p| {
                try self.writeUniSkipFirstRoundProof(p);
            }
            try self.writeSumcheckInstanceProof(&proof.stage1_sumcheck_proof);

            // 4. Stage 2
            if (proof.stage2_uni_skip_first_round_proof) |*p| {
                try self.writeUniSkipFirstRoundProof(p);
            }
            try self.writeSumcheckInstanceProof(&proof.stage2_sumcheck_proof);

            // 5. Stages 3-7
            try self.writeSumcheckInstanceProof(&proof.stage3_sumcheck_proof);
            try self.writeSumcheckInstanceProof(&proof.stage4_sumcheck_proof);
            try self.writeSumcheckInstanceProof(&proof.stage5_sumcheck_proof);
            try self.writeSumcheckInstanceProof(&proof.stage6_sumcheck_proof);
            try self.writeSumcheckInstanceProof(&proof.stage7_sumcheck_proof);

            // 6. Joint opening proof
            if (proof.joint_opening_proof) |p| {
                try writeProof(self, p);
            }

            // 7. Advice proofs (all optional)
            if (proof.trusted_advice_val_evaluation_proof) |p| {
                try self.writeU8(1);
                try writeProof(self, p);
            } else {
                try self.writeU8(0);
            }

            if (proof.trusted_advice_val_final_proof) |p| {
                try self.writeU8(1);
                try writeProof(self, p);
            } else {
                try self.writeU8(0);
            }

            if (proof.untrusted_advice_val_evaluation_proof) |p| {
                try self.writeU8(1);
                try writeProof(self, p);
            } else {
                try self.writeU8(0);
            }

            if (proof.untrusted_advice_val_final_proof) |p| {
                try self.writeU8(1);
                try writeProof(self, p);
            } else {
                try self.writeU8(0);
            }

            if (proof.untrusted_advice_commitment) |c| {
                try self.writeU8(1);
                try writeCommitment(self, c);
            } else {
                try self.writeU8(0);
            }

            // 8. Configuration
            try self.writeUsize(proof.trace_length);
            try self.writeUsize(proof.ram_K);
            try self.writeUsize(proof.bytecode_K);
            try self.writeUsize(proof.log_k_chunk);
            try self.writeUsize(proof.lookups_ra_virtual_log_k_chunk);
        }

        /// Write a JoltProof using Dory commitments (GT elements)
        /// This is a convenience wrapper for proofs with DoryCommitment type
        pub fn writeJoltDoryProof(
            self: *Self,
            comptime DoryProofT: type,
            proof: *const jolt_types.JoltProof(F, DoryCommitment, DoryProofT),
            writeDoryProofFn: *const fn (*Self, DoryProofT) anyerror!void,
        ) !void {
            const writeDoryCommWrapper = struct {
                fn f(ser: *Self, c: DoryCommitment) !void {
                    try ser.writeDoryCommitment(c);
                }
            }.f;

            try self.writeJoltProof(
                DoryCommitment,
                DoryProofT,
                proof,
                writeDoryCommWrapper,
                writeDoryProofFn,
            );
        }
    };
}

/// Arkworks-compatible deserializer
pub fn ArkworksDeserializer(comptime F: type) type {
    return struct {
        const Self = @This();

        data: []const u8,
        pos: usize,

        pub fn init(data: []const u8) Self {
            return Self{
                .data = data,
                .pos = 0,
            };
        }

        /// Read a usize from u64 little-endian
        pub fn readUsize(self: *Self) !usize {
            if (self.pos + 8 > self.data.len) return error.UnexpectedEof;
            const value = std.mem.readInt(u64, self.data[self.pos..][0..8], .little);
            self.pos += 8;
            return @intCast(value);
        }

        /// Read a u8
        pub fn readU8(self: *Self) !u8 {
            if (self.pos >= self.data.len) return error.UnexpectedEof;
            const value = self.data[self.pos];
            self.pos += 1;
            return value;
        }

        /// Read bytes
        pub fn readBytes(self: *Self, len: usize) ![]const u8 {
            if (self.pos + len > self.data.len) return error.UnexpectedEof;
            const result = self.data[self.pos..][0..len];
            self.pos += len;
            return result;
        }

        /// Read a field element from arkworks format (32 bytes LE)
        /// The bytes are in standard form; we convert to Montgomery form.
        pub fn readFieldElement(self: *Self) !F {
            const bytes = try self.readBytes(32);
            // fromBytes converts from standard form to Montgomery form
            return F.fromBytes(bytes);
        }

        /// Read a Vec of field elements
        pub fn readVecFieldElements(self: *Self, allocator: Allocator) ![]F {
            const len = try self.readUsize();
            const result = try allocator.alloc(F, len);
            errdefer allocator.free(result);

            for (0..len) |i| {
                result[i] = try self.readFieldElement();
            }
            return result;
        }

        /// Read a GT element (Dory commitment) from arkworks format (384 bytes)
        pub fn readGT(self: *Self) !GT {
            if (self.pos + 384 > self.data.len) return error.UnexpectedEof;
            const bytes = self.data[self.pos..][0..384];
            self.pos += 384;
            return GT.fromBytes(bytes);
        }

        /// Read a Dory commitment (alias for readGT)
        pub fn readDoryCommitment(self: *Self) !DoryCommitment {
            return self.readGT();
        }

        /// Read a Vec of Dory commitments
        pub fn readVecDoryCommitments(self: *Self, allocator: Allocator) ![]DoryCommitment {
            const len = try self.readUsize();
            const result = try allocator.alloc(DoryCommitment, len);
            errdefer allocator.free(result);

            for (0..len) |i| {
                result[i] = try self.readDoryCommitment();
            }
            return result;
        }

        /// Read a SumcheckInstanceProof
        pub fn readSumcheckInstanceProof(self: *Self, allocator: Allocator) !jolt_types.SumcheckInstanceProof(F) {
            var proof = jolt_types.SumcheckInstanceProof(F).init(allocator);
            errdefer proof.deinit();

            const num_polys = try self.readUsize();
            for (0..num_polys) |_| {
                const coeffs = try self.readVecFieldElements(allocator);
                defer allocator.free(coeffs);
                try proof.addRoundPoly(coeffs);
            }

            return proof;
        }

        /// Read a UniSkipFirstRoundProof
        pub fn readUniSkipFirstRoundProof(self: *Self, allocator: Allocator) !jolt_types.UniSkipFirstRoundProof(F) {
            const coeffs = try self.readVecFieldElements(allocator);
            return jolt_types.UniSkipFirstRoundProof(F){
                .uni_poly = coeffs,
                .allocator = allocator,
            };
        }
    };
}

/// Write a proof to a file in Jolt-compatible format
pub fn writeJoltProofToFile(
    comptime F: type,
    comptime Commitment: type,
    comptime Proof: type,
    proof: *const jolt_types.JoltProof(F, Commitment, Proof),
    path: []const u8,
    allocator: Allocator,
    writeCommitment: *const fn (*ArkworksSerializer(F), Commitment) anyerror!void,
    writeProof: *const fn (*ArkworksSerializer(F), Proof) anyerror!void,
) !void {
    var serializer = ArkworksSerializer(F).init(allocator);
    defer serializer.deinit();

    try serializer.writeJoltProof(Commitment, Proof, proof, writeCommitment, writeProof);

    const file = try std.fs.cwd().createFile(path, .{});
    defer file.close();

    try file.writeAll(serializer.bytes());
}

// =============================================================================
// Tests
// =============================================================================

const testing = std.testing;
const BN254Scalar = @import("../field/mod.zig").BN254Scalar;

test "arkworks serializer: field element format" {
    var serializer = ArkworksSerializer(BN254Scalar).init(testing.allocator);
    defer serializer.deinit();

    // Test with value 42
    const scalar = BN254Scalar.fromU64(42);
    try serializer.writeFieldElement(scalar);

    // Should be 32 bytes
    try testing.expectEqual(@as(usize, 32), serializer.bytes().len);

    // Should match Jolt's output: [2a, 00, 00, 00, ...] (42 in LE)
    try testing.expectEqual(@as(u8, 0x2a), serializer.bytes()[0]);
    for (1..32) |i| {
        try testing.expectEqual(@as(u8, 0), serializer.bytes()[i]);
    }
}

// =============================================================================
// Jolt Compatibility Test Vectors
// =============================================================================
// These test vectors are generated from Jolt to verify byte-level compatibility

test "jolt serialization: Fr(42) matches Jolt" {
    var serializer = ArkworksSerializer(BN254Scalar).init(testing.allocator);
    defer serializer.deinit();

    const scalar = BN254Scalar.fromU64(42);
    try serializer.writeFieldElement(scalar);

    // Expected from Jolt: [2a, 00, 00, 00, ...]
    const expected = [_]u8{
        0x2a, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    };

    try testing.expectEqualSlices(u8, &expected, serializer.bytes());
}

test "jolt serialization: Fr(0) matches Jolt" {
    var serializer = ArkworksSerializer(BN254Scalar).init(testing.allocator);
    defer serializer.deinit();

    const scalar = BN254Scalar.zero();
    try serializer.writeFieldElement(scalar);

    // Expected from Jolt: all zeros
    const expected = [_]u8{0} ** 32;

    try testing.expectEqualSlices(u8, &expected, serializer.bytes());
}

test "jolt serialization: Fr(1) matches Jolt" {
    var serializer = ArkworksSerializer(BN254Scalar).init(testing.allocator);
    defer serializer.deinit();

    const scalar = BN254Scalar.one();
    try serializer.writeFieldElement(scalar);

    // Expected from Jolt: [01, 00, 00, 00, ...]
    const expected = [_]u8{
        0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    };

    try testing.expectEqualSlices(u8, &expected, serializer.bytes());
}

test "jolt serialization: Fr(0xDEADBEEF) matches Jolt" {
    var serializer = ArkworksSerializer(BN254Scalar).init(testing.allocator);
    defer serializer.deinit();

    const scalar = BN254Scalar.fromU64(0xDEADBEEF);
    try serializer.writeFieldElement(scalar);

    // Expected from Jolt: [ef, be, ad, de, 00, 00, 00, 00, ...]
    const expected = [_]u8{
        0xef, 0xbe, 0xad, 0xde, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    };

    try testing.expectEqualSlices(u8, &expected, serializer.bytes());
}

test "jolt serialization: usize(1234567890) matches Jolt" {
    var serializer = ArkworksSerializer(BN254Scalar).init(testing.allocator);
    defer serializer.deinit();

    try serializer.writeUsize(1234567890);

    // Expected from Jolt: [d2, 02, 96, 49, 00, 00, 00, 00]
    const expected = [_]u8{ 0xd2, 0x02, 0x96, 0x49, 0x00, 0x00, 0x00, 0x00 };

    try testing.expectEqualSlices(u8, &expected, serializer.bytes());
}

test "arkworks serializer: usize as u64" {
    var serializer = ArkworksSerializer(BN254Scalar).init(testing.allocator);
    defer serializer.deinit();

    try serializer.writeUsize(1234567890);

    try testing.expectEqual(@as(usize, 8), serializer.bytes().len);

    const value = std.mem.readInt(u64, serializer.bytes()[0..8], .little);
    try testing.expectEqual(@as(u64, 1234567890), value);
}

test "arkworks deserializer: roundtrip" {
    var serializer = ArkworksSerializer(BN254Scalar).init(testing.allocator);
    defer serializer.deinit();

    const original = BN254Scalar.fromU64(0xDEADBEEF);
    try serializer.writeFieldElement(original);

    var deserializer = ArkworksDeserializer(BN254Scalar).init(serializer.bytes());
    const decoded = try deserializer.readFieldElement();

    try testing.expect(original.eql(decoded));
}

test "arkworks serializer: sumcheck instance proof" {
    const Proof = jolt_types.SumcheckInstanceProof(BN254Scalar);
    var proof = Proof.init(testing.allocator);
    defer proof.deinit();

    const coeffs = [_]BN254Scalar{
        BN254Scalar.fromU64(1),
        BN254Scalar.fromU64(2),
        BN254Scalar.fromU64(3),
    };
    try proof.addRoundPoly(&coeffs);
    try proof.addRoundPoly(&coeffs);

    var serializer = ArkworksSerializer(BN254Scalar).init(testing.allocator);
    defer serializer.deinit();

    try serializer.writeSumcheckInstanceProof(&proof);

    // Expected: 8 (num_polys) + 2 * (8 (len) + 3 * 32 (coeffs)) = 8 + 2 * 104 = 216
    try testing.expectEqual(@as(usize, 216), serializer.bytes().len);
}

test "arkworks serializer: opening claims" {
    const Claims = jolt_types.OpeningClaims(BN254Scalar);
    var claims = Claims.init(testing.allocator);
    defer claims.deinit();

    try claims.insert(.{ .UntrustedAdvice = .SpartanOuter }, BN254Scalar.fromU64(100));
    try claims.insert(.{ .TrustedAdvice = .RamValEvaluation }, BN254Scalar.fromU64(200));

    var serializer = ArkworksSerializer(BN254Scalar).init(testing.allocator);
    defer serializer.deinit();

    try serializer.writeOpeningClaims(&claims);

    // Expected: 8 (len) + 2 * (1 (id) + 32 (claim)) = 8 + 66 = 74
    try testing.expectEqual(@as(usize, 74), serializer.bytes().len);
}

test "arkworks serializer: uni skip first round proof" {
    const coeffs = [_]BN254Scalar{
        BN254Scalar.fromU64(1),
        BN254Scalar.fromU64(2),
        BN254Scalar.fromU64(3),
        BN254Scalar.fromU64(4),
    };

    var proof = try jolt_types.UniSkipFirstRoundProof(BN254Scalar).init(testing.allocator, &coeffs);
    defer proof.deinit();

    var serializer = ArkworksSerializer(BN254Scalar).init(testing.allocator);
    defer serializer.deinit();

    try serializer.writeUniSkipFirstRoundProof(&proof);

    // Expected: 8 (len) + 4 * 32 (coeffs) = 136
    try testing.expectEqual(@as(usize, 136), serializer.bytes().len);
}

// =============================================================================
// End-to-End JoltProof Serialization Test
// =============================================================================

test "e2e: JoltProof serialization matches Jolt format" {
    // This test creates a complete JoltProof and serializes it,
    // verifying the output structure matches Jolt's expectations.

    const F = BN254Scalar;

    // Use simple dummy types for commitment and proof
    const DummyCommitment = struct {
        value: u64,
        fn toBytes(self: @This()) [32]u8 {
            var buf: [32]u8 = [_]u8{0} ** 32;
            std.mem.writeInt(u64, buf[0..8], self.value, .little);
            return buf;
        }
    };
    const DummyProof = struct {
        data: [32]u8,
    };

    // Create a JoltProof with some test data
    var jolt_proof = jolt_types.JoltProof(F, DummyCommitment, DummyProof).init(testing.allocator);
    defer jolt_proof.deinit();

    // Set configuration
    jolt_proof.trace_length = 16;
    jolt_proof.ram_K = 1024;
    jolt_proof.bytecode_K = 65536;
    jolt_proof.log_k_chunk = 4; // Must be <= 8 to match Jolt
    jolt_proof.lookups_ra_virtual_log_k_chunk = 16; // LOG_K / 8 = 128 / 8

    // Add some commitments
    try jolt_proof.commitments.append(testing.allocator, .{ .value = 123 });
    try jolt_proof.commitments.append(testing.allocator, .{ .value = 456 });

    // Add some opening claims
    try jolt_proof.opening_claims.insert(
        .{ .UntrustedAdvice = .SpartanOuter },
        F.fromU64(100),
    );
    try jolt_proof.opening_claims.insert(
        .{ .Virtual = .{ .poly = .Product, .sumcheck_id = .SpartanOuter } },
        F.fromU64(200),
    );

    // Add round polynomials to stage 1
    const coeffs1 = [_]F{ F.fromU64(1), F.fromU64(2), F.fromU64(3) };
    try jolt_proof.stage1_sumcheck_proof.addRoundPoly(&coeffs1);

    // Create UniSkip for stage 1
    jolt_proof.stage1_uni_skip_first_round_proof = try jolt_types.UniSkipFirstRoundProof(F).init(
        testing.allocator,
        &coeffs1,
    );

    // Add round polynomials to other stages
    const coeffs2 = [_]F{ F.fromU64(10), F.fromU64(20) };
    try jolt_proof.stage2_sumcheck_proof.addRoundPoly(&coeffs2);
    try jolt_proof.stage3_sumcheck_proof.addRoundPoly(&coeffs2);
    try jolt_proof.stage4_sumcheck_proof.addRoundPoly(&coeffs2);
    try jolt_proof.stage5_sumcheck_proof.addRoundPoly(&coeffs2);
    try jolt_proof.stage6_sumcheck_proof.addRoundPoly(&coeffs2);
    try jolt_proof.stage7_sumcheck_proof.addRoundPoly(&coeffs2);

    // Create UniSkip for stage 2
    jolt_proof.stage2_uni_skip_first_round_proof = try jolt_types.UniSkipFirstRoundProof(F).init(
        testing.allocator,
        &coeffs2,
    );

    // Serialize the proof
    var serializer = ArkworksSerializer(F).init(testing.allocator);
    defer serializer.deinit();

    // Define serialization functions for dummy types
    const writeCommitment = struct {
        fn f(ser: *ArkworksSerializer(F), c: DummyCommitment) !void {
            try ser.writeBytes(&c.toBytes());
        }
    }.f;

    const writeProof = struct {
        fn f(ser: *ArkworksSerializer(F), p: DummyProof) !void {
            try ser.writeBytes(&p.data);
        }
    }.f;

    try serializer.writeJoltProof(
        DummyCommitment,
        DummyProof,
        &jolt_proof,
        writeCommitment,
        writeProof,
    );

    const bytes = serializer.bytes();

    // Verify the serialized output has reasonable size
    // Minimum expected:
    // - Opening claims: 8 (len) + 2 * (1 + 32) = 74
    // - Commitments: 8 (len) + 2 * 32 = 72
    // - UniSkip proofs: 2 * (8 + n*32)
    // - 7 sumcheck proofs: each has length prefix + polynomials
    // - Config: 5 * 8 = 40
    try testing.expect(bytes.len > 200);

    // Verify the first 8 bytes are the opening claims length (2)
    const claims_len = std.mem.readInt(u64, bytes[0..8], .little);
    try testing.expectEqual(@as(u64, 2), claims_len);

    // Skip past the opening claims and verify commitments length
    // Claims: 2 entries * (1 + 32) = 66 bytes, plus 8 byte length prefix
    const commitments_offset: usize = 8 + 66;
    const commitments_len = std.mem.readInt(u64, bytes[commitments_offset..][0..8], .little);
    try testing.expectEqual(@as(u64, 2), commitments_len);
}

test "e2e: empty JoltProof serialization" {
    // Test that an empty proof serializes correctly
    const F = BN254Scalar;
    const DummyCommitment = struct { value: u64 };
    const DummyProof = struct { data: [32]u8 };

    var jolt_proof = jolt_types.JoltProof(F, DummyCommitment, DummyProof).init(testing.allocator);
    defer jolt_proof.deinit();

    // Set minimum configuration
    jolt_proof.trace_length = 0;
    jolt_proof.ram_K = 0;
    jolt_proof.bytecode_K = 0;
    jolt_proof.log_k_chunk = 0;
    jolt_proof.lookups_ra_virtual_log_k_chunk = 0;

    var serializer = ArkworksSerializer(F).init(testing.allocator);
    defer serializer.deinit();

    const writeCommitment = struct {
        fn f(_: *ArkworksSerializer(F), _: DummyCommitment) !void {}
    }.f;
    const writeProof = struct {
        fn f(_: *ArkworksSerializer(F), _: DummyProof) !void {}
    }.f;

    try serializer.writeJoltProof(
        DummyCommitment,
        DummyProof,
        &jolt_proof,
        writeCommitment,
        writeProof,
    );

    const bytes = serializer.bytes();

    // Empty proof should still have:
    // - Opening claims length (8 bytes, value 0)
    // - Commitments length (8 bytes, value 0)
    // - 7 sumcheck proofs with 0 polynomials each (7 * 8 = 56 bytes)
    // - Optional proofs (5 bytes for None markers)
    // - Config (5 * 8 = 40 bytes)
    try testing.expect(bytes.len >= 8 + 8 + 56 + 5 + 40);

    // First 8 bytes should be 0 (empty opening claims)
    const claims_len = std.mem.readInt(u64, bytes[0..8], .little);
    try testing.expectEqual(@as(u64, 0), claims_len);
}

test "dory commitment serialization" {
    // Test GT/Dory commitment serialization
    var serializer = ArkworksSerializer(BN254Scalar).init(testing.allocator);
    defer serializer.deinit();

    // Create a simple GT element (one)
    const gt = GT.one();

    // Serialize
    try serializer.writeDoryCommitment(gt);

    // Should be 384 bytes
    try testing.expectEqual(@as(usize, 384), serializer.bytes().len);

    // First 32 bytes should be [1, 0, 0, ...] (c0.c0.c0 = Fp.one())
    const first_value = std.mem.readInt(u64, serializer.bytes()[0..8], .little);
    try testing.expectEqual(@as(u64, 1), first_value);
}

test "dory commitment serialization roundtrip" {
    // Create a non-trivial GT element
    const fp = @import("../field/mod.zig").BN254BaseField;
    const Fp2 = pairing.Fp2;
    const Fp6 = pairing.Fp6;

    const gt = GT{
        .c0 = Fp6{
            .c0 = Fp2.init(fp.fromU64(11), fp.fromU64(22)),
            .c1 = Fp2.init(fp.fromU64(33), fp.fromU64(44)),
            .c2 = Fp2.init(fp.fromU64(55), fp.fromU64(66)),
        },
        .c1 = Fp6{
            .c0 = Fp2.init(fp.fromU64(77), fp.fromU64(88)),
            .c1 = Fp2.init(fp.fromU64(99), fp.fromU64(111)),
            .c2 = Fp2.init(fp.fromU64(122), fp.fromU64(133)),
        },
    };

    // Serialize
    var serializer = ArkworksSerializer(BN254Scalar).init(testing.allocator);
    defer serializer.deinit();
    try serializer.writeDoryCommitment(gt);

    // Deserialize
    var deserializer = ArkworksDeserializer(BN254Scalar).init(serializer.bytes());
    const decoded = try deserializer.readDoryCommitment();

    // Verify equality
    try testing.expect(gt.eql(decoded));
}
