//! Proof serialization for Jolt proofs.
//!
//! Binary format for efficient storage and transmission.
//! This module imports from subprotocols, poly, and zkvm/spartan.

const std = @import("std");
const Allocator = std.mem.Allocator;
const Serialize = @import("mod.zig").Serialize;

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
        pub fn serializeField(writer: *std.Io.Writer, value: F) !void {
            for (value.limbs) |limb| {
                try Serialize.writeU64(writer, limb);
            }
        }

        /// Deserialize a field element from bytes
        pub fn deserializeField(reader: *std.Io.Reader) !F {
            var limbs: [4]u64 = undefined;
            for (&limbs) |*limb| {
                limb.* = try Serialize.readU64(reader);
            }
            return F{ .limbs = limbs };
        }

        /// Serialize an array of field elements
        pub fn serializeFieldArray(writer: *std.Io.Writer, values: []const F) !void {
            try Serialize.writeU64(writer, values.len);
            for (values) |value| {
                try serializeField(writer, value);
            }
        }

        /// Deserialize an array of field elements
        pub fn deserializeFieldArray(reader: *std.Io.Reader, allocator: Allocator) ![]F {
            const len = try Serialize.readU64(reader);
            const values = try allocator.alloc(F, @intCast(len));
            errdefer allocator.free(values);

            for (values) |*value| {
                value.* = try deserializeField(reader);
            }
            return values;
        }

        /// Write proof header with magic and version
        pub fn writeHeader(writer: *std.Io.Writer) !void {
            try writer.writeAll(&MAGIC);
            try Serialize.writeU32(writer, VERSION);
        }

        /// Read and verify proof header
        pub fn readHeader(reader: *std.Io.Reader) !void {
            var magic: [4]u8 = undefined;
            try reader.readSliceAll(&magic);
            if (!std.mem.eql(u8, &magic, &MAGIC)) {
                return error.InvalidProofFormat;
            }

            const version = try Serialize.readU32(reader);
            if (version != VERSION) {
                return error.UnsupportedProofVersion;
            }
        }

        /// Serialize a sumcheck proof
        pub fn serializeSumcheckProof(writer: *std.Io.Writer, proof: anytype) !void {
            try serializeField(writer, proof.claim);

            try Serialize.writeU64(writer, proof.rounds.len);

            for (proof.rounds) |round| {
                try serializeFieldArray(writer, round.poly.coeffs);
            }

            try serializeFieldArray(writer, proof.final_point);

            try serializeField(writer, proof.final_eval);
        }

        /// Deserialize a sumcheck proof
        pub fn deserializeSumcheckProof(reader: *std.Io.Reader, allocator: Allocator) !@import("zolt_arith").subprotocols.Sumcheck(F).Proof {
            const poly_mod = @import("zolt_arith").poly;
            const subprotocols = @import("zolt_arith").subprotocols;

            const claim = try deserializeField(reader);

            const num_rounds = try Serialize.readU64(reader);
            const rounds = try allocator.alloc(subprotocols.Sumcheck(F).Round, @intCast(num_rounds));
            errdefer allocator.free(rounds);

            for (rounds) |*round| {
                const coeffs = try deserializeFieldArray(reader, allocator);
                round.* = .{
                    .poly = poly_mod.UniPoly(F){
                        .coeffs = coeffs,
                        .allocator = allocator,
                    },
                };
            }

            const final_point = try deserializeFieldArray(reader, allocator);

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
        pub fn serializeR1CSProof(writer: *std.Io.Writer, proof: anytype) !void {
            try serializeFieldArray(writer, proof.tau);

            try serializeSumcheckProof(writer, proof.sumcheck_proof);

            for (proof.eval_claims) |eval_claim| {
                try serializeField(writer, eval_claim);
            }

            try serializeFieldArray(writer, proof.eval_point);
        }

        /// Deserialize an R1CS/Spartan proof
        pub fn deserializeR1CSProof(reader: *std.Io.Reader, allocator: Allocator) !@import("../zkvm/spartan/mod.zig").R1CSProof(F) {
            const spartan = @import("../zkvm/spartan/mod.zig");

            const tau = try deserializeFieldArray(reader, allocator);
            errdefer allocator.free(tau);

            const sumcheck_proof = try deserializeSumcheckProof(reader, allocator);

            var eval_claims: [3]F = undefined;
            for (&eval_claims) |*eval_claim| {
                eval_claim.* = try deserializeField(reader);
            }

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
            var list: std.ArrayListUnmanaged(u8) = .empty;
            errdefer list.deinit(allocator);

            var aw: std.Io.Writer.Allocating = .fromArrayList(allocator, &list);

            try writeHeader(&aw.writer);

            const ProofType = @TypeOf(proof);
            const type_info = @typeInfo(ProofType);

            if (type_info == .@"struct") {
                if (@hasField(ProofType, "sumcheck_proof") and @hasField(ProofType, "tau")) {
                    try aw.writer.writeByte(1);
                    try serializeR1CSProof(&aw.writer, proof);
                } else if (@hasField(ProofType, "rounds") and @hasField(ProofType, "claim")) {
                    try aw.writer.writeByte(0);
                    try serializeSumcheckProof(&aw.writer, proof);
                }
            }

            list = aw.toArrayList();
            return list.toOwnedSlice();
        }

        /// Compute a simple hash of proof bytes for integrity verification
        pub fn computeHash(data: []const u8) [32]u8 {
            var hash: [32]u8 = .{0} ** 32;

            for (data, 0..) |byte, i| {
                const idx = i % 32;
                hash[idx] ^= byte;
                hash[(idx + 1) % 32] ^= byte >> 4;
            }

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
    const field_mod = @import("zolt_arith").field;
    const F = field_mod.BN254Scalar;

    const PS = ProofSerializer(F);

    const original = F.fromU64(12345678);

    var buffer: [256]u8 = undefined;
    var writer = std.Io.Writer.fixed(&buffer);

    try PS.serializeField(&writer, original);

    var reader = std.Io.Reader.fixed(buffer[0..writer.end]);
    const deserialized = try PS.deserializeField(&reader);

    try std.testing.expect(original.eql(deserialized));
}

test "field array serialization" {
    const field_mod = @import("zolt_arith").field;
    const F = field_mod.BN254Scalar;
    const allocator = std.testing.allocator;

    const PS = ProofSerializer(F);

    const original = [_]F{
        F.fromU64(1),
        F.fromU64(2),
        F.fromU64(3),
    };

    var buffer: [512]u8 = undefined;
    var writer = std.Io.Writer.fixed(&buffer);

    try PS.serializeFieldArray(&writer, &original);

    var reader = std.Io.Reader.fixed(buffer[0..writer.end]);
    const deserialized = try PS.deserializeFieldArray(&reader, allocator);
    defer allocator.free(deserialized);

    try std.testing.expectEqual(original.len, deserialized.len);
    for (0..original.len) |i| {
        try std.testing.expect(original[i].eql(deserialized[i]));
    }
}

test "proof hash computation" {
    const F = @import("zolt_arith").field.BN254Scalar;
    const PS = ProofSerializer(F);

    const data1 = "Hello, Zolt!";
    const data2 = "Hello, Zolt?";

    const hash1 = PS.computeHash(data1);
    const hash2 = PS.computeHash(data2);

    try std.testing.expect(!std.mem.eql(u8, &hash1, &hash2));

    const hash1_again = PS.computeHash(data1);
    try std.testing.expect(std.mem.eql(u8, &hash1, &hash1_again));
}
