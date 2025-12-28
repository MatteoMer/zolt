//! Commitment types for Jolt proofs
//!
//! This module defines the commitment types used in Jolt proofs.
//! These wrap the underlying polynomial commitment scheme (HyperKZG).

const std = @import("std");
const msm = @import("../msm/mod.zig");
const field = @import("../field/mod.zig");

/// Base field for G1 point coordinates
pub const Fp = field.BN254BaseField;

/// G1 Point type used for commitments
pub const G1Point = msm.AffinePoint(Fp);

/// A polynomial commitment (G1 point)
///
/// This is the type used throughout Jolt proofs to represent
/// polynomial commitments. It wraps a G1 point from the elliptic curve.
pub const PolyCommitment = struct {
    const Self = @This();

    /// The underlying G1 point
    point: G1Point,

    /// Create a commitment from a G1 point
    pub fn fromPoint(point: G1Point) Self {
        return .{ .point = point };
    }

    /// Create a zero/identity commitment
    pub fn zero() Self {
        return .{ .point = G1Point.identity() };
    }

    /// Check if two commitments are equal
    pub fn eql(self: Self, other: Self) bool {
        return self.point.x.eql(other.point.x) and
            self.point.y.eql(other.point.y) and
            self.point.infinity == other.point.infinity;
    }

    /// Check if this is the identity/zero commitment
    pub fn isZero(self: Self) bool {
        return self.point.infinity;
    }

    /// Convert to bytes for serialization
    pub fn toBytes(self: Self) [64]u8 {
        var result: [64]u8 = undefined;
        result[0..32].* = self.point.x.toBytes();
        result[32..64].* = self.point.y.toBytes();
        return result;
    }

    /// Create from bytes
    pub fn fromBytes(bytes: [64]u8) Self {
        return .{
            .point = .{
                .x = Fp.fromBytes(bytes[0..32].*),
                .y = Fp.fromBytes(bytes[32..64].*),
                .infinity = false,
            },
        };
    }
};

/// Opening proof for a polynomial commitment
///
/// Contains the quotient commitments and final evaluation.
pub const OpeningProof = struct {
    const Self = @This();

    /// Quotient polynomial commitments (one per variable for HyperKZG)
    quotients: []PolyCommitment,
    /// Final evaluation after all folding
    final_eval: field.BN254Scalar,
    /// Allocator used for quotients array
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, num_vars: usize) !Self {
        const quotients = try allocator.alloc(PolyCommitment, num_vars);
        for (quotients) |*q| {
            q.* = PolyCommitment.zero();
        }
        return .{
            .quotients = quotients,
            .final_eval = field.BN254Scalar.zero(),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        if (self.quotients.len > 0) {
            self.allocator.free(self.quotients);
        }
    }
};

test "poly commitment basic operations" {
    const identity = PolyCommitment.zero();
    try std.testing.expect(identity.isZero());

    const gen = PolyCommitment.fromPoint(G1Point.generator());
    try std.testing.expect(!gen.isZero());
    try std.testing.expect(gen.eql(gen));
    try std.testing.expect(!gen.eql(identity));
}

test "opening proof init and deinit" {
    const allocator = std.testing.allocator;
    var proof = try OpeningProof.init(allocator, 5);
    defer proof.deinit();

    try std.testing.expectEqual(@as(usize, 5), proof.quotients.len);
    for (proof.quotients) |q| {
        try std.testing.expect(q.isZero());
    }
}
