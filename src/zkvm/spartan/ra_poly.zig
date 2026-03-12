//! RaPolynomial: compressed read-address polynomial representation.
//!
//! During Stage 6 init, ra polynomials are computed as eq_table[chunk_val(j)]
//! for each cycle j. Instead of storing T field elements (32 bytes each),
//! store u8 indices into a small eq table (k_chunk entries, typically 16).
//! This reduces init memory from 32T to ~T bytes per polynomial.
//!
//! After the first sumcheck round (bind), materializes to dense []F at half size.

const std = @import("std");
const Allocator = std.mem.Allocator;

/// Maximum log_k_chunk supported by u8 index representation.
pub const MAX_LOG_K_CHUNK: usize = 8;

pub fn RaPolynomial(comptime F: type) type {
    return union(enum) {
        /// Round 1: compressed u8 indices + small eq table
        round1: Round1,
        /// After first bind: dense field element array
        dense: DenseState,

        const Round1 = struct {
            /// T-sized array of u8 indices into eq_table. null = zero (no valid chunk).
            indices: []?u8,
            /// Small eq table (k_chunk entries, typically 16). Owned.
            eq_table: []F,
            /// Pre-scaling factor (e.g., gamma power for first poly in batch)
            scale: F,

            pub inline fn getBoundCoeff(self: @This(), j: usize) F {
                return if (self.indices[j]) |idx|
                    self.eq_table[idx].mul(self.scale)
                else
                    F.zero();
            }

            pub inline fn len(self: @This()) usize {
                return self.indices.len;
            }
        };

        const DenseState = struct {
            /// Full allocation (never resized). Only coeffs[0..current_len] is valid.
            coeffs: []F,
            /// Current logical length (halved each bind).
            current_len: usize,

            pub inline fn getBoundCoeff(self: @This(), j: usize) F {
                return self.coeffs[j];
            }
        };

        pub inline fn getBoundCoeff(self: @This(), j: usize) F {
            return switch (self) {
                .round1 => |s| s.getBoundCoeff(j),
                .dense => |s| s.getBoundCoeff(j),
            };
        }

        /// Bind one sumcheck variable: transitions round1 → dense (materialized at half size).
        /// For dense state, performs in-place MLE bind.
        pub fn bind(self: *@This(), r: F, allocator: Allocator) !void {
            switch (self.*) {
                .round1 => |*s| {
                    // Materialize to dense at half size
                    const half = s.indices.len / 2;
                    const coeffs = try allocator.alloc(F, half);
                    for (0..half) |j| {
                        const v0 = s.getBoundCoeff(2 * j);
                        const v1 = s.getBoundCoeff(2 * j + 1);
                        coeffs[j] = v0.add(r.mul(v1.sub(v0)));
                    }
                    allocator.free(s.indices);
                    allocator.free(s.eq_table);
                    self.* = .{ .dense = .{ .coeffs = coeffs, .current_len = half } };
                },
                .dense => |*s| {
                    const half = s.current_len / 2;
                    for (0..half) |j| {
                        s.coeffs[j] = s.coeffs[2 * j].add(r.mul(s.coeffs[2 * j + 1].sub(s.coeffs[2 * j])));
                    }
                    s.current_len = half;
                },
            }
        }

        /// Get the final scalar value after all rounds complete.
        pub inline fn finalClaim(self: @This()) F {
            return switch (self) {
                .round1 => |s| s.getBoundCoeff(0),
                .dense => |s| s.coeffs[0],
            };
        }

        /// Free resources.
        pub fn deinit(self: *@This(), allocator: Allocator) void {
            switch (self.*) {
                .round1 => |*s| {
                    allocator.free(s.indices);
                    allocator.free(s.eq_table);
                },
                .dense => |*s| {
                    allocator.free(s.coeffs);
                },
            }
        }
    };
}

// =============================================================================
// Tests
// =============================================================================

test "RaPolynomial round1 getBoundCoeff" {
    const BN254Scalar = @import("../../field/mod.zig").BN254Scalar;
    const allocator = std.testing.allocator;

    // eq_table with 4 entries
    var eq_table = try allocator.alloc(BN254Scalar, 4);
    defer allocator.free(eq_table);
    eq_table[0] = BN254Scalar.fromU64(10);
    eq_table[1] = BN254Scalar.fromU64(20);
    eq_table[2] = BN254Scalar.fromU64(30);
    eq_table[3] = BN254Scalar.fromU64(40);

    const scale = BN254Scalar.fromU64(3);

    var indices = try allocator.alloc(?u8, 4);
    defer allocator.free(indices);
    indices[0] = 0; // → eq_table[0] * scale = 30
    indices[1] = 2; // → eq_table[2] * scale = 90
    indices[2] = null; // → zero
    indices[3] = 1; // → eq_table[1] * scale = 60

    const RaPoly = RaPolynomial(BN254Scalar);
    const poly: RaPoly = .{ .round1 = .{
        .indices = indices,
        .eq_table = eq_table,
        .scale = scale,
    } };

    try std.testing.expect(poly.getBoundCoeff(0).eql(BN254Scalar.fromU64(30)));
    try std.testing.expect(poly.getBoundCoeff(1).eql(BN254Scalar.fromU64(90)));
    try std.testing.expect(poly.getBoundCoeff(2).eql(BN254Scalar.zero()));
    try std.testing.expect(poly.getBoundCoeff(3).eql(BN254Scalar.fromU64(60)));
}

test "RaPolynomial round1 bind matches dense computation" {
    const BN254Scalar = @import("../../field/mod.zig").BN254Scalar;
    const allocator = std.testing.allocator;

    // Build a round1 poly with 8 elements
    var eq_table = try allocator.alloc(BN254Scalar, 4);
    eq_table[0] = BN254Scalar.fromU64(10);
    eq_table[1] = BN254Scalar.fromU64(20);
    eq_table[2] = BN254Scalar.fromU64(30);
    eq_table[3] = BN254Scalar.fromU64(40);

    const scale = BN254Scalar.fromU64(2);
    var indices = try allocator.alloc(?u8, 8);
    indices[0] = 0;
    indices[1] = 1;
    indices[2] = null;
    indices[3] = 3;
    indices[4] = 2;
    indices[5] = 0;
    indices[6] = 1;
    indices[7] = null;

    // Compute expected dense values before bind
    const expected_dense = [8]BN254Scalar{
        BN254Scalar.fromU64(20), // eq[0]*2 = 20
        BN254Scalar.fromU64(40), // eq[1]*2 = 40
        BN254Scalar.zero(), // null
        BN254Scalar.fromU64(80), // eq[3]*2 = 80
        BN254Scalar.fromU64(60), // eq[2]*2 = 60
        BN254Scalar.fromU64(20), // eq[0]*2 = 20
        BN254Scalar.fromU64(40), // eq[1]*2 = 40
        BN254Scalar.zero(), // null
    };

    // Compute expected bind result: v0 + r*(v1-v0) for each pair
    const r = BN254Scalar.fromU64(7);
    var expected_bound: [4]BN254Scalar = undefined;
    for (0..4) |j| {
        const v0 = expected_dense[2 * j];
        const v1 = expected_dense[2 * j + 1];
        expected_bound[j] = v0.add(r.mul(v1.sub(v0)));
    }

    // Bind the round1 poly
    const RaPoly = RaPolynomial(BN254Scalar);
    var poly: RaPoly = .{ .round1 = .{
        .indices = indices,
        .eq_table = eq_table,
        .scale = scale,
    } };
    try poly.bind(r, allocator);
    defer poly.deinit(allocator);

    // Should now be dense
    try std.testing.expect(poly == .dense);
    try std.testing.expectEqual(@as(usize, 4), poly.dense.current_len);

    for (0..4) |j| {
        try std.testing.expect(poly.dense.coeffs[j].eql(expected_bound[j]));
    }
}

test "RaPolynomial full sumcheck simulation" {
    const BN254Scalar = @import("../../field/mod.zig").BN254Scalar;
    const allocator = std.testing.allocator;

    // 4-element poly (2 rounds of bind to reach scalar)
    var eq_table = try allocator.alloc(BN254Scalar, 2);
    eq_table[0] = BN254Scalar.fromU64(5);
    eq_table[1] = BN254Scalar.fromU64(15);

    var indices = try allocator.alloc(?u8, 4);
    indices[0] = 0; // 5
    indices[1] = 1; // 15
    indices[2] = 1; // 15
    indices[3] = null; // 0

    const scale = BN254Scalar.one();
    const RaPoly = RaPolynomial(BN254Scalar);
    var poly: RaPoly = .{ .round1 = .{
        .indices = indices,
        .eq_table = eq_table,
        .scale = scale,
    } };

    // Round 1 bind (round1 → dense, 4 → 2)
    const r1 = BN254Scalar.fromU64(3);
    try poly.bind(r1, allocator);
    try std.testing.expect(poly == .dense);
    try std.testing.expectEqual(@as(usize, 2), poly.dense.current_len);

    // Round 2 bind (dense → dense, 2 → 1)
    const r2 = BN254Scalar.fromU64(11);
    try poly.bind(r2, allocator);
    try std.testing.expectEqual(@as(usize, 1), poly.dense.current_len);

    // finalClaim should match MLE evaluation
    const final = poly.finalClaim();

    // Compute MLE directly: f(r1, r2) = Σ_{x0,x1} eq((r1,r2),(x0,x1)) * val(x0,x1)
    // val = [5, 15, 15, 0] with LE indexing (bit 0 = x0, bit 1 = x1)
    // eq((r1,r2),(x0,x1)) = eq_bit(r1,x0) * eq_bit(r2,x1)
    // where eq_bit(a,0) = 1-a, eq_bit(a,1) = a
    const vals = [4]BN254Scalar{
        BN254Scalar.fromU64(5),
        BN254Scalar.fromU64(15),
        BN254Scalar.fromU64(15),
        BN254Scalar.zero(),
    };
    const one = BN254Scalar.one();
    var expected = BN254Scalar.zero();
    for (0..4) |idx| {
        const x0: u1 = @truncate(idx);
        const x1: u1 = @truncate(idx >> 1);
        const eq_x0 = if (x0 == 1) r1 else one.sub(r1);
        const eq_x1 = if (x1 == 1) r2 else one.sub(r2);
        expected = expected.add(eq_x0.mul(eq_x1).mul(vals[idx]));
    }

    try std.testing.expect(final.eql(expected));
    poly.deinit(allocator);
}

test "RaPolynomial deinit in round1 state" {
    const BN254Scalar = @import("../../field/mod.zig").BN254Scalar;
    const allocator = std.testing.allocator;

    var eq_table = try allocator.alloc(BN254Scalar, 2);
    eq_table[0] = BN254Scalar.one();
    eq_table[1] = BN254Scalar.fromU64(2);

    var indices = try allocator.alloc(?u8, 4);
    indices[0] = 0;
    indices[1] = 1;
    indices[2] = null;
    indices[3] = 0;

    const RaPoly = RaPolynomial(BN254Scalar);
    var poly: RaPoly = .{ .round1 = .{
        .indices = indices,
        .eq_table = eq_table,
        .scale = BN254Scalar.one(),
    } };

    // deinit without binding — should free round1 resources
    poly.deinit(allocator);
}
