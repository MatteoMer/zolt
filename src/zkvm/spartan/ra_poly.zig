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
            /// Small eq table (k_chunk entries, typically 16). Prescaled by `scale`. Owned.
            eq_table: []F,

            pub inline fn getBoundCoeff(self: @This(), j: usize) F {
                return if (self.indices[j]) |idx|
                    self.eq_table[idx]
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
                std.debug.assert(j < self.current_len);
                return self.coeffs[j];
            }
        };

        /// Initialize a round1 RaPolynomial. Takes ownership of both `indices` and `eq_table`;
        /// caller must not access either array after this call. Prescales `eq_table` entries
        /// in-place by `scale` so that per-access multiplication is avoided.
        /// Asserts eq_table fits in u8 index range, indices length is a power of two,
        /// and all non-null indices are within eq_table bounds.
        pub fn initRound1(indices: []?u8, eq_table: []F, scale: F) @This() {
            std.debug.assert(indices.len > 0 and std.math.isPowerOfTwo(indices.len));
            std.debug.assert(eq_table.len <= (@as(usize, 1) << MAX_LOG_K_CHUNK));
            // Validate all non-null indices are within eq_table bounds
            if (std.debug.runtime_safety) {
                for (indices) |maybe_idx| {
                    if (maybe_idx) |idx| {
                        std.debug.assert(idx < eq_table.len);
                    }
                }
            }
            // Prescale eq_table entries
            for (eq_table) |*entry| {
                entry.* = entry.*.mul(scale);
            }
            return .{ .round1 = .{
                .indices = indices,
                .eq_table = eq_table,
            } };
        }

        pub inline fn getBoundCoeff(self: @This(), j: usize) F {
            return switch (self) {
                .round1 => |s| s.getBoundCoeff(j),
                .dense => |s| s.getBoundCoeff(j),
            };
        }

        pub inline fn currentLen(self: @This()) usize {
            return switch (self) {
                .round1 => |s| s.len(),
                .dense => |s| s.current_len,
            };
        }

        /// Bind one sumcheck variable: transitions round1 → dense (materialized at half size).
        /// For dense state, performs in-place MLE bind.
        /// The only possible error is OOM during the round1→dense allocation, in which
        /// case `self` is unchanged and must still be deinit'd.
        pub fn bind(self: *@This(), r: F, allocator: Allocator) !void {
            switch (self.*) {
                .round1 => |*s| {
                    std.debug.assert(s.indices.len > 0 and std.math.isPowerOfTwo(s.indices.len));
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
                    std.debug.assert(s.current_len > 0 and std.math.isPowerOfTwo(s.current_len));
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
                .round1 => |s| {
                    std.debug.assert(s.indices.len == 1);
                    return s.getBoundCoeff(0);
                },
                .dense => |s| {
                    std.debug.assert(s.current_len == 1);
                    return s.coeffs[0];
                },
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
    const poly = RaPoly.initRound1(indices, eq_table, scale);

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

    // Compute expected dense values before bind (eq_table prescaled by 2)
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
    var poly = RaPoly.initRound1(indices, eq_table, scale);
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

    const RaPoly = RaPolynomial(BN254Scalar);
    var poly = RaPoly.initRound1(indices, eq_table, BN254Scalar.one());

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
    var poly = RaPoly.initRound1(indices, eq_table, BN254Scalar.one());

    // deinit without binding — should free round1 resources
    poly.deinit(allocator);
}

test "RaPolynomial scale zero returns all zeros" {
    const BN254Scalar = @import("../../field/mod.zig").BN254Scalar;
    const allocator = std.testing.allocator;

    var eq_table = try allocator.alloc(BN254Scalar, 2);
    eq_table[0] = BN254Scalar.fromU64(100);
    eq_table[1] = BN254Scalar.fromU64(200);

    var indices = try allocator.alloc(?u8, 4);
    indices[0] = 0;
    indices[1] = 1;
    indices[2] = null;
    indices[3] = 1;

    const RaPoly = RaPolynomial(BN254Scalar);
    var poly = RaPoly.initRound1(indices, eq_table, BN254Scalar.zero());
    defer poly.deinit(allocator);

    // All getBoundCoeff should return zero when scale is zero
    for (0..4) |j| {
        try std.testing.expect(poly.getBoundCoeff(j).eql(BN254Scalar.zero()));
    }
}

test "RaPolynomial multi-round dense bind (16 elements, 4 rounds)" {
    const BN254Scalar = @import("../../field/mod.zig").BN254Scalar;
    const allocator = std.testing.allocator;

    // Build 16-element poly with 2-entry eq_table
    var eq_table = try allocator.alloc(BN254Scalar, 2);
    eq_table[0] = BN254Scalar.fromU64(3);
    eq_table[1] = BN254Scalar.fromU64(7);

    var indices = try allocator.alloc(?u8, 16);
    // Pattern: alternating indices and nulls
    for (0..16) |j| {
        indices[j] = if (j % 3 == 0) null else @intCast(j % 2);
    }

    // Manually compute expected dense values (prescaled by scale=1)
    var expected: [16]BN254Scalar = .{BN254Scalar.zero()} ** 16;
    for (0..16) |j| {
        expected[j] = if (indices[j]) |idx| eq_table[idx] else BN254Scalar.zero();
    }

    const challenges = [4]BN254Scalar{
        BN254Scalar.fromU64(5),
        BN254Scalar.fromU64(13),
        BN254Scalar.fromU64(2),
        BN254Scalar.fromU64(19),
    };

    // Compute expected result by applying 4 rounds of MLE bind
    var current = expected;
    var cur_len: usize = 16;
    for (challenges) |r| {
        const half = cur_len / 2;
        for (0..half) |j| {
            current[j] = current[2 * j].add(r.mul(current[2 * j + 1].sub(current[2 * j])));
        }
        cur_len = half;
    }

    const RaPoly = RaPolynomial(BN254Scalar);
    var poly = RaPoly.initRound1(indices, eq_table, BN254Scalar.one());

    // Bind 4 rounds (round1→dense on first, then 3 dense→dense)
    for (challenges) |r| {
        try poly.bind(r, allocator);
    }

    try std.testing.expectEqual(@as(usize, 1), poly.dense.current_len);
    try std.testing.expect(poly.finalClaim().eql(current[0]));
    poly.deinit(allocator);
}

test "RaPolynomial single element finalClaim in round1 state" {
    const BN254Scalar = @import("../../field/mod.zig").BN254Scalar;
    const allocator = std.testing.allocator;

    var eq_table = try allocator.alloc(BN254Scalar, 2);
    eq_table[0] = BN254Scalar.fromU64(42);
    eq_table[1] = BN254Scalar.fromU64(99);

    var indices = try allocator.alloc(?u8, 1);
    indices[0] = 1;

    const RaPoly = RaPolynomial(BN254Scalar);
    var poly = RaPoly.initRound1(indices, eq_table, BN254Scalar.fromU64(3));
    defer poly.deinit(allocator);

    // Single element, no bind needed — finalClaim should return eq_table[1] * scale = 297
    try std.testing.expect(poly.finalClaim().eql(BN254Scalar.fromU64(297)));
}

test "RaPolynomial non-unit scale full sumcheck to scalar" {
    const BN254Scalar = @import("../../field/mod.zig").BN254Scalar;
    const allocator = std.testing.allocator;

    // 8-element poly with scale=7, bind 3 rounds to a scalar, verify against direct MLE
    const scale = BN254Scalar.fromU64(7);
    var eq_table = try allocator.alloc(BN254Scalar, 4);
    eq_table[0] = BN254Scalar.fromU64(10);
    eq_table[1] = BN254Scalar.fromU64(20);
    eq_table[2] = BN254Scalar.fromU64(30);
    eq_table[3] = BN254Scalar.fromU64(40);

    var indices = try allocator.alloc(?u8, 8);
    indices[0] = 0; // val = 10*7 = 70
    indices[1] = 3; // val = 40*7 = 280
    indices[2] = null; // val = 0
    indices[3] = 1; // val = 20*7 = 140
    indices[4] = 2; // val = 30*7 = 210
    indices[5] = null; // val = 0
    indices[6] = 0; // val = 10*7 = 70
    indices[7] = 3; // val = 40*7 = 280

    // Compute expected dense values (prescaled)
    var expected_vals: [8]BN254Scalar = undefined;
    for (0..8) |j| {
        expected_vals[j] = if (indices[j]) |idx| eq_table[idx].mul(scale) else BN254Scalar.zero();
    }

    const challenges = [3]BN254Scalar{
        BN254Scalar.fromU64(13),
        BN254Scalar.fromU64(29),
        BN254Scalar.fromU64(41),
    };

    // Compute expected MLE: f(r0,r1,r2) = Σ_{x} eq((r0,r1,r2), x) * val(x)  (LE indexing)
    const one = BN254Scalar.one();
    var expected = BN254Scalar.zero();
    for (0..8) |idx| {
        var eq_prod = BN254Scalar.one();
        for (0..3) |bit| {
            const xi: u1 = @truncate(idx >> @intCast(bit));
            eq_prod = eq_prod.mul(if (xi == 1) challenges[bit] else one.sub(challenges[bit]));
        }
        expected = expected.add(eq_prod.mul(expected_vals[idx]));
    }

    const RaPoly = RaPolynomial(BN254Scalar);
    var poly = RaPoly.initRound1(indices, eq_table, scale);

    for (challenges) |r| {
        try poly.bind(r, allocator);
    }

    try std.testing.expectEqual(@as(usize, 1), poly.dense.current_len);
    try std.testing.expect(poly.finalClaim().eql(expected));
    poly.deinit(allocator);
}

test "RaPolynomial currentLen tracks through transitions" {
    const BN254Scalar = @import("../../field/mod.zig").BN254Scalar;
    const allocator = std.testing.allocator;

    var eq_table = try allocator.alloc(BN254Scalar, 2);
    eq_table[0] = BN254Scalar.fromU64(1);
    eq_table[1] = BN254Scalar.fromU64(2);

    var indices = try allocator.alloc(?u8, 8);
    for (0..8) |j| indices[j] = @intCast(j % 2);

    const RaPoly = RaPolynomial(BN254Scalar);
    var poly = RaPoly.initRound1(indices, eq_table, BN254Scalar.one());

    try std.testing.expectEqual(@as(usize, 8), poly.currentLen());

    try poly.bind(BN254Scalar.fromU64(5), allocator);
    try std.testing.expectEqual(@as(usize, 4), poly.currentLen());
    try std.testing.expect(poly == .dense);

    try poly.bind(BN254Scalar.fromU64(7), allocator);
    try std.testing.expectEqual(@as(usize, 2), poly.currentLen());

    try poly.bind(BN254Scalar.fromU64(11), allocator);
    try std.testing.expectEqual(@as(usize, 1), poly.currentLen());

    poly.deinit(allocator);
}

test "RaPolynomial all null indices produces zero polynomial" {
    const BN254Scalar = @import("../../field/mod.zig").BN254Scalar;
    const allocator = std.testing.allocator;

    var eq_table = try allocator.alloc(BN254Scalar, 4);
    eq_table[0] = BN254Scalar.fromU64(10);
    eq_table[1] = BN254Scalar.fromU64(20);
    eq_table[2] = BN254Scalar.fromU64(30);
    eq_table[3] = BN254Scalar.fromU64(40);

    var indices = try allocator.alloc(?u8, 8);
    for (0..8) |j| indices[j] = null;

    const RaPoly = RaPolynomial(BN254Scalar);
    var poly = RaPoly.initRound1(indices, eq_table, BN254Scalar.fromU64(5));

    // All coefficients should be zero
    for (0..8) |j| {
        try std.testing.expect(poly.getBoundCoeff(j).eql(BN254Scalar.zero()));
    }

    // Bind all 3 rounds to a scalar
    const challenges = [3]BN254Scalar{
        BN254Scalar.fromU64(7),
        BN254Scalar.fromU64(13),
        BN254Scalar.fromU64(29),
    };
    for (challenges) |r| {
        try poly.bind(r, allocator);
    }

    // Final claim should be zero
    try std.testing.expect(poly.finalClaim().eql(BN254Scalar.zero()));
    poly.deinit(allocator);
}
