//! RaPolynomial: compressed read-address polynomial representation.
//!
//! During Stage 6 init, ra polynomials are computed as eq_table[chunk_val(j)]
//! for each cycle j. Instead of storing T field elements (32 bytes each),
//! store u8 indices into a small eq table (k_chunk entries, typically 16).
//! This reduces init memory from 32T to ~T bytes per polynomial.
//!
//! Lazy materialization through 3 rounds:
//!   round1 → round2 → round3 → dense
//! Each transition does O(K) work on the small eq tables, deferring the O(T)
//! dense allocation until round3→dense. This keeps the working set in L1 cache
//! during the first 3 sumcheck rounds and avoids OOM at T=2^30.

const std = @import("std");
const Allocator = std.mem.Allocator;

/// Maximum log_k_chunk supported by u8 index representation.
pub const MAX_LOG_K_CHUNK: usize = 8;

pub fn RaPolynomial(comptime F: type) type {
    return union(enum) {
        /// Round 1: compressed u8 indices + small eq table
        round1: Round1,
        /// Round 2: compressed u8 indices + two lookup tables F_0, F_1
        round2: Round2,
        /// Round 3: compressed u8 indices + four lookup tables F_00..F_11
        round3: Round3,
        /// After round 3 bind: dense field element array
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

        const Round2 = struct {
            /// T/2-logically-sized view of original T-sized indices (shared, NOT owned).
            /// Access pattern: indices[2*j] and indices[2*j+1] for j in 0..T/2.
            indices: []?u8,
            /// F_0[k] = (1-r0)*eq[k], F_1[k] = r0*eq[k]. Owned.
            F_0: []F,
            F_1: []F,

            pub inline fn getBoundCoeff(self: @This(), j: usize) F {
                const v0 = if (self.indices[2 * j]) |idx| self.F_0[idx] else F.zero();
                const v1 = if (self.indices[2 * j + 1]) |idx| self.F_1[idx] else F.zero();
                return v0.add(v1);
            }

            pub inline fn len(self: @This()) usize {
                return self.indices.len / 2;
            }
        };

        const Round3 = struct {
            /// T/4-logically-sized view of original T-sized indices (shared, NOT owned).
            /// Access: indices[4*j..4*j+3] for j in 0..T/4.
            indices: []?u8,
            /// Four lookup tables. Owned.
            F_00: []F,
            F_01: []F,
            F_10: []F,
            F_11: []F,

            /// LE bit ordering: position offset g has bits [b1, b0] where
            /// b0 = round0 selector (F_?0 vs F_?1), b1 = round1 selector (F_0? vs F_1?).
            /// So: base+0 → F_00, base+1 → F_10, base+2 → F_01, base+3 → F_11.
            pub inline fn getBoundCoeff(self: @This(), j: usize) F {
                const base = 4 * j;
                const v00 = if (self.indices[base]) |idx| self.F_00[idx] else F.zero();
                const v10 = if (self.indices[base + 1]) |idx| self.F_10[idx] else F.zero();
                const v01 = if (self.indices[base + 2]) |idx| self.F_01[idx] else F.zero();
                const v11 = if (self.indices[base + 3]) |idx| self.F_11[idx] else F.zero();
                return v00.add(v10).add(v01).add(v11);
            }

            pub inline fn len(self: @This()) usize {
                return self.indices.len / 4;
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
                .round2 => |s| s.getBoundCoeff(j),
                .round3 => |s| s.getBoundCoeff(j),
                .dense => |s| s.getBoundCoeff(j),
            };
        }

        pub inline fn currentLen(self: @This()) usize {
            return switch (self) {
                .round1 => |s| s.len(),
                .round2 => |s| s.len(),
                .round3 => |s| s.len(),
                .dense => |s| s.current_len,
            };
        }

        /// Check if the polynomial is in dense state.
        pub inline fn isDense(self: @This()) bool {
            return self == .dense;
        }

        /// Bind one sumcheck variable. Transitions:
        ///   round1 → round2 (O(K) work, no allocation)
        ///   round2 → round3 (O(K) work, allocates 4 tables of size K)
        ///   round3 → dense  (O(T/8) work, allocates T/8 dense array, frees indices+tables)
        ///   dense  → dense  (in-place MLE bind)
        pub fn bind(self: *@This(), r: F, allocator: Allocator) !void {
            switch (self.*) {
                .round1 => |*s| {
                    const K = s.eq_table.len;
                    const one_minus_r = F.one().sub(r);

                    // Compute F_0[k] = (1-r0)*eq[k], F_1[k] = r0*eq[k]
                    const F_0 = try allocator.alloc(F, K);
                    errdefer allocator.free(F_0);
                    const F_1 = try allocator.alloc(F, K);
                    for (0..K) |k| {
                        F_0[k] = one_minus_r.mul(s.eq_table[k]);
                        F_1[k] = r.mul(s.eq_table[k]);
                    }

                    // Free eq_table (replaced by F_0, F_1)
                    allocator.free(s.eq_table);

                    // indices pointer is shared into round2 (NOT freed)
                    self.* = .{ .round2 = .{
                        .indices = s.indices,
                        .F_0 = F_0,
                        .F_1 = F_1,
                    } };
                },
                .round2 => |*s| {
                    const K = s.F_0.len;
                    const one_minus_r = F.one().sub(r);

                    // Compute 4 tables from 2
                    const F_00 = try allocator.alloc(F, K);
                    errdefer allocator.free(F_00);
                    const F_01 = try allocator.alloc(F, K);
                    errdefer allocator.free(F_01);
                    const F_10 = try allocator.alloc(F, K);
                    errdefer allocator.free(F_10);
                    const F_11 = try allocator.alloc(F, K);
                    errdefer allocator.free(F_11);

                    for (0..K) |k| {
                        F_00[k] = one_minus_r.mul(s.F_0[k]);
                        F_01[k] = r.mul(s.F_0[k]);
                        F_10[k] = one_minus_r.mul(s.F_1[k]);
                        F_11[k] = r.mul(s.F_1[k]);
                    }

                    // Free F_0, F_1
                    allocator.free(s.F_0);
                    allocator.free(s.F_1);

                    self.* = .{ .round3 = .{
                        .indices = s.indices,
                        .F_00 = F_00,
                        .F_01 = F_01,
                        .F_10 = F_10,
                        .F_11 = F_11,
                    } };
                },
                .round3 => |*s| {
                    // Materialize dense array at T/8 (indices.len/4 pairs → /2 after bind)
                    const n = s.indices.len / 4; // logical length before this bind
                    const half = n / 2; // length after bind
                    const one_minus_r = F.one().sub(r);

                    const coeffs = try allocator.alloc(F, half);

                    // For each output j, compute bound coefficient from 8 consecutive indices
                    for (0..half) |j| {
                        // v0 = getBoundCoeff(2*j), v1 = getBoundCoeff(2*j+1)
                        // Then coeffs[j] = v0 + r*(v1 - v0) = (1-r)*v0 + r*v1
                        const base0 = 4 * (2 * j);
                        const base1 = 4 * (2 * j + 1);

                        // v0 = sum of 4 table lookups at base0..base0+3
                        const a00 = if (s.indices[base0]) |idx| s.F_00[idx] else F.zero();
                        const a10 = if (s.indices[base0 + 1]) |idx| s.F_10[idx] else F.zero();
                        const a01 = if (s.indices[base0 + 2]) |idx| s.F_01[idx] else F.zero();
                        const a11 = if (s.indices[base0 + 3]) |idx| s.F_11[idx] else F.zero();
                        const v0 = a00.add(a10).add(a01).add(a11);

                        // v1 = sum of 4 table lookups at base1..base1+3
                        const b00 = if (s.indices[base1]) |idx| s.F_00[idx] else F.zero();
                        const b10 = if (s.indices[base1 + 1]) |idx| s.F_10[idx] else F.zero();
                        const b01 = if (s.indices[base1 + 2]) |idx| s.F_01[idx] else F.zero();
                        const b11 = if (s.indices[base1 + 3]) |idx| s.F_11[idx] else F.zero();
                        const v1 = b00.add(b10).add(b01).add(b11);

                        coeffs[j] = one_minus_r.mul(v0).add(r.mul(v1));
                    }

                    // Free indices and 4 tables
                    allocator.free(s.indices);
                    allocator.free(s.F_00);
                    allocator.free(s.F_01);
                    allocator.free(s.F_10);
                    allocator.free(s.F_11);

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
                .round2, .round3 => unreachable, // Should never reach final claim in these states
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
                .round2 => |*s| {
                    allocator.free(s.indices);
                    allocator.free(s.F_0);
                    allocator.free(s.F_1);
                },
                .round3 => |*s| {
                    allocator.free(s.indices);
                    allocator.free(s.F_00);
                    allocator.free(s.F_01);
                    allocator.free(s.F_10);
                    allocator.free(s.F_11);
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

    // Bind the round1 poly — now goes to round2, not dense
    const RaPoly = RaPolynomial(BN254Scalar);
    var poly = RaPoly.initRound1(indices, eq_table, scale);
    try poly.bind(r, allocator);

    // Should now be round2
    try std.testing.expect(poly == .round2);

    // Verify getBoundCoeff matches expected values at round2 logical indices
    for (0..4) |j| {
        try std.testing.expect(poly.getBoundCoeff(j).eql(expected_bound[j]));
    }

    poly.deinit(allocator);
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

    // Round 1 bind (round1 → round2, stays at 4 indices, logical len 2)
    const r1 = BN254Scalar.fromU64(3);
    try poly.bind(r1, allocator);
    try std.testing.expect(poly == .round2);
    try std.testing.expectEqual(@as(usize, 2), poly.currentLen());

    // Round 2 bind (round2 → round3, stays at 4 indices, logical len 1)
    const r2 = BN254Scalar.fromU64(11);
    try poly.bind(r2, allocator);

    // For a 4-element poly with 2 rounds: round1→round2→round3
    // But round3 logical len = indices.len/4 = 1, and we need finalClaim
    // Actually for 4 elements: round1(4)→round2(2)→round3(1)
    // round3 has logical len 1, so we can get getBoundCoeff(0)
    try std.testing.expect(poly == .round3);
    try std.testing.expectEqual(@as(usize, 1), poly.currentLen());

    // Get the result directly from round3 getBoundCoeff
    const final_val = poly.getBoundCoeff(0);

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

    try std.testing.expect(final_val.eql(expected));
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

    // Bind 4 rounds: round1→round2→round3→dense on round 3, then 1 dense→dense
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

    // Bind 3 rounds: round1→round2→round3→dense(1)
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
    try std.testing.expect(poly == .round2);

    try poly.bind(BN254Scalar.fromU64(7), allocator);
    try std.testing.expectEqual(@as(usize, 2), poly.currentLen());
    try std.testing.expect(poly == .round3);

    try poly.bind(BN254Scalar.fromU64(11), allocator);
    try std.testing.expectEqual(@as(usize, 1), poly.currentLen());
    try std.testing.expect(poly == .dense);

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

test "RaPolynomial deinit in round2 state" {
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
    try poly.bind(BN254Scalar.fromU64(5), allocator);
    try std.testing.expect(poly == .round2);
    poly.deinit(allocator);
}

test "RaPolynomial deinit in round3 state" {
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
    try poly.bind(BN254Scalar.fromU64(5), allocator);
    try poly.bind(BN254Scalar.fromU64(7), allocator);
    try std.testing.expect(poly == .round3);
    poly.deinit(allocator);
}

test "RaPolynomial 32 elements (5 rounds) full sumcheck" {
    const BN254Scalar = @import("../../field/mod.zig").BN254Scalar;
    const allocator = std.testing.allocator;

    // 32-element poly with 4-entry eq_table
    var eq_table = try allocator.alloc(BN254Scalar, 4);
    eq_table[0] = BN254Scalar.fromU64(1);
    eq_table[1] = BN254Scalar.fromU64(3);
    eq_table[2] = BN254Scalar.fromU64(5);
    eq_table[3] = BN254Scalar.fromU64(7);

    const scale = BN254Scalar.fromU64(11);
    var indices = try allocator.alloc(?u8, 32);
    for (0..32) |j| {
        indices[j] = if (j % 5 == 0) null else @intCast(j % 4);
    }

    // Compute expected dense values
    var expected: [32]BN254Scalar = undefined;
    for (0..32) |j| {
        expected[j] = if (indices[j]) |idx| eq_table[idx].mul(scale) else BN254Scalar.zero();
    }

    const challenges = [5]BN254Scalar{
        BN254Scalar.fromU64(3),
        BN254Scalar.fromU64(17),
        BN254Scalar.fromU64(41),
        BN254Scalar.fromU64(7),
        BN254Scalar.fromU64(23),
    };

    // Compute expected by sequential MLE bind
    var cur = expected;
    var cur_len: usize = 32;
    for (challenges) |r| {
        const half = cur_len / 2;
        for (0..half) |j| {
            cur[j] = cur[2 * j].add(r.mul(cur[2 * j + 1].sub(cur[2 * j])));
        }
        cur_len = half;
    }

    const RaPoly = RaPolynomial(BN254Scalar);
    var poly = RaPoly.initRound1(indices, eq_table, scale);

    // 5 binds: round1→round2→round3→dense(4)→dense(2)→dense(1)
    for (challenges) |r| {
        try poly.bind(r, allocator);
    }

    try std.testing.expectEqual(@as(usize, 1), poly.dense.current_len);
    try std.testing.expect(poly.finalClaim().eql(cur[0]));
    poly.deinit(allocator);
}
