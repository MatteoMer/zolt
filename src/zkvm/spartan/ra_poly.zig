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

pub fn RaPolynomial(comptime F: type) type {
    return union(enum) {
        /// Round 1: compressed u8 indices + small eq table
        round1: Round1,
        /// After first bind: dense field element array
        dense: DenseState,

        const Round1 = struct {
            /// T-sized array of u8 indices into eq_table. null = zero (no valid chunk).
            indices: []?u8,
            /// Small eq table (k_chunk entries, typically 16)
            eq_table: []const F,
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
                    allocator.free(@constCast(s.eq_table));
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
                    allocator.free(@constCast(s.eq_table));
                },
                .dense => |*s| {
                    allocator.free(s.coeffs);
                },
            }
        }
    };
}
