//! Prefix-Suffix Sumcheck Optimization for Stage 3
//!
//! This module implements Jolt's prefix-suffix sumcheck optimization used in Stage 3's
//! ShiftSumcheck and related instances. The optimization splits the eq+1 polynomial
//! evaluation into prefix and suffix components, enabling efficient computation during
//! the first n/2 sumcheck rounds.
//!
//! ## Algorithm Overview
//!
//! The eq+1 polynomial eq+1(r, x) is decomposed as:
//!   eq+1((r_hi, r_lo), (x_hi, x_lo)) = prefix_0(r_lo, x_lo) * suffix_0(r_hi, x_hi)
//!                                    + prefix_1(r_lo, x_lo) * suffix_1(r_hi, x_hi)
//!
//! This allows us to represent the sumcheck polynomial as P(x_lo) * Q(x_lo) where:
//! - P comes from the prefix decomposition
//! - Q accumulates trace values weighted by suffix evaluations
//!
//! ## Phase 1 (first n/2 rounds)
//!
//! Round polynomial: g(X) = Σ_i P[2i] * Q[2i] when X=0, Σ_i P[2i+1] * Q[2i+1] when X=1
//! Binding: P_new[i] = P[2i] + r * (P[2i+1] - P[2i])
//!
//! ## Phase 2 (remaining n/2 rounds)
//!
//! Materialize eq+1_r = prefix_0_eval * suffix_0 + prefix_1_eval * suffix_1
//! Run standard sumcheck on the materialized polynomial.

const std = @import("std");
const Allocator = std.mem.Allocator;
const poly_mod = @import("../../poly/mod.zig");

/// Phase 1 Prover for prefix-suffix sumcheck optimization
///
/// Operates on P/Q buffer pairs during the first n/2 sumcheck rounds.
pub fn Phase1Prover(comptime F: type) type {
    return struct {
        const Self = @This();
        const EqPlusOnePrefixSuffixPoly = poly_mod.EqPlusOnePrefixSuffixPoly(F);

        /// P/Q pair for one eq+1 polynomial (e.g., for r_outer)
        const PQPair = struct {
            /// P buffer - from prefix decomposition
            P: []F,
            /// Q buffer - accumulated trace values weighted by suffix
            Q: []F,
        };

        allocator: Allocator,
        /// Prefix-suffix pairs (one per eq+1 polynomial used)
        pairs: std.ArrayList(PQPair),
        /// Sumcheck challenges collected so far
        challenges: std.ArrayList(F),
        /// Current size of P/Q buffers
        current_size: usize,

        pub fn init(allocator: Allocator) Self {
            return Self{
                .allocator = allocator,
                .pairs = std.ArrayList(PQPair).init(allocator),
                .challenges = std.ArrayList(F).init(allocator),
                .current_size = 0,
            };
        }

        pub fn deinit(self: *Self) void {
            for (self.pairs.items) |pair| {
                self.allocator.free(pair.P);
                self.allocator.free(pair.Q);
            }
            self.pairs.deinit();
            self.challenges.deinit();
        }

        /// Add a P/Q pair for an eq+1 polynomial
        /// P comes from the prefix decomposition (prefix_0 combined with prefix_1)
        /// Q is accumulated from trace values weighted by suffix evaluations
        pub fn addPair(self: *Self, P: []F, Q: []F) !void {
            std.debug.assert(P.len == Q.len);
            if (self.current_size == 0) {
                self.current_size = P.len;
            } else {
                std.debug.assert(P.len == self.current_size);
            }
            try self.pairs.append(.{ .P = P, .Q = Q });
        }

        /// Check if we should transition to Phase 2
        /// Transition occurs when only 2 elements remain (log2 size == 1)
        pub fn shouldTransition(self: *const Self) bool {
            return self.current_size <= 2;
        }

        /// Compute the round polynomial evaluations at 0 and 1
        /// Returns [g(0), g(1)] for degree-2 polynomial construction
        pub fn computeRoundEvals(self: *const Self) [2]F {
            var g0 = F.zero();
            var g1 = F.zero();

            const half_size = self.current_size / 2;
            for (self.pairs.items) |pair| {
                for (0..half_size) |i| {
                    // g(0) = Σ P[2i] * Q[2i]
                    // g(1) = Σ P[2i+1] * Q[2i+1]
                    g0 = g0.add(pair.P[2 * i].mul(pair.Q[2 * i]));
                    g1 = g1.add(pair.P[2 * i + 1].mul(pair.Q[2 * i + 1]));
                }
            }

            return .{ g0, g1 };
        }

        /// Bind all P/Q pairs at the challenge point
        /// P_new[i] = P[2i] + r * (P[2i+1] - P[2i])
        pub fn bind(self: *Self, r: F) !void {
            try self.challenges.append(r);

            const half_size = self.current_size / 2;
            for (self.pairs.items) |*pair| {
                for (0..half_size) |i| {
                    const p_low = pair.P[2 * i];
                    const p_high = pair.P[2 * i + 1];
                    pair.P[i] = p_low.add(r.mul(p_high.sub(p_low)));

                    const q_low = pair.Q[2 * i];
                    const q_high = pair.Q[2 * i + 1];
                    pair.Q[i] = q_low.add(r.mul(q_high.sub(q_low)));
                }
            }

            self.current_size = half_size;
        }

        /// Get collected challenges for Phase 2 transition
        pub fn getChallenges(self: *const Self) []const F {
            return self.challenges.items;
        }
    };
}

/// Initialize Q buffers for ShiftSumcheck
///
/// For ShiftSumcheck, we need:
/// - Q_0_outer[x_lo] = Σ_{x_hi} v(x) * suffix_0_outer[x_hi]
/// - Q_1_outer[x_lo] = Σ_{x_hi} v(x) * suffix_1_outer[x_hi]
/// - Q_0_product[x_lo] = Σ_{x_hi} (1-noop(x)) * suffix_0_product[x_hi]
/// - Q_1_product[x_lo] = Σ_{x_hi} (1-noop(x)) * suffix_1_product[x_hi]
///
/// where v(x) = upc(x) + γ¹*pc(x) + γ²*virt(x) + γ³*first(x)
pub fn initShiftQBuffers(
    comptime F: type,
    allocator: Allocator,
    // Trace data
    unexpanded_pc: []const F,
    pc: []const F,
    is_virtual: []const F,
    is_first_in_sequence: []const F,
    is_noop: []const F,
    // Decomposed eq+1 for r_outer
    suffix_0_outer: []const F,
    suffix_1_outer: []const F,
    // Decomposed eq+1 for r_product
    suffix_0_product: []const F,
    suffix_1_product: []const F,
    // Gamma powers
    gamma_powers: []const F,
    // Output size (prefix size = 2^(n/2))
    prefix_size: usize,
) !struct { Q_0_outer: []F, Q_1_outer: []F, Q_0_product: []F, Q_1_product: []F } {
    const trace_len = unexpanded_pc.len;
    const suffix_size = suffix_0_outer.len;

    std.debug.assert(trace_len == prefix_size * suffix_size);
    std.debug.assert(gamma_powers.len >= 5);

    // Allocate Q buffers
    const Q_0_outer = try allocator.alloc(F, prefix_size);
    errdefer allocator.free(Q_0_outer);
    const Q_1_outer = try allocator.alloc(F, prefix_size);
    errdefer allocator.free(Q_1_outer);
    const Q_0_product = try allocator.alloc(F, prefix_size);
    errdefer allocator.free(Q_0_product);
    const Q_1_product = try allocator.alloc(F, prefix_size);
    errdefer allocator.free(Q_1_product);

    // Initialize to zero
    @memset(Q_0_outer, F.zero());
    @memset(Q_1_outer, F.zero());
    @memset(Q_0_product, F.zero());
    @memset(Q_1_product, F.zero());

    // Accumulate over all trace indices
    // x = x_lo + (x_hi << prefix_bits)
    // x_lo ∈ [0, prefix_size), x_hi ∈ [0, suffix_size)
    for (0..suffix_size) |x_hi| {
        for (0..prefix_size) |x_lo| {
            const x = x_lo + (x_hi * prefix_size);

            // v(x) = upc + γ¹*pc + γ²*virt + γ³*first
            var v = unexpanded_pc[x];
            v = v.add(gamma_powers[1].mul(pc[x]));
            v = v.add(gamma_powers[2].mul(is_virtual[x]));
            v = v.add(gamma_powers[3].mul(is_first_in_sequence[x]));

            // Accumulate for r_outer
            Q_0_outer[x_lo] = Q_0_outer[x_lo].add(v.mul(suffix_0_outer[x_hi]));
            Q_1_outer[x_lo] = Q_1_outer[x_lo].add(v.mul(suffix_1_outer[x_hi]));

            // Accumulate for r_product (weighted by 1-noop)
            const noop_factor = F.one().sub(is_noop[x]);
            Q_0_product[x_lo] = Q_0_product[x_lo].add(noop_factor.mul(suffix_0_product[x_hi]));
            Q_1_product[x_lo] = Q_1_product[x_lo].add(noop_factor.mul(suffix_1_product[x_hi]));
        }
    }

    // Scale Q_product by gamma^4
    for (0..prefix_size) |i| {
        Q_0_product[i] = Q_0_product[i].mul(gamma_powers[4]);
        Q_1_product[i] = Q_1_product[i].mul(gamma_powers[4]);
    }

    return .{
        .Q_0_outer = Q_0_outer,
        .Q_1_outer = Q_1_outer,
        .Q_0_product = Q_0_product,
        .Q_1_product = Q_1_product,
    };
}

/// Combine P buffers from prefix decomposition
///
/// The combined P buffer represents:
/// P[j] = prefix_0[j] (for the main term)
/// We handle prefix_1 separately since it's sparse (only j=0 is non-zero)
pub fn combinePBuffers(
    comptime F: type,
    allocator: Allocator,
    prefix_0: []const F,
    prefix_1: []const F,
) ![]F {
    _ = prefix_1; // prefix_1 is sparse, handle separately
    const P = try allocator.alloc(F, prefix_0.len);
    @memcpy(P, prefix_0);
    return P;
}

test "Phase1Prover basic" {
    const F = @import("../../field/mod.zig").BN254Scalar;
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var prover = Phase1Prover(F).init(allocator);
    defer prover.deinit();

    // Create simple P/Q pair
    const P = try allocator.alloc(F, 4);
    const Q = try allocator.alloc(F, 4);
    P[0] = F.fromU64(1);
    P[1] = F.fromU64(2);
    P[2] = F.fromU64(3);
    P[3] = F.fromU64(4);
    Q[0] = F.fromU64(5);
    Q[1] = F.fromU64(6);
    Q[2] = F.fromU64(7);
    Q[3] = F.fromU64(8);

    try prover.addPair(P, Q);

    // Compute round evaluations
    const evals = prover.computeRoundEvals();
    // g(0) = P[0]*Q[0] + P[2]*Q[2] = 1*5 + 3*7 = 5 + 21 = 26
    // g(1) = P[1]*Q[1] + P[3]*Q[3] = 2*6 + 4*8 = 12 + 32 = 44
    try std.testing.expect(evals[0].eql(F.fromU64(26)));
    try std.testing.expect(evals[1].eql(F.fromU64(44)));

    // Bind at r = 2
    const r = F.fromU64(2);
    try prover.bind(r);

    // Check size reduced
    try std.testing.expectEqual(@as(usize, 2), prover.current_size);

    // P_new[0] = P[0] + 2*(P[1] - P[0]) = 1 + 2*1 = 3
    // P_new[1] = P[2] + 2*(P[3] - P[2]) = 3 + 2*1 = 5
    try std.testing.expect(prover.pairs.items[0].P[0].eql(F.fromU64(3)));
    try std.testing.expect(prover.pairs.items[0].P[1].eql(F.fromU64(5)));
}
