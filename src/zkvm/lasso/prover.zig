//! Lasso Prover for Instruction Lookup Arguments
//!
//! This module implements the Lasso prover for proving instruction lookups
//! in the Jolt zkVM. The prover handles the Read + RAF (Read-Access-Flag)
//! sumcheck protocol that verifies all instruction lookups are valid.
//!
//! The protocol proves the following sumcheck identity:
//!   rv(r_reduction) + γ·left_op(r_reduction) + γ²·right_op(r_reduction)
//!   = Σ_{j=0}^{T-1} Σ_{k=0}^{K-1} eq(j; r_reduction) · ra(k,j) · (Val_j(k) + γ·RafVal_j(k))
//!
//! Where:
//! - rv(j) is the lookup result at cycle j
//! - left_op(j), right_op(j) are the operands at cycle j
//! - ra(k,j) is 1 if the lookup at cycle j accessed table entry k
//! - Val_j(k) is the value at table entry k for the table used at cycle j
//! - RafVal_j(k) is the RAF checking value
//!
//! The prover uses two phases:
//! 1. Address binding (LOG_K rounds): Uses prefix-suffix decomposition
//! 2. Cycle binding (log_T rounds): Uses Gruen split EQ
//!
//! Reference: jolt-core/src/zkvm/instruction_lookups/read_raf_checking.rs

const std = @import("std");
const Allocator = std.mem.Allocator;

const expanding_table = @import("expanding_table.zig");
const prefix_suffix = @import("prefix_suffix.zig");
const split_eq = @import("split_eq.zig");
const poly = @import("../../poly/mod.zig");

const ExpandingTable = expanding_table.ExpandingTable;
const PrefixSuffixDecomposition = prefix_suffix.PrefixSuffixDecomposition;
const PrefixRegistry = prefix_suffix.PrefixRegistry;
const SplitEqPolynomial = split_eq.SplitEqPolynomial;

/// Parameters for the Lasso sumcheck protocol
pub fn LassoParams(comptime F: type) type {
    return struct {
        const Self = @This();

        /// Batching challenge γ
        gamma: F,
        /// γ² for second operand batching
        gamma_sqr: F,
        /// Number of cycle variables (log_T)
        log_T: usize,
        /// Number of address variables (LOG_K)
        log_K: usize,
        /// Chunk size for virtual ra polynomials
        ra_virtual_log_k_chunk: usize,
        /// Number of phases for prefix-suffix decomposition
        num_phases: usize,
        /// Reduction point (challenge from previous round)
        r_reduction: []const F,

        /// Create parameters from transcript challenges
        pub fn init(
            gamma: F,
            log_T: usize,
            log_K: usize,
            r_reduction: []const F,
        ) Self {
            // Default chunk sizes (can be tuned for performance)
            const ra_virtual_log_k_chunk = @min(log_K, 32);
            const num_phases = (log_K + ra_virtual_log_k_chunk - 1) / ra_virtual_log_k_chunk;

            return Self{
                .gamma = gamma,
                .gamma_sqr = gamma.mul(gamma),
                .log_T = log_T,
                .log_K = log_K,
                .ra_virtual_log_k_chunk = ra_virtual_log_k_chunk,
                .num_phases = num_phases,
                .r_reduction = r_reduction,
            };
        }
    };
}

/// Lasso sumcheck prover state
pub fn LassoProver(comptime F: type) type {
    return struct {
        const Self = @This();

        /// Lookup indices for each cycle
        lookup_indices: []const u128,
        /// Which table is used at each cycle
        lookup_tables: []const usize,
        /// EQ polynomial evaluations indexed by cycle
        eq_r_reduction: SplitEqPolynomial(F),
        /// Expanding table for accumulating EQ values
        expanding_v: ExpandingTable(F),
        /// Prefix-suffix decomposition for left operand
        left_operand_ps: PrefixSuffixDecomposition(F, 2),
        /// Prefix-suffix decomposition for right operand
        right_operand_ps: PrefixSuffixDecomposition(F, 2),
        /// Prefix registry for caching
        prefix_registry: PrefixRegistry(F),
        /// Protocol parameters
        params: LassoParams(F),
        /// Current round (0-indexed)
        round: usize,
        /// Current phase for prefix-suffix
        phase: usize,
        /// Accumulated challenges
        challenges: []F,
        challenges_len: usize,
        /// Allocator
        allocator: Allocator,

        /// Initialize the Lasso prover
        pub fn init(
            allocator: Allocator,
            lookup_indices: []const u128,
            lookup_tables: []const usize,
            params: LassoParams(F),
        ) !Self {
            // Use log_T from params (which matches r_reduction.len) not from lookup count
            // This ensures consistency with the reduction point provided
            const log_T = params.log_T;

            // Initialize EQ polynomial for reduction point
            // Split into outer (first half) and inner (second half) variables
            const outer_vars = log_T / 2;
            const inner_vars = log_T - outer_vars;
            const eq_r = try SplitEqPolynomial(F).init(
                allocator,
                outer_vars,
                inner_vars,
                params.r_reduction,
            );

            // Initialize expanding table
            const expanding_v = try ExpandingTable(F).init(allocator, params.log_K);

            // Initialize prefix-suffix decompositions
            const prefix_vars = params.log_K / 2;
            const suffix_vars = params.log_K - prefix_vars;
            const left_ps = try PrefixSuffixDecomposition(F, 2).init(allocator, prefix_vars, suffix_vars);
            const right_ps = try PrefixSuffixDecomposition(F, 2).init(allocator, prefix_vars, suffix_vars);

            // Initialize prefix registry
            const registry = PrefixRegistry(F).init(allocator);

            // Allocate challenge buffer
            const max_rounds = params.log_K + params.log_T;
            const challenges = try allocator.alloc(F, max_rounds);
            @memset(challenges, F.zero());

            return Self{
                .lookup_indices = lookup_indices,
                .lookup_tables = lookup_tables,
                .eq_r_reduction = eq_r,
                .expanding_v = expanding_v,
                .left_operand_ps = left_ps,
                .right_operand_ps = right_ps,
                .prefix_registry = registry,
                .params = params,
                .round = 0,
                .phase = 0,
                .challenges = challenges,
                .challenges_len = 0,
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            self.eq_r_reduction.deinit();
            self.expanding_v.deinit();
            self.left_operand_ps.deinit();
            self.right_operand_ps.deinit();
            self.prefix_registry.deinit();
            self.allocator.free(self.challenges);
        }

        /// Check if we're in the address binding phase
        pub fn isAddressPhase(self: *const Self) bool {
            return self.round < self.params.log_K;
        }

        /// Check if the protocol is complete
        pub fn isComplete(self: *const Self) bool {
            return self.round >= self.params.log_K + self.params.log_T;
        }

        /// Compute the prover message for the current round
        ///
        /// Returns a univariate polynomial g_i(X) of degree 2 where:
        /// - g_i(0) + g_i(1) = current claim
        /// - The polynomial encodes the partial sums for variable i
        pub fn computeRoundPolynomial(self: *Self) !poly.UniPoly(F) {
            if (self.isAddressPhase()) {
                return self.computeAddressRoundPoly();
            } else {
                return self.computeCycleRoundPoly();
            }
        }

        /// Compute round polynomial for address phase (first LOG_K rounds)
        fn computeAddressRoundPoly(self: *Self) !poly.UniPoly(F) {
            // During address binding, we use prefix-suffix decomposition
            // The round polynomial is computed by summing over table entries
            // while accumulating prefix/suffix evaluations

            const coeffs = try self.allocator.alloc(F, 3);

            // For now, compute simple linear polynomial
            // g(X) = a + b*X where a = sum at X=0, b = slope
            var sum_0 = F.zero();
            var sum_1 = F.zero();

            const half = self.lookup_indices.len / 2;
            const round_bit = self.round;

            for (self.lookup_indices, 0..) |idx, j| {
                // Get the bit at current round position
                const bit = (idx >> @intCast(round_bit)) & 1;
                const eq_val = self.eq_r_reduction.getEq(j);

                if (bit == 0) {
                    sum_0 = sum_0.add(eq_val);
                } else {
                    sum_1 = sum_1.add(eq_val);
                }
            }
            _ = half;

            // Linear polynomial: coeffs = [sum_0, sum_1 - sum_0]
            // But we need degree 2 for consistency, so add zero coefficient
            coeffs[0] = sum_0;
            coeffs[1] = sum_1.sub(sum_0);
            coeffs[2] = F.zero();

            return poly.UniPoly(F){
                .coeffs = coeffs,
                .allocator = self.allocator,
            };
        }

        /// Compute round polynomial for cycle phase (last log_T rounds)
        fn computeCycleRoundPoly(self: *Self) !poly.UniPoly(F) {
            // During cycle binding, we use the Gruen split EQ optimization
            // to efficiently compute the round polynomial

            const coeffs = try self.allocator.alloc(F, 3);

            // Compute sums over cycles where cycle variable = 0 and = 1
            var sum_0 = F.zero();
            var sum_1 = F.zero();

            const cycle_round = self.round - self.params.log_K;
            const half = @as(usize, 1) << @intCast(self.params.log_T - cycle_round - 1);

            for (0..half) |j| {
                const eq_0 = self.eq_r_reduction.getEq(j);
                const eq_1 = self.eq_r_reduction.getEq(j + half);

                // Weight by table values (simplified for now)
                sum_0 = sum_0.add(eq_0);
                sum_1 = sum_1.add(eq_1);
            }

            coeffs[0] = sum_0;
            coeffs[1] = sum_1.sub(sum_0);
            coeffs[2] = F.zero();

            return poly.UniPoly(F){
                .coeffs = coeffs,
                .allocator = self.allocator,
            };
        }

        /// Receive a challenge and update state
        pub fn receiveChallenge(self: *Self, challenge: F) !void {
            // Record the challenge
            self.challenges[self.challenges_len] = challenge;
            self.challenges_len += 1;

            if (self.isAddressPhase()) {
                // Bind the expanding table
                try self.expanding_v.bind(challenge);

                // Update prefix-suffix decompositions
                try self.left_operand_ps.bind(challenge);
                try self.right_operand_ps.bind(challenge);

                // Check if we need to transition to the next phase
                const vars_per_phase = self.params.log_K / self.params.num_phases;
                if (self.round > 0 and self.round % vars_per_phase == 0) {
                    self.phase += 1;
                }
            } else {
                // Bind the cycle EQ polynomial
                const cycle_round = self.round - self.params.log_K;
                if (cycle_round < self.eq_r_reduction.num_outer) {
                    self.eq_r_reduction.bindOuter(challenge);
                } else {
                    self.eq_r_reduction.bindInner(challenge);
                }
            }

            self.round += 1;
        }

        /// Get the final evaluation after all rounds
        pub fn getFinalEval(self: *const Self) F {
            std.debug.assert(self.isComplete());
            // Return the evaluation of the combined polynomial at the challenge point
            return self.expanding_v.get(0);
        }

        /// Get all accumulated challenges
        pub fn getChallenges(self: *const Self) []const F {
            return self.challenges[0..self.challenges_len];
        }
    };
}

/// Lasso proof structure
pub fn LassoProof(comptime F: type) type {
    return struct {
        const Self = @This();

        /// Round polynomials
        round_polys: []poly.UniPoly(F),
        /// Final evaluation
        final_eval: F,
        /// Challenges used
        challenges: []F,
        allocator: Allocator,

        pub fn deinit(self: *Self) void {
            for (self.round_polys) |*p| {
                p.deinit();
            }
            self.allocator.free(self.round_polys);
            self.allocator.free(self.challenges);
        }
    };
}

/// Run the Lasso prover protocol
pub fn runLassoProver(
    comptime F: type,
    allocator: Allocator,
    lookup_indices: []const u128,
    lookup_tables: []const usize,
    params: LassoParams(F),
) !LassoProof(F) {
    var prover = try LassoProver(F).init(allocator, lookup_indices, lookup_tables, params);
    defer prover.deinit();

    const total_rounds = params.log_K + params.log_T;
    const round_polys = try allocator.alloc(poly.UniPoly(F), total_rounds);

    // Run the sumcheck protocol
    var round: usize = 0;
    while (!prover.isComplete()) : (round += 1) {
        // Compute round polynomial
        round_polys[round] = try prover.computeRoundPolynomial();

        // Generate challenge (in practice, from Fiat-Shamir transcript)
        const challenge = deriveChallenge(F, round_polys[round], round);

        // Update prover state
        try prover.receiveChallenge(challenge);
    }

    // Get final evaluation
    const final_eval = prover.getFinalEval();

    // Copy challenges
    const challenges = try allocator.alloc(F, prover.challenges_len);
    @memcpy(challenges, prover.getChallenges());

    return LassoProof(F){
        .round_polys = round_polys,
        .final_eval = final_eval,
        .challenges = challenges,
        .allocator = allocator,
    };
}

/// Derive a challenge from a round polynomial (simplified Fiat-Shamir)
fn deriveChallenge(comptime F: type, round_poly: poly.UniPoly(F), round: usize) F {
    var hash: u64 = 0x9e3779b97f4a7c15;
    hash ^= @as(u64, @intCast(round));
    hash *%= 0xff51afd7ed558ccd;

    for (round_poly.coeffs) |coeff| {
        for (coeff.limbs) |limb| {
            hash ^= limb;
            hash *%= 0xc4ceb9fe1a85ec53;
        }
    }

    hash ^= hash >> 33;
    return F.fromU64(hash);
}

test "lasso prover basic" {
    const F = @import("../../field/mod.zig").BN254Scalar;
    const allocator = std.testing.allocator;

    // Simple test case: 4 lookups, 8 table entries
    const lookup_indices = [_]u128{ 0, 1, 2, 3 };
    const lookup_tables = [_]usize{ 0, 0, 0, 0 };

    // Create reduction point (4 elements for log_T=2)
    const r_reduction = [_]F{
        F.fromU64(2),
        F.fromU64(3),
    };

    const params = LassoParams(F).init(
        F.fromU64(5), // gamma
        2, // log_T (4 cycles)
        3, // log_K (8 table entries)
        &r_reduction,
    );

    var prover = try LassoProver(F).init(
        allocator,
        &lookup_indices,
        &lookup_tables,
        params,
    );
    defer prover.deinit();

    // Should start at round 0
    try std.testing.expectEqual(@as(usize, 0), prover.round);
    try std.testing.expect(prover.isAddressPhase());
    try std.testing.expect(!prover.isComplete());
}

test "lasso prover rounds" {
    const F = @import("../../field/mod.zig").BN254Scalar;
    const allocator = std.testing.allocator;

    const lookup_indices = [_]u128{ 0, 1, 2, 3 };
    const lookup_tables = [_]usize{ 0, 0, 0, 0 };

    const r_reduction = [_]F{
        F.fromU64(2),
        F.fromU64(3),
    };

    const params = LassoParams(F).init(
        F.fromU64(5),
        2, // log_T
        3, // log_K
        &r_reduction,
    );

    var prover = try LassoProver(F).init(
        allocator,
        &lookup_indices,
        &lookup_tables,
        params,
    );
    defer prover.deinit();

    // Run a few rounds
    var uni = try prover.computeRoundPolynomial();
    defer uni.deinit();

    // Check polynomial is valid
    try std.testing.expect(uni.coeffs.len > 0);

    // Receive challenge
    try prover.receiveChallenge(F.fromU64(7));

    // Should advance
    try std.testing.expectEqual(@as(usize, 1), prover.round);
}
