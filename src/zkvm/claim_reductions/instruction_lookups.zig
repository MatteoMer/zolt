//! Instruction Lookups Claim Reduction Sumcheck Prover
//!
//! This implements the InstructionLookupsClaimReduction sumcheck for Stage 2.
//! It proves the aggregation of instruction lookup claims from Spartan outer.
//!
//! The sumcheck proves:
//! Σ_j eq(r_spartan, j) * (LookupOutput(j) + γ*LeftOperand(j) + γ²*RightOperand(j)) = input_claim
//!
//! This is a degree-2 sumcheck with log_T rounds.

const std = @import("std");
const Allocator = std.mem.Allocator;

/// Parameters for instruction lookups claim reduction
pub fn InstructionLookupsParams(comptime F: type) type {
    return struct {
        const Self = @This();

        /// Gamma challenge for batching
        gamma: F,
        /// Gamma squared (γ²)
        gamma_sqr: F,
        /// Challenges from SpartanOuter (r_spartan)
        r_spartan: []const F,
        /// Number of cycle variables (log_T)
        n_cycle_vars: usize,
        /// Allocator
        allocator: Allocator,

        pub fn init(
            allocator: Allocator,
            gamma: F,
            r_spartan: []const F,
            n_cycle_vars: usize,
        ) !Self {
            const r_copy = try allocator.alloc(F, r_spartan.len);
            @memcpy(r_copy, r_spartan);

            return Self{
                .gamma = gamma,
                .gamma_sqr = gamma.mul(gamma),
                .r_spartan = r_copy,
                .n_cycle_vars = n_cycle_vars,
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.r_spartan);
        }

        pub fn numRounds(self: *const Self) usize {
            return self.n_cycle_vars;
        }
    };
}

/// Instruction Lookups Claim Reduction Prover
pub fn InstructionLookupsProver(comptime F: type) type {
    return struct {
        const Self = @This();

        /// Parameters
        params: InstructionLookupsParams(F),
        /// Current claim
        current_claim: F,
        /// Current round
        round: usize,
        /// Eq polynomial evaluations (eq(r_spartan, j) for each j)
        eq_evals: []F,
        /// Lookup output values per cycle
        lookup_outputs: []F,
        /// Left operand values per cycle
        left_operands: []F,
        /// Right operand values per cycle
        right_operands: []F,
        /// Bound challenges
        challenges: std.ArrayListUnmanaged(F),
        /// Allocator
        allocator: Allocator,
        /// Original allocation size (for proper deallocation after slicing)
        original_size: usize,

        pub fn init(
            allocator: Allocator,
            params: InstructionLookupsParams(F),
            initial_claim: F,
            lookup_outputs: []const F,
            left_operands: []const F,
            right_operands: []const F,
        ) !Self {
            const T = @as(usize, 1) << @intCast(params.n_cycle_vars);

            // Copy witness data (padded if necessary)
            const lo = try allocator.alloc(F, T);
            const left = try allocator.alloc(F, T);
            const right = try allocator.alloc(F, T);

            @memset(lo, F.zero());
            @memset(left, F.zero());
            @memset(right, F.zero());

            const copy_len = @min(lookup_outputs.len, T);
            if (copy_len > 0) {
                @memcpy(lo[0..copy_len], lookup_outputs[0..copy_len]);
                @memcpy(left[0..copy_len], left_operands[0..copy_len]);
                @memcpy(right[0..copy_len], right_operands[0..copy_len]);
            }

            // Compute eq(r_spartan, j) for each cycle j
            const eq_evals = try allocator.alloc(F, T);
            for (0..T) |j| {
                eq_evals[j] = computeEq(F, params.r_spartan, j);
            }

            const challenges_list = std.ArrayListUnmanaged(F){};

            return Self{
                .params = params,
                .current_claim = initial_claim,
                .round = 0,
                .eq_evals = eq_evals,
                .lookup_outputs = lo,
                .left_operands = left,
                .right_operands = right,
                .challenges = challenges_list,
                .allocator = allocator,
                .original_size = T,
            };
        }

        pub fn deinit(self: *Self) void {
            // Restore original slice lengths for proper deallocation
            // (bindChallenge may have shrunk them)
            self.allocator.free(self.eq_evals.ptr[0..self.original_size]);
            self.allocator.free(self.lookup_outputs.ptr[0..self.original_size]);
            self.allocator.free(self.left_operands.ptr[0..self.original_size]);
            self.allocator.free(self.right_operands.ptr[0..self.original_size]);
            self.challenges.deinit(self.allocator);
            self.params.deinit();
        }

        /// Compute round polynomial [s(0), s(1), s(2), s(3)] for batched cubic sumcheck
        /// Note: This is actually degree-2, but we pad to degree-3 for batching
        /// Uses LowToHigh binding order: pairs (2j, 2j+1) to bind LSB first
        pub fn computeRoundPolynomialCubic(self: *Self) [4]F {
            const gamma = self.params.gamma;
            const gamma_sqr = self.params.gamma_sqr;

            var s0: F = F.zero();
            var s2: F = F.zero();

            const current_len = self.eq_evals.len;
            const half = current_len / 2;

            // Sum over pairs (2j, 2j+1) - LowToHigh binding order
            for (0..half) |idx| {
                // LowToHigh: use consecutive pairs (2*idx, 2*idx+1)
                const lo_idx = 2 * idx;
                const hi_idx = 2 * idx + 1;

                // Get eq evaluations
                const eq_lo = self.eq_evals[lo_idx];
                const eq_hi = self.eq_evals[hi_idx];
                // eq(x) = (1-x)*eq_lo + x*eq_hi
                // eq(0) = eq_lo
                // eq(1) = eq_hi
                // eq(2) = 2*eq_hi - eq_lo

                // Get combined polynomial values
                // combined(j) = lookup_out(j) + γ*left(j) + γ²*right(j)
                const combined_lo = self.lookup_outputs[lo_idx]
                    .add(gamma.mul(self.left_operands[lo_idx]))
                    .add(gamma_sqr.mul(self.right_operands[lo_idx]));
                const combined_hi = self.lookup_outputs[hi_idx]
                    .add(gamma.mul(self.left_operands[hi_idx]))
                    .add(gamma_sqr.mul(self.right_operands[hi_idx]));
                // combined(x) = (1-x)*combined_lo + x*combined_hi (linear)
                // combined(0) = combined_lo
                // combined(1) = combined_hi
                // combined(2) = 2*combined_hi - combined_lo

                // Product evaluations (degree 2)
                // s(0) = eq(0) * combined(0)
                // s(1) = eq(1) * combined(1)
                // s(2) = eq(2) * combined(2)
                const prod_0 = eq_lo.mul(combined_lo);
                const eq_2 = eq_hi.add(eq_hi).sub(eq_lo);
                const combined_2 = combined_hi.add(combined_hi).sub(combined_lo);
                const prod_2 = eq_2.mul(combined_2);

                s0 = s0.add(prod_0);
                s2 = s2.add(prod_2);
            }

            // Compute s(1) from constraint: s(0) + s(1) = current_claim
            const s1 = self.current_claim.sub(s0);

            // For degree-2 polynomial extrapolated to degree-3:
            // s(3) = s(0) - 3*s(1) + 3*s(2)
            const s3 = s0.sub(s1.mul(F.fromU64(3))).add(s2.mul(F.fromU64(3)));

            return [4]F{ s0, s1, s2, s3 };
        }

        /// Bind a challenge after round polynomial computation
        /// Uses LowToHigh binding order: fold pairs (2i, 2i+1) into position i
        pub fn bindChallenge(self: *Self, challenge: F) !void {
            try self.challenges.append(self.allocator, challenge);

            // Fold all polynomials using the challenge - LowToHigh order
            const current_len = self.eq_evals.len;
            const half = current_len / 2;

            for (0..half) |i| {
                // LowToHigh: fold pairs (2i, 2i+1) into position i
                const lo_idx = 2 * i;
                const hi_idx = 2 * i + 1;

                // Linear interpolation: new[i] = (1-r)*lo + r*hi = lo + r*(hi - lo)
                const eq_lo = self.eq_evals[lo_idx];
                const eq_hi = self.eq_evals[hi_idx];
                self.eq_evals[i] = eq_lo.add(challenge.mul(eq_hi.sub(eq_lo)));

                const lo_lo = self.lookup_outputs[lo_idx];
                const lo_hi = self.lookup_outputs[hi_idx];
                self.lookup_outputs[i] = lo_lo.add(challenge.mul(lo_hi.sub(lo_lo)));

                const left_lo = self.left_operands[lo_idx];
                const left_hi = self.left_operands[hi_idx];
                self.left_operands[i] = left_lo.add(challenge.mul(left_hi.sub(left_lo)));

                const right_lo = self.right_operands[lo_idx];
                const right_hi = self.right_operands[hi_idx];
                self.right_operands[i] = right_lo.add(challenge.mul(right_hi.sub(right_lo)));
            }

            // Reduce the effective length of all arrays to half
            // Since these are slices, we update the len field directly
            self.eq_evals = self.eq_evals[0..half];
            self.lookup_outputs = self.lookup_outputs[0..half];
            self.left_operands = self.left_operands[0..half];
            self.right_operands = self.right_operands[0..half];

            self.round += 1;
        }

        /// Update claim after evaluating polynomial at challenge
        pub fn updateClaim(self: *Self, evals: [4]F, challenge: F) void {
            // Lagrange interpolation at challenge from evals at 0, 1, 2, 3
            const c = challenge;
            const c_minus_1 = c.sub(F.one());
            const c_minus_2 = c.sub(F.fromU64(2));
            const c_minus_3 = c.sub(F.fromU64(3));

            const neg6 = F.zero().sub(F.fromU64(6));
            const L0 = c_minus_1.mul(c_minus_2).mul(c_minus_3).mul(neg6.inverse().?);

            const L1 = c.mul(c_minus_2).mul(c_minus_3).mul(F.fromU64(2).inverse().?);

            const neg2 = F.zero().sub(F.fromU64(2));
            const L2 = c.mul(c_minus_1).mul(c_minus_3).mul(neg2.inverse().?);

            const L3 = c.mul(c_minus_1).mul(c_minus_2).mul(F.fromU64(6).inverse().?);

            self.current_claim = evals[0].mul(L0)
                .add(evals[1].mul(L1))
                .add(evals[2].mul(L2))
                .add(evals[3].mul(L3));
        }

        /// Check if all rounds complete
        pub fn isComplete(self: *const Self) bool {
            return self.round >= self.params.numRounds();
        }
    };
}

/// Compute eq(r, x) for a binary index x using BIG ENDIAN indexing
/// r[0] corresponds to MSB of x, r[n-1] corresponds to LSB
fn computeEq(comptime F: type, r: []const F, x: usize) F {
    var result = F.one();
    const n = r.len;
    for (r, 0..) |ri, i| {
        // BIG ENDIAN: r[0] is MSB, r[n-1] is LSB
        const xi: u1 = @truncate(x >> @intCast(n - 1 - i));
        if (xi == 1) {
            result = result.mul(ri);
        } else {
            result = result.mul(F.one().sub(ri));
        }
    }
    return result;
}

test "instruction lookups prover initialization" {
    const allocator = std.testing.allocator;
    const field = @import("../../field/mod.zig");
    const F = field.BN254Scalar;

    const r_spartan = [_]F{ F.fromU64(1), F.fromU64(2), F.fromU64(3) };
    var params = try InstructionLookupsParams(F).init(
        allocator,
        F.fromU64(12345), // gamma
        &r_spartan,
        3, // n_cycle_vars (8 cycles)
    );
    defer params.deinit();

    const lookup_outputs = [_]F{ F.fromU64(1), F.fromU64(2), F.fromU64(3), F.fromU64(4) };
    const left_operands = [_]F{ F.fromU64(10), F.fromU64(20), F.fromU64(30), F.fromU64(40) };
    const right_operands = [_]F{ F.fromU64(100), F.fromU64(200), F.fromU64(300), F.fromU64(400) };

    var prover = try InstructionLookupsProver(F).init(
        allocator,
        params,
        F.fromU64(1000), // initial_claim
        &lookup_outputs,
        &left_operands,
        &right_operands,
    );
    defer prover.deinit();

    try std.testing.expect(!prover.isComplete());
}
