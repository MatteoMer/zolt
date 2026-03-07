//! Instruction Lookups Claim Reduction Sumcheck Prover
//!
//! This implements the InstructionLookupsClaimReduction sumcheck for Stage 2.
//! It proves the aggregation of instruction lookup claims from Spartan outer.
//!
//! The sumcheck proves:
//! Σ_j eq(r_spartan, j) * (LookupOutput(j) + γ*LeftOp(j) + γ²*RightOp(j) + γ³*LeftInstr(j) + γ⁴*RightInstr(j)) = input_claim
//!
//! This is a degree-2 sumcheck with log_T rounds.

const std = @import("std");

// Debug output control - set to true to enable verbose debug prints
const debug_verbose = true;
fn dbg(comptime fmt: []const u8, args: anytype) void {
    if (debug_verbose) std.debug.print(fmt, args);
}

const Allocator = std.mem.Allocator;

/// Parameters for instruction lookups claim reduction
pub fn InstructionLookupsParams(comptime F: type) type {
    return struct {
        const Self = @This();

        /// Gamma challenge for batching
        gamma: F,
        /// Gamma squared (γ²)
        gamma_sqr: F,
        /// Gamma cubed (γ³)
        gamma_cub: F,
        /// Gamma to the fourth (γ⁴)
        gamma_quart: F,
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

            const gamma_sqr = gamma.mul(gamma);
            return Self{
                .gamma = gamma,
                .gamma_sqr = gamma_sqr,
                .gamma_cub = gamma_sqr.mul(gamma),
                .gamma_quart = gamma_sqr.mul(gamma_sqr),
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
        /// Left instruction input values per cycle
        left_instr_inputs: []F,
        /// Right instruction input values per cycle
        right_instr_inputs: []F,
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
            left_instr_inputs: []const F,
            right_instr_inputs: []const F,
        ) !Self {
            const T = @as(usize, 1) << @intCast(params.n_cycle_vars);

            // Copy witness data (padded if necessary)
            const lo = try allocator.alloc(F, T);
            const left = try allocator.alloc(F, T);
            const right = try allocator.alloc(F, T);
            const left_ii = try allocator.alloc(F, T);
            const right_ii = try allocator.alloc(F, T);

            @memset(lo, F.zero());
            @memset(left, F.zero());
            @memset(right, F.zero());
            @memset(left_ii, F.zero());
            @memset(right_ii, F.zero());

            const copy_len = @min(lookup_outputs.len, T);
            if (copy_len > 0) {
                @memcpy(lo[0..copy_len], lookup_outputs[0..copy_len]);
                @memcpy(left[0..copy_len], left_operands[0..copy_len]);
                @memcpy(right[0..copy_len], right_operands[0..copy_len]);
                @memcpy(left_ii[0..copy_len], left_instr_inputs[0..@min(left_instr_inputs.len, T)]);
                @memcpy(right_ii[0..copy_len], right_instr_inputs[0..@min(right_instr_inputs.len, T)]);
            }

            // Compute eq(r_spartan, j) for each cycle j
            const eq_evals = try allocator.alloc(F, T);
            for (0..T) |j| {
                eq_evals[j] = computeEq(F, params.r_spartan, j);
            }

            // Debug: print initial sums before any binding (this is MLE at r_spartan)
            var init_left_sum = F.zero();
            var init_right_sum = F.zero();
            var init_output_sum = F.zero();
            for (0..T) |j| {
                init_left_sum = init_left_sum.add(eq_evals[j].mul(left[j]));
                init_right_sum = init_right_sum.add(eq_evals[j].mul(right[j]));
                init_output_sum = init_output_sum.add(eq_evals[j].mul(lo[j]));
            }
            dbg("[INSTR_LOOKUPS INIT] MLE at r_spartan (before sumcheck):\n", .{});
            dbg("  init_output_sum = {x}\n", .{init_output_sum.toBytesBE()[16..32].*});
            dbg("  init_left_sum = {x}\n", .{init_left_sum.toBytesBE()[16..32].*});
            dbg("  init_right_sum = {x}\n", .{init_right_sum.toBytesBE()[16..32].*});
            dbg("  r_spartan[0] = {x}\n", .{params.r_spartan[0].toBytesBE()[16..32].*});
            dbg("  Stage2 eq_evals[0..3]: {x}, {x}, {x}\n", .{
                eq_evals[0].toBytesBE()[16..32].*,
                eq_evals[1].toBytesBE()[16..32].*,
                eq_evals[2].toBytesBE()[16..32].*,
            });

            const challenges_list = std.ArrayListUnmanaged(F){};

            return Self{
                .params = params,
                .current_claim = initial_claim,
                .round = 0,
                .eq_evals = eq_evals,
                .lookup_outputs = lo,
                .left_operands = left,
                .right_operands = right,
                .left_instr_inputs = left_ii,
                .right_instr_inputs = right_ii,
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
            self.allocator.free(self.left_instr_inputs.ptr[0..self.original_size]);
            self.allocator.free(self.right_instr_inputs.ptr[0..self.original_size]);
            self.challenges.deinit(self.allocator);
            self.params.deinit();
        }

        /// Compute round polynomial [s(0), s(1), s(2), s(3)] for batched cubic sumcheck
        /// Note: This is actually degree-2, but we pad to degree-3 for batching
        /// Uses LowToHigh binding order: pairs (2j, 2j+1) to bind LSB first
        pub fn computeRoundPolynomialCubic(self: *Self) [4]F {
            const gamma = self.params.gamma;
            const gamma_sqr = self.params.gamma_sqr;
            const gamma_cub = self.params.gamma_cub;
            const gamma_quart = self.params.gamma_quart;

            var s0: F = F.zero();
            var s2: F = F.zero();

            const current_len = self.eq_evals.len;
            const half = current_len / 2;

            // Sum over pairs (2j, 2j+1) - LowToHigh binding order
            for (0..half) |idx| {
                const lo_idx = 2 * idx;
                const hi_idx = 2 * idx + 1;

                const eq_lo = self.eq_evals[lo_idx];
                const eq_hi = self.eq_evals[hi_idx];

                // combined(j) = lookup_out(j) + γ*left_op(j) + γ²*right_op(j) + γ³*left_instr(j) + γ⁴*right_instr(j)
                const combined_lo = self.lookup_outputs[lo_idx]
                    .add(gamma.mul(self.left_operands[lo_idx]))
                    .add(gamma_sqr.mul(self.right_operands[lo_idx]))
                    .add(gamma_cub.mul(self.left_instr_inputs[lo_idx]))
                    .add(gamma_quart.mul(self.right_instr_inputs[lo_idx]));
                const combined_hi = self.lookup_outputs[hi_idx]
                    .add(gamma.mul(self.left_operands[hi_idx]))
                    .add(gamma_sqr.mul(self.right_operands[hi_idx]))
                    .add(gamma_cub.mul(self.left_instr_inputs[hi_idx]))
                    .add(gamma_quart.mul(self.right_instr_inputs[hi_idx]));

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

                const lii_lo = self.left_instr_inputs[lo_idx];
                const lii_hi = self.left_instr_inputs[hi_idx];
                self.left_instr_inputs[i] = lii_lo.add(challenge.mul(lii_hi.sub(lii_lo)));

                const rii_lo = self.right_instr_inputs[lo_idx];
                const rii_hi = self.right_instr_inputs[hi_idx];
                self.right_instr_inputs[i] = rii_lo.add(challenge.mul(rii_hi.sub(rii_lo)));
            }

            // Reduce the effective length of all arrays to half
            self.eq_evals = self.eq_evals[0..half];
            self.lookup_outputs = self.lookup_outputs[0..half];
            self.left_operands = self.left_operands[0..half];
            self.right_operands = self.right_operands[0..half];
            self.left_instr_inputs = self.left_instr_inputs[0..half];
            self.right_instr_inputs = self.right_instr_inputs[0..half];

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

        /// Get the individual opening claims after all rounds are complete
        pub fn getOpeningClaims(self: *const Self) struct { lookup_output: F, left_operand: F, right_operand: F, left_instr_input: F, right_instr_input: F } {
            // After all rounds, the arrays have been folded down to length 1
            const lookup_output = if (self.lookup_outputs.len > 0) self.lookup_outputs[0] else F.zero();
            const left_operand = if (self.left_operands.len > 0) self.left_operands[0] else F.zero();
            const right_operand = if (self.right_operands.len > 0) self.right_operands[0] else F.zero();
            const left_instr_input = if (self.left_instr_inputs.len > 0) self.left_instr_inputs[0] else F.zero();
            const right_instr_input = if (self.right_instr_inputs.len > 0) self.right_instr_inputs[0] else F.zero();

            dbg("[INSTR_LOOKUPS FINAL] After {} rounds of binding:\n", .{self.challenges.items.len});
            dbg("  lookup_output = {x}\n", .{lookup_output.toBytesBE()[16..32].*});
            dbg("  left_operand = {x}\n", .{left_operand.toBytesBE()[16..32].*});
            dbg("  right_operand = {x}\n", .{right_operand.toBytesBE()[16..32].*});
            dbg("  left_instr_input = {x}\n", .{left_instr_input.toBytesBE()[16..32].*});
            dbg("  right_instr_input = {x}\n", .{right_instr_input.toBytesBE()[16..32].*});

            return .{
                .lookup_output = lookup_output,
                .left_operand = left_operand,
                .right_operand = right_operand,
                .left_instr_input = left_instr_input,
                .right_instr_input = right_instr_input,
            };
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

    const left_instr = [_]F{ F.fromU64(5), F.fromU64(15), F.fromU64(25), F.fromU64(35) };
    const right_instr = [_]F{ F.fromU64(50), F.fromU64(150), F.fromU64(250), F.fromU64(350) };

    var prover = try InstructionLookupsProver(F).init(
        allocator,
        params,
        F.fromU64(1000), // initial_claim
        &lookup_outputs,
        &left_operands,
        &right_operands,
        &left_instr,
        &right_instr,
    );
    defer prover.deinit();

    try std.testing.expect(!prover.isComplete());
}
