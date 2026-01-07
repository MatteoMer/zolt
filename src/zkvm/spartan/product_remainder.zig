//! ProductVirtualRemainderProver - Stage 2 Product Virtualization Sumcheck
//!
//! This module implements the sumcheck prover for the "remaining" rounds of
//! product virtualization after the univariate skip first round.
//!
//! ## Protocol Overview
//!
//! Stage 2 proves the 5 product constraints:
//! 1. Product = LeftInstructionInput * RightInstructionInput
//! 2. WriteLookupOutputToRD = IsRdNotZero * OpFlags(WriteLookupOutputToRD)
//! 3. WritePCtoRD = IsRdNotZero * OpFlags(Jump)
//! 4. ShouldBranch = LookupOutput * InstructionFlags(Branch)
//! 5. ShouldJump = OpFlags(Jump) * (1 - NextIsNoop)
//!
//! ## Fused Sumcheck
//!
//! These 5 constraints are fused into 2 polynomials (left/right) using Lagrange weights
//! from the first-round challenge r0:
//!
//!   fused_left(x) = Σᵢ wᵢ * leftᵢ(x)
//!   fused_right(x) = Σᵢ wᵢ * rightᵢ(x)
//!
//! Where wᵢ = Lᵢ(r0) are Lagrange basis polynomials evaluated at r0 over the 5-point domain.
//!
//! ## Round Polynomial
//!
//! The sumcheck proves:
//!   Σ_x L(τ_high, r0) * Eq(τ_low, x) * fused_left(x) * fused_right(x) = claim
//!
//! Each round, the prover sends s(X) = [s(0), s(2), s(3)] (cubic polynomial, linear omitted).
//!
//! Reference: jolt-core/src/zkvm/spartan/product.rs

const std = @import("std");
const Allocator = std.mem.Allocator;

const constraints = @import("../r1cs/constraints.zig");
const univariate_skip = @import("../r1cs/univariate_skip.zig");
const poly_mod = @import("../../poly/mod.zig");
const GruenSplitEqPolynomial = poly_mod.GruenSplitEqPolynomial;
const DensePolynomial = poly_mod.DensePolynomial;
const utils = @import("../../utils/mod.zig");

/// Number of product constraints
pub const NUM_PRODUCT_CONSTRAINTS: usize = 5;

/// Domain size for product virtualization univariate skip
pub const DOMAIN_SIZE: usize = univariate_skip.PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DOMAIN_SIZE;

/// Degree of the remainder sumcheck (product of 2 multilinear = degree 2, plus eq = degree 3)
pub const REMAINDER_DEGREE: usize = 3;

/// The 8 unique factor polynomial indices that appear in the 5 product constraints
/// Matches Jolt's PRODUCT_UNIQUE_FACTOR_VIRTUALS
pub const ProductFactorIndex = enum(usize) {
    LeftInstructionInput = 0,
    RightInstructionInput = 1,
    IsRdNotZero = 2, // InstructionFlags::IsRdNotZero
    WriteLookupOutputToRDFlag = 3, // OpFlags::WriteLookupOutputToRD
    JumpFlag = 4, // OpFlags::Jump
    LookupOutput = 5,
    BranchFlag = 6, // InstructionFlags::Branch
    NextIsNoop = 7,
};

/// Per-cycle inputs for product virtualization
/// Compact representation of the 8 unique factors
pub fn ProductCycleInputs(comptime F: type) type {
    return struct {
        const Self = @This();

        /// The 8 factor values for this cycle
        factors: [8]F,

        /// Create from R1CS cycle inputs
        pub fn fromR1CSInputs(r1cs_inputs: *const constraints.R1CSCycleInputs(F)) Self {
            return Self{
                .factors = [8]F{
                    // 0: LeftInstructionInput
                    r1cs_inputs.values[constraints.R1CSInputIndex.LeftInstructionInput.toIndex()],
                    // 1: RightInstructionInput
                    r1cs_inputs.values[constraints.R1CSInputIndex.RightInstructionInput.toIndex()],
                    // 2: IsRdNotZero (extracted from FlagIsRdNotZero or computed)
                    // Note: We need to get this from the instruction flags, not a direct R1CS input
                    // In Jolt, this is InstructionFlags::IsRdNotZero which is a derived value
                    // For now, we compute it: rd != 0 means is_rd_not_zero = 1
                    r1cs_inputs.values[constraints.R1CSInputIndex.RdWriteValue.toIndex()].eqlZero() == false,
                    // 3: WriteLookupOutputToRDFlag
                    r1cs_inputs.values[constraints.R1CSInputIndex.FlagWriteLookupOutputToRD.toIndex()],
                    // 4: JumpFlag
                    r1cs_inputs.values[constraints.R1CSInputIndex.FlagJump.toIndex()],
                    // 5: LookupOutput
                    r1cs_inputs.values[constraints.R1CSInputIndex.LookupOutput.toIndex()],
                    // 6: BranchFlag - needs to be derived from opcode or instruction flags
                    // For branches, this is 1 when the instruction is a branch type
                    // We approximate using ShouldBranch/LookupOutput when LookupOutput != 0
                    r1cs_inputs.values[constraints.R1CSInputIndex.ShouldBranch.toIndex()],
                    // 7: NextIsNoop
                    F.zero(), // Will be set properly from trace context
                },
            };
        }

        /// Get factor value by index
        pub fn getFactor(self: *const Self, idx: ProductFactorIndex) F {
            return self.factors[@intFromEnum(idx)];
        }

        /// Compute fused left polynomial value at this cycle using Lagrange weights
        ///
        /// left = w[0]*LeftInstructionInput + w[1]*IsRdNotZero + w[2]*IsRdNotZero
        ///      + w[3]*LookupOutput + w[4]*JumpFlag
        pub fn fusedLeft(self: *const Self, weights: *const [5]F) F {
            return weights[0].mul(self.factors[0]) // LeftInstructionInput
                .add(weights[1].mul(self.factors[2])) // IsRdNotZero
                .add(weights[2].mul(self.factors[2])) // IsRdNotZero (again)
                .add(weights[3].mul(self.factors[5])) // LookupOutput
                .add(weights[4].mul(self.factors[4])); // JumpFlag
        }

        /// Compute fused right polynomial value at this cycle using Lagrange weights
        ///
        /// right = w[0]*RightInstructionInput + w[1]*WriteLookupOutputToRDFlag + w[2]*JumpFlag
        ///       + w[3]*BranchFlag + w[4]*(1 - NextIsNoop)
        pub fn fusedRight(self: *const Self, weights: *const [5]F) F {
            const one_minus_noop = F.one().sub(self.factors[7]);
            return weights[0].mul(self.factors[1]) // RightInstructionInput
                .add(weights[1].mul(self.factors[3])) // WriteLookupOutputToRDFlag
                .add(weights[2].mul(self.factors[4])) // JumpFlag
                .add(weights[3].mul(self.factors[6])) // BranchFlag
                .add(weights[4].mul(one_minus_noop)); // (1 - NextIsNoop)
        }
    };
}

/// ProductVirtualRemainderProver - sumcheck prover for product virtualization
pub fn ProductVirtualRemainderProver(comptime F: type) type {
    return struct {
        const Self = @This();

        /// Lagrange basis evaluations at r0 over the 5-point domain
        lagrange_weights: [5]F,
        /// Split eq polynomial for efficient factored evaluation
        split_eq: GruenSplitEqPolynomial(F),
        /// Bound left polynomial (interleaved lo/hi)
        left_poly: DensePolynomial(F),
        /// Bound right polynomial (interleaved lo/hi)
        right_poly: DensePolynomial(F),
        /// Number of cycle variables
        num_cycle_vars: usize,
        /// Current round
        current_round: usize,
        /// Current claim
        current_claim: F,
        /// Allocator
        allocator: Allocator,

        /// Initialize the prover after univariate skip
        ///
        /// r0: First-round challenge from univariate skip
        /// tau: Full tau vector (length = num_cycle_vars + 1)
        ///      tau[0..num_cycle_vars] = tau_low (cycle vars)
        ///      tau[num_cycle_vars] = tau_high (used in UniSkip)
        /// uni_skip_claim: Evaluation of UniSkip polynomial at r0
        /// cycle_witnesses: Per-cycle R1CS inputs
        pub fn init(
            allocator: Allocator,
            r0: F,
            tau: []const F,
            uni_skip_claim: F,
            cycle_witnesses: []const constraints.R1CSCycleInputs(F),
        ) !Self {
            if (cycle_witnesses.len == 0) {
                return error.EmptyTrace;
            }

            // Pad to power of 2
            const padded_len = nextPowerOfTwo(cycle_witnesses.len);
            const num_cycle_vars = std.math.log2_int(usize, padded_len);

            // Compute Lagrange weights at r0 over the 5-point domain {-2, -1, 0, 1, 2}
            const lagrange_weights = try computeLagrangeWeightsGeneric(F, allocator, r0);

            // Extract tau_low and tau_high
            const tau_high = if (tau.len > 0) tau[tau.len - 1] else F.zero();
            const tau_low = if (tau.len > 0) tau[0 .. tau.len - 1] else tau;

            // Compute Lagrange kernel L(tau_high, r0) for scaling
            const lagrange_kernel = try univariate_skip.LagrangePolynomial(F).lagrangeKernel(
                DOMAIN_SIZE,
                r0,
                tau_high,
                allocator,
            );

            // Initialize split eq with scaling
            const split_eq = try GruenSplitEqPolynomial(F).initWithScaling(
                allocator,
                tau_low,
                lagrange_kernel,
            );

            // Materialize fused left/right polynomials
            var left_evals = try allocator.alloc(F, padded_len);
            errdefer allocator.free(left_evals);
            var right_evals = try allocator.alloc(F, padded_len);
            errdefer allocator.free(right_evals);

            // Compute fused left/right for each cycle
            for (0..padded_len) |idx| {
                if (idx < cycle_witnesses.len) {
                    const witness = &cycle_witnesses[idx];

                    // Extract product factors
                    const product_inputs = extractProductInputs(F, witness, cycle_witnesses, idx);

                    // Compute fused values
                    left_evals[idx] = product_inputs.fusedLeft(&lagrange_weights);
                    right_evals[idx] = product_inputs.fusedRight(&lagrange_weights);
                } else {
                    // Pad with zeros
                    left_evals[idx] = F.zero();
                    right_evals[idx] = F.zero();
                }
            }

            const left_poly = try DensePolynomial(F).init(allocator, left_evals);
            allocator.free(left_evals);

            const right_poly = try DensePolynomial(F).init(allocator, right_evals);
            allocator.free(right_evals);

            return Self{
                .lagrange_weights = lagrange_weights,
                .split_eq = split_eq,
                .left_poly = left_poly,
                .right_poly = right_poly,
                .num_cycle_vars = num_cycle_vars,
                .current_round = 0,
                .current_claim = uni_skip_claim,
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            self.split_eq.deinit();
            self.left_poly.deinit();
            self.right_poly.deinit();
        }

        /// Number of rounds (= num_cycle_vars)
        pub fn numRounds(self: *const Self) usize {
            return self.num_cycle_vars;
        }

        /// Degree bound (cubic = 3)
        pub fn degreeBound() usize {
            return REMAINDER_DEGREE;
        }

        /// Compute the round polynomial for the current round using Gruen's method
        ///
        /// This matches Jolt's ProductVirtualRemainderProver::compute_message exactly:
        /// 1. Compute t0 = Σ eq * left_lo * right_lo (constant coefficient)
        /// 2. Compute t_inf = Σ eq * (left_hi - left_lo) * (right_hi - right_lo) (quadratic coefficient)
        /// 3. Use split_eq.computeCubicRoundPoly(t0, t_inf, current_claim)
        ///
        /// Returns [s(0), s(2), s(3)] - the compressed cubic polynomial
        pub fn computeRoundPolynomial(self: *Self) ![3]F {
            // Number of groups - each group has 2 adjacent values (lo, hi)
            const n = self.left_poly.boundLen();
            const num_groups = n / 2;

            if (num_groups == 0) {
                return [3]F{ self.current_claim, F.zero(), F.zero() };
            }

            // Get eq table projections for this round (matching Jolt's E_out_in_for_window)
            const eq_tables = self.split_eq.getWindowEqTables(self.current_round, 1);
            const E_out = eq_tables.E_out;
            const E_in = eq_tables.E_in;
            const num_xin_bits: u6 = if (E_in.len > 1) @intCast(std.math.log2_int(usize, E_in.len)) else 0;

            // Compute t0 and t_inf using the Gruen structure (matching Jolt's remaining_quadratic_evals)
            // Uses interleaved format: left[2*g] = lo, left[2*g+1] = hi
            var t0_sum: F = F.zero();
            var t_inf_sum: F = F.zero();

            for (0..E_out.len) |x_out| {
                var inner_t0: F = F.zero();
                var inner_t_inf: F = F.zero();

                for (0..E_in.len) |x_in| {
                    const g = (x_out << num_xin_bits) | x_in;

                    // Interleaved layout: lo at 2*g, hi at 2*g+1
                    if (g < num_groups) {
                        // Get left/right at lo and hi positions (adjacent pairs)
                        const l_lo = self.left_poly.evaluations[2 * g];
                        const l_hi = self.left_poly.evaluations[2 * g + 1];
                        const r_lo = self.right_poly.evaluations[2 * g];
                        const r_hi = self.right_poly.evaluations[2 * g + 1];

                        // t0 = left_lo * right_lo
                        const p0 = l_lo.mul(r_lo);

                        // t_inf = (left_hi - left_lo) * (right_hi - right_lo)
                        // This is the "slope" term that becomes the quadratic coefficient
                        const slope = l_hi.sub(l_lo).mul(r_hi.sub(r_lo));

                        // Weight by E_in
                        const e_in = E_in[x_in];
                        inner_t0 = inner_t0.add(p0.mul(e_in));
                        inner_t_inf = inner_t_inf.add(slope.mul(e_in));
                    }
                }

                // Weight by E_out
                const e_out = E_out[x_out];
                t0_sum = t0_sum.add(inner_t0.mul(e_out));
                t_inf_sum = t_inf_sum.add(inner_t_inf.mul(e_out));
            }

            // Debug output for first 3 rounds (matching Jolt's debug)
            if (self.current_round < 3) {
                const t0_be = t0_sum.toBytesBE();
                const tinf_be = t_inf_sum.toBytesBE();
                const claim_be = self.current_claim.toBytesBE();
                std.debug.print("[ZOLT PRODUCT round {}] t0 last 8 bytes (LE): {x:0>2} {x:0>2} {x:0>2} {x:0>2} {x:0>2} {x:0>2} {x:0>2} {x:0>2}\n", .{
                    self.current_round,
                    t0_be[31], t0_be[30], t0_be[29], t0_be[28], t0_be[27], t0_be[26], t0_be[25], t0_be[24],
                });
                std.debug.print("[ZOLT PRODUCT round {}] t_inf last 8 bytes (LE): {x:0>2} {x:0>2} {x:0>2} {x:0>2} {x:0>2} {x:0>2} {x:0>2} {x:0>2}\n", .{
                    self.current_round,
                    tinf_be[31], tinf_be[30], tinf_be[29], tinf_be[28], tinf_be[27], tinf_be[26], tinf_be[25], tinf_be[24],
                });
                std.debug.print("[ZOLT PRODUCT round {}] previous_claim last 8 bytes (LE): {x:0>2} {x:0>2} {x:0>2} {x:0>2} {x:0>2} {x:0>2} {x:0>2} {x:0>2}\n", .{
                    self.current_round,
                    claim_be[31], claim_be[30], claim_be[29], claim_be[28], claim_be[27], claim_be[26], claim_be[25], claim_be[24],
                });
            }

            // Use Gruen's polynomial construction to get the cubic round polynomial
            const evals = self.split_eq.computeCubicRoundPoly(t0_sum, t_inf_sum, self.current_claim);

            // Convert evaluations to compressed coefficients [c0, c2, c3]
            return poly_mod.UniPoly(F).evalsToCompressed(evals);
        }

        /// Bind the challenge for this round and update state
        pub fn bindChallenge(self: *Self, challenge: F) !void {
            // Bind left and right polynomials
            self.left_poly.bindLow(challenge);
            self.right_poly.bindLow(challenge);

            // Bind split eq
            self.split_eq.bind(challenge);

            self.current_round += 1;
        }

        /// Update the claim based on round polynomial evaluation at challenge
        pub fn updateClaim(self: *Self, round_evals: [4]F, challenge: F) void {
            // Compute s(challenge) using Lagrange interpolation
            self.current_claim = evaluateCubicAtPointGeneric(F, round_evals, challenge);
        }

        /// Get the final claim after all rounds
        pub fn getFinalClaim(self: *const Self) F {
            if (self.left_poly.boundLen() == 0) return F.zero();

            // Final claim = left(r) * right(r) * eq(tau, r)
            // At this point, polynomials are fully bound, so we get single values
            const left_final = self.left_poly.evaluations[0];
            const right_final = self.right_poly.evaluations[0];

            return left_final.mul(right_final);
        }

        /// Compute opening claims for the 8 unique factor polynomials
        ///
        /// These are the MLE evaluations at r_cycle (the accumulated challenges)
        pub fn computeOpeningClaims(
            self: *const Self,
            cycle_witnesses: []const constraints.R1CSCycleInputs(F),
            r_cycle: []const F,
        ) ![8]F {
            // Compute MLE evaluations of each factor polynomial at r_cycle
            var claims: [8]F = [_]F{F.zero()} ** 8;

            // Compute eq(r_cycle, j) for all cycles j
            const eq_evals = try computeEqEvalsGeneric(F, self.allocator, r_cycle, cycle_witnesses.len);
            defer self.allocator.free(eq_evals);

            // Accumulate each factor
            for (cycle_witnesses, 0..) |witness, j| {
                const eq_j = if (j < eq_evals.len) eq_evals[j] else F.zero();

                const product_inputs = extractProductInputs(F, &witness, cycle_witnesses, j);

                for (0..8) |factor_idx| {
                    claims[factor_idx] = claims[factor_idx].add(
                        product_inputs.factors[factor_idx].mul(eq_j),
                    );
                }
            }

            return claims;
        }
    };
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Extract product constraint inputs from R1CS cycle inputs
fn extractProductInputs(
    comptime F: type,
    witness: *const constraints.R1CSCycleInputs(F),
    all_witnesses: []const constraints.R1CSCycleInputs(F),
    cycle_idx: usize,
) ProductCycleInputs(F) {
    const inputs = ProductCycleInputs(F){
        .factors = [8]F{
            // 0: LeftInstructionInput
            witness.values[constraints.R1CSInputIndex.LeftInstructionInput.toIndex()],
            // 1: RightInstructionInput
            witness.values[constraints.R1CSInputIndex.RightInstructionInput.toIndex()],
            // 2: IsRdNotZero - directly from FlagIsRdNotZero (rd register index != 0)
            // In Jolt, this is InstructionFlags::IsRdNotZero
            witness.values[constraints.R1CSInputIndex.FlagIsRdNotZero.toIndex()],
            // 3: WriteLookupOutputToRDFlag (OpFlags::WriteLookupOutputToRD)
            witness.values[constraints.R1CSInputIndex.FlagWriteLookupOutputToRD.toIndex()],
            // 4: JumpFlag (OpFlags::Jump)
            witness.values[constraints.R1CSInputIndex.FlagJump.toIndex()],
            // 5: LookupOutput
            witness.values[constraints.R1CSInputIndex.LookupOutput.toIndex()],
            // 6: BranchFlag (InstructionFlags::Branch)
            // Directly from FlagBranch (opcode == 0x63)
            witness.values[constraints.R1CSInputIndex.FlagBranch.toIndex()],
            // 7: NextIsNoop - 1 if next instruction is a noop
            // NextIsNoop = !not_next_noop = trace[t+1].IsNoop (for t+1 < len)
            // For last cycle, NextIsNoop = true (not_next_noop = false)
            blk: {
                if (cycle_idx + 1 < all_witnesses.len) {
                    // Use next cycle's IsNoop flag
                    const next_witness = &all_witnesses[cycle_idx + 1];
                    break :blk next_witness.values[constraints.R1CSInputIndex.FlagIsNoop.toIndex()];
                }
                // Last cycle: not_next_noop = false, so NextIsNoop = true
                break :blk F.one();
            },
        },
    };
    return inputs;
}

/// Compute Lagrange basis evaluations at r0 over the 5-point domain {-2, -1, 0, 1, 2}
fn computeLagrangeWeightsGeneric(comptime F: type, allocator: Allocator, r0: F) ![5]F {
    const LagrangePoly = univariate_skip.LagrangePolynomial(F);
    const weights = try LagrangePoly.evals(DOMAIN_SIZE, r0, allocator);
    defer allocator.free(weights);

    var result: [5]F = undefined;
    for (0..5) |i| {
        result[i] = weights[i];
    }
    return result;
}

/// Compute eq(r, j) for all j in [0, n)
fn computeEqEvalsGeneric(comptime F: type, allocator: Allocator, r: []const F, n: usize) ![]F {
    const padded_n = nextPowerOfTwo(n);
    const log_n = std.math.log2_int(usize, padded_n);

    const result = try allocator.alloc(F, padded_n);
    errdefer allocator.free(result);

    // Initialize with 1
    result[0] = F.one();
    var current_size: usize = 1;

    // Build eq table iteratively
    for (0..log_n) |i| {
        const ri = if (i < r.len) r[i] else F.zero();
        const one_minus_ri = F.one().sub(ri);

        // Process in reverse to avoid overwriting
        var j = current_size;
        while (j > 0) {
            j -= 1;
            const val = result[j];
            result[j] = val.mul(one_minus_ri); // j with bit i = 0
            result[j + current_size] = val.mul(ri); // j with bit i = 1
        }
        current_size *= 2;
    }

    return result;
}

/// Evaluate cubic polynomial at a point using Lagrange interpolation
fn evaluateCubicAtPointGeneric(comptime F: type, evals: [4]F, x: F) F {
    // Lagrange interpolation at points 0, 1, 2, 3
    // L_i(x) = Π_{j≠i} (x - j) / (i - j)

    const x_minus_0 = x;
    const x_minus_1 = x.sub(F.one());
    const x_minus_2 = x.sub(F.fromU64(2));
    const x_minus_3 = x.sub(F.fromU64(3));

    // L_0(x) = (x-1)(x-2)(x-3) / (0-1)(0-2)(0-3) = (x-1)(x-2)(x-3) / (-6)
    const L0 = x_minus_1.mul(x_minus_2).mul(x_minus_3).mul(F.fromU64(6).neg().inverse().?);

    // L_1(x) = x(x-2)(x-3) / (1-0)(1-2)(1-3) = x(x-2)(x-3) / (1*-1*-2) = x(x-2)(x-3) / 2
    const L1 = x_minus_0.mul(x_minus_2).mul(x_minus_3).mul(F.fromU64(2).inverse().?);

    // L_2(x) = x(x-1)(x-3) / (2-0)(2-1)(2-3) = x(x-1)(x-3) / (2*1*-1) = x(x-1)(x-3) / (-2)
    const L2 = x_minus_0.mul(x_minus_1).mul(x_minus_3).mul(F.fromU64(2).neg().inverse().?);

    // L_3(x) = x(x-1)(x-2) / (3-0)(3-1)(3-2) = x(x-1)(x-2) / (3*2*1) = x(x-1)(x-2) / 6
    const L3 = x_minus_0.mul(x_minus_1).mul(x_minus_2).mul(F.fromU64(6).inverse().?);

    return evals[0].mul(L0)
        .add(evals[1].mul(L1))
        .add(evals[2].mul(L2))
        .add(evals[3].mul(L3));
}

/// Round up to next power of two
fn nextPowerOfTwo(n: usize) usize {
    if (n == 0) return 1;
    var v = n - 1;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v |= v >> 32;
    return v + 1;
}

// ============================================================================
// Tests
// ============================================================================

const testing = std.testing;
const BN254Scalar = @import("../../field/mod.zig").BN254Scalar;

test "product remainder prover: lagrange weights computation" {
    const F = BN254Scalar;

    // Test with r0 = 0 (should give L_i(0) = δ_{i,2} since domain point 2 maps to value 0)
    const weights = try computeLagrangeWeightsGeneric(F, testing.allocator, F.zero());

    // Sum of Lagrange weights at any point should be 1
    var sum = F.zero();
    for (weights) |w| {
        sum = sum.add(w);
    }
    try testing.expect(sum.eql(F.one()));
}

test "product remainder prover: cubic evaluation" {
    const F = BN254Scalar;

    // Test polynomial s(X) = 1 + 2X + 3X^2 + 4X^3
    // s(0) = 1, s(1) = 1+2+3+4 = 10, s(2) = 1+4+12+32 = 49, s(3) = 1+6+27+108 = 142
    const evals = [4]F{
        F.fromU64(1),
        F.fromU64(10),
        F.fromU64(49),
        F.fromU64(142),
    };

    // Evaluate at X = 2
    const result = evaluateCubicAtPointGeneric(F, evals, F.fromU64(2));
    try testing.expect(result.eql(F.fromU64(49)));

    // Evaluate at X = 0
    const result0 = evaluateCubicAtPointGeneric(F, evals, F.zero());
    try testing.expect(result0.eql(F.fromU64(1)));
}

test "product remainder prover: eq evals" {
    const F = BN254Scalar;

    // eq(r, j) where r = [r0, r1] and j ∈ {0, 1, 2, 3}
    // eq(r, 0) = (1-r0)(1-r1)
    // eq(r, 1) = r0(1-r1)
    // eq(r, 2) = (1-r0)r1
    // eq(r, 3) = r0*r1

    const r = [_]F{ F.fromU64(2), F.fromU64(3) };
    const evals = try computeEqEvalsGeneric(F, testing.allocator, &r, 4);
    defer testing.allocator.free(evals);

    try testing.expectEqual(@as(usize, 4), evals.len);

    // Verify eq(r, 0) = (1-2)(1-3) = (-1)(-2) = 2
    const expected_0 = F.one().sub(F.fromU64(2)).mul(F.one().sub(F.fromU64(3)));
    try testing.expect(evals[0].eql(expected_0));
}
