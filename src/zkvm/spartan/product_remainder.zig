//! ProductVirtualRemainderProver - Stage 2 Product Virtualization Sumcheck
//!
//! This module implements the sumcheck prover for the "remaining" rounds of
//! product virtualization after the univariate skip first round.
//!
//! ## Protocol Overview
//!
//! Stage 2 proves the 3 product constraints:
//! 1. Product = LeftInstructionInput * RightInstructionInput
//! 2. ShouldBranch = LookupOutput * InstructionFlags(Branch)
//! 3. ShouldJump = OpFlags(Jump) * (1 - NextIsNoop)
//!
//! ## Fused Sumcheck
//!
//! These 3 constraints are fused into 2 polynomials (left/right) using Lagrange weights
//! from the first-round challenge r0:
//!
//!   fused_left(x) = Σᵢ wᵢ * leftᵢ(x)
//!   fused_right(x) = Σᵢ wᵢ * rightᵢ(x)
//!
//! Where wᵢ = Lᵢ(r0) are Lagrange basis polynomials evaluated at r0 over the 3-point domain.
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

// Debug output control - set to true to enable verbose debug prints
const debug_verbose = false;
fn dbg(comptime fmt: []const u8, args: anytype) void {
    if (debug_verbose) std.debug.print(fmt, args);
}

const Allocator = std.mem.Allocator;
const ThreadPool = @import("../../utils/thread_pool.zig").ThreadPool;
const field_mod = @import("../../field/mod.zig");
const UnreducedProductAccum = field_mod.UnreducedProductAccum;

const constraints = @import("../r1cs/constraints.zig");
const univariate_skip = @import("../r1cs/univariate_skip.zig");
const poly_mod = @import("../../poly/mod.zig");
const GruenSplitEqPolynomial = poly_mod.GruenSplitEqPolynomial;
const DensePolynomial = poly_mod.DensePolynomial;
const utils = @import("../../utils/mod.zig");

/// Number of product constraints
pub const NUM_PRODUCT_CONSTRAINTS: usize = 3;

/// Domain size for product virtualization univariate skip
pub const DOMAIN_SIZE: usize = univariate_skip.PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DOMAIN_SIZE;

/// Degree of the remainder sumcheck (product of 2 multilinear = degree 2, plus eq = degree 3)
pub const REMAINDER_DEGREE: usize = 3;

/// The 8 unique factor polynomial indices that appear in the 3 product constraints
/// Matches upstream Jolt's PRODUCT_UNIQUE_FACTOR_VIRTUALS
/// Note: WriteLookupOutputToRDFlag and VirtualInstructionFlag are opened at the product
/// cycle point for downstream stages, even though they don't appear in the 3 products.
pub const ProductFactorIndex = enum(usize) {
    LeftInstructionInput = 0,
    RightInstructionInput = 1,
    JumpFlag = 2, // OpFlags::Jump
    WriteLookupOutputToRDFlag = 3, // OpFlags::WriteLookupOutputToRD
    LookupOutput = 4,
    BranchFlag = 5, // InstructionFlags::Branch
    NextIsNoop = 6,
    VirtualInstructionFlag = 7, // OpFlags::VirtualInstruction
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
                    // 2: JumpFlag (OpFlags::Jump)
                    r1cs_inputs.values[constraints.R1CSInputIndex.FlagJump.toIndex()],
                    // 3: WriteLookupOutputToRDFlag (OpFlags::WriteLookupOutputToRD)
                    r1cs_inputs.values[constraints.R1CSInputIndex.FlagWriteLookupOutputToRD.toIndex()],
                    // 4: LookupOutput
                    r1cs_inputs.values[constraints.R1CSInputIndex.LookupOutput.toIndex()],
                    // 5: BranchFlag (InstructionFlags::Branch)
                    r1cs_inputs.values[constraints.R1CSInputIndex.FlagBranch.toIndex()],
                    // 6: NextIsNoop
                    r1cs_inputs.values[constraints.R1CSInputIndex.FlagIsNoop.toIndex()], // Will be set properly from trace context
                    // 7: VirtualInstructionFlag (OpFlags::VirtualInstruction)
                    r1cs_inputs.values[constraints.R1CSInputIndex.FlagVirtualInstruction.toIndex()],
                },
            };
        }

        /// Get factor value by index
        pub fn getFactor(self: *const Self, idx: ProductFactorIndex) F {
            return self.factors[@intFromEnum(idx)];
        }

        /// Compute fused left polynomial value at this cycle using Lagrange weights
        ///
        /// fused_left = w[0]*LeftInstructionInput + w[1]*LookupOutput + w[2]*JumpFlag
        pub fn fusedLeft(self: *const Self, weights: *const [3]F) F {
            return weights[0].mul(self.factors[0]) // LeftInstructionInput
                .add(weights[1].mul(self.factors[4])) // LookupOutput
                .add(weights[2].mul(self.factors[2])); // JumpFlag
        }

        /// Compute fused right polynomial value at this cycle using Lagrange weights
        ///
        /// fused_right = w[0]*RightInstructionInput + w[1]*BranchFlag + w[2]*(1 - NextIsNoop)
        pub fn fusedRight(self: *const Self, weights: *const [3]F) F {
            const one_minus_noop = F.one().sub(self.factors[6]);
            return weights[0].mul(self.factors[1]) // RightInstructionInput
                .add(weights[1].mul(self.factors[5])) // BranchFlag
                .add(weights[2].mul(one_minus_noop)); // (1 - NextIsNoop)
        }
    };
}

/// ProductVirtualRemainderProver - sumcheck prover for product virtualization
pub fn ProductVirtualRemainderProver(comptime F: type) type {
    return struct {
        const Self = @This();

        /// Lagrange basis evaluations at r0 over the 3-point domain
        lagrange_weights: [3]F,
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
        /// Thread pool for parallel compute/bind
        thread_pool: ?*ThreadPool = null,

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

            // Compute Lagrange weights at r0 over the 3-point domain {-1, 0, 1}
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
            // Flattened parallel reduce over all groups g = (x_out << num_xin_bits) | x_in
            const ComputeCtx = struct {
                left: []const F,
                right: []const F,
                e_out: []const F,
                e_in: []const F,
                num_xin_bits: u6,
            };
            const ctx = ComputeCtx{
                .left = self.left_poly.evaluations,
                .right = self.right_poly.evaluations,
                .e_out = E_out,
                .e_in = E_in,
                .num_xin_bits = num_xin_bits,
            };

            const mapFn = struct {
                fn f(c: ComputeCtx, start: usize, end: usize) [2]F {
                    @setEvalBranchQuota(10000);
                    const use_deferred = comptime @hasDecl(F, "mulToProductAccum");
                    const e_in_len = c.e_in.len;
                    const xin_mask: usize = if (e_in_len > 1) e_in_len - 1 else 0;

                    if (use_deferred) {
                        var acc_t0 = UnreducedProductAccum.zero();
                        var acc_tinf = UnreducedProductAccum.zero();
                        for (start..end) |g| {
                            const l_lo = c.left[2 * g];
                            const l_hi = c.left[2 * g + 1];
                            const r_lo = c.right[2 * g];
                            const r_hi = c.right[2 * g + 1];
                            const p0 = l_lo.mul(r_lo);
                            const slope = l_hi.sub(l_lo).mul(r_hi.sub(r_lo));
                            const x_in = g & xin_mask;
                            const x_out = g >> @intCast(c.num_xin_bits);
                            const e_weight = if (x_out < c.e_out.len and (e_in_len <= 1 or x_in < e_in_len))
                                c.e_out[x_out].mul(if (e_in_len <= 1) F.one() else c.e_in[x_in])
                            else
                                F.zero();
                            acc_t0.addAssign(p0.mulToProductAccum(e_weight));
                            acc_tinf.addAssign(slope.mulToProductAccum(e_weight));
                        }
                        return .{ acc_t0.reduce(), acc_tinf.reduce() };
                    } else {
                        var t0_local: F = F.zero();
                        var tinf_local: F = F.zero();
                        for (start..end) |g| {
                            const l_lo = c.left[2 * g];
                            const l_hi = c.left[2 * g + 1];
                            const r_lo = c.right[2 * g];
                            const r_hi = c.right[2 * g + 1];
                            const p0 = l_lo.mul(r_lo);
                            const slope = l_hi.sub(l_lo).mul(r_hi.sub(r_lo));
                            const x_in = g & xin_mask;
                            const x_out = g >> @intCast(c.num_xin_bits);
                            const e_weight = if (x_out < c.e_out.len and (e_in_len <= 1 or x_in < e_in_len))
                                c.e_out[x_out].mul(if (e_in_len <= 1) F.one() else c.e_in[x_in])
                            else
                                F.zero();
                            t0_local = t0_local.add(p0.mul(e_weight));
                            tinf_local = tinf_local.add(slope.mul(e_weight));
                        }
                        return .{ t0_local, tinf_local };
                    }
                }
            }.f;

            const reduceFn = struct {
                fn f(a: [2]F, b: [2]F) [2]F {
                    return .{ a[0].add(b[0]), a[1].add(b[1]) };
                }
            }.f;

            const identity = [2]F{ F.zero(), F.zero() };
            const sums = if (self.thread_pool) |tp|
                tp.parallelReduce([2]F, num_groups, identity, ctx, mapFn, reduceFn)
            else
                mapFn(ctx, 0, num_groups);

            const t0_sum = sums[0];
            const t_inf_sum = sums[1];

            // Debug output for first 3 rounds (matching Jolt's debug)
            if (self.current_round < 3) {
                const t0_be = t0_sum.toBytesBE();
                const tinf_be = t_inf_sum.toBytesBE();
                const claim_be = self.current_claim.toBytesBE();
                dbg("[ZOLT PRODUCT round {}] t0 last 8 bytes (LE): {x:0>2} {x:0>2} {x:0>2} {x:0>2} {x:0>2} {x:0>2} {x:0>2} {x:0>2}\n", .{
                    self.current_round,
                    t0_be[31], t0_be[30], t0_be[29], t0_be[28], t0_be[27], t0_be[26], t0_be[25], t0_be[24],
                });
                dbg("[ZOLT PRODUCT round {}] t_inf last 8 bytes (LE): {x:0>2} {x:0>2} {x:0>2} {x:0>2} {x:0>2} {x:0>2} {x:0>2} {x:0>2}\n", .{
                    self.current_round,
                    tinf_be[31], tinf_be[30], tinf_be[29], tinf_be[28], tinf_be[27], tinf_be[26], tinf_be[25], tinf_be[24],
                });
                dbg("[ZOLT PRODUCT round {}] previous_claim last 8 bytes (LE): {x:0>2} {x:0>2} {x:0>2} {x:0>2} {x:0>2} {x:0>2} {x:0>2} {x:0>2}\n", .{
                    self.current_round,
                    claim_be[31], claim_be[30], claim_be[29], claim_be[28], claim_be[27], claim_be[26], claim_be[25], claim_be[24],
                });
                // Print split_eq state
                dbg("[ZOLT PRODUCT round {}] split_eq.current_scalar = {any}\n", .{
                    self.current_round,
                    self.split_eq.current_scalar.toBytesBE(),
                });
                dbg("[ZOLT PRODUCT round {}] E_out.len = {}, E_in.len = {}\n", .{
                    self.current_round,
                    E_out.len,
                    E_in.len,
                });
            }

            // Print final state after last round
            if (self.current_round + 1 == self.num_cycle_vars) {
                dbg("[ZOLT PRODUCT FINAL] left[0] = {any}\n", .{self.left_poly.evaluations[0].toBytesBE()});
                dbg("[ZOLT PRODUCT FINAL] right[0] = {any}\n", .{self.right_poly.evaluations[0].toBytesBE()});
                dbg("[ZOLT PRODUCT FINAL] left[1] = {any}\n", .{self.left_poly.evaluations[1].toBytesBE()});
                dbg("[ZOLT PRODUCT FINAL] right[1] = {any}\n", .{self.right_poly.evaluations[1].toBytesBE()});
            }

            // Use Gruen's polynomial construction to get the cubic round polynomial
            const evals = self.split_eq.computeCubicRoundPoly(t0_sum, t_inf_sum, self.current_claim);

            // Convert evaluations to compressed coefficients [c0, c2, c3]
            return poly_mod.UniPoly(F).evalsToCompressed(evals);
        }

        /// Bind the challenge for this round and update state
        pub fn bindChallenge(self: *Self, challenge: F) !void {
            // Bind left and right polynomials concurrently
            if (self.thread_pool) |tp| {
                const BindCtx = struct { left: *DensePolynomial(F), right: *DensePolynomial(F), c: F };
                const bctx = BindCtx{ .left = &self.left_poly, .right = &self.right_poly, .c = challenge };
                tp.parallelForForce(2, bctx, struct {
                    fn f(bc: BindCtx, idx: usize) void {
                        if (idx == 0) bc.left.bindLow(bc.c) else bc.right.bindLow(bc.c);
                    }
                }.f);
            } else {
                self.left_poly.bindLow(challenge);
                self.right_poly.bindLow(challenge);
            }

            // Bind split eq (fast, O(window_size))
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
            // 2: JumpFlag (OpFlags::Jump)
            witness.values[constraints.R1CSInputIndex.FlagJump.toIndex()],
            // 3: WriteLookupOutputToRDFlag (OpFlags::WriteLookupOutputToRD)
            witness.values[constraints.R1CSInputIndex.FlagWriteLookupOutputToRD.toIndex()],
            // 4: LookupOutput
            witness.values[constraints.R1CSInputIndex.LookupOutput.toIndex()],
            // 5: BranchFlag (InstructionFlags::Branch)
            witness.values[constraints.R1CSInputIndex.FlagBranch.toIndex()],
            // 6: NextIsNoop - 1 if next instruction is a noop
            blk: {
                if (cycle_idx + 1 < all_witnesses.len) {
                    const next_witness = &all_witnesses[cycle_idx + 1];
                    break :blk next_witness.values[constraints.R1CSInputIndex.FlagIsNoop.toIndex()];
                }
                // Last cycle: not_next_noop = false, so NextIsNoop = true
                break :blk F.one();
            },
            // 7: VirtualInstructionFlag (OpFlags::VirtualInstruction)
            witness.values[constraints.R1CSInputIndex.FlagVirtualInstruction.toIndex()],
        },
    };
    return inputs;
}

/// Compute Lagrange basis evaluations at r0 over the 3-point domain {-1, 0, 1}
fn computeLagrangeWeightsGeneric(comptime F: type, allocator: Allocator, r0: F) ![3]F {
    const LagrangePoly = univariate_skip.LagrangePolynomial(F);
    const weights = try LagrangePoly.evals(DOMAIN_SIZE, r0, allocator);
    defer allocator.free(weights);

    var result: [3]F = undefined;
    for (0..3) |i| {
        result[i] = weights[i];
    }
    return result;
}

/// Compute eq(r, j) for all j in [0, n)
///
/// Uses BIG ENDIAN indexing to match Jolt's EqPolynomial::evals:
/// - r[0] controls the MSB of the index
/// - r[n-1] controls the LSB of the index
/// - result[idx] = Π_i eq(bit_i(idx), r[i]) where bit_i is from MSB
fn computeEqEvalsGeneric(comptime F: type, allocator: Allocator, r: []const F, n: usize) ![]F {
    const padded_n = nextPowerOfTwo(n);
    const log_n = std.math.log2_int(usize, padded_n);

    const result = try allocator.alloc(F, padded_n);
    errdefer allocator.free(result);

    // Initialize with 1
    result[0] = F.one();
    var current_size: usize = 1;

    // Build eq table iteratively using BIG ENDIAN indexing (like Jolt)
    // Process r[0] first, which controls the MSB of the index
    // This means in each iteration, we process r[j] and double the table size
    // with r[j] controlling the newly added bit (which is the MSB of the new indices)
    for (0..log_n) |j| {
        const rj = if (j < r.len) r[j] else F.zero();
        const one_minus_rj = F.one().sub(rj);

        // Process from size-1 down to 0, setting:
        // - result[2*i+1] = result[i] * rj     (odd index = bit set = rj factor)
        // - result[2*i] = result[i] * (1-rj)  (even index = bit clear = (1-rj) factor)
        // But we need to iterate in reverse to avoid overwriting
        var i = current_size;
        while (i > 0) {
            i -= 1;
            const val = result[i];
            // In BIG ENDIAN: odd indices (bit=1) get rj, even indices (bit=0) get (1-rj)
            result[2 * i + 1] = val.mul(rj);
            result[2 * i] = val.mul(one_minus_rj);
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
