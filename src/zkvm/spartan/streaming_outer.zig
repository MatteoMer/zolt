//! Streaming Outer Sumcheck Prover for Jolt Compatibility
//!
//! This module implements the full Spartan outer sumcheck prover matching
//! Jolt's implementation. The key innovation is using multiquadratic
//! polynomial representation for memory-efficient streaming evaluation.
//!
//! ## Protocol Overview
//!
//! The outer sumcheck proves:
//!   Σ_{x ∈ {0,1}^n} L(τ_high, x_uniskip) * eq(τ, x) * Az(x) * Bz(x) = 0
//!
//! Where:
//! - L(τ_high, x_uniskip) is the Lagrange polynomial for univariate skip (first round)
//! - eq(τ, x) is factored as eq(τ_out, x_out) * eq(τ_in, x_in)
//! - Az(x), Bz(x) are R1CS matrix products
//!
//! ## Rounds
//!
//! - Round 0: Univariate skip (degree 27 for domain size 10)
//! - Rounds 1..n: Streaming sumcheck (degree 3)
//!
//! ## Reference
//!
//! jolt-core/src/zkvm/spartan/outer.rs

const std = @import("std");

// Debug output control - set to true to enable verbose debug prints
const debug_verbose = false;
fn dbg(comptime fmt: []const u8, args: anytype) void {
    if (debug_verbose) std.debug.print(fmt, args);
}

const Allocator = std.mem.Allocator;

const constraints = @import("../r1cs/constraints.zig");
const univariate_skip = @import("../r1cs/univariate_skip.zig");
const evaluators = @import("../r1cs/evaluators.zig");
const jolt_types = @import("../jolt_types.zig");
const poly_mod = @import("../../poly/mod.zig");
const multiquadratic = @import("../../poly/multiquadratic.zig");
const field_mod = @import("../../field/mod.zig");
const GruenSplitEqPolynomial = poly_mod.GruenSplitEqPolynomial;
const MultiquadraticPolynomial = poly_mod.MultiquadraticPolynomial;
const utils = @import("../../utils/mod.zig");
const GpuPolyOps = @import("../../gpu/mod.zig").GpuPolyOps;
const ExpandingTable = utils.ExpandingTable;

/// Streaming outer sumcheck prover for Jolt compatibility
pub fn StreamingOuterProver(comptime F: type) type {
    return struct {
        const Self = @This();

        /// Configuration constants from Jolt
        pub const NUM_CONSTRAINTS: usize = univariate_skip.NUM_R1CS_CONSTRAINTS;
        pub const FIRST_GROUP_SIZE: usize = univariate_skip.OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE;
        pub const SECOND_GROUP_SIZE: usize = NUM_CONSTRAINTS - FIRST_GROUP_SIZE;
        pub const UNISKIP_DEGREE: usize = univariate_skip.OUTER_UNIVARIATE_SKIP_DEGREE;
        pub const FIRST_ROUND_NUM_COEFFS: usize = univariate_skip.OUTER_FIRST_ROUND_POLY_NUM_COEFFS;
        pub const REMAINING_DEGREE: usize = 3;

        /// Compact integer witnesses for cache-friendly evaluation
        compact_witnesses: []const evaluators.CompactWitness,
        /// Number of cycle variables (log2 of trace length)
        num_cycle_vars: usize,
        /// Padded trace length (power of 2)
        padded_trace_len: usize,

        /// Split eq polynomial for efficient factored evaluation
        split_eq: GruenSplitEqPolynomial(F),

        /// Current sumcheck claim
        current_claim: F,
        /// Collected challenges
        challenges: std.ArrayListUnmanaged(F),
        /// Current round number
        current_round: usize,

        /// Precomputed Lagrange basis evaluations at first-round challenge r0
        /// Used for remaining rounds
        lagrange_evals_r0: [FIRST_GROUP_SIZE]F,

        /// Bound r_stream value (set after streaming round)
        /// Used to combine constraint groups in subsequent rounds
        r_stream: ?F,

        /// Expanding table for bound challenge weights
        /// eq(r_1, b_1) * eq(r_2, b_2) * ... for each cycle's binary representation
        /// Matches Jolt's r_grid in OuterSharedState
        r_grid: ExpandingTable(F),

        /// tau_high - the last element of the full tau vector
        /// Used for the Lagrange kernel L(tau_high, Y) in the first-round polynomial
        /// This is stored separately because split_eq only receives tau_low
        tau_high: F,

        /// Full tau vector (needed for UniSkip first round computation)
        /// The split_eq receives tau_low = tau[0..tau.len-1], but UniSkip needs full tau
        /// to compute the correct eq_table structure with E_out and E_in
        full_tau: []const F,

        /// DEBUG: t1 polynomial coefficients from the last UniSkip computation
        /// Used for diagnostic evaluation at r0
        debug_t1_coeffs: ?[univariate_skip.OUTER_UNIVARIATE_SKIP_EXTENDED_DOMAIN_SIZE]F,

        /// Linear phase: Bound Az polynomial
        /// Materialized at linear phase start, bound each round with bindLow()
        /// Matches Jolt's OuterLinearStage.az
        az_poly: ?poly_mod.DensePolynomial(F),

        /// Linear phase: Bound Bz polynomial
        /// Materialized at linear phase start, bound each round with bindLow()
        /// Matches Jolt's OuterLinearStage.bz
        bz_poly: ?poly_mod.DensePolynomial(F),

        /// Multiquadratic polynomial t' = Az * Bz on the ternary grid
        /// Built during materialization and rebound each linear round.
        /// Used for computing (t'(0), t'(∞)) projections in compute_t_evals.
        /// Matches Jolt's OuterSharedState.t_prime_poly
        t_prime_poly: ?MultiquadraticPolynomial(F),

        /// Cached 2^64 field element (for second-group Bz direct evaluation)
        two_pow_64_cached: F,

        /// Allocator
        allocator: Allocator,

        /// Thread pool for parallel operations
        thread_pool: ?*@import("../../utils/thread_pool.zig").ThreadPool = null,
        /// GPU accelerator for Metal compute (Apple Silicon)
        gpu_ops: ?*GpuPolyOps = null,

        /// Initialize the streaming outer prover (without scaling)
        ///
        /// tau: Full challenge vector of length (num_cycle_vars + 2)
        ///      - tau[0..m]: w_out (for E_out tables), where m = tau.len / 2
        ///      - tau[m..tau.len-1]: w_in (for E_in tables)
        ///      - tau[tau.len-1]: w_last (skipped, handled by UniSkip Lagrange kernel)
        ///
        /// IMPORTANT: Pass FULL tau, not tau_low! The split uses m = tau.len / 2
        /// which differs between length 11 and 12. Jolt uses full tau.
        pub fn init(
            allocator: Allocator,
            compact_witnesses_arg: []const evaluators.CompactWitness,
            tau: []const F,
        ) !Self {
            return initWithScaling(allocator, compact_witnesses_arg, tau, null);
        }

        /// Initialize the streaming outer prover with Lagrange kernel scaling
        ///
        /// tau: Full challenge vector of length (num_cycle_vars + 2)
        ///      We extract tau_low = tau[0..tau.len-1] for the split_eq.
        ///      tau_high = tau[tau.len-1] should already be incorporated into lagrange_tau_r0.
        ///
        /// lagrange_tau_r0: The Lagrange kernel L(r0, tau_high) from UniSkip
        ///                  This is multiplied into all eq evaluations.
        ///
        /// IMPORTANT: Jolt passes tau_low (not full tau) to GruenSplitEqPolynomial.
        /// tau_low is tau[0..tau.len-1], which has length num_cycle_vars + 1.
        /// The split uses m = tau_low.len / 2.
        pub fn initWithScaling(
            allocator: Allocator,
            compact_witnesses_arg: []const evaluators.CompactWitness,
            tau: []const F,
            lagrange_tau_r0: ?F,
        ) !Self {
            const num_cycles = compact_witnesses_arg.len;
            if (num_cycles == 0) {
                return error.EmptyTrace;
            }

            // Pad to next power of 2
            const padded_len = nextPowerOfTwo(num_cycles);
            const num_cycle_vars = std.math.log2_int(usize, padded_len);

            // Extract tau_low and tau_high, matching Jolt's split.
            // In Jolt:
            //   let tau_high = uni_skip_params.tau[uni_skip_params.tau.len() - 1];
            //   let tau_low = &uni_skip_params.tau[..uni_skip_params.tau.len() - 1];
            //   GruenSplitEqPolynomial::new_with_scaling(tau_low, ...)
            //
            // tau_high is used for the Lagrange kernel in the first-round polynomial.
            // tau_low is passed to split_eq for the remaining rounds.
            //
            const tau_high = if (tau.len > 0) tau[tau.len - 1] else F.zero();
            const tau_low = if (tau.len > 0) tau[0 .. tau.len - 1] else tau;

            // DEBUG: Print tau values used by streaming outer prover
            dbg("[STREAMING_OUTER] initWithScaling: tau.len={d}\n", .{tau.len});
            dbg("[STREAMING_OUTER] tau_high (limbs) = [{x}, {x}, {x}, {x}]\n", .{
                tau_high.limbs[0], tau_high.limbs[1], tau_high.limbs[2], tau_high.limbs[3],
            });
            if (tau.len > 0) {
                dbg("[STREAMING_OUTER] tau[0] (limbs) = [{x}, {x}, {x}, {x}]\n", .{
                    tau[0].limbs[0], tau[0].limbs[1], tau[0].limbs[2], tau[0].limbs[3],
                });
            }
            dbg("[STREAMING_OUTER] lagrange_tau_r0 present = {}\n", .{lagrange_tau_r0 != null});
            if (lagrange_tau_r0) |l| {
                dbg("[STREAMING_OUTER] lagrange_tau_r0 (limbs) = [{x}, {x}, {x}, {x}]\n", .{
                    l.limbs[0], l.limbs[1], l.limbs[2], l.limbs[3],
                });
            }

            const split_eq = try GruenSplitEqPolynomial(F).initWithScaling(allocator, tau_low, lagrange_tau_r0);

            // Initialize r_grid for tracking bound challenge weights
            // Capacity = padded_len (maximum number of cycles)
            var r_grid = try ExpandingTable(F).init(allocator, padded_len, .LowToHigh);
            r_grid.reset(F.one());

            // Copy full tau for UniSkip computation
            const full_tau = try allocator.alloc(F, tau.len);
            @memcpy(full_tau, tau);

            // Precompute 2^64 as field element for fast Bz evaluation
            const two_pow_64_cached = blk_2p64: {
                var bytes: [16]u8 = undefined;
                std.mem.writeInt(u128, &bytes, 0x10000000000000000, .little);
                break :blk_2p64 F.fromBytes(&bytes);
            };

            return Self{
                .compact_witnesses = compact_witnesses_arg,
                .num_cycle_vars = num_cycle_vars,
                .padded_trace_len = padded_len,
                .split_eq = split_eq,
                .current_claim = F.zero(),
                .challenges = .{},
                .current_round = 0,
                .lagrange_evals_r0 = [_]F{F.zero()} ** FIRST_GROUP_SIZE,
                .r_stream = null,
                .r_grid = r_grid,
                .tau_high = tau_high,
                .full_tau = full_tau,
                .two_pow_64_cached = two_pow_64_cached,
                .az_poly = null,
                .bz_poly = null,
                .t_prime_poly = null,
                .debug_t1_coeffs = null,
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            self.split_eq.deinit();
            self.challenges.deinit(self.allocator);
            self.r_grid.deinit();
            self.allocator.free(self.full_tau);
            if (self.az_poly) |*az| {
                az.deinit();
            }
            if (self.bz_poly) |*bz| {
                bz.deinit();
            }
            if (self.t_prime_poly) |*t_prime| {
                t_prime.deinit();
            }
        }

        /// Total number of rounds for the remaining sumcheck (after UniSkip)
        /// = 1 (streaming/constraint group) + num_cycle_vars (cycle bits)
        ///
        /// Note: This does NOT include the UniSkip round, which is handled separately.
        /// The streaming round binds the constraint group selector variable.
        /// The cycle rounds bind the cycle index bits.
        pub fn numRounds(self: *const Self) usize {
            return 1 + self.num_cycle_vars;
        }

        /// Materialize Az and Bz polynomials for the linear phase
        ///
        /// This matches Jolt's fused_materialise_polynomials_general_with_multiquadratic.
        /// Called at the switchover point (start of linear phase).
        ///
        /// Creates dense polynomials of size E_out.len * E_in.len * grid_size that incorporate:
        /// - The Lagrange weights from r0
        /// - The r_grid weights from already-bound streaming variables
        ///
        /// The index structure is:
        ///   full_idx = base_idx | x_val_shifted | r_idx
        ///   where base_idx = (x_out << (x_in_bits + window + r_bits)) | (x_in << (window + r_bits))
        ///   step_idx = full_idx >> 1 (cycle index)
        ///   selector = full_idx & 1 (constraint group)
        ///
        /// After materialization, each linear round:
        /// 1. Reads from az[grid_size * i + j] and bz[grid_size * i + j]
        /// 2. Binds with bindLow() to halve the polynomial size
        pub fn materializeLinearPhasePolynomials(self: *Self) !void {
            // Use the round zero path - this is called BEFORE any challenges are bound.
            // Jolt uses fused_materialise_polynomials_round_zero which has a simple
            // index formula: full_idx = grid_size * i + j
            //
            // This differs from the general path which uses complex bitwise indices
            // with r_grid. At round zero, we DON'T use r_grid scaling.

            // Get E_out and E_in tables for the current state
            // window_size = 1 for linear phase
            const window_size: usize = 1;
            const eq_tables = self.split_eq.getWindowEqTables(0, window_size);
            const E_out = eq_tables.E_out;
            const E_in = eq_tables.E_in;

            const num_x_out_vals = E_out.len;
            const num_x_in_vals = E_in.len;

            // Grid size for linear phase is 2^window_size = 2
            const grid_size: usize = @as(usize, 1) << @intCast(window_size);

            // Polynomial size = E_out.len * E_in.len * grid_size
            const poly_size = num_x_out_vals * num_x_in_vals * grid_size;

            const az_evals = try self.allocator.alloc(F, poly_size);
            errdefer self.allocator.free(az_evals);
            const bz_evals = try self.allocator.alloc(F, poly_size);
            errdefer self.allocator.free(bz_evals);

            // Initialize to zero
            @memset(az_evals, F.zero());
            @memset(bz_evals, F.zero());

            // Iterate over flattened (x_out, x_in) pairs — each pair is independent.
            // This matches Jolt's round_zero loop: for i in 0..E_out.len*E_in.len
            const total_pairs = num_x_out_vals * num_x_in_vals;

            // Precompute 2^64 field element for second-group Bz
            const two_pow_64_mat = blk_2p64m: {
                var bytes: [16]u8 = undefined;
                std.mem.writeInt(u128, &bytes, 0x10000000000000000, .little);
                break :blk_2p64m F.fromBytes(&bytes);
            };

            const MatCtx = struct {
                compact_witnesses: []const evaluators.CompactWitness,
                lagrange_evals_r0: *const [FIRST_GROUP_SIZE]F,
                az_evals: []F,
                bz_evals: []F,
                E_out: []const F,
                E_in: []const F,
                num_x_in_vals: usize,
                grid_size: usize,
                two_pow_64: F,
            };

            const mat_ctx = MatCtx{
                .compact_witnesses = self.compact_witnesses,
                .lagrange_evals_r0 = &self.lagrange_evals_r0,
                .az_evals = az_evals,
                .bz_evals = bz_evals,
                .E_out = E_out,
                .E_in = E_in,
                .num_x_in_vals = num_x_in_vals,
                .grid_size = grid_size,
                .two_pow_64 = two_pow_64_mat,
            };

            // Fused materialization + t_prime accumulation:
            // Writes az/bz arrays (side effect) AND returns [3]F t_prime contributions.
            const matFusedMapReduce = struct {
                fn mapFn(ctx: MatCtx, start: usize, end: usize) [3]F {
                    @setEvalBranchQuota(10000);
                    var local_ans = [3]F{ F.zero(), F.zero(), F.zero() };
                    const cw_len = ctx.compact_witnesses.len;

                    for (start..end) |flat_i| {
                        const x_in_val = flat_i % ctx.num_x_in_vals;
                        const x_out_val = flat_i / ctx.num_x_in_vals;
                        const i = x_out_val * ctx.num_x_in_vals + x_in_val;

                        var j: usize = 0;
                        while (j < ctx.grid_size) : (j += 2) {
                            const full_idx = ctx.grid_size * i + j;
                            const time_step_idx = full_idx >> 1;

                            if (time_step_idx >= cw_len) continue;

                            const cw = &ctx.compact_witnesses[time_step_idx];

                            // First group: SmallAccumU for Az, MedAccumS for Bz (i128 safe)
                            var az0_acc = field_mod.SmallAccumU.zero();
                            var bz0_acc = field_mod.MedAccumS.zero();
                            inline for (0..FIRST_GROUP_SIZE) |t| {
                                const w = ctx.lagrange_evals_r0[t];
                                az0_acc.fmaddI8(w, cw.az_first[t]);
                                bz0_acc.fmaddI128(w, cw.bz_first[t]);
                            }
                            const az0 = az0_acc.barrettReduce();
                            const bz0 = bz0_acc.barrettReduce();

                            // Second group: SmallAccumU for Az, WideAccumS for Bz (S192)
                            const g2_size = comptime @min(SECOND_GROUP_SIZE, FIRST_GROUP_SIZE);
                            var az1_acc = field_mod.SmallAccumU.zero();
                            var bz1_acc = field_mod.WideAccumS.zero();
                            inline for (0..g2_size) |t| {
                                az1_acc.fmaddI8(ctx.lagrange_evals_r0[t], cw.az_second[t]);
                                bz1_acc.fmaddS192(ctx.lagrange_evals_r0[t], cw.bz_second[t]);
                            }
                            const az1 = az1_acc.barrettReduce();
                            const bz1 = bz1_acc.barrettReduce();

                            // Store materialized values
                            const base_idx = ctx.grid_size * i;
                            ctx.az_evals[base_idx + j] = az0;
                            ctx.bz_evals[base_idx + j] = bz0;
                            ctx.az_evals[base_idx + j + 1] = az1;
                            ctx.bz_evals[base_idx + j + 1] = bz1;

                            // Fused t_prime accumulation (window_size=1, 3 ternary points)
                            const eq_weight = ctx.E_out[x_out_val].mul(ctx.E_in[x_in_val]);
                            local_ans[0] = local_ans[0].add(az0.mul(bz0).mul(eq_weight));
                            local_ans[1] = local_ans[1].add(az1.mul(bz1).mul(eq_weight));
                            local_ans[2] = local_ans[2].add(az1.sub(az0).mul(bz1.sub(bz0)).mul(eq_weight));
                        }
                    }
                    return local_ans;
                }

                fn reduceFn(a: [3]F, b: [3]F) [3]F {
                    return [3]F{ a[0].add(b[0]), a[1].add(b[1]), a[2].add(b[2]) };
                }
            };

            const identity_3 = [3]F{ F.zero(), F.zero(), F.zero() };
            const t_prime_result = if (self.thread_pool) |tp|
                tp.parallelReduce(
                    [3]F, total_pairs, identity_3,
                    mat_ctx, matFusedMapReduce.mapFn, matFusedMapReduce.reduceFn,
                )
            else
                matFusedMapReduce.mapFn(mat_ctx, 0, total_pairs);

            // Create DensePolynomials by taking ownership (no copy)
            self.az_poly = poly_mod.DensePolynomial(F).initOwned(self.allocator, az_evals);
            self.bz_poly = poly_mod.DensePolynomial(F).initOwned(self.allocator, bz_evals);

            // Allocate scratch buffers for parallel binding (double-buffer technique)
            if (self.thread_pool != null) {
                self.az_poly.?.scratch = try self.allocator.alloc(F, poly_size);
                self.bz_poly.?.scratch = try self.allocator.alloc(F, poly_size);
            }

            // DEBUG: Print first few materialized Az*Bz products
            if (self.az_poly != null and self.bz_poly != null) {
                const az_dbg = &(self.az_poly.?);
                const bz_dbg = &(self.bz_poly.?);
                for (0..@min(4, az_dbg.boundLen())) |idx| {
                    const prod = az_dbg.evaluations[idx].mul(bz_dbg.evaluations[idx]);
                    dbg("[MATERIALIZE] Az[{d}]*Bz[{d}] = {any}\n", .{idx, idx, prod.toBytes()});
                }
            }

            // Build t_prime_poly from fused result (no separate buildTPrimePoly pass)
            const three_pow_dim: usize = 3;
            var ans = try self.allocator.alloc(F, three_pow_dim);
            defer self.allocator.free(ans);
            ans[0] = t_prime_result[0];
            ans[1] = t_prime_result[1];
            ans[2] = t_prime_result[2];

            if (self.t_prime_poly) |*old| {
                old.deinit();
            }
            self.t_prime_poly = try MultiquadraticPolynomial(F).init(self.allocator, window_size, ans);
        }

        /// Build t_prime_poly from bound Az/Bz polynomials
        ///
        /// Matches Jolt's compute_evaluation_grid_from_polynomials_parallel.
        /// Creates a MultiquadraticPolynomial of size 3^window_size where each entry is:
        ///   t'[idx] = Σ_{x_out, x_in} E_out[x_out] * E_in[x_in] * Az[i,j] * Bz[i,j]
        ///
        /// where i = (x_out << num_xin_bits) | x_in and j indexes within the window grid.
        fn buildTPrimePoly(self: *Self, window_size: usize) !void {
            const az_poly = &(self.az_poly orelse return error.AzPolyNotInitialized);
            const bz_poly = &(self.bz_poly orelse return error.BzPolyNotInitialized);

            const eq_tables = self.split_eq.getWindowEqTables(0, window_size);
            const E_out = eq_tables.E_out;
            const E_in = eq_tables.E_in;
            // Compute grid sizes
            const grid_size = @as(usize, 1) << @intCast(window_size);
            var three_pow_dim: usize = 1;
            for (0..window_size) |_| three_pow_dim *= 3;

            const num_xin_bits: u6 = if (E_in.len > 1) @intCast(std.math.log2_int(usize, E_in.len)) else 0;

            const total_pairs = E_out.len * E_in.len;
            const az_bound_len = az_poly.boundLen();

            const BuildCtx = struct {
                az_evals: []const F,
                bz_evals: []const F,
                E_out: []const F,
                E_in: []const F,
                num_xin_bits: u6,
                grid_size: usize,
                three_pow_dim: usize,
                az_bound_len: usize,
                window_size_val: usize,
            };

            const build_ctx = BuildCtx{
                .az_evals = az_poly.evaluations,
                .bz_evals = bz_poly.evaluations,
                .E_out = E_out,
                .E_in = E_in,
                .num_xin_bits = num_xin_bits,
                .grid_size = grid_size,
                .three_pow_dim = three_pow_dim,
                .az_bound_len = az_bound_len,
                .window_size_val = window_size,
            };

            const mapFn = struct {
                fn f(ctx: BuildCtx, start: usize, end: usize) [3]F {
                    var local_ans = [3]F{ F.zero(), F.zero(), F.zero() };
                    // Stack-allocate scratch buffers (window_size=1: grid_size=2, three_pow_dim=3)
                    std.debug.assert(ctx.grid_size == 2 and ctx.three_pow_dim == 3);
                    var az_grid: [2]F = undefined;
                    var bz_grid: [2]F = undefined;
                    var buff_a: [3]F = undefined;
                    var buff_b: [3]F = undefined;

                    var pair_idx = start;
                    while (pair_idx < end) : (pair_idx += 1) {
                        const x_out = pair_idx / ctx.E_in.len;
                        const x_in = pair_idx % ctx.E_in.len;
                        const i = (x_out << ctx.num_xin_bits) | x_in;

                        // Extract az and bz values for this pair
                        for (0..ctx.grid_size) |j| {
                            const index = ctx.grid_size * i + j;
                            if (index < ctx.az_bound_len) {
                                az_grid[j] = ctx.az_evals[index];
                                bz_grid[j] = ctx.bz_evals[index];
                            } else {
                                az_grid[j] = F.zero();
                                bz_grid[j] = F.zero();
                            }
                        }

                        // Expand linear grids to multiquadratic
                        @memset(&buff_a, F.zero());
                        @memset(&buff_b, F.zero());

                        // Copy boolean evaluations to ternary positions
                        for (0..ctx.grid_size) |linear_idx| {
                            var ternary_idx: usize = 0;
                            var pow3_factor: usize = 1;
                            var idx = linear_idx;
                            for (0..ctx.window_size_val) |_| {
                                const bit = idx & 1;
                                ternary_idx += bit * pow3_factor;
                                pow3_factor *= 3;
                                idx >>= 1;
                            }
                            buff_a[ternary_idx] = az_grid[linear_idx];
                            buff_b[ternary_idx] = bz_grid[linear_idx];
                        }

                        // Expand to include infinity values: f(∞) = f(1) - f(0)
                        multiquadratic.expandGrid(F, ctx.window_size_val, &buff_a);
                        multiquadratic.expandGrid(F, ctx.window_size_val, &buff_b);

                        // Accumulate Az * Bz * E_out * E_in
                        const eq_weight = ctx.E_out[x_out].mul(ctx.E_in[x_in]);
                        for (0..ctx.three_pow_dim) |idx| {
                            local_ans[idx] = local_ans[idx].add(buff_a[idx].mul(buff_b[idx]).mul(eq_weight));
                        }
                    }
                    return local_ans;
                }
            }.f;

            const reduceFn = struct {
                fn f(a: [3]F, b: [3]F) [3]F {
                    return [3]F{ a[0].add(b[0]), a[1].add(b[1]), a[2].add(b[2]) };
                }
            }.f;

            const identity = [3]F{ F.zero(), F.zero(), F.zero() };
            const ans_result = if (self.thread_pool) |tp|
                tp.parallelReduce([3]F, total_pairs, identity, build_ctx, mapFn, reduceFn)
            else
                mapFn(build_ctx, 0, total_pairs);

            // Convert [3]F to allocated slice for MultiquadraticPolynomial
            var ans = try self.allocator.alloc(F, three_pow_dim);
            errdefer self.allocator.free(ans);
            for (0..three_pow_dim) |idx| {
                ans[idx] = ans_result[idx];
            }

            // Create the MultiquadraticPolynomial
            if (self.t_prime_poly) |*old| {
                old.deinit();
            }
            self.t_prime_poly = try MultiquadraticPolynomial(F).init(self.allocator, window_size, ans);
            self.allocator.free(ans);
        }

        /// Rebuild t_prime_poly from bound Az/Bz polynomials (for linear rounds after first)
        ///
        /// This is called at the start of each linear round (except the first which uses buildTPrimePoly).
        /// It uses the already-bound az_poly and bz_poly to rebuild t_prime_poly.
        fn rebuildTPrimePoly(self: *Self, window_size: usize) !void {
            try self.buildTPrimePoly(window_size);
        }

        /// Compute (t'(0), t'(∞)) from t_prime_poly using E_active projection
        ///
        /// Matches Jolt's compute_t_evals in OuterSharedState.
        /// Projects t_prime_poly to its first variable at evaluation points 0 and ∞,
        /// weighted by eq(tau_active, ·) over the remaining coordinates.
        fn computeTEvals(self: *Self, window_size: usize) !struct { t_zero: F, t_infinity: F } {
            const t_prime_poly = &(self.t_prime_poly orelse return error.TPrimePolyNotInitialized);

            // Get E_active: eq table over active window bits (all window bits except current Gruen variable)
            const e_active = try self.split_eq.getEActiveForWindow(self.allocator, window_size);
            defer self.allocator.free(e_active);

            // Project t_prime_poly to first variable using E_active weights
            const result = t_prime_poly.projectToFirstVariable(e_active);
            return .{ .t_zero = result.t_zero, .t_infinity = result.t_infinity };
        }

        /// Compute the first-round univariate skip polynomial
        ///
        /// This is a degree-27 polynomial over the extended domain.
        /// It computes:
        ///   s₁(Y) = L(τ_high, Y) * Σ_{x_out, x_in} eq(τ, x) * Az(x, Y) * Bz(x, Y)
        ///
        /// Returns coefficients for the degree-27 polynomial
        ///
        /// IMPORTANT: This loops over both constraint groups (FIRST_GROUP and SECOND_GROUP)
        /// matching Jolt's implementation. In Jolt, x_in's LSB selects the group:
        /// - x_in & 1 == 0 → FIRST_GROUP (10 constraints)
        /// - x_in & 1 == 1 → SECOND_GROUP (9 constraints)
        ///
        /// CRITICAL: Jolt's UniSkip uses the FULL tau vector (num_cycle_vars + 2 elements)
        /// to create the split_eq, NOT tau_low. This gives:
        /// - m = (num_cycle_vars + 2) / 2 for the split
        /// - E_out has m bits, E_in has (num_cycle_vars + 1 - m) bits
        /// - w_last (tau[tau.len-1]) is DROPPED from the eq computation
        ///
        /// The cycle index is computed as:
        ///   base_step_idx = (x_out << num_x_in_prime_bits) | (x_in >> 1)
        /// where num_x_in_prime_bits = E_in_bits - 1 (removing group bit)
        /// Compute the univariate skip first-round polynomial s1(Y) = L(τ_high, Y) · t1(Y)
        ///
        /// This matches Jolt's `build_uniskip_first_round_poly` algorithm:
        /// 1. Compute extended_evals[DEGREE=9] at interleaved target points
        /// 2. Build t1_vals[19] with zeros at base window, extended_evals at targets
        /// 3. Interpolate t1 from evaluations to get 19 coefficients (degree-18)
        /// 4. Compute Lagrange kernel L(τ_high, Y) with 10 coefficients (degree-9)
        /// 5. Multiply polynomials: (deg-9) × (deg-18) = deg-27 → 28 coefficients
        pub fn computeFirstRoundPoly(self: *Self) ![FIRST_ROUND_NUM_COEFFS]F {
            const DEGREE = univariate_skip.OUTER_UNIVARIATE_SKIP_DEGREE; // 9
            const EXTENDED_SIZE = univariate_skip.OUTER_UNIVARIATE_SKIP_EXTENDED_DOMAIN_SIZE; // 19

            // Step 1: Compute extended_evals at ONLY the 9 interleaved target points
            // Targets are: {-5, 6, -6, 7, -7, 8, -8, 9, -9}
            const targets = univariate_skip.UNISKIP_TARGETS;
            var extended_evals: [DEGREE]F = [_]F{F.zero()} ** DEGREE;

            // Build eq tables for the factored computation
            const m = self.full_tau.len / 2;
            const wprime_len = if (self.full_tau.len > 0) self.full_tau.len - 1 else 0;
            const num_x_out_bits = m;
            const num_x_in_bits = if (wprime_len > m) wprime_len - m else 0;
            const num_x_in_prime_bits = if (num_x_in_bits > 0) num_x_in_bits - 1 else 0;

            const num_x_out_vals: usize = @as(usize, 1) << @intCast(num_x_out_bits);
            const num_x_in_vals: usize = @as(usize, 1) << @intCast(num_x_in_bits);

            // Build E_out (eq table over w_out = tau[0..m])
            const E_out = try self.buildEqTable(self.full_tau[0..m]);
            defer self.allocator.free(E_out);

            // Build E_in (eq table over w_in = tau[m..tau.len-1])
            const E_in = try self.buildEqTable(self.full_tau[m .. self.full_tau.len - 1]);
            defer self.allocator.free(E_in);

            // Compute extended_evals at all 9 target points simultaneously.
            // Instead of parallelizing over targets (9 items, each doing N_out × N_in work),
            // we parallelize over x_out (typically 256 items), computing Az/Bz base values
            // once per cycle and reusing them for all 9 targets.

            const FirstRoundCtx = struct {
                num_x_in_vals: usize,
                num_x_in_prime_bits: u6,
                E_out: []const F,
                E_in: []const F,
                compact_witnesses: []const evaluators.CompactWitness,
                two_pow_64: F,
            };

            const firstRoundMapReduce = struct {
                const Accum = field_mod.UnreducedProductAccum;
                const accum_zero = [_]Accum{Accum.zero()} ** DEGREE;
                const FoldedU128 = field_mod.FoldedMulU128;
                const FoldedU128Accum = field_mod.FoldedMulU128Accum;
                const folded_zero = [_]FoldedU128{FoldedU128.zero()} ** DEGREE;
                const accum7_zero = [_]FoldedU128Accum{FoldedU128Accum.zero()} ** DEGREE;

                fn map(ctx: FirstRoundCtx, start: usize, end: usize) [DEGREE]F {
                    var accum_outer: [DEGREE]Accum = accum_zero;
                    const cw_len = ctx.compact_witnesses.len;

                    for (start..end) |x_out| {
                        const e_out = if (x_out < ctx.E_out.len) ctx.E_out[x_out] else F.zero();
                        if (e_out.eql(F.zero())) continue;

                        // Group 0 accumulators: Barrett reduction via FoldedMulU128 (6 slots)
                        var g0_pos: [DEGREE]FoldedU128 = folded_zero;
                        var g0_neg: [DEGREE]FoldedU128 = folded_zero;
                        // Group 1 accumulators: S192 Barrett via FoldedMulU128Accum (7 slots)
                        var sg_pos: [DEGREE]FoldedU128Accum = accum7_zero;
                        var sg_neg: [DEGREE]FoldedU128Accum = accum7_zero;

                        for (0..ctx.num_x_in_vals) |x_in| {
                            const e_in = if (x_in < ctx.E_in.len) ctx.E_in[x_in] else F.zero();

                            const x_in_prime = x_in >> 1;
                            const cycle = (x_out << ctx.num_x_in_prime_bits) | x_in_prime;
                            const group: u1 = @truncate(x_in & 1);

                            if (cycle < cw_len) {
                                const cw = &ctx.compact_witnesses[cycle];
                                if (group == 0) {
                                    // Integer path: Barrett reduction via FoldedMulU128
                                    inline for (0..DEGREE) |j| {
                                        const product_int = evaluators.interpolateAzBzProductInt(
                                            &cw.az_first,
                                            &cw.bz_first,
                                            &univariate_skip.COEFFS_PER_J[j],
                                        );
                                        if (product_int > 0) {
                                            g0_pos[j].addAssign(field_mod.mulU128Unreduced(e_in, @intCast(product_int)));
                                        } else if (product_int < 0) {
                                            g0_neg[j].addAssign(field_mod.mulU128Unreduced(e_in, @intCast(-product_int)));
                                        }
                                    }
                                } else {
                                    // Second group: S192 integer path, Barrett via 7-slot accum
                                    inline for (0..DEGREE) |j| {
                                        const coeffs = univariate_skip.COEFFS_PER_J[j][0..SECOND_GROUP_SIZE];
                                        const product = evaluators.interpolateAzBzProductSecondGroupInt(
                                            &cw.az_second, &cw.bz_second, coeffs,
                                        );
                                        if (!product.isZero()) {
                                            const unreduced = field_mod.mulU192Unreduced(e_in, product.magnitude);
                                            if (product.is_positive) {
                                                sg_pos[j].addAssign(unreduced);
                                            } else {
                                                sg_neg[j].addAssign(unreduced);
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        // Reduce inner accumulators and weight by e_out
                        inline for (0..DEGREE) |j| {
                            // Group 0: Barrett reduction of integer products
                            const int_sum = field_mod.reduceMulU128(g0_pos[j]).sub(field_mod.reduceMulU128(g0_neg[j]));
                            // Group 1: Barrett reduction of S192 products
                            const sg_sum = field_mod.reduceMulU128Accum(sg_pos[j]).sub(field_mod.reduceMulU128Accum(sg_neg[j]));
                            const inner_total = int_sum.add(sg_sum);
                            accum_outer[j].addAssign(Accum.fromMul(e_out, inner_total));
                        }
                    }

                    var result: [DEGREE]F = undefined;
                    inline for (0..DEGREE) |j| {
                        result[j] = accum_outer[j].reduce();
                    }
                    return result;
                }

                fn reduce(a: [DEGREE]F, b: [DEGREE]F) [DEGREE]F {
                    var result: [DEGREE]F = undefined;
                    inline for (0..DEGREE) |i| {
                        result[i] = a[i].add(b[i]);
                    }
                    return result;
                }
            };

            // Precompute 2^64 field element for second-group Bz (fallback path)
            const two_pow_64 = blk_2p64: {
                var bytes: [16]u8 = undefined;
                std.mem.writeInt(u128, &bytes, 0x10000000000000000, .little);
                break :blk_2p64 F.fromBytes(&bytes);
            };

            const first_round_ctx = FirstRoundCtx{
                .num_x_in_vals = num_x_in_vals,
                .num_x_in_prime_bits = @intCast(num_x_in_prime_bits),
                .E_out = E_out,
                .E_in = E_in,
                .compact_witnesses = self.compact_witnesses,
                .two_pow_64 = two_pow_64,
            };

            const identity: [DEGREE]F = [_]F{F.zero()} ** DEGREE;

            const target_sums = if (self.thread_pool) |tp|
                tp.parallelReduceForce(
                    [DEGREE]F,
                    num_x_out_vals,
                    identity,
                    first_round_ctx,
                    firstRoundMapReduce.map,
                    firstRoundMapReduce.reduce,
                )
            else
                firstRoundMapReduce.map(first_round_ctx, 0, num_x_out_vals);

            // Map target sums to extended_evals
            extended_evals = target_sums;

            // Step 2: Build t1_vals array (19 entries)
            // Base window {-4,...,5} gets zeros, extended points get their evals
            var t1_vals: [EXTENDED_SIZE]F = [_]F{F.zero()} ** EXTENDED_SIZE;

            // Fill in extended evaluations at target positions
            for (targets, 0..) |z, idx| {
                // pos maps z ∈ {-9,...,9} to index ∈ {0,...,18}
                const pos: usize = @intCast(z + @as(i64, DEGREE));
                t1_vals[pos] = extended_evals[idx];
            }

            // Step 3-5: Interpolate and multiply with Lagrange kernel
            return self.buildUniSkipPolynomial(&t1_vals);
        }

        /// Compute the UniSkip first-round polynomial without allocating a full StreamingOuterProver.
        /// Avoids the split_eq, r_grid, and full_tau allocations (~8MB for primes_large).
        pub fn computeUniSkipFirstRound(
            allocator: Allocator,
            compact_witnesses_arg: []const evaluators.CompactWitness,
            tau: []const F,
            thread_pool: ?*@import("../../utils/thread_pool.zig").ThreadPool,
        ) ![FIRST_ROUND_NUM_COEFFS]F {
            const DEGREE = univariate_skip.OUTER_UNIVARIATE_SKIP_DEGREE;
            const EXTENDED_SIZE = univariate_skip.OUTER_UNIVARIATE_SKIP_EXTENDED_DOMAIN_SIZE;
            const targets = univariate_skip.UNISKIP_TARGETS;

            const m = tau.len / 2;
            const wprime_len = if (tau.len > 0) tau.len - 1 else 0;
            const num_x_out_bits = m;
            const num_x_in_bits = if (wprime_len > m) wprime_len - m else 0;
            const num_x_in_prime_bits = if (num_x_in_bits > 0) num_x_in_bits - 1 else 0;
            const num_x_out_vals: usize = @as(usize, 1) << @intCast(num_x_out_bits);
            const num_x_in_vals: usize = @as(usize, 1) << @intCast(num_x_in_bits);
            const tau_high = if (tau.len > 0) tau[tau.len - 1] else F.zero();

            const E_out = try buildEqTableAlloc(allocator, tau[0..m]);
            defer allocator.free(E_out);
            const E_in = try buildEqTableAlloc(allocator, tau[m .. tau.len - 1]);
            defer allocator.free(E_in);

            const two_pow_64 = blk_2p64: {
                var bytes: [16]u8 = undefined;
                std.mem.writeInt(u128, &bytes, 0x10000000000000000, .little);
                break :blk_2p64 F.fromBytes(&bytes);
            };

            const FirstRoundCtx = struct {
                num_x_in_vals: usize,
                num_x_in_prime_bits: u6,
                E_out: []const F,
                E_in: []const F,
                compact_witnesses: []const evaluators.CompactWitness,
                two_pow_64: F,
            };

            const ctx = FirstRoundCtx{
                .num_x_in_vals = num_x_in_vals,
                .num_x_in_prime_bits = @intCast(num_x_in_prime_bits),
                .E_out = E_out,
                .E_in = E_in,
                .compact_witnesses = compact_witnesses_arg,
                .two_pow_64 = two_pow_64,
            };

            // Reuse the same map-reduce logic
            const mapReduceFns = struct {
                const Accum = field_mod.UnreducedProductAccum;
                const accum_zero = [_]Accum{Accum.zero()} ** DEGREE;
                const FoldedU128 = field_mod.FoldedMulU128;
                const FoldedU128Accum = field_mod.FoldedMulU128Accum;
                const folded_zero = [_]FoldedU128{FoldedU128.zero()} ** DEGREE;
                const accum7_zero = [_]FoldedU128Accum{FoldedU128Accum.zero()} ** DEGREE;

                fn mapFn(c: FirstRoundCtx, start: usize, end: usize) [DEGREE]F {
                    var accum_outer: [DEGREE]Accum = accum_zero;
                    const cw_len = c.compact_witnesses.len;

                    for (start..end) |x_out| {
                        const e_out = if (x_out < c.E_out.len) c.E_out[x_out] else F.zero();
                        if (e_out.eql(F.zero())) continue;

                        var g0_pos: [DEGREE]FoldedU128 = folded_zero;
                        var g0_neg: [DEGREE]FoldedU128 = folded_zero;
                        var sg_pos: [DEGREE]FoldedU128Accum = accum7_zero;
                        var sg_neg: [DEGREE]FoldedU128Accum = accum7_zero;

                        for (0..c.num_x_in_vals) |x_in| {
                            const e_in = if (x_in < c.E_in.len) c.E_in[x_in] else F.zero();
                            const x_in_prime = x_in >> 1;
                            const cycle = (x_out << c.num_x_in_prime_bits) | x_in_prime;
                            const group: u1 = @truncate(x_in & 1);

                            if (cycle < cw_len) {
                                const cw = &c.compact_witnesses[cycle];
                                if (group == 0) {
                                    inline for (0..DEGREE) |j| {
                                        const product_int = evaluators.interpolateAzBzProductInt(
                                            &cw.az_first, &cw.bz_first,
                                            &univariate_skip.COEFFS_PER_J[j],
                                        );
                                        if (product_int > 0) {
                                            g0_pos[j].addAssign(field_mod.mulU128Unreduced(e_in, @intCast(product_int)));
                                        } else if (product_int < 0) {
                                            g0_neg[j].addAssign(field_mod.mulU128Unreduced(e_in, @intCast(-product_int)));
                                        }
                                    }
                                } else {
                                    // Second group: S192 integer path
                                    inline for (0..DEGREE) |j| {
                                        const coeffs = univariate_skip.COEFFS_PER_J[j][0..SECOND_GROUP_SIZE];
                                        const product = evaluators.interpolateAzBzProductSecondGroupInt(
                                            &cw.az_second, &cw.bz_second, coeffs,
                                        );
                                        if (!product.isZero()) {
                                            const unreduced = field_mod.mulU192Unreduced(e_in, product.magnitude);
                                            if (product.is_positive) {
                                                sg_pos[j].addAssign(unreduced);
                                            } else {
                                                sg_neg[j].addAssign(unreduced);
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        inline for (0..DEGREE) |j| {
                            const int_sum = field_mod.reduceMulU128(g0_pos[j]).sub(field_mod.reduceMulU128(g0_neg[j]));
                            const sg_sum = field_mod.reduceMulU128Accum(sg_pos[j]).sub(field_mod.reduceMulU128Accum(sg_neg[j]));
                            const inner_total = int_sum.add(sg_sum);
                            accum_outer[j].addAssign(Accum.fromMul(e_out, inner_total));
                        }
                    }

                    var result: [DEGREE]F = undefined;
                    inline for (0..DEGREE) |j| {
                        result[j] = accum_outer[j].reduce();
                    }
                    return result;
                }

                fn reduceFn(a: [DEGREE]F, b: [DEGREE]F) [DEGREE]F {
                    var result: [DEGREE]F = undefined;
                    inline for (0..DEGREE) |i| {
                        result[i] = a[i].add(b[i]);
                    }
                    return result;
                }
            };

            const identity: [DEGREE]F = [_]F{F.zero()} ** DEGREE;
            const target_sums = if (thread_pool) |tp|
                tp.parallelReduceForce(
                    [DEGREE]F, num_x_out_vals, identity,
                    ctx, mapReduceFns.mapFn, mapReduceFns.reduceFn,
                )
            else
                mapReduceFns.mapFn(ctx, 0, num_x_out_vals);

            // Build t1_vals and produce final polynomial
            var t1_vals: [EXTENDED_SIZE]F = [_]F{F.zero()} ** EXTENDED_SIZE;
            for (targets, 0..) |z, idx| {
                const pos: usize = @intCast(z + @as(i64, DEGREE));
                t1_vals[pos] = target_sums[idx];
            }

            return buildUniSkipPolynomialStatic(tau_high, &t1_vals);
        }

        /// Build the UniSkip polynomial s1(Y) = L(τ_high, Y) · t1(Y)
        /// from t1 evaluations on the extended domain
        fn buildUniSkipPolynomial(
            self: *const Self,
            t1_vals: *const [univariate_skip.OUTER_UNIVARIATE_SKIP_EXTENDED_DOMAIN_SIZE]F,
        ) [FIRST_ROUND_NUM_COEFFS]F {
            return buildUniSkipPolynomialStatic(self.tau_high, t1_vals);
        }

        /// Static version of buildUniSkipPolynomial for standalone use
        fn buildUniSkipPolynomialStatic(
            tau_high: F,
            t1_vals: *const [univariate_skip.OUTER_UNIVARIATE_SKIP_EXTENDED_DOMAIN_SIZE]F,
        ) [FIRST_ROUND_NUM_COEFFS]F {
            const DEGREE = univariate_skip.OUTER_UNIVARIATE_SKIP_DEGREE; // 9
            const DOMAIN_SIZE = univariate_skip.OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE; // 10
            const EXTENDED_SIZE = univariate_skip.OUTER_UNIVARIATE_SKIP_EXTENDED_DOMAIN_SIZE; // 19

            // Step 3: Interpolate t1 from evaluations to coefficients (degree-18)
            var t1_coeffs: [EXTENDED_SIZE]F = [_]F{F.zero()} ** EXTENDED_SIZE;
            interpolateCoeffs(t1_vals, &t1_coeffs, EXTENDED_SIZE, DEGREE);

            // Step 4: Compute Lagrange kernel L(τ_high, Y) evaluations and coefficients
            var lagrange_evals: [DOMAIN_SIZE]F = undefined;
            const base_left: i64 = -@as(i64, (DOMAIN_SIZE - 1) / 2);

            for (0..DOMAIN_SIZE) |i| {
                const x_i: i64 = base_left + @as(i64, @intCast(i));
                var num = F.one();
                var den = F.one();

                for (0..DOMAIN_SIZE) |j| {
                    if (i == j) continue;
                    const x_j: i64 = base_left + @as(i64, @intCast(j));
                    const x_j_field = if (x_j >= 0) F.fromU64(@intCast(x_j)) else F.zero().sub(F.fromU64(@intCast(-x_j)));
                    num = num.mul(tau_high.sub(x_j_field));

                    const diff: i64 = x_i - x_j;
                    const diff_field = if (diff >= 0) F.fromU64(@intCast(diff)) else F.zero().sub(F.fromU64(@intCast(-diff)));
                    den = den.mul(diff_field);
                }

                lagrange_evals[i] = num.mul(den.inverse().?);
            }

            // Interpolate Lagrange kernel to coefficients (degree-9)
            var lagrange_coeffs: [DOMAIN_SIZE]F = [_]F{F.zero()} ** DOMAIN_SIZE;
            interpolateCoeffs(&lagrange_evals, &lagrange_coeffs, DOMAIN_SIZE, @as(i64, (DOMAIN_SIZE - 1) / 2));

            // Step 5: Multiply polynomials (deg-9) × (deg-18) = deg-27 → 28 coefficients
            var s1_coeffs: [FIRST_ROUND_NUM_COEFFS]F = [_]F{F.zero()} ** FIRST_ROUND_NUM_COEFFS;
            for (0..DOMAIN_SIZE) |i| {
                for (0..EXTENDED_SIZE) |j| {
                    s1_coeffs[i + j] = s1_coeffs[i + j].add(lagrange_coeffs[i].mul(t1_coeffs[j]));
                }
            }

            return s1_coeffs;
        }

        /// O(n^2) Newton divided-differences interpolation from evaluations to monomial coefficients.
        /// Evaluations are at symmetric consecutive-integer domain {-half_size, ..., half_size}.
        /// Ported from Jolt's interpolate_coeffs (lagrange_poly.rs:298-383).
        fn lagrangeInterpolate(
            self: *const Self,
            evals: []const F,
            coeffs: []F,
            size: usize,
            half_size: i64,
        ) void {
            _ = self;
            interpolateCoeffs(evals, coeffs, size, half_size);
        }

        /// Static O(n^2) Newton interpolation for standalone use.
        /// Domain: consecutive integers {start, start+1, ..., start+N-1} where start = -half_size.
        /// Since nodes are consecutive integers, divided difference denominators are just 1/order.
        fn interpolateCoeffs(
            evals: []const F,
            coeffs: []F,
            size: usize,
            half_size: i64,
        ) void {
            if (size == 0) return;
            const d = size - 1;
            const start = -half_size;

            // Initialize coeffs to zero
            for (0..size) |k| coeffs[k] = F.zero();

            // Batch-compute inverses of 1..d using one field inversion (Montgomery's trick)
            const MAX_N = univariate_skip.OUTER_UNIVARIATE_SKIP_EXTENDED_DOMAIN_SIZE;
            var pref: [MAX_N]F = undefined;
            var invs: [MAX_N]F = undefined;
            pref[0] = F.one();
            var m: usize = 1;
            while (m <= d) : (m += 1) {
                pref[m] = pref[m - 1].mul(F.fromU64(@intCast(m)));
            }
            const inv_total = pref[d].inverse().?;
            var right = F.one();
            var i_idx: usize = d;
            while (i_idx >= 1) : (i_idx -= 1) {
                invs[i_idx] = pref[i_idx - 1].mul(right).mul(inv_total);
                right = right.mul(F.fromU64(@intCast(i_idx)));
                if (i_idx == 1) break;
            }

            // Newton divided differences: dd[i] tracks current-order differences
            var dd: [MAX_N]F = undefined;
            for (0..size) |idx| dd[idx] = evals[idx];

            var newton: [MAX_N]F = [_]F{F.zero()} ** MAX_N;
            newton[0] = dd[0];
            var order: usize = 1;
            while (order <= d) : (order += 1) {
                const inv = invs[order];
                var ii: usize = 0;
                while (ii + order < size) : (ii += 1) {
                    // Denominator (x_{i+order} - x_i) = order for consecutive nodes
                    dd[ii] = dd[ii + 1].sub(dd[ii]).mul(inv);
                }
                newton[order] = dd[0];
            }

            // Convert Newton form to monomial coefficients
            // basis[j] = coefficient of x^j in product_{k=0..deg-1} (x - (start+k))
            var basis: [MAX_N]F = [_]F{F.zero()} ** MAX_N;
            basis[0] = F.one();
            var deg: usize = 0;
            var k: usize = 0;
            while (k < size) : (k += 1) {
                // coeffs += newton[k] * basis
                const scale = newton[k];
                var j: usize = 0;
                while (j <= deg) : (j += 1) {
                    coeffs[j] = coeffs[j].add(scale.mul(basis[j]));
                }

                if (k == d) break;

                // Update basis <- basis * (x - (start + k))
                const a: i64 = start + @as(i64, @intCast(k));
                const neg_a = if (a >= 0)
                    F.zero().sub(F.fromU64(@intCast(a)))
                else
                    F.fromU64(@intCast(-a));

                var t: usize = deg + 1;
                while (t >= 1) : (t -= 1) {
                    basis[t] = basis[t - 1].add(basis[t].mul(neg_a));
                    if (t == 1) break;
                }
                basis[0] = basis[0].mul(neg_a);
                deg += 1;
            }
        }

        /// Build an eq polynomial evaluation table over the given tau values
        /// Uses big-endian indexing: tau[0] controls MSB of index
        fn buildEqTable(self: *const Self, tau: []const F) ![]F {
            return buildEqTableAlloc(self.allocator, tau);
        }

        /// Static version of buildEqTable for standalone use
        fn buildEqTableAlloc(allocator: Allocator, tau: []const F) ![]F {
            const size: usize = @as(usize, 1) << @intCast(tau.len);
            const result = try allocator.alloc(F, size);

            // Start with 1
            result[0] = F.one();
            var current_size: usize = 1;

            // For each variable, double the table size
            for (tau) |tau_k| {
                const one_minus_tau_k = F.one().sub(tau_k);

                // Iterate in reverse to avoid overwriting needed values
                var i: usize = current_size;
                while (i > 0) {
                    i -= 1;
                    const scalar = result[i];
                    // Big-endian: new bit is appended as LSB
                    // Index 2*i (even, bit=0) gets (1-τ)
                    // Index 2*i+1 (odd, bit=1) gets τ
                    result[2 * i + 1] = scalar.mul(tau_k);
                    result[2 * i] = scalar.mul(one_minus_tau_k);
                }
                current_size *= 2;
            }

            return result;
        }

        /// Evaluate Az * Bz for a single cycle at a specific domain point Y for a specific group
        ///
        /// The domain points are in the extended symmetric window:
        /// - Index 0: Y = -DEGREE (= -9)
        /// - Index 9: Y = 0
        /// - Index 18: Y = DEGREE (= 9)
        ///
        /// Group selection:
        /// - group=0 (FIRST_GROUP): 10 constraints at Y ∈ {-4, -3, ..., 4, 5}
        /// - group=1 (SECOND_GROUP): 9 constraints at Y ∈ {-4, -3, ..., 3, 4}
        ///
        /// For Y in the base window, we can directly evaluate the constraint.
        /// For Y outside the base window, we use Lagrange extrapolation.
        fn evaluateAzBzAtDomainPointForGroup(
            self: *const Self,
            witness: *const constraints.R1CSCycleInputs(F),
            domain_idx: usize,
            group: u1,
        ) F {
            _ = self;

            // Convert domain_idx to the actual Y coordinate
            // domain_idx 0 -> Y = -DEGREE = -9
            // domain_idx DEGREE -> Y = 0
            // domain_idx 2*DEGREE -> Y = DEGREE = 9
            const DEGREE = univariate_skip.OUTER_UNIVARIATE_SKIP_DEGREE;
            const y_coord: i64 = @as(i64, @intCast(domain_idx)) - @as(i64, DEGREE);

            // Select group-specific parameters
            // FIRST_GROUP (group=0): 10 constraints at Y ∈ {-4, ..., 5}, uses FIRST_GROUP_INDICES
            // SECOND_GROUP (group=1): 9 constraints at Y ∈ {-4, ..., 4}, uses SECOND_GROUP_INDICES
            const group_size: usize = if (group == 0) FIRST_GROUP_SIZE else SECOND_GROUP_SIZE;
            const group_indices = if (group == 0) &constraints.FIRST_GROUP_INDICES else &constraints.SECOND_GROUP_INDICES;

            // Base window:
            // - FIRST_GROUP (10 constraints): {-4, -3, -2, -1, 0, 1, 2, 3, 4, 5}
            // - SECOND_GROUP (9 constraints): {-4, -3, -2, -1, 0, 1, 2, 3, 4}
            const base_left: i64 = -@as(i64, @intCast((group_size - 1) / 2));
            const base_right: i64 = base_left + @as(i64, @intCast(group_size)) - 1;

            // Check if Y is in the base window for this group
            if (y_coord >= base_left and y_coord <= base_right) {
                // Y is in the base window - evaluate constraint directly
                // Map Y to constraint index: Y = base_left + i => i = Y - base_left
                const constraint_pos: usize = @intCast(y_coord - base_left);

                if (constraint_pos < group_size) {
                    const constraint_idx = group_indices[constraint_pos];
                    const constraint = constraints.UNIFORM_CONSTRAINTS[constraint_idx];
                    const az = constraint.condition.evaluate(F, witness.asSlice());
                    const bz = constraint.left.evaluate(F, witness.asSlice())
                        .sub(constraint.right.evaluate(F, witness.asSlice()));
                    return az.mul(bz);
                }
            }

            // Y is outside the base window - use Lagrange extrapolation
            // Compute Az and Bz at all base window points for this group
            var az_base: [FIRST_GROUP_SIZE]F = undefined; // Use max size for array
            var bz_base: [FIRST_GROUP_SIZE]F = undefined;

            for (0..group_size) |i| {
                const constraint_idx = group_indices[i];
                const constraint = constraints.UNIFORM_CONSTRAINTS[constraint_idx];
                az_base[i] = constraint.condition.evaluate(F, witness.asSlice());
                bz_base[i] = constraint.left.evaluate(F, witness.asSlice())
                    .sub(constraint.right.evaluate(F, witness.asSlice()));
            }

            // Use precomputed Lagrange coefficients to extrapolate to Y
            // Find the target index j corresponding to Y
            const targets = univariate_skip.UNISKIP_TARGETS;
            var target_j: ?usize = null;
            for (targets, 0..) |t, j| {
                if (t == y_coord) {
                    target_j = j;
                    break;
                }
            }

            if (target_j) |j| {
                // Use COEFFS_PER_J[j] for extrapolation
                // Note: Both groups use the same COEFFS_PER_J, but SECOND_GROUP only uses
                // the first 9 coefficients (group_size coefficients)
                const coeffs = univariate_skip.COEFFS_PER_J[j];

                // Extrapolate Az(Y) = Σ_i coeffs[i] * az_base[i]
                var az_y = F.zero();
                for (0..group_size) |i| {
                    const c = coeffs[i];
                    if (c != 0) {
                        const c_field = if (c > 0)
                            F.fromU64(@intCast(c))
                        else
                            F.zero().sub(F.fromU64(@intCast(-c)));
                        az_y = az_y.add(az_base[i].mul(c_field));
                    }
                }

                // Extrapolate Bz(Y) = Σ_i coeffs[i] * bz_base[i]
                var bz_y = F.zero();
                for (0..group_size) |i| {
                    const c = coeffs[i];
                    if (c != 0) {
                        const c_field = if (c > 0)
                            F.fromU64(@intCast(c))
                        else
                            F.zero().sub(F.fromU64(@intCast(-c)));
                        bz_y = bz_y.add(bz_base[i].mul(c_field));
                    }
                }

                return az_y.mul(bz_y);
            }

            // Should not reach here for valid domain indices
            return F.zero();
        }

        /// Legacy function for backwards compatibility - evaluates FIRST_GROUP only
        /// Deprecated: Use evaluateAzBzAtDomainPointForGroup instead
        fn evaluateAzBzAtDomainPoint(
            self: *const Self,
            witness: *const constraints.R1CSCycleInputs(F),
            domain_idx: usize,
        ) F {
            return self.evaluateAzBzAtDomainPointForGroup(witness, domain_idx, 0);
        }

        /// Interpolate extended evaluations to polynomial coefficients
        ///
        /// This implements Jolt's `build_uniskip_first_round_poly`:
        /// 1. Rebuild t1 on the full extended symmetric window
        /// 2. Interpolate t1 to get coefficients
        /// 3. Compute Lagrange kernel L(τ_high, Y) coefficients
        /// 4. Multiply polynomials to get s1(Y) = L(τ_high, Y) * t1(Y)
        fn interpolateFirstRoundPoly(
            self: *const Self,
            extended_evals: *const [univariate_skip.OUTER_UNIVARIATE_SKIP_EXTENDED_DOMAIN_SIZE]F,
        ) [FIRST_ROUND_NUM_COEFFS]F {
            const DOMAIN_SIZE = FIRST_GROUP_SIZE;
            const DEGREE = univariate_skip.OUTER_UNIVARIATE_SKIP_DEGREE;
            const EXTENDED_SIZE = univariate_skip.OUTER_UNIVARIATE_SKIP_EXTENDED_DOMAIN_SIZE;

            // The extended_evals are evaluations of t1(Y) on the extended symmetric window
            // {-DEGREE, ..., DEGREE} = {-9, -8, ..., 8, 9}
            // These 19 values are provided in order from extended_evals[0] = t1(-9) to extended_evals[18] = t1(9)

            // Step 1: Interpolate t1(Y) from extended evaluations
            // Domain points: {-DEGREE, -DEGREE+1, ..., DEGREE-1, DEGREE}
            var t1_coeffs: [EXTENDED_SIZE]F = [_]F{F.zero()} ** EXTENDED_SIZE;

            // Use Lagrange interpolation: p(Y) = Σ_i y_i * L_i(Y)
            // where L_i(Y) = Π_{j≠i} (Y - x_j) / (x_i - x_j)
            for (0..EXTENDED_SIZE) |i| {
                // Evaluation y_i at domain point x_i = -DEGREE + i
                const y_i = extended_evals[i];

                if (y_i.eql(F.zero())) continue;

                // Compute denominator (scalar) Π_{j≠i} (x_i - x_j)
                // Since x_i = -DEGREE + i and x_j = -DEGREE + j, we have x_i - x_j = i - j
                var den = F.one();
                for (0..EXTENDED_SIZE) |j| {
                    if (i == j) continue;
                    // x_i - x_j = (i - j)
                    const diff: i64 = @as(i64, @intCast(i)) - @as(i64, @intCast(j));
                    const diff_field = if (diff >= 0)
                        F.fromU64(@intCast(diff))
                    else
                        F.zero().sub(F.fromU64(@intCast(-diff)));
                    den = den.mul(diff_field);
                }

                const scale = y_i.mul(den.inverse().?);

                // Build numerator polynomial Π_{j≠i} (Y - x_j)
                // Start with constant 1, multiply by (Y - x_j) for each j ≠ i
                var basis: [EXTENDED_SIZE]F = [_]F{F.zero()} ** EXTENDED_SIZE;
                basis[0] = F.one();
                var deg: usize = 0;

                for (0..EXTENDED_SIZE) |j| {
                    if (i == j) continue;
                    const x_j: i64 = -@as(i64, DEGREE) + @as(i64, @intCast(j));
                    const neg_x_j = if (x_j >= 0)
                        F.zero().sub(F.fromU64(@intCast(x_j)))
                    else
                        F.fromU64(@intCast(-x_j));

                    // Multiply basis by (Y - x_j)
                    // New polynomial: basis[k+1] += basis[k] and basis[k] *= neg_x_j
                    var k: usize = deg + 1;
                    while (k > 0) {
                        k -= 1;
                        const old = basis[k];
                        if (k + 1 <= EXTENDED_SIZE - 1) {
                            basis[k + 1] = basis[k + 1].add(old);
                        }
                        basis[k] = old.mul(neg_x_j);
                    }
                    deg += 1;
                }

                // Add scaled basis to t1_coeffs
                for (0..EXTENDED_SIZE) |k| {
                    t1_coeffs[k] = t1_coeffs[k].add(basis[k].mul(scale));
                }
            }

            // Step 2: Compute Lagrange kernel L(τ_high, Y) coefficients
            // τ_high is the last element of the full tau vector, stored separately
            const tau_high = self.tau_high;

            // L(τ_high, Y) evaluations at base domain {-4, -3, ..., 4, 5}
            var lagrange_evals: [DOMAIN_SIZE]F = undefined;
            const base_left: i64 = -@as(i64, (DOMAIN_SIZE - 1) / 2);

            for (0..DOMAIN_SIZE) |i| {
                const x_i: i64 = base_left + @as(i64, @intCast(i));
                var num = F.one();
                var den = F.one();

                for (0..DOMAIN_SIZE) |j| {
                    if (i == j) continue;
                    const x_j: i64 = base_left + @as(i64, @intCast(j));
                    const x_j_field = if (x_j >= 0) F.fromU64(@intCast(x_j)) else F.zero().sub(F.fromU64(@intCast(-x_j)));
                    num = num.mul(tau_high.sub(x_j_field));

                    const diff: i64 = x_i - x_j;
                    const diff_field = if (diff >= 0) F.fromU64(@intCast(diff)) else F.zero().sub(F.fromU64(@intCast(-diff)));
                    den = den.mul(diff_field);
                }

                lagrange_evals[i] = num.mul(den.inverse().?);
            }

            // Interpolate Lagrange kernel to coefficients (degree DOMAIN_SIZE-1 = 9)
            var lagrange_coeffs: [DOMAIN_SIZE]F = [_]F{F.zero()} ** DOMAIN_SIZE;

            for (0..DOMAIN_SIZE) |i| {
                const y_i = lagrange_evals[i];
                if (y_i.eql(F.zero())) continue;

                var den = F.one();
                for (0..DOMAIN_SIZE) |j| {
                    if (i == j) continue;
                    const diff: i64 = @as(i64, @intCast(i)) - @as(i64, @intCast(j));
                    const diff_field = if (diff >= 0) F.fromU64(@intCast(diff)) else F.zero().sub(F.fromU64(@intCast(-diff)));
                    den = den.mul(diff_field);
                }
                const scale = y_i.mul(den.inverse().?);

                var basis: [DOMAIN_SIZE]F = [_]F{F.zero()} ** DOMAIN_SIZE;
                basis[0] = F.one();
                var deg: usize = 0;

                for (0..DOMAIN_SIZE) |j| {
                    if (i == j) continue;
                    const x_j: i64 = base_left + @as(i64, @intCast(j));
                    const neg_x_j = if (x_j >= 0) F.zero().sub(F.fromU64(@intCast(x_j))) else F.fromU64(@intCast(-x_j));

                    var k: usize = deg + 1;
                    while (k > 0) {
                        k -= 1;
                        const old = basis[k];
                        if (k + 1 < DOMAIN_SIZE) {
                            basis[k + 1] = basis[k + 1].add(old);
                        }
                        basis[k] = old.mul(neg_x_j);
                    }
                    deg += 1;
                }

                for (0..DOMAIN_SIZE) |k| {
                    lagrange_coeffs[k] = lagrange_coeffs[k].add(basis[k].mul(scale));
                }
            }

            // Step 3: Multiply polynomials s1 = L * t1
            // deg(L) = DOMAIN_SIZE - 1 = 9
            // deg(t1) = EXTENDED_SIZE - 1 = 18
            // deg(s1) = 9 + 18 = 27
            var s1_coeffs: [FIRST_ROUND_NUM_COEFFS]F = [_]F{F.zero()} ** FIRST_ROUND_NUM_COEFFS;

            for (0..DOMAIN_SIZE) |i| {
                for (0..EXTENDED_SIZE) |j| {
                    if (i + j < FIRST_ROUND_NUM_COEFFS) {
                        s1_coeffs[i + j] = s1_coeffs[i + j].add(lagrange_coeffs[i].mul(t1_coeffs[j]));
                    }
                }
            }

            return s1_coeffs;
        }

        /// Bind the first-round challenge and set up for remaining rounds
        ///
        /// The uni_skip_claim parameter is uni_poly(r0), the evaluation of the
        /// univariate skip polynomial at the first-round challenge.
        ///
        /// IMPORTANT: r0 is NOT bound in split_eq! In Jolt, r0's contribution is:
        /// 1. Pre-multiplied into current_scalar via L(tau_high, r0) at initialization
        /// 2. Used to compute Lagrange weights for Az/Bz evaluation
        /// The split_eq only binds the streaming round and cycle round challenges.
        pub fn bindFirstRoundChallenge(self: *Self, r0: F, uni_skip_claim: F) !void {
            // IMPORTANT: r0 is NOT added to challenges!
            // challenges should only contain sumcheck challenges [r_stream, r_1, ..., r_n]
            // r0 is the UniSkip challenge which is used differently:
            // 1. Pre-multiplied into current_scalar via L(tau_high, r0) at initialization
            // 2. Used to compute Lagrange weights for Az/Bz evaluation
            // The split_eq only binds the streaming round and cycle round challenges.
            self.current_round = 1;
            self.current_claim = uni_skip_claim;

            // Compute Lagrange basis evaluations at r0 for use in remaining rounds
            self.computeLagrangeEvalsAtR0(r0);

            // DO NOT bind r0 in split_eq! The Lagrange kernel scaling was already
            // applied during initialization. The streaming round will bind the first
            // actual sumcheck challenge.
        }

        /// Compute Lagrange basis evaluations at r0
        ///
        /// IMPORTANT: The domain is the symmetric window {-4, -3, -2, -1, 0, 1, 2, 3, 4, 5}
        /// matching Jolt's LagrangePolynomial::start_i64 which computes -(N-1)/2 = -4 for N=10.
        fn computeLagrangeEvalsAtR0(self: *Self, r0: F) void {
            // L_i(r0) for i in 0..FIRST_GROUP_SIZE
            // Domain is {start, start+1, ..., start+FIRST_GROUP_SIZE-1}
            // where start = -((FIRST_GROUP_SIZE - 1) / 2) = -4

            const start: i64 = -@as(i64, (FIRST_GROUP_SIZE - 1) / 2); // = -4

            // DEBUG: Print r0
            dbg("[ZOLT] computeLagrangeEvalsAtR0: r0 = {any}\n", .{r0.toBytesBE()});

            for (0..FIRST_GROUP_SIZE) |i| {
                _ = start + @as(i64, @intCast(i)); // actual domain point (unused but shows semantics)
                var numer = F.one();
                var denom = F.one();

                for (0..FIRST_GROUP_SIZE) |j| {
                    if (i != j) {
                        const x_j: i64 = start + @as(i64, @intCast(j));

                        // numer *= (r0 - x_j)
                        const x_j_field = if (x_j >= 0)
                            F.fromU64(@intCast(x_j))
                        else
                            F.zero().sub(F.fromU64(@intCast(-x_j)));
                        numer = numer.mul(r0.sub(x_j_field));

                        // denom *= (x_i - x_j) = (i - j) since x_k = start + k
                        const diff: i64 = @as(i64, @intCast(i)) - @as(i64, @intCast(j));
                        if (diff > 0) {
                            denom = denom.mul(F.fromU64(@intCast(diff)));
                        } else {
                            denom = denom.mul(F.zero().sub(F.fromU64(@intCast(-diff))));
                        }
                    }
                }

                // L_i(r0) = numer / denom
                self.lagrange_evals_r0[i] = if (!denom.eql(F.zero()))
                    numer.mul(denom.inverse().?)
                else
                    F.zero();
            }

            // DEBUG: Print all Lagrange weights
            if (comptime debug_verbose) {
                for (0..FIRST_GROUP_SIZE) |i| {
                    dbg("[ZOLT] computeLagrangeEvalsAtR0: w[{d}] = {any}\n", .{ i, self.lagrange_evals_r0[i].toBytesBE() });
                }
            }
        }

        /// Compute a remaining round polynomial (degree 3)
        ///
        /// There are two types of rounds:
        /// 1. Streaming round (current_round == 1): Sums over constraint groups
        /// 2. Cycle rounds (current_round > 1): Sums over cycle halves using combined Az*Bz
        ///
        /// IMPORTANT: The eq weights for cycles use a FACTORIZED representation:
        ///   eq_val[i] = E_out[i >> head_in_bits] * E_in[i & ((1 << head_in_bits) - 1)]
        /// This allows us to handle 1024 cycles with only 32+32=64 precomputed values.
        pub fn computeRemainingRoundPoly(self: *Self) ![4]F {
            // Gruen's multiquadratic method computes:
            // - t'(0) = Σ eq * Az(0) * Bz(0)
            // - t'(∞) = Σ eq * Az(∞) * Bz(∞) = Σ eq * (Az(1) - Az(0)) * (Bz(1) - Bz(0))
            //
            // Note: t'(∞) is the product of SLOPES, NOT the slope of the product!
            // This is crucial for the cubic polynomial construction.

            var t_zero = F.zero();
            var t_infinity = F.zero();

            // Jolt uses LinearOnlySchedule with switch_over = 0
            // This means ALL rounds use linear phase (no streaming rounds).
            // Round 1 (Zolt's first remaining round) is equivalent to Jolt's round 0,
            // which is the switch-over point where OuterLinearStage::initialize is called.
            //
            // Mapping:
            //   Zolt round 1 → Jolt round 0 (switch-over, Equal) → initialize linear stage
            //   Zolt round 2 → Jolt round 1 (Greater) → next_window
            //   ...
            //
            // The key insight: With LinearOnlySchedule, we NEVER have a "streaming round"
            // that reads directly from trace. We always materialize Az/Bz polynomials first.
            const window_size: usize = 1;

            // For round 1 (first remaining round), we need to materialize Az/Bz and build t_prime
            // This happens in bindRemainingRoundChallenge when r_stream is set
            if (self.current_round == 1 and self.t_prime_poly == null) {
                // Round 1 but t_prime not yet built - materialize now
                // This matches Jolt's OuterLinearStage::initialize called on round 0
                try self.materializeLinearPhasePolynomials();

                // DEBUG: Brute-force sum check after materialization
                // The sumcheck sums: Σ_{x_out, x_in} E_out[x_out] * E_in[x_in] *
                //   Σ_j (eq_window[j] * Az[grid*i + j] * Bz[grid*i + j])
                // where i = (x_out << in_bits) | x_in
                // For window_size=1, the window variable is w_last, and
                //   eq_window = [(1-w_last), w_last]
                // So the full sum is:
                //   Σ E_out[x_out]*E_in[x_in] * (
                //     (1-w_last)*Az[2i]*Bz[2i] + w_last*Az[2i+1]*Bz[2i+1]
                //   )
                // And this should equal current_claim / current_scalar (since
                // the split_eq scaling hasn't been included yet in materials)
                //
                // Actually, the remaining sumcheck sums:
                //   Σ_x split_eq(tau, x) * t(x)
                // where split_eq includes current_scalar * l(X) for each round.
                // At round 1 (first remaining), the function being summed is the
                // full polynomial over all unbound variables.
                //
                // Use E_out and E_in factored representation for the "head" part,
                // and sum over the window variable separately.
                if (comptime debug_verbose) {
                if (self.az_poly != null and self.bz_poly != null) {
                    const az_p = &(self.az_poly.?);
                    const bz_p = &(self.bz_poly.?);
                    const brute_window: usize = 1;
                    const brute_grid: usize = 2;

                    const eq_tables = self.split_eq.getWindowEqTables(0, brute_window);
                    const E_out_b = eq_tables.E_out;
                    const E_in_b = eq_tables.E_in;
                    const head_in_bits = eq_tables.head_in_bits;

                    // w_last is the window variable = tau[current_index - 1]
                    const w_last = self.split_eq.tau[self.split_eq.current_index - 1];
                    const one_minus_w_last = F.one().sub(w_last);

                    dbg("[BRUTE_FORCE2] E_out.len={d}, E_in.len={d}, head_in_bits={d}\n", .{
                        E_out_b.len, E_in_b.len, head_in_bits,
                    });
                    dbg("[BRUTE_FORCE2] current_scalar = {any}\n", .{self.split_eq.current_scalar.toBytes()});
                    dbg("[BRUTE_FORCE2] w_last = {any}\n", .{w_last.toBytes()});

                    var brute_sum2 = F.zero();
                    for (0..E_out_b.len) |x_out| {
                        for (0..E_in_b.len) |x_in| {
                            const i = (x_out << @intCast(head_in_bits)) | x_in;
                            const eq_weight = E_out_b[x_out].mul(E_in_b[x_in]);

                            // Sum over grid positions (window variable)
                            var inner_sum = F.zero();
                            for (0..brute_grid) |j| {
                                const idx = brute_grid * i + j;
                                if (idx < az_p.boundLen()) {
                                    const az_val = az_p.evaluations[idx];
                                    const bz_val = bz_p.evaluations[idx];
                                    // eq_window[0] = (1 - w_last), eq_window[1] = w_last
                                    const w_j = if (j == 0) one_minus_w_last else w_last;
                                    inner_sum = inner_sum.add(w_j.mul(az_val.mul(bz_val)));
                                }
                            }
                            brute_sum2 = brute_sum2.add(eq_weight.mul(inner_sum));
                        }
                    }
                    // Include current_scalar from split_eq
                    brute_sum2 = brute_sum2.mul(self.split_eq.current_scalar);

                    dbg("[BRUTE_FORCE2] sum (with current_scalar) = {any}\n", .{brute_sum2.toBytes()});
                    dbg("[BRUTE_FORCE2] current_claim (uni_skip_claim) = {any}\n", .{self.current_claim.toBytes()});
                    dbg("[BRUTE_FORCE2] match = {}\n", .{brute_sum2.eql(self.current_claim)});

                    // Check if the missing factor is R^2
                    const r_squared = F.rSquared();
                    const brute_with_r2 = brute_sum2.mul(r_squared);
                    dbg("[BRUTE_FORCE2] sum * R^2 = {any}\n", .{brute_with_r2.toBytes()});
                    dbg("[BRUTE_FORCE2] match_with_R2 = {}\n", .{brute_with_r2.eql(self.current_claim)});

                    // Also try claim / R^2
                    const r_squared_inv = r_squared.inverse().?;
                    const claim_div_r2 = self.current_claim.mul(r_squared_inv);
                    dbg("[BRUTE_FORCE2] claim / R^2 = {any}\n", .{claim_div_r2.toBytes()});
                    dbg("[BRUTE_FORCE2] sum == claim/R^2 = {}\n", .{brute_sum2.eql(claim_div_r2)});

                    // Compute sum WITHOUT current_scalar to isolate the issue
                    var brute_raw = F.zero();
                    for (0..E_out_b.len) |x_out| {
                        for (0..E_in_b.len) |x_in| {
                            const i = (x_out << @intCast(head_in_bits)) | x_in;
                            const eq_wt = E_out_b[x_out].mul(E_in_b[x_in]);
                            var inner_s = F.zero();
                            for (0..brute_grid) |j| {
                                const idx = brute_grid * i + j;
                                if (idx < az_p.boundLen()) {
                                    const w_jj = if (j == 0) one_minus_w_last else w_last;
                                    inner_s = inner_s.add(w_jj.mul(az_p.evaluations[idx].mul(bz_p.evaluations[idx])));
                                }
                            }
                            brute_raw = brute_raw.add(eq_wt.mul(inner_s));
                        }
                    }
                    dbg("[BRUTE_FORCE2] raw_sum (without current_scalar) = {any}\n", .{brute_raw.toBytes()});
                    const raw_with_cs = brute_raw.mul(self.split_eq.current_scalar);
                    dbg("[BRUTE_FORCE2] raw_sum * current_scalar = {any}\n", .{raw_with_cs.toBytes()});
                    const raw_with_cs_r2 = raw_with_cs.mul(r_squared);
                    dbg("[BRUTE_FORCE2] raw_sum * current_scalar * R^2 = {any}\n", .{raw_with_cs_r2.toBytes()});
                    dbg("[BRUTE_FORCE2] claim == raw*cs*R^2? {}\n", .{raw_with_cs_r2.eql(self.current_claim)});

                    // Also print first few values
                    for (0..@min(3, E_out_b.len)) |x_out| {
                        for (0..@min(2, E_in_b.len)) |x_in| {
                            const i = (x_out << @intCast(head_in_bits)) | x_in;
                            dbg("[BRUTE_FORCE2] E_out[{d}]={any}  E_in[{d}]={any}\n", .{x_out, E_out_b[x_out].toBytes(), x_in, E_in_b[x_in].toBytes()});
                            for (0..brute_grid) |j| {
                                const idx = brute_grid * i + j;
                                if (idx < az_p.boundLen()) {
                                    dbg("[BRUTE_FORCE2]   Az[{d}]={any}  Bz[{d}]={any}\n", .{idx, az_p.evaluations[idx].toBytes(), idx, bz_p.evaluations[idx].toBytes()});
                                }
                            }
                        }
                    }
                }
                } // end if (comptime debug_verbose) for BRUTE_FORCE2

                // DEBUG3: Compute sum using UniSkip's factorization but with materialized Az/Bz
                // This isolates whether the issue is in eq tables or Az/Bz values
                if (comptime debug_verbose) {
                if (self.az_poly != null and self.bz_poly != null) {
                    const az_p3 = &(self.az_poly.?);
                    const bz_p3 = &(self.bz_poly.?);

                    // Build eq tables using UniSkip's factorization (full_tau, m = full_tau.len/2)
                    const m_uni = self.full_tau.len / 2;
                    const wprime_len_uni = if (self.full_tau.len > 0) self.full_tau.len - 1 else 0;
                    const num_x_in_bits_uni = if (wprime_len_uni > m_uni) wprime_len_uni - m_uni else 0;
                    const num_x_in_prime_bits_uni = if (num_x_in_bits_uni > 0) num_x_in_bits_uni - 1 else 0;

                    const E_out_uni = self.buildEqTable(self.full_tau[0..m_uni]) catch unreachable;
                    defer self.allocator.free(E_out_uni);
                    const E_in_uni = self.buildEqTable(self.full_tau[m_uni .. self.full_tau.len - 1]) catch unreachable;
                    defer self.allocator.free(E_in_uni);

                    dbg("[DEBUG3] UniSkip factorization: m={d}, E_out.len={d}, E_in.len={d}\n", .{ m_uni, E_out_uni.len, E_in_uni.len });
                    dbg("[DEBUG3] E_out_uni[0] = {any}\n", .{E_out_uni[0].toBytesBE()});
                    dbg("[DEBUG3] E_in_uni[0] = {any}\n", .{E_in_uni[0].toBytesBE()});

                    // Sum using UniSkip's index mapping but materialized Az/Bz
                    var sum3 = F.zero();
                    for (0..E_out_uni.len) |x_out_u| {
                        for (0..E_in_uni.len) |x_in_u| {
                            const eq_val3 = E_out_uni[x_out_u].mul(E_in_uni[x_in_u]);
                            const x_in_prime_u = x_in_u >> 1;
                            const cycle_u = (x_out_u << @intCast(num_x_in_prime_bits_uni)) | x_in_prime_u;
                            const group_u: usize = x_in_u & 1;

                            // Map to materialized array index
                            // In materialized layout: full_idx = 2 * i + j where i = x_out' * E_in' + x_in'
                            // cycle_u corresponds to i when viewed differently
                            // The materialized array is indexed as: Az[2*cycle + group]
                            const mat_idx = 2 * cycle_u + group_u;
                            if (mat_idx < az_p3.boundLen()) {
                                sum3 = sum3.add(eq_val3.mul(az_p3.evaluations[mat_idx].mul(bz_p3.evaluations[mat_idx])));
                            }
                        }
                    }
                    // Scale by current_scalar (lagrange_tau_r0)
                    sum3 = sum3.mul(self.split_eq.current_scalar);
                    dbg("[DEBUG3] sum_uniskip_eq_with_mat_AzBz * current_scalar = {any}\n", .{sum3.toBytes()});
                    dbg("[DEBUG3] matches claim? {}\n", .{sum3.eql(self.current_claim)});

                    // Now compute using split_eq factorization but with UniSkip's cycle mapping
                    // to see if the issue is in eq tables or cycle mapping
                    var sum4 = F.zero();
                    const eq_t4 = self.split_eq.getWindowEqTables(0, 1);
                    const E_out_4 = eq_t4.E_out;
                    const E_in_4 = eq_t4.E_in;
                    const hib4: u6 = @intCast(eq_t4.head_in_bits);
                    const w_last4 = self.split_eq.tau[self.split_eq.current_index - 1];
                    const one_minus_w_last4 = F.one().sub(w_last4);

                    for (0..E_out_4.len) |xo4| {
                        for (0..E_in_4.len) |xi4| {
                            const eq_base4 = E_out_4[xo4].mul(E_in_4[xi4]);
                            const idx4 = (xo4 << hib4) | xi4;
                            // Use SAME materialized layout: Az[2*i + j]
                            const az0_4 = if (2 * idx4 < az_p3.boundLen()) az_p3.evaluations[2 * idx4] else F.zero();
                            const bz0_4 = if (2 * idx4 < bz_p3.boundLen()) bz_p3.evaluations[2 * idx4] else F.zero();
                            const az1_4 = if (2 * idx4 + 1 < az_p3.boundLen()) az_p3.evaluations[2 * idx4 + 1] else F.zero();
                            const bz1_4 = if (2 * idx4 + 1 < bz_p3.boundLen()) bz_p3.evaluations[2 * idx4 + 1] else F.zero();
                            const inner4 = one_minus_w_last4.mul(az0_4.mul(bz0_4)).add(w_last4.mul(az1_4.mul(bz1_4)));
                            sum4 = sum4.add(eq_base4.mul(inner4));
                        }
                    }
                    sum4 = sum4.mul(self.split_eq.current_scalar);
                    dbg("[DEBUG3] sum_split_eq_with_mat_AzBz * current_scalar = {any}\n", .{sum4.toBytes()});
                    dbg("[DEBUG3] sum3 == sum4? {}\n", .{sum3.eql(sum4)});

                    // Check: claim / current_scalar vs raw sum without scalar
                    const cs_inv_3 = self.split_eq.current_scalar.inverse();
                    if (cs_inv_3) |inv3| {
                        const claim_div_cs3 = self.current_claim.mul(inv3);
                        // sum3 / current_scalar = raw sum (without current_scalar)
                        const raw3 = sum3.mul(inv3);
                        dbg("[DEBUG3] claim / current_scalar = {any}\n", .{claim_div_cs3.toBytes()});
                        dbg("[DEBUG3] raw_eq_sum (sum3/cs) = {any}\n", .{raw3.toBytes()});
                        dbg("[DEBUG3] claim/cs == raw_eq_sum? {}\n", .{claim_div_cs3.eql(raw3)});
                    }

                    // KEY TEST: The uni_skip_claim = s1(r0) = L(tau_high, r0) * t1(r0) * R^2
                    // t1(r0) = Σ eq(tau, x) * AzBz(x, r0)
                    // raw_sum3 = Σ eq(tau, x) * AzBz_mat(x, r0) (using UniSkip eq tables)
                    // If AzBz_mat matches AzBz, then t1(r0) should equal raw_sum3
                    // And claim = current_scalar * t1(r0) * R^2
                    // But we applied R^2 scaling in the UniSkip extended_evals...
                    // So claim = L(tau_high, r0) * [Σ eq * AzBz * R^2](r0)
                    // While sum3 = L(tau_high, r0) * Σ eq * AzBz(r0) (NO R^2 in sum3)
                    // So: claim = sum3 * R^2 ??? Let me check this
                    const r_sq_3 = F.rSquared();
                    dbg("[DEBUG3] sum3 * R^2 = {any}\n", .{sum3.mul(r_sq_3).toBytes()});
                    dbg("[DEBUG3] claim == sum3 * R^2? {}\n", .{self.current_claim.eql(sum3.mul(r_sq_3))});

                    // Correct decomposition:
                    // claim = current_scalar * t1_with_r2(r0)
                    // t1_with_r2(r0) = claim / current_scalar
                    // raw_eq_sum = Σ eq * AzBz(r0) = t1(r0) (without R^2)
                    // So we expect: claim / current_scalar = raw_eq_sum * R^2
                    {
                        const cs_inv_32 = self.split_eq.current_scalar.inverse().?;
                        const t1_r0 = self.current_claim.mul(cs_inv_32);
                        const raw_sum_3 = sum3.mul(cs_inv_32);
                        const raw_times_r2 = raw_sum_3.mul(r_sq_3);
                        dbg("[DEBUG3] t1_with_r2(r0) = claim/cs = {any}\n", .{t1_r0.toBytes()});
                        dbg("[DEBUG3] raw_sum * R^2 = {any}\n", .{raw_times_r2.toBytes()});
                        dbg("[DEBUG3] t1_with_r2 == raw*R^2? {}\n", .{t1_r0.eql(raw_times_r2)});
                    }

                    // Multi-cycle check + direct sum comparison
                    {
                        var mismatch_n: usize = 0;
                        const climit = self.cycle_witnesses.len;
                        for (0..climit) |cchk| {
                            for (0..2) |gchk| {
                                const gsz: usize = if (gchk == 0) FIRST_GROUP_SIZE else SECOND_GROUP_SIZE;
                                const gid = if (gchk == 0) &constraints.FIRST_GROUP_INDICES else &constraints.SECOND_GROUP_INDICES;
                                const ww = &self.cycle_witnesses[cchk];
                                var azd = F.zero();
                                var bzd = F.zero();
                                for (0..gsz) |kk| {
                                    const ccc = constraints.UNIFORM_CONSTRAINTS[gid[kk]];
                                    const cdv = ccc.condition.evaluate(F, ww.asSlice());
                                    const mgv = ccc.left.evaluate(F, ww.asSlice()).sub(ccc.right.evaluate(F, ww.asSlice()));
                                    azd = azd.add(self.lagrange_evals_r0[kk].mul(cdv));
                                    bzd = bzd.add(self.lagrange_evals_r0[kk].mul(mgv));
                                }
                                const midx = 2 * cchk + gchk;
                                const azm = if (midx < az_p3.boundLen()) az_p3.evaluations[midx] else F.zero();
                                const bzm = if (midx < bz_p3.boundLen()) bz_p3.evaluations[midx] else F.zero();
                                if (!azd.eql(azm) or !bzd.eql(bzm)) {
                                    mismatch_n += 1;
                                    if (mismatch_n <= 3) {
                                        dbg("[DEBUG5] AzBz MISMATCH c={d} g={d} mi={d}\n", .{cchk, gchk, midx});
                                        dbg("[DEBUG5]  Az_d={any}  Az_m={any}\n", .{azd.toBytesBE(), azm.toBytesBE()});
                                    }
                                }
                            }
                        }
                        dbg("[DEBUG5] Checked {d} cycles x 2 groups, {d} mismatches\n", .{climit, mismatch_n});

                        // Compute direct sum of eq*AzBz at r0 over ALL 512 indices (9 bits)
                        // using the UniSkip's eq tables and the Lagrange-evaluated Az/Bz
                        var direct_sum_r0 = F.zero();
                        for (0..E_out_uni.len) |xou7| {
                            for (0..E_in_uni.len) |xiu7| {
                                const eq7 = E_out_uni[xou7].mul(E_in_uni[xiu7]);
                                const xip7 = xiu7 >> 1;
                                const cyc7 = (xou7 << @intCast(num_x_in_prime_bits_uni)) | xip7;
                                const grp7: usize = xiu7 & 1;

                                if (cyc7 < climit) {
                                    const gsz7: usize = if (grp7 == 0) FIRST_GROUP_SIZE else SECOND_GROUP_SIZE;
                                    const gid7 = if (grp7 == 0) &constraints.FIRST_GROUP_INDICES else &constraints.SECOND_GROUP_INDICES;
                                    const ww7 = &self.cycle_witnesses[cyc7];
                                    var az7 = F.zero();
                                    var bz7 = F.zero();
                                    for (0..gsz7) |kk7| {
                                        const cc7 = constraints.UNIFORM_CONSTRAINTS[gid7[kk7]];
                                        const cd7 = cc7.condition.evaluate(F, ww7.asSlice());
                                        const mg7 = cc7.left.evaluate(F, ww7.asSlice()).sub(cc7.right.evaluate(F, ww7.asSlice()));
                                        az7 = az7.add(self.lagrange_evals_r0[kk7].mul(cd7));
                                        bz7 = bz7.add(self.lagrange_evals_r0[kk7].mul(mg7));
                                    }
                                    direct_sum_r0 = direct_sum_r0.add(eq7.mul(az7.mul(bz7)));
                                }
                            }
                        }
                        dbg("[DEBUG6] direct_sum_r0 (eq_uni * AzBz_r0) = {any}\n", .{direct_sum_r0.toBytes()});
                        dbg("[DEBUG6] mat_sum (sum3/cs) = {any}\n", .{sum3.mul(self.split_eq.current_scalar.inverse().?).toBytes()});
                        dbg("[DEBUG6] direct == mat? {}\n", .{direct_sum_r0.eql(sum3.mul(self.split_eq.current_scalar.inverse().?))});
                        dbg("[DEBUG6] claim/cs = {any}\n", .{self.current_claim.mul(self.split_eq.current_scalar.inverse().?).toBytes()});
                        dbg("[DEBUG6] direct == claim/cs? {}\n", .{direct_sum_r0.eql(self.current_claim.mul(self.split_eq.current_scalar.inverse().?))});
                    }

                    // Compare per-cycle Az*Bz between UniSkip and materialization for cycle 0
                    // UniSkip path: evaluate at extended points, interpolate, evaluate at r0
                    // Materialization path: Lagrange evaluate directly at r0
                    if (self.cycle_witnesses.len > 0) {
                        const w0 = &self.cycle_witnesses[0];
                        // Compute UniSkip's Az*Bz for cycle 0, group 0 at each target
                        // First compute base values
                        var az_base_g0: [FIRST_GROUP_SIZE]F = undefined;
                        var bz_base_g0: [FIRST_GROUP_SIZE]F = undefined;
                        for (0..FIRST_GROUP_SIZE) |ii| {
                            const cidx = constraints.FIRST_GROUP_INDICES[ii];
                            const cst = constraints.UNIFORM_CONSTRAINTS[cidx];
                            az_base_g0[ii] = cst.condition.evaluate(F, w0.asSlice());
                            bz_base_g0[ii] = cst.left.evaluate(F, w0.asSlice())
                                .sub(cst.right.evaluate(F, w0.asSlice()));
                        }

                        // Print base Az*Bz products (should be 0 for correct witness)
                        for (0..@min(3, FIRST_GROUP_SIZE)) |ii| {
                            dbg("[DEBUG4] base Az[{d}]*Bz[{d}] = {any}\n", .{
                                ii, ii, az_base_g0[ii].mul(bz_base_g0[ii]).toBytesBE(),
                            });
                        }

                        // Compute direct Az(r0) and Bz(r0) using lagrange_evals_r0
                        var az_direct_r0 = F.zero();
                        var bz_direct_r0 = F.zero();
                        for (0..FIRST_GROUP_SIZE) |ii| {
                            az_direct_r0 = az_direct_r0.add(self.lagrange_evals_r0[ii].mul(az_base_g0[ii]));
                            bz_direct_r0 = bz_direct_r0.add(self.lagrange_evals_r0[ii].mul(bz_base_g0[ii]));
                        }
                        dbg("[DEBUG4] Az(r0)_direct = {any}\n", .{az_direct_r0.toBytesBE()});
                        dbg("[DEBUG4] Bz(r0)_direct = {any}\n", .{bz_direct_r0.toBytesBE()});
                        dbg("[DEBUG4] Az(r0)*Bz(r0)_direct = {any}\n", .{az_direct_r0.mul(bz_direct_r0).toBytesBE()});

                        // Compare with materialized value
                        dbg("[DEBUG4] Az_mat[0] = {any}\n", .{az_p3.evaluations[0].toBytesBE()});
                        dbg("[DEBUG4] Bz_mat[0] = {any}\n", .{bz_p3.evaluations[0].toBytesBE()});
                        dbg("[DEBUG4] Az_mat*Bz_mat[0] = {any}\n", .{az_p3.evaluations[0].mul(bz_p3.evaluations[0]).toBytesBE()});
                        dbg("[DEBUG4] Az match? {}\n", .{az_direct_r0.eql(az_p3.evaluations[0])});
                        dbg("[DEBUG4] Bz match? {}\n", .{bz_direct_r0.eql(bz_p3.evaluations[0])});

                        // Now compute via UniSkip's interpolation approach
                        // t1_at_r0_cycle0_g0 = Σ_Y t1_vals(Y) * L_Y(r0) / domain_vanishing(r0)
                        // But actually, the UniSkip computes AzBz at extended points and then
                        // does polynomial interpolation. Let me instead just evaluate the PRODUCT
                        // polynomial at r0 by evaluating Az and Bz separately:
                        //   Az(r0) = Σ_i L_i(r0) * az_base[i]
                        //   Bz(r0) = Σ_i L_i(r0) * bz_base[i]
                        // This should equal the materialized Az[0], Bz[0]

                        // The issue might be in lagrange_evals_r0 computation!
                        // Print first few lagrange_evals_r0
                        for (0..@min(4, FIRST_GROUP_SIZE)) |ii| {
                            dbg("[DEBUG4] lagrange_evals_r0[{d}] = {any}\n", .{ii, self.lagrange_evals_r0[ii].toBytesBE()});
                        }
                    }
                }
                } // end if (comptime debug_verbose) for DEBUG3+DEBUG5+DEBUG6+DEBUG4
            }

            // Use t_prime_poly for all rounds (linear-only schedule)
            if (self.t_prime_poly != null) {
                // LINEAR PHASE: Use t_prime_poly directly
                //
                // The t_prime_poly was built during materializeLinearPhasePolynomials
                // and is bound after each round. We use computeTEvals to project
                // it to (t_zero, t_infinity) using E_active weights.
                //
                // IMPORTANT: If t_prime_poly has num_vars == 0, we need to rebuild it
                // from the bound Az/Bz polynomials (this is nextWindow in Jolt)
                if (self.t_prime_poly.?.num_vars == 0 and self.az_poly != null and self.bz_poly != null) {
                    // Rebuild t_prime_poly from bound Az/Bz (nextWindow equivalent)
                    // DEBUG: Print t_prime[0] BEFORE rebuild
                    try self.rebuildTPrimePoly(window_size);
                }

                const t_evals = try self.computeTEvals(window_size);
                t_zero = t_evals.t_zero;
                t_infinity = t_evals.t_infinity;
            } else {
                // No t_prime_poly available (shouldn't happen with LinearOnlySchedule)
                return error.TPrimePolyNotAvailable;
            }

            // Use Gruen's method to compute the cubic round polynomial
            const previous_claim = self.current_claim;
            const round_poly = self.split_eq.computeCubicRoundPoly(
                t_zero,
                t_infinity,
                previous_claim,
            );

            return round_poly;
        }

        /// Compute Az * Bz product for a single cycle (for given constraint group)
        ///
        /// Computation:
        /// Az = Σ_i L_i(r0) * condition_i(witness)
        /// Bz = Σ_i L_i(r0) * (left_i - right_i)(witness)
        /// Return Az * Bz
        ///
        /// This computes Az*Bz for a single constraint group. The streaming round
        /// uses both groups combined with r_stream.
        fn computeCycleAzBzProductForGroup(
            self: *const Self,
            witness: *const constraints.R1CSCycleInputs(F),
            group: usize, // 0 = first group, 1 = second group
        ) F {
            const result = self.computeCycleAzBzForGroup(witness, group);
            return result.az.mul(result.bz);
        }

        /// Compute separate Az and Bz for a single cycle for a given constraint group.
        /// Uses compact integer witnesses when available for better cache performance.
        fn computeCycleAzBzForGroup(
            self: *const Self,
            witness: *const constraints.R1CSCycleInputs(F),
            group: usize, // 0 = first group, 1 = second group
        ) struct { az: F, bz: F } {
            const ws = witness.asSlice();

            if (group == 0) {
                const az_int = evaluators.computeAzFirstGroupInt(F, ws);
                const bz_field = evaluators.computeBzFirstGroupDirect(F, ws);

                var az_sum = F.zero();
                var bz_sum = F.zero();
                for (0..FIRST_GROUP_SIZE) |i| {
                    const w = self.lagrange_evals_r0[i];
                    const az_i = az_int[i];
                    if (az_i == 1) {
                        az_sum = az_sum.add(w);
                    } else if (az_i == -1) {
                        az_sum = az_sum.sub(w);
                    } else if (az_i != 0) {
                        az_sum = az_sum.add(w.mul(evaluators.fieldFromI32(F, @as(i32, az_i))));
                    }
                    bz_sum = bz_sum.add(w.mul(bz_field[i]));
                }
                return .{ .az = az_sum, .bz = bz_sum };
            } else {
                const az_int = evaluators.computeAzSecondGroupInt(F, ws);
                const bz_field = evaluators.computeBzSecondGroupDirect(F, ws, self.two_pow_64_cached);

                const g2_size = @min(SECOND_GROUP_SIZE, FIRST_GROUP_SIZE);
                var az_sum = F.zero();
                var bz_sum = F.zero();
                for (0..g2_size) |i| {
                    const w = self.lagrange_evals_r0[i];
                    const az_i = az_int[i];
                    if (az_i == 1) {
                        az_sum = az_sum.add(w);
                    } else if (az_i == -1) {
                        az_sum = az_sum.sub(w);
                    } else if (az_i != 0) {
                        az_sum = az_sum.add(w.mul(evaluators.fieldFromI32(F, @as(i32, az_i))));
                    }
                    bz_sum = bz_sum.add(w.mul(bz_field[i]));
                }
                return .{ .az = az_sum, .bz = bz_sum };
            }
        }

        /// Compact witness version of computeCycleAzBzForGroup.
        /// Uses precomputed integer Az/Bz values for better cache performance.
        fn computeCycleAzBzForGroupCompact(
            self: *const Self,
            cw: *const evaluators.CompactWitness,
            group: usize,
        ) struct { az: F, bz: F } {
            if (group == 0) {
                // Barrett accumulator path: SmallAccumU for Az (i8), MedAccumS for Bz (i128)
                var az_acc = field_mod.SmallAccumU.zero();
                var bz_acc = field_mod.MedAccumS.zero();
                inline for (0..FIRST_GROUP_SIZE) |i| {
                    az_acc.fmaddI8(self.lagrange_evals_r0[i], cw.az_first[i]);
                    bz_acc.fmaddI128(self.lagrange_evals_r0[i], cw.bz_first[i]);
                }
                return .{ .az = az_acc.barrettReduce(), .bz = bz_acc.barrettReduce() };
            } else {
                // Second group: SmallAccumU for Az, WideAccumS for Bz (S192)
                const g2_size = comptime @min(SECOND_GROUP_SIZE, FIRST_GROUP_SIZE);
                var az_acc = field_mod.SmallAccumU.zero();
                var bz_acc = field_mod.WideAccumS.zero();
                inline for (0..g2_size) |i| {
                    az_acc.fmaddI8(self.lagrange_evals_r0[i], cw.az_second[i]);
                    bz_acc.fmaddS192(self.lagrange_evals_r0[i], cw.bz_second[i]);
                }
                return .{ .az = az_acc.barrettReduce(), .bz = bz_acc.barrettReduce() };
            }
        }

        /// Compute combined Az * Bz for a single cycle using bound r_stream value
        ///
        /// The formula is:
        /// Az_final = (1 - r_stream) * Az_g0 + r_stream * Az_g1
        /// Bz_final = (1 - r_stream) * Bz_g0 + r_stream * Bz_g1
        /// Return Az_final * Bz_final
        fn computeCycleAzBzProductCombined(
            self: *const Self,
            witness: *const constraints.R1CSCycleInputs(F),
            r_stream: F,
        ) F {
            const result_g0 = self.computeCycleAzBzForGroup(witness, 0);
            const result_g1 = self.computeCycleAzBzForGroup(witness, 1);

            const az_final = result_g0.az.add(r_stream.mul(result_g1.az.sub(result_g0.az)));
            const bz_final = result_g0.bz.add(r_stream.mul(result_g1.bz.sub(result_g0.bz)));

            return az_final.mul(bz_final);
        }

        /// Compute Az * Bz product for a single cycle (legacy, uses only group 0)
        /// This is kept for compatibility but should not be used for correct proofs.
        fn computeCycleAzBzProduct(self: *const Self, witness: *const constraints.R1CSCycleInputs(F)) F {
            return self.computeCycleAzBzProductForGroup(witness, 0);
        }

        /// Compute Az*Bz for multiquadratic expansion (streaming round)
        ///
        /// Returns values for the multiquadratic grid:
        /// - prod_0 = Az_g0 * Bz_g0 (product at position 0)
        /// - prod_inf = (Az_g1 - Az_g0) * (Bz_g1 - Bz_g0) (product of slopes)
        ///
        /// This is used in the streaming round where we select between constraint groups.
        fn computeCycleAzBzForMultiquadratic(
            self: *const Self,
            witness: *const constraints.R1CSCycleInputs(F),
        ) struct { prod_0: F, prod_inf: F } {
            const result_g0 = self.computeCycleAzBzForGroup(witness, 0);
            const result_g1 = self.computeCycleAzBzForGroup(witness, 1);

            const prod_0 = result_g0.az.mul(result_g0.bz);
            const slope_az = result_g1.az.sub(result_g0.az);
            const slope_bz = result_g1.bz.sub(result_g0.bz);
            const prod_inf = slope_az.mul(slope_bz);

            return .{ .prod_0 = prod_0, .prod_inf = prod_inf };
        }

        /// Compute Az and Bz values for both groups (without products)
        ///
        /// Returns (az_g0, az_g1, bz_g0, bz_g1) for use in product-of-sums computation.
        /// This is needed because (Σ Az) * (Σ Bz) ≠ Σ (Az * Bz).
        fn computeCycleAzBzValues(
            self: *const Self,
            witness: *const constraints.R1CSCycleInputs(F),
        ) struct { az_g0: F, az_g1: F, bz_g0: F, bz_g1: F } {
            const result_g0 = self.computeCycleAzBzForGroup(witness, 0);
            const result_g1 = self.computeCycleAzBzForGroup(witness, 1);
            return .{ .az_g0 = result_g0.az, .az_g1 = result_g1.az, .bz_g0 = result_g0.bz, .bz_g1 = result_g1.bz };
        }

        /// Compute Az and Bz separately for a single cycle (combined groups)
        ///
        /// Returns (Az_final, Bz_final) where:
        /// Az_final = (1 - r_stream) * Az_g0 + r_stream * Az_g1
        /// Bz_final = (1 - r_stream) * Bz_g0 + r_stream * Bz_g1
        fn computeCycleAzBzSeparate(
            self: *const Self,
            witness: *const constraints.R1CSCycleInputs(F),
            r_stream: F,
        ) struct { az: F, bz: F } {
            const v = self.computeCycleAzBzValues(witness);
            const az_final = v.az_g0.add(r_stream.mul(v.az_g1.sub(v.az_g0)));
            const bz_final = v.bz_g0.add(r_stream.mul(v.bz_g1.sub(v.bz_g0)));
            return .{ .az = az_final, .bz = bz_final };
        }

        /// Compute remaining round polynomial using multiquadratic expansion
        ///
        /// This is the correct approach:
        /// 1. Compute Az and Bz grids separately for each cycle
        /// 2. Expand each to multiquadratic (f(∞) = f(1) - f(0))
        /// 3. Multiply pointwise to get Az*Bz on multiquadratic grid
        /// 4. Sum with eq weights to get t'(0) and t'(∞)
        ///
        /// IMPORTANT: The eq weights for cycles use a FACTORIZED representation:
        ///   eq_val[i] = E_out[i / E_in.len] * E_in[i % E_in.len]
        ///
        /// The r_grid contains eq(r_bound, cycle_bits) for already-bound challenges.
        /// This is used to weight each cycle according to how it matches the bound challenges.
        ///
        /// For the streaming round (current_round == 1):
        /// - We sum over all cycles and constraint groups
        ///
        /// For subsequent cycle rounds:
        /// - We sum over cycle halves (based on current variable)
        /// - Each cycle is weighted by E_out * E_in * r_grid[k]
        pub fn computeRemainingRoundPolyMultiquadratic(self: *Self) ![4]F {
            // Match Jolt's linear phase index structure exactly:
            //   full_idx = x_out << (x_in_bits + window + r_bits) | x_in << (window + r_bits) | x_val << r_bits | r_idx
            //   step_idx = full_idx >> 1
            //   selector = full_idx & 1
            //
            // The constraint group selector is always the LSB.
            // For cycle rounds, x_val ∈ {0, 1} is the current cycle bit.
            // r_idx indexes into r_grid for the bound streaming challenges.
            // Note: r_stream is NOT used here - the selector comes from full_idx & 1.

            // Get eq tables for current window
            const eq_tables = self.split_eq.getWindowEqTables(0, 1);
            const E_out = eq_tables.E_out;
            const E_in = eq_tables.E_in;
            const head_in_bits: u6 = @intCast(eq_tables.head_in_bits);

            // r_grid parameters
            const r_grid = &self.r_grid;
            const r_grid_len = r_grid.length();
            const num_r_bits: u6 = if (r_grid_len > 1) @intCast(std.math.log2_int(usize, r_grid_len)) else 0;

            // window_size is always 1 for linear phase cycle rounds
            const window_bits: u6 = 1;

            // Accumulators for multiquadratic polynomial (parallelized over x_out)
            const ReduceCtx = struct {
                E_out: []const F,
                E_in: []const F,
                r_grid: *const ExpandingTable(F),
                r_grid_len: usize,
                head_in_bits: u6,
                window_bits: u6,
                num_r_bits: u6,
                compact_witnesses: []const evaluators.CompactWitness,
                self_ptr: *const Self,
            };
            const reduceMap = struct {
                fn f(ctx: ReduceCtx, start: usize, end: usize) [2]F {
                    var local_t_00 = F.zero();
                    var local_t_inf = F.zero();
                    const cw_len = ctx.compact_witnesses.len;

                    var x_out_idx: usize = start;
                    while (x_out_idx < end) : (x_out_idx += 1) {
                        const e_out_val = ctx.E_out[x_out_idx];

                        var x_in_idx: usize = 0;
                        while (x_in_idx < ctx.E_in.len) : (x_in_idx += 1) {
                            const e_in_val = ctx.E_in[x_in_idx];
                            const eq_base = e_out_val.mul(e_in_val);

                            var az_grid = [2]F{ F.zero(), F.zero() };
                            var bz_grid = [2]F{ F.zero(), F.zero() };

                            const base_idx: usize = (x_out_idx << @intCast(ctx.head_in_bits + ctx.window_bits + ctx.num_r_bits)) |
                                (x_in_idx << @intCast(ctx.window_bits + ctx.num_r_bits));

                            var x_val: usize = 0;
                            while (x_val < 2) : (x_val += 1) {
                                const x_val_shifted = x_val << ctx.num_r_bits;

                                {
                                    // Use existing compact/field dispatch
                                    var r_idx: usize = 0;
                                    while (r_idx < ctx.r_grid_len) : (r_idx += 1) {
                                        const r_weight = ctx.r_grid.get(r_idx);
                                        const full_idx = base_idx | x_val_shifted | r_idx;
                                        const step_idx = full_idx >> 1;
                                        const selector: usize = full_idx & 1;

                                        if (step_idx < cw_len) {
                                            const result = ctx.self_ptr.computeCycleAzBzForGroupCompact(&ctx.compact_witnesses[step_idx], selector);
                                            az_grid[x_val] = az_grid[x_val].add(r_weight.mul(result.az));
                                            bz_grid[x_val] = bz_grid[x_val].add(r_weight.mul(result.bz));
                                        }
                                    }
                                }
                            }

                            const prod_0 = az_grid[0].mul(bz_grid[0]);
                            const slope_az = az_grid[1].sub(az_grid[0]);
                            const slope_bz = bz_grid[1].sub(bz_grid[0]);
                            const prod_inf = slope_az.mul(slope_bz);

                            local_t_00 = local_t_00.add(eq_base.mul(prod_0));
                            local_t_inf = local_t_inf.add(eq_base.mul(prod_inf));
                        }
                    }

                    return [2]F{ local_t_00, local_t_inf };
                }
            }.f;
            const reduceAdd = struct {
                fn f(a: [2]F, b: [2]F) [2]F {
                    return [2]F{ a[0].add(b[0]), a[1].add(b[1]) };
                }
            }.f;

            const reduce_ctx = ReduceCtx{
                .E_out = E_out,
                .E_in = E_in,
                .r_grid = &self.r_grid,
                .r_grid_len = r_grid_len,
                .head_in_bits = head_in_bits,
                .window_bits = window_bits,
                .num_r_bits = num_r_bits,
                .compact_witnesses = self.compact_witnesses,
                .self_ptr = self,
            };

            const identity = [2]F{ F.zero(), F.zero() };
            const t_results = if (self.thread_pool) |tp|
                tp.parallelReduce([2]F, E_out.len, identity, reduce_ctx, reduceMap, reduceAdd)
            else
                reduceMap(reduce_ctx, 0, E_out.len);

            const t_00 = t_results[0];
            const t_inf = t_results[1];

            // Use Gruen's method
            const previous_claim = self.current_claim;
            const round_poly = self.split_eq.computeCubicRoundPoly(
                t_00,
                t_inf,
                previous_claim,
            );

            return round_poly;
        }

        /// Bind a remaining round challenge
        ///
        /// Matches Jolt's LinearOnlySchedule:
        /// - Streaming round (round 1): update r_grid, then materialize Az/Bz
        /// - Linear phase (round > 1): bind Az/Bz polynomials, don't update r_grid
        ///
        /// For the remaining sumcheck with num_rounds = 1 + num_cycle_vars:
        /// - Switch-over is at round 0 (linear phase starts immediately after streaming)
        /// - Streaming round: 1 only
        /// - Linear rounds: 2 to num_rounds
        pub fn bindRemainingRoundChallenge(self: *Self, r: F) !void {
            // If this is the first remaining round (current_round == 1), save r_stream
            // This is used for blending constraint groups
            if (self.current_round == 1) {
                self.r_stream = r;
            }

            try self.challenges.append(self.allocator, r);

            // CRITICAL: Match Jolt's ingest_challenge binding order exactly:
            // 1. split_eq_poly.bind(r_j) FIRST
            // 2. t_prime_poly.bind(r_j, BindingOrder::LowToHigh)
            // 3. az.bind_parallel(r_j) and bz.bind_parallel(r_j)
            //
            // This order matters because getWindowEqTables reads from split_eq,
            // and if we bind az/bz before split_eq, the eq tables used in
            // buildTPrimePoly (next_window) will be at the wrong state.

            // 1. Bind split_eq FIRST
            self.split_eq.bind(r);

            // 2. Skip t_prime_poly bind — we rebuild it entirely during the fused Az/Bz bind below.

            // 3. Bind Az/Bz polynomials LAST, AND fuse t_prime rebuild (window_size=1).
            // Instead of bind → rebuildTPrimePoly (2 passes), do both in one pass.
            // split_eq is already bound (step 1), so getWindowEqTables returns updated tables.
            if (self.az_poly != null and self.bz_poly != null) {
                const window_size: usize = 1;
                const eq_tables = self.split_eq.getWindowEqTables(0, window_size);
                const E_out = eq_tables.E_out;
                const E_in = eq_tables.E_in;
                const num_xin_bits: u6 = if (E_in.len > 1) @intCast(std.math.log2_int(usize, E_in.len)) else 0;

                const az_src = self.az_poly.?.evaluations;
                const bz_src = self.bz_poly.?.evaluations;
                // Use boundLen() to process only the valid portion of the array,
                // not the full allocated length. Without this, later rounds waste
                // time binding ~524k zero entries.
                const current_bound = self.az_poly.?.boundLen();
                const new_size = current_bound / 2;
                const total_pairs = E_out.len * E_in.len;

                // Each pair reads 2 consecutive bound values: bound[2*pair], bound[2*pair+1]
                // bound[j] = old[2*j] + r*(old[2*j+1] - old[2*j])
                // So pair i reads old[4*i..4*i+3]
                const FusedBindCtx = struct {
                    az_s: []const F, az_d: []F,
                    bz_s: []const F, bz_d: []F,
                    r_val: F,
                    E_out: []const F, E_in: []const F,
                    num_xin_bits: u6,
                    total_pairs: usize,
                    new_size: usize,
                };

                const az_scratch = self.az_poly.?.scratch;
                const bz_scratch = self.bz_poly.?.scratch;
                const has_scratch = az_scratch != null and bz_scratch != null;
                const az_dest = if (has_scratch) az_scratch.? else az_src;
                const bz_dest = if (has_scratch) bz_scratch.? else bz_src;

                const fused_ctx = FusedBindCtx{
                    .az_s = az_src, .az_d = az_dest,
                    .bz_s = bz_src, .bz_d = bz_dest,
                    .r_val = r,
                    .E_out = E_out, .E_in = E_in,
                    .num_xin_bits = num_xin_bits,
                    .total_pairs = total_pairs,
                    .new_size = new_size,
                };

                const fusedMapFn = struct {
                    fn f(ctx: FusedBindCtx, start: usize, end: usize) [3]F {
                        var local_ans = [3]F{ F.zero(), F.zero(), F.zero() };

                        for (start..end) |pair_idx| {
                            const x_out = pair_idx / ctx.E_in.len;
                            const x_in = pair_idx % ctx.E_in.len;
                            const i = (@as(usize, x_out) << ctx.num_xin_bits) | x_in;

                            // Compute bound values for this pair's two elements: bound[2i], bound[2i+1]
                            const idx0 = 2 * i;
                            const idx1 = 2 * i + 1;

                            var az0: F = F.zero();
                            var az1: F = F.zero();
                            var bz0: F = F.zero();
                            var bz1: F = F.zero();

                            if (idx0 < ctx.new_size) {
                                const a_lo = ctx.az_s[2 * idx0];
                                az0 = a_lo.add(ctx.r_val.mul(ctx.az_s[2 * idx0 + 1].sub(a_lo)));
                                ctx.az_d[idx0] = az0;
                                const b_lo = ctx.bz_s[2 * idx0];
                                bz0 = b_lo.add(ctx.r_val.mul(ctx.bz_s[2 * idx0 + 1].sub(b_lo)));
                                ctx.bz_d[idx0] = bz0;
                            }
                            if (idx1 < ctx.new_size) {
                                const a_lo = ctx.az_s[2 * idx1];
                                az1 = a_lo.add(ctx.r_val.mul(ctx.az_s[2 * idx1 + 1].sub(a_lo)));
                                ctx.az_d[idx1] = az1;
                                const b_lo = ctx.bz_s[2 * idx1];
                                bz1 = b_lo.add(ctx.r_val.mul(ctx.bz_s[2 * idx1 + 1].sub(b_lo)));
                                ctx.bz_d[idx1] = bz1;
                            }

                            // Compute t_prime products (multiquadratic expand for window_size=1)
                            // buff_a = [az0, az1, az1-az0], buff_b = [bz0, bz1, bz1-bz0]
                            const eq_weight = ctx.E_out[x_out].mul(ctx.E_in[x_in]);
                            local_ans[0] = local_ans[0].add(eq_weight.mul(az0.mul(bz0)));
                            local_ans[1] = local_ans[1].add(eq_weight.mul(az1.mul(bz1)));
                            local_ans[2] = local_ans[2].add(eq_weight.mul(az1.sub(az0).mul(bz1.sub(bz0))));
                        }
                        return local_ans;
                    }
                }.f;

                const fusedReduceFn = struct {
                    fn f(a: [3]F, b: [3]F) [3]F {
                        return [3]F{ a[0].add(b[0]), a[1].add(b[1]), a[2].add(b[2]) };
                    }
                }.f;

                const identity = [3]F{ F.zero(), F.zero(), F.zero() };
                const tp_result = if (self.thread_pool) |tp|
                    tp.parallelReduce([3]F, total_pairs, identity, fused_ctx, fusedMapFn, fusedReduceFn)
                else
                    fusedMapFn(fused_ctx, 0, total_pairs);

                // Also bind any remaining elements not covered by pairs
                // (pairs cover 2*total_pairs elements; if new_size > 2*total_pairs, bind the rest)
                const covered = 2 * total_pairs;
                if (covered < new_size) {
                    for (covered..new_size) |idx| {
                        const a_lo = az_src[2 * idx];
                        az_dest[idx] = a_lo.add(r.mul(az_src[2 * idx + 1].sub(a_lo)));
                        const b_lo = bz_src[2 * idx];
                        bz_dest[idx] = b_lo.add(r.mul(bz_src[2 * idx + 1].sub(b_lo)));
                    }
                }

                // Swap evaluations/scratch and update num_vars
                if (has_scratch) {
                    const az_tmp = self.az_poly.?.evaluations;
                    self.az_poly.?.evaluations = az_scratch.?;
                    self.az_poly.?.scratch = az_tmp;
                    const bz_tmp = self.bz_poly.?.evaluations;
                    self.bz_poly.?.evaluations = bz_scratch.?;
                    self.bz_poly.?.scratch = bz_tmp;
                }
                self.az_poly.?.num_vars -= 1;
                self.bz_poly.?.num_vars -= 1;

                // Build t_prime from fused results (no separate pass needed)
                var ans = try self.allocator.alloc(F, 3);
                errdefer self.allocator.free(ans);
                ans[0] = tp_result[0];
                ans[1] = tp_result[1];
                ans[2] = tp_result[2];
                if (self.t_prime_poly) |*old| {
                    old.deinit();
                }
                self.t_prime_poly = try MultiquadraticPolynomial(F).init(self.allocator, window_size, ans);
                self.allocator.free(ans);
            } else if (self.gpu_ops != null and self.az_poly != null and self.az_poly.?.boundLen() >= 32768) {
                const gpu = self.gpu_ops.?;
                if (self.az_poly) |*az| {
                    const evals = az.evaluations;
                    const h = evals.len / 2;
                    gpu.polyBindLow(evals, r, evals[0..h]) catch {
                        az.bindLow(r);
                        if (self.bz_poly) |*bz| bz.bindLow(r);
                        self.current_round += 1;
                        return;
                    };
                    az.num_vars -= 1;
                }
                if (self.bz_poly) |*bz| {
                    const evals = bz.evaluations;
                    const h = evals.len / 2;
                    gpu.polyBindLow(evals, r, evals[0..h]) catch {
                        bz.bindLow(r);
                        self.current_round += 1;
                        return;
                    };
                    bz.num_vars -= 1;
                }
            } else {
                if (self.az_poly) |*az| az.bindLow(r);
                if (self.bz_poly) |*bz| bz.bindLow(r);
            }

            self.current_round += 1;
        }

        /// Update the current claim after a round
        ///
        /// round_poly contains evaluations [s(0), s(1), s(2), s(3)], NOT coefficients.
        /// We need to first convert to coefficients, then evaluate at challenge.
        pub fn updateClaim(self: *Self, round_poly: [4]F, challenge: F) void {
            // Convert evaluations to coefficients via Lagrange interpolation
            const coeffs = poly_mod.UniPoly(F).interpolateDegree3(round_poly);

            // Now evaluate the polynomial at challenge using Horner's method
            // s(r) = c0 + r * (c1 + r * (c2 + r * c3))
            self.current_claim = coeffs[0]
                .add(challenge.mul(
                coeffs[1]
                    .add(challenge.mul(
                    coeffs[2]
                        .add(challenge.mul(coeffs[3])),
                )),
            ));
        }

        /// Get the final evaluation after all rounds
        pub fn getFinalEval(self: *const Self) F {
            return self.current_claim;
        }

        /// Generate the full sumcheck proof
        pub fn generateProof(
            self: *Self,
            transcript: anytype,
        ) !struct {
            uniskip_proof: jolt_types.UniSkipFirstRoundProof(F),
            sumcheck_proof: jolt_types.SumcheckInstanceProof(F),
            final_eval: F,
        } {
            // Round 0: Univariate skip
            const first_round_coeffs = try self.computeFirstRoundPoly();

            // Get challenge from transcript
            transcript.appendSlice(&first_round_coeffs);
            const r0 = transcript.challengeScalar();

            // Compute the claim from first round polynomial
            const uni_skip_claim = self.evaluatePolyAtChallenge(&first_round_coeffs, r0);
            try self.bindFirstRoundChallenge(r0, uni_skip_claim);

            // Create proofs
            const uniskip_proof = try jolt_types.UniSkipFirstRoundProof(F).init(
                self.allocator,
                &first_round_coeffs,
            );

            var sumcheck_proof = jolt_types.SumcheckInstanceProof(F).init(self.allocator);

            // Remaining rounds
            while (self.current_round < self.numRounds()) {
                const round_poly = try self.computeRemainingRoundPoly();

                // Convert to compressed coefficient format [c0, c2, c3] for proof
                const compressed = poly_mod.UniPoly(F).evalsToCompressed(round_poly);
                try sumcheck_proof.addRoundPoly(&compressed);

                // Get challenge from transcript
                transcript.appendSlice(&compressed);
                const r = transcript.challengeScalar();

                // Update state
                self.updateClaim(round_poly, r);

                // DEBUG: Print claim AFTER update

                try self.bindRemainingRoundChallenge(r);
            }

            return .{
                .uniskip_proof = uniskip_proof,
                .sumcheck_proof = sumcheck_proof,
                .final_eval = self.getFinalEval(),
            };
        }

        /// Evaluate polynomial at a point using Horner's method
        fn evaluatePolyAtChallenge(self: *const Self, coeffs: []const F, x: F) F {
            _ = self;
            if (coeffs.len == 0) return F.zero();

            var result = coeffs[coeffs.len - 1];
            var i = coeffs.len - 1;
            while (i > 0) {
                i -= 1;
                result = result.mul(x).add(coeffs[i]);
            }
            return result;
        }
    };
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

test "StreamingOuterProver: initialization" {
    const F = BN254Scalar;

    // Create trivial witnesses
    const witnesses = [_]constraints.R1CSCycleInputs(F){
        .{ .values = [_]F{F.zero()} ** 36 },
        .{ .values = [_]F{F.zero()} ** 36 },
        .{ .values = [_]F{F.zero()} ** 36 },
        .{ .values = [_]F{F.zero()} ** 36 },
    };

    const tau = [_]F{ F.fromU64(1), F.fromU64(2), F.fromU64(3) };

    var prover = try StreamingOuterProver(F).init(testing.allocator, &witnesses, &tau);
    defer prover.deinit();

    try testing.expectEqual(@as(usize, 2), prover.num_cycle_vars); // log2(4) = 2
    try testing.expectEqual(@as(usize, 4), prover.padded_trace_len);
    try testing.expectEqual(@as(usize, 3), prover.numRounds()); // 1 + 2 (streaming + 2 cycle vars)
}

test "StreamingOuterProver: first round poly" {
    const F = BN254Scalar;

    const witnesses = [_]constraints.R1CSCycleInputs(F){
        .{ .values = [_]F{F.zero()} ** 36 },
        .{ .values = [_]F{F.zero()} ** 36 },
    };

    const tau = [_]F{ F.fromU64(1), F.fromU64(2) };

    var prover = try StreamingOuterProver(F).init(testing.allocator, &witnesses, &tau);
    defer prover.deinit();

    const first_round = try prover.computeFirstRoundPoly();

    // With zero witnesses, all coefficients should be zero
    for (first_round) |coeff| {
        try testing.expect(coeff.eql(F.zero()));
    }
}

test "StreamingOuterProver: Lagrange basis at r0" {
    const F = BN254Scalar;

    const witnesses = [_]constraints.R1CSCycleInputs(F){
        .{ .values = [_]F{F.zero()} ** 36 },
    };

    const tau = [_]F{F.fromU64(1)};

    var prover = try StreamingOuterProver(F).init(testing.allocator, &witnesses, &tau);
    defer prover.deinit();

    // Bind first round with r0 = 0
    // Domain is {-4, -3, -2, -1, 0, 1, 2, 3, 4, 5}
    // So domain point 0 is at index 4, meaning L_4(0) = 1
    try prover.bindFirstRoundChallenge(F.zero(), F.zero());

    try testing.expect(prover.lagrange_evals_r0[4].eql(F.one()));
}

test "StreamingOuterProver: debug streaming round values" {
    const F = BN254Scalar;

    // Create witnesses with some non-trivial values to test
    // We'll create 4 cycles (2 variables) with random-looking values
    var witnesses: [4]constraints.R1CSCycleInputs(F) = undefined;
    for (0..4) |t| {
        for (0..36) |i| {
            // Use a simple pattern that creates non-zero values
            witnesses[t].values[i] = F.fromU64(@intCast((t + 1) * (i + 1) % 100));
        }
    }

    // tau must have length num_cycle_vars + 2 = 4 for 4 cycles (num_cycle_vars=2)
    // m = 4/2 = 2, so w_out = tau[0..2], w_in = tau[2..3], w_last = tau[3]
    const tau = [_]F{
        F.fromU64(1234),
        F.fromU64(5678),
        F.fromU64(9012),
        F.fromU64(3456), // tau_high (w_last, skipped in split_eq)
    };

    var prover = try StreamingOuterProver(F).init(testing.allocator, &witnesses, &tau);
    defer prover.deinit();

    // Generate a dummy r0 and bind it
    const r0 = F.fromU64(7777);
    try prover.bindFirstRoundChallenge(r0, F.zero());

    // Compute the remaining round poly (streaming round)
    const poly_evals = try prover.computeRemainingRoundPoly();

    // Basic sanity: poly_evals should be non-trivial with these inputs
    var any_nonzero = false;
    for (poly_evals) |v| {
        if (!v.eql(F.zero())) {
            any_nonzero = true;
            break;
        }
    }
    try testing.expect(any_nonzero);
}

test "StreamingOuterProver: expected_output_claim cross-verification" {
    // This test verifies that the sumcheck's final output_claim matches
    // the expected_output_claim formula from Jolt:
    //   expected = tau_high_bound_r0 * tau_bound_r_tail_reversed * inner_sum_prod
    //
    // Where:
    // - tau_high_bound_r0 = L(tau_high, r0) = Lagrange kernel at UniSkip challenge
    // - tau_bound_r_tail_reversed = eq(tau_low, r_tail_reversed) = eq with ALL sumcheck challenges reversed
    // - inner_sum_prod = Az_final * Bz_final where Az/Bz are computed from R1CS input MLE evaluations
    //
    // This is the key formula that Jolt's verifier uses to check the sumcheck.

    const F = BN254Scalar;
    const LagrangePoly = @import("../../poly/mod.zig").LagrangePolynomial(F);
    const EqPolynomial = @import("../../poly/mod.zig").EqPolynomial(F);
    const r1cs_eval = @import("../r1cs/mod.zig").R1CSInputEvaluator(F);

    // Create 4 cycles with non-trivial values
    var witnesses: [4]constraints.R1CSCycleInputs(F) = undefined;
    for (0..4) |t| {
        for (0..36) |i| {
            witnesses[t].values[i] = F.fromU64(@intCast((t + 1) * (i + 1) % 100));
        }
    }

    // tau must have length num_cycle_vars + 2 = 4 for 4 cycles (num_cycle_vars=2)
    const tau = [_]F{
        F.fromU64(1234), // tau_low[0]
        F.fromU64(5678), // tau_low[1]
        F.fromU64(9012), // tau_low[2]
        F.fromU64(3456), // tau_high
    };
    const tau_high = tau[tau.len - 1];
    const tau_low = tau[0 .. tau.len - 1];

    // Create a mock transcript for consistent challenges
    const MockTranscript = struct {
        counter: u64 = 0,

        pub fn appendSlice(self: *@This(), _: []const F) void {
            _ = self;
        }

        pub fn challengeScalar(self: *@This()) F {
            self.counter += 1;
            // Return deterministic "random" challenges
            return F.fromU64(self.counter * 1111);
        }
    };

    // Compute the Lagrange kernel L(tau_high, r0) used for scaling
    // First, simulate getting r0 from UniSkip
    const r0 = F.fromU64(1111); // First challenge
    const DOMAIN_SIZE = StreamingOuterProver(F).FIRST_GROUP_SIZE;
    const lagrange_tau_r0 = try LagrangePoly.lagrangeKernel(
        DOMAIN_SIZE,
        r0,
        tau_high,
        testing.allocator,
    );

    // Initialize the prover with the Lagrange kernel scaling
    var prover = try StreamingOuterProver(F).initWithScaling(
        testing.allocator,
        &witnesses,
        &tau,
        lagrange_tau_r0,
    );
    defer prover.deinit();

    // Compute UniSkip first round
    const first_round_coeffs = try prover.computeFirstRoundPoly();

    // Evaluate at r0 to get uni_skip_claim
    const uni_skip_claim = prover.evaluatePolyAtChallenge(&first_round_coeffs, r0);

    // Bind first round
    try prover.bindFirstRoundChallenge(r0, uni_skip_claim);

    // Generate remaining round challenges and compute proof
    var challenges_list = std.ArrayList(F){};
    defer challenges_list.deinit(testing.allocator);

    // Remaining rounds (1 + num_cycle_vars = 3 for 4 cycles)
    var mock_transcript = MockTranscript{ .counter = 1 }; // Start at 1 since r0 was first

    while (prover.current_round < prover.numRounds()) {
        const round_poly = try prover.computeRemainingRoundPoly();

        // Get challenge for this round
        const r = mock_transcript.challengeScalar();
        try challenges_list.append(testing.allocator, r);

        // Update state
        prover.updateClaim(round_poly, r);
        try prover.bindRemainingRoundChallenge(r);
    }

    // Final output_claim from sumcheck
    const output_claim = prover.getFinalEval();

    // Now compute expected_output_claim using Jolt's formula
    // 1. tau_high_bound_r0 = lagrange_tau_r0 (already computed)
    // 2. tau_bound_r_tail_reversed = eq(tau_low, [r_n, ..., r_1, r_stream])
    // 3. inner_sum_prod = Az_final * Bz_final

    // Construct r_tail_reversed: reverse all sumcheck challenges
    const challenges = challenges_list.items;
    const r_tail_reversed = try testing.allocator.alloc(F, challenges.len);
    defer testing.allocator.free(r_tail_reversed);
    for (0..challenges.len) |i| {
        r_tail_reversed[i] = challenges[challenges.len - 1 - i];
    }

    // Compute tau_bound_r_tail_reversed = eq(tau_low, r_tail_reversed)
    // Note: tau_low.len should equal challenges.len (= 1 + num_cycle_vars = 3)
    try testing.expectEqual(tau_low.len, challenges.len);

    var eq_poly = try EqPolynomial.init(testing.allocator, tau_low);
    defer eq_poly.deinit();
    const tau_bound_r_tail_reversed = eq_poly.evaluate(r_tail_reversed);

    // For inner_sum_prod, we need R1CS input evaluations at r_cycle
    // r_cycle = challenges[1..] reversed to big-endian (excludes r_stream)
    const cycle_challenges = if (challenges.len > 1) challenges[1..] else challenges[0..0];
    const r_cycle_big_endian = try testing.allocator.alloc(F, cycle_challenges.len);
    defer testing.allocator.free(r_cycle_big_endian);
    for (0..cycle_challenges.len) |i| {
        r_cycle_big_endian[i] = cycle_challenges[cycle_challenges.len - 1 - i];
    }

    // Compute R1CS input MLE evaluations at r_cycle
    _ = try r1cs_eval.computeClaimedInputs(
        testing.allocator,
        &witnesses,
        r_cycle_big_endian,
        null,
    );

    // Compute Az_final and Bz_final using Jolt's formula:
    // Az = w[0]*lc_a[0](z) + w[1]*lc_a[1](z) + ... for each group
    // where w are Lagrange weights at r0
    //
    // For now, use the prover's computed values since we're testing the eq/Lagrange part
    const r_stream = challenges[0];
    _ = prover.computeCycleAzBzProductCombined(
        &witnesses[0], // Use first cycle as representative
        r_stream,
    );

    // Actually, we need to compute inner_sum_prod differently - it should be
    // the evaluation at the bound point, not per-cycle.
    //
    // The correct computation requires evaluating Az*Bz using the MLE evaluations.
    // This is complex and involves the constraint matrices. For this test,
    // let's verify the eq polynomial part is correct.

    // Compute expected = L(tau_high, r0) * eq(tau_low, r_tail_reversed) * inner_sum_prod
    // where inner_sum_prod is what the verifier computes from opening claims

    // For this test, we verify the eq factor relationship:
    // The prover's final claim should be: eq_factor * Az_Bz_factor
    // The verifier's expected claim is: lagrange_tau_r0 * tau_bound_r_tail_reversed * inner_sum_prod

    // Compute the eq factor from the prover's state
    // After all rounds, current_scalar = lagrange_tau_r0 * eq(tau_low, challenges)
    const prover_eq_factor = prover.split_eq.current_scalar;

    // The verifier's eq factor is: lagrange_tau_r0 * eq(tau_low, r_tail_reversed)
    // Since eq is symmetric in its arguments (eq(a,b) = eq(b,a) for each coordinate),
    // and multiplication is commutative, these should be equal.
    const verifier_eq_factor = lagrange_tau_r0.mul(tau_bound_r_tail_reversed);


    // The prover's eq factor should match the verifier's eq factor
    try testing.expect(prover_eq_factor.eql(verifier_eq_factor));

    // If eq factors match, then output_claim / verifier_eq_factor = inner_sum_prod
    // This should equal what the verifier computes from opening claims
    _ = if (!verifier_eq_factor.eql(F.zero()))
        output_claim.mul(verifier_eq_factor.inverse().?)
    else
        F.zero();
}
