//! Stage 3 Prefix-Suffix Instance Provers
//!
//! Contains the two prefix-suffix sumcheck instances used in Stage 3:
//! - ShiftPrefixSuffixProver: EqPlusOne prefix-suffix with 4 (P,Q) pairs (degree 2)
//! - RegistersPrefixSuffixProver: Eq prefix-suffix with 1 (P,Q) pair (degree 2)
//!
//! Both use the same two-phase approach:
//! Phase 1: Operate on compact prefix/suffix buffers
//! Phase 2: Transition to materialized witness MLEs when prefix size reaches 2

const std = @import("std");

const zkvm_debug = @import("../debug.zig");
const dbg = zkvm_debug.dbg;
const debug_verbose = zkvm_debug.verbose;

const Allocator = std.mem.Allocator;
const ThreadPool = @import("zolt_pool").ThreadPool;
const pool_helpers = @import("zolt_pool").helpers;
const GpuPolyOps = @import("zolt_arith").gpu.GpuPolyOps;
const poly_mod = @import("zolt_arith").poly;
const field_mod = @import("zolt_arith").field;
const UnreducedProductAccum = field_mod.UnreducedProductAccum;
const FoldedMulU64 = field_mod.FoldedMulU64;
const RawR1CSInputs = @import("../r1cs/evaluators.zig").RawR1CSInputs;

// =============================================================================
// ShiftSumcheck Prefix-Suffix Prover
// =============================================================================
//
// Uses EqPlusOnePrefixSuffixPoly decomposition with 4 (P,Q) pairs:
// - 2 pairs for r_outer (prefix_0/suffix_0, prefix_1/suffix_1)
// - 2 pairs for r_product (prefix_0/suffix_0, prefix_1/suffix_1)
//
// Phase1: First half of rounds use prefix-suffix optimization
// Phase2: Second half of rounds use materialized MLEs
// Transition: When prefix buffer size == 2

pub fn ShiftPrefixSuffixProver(comptime F: type) type {
    return struct {
        const Self = @This();
        const EqPlusOnePrefixSuffixPoly = poly_mod.EqPlusOnePrefixSuffixPoly;
        const EqPolynomial = poly_mod.EqPolynomial;
        const EqPlusOnePolynomial = poly_mod.EqPlusOnePolynomial;

        // P buffers (prefix polynomials) and Q buffers (accumulated witness * suffix)
        // 4 pairs: (P_0_outer, Q_0_outer), (P_1_outer, Q_1_outer),
        //          (P_0_prod, Q_0_prod), (P_1_prod, Q_1_prod)
        P_0_outer: []F,
        Q_0_outer: []F,
        P_1_outer: []F,
        Q_1_outer: []F,
        P_0_prod: []F,
        Q_0_prod: []F,
        P_1_prod: []F,
        Q_1_prod: []F,

        // Gamma powers for batching
        gamma_powers: []const F,

        // Witness MLEs (for final claims computation)
        unexpanded_pc: []F,
        pc: []F,
        is_virtual: []F,
        is_first_in_sequence: []F,
        is_noop: []F,

        // State tracking
        prefix_n_vars: usize,
        suffix_n_vars: usize,
        current_prefix_size: usize,
        current_witness_size: usize, // Track witness MLE size separately
        sumcheck_challenges: std.ArrayListUnmanaged(F),
        in_phase2: bool,

        // Original points (needed for Phase 2 transition)
        r_outer: []const F,
        r_product: []const F,

        // Original trace (needed for Phase 2 witness MLE reconstruction)
        raw_inputs: []const RawR1CSInputs,
        trace_len: usize,

        // Phase2 materialized polynomials (only allocated when transitioning)
        phase2_eq_plus_one_outer: ?[]F,
        phase2_eq_plus_one_prod: ?[]F,

        allocator: Allocator,
        thread_pool: ?*ThreadPool = null,
        gpu_ops: ?*GpuPolyOps = null,

        pub fn init(
            allocator: Allocator,
            raw_inputs: []const RawR1CSInputs,
            trace_len: usize,
            r_outer: []const F,
            r_product: []const F,
            gamma_powers: []const F,
            thread_pool: ?*ThreadPool,
        ) !Self {
            const n_vars = r_outer.len;
            // Split r into hi (first half) and lo (second half)
            // Jolt convention: PREFIX uses r_lo, SUFFIX uses r_hi
            const split_point = n_vars / 2;
            const r_outer_hi = r_outer[0..split_point]; // First half -> used for SUFFIX
            const r_outer_lo = r_outer[split_point..]; // Second half -> used for PREFIX
            const r_prod_hi = r_product[0..split_point];
            const r_prod_lo = r_product[split_point..];

            // Sizes: prefix_size = 2^len(r_lo), suffix_size = 2^len(r_hi)
            const prefix_n_vars = r_outer_lo.len; // = n_vars - split_point
            const suffix_n_vars = r_outer_hi.len; // = split_point
            const prefix_size: usize = @as(usize, 1) << @intCast(prefix_n_vars);
            const suffix_size: usize = @as(usize, 1) << @intCast(suffix_n_vars);

            // Initialize P buffers (prefix polynomials)
            // PREFIX uses r_lo (Jolt convention)
            // P_0 = eq+1(r_lo, j) for j in [0, prefix_size)
            // P_1 = is_max(r_lo) * delta(j=0)
            const P_0_outer = try allocator.alloc(F, prefix_size);
            const P_1_outer = try allocator.alloc(F, prefix_size);
            const P_0_prod = try allocator.alloc(F, prefix_size);
            const P_1_prod = try allocator.alloc(F, prefix_size);

            // Compute eq+1(r_lo, j) - PREFIX uses r_lo (Jolt convention)
            poly_mod.EqPolynomial(F).buildEqPlusOneTableInPlace(r_outer_lo, P_0_outer);
            poly_mod.EqPolynomial(F).buildEqPlusOneTableInPlace(r_prod_lo, P_0_prod);

            // Compute is_max(r_lo) for prefix_1
            // is_max(x) = eq((1,1,...,1), x) = product of x[i]
            var is_max_outer = F.one();
            for (r_outer_lo) |r_i| {
                is_max_outer = is_max_outer.mul(r_i);
            }
            @memset(P_1_outer, F.zero());
            P_1_outer[0] = is_max_outer;

            var is_max_prod = F.one();
            for (r_prod_lo) |r_i| {
                is_max_prod = is_max_prod.mul(r_i);
            }
            @memset(P_1_prod, F.zero());
            P_1_prod[0] = is_max_prod;

            // Compute suffix evaluations (needed for Q buffer construction)
            // SUFFIX uses r_hi (Jolt convention)
            // suffix evaluations are indexed by x_hi in [0, suffix_size)
            const suffix_0_outer = try allocator.alloc(F, suffix_size);
            defer allocator.free(suffix_0_outer);
            const suffix_1_outer = try allocator.alloc(F, suffix_size);
            defer allocator.free(suffix_1_outer);
            const suffix_0_prod = try allocator.alloc(F, suffix_size);
            defer allocator.free(suffix_0_prod);
            const suffix_1_prod = try allocator.alloc(F, suffix_size);
            defer allocator.free(suffix_1_prod);

            // SUFFIX uses r_hi (Jolt convention)
            poly_mod.EqPolynomial(F).buildEqAndEqPlusOneInPlace(r_outer_hi, suffix_0_outer, suffix_1_outer);
            poly_mod.EqPolynomial(F).buildEqAndEqPlusOneInPlace(r_prod_hi, suffix_0_prod, suffix_1_prod);

            // Initialize Q buffers to zero
            const Q_0_outer = try allocator.alloc(F, prefix_size);
            const Q_1_outer = try allocator.alloc(F, prefix_size);
            const Q_0_prod = try allocator.alloc(F, prefix_size);
            const Q_1_prod = try allocator.alloc(F, prefix_size);
            @memset(Q_0_outer, F.zero());
            @memset(Q_1_outer, F.zero());
            @memset(Q_0_prod, F.zero());
            @memset(Q_1_prod, F.zero());

            // Allocate witness MLEs
            const unexpanded_pc = try allocator.alloc(F, trace_len);
            const pc = try allocator.alloc(F, trace_len);
            const is_virtual = try allocator.alloc(F, trace_len);
            const is_first_in_sequence = try allocator.alloc(F, trace_len);
            const is_noop = try allocator.alloc(F, trace_len);

            // Compute Q buffers by accumulating witness * suffix
            // Q[x_lo] = sum over x_hi of: witness(x) * suffix[x_hi]
            // where x = x_lo + (x_hi << prefix_n_vars)
            // Each x_lo writes to disjoint Q indices and disjoint witness indices.
            // Optimization: read from RawR1CSInputs (u64/bool) and use mulU64Unreduced
            // for ~4x fewer multiply instructions per accumulation.
            const ShiftInitCtx = struct {
                raw_inputs: []const RawR1CSInputs,
                suffix_0_outer: []const F,
                suffix_1_outer: []const F,
                suffix_0_prod: []const F,
                suffix_1_prod: []const F,
                Q_0_outer: []F,
                Q_1_outer: []F,
                Q_0_prod: []F,
                Q_1_prod: []F,
                unexpanded_pc: []F,
                pc_arr: []F,
                is_virtual: []F,
                is_first_in_sequence: []F,
                is_noop: []F,
                gamma_powers: []const F,
                prefix_n_vars: usize,
                suffix_size: usize,
                trace_len: usize,
            };
            const shift_init_ctx = ShiftInitCtx{
                .raw_inputs = raw_inputs,
                .suffix_0_outer = suffix_0_outer,
                .suffix_1_outer = suffix_1_outer,
                .suffix_0_prod = suffix_0_prod,
                .suffix_1_prod = suffix_1_prod,
                .Q_0_outer = Q_0_outer,
                .Q_1_outer = Q_1_outer,
                .Q_0_prod = Q_0_prod,
                .Q_1_prod = Q_1_prod,
                .unexpanded_pc = unexpanded_pc,
                .pc_arr = pc,
                .is_virtual = is_virtual,
                .is_first_in_sequence = is_first_in_sequence,
                .is_noop = is_noop,
                .gamma_powers = gamma_powers,
                .prefix_n_vars = prefix_n_vars,
                .suffix_size = suffix_size,
                .trace_len = trace_len,
            };
            const shiftInitWorker = struct {
                fn f(c: ShiftInitCtx, x_lo: usize) void {
                    // Use deferred-reduction accumulators for Q buffers
                    var q_0_outer_acc = UnreducedProductAccum.zero();
                    var q_1_outer_acc = UnreducedProductAccum.zero();
                    // For prod Q buffers we just sum suffix values (when !noop), use FoldedMulU64
                    var q_0_prod_acc = FoldedMulU64.zero();
                    var q_1_prod_acc = FoldedMulU64.zero();

                    for (0..c.suffix_size) |x_hi| {
                        const x = x_lo + (x_hi << @intCast(c.prefix_n_vars));
                        if (x >= c.trace_len) continue;

                        // Read from RawR1CSInputs (u64/bool) instead of F
                        const raw = &c.raw_inputs[x];
                        const upc = raw.u64_values[2]; // UnexpandedPC
                        const pc_val = raw.u64_values[1]; // PC
                        const virt = raw.bool_flags[11]; // FlagVirtualInstruction
                        const first_flag = raw.bool_flags[16]; // FlagIsFirstInSequence
                        const noop = raw.bool_flags[20]; // FlagIsNoop

                        // Fill witness MLE arrays (needed for Phase 2 claims)
                        c.unexpanded_pc[x] = F.fromU64(upc);
                        c.pc_arr[x] = F.fromU64(pc_val);
                        c.is_virtual[x] = if (virt) F.one() else F.zero();
                        c.is_first_in_sequence[x] = if (first_flag) F.one() else F.zero();
                        c.is_noop[x] = if (noop) F.one() else F.zero();

                        // Compute v = gamma[0]*upc + gamma[1]*pc + gamma[2]*virt + gamma[3]*first
                        // using mulU64Unreduced (4 mulq each) + bool conditional-add (0 mulq)
                        var v_accum = field_mod.mulU64Unreduced(c.gamma_powers[0], upc);
                        v_accum.addAssign(field_mod.mulU64Unreduced(c.gamma_powers[1], pc_val));
                        if (virt) v_accum.addBigInt4(c.gamma_powers[2].limbs);
                        if (first_flag) v_accum.addBigInt4(c.gamma_powers[3].limbs);
                        const v = field_mod.reduceMulU64(v_accum);

                        // Accumulate v * suffix into Q buffers (deferred Montgomery)
                        q_0_outer_acc.addAssign(v.mulToProductAccum(c.suffix_0_outer[x_hi]));
                        q_1_outer_acc.addAssign(v.mulToProductAccum(c.suffix_1_outer[x_hi]));

                        // For prod Q buffers: (1-noop) * suffix
                        // When noop=false (the common case), just add the suffix value directly
                        if (!noop) {
                            q_0_prod_acc.addBigInt4(c.suffix_0_prod[x_hi].limbs);
                            q_1_prod_acc.addBigInt4(c.suffix_1_prod[x_hi].limbs);
                        }
                    }

                    c.Q_0_outer[x_lo] = q_0_outer_acc.reduce();
                    c.Q_1_outer[x_lo] = q_1_outer_acc.reduce();
                    c.Q_0_prod[x_lo] = field_mod.reduceMulU64(q_0_prod_acc).mul(c.gamma_powers[4]);
                    c.Q_1_prod[x_lo] = field_mod.reduceMulU64(q_1_prod_acc).mul(c.gamma_powers[4]);
                }
            }.f;

            pool_helpers.parallelForOptional(thread_pool, prefix_size, shift_init_ctx, shiftInitWorker);

            if (comptime debug_verbose) {
                // DEBUG: Print initial witness MLE values
                dbg("\n[ZOLT] SHIFT_INIT: trace_len={d}, prefix_size={d}, suffix_size={d}\n", .{ trace_len, prefix_size, suffix_size });
                dbg("[ZOLT] SHIFT_INIT: unexpanded_pc[0..4] = ", .{});
                for (0..@min(4, trace_len)) |i| {
                    dbg("{any} ", .{unexpanded_pc[i].toBytes()[0..8]});
                }
                dbg("\n", .{});

                // DEBUG: Verify grand sum = Σ P[j]*Q[j]
                var grand_sum = F.zero();
                for (0..prefix_size) |j| {
                    grand_sum = grand_sum.add(P_0_outer[j].mul(Q_0_outer[j]));
                    grand_sum = grand_sum.add(P_1_outer[j].mul(Q_1_outer[j]));
                    grand_sum = grand_sum.add(P_0_prod[j].mul(Q_0_prod[j]));
                    grand_sum = grand_sum.add(P_1_prod[j].mul(Q_1_prod[j]));
                }
                dbg("[ZOLT] SHIFT_INIT: grand_sum(P*Q) = {{ {any} }}\n", .{grand_sum.toBytes()});
                dbg("[ZOLT] SHIFT_INIT: r_outer[0] = {{ {any} }}\n", .{r_outer[0].toBytes()[0..8]});
                dbg("[ZOLT] SHIFT_INIT: r_outer[last] = {{ {any} }}\n", .{r_outer[r_outer.len - 1].toBytes()[0..8]});
            }

            return Self{
                .P_0_outer = P_0_outer,
                .Q_0_outer = Q_0_outer,
                .P_1_outer = P_1_outer,
                .Q_1_outer = Q_1_outer,
                .P_0_prod = P_0_prod,
                .Q_0_prod = Q_0_prod,
                .P_1_prod = P_1_prod,
                .Q_1_prod = Q_1_prod,
                .gamma_powers = gamma_powers,
                .unexpanded_pc = unexpanded_pc,
                .pc = pc,
                .is_virtual = is_virtual,
                .is_first_in_sequence = is_first_in_sequence,
                .is_noop = is_noop,
                .prefix_n_vars = prefix_n_vars,
                .suffix_n_vars = suffix_n_vars,
                .current_prefix_size = prefix_size,
                .current_witness_size = trace_len,
                .sumcheck_challenges = .{},
                .in_phase2 = false,
                .r_outer = r_outer,
                .r_product = r_product,
                .raw_inputs = raw_inputs,
                .trace_len = trace_len,
                .phase2_eq_plus_one_outer = null,
                .phase2_eq_plus_one_prod = null,
                .allocator = allocator,
                .thread_pool = thread_pool,
            };
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.P_0_outer);
            self.allocator.free(self.Q_0_outer);
            self.allocator.free(self.P_1_outer);
            self.allocator.free(self.Q_1_outer);
            self.allocator.free(self.P_0_prod);
            self.allocator.free(self.Q_0_prod);
            self.allocator.free(self.P_1_prod);
            self.allocator.free(self.Q_1_prod);
            self.allocator.free(self.unexpanded_pc);
            self.allocator.free(self.pc);
            self.allocator.free(self.is_virtual);
            self.allocator.free(self.is_first_in_sequence);
            self.allocator.free(self.is_noop);
            self.sumcheck_challenges.deinit(self.allocator);
            if (self.phase2_eq_plus_one_outer) |p| self.allocator.free(p);
            if (self.phase2_eq_plus_one_prod) |p| self.allocator.free(p);
        }

        /// Compute round evaluations [p(0), p(1), p(2)] for degree-2 polynomial
        pub fn computeRoundEvals(self: *Self, previous_claim: F) [3]F {
            if (self.in_phase2) {
                return self.computeRoundEvalsPhase2(previous_claim);
            } else {
                return self.computeRoundEvalsPhase1(previous_claim);
            }
        }

        fn computeRoundEvalsPhase1(self: *Self, previous_claim: F) [3]F {
            const half = self.current_prefix_size / 2;

            const P1Ctx = struct {
                P0o: []const F,
                Q0o: []const F,
                P1o: []const F,
                Q1o: []const F,
                P0p: []const F,
                Q0p: []const F,
                P1p: []const F,
                Q1p: []const F,
            };
            const ctx = P1Ctx{
                .P0o = self.P_0_outer,
                .Q0o = self.Q_0_outer,
                .P1o = self.P_1_outer,
                .Q1o = self.Q_1_outer,
                .P0p = self.P_0_prod,
                .Q0p = self.Q_0_prod,
                .P1p = self.P_1_prod,
                .Q1p = self.Q_1_prod,
            };

            const mapFn = struct {
                fn f(c: P1Ctx, start: usize, end: usize) [3]F {
                    @setEvalBranchQuota(10000);
                    const use_deferred = comptime @hasDecl(F, "mulToProductAccum");
                    if (use_deferred) {
                        var accum: [3]UnreducedProductAccum = .{ UnreducedProductAccum.zero(), UnreducedProductAccum.zero(), UnreducedProductAccum.zero() };
                        const all_pairs = [4][2][]const F{
                            .{ c.P0o, c.Q0o }, .{ c.P1o, c.Q1o },
                            .{ c.P0p, c.Q0p }, .{ c.P1p, c.Q1p },
                        };
                        for (all_pairs) |pair| {
                            const P = pair[0];
                            const Q = pair[1];
                            for (start..end) |i| {
                                const p0 = P[2 * i];
                                const p1 = P[2 * i + 1];
                                const q0 = Q[2 * i];
                                const q1 = Q[2 * i + 1];
                                const p2 = p1.add(p1).sub(p0);
                                const q2 = q1.add(q1).sub(q0);
                                accum[0].addAssign(p0.mulToProductAccum(q0));
                                accum[1].addAssign(p1.mulToProductAccum(q1));
                                accum[2].addAssign(p2.mulToProductAccum(q2));
                            }
                        }
                        return .{ accum[0].reduce(), accum[1].reduce(), accum[2].reduce() };
                    } else {
                        var local: [3]F = .{ F.zero(), F.zero(), F.zero() };
                        const all_pairs = [4][2][]const F{
                            .{ c.P0o, c.Q0o }, .{ c.P1o, c.Q1o },
                            .{ c.P0p, c.Q0p }, .{ c.P1p, c.Q1p },
                        };
                        for (all_pairs) |pair| {
                            const P = pair[0];
                            const Q = pair[1];
                            for (start..end) |i| {
                                const p0 = P[2 * i];
                                const p1 = P[2 * i + 1];
                                const q0 = Q[2 * i];
                                const q1 = Q[2 * i + 1];
                                const p2 = p1.add(p1).sub(p0);
                                const q2 = q1.add(q1).sub(q0);
                                local[0] = local[0].add(p0.mul(q0));
                                local[1] = local[1].add(p1.mul(q1));
                                local[2] = local[2].add(p2.mul(q2));
                            }
                        }
                        return local;
                    }
                }
            }.f;

            const reduceFn = struct {
                fn f(a: [3]F, b: [3]F) [3]F {
                    return .{ a[0].add(b[0]), a[1].add(b[1]), a[2].add(b[2]) };
                }
            }.f;

            const identity = [3]F{ F.zero(), F.zero(), F.zero() };
            const evals = pool_helpers.parallelReduceOptional([3]F, self.thread_pool, half, identity, ctx, mapFn, reduceFn);

            // DEBUG: Verify sumcheck invariant p(0) + p(1) = previous_claim
            if (comptime debug_verbose) {
                const computed_sum = evals[0].add(evals[1]);
                if (!computed_sum.eql(previous_claim)) {
                    dbg("[ZOLT] SHIFT INVARIANT FAIL: p(0)+p(1) = {{ {any} }}, expected = {{ {any} }}\n", .{ computed_sum.toBytes(), previous_claim.toBytes() });
                }
            }

            return evals;
        }

        fn computeRoundEvalsPhase2(self: *Self, previous_claim: F) [3]F {
            // Phase2: Use materialized eq+1 polynomials with witness MLEs
            const eq_outer = self.phase2_eq_plus_one_outer.?;
            const eq_prod = self.phase2_eq_plus_one_prod.?;
            // CRITICAL FIX: Use current_witness_size, NOT eq_outer.len!
            // eq_outer.len is the original suffix_size allocation, but current_witness_size
            // shrinks after each bindPhase2 call.
            const half = self.current_witness_size / 2;

            var evals: [2]F = .{ F.zero(), F.zero() };

            // Debug: Print arrays when they're size 2 (last Phase 2 round)
            if (comptime debug_verbose) {
                if (self.current_witness_size == 2) {
                    dbg("[ZOLT] SHIFT_LAST_ROUND: eq_outer[0] = {{ {any} }}\n", .{eq_outer[0].toBytes()});
                    dbg("[ZOLT] SHIFT_LAST_ROUND: eq_outer[1] = {{ {any} }}\n", .{eq_outer[1].toBytes()});
                    dbg("[ZOLT] SHIFT_LAST_ROUND: eq_prod[0] = {{ {any} }}\n", .{eq_prod[0].toBytes()});
                    dbg("[ZOLT] SHIFT_LAST_ROUND: eq_prod[1] = {{ {any} }}\n", .{eq_prod[1].toBytes()});
                    dbg("[ZOLT] SHIFT_LAST_ROUND: upc[0] = {{ {any} }}\n", .{self.unexpanded_pc[0].toBytes()});
                    dbg("[ZOLT] SHIFT_LAST_ROUND: upc[1] = {{ {any} }}\n", .{self.unexpanded_pc[1].toBytes()});
                    dbg("[ZOLT] SHIFT_LAST_ROUND: noop[0] = {{ {any} }}\n", .{self.is_noop[0].toBytes()});
                    dbg("[ZOLT] SHIFT_LAST_ROUND: noop[1] = {{ {any} }}\n", .{self.is_noop[1].toBytes()});
                    dbg("[ZOLT] SHIFT_LAST_ROUND: previous_claim = {{ {any} }}\n", .{previous_claim.toBytes()});
                }
            }

            for (0..half) |j| {
                const eq_out_0 = eq_outer[2 * j];
                const eq_out_1 = eq_outer[2 * j + 1];
                const eq_prod_0 = eq_prod[2 * j];
                const eq_prod_1 = eq_prod[2 * j + 1];

                const upc_0 = self.unexpanded_pc[2 * j];
                const upc_1 = self.unexpanded_pc[2 * j + 1];
                const pc_0 = self.pc[2 * j];
                const pc_1 = self.pc[2 * j + 1];
                const virt_0 = self.is_virtual[2 * j];
                const virt_1 = self.is_virtual[2 * j + 1];
                const first_0 = self.is_first_in_sequence[2 * j];
                const first_1 = self.is_first_in_sequence[2 * j + 1];
                const noop_0 = self.is_noop[2 * j];
                const noop_1 = self.is_noop[2 * j + 1];

                // Extrapolate to X=2
                const eq_out_2 = eq_out_1.add(eq_out_1).sub(eq_out_0);
                const eq_prod_2 = eq_prod_1.add(eq_prod_1).sub(eq_prod_0);
                const upc_2 = upc_1.add(upc_1).sub(upc_0);
                const pc_2 = pc_1.add(pc_1).sub(pc_0);
                const virt_2 = virt_1.add(virt_1).sub(virt_0);
                const first_2 = first_1.add(first_1).sub(first_0);
                const noop_2 = noop_1.add(noop_1).sub(noop_0);

                // Compute f at X=0
                const val_0 = upc_0.add(self.gamma_powers[1].mul(pc_0))
                    .add(self.gamma_powers[2].mul(virt_0))
                    .add(self.gamma_powers[3].mul(first_0));
                const term1_0 = eq_out_0.mul(val_0);
                const term2_0 = self.gamma_powers[4].mul(F.one().sub(noop_0)).mul(eq_prod_0);
                const f_0 = term1_0.add(term2_0);

                // Compute f at X=2
                const val_2 = upc_2.add(self.gamma_powers[1].mul(pc_2))
                    .add(self.gamma_powers[2].mul(virt_2))
                    .add(self.gamma_powers[3].mul(first_2));
                const term1_2 = eq_out_2.mul(val_2);
                const term2_2 = self.gamma_powers[4].mul(F.one().sub(noop_2)).mul(eq_prod_2);
                const f_2 = term1_2.add(term2_2);

                evals[0] = evals[0].add(f_0);
                evals[1] = evals[1].add(f_2);

                // Debug: last round details
                if (comptime debug_verbose) {
                    if (eq_outer.len == 2) {
                        dbg("[ZOLT] SHIFT_LAST_ROUND: f(0) = {{ {any} }}\n", .{f_0.toBytes()});
                        dbg("[ZOLT] SHIFT_LAST_ROUND: f(2) = {{ {any} }}\n", .{f_2.toBytes()});

                        // Compute f(1) = eq_out_1 * val_1 + γ⁴*(1-noop_1)*eq_prod_1
                        const val_1 = upc_1.add(self.gamma_powers[1].mul(pc_1))
                            .add(self.gamma_powers[2].mul(virt_1))
                            .add(self.gamma_powers[3].mul(first_1));
                        const actual_f1 = eq_out_1.mul(val_1)
                            .add(self.gamma_powers[4].mul(F.one().sub(noop_1)).mul(eq_prod_1));
                        const derived_f1 = previous_claim.sub(f_0);
                        dbg("[ZOLT] SHIFT_LAST_ROUND: actual_f(1) = {{ {any} }}\n", .{actual_f1.toBytes()});
                        dbg("[ZOLT] SHIFT_LAST_ROUND: derived_f(1) = {{ {any} }}\n", .{derived_f1.toBytes()});
                        dbg("[ZOLT] SHIFT_LAST_ROUND: f(1) match = {}\n", .{actual_f1.eql(derived_f1)});
                        dbg("[ZOLT] SHIFT_LAST_ROUND: f(0)+actual_f(1) = {{ {any} }}\n", .{f_0.add(actual_f1).toBytes()});
                        dbg("[ZOLT] SHIFT_LAST_ROUND: claim = {{ {any} }}\n", .{previous_claim.toBytes()});
                        dbg("[ZOLT] SHIFT_LAST_ROUND: f(0)+actual_f(1)==claim: {}\n", .{f_0.add(actual_f1).eql(previous_claim)});
                    }
                }
            }

            const p_1 = previous_claim.sub(evals[0]);
            return [3]F{ evals[0], p_1, evals[1] };
        }

        /// Bind the prover at challenge r_j
        pub fn bind(self: *Self, r_j: F) void {
            if (self.in_phase2) {
                self.bindPhase2(r_j);
            } else {
                // Check if we should transition to Phase2
                if (self.shouldTransitionToPhase2()) {
                    // transitionToPhase2 handles appending the challenge itself
                    self.transitionToPhase2(r_j);
                } else {
                    // Append challenge for Phase 1 binding
                    self.sumcheck_challenges.append(self.allocator, r_j) catch unreachable;
                    self.bindPhase1(r_j);
                }
            }
        }

        fn shouldTransitionToPhase2(self: *Self) bool {
            // Transition when prefix size is 2 (log2 == 1)
            return std.math.log2_int(usize, self.current_prefix_size) == 1;
        }

        fn bindPhase1(self: *Self, r_j: F) void {
            const new_prefix_size = self.current_prefix_size / 2;

            const BindP1Ctx = struct {
                slices: [8][]F,
                r: F,
                n: usize,
            };
            const bctx = BindP1Ctx{
                .slices = .{
                    self.P_0_outer, self.Q_0_outer,
                    self.P_1_outer, self.Q_1_outer,
                    self.P_0_prod,  self.Q_0_prod,
                    self.P_1_prod,  self.Q_1_prod,
                },
                .r = r_j,
                .n = new_prefix_size,
            };
            const bindOneFn = struct {
                fn f(c: BindP1Ctx, idx: usize) void {
                    const arr = c.slices[idx];
                    for (0..c.n) |i| {
                        arr[i] = arr[2 * i].add(c.r.mul(arr[2 * i + 1].sub(arr[2 * i])));
                    }
                }
            }.f;

            if (self.gpu_ops) |gpu| {
                if (new_prefix_size >= 16384) {
                    for (bctx.slices) |arr| {
                        gpu.polyBindLow(arr[0 .. new_prefix_size * 2], r_j, arr[0..new_prefix_size]) catch {
                            for (0..new_prefix_size) |i| {
                                arr[i] = arr[2 * i].add(r_j.montgomeryMul(arr[2 * i + 1].sub(arr[2 * i])));
                            }
                        };
                    }
                } else {
                    for (0..8) |idx| bindOneFn(bctx, idx);
                }
            } else {
                pool_helpers.parallelForOptional(self.thread_pool, 8, bctx, bindOneFn);
            }

            self.current_prefix_size = new_prefix_size;
            // Note: Witness MLEs are NOT bound in Phase 1. They are reconstructed
            // from scratch during transitionToPhase2 using Eq(r_prefix, i) weighting.
        }

        fn transitionToPhase2(self: *Self, r_j: F) void {
            // The transition happens AFTER binding with the final Phase 1 challenge r_j
            // First, bind the P/Q buffers one last time (they become size 1)
            const new_prefix_size = self.current_prefix_size / 2;
            for (0..new_prefix_size) |i| {
                self.P_0_outer[i] = self.P_0_outer[2 * i].add(r_j.mul(self.P_0_outer[2 * i + 1].sub(self.P_0_outer[2 * i])));
                self.Q_0_outer[i] = self.Q_0_outer[2 * i].add(r_j.mul(self.Q_0_outer[2 * i + 1].sub(self.Q_0_outer[2 * i])));
                self.P_1_outer[i] = self.P_1_outer[2 * i].add(r_j.mul(self.P_1_outer[2 * i + 1].sub(self.P_1_outer[2 * i])));
                self.Q_1_outer[i] = self.Q_1_outer[2 * i].add(r_j.mul(self.Q_1_outer[2 * i + 1].sub(self.Q_1_outer[2 * i])));
                self.P_0_prod[i] = self.P_0_prod[2 * i].add(r_j.mul(self.P_0_prod[2 * i + 1].sub(self.P_0_prod[2 * i])));
                self.Q_0_prod[i] = self.Q_0_prod[2 * i].add(r_j.mul(self.Q_0_prod[2 * i + 1].sub(self.Q_0_prod[2 * i])));
                self.P_1_prod[i] = self.P_1_prod[2 * i].add(r_j.mul(self.P_1_prod[2 * i + 1].sub(self.P_1_prod[2 * i])));
                self.Q_1_prod[i] = self.Q_1_prod[2 * i].add(r_j.mul(self.Q_1_prod[2 * i + 1].sub(self.Q_1_prod[2 * i])));
            }
            self.current_prefix_size = new_prefix_size;

            // Store final challenge
            self.sumcheck_challenges.append(self.allocator, r_j) catch unreachable;
            self.in_phase2 = true;

            // Collect all Phase 1 challenges as r_prefix
            // CRITICAL: Jolt converts from LITTLE_ENDIAN (sumcheck order) to BIG_ENDIAN (MLE indexing)
            // by reversing the challenges array. We must do the same.
            // sumcheck_challenges[0] = first round = binds LSB variable
            // After reversal: r_prefix_be[0] = last challenge = MSB variable
            const r_prefix_le = self.sumcheck_challenges.items;
            const r_prefix_be = self.allocator.alloc(F, r_prefix_le.len) catch unreachable;
            defer self.allocator.free(r_prefix_be);
            for (0..r_prefix_le.len) |i| {
                r_prefix_be[i] = r_prefix_le[r_prefix_le.len - 1 - i];
            }
            const n_remaining_rounds = self.suffix_n_vars;
            const suffix_size: usize = @as(usize, 1) << @intCast(n_remaining_rounds);

            if (comptime debug_verbose) {
                dbg("\n[ZOLT] SHIFT_PHASE2_START: n_remaining_rounds={d}, suffix_size={d}\n", .{ n_remaining_rounds, suffix_size });
                dbg("[ZOLT] SHIFT_PHASE2_START: r_prefix_be.len={d}\n", .{r_prefix_be.len});
            }

            // =====================================================================
            // Step 1: Regenerate prefix-suffix decomposition from original r_outer/r_product
            // and evaluate prefix at r_prefix to get scalar values
            // =====================================================================

            // For r_outer: split into hi and lo parts (Jolt convention)
            // r_hi (first half) -> used for SUFFIX
            // r_lo (second half) -> used for PREFIX
            // split_point = suffix_n_vars (original n_vars / 2)
            const r_outer_hi = self.r_outer[0..self.suffix_n_vars]; // For SUFFIX
            const r_outer_lo = self.r_outer[self.suffix_n_vars..]; // For PREFIX

            // Regenerate prefix polynomials for r_outer
            // PREFIX uses r_lo (Jolt convention)
            const prefix_size_outer: usize = @as(usize, 1) << @intCast(r_outer_lo.len);
            const prefix_0_outer = self.allocator.alloc(F, prefix_size_outer) catch unreachable;
            defer self.allocator.free(prefix_0_outer);
            poly_mod.EqPolynomial(F).buildEqPlusOneTableInPlace(r_outer_lo, prefix_0_outer);

            const prefix_1_outer = self.allocator.alloc(F, prefix_size_outer) catch unreachable;
            defer self.allocator.free(prefix_1_outer);
            @memset(prefix_1_outer, F.zero());
            var is_max_outer = F.one();
            for (r_outer_lo) |r_i| {
                is_max_outer = is_max_outer.mul(r_i);
            }
            prefix_1_outer[0] = is_max_outer;

            // Evaluate prefix polynomials at r_prefix
            // NOTE: evaluateMle binds point[0] to the LSB of the table index.
            // The prefix table is in BIG_ENDIAN order (buildEqPlusOneTableInPlace uses BIG_ENDIAN bits).
            // So evaluateMle(table, [r_0, r_1, r_2]) binds r_0→LSB, r_1→mid, r_2→MSB.
            // This matches Jolt's MultilinearPolynomial::evaluate(r_prefix_BE) which also
            // evaluates Σ table[i] * Eq(r_prefix_BE, i_BE), since eq_evals uses BIG_ENDIAN.
            // r_prefix_le = [r_0, r_1, r_2] (LowToHigh: LSB challenge first)
            const prefix_0_eval_outer = evaluateMle(prefix_0_outer, r_prefix_le);
            const prefix_1_eval_outer = evaluateMle(prefix_1_outer, r_prefix_le);

            // DEBUG: Print prefix evaluation details
            if (comptime debug_verbose) {
                dbg("[ZOLT] SHIFT_PREFIX: r_prefix_be[0] = {any}\n", .{r_prefix_be[0].toBytes()[0..8]});
                dbg("[ZOLT] SHIFT_PREFIX: r_prefix_be[last] = {any}\n", .{r_prefix_be[r_prefix_be.len - 1].toBytes()[0..8]});
                dbg("[ZOLT] SHIFT_PREFIX: prefix_0_eval_outer = {any}\n", .{prefix_0_eval_outer.toBytes()[0..8]});
                dbg("[ZOLT] SHIFT_PREFIX: prefix_1_eval_outer = {any}\n", .{prefix_1_eval_outer.toBytes()[0..8]});

                // DEBUG: Print r_outer_hi and r_outer_lo (the fixed points from Stage 1)
                // Using BE for comparison with STAGE1_CHALLENGES output
                dbg("[ZOLT] SHIFT_PREFIX: r_outer_hi[0] (BE) = {any}\n", .{r_outer_hi[0].toBytesBE()[0..8]});
                dbg("[ZOLT] SHIFT_PREFIX: r_outer_hi[last] (BE) = {any}\n", .{r_outer_hi[r_outer_hi.len - 1].toBytesBE()[0..8]});
                dbg("[ZOLT] SHIFT_PREFIX: r_outer_lo[0] (BE) = {any}\n", .{r_outer_lo[0].toBytesBE()[0..8]});
                dbg("[ZOLT] SHIFT_PREFIX: r_outer_lo[last] (BE) = {any}\n", .{r_outer_lo[r_outer_lo.len - 1].toBytesBE()[0..8]});
            }

            // Regenerate suffix polynomials for r_outer
            // SUFFIX uses r_hi (Jolt convention)
            const suffix_0_outer = self.allocator.alloc(F, suffix_size) catch unreachable;
            defer self.allocator.free(suffix_0_outer);
            const suffix_1_outer = self.allocator.alloc(F, suffix_size) catch unreachable;
            defer self.allocator.free(suffix_1_outer);
            poly_mod.EqPolynomial(F).buildEqAndEqPlusOneInPlace(r_outer_hi, suffix_0_outer, suffix_1_outer);

            // Same for r_product
            const r_prod_hi = self.r_product[0..self.suffix_n_vars]; // For SUFFIX
            const r_prod_lo = self.r_product[self.suffix_n_vars..]; // For PREFIX

            // PREFIX uses r_lo (Jolt convention)
            const prefix_size_prod: usize = @as(usize, 1) << @intCast(r_prod_lo.len);
            const prefix_0_prod = self.allocator.alloc(F, prefix_size_prod) catch unreachable;
            defer self.allocator.free(prefix_0_prod);
            poly_mod.EqPolynomial(F).buildEqPlusOneTableInPlace(r_prod_lo, prefix_0_prod);

            const prefix_1_prod = self.allocator.alloc(F, prefix_size_prod) catch unreachable;
            defer self.allocator.free(prefix_1_prod);
            @memset(prefix_1_prod, F.zero());
            var is_max_prod = F.one();
            for (r_prod_lo) |r_i| {
                is_max_prod = is_max_prod.mul(r_i);
            }
            prefix_1_prod[0] = is_max_prod;

            // Use r_prefix_le for MLE evaluation (LSB challenge first to match evaluateMle convention)
            const prefix_0_eval_prod = evaluateMle(prefix_0_prod, r_prefix_le);
            const prefix_1_eval_prod = evaluateMle(prefix_1_prod, r_prefix_le);

            // SUFFIX uses r_hi (Jolt convention)
            const suffix_0_prod = self.allocator.alloc(F, suffix_size) catch unreachable;
            defer self.allocator.free(suffix_0_prod);
            const suffix_1_prod = self.allocator.alloc(F, suffix_size) catch unreachable;
            defer self.allocator.free(suffix_1_prod);
            poly_mod.EqPolynomial(F).buildEqAndEqPlusOneInPlace(r_prod_hi, suffix_0_prod, suffix_1_prod);

            // =====================================================================
            // Step 2: Construct eq+1(r_outer, (r_prefix, j)) for all j in suffix domain
            // eq+1(r, (r_prefix, j)) = prefix_0_eval * suffix_0[j] + prefix_1_eval * suffix_1[j]
            // =====================================================================

            self.phase2_eq_plus_one_outer = self.allocator.alloc(F, suffix_size) catch unreachable;
            self.phase2_eq_plus_one_prod = self.allocator.alloc(F, suffix_size) catch unreachable;

            // Parallelize eq+1 materialization
            const EqMatCtx = struct {
                eq_outer: []F,
                eq_prod: []F,
                s0o: []const F,
                s1o: []const F,
                s0p: []const F,
                s1p: []const F,
                p0o: F,
                p1o: F,
                p0p: F,
                p1p: F,
            };
            const eq_mat_ctx = EqMatCtx{
                .eq_outer = self.phase2_eq_plus_one_outer.?,
                .eq_prod = self.phase2_eq_plus_one_prod.?,
                .s0o = suffix_0_outer,
                .s1o = suffix_1_outer,
                .s0p = suffix_0_prod,
                .s1p = suffix_1_prod,
                .p0o = prefix_0_eval_outer,
                .p1o = prefix_1_eval_outer,
                .p0p = prefix_0_eval_prod,
                .p1p = prefix_1_eval_prod,
            };
            const eqMatWorker = struct {
                fn f(c: EqMatCtx, j: usize) void {
                    c.eq_outer[j] = c.p0o.mul(c.s0o[j]).add(c.p1o.mul(c.s1o[j]));
                    c.eq_prod[j] = c.p0p.mul(c.s0p[j]).add(c.p1p.mul(c.s1p[j]));
                }
            }.f;
            pool_helpers.parallelForOptional(self.thread_pool, suffix_size, eq_mat_ctx, eqMatWorker);

            // =====================================================================
            // Step 3: Construct witness MLEs by summing over prefix domain weighted by Eq(r_prefix, j)
            // poly[j] = Σ_i Eq(r_prefix, i) * witness[i * suffix_size + j]
            // =====================================================================

            // Compute Eq(r_prefix, i) for all i in prefix domain (using BIG_ENDIAN version)
            const prefix_domain_size: usize = @as(usize, 1) << @intCast(r_prefix_be.len);
            const eq_evals = self.allocator.alloc(F, prefix_domain_size) catch unreachable;
            defer self.allocator.free(eq_evals);
            computeEqEvals(self.allocator, r_prefix_be, eq_evals) catch unreachable;

            // Reallocate witness MLEs to suffix_size
            self.allocator.free(self.unexpanded_pc);
            self.allocator.free(self.pc);
            self.allocator.free(self.is_virtual);
            self.allocator.free(self.is_first_in_sequence);
            self.allocator.free(self.is_noop);

            self.unexpanded_pc = self.allocator.alloc(F, suffix_size) catch unreachable;
            self.pc = self.allocator.alloc(F, suffix_size) catch unreachable;
            self.is_virtual = self.allocator.alloc(F, suffix_size) catch unreachable;
            self.is_first_in_sequence = self.allocator.alloc(F, suffix_size) catch unreachable;
            self.is_noop = self.allocator.alloc(F, suffix_size) catch unreachable;

            @memset(self.unexpanded_pc, F.zero());
            @memset(self.pc, F.zero());
            @memset(self.is_virtual, F.zero());
            @memset(self.is_first_in_sequence, F.zero());
            @memset(self.is_noop, F.zero());

            // Sum over prefix domain (parallelized — each j is independent)
            // Optimization: use mulU64Unreduced for u64 witnesses and conditional-add for booleans
            const WitReconCtx = struct {
                raw_inputs_ptr: []const RawR1CSInputs,
                eq_ev: []const F,
                upc_out: []F,
                pc_out: []F,
                virt_out: []F,
                first_out: []F,
                noop_out: []F,
                prefix_dom_size: usize,
                tl: usize,
            };
            const wit_ctx = WitReconCtx{
                .raw_inputs_ptr = self.raw_inputs,
                .eq_ev = eq_evals,
                .upc_out = self.unexpanded_pc,
                .pc_out = self.pc,
                .virt_out = self.is_virtual,
                .first_out = self.is_first_in_sequence,
                .noop_out = self.is_noop,
                .prefix_dom_size = prefix_domain_size,
                .tl = self.trace_len,
            };
            const witReconWorker = struct {
                fn f(c: WitReconCtx, j: usize) void {
                    // Use FoldedMulU64 accumulators for u64 witnesses (4 mulq each)
                    var upc_acc = FoldedMulU64.zero();
                    var pc_acc = FoldedMulU64.zero();
                    // Use FoldedMulU64 for booleans too (conditional addBigInt4, 0 mulq)
                    var virt_acc = FoldedMulU64.zero();
                    var first_acc = FoldedMulU64.zero();
                    var noop_acc = FoldedMulU64.zero();

                    for (0..c.prefix_dom_size) |i| {
                        const trace_idx = j * c.prefix_dom_size + i;
                        if (trace_idx >= c.tl) continue;

                        const raw = &c.raw_inputs_ptr[trace_idx];
                        const eq_eval = c.eq_ev[i];

                        // u64 witnesses: mulU64Unreduced (4 mulq each)
                        upc_acc.addAssign(field_mod.mulU64Unreduced(eq_eval, raw.u64_values[2])); // UnexpandedPC
                        pc_acc.addAssign(field_mod.mulU64Unreduced(eq_eval, raw.u64_values[1])); // PC

                        // Boolean witnesses: conditional add (0 mulq)
                        if (raw.bool_flags[11]) virt_acc.addBigInt4(eq_eval.limbs); // FlagVirtualInstruction
                        if (raw.bool_flags[16]) first_acc.addBigInt4(eq_eval.limbs); // FlagIsFirstInSequence
                        if (raw.bool_flags[20]) noop_acc.addBigInt4(eq_eval.limbs); // FlagIsNoop
                    }

                    c.upc_out[j] = field_mod.reduceMulU64(upc_acc);
                    c.pc_out[j] = field_mod.reduceMulU64(pc_acc);
                    c.virt_out[j] = field_mod.reduceMulU64(virt_acc);
                    c.first_out[j] = field_mod.reduceMulU64(first_acc);
                    c.noop_out[j] = field_mod.reduceMulU64(noop_acc);
                }
            }.f;
            pool_helpers.parallelForOptional(self.thread_pool, suffix_size, wit_ctx, witReconWorker);

            self.current_witness_size = suffix_size;

            if (comptime debug_verbose) {
                dbg("[ZOLT] SHIFT_PHASE2_START: eq+1_outer[0] = {{ {any} }}\n", .{self.phase2_eq_plus_one_outer.?[0].toBytes()[0..8]});
                dbg("[ZOLT] SHIFT_PHASE2_START: unexpanded_pc[0] = {{ {any} }}\n", .{self.unexpanded_pc[0].toBytes()[0..8]});

                // CRITICAL VERIFICATION: Compute Σ_j f(j) using Phase 2 data
                // This should equal the current_shift_claim at the start of Phase 2
                {
                    var phase2_total_sum = F.zero();
                    for (0..suffix_size) |j| {
                        const eq_out = self.phase2_eq_plus_one_outer.?[j];
                        const eq_prod_v = self.phase2_eq_plus_one_prod.?[j];
                        const upc = self.unexpanded_pc[j];
                        const pc_val = self.pc[j];
                        const virt = self.is_virtual[j];
                        const first = self.is_first_in_sequence[j];
                        const noop = self.is_noop[j];

                        const val = upc.add(self.gamma_powers[1].mul(pc_val))
                            .add(self.gamma_powers[2].mul(virt))
                            .add(self.gamma_powers[3].mul(first));
                        const term1 = eq_out.mul(val);
                        const term2 = self.gamma_powers[4].mul(F.one().sub(noop)).mul(eq_prod_v);
                        phase2_total_sum = phase2_total_sum.add(term1).add(term2);
                    }
                    dbg("[ZOLT] SHIFT_PHASE2_VERIFY: phase2_total_sum = {{ {any} }}\n", .{phase2_total_sum.toBytes()});
                    dbg("[ZOLT] SHIFT_PHASE2_VERIFY: (compare with current_shift_claim at Phase2 start)\n", .{});
                }

                // DEBUG: Verify eq+1_outer initialization by direct evaluation
                {
                    const full_y = self.allocator.alloc(F, self.r_outer.len) catch unreachable;
                    defer self.allocator.free(full_y);

                    // Big-endian: first half = suffix = zeros (for j=0), second half = prefix = r_prefix_be
                    for (0..self.suffix_n_vars) |i| {
                        full_y[i] = F.zero();
                    }
                    for (0..r_prefix_be.len) |i| {
                        full_y[self.suffix_n_vars + i] = r_prefix_be[i];
                    }

                    // Direct evaluation
                    const direct_eq_plus_one = poly_mod.EqPlusOnePolynomial(F).mle(self.r_outer, full_y);
                    dbg("[ZOLT] SHIFT_CRITICAL: direct eq+1(r_outer, (zeros, r_prefix)) = {any}\n", .{direct_eq_plus_one.toBytes()});
                    dbg("[ZOLT] SHIFT_CRITICAL: phase2_eq+1_outer[0] = {any}\n", .{self.phase2_eq_plus_one_outer.?[0].toBytes()});
                    dbg("[ZOLT] SHIFT_CRITICAL: match = {}\n", .{direct_eq_plus_one.eql(self.phase2_eq_plus_one_outer.?[0])});

                    // Debug: check formula components
                    const expected_from_formula = prefix_0_eval_outer.mul(suffix_0_outer[0])
                        .add(prefix_1_eval_outer.mul(suffix_1_outer[0]));
                    dbg("[ZOLT] SHIFT_CRITICAL: from_formula = {any}\n", .{expected_from_formula.toBytes()});
                    dbg("[ZOLT] SHIFT_CRITICAL: formula_match = {}\n", .{expected_from_formula.eql(self.phase2_eq_plus_one_outer.?[0])});

                    // Debug: prefix_0 and suffix_0 individually
                    const direct_prefix_eval = poly_mod.EqPlusOnePolynomial(F).mle(r_outer_lo, r_prefix_be);
                    dbg("[ZOLT] SHIFT_CRITICAL: direct eq+1(r_lo, y_lo) = {any}\n", .{direct_prefix_eval.toBytes()});
                    dbg("[ZOLT] SHIFT_CRITICAL: prefix_0_eval_outer = {any}\n", .{prefix_0_eval_outer.toBytes()});
                    dbg("[ZOLT] SHIFT_CRITICAL: prefix_match = {}\n", .{direct_prefix_eval.eql(prefix_0_eval_outer)});

                    // Direct eq(r_hi, y_hi) where y_hi = zeros
                    const zeros_hi = self.allocator.alloc(F, self.suffix_n_vars) catch unreachable;
                    defer self.allocator.free(zeros_hi);
                    @memset(zeros_hi, F.zero());
                    const direct_suffix_eval = poly_mod.EqPolynomial(F).mle(r_outer_hi, zeros_hi);
                    dbg("[ZOLT] SHIFT_CRITICAL: direct eq(r_hi, zeros) = {any}\n", .{direct_suffix_eval.toBytes()});
                    dbg("[ZOLT] SHIFT_CRITICAL: suffix_0[0] = {any}\n", .{suffix_0_outer[0].toBytes()});
                    dbg("[ZOLT] SHIFT_CRITICAL: suffix_match = {}\n", .{direct_suffix_eval.eql(suffix_0_outer[0])});
                }
            }
        }

        fn bindPhase2(self: *Self, r_j: F) void {
            const new_size = self.current_witness_size / 2;

            var num_arrays: usize = 5;
            if (self.phase2_eq_plus_one_outer != null) num_arrays += 1;
            if (self.phase2_eq_plus_one_prod != null) num_arrays += 1;

            const BindP2Ctx = struct {
                slices: [7]?[]F,
                r: F,
                n: usize,
            };
            const bctx = BindP2Ctx{
                .slices = .{
                    self.unexpanded_pc,           self.pc,      self.is_virtual,
                    self.is_first_in_sequence,    self.is_noop, self.phase2_eq_plus_one_outer,
                    self.phase2_eq_plus_one_prod,
                },
                .r = r_j,
                .n = new_size,
            };
            const bindOneFn = struct {
                fn f(c: BindP2Ctx, idx: usize) void {
                    const arr = c.slices[idx] orelse return;
                    for (0..c.n) |i| {
                        arr[i] = arr[2 * i].add(c.r.mul(arr[2 * i + 1].sub(arr[2 * i])));
                    }
                }
            }.f;

            if (self.gpu_ops) |gpu| {
                if (new_size >= 16384) {
                    for (bctx.slices) |maybe_arr| {
                        const arr = maybe_arr orelse continue;
                        gpu.polyBindLow(arr[0 .. new_size * 2], r_j, arr[0..new_size]) catch {
                            for (0..new_size) |i| {
                                arr[i] = arr[2 * i].add(r_j.montgomeryMul(arr[2 * i + 1].sub(arr[2 * i])));
                            }
                        };
                    }
                } else {
                    for (0..num_arrays) |idx| bindOneFn(bctx, idx);
                }
            } else {
                pool_helpers.parallelForOptional(self.thread_pool, num_arrays, bctx, bindOneFn);
            }

            self.current_witness_size = new_size;
        }

        // Helper: Evaluate MLE at a point
        fn evaluateMle(coeffs: []const F, point: []const F) F {
            if (coeffs.len == 1) return coeffs[0];
            if (point.len == 0) return coeffs[0];

            const temp = std.heap.page_allocator.alloc(F, coeffs.len) catch unreachable;
            defer std.heap.page_allocator.free(temp);
            @memcpy(temp, coeffs);

            var current_len = coeffs.len;
            for (point) |r_i| {
                const half = current_len / 2;
                for (0..half) |i| {
                    temp[i] = temp[2 * i].add(r_i.mul(temp[2 * i + 1].sub(temp[2 * i])));
                }
                current_len = half;
            }
            return temp[0];
        }

        // Helper: Compute Eq(r, j) for all j
        fn computeEqEvals(allocator: Allocator, r: []const F, out: []F) !void {
            const n = r.len;
            const size = out.len;
            std.debug.assert(size == @as(usize, 1) << @intCast(n));

            const j_bits = try allocator.alloc(F, n);
            defer allocator.free(j_bits);

            for (0..size) |j| {
                // Convert j to binary (BIG_ENDIAN: bit 0 is MSB)
                for (0..n) |k| {
                    const bit_pos: std.math.Log2Int(usize) = @intCast(n - 1 - k);
                    j_bits[k] = if ((j >> bit_pos) & 1 == 1) F.one() else F.zero();
                }
                out[j] = poly_mod.EqPolynomial(F).mle(r, j_bits);
            }
        }

        /// Get final claims after all rounds
        /// After all rounds, current_witness_size should be 1
        pub fn finalClaims(self: *const Self) struct {
            unexpanded_pc: F,
            pc: F,
            is_virtual: F,
            is_first_in_sequence: F,
            is_noop: F,
        } {
            std.debug.assert(self.current_witness_size == 1);
            return .{
                .unexpanded_pc = self.unexpanded_pc[0],
                .pc = self.pc[0],
                .is_virtual = self.is_virtual[0],
                .is_first_in_sequence = self.is_first_in_sequence[0],
                .is_noop = self.is_noop[0],
            };
        }
    };
}

// =============================================================================
// RegistersClaimReduction Prefix-Suffix Prover
// =============================================================================

pub fn RegistersPrefixSuffixProver(comptime F: type) type {
    return struct {
        const Self = @This();

        // Single (P, Q) pair for eq polynomial
        P: []F, // Prefix eq evals
        Q: []F, // Accumulated witness * suffix

        // Witness MLEs (only allocated at Phase 2 transition, suffix_size)
        rd_write_value: ?[]F,
        rs1_value: ?[]F,
        rs2_value: ?[]F,

        gamma: F,
        gamma_sqr: F,

        prefix_n_vars: usize,
        suffix_n_vars: usize,
        current_prefix_size: usize,
        current_witness_size: usize,
        in_phase2: bool,

        // Phase 2 eq polynomial
        phase2_eq: ?[]F,

        // r_hi for Phase 2 initialization (eq suffix)
        r_hi: []const F,
        // r_lo for prefix evaluation in Phase 2
        r_lo: []const F,
        // Accumulated prefix challenges
        prefix_challenges: std.ArrayListUnmanaged(F),

        // Raw integer witness data (for Phase 2 reconstruction)
        raw_inputs: []const RawR1CSInputs,
        trace_len: usize,

        allocator: Allocator,
        thread_pool: ?*ThreadPool = null,
        gpu_ops: ?*GpuPolyOps = null,

        pub fn init(
            allocator: Allocator,
            raw_inputs: []const RawR1CSInputs,
            trace_len: usize,
            r_spartan: []const F,
            gamma: F,
            gamma_sqr: F,
            thread_pool: ?*ThreadPool,
        ) !Self {
            const n_vars = r_spartan.len;
            // Split r into hi (first half) and lo (second half)
            // Jolt convention: PREFIX uses r_lo, SUFFIX uses r_hi
            const split_point = n_vars / 2;
            const r_hi = r_spartan[0..split_point]; // First half -> used for SUFFIX
            const r_lo = r_spartan[split_point..]; // Second half -> used for PREFIX

            // Sizes: prefix_size = 2^len(r_lo), suffix_size = 2^len(r_hi)
            const prefix_n_vars = r_lo.len; // = n_vars - split_point
            const suffix_n_vars = r_hi.len; // = split_point
            const prefix_size: usize = @as(usize, 1) << @intCast(prefix_n_vars);
            const suffix_size: usize = @as(usize, 1) << @intCast(suffix_n_vars);

            // P = eq(r_lo, j) for PREFIX (Jolt convention)
            const P = try allocator.alloc(F, prefix_size);
            var eq_lo = try poly_mod.EqPolynomial(F).init(allocator, r_lo);
            defer eq_lo.deinit();
            const eq_lo_evals = try eq_lo.evals(allocator);
            defer allocator.free(eq_lo_evals);
            @memcpy(P, eq_lo_evals);

            // Suffix evals = eq(r_hi, j) for SUFFIX (Jolt convention)
            var eq_hi = try poly_mod.EqPolynomial(F).init(allocator, r_hi);
            defer eq_hi.deinit();
            const suffix_evals = try eq_hi.evals(allocator);
            defer allocator.free(suffix_evals);

            // Initialize Q buffer using RawR1CSInputs with mulU64Unreduced
            // No witness MLEs allocated here — they're reconstructed at Phase 2 transition
            const Q = try allocator.alloc(F, prefix_size);
            @memset(Q, F.zero());

            if (comptime debug_verbose) {
                dbg("[STAGE3] RegistersClaimReduction: trace_len={}, prefix_size={}, suffix_size={}\n", .{ trace_len, prefix_size, suffix_size });
            }

            const RegInitCtx = struct {
                raw_inputs: []const RawR1CSInputs,
                suffix_evals: []const F,
                Q_buf: []F,
                gamma_val: F,
                gamma_sqr_val: F,
                prefix_n_vars_val: usize,
                suffix_size_val: usize,
                trace_len_val: usize,
            };
            const reg_init_ctx = RegInitCtx{
                .raw_inputs = raw_inputs,
                .suffix_evals = suffix_evals,
                .Q_buf = Q,
                .gamma_val = gamma,
                .gamma_sqr_val = gamma_sqr,
                .prefix_n_vars_val = prefix_n_vars,
                .suffix_size_val = suffix_size,
                .trace_len_val = trace_len,
            };
            const regInitWorker = struct {
                fn f(c: RegInitCtx, x_lo: usize) void {
                    var q_acc = UnreducedProductAccum.zero();
                    for (0..c.suffix_size_val) |x_hi| {
                        const x = x_lo + (x_hi << @intCast(c.prefix_n_vars_val));
                        if (x >= c.trace_len_val) continue;

                        const raw = &c.raw_inputs[x];
                        const rd = raw.u64_values[6]; // RdWriteValue
                        const rs1 = raw.u64_values[4]; // Rs1Value
                        const rs2 = raw.u64_values[5]; // Rs2Value

                        // v = rd + gamma*rs1 + gamma^2*rs2 using mulU64Unreduced
                        var v_accum = FoldedMulU64.zero();
                        v_accum.addAssign(field_mod.mulU64Unreduced(F.one(), rd));
                        v_accum.addAssign(field_mod.mulU64Unreduced(c.gamma_val, rs1));
                        v_accum.addAssign(field_mod.mulU64Unreduced(c.gamma_sqr_val, rs2));
                        const v = field_mod.reduceMulU64(v_accum);

                        // Accumulate v * suffix (deferred Montgomery)
                        q_acc.addAssign(v.mulToProductAccum(c.suffix_evals[x_hi]));
                    }
                    c.Q_buf[x_lo] = q_acc.reduce();
                }
            }.f;

            pool_helpers.parallelForOptional(thread_pool, prefix_size, reg_init_ctx, regInitWorker);

            const padded_size = prefix_size * suffix_size;
            return Self{
                .P = P,
                .Q = Q,
                .rd_write_value = null, // Allocated at Phase 2 transition
                .rs1_value = null,
                .rs2_value = null,
                .gamma = gamma,
                .gamma_sqr = gamma_sqr,
                .prefix_n_vars = prefix_n_vars,
                .suffix_n_vars = suffix_n_vars,
                .current_prefix_size = prefix_size,
                .current_witness_size = padded_size,
                .in_phase2 = false,
                .phase2_eq = null,
                .r_hi = r_hi,
                .r_lo = r_lo,
                .prefix_challenges = std.ArrayListUnmanaged(F).initCapacity(allocator, @intCast(prefix_n_vars)) catch unreachable,
                .raw_inputs = raw_inputs,
                .trace_len = trace_len,
                .allocator = allocator,
                .thread_pool = thread_pool,
            };
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.P);
            self.allocator.free(self.Q);
            self.prefix_challenges.deinit(self.allocator);
            if (self.rd_write_value) |v| self.allocator.free(v);
            if (self.rs1_value) |v| self.allocator.free(v);
            if (self.rs2_value) |v| self.allocator.free(v);
            if (self.phase2_eq) |eq| self.allocator.free(eq);
        }

        pub fn computeRoundEvals(self: *Self, previous_claim: F) [3]F {
            if (self.in_phase2) {
                return self.computeRoundEvalsPhase2(previous_claim);
            } else {
                return self.computeRoundEvalsPhase1(previous_claim);
            }
        }

        fn computeRoundEvalsPhase1(self: *Self, previous_claim: F) [3]F {
            const half = self.current_prefix_size / 2;
            const use_deferred = comptime @hasDecl(F, "mulToProductAccum");
            var evals: [2]F = undefined;

            if (use_deferred) {
                var accum: [2]UnreducedProductAccum = .{ UnreducedProductAccum.zero(), UnreducedProductAccum.zero() };
                for (0..half) |i| {
                    const p_0 = self.P[2 * i];
                    const p_1 = self.P[2 * i + 1];
                    const q_0 = self.Q[2 * i];
                    const q_1 = self.Q[2 * i + 1];
                    const p_2 = p_1.add(p_1).sub(p_0);
                    const q_2 = q_1.add(q_1).sub(q_0);

                    accum[0].addAssign(p_0.mulToProductAccum(q_0));
                    accum[1].addAssign(p_2.mulToProductAccum(q_2));
                }
                evals = .{ accum[0].reduce(), accum[1].reduce() };
            } else {
                evals = .{ F.zero(), F.zero() };
                for (0..half) |i| {
                    const p_0 = self.P[2 * i];
                    const p_1 = self.P[2 * i + 1];
                    const q_0 = self.Q[2 * i];
                    const q_1 = self.Q[2 * i + 1];
                    const p_2 = p_1.add(p_1).sub(p_0);
                    const q_2 = q_1.add(q_1).sub(q_0);

                    evals[0] = evals[0].add(p_0.mul(q_0));
                    evals[1] = evals[1].add(p_2.mul(q_2));
                }
            }

            const p_1 = previous_claim.sub(evals[0]);
            return [3]F{ evals[0], p_1, evals[1] };
        }

        fn computeRoundEvalsPhase2(self: *Self, previous_claim: F) [3]F {
            const eq = self.phase2_eq.?;
            const rd = self.rd_write_value.?;
            const rs1 = self.rs1_value.?;
            const rs2 = self.rs2_value.?;
            const half = self.current_witness_size / 2;
            var evals: [2]F = .{ F.zero(), F.zero() };

            for (0..half) |j| {
                const eq_0 = eq[2 * j];
                const eq_1 = eq[2 * j + 1];
                const rd_0 = rd[2 * j];
                const rd_1 = rd[2 * j + 1];
                const rs1_0 = rs1[2 * j];
                const rs1_1 = rs1[2 * j + 1];
                const rs2_0 = rs2[2 * j];
                const rs2_1 = rs2[2 * j + 1];

                // Extrapolate
                const eq_2 = eq_1.add(eq_1).sub(eq_0);
                const rd_2 = rd_1.add(rd_1).sub(rd_0);
                const rs1_2 = rs1_1.add(rs1_1).sub(rs1_0);
                const rs2_2 = rs2_1.add(rs2_1).sub(rs2_0);

                const v_0 = rd_0.add(self.gamma.mul(rs1_0)).add(self.gamma_sqr.mul(rs2_0));
                const v_2 = rd_2.add(self.gamma.mul(rs1_2)).add(self.gamma_sqr.mul(rs2_2));

                evals[0] = evals[0].add(eq_0.mul(v_0));
                evals[1] = evals[1].add(eq_2.mul(v_2));
            }

            const p_1 = previous_claim.sub(evals[0]);
            return [3]F{ evals[0], p_1, evals[1] };
        }

        pub fn bind(self: *Self, r_j: F) void {
            if (self.in_phase2) {
                self.bindPhase2(r_j);
            } else {
                if (self.shouldTransitionToPhase2()) {
                    self.transitionToPhase2(r_j);
                } else {
                    self.bindPhase1(r_j);
                }
            }
        }

        fn shouldTransitionToPhase2(self: *Self) bool {
            return std.math.log2_int(usize, self.current_prefix_size) == 1;
        }

        fn bindPhase1(self: *Self, r_j: F) void {
            const new_prefix_size = self.current_prefix_size / 2;

            // Only bind P and Q (prefix-sized arrays)
            // Witness MLEs are NOT bound during Phase 1 — they're reconstructed at transition
            for (0..new_prefix_size) |i| {
                self.P[i] = self.P[2 * i].add(r_j.mul(self.P[2 * i + 1].sub(self.P[2 * i])));
                self.Q[i] = self.Q[2 * i].add(r_j.mul(self.Q[2 * i + 1].sub(self.Q[2 * i])));
            }

            self.current_prefix_size = new_prefix_size;

            // Record challenge for Phase 2 initialization
            self.prefix_challenges.append(self.allocator, r_j) catch unreachable;
        }

        fn transitionToPhase2(self: *Self, r_j: F) void {
            // Final bind and record challenge
            self.bindPhase1(r_j);
            self.in_phase2 = true;

            const suffix_size: usize = @as(usize, 1) << @intCast(self.suffix_n_vars);

            // Materialize eq polynomial for Phase 2:
            // eq(r_spartan, (r_prefix, j)) = eq(r_lo, r_prefix) * eq(r_hi, j)

            // Reverse prefix challenges (LE → BE) for Jolt compatibility
            const reversed_prefix = self.allocator.alloc(F, self.prefix_challenges.items.len) catch unreachable;
            defer self.allocator.free(reversed_prefix);
            for (0..self.prefix_challenges.items.len) |i| {
                reversed_prefix[i] = self.prefix_challenges.items[self.prefix_challenges.items.len - 1 - i];
            }

            var eq_prefix = poly_mod.EqPolynomial(F).init(self.allocator, self.r_lo) catch unreachable;
            defer eq_prefix.deinit();
            const eq_prefix_eval = eq_prefix.evaluate(reversed_prefix);

            // Compute eq(r_hi, j) for each j in suffix domain
            var eq_suffix = poly_mod.EqPolynomial(F).init(self.allocator, self.r_hi) catch unreachable;
            defer eq_suffix.deinit();
            const eq_suffix_evals = eq_suffix.evals(self.allocator) catch unreachable;
            defer self.allocator.free(eq_suffix_evals);

            // phase2_eq[j] = eq_suffix[j] * eq_prefix_eval
            self.phase2_eq = self.allocator.alloc(F, suffix_size) catch unreachable;
            for (0..suffix_size) |j| {
                self.phase2_eq.?[j] = eq_suffix_evals[j].mul(eq_prefix_eval);
            }

            // Reconstruct witness MLEs at suffix_size by contracting over prefix domain
            // using raw_inputs + mulU64Unreduced
            const prefix_domain_size: usize = @as(usize, 1) << @intCast(self.prefix_challenges.items.len);

            // Compute eq(r_prefix_be, i) for all prefix indices using EqPolynomial
            var eq_poly = poly_mod.EqPolynomial(F).init(self.allocator, reversed_prefix) catch unreachable;
            defer eq_poly.deinit();
            const eq_evals_alloc = eq_poly.evals(self.allocator) catch unreachable;
            defer self.allocator.free(eq_evals_alloc);
            const eq_evals = eq_evals_alloc[0..prefix_domain_size];

            // Allocate witness MLEs at suffix_size (not T)
            self.rd_write_value = self.allocator.alloc(F, suffix_size) catch unreachable;
            self.rs1_value = self.allocator.alloc(F, suffix_size) catch unreachable;
            self.rs2_value = self.allocator.alloc(F, suffix_size) catch unreachable;

            // Contract: witness_mle[j] = Sum_i eq_evals[i] * raw_inputs[j*prefix + i].u64_values[idx]
            const RegReconCtx = struct {
                raw_inputs_ptr: []const RawR1CSInputs,
                eq_ev: []const F,
                rd_out: []F,
                rs1_out: []F,
                rs2_out: []F,
                prefix_dom_size: usize,
                tl: usize,
            };
            const recon_ctx = RegReconCtx{
                .raw_inputs_ptr = self.raw_inputs,
                .eq_ev = eq_evals,
                .rd_out = self.rd_write_value.?,
                .rs1_out = self.rs1_value.?,
                .rs2_out = self.rs2_value.?,
                .prefix_dom_size = prefix_domain_size,
                .tl = self.trace_len,
            };
            const regReconWorker = struct {
                fn f(c: RegReconCtx, j: usize) void {
                    var rd_acc = FoldedMulU64.zero();
                    var rs1_acc = FoldedMulU64.zero();
                    var rs2_acc = FoldedMulU64.zero();

                    for (0..c.prefix_dom_size) |i| {
                        const trace_idx = j * c.prefix_dom_size + i;
                        if (trace_idx >= c.tl) continue;

                        const raw = &c.raw_inputs_ptr[trace_idx];
                        const eq_eval = c.eq_ev[i];

                        rd_acc.addAssign(field_mod.mulU64Unreduced(eq_eval, raw.u64_values[6])); // RdWriteValue
                        rs1_acc.addAssign(field_mod.mulU64Unreduced(eq_eval, raw.u64_values[4])); // Rs1Value
                        rs2_acc.addAssign(field_mod.mulU64Unreduced(eq_eval, raw.u64_values[5])); // Rs2Value
                    }

                    c.rd_out[j] = field_mod.reduceMulU64(rd_acc);
                    c.rs1_out[j] = field_mod.reduceMulU64(rs1_acc);
                    c.rs2_out[j] = field_mod.reduceMulU64(rs2_acc);
                }
            }.f;
            pool_helpers.parallelForOptional(self.thread_pool, suffix_size, recon_ctx, regReconWorker);

            self.current_witness_size = suffix_size;
        }

        fn bindPhase2(self: *Self, r_j: F) void {
            const new_size = self.current_witness_size / 2;

            var num_arrays: usize = 3;
            if (self.phase2_eq != null) num_arrays = 4;

            const RegBP2Ctx = struct {
                slices: [4]?[]F,
                r: F,
                n: usize,
            };
            const bctx = RegBP2Ctx{
                .slices = .{ self.rd_write_value, self.rs1_value, self.rs2_value, self.phase2_eq },
                .r = r_j,
                .n = new_size,
            };
            const bindOneFn = struct {
                fn f(c: RegBP2Ctx, idx: usize) void {
                    const arr = c.slices[idx] orelse return;
                    for (0..c.n) |i| {
                        arr[i] = arr[2 * i].add(c.r.mul(arr[2 * i + 1].sub(arr[2 * i])));
                    }
                }
            }.f;

            if (self.gpu_ops) |gpu| {
                if (new_size >= 16384) {
                    for (bctx.slices) |maybe_arr| {
                        const arr = maybe_arr orelse continue;
                        gpu.polyBindLow(arr[0 .. new_size * 2], r_j, arr[0..new_size]) catch {
                            for (0..new_size) |i| {
                                arr[i] = arr[2 * i].add(r_j.montgomeryMul(arr[2 * i + 1].sub(arr[2 * i])));
                            }
                        };
                    }
                } else {
                    for (0..num_arrays) |idx| bindOneFn(bctx, idx);
                }
            } else {
                pool_helpers.parallelForOptional(self.thread_pool, num_arrays, bctx, bindOneFn);
            }

            self.current_witness_size = new_size;
        }

        pub fn finalClaims(self: *const Self) struct {
            rd_write_value: F,
            rs1_value: F,
            rs2_value: F,
        } {
            std.debug.assert(self.current_witness_size == 1);
            return .{
                .rd_write_value = self.rd_write_value.?[0],
                .rs1_value = self.rs1_value.?[0],
                .rs2_value = self.rs2_value.?[0],
            };
        }
    };
}
