//! Instruction Lookups Claim Reduction Sumcheck Prover
//!
//! This implements the InstructionLookupsClaimReduction sumcheck for Stage 2.
//! It proves the aggregation of instruction lookup claims from Spartan outer.
//!
//! The sumcheck proves:
//! Σ_j eq(r_spartan, j) * (LookupOutput(j) + γ*LeftOp(j) + γ²*RightOp(j) + γ³*LeftInstr(j) + γ⁴*RightInstr(j)) = input_claim
//!
//! Uses the prefix-suffix P/Q trick from ePrint 2025/611 Appendix A:
//! Split cycle index j = (j_lo, j_hi), work with sqrt(T)-sized buffers in Phase 1,
//! then materialize suffix-sized arrays in Phase 2.

const std = @import("std");

// Debug output control - set to true to enable verbose debug prints
const debug_verbose = false;
fn dbg(comptime fmt: []const u8, args: anytype) void {
    if (debug_verbose) std.debug.print(fmt, args);
}

const Allocator = std.mem.Allocator;
const ThreadPool = @import("../../utils/thread_pool.zig").ThreadPool;
const poly_mod = @import("../../poly/mod.zig");
const EqPolynomial = poly_mod.EqPolynomial;
const UniPoly = poly_mod.UniPoly;
const R1CSInputIndex = @import("../r1cs/constraints.zig").R1CSInputIndex;
const RawR1CSInputs = @import("../r1cs/evaluators.zig").RawR1CSInputs;

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
        /// Challenges from SpartanOuter (r_spartan), BIG ENDIAN
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

/// Witness field indices for the 5 lookup signals
const WitnessField = enum(usize) {
    LookupOutput = 0,
    LeftLookupOperand = 1,
    RightLookupOperand = 2,
    LeftInstructionInput = 3,
    RightInstructionInput = 4,

    fn r1csIndex(self: WitnessField) usize {
        return switch (self) {
            .LookupOutput => R1CSInputIndex.LookupOutput.toIndex(),
            .LeftLookupOperand => R1CSInputIndex.LeftLookupOperand.toIndex(),
            .RightLookupOperand => R1CSInputIndex.RightLookupOperand.toIndex(),
            .LeftInstructionInput => R1CSInputIndex.LeftInstructionInput.toIndex(),
            .RightInstructionInput => R1CSInputIndex.RightInstructionInput.toIndex(),
        };
    }
};
const NUM_WITNESS_FIELDS = 5;

/// R1CS cycle inputs type (generic over field) — kept for tests
fn R1CSCycleInputs(comptime F: type) type {
    return @import("../r1cs/constraints.zig").R1CSCycleInputs(F);
}

/// Instruction Lookups Claim Reduction Prover with P/Q prefix-suffix optimization
pub fn InstructionLookupsProver(comptime F: type) type {
    return struct {
        const Self = @This();

        /// Parameters
        params: InstructionLookupsParams(F),
        /// Current claim
        current_claim: F,
        /// Current round
        round: usize,
        /// Phase state
        phase: Phase,
        /// Bound challenges (collected for Phase 2 transition)
        challenges: std.ArrayListUnmanaged(F),
        /// Allocator
        allocator: Allocator,
        /// Thread pool for parallelism
        thread_pool: ?*ThreadPool = null,

        const Phase = union(enum) {
            phase1: Phase1State,
            phase2: Phase2State,
        };

        const Phase1State = struct {
            /// P[j_lo] = eq(r_lo, j_lo) — prefix eq evals, shrinks each round
            P: []F,
            /// Q[j_lo] = Σ_{j_hi} eq_hi[j_hi] * combined(j_lo + j_hi * prefix_size)
            Q: []F,
            /// Number of prefix variables (low bits of cycle index)
            prefix_n_vars: usize,
            /// Number of suffix variables (high bits of cycle index)
            suffix_n_vars: usize,
            /// Original prefix size (2^prefix_n_vars) for phase2 materialization
            original_prefix_size: usize,
            /// Reference to raw integer witness data for Phase 2 materialization
            raw_inputs: []const RawR1CSInputs,
            /// Original P allocation size for dealloc
            original_P_size: usize,
            /// Original Q allocation size for dealloc
            original_Q_size: usize,
        };

        const Phase2State = struct {
            /// eq(suffix) evals
            eq_evals: []F,
            /// 5 witness polynomial arrays
            lookup_outputs: []F,
            left_operands: []F,
            right_operands: []F,
            left_instr_inputs: []F,
            right_instr_inputs: []F,
            /// Original allocation size for dealloc
            original_size: usize,
        };

        pub fn init(
            allocator: Allocator,
            params: InstructionLookupsParams(F),
            initial_claim: F,
            raw_inputs: []const RawR1CSInputs,
            thread_pool: ?*ThreadPool,
        ) !Self {
            const n = params.n_cycle_vars;
            const prefix_n_vars = n / 2;
            const suffix_n_vars = n - prefix_n_vars;
            const prefix_size = @as(usize, 1) << @intCast(prefix_n_vars);
            const suffix_size = @as(usize, 1) << @intCast(suffix_n_vars);

            // r_spartan is BIG ENDIAN: first suffix_n_vars elements are high bits,
            // remaining prefix_n_vars elements are low bits
            const r_hi = params.r_spartan[0..suffix_n_vars]; // suffix (high bits)
            const r_lo = params.r_spartan[suffix_n_vars..]; // prefix (low bits)

            // Build P = eq(r_lo, ·) — size prefix_size, O(prefix_size) work
            const P = try allocator.alloc(F, prefix_size);
            EqPolynomial(F).buildEqTableInPlace(r_lo, P, null);

            // Build eq_hi = eq(r_hi, ·) — size suffix_size, O(suffix_size) work
            const eq_hi = try allocator.alloc(F, suffix_size);
            defer allocator.free(eq_hi);
            EqPolynomial(F).buildEqTableInPlace(r_hi, eq_hi, null);

            // Build Q via blocked accumulation
            // Q[x_lo] = Σ_{x_hi} eq_hi[x_hi] * combined(x_lo + x_hi * prefix_size)
            const Q = try allocator.alloc(F, prefix_size);
            const trace_len = raw_inputs.len;
            const padded_T = @as(usize, 1) << @intCast(n);

            const QCtx = struct {
                Q_buf: []F,
                eq_hi_buf: []const F,
                raw_inputs: []const RawR1CSInputs,
                gamma: F,
                gamma_sqr: F,
                gamma_cub: F,
                gamma_quart: F,
                p_size: usize,
                s_size: usize,
                t_len: usize,
                padded: usize,
            };
            const qctx = QCtx{
                .Q_buf = Q,
                .eq_hi_buf = eq_hi,
                .raw_inputs = raw_inputs,
                .gamma = params.gamma,
                .gamma_sqr = params.gamma_sqr,
                .gamma_cub = params.gamma_cub,
                .gamma_quart = params.gamma_quart,
                .p_size = prefix_size,
                .s_size = suffix_size,
                .t_len = trace_len,
                .padded = padded_T,
            };

            const buildQFn = struct {
                fn f(c: QCtx, block_idx: usize) void {
                    // Each block handles one x_lo position
                    const x_lo = block_idx;
                    var acc_lo = F.zero();
                    var acc_left = F.zero();
                    var acc_right = F.zero();
                    var acc_li = F.zero();
                    var acc_ri = F.zero();

                    for (0..c.s_size) |x_hi| {
                        const j = x_lo + x_hi * c.p_size;
                        if (j >= c.padded) break;
                        const e = c.eq_hi_buf[x_hi];
                        if (j < c.t_len) {
                            const raw = &c.raw_inputs[j];
                            acc_lo = acc_lo.add(e.mul(raw.toFieldValue(F, .LookupOutput)));
                            acc_left = acc_left.add(e.mul(raw.toFieldValue(F, .LeftLookupOperand)));
                            acc_right = acc_right.add(e.mul(raw.toFieldValue(F, .RightLookupOperand)));
                            acc_li = acc_li.add(e.mul(raw.toFieldValue(F, .LeftInstructionInput)));
                            acc_ri = acc_ri.add(e.mul(raw.toFieldValue(F, .RightInstructionInput)));
                        }
                        // else: padded cycle, all witness values are 0, contributes nothing
                    }

                    // Combine: LookupOutput + γ*Left + γ²*Right + γ³*LeftInstr + γ⁴*RightInstr
                    c.Q_buf[x_lo] = acc_lo
                        .add(c.gamma.mul(acc_left))
                        .add(c.gamma_sqr.mul(acc_right))
                        .add(c.gamma_cub.mul(acc_li))
                        .add(c.gamma_quart.mul(acc_ri));
                }
            }.f;

            if (thread_pool) |tp| {
                tp.parallelForForce(prefix_size, qctx, buildQFn);
            } else {
                for (0..prefix_size) |x_lo| buildQFn(qctx, x_lo);
            }

            return Self{
                .params = params,
                .current_claim = initial_claim,
                .round = 0,
                .phase = .{ .phase1 = .{
                    .P = P,
                    .Q = Q,
                    .prefix_n_vars = prefix_n_vars,
                    .suffix_n_vars = suffix_n_vars,
                    .original_prefix_size = prefix_size,
                    .raw_inputs = raw_inputs,
                    .original_P_size = prefix_size,
                    .original_Q_size = prefix_size,
                } },
                .challenges = std.ArrayListUnmanaged(F){},
                .allocator = allocator,
                .thread_pool = thread_pool,
            };
        }

        pub fn deinit(self: *Self) void {
            switch (self.phase) {
                .phase1 => |s| {
                    self.allocator.free(s.P.ptr[0..s.original_P_size]);
                    self.allocator.free(s.Q.ptr[0..s.original_Q_size]);
                },
                .phase2 => |s| {
                    self.allocator.free(s.eq_evals.ptr[0..s.original_size]);
                    self.allocator.free(s.lookup_outputs.ptr[0..s.original_size]);
                    self.allocator.free(s.left_operands.ptr[0..s.original_size]);
                    self.allocator.free(s.right_operands.ptr[0..s.original_size]);
                    self.allocator.free(s.left_instr_inputs.ptr[0..s.original_size]);
                    self.allocator.free(s.right_instr_inputs.ptr[0..s.original_size]);
                },
            }
            self.challenges.deinit(self.allocator);
            self.params.deinit();
        }

        /// Compute round polynomial [s(0), s(1), s(2), s(3)] for batched cubic sumcheck
        /// Degree-2 sumcheck padded to degree-3 for batching compatibility
        pub fn computeRoundPolynomialCubic(self: *Self) [4]F {
            return switch (self.phase) {
                .phase1 => |s| self.computePhase1(s),
                .phase2 => |s| self.computePhase2(s),
            };
        }

        fn computePhase1(self: *Self, s: Phase1State) [4]F {
            // Inner product of P and Q — only prefix_size/2 pairs (tiny: ~256 for n=19)
            const half = s.P.len / 2;
            var s0 = F.zero();
            var s2 = F.zero();

            for (0..half) |idx| {
                const p_lo = s.P[2 * idx];
                const p_hi = s.P[2 * idx + 1];
                const q_lo = s.Q[2 * idx];
                const q_hi = s.Q[2 * idx + 1];

                s0 = s0.add(p_lo.mul(q_lo));

                const p_2 = p_hi.add(p_hi).sub(p_lo);
                const q_2 = q_hi.add(q_hi).sub(q_lo);
                s2 = s2.add(p_2.mul(q_2));
            }

            const s1 = self.current_claim.sub(s0);
            const s3 = s0.sub(s1.mul(F.fromU64(3))).add(s2.mul(F.fromU64(3)));
            return [4]F{ s0, s1, s2, s3 };
        }

        fn computePhase2(self: *Self, s: Phase2State) [4]F {
            const half = s.eq_evals.len / 2;

            const P2Ctx = struct {
                eq: []const F,
                lo: []const F,
                left: []const F,
                right: []const F,
                li: []const F,
                ri: []const F,
                gamma: F,
                gamma_sqr: F,
                gamma_cub: F,
                gamma_quart: F,
            };
            const ctx = P2Ctx{
                .eq = s.eq_evals,
                .lo = s.lookup_outputs,
                .left = s.left_operands,
                .right = s.right_operands,
                .li = s.left_instr_inputs,
                .ri = s.right_instr_inputs,
                .gamma = self.params.gamma,
                .gamma_sqr = self.params.gamma_sqr,
                .gamma_cub = self.params.gamma_cub,
                .gamma_quart = self.params.gamma_quart,
            };

            const mapFn = struct {
                fn f(c: P2Ctx, start: usize, end: usize) [2]F {
                    var local_s0 = F.zero();
                    var local_s2 = F.zero();
                    for (start..end) |idx| {
                        const lo_idx = 2 * idx;
                        const hi_idx = 2 * idx + 1;
                        const eq_lo = c.eq[lo_idx];
                        const eq_hi = c.eq[hi_idx];
                        const combined_lo = c.lo[lo_idx]
                            .add(c.gamma.mul(c.left[lo_idx]))
                            .add(c.gamma_sqr.mul(c.right[lo_idx]))
                            .add(c.gamma_cub.mul(c.li[lo_idx]))
                            .add(c.gamma_quart.mul(c.ri[lo_idx]));
                        const combined_hi = c.lo[hi_idx]
                            .add(c.gamma.mul(c.left[hi_idx]))
                            .add(c.gamma_sqr.mul(c.right[hi_idx]))
                            .add(c.gamma_cub.mul(c.li[hi_idx]))
                            .add(c.gamma_quart.mul(c.ri[hi_idx]));
                        local_s0 = local_s0.add(eq_lo.mul(combined_lo));
                        const eq_2 = eq_hi.add(eq_hi).sub(eq_lo);
                        const combined_2 = combined_hi.add(combined_hi).sub(combined_lo);
                        local_s2 = local_s2.add(eq_2.mul(combined_2));
                    }
                    return .{ local_s0, local_s2 };
                }
            }.f;

            const reduceFn = struct {
                fn f(a: [2]F, b: [2]F) [2]F { return .{ a[0].add(b[0]), a[1].add(b[1]) }; }
            }.f;

            const identity = [2]F{ F.zero(), F.zero() };
            const sums = if (self.thread_pool) |tp|
                tp.parallelReduce([2]F, half, identity, ctx, mapFn, reduceFn)
            else
                mapFn(ctx, 0, half);

            const s0 = sums[0];
            const s1 = self.current_claim.sub(s0);
            const s3 = s0.sub(s1.mul(F.fromU64(3))).add(sums[1].mul(F.fromU64(3)));
            return [4]F{ s0, s1, sums[1], s3 };
        }

        /// Bind a challenge after round polynomial computation
        pub fn bindChallenge(self: *Self, challenge: F) !void {
            try self.challenges.append(self.allocator, challenge);

            switch (self.phase) {
                .phase1 => |*s| {
                    // Check if we should transition to Phase 2
                    // Transition when P.len == 2 (one round left in prefix)
                    if (s.P.len == 2) {
                        try self.transitionToPhase2(challenge, s.*);
                        return;
                    }
                    // Bind P and Q (tiny arrays)
                    const half = s.P.len / 2;
                    bindLow(F, s.P, challenge);
                    bindLow(F, s.Q, challenge);
                    s.P = s.P[0..half];
                    s.Q = s.Q[0..half];
                },
                .phase2 => |*s| {
                    const half = s.eq_evals.len / 2;
                    // Bind 6 arrays
                    const ILBindCtx = struct {
                        slices: [6][]F,
                        r: F,
                        n: usize,
                    };
                    const bctx = ILBindCtx{
                        .slices = .{
                            s.eq_evals, s.lookup_outputs, s.left_operands,
                            s.right_operands, s.left_instr_inputs, s.right_instr_inputs,
                        },
                        .r = challenge,
                        .n = half,
                    };
                    const bindOneFn = struct {
                        fn f(c: ILBindCtx, idx: usize) void {
                            const arr = c.slices[idx];
                            for (0..c.n) |i| {
                                arr[i] = arr[2 * i].add(c.r.mul(arr[2 * i + 1].sub(arr[2 * i])));
                            }
                        }
                    }.f;

                    if (self.thread_pool) |tp| {
                        tp.parallelForForce(6, bctx, bindOneFn);
                    } else {
                        for (0..6) |idx| bindOneFn(bctx, idx);
                    }

                    s.eq_evals = s.eq_evals[0..half];
                    s.lookup_outputs = s.lookup_outputs[0..half];
                    s.left_operands = s.left_operands[0..half];
                    s.right_operands = s.right_operands[0..half];
                    s.left_instr_inputs = s.left_instr_inputs[0..half];
                    s.right_instr_inputs = s.right_instr_inputs[0..half];
                },
            }

            self.round += 1;
        }

        /// Transition from Phase 1 to Phase 2
        /// Called when P.len == 2 with the final prefix challenge
        fn transitionToPhase2(self: *Self, _: F, s1: Phase1State) !void {
            const prefix_n_vars = s1.prefix_n_vars;
            const suffix_n_vars = s1.suffix_n_vars;
            const prefix_size = s1.original_prefix_size;
            const suffix_size = @as(usize, 1) << @intCast(suffix_n_vars);

            // Build eq_prefix from collected challenges (LE order from sumcheck)
            // Convert to BE by reversing, then build eq table
            const n_phase1_challenges = self.challenges.items.len; // includes final_challenge just appended
            const eq_prefix_challenges = try self.allocator.alloc(F, n_phase1_challenges);
            defer self.allocator.free(eq_prefix_challenges);
            // Reverse for LE→BE conversion
            for (0..n_phase1_challenges) |i| {
                eq_prefix_challenges[i] = self.challenges.items[n_phase1_challenges - 1 - i];
            }

            // Build eq_prefix table
            const eq_prefix = try self.allocator.alloc(F, prefix_size);
            defer self.allocator.free(eq_prefix);
            EqPolynomial(F).buildEqTableInPlace(eq_prefix_challenges, eq_prefix, null);

            // Compute eq_prefix_at_r_lo = eq(collected_challenges_BE, r_lo)
            // where r_lo = r_spartan[suffix_n_vars..]
            const r_lo = self.params.r_spartan[suffix_n_vars..];
            var eq_prefix_at_r_lo = F.one();
            for (0..prefix_n_vars) |i| {
                const xi = eq_prefix_challenges[i];
                const ri = r_lo[i];
                const xi_ri = xi.mul(ri);
                const one_minus_xi = F.one().sub(xi);
                const one_minus_ri = F.one().sub(ri);
                eq_prefix_at_r_lo = eq_prefix_at_r_lo.mul(xi_ri.add(one_minus_xi.mul(one_minus_ri)));
            }

            // Build eq_suffix = eq(r_hi, ·) scaled by eq_prefix_at_r_lo
            const r_hi = self.params.r_spartan[0..suffix_n_vars];
            const eq_suffix = try self.allocator.alloc(F, suffix_size);
            EqPolynomial(F).buildEqTableInPlace(r_hi, eq_suffix, eq_prefix_at_r_lo);

            // Materialize 5 witness polynomials of size suffix_size
            // w[j_hi] = Σ_{j_lo} eq_prefix[j_lo] * witness_field(j_lo + j_hi * prefix_size)
            const lo_poly = try self.allocator.alloc(F, suffix_size);
            const left_poly = try self.allocator.alloc(F, suffix_size);
            const right_poly = try self.allocator.alloc(F, suffix_size);
            const li_poly = try self.allocator.alloc(F, suffix_size);
            const ri_poly = try self.allocator.alloc(F, suffix_size);

            const MatCtx = struct {
                eq_pref: []const F,
                raw_inputs: []const RawR1CSInputs,
                p_size: usize,
                t_len: usize,
                lo_buf: []F,
                left_buf: []F,
                right_buf: []F,
                li_buf: []F,
                ri_buf: []F,
            };
            const mctx = MatCtx{
                .eq_pref = eq_prefix,
                .raw_inputs = s1.raw_inputs,
                .p_size = prefix_size,
                .t_len = s1.raw_inputs.len,
                .lo_buf = lo_poly,
                .left_buf = left_poly,
                .right_buf = right_poly,
                .li_buf = li_poly,
                .ri_buf = ri_poly,
            };

            const materializeFn = struct {
                fn f(c: MatCtx, j_hi: usize) void {
                    var sum_lo = F.zero();
                    var sum_left = F.zero();
                    var sum_right = F.zero();
                    var sum_li = F.zero();
                    var sum_ri = F.zero();

                    for (0..c.p_size) |j_lo| {
                        const j = j_lo + j_hi * c.p_size;
                        const eq_val = c.eq_pref[j_lo];
                        if (j < c.t_len) {
                            const raw = &c.raw_inputs[j];
                            sum_lo = sum_lo.add(eq_val.mul(raw.toFieldValue(F, .LookupOutput)));
                            sum_left = sum_left.add(eq_val.mul(raw.toFieldValue(F, .LeftLookupOperand)));
                            sum_right = sum_right.add(eq_val.mul(raw.toFieldValue(F, .RightLookupOperand)));
                            sum_li = sum_li.add(eq_val.mul(raw.toFieldValue(F, .LeftInstructionInput)));
                            sum_ri = sum_ri.add(eq_val.mul(raw.toFieldValue(F, .RightInstructionInput)));
                        }
                    }

                    c.lo_buf[j_hi] = sum_lo;
                    c.left_buf[j_hi] = sum_left;
                    c.right_buf[j_hi] = sum_right;
                    c.li_buf[j_hi] = sum_li;
                    c.ri_buf[j_hi] = sum_ri;
                }
            }.f;

            if (self.thread_pool) |tp| {
                tp.parallelForForce(suffix_size, mctx, materializeFn);
            } else {
                for (0..suffix_size) |j_hi| materializeFn(mctx, j_hi);
            }

            // Free Phase 1 state
            self.allocator.free(s1.P.ptr[0..s1.original_P_size]);
            self.allocator.free(s1.Q.ptr[0..s1.original_Q_size]);

            // Set Phase 2
            self.phase = .{ .phase2 = .{
                .eq_evals = eq_suffix,
                .lookup_outputs = lo_poly,
                .left_operands = left_poly,
                .right_operands = right_poly,
                .left_instr_inputs = li_poly,
                .right_instr_inputs = ri_poly,
                .original_size = suffix_size,
            } };

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

            const L1 = c.mul(c_minus_2).mul(c_minus_3).mul(UniPoly(F).INV2);

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
            const s = switch (self.phase) {
                .phase2 => |s| s,
                .phase1 => unreachable, // should always finish in phase2
            };

            const lookup_output = if (s.lookup_outputs.len > 0) s.lookup_outputs[0] else F.zero();
            const left_operand = if (s.left_operands.len > 0) s.left_operands[0] else F.zero();
            const right_operand = if (s.right_operands.len > 0) s.right_operands[0] else F.zero();
            const left_instr_input = if (s.left_instr_inputs.len > 0) s.left_instr_inputs[0] else F.zero();
            const right_instr_input = if (s.right_instr_inputs.len > 0) s.right_instr_inputs[0] else F.zero();

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

/// Bind low (LowToHigh): fold pairs (2i, 2i+1) into position i
fn bindLow(comptime F: type, arr: []F, r: F) void {
    const half = arr.len / 2;
    for (0..half) |i| {
        arr[i] = arr[2 * i].add(r.mul(arr[2 * i + 1].sub(arr[2 * i])));
    }
}

test "instruction lookups prover initialization" {
    const allocator = std.testing.allocator;
    const field = @import("../../field/mod.zig");
    const F = field.BN254Scalar;
    const constraints = @import("../r1cs/constraints.zig");
    const CycleInputs = constraints.R1CSCycleInputs(F);

    const r_spartan = [_]F{ F.fromU64(1), F.fromU64(2), F.fromU64(3), F.fromU64(4) };
    const params = try InstructionLookupsParams(F).init(
        allocator,
        F.fromU64(12345), // gamma
        &r_spartan,
        4, // n_cycle_vars (16 cycles)
    );

    // Create cycle witnesses
    var witnesses: [4]CycleInputs = undefined;
    for (&witnesses) |*w| {
        w.* = CycleInputs.init();
    }
    witnesses[0].setInput(.LookupOutput, F.fromU64(1));
    witnesses[1].setInput(.LookupOutput, F.fromU64(2));
    witnesses[2].setInput(.LookupOutput, F.fromU64(3));
    witnesses[3].setInput(.LookupOutput, F.fromU64(4));

    var prover = try InstructionLookupsProver(F).init(
        allocator,
        params,
        F.fromU64(1000), // initial_claim
        &witnesses,
        null, // no thread pool
    );
    defer prover.deinit();

    try std.testing.expect(!prover.isComplete());
}
