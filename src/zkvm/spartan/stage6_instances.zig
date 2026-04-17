//! Stage 6 Sumcheck Instance Provers (extracted from stage6_prover.zig)
//!
//! Contains 5 of the 6 sumcheck instance provers:
//! - IncClaimReductionProver (Instance 5): two-phase prefix-suffix prover
//! - HammingBooleanityProver (Instance 1): split-eq + GruenSplitEq
//! - RamRaVirtualProver (Instance 3): compressed RaPolynomial + GruenSplitEq
//! - BooleanityProver (Instance 2): expanding table + lazy/dense phases
//! - LookupsRaVirtualProver (Instance 4): compressed RaPolynomial + GruenSplitEq + Toom-Cook

const std = @import("std");

const Allocator = std.mem.Allocator;
const ThreadPool = @import("zolt_pool").ThreadPool;
const pool_helpers = @import("zolt_pool").helpers;
const GpuPolyOps = @import("zolt_arith").gpu.GpuPolyOps;

const poly_mod = @import("zolt_arith").poly;
const tracer = @import("../../tracer/mod.zig");
const ExecutionTrace = tracer.ExecutionTrace;
const jolt_device = @import("../jolt_device.zig");
const preprocessing = @import("../preprocessing.zig");
const BytecodePCMapper = preprocessing.BytecodePCMapper;
const ra_poly_mod = @import("ra_poly.zig");
const UnreducedProductAccum = @import("zolt_arith").field.UnreducedProductAccum;

const stage6_helpers = @import("stage6_helpers.zig");
const computeEqTable = stage6_helpers.computeEqTable;
const computeEqTableParallel = stage6_helpers.computeEqTableParallel;
const fieldFromI128 = stage6_helpers.fieldFromI128;
const extractChunkMSB = stage6_helpers.extractChunkMSB;
const computeLookupIndex = stage6_helpers.computeLookupIndex;
const dropInBackground = stage6_helpers.dropInBackground;

// Maximum evaluation points for parallelReduce accumulator.
// Covers all sub-provers: LookupsRa (M+2 <= 10), RamRa (d+2 <= 6), BytecodeReadRaf (d+2 <= 4).
const MAX_RA_EVALS = 16;

const zkvm_debug = @import("../debug.zig");
const dbg = zkvm_debug.dbg;
const debug_verbose = zkvm_debug.verbose;

// =============================================================================
// IncClaimReduction Sumcheck Instance (Instance 5)
// =============================================================================
// Proves: Sigma_j [RamInc(j) * eq_ram_combined(j) + gamma^2 * RdInc(j) * eq_rd_combined(j)] = input_claim
// where eq_ram_combined = eq(r_stage2, j) + gamma * eq(r_stage4, j)
//       eq_rd_combined  = eq(s_stage4, j) + gamma * eq(s_stage5, j)
// Degree 2: product of two linear polys (Inc x eq)
//
// Two-phase P/Q prefix-suffix split (matches upstream Jolt):
// Phase 1: Operates on sqrt(T)-sized P (prefix eq) and Q (suffix-folded inc) arrays
//   p(t) = Σ_j [P_r2·Q_r2 + γ·P_r4·Q_r4 + γ²·P_s4·Q_s4 + γ³·P_s5·Q_s5]
//   Runs for prefix_n_vars rounds.
// Phase 2: Materializes suffix-sized ram_inc, rd_inc, eq_ram, eq_rd arrays
//   p(t) = Σ_j [ram_inc·eq_ram + γ²·rd_inc·eq_rd]
//   Runs for suffix_n_vars rounds.
pub fn IncClaimReductionProver(comptime F: type) type {
    return struct {
        const Self = @This();
        const Phase = enum { phase1, phase2 };

        phase: Phase,
        // Phase 1 state: prefix eq tables (P) and suffix-folded polys (Q)
        // Indices: [0]=r_stage2, [1]=r_stage4, [2]=s_stage4, [3]=s_stage5
        P: [4][]F, // prefix eq tables, prefix_len each
        Q: [4][]F, // suffix-folded inc polys, prefix_len each
        eq_hi: [4][]F, // suffix eq tables, suffix_len each (kept for Phase 2 transition)
        p1_current_len: usize, // Phase 1 current len (prefix_len → prefix_len/2 → ...)
        challenges: []F, // pre-allocated for prefix_n_vars challenges
        num_challenges: usize,

        // Phase 2 state: suffix-sized dense arrays
        ram_inc: []F,
        rd_inc: []F,
        eq_ram: []F,
        eq_rd: []F,
        p2_current_len: usize,

        // Shared
        gamma: F,
        gamma_sqr: F,
        gamma_cub: F,
        prefix_n_vars: usize,
        suffix_n_vars: usize,
        n_vars: usize,
        /// Caller-owned trace; must outlive Phase 1→2 transition (not accessed after).
        trace: *const ExecutionTrace,
        /// Original BE opening points (caller-owned); must outlive Phase 1→2 transition.
        points_be: [4][]const F,
        allocator: Allocator,
        pool: ?*ThreadPool = null,
        gpu: ?*GpuPolyOps = null,

        /// Compute scalar MLE: eq(a, b) = Π_i (a_i·b_i + (1-a_i)·(1-b_i))
        fn computeMle(a: []const F, b: []const F) F {
            var result = F.one();
            for (0..a.len) |i| {
                const prod = a[i].mul(b[i]);
                const sum = a[i].add(b[i]);
                // a·b + (1-a)·(1-b) = 2·a·b + 1 - a - b
                result = result.mul(prod.add(prod).add(F.one()).sub(sum));
            }
            return result;
        }

        pub fn init(
            allocator: Allocator,
            trace: *const ExecutionTrace,
            gamma: F,
            r_cycle_stage2: []const F,
            r_cycle_stage4: []const F,
            s_cycle_stage4: []const F,
            s_cycle_stage5: []const F,
            pool: ?*ThreadPool,
        ) !Self {
            const T = trace.steps.items.len;
            const n_vars = std.math.log2_int(usize, T);
            const prefix_n_vars = n_vars / 2;
            const suffix_n_vars = n_vars - prefix_n_vars;
            const prefix_len: usize = @as(usize, 1) << @intCast(prefix_n_vars);
            const suffix_len: usize = @as(usize, 1) << @intCast(suffix_n_vars);

            const points_be = [4][]const F{
                r_cycle_stage2, r_cycle_stage4, s_cycle_stage4, s_cycle_stage5,
            };

            // For each point, split into lo (prefix) and hi (suffix) BE halves,
            // reverse each to LE, then compute eq tables.
            // BE: [0..suffix_n_vars] = hi (MSB), [suffix_n_vars..n_vars] = lo (LSB)
            // LE lo = reverse(be_lo), LE hi = reverse(be_hi)
            var P: [4][]F = undefined;
            var eq_hi: [4][]F = undefined;

            var rev_lo = try allocator.alloc(F, prefix_n_vars);
            defer allocator.free(rev_lo);
            var rev_hi = try allocator.alloc(F, suffix_n_vars);
            defer allocator.free(rev_hi);

            for (0..4) |i| {
                // LE lo: reverse of BE[suffix_n_vars..n_vars]
                for (0..prefix_n_vars) |k| {
                    rev_lo[k] = points_be[i][n_vars - 1 - k];
                }
                P[i] = try computeEqTableParallel(F, allocator, rev_lo, prefix_n_vars, pool);

                // LE hi: reverse of BE[0..suffix_n_vars]
                for (0..suffix_n_vars) |k| {
                    rev_hi[k] = points_be[i][suffix_n_vars - 1 - k];
                }
                eq_hi[i] = try computeEqTableParallel(F, allocator, rev_hi, suffix_n_vars, pool);
            }

            // Q[i][x_lo] = Σ_{x_hi} Inc(x_lo + x_hi << prefix_n_vars) * eq_hi[i][x_hi]
            // Q[0], Q[1] for RamInc at points 0,1
            // Q[2], Q[3] for RdInc at points 2,3
            var Q: [4][]F = undefined;
            for (0..4) |i| {
                Q[i] = try allocator.alloc(F, prefix_len);
            }

            const QCtx = struct {
                steps: []const tracer.TraceStep,
                eq_hi: [4][]const F,
                Q: [4][]F,
                prefix_n_vars: std.math.Log2Int(usize),
                suffix_len: usize,
            };
            const q_ctx = QCtx{
                .steps = trace.steps.items,
                .eq_hi = .{ eq_hi[0], eq_hi[1], eq_hi[2], eq_hi[3] },
                .Q = Q,
                .prefix_n_vars = @intCast(prefix_n_vars),
                .suffix_len = suffix_len,
            };
            const qFn = struct {
                fn f(c: QCtx, x_lo: usize) void {
                    var acc: [4]F = .{ F.zero(), F.zero(), F.zero(), F.zero() };
                    for (0..c.suffix_len) |x_hi| {
                        const x = x_lo + (x_hi << c.prefix_n_vars);
                        const step = c.steps[x];

                        // RamInc
                        var ram_inc = F.zero();
                        if (step.is_memory_write) {
                            const mem_post: i128 = @intCast(step.memory_value orelse 0);
                            const mem_pre: i128 = @intCast(step.memory_pre_value orelse 0);
                            ram_inc = fieldFromI128(F, mem_post - mem_pre);
                        }
                        // RdInc
                        var rd_inc_val = F.zero();
                        if (!step.is_noop and step.rd_written and step.rd_index != 0) {
                            rd_inc_val = F.fromU64(step.rd_value).sub(F.fromU64(step.rd_pre_value));
                        }

                        acc[0] = acc[0].add(c.eq_hi[0][x_hi].mul(ram_inc));
                        acc[1] = acc[1].add(c.eq_hi[1][x_hi].mul(ram_inc));
                        acc[2] = acc[2].add(c.eq_hi[2][x_hi].mul(rd_inc_val));
                        acc[3] = acc[3].add(c.eq_hi[3][x_hi].mul(rd_inc_val));
                    }
                    c.Q[0][x_lo] = acc[0];
                    c.Q[1][x_lo] = acc[1];
                    c.Q[2][x_lo] = acc[2];
                    c.Q[3][x_lo] = acc[3];
                }
            }.f;

            pool_helpers.parallelForOptional(pool, prefix_len, q_ctx, qFn);

            const challenges_buf = try allocator.alloc(F, prefix_n_vars);
            @memset(challenges_buf, F.zero());

            return Self{
                .phase = .phase1,
                .P = P,
                .Q = Q,
                .eq_hi = eq_hi,
                .p1_current_len = prefix_len,
                .challenges = challenges_buf,
                .num_challenges = 0,
                .ram_inc = &[_]F{},
                .rd_inc = &[_]F{},
                .eq_ram = &[_]F{},
                .eq_rd = &[_]F{},
                .p2_current_len = 0,
                .gamma = gamma,
                .gamma_sqr = gamma.mul(gamma),
                .gamma_cub = gamma.mul(gamma).mul(gamma),
                .prefix_n_vars = prefix_n_vars,
                .suffix_n_vars = suffix_n_vars,
                .n_vars = n_vars,
                .trace = trace,
                .points_be = points_be,
                .allocator = allocator,
                .pool = pool,
            };
        }

        pub fn deinit(self: *Self) void {
            switch (self.phase) {
                .phase1 => {
                    for (0..4) |i| {
                        self.allocator.free(self.P[i]);
                        self.allocator.free(self.Q[i]);
                        self.allocator.free(self.eq_hi[i]);
                    }
                    self.allocator.free(self.challenges);
                },
                .phase2 => {
                    self.allocator.free(self.ram_inc);
                    self.allocator.free(self.rd_inc);
                    self.allocator.free(self.eq_ram);
                    self.allocator.free(self.eq_rd);
                },
            }
        }

        /// Phase 1 round polynomial: P·Q products with gamma weighting
        fn computeRoundPolyPhase1(self: *Self) [3]F {
            const half = self.p1_current_len / 2;

            const Ctx = struct {
                P: [4][]const F,
                Q: [4][]const F,
                gamma: F,
                gamma_sqr: F,
                gamma_cub: F,
            };
            const ctx = Ctx{
                .P = .{ self.P[0], self.P[1], self.P[2], self.P[3] },
                .Q = .{ self.Q[0], self.Q[1], self.Q[2], self.Q[3] },
                .gamma = self.gamma,
                .gamma_sqr = self.gamma_sqr,
                .gamma_cub = self.gamma_cub,
            };

            const mapFn = struct {
                fn f(c: Ctx, start: usize, end: usize) [3]F {
                    var e0 = F.zero();
                    var e1 = F.zero();
                    var e2 = F.zero();
                    const weights = [4]F{ F.one(), c.gamma, c.gamma_sqr, c.gamma_cub };

                    for (start..end) |j| {
                        inline for (0..4) |k| {
                            const p0 = c.P[k][2 * j];
                            const p1 = c.P[k][2 * j + 1];
                            const q0 = c.Q[k][2 * j];
                            const q1 = c.Q[k][2 * j + 1];

                            e0 = e0.add(weights[k].mul(p0.mul(q0)));
                            e1 = e1.add(weights[k].mul(p1.mul(q1)));

                            const p2 = p1.add(p1).sub(p0);
                            const q2 = q1.add(q1).sub(q0);
                            e2 = e2.add(weights[k].mul(p2.mul(q2)));
                        }
                    }
                    return [3]F{ e0, e1, e2 };
                }
            }.f;

            const reduceFn = struct {
                fn f(a: [3]F, b: [3]F) [3]F {
                    return [3]F{ a[0].add(b[0]), a[1].add(b[1]), a[2].add(b[2]) };
                }
            }.f;

            return pool_helpers.parallelReduceOptional([3]F, self.pool, half, [3]F{ F.zero(), F.zero(), F.zero() }, ctx, mapFn, reduceFn);
        }

        /// Phase 2 round polynomial: ram_inc·eq_ram + γ²·rd_inc·eq_rd
        fn computeRoundPolyPhase2(self: *Self) [3]F {
            const half = self.p2_current_len / 2;

            const Ctx = struct {
                ram_inc: []const F,
                rd_inc: []const F,
                eq_ram: []const F,
                eq_rd: []const F,
                gamma_sqr: F,
            };
            const ctx = Ctx{
                .ram_inc = self.ram_inc,
                .rd_inc = self.rd_inc,
                .eq_ram = self.eq_ram,
                .eq_rd = self.eq_rd,
                .gamma_sqr = self.gamma_sqr,
            };

            const mapFn = struct {
                fn f(c: Ctx, start: usize, end: usize) [3]F {
                    var e0 = F.zero();
                    var e1 = F.zero();
                    var e2 = F.zero();
                    for (start..end) |j| {
                        const ram0 = c.ram_inc[2 * j];
                        const ram1 = c.ram_inc[2 * j + 1];
                        const eq_r0 = c.eq_ram[2 * j];
                        const eq_r1 = c.eq_ram[2 * j + 1];
                        const rd0 = c.rd_inc[2 * j];
                        const rd1 = c.rd_inc[2 * j + 1];
                        const eq_d0 = c.eq_rd[2 * j];
                        const eq_d1 = c.eq_rd[2 * j + 1];

                        e0 = e0.add(ram0.mul(eq_r0).add(c.gamma_sqr.mul(rd0.mul(eq_d0))));
                        e1 = e1.add(ram1.mul(eq_r1).add(c.gamma_sqr.mul(rd1.mul(eq_d1))));

                        const ram2 = ram1.add(ram1).sub(ram0);
                        const eq_r2 = eq_r1.add(eq_r1).sub(eq_r0);
                        const rd2 = rd1.add(rd1).sub(rd0);
                        const eq_d2 = eq_d1.add(eq_d1).sub(eq_d0);
                        e2 = e2.add(ram2.mul(eq_r2).add(c.gamma_sqr.mul(rd2.mul(eq_d2))));
                    }
                    return [3]F{ e0, e1, e2 };
                }
            }.f;

            const reduceFn = struct {
                fn f(a: [3]F, b: [3]F) [3]F {
                    return [3]F{ a[0].add(b[0]), a[1].add(b[1]), a[2].add(b[2]) };
                }
            }.f;

            return pool_helpers.parallelReduceOptional([3]F, self.pool, half, [3]F{ F.zero(), F.zero(), F.zero() }, ctx, mapFn, reduceFn);
        }

        pub fn computeRoundPoly(self: *Self) [3]F {
            return switch (self.phase) {
                .phase1 => self.computeRoundPolyPhase1(),
                .phase2 => self.computeRoundPolyPhase2(),
            };
        }

        /// Transition from Phase 1 to Phase 2
        fn transitionToPhase2(self: *Self, last_challenge: F) !void {
            // Store the final challenge
            self.challenges[self.num_challenges] = last_challenge;
            self.num_challenges += 1;

            const prefix_n_vars = self.prefix_n_vars;
            const suffix_n_vars = self.suffix_n_vars;
            const n_vars = self.n_vars;
            const suffix_len: usize = @as(usize, 1) << @intCast(suffix_n_vars);
            const prefix_len: usize = @as(usize, 1) << @intCast(prefix_n_vars);

            // Compute eq_prefix table: eq(challenges, x_lo) for each x_lo
            const eq_prefix = try computeEqTableParallel(F, self.allocator, self.challenges[0..prefix_n_vars], prefix_n_vars, self.pool);
            defer self.allocator.free(eq_prefix);

            // Compute scalar MLE: eq(challenges_LE → BE, point_lo_BE) for each point
            // challenges are in LE order; point_lo_LE = reverse(point_be[suffix_n_vars..])
            var point_lo_le = try self.allocator.alloc(F, prefix_n_vars);
            defer self.allocator.free(point_lo_le);

            var eq_prefix_scalars: [4]F = undefined;
            for (0..4) |i| {
                for (0..prefix_n_vars) |k| {
                    point_lo_le[k] = self.points_be[i][n_vars - 1 - k];
                }
                eq_prefix_scalars[i] = computeMle(self.challenges[0..prefix_n_vars], point_lo_le);
            }

            // Build combined eq arrays: eq_ram[x_hi] = scalar_r2·eq_hi_r2[x_hi] + γ·scalar_r4·eq_hi_r4[x_hi]
            const eq_ram_arr = try self.allocator.alloc(F, suffix_len);
            errdefer self.allocator.free(eq_ram_arr);
            const eq_rd_arr = try self.allocator.alloc(F, suffix_len);
            errdefer self.allocator.free(eq_rd_arr);

            const scale_r2 = eq_prefix_scalars[0];
            const scale_r4 = eq_prefix_scalars[1];
            const scale_s4 = eq_prefix_scalars[2];
            const scale_s5 = eq_prefix_scalars[3];

            const EqP2Ctx = struct {
                eq_hi_0: []const F,
                eq_hi_1: []const F,
                eq_hi_2: []const F,
                eq_hi_3: []const F,
                eq_ram_out: []F,
                eq_rd_out: []F,
                scale_r2: F,
                scale_r4: F,
                scale_s4: F,
                scale_s5: F,
                gamma: F,
            };
            const eq_ctx = EqP2Ctx{
                .eq_hi_0 = self.eq_hi[0],
                .eq_hi_1 = self.eq_hi[1],
                .eq_hi_2 = self.eq_hi[2],
                .eq_hi_3 = self.eq_hi[3],
                .eq_ram_out = eq_ram_arr,
                .eq_rd_out = eq_rd_arr,
                .scale_r2 = scale_r2,
                .scale_r4 = scale_r4,
                .scale_s4 = scale_s4,
                .scale_s5 = scale_s5,
                .gamma = self.gamma,
            };
            const eqP2Fn = struct {
                fn f(c: EqP2Ctx, x_hi: usize) void {
                    c.eq_ram_out[x_hi] = c.scale_r2.mul(c.eq_hi_0[x_hi]).add(c.gamma.mul(c.scale_r4.mul(c.eq_hi_1[x_hi])));
                    c.eq_rd_out[x_hi] = c.scale_s4.mul(c.eq_hi_2[x_hi]).add(c.gamma.mul(c.scale_s5.mul(c.eq_hi_3[x_hi])));
                }
            }.f;
            pool_helpers.parallelForOptional(self.pool, suffix_len, eq_ctx, eqP2Fn);

            // Materialize ram_inc and rd_inc by folding trace over prefix dimension
            const ram_inc_arr = try self.allocator.alloc(F, suffix_len);
            errdefer self.allocator.free(ram_inc_arr);
            const rd_inc_arr = try self.allocator.alloc(F, suffix_len);
            errdefer self.allocator.free(rd_inc_arr);

            const IncP2Ctx = struct {
                steps: []const tracer.TraceStep,
                eq_prefix: []const F,
                ram_inc_out: []F,
                rd_inc_out: []F,
                prefix_len: usize,
                prefix_n_vars: std.math.Log2Int(usize),
            };
            const inc_ctx = IncP2Ctx{
                .steps = self.trace.steps.items,
                .eq_prefix = eq_prefix,
                .ram_inc_out = ram_inc_arr,
                .rd_inc_out = rd_inc_arr,
                .prefix_len = prefix_len,
                .prefix_n_vars = @intCast(prefix_n_vars),
            };
            const incP2Fn = struct {
                fn f(c: IncP2Ctx, x_hi: usize) void {
                    var acc_ram = F.zero();
                    var acc_rd = F.zero();
                    for (0..c.prefix_len) |x_lo| {
                        const x = x_lo + (x_hi << c.prefix_n_vars);
                        const step = c.steps[x];
                        const eq_val = c.eq_prefix[x_lo];

                        if (step.is_memory_write) {
                            const mem_post: i128 = @intCast(step.memory_value orelse 0);
                            const mem_pre: i128 = @intCast(step.memory_pre_value orelse 0);
                            acc_ram = acc_ram.add(eq_val.mul(fieldFromI128(F, mem_post - mem_pre)));
                        }
                        if (!step.is_noop and step.rd_written and step.rd_index != 0) {
                            acc_rd = acc_rd.add(eq_val.mul(F.fromU64(step.rd_value).sub(F.fromU64(step.rd_pre_value))));
                        }
                    }
                    c.ram_inc_out[x_hi] = acc_ram;
                    c.rd_inc_out[x_hi] = acc_rd;
                }
            }.f;
            pool_helpers.parallelForOptional(self.pool, suffix_len, inc_ctx, incP2Fn);

            // Free Phase 1 arrays
            for (0..4) |i| {
                self.allocator.free(self.P[i]);
                self.allocator.free(self.Q[i]);
                self.allocator.free(self.eq_hi[i]);
            }
            self.allocator.free(self.challenges);

            // Set Phase 2 state
            self.ram_inc = ram_inc_arr;
            self.rd_inc = rd_inc_arr;
            self.eq_ram = eq_ram_arr;
            self.eq_rd = eq_rd_arr;
            self.p2_current_len = suffix_len;
            self.phase = .phase2;
        }

        pub fn bindChallenge(self: *Self, r: F) !void {
            const bindOne = struct {
                fn f(arr: []F, h: usize, challenge: F) void {
                    for (0..h) |j| {
                        arr[j] = arr[2 * j].add(challenge.mul(arr[2 * j + 1].sub(arr[2 * j])));
                    }
                }
            }.f;

            switch (self.phase) {
                .phase1 => {
                    // Check if we should transition (P has length 2 → last Phase 1 round)
                    if (self.p1_current_len == 2) {
                        try self.transitionToPhase2(r);
                        return;
                    }

                    // Normal Phase 1 bind: bind all 8 P/Q arrays
                    const half = self.p1_current_len / 2;
                    self.challenges[self.num_challenges] = r;
                    self.num_challenges += 1;

                    if (self.gpu) |gpu| {
                        if (half >= 16384) {
                            inline for (0..4) |i| {
                                gpu.polyBindLow(self.P[i][0 .. half * 2], r, self.P[i][0..half]) catch bindOne(self.P[i], half, r);
                                gpu.polyBindLow(self.Q[i][0 .. half * 2], r, self.Q[i][0..half]) catch bindOne(self.Q[i], half, r);
                            }
                        } else {
                            inline for (0..4) |i| {
                                bindOne(self.P[i], half, r);
                                bindOne(self.Q[i], half, r);
                            }
                        }
                    } else if (self.pool) |pool| {
                        const arrays = [8][]F{ self.P[0], self.P[1], self.P[2], self.P[3], self.Q[0], self.Q[1], self.Q[2], self.Q[3] };
                        const Ctx = struct { arrs: [8][]F, half: usize, r: F };
                        const ctx = Ctx{ .arrs = arrays, .half = half, .r = r };
                        pool.parallelForForce(8, ctx, struct {
                            fn f(c: Ctx, idx: usize) void {
                                bindOne(c.arrs[idx], c.half, c.r);
                            }
                        }.f);
                    } else {
                        inline for (0..4) |i| {
                            bindOne(self.P[i], half, r);
                            bindOne(self.Q[i], half, r);
                        }
                    }
                    self.p1_current_len = half;
                },
                .phase2 => {
                    const half = self.p2_current_len / 2;
                    if (self.gpu) |gpu| {
                        if (half >= 16384) {
                            gpu.polyBindLow(self.ram_inc[0 .. half * 2], r, self.ram_inc[0..half]) catch bindOne(self.ram_inc, half, r);
                            gpu.polyBindLow(self.rd_inc[0 .. half * 2], r, self.rd_inc[0..half]) catch bindOne(self.rd_inc, half, r);
                            gpu.polyBindLow(self.eq_ram[0 .. half * 2], r, self.eq_ram[0..half]) catch bindOne(self.eq_ram, half, r);
                            gpu.polyBindLow(self.eq_rd[0 .. half * 2], r, self.eq_rd[0..half]) catch bindOne(self.eq_rd, half, r);
                        } else {
                            bindOne(self.ram_inc, half, r);
                            bindOne(self.rd_inc, half, r);
                            bindOne(self.eq_ram, half, r);
                            bindOne(self.eq_rd, half, r);
                        }
                    } else if (self.pool) |pool| {
                        const arrays = [4][]F{ self.ram_inc, self.rd_inc, self.eq_ram, self.eq_rd };
                        const Ctx = struct { arrs: [4][]F, half: usize, r: F };
                        const ctx = Ctx{ .arrs = arrays, .half = half, .r = r };
                        pool.parallelForForce(4, ctx, struct {
                            fn f(c: Ctx, idx: usize) void {
                                bindOne(c.arrs[idx], c.half, c.r);
                            }
                        }.f);
                    } else {
                        bindOne(self.ram_inc, half, r);
                        bindOne(self.rd_inc, half, r);
                        bindOne(self.eq_ram, half, r);
                        bindOne(self.eq_rd, half, r);
                    }
                    self.p2_current_len = half;
                },
            }
        }

        pub fn openingClaims(self: *const Self) struct { ram_inc: F, rd_inc: F } {
            std.debug.assert(self.phase == .phase2);
            return .{
                .ram_inc = self.ram_inc[0],
                .rd_inc = self.rd_inc[0],
            };
        }
    };
}

// =============================================================================
// HammingBooleanity Sumcheck Instance (Instance 1)
// =============================================================================
// Proves: Sigma_j eq(r_cycle, j) * (H(j)^2 - H(j)) = 0
// Degree 3: eq is linear * (H^2 - H is quadratic)
//
// Split-eq optimization: replaces T-sized eq table with sqrt(T)-sized E_lo/E_hi.
// Phase 1: eq factored as eq_lo(x_lo) * eq_hi(x_hi), bind eq_lo + H for prefix_n_vars rounds.
// Phase 2: GruenSplitEq with O(1) bind + factored eq compute, bind only H for suffix_n_vars rounds.
pub fn HammingBooleanityProver(comptime F: type) type {
    return struct {
        const Self = @This();
        const Phase = enum { phase1, phase2 };

        H: []F,
        phase: Phase,
        // Phase 1: split eq tables
        eq_lo: []F, // prefix eq, prefix_len → prefix_len/2 → ... → 1
        eq_hi: []F, // suffix eq, constant during Phase 1 (freed at transition)
        // Phase 2: GruenSplitEq replaces dense merged eq (O(1) bind, factored compute)
        gruen_eq: ?poly_mod.GruenSplitEqPolynomial(F),
        // BE suffix challenges for GruenSplitEq construction at Phase 1→2 transition
        r_cycle_suffix: []F,
        current_len: usize, // H length
        prefix_current_len: usize, // eq_lo length (>0 in Phase 1)
        suffix_len: usize,
        allocator: Allocator,
        pool: ?*ThreadPool = null,
        gpu: ?*GpuPolyOps = null,

        pub fn init(
            allocator: Allocator,
            trace: *const ExecutionTrace,
            r_cycle: []const F,
            pool: ?*ThreadPool,
        ) !Self {
            const T = trace.steps.items.len;
            const n_vars = std.math.log2_int(usize, T);
            const prefix_n_vars = n_vars / 2;
            const suffix_n_vars = n_vars - prefix_n_vars;
            const prefix_len: usize = @as(usize, 1) << @intCast(prefix_n_vars);
            const suffix_len: usize = @as(usize, 1) << @intCast(suffix_n_vars);

            const H_arr = try allocator.alloc(F, T);
            const HInitCtx = struct {
                steps: []const tracer.TraceStep,
                H_out: []F,
            };
            const h_init_ctx = HInitCtx{ .steps = trace.steps.items, .H_out = H_arr };
            const hInitFn = struct {
                fn f(c: HInitCtx, j: usize) void {
                    const step = c.steps[j];
                    if (step.memory_addr) |addr| {
                        c.H_out[j] = if (addr != 0) F.one() else F.zero();
                    } else {
                        c.H_out[j] = F.zero();
                    }
                }
            }.f;
            pool_helpers.parallelForOptional(pool, T, h_init_ctx, hInitFn);

            // r_cycle is in BE order; reverse for LE
            var r_cycle_rev = try allocator.alloc(F, n_vars);
            defer allocator.free(r_cycle_rev);
            for (0..n_vars) |i| r_cycle_rev[i] = r_cycle[n_vars - 1 - i];

            // Split eq: E_lo over first prefix_n_vars LE vars, E_hi over remaining
            const eq_lo = try computeEqTableParallel(F, allocator, r_cycle_rev[0..prefix_n_vars], prefix_n_vars, pool);
            const eq_hi = try computeEqTableParallel(F, allocator, r_cycle_rev[prefix_n_vars..n_vars], suffix_n_vars, pool);

            // Store BE suffix challenges for GruenSplitEq construction at Phase 1→2 transition.
            // Suffix variables are bits prefix_n_vars..n_vars-1 of the flat index.
            // In BE order these are r_cycle[0..suffix_n_vars].
            const r_cycle_suffix = try allocator.alloc(F, suffix_n_vars);
            @memcpy(r_cycle_suffix, r_cycle[0..suffix_n_vars]);

            return Self{
                .H = H_arr,
                .phase = .phase1,
                .eq_lo = eq_lo,
                .eq_hi = eq_hi,
                .gruen_eq = null,
                .r_cycle_suffix = r_cycle_suffix,
                .current_len = T,
                .prefix_current_len = prefix_len,
                .suffix_len = suffix_len,
                .allocator = allocator,
                .pool = pool,
            };
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.H);
            self.allocator.free(self.r_cycle_suffix);
            switch (self.phase) {
                .phase1 => {
                    self.allocator.free(self.eq_lo);
                    self.allocator.free(self.eq_hi);
                },
                .phase2 => {
                    if (self.gruen_eq) |*g| g.deinit();
                },
            }
        }

        /// Phase 1: double loop with factored eq = eq_lo(x_lo) * eq_hi(x_hi)
        fn computeRoundPolyPhase1(self: *Self, previous_claim: F) [4]F {
            _ = previous_claim;
            const half_lo = self.prefix_current_len / 2;
            const suffix_len = self.suffix_len;

            const Ctx = struct {
                H: []const F,
                eq_lo: []const F,
                eq_hi: []const F,
                half_lo: usize,
                suffix_len: usize,
            };
            const ctx = Ctx{
                .H = self.H,
                .eq_lo = self.eq_lo,
                .eq_hi = self.eq_hi,
                .half_lo = half_lo,
                .suffix_len = suffix_len,
            };

            // Compute s(0), s(1), s(2), s(3) using deferred reduction (UPA) to
            // amortize Montgomery reduction across multiple products.
            const UPA = UnreducedProductAccum;
            const mapFn = struct {
                fn f(c: Ctx, start: usize, end: usize) [4]F {
                    var upa0 = UPA.zero();
                    var upa1 = UPA.zero();
                    var upa2 = UPA.zero();
                    var upa3 = UPA.zero();

                    for (start..end) |j_outer| {
                        const eq_hi_val = c.eq_hi[j_outer];
                        for (0..c.half_lo) |j_inner| {
                            const j = j_inner + j_outer * c.half_lo;
                            const h0 = c.H[2 * j];
                            const h1 = c.H[2 * j + 1];
                            const h_delta = h1.sub(h0);

                            const eq_lo_0 = c.eq_lo[2 * j_inner];
                            const eq_lo_1 = c.eq_lo[2 * j_inner + 1];
                            const eq0 = eq_lo_0.mul(eq_hi_val);
                            const eq1 = eq_lo_1.mul(eq_hi_val);
                            const e_delta = eq1.sub(eq0);

                            upa0.addAssign(UPA.fromMul(eq0, h0.mul(h0).sub(h0)));
                            upa1.addAssign(UPA.fromMul(eq1, h1.mul(h1).sub(h1)));

                            const h_at_2 = h1.add(h_delta);
                            const e_at_2 = eq1.add(e_delta);
                            upa2.addAssign(UPA.fromMul(e_at_2, h_at_2.mul(h_at_2).sub(h_at_2)));

                            const h_at_3 = h_at_2.add(h_delta);
                            const e_at_3 = e_at_2.add(e_delta);
                            upa3.addAssign(UPA.fromMul(e_at_3, h_at_3.mul(h_at_3).sub(h_at_3)));
                        }
                    }
                    return [4]F{ upa0.reduce(), upa1.reduce(), upa2.reduce(), upa3.reduce() };
                }
            }.f;

            const reduceFn = struct {
                fn f(a: [4]F, b: [4]F) [4]F {
                    return [4]F{ a[0].add(b[0]), a[1].add(b[1]), a[2].add(b[2]), a[3].add(b[3]) };
                }
            }.f;

            if (self.pool) |pool| {
                return pool.parallelReduce([4]F, suffix_len, [4]F{ F.zero(), F.zero(), F.zero(), F.zero() }, ctx, mapFn, reduceFn);
            }
            return mapFn(ctx, 0, suffix_len);
        }

        /// Phase 2: GruenSplitEq factored eq with deferred E_out pattern
        /// Computes quotient q(x) = f(x)/eq(x,r) at {0, ∞} then reconstructs via computeCubicRoundPoly.
        /// q_constant = Σ_j eq_prefix(j) * (H[2j]² - H[2j])
        /// q_quadratic_coeff = Σ_j eq_prefix(j) * (H[2j+1] - H[2j])²
        fn computeRoundPolyPhase2(self: *Self, previous_claim: F) [4]F {
            const gruen = &self.gruen_eq.?;
            const half = self.current_len / 2;

            const eq_tables = gruen.getWindowEqTables(gruen.current_index, 1);
            const E_out = eq_tables.E_out;
            const E_in = eq_tables.E_in;
            const head_in_bits = eq_tables.head_in_bits;
            const in_mask = (@as(usize, 1) << @intCast(head_in_bits)) -| 1;

            const Ctx = struct {
                H: []const F,
                E_out: []const F,
                E_in: []const F,
                in_mask: usize,
                head_in_bits: usize,
            };
            const ctx = Ctx{
                .H = self.H,
                .E_out = E_out,
                .E_in = E_in,
                .in_mask = in_mask,
                .head_in_bits = head_in_bits,
            };

            const UPA = UnreducedProductAccum;
            const mapFn = struct {
                fn f(c: Ctx, start: usize, end: usize) [2]F {
                    var q_const_outer = UPA.zero();
                    var q_quad_outer = UPA.zero();
                    var q_const_inner = UPA.zero();
                    var q_quad_inner = UPA.zero();
                    var prev_x_out: usize = if (start > 0) start >> @intCast(c.head_in_bits) else 0;
                    var started = false;

                    for (start..end) |j| {
                        const x_out = j >> @intCast(c.head_in_bits);
                        const x_in = j & c.in_mask;

                        // Flush inner accumulators when x_out changes
                        if (started and x_out != prev_x_out) {
                            const e_out = if (prev_x_out < c.E_out.len) c.E_out[prev_x_out] else F.one();
                            q_const_outer.addAssign(e_out.mulToProductAccum(q_const_inner.reduce()));
                            q_quad_outer.addAssign(e_out.mulToProductAccum(q_quad_inner.reduce()));
                            q_const_inner = UPA.zero();
                            q_quad_inner = UPA.zero();
                        }
                        prev_x_out = x_out;
                        started = true;

                        const e_in = if (x_in < c.E_in.len) c.E_in[x_in] else F.one();

                        const h0 = c.H[2 * j];
                        const h1 = c.H[2 * j + 1];
                        const delta = h1.sub(h0);

                        // q_constant contribution: e_in * (h0² - h0)
                        q_const_inner.addAssign(e_in.mulToProductAccum(h0.mul(h0).sub(h0)));
                        // q_quadratic_coeff contribution: e_in * delta²
                        q_quad_inner.addAssign(e_in.mulToProductAccum(delta.mul(delta)));
                    }
                    // Flush final block
                    if (started) {
                        const e_out = if (prev_x_out < c.E_out.len) c.E_out[prev_x_out] else F.one();
                        q_const_outer.addAssign(e_out.mulToProductAccum(q_const_inner.reduce()));
                        q_quad_outer.addAssign(e_out.mulToProductAccum(q_quad_inner.reduce()));
                    }
                    return .{ q_const_outer.reduce(), q_quad_outer.reduce() };
                }
            }.f;

            const reduceFn = struct {
                fn f(a: [2]F, b: [2]F) [2]F {
                    return .{ a[0].add(b[0]), a[1].add(b[1]) };
                }
            }.f;

            const result = if (self.pool) |pool|
                pool.parallelReduce([2]F, half, [2]F{ F.zero(), F.zero() }, ctx, mapFn, reduceFn)
            else
                mapFn(ctx, 0, half);

            return gruen.computeCubicRoundPoly(result[0], result[1], previous_claim);
        }

        pub fn computeRoundPoly(self: *Self, previous_claim: F) [4]F {
            return switch (self.phase) {
                .phase1 => self.computeRoundPolyPhase1(previous_claim),
                .phase2 => self.computeRoundPolyPhase2(previous_claim),
            };
        }

        pub fn bindChallenge(self: *Self, r: F) !void {
            const bindOne = struct {
                fn f(arr: []F, h: usize, challenge: F) void {
                    for (0..h) |j| {
                        arr[j] = arr[2 * j].add(challenge.mul(arr[2 * j + 1].sub(arr[2 * j])));
                    }
                }
            }.f;

            switch (self.phase) {
                .phase1 => {
                    const half = self.current_len / 2;
                    const half_lo = self.prefix_current_len / 2;

                    // eq_lo is tiny (sqrt(T)), bind inline. H is large — use GPU if available.
                    bindOne(self.eq_lo, half_lo, r);
                    if (self.gpu) |gpu| {
                        if (half >= 16384) {
                            gpu.polyBindLow(self.H[0 .. half * 2], r, self.H[0..half]) catch bindOne(self.H, half, r);
                        } else {
                            bindOne(self.H, half, r);
                        }
                    } else {
                        bindOne(self.H, half, r);
                    }
                    self.current_len = half;
                    self.prefix_current_len = half_lo;

                    // Transition to Phase 2 when eq_lo reaches length 1
                    if (half_lo == 1) {
                        const eq_lo_scalar = self.eq_lo[0];
                        self.allocator.free(self.eq_lo);
                        self.allocator.free(self.eq_hi);

                        // Create GruenSplitEq with the accumulated eq_lo scalar as scaling factor.
                        // r_cycle_suffix is in BE order (r_cycle[0..suffix_n_vars]),
                        // matching GruenSplitEq's expected convention.
                        self.gruen_eq = try poly_mod.GruenSplitEqPolynomial(F).initWithScaling(
                            self.allocator,
                            self.r_cycle_suffix,
                            eq_lo_scalar,
                        );
                        self.phase = .phase2;
                    }
                },
                .phase2 => {
                    const half = self.current_len / 2;
                    // GruenSplitEq: O(1) bind (no dense array to halve)
                    self.gruen_eq.?.bind(r);
                    // H is dense — bind with GPU or CPU
                    if (self.gpu) |gpu| {
                        if (half >= 16384) {
                            gpu.polyBindLow(self.H[0 .. half * 2], r, self.H[0..half]) catch bindOne(self.H, half, r);
                        } else {
                            bindOne(self.H, half, r);
                        }
                    } else {
                        bindOne(self.H, half, r);
                    }
                    self.current_len = half;
                },
            }
        }

        pub fn openingClaim(self: *const Self) F {
            return self.H[0];
        }
    };
}

// =============================================================================
// RamRaVirtual Sumcheck Instance (Instance 3)
// =============================================================================
// Proves: Sigma_c eq(r_cycle_reduced, c) * Prod_{i=0}^{d-1} ra_i(r_addr_chunk_i, c) = claim
// Variables: n_cycle_vars
// Degree: d+1 (product of d linear ra_i * one linear eq)
pub fn RamRaVirtualProver(comptime F: type) type {
    const RaPoly = ra_poly_mod.RaPolynomial(F);

    return struct {
        const Self = @This();

        /// In-place MLE bind: arr[j] = arr[2j] + challenge*(arr[2j+1] - arr[2j]) for j < h.
        /// Sequential only (write[j] aliases future read[2j], cannot parallelize within one array).
        fn bindSlice(arr: []F, h: usize, challenge: F) void {
            for (0..h) |j| {
                arr[j] = arr[2 * j].add(challenge.mul(arr[2 * j + 1].sub(arr[2 * j])));
            }
        }

        /// Compressed ra polynomials (u8 indices in round 1, dense after bind)
        ra_polys: []RaPoly,
        /// GruenSplitEq for eq(r_cycle, .) — O(1) bind
        gruen_eq: poly_mod.GruenSplitEqPolynomial(F),
        d: usize,
        current_len: usize,
        allocator: Allocator,
        pool: ?*ThreadPool = null,
        gpu: ?*GpuPolyOps = null,

        pub fn init(
            allocator: Allocator,
            trace: *const ExecutionTrace,
            r_cycle: []const F, // BIG_ENDIAN
            r_addr_chunks: []const []const F, // r_addr_chunks[i] is BIG_ENDIAN
            d: usize,
            memory_layout: *const jolt_device.MemoryLayout,
            log_k_chunk: usize,
            init_pool: ?*ThreadPool,
        ) !Self {
            const T = trace.steps.items.len;
            const n_vars = std.math.log2_int(usize, T);
            const k_chunk: usize = @as(usize, 1) << @intCast(log_k_chunk);

            // u8 indices can represent chunk values up to 255
            std.debug.assert(log_k_chunk <= ra_poly_mod.MAX_LOG_K_CHUNK);

            var ra_polys = try allocator.alloc(RaPoly, d);
            // Track how many ra_polys have been assembled for safe errdefer cleanup
            var ra_polys_assembled: usize = 0;
            errdefer {
                for (ra_polys[0..ra_polys_assembled]) |*rp| rp.deinit(allocator);
                allocator.free(ra_polys);
            }

            // Pre-allocate all d index arrays and eq_tables
            // eq_tables are owned by RaPolynomial (freed on bind/deinit)
            var indices_arr = try allocator.alloc([]?u8, d);
            defer allocator.free(indices_arr);
            var eq_tables = try allocator.alloc([]F, d);
            defer allocator.free(eq_tables); // only frees the pointer array, not contents

            // Track allocation progress for errdefer cleanup (before assembly into RaPolys)
            var indices_allocated: usize = 0;
            var eq_tables_allocated: usize = 0;
            errdefer {
                for (0..eq_tables_allocated) |i| allocator.free(eq_tables[i]);
                for (0..indices_allocated) |i| allocator.free(indices_arr[i]);
            }

            for (0..d) |i| {
                indices_arr[i] = try allocator.alloc(?u8, T);
                indices_allocated += 1;
                var r_chunk_rev = try allocator.alloc(F, log_k_chunk);
                defer allocator.free(r_chunk_rev);
                for (0..log_k_chunk) |ci| r_chunk_rev[ci] = r_addr_chunks[i][log_k_chunk - 1 - ci];
                eq_tables[i] = try computeEqTable(F, allocator, r_chunk_rev, log_k_chunk);
                eq_tables_allocated += 1;
            }

            // Parallel fill: each chunk i is independent
            const RamRaInitCtx = struct {
                steps: []const tracer.TraceStep,
                indices_arr: [][]?u8,
                memory_layout: *const jolt_device.MemoryLayout,
                d: usize,
                log_k_chunk: usize,
                k_chunk: usize,
            };
            const ram_ra_ctx = RamRaInitCtx{
                .steps = trace.steps.items,
                .indices_arr = indices_arr,
                .memory_layout = memory_layout,
                .d = d,
                .log_k_chunk = log_k_chunk,
                .k_chunk = k_chunk,
            };
            const ramRaInitFn = struct {
                fn f(c: RamRaInitCtx, i: usize) void {
                    for (0..c.steps.len) |j| {
                        const step = c.steps[j];
                        if (step.memory_addr) |addr| {
                            if (addr == 0) {
                                c.indices_arr[i][j] = null;
                            } else {
                                const remapped = c.memory_layout.remapAddress(addr);
                                if (remapped) |raddr| {
                                    const chunk_val = extractChunkMSB(raddr, i, c.d, c.log_k_chunk);
                                    if (chunk_val < c.k_chunk) {
                                        c.indices_arr[i][j] = @intCast(chunk_val);
                                    } else {
                                        c.indices_arr[i][j] = null;
                                    }
                                } else {
                                    c.indices_arr[i][j] = null;
                                }
                            }
                        } else {
                            c.indices_arr[i][j] = null;
                        }
                    }
                }
            }.f;
            pool_helpers.parallelForOptional(init_pool, d, ram_ra_ctx, ramRaInitFn);

            // Assemble RaPolynomials (prescales eq_table by scale=1, validates invariants).
            // Ownership of indices_arr[i] and eq_tables[i] transfers to the RaPoly;
            // clear allocation counters so the pre-assembly errdefer won't double-free.
            for (0..d) |i| {
                ra_polys[i] = RaPoly.initRound1(indices_arr[i], eq_tables[i], F.one());
                ra_polys_assembled += 1;
            }
            indices_allocated = 0;
            eq_tables_allocated = 0;

            // r_cycle is in BE order; pass directly to GruenSplitEq
            const gruen_eq = try poly_mod.GruenSplitEqPolynomial(F).init(allocator, r_cycle[0..n_vars]);

            return Self{
                .ra_polys = ra_polys,
                .gruen_eq = gruen_eq,
                .d = d,
                .current_len = T,
                .allocator = allocator,
                .pool = init_pool,
            };
        }

        pub fn deinit(self: *Self) void {
            for (self.ra_polys) |*rp| rp.deinit(self.allocator);
            self.allocator.free(self.ra_polys);
            self.gruen_eq.deinit();
        }

        /// f(x) = eq(x,r) * Prod_i ra_i(x), degree = d + 1
        /// Uses quotient polynomial approach: factor out eq(x,r), compute quotient
        /// q(x) = Prod_i ra_i(x) at Toom points {1, 2, ..., d-1, ∞}, then
        /// reconstruct f(x) via finishMlesProductSumFromEvals.
        /// Returns monomial coefficients.
        pub fn computeRoundPoly(self: *Self, allocator: Allocator, claim: F) ![]F {
            const half = self.current_len / 2;
            const n_toom_evals = self.d;

            // Get factored eq tables from GruenSplitEq
            const eq_tables = self.gruen_eq.getWindowEqTables(self.gruen_eq.current_index, 1);
            const E_out = eq_tables.E_out;
            const E_in = eq_tables.E_in;
            const head_in_bits = eq_tables.head_in_bits;
            const in_mask = (@as(usize, 1) << @intCast(head_in_bits)) -| 1;

            const Ctx = struct {
                ra_polys: []RaPoly,
                E_out: []const F,
                E_in: []const F,
                in_mask: usize,
                head_in_bits: usize,
                d: usize,
                n_toom_evals: usize,
            };
            const ctx = Ctx{
                .ra_polys = self.ra_polys,
                .E_out = E_out,
                .E_in = E_in,
                .in_mask = in_mask,
                .head_in_bits = head_in_bits,
                .d = self.d,
                .n_toom_evals = n_toom_evals,
            };

            const UPA = UnreducedProductAccum;
            const mapFn = struct {
                fn f(c: Ctx, start: usize, end: usize) [MAX_RA_EVALS]F {
                    // Dispatch once on ra_poly state to avoid per-access tag checks.
                    // All ra_polys are in the same state at any given round.
                    return switch (c.ra_polys[0]) {
                        inline else => |_, comptime_tag| fInner(c, start, end, comptime_tag),
                    };
                }

                inline fn fInner(c: Ctx, start: usize, end: usize, comptime tag: anytype) [MAX_RA_EVALS]F {
                    const MAX_D = 8;
                    var outer_acc: [MAX_RA_EVALS]UPA = .{UPA.zero()} ** MAX_RA_EVALS;
                    var inner_acc: [MAX_RA_EVALS]UPA = .{UPA.zero()} ** MAX_RA_EVALS;
                    var prev_x_out: usize = if (start > 0) start >> @intCast(c.head_in_bits) else 0;
                    var started = false;

                    for (start..end) |j| {
                        const x_out = j >> @intCast(c.head_in_bits);
                        const x_in = j & c.in_mask;

                        // Flush inner_acc when x_out changes
                        if (started and x_out != prev_x_out) {
                            const e_out = if (prev_x_out < c.E_out.len) c.E_out[prev_x_out] else F.one();
                            for (0..c.n_toom_evals) |k| {
                                outer_acc[k].addAssign(e_out.mulToProductAccum(inner_acc[k].reduce()));
                                inner_acc[k] = UPA.zero();
                            }
                        }
                        prev_x_out = x_out;
                        started = true;

                        const e_in = if (x_in < c.E_in.len) c.E_in[x_in] else F.one();

                        var lo: [MAX_D]F = undefined;
                        var delta: [MAX_D]F = undefined;
                        for (0..c.d) |i| {
                            lo[i] = @field(c.ra_polys[i], @tagName(tag)).getBoundCoeff(2 * j);
                            delta[i] = @field(c.ra_polys[i], @tagName(tag)).getBoundCoeff(2 * j + 1).sub(lo[i]);
                        }

                        var cur: [MAX_D]F = undefined;
                        for (0..c.d) |i| cur[i] = lo[i].add(delta[i]);

                        for (0..c.n_toom_evals -| 1) |k| {
                            if (k > 0) {
                                for (0..c.d) |i| cur[i] = cur[i].add(delta[i]);
                            }
                            var product = cur[0];
                            for (1..c.d) |i| product = product.mul(cur[i]);
                            inner_acc[k].addAssign(e_in.mulToProductAccum(product));
                        }

                        var product = delta[0];
                        for (1..c.d) |i| product = product.mul(delta[i]);
                        inner_acc[c.n_toom_evals - 1].addAssign(e_in.mulToProductAccum(product));
                    }
                    // Flush final block
                    if (started) {
                        const e_out = if (prev_x_out < c.E_out.len) c.E_out[prev_x_out] else F.one();
                        for (0..c.n_toom_evals) |k| {
                            outer_acc[k].addAssign(e_out.mulToProductAccum(inner_acc[k].reduce()));
                        }
                    }
                    var acc: [MAX_RA_EVALS]F = undefined;
                    for (0..MAX_RA_EVALS) |i| acc[i] = outer_acc[i].reduce();
                    return acc;
                }
            }.f;

            const reduceFn = struct {
                fn f(a: [MAX_RA_EVALS]F, b: [MAX_RA_EVALS]F) [MAX_RA_EVALS]F {
                    var r: [MAX_RA_EVALS]F = undefined;
                    for (0..MAX_RA_EVALS) |i| {
                        r[i] = a[i].add(b[i]);
                    }
                    return r;
                }
            }.f;

            const result = if (self.pool) |pool|
                pool.parallelReduce([MAX_RA_EVALS]F, half, .{F.zero()} ** MAX_RA_EVALS, ctx, mapFn, reduceFn)
            else
                mapFn(ctx, 0, half);

            // Scale by current_scalar (accumulated eq from previously bound variables)
            const scalar = self.gruen_eq.current_scalar;
            var toom_evals = try allocator.alloc(F, n_toom_evals);
            defer allocator.free(toom_evals);
            for (0..n_toom_evals) |i| toom_evals[i] = result[i].mul(scalar);

            // Extract r_round and reconstruct full polynomial
            const r_round = self.gruen_eq.tau[self.gruen_eq.current_index - 1];
            return poly_mod.UniPoly(F).finishMlesProductSumFromEvals(allocator, toom_evals, claim, r_round);
        }

        pub fn bindChallenge(self: *Self, r: F) !void {
            const half = self.current_len / 2;

            // After the first bind(), all ra_polys transition from round1→dense simultaneously
            // (they all have the same length T). Check index 0 as representative.
            const all_dense = self.ra_polys.len > 0 and self.ra_polys[0] == .dense;
            if (std.debug.runtime_safety and all_dense) {
                for (self.ra_polys) |rp| std.debug.assert(rp == .dense);
            }

            if (all_dense and self.gpu != null and half >= 16384) {
                // GPU bind: d ra_poly dense arrays + e_out
                const gpu = self.gpu.?;
                for (self.ra_polys) |*rp| {
                    const dense = &rp.dense;
                    const h = dense.current_len / 2;
                    gpu.polyBindLow(dense.coeffs[0 .. h * 2], r, dense.coeffs[0..h]) catch bindSlice(dense.coeffs[0 .. h * 2], h, r);
                    dense.current_len = h;
                }
            } else if (all_dense and self.pool != null) {
                // Parallel bind: d ra_poly dense arrays (eq is O(√T), done separately below)
                const RaBindCtx = struct { ra: []RaPoly, d: usize, half: usize, r: F };
                const ctx = RaBindCtx{ .ra = self.ra_polys, .d = self.d, .half = half, .r = r };
                self.pool.?.parallelForForce(self.d, ctx, struct {
                    fn f(c: RaBindCtx, idx: usize) void {
                        std.debug.assert(c.ra[idx] == .dense);
                        bindSlice(c.ra[idx].dense.coeffs[0 .. c.half * 2], c.half, c.r);
                        c.ra[idx].dense.current_len = c.half;
                    }
                }.f);
            } else {
                // First round (round1→dense transition) or no pool: sequential.
                for (self.ra_polys) |*rp| {
                    try rp.bind(r, self.allocator);
                }
            }

            // GruenSplitEq bind — O(1) instead of O(T/2^round)
            self.gruen_eq.bind(r);

            self.current_len = half;
        }

        pub fn getOpeningClaims(self: *const Self, allocator: Allocator) ![]F {
            var claims = try allocator.alloc(F, self.d);
            for (0..self.d) |i| {
                claims[i] = self.ra_polys[i].finalClaim();
            }
            return claims;
        }
    };
}

// =============================================================================
// Booleanity Sumcheck Instance (Instance 2) - REAL prover
// =============================================================================
// Proves: 0 = Σ_{k,j} eq(r_addr, k) · eq(r_cycle, j) · Σ_i γ^{2i} · (ra_i(k,j)² - ra_i(k,j))
//
// Phase 1: log_k_chunk address rounds (degree 3)
//   Uses G tables (full size K, never halved), expanding table F, and split-eq B.
//   At round m: G stays full, F has size 2^m, B tracks eq(r_addr_fixed, ...).
//   p(X_m) = l(X) * q(X) where l = eq linear part, q = Σ γ^{2i} * G*F*(G*F-1)
//
// Phase 2: n_cycle_vars cycle rounds (degree 3)
//   Uses H tables (initialized from F at transition, halved each round) and eq_cycle D.
//   H[i][j] = eq(r_addr_bound, chunk_i(j)), scaled by eq_r_r.
//
// r_addr and r_cycle are FIXED reference points from Stage 5 InstructionReadRaf.
// G_i[k] = Σ_j eq(r_cycle_fixed, j) * [chunk_i(j) == k]  (pushforward)
pub fn BooleanityProver(comptime F: type) type {
    return struct {
        const Self = @This();

        /// G_i tables (pushforward): G_i[k] = Σ_j eq(r_cycle, j) * [chunk_i(j) == k]
        /// Stays at FULL size K throughout Phase 1 (never halved).
        G: [][]F,
        /// Expanding table F: F[k] = eq(r_bound_so_far, k). Starts size 1, doubles each round.
        F_table: []F,
        /// Current size of F_table
        F_size: usize,
        /// r_address (LE, LowToHigh order) - fixed reference point
        r_address_le: []F,
        /// B_scalar: accumulated eq(r_addr_fixed[bound_vars], r_challenges[bound_vars])
        B_scalar: F,
        /// GruenSplitEq for eq(r_cycle, .) — O(1) bind, factored eq compute
        gruen_eq_cycle: poly_mod.GruenSplitEqPolynomial(F),
        /// Flat eq table (BE convention) for Phase 2 compute
        eq_cycle: []F,
        /// γ^{2i} powers for batching
        gamma_powers_sq: []F,
        /// γ^i powers for pre-scaling (used in Phase 2 Gruen optimization)
        gamma_powers: []F,
        /// Number of RA polynomials
        N: usize,
        /// K = 2^log_k_chunk (address table size)
        K: usize,
        /// log_k_chunk (address rounds)
        log_k_chunk: usize,
        /// n_cycle_vars (cycle rounds)
        n_cycle_vars: usize,
        /// Current round number (0-indexed)
        round: usize,
        /// eq(r_addr_fixed, r_addr_bound) - set at Phase 1→2 transition
        eq_r_r: F,
        /// H tables for Phase 2: H[i][j] = eq(r_addr_bound, chunk_i(j))
        /// Initialized at Phase 1→2 transition. Halved each Phase 2 round.
        /// null during lazy rounds (first 3 Phase 2 rounds).
        H: ?[][]F,
        /// Current table length for Phase 2 (T, then T/2, etc.)
        phase2_len: usize,
        /// Chunk indices for lazy H evaluation: chunk_indices[i][j] = chunk index for poly i, cycle j
        /// Only allocated during lazy rounds (first 3 Phase 2 rounds).
        chunk_indices: ?[][]u8,
        /// Lookup tables for lazy H evaluation. In round1: tables[0][k] = F_table[k].
        /// In round2: tables[0][k]=(1-r)*F[k], tables[1][k]=r*F[k].
        /// In round3: tables[0..4][k] = (1-r1)(1-r0)F[k], (1-r1)(r0)F[k], etc.
        lazy_tables: [4][]F,
        /// Number of valid tables in lazy_tables (1=round1, 2=round2, 4=round3, 0=dense)
        lazy_num_tables: u8,
        /// Flat array holding per-poly pre-scaled tables: prescaled[t * N * K_ext + i * K_ext + k] = gamma^i * table_t[k]
        prescaled_lazy_flat: ?[]F,
        /// K+1 (includes sentinel entry at index K)
        K_ext: usize,
        /// Trace reference for building H tables at transition
        trace: *const ExecutionTrace,
        /// Parameters needed for H table construction
        instruction_d: usize,
        bytecode_d: usize,
        ram_d: usize,
        memory_layout: *const jolt_device.MemoryLayout,
        pc_map: *const BytecodePCMapper,
        allocator: std.mem.Allocator,
        pool: ?*ThreadPool = null,
        gpu: ?*GpuPolyOps = null,
        /// Flat backing storage for chunk_indices (single alloc for all N arrays)
        ci_flat_storage: ?[]u8 = null,

        pub fn init(
            allocator: std.mem.Allocator,
            G_tables: [][]F,
            r_addr_le: []F,
            gruen_eq_in: poly_mod.GruenSplitEqPolynomial(F),
            eq_cycle_table: []F,
            gamma_sq: []F,
            gamma_unsq: []F,
            N_val: usize,
            log_k: usize,
            n_cycle: usize,
            trace: *const ExecutionTrace,
            instr_d: usize,
            bc_d: usize,
            ram_d_val: usize,
            mem_layout: *const jolt_device.MemoryLayout,
            pc_mapper: *const BytecodePCMapper,
        ) !Self {
            const K_val = @as(usize, 1) << @intCast(log_k);
            // Initialize expanding table F with F[0] = 1
            const f_table = try allocator.alloc(F, K_val);
            @memset(f_table, F.zero());
            f_table[0] = F.one();

            // Reverse r_addr_le to match Jolt's binding order (MSB first).
            // Jolt's GruenSplitEqPolynomial binds variables from high index to low:
            //   round 0 binds w[n-1] (MSB), round 1 binds w[n-2], ..., round n-1 binds w[0] (LSB).
            // The G table inner loop uses bit m of k at round m, paired with the
            // eq factor from w[n-1-m]. By reversing, r_addr[m] = w[n-1-m].
            // Debug: print BEFORE reversal
            {
                for (0..log_k) |dbg_i| {
                    const dbg_b = r_addr_le[dbg_i].toBytesBE();
                    dbg("[BOOL_INIT] r_addr_BEFORE[{}] LE=[{x:0>2}{x:0>2}{x:0>2}{x:0>2}]\n", .{
                        dbg_i, dbg_b[31], dbg_b[30], dbg_b[29], dbg_b[28],
                    });
                }
            }

            std.mem.reverse(F, r_addr_le);

            // Debug: print AFTER reversal
            {
                for (0..log_k) |dbg_i| {
                    const dbg_b = r_addr_le[dbg_i].toBytesBE();
                    dbg("[BOOL_INIT] r_addr_AFTER[{}] LE=[{x:0>2}{x:0>2}{x:0>2}{x:0>2}]\n", .{
                        dbg_i, dbg_b[31], dbg_b[30], dbg_b[29], dbg_b[28],
                    });
                }
            }

            return Self{
                .G = G_tables,
                .F_table = f_table,
                .F_size = 1,
                .r_address_le = r_addr_le,
                .B_scalar = F.one(),
                .gruen_eq_cycle = gruen_eq_in,
                .eq_cycle = eq_cycle_table,
                .gamma_powers_sq = gamma_sq,
                .gamma_powers = gamma_unsq,
                .N = N_val,
                .K = K_val,
                .log_k_chunk = log_k,
                .n_cycle_vars = n_cycle,
                .round = 0,
                .eq_r_r = F.zero(),
                .H = null,
                .phase2_len = 0,
                .chunk_indices = null,
                .lazy_tables = .{ &.{}, &.{}, &.{}, &.{} },
                .lazy_num_tables = 0,
                .prescaled_lazy_flat = null,
                .K_ext = K_val + 1,
                .trace = trace,
                .instruction_d = instr_d,
                .bytecode_d = bc_d,
                .ram_d = ram_d_val,
                .memory_layout = mem_layout,
                .pc_map = pc_mapper,
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            // G tables may already be freed in background at Phase 1→2 transition.
            if (self.G.len > 0) {
                for (self.G) |g| self.allocator.free(g);
                self.allocator.free(self.G);
            }
            self.allocator.free(self.F_table);
            self.allocator.free(self.r_address_le);
            self.gruen_eq_cycle.deinit();
            if (self.eq_cycle.len > 0) self.allocator.free(self.eq_cycle);
            self.allocator.free(self.gamma_powers_sq);
            self.allocator.free(self.gamma_powers);
            if (self.H) |ht| {
                for (ht) |h| self.allocator.free(h);
                self.allocator.free(ht);
            }
            if (self.prescaled_lazy_flat) |pl| self.allocator.free(pl);
            if (self.chunk_indices) |ci| {
                if (self.ci_flat_storage) |flat| {
                    self.allocator.free(flat);
                } else {
                    for (ci) |c| self.allocator.free(c);
                }
                self.allocator.free(ci);
            }
            for (0..@as(usize, self.lazy_num_tables)) |i| {
                if (self.lazy_tables[i].len > 0) self.allocator.free(self.lazy_tables[i]);
            }
        }

        /// Get the opening claims from the final H state after all sumcheck rounds.
        /// H[i][0] gives ra_i(ρ_addr, ρ_cycle) after all bindings.
        /// H tables are pre-scaled by γ^i, so we unscale via γ^{-i} = (γ^{-1})^i.
        /// This uses 1 inversion + (N-1) muls instead of N inversions.
        pub fn getBooleanityClaims(self: *const Self, allocator: std.mem.Allocator) ![]F {
            const claims = try allocator.alloc(F, self.N);
            dbg("[BOOL_CLAIMS] phase2_len={}, round={}, N={}\n", .{ self.phase2_len, self.round, self.N });
            if (self.H) |ht| {
                // gamma_powers[1] = γ, so γ^{-1} is a single inversion
                const gamma_inv = self.gamma_powers[1].inverse().?;
                var gamma_inv_i = F.one(); // γ^{-0} = 1

                var all_same_claims = true;
                for (0..self.N) |i| {
                    claims[i] = ht[i][0].mul(gamma_inv_i);
                    gamma_inv_i = gamma_inv_i.mul(gamma_inv);
                    if (i < 5 or i >= self.N - 5 or (i >= 28 and i < 34)) {
                        const hbe = claims[i].toBytesBE();
                        dbg("[BOOL_CLAIMS] H[{}][0]_LE=[{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}]\n", .{
                            i, hbe[31], hbe[30], hbe[29], hbe[28], hbe[27], hbe[26], hbe[25], hbe[24],
                        });
                    }
                    if (i > 0 and !claims[i].eql(claims[0])) all_same_claims = false;
                }
                dbg("[BOOL_CLAIMS] all_same={}\n", .{@intFromBool(all_same_claims)});
            } else {
                @memset(claims, F.zero());
            }
            return claims;
        }

        /// Compute round polynomial evaluations: [s(0), s(1), s(2), s(3)]
        /// Returns 4 evaluation points (NOT [s(0), s(1), s(2), p_inf]).
        /// Phase 1 uses gruen_poly_deg_3 approach (derive Q(1) from previous_claim).
        pub fn computeRoundPoly(self: *Self, allocator: std.mem.Allocator, claim: F) ![]F {
            const evals = try allocator.alloc(F, 4);
            @memset(evals, F.zero());

            if (self.round < self.log_k_chunk) {
                self.computePhase1Poly(evals, claim);
            } else {
                self.computePhase2Poly(evals, claim);
            }

            return evals;
        }

        fn computePhase1Poly(self: *Self, evals: []F, previous_claim: F) void {
            // Gruen poly deg 3 approach (matching Jolt's gruen_poly_deg_3):
            //
            // Compute c = Q(0) (constant of quadratic Q) and e (X² coeff of Q).
            // Derive Q(1) from previous_claim to guarantee s(0)+s(1) = claim.
            // Extrapolate Q(2), compute s(2) and p_inf.
            // Return [s(0), s(1), s(2), p_inf] (Toom-Cook format).

            const m = self.round;
            const f_mask = if (m == 0) 0 else (@as(usize, 1) << @intCast(m)) - 1;
            const upper_bits = self.log_k_chunk - m - 1;

            // Build eq_upper table for the head (unbound, non-summed) address variables.
            // In Jolt's LowToHigh convention, the unbound variables at round m are
            // w[0..n-m-1] in LE order. The head = w[0..n-m-2], Gruen = w[n-m-1].
            // After reversal, r_address_le[i] = w[n-1-i], so:
            //   w[0] = r_address_le[n-1], w[1] = r_address_le[n-2], ..., w[n-m-2] = r_address_le[m+1]
            // We process from w[0] (MSB of eq_upper index) to w[n-m-2] (LSB),
            // i.e., from r_address_le[n-1] down to r_address_le[m+1].
            var eq_upper: [16]F = undefined;
            if (upper_bits == 0) {
                eq_upper[0] = F.one();
            } else {
                eq_upper[0] = F.one();
                var eq_upper_len: usize = 1;
                // Process in DESCENDING order: r_address_le[log_k-1] down to r_address_le[m+1]
                var bit: usize = self.log_k_chunk - 1;
                while (bit >= m + 1) : (bit -= 1) {
                    const w = self.r_address_le[bit];
                    const one_minus_w = F.one().sub(w);
                    var idx: usize = eq_upper_len;
                    while (idx > 0) {
                        idx -= 1;
                        eq_upper[2 * idx + 1] = eq_upper[idx].mul(w);
                        eq_upper[2 * idx] = eq_upper[idx].mul(one_minus_w);
                    }
                    eq_upper_len *= 2;
                    if (bit == 0) break; // prevent underflow on usize
                }
            }

            // Inner loop: compute c (=Q(0), constant of Q) and e (X² coeff of Q)
            // c = Σ_{k:k_m=0} eu * Σ_i γ^{2i} * G*F*(F-1)
            // e = Σ_{all k} eu * Σ_i γ^{2i} * G*F²
            var c = F.zero();
            var e = F.zero();

            for (0..self.K) |k| {
                const k_m = (k >> @intCast(m)) & 1;
                const k_bound = k & f_mask;
                const k_upper = k >> @intCast(m + 1);
                const f_k = if (m == 0) F.one() else self.F_table[k_bound];
                const eu = eq_upper[k_upper];
                const f_sq = f_k.mul(f_k);

                var gamma_G_sum = F.zero();
                for (0..self.N) |i| {
                    gamma_G_sum = gamma_G_sum.add(self.gamma_powers_sq[i].mul(self.G[i][k]));
                }

                // e contribution (all k): eu * Σ_i γ^{2i} * G * F²
                e = e.add(eu.mul(gamma_G_sum).mul(f_sq));

                // c contribution (k_m=0 only): eu * Σ_i γ^{2i} * G*F*(F-1)
                if (k_m == 0) {
                    const G_times_F = gamma_G_sum.mul(f_k);
                    c = c.add(eu.mul(G_times_F.mul(f_k).sub(G_times_F)));
                }
            }

            // Linear eq evaluations: l(X) = eq_0 + b*X where b = eq_slope
            const w_m = self.r_address_le[m];
            const eq_eval_1 = self.B_scalar.mul(w_m);
            const eq_eval_0 = self.B_scalar.sub(eq_eval_1);
            const eq_slope = eq_eval_1.sub(eq_eval_0);
            const eq_eval_2 = eq_eval_1.add(eq_slope);

            // Derive Q(1) from previous_claim (Jolt's gruen_poly_deg_3 approach):
            // s(0) = eq_eval_0 * Q(0) = eq_eval_0 * c
            // s(1) = previous_claim - s(0)
            // Q(1) = s(1) / eq_eval_1
            const s0 = eq_eval_0.mul(c);
            const s1 = previous_claim.sub(s0);
            const q1 = if (eq_eval_1.eql(F.zero())) F.zero() else s1.mul(eq_eval_1.inverse().?);

            // Extrapolate: Q(2) = 2*Q(1) - Q(0) + 2*e
            const e_times_2 = e.add(e);
            const q2 = q1.add(q1).sub(c).add(e_times_2);

            // Extrapolate: Q(3) = 3*Q(1) - 2*Q(0) + 6*e
            const three = F.fromU64(3);
            const six = F.fromU64(6);
            const q3 = three.mul(q1).sub(c.add(c)).add(six.mul(e));

            // l(3) = eq_eval_0 + 3 * eq_slope
            const eq_eval_3 = eq_eval_0.add(three.mul(eq_slope));

            // Return Vandermonde format [s(0), s(1), s(2), s(3)]
            evals[0] = s0;
            evals[1] = s1;
            evals[2] = eq_eval_2.mul(q2);
            evals[3] = eq_eval_3.mul(q3);
        }

        /// Look up the bound coefficient for poly i at position pos in lazy state.
        /// In round1 (1 table): h = tables[0][chunk_indices[i][pos]]
        /// In round2 (2 tables): h = tables[0][ci[i][2*pos]] + tables[1][ci[i][2*pos+1]]
        /// In round3 (4 tables): h = sum of tables[t][ci[i][4*pos+t]] for t=0..3
        inline fn lazyGetCoeff(
            ci: []const []const u8,
            tables: [4][]const F,
            num_tables: u8,
            i: usize,
            pos: usize,
        ) F {
            switch (num_tables) {
                1 => return tables[0][ci[i][pos]],
                2 => return tables[0][ci[i][2 * pos]].add(tables[1][ci[i][2 * pos + 1]]),
                4 => return tables[0][ci[i][4 * pos]].add(tables[1][ci[i][4 * pos + 1]])
                    .add(tables[2][ci[i][4 * pos + 2]]).add(tables[3][ci[i][4 * pos + 3]]),
                else => unreachable,
            }
        }

        /// Like lazyGetCoeff, but reads from the pre-scaled flat array where
        /// prescaled[t * N * K_ext + i * K_ext + k] = gamma^i * table_t[k].
        /// Returns sum over tables, already multiplied by gamma^i (eliminates 2 muls per (j,i)).
        inline fn lazyGetCoeffPrescaled(
            ci: []const []const u8,
            prescaled: []const F,
            N: usize,
            K_ext: usize,
            num_tables: u8,
            i: usize,
            pos: usize,
        ) F {
            switch (num_tables) {
                1 => {
                    const off0 = i * K_ext;
                    return prescaled[off0 + ci[i][pos]];
                },
                2 => {
                    const stride = N * K_ext;
                    const off0 = i * K_ext;
                    const off1 = stride + i * K_ext;
                    return prescaled[off0 + ci[i][2 * pos]].add(prescaled[off1 + ci[i][2 * pos + 1]]);
                },
                4 => {
                    const stride = N * K_ext;
                    const off0 = i * K_ext;
                    const off1 = stride + i * K_ext;
                    const off2 = 2 * stride + i * K_ext;
                    const off3 = 3 * stride + i * K_ext;
                    return prescaled[off0 + ci[i][4 * pos]].add(prescaled[off1 + ci[i][4 * pos + 1]])
                        .add(prescaled[off2 + ci[i][4 * pos + 2]]).add(prescaled[off3 + ci[i][4 * pos + 3]]);
                },
                else => unreachable,
            }
        }

        fn computePhase2Poly(self: *Self, evals: []F, previous_claim: F) void {
            const half = self.phase2_len / 2;

            if (self.chunk_indices != null) {
                // Lazy evaluation: use chunk_indices + lookup tables
                self.computePhase2PolyLazy(evals, half, previous_claim);
            } else {
                // Dense evaluation: use materialized H arrays (Gruen c/e optimization)
                self.computePhase2PolyDense(evals, half, previous_claim);
            }
        }

        fn computePhase2PolyLazy(self: *Self, evals: []F, half: usize, previous_claim: F) void {
            const ci = self.chunk_indices.?;
            const num_tables = self.lazy_num_tables;
            const prescaled = self.prescaled_lazy_flat.?;

            // Use flat eq table for lazy compute, GruenSplitEq for bind only
            const LazyCtx = struct {
                ci: []const []const u8,
                prescaled: []const F,
                num_tables: u8,
                eq_cycle: []const F,
                gamma_powers: []const F,
                N: usize,
                K_ext: usize,
            };
            const ctx = LazyCtx{
                .ci = ci,
                .prescaled = prescaled,
                .num_tables = num_tables,
                .eq_cycle = self.eq_cycle,
                .gamma_powers = self.gamma_powers,
                .N = self.N,
                .K_ext = self.K_ext,
            };

            const mapFn = struct {
                fn f(c: LazyCtx, start: usize, end: usize) [4]F {
                    const UPA = UnreducedProductAccum;
                    var c_weighted = F.zero();
                    var e_weighted = F.zero();
                    var eq_sum_0 = F.zero();
                    var eq_sum_1 = F.zero();
                    for (start..end) |j| {
                        const d0 = c.eq_cycle[2 * j];
                        const d1 = c.eq_cycle[2 * j + 1];
                        var acc_c = UPA.zero();
                        var acc_e = UPA.zero();
                        for (0..c.N) |i| {
                            // Pre-scaled: h0/h1 already contain gamma^i factor
                            const h0 = lazyGetCoeffPrescaled(c.ci, c.prescaled, c.N, c.K_ext, c.num_tables, i, 2 * j);
                            const h1 = lazyGetCoeffPrescaled(c.ci, c.prescaled, c.N, c.K_ext, c.num_tables, i, 2 * j + 1);
                            const rho = c.gamma_powers[i];
                            const b = h1.sub(h0);
                            acc_c.addAssign(h0.mulToProductAccum(h0.sub(rho)));
                            acc_e.addAssign(b.mulToProductAccum(b));
                        }
                        const q_c = acc_c.reduce();
                        const q_e = acc_e.reduce();
                        c_weighted = c_weighted.add(d0.mul(q_c));
                        e_weighted = e_weighted.add(d0.mul(q_e));
                        eq_sum_0 = eq_sum_0.add(d0);
                        eq_sum_1 = eq_sum_1.add(d1);
                    }
                    return [4]F{ c_weighted, e_weighted, eq_sum_0, eq_sum_1 };
                }
            }.f;

            const reduceFn = struct {
                fn f(a: [4]F, b: [4]F) [4]F {
                    return [4]F{ a[0].add(b[0]), a[1].add(b[1]), a[2].add(b[2]), a[3].add(b[3]) };
                }
            }.f;

            const result = if (self.pool) |pool|
                pool.parallelReduce([4]F, half, [4]F{ F.zero(), F.zero(), F.zero(), F.zero() }, ctx, mapFn, reduceFn)
            else
                mapFn(ctx, 0, half);

            const c_weighted = result[0];
            const e_weighted = result[1];
            const eq_eval_0 = result[2];
            const eq_eval_1 = result[3];
            const adjusted_claim = previous_claim.mul(self.eq_r_r.inverse().?);
            const s0_inner = c_weighted;
            const s1_inner = adjusted_claim.sub(c_weighted);
            const eq0_inv = eq_eval_0.inverse().?;
            const eq1_inv = eq_eval_1.inverse().?;
            const q_total_0 = c_weighted.mul(eq0_inv);
            const q_total_1 = s1_inner.mul(eq1_inv);
            const q_total_e = e_weighted.mul(eq0_inv);
            const e_times_2 = q_total_e.add(q_total_e);
            const q_total_2 = q_total_1.add(q_total_1).sub(q_total_0).add(e_times_2);
            const q_total_3 = q_total_2.add(q_total_1).sub(q_total_0).add(e_times_2.add(e_times_2));
            const eq_slope = eq_eval_1.sub(eq_eval_0);
            const eq_eval_2 = eq_eval_1.add(eq_slope);
            const eq_eval_3 = eq_eval_2.add(eq_slope);
            evals[0] = s0_inner.mul(self.eq_r_r);
            evals[1] = s1_inner.mul(self.eq_r_r);
            evals[2] = eq_eval_2.mul(q_total_2).mul(self.eq_r_r);
            evals[3] = eq_eval_3.mul(q_total_3).mul(self.eq_r_r);
        }

        fn computePhase2PolyDense(self: *Self, evals: []F, half: usize, previous_claim: F) void {
            const ht = self.H orelse return;

            // GruenSplitEq factored eq — E_out * E_in decomposition + computeCubicRoundPoly
            const eq_tables_d = self.gruen_eq_cycle.getWindowEqTables(self.gruen_eq_cycle.current_index, 1);
            const BoolP2Ctx = struct {
                ht: [][]F,
                E_out: []const F,
                E_in: []const F,
                in_mask: usize,
                head_in_bits: usize,
                gamma_powers: []const F,
                N: usize,
            };
            const ctx = BoolP2Ctx{
                .ht = ht,
                .E_out = eq_tables_d.E_out,
                .E_in = eq_tables_d.E_in,
                .in_mask = (@as(usize, 1) << @intCast(eq_tables_d.head_in_bits)) -| 1,
                .head_in_bits = eq_tables_d.head_in_bits,
                .gamma_powers = self.gamma_powers,
                .N = self.N,
            };

            const mapFn = struct {
                fn f(c: BoolP2Ctx, start: usize, end: usize) [2]F {
                    const UPA = UnreducedProductAccum;
                    var q_const_upa = UPA.zero();
                    var q_quad_upa = UPA.zero();
                    for (start..end) |j| {
                        const x_out = j >> @intCast(c.head_in_bits);
                        const x_in = j & c.in_mask;
                        const eq_prefix = (if (x_out < c.E_out.len) c.E_out[x_out] else F.one())
                            .mul(if (x_in < c.E_in.len) c.E_in[x_in] else F.one());
                        var acc_c = UPA.zero();
                        var acc_e = UPA.zero();
                        for (0..c.N) |i| {
                            const h0 = c.ht[i][2 * j];
                            const h1 = c.ht[i][2 * j + 1];
                            const b = h1.sub(h0);
                            const rho = c.gamma_powers[i];
                            acc_c.addAssign(h0.mulToProductAccum(h0.sub(rho)));
                            acc_e.addAssign(b.mulToProductAccum(b));
                        }
                        q_const_upa.addAssign(eq_prefix.mulToProductAccum(acc_c.reduce()));
                        q_quad_upa.addAssign(eq_prefix.mulToProductAccum(acc_e.reduce()));
                    }
                    return .{ q_const_upa.reduce(), q_quad_upa.reduce() };
                }
            }.f;

            const reduceFn = struct {
                fn f(a: [2]F, b: [2]F) [2]F {
                    return .{ a[0].add(b[0]), a[1].add(b[1]) };
                }
            }.f;

            const result = if (self.pool) |pool|
                pool.parallelReduce([2]F, half, [2]F{ F.zero(), F.zero() }, ctx, mapFn, reduceFn)
            else
                mapFn(ctx, 0, half);

            const adjusted_claim = previous_claim.mul(self.eq_r_r.inverse().?);
            const gruen_evals = self.gruen_eq_cycle.computeCubicRoundPoly(result[0], result[1], adjusted_claim);
            evals[0] = gruen_evals[0].mul(self.eq_r_r);
            evals[1] = gruen_evals[1].mul(self.eq_r_r);
            evals[2] = gruen_evals[2].mul(self.eq_r_r);
            evals[3] = gruen_evals[3].mul(self.eq_r_r);
        }

        pub fn bindChallenge(self: *Self, r: F) !void {
            if (self.round < self.log_k_chunk) {
                // Phase 1: update B_scalar and F_table
                const w_m = self.r_address_le[self.round];
                // B_scalar *= eq(w_m, r) = w_m*r + (1-w_m)*(1-r) = 1 - w_m - r + 2*w_m*r
                const prod = w_m.mul(r);
                self.B_scalar = self.B_scalar.mul(F.one().sub(w_m).sub(r).add(prod.add(prod)));

                // Update F: double size from 2^m to 2^(m+1)
                // Match Jolt's LowToHigh ExpandingTable: new entries go in the UPPER half.
                // This ensures bit j of the F index corresponds to sumcheck challenge r_j.
                // Jolt: F[i+len] = F[i]*r, F[i] = F[i]*(1-r)  (for i in 0..len)
                for (0..self.F_size) |idx| {
                    self.F_table[idx + self.F_size] = self.F_table[idx].mul(r);
                    self.F_table[idx] = self.F_table[idx].sub(self.F_table[idx + self.F_size]);
                }
                self.F_size *= 2;

                // Phase 1→2 transition after last address round
                if (self.round == self.log_k_chunk - 1) {
                    self.eq_r_r = self.B_scalar; // eq(r_addr_fixed, r_addr_bound)
                    try self.transitionToPhase2();
                }
            } else {
                // Phase 2: bind cycle variable
                const half = self.phase2_len / 2;

                const bindOne = struct {
                    fn f(arr: []F, h: usize, challenge: F) void {
                        for (0..h) |j| {
                            arr[j] = arr[2 * j].add(challenge.mul(arr[2 * j + 1].sub(arr[2 * j])));
                        }
                    }
                }.f;

                if (self.chunk_indices != null) {
                    // Lazy state: split tables, don't bind dense arrays
                    const K = self.K;

                    if (self.lazy_num_tables == 1) {
                        // Round1 → Round2: split into tables_0 = (1-r)*table, tables_1 = r*table
                        const old_table = self.lazy_tables[0];
                        const tbl_len = K + 1; // includes sentinel entry
                        const t0 = try self.allocator.alloc(F, tbl_len);
                        const t1 = try self.allocator.alloc(F, tbl_len);
                        for (0..K) |k| {
                            t1[k] = r.mul(old_table[k]);
                            t0[k] = old_table[k].sub(t1[k]);
                        }
                        t0[K] = F.zero(); // sentinel stays zero
                        t1[K] = F.zero();
                        self.allocator.free(old_table);
                        self.lazy_tables[0] = t0;
                        self.lazy_tables[1] = t1;
                        self.lazy_num_tables = 2;

                        // Update prescaled flat array for tables 0 and 1
                        if (self.prescaled_lazy_flat) |pf| {
                            const K_ext = self.K_ext;
                            const N = self.N;
                            const stride = N * K_ext;
                            for (0..N) |i| {
                                const gp = self.gamma_powers[i];
                                const row0 = 0 * stride + i * K_ext;
                                const row1 = 1 * stride + i * K_ext;
                                for (0..K_ext) |k| {
                                    pf[row0 + k] = gp.mul(t0[k]);
                                    pf[row1 + k] = gp.mul(t1[k]);
                                }
                            }
                        }
                    } else if (self.lazy_num_tables == 2) {
                        // Round2 → Round3: split each of 2 tables into 2
                        // After Round1→Round2: tables[0]=(1-r0)*F, tables[1]=r0*F
                        // Binding with r1, the position offset within a group of 4 encodes:
                        //   bit 0 = r0 selector, bit 1 = r1 selector
                        // So tables must be ordered by [bit1, bit0] matching position offsets:
                        //   [0]=pos0=(1-r1)(1-r0), [1]=pos1=(1-r1)*r0,
                        //   [2]=pos2=r1*(1-r0),    [3]=pos3=r1*r0
                        const old_t0 = self.lazy_tables[0]; // (1-r0)*F
                        const old_t1 = self.lazy_tables[1]; // r0*F
                        const tbl_len = K + 1;
                        const t_pos0 = try self.allocator.alloc(F, tbl_len); // (1-r1)(1-r0)
                        const t_pos1 = try self.allocator.alloc(F, tbl_len); // (1-r1)*r0
                        const t_pos2 = try self.allocator.alloc(F, tbl_len); // r1*(1-r0)
                        const t_pos3 = try self.allocator.alloc(F, tbl_len); // r1*r0
                        for (0..K) |k| {
                            t_pos2[k] = r.mul(old_t0[k]); // r1*(1-r0)*F
                            t_pos3[k] = r.mul(old_t1[k]); // r1*r0*F
                            t_pos0[k] = old_t0[k].sub(t_pos2[k]); // (1-r1)(1-r0)*F
                            t_pos1[k] = old_t1[k].sub(t_pos3[k]); // (1-r1)*r0*F
                        }
                        t_pos0[K] = F.zero();
                        t_pos1[K] = F.zero();
                        t_pos2[K] = F.zero();
                        t_pos3[K] = F.zero();
                        self.allocator.free(old_t0);
                        self.allocator.free(old_t1);
                        self.lazy_tables[0] = t_pos0;
                        self.lazy_tables[1] = t_pos1;
                        self.lazy_tables[2] = t_pos2;
                        self.lazy_tables[3] = t_pos3;
                        self.lazy_num_tables = 4;

                        // Update prescaled flat array for tables 0,1,2,3
                        if (self.prescaled_lazy_flat) |pf| {
                            const K_ext = self.K_ext;
                            const N = self.N;
                            const stride = N * K_ext;
                            const tbls = [4][]const F{ t_pos0, t_pos1, t_pos2, t_pos3 };
                            for (0..N) |i| {
                                const gp = self.gamma_powers[i];
                                inline for (0..4) |t| {
                                    const row = t * stride + i * K_ext;
                                    for (0..K_ext) |k| {
                                        pf[row + k] = gp.mul(tbls[t][k]);
                                    }
                                }
                            }
                        }
                    } else {
                        // Round3 → Dense: materialize H[N][T/8] and free indices/tables.
                        // materializeDense frees eq_cycle (no longer needed in dense mode).
                        try self.materializeDense(r);
                        // Only bind gruen_eq_cycle (O(1)); eq_cycle was freed above.
                        self.gruen_eq_cycle.bind(r);
                        self.phase2_len = half;
                        self.round += 1;
                        return;
                    }

                    // Bind eq_cycle + GruenSplitEq (lazy state, eq_cycle still alive)
                    bindOne(self.eq_cycle, half, r);
                    self.gruen_eq_cycle.bind(r);
                } else if (self.H) |ht| {
                    // Dense state: bind H arrays only (eq_cycle is NOT needed in dense
                    // compute — computePhase2PolyDense uses gruen_eq_cycle exclusively).
                    if (self.gpu) |gpu| {
                        if (half >= 16384) {
                            for (0..self.N) |i| {
                                gpu.polyBindLow(ht[i][0 .. half * 2], r, ht[i][0..half]) catch bindOne(ht[i], half, r);
                            }
                        } else {
                            for (0..self.N) |i| {
                                bindOne(ht[i], half, r);
                            }
                        }
                    } else if (self.pool) |pool| {
                        const Ctx2 = struct { ht: [][]F, half: usize, challenge: F };
                        const ctx2 = Ctx2{ .ht = ht, .half = half, .challenge = r };
                        pool.parallelForForce(self.N, ctx2, struct {
                            fn f2(c: Ctx2, idx: usize) void {
                                bindOne(c.ht[idx], c.half, c.challenge);
                            }
                        }.f2);
                    } else {
                        for (0..self.N) |i| {
                            bindOne(ht[i], half, r);
                        }
                    }
                    self.gruen_eq_cycle.bind(r);
                } else {
                    // Fallback: no H arrays, no chunk indices (shouldn't happen in normal flow)
                    if (self.eq_cycle.len > 0) bindOne(self.eq_cycle, half, r);
                    self.gruen_eq_cycle.bind(r);
                }
                self.phase2_len = half;
            }
            self.round += 1;
        }

        /// Materialize dense H[N][dense_len] from Round3 (4 tables) + chunk_indices,
        /// binding with challenge r in the process. After this, chunk_indices and
        /// lazy_tables are freed, and self.H is set.
        fn materializeDense(self: *Self, r: F) !void {
            const ci = self.chunk_indices.?;
            const K = self.K;
            const T_orig = ci[0].len;
            // After 3 lazy rounds, the "current length" is T/4 (phase2_len was already
            // halved 2 times; this is the 3rd bind). The dense materialization performs
            // the bind as part of the materialization, producing T/8 entries.
            const dense_len = T_orig / 8;

            // Build 8 combined tables for the 8 original positions within each group.
            // Position offset g within group of 8 has bits [b2, b1, b0]:
            //   b0 = r0 selector, b1 = r1 selector, b2 = r2 selector (current bind)
            // lazy_tables[0..4] are already ordered by position offset (matching bit pattern):
            //   [0]=(1-r1)(1-r0), [1]=(1-r1)*r0, [2]=r1*(1-r0), [3]=r1*r0
            // The current bind with r2 adds the b2 dimension:
            //   combined[g] = ((g & 4) ? r2 : (1-r2)) * lazy_tables[g & 3]
            const tbl_len = K + 1; // includes sentinel
            var combined_tables: [8][]F = undefined;
            for (0..8) |g| {
                combined_tables[g] = try self.allocator.alloc(F, tbl_len);
            }
            errdefer for (combined_tables) |ct| self.allocator.free(ct);

            for (0..K) |k| {
                inline for (0..4) |g| {
                    combined_tables[g + 4][k] = r.mul(self.lazy_tables[g][k]);
                    combined_tables[g][k] = self.lazy_tables[g][k].sub(combined_tables[g + 4][k]);
                }
            }
            // Sentinel entries stay zero
            inline for (0..8) |g| {
                combined_tables[g][K] = F.zero();
            }

            // Materialize dense H arrays, pre-scaled by γ^i for Gruen optimization.
            // Pre-allocate all arrays first (allocator calls can't be parallelized),
            // then fill in parallel across N polynomials.
            var ht = try self.allocator.alloc([]F, self.N);
            for (0..self.N) |i| {
                ht[i] = try self.allocator.alloc(F, dense_len);
            }

            // Pre-compute per-poly γ-scaled combined tables in a flat array:
            //   scaled[i * stride8 + g * tbl_len + k] = γ^i * combined_tables[g][k]
            // This eliminates the per-element multiply by γ^i in the inner loop
            // (matches Jolt's approach of pre-baking gamma into the eq tables).
            // Cost: N * 8 * K muls ≈ 5000 muls — negligible vs 2.5M inner-loop adds.
            const stride8 = 8 * tbl_len;
            const scaled = try self.allocator.alloc(F, self.N * stride8);
            defer self.allocator.free(scaled);
            for (0..self.N) |i| {
                const rho = self.gamma_powers[i];
                const base_off = i * stride8;
                for (0..8) |g| {
                    const g_off = base_off + g * tbl_len;
                    for (0..tbl_len) |k| {
                        scaled[g_off + k] = rho.mul(combined_tables[g][k]);
                    }
                }
            }

            // Parallel fill: each task materializes one poly's H array
            if (self.pool) |pool| {
                const MdCtx = struct {
                    ht: [][]F,
                    ci: []const []const u8,
                    scaled: []const F,
                    stride8: usize,
                    tbl_len: usize,
                    dense_len: usize,
                };
                const md_ctx = MdCtx{
                    .ht = ht,
                    .ci = ci,
                    .scaled = scaled,
                    .stride8 = stride8,
                    .tbl_len = tbl_len,
                    .dense_len = dense_len,
                };
                pool.parallelForForce(self.N, md_ctx, struct {
                    fn f(c: MdCtx, i: usize) void {
                        const idx = c.ci[i];
                        const my_off = i * c.stride8;
                        for (0..c.dense_len) |j| {
                            const base = j * 8;
                            var val = F.zero();
                            inline for (0..8) |g| {
                                val = val.add(c.scaled[my_off + g * c.tbl_len + idx[base + g]]);
                            }
                            c.ht[i][j] = val; // no per-element gamma multiply
                        }
                    }
                }.f);
            } else {
                // Sequential fallback
                for (0..self.N) |i| {
                    const idx = ci[i];
                    const my_off = i * stride8;
                    for (0..dense_len) |j| {
                        const base = j * 8;
                        var val = F.zero();
                        inline for (0..8) |g| {
                            val = val.add(scaled[my_off + g * tbl_len + idx[base + g]]);
                        }
                        ht[i][j] = val;
                    }
                }
            }

            // Free chunk indices: the flat backing storage + the slice-of-slices header.
            if (self.ci_flat_storage) |flat| {
                dropInBackground(self.allocator, flat);
                self.ci_flat_storage = null;
            }
            self.allocator.free(ci);
            self.chunk_indices = null;
            // Free lazy tables and prescaled flat array (small, sync is fine)
            for (0..4) |i| {
                self.allocator.free(self.lazy_tables[i]);
                self.lazy_tables[i] = &.{};
            }
            self.lazy_num_tables = 0;
            if (self.prescaled_lazy_flat) |pf| {
                dropInBackground(self.allocator, pf);
                self.prescaled_lazy_flat = null;
            }
            for (combined_tables) |ct| self.allocator.free(ct);

            self.H = ht;

            // Free eq_cycle — no longer needed in dense mode.
            // computePhase2PolyDense uses gruen_eq_cycle exclusively.
            // The final eq value is available via gruen_eq_cycle.current_scalar.
            self.allocator.free(self.eq_cycle);
            self.eq_cycle = &[_]F{};
        }

        fn transitionToPhase2(self: *Self) !void {
            // F_table now has K entries: F[k] = eq(r_challenges, k) for k ∈ [0, K)
            // Instead of materializing full H[N][T] dense arrays (76MB for N=38, T=65536),
            // store u8 chunk indices (2.5MB) and look up F_table values lazily.
            // This reduces working set from 76MB to ~2.5MB, fitting in L2 cache.
            // After 3 Phase 2 rounds, materialize dense arrays of size T/8 (9.4MB).
            const T_val = @as(usize, 1) << @intCast(self.n_cycle_vars);
            const trace = self.trace;
            const instr_d = self.instruction_d;
            const bc_d = self.bytecode_d;
            const ram_d_val = self.ram_d;
            const K = self.K;

            // Allocate chunk index arrays: N arrays of T u8 entries.
            // Use K as sentinel for "no value" (F.zero()), with tables extended by 1 entry.
            // Single flat allocation reduces mmap overhead vs N separate allocs.
            std.debug.assert(K <= 255); // K+1 must fit in u8
            const sentinel: u8 = @intCast(K);
            const ci_flat = try self.allocator.alloc(u8, self.N * T_val);
            @memset(ci_flat, sentinel);
            var ci = try self.allocator.alloc([]u8, self.N);
            errdefer {
                self.allocator.free(ci_flat);
                self.allocator.free(ci);
            }
            for (0..self.N) |i| {
                ci[i] = ci_flat[i * T_val .. (i + 1) * T_val];
            }
            self.ci_flat_storage = ci_flat; // save for bulk free in materializeDense/deinit

            // Parallel chunk_indices build — each j writes to unique ci[*][j] positions
            const CiCtx = struct {
                steps: []const tracer.TraceStep,
                ci_arr: [][]u8,
                pc_map_ptr: *const BytecodePCMapper,
                mem_layout: *const jolt_device.MemoryLayout,
                log_kc: usize,
                instr_d_v: usize,
                bc_d_v: usize,
                ram_d_v: usize,
                K_v: usize,
                sentinel_v: u8,
            };
            const ci_ctx = CiCtx{
                .steps = trace.steps.items,
                .ci_arr = ci,
                .pc_map_ptr = self.pc_map,
                .mem_layout = self.memory_layout,
                .log_kc = self.log_k_chunk,
                .instr_d_v = instr_d,
                .bc_d_v = bc_d,
                .ram_d_v = ram_d_val,
                .K_v = K,
                .sentinel_v = sentinel,
            };
            const ciFn = struct {
                fn f(c: CiCtx, j: usize) void {
                    const step = c.steps[j];
                    // InstructionRa chunks
                    {
                        const lookup_idx = computeLookupIndex(step);
                        for (0..c.instr_d_v) |i| {
                            const shift = c.log_kc * (c.instr_d_v - 1 - i);
                            const mask: u128 = (@as(u128, 1) << @intCast(c.log_kc)) - 1;
                            const chunk_val: usize = @intCast((lookup_idx >> @intCast(shift)) & mask);
                            c.ci_arr[i][j] = if (chunk_val < c.K_v) @intCast(chunk_val) else c.sentinel_v;
                        }
                    }
                    // BytecodeRa chunks
                    {
                        const pc_idx: u64 = @intCast(c.pc_map_ptr.getPCForStep(step));
                        for (0..c.bc_d_v) |i| {
                            const chunk_val = extractChunkMSB(pc_idx, i, c.bc_d_v, c.log_kc);
                            c.ci_arr[c.instr_d_v + i][j] = if (chunk_val < c.K_v) @intCast(chunk_val) else c.sentinel_v;
                        }
                    }
                    // RamRa chunks
                    {
                        if (step.memory_addr) |addr| {
                            if (addr != 0) {
                                if (c.mem_layout.remapAddress(addr)) |raddr| {
                                    for (0..c.ram_d_v) |i| {
                                        const chunk_val = extractChunkMSB(raddr, i, c.ram_d_v, c.log_kc);
                                        c.ci_arr[c.instr_d_v + c.bc_d_v + i][j] = if (chunk_val < c.K_v) @intCast(chunk_val) else c.sentinel_v;
                                    }
                                }
                            }
                        }
                    }
                }
            }.f;
            if (self.pool) |pool| {
                pool.parallelFor(T_val, ci_ctx, ciFn);
            } else {
                for (0..T_val) |j| ciFn(ci_ctx, j);
            }

            self.chunk_indices = ci;
            // Copy F_table as the initial lazy lookup table (round1 state)
            // Extra entry at index K = F.zero() for sentinel "no value" positions
            const lt = try self.allocator.alloc(F, K + 1);
            @memcpy(lt[0..K], self.F_table[0..K]);
            lt[K] = F.zero();
            self.lazy_tables[0] = lt;
            self.lazy_num_tables = 1;

            // Pre-scale table 0 for all N polys: prescaled[i*K_ext + k] = gamma^i * lt[k]
            {
                const K_ext = self.K_ext;
                const N = self.N;
                const pf = try self.allocator.alloc(F, 4 * N * K_ext);
                @memset(pf, F.zero());
                // Only table 0 is active initially
                const base0: usize = 0; // table 0 offset = 0 * N * K_ext
                for (0..N) |i| {
                    const row_off = base0 + i * K_ext;
                    const gp = self.gamma_powers[i];
                    for (0..K) |k| {
                        pf[row_off + k] = gp.mul(lt[k]);
                    }
                    // sentinel at K stays zero (already memset)
                }
                self.prescaled_lazy_flat = pf;
            }

            self.H = null;
            self.phase2_len = T_val;

            // Free G tables in background — no longer needed after Phase 1.
            // G tables hold N arrays of K field elements (e.g. 38 × 16 = 608 entries).
            // Ownership transfers to background thread; null out to prevent double-free in deinit.
            dropInBackground(self.allocator, self.G);
            self.G = &.{};

            // Debug: print F_table values at transition
            if (comptime debug_verbose) {
                dbg("[BOOL_H_INIT] T={}, using lazy chunk indices\n", .{T_val});
                dbg("[BOOL_H_INIT] F_size={}\n", .{self.F_size});
                for (0..@min(self.F_size, 8)) |fi| {
                    const fb = self.F_table[fi].toBytesBE();
                    dbg("[BOOL_H_INIT] F[{}]_LE=[{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}]\n", .{
                        fi, fb[31], fb[30], fb[29], fb[28], fb[27], fb[26], fb[25], fb[24],
                    });
                }
            }

            dbg("[BOOL_PROVER] Phase 1→2 transition: eq_r_r=", .{});
            const err_be = self.eq_r_r.toBytesBE();
            for (0..8) |bi| dbg("{x:0>2}", .{err_be[31 - bi]});
            dbg(", lazy_num_tables={}\n", .{self.lazy_num_tables});
        }
    };
}

// =============================================================================
// LookupsRaVirtual Sumcheck Instance (Instance 4)
// =============================================================================
// Proves: Sigma_c eq(r_cycle, c) * Sum_{v=0}^{N-1} gamma^v * Prod_{j=0}^{M-1} ra_{v*M+j}(c)
// Variables: n_cycle_vars
// Degree: M+1 (product of M linear ra polys * one linear eq)
//
// NOTE: Unlike RamRaVirtualProver, this uses dense []F arrays rather than RaPolynomial
// compression. The gamma scale is baked into the first poly of each virtual batch at
// init time (ra_bound[v*M] *= gamma^v), so the compressed representation would need
// per-index scaling, not a single shared eq_table scale. Future optimization could
// store separate gamma-scaled and unscaled eq_tables per batch.
/// Evaluate the product of 4 linear polynomials at the grid {1, 2, 3, ∞}.
/// Each pair[i] = { p_i(0), p_i(1) }. Returns { P(1), P(2), P(3), P(∞) }
/// where P(x) = p_0(x) * p_1(x) * p_2(x) * p_3(x).
/// Uses Toom-Cook factoring: 10 field muls total.
/// Ported from Jolt's eval_prod_4_assign (mles_product_sum.rs:453-462).
fn evalLinearProd4(comptime F: type, pairs: [4][2]F) [4]F {
    // eval_linear_prod_2_internal on first pair (p[0], p[1]):
    // For linear poly p(x) = p0 + (p1-p0)*x: p(1)=p1, p(∞)=p1-p0, p(2)=p1+(p1-p0)
    const p0_inf = pairs[0][1].sub(pairs[0][0]); // slope of p0
    const p1_inf = pairs[1][1].sub(pairs[1][0]); // slope of p1
    const a1 = pairs[0][1].mul(pairs[1][1]); // A(1) = p0(1)*p1(1)
    const a2 = p0_inf.add(pairs[0][1]).mul(p1_inf.add(pairs[1][1])); // A(2) = p0(2)*p1(2)
    const a_inf = p0_inf.mul(p1_inf); // A(∞) = leading coeff

    // ex2 extrapolation: A(3) = 2*(A(2) + A(∞)) - A(1)  (0 muls, pure adds)
    const a3 = a2.add(a_inf).add(a2.add(a_inf)).sub(a1);

    // eval_linear_prod_2_internal on second pair (p[2], p[3]):
    const p2_inf = pairs[2][1].sub(pairs[2][0]);
    const p3_inf = pairs[3][1].sub(pairs[3][0]);
    const b1 = pairs[2][1].mul(pairs[3][1]);
    const b2 = p2_inf.add(pairs[2][1]).mul(p3_inf.add(pairs[3][1]));
    const b_inf = p2_inf.mul(p3_inf);
    const b3 = b2.add(b_inf).add(b2.add(b_inf)).sub(b1);

    // Pointwise multiply: 4 muls
    return .{
        a1.mul(b1), // P(1)
        a2.mul(b2), // P(2)
        a3.mul(b3), // P(3)
        a_inf.mul(b_inf), // P(∞)
    };
}

pub fn LookupsRaVirtualProver(comptime F: type) type {
    const RaPoly = ra_poly_mod.RaPolynomial(F);

    return struct {
        const Self = @This();

        /// In-place MLE bind (same as RamRaVirtualProver.bindSlice).
        fn bindSlice(arr: []F, h: usize, challenge: F) void {
            if (challenge.limbs[0] == 0 and challenge.limbs[1] == 0) {
                for (0..h) |j| {
                    arr[j] = arr[2 * j].add(arr[2 * j + 1].sub(arr[2 * j]).mulHiBigIntU128(challenge.limbs));
                }
            } else {
                for (0..h) |j| {
                    arr[j] = arr[2 * j].add(challenge.mul(arr[2 * j + 1].sub(arr[2 * j])));
                }
            }
        }

        /// Compressed RA polynomials (lazy materialization through round1→round2→round3→dense)
        ra_polys: []RaPoly,
        /// GruenSplitEq for eq(r_cycle, .) — O(1) bind
        gruen_eq: poly_mod.GruenSplitEqPolynomial(F),
        M: usize,
        N: usize,
        total_committed: usize,
        current_len: usize,
        allocator: Allocator,
        pool: ?*ThreadPool = null,
        gpu: ?*GpuPolyOps = null,

        pub fn init(
            allocator: Allocator,
            trace: *const ExecutionTrace,
            r_cycle: []const F, // BIG_ENDIAN
            r_addr_chunks: []const []const F, // r_addr_chunks[i] for each committed poly
            gamma_powers: []const F, // gamma^v for v in 0..N
            M: usize,
            N: usize,
            log_k_chunk: usize,
            instruction_d: usize,
            init_pool: ?*ThreadPool,
        ) !Self {
            std.debug.assert(log_k_chunk <= ra_poly_mod.MAX_LOG_K_CHUNK);
            const T = trace.steps.items.len;
            const total_committed = M * N;
            const k_chunk: usize = @as(usize, 1) << @intCast(log_k_chunk);

            // Build RaPolynomials with compressed u8 indices + small eq tables
            var ra_polys_arr = try allocator.alloc(RaPoly, total_committed);
            errdefer allocator.free(ra_polys_arr);

            // Pre-allocate index arrays for all committed polys
            var indices_arr = try allocator.alloc([]?u8, total_committed);
            defer allocator.free(indices_arr);

            for (0..total_committed) |i| {
                indices_arr[i] = try allocator.alloc(?u8, T);
            }

            // Parallel fill: compute ALL index arrays in a single pass over trace.
            // For each step j, compute lookup_index ONCE, then extract all total_committed chunks.
            // This is ~32x fewer computeLookupIndex calls vs the old per-poly approach.
            const LkRaInitCtx = struct {
                steps: []const tracer.TraceStep,
                indices: [][]?u8,
                log_k_chunk: usize,
                k_chunk: usize,
                instruction_d: usize,
                total_committed: usize,
            };
            const lk_ra_ctx = LkRaInitCtx{
                .steps = trace.steps.items,
                .indices = indices_arr,
                .log_k_chunk = log_k_chunk,
                .k_chunk = k_chunk,
                .instruction_d = instruction_d,
                .total_committed = total_committed,
            };
            const lkRaInitFn = struct {
                fn f(c: LkRaInitCtx, j: usize) void {
                    const step = c.steps[j];
                    const lookup_index = computeLookupIndex(step);
                    const mask: u128 = (@as(u128, 1) << @intCast(c.log_k_chunk)) - 1;
                    for (0..c.total_committed) |i| {
                        // MSB-first: shift = log_k_chunk * (instruction_d - 1 - i)
                        const shift_amount = c.log_k_chunk * (c.instruction_d - 1 - i);
                        const chunk_val: usize = if (shift_amount < 128) @intCast((lookup_index >> @intCast(shift_amount)) & mask) else 0;
                        c.indices[i][j] = if (chunk_val < c.k_chunk) @intCast(chunk_val) else null;
                    }
                }
            }.f;
            pool_helpers.parallelForOptional(init_pool, T, lk_ra_ctx, lkRaInitFn);

            // Build eq tables and create RaPolynomials
            for (0..total_committed) |i| {
                var r_chunk_rev = try allocator.alloc(F, log_k_chunk);
                defer allocator.free(r_chunk_rev);
                for (0..log_k_chunk) |ci| r_chunk_rev[ci] = r_addr_chunks[i][log_k_chunk - 1 - ci];
                const eq_table = try computeEqTable(F, allocator, r_chunk_rev, log_k_chunk);

                const virtual_batch = i / M;
                const is_first_in_batch = (i % M == 0);
                const gamma_scale = if (is_first_in_batch) gamma_powers[virtual_batch] else F.one();

                // initRound1 takes ownership of indices and eq_table, prescales by gamma_scale
                ra_polys_arr[i] = RaPoly.initRound1(indices_arr[i], eq_table, gamma_scale);
            }

            // r_cycle is in BE order; pass directly to GruenSplitEq (same as Stage 3)
            const n_vars = std.math.log2_int(usize, T);
            const gruen_eq = try poly_mod.GruenSplitEqPolynomial(F).init(allocator, r_cycle[0..n_vars]);

            return Self{
                .ra_polys = ra_polys_arr,
                .gruen_eq = gruen_eq,
                .M = M,
                .N = N,
                .total_committed = total_committed,
                .current_len = T,
                .allocator = allocator,
                .pool = init_pool,
            };
        }

        pub fn deinit(self: *Self) void {
            for (self.ra_polys) |*p| p.deinit(self.allocator);
            self.allocator.free(self.ra_polys);
            self.gruen_eq.deinit();
        }

        /// f(x) = eq(x,r) * Sum_v Prod_{j=0}^{M-1} ra_{v*M+j}(x)
        /// Uses quotient polynomial approach: compute q(x) = f(x)/eq(x,r) at Toom points {1,2,3,∞}
        /// via evalLinearProd4, then reconstruct f(x) via finishMlesProductSumFromEvals.
        /// Returns monomial coefficients (degree M+1, i.e. 6 coefficients for M=4).
        pub fn computeRoundPoly(self: *Self, allocator: Allocator, claim: F) ![]F {
            const half = self.current_len / 2;

            // Get factored eq tables from GruenSplitEq (same pattern as Stage 3)
            const eq_tables = self.gruen_eq.getWindowEqTables(self.gruen_eq.current_index, 1);
            const E_out = eq_tables.E_out;
            const E_in = eq_tables.E_in;
            const head_in_bits = eq_tables.head_in_bits;
            const in_mask = (@as(usize, 1) << @intCast(head_in_bits)) -| 1;

            const Ctx = struct {
                ra_polys: []RaPoly,
                E_out: []const F,
                E_in: []const F,
                in_mask: usize,
                head_in_bits: usize,
                M: usize,
                N: usize,
            };
            const ctx = Ctx{
                .ra_polys = self.ra_polys,
                .E_out = E_out,
                .E_in = E_in,
                .in_mask = in_mask,
                .head_in_bits = head_in_bits,
                .M = self.M,
                .N = self.N,
            };

            // Compute quotient q(x) = Σ_j E_prefix(j) * Σ_v Π_k ra_{v*M+k}(x)
            // at 4 Toom points {1, 2, 3, ∞}
            // Uses deferred E_out pattern: accumulate E_in*val as unreduced within each x_out
            // block, then reduce and scale by E_out once per block (saves 1 mul per j).
            const UPA = UnreducedProductAccum;
            const mapFn = struct {
                fn f(c: Ctx, start: usize, end: usize) [4]F {
                    // Dispatch once on ra_poly state to avoid per-access tag checks.
                    // All ra_polys are in the same state at any given round.
                    return switch (c.ra_polys[0]) {
                        inline else => |_, comptime_tag| fInner(c, start, end, comptime_tag),
                    };
                }

                inline fn fInner(c: Ctx, start: usize, end: usize, comptime tag: anytype) [4]F {
                    var outer_acc: [4]UPA = .{UPA.zero()} ** 4;
                    var inner_acc: [4]UPA = .{UPA.zero()} ** 4;
                    var prev_x_out: usize = if (start > 0) start >> @intCast(c.head_in_bits) else 0;
                    var started = false;

                    for (start..end) |j| {
                        const x_out = j >> @intCast(c.head_in_bits);
                        const x_in = j & c.in_mask;

                        // Flush inner_acc when x_out changes
                        if (started and x_out != prev_x_out) {
                            const e_out = if (prev_x_out < c.E_out.len) c.E_out[prev_x_out] else F.one();
                            for (0..4) |k| {
                                outer_acc[k].addAssign(e_out.mulToProductAccum(inner_acc[k].reduce()));
                                inner_acc[k] = UPA.zero();
                            }
                        }
                        prev_x_out = x_out;
                        started = true;

                        const e_in = if (x_in < c.E_in.len) c.E_in[x_in] else F.one();

                        // Accumulate sum of products across all virtual batches
                        var virtual_sum = [4]F{ F.zero(), F.zero(), F.zero(), F.zero() };
                        for (0..c.N) |v| {
                            var pairs: [4][2]F = undefined;
                            for (0..c.M) |m_idx| {
                                const idx = v * c.M + m_idx;
                                pairs[m_idx] = .{
                                    @field(c.ra_polys[idx], @tagName(tag)).getBoundCoeff(2 * j),
                                    @field(c.ra_polys[idx], @tagName(tag)).getBoundCoeff(2 * j + 1),
                                };
                            }
                            const prod_evals = evalLinearProd4(F, pairs);
                            for (0..4) |k| virtual_sum[k] = virtual_sum[k].add(prod_evals[k]);
                        }

                        // Accumulate with E_in only (defer E_out to block boundary)
                        for (0..4) |k| {
                            inner_acc[k].addAssign(e_in.mulToProductAccum(virtual_sum[k]));
                        }
                    }
                    // Flush final block
                    if (started) {
                        const e_out = if (prev_x_out < c.E_out.len) c.E_out[prev_x_out] else F.one();
                        for (0..4) |k| {
                            outer_acc[k].addAssign(e_out.mulToProductAccum(inner_acc[k].reduce()));
                        }
                    }
                    return .{ outer_acc[0].reduce(), outer_acc[1].reduce(), outer_acc[2].reduce(), outer_acc[3].reduce() };
                }
            }.f;

            const reduceFn = struct {
                fn f(a: [4]F, b: [4]F) [4]F {
                    return .{ a[0].add(b[0]), a[1].add(b[1]), a[2].add(b[2]), a[3].add(b[3]) };
                }
            }.f;

            const sum_evals = if (self.pool) |pool|
                pool.parallelReduce([4]F, half, .{F.zero()} ** 4, ctx, mapFn, reduceFn)
            else
                mapFn(ctx, 0, half);

            // Scale by current_scalar (accumulated eq from all previously bound variables)
            // The inner loop computed quotient' = Σ E_out*E_in*product (without current_scalar).
            // The actual quotient = current_scalar * quotient'.
            const scalar = self.gruen_eq.current_scalar;
            var scaled_evals = [4]F{
                sum_evals[0].mul(scalar),
                sum_evals[1].mul(scalar),
                sum_evals[2].mul(scalar),
                sum_evals[3].mul(scalar),
            };

            // Extract r_round from gruen_eq and reconstruct full polynomial
            const r_round = self.gruen_eq.tau[self.gruen_eq.current_index - 1];
            return poly_mod.UniPoly(F).finishMlesProductSumFromEvals(allocator, &scaled_evals, claim, r_round);
        }

        pub fn bindChallenge(self: *Self, r: F) !void {
            const half = self.current_len / 2;

            // Bind RA polynomials — O(K) for compressed states, O(T/2^round) for dense
            const all_dense = self.ra_polys[0].isDense();
            if (all_dense and self.gpu != null and half >= 16384) {
                // GPU bind: total_committed ra_poly dense arrays
                const gpu = self.gpu.?;
                for (self.ra_polys) |*rp| {
                    const dense = &rp.dense;
                    const h = dense.current_len / 2;
                    gpu.polyBindLow(dense.coeffs[0 .. h * 2], r, dense.coeffs[0..h]) catch {
                        for (0..h) |jj| {
                            dense.coeffs[jj] = dense.coeffs[2 * jj].add(r.mul(dense.coeffs[2 * jj + 1].sub(dense.coeffs[2 * jj])));
                        }
                    };
                    dense.current_len = h;
                }
            } else if (all_dense) {
                if (self.pool) |pool| {
                    const BindCtx = struct { ra: []RaPoly, tc: usize, half: usize, r: F };
                    const ctx = BindCtx{ .ra = self.ra_polys, .tc = self.total_committed, .half = half, .r = r };
                    pool.parallelForForce(self.total_committed, ctx, struct {
                        fn f(c: BindCtx, idx: usize) void {
                            const dense = &c.ra[idx].dense;
                            const h = dense.current_len / 2;
                            for (0..h) |jj| {
                                dense.coeffs[jj] = dense.coeffs[2 * jj].add(c.r.mul(dense.coeffs[2 * jj + 1].sub(dense.coeffs[2 * jj])));
                            }
                            dense.current_len = h;
                        }
                    }.f);
                } else {
                    for (self.ra_polys) |*p| try p.bind(r, self.allocator);
                }
            } else {
                for (self.ra_polys) |*p| try p.bind(r, self.allocator);
            }

            // GruenSplitEq bind — O(1) instead of O(T/2^round)
            self.gruen_eq.bind(r);

            self.current_len = half;
        }

        pub fn getOpeningClaims(self: *const Self, allocator: Allocator, gamma_powers: []const F) ![]F {
            var claims = try allocator.alloc(F, self.total_committed);
            for (0..self.total_committed) |i| {
                var claim = self.ra_polys[i].finalClaim();
                // Undo gamma pre-scaling for first poly in each batch
                const is_first_in_batch = (i % self.M == 0);
                if (is_first_in_batch) {
                    const virtual_batch = i / self.M;
                    claim = claim.mul(gamma_powers[virtual_batch].inverse().?);
                }
                claims[i] = claim;
            }
            return claims;
        }
    };
}

// =============================================================================
// Booleanity Prover Factory (extracted from stage6_prover.generateStage6Proof)
// =============================================================================
// Builds G-tables, eq-tables, and creates a BooleanityProver in a single call.
// This was previously an ~1180-line inline block in generateStage6Proof.

/// Initialize a BooleanityProver by building G-tables, eq-tables, and all
/// supporting data structures from execution trace and Stage 5 challenge data.
///
/// Ownership: `booleanity_gammas_sq` and `booleanity_gammas_unsq` ownership
/// transfers to the returned BooleanityProver (freed by its deinit).
pub fn initBooleanityProver(
    comptime F: type,
    allocator: std.mem.Allocator,
    thread_pool: ?*ThreadPool,
    gpu_ops: ?*GpuPolyOps,
    trace: *const ExecutionTrace,
    stage5_challenges: []const F,
    lookups_ra_r_cycle: []const F,
    booleanity_gammas_sq: []F,
    booleanity_gammas_unsq: []F,
    instruction_d: usize,
    bytecode_d: usize,
    ram_d: usize,
    log_k_chunk: usize,
    n_cycle_vars: usize,
    memory_layout: *const jolt_device.MemoryLayout,
    pc_map: *const BytecodePCMapper,
) !BooleanityProver(F) {
    const total_bool_polys = instruction_d + bytecode_d + ram_d;

    // r_address_bool: last log_k_chunk of Stage 5 address in LE
    // Stage 5 address in BE: stage5_challenges[0..128] (MSB first since HighToLow binding)
    // Reverse to LE: [ch[127], ch[126], ..., ch[0]]
    // Last log_k_chunk: [ch[log_k_chunk-1], ..., ch[0]] = MSB bits in LE
    var r_address_bool_le = try allocator.alloc(F, log_k_chunk);
    // No defer free - BooleanityProver takes ownership of r_address_bool_le
    for (0..log_k_chunk) |i| {
        // In LE, element i corresponds to Stage5 address challenge (LOOKUPS_LOG_K - 1 - (LOOKUPS_LOG_K - log_k_chunk + i))
        // = log_k_chunk - 1 - i
        r_address_bool_le[i] = stage5_challenges[log_k_chunk - 1 - i];
    }

    // r_cycle_bool_le: same as lookups_ra_r_cycle (already LE)
    // lookups_ra_r_cycle[i] = stage5_challenges[LOOKUPS_LOG_K + n_cycle_vars - 1 - i]

    // Build eq_addr table for Phase 1 (LowToHigh binding)
    // computeEqTable expects BE input (MSB-first) for its internal convention.
    // Since r_address_bool_le is LE and we want LowToHigh binding,
    // the eq table should be indexed such that eq_addr[k] = eq(r_addr_le, k)
    // where bit 0 of k is the LSB, bound first.
    // Jolt's LowToHigh EqPolynomial: eq(r, k) = Π_i (r[i]*k_i + (1-r[i])*(1-k_i))
    // where r[0] corresponds to the LSB of k.
    // For computeEqTable: it expects r in "BE" (MSB first), so reverse LE to BE.
    var r_addr_bool_be_for_eq = try allocator.alloc(F, log_k_chunk);
    defer allocator.free(r_addr_bool_be_for_eq);
    for (0..log_k_chunk) |i| {
        r_addr_bool_be_for_eq[i] = r_address_bool_le[log_k_chunk - 1 - i];
    }
    const eq_addr_bool_phase1 = try computeEqTable(F, allocator, r_addr_bool_be_for_eq, log_k_chunk);
    defer allocator.free(eq_addr_bool_phase1); // Only used for debug verification below

    // Build a SINGLE eq_cycle table used for BOTH G construction AND Phase 2 halving.
    //
    // The table ordering must match Jolt's evals_parallel which iterates .rev():
    //   bit 0 of index j -> r_cycle[n-1] (MSB)
    // For our computeEqTable (forward iteration), input[0] must be MSB = lookups[0].
    // So input = lookups_ra_r_cycle directly (BE, MSB first).
    //
    // Using the SAME table for G construction and Phase 2 ensures consistency:
    // Phase 1 reduces address variables with G tables weighted by eq_cycle[j],
    // and Phase 2 halves the same eq_cycle[j] table. The running claim from Phase 1
    // equals the initial Phase 2 polynomial sum, satisfying the transition.
    //
    // After Phase 2 halving with LowToHigh binding, the final eq value equals
    // eq(challenges, r_cycle_BE) = eq(challenges, rev(r_cycle_LE)), matching
    // Jolt's verifier which computes combined_r_cycle = rev(r_cycle_LE).
    // Build GruenSplitEq for Booleanity Phase 2 (O(1) bind)
    // Build flat eq table (LE convention, proven correct for G-tables)
    const eq_cycle_bool_phase2 = try computeEqTableParallel(F, allocator, lookups_ra_r_cycle, n_cycle_vars, thread_pool);
    // Build GruenSplitEq with REVERSED r_cycle so its binding order matches
    // the LE flat table: GruenSplitEq binds tau[n-1] first, which is
    // reversed[n-1] = lookups_ra_r_cycle[0] = challenge MSB = bit 0 in LE.
    var r_cycle_for_gruen = try allocator.alloc(F, n_cycle_vars);
    defer allocator.free(r_cycle_for_gruen);
    for (0..n_cycle_vars) |ri| r_cycle_for_gruen[ri] = lookups_ra_r_cycle[n_cycle_vars - 1 - ri];
    const bool_gruen_eq = try poly_mod.GruenSplitEqPolynomial(F).init(allocator, r_cycle_for_gruen);
    // eq_cycle_bool_phase2 is NOT deferred - shared with BooleanityProver

    // Build G tables: G_i[k] = Sigma_j eq(r_cycle_fixed, j) * [chunk_i(j) == k]
    const T_val: usize = @as(usize, 1) << @intCast(n_cycle_vars);
    const K_val: usize = @as(usize, 1) << @intCast(log_k_chunk);
    var G_tables = try allocator.alloc([]F, total_bool_polys);
    for (0..total_bool_polys) |i| {
        G_tables[i] = try allocator.alloc(F, K_val);
        @memset(G_tables[i], F.zero());
    }

    // OPTIMIZATION: Pre-compute chunk indices for all T steps in ONE parallel pass.
    // This avoids calling computeLookupIndex 38 times per step (once per poly).
    // Each step produces: instruction chunks [0..instr_d], bytecode chunks [0..bc_d], ram chunks [0..ram_d]
    // Stored as u8 per chunk (K < 256).
    const MAX_BOOL_POLYS = 48; // instruction_d(32) + bytecode_d(~3-5) + ram_d(~2-3)
    std.debug.assert(total_bool_polys <= MAX_BOOL_POLYS);

    // Allocate per-step chunk index arrays: chunk_idx[j][poly_i] = chunk value (or K_val for invalid)
    const chunk_idx = try allocator.alloc([MAX_BOOL_POLYS]u8, T_val);
    defer allocator.free(chunk_idx);

    // Phase 1: Single-pass pre-compute all chunk indices (parallel over T)
    {
        const ChunkPreCtx = struct {
            steps: []const tracer.TraceStep,
            pc_map_ptr: *const BytecodePCMapper,
            mem_layout: *const jolt_device.MemoryLayout,
            instr_d: usize,
            bc_d: usize,
            rm_d: usize,
            lkc: usize,
            K: usize,
            total_polys: usize,
            chunk_idx: [][MAX_BOOL_POLYS]u8,
        };
        const pre_ctx = ChunkPreCtx{
            .steps = trace.steps.items,
            .pc_map_ptr = pc_map,
            .mem_layout = memory_layout,
            .instr_d = instruction_d,
            .bc_d = bytecode_d,
            .rm_d = ram_d,
            .lkc = log_k_chunk,
            .K = K_val,
            .total_polys = total_bool_polys,
            .chunk_idx = chunk_idx,
        };
        const precomputeFn = struct {
            fn f(c: ChunkPreCtx, j: usize) void {
                const step = c.steps[j];
                const sentinel: u8 = @intCast(c.K); // K < 256, use K as "invalid" sentinel

                // InstructionRa: compute lookup_idx ONCE, extract all chunks
                const lookup_idx = computeLookupIndex(step);
                const mask: u128 = (@as(u128, 1) << @intCast(c.lkc)) - 1;
                for (0..c.instr_d) |i| {
                    const shift = c.lkc * (c.instr_d - 1 - i);
                    const chunk_val: usize = @intCast((lookup_idx >> @intCast(shift)) & mask);
                    c.chunk_idx[j][i] = if (chunk_val < c.K) @intCast(chunk_val) else sentinel;
                }

                // BytecodeRa: compute PC ONCE, extract all chunks
                const pc_idx: u64 = @intCast(c.pc_map_ptr.getPCForStep(step));
                for (0..c.bc_d) |i| {
                    const chunk_val = extractChunkMSB(pc_idx, i, c.bc_d, c.lkc);
                    c.chunk_idx[j][c.instr_d + i] = if (chunk_val < c.K) @intCast(chunk_val) else sentinel;
                }

                // RamRa: compute address ONCE, extract all chunks
                if (step.memory_addr) |addr| {
                    if (addr != 0) {
                        if (c.mem_layout.remapAddress(addr)) |raddr| {
                            for (0..c.rm_d) |i| {
                                const chunk_val = extractChunkMSB(raddr, i, c.rm_d, c.lkc);
                                c.chunk_idx[j][c.instr_d + c.bc_d + i] = if (chunk_val < c.K) @intCast(chunk_val) else sentinel;
                            }
                        } else {
                            for (0..c.rm_d) |i| c.chunk_idx[j][c.instr_d + c.bc_d + i] = sentinel;
                        }
                    } else {
                        for (0..c.rm_d) |i| c.chunk_idx[j][c.instr_d + c.bc_d + i] = sentinel;
                    }
                } else {
                    for (0..c.rm_d) |i| c.chunk_idx[j][c.instr_d + c.bc_d + i] = sentinel;
                }
            }
        }.f;
        if (thread_pool) |pool| {
            pool.parallelForForce(T_val, pre_ctx, precomputeFn);
        } else {
            for (0..T_val) |j| precomputeFn(pre_ctx, j);
        }
    }

    // Phase 2: Build G tables using pre-computed indices (parallel over polys)
    // Each poly's inner loop is now a simple scatter-add with O(1) index lookup.
    if (thread_pool) |pool| {
        const GBuildCtx = struct {
            eq_cycle: []const F,
            chunk_idx: [][MAX_BOOL_POLYS]u8,
            K: usize,
            T: usize,
            G_out: [][]F,
        };
        const g_ctx = GBuildCtx{
            .eq_cycle = eq_cycle_bool_phase2,
            .chunk_idx = chunk_idx,
            .K = K_val,
            .T = T_val,
            .G_out = G_tables,
        };
        pool.parallelForForce(total_bool_polys, g_ctx, struct {
            fn f(c: GBuildCtx, poly_i: usize) void {
                const G_i = c.G_out[poly_i];
                const sentinel: u8 = @intCast(c.K);
                for (0..c.T) |j| {
                    const cv = c.chunk_idx[j][poly_i];
                    if (cv != sentinel) {
                        const eq_j = c.eq_cycle[j];
                        G_i[cv] = G_i[cv].add(eq_j);
                    }
                }
            }
        }.f);
    } else {
        // Sequential: single pass over T, scatter to all polys per step
        for (0..T_val) |j| {
            const eq_j = eq_cycle_bool_phase2[j];
            if (eq_j.eql(F.zero())) continue;
            const sentinel: u8 = @intCast(K_val);
            for (0..total_bool_polys) |i| {
                const cv = chunk_idx[j][i];
                if (cv != sentinel) {
                    G_tables[i][cv] = G_tables[i][cv].add(eq_j);
                }
            }
        }
    }

    // Use the independently sampled gammas directly (matching Jolt's challenge_vector_optimized)
    // Jolt formula: Sigma_i gamma_i * (ra_i^2 - ra_i), where gamma_i are independent challenges
    // booleanity_gammas ownership transfers to BooleanityProver (freed by deinit)
    const gamma_sq = booleanity_gammas_sq;

    // Verify G tables: Sigma_k G_i[k] should equal Sigma_j eq(r_cycle, j) = 1
    // Actually Sigma_k G_i[k] = Sigma_j eq(r_cycle, j) * Sigma_k [chunk_i(j)==k]
    //                     = Sigma_j eq(r_cycle, j) * 1 = 1 (since chunk_i(j) always hits exactly one k)
    // Wait no: only if all j have valid chunks. Noop steps may have chunk_val=0 added.
    // Let's just print the first few G tables for debug.
    dbg("[BOOL_PROVER] G tables built: N={}, K={}, T={}\n", .{ total_bool_polys, K_val, T_val });
    for (0..@min(3, total_bool_polys)) |i| {
        var g_sum = F.zero();
        for (0..K_val) |k| g_sum = g_sum.add(G_tables[i][k]);
        const gs_be = g_sum.toBytesBE();
        dbg("  G[{}] sum_LE=[{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}]\n", .{
            i, gs_be[31], gs_be[30], gs_be[29], gs_be[28], gs_be[27], gs_be[26], gs_be[25], gs_be[24],
        });
    }

    // Initial claim verification: Sigma_k eq_addr[k] * Sigma_i gamma^{2i} * (G_i[k]^2 - G_i[k])
    // This should be zero since ra_i(k,j) is binary.
    // Actually that's the FULL sum; at random r it won't be zero for individual terms.
    // But the initial claim IS zero.
    {
        var init_sum = F.zero();
        for (0..K_val) |k| {
            var q_val = F.zero();
            for (0..total_bool_polys) |i| {
                const g_k = G_tables[i][k];
                q_val = q_val.add(gamma_sq[i].mul(g_k.mul(g_k).sub(g_k)));
            }
            init_sum = init_sum.add(eq_addr_bool_phase1[k].mul(q_val));
        }
        const is_be = init_sum.toBytesBE();
        dbg("[BOOL_PROVER] Initial sum (should be ~0) LE=[{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}]\n", .{
            is_be[31], is_be[30], is_be[29], is_be[28], is_be[27], is_be[26], is_be[25], is_be[24],
        });
    }

    var prover = try BooleanityProver(F).init(
        allocator,
        G_tables,
        r_address_bool_le,
        bool_gruen_eq,
        eq_cycle_bool_phase2,
        gamma_sq,
        booleanity_gammas_unsq,
        total_bool_polys,
        log_k_chunk,
        n_cycle_vars,
        trace,
        instruction_d,
        bytecode_d,
        ram_d,
        memory_layout,
        pc_map,
    );
    prover.pool = thread_pool;
    prover.gpu = gpu_ops;
    return prover;
}
