//! BytecodeReadRaf Sumcheck Instance Prover (Instance 0 of Stage 6)
//!
//! Most complex instance. Two phases:
//! Phase 1: Address binding (bytecode_log_k rounds)
//!   Polynomial: H(k) = Sum_s gamma^s * F_s[k] * (Val_s(k) + RAF_s(k))
//!   where F_s[k] = Sum_c eq(r_cycle_s, c) * delta(PC(c)=k)
//!   Both F_s and Val are linear in the bound address variable, so the product
//!   gives a DEGREE 2 round polynomial.
//!
//! Phase 2: Cycle binding (n_cycle_vars rounds)
//!   After binding address to r_addr, polynomial becomes:
//!   f(c) = [Prod_i ra_chunk_i(c)] * [Sum_s gamma^s * bound_val_s * eq_s(c)]
//!   Degree = bytecode_d + 1

const std = @import("std");
const Allocator = std.mem.Allocator;
const ThreadPool = @import("zolt_pool").ThreadPool;
const GpuPolyOps = @import("zolt_arith").gpu.GpuPolyOps;
const poly_mod = @import("zolt_arith").poly;
const UnreducedProductAccum = @import("zolt_arith").field.UnreducedProductAccum;
const jolt_types = @import("../jolt_types.zig");
const OpeningClaims = jolt_types.OpeningClaims;
const OpeningId = jolt_types.OpeningId;
const tracer = @import("../../tracer/mod.zig");
const ExecutionTrace = tracer.ExecutionTrace;
const preprocessing = @import("../preprocessing.zig");
const BytecodePCMapper = preprocessing.BytecodePCMapper;
const stage6_helpers = @import("stage6_helpers.zig");
const computeEqTable = stage6_helpers.computeEqTable;
const computeEqTableParallel = stage6_helpers.computeEqTableParallel;
const extractChunkMSB = stage6_helpers.extractChunkMSB;
const fieldFromI128 = stage6_helpers.fieldFromI128;
const zkvm_debug = @import("../debug.zig");
const dbg = zkvm_debug.dbg;
const debug_verbose = zkvm_debug.verbose;
const bytecode_entry_mod = @import("bytecode_entries.zig");
const BytecodeEntry = bytecode_entry_mod.BytecodeEntry;
const instruction_mod = @import("../instruction/mod.zig");
const CircuitFlags = instruction_mod.CircuitFlags;
const InstructionFlags = instruction_mod.InstructionFlags;

// Maximum evaluation points for parallelReduce accumulator.
// Covers all sub-provers: LookupsRa (M+2 ≤ 10), RamRa (d+2 ≤ 6), BytecodeReadRaf (d+2 ≤ 4).
const MAX_RA_EVALS = 16;

pub fn BytecodeReadRafProver(comptime F: type) type {
    return struct {
        const Self = @This();

        /// Phase 1: Separate F_s and val_with_raf arrays per stage
        /// F_s_arrs[s][k] = Sum_c:PC(c)=k eq(r_cycle_s, c)
        F_s_arrs: [5][]F,
        /// val_with_raf[s][k] = Val_s(k) + RAF_s(k)
        val_with_raf: [5][]F,
        /// Per-stage running claims for Phase 1
        stage_claims: [5]F,

        /// Phase 2: 5 GruenSplitEq instances for per-stage eq(r_cycle_s, .)
        /// Replaces flat combined[] with O(1) bind per round.
        stage_gruen_eqs: [5]?poly_mod.GruenSplitEqPolynomial(F),
        /// Phase 2: per-stage bound_vals (gamma^s * val_with_raf[s][0])
        bound_vals_phase2: [5]F,
        /// Phase 2: entry correction scalar, starts at entry_gamma * bound_f_entry,
        /// multiplied by (1-r) each bind round (tracks eq(0...0, challenges))
        entry_correction_scalar: F,

        /// Phase 2: RA chunk polynomials ra_chunks[i][c]
        ra_chunks: ?[][]F,

        /// Phase tracking
        phase: u8,
        bytecode_log_k: usize,
        n_cycle_vars: usize,
        bytecode_d: usize,
        log_k_chunk: usize,
        current_len: usize,
        addr_rounds_done: usize,

        /// Stored from Phase 1→2 transition for diagnostics
        bound_vals_stored: [5]F,
        /// F_s[0] values saved before freeing F_s_arrs (for Phase 2 consistency check)
        f_s_bound_saved: [5]F,

        /// Data needed for phase transition
        trace: *const ExecutionTrace,
        pc_map: *const BytecodePCMapper,
        stage_r_cycles: [5][]const F,
        gamma_powers: [8]F,
        /// Val polynomials per stage: val_polys[s][k]
        val_polys: [5][]F,
        /// Identity polynomial: int_poly[k] = k as field element
        int_poly: []F,

        entry_gamma: F,
        entry_val: F,
        entry_ri: usize,
        bound_f_entry: F,
        eq_zero_scalar: F,

        allocator: Allocator,
        pool: ?*ThreadPool = null,
        gpu: ?*GpuPolyOps = null,

        pub fn init(
            allocator: Allocator,
            trace: *const ExecutionTrace,
            pc_map: *const BytecodePCMapper,
            val_polys: [5][]F, // Val_s(k) for each stage, length bytecode_K each
            bytecode_log_k: usize,
            n_cycle_vars: usize,
            bytecode_d: usize,
            log_k_chunk: usize,
            gamma_powers: [8]F,
            stage_r_cycles: [5][]const F,
            int_poly: []F,
            external_stage_claims: [5]F, // From opening claims: claim_per_stage[s]
            entry_bytecode_index: usize,
            init_pool: ?*ThreadPool,
        ) !Self {
            const bytecode_K: usize = @as(usize, 1) << @intCast(bytecode_log_k);

            // Phase 1: Build separate F_s and val_with_raf arrays per stage
            var F_s_arrs: [5][]F = undefined;
            var val_with_raf_arrs: [5][]F = undefined;
            var stage_claims_init: [5]F = undefined;

            // Split-eq F_s computation: replaces T-sized eq tables with sqrt(T)-sized E_lo/E_hi
            // F_s[s][k] = Σ_c eq(r_cycle_s, c) * δ(PC(c)=k)
            //           = Σ_{c_hi} E_hi[c_hi] * (Σ_{c_lo: PC(c)=k} E_lo[c_lo])
            // Inner loop over c_lo is additions only; one mul per touched PC per c_hi block.
            const lo_bits = n_cycle_vars / 2;
            const hi_bits = n_cycle_vars - lo_bits;
            const in_len: usize = @as(usize, 1) << @intCast(lo_bits);
            const out_len: usize = @as(usize, 1) << @intCast(hi_bits);

            // Compute all 5 stages' E_lo and E_hi tables, then run all 5 double-loops.
            // Each stage has its own buffers to enable parallel execution.
            var E_lo_arr: [5][]F = undefined;
            var E_hi_arr: [5][]F = undefined;

            for (0..5) |s| {
                var r_cycle_rev_s = try allocator.alloc(F, n_cycle_vars);
                defer allocator.free(r_cycle_rev_s);
                for (0..n_cycle_vars) |i| {
                    r_cycle_rev_s[i] = stage_r_cycles[s][n_cycle_vars - 1 - i];
                }
                E_lo_arr[s] = try computeEqTableParallel(F, allocator, r_cycle_rev_s[0..lo_bits], lo_bits, init_pool);
                E_hi_arr[s] = try computeEqTableParallel(F, allocator, r_cycle_rev_s[lo_bits..n_cycle_vars], hi_bits, init_pool);
            }
            defer for (0..5) |s| {
                allocator.free(E_lo_arr[s]);
                allocator.free(E_hi_arr[s]);
            };

            // Allocate F_s output and per-stage temp buffers
            for (0..5) |s| {
                F_s_arrs[s] = try allocator.alloc(F, bytecode_K);
                @memset(F_s_arrs[s], F.zero());
            }

            // Run all 5 stages' double-loops in parallel (each stage independent)
            // Pre-allocate per-stage heap buffers (bytecode_K can be >256 for large programs)
            var per_stage_inner: [5][]F = undefined;
            var per_stage_touched: [5][]usize = undefined;
            var per_stage_tset: [5][]bool = undefined;
            for (0..5) |s| {
                per_stage_inner[s] = try allocator.alloc(F, bytecode_K);
                @memset(per_stage_inner[s], F.zero());
                per_stage_touched[s] = try allocator.alloc(usize, bytecode_K);
                per_stage_tset[s] = try allocator.alloc(bool, bytecode_K);
                @memset(per_stage_tset[s], false);
            }
            defer for (0..5) |s| {
                allocator.free(per_stage_inner[s]);
                allocator.free(per_stage_touched[s]);
                allocator.free(per_stage_tset[s]);
            };

            if (init_pool) |pool| {
                const FsCtx = struct {
                    F_s_out: *[5][]F,
                    E_lo_a: *[5][]F,
                    E_hi_a: *[5][]F,
                    steps: []const tracer.TraceStep,
                    pc_map_ptr: *const BytecodePCMapper,
                    in_len: usize,
                    out_len: usize,
                    lo_bits: usize,
                    bK: usize,
                    inner_bufs: *[5][]F,
                    touched_bufs: *[5][]usize,
                    tset_bufs: *[5][]bool,
                };
                const fs_ctx = FsCtx{
                    .F_s_out = &F_s_arrs,
                    .E_lo_a = &E_lo_arr,
                    .E_hi_a = &E_hi_arr,
                    .steps = trace.steps.items,
                    .pc_map_ptr = pc_map,
                    .in_len = in_len,
                    .out_len = out_len,
                    .lo_bits = lo_bits,
                    .bK = bytecode_K,
                    .inner_bufs = &per_stage_inner,
                    .touched_bufs = &per_stage_touched,
                    .tset_bufs = &per_stage_tset,
                };
                pool.parallelForForce(5, fs_ctx, struct {
                    fn f(c: FsCtx, s: usize) void {
                        const E_lo = c.E_lo_a[s];
                        const E_hi = c.E_hi_a[s];
                        const F_s = c.F_s_out[s];
                        const inner_buf = c.inner_bufs[s];
                        const touched_buf = c.touched_bufs[s];
                        const touched_set = c.tset_bufs[s];

                        for (0..c.out_len) |c_hi| {
                            var touched_count: usize = 0;
                            for (0..c.in_len) |c_lo| {
                                const idx = c_lo + (c_hi << @intCast(c.lo_bits));
                                const step = c.steps[idx];
                                const pc_idx = c.pc_map_ptr.getPCForStep(step);
                                if (pc_idx < c.bK) {
                                    if (!touched_set[pc_idx]) {
                                        touched_set[pc_idx] = true;
                                        touched_buf[touched_count] = pc_idx;
                                        touched_count += 1;
                                    }
                                    inner_buf[pc_idx] = inner_buf[pc_idx].add(E_lo[c_lo]);
                                }
                            }
                            const e_hi_val = E_hi[c_hi];
                            for (0..touched_count) |ti| {
                                const pc = touched_buf[ti];
                                F_s[pc] = F_s[pc].add(e_hi_val.mul(inner_buf[pc]));
                                inner_buf[pc] = F.zero();
                                touched_set[pc] = false;
                            }
                        }
                    }
                }.f);
            } else {
                // Sequential fallback
                var inner_buf = try allocator.alloc(F, bytecode_K);
                defer allocator.free(inner_buf);
                var touched_buf = try allocator.alloc(usize, bytecode_K);
                defer allocator.free(touched_buf);
                var touched_set = try allocator.alloc(bool, bytecode_K);
                defer allocator.free(touched_set);

                for (0..5) |s| {
                    @memset(inner_buf, F.zero());
                    @memset(touched_set, false);

                    for (0..out_len) |c_hi| {
                        var touched_count: usize = 0;
                        for (0..in_len) |c_lo| {
                            const c = c_lo + (c_hi << @intCast(lo_bits));
                            const step = trace.steps.items[c];
                            const pc_idx = pc_map.getPCForStep(step);
                            if (pc_idx < bytecode_K) {
                                if (!touched_set[pc_idx]) {
                                    touched_set[pc_idx] = true;
                                    touched_buf[touched_count] = pc_idx;
                                    touched_count += 1;
                                }
                                inner_buf[pc_idx] = inner_buf[pc_idx].add(E_lo_arr[s][c_lo]);
                            }
                        }
                        const e_hi_val = E_hi_arr[s][c_hi];
                        for (0..touched_count) |ti| {
                            const pc = touched_buf[ti];
                            F_s_arrs[s][pc] = F_s_arrs[s][pc].add(e_hi_val.mul(inner_buf[pc]));
                            inner_buf[pc] = F.zero();
                            touched_set[pc] = false;
                        }
                    }
                }
            }

            // Build val_with_raf and compute claims for each stage
            for (0..5) |s| {
                // val_with_raf[s][k] = Val_s(k) + RAF_s(k)
                val_with_raf_arrs[s] = try allocator.alloc(F, bytecode_K);
                for (0..bytecode_K) |k| {
                    var val_plus_raf = if (val_polys[s].len > k) val_polys[s][k] else F.zero();
                    // RAF terms
                    if (s == 0) {
                        val_plus_raf = val_plus_raf.add(gamma_powers[5].mul(int_poly[k]));
                    } else if (s == 2) {
                        val_plus_raf = val_plus_raf.add(gamma_powers[4].mul(int_poly[k]));
                    }
                    val_with_raf_arrs[s][k] = val_plus_raf;
                }

                // Compute claim from val_polys and F_s
                var recomputed_claim = F.zero();
                for (0..bytecode_K) |k| {
                    recomputed_claim = recomputed_claim.add(F_s_arrs[s][k].mul(val_with_raf_arrs[s][k]));
                }

                // Use val_poly-derived claims for sumcheck consistency
                // The sumcheck polynomial must sum to the claimed value,
                // and the polynomial is built from val_polys and F_s.
                // If we use external claims that differ from the actual polynomial sum,
                // the sumcheck will be inconsistent.
                stage_claims_init[s] = recomputed_claim;

                // Debug: Check if recomputed matches external
                if (comptime debug_verbose) {
                    const match_ext = @as(u8, if (recomputed_claim.eql(external_stage_claims[s])) 1 else 0);
                    if (match_ext == 0) {
                        const rc_full = recomputed_claim.toBytesBE();
                        const ec_full = external_stage_claims[s].toBytesBE();
                        dbg("[BCRAF_MISMATCH] Stage {d}: recomputed != external!\n", .{s});
                        dbg("  recomputed_LE=[", .{});
                        for (0..32) |bi| dbg("{x:0>2}", .{rc_full[31 - bi]});
                        dbg("]\n  external_LE=[", .{});
                        for (0..32) |bi| dbg("{x:0>2}", .{ec_full[31 - bi]});
                        dbg("]\n", .{});
                    }
                }
            }

            // Debug: print per-stage claims with full aggregation detail
            if (comptime debug_verbose) {
                var total = F.zero();
                for (0..5) |s| {
                    const term = gamma_powers[s].mul(stage_claims_init[s]);
                    total = total.add(term);
                    const sc_le = stage_claims_init[s].toBytes();
                    const gp_le = gamma_powers[s].toBytes();
                    const tm_le = term.toBytes();
                    dbg("[BCRAF_AGG_PR] s={} sc==ext={}", .{
                        s, @as(u8, if (stage_claims_init[s].eql(external_stage_claims[s])) 1 else 0),
                    });
                    dbg(" gp=[{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}]", .{
                        gp_le[0], gp_le[1], gp_le[2], gp_le[3], gp_le[4], gp_le[5], gp_le[6], gp_le[7],
                    });
                    dbg(" sc=[{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}]", .{
                        sc_le[0], sc_le[1], sc_le[2], sc_le[3], sc_le[4], sc_le[5], sc_le[6], sc_le[7],
                    });
                    dbg(" term=[{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}]", .{
                        tm_le[0], tm_le[1], tm_le[2], tm_le[3], tm_le[4], tm_le[5], tm_le[6], tm_le[7],
                    });
                    dbg("\n", .{});
                }
                const tl = total.toBytes();
                dbg("[BCRAF_AGG_PR] total_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{ tl[0], tl[1], tl[2], tl[3], tl[4], tl[5], tl[6], tl[7] });
            }

            return Self{
                .F_s_arrs = F_s_arrs,
                .val_with_raf = val_with_raf_arrs,
                .stage_claims = stage_claims_init,
                .stage_gruen_eqs = [_]?poly_mod.GruenSplitEqPolynomial(F){null} ** 5,
                .bound_vals_phase2 = [_]F{F.zero()} ** 5,
                .entry_correction_scalar = F.zero(),
                .ra_chunks = null,
                .phase = 0,
                .bytecode_log_k = bytecode_log_k,
                .n_cycle_vars = n_cycle_vars,
                .bytecode_d = bytecode_d,
                .log_k_chunk = log_k_chunk,
                .current_len = bytecode_K,
                .addr_rounds_done = 0,
                .bound_vals_stored = [_]F{F.zero()} ** 5,
                .f_s_bound_saved = [_]F{F.zero()} ** 5,
                .trace = trace,
                .pc_map = pc_map,
                .stage_r_cycles = stage_r_cycles,
                .gamma_powers = gamma_powers,
                .val_polys = val_polys,
                .int_poly = int_poly,
                .entry_gamma = gamma_powers[7],
                .entry_val = F.one(),
                .entry_ri = entry_bytecode_index,
                .bound_f_entry = F.zero(),
                .eq_zero_scalar = F.one(),
                .allocator = allocator,
                .pool = init_pool,
            };
        }

        pub fn deinit(self: *Self) void {
            // Phase 1 arrays (freed during transition if we got that far)
            for (0..5) |s| {
                if (self.F_s_arrs[s].len > 0) self.allocator.free(self.F_s_arrs[s]);
                if (self.val_with_raf[s].len > 0) self.allocator.free(self.val_with_raf[s]);
            }
            // Phase 2: GruenSplitEq instances
            for (0..5) |s| {
                if (self.stage_gruen_eqs[s]) |*g| {
                    var gruen = g.*;
                    gruen.deinit();
                }
            }
            if (self.ra_chunks) |chunks| {
                for (chunks) |arr| self.allocator.free(arr);
                self.allocator.free(chunks);
            }
            for (&self.val_polys) |vp| {
                if (vp.len > 0) self.allocator.free(vp);
            }
            self.allocator.free(self.int_poly);
        }

        /// Phase 1: degree-2 round poly over address vars
        /// Returns .{ agg=[p(0), p(2)], per_stage=[5][eval_0, eval_2] }
        /// Matches Jolt's approach: product of F_s (linear) and val_with_raf (linear) = degree 2
        pub fn computeRoundPolyPhase1(self: *Self) struct { agg: [2]F, per_stage: [5][2]F } {
            const half = self.current_len / 2;
            var per_stage: [5][2]F = undefined;

            if (self.pool) |pool| {
                // Compute 5 stages in parallel, each accumulating [2]F
                const Ctx = struct {
                    F_s_arrs: *const [5][]F,
                    val_with_raf: *const [5][]F,
                    half: usize,
                    results: *[5][2]F,
                };
                const ctx = Ctx{
                    .F_s_arrs = &self.F_s_arrs,
                    .val_with_raf = &self.val_with_raf,
                    .half = half,
                    .results = &per_stage,
                };
                pool.parallelForForce(5, ctx, struct {
                    fn f(c: Ctx, s: usize) void {
                        var eval_at_0 = F.zero();
                        var eval_at_2 = F.zero();
                        for (0..c.half) |k| {
                            const f_lo = c.F_s_arrs[s][2 * k];
                            const f_hi = c.F_s_arrs[s][2 * k + 1];
                            const v_lo = c.val_with_raf[s][2 * k];
                            const v_hi = c.val_with_raf[s][2 * k + 1];

                            eval_at_0 = eval_at_0.add(f_lo.mul(v_lo));

                            const f_at_2 = f_hi.add(f_hi).sub(f_lo);
                            const v_at_2 = v_hi.add(v_hi).sub(v_lo);
                            eval_at_2 = eval_at_2.add(f_at_2.mul(v_at_2));
                        }
                        c.results[s] = [2]F{ eval_at_0, eval_at_2 };
                    }
                }.f);
            } else {
                for (0..5) |s| {
                    var eval_at_0 = F.zero();
                    var eval_at_2 = F.zero();

                    for (0..half) |k| {
                        const f_lo = self.F_s_arrs[s][2 * k];
                        const f_hi = self.F_s_arrs[s][2 * k + 1];
                        const v_lo = self.val_with_raf[s][2 * k];
                        const v_hi = self.val_with_raf[s][2 * k + 1];

                        eval_at_0 = eval_at_0.add(f_lo.mul(v_lo));

                        const f_at_2 = f_hi.add(f_hi).sub(f_lo);
                        const v_at_2 = v_hi.add(v_hi).sub(v_lo);
                        eval_at_2 = eval_at_2.add(f_at_2.mul(v_at_2));
                    }

                    per_stage[s] = [2]F{ eval_at_0, eval_at_2 };
                }
            }

            var agg_eval_0 = F.zero();
            var agg_eval_2 = F.zero();
            for (0..5) |s| {
                agg_eval_0 = agg_eval_0.add(self.gamma_powers[s].mul(per_stage[s][0]));
                agg_eval_2 = agg_eval_2.add(self.gamma_powers[s].mul(per_stage[s][1]));
            }

            const entry_bit = self.entry_ri & 1;
            const ev_sq = self.entry_val.mul(self.entry_val);
            const eg_ev = self.entry_gamma.mul(ev_sq);
            if (entry_bit == 0) {
                agg_eval_0 = agg_eval_0.add(eg_ev);
                agg_eval_2 = agg_eval_2.add(eg_ev);
            } else {
                agg_eval_2 = agg_eval_2.add(eg_ev.add(eg_ev).add(eg_ev).add(eg_ev));
            }

            return .{ .agg = [2]F{ agg_eval_0, agg_eval_2 }, .per_stage = per_stage };
        }

        /// Bind challenge and update per-stage claims from polynomial evaluation
        /// per_stage_evals: [5][eval_0, eval_2] from computeRoundPolyPhase1
        pub fn bindChallengePhase1(self: *Self, r: F, per_stage_evals: [5][2]F) void {
            const half = self.current_len / 2;
            const two = F.fromU64(2);
            const two_inv = two.inverse().?;

            const bindStage = struct {
                fn f(
                    F_s: []F,
                    vwr: []F,
                    stage_claim: *F,
                    h: usize,
                    challenge: F,
                    pse: [2]F,
                    t_inv: F,
                ) void {
                    // Bind F_s and val_with_raf arrays
                    for (0..h) |k| {
                        F_s[k] = F_s[2 * k].add(challenge.mul(F_s[2 * k + 1].sub(F_s[2 * k])));
                        vwr[k] = vwr[2 * k].add(challenge.mul(vwr[2 * k + 1].sub(vwr[2 * k])));
                    }

                    // Update per-stage claim
                    const p0 = pse[0];
                    const p2 = pse[1];
                    const p1 = stage_claim.*.sub(p0);
                    const a0 = p0;
                    const a2 = p2.sub(p1.add(p1)).add(p0).mul(t_inv);
                    const a1 = p1.sub(p0).sub(a2);
                    stage_claim.* = a0.add(challenge.mul(a1.add(challenge.mul(a2))));
                }
            }.f;

            const bindOneArr = struct {
                fn f(arr: []F, h: usize, challenge: F) void {
                    for (0..h) |k| {
                        arr[k] = arr[2 * k].add(challenge.mul(arr[2 * k + 1].sub(arr[2 * k])));
                    }
                }
            }.f;

            const updateClaim = struct {
                fn f(stage_claim: *F, pse: [2]F, challenge: F, t_inv: F) void {
                    const p0 = pse[0];
                    const p2 = pse[1];
                    const p1 = stage_claim.*.sub(p0);
                    const a0 = p0;
                    const a2 = p2.sub(p1.add(p1)).add(p0).mul(t_inv);
                    const a1 = p1.sub(p0).sub(a2);
                    stage_claim.* = a0.add(challenge.mul(a1.add(challenge.mul(a2))));
                }
            }.f;

            if (self.gpu) |gpu| {
                if (half >= 16384) {
                    // GPU bind: 5 stages x 2 arrays each, then update claims on CPU
                    for (0..5) |s| {
                        gpu.polyBindLow(self.F_s_arrs[s][0 .. half * 2], r, self.F_s_arrs[s][0..half]) catch bindOneArr(self.F_s_arrs[s], half, r);
                        gpu.polyBindLow(self.val_with_raf[s][0 .. half * 2], r, self.val_with_raf[s][0..half]) catch bindOneArr(self.val_with_raf[s], half, r);
                        updateClaim(&self.stage_claims[s], per_stage_evals[s], r, two_inv);
                    }
                } else {
                    for (0..5) |s| {
                        bindStage(
                            self.F_s_arrs[s],
                            self.val_with_raf[s],
                            &self.stage_claims[s],
                            half,
                            r,
                            per_stage_evals[s],
                            two_inv,
                        );
                    }
                }
            } else if (self.pool) |pool| {
                const Ctx = struct {
                    F_s_arrs: *[5][]F,
                    val_with_raf: *[5][]F,
                    stage_claims: *[5]F,
                    half: usize,
                    r: F,
                    per_stage_evals: [5][2]F,
                    two_inv: F,
                };
                const ctx = Ctx{
                    .F_s_arrs = &self.F_s_arrs,
                    .val_with_raf = &self.val_with_raf,
                    .stage_claims = &self.stage_claims,
                    .half = half,
                    .r = r,
                    .per_stage_evals = per_stage_evals,
                    .two_inv = two_inv,
                };
                pool.parallelForForce(5, ctx, struct {
                    fn f(c: Ctx, s: usize) void {
                        bindStage(
                            c.F_s_arrs[s],
                            c.val_with_raf[s],
                            &c.stage_claims[s],
                            c.half,
                            c.r,
                            c.per_stage_evals[s],
                            c.two_inv,
                        );
                    }
                }.f);
            } else {
                for (0..5) |s| {
                    bindStage(
                        self.F_s_arrs[s],
                        self.val_with_raf[s],
                        &self.stage_claims[s],
                        half,
                        r,
                        per_stage_evals[s],
                        two_inv,
                    );
                }
            }

            const entry_bit = self.entry_ri & 1;
            if (entry_bit == 0) {
                self.entry_val = self.entry_val.mul(F.one().sub(r));
            } else {
                self.entry_val = self.entry_val.mul(r);
            }
            self.entry_ri >>= 1;

            self.current_len = half;
            self.addr_rounds_done += 1;
        }

        /// Transition from Phase 1 to Phase 2 after binding all address vars
        /// r_address_challenges are the challenges from Phase 1 in binding order (low-to-high)
        pub fn transitionToPhase2(
            self: *Self,
            r_address_challenges: []const F, // Low-to-high binding order from the sumcheck
        ) !void {
            const T: usize = @as(usize, 1) << @intCast(self.n_cycle_vars);
            const bytecode_K: usize = @as(usize, 1) << @intCast(self.bytecode_log_k);

            // The Phase 1 sumcheck binds variables in LowToHigh order:
            // r_address_challenges[0] = r_0 (bound to LSB of index), ..., [n-1] = r_{n-1} (MSB).
            //
            // For computing val_eval = Σ_k val[k] * eq(k, r), we need:
            // eq[k] = Π_j (r_j if bit j of k is 1, else 1-r_j).
            // Our computeEqTable with LE indexing gives exactly this when passed
            // the challenges in LH order.
            //
            // For RA chunk computation, Jolt uses r_address_BE = [r_{n-1},...,r_0]
            // and chunks sequentially: chunk 0 = MSB vars, chunk d-1 = LSB vars.
            // We keep r_address_be for RA chunk slicing (same convention as before).

            // Print address challenges (ALWAYS ON for debugging)
            {
                dbg("[BCRAF_TRANS] r_address_challenges (len={}, LH order):\n", .{self.bytecode_log_k});
                for (0..self.bytecode_log_k) |i| {
                    const ch_be = r_address_challenges[i].toBytesBE();
                    dbg("  ch[{d}]_LE=[", .{i});
                    for (0..32) |bi| dbg("{x:0>2}", .{ch_be[31 - bi]});
                    dbg("]\n", .{});
                }
            }

            // Compute r_address_be for RA chunk slicing (same as before)
            var r_address_be = try self.allocator.alloc(F, self.bytecode_log_k);
            defer self.allocator.free(r_address_be);
            for (0..self.bytecode_log_k) |i| {
                r_address_be[i] = r_address_challenges[self.bytecode_log_k - 1 - i];
            }

            // Compute bound_vals[s] = Val_s(r_address) + RAF_s(r_address)
            // The sumcheck binds variables MSB-first: r_address_challenges[0] = MSB.
            // But val_poly coefficients are indexed with bit 0 = LSB.
            // Jolt's verifier reverses challenges (normalize_opening_point) before evaluate,
            // so r[0] = LSB challenge maps to bit 0 of coefficient index.
            // Use r_address_be (reversed) for the eq table, matching Jolt's normalize_opening_point.
            const eq_addr = try computeEqTableParallel(F, self.allocator, r_address_be, self.bytecode_log_k, self.pool);
            defer self.allocator.free(eq_addr);

            // Debug: eq_addr entries
            if (comptime debug_verbose) {
                for (0..bytecode_K) |ek| {
                    const eab = eq_addr[ek].toBytesBE();
                    dbg("[ZOLT_EQ_ADDR] eq[{d}]_LE=[", .{ek});
                    for (0..32) |bi| dbg("{x:0>2}", .{eab[31 - bi]});
                    dbg("]\n", .{});
                }
            }

            // Debug: val_polys entries
            if (comptime debug_verbose) {
                for (0..5) |vs| {
                    for (0..bytecode_K) |kk| {
                        const vpk = self.val_polys[vs][kk].toBytesBE();
                        dbg("[ZOLT_VP] Val[{d}][{d}]_LE=[", .{ vs, kk });
                        for (0..8) |bi| dbg("{x:0>2}", .{vpk[31 - bi]});
                        dbg("]\n", .{});
                    }
                }
            }

            var bound_vals: [5]F = undefined;
            for (0..5) |s| {
                var val_eval = F.zero();
                const max_k = @min(self.val_polys[s].len, bytecode_K);
                for (0..max_k) |k| {
                    val_eval = val_eval.add(self.val_polys[s][k].mul(eq_addr[k]));
                }

                // Add RAF terms (identity polynomial contribution)
                // Stage 0: RAF = gamma^5 * identity_eval
                // Stage 2: RAF = gamma^4 * identity_eval
                if (s == 0) {
                    var identity_eval = F.zero();
                    for (0..bytecode_K) |k| {
                        identity_eval = identity_eval.add(self.int_poly[k].mul(eq_addr[k]));
                    }
                    const raf_contrib = self.gamma_powers[5].mul(identity_eval);
                    val_eval = val_eval.add(raf_contrib);
                    // Print identity_eval, gamma[5], RAF contribution, val_before_raf
                    const ie_be = identity_eval.toBytesBE();
                    const g5_be = self.gamma_powers[5].toBytesBE();
                    const rc_be = raf_contrib.toBytesBE();
                    dbg("[TRANS_RAF] s=0: identity_eval_LE=[", .{});
                    for (0..32) |bi| dbg("{x:0>2}", .{ie_be[31 - bi]});
                    dbg("] gamma5_LE=[", .{});
                    for (0..32) |bi| dbg("{x:0>2}", .{g5_be[31 - bi]});
                    dbg("] raf_LE=[", .{});
                    for (0..32) |bi| dbg("{x:0>2}", .{rc_be[31 - bi]});
                    dbg("]\n", .{});
                } else if (s == 2) {
                    var identity_eval = F.zero();
                    for (0..bytecode_K) |k| {
                        identity_eval = identity_eval.add(self.int_poly[k].mul(eq_addr[k]));
                    }
                    const raf_contrib = self.gamma_powers[4].mul(identity_eval);
                    val_eval = val_eval.add(raf_contrib);
                    const ie_be = identity_eval.toBytesBE();
                    const g4_be = self.gamma_powers[4].toBytesBE();
                    const rc_be = raf_contrib.toBytesBE();
                    dbg("[TRANS_RAF] s=2: identity_eval_LE=[", .{});
                    for (0..32) |bi| dbg("{x:0>2}", .{ie_be[31 - bi]});
                    dbg("] gamma4_LE=[", .{});
                    for (0..32) |bi| dbg("{x:0>2}", .{g4_be[31 - bi]});
                    dbg("] raf_LE=[", .{});
                    for (0..32) |bi| dbg("{x:0>2}", .{rc_be[31 - bi]});
                    dbg("]\n", .{});
                }

                // bound_vals[s] = gamma^s * val_with_raf[s][0]
                // After Phase 1 binding of all address variables, val_with_raf[s][0] is the
                // MLE evaluation at the binding point. stage_claims[s] = F_s[0]*val_with_raf[s][0]
                // since both are reduced to single elements.
                bound_vals[s] = self.gamma_powers[s].mul(self.val_with_raf[s][0]);
                self.bound_vals_stored[s] = bound_vals[s];

                // DIAGNOSTIC: compare re-computed val_eval with Phase 1 bound val_with_raf[s][0]
                if (comptime debug_verbose) {
                    const phase1_bound = self.val_with_raf[s][0];
                    dbg("[TRANS_CHECK] stage[{}]: match={}\n", .{ s, @as(u8, if (val_eval.eql(phase1_bound)) 1 else 0) });
                }

                // Debug: Print val_eval and bound_val for comparison with Jolt verifier
                if (comptime debug_verbose) {
                    const ve_be = val_eval.toBytesBE();
                    const bv_be = bound_vals[s].toBytesBE();
                    dbg("[BCRAF_TRANS] stage[{}]: val_eval_LE=[", .{s});
                    for (0..32) |bi| dbg("{x:0>2}", .{ve_be[31 - bi]});
                    dbg("] bound_val_LE=[", .{});
                    for (0..32) |bi| dbg("{x:0>2}", .{bv_be[31 - bi]});
                    dbg("]\n", .{});
                }
            }

            // Build RA chunk polynomials for cycle binding
            // ra_chunks[i][c] = eq(r_addr_chunk_i, PC_chunk_i(c))
            //
            // Like Jolt's compute_r_address_chunks: pad r_address_be with zeros at MSB
            // to make length a multiple of log_k_chunk, then split into d chunks of
            // exactly log_k_chunk variables each.
            const padded_len = self.bytecode_d * self.log_k_chunk;
            const pad_count = padded_len - self.bytecode_log_k;
            var r_address_be_padded = try self.allocator.alloc(F, padded_len);
            defer self.allocator.free(r_address_be_padded);
            // Pad MSB end with zeros (Jolt prepends zeros to r_address which is BE)
            for (0..pad_count) |i| {
                r_address_be_padded[i] = F.zero();
            }
            for (0..self.bytecode_log_k) |i| {
                r_address_be_padded[pad_count + i] = r_address_be[i];
            }

            self.ra_chunks = try self.allocator.alloc([]F, self.bytecode_d);
            const chunk_K: usize = @as(usize, 1) << @intCast(self.log_k_chunk);

            // Pre-compute eq tables for each chunk (small, sequential is fine)
            var eq_tables: [8][]F = undefined; // max bytecode_d = 8
            for (0..self.bytecode_d) |i| {
                self.ra_chunks.?[i] = try self.allocator.alloc(F, T);
                const chunk_start = i * self.log_k_chunk;
                const chunk_end = chunk_start + self.log_k_chunk;
                const r_chunk_be = r_address_be_padded[chunk_start..chunk_end];
                var r_chunk_rev = try self.allocator.alloc(F, self.log_k_chunk);
                defer self.allocator.free(r_chunk_rev);
                for (0..self.log_k_chunk) |ci| {
                    r_chunk_rev[ci] = r_chunk_be[self.log_k_chunk - 1 - ci];
                }
                eq_tables[i] = try computeEqTable(F, self.allocator, r_chunk_rev, self.log_k_chunk);
            }
            defer for (0..self.bytecode_d) |i| self.allocator.free(eq_tables[i]);

            // Fill ra_chunks in parallel across T elements
            if (self.pool) |pool| {
                const RaFillCtx = struct {
                    ra_chunks: [][]F,
                    eq_tables: *const [8][]F,
                    steps: []const tracer.TraceStep,
                    pc_map_ptr: *const BytecodePCMapper,
                    bytecode_d: usize,
                    bytecode_K: usize,
                    chunk_K: usize,
                    log_k_chunk: usize,
                };
                const fill_ctx = RaFillCtx{
                    .ra_chunks = self.ra_chunks.?,
                    .eq_tables = &eq_tables,
                    .steps = self.trace.steps.items,
                    .pc_map_ptr = self.pc_map,
                    .bytecode_d = self.bytecode_d,
                    .bytecode_K = bytecode_K,
                    .chunk_K = chunk_K,
                    .log_k_chunk = self.log_k_chunk,
                };
                pool.parallelFor(T, fill_ctx, struct {
                    fn f(c: RaFillCtx, cycle: usize) void {
                        const step = c.steps[cycle];
                        const pc = c.pc_map_ptr.getPCForStep(step);
                        for (0..c.bytecode_d) |i| {
                            if (pc < c.bytecode_K) {
                                const chunk_val = extractChunkMSB(pc, i, c.bytecode_d, c.log_k_chunk);
                                c.ra_chunks[i][cycle] = if (chunk_val < c.chunk_K) c.eq_tables.*[i][chunk_val] else F.zero();
                            } else {
                                c.ra_chunks[i][cycle] = F.zero();
                            }
                        }
                    }
                }.f);
            } else {
                for (0..T) |c| {
                    const step = self.trace.steps.items[c];
                    const pc = self.pc_map.getPCForStep(step);
                    for (0..self.bytecode_d) |i| {
                        if (pc < bytecode_K) {
                            const chunk_val = extractChunkMSB(pc, i, self.bytecode_d, self.log_k_chunk);
                            self.ra_chunks.?[i][c] = if (chunk_val < chunk_K) eq_tables[i][chunk_val] else F.zero();
                        } else {
                            self.ra_chunks.?[i][c] = F.zero();
                        }
                    }
                }
            }

            // Build 5 GruenSplitEq instances for per-stage eq(r_cycle_s, .)
            // stage_r_cycles[s] is in BE order; pass DIRECTLY to GruenSplitEq.
            // GruenSplitEq uses BE convention internally: E_out/E_in tables are BE-indexed,
            // and bind() processes tau[n-1] first (the LSB in BE = bit 0 of position index).
            // This matches the flat sumcheck's LowToHigh binding which groups positions
            // by bit 0 first. Same pattern as LookupsRaVirtualProver (line ~4446).
            for (0..5) |s| {
                self.stage_gruen_eqs[s] = try poly_mod.GruenSplitEqPolynomial(F).initWithScaling(
                    self.allocator,
                    self.stage_r_cycles[s][0..self.n_cycle_vars],
                    null,
                );
                self.bound_vals_phase2[s] = bound_vals[s];
            }

            // Debug: verify Π_i ra_chunk_i(c) = eq_addr[PC(c)] for each cycle
            // Debug: verify RA product vs eq_binding (behind comptime debug_verbose)
            if (comptime debug_verbose) {
                // Check RA product vs eq_binding (using LH challenges directly)
                const eq_binding_check = try computeEqTableParallel(F, self.allocator, r_address_challenges, self.bytecode_log_k, self.pool);
                defer self.allocator.free(eq_binding_check);
                var mismatch_count: usize = 0;
                for (0..T) |c| {
                    var ra_prod = F.one();
                    for (0..self.bytecode_d) |i| {
                        ra_prod = ra_prod.mul(self.ra_chunks.?[i][c]);
                    }
                    const step = self.trace.steps.items[c];
                    const pc = self.pc_map.getPCForStep(step);
                    const full_eq = if (pc < bytecode_K) eq_binding_check[pc] else F.zero();
                    if (!ra_prod.eql(full_eq)) mismatch_count += 1;
                }
                dbg("[BCRAF_RA] total mismatches: {}/{}\n", .{ mismatch_count, T });
            }

            // Save F_s[0] before freeing
            for (0..5) |s| {
                self.f_s_bound_saved[s] = if (self.F_s_arrs[s].len > 0) self.F_s_arrs[s][0] else F.zero();
            }

            // Free Phase 1 arrays (no longer needed)
            // Replace with zero-length allocations so deinit doesn't double-free
            for (0..5) |s| {
                self.allocator.free(self.F_s_arrs[s]);
                self.F_s_arrs[s] = try self.allocator.alloc(F, 0);
                self.allocator.free(self.val_with_raf[s]);
                self.val_with_raf[s] = try self.allocator.alloc(F, 0);
            }

            // Entry correction: track as a separate scalar that gets bound each round
            self.bound_f_entry = self.entry_val;
            self.entry_correction_scalar = self.entry_gamma.mul(self.bound_f_entry);

            self.current_len = T;
            self.phase = 1;
        }

        /// Phase 2: degree bytecode_d+1 round poly
        /// Returns evals in Toom-Cook format: [p(0), p(1), ..., p(d), p_inf]
        /// Phase 2: compute round polynomial using Toom-Cook delta approach.
        /// Evaluates ∏_i ra_chunk_i(x) * combined(x) at Toom points {0, 1, ..., d, ∞}
        /// using lo/delta decomposition (same pattern as RamRaVirtual).
        ///
        /// combined(pos) is computed on-the-fly from 5 GruenSplitEq instances:
        ///   combined(pos) = Σ_s eff_lo_s * E_out_s[x_out] * E_in_s[x_in] + entry_correction * δ(pos,0)
        /// where eff_lo_s = bound_vals[s] * scalar_s * (1 - tau_s_window)
        ///       eff_hi_s = bound_vals[s] * scalar_s * tau_s_window
        /// Returns evaluations at Toom points (not monomials).
        pub fn computeRoundPolyPhase2(self: *Self, allocator: Allocator) ![]F {
            const half = self.current_len / 2;
            const ra_chunks = self.ra_chunks.?;
            const d_total = self.bytecode_d + 1; // ra_chunks + combined = d_total linear polys
            const n_toom_evals = d_total; // Toom grid: {1, 2, ..., d_total-1, ∞}

            // Precompute per-stage effective scalars for lo (active var=0) and hi (active var=1)
            // For each stage s with GruenSplitEq:
            //   eq_s(2j) = scalar_s * E_out_s[x_out] * E_in_s[x_in] * (1 - tau_s_window)
            //   eq_s(2j+1) = scalar_s * E_out_s[x_out] * E_in_s[x_in] * tau_s_window
            // So: eff_lo_s = bound_vals[s] * scalar_s * (1 - tau_s_window)
            //     eff_hi_s = bound_vals[s] * scalar_s * tau_s_window
            var eff_lo: [5]F = undefined;
            var eff_hi: [5]F = undefined;
            var E_out_arr: [5][]const F = undefined;
            var E_in_arr: [5][]const F = undefined;
            var head_in_bits: usize = 0;
            for (0..5) |s| {
                const gruen = &self.stage_gruen_eqs[s].?;
                const tau_window = gruen.tau[gruen.current_index - 1];
                const one_minus_tau = F.one().sub(tau_window);
                const base = self.bound_vals_phase2[s].mul(gruen.current_scalar);
                eff_lo[s] = base.mul(one_minus_tau);
                eff_hi[s] = base.mul(tau_window);

                const eq_tables = gruen.getWindowEqTables(gruen.current_index, 1);
                E_out_arr[s] = eq_tables.E_out;
                E_in_arr[s] = eq_tables.E_in;
                head_in_bits = eq_tables.head_in_bits; // Same for all 5 stages
            }
            const in_mask: usize = (@as(usize, 1) << @intCast(head_in_bits)) - 1;
            const entry_corr = self.entry_correction_scalar;

            const Ctx = struct {
                ra_chunks: [][]F,
                bytecode_d: usize,
                d_total: usize,
                n_toom_evals: usize,
                eff_lo: [5]F,
                eff_hi: [5]F,
                E_out_arr: [5][]const F,
                E_in_arr: [5][]const F,
                head_in_bits: usize,
                in_mask: usize,
                entry_corr: F,
            };
            const ctx = Ctx{
                .ra_chunks = ra_chunks,
                .bytecode_d = self.bytecode_d,
                .d_total = d_total,
                .n_toom_evals = n_toom_evals,
                .eff_lo = eff_lo,
                .eff_hi = eff_hi,
                .E_out_arr = E_out_arr,
                .E_in_arr = E_in_arr,
                .head_in_bits = head_in_bits,
                .in_mask = in_mask,
                .entry_corr = entry_corr,
            };

            const UPA = UnreducedProductAccum;
            const mapFn = struct {
                fn f(c: Ctx, start: usize, end: usize) [MAX_RA_EVALS]F {
                    const MAX_D = 8;
                    var upa_acc: [MAX_RA_EVALS]UPA = .{UPA.zero()} ** MAX_RA_EVALS;

                    // Factored E_out/E_in: for each x_out block, pre-combine
                    // eff_lo/hi with E_out (saves 5 muls per j vs computing E_out*E_in per j).
                    var prev_x_out: usize = std.math.maxInt(usize);
                    var eff_lo_out: [5]F = undefined;
                    var eff_hi_out: [5]F = undefined;

                    for (start..end) |j| {
                        const x_out = j >> @intCast(c.head_in_bits);
                        const x_in = j & c.in_mask;

                        // Re-compute eff_lo/hi_out only when x_out changes
                        if (x_out != prev_x_out) {
                            prev_x_out = x_out;
                            inline for (0..5) |s| {
                                eff_lo_out[s] = c.eff_lo[s].mul(c.E_out_arr[s][x_out]);
                                eff_hi_out[s] = c.eff_hi[s].mul(c.E_out_arr[s][x_out]);
                            }
                        }

                        var combined_lo = F.zero();
                        var combined_hi = F.zero();
                        inline for (0..5) |s| {
                            const e_in = c.E_in_arr[s][x_in];
                            combined_lo = combined_lo.add(eff_lo_out[s].mul(e_in));
                            combined_hi = combined_hi.add(eff_hi_out[s].mul(e_in));
                        }
                        // Entry correction: only at position 0 (j=0, even position)
                        if (j == 0) {
                            combined_lo = combined_lo.add(c.entry_corr);
                        }

                        // Compute lo[i] = poly_i(2j), delta[i] = poly_i(2j+1) - poly_i(2j)
                        var lo: [MAX_D]F = undefined;
                        var delta: [MAX_D]F = undefined;
                        for (0..c.bytecode_d) |i| {
                            lo[i] = c.ra_chunks[i][2 * j];
                            delta[i] = c.ra_chunks[i][2 * j + 1].sub(lo[i]);
                        }
                        // combined is the last polynomial in the product
                        lo[c.bytecode_d] = combined_lo;
                        delta[c.bytecode_d] = combined_hi.sub(combined_lo);

                        // Evaluate product at point 0: ∏ lo[i]
                        {
                            var product = lo[0];
                            for (1..c.d_total) |i| product = product.mul(lo[i]);
                            upa_acc[0].addAssign(UPA.fromMul(product, F.one()));
                        }

                        // Evaluate product at points 1, 2, ..., d_total-2
                        // cur[i] = lo[i] + k*delta[i] for point k
                        var cur: [MAX_D]F = undefined;
                        for (0..c.d_total) |i| cur[i] = lo[i].add(delta[i]); // point 1
                        {
                            var product = cur[0];
                            for (1..c.d_total) |i| product = product.mul(cur[i]);
                            upa_acc[1].addAssign(UPA.fromMul(product, F.one()));
                        }
                        for (2..c.n_toom_evals) |k| {
                            for (0..c.d_total) |i| cur[i] = cur[i].add(delta[i]);
                            var product = cur[0];
                            for (1..c.d_total) |i| product = product.mul(cur[i]);
                            upa_acc[k].addAssign(UPA.fromMul(product, F.one()));
                        }

                        // Evaluate at ∞: ∏ delta[i] (leading coefficient)
                        {
                            var product = delta[0];
                            for (1..c.d_total) |i| product = product.mul(delta[i]);
                            upa_acc[c.n_toom_evals].addAssign(UPA.fromMul(product, F.one()));
                        }
                    }
                    var acc: [MAX_RA_EVALS]F = undefined;
                    for (0..MAX_RA_EVALS) |i| acc[i] = upa_acc[i].reduce();
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

            // Return Toom evaluations: [p(0), p(1), ..., p(d_total-1), p(∞)]
            const n_evals_out = n_toom_evals + 1;
            var evals = try allocator.alloc(F, n_evals_out);
            for (0..n_evals_out) |i| {
                evals[i] = result[i];
            }
            return evals;
        }

        pub fn bindChallengePhase2(self: *Self, r: F) void {
            const half = self.current_len / 2;
            const ra_chunks = self.ra_chunks.?;

            const bindOne = struct {
                fn f(arr: []F, h: usize, challenge: F) void {
                    for (0..h) |j| {
                        arr[j] = arr[2 * j].add(challenge.mul(arr[2 * j + 1].sub(arr[2 * j])));
                    }
                }
            }.f;

            // Bind ra_chunks (dense arrays)
            if (self.gpu) |gpu| {
                if (half >= 16384) {
                    for (0..self.bytecode_d) |i| {
                        gpu.polyBindLow(ra_chunks[i][0 .. half * 2], r, ra_chunks[i][0..half]) catch bindOne(ra_chunks[i], half, r);
                    }
                } else {
                    for (0..self.bytecode_d) |i| {
                        bindOne(ra_chunks[i], half, r);
                    }
                }
            } else if (self.pool) |pool| {
                const Ctx = struct { ra: [][]F, d: usize, half: usize, r: F };
                const ctx = Ctx{ .ra = ra_chunks, .d = self.bytecode_d, .half = half, .r = r };
                pool.parallelForForce(self.bytecode_d, ctx, struct {
                    fn f(c: Ctx, idx: usize) void {
                        bindOne(c.ra[idx], c.half, c.r);
                    }
                }.f);
            } else {
                for (0..self.bytecode_d) |i| {
                    bindOne(ra_chunks[i], half, r);
                }
            }

            // Bind 5 GruenSplitEq instances — O(1) each
            for (0..5) |s| {
                self.stage_gruen_eqs[s].?.bind(r);
            }

            // Bind entry correction scalar: entry(pos=0) * (1-r) + entry(pos=1) * r = entry * (1-r)
            self.entry_correction_scalar = self.entry_correction_scalar.mul(F.one().sub(r));

            self.current_len = half;
        }

        pub fn getOpeningClaims(self: *const Self, allocator: Allocator) ![]F {
            var claims = try allocator.alloc(F, self.bytecode_d);
            for (0..self.bytecode_d) |i| {
                claims[i] = self.ra_chunks.?[i][0];
            }
            return claims;
        }
    };
}

/// Compute BytecodeReadRaf input claim and per-stage claims
/// Returns .{ total_claim, [5]per_stage_claims }
pub fn computeBytecodeReadRafInputClaim(
    comptime F: type,
    opening_claims: *OpeningClaims(F),
    gamma_powers: []const F,
    stage1_gammas: []const F,
    stage2_gammas: []const F,
    stage3_gammas: []const F,
    stage4_gammas: []const F,
    stage5_gammas: []const F,
) struct { total: F, per_stage: [5]F } {
    const getClaim = struct {
        fn get(oc: *OpeningClaims(F), key: OpeningId) F {
            return oc.get(key) orelse F.zero();
        }
    }.get;

    // rv_claim_1 (Stage 1 / SpartanOuter)
    var rv1 = F.zero();
    const oc_upc = getClaim(opening_claims, .{ .Virtual = .{ .poly = .UnexpandedPC, .sumcheck_id = .SpartanOuter } });
    rv1 = rv1.add(oc_upc); // No gamma[0] - Jolt formula: unexpanded_pc + γ¹·imm + Σγ^(2+i)·cf[i]
    const oc_imm = getClaim(opening_claims, .{ .Virtual = .{ .poly = .Imm, .sumcheck_id = .SpartanOuter } });
    rv1 = rv1.add(stage1_gammas[1].mul(oc_imm));
    var oc_flags: [14]F = undefined;
    for (0..14) |i| {
        oc_flags[i] = getClaim(opening_claims, .{ .Virtual = .{ .poly = .{ .OpFlags = @intCast(i) }, .sumcheck_id = .SpartanOuter } });
        rv1 = rv1.add(stage1_gammas[2 + i].mul(oc_flags[i]));
    }
    // Debug: print each opening claim component for rv1
    {
        const upc_le = oc_upc.toBytes();
        const imm_le = oc_imm.toBytes();
        dbg("[BCRAF_RV1_DETAIL] oc_UnexpandedPC_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
            upc_le[0], upc_le[1], upc_le[2], upc_le[3], upc_le[4], upc_le[5], upc_le[6], upc_le[7],
        });
        dbg("[BCRAF_RV1_DETAIL] oc_Imm_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
            imm_le[0], imm_le[1], imm_le[2], imm_le[3], imm_le[4], imm_le[5], imm_le[6], imm_le[7],
        });
        for (0..14) |i| {
            const fl = oc_flags[i].toBytes();
            dbg("[BCRAF_RV1_DETAIL] oc_OpFlag[{}]_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                i, fl[0], fl[1], fl[2], fl[3], fl[4], fl[5], fl[6], fl[7],
            });
        }
        // Also print the oc_PC claim (used for RAF) and FlagIsNoop
        const oc_pc = getClaim(opening_claims, .{ .Virtual = .{ .poly = .PC, .sumcheck_id = .SpartanOuter } });
        const pc_le = oc_pc.toBytes();
        dbg("[BCRAF_RV1_DETAIL] oc_PC_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
            pc_le[0], pc_le[1], pc_le[2], pc_le[3], pc_le[4], pc_le[5], pc_le[6], pc_le[7],
        });
    }

    // rv_claim_2 (Stage 2 / SpartanProductVirtualization)
    var rv2 = F.zero();
    // Upstream: Jump + γ·Branch + γ²·WriteLookupOutputToRD + γ³·VirtualInstruction
    rv2 = rv2.add(stage2_gammas[0].mul(getClaim(opening_claims, .{ .Virtual = .{ .poly = .{ .OpFlags = 5 }, .sumcheck_id = .SpartanProductVirtualization } })));
    rv2 = rv2.add(stage2_gammas[1].mul(getClaim(opening_claims, .{ .Virtual = .{ .poly = .{ .InstructionFlags = 4 }, .sumcheck_id = .SpartanProductVirtualization } })));
    rv2 = rv2.add(stage2_gammas[2].mul(getClaim(opening_claims, .{ .Virtual = .{ .poly = .{ .OpFlags = 6 }, .sumcheck_id = .SpartanProductVirtualization } })));
    rv2 = rv2.add(stage2_gammas[3].mul(getClaim(opening_claims, .{ .Virtual = .{ .poly = .{ .OpFlags = 7 }, .sumcheck_id = .SpartanProductVirtualization } })));

    // rv_claim_3 (Stage 3)
    var rv3 = F.zero();
    rv3 = rv3.add(getClaim(opening_claims, .{ .Virtual = .{ .poly = .Imm, .sumcheck_id = .InstructionInputVirtualization } })); // No gamma[0] - Jolt formula: imm + γ¹·unexpanded_pc + ...
    rv3 = rv3.add(stage3_gammas[1].mul(getClaim(opening_claims, .{ .Virtual = .{ .poly = .UnexpandedPC, .sumcheck_id = .SpartanShift } })));
    rv3 = rv3.add(stage3_gammas[2].mul(getClaim(opening_claims, .{ .Virtual = .{ .poly = .{ .InstructionFlags = 2 }, .sumcheck_id = .InstructionInputVirtualization } })));
    rv3 = rv3.add(stage3_gammas[3].mul(getClaim(opening_claims, .{ .Virtual = .{ .poly = .{ .InstructionFlags = 0 }, .sumcheck_id = .InstructionInputVirtualization } })));
    rv3 = rv3.add(stage3_gammas[4].mul(getClaim(opening_claims, .{ .Virtual = .{ .poly = .{ .InstructionFlags = 3 }, .sumcheck_id = .InstructionInputVirtualization } })));
    rv3 = rv3.add(stage3_gammas[5].mul(getClaim(opening_claims, .{ .Virtual = .{ .poly = .{ .InstructionFlags = 1 }, .sumcheck_id = .InstructionInputVirtualization } })));
    rv3 = rv3.add(stage3_gammas[6].mul(getClaim(opening_claims, .{ .Virtual = .{ .poly = .{ .InstructionFlags = 5 }, .sumcheck_id = .SpartanShift } })));
    rv3 = rv3.add(stage3_gammas[7].mul(getClaim(opening_claims, .{ .Virtual = .{ .poly = .{ .OpFlags = 7 }, .sumcheck_id = .SpartanShift } })));
    rv3 = rv3.add(stage3_gammas[8].mul(getClaim(opening_claims, .{ .Virtual = .{ .poly = .{ .OpFlags = 12 }, .sumcheck_id = .SpartanShift } })));

    // rv_claim_4 (Stage 4)
    var rv4 = F.zero();
    rv4 = rv4.add(stage4_gammas[0].mul(getClaim(opening_claims, .{ .Virtual = .{ .poly = .RdWa, .sumcheck_id = .RegistersReadWriteChecking } })));
    rv4 = rv4.add(stage4_gammas[1].mul(getClaim(opening_claims, .{ .Virtual = .{ .poly = .Rs1Ra, .sumcheck_id = .RegistersReadWriteChecking } })));
    rv4 = rv4.add(stage4_gammas[2].mul(getClaim(opening_claims, .{ .Virtual = .{ .poly = .Rs2Ra, .sumcheck_id = .RegistersReadWriteChecking } })));

    // rv_claim_5 (Stage 5)
    const NUM_LOOKUP_TABLES: usize = 40;
    var rv5 = F.zero();
    const rv5_rdwa = getClaim(opening_claims, .{ .Virtual = .{ .poly = .RdWa, .sumcheck_id = .RegistersValEvaluation } });
    rv5 = rv5.add(rv5_rdwa); // No gamma[0] - Jolt formula: eq(rd,r) + γ¹·!interleaved + ...
    const rv5_raf_flag = getClaim(opening_claims, .{ .Virtual = .{ .poly = .InstructionRafFlag, .sumcheck_id = .InstructionReadRaf } });
    rv5 = rv5.add(stage5_gammas[1].mul(rv5_raf_flag));
    for (0..NUM_LOOKUP_TABLES) |i| {
        const lt_claim = getClaim(opening_claims, .{ .Virtual = .{ .poly = .{ .LookupTableFlag = i }, .sumcheck_id = .InstructionReadRaf } });
        rv5 = rv5.add(stage5_gammas[2 + i].mul(lt_claim));
        if (!lt_claim.eql(F.zero())) {
            const ltb = lt_claim.toBytes();
            dbg("[BCRAF_RV5] LookupTableFlag({})_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                i, ltb[0], ltb[1], ltb[2], ltb[3], ltb[4], ltb[5], ltb[6], ltb[7],
            });
        }
    }
    {
        const rdwa_le = rv5_rdwa.toBytes();
        const rff_le = rv5_raf_flag.toBytes();
        const rv5_le = rv5.toBytes();
        dbg("[BCRAF_RV5] RdWa_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
            rdwa_le[0], rdwa_le[1], rdwa_le[2], rdwa_le[3], rdwa_le[4], rdwa_le[5], rdwa_le[6], rdwa_le[7],
        });
        dbg("[BCRAF_RV5] InstructionRafFlag_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
            rff_le[0], rff_le[1], rff_le[2], rff_le[3], rff_le[4], rff_le[5], rff_le[6], rff_le[7],
        });
        dbg("[BCRAF_RV5] total_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
            rv5_le[0], rv5_le[1], rv5_le[2], rv5_le[3], rv5_le[4], rv5_le[5], rv5_le[6], rv5_le[7],
        });
    }

    // RAF claims
    const raf_claim = getClaim(opening_claims, .{ .Virtual = .{ .poly = .PC, .sumcheck_id = .SpartanOuter } });
    const raf_shift_claim = getClaim(opening_claims, .{ .Virtual = .{ .poly = .PC, .sumcheck_id = .SpartanShift } });

    // Debug: print per-stage rv_claims and raf_claims
    {
        const rv_arr = [5]F{ rv1, rv2, rv3, rv4, rv5 };
        for (0..5) |s| {
            const rvl = rv_arr[s].toBytes();
            dbg("[BCRAF_INPUT] rv_claim[{}]_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                s, rvl[0], rvl[1], rvl[2], rvl[3], rvl[4], rvl[5], rvl[6], rvl[7],
            });
        }
        const raf_le = raf_claim.toBytes();
        const rafs_le = raf_shift_claim.toBytes();
        dbg("[BCRAF_INPUT] raf_claim_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
            raf_le[0], raf_le[1], raf_le[2], raf_le[3], raf_le[4], raf_le[5], raf_le[6], raf_le[7],
        });
        dbg("[BCRAF_INPUT] raf_shift_claim_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
            rafs_le[0], rafs_le[1], rafs_le[2], rafs_le[3], rafs_le[4], rafs_le[5], rafs_le[6], rafs_le[7],
        });
        // Also print per-stage claims with RAF folded in (like Jolt's claim_per_stage)
        const cps0 = rv1.add(gamma_powers[5].mul(raf_claim));
        const cps2 = rv3.add(gamma_powers[4].mul(raf_shift_claim));
        const cps0l = cps0.toBytes();
        const cps2l = cps2.toBytes();
        dbg("[BCRAF_INPUT] claim_per_stage[0]_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
            cps0l[0], cps0l[1], cps0l[2], cps0l[3], cps0l[4], cps0l[5], cps0l[6], cps0l[7],
        });
        dbg("[BCRAF_INPUT] claim_per_stage[2]_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
            cps2l[0], cps2l[1], cps2l[2], cps2l[3], cps2l[4], cps2l[5], cps2l[6], cps2l[7],
        });
    }

    // Per-stage claims (like Jolt's claim_per_stage)
    // claim_per_stage[s] = rv_claim[s] + RAF_s contribution
    const per_stage = [5]F{
        rv1.add(gamma_powers[5].mul(raf_claim)), // Stage 0: rv1 + gamma^5 * raf
        rv2, // Stage 1: rv2
        rv3.add(gamma_powers[4].mul(raf_shift_claim)), // Stage 2: rv3 + gamma^4 * raf_shift
        rv4, // Stage 3: rv4
        rv5, // Stage 4: rv5
    };

    // Combine: total = Σ_s gamma^s * per_stage[s]
    var result = F.zero();
    for (0..5) |s| {
        const term = gamma_powers[s].mul(per_stage[s]);
        result = result.add(term);
        const ps_le = per_stage[s].toBytes();
        const gp_le = gamma_powers[s].toBytes();
        const tm_le = term.toBytes();
        dbg("[BCRAF_AGG_OC] s={} gp_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}] ps_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}] term_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
            s,
            gp_le[0],
            gp_le[1],
            gp_le[2],
            gp_le[3],
            gp_le[4],
            gp_le[5],
            gp_le[6],
            gp_le[7],
            ps_le[0],
            ps_le[1],
            ps_le[2],
            ps_le[3],
            ps_le[4],
            ps_le[5],
            ps_le[6],
            ps_le[7],
            tm_le[0],
            tm_le[1],
            tm_le[2],
            tm_le[3],
            tm_le[4],
            tm_le[5],
            tm_le[6],
            tm_le[7],
        });
    }
    const res_le = result.toBytes();
    dbg("[BCRAF_AGG_OC] total_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
        res_le[0], res_le[1], res_le[2], res_le[3], res_le[4], res_le[5], res_le[6], res_le[7],
    });

    return .{ .total = result, .per_stage = per_stage };
}

/// Compute the 5 Val polynomials for BytecodeReadRaf from bytecode entries and per-stage gammas.
///
/// Val_s(k) encodes the bytecode entry at address k using the gamma-weighted formula for stage s.
/// Also computes eq tables for Stages 4/5 register addresses (returned for use by debug comparisons).
///
/// Caller must free each val_polys[s] and the two eq_tables when done.
pub fn computeBytecodeValPolys(
    comptime F: type,
    allocator: Allocator,
    bytecode_entries: []const BytecodeEntry,
    bytecode_log_k: usize,
    stage1_gammas: []const F,
    stage2_gammas: []const F,
    stage3_gammas: []const F,
    stage4_gammas: []const F,
    stage5_gammas: []const F,
    r_register_4: []const F,
    r_register_5: []const F,
) !struct { val_polys: [5][]F, eq_table_4: []F, eq_table_5: []F } {
    const bytecode_K: usize = @as(usize, 1) << @intCast(bytecode_log_k);

    // Precompute eq tables for Stages 4 and 5 register addresses
    // r_register_4 and r_register_5 are the address portions from
    // RegistersReadWriteChecking and RegistersValEvaluation opening points
    const REGISTER_COUNT_LOG2: usize = 7; // log2(128 registers: 32 RISC-V + 96 virtual)
    dbg("[STAGE6] r_register_4 (len={}):\n", .{r_register_4.len});
    for (r_register_4, 0..) |rv, i| {
        dbg("  r_register_4[{}] mont_limbs=[0x{x:0>16}, 0x{x:0>16}, 0x{x:0>16}, 0x{x:0>16}]\n", .{ i, rv.limbs[0], rv.limbs[1], rv.limbs[2], rv.limbs[3] });
    }
    dbg("[STAGE6] r_register_5 (len={}):\n", .{r_register_5.len});
    for (r_register_5, 0..) |rv, i| {
        dbg("  r_register_5[{}] mont_limbs=[0x{x:0>16}, 0x{x:0>16}, 0x{x:0>16}, 0x{x:0>16}]\n", .{ i, rv.limbs[0], rv.limbs[1], rv.limbs[2], rv.limbs[3] });
    }
    // Jolt's EqPolynomial::evals uses BIG-ENDIAN bit indexing:
    // r[0] maps to MSB of index, r[n-1] maps to LSB.
    // Our computeEqTable uses LITTLE-ENDIAN: r[0] maps to LSB.
    // Fix: reverse the input array so our LE computation produces BE-indexed results.
    var r_register_4_rev = try allocator.alloc(F, r_register_4.len);
    defer allocator.free(r_register_4_rev);
    for (0..r_register_4.len) |i| {
        r_register_4_rev[i] = r_register_4[r_register_4.len - 1 - i];
    }
    var r_register_5_rev = try allocator.alloc(F, r_register_5.len);
    defer allocator.free(r_register_5_rev);
    for (0..r_register_5.len) |i| {
        r_register_5_rev[i] = r_register_5[r_register_5.len - 1 - i];
    }
    const eq_table_4 = try computeEqTable(F, allocator, r_register_4_rev, REGISTER_COUNT_LOG2);
    const eq_table_5 = try computeEqTable(F, allocator, r_register_5_rev, REGISTER_COUNT_LOG2);
    // Print eq_table_4 entries in LE hex for comparison with Jolt
    dbg("[STAGE6] eq_table_4 (len={}):\n", .{eq_table_4.len});
    for ([_]usize{ 0, 1, 2, 8, 10, 15, 31, 127 }) |idx| {
        if (idx < eq_table_4.len) {
            const vbe = eq_table_4[idx].toBytesBE();
            dbg("  eq4[{}]_LE=[", .{idx});
            for (0..32) |bi| dbg("{x:0>2}", .{vbe[31 - bi]});
            dbg("]\n", .{});
        }
    }
    // Print stage4_gammas in LE hex
    dbg("[STAGE6] stage4_gammas:\n", .{});
    for (0..3) |i| {
        const gbe = stage4_gammas[i].toBytesBE();
        dbg("  gamma4[{}]_LE=[", .{i});
        for (0..32) |bi| dbg("{x:0>2}", .{gbe[31 - bi]});
        dbg("]\n", .{});
    }

    var val_polys: [5][]F = undefined;
    for (0..5) |s| {
        val_polys[s] = try allocator.alloc(F, bytecode_K);
        @memset(val_polys[s], F.zero());
    }

    for (0..bytecode_K) |k| {
        if (k >= bytecode_entries.len) break;
        const entry = bytecode_entries[k];

        // Stage 1: unexpanded_pc + γ₁¹·imm + Σ γ₁^(2+i)·circuit_flag_i
        // CRITICAL: The Imm encoding must match Jolt's vanilla verifier exactly.
        // Jolt's NormalizedOperands.imm is i128, but how it gets there depends
        // on the instruction FORMAT type:
        //   FormatI (I-type): u64 as i128 → zero-extended (always positive)
        //   FormatU (U-type): u64 as i128 → zero-extended (always positive)
        //   FormatJ (J-type): u64 as i128 → zero-extended (always positive)
        //   FormatB (B-type): i128 directly → signed
        //   FormatS (S-type): i64 as i128 → sign-extended (signed)
        //   Virtual (0x0B, 0x2B): u64 as i128 (from emit_i helper)
        // Then Jolt calls from_i128(operands.imm) to get the field element.
        const imm_field: F = blk: {
            const opcode_for_imm = entry.opcode;
            // Signed encoding: must match R1CS witness and Jolt verifier.
            const is_signed_format = (opcode_for_imm == 0x63) or // B-type (branches: FormatB i128)
                (opcode_for_imm == 0x23) or // S-type (stores: FormatS i64)
                (opcode_for_imm == 0x03) or // Load (FormatLoad: i64 sign-extended to i128)
                (opcode_for_imm == 0x22); // VirtualAssert (FormatAssert: signed i64)
            if (is_signed_format) {
                break :blk fieldFromI128(F, @as(i128, entry.imm));
            } else {
                // I-type, U-type, J-type, Virtual: u64 zero-extended to i128.
                break :blk F.fromU64(@as(u64, @bitCast(entry.imm)));
            }
        };
        var val1 = F.fromU64(entry.address);
        val1 = val1.add(stage1_gammas[1].mul(imm_field));
        for (0..14) |i| {
            if (entry.circuit_flags[i]) {
                val1 = val1.add(stage1_gammas[2 + i]);
            }
        }
        val_polys[0][k] = val1;

        // Stage 2: γ₂⁰·jump + γ₂¹·branch + γ₂²·write_lookup_to_rd + γ₂³·virtual_instruction
        var val2 = F.zero();
        if (entry.circuit_flags[@intFromEnum(CircuitFlags.Jump)]) {
            val2 = val2.add(stage2_gammas[0]);
        }
        if (entry.instruction_flags[@intFromEnum(InstructionFlags.Branch)]) {
            val2 = val2.add(stage2_gammas[1]);
        }
        if (entry.circuit_flags[@intFromEnum(CircuitFlags.WriteLookupOutputToRD)]) {
            val2 = val2.add(stage2_gammas[2]);
        }
        if (entry.circuit_flags[@intFromEnum(CircuitFlags.VirtualInstruction)]) {
            val2 = val2.add(stage2_gammas[3]);
        }
        val_polys[1][k] = val2;

        // Stage 3: γ₃⁰·imm + γ₃¹·unexpanded_pc + γ₃²·L_is_rs1 + γ₃³·L_is_pc
        //         + γ₃⁴·R_is_rs2 + γ₃⁵·R_is_imm + γ₃⁶·is_noop
        //         + γ₃⁷·virtual_instruction + γ₃⁸·is_first_in_sequence
        var val3 = imm_field;
        val3 = val3.add(stage3_gammas[1].mul(F.fromU64(entry.address)));
        if (entry.instruction_flags[@intFromEnum(InstructionFlags.LeftOperandIsRs1Value)]) {
            val3 = val3.add(stage3_gammas[2]);
        }
        if (entry.instruction_flags[@intFromEnum(InstructionFlags.LeftOperandIsPC)]) {
            val3 = val3.add(stage3_gammas[3]);
        }
        if (entry.instruction_flags[@intFromEnum(InstructionFlags.RightOperandIsRs2Value)]) {
            val3 = val3.add(stage3_gammas[4]);
        }
        if (entry.instruction_flags[@intFromEnum(InstructionFlags.RightOperandIsImm)]) {
            val3 = val3.add(stage3_gammas[5]);
        }
        if (entry.instruction_flags[@intFromEnum(InstructionFlags.IsNoop)]) {
            val3 = val3.add(stage3_gammas[6]);
        }
        if (entry.circuit_flags[@intFromEnum(CircuitFlags.VirtualInstruction)]) {
            val3 = val3.add(stage3_gammas[7]);
        }
        if (entry.is_first_in_sequence) {
            val3 = val3.add(stage3_gammas[8]);
        }
        val_polys[2][k] = val3;

        // Stage 4: γ₄⁰·eq(rd, r_reg4) + γ₄¹·eq(rs1, r_reg4) + γ₄²·eq(rs2, r_reg4)
        const REGISTER_COUNT: usize = 128; // 32 RISC-V + 96 virtual
        var val4 = F.zero();
        if (entry.rd < REGISTER_COUNT) {
            val4 = val4.add(stage4_gammas[0].mul(eq_table_4[entry.rd]));
        }
        if (entry.rs1 < REGISTER_COUNT) {
            val4 = val4.add(stage4_gammas[1].mul(eq_table_4[entry.rs1]));
        }
        if (entry.rs2 < REGISTER_COUNT) {
            val4 = val4.add(stage4_gammas[2].mul(eq_table_4[entry.rs2]));
        }
        val_polys[3][k] = val4;

        // Stage 5: eq(rd, r_reg5) + γ₅¹·!is_interleaved + Σ γ₅^(2+i)·table_flag_i
        var val5 = F.zero();
        if (entry.rd < REGISTER_COUNT) {
            val5 = val5.add(eq_table_5[entry.rd]);
        }
        if (!entry.is_interleaved) {
            val5 = val5.add(stage5_gammas[1]);
        }
        if (entry.lookup_table_index < 40) {
            val5 = val5.add(stage5_gammas[2 + @as(usize, entry.lookup_table_index)]);
        }
        val_polys[4][k] = val5;
    }

    // Debug: Print Val poly entries for comparison with Jolt verifier
    if (comptime debug_verbose) {
        dbg("[STAGE6] Val[3] (Stage 4/RegistersRWC) entries:\n", .{});
        for (0..bytecode_K) |k| {
            const vbe = val_polys[3][k].toBytesBE();
            dbg("  Val[3][{}]_LE=[", .{k});
            for (0..32) |bi| dbg("{x:0>2}", .{vbe[31 - bi]});
            dbg("]\n", .{});
        }
    }
    if (debug_verbose) {
        for ([_]usize{ 0, 1, 2, 4 }) |s| {
            for (0..bytecode_K) |k| {
                const vbe = val_polys[s][k].toBytesBE();
                dbg("  Val[{}][{}]_LE=[", .{ s, k });
                for (0..32) |bi| dbg("{x:0>2}", .{vbe[31 - bi]});
                dbg("]\n", .{});
            }
        }
    }

    // Debug: Dump bytecode entries
    if (comptime debug_verbose) {
        dbg("[STAGE6] Bytecode entries (ALL k=0..{}):\n", .{bytecode_K});
        for (0..@min(bytecode_K, 64)) |k| {
            if (k >= bytecode_entries.len) break;
            const entry = bytecode_entries[k];
            dbg("[STAGE6] entry[{}]: addr=0x{x:0>8} rd={} rs1={} rs2={} imm={} cf=[", .{ k, entry.address, entry.rd, entry.rs1, entry.rs2, entry.imm });
            for (0..14) |i| {
                if (i > 0) dbg(",", .{});
                if (entry.circuit_flags[i]) dbg("1", .{}) else dbg("0", .{});
            }
            dbg("] if=[", .{});
            for (0..7) |i| {
                if (i > 0) dbg(",", .{});
                if (entry.instruction_flags[i]) dbg("1", .{}) else dbg("0", .{});
            }
            dbg("] lt={} interleaved={}\n", .{ entry.lookup_table_index, @intFromBool(entry.is_interleaved) });
        }
    }

    return .{ .val_polys = val_polys, .eq_table_4 = eq_table_4, .eq_table_5 = eq_table_5 };
}
