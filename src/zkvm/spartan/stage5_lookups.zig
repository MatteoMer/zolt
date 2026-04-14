//! LookupsReadRaf Prover (Instance 2 of Stage 5 Batched Sumcheck)
//!
//! Extracted from stage5_prover.zig to encapsulate the LookupsReadRaf
//! sumcheck logic. This instance proves:
//!   Sum_{j,k} eq(j, r_reduction) * ra(k, j) * combined(k, j) = lookups_input
//!
//! The sumcheck has 2 phases:
//!   - Address rounds (0..127): sparse 128-bit address variable binding using
//!     prefix-suffix polynomial decomposition and RAF (Read-Address Flag)
//!   - Cycle rounds (128..135): dense cycle variable binding with degree-10
//!     polynomials (product of 9 linear factors times eq)
//!
//! A rematerialization step at round 128 transitions between the two phases.

const std = @import("std");

const zkvm_debug = @import("../debug.zig");
const is_wasm = zkvm_debug.is_wasm;
const dbg = zkvm_debug.dbg;
const debug_verbose = zkvm_debug.verbose;
const platformGetenv = zkvm_debug.getenv;

const Allocator = std.mem.Allocator;
const ThreadPool = @import("zolt_pool").ThreadPool;
const pool_helpers = @import("zolt_pool").helpers;
const GpuPolyOps = @import("zolt_arith").gpu.GpuPolyOps;

const poly_mod = @import("zolt_arith").poly;
const UniPoly = poly_mod.UniPoly;
const UnreducedProductAccum = @import("zolt_arith").field.UnreducedProductAccum;

const stage5_inst = @import("stage5_instances.zig");
const stage5_prover_mod = @import("stage5_prover.zig");
const LOOKUPS_LOG_K = stage5_prover_mod.LOOKUPS_LOG_K;
const getBit128 = stage5_prover_mod.getBit128;

const lookup_table_mod = @import("../lookup_table/mod.zig");
const AllSuffixPolys = lookup_table_mod.AllSuffixPolys;
const PrefixCheckpointsState = lookup_table_mod.PrefixCheckpointsState;
const proverMsgReadChecking = lookup_table_mod.proverMsgReadChecking;
const RafDecomposition = lookup_table_mod.RafDecomposition;
const initQRaf = lookup_table_mod.initQRaf;
const proverMsgRaf = lookup_table_mod.proverMsgRaf;
const ExpandingTable = lookup_table_mod.ExpandingTable;
const condenseUEvals = lookup_table_mod.condenseUEvals;
const computeTableValuesAtRAddress = lookup_table_mod.computeTableValuesAtRAddress;
const NUM_TABLES = lookup_table_mod.NUM_TABLES;

const evaluateLeftOperand = stage5_inst.evaluateLeftOperand;
const evaluateRightOperand = stage5_inst.evaluateRightOperand;
const evaluateIdentity = stage5_inst.evaluateIdentity;
const computeEqPolynomial = stage5_inst.computeEqPolynomial;

/// Result of the opening claims extraction after the sumcheck completes.
pub fn LookupsOpeningClaims(comptime F: type) type {
    return struct {
        ra_chunks: []F,
        table_flags: []F,
        raf_flag: F,
    };
}

/// LookupsReadRaf prover for Instance 2 of Stage 5.
///
/// Encapsulates all state and methods for the 136-round LookupsReadRaf
/// sumcheck, using prefix-suffix decomposition for address rounds (0-127)
/// and dense degree-10 product sumcheck for cycle rounds (128-135).
pub fn LookupsReadRafProver(comptime F: type) type {
    return struct {
        const Self = @This();
        const Inst = stage5_inst.Helpers(F);

        const MAX_RA_CHUNKS = 8;
        const MAX_LOOKUP_TABLES: usize = 40;
        const MAX_SPLIT_EQ_SIZE = 1 << 10;

        // ---- Allocator / thread pool / GPU ----
        allocator: Allocator,
        thread_pool: ?*ThreadPool,
        gpu_ops: ?*GpuPolyOps,

        // ---- Gamma values ----
        gamma_raf: F,
        gamma_raf2: F,

        // ---- Current claim (tracks Instance 2 polynomial chain) ----
        current_claim: F,

        // ---- Core polynomial tables (size T) ----
        lookups_eq_evals: []F,
        lookups_combined_vals: []F,
        lookups_indices_lo: []u64,
        lookups_indices_hi: []u64,
        lookup_indices_u128: []u128,
        cycle_table_indices: []i8,
        cycle_is_identity_path: []bool,
        is_interleaved_operands: []bool,
        T: usize,

        // ---- RA chunk weights (allocated lazily at rematerialization) ----
        ra_chunk_weights: [MAX_RA_CHUNKS][]F,
        ra_chunk_weights_allocated: bool,
        ra_num_chunks: usize,
        lookups_ra_virtual_log_k_chunk: usize,

        // ---- Table values at r_address (populated during rematerialization) ----
        stored_table_values: [MAX_LOOKUP_TABLES]F,

        // ---- Prefix-suffix decomposition state ----
        suffix_polys: AllSuffixPolys(F),
        prefix_checkpoints: PrefixCheckpointsState(F),
        left_raf: RafDecomposition(F),
        right_raf: RafDecomposition(F),
        identity_raf: RafDecomposition(F),
        expanding_tables: [16]ExpandingTable(F),
        tables_initialized: usize,
        num_phases: usize,
        log_m: usize,
        initial_m: usize,
        current_phase: usize,

        // ---- Lookup indices by table (for parallel initPhase) ----
        lookup_indices_by_table: [NUM_TABLES][]usize,
        lookup_indices_by_table_count: usize,

        // ---- Cycle round state ----
        lookups_current_scalar: F,
        split_eq_E_in: [MAX_SPLIT_EQ_SIZE]F,
        split_eq_E_out: [MAX_SPLIT_EQ_SIZE]F,
        split_eq_m_in: usize,
        split_eq_E_in_len: usize,
        split_eq_E_out_len: usize,
        split_eq_initialized: bool,

        // ---- Configuration ----
        n_cycle_vars: usize,
        r_reduction: []const F,

        // ---- Debug-only fields ----
        lookups_ra_weights: []F, // Only used for debug verification
        bf_weights: []F, // Only used for brute-force verification

        /// Initialize the LookupsReadRaf prover with trace-derived data.
        ///
        /// This performs all the initialization that was previously inline in
        /// generateStage5ProofWithTrace: building eq tables, combined values,
        /// suffix polys, RAF decompositions, expanding tables, etc.
        pub fn init(
            allocator: Allocator,
            thread_pool: ?*ThreadPool,
            gpu_ops: ?*GpuPolyOps,
            gamma_raf: F,
            lookups_input: F,
            n_cycle_vars: usize,
            lookups_ra_virtual_log_k_chunk: usize,
            r_reduction: []const F,
            // Pre-populated arrays (ownership transferred to this struct)
            lookups_eq_evals: []F,
            lookups_combined_vals: []F,
            lookups_indices_lo: []u64,
            lookups_indices_hi: []u64,
            lookup_indices_u128: []u128,
            cycle_table_indices: []i8,
            cycle_is_identity_path: []bool,
            is_interleaved_operands: []bool,
            lookup_indices_by_table: [NUM_TABLES][]usize,
        ) !Self {
            const T = @as(usize, 1) << @intCast(n_cycle_vars);
            const gamma_raf2 = gamma_raf.mul(gamma_raf);
            const ra_num_chunks = LOOKUPS_LOG_K / lookups_ra_virtual_log_k_chunk;
            std.debug.assert(ra_num_chunks <= MAX_RA_CHUNKS);

            // Jolt uses 16 phases for log_T < 24 (traces < 2^24), 8 phases otherwise
            const INSTRUCTION_PHASES_THRESHOLD_LOG_T = 24;
            const log_T = std.math.log2_int(usize, T);
            const num_phases: usize = if (log_T < INSTRUCTION_PHASES_THRESHOLD_LOG_T) 16 else 8;
            const log_m = LOOKUPS_LOG_K / num_phases;
            const initial_m = @as(usize, 1) << @intCast(log_m);

            // Initialize suffix polynomials for phase 0
            var suffix_polys = AllSuffixPolys(F).init(allocator);
            errdefer suffix_polys.deinit();
            const prefix_checkpoints = PrefixCheckpointsState(F).init();

            // Initialize RAF decompositions
            var left_raf = try RafDecomposition(F).init(allocator, initial_m, log_m, LOOKUPS_LOG_K, .LeftOperand);
            errdefer left_raf.deinit();
            var right_raf = try RafDecomposition(F).init(allocator, initial_m, log_m, LOOKUPS_LOG_K, .RightOperand);
            errdefer right_raf.deinit();
            var identity_raf = try RafDecomposition(F).init(allocator, initial_m, log_m, LOOKUPS_LOG_K, .Identity);
            errdefer identity_raf.deinit();

            // Initialize Q polynomials for read-checking AND RAF Q accumulators concurrently
            if (thread_pool) |tp| {
                const SuffixInitCtx = struct {
                    polys: *AllSuffixPolys(F),
                    phases: usize,
                    eq: []const F,
                    indices: []const u128,
                    table_indices: []const i8,
                    pool: *ThreadPool,
                    alloc_inner: Allocator,
                    ibt: *const [NUM_TABLES][]usize,
                };
                const suffix_ctx = SuffixInitCtx{
                    .polys = &suffix_polys,
                    .phases = num_phases,
                    .eq = lookups_eq_evals,
                    .indices = lookup_indices_u128,
                    .table_indices = cycle_table_indices,
                    .pool = tp,
                    .alloc_inner = allocator,
                    .ibt = &lookup_indices_by_table,
                };
                const RafInitCtx = struct {
                    left_raf: *RafDecomposition(F),
                    right_raf: *RafDecomposition(F),
                    identity_raf: *RafDecomposition(F),
                    eq: []const F,
                    indices: []const u128,
                    is_interleaved: []const bool,
                    pool: *ThreadPool,
                    alloc_inner: Allocator,
                };
                const raf_ctx = RafInitCtx{
                    .left_raf = &left_raf,
                    .right_raf = &right_raf,
                    .identity_raf = &identity_raf,
                    .eq = lookups_eq_evals,
                    .indices = lookup_indices_u128,
                    .is_interleaved = is_interleaved_operands,
                    .pool = tp,
                    .alloc_inner = allocator,
                };
                _ = tp.join(
                    void,
                    void,
                    suffix_ctx,
                    struct {
                        fn f(c: SuffixInitCtx) void {
                            c.polys.initPhase(0, c.phases, c.eq, c.indices, c.table_indices, c.pool, c.alloc_inner, c.ibt) catch unreachable;
                        }
                    }.f,
                    raf_ctx,
                    struct {
                        fn f(c: RafInitCtx) void {
                            initQRaf(F, c.left_raf, c.right_raf, c.identity_raf, c.eq, c.indices, c.is_interleaved, c.pool, c.alloc_inner) catch unreachable;
                        }
                    }.f,
                );
            } else {
                try suffix_polys.initPhase(0, num_phases, lookups_eq_evals, lookup_indices_u128, cycle_table_indices, null, allocator, null);
                try initQRaf(F, &left_raf, &right_raf, &identity_raf, lookups_eq_evals, lookup_indices_u128, is_interleaved_operands, null, allocator);
            }

            // Materialize prefix MLE tables for phase 0
            left_raf.initPrefix();
            right_raf.initPrefix();
            identity_raf.initPrefix();

            // Initialize expanding tables for each phase
            var expanding_tables: [16]ExpandingTable(F) = undefined;
            var tables_initialized: usize = 0;
            errdefer {
                for (0..tables_initialized) |phase_idx| {
                    expanding_tables[phase_idx].deinit();
                }
            }
            for (0..num_phases) |phase_idx| {
                expanding_tables[phase_idx] = try ExpandingTable(F).init(allocator, initial_m);
                tables_initialized = phase_idx + 1;
            }

            // Reset phase 0's expanding table to 1
            expanding_tables[0].reset(F.one());

            // Debug-only allocations
            var lookups_ra_weights: []F = &[_]F{};
            var bf_weights_alloc: []F = &[_]F{};
            if (comptime debug_verbose) {
                lookups_ra_weights = try allocator.alloc(F, T);
                @memset(lookups_ra_weights, F.one());
                bf_weights_alloc = try allocator.alloc(F, T);
                for (0..T) |j| {
                    bf_weights_alloc[j] = lookups_eq_evals[j].mul(lookups_combined_vals[j]);
                }
            }

            return Self{
                .allocator = allocator,
                .thread_pool = thread_pool,
                .gpu_ops = gpu_ops,
                .gamma_raf = gamma_raf,
                .gamma_raf2 = gamma_raf2,
                .current_claim = lookups_input,
                .lookups_eq_evals = lookups_eq_evals,
                .lookups_combined_vals = lookups_combined_vals,
                .lookups_indices_lo = lookups_indices_lo,
                .lookups_indices_hi = lookups_indices_hi,
                .lookup_indices_u128 = lookup_indices_u128,
                .cycle_table_indices = cycle_table_indices,
                .cycle_is_identity_path = cycle_is_identity_path,
                .is_interleaved_operands = is_interleaved_operands,
                .T = T,
                .ra_chunk_weights = undefined,
                .ra_chunk_weights_allocated = false,
                .ra_num_chunks = ra_num_chunks,
                .lookups_ra_virtual_log_k_chunk = lookups_ra_virtual_log_k_chunk,
                .stored_table_values = [_]F{F.zero()} ** MAX_LOOKUP_TABLES,
                .suffix_polys = suffix_polys,
                .prefix_checkpoints = prefix_checkpoints,
                .left_raf = left_raf,
                .right_raf = right_raf,
                .identity_raf = identity_raf,
                .expanding_tables = expanding_tables,
                .tables_initialized = tables_initialized,
                .num_phases = num_phases,
                .log_m = log_m,
                .initial_m = initial_m,
                .current_phase = 0,
                .lookup_indices_by_table = lookup_indices_by_table,
                .lookup_indices_by_table_count = NUM_TABLES,
                .lookups_current_scalar = F.one(),
                .split_eq_E_in = undefined,
                .split_eq_E_out = undefined,
                .split_eq_m_in = 0,
                .split_eq_E_in_len = 0,
                .split_eq_E_out_len = 0,
                .split_eq_initialized = false,
                .n_cycle_vars = n_cycle_vars,
                .r_reduction = r_reduction,
                .lookups_ra_weights = lookups_ra_weights,
                .bf_weights = bf_weights_alloc,
            };
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.lookups_eq_evals);
            self.allocator.free(self.lookups_combined_vals);
            self.allocator.free(self.lookups_indices_lo);
            self.allocator.free(self.lookups_indices_hi);
            self.allocator.free(self.lookup_indices_u128);
            self.allocator.free(self.cycle_table_indices);
            self.allocator.free(self.cycle_is_identity_path);
            self.allocator.free(self.is_interleaved_operands);
            if (self.ra_chunk_weights_allocated) {
                for (0..self.ra_num_chunks) |chunk_idx| {
                    self.allocator.free(self.ra_chunk_weights[chunk_idx]);
                }
            }
            self.suffix_polys.deinit();
            self.left_raf.deinit();
            self.right_raf.deinit();
            self.identity_raf.deinit();
            for (0..self.num_phases) |phase_idx| {
                self.expanding_tables[phase_idx].deinit();
            }
            for (0..self.lookup_indices_by_table_count) |t| {
                self.allocator.free(self.lookup_indices_by_table[t]);
            }
            if (comptime debug_verbose) {
                if (self.lookups_ra_weights.len > 0) self.allocator.free(self.lookups_ra_weights);
                if (self.bf_weights.len > 0) self.allocator.free(self.bf_weights);
            }
        }

        /// Compute the address round polynomial for Instance 2.
        /// Returns [eval_0, eval_1, eval_2] where eval_1 = claim - eval_0.
        /// The polynomial is degree-2 (from prefix-suffix decomposition + RAF).
        /// `challenges` is the full challenges array (needed for r_x on odd rounds).
        pub fn computeAddressRoundPoly(self: *Self, round: usize, challenges: []const F) [3]F {
            // Compute read-checking and RAF contributions concurrently via tp.join
            const RcCtx = struct {
                round: usize,
                sp: *AllSuffixPolys(F),
                pc: *PrefixCheckpointsState(F),
                rx: ?F,
                tp: ?*ThreadPool,
            };
            const RafCtx = struct {
                left: *RafDecomposition(F),
                right: *RafDecomposition(F),
                identity: *RafDecomposition(F),
                g1: F,
                g2: F,
                tp: ?*ThreadPool,
            };
            const r_x_val: ?F = if (round % 2 == 1) challenges[round - 1] else null;
            const rc_join_ctx = RcCtx{ .round = round, .sp = &self.suffix_polys, .pc = &self.prefix_checkpoints, .rx = r_x_val, .tp = self.thread_pool };
            const raf_join_ctx = RafCtx{ .left = &self.left_raf, .right = &self.right_raf, .identity = &self.identity_raf, .g1 = self.gamma_raf, .g2 = self.gamma_raf2, .tp = self.thread_pool };
            const read_checking_evals, const raf_evals = if (self.thread_pool) |tp|
                tp.join([2]F, [2]F, rc_join_ctx, struct {
                    fn f(c: RcCtx) [2]F {
                        return proverMsgReadChecking(F, c.round, c.sp, c.pc, c.rx, c.tp);
                    }
                }.f, raf_join_ctx, struct {
                    fn f(c: RafCtx) [2]F {
                        return proverMsgRaf(F, c.left, c.right, c.identity, c.g1, c.g2, c.tp);
                    }
                }.f)
            else
                .{ proverMsgReadChecking(F, round, &self.suffix_polys, &self.prefix_checkpoints, r_x_val, self.thread_pool), proverMsgRaf(F, &self.left_raf, &self.right_raf, &self.identity_raf, self.gamma_raf, self.gamma_raf2, self.thread_pool) };

            // Combined: read_checking + raf
            const eval_0 = read_checking_evals[0].add(raf_evals[0]);
            const eval_2 = read_checking_evals[1].add(raf_evals[1]);
            const eval_1 = self.current_claim.sub(eval_0);

            if (comptime debug_verbose) if (round < 5 or round == 127) {
                const print = std.debug.print;
                print("[ZOLT INST2 R{}] previous_claim = {any}\n", .{ round, self.current_claim.toBytes()[0..16].* });
                print("[ZOLT INST2 R{}] eval_at_0 = {any}\n", .{ round, eval_0.toBytes()[0..16].* });
                print("[ZOLT INST2 R{}] eval_at_1 = {any}\n", .{ round, eval_1.toBytes()[0..16].* });
                print("[ZOLT INST2 R{}] eval_at_2 = {any}\n", .{ round, eval_2.toBytes()[0..16].* });
            };

            // DIAGNOSTIC: at round 0, brute-force compute eval_0 and eval_2 from
            // the underlying cycle data and compare with the decomposition-based
            // values. Round 0 is before any binding, so the polynomial is just
            // `Σ_j u[j] * (combined_vals[j] + γ*left_op + γ²*(identity+right))` for
            // cycles whose lookup index has MSB=0 (eval_0) or MSB=1 (eval_1).
            // eval_2 is extrapolated as 2*eval_1 - eval_0 for the linear-in-X
            // part plus degree-2 correction (because f(X,rest) = ra(X,rest) *
            // poly(X,rest) is degree 2 in X — eq(X, msb) contributes one X).
            if (comptime !is_wasm) {
                if (round == 0 and platformGetenv("ZOLT_S5_BRUTE") != null) {
                    var bf_eval_0 = F.zero();
                    var bf_eval_1 = F.zero();
                    // Break down by component to isolate the drift source.
                    var bf_rv_total = F.zero();
                    var bf_left_total = F.zero();
                    var bf_right_total = F.zero();
                    // Per-table rv_total to identify which table is buggy
                    var bf_rv_per_table: [40]F = .{F.zero()} ** 40;
                    const T_local = self.T;
                    for (0..T_local) |j| {
                        const lookup_idx = self.lookup_indices_u128[j];
                        const msb: u1 = @intCast((lookup_idx >> 127) & 1);
                        const t_idx = self.cycle_table_indices[j];
                        var table_val = F.zero();
                        if (t_idx >= 0) {
                            const ti: usize = @intCast(t_idx);
                            const TableMod_local = @import("../lookup_table/mod.zig").LookupTable(F, 64);
                            const entry = TableMod_local.materializeTableEntry(ti, lookup_idx);
                            table_val = F.fromU64(entry);
                        }
                        var left_val = F.zero();
                        var right_val = F.zero();
                        if (self.is_interleaved_operands[j]) {
                            const left_bits: u64 = blk_left: {
                                var x_bits: u128 = (lookup_idx >> 1) & 0x5555_5555_5555_5555_5555_5555_5555_5555;
                                x_bits = (x_bits | (x_bits >> 1)) & 0x3333_3333_3333_3333_3333_3333_3333_3333;
                                x_bits = (x_bits | (x_bits >> 2)) & 0x0F0F_0F0F_0F0F_0F0F_0F0F_0F0F_0F0F_0F0F;
                                x_bits = (x_bits | (x_bits >> 4)) & 0x00FF_00FF_00FF_00FF_00FF_00FF_00FF_00FF;
                                x_bits = (x_bits | (x_bits >> 8)) & 0x0000_FFFF_0000_FFFF_0000_FFFF_0000_FFFF;
                                x_bits = (x_bits | (x_bits >> 16)) & 0x0000_0000_FFFF_FFFF_0000_0000_FFFF_FFFF;
                                x_bits = (x_bits | (x_bits >> 32)) & 0x0000_0000_0000_0000_FFFF_FFFF_FFFF_FFFF;
                                break :blk_left @truncate(x_bits);
                            };
                            const right_bits: u64 = blk_right: {
                                var y_bits: u128 = lookup_idx & 0x5555_5555_5555_5555_5555_5555_5555_5555;
                                y_bits = (y_bits | (y_bits >> 1)) & 0x3333_3333_3333_3333_3333_3333_3333_3333;
                                y_bits = (y_bits | (y_bits >> 2)) & 0x0F0F_0F0F_0F0F_0F0F_0F0F_0F0F_0F0F_0F0F;
                                y_bits = (y_bits | (y_bits >> 4)) & 0x00FF_00FF_00FF_00FF_00FF_00FF_00FF_00FF;
                                y_bits = (y_bits | (y_bits >> 8)) & 0x0000_FFFF_0000_FFFF_0000_FFFF_0000_FFFF;
                                y_bits = (y_bits | (y_bits >> 16)) & 0x0000_0000_FFFF_FFFF_0000_0000_FFFF_FFFF;
                                y_bits = (y_bits | (y_bits >> 32)) & 0x0000_0000_0000_0000_FFFF_FFFF_FFFF_FFFF;
                                break :blk_right @truncate(y_bits);
                            };
                            left_val = F.fromU64(left_bits);
                            right_val = F.fromU64(right_bits);
                        } else {
                            // Identity path: left_op = 0, right_op = full lookup_idx (u128).
                            left_val = F.zero();
                            right_val = F.fromU128(lookup_idx);
                        }
                        const raf_val = self.gamma_raf.mul(left_val).add(self.gamma_raf2.mul(right_val));
                        const u_val = self.lookups_eq_evals[j];
                        const contribution = u_val.mul(table_val.add(raf_val));
                        if (msb == 0) {
                            bf_eval_0 = bf_eval_0.add(contribution);
                        } else {
                            bf_eval_1 = bf_eval_1.add(contribution);
                        }
                        // Component breakdown (summed over ALL cycles, no msb split):
                        bf_rv_total = bf_rv_total.add(u_val.mul(table_val));
                        bf_left_total = bf_left_total.add(u_val.mul(left_val));
                        bf_right_total = bf_right_total.add(u_val.mul(right_val));
                        if (t_idx >= 0 and @as(usize, @intCast(t_idx)) < 40) {
                            const ti_local: usize = @intCast(t_idx);
                            bf_rv_per_table[ti_local] = bf_rv_per_table[ti_local].add(u_val.mul(table_val));
                        }
                        // Diff materializeTableEntry vs the "true" lookup_output (rd_value from the step).
                        // We only have direct access to lookup_indices_u128 here, so we use
                        // materializeTableEntry as the reference and the `lookups_combined_vals`
                        // initial values as the prover's witness value.
                    }
                    const bf_total = bf_eval_0.add(bf_eval_1);
                    const bf_from_components = bf_rv_total
                        .add(self.gamma_raf.mul(bf_left_total))
                        .add(self.gamma_raf2.mul(bf_right_total));
                    const print = std.debug.print;
                    print("[S5_BRUTE R0] prover_eval_0  = ", .{});
                    for (0..32) |bi| print("{x:0>2}", .{eval_0.toBytesBE()[31 - bi]});
                    print("\n[S5_BRUTE R0] brute_eval_0   = ", .{});
                    for (0..32) |bi| print("{x:0>2}", .{bf_eval_0.toBytesBE()[31 - bi]});
                    print("\n[S5_BRUTE R0] prover_eval_1  = ", .{});
                    for (0..32) |bi| print("{x:0>2}", .{eval_1.toBytesBE()[31 - bi]});
                    print("\n[S5_BRUTE R0] brute_eval_1   = ", .{});
                    for (0..32) |bi| print("{x:0>2}", .{bf_eval_1.toBytesBE()[31 - bi]});
                    print("\n[S5_BRUTE R0] current_claim  = ", .{});
                    for (0..32) |bi| print("{x:0>2}", .{self.current_claim.toBytesBE()[31 - bi]});
                    print("\n[S5_BRUTE R0] brute_total    = ", .{});
                    for (0..32) |bi| print("{x:0>2}", .{bf_total.toBytesBE()[31 - bi]});
                    print("\n[S5_BRUTE R0] bf_from_comps  = ", .{});
                    for (0..32) |bi| print("{x:0>2}", .{bf_from_components.toBytesBE()[31 - bi]});
                    print("\n[S5_BRUTE R0] bf_rv_total    = ", .{});
                    for (0..32) |bi| print("{x:0>2}", .{bf_rv_total.toBytesBE()[31 - bi]});
                    print("\n[S5_BRUTE R0] bf_rv_per_table (non-zero):\n", .{});
                    for (0..40) |ti_print| {
                        if (!bf_rv_per_table[ti_print].eql(F.zero())) {
                            print("  table[{d:2}] = ", .{ti_print});
                            for (0..32) |bi_pt| print("{x:0>2}", .{bf_rv_per_table[ti_print].toBytesBE()[31 - bi_pt]});
                            print("\n", .{});
                        }
                    }
                    print("\n[S5_BRUTE R0] bf_left_total  = ", .{});
                    for (0..32) |bi| print("{x:0>2}", .{bf_left_total.toBytesBE()[31 - bi]});
                    print("\n[S5_BRUTE R0] bf_right_total = ", .{});
                    for (0..32) |bi| print("{x:0>2}", .{bf_right_total.toBytesBE()[31 - bi]});
                    print("\n[S5_BRUTE R0] eval_0_match   = {}\n", .{eval_0.eql(bf_eval_0)});
                    print("[S5_BRUTE R0] eval_1_match   = {}\n", .{eval_1.eql(bf_eval_1)});
                    print("[S5_BRUTE R0] claim_match    = {}\n", .{self.current_claim.eql(bf_total)});
                }
            }

            return .{ eval_0, eval_1, eval_2 };
        }

        /// Bind an address round challenge and update internal state.
        /// This binds suffix polys, RAF decompositions, expanding tables,
        /// updates prefix checkpoints, and handles phase transitions.
        /// NOTE: The caller must update current_claim (via setClaim) BEFORE calling this,
        /// using the result of evaluating the Instance 2 polynomial at the challenge.
        pub fn bindAddressChallenge(self: *Self, challenge: F, round: usize, challenges: []const F) !void {
            // Bind challenge to suffix polys, RAF decompositions, and expanding table
            if (self.thread_pool) |tp| {
                const BindAllCtx = struct {
                    sp: *AllSuffixPolys(F),
                    left: *RafDecomposition(F),
                    right: *RafDecomposition(F),
                    ident: *RafDecomposition(F),
                    exp_table: *ExpandingTable(F),
                    chal: F,
                    pool: *ThreadPool,
                };
                const bind_ctx = BindAllCtx{
                    .sp = &self.suffix_polys,
                    .left = &self.left_raf,
                    .right = &self.right_raf,
                    .ident = &self.identity_raf,
                    .exp_table = &self.expanding_tables[self.current_phase],
                    .chal = challenge,
                    .pool = tp,
                };
                tp.parallelForForce(5, bind_ctx, struct {
                    fn f(c: BindAllCtx, task_idx: usize) void {
                        switch (task_idx) {
                            0 => c.sp.bindAllParallel(c.chal, c.pool),
                            1 => c.left.bind(c.chal),
                            2 => c.right.bind(c.chal),
                            3 => c.ident.bind(c.chal),
                            4 => c.exp_table.update(c.chal),
                            else => unreachable,
                        }
                    }
                }.f);
            } else {
                self.suffix_polys.bindAll(challenge);
                self.left_raf.bind(challenge);
                self.right_raf.bind(challenge);
                self.identity_raf.bind(challenge);
                self.expanding_tables[self.current_phase].update(challenge);
            }

            // Update prefix checkpoints every 2 rounds (after binding X and Y)
            const round_in_phase = round % self.log_m;
            if (round_in_phase % 2 == 1) {
                const checkpoint_r_x = challenges[round - 1];
                const r_y = challenge;
                const suffix_len = LOOKUPS_LOG_K - (self.current_phase + 1) * self.log_m;
                self.prefix_checkpoints.update(checkpoint_r_x, r_y, round, suffix_len);
            }

            // Debug: update bf_weights and ra_weights
            if (comptime debug_verbose) {
                const bit_index = LOOKUPS_LOG_K - 1 - round;
                const one_minus_r = F.one().sub(challenge);
                for (0..self.T) |j| {
                    const bit = getBit128(self.lookups_indices_lo[j], self.lookups_indices_hi[j], bit_index);
                    const factor = if (bit == 0) one_minus_r else challenge;
                    self.lookups_ra_weights[j] = self.lookups_ra_weights[j].mul(factor);
                    self.bf_weights[j] = self.bf_weights[j].mul(factor);
                }
            }

            // Check for phase transition (every log_m rounds)
            if ((round + 1) % self.log_m == 0 and round + 1 < LOOKUPS_LOG_K) {
                const prev_phase = self.current_phase;
                self.current_phase += 1;

                // Condense u_evals using expanding table from previous phase
                condenseUEvals(F, self.lookups_eq_evals, &self.expanding_tables[prev_phase], self.lookup_indices_u128, self.current_phase, self.num_phases, self.thread_pool);

                // Save prefix checkpoints before resetting RAF decompositions
                self.left_raf.updateCheckpoint();
                self.right_raf.updateCheckpoint();
                self.identity_raf.updateCheckpoint();

                // Reset RAF decompositions for new phase
                self.left_raf.resetForPhase(self.current_phase, self.initial_m);
                self.right_raf.resetForPhase(self.current_phase, self.initial_m);
                self.identity_raf.resetForPhase(self.current_phase, self.initial_m);

                // Run initPhase and initQRaf concurrently
                if (self.thread_pool) |tp| {
                    const SuffixInitCtx = struct {
                        polys: *AllSuffixPolys(F),
                        phase: usize,
                        phases: usize,
                        eq: []const F,
                        indices: []const u128,
                        table_indices: []const i8,
                        pool: *ThreadPool,
                        alloc_inner: Allocator,
                        ibt: *const [NUM_TABLES][]usize,
                    };
                    const suffix_ctx = SuffixInitCtx{
                        .polys = &self.suffix_polys,
                        .phase = self.current_phase,
                        .phases = self.num_phases,
                        .eq = self.lookups_eq_evals,
                        .indices = self.lookup_indices_u128,
                        .table_indices = self.cycle_table_indices,
                        .pool = tp,
                        .alloc_inner = self.allocator,
                        .ibt = &self.lookup_indices_by_table,
                    };
                    const RafInitCtx = struct {
                        left_raf: *RafDecomposition(F),
                        right_raf: *RafDecomposition(F),
                        identity_raf: *RafDecomposition(F),
                        eq: []const F,
                        indices: []const u128,
                        is_interleaved: []const bool,
                        pool: *ThreadPool,
                        alloc_inner: Allocator,
                    };
                    const raf_ctx = RafInitCtx{
                        .left_raf = &self.left_raf,
                        .right_raf = &self.right_raf,
                        .identity_raf = &self.identity_raf,
                        .eq = self.lookups_eq_evals,
                        .indices = self.lookup_indices_u128,
                        .is_interleaved = self.is_interleaved_operands,
                        .pool = tp,
                        .alloc_inner = self.allocator,
                    };
                    _ = tp.join(
                        void,
                        void,
                        suffix_ctx,
                        struct {
                            fn f(c: SuffixInitCtx) void {
                                c.polys.initPhase(c.phase, c.phases, c.eq, c.indices, c.table_indices, c.pool, c.alloc_inner, c.ibt) catch unreachable;
                            }
                        }.f,
                        raf_ctx,
                        struct {
                            fn f(c: RafInitCtx) void {
                                initQRaf(F, c.left_raf, c.right_raf, c.identity_raf, c.eq, c.indices, c.is_interleaved, c.pool, c.alloc_inner) catch unreachable;
                            }
                        }.f,
                    );
                } else {
                    try self.suffix_polys.initPhase(self.current_phase, self.num_phases, self.lookups_eq_evals, self.lookup_indices_u128, self.cycle_table_indices, null, self.allocator, null);
                    try initQRaf(F, &self.left_raf, &self.right_raf, &self.identity_raf, self.lookups_eq_evals, self.lookup_indices_u128, self.is_interleaved_operands, null, self.allocator);
                }

                // Materialize prefix MLE tables for new phase
                self.left_raf.initPrefix();
                self.right_raf.initPrefix();
                self.identity_raf.initPrefix();

                // Reset the new phase's expanding table to 1
                self.expanding_tables[self.current_phase].reset(F.one());
            }
        }

        /// Set the current claim directly (used by the main loop after evaluating
        /// the Instance 2 polynomial at the challenge).
        pub fn setClaim(self: *Self, new_claim: F) void {
            self.current_claim = new_claim;
        }

        /// Rematerialize state at the transition from address rounds to cycle rounds (round 128).
        /// This computes table values at r_address, rematerializes combined_vals,
        /// allocates and populates ra_chunk_weights, and initializes split-eq tables.
        pub fn rematerialize(self: *Self, challenges: []const F) !void {
            const T = self.T;

            // Get bound prefix values from RAF decompositions
            self.left_raf.updateCheckpoint();
            self.right_raf.updateCheckpoint();
            self.identity_raf.updateCheckpoint();
            const left_prefix = self.left_raf.bound_value;
            const right_prefix = self.right_raf.bound_value;
            const identity_prefix = self.identity_raf.bound_value;

            if (comptime debug_verbose) {
                dbg("[STAGE5 REMATERIALIZE] round=128, left_prefix={x}, right_prefix={x}, identity_prefix={x}\n", .{
                    left_prefix.toBytesBE()[16..32].*,
                    right_prefix.toBytesBE()[16..32].*,
                    identity_prefix.toBytesBE()[16..32].*,
                });
            }

            // Compute RAF scalar values
            const raf_interleaved = self.gamma_raf.mul(left_prefix).add(self.gamma_raf2.mul(right_prefix));
            const raf_identity = self.gamma_raf2.mul(identity_prefix);

            // Compute table_values_at_r_addr using bound prefix checkpoints
            const table_values = computeTableValuesAtRAddress(F, &self.prefix_checkpoints);

            // Store for later use in opening claims
            for (0..NUM_TABLES) |ti| {
                self.stored_table_values[ti] = table_values[ti];
            }

            // Rematerialize combined_vals for cycle rounds
            {
                const RematCtx = struct {
                    combined: []F,
                    t_indices: []const i8,
                    is_identity: []const bool,
                    table_vals: *const [MAX_LOOKUP_TABLES]F,
                    raf_il: F,
                    raf_id: F,
                };
                const rctx = RematCtx{
                    .combined = self.lookups_combined_vals,
                    .t_indices = self.cycle_table_indices,
                    .is_identity = self.cycle_is_identity_path,
                    .table_vals = &self.stored_table_values,
                    .raf_il = raf_interleaved,
                    .raf_id = raf_identity,
                };
                const rematFn = struct {
                    fn f(c: RematCtx, j: usize) void {
                        var val = F.zero();
                        const t_idx_j = c.t_indices[j];
                        if (t_idx_j >= 0) {
                            const ti: usize = @intCast(t_idx_j);
                            if (ti < NUM_TABLES) {
                                val = c.table_vals[ti];
                            }
                        }
                        if (!c.is_identity[j]) {
                            val = val.add(c.raf_il);
                        } else {
                            val = val.add(c.raf_id);
                        }
                        c.combined[j] = val;
                    }
                }.f;
                pool_helpers.parallelForOptional(self.thread_pool, T, rctx, rematFn);
            }

            // Reinitialize lookups_eq_evals for cycle rounds (debug only needs full rebuild)
            if (comptime debug_verbose) {
                Inst.buildFullEqTable(self.r_reduction, self.lookups_eq_evals[0..T], self.thread_pool);
            }

            // Allocate ra_chunk_weights
            for (0..self.ra_num_chunks) |chunk_idx| {
                self.ra_chunk_weights[chunk_idx] = self.allocator.alloc(F, T) catch unreachable;
            }
            self.ra_chunk_weights_allocated = true;

            // Materialize ra_chunk_weights from expanding tables
            const phases_per_chunk = self.num_phases / self.ra_num_chunks;
            {
                const RaChunkRematCtx = struct {
                    indices_lo: []const u64,
                    indices_hi: []const u64,
                    chunks: *[MAX_RA_CHUNKS][]F,
                    tables: *const [16]ExpandingTable(F),
                    n_phases: usize,
                    ppc: usize,
                    n_chunks: usize,
                    lm: usize,
                };
                const rctx = RaChunkRematCtx{
                    .indices_lo = self.lookups_indices_lo,
                    .indices_hi = self.lookups_indices_hi,
                    .chunks = &self.ra_chunk_weights,
                    .tables = &self.expanding_tables,
                    .n_phases = self.num_phases,
                    .ppc = phases_per_chunk,
                    .n_chunks = self.ra_num_chunks,
                    .lm = self.log_m,
                };
                const rematChunkFn = struct {
                    fn f(c: RaChunkRematCtx, j: usize) void {
                        const k_lo = c.indices_lo[j];
                        const k_hi = c.indices_hi[j];
                        const phase_mask = (@as(usize, 1) << @intCast(c.lm)) - 1;
                        for (0..c.n_chunks) |chunk_idx| {
                            var ra_val = F.one();
                            const phase_start = chunk_idx * c.ppc;
                            const phase_end = @min((chunk_idx + 1) * c.ppc, c.n_phases);
                            for (phase_start..phase_end) |phase| {
                                const shift = (c.n_phases - 1 - phase) * c.lm;
                                var k_phase: usize = undefined;
                                if (shift >= 64) {
                                    k_phase = @truncate(k_hi >> @intCast(shift - 64));
                                } else if (shift + c.lm <= 64) {
                                    k_phase = @truncate(k_lo >> @intCast(shift));
                                } else {
                                    const lo_bits = k_lo >> @intCast(shift);
                                    const hi_bits = k_hi << @intCast(64 - shift);
                                    k_phase = @truncate(lo_bits | hi_bits);
                                }
                                k_phase &= phase_mask;
                                ra_val = ra_val.mul(c.tables[phase].get(k_phase));
                            }
                            c.chunks[chunk_idx][j] = ra_val;
                        }
                    }
                }.f;
                pool_helpers.parallelForOptional(self.thread_pool, T, rctx, rematChunkFn);
            }

            // Free data structures no longer needed after rematerialization
            self.suffix_polys.deinit();
            self.suffix_polys = AllSuffixPolys(F).init(self.allocator);
            for (0..self.num_phases) |phase_idx| {
                self.expanding_tables[phase_idx].deinit();
                self.expanding_tables[phase_idx] = ExpandingTable(F).init(self.allocator, 1) catch unreachable;
            }

            // Debug: update ra_weights from chunks
            if (comptime debug_verbose) {
                for (0..T) |j| {
                    var ra_prod = F.one();
                    for (0..self.ra_num_chunks) |c| {
                        ra_prod = ra_prod.mul(self.ra_chunk_weights[c][j]);
                    }
                    self.lookups_ra_weights[j] = ra_prod;
                }
            }

            // Free lookup_indices_by_table (no longer needed)
            for (0..self.lookup_indices_by_table_count) |t| {
                self.allocator.free(self.lookup_indices_by_table[t]);
            }
            self.lookup_indices_by_table_count = 0;

            // Initialize split-eq tables
            _ = challenges; // challenges are accessible from the caller
            // Split-eq initialization happens on the first call to computeCycleRoundPoly
        }

        /// Compute the cycle round polynomial for Instance 2.
        /// Returns the full degree-10 coefficient array (11 elements, caller must free).
        /// lookups_round is the cycle-relative round (0..n_cycle_vars-1).
        pub fn computeCycleRoundPoly(self: *Self, lookups_round: usize) ![]F {
            const T = self.T;

            // Initialize split-eq tables on first cycle round
            if (!self.split_eq_initialized) {
                const init_vars = self.n_cycle_vars - 1;
                self.split_eq_m_in = init_vars / 2;
                const m_out = init_vars - self.split_eq_m_in;
                self.split_eq_E_in_len = @as(usize, 1) << @intCast(self.split_eq_m_in);
                self.split_eq_E_out_len = @as(usize, 1) << @intCast(m_out);
                std.debug.assert(self.split_eq_E_in_len <= MAX_SPLIT_EQ_SIZE);
                std.debug.assert(self.split_eq_E_out_len <= MAX_SPLIT_EQ_SIZE);

                // Build E_out from r_reduction[0..m_out]
                self.split_eq_E_out[0] = F.one();
                for (0..m_out) |i| {
                    const ri = self.r_reduction[i];
                    const len = @as(usize, 1) << @intCast(i);
                    var j2: usize = len;
                    while (j2 > 0) {
                        j2 -= 1;
                        self.split_eq_E_out[2 * j2 + 1] = self.split_eq_E_out[j2].mul(ri);
                        self.split_eq_E_out[2 * j2] = self.split_eq_E_out[j2].sub(self.split_eq_E_out[2 * j2 + 1]);
                    }
                }
                // Build E_in from r_reduction[m_out..m_out+m_in]
                self.split_eq_E_in[0] = F.one();
                for (0..self.split_eq_m_in) |i| {
                    const ri = self.r_reduction[m_out + i];
                    const len = @as(usize, 1) << @intCast(i);
                    var j2: usize = len;
                    while (j2 > 0) {
                        j2 -= 1;
                        self.split_eq_E_in[2 * j2 + 1] = self.split_eq_E_in[j2].mul(ri);
                        self.split_eq_E_in[2 * j2] = self.split_eq_E_in[j2].sub(self.split_eq_E_in[2 * j2 + 1]);
                    }
                }
                self.split_eq_initialized = true;
            }

            // Get r_round for this cycle variable (LowToHigh binding)
            const r_round = self.r_reduction[self.n_cycle_vars - 1 - lookups_round];

            // Accumulate sum of 9-factor products via flat parallelReduce with split-eq
            const current_half_size = T >> @intCast(lookups_round + 1);
            var sum_evals: [9]F = blk: {
                const EvalCtx = struct {
                    combined: []const F,
                    chunks: *const [MAX_RA_CHUNKS][]F,
                    e_in: [*]const F,
                    e_out: [*]const F,
                    e_in_mask: usize,
                    e_in_shift: std.math.Log2Int(usize),
                    n_chunks: usize,
                };
                const e_in_shift: std.math.Log2Int(usize) = @intCast(@ctz(self.split_eq_E_in_len));
                const ectx = EvalCtx{
                    .combined = self.lookups_combined_vals,
                    .chunks = &self.ra_chunk_weights,
                    .e_in = &self.split_eq_E_in,
                    .e_out = &self.split_eq_E_out,
                    .e_in_mask = self.split_eq_E_in_len - 1,
                    .e_in_shift = e_in_shift,
                    .n_chunks = self.ra_num_chunks,
                };
                const identity = [_]F{F.zero()} ** 9;
                const mapFn = struct {
                    fn f(c: EvalCtx, start: usize, end: usize) [9]F {
                        var acc = [_]UnreducedProductAccum{UnreducedProductAccum.zero()} ** 9;
                        for (start..end) |j| {
                            var pairs: [9][2]F = undefined;
                            const x_in = j & c.e_in_mask;
                            const x_out = j >> c.e_in_shift;
                            const eq_prefix = c.e_out[x_out].mul(c.e_in[x_in]);
                            pairs[0][0] = eq_prefix.mul(c.combined[2 * j]);
                            pairs[0][1] = eq_prefix.mul(c.combined[2 * j + 1]);
                            for (0..c.n_chunks) |ci| {
                                pairs[ci + 1][0] = c.chunks[ci][2 * j];
                                pairs[ci + 1][1] = c.chunks[ci][2 * j + 1];
                            }
                            UniPoly(F).evalProd9Accumulate(pairs, &acc);
                        }
                        var result: [9]F = undefined;
                        inline for (0..9) |k| result[k] = acc[k].reduce();
                        return result;
                    }
                }.f;
                const reduceFn = struct {
                    fn f(a: [9]F, b: [9]F) [9]F {
                        var r: [9]F = undefined;
                        for (0..9) |k| r[k] = a[k].add(b[k]);
                        return r;
                    }
                }.f;
                break :blk pool_helpers.parallelReduceOptional([9]F, self.thread_pool, current_half_size, identity, ectx, mapFn, reduceFn);
            };

            // Multiply sum_evals by lookups_current_scalar
            for (&sum_evals) |*eval| {
                eval.* = eval.*.mul(self.lookups_current_scalar);
            }

            // Use finish_mles_product_sum_from_evals to recover degree-10 polynomial
            const full_coeffs = try UniPoly(F).finishMlesProductSumFromEvals(
                self.allocator,
                &sum_evals,
                self.current_claim,
                r_round,
            );

            return full_coeffs;
        }

        /// Bind a cycle round challenge for lookups.
        /// Updates combined_vals, ra_chunk_weights, current_scalar, split-eq tables,
        /// and the current claim.
        pub fn bindCycleChallenge(self: *Self, challenge: F, lookups_round: usize, full_coeffs: []const F) void {
            // Update current_claim by evaluating Instance 2's polynomial at challenge
            {
                var p2_at_r = full_coeffs[full_coeffs.len - 1];
                var idx = full_coeffs.len - 1;
                while (idx > 0) {
                    idx -= 1;
                    p2_at_r = p2_at_r.mulHiBigIntU128(challenge.limbs).add(full_coeffs[idx]);
                }
                self.current_claim = p2_at_r;
            }

            // Bind combined_vals + all ra_chunk_weights
            if (self.gpu_ops) |gpu| {
                const lk_n = self.lookups_combined_vals.len >> @intCast(lookups_round);
                const lk_half = lk_n / 2;
                if (lk_half >= 16384) {
                    gpu.polyBindLow(self.lookups_combined_vals[0 .. lk_half * 2], challenge, self.lookups_combined_vals[0..lk_half]) catch {
                        for (0..lk_half) |i| {
                            self.lookups_combined_vals[i] = self.lookups_combined_vals[2 * i].add(challenge.mul(self.lookups_combined_vals[2 * i + 1].sub(self.lookups_combined_vals[2 * i])));
                        }
                    };
                    for (lk_half..lk_n) |i| {
                        self.lookups_combined_vals[i] = F.zero();
                    }
                } else {
                    Inst.bindSinglePolynomial(self.lookups_combined_vals, lookups_round, challenge, self.thread_pool, self.gpu_ops);
                }
                for (0..self.ra_num_chunks) |chunk_idx| {
                    const ra_poly = self.ra_chunk_weights[chunk_idx];
                    const ra_n = ra_poly.len >> @intCast(lookups_round);
                    const ra_half = ra_n / 2;
                    if (ra_half >= 16384) {
                        gpu.polyBindLow(ra_poly[0 .. ra_half * 2], challenge, ra_poly[0..ra_half]) catch {
                            for (0..ra_half) |i| {
                                ra_poly[i] = ra_poly[2 * i].add(challenge.mul(ra_poly[2 * i + 1].sub(ra_poly[2 * i])));
                            }
                        };
                        for (ra_half..ra_n) |i| {
                            ra_poly[i] = F.zero();
                        }
                    } else {
                        Inst.bindSinglePolynomial(ra_poly, lookups_round, challenge, self.thread_pool, self.gpu_ops);
                    }
                }
            } else {
                const BindArraysCtx = struct {
                    combined: []F,
                    chunks: *[MAX_RA_CHUNKS][]F,
                    n_chunks: usize,
                    lround: usize,
                    chal: F,
                };
                const bactx = BindArraysCtx{
                    .combined = self.lookups_combined_vals,
                    .chunks = &self.ra_chunk_weights,
                    .n_chunks = self.ra_num_chunks,
                    .lround = lookups_round,
                    .chal = challenge,
                };
                pool_helpers.parallelForOptional(self.thread_pool, self.ra_num_chunks + 1, bactx, struct {
                    fn f(c: BindArraysCtx, arr_idx: usize) void {
                        const poly = if (arr_idx == 0) c.combined else c.chunks[arr_idx - 1];
                        const n = poly.len >> @intCast(c.lround);
                        const half = n / 2;
                        if (half == 0) return;
                        for (0..half) |i| {
                            poly[i] = poly[2 * i].add(c.chal.mul(poly[2 * i + 1].sub(poly[2 * i])));
                        }
                        for (half..n) |i| {
                            poly[i] = F.zero();
                        }
                    }
                }.f);
            }

            // Update lookups_current_scalar with eq(w_i, challenge)
            const w_i = self.r_reduction[self.n_cycle_vars - 1 - lookups_round];
            const prod_w_r = w_i.mulHiBigIntU128(challenge.limbs);
            const one_minus_r_scalar = F.one().sub(challenge);
            const eq_factor = one_minus_r_scalar.sub(w_i).add(prod_w_r).add(prod_w_r);
            self.lookups_current_scalar = self.lookups_current_scalar.mul(eq_factor);

            // Halve split-eq tables
            if (self.split_eq_E_in_len > 1) {
                const new_len = self.split_eq_E_in_len / 2;
                for (0..new_len) |k| {
                    self.split_eq_E_in[k] = self.split_eq_E_in[2 * k].add(self.split_eq_E_in[2 * k + 1]);
                }
                self.split_eq_E_in_len = new_len;
            } else if (self.split_eq_E_out_len > 1) {
                const new_len = self.split_eq_E_out_len / 2;
                for (0..new_len) |k| {
                    self.split_eq_E_out[k] = self.split_eq_E_out[2 * k].add(self.split_eq_E_out[2 * k + 1]);
                }
                self.split_eq_E_out_len = new_len;
            }
        }

        /// Extract opening claims after all 136 sumcheck rounds.
        /// Returns ra_chunks, table_flags, and raf_flag.
        /// Caller owns ra_chunks and table_flags slices.
        pub fn getOpeningClaims(
            self: *Self,
            challenges: []const F,
        ) !LookupsOpeningClaims(F) {
            const num_lookup_tables: usize = 40;
            const lookups_ra_d = LOOKUPS_LOG_K / self.lookups_ra_virtual_log_k_chunk;
            const T = self.T;

            // Extract r_cycle_prime (last n_cycle_vars challenges)
            const r_cycle_prime = challenges[LOOKUPS_LOG_K..];

            // Reverse to BIG_ENDIAN for eq computation
            var r_cycle_prime_be = try self.allocator.alloc(F, self.n_cycle_vars);
            defer self.allocator.free(r_cycle_prime_be);
            for (0..self.n_cycle_vars) |i| {
                r_cycle_prime_be[i] = r_cycle_prime[self.n_cycle_vars - 1 - i];
            }

            // ra_claims[i] = ra_chunk_weights[i][0] after all bindings
            const ra_chunks = try self.allocator.alloc(F, lookups_ra_d);
            for (0..lookups_ra_d) |i| {
                ra_chunks[i] = self.ra_chunk_weights[i][0];
            }

            // DEBUG: dump final-state bound values and compute self-consistency check.
            // current_claim should equal:
            //   lookups_current_scalar * E_in[0] * E_out[0] * combined_vals[0] * Π ra_chunks[i][0]
            if (comptime !is_wasm) {
                if (platformGetenv("ZOLT_S5_DEBUG") != null) {
                    const cur_be = self.current_claim.toBytesBE();
                    std.debug.print("[LOOKUPS_END] current_claim_LE=", .{});
                    for (0..32) |bi| std.debug.print("{x:0>2}", .{cur_be[31 - bi]});
                    std.debug.print("\n", .{});

                    const cv0_be = self.lookups_combined_vals[0].toBytesBE();
                    std.debug.print("[LOOKUPS_END] combined_vals[0]_LE=", .{});
                    for (0..32) |bi| std.debug.print("{x:0>2}", .{cv0_be[31 - bi]});
                    std.debug.print("\n", .{});

                    const lcs_be = self.lookups_current_scalar.toBytesBE();
                    std.debug.print("[LOOKUPS_END] lookups_current_scalar_LE=", .{});
                    for (0..32) |bi| std.debug.print("{x:0>2}", .{lcs_be[31 - bi]});
                    std.debug.print("\n", .{});

                    const ei0 = if (self.split_eq_E_in_len > 0) self.split_eq_E_in[0] else F.one();
                    const eo0 = if (self.split_eq_E_out_len > 0) self.split_eq_E_out[0] else F.one();
                    const ei_be = ei0.toBytesBE();
                    const eo_be = eo0.toBytesBE();
                    std.debug.print("[LOOKUPS_END] E_in[0]_LE=", .{});
                    for (0..32) |bi| std.debug.print("{x:0>2}", .{ei_be[31 - bi]});
                    std.debug.print("\n[LOOKUPS_END] E_out[0]_LE=", .{});
                    for (0..32) |bi| std.debug.print("{x:0>2}", .{eo_be[31 - bi]});
                    std.debug.print("\n", .{});

                    var ra_prod = F.one();
                    for (0..lookups_ra_d) |i| ra_prod = ra_prod.mul(ra_chunks[i]);
                    const rp_be = ra_prod.toBytesBE();
                    std.debug.print("[LOOKUPS_END] ra_product_LE=", .{});
                    for (0..32) |bi| std.debug.print("{x:0>2}", .{rp_be[31 - bi]});
                    std.debug.print("\n", .{});

                    // Expected self-consistent value
                    const self_check = self.lookups_current_scalar
                        .mul(ei0)
                        .mul(eo0)
                        .mul(self.lookups_combined_vals[0])
                        .mul(ra_prod);
                    const sc_be = self_check.toBytesBE();
                    std.debug.print("[LOOKUPS_END] self_check_LE=", .{});
                    for (0..32) |bi| std.debug.print("{x:0>2}", .{sc_be[31 - bi]});
                    std.debug.print("\n[LOOKUPS_END] self_check_eq_current={}\n", .{self_check.eql(self.current_claim)});
                }
            }

            // Compute table flags using split-eq parallel histogram
            const table_flags = try self.allocator.alloc(F, num_lookup_tables);
            @memset(table_flags, F.zero());

            const flag_n_hi = self.n_cycle_vars / 2;
            const flag_n_lo = self.n_cycle_vars - flag_n_hi;
            const flag_hi_len = @as(usize, 1) << @intCast(flag_n_hi);
            const flag_lo_len = @as(usize, 1) << @intCast(flag_n_lo);

            // Build E_lo and E_hi
            const E_lo_flags = self.lookups_eq_evals[0..flag_lo_len];
            Inst.buildFullEqTable(r_cycle_prime_be[flag_n_hi..self.n_cycle_vars], E_lo_flags, self.thread_pool);
            const E_hi_flags = try self.allocator.alloc(F, flag_hi_len);
            defer self.allocator.free(E_hi_flags);
            Inst.buildFullEqTable(r_cycle_prime_be[0..flag_n_hi], E_hi_flags, self.thread_pool);

            // Parallel histogram accumulation
            const FlagResult = struct {
                flags: [NUM_TABLES]F,
                raf: F,
            };
            const FlagCtx = struct {
                e_hi: []const F,
                e_lo: []const F,
                lo_len: usize,
                tbl_ids: []const i8,
                is_id: []const bool,
                n_tables: usize,
                total_T: usize,
            };
            const flag_ctx = FlagCtx{
                .e_hi = E_hi_flags,
                .e_lo = E_lo_flags,
                .lo_len = flag_lo_len,
                .tbl_ids = self.cycle_table_indices,
                .is_id = self.cycle_is_identity_path,
                .n_tables = num_lookup_tables,
                .total_T = T,
            };
            const flag_identity = FlagResult{
                .flags = [_]F{F.zero()} ** NUM_TABLES,
                .raf = F.zero(),
            };
            const flagMapFn = struct {
                fn f(c: FlagCtx, hi_start: usize, hi_end: usize) FlagResult {
                    var local_flags = [_]F{F.zero()} ** NUM_TABLES;
                    var local_raf = F.zero();
                    for (hi_start..hi_end) |c_hi| {
                        const base = c_hi * c.lo_len;
                        var inner_flags = [_]F{F.zero()} ** NUM_TABLES;
                        var inner_raf = F.zero();
                        const inner_end = @min(base + c.lo_len, c.total_T);
                        for (base..inner_end) |j| {
                            const c_lo = j - base;
                            const eq_lo = c.e_lo[c_lo];
                            const table_idx = c.tbl_ids[j];
                            if (table_idx >= 0 and @as(usize, @intCast(table_idx)) < c.n_tables) {
                                inner_flags[@intCast(table_idx)] = inner_flags[@intCast(table_idx)].add(eq_lo);
                            }
                            if (c.is_id[j]) {
                                inner_raf = inner_raf.add(eq_lo);
                            }
                        }
                        const e_hi = c.e_hi[c_hi];
                        for (0..c.n_tables) |t| {
                            if (!inner_flags[t].eql(F.zero())) {
                                local_flags[t] = local_flags[t].add(e_hi.mul(inner_flags[t]));
                            }
                        }
                        if (!inner_raf.eql(F.zero())) {
                            local_raf = local_raf.add(e_hi.mul(inner_raf));
                        }
                    }
                    return FlagResult{ .flags = local_flags, .raf = local_raf };
                }
            }.f;
            const flagReduceFn = struct {
                fn f(a: FlagResult, b: FlagResult) FlagResult {
                    var r: FlagResult = undefined;
                    for (0..NUM_TABLES) |t| {
                        r.flags[t] = a.flags[t].add(b.flags[t]);
                    }
                    r.raf = a.raf.add(b.raf);
                    return r;
                }
            }.f;

            const flag_result = pool_helpers.parallelReduceOptional(FlagResult, self.thread_pool, flag_hi_len, flag_identity, flag_ctx, flagMapFn, flagReduceFn);

            @memcpy(table_flags[0..num_lookup_tables], flag_result.flags[0..num_lookup_tables]);
            const computed_raf_flag = flag_result.raf;

            return LookupsOpeningClaims(F){
                .ra_chunks = ra_chunks,
                .table_flags = table_flags,
                .raf_flag = computed_raf_flag,
            };
        }
    };
}

test "stage5_lookups compiles" {
    const F = @import("zolt_arith").field.BN254Scalar;
    // Just verify the type compiles
    const Prover = LookupsReadRafProver(F);
    _ = Prover;
}
