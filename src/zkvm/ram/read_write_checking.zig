//! RAM Read/Write Checking Sumcheck Prover
//!
//! This implements the RamReadWriteChecking sumcheck for Stage 2 verification.
//! It proves the consistency of RAM read/write operations across the execution trace.
//!
//! The sumcheck proves:
//! Σ_{k,j} eq(r_cycle, j) * ra(k,j) * (Val(k,j) + γ*(Val(k,j) + inc(j))) = rv_claim + γ*wv_claim
//!
//! This is a 2-phase prover:
//! - Phase 1 (rounds 0 to log_T-1): Cycle-major sparse matrix, binds cycle variables
//! - Phase 2 (rounds log_T to log_T+log_K-1): Address-major, binds address variables

const std = @import("std");

const zkvm_debug = @import("../debug.zig");
const dbg = zkvm_debug.dbg;
const debug_verbose = zkvm_debug.verbose;

const Allocator = std.mem.Allocator;
const zolt_pool = @import("zolt_pool");
const ThreadPool = zolt_pool.ThreadPool;
const parallelReduceOptional = zolt_pool.parallelReduceOptional;
const parallelForOptional = zolt_pool.parallelForOptional;
const MemoryTrace = @import("mod.zig").MemoryTrace;
const MemoryAccess = @import("mod.zig").MemoryAccess;
const MemoryOp = @import("mod.zig").MemoryOp;
const split_eq = @import("zolt_arith").poly.split_eq;

/// Parameters for RAM read/write checking
pub fn RamReadWriteCheckingParams(comptime F: type) type {
    return struct {
        const Self = @This();

        /// Random challenge for combining read and write values
        gamma: F,
        /// Challenges from SpartanOuter sumcheck (cycle dimension)
        r_cycle: []const F,
        /// Log2 of number of addresses
        log_k: usize,
        /// Log2 of trace length
        log_t: usize,
        /// Number of rounds in Phase 1 (cycle binding) - determines when Phase 2 starts
        /// This allows the prover to match the batched sumcheck's 3-phase structure:
        /// - Phase 1: phase1_num_rounds (some cycle vars)
        /// - Phase 2: log_k (all address vars)
        /// - Phase 3: remaining (rest of cycle vars)
        phase1_num_rounds: usize,
        /// Start address of RAM region
        start_address: u64,
        /// Allocator for internal use
        allocator: Allocator,

        pub fn init(
            allocator: Allocator,
            gamma: F,
            r_cycle: []const F,
            log_k: usize,
            log_t: usize,
            start_address: u64,
        ) !Self {
            // Default: use log_t for backward compatibility
            return initWithPhaseConfig(allocator, gamma, r_cycle, log_k, log_t, log_t, start_address);
        }

        pub fn initWithPhaseConfig(
            allocator: Allocator,
            gamma: F,
            r_cycle: []const F,
            log_k: usize,
            log_t: usize,
            phase1_num_rounds: usize,
            start_address: u64,
        ) !Self {
            const r_cycle_copy = try allocator.alloc(F, r_cycle.len);
            @memcpy(r_cycle_copy, r_cycle);

            return Self{
                .gamma = gamma,
                .r_cycle = r_cycle_copy,
                .log_k = log_k,
                .log_t = log_t,
                .phase1_num_rounds = phase1_num_rounds,
                .start_address = start_address,
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.r_cycle);
        }

        /// Total number of sumcheck rounds
        pub fn numRounds(self: *const Self) usize {
            return self.log_k + self.log_t;
        }
    };
}

/// Sparse matrix entry for cycle-major ordering
pub fn CycleMajorEntry(comptime F: type) type {
    return struct {
        const Self = @This();

        /// Cycle index (row)
        cycle: usize,
        /// Address index (column)
        address: usize,
        /// ra coefficient (1 for accessed entries)
        ra_coeff: F,
        /// Value coefficient (memory value before access)
        val_coeff: F,
        /// Previous value (for tracking) - field element, bound during Phase 2
        prev_val: F,
        /// Next value (for tracking write increments) - field element, bound during Phase 2
        next_val: F,

        /// Bind entries at even and odd rows to create a new entry at row/2
        /// Matches Jolt's CycleMajorMatrixEntry::bind_entries
        pub fn bindEntries(even: ?*const Self, odd: ?*const Self, r: F) ?Self {
            if (even != null and odd != null) {
                // Both entries exist
                const e = even.?.*;
                const o = odd.?.*;
                std.debug.assert(e.cycle % 2 == 0);
                std.debug.assert(o.cycle % 2 == 1);
                std.debug.assert(e.address == o.address);
                const new_val = e.val_coeff.add(r.mul(o.val_coeff.sub(e.val_coeff)));
                dbg("[BIND CYCLE] BOTH: even_cycle={}, even_val={any}, odd_val={any}, r={any}, result_val={any}\n", .{ e.cycle, e.val_coeff.toBytesBE()[0..8], o.val_coeff.toBytesBE()[0..8], r.toBytesBE()[0..8], new_val.toBytesBE()[0..8] });
                return Self{
                    .cycle = e.cycle / 2,
                    .address = e.address,
                    .ra_coeff = e.ra_coeff.add(r.mul(o.ra_coeff.sub(e.ra_coeff))),
                    .val_coeff = new_val,
                    .prev_val = e.prev_val,
                    .next_val = o.next_val,
                };
            } else if (even != null) {
                // Only even entry exists - odd is implicit
                const e = even.?.*;
                const odd_val_coeff = e.next_val;
                const new_val = e.val_coeff.add(r.mul(odd_val_coeff.sub(e.val_coeff)));
                dbg("[BIND CYCLE] EVEN_ONLY: even_cycle={}, even_val={any}, odd_implicit_val={any}, r={any}, result_val={any}\n", .{ e.cycle, e.val_coeff.toBytesBE()[0..8], odd_val_coeff.toBytesBE()[0..8], r.toBytesBE()[0..8], new_val.toBytesBE()[0..8] });
                return Self{
                    .cycle = e.cycle / 2,
                    .address = e.address,
                    .ra_coeff = F.one().sub(r).mul(e.ra_coeff),
                    .val_coeff = new_val,
                    .prev_val = e.prev_val,
                    .next_val = e.next_val,
                };
            } else if (odd != null) {
                // Only odd entry exists - even is implicit
                const o = odd.?.*;
                const even_val_coeff = o.prev_val;
                const new_val = even_val_coeff.add(r.mul(o.val_coeff.sub(even_val_coeff)));
                dbg("[BIND CYCLE] ODD_ONLY: odd_cycle={}, even_implicit_val={any}, odd_val={any}, r={any}, result_val={any}\n", .{ o.cycle, even_val_coeff.toBytesBE()[0..8], o.val_coeff.toBytesBE()[0..8], r.toBytesBE()[0..8], new_val.toBytesBE()[0..8] });
                return Self{
                    .cycle = o.cycle / 2,
                    .address = o.address,
                    .ra_coeff = r.mul(o.ra_coeff),
                    .val_coeff = new_val,
                    .prev_val = o.prev_val,
                    .next_val = o.next_val,
                };
            } else {
                return null;
            }
        }
    };
}

/// RAM Read/Write Checking Prover
///
/// This prover handles the sumcheck for verifying RAM consistency.
pub fn RamReadWriteCheckingProver(comptime F: type) type {
    return struct {
        const Self = @This();
        const Entry = CycleMajorEntry(F);
        const GruenSplitEq = split_eq.GruenSplitEqPolynomial(F);

        /// Parameters
        params: RamReadWriteCheckingParams(F),
        /// Current claim (updated after each round)
        current_claim: F,
        /// Current round
        round: usize,
        /// Sparse matrix entries (cycle-major)
        entries: std.ArrayListUnmanaged(Entry),
        /// Inc polynomial evaluations (one per cycle)
        inc: []F,
        /// Initial memory values (one per address)
        val_init: []F,
        /// Challenges bound so far
        challenges: std.ArrayListUnmanaged(F),
        /// Eq polynomial evaluations that get folded each round
        eq_evals: []F,
        /// Current effective size of eq_evals (halves each round in Phase 1)
        eq_size: usize,
        /// Gruen split eq polynomial for optimized round polynomial computation
        gruen_eq: ?GruenSplitEq,
        /// Memory trace (kept for dense val_claim computation)
        trace: *const MemoryTrace,
        /// Allocator
        allocator: Allocator,
        thread_pool: ?*ThreadPool = null,

        pub fn init(
            allocator: Allocator,
            trace: *const MemoryTrace,
            params: RamReadWriteCheckingParams(F),
            initial_claim: F,
            initial_ram: ?*const std.AutoHashMapUnmanaged(u64, u64),
            memory_layout: ?*const @import("../jolt_device.zig").MemoryLayout,
            is_panicking: bool,
        ) !Self {
            const K = @as(usize, 1) << @intCast(params.log_k);
            const T = @as(usize, 1) << @intCast(params.log_t);

            // Initialize inc polynomial (zero for all cycles by default)
            const inc = try allocator.alloc(F, T);
            @memset(inc, F.zero());

            // Initialize val_init from initial RAM state
            const val_init = try allocator.alloc(F, K);
            @memset(val_init, F.zero());

            dbg("[RWC INIT] params.start_address = 0x{x:0>16}\n", .{params.start_address});
            dbg("[RWC INIT] K = {}, initial_ram entries = {}\n", .{ K, if (initial_ram) |ram| ram.count() else 0 });

            if (initial_ram) |ram| {
                var iter = ram.iterator();
                var populated_count: usize = 0;
                dbg("[RWC INIT] Populating val_init from initial_ram:\n", .{});
                while (iter.next()) |entry| {
                    const addr = entry.key_ptr.*;
                    const val = entry.value_ptr.*;
                    if (addr >= params.start_address) {
                        const idx: usize = @intCast((addr - params.start_address) / 8);
                        if (idx < K) {
                            val_init[idx] = F.fromU64(val);
                            if (populated_count < 5) {
                                dbg("[RWC INIT]   addr=0x{x:0>16}, idx={}, val={}\n", .{ addr, idx, val });
                            }
                            populated_count += 1;
                        }
                    }
                }
                dbg("[RWC INIT] Populated {} val_init entries (shown first 5)\n", .{populated_count});
            }

            // NOTE: We do NOT add termination or panic bits to val_init here.
            //
            // Jolt's initial RAM state (used for eval_initial_ram_mle in verification)
            // includes: bytecode, inputs, trusted/untrusted advice.
            // It does NOT include: outputs, panic bit, termination bit.
            //
            // The termination bit is only in the FINAL RAM state (val_final), which is
            // used by OutputSumcheck, not RamReadWriteChecking.
            //
            // For programs without RAM operations (like Fibonacci):
            //   - rwc_val_claim = MLE(initial_ram) @ r_address
            //   - init_eval = MLE(initial_ram) @ r_address (computed by verifier)
            //   - input_claim = rwc_val_claim - init_eval = 0
            //
            // This matches Jolt's behavior exactly.
            _ = memory_layout;
            _ = is_panicking;

            // Build sparse matrix entries from trace
            // Track current value per address to compute inc = new_value - prev_value
            var current_val_per_addr = std.AutoHashMapUnmanaged(usize, u64){};
            defer current_val_per_addr.deinit(allocator);

            // Initialize current values from initial RAM state
            if (initial_ram) |ram| {
                var iter = ram.iterator();
                while (iter.next()) |entry| {
                    const addr = entry.key_ptr.*;
                    const val = entry.value_ptr.*;
                    if (addr >= params.start_address) {
                        const idx: usize = @intCast((addr - params.start_address) / 8);
                        if (idx < K) {
                            try current_val_per_addr.put(allocator, idx, val);
                        }
                    }
                }
            }

            var entries = std.ArrayListUnmanaged(Entry).empty;
            for (trace.accesses.items) |access| {
                if (access.timestamp >= T) continue;

                const addr_idx: usize = blk: {
                    if (access.address >= params.start_address) {
                        const idx: usize = @intCast((access.address - params.start_address) / 8);
                        if (idx < K) break :blk idx;
                    }
                    continue;
                };

                // Get previous value at this address
                const prev_val = current_val_per_addr.get(addr_idx) orelse 0;

                // For writes, compute inc = new_value - prev_value (as signed difference)
                if (access.op == .Write) {
                    // inc = new_value - prev_value (can be negative, use field arithmetic)
                    const new_val = access.value;
                    const inc_val = if (new_val >= prev_val)
                        F.fromU64(new_val - prev_val)
                    else
                        F.zero().sub(F.fromU64(prev_val - new_val));
                    inc[@intCast(access.timestamp)] = inc_val;
                    dbg("[RWC INC SET] cycle={}, new_val={}, prev_val={}, inc={any}\n", .{
                        access.timestamp,
                        new_val,
                        prev_val,
                        inc_val.toBytesBE(),
                    });
                    // Update current value for this address
                    try current_val_per_addr.put(allocator, addr_idx, new_val);
                }

                // val_coeff should be the value BEFORE the access (Val(k,j))
                // For reads: pre-value == value read
                // For writes: pre-value (not the post-value)
                const val_coeff = if (access.op == .Write)
                    F.fromU64(prev_val) // Use pre-value for writes
                else
                    F.fromU64(access.value); // Use value for reads

                try entries.append(allocator, Entry{
                    .cycle = @intCast(access.timestamp),
                    .address = addr_idx,
                    .ra_coeff = F.one(),
                    .val_coeff = val_coeff,
                    .prev_val = F.fromU64(prev_val),
                    .next_val = F.fromU64(access.value),
                });

                dbg("[RWC INIT] entry: cycle={}, addr={}, op={}, prev_val={}, next_val={}, inc[{}]={any}\n", .{
                    access.timestamp,
                    addr_idx,
                    @intFromEnum(access.op),
                    prev_val,
                    access.value,
                    access.timestamp,
                    if (access.timestamp < T) inc[@as(usize, @intCast(access.timestamp))].toBytesBE()[0..8] else &[_]u8{ 0, 0, 0, 0, 0, 0, 0, 0 },
                });
            }

            // Sort entries by (cycle, address) for cycle-major ordering
            // Use unstable sort (pdq) — no duplicate (cycle, address) pairs exist
            std.mem.sortUnstable(Entry, entries.items, {}, struct {
                fn lessThan(_: void, a: Entry, b: Entry) bool {
                    if (a.cycle != b.cycle) return a.cycle < b.cycle;
                    return a.address < b.address;
                }
            }.lessThan);

            // Initialize eq polynomial evaluations: eq(r_cycle, j) for each cycle j
            // r_cycle is in BIG_ENDIAN order (MSB first, as stored in tau)
            // Use O(T) table construction instead of O(T·logT) per-element computation
            const poly_mod = @import("zolt_arith").poly;
            const EqPoly = poly_mod.EqPolynomial(F);
            const eq_evals = try EqPoly.evalsSliceWithScaling(F, allocator, params.r_cycle, null);

            const challenges_list = std.ArrayListUnmanaged(F).empty;

            // Initialize GruenSplitEqPolynomial for Phase 1 optimization
            // This matches Jolt's structure for computing round polynomials
            const gruen_eq = try GruenSplitEq.init(allocator, params.r_cycle);

            dbg("[RWC INIT] tau.len = {}, current_index = {}\n", .{ params.r_cycle.len, gruen_eq.current_index });
            if (params.r_cycle.len > 0) {
                dbg("[RWC INIT] tau[0] = {any}\n", .{params.r_cycle[0].toBytesBE()[0..8]});
                if (params.r_cycle.len > 1) {
                    dbg("[RWC INIT] tau[last] = {any}\n", .{params.r_cycle[params.r_cycle.len - 1].toBytesBE()[0..8]});
                }
            }

            // VERIFY: Check initial sum (debug only — O(N) + O(T) allocation)
            if (comptime debug_verbose) {
                var verify_init_sum = F.zero();
                for (entries.items) |ve| {
                    const eq_j = if (ve.cycle < T) eq_evals[ve.cycle] else F.zero();
                    const inc_j = if (ve.cycle < T) inc[ve.cycle] else F.zero();
                    const inner = ve.val_coeff.add(params.gamma.mul(ve.val_coeff.add(inc_j)));
                    verify_init_sum = verify_init_sum.add(eq_j.mul(ve.ra_coeff).mul(inner));
                }
                dbg("[RWC INIT VERIFY] initial_claim = {any}\n", .{initial_claim.toBytesBE()});
                dbg("[RWC INIT VERIFY] sum_of_entries = {any}\n", .{verify_init_sum.toBytesBE()});
                dbg("[RWC INIT VERIFY] match = {}\n", .{verify_init_sum.eql(initial_claim)});
                var rv_sum = F.zero();
                var wv_sum = F.zero();
                for (entries.items) |ve| {
                    const eq_j = if (ve.cycle < T) eq_evals[ve.cycle] else F.zero();
                    const inc_j = if (ve.cycle < T) inc[ve.cycle] else F.zero();
                    rv_sum = rv_sum.add(eq_j.mul(ve.ra_coeff).mul(ve.val_coeff));
                    wv_sum = wv_sum.add(eq_j.mul(ve.ra_coeff).mul(ve.val_coeff.add(inc_j)));
                }
                dbg("[RWC INIT VERIFY] rv_sum = {any}\n", .{rv_sum.toBytesBE()});
                dbg("[RWC INIT VERIFY] wv_sum = {any}\n", .{wv_sum.toBytesBE()});
                dbg("[RWC INIT VERIFY] rv + gamma*wv = {any}\n", .{rv_sum.add(params.gamma.mul(wv_sum)).toBytesBE()});
                const poly_mod_dbg = @import("zolt_arith").poly;
                const EqPolyDbg = poly_mod_dbg.EqPolynomial(F);
                const eq_le_evals = try EqPolyDbg.evalsSliceWithScaling(F, allocator, params.r_cycle, null);
                defer allocator.free(eq_le_evals);
                const eq_le_54 = if (54 < eq_le_evals.len) eq_le_evals[54] else F.zero();
                dbg("[RWC INIT VERIFY] eq_BE[54] = {any}\n", .{eq_evals[54].toBytesBE()});
                dbg("[RWC INIT VERIFY] eq_LE[54] (EqPoly) = {any}\n", .{eq_le_54.toBytesBE()});
                const sum_le = eq_le_54.mul(params.gamma);
                dbg("[RWC INIT VERIFY] gamma*eq_LE[54] = {any}\n", .{sum_le.toBytesBE()});
            }

            return Self{
                .params = params,
                .current_claim = initial_claim,
                .round = 0,
                .entries = entries,
                .inc = inc,
                .val_init = val_init,
                .challenges = challenges_list,
                .eq_evals = eq_evals,
                .eq_size = T,
                .gruen_eq = gruen_eq,
                .trace = trace,
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            self.entries.deinit(self.allocator);
            self.allocator.free(self.inc);
            self.allocator.free(self.val_init);
            self.allocator.free(self.eq_evals);
            self.challenges.deinit(self.allocator);
            if (self.gruen_eq) |*geq| {
                geq.deinit();
            }
            self.params.deinit();
        }

        /// Compute round polynomial [s(0), s(1), s(2), s(3)] for batched cubic sumcheck
        pub fn computeRoundPolynomialCubic(self: *Self) [4]F {
            const gamma = self.params.gamma;
            const phase1_end = self.params.phase1_num_rounds;
            const phase2_end = phase1_end + self.params.log_k;

            // Phase structure for batched sumcheck:
            // - Phase 1: phase1_num_rounds (some cycle vars) - cycle-major entries
            // - Phase 2: log_k rounds (all address vars) - address-major entries + val_init
            // - Phase 3: remaining (rest of cycle vars) - cycle-major entries
            const in_cycle_phase = self.round < phase1_end or self.round >= phase2_end;

            const result = if (in_cycle_phase)
                self.computePhase1Polynomial(gamma)
            else
                self.computePhase2Polynomial(gamma);

            return result;
        }

        fn computePhase1Polynomial(self: *Self, gamma: F) [4]F {
            var gruen_eq = &self.gruen_eq.?;
            const tables = gruen_eq.getWindowEqTables(gruen_eq.current_index, 1);
            const E_out = tables.E_out;
            const E_in = tables.E_in;
            const head_in_bits = tables.head_in_bits;

            if (self.entries.items.len == 0) {
                return gruen_eq.computeCubicRoundPoly(F.zero(), F.zero(), self.current_claim);
            }

            const RWCCMCtx = struct {
                entries: []const CycleMajorEntry(F),
                inc: []const F,
                E_out: []const F,
                E_in: []const F,
                head_in_bits: usize,
                gamma: F,
            };
            const ctx = RWCCMCtx{
                .entries = self.entries.items,
                .inc = self.inc,
                .E_out = E_out,
                .E_in = E_in,
                .head_in_bits = head_in_bits,
                .gamma = gamma,
            };

            // Split by entry range (not group index) to avoid O(N) skip-to-start scan
            const mapFn = struct {
                fn f(c: RWCCMCtx, entry_start: usize, entry_end: usize) [2]F {
                    var local_qc = F.zero();
                    var local_qq = F.zero();

                    // Align start to group boundary
                    var scan = entry_start;
                    if (scan > 0 and scan < c.entries.len) {
                        const prev_pid = c.entries[scan - 1].cycle / 2;
                        const cur_pid = c.entries[scan].cycle / 2;
                        if (cur_pid == prev_pid) {
                            while (scan < entry_end and c.entries[scan].cycle / 2 == cur_pid) scan += 1;
                        }
                    }

                    while (scan < entry_end) {
                        const row_pair_idx = c.entries[scan].cycle / 2;
                        const group_start = scan;
                        while (scan < c.entries.len and c.entries[scan].cycle / 2 == row_pair_idx) scan += 1;

                        const x_out = row_pair_idx >> @intCast(c.head_in_bits);
                        const x_in_mask = (@as(usize, 1) << @intCast(c.head_in_bits)) - 1;
                        const x_in = row_pair_idx & x_in_mask;
                        const E_prefix = (if (x_out < c.E_out.len) c.E_out[x_out] else F.one())
                            .mul(if (x_in < c.E_in.len) c.E_in[x_in] else F.one());

                        const j_prime = row_pair_idx * 2;
                        const inc_0 = if (j_prime < c.inc.len) c.inc[j_prime] else F.zero();
                        const inc_1 = if (j_prime + 1 < c.inc.len) c.inc[j_prime + 1] else F.zero();
                        const inc_inf = inc_1.sub(inc_0);

                        const pair_entries = c.entries[group_start..scan];
                        var odd_start: usize = 0;
                        while (odd_start < pair_entries.len and pair_entries[odd_start].cycle % 2 == 0) odd_start += 1;
                        const even_row = pair_entries[0..odd_start];
                        const odd_row = pair_entries[odd_start..];

                        var ei: usize = 0;
                        var oi: usize = 0;
                        while (ei < even_row.len and oi < odd_row.len) {
                            var ra_0: F = undefined;
                            var ra_i: F = undefined;
                            var val_0: F = undefined;
                            var val_i: F = undefined;
                            if (even_row[ei].address == odd_row[oi].address) {
                                ra_0 = even_row[ei].ra_coeff;
                                ra_i = odd_row[oi].ra_coeff.sub(even_row[ei].ra_coeff);
                                val_0 = even_row[ei].val_coeff;
                                val_i = odd_row[oi].val_coeff.sub(even_row[ei].val_coeff);
                                ei += 1;
                                oi += 1;
                            } else if (even_row[ei].address < odd_row[oi].address) {
                                ra_0 = even_row[ei].ra_coeff;
                                ra_i = F.zero().sub(even_row[ei].ra_coeff);
                                val_0 = even_row[ei].val_coeff;
                                val_i = even_row[ei].next_val.sub(even_row[ei].val_coeff);
                                ei += 1;
                            } else {
                                ra_0 = F.zero();
                                ra_i = odd_row[oi].ra_coeff;
                                val_0 = odd_row[oi].prev_val;
                                val_i = odd_row[oi].val_coeff.sub(odd_row[oi].prev_val);
                                oi += 1;
                            }
                            local_qc = local_qc.add(E_prefix.mul(ra_0).mul(val_0.add(c.gamma.mul(inc_0.add(val_0)))));
                            local_qq = local_qq.add(E_prefix.mul(ra_i).mul(val_i.add(c.gamma.mul(inc_inf.add(val_i)))));
                        }
                        while (ei < even_row.len) : (ei += 1) {
                            const r0 = even_row[ei].ra_coeff;
                            const v0 = even_row[ei].val_coeff;
                            const vi = even_row[ei].next_val.sub(v0);
                            local_qc = local_qc.add(E_prefix.mul(r0).mul(v0.add(c.gamma.mul(inc_0.add(v0)))));
                            local_qq = local_qq.add(E_prefix.mul(F.zero().sub(r0)).mul(vi.add(c.gamma.mul(inc_inf.add(vi)))));
                        }
                        while (oi < odd_row.len) : (oi += 1) {
                            const ri = odd_row[oi].ra_coeff;
                            const v0 = odd_row[oi].prev_val;
                            const vi = odd_row[oi].val_coeff.sub(v0);
                            local_qc = local_qc.add(E_prefix.mul(F.zero()).mul(v0.add(c.gamma.mul(inc_0.add(v0)))));
                            local_qq = local_qq.add(E_prefix.mul(ri).mul(vi.add(c.gamma.mul(inc_inf.add(vi)))));
                        }
                    }
                    return .{ local_qc, local_qq };
                }
            }.f;

            const reduceFn = struct {
                fn f(a: [2]F, b: [2]F) [2]F {
                    return .{ a[0].add(b[0]), a[1].add(b[1]) };
                }
            }.f;

            const identity = [2]F{ F.zero(), F.zero() };
            const sums = parallelReduceOptional([2]F, self.thread_pool, self.entries.items.len, identity, ctx, mapFn, reduceFn);

            const result = gruen_eq.computeCubicRoundPoly(sums[0], sums[1], self.current_claim);

            dbg("[RWC PHASE1] round={}, q_constant={any}\n", .{ self.round, sums[0].toBytesBE()[0..8] });
            dbg("[RWC PHASE1] q_quadratic={any}, current_claim={any}\n", .{ sums[1].toBytesBE()[0..8], self.current_claim.toBytesBE()[0..8] });
            dbg("[RWC PHASE1] result: s0={any}, s1={any}\n", .{ result[0].toBytesBE()[0..8], result[1].toBytesBE()[0..8] });

            return result;
        }

        fn computePhase2Polynomial(self: *Self, gamma: F) [4]F {
            // Phase 2: Binding address variables using AddressMajor ordering
            //
            // Jolt's approach:
            // 1. Entries sorted by (column, row) = (address, cycle)
            // 2. Group by column pairs (2k, 2k+1)
            // 3. For each pair, merge by row with checkpoint tracking
            // 4. Return [s(0), s(2)] and derive s(1) from current_claim
            //
            // At start of Phase 2, entries should be in AddressMajor order.

            const phase1_end = self.params.phase1_num_rounds;
            const addr_round = self.round - phase1_end;

            dbg("[RWC PHASE2] round={}, addr_round={}, entries.len={}\n", .{ self.round, addr_round, self.entries.items.len });

            // Convert to AddressMajor at start of Phase 2
            if (addr_round == 0) {
                // Sort entries by (address, cycle) for AddressMajor ordering
                // Use unstable sort (pdq) — no duplicate (address, cycle) pairs exist
                std.mem.sortUnstable(Entry, self.entries.items, {}, struct {
                    fn lessThan(_: void, a: Entry, b: Entry) bool {
                        if (a.address != b.address) return a.address < b.address;
                        return a.cycle < b.cycle;
                    }
                }.lessThan);

                dbg("[RWC PHASE2] Converted to AddressMajor order\n", .{});
                dbg("[RWC PHASE2] eq_cycle_scalar = {any}\n", .{self.eq_evals[0].toBytesBE()[0..8]});
                dbg("[RWC PHASE2] inc_scalar = {any}\n", .{self.inc[0].toBytesBE()[0..8]});
                if (self.entries.items.len > 0) {
                    const e = self.entries.items[0];
                    dbg("[RWC PHASE2] entry[0]: addr={}, ra_coeff={any}, val_coeff={any}\n", .{
                        e.address,
                        e.ra_coeff.toBytesBE()[0..8],
                        e.val_coeff.toBytesBE()[0..8],
                    });
                }
                // VERIFICATION: At start of Phase 2, compute the actual sum of all entries
                // and check it matches current_claim.
                // Sum = eq_cycle * Σ_k ra(k) * (val(k) + gamma*(val(k) + inc))
                const eq_s = self.eq_evals[0];
                const inc_s = self.inc[0];
                const gamma_s = self.params.gamma;
                var verify_sum = F.zero();
                for (self.entries.items) |ve| {
                    const inner = ve.val_coeff.add(gamma_s.mul(ve.val_coeff.add(inc_s)));
                    verify_sum = verify_sum.add(eq_s.mul(ve.ra_coeff).mul(inner));
                }
                dbg("[RWC PHASE2 VERIFY] Sum of entries = {any}\n", .{verify_sum.toBytesBE()});
                dbg("[RWC PHASE2 VERIFY] current_claim = {any}\n", .{self.current_claim.toBytesBE()});
                dbg("[RWC PHASE2 VERIFY] match = {}\n", .{verify_sum.eql(self.current_claim)});
            }

            // After all cycle variables are bound:
            // - eq_evals[0] is the scalar eq(r_cycle_params, r_cycle_sumcheck)
            // - inc[0] is the scalar inc(r_cycle_sumcheck) (after Phase 1 folding)
            const eq_cycle_scalar = self.eq_evals[0];
            const inc_scalar = self.inc[0];

            const K = @as(usize, 1) << @intCast(self.params.log_k);
            const val_init_current_size = K >> @intCast(addr_round);

            const s0s2 = blk: {
                if (self.entries.items.len == 0) break :blk [2]F{ F.zero(), F.zero() };

                const RWCP2Ctx = struct {
                    entries: []const CycleMajorEntry(F),
                    val_init: []const F,
                    vi_size: usize,
                    inc_scalar: F,
                    eq_scalar: F,
                    gamma: F,
                    addr_round: usize,
                    phase1_end: usize,
                    challenges: []const F,
                };
                const p2ctx = RWCP2Ctx{
                    .entries = self.entries.items,
                    .val_init = self.val_init,
                    .vi_size = val_init_current_size,
                    .inc_scalar = inc_scalar,
                    .eq_scalar = eq_cycle_scalar,
                    .gamma = gamma,
                    .addr_round = addr_round,
                    .phase1_end = phase1_end,
                    .challenges = self.challenges.items,
                };

                // Split by entry range, align to group boundaries
                const p2MapFn = struct {
                    fn f(c: RWCP2Ctx, entry_start: usize, entry_end: usize) [2]F {
                        var ls0 = F.zero();
                        var ls2 = F.zero();

                        var scan = entry_start;
                        if (scan > 0 and scan < c.entries.len) {
                            const prev_pid = c.entries[scan - 1].address / 2;
                            const cur_pid = c.entries[scan].address / 2;
                            if (cur_pid == prev_pid) {
                                while (scan < entry_end and c.entries[scan].address / 2 == cur_pid) scan += 1;
                            }
                        }

                        while (scan < entry_end) {
                            const col_pair = c.entries[scan].address / 2;
                            const pair_start = scan;
                            while (scan < c.entries.len and c.entries[scan].address / 2 == col_pair) scan += 1;

                            const even_col_idx = col_pair * 2;
                            const odd_col_idx = even_col_idx + 1;
                            var even_cp = if (even_col_idx < c.vi_size) c.val_init[even_col_idx] else F.zero();
                            var odd_cp = if (odd_col_idx < c.vi_size) c.val_init[odd_col_idx] else F.zero();

                            var odd_start: usize = pair_start;
                            while (odd_start < scan and c.entries[odd_start].address % 2 == 0) odd_start += 1;

                            var ei = pair_start;
                            var oi = odd_start;
                            while (ei < odd_start and oi < scan) {
                                const ee = &c.entries[ei];
                                const oe = &c.entries[oi];
                                if (ee.cycle == oe.cycle) {
                                    const ev = computePhase2Evals(ee, oe, even_cp, odd_cp, c.inc_scalar, c.eq_scalar, c.gamma, c.addr_round, c.phase1_end, c.challenges);
                                    ls0 = ls0.add(ev[0]);
                                    ls2 = ls2.add(ev[1]);
                                    even_cp = ee.next_val;
                                    odd_cp = oe.next_val;
                                    ei += 1;
                                    oi += 1;
                                } else if (ee.cycle < oe.cycle) {
                                    const ev = computePhase2EvalsEvenOnly(ee, even_cp, odd_cp, c.inc_scalar, c.eq_scalar, c.gamma, c.addr_round, c.phase1_end, c.challenges);
                                    ls0 = ls0.add(ev[0]);
                                    ls2 = ls2.add(ev[1]);
                                    even_cp = ee.next_val;
                                    ei += 1;
                                } else {
                                    const ev = computePhase2EvalsOddOnly(oe, even_cp, odd_cp, c.inc_scalar, c.eq_scalar, c.gamma, c.addr_round, c.phase1_end, c.challenges);
                                    ls0 = ls0.add(ev[0]);
                                    ls2 = ls2.add(ev[1]);
                                    odd_cp = oe.next_val;
                                    oi += 1;
                                }
                            }
                            while (ei < odd_start) : (ei += 1) {
                                const ev = computePhase2EvalsEvenOnly(&c.entries[ei], even_cp, odd_cp, c.inc_scalar, c.eq_scalar, c.gamma, c.addr_round, c.phase1_end, c.challenges);
                                ls0 = ls0.add(ev[0]);
                                ls2 = ls2.add(ev[1]);
                                even_cp = c.entries[ei].next_val;
                            }
                            while (oi < scan) : (oi += 1) {
                                const ev = computePhase2EvalsOddOnly(&c.entries[oi], even_cp, odd_cp, c.inc_scalar, c.eq_scalar, c.gamma, c.addr_round, c.phase1_end, c.challenges);
                                ls0 = ls0.add(ev[0]);
                                ls2 = ls2.add(ev[1]);
                                odd_cp = c.entries[oi].next_val;
                            }
                        }
                        return .{ ls0, ls2 };
                    }
                }.f;

                const p2ReduceFn = struct {
                    fn f(a: [2]F, b: [2]F) [2]F {
                        return .{ a[0].add(b[0]), a[1].add(b[1]) };
                    }
                }.f;

                const p2id = [2]F{ F.zero(), F.zero() };
                break :blk parallelReduceOptional([2]F, self.thread_pool, self.entries.items.len, p2id, p2ctx, p2MapFn, p2ReduceFn);
            };

            const s0 = s0s2[0];
            const s1 = self.current_claim.sub(s0);
            const s2 = s0s2[1];

            // Extrapolate s(3) for degree-2 polynomial
            const s3 = s2.mul(F.fromU64(3)).sub(s1.mul(F.fromU64(3))).add(s0);

            if (addr_round < 3) {
                dbg("[RWC PHASE2] result: s0={any}, s1={any}, s2={any}\n", .{
                    s0.toBytesBE()[0..8],
                    s1.toBytesBE()[0..8],
                    s2.toBytesBE()[0..8],
                });
            }

            return [4]F{ s0, s1, s2, s3 };
        }

        /// Compute [s(0), s(2)] contribution for both even and odd entries at same row
        /// In Jolt's Phase 2, the eq factor is ONLY eq(r_cycle, row) from the bound eq polynomial.
        /// There is NO separate eq_addr factor - the address dimension is handled purely
        /// through the entry pairing structure (even/odd with checkpoints).
        fn computePhase2Evals(
            even_entry: *const Entry,
            odd_entry: *const Entry,
            even_checkpoint: F,
            odd_checkpoint: F,
            inc_eval: F,
            eq_eval: F,
            gamma: F,
            addr_round: usize,
            phase1_end: usize,
            challenges: []const F,
        ) [2]F {
            _ = even_checkpoint;
            _ = odd_checkpoint;
            _ = addr_round;
            _ = phase1_end;
            _ = challenges;

            // Following Jolt exactly: eq_eval is the only eq factor (from Phase 1 cycle binding)
            // ra_evals = [ra_even, 2*ra_odd - ra_even]
            const ra_0 = even_entry.ra_coeff;
            const ra_2 = odd_entry.ra_coeff.add(odd_entry.ra_coeff).sub(even_entry.ra_coeff);

            // val_evals = [val_even, 2*val_odd - val_even]
            const val_0 = even_entry.val_coeff;
            const val_2 = odd_entry.val_coeff.add(odd_entry.val_coeff).sub(even_entry.val_coeff);

            const one_plus_gamma = F.one().add(gamma);
            const s0_contrib = eq_eval.mul(ra_0).mul(val_0.mul(one_plus_gamma).add(gamma.mul(inc_eval)));
            const s2_contrib = eq_eval.mul(ra_2).mul(val_2.mul(one_plus_gamma).add(gamma.mul(inc_eval)));

            return [2]F{ s0_contrib, s2_contrib };
        }

        /// Compute [s(0), s(2)] contribution when only even entry exists
        fn computePhase2EvalsEvenOnly(
            even_entry: *const Entry,
            even_checkpoint: F,
            odd_checkpoint: F,
            inc_eval: F,
            eq_eval: F,
            gamma: F,
            addr_round: usize,
            phase1_end: usize,
            challenges: []const F,
        ) [2]F {
            _ = even_checkpoint;
            _ = addr_round;
            _ = phase1_end;
            _ = challenges;

            // Implicit odd entry has ra=0, val=odd_checkpoint
            // ra_evals = [ra_even, -ra_even] (since odd ra = 0)
            const ra_0 = even_entry.ra_coeff;
            const ra_2 = F.zero().sub(even_entry.ra_coeff); // 2*0 - ra_even = -ra_even

            // val_evals = [val_even, 2*odd_checkpoint - val_even]
            const val_0 = even_entry.val_coeff;
            const val_2 = odd_checkpoint.add(odd_checkpoint).sub(even_entry.val_coeff);

            const one_plus_gamma = F.one().add(gamma);
            const s0_contrib = eq_eval.mul(ra_0).mul(val_0.mul(one_plus_gamma).add(gamma.mul(inc_eval)));
            const s2_contrib = eq_eval.mul(ra_2).mul(val_2.mul(one_plus_gamma).add(gamma.mul(inc_eval)));

            return [2]F{ s0_contrib, s2_contrib };
        }

        /// Compute [s(0), s(2)] contribution when only odd entry exists
        fn computePhase2EvalsOddOnly(
            odd_entry: *const Entry,
            even_checkpoint: F,
            odd_checkpoint: F,
            inc_eval: F,
            eq_eval: F,
            gamma: F,
            addr_round: usize,
            phase1_end: usize,
            challenges: []const F,
        ) [2]F {
            _ = odd_checkpoint;
            _ = addr_round;
            _ = phase1_end;
            _ = challenges;

            // Implicit even entry has ra=0, val=even_checkpoint
            // ra_evals = [0, 2*ra_odd] (since even ra = 0)
            const ra_0 = F.zero();
            const ra_2 = odd_entry.ra_coeff.add(odd_entry.ra_coeff); // 2*ra_odd - 0

            // val_evals = [even_checkpoint, 2*val_odd - even_checkpoint]
            const val_0 = even_checkpoint;
            const val_2 = odd_entry.val_coeff.add(odd_entry.val_coeff).sub(even_checkpoint);

            const one_plus_gamma = F.one().add(gamma);
            // s(0) = 0 since ra(0) = 0
            const s0_contrib = eq_eval.mul(ra_0).mul(val_0.mul(one_plus_gamma).add(gamma.mul(inc_eval)));
            const s2_contrib = eq_eval.mul(ra_2).mul(val_2.mul(one_plus_gamma).add(gamma.mul(inc_eval)));

            return [2]F{ s0_contrib, s2_contrib };
        }

        /// Bind a challenge after round polynomial computation
        pub fn bindChallenge(self: *Self, challenge: F) !void {
            try self.challenges.append(self.allocator, challenge);

            // Phase structure for batched sumcheck:
            // - Phase 1: phase1_num_rounds (some cycle vars) - eq and inc binding
            // - Phase 2: log_k rounds (all address vars) - val_init binding
            // - Phase 3: remaining (rest of cycle vars + any extra address vars)
            const phase1_end = self.params.phase1_num_rounds;
            const phase2_end = phase1_end + self.params.log_k;

            // Fold eq_evals and inc in Phase 1 (and Phase 3 for remaining cycle vars)
            // In the 3-phase structure, cycle vars are bound in Phase 1 and Phase 3
            const in_cycle_phase = self.round < phase1_end or self.round >= phase2_end;
            if (in_cycle_phase and self.eq_size > 1) {
                const half = self.eq_size / 2;

                // Fold eq_evals and inc in parallel (independent arrays)
                const RWCBindCtx = struct { slices: [2][]F, r: F, n: usize };
                const rwc_bctx = RWCBindCtx{
                    .slices = .{ self.eq_evals, self.inc },
                    .r = challenge,
                    .n = half,
                };
                const rwcBindFn = struct {
                    fn f(c: RWCBindCtx, idx: usize) void {
                        const arr = c.slices[idx];
                        for (0..c.n) |i| {
                            arr[i] = arr[2 * i].add(c.r.mul(arr[2 * i + 1].sub(arr[2 * i])));
                        }
                    }
                }.f;
                parallelForOptional(self.thread_pool, 2, rwc_bctx, rwcBindFn);

                self.eq_size = half;

                // Bind the Gruen eq polynomial to update current_scalar and E tables
                if (self.gruen_eq) |*geq| {
                    geq.bind(challenge);
                }

                // Bind entries: group by (row/2, col), create bound entries
                // This matches Jolt's ReadWriteMatrixCycleMajor::bind
                try self.bindEntries(challenge);

                // VERIFY: After binding, check if sum of entries matches current_claim
                // Sum should be: Σ_k Σ_j eq_eval[j] * ra_coeff(k,j) * (val_coeff(k,j) + gamma*(val_coeff(k,j) + inc[j]))
                if (self.round < 3 or self.round == phase1_end - 1) {
                    const verify_gamma = self.params.gamma;
                    var verify_sum_phase1 = F.zero();
                    for (self.entries.items) |ve| {
                        const eq_j = if (ve.cycle < self.eq_size) self.eq_evals[ve.cycle] else F.zero();
                        const inc_j = if (ve.cycle < self.eq_size) self.inc[ve.cycle] else F.zero();
                        const inner = ve.val_coeff.add(verify_gamma.mul(ve.val_coeff.add(inc_j)));
                        verify_sum_phase1 = verify_sum_phase1.add(eq_j.mul(ve.ra_coeff).mul(inner));
                    }
                    dbg("[RWC PHASE1 VERIFY] round={}, sum={any}\n", .{ self.round, verify_sum_phase1.toBytesBE()[0..8] });
                    dbg("[RWC PHASE1 VERIFY] round={}, claim={any}\n", .{ self.round, self.current_claim.toBytesBE()[0..8] });
                    dbg("[RWC PHASE1 VERIFY] round={}, match={}\n", .{ self.round, verify_sum_phase1.eql(self.current_claim) });
                }
            }

            // Phase 2: Address binding phase
            // CRITICAL: Jolt binds entries FIRST (using current val_init for checkpoints),
            // then binds val_init LAST. We must follow the same order.
            const in_phase2 = self.round >= phase1_end and self.round < phase2_end;
            if (in_phase2) {
                const addr_round = self.round - phase1_end;

                // Bind entries in AddressMajor format (by column pairs)
                // This uses val_init for checkpoints, so must happen BEFORE val_init binding
                try self.bindEntriesAddressMajor(challenge, addr_round);

                // NOW bind val_init (after entries have used the unbound checkpoints)
                // This matches Jolt's val_init.bind_parallel(r, BindingOrder::LowToHigh)
                const K = @as(usize, 1) << @intCast(self.params.log_k);
                const current_size = K >> @intCast(addr_round);
                if (current_size > 1) {
                    const half = current_size / 2;
                    for (0..half) |i| {
                        const lo = self.val_init[2 * i];
                        const hi = self.val_init[2 * i + 1];
                        self.val_init[i] = lo.add(challenge.mul(hi.sub(lo)));
                    }
                }

                // PHASE 2 DIAGNOSTIC: Check val_coeff consistency after binding
                if (addr_round < 3 or addr_round == 15) {
                    const gamma_v = self.params.gamma;
                    const eq_v = self.eq_evals[0];
                    const inc_v = self.inc[0];
                    var ra_v: F = F.zero();
                    var val_v: F = F.zero();
                    for (self.entries.items) |entry| {
                        ra_v = ra_v.add(entry.ra_coeff);
                        val_v = val_v.add(entry.val_coeff);
                    }
                    if (self.entries.items.len == 0) val_v = self.val_init[0];
                    const expected_v = eq_v.mul(ra_v).mul(val_v.add(gamma_v.mul(val_v.add(inc_v))));
                    dbg("[RWC PHASE2 BIND CHECK] addr_round={}, match={}\n", .{ addr_round, expected_v.eql(self.current_claim) });
                    if (addr_round < 3) {
                        dbg("[RWC PHASE2 BIND CHECK]   val_coeff={any}\n", .{val_v.toBytesBE()[0..8]});
                        dbg("[RWC PHASE2 BIND CHECK]   ra_coeff={any}\n", .{ra_v.toBytesBE()[0..8]});
                        dbg("[RWC PHASE2 BIND CHECK]   current_claim={any}\n", .{self.current_claim.toBytesBE()[0..8]});
                        dbg("[RWC PHASE2 BIND CHECK]   expected={any}\n", .{expected_v.toBytesBE()[0..8]});
                    }
                }
            }

            self.round += 1;
        }

        /// Bind entries in AddressMajor format for Phase 2
        /// Groups entries by column pairs (2k, 2k+1) and merges by row
        fn bindEntriesAddressMajor(self: *Self, r: F, addr_round: usize) !void {
            const K = @as(usize, 1) << @intCast(self.params.log_k);
            const val_init_current_size = K >> @intCast(addr_round);

            var new_entries = std.ArrayListUnmanaged(Entry).empty;
            try new_entries.ensureTotalCapacity(self.allocator, self.entries.items.len);

            var entry_idx: usize = 0;
            while (entry_idx < self.entries.items.len) {
                const entry = self.entries.items[entry_idx];

                // Determine column pair for this entry
                // CRITICAL FIX: entry.address has ALREADY been divided by 2 at each previous round
                // So entry.address IS the current column. Do NOT shift by addr_round!
                const col = entry.address;
                const col_pair = col / 2;
                const even_col_idx = col_pair * 2;
                const odd_col_idx = even_col_idx + 1;

                // Get checkpoints from bound val_init
                var even_checkpoint = if (even_col_idx < val_init_current_size)
                    self.val_init[even_col_idx]
                else
                    F.zero();
                var odd_checkpoint = if (odd_col_idx < val_init_current_size)
                    self.val_init[odd_col_idx]
                else
                    F.zero();

                // Find all entries in this column pair
                var pair_end = entry_idx;
                while (pair_end < self.entries.items.len) {
                    const e = self.entries.items[pair_end];
                    const e_col = e.address;
                    const e_pair = e_col / 2;
                    if (e_pair != col_pair) break;
                    pair_end += 1;
                }

                // Split into even and odd columns within this pair
                var j = entry_idx;
                while (j < pair_end) {
                    const e = self.entries.items[j];
                    const e_col = e.address;
                    if (e_col % 2 == 1) break;
                    j += 1;
                }

                const even_start = entry_idx;
                const even_end = j;
                const odd_start = j;
                const odd_end = pair_end;

                var even_idx = even_start;
                var odd_idx = odd_start;

                // Merge by row with checkpoint tracking
                while (even_idx < even_end and odd_idx < odd_end) {
                    const even_entry = &self.entries.items[even_idx];
                    const odd_entry = &self.entries.items[odd_idx];

                    if (even_entry.cycle == odd_entry.cycle) {
                        if (bindAddressMajorPair(even_entry, odd_entry, even_checkpoint, odd_checkpoint, r)) |bound| {
                            new_entries.appendAssumeCapacity(bound);
                        }
                        even_checkpoint = even_entry.next_val;
                        odd_checkpoint = odd_entry.next_val;
                        even_idx += 1;
                        odd_idx += 1;
                    } else if (even_entry.cycle < odd_entry.cycle) {
                        if (bindAddressMajorEvenOnly(even_entry, even_checkpoint, odd_checkpoint, r)) |bound| {
                            new_entries.appendAssumeCapacity(bound);
                        }
                        even_checkpoint = even_entry.next_val;
                        even_idx += 1;
                    } else {
                        if (bindAddressMajorOddOnly(odd_entry, even_checkpoint, odd_checkpoint, r)) |bound| {
                            new_entries.appendAssumeCapacity(bound);
                        }
                        odd_checkpoint = odd_entry.next_val;
                        odd_idx += 1;
                    }
                }

                // Process remaining even entries
                while (even_idx < even_end) {
                    const even_entry = &self.entries.items[even_idx];
                    if (bindAddressMajorEvenOnly(even_entry, even_checkpoint, odd_checkpoint, r)) |bound| {
                        new_entries.appendAssumeCapacity(bound);
                    }
                    even_checkpoint = even_entry.next_val;
                    even_idx += 1;
                }

                // Process remaining odd entries
                while (odd_idx < odd_end) {
                    const odd_entry = &self.entries.items[odd_idx];
                    if (bindAddressMajorOddOnly(odd_entry, even_checkpoint, odd_checkpoint, r)) |bound| {
                        new_entries.appendAssumeCapacity(bound);
                    }
                    odd_checkpoint = odd_entry.next_val;
                    odd_idx += 1;
                }

                entry_idx = pair_end;
            }

            // Replace old entries with bound entries
            self.entries.deinit(self.allocator);
            self.entries = new_entries;

            dbg("[RWC BIND PHASE2] addr_round={}, entries.len after bind={}\n", .{ addr_round, self.entries.items.len });
        }

        /// Bind two adjacent column entries together (both even and odd exist)
        fn bindAddressMajorPair(even: *const Entry, odd: *const Entry, even_checkpoint: F, odd_checkpoint: F, r: F) ?Entry {
            _ = even_checkpoint;
            _ = odd_checkpoint;

            // Matching Jolt's RamAddressMajorEntry::bind_entries (Some, Some) case
            // prev_val and next_val are field elements that get bound like val_coeff
            return Entry{
                .cycle = even.cycle,
                .address = even.address / 2,
                .ra_coeff = even.ra_coeff.add(r.mul(odd.ra_coeff.sub(even.ra_coeff))),
                .val_coeff = even.val_coeff.add(r.mul(odd.val_coeff.sub(even.val_coeff))),
                .prev_val = even.prev_val.add(r.mul(odd.prev_val.sub(even.prev_val))),
                .next_val = even.next_val.add(r.mul(odd.next_val.sub(even.next_val))),
            };
        }

        /// Bind when only even entry exists (odd is implicit with ra=0, val=checkpoint)
        fn bindAddressMajorEvenOnly(even: *const Entry, even_checkpoint: F, odd_checkpoint: F, r: F) ?Entry {
            _ = even_checkpoint;

            // Matching Jolt's (Some(even), None) case
            // Implicit odd has ra=0, val=odd_checkpoint, prev_val=odd_checkpoint, next_val=odd_checkpoint
            const one_minus_r = F.one().sub(r);
            const new_val = even.val_coeff.add(r.mul(odd_checkpoint.sub(even.val_coeff)));
            dbg("[BIND ADDR] EVEN_ONLY: addr={}, even_val={any}, odd_chkpt={any}, r={any}, result_val={any}\n", .{ even.address, even.val_coeff.toBytesBE()[0..8], odd_checkpoint.toBytesBE()[0..8], r.toBytesBE()[0..8], new_val.toBytesBE()[0..8] });
            return Entry{
                .cycle = even.cycle,
                .address = even.address / 2,
                .ra_coeff = one_minus_r.mul(even.ra_coeff), // (1-r)*ra_even + r*0
                .val_coeff = new_val,
                .prev_val = even.prev_val.add(r.mul(odd_checkpoint.sub(even.prev_val))),
                .next_val = even.next_val.add(r.mul(odd_checkpoint.sub(even.next_val))),
            };
        }

        /// Bind when only odd entry exists (even is implicit with ra=0, val=checkpoint)
        fn bindAddressMajorOddOnly(odd: *const Entry, even_checkpoint: F, odd_checkpoint: F, r: F) ?Entry {
            _ = odd_checkpoint;

            // Matching Jolt's (None, Some(odd)) case
            // Implicit even has ra=0, val=even_checkpoint, prev_val=even_checkpoint, next_val=even_checkpoint
            const new_val = even_checkpoint.add(r.mul(odd.val_coeff.sub(even_checkpoint)));
            dbg("[BIND ADDR] ODD_ONLY: addr={}, even_chkpt={any}, odd_val={any}, r={any}, result_val={any}\n", .{ odd.address, even_checkpoint.toBytesBE()[0..8], odd.val_coeff.toBytesBE()[0..8], r.toBytesBE()[0..8], new_val.toBytesBE()[0..8] });
            return Entry{
                .cycle = odd.cycle,
                .address = odd.address / 2,
                .ra_coeff = r.mul(odd.ra_coeff), // (1-r)*0 + r*ra_odd
                .val_coeff = new_val,
                .prev_val = even_checkpoint.add(r.mul(odd.prev_val.sub(even_checkpoint))),
                .next_val = even_checkpoint.add(r.mul(odd.next_val.sub(even_checkpoint))),
            };
        }

        /// Bind entries by grouping into row pairs, splitting into even/odd rows,
        /// and merge-joining by column (address).
        /// This matches Jolt's ReadWriteMatrixCycleMajor::bind which merges
        /// even_row and odd_row slices sorted by column.
        fn bindEntries(self: *Self, r: F) !void {
            const entries = self.entries.items;
            if (entries.len == 0) {
                return;
            }

            // Pass 1: Find row-pair group boundaries (sequential O(N) scan)
            var group_starts = std.ArrayListUnmanaged(usize).empty;
            defer group_starts.deinit(self.allocator);
            try group_starts.append(self.allocator, 0);
            for (1..entries.len) |i| {
                if (entries[i].cycle / 2 != entries[i - 1].cycle / 2) {
                    try group_starts.append(self.allocator, i);
                }
            }
            const num_groups = group_starts.items.len;

            // Compute output offsets: each group's upper bound = its input size
            const offsets = try self.allocator.alloc(usize, num_groups + 1);
            defer self.allocator.free(offsets);
            const actual_counts = try self.allocator.alloc(usize, num_groups);
            defer self.allocator.free(actual_counts);
            offsets[0] = 0;
            for (0..num_groups) |g| {
                const g_start = group_starts.items[g];
                const g_end = if (g + 1 < num_groups) group_starts.items[g + 1] else entries.len;
                offsets[g + 1] = offsets[g] + (g_end - g_start);
            }

            // Pre-allocate output buffer
            const output = try self.allocator.alloc(Entry, entries.len);

            // Pass 2: Parallel merge-join per group
            const BindGroupCtx = struct {
                entries_buf: []const Entry,
                output_buf: []Entry,
                group_starts_buf: []const usize,
                offsets_buf: []const usize,
                actual_counts_buf: []usize,
                num_entries: usize,
                num_groups_total: usize,
                challenge: F,
            };
            const bgctx = BindGroupCtx{
                .entries_buf = entries,
                .output_buf = output,
                .group_starts_buf = group_starts.items,
                .offsets_buf = offsets,
                .actual_counts_buf = actual_counts,
                .num_entries = entries.len,
                .num_groups_total = num_groups,
                .challenge = r,
            };

            const bindGroupFn = struct {
                fn f(c: BindGroupCtx, g: usize) void {
                    const g_start = c.group_starts_buf[g];
                    const g_end = if (g + 1 < c.num_groups_total) c.group_starts_buf[g + 1] else c.num_entries;
                    const pair_entries = c.entries_buf[g_start..g_end];
                    const out_start = c.offsets_buf[g];
                    const out_slice = c.output_buf[out_start .. out_start + pair_entries.len];

                    // Split into even-row and odd-row
                    var odd_start: usize = 0;
                    while (odd_start < pair_entries.len and pair_entries[odd_start].cycle % 2 == 0) {
                        odd_start += 1;
                    }
                    const even_row = pair_entries[0..odd_start];
                    const odd_row = pair_entries[odd_start..];

                    // Merge-join by address
                    var ei: usize = 0;
                    var oi: usize = 0;
                    var out_idx: usize = 0;
                    while (ei < even_row.len and oi < odd_row.len) {
                        if (even_row[ei].address == odd_row[oi].address) {
                            if (Entry.bindEntries(&even_row[ei], &odd_row[oi], c.challenge)) |bound| {
                                out_slice[out_idx] = bound;
                                out_idx += 1;
                            }
                            ei += 1;
                            oi += 1;
                        } else if (even_row[ei].address < odd_row[oi].address) {
                            if (Entry.bindEntries(&even_row[ei], null, c.challenge)) |bound| {
                                out_slice[out_idx] = bound;
                                out_idx += 1;
                            }
                            ei += 1;
                        } else {
                            if (Entry.bindEntries(null, &odd_row[oi], c.challenge)) |bound| {
                                out_slice[out_idx] = bound;
                                out_idx += 1;
                            }
                            oi += 1;
                        }
                    }
                    while (ei < even_row.len) : (ei += 1) {
                        if (Entry.bindEntries(&even_row[ei], null, c.challenge)) |bound| {
                            out_slice[out_idx] = bound;
                            out_idx += 1;
                        }
                    }
                    while (oi < odd_row.len) : (oi += 1) {
                        if (Entry.bindEntries(null, &odd_row[oi], c.challenge)) |bound| {
                            out_slice[out_idx] = bound;
                            out_idx += 1;
                        }
                    }
                    c.actual_counts_buf[g] = out_idx;
                }
            }.f;

            parallelForOptional(self.thread_pool, num_groups, bgctx, bindGroupFn);

            // Compact: move groups together (removing gaps from upper-bound slack)
            var total_out: usize = 0;
            for (0..num_groups) |g| {
                const src_start = offsets[g];
                const count = actual_counts[g];
                if (src_start != total_out and count > 0) {
                    std.mem.copyForwards(Entry, output[total_out .. total_out + count], output[src_start .. src_start + count]);
                }
                total_out += count;
            }

            // Replace old entries
            self.entries.deinit(self.allocator);
            self.entries = .empty;
            self.entries.items = output[0..total_out];
            self.entries.capacity = output.len;

            // Transfer allocator ownership manually — the output buffer is now owned by entries
            // (ArrayList.deinit will free it)

            dbg("[RWC BIND] round={}, entries.len after bind={}\n", .{ self.round, self.entries.items.len });
        }

        /// Update claim after evaluating polynomial at challenge
        pub fn updateClaim(self: *Self, evals: [4]F, challenge: F) void {
            const c = challenge;
            const c_minus_1 = c.sub(F.one());
            const c_minus_2 = c.sub(F.fromU64(2));
            const c_minus_3 = c.sub(F.fromU64(3));

            const neg6 = F.zero().sub(F.fromU64(6));
            const L0 = c_minus_1.mul(c_minus_2).mul(c_minus_3).mul(neg6.inverse().?);
            const L1 = c.mul(c_minus_2).mul(c_minus_3).mul(@import("zolt_arith").poly.UniPoly(F).INV2);
            const neg2 = F.zero().sub(F.fromU64(2));
            const L2 = c.mul(c_minus_1).mul(c_minus_3).mul(neg2.inverse().?);
            const L3 = c.mul(c_minus_1).mul(c_minus_2).mul(F.fromU64(6).inverse().?);

            self.current_claim = evals[0].mul(L0)
                .add(evals[1].mul(L1))
                .add(evals[2].mul(L2))
                .add(evals[3].mul(L3));
        }

        pub fn isComplete(self: *const Self) bool {
            return self.round >= self.params.numRounds();
        }

        pub fn getOpeningClaims(self: *const Self, r_sumcheck: []const F) OpeningClaims(F) {
            _ = r_sumcheck;

            // After all sumcheck rounds, the polynomials have been fully bound:
            // - ra and val: The entries have been progressively bound through Phase 1 (cycle vars)
            //   and Phase 2 (address vars). Each entry's ra_coeff and val_coeff now represent
            //   the fully-bound polynomial value.
            // - inc: Bound through Phase 1 into inc[0]
            // - val_init: Bound through Phase 2 into val_init[0]
            //
            // Following Jolt's approach: final_sumcheck_claim() just returns the single
            // remaining coefficient after all binding rounds.
            //
            // For ra: sum of all bound entries' ra_coeff (each entry contributes its bound weight)
            // For val: val_init[0] (base) + Σ entry contributions
            //   After full binding, val(r_addr, r_cycle) = val_init(r_addr) + Σ_entries bound_correction
            //   But the entries' val_coeff has also been bound, so we need to be careful.
            //
            // Actually, the simplest correct approach matching Jolt:
            // - ra_claim = sum of entry.ra_coeff for all remaining entries
            //   (after full binding, entries encode the bound polynomial evaluation)
            // - val_claim = val_init[0] + sum of entry.val_coeff for all remaining entries
            //   But this isn't quite right either, because val is the FULL val polynomial,
            //   not just the delta from val_init.
            //
            // The correct decomposition: After full binding of the sumcheck polynomial
            //   S(r) = eq(r_cycle_s1, r_cycle) * Σ_k ra(k, r_cycle) * (val(k, r_cycle) + gamma * (val(k, r_cycle) + inc(r_cycle)))
            //
            // The verifier expects:
            //   expected = eq_eval * ra_claim * (val_claim + gamma * (val_claim + inc_claim))
            //
            // So ra_claim should be the evaluation of ra at (r_addr, r_cycle),
            //    val_claim should be the evaluation of val at (r_addr, r_cycle),
            //    inc_claim should be the evaluation of inc at r_cycle.
            //
            // After binding, entries are at (addr=0, cycle=0) with bound coefficients.
            // The sumcheck binding folds pairs using: bound = (1-r)*even + r*odd
            // After all rounds, the entry's ra_coeff IS the evaluation ra(r_addr, r_cycle)
            // (because binding is equivalent to MLE evaluation).
            //
            // So: ra_claim = sum of all entries' ra_coeff (usually just one entry after binding)
            // And val has two components:
            //   val(r_addr, r_cycle) = val_init(r_addr) + Σ delta(k,j) * eq_addr(k) * eq_cycle(j)
            // After binding: val_init(r_addr) = val_init[0]
            // And the delta contributions are bound into the entries.
            //
            // For val_claim: The entry's val_coeff after binding represents the bound
            // contribution of the original (val_coeff - val_init[addr]) * eq factors.
            // So val_claim = val_init[0] + sum of bound entry val contributions.
            //
            // But actually, looking at Jolt more carefully: val is a single polynomial that gets
            // bound directly. It's not decomposed as val_init + delta. Jolt's val polynomial
            // is initialized as a sparse matrix and then materialized and bound.
            // After all binding: val.final_sumcheck_claim() = val[0] = val(r_addr, r_cycle).

            // After all sumcheck rounds, all variables are bound:
            // - ra: entries' ra_coeff has been bound through Phase 1 (cycle) and Phase 2 (address)
            // - val: entries' val_coeff has been bound independently (not multiplied by ra)
            //   During Phase 1, implicit entries use prev_val/next_val for interpolation.
            //   During Phase 2, implicit entries use val_init checkpoints.
            //   After full binding, entry.val_coeff = val(r_addr, r_cycle) = true polynomial evaluation.
            // - inc: bound through Phase 1 into inc[0]
            // - val_init: bound through Phase 2 into val_init[0]
            //
            // Following Jolt: val.final_sumcheck_claim() returns the single remaining coefficient
            // after all binding. In our sparse representation:
            // - If entries remain: entry.val_coeff IS the val evaluation (includes val_init via checkpoints)
            // - If no entries: val_claim = val_init[0] (the bound initial state)

            // ra_claim = sum of bound entry ra_coeffs (= ra(r_addr, r_cycle))
            var ra_claim = F.zero();
            for (self.entries.items) |entry| {
                ra_claim = ra_claim.add(entry.ra_coeff);
            }

            // inc_claim = bound inc[0] (after Phase 1 binding)
            const inc_claim = self.inc[0];

            // Following Jolt: val.final_sumcheck_claim() returns val.Z[0]
            // After full binding of all variables, the sparse entry's val_coeff
            // should be the evaluation val(r_addr, r_cycle).
            const eq_eval = self.eq_evals[0];
            const gamma = self.params.gamma;

            // val_claim from sparse binding (now correct with merge-join fix)
            var val_claim: F = self.val_init[0];
            if (self.entries.items.len > 0) {
                val_claim = F.zero();
                for (self.entries.items) |entry| {
                    val_claim = val_claim.add(entry.val_coeff);
                }
            }

            // Verify: current_claim should equal eq * ra * (val + gamma * (val + inc))
            const expected_claim = eq_eval.mul(ra_claim).mul(
                val_claim.add(gamma.mul(val_claim.add(inc_claim))),
            );

            dbg("[RWC GET_OPENING] ra_claim = {any}\n", .{ra_claim.toBytesBE()});
            dbg("[RWC GET_OPENING] val_claim (entry) = {any}\n", .{val_claim.toBytesBE()});
            dbg("[RWC GET_OPENING] inc_claim = {any}\n", .{inc_claim.toBytesBE()});
            dbg("[RWC GET_OPENING] eq_eval = {any}\n", .{eq_eval.toBytesBE()});
            dbg("[RWC GET_OPENING] gamma = {any}\n", .{gamma.toBytesBE()});
            dbg("[RWC GET_OPENING] val_init[0] = {any}\n", .{self.val_init[0].toBytesBE()});
            dbg("[RWC GET_OPENING] current_claim = {any}\n", .{self.current_claim.toBytesBE()});
            dbg("[RWC GET_OPENING] expected = eq*ra*(v+g*(v+i)) = {any}\n", .{expected_claim.toBytesBE()});
            dbg("[RWC GET_OPENING] MATCH = {}\n", .{expected_claim.eql(self.current_claim)});

            return OpeningClaims(F){
                .ra_claim = ra_claim,
                .val_claim = val_claim,
                .inc_claim = inc_claim,
            };
        }

        /// Compute val(r_addr, r_cycle) via dense evaluation using eq tables and suffix sums.
        ///
        /// The val polynomial is an MLE over (addr, cycle) dimensions where val[k][j] is the
        /// memory value at address k just before cycle j. Between writes the value is constant,
        /// so we decompose the evaluation as:
        ///
        ///   val(r_addr, r_cycle) = Σ_k eq_addr[k] * val_init[k]            (initial values)
        ///                        + Σ_writes eq_addr[k] * (new-old) * suffix_eq_cycle[c]  (deltas)
        ///
        /// where suffix_eq_cycle[c] = Σ_{j>=c} eq_cycle[j] accounts for the write at cycle c
        /// affecting all subsequent cycles.
        fn computeDenseValClaim(self: *const Self) !F {
            const poly_mod = @import("zolt_arith").poly;
            const EqPoly = poly_mod.EqPolynomial(F);

            const phase1_end = self.params.phase1_num_rounds;
            const r_cycle_le = self.challenges.items[0..phase1_end];
            const r_addr_le = self.challenges.items[phase1_end..];
            const K: usize = @as(usize, 1) << @intCast(self.params.log_k);
            const T: usize = @as(usize, 1) << @intCast(r_cycle_le.len);

            // buildEqTableInPlace: bit 0 of idx corresponds to r[n-1].
            // Binding convention: bit 0 of cycle/addr corresponds to challenge[0].
            // So we need r = [c_{n-1}, ..., c_0] (reversed from LE) to get eq[idx]
            // with bit 0 of idx = c_0.
            // But wait — for the val polynomial, the "bit 0" of the cycle timestamp is
            // what was bound in round 0. The cycle timestamp IS the raw RISC-V cycle number.
            // After binding round 0 with challenge c_0, entries at even cycles get weight (1-c_0)
            // and entries at odd cycles get weight c_0. So bit 0 of the timestamp corresponds to c_0.
            // Therefore eq_cycle[timestamp] should have: bit 0 = c_0.
            // With buildEqTableInPlace: bit 0 = r[n-1]. So r = [c_{n-1}, ..., c_0] → r[n-1] = c_0.
            // This means we need to REVERSE the LE challenges.
            const r_addr_rev = try self.allocator.alloc(F, r_addr_le.len);
            defer self.allocator.free(r_addr_rev);
            for (0..r_addr_le.len) |i| r_addr_rev[i] = r_addr_le[r_addr_le.len - 1 - i];

            const r_cycle_rev = try self.allocator.alloc(F, r_cycle_le.len);
            defer self.allocator.free(r_cycle_rev);
            for (0..r_cycle_le.len) |i| r_cycle_rev[i] = r_cycle_le[r_cycle_le.len - 1 - i];

            const eq_addr_evals = try EqPoly.evalsSliceWithScaling(F, self.allocator, r_addr_rev, null);
            defer self.allocator.free(eq_addr_evals);

            const eq_cycle_evals = try EqPoly.evalsSliceWithScaling(F, self.allocator, r_cycle_rev, null);
            defer self.allocator.free(eq_cycle_evals);

            // val_init contribution: Σ_k eq_addr[k] * val_init[k]
            // (Σ_j eq_cycle[j] = 1 by partition of unity, so val_init is weighted only by eq_addr)
            var val_init_contrib = F.zero();
            for (0..@min(K, self.val_init.len)) |k| {
                if (!self.val_init[k].eql(F.zero())) {
                    if (k < eq_addr_evals.len) {
                        val_init_contrib = val_init_contrib.add(eq_addr_evals[k].mul(self.val_init[k]));
                    }
                }
            }

            // Build suffix sums of eq_cycle: suffix[c] = Σ_{j=c}^{T-1} eq_cycle[j]
            const suffix_sums = try self.allocator.alloc(F, T + 1);
            defer self.allocator.free(suffix_sums);
            suffix_sums[T] = F.zero();
            var si: usize = T;
            while (si > 0) {
                si -= 1;
                suffix_sums[si] = suffix_sums[si + 1].add(
                    if (si < eq_cycle_evals.len) eq_cycle_evals[si] else F.zero(),
                );
            }

            // Write delta contribution: for each write at (addr k, cycle c) changing old→new,
            // the val polynomial changes by (new - old) for all cycles >= c at address k.
            // Contribution = eq_addr[k] * (new - old) * suffix_sums[c+1]
            // (the write AT cycle c uses old_val; cycles > c use new_val)
            var write_delta_contrib = F.zero();
            const trace = self.trace;
            for (trace.accesses.items) |access| {
                if (access.op != .Write) continue;
                if (access.address < self.params.start_address) continue;
                const k: usize = @intCast((access.address - self.params.start_address) / 8);
                if (k >= K) continue;
                const c: usize = @intCast(access.timestamp);
                if (c >= T) continue;

                const old_v = access.pre_value;
                const new_v = access.value;
                if (new_v == old_v) continue;

                const delta = if (new_v >= old_v)
                    F.fromU64(new_v - old_v)
                else
                    F.zero().sub(F.fromU64(old_v - new_v));

                const eq_k = if (k < eq_addr_evals.len) eq_addr_evals[k] else F.zero();
                write_delta_contrib = write_delta_contrib.add(eq_k.mul(delta).mul(suffix_sums[c + 1]));
            }

            return val_init_contrib.add(write_delta_contrib);
        }
    };
}

pub fn OpeningClaims(comptime F: type) type {
    return struct {
        ra_claim: F,
        val_claim: F,
        inc_claim: F,
    };
}

/// Compute eq(r, x) where r is in BIG_ENDIAN order (MSB first)
/// and x is a binary index
/// This matches Jolt's convention where tau is stored as [r_MSB, ..., r_LSB]
fn computeEqBigEndian(comptime F: type, r: []const F, x: usize, n: usize) F {
    return @import("../eq_utils.zig").computeEqAtPointBE(F, r[0..n], x);
}

/// Compute eq(r, x) for a binary index x
/// r is in BIG-ENDIAN order: r[0] is MSB, r[n-1] is LSB
fn computeEq(comptime F: type, r: []const F, x: usize) F {
    return @import("../eq_utils.zig").computeEqAtPointBE(F, r, x);
}

/// Remap address to index (matches logic from output_check.zig)
fn remapAddress(address: u64, memory_layout: *const @import("../jolt_device.zig").MemoryLayout, start_address: u64) ?usize {
    const lowest = memory_layout.getLowestAddress();
    if (address < lowest) return null;
    const offset = address - lowest;
    if (offset % 8 != 0) return null;
    const index = @as(usize, @intCast(offset / 8));

    // Validate that this index is within the RAM range starting at start_address
    if (lowest + offset < start_address) return null;

    return index;
}

test "ram read write checking prover initialization" {
    const allocator = std.testing.allocator;
    const field = @import("zolt_arith").field;
    const F = field.BN254Scalar;

    var trace = MemoryTrace.init(allocator);
    defer trace.deinit();

    try trace.recordWrite(0x80000000, 42, 0);
    try trace.recordRead(0x80000000, 42, 1);

    const r_cycle = [_]F{ F.fromU64(1), F.fromU64(2) };
    var params = try RamReadWriteCheckingParams(F).init(
        allocator,
        F.fromU64(12345),
        &r_cycle,
        4,
        2,
        0x80000000,
    );
    defer params.deinit();

    var prover = try RamReadWriteCheckingProver(F).init(
        allocator,
        &trace,
        params,
        F.fromU64(100),
        null,
        null, // memory_layout
        false, // is_panicking
    );
    defer prover.deinit();

    try std.testing.expect(!prover.isComplete());
}
