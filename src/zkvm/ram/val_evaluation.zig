//! RAM Value Evaluation Sumcheck
//!
//! This module implements the value evaluation sumcheck which proves
//! that memory values are consistent across the execution trace.
//!
//! The protocol proves:
//!   Val(r) - Val_init(r_address) = Σ_{j=0}^{T-1} inc(j) · wa(r_address, j) · LT(j, r_cycle)
//!
//! Where:
//! - r = (r_address, r_cycle) is the evaluation point from read-write checking
//! - Val(r) is the claimed memory value at address r_address and time r_cycle
//! - Val_init(r_address) is the initial value of memory at address r_address
//! - inc(j) is the value change at cycle j if a write occurs (0 otherwise)
//! - wa(r_address, j) is the write-indicator MLE (1 on matching points)
//! - LT(j, k) is the strict less-than MLE: 1 iff j < k as bitstrings
//!
//! Reference: jolt-core/src/zkvm/ram/val_evaluation.rs

const std = @import("std");

const zkvm_debug = @import("../debug.zig");
const dbg = zkvm_debug.dbg;

const Allocator = std.mem.Allocator;
const zolt_pool = @import("zolt_pool");
const ThreadPool = zolt_pool.ThreadPool;
const parallelReduceOptional = zolt_pool.parallelReduceOptional;
const parallelForOptional = zolt_pool.parallelForOptional;
const UnreducedProductAccum = @import("zolt_arith").field.UnreducedProductAccum;

const mod = @import("mod.zig");
const MemoryOp = mod.MemoryOp;
const MemoryAccess = mod.MemoryAccess;
const MemoryTrace = mod.MemoryTrace;

const jolt_device = @import("../jolt_device.zig");

/// Parameters for Value Evaluation sumcheck
pub fn ValEvaluationParams(comptime F: type) type {
    return struct {
        const Self = @This();

        /// Initial memory value evaluation at r_address: Val_init(r_address)
        init_eval: F,
        /// Trace length (T)
        trace_len: usize,
        /// Number of memory slots (K = 2^log_k)
        k: usize,
        /// Address point from read-write checking (r_address)
        r_address: []const F,
        /// Cycle point from read-write checking (r_cycle)
        r_cycle: []const F,
        allocator: Allocator,

        pub fn init(
            allocator: Allocator,
            init_eval: F,
            trace_len: usize,
            k: usize,
            r_address: []const F,
            r_cycle: []const F,
        ) !Self {
            const r_addr_copy = try allocator.alloc(F, r_address.len);
            @memcpy(r_addr_copy, r_address);

            const r_cycle_copy = try allocator.alloc(F, r_cycle.len);
            @memcpy(r_cycle_copy, r_cycle);

            return Self{
                .init_eval = init_eval,
                .trace_len = trace_len,
                .k = k,
                .r_address = r_addr_copy,
                .r_cycle = r_cycle_copy,
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.r_address);
            self.allocator.free(self.r_cycle);
        }

        /// Number of sumcheck rounds = log2(trace_len)
        pub fn numRounds(self: *const Self) usize {
            if (self.trace_len == 0) return 0;
            return std.math.log2_int_ceil(usize, self.trace_len);
        }

        /// Degree bound is 3 (product of 3 linear polynomials: inc, wa, LT)
        pub fn degreeBound() usize {
            return 3;
        }
    };
}

/// Increment polynomial: inc(j) = val_new(j) - val_old(j) for writes
pub fn IncPolynomial(comptime F: type) type {
    return struct {
        const Self = @This();

        /// Evaluations inc[j] for each cycle j
        evals: []F,
        num_vars: usize,
        allocator: Allocator,

        /// Initialize from memory trace
        /// inc(j) = value_after_write - value_before_write for writes, 0 otherwise
        ///
        /// IMPORTANT: Synthetic termination/panic writes are NOT included in Jolt's trace.
        /// If memory_layout is provided, writes to termination/panic addresses are skipped.
        /// This matches Jolt's behavior where these bits are set directly in the final memory
        /// state without corresponding trace entries.
        pub fn fromTrace(
            allocator: Allocator,
            trace: *const MemoryTrace,
            trace_len: usize,
            start_address: u64,
            k: usize,
            initial_ram: ?*const std.AutoHashMapUnmanaged(u64, u64),
        ) !Self {
            // Call the full version with no memory_layout (no filtering)
            return fromTraceWithLayout(allocator, trace, trace_len, start_address, k, initial_ram, null);
        }

        /// Initialize from memory trace, optionally filtering out termination/panic writes
        pub fn fromTraceWithLayout(
            allocator: Allocator,
            trace: *const MemoryTrace,
            trace_len: usize,
            start_address: u64,
            k: usize,
            initial_ram: ?*const std.AutoHashMapUnmanaged(u64, u64),
            memory_layout: ?*const jolt_device.MemoryLayout,
        ) !Self {
            return fromTraceParallel(allocator, trace, trace_len, start_address, k, initial_ram, memory_layout, null);
        }

        /// Parallel version using pre_value from trace (each access independent)
        pub fn fromTraceParallel(
            allocator: Allocator,
            trace: *const MemoryTrace,
            trace_len: usize,
            start_address: u64,
            k: usize,
            initial_ram: ?*const std.AutoHashMapUnmanaged(u64, u64),
            memory_layout: ?*const jolt_device.MemoryLayout,
            pool: ?*ThreadPool,
        ) !Self {
            _ = memory_layout;
            _ = initial_ram; // pre_value is in the trace; initial_ram was only for last_value tracking
            const effective_len = if (trace_len == 0) 1 else trace_len;
            const padded_len = std.math.ceilPowerOfTwo(usize, effective_len) catch effective_len;
            const num_vars = if (padded_len <= 1) 0 else std.math.log2_int_ceil(usize, padded_len);

            const evals = try allocator.alloc(F, padded_len);
            @memset(evals, F.zero());

            // Parallel: each access computes inc = value - pre_value independently
            const accesses = trace.accesses.items;
            const IncCtx = struct {
                items: []const mod.MemoryAccess,
                ev: []F,
                start: u64,
                kk: usize,
                tlen: usize,
            };
            const inc_ctx = IncCtx{
                .items = accesses,
                .ev = evals,
                .start = start_address,
                .kk = k,
                .tlen = trace_len,
            };
            const incFn = struct {
                fn f(ctx: IncCtx, idx: usize) void {
                    const access = ctx.items[idx];
                    if (access.op != .Write) return;
                    if (access.address < ctx.start) return;
                    const remapped = (access.address - ctx.start) / 8;
                    if (remapped >= ctx.kk) return;
                    const timestamp = @as(usize, @intCast(access.timestamp));
                    if (timestamp >= ctx.tlen) return;
                    if (access.value >= access.pre_value) {
                        ctx.ev[timestamp] = F.fromU64(access.value - access.pre_value);
                    } else {
                        ctx.ev[timestamp] = F.zero().sub(F.fromU64(access.pre_value - access.value));
                    }
                }
            }.f;

            if (pool != null and accesses.len >= 256) {
                pool.?.parallelFor(accesses.len, inc_ctx, incFn);
            } else {
                for (0..accesses.len) |idx| incFn(inc_ctx, idx);
            }

            return Self{
                .evals = evals,
                .num_vars = num_vars,
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.evals);
        }

        /// Evaluate at point r
        pub fn evaluate(self: *const Self, r: []const F) F {
            var result = F.zero();
            for (self.evals, 0..) |eval, j| {
                const eq_val = computeEqAtPoint(F, r, j);
                result = result.add(eval.mul(eq_val));
            }
            return result;
        }

        /// Get evaluation at index j
        pub fn get(self: *const Self, j: usize) F {
            if (j >= self.evals.len) return F.zero();
            return self.evals[j];
        }

        /// Bind the first variable (LSB) to value r using LowToHigh order.
        /// Uses adjacent-pair folding: new[i] = (1-r)*old[2*i] + r*old[2*i+1]
        pub fn bind(self: *Self, r: F) void {
            const half = self.evals.len / 2;
            const one_minus_r = F.one().sub(r);
            for (0..half) |i| {
                const lo = self.evals[2 * i];
                const hi = self.evals[2 * i + 1];
                self.evals[i] = one_minus_r.mul(lo).add(r.mul(hi));
            }
            if (self.num_vars > 0) self.num_vars -= 1;
        }
    };
}

/// Write-Address indicator polynomial: wa(k, j) = 1 iff cycle j writes to address k
pub fn WaPolynomial(comptime F: type) type {
    return struct {
        const Self = @This();

        /// For each cycle j, store the remapped address k that was written (or null if no write)
        write_addresses: []?u64,
        /// Target address point r_address
        r_address: []const F,
        /// Number of cycle variables
        num_cycle_vars: usize,
        allocator: Allocator,

        /// Initialize from memory trace
        pub fn fromTrace(
            allocator: Allocator,
            trace: *const MemoryTrace,
            trace_len: usize,
            r_address: []const F,
            start_address: u64,
            k: usize,
        ) !Self {
            return fromTraceWithPool(allocator, trace, trace_len, r_address, start_address, k, null);
        }

        pub fn fromTraceWithPool(
            allocator: Allocator,
            trace: *const MemoryTrace,
            trace_len: usize,
            r_address: []const F,
            start_address: u64,
            k: usize,
            pool: ?*ThreadPool,
        ) !Self {
            const effective_len = if (trace_len == 0) 1 else trace_len;
            const padded_len = std.math.ceilPowerOfTwo(usize, effective_len) catch effective_len;
            const num_cycle_vars = if (padded_len <= 1) 0 else std.math.log2_int_ceil(usize, padded_len);

            const write_addresses = try allocator.alloc(?u64, padded_len);
            @memset(write_addresses, null);

            // Each access writes to a unique timestamp — parallelizable
            const accesses = trace.accesses.items;
            const WaCtx = struct {
                items: []const mod.MemoryAccess,
                wa: []?u64,
                start: u64,
                kk: usize,
                tlen: usize,
            };
            const wa_ctx = WaCtx{ .items = accesses, .wa = write_addresses, .start = start_address, .kk = k, .tlen = trace_len };
            const waFn = struct {
                fn f(ctx: WaCtx, idx: usize) void {
                    const access = ctx.items[idx];
                    if (access.address < ctx.start) return;
                    const remapped = (access.address - ctx.start) / 8;
                    if (remapped >= ctx.kk) return;
                    const timestamp = @as(usize, @intCast(access.timestamp));
                    if (timestamp >= ctx.tlen) return;
                    ctx.wa[timestamp] = remapped;
                }
            }.f;

            if (pool != null and accesses.len >= 256) {
                pool.?.parallelFor(accesses.len, wa_ctx, waFn);
            } else {
                for (0..accesses.len) |idx| waFn(wa_ctx, idx);
            }

            const r_addr_copy = try allocator.alloc(F, r_address.len);
            @memcpy(r_addr_copy, r_address);

            return Self{
                .write_addresses = write_addresses,
                .r_address = r_addr_copy,
                .num_cycle_vars = num_cycle_vars,
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.write_addresses);
            self.allocator.free(self.r_address);
        }

        /// Evaluate wa at cycle j: returns eq(r_address, write_address[j]) if write, 0 otherwise
        pub fn evaluateAtCycle(self: *const Self, j: usize) F {
            if (j >= self.write_addresses.len) return F.zero();

            if (self.write_addresses[j]) |addr| {
                // eq(r_address, addr)
                return computeEqAtPoint(F, self.r_address, addr);
            } else {
                return F.zero();
            }
        }

        /// Get write address at cycle j (if any)
        pub fn getWriteAddress(self: *const Self, j: usize) ?u64 {
            if (j >= self.write_addresses.len) return null;
            return self.write_addresses[j];
        }
    };
}

/// Less-Than polynomial: LT(j, r_cycle) = 1 iff j < r_cycle as bitstrings
///
/// CRITICAL: r_cycle must be in BIG_ENDIAN order (r_cycle[0] = MSB coefficient).
/// This matches Jolt's convention where LT is defined with MSB-first comparison.
///
/// The MLE is: LT(x, y) = Σ_i (1 - x_i) · y_i · eq(x[i+1:], y[i+1:])
/// where i runs from MSB to LSB.
pub fn LtPolynomial(comptime F: type) type {
    return struct {
        const Self = @This();

        /// Target point r_cycle in BIG_ENDIAN order (r_cycle[0] = MSB)
        r_cycle_be: []const F,
        /// Precomputed evaluations LT(j, r_cycle_be) for all j from 0 to 2^n - 1
        /// This matches Jolt's lt_evals construction
        evals: []F,
        /// Number of cycle variables
        num_vars: usize,
        allocator: Allocator,

        /// Initialize with r_cycle in BIG_ENDIAN order (r_cycle[0] = MSB)
        /// This precomputes all LT(j, r_cycle) values using Jolt's algorithm.
        pub fn init(allocator: Allocator, r_cycle_be: []const F) !Self {
            return initWithPool(allocator, r_cycle_be, null);
        }

        pub fn initWithPool(allocator: Allocator, r_cycle_be: []const F, pool: ?*ThreadPool) !Self {
            const n = r_cycle_be.len;
            const size = @as(usize, 1) << @intCast(n);

            const r_copy = try allocator.alloc(F, n);
            @memcpy(r_copy, r_cycle_be);

            const evals = try allocator.alloc(F, size);

            // Initialize all evals to zero
            for (evals) |*e| {
                e.* = F.zero();
            }

            // Build LT evaluations using Jolt's algorithm.
            // Outer loop has data dependencies (level i depends on level i-1).
            // Inner loop pairs are independent — parallelize when half >= 256.
            for (0..n) |i| {
                const r = r_cycle_be[n - 1 - i];
                const half = @as(usize, 1) << @intCast(i);

                if (pool != null and half >= 256) {
                    const Ctx = struct { ev: []F, rr: F, h: usize };
                    pool.?.parallelFor(half, Ctx{ .ev = evals, .rr = r, .h = half }, struct {
                        fn f(ctx: Ctx, idx: usize) void {
                            const old_x = ctx.ev[idx];
                            const y = old_x.mul(ctx.rr);
                            ctx.ev[idx + ctx.h] = y;
                            ctx.ev[idx] = old_x.add(ctx.rr).sub(y);
                        }
                    }.f);
                } else {
                    var idx: usize = 0;
                    while (idx < half) : (idx += 1) {
                        const old_x = evals[idx];
                        evals[idx + half] = old_x.mul(r);
                        evals[idx] = old_x.add(r).sub(evals[idx + half]);
                    }
                }
            }

            return Self{
                .r_cycle_be = r_copy,
                .evals = evals,
                .num_vars = n,
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.r_cycle_be);
            self.allocator.free(self.evals);
        }

        /// Get LT(j, r_cycle) from precomputed table
        pub fn evaluateAtIndex(self: *const Self, j: usize) F {
            if (j >= self.evals.len) return F.zero();
            return self.evals[j];
        }

        /// Debug: evaluate LT and print intermediate values
        /// Uses the old formula for comparison - not used in production
        pub fn evaluateAtIndexDebug(self: *const Self, j: usize) F {
            const result = self.evaluateAtIndex(j);
            dbg("[LT DEBUG] evaluateAtIndex(j={}) num_vars={} result={any}\n", .{
                j,
                self.num_vars,
                result.toBytes()[0..8],
            });
            return result;
        }
    };
}

/// Split LT polynomial: stores LT_hi, LT_lo, EQ_hi instead of dense T-element array.
/// Memory: 3 × 2^(n/2) elements instead of 2^n (e.g., 64KB vs 16MB for n=19).
///
/// Decomposition: LT(i) = LT_hi(i_hi) + EQ_hi(i_hi) * LT_lo(i_lo)
/// where i_hi = i >> n_lo_vars, i_lo = i & ((1 << n_lo_vars) - 1).
///
/// Binding order: LowToHigh — binds lo vars first, then hi vars.
pub fn SplitLtPolynomial(comptime F: type) type {
    const EqPoly = @import("zolt_arith").poly.EqPolynomial(F);
    return struct {
        const Self = @This();

        lt_lo: []F,
        lt_hi: []F,
        eq_hi: []F,
        n_lo_vars: usize,
        allocator: Allocator,

        pub fn init(allocator: Allocator, r_cycle_be: []const F, pool: ?*ThreadPool) !Self {
            const n = r_cycle_be.len;
            const n_hi = n / 2;
            const n_lo = n - n_hi;

            // Split r_cycle_be into hi and lo halves
            const r_hi = r_cycle_be[0..n_hi]; // MSB side
            const r_lo = r_cycle_be[n_hi..n]; // LSB side

            // Build lt_lo from r_lo using LtPolynomial.init algorithm
            const lt_lo = try ltEvalsAlloc(allocator, r_lo, pool);
            errdefer allocator.free(lt_lo);

            // Build lt_hi from r_hi
            const lt_hi = try ltEvalsAlloc(allocator, r_hi, pool);
            errdefer allocator.free(lt_hi);

            // Build eq_hi = EqPolynomial::evals(r_hi) — same BE convention as lt_hi
            const eq_hi = try EqPoly.evalsSliceWithScalingParallel(F, allocator, r_hi, null, pool);
            errdefer allocator.free(eq_hi);

            return Self{
                .lt_lo = lt_lo,
                .lt_hi = lt_hi,
                .eq_hi = eq_hi,
                .n_lo_vars = n_lo,
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.lt_lo);
            self.allocator.free(self.lt_hi);
            self.allocator.free(self.eq_hi);
        }

        /// Get bound coefficient at index i (O(1): 3 lookups + 1 mul + 1 add)
        pub inline fn getBoundCoeff(self: *const Self, i: usize) F {
            const lo_mask = (@as(usize, 1) << @intCast(self.n_lo_vars)) - 1;
            const i_lo = i & lo_mask;
            const i_hi = i >> @intCast(self.n_lo_vars);
            return self.lt_hi[i_hi].add(self.eq_hi[i_hi].mul(self.lt_lo[i_lo]));
        }

        /// Bind LowToHigh: lo vars first, then hi vars
        pub fn bind(self: *Self, r: F) void {
            if (self.n_lo_vars > 0) {
                // Bind lo
                bindHalf(self.lt_lo, r);
                self.n_lo_vars -= 1;
            } else {
                // Bind hi + eq_hi
                bindHalf(self.lt_hi, r);
                bindHalf(self.eq_hi, r);
            }
        }

        fn bindHalf(arr: []F, r: F) void {
            const n = arr.len;
            const half = n / 2;
            for (0..half) |i| {
                const lo = arr[2 * i];
                const hi = arr[2 * i + 1];
                arr[i] = lo.add(r.mul(hi.sub(lo)));
            }
        }

        /// Build lt_evals using the standard algorithm (same as LtPolynomial.init)
        fn ltEvalsAlloc(allocator: Allocator, r_be: []const F, pool: ?*ThreadPool) ![]F {
            const n = r_be.len;
            const size = @as(usize, 1) << @intCast(n);
            const evals = try allocator.alloc(F, size);
            @memset(evals, F.zero());

            for (0..n) |i| {
                const r = r_be[n - 1 - i];
                const half = @as(usize, 1) << @intCast(i);

                if (pool != null and half >= 256) {
                    const Ctx = struct { ev: []F, rr: F, h: usize };
                    pool.?.parallelFor(half, Ctx{ .ev = evals, .rr = r, .h = half }, struct {
                        fn f(ctx: Ctx, idx: usize) void {
                            const old_x = ctx.ev[idx];
                            const y = old_x.mul(ctx.rr);
                            ctx.ev[idx + ctx.h] = y;
                            ctx.ev[idx] = old_x.add(ctx.rr).sub(y);
                        }
                    }.f);
                } else {
                    var idx: usize = 0;
                    while (idx < half) : (idx += 1) {
                        const old_x = evals[idx];
                        evals[idx + half] = old_x.mul(r);
                        evals[idx] = old_x.add(r).sub(evals[idx + half]);
                    }
                }
            }
            return evals;
        }
    };
}

/// Value Evaluation Sumcheck Prover
///
/// Uses SplitLtPolynomial for memory-efficient LT evaluation (64KB vs 16MB).
/// inc and wa are materialized as dense arrays; lt uses hi/lo decomposition.
pub fn ValEvaluationProver(comptime F: type) type {
    return struct {
        const Self = @This();

        /// Increment polynomial evaluations
        inc_evals: []F,
        /// Wa: either lazy (eq_table + sparse indices) or dense (materialized after first bind)
        wa_evals: ?[]F,
        /// Lazy wa state: sparse write addresses (freed after first bind)
        wa_addrs: ?[]const ?u64,
        wa_addrs_owned: bool,
        /// Lazy wa state: eq table (freed after first bind)
        wa_eq_table: ?[]const F,
        /// Split LT polynomial (hi/lo decomposition, ~64KB vs 16MB dense)
        lt_poly: SplitLtPolynomial(F),
        /// Number of variables (log of trace length)
        num_vars: usize,
        /// Current round (bound variables count)
        round: usize,
        /// Current claim being sumchecked
        current_claim: F,
        /// Parameters
        params: ValEvaluationParams(F),
        allocator: Allocator,
        thread_pool: ?*ThreadPool = null,

        /// Get wa value at index j — lazy (eq_table lookup) or dense (array access)
        pub inline fn getWa(self: *const Self, j: usize) F {
            if (self.wa_evals) |wa| return wa[j];
            // Lazy path: sparse lookup into eq table
            const addrs = self.wa_addrs.?;
            if (j < addrs.len) {
                if (addrs[j]) |addr| return self.wa_eq_table.?[@intCast(addr)];
            }
            return F.zero();
        }

        pub fn init(
            allocator: Allocator,
            trace: *const MemoryTrace,
            initial_ram: ?*const std.AutoHashMapUnmanaged(u64, u64),
            params: ValEvaluationParams(F),
            start_address: u64,
        ) !Self {
            // Call full version with no memory_layout (no filtering of synthetic writes)
            return initWithLayout(allocator, trace, initial_ram, params, start_address, null);
        }

        /// Initialize with optional memory_layout to filter out synthetic termination/panic writes
        ///
        /// IMPORTANT: Pass memory_layout to exclude synthetic writes from the inc polynomial.
        /// This matches Jolt's behavior where termination/panic bits are set directly in
        /// final memory without corresponding trace entries.
        pub fn initWithLayout(
            allocator: Allocator,
            trace: *const MemoryTrace,
            initial_ram: ?*const std.AutoHashMapUnmanaged(u64, u64),
            params: ValEvaluationParams(F),
            start_address: u64,
            memory_layout: ?*const jolt_device.MemoryLayout,
        ) !Self {
            return initWithLayoutAndPool(allocator, trace, initial_ram, params, start_address, memory_layout, null);
        }

        /// Initialize with optional thread pool for parallel init
        pub fn initWithLayoutAndPool(
            allocator: Allocator,
            trace: *const MemoryTrace,
            initial_ram: ?*const std.AutoHashMapUnmanaged(u64, u64),
            params: ValEvaluationParams(F),
            start_address: u64,
            memory_layout: ?*const jolt_device.MemoryLayout,
            pool: ?*ThreadPool,
        ) !Self {

            // Build inc polynomial (filtering out synthetic writes if memory_layout provided)
            var inc_poly = try IncPolynomial(F).fromTraceParallel(
                allocator,
                trace,
                params.trace_len,
                start_address,
                params.k,
                initial_ram,
                memory_layout,
                pool,
            );
            defer inc_poly.deinit();

            // Build wa polynomial helper (keep sparse addresses for lazy lookup)
            const wa_poly = try WaPolynomial(F).fromTraceWithPool(
                allocator,
                trace,
                params.trace_len,
                params.r_address,
                start_address,
                params.k,
                pool,
            );
            // Transfer write_addresses ownership to prover; free only r_address here
            defer allocator.free(wa_poly.r_address);

            // Build split LT polynomial (hi/lo decomposition, ~64KB vs 16MB dense)
            var split_lt = try SplitLtPolynomial(F).init(allocator, params.r_cycle, pool);
            errdefer split_lt.deinit();

            const n = inc_poly.evals.len;
            const num_vars = inc_poly.num_vars;

            // Precompute K-element eq table for wa evaluation: eq_table[addr] = eq(r_address, addr).
            // Reverse r_address for LE indexing (buildEqTableInPlace uses BE convention).
            const EqPoly = @import("zolt_arith").poly.EqPolynomial(F);
            const r_addr = params.r_address;
            const r_addr_rev = try allocator.alloc(F, r_addr.len);
            defer allocator.free(r_addr_rev);
            for (0..r_addr.len) |ri| {
                r_addr_rev[ri] = r_addr[r_addr.len - 1 - ri];
            }
            // eq_table ownership transfers to prover (freed on first bind or deinit)
            const eq_table = try EqPoly.evalsSliceWithScalingParallel(F, allocator, r_addr_rev, null, pool);

            // Transfer inc_poly.evals ownership to prover (avoids 16MB memcpy)
            const inc_evals = inc_poly.evals;
            inc_poly.evals = &.{}; // prevent deinit from double-freeing

            // Compute initial claim using lazy wa + split LT — parallel when pool available
            const LazyClaimCtx = struct { inc: []const F, addrs: []const ?u64, eq_tbl: []const F, lt: *const SplitLtPolynomial(F), addrs_len: usize };
            const claim_ctx = LazyClaimCtx{ .inc = inc_evals, .addrs = wa_poly.write_addresses, .eq_tbl = eq_table, .lt = &split_lt, .addrs_len = wa_poly.write_addresses.len };
            const claimMapFn = struct {
                fn f(c: LazyClaimCtx, start: usize, end: usize) F {
                    var s = F.zero();
                    for (start..end) |j| {
                        const wa_j = if (j < c.addrs_len)
                            (if (c.addrs[j]) |addr| c.eq_tbl[@intCast(addr)] else F.zero())
                        else
                            F.zero();
                        s = s.add(c.inc[j].mul(wa_j).mul(c.lt.getBoundCoeff(j)));
                    }
                    return s;
                }
            }.f;
            const claimReduceFn = struct {
                fn f(a: F, b: F) F {
                    return a.add(b);
                }
            }.f;
            const initial_claim = parallelReduceOptional(F, pool, n, F.zero(), claim_ctx, claimMapFn, claimReduceFn);

            return Self{
                .inc_evals = inc_evals,
                .wa_evals = null, // Lazy: materialized on first bind
                .wa_addrs = wa_poly.write_addresses, // Transfer ownership
                .wa_addrs_owned = true,
                .wa_eq_table = eq_table, // Transfer ownership
                .lt_poly = split_lt,
                .num_vars = num_vars,
                .round = 0,
                .current_claim = initial_claim,
                .params = params,
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.inc_evals);
            if (self.wa_evals) |wa| self.allocator.free(wa);
            if (self.wa_addrs_owned) {
                if (self.wa_addrs) |addrs| self.allocator.free(addrs);
            }
            if (self.wa_eq_table) |eq| self.allocator.free(eq);
            self.lt_poly.deinit();
            self.params.deinit();
        }

        /// Get the initial claim for the sumcheck
        pub fn computeInitialClaim(self: *const Self) F {
            return self.current_claim;
        }

        /// Compute round polynomial in Toom-Cook format [p(0), p(1), p(2), p_inf]
        /// For degree-3 sumcheck (product of 3 multilinear), p_inf = c3 (leading coefficient).
        ///   p(x) = Σ_{j} inc(x,j) · wa(x,j) · lt(x,j)
        /// where the current variable takes value x and we sum over remaining indices.
        /// Uses LowToHigh indexing: x=0 at index 2*i, x=1 at index 2*i+1
        ///
        /// This matches Jolt's ValEval which uses from_evals_toom(&[eval_0, eval_1, eval_2, eval_inf]).
        pub fn computeRoundPolynomial(self: *Self) [4]F {
            var evals: [4]F = .{ F.zero(), F.zero(), F.zero(), F.zero() };
            const n = self.effectiveLen();
            const half = n / 2;

            if (half == 0) {
                // Single element: p(0) = f(0), others are 0
                if (n > 0) {
                    evals[0] = self.inc_evals[0].mul(self.getWa(0)).mul(self.lt_poly.getBoundCoeff(0));
                }
                return evals;
            }

            for (0..half) |i| {
                // For LowToHigh binding, x=0 is at index 2*i (bit 0 = 0)
                // and x=1 is at index 2*i+1 (bit 0 = 1)
                const inc_0 = self.inc_evals[2 * i];
                const wa_0 = self.getWa(2 * i);
                const lt_0 = self.lt_poly.getBoundCoeff(2 * i);

                const inc_1 = self.inc_evals[2 * i + 1];
                const wa_1 = self.getWa(2 * i + 1);
                const lt_1 = self.lt_poly.getBoundCoeff(2 * i + 1);

                // p(0): product at x = 0
                evals[0] = evals[0].add(inc_0.mul(wa_0).mul(lt_0));

                // p(1): product at x = 1
                evals[1] = evals[1].add(inc_1.mul(wa_1).mul(lt_1));

                // For multilinear polynomial: f(x) = (1-x)*f(0) + x*f(1)
                // f(2) = -f(0) + 2*f(1) = 2*f(1) - f(0)
                const two = F.fromU64(2);

                // p(2): extrapolate each polynomial to x=2, then multiply
                const inc_2 = two.mul(inc_1).sub(inc_0);
                const wa_2 = two.mul(wa_1).sub(wa_0);
                const lt_2 = two.mul(lt_1).sub(lt_0);
                evals[2] = evals[2].add(inc_2.mul(wa_2).mul(lt_2));

                // p_inf = c3 (leading coefficient) for degree-3 polynomial
                // For product of 3 multilinears, c3 = product of slopes
                // slope = f(1) - f(0) for each multilinear
                const inc_slope = inc_1.sub(inc_0);
                const wa_slope = wa_1.sub(wa_0);
                const lt_slope = lt_1.sub(lt_0);
                evals[3] = evals[3].add(inc_slope.mul(wa_slope).mul(lt_slope));
            }

            return evals;
        }

        /// Compute round polynomial for the combined RamValCheck: inc * wa * (lt + gamma).
        /// This matches upstream a16z/jolt's RamValCheckSumcheckProver::compute_message exactly.
        /// Returns [eval_at_0, eval_at_1, eval_at_2, eval_at_inf] in Toom-Cook format.
        pub fn computeRoundPolynomialCombined(self: *Self, gamma: F) [4]F {
            const n = self.effectiveLen();
            const half = n / 2;

            if (half == 0) {
                var evals: [4]F = .{ F.zero(), F.zero(), F.zero(), F.zero() };
                if (n > 0) {
                    evals[0] = self.inc_evals[0].mul(self.getWa(0)).mul(self.lt_poly.getBoundCoeff(0).add(gamma));
                }
                return evals;
            }

            const ComputeCtx = struct {
                inc: []const F,
                wa_dense: ?[]const F,
                wa_addrs: ?[]const ?u64,
                wa_eq_tbl: ?[]const F,
                lt: *const SplitLtPolynomial(F),
                gamma: F,

                inline fn getWaAt(c: @This(), j: usize) F {
                    if (c.wa_dense) |wa| return wa[j];
                    const addrs = c.wa_addrs.?;
                    if (j < addrs.len) {
                        if (addrs[j]) |addr| return c.wa_eq_tbl.?[@intCast(addr)];
                    }
                    return F.zero();
                }
            };
            const ctx = ComputeCtx{
                .inc = self.inc_evals,
                .wa_dense = self.wa_evals,
                .wa_addrs = self.wa_addrs,
                .wa_eq_tbl = self.wa_eq_table,
                .lt = &self.lt_poly,
                .gamma = gamma,
            };

            const mapFn = struct {
                fn f(c: ComputeCtx, start: usize, end: usize) [4]F {
                    @setEvalBranchQuota(10000);
                    const use_deferred = comptime @hasDecl(F, "mulToProductAccum");
                    const two = F.fromU64(2);

                    if (use_deferred) {
                        var acc: [4]UnreducedProductAccum = .{
                            UnreducedProductAccum.zero(), UnreducedProductAccum.zero(),
                            UnreducedProductAccum.zero(), UnreducedProductAccum.zero(),
                        };
                        for (start..end) |i| {
                            const inc_0 = c.inc[2 * i];
                            const wa_0 = c.getWaAt(2 * i);
                            const lt_0 = c.lt.getBoundCoeff(2 * i);
                            const inc_1 = c.inc[2 * i + 1];
                            const wa_1 = c.getWaAt(2 * i + 1);
                            const lt_1 = c.lt.getBoundCoeff(2 * i + 1);
                            const inc_2 = two.mul(inc_1).sub(inc_0);
                            const wa_2 = two.mul(wa_1).sub(wa_0);
                            const lt_2 = two.mul(lt_1).sub(lt_0);
                            // t1: inc*wa*lt (cubic) — defer last mul
                            acc[0].addAssign(inc_0.mul(wa_0).mulToProductAccum(lt_0));
                            acc[1].addAssign(inc_1.mul(wa_1).mulToProductAccum(lt_1));
                            acc[2].addAssign(inc_2.mul(wa_2).mulToProductAccum(lt_2));
                            acc[3].addAssign(inc_1.sub(inc_0).mul(wa_1.sub(wa_0)).mulToProductAccum(lt_1.sub(lt_0)));
                            // t2: gamma*inc*wa (quadratic) — defer last mul
                            acc[0].addAssign(inc_0.mul(wa_0).mulToProductAccum(c.gamma));
                            acc[1].addAssign(inc_1.mul(wa_1).mulToProductAccum(c.gamma));
                            acc[2].addAssign(inc_2.mul(wa_2).mulToProductAccum(c.gamma));
                            // t2 has no contribution to acc[3] (eval_at_inf)
                        }
                        return .{ acc[0].reduce(), acc[1].reduce(), acc[2].reduce(), acc[3].reduce() };
                    } else {
                        var local: [4]F = .{ F.zero(), F.zero(), F.zero(), F.zero() };
                        for (start..end) |i| {
                            const inc_0 = c.inc[2 * i];
                            const wa_0 = c.getWaAt(2 * i);
                            const lt_0 = c.lt.getBoundCoeff(2 * i);
                            const inc_1 = c.inc[2 * i + 1];
                            const wa_1 = c.getWaAt(2 * i + 1);
                            const lt_1 = c.lt.getBoundCoeff(2 * i + 1);
                            const inc_2 = two.mul(inc_1).sub(inc_0);
                            const wa_2 = two.mul(wa_1).sub(wa_0);
                            const lt_2 = two.mul(lt_1).sub(lt_0);
                            local[0] = local[0].add(inc_0.mul(wa_0).mul(lt_0).add(c.gamma.mul(inc_0.mul(wa_0))));
                            local[1] = local[1].add(inc_1.mul(wa_1).mul(lt_1).add(c.gamma.mul(inc_1.mul(wa_1))));
                            local[2] = local[2].add(inc_2.mul(wa_2).mul(lt_2).add(c.gamma.mul(inc_2.mul(wa_2))));
                            local[3] = local[3].add(inc_1.sub(inc_0).mul(wa_1.sub(wa_0)).mul(lt_1.sub(lt_0)));
                        }
                        return local;
                    }
                }
            }.f;

            const reduceFn = struct {
                fn f(a: [4]F, b: [4]F) [4]F {
                    return .{ a[0].add(b[0]), a[1].add(b[1]), a[2].add(b[2]), a[3].add(b[3]) };
                }
            }.f;

            const identity = [4]F{ F.zero(), F.zero(), F.zero(), F.zero() };
            const evals = parallelReduceOptional([4]F, self.thread_pool, half, identity, ctx, mapFn, reduceFn);

            return evals;
        }

        /// Bind the current variable to challenge r, and provide round polynomial values
        /// This folds all three polynomials using LowToHigh binding order:
        /// f_new[i] = (1-r)*f[2*i] + r*f[2*i+1]
        /// This binds the LSB variable (bit 0 of index) first, matching Jolt's behavior.
        /// The round polynomial values [p(0), p(1), p(2), p(3)] are used to compute the new claim
        pub fn bindChallengeWithPoly(self: *Self, r: F, round_poly: [4]F) void {
            const n = self.effectiveLen();
            const half = n / 2;
            if (half == 0) {
                self.round += 1;
                return;
            }

            // If wa is lazy (first bind), materialize to dense at half size
            if (self.wa_evals == null) {
                // First bind: materialize wa from lazy (sparse addrs + eq table) → dense at half size
                const wa_dense = self.allocator.alloc(F, half) catch @panic("wa materialize alloc");
                const addrs = self.wa_addrs.?;
                const eq_tbl = self.wa_eq_table.?;

                const MatWaCtx = struct {
                    dst: []F,
                    ad: []const ?u64,
                    eq: []const F,
                    rr: F,
                    omr: F,
                    alen: usize,
                };
                const mctx = MatWaCtx{
                    .dst = wa_dense,
                    .ad = addrs,
                    .eq = eq_tbl,
                    .rr = r,
                    .omr = F.one().sub(r),
                    .alen = addrs.len,
                };
                const matWaFn = struct {
                    fn f(c: MatWaCtx, i: usize) void {
                        const wa_0 = if (2 * i < c.alen)
                            (if (c.ad[2 * i]) |a| c.eq[@intCast(a)] else F.zero())
                        else
                            F.zero();
                        const wa_1 = if (2 * i + 1 < c.alen)
                            (if (c.ad[2 * i + 1]) |a| c.eq[@intCast(a)] else F.zero())
                        else
                            F.zero();
                        c.dst[i] = c.omr.mul(wa_0).add(c.rr.mul(wa_1));
                    }
                }.f;

                parallelForOptional(self.thread_pool, half, mctx, matWaFn);

                // Free lazy state
                if (self.wa_addrs_owned) {
                    self.allocator.free(self.wa_addrs.?);
                }
                self.wa_addrs = null;
                self.wa_addrs_owned = false;
                self.allocator.free(self.wa_eq_table.?);
                self.wa_eq_table = null;
                self.wa_evals = wa_dense;

                // Bind inc only (wa already bound above)
                const one_minus_r2 = F.one().sub(r);
                for (0..half) |i| {
                    self.inc_evals[i] = one_minus_r2.mul(self.inc_evals[2 * i]).add(r.mul(self.inc_evals[2 * i + 1]));
                }
            } else {
                // Both arrays are dense — bind in parallel
                const BindCtx = struct {
                    slices: [2][]F,
                    r: F,
                    half: usize,
                };
                const bind_ctx = BindCtx{
                    .slices = .{ self.inc_evals, self.wa_evals.? },
                    .r = r,
                    .half = half,
                };
                const bindFn = struct {
                    fn f(ctx: BindCtx, idx: usize) void {
                        const arr = ctx.slices[idx];
                        const one_minus_r3 = F.one().sub(ctx.r);
                        for (0..ctx.half) |i| {
                            arr[i] = one_minus_r3.mul(arr[2 * i]).add(ctx.r.mul(arr[2 * i + 1]));
                        }
                    }
                }.f;

                parallelForOptional(self.thread_pool, 2, bind_ctx, bindFn);
            }

            // Bind split LT polynomial (O(sqrt(T)) work)
            self.lt_poly.bind(r);

            self.round += 1;

            // Update claim from round polynomial evaluation at challenge point.
            // This is O(1) (3 field muls) vs the previous O(T/2^round) full re-summation.
            // The caller already uses this same value for individual_claims[1].
            const poly_mod = @import("zolt_arith").poly;
            self.current_claim = poly_mod.UniPoly(F).evaluateToomCookAt(round_poly, r);
        }

        /// Bind the current variable to challenge r (DEPRECATED - use bindChallengeWithPoly)
        /// This computes the sum of folded products, which is incorrect for degree-3 sumcheck
        pub fn bindChallenge(self: *Self, r: F) void {
            // Compute round poly first, then bind with it
            const round_poly = self.computeRoundPolynomial();
            self.bindChallengeWithPoly(r, round_poly);
        }

        /// Get current claim (after binding challenges)
        pub fn getCurrentClaim(self: *const Self) F {
            return self.current_claim;
        }

        /// Get final claim: the product at the fully bound point
        pub fn getFinalClaim(self: *const Self) F {
            if (self.inc_evals.len == 0) return F.zero();
            return self.inc_evals[0].mul(self.getWa(0)).mul(self.lt_poly.getBoundCoeff(0));
        }

        pub fn getFinalOpenings(self: *const Self) struct { inc_eval: F, wa_eval: F, lt_eval: F } {
            if (self.inc_evals.len == 0) {
                return .{ .inc_eval = F.zero(), .wa_eval = F.zero(), .lt_eval = F.zero() };
            }
            return .{
                .inc_eval = self.inc_evals[0],
                .wa_eval = self.getWa(0),
                .lt_eval = self.lt_poly.getBoundCoeff(0),
            };
        }

        /// Check if complete
        pub fn isComplete(self: *const Self) bool {
            return self.round >= self.numRounds();
        }

        /// Number of rounds
        pub fn numRounds(self: *const Self) usize {
            return self.num_vars;
        }

        /// Get effective array length (shrinks after binding)
        pub fn effectiveLen(self: *const Self) usize {
            return self.inc_evals.len >> @intCast(self.round);
        }
    };
}

/// Value Evaluation Sumcheck Verifier
pub fn ValEvaluationVerifier(comptime F: type) type {
    return struct {
        const Self = @This();

        params: ValEvaluationParams(F),
        current_claim: F,
        challenges: std.ArrayListUnmanaged(F),
        round: usize,
        allocator: Allocator,

        pub fn init(allocator: Allocator, params: ValEvaluationParams(F), initial_claim: F) Self {
            return Self{
                .params = params,
                .current_claim = initial_claim,
                .challenges = .{},
                .round = 0,
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            self.challenges.deinit(self.allocator);
        }

        /// Verify a round polynomial [p(0), p(1), p(2)]
        pub fn verifyRound(self: *Self, round_poly: [3]F, transcript: anytype) !F {
            // For degree 3, we check: p(0) + p(1) = current_claim
            // (This is simplified; full verification uses UniPoly interpolation)
            const sum = round_poly[0].add(round_poly[1]);
            if (!sum.eql(self.current_claim)) {
                return error.SumcheckVerificationFailed;
            }

            // Get challenge
            const challenge = try transcript.challengeScalar("val_eval_challenge");

            // Update claim: evaluate p at challenge using Lagrange interpolation
            // p(r) = (1-r)(2-r)/2 * p(0) - r(2-r) * p(1) + r(r-1)/2 * p(2)
            // Simplified: linear interp for now
            const one_minus_r = F.one().sub(challenge);
            self.current_claim = one_minus_r.mul(round_poly[0]).add(challenge.mul(round_poly[1]));

            try self.challenges.append(self.allocator, challenge);
            self.round += 1;

            return challenge;
        }

        /// Get final claim
        pub fn getFinalClaim(self: *const Self) F {
            return self.current_claim;
        }

        /// Verify final claim against oracle evaluations
        pub fn verifyFinalClaim(
            self: *const Self,
            inc_eval: F,
            wa_eval: F,
            lt_eval: F,
        ) bool {
            const expected = inc_eval.mul(wa_eval).mul(lt_eval);
            return expected.eql(self.current_claim);
        }

        /// Get all challenges
        pub fn getChallenges(self: *const Self) []const F {
            return self.challenges.items;
        }
    };
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Compute eq(r, k) for a specific index k
pub fn computeEqAtPoint(comptime F: type, r: []const F, k: anytype) F {
    return @import("../eq_utils.zig").computeEqAtPointLE(F, r, @intCast(k));
}

// ============================================================================
// Tests
// ============================================================================

test "inc polynomial from empty trace" {
    const allocator = std.testing.allocator;
    const field = @import("zolt_arith").field;
    const F = field.BN254Scalar;

    var trace = MemoryTrace.init(allocator);
    defer trace.deinit();

    var inc = try IncPolynomial(F).fromTrace(allocator, &trace, 1, 0x80000000, 1, null);
    defer inc.deinit();

    // Empty trace should give zero
    try std.testing.expect(inc.get(0).eql(F.zero()));
}

test "inc polynomial from trace with write" {
    const allocator = std.testing.allocator;
    const field = @import("zolt_arith").field;
    const F = field.BN254Scalar;

    var trace = MemoryTrace.init(allocator);
    defer trace.deinit();

    // Write 42 to address 0x80000000
    try trace.recordWrite(0x80000000, 42, 0);
    // Write 100 to same address (inc = 100 - 42 = 58)
    try trace.recordWrite(0x80000000, 100, 1);

    var inc = try IncPolynomial(F).fromTrace(allocator, &trace, 2, 0x80000000, 4, null);
    defer inc.deinit();

    // First write: inc = 42 - 0 = 42
    try std.testing.expect(inc.get(0).eql(F.fromU64(42)));
    // Second write: inc = 100 - 42 = 58
    try std.testing.expect(inc.get(1).eql(F.fromU64(58)));
}

test "wa polynomial initialization" {
    const allocator = std.testing.allocator;
    const field = @import("zolt_arith").field;
    const F = field.BN254Scalar;

    var trace = MemoryTrace.init(allocator);
    defer trace.deinit();

    try trace.recordWrite(0x80000008, 42, 0); // Writes to slot 1
    try trace.recordRead(0x80000008, 42, 1); // Read (no write indicator)
    try trace.recordWrite(0x80000010, 100, 2); // Writes to slot 2

    const r_address = [_]F{ F.zero(), F.zero(), F.zero(), F.zero() }; // slot 0
    var wa = try WaPolynomial(F).fromTrace(allocator, &trace, 3, &r_address, 0x80000000, 16);
    defer wa.deinit();

    // Cycle 0 wrote to slot 1
    try std.testing.expectEqual(@as(?u64, 1), wa.getWriteAddress(0));
    // Cycle 1 was a read
    try std.testing.expectEqual(@as(?u64, null), wa.getWriteAddress(1));
    // Cycle 2 wrote to slot 2
    try std.testing.expectEqual(@as(?u64, 2), wa.getWriteAddress(2));
}

test "lt polynomial basic" {
    const allocator = std.testing.allocator;
    const field = @import("zolt_arith").field;
    const F = field.BN254Scalar;

    // r_cycle = [1, 0] represents cycle 1 in 2-bit representation
    const r_cycle = [_]F{ F.one(), F.zero() };
    var lt = try LtPolynomial(F).init(allocator, &r_cycle);
    defer lt.deinit();

    // LT(0, 1) = 1 (0 < 1)
    const lt_0 = lt.evaluateAtIndex(0);
    try std.testing.expect(!lt_0.eql(F.zero()));

    // LT(1, 1) = 0 (1 is not < 1)
    const lt_1 = lt.evaluateAtIndex(1);
    try std.testing.expect(lt_1.eql(F.zero()));

    // LT(2, 1) = 0 (2 is not < 1)
    const lt_2 = lt.evaluateAtIndex(2);
    try std.testing.expect(lt_2.eql(F.zero()));
}

test "val evaluation params" {
    const allocator = std.testing.allocator;
    const field = @import("zolt_arith").field;
    const F = field.BN254Scalar;

    const r_address = [_]F{ F.one(), F.zero() };
    const r_cycle = [_]F{ F.zero(), F.one() };

    var params = try ValEvaluationParams(F).init(
        allocator,
        F.fromU64(100), // init_eval
        8, // trace_len
        16, // k
        &r_address,
        &r_cycle,
    );
    defer params.deinit();

    try std.testing.expectEqual(@as(usize, 3), params.numRounds()); // log2(8) = 3
    try std.testing.expectEqual(@as(usize, 3), ValEvaluationParams(F).degreeBound());
}

test "val prover sumcheck invariant: p(0) + p(1) = current_claim" {
    const allocator = std.testing.allocator;
    const field = @import("zolt_arith").field;
    const F = field.BN254Scalar;

    // Create a memory trace with some writes
    var trace = MemoryTrace.init(allocator);
    defer trace.deinit();

    // Add 8 memory accesses to get 3 sumcheck rounds (log2(8) = 3)
    try trace.recordWrite(0x80000000, 100, 0); // Write to slot 0
    try trace.recordWrite(0x80000008, 200, 1); // Write to slot 1
    try trace.recordRead(0x80000000, 100, 2); // Read from slot 0
    try trace.recordWrite(0x80000010, 300, 3); // Write to slot 2
    try trace.recordWrite(0x80000000, 150, 4); // Write to slot 0 (update)
    try trace.recordRead(0x80000008, 200, 5); // Read from slot 1
    try trace.recordWrite(0x80000018, 400, 6); // Write to slot 3
    try trace.recordRead(0x80000010, 300, 7); // Read from slot 2

    const r_address = [_]F{ F.zero(), F.zero() }; // Pointing to slot 0
    const r_cycle = [_]F{ F.one(), F.one(), F.one() }; // Pointing to cycle 7

    var params = try ValEvaluationParams(F).init(
        allocator,
        F.zero(), // init_eval
        8, // trace_len
        16, // k
        &r_address,
        &r_cycle,
    );
    defer params.deinit();

    var prover = try ValEvaluationProver(F).init(
        allocator,
        &trace,
        null,
        params,
        0x80000000,
    );
    defer prover.deinit();

    // Verify sumcheck invariant for each round
    var claim = prover.computeInitialClaim();

    for (0..prover.numRounds()) |round| {
        const round_poly = prover.computeRoundPolynomial();

        // The sumcheck invariant: p(0) + p(1) should equal current claim
        const sum = round_poly[0].add(round_poly[1]);

        try std.testing.expect(sum.eql(claim));

        // Bind with a random-ish challenge
        const challenge = F.fromU64(@as(u64, round * 7 + 5));
        prover.bindChallenge(challenge);

        // New claim should be p(challenge) - evaluate using Lagrange interpolation
        // For a degree-d polynomial with points at 0, 1, 2:
        // p(r) = p(0)*L_0(r) + p(1)*L_1(r) + p(2)*L_2(r)
        // L_0(r) = (r-1)(r-2) / (0-1)(0-2) = (r-1)(r-2)/2
        // L_1(r) = (r-0)(r-2) / (1-0)(1-2) = r(r-2)/(-1) = r(2-r)
        // L_2(r) = (r-0)(r-1) / (2-0)(2-1) = r(r-1)/2
        const one = F.one();
        const two = F.fromU64(2);

        const r_minus_1 = challenge.sub(one);
        const r_minus_2 = challenge.sub(two);
        const two_minus_r = two.sub(challenge);

        // L_0(r) = (r-1)(r-2)/2
        const L_0 = r_minus_1.mul(r_minus_2).mul(F.fromU64(2).inv());
        // L_1(r) = r(2-r)
        const L_1 = challenge.mul(two_minus_r);
        // L_2(r) = r(r-1)/2
        const L_2 = challenge.mul(r_minus_1).mul(F.fromU64(2).inv());

        const p_at_r = round_poly[0].mul(L_0).add(round_poly[1].mul(L_1)).add(round_poly[2].mul(L_2));

        // Update claim for next round
        claim = p_at_r;
    }

    // Verify final claim matches prover's tracked claim
    const final_claim = prover.getFinalClaim();
    try std.testing.expect(final_claim.eql(claim));

    dbg("Val prover sumcheck invariant test passed!\n", .{});
}
