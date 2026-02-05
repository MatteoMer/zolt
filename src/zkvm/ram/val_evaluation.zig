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
const Allocator = std.mem.Allocator;

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
            const effective_len = if (trace_len == 0) 1 else trace_len;
            const padded_len = std.math.ceilPowerOfTwo(usize, effective_len) catch effective_len;
            const num_vars = if (padded_len <= 1) 0 else std.math.log2_int_ceil(usize, padded_len);

            const evals = try allocator.alloc(F, padded_len);
            for (evals) |*e| {
                e.* = F.zero();
            }

            // Track last written value per address for computing increments.
            // Initialize from the provided initial RAM map (if any).
            var last_value = std.AutoHashMap(u64, u64).init(allocator);
            defer last_value.deinit();

            if (initial_ram) |ram| {
                var iter = ram.iterator();
                while (iter.next()) |entry| {
                    const addr = entry.key_ptr.*;
                    if (addr < start_address) continue;
                    const idx = (addr - start_address) / 8;
                    if (idx >= k) continue;
                    try last_value.put(addr, entry.value_ptr.*);
                }
            }

            std.debug.print("[IncPolynomial] Processing {} accesses, trace_len={}, start_address=0x{X:0>16}, k={}\n", .{ trace.accesses.items.len, trace_len, start_address, k });
            for (trace.accesses.items) |access| {
                if (access.op != .Write) continue;
                if (access.address < start_address) {
                    std.debug.print("[IncPolynomial] Skipping write at 0x{X:0>16}: address < start_address\n", .{access.address});
                    continue;
                }

                // NOTE: Jolt DOES include termination/panic writes in the trace.
                // The guest program writes to the termination address via `core::ptr::write_volatile`,
                // which is captured by the tracer as a normal RAM access. The Inc polynomial must
                // include this write's increment (0 -> 1 = 1) for the ValFinal sumcheck to verify:
                //   val_final(r) - val_init(r) = Σ_j inc(j) * wa(r, j)
                //
                // Previously we filtered these out, but that was incorrect - Jolt includes them.
                _ = memory_layout; // unused after removing the filter

                const idx = (access.address - start_address) / 8;
                if (idx >= k) {
                    std.debug.print("[IncPolynomial] Skipping write at 0x{X:0>16}: idx {} >= k {}\n", .{ access.address, idx, k });
                    continue;
                }

                const timestamp = @as(usize, @intCast(access.timestamp));
                if (timestamp >= trace_len) {
                    std.debug.print("[IncPolynomial] Skipping write at 0x{X:0>16}: timestamp {} >= trace_len {}\n", .{ access.address, timestamp, trace_len });
                    continue;
                }

                const old_val = last_value.get(access.address) orelse 0;
                const new_val = access.value;

                // inc = new_val - old_val (as field element)
                if (new_val >= old_val) {
                    evals[timestamp] = F.fromU64(new_val - old_val);
                } else {
                    // Negative difference: -|diff|
                    evals[timestamp] = F.zero().sub(F.fromU64(old_val - new_val));
                }

                std.debug.print("[IncPolynomial] Write at idx={}, timestamp={}, old_val={}, new_val={}, inc={}\n", .{ idx, timestamp, old_val, new_val, if (new_val >= old_val) new_val - old_val else 0 });

                try last_value.put(access.address, new_val);
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
            const effective_len = if (trace_len == 0) 1 else trace_len;
            const padded_len = std.math.ceilPowerOfTwo(usize, effective_len) catch effective_len;
            const num_cycle_vars = if (padded_len <= 1) 0 else std.math.log2_int_ceil(usize, padded_len);

            const write_addresses = try allocator.alloc(?u64, padded_len);
            for (write_addresses) |*w| {
                w.* = null;
            }

            for (trace.accesses.items) |access| {
                if (access.op != .Write) continue;
                if (access.address < start_address) continue;
                const remapped = (access.address - start_address) / 8;
                if (remapped >= k) continue;

                const timestamp = @as(usize, @intCast(access.timestamp));
                if (timestamp >= trace_len) continue;
                write_addresses[timestamp] = remapped;
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
            const n = r_cycle_be.len;
            const size = @as(usize, 1) << @intCast(n);

            const r_copy = try allocator.alloc(F, n);
            @memcpy(r_copy, r_cycle_be);

            const evals = try allocator.alloc(F, size);

            // Initialize all evals to zero
            for (evals) |*e| {
                e.* = F.zero();
            }

            // Build LT evaluations using Jolt's algorithm:
            // for (i, r) in r.r.iter().rev().enumerate() {
            //     let (evals_left, evals_right) = evals.split_at_mut(1 << i);
            //     zip(evals_left, evals_right).for_each(|(x, y)| {
            //         *y = *x * r;
            //         *x += *r - *y;
            //     });
            // }
            //
            // r.r.iter().rev() means we iterate from r_cycle_be[n-1] (LSB) to r_cycle_be[0] (MSB)
            // i=0 corresponds to r_cycle_be[n-1], i=1 to r_cycle_be[n-2], etc.
            for (0..n) |i| {
                // r_cycle_be is BE, so r_cycle_be[n-1-i] is the coefficient for bit position i (LSB=0)
                const r = r_cycle_be[n - 1 - i];
                const half = @as(usize, 1) << @intCast(i);

                // Split evals at position 'half' and process pairs
                var idx: usize = 0;
                while (idx < half) : (idx += 1) {
                    const left_idx = idx;
                    const right_idx = idx + half;

                    const old_x = evals[left_idx];
                    // y = old_x * r
                    evals[right_idx] = old_x.mul(r);
                    // x = old_x + r - y = old_x + r - old_x * r = old_x * (1 - r) + r
                    evals[left_idx] = old_x.add(r).sub(evals[right_idx]);
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
            std.debug.print("[LT DEBUG] evaluateAtIndex(j={}) num_vars={} result={any}\n", .{
                j,
                self.num_vars,
                result.toBytes()[0..8],
            });
            return result;
        }
    };
}

/// Value Evaluation Sumcheck Prover
///
/// This prover implements the sumcheck for the value evaluation:
///   Σ_{j=0}^{T-1} inc(j) · wa(j) · LT(j, r_cycle)
///
/// The key insight is that wa(j) and LT(j) depend on the *full* index j,
/// but after binding variables, the indices are constructed from:
/// - bound challenges (for already-summed variables)
/// - the current free variable (0 or 1)
/// - remaining free variables (summed over)
///
/// To correctly implement this, we:
/// 1. Materialize wa and lt evaluations upfront (same as inc)
/// 2. Bind all three polynomials together after each challenge
pub fn ValEvaluationProver(comptime F: type) type {
    return struct {
        const Self = @This();

        /// Increment polynomial evaluations
        inc_evals: []F,
        /// Write-address indicator evaluations: wa(r_address, j) for each j
        wa_evals: []F,
        /// Less-than evaluations: LT(j, r_cycle) for each j
        lt_evals: []F,
        /// Number of variables (log of trace length)
        num_vars: usize,
        /// Current round (bound variables count)
        round: usize,
        /// Current claim being sumchecked
        current_claim: F,
        /// Parameters
        params: ValEvaluationParams(F),
        allocator: Allocator,

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

            // Build inc polynomial (filtering out synthetic writes if memory_layout provided)
            var inc_poly = try IncPolynomial(F).fromTraceWithLayout(
                allocator,
                trace,
                params.trace_len,
                start_address,
                params.k,
                initial_ram,
                memory_layout,
            );
            defer inc_poly.deinit();

            // Build wa polynomial helper
            var wa_poly = try WaPolynomial(F).fromTrace(
                allocator,
                trace,
                params.trace_len,
                params.r_address,
                start_address,
                params.k,
            );
            defer wa_poly.deinit();

            // Build lt polynomial helper
            var lt_poly = try LtPolynomial(F).init(allocator, params.r_cycle);
            defer lt_poly.deinit();

            const n = inc_poly.evals.len;
            const num_vars = inc_poly.num_vars;

            // Allocate evaluation arrays
            const inc_evals = try allocator.alloc(F, n);
            const wa_evals = try allocator.alloc(F, n);
            const lt_evals = try allocator.alloc(F, n);

            // Materialize all polynomial evaluations
            for (0..n) |j| {
                inc_evals[j] = inc_poly.get(j);
                wa_evals[j] = wa_poly.evaluateAtCycle(j);
                lt_evals[j] = lt_poly.evaluateAtIndex(j);
            }

            // Debug: print r_address used by this prover (first and last 4)
            // Note: r_address uses LE for eq polynomial (symmetric, order doesn't matter)
            std.debug.print("[VALEVAL_INIT] r_address from params, len={}:\n", .{params.r_address.len});
            for (0..@min(4, params.r_address.len)) |i| {
                std.debug.print("  r_address[{}] = {any}\n", .{ i, params.r_address[i].toBytes()[0..8] });
            }
            // Also print last 4 to verify full array
            if (params.r_address.len > 4) {
                for ((params.r_address.len - 4)..params.r_address.len) |i| {
                    std.debug.print("  r_address[{}] = {any}\n", .{ i, params.r_address[i].toBytes()[0..8] });
                }
            }

            // Debug: print initial LT evaluations for indices 0, 1, 128 (to check pattern)
            std.debug.print("[VALEVAL_INIT] LT polynomial values:\n", .{});
            std.debug.print("  lt_evals[0] = {any}\n", .{lt_evals[0].toBytes()[0..8]});
            std.debug.print("  lt_evals[1] = {any}\n", .{lt_evals[1].toBytes()[0..8]});
            std.debug.print("  lt_evals[128] = {any}\n", .{if (n > 128) lt_evals[128].toBytes()[0..8] else lt_evals[0].toBytes()[0..8]});
            std.debug.print("  r_cycle values (from params, BIG_ENDIAN - r[0]=MSB):\n", .{});
            for (0..@min(3, params.r_cycle.len)) |i| {
                std.debug.print("    r_cycle_be[{}] = {any}\n", .{ i, params.r_cycle[i].toBytes()[0..8] });
            }
            // Verify LT(0, r_cycle_be) using Jolt's verifier formula:
            // LT(0, r) = Σ_i (1 - 0_i) · r_i · eq(0[i+1:], r[i+1:])
            //          = Σ_i r_i · (1-r[i+1]) · (1-r[i+2]) · ... · (1-r[n-1])
            // where i runs from MSB (index 0 in BE) to LSB
            std.debug.print("  Verifying LT(0, r_cycle_be) directly (BE formula):\n", .{});
            var lt_0_direct = F.zero();
            var eq_suffix = F.one();
            // Iterate from MSB (index 0) to LSB (index n-1)
            // But we need eq(0[i+1:], r[i+1:]) which is the product for indices > i
            // So we iterate backwards to compute the running product
            var i: usize = num_vars;
            while (i > 0) {
                i -= 1;
                // At position i (BE: 0=MSB), contribution is r[i] * eq_suffix
                // For j=0, all bits are 0, so (1 - j_i) = 1
                lt_0_direct = lt_0_direct.add(params.r_cycle[i].mul(eq_suffix));
                // Update eq_suffix for next iteration (which is position i-1)
                // eq_suffix *= eq(0, r[i]) = (1-r[i])
                eq_suffix = eq_suffix.mul(F.one().sub(params.r_cycle[i]));
            }
            std.debug.print("    LT(0, r_cycle_be) direct = {any}\n", .{lt_0_direct.toBytes()[0..8]});
            std.debug.print("    lt_evals[0] = {any}\n", .{lt_evals[0].toBytes()[0..8]});
            std.debug.print("    Match? {}\n", .{lt_0_direct.eql(lt_evals[0])});

            // Compute initial claim
            var initial_claim = F.zero();
            for (0..n) |j| {
                initial_claim = initial_claim.add(inc_evals[j].mul(wa_evals[j]).mul(lt_evals[j]));
            }

            // Debug: print value at j=54 (known termination write cycle)
            if (n > 54) {
                std.debug.print("[VALEVAL_INIT] At j=54: inc={any}, wa={any}, lt={any}\n", .{
                    inc_evals[54].toBytes()[0..8],
                    wa_evals[54].toBytes()[0..8],
                    lt_evals[54].toBytes()[0..8],
                });
            }
            std.debug.print("[VALEVAL_INIT] n={}, initial_claim={any}\n", .{ n, initial_claim.toBytes()[0..8] });

            return Self{
                .inc_evals = inc_evals,
                .wa_evals = wa_evals,
                .lt_evals = lt_evals,
                .num_vars = num_vars,
                .round = 0,
                .current_claim = initial_claim,
                .params = params,
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.inc_evals);
            self.allocator.free(self.wa_evals);
            self.allocator.free(self.lt_evals);
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
                    evals[0] = self.inc_evals[0].mul(self.wa_evals[0]).mul(self.lt_evals[0]);
                }
                return evals;
            }

            for (0..half) |i| {
                // For LowToHigh binding, x=0 is at index 2*i (bit 0 = 0)
                // and x=1 is at index 2*i+1 (bit 0 = 1)
                const inc_0 = self.inc_evals[2 * i];
                const wa_0 = self.wa_evals[2 * i];
                const lt_0 = self.lt_evals[2 * i];

                const inc_1 = self.inc_evals[2 * i + 1];
                const wa_1 = self.wa_evals[2 * i + 1];
                const lt_1 = self.lt_evals[2 * i + 1];

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

            const one_minus_r = F.one().sub(r);

            // Fold all three polynomials using LowToHigh binding:
            // new[i] = (1-r)*old[2*i] + r*old[2*i+1]
            // This binds the variable corresponding to bit 0 of the index (LSB).
            for (0..half) |i| {
                // inc: interpolate between adjacent pairs
                self.inc_evals[i] = one_minus_r.mul(self.inc_evals[2 * i]).add(r.mul(self.inc_evals[2 * i + 1]));
                // wa: interpolate
                self.wa_evals[i] = one_minus_r.mul(self.wa_evals[2 * i]).add(r.mul(self.wa_evals[2 * i + 1]));
                // lt: interpolate
                self.lt_evals[i] = one_minus_r.mul(self.lt_evals[2 * i]).add(r.mul(self.lt_evals[2 * i + 1]));
            }

            // Conceptually shrink the arrays (we'll use fewer elements)
            // In practice we just track via round and use effectiveLen
            // Zero out the upper half that we just folded from
            for (half..n) |i| {
                self.inc_evals[i] = F.zero();
                self.wa_evals[i] = F.zero();
                self.lt_evals[i] = F.zero();
            }

            _ = round_poly; // The round_poly is only used for the transcript, not for internal claim tracking

            self.round += 1;

            // CRITICAL FIX: After binding, the new claim is the actual sum of products over the bound arrays.
            // The round_poly parameter contains the hinted polynomial (for the transcript), but the
            // prover's internal claim must track the actual polynomial sum: Σ inc[j]*wa[j]*lt[j]
            // This is because the hint mechanism modifies H(1) = claim - H(0) for the sumcheck invariant,
            // but the actual polynomial arrays are independent of this hint.
            var new_claim = F.zero();
            const new_len = self.effectiveLen();
            for (0..new_len) |j| {
                new_claim = new_claim.add(self.inc_evals[j].mul(self.wa_evals[j]).mul(self.lt_evals[j]));
            }
            self.current_claim = new_claim;
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
            return self.inc_evals[0].mul(self.wa_evals[0]).mul(self.lt_evals[0]);
        }

        pub fn getFinalOpenings(self: *const Self) struct { inc_eval: F, wa_eval: F, lt_eval: F } {
            if (self.inc_evals.len == 0) {
                return .{ .inc_eval = F.zero(), .wa_eval = F.zero(), .lt_eval = F.zero() };
            }
            return .{
                .inc_eval = self.inc_evals[0],
                .wa_eval = self.wa_evals[0],
                .lt_eval = self.lt_evals[0],
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
    const k_val: usize = @intCast(k);
    var result = F.one();
    for (r, 0..) |ri, i| {
        const ki = (k_val >> @intCast(i)) & 1;
        if (ki == 1) {
            result = result.mul(ri);
        } else {
            result = result.mul(F.one().sub(ri));
        }
    }
    return result;
}

// ============================================================================
// Tests
// ============================================================================

test "inc polynomial from empty trace" {
    const allocator = std.testing.allocator;
    const field = @import("../../field/mod.zig");
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
    const field = @import("../../field/mod.zig");
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
    const field = @import("../../field/mod.zig");
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
    const field = @import("../../field/mod.zig");
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
    const field = @import("../../field/mod.zig");
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
    const field = @import("../../field/mod.zig");
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

    std.debug.print("Val prover sumcheck invariant test passed!\n", .{});
}
