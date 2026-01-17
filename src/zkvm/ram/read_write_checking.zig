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
const Allocator = std.mem.Allocator;
const MemoryTrace = @import("mod.zig").MemoryTrace;
const MemoryAccess = @import("mod.zig").MemoryAccess;
const MemoryOp = @import("mod.zig").MemoryOp;
const split_eq = @import("../../poly/split_eq.zig");

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
            const r_cycle_copy = try allocator.alloc(F, r_cycle.len);
            @memcpy(r_cycle_copy, r_cycle);

            return Self{
                .gamma = gamma,
                .r_cycle = r_cycle_copy,
                .log_k = log_k,
                .log_t = log_t,
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
        /// Previous value (for tracking)
        prev_val: u64,
        /// Next value (for tracking write increments)
        next_val: u64,

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
                return Self{
                    .cycle = e.cycle / 2,
                    .address = e.address,
                    .ra_coeff = e.ra_coeff.add(r.mul(o.ra_coeff.sub(e.ra_coeff))),
                    .val_coeff = e.val_coeff.add(r.mul(o.val_coeff.sub(e.val_coeff))),
                    .prev_val = e.prev_val,
                    .next_val = o.next_val,
                };
            } else if (even != null) {
                // Only even entry exists - odd is implicit
                const e = even.?.*;
                const odd_val_coeff = F.fromU64(e.next_val);
                return Self{
                    .cycle = e.cycle / 2,
                    .address = e.address,
                    .ra_coeff = F.one().sub(r).mul(e.ra_coeff),
                    .val_coeff = e.val_coeff.add(r.mul(odd_val_coeff.sub(e.val_coeff))),
                    .prev_val = e.prev_val,
                    .next_val = e.next_val,
                };
            } else if (odd != null) {
                // Only odd entry exists - even is implicit
                const o = odd.?.*;
                const even_val_coeff = F.fromU64(o.prev_val);
                return Self{
                    .cycle = o.cycle / 2,
                    .address = o.address,
                    .ra_coeff = r.mul(o.ra_coeff),
                    .val_coeff = even_val_coeff.add(r.mul(o.val_coeff.sub(even_val_coeff))),
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
        /// Allocator
        allocator: Allocator,

        pub fn init(
            allocator: Allocator,
            trace: *const MemoryTrace,
            params: RamReadWriteCheckingParams(F),
            initial_claim: F,
            initial_ram: ?*const std.AutoHashMapUnmanaged(u64, u64),
        ) !Self {
            const K = @as(usize, 1) << @intCast(params.log_k);
            const T = @as(usize, 1) << @intCast(params.log_t);

            // Initialize inc polynomial (zero for all cycles by default)
            const inc = try allocator.alloc(F, T);
            @memset(inc, F.zero());

            // Initialize val_init from initial RAM state
            const val_init = try allocator.alloc(F, K);
            @memset(val_init, F.zero());

            if (initial_ram) |ram| {
                var iter = ram.iterator();
                while (iter.next()) |entry| {
                    const addr = entry.key_ptr.*;
                    const val = entry.value_ptr.*;
                    if (addr >= params.start_address) {
                        const idx = (addr - params.start_address) / 8;
                        if (idx < K) {
                            val_init[idx] = F.fromU64(val);
                        }
                    }
                }
            }

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
                        const idx = (addr - params.start_address) / 8;
                        if (idx < K) {
                            try current_val_per_addr.put(allocator, idx, val);
                        }
                    }
                }
            }

            var entries = std.ArrayListUnmanaged(Entry){};
            for (trace.accesses.items) |access| {
                if (access.timestamp >= T) continue;

                const addr_idx = blk: {
                    if (access.address >= params.start_address) {
                        const idx = (access.address - params.start_address) / 8;
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
                    inc[access.timestamp] = inc_val;
                    std.debug.print("[RWC INC SET] cycle={}, new_val={}, prev_val={}, inc={any}\n", .{
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
                    .cycle = access.timestamp,
                    .address = addr_idx,
                    .ra_coeff = F.one(),
                    .val_coeff = val_coeff,
                    .prev_val = prev_val,
                    .next_val = access.value,
                });

                std.debug.print("[RWC INIT] entry: cycle={}, addr={}, op={}, prev_val={}, next_val={}, inc[{}]={any}\n", .{
                    access.timestamp,
                    addr_idx,
                    @intFromEnum(access.op),
                    prev_val,
                    access.value,
                    access.timestamp,
                    if (access.timestamp < T) inc[access.timestamp].toBytesBE()[0..8] else &[_]u8{ 0, 0, 0, 0, 0, 0, 0, 0 },
                });
            }

            // Sort entries by (cycle, address) for cycle-major ordering
            std.mem.sort(Entry, entries.items, {}, struct {
                fn lessThan(_: void, a: Entry, b: Entry) bool {
                    if (a.cycle != b.cycle) return a.cycle < b.cycle;
                    return a.address < b.address;
                }
            }.lessThan);


            // Initialize eq polynomial evaluations: eq(r_cycle, j) for each cycle j
            // r_cycle is in BIG_ENDIAN order (MSB first, as stored in tau)
            const eq_evals = try allocator.alloc(F, T);
            for (0..T) |j| {
                eq_evals[j] = computeEqBigEndian(F, params.r_cycle, j, params.log_t);
            }

            const challenges_list = std.ArrayListUnmanaged(F){};

            // Initialize GruenSplitEqPolynomial for Phase 1 optimization
            // This matches Jolt's structure for computing round polynomials
            const gruen_eq = try GruenSplitEq.init(allocator, params.r_cycle);

            std.debug.print("[RWC INIT] tau.len = {}, current_index = {}\n", .{ params.r_cycle.len, gruen_eq.current_index });
            if (params.r_cycle.len > 0) {
                std.debug.print("[RWC INIT] tau[0] = {any}\n", .{params.r_cycle[0].toBytesBE()[0..8]});
                if (params.r_cycle.len > 1) {
                    std.debug.print("[RWC INIT] tau[last] = {any}\n", .{params.r_cycle[params.r_cycle.len - 1].toBytesBE()[0..8]});
                }
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
            const in_phase1 = self.round < self.params.log_t;

            const result = if (in_phase1)
                self.computePhase1Polynomial(gamma)
            else
                self.computePhase2Polynomial(gamma);

            return result;
        }

        fn computePhase1Polynomial(self: *Self, gamma: F) [4]F {
            // Phase 1: Using Gruen's optimization matching Jolt exactly
            //
            // The polynomial has the form s(X) = l(X) * q(X) where:
            // - l(X) is the linear eq factor for the current variable
            // - q(X) is quadratic: q(X) = c + d*X + e*X²
            //
            // Jolt's formula for each (even, odd) entry pair at column k:
            //   ra_evals = [ra_even, ra_odd - ra_even]  // [ra_0, ra_infty]
            //   val_evals = [val_even, val_odd - val_even]  // [val_0, val_infty]
            //   inc_evals = [inc_even, inc_odd - inc_even]  // [inc_0, inc_infty]
            //
            //   q_constant += E_prefix * ra_0 * (val_0 + gamma * (inc_0 + val_0))
            //   q_quadratic += E_prefix * ra_infty * (val_infty + gamma * (inc_infty + val_infty))

            var gruen_eq = &self.gruen_eq.?;

            // Get E_out and E_in tables for the current window
            const tables = gruen_eq.getWindowEqTables(gruen_eq.current_index, 1);
            const E_out = tables.E_out;
            const E_in = tables.E_in;
            const head_in_bits = tables.head_in_bits;

            var q_constant: F = F.zero();
            var q_quadratic: F = F.zero();

            // Group entries by row pair (rows 2j and 2j+1 for each j)
            // Entries are bound after each round, so entry.cycle is the current row
            var entry_idx: usize = 0;
            while (entry_idx < self.entries.items.len) {
                const entry = self.entries.items[entry_idx];
                // Since entries are bound, entry.cycle is the current effective cycle
                const effective_cycle = entry.cycle;
                const row_pair_idx = effective_cycle / 2; // j = cycle / 2

                // Find even and odd entries for this column at this row pair
                const is_even_row = effective_cycle % 2 == 0;

                // Compute E_prefix = E_out[x_out] * E_in[x_in]
                const x_out = row_pair_idx >> @intCast(head_in_bits);
                const x_in_mask = (@as(usize, 1) << @intCast(head_in_bits)) - 1;
                const x_in = row_pair_idx & x_in_mask;
                const E_out_val = if (x_out < E_out.len) E_out[x_out] else F.one();
                const E_in_val = if (x_in < E_in.len) E_in[x_in] else F.one();
                const E_prefix = E_out_val.mul(E_in_val);

                // Get inc values for this row pair
                const j_prime = row_pair_idx * 2; // even row index
                const inc_0 = if (j_prime < self.inc.len) self.inc[j_prime] else F.zero();
                const inc_1 = if (j_prime + 1 < self.inc.len) self.inc[j_prime + 1] else F.zero();
                const inc_infty = inc_1.sub(inc_0);
                const inc_evals = [2]F{ inc_0, inc_infty };

                // Compute ra_evals and val_evals based on which entries exist
                // Case 1: Only even entry exists (common case for sparse matrix)
                // Case 2: Only odd entry exists
                // Case 3: Both entries exist (check if next entry is in same row pair)

                var ra_0: F = undefined;
                var ra_infty: F = undefined;
                var val_0: F = undefined;
                var val_infty: F = undefined;

                if (is_even_row) {
                    // Entry is at even row - check if there's a matching odd entry
                    const has_odd = blk: {
                        if (entry_idx + 1 < self.entries.items.len) {
                            const next = self.entries.items[entry_idx + 1];
                            // Entries are bound, so next.cycle is the current row
                            const next_effective = next.cycle;
                            const next_pair = next_effective / 2;
                            break :blk (next_pair == row_pair_idx and next.address == entry.address and next_effective % 2 == 1);
                        }
                        break :blk false;
                    };

                    if (has_odd) {
                        // Both even and odd entries exist
                        const odd_entry = self.entries.items[entry_idx + 1];
                        ra_0 = entry.ra_coeff;
                        ra_infty = odd_entry.ra_coeff.sub(entry.ra_coeff);
                        val_0 = entry.val_coeff;
                        val_infty = odd_entry.val_coeff.sub(entry.val_coeff);
                        entry_idx += 2; // Skip both entries
                    } else {
                        // Only even entry exists - odd entry is implicit
                        // odd_val_coeff = entry.next_val (value after this access)
                        const odd_val_coeff = F.fromU64(entry.next_val);
                        ra_0 = entry.ra_coeff;
                        ra_infty = F.zero().sub(entry.ra_coeff); // -ra_even
                        val_0 = entry.val_coeff;
                        val_infty = odd_val_coeff.sub(entry.val_coeff);
                        entry_idx += 1;
                    }
                } else {
                    // Entry is at odd row - even entry is implicit
                    // even_val_coeff = entry.prev_val (value before this access)
                    const even_val_coeff = F.fromU64(entry.prev_val);
                    ra_0 = F.zero(); // No access at even row
                    ra_infty = entry.ra_coeff;
                    val_0 = even_val_coeff;
                    val_infty = entry.val_coeff.sub(even_val_coeff);
                    entry_idx += 1;
                }

                // Apply Jolt's formula:
                // q_constant += E_prefix * ra_0 * (val_0 + gamma * (inc_0 + val_0))
                // q_quadratic += E_prefix * ra_infty * (val_infty + gamma * (inc_infty + val_infty))
                const inner_0 = val_0.add(gamma.mul(inc_evals[0].add(val_0)));
                const inner_infty = val_infty.add(gamma.mul(inc_evals[1].add(val_infty)));

                const contrib_0 = E_prefix.mul(ra_0).mul(inner_0);
                const contrib_infty = E_prefix.mul(ra_infty).mul(inner_infty);

                q_constant = q_constant.add(contrib_0);
                q_quadratic = q_quadratic.add(contrib_infty);
            }

            // Use Gruen's formula to compute s(X) = l(X) * q(X)
            const result = gruen_eq.computeCubicRoundPoly(q_constant, q_quadratic, self.current_claim);

            std.debug.print("[RWC PHASE1] round={}, q_constant={any}\n", .{ self.round, q_constant.toBytesBE()[0..8] });
            std.debug.print("[RWC PHASE1] q_quadratic={any}, current_claim={any}\n", .{ q_quadratic.toBytesBE()[0..8], self.current_claim.toBytesBE()[0..8] });
            std.debug.print("[RWC PHASE1] result: s0={any}, s1={any}\n", .{ result[0].toBytesBE()[0..8], result[1].toBytesBE()[0..8] });

            return result;
        }

        fn computePhase2Polynomial(self: *Self, gamma: F) [4]F {
            // Phase 2: Binding address variables
            // All cycle variables are bound, so eq over cycles is a scalar
            // After Phase 1, eq_evals[0] contains eq(r_cycle_params, r_sumcheck_challenges)
            // and inc has been folded to a scalar inc_scalar = inc(r_sumcheck_challenges)

            std.debug.print("[RWC PHASE2] round={}, entries.len={}\n", .{ self.round, self.entries.items.len });

            var s0: F = F.zero();
            var s1: F = F.zero();

            const log_t = self.params.log_t;
            const addr_round = self.round - log_t;

            // Debug: show eq_cycle_scalar and inc_scalar at the start of Phase 2
            if (addr_round == 0) {
                std.debug.print("[RWC PHASE2] eq_cycle_scalar = {any}\n", .{self.eq_evals[0].toBytesBE()[0..8]});
                std.debug.print("[RWC PHASE2] inc_scalar = {any}\n", .{self.inc[0].toBytesBE()[0..8]});
                if (self.entries.items.len > 0) {
                    const e = self.entries.items[0];
                    std.debug.print("[RWC PHASE2] entry[0]: addr={}, ra_coeff={any}, val_coeff={any}\n", .{
                        e.address,
                        e.ra_coeff.toBytesBE()[0..8],
                        e.val_coeff.toBytesBE()[0..8],
                    });
                }
            }

            // After all cycle variables are bound:
            // - eq_evals[0] is the scalar eq(r_cycle_params, r_cycle_sumcheck)
            // - inc[0] is the scalar inc(r_cycle_sumcheck) (after Phase 1 folding)
            const eq_cycle_scalar = self.eq_evals[0];
            const inc_scalar = self.inc[0];

            // Compute s(0), s(1), s(2) directly as polynomial evaluations
            // For each entry, ra and val are linear in the address variable X
            // s(X) = eq * ra(X) * (val(X) + γ*(inc + val(X))) is degree-2 in X
            var s2: F = F.zero();

            // Get val_init checkpoint at the current bound point
            // After binding addr_round variables, val_init is at current_size = K / 2^addr_round
            const K = @as(usize, 1) << @intCast(self.params.log_k);
            const val_init_current_size = K >> @intCast(addr_round);

            for (self.entries.items) |entry| {
                // The current address bit being bound determines if this entry contributes to s(0) or s(1)
                const current_addr_bit: u1 = @truncate(entry.address >> @intCast(addr_round));

                // For checkpoint lookup, we need the col_pair index in the bound val_init
                // entry.address / 2^(addr_round+1) gives the pair index after binding current round
                const col_pair = entry.address >> @intCast(addr_round + 1);
                const even_col_idx = col_pair * 2;
                const odd_col_idx = even_col_idx + 1;

                // Get checkpoint from bound val_init
                const even_checkpoint = if (even_col_idx < val_init_current_size)
                    self.val_init[even_col_idx]
                else
                    F.zero();
                const odd_checkpoint = if (odd_col_idx < val_init_current_size)
                    self.val_init[odd_col_idx]
                else
                    F.zero();

                // Compute eq over bound address variables only
                var eq_addr = F.one();
                for (0..addr_round) |i| {
                    const bit_i: u1 = @truncate(entry.address >> @intCast(i));
                    const r_i = self.challenges.items[log_t + i];
                    if (bit_i == 1) {
                        eq_addr = eq_addr.mul(r_i);
                    } else {
                        eq_addr = eq_addr.mul(F.one().sub(r_i));
                    }
                }

                // eq_partial = eq_cycle_scalar * eq_addr
                const eq_partial = eq_cycle_scalar.mul(eq_addr);

                const ra_coeff = entry.ra_coeff;
                const val_coeff = entry.val_coeff;
                const one_plus_gamma = F.one().add(gamma);

                if (current_addr_bit == 0) {
                    // Entry at even column: ra(0) = ra_coeff, ra(1) = 0
                    // val(0) = val_coeff, val(1) = odd_checkpoint
                    const ra_0 = ra_coeff;
                    const val_0 = val_coeff;
                    const val_1 = odd_checkpoint;
                    const ra_2 = F.zero().sub(ra_coeff); // ra(2) = 2*ra(1) - ra(0) = 0 - ra_coeff = -ra_coeff
                    const val_2 = val_1.add(val_1).sub(val_0); // val(2) = 2*val(1) - val(0)

                    const inner_0 = val_0.mul(one_plus_gamma).add(gamma.mul(inc_scalar));
                    const inner_2 = val_2.mul(one_plus_gamma).add(gamma.mul(inc_scalar));

                    s0 = s0.add(eq_partial.mul(ra_0).mul(inner_0));
                    // s1 contribution = 0 (ra(1) = 0)
                    s2 = s2.add(eq_partial.mul(ra_2).mul(inner_2));
                } else {
                    // Entry at odd column: ra(0) = 0, ra(1) = ra_coeff
                    // val(0) = even_checkpoint, val(1) = val_coeff
                    const ra_1 = ra_coeff;
                    const val_0 = even_checkpoint;
                    const val_1 = val_coeff;
                    const ra_2 = ra_1.add(ra_1); // ra(2) = 2*ra(1) - ra(0) = 2*ra_coeff - 0
                    const val_2 = val_1.add(val_1).sub(val_0); // val(2) = 2*val(1) - val(0)

                    const inner_1 = val_1.mul(one_plus_gamma).add(gamma.mul(inc_scalar));
                    const inner_2 = val_2.mul(one_plus_gamma).add(gamma.mul(inc_scalar));

                    // s0 contribution = 0 (ra(0) = 0)
                    s1 = s1.add(eq_partial.mul(ra_1).mul(inner_1));
                    s2 = s2.add(eq_partial.mul(ra_2).mul(inner_2));
                }
            }

            // For degree-2 polynomial: s(3) = 3*s(2) - 3*s(1) + s(0) (Lagrange extrapolation)
            const s3 = s2.mul(F.fromU64(3)).sub(s1.mul(F.fromU64(3))).add(s0);

            if (addr_round < 3) {
                std.debug.print("[RWC PHASE2] result: s0={any}, s1={any}, s2={any}\n", .{
                    s0.toBytesBE()[0..8],
                    s1.toBytesBE()[0..8],
                    s2.toBytesBE()[0..8],
                });
            }

            return [4]F{ s0, s1, s2, s3 };
        }

        /// Bind a challenge after round polynomial computation
        pub fn bindChallenge(self: *Self, challenge: F) !void {
            try self.challenges.append(self.allocator, challenge);

            // Fold eq_evals and inc in Phase 1
            const in_phase1 = self.round < self.params.log_t;
            if (in_phase1 and self.eq_size > 1) {
                const half = self.eq_size / 2;

                // Fold eq_evals using LowToHigh binding to match inc and entries:
                // bound[i] = (1-r)*coeff[2*i] + r*coeff[2*i+1]
                // This matches Jolt's merged_eq.bind_parallel(r_j, BindingOrder::LowToHigh)
                for (0..half) |i| {
                    const lo = self.eq_evals[2 * i];
                    const hi = self.eq_evals[2 * i + 1];
                    self.eq_evals[i] = lo.add(challenge.mul(hi.sub(lo)));
                }

                // Fold inc using LowToHigh binding to match Jolt:
                // bound[i] = (1-r)*coeff[2*i] + r*coeff[2*i+1]
                // This matches Jolt's inc.bind_parallel(r_j, BindingOrder::LowToHigh)
                for (0..half) |i| {
                    const lo = self.inc[2 * i];
                    const hi = self.inc[2 * i + 1];
                    self.inc[i] = lo.add(challenge.mul(hi.sub(lo)));
                }

                self.eq_size = half;

                // Bind the Gruen eq polynomial to update current_scalar and E tables
                if (self.gruen_eq) |*geq| {
                    geq.bind(challenge);
                }

                // Bind entries: group by (row/2, col), create bound entries
                // This matches Jolt's ReadWriteMatrixCycleMajor::bind
                try self.bindEntries(challenge);
            }

            // Fold val_init in Phase 2
            // This matches Jolt's val_init.bind_parallel(r, BindingOrder::LowToHigh)
            const in_phase2 = self.round >= self.params.log_t and
                self.round < self.params.log_t + self.params.log_k;
            if (in_phase2) {
                const addr_round = self.round - self.params.log_t;
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
            }

            self.round += 1;
        }

        /// Bind entries by grouping (row/2, col) and creating bound entries
        fn bindEntries(self: *Self, r: F) !void {
            var new_entries = std.ArrayListUnmanaged(Entry){};

            var i: usize = 0;
            while (i < self.entries.items.len) {
                const entry = &self.entries.items[i];
                const pair_key = entry.cycle / 2;
                const addr_key = entry.address;
                const is_even = entry.cycle % 2 == 0;

                // Look for matching odd entry at next position
                var odd_entry: ?*const Entry = null;
                var even_entry: ?*const Entry = null;

                if (is_even) {
                    even_entry = entry;
                    // Check if next entry is the odd counterpart
                    if (i + 1 < self.entries.items.len) {
                        const next = &self.entries.items[i + 1];
                        if (next.cycle / 2 == pair_key and next.address == addr_key and next.cycle % 2 == 1) {
                            odd_entry = next;
                            i += 1; // Skip next entry
                        }
                    }
                } else {
                    odd_entry = entry;
                }

                // Bind the entry pair
                if (Entry.bindEntries(even_entry, odd_entry, r)) |bound| {
                    try new_entries.append(self.allocator, bound);
                }

                i += 1;
            }

            // Replace old entries with bound entries
            self.entries.deinit(self.allocator);
            self.entries = new_entries;

            std.debug.print("[RWC BIND] round={}, entries.len after bind={}\n", .{ self.round, self.entries.items.len });
            if (self.entries.items.len > 0) {
                const e = self.entries.items[0];
                std.debug.print("[RWC BIND]   entry[0]: cycle={}, addr={}, ra_coeff={any}\n", .{ e.cycle, e.address, e.ra_coeff.toBytesBE()[0..8] });
            }
        }

        /// Update claim after evaluating polynomial at challenge
        pub fn updateClaim(self: *Self, evals: [4]F, challenge: F) void {
            const c = challenge;
            const c_minus_1 = c.sub(F.one());
            const c_minus_2 = c.sub(F.fromU64(2));
            const c_minus_3 = c.sub(F.fromU64(3));

            const neg6 = F.zero().sub(F.fromU64(6));
            const L0 = c_minus_1.mul(c_minus_2).mul(c_minus_3).mul(neg6.inverse().?);
            const L1 = c.mul(c_minus_2).mul(c_minus_3).mul(F.fromU64(2).inverse().?);
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
            const log_k = self.params.log_k;
            const log_t = self.params.log_t;

            var r_cycle: [32]F = undefined;
            var r_address: [32]F = undefined;
            @memset(&r_cycle, F.zero());
            @memset(&r_address, F.zero());

            for (0..@min(log_t, r_sumcheck.len)) |i| {
                r_cycle[log_t - 1 - i] = r_sumcheck[i];
            }
            for (0..@min(log_k, r_sumcheck.len -| log_t)) |i| {
                r_address[log_k - 1 - i] = r_sumcheck[log_t + i];
            }

            var ra_claim = F.zero();
            for (self.entries.items) |entry| {
                const eq_addr = computeEq(F, r_address[0..log_k], entry.address);
                const eq_cycle = computeEq(F, r_cycle[0..log_t], entry.cycle);
                ra_claim = ra_claim.add(eq_addr.mul(eq_cycle).mul(entry.ra_coeff));
            }

            // Compute base: val_init.evaluate(r_address) = Σ_k eq(r_address, k) * val_init[k]
            // This is the "background" value from initial RAM state
            var val_claim = F.zero();
            const K = @as(usize, 1) << @intCast(log_k);
            for (0..@min(K, self.val_init.len)) |k| {
                const eq_addr = computeEq(F, r_address[0..log_k], k);
                val_claim = val_claim.add(eq_addr.mul(self.val_init[k]));
            }

            // Add entry contributions: eq(r_addr, addr) * eq(r_cycle, cycle) * (entry.val_coeff - val_init[addr])
            // Each entry represents a RAM operation that overwrites the initial value at that (addr, cycle)
            for (self.entries.items) |entry| {
                const eq_addr = computeEq(F, r_address[0..log_k], entry.address);
                const eq_cycle = computeEq(F, r_cycle[0..log_t], entry.cycle);
                const delta = entry.val_coeff.sub(self.val_init[entry.address]);
                val_claim = val_claim.add(eq_addr.mul(eq_cycle).mul(delta));
            }

            var inc_claim = F.zero();
            const T = @as(usize, 1) << @intCast(log_t);
            for (0..@min(T, self.inc.len)) |j| {
                const eq_cycle = computeEq(F, r_cycle[0..log_t], j);
                inc_claim = inc_claim.add(eq_cycle.mul(self.inc[j]));
            }

            return OpeningClaims(F){
                .ra_claim = ra_claim,
                .val_claim = val_claim,
                .inc_claim = inc_claim,
            };
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
    var result = F.one();
    for (0..n) |i| {
        // r[i] corresponds to bit (n-1-i) of x (MSB to LSB)
        const bit_pos = n - 1 - i;
        const xi: u1 = @truncate(x >> @intCast(bit_pos));
        if (xi == 1) {
            result = result.mul(r[i]);
        } else {
            result = result.mul(F.one().sub(r[i]));
        }
    }
    return result;
}

/// Compute eq(r, x) for a binary index x
/// r is in BIG-ENDIAN order: r[0] is MSB, r[n-1] is LSB
fn computeEq(comptime F: type, r: []const F, x: usize) F {
    var result = F.one();
    const n = r.len;
    for (0..n) |i| {
        const xi: u1 = @truncate(x >> @intCast(n - 1 - i));
        if (xi == 1) {
            result = result.mul(r[i]);
        } else {
            result = result.mul(F.one().sub(r[i]));
        }
    }
    return result;
}

test "ram read write checking prover initialization" {
    const allocator = std.testing.allocator;
    const field = @import("../../field/mod.zig");
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
    );
    defer prover.deinit();

    try std.testing.expect(!prover.isComplete());
}
