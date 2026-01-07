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
    };
}

/// RAM Read/Write Checking Prover
///
/// This prover handles the sumcheck for verifying RAM consistency.
pub fn RamReadWriteCheckingProver(comptime F: type) type {
    return struct {
        const Self = @This();
        const Entry = CycleMajorEntry(F);

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
                    if (new_val >= prev_val) {
                        inc[access.timestamp] = F.fromU64(new_val - prev_val);
                    } else {
                        // new_val < prev_val: inc = -(prev_val - new_val) = p - (prev_val - new_val)
                        inc[access.timestamp] = F.zero().sub(F.fromU64(prev_val - new_val));
                    }
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
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            self.entries.deinit(self.allocator);
            self.allocator.free(self.inc);
            self.allocator.free(self.val_init);
            self.allocator.free(self.eq_evals);
            self.challenges.deinit(self.allocator);
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
            // Phase 1: Binding cycle variables
            // For round r, we partition entries by bit r of their cycle index
            // s(0) = sum over entries where bit r = 0
            // s(1) = sum over entries where bit r = 1
            //
            // The eq polynomial contribution is eq_evals[effective_cycle] which has been
            // folded to reflect bound variables. Similarly, inc[effective_cycle].

            var s0: F = F.zero();
            var s1: F = F.zero();

            const half_size = self.eq_size / 2;

            for (self.entries.items) |entry| {
                // For LowToHigh binding, round r binds bit r (from LSB)
                // After r rounds, effective_cycle = entry.cycle >> r = entry.cycle % eq_size

                const effective_cycle = entry.cycle % self.eq_size;
                const in_lower_half = effective_cycle < half_size;

                // Get eq contribution from the folded eq_evals
                const eq_j = self.eq_evals[effective_cycle];

                // Get inc contribution from the folded inc polynomial
                // Note: inc has also been folded to size eq_size
                const inc_term = self.inc[effective_cycle];

                // Compute contribution: eq * ra * (val + γ*(val + inc))
                const val_term = entry.val_coeff;
                const inner = val_term.add(gamma.mul(val_term.add(inc_term)));
                const contribution = eq_j.mul(entry.ra_coeff).mul(inner);

                if (in_lower_half) {
                    s0 = s0.add(contribution);
                } else {
                    s1 = s1.add(contribution);
                }
            }

            // For RWC Phase 1, the round polynomial is LINEAR in X
            // s(X) = s(0) + (s(1) - s(0)) * X
            // s(2) = s(0) + 2*(s(1) - s(0)) = 2*s(1) - s(0)
            // s(3) = s(0) + 3*(s(1) - s(0)) = 3*s(1) - 2*s(0)
            const s2 = s1.add(s1).sub(s0);
            const s3 = s1.mul(F.fromU64(3)).sub(s0.add(s0));

            return [4]F{ s0, s1, s2, s3 };
        }

        fn computePhase2Polynomial(self: *Self, gamma: F) [4]F {
            // Phase 2: Binding address variables
            // All cycle variables are bound, so eq over cycles is a scalar
            // After Phase 1, eq_evals[0] contains eq(r_cycle_params, r_sumcheck_challenges)
            // and inc has been folded to a scalar inc_scalar = inc(r_sumcheck_challenges)

            var s0: F = F.zero();
            var s1: F = F.zero();

            const log_t = self.params.log_t;
            const addr_round = self.round - log_t;

            // After all cycle variables are bound:
            // - eq_evals[0] is the scalar eq(r_cycle_params, r_cycle_sumcheck)
            // - inc[0] is the scalar inc(r_cycle_sumcheck) (after Phase 1 folding)
            const eq_cycle_scalar = self.eq_evals[0];
            const inc_scalar = self.inc[0];

            for (self.entries.items) |entry| {
                const current_addr_bit: u1 = @truncate(entry.address >> @intCast(addr_round));

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
                // Note: eq_cycle_scalar is constant for all entries
                const eq_partial = eq_cycle_scalar.mul(eq_addr);

                // Compute contribution: eq * ra * (val + γ*(inc + val))
                // Note: inc_scalar is constant (already bound to sumcheck point)
                const val_term = entry.val_coeff;
                const inner = val_term.add(gamma.mul(inc_scalar.add(val_term)));
                const contribution = eq_partial.mul(entry.ra_coeff).mul(inner);

                if (current_addr_bit == 0) {
                    s0 = s0.add(contribution);
                } else {
                    s1 = s1.add(contribution);
                }
            }

            const s2 = s1.add(s1).sub(s0);
            const s3 = s1.mul(F.fromU64(3)).sub(s0.add(s0));

            return [4]F{ s0, s1, s2, s3 };
        }

        /// Bind a challenge after round polynomial computation
        pub fn bindChallenge(self: *Self, challenge: F) !void {
            try self.challenges.append(self.allocator, challenge);

            // Fold eq_evals and inc in Phase 1
            const in_phase1 = self.round < self.params.log_t;
            if (in_phase1 and self.eq_size > 1) {
                const half = self.eq_size / 2;

                // Fold eq_evals: eq_new[i] = (1-r)*eq_old[i] + r*eq_old[i + half]
                for (0..half) |i| {
                    const lo = self.eq_evals[i];
                    const hi = self.eq_evals[i + half];
                    self.eq_evals[i] = lo.add(challenge.mul(hi.sub(lo)));
                }

                // Fold inc: inc_new[i] = (1-r)*inc_old[i] + r*inc_old[i + half]
                // This binds inc polynomial to the sumcheck challenge (LowToHigh binding)
                for (0..half) |i| {
                    const lo = self.inc[i];
                    const hi = self.inc[i + half];
                    self.inc[i] = lo.add(challenge.mul(hi.sub(lo)));
                }

                self.eq_size = half;
            }

            self.round += 1;
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

            var val_claim = F.zero();
            for (self.entries.items) |entry| {
                const eq_addr = computeEq(F, r_address[0..log_k], entry.address);
                const eq_cycle = computeEq(F, r_cycle[0..log_t], entry.cycle);
                val_claim = val_claim.add(eq_addr.mul(eq_cycle).mul(entry.val_coeff));
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
