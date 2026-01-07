//! RAM Read/Write Checking Sumcheck Prover
//!
//! This implements the RamReadWriteChecking sumcheck for Stage 2 verification.
//! It proves the consistency of RAM read/write operations across the execution trace.
//!
//! The sumcheck proves:
//! Σ_{k,j} eq(r_cycle, j) * ra(k,j) * (Val(k,j) + γ*(Val(k,j) + inc(j))) = rv_claim + γ*wv_claim
//!
//! This is a 3-phase prover:
//! - Phase 1 (rounds 0 to log_T-1): Cycle-major sparse matrix, binds cycle variables
//! - Phase 2 (rounds log_T to log_T+log_K-1): Address-major sparse matrix, binds address variables
//! - Phase 3 (remaining rounds): Dense polynomials

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
/// For Fibonacci programs with minimal RAM usage, most entries are zero.
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
        /// Eq polynomial evaluations at current binding
        eq_evals: []F,
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

                // Determine inc value for writes
                if (access.op == .Write) {
                    // For writes: inc = new_value - old_value
                    // We store the new value as inc for simplicity
                    inc[access.timestamp] = F.fromU64(access.value);
                }

                try entries.append(allocator, Entry{
                    .cycle = access.timestamp,
                    .address = addr_idx,
                    .ra_coeff = F.one(),
                    .val_coeff = F.fromU64(access.value),
                    .prev_val = access.value,
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

            // Initialize eq polynomial evaluations
            const eq_evals = try allocator.alloc(F, T);
            // Compute eq(r_cycle, j) for each cycle j
            for (0..T) |j| {
                eq_evals[j] = computeEq(F, params.r_cycle, j);
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

            // Determine which phase we're in
            const in_phase1 = self.round < self.params.log_t;

            if (in_phase1) {
                // Phase 1: Binding cycle variables
                return self.computePhase1Polynomial(gamma);
            } else {
                // Phase 2 or 3: Binding address variables
                return self.computePhase2Polynomial(gamma);
            }
        }

        fn computePhase1Polynomial(self: *Self, gamma: F) [4]F {
            // In phase 1, we're binding cycle variables.
            // The sumcheck is over: Σ_{k,j} eq(r_cycle_params, j) * ra(k,j) * (val(k,j) + γ*(val+inc))
            //
            // For round r, we're computing s(X) where X binds variable r of the cycle.
            // Variables 0..r-1 are already bound by challenges in self.challenges.
            // Variables r+1..log_t-1 are still being summed over (contribute to s(0) and s(1)).
            //
            // For each entry with cycle j:
            //   - Extract bit r of j to determine if this contributes to s(0) or s(1)
            //   - Compute eq contribution from:
            //     * Bound variables 0..r-1: use challenges already bound
            //     * Current variable r: contributes 1 (we're evaluating at both 0 and 1)
            //     * Remaining variables r+1..log_t-1: use r_cycle_params[r+1..]

            var s0: F = F.zero();
            var s1: F = F.zero();

            const log_t = self.params.log_t;
            const r = self.round; // Current round (binding variable r)

            for (self.entries.items) |entry| {
                // Extract bit r of the cycle (this determines s(0) vs s(1))
                // We use big-endian: bit 0 is MSB, so cycle[log_t-1-r] is the bit for round r
                const current_bit: u1 = @truncate(entry.cycle >> @intCast(log_t - 1 - r));

                // The eq polynomial is eq(params.r_cycle, j).
                // We precomputed eq_evals[j] = eq(params.r_cycle, j) at init.
                // However, during sumcheck, we need to "fold" this polynomial.
                //
                // Use precomputed eq_evals which already has eq(params.r_cycle, j)
                const eq_j = self.eq_evals[entry.cycle];

                // Compute polynomial contribution: eq(w, j) * ra(j) * (val(j) + γ*(val(j) + inc(j)))
                const val_term = entry.val_coeff;
                const inc_term = self.inc[entry.cycle];
                const inner = val_term.add(gamma.mul(val_term.add(inc_term)));
                const contribution = eq_j.mul(entry.ra_coeff).mul(inner);

                // Accumulate to s(0) or s(1) based on which half this entry is in
                if (current_bit == 0) {
                    s0 = s0.add(contribution);
                } else {
                    s1 = s1.add(contribution);
                }
            }

            // For RWC, the round polynomial is linear in the bound variable (degree 1).
            // After summing contributions, s(0) and s(1) should satisfy s(0) + s(1) = current_claim.
            // Since the polynomial is linear:
            // s(2) = 2*s(1) - s(0)
            // s(3) = 3*s(1) - 2*s(0)
            const s2 = s1.add(s1).sub(s0);
            const s3 = s1.mul(F.fromU64(3)).sub(s0.add(s0));

            return [4]F{ s0, s1, s2, s3 };
        }

        fn computePhase2Polynomial(self: *Self, gamma: F) [4]F {
            // Phase 2: Binding address variables
            // After Phase 1, all cycle variables are bound. The eq polynomial over cycle
            // variables is now a scalar: eq(r_cycle_params, r_cycle_challenges).
            //
            // In Phase 2, we're summing over address pairs, grouped by which address bit
            // is being bound.

            var s0: F = F.zero();
            var s1: F = F.zero();

            const log_t = self.params.log_t;
            const log_k = self.params.log_k;
            const addr_round = self.round - log_t; // Round within address phase (0-indexed)

            for (self.entries.items) |entry| {
                // Extract the current address bit being bound
                const current_addr_bit: u1 = @truncate(entry.address >> @intCast(log_k - 1 - addr_round));

                // Compute eq contribution over cycle variables (all bound now)
                // eq(r_cycle_params, r_cycle_challenges) where challenges are in self.challenges[0..log_t]
                var eq_cycle = F.one();
                for (0..log_t) |i| {
                    const bit_i: u1 = @truncate(entry.cycle >> @intCast(log_t - 1 - i));
                    const r_i = self.challenges.items[i];
                    if (bit_i == 1) {
                        eq_cycle = eq_cycle.mul(r_i);
                    } else {
                        eq_cycle = eq_cycle.mul(F.one().sub(r_i));
                    }
                }

                // Also need eq contribution from params.r_cycle
                for (0..log_t) |i| {
                    const bit_i: u1 = @truncate(entry.cycle >> @intCast(log_t - 1 - i));
                    const r_param_i = self.params.r_cycle[i];
                    if (bit_i == 1) {
                        eq_cycle = eq_cycle.mul(r_param_i);
                    } else {
                        eq_cycle = eq_cycle.mul(F.one().sub(r_param_i));
                    }
                }

                // Compute eq contribution from bound address variables (addr_round > 0 means we've bound some)
                var eq_addr_bound = F.one();
                for (0..addr_round) |i| {
                    const bit_i: u1 = @truncate(entry.address >> @intCast(log_k - 1 - i));
                    const r_i = self.challenges.items[log_t + i]; // Address challenges come after cycle
                    if (bit_i == 1) {
                        eq_addr_bound = eq_addr_bound.mul(r_i);
                    } else {
                        eq_addr_bound = eq_addr_bound.mul(F.one().sub(r_i));
                    }
                }

                // Total eq contribution (excluding the current address variable)
                const eq_partial = eq_cycle.mul(eq_addr_bound);

                // Compute polynomial contribution
                const val_term = entry.val_coeff;
                const inc_term = self.inc[entry.cycle];
                const inner = val_term.add(gamma.mul(val_term.add(inc_term)));
                const base_contribution = eq_partial.mul(entry.ra_coeff).mul(inner);

                // Note: For Phase 2, the "eq" for the current address variable is just
                // counting which half the entry belongs to (no r_cycle_params factor for addresses)
                if (current_addr_bit == 0) {
                    s0 = s0.add(base_contribution);
                } else {
                    s1 = s1.add(base_contribution);
                }
            }

            // Extrapolate to s(2) and s(3)
            // For Phase 2, the polynomial is linear in the current variable
            // s(2) = 2*s(1) - s(0), s(3) = 3*s(1) - 2*s(0)
            const s2 = s1.add(s1).sub(s0);
            const s3 = s1.mul(F.fromU64(3)).sub(s0.add(s0));

            return [4]F{ s0, s1, s2, s3 };
        }

        /// Bind a challenge after round polynomial computation
        pub fn bindChallenge(self: *Self, challenge: F) !void {
            try self.challenges.append(self.allocator, challenge);

            // Update eq polynomial evaluations based on the challenge
            const in_phase1 = self.round < self.params.log_t;

            if (in_phase1) {
                // Fold eq_evals using the challenge
                const half = self.eq_evals.len / 2;
                for (0..half) |i| {
                    const lo = self.eq_evals[i];
                    const hi = self.eq_evals[i + half];
                    // Linear interpolation: eq'(i) = (1-r)*lo + r*hi
                    self.eq_evals[i] = lo.add(challenge.mul(hi.sub(lo)));
                }
                // Resize to half (in place)
                // Actually we can't resize in place easily, so we mark the effective length
            }

            self.round += 1;
        }

        /// Update claim after evaluating polynomial at challenge
        pub fn updateClaim(self: *Self, evals: [4]F, challenge: F) void {
            // Lagrange interpolation at challenge from evals at 0, 1, 2, 3
            const c = challenge;
            const c_minus_1 = c.sub(F.one());
            const c_minus_2 = c.sub(F.fromU64(2));
            const c_minus_3 = c.sub(F.fromU64(3));

            // L0(c) = (c-1)(c-2)(c-3) / (-6)
            const neg6 = F.zero().sub(F.fromU64(6));
            const L0 = c_minus_1.mul(c_minus_2).mul(c_minus_3).mul(neg6.inverse().?);

            // L1(c) = c(c-2)(c-3) / 2
            const L1 = c.mul(c_minus_2).mul(c_minus_3).mul(F.fromU64(2).inverse().?);

            // L2(c) = c(c-1)(c-3) / (-2)
            const neg2 = F.zero().sub(F.fromU64(2));
            const L2 = c.mul(c_minus_1).mul(c_minus_3).mul(neg2.inverse().?);

            // L3(c) = c(c-1)(c-2) / 6
            const L3 = c.mul(c_minus_1).mul(c_minus_2).mul(F.fromU64(6).inverse().?);

            self.current_claim = evals[0].mul(L0)
                .add(evals[1].mul(L1))
                .add(evals[2].mul(L2))
                .add(evals[3].mul(L3));
        }

        /// Check if all rounds are complete
        pub fn isComplete(self: *const Self) bool {
            return self.round >= self.params.numRounds();
        }

        /// Get individual polynomial opening claims after sumcheck completes.
        /// Returns (ra_claim, val_claim, inc_claim) which can be used to verify the expected output claim.
        ///
        /// These are computed by evaluating each polynomial at the opening point:
        /// - ra: evaluated at r_address || r_cycle (full opening point)
        /// - val: evaluated at r_address || r_cycle (full opening point)
        /// - inc: evaluated at r_cycle (only cycle variables)
        pub fn getOpeningClaims(self: *const Self, r_sumcheck: []const F) OpeningClaims(F) {
            const log_k = self.params.log_k;
            const log_t = self.params.log_t;

            // Parse sumcheck challenges into r_address and r_cycle parts
            // RWC uses: r_address || r_cycle ordering in the opening point
            // Phase 1 binds cycle vars, Phase 2 binds address vars
            // The challenges need to be reorganized into big-endian order

            // For simplicity, assume:
            // - First log_t challenges are for cycle variables (Phase 1)
            // - Next log_k challenges are for address variables (Phase 2)
            // Then reverse to get big-endian

            var r_cycle: [32]F = undefined;
            var r_address: [32]F = undefined;
            @memset(&r_cycle, F.zero());
            @memset(&r_address, F.zero());

            // Phase 1 challenges (cycle) - need to be reversed for big-endian
            for (0..@min(log_t, r_sumcheck.len)) |i| {
                r_cycle[log_t - 1 - i] = r_sumcheck[i];
            }
            // Phase 2 challenges (address) - need to be reversed for big-endian
            for (0..@min(log_k, r_sumcheck.len -| log_t)) |i| {
                r_address[log_k - 1 - i] = r_sumcheck[log_t + i];
            }

            // Compute ra_claim = MLE(ra)(r_address, r_cycle)
            // ra is a sparse polynomial: ra(k,j) = 1 if entry (k,j) exists, 0 otherwise
            var ra_claim = F.zero();
            for (self.entries.items) |entry| {
                const eq_addr = computeEq(F, r_address[0..log_k], entry.address);
                const eq_cycle = computeEq(F, r_cycle[0..log_t], entry.cycle);
                ra_claim = ra_claim.add(eq_addr.mul(eq_cycle).mul(entry.ra_coeff));
            }

            // Compute val_claim = MLE(val)(r_address, r_cycle)
            // val is also sparse: val(k,j) = entry.val_coeff for existing entries
            var val_claim = F.zero();
            for (self.entries.items) |entry| {
                const eq_addr = computeEq(F, r_address[0..log_k], entry.address);
                const eq_cycle = computeEq(F, r_cycle[0..log_t], entry.cycle);
                val_claim = val_claim.add(eq_addr.mul(eq_cycle).mul(entry.val_coeff));
            }

            // Compute inc_claim = MLE(inc)(r_cycle)
            // inc is a dense polynomial over cycles only
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

/// Opening claims for RamReadWriteChecking verification
pub fn OpeningClaims(comptime F: type) type {
    return struct {
        ra_claim: F,
        val_claim: F,
        inc_claim: F,
    };
}

/// Compute eq(r, x) for a binary index x
/// r is in BIG-ENDIAN order: r[0] is MSB, r[n-1] is LSB
/// x is interpreted with bit i mapped to r[i] where bit i = bit (n-1-i) of x
fn computeEq(comptime F: type, r: []const F, x: usize) F {
    var result = F.one();
    const n = r.len;
    for (0..n) |i| {
        // r[i] corresponds to bit (n-1-i) of x (MSB first in r)
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

    // Create a simple memory trace
    var trace = MemoryTrace.init(allocator);
    defer trace.deinit();

    try trace.recordWrite(0x80000000, 42, 0);
    try trace.recordRead(0x80000000, 42, 1);

    // Create params
    const r_cycle = [_]F{ F.fromU64(1), F.fromU64(2) };
    var params = try RamReadWriteCheckingParams(F).init(
        allocator,
        F.fromU64(12345), // gamma
        &r_cycle,
        4, // log_k
        2, // log_t
        0x80000000,
    );
    defer params.deinit();

    // Initialize prover
    var prover = try RamReadWriteCheckingProver(F).init(
        allocator,
        &trace,
        params,
        F.fromU64(100), // initial_claim
        null,
    );
    defer prover.deinit();

    try std.testing.expect(!prover.isComplete());
}
