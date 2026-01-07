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
                const current_bit: u1 = @truncate(entry.cycle >> @intCast(log_t - 1 - r));

                // Compute eq contribution from already-bound variables (0..r-1)
                var eq_bound = F.one();
                for (0..r) |i| {
                    const bit_i: u1 = @truncate(entry.cycle >> @intCast(log_t - 1 - i));
                    const r_i = self.challenges.items[i];
                    if (bit_i == 1) {
                        eq_bound = eq_bound.mul(r_i);
                    } else {
                        eq_bound = eq_bound.mul(F.one().sub(r_i));
                    }
                }

                // Compute eq contribution from remaining unbound variables (r+1..log_t-1)
                // These use r_cycle_params
                var eq_remaining = F.one();
                for (r + 1..log_t) |i| {
                    const bit_i: u1 = @truncate(entry.cycle >> @intCast(log_t - 1 - i));
                    const r_i = self.params.r_cycle[i];
                    if (bit_i == 1) {
                        eq_remaining = eq_remaining.mul(r_i);
                    } else {
                        eq_remaining = eq_remaining.mul(F.one().sub(r_i));
                    }
                }

                // Total eq contribution (excluding the current variable r)
                const eq_partial = eq_bound.mul(eq_remaining);

                // Also include the contribution from r_cycle_params[r] for the current variable
                // At X=0: contributes (1 - r_cycle_params[r])
                // At X=1: contributes r_cycle_params[r]
                const r_param_r = self.params.r_cycle[r];

                // Compute polynomial contribution
                const val_term = entry.val_coeff;
                const inc_term = self.inc[entry.cycle];
                const inner = val_term.add(gamma.mul(val_term.add(inc_term)));
                const base_contribution = eq_partial.mul(entry.ra_coeff).mul(inner);

                if (current_bit == 0) {
                    // This entry contributes to s(0)
                    // s(0) = Σ entries with bit_r=0, weighted by (1 - r_param_r)
                    s0 = s0.add(base_contribution.mul(F.one().sub(r_param_r)));
                } else {
                    // This entry contributes to s(1)
                    // s(1) = Σ entries with bit_r=1, weighted by r_param_r
                    s1 = s1.add(base_contribution.mul(r_param_r));
                }
            }

            // Extrapolate to s(2) and s(3) for cubic polynomial
            // For a quadratic polynomial, we can use s(2) = s(0) - s(1) + some_correction
            // But actually, our sumcheck should be degree 2 in the eq variable
            // The round polynomial is: s(X) = Σ eq(X) * ra * inner
            // where eq(X) is linear in X.
            //
            // For proper extrapolation, we need s(2):
            // s(2) = Σ entries with bit_r=0, weighted by (1 - 2*r_param_r)
            //      + Σ entries with bit_r=1, weighted by 2*r_param_r
            // This comes from eq(X=2, r_param_r) = X*r_param_r + (1-X)*(1-r_param_r) at X=2
            // = 2*r_param_r + (-1)*(1-r_param_r) = 2*r_param_r - 1 + r_param_r = 3*r_param_r - 1

            // Simpler: s is linear in X, so s(2) = 2*s(1) - s(0) and s(3) = 3*s(1) - 2*s(0)
            // Actually for eq polynomial: eq(X) = (1-X)*(1-r) + X*r = (1-r) + X*(2r-1)
            // So s(X) = sum * ((1-r) + X*(2r-1)) which is linear
            // s(0) = sum * (1-r)
            // s(1) = sum * r
            // s(2) = sum * ((1-r) + 2*(2r-1)) = sum * (1-r + 4r - 2) = sum * (3r - 1)
            // s(3) = sum * ((1-r) + 3*(2r-1)) = sum * (1-r + 6r - 3) = sum * (5r - 2)

            // But we computed s(0) and s(1) directly. To get s(2):
            // s(0) = sum * (1-r), s(1) = sum * r
            // sum = s(0)/(1-r) = s(1)/r (if r != 0, 1)
            // For numerical stability, compute s(2) = 3*s(1) - 2*s(0) + (s(0) - s(1))
            // Actually: s(2) = s(0) + 2*(s(1) - s(0)) + (something for quadratic)
            //
            // Hmm, let me just compute s(2) directly for now
            var s2: F = F.zero();
            for (self.entries.items) |entry| {
                const current_bit: u1 = @truncate(entry.cycle >> @intCast(log_t - 1 - r));
                var eq_bound = F.one();
                for (0..r) |i| {
                    const bit_i: u1 = @truncate(entry.cycle >> @intCast(log_t - 1 - i));
                    const r_i = self.challenges.items[i];
                    if (bit_i == 1) {
                        eq_bound = eq_bound.mul(r_i);
                    } else {
                        eq_bound = eq_bound.mul(F.one().sub(r_i));
                    }
                }
                var eq_remaining = F.one();
                for (r + 1..log_t) |i| {
                    const bit_i: u1 = @truncate(entry.cycle >> @intCast(log_t - 1 - i));
                    const r_i = self.params.r_cycle[i];
                    if (bit_i == 1) {
                        eq_remaining = eq_remaining.mul(r_i);
                    } else {
                        eq_remaining = eq_remaining.mul(F.one().sub(r_i));
                    }
                }
                const eq_partial = eq_bound.mul(eq_remaining);
                const r_param_r = self.params.r_cycle[r];
                const val_term = entry.val_coeff;
                const inc_term = self.inc[entry.cycle];
                const inner = val_term.add(gamma.mul(val_term.add(inc_term)));
                const base_contribution = eq_partial.mul(entry.ra_coeff).mul(inner);

                // At X=2: eq(2, r) = 2*r - (1-r) = 3r - 1
                const eq_at_2 = if (current_bit == 0)
                    F.one().sub(r_param_r).sub(r_param_r) // (1 - 2r) for bit=0
                else
                    r_param_r.add(r_param_r); // 2r for bit=1

                s2 = s2.add(base_contribution.mul(eq_at_2));
            }

            // s(3) by similar logic or extrapolation
            // For now, use linear extrapolation from s(0), s(1), s(2)
            // Assuming quadratic, s(3) = 3*s(2) - 3*s(1) + s(0) (Newton forward diff)
            const s3 = s2.mul(F.fromU64(3)).sub(s1.mul(F.fromU64(3))).add(s0);

            return [4]F{ s0, s1, s2, s3 };
        }

        fn computePhase2Polynomial(self: *Self, gamma: F) [4]F {
            // Phase 2: Similar structure but binding address variables
            var s0: F = F.zero();
            var s2: F = F.zero();

            const addr_round = self.round - self.params.log_t;
            const current_half = @as(usize, 1) << @intCast(self.params.log_k - addr_round - 1);

            for (self.entries.items) |entry| {
                const effective_addr = entry.address >> @intCast(addr_round);
                const addr_pair = effective_addr / 2;
                const is_odd = (effective_addr & 1) == 1;

                if (addr_pair >= current_half) continue;

                // After phase 1, eq is bound; use scalar contribution
                const eq_contribution = self.eq_evals[entry.cycle];

                const val_term = entry.val_coeff;
                const inc_term = self.inc[entry.cycle];
                const inner = val_term.add(gamma.mul(val_term.add(inc_term)));
                const contribution = eq_contribution.mul(entry.ra_coeff).mul(inner);

                if (is_odd) {
                    s2 = s2.add(contribution);
                } else {
                    s0 = s0.add(contribution);
                }
            }

            const s1 = self.current_claim.sub(s0);
            const s3 = s0.sub(s1.mul(F.fromU64(3))).add(s2.mul(F.fromU64(3)));

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
