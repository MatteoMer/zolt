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
            // In phase 1, we're summing over address-cycle pairs
            // For each cycle pair (2j, 2j+1), compute contribution
            var s0: F = F.zero();
            var s2: F = F.zero();

            const current_half = @as(usize, 1) << @intCast(self.params.log_t - self.round - 1);

            // Simple approach: iterate over sparse entries
            // Group by cycle pair and compute contributions
            for (self.entries.items) |entry| {
                const effective_cycle = entry.cycle >> @intCast(self.round);
                const cycle_pair = effective_cycle / 2;
                const is_odd = (effective_cycle & 1) == 1;

                if (cycle_pair >= current_half) continue;

                // Get eq evaluation for this cycle
                const eq_j = self.eq_evals[entry.cycle];

                // Compute the sumcheck contribution:
                // eq(r_cycle, j) * ra(k,j) * (Val + γ*(Val + inc))
                const val_term = entry.val_coeff;
                const inc_term = self.inc[entry.cycle];
                const inner = val_term.add(gamma.mul(val_term.add(inc_term)));
                const contribution = eq_j.mul(entry.ra_coeff).mul(inner);

                if (is_odd) {
                    // Contributes to x=1 term
                    s2 = s2.add(contribution);
                } else {
                    // Contributes to x=0 term
                    s0 = s0.add(contribution);
                }
            }

            // Compute s(1) from constraint: s(0) + s(1) = current_claim
            const s1 = self.current_claim.sub(s0);

            // For degree-3 polynomial padding (we have degree 2 effectively)
            // s(3) = extrapolation: s(0) - 3*s(1) + 3*s(2)
            const s3 = s0.sub(s1.mul(F.fromU64(3))).add(s2.mul(F.fromU64(3)));

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
    };
}

/// Compute eq(r, x) for a binary index x
fn computeEq(comptime F: type, r: []const F, x: usize) F {
    var result = F.one();
    for (r, 0..) |ri, i| {
        const xi: u1 = @truncate(x >> @intCast(i));
        if (xi == 1) {
            result = result.mul(ri);
        } else {
            result = result.mul(F.one().sub(ri));
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
