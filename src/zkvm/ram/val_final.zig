//! RAM Final Value Evaluation Sumcheck
//!
//! Proves:
//!   Val_final(r_address) - Val_init(r_address) = Σ_{j=0}^{T-1} inc(j) · wa(r_address, j)
//!
//! This is degree-2 (product of two multilinear polynomials).

const std = @import("std");

const zkvm_debug = @import("../debug.zig");
const dbg = zkvm_debug.dbg;

const Allocator = std.mem.Allocator;

const mod = @import("mod.zig");
const MemoryTrace = mod.MemoryTrace;
const val_evaluation = @import("val_evaluation.zig");

const IncPolynomial = val_evaluation.IncPolynomial;
const WaPolynomial = val_evaluation.WaPolynomial;

/// Parameters for RAM final value evaluation
pub fn ValFinalParams(comptime F: type) type {
    return struct {
        const Self = @This();

        /// Trace length (T)
        trace_len: usize,
        /// Address point from OutputSumcheck (r_address)
        r_address: []const F,
        allocator: Allocator,

        pub fn init(allocator: Allocator, trace_len: usize, r_address: []const F) !Self {
            const r_addr_copy = try allocator.alloc(F, r_address.len);
            @memcpy(r_addr_copy, r_address);

            return Self{
                .trace_len = trace_len,
                .r_address = r_addr_copy,
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.r_address);
        }

        /// Number of sumcheck rounds = log2(trace_len)
        pub fn numRounds(self: *const Self) usize {
            if (self.trace_len == 0) return 0;
            return std.math.log2_int_ceil(usize, self.trace_len);
        }

        /// Degree bound is 2 (product of 2 linear polynomials: inc, wa)
        pub fn degreeBound() usize {
            return 2;
        }
    };
}

/// RAM final value evaluation sumcheck prover
pub fn ValFinalProver(comptime F: type) type {
    return struct {
        const Self = @This();

        /// Increment polynomial evaluations
        inc_evals: []F,
        /// Write-address indicator evaluations: wa(r_address, j) for each j
        wa_evals: []F,
        /// Number of variables
        num_vars: usize,
        /// Current round
        round: usize,
        /// Current claim
        current_claim: F,
        /// Parameters
        params: ValFinalParams(F),
        allocator: Allocator,

        /// Synthetic write descriptor for injecting virtual writes (e.g., termination bit)
        /// that are not in the memory trace but must be accounted for in ValFinal.
        pub const SyntheticWrite = struct {
            /// Cycle index at which the synthetic write occurs
            cycle: usize,
            /// Memory address of the write
            address: u64,
            /// Value before write
            old_value: u64,
            /// Value after write
            new_value: u64,
        };

        pub fn init(
            allocator: Allocator,
            trace: *const MemoryTrace,
            initial_ram: ?*const std.AutoHashMapUnmanaged(u64, u64),
            params: ValFinalParams(F),
            start_address: u64,
        ) !Self {
            return initWithSyntheticWrites(allocator, trace, initial_ram, params, start_address, null);
        }

        /// Initialize with optional synthetic writes that are injected into the inc/wa polynomials.
        ///
        /// This is needed for the termination bit: Zolt treats the termination step as a NoOp
        /// in R1CS (so it's not in the RAM trace), but the OutputSumcheck sets
        /// val_final[termination_index] = 1. The ValFinal sumcheck must prove:
        ///   val_final(r) - val_init(r) = Σ inc(j) * wa(r, j)
        /// So we need inc(termination_cycle) = 1 and wa(r, termination_cycle) = eq(r, term_addr).
        pub fn initWithSyntheticWrites(
            allocator: Allocator,
            trace: *const MemoryTrace,
            initial_ram: ?*const std.AutoHashMapUnmanaged(u64, u64),
            params: ValFinalParams(F),
            start_address: u64,
            synthetic_writes: ?[]const SyntheticWrite,
        ) !Self {
            const k = @as(usize, 1) << @intCast(params.r_address.len);

            // Build inc polynomial
            var inc_poly = try IncPolynomial(F).fromTrace(
                allocator,
                trace,
                params.trace_len,
                start_address,
                k,
                initial_ram,
            );
            defer inc_poly.deinit();

            // Build wa polynomial helper
            var wa_poly = try WaPolynomial(F).fromTrace(
                allocator,
                trace,
                params.trace_len,
                params.r_address,
                start_address,
                k,
            );
            defer wa_poly.deinit();

            const n = inc_poly.evals.len;
            const num_vars = inc_poly.num_vars;

            // Allocate evaluation arrays
            const inc_evals = try allocator.alloc(F, n);
            const wa_evals = try allocator.alloc(F, n);

            // Materialize all polynomial evaluations
            for (0..n) |j| {
                inc_evals[j] = inc_poly.get(j);
                wa_evals[j] = wa_poly.evaluateAtCycle(j);
            }

            // Inject synthetic writes into inc and wa polynomials
            if (synthetic_writes) |writes| {
                for (writes) |sw| {
                    if (sw.cycle >= n) {
                        dbg("[ValFinal] WARNING: synthetic write cycle {} >= n {}, skipping\n", .{ sw.cycle, n });
                        continue;
                    }
                    if (sw.address < start_address) {
                        dbg("[ValFinal] WARNING: synthetic write address 0x{X} < start_address 0x{X}, skipping\n", .{ sw.address, start_address });
                        continue;
                    }
                    const remapped = (sw.address - start_address) / 8;
                    if (remapped >= k) {
                        dbg("[ValFinal] WARNING: synthetic write remapped address {} >= k {}, skipping\n", .{ remapped, k });
                        continue;
                    }

                    // Compute inc = new_value - old_value
                    const inc_val = if (sw.new_value >= sw.old_value)
                        F.fromU64(sw.new_value - sw.old_value)
                    else
                        F.zero().sub(F.fromU64(sw.old_value - sw.new_value));

                    // Compute wa = eq(r_address, remapped_address)
                    const wa_val = val_evaluation.computeEqAtPoint(F, params.r_address, remapped);

                    dbg("[ValFinal] Injecting synthetic write: cycle={}, addr=0x{X}, remapped={}, inc={}, wa={any}\n", .{
                        sw.cycle, sw.address, remapped,
                        if (sw.new_value >= sw.old_value) sw.new_value - sw.old_value else 0,
                        wa_val.toBytesBE(),
                    });

                    // Add to the polynomial evaluations at this cycle
                    inc_evals[sw.cycle] = inc_evals[sw.cycle].add(inc_val);
                    wa_evals[sw.cycle] = wa_val; // Override (should be zero from trace since not in trace)
                }
            }

            // Compute initial claim
            var initial_claim = F.zero();
            for (0..n) |j| {
                initial_claim = initial_claim.add(inc_evals[j].mul(wa_evals[j]));
            }

            dbg("[ValFinal] initial_claim = {any}\n", .{initial_claim.toBytesBE()});

            return Self{
                .inc_evals = inc_evals,
                .wa_evals = wa_evals,
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
            self.params.deinit();
        }

        pub fn computeInitialClaim(self: *const Self) F {
            return self.current_claim;
        }

        /// Compute round polynomial in Toom-Cook format: [p(0), p(1), p(2), p_inf]
        /// For degree-2 polynomial (product of 2 multilinears), p_inf = c2 (leading coefficient).
        /// Uses LowToHigh indexing: x=0 at index 2*i, x=1 at index 2*i+1
        pub fn computeRoundPolynomial(self: *Self) [4]F {
            var evals: [4]F = .{ F.zero(), F.zero(), F.zero(), F.zero() };
            const n = self.effectiveLen();
            const half = n / 2;

            if (half == 0) {
                if (n > 0) {
                    evals[0] = self.inc_evals[0].mul(self.wa_evals[0]);
                }
                return evals;
            }

            const two = F.fromU64(2);

            for (0..half) |i| {
                // For LowToHigh binding, x=0 is at index 2*i (bit 0 = 0)
                // and x=1 is at index 2*i+1 (bit 0 = 1)
                const inc_0 = self.inc_evals[2 * i];
                const wa_0 = self.wa_evals[2 * i];
                const inc_1 = self.inc_evals[2 * i + 1];
                const wa_1 = self.wa_evals[2 * i + 1];

                // p(0) and p(1)
                evals[0] = evals[0].add(inc_0.mul(wa_0));
                evals[1] = evals[1].add(inc_1.mul(wa_1));

                // Extrapolate to x=2 for each polynomial, then multiply
                // For multilinear: f(2) = 2*f(1) - f(0)
                const inc_2 = two.mul(inc_1).sub(inc_0);
                const wa_2 = two.mul(wa_1).sub(wa_0);
                evals[2] = evals[2].add(inc_2.mul(wa_2));

                // p_inf = c2 (leading coefficient) for degree-2 polynomial
                // For product of 2 multilinears, c2 = product of slopes
                // slope = f(1) - f(0) for each multilinear
                const inc_slope = inc_1.sub(inc_0);
                const wa_slope = wa_1.sub(wa_0);
                evals[3] = evals[3].add(inc_slope.mul(wa_slope));
            }

            return evals;
        }

        /// Uses LowToHigh binding: new[i] = (1-r)*old[2*i] + r*old[2*i+1]
        pub fn bindChallengeWithPoly(self: *Self, r: F, round_poly: [4]F) void {
            _ = round_poly; // The round_poly is only used for the transcript, not for internal claim tracking

            const n = self.effectiveLen();
            const half = n / 2;
            if (half == 0) {
                self.round += 1;
                return;
            }

            const one_minus_r = F.one().sub(r);
            for (0..half) |i| {
                self.inc_evals[i] = one_minus_r.mul(self.inc_evals[2 * i]).add(r.mul(self.inc_evals[2 * i + 1]));
                self.wa_evals[i] = one_minus_r.mul(self.wa_evals[2 * i]).add(r.mul(self.wa_evals[2 * i + 1]));
            }
            for (half..n) |i| {
                self.inc_evals[i] = F.zero();
                self.wa_evals[i] = F.zero();
            }

            self.round += 1;

            // CRITICAL FIX: After binding, the new claim is the actual sum of products over the bound arrays.
            // The round_poly parameter contains the hinted polynomial (for the transcript), but the
            // prover's internal claim must track the actual polynomial sum: Σ inc[j]*wa[j]
            // This is because the hint mechanism modifies H(1) = claim - H(0) for the sumcheck invariant,
            // but the actual polynomial arrays are independent of this hint.
            var new_claim = F.zero();
            const new_len = self.effectiveLen();
            for (0..new_len) |j| {
                new_claim = new_claim.add(self.inc_evals[j].mul(self.wa_evals[j]));
            }
            self.current_claim = new_claim;
        }

        pub fn bindChallenge(self: *Self, r: F) void {
            const round_poly = self.computeRoundPolynomial();
            self.bindChallengeWithPoly(r, round_poly);
        }

        pub fn getFinalClaim(self: *const Self) F {
            return self.current_claim;
        }

        pub fn getFinalOpenings(self: *const Self) struct { inc_eval: F, wa_eval: F } {
            if (self.inc_evals.len == 0) {
                return .{ .inc_eval = F.zero(), .wa_eval = F.zero() };
            }
            return .{
                .inc_eval = self.inc_evals[0],
                .wa_eval = self.wa_evals[0],
            };
        }

        pub fn numRounds(self: *const Self) usize {
            return self.num_vars;
        }

        pub fn effectiveLen(self: *const Self) usize {
            return self.inc_evals.len >> @intCast(self.round);
        }
    };
}
