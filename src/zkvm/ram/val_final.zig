//! RAM Final Value Evaluation Sumcheck
//!
//! Proves:
//!   Val_final(r_address) - Val_init(r_address) = Σ_{j=0}^{T-1} inc(j) · wa(r_address, j)
//!
//! This is degree-2 (product of two multilinear polynomials).

const std = @import("std");
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

        pub fn init(
            allocator: Allocator,
            trace: *const MemoryTrace,
            initial_ram: ?*const std.AutoHashMapUnmanaged(u64, u64),
            params: ValFinalParams(F),
            start_address: u64,
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

            // Compute initial claim
            var initial_claim = F.zero();
            for (0..n) |j| {
                initial_claim = initial_claim.add(inc_evals[j].mul(wa_evals[j]));
            }

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

        /// Compute round polynomial [p(0), p(1), p(2), p(3)]
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
            const three = F.fromU64(3);

            for (0..half) |i| {
                const inc_0 = self.inc_evals[i];
                const wa_0 = self.wa_evals[i];
                const inc_1 = self.inc_evals[i + half];
                const wa_1 = self.wa_evals[i + half];

                // p(0) and p(1)
                evals[0] = evals[0].add(inc_0.mul(wa_0));
                evals[1] = evals[1].add(inc_1.mul(wa_1));

                // Extrapolate to x=2,3 for each polynomial, then multiply
                const inc_2 = two.mul(inc_1).sub(inc_0);
                const wa_2 = two.mul(wa_1).sub(wa_0);
                evals[2] = evals[2].add(inc_2.mul(wa_2));

                const inc_3 = three.mul(inc_1).sub(two.mul(inc_0));
                const wa_3 = three.mul(wa_1).sub(two.mul(wa_0));
                evals[3] = evals[3].add(inc_3.mul(wa_3));
            }

            return evals;
        }

        pub fn bindChallengeWithPoly(self: *Self, r: F, round_poly: [4]F) void {
            const n = self.effectiveLen();
            const half = n / 2;
            if (half == 0) {
                self.round += 1;
                return;
            }

            const one_minus_r = F.one().sub(r);
            for (0..half) |i| {
                self.inc_evals[i] = one_minus_r.mul(self.inc_evals[i]).add(r.mul(self.inc_evals[i + half]));
                self.wa_evals[i] = one_minus_r.mul(self.wa_evals[i]).add(r.mul(self.wa_evals[i + half]));
            }
            for (half..n) |i| {
                self.inc_evals[i] = F.zero();
                self.wa_evals[i] = F.zero();
            }

            // Update current claim using cubic interpolation (degree <= 2 is safe).
            const two = F.fromU64(2);
            const three = F.fromU64(3);
            const six = F.fromU64(6);

            const r_minus_1 = r.sub(F.one());
            const r_minus_2 = r.sub(two);
            const r_minus_3 = r.sub(three);

            const L0 = r_minus_1.mul(r_minus_2).mul(r_minus_3).mul(six.neg().inverse().?);
            const L1 = r.mul(r_minus_2).mul(r_minus_3).mul(two.inverse().?);
            const L2 = r.mul(r_minus_1).mul(r_minus_3).mul(two.neg().inverse().?);
            const L3 = r.mul(r_minus_1).mul(r_minus_2).mul(six.inverse().?);

            self.current_claim = round_poly[0].mul(L0)
                .add(round_poly[1].mul(L1))
                .add(round_poly[2].mul(L2))
                .add(round_poly[3].mul(L3));

            self.round += 1;
        }

        pub fn bindChallenge(self: *Self, r: F) void {
            const round_poly = self.computeRoundPolynomial();
            self.bindChallengeWithPoly(r, round_poly);
        }

        pub fn getFinalClaim(self: *const Self) F {
            if (self.inc_evals.len == 0) return F.zero();
            return self.inc_evals[0].mul(self.wa_evals[0]);
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
