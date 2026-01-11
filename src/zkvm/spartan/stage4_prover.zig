const std = @import("std");
const Allocator = std.mem.Allocator;
const TraceStep = @import("../../tracer/mod.zig").TraceStep;
const ExecutionTrace = @import("../../tracer/mod.zig").ExecutionTrace;

/// LOG_K = 7 (128 registers in RISC-V register file)
pub const LOG_K: usize = 7;
pub const K: usize = 1 << LOG_K; // 128

/// Degree bound for RegistersReadWriteChecking round polynomials
pub const DEGREE_BOUND: usize = 3;

/// Round polynomial for Stage 4 sumcheck
/// Stores full coefficients [c0, c1, c2, c3] for a degree-3 polynomial
pub fn RoundPoly(comptime F: type) type {
    return struct {
        /// Full coefficients: c0, c1, c2, c3 for p(x) = c0 + c1*x + c2*x² + c3*x³
        coeffs: [4]F,

        /// Evaluate at a point: p(x) = c0 + c1*x + c2*x² + c3*x³
        pub fn evaluateAt(self: @This(), x: F) F {
            // Horner's method: c0 + x*(c1 + x*(c2 + x*c3))
            var result = self.coeffs[3];
            result = result.mul(x).add(self.coeffs[2]);
            result = result.mul(x).add(self.coeffs[1]);
            result = result.mul(x).add(self.coeffs[0]);
            return result;
        }
    };
}

/// Result from Stage 4 sumcheck prover
pub fn Stage4Result(comptime F: type) type {
    return struct {
        const Self = @This();

        /// Round polynomials as full coefficient arrays
        round_polys: []RoundPoly(F),
        /// Final claims at the opening point
        val_claim: F,
        rs1_ra_claim: F,
        rs2_ra_claim: F,
        rd_wa_claim: F,
        inc_claim: F,
        /// Sumcheck challenges (for debugging/verification)
        challenges: []F,

        allocator: Allocator,

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.round_polys);
            self.allocator.free(self.challenges);
        }
    };
}

/// Stage 4 Registers Read/Write Checking Prover
///
/// This sumcheck proves that register reads and writes are consistent:
/// - rd_write_value = rd_wa * (inc + val)
/// - rs1_value = rs1_ra * val
/// - rs2_value = rs2_ra * val
///
/// The sumcheck is over (address, cycle) pairs where address ∈ [0, K-1] and cycle ∈ [0, T-1].
pub fn Stage4Prover(comptime F: type) type {
    const BlakeTranscript = @import("../../transcripts/mod.zig").Blake2bTranscript(F);

    return struct {
        const Self = @This();

        allocator: Allocator,

        /// Trace length (power of 2)
        T: usize,
        log_T: usize,

        /// Total number of rounds = LOG_K + log_T
        num_rounds: usize,

        /// gamma challenge from transcript (for batching rs1/rs2)
        gamma: F,

        /// r_cycle from Stage 3 (the cycle point we're reducing to)
        r_cycle: []const F,

        // Polynomial evaluations (dense representation)
        // These are indexed as: poly[k * T + j] for address k, cycle j

        /// val(k, j) = value of register k right before cycle j
        val_poly: []F,

        /// rd_wa(k, j) = 1 if at cycle j, destination register is k
        rd_wa_poly: []F,

        /// rs1_ra(k, j) = 1 if at cycle j, source register 1 is k
        rs1_ra_poly: []F,

        /// rs2_ra(k, j) = 1 if at cycle j, source register 2 is k
        rs2_ra_poly: []F,

        /// inc(j) = change in value at cycle j (post_value - pre_value)
        /// This is only over cycles, not addresses
        inc_poly: []F,

        /// eq(r_cycle', j) evaluations - precomputed equality polynomial
        eq_cycle_evals: []F,

        pub fn init(
            allocator: Allocator,
            trace: *const ExecutionTrace,
            gamma: F,
            r_cycle: []const F,
        ) !Self {
            const trace_len = trace.steps.items.len;
            if (trace_len == 0) return error.EmptyTrace;

            // Pad to power of 2
            const T = std.math.ceilPowerOfTwo(usize, trace_len) catch trace_len;
            const log_T = @ctz(T);

            if (r_cycle.len != log_T) {
                std.debug.print("[STAGE4] r_cycle.len = {}, expected log_T = {}\n", .{r_cycle.len, log_T});
                return error.InvalidRCycleLength;
            }

            const num_rounds = LOG_K + log_T;
            const total_size = K * T;

            // Allocate polynomial arrays
            const val_poly = try allocator.alloc(F, total_size);
            const rd_wa_poly = try allocator.alloc(F, total_size);
            const rs1_ra_poly = try allocator.alloc(F, total_size);
            const rs2_ra_poly = try allocator.alloc(F, total_size);
            const inc_poly = try allocator.alloc(F, T);

            // Initialize to zero
            @memset(val_poly, F.zero());
            @memset(rd_wa_poly, F.zero());
            @memset(rs1_ra_poly, F.zero());
            @memset(rs2_ra_poly, F.zero());
            @memset(inc_poly, F.zero());

            // Track register values across cycles
            var register_values: [32]u64 = [_]u64{0} ** 32;

            // Build polynomial evaluations from trace
            for (trace.steps.items, 0..) |step, cycle| {
                if (step.is_noop) continue;

                // Extract register indices from instruction encoding
                const instr = step.instruction;
                const rd: u5 = @truncate((instr >> 7) & 0x1f);
                const rs1: u5 = @truncate((instr >> 15) & 0x1f);
                const rs2: u5 = @truncate((instr >> 20) & 0x1f);

                // Determine if registers are actually used based on instruction type
                const opcode = instr & 0x7f;

                // Set val(k, j) for all registers k - value before this cycle
                for (0..32) |k| {
                    val_poly[k * T + cycle] = F.fromU64(register_values[k]);
                }
                // Extend to full K registers (Jolt uses 128)
                for (32..K) |k| {
                    val_poly[k * T + cycle] = F.zero();
                }

                // Source register 1: used in most R-type, I-type, S-type, B-type instructions
                const rs1_used = switch (opcode) {
                    0x13, 0x03, 0x23, 0x63, 0x33, 0x3B, 0x1B, 0x67 => true, // I, LOAD, STORE, BRANCH, R, etc.
                    else => false,
                };
                if (rs1_used and rs1 < 32) {
                    rs1_ra_poly[@as(usize, rs1) * T + cycle] = F.one();
                }

                // Source register 2: used in R-type, S-type, B-type instructions
                const rs2_used = switch (opcode) {
                    0x33, 0x3B, 0x23, 0x63 => true, // R-type, S-type, B-type
                    else => false,
                };
                if (rs2_used and rs2 < 32) {
                    rs2_ra_poly[@as(usize, rs2) * T + cycle] = F.one();
                }

                // Destination register: used in most instructions except STORE, BRANCH
                const rd_used = switch (opcode) {
                    0x23, 0x63 => false, // STORE, BRANCH don't write
                    else => true,
                };
                if (rd_used and rd != 0 and rd < 32) {
                    rd_wa_poly[@as(usize, rd) * T + cycle] = F.one();

                    // Compute inc = post_value - pre_value
                    const pre_value = register_values[rd];
                    const post_value = step.rd_value;
                    // inc = post - pre in field
                    inc_poly[cycle] = F.fromU64(post_value).sub(F.fromU64(pre_value));

                    // Update register value for next cycle
                    register_values[rd] = post_value;
                }
            }

            // Precompute eq(r_cycle', j) evaluations
            const eq_cycle_evals = try allocator.alloc(F, T);
            try computeEqEvals(F, eq_cycle_evals, r_cycle);

            return Self{
                .allocator = allocator,
                .T = T,
                .log_T = log_T,
                .num_rounds = num_rounds,
                .gamma = gamma,
                .r_cycle = r_cycle,
                .val_poly = val_poly,
                .rd_wa_poly = rd_wa_poly,
                .rs1_ra_poly = rs1_ra_poly,
                .rs2_ra_poly = rs2_ra_poly,
                .inc_poly = inc_poly,
                .eq_cycle_evals = eq_cycle_evals,
            };
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.val_poly);
            self.allocator.free(self.rd_wa_poly);
            self.allocator.free(self.rs1_ra_poly);
            self.allocator.free(self.rs2_ra_poly);
            self.allocator.free(self.inc_poly);
            self.allocator.free(self.eq_cycle_evals);
        }

        /// Run the Stage 4 sumcheck and generate proof
        pub fn prove(self: *Self, transcript: *BlakeTranscript) !Stage4Result(F) {
            var round_polys = try self.allocator.alloc(RoundPoly(F), self.num_rounds);
            var challenges = try self.allocator.alloc(F, self.num_rounds);

            // Current claim starts from input_claim
            var current_claim = self.computeInputClaim();

            std.debug.print("[STAGE4] Starting sumcheck with {} rounds\n", .{self.num_rounds});
            std.debug.print("[STAGE4] Input claim = {any}\n", .{current_claim.toBytes()});

            for (0..self.num_rounds) |round| {
                // Compute round polynomial (returns full coefficients)
                const round_poly = self.computeRoundPolynomial(round, current_claim);

                // Append compressed form to transcript: [c0, c2, c3] (skip c1)
                // This matches Jolt's CompressedUniPoly format
                transcript.appendScalar(round_poly.coeffs[0]);
                transcript.appendScalar(round_poly.coeffs[2]);
                transcript.appendScalar(round_poly.coeffs[3]);

                // Get challenge
                const challenge = transcript.challengeScalar();
                challenges[round] = challenge;

                // Bind variable and update claim
                current_claim = round_poly.evaluateAt(challenge);

                // Bind polynomials (halve their size)
                self.bindPolynomials(round, challenge);

                round_polys[round] = round_poly;

                if (round < 3 or round >= self.num_rounds - 3) {
                    std.debug.print("[STAGE4] Round {}: claim = {any}, challenge = {any}\n",
                        .{round, current_claim.toBytes()[0..8], challenge.toBytes()[0..8]});
                }
            }

            std.debug.print("[STAGE4] Final claim = {any}\n", .{current_claim.toBytes()});

            // Extract final claims
            return Stage4Result(F){
                .round_polys = round_polys,
                .val_claim = self.val_poly[0],
                .rs1_ra_claim = self.rs1_ra_poly[0],
                .rs2_ra_claim = self.rs2_ra_poly[0],
                .rd_wa_claim = self.rd_wa_poly[0],
                .inc_claim = self.inc_poly[0],
                .challenges = challenges,
                .allocator = self.allocator,
            };
        }

        /// Compute the initial input claim
        fn computeInputClaim(self: *Self) F {
            var claim = F.zero();
            const gamma_sq = self.gamma.mul(self.gamma);

            for (0..K) |k| {
                for (0..self.T) |j| {
                    const idx = k * self.T + j;
                    const eq_j = self.eq_cycle_evals[j];

                    const val = self.val_poly[idx];
                    const rd_wa = self.rd_wa_poly[idx];
                    const rs1_ra = self.rs1_ra_poly[idx];
                    const rs2_ra = self.rs2_ra_poly[idx];
                    const inc = self.inc_poly[j];

                    // rd_write_value = rd_wa * (inc + val)
                    const rd_wv = rd_wa.mul(inc.add(val));
                    // rs1_value = rs1_ra * val
                    const rs1_v = rs1_ra.mul(val);
                    // rs2_value = rs2_ra * val
                    const rs2_v = rs2_ra.mul(val);

                    // Combined: rd_wv + γ*rs1_v + γ²*rs2_v
                    const combined = rd_wv.add(self.gamma.mul(rs1_v)).add(gamma_sq.mul(rs2_v));

                    claim = claim.add(eq_j.mul(combined));
                }
            }

            return claim;
        }

        /// Compute round polynomial for the given round
        /// Returns full coefficients [c0, c1, c2, c3] as a RoundPoly
        ///
        /// IMPORTANT: Jolt binds CYCLE variables first (log_T rounds), then ADDRESS variables (LOG_K rounds)
        fn computeRoundPolynomial(self: *Self, round: usize, current_claim: F) RoundPoly(F) {
            const is_cycle_round = round < self.log_T;

            // Compute evaluations at 0, 1
            var evals: [4]F = .{ F.zero(), F.zero(), F.zero(), F.zero() };

            if (is_cycle_round) {
                // Binding cycle variable (first log_T rounds)
                const cycle_var = round;
                const cycle_half = @as(usize, 1) << @intCast(self.log_T - 1 - cycle_var);

                for (0..K) |k| {
                    for (0..cycle_half) |j_lo| {
                        // X = 0
                        const j0 = j_lo;
                        const idx0 = k * self.T + j0;
                        const eq_j0 = self.eq_cycle_evals[j0];
                        const contrib0 = self.computeContribution(idx0, j0);
                        evals[0] = evals[0].add(eq_j0.mul(contrib0));

                        // X = 1
                        const j1 = j_lo + cycle_half;
                        const idx1 = k * self.T + j1;
                        const eq_j1 = self.eq_cycle_evals[j1];
                        const contrib1 = self.computeContribution(idx1, j1);
                        evals[1] = evals[1].add(eq_j1.mul(contrib1));
                    }
                }
            } else {
                // Binding address variable (next LOG_K rounds)
                const addr_var = round - self.log_T;
                const addr_half = @as(usize, 1) << @intCast(LOG_K - 1 - addr_var);

                for (0..addr_half) |addr_lo| {
                    for (0..self.T) |j| {
                        const eq_j = self.eq_cycle_evals[j];

                        // X = 0: use addresses with bit addr_var = 0
                        const k0 = addr_lo;
                        const idx0 = k0 * self.T + j;
                        const contrib0 = self.computeContribution(idx0, j);
                        evals[0] = evals[0].add(eq_j.mul(contrib0));

                        // X = 1: use addresses with bit addr_var = 1
                        const k1 = addr_lo + addr_half;
                        const idx1 = k1 * self.T + j;
                        const contrib1 = self.computeContribution(idx1, j);
                        evals[1] = evals[1].add(eq_j.mul(contrib1));
                    }
                }
            }

            // For degree 2 polynomial, extrapolate to get p(2), p(3)
            // p(X) = c0 + c1*X + c2*X² where p(0)=evals[0], p(1)=evals[1]
            // p(2) = c0 + 2*c1 + 4*c2
            // For linear extrapolation: p(2) = 2*p(1) - p(0)
            evals[2] = evals[1].mul(F.fromU64(2)).sub(evals[0]);
            evals[3] = evals[2].mul(F.fromU64(2)).sub(evals[1]);

            // Verify p(0) + p(1) = current_claim
            const sum = evals[0].add(evals[1]);
            if (!sum.eql(current_claim)) {
                std.debug.print("[STAGE4] Warning: p(0)+p(1) = {any}, expected = {any}\n",
                    .{sum.toBytes(), current_claim.toBytes()});
            }

            // Compute full coefficients [c0, c1, c2, c3]
            // For p(X) = c0 + c1*X + c2*X² + c3*X³:
            // p(0) = c0
            // p(1) = c0 + c1 + c2 + c3
            // p(0) + p(1) = claim => 2*c0 + c1 + c2 + c3 = claim
            // => c1 = claim - 2*c0 - c2 - c3
            const c0 = evals[0];
            const c3 = F.zero(); // Assume degree 2 (quadratic) for simplicity

            // For quadratic: p(2) = c0 + 2*c1 + 4*c2
            // We have c1 = claim - 2*c0 - c2
            // p(2) = c0 + 2*(claim - 2*c0 - c2) + 4*c2
            //      = c0 + 2*claim - 4*c0 - 2*c2 + 4*c2
            //      = c0 + 2*claim - 4*c0 + 2*c2
            //      = 2*claim - 3*c0 + 2*c2
            // => c2 = (p(2) - 2*claim + 3*c0) / 2
            const two = F.fromU64(2);
            const three = F.fromU64(3);
            const two_inv = two.inverse() orelse F.one(); // 2^-1 mod p
            const c2 = evals[2].sub(current_claim.mul(two)).add(c0.mul(three)).mul(two_inv);

            // c1 = claim - 2*c0 - c2 - c3
            const c1 = current_claim.sub(c0.mul(two)).sub(c2).sub(c3);

            return RoundPoly(F){
                .coeffs = .{ c0, c1, c2, c3 },
            };
        }

        /// Compute contribution for a single (address, cycle) pair
        fn computeContribution(self: *Self, idx: usize, j: usize) F {
            const val = self.val_poly[idx];
            const rd_wa = self.rd_wa_poly[idx];
            const rs1_ra = self.rs1_ra_poly[idx];
            const rs2_ra = self.rs2_ra_poly[idx];
            const inc = self.inc_poly[j];

            const gamma_sq = self.gamma.mul(self.gamma);

            const rd_wv = rd_wa.mul(inc.add(val));
            const rs1_v = rs1_ra.mul(val);
            const rs2_v = rs2_ra.mul(val);

            return rd_wv.add(self.gamma.mul(rs1_v)).add(gamma_sq.mul(rs2_v));
        }

        /// Bind polynomials after receiving a challenge
        /// IMPORTANT: Jolt binds CYCLE variables first (log_T rounds), then ADDRESS variables (LOG_K rounds)
        fn bindPolynomials(self: *Self, round: usize, challenge: F) void {
            const is_cycle_round = round < self.log_T;

            if (is_cycle_round) {
                // Binding cycle variable (first log_T rounds)
                const cycle_var = round;
                const cycle_half = @as(usize, 1) << @intCast(self.log_T - 1 - cycle_var);
                const one_minus_c = F.one().sub(challenge);

                for (0..K) |k| {
                    for (0..cycle_half) |j| {
                        const idx_lo = k * self.T + j;
                        const idx_hi = k * self.T + j + cycle_half;

                        self.val_poly[idx_lo] = self.val_poly[idx_lo].mul(one_minus_c).add(self.val_poly[idx_hi].mul(challenge));
                        self.rd_wa_poly[idx_lo] = self.rd_wa_poly[idx_lo].mul(one_minus_c).add(self.rd_wa_poly[idx_hi].mul(challenge));
                        self.rs1_ra_poly[idx_lo] = self.rs1_ra_poly[idx_lo].mul(one_minus_c).add(self.rs1_ra_poly[idx_hi].mul(challenge));
                        self.rs2_ra_poly[idx_lo] = self.rs2_ra_poly[idx_lo].mul(one_minus_c).add(self.rs2_ra_poly[idx_hi].mul(challenge));
                    }
                }

                // Also bind inc_poly and eq_cycle_evals (only during cycle rounds)
                for (0..cycle_half) |j| {
                    self.inc_poly[j] = self.inc_poly[j].mul(one_minus_c).add(self.inc_poly[j + cycle_half].mul(challenge));
                    self.eq_cycle_evals[j] = self.eq_cycle_evals[j].mul(one_minus_c).add(self.eq_cycle_evals[j + cycle_half].mul(challenge));
                }
            } else {
                // Binding address variable (next LOG_K rounds)
                const addr_var = round - self.log_T;
                const addr_half = @as(usize, 1) << @intCast(LOG_K - 1 - addr_var);
                const one_minus_c = F.one().sub(challenge);

                for (0..addr_half) |k| {
                    for (0..self.T) |j| {
                        const idx_lo = k * self.T + j;
                        const idx_hi = (k + addr_half) * self.T + j;

                        self.val_poly[idx_lo] = self.val_poly[idx_lo].mul(one_minus_c).add(self.val_poly[idx_hi].mul(challenge));
                        self.rd_wa_poly[idx_lo] = self.rd_wa_poly[idx_lo].mul(one_minus_c).add(self.rd_wa_poly[idx_hi].mul(challenge));
                        self.rs1_ra_poly[idx_lo] = self.rs1_ra_poly[idx_lo].mul(one_minus_c).add(self.rs1_ra_poly[idx_hi].mul(challenge));
                        self.rs2_ra_poly[idx_lo] = self.rs2_ra_poly[idx_lo].mul(one_minus_c).add(self.rs2_ra_poly[idx_hi].mul(challenge));
                    }
                }
            }
        }
    };
}

/// Compute eq polynomial evaluations: eq(r, i) for i in [0, 2^n)
fn computeEqEvals(comptime F: type, output: []F, r: []const F) !void {
    const n = r.len;
    const size = @as(usize, 1) << @intCast(n);

    if (output.len < size) return error.OutputTooSmall;

    // Initialize
    output[0] = F.one();

    for (r, 0..) |r_i, i| {
        const half = @as(usize, 1) << @intCast(i);
        const one_minus_r_i = F.one().sub(r_i);

        var j = half;
        while (j > 0) {
            j -= 1;
            output[j + half] = output[j].mul(r_i);
            output[j] = output[j].mul(one_minus_r_i);
        }
    }
}
