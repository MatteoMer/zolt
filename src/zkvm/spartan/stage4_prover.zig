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

/// Stage 3 claims passed to Stage 4
pub fn Stage3Claims(comptime F: type) type {
    return struct {
        rd_write_value: F,
        rs1_value: F,
        rs2_value: F,
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

        /// Stage 3 claims for input claim computation
        stage3_claims: ?Stage3Claims(F),

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

        /// Current effective sizes (halve after each binding)
        current_T: usize,
        current_K: usize,

        pub fn init(
            allocator: Allocator,
            trace: *const ExecutionTrace,
            gamma: F,
            r_cycle: []const F,
        ) !Self {
            return initWithClaims(allocator, trace, gamma, r_cycle, null);
        }

        pub fn initWithClaims(
            allocator: Allocator,
            trace: *const ExecutionTrace,
            gamma: F,
            r_cycle: []const F,
            stage3_claims: ?Stage3Claims(F),
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
            // Debug: Print trace info
            std.debug.print("[STAGE4] Building polynomials from {} trace steps, T={}\n", .{ trace.steps.items.len, T });

            for (trace.steps.items, 0..) |step, cycle| {
                if (step.is_noop) {
                    if (cycle < 5) {
                        std.debug.print("[STAGE4] Cycle {}: NOOP (skipping)\n", .{cycle});
                    }
                    continue;
                }

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

                // Compute rd_write_value for debug comparison
                var stage4_rd_wv: u64 = 0;
                if (rd_used and rd != 0 and rd < 32) {
                    rd_wa_poly[@as(usize, rd) * T + cycle] = F.one();

                    // Compute inc = post_value - pre_value
                    const pre_value = register_values[rd];
                    const post_value = step.rd_value;
                    // inc = post - pre in field
                    inc_poly[cycle] = F.fromU64(post_value).sub(F.fromU64(pre_value));

                    // rd_write_value = post_value (what Stage 3 should have)
                    stage4_rd_wv = post_value;

                    // Update register value for next cycle
                    register_values[rd] = post_value;
                }

                // Debug: Print first few cycles for comparison
                if (cycle < 5) {
                    std.debug.print("[STAGE4] Cycle {}: opcode=0x{x}, rd={}, rd_used={}, rd_wv={}\n", .{ cycle, opcode, rd, rd_used, stage4_rd_wv });
                    std.debug.print("[STAGE4]   rs1={}, rs1_used={}, rs1_val={}\n", .{ rs1, rs1_used, register_values[if (rs1 < 32) rs1 else 0] });
                    std.debug.print("[STAGE4]   rs2={}, rs2_used={}, rs2_val={}\n", .{ rs2, rs2_used, register_values[if (rs2 < 32) rs2 else 0] });
                }
            }

            // Precompute eq(r_cycle', j) evaluations
            const eq_cycle_evals = try allocator.alloc(F, T);
            try computeEqEvals(F, eq_cycle_evals, r_cycle);

            // Debug: Compute simple MLE sums to verify eq polynomial
            // These should match Stage 3's claims
            var rd_wv_sum = F.zero();
            var rs1_v_sum = F.zero();
            var rs2_v_sum = F.zero();

            // Need to track rs1/rs2 values like Stage 3 does (from R1CS witness, not trace reconstruction)
            // Stage 3 uses: witness[R1CSInputIndex.Rs1Value] and witness[R1CSInputIndex.Rs2Value]
            // These are step.rs1_value and step.rs2_value from trace
            for (trace.steps.items, 0..) |step, cycle| {
                if (step.is_noop) continue;
                const instr = step.instruction;
                const rd: u5 = @truncate((instr >> 7) & 0x1f);
                const opcode = instr & 0x7f;
                const rd_used = switch (opcode) {
                    0x23, 0x63 => false,
                    else => true,
                };
                const rd_wv = if (rd_used and rd != 0) F.fromU64(step.rd_value) else F.zero();
                rd_wv_sum = rd_wv_sum.add(eq_cycle_evals[cycle].mul(rd_wv));

                // Rs1Value and Rs2Value from R1CS witness - only for instructions that read them
                // This matches Jolt's cycle.rs1_read().unwrap_or_default().1 behavior
                const reads_rs1 = switch (opcode) {
                    0x13, 0x03, 0x67, 0x1b, 0x33, 0x3b, 0x23, 0x63 => true,
                    else => false,
                };
                if (reads_rs1) {
                    rs1_v_sum = rs1_v_sum.add(eq_cycle_evals[cycle].mul(F.fromU64(step.rs1_value)));
                }

                const reads_rs2 = switch (opcode) {
                    0x33, 0x3b, 0x23, 0x63 => true,
                    else => false,
                };
                if (reads_rs2) {
                    rs2_v_sum = rs2_v_sum.add(eq_cycle_evals[cycle].mul(F.fromU64(step.rs2_value)));
                }
            }
            std.debug.print("[STAGE4] Simple rd_wv MLE sum = {any}\n", .{rd_wv_sum.toBytes()});
            std.debug.print("[STAGE4] Simple rs1_v MLE sum = {any}\n", .{rs1_v_sum.toBytes()});
            std.debug.print("[STAGE4] Simple rs2_v MLE sum = {any}\n", .{rs2_v_sum.toBytes()});
            if (stage3_claims) |claims| {
                std.debug.print("[STAGE4] Expected rd_wv from Stage 3 = {any}\n", .{claims.rd_write_value.toBytes()});
                std.debug.print("[STAGE4] Expected rs1_v from Stage 3 = {any}\n", .{claims.rs1_value.toBytes()});
                std.debug.print("[STAGE4] Expected rs2_v from Stage 3 = {any}\n", .{claims.rs2_value.toBytes()});

                // Compute expected input_claim the simple way
                const simple_input_claim = rd_wv_sum.add(gamma.mul(rs1_v_sum)).add(gamma.mul(gamma).mul(rs2_v_sum));
                std.debug.print("[STAGE4] Simple input_claim = {any}\n", .{simple_input_claim.toBytes()});
            }

            return Self{
                .allocator = allocator,
                .T = T,
                .log_T = log_T,
                .num_rounds = num_rounds,
                .gamma = gamma,
                .r_cycle = r_cycle,
                .stage3_claims = stage3_claims,
                .val_poly = val_poly,
                .rd_wa_poly = rd_wa_poly,
                .rs1_ra_poly = rs1_ra_poly,
                .rs2_ra_poly = rs2_ra_poly,
                .inc_poly = inc_poly,
                .eq_cycle_evals = eq_cycle_evals,
                .current_T = T,
                .current_K = K,
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

            // Compute input claim from trace polynomials (for comparison)
            const computed_claim = self.computeInputClaim();

            // If Stage 3 claims are provided, compute expected input claim from them
            const gamma_sq = self.gamma.mul(self.gamma);
            var expected_claim: ?F = null;
            if (self.stage3_claims) |claims| {
                // input_claim = rd_wv + gamma * (rs1_v + gamma * rs2_v)
                //             = rd_wv + gamma * rs1_v + gamma^2 * rs2_v
                const gamma_rs1 = self.gamma.mul(claims.rs1_value);
                const gamma_sq_rs2 = gamma_sq.mul(claims.rs2_value);
                const sum_gamma = gamma_rs1.add(gamma_sq_rs2);
                expected_claim = claims.rd_write_value.add(sum_gamma);

                std.debug.print("[STAGE4] Stage 3 claims:\n", .{});
                std.debug.print("[STAGE4]   rd_wv = {any}\n", .{claims.rd_write_value.toBytes()});
                std.debug.print("[STAGE4]   rs1_v = {any}\n", .{claims.rs1_value.toBytes()});
                std.debug.print("[STAGE4]   rs1_v LIMBS (Montgomery) = {x}, {x}, {x}, {x}\n", .{
                    claims.rs1_value.limbs[0], claims.rs1_value.limbs[1],
                    claims.rs1_value.limbs[2], claims.rs1_value.limbs[3],
                });
                std.debug.print("[STAGE4]   rs2_v = {any}\n", .{claims.rs2_value.toBytes()});
                std.debug.print("[STAGE4]   gamma = {any}\n", .{self.gamma.toBytes()});
                std.debug.print("[STAGE4]   gamma LIMBS (Montgomery) = {x}, {x}, {x}, {x}\n", .{
                    self.gamma.limbs[0], self.gamma.limbs[1],
                    self.gamma.limbs[2], self.gamma.limbs[3],
                });
                std.debug.print("[STAGE4]   gamma^2 = {any}\n", .{gamma_sq.toBytes()});
                std.debug.print("[STAGE4]   gamma * rs1_v = {any}\n", .{gamma_rs1.toBytes()});
                std.debug.print("[STAGE4]   gamma * rs1_v LIMBS = {x}, {x}, {x}, {x}\n", .{
                    gamma_rs1.limbs[0], gamma_rs1.limbs[1],
                    gamma_rs1.limbs[2], gamma_rs1.limbs[3],
                });
                std.debug.print("[STAGE4]   gamma^2 * rs2_v = {any}\n", .{gamma_sq_rs2.toBytes()});
                std.debug.print("[STAGE4]   expected_input_claim = {any}\n", .{expected_claim.?.toBytes()});

                // Verify alternative computation: gamma * (rs1_v + gamma * rs2_v)
                const gamma_rs2 = self.gamma.mul(claims.rs2_value);
                const inner_sum = claims.rs1_value.add(gamma_rs2);
                const gamma_inner = self.gamma.mul(inner_sum);
                const alt_result = claims.rd_write_value.add(gamma_inner);
                std.debug.print("[STAGE4]   alt_result (using gamma*(rs1+gamma*rs2)) = {any}\n", .{alt_result.toBytes()});
                std.debug.print("[STAGE4]   alt_result matches expected? {}\n", .{alt_result.eql(expected_claim.?)});
            }

            // Use expected claim from Stage 3 if available, otherwise use computed
            var current_claim = if (expected_claim) |exp| exp else computed_claim;

            std.debug.print("[STAGE4] Starting sumcheck with {} rounds (log_T={}, LOG_K={})\n", .{ self.num_rounds, self.log_T, LOG_K });
            std.debug.print("[STAGE4] T={}, K={}\n", .{ self.T, K });
            std.debug.print("[STAGE4] Computed input claim = {any}\n", .{computed_claim.toBytes()});
            std.debug.print("[STAGE4] Using input claim = {any}\n", .{current_claim.toBytes()});
            std.debug.print("[STAGE4] gamma = {any}\n", .{self.gamma.toBytes()});
            std.debug.print("[STAGE4] r_cycle.len = {}, r_cycle[0] = {any}\n", .{ self.r_cycle.len, self.r_cycle[0].toBytes()[0..8] });

            // Check if computed claim matches expected claim
            if (expected_claim) |exp| {
                if (!computed_claim.eql(exp)) {
                    std.debug.print("[STAGE4] WARNING: Computed claim != Expected claim from Stage 3!\n", .{});
                    std.debug.print("[STAGE4]   This means the polynomial construction doesn't match Stage 3 witnesses\n", .{});
                } else {
                    std.debug.print("[STAGE4] OK: Computed claim matches expected claim from Stage 3\n", .{});
                }
            }

            for (0..self.num_rounds) |round| {
                // Compute round polynomial (returns full coefficients)
                const round_poly = self.computeRoundPolynomial(round, current_claim);

                // Debug: print round polynomial coefficients for comparison with Jolt
                if (round < 3) {
                    std.debug.print("[STAGE4_COEFF] Round {}: c0 = {any}\n", .{ round, round_poly.coeffs[0].toBytes() });
                    std.debug.print("[STAGE4_COEFF] Round {}: c1 = {any}\n", .{ round, round_poly.coeffs[1].toBytes() });
                    std.debug.print("[STAGE4_COEFF] Round {}: c2 = {any}\n", .{ round, round_poly.coeffs[2].toBytes() });
                    std.debug.print("[STAGE4_COEFF] Round {}: c3 = {any}\n", .{ round, round_poly.coeffs[3].toBytes() });
                    std.debug.print("[STAGE4_COEFF] Round {}: current_claim = {any}\n", .{ round, current_claim.toBytes() });

                    // Verify p(0) + p(1) = current_claim
                    const p0 = round_poly.coeffs[0];
                    const p1 = round_poly.coeffs[0].add(round_poly.coeffs[1]).add(round_poly.coeffs[2]).add(round_poly.coeffs[3]);
                    const sum = p0.add(p1);
                    std.debug.print("[STAGE4_COEFF] Round {}: p(0) = {any}\n", .{ round, p0.toBytes() });
                    std.debug.print("[STAGE4_COEFF] Round {}: p(1) = {any}\n", .{ round, p1.toBytes() });
                    std.debug.print("[STAGE4_COEFF] Round {}: p(0)+p(1) = {any}\n", .{ round, sum.toBytes() });
                    std.debug.print("[STAGE4_COEFF] Round {}: matches claim? {}\n", .{ round, sum.eql(current_claim) });
                }

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

            // Debug: print final polynomial evaluations and verify expected formula
            const val_claim = self.val_poly[0];
            const rs1_ra_claim = self.rs1_ra_poly[0];
            const rs2_ra_claim = self.rs2_ra_poly[0];
            const rd_wa_claim = self.rd_wa_poly[0];
            const inc_claim = self.inc_poly[0];
            const eq_claim = self.eq_cycle_evals[0];

            std.debug.print("[STAGE4_FINAL] val_claim = {any}\n", .{val_claim.toBytes()});
            std.debug.print("[STAGE4_FINAL] rs1_ra_claim = {any}\n", .{rs1_ra_claim.toBytes()});
            std.debug.print("[STAGE4_FINAL] rs2_ra_claim = {any}\n", .{rs2_ra_claim.toBytes()});
            std.debug.print("[STAGE4_FINAL] rd_wa_claim = {any}\n", .{rd_wa_claim.toBytes()});
            std.debug.print("[STAGE4_FINAL] inc_claim = {any}\n", .{inc_claim.toBytes()});
            std.debug.print("[STAGE4_FINAL] eq_claim (after binding) = {any}\n", .{eq_claim.toBytes()});

            // Compute expected output claim using the verifier's formula:
            // eq(r_cycle, r_cycle') * (rd_wa * (inc + val) + γ * rs1_ra * val + γ² * rs2_ra * val)
            const rd_wv = rd_wa_claim.mul(inc_claim.add(val_claim));
            const rs1_v = rs1_ra_claim.mul(val_claim);
            const rs2_v = rs2_ra_claim.mul(val_claim);
            const gamma_sq_final = self.gamma.mul(self.gamma);
            const combined = rd_wv.add(self.gamma.mul(rs1_v)).add(gamma_sq_final.mul(rs2_v));
            const computed_expected_claim = eq_claim.mul(combined);

            std.debug.print("[STAGE4_FINAL] rd_wv = {any}\n", .{rd_wv.toBytes()});
            std.debug.print("[STAGE4_FINAL] rs1_v = {any}\n", .{rs1_v.toBytes()});
            std.debug.print("[STAGE4_FINAL] rs2_v = {any}\n", .{rs2_v.toBytes()});
            std.debug.print("[STAGE4_FINAL] combined (rd_wv + γ*rs1_v + γ²*rs2_v) = {any}\n", .{combined.toBytes()});
            std.debug.print("[STAGE4_FINAL] computed expected claim (eq * combined) = {any}\n", .{computed_expected_claim.toBytes()});
            std.debug.print("[STAGE4_FINAL] actual final claim = {any}\n", .{current_claim.toBytes()});
            std.debug.print("[STAGE4_FINAL] match? {}\n", .{computed_expected_claim.eql(current_claim)});

            // Extract final claims
            return Stage4Result(F){
                .round_polys = round_polys,
                .val_claim = val_claim,
                .rs1_ra_claim = rs1_ra_claim,
                .rs2_ra_claim = rs2_ra_claim,
                .rd_wa_claim = rd_wa_claim,
                .inc_claim = inc_claim,
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
        /// After each binding, current_T or current_K halves, and live values are at positions 0..current/2-1
        ///
        /// KEY INSIGHT: For a product of multilinear polynomials, we must compute univariate
        /// restrictions for each component separately, then multiply pointwise. This is because:
        ///   (f * g)(t) = f(t) * g(t)  for each evaluation point t
        /// But the product's univariate polynomial is NOT the product of the univariate polynomials
        /// evaluated separately at different indices.
        fn computeRoundPolynomial(self: *Self, round: usize, current_claim: F) RoundPoly(F) {
            const is_cycle_round = round < self.log_T;
            const gamma_sq = self.gamma.mul(self.gamma);

            // We need evaluations at 0, 1, 2, 3 for a degree-3 polynomial
            // The polynomial is: eq * [rd_wa * (val + inc) + gamma * rs1_ra * val + gamma^2 * rs2_ra * val]
            // This is degree 3: eq (degree 1) * [ products of degree-1 terms ]
            var evals: [4]F = .{ F.zero(), F.zero(), F.zero(), F.zero() };

            if (is_cycle_round) {
                // Binding cycle variable (first log_T rounds)
                // For each pair (j_even, j_odd), compute univariate restriction evals
                const half_T = self.current_T / 2;

                for (0..self.current_K) |k| {
                    for (0..half_T) |i| {
                        const j0 = 2 * i; // Even index -> X = 0
                        const j1 = 2 * i + 1; // Odd index -> X = 1
                        const idx0 = k * self.T + j0;
                        const idx1 = k * self.T + j1;

                        // Get component values at j0 (X=0) and j1 (X=1)
                        const eq_0 = self.eq_cycle_evals[j0];
                        const eq_1 = self.eq_cycle_evals[j1];
                        const val_0 = self.val_poly[idx0];
                        const val_1 = self.val_poly[idx1];
                        const rd_wa_0 = self.rd_wa_poly[idx0];
                        const rd_wa_1 = self.rd_wa_poly[idx1];
                        const rs1_ra_0 = self.rs1_ra_poly[idx0];
                        const rs1_ra_1 = self.rs1_ra_poly[idx1];
                        const rs2_ra_0 = self.rs2_ra_poly[idx0];
                        const rs2_ra_1 = self.rs2_ra_poly[idx1];
                        const inc_0 = self.inc_poly[j0];
                        const inc_1 = self.inc_poly[j1];

                        // Compute univariate restriction evals at t = 0, 1, 2, 3
                        // For a linear polynomial with values f0, f1:
                        // f(t) = f0 + t * (f1 - f0) = (1-t)*f0 + t*f1
                        inline for (0..4) |t| {
                            const t_field = F.fromU64(t);
                            const one_minus_t = F.one().sub(t_field);

                            // Interpolate each component: comp(t) = (1-t)*comp_0 + t*comp_1
                            const eq_t = one_minus_t.mul(eq_0).add(t_field.mul(eq_1));
                            const val_t = one_minus_t.mul(val_0).add(t_field.mul(val_1));
                            const rd_wa_t = one_minus_t.mul(rd_wa_0).add(t_field.mul(rd_wa_1));
                            const rs1_ra_t = one_minus_t.mul(rs1_ra_0).add(t_field.mul(rs1_ra_1));
                            const rs2_ra_t = one_minus_t.mul(rs2_ra_0).add(t_field.mul(rs2_ra_1));
                            const inc_t = one_minus_t.mul(inc_0).add(t_field.mul(inc_1));

                            // Compute the combined contribution at this evaluation point
                            // rd_wv = rd_wa * (val + inc)
                            // rs1_v = rs1_ra * val
                            // rs2_v = rs2_ra * val
                            // combined = rd_wv + gamma * rs1_v + gamma^2 * rs2_v
                            const rd_wv_t = rd_wa_t.mul(val_t.add(inc_t));
                            const rs1_v_t = rs1_ra_t.mul(val_t);
                            const rs2_v_t = rs2_ra_t.mul(val_t);
                            const combined_t = rd_wv_t.add(self.gamma.mul(rs1_v_t)).add(gamma_sq.mul(rs2_v_t));

                            // Multiply by eq and accumulate
                            evals[t] = evals[t].add(eq_t.mul(combined_t));
                        }
                    }
                }
            } else {
                // Binding address variable (next LOG_K rounds)
                // For each pair (k_even, k_odd), compute univariate restriction evals
                const half_K = self.current_K / 2;

                for (0..half_K) |i| {
                    const k0 = 2 * i; // Even address -> X = 0
                    const k1 = 2 * i + 1; // Odd address -> X = 1

                    for (0..self.current_T) |j| {
                        const idx0 = k0 * self.T + j;
                        const idx1 = k1 * self.T + j;
                        const eq_j = self.eq_cycle_evals[j];
                        const inc_j = self.inc_poly[j];

                        // Get component values at k0 (X=0) and k1 (X=1)
                        const val_0 = self.val_poly[idx0];
                        const val_1 = self.val_poly[idx1];
                        const rd_wa_0 = self.rd_wa_poly[idx0];
                        const rd_wa_1 = self.rd_wa_poly[idx1];
                        const rs1_ra_0 = self.rs1_ra_poly[idx0];
                        const rs1_ra_1 = self.rs1_ra_poly[idx1];
                        const rs2_ra_0 = self.rs2_ra_poly[idx0];
                        const rs2_ra_1 = self.rs2_ra_poly[idx1];

                        // Compute univariate restriction evals at t = 0, 1, 2, 3
                        // eq and inc don't depend on k, so they're constant for this variable
                        inline for (0..4) |t| {
                            const t_field = F.fromU64(t);
                            const one_minus_t = F.one().sub(t_field);

                            // Interpolate address-dependent components
                            const val_t = one_minus_t.mul(val_0).add(t_field.mul(val_1));
                            const rd_wa_t = one_minus_t.mul(rd_wa_0).add(t_field.mul(rd_wa_1));
                            const rs1_ra_t = one_minus_t.mul(rs1_ra_0).add(t_field.mul(rs1_ra_1));
                            const rs2_ra_t = one_minus_t.mul(rs2_ra_0).add(t_field.mul(rs2_ra_1));

                            // Compute combined contribution
                            const rd_wv_t = rd_wa_t.mul(val_t.add(inc_j));
                            const rs1_v_t = rs1_ra_t.mul(val_t);
                            const rs2_v_t = rs2_ra_t.mul(val_t);
                            const combined_t = rd_wv_t.add(self.gamma.mul(rs1_v_t)).add(gamma_sq.mul(rs2_v_t));

                            // Multiply by eq (constant for address rounds) and accumulate
                            evals[t] = evals[t].add(eq_j.mul(combined_t));
                        }
                    }
                }
            }

            // Verify p(0) + p(1) = current_claim
            const sum = evals[0].add(evals[1]);
            if (!sum.eql(current_claim)) {
                if (round < 3) {
                    std.debug.print("[STAGE4] Round {} mismatch: p(0)+p(1) = {any}\n", .{ round, sum.toBytes()[0..16] });
                    std.debug.print("[STAGE4]   expected = {any}\n", .{current_claim.toBytes()[0..16]});
                    std.debug.print("[STAGE4]   is_cycle={}, current_T={}, current_K={}\n", .{ is_cycle_round, self.current_T, self.current_K });
                }
            }

            // Compute coefficients from evaluations using Lagrange interpolation
            // For degree-3 polynomial with evals at 0, 1, 2, 3:
            // p(x) = sum_{i=0}^{3} evals[i] * L_i(x)
            // where L_i(x) = prod_{j != i} (x - j) / (i - j)
            //
            // We need coefficients c0, c1, c2, c3 where p(x) = c0 + c1*x + c2*x² + c3*x³
            const c0 = evals[0];

            // Use the constraint p(0) + p(1) = claim to help verify
            // c1 can be derived from the other evaluations

            // Lagrange interpolation for coefficients:
            // Given p(0), p(1), p(2), p(3), we can solve for c0, c1, c2, c3
            //
            // p(0) = c0
            // p(1) = c0 + c1 + c2 + c3
            // p(2) = c0 + 2*c1 + 4*c2 + 8*c3
            // p(3) = c0 + 3*c1 + 9*c2 + 27*c3
            //
            // Solving:
            // c1 = (-11*p(0) + 18*p(1) - 9*p(2) + 2*p(3)) / 6
            // c2 = (2*p(0) - 5*p(1) + 4*p(2) - p(3)) / 2
            // c3 = (-p(0) + 3*p(1) - 3*p(2) + p(3)) / 6
            const six = F.fromU64(6);
            const six_inv = six.inverse() orelse F.one();
            const two = F.fromU64(2);
            const two_inv = two.inverse() orelse F.one();

            // c3 = (-p(0) + 3*p(1) - 3*p(2) + p(3)) / 6
            const c3 = evals[0].neg()
                .add(evals[1].mul(F.fromU64(3)))
                .sub(evals[2].mul(F.fromU64(3)))
                .add(evals[3])
                .mul(six_inv);

            // c2 = (2*p(0) - 5*p(1) + 4*p(2) - p(3)) / 2
            const c2 = evals[0].mul(two)
                .sub(evals[1].mul(F.fromU64(5)))
                .add(evals[2].mul(F.fromU64(4)))
                .sub(evals[3])
                .mul(two_inv);

            // c1 = p(1) - p(0) - c2 - c3 (from p(1) = c0 + c1 + c2 + c3)
            const c1 = evals[1].sub(evals[0]).sub(c2).sub(c3);

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
        /// Uses low-to-high binding: fold pairs (2i, 2i+1) into position i
        fn bindPolynomials(self: *Self, round: usize, challenge: F) void {
            const is_cycle_round = round < self.log_T;
            const one_minus_c = F.one().sub(challenge);

            if (is_cycle_round) {
                // Binding cycle variable (first log_T rounds)
                // Low-to-high binding: fold pairs (2i, 2i+1) into position i
                const half_T = self.current_T / 2;

                for (0..self.current_K) |k| {
                    for (0..half_T) |i| {
                        const j_lo = 2 * i;
                        const j_hi = 2 * i + 1;
                        const idx_lo = k * self.T + j_lo;
                        const idx_hi = k * self.T + j_hi;

                        // Fold: new[i] = (1-c)*old[2i] + c*old[2i+1]
                        // Store result at position i (not j_lo)
                        const new_idx = k * self.T + i;
                        self.val_poly[new_idx] = self.val_poly[idx_lo].mul(one_minus_c).add(self.val_poly[idx_hi].mul(challenge));
                        self.rd_wa_poly[new_idx] = self.rd_wa_poly[idx_lo].mul(one_minus_c).add(self.rd_wa_poly[idx_hi].mul(challenge));
                        self.rs1_ra_poly[new_idx] = self.rs1_ra_poly[idx_lo].mul(one_minus_c).add(self.rs1_ra_poly[idx_hi].mul(challenge));
                        self.rs2_ra_poly[new_idx] = self.rs2_ra_poly[idx_lo].mul(one_minus_c).add(self.rs2_ra_poly[idx_hi].mul(challenge));
                    }
                }

                // Also bind inc_poly and eq_cycle_evals
                for (0..half_T) |i| {
                    const j_lo = 2 * i;
                    const j_hi = 2 * i + 1;
                    self.inc_poly[i] = self.inc_poly[j_lo].mul(one_minus_c).add(self.inc_poly[j_hi].mul(challenge));
                    self.eq_cycle_evals[i] = self.eq_cycle_evals[j_lo].mul(one_minus_c).add(self.eq_cycle_evals[j_hi].mul(challenge));
                }

                // Update current cycle size
                self.current_T = half_T;
            } else {
                // Binding address variable (next LOG_K rounds)
                // Low-to-high binding: fold pairs (2i, 2i+1) into position i
                const half_K = self.current_K / 2;

                for (0..half_K) |i| {
                    const k_lo = 2 * i;
                    const k_hi = 2 * i + 1;

                    for (0..self.current_T) |j| {
                        const idx_lo = k_lo * self.T + j;
                        const idx_hi = k_hi * self.T + j;
                        const new_idx = i * self.T + j;

                        self.val_poly[new_idx] = self.val_poly[idx_lo].mul(one_minus_c).add(self.val_poly[idx_hi].mul(challenge));
                        self.rd_wa_poly[new_idx] = self.rd_wa_poly[idx_lo].mul(one_minus_c).add(self.rd_wa_poly[idx_hi].mul(challenge));
                        self.rs1_ra_poly[new_idx] = self.rs1_ra_poly[idx_lo].mul(one_minus_c).add(self.rs1_ra_poly[idx_hi].mul(challenge));
                        self.rs2_ra_poly[new_idx] = self.rs2_ra_poly[idx_lo].mul(one_minus_c).add(self.rs2_ra_poly[idx_hi].mul(challenge));
                    }
                }

                // Update current address size
                self.current_K = half_K;
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
