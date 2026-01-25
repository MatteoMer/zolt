const std = @import("std");
const Allocator = std.mem.Allocator;
const TraceStep = @import("../../tracer/mod.zig").TraceStep;
const ExecutionTrace = @import("../../tracer/mod.zig").ExecutionTrace;
const gruen_eq = @import("gruen_eq.zig");
const GruenSplitEqPolynomial = gruen_eq.GruenSplitEqPolynomial;

/// LOG_K = 7 (128 registers in RISC-V register file) - matches Jolt
pub const LOG_K: usize = 7;
pub const K: usize = 1 << LOG_K; // 128

/// Degree bound for RegistersReadWriteChecking round polynomials
pub const DEGREE_BOUND: usize = 3;

/// Round polynomial for Stage 4 sumcheck
pub fn RoundPoly(comptime F: type) type {
    return struct {
        coeffs: [4]F,

        pub fn evaluateAt(self: @This(), x: F) F {
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
        round_polys: []RoundPoly(F),
        val_claim: F,
        rs1_ra_claim: F,
        rs2_ra_claim: F,
        rd_wa_claim: F,
        inc_claim: F,
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

/// Stage 4 Prover using Gruen optimization (matches Jolt exactly)
///
/// Key differences from the original Stage4Prover:
/// 1. Uses GruenSplitEqPolynomial for eq polynomial factorization
/// 2. Computes quadratic coefficients [q(0), q_X2] instead of direct evaluations
/// 3. Uses gruenPolyDeg3 to convert to cubic polynomial
pub fn Stage4GruenProver(comptime F: type) type {
    const BlakeTranscript = @import("../../transcripts/mod.zig").Blake2bTranscript(F);

    return struct {
        const Self = @This();

        allocator: Allocator,

        /// Trace length (power of 2)
        T: usize,
        log_T: usize,

        /// Total number of rounds = LOG_K + log_T
        num_rounds: usize,

        /// gamma challenge (for batching ra = gamma*rs1_ra + gamma^2*rs2_ra)
        gamma: F,
        gamma_sq: F,

        /// r_cycle from Stage 3 (big-endian: r_cycle[0] = MSB)
        r_cycle: []const F,

        /// GruenSplitEqPolynomial for efficient eq computation during cycle binding
        gruen_eq: ?GruenSplitEqPolynomial(F),

        /// Stage 3 claims for input claim computation
        stage3_claims: ?Stage3Claims(F),

        // Dense polynomial evaluations indexed as: poly[k * T + j]
        val_poly: []F,
        rd_wa_poly: []F,
        /// Combined ra = gamma*rs1_ra + gamma^2*rs2_ra (matches Jolt's ra_coeff)
        ra_poly: []F,
        inc_poly: []F,

        // For final claims, we also need individual rs1_ra and rs2_ra
        rs1_ra_poly: []F,
        rs2_ra_poly: []F,

        /// Current effective sizes
        current_T: usize,
        current_K: usize,

        /// Batching coefficient
        batching_coeff: F,

        /// Alias for compatibility with existing code
        pub fn initWithClaims(
            allocator: Allocator,
            trace: *const ExecutionTrace,
            gamma: F,
            r_cycle: []const F,
            stage3_claims: ?Stage3Claims(F),
            batching_coeff: F,
        ) !Self {
            return init(allocator, trace, gamma, r_cycle, stage3_claims, batching_coeff);
        }

        pub fn init(
            allocator: Allocator,
            trace: *const ExecutionTrace,
            gamma: F,
            r_cycle: []const F,
            stage3_claims: ?Stage3Claims(F),
            batching_coeff: F,
        ) !Self {
            const trace_len = trace.steps.items.len;
            if (trace_len == 0) return error.EmptyTrace;

            const T = std.math.ceilPowerOfTwo(usize, trace_len) catch trace_len;
            const log_T = @ctz(T);

            if (r_cycle.len != log_T) {
                std.debug.print("[STAGE4_GRUEN] r_cycle.len = {}, expected log_T = {}\n", .{ r_cycle.len, log_T });
                return error.InvalidRCycleLength;
            }

            const num_rounds = LOG_K + log_T;
            const total_size = K * T;
            const gamma_sq = gamma.mul(gamma);

            // Allocate polynomial arrays
            const val_poly = try allocator.alloc(F, total_size);
            const rd_wa_poly = try allocator.alloc(F, total_size);
            const ra_poly = try allocator.alloc(F, total_size);
            const rs1_ra_poly = try allocator.alloc(F, total_size);
            const rs2_ra_poly = try allocator.alloc(F, total_size);
            const inc_poly = try allocator.alloc(F, T);

            @memset(val_poly, F.zero());
            @memset(rd_wa_poly, F.zero());
            @memset(ra_poly, F.zero());
            @memset(rs1_ra_poly, F.zero());
            @memset(rs2_ra_poly, F.zero());
            @memset(inc_poly, F.zero());

            // Track register values across cycles
            var register_values: [32]u64 = [_]u64{0} ** 32;

            // Build polynomial evaluations from trace (matching Jolt's RegistersCycleMajorEntry)
            for (trace.steps.items, 0..) |step, cycle| {
                // Set val(k, j) for all registers - value BEFORE this cycle
                for (0..32) |k| {
                    val_poly[k * T + cycle] = F.fromU64(register_values[k]);
                }
                // Extend to full K registers
                for (32..K) |k| {
                    val_poly[k * T + cycle] = F.zero();
                }

                if (step.is_noop) continue;

                const instr = step.instruction;
                const rd: u5 = @truncate((instr >> 7) & 0x1f);
                const rs1: u5 = @truncate((instr >> 15) & 0x1f);
                const rs2: u5 = @truncate((instr >> 20) & 0x1f);
                const opcode = instr & 0x7f;

                // rs1_ra: gamma coefficient for rs1 reads
                const reads_rs1 = switch (opcode) {
                    0x13, 0x03, 0x67, 0x1b, 0x33, 0x3b, 0x23, 0x63 => true,
                    else => false,
                };
                if (reads_rs1 and rs1 < 32) {
                    // CRITICAL: Verify tracked value matches trace value
                    if (cycle < 5 and register_values[rs1] != step.rs1_value) {
                        std.debug.print("[VAL_MISMATCH] cycle={}, rs1={}: tracked={}, trace={}\n", .{
                            cycle, rs1, register_values[rs1], step.rs1_value,
                        });
                    }
                    rs1_ra_poly[@as(usize, rs1) * T + cycle] = F.one();
                    ra_poly[@as(usize, rs1) * T + cycle] = ra_poly[@as(usize, rs1) * T + cycle].add(gamma);
                }

                // rs2_ra: gamma^2 coefficient for rs2 reads
                const reads_rs2 = switch (opcode) {
                    0x33, 0x3b, 0x23, 0x63 => true,
                    else => false,
                };
                if (reads_rs2 and rs2 < 32) {
                    // CRITICAL: Verify tracked value matches trace value
                    if (cycle < 5 and register_values[rs2] != step.rs2_value) {
                        std.debug.print("[VAL_MISMATCH] cycle={}, rs2={}: tracked={}, trace={}\n", .{
                            cycle, rs2, register_values[rs2], step.rs2_value,
                        });
                    }
                    rs2_ra_poly[@as(usize, rs2) * T + cycle] = F.one();
                    ra_poly[@as(usize, rs2) * T + cycle] = ra_poly[@as(usize, rs2) * T + cycle].add(gamma_sq);
                }

                // rd_wa and inc
                const rd_used = switch (opcode) {
                    0x23, 0x63 => false,
                    else => true,
                };
                if (rd_used and rd != 0 and rd < 32) {
                    rd_wa_poly[@as(usize, rd) * T + cycle] = F.one();
                    const pre_value = register_values[rd];
                    const post_value = step.rd_value;
                    inc_poly[cycle] = F.fromU64(post_value).sub(F.fromU64(pre_value));
                    register_values[rd] = post_value;
                }
            }

            // Fill padding cycles with final register values
            if (trace_len < T) {
                for (trace_len..T) |cycle| {
                    for (0..32) |k| {
                        val_poly[k * T + cycle] = F.fromU64(register_values[k]);
                    }
                    for (32..K) |k| {
                        val_poly[k * T + cycle] = F.zero();
                    }
                }
            }

            // Convert r_cycle from LE (round order) to BE (MSB first) for GruenSplitEqPolynomial
            // Stage 3 challenges are in round order: r_cycle[0] = round 0 challenge (binds LSB)
            // GruenSplitEqPolynomial expects big-endian: w[0] = MSB
            const r_cycle_be = try allocator.alloc(F, r_cycle.len);
            for (0..r_cycle.len) |i| {
                r_cycle_be[i] = r_cycle[r_cycle.len - 1 - i];
            }

            // Debug: Print r_cycle_be values to compare with Jolt's params.r_cycle
            std.debug.print("\n[STAGE4_GRUEN_INIT] r_cycle_be (should match Jolt's params.r_cycle):\n", .{});
            for (0..r_cycle_be.len) |i| {
                std.debug.print("[STAGE4_GRUEN_INIT]   r_cycle_be[{}] = {any}\n", .{ i, r_cycle_be[i].toBytes() });
            }

            const gruen_eq_poly = try GruenSplitEqPolynomial(F).init(allocator, r_cycle_be);

            // Debug: Print E_out and E_in table details for comparison with Jolt
            {
                const m = r_cycle_be.len / 2;
                std.debug.print("\n[STAGE4_GRUEN_INIT] GruenSplitEqPolynomial structure:\n", .{});
                std.debug.print("[STAGE4_GRUEN_INIT]   n={}, m={}\n", .{ r_cycle_be.len, m });
                std.debug.print("[STAGE4_GRUEN_INIT]   w_out = r_cycle_be[0..{}] (indices 0..{})\n", .{ m, m - 1 });
                std.debug.print("[STAGE4_GRUEN_INIT]   w_in = r_cycle_be[{}..{}]\n", .{ m, r_cycle_be.len - 1 });
                std.debug.print("[STAGE4_GRUEN_INIT]   w_last = r_cycle_be[{}]\n", .{r_cycle_be.len - 1});

                const E_out = gruen_eq_poly.E_out_current();
                const E_in = gruen_eq_poly.E_in_current();

                std.debug.print("[STAGE4_GRUEN_INIT]   E_out.len={}, E_in.len={}\n", .{ E_out.len, E_in.len });

                // Print first 4 E_out entries (serialized as 32-byte arrays)
                std.debug.print("[STAGE4_GRUEN_INIT]   E_out[0..min(4, len)]:\n", .{});
                for (0..@min(4, E_out.len)) |i| {
                    std.debug.print("[STAGE4_GRUEN_INIT]     E_out[{}] = {any}\n", .{ i, E_out[i].toBytes() });
                }

                // Print first 4 E_in entries
                std.debug.print("[STAGE4_GRUEN_INIT]   E_in[0..min(4, len)]:\n", .{});
                for (0..@min(4, E_in.len)) |i| {
                    std.debug.print("[STAGE4_GRUEN_INIT]     E_in[{}] = {any}\n", .{ i, E_in[i].toBytes() });
                }

                // Print current_scalar and current_w
                std.debug.print("[STAGE4_GRUEN_INIT]   current_scalar = {any}\n", .{gruen_eq_poly.current_scalar.toBytes()});
                std.debug.print("[STAGE4_GRUEN_INIT]   current_w (w_last) = {any}\n", .{gruen_eq_poly.get_current_w().toBytes()});
            }
            allocator.free(r_cycle_be); // GruenSplitEqPolynomial makes its own copy

            // Copy r_cycle (keep original LE order for other uses)
            const r_cycle_copy = try allocator.alloc(F, r_cycle.len);
            @memcpy(r_cycle_copy, r_cycle);

            // Debug: Print first few polynomial values for comparison with Jolt
            std.debug.print("[STAGE4 INIT] T={}, K={}, gamma={any}\n", .{ T, K, gamma.toBytes()[0..8] });
            std.debug.print("[STAGE4 INIT] First 4 entries for register k=2 (j=0..3):\n", .{});
            for (0..4) |j| {
                const idx = 2 * T + j;
                std.debug.print("  j={}: val={any}, ra={any}, wa={any}\n", .{
                    j,
                    val_poly[idx].toBytes()[0..8],
                    ra_poly[idx].toBytes()[0..8],
                    rd_wa_poly[idx].toBytes()[0..8],
                });
            }
            std.debug.print("[STAGE4 INIT] inc_poly first 4: ", .{});
            for (0..4) |j| {
                std.debug.print("{any} ", .{inc_poly[j].toBytes()[0..8]});
            }
            std.debug.print("\n", .{});

            return Self{
                .allocator = allocator,
                .T = T,
                .log_T = log_T,
                .num_rounds = num_rounds,
                .gamma = gamma,
                .gamma_sq = gamma_sq,
                .r_cycle = r_cycle_copy,
                .gruen_eq = gruen_eq_poly,
                .stage3_claims = stage3_claims,
                .val_poly = val_poly,
                .rd_wa_poly = rd_wa_poly,
                .ra_poly = ra_poly,
                .rs1_ra_poly = rs1_ra_poly,
                .rs2_ra_poly = rs2_ra_poly,
                .inc_poly = inc_poly,
                .current_T = T,
                .current_K = K,
                .batching_coeff = batching_coeff,
            };
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.val_poly);
            self.allocator.free(self.rd_wa_poly);
            self.allocator.free(self.ra_poly);
            self.allocator.free(self.rs1_ra_poly);
            self.allocator.free(self.rs2_ra_poly);
            self.allocator.free(self.inc_poly);
            self.allocator.free(@constCast(self.r_cycle));
            if (self.gruen_eq) |*g| {
                g.deinit();
            }
        }

        /// Compute input claim from Stage 3 claims or polynomial sum
        fn computeInputClaim(self: *Self) F {
            if (self.stage3_claims) |claims| {
                // input_claim = rd_wv + gamma * (rs1_v + gamma * rs2_v)
                return claims.rd_write_value.add(
                    self.gamma.mul(claims.rs1_value.add(self.gamma.mul(claims.rs2_value))),
                );
            }

            // Compute from polynomials using GruenSplitEqPolynomial
            var claim = F.zero();
            const gruen = &self.gruen_eq.?;
            const E_out = gruen.E_out_current();
            const E_in = gruen.E_in_current();
            const num_x_in_bits = if (E_in.len > 0) @ctz(E_in.len) else 0;

            for (0..K) |k| {
                for (0..self.T) |j| {
                    const idx = k * self.T + j;
                    // Factor j into (x_out, x_in, x_last)
                    const x_last = j & 1;
                    const j_upper = j >> 1;
                    const x_in = j_upper & (((@as(usize, 1) << num_x_in_bits) - 1));
                    const x_out = j_upper >> num_x_in_bits;

                    // eq_j = E_out[x_out] * E_in[x_in] * eq_linear(x_last)
                    const E_out_val = if (x_out < E_out.len) E_out[x_out] else F.one();
                    const E_in_val = if (x_in < E_in.len) E_in[x_in] else F.one();
                    const w_last = gruen.get_current_w();
                    const eq_linear = if (x_last == 0)
                        gruen.current_scalar.sub(gruen.current_scalar.mul(w_last))
                    else
                        gruen.current_scalar.mul(w_last);
                    const eq_j = E_out_val.mul(E_in_val).mul(eq_linear);

                    // Combined contribution
                    const val = self.val_poly[idx];
                    const ra = self.ra_poly[idx];
                    const wa = self.rd_wa_poly[idx];
                    const inc = self.inc_poly[j];

                    const combined = ra.mul(val).add(wa.mul(val.add(inc)));
                    claim = claim.add(eq_j.mul(combined));
                }
            }
            return claim;
        }

        /// Run the Stage 4 sumcheck using Gruen optimization
        pub fn prove(self: *Self, transcript: *BlakeTranscript) !Stage4Result(F) {
            var round_polys = try self.allocator.alloc(RoundPoly(F), self.num_rounds);
            var challenges = try self.allocator.alloc(F, self.num_rounds);

            const input_claim = self.computeInputClaim();
            var current_claim = input_claim.mul(self.batching_coeff);

            std.debug.print("[STAGE4_GRUEN] Starting with {} rounds, T={}, K={}\n", .{ self.num_rounds, self.T, K });
            std.debug.print("[STAGE4_GRUEN] Input claim = {any}\n", .{input_claim.toBytes()[0..8]});
            std.debug.print("[STAGE4_GRUEN] Batched claim = {any}\n", .{current_claim.toBytes()[0..8]});

            for (0..self.num_rounds) |round| {
                const round_poly = self.computeRoundPolynomialGruen(round, current_claim);

                // Debug first few rounds
                if (round < 2) {
                    std.debug.print("[STAGE4_GRUEN] Round {}: coeffs = [{any}, {any}, ...]\n", .{
                        round,
                        round_poly.coeffs[0].toBytes()[0..8],
                        round_poly.coeffs[1].toBytes()[0..8],
                    });
                }

                // Append to transcript (batched coefficients)
                transcript.appendScalar(round_poly.coeffs[0].mul(self.batching_coeff));
                transcript.appendScalar(round_poly.coeffs[2].mul(self.batching_coeff));
                transcript.appendScalar(round_poly.coeffs[3].mul(self.batching_coeff));

                const challenge = transcript.challengeScalar();
                challenges[round] = challenge;

                current_claim = round_poly.evaluateAt(challenge).mul(self.batching_coeff);

                self.bindPolynomials(round, challenge);

                // Store UNBATCHED polynomial (batching is only for transcript/challenge)
                round_polys[round] = round_poly;
            }

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

        /// Compute round polynomial using Gruen optimization
        fn computeRoundPolynomialGruen(self: *Self, round: usize, current_claim: F) RoundPoly(F) {
            if (round < self.log_T) {
                // Phase 1: Binding cycle variables using Gruen
                return self.phase1ComputeMessage(current_claim);
            } else {
                // Phase 2/3: Binding address variables (dense computation)
                return self.phase23ComputeMessage(round, current_claim);
            }
        }

        /// Phase 1: Compute round polynomial for cycle variable binding using Gruen
        ///
        /// This computes [q(0), q_X2_coeff] where q(X) is the "body" polynomial,
        /// then uses gruenPolyDeg3 to get the full cubic s(X) = eq_linear(X) * q(X).
        fn phase1ComputeMessage(self: *Self, previous_claim: F) RoundPoly(F) {
            const gruen = &self.gruen_eq.?;
            const E_in = gruen.E_in_current();
            const E_out = gruen.E_out_current();
            const E_in_len = E_in.len;
            const num_x_in_bits: usize = if (E_in_len > 1) @ctz(E_in_len) else 0;
            const x_bitmask = if (num_x_in_bits < 64) ((@as(usize, 1) << @intCast(num_x_in_bits)) - 1) else @as(usize, 0);

            // Debug: Print Gruen state for first few calls
            if (self.current_T >= self.T / 2) {
                std.debug.print("[GRUEN_DEBUG] Phase 1: current_T={}, T={}, current_scalar={any}\n", .{
                    self.current_T,
                    self.T,
                    gruen.current_scalar.toBytes()[0..8],
                });
                std.debug.print("[GRUEN_DEBUG]   current_index={}, w[{}]={any}\n", .{
                    gruen.current_index,
                    gruen.current_index - 1,
                    gruen.get_current_w().toBytes()[0..8],
                });
                std.debug.print("[GRUEN_DEBUG]   E_in.len={}, E_out.len={}, num_x_in_bits={}\n", .{
                    E_in_len,
                    E_out.len,
                    num_x_in_bits,
                });
                if (E_out.len > 0) {
                    std.debug.print("[GRUEN_DEBUG]   E_out[0]={any}\n", .{E_out[0].toBytes()[0..8]});
                }
                if (E_in_len > 0) {
                    std.debug.print("[GRUEN_DEBUG]   E_in[0]={any}\n", .{E_in[0].toBytes()[0..8]});
                }
            }

            // Accumulate [q(0), q_X2_coeff]
            var q_0 = F.zero();
            var q_X2 = F.zero();
            var nonzero_count: usize = 0;
            const is_round_0 = self.current_T == self.T;

            const half_T = self.current_T / 2;

            // Process pairs (j_even, j_odd) where they differ only in LSB
            for (0..half_T) |i| {
                const j_prime = 2 * i; // j_even
                const j_odd = j_prime + 1;

                // Compute x_out and x_in from j_prime
                const x_in = if (num_x_in_bits > 0) (i & x_bitmask) else 0;
                const x_out = if (num_x_in_bits < 64) (i >> @as(u6, @intCast(num_x_in_bits))) else 0;

                const E_in_eval = if (x_in < E_in_len) E_in[x_in] else F.one();
                const E_out_eval = if (x_out < E_out.len) E_out[x_out] else F.one();
                const E_combined = E_out_eval.mul(E_in_eval);

                // Get inc evaluations: [inc_0, inc_slope]
                // Jolt uses: inc_evals = [inc(j_even), inc(j_odd) - inc(j_even)]
                const inc_0 = self.inc_poly[j_prime];
                const inc_1 = self.inc_poly[j_odd];
                const inc_slope = inc_1.sub(inc_0);

                // Sum over all registers
                for (0..self.current_K) |k| {
                    const idx_even = k * self.T + j_prime;
                    const idx_odd = k * self.T + j_odd;

                    // Get polynomial values for this (k, j) pair
                    const ra_even = self.ra_poly[idx_even];
                    const ra_odd = self.ra_poly[idx_odd];
                    const wa_even = self.rd_wa_poly[idx_even];
                    const wa_odd = self.rd_wa_poly[idx_odd];
                    const val_even = self.val_poly[idx_even];
                    const val_odd = self.val_poly[idx_odd];

                    // Compute slopes (coefficient of X in linear interpolation)
                    const ra_slope = ra_odd.sub(ra_even);
                    const wa_slope = wa_odd.sub(wa_even);
                    const val_slope = val_odd.sub(val_even);

                    // C(X) = ra(X)*val(X) + wa(X)*(val(X)+inc(X))
                    // C(0) = ra(0)*val(0) + wa(0)*(val(0)+inc(0))
                    const c_0 = ra_even.mul(val_even).add(wa_even.mul(val_even.add(inc_0)));

                    // C_X2_coeff = ra_slope*val_slope + wa_slope*(val_slope+inc_slope)
                    // This matches Jolt's compute_evals which uses inc_evals[1] for the slope term.
                    const c_X2 = ra_slope.mul(val_slope).add(wa_slope.mul(val_slope.add(inc_slope)));

                    // Debug: print first few nonzero contributions in Round 0
                    const has_nonzero = !c_0.eql(F.zero()) or !c_X2.eql(F.zero());
                    if (is_round_0 and has_nonzero and nonzero_count < 5) {
                        nonzero_count += 1;
                        std.debug.print("[STAGE4_CONTRIB #{} i={}, k={}] j_pair=({},{}): idx_even={}, idx_odd={}\n", .{
                            nonzero_count, i, k, j_prime, j_odd, idx_even, idx_odd,
                        });
                        std.debug.print("[STAGE4_CONTRIB]   EVEN: ra={any}, wa={any}, val={any}\n", .{
                            ra_even.toBytes()[0..8],
                            wa_even.toBytes()[0..8],
                            val_even.toBytes()[0..8],
                        });
                        std.debug.print("[STAGE4_CONTRIB]   ODD:  ra={any}, wa={any}, val={any}\n", .{
                            ra_odd.toBytes()[0..8],
                            wa_odd.toBytes()[0..8],
                            val_odd.toBytes()[0..8],
                        });
                        std.debug.print("[STAGE4_CONTRIB]   SLOPE: ra={any}, wa={any}, val={any}\n", .{
                            ra_slope.toBytes()[0..8],
                            wa_slope.toBytes()[0..8],
                            val_slope.toBytes()[0..8],
                        });
                        std.debug.print("[STAGE4_CONTRIB]   inc_0={any}, inc_slope={any}\n", .{
                            inc_0.toBytes()[0..8],
                            inc_slope.toBytes()[0..8],
                        });
                        std.debug.print("[STAGE4_CONTRIB]   c_0={any}, c_X2={any}, E_combined={any}\n", .{
                            c_0.toBytes()[0..8],
                            c_X2.toBytes()[0..8],
                            E_combined.toBytes()[0..8],
                        });
                        // Also print x_out, x_in for this contribution
                        std.debug.print("[STAGE4_CONTRIB]   x_out={}, x_in={}, E_out={any}, E_in={any}\n", .{
                            x_out,
                            x_in,
                            E_out_eval.toBytes()[0..8],
                            E_in_eval.toBytes()[0..8],
                        });
                    }

                    // Additional debug: print ALL contributions for i=0 (first pair) in round 0
                    if (is_round_0 and i == 0 and (!ra_even.eql(F.zero()) or !wa_even.eql(F.zero()) or !ra_odd.eql(F.zero()) or !wa_odd.eql(F.zero()))) {
                        std.debug.print("[STAGE4_PAIR0] k={}: ra_even={any}, wa_even={any}, val_even={any}\n", .{
                            k,
                            ra_even.toBytes()[0..8],
                            wa_even.toBytes()[0..8],
                            val_even.toBytes()[0..8],
                        });
                        std.debug.print("[STAGE4_PAIR0]   ra_odd={any}, wa_odd={any}, val_odd={any}\n", .{
                            ra_odd.toBytes()[0..8],
                            wa_odd.toBytes()[0..8],
                            val_odd.toBytes()[0..8],
                        });
                        std.debug.print("[STAGE4_PAIR0]   c_0={any}, c_X2={any}\n", .{
                            c_0.toBytes()[0..8],
                            c_X2.toBytes()[0..8],
                        });
                    }

                    // Accumulate with E_out * E_in factor
                    const contrib_0 = E_combined.mul(c_0);
                    const contrib_X2 = E_combined.mul(c_X2);
                    q_0 = q_0.add(contrib_0);
                    q_X2 = q_X2.add(contrib_X2);

                    // Debug: print first 3 accumulations in Round 0
                    if (is_round_0 and (!contrib_0.eql(F.zero()) or !contrib_X2.eql(F.zero())) and nonzero_count <= 3) {
                        std.debug.print("[ZOLT ACCUM] contrib #{}: E_combined*c_0={any}\n", .{
                            nonzero_count,
                            contrib_0.toBytes()[0..16],
                        });
                        std.debug.print("[ZOLT ACCUM] q_0 (after)={any}\n", .{q_0.toBytes()[0..16]});
                        std.debug.print("[ZOLT ACCUM] contrib #{}: E_combined*c_X2={any}\n", .{
                            nonzero_count,
                            contrib_X2.toBytes()[0..16],
                        });
                        std.debug.print("[ZOLT ACCUM] q_X2 (after)={any}\n", .{q_X2.toBytes()[0..16]});
                    }
                }
            }

            // Debug: Print q_0 and q_X2 for first round (full 32 bytes)
            if (self.current_T >= self.T / 2) {
                std.debug.print("[ZOLT PHASE1 q_0] = {any}\n", .{q_0.toBytes()});
                std.debug.print("[ZOLT PHASE1 q_X2] = {any}\n", .{q_X2.toBytes()});
                std.debug.print("[ZOLT PHASE1 previous_claim] = {any}\n", .{previous_claim.toBytes()});
                std.debug.print("[ZOLT PHASE1] Round completed, current_T={}, half_T={}\n", .{ self.current_T, half_T });
            }

            // Use gruenPolyDeg3 to convert [q(0), q_X2] to cubic coefficients
            if (self.current_T >= self.T / 2) {
                std.debug.print("[ZOLT PHASE1] Before gruenPolyDeg3: current_index={}, w.len={}\n", .{
                    gruen.current_index,
                    gruen.w.len,
                });
            }
            const coeffs = gruen.gruenPolyDeg3(q_0, q_X2, previous_claim);

            // Debug: Print final coefficients for first round
            if (self.current_T >= self.T / 2) {
                std.debug.print("[ZOLT PHASE1 COEFFS] c0={any}\n", .{coeffs[0].toBytes()});
                std.debug.print("[ZOLT PHASE1 COEFFS] c1={any}\n", .{coeffs[1].toBytes()});
                std.debug.print("[ZOLT PHASE1 COEFFS] c2={any}\n", .{coeffs[2].toBytes()});
                std.debug.print("[ZOLT PHASE1 COEFFS] c3={any}\n", .{coeffs[3].toBytes()});
            }

            return RoundPoly(F){ .coeffs = coeffs };
        }

        /// Phase 2/3: Compute round polynomial for address variable binding (dense)
        fn phase23ComputeMessage(self: *Self, round: usize, previous_claim: F) RoundPoly(F) {
            _ = round;
            _ = previous_claim; // Used for verification, not computation

            // For address binding, use direct evaluation at 0, 1, 2, 3
            const half_K = self.current_K / 2;
            var evals: [4]F = .{ F.zero(), F.zero(), F.zero(), F.zero() };

            // We need the merged eq polynomial after all cycle variables are bound
            // At this point, gruen_eq.current_scalar contains eq(w_bound, r_bound)
            // and we need to compute eq over remaining address variables
            const gruen = &self.gruen_eq.?;
            const eq_scalar = gruen.current_scalar;

            for (0..half_K) |i| {
                const k_even = 2 * i;
                const k_odd = k_even + 1;

                for (0..self.current_T) |j| {
                    const idx_even = k_even * self.T + j;
                    const idx_odd = k_odd * self.T + j;
                    const inc_j = self.inc_poly[j];

                    // Get polynomial values
                    const ra_even = self.ra_poly[idx_even];
                    const ra_odd = self.ra_poly[idx_odd];
                    const wa_even = self.rd_wa_poly[idx_even];
                    const wa_odd = self.rd_wa_poly[idx_odd];
                    const val_even = self.val_poly[idx_even];
                    const val_odd = self.val_poly[idx_odd];

                    // Evaluate at X = 0, 1, 2, 3
                    inline for (0..4) |t| {
                        const t_field = F.fromU64(t);
                        const one_minus_t = F.one().sub(t_field);

                        const ra_t = one_minus_t.mul(ra_even).add(t_field.mul(ra_odd));
                        const wa_t = one_minus_t.mul(wa_even).add(t_field.mul(wa_odd));
                        const val_t = one_minus_t.mul(val_even).add(t_field.mul(val_odd));

                        const combined_t = ra_t.mul(val_t).add(wa_t.mul(val_t.add(inc_j)));
                        evals[t] = evals[t].add(eq_scalar.mul(combined_t));
                    }
                }
            }

            // Convert evaluations to coefficients
            return coeffsFromEvals(evals);
        }

        fn coeffsFromEvals(evals: [4]F) RoundPoly(F) {
            const c0 = evals[0];
            const six = F.fromU64(6);
            const six_inv = six.inverse() orelse F.one();
            const two = F.fromU64(2);
            const two_inv = two.inverse() orelse F.one();

            const c3 = evals[0].neg()
                .add(evals[1].mul(F.fromU64(3)))
                .sub(evals[2].mul(F.fromU64(3)))
                .add(evals[3])
                .mul(six_inv);

            const c2 = evals[0].mul(two)
                .sub(evals[1].mul(F.fromU64(5)))
                .add(evals[2].mul(F.fromU64(4)))
                .sub(evals[3])
                .mul(two_inv);

            const c1 = evals[1].sub(evals[0]).sub(c2).sub(c3);

            return RoundPoly(F){ .coeffs = .{ c0, c1, c2, c3 } };
        }

        /// Bind polynomials after receiving a challenge
        fn bindPolynomials(self: *Self, round: usize, challenge: F) void {
            const is_cycle_round = round < self.log_T;
            const one_minus_c = F.one().sub(challenge);

            if (is_cycle_round) {
                const half_T = self.current_T / 2;

                // Bind cycle variable using low-to-high
                for (0..self.current_K) |k| {
                    for (0..half_T) |i| {
                        const j_lo = 2 * i;
                        const j_hi = j_lo + 1;
                        const idx_lo = k * self.T + j_lo;
                        const idx_hi = k * self.T + j_hi;
                        const new_idx = k * self.T + i;

                        self.val_poly[new_idx] = self.val_poly[idx_lo].mul(one_minus_c).add(self.val_poly[idx_hi].mul(challenge));
                        self.rd_wa_poly[new_idx] = self.rd_wa_poly[idx_lo].mul(one_minus_c).add(self.rd_wa_poly[idx_hi].mul(challenge));
                        self.ra_poly[new_idx] = self.ra_poly[idx_lo].mul(one_minus_c).add(self.ra_poly[idx_hi].mul(challenge));
                        self.rs1_ra_poly[new_idx] = self.rs1_ra_poly[idx_lo].mul(one_minus_c).add(self.rs1_ra_poly[idx_hi].mul(challenge));
                        self.rs2_ra_poly[new_idx] = self.rs2_ra_poly[idx_lo].mul(one_minus_c).add(self.rs2_ra_poly[idx_hi].mul(challenge));
                    }
                }

                // Bind inc_poly
                for (0..half_T) |i| {
                    const j_lo = 2 * i;
                    const j_hi = j_lo + 1;
                    self.inc_poly[i] = self.inc_poly[j_lo].mul(one_minus_c).add(self.inc_poly[j_hi].mul(challenge));
                }

                self.current_T = half_T;

                // Bind the Gruen eq polynomial
                self.gruen_eq.?.bind(challenge);
            } else {
                const half_K = self.current_K / 2;

                // Bind address variable
                for (0..half_K) |i| {
                    const k_lo = 2 * i;
                    const k_hi = k_lo + 1;

                    for (0..self.current_T) |j| {
                        const idx_lo = k_lo * self.T + j;
                        const idx_hi = k_hi * self.T + j;
                        const new_idx = i * self.T + j;

                        self.val_poly[new_idx] = self.val_poly[idx_lo].mul(one_minus_c).add(self.val_poly[idx_hi].mul(challenge));
                        self.rd_wa_poly[new_idx] = self.rd_wa_poly[idx_lo].mul(one_minus_c).add(self.rd_wa_poly[idx_hi].mul(challenge));
                        self.ra_poly[new_idx] = self.ra_poly[idx_lo].mul(one_minus_c).add(self.ra_poly[idx_hi].mul(challenge));
                        self.rs1_ra_poly[new_idx] = self.rs1_ra_poly[idx_lo].mul(one_minus_c).add(self.rs1_ra_poly[idx_hi].mul(challenge));
                        self.rs2_ra_poly[new_idx] = self.rs2_ra_poly[idx_lo].mul(one_minus_c).add(self.rs2_ra_poly[idx_hi].mul(challenge));
                    }
                }

                self.current_K = half_K;
            }
        }

        /// Compute round polynomial evaluations at 0, 1, 2, 3
        /// Compatible with the existing proof_converter interface
        pub fn computeRoundEvals(self: *Self, round: usize, current_claim: F) [4]F {
            const round_poly = self.computeRoundPolynomialGruen(round, current_claim);
            // Convert coefficients back to evaluations
            // p(0) = c0
            // p(1) = c0 + c1 + c2 + c3
            // p(2) = c0 + 2*c1 + 4*c2 + 8*c3
            // p(3) = c0 + 3*c1 + 9*c2 + 27*c3
            const c = round_poly.coeffs;
            const p0 = c[0];
            const p1 = c[0].add(c[1]).add(c[2]).add(c[3]);
            const p2 = c[0].add(c[1].mul(F.fromU64(2))).add(c[2].mul(F.fromU64(4))).add(c[3].mul(F.fromU64(8)));
            const p3 = c[0].add(c[1].mul(F.fromU64(3))).add(c[2].mul(F.fromU64(9))).add(c[3].mul(F.fromU64(27)));
            return .{ p0, p1, p2, p3 };
        }

        /// Bind challenge after getting round evaluations (compatible with existing interface)
        pub fn bindChallenge(self: *Self, round: usize, challenge: F) void {
            self.bindPolynomials(round, challenge);
        }

        pub fn getFinalClaims(self: *const Self) struct {
            val_claim: F,
            rs1_ra_claim: F,
            rs2_ra_claim: F,
            rd_wa_claim: F,
            inc_claim: F,
        } {
            return .{
                .val_claim = if (self.val_poly.len > 0) self.val_poly[0] else F.zero(),
                .rs1_ra_claim = if (self.rs1_ra_poly.len > 0) self.rs1_ra_poly[0] else F.zero(),
                .rs2_ra_claim = if (self.rs2_ra_poly.len > 0) self.rs2_ra_poly[0] else F.zero(),
                .rd_wa_claim = if (self.rd_wa_poly.len > 0) self.rd_wa_poly[0] else F.zero(),
                .inc_claim = if (self.inc_poly.len > 0) self.inc_poly[0] else F.zero(),
            };
        }
    };
}

test "stage4 gruen prover basic" {
    const BN254Scalar = @import("../../field/mod.zig").BN254Scalar;
    const allocator = std.testing.allocator;
    const Transcript = @import("../../transcripts/mod.zig").Blake2bTranscript(BN254Scalar);

    // Create minimal trace
    var trace = ExecutionTrace.init(allocator);
    defer trace.deinit();

    // Add a simple instruction
    try trace.steps.append(.{
        .pc = 0,
        .instruction = 0x00500093, // addi x1, x0, 5
        .rs1_value = 0,
        .rs2_value = 0,
        .rd_value = 5,
        .is_noop = false,
    });

    // Pad to power of 2
    try trace.steps.append(.{
        .pc = 4,
        .instruction = 0,
        .rs1_value = 0,
        .rs2_value = 0,
        .rd_value = 0,
        .is_noop = true,
    });

    const gamma = BN254Scalar.fromU64(123);
    const r_cycle = [_]BN254Scalar{BN254Scalar.fromU64(456)}; // log_T = 1

    var prover = try Stage4GruenProver(BN254Scalar).init(
        allocator,
        &trace,
        gamma,
        &r_cycle,
        null,
        BN254Scalar.one(),
    );
    defer prover.deinit();

    var transcript = Transcript.init();
    var result = try prover.prove(&transcript);
    defer result.deinit();

    // Basic sanity check: should have correct number of round polys
    try std.testing.expectEqual(@as(usize, 1 + LOG_K), result.round_polys.len);
}
