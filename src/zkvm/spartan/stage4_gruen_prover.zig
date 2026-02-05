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

        /// Phase configuration for Gruen binding (from ReadWriteConfig)
        /// Phase 1: first phase1_num_rounds cycle vars
        /// Phase 2: next phase2_num_rounds address vars
        /// Phase 3: remaining (log_T - phase1_num_rounds) cycle vars
        phase1_num_rounds: usize,
        phase2_num_rounds: usize,

        /// Merged eq polynomial for Phase 3 (after Gruen eq is fully bound)
        merged_eq: ?[]F,

        /// Trace reference for debugging
        trace_ref: ?*const ExecutionTrace,

        /// Alias for compatibility with existing code (uses default phase config)
        pub fn initWithClaims(
            allocator: Allocator,
            trace: *const ExecutionTrace,
            gamma: F,
            r_cycle: []const F,
            stage3_claims: ?Stage3Claims(F),
            batching_coeff: F,
        ) !Self {
            // CRITICAL: Must match Jolt's phase config exactly for verification compatibility
            // Jolt uses: phase1 = T.log_2() (ALL cycle vars), phase2 = K.log_2() (ALL address vars)
            // This means Phase 3 has 0 rounds - everything is bound in Phase 1 and 2.
            // See jolt-core/src/zkvm/registers/read_write_checking.rs:200-207
            const T = std.math.ceilPowerOfTwo(usize, trace.steps.items.len) catch trace.steps.items.len;
            const log_T = @ctz(T);
            const phase1_num_rounds = log_T; // ALL cycle vars in Phase 1 (matches Jolt)
            const phase2_num_rounds = LOG_K; // ALL address vars in Phase 2 (matches Jolt)
            return initWithPhaseConfig(allocator, trace, gamma, r_cycle, stage3_claims, batching_coeff, phase1_num_rounds, phase2_num_rounds);
        }

        /// Initialize with explicit phase configuration (from ReadWriteConfig)
        pub fn initWithPhaseConfig(
            allocator: Allocator,
            trace: *const ExecutionTrace,
            gamma: F,
            r_cycle: []const F,
            stage3_claims: ?Stage3Claims(F),
            batching_coeff: F,
            phase1_num_rounds: usize,
            phase2_num_rounds: usize,
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

                // Skip pure NoOp padding cycles.
                // Termination store steps with is_noop=false (LUI, ADDI, SB) are processed
                // normally below using their real register indices (x30, x31).
                // The noop step (is_noop=true, is_termination_store=true) is also skipped.
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
                    rs1_ra_poly[@as(usize, rs1) * T + cycle] = F.one();
                    ra_poly[@as(usize, rs1) * T + cycle] = ra_poly[@as(usize, rs1) * T + cycle].add(gamma);
                }

                // rs2_ra: gamma^2 coefficient for rs2 reads
                const reads_rs2 = switch (opcode) {
                    0x33, 0x3b, 0x23, 0x63 => true,
                    else => false,
                };
                if (reads_rs2 and rs2 < 32) {
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
            // NOTE: r_cycle_be values are MontU128Challenge-style [0, 0, low, high] limbs
            std.debug.print("\n[STAGE4_GRUEN_INIT] r_cycle_be (should match Jolt's params.r_cycle):\n", .{});
            for (0..r_cycle_be.len) |i| {
                const c = r_cycle_be[i];
                // Print in Jolt Challenge format: [16 zeros, low_LE, high_LE]
                var jolt_format: [32]u8 = [_]u8{0} ** 32;
                std.mem.writeInt(u64, jolt_format[16..24], c.limbs[2], .little);
                std.mem.writeInt(u64, jolt_format[24..32], c.limbs[3], .little);
                std.debug.print("[STAGE4_GRUEN_INIT]   r_cycle_be[{}] = {{ ", .{i});
                for (jolt_format) |b| std.debug.print("{x:0>2} ", .{b});
                std.debug.print("}}\n", .{});
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

            std.debug.print("[STAGE4 INIT] Phase config: phase1={}, phase2={}, phase3_cycle={}\n", .{
                phase1_num_rounds,
                phase2_num_rounds,
                log_T - phase1_num_rounds,
            });

            // CRITICAL DIAGNOSTIC: Compute polynomial sum using brute-force eq(r_cycle, j)
            // This does NOT use Gruen decomposition - direct eq table computation
            {
                const eq_tbl = allocator.alloc(F, T) catch unreachable;
                defer allocator.free(eq_tbl);
                eq_tbl[0] = F.one();
                var eq_sz: usize = 1;
                // CRITICAL: For LowToHigh binding, r_cycle[0] must bind bit 0 (LSB) of j.
                // The backward-loop construction assigns the FIRST processed challenge to MSB.
                // So we must process challenges in REVERSE order (r_cycle[n-1] first)
                // to get r_cycle[0] → LSB.
                var bit_i: usize = r_cycle.len;
                while (bit_i > 0) {
                    bit_i -= 1;
                    const ri = r_cycle[bit_i];
                    const one_m_r = F.one().sub(ri);
                    var ii: usize = eq_sz;
                    while (ii > 0) {
                        ii -= 1;
                        eq_tbl[2 * ii + 1] = eq_tbl[ii].mul(ri);
                        eq_tbl[2 * ii] = eq_tbl[ii].mul(one_m_r);
                    }
                    eq_sz *= 2;
                }

                // Compute: Σ_j Σ_k eq(r_cycle, j) * [ra(k,j)*val(k,j) + wa(k,j)*(val(k,j)+inc(j))]
                var brute_sum = F.zero();
                var rd_write_sum = F.zero();
                var rs1_read_sum = F.zero();
                var rs2_read_sum = F.zero();
                for (0..K) |kk| {
                    for (0..T) |jj| {
                        const pidx = kk * T + jj;
                        const v = val_poly[pidx];
                        const ra_v = ra_poly[pidx];
                        const wa_v = rd_wa_poly[pidx];
                        const inc_v = inc_poly[jj];
                        const body = ra_v.mul(v).add(wa_v.mul(v.add(inc_v)));
                        brute_sum = brute_sum.add(eq_tbl[jj].mul(body));

                        // Component breakdown
                        rd_write_sum = rd_write_sum.add(eq_tbl[jj].mul(wa_v.mul(v.add(inc_v))));
                        rs1_read_sum = rs1_read_sum.add(eq_tbl[jj].mul(rs1_ra_poly[pidx].mul(v)));
                        rs2_read_sum = rs2_read_sum.add(eq_tbl[jj].mul(rs2_ra_poly[pidx].mul(v)));
                    }
                }

                // Compute Stage 3 claim
                var s3_claim = F.zero();
                if (stage3_claims) |claims| {
                    s3_claim = claims.rd_write_value.add(
                        gamma.mul(claims.rs1_value.add(gamma.mul(claims.rs2_value))),
                    );
                }

                // Also compute sum using raw trace values (independent of Stage 4 polynomial construction)
                var direct_rd_sum = F.zero();
                var direct_rs1_sum = F.zero();
                var direct_rs2_sum = F.zero();
                for (0..T) |jj| {
                    if (jj < trace.steps.items.len) {
                        const step = trace.steps.items[jj];
                        if (!step.is_noop or step.is_termination_store) {
                            const op = step.instruction & 0x7f;
                            const rd_reg: u5 = @truncate((step.instruction >> 7) & 0x1f);
                            const writes_rd = switch (op) {
                                0x23, 0x63 => false,
                                else => true,
                            };
                            if (writes_rd and rd_reg != 0) {
                                direct_rd_sum = direct_rd_sum.add(eq_tbl[jj].mul(F.fromU64(step.rd_value)));
                            }
                            const reads_rs1_dir = switch (op) {
                                0x13, 0x03, 0x67, 0x1b, 0x33, 0x3b, 0x23, 0x63 => true,
                                else => false,
                            };
                            if (reads_rs1_dir) {
                                direct_rs1_sum = direct_rs1_sum.add(eq_tbl[jj].mul(F.fromU64(step.rs1_value)));
                            }
                            const reads_rs2_dir = switch (op) {
                                0x33, 0x3b, 0x23, 0x63 => true,
                                else => false,
                            };
                            if (reads_rs2_dir) {
                                direct_rs2_sum = direct_rs2_sum.add(eq_tbl[jj].mul(F.fromU64(step.rs2_value)));
                            }
                        }
                    }
                }

                std.debug.print("\n[STAGE4 INIT BRUTE FORCE]\n", .{});
                std.debug.print("[STAGE4 INIT]   brute_sum     = {any}\n", .{brute_sum.toBytes()});
                std.debug.print("[STAGE4 INIT]   stage3_claim  = {any}\n", .{s3_claim.toBytes()});
                std.debug.print("[STAGE4 INIT]   direct_rd_sum = {any}\n", .{direct_rd_sum.toBytes()});
                std.debug.print("[STAGE4 INIT]   MATCH? {}\n", .{brute_sum.eql(s3_claim)});
                std.debug.print("[STAGE4 INIT]   direct_rd==s3.rd_wv? {}\n", .{direct_rd_sum.eql(if (stage3_claims) |c| c.rd_write_value else F.zero())});
                std.debug.print("[STAGE4 INIT]   direct_rd==stage4_rd? {}\n", .{direct_rd_sum.eql(rd_write_sum)});
                std.debug.print("[STAGE4 INIT]   direct_rs1==s3.rs1_v? {}\n", .{direct_rs1_sum.eql(if (stage3_claims) |c| c.rs1_value else F.zero())});
                std.debug.print("[STAGE4 INIT]   direct_rs1==stage4_rs1? {}\n", .{direct_rs1_sum.eql(rs1_read_sum)});
                std.debug.print("[STAGE4 INIT]   direct_rs2==s3.rs2_v? {}\n", .{direct_rs2_sum.eql(if (stage3_claims) |c| c.rs2_value else F.zero())});
                std.debug.print("[STAGE4 INIT]   direct_rs2==stage4_rs2? {}\n", .{direct_rs2_sum.eql(rs2_read_sum)});
                std.debug.print("[STAGE4 INIT]   direct_rs1_sum = {any}\n", .{direct_rs1_sum.toBytes()});
                std.debug.print("[STAGE4 INIT]   direct_rs2_sum = {any}\n", .{direct_rs2_sum.toBytes()});
                // Print first few non-zero rs1 direct contributions
                {
                    var rs1_ccount: usize = 0;
                    var rs1_nz_count: usize = 0;
                    for (0..T) |jj| {
                        if (jj < trace.steps.items.len and (!trace.steps.items[jj].is_noop or trace.steps.items[jj].is_termination_store)) {
                            const step = trace.steps.items[jj];
                            const op_c = step.instruction & 0x7f;
                            const reads_rs1_c = switch (op_c) {
                                0x13, 0x03, 0x67, 0x1b, 0x33, 0x3b, 0x23, 0x63 => true,
                                else => false,
                            };
                            if (reads_rs1_c) {
                                const rs1_f = F.fromU64(step.rs1_value);
                                if (!rs1_f.eql(F.zero())) {
                                    rs1_nz_count += 1;
                                    if (rs1_ccount < 8) {
                                        const contrib = eq_tbl[jj].mul(rs1_f);
                                        std.debug.print("[STAGE4 INIT]   rs1_contrib[{}]: val={any}, eq={any}, contrib={any}\n", .{
                                            jj, rs1_f.toBytes()[0..8], eq_tbl[jj].toBytes()[0..8], contrib.toBytes()[0..8],
                                        });
                                        rs1_ccount += 1;
                                    }
                                }
                            }
                        }
                    }
                    std.debug.print("[STAGE4 INIT]   total rs1 nonzero entries: {}\n", .{rs1_nz_count});
                    // Check cycle 54 specifically
                    if (54 < trace.steps.items.len) {
                        const step54 = trace.steps.items[54];
                        const rs1_54: u5 = @truncate((step54.instruction >> 15) & 0x1f);
                        const rs2_54: u5 = @truncate((step54.instruction >> 20) & 0x1f);
                        std.debug.print("[STAGE4 INIT] CYCLE 54: is_noop={}, is_term_store={}, opcode=0x{x:0>2}, rs1_reg={}, rs2_reg={}, rs1_value={}, rs2_value={}\n", .{
                            step54.is_noop, step54.is_termination_store, step54.instruction & 0x7f,
                            rs1_54, rs2_54, step54.rs1_value, step54.rs2_value,
                        });
                        // Check polynomial values at cycle 54 for rs1 register
                        const val_at_rs1 = val_poly[@as(usize, rs1_54) * T + 54];
                        const ra_at_rs1 = rs1_ra_poly[@as(usize, rs1_54) * T + 54];
                        const val_at_rs2 = val_poly[@as(usize, rs2_54) * T + 54];
                        const ra_at_rs2 = rs2_ra_poly[@as(usize, rs2_54) * T + 54];
                        std.debug.print("[STAGE4 INIT] CYCLE 54: val[rs1={},54]={any}, rs1_ra[rs1,54]={any}\n", .{
                            rs1_54, val_at_rs1.toBytes()[0..8], ra_at_rs1.toBytes()[0..8],
                        });
                        std.debug.print("[STAGE4 INIT] CYCLE 54: val[rs2={},54]={any}, rs2_ra[rs2,54]={any}\n", .{
                            rs2_54, val_at_rs2.toBytes()[0..8], ra_at_rs2.toBytes()[0..8],
                        });
                    }
                    // Print ALL non-zero rs1 contributions
                    var rs1_all_c: usize = 0;
                    for (0..T) |jj| {
                        if (jj < trace.steps.items.len) {
                            const step_a = trace.steps.items[jj];
                            if (!step_a.is_noop or step_a.is_termination_store) {
                                const op_a = step_a.instruction & 0x7f;
                                const reads_rs1_a = switch (op_a) {
                                    0x13, 0x03, 0x67, 0x1b, 0x33, 0x3b, 0x23, 0x63 => true,
                                    else => false,
                                };
                                if (reads_rs1_a) {
                                    const rs1_fa = F.fromU64(step_a.rs1_value);
                                    if (!rs1_fa.eql(F.zero())) {
                                        rs1_all_c += 1;
                                        std.debug.print("[STAGE4 INIT]   rs1_all[{}/cycle={}]: val={any}\n", .{
                                            rs1_all_c, jj, rs1_fa.toBytes()[0..8],
                                        });
                                    }
                                }
                            }
                        }
                    }
                }
                // Print first 10 trace rs1/rd values for comparison with Stage 3 witnesses
                for (0..@min(10, trace.steps.items.len)) |jj| {
                    const step = trace.steps.items[jj];
                    if (!step.is_noop or step.is_termination_store) {
                        const op = step.instruction & 0x7f;
                        const reads_rs1_p = switch (op) {
                            0x13, 0x03, 0x67, 0x1b, 0x33, 0x3b, 0x23, 0x63 => true,
                            else => false,
                        };
                        const rs1_f = if (reads_rs1_p) F.fromU64(step.rs1_value) else F.zero();
                        const writes_rd_p = switch (op) {
                            0x23, 0x63 => false,
                            else => true,
                        };
                        const rd_reg: u5 = @truncate((step.instruction >> 7) & 0x1f);
                        const rd_f = if (writes_rd_p and rd_reg != 0) F.fromU64(step.rd_value) else F.zero();
                        if (!rs1_f.eql(F.zero()) or !rd_f.eql(F.zero())) {
                            std.debug.print("[STAGE4 INIT]   trace[{}]: rd={any}, rs1={any}\n", .{
                                jj, rd_f.toBytes()[0..8], rs1_f.toBytes()[0..8],
                            });
                        }
                    }
                }
                // Print eq_tbl at a few key indices
                std.debug.print("[STAGE4 INIT]   eq_tbl[0]     = {any}\n", .{eq_tbl[0].toBytes()[0..8]});
                std.debug.print("[STAGE4 INIT]   eq_tbl[1]     = {any}\n", .{eq_tbl[1].toBytes()[0..8]});
                std.debug.print("[STAGE4 INIT]   eq_tbl[63]    = {any}\n", .{eq_tbl[63].toBytes()[0..8]});
                std.debug.print("[STAGE4 INIT]   eq_tbl[64]    = {any}\n", .{eq_tbl[64].toBytes()[0..8]});
                std.debug.print("[STAGE4 INIT]   r_cycle[0]    = {any}\n", .{r_cycle[0].toBytes()[0..8]});
                std.debug.print("[STAGE4 INIT]   r_cycle[7]    = {any}\n", .{r_cycle[7].toBytes()[0..8]});

                if (!brute_sum.eql(s3_claim)) {
                    std.debug.print("[STAGE4 INIT]   rd_write_sum  = {any}\n", .{rd_write_sum.toBytes()});
                    std.debug.print("[STAGE4 INIT]   rs1_read_sum  = {any}\n", .{rs1_read_sum.toBytes()});
                    std.debug.print("[STAGE4 INIT]   rs2_read_sum  = {any}\n", .{rs2_read_sum.toBytes()});

                    // Debug: Check per-cycle values at first few active cycles
                    // AND compare with raw trace values
                    var all_rs1_match = true;
                    var all_rs2_match = true;
                    for (0..T) |jj| {
                        var rd_wv_j = F.zero();
                        var rs1_v_j = F.zero();
                        var rs2_v_j = F.zero();
                        for (0..K) |kk| {
                            const pidx = kk * T + jj;
                            const v = val_poly[pidx];
                            const wa_v = rd_wa_poly[pidx];
                            const inc_v = inc_poly[jj];
                            const rs1_v = rs1_ra_poly[pidx];
                            const rs2_v = rs2_ra_poly[pidx];
                            rd_wv_j = rd_wv_j.add(wa_v.mul(v.add(inc_v)));
                            rs1_v_j = rs1_v_j.add(rs1_v.mul(v));
                            rs2_v_j = rs2_v_j.add(rs2_v.mul(v));
                        }
                        // Compare with trace values
                        if (jj < trace.steps.items.len and (!trace.steps.items[jj].is_noop or trace.steps.items[jj].is_termination_store)) {
                            const step = trace.steps.items[jj];
                            const op = step.instruction & 0x7f;
                            const reads_rs1_cmp = switch (op) {
                                0x13, 0x03, 0x67, 0x1b, 0x33, 0x3b, 0x23, 0x63 => true,
                                else => false,
                            };
                            const expected_rs1 = if (reads_rs1_cmp) F.fromU64(step.rs1_value) else F.zero();
                            if (!rs1_v_j.eql(expected_rs1)) {
                                all_rs1_match = false;
                                if (jj < 60) {
                                    std.debug.print("[STAGE4 INIT]   RS1 MISMATCH cycle {}: poly={any}, trace={any}\n", .{
                                        jj, rs1_v_j.toBytes()[0..8], expected_rs1.toBytes()[0..8],
                                    });
                                }
                            }
                            const reads_rs2_cmp = switch (op) {
                                0x33, 0x3b, 0x23, 0x63 => true,
                                else => false,
                            };
                            const expected_rs2 = if (reads_rs2_cmp) F.fromU64(step.rs2_value) else F.zero();
                            if (!rs2_v_j.eql(expected_rs2)) {
                                all_rs2_match = false;
                                if (jj < 60) {
                                    std.debug.print("[STAGE4 INIT]   RS2 MISMATCH cycle {}: poly={any}, trace={any}\n", .{
                                        jj, rs2_v_j.toBytes()[0..8], expected_rs2.toBytes()[0..8],
                                    });
                                }
                            }
                        }
                        // Only print non-zero cycles (first 8)
                        if (jj < 8 and (!rd_wv_j.eql(F.zero()) or !rs1_v_j.eql(F.zero()) or !rs2_v_j.eql(F.zero()))) {
                            std.debug.print("[STAGE4 INIT]   cycle {}: rd_wv={any}, rs1_v={any}, rs2_v={any}\n", .{
                                jj, rd_wv_j.toBytes()[0..8], rs1_v_j.toBytes()[0..8], rs2_v_j.toBytes()[0..8],
                            });
                        }
                    }
                    std.debug.print("[STAGE4 INIT]   all_rs1_per_cycle_match? {}\n", .{all_rs1_match});
                    std.debug.print("[STAGE4 INIT]   all_rs2_per_cycle_match? {}\n", .{all_rs2_match});
                    if (stage3_claims) |claims| {
                        std.debug.print("[STAGE4 INIT]   s3.rd_wv      = {any}\n", .{claims.rd_write_value.toBytes()});
                        std.debug.print("[STAGE4 INIT]   s3.rs1_v      = {any}\n", .{claims.rs1_value.toBytes()});
                        std.debug.print("[STAGE4 INIT]   s3.rs2_v      = {any}\n", .{claims.rs2_value.toBytes()});

                        // Check: combined_brute = rd_write + gamma*rs1 + gamma^2*rs2
                        const combined_brute = rd_write_sum.add(
                            gamma.mul(rs1_read_sum)).add(
                            gamma_sq.mul(rs2_read_sum));
                        std.debug.print("[STAGE4 INIT]   combined_brute = {any}\n", .{combined_brute.toBytes()});
                        std.debug.print("[STAGE4 INIT]   combined==brute_sum? {}\n", .{combined_brute.eql(brute_sum)});

                        // Per-component match check
                        std.debug.print("[STAGE4 INIT]   rd_write MATCH? {}\n", .{rd_write_sum.eql(claims.rd_write_value)});
                        std.debug.print("[STAGE4 INIT]   rs1_read MATCH? {}\n", .{rs1_read_sum.eql(claims.rs1_value)});
                        std.debug.print("[STAGE4 INIT]   rs2_read MATCH? {}\n", .{rs2_read_sum.eql(claims.rs2_value)});
                    }
                }
            }

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
                .phase1_num_rounds = phase1_num_rounds,
                .phase2_num_rounds = phase2_num_rounds,
                .merged_eq = null,
                .trace_ref = trace,
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
            if (self.merged_eq) |merged| {
                self.allocator.free(merged);
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
            std.debug.print("[STAGE4_GRUEN] Input claim (from stage3) = {any}\n", .{input_claim.toBytes()[0..8]});
            std.debug.print("[STAGE4_GRUEN] Batched claim = {any}\n", .{current_claim.toBytes()[0..8]});

            // CRITICAL DIAGNOSTIC: Compute polynomial sum using brute-force eq(r_cycle, j)
            // This does NOT use Gruen decomposition - it computes eq directly from r_cycle
            {
                // Build standard eq(r_cycle, j) table
                const eq_table = self.allocator.alloc(F, self.T) catch unreachable;
                defer self.allocator.free(eq_table);
                eq_table[0] = F.one();
                var eq_len: usize = 1;
                // r_cycle is in LE order: r_cycle[0] = LSB, binds bit 0 of j
                for (0..self.r_cycle.len) |bit_idx| {
                    const r_i = self.r_cycle[bit_idx];
                    const one_minus_r = F.one().sub(r_i);
                    // Expand: for each existing entry, split into (entry*(1-r), entry*r)
                    var idx: usize = eq_len;
                    while (idx > 0) {
                        idx -= 1;
                        eq_table[2 * idx + 1] = eq_table[idx].mul(r_i);
                        eq_table[2 * idx] = eq_table[idx].mul(one_minus_r);
                    }
                    eq_len *= 2;
                }

                // Now compute Σ_j Σ_k eq(r_cycle, j) * body(k, j)
                var brute_claim = F.zero();
                for (0..K) |k| {
                    for (0..self.T) |j| {
                        const poly_idx = k * self.T + j;
                        const val_v = self.val_poly[poly_idx];
                        const ra_v = self.ra_poly[poly_idx];
                        const wa_v = self.rd_wa_poly[poly_idx];
                        const inc_v = self.inc_poly[j];
                        const body = ra_v.mul(val_v).add(wa_v.mul(val_v.add(inc_v)));
                        brute_claim = brute_claim.add(eq_table[j].mul(body));
                    }
                }

                std.debug.print("[STAGE4_GRUEN] BRUTE FORCE poly sum   = {any}\n", .{brute_claim.toBytes()[0..8]});
                std.debug.print("[STAGE4_GRUEN] Stage 3 input_claim    = {any}\n", .{input_claim.toBytes()[0..8]});
                if (!brute_claim.eql(input_claim)) {
                    std.debug.print("[STAGE4_GRUEN] *** MISMATCH: poly arrays don't match stage3 claim! ***\n", .{});

                    // Break down by component
                    var rd_write_sum = F.zero();
                    var rs1_read_sum = F.zero();
                    var rs2_read_sum = F.zero();
                    for (0..K) |k| {
                        for (0..self.T) |j| {
                            const poly_idx = k * self.T + j;
                            const val_v = self.val_poly[poly_idx];
                            const wa_v = self.rd_wa_poly[poly_idx];
                            const inc_v = self.inc_poly[j];
                            const rs1_ra_v = self.rs1_ra_poly[poly_idx];
                            const rs2_ra_v = self.rs2_ra_poly[poly_idx];

                            // rd write: wa * (val + inc)
                            rd_write_sum = rd_write_sum.add(eq_table[j].mul(wa_v.mul(val_v.add(inc_v))));
                            // rs1 read: rs1_ra * val
                            rs1_read_sum = rs1_read_sum.add(eq_table[j].mul(rs1_ra_v.mul(val_v)));
                            // rs2 read: rs2_ra * val
                            rs2_read_sum = rs2_read_sum.add(eq_table[j].mul(rs2_ra_v.mul(val_v)));
                        }
                    }
                    std.debug.print("[STAGE4_GRUEN]   rd_write_sum  = {any}\n", .{rd_write_sum.toBytes()[0..8]});
                    std.debug.print("[STAGE4_GRUEN]   rs1_read_sum  = {any}\n", .{rs1_read_sum.toBytes()[0..8]});
                    std.debug.print("[STAGE4_GRUEN]   rs2_read_sum  = {any}\n", .{rs2_read_sum.toBytes()[0..8]});

                    // Compare with Stage 3 claims
                    if (self.stage3_claims) |claims| {
                        std.debug.print("[STAGE4_GRUEN]   stage3 rd_wv  = {any}\n", .{claims.rd_write_value.toBytes()[0..8]});
                        std.debug.print("[STAGE4_GRUEN]   stage3 rs1_v  = {any}\n", .{claims.rs1_value.toBytes()[0..8]});
                        std.debug.print("[STAGE4_GRUEN]   stage3 rs2_v  = {any}\n", .{claims.rs2_value.toBytes()[0..8]});

                        // Check combined: rd_wv + gamma*rs1_v + gamma^2*rs2_v
                        const combined_brute = rd_write_sum.add(
                            self.gamma.mul(rs1_read_sum)).add(
                            self.gamma_sq.mul(rs2_read_sum));
                        std.debug.print("[STAGE4_GRUEN]   combined_brute = {any}\n", .{combined_brute.toBytes()[0..8]});

                        // Check: does rd_write_sum == stage3.rd_write_value ?
                        if (rd_write_sum.eql(claims.rd_write_value)) {
                            std.debug.print("[STAGE4_GRUEN]   rd_write MATCHES stage3 ✓\n", .{});
                        } else {
                            std.debug.print("[STAGE4_GRUEN]   rd_write DIFFERS from stage3 ✗\n", .{});
                        }
                        if (rs1_read_sum.eql(claims.rs1_value)) {
                            std.debug.print("[STAGE4_GRUEN]   rs1_read MATCHES stage3 ✓\n", .{});
                        } else {
                            std.debug.print("[STAGE4_GRUEN]   rs1_read DIFFERS from stage3 ✗\n", .{});
                        }
                        if (rs2_read_sum.eql(claims.rs2_value)) {
                            std.debug.print("[STAGE4_GRUEN]   rs2_read MATCHES stage3 ✓\n", .{});
                        } else {
                            std.debug.print("[STAGE4_GRUEN]   rs2_read DIFFERS from stage3 ✗\n", .{});
                        }
                    }
                } else {
                    std.debug.print("[STAGE4_GRUEN] ✓ Brute force poly sum MATCHES stage3 claim\n", .{});
                }
            }

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

            // BRUTE FORCE CHECK: Verify val_poly[0] matches MLE evaluation
            // The sumcheck challenges define the evaluation point.
            // Phase 1 challenges (0..phase1) bind cycle vars LSB first
            // Phase 2 challenges (phase1..phase1+phase2) bind address vars LSB first
            // Phase 3 challenges (phase1+phase2..) bind remaining cycle vars
            //
            // For default config: phase1=8 (all cycle), phase2=7 (all address), phase3=0
            // So r_cycle = challenges[0..8] (LSB first), r_address = challenges[8..15] (LSB first)
            //
            // MLE evaluation: val(r_address, r_cycle) = Σ_{k,j} eq(r_address,k)*eq(r_cycle,j)*val_orig(k,j)
            // But val_poly has been modified by binding! So we can't do this check directly.
            //
            // Instead, check that the ORIGINAL val_poly (before binding) would give the same answer
            // as the folded val_poly[0]. We rebuild from trace.
            {
                // Rebuild val_poly from scratch
                var brute_regs: [32]u64 = [_]u64{0} ** 32;
                var brute_val_claim = F.zero();

                // Compute eq tables for the opening point
                // Phase 1 challenges bind cycle LSB first: challenges[0] = cycle bit 0 (LSB)
                // Phase 2 challenges bind address LSB first: challenges[phase1] = address bit 0 (LSB)
                // After normalize_opening_point: r_cycle = reversed(phase3_cycle) ++ reversed(phase1)
                // For phase3=0: r_cycle_be = reversed(challenges[0..8]) = [c7,c6,...,c0] (BIG_ENDIAN)
                // r_address_be = reversed(challenges[8..15]) = [c14,c13,...,c8] (BIG_ENDIAN)

                // Build eq_cycle table (BIG_ENDIAN: r[0]=MSB)
                var r_cycle_be2: [8]F = undefined;
                for (0..self.phase1_num_rounds) |i| {
                    r_cycle_be2[i] = challenges[self.phase1_num_rounds - 1 - i];
                }
                var eq_cycle2 = try self.allocator.alloc(F, self.T);
                defer self.allocator.free(eq_cycle2);
                // Use same algorithm as Stage 5's computeEqAtIndex but for all j
                for (0..self.T) |j| {
                    var eq_val = F.one();
                    for (0..self.phase1_num_rounds) |bit_idx| {
                        const bj: u1 = @truncate(j >> @intCast(self.phase1_num_rounds - 1 - bit_idx));
                        const rj = r_cycle_be2[bit_idx];
                        if (bj == 1) {
                            eq_val = eq_val.mulHiBigIntU128(rj.limbs);
                        } else {
                            eq_val = eq_val.mul(F.one().sub(rj));
                        }
                    }
                    eq_cycle2[j] = eq_val;
                }

                // Build eq_addr table
                const addr_bits = self.phase2_num_rounds;
                var r_addr_be2: [7]F = undefined;
                for (0..addr_bits) |i| {
                    r_addr_be2[i] = challenges[self.phase1_num_rounds + addr_bits - 1 - i];
                }
                var eq_addr2: [128]F = undefined;
                for (0..K) |k| {
                    var eq_val = F.one();
                    for (0..addr_bits) |bit_idx| {
                        const bk: u1 = @truncate(k >> @intCast(addr_bits - 1 - bit_idx));
                        const rk = r_addr_be2[bit_idx];
                        if (bk == 1) {
                            eq_val = eq_val.mulHiBigIntU128(rk.limbs);
                        } else {
                            eq_val = eq_val.mul(F.one().sub(rk));
                        }
                    }
                    eq_addr2[k] = eq_val;
                }

                // Compute val(r_address, r_cycle) = Σ_{k,j} eq_addr(k) * eq_cycle(j) * val_orig(k,j)
                const trace_items = self.trace_ref.?.steps.items;
                for (trace_items, 0..) |_, j| {
                    for (0..32) |k| {
                        if (!eq_addr2[k].eql(F.zero()) and !eq_cycle2[j].eql(F.zero())) {
                            brute_val_claim = brute_val_claim.add(
                                eq_addr2[k].mul(eq_cycle2[j]).mul(F.fromU64(brute_regs[k]))
                            );
                        }
                    }
                    // Update registers
                    const step2 = trace_items[j];
                    if (!step2.is_noop or step2.is_termination_store) {
                        if (!step2.is_termination_store) {
                            const instr2 = step2.instruction;
                            const rd2: u5 = @truncate((instr2 >> 7) & 0x1f);
                            const opcode2 = instr2 & 0x7f;
                            const rd_used2 = switch (opcode2) {
                                0x23, 0x63 => false,
                                else => true,
                            };
                            if (rd_used2 and rd2 != 0 and rd2 < 32) {
                                brute_regs[rd2] = step2.rd_value;
                            }
                        }
                    }
                }
                // Add padding cycles
                for (trace_items.len..self.T) |j| {
                    for (0..32) |k| {
                        if (!eq_addr2[k].eql(F.zero()) and !eq_cycle2[j].eql(F.zero())) {
                            brute_val_claim = brute_val_claim.add(
                                eq_addr2[k].mul(eq_cycle2[j]).mul(F.fromU64(brute_regs[k]))
                            );
                        }
                    }
                }

                std.debug.print("\n[STAGE4 BRUTE] val_poly[0] after folding = {any}\n", .{self.val_poly[0].toBytesBE()[0..16]});
                std.debug.print("[STAGE4 BRUTE] brute force val(r_addr,r_cycle) = {any}\n", .{brute_val_claim.toBytesBE()[0..16]});
                std.debug.print("[STAGE4 BRUTE] match? {}\n", .{self.val_poly[0].eql(brute_val_claim)});
            }

            // Final claims after all rounds
            const val_claim = self.val_poly[0];
            const rs1_ra_claim = self.rs1_ra_poly[0];
            const rs2_ra_claim = self.rs2_ra_poly[0];
            const rd_wa_claim = self.rd_wa_poly[0];
            const inc_claim = self.inc_poly[0];

            // Debug: Print final eq scalar for comparison with Jolt's eq_eval
            if (self.gruen_eq) |gruen| {
                std.debug.print("\n[ZOLT STAGE4 EQ_EVAL] gruen.current_scalar = {any}\n", .{gruen.current_scalar.toBytes()});
                std.debug.print("[ZOLT STAGE4 EQ_EVAL] This should match Jolt's eq_eval in expected_output_claim\n", .{});
            }

            // Debug: Print final merged_eq[0] after all Phase 3 bindings
            if (self.merged_eq) |merged| {
                std.debug.print("\n[ZOLT STAGE4 FINAL] merged_eq[0] (final eq scalar) = {any}\n", .{merged[0].toBytes()});
                std.debug.print("[ZOLT STAGE4 FINAL] This should match Jolt's eq_eval\n", .{});

                // Compute what the final polynomial sum should be
                const eq_scalar = merged[0];
                const ra = self.ra_poly[0];
                const wa = self.rd_wa_poly[0];
                const val = self.val_poly[0];
                const inc = self.inc_poly[0];
                const expected_combined = ra.mul(val).add(wa.mul(val.add(inc)));
                const expected_poly_final = eq_scalar.mul(expected_combined);
                std.debug.print("[ZOLT STAGE4 FINAL] computed eq * combined = {any}\n", .{expected_poly_final.toBytes()});
            }

            // Debug: Print final claims for comparison with Jolt
            std.debug.print("\n[ZOLT STAGE4 FINAL CLAIMS]\n", .{});
            std.debug.print("[ZOLT STAGE4]   val_claim = {any}\n", .{val_claim.toBytes()});
            std.debug.print("[ZOLT STAGE4]   rs1_ra_claim = {any}\n", .{rs1_ra_claim.toBytes()});
            std.debug.print("[ZOLT STAGE4]   rs2_ra_claim = {any}\n", .{rs2_ra_claim.toBytes()});
            std.debug.print("[ZOLT STAGE4]   rd_wa_claim = {any}\n", .{rd_wa_claim.toBytes()});
            std.debug.print("[ZOLT STAGE4]   inc_claim = {any}\n", .{inc_claim.toBytes()});

            // Compute expected output claim using Jolt's formula
            const rd_write_value_claim = rd_wa_claim.mul(inc_claim.add(val_claim));
            const rs1_value_claim = rs1_ra_claim.mul(val_claim);
            const rs2_value_claim = rs2_ra_claim.mul(val_claim);
            const combined = rd_write_value_claim.add(
                self.gamma.mul(rs1_value_claim.add(self.gamma.mul(rs2_value_claim)))
            );
            std.debug.print("[ZOLT STAGE4]   rd_write_value_claim = {any}\n", .{rd_write_value_claim.toBytes()});
            std.debug.print("[ZOLT STAGE4]   rs1_value_claim = {any}\n", .{rs1_value_claim.toBytes()});
            std.debug.print("[ZOLT STAGE4]   rs2_value_claim = {any}\n", .{rs2_value_claim.toBytes()});
            std.debug.print("[ZOLT STAGE4]   combined = {any}\n", .{combined.toBytes()});

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

        /// Compute round polynomial using Gruen optimization with 3-phase structure
        ///
        /// Phase 1 (rounds 0 to phase1-1): Bind first phase1_num_rounds cycle vars via Gruen
        /// Phase 2 (rounds phase1 to phase1+phase2-1): Bind address vars (eq NOT bound)
        /// Phase 3 (rounds phase1+phase2 to end): Bind remaining cycle vars via merged eq
        fn computeRoundPolynomialGruen(self: *Self, round: usize, current_claim: F) RoundPoly(F) {
            const phase1_end = self.phase1_num_rounds;
            const phase2_end = phase1_end + self.phase2_num_rounds;

            if (round < phase1_end) {
                // Phase 1: Binding first cycle variables using Gruen
                std.debug.print("[STAGE4] Round {} -> Phase 1 (cycle via Gruen)\n", .{round});
                return self.phase1ComputeMessage(current_claim);
            } else if (round < phase2_end) {
                // Phase 2: Binding address variables (eq polynomial NOT bound here)
                std.debug.print("[STAGE4] Round {} -> Phase 2 (address, eq not bound)\n", .{round});
                return self.phase2ComputeMessage(round, current_claim);
            } else {
                // Phase 3: Binding remaining cycle variables using merged dense eq
                std.debug.print("[STAGE4] Round {} -> Phase 3 (cycle via dense eq)\n", .{round});
                return self.phase3ComputeMessage(round, current_claim);
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

            // Compute ACTUAL q(1) for diagnostic
            var q_1_actual = F.zero();
            for (0..half_T) |ia| {
                const j_odd_a = 2 * ia + 1;
                const x_in_a = if (num_x_in_bits > 0) (ia & x_bitmask) else 0;
                const x_out_a = if (num_x_in_bits < 64) (ia >> @as(u6, @intCast(num_x_in_bits))) else 0;
                const E_in_a = if (x_in_a < E_in_len) E_in[x_in_a] else F.one();
                const E_out_a = if (x_out_a < E_out.len) E_out[x_out_a] else F.one();
                const E_comb_a = E_out_a.mul(E_in_a);
                const inc_a = self.inc_poly[j_odd_a];
                for (0..self.current_K) |ka| {
                    const idx_a = ka * self.T + j_odd_a;
                    const ra_a = self.ra_poly[idx_a];
                    const wa_a = self.rd_wa_poly[idx_a];
                    const val_a = self.val_poly[idx_a];
                    const body_a = ra_a.mul(val_a).add(wa_a.mul(val_a.add(inc_a)));
                    q_1_actual = q_1_actual.add(E_comb_a.mul(body_a));
                }
            }
            // Now check: s(0) + s(1)_actual should equal previous_claim
            {
                const g = &self.gruen_eq.?;
                const w_curr = g.w[g.current_index - 1];
                const eq_1 = g.current_scalar.mul(w_curr);
                const eq_0 = g.current_scalar.sub(eq_1);
                const s0 = eq_0.mul(q_0);
                const s1_actual = eq_1.mul(q_1_actual);
                const actual_sum = s0.add(s1_actual);
                if (!actual_sum.eql(previous_claim)) {
                    std.debug.print("[PHASE1 ACTUAL MISMATCH] Round: current_T={}\n", .{self.current_T});
                    std.debug.print("  s(0) = {any}\n", .{s0.toBytes()[0..8]});
                    std.debug.print("  s(1)_actual = {any}\n", .{s1_actual.toBytes()[0..8]});
                    std.debug.print("  s(0)+s(1)_actual = {any}\n", .{actual_sum.toBytes()[0..8]});
                    std.debug.print("  previous_claim = {any}\n", .{previous_claim.toBytes()[0..8]});
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

        /// Phase 2: Compute round polynomial for address variable binding
        ///
        /// In Phase 2, the merged_eq polynomial has been computed (from Gruen) but is NOT bound.
        /// We iterate over address pairs (k_even, k_odd) and sum over all cycle indices j,
        /// multiplying by merged_eq[j].
        ///
        /// Uses the from_evals_and_hint pattern: compute p(0) and p(2), recover p(1) = previous_claim - p(0)
        /// Returns degree-2 polynomial: [c0, c1, c2, 0]
        fn phase2ComputeMessage(self: *Self, round: usize, previous_claim: F) RoundPoly(F) {
            const merged_eq = self.merged_eq orelse @panic("merged_eq not initialized for Phase 2");

            const half_K = self.current_K / 2;

            // Accumulate evaluations at X = 0 and X = 2 only (Jolt optimization)
            // p(1) will be recovered from previous_claim
            var eval_0 = F.zero();
            var eval_2 = F.zero();

            for (0..half_K) |i| {
                const k_even = 2 * i;
                const k_odd = k_even + 1;

                for (0..self.current_T) |j| {
                    const idx_even = k_even * self.T + j;
                    const idx_odd = k_odd * self.T + j;
                    const inc_j = self.inc_poly[j];
                    const eq_j = merged_eq[j];

                    // Get polynomial values
                    const ra_even = self.ra_poly[idx_even];
                    const ra_odd = self.ra_poly[idx_odd];
                    const wa_even = self.rd_wa_poly[idx_even];
                    const wa_odd = self.rd_wa_poly[idx_odd];
                    const val_even = self.val_poly[idx_even];
                    const val_odd = self.val_poly[idx_odd];

                    // Compute slopes for linear interpolation
                    const ra_slope = ra_odd.sub(ra_even);
                    const wa_slope = wa_odd.sub(wa_even);
                    const val_slope = val_odd.sub(val_even);

                    // Evaluate combined polynomial at X = 0
                    // C(0) = ra_even*val_even + wa_even*(val_even+inc)
                    const combined_0 = ra_even.mul(val_even).add(wa_even.mul(val_even.add(inc_j)));
                    eval_0 = eval_0.add(eq_j.mul(combined_0));

                    // Evaluate combined polynomial at X = 2
                    const ra_2 = ra_even.add(F.fromU64(2).mul(ra_slope));
                    const wa_2 = wa_even.add(F.fromU64(2).mul(wa_slope));
                    const val_2 = val_even.add(F.fromU64(2).mul(val_slope));
                    const combined_2 = ra_2.mul(val_2).add(wa_2.mul(val_2.add(inc_j)));
                    eval_2 = eval_2.add(eq_j.mul(combined_2));
                }
            }

            // Compute ACTUAL p(1) for debugging (before hint)
            var eval_1_actual = F.zero();
            for (0..half_K) |ii| {
                const k_even2 = 2 * ii;
                const k_odd2 = k_even2 + 1;
                for (0..self.current_T) |jj| {
                    const idx_odd2 = k_odd2 * self.T + jj;
                    const inc_jj = self.inc_poly[jj];
                    const eq_jj = merged_eq[jj];
                    const ra_odd2 = self.ra_poly[idx_odd2];
                    const wa_odd2 = self.rd_wa_poly[idx_odd2];
                    const val_odd2 = self.val_poly[idx_odd2];
                    const combined_1 = ra_odd2.mul(val_odd2).add(wa_odd2.mul(val_odd2.add(inc_jj)));
                    eval_1_actual = eval_1_actual.add(eq_jj.mul(combined_1));
                }
            }

            const actual_sum = eval_0.add(eval_1_actual);
            const hint_needed = !actual_sum.eql(previous_claim);
            if (hint_needed) {
                std.debug.print("[STAGE4 PHASE2 HINT WARNING] Round {}: actual p(0)+p(1) != previous_claim!\n", .{round});
                std.debug.print("  actual p(0)+p(1) = {any}\n", .{actual_sum.toBytes()[0..8]});
                std.debug.print("  previous_claim = {any}\n", .{previous_claim.toBytes()[0..8]});
                std.debug.print("  diff = {any}\n", .{previous_claim.sub(actual_sum).toBytes()[0..8]});
            }

            // Recover p(1) using the hint: p(1) = previous_claim - p(0)
            const eval_1 = previous_claim.sub(eval_0);

            // Debug for first phase 2 round
            if (round == self.phase1_num_rounds) {
                std.debug.print("[STAGE4 PHASE2] First round: evals[0,1,2] = [{any}, {any}, {any}]\n", .{
                    eval_0.toBytes()[0..8],
                    eval_1.toBytes()[0..8],
                    eval_2.toBytes()[0..8],
                });
                std.debug.print("[STAGE4 PHASE2] actual p(1) = {any}\n", .{eval_1_actual.toBytes()[0..8]});
                std.debug.print("[STAGE4 PHASE2] p(0)+p(1)_actual={any}, previous_claim={any}, match={}\n", .{
                    actual_sum.toBytes()[0..8],
                    previous_claim.toBytes()[0..8],
                    actual_sum.eql(previous_claim),
                });
            }

            // Convert 3 evaluations to degree-2 coefficients, then pad to degree-3
            // p(X) = c0 + c1*X + c2*X^2 + 0*X^3
            // p(0) = c0
            // p(1) = c0 + c1 + c2
            // p(2) = c0 + 2*c1 + 4*c2
            const c0 = eval_0;
            // c2 = (p(0) - 2*p(1) + p(2)) / 2
            const two = F.fromU64(2);
            const two_inv = two.inverse() orelse F.one();
            const c2 = eval_0.sub(eval_1.mul(two)).add(eval_2).mul(two_inv);
            // c1 = p(1) - p(0) - c2
            const c1 = eval_1.sub(eval_0).sub(c2);

            return RoundPoly(F){ .coeffs = .{ c0, c1, c2, F.zero() } };
        }

        /// Phase 3: Compute round polynomial for remaining cycle variable binding
        ///
        /// In Phase 3, both merged_eq and inc_poly are bound along with the value polynomials.
        /// This handles the remaining cycle variables after Phase 1/2.
        ///
        /// CRITICAL: Phase 3 has TWO cases (matching Jolt exactly):
        /// 1. Cycle variables remaining (current_T > 1): Degree 3, compute [p(0), p(2), p(3)]
        /// 2. Cycle variables fully bound (current_T == 1): Degree 2, compute [p(0), p(2)]
        ///
        /// Uses the from_evals_and_hint pattern: recover p(1) = previous_claim - p(0)
        fn phase3ComputeMessage(self: *Self, round: usize, previous_claim: F) RoundPoly(F) {
            const merged_eq = self.merged_eq orelse @panic("merged_eq not initialized for Phase 3");

            const half_T = self.current_T / 2;
            const K_prime = self.current_K;

            // Debug Phase 3 state
            std.debug.print("[STAGE4 PHASE3] Round {}: current_T={}, half_T={}, K_prime={}, merged_eq.len={}\n", .{
                round,
                self.current_T,
                half_T,
                K_prime,
                merged_eq.len,
            });

            // Check if cycle variables remain or are fully bound
            // Jolt: if inc.len() > 1, we have cycle vars remaining → degree 3
            //       if inc.len() == 1, cycles fully bound → degree 2
            const cycles_remaining = self.current_T > 1;

            if (cycles_remaining) {
                // DEGREE 3: Cycle variables remaining
                // Compute evaluations at X = 0, 2, 3 (Jolt uses [p(0), p(2), p(3)])
                var eval_0 = F.zero();
                var eval_2 = F.zero();
                var eval_3 = F.zero();

                for (0..half_T) |j| {
                    const j_even = 2 * j;
                    const j_odd = j_even + 1;

                    // Get inc and eq evaluations for this pair
                    const inc_even = self.inc_poly[j_even];
                    const inc_odd = self.inc_poly[j_odd];
                    const inc_slope = inc_odd.sub(inc_even);

                    const eq_even = merged_eq[j_even];
                    const eq_odd = merged_eq[j_odd];
                    const eq_slope = eq_odd.sub(eq_even);

                    // Compute inc and eq at evaluation points
                    const inc_2 = inc_even.add(F.fromU64(2).mul(inc_slope));
                    const inc_3 = inc_even.add(F.fromU64(3).mul(inc_slope));
                    const eq_2 = eq_even.add(F.fromU64(2).mul(eq_slope));
                    const eq_3 = eq_even.add(F.fromU64(3).mul(eq_slope));

                    // Sum over all registers for this cycle pair
                    var inner_0 = F.zero();
                    var inner_2 = F.zero();
                    var inner_3 = F.zero();

                    for (0..K_prime) |k| {
                        const idx_even = k * self.T + j_even;
                        const idx_odd = k * self.T + j_odd;

                        const ra_even = self.ra_poly[idx_even];
                        const ra_odd = self.ra_poly[idx_odd];
                        const wa_even = self.rd_wa_poly[idx_even];
                        const wa_odd = self.rd_wa_poly[idx_odd];
                        const val_even = self.val_poly[idx_even];
                        const val_odd = self.val_poly[idx_odd];

                        const ra_slope = ra_odd.sub(ra_even);
                        const wa_slope = wa_odd.sub(wa_even);
                        const val_slope = val_odd.sub(val_even);

                        // Evaluate combined at X = 0, 2, 3
                        const combined_0 = ra_even.mul(val_even).add(wa_even.mul(val_even.add(inc_even)));
                        inner_0 = inner_0.add(combined_0);

                        const ra_2 = ra_even.add(F.fromU64(2).mul(ra_slope));
                        const wa_2 = wa_even.add(F.fromU64(2).mul(wa_slope));
                        const val_2 = val_even.add(F.fromU64(2).mul(val_slope));
                        const combined_2 = ra_2.mul(val_2).add(wa_2.mul(val_2.add(inc_2)));
                        inner_2 = inner_2.add(combined_2);

                        const ra_3 = ra_even.add(F.fromU64(3).mul(ra_slope));
                        const wa_3 = wa_even.add(F.fromU64(3).mul(wa_slope));
                        const val_3 = val_even.add(F.fromU64(3).mul(val_slope));
                        const combined_3 = ra_3.mul(val_3).add(wa_3.mul(val_3.add(inc_3)));
                        inner_3 = inner_3.add(combined_3);
                    }

                    // Multiply by eq at each evaluation point (Jolt: eq_evals * inner)
                    eval_0 = eval_0.add(eq_even.mul(inner_0));
                    eval_2 = eval_2.add(eq_2.mul(inner_2));
                    eval_3 = eval_3.add(eq_3.mul(inner_3));
                }

                // Recover p(1) using the hint: p(1) = previous_claim - p(0)
                const eval_1 = previous_claim.sub(eval_0);

                std.debug.print("[STAGE4 PHASE3] Round {} (deg3) evals: [0,1,2,3] = [{any}, {any}, {any}, {any}]\n", .{
                    round,
                    eval_0.toBytes()[0..8],
                    eval_1.toBytes()[0..8],
                    eval_2.toBytes()[0..8],
                    eval_3.toBytes()[0..8],
                });

                // Convert 4 evaluations to degree-3 coefficients
                return coeffsFromEvals(.{ eval_0, eval_1, eval_2, eval_3 });
            } else {
                // DEGREE 2: Cycle variables fully bound
                // Compute evaluations at X = 0, 2 only
                const inc_eval = self.inc_poly[0];
                const eq_eval = merged_eq[0];

                var eval_0 = F.zero();
                var eval_2 = F.zero();

                for (0..K_prime / 2) |i| {
                    const k_even = 2 * i;
                    const k_odd = k_even + 1;

                    const ra_even = self.ra_poly[k_even * self.T];
                    const ra_odd = self.ra_poly[k_odd * self.T];
                    const wa_even = self.rd_wa_poly[k_even * self.T];
                    const wa_odd = self.rd_wa_poly[k_odd * self.T];
                    const val_even = self.val_poly[k_even * self.T];
                    const val_odd = self.val_poly[k_odd * self.T];

                    const ra_slope = ra_odd.sub(ra_even);
                    const wa_slope = wa_odd.sub(wa_even);
                    const val_slope = val_odd.sub(val_even);

                    // Evaluate combined at X = 0
                    const combined_0 = ra_even.mul(val_even).add(wa_even.mul(val_even.add(inc_eval)));
                    eval_0 = eval_0.add(combined_0);

                    // Evaluate combined at X = 2
                    const ra_2 = ra_even.add(F.fromU64(2).mul(ra_slope));
                    const wa_2 = wa_even.add(F.fromU64(2).mul(wa_slope));
                    const val_2 = val_even.add(F.fromU64(2).mul(val_slope));
                    const combined_2 = ra_2.mul(val_2).add(wa_2.mul(val_2.add(inc_eval)));
                    eval_2 = eval_2.add(combined_2);
                }

                // Multiply by eq scalar
                eval_0 = eq_eval.mul(eval_0);
                eval_2 = eq_eval.mul(eval_2);

                // Recover p(1) using the hint: p(1) = previous_claim - p(0)
                const eval_1 = previous_claim.sub(eval_0);

                std.debug.print("[STAGE4 PHASE3] Round {} (deg2) evals: [0,1,2] = [{any}, {any}, {any}]\n", .{
                    round,
                    eval_0.toBytes()[0..8],
                    eval_1.toBytes()[0..8],
                    eval_2.toBytes()[0..8],
                });

                // Convert 3 evaluations to degree-2 coefficients, padded to degree-3
                const c0 = eval_0;
                const two = F.fromU64(2);
                const two_inv = two.inverse() orelse F.one();
                const c2 = eval_0.sub(eval_1.mul(two)).add(eval_2).mul(two_inv);
                const c1 = eval_1.sub(eval_0).sub(c2);

                return RoundPoly(F){ .coeffs = .{ c0, c1, c2, F.zero() } };
            }
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
        ///
        /// Phase 1: Bind cycle vars (inc, gruen_eq, value polys over cycle dim)
        ///          At end of Phase 1, merge gruen_eq into dense merged_eq
        /// Phase 2: Bind address vars (value polys over address dim). Eq NOT bound.
        /// Phase 3: Bind remaining cycle vars (inc, merged_eq, value polys over cycle dim)
        fn bindPolynomials(self: *Self, round: usize, challenge: F) void {
            const phase1_end = self.phase1_num_rounds;
            const phase2_end = phase1_end + self.phase2_num_rounds;
            const one_minus_c = F.one().sub(challenge);

            if (round < phase1_end) {
                // Phase 1: Bind cycle variable
                const half_T = self.current_T / 2;

                // Bind cycle variable using low-to-high for value polys
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

                // At the end of Phase 1, merge the Gruen eq into dense form
                if (round == phase1_end - 1) {
                    std.debug.print("[STAGE4 BIND] End of Phase 1 (round {}), merging Gruen eq\n", .{round});
                    const merged = self.gruen_eq.?.merge(self.allocator) catch @panic("Failed to merge eq");
                    self.merged_eq = merged;

                    // Debug: print merged eq info
                    std.debug.print("[STAGE4 BIND]   merged_eq.len = {}, current_T = {}\n", .{
                        merged.len,
                        self.current_T,
                    });
                    if (merged.len > 0) {
                        std.debug.print("[STAGE4 BIND]   merged_eq[0] = {any}\n", .{merged[0].toBytes()[0..8]});
                    }

                    // CRITICAL DEBUG: Compute actual claim from bound arrays at Phase 1->2 transition
                    var check_claim = F.zero();
                    for (0..self.current_K) |kk| {
                        for (0..self.current_T) |jj| {
                            const idx = kk * self.T + jj;
                            const ra_val = self.ra_poly[idx];
                            const wa_val = self.rd_wa_poly[idx];
                            const val_val = self.val_poly[idx];
                            const inc_val = self.inc_poly[jj];
                            const eq_val = merged[jj];
                            const body = ra_val.mul(val_val).add(wa_val.mul(val_val.add(inc_val)));
                            check_claim = check_claim.add(eq_val.mul(body));
                        }
                    }
                    std.debug.print("[STAGE4 BIND] TRANSITION CHECK: computed_claim = {any}\n", .{check_claim.toBytes()[0..8]});
                    std.debug.print("[STAGE4 BIND] TRANSITION CHECK: current_T = {}, current_K = {}\n", .{ self.current_T, self.current_K });
                }
            } else if (round < phase2_end) {
                // Phase 2: Bind address variable (eq NOT bound)
                const half_K = self.current_K / 2;

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
            } else {
                // Phase 3: Bind remaining cycle variables (including inc and merged_eq)
                const half_T = self.current_T / 2;

                // Bind cycle variable for value polys
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

                // Bind merged_eq
                if (self.merged_eq) |merged_eq| {
                    for (0..half_T) |i| {
                        const j_lo = 2 * i;
                        const j_hi = j_lo + 1;
                        merged_eq[i] = merged_eq[j_lo].mul(one_minus_c).add(merged_eq[j_hi].mul(challenge));
                    }
                }

                self.current_T = half_T;
            }
        }

        /// Compute round polynomial evaluations in Toom-Cook format: [p(0), p(1), p(2), p_inf]
        /// where p_inf is the leading coefficient c3 (evaluation at infinity).
        /// This matches Jolt's format for compressed polynomials.
        pub fn computeRoundEvals(self: *Self, round: usize, current_claim: F) [4]F {
            const round_poly = self.computeRoundPolynomialGruen(round, current_claim);
            // Convert coefficients to Toom-Cook evaluations:
            // p(0) = c0
            // p(1) = c0 + c1 + c2 + c3
            // p(2) = c0 + 2*c1 + 4*c2 + 8*c3
            // p_inf = c3 (leading coefficient)
            const c = round_poly.coeffs;
            const p0 = c[0];
            const p1 = c[0].add(c[1]).add(c[2]).add(c[3]);
            const p2 = c[0].add(c[1].mul(F.fromU64(2))).add(c[2].mul(F.fromU64(4))).add(c[3].mul(F.fromU64(8)));
            const p_inf = c[3]; // Leading coefficient (Toom-Cook format)

            // Debug: Verify sumcheck constraint p(0)+p(1) = current_claim
            const sum01 = p0.add(p1);
            if (!sum01.eql(current_claim)) {
                std.debug.print("[STAGE4 SUMCHECK CONSTRAINT VIOLATION] Round {}: p(0)+p(1) != claim\n", .{round});
                std.debug.print("  p(0) = {any}\n", .{p0.toBytes()[0..8]});
                std.debug.print("  p(1) = {any}\n", .{p1.toBytes()[0..8]});
                std.debug.print("  p(0)+p(1) = {any}\n", .{sum01.toBytes()[0..8]});
                std.debug.print("  current_claim = {any}\n", .{current_claim.toBytes()[0..8]});
            }

            return .{ p0, p1, p2, p_inf };
        }

        /// Bind challenge after getting round evaluations (compatible with existing interface)
        pub fn bindChallenge(self: *Self, round: usize, challenge: F) void {
            self.bindPolynomials(round, challenge);

            // Debug: After last round, print final eq scalar
            if (round == self.num_rounds - 1) {
                std.debug.print("\n[ZOLT STAGE4 FINAL BIND] After round {}, merged_eq[0] = ", .{round});
                if (self.merged_eq) |merged| {
                    std.debug.print("{any}\n", .{merged[0].toBytes()});
                    std.debug.print("[ZOLT STAGE4 FINAL BIND] This should match Jolt's eq_eval\n", .{});

                    // Compute expected final sum
                    const eq_scalar = merged[0];
                    const ra = self.ra_poly[0];
                    const wa = self.rd_wa_poly[0];
                    const val = self.val_poly[0];
                    const inc = self.inc_poly[0];
                    const combined = ra.mul(val).add(wa.mul(val.add(inc)));
                    const expected = eq_scalar.mul(combined);
                    std.debug.print("[ZOLT STAGE4 FINAL BIND] eq_scalar = {any}\n", .{eq_scalar.toBytes()});
                    std.debug.print("[ZOLT STAGE4 FINAL BIND] combined (ra*val + wa*(val+inc)) = {any}\n", .{combined.toBytes()});
                    std.debug.print("[ZOLT STAGE4 FINAL BIND] expected (eq * combined) = {any}\n", .{expected.toBytes()});
                } else {
                    std.debug.print("(null)\n", .{});
                }
            }
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
