const std = @import("std");

const Allocator = std.mem.Allocator;
const ThreadPool = @import("zolt_pool").ThreadPool;
const GpuPolyOps = @import("zolt_arith").gpu.GpuPolyOps;
const TraceStep = @import("../../tracer/mod.zig").TraceStep;
const ExecutionTrace = @import("../../tracer/mod.zig").ExecutionTrace;
const gruen_eq = @import("gruen_eq.zig");
const GruenSplitEqPolynomial = gruen_eq.GruenSplitEqPolynomial;
const sparse_registers = @import("sparse_registers.zig");

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

/// Stage 4 Prover using sparse register matrix + Gruen optimization.
///
/// Replaces dense K×T arrays (128×T = 1.5GB for T=65536) with sparse entries
/// (~3 per cycle, ~14MB total). Uses OneHotCoeff lookup tables for the first
/// ~3 cycle rounds, then derefs to field elements.
pub fn Stage4GruenProver(comptime F: type) type {
    const SparseRegsLookup = sparse_registers.SparseRegistersCycleMajor(F, true);
    const SparseRegsField = sparse_registers.SparseRegistersCycleMajor(F, false);
    const SparseRegsAddr = sparse_registers.SparseRegistersAddressMajor(F);

    return struct {
        const Self = @This();

        allocator: Allocator,
        thread_pool: ?*ThreadPool = null,
        gpu_ops: ?*GpuPolyOps = null,

        /// Trace length (power of 2)
        T: usize,
        log_T: usize,

        /// Total number of rounds = LOG_K + log_T
        num_rounds: usize,

        /// gamma challenge (for batching ra = gamma*rs1_ra + gamma^2*rs2_ra)
        gamma: F,
        gamma_sq: F,

        /// r_cycle from Stage 3 (LE order: r_cycle[0] = LSB)
        r_cycle: []const F,

        /// GruenSplitEqPolynomial for efficient eq computation during cycle binding
        gruen_eq: ?GruenSplitEqPolynomial(F),

        /// Stage 3 claims for input claim computation
        stage3_claims: ?Stage3Claims(F),

        /// Dense inc polynomial (T elements, bound during Phase 1)
        inc_poly: []F,

        /// Batching coefficient
        batching_coeff: F,

        /// Phase configuration
        phase1_num_rounds: usize,
        phase2_num_rounds: usize,

        /// Merged eq polynomial for Phase 2 (after Gruen eq is fully bound in Phase 1)
        merged_eq: ?[]F,

        /// Trace reference for computing rs2_ra claim
        trace_ref: ?*const ExecutionTrace,

        /// Current effective sizes
        current_T: usize,
        current_K: usize,

        // ---- Sparse matrix state (exactly one is active at a time) ----
        sparse_lookup: ?SparseRegsLookup,
        sparse_field: ?SparseRegsField,
        address_major: ?SparseRegsAddr,

        /// Stored challenges for getFinalClaims opening point reconstruction
        challenges_buf: []F,
        rounds_completed: usize,

        /// Alias for compatibility with existing code (uses default phase config)
        pub fn initWithClaims(
            allocator: Allocator,
            trace: *const ExecutionTrace,
            gamma: F,
            r_cycle: []const F,
            stage3_claims: ?Stage3Claims(F),
            batching_coeff: F,
            pool: ?*ThreadPool,
        ) !Self {
            const T = std.math.ceilPowerOfTwo(usize, trace.steps.items.len) catch trace.steps.items.len;
            const log_T = @ctz(T);
            // Default config: phase1 = log_T (ALL cycle vars), phase2 = LOG_K (ALL address vars)
            return initWithPhaseConfig(allocator, trace, gamma, r_cycle, stage3_claims, batching_coeff, log_T, LOG_K, pool);
        }

        /// Initialize with explicit phase configuration
        pub fn initWithPhaseConfig(
            allocator: Allocator,
            trace: *const ExecutionTrace,
            gamma: F,
            r_cycle: []const F,
            stage3_claims: ?Stage3Claims(F),
            batching_coeff: F,
            phase1_num_rounds: usize,
            phase2_num_rounds: usize,
            pool: ?*ThreadPool,
        ) !Self {
            const trace_len = trace.steps.items.len;
            if (trace_len == 0) return error.EmptyTrace;

            const T = std.math.ceilPowerOfTwo(usize, trace_len) catch trace_len;
            const log_T = @ctz(T);

            if (r_cycle.len != log_T) return error.InvalidRCycleLength;

            const num_rounds = LOG_K + log_T;
            const gamma_sq = gamma.mul(gamma);

            // Build sparse matrix from trace
            var sparse_lookup = try SparseRegsLookup.fromTrace(allocator, trace, gamma, pool);
            errdefer sparse_lookup.deinit();

            // Batch allocation: inc_poly (T) + r_cycle_copy (log_T) + challenges_buf (num_rounds)
            const combined_len = T + r_cycle.len + num_rounds;
            const combined = try allocator.alloc(F, combined_len);
            const inc_poly = combined[0..T];
            const r_cycle_copy = combined[T .. T + r_cycle.len];
            const challenges_buf = combined[T + r_cycle.len ..][0..num_rounds];

            // Fill r_cycle_copy (LE order) and zero challenges_buf
            @memcpy(r_cycle_copy, r_cycle);
            @memset(challenges_buf, F.zero());

            // Build dense inc_poly from trace — parallel, each cycle independent
            const IncCtx = struct { steps: []const TraceStep, out: []F, tlen: usize };
            const inc_ctx = IncCtx{ .steps = trace.steps.items, .out = inc_poly, .tlen = trace_len };
            const incFn = struct {
                fn f(ctx: IncCtx, cycle: usize) void {
                    if (cycle >= ctx.tlen) {
                        ctx.out[cycle] = F.zero();
                        return;
                    }
                    const step = ctx.steps[cycle];
                    if (!step.is_noop and step.rd_written and step.rd_index != 0) {
                        ctx.out[cycle] = F.fromU64(step.rd_value).sub(F.fromU64(step.rd_pre_value));
                    } else {
                        ctx.out[cycle] = F.zero();
                    }
                }
            }.f;
            if (pool) |tp| {
                tp.parallelFor(T, inc_ctx, incFn);
            } else {
                for (0..T) |cycle| incFn(inc_ctx, cycle);
            }

            // Build GruenSplitEqPolynomial from r_cycle in BE order.
            // Reverse r_cycle_copy in-place, build eq poly (which copies internally), then restore.
            std.mem.reverse(F, r_cycle_copy);
            const gruen_eq_poly = try GruenSplitEqPolynomial(F).init(allocator, r_cycle_copy);
            std.mem.reverse(F, r_cycle_copy); // restore LE order

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
                .inc_poly = inc_poly,
                .batching_coeff = batching_coeff,
                .phase1_num_rounds = phase1_num_rounds,
                .phase2_num_rounds = phase2_num_rounds,
                .merged_eq = null,
                .trace_ref = trace,
                .current_T = T,
                .current_K = K,
                .sparse_lookup = sparse_lookup,
                .sparse_field = null,
                .address_major = null,
                .challenges_buf = challenges_buf,
                .rounds_completed = 0,
            };
        }

        pub fn deinit(self: *Self) void {
            // inc_poly, r_cycle, and challenges_buf are a single batched allocation
            const combined_len = self.T + self.r_cycle.len + self.num_rounds;
            self.allocator.free(self.inc_poly.ptr[0..combined_len]);
            if (self.gruen_eq) |*g| g.deinit();
            if (self.merged_eq) |merged| self.allocator.free(merged);
            if (self.sparse_lookup) |*s| s.deinit();
            if (self.sparse_field) |*s| s.deinit();
            if (self.address_major) |*a| a.deinit();
        }

        /// Compute input claim from Stage 3 claims
        fn computeInputClaim(self: *Self) F {
            if (self.stage3_claims) |claims| {
                return claims.rd_write_value.add(
                    self.gamma.mul(claims.rs1_value.add(self.gamma.mul(claims.rs2_value))),
                );
            }
            // If no stage3 claims, return zero (shouldn't happen in practice)
            return F.zero();
        }

        /// Compute round polynomial evaluations in Toom-Cook format: [p(0), p(1), p(2), p_inf]
        pub fn computeRoundEvals(self: *Self, round: usize, current_claim: F) [4]F {
            const round_poly = self.computeRoundPolynomialGruen(round, current_claim);
            const c = round_poly.coeffs;
            const p0 = c[0];
            const p1 = c[0].add(c[1]).add(c[2]).add(c[3]);
            const two = F.fromU64(2);
            const four = F.fromU64(4);
            const eight = F.fromU64(8);
            const p2 = c[0].add(c[1].mul(two)).add(c[2].mul(four)).add(c[3].mul(eight));
            const p_inf = c[3];
            return .{ p0, p1, p2, p_inf };
        }

        /// Bind challenge after getting round evaluations
        pub fn bindChallenge(self: *Self, round: usize, challenge: F) void {
            self.challenges_buf[round] = challenge;
            self.rounds_completed = round + 1;
            self.bindPolynomials(round, challenge);
        }

        pub fn getFinalClaims(self: *const Self) struct {
            val_claim: F,
            rs1_ra_claim: F,
            rs2_ra_claim: F,
            rd_wa_claim: F,
            inc_claim: F,
        } {
            const inc_claim = self.inc_poly[0];

            // Get combined claims from address-major sparse matrix
            if (self.address_major) |*am| {
                const sparse_claims = am.getFinalClaims();

                // Reconstruct r_address_be and r_cycle_be from stored challenges
                // Phase 1 challenges = challenges[0..phase1] bind cycle vars (LE)
                // Phase 2 challenges = challenges[phase1..phase1+phase2] bind address vars (LE)
                // r_cycle_be = reverse(phase1 challenges)
                // r_address_be = reverse(phase2 challenges)
                var r_cycle_be: [24]F = undefined; // max log_T = 24
                var r_address_be: [7]F = undefined; // LOG_K = 7
                for (0..self.phase1_num_rounds) |i| {
                    r_cycle_be[i] = self.challenges_buf[self.phase1_num_rounds - 1 - i];
                }
                for (0..self.phase2_num_rounds) |i| {
                    r_address_be[i] = self.challenges_buf[self.phase1_num_rounds + self.phase2_num_rounds - 1 - i];
                }

                // Compute rs2_ra from trace using split-eq (matches Jolt compute_rs2_ra_claim).
                // rs2_ra(r_addr, r_cycle) = Σ_j [has_rs2[j]] * eq(r_cycle, j) * eq(r_addr, rs2[j])
                // Uses 2-way split over joint (cycle, address) space for parallelism.
                const rs2_ra_claim = blk: {
                    const trace = self.trace_ref.?;
                    const addr_bits = self.phase2_num_rounds; // = 7

                    // r_joint = [r_cycle_LE..., r_address_LE...]
                    // Split: hi_bits from cycle, lo_bits from remaining cycle + all address
                    const n = self.phase1_num_rounds + addr_bits;
                    const hi_bits = @min(self.phase1_num_rounds, (n + 1) / 2);
                    const lo_bits = n - hi_bits;
                    const cycle_bits_in_lo = lo_bits - addr_bits;

                    // Build E_hi from the high cycle challenge bits (LE order)
                    const e_hi_size = @as(usize, 1) << @intCast(hi_bits);
                    const E_hi = self.allocator.alloc(F, e_hi_size) catch break :blk F.zero();
                    defer self.allocator.free(E_hi);
                    E_hi[0] = F.one();
                    {
                        var len: usize = 1;
                        // hi bits are the top cycle challenges: challenges_buf[cycle_bits_in_lo .. phase1_num_rounds]
                        for (0..hi_bits) |bit| {
                            const ri = self.challenges_buf[cycle_bits_in_lo + bit];
                            const one_m_r = F.one().sub(ri);
                            var j: usize = 0;
                            while (j < len) : (j += 1) {
                                const old = E_hi[j];
                                E_hi[j + len] = old.mul(ri);
                                E_hi[j] = old.mul(one_m_r);
                            }
                            len *= 2;
                        }
                    }

                    // Build E_lo from low cycle bits + all address bits
                    const e_lo_size = @as(usize, 1) << @intCast(lo_bits);
                    const E_lo = self.allocator.alloc(F, e_lo_size) catch break :blk F.zero();
                    defer self.allocator.free(E_lo);
                    E_lo[0] = F.one();
                    {
                        var len: usize = 1;
                        // Joint LE order: [addr_0, ..., addr_6, cycle_0, ..., cycle_{lo-addr-1}]
                        // Address bits first (LSB of joint index)
                        for (0..addr_bits) |bit| {
                            const ri = self.challenges_buf[self.phase1_num_rounds + bit];
                            const one_m_r = F.one().sub(ri);
                            var j: usize = 0;
                            while (j < len) : (j += 1) {
                                const old = E_lo[j];
                                E_lo[j + len] = old.mul(ri);
                                E_lo[j] = old.mul(one_m_r);
                            }
                            len *= 2;
                        }
                        // Low cycle bits next
                        for (0..cycle_bits_in_lo) |bit| {
                            const ri = self.challenges_buf[bit];
                            const one_m_r = F.one().sub(ri);
                            var j: usize = 0;
                            while (j < len) : (j += 1) {
                                const old = E_lo[j];
                                E_lo[j + len] = old.mul(ri);
                                E_lo[j] = old.mul(one_m_r);
                            }
                            len *= 2;
                        }
                    }

                    // Parallel reduce over E_hi blocks
                    const cycles_per_block = @as(usize, 1) << @intCast(cycle_bits_in_lo);
                    const cycle_lo_mask = cycles_per_block - 1;
                    const trace_len = trace.steps.items.len;

                    const SplitEqCtx = struct {
                        e_hi: []const F,
                        e_lo: []const F,
                        steps: []const @import("../../tracer/mod.zig").TraceStep,
                        cpb: usize,
                        clm: usize,
                        ab: usize,
                        tlen: usize,
                    };
                    const ctx = SplitEqCtx{
                        .e_hi = E_hi,
                        .e_lo = E_lo,
                        .steps = trace.steps.items,
                        .cpb = cycles_per_block,
                        .clm = cycle_lo_mask,
                        .ab = addr_bits,
                        .tlen = trace_len,
                    };
                    const mapFn = struct {
                        fn f(c: SplitEqCtx, start: usize, end: usize) F {
                            var acc = F.zero();
                            for (start..end) |idx_hi| {
                                const block_start = idx_hi * c.cpb;
                                if (block_start >= c.tlen) continue;
                                const block_end = @min(block_start + c.cpb, c.tlen);
                                var inner = F.zero();
                                for (block_start..block_end) |j| {
                                    const step = c.steps[j];
                                    if (step.is_noop or !step.rs2_read) continue;
                                    const j_in_block = j & c.clm;
                                    const idx_lo = (j_in_block << @intCast(c.ab)) | @as(usize, step.rs2_index);
                                    inner = inner.add(c.e_lo[idx_lo]);
                                }
                                acc = acc.add(c.e_hi[idx_hi].mul(inner));
                            }
                            return acc;
                        }
                    }.f;
                    const reduceFn = struct {
                        fn f(a: F, b: F) F { return a.add(b); }
                    }.f;

                    if (self.thread_pool) |tp| {
                        break :blk tp.parallelReduce(F, e_hi_size, F.zero(), ctx, mapFn, reduceFn);
                    } else {
                        break :blk mapFn(ctx, 0, e_hi_size);
                    }
                };

                // Derive rs1_ra = (combined_ra - gamma^2 * rs2_ra) / gamma
                const gamma_inv = self.gamma.inverse() orelse F.zero();
                const rs1_ra_claim = sparse_claims.combined_ra.sub(self.gamma_sq.mul(rs2_ra_claim)).mul(gamma_inv);

                return .{
                    .val_claim = sparse_claims.val,
                    .rs1_ra_claim = rs1_ra_claim,
                    .rs2_ra_claim = rs2_ra_claim,
                    .rd_wa_claim = sparse_claims.rd_wa,
                    .inc_claim = inc_claim,
                };
            }

            // Fallback: shouldn't happen if sumcheck completed normally
            return .{
                .val_claim = F.zero(),
                .rs1_ra_claim = F.zero(),
                .rs2_ra_claim = F.zero(),
                .rd_wa_claim = F.zero(),
                .inc_claim = inc_claim,
            };
        }

        // ================================================================
        // Internal: Round polynomial computation
        // ================================================================

        fn computeRoundPolynomialGruen(self: *Self, round: usize, current_claim: F) RoundPoly(F) {
            const phase1_end = self.phase1_num_rounds;
            const phase2_end = phase1_end + self.phase2_num_rounds;

            if (round < phase1_end) {
                return self.phase1ComputeMessage(current_claim);
            } else if (round < phase2_end) {
                return self.phase2ComputeMessage(current_claim);
            } else {
                // Phase 3: should not happen with default config (phase3=0)
                unreachable;
            }
        }

        /// Phase 1: Compute round polynomial for cycle variable binding using Gruen + sparse matrix
        fn phase1ComputeMessage(self: *Self, previous_claim: F) RoundPoly(F) {
            const gruen = &self.gruen_eq.?;
            const inc = self.inc_poly[0..self.current_T];

            if (self.sparse_lookup) |*sl| {
                const coeffs = sl.computeMessage(inc, gruen, previous_claim, self.thread_pool);
                return RoundPoly(F){ .coeffs = coeffs };
            } else if (self.sparse_field) |*sf| {
                const coeffs = sf.computeMessage(inc, gruen, previous_claim, self.thread_pool);
                return RoundPoly(F){ .coeffs = coeffs };
            } else unreachable;
        }

        /// Phase 2: Compute round polynomial for address variable binding using address-major sparse matrix
        fn phase2ComputeMessage(self: *Self, previous_claim: F) RoundPoly(F) {
            const merged_eq = self.merged_eq orelse @panic("merged_eq not initialized for Phase 2");
            const inc = self.inc_poly[0..self.current_T];
            const am = &self.address_major.?;

            // Compute [eval_0, eval_2] from sparse address-major entries
            const evals_02 = am.computeRoundEvals(inc, merged_eq[0..self.current_T], self.thread_pool);

            const eval_0 = evals_02[0];
            const eval_2 = evals_02[1];

            // Recover eval_1 using hint: p(1) = previous_claim - p(0)
            const eval_1 = previous_claim.sub(eval_0);

            // Convert 3 evaluations to degree-2 coefficients, padded to degree-3
            const c0 = eval_0;
            const two = F.fromU64(2);
            const two_inv = two.inverse().?;
            const c2 = eval_0.sub(eval_1.mul(two)).add(eval_2).mul(two_inv);
            const c1 = eval_1.sub(eval_0).sub(c2);

            return RoundPoly(F){ .coeffs = .{ c0, c1, c2, F.zero() } };
        }

        // ================================================================
        // Internal: Bind polynomials
        // ================================================================

        fn bindPolynomials(self: *Self, round: usize, challenge: F) void {
            const phase1_end = self.phase1_num_rounds;
            const phase2_end = phase1_end + self.phase2_num_rounds;

            if (round < phase1_end) {
                self.phase1Bind(round, challenge);
            } else if (round < phase2_end) {
                self.phase2Bind(challenge);
            } else {
                // Phase 3: should not happen with default config
                unreachable;
            }
        }

        fn phase1Bind(self: *Self, round: usize, challenge: F) void {
            const half_T = self.current_T / 2;

            // Bind inc_poly (dense) — parallel when pool available, GPU-accelerated for large polys
            if (self.gpu_ops) |gpu| {
                if (half_T >= 16384) {
                    gpu.polyBindLow(self.inc_poly[0..self.current_T], challenge, self.inc_poly[0..half_T]) catch {
                        self.bindIncPolyCpu(half_T, challenge);
                    };
                } else {
                    self.bindIncPolyCpu(half_T, challenge);
                }
            } else {
                self.bindIncPolyCpu(half_T, challenge);
            }

            // Bind sparse matrix entries
            if (self.sparse_lookup) |*sl| {
                sl.bind(challenge, self.thread_pool) catch @panic("sparse bind failed");

                // Check if lookup tables are saturated after bind
                if (sl.isLookupSaturated()) {
                    // Deref: convert LookupTableIndex entries to F entries
                    var field_matrix = sl.derefCoeffs() catch @panic("deref failed");
                    self.sparse_lookup = null;
                    self.sparse_field = field_matrix;
                    _ = &field_matrix;
                }
            } else if (self.sparse_field) |*sf| {
                sf.bind(challenge, self.thread_pool) catch @panic("sparse bind failed");
            } else unreachable;

            self.current_T = half_T;

            // Bind the Gruen eq polynomial
            self.gruen_eq.?.bind(challenge);

            // At the end of Phase 1, merge Gruen eq and convert to address-major
            if (round == self.phase1_num_rounds - 1) {
                const merged = self.gruen_eq.?.merge(self.allocator) catch @panic("Failed to merge eq");
                self.merged_eq = merged;

                // Convert to address-major
                if (self.sparse_lookup) |*sl| {
                    var am = sl.toAddressMajor() catch @panic("toAddressMajor failed");
                    self.sparse_lookup = null;
                    self.address_major = am;
                    _ = &am;
                } else if (self.sparse_field) |*sf| {
                    var am = sf.toAddressMajor() catch @panic("toAddressMajor failed");
                    self.sparse_field = null;
                    self.address_major = am;
                    _ = &am;
                } else unreachable;
            }
        }

        fn phase2Bind(self: *Self, challenge: F) void {
            // Bind address-major sparse matrix (includes val_init binding)
            if (self.address_major) |*am| {
                am.bind(challenge, self.thread_pool) catch @panic("address-major bind failed");
            } else unreachable;

            self.current_K = self.current_K / 2;
            // Note: merged_eq and inc_poly are NOT bound during Phase 2 (address binding)
        }

        /// Dense inc_poly bind — uses scratch buffer for parallel execution.
        fn bindIncPolyCpu(self: *Self, half_T: usize, challenge: F) void {
            if (self.thread_pool != null and half_T >= 512) {
                // Parallel: write to scratch buffer, then copy back.
                // In-place is unsafe because thread A reading inc_poly[2i] may race
                // with thread B writing inc_poly[j] when 2i == j.
                const scratch = self.allocator.alloc(F, half_T) catch {
                    // Allocation failed — fall back to sequential
                    self.bindIncPolyCpuSeq(half_T, challenge);
                    return;
                };
                defer self.allocator.free(scratch);
                const Ctx = struct { src: []const F, dst: []F, ch: F };
                self.thread_pool.?.parallelFor(half_T, Ctx{ .src = self.inc_poly, .dst = scratch, .ch = challenge }, struct {
                    fn f(ctx: Ctx, i: usize) void {
                        const lo = ctx.src[2 * i];
                        const hi = ctx.src[2 * i + 1];
                        ctx.dst[i] = lo.add(ctx.ch.mul(hi.sub(lo)));
                    }
                }.f);
                @memcpy(self.inc_poly[0..half_T], scratch);
            } else {
                self.bindIncPolyCpuSeq(half_T, challenge);
            }
        }

        fn bindIncPolyCpuSeq(self: *Self, half_T: usize, challenge: F) void {
            for (0..half_T) |i| {
                const lo = self.inc_poly[2 * i];
                const hi = self.inc_poly[2 * i + 1];
                self.inc_poly[i] = lo.add(challenge.mul(hi.sub(lo)));
            }
        }
    };
}
