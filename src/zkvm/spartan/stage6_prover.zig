//! Stage 6 Batched Sumcheck Prover
//!
//! Stage 6 is a batched sumcheck with 6 instances:
//! 0. BytecodeReadRaf: bytecode_log_k + n_cycle_vars rounds, degree bytecode_d + 1
//! 1. Booleanity: log_k_chunk + n_cycle_vars rounds, degree 3 (input_claim = 0)
//! 2. HammingBooleanity: n_cycle_vars rounds, degree 3 (input_claim = 0)
//! 3. RamRaVirtual: n_cycle_vars rounds, degree ram_d + 1
//! 4. LookupsRaVirtual: n_cycle_vars rounds, degree n_committed_per_virtual + 1
//! 5. IncClaimReduction: n_cycle_vars rounds, degree 2
//!
//! ALL instances use real sumcheck provers with actual polynomial materialization
//! from execution trace data. No shortcuts, no placeholders.

const std = @import("std");

const zkvm_debug = @import("../debug.zig");
const dbg = zkvm_debug.dbg;
const debug_verbose = zkvm_debug.verbose;
// Stage 6 fine-grained bench timing — enabled at runtime via ZOLT_BENCH=1
const s6_bench_timing = true;

// Maximum evaluation points for parallelReduce accumulator.
// Covers all sub-provers: LookupsRa (M+2 ≤ 10), RamRa (d+2 ≤ 6), BytecodeReadRaf (d+2 ≤ 4).
const MAX_RA_EVALS = 16;

const Allocator = std.mem.Allocator;
const ThreadPool = @import("zolt_pool").ThreadPool;
const pool_helpers = @import("zolt_pool").helpers;
const GpuPolyOps = @import("zolt_arith").gpu.GpuPolyOps;

const poly_mod = @import("zolt_arith").poly;
const UniPoly = poly_mod.UniPoly;
const transcripts = @import("zolt_arith").transcripts;
const Blake2bTranscript = transcripts.Blake2bTranscript;
const jolt_types = @import("../jolt_types.zig");
const SumcheckInstanceProof = jolt_types.SumcheckInstanceProof;
const OpeningClaims = jolt_types.OpeningClaims;
const OpeningId = jolt_types.OpeningId;
const tracer = @import("../../tracer/mod.zig");
const ExecutionTrace = tracer.ExecutionTrace;
const ram = @import("../ram/mod.zig");
const jolt_device = @import("../jolt_device.zig");
const instruction_mod = @import("../instruction/mod.zig");
const CircuitFlags = instruction_mod.CircuitFlags;
const InstructionFlags = instruction_mod.InstructionFlags;
const preprocessing = @import("../preprocessing.zig");
const BytecodePCMapper = preprocessing.BytecodePCMapper;
const ra_poly_mod = @import("ra_poly.zig");
const UnreducedProductAccum = @import("zolt_arith").field.UnreducedProductAccum;
const sumcheck_helpers = @import("sumcheck_helpers.zig");

/// Free a large allocation on a detached background thread so the caller doesn't block.
/// Falls back to synchronous free if thread spawn fails.
/// Supports flat slices ([]T) and slices-of-slices ([][]T).
pub fn dropInBackground(allocator: Allocator, slice: anytype) void {
    const T = @TypeOf(slice);
    const SpawnCtx = struct { alloc: Allocator, ptr: T };
    const info = @typeInfo(T);
    const is_slice_of_slices = comptime blk: {
        if (info != .pointer) break :blk false;
        const child_info = @typeInfo(info.pointer.child);
        break :blk (child_info == .pointer and child_info.pointer.size == .slice);
    };
    const ctx = SpawnCtx{ .alloc = allocator, .ptr = slice };
    const thread = std.Thread.spawn(.{}, struct {
        fn run(c: SpawnCtx) void {
            if (is_slice_of_slices) {
                for (c.ptr) |inner| c.alloc.free(inner);
            }
            c.alloc.free(c.ptr);
        }
    }.run, .{ctx}) catch {
        // Fallback: free synchronously if spawn fails
        if (is_slice_of_slices) {
            for (slice) |inner| allocator.free(inner);
        }
        allocator.free(slice);
        return;
    };
    thread.detach();
}


// Bytecode entry construction — extracted to bytecode_entries.zig
pub const bytecode_entry_mod = @import("bytecode_entries.zig");
pub const BytecodeEntry = bytecode_entry_mod.BytecodeEntry;
pub const buildBytecodeEntries = bytecode_entry_mod.buildBytecodeEntries;
const hasLookupTable = bytecode_entry_mod.hasLookupTable;
const getLookupTableIndex = bytecode_entry_mod.getLookupTableIndex;


/// Result of Stage 6 sumcheck
pub fn Stage6Result(comptime F: type) type {
    return struct {
        const Self = @This();

        /// All sumcheck challenges (stage6_max_rounds elements)
        challenges: []F,

        /// BytecodeReadRaf opening claims: BytecodeRa(i) for i in 0..bytecode_d
        bytecode_ra_claims: []F,

        /// HammingBooleanity opening claim: RamHammingWeight
        hamming_weight_claim: F,

        /// Booleanity opening claims: all RA polys [InstructionRa(0..inst_d), BytecodeRa(0..bc_d), RamRa(0..ram_d)]
        booleanity_ra_claims: []F,

        /// RamRaVirtualization opening claims: RamRa(i) for i in 0..ram_d
        ram_ra_virtual_claims: []F,

        /// InstructionRaVirtualization opening claims: InstructionRa(i) for i in 0..instruction_d
        instruction_ra_virtual_claims: []F,

        /// IncClaimReduction opening claims: [RamInc, RdInc]
        ram_inc_claim: F,
        rd_inc_claim: F,

        /// Stage 6 configuration for Stage 7 opening point extraction
        bytecode_log_k: usize,
        log_k_chunk: usize,
        n_cycle_vars: usize,
        bytecode_d: usize,
        ram_d: usize,
        instruction_d: usize,

        allocator: Allocator,

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.challenges);
            self.allocator.free(self.bytecode_ra_claims);
            self.allocator.free(self.booleanity_ra_claims);
            self.allocator.free(self.ram_ra_virtual_claims);
            self.allocator.free(self.instruction_ra_virtual_claims);
        }
    };
}

// Instance provers extracted to stage6_instances.zig
const stage6_instances = @import("stage6_instances.zig");
pub const IncClaimReductionProver = stage6_instances.IncClaimReductionProver;
pub const HammingBooleanityProver = stage6_instances.HammingBooleanityProver;
pub const RamRaVirtualProver = stage6_instances.RamRaVirtualProver;
pub const BooleanityProver = stage6_instances.BooleanityProver;

// =============================================================================
// LookupsRaVirtual Sumcheck Instance (Instance 4)
// =============================================================================
// Proves: Sigma_c eq(r_cycle, c) * Sum_{v=0}^{N-1} gamma^v * Prod_{j=0}^{M-1} ra_{v*M+j}(c)
// Variables: n_cycle_vars
// Degree: M+1 (product of M linear ra polys * one linear eq)
//
// NOTE: Unlike RamRaVirtualProver, this uses dense []F arrays rather than RaPolynomial
// compression. The gamma scale is baked into the first poly of each virtual batch at
// init time (ra_bound[v*M] *= gamma^v), so the compressed representation would need
// per-index scaling, not a single shared eq_table scale. Future optimization could
// store separate gamma-scaled and unscaled eq_tables per batch.
/// Evaluate the product of 4 linear polynomials at the grid {1, 2, 3, ∞}.
/// Each pair[i] = { p_i(0), p_i(1) }. Returns { P(1), P(2), P(3), P(∞) }
/// where P(x) = p_0(x) * p_1(x) * p_2(x) * p_3(x).
/// Uses Toom-Cook factoring: 10 field muls total.
/// Ported from Jolt's eval_prod_4_assign (mles_product_sum.rs:453-462).
fn evalLinearProd4(comptime F: type, pairs: [4][2]F) [4]F {
    // eval_linear_prod_2_internal on first pair (p[0], p[1]):
    // For linear poly p(x) = p0 + (p1-p0)*x: p(1)=p1, p(∞)=p1-p0, p(2)=p1+(p1-p0)
    const p0_inf = pairs[0][1].sub(pairs[0][0]); // slope of p0
    const p1_inf = pairs[1][1].sub(pairs[1][0]); // slope of p1
    const a1 = pairs[0][1].mul(pairs[1][1]); // A(1) = p0(1)*p1(1)
    const a2 = p0_inf.add(pairs[0][1]).mul(p1_inf.add(pairs[1][1])); // A(2) = p0(2)*p1(2)
    const a_inf = p0_inf.mul(p1_inf); // A(∞) = leading coeff

    // ex2 extrapolation: A(3) = 2*(A(2) + A(∞)) - A(1)  (0 muls, pure adds)
    const a3 = a2.add(a_inf).add(a2.add(a_inf)).sub(a1);

    // eval_linear_prod_2_internal on second pair (p[2], p[3]):
    const p2_inf = pairs[2][1].sub(pairs[2][0]);
    const p3_inf = pairs[3][1].sub(pairs[3][0]);
    const b1 = pairs[2][1].mul(pairs[3][1]);
    const b2 = p2_inf.add(pairs[2][1]).mul(p3_inf.add(pairs[3][1]));
    const b_inf = p2_inf.mul(p3_inf);
    const b3 = b2.add(b_inf).add(b2.add(b_inf)).sub(b1);

    // Pointwise multiply: 4 muls
    return .{
        a1.mul(b1), // P(1)
        a2.mul(b2), // P(2)
        a3.mul(b3), // P(3)
        a_inf.mul(b_inf), // P(∞)
    };
}

/// Reconstruct the full round polynomial from quotient evaluations.
/// Implements Jolt's finish_mles_product_sum_from_evals (mles_product_sum.rs:235-269).
///
/// sum_evals: quotient g(x)/eq(x, r_round) evaluated at [1, 2, 3, ∞]
/// claim: the sumcheck claim p(0) + p(1) for this round
/// gruen_eq: the split-eq polynomial (provides r_round = tau[current_index - 1])
///
/// Returns: monomial coefficients of the full round polynomial g(x) (degree M+1=5, 6 coefficients).
/// Caller owns returned slice.
fn finishMlesProductSum(
    comptime F: type,
    allocator: std.mem.Allocator,
    sum_evals: [4]F,
    claim: F,
    gruen_eq: *const poly_mod.GruenSplitEqPolynomial(F),
) ![]F {
    const UniPolyT = poly_mod.UniPoly(F);

    // 1. Get r_round from the split-eq's current tau value
    const r_round = gruen_eq.tau[gruen_eq.current_index - 1];
    const eq_at_0 = F.one().sub(r_round); // eq(0, r) = 1 - r
    const eq_at_1 = r_round; // eq(1, r) = r

    // 2. Recover quotient(0) from claim:
    //    claim = eq(0,r)*q(0) + eq(1,r)*q(1)
    //    q(0) = (claim - r * q(1)) / (1 - r)
    const q_at_0 = claim.sub(eq_at_1.mul(sum_evals[0])).mul(eq_at_0.inverse().?);

    // 3. Interpolate quotient poly from [q(0), q(1), q(2), q(3), q(∞)]
    //    fromEvalsToom handles grid [0, 1, ..., n-2, ∞] where last eval = leading coeff
    var toom_evals = [5]F{ q_at_0, sum_evals[0], sum_evals[1], sum_evals[2], sum_evals[3] };
    const quotient_coeffs = try UniPolyT.fromEvalsToom(allocator, &toom_evals);
    defer allocator.free(quotient_coeffs);

    // 4. Multiply back by eq(x, r_round) = (1-r) + (2r-1)*x
    //    This produces the full g(x) of degree d = len(quotient_coeffs)
    const constant_coeff = eq_at_0; // 1 - r
    const x_coeff = r_round.add(r_round).sub(F.one()); // 2r - 1
    var final_coeffs = try allocator.alloc(F, quotient_coeffs.len + 1);
    @memset(final_coeffs, F.zero());
    for (0..quotient_coeffs.len) |i| {
        final_coeffs[i] = final_coeffs[i].add(quotient_coeffs[i].mul(constant_coeff));
        final_coeffs[i + 1] = final_coeffs[i + 1].add(quotient_coeffs[i].mul(x_coeff));
    }

    return final_coeffs;
}

fn LookupsRaVirtualProver(comptime F: type) type {
    const RaPoly = ra_poly_mod.RaPolynomial(F);

    return struct {
        const Self = @This();

        /// In-place MLE bind (same as RamRaVirtualProver.bindSlice).
        fn bindSlice(arr: []F, h: usize, challenge: F) void {
            if (challenge.limbs[0] == 0 and challenge.limbs[1] == 0) {
                for (0..h) |j| {
                    arr[j] = arr[2 * j].add(arr[2 * j + 1].sub(arr[2 * j]).mulHiBigIntU128(challenge.limbs));
                }
            } else {
                for (0..h) |j| {
                    arr[j] = arr[2 * j].add(challenge.mul(arr[2 * j + 1].sub(arr[2 * j])));
                }
            }
        }

        /// Compressed RA polynomials (lazy materialization through round1→round2→round3→dense)
        ra_polys: []RaPoly,
        /// GruenSplitEq for eq(r_cycle, .) — O(1) bind
        gruen_eq: poly_mod.GruenSplitEqPolynomial(F),
        M: usize,
        N: usize,
        total_committed: usize,
        current_len: usize,
        allocator: Allocator,
        pool: ?*ThreadPool = null,
        gpu: ?*GpuPolyOps = null,

        pub fn init(
            allocator: Allocator,
            trace: *const ExecutionTrace,
            r_cycle: []const F, // BIG_ENDIAN
            r_addr_chunks: []const []const F, // r_addr_chunks[i] for each committed poly
            gamma_powers: []const F, // gamma^v for v in 0..N
            M: usize,
            N: usize,
            log_k_chunk: usize,
            instruction_d: usize,
            init_pool: ?*ThreadPool,
        ) !Self {
            std.debug.assert(log_k_chunk <= ra_poly_mod.MAX_LOG_K_CHUNK);
            const T = trace.steps.items.len;
            const total_committed = M * N;
            const k_chunk: usize = @as(usize, 1) << @intCast(log_k_chunk);

            // Build RaPolynomials with compressed u8 indices + small eq tables
            var ra_polys_arr = try allocator.alloc(RaPoly, total_committed);
            errdefer allocator.free(ra_polys_arr);

            // Pre-allocate index arrays for all committed polys
            var indices_arr = try allocator.alloc([]?u8, total_committed);
            defer allocator.free(indices_arr);

            for (0..total_committed) |i| {
                indices_arr[i] = try allocator.alloc(?u8, T);
            }

            // Parallel fill: compute ALL index arrays in a single pass over trace.
            // For each step j, compute lookup_index ONCE, then extract all total_committed chunks.
            // This is ~32x fewer computeLookupIndex calls vs the old per-poly approach.
            const LkRaInitCtx = struct {
                steps: []const tracer.TraceStep,
                indices: [][]?u8,
                log_k_chunk: usize,
                k_chunk: usize,
                instruction_d: usize,
                total_committed: usize,
            };
            const lk_ra_ctx = LkRaInitCtx{
                .steps = trace.steps.items,
                .indices = indices_arr,
                .log_k_chunk = log_k_chunk,
                .k_chunk = k_chunk,
                .instruction_d = instruction_d,
                .total_committed = total_committed,
            };
            const lkRaInitFn = struct {
                fn f(c: LkRaInitCtx, j: usize) void {
                    const step = c.steps[j];
                    const lookup_index = computeLookupIndex(step);
                    const mask: u128 = (@as(u128, 1) << @intCast(c.log_k_chunk)) - 1;
                    for (0..c.total_committed) |i| {
                        // MSB-first: shift = log_k_chunk * (instruction_d - 1 - i)
                        const shift_amount = c.log_k_chunk * (c.instruction_d - 1 - i);
                        const chunk_val: usize = if (shift_amount < 128) @intCast((lookup_index >> @intCast(shift_amount)) & mask) else 0;
                        c.indices[i][j] = if (chunk_val < c.k_chunk) @intCast(chunk_val) else null;
                    }
                }
            }.f;
            pool_helpers.parallelForOptional(init_pool, T, lk_ra_ctx, lkRaInitFn);

            // Build eq tables and create RaPolynomials
            for (0..total_committed) |i| {
                var r_chunk_rev = try allocator.alloc(F, log_k_chunk);
                defer allocator.free(r_chunk_rev);
                for (0..log_k_chunk) |ci| r_chunk_rev[ci] = r_addr_chunks[i][log_k_chunk - 1 - ci];
                const eq_table = try computeEqTable(F, allocator, r_chunk_rev, log_k_chunk);

                const virtual_batch = i / M;
                const is_first_in_batch = (i % M == 0);
                const gamma_scale = if (is_first_in_batch) gamma_powers[virtual_batch] else F.one();

                // initRound1 takes ownership of indices and eq_table, prescales by gamma_scale
                ra_polys_arr[i] = RaPoly.initRound1(indices_arr[i], eq_table, gamma_scale);
            }

            // r_cycle is in BE order; pass directly to GruenSplitEq (same as Stage 3)
            const n_vars = std.math.log2_int(usize, T);
            const gruen_eq = try poly_mod.GruenSplitEqPolynomial(F).init(allocator, r_cycle[0..n_vars]);

            return Self{
                .ra_polys = ra_polys_arr,
                .gruen_eq = gruen_eq,
                .M = M,
                .N = N,
                .total_committed = total_committed,
                .current_len = T,
                .allocator = allocator,
                .pool = init_pool,
            };
        }

        pub fn deinit(self: *Self) void {
            for (self.ra_polys) |*p| p.deinit(self.allocator);
            self.allocator.free(self.ra_polys);
            self.gruen_eq.deinit();
        }

        /// f(x) = eq(x,r) * Sum_v Prod_{j=0}^{M-1} ra_{v*M+j}(x)
        /// Uses quotient polynomial approach: compute q(x) = f(x)/eq(x,r) at Toom points {1,2,3,∞}
        /// via evalLinearProd4, then reconstruct f(x) via finishMlesProductSumFromEvals.
        /// Returns monomial coefficients (degree M+1, i.e. 6 coefficients for M=4).
        pub fn computeRoundPoly(self: *Self, allocator: Allocator, claim: F) ![]F {
            const half = self.current_len / 2;

            // Get factored eq tables from GruenSplitEq (same pattern as Stage 3)
            const eq_tables = self.gruen_eq.getWindowEqTables(self.gruen_eq.current_index, 1);
            const E_out = eq_tables.E_out;
            const E_in = eq_tables.E_in;
            const head_in_bits = eq_tables.head_in_bits;
            const in_mask = (@as(usize, 1) << @intCast(head_in_bits)) -| 1;



            const Ctx = struct {
                ra_polys: []RaPoly,
                E_out: []const F,
                E_in: []const F,
                in_mask: usize,
                head_in_bits: usize,
                M: usize,
                N: usize,
            };
            const ctx = Ctx{
                .ra_polys = self.ra_polys,
                .E_out = E_out,
                .E_in = E_in,
                .in_mask = in_mask,
                .head_in_bits = head_in_bits,
                .M = self.M,
                .N = self.N,
            };

            // Compute quotient q(x) = Σ_j E_prefix(j) * Σ_v Π_k ra_{v*M+k}(x)
            // at 4 Toom points {1, 2, 3, ∞}
            // Uses deferred E_out pattern: accumulate E_in*val as unreduced within each x_out
            // block, then reduce and scale by E_out once per block (saves 1 mul per j).
            const UPA = UnreducedProductAccum;
            const mapFn = struct {
                fn f(c: Ctx, start: usize, end: usize) [4]F {
                    var outer_acc: [4]UPA = .{UPA.zero()} ** 4;
                    var inner_acc: [4]UPA = .{UPA.zero()} ** 4;
                    var prev_x_out: usize = if (start > 0) start >> @intCast(c.head_in_bits) else 0;
                    var started = false;

                    for (start..end) |j| {
                        const x_out = j >> @intCast(c.head_in_bits);
                        const x_in = j & c.in_mask;

                        // Flush inner_acc when x_out changes
                        if (started and x_out != prev_x_out) {
                            const e_out = if (prev_x_out < c.E_out.len) c.E_out[prev_x_out] else F.one();
                            for (0..4) |k| {
                                outer_acc[k].addAssign(e_out.mulToProductAccum(inner_acc[k].reduce()));
                                inner_acc[k] = UPA.zero();
                            }
                        }
                        prev_x_out = x_out;
                        started = true;

                        const e_in = if (x_in < c.E_in.len) c.E_in[x_in] else F.one();

                        // Accumulate sum of products across all virtual batches
                        var virtual_sum = [4]F{ F.zero(), F.zero(), F.zero(), F.zero() };
                        for (0..c.N) |v| {
                            var pairs: [4][2]F = undefined;
                            for (0..c.M) |m_idx| {
                                const idx = v * c.M + m_idx;
                                pairs[m_idx] = .{
                                    c.ra_polys[idx].getBoundCoeff(2 * j),
                                    c.ra_polys[idx].getBoundCoeff(2 * j + 1),
                                };
                            }
                            const prod_evals = evalLinearProd4(F, pairs);
                            for (0..4) |k| virtual_sum[k] = virtual_sum[k].add(prod_evals[k]);
                        }

                        // Accumulate with E_in only (defer E_out to block boundary)
                        for (0..4) |k| {
                            inner_acc[k].addAssign(e_in.mulToProductAccum(virtual_sum[k]));
                        }
                    }
                    // Flush final block
                    if (started) {
                        const e_out = if (prev_x_out < c.E_out.len) c.E_out[prev_x_out] else F.one();
                        for (0..4) |k| {
                            outer_acc[k].addAssign(e_out.mulToProductAccum(inner_acc[k].reduce()));
                        }
                    }
                    return .{ outer_acc[0].reduce(), outer_acc[1].reduce(), outer_acc[2].reduce(), outer_acc[3].reduce() };
                }
            }.f;

            const reduceFn = struct {
                fn f(a: [4]F, b: [4]F) [4]F {
                    return .{ a[0].add(b[0]), a[1].add(b[1]), a[2].add(b[2]), a[3].add(b[3]) };
                }
            }.f;

            const sum_evals = if (self.pool) |pool|
                pool.parallelReduce([4]F, half, .{F.zero()} ** 4, ctx, mapFn, reduceFn)
            else
                mapFn(ctx, 0, half);

            // Scale by current_scalar (accumulated eq from all previously bound variables)
            // The inner loop computed quotient' = Σ E_out*E_in*product (without current_scalar).
            // The actual quotient = current_scalar * quotient'.
            const scalar = self.gruen_eq.current_scalar;
            var scaled_evals = [4]F{
                sum_evals[0].mul(scalar),
                sum_evals[1].mul(scalar),
                sum_evals[2].mul(scalar),
                sum_evals[3].mul(scalar),
            };

            // Extract r_round from gruen_eq and reconstruct full polynomial
            const r_round = self.gruen_eq.tau[self.gruen_eq.current_index - 1];
            return poly_mod.UniPoly(F).finishMlesProductSumFromEvals(allocator, &scaled_evals, claim, r_round);
        }

        pub fn bindChallenge(self: *Self, r: F) !void {
            const half = self.current_len / 2;

            // Bind RA polynomials — O(K) for compressed states, O(T/2^round) for dense
            const all_dense = self.ra_polys[0].isDense();
            if (all_dense and self.gpu != null and half >= 16384) {
                // GPU bind: total_committed ra_poly dense arrays
                const gpu = self.gpu.?;
                for (self.ra_polys) |*rp| {
                    const dense = &rp.dense;
                    const h = dense.current_len / 2;
                    gpu.polyBindLow(dense.coeffs[0 .. h * 2], r, dense.coeffs[0..h]) catch {
                        for (0..h) |jj| {
                            dense.coeffs[jj] = dense.coeffs[2 * jj].add(r.mul(dense.coeffs[2 * jj + 1].sub(dense.coeffs[2 * jj])));
                        }
                    };
                    dense.current_len = h;
                }
            } else if (all_dense) {
                if (self.pool) |pool| {
                    const BindCtx = struct { ra: []RaPoly, tc: usize, half: usize, r: F };
                    const ctx = BindCtx{ .ra = self.ra_polys, .tc = self.total_committed, .half = half, .r = r };
                    pool.parallelForForce(self.total_committed, ctx, struct {
                        fn f(c: BindCtx, idx: usize) void {
                            const dense = &c.ra[idx].dense;
                            const h = dense.current_len / 2;
                            for (0..h) |jj| {
                                dense.coeffs[jj] = dense.coeffs[2 * jj].add(c.r.mul(dense.coeffs[2 * jj + 1].sub(dense.coeffs[2 * jj])));
                            }
                            dense.current_len = h;
                        }
                    }.f);
                } else {
                    for (self.ra_polys) |*p| try p.bind(r, self.allocator);
                }
            } else {
                for (self.ra_polys) |*p| try p.bind(r, self.allocator);
            }

            // GruenSplitEq bind — O(1) instead of O(T/2^round)
            self.gruen_eq.bind(r);

            self.current_len = half;
        }

        pub fn getOpeningClaims(self: *const Self, allocator: Allocator, gamma_powers: []const F) ![]F {
            var claims = try allocator.alloc(F, self.total_committed);
            for (0..self.total_committed) |i| {
                var claim = self.ra_polys[i].finalClaim();
                // Undo gamma pre-scaling for first poly in each batch
                const is_first_in_batch = (i % self.M == 0);
                if (is_first_in_batch) {
                    const virtual_batch = i / self.M;
                    claim = claim.mul(gamma_powers[virtual_batch].inverse().?);
                }
                claims[i] = claim;
            }
            return claims;
        }
    };
}

// =============================================================================
// BytecodeReadRaf Sumcheck Instance (Instance 0)
// =============================================================================
// Most complex instance. Two phases:
// Phase 1: Address binding (bytecode_log_k rounds)
//   Polynomial: H(k) = Sum_s gamma^s * F_s[k] * (Val_s(k) + RAF_s(k))
//   where F_s[k] = Sum_c eq(r_cycle_s, c) * delta(PC(c)=k)
//   Both F_s and Val are linear in the bound address variable, so the product
//   gives a DEGREE 2 round polynomial.
//
// Phase 2: Cycle binding (n_cycle_vars rounds)
//   After binding address to r_addr, polynomial becomes:
//   f(c) = [Prod_i ra_chunk_i(c)] * [Sum_s gamma^s * bound_val_s * eq_s(c)]
//   Degree = bytecode_d + 1
fn BytecodeReadRafProver(comptime F: type) type {
    return struct {
        const Self = @This();

        /// Phase 1: Separate F_s and val_with_raf arrays per stage
        /// F_s_arrs[s][k] = Sum_c:PC(c)=k eq(r_cycle_s, c)
        F_s_arrs: [5][]F,
        /// val_with_raf[s][k] = Val_s(k) + RAF_s(k)
        val_with_raf: [5][]F,
        /// Per-stage running claims for Phase 1
        stage_claims: [5]F,

        /// Phase 2: 5 GruenSplitEq instances for per-stage eq(r_cycle_s, .)
        /// Replaces flat combined[] with O(1) bind per round.
        stage_gruen_eqs: [5]?poly_mod.GruenSplitEqPolynomial(F),
        /// Phase 2: per-stage bound_vals (gamma^s * val_with_raf[s][0])
        bound_vals_phase2: [5]F,
        /// Phase 2: entry correction scalar, starts at entry_gamma * bound_f_entry,
        /// multiplied by (1-r) each bind round (tracks eq(0...0, challenges))
        entry_correction_scalar: F,

        /// Phase 2: RA chunk polynomials ra_chunks[i][c]
        ra_chunks: ?[][]F,

        /// Phase tracking
        phase: u8,
        bytecode_log_k: usize,
        n_cycle_vars: usize,
        bytecode_d: usize,
        log_k_chunk: usize,
        current_len: usize,
        addr_rounds_done: usize,

        /// Stored from Phase 1→2 transition for diagnostics
        bound_vals_stored: [5]F,
        /// F_s[0] values saved before freeing F_s_arrs (for Phase 2 consistency check)
        f_s_bound_saved: [5]F,

        /// Data needed for phase transition
        trace: *const ExecutionTrace,
        pc_map: *const BytecodePCMapper,
        stage_r_cycles: [5][]const F,
        gamma_powers: [8]F,
        /// Val polynomials per stage: val_polys[s][k]
        val_polys: [5][]F,
        /// Identity polynomial: int_poly[k] = k as field element
        int_poly: []F,

        entry_gamma: F,
        entry_val: F,
        entry_ri: usize,
        bound_f_entry: F,
        eq_zero_scalar: F,

        allocator: Allocator,
        pool: ?*ThreadPool = null,
        gpu: ?*GpuPolyOps = null,

        pub fn init(
            allocator: Allocator,
            trace: *const ExecutionTrace,
            pc_map: *const BytecodePCMapper,
            val_polys: [5][]F, // Val_s(k) for each stage, length bytecode_K each
            bytecode_log_k: usize,
            n_cycle_vars: usize,
            bytecode_d: usize,
            log_k_chunk: usize,
            gamma_powers: [8]F,
            stage_r_cycles: [5][]const F,
            int_poly: []F,
            external_stage_claims: [5]F, // From opening claims: claim_per_stage[s]
            entry_bytecode_index: usize,
            init_pool: ?*ThreadPool,
        ) !Self {
            const bytecode_K: usize = @as(usize, 1) << @intCast(bytecode_log_k);

            // Phase 1: Build separate F_s and val_with_raf arrays per stage
            var F_s_arrs: [5][]F = undefined;
            var val_with_raf_arrs: [5][]F = undefined;
            var stage_claims_init: [5]F = undefined;

            // Split-eq F_s computation: replaces T-sized eq tables with sqrt(T)-sized E_lo/E_hi
            // F_s[s][k] = Σ_c eq(r_cycle_s, c) * δ(PC(c)=k)
            //           = Σ_{c_hi} E_hi[c_hi] * (Σ_{c_lo: PC(c)=k} E_lo[c_lo])
            // Inner loop over c_lo is additions only; one mul per touched PC per c_hi block.
            const lo_bits = n_cycle_vars / 2;
            const hi_bits = n_cycle_vars - lo_bits;
            const in_len: usize = @as(usize, 1) << @intCast(lo_bits);
            const out_len: usize = @as(usize, 1) << @intCast(hi_bits);

            // Compute all 5 stages' E_lo and E_hi tables, then run all 5 double-loops.
            // Each stage has its own buffers to enable parallel execution.
            var E_lo_arr: [5][]F = undefined;
            var E_hi_arr: [5][]F = undefined;

            for (0..5) |s| {
                var r_cycle_rev_s = try allocator.alloc(F, n_cycle_vars);
                defer allocator.free(r_cycle_rev_s);
                for (0..n_cycle_vars) |i| {
                    r_cycle_rev_s[i] = stage_r_cycles[s][n_cycle_vars - 1 - i];
                }
                E_lo_arr[s] = try computeEqTableParallel(F, allocator, r_cycle_rev_s[0..lo_bits], lo_bits, init_pool);
                E_hi_arr[s] = try computeEqTableParallel(F, allocator, r_cycle_rev_s[lo_bits..n_cycle_vars], hi_bits, init_pool);
            }
            defer for (0..5) |s| {
                allocator.free(E_lo_arr[s]);
                allocator.free(E_hi_arr[s]);
            };

            // Allocate F_s output and per-stage temp buffers
            for (0..5) |s| {
                F_s_arrs[s] = try allocator.alloc(F, bytecode_K);
                @memset(F_s_arrs[s], F.zero());
            }

            // Run all 5 stages' double-loops in parallel (each stage independent)
            // Pre-allocate per-stage heap buffers (bytecode_K can be >256 for large programs)
            var per_stage_inner: [5][]F = undefined;
            var per_stage_touched: [5][]usize = undefined;
            var per_stage_tset: [5][]bool = undefined;
            for (0..5) |s| {
                per_stage_inner[s] = try allocator.alloc(F, bytecode_K);
                @memset(per_stage_inner[s], F.zero());
                per_stage_touched[s] = try allocator.alloc(usize, bytecode_K);
                per_stage_tset[s] = try allocator.alloc(bool, bytecode_K);
                @memset(per_stage_tset[s], false);
            }
            defer for (0..5) |s| {
                allocator.free(per_stage_inner[s]);
                allocator.free(per_stage_touched[s]);
                allocator.free(per_stage_tset[s]);
            };

            if (init_pool) |pool| {
                const FsCtx = struct {
                    F_s_out: *[5][]F,
                    E_lo_a: *[5][]F,
                    E_hi_a: *[5][]F,
                    steps: []const tracer.TraceStep,
                    pc_map_ptr: *const BytecodePCMapper,
                    in_len: usize,
                    out_len: usize,
                    lo_bits: usize,
                    bK: usize,
                    inner_bufs: *[5][]F,
                    touched_bufs: *[5][]usize,
                    tset_bufs: *[5][]bool,
                };
                const fs_ctx = FsCtx{
                    .F_s_out = &F_s_arrs,
                    .E_lo_a = &E_lo_arr,
                    .E_hi_a = &E_hi_arr,
                    .steps = trace.steps.items,
                    .pc_map_ptr = pc_map,
                    .in_len = in_len,
                    .out_len = out_len,
                    .lo_bits = lo_bits,
                    .bK = bytecode_K,
                    .inner_bufs = &per_stage_inner,
                    .touched_bufs = &per_stage_touched,
                    .tset_bufs = &per_stage_tset,
                };
                pool.parallelForForce(5, fs_ctx, struct {
                    fn f(c: FsCtx, s: usize) void {
                        const E_lo = c.E_lo_a[s];
                        const E_hi = c.E_hi_a[s];
                        const F_s = c.F_s_out[s];
                        const inner_buf = c.inner_bufs[s];
                        const touched_buf = c.touched_bufs[s];
                        const touched_set = c.tset_bufs[s];

                        for (0..c.out_len) |c_hi| {
                            var touched_count: usize = 0;
                            for (0..c.in_len) |c_lo| {
                                const idx = c_lo + (c_hi << @intCast(c.lo_bits));
                                const step = c.steps[idx];
                                const pc_idx = c.pc_map_ptr.getPCForStep(step);
                                if (pc_idx < c.bK) {
                                    if (!touched_set[pc_idx]) {
                                        touched_set[pc_idx] = true;
                                        touched_buf[touched_count] = pc_idx;
                                        touched_count += 1;
                                    }
                                    inner_buf[pc_idx] = inner_buf[pc_idx].add(E_lo[c_lo]);
                                }
                            }
                            const e_hi_val = E_hi[c_hi];
                            for (0..touched_count) |ti| {
                                const pc = touched_buf[ti];
                                F_s[pc] = F_s[pc].add(e_hi_val.mul(inner_buf[pc]));
                                inner_buf[pc] = F.zero();
                                touched_set[pc] = false;
                            }
                        }
                    }
                }.f);
            } else {
                // Sequential fallback
                var inner_buf = try allocator.alloc(F, bytecode_K);
                defer allocator.free(inner_buf);
                var touched_buf = try allocator.alloc(usize, bytecode_K);
                defer allocator.free(touched_buf);
                var touched_set = try allocator.alloc(bool, bytecode_K);
                defer allocator.free(touched_set);

                for (0..5) |s| {
                    @memset(inner_buf, F.zero());
                    @memset(touched_set, false);

                    for (0..out_len) |c_hi| {
                        var touched_count: usize = 0;
                        for (0..in_len) |c_lo| {
                            const c = c_lo + (c_hi << @intCast(lo_bits));
                            const step = trace.steps.items[c];
                            const pc_idx = pc_map.getPCForStep(step);
                            if (pc_idx < bytecode_K) {
                                if (!touched_set[pc_idx]) {
                                    touched_set[pc_idx] = true;
                                    touched_buf[touched_count] = pc_idx;
                                    touched_count += 1;
                                }
                                inner_buf[pc_idx] = inner_buf[pc_idx].add(E_lo_arr[s][c_lo]);
                            }
                        }
                        const e_hi_val = E_hi_arr[s][c_hi];
                        for (0..touched_count) |ti| {
                            const pc = touched_buf[ti];
                            F_s_arrs[s][pc] = F_s_arrs[s][pc].add(e_hi_val.mul(inner_buf[pc]));
                            inner_buf[pc] = F.zero();
                            touched_set[pc] = false;
                        }
                    }
                }
            }

            // Build val_with_raf and compute claims for each stage
            for (0..5) |s| {
                // val_with_raf[s][k] = Val_s(k) + RAF_s(k)
                val_with_raf_arrs[s] = try allocator.alloc(F, bytecode_K);
                for (0..bytecode_K) |k| {
                    var val_plus_raf = if (val_polys[s].len > k) val_polys[s][k] else F.zero();
                    // RAF terms
                    if (s == 0) {
                        val_plus_raf = val_plus_raf.add(gamma_powers[5].mul(int_poly[k]));
                    } else if (s == 2) {
                        val_plus_raf = val_plus_raf.add(gamma_powers[4].mul(int_poly[k]));
                    }
                    val_with_raf_arrs[s][k] = val_plus_raf;
                }

                // Compute claim from val_polys and F_s
                var recomputed_claim = F.zero();
                for (0..bytecode_K) |k| {
                    recomputed_claim = recomputed_claim.add(F_s_arrs[s][k].mul(val_with_raf_arrs[s][k]));
                }

                // Use val_poly-derived claims for sumcheck consistency
                // The sumcheck polynomial must sum to the claimed value,
                // and the polynomial is built from val_polys and F_s.
                // If we use external claims that differ from the actual polynomial sum,
                // the sumcheck will be inconsistent.
                stage_claims_init[s] = recomputed_claim;

                // Debug: Check if recomputed matches external
                if (comptime debug_verbose) {
                    const match_ext = @as(u8, if (recomputed_claim.eql(external_stage_claims[s])) 1 else 0);
                    if (match_ext == 0) {
                        const rc_full = recomputed_claim.toBytesBE();
                        const ec_full = external_stage_claims[s].toBytesBE();
                        dbg("[BCRAF_MISMATCH] Stage {d}: recomputed != external!\n", .{s});
                        dbg("  recomputed_LE=[", .{});
                        for (0..32) |bi| dbg("{x:0>2}", .{rc_full[31 - bi]});
                        dbg("]\n  external_LE=[", .{});
                        for (0..32) |bi| dbg("{x:0>2}", .{ec_full[31 - bi]});
                        dbg("]\n", .{});
                    }
                }
            }

            // Debug: print per-stage claims with full aggregation detail
            if (comptime debug_verbose) {
                var total = F.zero();
                for (0..5) |s| {
                    const term = gamma_powers[s].mul(stage_claims_init[s]);
                    total = total.add(term);
                    const sc_le = stage_claims_init[s].toBytes();
                    const gp_le = gamma_powers[s].toBytes();
                    const tm_le = term.toBytes();
                    dbg("[BCRAF_AGG_PR] s={} sc==ext={}", .{
                        s, @as(u8, if (stage_claims_init[s].eql(external_stage_claims[s])) 1 else 0),
                    });
                    dbg(" gp=[{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}]", .{
                        gp_le[0], gp_le[1], gp_le[2], gp_le[3], gp_le[4], gp_le[5], gp_le[6], gp_le[7],
                    });
                    dbg(" sc=[{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}]", .{
                        sc_le[0], sc_le[1], sc_le[2], sc_le[3], sc_le[4], sc_le[5], sc_le[6], sc_le[7],
                    });
                    dbg(" term=[{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}]", .{
                        tm_le[0], tm_le[1], tm_le[2], tm_le[3], tm_le[4], tm_le[5], tm_le[6], tm_le[7],
                    });
                    dbg("\n", .{});
                }
                const tl = total.toBytes();
                dbg("[BCRAF_AGG_PR] total_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{tl[0], tl[1], tl[2], tl[3], tl[4], tl[5], tl[6], tl[7]});
            }

            return Self{
                .F_s_arrs = F_s_arrs,
                .val_with_raf = val_with_raf_arrs,
                .stage_claims = stage_claims_init,
                .stage_gruen_eqs = [_]?poly_mod.GruenSplitEqPolynomial(F){null} ** 5,
                .bound_vals_phase2 = [_]F{F.zero()} ** 5,
                .entry_correction_scalar = F.zero(),
                .ra_chunks = null,
                .phase = 0,
                .bytecode_log_k = bytecode_log_k,
                .n_cycle_vars = n_cycle_vars,
                .bytecode_d = bytecode_d,
                .log_k_chunk = log_k_chunk,
                .current_len = bytecode_K,
                .addr_rounds_done = 0,
                .bound_vals_stored = [_]F{F.zero()} ** 5,
                .f_s_bound_saved = [_]F{F.zero()} ** 5,
                .trace = trace,
                .pc_map = pc_map,
                .stage_r_cycles = stage_r_cycles,
                .gamma_powers = gamma_powers,
                .val_polys = val_polys,
                .int_poly = int_poly,
                .entry_gamma = gamma_powers[7],
                .entry_val = F.one(),
                .entry_ri = entry_bytecode_index,
                .bound_f_entry = F.zero(),
                .eq_zero_scalar = F.one(),
                .allocator = allocator,
                .pool = init_pool,
            };
        }

        pub fn deinit(self: *Self) void {
            // Phase 1 arrays (freed during transition if we got that far)
            for (0..5) |s| {
                if (self.F_s_arrs[s].len > 0) self.allocator.free(self.F_s_arrs[s]);
                if (self.val_with_raf[s].len > 0) self.allocator.free(self.val_with_raf[s]);
            }
            // Phase 2: GruenSplitEq instances
            for (0..5) |s| {
                if (self.stage_gruen_eqs[s]) |*g| {
                    var gruen = g.*;
                    gruen.deinit();
                }
            }
            if (self.ra_chunks) |chunks| {
                for (chunks) |arr| self.allocator.free(arr);
                self.allocator.free(chunks);
            }
            for (&self.val_polys) |vp| {
                if (vp.len > 0) self.allocator.free(vp);
            }
            self.allocator.free(self.int_poly);
        }

        /// Phase 1: degree-2 round poly over address vars
        /// Returns .{ agg=[p(0), p(2)], per_stage=[5][eval_0, eval_2] }
        /// Matches Jolt's approach: product of F_s (linear) and val_with_raf (linear) = degree 2
        pub fn computeRoundPolyPhase1(self: *Self) struct { agg: [2]F, per_stage: [5][2]F } {
            const half = self.current_len / 2;
            var per_stage: [5][2]F = undefined;

            if (self.pool) |pool| {
                // Compute 5 stages in parallel, each accumulating [2]F
                const Ctx = struct {
                    F_s_arrs: *const [5][]F,
                    val_with_raf: *const [5][]F,
                    half: usize,
                    results: *[5][2]F,
                };
                const ctx = Ctx{
                    .F_s_arrs = &self.F_s_arrs,
                    .val_with_raf = &self.val_with_raf,
                    .half = half,
                    .results = &per_stage,
                };
                pool.parallelForForce(5, ctx, struct {
                    fn f(c: Ctx, s: usize) void {
                        var eval_at_0 = F.zero();
                        var eval_at_2 = F.zero();
                        for (0..c.half) |k| {
                            const f_lo = c.F_s_arrs[s][2 * k];
                            const f_hi = c.F_s_arrs[s][2 * k + 1];
                            const v_lo = c.val_with_raf[s][2 * k];
                            const v_hi = c.val_with_raf[s][2 * k + 1];

                            eval_at_0 = eval_at_0.add(f_lo.mul(v_lo));

                            const f_at_2 = f_hi.add(f_hi).sub(f_lo);
                            const v_at_2 = v_hi.add(v_hi).sub(v_lo);
                            eval_at_2 = eval_at_2.add(f_at_2.mul(v_at_2));
                        }
                        c.results[s] = [2]F{ eval_at_0, eval_at_2 };
                    }
                }.f);
            } else {
                for (0..5) |s| {
                    var eval_at_0 = F.zero();
                    var eval_at_2 = F.zero();

                    for (0..half) |k| {
                        const f_lo = self.F_s_arrs[s][2 * k];
                        const f_hi = self.F_s_arrs[s][2 * k + 1];
                        const v_lo = self.val_with_raf[s][2 * k];
                        const v_hi = self.val_with_raf[s][2 * k + 1];

                        eval_at_0 = eval_at_0.add(f_lo.mul(v_lo));

                        const f_at_2 = f_hi.add(f_hi).sub(f_lo);
                        const v_at_2 = v_hi.add(v_hi).sub(v_lo);
                        eval_at_2 = eval_at_2.add(f_at_2.mul(v_at_2));
                    }

                    per_stage[s] = [2]F{ eval_at_0, eval_at_2 };
                }
            }

            var agg_eval_0 = F.zero();
            var agg_eval_2 = F.zero();
            for (0..5) |s| {
                agg_eval_0 = agg_eval_0.add(self.gamma_powers[s].mul(per_stage[s][0]));
                agg_eval_2 = agg_eval_2.add(self.gamma_powers[s].mul(per_stage[s][1]));
            }

            const entry_bit = self.entry_ri & 1;
            const ev_sq = self.entry_val.mul(self.entry_val);
            const eg_ev = self.entry_gamma.mul(ev_sq);
            if (entry_bit == 0) {
                agg_eval_0 = agg_eval_0.add(eg_ev);
                agg_eval_2 = agg_eval_2.add(eg_ev);
            } else {
                agg_eval_2 = agg_eval_2.add(eg_ev.add(eg_ev).add(eg_ev).add(eg_ev));
            }

            return .{ .agg = [2]F{ agg_eval_0, agg_eval_2 }, .per_stage = per_stage };
        }

        /// Bind challenge and update per-stage claims from polynomial evaluation
        /// per_stage_evals: [5][eval_0, eval_2] from computeRoundPolyPhase1
        pub fn bindChallengePhase1(self: *Self, r: F, per_stage_evals: [5][2]F) void {
            const half = self.current_len / 2;
            const two = F.fromU64(2);
            const two_inv = two.inverse().?;

            const bindStage = struct {
                fn f(
                    F_s: []F,
                    vwr: []F,
                    stage_claim: *F,
                    h: usize,
                    challenge: F,
                    pse: [2]F,
                    t_inv: F,
                ) void {
                    // Bind F_s and val_with_raf arrays
                    for (0..h) |k| {
                        F_s[k] = F_s[2 * k].add(challenge.mul(F_s[2 * k + 1].sub(F_s[2 * k])));
                        vwr[k] = vwr[2 * k].add(challenge.mul(vwr[2 * k + 1].sub(vwr[2 * k])));
                    }

                    // Update per-stage claim
                    const p0 = pse[0];
                    const p2 = pse[1];
                    const p1 = stage_claim.*.sub(p0);
                    const a0 = p0;
                    const a2 = p2.sub(p1.add(p1)).add(p0).mul(t_inv);
                    const a1 = p1.sub(p0).sub(a2);
                    stage_claim.* = a0.add(challenge.mul(a1.add(challenge.mul(a2))));
                }
            }.f;

            const bindOneArr = struct {
                fn f(arr: []F, h: usize, challenge: F) void {
                    for (0..h) |k| {
                        arr[k] = arr[2 * k].add(challenge.mul(arr[2 * k + 1].sub(arr[2 * k])));
                    }
                }
            }.f;

            const updateClaim = struct {
                fn f(stage_claim: *F, pse: [2]F, challenge: F, t_inv: F) void {
                    const p0 = pse[0];
                    const p2 = pse[1];
                    const p1 = stage_claim.*.sub(p0);
                    const a0 = p0;
                    const a2 = p2.sub(p1.add(p1)).add(p0).mul(t_inv);
                    const a1 = p1.sub(p0).sub(a2);
                    stage_claim.* = a0.add(challenge.mul(a1.add(challenge.mul(a2))));
                }
            }.f;

            if (self.gpu) |gpu| {
                if (half >= 16384) {
                    // GPU bind: 5 stages x 2 arrays each, then update claims on CPU
                    for (0..5) |s| {
                        gpu.polyBindLow(self.F_s_arrs[s][0 .. half * 2], r, self.F_s_arrs[s][0..half]) catch bindOneArr(self.F_s_arrs[s], half, r);
                        gpu.polyBindLow(self.val_with_raf[s][0 .. half * 2], r, self.val_with_raf[s][0..half]) catch bindOneArr(self.val_with_raf[s], half, r);
                        updateClaim(&self.stage_claims[s], per_stage_evals[s], r, two_inv);
                    }
                } else {
                    for (0..5) |s| {
                        bindStage(
                            self.F_s_arrs[s],
                            self.val_with_raf[s],
                            &self.stage_claims[s],
                            half,
                            r,
                            per_stage_evals[s],
                            two_inv,
                        );
                    }
                }
            } else if (self.pool) |pool| {
                const Ctx = struct {
                    F_s_arrs: *[5][]F,
                    val_with_raf: *[5][]F,
                    stage_claims: *[5]F,
                    half: usize,
                    r: F,
                    per_stage_evals: [5][2]F,
                    two_inv: F,
                };
                const ctx = Ctx{
                    .F_s_arrs = &self.F_s_arrs,
                    .val_with_raf = &self.val_with_raf,
                    .stage_claims = &self.stage_claims,
                    .half = half,
                    .r = r,
                    .per_stage_evals = per_stage_evals,
                    .two_inv = two_inv,
                };
                pool.parallelForForce(5, ctx, struct {
                    fn f(c: Ctx, s: usize) void {
                        bindStage(
                            c.F_s_arrs[s],
                            c.val_with_raf[s],
                            &c.stage_claims[s],
                            c.half,
                            c.r,
                            c.per_stage_evals[s],
                            c.two_inv,
                        );
                    }
                }.f);
            } else {
                for (0..5) |s| {
                    bindStage(
                        self.F_s_arrs[s],
                        self.val_with_raf[s],
                        &self.stage_claims[s],
                        half,
                        r,
                        per_stage_evals[s],
                        two_inv,
                    );
                }
            }


            const entry_bit = self.entry_ri & 1;
            if (entry_bit == 0) {
                self.entry_val = self.entry_val.mul(F.one().sub(r));
            } else {
                self.entry_val = self.entry_val.mul(r);
            }
            self.entry_ri >>= 1;

            self.current_len = half;
            self.addr_rounds_done += 1;
        }

        /// Transition from Phase 1 to Phase 2 after binding all address vars
        /// r_address_challenges are the challenges from Phase 1 in binding order (low-to-high)
        pub fn transitionToPhase2(
            self: *Self,
            r_address_challenges: []const F, // Low-to-high binding order from the sumcheck
        ) !void {
            const T: usize = @as(usize, 1) << @intCast(self.n_cycle_vars);
            const bytecode_K: usize = @as(usize, 1) << @intCast(self.bytecode_log_k);

            // The Phase 1 sumcheck binds variables in LowToHigh order:
            // r_address_challenges[0] = r_0 (bound to LSB of index), ..., [n-1] = r_{n-1} (MSB).
            //
            // For computing val_eval = Σ_k val[k] * eq(k, r), we need:
            // eq[k] = Π_j (r_j if bit j of k is 1, else 1-r_j).
            // Our computeEqTable with LE indexing gives exactly this when passed
            // the challenges in LH order.
            //
            // For RA chunk computation, Jolt uses r_address_BE = [r_{n-1},...,r_0]
            // and chunks sequentially: chunk 0 = MSB vars, chunk d-1 = LSB vars.
            // We keep r_address_be for RA chunk slicing (same convention as before).

            // Print address challenges (ALWAYS ON for debugging)
            {
                dbg("[BCRAF_TRANS] r_address_challenges (len={}, LH order):\n", .{self.bytecode_log_k});
                for (0..self.bytecode_log_k) |i| {
                    const ch_be = r_address_challenges[i].toBytesBE();
                    dbg("  ch[{d}]_LE=[", .{i});
                    for (0..32) |bi| dbg("{x:0>2}", .{ch_be[31 - bi]});
                    dbg("]\n", .{});
                }
            }

            // Compute r_address_be for RA chunk slicing (same as before)
            var r_address_be = try self.allocator.alloc(F, self.bytecode_log_k);
            defer self.allocator.free(r_address_be);
            for (0..self.bytecode_log_k) |i| {
                r_address_be[i] = r_address_challenges[self.bytecode_log_k - 1 - i];
            }

            // Compute bound_vals[s] = Val_s(r_address) + RAF_s(r_address)
            // The sumcheck binds variables MSB-first: r_address_challenges[0] = MSB.
            // But val_poly coefficients are indexed with bit 0 = LSB.
            // Jolt's verifier reverses challenges (normalize_opening_point) before evaluate,
            // so r[0] = LSB challenge maps to bit 0 of coefficient index.
            // Use r_address_be (reversed) for the eq table, matching Jolt's normalize_opening_point.
            const eq_addr = try computeEqTableParallel(F, self.allocator, r_address_be, self.bytecode_log_k, self.pool);
            defer self.allocator.free(eq_addr);

            // Debug: eq_addr entries
            if (comptime debug_verbose) {
                for (0..bytecode_K) |ek| {
                    const eab = eq_addr[ek].toBytesBE();
                    dbg("[ZOLT_EQ_ADDR] eq[{d}]_LE=[", .{ek});
                    for (0..32) |bi| dbg("{x:0>2}", .{eab[31 - bi]});
                    dbg("]\n", .{});
                }
            }

            // Debug: val_polys entries
            if (comptime debug_verbose) {
                for (0..5) |vs| {
                    for (0..bytecode_K) |kk| {
                        const vpk = self.val_polys[vs][kk].toBytesBE();
                        dbg("[ZOLT_VP] Val[{d}][{d}]_LE=[", .{ vs, kk });
                        for (0..8) |bi| dbg("{x:0>2}", .{vpk[31 - bi]});
                        dbg("]\n", .{});
                    }
                }
            }

            var bound_vals: [5]F = undefined;
            for (0..5) |s| {
                var val_eval = F.zero();
                const max_k = @min(self.val_polys[s].len, bytecode_K);
                for (0..max_k) |k| {
                    val_eval = val_eval.add(self.val_polys[s][k].mul(eq_addr[k]));
                }

                // Add RAF terms (identity polynomial contribution)
                // Stage 0: RAF = gamma^5 * identity_eval
                // Stage 2: RAF = gamma^4 * identity_eval
                if (s == 0) {
                    var identity_eval = F.zero();
                    for (0..bytecode_K) |k| {
                        identity_eval = identity_eval.add(self.int_poly[k].mul(eq_addr[k]));
                    }
                    const raf_contrib = self.gamma_powers[5].mul(identity_eval);
                    val_eval = val_eval.add(raf_contrib);
                    // Print identity_eval, gamma[5], RAF contribution, val_before_raf
                    const ie_be = identity_eval.toBytesBE();
                    const g5_be = self.gamma_powers[5].toBytesBE();
                    const rc_be = raf_contrib.toBytesBE();
                    dbg("[TRANS_RAF] s=0: identity_eval_LE=[", .{});
                    for (0..32) |bi| dbg("{x:0>2}", .{ie_be[31 - bi]});
                    dbg("] gamma5_LE=[", .{});
                    for (0..32) |bi| dbg("{x:0>2}", .{g5_be[31 - bi]});
                    dbg("] raf_LE=[", .{});
                    for (0..32) |bi| dbg("{x:0>2}", .{rc_be[31 - bi]});
                    dbg("]\n", .{});
                } else if (s == 2) {
                    var identity_eval = F.zero();
                    for (0..bytecode_K) |k| {
                        identity_eval = identity_eval.add(self.int_poly[k].mul(eq_addr[k]));
                    }
                    const raf_contrib = self.gamma_powers[4].mul(identity_eval);
                    val_eval = val_eval.add(raf_contrib);
                    const ie_be = identity_eval.toBytesBE();
                    const g4_be = self.gamma_powers[4].toBytesBE();
                    const rc_be = raf_contrib.toBytesBE();
                    dbg("[TRANS_RAF] s=2: identity_eval_LE=[", .{});
                    for (0..32) |bi| dbg("{x:0>2}", .{ie_be[31 - bi]});
                    dbg("] gamma4_LE=[", .{});
                    for (0..32) |bi| dbg("{x:0>2}", .{g4_be[31 - bi]});
                    dbg("] raf_LE=[", .{});
                    for (0..32) |bi| dbg("{x:0>2}", .{rc_be[31 - bi]});
                    dbg("]\n", .{});
                }

                // bound_vals[s] = gamma^s * val_with_raf[s][0]
                // After Phase 1 binding of all address variables, val_with_raf[s][0] is the
                // MLE evaluation at the binding point. stage_claims[s] = F_s[0]*val_with_raf[s][0]
                // since both are reduced to single elements.
                bound_vals[s] = self.gamma_powers[s].mul(self.val_with_raf[s][0]);
                self.bound_vals_stored[s] = bound_vals[s];

                // DIAGNOSTIC: compare re-computed val_eval with Phase 1 bound val_with_raf[s][0]
                if (comptime debug_verbose) {
                    const phase1_bound = self.val_with_raf[s][0];
                    dbg("[TRANS_CHECK] stage[{}]: match={}\n", .{ s, @as(u8, if (val_eval.eql(phase1_bound)) 1 else 0) });
                }

                // Debug: Print val_eval and bound_val for comparison with Jolt verifier
                if (comptime debug_verbose) {
                    const ve_be = val_eval.toBytesBE();
                    const bv_be = bound_vals[s].toBytesBE();
                    dbg("[BCRAF_TRANS] stage[{}]: val_eval_LE=[", .{s});
                    for (0..32) |bi| dbg("{x:0>2}", .{ve_be[31 - bi]});
                    dbg("] bound_val_LE=[", .{});
                    for (0..32) |bi| dbg("{x:0>2}", .{bv_be[31 - bi]});
                    dbg("]\n", .{});
                }
            }

            // Build RA chunk polynomials for cycle binding
            // ra_chunks[i][c] = eq(r_addr_chunk_i, PC_chunk_i(c))
            //
            // Like Jolt's compute_r_address_chunks: pad r_address_be with zeros at MSB
            // to make length a multiple of log_k_chunk, then split into d chunks of
            // exactly log_k_chunk variables each.
            const padded_len = self.bytecode_d * self.log_k_chunk;
            const pad_count = padded_len - self.bytecode_log_k;
            var r_address_be_padded = try self.allocator.alloc(F, padded_len);
            defer self.allocator.free(r_address_be_padded);
            // Pad MSB end with zeros (Jolt prepends zeros to r_address which is BE)
            for (0..pad_count) |i| {
                r_address_be_padded[i] = F.zero();
            }
            for (0..self.bytecode_log_k) |i| {
                r_address_be_padded[pad_count + i] = r_address_be[i];
            }

            self.ra_chunks = try self.allocator.alloc([]F, self.bytecode_d);
            const chunk_K: usize = @as(usize, 1) << @intCast(self.log_k_chunk);

            // Pre-compute eq tables for each chunk (small, sequential is fine)
            var eq_tables: [8][]F = undefined; // max bytecode_d = 8
            for (0..self.bytecode_d) |i| {
                self.ra_chunks.?[i] = try self.allocator.alloc(F, T);
                const chunk_start = i * self.log_k_chunk;
                const chunk_end = chunk_start + self.log_k_chunk;
                const r_chunk_be = r_address_be_padded[chunk_start..chunk_end];
                var r_chunk_rev = try self.allocator.alloc(F, self.log_k_chunk);
                defer self.allocator.free(r_chunk_rev);
                for (0..self.log_k_chunk) |ci| {
                    r_chunk_rev[ci] = r_chunk_be[self.log_k_chunk - 1 - ci];
                }
                eq_tables[i] = try computeEqTable(F, self.allocator, r_chunk_rev, self.log_k_chunk);
            }
            defer for (0..self.bytecode_d) |i| self.allocator.free(eq_tables[i]);

            // Fill ra_chunks in parallel across T elements
            if (self.pool) |pool| {
                const RaFillCtx = struct {
                    ra_chunks: [][]F,
                    eq_tables: *const [8][]F,
                    steps: []const tracer.TraceStep,
                    pc_map_ptr: *const BytecodePCMapper,
                    bytecode_d: usize,
                    bytecode_K: usize,
                    chunk_K: usize,
                    log_k_chunk: usize,
                };
                const fill_ctx = RaFillCtx{
                    .ra_chunks = self.ra_chunks.?,
                    .eq_tables = &eq_tables,
                    .steps = self.trace.steps.items,
                    .pc_map_ptr = self.pc_map,
                    .bytecode_d = self.bytecode_d,
                    .bytecode_K = bytecode_K,
                    .chunk_K = chunk_K,
                    .log_k_chunk = self.log_k_chunk,
                };
                pool.parallelFor(T, fill_ctx, struct {
                    fn f(c: RaFillCtx, cycle: usize) void {
                        const step = c.steps[cycle];
                        const pc = c.pc_map_ptr.getPCForStep(step);
                        for (0..c.bytecode_d) |i| {
                            if (pc < c.bytecode_K) {
                                const chunk_val = extractChunkMSB(pc, i, c.bytecode_d, c.log_k_chunk);
                                c.ra_chunks[i][cycle] = if (chunk_val < c.chunk_K) c.eq_tables.*[i][chunk_val] else F.zero();
                            } else {
                                c.ra_chunks[i][cycle] = F.zero();
                            }
                        }
                    }
                }.f);
            } else {
                for (0..T) |c| {
                    const step = self.trace.steps.items[c];
                    const pc = self.pc_map.getPCForStep(step);
                    for (0..self.bytecode_d) |i| {
                        if (pc < bytecode_K) {
                            const chunk_val = extractChunkMSB(pc, i, self.bytecode_d, self.log_k_chunk);
                            self.ra_chunks.?[i][c] = if (chunk_val < chunk_K) eq_tables[i][chunk_val] else F.zero();
                        } else {
                            self.ra_chunks.?[i][c] = F.zero();
                        }
                    }
                }
            }

            // Build 5 GruenSplitEq instances for per-stage eq(r_cycle_s, .)
            // stage_r_cycles[s] is in BE order; pass DIRECTLY to GruenSplitEq.
            // GruenSplitEq uses BE convention internally: E_out/E_in tables are BE-indexed,
            // and bind() processes tau[n-1] first (the LSB in BE = bit 0 of position index).
            // This matches the flat sumcheck's LowToHigh binding which groups positions
            // by bit 0 first. Same pattern as LookupsRaVirtualProver (line ~4446).
            for (0..5) |s| {
                self.stage_gruen_eqs[s] = try poly_mod.GruenSplitEqPolynomial(F).initWithScaling(
                    self.allocator,
                    self.stage_r_cycles[s][0..self.n_cycle_vars],
                    null,
                );
                self.bound_vals_phase2[s] = bound_vals[s];
            }

            // Debug: verify Π_i ra_chunk_i(c) = eq_addr[PC(c)] for each cycle
            // Debug: verify RA product vs eq_binding (behind comptime debug_verbose)
            if (comptime debug_verbose) {
                // Check RA product vs eq_binding (using LH challenges directly)
                const eq_binding_check = try computeEqTableParallel(F, self.allocator, r_address_challenges, self.bytecode_log_k, self.pool);
                defer self.allocator.free(eq_binding_check);
                var mismatch_count: usize = 0;
                for (0..T) |c| {
                    var ra_prod = F.one();
                    for (0..self.bytecode_d) |i| {
                        ra_prod = ra_prod.mul(self.ra_chunks.?[i][c]);
                    }
                    const step = self.trace.steps.items[c];
                    const pc = self.pc_map.getPCForStep(step);
                    const full_eq = if (pc < bytecode_K) eq_binding_check[pc] else F.zero();
                    if (!ra_prod.eql(full_eq)) mismatch_count += 1;
                }
                dbg("[BCRAF_RA] total mismatches: {}/{}\n", .{ mismatch_count, T });
            }

            // Save F_s[0] before freeing
            for (0..5) |s| {
                self.f_s_bound_saved[s] = if (self.F_s_arrs[s].len > 0) self.F_s_arrs[s][0] else F.zero();
            }

            // Free Phase 1 arrays (no longer needed)
            // Replace with zero-length allocations so deinit doesn't double-free
            for (0..5) |s| {
                self.allocator.free(self.F_s_arrs[s]);
                self.F_s_arrs[s] = try self.allocator.alloc(F, 0);
                self.allocator.free(self.val_with_raf[s]);
                self.val_with_raf[s] = try self.allocator.alloc(F, 0);
            }

            // Entry correction: track as a separate scalar that gets bound each round
            self.bound_f_entry = self.entry_val;
            self.entry_correction_scalar = self.entry_gamma.mul(self.bound_f_entry);

            self.current_len = T;
            self.phase = 1;
        }

        /// Phase 2: degree bytecode_d+1 round poly
        /// Returns evals in Toom-Cook format: [p(0), p(1), ..., p(d), p_inf]
        /// Phase 2: compute round polynomial using Toom-Cook delta approach.
        /// Evaluates ∏_i ra_chunk_i(x) * combined(x) at Toom points {0, 1, ..., d, ∞}
        /// using lo/delta decomposition (same pattern as RamRaVirtual).
        ///
        /// combined(pos) is computed on-the-fly from 5 GruenSplitEq instances:
        ///   combined(pos) = Σ_s eff_lo_s * E_out_s[x_out] * E_in_s[x_in] + entry_correction * δ(pos,0)
        /// where eff_lo_s = bound_vals[s] * scalar_s * (1 - tau_s_window)
        ///       eff_hi_s = bound_vals[s] * scalar_s * tau_s_window
        /// Returns evaluations at Toom points (not monomials).
        pub fn computeRoundPolyPhase2(self: *Self, allocator: Allocator) ![]F {
            const half = self.current_len / 2;
            const ra_chunks = self.ra_chunks.?;
            const d_total = self.bytecode_d + 1; // ra_chunks + combined = d_total linear polys
            const n_toom_evals = d_total; // Toom grid: {1, 2, ..., d_total-1, ∞}

            // Precompute per-stage effective scalars for lo (active var=0) and hi (active var=1)
            // For each stage s with GruenSplitEq:
            //   eq_s(2j) = scalar_s * E_out_s[x_out] * E_in_s[x_in] * (1 - tau_s_window)
            //   eq_s(2j+1) = scalar_s * E_out_s[x_out] * E_in_s[x_in] * tau_s_window
            // So: eff_lo_s = bound_vals[s] * scalar_s * (1 - tau_s_window)
            //     eff_hi_s = bound_vals[s] * scalar_s * tau_s_window
            var eff_lo: [5]F = undefined;
            var eff_hi: [5]F = undefined;
            var E_out_arr: [5][]const F = undefined;
            var E_in_arr: [5][]const F = undefined;
            var head_in_bits: usize = 0;
            for (0..5) |s| {
                const gruen = &self.stage_gruen_eqs[s].?;
                const tau_window = gruen.tau[gruen.current_index - 1];
                const one_minus_tau = F.one().sub(tau_window);
                const base = self.bound_vals_phase2[s].mul(gruen.current_scalar);
                eff_lo[s] = base.mul(one_minus_tau);
                eff_hi[s] = base.mul(tau_window);

                const eq_tables = gruen.getWindowEqTables(gruen.current_index, 1);
                E_out_arr[s] = eq_tables.E_out;
                E_in_arr[s] = eq_tables.E_in;
                head_in_bits = eq_tables.head_in_bits; // Same for all 5 stages
            }
            const in_mask: usize = (@as(usize, 1) << @intCast(head_in_bits)) - 1;
            const entry_corr = self.entry_correction_scalar;

            const Ctx = struct {
                ra_chunks: [][]F,
                bytecode_d: usize,
                d_total: usize,
                n_toom_evals: usize,
                eff_lo: [5]F,
                eff_hi: [5]F,
                E_out_arr: [5][]const F,
                E_in_arr: [5][]const F,
                head_in_bits: usize,
                in_mask: usize,
                entry_corr: F,
            };
            const ctx = Ctx{
                .ra_chunks = ra_chunks,
                .bytecode_d = self.bytecode_d,
                .d_total = d_total,
                .n_toom_evals = n_toom_evals,
                .eff_lo = eff_lo,
                .eff_hi = eff_hi,
                .E_out_arr = E_out_arr,
                .E_in_arr = E_in_arr,
                .head_in_bits = head_in_bits,
                .in_mask = in_mask,
                .entry_corr = entry_corr,
            };

            const UPA = UnreducedProductAccum;
            const mapFn = struct {
                fn f(c: Ctx, start: usize, end: usize) [MAX_RA_EVALS]F {
                    const MAX_D = 8;
                    var upa_acc: [MAX_RA_EVALS]UPA = .{UPA.zero()} ** MAX_RA_EVALS;

                    // Factored E_out/E_in: for each x_out block, pre-combine
                    // eff_lo/hi with E_out (saves 5 muls per j vs computing E_out*E_in per j).
                    var prev_x_out: usize = std.math.maxInt(usize);
                    var eff_lo_out: [5]F = undefined;
                    var eff_hi_out: [5]F = undefined;

                    for (start..end) |j| {
                        const x_out = j >> @intCast(c.head_in_bits);
                        const x_in = j & c.in_mask;

                        // Re-compute eff_lo/hi_out only when x_out changes
                        if (x_out != prev_x_out) {
                            prev_x_out = x_out;
                            inline for (0..5) |s| {
                                eff_lo_out[s] = c.eff_lo[s].mul(c.E_out_arr[s][x_out]);
                                eff_hi_out[s] = c.eff_hi[s].mul(c.E_out_arr[s][x_out]);
                            }
                        }

                        var combined_lo = F.zero();
                        var combined_hi = F.zero();
                        inline for (0..5) |s| {
                            const e_in = c.E_in_arr[s][x_in];
                            combined_lo = combined_lo.add(eff_lo_out[s].mul(e_in));
                            combined_hi = combined_hi.add(eff_hi_out[s].mul(e_in));
                        }
                        // Entry correction: only at position 0 (j=0, even position)
                        if (j == 0) {
                            combined_lo = combined_lo.add(c.entry_corr);
                        }

                        // Compute lo[i] = poly_i(2j), delta[i] = poly_i(2j+1) - poly_i(2j)
                        var lo: [MAX_D]F = undefined;
                        var delta: [MAX_D]F = undefined;
                        for (0..c.bytecode_d) |i| {
                            lo[i] = c.ra_chunks[i][2 * j];
                            delta[i] = c.ra_chunks[i][2 * j + 1].sub(lo[i]);
                        }
                        // combined is the last polynomial in the product
                        lo[c.bytecode_d] = combined_lo;
                        delta[c.bytecode_d] = combined_hi.sub(combined_lo);

                        // Evaluate product at point 0: ∏ lo[i]
                        {
                            var product = lo[0];
                            for (1..c.d_total) |i| product = product.mul(lo[i]);
                            upa_acc[0].addAssign(UPA.fromMul(product, F.one()));
                        }

                        // Evaluate product at points 1, 2, ..., d_total-2
                        // cur[i] = lo[i] + k*delta[i] for point k
                        var cur: [MAX_D]F = undefined;
                        for (0..c.d_total) |i| cur[i] = lo[i].add(delta[i]); // point 1
                        {
                            var product = cur[0];
                            for (1..c.d_total) |i| product = product.mul(cur[i]);
                            upa_acc[1].addAssign(UPA.fromMul(product, F.one()));
                        }
                        for (2..c.n_toom_evals) |k| {
                            for (0..c.d_total) |i| cur[i] = cur[i].add(delta[i]);
                            var product = cur[0];
                            for (1..c.d_total) |i| product = product.mul(cur[i]);
                            upa_acc[k].addAssign(UPA.fromMul(product, F.one()));
                        }

                        // Evaluate at ∞: ∏ delta[i] (leading coefficient)
                        {
                            var product = delta[0];
                            for (1..c.d_total) |i| product = product.mul(delta[i]);
                            upa_acc[c.n_toom_evals].addAssign(UPA.fromMul(product, F.one()));
                        }
                    }
                    var acc: [MAX_RA_EVALS]F = undefined;
                    for (0..MAX_RA_EVALS) |i| acc[i] = upa_acc[i].reduce();
                    return acc;
                }
            }.f;

            const reduceFn = struct {
                fn f(a: [MAX_RA_EVALS]F, b: [MAX_RA_EVALS]F) [MAX_RA_EVALS]F {
                    var r: [MAX_RA_EVALS]F = undefined;
                    for (0..MAX_RA_EVALS) |i| {
                        r[i] = a[i].add(b[i]);
                    }
                    return r;
                }
            }.f;

            const result = if (self.pool) |pool|
                pool.parallelReduce([MAX_RA_EVALS]F, half, .{F.zero()} ** MAX_RA_EVALS, ctx, mapFn, reduceFn)
            else
                mapFn(ctx, 0, half);

            // Return Toom evaluations: [p(0), p(1), ..., p(d_total-1), p(∞)]
            const n_evals_out = n_toom_evals + 1;
            var evals = try allocator.alloc(F, n_evals_out);
            for (0..n_evals_out) |i| {
                evals[i] = result[i];
            }
            return evals;
        }

        pub fn bindChallengePhase2(self: *Self, r: F) void {
            const half = self.current_len / 2;
            const ra_chunks = self.ra_chunks.?;

            const bindOne = struct {
                fn f(arr: []F, h: usize, challenge: F) void {
                    for (0..h) |j| {
                        arr[j] = arr[2 * j].add(challenge.mul(arr[2 * j + 1].sub(arr[2 * j])));
                    }
                }
            }.f;

            // Bind ra_chunks (dense arrays)
            if (self.gpu) |gpu| {
                if (half >= 16384) {
                    for (0..self.bytecode_d) |i| {
                        gpu.polyBindLow(ra_chunks[i][0 .. half * 2], r, ra_chunks[i][0..half]) catch bindOne(ra_chunks[i], half, r);
                    }
                } else {
                    for (0..self.bytecode_d) |i| {
                        bindOne(ra_chunks[i], half, r);
                    }
                }
            } else if (self.pool) |pool| {
                const Ctx = struct { ra: [][]F, d: usize, half: usize, r: F };
                const ctx = Ctx{ .ra = ra_chunks, .d = self.bytecode_d, .half = half, .r = r };
                pool.parallelForForce(self.bytecode_d, ctx, struct {
                    fn f(c: Ctx, idx: usize) void {
                        bindOne(c.ra[idx], c.half, c.r);
                    }
                }.f);
            } else {
                for (0..self.bytecode_d) |i| {
                    bindOne(ra_chunks[i], half, r);
                }
            }

            // Bind 5 GruenSplitEq instances — O(1) each
            for (0..5) |s| {
                self.stage_gruen_eqs[s].?.bind(r);
            }

            // Bind entry correction scalar: entry(pos=0) * (1-r) + entry(pos=1) * r = entry * (1-r)
            self.entry_correction_scalar = self.entry_correction_scalar.mul(F.one().sub(r));

            self.current_len = half;
        }

        pub fn getOpeningClaims(self: *const Self, allocator: Allocator) ![]F {
            var claims = try allocator.alloc(F, self.bytecode_d);
            for (0..self.bytecode_d) |i| {
                claims[i] = self.ra_chunks.?[i][0];
            }
            return claims;
        }
    };
}

// =============================================================================
// Stage 6 Batched Sumcheck Prover (Main)
// =============================================================================
pub fn Stage6BatchedProver(comptime F: type) type {
    return struct {
        const Self = @This();

        allocator: Allocator,
        thread_pool: ?*ThreadPool = null,
        gpu_ops: ?*GpuPolyOps = null,

        pub fn init(allocator: Allocator) Self {
            return .{ .allocator = allocator };
        }

        /// GPU-accelerated bindLow: arr[j] = arr[2j] + r*(arr[2j+1] - arr[2j])
        /// Falls back to CPU when GPU unavailable or array too small.
        fn gpuBindLow(arr: []F, half: usize, r: F, gpu_ops: ?*GpuPolyOps) void {
            if (gpu_ops) |gpu| {
                if (half >= 16384) {
                    gpu.polyBindLow(arr[0 .. half * 2], r, arr[0..half]) catch {
                        cpuBindLow(arr, half, r);
                        return;
                    };
                    return;
                }
            }
            cpuBindLow(arr, half, r);
        }

        fn cpuBindLow(arr: []F, half: usize, r: F) void {
            for (0..half) |j| {
                arr[j] = arr[2 * j].add(r.montgomeryMul(arr[2 * j + 1].sub(arr[2 * j])));
            }
        }

        /// Generate Stage 6 batched sumcheck proof with real polynomial evaluation
        pub fn generateStage6Proof(
            self: *Self,
            proof: *SumcheckInstanceProof(F),
            transcript: *Blake2bTranscript(F),
            opening_claims: *OpeningClaims(F),
            // Parameters
            n_cycle_vars: usize,
            bytecode_log_k: usize,
            log_k_chunk: usize,
            bytecode_d: usize,
            ram_d: usize,
            instruction_d: usize,
            lookups_ra_virtual_log_k_chunk: usize,
            // Execution trace
            trace: *const ExecutionTrace,
            // Opening points for BytecodeReadRaf (all BIG_ENDIAN)
            r_cycle_bc1_spartan_outer: []const F,
            r_cycle_bc2_product_virt: []const F,
            r_cycle_bc3_spartan_shift: []const F,
            r_cycle_bc4_regs_rwc: []const F,
            r_cycle_bc5_regs_val: []const F,
            // Opening points for IncClaimReduction (all BIG_ENDIAN)
            r_cycle_inc_ram_rwc: []const F, // RamReadWriteChecking
            r_cycle_inc_ram_val: []const F, // RamValEvaluation
            // Stage 5 challenges for deriving LookupsRaVirtual and RamRaVirtual points
            stage5_challenges: []const F,
            // RAM r_address from Stage 2 (BIG_ENDIAN) — the aligned address used by RamRaClaimReduction
            ram_r_address_stage2_be: []const F,
            // Memory layout for address remapping
            memory_layout: *const jolt_device.MemoryLayout,
            // Bytecode entry table for Val polynomial computation
            bytecode_entries: []const BytecodeEntry,
            // Register address opening points for Stages 4 and 5 (BIG_ENDIAN)
            r_register_4: []const F, // From RegistersReadWriteChecking (address portion)
            r_register_5: []const F, // From RegistersValEvaluation (address portion)
            // BytecodePCMapper for converting ELF addresses to bytecode array indices
            pc_map: *const BytecodePCMapper,
            entry_address: u64,
            // Stage 4 inc_poly copy for diagnostic comparison (pass null slice to skip)
            stage4_inc_poly_copy: []const F,
        ) !Stage6Result(F) {
            // Instance round counts
            const bytecodeReadRaf_rounds = bytecode_log_k + n_cycle_vars;
            const hammingBooleanity_rounds = n_cycle_vars;
            const booleanity_rounds = log_k_chunk + n_cycle_vars;
            const ramRaVirtual_rounds = n_cycle_vars;
            const lookupsRaVirtual_rounds = n_cycle_vars;
            const incClaimReduction_rounds = n_cycle_vars;

            const max_num_rounds = bytecodeReadRaf_rounds;

            // Instance degrees
            const bytecodeReadRaf_degree = bytecode_d + 1;
            const hammingBooleanity_degree: usize = 3;
            const booleanity_degree: usize = 3;
            const ramRaVirtual_degree = ram_d + 1;
            const n_committed_per_virtual = lookups_ra_virtual_log_k_chunk / log_k_chunk;
            const n_virtual_ra_polys = 128 / lookups_ra_virtual_log_k_chunk;
            const lookupsRaVirtual_degree = n_committed_per_virtual + 1;
            const incClaimReduction_degree: usize = 2;

            const max_degree = @max(
                @max(@max(bytecodeReadRaf_degree, hammingBooleanity_degree), @max(booleanity_degree, ramRaVirtual_degree)),
                @max(lookupsRaVirtual_degree, incClaimReduction_degree),
            );

            dbg("[STAGE6] Configuration:\n", .{});
            dbg("  bytecodeReadRaf: {} rounds (addr={}, cycle={}), degree {}\n", .{ bytecodeReadRaf_rounds, bytecode_log_k, n_cycle_vars, bytecodeReadRaf_degree });
            dbg("  hammingBooleanity: {} rounds, degree {}\n", .{ hammingBooleanity_rounds, hammingBooleanity_degree });
            dbg("  booleanity: {} rounds, degree {}\n", .{ booleanity_rounds, booleanity_degree });
            dbg("  ramRaVirtual: {} rounds, degree {}\n", .{ ramRaVirtual_rounds, ramRaVirtual_degree });
            dbg("  lookupsRaVirtual: {} rounds, degree {}\n", .{ lookupsRaVirtual_rounds, lookupsRaVirtual_degree });
            dbg("  incClaimReduction: {} rounds, degree {}\n", .{ incClaimReduction_rounds, incClaimReduction_degree });
            dbg("  max_num_rounds: {}, max_degree: {}\n", .{ max_num_rounds, max_degree });

            // ====================================================================
            // Sample gammas (must match Jolt verifier)
            // ====================================================================

            // Debug: dump transcript state at Stage 6 entry
            if (comptime debug_verbose) {
                dbg("[STAGE6] Transcript state at entry: {{ ", .{});
                for (transcript.state) |b| dbg("{x:0>2} ", .{b});
                dbg("}}, round={}\n", .{transcript.n_rounds});
            }

            dbg("[STAGE6] Transcript at entry: round={}\n", .{transcript.n_rounds});
            const bytecode_raf_gamma_powers = try transcript.challengeScalarPowers(self.allocator, 8);
            defer self.allocator.free(bytecode_raf_gamma_powers);

            // Debug: print first gamma to verify transcript sync
            {
                const g0_be = bytecode_raf_gamma_powers[1].toBytesBE(); // [1] is gamma itself
                dbg("[STAGE6] bytecodeRaf_gamma = [{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                    g0_be[31], g0_be[30], g0_be[29], g0_be[28], g0_be[27], g0_be[26], g0_be[25], g0_be[24],
                });
            }

            const NUM_CIRCUIT_FLAGS: usize = 14;
            const stage1_gammas = try transcript.challengeScalarPowers(self.allocator, 2 + NUM_CIRCUIT_FLAGS);
            defer self.allocator.free(stage1_gammas);

            const stage2_gammas = try transcript.challengeScalarPowers(self.allocator, 4);
            defer self.allocator.free(stage2_gammas);

            const stage3_gammas = try transcript.challengeScalarPowers(self.allocator, 9);
            defer self.allocator.free(stage3_gammas);

            const stage4_gammas = try transcript.challengeScalarPowers(self.allocator, 3);
            defer self.allocator.free(stage4_gammas);

            const NUM_LOOKUP_TABLES: usize = 40;
            const stage5_gammas = try transcript.challengeScalarPowers(self.allocator, 2 + NUM_LOOKUP_TABLES);
            defer self.allocator.free(stage5_gammas);

            dbg("[STAGE6] Sampled BytecodeReadRaf gammas\n", .{});

            // BooleanitySumcheckParams::new() - conditional extra challenges
            // When Stage 5 address variables < log_k_chunk, Jolt samples extra challenges
            // to pad r_address to log_k_chunk length. This happens when LOOKUPS_LOG_K is
            // smaller than log_k_chunk, which doesn't happen in practice (128 > 4).
            if (lookups_ra_virtual_log_k_chunk < log_k_chunk) {
                const extra_count = log_k_chunk - lookups_ra_virtual_log_k_chunk;
                for (0..extra_count) |_| {
                    _ = transcript.challengeScalar();
                }
            }
            // Jolt samples 1 gamma via challenge_scalar_optimized() and derives powers:
            //   gamma_powers_square[i] = γ^(2i) for i = 0..total_d
            // The prover uses gamma_powers[i] = γ^i internally for polynomial scaling,
            // and the verifier uses gamma_powers_square[i] = γ^(2i) for expected_output_claim.
            const total_d = instruction_d + bytecode_d + ram_d;
            const booleanity_gamma = transcript.challengeScalar();
            // Handle degenerate gamma=0 case (same as Jolt: replace with 1)
            const booleanity_gamma_f: F = if (booleanity_gamma.isZero()) F.one() else booleanity_gamma;
            const booleanity_gamma_sq = booleanity_gamma_f.mul(booleanity_gamma_f);
            const booleanity_gammas = try self.allocator.alloc(F, total_d);
            booleanity_gammas[0] = F.one(); // γ^0 = 1
            for (1..total_d) |i| {
                booleanity_gammas[i] = booleanity_gammas[i - 1].mul(booleanity_gamma_sq); // γ^(2i)
            }
            // Also compute γ^i powers for Phase 2 pre-scaling optimization
            const booleanity_gamma_unsq = try self.allocator.alloc(F, total_d);
            booleanity_gamma_unsq[0] = F.one(); // γ^0 = 1
            for (1..total_d) |i| {
                booleanity_gamma_unsq[i] = booleanity_gamma_unsq[i - 1].mul(booleanity_gamma_f); // γ^i
            }

            // LookupsRa::new() - gamma powers for virtual RA batching
            const lookups_ra_gamma_powers = try transcript.challengeScalarPowers(self.allocator, n_virtual_ra_polys);
            defer self.allocator.free(lookups_ra_gamma_powers);
            {
                dbg("[STAGE6] lookups_ra_gamma_powers:\n", .{});
                for (0..@min(n_virtual_ra_polys, 4)) |i| {
                    const gp_le = lookups_ra_gamma_powers[i].toBytes();
                    dbg("  gamma_powers[{}]_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                        i, gp_le[0], gp_le[1], gp_le[2], gp_le[3], gp_le[4], gp_le[5], gp_le[6], gp_le[7],
                    });
                }
            }

            // IncClaimReduction::new() - gamma
            // Jolt uses challenge_scalar() (FULL 128-bit) for inc gamma, not optimized
            const inc_gamma = transcript.challengeScalarFull();

            // ====================================================================
            // Compute input claims
            // ====================================================================

            const bcraf_result = self.computeBytecodeReadRafInputClaim(
                opening_claims,
                bytecode_raf_gamma_powers,
                stage1_gammas,
                stage2_gammas,
                stage3_gammas,
                stage4_gammas,
                stage5_gammas,
            );
            var bytecodeReadRaf_input = bcraf_result.total.add(bytecode_raf_gamma_powers[7]);
            const bcraf_per_stage_claims = bcraf_result.per_stage;

            const hammingBooleanity_input = F.zero();
            const booleanity_input = F.zero();

            const ramRaVirtual_input = opening_claims.get(
                .{ .Virtual = .{ .poly = .RamRa, .sumcheck_id = .RamRaClaimReduction } },
            ) orelse F.zero();

            var lookupsRaVirtual_input = F.zero();
            for (0..n_virtual_ra_polys) |i| {
                const ra_claim = opening_claims.get(
                    .{ .Virtual = .{ .poly = .{ .InstructionRa = i }, .sumcheck_id = .InstructionReadRaf } },
                ) orelse F.zero();
                lookupsRaVirtual_input = lookupsRaVirtual_input.add(lookups_ra_gamma_powers[i].mul(ra_claim));
            }

            const inc_gamma2 = inc_gamma.mul(inc_gamma);
            const inc_gamma3 = inc_gamma2.mul(inc_gamma);

            const v1_claim = opening_claims.get(
                .{ .Committed = .{ .poly = .RamInc, .sumcheck_id = .RamReadWriteChecking } },
            ) orelse F.zero();
            const v2_claim = opening_claims.get(
                .{ .Committed = .{ .poly = .RamInc, .sumcheck_id = .RamValCheck } },
            ) orelse F.zero();
            const w1_claim = opening_claims.get(
                .{ .Committed = .{ .poly = .RdInc, .sumcheck_id = .RegistersReadWriteChecking } },
            ) orelse F.zero();
            const w2_claim = opening_claims.get(
                .{ .Committed = .{ .poly = .RdInc, .sumcheck_id = .RegistersValEvaluation } },
            ) orelse F.zero();

            // Debug: dump inc_gamma and individual claims
            {
                const ig_be = inc_gamma.toBytesBE();
                const v1_be = v1_claim.toBytesBE();
                const v2_be = v2_claim.toBytesBE();
                const w1_be = w1_claim.toBytesBE();
                const w2_be = w2_claim.toBytesBE();
                dbg("[STAGE6] inc_gamma = [{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                    ig_be[31], ig_be[30], ig_be[29], ig_be[28], ig_be[27], ig_be[26], ig_be[25], ig_be[24],
                });
                dbg("[STAGE6] IncClaim v1(RamInc@RamRWC) = [{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                    v1_be[31], v1_be[30], v1_be[29], v1_be[28], v1_be[27], v1_be[26], v1_be[25], v1_be[24],
                });
                dbg("[STAGE6] IncClaim v2(RamInc@RamVal) = [{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                    v2_be[31], v2_be[30], v2_be[29], v2_be[28], v2_be[27], v2_be[26], v2_be[25], v2_be[24],
                });
                dbg("[STAGE6] IncClaim w1(RdInc@RegsRWC) = [{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                    w1_be[31], w1_be[30], w1_be[29], w1_be[28], w1_be[27], w1_be[26], w1_be[25], w1_be[24],
                });
                dbg("[STAGE6] IncClaim w2(RdInc@RegsVal) = [{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                    w2_be[31], w2_be[30], w2_be[29], w2_be[28], w2_be[27], w2_be[26], w2_be[25], w2_be[24],
                });
            }

            const incClaimReduction_input = v1_claim
                .add(inc_gamma.mul(v2_claim))
                .add(inc_gamma2.mul(w1_claim))
                .add(inc_gamma3.mul(w2_claim));

            dbg("[STAGE6] Input claims (LE first 8):\n", .{});
            // Print components for IncClaimReduction
            {
                const v1_be = v1_claim.toBytesBE();
                const v2_be = v2_claim.toBytesBE();
                const w1_be = w1_claim.toBytesBE();
                const w2_be = w2_claim.toBytesBE();
                dbg("  IncClaim components: v1=[{x:0>2},{x:0>2},...] v2=[{x:0>2},{x:0>2},...] w1=[{x:0>2},{x:0>2},...] w2=[{x:0>2},{x:0>2},...]\n", .{
                    v1_be[31], v1_be[30], v2_be[31], v2_be[30], w1_be[31], w1_be[30], w2_be[31], w2_be[30],
                });
            }
            // Print LookupsRa claims
            for (0..@min(n_virtual_ra_polys, 4)) |i| {
                const ra_c = opening_claims.get(
                    .{ .Virtual = .{ .poly = .{ .InstructionRa = i }, .sumcheck_id = .InstructionReadRaf } },
                ) orelse F.zero();
                const ra_be = ra_c.toBytesBE();
                dbg("  InstructionRa[{}] = [{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                    i, ra_be[31], ra_be[30], ra_be[29], ra_be[28], ra_be[27], ra_be[26], ra_be[25], ra_be[24],
                });
            }
            // Print BytecodeReadRaf components
            {
                const bc_be = bytecodeReadRaf_input.toBytesBE();
                dbg("  bytecodeReadRaf_input = [{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                    bc_be[31], bc_be[30], bc_be[29], bc_be[28], bc_be[27], bc_be[26], bc_be[25], bc_be[24],
                });
            }
            {
                const ram_be = ramRaVirtual_input.toBytesBE();
                dbg("  ramRaVirtual_input = [{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                    ram_be[31], ram_be[30], ram_be[29], ram_be[28], ram_be[27], ram_be[26], ram_be[25], ram_be[24],
                });
            }
            {
                const look_be = lookupsRaVirtual_input.toBytesBE();
                dbg("  lookupsRaVirtual_input = [{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                    look_be[31], look_be[30], look_be[29], look_be[28], look_be[27], look_be[26], look_be[25], look_be[24],
                });
            }
            {
                const inc_be = incClaimReduction_input.toBytesBE();
                dbg("  incClaimReduction_input = [{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                    inc_be[31], inc_be[30], inc_be[29], inc_be[28], inc_be[27], inc_be[26], inc_be[25], inc_be[24],
                });
            }

            // ====================================================================
            // Derive opening points for RamRaVirtual and LookupsRaVirtual from Stage 5
            // ====================================================================

            const LOOKUPS_LOG_K: usize = 128;
            const ram_log_k: usize = ram_r_address_stage2_be.len;

            // RamRaVirtual: r_cycle from Stage 5 RamRaClaimReduction, r_address from Stage 2
            // RamRaClaimReduction is cycle-only (log_T rounds), NOT address+cycle.
            // The r_address comes from Stage 2's aligned RAM address, stored in ram_r_address_stage2_be.
            const stage5_max_rounds = LOOKUPS_LOG_K + n_cycle_vars;
            // RamRaClaimReduction has n_cycle_vars rounds (cycle-only), offset = stage5_max - n_cycle_vars
            const ram_ra_offset = stage5_max_rounds - n_cycle_vars;
            dbg("[STAGE6] RamRa challenge offset: stage5_max={}, ram_ra_rounds={}, offset={}\n", .{
                stage5_max_rounds, n_cycle_vars, ram_ra_offset,
            });
            var ram_ra_r_cycle = try self.allocator.alloc(F, n_cycle_vars);
            defer self.allocator.free(ram_ra_r_cycle);
            for (0..n_cycle_vars) |i| {
                // Reverse cycle part: challenges[offset..offset+n_cycle_vars] reversed (BE)
                ram_ra_r_cycle[i] = stage5_challenges[ram_ra_offset + n_cycle_vars - 1 - i];
            }

            // r_address for RamRa: from Stage 2 aligned RAM address (already BIG_ENDIAN)
            // Pad with leading zeros to make length a multiple of log_k_chunk (matching Jolt's compute_r_address_chunks)
            const padded_ram_len = ((ram_log_k + log_k_chunk - 1) / log_k_chunk) * log_k_chunk;
            var ram_ra_r_address_be: []F = undefined;
            var ram_ra_r_address_allocated = false;
            if (padded_ram_len != ram_log_k) {
                ram_ra_r_address_be = try self.allocator.alloc(F, padded_ram_len);
                ram_ra_r_address_allocated = true;
                const pad_count = padded_ram_len - ram_log_k;
                @memset(ram_ra_r_address_be[0..pad_count], F.zero());
                @memcpy(ram_ra_r_address_be[pad_count..], ram_r_address_stage2_be);
            } else {
                ram_ra_r_address_be = @constCast(ram_r_address_stage2_be);
            }
            defer if (ram_ra_r_address_allocated) self.allocator.free(ram_ra_r_address_be);

            // Split r_address into chunks (BIG_ENDIAN, chunk[0] = MSB)
            var ram_ra_addr_chunks = try self.allocator.alloc([]const F, ram_d);
            defer self.allocator.free(ram_ra_addr_chunks);
            for (0..ram_d) |i| {
                const chunk_start = i * log_k_chunk;
                const chunk_end = chunk_start + log_k_chunk;
                ram_ra_addr_chunks[i] = ram_ra_r_address_be[chunk_start..chunk_end];
            }

            // LookupsRaVirtual: r_cycle and r_addr_chunks from InstructionReadRaf (Stage 5 Instance 1)
            // InstructionReadRaf has LOOKUPS_LOG_K + n_cycle_vars = 136 rounds
            // normalize_opening_point: address NOT reversed, cycle IS reversed
            var lookups_ra_r_cycle = try self.allocator.alloc(F, n_cycle_vars);
            defer self.allocator.free(lookups_ra_r_cycle);
            for (0..n_cycle_vars) |i| {
                // Reverse cycle part: challenges[128..136] reversed
                lookups_ra_r_cycle[i] = stage5_challenges[LOOKUPS_LOG_K + n_cycle_vars - 1 - i];
            }
            // Debug: print lookups_ra_r_cycle to compare with Jolt's r_cycle
            for (0..n_cycle_vars) |dbg_i| {
                const dbg_b = lookups_ra_r_cycle[dbg_i].toBytesBE();
                dbg("[S6_RCYCLE] lookups_ra_r_cycle[{}] LE=[", .{dbg_i});
                for (0..8) |bi| dbg("{x:0>2}", .{dbg_b[31 - bi]});
                dbg("]\n", .{});
            }

            // r_address for Lookups: challenges[0..128] NOT reversed (stays LITTLE_ENDIAN)
            // Then compute_r_address_chunks splits into log_k_chunk-sized pieces
            var lookups_ra_addr_chunks = try self.allocator.alloc([]const F, instruction_d);
            defer self.allocator.free(lookups_ra_addr_chunks);
            for (0..instruction_d) |i| {
                const chunk_start = i * log_k_chunk;
                const chunk_end = @min(chunk_start + log_k_chunk, LOOKUPS_LOG_K);
                lookups_ra_addr_chunks[i] = stage5_challenges[chunk_start..chunk_end];
            }

            // ====================================================================
            // Initialize ALL sumcheck instances
            // ====================================================================
            const bench_s6 = (std.posix.getenv("ZOLT_BENCH") != null);
            const t_s6_overall_start = if (bench_s6) std.time.nanoTimestamp() else 0;
            var s6_init_timer: if (s6_bench_timing) std.time.Timer else void = if (comptime s6_bench_timing) std.time.Timer.start() catch unreachable else {};

            // Instance 5: IncClaimReduction (degree 2)
            // IncClaimReduction uses RAM r_cycles (not BytecodeReadRaf r_cycles)
            const t_init_inc = if (bench_s6) std.time.nanoTimestamp() else 0;
            var inc_prover = try IncClaimReductionProver(F).init(
                self.allocator, trace, inc_gamma,
                r_cycle_inc_ram_rwc, r_cycle_inc_ram_val,
                r_cycle_bc4_regs_rwc, r_cycle_bc5_regs_val,
                self.thread_pool,
            );
            inc_prover.gpu = self.gpu_ops;
            defer inc_prover.deinit();
            const t_after_inc = if (bench_s6) std.time.nanoTimestamp() else 0;

            // Direct comparison: Stage 6 rd_inc vs Stage 4 inc_poly
            if (comptime debug_verbose) if (stage4_inc_poly_copy.len > 0) {
                var inc_diff_count: usize = 0;
                const cmp_len = @min(inc_prover.rd_inc.len, stage4_inc_poly_copy.len);
                for (0..cmp_len) |j| {
                    if (!inc_prover.rd_inc[j].eql(stage4_inc_poly_copy[j])) {
                        if (inc_diff_count < 8) {
                            const a = inc_prover.rd_inc[j].toBytes();
                            const b = stage4_inc_poly_copy[j].toBytes();
                            const step_j = trace.steps.items[j];
                            std.debug.print("[S6 vs S4 INC] j={} rd={} noop={} wr={} s6_LE={x:0>16} s4_LE={x:0>16}\n", .{
                                j, step_j.rd_index,
                                @as(u8, if (step_j.is_noop) 1 else 0),
                                @as(u8, if (step_j.rd_written) 1 else 0),
                                @as(u64, @bitCast(a[0..8].*)),
                                @as(u64, @bitCast(b[0..8].*)),
                            });
                        }
                        inc_diff_count += 1;
                    }
                }
                std.debug.print("[S6 vs S4 INC] total differences: {}\n", .{inc_diff_count});
            };

            // Diagnostic: verify IncClaimReduction individual component sums
            if (comptime debug_verbose) {
                const T_inc = inc_prover.current_len;
                // Recompute individual eq tables for diagnosis
                var rev_buf2 = try self.allocator.alloc(F, n_cycle_vars);
                defer self.allocator.free(rev_buf2);

                for (0..n_cycle_vars) |i| rev_buf2[i] = r_cycle_inc_ram_rwc[n_cycle_vars - 1 - i];
                const eq_r2_diag = try computeEqTable(F, self.allocator, rev_buf2, n_cycle_vars);
                defer self.allocator.free(eq_r2_diag);

                for (0..n_cycle_vars) |i| rev_buf2[i] = r_cycle_inc_ram_val[n_cycle_vars - 1 - i];
                const eq_r4_diag = try computeEqTable(F, self.allocator, rev_buf2, n_cycle_vars);
                defer self.allocator.free(eq_r4_diag);

                for (0..n_cycle_vars) |i| rev_buf2[i] = r_cycle_bc4_regs_rwc[n_cycle_vars - 1 - i];
                const eq_s4_diag = try computeEqTable(F, self.allocator, rev_buf2, n_cycle_vars);
                defer self.allocator.free(eq_s4_diag);

                for (0..n_cycle_vars) |i| rev_buf2[i] = r_cycle_bc5_regs_val[n_cycle_vars - 1 - i];
                const eq_s5_diag = try computeEqTable(F, self.allocator, rev_buf2, n_cycle_vars);
                defer self.allocator.free(eq_s5_diag);

                var sv1 = F.zero();
                var sv2 = F.zero();
                var sw1 = F.zero();
                var sw2 = F.zero();
                for (0..T_inc) |j| {
                    sv1 = sv1.add(inc_prover.ram_inc[j].mul(eq_r2_diag[j]));
                    sv2 = sv2.add(inc_prover.ram_inc[j].mul(eq_r4_diag[j]));
                    sw1 = sw1.add(inc_prover.rd_inc[j].mul(eq_s4_diag[j]));
                    sw2 = sw2.add(inc_prover.rd_inc[j].mul(eq_s5_diag[j]));
                }
                const v1_ok: u8 = if (std.mem.eql(u8, &sv1.toBytesBE(), &v1_claim.toBytesBE())) 1 else 0;
                const v2_ok: u8 = if (std.mem.eql(u8, &sv2.toBytesBE(), &v2_claim.toBytesBE())) 1 else 0;
                const w1_ok: u8 = if (std.mem.eql(u8, &sw1.toBytesBE(), &w1_claim.toBytesBE())) 1 else 0;
                const w2_ok: u8 = if (std.mem.eql(u8, &sw2.toBytesBE(), &w2_claim.toBytesBE())) 1 else 0;
                std.debug.print("[INC_DIAG] v1_match={} v2_match={} w1_match={} w2_match={}\n", .{ v1_ok, v2_ok, w1_ok, w2_ok });
                if (v1_ok == 0) {
                    const a = sv1.toBytesBE();
                    const b = v1_claim.toBytesBE();
                    std.debug.print("[INC_DIAG] v1: sum_LE={x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2} claim_LE={x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}\n", .{
                        a[31], a[30], a[29], a[28], a[27], a[26], a[25], a[24],
                        b[31], b[30], b[29], b[28], b[27], b[26], b[25], b[24],
                    });
                }
                if (v2_ok == 0) {
                    const a = sv2.toBytesBE();
                    const b = v2_claim.toBytesBE();
                    std.debug.print("[INC_DIAG] v2: sum_LE={x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2} claim_LE={x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}\n", .{
                        a[31], a[30], a[29], a[28], a[27], a[26], a[25], a[24],
                        b[31], b[30], b[29], b[28], b[27], b[26], b[25], b[24],
                    });
                }
                if (w1_ok == 0) {
                    const a = sw1.toBytesBE();
                    const b = w1_claim.toBytesBE();
                    std.debug.print("[INC_DIAG] w1: sum_LE={x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2} claim_LE={x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}\n", .{
                        a[31], a[30], a[29], a[28], a[27], a[26], a[25], a[24],
                        b[31], b[30], b[29], b[28], b[27], b[26], b[25], b[24],
                    });
                }
                if (w2_ok == 0) {
                    const a = sw2.toBytesBE();
                    const b = w2_claim.toBytesBE();
                    std.debug.print("[INC_DIAG] w2: sum_LE={x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2} claim_LE={x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}\n", .{
                        a[31], a[30], a[29], a[28], a[27], a[26], a[25], a[24],
                        b[31], b[30], b[29], b[28], b[27], b[26], b[25], b[24],
                    });
                }
            }

            // Instance 1: HammingBooleanity (degree 3)
            const t_init_hamming = if (bench_s6) std.time.nanoTimestamp() else 0;
            var hamming_prover = try HammingBooleanityProver(F).init(
                self.allocator, trace, r_cycle_bc1_spartan_outer,
                self.thread_pool,
            );
            hamming_prover.gpu = self.gpu_ops;
            defer hamming_prover.deinit();
            const t_after_hamming = if (bench_s6) std.time.nanoTimestamp() else 0;

            // Instance 3: RamRaVirtual (degree ram_d+1)
            const t_init_ram = if (bench_s6) std.time.nanoTimestamp() else 0;
            var ram_ra_prover = try RamRaVirtualProver(F).init(
                self.allocator, trace, ram_ra_r_cycle,
                ram_ra_addr_chunks, ram_d, memory_layout, log_k_chunk,
                self.thread_pool,
            );
            ram_ra_prover.gpu = self.gpu_ops;
            defer ram_ra_prover.deinit();
            const t_after_ram = if (bench_s6) std.time.nanoTimestamp() else 0;

            // Instance 4: LookupsRaVirtual (degree n_committed_per_virtual+1)
            const t_init_lookups = if (bench_s6) std.time.nanoTimestamp() else 0;
            var lookups_ra_prover = try LookupsRaVirtualProver(F).init(
                self.allocator, trace, lookups_ra_r_cycle,
                lookups_ra_addr_chunks, lookups_ra_gamma_powers,
                n_committed_per_virtual, n_virtual_ra_polys,
                log_k_chunk, instruction_d,
                self.thread_pool,
            );
            lookups_ra_prover.gpu = self.gpu_ops;
            defer lookups_ra_prover.deinit();
            const t_after_lookups = if (bench_s6) std.time.nanoTimestamp() else 0;

            // Verify: eq table partition of unity (Σ eq[j] = 1)
            if (comptime debug_verbose) {
                var eq_sum = F.zero();
                for (0..lookups_ra_prover.current_len) |j| {
                    eq_sum = eq_sum.add(lookups_ra_prover.e_out[j]);
                }
                dbg("[LR_EQ] Σeq==1? {} T={}\n", .{eq_sum.eql(F.one()), lookups_ra_prover.current_len});
            }

            // Instance 2: Booleanity (degree 3, two-phase)
            const t_init_booleanity = if (bench_s6) std.time.nanoTimestamp() else 0;
            // Build BooleanityProver with G tables and eq tables from Stage 5 opening point.
            //
            // In Jolt, r_address for booleanity = last log_k_chunk elements of Stage 5
            // InstructionReadRaf address (reversed to LE). Stage 5 address uses HighToLow
            // binding, so stage5_challenges[0]=MSB. After reverse to LE: [ch[127],...,ch[0]].
            // Last log_k_chunk elements = [ch[3],ch[2],ch[1],ch[0]] = MSB bits in LE.
            //
            // r_cycle for booleanity = same as InstructionReadRaf cycle (LE) = lookups_ra_r_cycle
            //
            // Binding order: LowToHigh for both Phase 1 (address) and Phase 2 (cycle)
            var booleanity_prover = blk_bool: {
                const total_bool_polys = instruction_d + bytecode_d + ram_d;

                // r_address_bool: last log_k_chunk of Stage 5 address in LE
                // Stage 5 address in BE: stage5_challenges[0..128] (MSB first since HighToLow binding)
                // Reverse to LE: [ch[127], ch[126], ..., ch[0]]
                // Last log_k_chunk: [ch[log_k_chunk-1], ..., ch[0]] = MSB bits in LE
                var r_address_bool_le = try self.allocator.alloc(F, log_k_chunk);
                // No defer free - BooleanityProver takes ownership of r_address_bool_le
                for (0..log_k_chunk) |i| {
                    // In LE, element i corresponds to Stage5 address challenge (LOOKUPS_LOG_K - 1 - (LOOKUPS_LOG_K - log_k_chunk + i))
                    // = log_k_chunk - 1 - i
                    r_address_bool_le[i] = stage5_challenges[log_k_chunk - 1 - i];
                }

                // r_cycle_bool_le: same as lookups_ra_r_cycle (already LE)
                // lookups_ra_r_cycle[i] = stage5_challenges[LOOKUPS_LOG_K + n_cycle_vars - 1 - i]

                // Build eq_addr table for Phase 1 (LowToHigh binding)
                // computeEqTable expects BE input (MSB-first) for its internal convention.
                // Since r_address_bool_le is LE and we want LowToHigh binding,
                // the eq table should be indexed such that eq_addr[k] = eq(r_addr_le, k)
                // where bit 0 of k is the LSB, bound first.
                // Jolt's LowToHigh EqPolynomial: eq(r, k) = Π_i (r[i]*k_i + (1-r[i])*(1-k_i))
                // where r[0] corresponds to the LSB of k.
                // For computeEqTable: it expects r in "BE" (MSB first), so reverse LE to BE.
                var r_addr_bool_be_for_eq = try self.allocator.alloc(F, log_k_chunk);
                defer self.allocator.free(r_addr_bool_be_for_eq);
                for (0..log_k_chunk) |i| {
                    r_addr_bool_be_for_eq[i] = r_address_bool_le[log_k_chunk - 1 - i];
                }
                const eq_addr_bool_phase1 = try computeEqTable(F, self.allocator, r_addr_bool_be_for_eq, log_k_chunk);
                defer self.allocator.free(eq_addr_bool_phase1); // Only used for debug verification below

                // Build a SINGLE eq_cycle table used for BOTH G construction AND Phase 2 halving.
                //
                // The table ordering must match Jolt's evals_parallel which iterates .rev():
                //   bit 0 of index j → r_cycle[n-1] (MSB)
                // For our computeEqTable (forward iteration), input[0] must be MSB = lookups[0].
                // So input = lookups_ra_r_cycle directly (BE, MSB first).
                //
                // Using the SAME table for G construction and Phase 2 ensures consistency:
                // Phase 1 reduces address variables with G tables weighted by eq_cycle[j],
                // and Phase 2 halves the same eq_cycle[j] table. The running claim from Phase 1
                // equals the initial Phase 2 polynomial sum, satisfying the transition.
                //
                // After Phase 2 halving with LowToHigh binding, the final eq value equals
                // eq(challenges, r_cycle_BE) = eq(challenges, rev(r_cycle_LE)), matching
                // Jolt's verifier which computes combined_r_cycle = rev(r_cycle_LE).
                // Build GruenSplitEq for Booleanity Phase 2 (O(1) bind)
                // Build flat eq table (LE convention, proven correct for G-tables)
                const eq_cycle_bool_phase2 = try computeEqTableParallel(F, self.allocator, lookups_ra_r_cycle, n_cycle_vars, self.thread_pool);
                // Build GruenSplitEq with REVERSED r_cycle so its binding order matches
                // the LE flat table: GruenSplitEq binds tau[n-1] first, which is
                // reversed[n-1] = lookups_ra_r_cycle[0] = challenge MSB = bit 0 in LE.
                var r_cycle_for_gruen = try self.allocator.alloc(F, n_cycle_vars);
                defer self.allocator.free(r_cycle_for_gruen);
                for (0..n_cycle_vars) |ri| r_cycle_for_gruen[ri] = lookups_ra_r_cycle[n_cycle_vars - 1 - ri];
                const bool_gruen_eq = try poly_mod.GruenSplitEqPolynomial(F).init(self.allocator, r_cycle_for_gruen);
                // eq_cycle_bool_phase2 is NOT deferred - shared with BooleanityProver

                // Build G tables: G_i[k] = Σ_j eq(r_cycle_fixed, j) * [chunk_i(j) == k]
                const T_val: usize = @as(usize, 1) << @intCast(n_cycle_vars);
                const K_val: usize = @as(usize, 1) << @intCast(log_k_chunk);
                var G_tables = try self.allocator.alloc([]F, total_bool_polys);
                for (0..total_bool_polys) |i| {
                    G_tables[i] = try self.allocator.alloc(F, K_val);
                    @memset(G_tables[i], F.zero());
                }

                // OPTIMIZATION: Pre-compute chunk indices for all T steps in ONE parallel pass.
                // This avoids calling computeLookupIndex 38 times per step (once per poly).
                // Each step produces: instruction chunks [0..instr_d], bytecode chunks [0..bc_d], ram chunks [0..ram_d]
                // Stored as u8 per chunk (K < 256).
                const MAX_BOOL_POLYS = 48; // instruction_d(32) + bytecode_d(~3-5) + ram_d(~2-3)
                std.debug.assert(total_bool_polys <= MAX_BOOL_POLYS);

                // Allocate per-step chunk index arrays: chunk_idx[j][poly_i] = chunk value (or K_val for invalid)
                const chunk_idx = try self.allocator.alloc([MAX_BOOL_POLYS]u8, T_val);
                defer self.allocator.free(chunk_idx);

                // Phase 1: Single-pass pre-compute all chunk indices (parallel over T)
                {
                    const ChunkPreCtx = struct {
                        steps: []const tracer.TraceStep,
                        pc_map_ptr: *const BytecodePCMapper,
                        mem_layout: *const jolt_device.MemoryLayout,
                        instr_d: usize,
                        bc_d: usize,
                        rm_d: usize,
                        lkc: usize,
                        K: usize,
                        total_polys: usize,
                        chunk_idx: [][MAX_BOOL_POLYS]u8,
                    };
                    const pre_ctx = ChunkPreCtx{
                        .steps = trace.steps.items,
                        .pc_map_ptr = pc_map,
                        .mem_layout = memory_layout,
                        .instr_d = instruction_d,
                        .bc_d = bytecode_d,
                        .rm_d = ram_d,
                        .lkc = log_k_chunk,
                        .K = K_val,
                        .total_polys = total_bool_polys,
                        .chunk_idx = chunk_idx,
                    };
                    const precomputeFn = struct {
                        fn f(c: ChunkPreCtx, j: usize) void {
                            const step = c.steps[j];
                            const sentinel: u8 = @intCast(c.K); // K < 256, use K as "invalid" sentinel

                            // InstructionRa: compute lookup_idx ONCE, extract all chunks
                            const lookup_idx = computeLookupIndex(step);
                            const mask: u128 = (@as(u128, 1) << @intCast(c.lkc)) - 1;
                            for (0..c.instr_d) |i| {
                                const shift = c.lkc * (c.instr_d - 1 - i);
                                const chunk_val: usize = @intCast((lookup_idx >> @intCast(shift)) & mask);
                                c.chunk_idx[j][i] = if (chunk_val < c.K) @intCast(chunk_val) else sentinel;
                            }

                            // BytecodeRa: compute PC ONCE, extract all chunks
                            const pc_idx: u64 = @intCast(c.pc_map_ptr.getPCForStep(step));
                            for (0..c.bc_d) |i| {
                                const chunk_val = extractChunkMSB(pc_idx, i, c.bc_d, c.lkc);
                                c.chunk_idx[j][c.instr_d + i] = if (chunk_val < c.K) @intCast(chunk_val) else sentinel;
                            }

                            // RamRa: compute address ONCE, extract all chunks
                            if (step.memory_addr) |addr| {
                                if (addr != 0) {
                                    if (c.mem_layout.remapAddress(addr)) |raddr| {
                                        for (0..c.rm_d) |i| {
                                            const chunk_val = extractChunkMSB(raddr, i, c.rm_d, c.lkc);
                                            c.chunk_idx[j][c.instr_d + c.bc_d + i] = if (chunk_val < c.K) @intCast(chunk_val) else sentinel;
                                        }
                                    } else {
                                        for (0..c.rm_d) |i| c.chunk_idx[j][c.instr_d + c.bc_d + i] = sentinel;
                                    }
                                } else {
                                    for (0..c.rm_d) |i| c.chunk_idx[j][c.instr_d + c.bc_d + i] = sentinel;
                                }
                            } else {
                                for (0..c.rm_d) |i| c.chunk_idx[j][c.instr_d + c.bc_d + i] = sentinel;
                            }
                        }
                    }.f;
                    if (self.thread_pool) |pool| {
                        pool.parallelForForce(T_val, pre_ctx, precomputeFn);
                    } else {
                        for (0..T_val) |j| precomputeFn(pre_ctx, j);
                    }
                }

                // Phase 2: Build G tables using pre-computed indices (parallel over polys)
                // Each poly's inner loop is now a simple scatter-add with O(1) index lookup.
                if (self.thread_pool) |pool| {
                    const GBuildCtx = struct {
                        eq_cycle: []const F,
                        chunk_idx: [][MAX_BOOL_POLYS]u8,
                        K: usize,
                        T: usize,
                        G_out: [][]F,
                    };
                    const g_ctx = GBuildCtx{
                        .eq_cycle = eq_cycle_bool_phase2,
                        .chunk_idx = chunk_idx,
                        .K = K_val,
                        .T = T_val,
                        .G_out = G_tables,
                    };
                    pool.parallelForForce(total_bool_polys, g_ctx, struct {
                        fn f(c: GBuildCtx, poly_i: usize) void {
                            const G_i = c.G_out[poly_i];
                            const sentinel: u8 = @intCast(c.K);
                            for (0..c.T) |j| {
                                const cv = c.chunk_idx[j][poly_i];
                                if (cv != sentinel) {
                                    const eq_j = c.eq_cycle[j];
                                    G_i[cv] = G_i[cv].add(eq_j);
                                }
                            }
                        }
                    }.f);
                } else {
                    // Sequential: single pass over T, scatter to all polys per step
                    for (0..T_val) |j| {
                        const eq_j = eq_cycle_bool_phase2[j];
                        if (eq_j.eql(F.zero())) continue;
                        const sentinel: u8 = @intCast(K_val);
                        for (0..total_bool_polys) |i| {
                            const cv = chunk_idx[j][i];
                            if (cv != sentinel) {
                                G_tables[i][cv] = G_tables[i][cv].add(eq_j);
                            }
                        }
                    }
                }

                // Use the independently sampled gammas directly (matching Jolt's challenge_vector_optimized)
                // Jolt formula: Σ_i γ_i * (ra_i² - ra_i), where γ_i are independent challenges
                // booleanity_gammas ownership transfers to BooleanityProver (freed by deinit)
                const gamma_sq = booleanity_gammas;

                // Verify G tables: Σ_k G_i[k] should equal Σ_j eq(r_cycle, j) = 1
                // Actually Σ_k G_i[k] = Σ_j eq(r_cycle, j) * Σ_k [chunk_i(j)==k]
                //                     = Σ_j eq(r_cycle, j) * 1 = 1 (since chunk_i(j) always hits exactly one k)
                // Wait no: only if all j have valid chunks. Noop steps may have chunk_val=0 added.
                // Let's just print the first few G tables for debug.
                dbg("[BOOL_PROVER] G tables built: N={}, K={}, T={}\n", .{ total_bool_polys, K_val, T_val });
                for (0..@min(3, total_bool_polys)) |i| {
                    var g_sum = F.zero();
                    for (0..K_val) |k| g_sum = g_sum.add(G_tables[i][k]);
                    const gs_be = g_sum.toBytesBE();
                    dbg("  G[{}] sum_LE=[{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}]\n", .{
                        i, gs_be[31], gs_be[30], gs_be[29], gs_be[28], gs_be[27], gs_be[26], gs_be[25], gs_be[24],
                    });
                }

                // Initial claim verification: Σ_k eq_addr[k] * Σ_i γ^{2i} * (G_i[k]^2 - G_i[k])
                // This should be zero since ra_i(k,j) is binary.
                // Actually that's the FULL sum; at random r it won't be zero for individual terms.
                // But the initial claim IS zero.
                {
                    var init_sum = F.zero();
                    for (0..K_val) |k| {
                        var q_val = F.zero();
                        for (0..total_bool_polys) |i| {
                            const g_k = G_tables[i][k];
                            q_val = q_val.add(gamma_sq[i].mul(g_k.mul(g_k).sub(g_k)));
                        }
                        init_sum = init_sum.add(eq_addr_bool_phase1[k].mul(q_val));
                    }
                    const is_be = init_sum.toBytesBE();
                    dbg("[BOOL_PROVER] Initial sum (should be ~0) LE=[{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}]\n", .{
                        is_be[31], is_be[30], is_be[29], is_be[28], is_be[27], is_be[26], is_be[25], is_be[24],
                    });
                }

                break :blk_bool try BooleanityProver(F).init(
                    self.allocator,
                    G_tables,
                    r_address_bool_le,
                    bool_gruen_eq,
                    eq_cycle_bool_phase2,
                    gamma_sq,
                    booleanity_gamma_unsq,
                    total_bool_polys,
                    log_k_chunk,
                    n_cycle_vars,
                    trace,
                    instruction_d,
                    bytecode_d,
                    ram_d,
                    memory_layout,
                    pc_map,
                );
            };
            booleanity_prover.pool = self.thread_pool;
            booleanity_prover.gpu = self.gpu_ops;
            defer booleanity_prover.deinit();
            const t_after_booleanity = if (bench_s6) std.time.nanoTimestamp() else 0;

            // Instance 0: BytecodeReadRaf (degree bytecode_d+1)
            // Compute Val polynomials from bytecode entries and stage gammas
            const bytecode_K: usize = @as(usize, 1) << @intCast(bytecode_log_k);
            var bytecode_val_polys: [5][]F = undefined;

            // Precompute eq tables for Stages 4 and 5 register addresses
            // r_register_4 and r_register_5 are the address portions from
            // RegistersReadWriteChecking and RegistersValEvaluation opening points
            const REGISTER_COUNT_LOG2: usize = 7; // log2(128 registers: 32 RISC-V + 96 virtual)
            dbg("[STAGE6] r_register_4 (len={}):\n", .{r_register_4.len});
            for (r_register_4, 0..) |rv, i| {
                dbg("  r_register_4[{}] mont_limbs=[0x{x:0>16}, 0x{x:0>16}, 0x{x:0>16}, 0x{x:0>16}]\n", .{i, rv.limbs[0], rv.limbs[1], rv.limbs[2], rv.limbs[3]});
            }
            dbg("[STAGE6] r_register_5 (len={}):\n", .{r_register_5.len});
            for (r_register_5, 0..) |rv, i| {
                dbg("  r_register_5[{}] mont_limbs=[0x{x:0>16}, 0x{x:0>16}, 0x{x:0>16}, 0x{x:0>16}]\n", .{i, rv.limbs[0], rv.limbs[1], rv.limbs[2], rv.limbs[3]});
            }
            // Jolt's EqPolynomial::evals uses BIG-ENDIAN bit indexing:
            // r[0] maps to MSB of index, r[n-1] maps to LSB.
            // Our computeEqTable uses LITTLE-ENDIAN: r[0] maps to LSB.
            // Fix: reverse the input array so our LE computation produces BE-indexed results.
            var r_register_4_rev = try self.allocator.alloc(F, r_register_4.len);
            defer self.allocator.free(r_register_4_rev);
            for (0..r_register_4.len) |i| {
                r_register_4_rev[i] = r_register_4[r_register_4.len - 1 - i];
            }
            var r_register_5_rev = try self.allocator.alloc(F, r_register_5.len);
            defer self.allocator.free(r_register_5_rev);
            for (0..r_register_5.len) |i| {
                r_register_5_rev[i] = r_register_5[r_register_5.len - 1 - i];
            }
            const eq_table_4 = try computeEqTable(F, self.allocator, r_register_4_rev, REGISTER_COUNT_LOG2);
            defer self.allocator.free(eq_table_4);
            const eq_table_5 = try computeEqTable(F, self.allocator, r_register_5_rev, REGISTER_COUNT_LOG2);
            defer self.allocator.free(eq_table_5);
            // Print eq_table_4 entries in LE hex for comparison with Jolt
            dbg("[STAGE6] eq_table_4 (len={}):\n", .{eq_table_4.len});
            for ([_]usize{0, 1, 2, 8, 10, 15, 31, 127}) |idx| {
                if (idx < eq_table_4.len) {
                    const vbe = eq_table_4[idx].toBytesBE();
                    dbg("  eq4[{}]_LE=[", .{idx});
                    for (0..32) |bi| dbg("{x:0>2}", .{vbe[31 - bi]});
                    dbg("]\n", .{});
                }
            }
            // Print stage4_gammas in LE hex
            dbg("[STAGE6] stage4_gammas:\n", .{});
            for (0..3) |i| {
                const gbe = stage4_gammas[i].toBytesBE();
                dbg("  gamma4[{}]_LE=[", .{i});
                for (0..32) |bi| dbg("{x:0>2}", .{gbe[31 - bi]});
                dbg("]\n", .{});
            }

            for (0..5) |s| {
                bytecode_val_polys[s] = try self.allocator.alloc(F, bytecode_K);
                @memset(bytecode_val_polys[s], F.zero());
            }

            for (0..bytecode_K) |k| {
                if (k >= bytecode_entries.len) break;
                const entry = bytecode_entries[k];

                // Stage 1: unexpanded_pc + γ₁¹·imm + Σ γ₁^(2+i)·circuit_flag_i
                // CRITICAL: The Imm encoding must match Jolt's vanilla verifier exactly.
                // Jolt's NormalizedOperands.imm is i128, but how it gets there depends
                // on the instruction FORMAT type:
                //   FormatI (I-type): u64 as i128 → zero-extended (always positive)
                //   FormatU (U-type): u64 as i128 → zero-extended (always positive)
                //   FormatJ (J-type): u64 as i128 → zero-extended (always positive)
                //   FormatB (B-type): i128 directly → signed
                //   FormatS (S-type): i64 as i128 → sign-extended (signed)
                //   Virtual (0x0B, 0x2B): u64 as i128 (from emit_i helper)
                // Then Jolt calls from_i128(operands.imm) to get the field element.
                const imm_field: F = blk: {
                    const opcode_for_imm = entry.opcode;
                    // Jolt stores imm as i128 in NormalizedOperands, then uses from_i128().
                    // The i128 value depends on the instruction format's source type:
                    //   FormatI (u64): u64 as i128 → zero-extended (always positive)
                    //   FormatU (u64): u64 as i128 → zero-extended (always positive)
                    //   FormatJ (u64): u64 as i128 → zero-extended (always positive)
                    //   FormatB (i128): direct → can be negative
                    //   FormatS (i64): i64 as i128 → sign-extended (can be negative)
                    //   FormatLoad (i64): i64 as i128 → sign-extended (can be negative)
                    // We must match: signed formats use fieldFromI128, unsigned use fromU64.
                    // Signed encoding: must match R1CS witness and Jolt verifier.
                    const is_signed_format = (opcode_for_imm == 0x63) or // B-type (branches: FormatB i128)
                        (opcode_for_imm == 0x23) or // S-type (stores: FormatS i64)
                        (opcode_for_imm == 0x03) or // Load (FormatLoad: i64 sign-extended to i128)
                        (opcode_for_imm == 0x22); // VirtualAssert (FormatAssert: signed i64)
                    if (is_signed_format) {
                        break :blk fieldFromI128(F, @as(i128, entry.imm));
                    } else {
                        // I-type, U-type, J-type, Virtual: u64 zero-extended to i128.
                        // from_i128(u64 as i128) = from_u64(u64), so fromU64(@bitCast) matches.
                        break :blk F.fromU64(@as(u64, @bitCast(entry.imm)));
                    }
                };
                var val1 = F.fromU64(entry.address); // No gamma[0] - Jolt formula: unexpanded_pc + γ¹·imm + Σγ^(2+i)·cf[i]
                val1 = val1.add(stage1_gammas[1].mul(imm_field));
                for (0..14) |i| {
                    if (entry.circuit_flags[i]) {
                        val1 = val1.add(stage1_gammas[2 + i]);
                    }
                }
                bytecode_val_polys[0][k] = val1;

                // Debug: print details for mismatching entries
                if (k == 3 or k == 4 or k == 10 or k == 16 or k == 18 or k == 27 or k == 29 or k == 35) {
                    const addr_be = F.fromU64(entry.address).toBytesBE();
                    const imm_be = imm_field.toBytesBE();
                    dbg("[ZOLT_BC_ENTRY] k={}: addr=0x{x:0>8} imm_LE=[", .{k, entry.address});
                    for (0..8) |bi| dbg("{x:0>2}", .{imm_be[31 - bi]});
                    dbg("] opcode=0x{x:0>2} raw_imm={} cf=[", .{entry.opcode, entry.imm});
                    for (0..14) |ci| {
                        if (entry.circuit_flags[ci]) dbg("1", .{}) else dbg("0", .{});
                    }
                    dbg("]\n", .{});
                    _ = addr_be;
                }

                // Stage 2: γ₂⁰·jump + γ₂¹·branch + γ₂²·write_lookup_to_rd + γ₂³·virtual_instruction
                // Matches upstream a16z/jolt (no IsRdNotZero — that was fork-only)
                var val2 = F.zero();
                if (entry.circuit_flags[@intFromEnum(CircuitFlags.Jump)]) {
                    val2 = val2.add(stage2_gammas[0]);
                }
                if (entry.instruction_flags[@intFromEnum(InstructionFlags.Branch)]) {
                    val2 = val2.add(stage2_gammas[1]);
                }
                if (entry.circuit_flags[@intFromEnum(CircuitFlags.WriteLookupOutputToRD)]) {
                    val2 = val2.add(stage2_gammas[2]);
                }
                if (entry.circuit_flags[@intFromEnum(CircuitFlags.VirtualInstruction)]) {
                    val2 = val2.add(stage2_gammas[3]);
                }
                bytecode_val_polys[1][k] = val2;

                // Stage 3: γ₃⁰·imm + γ₃¹·unexpanded_pc + γ₃²·L_is_rs1 + γ₃³·L_is_pc
                //         + γ₃⁴·R_is_rs2 + γ₃⁵·R_is_imm + γ₃⁶·is_noop
                //         + γ₃⁷·virtual_instruction + γ₃⁸·is_first_in_sequence
                // Uses same signed Imm encoding as Stage 1 (see comment above)
                var val3 = imm_field; // No gamma[0] - Jolt formula: imm + γ¹·unexpanded_pc + Σγ^(2+i)·flags[i]
                val3 = val3.add(stage3_gammas[1].mul(F.fromU64(entry.address)));
                if (entry.instruction_flags[@intFromEnum(InstructionFlags.LeftOperandIsRs1Value)]) {
                    val3 = val3.add(stage3_gammas[2]);
                }
                if (entry.instruction_flags[@intFromEnum(InstructionFlags.LeftOperandIsPC)]) {
                    val3 = val3.add(stage3_gammas[3]);
                }
                if (entry.instruction_flags[@intFromEnum(InstructionFlags.RightOperandIsRs2Value)]) {
                    val3 = val3.add(stage3_gammas[4]);
                }
                if (entry.instruction_flags[@intFromEnum(InstructionFlags.RightOperandIsImm)]) {
                    val3 = val3.add(stage3_gammas[5]);
                }
                if (entry.instruction_flags[@intFromEnum(InstructionFlags.IsNoop)]) {
                    val3 = val3.add(stage3_gammas[6]);
                }
                if (entry.circuit_flags[@intFromEnum(CircuitFlags.VirtualInstruction)]) {
                    val3 = val3.add(stage3_gammas[7]);
                }
                if (entry.is_first_in_sequence) {
                    val3 = val3.add(stage3_gammas[8]);
                }
                bytecode_val_polys[2][k] = val3;

                // Stage 4: γ₄⁰·eq(rd, r_reg4) + γ₄¹·eq(rs1, r_reg4) + γ₄²·eq(rs2, r_reg4)
                const REGISTER_COUNT: usize = 128; // 32 RISC-V + 96 virtual
                var val4 = F.zero();
                if (entry.rd < REGISTER_COUNT) {
                    val4 = val4.add(stage4_gammas[0].mul(eq_table_4[entry.rd]));
                }
                if (entry.rs1 < REGISTER_COUNT) {
                    val4 = val4.add(stage4_gammas[1].mul(eq_table_4[entry.rs1]));
                }
                if (entry.rs2 < REGISTER_COUNT) {
                    val4 = val4.add(stage4_gammas[2].mul(eq_table_4[entry.rs2]));
                }
                bytecode_val_polys[3][k] = val4;

                // Stage 5: eq(rd, r_reg5) + γ₅¹·!is_interleaved + Σ γ₅^(2+i)·table_flag_i
                var val5 = F.zero();
                if (entry.rd < REGISTER_COUNT) {
                    val5 = val5.add(eq_table_5[entry.rd]);
                }
                if (!entry.is_interleaved) {
                    val5 = val5.add(stage5_gammas[1]);
                }
                if (entry.lookup_table_index < 40) {
                    val5 = val5.add(stage5_gammas[2 + @as(usize, entry.lookup_table_index)]);
                }
                bytecode_val_polys[4][k] = val5;
            }

            // Debug: Print Stage 3 Val poly for comparison with Jolt verifier
            if (comptime debug_verbose) {
                dbg("[STAGE6] Val[3] (Stage 4/RegistersRWC) entries:\n", .{});
                for (0..bytecode_K) |k| {
                    const vbe = bytecode_val_polys[3][k].toBytesBE();
                    dbg("  Val[3][{}]_LE=[", .{k});
                    for (0..32) |bi| dbg("{x:0>2}", .{vbe[31 - bi]});
                    dbg("]\n", .{});
                }
            }
            if (debug_verbose) {
                for ([_]usize{0, 1, 2, 4}) |s| {
                    for (0..bytecode_K) |k| {
                        const vbe = bytecode_val_polys[s][k].toBytesBE();
                        dbg("  Val[{}][{}]_LE=[", .{s, k});
                        for (0..32) |bi| dbg("{x:0>2}", .{vbe[31 - bi]});
                        dbg("]\n", .{});
                    }
                }
            }

            // Debug: Dump bytecode entries
            if (comptime debug_verbose) {
                dbg("[STAGE6] Bytecode entries (ALL k=0..{}):\n", .{bytecode_K});
                for (0..@min(bytecode_K, 64)) |k| {
                    if (k >= bytecode_entries.len) break;
                    const entry = bytecode_entries[k];
                    dbg("[STAGE6] entry[{}]: addr=0x{x:0>8} rd={} rs1={} rs2={} imm={} cf=[", .{ k, entry.address, entry.rd, entry.rs1, entry.rs2, entry.imm });
                    for (0..14) |i| {
                        if (i > 0) dbg(",", .{});
                        if (entry.circuit_flags[i]) dbg("1", .{}) else dbg("0", .{});
                    }
                    dbg("] if=[", .{});
                    for (0..7) |i| {
                        if (i > 0) dbg(",", .{});
                        if (entry.instruction_flags[i]) dbg("1", .{}) else dbg("0", .{});
                    }
                    dbg("] lt={} interleaved={}\n", .{ entry.lookup_table_index, @intFromBool(entry.is_interleaved) });
                }
            }

            // Build identity polynomial
            var bytecode_int_poly = try self.allocator.alloc(F, bytecode_K);
            for (0..bytecode_K) |k| {
                bytecode_int_poly[k] = F.fromU64(@intCast(k));
            }

            // DEBUG: Per-field comparison for Stage 1 (SpartanOuter)
            if (comptime debug_verbose) {
                // Compute eq table for Stage 1's r_cycle
                const n_vars = n_cycle_vars;
                const T = @as(usize, 1) << @intCast(n_vars);
                var r_cycle_rev = try self.allocator.alloc(F, n_vars);
                defer self.allocator.free(r_cycle_rev);
                for (0..n_vars) |i| r_cycle_rev[i] = r_cycle_bc1_spartan_outer[n_vars - 1 - i];
                const eq_table_s1 = try computeEqTableParallel(F, self.allocator, r_cycle_rev, n_vars, self.thread_pool);
                defer self.allocator.free(eq_table_s1);

                // Compute F_s[k] = Σ_{c:PC(c)=k} eq(r_cycle, c) for Stage 1
                var F_s_s1 = try self.allocator.alloc(F, bytecode_K);
                defer self.allocator.free(F_s_s1);
                @memset(F_s_s1, F.zero());
                for (0..T) |c| {
                    const step = trace.steps.items[c];
                    const pc_idx = pc_map.getPCForStep(step);
                    if (pc_idx < bytecode_K) {
                        F_s_s1[pc_idx] = F_s_s1[pc_idx].add(eq_table_s1[c]);
                    }
                }

                // Compute per-field bytecode-weighted sums for Stage 1:
                // Stage 1 = γ₁⁰·address + γ₁¹·imm + Σ_i γ₁^(2+i)·cf[i]
                var bc_addr_sum = F.zero();
                var bc_imm_sum = F.zero();
                var bc_cf_sums: [14]F = [_]F{F.zero()} ** 14;

                for (0..bytecode_K) |k| {
                    if (k >= bytecode_entries.len) break;
                    const entry = bytecode_entries[k];
                    bc_addr_sum = bc_addr_sum.add(F_s_s1[k].mul(F.fromU64(entry.address)));
                    const debug_imm_field: F = if (entry.opcode == 0x63 or entry.opcode == 0x23)
                        fieldFromI128(F, @as(i128, entry.imm))
                    else
                        F.fromU64(@as(u64, @bitCast(entry.imm)));
                    bc_imm_sum = bc_imm_sum.add(F_s_s1[k].mul(debug_imm_field));
                    for (0..14) |fi| {
                        if (entry.circuit_flags[fi]) {
                            bc_cf_sums[fi] = bc_cf_sums[fi].add(F_s_s1[k]);
                        }
                    }
                }

                // Get corresponding opening claims for SpartanOuter
                const getClaim = struct {
                    fn get(oc: *OpeningClaims(F), key: OpeningId) F {
                        return oc.get(key) orelse F.zero();
                    }
                }.get;
                const oc_addr = getClaim(opening_claims, .{ .Virtual = .{ .poly = .UnexpandedPC, .sumcheck_id = .SpartanOuter } });
                const oc_imm = getClaim(opening_claims, .{ .Virtual = .{ .poly = .Imm, .sumcheck_id = .SpartanOuter } });

                // Compare and print mismatches
                const addr_match = bc_addr_sum.eql(oc_addr);
                const imm_match = bc_imm_sum.eql(oc_imm);
                dbg("\n[BCRAF_FIELD_CMP] Stage 1 field-by-field comparison:\n", .{});
                dbg("  address: match={}\n", .{@as(u8, if (addr_match) 1 else 0)});
                if (!addr_match) {
                    const a1 = bc_addr_sum.toBytes();
                    const a2 = oc_addr.toBytes();
                    dbg("    bc_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{a1[0],a1[1],a1[2],a1[3],a1[4],a1[5],a1[6],a1[7]});
                    dbg("    oc_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{a2[0],a2[1],a2[2],a2[3],a2[4],a2[5],a2[6],a2[7]});
                }
                dbg("  imm: match={}\n", .{@as(u8, if (imm_match) 1 else 0)});
                if (!imm_match) {
                    const ib1 = bc_imm_sum.toBytes();
                    const ib2 = oc_imm.toBytes();
                    dbg("    bc_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{ib1[0],ib1[1],ib1[2],ib1[3],ib1[4],ib1[5],ib1[6],ib1[7]});
                    dbg("    oc_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{ib2[0],ib2[1],ib2[2],ib2[3],ib2[4],ib2[5],ib2[6],ib2[7]});
                }
                const cf_names = [14][]const u8{ "AddOp", "SubOp", "MulOp", "Load", "Store", "Jump", "WrLookup", "VirtInstr", "Assert", "NoUpdateUPC", "Advice", "IsCompr", "IsFirst", "IsLast" };
                for (0..14) |fi| {
                    const oc_cf = getClaim(opening_claims, .{ .Virtual = .{ .poly = .{ .OpFlags = @intCast(fi) }, .sumcheck_id = .SpartanOuter } });
                    const cf_match = bc_cf_sums[fi].eql(oc_cf);
                    if (!cf_match) {
                        dbg("  cf[{}] ({s}): MISMATCH\n", .{fi, cf_names[fi]});
                        const c1 = bc_cf_sums[fi].toBytes();
                        const c2 = oc_cf.toBytes();
                        dbg("    bc_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{c1[0],c1[1],c1[2],c1[3],c1[4],c1[5],c1[6],c1[7]});
                        dbg("    oc_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{c2[0],c2[1],c2[2],c2[3],c2[4],c2[5],c2[6],c2[7]});
                    }
                }
                // Also check non-RAF rv_claim_1 directly
                var rv1_recomp = F.zero();
                rv1_recomp = rv1_recomp.add(bc_addr_sum); // No gamma[0] - matches Jolt formula
                rv1_recomp = rv1_recomp.add(stage1_gammas[1].mul(bc_imm_sum));
                for (0..14) |fi| {
                    rv1_recomp = rv1_recomp.add(stage1_gammas[2 + fi].mul(bc_cf_sums[fi]));
                }
                const rv1_ext = getClaim(opening_claims, .{ .Virtual = .{ .poly = .UnexpandedPC, .sumcheck_id = .SpartanOuter } });
                _ = rv1_ext;
                // Compare rv1_recomp with rv_claim_1 from computeBytecodeReadRafInputClaim
                // rv1_recomp = Σ_k F_s[k] * val_1_no_raf(k) (the non-RAF part of recomputed)
                // rv1_opening = Σ_i gamma_i * opening_claim_i (from opening_claims)
                var rv1_opening = F.zero();
                rv1_opening = rv1_opening.add(oc_addr); // No gamma[0] - matches Jolt formula
                rv1_opening = rv1_opening.add(stage1_gammas[1].mul(oc_imm));
                for (0..14) |fi| {
                    const oc_cf_fi = getClaim(opening_claims, .{ .Virtual = .{ .poly = .{ .OpFlags = @intCast(fi) }, .sumcheck_id = .SpartanOuter } });
                    rv1_opening = rv1_opening.add(stage1_gammas[2 + fi].mul(oc_cf_fi));
                }
                const rv1_match = rv1_recomp.eql(rv1_opening);
                dbg("  rv1 non-RAF match: {}\n", .{@as(u8, if (rv1_match) 1 else 0)});

                // Check RAF contribution
                const raf_oc = getClaim(opening_claims, .{ .Virtual = .{ .poly = .PC, .sumcheck_id = .SpartanOuter } });
                var bc_pc_sum = F.zero();
                for (0..bytecode_K) |k| {
                    bc_pc_sum = bc_pc_sum.add(F_s_s1[k].mul(F.fromU64(@intCast(k))));
                }
                const raf_match = bc_pc_sum.eql(raf_oc);
                dbg("  PC/RAF match: {}\n", .{@as(u8, if (raf_match) 1 else 0)});
                if (!raf_match) {
                    const r1 = bc_pc_sum.toBytes();
                    const r2 = raf_oc.toBytes();
                    dbg("    bc_pc_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{r1[0],r1[1],r1[2],r1[3],r1[4],r1[5],r1[6],r1[7]});
                    dbg("    oc_pc_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{r2[0],r2[1],r2[2],r2[3],r2[4],r2[5],r2[6],r2[7]});
                }
                // Total claim check
                const total_recomp = rv1_recomp.add(bytecode_raf_gamma_powers[5].mul(bc_pc_sum));
                const total_ext = rv1_opening.add(bytecode_raf_gamma_powers[5].mul(raf_oc));
                dbg("  total_stage1_recomp match total_ext: {}\n", .{@as(u8, if (total_recomp.eql(total_ext)) 1 else 0)});
                dbg("  total_stage1_recomp match bcraf_per_stage_claims[0]: {}\n", .{@as(u8, if (total_recomp.eql(bcraf_per_stage_claims[0])) 1 else 0)});

                dbg("[BCRAF_FIELD_CMP] Done\n\n", .{});
            }

            // DEBUG: Per-field comparison for Stage 2 (SpartanProductVirtualization)
            if (comptime debug_verbose) {
                const n_vars = n_cycle_vars;
                const T = @as(usize, 1) << @intCast(n_vars);
                var r_cycle_rev2 = try self.allocator.alloc(F, n_vars);
                defer self.allocator.free(r_cycle_rev2);
                for (0..n_vars) |i| r_cycle_rev2[i] = r_cycle_bc2_product_virt[n_vars - 1 - i];
                const eq_table_s2 = try computeEqTableParallel(F, self.allocator, r_cycle_rev2, n_vars, self.thread_pool);
                defer self.allocator.free(eq_table_s2);

                // Compute per-field sums: Σ_c eq(r_cycle_2, c) * witness_field[c]
                // Stage 2 witnesses: JumpFlag, BranchFlag, IsRdNotZero, WriteLookupToRD
                var cycle_jump_sum = F.zero();
                var cycle_branch_sum = F.zero();
                var cycle_isrdnz_sum = F.zero();
                var cycle_wrlookup_sum = F.zero();

                for (0..T) |c| {
                    const step = trace.steps.items[c];
                    const pc_idx = pc_map.getPCForStep(step);
                    if (pc_idx < bytecode_K and pc_idx < bytecode_entries.len) {
                        const entry = bytecode_entries[pc_idx];
                        if (entry.circuit_flags[@intFromEnum(CircuitFlags.Jump)]) {
                            cycle_jump_sum = cycle_jump_sum.add(eq_table_s2[c]);
                        }
                        if (entry.instruction_flags[@intFromEnum(InstructionFlags.Branch)]) {
                            cycle_branch_sum = cycle_branch_sum.add(eq_table_s2[c]);
                        }
                        if (entry.instruction_flags[@intFromEnum(InstructionFlags.IsRdNotZero)]) {
                            cycle_isrdnz_sum = cycle_isrdnz_sum.add(eq_table_s2[c]);
                        }
                        if (entry.circuit_flags[@intFromEnum(CircuitFlags.WriteLookupOutputToRD)]) {
                            cycle_wrlookup_sum = cycle_wrlookup_sum.add(eq_table_s2[c]);
                        }
                    }
                }

                const getClaim2 = struct {
                    fn get(oc: *OpeningClaims(F), key: OpeningId) F {
                        return oc.get(key) orelse F.zero();
                    }
                }.get;

                const oc_jump = getClaim2(opening_claims, .{ .Virtual = .{ .poly = .{ .OpFlags = 5 }, .sumcheck_id = .SpartanProductVirtualization } });
                const oc_branch = getClaim2(opening_claims, .{ .Virtual = .{ .poly = .{ .InstructionFlags = 4 }, .sumcheck_id = .SpartanProductVirtualization } });
                const oc_isrdnz = getClaim2(opening_claims, .{ .Virtual = .{ .poly = .{ .InstructionFlags = 6 }, .sumcheck_id = .SpartanProductVirtualization } });
                const oc_wrlookup = getClaim2(opening_claims, .{ .Virtual = .{ .poly = .{ .OpFlags = 6 }, .sumcheck_id = .SpartanProductVirtualization } });

                dbg("\n[BCRAF_FIELD_CMP2] Stage 2 (SpartanProductVirt) field comparison:\n", .{});
                const fields2 = [4]struct { name: []const u8, bc: F, oc: F }{
                    .{ .name = "Jump(OpFlags=5)", .bc = cycle_jump_sum, .oc = oc_jump },
                    .{ .name = "Branch(InstrFlags=4)", .bc = cycle_branch_sum, .oc = oc_branch },
                    .{ .name = "IsRdNotZero(InstrFlags=6)", .bc = cycle_isrdnz_sum, .oc = oc_isrdnz },
                    .{ .name = "WriteLookupToRD(OpFlags=6)", .bc = cycle_wrlookup_sum, .oc = oc_wrlookup },
                };
                for (fields2) |f| {
                    const match2 = f.bc.eql(f.oc);
                    const b1 = f.bc.toBytes();
                    const b2 = f.oc.toBytes();
                    dbg("  {s}: {s}\n", .{f.name, if (match2) "MATCH" else "MISMATCH"});
                    dbg("    bc_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{b1[0],b1[1],b1[2],b1[3],b1[4],b1[5],b1[6],b1[7]});
                    dbg("    oc_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{b2[0],b2[1],b2[2],b2[3],b2[4],b2[5],b2[6],b2[7]});
                }

                // Compute rv2 from recomputed per-field values vs rv2 from opening claims
                var rv2_recomp = F.zero();
                rv2_recomp = rv2_recomp.add(stage2_gammas[0].mul(cycle_jump_sum));
                rv2_recomp = rv2_recomp.add(stage2_gammas[1].mul(cycle_branch_sum));
                rv2_recomp = rv2_recomp.add(stage2_gammas[2].mul(cycle_isrdnz_sum));
                rv2_recomp = rv2_recomp.add(stage2_gammas[3].mul(cycle_wrlookup_sum));

                var rv2_ext = F.zero();
                rv2_ext = rv2_ext.add(stage2_gammas[0].mul(oc_jump));
                rv2_ext = rv2_ext.add(stage2_gammas[1].mul(oc_branch));
                rv2_ext = rv2_ext.add(stage2_gammas[2].mul(oc_isrdnz));
                rv2_ext = rv2_ext.add(stage2_gammas[3].mul(oc_wrlookup));

                const rv2r = rv2_recomp.toBytes();
                const rv2e = rv2_ext.toBytes();
                dbg("  rv2_recomp_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{rv2r[0],rv2r[1],rv2r[2],rv2r[3],rv2r[4],rv2r[5],rv2r[6],rv2r[7]});
                dbg("  rv2_ext_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{rv2e[0],rv2e[1],rv2e[2],rv2e[3],rv2e[4],rv2e[5],rv2e[6],rv2e[7]});
                dbg("  rv2_match: {}\n", .{@as(u8, if (rv2_recomp.eql(rv2_ext)) 1 else 0)});

                dbg("[BCRAF_FIELD_CMP2] Done\n\n", .{});
            }

            // DEBUG: Per-field comparison for Stage 3 (RegistersReadWriteChecking)
            if (comptime debug_verbose) {
                const n_vars = n_cycle_vars;
                const T = @as(usize, 1) << @intCast(n_vars);
                var r_cycle_rev4 = try self.allocator.alloc(F, n_vars);
                defer self.allocator.free(r_cycle_rev4);
                for (0..n_vars) |i| r_cycle_rev4[i] = r_cycle_bc4_regs_rwc[n_vars - 1 - i];
                const eq_table_s4 = try computeEqTableParallel(F, self.allocator, r_cycle_rev4, n_vars, self.thread_pool);
                defer self.allocator.free(eq_table_s4);

                // For each field (rd, rs1, rs2), compute Σ_k F_s[k] * eq(entry[k].reg, r_register_4)
                // F_s[k] = Σ_c:PC(c)=k eq(r_cycle_4, c)
                // First compute F_s[k] for all k
                var F_s = try self.allocator.alloc(F, bytecode_K);
                defer self.allocator.free(F_s);
                @memset(F_s, F.zero());
                for (0..T) |c| {
                    const step = trace.steps.items[c];
                    const pc_idx = pc_map.getPCForStep(step);
                    if (pc_idx < bytecode_K) {
                        F_s[pc_idx] = F_s[pc_idx].add(eq_table_s4[c]);
                    }
                }

                var bc_rd_sum = F.zero();
                var bc_rs1_sum = F.zero();
                var bc_rs2_sum = F.zero();
                const REG_COUNT: usize = 128;
                for (0..bytecode_K) |k| {
                    if (k >= bytecode_entries.len) break;
                    const entry = bytecode_entries[k];
                    if (entry.rd < REG_COUNT) {
                        bc_rd_sum = bc_rd_sum.add(F_s[k].mul(eq_table_4[entry.rd]));
                    }
                    if (entry.rs1 < REG_COUNT) {
                        bc_rs1_sum = bc_rs1_sum.add(F_s[k].mul(eq_table_4[entry.rs1]));
                    }
                    if (entry.rs2 < REG_COUNT) {
                        bc_rs2_sum = bc_rs2_sum.add(F_s[k].mul(eq_table_4[entry.rs2]));
                    }
                }

                const getClaim3 = struct {
                    fn get(oc: *OpeningClaims(F), key: OpeningId) F {
                        return oc.get(key) orelse F.zero();
                    }
                }.get;

                const oc_rd = getClaim3(opening_claims, .{ .Virtual = .{ .poly = .RdWa, .sumcheck_id = .RegistersReadWriteChecking } });
                const oc_rs1 = getClaim3(opening_claims, .{ .Virtual = .{ .poly = .Rs1Ra, .sumcheck_id = .RegistersReadWriteChecking } });
                const oc_rs2 = getClaim3(opening_claims, .{ .Virtual = .{ .poly = .Rs2Ra, .sumcheck_id = .RegistersReadWriteChecking } });

                dbg("\n[BCRAF_FIELD_CMP3] Stage 3 (RegistersRWC) field comparison:\n", .{});
                const fields3 = [3]struct { name: []const u8, bc: F, oc: F }{
                    .{ .name = "RdWa", .bc = bc_rd_sum, .oc = oc_rd },
                    .{ .name = "Rs1Ra", .bc = bc_rs1_sum, .oc = oc_rs1 },
                    .{ .name = "Rs2Ra", .bc = bc_rs2_sum, .oc = oc_rs2 },
                };
                for (fields3) |f| {
                    const match3 = f.bc.eql(f.oc);
                    const b1 = f.bc.toBytesBE();
                    const b2 = f.oc.toBytesBE();
                    dbg("  {s}: {s}\n", .{ f.name, if (match3) "MATCH" else "MISMATCH" });
                    dbg("    bc_LE=[", .{});
                    for (0..8) |bi| dbg("{x:0>2}", .{b1[31 - bi]});
                    dbg("]\n", .{});
                    dbg("    oc_LE=[", .{});
                    for (0..8) |bi| dbg("{x:0>2}", .{b2[31 - bi]});
                    dbg("]\n", .{});
                }

                // Also compute and show combined claim
                var rv4_bc = F.zero();
                rv4_bc = rv4_bc.add(stage4_gammas[0].mul(bc_rd_sum));
                rv4_bc = rv4_bc.add(stage4_gammas[1].mul(bc_rs1_sum));
                rv4_bc = rv4_bc.add(stage4_gammas[2].mul(bc_rs2_sum));
                var rv4_oc = F.zero();
                rv4_oc = rv4_oc.add(stage4_gammas[0].mul(oc_rd));
                rv4_oc = rv4_oc.add(stage4_gammas[1].mul(oc_rs1));
                rv4_oc = rv4_oc.add(stage4_gammas[2].mul(oc_rs2));
                dbg("  rv4_bc match rv4_oc: {}\n", .{@as(u8, if (rv4_bc.eql(rv4_oc)) 1 else 0)});
                dbg("  rv4_bc match bcraf_per_stage[3]: {}\n", .{@as(u8, if (rv4_bc.eql(bcraf_per_stage_claims[3])) 1 else 0)});

                // Compute trace-based rd using val polys (should match bc-based)
                var trace_rd_sum = F.zero();
                var trace_rs1_sum = F.zero();
                var trace_rs2_sum = F.zero();
                var trace_rd_valpoly = F.zero(); // Using bytecode val poly like bc-based
                var trace_rs1_valpoly = F.zero();
                var trace_rs2_valpoly = F.zero();
                var n_mismatch: usize = 0;
                for (0..T) |c| {
                    const step = trace.steps.items[c];
                    const pc_idx = pc_map.getPCForStep(step);

                    // Val-poly-based (should match bc-based Σ_k F_s[k] * eq4[rd_k])
                    if (pc_idx < bytecode_K and pc_idx < bytecode_entries.len) {
                        const ent = bytecode_entries[pc_idx];
                        if (ent.rd < REG_COUNT) {
                            trace_rd_valpoly = trace_rd_valpoly.add(eq_table_s4[c].mul(eq_table_4[ent.rd]));
                        }
                        if (ent.rs1 < REG_COUNT) {
                            trace_rs1_valpoly = trace_rs1_valpoly.add(eq_table_s4[c].mul(eq_table_4[ent.rs1]));
                        }
                        if (ent.rs2 < REG_COUNT) {
                            trace_rs2_valpoly = trace_rs2_valpoly.add(eq_table_s4[c].mul(eq_table_4[ent.rs2]));
                        }
                    }

                    // Opening-claim-based (from trace raw instruction)
                    if (step.is_noop and !step.is_termination_store) continue;
                    const instr = step.instruction;
                    const opcode = instr & 0x7f;
                    const rd_raw: u8 = @truncate((instr >> 7) & 0x1f);
                    const rs1_raw: u8 = @truncate((instr >> 15) & 0x1f);
                    const rs2_raw: u8 = @truncate((instr >> 20) & 0x1f);

                    const writes_rd = switch (opcode) {
                        0x23, 0x63 => false,
                        else => true,
                    };
                    if (writes_rd and rd_raw != 0) {
                        trace_rd_sum = trace_rd_sum.add(eq_table_s4[c].mul(eq_table_4[rd_raw]));
                    }
                    const reads_rs1 = switch (opcode) {
                        0x13, 0x03, 0x67, 0x1b, 0x33, 0x3b, 0x23, 0x63 => true,
                        else => false,
                    };
                    if (reads_rs1) {
                        trace_rs1_sum = trace_rs1_sum.add(eq_table_s4[c].mul(eq_table_4[rs1_raw]));
                    }
                    const reads_rs2 = switch (opcode) {
                        0x33, 0x3b, 0x23, 0x63 => true,
                        else => false,
                    };
                    if (reads_rs2) {
                        trace_rs2_sum = trace_rs2_sum.add(eq_table_s4[c].mul(eq_table_4[rs2_raw]));
                    }
                    // Check for per-cycle rd contribution divergence
                    if (pc_idx < bytecode_K and pc_idx < bytecode_entries.len) {
                        const ent2 = bytecode_entries[pc_idx];
                        // Compute val-poly rd contribution for this cycle
                        const vp_rd_contrib = if (ent2.rd < REG_COUNT) eq_table_4[ent2.rd] else F.zero();
                        // Compute trace-based rd contribution for this cycle
                        const tr_rd_contrib = if (writes_rd and rd_raw != 0 and rd_raw < REG_COUNT)
                            eq_table_4[rd_raw]
                        else
                            F.zero();
                        if (!vp_rd_contrib.eql(tr_rd_contrib) and n_mismatch < 15) {
                            dbg("  [RD_DIVERGE] c={} k={} pc=0x{x} opc=0x{x:0>2} bc_rd={} raw_rd={} writes={} noop={} term={}\n", .{
                                c, pc_idx, step.pc, opcode, ent2.rd, rd_raw, @intFromBool(writes_rd),
                                @intFromBool(step.is_noop), @intFromBool(step.is_termination_store),
                            });
                            n_mismatch += 1;
                        }
                    }
                }
                dbg("  valpoly_rd match bc_rd: {}\n", .{@as(u8, if (trace_rd_valpoly.eql(bc_rd_sum)) 1 else 0)});
                dbg("  valpoly_rs1 match bc_rs1: {}\n", .{@as(u8, if (trace_rs1_valpoly.eql(bc_rs1_sum)) 1 else 0)});
                dbg("  valpoly_rs2 match bc_rs2: {}\n", .{@as(u8, if (trace_rs2_valpoly.eql(bc_rs2_sum)) 1 else 0)});
                dbg("  trace_rd match oc_rd: {}\n", .{@as(u8, if (trace_rd_sum.eql(oc_rd)) 1 else 0)});
                dbg("  valpoly_rd match oc_rd: {}\n", .{@as(u8, if (trace_rd_valpoly.eql(oc_rd)) 1 else 0)});
                // Critical: Does bc_rs1 match oc_rs1? This is the actual BCRAF check.
                dbg("  [RS1_MATCH] bc_rs1 == oc_rs1: {}\n", .{@as(u8, if (bc_rs1_sum.eql(oc_rs1)) 1 else 0)});
                dbg("  [RS1_MATCH] valpoly_rs1 == oc_rs1: {}\n", .{@as(u8, if (trace_rs1_valpoly.eql(oc_rs1)) 1 else 0)});
                // Per-cycle rs1 divergence: compare bytecode entry rs1 vs trace step rs1_index
                {
                    var rs1_div: usize = 0;
                    for (0..T) |c2| {
                        const step_c = trace.steps.items[c2];
                        if (step_c.is_noop and !step_c.is_termination_store) continue;
                        const pc_c = pc_map.getPCForStep(step_c);
                        if (pc_c >= bytecode_K or pc_c >= bytecode_entries.len) continue;
                        const bc_ent = bytecode_entries[pc_c];
                        // bc_ent.rs1 = bytecode entry rs1 (used in BCRAF)
                        // step_c.rs1_index = trace step rs1 (used in opening claim)
                        // step_c.rs1_read = whether rs1 is actually read
                        if (step_c.rs1_read) {
                            // Bytecode says rs1=bc_ent.rs1, trace says rs1=step_c.rs1_index
                            if (bc_ent.rs1 != step_c.rs1_index and rs1_div < 20) {
                                dbg("  [RS1_DIVERGE] c={} k={} pc=0x{x:0>8} bc_rs1={} trace_rs1={} opc=0x{x:0>2}\n", .{
                                    c2, pc_c, step_c.pc, bc_ent.rs1, step_c.rs1_index,
                                    step_c.instruction & 0x7f,
                                });
                                rs1_div += 1;
                            }
                        }
                    }
                    dbg("  [RS1_DIVERGE] total divergences: {}\n", .{rs1_div});
                    // Check for cycles where rs1_read=false but bytecode entry has rs1 < 128
                    var phantom_count: usize = 0;
                    var phantom_contrib = F.zero();
                    for (0..T) |c3| {
                        const step_d = trace.steps.items[c3];
                        if (step_d.is_noop and !step_d.is_termination_store) continue;
                        if (!step_d.rs1_read) {
                            const pc_d = pc_map.getPCForStep(step_d);
                            if (pc_d < bytecode_K and pc_d < bytecode_entries.len) {
                                const bc_d = bytecode_entries[pc_d];
                                if (bc_d.rs1 < REG_COUNT) {
                                    const contrib = eq_table_s4[c3].mul(eq_table_4[bc_d.rs1]);
                                    phantom_contrib = phantom_contrib.add(contrib);
                                    if (phantom_count < 10) {
                                        dbg("  [RS1_PHANTOM] c={} k={} opc=0x{x:0>2} bc_rs1={} rs1_read=false\n", .{
                                            c3, pc_d, step_d.instruction & 0x7f, bc_d.rs1,
                                        });
                                    }
                                    phantom_count += 1;
                                }
                            }
                        }
                    }
                    dbg("  [RS1_PHANTOM] count={}, nonzero={}\n", .{phantom_count, @as(u8, if (!phantom_contrib.eql(F.zero())) 1 else 0)});
                    // If bc_rs1 - phantom_contrib == oc_rs1, then the phantom entries explain the mismatch
                    const adjusted = bc_rs1_sum.sub(phantom_contrib);
                    dbg("  [RS1_PHANTOM] bc_rs1 - phantom == oc_rs1: {}\n", .{@as(u8, if (adjusted.eql(oc_rs1)) 1 else 0)});
                }
                const t_rd = trace_rd_sum.toBytesBE();
                const t_rs1 = trace_rs1_sum.toBytesBE();
                const t_rs2 = trace_rs2_sum.toBytesBE();
                dbg("  trace_rd_LE=[", .{});
                for (0..8) |bi| dbg("{x:0>2}", .{t_rd[31 - bi]});
                dbg("] match_oc={}\n", .{@as(u8, if (trace_rd_sum.eql(oc_rd)) 1 else 0)});
                dbg("  trace_rs1_LE=[", .{});
                for (0..8) |bi| dbg("{x:0>2}", .{t_rs1[31 - bi]});
                dbg("] match_oc={}\n", .{@as(u8, if (trace_rs1_sum.eql(oc_rs1)) 1 else 0)});
                dbg("  trace_rs2_LE=[", .{});
                for (0..8) |bi| dbg("{x:0>2}", .{t_rs2[31 - bi]});
                dbg("] match_oc={}\n", .{@as(u8, if (trace_rs2_sum.eql(oc_rs2)) 1 else 0)});
                // CRITICAL: Compute RdWa claim using EXACT same logic as Stage 4 prover
                // Stage 4 sets rd_wa_poly[rd * T + cycle] = 1 when step.rd_written (including rd=0)
                // After sumcheck: rd_wa_claim = Σ_c eq(r_cycle, c) * eq(rd_index(c), r_addr) * 1{rd_written(c)}
                {
                    var direct_rd_claim = F.zero();
                    var rd_written_0_count: usize = 0;
                    var rd_not_written_but_bc_has_rd: usize = 0;
                    for (0..T) |c4| {
                        const step_e = trace.steps.items[c4];
                        if (step_e.is_noop) {
                            // Stage 4 prover skips noop cycles
                            continue;
                        }
                        if (step_e.rd_written) {
                            const rd_idx = @as(usize, step_e.rd_index);
                            if (rd_idx < REG_COUNT) {
                                direct_rd_claim = direct_rd_claim.add(eq_table_s4[c4].mul(eq_table_4[rd_idx]));
                            }
                            if (rd_idx == 0) rd_written_0_count += 1;
                        } else {
                            // Check if bytecode entry has rd < 128 for this cycle
                            const pc_e = pc_map.getPCForStep(step_e);
                            if (pc_e < bytecode_K and pc_e < bytecode_entries.len) {
                                if (bytecode_entries[pc_e].rd < REG_COUNT) {
                                    rd_not_written_but_bc_has_rd += 1;
                                    if (rd_not_written_but_bc_has_rd <= 5) {
                                        dbg("  [RD_GHOST] c={} k={} pc=0x{x:0>8} opc=0x{x:0>2} bc_rd={} step.rd_idx={} rd_written=0\n", .{
                                            c4, pc_e, step_e.pc, step_e.instruction & 0x7f,
                                            bytecode_entries[pc_e].rd, step_e.rd_index,
                                        });
                                    }
                                }
                            }
                        }
                    }
                    const drcl = direct_rd_claim.toBytesBE();
                    dbg("  [DIRECT_RD] claim_LE=[", .{});
                    for (0..8) |bi| dbg("{x:0>2}", .{drcl[31 - bi]});
                    dbg("] match_oc={} match_bc={}\n", .{
                        @as(u8, if (direct_rd_claim.eql(oc_rd)) 1 else 0),
                        @as(u8, if (direct_rd_claim.eql(bc_rd_sum)) 1 else 0),
                    });
                    dbg("  [DIRECT_RD] rd_written_0_count={} rd_not_written_but_bc_has_rd={}\n", .{
                        rd_written_0_count, rd_not_written_but_bc_has_rd,
                    });
                    // Compute difference
                    const diff = bc_rd_sum.sub(direct_rd_claim);
                    const diff_le = diff.toBytesBE();
                    dbg("  [DIRECT_RD] bc_rd - direct = [", .{});
                    for (0..8) |bi| dbg("{x:0>2}", .{diff_le[31 - bi]});
                    dbg("]\n", .{});
                    // Check: does direct_rd match oc_rd? If not, Stage 4 prover has a bug
                    const diff2 = direct_rd_claim.sub(oc_rd);
                    const diff2_le = diff2.toBytesBE();
                    dbg("  [DIRECT_RD] direct - oc_rd = [", .{});
                    for (0..8) |bi| dbg("{x:0>2}", .{diff2_le[31 - bi]});
                    dbg("]\n", .{});
                }
                dbg("[BCRAF_FIELD_CMP3] Done\n\n", .{});
            }

            // DEBUG: Per-field comparison for Stage 4 (RegistersValEval + InstructionReadRaf)
            if (comptime debug_verbose) {
                const n_vars = n_cycle_vars;
                const T = @as(usize, 1) << @intCast(n_vars);
                var r_cycle_rev5 = try self.allocator.alloc(F, n_vars);
                defer self.allocator.free(r_cycle_rev5);
                for (0..n_vars) |i| r_cycle_rev5[i] = r_cycle_bc5_regs_val[n_vars - 1 - i];
                const eq_table_s5 = try computeEqTableParallel(F, self.allocator, r_cycle_rev5, n_vars, self.thread_pool);
                defer self.allocator.free(eq_table_s5);

                var F_s5 = try self.allocator.alloc(F, bytecode_K);
                defer self.allocator.free(F_s5);
                @memset(F_s5, F.zero());
                for (0..T) |c| {
                    const step = trace.steps.items[c];
                    const pc_idx = pc_map.getPCForStep(step);
                    if (pc_idx < bytecode_K) {
                        F_s5[pc_idx] = F_s5[pc_idx].add(eq_table_s5[c]);
                    }
                }

                const REG_COUNT5: usize = 128;
                var bc_rd5_sum = F.zero();
                var bc_iraf_sum = F.zero();
                var bc_table_sums: [40]F = undefined;
                for (0..40) |t| bc_table_sums[t] = F.zero();
                for (0..bytecode_K) |k| {
                    if (k >= bytecode_entries.len) break;
                    const entry = bytecode_entries[k];
                    if (entry.rd < REG_COUNT5) {
                        bc_rd5_sum = bc_rd5_sum.add(F_s5[k].mul(eq_table_5[entry.rd]));
                    }
                    if (!entry.is_interleaved) {
                        bc_iraf_sum = bc_iraf_sum.add(F_s5[k]);
                    }
                    if (entry.lookup_table_index < 40) {
                        bc_table_sums[entry.lookup_table_index] = bc_table_sums[entry.lookup_table_index].add(F_s5[k]);
                    }
                }

                const getClaim5 = struct {
                    fn get(oc: *OpeningClaims(F), key: OpeningId) F {
                        return oc.get(key) orelse F.zero();
                    }
                }.get;

                const oc_rd5 = getClaim5(opening_claims, .{ .Virtual = .{ .poly = .RdWa, .sumcheck_id = .RegistersValEvaluation } });
                const oc_iraf = getClaim5(opening_claims, .{ .Virtual = .{ .poly = .InstructionRafFlag, .sumcheck_id = .InstructionReadRaf } });

                dbg("\n[BCRAF_FIELD_CMP4] Stage 4 (RegistersValEval+InstrReadRaf) field comparison:\n", .{});
                const rd5_match = bc_rd5_sum.eql(oc_rd5);
                const iraf_match = bc_iraf_sum.eql(oc_iraf);
                const b1r = bc_rd5_sum.toBytesBE();
                const b2r = oc_rd5.toBytesBE();
                dbg("  RdWa: {s}\n", .{if (rd5_match) "MATCH" else "MISMATCH"});
                dbg("    bc_LE=[", .{});
                for (0..8) |bi| dbg("{x:0>2}", .{b1r[31 - bi]});
                dbg("]\n", .{});
                dbg("    oc_LE=[", .{});
                for (0..8) |bi| dbg("{x:0>2}", .{b2r[31 - bi]});
                dbg("]\n", .{});
                const b1i = bc_iraf_sum.toBytesBE();
                const b2i = oc_iraf.toBytesBE();
                dbg("  InstructionRafFlag: {s}\n", .{if (iraf_match) "MATCH" else "MISMATCH"});
                dbg("    bc_LE=[", .{});
                for (0..8) |bi| dbg("{x:0>2}", .{b1i[31 - bi]});
                dbg("]\n", .{});
                dbg("    oc_LE=[", .{});
                for (0..8) |bi| dbg("{x:0>2}", .{b2i[31 - bi]});
                dbg("]\n", .{});

                // Check first few table flags
                var table_mismatches: usize = 0;
                for (0..40) |t| {
                    const oc_tf = getClaim5(opening_claims, .{ .Virtual = .{ .poly = .{ .LookupTableFlag = t }, .sumcheck_id = .InstructionReadRaf } });
                    if (!bc_table_sums[t].eql(oc_tf)) {
                        table_mismatches += 1;
                        if (table_mismatches <= 5) {
                            const bt1 = bc_table_sums[t].toBytesBE();
                            const bt2 = oc_tf.toBytesBE();
                            dbg("  LookupTableFlag[{}]: MISMATCH\n", .{t});
                            dbg("    bc_LE=[", .{});
                            for (0..8) |bi| dbg("{x:0>2}", .{bt1[31 - bi]});
                            dbg("]\n", .{});
                            dbg("    oc_LE=[", .{});
                            for (0..8) |bi| dbg("{x:0>2}", .{bt2[31 - bi]});
                            dbg("]\n", .{});
                        }
                    }
                }
                dbg("  Total LookupTableFlag mismatches: {}\n", .{table_mismatches});

                // Compute per-cycle iraf sum by iterating trace and checking opcode-based identity path
                // This mirrors Stage 5's cycle_is_identity_path logic
                var trace_iraf_sum = F.zero();
                var bc_vs_trace_mismatches: usize = 0;
                for (0..T) |c| {
                    const step = trace.steps.items[c];
                    const pc_idx = pc_map.getPCForStep(step);

                    // Compute identity path from instruction opcode (same as Stage 5)
                    const instr = step.instruction;
                    const opcode_7: u8 = @truncate(instr & 0x7F);
                    const funct3_3: u3 = @truncate((instr >> 12) & 0x7);
                    const funct7_7: u7 = @truncate(instr >> 25);
                    const trace_is_identity = switch (opcode_7) {
                        0x33 => (funct3_3 == 0 and funct7_7 == 0) or // ADD
                            (funct3_3 == 0 and funct7_7 == 0x20) or // SUB
                            (funct7_7 == 0x01 and funct3_3 == 0) or // MUL
                            (funct7_7 == 0x01 and funct3_3 == 3), // MULHU
                        0x13 => (funct3_3 == 0), // ADDI
                        0x1b => (funct3_3 == 0), // ADDIW
                        0x3b => (funct3_3 == 0 and funct7_7 == 0) or // ADDW
                            (funct3_3 == 0 and funct7_7 == 0x20), // SUBW
                        0x37 => true, // LUI
                        0x17 => true, // AUIPC
                        0x6f => true, // JAL
                        0x67 => true, // JALR
                        0x02 => true, // VirtualAdvice (Advice → identity path)
                        0x42 => true, // VirtualZeroExtendWord (AddOperands → identity path)
                        0x0B => true, // VirtualSignExtendWord (AddOperands → identity path)
                        0x2B => true, // VirtualMULI (MultiplyOperands → identity path)
                        else => false,
                    };

                    // bytecode path
                    const bc_raf: bool = if (pc_idx < bytecode_entries.len) !bytecode_entries[pc_idx].is_interleaved else false;

                    if (trace_is_identity) {
                        trace_iraf_sum = trace_iraf_sum.add(eq_table_s5[c]);
                    }

                    if (trace_is_identity != bc_raf and bc_vs_trace_mismatches < 10) {
                        dbg("  [IRAF_MISMATCH] c={} pc_idx={} noop={} trace_ident={} bc_raf={} opcode=0x{x:0>2}\n", .{
                            c, pc_idx, @intFromBool(step.is_noop), @intFromBool(trace_is_identity), @intFromBool(bc_raf), opcode_7,
                        });
                        if (pc_idx < bytecode_entries.len) {
                            dbg("    bc_cf=[", .{});
                            for (0..14) |fi| {
                                if (fi > 0) dbg(",", .{});
                                dbg("{}", .{@intFromBool(bytecode_entries[pc_idx].circuit_flags[fi])});
                            }
                            dbg("] bc_is_interleaved={}\n", .{@intFromBool(bytecode_entries[pc_idx].is_interleaved)});
                        }
                        bc_vs_trace_mismatches += 1;
                    }
                }
                const ti_le = trace_iraf_sum.toBytesBE();
                dbg("  trace_iraf_sum_LE=[", .{});
                for (0..8) |bi| dbg("{x:0>2}", .{ti_le[31 - bi]});
                dbg("] match_oc={} match_bc={}\n", .{
                    @intFromBool(trace_iraf_sum.eql(oc_iraf)),
                    @intFromBool(trace_iraf_sum.eql(bc_iraf_sum)),
                });
                dbg("  bc_vs_trace mismatches: {}\n", .{bc_vs_trace_mismatches});

                dbg("[BCRAF_FIELD_CMP4] Done\n\n", .{});
            }

            var bytecode_gamma_arr: [8]F = undefined;
            for (0..8) |i| {
                bytecode_gamma_arr[i] = bytecode_raf_gamma_powers[i];
            }
            const entry_bytecode_index = pc_map.getPC(entry_address, 0);
            const t_init_bcraf = if (bench_s6) std.time.nanoTimestamp() else 0;
            var bytecode_prover = try BytecodeReadRafProver(F).init(
                self.allocator, trace, pc_map, bytecode_val_polys,
                bytecode_log_k, n_cycle_vars, bytecode_d, log_k_chunk,
                bytecode_gamma_arr,
                [5][]const F{
                    r_cycle_bc1_spartan_outer,
                    r_cycle_bc2_product_virt,
                    r_cycle_bc3_spartan_shift,
                    r_cycle_bc4_regs_rwc,
                    r_cycle_bc5_regs_val,
                },
                bytecode_int_poly,
                bcraf_per_stage_claims,
                entry_bytecode_index,
                self.thread_pool,
            );
            bytecode_prover.gpu = self.gpu_ops;
            defer bytecode_prover.deinit();
            const t_after_bcraf = if (bench_s6) std.time.nanoTimestamp() else 0;

            // pc_maps now consistent — no override needed

            // Debug: Compare prover's initial BytecodeReadRaf claim with opening-claims-derived claim
            if (comptime debug_verbose) {
                var prover_initial = F.zero();
                for (0..5) |s| {
                    prover_initial = prover_initial.add(bytecode_prover.gamma_powers[s].mul(bytecode_prover.stage_claims[s]));
                }
                const pi_be = prover_initial.toBytesBE();
                const oc_be = bytecodeReadRaf_input.toBytesBE();
                dbg("\n[S6P_BCRAF_COMPARE] prover_initial_LE=[", .{});
                for (0..32) |bi| dbg("{x:0>2}", .{pi_be[31 - bi]});
                dbg("]\n[S6P_BCRAF_COMPARE] opening_claims_LE=[", .{});
                for (0..32) |bi| dbg("{x:0>2}", .{oc_be[31 - bi]});
                dbg("]\n[S6P_BCRAF_COMPARE] match={}\n", .{@as(u8, if (prover_initial.eql(bytecodeReadRaf_input)) 1 else 0)});

                for (0..5) |s| {
                    const ps_be = bytecode_prover.stage_claims[s].toBytesBE();
                    const os_be = bcraf_per_stage_claims[s].toBytesBE();
                    const sm = @as(u8, if (bytecode_prover.stage_claims[s].eql(bcraf_per_stage_claims[s])) 1 else 0);
                    if (sm == 0) {
                        dbg("[S6P_BCRAF_COMPARE] stage[{}] MISMATCH! prover_LE=[", .{s});
                        for (0..32) |bi| dbg("{x:0>2}", .{ps_be[31 - bi]});
                        dbg("] opening_LE=[", .{});
                        for (0..32) |bi| dbg("{x:0>2}", .{os_be[31 - bi]});
                        dbg("]\n", .{});
                    }
                }
            }

            // Debug: print r_cycle values for comparison with Jolt
            {
                const r_cycles = [5][]const F{
                    r_cycle_bc1_spartan_outer,
                    r_cycle_bc2_product_virt,
                    r_cycle_bc3_spartan_shift,
                    r_cycle_bc4_regs_rwc,
                    r_cycle_bc5_regs_val,
                };
                for (0..5) |s| {
                    dbg("[ZOLT_BCRAF] r_cycle[{}] (len={}):", .{ s, r_cycles[s].len });
                    for (0..@min(r_cycles[s].len, 4)) |i| {
                        const v_le = r_cycles[s][i].toBytes();
                        dbg(" [{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}]", .{
                            v_le[0], v_le[1], v_le[2], v_le[3], v_le[4], v_le[5], v_le[6], v_le[7],
                        });
                    }
                    if (r_cycles[s].len > 4) dbg("...", .{});
                    dbg("\n", .{});
                }
            }

            // ====================================================================
            // Append input claims and get batching coefficients
            // ====================================================================

            dbg("[STAGE6] Transcript before input_claims: round={}\n", .{transcript.n_rounds});

            transcript.appendScalar("sumcheck_claim", bytecodeReadRaf_input);
            transcript.appendScalar("sumcheck_claim", booleanity_input);
            transcript.appendScalar("sumcheck_claim", hammingBooleanity_input);
            transcript.appendScalar("sumcheck_claim", ramRaVirtual_input);
            transcript.appendScalar("sumcheck_claim", lookupsRaVirtual_input);
            transcript.appendScalar("sumcheck_claim", incClaimReduction_input);

            const batch = try self.allocator.alloc(F, 6);
            defer self.allocator.free(batch);
            for (0..6) |i| {
                batch[i] = transcript.challengeScalarFull();
            }

            const input_claims = [6]F{
                bytecodeReadRaf_input,
                booleanity_input,
                hammingBooleanity_input,
                ramRaVirtual_input,
                lookupsRaVirtual_input,
                incClaimReduction_input,
            };
            const num_rounds_arr = [6]usize{
                bytecodeReadRaf_rounds,
                booleanity_rounds,
                hammingBooleanity_rounds,
                ramRaVirtual_rounds,
                lookupsRaVirtual_rounds,
                incClaimReduction_rounds,
            };

            var batched_claim = F.zero();
            for (0..6) |i| {
                const scale = max_num_rounds - num_rounds_arr[i];
                var scaled = input_claims[i];
                for (0..scale) |_| scaled = scaled.add(scaled);
                batched_claim = batched_claim.add(batch[i].mul(scaled));
            }

            // Debug: print the initial batched claim and all batch coefficients
            {
                const bc_be = batched_claim.toBytesBE();
                dbg("[S6P_BATCHED] initial_batched_claim_LE=[", .{});
                for (0..32) |bi| dbg("{x:0>2}", .{bc_be[31 - bi]});
                dbg("]\n", .{});
                for (0..6) |i| {
                    const b_be = batch[i].toBytesBE();
                    const ic_be = input_claims[i].toBytesBE();
                    dbg("[S6P_BATCHED] batch[{}]_LE=[", .{i});
                    for (0..32) |bi| dbg("{x:0>2}", .{b_be[31 - bi]});
                    dbg("] input_claim_LE=[", .{});
                    for (0..32) |bi| dbg("{x:0>2}", .{ic_be[31 - bi]});
                    dbg("] rounds={}\n", .{num_rounds_arr[i]});
                }
            }

            // ====================================================================
            // Run batched sumcheck
            // ====================================================================

            var challenges = try self.allocator.alloc(F, max_num_rounds);
            errdefer self.allocator.free(challenges);

            var instance_claims: [6]F = input_claims;
            var current_batched_claim = batched_claim;

            const num_compressed = max_degree;

            // Track Phase 1 address challenges for BytecodeReadRaf
            var bytecode_addr_challenges = try self.allocator.alloc(F, bytecode_log_k);
            defer self.allocator.free(bytecode_addr_challenges);

            // Stage 6 fine-grained timing (gated by ZOLT_BENCH env var)
            if (bench_s6) {
                const toMs = struct {
                    fn f(ns: i128) f64 {
                        return @as(f64, @floatFromInt(ns)) / 1_000_000.0;
                    }
                }.f;
                std.debug.print("    [STAGE6-BENCH] Init total: {d:7.1}ms\n", .{
                    @as(f64, @floatFromInt(s6_init_timer.read())) / 1_000_000.0,
                });
                std.debug.print("    [STAGE6-BENCH]   IncClaim init:       {d:7.1}ms\n", .{toMs(t_after_inc - t_init_inc)});
                std.debug.print("    [STAGE6-BENCH]   Hamming init:        {d:7.1}ms\n", .{toMs(t_after_hamming - t_init_hamming)});
                std.debug.print("    [STAGE6-BENCH]   RamRaVirtual init:   {d:7.1}ms\n", .{toMs(t_after_ram - t_init_ram)});
                std.debug.print("    [STAGE6-BENCH]   LookupsRa init:      {d:7.1}ms\n", .{toMs(t_after_lookups - t_init_lookups)});
                std.debug.print("    [STAGE6-BENCH]   Booleanity init:     {d:7.1}ms\n", .{toMs(t_after_booleanity - t_init_booleanity)});
                std.debug.print("    [STAGE6-BENCH]   BytecodeRaf init:    {d:7.1}ms\n", .{toMs(t_after_bcraf - t_init_bcraf)});
                std.debug.print("    [STAGE6-BENCH]   Val polys+eq+other:  {d:7.1}ms\n", .{
                    toMs((t_init_bcraf - t_after_booleanity) + (t_init_booleanity - t_after_lookups) + (t_init_lookups - t_after_ram) + (t_init_ram - t_after_hamming) + (t_init_hamming - t_after_inc)),
                });
            }
            var s6_t_compute: if (s6_bench_timing) [6]u64 else void = if (comptime s6_bench_timing) [6]u64{ 0, 0, 0, 0, 0, 0 } else {};
            var s6_t_bind: if (s6_bench_timing) [6]u64 else void = if (comptime s6_bench_timing) [6]u64{ 0, 0, 0, 0, 0, 0 } else {};
            var s6_t_transcript: if (s6_bench_timing) u64 else void = if (comptime s6_bench_timing) @as(u64, 0) else {};
            var s6_timer: if (s6_bench_timing) std.time.Timer else void = if (comptime s6_bench_timing) std.time.Timer.start() catch unreachable else {};

            for (0..max_num_rounds) |round| {
                const remaining_rounds = max_num_rounds - round;

                // Monomial-form batched polynomial: combined_coeffs[i] = coefficient of x^i
                // This matches Jolt's approach: each instance returns a UniPoly in monomial form,
                // and the batched poly is Σ batch[i] * poly_i in coefficient space.
                var combined_coeffs = try self.allocator.alloc(F, max_degree + 1);
                defer self.allocator.free(combined_coeffs);
                @memset(combined_coeffs, F.zero());

                // Per-instance cached round poly evals for claim tracking
                // We cache each instance's round poly so we don't recompute after challenge
                // Phase 1: degree-2 coefficients [a0, a1, a2] for p(x) = a0 + a1*x + a2*x^2
                var cached_bc_phase1_coeffs: [3]F = undefined;
                var cached_bc_phase1_per_stage: [5][2]F = undefined;
                var cached_bc_phase2: ?[]F = null;
                var cached_hamming: [4]F = undefined;
                var cached_ram_ra: ?[]F = null;
                var cached_lookups_ra: ?[]F = null;
                var cached_inc: [3]F = undefined; // Vandermonde: [p(0), p(1), p(2)]
                var cached_inc_p1: F = F.zero(); // recovered p(1)

                // Track which instances are active this round
                var inst_active: [6]bool = .{ false, false, false, false, false, false };
                const debug_r5 = (round == 5 or round == 6);
                // Debug: per-instance contribution to combined_coeffs[0] and [1]
                var dbg_inst_p0: [6]F = .{F.zero()} ** 6;
                var dbg_inst_p1: [6]F = .{F.zero()} ** 6;

                // Instance 0: BytecodeReadRaf - REAL prover
                if (bench_s6) s6_timer.reset();
                {
                    const inst = 0;
                    if (remaining_rounds > num_rounds_arr[inst]) {
                        // Not started yet - constant polynomial (degree 0)
                        const scaled = sumcheck_helpers.inactiveContribution(F, input_claims[inst], remaining_rounds, num_rounds_arr[inst]);
                        combined_coeffs[0] = combined_coeffs[0].add(batch[inst].mul(scaled));
                    } else {
                        inst_active[inst] = true;
                        if (bytecode_prover.phase == 0) {
                            // Phase 1: address binding (degree-2 poly)
                            // computeRoundPolyPhase1 returns aggregated [p(0), p(2)] and per-stage evals
                            const phase1_result = bytecode_prover.computeRoundPolyPhase1();
                            cached_bc_phase1_per_stage = phase1_result.per_stage;
                            const p0 = phase1_result.agg[0];
                            const p2 = phase1_result.agg[1];
                            // Recover p(1) from sumcheck constraint: p(0) + p(1) = claim
                            const p1 = instance_claims[inst].sub(p0);

                            if (round < 2) {
                                const bc_sum = p0.add(p1);
                                dbg("  [S6P] R{} BC_Phase1 p(0)={any} p(1)={any} p(2)={any} sum={any} claim={any}\n", .{
                                    round,
                                    p0.toBytesBE()[0..8],
                                    p1.toBytesBE()[0..8],
                                    p2.toBytesBE()[0..8],
                                    bc_sum.toBytesBE()[0..8],
                                    instance_claims[0].toBytesBE()[0..8],
                                });
                            }

                            // Interpolate degree-2 coefficients from evals at {0, 1, 2}
                            // p(x) = a0 + a1*x + a2*x^2
                            // a0 = p(0)
                            // a2 = (p(2) - 2*p(1) + p(0)) / 2
                            // a1 = p(1) - p(0) - a2
                            const two = F.fromU64(2);
                            const two_inv = two.inverse().?;
                            const a0 = p0;
                            const a2 = p2.sub(p1.add(p1)).add(p0).mul(two_inv);
                            const a1 = p1.sub(p0).sub(a2);
                            cached_bc_phase1_coeffs = [3]F{ a0, a1, a2 };

                            // Add degree-2 monomial coefficients [a0, a1, a2] to combined_coeffs
                            combined_coeffs[0] = combined_coeffs[0].add(batch[inst].mul(a0));
                            combined_coeffs[1] = combined_coeffs[1].add(batch[inst].mul(a1));
                            combined_coeffs[2] = combined_coeffs[2].add(batch[inst].mul(a2));
                        } else {
                            // Phase 2: cycle binding (degree bytecode_d+1)
                            // Returns Toom-Cook evals: [p(0), p(1), ..., p(d), p(∞)]
                            const polys = try bytecode_prover.computeRoundPolyPhase2(self.allocator);
                            cached_bc_phase2 = polys;
                            if (debug_r5) {
                                const p01 = polys[0].add(polys[1]);
                                const p01_ok: u8 = if (std.mem.eql(u8, &p01.toBytesBE(), &instance_claims[inst].toBytesBE())) 1 else 0;
                                dbg("  [R5_DBG] inst0_phase2 polys_len={} p(0)+p(1)=claim? {}\n", .{ polys.len, p01_ok });
                            }
                            // Convert Toom-Cook evaluations to monomial coefficients
                            const mono = try UniPoly(F).fromEvalsToom(self.allocator, polys);
                            defer self.allocator.free(mono);
                            for (0..mono.len) |ci| {
                                combined_coeffs[ci] = combined_coeffs[ci].add(batch[inst].mul(mono[ci]));
                            }
                        }
                    }
                }

                dbg_inst_p0[0] = combined_coeffs[0];
                dbg_inst_p1[0] = combined_coeffs[1];

                if (debug_r5) {
                    const e0 = combined_coeffs[0].toBytes();
                    const e1 = combined_coeffs[1].toBytes();
                    dbg("  [R5_DBG] after inst0: c[0]_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}] c[1]_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                        e0[0], e0[1], e0[2], e0[3], e0[4], e0[5], e0[6], e0[7],
                        e1[0], e1[1], e1[2], e1[3], e1[4], e1[5], e1[6], e1[7],
                    });
                }

                if (bench_s6) s6_t_compute[0] += s6_timer.read();
                // Instance 1: Booleanity - REAL prover (degree 3)
                if (bench_s6) s6_timer.reset();
                var cached_booleanity: ?[]F = null;
                {
                    const inst = 1;
                    if (remaining_rounds > num_rounds_arr[inst]) {
                        const scaled = sumcheck_helpers.inactiveContribution(F, input_claims[inst], remaining_rounds, num_rounds_arr[inst]);
                        combined_coeffs[0] = combined_coeffs[0].add(batch[inst].mul(scaled));
                    } else {
                        inst_active[inst] = true;
                        const polys = try booleanity_prover.computeRoundPoly(self.allocator, instance_claims[inst]);
                        cached_booleanity = polys;
                        {
                            const p01 = polys[0].add(polys[1]);
                            const p01_ok: u8 = if (std.mem.eql(u8, &p01.toBytesBE(), &instance_claims[inst].toBytesBE())) 1 else 0;
                            const p0b = polys[0].toBytesBE();
                            const p1b = polys[1].toBytesBE();
                            dbg("  [S6P] R{} Bool p(0)+p(1)=claim? {} phase={} p0=[{x:0>2}{x:0>2}{x:0>2}{x:0>2}] p1=[{x:0>2}{x:0>2}{x:0>2}{x:0>2}]\n", .{
                                round, p01_ok, if (booleanity_prover.round < booleanity_prover.log_k_chunk) @as(u8, 1) else 2,
                                p0b[31], p0b[30], p0b[29], p0b[28],
                                p1b[31], p1b[30], p1b[29], p1b[28],
                            });
                        }
                        // Convert degree-3 evals [p(0), p(1), p(2), p(3)] to monomial coefficients
                        // using finite differences, then add batch[inst] * coeffs to combined_coeffs
                        addEvalsAsMonomialToCoeffs(F, combined_coeffs, polys, 4, batch[inst]);
                    }
                }
                dbg_inst_p0[1] = combined_coeffs[0];
                dbg_inst_p1[1] = combined_coeffs[1];

                if (debug_r5) {
                    const e0 = combined_coeffs[0].toBytes();
                    const e1 = combined_coeffs[1].toBytes();
                    dbg("  [R5_DBG] after inst1: c[0]_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}] c[1]_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                        e0[0], e0[1], e0[2], e0[3], e0[4], e0[5], e0[6], e0[7],
                        e1[0], e1[1], e1[2], e1[3], e1[4], e1[5], e1[6], e1[7],
                    });
                }

                if (bench_s6) s6_t_compute[1] += s6_timer.read();
                // Instance 2: HammingBooleanity - REAL prover
                if (bench_s6) s6_timer.reset();
                {
                    const inst = 2;
                    if (remaining_rounds > num_rounds_arr[inst]) {
                        const scaled = sumcheck_helpers.inactiveContribution(F, input_claims[inst], remaining_rounds, num_rounds_arr[inst]);
                        combined_coeffs[0] = combined_coeffs[0].add(batch[inst].mul(scaled));
                    } else {
                        inst_active[inst] = true;
                        const polys = hamming_prover.computeRoundPoly(instance_claims[inst]);
                        cached_hamming = polys;
                        addEvalsAsMonomialToCoeffs(F, combined_coeffs, &polys, 4, batch[inst]);
                    }
                }
                dbg_inst_p0[2] = combined_coeffs[0];
                dbg_inst_p1[2] = combined_coeffs[1];

                if (debug_r5) {
                    const e0 = combined_coeffs[0].toBytes();
                    const e1 = combined_coeffs[1].toBytes();
                    dbg("  [R5_DBG] after inst2: c[0]_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}] c[1]_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                        e0[0], e0[1], e0[2], e0[3], e0[4], e0[5], e0[6], e0[7],
                        e1[0], e1[1], e1[2], e1[3], e1[4], e1[5], e1[6], e1[7],
                    });
                }

                if (bench_s6) s6_t_compute[2] += s6_timer.read();
                // Instance 3: RamRaVirtual - REAL prover
                if (bench_s6) s6_timer.reset();
                {
                    const inst = 3;
                    if (remaining_rounds > num_rounds_arr[inst]) {
                        const scaled = sumcheck_helpers.inactiveContribution(F, input_claims[inst], remaining_rounds, num_rounds_arr[inst]);
                        combined_coeffs[0] = combined_coeffs[0].add(batch[inst].mul(scaled));
                    } else {
                        inst_active[inst] = true;
                        // computeRoundPoly now returns monomial coefficients directly (Toom-Cook quotient approach)
                        const mono = try ram_ra_prover.computeRoundPoly(self.allocator, instance_claims[inst]);
                        cached_ram_ra = mono;
                        if (debug_r5) {
                            // Check p(0)+p(1)=claim for RamRaVirtual (mono format: eval via Horner)
                            var p0 = mono[mono.len - 1];
                            var ci_dbg: usize = mono.len - 1;
                            while (ci_dbg > 0) { ci_dbg -= 1; p0 = p0.mul(F.zero()).add(mono[ci_dbg]); }
                            var p1 = mono[mono.len - 1];
                            ci_dbg = mono.len - 1;
                            while (ci_dbg > 0) { ci_dbg -= 1; p1 = p1.mul(F.one()).add(mono[ci_dbg]); }
                            const p01 = p0.add(p1);
                            const p01_ok: u8 = if (std.mem.eql(u8, &p01.toBytesBE(), &instance_claims[inst].toBytesBE())) 1 else 0;
                            dbg("  [R5_DBG] inst3 polys_len={} p(0)+p(1)=claim? {}\n", .{ mono.len, p01_ok });
                        }
                        for (0..mono.len) |ci| {
                            combined_coeffs[ci] = combined_coeffs[ci].add(batch[inst].mul(mono[ci]));
                        }
                    }
                }
                dbg_inst_p0[3] = combined_coeffs[0];
                dbg_inst_p1[3] = combined_coeffs[1];

                if (debug_r5) {
                    const e0 = combined_coeffs[0].toBytes();
                    const e1 = combined_coeffs[1].toBytes();
                    dbg("  [R5_DBG] after inst3: c[0]_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}] c[1]_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                        e0[0], e0[1], e0[2], e0[3], e0[4], e0[5], e0[6], e0[7],
                        e1[0], e1[1], e1[2], e1[3], e1[4], e1[5], e1[6], e1[7],
                    });
                }

                if (bench_s6) s6_t_compute[3] += s6_timer.read();
                // Instance 4: LookupsRaVirtual - REAL prover
                // Overlap with previous instances via join when both are active
                if (bench_s6) s6_timer.reset();
                {
                    const inst = 4;
                    if (remaining_rounds > num_rounds_arr[inst]) {
                        const scaled = sumcheck_helpers.inactiveContribution(F, input_claims[inst], remaining_rounds, num_rounds_arr[inst]);
                        combined_coeffs[0] = combined_coeffs[0].add(batch[inst].mul(scaled));
                    } else {
                        inst_active[inst] = true;
                        // computeRoundPoly now returns monomial coefficients directly (Toom-Cook quotient approach)
                        const mono = try lookups_ra_prover.computeRoundPoly(self.allocator, instance_claims[inst]);
                        cached_lookups_ra = mono;
                        for (0..mono.len) |ci| {
                            combined_coeffs[ci] = combined_coeffs[ci].add(batch[inst].mul(mono[ci]));
                        }
                    }
                }
                dbg_inst_p0[4] = combined_coeffs[0];
                dbg_inst_p1[4] = combined_coeffs[1];

                if (debug_r5) {
                    const e0 = combined_coeffs[0].toBytes();
                    const e1 = combined_coeffs[1].toBytes();
                    dbg("  [R5_DBG] after inst4: c[0]_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}] c[1]_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                        e0[0], e0[1], e0[2], e0[3], e0[4], e0[5], e0[6], e0[7],
                        e1[0], e1[1], e1[2], e1[3], e1[4], e1[5], e1[6], e1[7],
                    });
                }

                if (bench_s6) s6_t_compute[4] += s6_timer.read();
                // Instance 5: IncClaimReduction - REAL prover
                if (bench_s6) s6_timer.reset();
                {
                    const inst = 5;
                    if (remaining_rounds > num_rounds_arr[inst]) {
                        const scaled = sumcheck_helpers.inactiveContribution(F, input_claims[inst], remaining_rounds, num_rounds_arr[inst]);
                        combined_coeffs[0] = combined_coeffs[0].add(batch[inst].mul(scaled));
                    } else {
                        inst_active[inst] = true;
                        const polys = inc_prover.computeRoundPoly();
                        cached_inc = polys;
                        // polys = [p(0), p(1), p(2)] in Vandermonde format for degree 2
                        const p0 = polys[0];
                        const p1 = polys[1];
                        cached_inc_p1 = p1;
                        if (debug_r5) {
                            const p01_ok: u8 = if (std.mem.eql(u8, &p0.add(p1).toBytesBE(), &instance_claims[inst].toBytesBE())) 1 else 0;
                            dbg("  [R5_DBG] inst5 p(0)+p(1)=claim? {} p(0)_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}] p(1)_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                                p01_ok,
                                p0.toBytes()[0], p0.toBytes()[1], p0.toBytes()[2], p0.toBytes()[3], p0.toBytes()[4], p0.toBytes()[5], p0.toBytes()[6], p0.toBytes()[7],
                                p1.toBytes()[0], p1.toBytes()[1], p1.toBytes()[2], p1.toBytes()[3], p1.toBytes()[4], p1.toBytes()[5], p1.toBytes()[6], p1.toBytes()[7],
                            });
                        }

                        // IncClaimReduction is degree 2 in Vandermonde format [p(0), p(1), p(2)].
                        // Interpolate monomial coefficients: a0 + a1*x + a2*x^2
                        const a0 = p0;
                        const two = F.fromU64(2);
                        const two_inv = two.inverse().?;
                        const a2_coeff = polys[2].sub(p1.add(p1)).add(p0).mul(two_inv);
                        const a1 = p1.sub(a0).sub(a2_coeff);

                        // Add monomial coefficients to combined_coeffs
                        combined_coeffs[0] = combined_coeffs[0].add(batch[inst].mul(a0));
                        combined_coeffs[1] = combined_coeffs[1].add(batch[inst].mul(a1));
                        combined_coeffs[2] = combined_coeffs[2].add(batch[inst].mul(a2_coeff));
                    }
                }
                dbg_inst_p0[5] = combined_coeffs[0];
                dbg_inst_p1[5] = combined_coeffs[1];

                if (debug_r5) {
                    const e0 = combined_coeffs[0].toBytes();
                    const e1 = combined_coeffs[1].toBytes();
                    dbg("  [R5_DBG] after inst5: c[0]_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}] c[1]_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                        e0[0], e0[1], e0[2], e0[3], e0[4], e0[5], e0[6], e0[7],
                        e1[0], e1[1], e1[2], e1[3], e1[4], e1[5], e1[6], e1[7],
                    });
                    // In monomial form, p(0)+p(1) = 2*c0 + c1 + c2 + ... + cd
                    var sum = combined_coeffs[0].add(combined_coeffs[0]); // 2*c0
                    for (1..max_degree + 1) |ci| sum = sum.add(combined_coeffs[ci]); // + c1 + c2 + ... + cd
                    const sum_le = sum.toBytes();
                    const claim_le = current_batched_claim.toBytes();
                    dbg("  [R5_DBG] sum=e[0]+e[1]_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                        sum_le[0], sum_le[1], sum_le[2], sum_le[3], sum_le[4], sum_le[5], sum_le[6], sum_le[7],
                    });
                    dbg("  [R5_DBG] claim_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                        claim_le[0], claim_le[1], claim_le[2], claim_le[3], claim_le[4], claim_le[5], claim_le[6], claim_le[7],
                    });
                    // Also check each instance's expected contribution to sum
                    for (0..6) |ii| {
                        const ic_le = instance_claims[ii].toBytes();
                        const ba_le = batch[ii].toBytes();
                        dbg("  [R5_DBG] inst[{}] claim_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}] batch_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}] rounds={}\n", .{
                            ii,
                            ic_le[0], ic_le[1], ic_le[2], ic_le[3], ic_le[4], ic_le[5], ic_le[6], ic_le[7],
                            ba_le[0], ba_le[1], ba_le[2], ba_le[3], ba_le[4], ba_le[5], ba_le[6], ba_le[7],
                            num_rounds_arr[ii],
                        });
                    }
                    // Recompute expected batched claim for round 5
                    // At round 5, remaining_rounds = 13-5 = 8
                    // inst 0 (13 rounds): active, scale = 0
                    // inst 1 (8 rounds): remaining 8 > 8? no, so active, scale = 0
                    // inst 2 (8 rounds): active, scale = 0
                    // inst 3 (8 rounds): active, scale = 0
                    // inst 4 (8 rounds): active, scale = 0
                    // inst 5 (8 rounds): active, scale = 0
                    // All active! Batched claim = Σ batch[i] * instance_claims[i]
                    var expected_sum = F.zero();
                    for (0..6) |ii| {
                        expected_sum = expected_sum.add(batch[ii].mul(instance_claims[ii]));
                    }
                    const exp_le = expected_sum.toBytes();
                    dbg("  [R5_DBG] expected_batched_Σ(b*c)_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                        exp_le[0], exp_le[1], exp_le[2], exp_le[3], exp_le[4], exp_le[5], exp_le[6], exp_le[7],
                    });
                }

                // Debug: check sumcheck invariant p(0)+p(1)=claim for ALL rounds
                // In monomial form: p(0)+p(1) = 2*c0 + c1 + c2 + ... + cd
                if (comptime debug_verbose) {
                    var p01_sum = combined_coeffs[0].add(combined_coeffs[0]); // 2*c0
                    for (1..max_degree + 1) |cii| p01_sum = p01_sum.add(combined_coeffs[cii]);
                    const p01_match = p01_sum.eql(current_batched_claim);
                    if (!p01_match) {
                        dbg("  [S6P] R{} *** SUMCHECK INVARIANT VIOLATED *** p(0)+p(1) != claim\n", .{round});
                        const ps = p01_sum.toBytes();
                        const cb = current_batched_claim.toBytes();
                        dbg("    p(0)+p(1)_LE=[", .{});
                        for (0..32) |bi| dbg("{x:0>2}", .{ps[bi]});
                        dbg("]\n    claim_LE=[", .{});
                        for (0..32) |bi| dbg("{x:0>2}", .{cb[bi]});
                        dbg("]\n", .{});
                        // Print each instance's contribution and per-instance p(0)+p(1) check
                        for (0..6) |di| {
                            const di_claim = instance_claims[di].toBytes();
                            dbg("    inst[{}] claim_LE=[", .{di});
                            for (0..32) |bi| dbg("{x:0>2}", .{di_claim[bi]});
                            dbg("] active={} rounds={}\n", .{@as(u8, if (inst_active[di]) 1 else 0), num_rounds_arr[di]});
                        }
                        // Recompute expected batched claim from per-instance claims
                        var recomp = F.zero();
                        for (0..6) |di| {
                            if (inst_active[di]) {
                                recomp = recomp.add(batch[di].mul(instance_claims[di]));
                            } else {
                                const scaled = sumcheck_helpers.inactiveContribution(F, input_claims[di], remaining_rounds, num_rounds_arr[di]);
                                recomp = recomp.add(batch[di].mul(scaled).add(batch[di].mul(scaled)));
                            }
                        }
                        const rc_le = recomp.toBytes();
                        dbg("    recomputed_LE=[", .{});
                        for (0..32) |bi| dbg("{x:0>2}", .{rc_le[bi]});
                        dbg("] match_claim={}\n", .{@as(u8, if (recomp.eql(current_batched_claim)) 1 else 0)});
                        // Per-instance p(0)+p(1) vs batch*claim check using cumulative deltas
                        var prev_p0 = F.zero();
                        var prev_p1 = F.zero();
                        for (0..6) |di| {
                            const inst_p0 = dbg_inst_p0[di].sub(prev_p0);
                            const inst_p1 = dbg_inst_p1[di].sub(prev_p1);
                            const inst_sum = inst_p0.add(inst_p1);
                            const expected_contrib = batch[di].mul(instance_claims[di]);
                            if (!inst_sum.eql(expected_contrib)) {
                                const is_le = inst_sum.toBytes();
                                const ex_le = expected_contrib.toBytes();
                                dbg("    *** MISMATCH inst[{}]: batch*(p0+p1)_LE=[", .{di});
                                for (0..32) |bi| dbg("{x:0>2}", .{is_le[bi]});
                                dbg("] batch*claim_LE=[", .{});
                                for (0..32) |bi| dbg("{x:0>2}", .{ex_le[bi]});
                                dbg("]\n", .{});
                            } else {
                                dbg("    inst[{}] p(0)+p(1)=claim OK\n", .{di});
                            }
                            prev_p0 = dbg_inst_p0[di];
                            prev_p1 = dbg_inst_p1[di];
                        }
                    }
                }

                // Debug: print monomial coefficients for round 7
                if (comptime debug_verbose) {
                    if (round == 7) {
                        dbg("  [S6P] R7 monomial coeffs:\n", .{});
                        for (0..max_degree + 1) |ci_idx| {
                            const ci_le = combined_coeffs[ci_idx].toBytes();
                            dbg("    c[{}]=[", .{ci_idx});
                            for (0..32) |bi| dbg("{x:0>2}", .{ci_le[bi]});
                            dbg("]\n", .{});
                        }
                        // p(0)+p(1) = 2*c0 + c1 + c2 + ... + cd
                        var sum01 = combined_coeffs[0].add(combined_coeffs[0]);
                        for (1..max_degree + 1) |ci_idx| sum01 = sum01.add(combined_coeffs[ci_idx]);
                        const sum_le = sum01.toBytes();
                        const hint_le = current_batched_claim.toBytes();
                        dbg("    p(0)+p(1)=[", .{});
                        for (0..32) |bi| dbg("{x:0>2}", .{sum_le[bi]});
                        dbg("]\n    hint    =[", .{});
                        for (0..32) |bi| dbg("{x:0>2}", .{hint_le[bi]});
                        dbg("]\n    match={}\n", .{sum01.eql(current_batched_claim)});
                    }
                }

                if (bench_s6) s6_t_compute[5] += s6_timer.read();
                if (bench_s6) s6_timer.reset();
                // Compress: strip c1 (linear term) from monomial coefficients
                // compressed = [c0, c2, c3, ..., c_d] (same as Jolt's UniPoly::compress)
                const compressed = try self.allocator.alloc(F, max_degree);
                defer self.allocator.free(compressed);
                compressed[0] = combined_coeffs[0]; // c0
                for (1..max_degree) |ci_idx| {
                    compressed[ci_idx] = combined_coeffs[ci_idx + 1]; // c2, c3, ..., c_d
                }

                // Debug: print compressed coefficients LE for ALL rounds
                if (comptime debug_verbose) {
                    var c_idx: usize = 0;
                    while (c_idx < compressed.len) : (c_idx += 1) {
                        const le = compressed[c_idx].toBytes();
                        dbg("  [S6P] R{} coeff[{}]=[", .{ round, c_idx });
                        for (0..32) |bi| dbg("{x:0>2}", .{le[bi]});
                        dbg("]\n", .{});
                    }
                }

                const coeffs = try self.allocator.alloc(F, num_compressed);
                for (0..num_compressed) |j| {
                    coeffs[j] = if (j < compressed.len) compressed[j] else F.zero();
                }

                try proof.compressed_polys.append(self.allocator, .{
                    .coeffs_except_linear_term = coeffs,
                    .allocator = self.allocator,
                });

                // Write diagnostic data to file for R0 - BEFORE appending to transcript
                if (comptime debug_verbose) {
                    if (round == 0) {
                        const diag_file = std.fs.cwd().createFile("/tmp/s6p_diag.bin", .{}) catch null;
                        if (diag_file) |f| {
                            defer f.close();
                            f.writeAll(&transcript.state) catch {};
                            for (0..num_compressed) |j| {
                                const le = coeffs[j].toBytes();
                                f.writeAll(&le) catch {};
                            }
                        }
                    }
                }

                transcript.appendScalars("sumcheck_poly", coeffs[0..num_compressed]);

                // Dump transcript state AFTER appending R0 polynomial
                if (comptime debug_verbose) {
                    if (round == 0) {
                        const diag_after = std.fs.cwd().createFile("/tmp/s6p_state_after_r0.bin", .{}) catch null;
                        if (diag_after) |fa| {
                            defer fa.close();
                            fa.writeAll(&transcript.state) catch {};
                            var nr_buf: [4]u8 = undefined;
                            std.mem.writeInt(u32, &nr_buf, transcript.n_rounds, .little);
                            fa.writeAll(&nr_buf) catch {};
                        }
                    }
                }

                const challenge = transcript.challengeScalar();
                challenges[round] = challenge;

                // Write R0 challenge to diagnostic file
                if (comptime debug_verbose) {
                    if (round == 0) {
                        const diag2 = std.fs.cwd().createFile("/tmp/s6p_r0_challenge.bin", .{}) catch null;
                        if (diag2) |f2| {
                            defer f2.close();
                            const ch_le = challenge.toBytes();
                            f2.writeAll(&ch_le) catch {};
                        }
                    }
                }

                // Evaluate combined polynomial at challenge using evalFromHintGeneral
                current_batched_claim = UniPoly(F).evalFromHintGeneral(coeffs[0..num_compressed], current_batched_claim, challenge);

                if (comptime debug_verbose) {
                    // Verify: directly evaluate combined_coeffs at challenge via Horner
                    var direct_eval = combined_coeffs[max_degree];
                    {
                        var ci_rev = max_degree;
                        while (ci_rev > 0) {
                            ci_rev -= 1;
                            direct_eval = direct_eval.mul(challenge).add(combined_coeffs[ci_rev]);
                        }
                    }
                    const efh_match = direct_eval.eql(current_batched_claim);
                    if (!efh_match) {
                        const efh_le = direct_eval.toBytes();
                        const vdm_le = current_batched_claim.toBytes();
                        dbg("  [S6P] R{} EVAL_MISMATCH! direct_eval=[", .{round});
                        for (0..32) |bi| dbg("{x:0>2}", .{efh_le[bi]});
                        dbg("]\n  [S6P] R{} EVAL_MISMATCH! evalFromHint=[", .{round});
                        for (0..32) |bi| dbg("{x:0>2}", .{vdm_le[bi]});
                        dbg("]\n", .{});
                        dbg("  [S6P] R{} num_compressed={}, compressed.len={}\n", .{ round, num_compressed, compressed.len });
                    }
                    dbg("  [S6P] R{} efh_match={}\n", .{ round, @intFromBool(efh_match) });
                }

                if (comptime debug_verbose) {
                    const ch_le = challenge.toBytes();
                    const cl_le = current_batched_claim.toBytes();
                    dbg("  [S6P] R{} challenge_LE=[", .{round});
                    for (0..32) |bi| dbg("{x:0>2}", .{ch_le[bi]});
                    dbg("]\n", .{});
                    dbg("  [S6P] R{} new_claim_LE=[", .{round});
                    for (0..32) |bi| dbg("{x:0>2}", .{cl_le[bi]});
                    dbg("]\n", .{});
                }

                if (bench_s6) s6_t_transcript += s6_timer.read();
                // Update per-instance claims from CACHED round polys and bind challenge
                // Instance 0: BytecodeReadRaf
                if (bench_s6) s6_timer.reset();
                if (inst_active[0]) {
                    if (bytecode_prover.phase == 0) {
                        // Phase 1: degree-2 poly, p(r) = a0 + a1*r + a2*r^2
                        const bc_a0 = cached_bc_phase1_coeffs[0];
                        const bc_a1 = cached_bc_phase1_coeffs[1];
                        const bc_a2 = cached_bc_phase1_coeffs[2];
                        instance_claims[0] = bc_a0.add(challenge.mul(bc_a1.add(challenge.mul(bc_a2))));
                        if (comptime debug_verbose) {
                            const ic_le = instance_claims[0].toBytes();
                            dbg("  [S6P] R{} inst0_from_poly_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                                round, ic_le[0], ic_le[1], ic_le[2], ic_le[3], ic_le[4], ic_le[5], ic_le[6], ic_le[7],
                            });
                        }
                        bytecode_addr_challenges[bytecode_prover.addr_rounds_done] = challenge;
                        bytecode_prover.bindChallengePhase1(challenge, cached_bc_phase1_per_stage);
                        if (comptime debug_verbose) {
                            // Check invariant: instance_claims[0] == Σ gamma^s * stage_claims[s]
                            var agg_check = F.zero();
                            for (0..5) |si| {
                                agg_check = agg_check.add(bytecode_prover.gamma_powers[si].mul(bytecode_prover.stage_claims[si]));
                            }
                            const ac_le = agg_check.toBytes();
                            const ic_le2 = instance_claims[0].toBytes();
                            for (0..5) |si| {
                                const scl = bytecode_prover.stage_claims[si].toBytes();
                                dbg("[INVARIANT_CHECK] R{} stage[{}]_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                                    round, si,
                                    scl[0], scl[1], scl[2], scl[3], scl[4], scl[5], scl[6], scl[7],
                                });
                            }
                            dbg("[INVARIANT_CHECK] R{} agg_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}] inst0_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}] match={}\n", .{
                                round,
                                ac_le[0], ac_le[1], ac_le[2], ac_le[3], ac_le[4], ac_le[5], ac_le[6], ac_le[7],
                                ic_le2[0], ic_le2[1], ic_le2[2], ic_le2[3], ic_le2[4], ic_le2[5], ic_le2[6], ic_le2[7],
                                @as(u8, if (agg_check.eql(instance_claims[0])) 1 else 0),
                            });
                            const bc_a0_ = cached_bc_phase1_coeffs[0];
                            const bc_a1_ = cached_bc_phase1_coeffs[1];
                            const bc_a2_ = cached_bc_phase1_coeffs[2];
                            const manual_eval = bc_a0_.add(challenge.mul(bc_a1_.add(challenge.mul(bc_a2_))));
                            const me_le = manual_eval.toBytes();
                            dbg("[INVARIANT_CHECK] R{} manual_eval_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}] match_inst={}\n", .{
                                round,
                                me_le[0], me_le[1], me_le[2], me_le[3], me_le[4], me_le[5], me_le[6], me_le[7],
                                @as(u8, if (manual_eval.eql(instance_claims[0])) 1 else 0),
                            });
                        }
                        if (bytecode_prover.addr_rounds_done == bytecode_log_k) {
                            if (comptime debug_verbose) {
                                // BEFORE transition: check Σ_s gamma^s * stage_claims[s] vs instance_claims[0]
                                var agg_from_stages = F.zero();
                                for (0..5) |si| {
                                    agg_from_stages = agg_from_stages.add(bytecode_prover.gamma_powers[si].mul(bytecode_prover.stage_claims[si]));
                                }
                                const afs_le = agg_from_stages.toBytes();
                                const ic0_le = instance_claims[0].toBytes();
                                dbg("[PHASE_TRANSITION_PRE] agg_stages_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}] inst0_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}] match={}\n", .{
                                    afs_le[0], afs_le[1], afs_le[2], afs_le[3], afs_le[4], afs_le[5], afs_le[6], afs_le[7],
                                    ic0_le[0], ic0_le[1], ic0_le[2], ic0_le[3], ic0_le[4], ic0_le[5], ic0_le[6], ic0_le[7],
                                    @as(u8, if (agg_from_stages.eql(instance_claims[0])) 1 else 0),
                                });
                                for (0..5) |si| {
                                    const sc_le2 = bytecode_prover.stage_claims[si].toBytes();
                                    dbg("[PHASE_TRANSITION_PRE] stage[{}]_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                                        si, sc_le2[0], sc_le2[1], sc_le2[2], sc_le2[3], sc_le2[4], sc_le2[5], sc_le2[6], sc_le2[7],
                                    });
                                }
                            }
                            try bytecode_prover.transitionToPhase2(bytecode_addr_challenges);
                            if (comptime debug_verbose) {
                                // After transition, check Phase 2 polynomial sum
                                // (combined[] replaced by GruenSplitEq — full verification
                                // requires materializing eq tables, skipped in debug mode)
                                dbg("[PHASE_TRANSITION] inst0 transition complete\n", .{});
                            }
                        }
                    } else {
                        // Phase 2: evaluate from Toom-Cook cached evals
                        // Convert to monomials, evaluate at challenge, free
                        const bc_p2_mono = try UniPoly(F).fromEvalsToom(self.allocator, cached_bc_phase2.?);
                        defer self.allocator.free(bc_p2_mono);
                        var bc_p2_val = F.zero();
                        var x_pow = F.one();
                        for (bc_p2_mono) |coeff| {
                            bc_p2_val = bc_p2_val.add(coeff.mul(x_pow));
                            x_pow = x_pow.mul(challenge);
                        }
                        instance_claims[0] = bc_p2_val;
                        self.allocator.free(cached_bc_phase2.?);
                        cached_bc_phase2 = null;
                        bytecode_prover.bindChallengePhase2(challenge);

                    }
                }

                if (bench_s6) s6_t_bind[0] += s6_timer.read();
                // Instance 1: Booleanity (real prover)
                if (bench_s6) s6_timer.reset();
                if (inst_active[1]) {
                    if (cached_booleanity) |polys| {
                        // Evaluate degree-3 poly at challenge from Vandermonde [p(0), p(1), p(2), p(3)]
                        const evals_arr = [4]F{ polys[0], polys[1], polys[2], polys[3] };
                        instance_claims[1] = UniPoly(F).evalFromEvalsDeg3(evals_arr, challenge);
                        self.allocator.free(polys);
                        cached_booleanity = null;
                    }
                    try booleanity_prover.bindChallenge(challenge);
                    if (comptime debug_verbose) {
                        if (booleanity_prover.round == booleanity_prover.log_k_chunk) {
                            const ic1_be = instance_claims[1].toBytesBE();
                            dbg("[BOOL_TRANSITION] inst_claim[1] after Ph1 LE=[{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}]\n", .{
                                ic1_be[31], ic1_be[30], ic1_be[29], ic1_be[28], ic1_be[27], ic1_be[26], ic1_be[25], ic1_be[24],
                            });
                        }
                    }
                }

                if (bench_s6) s6_t_bind[1] += s6_timer.read();
                // Instance 2: HammingBooleanity
                if (bench_s6) s6_timer.reset();
                if (inst_active[2]) {
                    instance_claims[2] = UniPoly(F).evalFromEvalsDeg3(cached_hamming, challenge);
                    try hamming_prover.bindChallenge(challenge);
                }

                if (bench_s6) s6_t_bind[2] += s6_timer.read();
                // Instance 3: RamRaVirtual
                if (bench_s6) s6_timer.reset();
                if (inst_active[3]) {
                    // Monomial coefficients — evaluate via Horner's method
                    const ram_mono = cached_ram_ra.?;
                    var ram_val = ram_mono[ram_mono.len - 1];
                    var ram_ci: usize = ram_mono.len - 1;
                    while (ram_ci > 0) {
                        ram_ci -= 1;
                        ram_val = ram_val.mul(challenge).add(ram_mono[ram_ci]);
                    }
                    instance_claims[3] = ram_val;
                    self.allocator.free(ram_mono);
                    cached_ram_ra = null;
                    try ram_ra_prover.bindChallenge(challenge);
                }

                if (bench_s6) s6_t_bind[3] += s6_timer.read();
                // Instance 4: LookupsRaVirtual
                if (bench_s6) s6_timer.reset();
                if (inst_active[4]) {
                    // Monomial coefficients — evaluate via Horner's method
                    const mono = cached_lookups_ra.?;
                    var val = mono[mono.len - 1];
                    var ci: usize = mono.len - 1;
                    while (ci > 0) {
                        ci -= 1;
                        val = val.mul(challenge).add(mono[ci]);
                    }
                    instance_claims[4] = val;
                    self.allocator.free(mono);
                    cached_lookups_ra = null;
                    try lookups_ra_prover.bindChallenge(challenge);
                }

                if (bench_s6) s6_t_bind[4] += s6_timer.read();
                // Instance 5: IncClaimReduction
                if (bench_s6) s6_timer.reset();
                if (inst_active[5]) {
                    instance_claims[5] = UniPoly(F).evalFromEvalsDeg2(cached_inc, challenge);

                    try inc_prover.bindChallenge(challenge);
                }
                if (bench_s6) s6_t_bind[5] += s6_timer.read();

                // NOTE: Instance claims for inactive instances are NOT halved here.
                // In Zolt, instance_claims starts at the UNSCALED input_claims (not 2^offset-scaled),
                // and the inactive round contributions are computed directly from input_claims with
                // the correct power-of-2 scaling. When an instance first becomes active,
                // instance_claims[i] = input_claims[i] = the correct unscaled claim.
            }


            if (bench_s6) {
                const names = [6][]const u8{ "BcRaf", "Bool ", "Hamm ", "RamRa", "LkRa ", "Inc  " };
                var total_compute: u64 = 0;
                var total_bind: u64 = 0;
                for (0..6) |i| {
                    total_compute += s6_t_compute[i];
                    total_bind += s6_t_bind[i];
                }
                const s6_sumcheck_wall_ns: i128 = std.time.nanoTimestamp() - t_s6_overall_start;
                const toMsU = struct {
                    fn f(ns: u64) f64 {
                        return @as(f64, @floatFromInt(ns)) / 1_000_000.0;
                    }
                }.f;
                const toMsI = struct {
                    fn f(ns: i128) f64 {
                        return @as(f64, @floatFromInt(ns)) / 1_000_000.0;
                    }
                }.f;
                std.debug.print("\n    [STAGE6-BENCH] Sumcheck loop ({} rounds):\n", .{max_num_rounds});
                for (0..6) |i| {
                    std.debug.print("    [STAGE6-BENCH]   {s}: compute={d:7.1}ms  bind={d:7.1}ms  total={d:7.1}ms\n", .{
                        names[i],
                        toMsU(s6_t_compute[i]),
                        toMsU(s6_t_bind[i]),
                        toMsU(s6_t_compute[i] + s6_t_bind[i]),
                    });
                }
                std.debug.print("    [STAGE6-BENCH]   transcript+compress: {d:7.1}ms\n", .{toMsU(s6_t_transcript)});
                std.debug.print("    [STAGE6-BENCH]   Sumcheck TOTAL: compute={d:7.1}ms  bind={d:7.1}ms  transcript={d:7.1}ms\n", .{
                    toMsU(total_compute),
                    toMsU(total_bind),
                    toMsU(s6_t_transcript),
                });
                std.debug.print("    [STAGE6-BENCH]   Stage 6 overall wall: {d:7.1}ms\n", .{toMsI(s6_sumcheck_wall_ns)});
            }

            // ====================================================================
            // Extract opening claims from all real provers
            // ====================================================================

            const inc_opening = inc_prover.openingClaims();
            const ram_inc_claim = inc_opening.ram_inc;
            const rd_inc_claim = inc_opening.rd_inc;
            if (comptime debug_verbose) {
                const eq_r = inc_prover.eq_ram[0];
                const eq_d = inc_prover.eq_rd[0];
                const recomp = ram_inc_claim.mul(eq_r).add(inc_gamma2.mul(rd_inc_claim.mul(eq_d)));
                const er_be = eq_r.toBytesBE();
                const ed_be = eq_d.toBytesBE();
                const rc_be = recomp.toBytesBE();
                dbg("[INC_DEBUG] eq_ram[0]_LE=[", .{});
                for (0..32) |bi| dbg("{x:0>2}", .{er_be[31 - bi]});
                dbg("]\n  eq_rd[0]_LE=[", .{});
                for (0..32) |bi| dbg("{x:0>2}", .{ed_be[31 - bi]});
                dbg("]\n  recomp_LE=[", .{});
                for (0..32) |bi| dbg("{x:0>2}", .{rc_be[31 - bi]});
                dbg("]\n  instance[5]_LE=[", .{});
                const i5_be = instance_claims[5].toBytesBE();
                for (0..32) |bi| dbg("{x:0>2}", .{i5_be[31 - bi]});
                dbg("]\n", .{});
            }

            const hamming_weight_claim = hamming_prover.openingClaim();

            const bytecode_ra_claims = try bytecode_prover.getOpeningClaims(self.allocator);
            if (comptime debug_verbose) {
                dbg("[S6P] Bytecode RA claims (d={d}):\n", .{bytecode_d});
                for (0..bytecode_d) |i| {
                    const be = bytecode_ra_claims[i].toBytesBE();
                    dbg("  ra[{d}]_LE=[", .{i});
                    for (0..32) |bi| dbg("{x:0>2}", .{be[31 - bi]});
                    dbg("]\n", .{});
                }
                // Compute combined[0] from GruenSplitEq final scalars + entry correction
                {
                    var comb0 = bytecode_prover.entry_correction_scalar;
                    for (0..5) |s| {
                        comb0 = comb0.add(bytecode_prover.bound_vals_phase2[s].mul(bytecode_prover.stage_gruen_eqs[s].?.current_scalar));
                    }
                    const comb0_be = comb0.toBytesBE();
                    dbg("  combined[0]_LE=[", .{});
                    for (0..32) |bi| dbg("{x:0>2}", .{comb0_be[31 - bi]});
                    dbg("]\n", .{});
                    // Compute val_from_prover = combined[0] * Π ra[i]
                    var val_ra_prod = comb0;
                    for (0..bytecode_d) |i| {
                        val_ra_prod = val_ra_prod.mul(bytecode_ra_claims[i]);
                    }
                    const vrp_be = val_ra_prod.toBytesBE();
                    dbg("  combined[0]*Π_ra_LE=[", .{});
                    for (0..32) |bi| dbg("{x:0>2}", .{vrp_be[31 - bi]});
                    dbg("]\n", .{});
                }
                // Compare with instance_claims[0]
                const ic0_be = instance_claims[0].toBytesBE();
                dbg("  instance_claims[0]_LE=[", .{});
                for (0..32) |bi| dbg("{x:0>2}", .{ic0_be[31 - bi]});
                dbg("]\n", .{});

                // === PER-STAGE DECOMPOSITION ===
                // Recompute combined[0] = Σ_s bound_vals[s] * eq_mle(r_cycle_s, r_cycle_prime)
                // r_cycle_prime = reversed Phase 2 challenges (matching Jolt's normalize_opening_point)
                const cycle_start = bytecode_log_k;
                var r_cycle_prime = try self.allocator.alloc(F, n_cycle_vars);
                defer self.allocator.free(r_cycle_prime);
                for (0..n_cycle_vars) |ci| {
                    r_cycle_prime[ci] = challenges[cycle_start + n_cycle_vars - 1 - ci];
                }
                // Print r_cycle_prime
                dbg("[DECOMP] r_cycle_prime (reversed cycle challenges, BE):\n", .{});
                for (0..@min(4, n_cycle_vars)) |ci| {
                    const rcp_be = r_cycle_prime[ci].toBytesBE();
                    dbg("  r_cycle_prime[{}]_LE=[", .{ci});
                    for (0..8) |bi| dbg("{x:0>2}", .{rcp_be[31 - bi]});
                    dbg("]\n", .{});
                }

                var decomp_sum = F.zero();
                for (0..5) |s| {
                    // Compute eq_mle(r_cycle_s, r_cycle_prime) = Π_i (r_s[i]*r_p[i] + (1-r_s[i])(1-r_p[i]))
                    // Both r_cycle_s and r_cycle_prime are in BE order
                    var eq_mle = F.one();
                    const r_s = bytecode_prover.stage_r_cycles[s];
                    for (0..n_cycle_vars) |ci| {
                        const a = r_s[ci];
                        const b = r_cycle_prime[ci];
                        // eq term: a*b + (1-a)*(1-b) = 1 - a - b + 2*a*b
                        const ab = a.mul(b);
                        const term = F.one().sub(a).sub(b).add(ab).add(ab);
                        eq_mle = eq_mle.mul(term);
                    }

                    const bv = bytecode_prover.bound_vals_stored[s];
                    const stage_contrib = bv.mul(eq_mle);
                    decomp_sum = decomp_sum.add(stage_contrib);

                    const bv_be = bv.toBytesBE();
                    const eq_be = eq_mle.toBytesBE();
                    const sc_be = stage_contrib.toBytesBE();
                    dbg("[DECOMP] stage[{}]: bound_val_LE=[", .{s});
                    for (0..8) |bi| dbg("{x:0>2}", .{bv_be[31 - bi]});
                    dbg("] eq_mle_LE=[", .{});
                    for (0..8) |bi| dbg("{x:0>2}", .{eq_be[31 - bi]});
                    dbg("] contrib_LE=[", .{});
                    for (0..8) |bi| dbg("{x:0>2}", .{sc_be[31 - bi]});
                    dbg("]\n", .{});
                }
                const ds_be = decomp_sum.toBytesBE();
                dbg("[DECOMP] val_sum_LE=[", .{});
                for (0..8) |bi| dbg("{x:0>2}", .{ds_be[31 - bi]});
                dbg("]\n", .{});

                // Also print val_with_raf bound values (without gamma)
                for (0..5) |s| {
                    const vwr = bytecode_prover.bound_vals_stored[s];
                    const gp = bytecode_prover.gamma_powers[s];
                    // val_with_raf[s][0] = bound_vals[s] / gamma[s]
                    // Print bound_val directly (it already includes gamma)
                    const vwr_be = vwr.toBytesBE();
                    const gp_be = gp.toBytesBE();
                    dbg("[DECOMP] stage[{}]: gamma_LE=[", .{s});
                    for (0..8) |bi| dbg("{x:0>2}", .{gp_be[31 - bi]});
                    dbg("] gamma*val_LE=[", .{});
                    for (0..8) |bi| dbg("{x:0>2}", .{vwr_be[31 - bi]});
                    dbg("]\n", .{});
                }
            }

            const ram_ra_virtual_claims = try ram_ra_prover.getOpeningClaims(self.allocator);

            const instruction_ra_virtual_claims = try lookups_ra_prover.getOpeningClaims(self.allocator, lookups_ra_gamma_powers);

            // Get booleanity claims directly from the prover's final H state.
            // After all Phase 2 rounds, H[i][0] = ra_i(ρ_addr, ρ_cycle).
            const booleanity_ra_claims = try booleanity_prover.getBooleanityClaims(self.allocator);
            if (comptime debug_verbose) {
                const total_booleanity_polys = instruction_d + bytecode_d + ram_d;
                dbg("[STAGE6] Booleanity claims from H final state:\n", .{});
                for (0..@min(5, total_booleanity_polys)) |i| {
                    const brc_be = booleanity_ra_claims[i].toBytesBE();
                    dbg("  bool_claim[{}]_LE=[", .{i});
                    for (0..8) |bi| dbg("{x:0>2}", .{brc_be[31 - bi]});
                    dbg("]\n", .{});
                }
            }

            if (comptime debug_verbose) {
            // Debug: compute what the verifier would compute for Instance 1 (Booleanity)
            // expected = eq(challenges, combined_r) * Σ gamma^{2i} * (ra_i^2 - ra_i)
            // combined_r = r_address.reversed ++ r_cycle.reversed
            // In Jolt: r_address reversed means the original r_address (from params) reversed.
            // The booleanity params store r_address in LE format. "reversed" in Jolt means
            // going from LE to reversed-LE. But actually Jolt stores r_address and r_cycle in a
            // specific order from BooleanitySumcheckParams::new, and then reverses them.
            {
                const total_booleanity_polys = instruction_d + bytecode_d + ram_d;
                // Jolt's BooleanitySumcheckParams stores r_address and r_cycle from Stage 5.
                // r_address = last log_k_chunk challenges from the InstructionReadRaf address.
                // r_cycle = cycle challenges from InstructionReadRaf.
                // The verifier reverses both: combined_r = rev(r_address) ++ rev(r_cycle).
                //
                // In our code:
                // r_address_bool_le = [ch[log_k-1], ch[log_k-2], ..., ch[0]] (from stage5 MSB-first)
                // But the Jolt params store them in a specific order based on Stage 5's binding.
                // Jolt's BooleanitySumcheckParams::new extracts r_address from the accumulator
                // which stores them in the binding order from Stage 5 InstructionReadRaf.
                //
                // For now, let me compute the expected claim using the data I have:
                // The sumcheck challenges for booleanity rounds are challenges[0..log_k+n_cycle].
                // Booleanity uses rounds 0..log_k for address, log_k..log_k+n_cycle for cycle.
                //
                // The actual output_claim from the sumcheck should be:
                //   eq_r_r * eq_cycle_final * Σ gamma^{2i} * (H[i][0]^2 - H[i][0])
                // where eq_cycle_final is what eq_cycle[0] becomes after all Phase 2 bindings.
                //
                // Let me just compute Σ gamma^{2i} * (ra_i^2 - ra_i) and the eq parts.
                var sum_gamma_ra = F.zero();
                for (0..total_booleanity_polys) |i| {
                    const ra = booleanity_ra_claims[i];
                    sum_gamma_ra = sum_gamma_ra.add(booleanity_prover.gamma_powers_sq[i].mul(ra.mul(ra).sub(ra)));
                }
                // Also, get the actual eq values from the prover
                const bp_eq_r_r = booleanity_prover.eq_r_r;
                const bp_eq_cycle_final = booleanity_prover.gruen_eq_cycle.current_scalar;
                const actual_output = bp_eq_r_r.mul(bp_eq_cycle_final).mul(sum_gamma_ra);

                const sg_be = sum_gamma_ra.toBytesBE();
                const err_be = bp_eq_r_r.toBytesBE();
                const ecf_be = bp_eq_cycle_final.toBytesBE();
                const ao_be = actual_output.toBytesBE();
                dbg("[BOOL_VERIFY] sum_gamma_ra_LE=[", .{});
                for (0..8) |bi| dbg("{x:0>2}", .{sg_be[31 - bi]});
                dbg("]\n", .{});
                dbg("[BOOL_VERIFY] eq_r_r_LE=[", .{});
                for (0..8) |bi| dbg("{x:0>2}", .{err_be[31 - bi]});
                dbg("]\n", .{});
                dbg("[BOOL_VERIFY] eq_cycle_final_LE=[", .{});
                for (0..8) |bi| dbg("{x:0>2}", .{ecf_be[31 - bi]});
                dbg("]\n", .{});
                dbg("[BOOL_VERIFY] actual_output_LE=[", .{});
                for (0..8) |bi| dbg("{x:0>2}", .{ao_be[31 - bi]});
                dbg("]\n", .{});

                // Compare with instance_claims[1] (the sumcheck output claim for booleanity)
                const ic1_be = instance_claims[1].toBytesBE();
                dbg("[BOOL_VERIFY] instance_claims[1]_LE=[", .{});
                for (0..8) |bi| dbg("{x:0>2}", .{ic1_be[31 - bi]});
                dbg("]\n", .{});
                dbg("[BOOL_VERIFY] match={}\n", .{@intFromBool(actual_output.eql(instance_claims[1]))});

                // Now compute eq(challenges, combined_r) directly, the way the verifier does.
                // combined_r = rev(r_address_LE) ++ rev(r_cycle_LE)
                // r_address_LE (in Jolt) = last log_k_chunk elements of Stage5 addr reversed to LE
                // In our code: the ORIGINAL r_address_bool_le (before reversal in init) is the LE version.
                // After init() reversed it, booleanity_prover.r_address_le[m] = MSB at m=0.
                // To get Jolt's LE r_address, we need to reverse it back.
                // Then rev(r_address_LE) = booleanity_prover.r_address_le (as-is, since it was reversed to BE)
                //
                // combined_r_addr[m] = r_address_LE[log_k-1-m] = booleanity_prover.r_address_le[m]
                // combined_r_cycle[m] = r_cycle_LE[n_cycle-1-m]
                //
                // r_cycle_LE = lookups_ra_r_cycle (the original, before computeEqTable)
                // combined_r_cycle[m] = lookups_ra_r_cycle[n_cycle-1-m]
                //
                // eq(ch[m], combined_r[m]) for m < log_k:
                //   = eq(ch[m], booleanity_prover.r_address_le[m])
                // eq(ch[log_k+m], combined_r[log_k+m]) for m < n_cycle:
                //   = eq(ch[log_k+m], lookups_ra_r_cycle[n_cycle-1-m])
                {
                    const bool_start_round = max_num_rounds - num_rounds_arr[1];
                    dbg("[BOOL_VERIFY] bool_start_round={}, log_k={}, n_cycle={}\n", .{
                        bool_start_round, log_k_chunk, n_cycle_vars,
                    });

                    // Print ALL eq factors matching Jolt's format
                    // Jolt: combined_r = rev(r_address_LE) ++ rev(r_cycle_LE)
                    // Zolt: r_address_le[m] = MSB at 0 (reversed in init) = rev(r_address_LE)[m]
                    // Zolt: combined_r_cycle[m] = r_cycle_LE[n_cycle-1-m] = lookups_ra_r_cycle[n_cycle-1-m]
                    var eq_direct = F.one();
                    for (0..log_k_chunk) |m| {
                        const ch_val = challenges[bool_start_round + m];
                        const w_val = booleanity_prover.r_address_le[m];
                        const prod = ch_val.mul(w_val);
                        const eq_factor = F.one().sub(ch_val).sub(w_val).add(prod.add(prod));
                        eq_direct = eq_direct.mul(eq_factor);

                        const ch_be = ch_val.toBytesBE();
                        const w_be = w_val.toBytesBE();
                        const ef_be = eq_factor.toBytesBE();
                        dbg("[BOOL_EQ_ZOLT] idx={} sc=[", .{m});
                        for (0..8) |bi| dbg("{x:0>2}", .{ch_be[31 - bi]});
                        dbg("] cr=[", .{});
                        for (0..8) |bi| dbg("{x:0>2}", .{w_be[31 - bi]});
                        dbg("] eq_i=[", .{});
                        for (0..8) |bi| dbg("{x:0>2}", .{ef_be[31 - bi]});
                        dbg("]\n", .{});
                    }
                    for (0..n_cycle_vars) |m| {
                        const ch_val = challenges[bool_start_round + log_k_chunk + m];
                        // Jolt: combined_r_cycle[m] = rev(r_cycle_LE)[m] = r_cycle_LE[n-1-m]
                        // Since lookups_ra_r_cycle is BE (MSB at 0), and Jolt r_cycle_LE[n-1-m] = lookups[m]
                        const w_val = lookups_ra_r_cycle[m]; // direct index, no reversal
                        const prod = ch_val.mul(w_val);
                        const eq_factor = F.one().sub(ch_val).sub(w_val).add(prod.add(prod));
                        eq_direct = eq_direct.mul(eq_factor);

                        const ch_be = ch_val.toBytesBE();
                        const w_be = w_val.toBytesBE();
                        const ef_be = eq_factor.toBytesBE();
                        dbg("[BOOL_EQ_ZOLT] idx={} sc=[", .{log_k_chunk + m});
                        for (0..8) |bi| dbg("{x:0>2}", .{ch_be[31 - bi]});
                        dbg("] cr=[", .{});
                        for (0..8) |bi| dbg("{x:0>2}", .{w_be[31 - bi]});
                        dbg("] eq_i=[", .{});
                        for (0..8) |bi| dbg("{x:0>2}", .{ef_be[31 - bi]});
                        dbg("]\n", .{});
                    }

                    const eq_from_prover = bp_eq_r_r.mul(bp_eq_cycle_final);
                    const ed_be = eq_direct.toBytesBE();
                    const ep_be = eq_from_prover.toBytesBE();
                    dbg("[BOOL_VERIFY] eq_direct_LE=[", .{});
                    for (0..8) |bi| dbg("{x:0>2}", .{ed_be[31 - bi]});
                    dbg("]\n", .{});
                    dbg("[BOOL_VERIFY] eq_from_prover_LE=[", .{});
                    for (0..8) |bi| dbg("{x:0>2}", .{ep_be[31 - bi]});
                    dbg("]\n", .{});
                    dbg("[BOOL_VERIFY] eq_match={}\n", .{@intFromBool(eq_direct.eql(eq_from_prover))});
                }
            }
            } // end if (comptime debug_verbose) for BOOL_VERIFY

            if (comptime debug_verbose) {
            dbg("[STAGE6] Opening claims (full LE hex):\n", .{});
            {
                const be = ram_inc_claim.toBytesBE();
                dbg("  ram_inc_LE=[", .{});
                for (0..32) |bi| dbg("{x:0>2}", .{be[31 - bi]});
                dbg("]\n", .{});
            }
            {
                const be = rd_inc_claim.toBytesBE();
                dbg("  rd_inc_LE=[", .{});
                for (0..32) |bi| dbg("{x:0>2}", .{be[31 - bi]});
                dbg("]\n", .{});
            }
            {
                const be = hamming_weight_claim.toBytesBE();
                dbg("  hamming_weight_LE=[", .{});
                for (0..32) |bi| dbg("{x:0>2}", .{be[31 - bi]});
                dbg("]\n", .{});
            }
            for (0..bytecode_d) |i| {
                const be = bytecode_ra_claims[i].toBytesBE();
                dbg("  bytecode_ra[{d}]_LE=[", .{i});
                for (0..32) |bi| dbg("{x:0>2}", .{be[31 - bi]});
                dbg("]\n", .{});
            }
            {
                const be = ram_ra_virtual_claims[0].toBytesBE();
                dbg("  ram_ra_virtual[0]_LE=[", .{});
                for (0..32) |bi| dbg("{x:0>2}", .{be[31 - bi]});
                dbg("]\n", .{});
            }
            {
                const be = instruction_ra_virtual_claims[0].toBytesBE();
                dbg("  instruction_ra_virtual[0]_LE=[", .{});
                for (0..32) |bi| dbg("{x:0>2}", .{be[31 - bi]});
                dbg("]\n", .{});
            }
            for (0..3) |i| {
                const be = booleanity_ra_claims[i].toBytesBE();
                dbg("  booleanity_ra[{d}]_LE=[", .{i});
                for (0..32) |bi| dbg("{x:0>2}", .{be[31 - bi]});
                dbg("]\n", .{});
            }

            // Consistency check: instance_claims[0] should equal val * Π ra[i]
            // where val = GruenSplitEq final scalar sum + entry correction
            {
                var bc_combined_val = bytecode_prover.entry_correction_scalar;
                for (0..5) |s| {
                    bc_combined_val = bc_combined_val.add(bytecode_prover.bound_vals_phase2[s].mul(bytecode_prover.stage_gruen_eqs[s].?.current_scalar));
                }
                var bc_ra_prod = F.one();
                for (bytecode_ra_claims) |c| bc_ra_prod = bc_ra_prod.mul(c);
                const bc_recomputed = bc_combined_val.mul(bc_ra_prod);
                dbg("[STAGE6] Consistency check Instance 0:\n", .{});
                // Print combined[0] as LE hex for comparison with Jolt's "val (sum)"
                const cval_be = bc_combined_val.toBytesBE();
                dbg("  combined[0]_LE=[", .{});
                for (0..32) |bi| dbg("{x:0>2}", .{cval_be[31 - bi]});
                dbg("]\n", .{});
                // Print ra claims
                for (0..bytecode_d) |i| {
                    const ra_be = bytecode_ra_claims[i].toBytesBE();
                    dbg("  ra[{}]_LE=[", .{i});
                    for (0..32) |bi| dbg("{x:0>2}", .{ra_be[31 - bi]});
                    dbg("]\n", .{});
                }
                dbg("  recomputed_LE=[", .{});
                const rc_be = bc_recomputed.toBytesBE();
                for (0..32) |bi| dbg("{x:0>2}", .{rc_be[31 - bi]});
                dbg("]\n", .{});
                dbg("  instance[0]_LE=[", .{});
                const ic_be = instance_claims[0].toBytesBE();
                for (0..32) |bi| dbg("{x:0>2}", .{ic_be[31 - bi]});
                dbg("]\n", .{});
                dbg("  match = {}\n", .{@as(u8, if (std.mem.eql(u8, &bc_recomputed.toBytesBE(), &instance_claims[0].toBytesBE())) 1 else 0)});
            }

            // Consistency check Instance 5 (IncClaimReduction):
            // expected = ram_inc * eq_ram_combined(rho) + gamma^2 * rd_inc * eq_rd_combined(rho)
            // where rho = reversed sumcheck challenges (opening point in BE)
            {
                // Build opening point: reverse challenges for LE->BE
                var opening_point = try self.allocator.alloc(F, n_cycle_vars);
                defer self.allocator.free(opening_point);
                // Instance 5 has n_cycle_vars rounds; offset = max_num_rounds - n_cycle_vars
                const inc_offset = max_num_rounds - n_cycle_vars;
                for (0..n_cycle_vars) |i| {
                    opening_point[n_cycle_vars - 1 - i] = challenges[inc_offset + i];
                }

                // Compute eq evaluations at opening_point vs each r_cycle
                // eq(a, b) = prod_i (a[i]*b[i] + (1-a[i])*(1-b[i]))
                const computeEqEval = struct {
                    fn eval(a: []const F, b: []const F) F {
                        var result = F.one();
                        for (0..a.len) |i| {
                            const prod = a[i].mul(b[i]);
                            const sum = a[i].add(b[i]);
                            result = result.mul(prod.add(prod).add(F.one()).sub(sum));
                        }
                        return result;
                    }
                }.eval;

                const eq_r2 = computeEqEval(opening_point, r_cycle_inc_ram_rwc);
                const eq_r4 = computeEqEval(opening_point, r_cycle_inc_ram_val);
                const eq_s4 = computeEqEval(opening_point, r_cycle_bc4_regs_rwc);
                const eq_s5 = computeEqEval(opening_point, r_cycle_bc5_regs_val);

                const eq_ram_combined = eq_r2.add(inc_gamma.mul(eq_r4));
                const eq_rd_combined = eq_s4.add(inc_gamma.mul(eq_s5));

                const expected_inc = ram_inc_claim.mul(eq_ram_combined).add(inc_gamma2.mul(rd_inc_claim.mul(eq_rd_combined)));

                dbg("[STAGE6] Inc consistency check:\n", .{});
                dbg("  ram_inc_claim_LE=[", .{});
                const ric_be = ram_inc_claim.toBytesBE();
                for (0..32) |bi| dbg("{x:0>2}", .{ric_be[31 - bi]});
                dbg("]\n", .{});
                dbg("  rd_inc_claim_LE=[", .{});
                const rdc_be = rd_inc_claim.toBytesBE();
                for (0..32) |bi| dbg("{x:0>2}", .{rdc_be[31 - bi]});
                dbg("]\n", .{});
                dbg("  eq_r2_LE=[", .{});
                const er2 = eq_r2.toBytesBE();
                for (0..32) |bi| dbg("{x:0>2}", .{er2[31 - bi]});
                dbg("]\n", .{});
                dbg("  eq_r4_LE=[", .{});
                const er4 = eq_r4.toBytesBE();
                for (0..32) |bi| dbg("{x:0>2}", .{er4[31 - bi]});
                dbg("]\n", .{});
                dbg("  eq_s4_LE=[", .{});
                const es4 = eq_s4.toBytesBE();
                for (0..32) |bi| dbg("{x:0>2}", .{es4[31 - bi]});
                dbg("]\n", .{});
                dbg("  eq_s5_LE=[", .{});
                const es5 = eq_s5.toBytesBE();
                for (0..32) |bi| dbg("{x:0>2}", .{es5[31 - bi]});
                dbg("]\n", .{});
                dbg("  expected_inc_LE=[", .{});
                const eibc = expected_inc.toBytesBE();
                for (0..32) |bi| dbg("{x:0>2}", .{eibc[31 - bi]});
                dbg("]\n", .{});
                dbg("  instance[5]_LE=[", .{});
                const i5_be = instance_claims[5].toBytesBE();
                for (0..32) |bi| dbg("{x:0>2}", .{i5_be[31 - bi]});
                dbg("]\n", .{});
                dbg("  match = {}\n", .{@as(u8, if (std.mem.eql(u8, &expected_inc.toBytesBE(), &instance_claims[5].toBytesBE())) 1 else 0)});

                // Also print the r_cycle values themselves
                dbg("  r_cycle_inc_ram_rwc[0]_LE=[", .{});
                const rr0 = r_cycle_inc_ram_rwc[0].toBytesBE();
                for (0..32) |bi| dbg("{x:0>2}", .{rr0[31 - bi]});
                dbg("]\n", .{});
                dbg("  r_cycle_inc_ram_val[0]_LE=[", .{});
                const rv0 = r_cycle_inc_ram_val[0].toBytesBE();
                for (0..32) |bi| dbg("{x:0>2}", .{rv0[31 - bi]});
                dbg("]\n", .{});
                dbg("  r_cycle_bc4_regs_rwc[0]_LE=[", .{});
                const rc0 = r_cycle_bc4_regs_rwc[0].toBytesBE();
                for (0..32) |bi| dbg("{x:0>2}", .{rc0[31 - bi]});
                dbg("]\n", .{});
                dbg("  r_cycle_bc5_regs_val[0]_LE=[", .{});
                const rv5 = r_cycle_bc5_regs_val[0].toBytesBE();
                for (0..32) |bi| dbg("{x:0>2}", .{rv5[31 - bi]});
                dbg("]\n", .{});
            }
            } // end if (comptime debug_verbose)

            // ====================================================================
            // Cache openings to transcript
            // ====================================================================

            dbg("[STAGE6] Transcript before cache_openings: round={}\n", .{transcript.n_rounds});

            // Instance 0: BytecodeReadRaf
            for (bytecode_ra_claims) |claim| {
                transcript.appendScalar("opening_claim", claim);
            }
            dbg("[STAGE6] After BytecodeReadRaf openings ({}): round={}\n", .{bytecode_ra_claims.len, transcript.n_rounds});

            // Instance 1: Booleanity
            // Upstream aliasing: when bytecode_log_k is a multiple of log_k_chunk,
            // BytecodeRa(0)/Booleanity has the same opening point as BytecodeRa(0)/BytecodeReadRaf
            // (no zero-padding in compute_r_address_chunks), so the verifier aliases it
            // and does NOT flush it to transcript.
            const bytecode_ra0_aliases = (bytecode_log_k % log_k_chunk == 0);
            const bool_skip_index = instruction_ra_virtual_claims.len; // BytecodeRa(0) is at index instruction_d in Booleanity's polynomial_types
            for (booleanity_ra_claims, 0..) |claim, i| {
                if (bytecode_ra0_aliases and i == bool_skip_index) continue;
                transcript.appendScalar("opening_claim", claim);
            }

            // Instance 2: HammingBooleanity
            transcript.appendScalar("opening_claim", hamming_weight_claim);

            // Instance 3: RamRaVirtual
            for (ram_ra_virtual_claims) |claim| {
                transcript.appendScalar("opening_claim", claim);
            }

            // Instance 4: LookupsRaVirtual
            for (instruction_ra_virtual_claims) |claim| {
                transcript.appendScalar("opening_claim", claim);
            }

            dbg("[STAGE6] After LookupsRaVirtual openings ({}): round={}\n", .{instruction_ra_virtual_claims.len, transcript.n_rounds});

            // Instance 5: IncClaimReduction
            transcript.appendScalar("opening_claim", ram_inc_claim);
            transcript.appendScalar("opening_claim", rd_inc_claim);
            dbg("[STAGE6] After ALL cache_openings: round={}\n", .{transcript.n_rounds});

            return Stage6Result(F){
                .challenges = challenges,
                .bytecode_ra_claims = bytecode_ra_claims,
                .hamming_weight_claim = hamming_weight_claim,
                .booleanity_ra_claims = booleanity_ra_claims,
                .ram_ra_virtual_claims = ram_ra_virtual_claims,
                .instruction_ra_virtual_claims = instruction_ra_virtual_claims,
                .ram_inc_claim = ram_inc_claim,
                .rd_inc_claim = rd_inc_claim,
                .bytecode_log_k = bytecode_log_k,
                .log_k_chunk = log_k_chunk,
                .n_cycle_vars = n_cycle_vars,
                .bytecode_d = bytecode_d,
                .ram_d = ram_d,
                .instruction_d = instruction_d,
                .allocator = self.allocator,
            };
        }

        /// Compute BytecodeReadRaf input claim and per-stage claims
        /// Returns .{ total_claim, [5]per_stage_claims }
        fn computeBytecodeReadRafInputClaim(
            self: *Self,
            opening_claims: *OpeningClaims(F),
            gamma_powers: []const F,
            stage1_gammas: []const F,
            stage2_gammas: []const F,
            stage3_gammas: []const F,
            stage4_gammas: []const F,
            stage5_gammas: []const F,
        ) struct { total: F, per_stage: [5]F } {
            _ = self;

            const getClaim = struct {
                fn get(oc: *OpeningClaims(F), key: OpeningId) F {
                    return oc.get(key) orelse F.zero();
                }
            }.get;

            // rv_claim_1 (Stage 1 / SpartanOuter)
            var rv1 = F.zero();
            const oc_upc = getClaim(opening_claims, .{ .Virtual = .{ .poly = .UnexpandedPC, .sumcheck_id = .SpartanOuter } });
            rv1 = rv1.add(oc_upc); // No gamma[0] - Jolt formula: unexpanded_pc + γ¹·imm + Σγ^(2+i)·cf[i]
            const oc_imm = getClaim(opening_claims, .{ .Virtual = .{ .poly = .Imm, .sumcheck_id = .SpartanOuter } });
            rv1 = rv1.add(stage1_gammas[1].mul(oc_imm));
            var oc_flags: [14]F = undefined;
            for (0..14) |i| {
                oc_flags[i] = getClaim(opening_claims, .{ .Virtual = .{ .poly = .{ .OpFlags = @intCast(i) }, .sumcheck_id = .SpartanOuter } });
                rv1 = rv1.add(stage1_gammas[2 + i].mul(oc_flags[i]));
            }
            // Debug: print each opening claim component for rv1
            {
                const upc_le = oc_upc.toBytes();
                const imm_le = oc_imm.toBytes();
                dbg("[BCRAF_RV1_DETAIL] oc_UnexpandedPC_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                    upc_le[0], upc_le[1], upc_le[2], upc_le[3], upc_le[4], upc_le[5], upc_le[6], upc_le[7],
                });
                dbg("[BCRAF_RV1_DETAIL] oc_Imm_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                    imm_le[0], imm_le[1], imm_le[2], imm_le[3], imm_le[4], imm_le[5], imm_le[6], imm_le[7],
                });
                for (0..14) |i| {
                    const fl = oc_flags[i].toBytes();
                    dbg("[BCRAF_RV1_DETAIL] oc_OpFlag[{}]_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                        i, fl[0], fl[1], fl[2], fl[3], fl[4], fl[5], fl[6], fl[7],
                    });
                }
                // Also print the oc_PC claim (used for RAF) and FlagIsNoop
                const oc_pc = getClaim(opening_claims, .{ .Virtual = .{ .poly = .PC, .sumcheck_id = .SpartanOuter } });
                const pc_le = oc_pc.toBytes();
                dbg("[BCRAF_RV1_DETAIL] oc_PC_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                    pc_le[0], pc_le[1], pc_le[2], pc_le[3], pc_le[4], pc_le[5], pc_le[6], pc_le[7],
                });
            }

            // rv_claim_2 (Stage 2 / SpartanProductVirtualization)
            var rv2 = F.zero();
            // Upstream: Jump + γ·Branch + γ²·WriteLookupOutputToRD + γ³·VirtualInstruction
            rv2 = rv2.add(stage2_gammas[0].mul(getClaim(opening_claims,
                .{ .Virtual = .{ .poly = .{ .OpFlags = 5 }, .sumcheck_id = .SpartanProductVirtualization } })));
            rv2 = rv2.add(stage2_gammas[1].mul(getClaim(opening_claims,
                .{ .Virtual = .{ .poly = .{ .InstructionFlags = 4 }, .sumcheck_id = .SpartanProductVirtualization } })));
            rv2 = rv2.add(stage2_gammas[2].mul(getClaim(opening_claims,
                .{ .Virtual = .{ .poly = .{ .OpFlags = 6 }, .sumcheck_id = .SpartanProductVirtualization } })));
            rv2 = rv2.add(stage2_gammas[3].mul(getClaim(opening_claims,
                .{ .Virtual = .{ .poly = .{ .OpFlags = 7 }, .sumcheck_id = .SpartanProductVirtualization } })));

            // rv_claim_3 (Stage 3)
            var rv3 = F.zero();
            rv3 = rv3.add(getClaim(opening_claims,
                .{ .Virtual = .{ .poly = .Imm, .sumcheck_id = .InstructionInputVirtualization } })); // No gamma[0] - Jolt formula: imm + γ¹·unexpanded_pc + ...
            rv3 = rv3.add(stage3_gammas[1].mul(getClaim(opening_claims,
                .{ .Virtual = .{ .poly = .UnexpandedPC, .sumcheck_id = .SpartanShift } })));
            rv3 = rv3.add(stage3_gammas[2].mul(getClaim(opening_claims,
                .{ .Virtual = .{ .poly = .{ .InstructionFlags = 2 }, .sumcheck_id = .InstructionInputVirtualization } })));
            rv3 = rv3.add(stage3_gammas[3].mul(getClaim(opening_claims,
                .{ .Virtual = .{ .poly = .{ .InstructionFlags = 0 }, .sumcheck_id = .InstructionInputVirtualization } })));
            rv3 = rv3.add(stage3_gammas[4].mul(getClaim(opening_claims,
                .{ .Virtual = .{ .poly = .{ .InstructionFlags = 3 }, .sumcheck_id = .InstructionInputVirtualization } })));
            rv3 = rv3.add(stage3_gammas[5].mul(getClaim(opening_claims,
                .{ .Virtual = .{ .poly = .{ .InstructionFlags = 1 }, .sumcheck_id = .InstructionInputVirtualization } })));
            rv3 = rv3.add(stage3_gammas[6].mul(getClaim(opening_claims,
                .{ .Virtual = .{ .poly = .{ .InstructionFlags = 5 }, .sumcheck_id = .SpartanShift } })));
            rv3 = rv3.add(stage3_gammas[7].mul(getClaim(opening_claims,
                .{ .Virtual = .{ .poly = .{ .OpFlags = 7 }, .sumcheck_id = .SpartanShift } })));
            rv3 = rv3.add(stage3_gammas[8].mul(getClaim(opening_claims,
                .{ .Virtual = .{ .poly = .{ .OpFlags = 12 }, .sumcheck_id = .SpartanShift } })));

            // rv_claim_4 (Stage 4)
            var rv4 = F.zero();
            rv4 = rv4.add(stage4_gammas[0].mul(getClaim(opening_claims,
                .{ .Virtual = .{ .poly = .RdWa, .sumcheck_id = .RegistersReadWriteChecking } })));
            rv4 = rv4.add(stage4_gammas[1].mul(getClaim(opening_claims,
                .{ .Virtual = .{ .poly = .Rs1Ra, .sumcheck_id = .RegistersReadWriteChecking } })));
            rv4 = rv4.add(stage4_gammas[2].mul(getClaim(opening_claims,
                .{ .Virtual = .{ .poly = .Rs2Ra, .sumcheck_id = .RegistersReadWriteChecking } })));

            // rv_claim_5 (Stage 5)
            const NUM_LOOKUP_TABLES: usize = 40;
            var rv5 = F.zero();
            const rv5_rdwa = getClaim(opening_claims,
                .{ .Virtual = .{ .poly = .RdWa, .sumcheck_id = .RegistersValEvaluation } });
            rv5 = rv5.add(rv5_rdwa); // No gamma[0] - Jolt formula: eq(rd,r) + γ¹·!interleaved + ...
            const rv5_raf_flag = getClaim(opening_claims,
                .{ .Virtual = .{ .poly = .InstructionRafFlag, .sumcheck_id = .InstructionReadRaf } });
            rv5 = rv5.add(stage5_gammas[1].mul(rv5_raf_flag));
            for (0..NUM_LOOKUP_TABLES) |i| {
                const lt_claim = getClaim(opening_claims,
                    .{ .Virtual = .{ .poly = .{ .LookupTableFlag = i }, .sumcheck_id = .InstructionReadRaf } });
                rv5 = rv5.add(stage5_gammas[2 + i].mul(lt_claim));
                if (!lt_claim.eql(F.zero())) {
                    const ltb = lt_claim.toBytes();
                    dbg("[BCRAF_RV5] LookupTableFlag({})_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                        i, ltb[0], ltb[1], ltb[2], ltb[3], ltb[4], ltb[5], ltb[6], ltb[7],
                    });
                }
            }
            {
                const rdwa_le = rv5_rdwa.toBytes();
                const rff_le = rv5_raf_flag.toBytes();
                const rv5_le = rv5.toBytes();
                dbg("[BCRAF_RV5] RdWa_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                    rdwa_le[0], rdwa_le[1], rdwa_le[2], rdwa_le[3], rdwa_le[4], rdwa_le[5], rdwa_le[6], rdwa_le[7],
                });
                dbg("[BCRAF_RV5] InstructionRafFlag_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                    rff_le[0], rff_le[1], rff_le[2], rff_le[3], rff_le[4], rff_le[5], rff_le[6], rff_le[7],
                });
                dbg("[BCRAF_RV5] total_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                    rv5_le[0], rv5_le[1], rv5_le[2], rv5_le[3], rv5_le[4], rv5_le[5], rv5_le[6], rv5_le[7],
                });
            }

            // RAF claims
            const raf_claim = getClaim(opening_claims,
                .{ .Virtual = .{ .poly = .PC, .sumcheck_id = .SpartanOuter } });
            const raf_shift_claim = getClaim(opening_claims,
                .{ .Virtual = .{ .poly = .PC, .sumcheck_id = .SpartanShift } });

            // Debug: print per-stage rv_claims and raf_claims
            {
                const rv_arr = [5]F{ rv1, rv2, rv3, rv4, rv5 };
                for (0..5) |s| {
                    const rvl = rv_arr[s].toBytes();
                    dbg("[BCRAF_INPUT] rv_claim[{}]_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                        s, rvl[0], rvl[1], rvl[2], rvl[3], rvl[4], rvl[5], rvl[6], rvl[7],
                    });
                }
                const raf_le = raf_claim.toBytes();
                const rafs_le = raf_shift_claim.toBytes();
                dbg("[BCRAF_INPUT] raf_claim_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                    raf_le[0], raf_le[1], raf_le[2], raf_le[3], raf_le[4], raf_le[5], raf_le[6], raf_le[7],
                });
                dbg("[BCRAF_INPUT] raf_shift_claim_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                    rafs_le[0], rafs_le[1], rafs_le[2], rafs_le[3], rafs_le[4], rafs_le[5], rafs_le[6], rafs_le[7],
                });
                // Also print per-stage claims with RAF folded in (like Jolt's claim_per_stage)
                const cps0 = rv1.add(gamma_powers[5].mul(raf_claim));
                const cps2 = rv3.add(gamma_powers[4].mul(raf_shift_claim));
                const cps0l = cps0.toBytes();
                const cps2l = cps2.toBytes();
                dbg("[BCRAF_INPUT] claim_per_stage[0]_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                    cps0l[0], cps0l[1], cps0l[2], cps0l[3], cps0l[4], cps0l[5], cps0l[6], cps0l[7],
                });
                dbg("[BCRAF_INPUT] claim_per_stage[2]_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                    cps2l[0], cps2l[1], cps2l[2], cps2l[3], cps2l[4], cps2l[5], cps2l[6], cps2l[7],
                });
            }

            // Per-stage claims (like Jolt's claim_per_stage)
            // claim_per_stage[s] = rv_claim[s] + RAF_s contribution
            const per_stage = [5]F{
                rv1.add(gamma_powers[5].mul(raf_claim)), // Stage 0: rv1 + gamma^5 * raf
                rv2, // Stage 1: rv2
                rv3.add(gamma_powers[4].mul(raf_shift_claim)), // Stage 2: rv3 + gamma^4 * raf_shift
                rv4, // Stage 3: rv4
                rv5, // Stage 4: rv5
            };

            // Combine: total = Σ_s gamma^s * per_stage[s]
            var result = F.zero();
            for (0..5) |s| {
                const term = gamma_powers[s].mul(per_stage[s]);
                result = result.add(term);
                const ps_le = per_stage[s].toBytes();
                const gp_le = gamma_powers[s].toBytes();
                const tm_le = term.toBytes();
                dbg("[BCRAF_AGG_OC] s={} gp_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}] ps_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}] term_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                    s,
                    gp_le[0], gp_le[1], gp_le[2], gp_le[3], gp_le[4], gp_le[5], gp_le[6], gp_le[7],
                    ps_le[0], ps_le[1], ps_le[2], ps_le[3], ps_le[4], ps_le[5], ps_le[6], ps_le[7],
                    tm_le[0], tm_le[1], tm_le[2], tm_le[3], tm_le[4], tm_le[5], tm_le[6], tm_le[7],
                });
            }
            const res_le = result.toBytes();
            dbg("[BCRAF_AGG_OC] total_LE=[{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2},{x:0>2}]\n", .{
                res_le[0], res_le[1], res_le[2], res_le[3], res_le[4], res_le[5], res_le[6], res_le[7],
            });

            return .{ .total = result, .per_stage = per_stage };
        }
    };
}

// =============================================================================
// Helper: Convert evaluations to monomial coefficients and add batch*coeffs to combined_coeffs
// =============================================================================
// Converts [p(0), p(1), ..., p(d)] (Vandermonde evals) to monomial [c0, c1, ..., cd]
// using finite differences for small degrees (d <= 3), then adds batch * c_i to combined_coeffs[i].
fn addEvalsAsMonomialToCoeffs(comptime F: type, combined_coeffs: []F, polys: []const F, n_evals: usize, batch_coeff: F) void {
    if (n_evals == 1) {
        // Degree 0: c0 = p(0)
        combined_coeffs[0] = combined_coeffs[0].add(batch_coeff.mul(polys[0]));
    } else if (n_evals == 2) {
        // Degree 1: c0 = p(0), c1 = p(1) - p(0)
        const c0 = polys[0];
        const c1 = polys[1].sub(polys[0]);
        combined_coeffs[0] = combined_coeffs[0].add(batch_coeff.mul(c0));
        combined_coeffs[1] = combined_coeffs[1].add(batch_coeff.mul(c1));
    } else if (n_evals == 3) {
        // Degree 2: c0 = p(0), c2 = (p(2) - 2p(1) + p(0)) / 2, c1 = p(1) - p(0) - c2
        const inv2 = UniPoly(F).INV2;
        const c0 = polys[0];
        const c2 = polys[2].sub(polys[1]).sub(polys[1]).add(polys[0]).mul(inv2);
        const c1 = polys[1].sub(polys[0]).sub(c2);
        combined_coeffs[0] = combined_coeffs[0].add(batch_coeff.mul(c0));
        combined_coeffs[1] = combined_coeffs[1].add(batch_coeff.mul(c1));
        combined_coeffs[2] = combined_coeffs[2].add(batch_coeff.mul(c2));
    } else if (n_evals == 4) {
        // Degree 3: finite differences
        const inv2 = UniPoly(F).INV2;
        const inv6 = F.fromU64(6).inverse().?;
        const c0 = polys[0];
        const d1 = polys[1].sub(polys[0]);
        const d2 = polys[2].sub(polys[1]);
        const d3 = polys[3].sub(polys[2]);
        const dd1 = d2.sub(d1);
        const dd2 = d3.sub(d2);
        const c3 = dd2.sub(dd1).mul(inv6);
        const c2 = dd1.mul(inv2).sub(c3.mul(F.fromU64(3)));
        const c1 = d1.sub(c2).sub(c3);
        combined_coeffs[0] = combined_coeffs[0].add(batch_coeff.mul(c0));
        combined_coeffs[1] = combined_coeffs[1].add(batch_coeff.mul(c1));
        combined_coeffs[2] = combined_coeffs[2].add(batch_coeff.mul(c2));
        combined_coeffs[3] = combined_coeffs[3].add(batch_coeff.mul(c3));
    } else {
        // General case: use Newton forward differences with static buffer
        // Supports up to degree 15 (16 eval points)
        std.debug.assert(n_evals <= 16);
        var dd: [16]F = undefined;
        for (0..n_evals) |i| dd[i] = polys[i];

        // Build forward difference table: dd[k] = k-th order forward difference at 0
        // After processing, dd[k] = Δ^k p(0)
        var coeffs_buf: [16]F = undefined;
        coeffs_buf[0] = dd[0]; // Δ^0 = p(0)

        var order: usize = 1;
        while (order < n_evals) : (order += 1) {
            // Compute order-th forward differences in-place
            var i = n_evals - 1;
            while (i >= order) : (i -= 1) {
                dd[i] = dd[i].sub(dd[i - 1]);
                if (i == order) break;
            }
            coeffs_buf[order] = dd[order]; // Δ^order p(0)
        }

        // Convert Newton forward differences to monomial coefficients
        // Newton form: p(x) = Σ_k Δ^k p(0) * C(x, k)
        // where C(x, k) = x(x-1)...(x-k+1) / k!
        // We need to convert to monomial c0 + c1*x + c2*x^2 + ...
        // Use the fact that Δ^k p(0) / k! is the leading coefficient contribution
        // Actually, the simplest approach for general n: use the Vandermonde solver result
        // which is already available via fromEvalsVandermonde. But since this is a non-allocating
        // path, we use Sterling numbers of the first kind.
        //
        // Actually for the general case, let's just compute monomial coefficients directly
        // from the forward differences using the Stirling number relationship.
        // c_j = Σ_{k=j}^{d} S1(k, j) * Δ^k p(0) / k!
        // This is complex. For now, fall back to evaluating the Newton form at integer points
        // and using the same approach as vandermondeToCompressed for n > 4.
        //
        // Simpler: we have forward differences. Convert via the standard formula:
        // The Newton forward difference interpolation gives:
        // c_k = Σ_{j=0}^{k} (-1)^{k-j} C(k,j) * Δ^j p(0) / ... no, this is circular.
        //
        // Let's just directly use finite-difference-to-monomial conversion:
        // Start with Newton basis coefficients dd[0..n] = [Δ^0 p(0)/0!, Δ^1 p(0)/1!, ...]
        // and convert to monomial via the standard algorithm.

        // Divide by factorials to get Newton basis coefficients
        var fact = F.one();
        for (1..n_evals) |k| {
            fact = fact.mul(F.fromU64(@intCast(k)));
            coeffs_buf[k] = coeffs_buf[k].mul(fact.inverse().?);
        }

        // Convert Newton basis to monomial: c(x) = Σ a_k * x*(x-1)*...*(x-k+1)
        // Process from highest to lowest degree, expanding x*(x-1)*...*(x-k+1) into monomials.
        // Use the recurrence: multiply running polynomial by (x - k) at each step.
        var mono: [16]F = .{F.zero()} ** 16;
        mono[0] = coeffs_buf[0];

        for (1..n_evals) |k| {
            // We need to add coeffs_buf[k] * x*(x-1)*...*(x-k+1) to mono
            // Build the falling factorial x*(x-1)*...*(x-k+1) incrementally
            // ff[k] = ff[k-1] * (x - (k-1))
            // We maintain ff_mono[0..k] = monomial coefficients of x*(x-1)*...*(x-k+1)
            // Start: ff_mono = [0, 1] for x
            // Multiply by (x - j) for j = 1, 2, ..., k-1
            var ff: [16]F = .{F.zero()} ** 16;
            ff[1] = F.one(); // x
            for (1..k) |j| {
                // Multiply ff by (x - j): new[i] = ff[i-1] - j*ff[i]
                const neg_j = F.zero().sub(F.fromU64(@intCast(j)));
                var i_rev = j + 1;
                while (i_rev > 0) {
                    i_rev -= 1;
                    const prev = if (i_rev > 0) ff[i_rev - 1] else F.zero();
                    ff[i_rev] = prev.add(neg_j.mul(ff[i_rev]));
                }
            }
            // Add coeffs_buf[k] * ff to mono
            for (0..k + 1) |i| {
                mono[i] = mono[i].add(coeffs_buf[k].mul(ff[i]));
            }
        }


        // Add batch * mono to combined_coeffs
        for (0..n_evals) |i| {
            combined_coeffs[i] = combined_coeffs[i].add(batch_coeff.mul(mono[i]));
        }
    }
}

// =============================================================================
// Helper: Add variable-length instance evals to combined_evals with interpolation (LEGACY)
// =============================================================================
// All evaluation arrays use Vandermonde format: [p(0), p(1), ..., p(d)]
// (evaluations at consecutive integer points, no p_inf)
fn addInstanceEvalsToCombibed(comptime F: type, combined_evals: []F, polys: []const F, batch_coeff: F, num_evals: usize) void {
    const inst_n_evals = polys.len;

    if (inst_n_evals >= num_evals) {
        // Instance has enough eval points - just add the first num_evals
        for (0..num_evals) |k| {
            combined_evals[k] = combined_evals[k].add(batch_coeff.mul(polys[k]));
        }
    } else {
        // Instance has fewer eval points - need Lagrange interpolation for missing points
        // polys format (Vandermonde): [p(0), p(1), ..., p(inst_n_evals-1)]
        // Need to interpolate p(inst_n_evals), ..., p(num_evals-1)

        // Add known evaluation points
        for (0..inst_n_evals) |k| {
            combined_evals[k] = combined_evals[k].add(batch_coeff.mul(polys[k]));
        }

        // Lagrange interpolation for missing points
        for (inst_n_evals..num_evals) |k| {
            const x = F.fromU64(@intCast(k));
            var lagrange_val = F.zero();
            for (0..inst_n_evals) |m| {
                var basis = F.one();
                const xm = F.fromU64(@intCast(m));
                for (0..inst_n_evals) |n| {
                    if (n != m) {
                        const xn = F.fromU64(@intCast(n));
                        basis = basis.mul(x.sub(xn)).mul(xm.sub(xn).inverse().?);
                    }
                }
                lagrange_val = lagrange_val.add(basis.mul(polys[m]));
            }
            combined_evals[k] = combined_evals[k].add(batch_coeff.mul(lagrange_val));
        }
    }
}

/// Add fixed-size instance evaluations to combined (for degree-3 instances like Hamming)
// All evaluation arrays use Vandermonde format: [p(0), p(1), ..., p(d)]
fn addFixedEvalsToCombibed(comptime F: type, combined_evals: []F, polys: []const F, n_polys: usize, batch_coeff: F, num_evals: usize) void {
    if (n_polys >= num_evals) {
        // Instance has enough eval points - add the first num_evals
        for (0..num_evals) |k| {
            combined_evals[k] = combined_evals[k].add(batch_coeff.mul(polys[k]));
        }
    } else {
        // Instance has fewer eval points - need Lagrange interpolation for missing points
        for (0..n_polys) |k| {
            combined_evals[k] = combined_evals[k].add(batch_coeff.mul(polys[k]));
        }

        // Lagrange interpolation for missing points
        for (n_polys..num_evals) |k| {
            const x = F.fromU64(@intCast(k));
            var lagrange_val = F.zero();
            for (0..n_polys) |m| {
                var basis = F.one();
                const xm = F.fromU64(@intCast(m));
                for (0..n_polys) |n| {
                    if (n != m) {
                        const xn = F.fromU64(@intCast(n));
                        basis = basis.mul(x.sub(xn)).mul(xm.sub(xn).inverse().?);
                    }
                }
                lagrange_val = lagrange_val.add(basis.mul(polys[m]));
            }
            combined_evals[k] = combined_evals[k].add(batch_coeff.mul(lagrange_val));
        }
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Compute eq polynomial table: eq(r, j) for all j in [0, 2^n_vars)
/// r is in BIG_ENDIAN order (r[0] is the most significant variable)
pub fn computeEqTable(comptime F: type, allocator: Allocator, r: []const F, n_vars: usize) ![]F {
    return computeEqTableParallel(F, allocator, r, n_vars, null);
}

/// Compute eq polynomial table with optional parallel inner loops.
/// Same as computeEqTable but parallelizes large levels via ThreadPool.
pub fn computeEqTableParallel(comptime F: type, allocator: Allocator, r: []const F, n_vars: usize, pool: ?*ThreadPool) ![]F {
    const size: usize = @as(usize, 1) << @intCast(n_vars);
    var table = try allocator.alloc(F, size);

    table[0] = F.one();

    for (0..n_vars) |i| {
        const r_i = r[i];
        const cur_size: usize = @as(usize, 1) << @intCast(i);

        if (pool != null and cur_size >= 256) {
            // Parallel: forward iteration, writes to disjoint halves [0..cur_size) and [cur_size..2*cur_size)
            const Ctx = struct {
                tbl: []F,
                ri: F,
                cs: usize,
            };
            const ctx = Ctx{ .tbl = table, .ri = r_i, .cs = cur_size };
            pool.?.parallelForForce(cur_size, ctx, struct {
                fn f(c: Ctx, j: usize) void {
                    const x = c.tbl[j];
                    const y = x.mul(c.ri);
                    c.tbl[j + c.cs] = y;
                    c.tbl[j] = x.sub(y);
                }
            }.f);
        } else {
            // Sequential: backward iteration (original)
            var j: usize = cur_size;
            while (j > 0) {
                j -= 1;
                const x = table[j];
                const y = x.mul(r_i);
                table[j + cur_size] = y;
                table[j] = x.sub(y);
            }
        }
    }

    return table;
}

/// Convert signed i128 to field element
pub fn fieldFromI128(comptime F: type, val: i128) F {
    if (val >= 0) {
        return F.fromU128(@intCast(val));
    } else {
        return F.fromU128(@intCast(-val)).neg();
    }
}

/// Extract chunk from address value using MSB-first ordering (matching Jolt)
/// chunk_idx=0 is the most significant chunk
pub fn extractChunkMSB(addr: u64, chunk_idx: usize, total_chunks: usize, log_k_chunk: usize) usize {
    // Jolt: shift = log_k_chunk * (d - 1 - chunk_idx)
    const shift_amount = log_k_chunk * (total_chunks - 1 - chunk_idx);
    if (shift_amount >= 64) return 0;
    const shift: u6 = @intCast(shift_amount);
    const mask: u64 = (@as(u64, 1) << @intCast(log_k_chunk)) - 1;
    return @intCast((addr >> shift) & mask);
}

/// Interleave bits of two 64-bit values to form a 128-bit lookup index
/// Matches Jolt's interleave_bits(even_bits, odd_bits): result = (even << 1) | odd
/// So even_bits (rs1) go to odd bit positions (1,3,5,...,127)
/// and odd_bits (rs2) go to even bit positions (0,2,4,...,126)
pub fn interleaveBits(rs1: u64, rs2: u64) u128 {
    // Spread rs1 bits to odd positions
    var x: u128 = @intCast(rs1);
    x = (x | (x << 32)) & 0x0000_0000_FFFF_FFFF_0000_0000_FFFF_FFFF;
    x = (x | (x << 16)) & 0x0000_FFFF_0000_FFFF_0000_FFFF_0000_FFFF;
    x = (x | (x << 8)) & 0x00FF_00FF_00FF_00FF_00FF_00FF_00FF_00FF;
    x = (x | (x << 4)) & 0x0F0F_0F0F_0F0F_0F0F_0F0F_0F0F_0F0F_0F0F;
    x = (x | (x << 2)) & 0x3333_3333_3333_3333_3333_3333_3333_3333;
    x = (x | (x << 1)) & 0x5555_5555_5555_5555_5555_5555_5555_5555;

    // Spread rs2 bits to even positions
    var y: u128 = @intCast(rs2);
    y = (y | (y << 32)) & 0x0000_0000_FFFF_FFFF_0000_0000_FFFF_FFFF;
    y = (y | (y << 16)) & 0x0000_FFFF_0000_FFFF_0000_FFFF_0000_FFFF;
    y = (y | (y << 8)) & 0x00FF_00FF_00FF_00FF_00FF_00FF_00FF_00FF;
    y = (y | (y << 4)) & 0x0F0F_0F0F_0F0F_0F0F_0F0F_0F0F_0F0F_0F0F;
    y = (y | (y << 2)) & 0x3333_3333_3333_3333_3333_3333_3333_3333;
    y = (y | (y << 1)) & 0x5555_5555_5555_5555_5555_5555_5555_5555;

    return (x << 1) | y;
}

/// Decode sign-extended immediate from RISC-V instruction encoding, returned as u64 (two's complement).
/// This matches Jolt's `to_instruction_inputs()` which sign-extends the immediate value.
fn decodeImmediateU64(instr: u32) u64 {
    const opcode: u8 = @truncate(instr & 0x7f);
    switch (opcode) {
        // I-type: imm[11:0] at bits [31:20], sign-extended
        0x13, 0x03, 0x67, 0x1b, 0x73 => {
            const imm12: u32 = instr >> 20;
            const imm_signed: i64 = @as(i64, @as(i32, @bitCast(imm12 << 20)) >> 20);
            return @bitCast(imm_signed);
        },
        // S-type: imm[11:5] at [31:25], imm[4:0] at [11:7], sign-extended
        0x23 => {
            const imm11_5 = (instr >> 25) & 0x7f;
            const imm4_0 = (instr >> 7) & 0x1f;
            const imm12: u32 = (imm11_5 << 5) | imm4_0;
            const imm_signed: i64 = @as(i64, @as(i32, @bitCast(imm12 << 20)) >> 20);
            return @bitCast(imm_signed);
        },
        // B-type: imm[12|10:5] at [31:25], imm[4:1|11] at [11:7], sign-extended, *2
        0x63 => {
            const imm12 = (instr >> 31) & 1;
            const imm10_5 = (instr >> 25) & 0x3f;
            const imm4_1 = (instr >> 8) & 0xf;
            const imm11 = (instr >> 7) & 1;
            const imm13: u32 = (imm12 << 12) | (imm11 << 11) | (imm10_5 << 5) | (imm4_1 << 1);
            const imm_signed: i64 = @as(i64, @as(i32, @bitCast(imm13 << 19)) >> 19);
            return @bitCast(imm_signed);
        },
        // U-type: imm[31:12] at [31:12], shifted left by 12, SIGN-EXTENDED to 64 bits
        // Matches Jolt's FormatU.parse: `as i32 as i64 as u64`
        0x37, 0x17 => {
            const imm_upper: u32 = instr & 0xFFFFF000;
            return @bitCast(@as(i64, @as(i32, @bitCast(imm_upper))));
        },
        // J-type: imm[20|10:1|11|19:12] at [31:12], sign-extended, *2
        0x6f => {
            const imm20 = (instr >> 31) & 1;
            const imm10_1 = (instr >> 21) & 0x3ff;
            const imm11 = (instr >> 20) & 1;
            const imm19_12 = (instr >> 12) & 0xff;
            const imm21: u32 = (imm20 << 20) | (imm19_12 << 12) | (imm11 << 11) | (imm10_1 << 1);
            const imm_signed: i64 = @as(i64, @as(i32, @bitCast(imm21 << 11)) >> 11);
            return @bitCast(imm_signed);
        },
        else => return 0,
    }
}

/// Compute the 128-bit lookup index for a trace step.
///
/// This matches Jolt's per-instruction `to_lookup_index()` method:
/// - AddOperands instructions (ADD, ADDI, etc.): returns raw sum as u128 (NO interleaving)
/// - SubtractOperands instructions (SUB, SUBW): returns raw shifted difference as u128
/// - MultiplyOperands instructions (MUL, MULHU): returns raw product as u128
/// - Standard instructions (XOR, AND, OR, SLT, branches): returns interleave_bits(x, y)
/// - No-lookup instructions (Load, Store, SLL, SRL): returns 0
/// - NoOp cycles: returns 0
pub fn computeLookupIndex(step: tracer.TraceStep) u128 {
    if (step.is_noop and !step.is_termination_store) return 0;

    const instr = step.instruction;
    const opcode: u8 = @truncate(instr & 0x7f);
    const funct3: u3 = @truncate((instr >> 12) & 0x7);
    const funct7: u7 = @truncate(instr >> 25);

    // Check if instruction has a lookup table at all
    if (!hasLookupTable(opcode, funct3, funct7)) return 0;

    // Virtual opcodes: handle specially since they don't follow standard RISC-V encoding
    if (opcode == 0x0B) {
        // VirtualSignExtendWord: AddOperands → rs1 + 0 = rs1
        // Jolt's to_lookup_index() returns rs1 directly (no interleaving)
        return @as(u128, step.rs1_value);
    }
    if (opcode == 0x2B) {
        if (funct3 == 0) {
            // VirtualMULI: MultiplyOperands → rs1 * (1 << shamt)
            const shamt_raw: u32 = instr >> 20;
            const shamt: u6 = @truncate(shamt_raw & 0x3F);
            const multiplier: u128 = @as(u128, 1) << shamt;
            return @as(u128, step.rs1_value) * multiplier;
        } else {
            // VirtualPow2 (funct3=1), VirtualShiftRightBitmask (funct3=2): AddOperands → rs1 + 0 = rs1
            return @as(u128, step.rs1_value);
        }
    }
    if (opcode == 0x5B) {
        if (step.rs2_read) {
            // VirtualSRL/VirtualSRA R-type: interleaved(rs1_value, rs2_value)
            return interleaveBits(step.rs1_value, step.rs2_value);
        } else {
            // VirtualSRLI/VirtualSRAI I-type: interleaved(rs1_value, bitmask)
            const total_shift_raw: u32 = instr >> 20;
            const total_shift: u7 = @truncate(total_shift_raw & 0x3F);
            const ones: u128 = (@as(u128, 1) << @intCast(64 - @as(u8, total_shift))) - 1;
            const bitmask: u64 = @truncate(ones << total_shift);
            return interleaveBits(step.rs1_value, bitmask);
        }
    }
    if (opcode == 0x02) {
        // VirtualAdvice: the lookup index is the advice value (rd_value)
        // Jolt's to_lookup_index() returns the second operand which is the advice value
        return @as(u128, step.rd_value);
    }
    if (opcode == 0x22) {
        if (funct3 == 2 or funct3 == 3) {
            // VirtualAssertHalfwordAlignment/WordAlignment: AddOperands → rs1 + imm
            const imm_raw: u32 = instr >> 20;
            const imm_signed: i64 = @as(i64, @as(i32, @bitCast(imm_raw << 20)) >> 20);
            return @as(u128, step.rs1_value +% @as(u64, @bitCast(imm_signed)));
        } else {
            // VirtualAssertEQ (funct3=0) / VirtualAssertValidDiv0 (funct3=1): interleaved
            return interleaveBits(step.rs1_value, step.rs2_value);
        }
    }
    if (opcode == 0x42) {
        // VirtualZeroExtendWord: AddOperands → rs1 + 0 = rs1
        // Jolt's to_lookup_index() returns rs1 directly (like SignExtendWord)
        return @as(u128, step.rs1_value);
    }
    if (opcode == 0x62) {
        // VirtualAssertValidUnsignedRemainder: interleaved(rs1_value, rs2_value)
        // LeftOperandIsRs1Value, RightOperandIsRs2Value → interleave
        return interleaveBits(step.rs1_value, step.rs2_value);
    }

    // Determine left_input and right_input (matching Jolt's to_instruction_inputs)
    const left_is_rs1: bool = switch (opcode) {
        0x33, 0x3b, 0x23, 0x63, 0x13, 0x03, 0x67, 0x1b => true,
        else => false,
    };
    const left_is_pc: bool = switch (opcode) {
        0x17, 0x6f => true,
        else => false,
    };
    const right_is_rs2: bool = switch (opcode) {
        0x33, 0x63, 0x3b => true,
        else => false,
    };
    const right_is_imm: bool = switch (opcode) {
        0x13, 0x03, 0x67, 0x23, 0x37, 0x17, 0x6f, 0x1b => true,
        else => false,
    };

    var left_input: u64 = 0;
    if (left_is_rs1) left_input = step.rs1_value;
    if (left_is_pc) left_input = step.unexpanded_pc;

    var right_input: u64 = 0;
    if (right_is_rs2) right_input = step.rs2_value;
    if (right_is_imm) right_input = decodeImmediateU64(instr);

    // Now compute the lookup index based on the instruction's operand mode
    switch (opcode) {
        0x33 => { // R-type
            if (funct7 == 0x01) {
                // M-extension
                if (funct3 == 0x0) {
                    // MUL: MultiplyOperands → raw product
                    return @as(u128, left_input) * @as(u128, right_input);
                } else if (funct3 == 0x3) {
                    // MULHU: MultiplyOperands → raw product
                    return @as(u128, left_input) * @as(u128, right_input);
                } else {
                    // Other M-ext: interleaved
                    return interleaveBits(left_input, right_input);
                }
            } else if (funct7 == 0x20 and funct3 == 0x0) {
                // SUB: SubtractOperands → x + (2^64 - y)
                return @as(u128, left_input) + (@as(u128, 1) << 64) - @as(u128, right_input);
            } else if (funct7 == 0 and funct3 == 0x0) {
                // ADD: AddOperands → raw sum
                return @as(u128, left_input) + @as(u128, right_input);
            } else {
                // Other R-type (AND, OR, XOR, SLT, SLTU): interleaved
                return interleaveBits(left_input, right_input);
            }
        },
        0x13 => { // I-type ALU
            if (funct3 == 0) {
                // ADDI: AddOperands → raw sum
                return @as(u128, left_input) + @as(u128, right_input);
            } else {
                // SLLI, SLTI, SLTIU, XORI, SRLI, SRAI, ORI, ANDI: interleaved
                return interleaveBits(left_input, right_input);
            }
        },
        0x37 => { // LUI: AddOperands → immediate directly (left=0)
            return @as(u128, left_input) + @as(u128, right_input);
        },
        0x17 => { // AUIPC: AddOperands → PC + imm
            return @as(u128, left_input) + @as(u128, right_input);
        },
        0x6f => { // JAL: AddOperands → PC + imm
            return @as(u128, left_input) + @as(u128, right_input);
        },
        0x67 => { // JALR: AddOperands → rs1 + imm
            return @as(u128, left_input) + @as(u128, right_input);
        },
        0x1b => { // I-type word ALU
            if (funct3 == 0) {
                // ADDIW: AddOperands → raw sum
                return @as(u128, left_input) + @as(u128, right_input);
            } else {
                // SLLIW, SRLIW, SRAIW: interleaved
                return interleaveBits(left_input, right_input);
            }
        },
        0x3b => { // OP-32
            if (funct3 == 0 and funct7 == 0) {
                // ADDW: AddOperands → raw sum
                return @as(u128, left_input) + @as(u128, right_input);
            } else if (funct3 == 0 and funct7 == 0x20) {
                // SUBW: SubtractOperands → x + (2^64 - y)
                return @as(u128, left_input) + (@as(u128, 1) << 64) - @as(u128, right_input);
            } else {
                // Other 0x3b: interleaved
                return interleaveBits(left_input, right_input);
            }
        },
        0x63 => { // Branch: interleaved
            return interleaveBits(left_input, right_input);
        },
        else => {
            // Default: interleaved
            return interleaveBits(left_input, right_input);
        },
    }
}

/// Get lookup index chunk from trace step.
/// This matches Jolt's lookup_index_chunk with instruction_shifts (MSB-first ordering).
/// Uses the instruction-type-aware computeLookupIndex to correctly handle
/// AddOperands, SubtractOperands, and MultiplyOperands instructions.
fn getLookupChunkInterleaved(step: tracer.TraceStep, chunk_idx: usize, log_k_chunk: usize, instruction_d: usize) usize {
    // Build the correct 128-bit lookup index based on instruction type
    const lookup_index = computeLookupIndex(step);

    // MSB-first: shift = log_k_chunk * (instruction_d - 1 - chunk_idx)
    const shift_amount = log_k_chunk * (instruction_d - 1 - chunk_idx);
    if (shift_amount >= 128) return 0;
    const shift: u7 = @intCast(shift_amount);
    const mask: u128 = (@as(u128, 1) << @intCast(log_k_chunk)) - 1;
    return @intCast((lookup_index >> shift) & mask);
}

/// Evaluate a polynomial at a point given its Toom-Cook evals format:
/// evals = [p(0), p(1), ..., p(d-1), p(inf)]
/// where p(inf) is the leading coefficient (coefficient of x^d).
/// The polynomial has degree d where d = evals.len - 1.
/// Uses Lagrange interpolation on the d finite points {0, 1, ..., d-1}
/// plus the leading coefficient correction.
/// Evaluate polynomial at challenge given Vandermonde evals [p(0), p(1), ..., p(d)]
/// Uses Lagrange interpolation through all n_evals points at consecutive integers.
fn evaluatePolyFromEvals(comptime F: type, evals: []const F, challenge: F) F {
    const n_evals = evals.len;

    // Lagrange interpolation through (0, p(0)), (1, p(1)), ..., (n_evals-1, p(n_evals-1))
    var result = F.zero();
    for (0..n_evals) |m| {
        var basis = F.one();
        const xm = F.fromU64(@intCast(m));
        for (0..n_evals) |n| {
            if (n != m) {
                const xn = F.fromU64(@intCast(n));
                basis = basis.mul(challenge.sub(xn)).mul(xm.sub(xn).inverse().?);
            }
        }
        result = result.add(basis.mul(evals[m]));
    }

    return result;
}

/// Evaluate degree-3 polynomial at challenge given Vandermonde evals [p(0), p(1), p(2), p(3)]
fn evaluateDeg3FromEvals(comptime F: type, evals: [4]F, challenge: F) F {
    const p0 = evals[0];
    const p1 = evals[1];
    const p2 = evals[2];
    const p3 = evals[3];

    // Lagrange interpolation through (0, p0), (1, p1), (2, p2), (3, p3)
    // L_0(x) = (x-1)(x-2)(x-3)/((0-1)(0-2)(0-3)) = (x-1)(x-2)(x-3)/(-6)
    // L_1(x) = (x-0)(x-2)(x-3)/((1-0)(1-2)(1-3)) = x(x-2)(x-3)/(2)
    // L_2(x) = (x-0)(x-1)(x-3)/((2-0)(2-1)(2-3)) = x(x-1)(x-3)/(-2)
    // L_3(x) = (x-0)(x-1)(x-2)/((3-0)(3-1)(3-2)) = x(x-1)(x-2)/(6)
    const x = challenge;
    const xm1 = x.sub(F.one());
    const xm2 = x.sub(F.fromU64(2));
    const xm3 = x.sub(F.fromU64(3));
    const six_inv = F.fromU64(6).inverse().?;
    const two_inv = UniPoly(F).INV2;

    const l0 = xm1.mul(xm2).mul(xm3).mul(six_inv).neg();
    const l1 = x.mul(xm2).mul(xm3).mul(two_inv);
    const l2 = x.mul(xm1).mul(xm3).mul(two_inv).neg();
    const l3 = x.mul(xm1).mul(xm2).mul(six_inv);

    return l0.mul(p0).add(l1.mul(p1)).add(l2.mul(p2)).add(l3.mul(p3));
}

// =============================================================================
// Tests
// =============================================================================

const testing = std.testing;
const BN254Scalar = @import("zolt_arith").field.BN254Scalar;

/// Helper: compute eq(r, x) directly for a boolean vector x and field vector r.
/// Both in LE order (r[0] = LSB, matching computeEqTable's output convention).
fn eqEvalDirect(r: []const BN254Scalar, x: usize) BN254Scalar {
    var result = BN254Scalar.one();
    for (0..r.len) |i| {
        const bit: u1 = @truncate(x >> @intCast(i));
        if (bit == 1) {
            result = result.mul(r[i]);
        } else {
            result = result.mul(BN254Scalar.one().sub(r[i]));
        }
    }
    return result;
}

test "split-eq factorization: eq_lo * eq_hi = eq_full" {
    // Verify the core split-eq identity:
    //   eq(r, x) = eq(r_lo, x_lo) * eq(r_hi, x_hi)
    // where x = x_lo + x_hi << prefix_n_vars
    //
    // computeEqTable takes BE input r[0..n], output table[j] has bit i → r[i].
    // For x = x_lo | (x_hi << prefix_n_vars):
    //   bits 0..prefix_n_vars-1 (x_lo) → r_be[0..prefix_n_vars]
    //   bits prefix_n_vars..n_vars-1 (x_hi) → r_be[prefix_n_vars..n_vars]
    const allocator = testing.allocator;
    const F = BN254Scalar;

    const n_vars = 4;
    const prefix_n_vars = 2;
    const suffix_n_vars = 2;
    const T: usize = 1 << n_vars;
    const prefix_len: usize = 1 << prefix_n_vars;
    const suffix_len: usize = 1 << suffix_n_vars;

    // Full BE challenge
    var r_be = [4]F{ F.fromU64(17), F.fromU64(31), F.fromU64(7), F.fromU64(53) };
    const eq_full = try computeEqTable(F, allocator, &r_be, n_vars);
    defer allocator.free(eq_full);

    // Split: prefix (x_lo bits) uses r_be[0..prefix_n_vars]
    var r_lo_be = [2]F{ r_be[0], r_be[1] };
    const eq_lo = try computeEqTable(F, allocator, &r_lo_be, prefix_n_vars);
    defer allocator.free(eq_lo);

    // Suffix (x_hi bits) uses r_be[prefix_n_vars..n_vars]
    var r_hi_be = [2]F{ r_be[2], r_be[3] };
    const eq_hi = try computeEqTable(F, allocator, &r_hi_be, suffix_n_vars);
    defer allocator.free(eq_hi);

    // Verify: eq_full[x] == eq_lo[x_lo] * eq_hi[x_hi] for all x
    for (0..T) |x| {
        const x_lo = x & (prefix_len - 1);
        const x_hi = x >> prefix_n_vars;
        const product = eq_lo[x_lo].mul(eq_hi[x_hi]);
        try testing.expect(eq_full[x].eql(product));
    }

    // Also verify: Σ_{x_hi} f(x_lo, x_hi) * eq_hi[x_hi] correctly folds suffix dimension
    var folded = [_]F{F.zero()} ** prefix_len;
    for (0..prefix_len) |x_lo| {
        for (0..suffix_len) |x_hi| {
            const x = x_lo + (x_hi << prefix_n_vars);
            folded[x_lo] = folded[x_lo].add(eq_hi[x_hi].mul(F.fromU64(@intCast(x))));
        }
    }
    // Verify: Σ_x_lo P[x_lo] * folded[x_lo] == Σ_x eq_full[x] * f(x)
    var sum_pq = F.zero();
    for (0..prefix_len) |x_lo| {
        sum_pq = sum_pq.add(eq_lo[x_lo].mul(folded[x_lo]));
    }
    var sum_direct = F.zero();
    for (0..T) |x| {
        sum_direct = sum_direct.add(eq_full[x].mul(F.fromU64(@intCast(x))));
    }
    try testing.expect(sum_pq.eql(sum_direct));
}

test "split-eq bind Phase 1 then Phase 2 matches flat eq bind" {
    // Verify that binding a split eq (Phase 1 prefix, then Phase 2 suffix)
    // produces the same result as binding the flat eq table.
    const allocator = testing.allocator;
    const F = BN254Scalar;

    const n_vars = 4;
    const prefix_n_vars = 2;
    const suffix_n_vars = 2;
    const prefix_len: usize = 1 << prefix_n_vars;
    const suffix_len: usize = 1 << suffix_n_vars;

    var r_be = [4]F{ F.fromU64(5), F.fromU64(13), F.fromU64(3), F.fromU64(19) };
    const challenges = [4]F{ F.fromU64(7), F.fromU64(11), F.fromU64(2), F.fromU64(17) };

    // Build flat eq table and bind sequentially
    var eq_flat = try computeEqTable(F, allocator, &r_be, n_vars);
    defer allocator.free(eq_flat);

    var flat_len: usize = 1 << n_vars;
    for (challenges) |ch| {
        const half = flat_len / 2;
        for (0..half) |j| {
            eq_flat[j] = eq_flat[2 * j].add(ch.mul(eq_flat[2 * j + 1].sub(eq_flat[2 * j])));
        }
        flat_len = half;
    }
    const flat_final = eq_flat[0];

    // Split: prefix uses r_be[0..prefix_n_vars], suffix uses r_be[prefix_n_vars..]
    var r_lo_be = [2]F{ r_be[0], r_be[1] };
    var eq_lo = try computeEqTable(F, allocator, &r_lo_be, prefix_n_vars);
    defer allocator.free(eq_lo);

    var r_hi_be = [2]F{ r_be[2], r_be[3] };
    var eq_hi = try computeEqTable(F, allocator, &r_hi_be, suffix_n_vars);
    defer allocator.free(eq_hi);

    // Phase 1: bind prefix rounds on eq_lo
    var lo_len = prefix_len;
    for (0..prefix_n_vars) |round| {
        const half = lo_len / 2;
        for (0..half) |j| {
            eq_lo[j] = eq_lo[2 * j].add(challenges[round].mul(eq_lo[2 * j + 1].sub(eq_lo[2 * j])));
        }
        lo_len = half;
    }
    const eq_lo_scalar = eq_lo[0];

    // Phase 2: scale eq_hi by eq_lo scalar and bind suffix rounds
    for (0..suffix_len) |j| {
        eq_hi[j] = eq_hi[j].mul(eq_lo_scalar);
    }
    var hi_len = suffix_len;
    for (0..suffix_n_vars) |round| {
        const half = hi_len / 2;
        for (0..half) |j| {
            eq_hi[j] = eq_hi[2 * j].add(challenges[prefix_n_vars + round].mul(eq_hi[2 * j + 1].sub(eq_hi[2 * j])));
        }
        hi_len = half;
    }
    const split_final = eq_hi[0];

    try testing.expect(flat_final.eql(split_final));
}

test "P*Q sum matches flat polynomial sum" {
    // Verify that Σ P[x_lo] * Q[x_lo] == Σ_x eq(r, x) * f(x)
    // where Q[x_lo] = Σ_{x_hi} eq_hi(r_hi, x_hi) * f(x_lo, x_hi)
    // This is the IncClaimReduction Phase 1 correctness property.
    const allocator = testing.allocator;
    const F = BN254Scalar;

    const n_vars = 6;
    const prefix_n_vars = 3;
    const suffix_n_vars = 3;
    const T: usize = 1 << n_vars;
    const prefix_len: usize = 1 << prefix_n_vars;
    const suffix_len: usize = 1 << suffix_n_vars;

    var r_be = [6]F{
        F.fromU64(3), F.fromU64(7), F.fromU64(11),
        F.fromU64(17), F.fromU64(23), F.fromU64(29),
    };

    const eq_full = try computeEqTable(F, allocator, &r_be, n_vars);
    defer allocator.free(eq_full);

    // Prefix uses r_be[0..prefix_n_vars], suffix uses r_be[prefix_n_vars..]
    var r_lo_be = [3]F{ r_be[0], r_be[1], r_be[2] };
    const eq_lo = try computeEqTable(F, allocator, &r_lo_be, prefix_n_vars);
    defer allocator.free(eq_lo);

    var r_hi_be = [3]F{ r_be[3], r_be[4], r_be[5] };
    const eq_hi = try computeEqTable(F, allocator, &r_hi_be, suffix_n_vars);
    defer allocator.free(eq_hi);

    // f(x) = x^2 + 3x + 1 (arbitrary polynomial for testing)
    var f_vals = try allocator.alloc(F, T);
    defer allocator.free(f_vals);
    for (0..T) |x| {
        const xf = F.fromU64(@intCast(x));
        f_vals[x] = xf.mul(xf).add(F.fromU64(3).mul(xf)).add(F.one());
    }

    // Q[x_lo] = Σ_{x_hi} eq_hi[x_hi] * f(x_lo + x_hi << prefix_n_vars)
    var Q = try allocator.alloc(F, prefix_len);
    defer allocator.free(Q);
    for (0..prefix_len) |x_lo| {
        Q[x_lo] = F.zero();
        for (0..suffix_len) |x_hi| {
            const x = x_lo + (x_hi << prefix_n_vars);
            Q[x_lo] = Q[x_lo].add(eq_hi[x_hi].mul(f_vals[x]));
        }
    }

    // Σ P[x_lo] * Q[x_lo]
    var sum_pq = F.zero();
    for (0..prefix_len) |x_lo| {
        sum_pq = sum_pq.add(eq_lo[x_lo].mul(Q[x_lo]));
    }

    // Σ eq_full[x] * f(x)
    var sum_direct = F.zero();
    for (0..T) |x| {
        sum_direct = sum_direct.add(eq_full[x].mul(f_vals[x]));
    }

    try testing.expect(sum_pq.eql(sum_direct));
}

test "P*Q Phase 1 sumcheck round polynomial matches flat" {
    // Verify that the Phase 1 round polynomial from the P*Q factorization
    // produces the same evaluations as computing from the flat polynomial.
    const allocator = testing.allocator;
    const F = BN254Scalar;

    const n_vars = 4;
    const prefix_n_vars = 2;
    const suffix_n_vars = 2;
    const T: usize = 1 << n_vars;
    const prefix_len: usize = 1 << prefix_n_vars;
    const suffix_len: usize = 1 << suffix_n_vars;

    var r_be = [4]F{ F.fromU64(5), F.fromU64(13), F.fromU64(3), F.fromU64(19) };

    // Build flat polynomial: poly[x] = eq(r, x) * f(x)
    const eq_full = try computeEqTable(F, allocator, &r_be, n_vars);
    defer allocator.free(eq_full);

    // f(x) = x + 1
    var poly = try allocator.alloc(F, T);
    defer allocator.free(poly);
    for (0..T) |x| {
        poly[x] = eq_full[x].mul(F.fromU64(@intCast(x + 1)));
    }

    // Flat round 1: p(0) = Σ poly[2j], p(1) = Σ poly[2j+1]
    var flat_p0 = F.zero();
    var flat_p1 = F.zero();
    for (0..T / 2) |j| {
        flat_p0 = flat_p0.add(poly[2 * j]);
        flat_p1 = flat_p1.add(poly[2 * j + 1]);
    }

    // Split: P * Q version (prefix = r_be[0..2], suffix = r_be[2..4])
    var r_lo_be = [2]F{ r_be[0], r_be[1] };
    const P = try computeEqTable(F, allocator, &r_lo_be, prefix_n_vars);
    defer allocator.free(P);

    var r_hi_be = [2]F{ r_be[2], r_be[3] };
    const eq_hi = try computeEqTable(F, allocator, &r_hi_be, suffix_n_vars);
    defer allocator.free(eq_hi);

    var Q = try allocator.alloc(F, prefix_len);
    defer allocator.free(Q);
    for (0..prefix_len) |x_lo| {
        Q[x_lo] = F.zero();
        for (0..suffix_len) |x_hi| {
            const x = x_lo + (x_hi << prefix_n_vars);
            Q[x_lo] = Q[x_lo].add(eq_hi[x_hi].mul(F.fromU64(@intCast(x + 1))));
        }
    }

    // Phase 1 round 1: p(t) = Σ_{x_lo} P(x_lo, t) * Q(x_lo, t)
    // P(x_lo, 0) = P[2*x_lo], P(x_lo, 1) = P[2*x_lo+1] (standard MLE bind)
    // Q same structure
    var split_p0 = F.zero();
    var split_p1 = F.zero();
    const half = prefix_len / 2;
    for (0..half) |j| {
        split_p0 = split_p0.add(P[2 * j].mul(Q[2 * j]));
        split_p1 = split_p1.add(P[2 * j + 1].mul(Q[2 * j + 1]));
    }

    try testing.expect(flat_p0.eql(split_p0));
    try testing.expect(flat_p1.eql(split_p1));
}

test "HammingBooleanity split-eq: Phase 1 sum matches flat" {
    // HammingBooleanity computes Σ_x eq(r, x) * H(x) * (H(x) - 1)
    // Verify split-eq Phase 1 round poly matches flat computation.
    const allocator = testing.allocator;
    const F = BN254Scalar;

    const n_vars = 4;
    const prefix_n_vars = 2;
    const suffix_n_vars = 2;
    const T: usize = 1 << n_vars;
    const prefix_len: usize = 1 << prefix_n_vars;
    const suffix_len: usize = 1 << suffix_n_vars;

    var r_be = [4]F{ F.fromU64(11), F.fromU64(23), F.fromU64(7), F.fromU64(41) };

    // Build flat eq
    const eq_full = try computeEqTable(F, allocator, &r_be, n_vars);
    defer allocator.free(eq_full);

    // H(x) = some test values (simulating Hamming weight or similar)
    var H = [16]F{
        F.fromU64(0), F.fromU64(1), F.fromU64(1), F.fromU64(2),
        F.fromU64(1), F.fromU64(2), F.fromU64(2), F.fromU64(3),
        F.fromU64(1), F.fromU64(2), F.fromU64(2), F.fromU64(3),
        F.fromU64(2), F.fromU64(3), F.fromU64(3), F.fromU64(4),
    };

    // Flat sum: Σ eq(r,x) * H(x) * (H(x) - 1) for degree 3 sumcheck
    // Round 1: p(t) at t=0 and t=1
    var flat_p0 = F.zero();
    var flat_p1 = F.zero();
    for (0..T / 2) |j| {
        flat_p0 = flat_p0.add(eq_full[2 * j].mul(H[2 * j]).mul(H[2 * j].sub(F.one())));
        flat_p1 = flat_p1.add(eq_full[2 * j + 1].mul(H[2 * j + 1]).mul(H[2 * j + 1].sub(F.one())));
    }

    // Split-eq: prefix = r_be[0..2], suffix = r_be[2..4]
    var r_lo_be = [2]F{ r_be[0], r_be[1] };
    const eq_lo = try computeEqTable(F, allocator, &r_lo_be, prefix_n_vars);
    defer allocator.free(eq_lo);

    var r_hi_be = [2]F{ r_be[2], r_be[3] };
    const eq_hi = try computeEqTable(F, allocator, &r_hi_be, suffix_n_vars);
    defer allocator.free(eq_hi);

    // Split round 1 (prefix dimension, bit 0):
    // p(t) = Σ_{x_lo_rest, x_hi} eq_lo(x_lo_rest, t) * eq_hi(x_hi) * H * (H-1)
    // At t=0: sum over even x_lo indices; at t=1: sum over odd x_lo indices
    var split_p0 = F.zero();
    var split_p1 = F.zero();
    const half_lo = prefix_len / 2;
    for (0..half_lo) |j_lo| {
        for (0..suffix_len) |j_hi| {
            const x0 = 2 * j_lo + (j_hi << prefix_n_vars);
            const x1 = 2 * j_lo + 1 + (j_hi << prefix_n_vars);
            const eq_term = eq_lo[2 * j_lo].mul(eq_hi[j_hi]);
            const eq_term1 = eq_lo[2 * j_lo + 1].mul(eq_hi[j_hi]);
            split_p0 = split_p0.add(eq_term.mul(H[x0]).mul(H[x0].sub(F.one())));
            split_p1 = split_p1.add(eq_term1.mul(H[x1]).mul(H[x1].sub(F.one())));
        }
    }

    try testing.expect(flat_p0.eql(split_p0));
    try testing.expect(flat_p1.eql(split_p1));
}

test "IncClaimReduction Phase 1→2 transition: folded suffix matches flat" {
    // Verify that the Phase 1→2 transition math produces the same result as flat computation.
    // All eq tables use LE convention (matching the actual prover which reverses BE→LE first).
    const allocator = testing.allocator;
    const F = BN254Scalar;

    const n_vars = 4;
    const prefix_n_vars = 2;
    const suffix_n_vars = 2;
    const T: usize = 1 << n_vars;
    const prefix_len: usize = 1 << prefix_n_vars;
    const suffix_len: usize = 1 << suffix_n_vars;

    const gamma = F.fromU64(13);
    const challenges = [2]F{ F.fromU64(7), F.fromU64(11) }; // prefix sumcheck challenges

    // 4 opening points in LE order (simulates the prover's reversed BE→LE points).
    // In the prover: r_cycle_rev[i] = r_cycle_be[n_vars - 1 - i].
    // Here we just define them directly in LE.
    var points_le: [4][4]F = undefined;
    points_le[0] = .{ F.fromU64(23), F.fromU64(5), F.fromU64(17), F.fromU64(3) };
    points_le[1] = .{ F.fromU64(19), F.fromU64(2), F.fromU64(11), F.fromU64(7) };
    points_le[2] = .{ F.fromU64(37), F.fromU64(31), F.fromU64(29), F.fromU64(13) };
    points_le[3] = .{ F.fromU64(53), F.fromU64(47), F.fromU64(43), F.fromU64(41) };

    // Build full eq tables for each point (LE input to computeEqTable)
    var eq_full: [4][]F = undefined;
    for (0..4) |i| {
        eq_full[i] = try computeEqTable(F, allocator, &points_le[i], n_vars);
    }
    defer for (0..4) |i| allocator.free(eq_full[i]);

    // Flat approach: eq_ram[x] = eq_0[x] + gamma*eq_1[x], eq_rd[x] = eq_2[x] + gamma*eq_3[x]
    // Then bind prefix variables with challenges to get suffix-sized arrays.
    var flat_eq_ram = try allocator.alloc(F, T);
    defer allocator.free(flat_eq_ram);
    var flat_eq_rd = try allocator.alloc(F, T);
    defer allocator.free(flat_eq_rd);
    for (0..T) |x| {
        flat_eq_ram[x] = eq_full[0][x].add(gamma.mul(eq_full[1][x]));
        flat_eq_rd[x] = eq_full[2][x].add(gamma.mul(eq_full[3][x]));
    }

    // Bind prefix_n_vars rounds (round 0 binds bit 0, round 1 binds bit 1)
    var flat_len: usize = T;
    for (challenges) |ch| {
        const half = flat_len / 2;
        for (0..half) |j| {
            flat_eq_ram[j] = flat_eq_ram[2 * j].add(ch.mul(flat_eq_ram[2 * j + 1].sub(flat_eq_ram[2 * j])));
            flat_eq_rd[j] = flat_eq_rd[2 * j].add(ch.mul(flat_eq_rd[2 * j + 1].sub(flat_eq_rd[2 * j])));
        }
        flat_len = half;
    }

    // Split approach: eq_lo from first prefix_n_vars LE vars, eq_hi from the rest.
    // This mirrors the prover's init which does:
    //   P[i] = computeEqTable(rev_lo, prefix_n_vars) where rev_lo[k] = points_be[n-1-k]
    //   eq_hi[i] = computeEqTable(rev_hi, suffix_n_vars) where rev_hi[k] = points_be[suffix-1-k]
    // In LE terms: lo = points_le[0..prefix_n_vars], hi = points_le[prefix_n_vars..n_vars]
    var eq_hi: [4][]F = undefined;
    for (0..4) |i| {
        var r_hi: [2]F = undefined;
        for (0..suffix_n_vars) |k| r_hi[k] = points_le[i][prefix_n_vars + k];
        eq_hi[i] = try computeEqTable(F, allocator, &r_hi, suffix_n_vars);
    }
    defer for (0..4) |i| allocator.free(eq_hi[i]);

    // Prefix scalars: eq(challenges, point_lo_i) where point_lo = points_le[0..prefix_n_vars]
    var eq_prefix_scalars: [4]F = undefined;
    for (0..4) |i| {
        var result = F.one();
        for (0..prefix_n_vars) |k| {
            const a = challenges[k];
            const b = points_le[i][k];
            const prod = a.mul(b);
            result = result.mul(prod.add(prod).add(F.one()).sub(a.add(b)));
        }
        eq_prefix_scalars[i] = result;
    }

    // Build split eq arrays and compare
    for (0..suffix_len) |x_hi| {
        const split_ram = eq_prefix_scalars[0].mul(eq_hi[0][x_hi]).add(gamma.mul(eq_prefix_scalars[1].mul(eq_hi[1][x_hi])));
        const split_rd = eq_prefix_scalars[2].mul(eq_hi[2][x_hi]).add(gamma.mul(eq_prefix_scalars[3].mul(eq_hi[3][x_hi])));
        try testing.expect(flat_eq_ram[x_hi].eql(split_ram));
        try testing.expect(flat_eq_rd[x_hi].eql(split_rd));
    }

    // Also verify the inc folding: Σ_{x_lo} eq_prefix[x_lo] * f(x_lo, x_hi) matches
    // flat bind of f(x) over prefix variables.
    const eq_prefix_table = try computeEqTable(F, allocator, &challenges, prefix_n_vars);
    defer allocator.free(eq_prefix_table);

    // f(x) = x + 1 (synthetic)
    var f_vals = try allocator.alloc(F, T);
    defer allocator.free(f_vals);
    for (0..T) |x| f_vals[x] = F.fromU64(@intCast(x + 1));

    // Flat bind of f over prefix
    var f_flat = try allocator.alloc(F, T);
    defer allocator.free(f_flat);
    @memcpy(f_flat, f_vals);
    var f_len: usize = T;
    for (challenges) |ch| {
        const half = f_len / 2;
        for (0..half) |j| {
            f_flat[j] = f_flat[2 * j].add(ch.mul(f_flat[2 * j + 1].sub(f_flat[2 * j])));
        }
        f_len = half;
    }

    // Split fold: Σ_{x_lo} eq_prefix[x_lo] * f(x_lo + x_hi << prefix_n_vars)
    for (0..suffix_len) |x_hi| {
        var acc = F.zero();
        for (0..prefix_len) |x_lo| {
            const x = x_lo + (x_hi << prefix_n_vars);
            acc = acc.add(eq_prefix_table[x_lo].mul(f_vals[x]));
        }
        try testing.expect(f_flat[x_hi].eql(acc));
    }
}

test "BytecodeReadRaf split-eq F_s: inner*outer matches flat eq pushforward" {
    // Verify F_s[pc] = Σ_c eq(r_cycle, c) * δ(PC(c)=pc) is the same whether computed
    // via a flat T-sized eq table or via the split-eq double loop with touched-PC tracking.
    const allocator = testing.allocator;
    const F = BN254Scalar;

    const n_vars = 4;
    const T: usize = 1 << n_vars;
    const lo_bits = n_vars / 2;
    const hi_bits = n_vars - lo_bits;
    const in_len: usize = 1 << lo_bits;
    const out_len: usize = 1 << hi_bits;
    const bytecode_K: usize = 8;

    // PC map: cycle c → pc_idx (some synthetic mapping)
    var pc_map_arr: [T]usize = undefined;
    for (0..T) |c| {
        pc_map_arr[c] = (c * 3 + 1) % bytecode_K;
    }

    // r_cycle in LE order (r[0]→LSB, as used by computeEqTable)
    var r_le = [4]F{ F.fromU64(5), F.fromU64(17), F.fromU64(31), F.fromU64(43) };

    // Method 1: Flat computation with full T-sized eq table
    const eq_flat = try computeEqTable(F, allocator, &r_le, n_vars);
    defer allocator.free(eq_flat);

    var F_s_flat: [bytecode_K]F = .{F.zero()} ** bytecode_K;
    for (0..T) |c| {
        F_s_flat[pc_map_arr[c]] = F_s_flat[pc_map_arr[c]].add(eq_flat[c]);
    }

    // Method 2: Split-eq double loop (same algorithm as BytecodeReadRafProver.init)
    // Split LE points into lo and hi halves

    var r_lo_arr = [2]F{ r_le[0], r_le[1] };
    const E_lo = try computeEqTable(F, allocator, &r_lo_arr, lo_bits);
    defer allocator.free(E_lo);

    var r_hi_arr = [2]F{ r_le[2], r_le[3] };
    const E_hi = try computeEqTable(F, allocator, &r_hi_arr, hi_bits);
    defer allocator.free(E_hi);

    var F_s_split: [bytecode_K]F = .{F.zero()} ** bytecode_K;
    var inner_buf: [bytecode_K]F = .{F.zero()} ** bytecode_K;
    var touched_buf: [bytecode_K]usize = undefined;
    var touched_set: [bytecode_K]bool = .{false} ** bytecode_K;

    for (0..out_len) |c_hi| {
        var touched_count: usize = 0;

        for (0..in_len) |c_lo| {
            const c = c_lo + (c_hi << @intCast(lo_bits));
            const pc = pc_map_arr[c];
            if (!touched_set[pc]) {
                touched_set[pc] = true;
                touched_buf[touched_count] = pc;
                touched_count += 1;
            }
            inner_buf[pc] = inner_buf[pc].add(E_lo[c_lo]);
        }

        const e_hi_val = E_hi[c_hi];
        for (0..touched_count) |ti| {
            const pc = touched_buf[ti];
            F_s_split[pc] = F_s_split[pc].add(e_hi_val.mul(inner_buf[pc]));
            inner_buf[pc] = F.zero();
            touched_set[pc] = false;
        }
    }

    for (0..bytecode_K) |k| {
        try testing.expect(F_s_flat[k].eql(F_s_split[k]));
    }
}

test "IncClaimReduction full multi-round: split P/Q matches flat across phase transition" {
    // Full multi-round sumcheck simulation for IncClaimReduction:
    // Phase 1 (prefix rounds on P/Q) → transition → Phase 2 (suffix rounds on dense arrays).
    // The sumcheck is degree 2 (product of two linear factors: eq × inc).
    // We keep the factors separate in the flat reference to properly evaluate the degree-2
    // round polynomial at 3 points [s(0), s(1), s(2)].
    const allocator = testing.allocator;
    const F = BN254Scalar;

    const n_vars = 6;
    const prefix_n_vars = 3;
    const suffix_n_vars = 3;
    const T: usize = 1 << n_vars;
    const prefix_len: usize = 1 << prefix_n_vars;
    const suffix_len: usize = 1 << suffix_n_vars;

    const gamma = F.fromU64(13);
    const gamma_sqr = gamma.mul(gamma);

    // 4 opening points in LE order
    const points_le = [4][6]F{
        .{ F.fromU64(3), F.fromU64(7), F.fromU64(11), F.fromU64(17), F.fromU64(23), F.fromU64(29) },
        .{ F.fromU64(5), F.fromU64(13), F.fromU64(19), F.fromU64(31), F.fromU64(37), F.fromU64(41) },
        .{ F.fromU64(2), F.fromU64(43), F.fromU64(47), F.fromU64(53), F.fromU64(59), F.fromU64(61) },
        .{ F.fromU64(67), F.fromU64(71), F.fromU64(73), F.fromU64(79), F.fromU64(83), F.fromU64(89) },
    };

    // Synthetic inc values
    var ram_inc_vals: [T]F = undefined;
    var rd_inc_vals: [T]F = undefined;
    for (0..T) |x| {
        ram_inc_vals[x] = F.fromU64(@intCast(x + 1));
        rd_inc_vals[x] = F.fromU64(@intCast(2 * x + 3));
    }

    // Build flat eq tables
    var eq_full: [4][]F = undefined;
    for (0..4) |i| {
        eq_full[i] = try computeEqTable(F, allocator, @constCast(&points_le[i]), n_vars);
    }
    defer for (0..4) |i| allocator.free(eq_full[i]);

    // Flat: keep eq and inc separate (4 eq arrays, 2 inc arrays) for degree-2 round poly
    var flat_ram_inc = try allocator.alloc(F, T);
    defer allocator.free(flat_ram_inc);
    var flat_rd_inc = try allocator.alloc(F, T);
    defer allocator.free(flat_rd_inc);
    @memcpy(flat_ram_inc, &ram_inc_vals);
    @memcpy(flat_rd_inc, &rd_inc_vals);

    // --- Split approach: build P, Q arrays ---
    var P: [4][]F = undefined;
    var eq_hi: [4][]F = undefined;
    for (0..4) |i| {
        var r_lo: [3]F = undefined;
        for (0..prefix_n_vars) |k| r_lo[k] = points_le[i][k];
        P[i] = try computeEqTable(F, allocator, &r_lo, prefix_n_vars);

        var r_hi: [3]F = undefined;
        for (0..suffix_n_vars) |k| r_hi[k] = points_le[i][prefix_n_vars + k];
        eq_hi[i] = try computeEqTable(F, allocator, &r_hi, suffix_n_vars);
    }
    defer for (0..4) |i| {
        allocator.free(P[i]);
        allocator.free(eq_hi[i]);
    };

    var Q: [4][]F = undefined;
    for (0..4) |i| {
        Q[i] = try allocator.alloc(F, prefix_len);
        for (0..prefix_len) |x_lo| {
            var acc = F.zero();
            for (0..suffix_len) |x_hi| {
                const x = x_lo + (x_hi << prefix_n_vars);
                const inc_val = if (i < 2) ram_inc_vals[x] else rd_inc_vals[x];
                acc = acc.add(eq_hi[i][x_hi].mul(inc_val));
            }
            Q[i][x_lo] = acc;
        }
    }
    defer for (0..4) |i| allocator.free(Q[i]);

    const gamma_cub = gamma_sqr.mul(gamma);
    const weights = [4]F{ F.one(), gamma, gamma_sqr, gamma_cub };

    var flat_len: usize = T;
    var p_len: usize = prefix_len;
    var challenges: [6]F = undefined;
    var in_phase2 = false;

    var p2_ram_inc: ?[]F = null;
    defer if (p2_ram_inc) |a| allocator.free(a);
    var p2_rd_inc: ?[]F = null;
    defer if (p2_rd_inc) |a| allocator.free(a);
    var p2_eq_ram: ?[]F = null;
    defer if (p2_eq_ram) |a| allocator.free(a);
    var p2_eq_rd: ?[]F = null;
    defer if (p2_eq_rd) |a| allocator.free(a);
    var p2_len: usize = 0;

    for (0..n_vars) |round| {
        const r = F.fromU64(@intCast(round * 7 + 3));
        challenges[round] = r;

        const flat_half = flat_len / 2;

        // --- Flat round poly (degree 2): 3 evaluation points ---
        // s(t) = Σ_j [ (eq_0(t) + γ·eq_1(t))·ram_inc(t) + γ²·(eq_2(t) + γ·eq_3(t))·rd_inc(t) ]
        var flat_evals: [3]F = .{ F.zero(), F.zero(), F.zero() };
        for (0..flat_half) |j| {
            // Values at t=0, t=1, t=2
            var eq_ram_at: [3]F = undefined;
            var eq_rd_at: [3]F = undefined;
            var ram_at: [3]F = undefined;
            var rd_at: [3]F = undefined;
            for (0..3) |t| {
                const tf = F.fromU64(@intCast(t));
                inline for (0..4) |k| {
                    const v0 = eq_full[k][2 * j];
                    const v1 = eq_full[k][2 * j + 1];
                    const interp = v0.add(tf.mul(v1.sub(v0)));
                    if (k == 0) eq_ram_at[t] = interp;
                    if (k == 1) eq_ram_at[t] = eq_ram_at[t].add(gamma.mul(interp));
                    if (k == 2) eq_rd_at[t] = interp;
                    if (k == 3) eq_rd_at[t] = eq_rd_at[t].add(gamma.mul(interp));
                }
                const r0 = flat_ram_inc[2 * j];
                const r1 = flat_ram_inc[2 * j + 1];
                ram_at[t] = r0.add(tf.mul(r1.sub(r0)));
                const d0 = flat_rd_inc[2 * j];
                const d1 = flat_rd_inc[2 * j + 1];
                rd_at[t] = d0.add(tf.mul(d1.sub(d0)));
            }
            for (0..3) |t| {
                flat_evals[t] = flat_evals[t].add(
                    ram_at[t].mul(eq_ram_at[t]).add(gamma_sqr.mul(rd_at[t].mul(eq_rd_at[t]))),
                );
            }
        }

        // --- Split round poly ---
        var split_evals: [3]F = .{ F.zero(), F.zero(), F.zero() };

        if (!in_phase2) {
            const half = p_len / 2;
            for (0..half) |j| {
                for (0..3) |t| {
                    const tf = F.fromU64(@intCast(t));
                    var term = F.zero();
                    for (0..4) |k| {
                        const p0 = P[k][2 * j];
                        const p1 = P[k][2 * j + 1];
                        const q0 = Q[k][2 * j];
                        const q1 = Q[k][2 * j + 1];
                        const p_t = p0.add(tf.mul(p1.sub(p0)));
                        const q_t = q0.add(tf.mul(q1.sub(q0)));
                        term = term.add(weights[k].mul(p_t.mul(q_t)));
                    }
                    split_evals[t] = split_evals[t].add(term);
                }
            }
        } else {
            const half = p2_len / 2;
            for (0..half) |j| {
                for (0..3) |t| {
                    const tf = F.fromU64(@intCast(t));
                    const ram_t = p2_ram_inc.?[2 * j].add(tf.mul(p2_ram_inc.?[2 * j + 1].sub(p2_ram_inc.?[2 * j])));
                    const eq_r_t = p2_eq_ram.?[2 * j].add(tf.mul(p2_eq_ram.?[2 * j + 1].sub(p2_eq_ram.?[2 * j])));
                    const rd_t = p2_rd_inc.?[2 * j].add(tf.mul(p2_rd_inc.?[2 * j + 1].sub(p2_rd_inc.?[2 * j])));
                    const eq_d_t = p2_eq_rd.?[2 * j].add(tf.mul(p2_eq_rd.?[2 * j + 1].sub(p2_eq_rd.?[2 * j])));
                    split_evals[t] = split_evals[t].add(
                        ram_t.mul(eq_r_t).add(gamma_sqr.mul(rd_t.mul(eq_d_t))),
                    );
                }
            }
        }

        for (0..3) |t| {
            try testing.expect(flat_evals[t].eql(split_evals[t]));
        }

        // --- Bind all arrays ---
        // Flat: bind 4 eq arrays + 2 inc arrays
        for (0..flat_half) |j| {
            for (0..4) |k| {
                eq_full[k][j] = eq_full[k][2 * j].add(r.mul(eq_full[k][2 * j + 1].sub(eq_full[k][2 * j])));
            }
            flat_ram_inc[j] = flat_ram_inc[2 * j].add(r.mul(flat_ram_inc[2 * j + 1].sub(flat_ram_inc[2 * j])));
            flat_rd_inc[j] = flat_rd_inc[2 * j].add(r.mul(flat_rd_inc[2 * j + 1].sub(flat_rd_inc[2 * j])));
        }
        flat_len = flat_half;

        if (!in_phase2) {
            if (p_len == 2) {
                // Transition to Phase 2
                const eq_prefix = try computeEqTable(F, allocator, challenges[0 .. round + 1], prefix_n_vars);
                defer allocator.free(eq_prefix);

                var eq_prefix_scalars: [4]F = undefined;
                for (0..4) |i| {
                    var result = F.one();
                    for (0..prefix_n_vars) |k| {
                        const a = challenges[k];
                        const b = points_le[i][k];
                        const prod = a.mul(b);
                        result = result.mul(prod.add(prod).add(F.one()).sub(a.add(b)));
                    }
                    eq_prefix_scalars[i] = result;
                }

                p2_eq_ram = try allocator.alloc(F, suffix_len);
                p2_eq_rd = try allocator.alloc(F, suffix_len);
                for (0..suffix_len) |x_hi| {
                    p2_eq_ram.?[x_hi] = eq_prefix_scalars[0].mul(eq_hi[0][x_hi]).add(
                        gamma.mul(eq_prefix_scalars[1].mul(eq_hi[1][x_hi])),
                    );
                    p2_eq_rd.?[x_hi] = eq_prefix_scalars[2].mul(eq_hi[2][x_hi]).add(
                        gamma.mul(eq_prefix_scalars[3].mul(eq_hi[3][x_hi])),
                    );
                }

                p2_ram_inc = try allocator.alloc(F, suffix_len);
                p2_rd_inc = try allocator.alloc(F, suffix_len);
                for (0..suffix_len) |x_hi| {
                    var acc_ram = F.zero();
                    var acc_rd = F.zero();
                    for (0..prefix_len) |x_lo| {
                        const x = x_lo + (x_hi << prefix_n_vars);
                        acc_ram = acc_ram.add(eq_prefix[x_lo].mul(ram_inc_vals[x]));
                        acc_rd = acc_rd.add(eq_prefix[x_lo].mul(rd_inc_vals[x]));
                    }
                    p2_ram_inc.?[x_hi] = acc_ram;
                    p2_rd_inc.?[x_hi] = acc_rd;
                }
                p2_len = suffix_len;
                in_phase2 = true;
            } else {
                const half = p_len / 2;
                for (0..4) |k| {
                    for (0..half) |j| {
                        P[k][j] = P[k][2 * j].add(r.mul(P[k][2 * j + 1].sub(P[k][2 * j])));
                        Q[k][j] = Q[k][2 * j].add(r.mul(Q[k][2 * j + 1].sub(Q[k][2 * j])));
                    }
                }
                p_len = half;
            }
        } else {
            const half = p2_len / 2;
            for (0..half) |j| {
                p2_ram_inc.?[j] = p2_ram_inc.?[2 * j].add(r.mul(p2_ram_inc.?[2 * j + 1].sub(p2_ram_inc.?[2 * j])));
                p2_rd_inc.?[j] = p2_rd_inc.?[2 * j].add(r.mul(p2_rd_inc.?[2 * j + 1].sub(p2_rd_inc.?[2 * j])));
                p2_eq_ram.?[j] = p2_eq_ram.?[2 * j].add(r.mul(p2_eq_ram.?[2 * j + 1].sub(p2_eq_ram.?[2 * j])));
                p2_eq_rd.?[j] = p2_eq_rd.?[2 * j].add(r.mul(p2_eq_rd.?[2 * j + 1].sub(p2_eq_rd.?[2 * j])));
            }
            p2_len = half;
        }
    }

    // Final scalar: split must match flat
    const flat_final = flat_ram_inc[0].mul(
        eq_full[0][0].add(gamma.mul(eq_full[1][0])),
    ).add(gamma_sqr.mul(flat_rd_inc[0].mul(
        eq_full[2][0].add(gamma.mul(eq_full[3][0])),
    )));
    const split_final = p2_ram_inc.?[0].mul(p2_eq_ram.?[0]).add(
        gamma_sqr.mul(p2_rd_inc.?[0].mul(p2_eq_rd.?[0])),
    );
    try testing.expect(flat_final.eql(split_final));
}

test "HammingBooleanity full multi-round: split-eq matches flat across phase transition" {
    // Full multi-round sumcheck simulation for HammingBooleanity:
    // Phase 1 (prefix rounds with factored eq_lo·eq_hi) → transition → Phase 2 (merged eq).
    // Verifies every round polynomial matches the flat (unsplit) computation.
    const allocator = testing.allocator;
    const F = BN254Scalar;

    const n_vars = 6;
    const prefix_n_vars = 3;
    const suffix_n_vars = 3;
    const T: usize = 1 << n_vars;
    const prefix_len: usize = 1 << prefix_n_vars;
    const suffix_len: usize = 1 << suffix_n_vars;

    // r_cycle in LE order
    var r_le = [6]F{
        F.fromU64(5), F.fromU64(13), F.fromU64(3),
        F.fromU64(19), F.fromU64(7), F.fromU64(11),
    };

    // H values: simulate Hamming weight (binary values for booleanity test)
    var H_flat: [T]F = undefined;
    var H_split: [T]F = undefined;
    for (0..T) |x| {
        // Mix of 0 and 1 with some non-boolean values to make test interesting
        const v: u64 = if (x % 5 == 0) 0 else if (x % 3 == 0) 1 else @intCast(x % 4);
        H_flat[x] = F.fromU64(v);
        H_split[x] = F.fromU64(v);
    }

    // Flat eq table
    var eq_flat = try computeEqTable(F, allocator, &r_le, n_vars);
    defer allocator.free(eq_flat);

    // Split eq tables
    var r_lo: [3]F = undefined;
    for (0..prefix_n_vars) |k| r_lo[k] = r_le[k];
    var eq_lo = try computeEqTable(F, allocator, &r_lo, prefix_n_vars);
    defer allocator.free(eq_lo);

    var r_hi: [3]F = undefined;
    for (0..suffix_n_vars) |k| r_hi[k] = r_le[prefix_n_vars + k];
    const eq_hi = try computeEqTable(F, allocator, &r_hi, suffix_n_vars);
    defer allocator.free(eq_hi);

    var flat_len: usize = T;
    var split_h_len: usize = T;
    var lo_len: usize = prefix_len;
    var in_phase2 = false;

    // Phase 2 state
    var eq_merged: ?[]F = null;
    defer if (eq_merged) |a| allocator.free(a);
    var merged_len: usize = 0;

    for (0..n_vars) |round| {
        const r = F.fromU64(@intCast(round * 11 + 2));
        const two = F.fromU64(2);
        const three = F.fromU64(3);

        // --- Flat round poly: [s(0), s(1), s(2), s(3)] ---
        const flat_half = flat_len / 2;
        var flat_evals: [4]F = .{ F.zero(), F.zero(), F.zero(), F.zero() };
        for (0..flat_half) |j| {
            const h0 = H_flat[2 * j];
            const h1 = H_flat[2 * j + 1];
            const h_delta = h1.sub(h0);
            const e0 = eq_flat[2 * j];
            const e1 = eq_flat[2 * j + 1];
            const e_delta = e1.sub(e0);

            flat_evals[0] = flat_evals[0].add(e0.mul(h0.mul(h0).sub(h0)));
            flat_evals[1] = flat_evals[1].add(e1.mul(h1.mul(h1).sub(h1)));

            const h_at_2 = h0.add(two.mul(h_delta));
            const e_at_2 = e0.add(two.mul(e_delta));
            flat_evals[2] = flat_evals[2].add(e_at_2.mul(h_at_2.mul(h_at_2).sub(h_at_2)));

            const h_at_3 = h0.add(three.mul(h_delta));
            const e_at_3 = e0.add(three.mul(e_delta));
            flat_evals[3] = flat_evals[3].add(e_at_3.mul(h_at_3.mul(h_at_3).sub(h_at_3)));
        }

        // --- Split round poly ---
        var split_evals: [4]F = .{ F.zero(), F.zero(), F.zero(), F.zero() };

        if (!in_phase2) {
            // Phase 1: double loop with factored eq = eq_lo(x_lo) * eq_hi(x_hi)
            const half_lo = lo_len / 2;
            for (0..suffix_len) |j_outer| {
                const eq_hi_val = eq_hi[j_outer];
                for (0..half_lo) |j_inner| {
                    const j = j_inner + j_outer * half_lo;
                    const h0 = H_split[2 * j];
                    const h1 = H_split[2 * j + 1];
                    const h_delta = h1.sub(h0);

                    const eq_lo_0 = eq_lo[2 * j_inner];
                    const eq_lo_1 = eq_lo[2 * j_inner + 1];
                    const e0 = eq_lo_0.mul(eq_hi_val);
                    const e1 = eq_lo_1.mul(eq_hi_val);
                    const e_delta = e1.sub(e0);

                    split_evals[0] = split_evals[0].add(e0.mul(h0.mul(h0).sub(h0)));
                    split_evals[1] = split_evals[1].add(e1.mul(h1.mul(h1).sub(h1)));

                    const h_at_2 = h0.add(two.mul(h_delta));
                    const e_at_2 = e0.add(two.mul(e_delta));
                    split_evals[2] = split_evals[2].add(e_at_2.mul(h_at_2.mul(h_at_2).sub(h_at_2)));

                    const h_at_3 = h0.add(three.mul(h_delta));
                    const e_at_3 = e0.add(three.mul(e_delta));
                    split_evals[3] = split_evals[3].add(e_at_3.mul(h_at_3.mul(h_at_3).sub(h_at_3)));
                }
            }
        } else {
            // Phase 2: flat loop with merged eq
            const half = split_h_len / 2;
            for (0..half) |j| {
                const h0 = H_split[2 * j];
                const h1 = H_split[2 * j + 1];
                const h_delta = h1.sub(h0);
                const e0 = eq_merged.?[2 * j];
                const e1 = eq_merged.?[2 * j + 1];
                const e_delta = e1.sub(e0);

                split_evals[0] = split_evals[0].add(e0.mul(h0.mul(h0).sub(h0)));
                split_evals[1] = split_evals[1].add(e1.mul(h1.mul(h1).sub(h1)));

                const h_at_2 = h0.add(two.mul(h_delta));
                const e_at_2 = e0.add(two.mul(e_delta));
                split_evals[2] = split_evals[2].add(e_at_2.mul(h_at_2.mul(h_at_2).sub(h_at_2)));

                const h_at_3 = h0.add(three.mul(h_delta));
                const e_at_3 = e0.add(three.mul(e_delta));
                split_evals[3] = split_evals[3].add(e_at_3.mul(h_at_3.mul(h_at_3).sub(h_at_3)));
            }
        }

        // All 4 evaluation points must match
        for (0..4) |k| {
            try testing.expect(flat_evals[k].eql(split_evals[k]));
        }

        // --- Bind ---
        // Flat: bind eq and H
        for (0..flat_half) |j| {
            eq_flat[j] = eq_flat[2 * j].add(r.mul(eq_flat[2 * j + 1].sub(eq_flat[2 * j])));
            H_flat[j] = H_flat[2 * j].add(r.mul(H_flat[2 * j + 1].sub(H_flat[2 * j])));
        }
        flat_len = flat_half;

        // Split: bind H always, plus eq_lo or merged eq
        const split_half = split_h_len / 2;
        for (0..split_half) |j| {
            H_split[j] = H_split[2 * j].add(r.mul(H_split[2 * j + 1].sub(H_split[2 * j])));
        }
        split_h_len = split_half;

        if (!in_phase2) {
            const half_lo = lo_len / 2;
            for (0..half_lo) |j| {
                eq_lo[j] = eq_lo[2 * j].add(r.mul(eq_lo[2 * j + 1].sub(eq_lo[2 * j])));
            }
            lo_len = half_lo;

            // Transition when eq_lo reaches length 1
            if (half_lo == 1) {
                const eq_lo_scalar = eq_lo[0];
                // Merge: eq_merged[j_hi] = eq_lo_scalar * eq_hi[j_hi]
                eq_merged = try allocator.alloc(F, suffix_len);
                for (0..suffix_len) |j| {
                    eq_merged.?[j] = eq_lo_scalar.mul(eq_hi[j]);
                }
                merged_len = suffix_len;
                in_phase2 = true;
            }
        } else {
            // Phase 2: bind merged eq
            const half = merged_len / 2;
            for (0..half) |j| {
                eq_merged.?[j] = eq_merged.?[2 * j].add(r.mul(eq_merged.?[2 * j + 1].sub(eq_merged.?[2 * j])));
            }
            merged_len = half;
        }
    }

    // Final scalars must match
    try testing.expect(H_flat[0].eql(H_split[0]));
    try testing.expect(eq_flat[0].eql(eq_merged.?[0]));
}
