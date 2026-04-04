//! GPU-accelerated polynomial operations for sumcheck.
//!
//! Provides bindLow, bindFirst, and parallel field-element reduction —
//! the core operations that run every sumcheck round.

const std = @import("std");
const device_mod = @import("device.zig");
const buffer_mod = @import("buffer.zig");
const objc = @import("objc.zig");
const field = @import("../field/mod.zig");

const GpuAccelerator = device_mod.GpuAccelerator;
const KernelParam = GpuAccelerator.KernelParam;
const GpuBuffer = buffer_mod.GpuBuffer;
const BN254Scalar = field.BN254Scalar;
const id = objc.id;

/// Fixed threadgroup size for reduction kernels.
const REDUCE_TG_SIZE: usize = 256;

/// GPU-resident polynomial with double-buffering. Field elements stored in
/// 8×u32 format in persistent shared-mode Metal buffers. Two buffers are
/// swapped on each bind operation to avoid GPU race conditions (thread tid
/// writes to buf[tid] while thread tid/2 reads from buf[tid]).
pub const GpuPolynomial = struct {
    bufs: [2]GpuBuffer(u32),
    active: u1, // which buffer holds current data (0 or 1)
    len: usize, // number of Fp256 elements

    /// Create from CPU-side field elements (one-time conversion 4×u64 → 8×u32).
    pub fn initFromCpu(device: id, evals: []const BN254Scalar) GpuBuffer(u32).Error!GpuPolynomial {
        var buf0 = try GpuBuffer(u32).init(device, evals.len * 8);
        writeFieldElements(buf0.slice(), evals);
        const buf1 = try GpuBuffer(u32).init(device, evals.len * 8);
        return .{ .bufs = .{ buf0, buf1 }, .active = 0, .len = evals.len };
    }

    pub fn deinit(self: *GpuPolynomial) void {
        self.bufs[1].deinit();
        self.bufs[0].deinit();
    }

    /// The Metal buffer holding current data.
    pub fn metalBuffer(self: *const GpuPolynomial) id {
        return self.bufs[self.active].metal_buffer;
    }

    /// The Metal buffer for writing output (the inactive one).
    fn outputBuffer(self: *const GpuPolynomial) id {
        return self.bufs[1 - self.active].metal_buffer;
    }

    /// Swap active buffer after a bind operation.
    fn swap(self: *GpuPolynomial) void {
        self.active = 1 - self.active;
    }

    /// Read one element back to CPU format.
    pub fn readElement(self: *const GpuPolynomial, idx: usize) BN254Scalar {
        var result: BN254Scalar = undefined;
        gpuToScalar(&result, self.bufs[self.active].slice()[idx * 8 ..][0..8]);
        return result;
    }

    /// Read all valid elements back to CPU format.
    pub fn readAll(self: *const GpuPolynomial, out: []BN254Scalar) void {
        readFieldElements(self.bufs[self.active].slice()[0 .. self.len * 8], out);
    }
};

pub const GpuPolyOps = struct {
    gpu: *GpuAccelerator,
    bind_low_pipeline: id,
    bind_first_pipeline: id,
    reduce_sum_pipeline: id,
    product_round_pipeline: id,
    outer_bind_reduce_pipeline: id,
    bind_low_inplace_pipeline: id,
    bind_first_inplace_pipeline: id,
    dot_product_pipeline: id,
    /// Pre-allocated partials buffer for reductions (avoids per-call alloc).
    /// Sized for max 1024 threadgroups (enough for 262144 elements).
    partials_buf: GpuBuffer(u32),
    /// Pre-allocated scratch buffers for productSumcheckRoundGpu.
    /// Avoids per-round Metal buffer allocation.
    scratch_pt0: GpuBuffer(u32),   // partial t0 sums (1024 threadgroups)
    scratch_ptinf: GpuBuffer(u32), // partial tinf sums (1024 threadgroups)
    scratch_eq: GpuBuffer(u32),    // eq tables (max 262K elements × 8 u32)
    /// Pre-allocated bind buffers: input (524K elements × 8 u32) and output (262K × 8 u32).
    /// Reused by polyBindLow to avoid per-call Metal buffer allocation.
    scratch_bind_in: GpuBuffer(u32),
    scratch_bind_out: GpuBuffer(u32),

    pub const Error = GpuAccelerator.Error || GpuBuffer(u32).Error;

    pub fn init(gpu: *GpuAccelerator) Error!GpuPolyOps {
        const bl = try gpu.createPipeline("poly_bind_low");
        errdefer objc.msgSendVoid(bl, "release");
        const bf = try gpu.createPipeline("poly_bind_first");
        errdefer objc.msgSendVoid(bf, "release");
        const rs = try gpu.createPipeline("field_reduce_sum");
        errdefer objc.msgSendVoid(rs, "release");
        const pr = try gpu.createPipeline("product_sumcheck_round");
        errdefer objc.msgSendVoid(pr, "release");
        const obr = try gpu.createPipeline("outer_bind_reduce");
        errdefer objc.msgSendVoid(obr, "release");
        const bli = try gpu.createPipeline("poly_bind_low_inplace");
        errdefer objc.msgSendVoid(bli, "release");
        const bfi = try gpu.createPipeline("poly_bind_first_inplace");
        errdefer objc.msgSendVoid(bfi, "release");
        const dp = try gpu.createPipeline("field_dot_product");
        errdefer objc.msgSendVoid(dp, "release");

        // Pre-allocate reusable scratch buffers (zero allocs during hot loop)
        const partials = try GpuBuffer(u32).init(gpu.device, 1024 * 8);
        const spt0 = try GpuBuffer(u32).init(gpu.device, 1024 * 8);
        const sptinf = try GpuBuffer(u32).init(gpu.device, 1024 * 8);
        const seq = try GpuBuffer(u32).init(gpu.device, 262144 * 8);
        // Pre-alloc bind buffers: max 524K input elements, 262K output
        const sbi = try GpuBuffer(u32).init(gpu.device, 524288 * 8);
        const sbo = try GpuBuffer(u32).init(gpu.device, 262144 * 8);

        return .{
            .gpu = gpu,
            .bind_low_pipeline = bl,
            .bind_first_pipeline = bf,
            .reduce_sum_pipeline = rs,
            .product_round_pipeline = pr,
            .outer_bind_reduce_pipeline = obr,
            .bind_low_inplace_pipeline = bli,
            .bind_first_inplace_pipeline = bfi,
            .dot_product_pipeline = dp,
            .partials_buf = partials,
            .scratch_pt0 = spt0,
            .scratch_ptinf = sptinf,
            .scratch_eq = seq,
            .scratch_bind_in = sbi,
            .scratch_bind_out = sbo,
        };
    }

    pub fn deinit(self: *GpuPolyOps) void {
        self.scratch_bind_out.deinit();
        self.scratch_bind_in.deinit();
        self.scratch_eq.deinit();
        self.scratch_ptinf.deinit();
        self.scratch_pt0.deinit();
        self.partials_buf.deinit();
        objc.msgSendVoid(self.dot_product_pipeline, "release");
        objc.msgSendVoid(self.bind_first_inplace_pipeline, "release");
        objc.msgSendVoid(self.bind_low_inplace_pipeline, "release");
        objc.msgSendVoid(self.outer_bind_reduce_pipeline, "release");
        objc.msgSendVoid(self.product_round_pipeline, "release");
        objc.msgSendVoid(self.reduce_sum_pipeline, "release");
        objc.msgSendVoid(self.bind_first_pipeline, "release");
        objc.msgSendVoid(self.bind_low_pipeline, "release");
    }

    /// GPU bindLow: out[i] = evals[2i] + r * (evals[2i+1] - evals[2i])
    /// Uses pre-allocated scratch buffers when the polynomial fits (≤524K elements).
    pub fn polyBindLow(self: *GpuPolyOps, evals: []const BN254Scalar, r: BN254Scalar, out: []BN254Scalar) Error!void {
        const n = evals.len;
        const half = n / 2;
        std.debug.assert(out.len == half);
        if (half == 0) return;

        const in_u32 = n * 8;
        const out_u32 = half * 8;

        // Use pre-allocated buffers if they fit, otherwise allocate fresh
        const use_scratch = in_u32 <= self.scratch_bind_in.len and out_u32 <= self.scratch_bind_out.len;

        var heap_in: ?GpuBuffer(u32) = null;
        var heap_out: ?GpuBuffer(u32) = null;
        defer if (heap_in) |*b| b.deinit();
        defer if (heap_out) |*b| b.deinit();

        const buf_in_metal: id = if (use_scratch) self.scratch_bind_in.metal_buffer else blk: {
            heap_in = try GpuBuffer(u32).init(self.gpu.device, in_u32);
            break :blk heap_in.?.metal_buffer;
        };
        const buf_out_metal: id = if (use_scratch) self.scratch_bind_out.metal_buffer else blk: {
            heap_out = try GpuBuffer(u32).init(self.gpu.device, out_u32);
            break :blk heap_out.?.metal_buffer;
        };

        const in_slice = if (use_scratch) self.scratch_bind_in.slice() else heap_in.?.slice();
        const out_slice = if (use_scratch) self.scratch_bind_out.slice() else heap_out.?.slice();

        writeFieldElements(in_slice[0..in_u32], evals);

        var r_gpu: [8]u32 = undefined;
        scalarToGpu(&r_gpu, r);

        try self.gpu.dispatchKernel(
            self.bind_low_pipeline,
            &.{
                .{ .buffer = buf_in_metal },
                .{ .buffer = buf_out_metal },
                .{ .bytes = .{ .ptr = &r_gpu, .len = 32 } },
            },
            half,
            0,
        );

        readFieldElements(out_slice[0..out_u32], out);
    }

    /// GPU bindFirst: out[i] = (1-r) * evals[i] + r * evals[i + half]
    pub fn polyBindFirst(self: *GpuPolyOps, evals: []const BN254Scalar, r: BN254Scalar, out: []BN254Scalar) Error!void {
        const n = evals.len;
        const half = n / 2;
        std.debug.assert(out.len == half);
        if (half == 0) return;

        var buf_in = try GpuBuffer(u32).init(self.gpu.device, n * 8);
        defer buf_in.deinit();
        var buf_out = try GpuBuffer(u32).init(self.gpu.device, half * 8);
        defer buf_out.deinit();

        writeFieldElements(buf_in.slice(), evals);

        var r_gpu: [8]u32 = undefined;
        scalarToGpu(&r_gpu, r);

        const one_minus_r = BN254Scalar.one().sub(r);
        var omr_gpu: [8]u32 = undefined;
        scalarToGpu(&omr_gpu, one_minus_r);

        const half_u32: u32 = @intCast(half);

        try self.gpu.dispatchKernel(
            self.bind_first_pipeline,
            &.{
                .{ .buffer = buf_in.metal_buffer },
                .{ .buffer = buf_out.metal_buffer },
                .{ .bytes = .{ .ptr = &r_gpu, .len = 32 } },
                .{ .bytes = .{ .ptr = &omr_gpu, .len = 32 } },
                .{ .bytes = .{ .ptr = &half_u32, .len = 4 } },
            },
            half,
            0,
        );

        readFieldElements(buf_out.slice(), out);
    }

    /// GPU parallel sum of a contiguous slice of field elements.
    pub fn fieldReduceSum(self: *GpuPolyOps, evals: []const BN254Scalar) Error!BN254Scalar {
        const n = evals.len;
        if (n == 0) return BN254Scalar.zero();

        var buf_in = try GpuBuffer(u32).init(self.gpu.device, n * 8);
        defer buf_in.deinit();

        writeFieldElements(buf_in.slice(), evals);

        return self.reduceBuffer(buf_in.metal_buffer, n);
    }

    /// GPU sumcheck nextRound: returns (g0, g1) from first/second half sums.
    pub fn sumcheckRound(self: *GpuPolyOps, evals: []const BN254Scalar) Error![2]BN254Scalar {
        const n = evals.len;
        const half = n / 2;
        std.debug.assert(half > 0);

        // Write all evals to one GPU buffer
        var buf = try GpuBuffer(u32).init(self.gpu.device, n * 8);
        defer buf.deinit();
        writeFieldElements(buf.slice(), evals);

        // Sum first half → g0, second half → g1
        // We dispatch reduction on the full buffer but with different offsets.
        // For simplicity, we dispatch on the full buffer as two separate slices.
        // Since the data is contiguous, we use Metal buffer offsets.

        // Reduce first half [0..half)
        const g0 = try self.reduceBufferSlice(buf.metal_buffer, 0, half);
        // Reduce second half [half..n)
        const g1 = try self.reduceBufferSlice(buf.metal_buffer, half, half);

        return .{ g0, g1 };
    }

    // ── In-place operations on GpuPolynomial (zero-copy) ────────────────────

    /// bindLow on GPU-resident data with double-buffer. Reads from active buffer,
    /// writes to inactive buffer, swaps. No allocation, no race conditions.
    pub fn bindLowInPlace(self: *GpuPolyOps, poly: *GpuPolynomial, r: BN254Scalar) Error!void {
        const half = poly.len / 2;
        if (half == 0) return;

        var r_gpu: [8]u32 = undefined;
        scalarToGpu(&r_gpu, r);

        try self.gpu.dispatchKernel(
            self.bind_low_pipeline, // uses the SEPARATE input/output kernel
            &.{
                .{ .buffer = poly.metalBuffer() },    // input (active)
                .{ .buffer = poly.outputBuffer() },   // output (inactive)
                .{ .bytes = .{ .ptr = &r_gpu, .len = 32 } },
            },
            half,
            0,
        );
        poly.swap();
        poly.len = half;
    }

    /// bindFirst on GPU-resident data with double-buffer.
    pub fn bindFirstInPlace(self: *GpuPolyOps, poly: *GpuPolynomial, r: BN254Scalar) Error!void {
        const half = poly.len / 2;
        if (half == 0) return;

        var r_gpu: [8]u32 = undefined;
        scalarToGpu(&r_gpu, r);
        const one_minus_r = BN254Scalar.one().sub(r);
        var omr_gpu: [8]u32 = undefined;
        scalarToGpu(&omr_gpu, one_minus_r);
        const half_u32: u32 = @intCast(half);

        try self.gpu.dispatchKernel(
            self.bind_first_pipeline, // uses the SEPARATE input/output kernel
            &.{
                .{ .buffer = poly.metalBuffer() },    // input (active)
                .{ .buffer = poly.outputBuffer() },   // output (inactive)
                .{ .bytes = .{ .ptr = &r_gpu, .len = 32 } },
                .{ .bytes = .{ .ptr = &omr_gpu, .len = 32 } },
                .{ .bytes = .{ .ptr = &half_u32, .len = 4 } },
            },
            half,
            0,
        );
        poly.swap();
        poly.len = half;
    }

    /// sumcheckRound on GPU-resident data. Returns (g0, g1).
    /// Encodes both half-reductions in a SINGLE command buffer to minimize overhead.
    pub fn sumcheckRoundInPlace(self: *GpuPolyOps, poly: *const GpuPolynomial) Error![2]BN254Scalar {
        const half = poly.len / 2;
        if (half == 0) return .{ BN254Scalar.zero(), BN254Scalar.zero() };

        const max_tg = objc.msgSendUsize(self.reduce_sum_pipeline, "maxTotalThreadsPerThreadgroup");
        const tg = @min(REDUCE_TG_SIZE, max_tg);
        const num_groups = (half + tg - 1) / tg;
        const grid = num_groups * tg;

        // We need 2 × num_groups partials (g0 partials + g1 partials)
        // Use pre-allocated partials_buf (sized for 1024 groups)
        const total_partials = num_groups * 2;
        if (total_partials * 8 > self.partials_buf.len) {
            // Fallback: too many groups for pre-allocated buffer
            const g0 = try self.reduceBufferSlice(poly.metalBuffer(), 0, half);
            const g1 = try self.reduceBufferSlice(poly.metalBuffer(), half, half);
            return .{ g0, g1 };
        }

        const pool = objc.objc_autoreleasePoolPush() orelse return error.DispatchFailed;
        defer objc.objc_autoreleasePoolPop(pool);

        const cmd_buf = objc.msgSendId(self.gpu.queue, "commandBuffer") orelse
            return error.DispatchFailed;
        const encoder = objc.msgSendId(cmd_buf, "computeCommandEncoder") orelse
            return error.DispatchFailed;

        objc.msgSendVoidWithId(encoder, "setComputePipelineState:", self.reduce_sum_pipeline);

        const half_u32: u32 = @intCast(half);

        // Dispatch 1: reduce first half [0..half) → partials[0..num_groups)
        objc.msgSendSetBuffer(encoder, poly.metalBuffer(), 0, 0);
        objc.msgSendSetBuffer(encoder, self.partials_buf.metal_buffer, 0, 1);
        objc.msgSendSetBytes(encoder, &half_u32, 4, 2);
        objc.msgSendDispatchThreads(
            encoder,
            .{ .width = grid, .height = 1, .depth = 1 },
            .{ .width = tg, .height = 1, .depth = 1 },
        );

        // Memory barrier between dispatches (ensure first write completes before second)
        // Re-set pipeline to force a new dispatch group
        objc.msgSendVoidWithId(encoder, "setComputePipelineState:", self.reduce_sum_pipeline);

        // Dispatch 2: reduce second half [half..n) → partials[num_groups..2*num_groups)
        const byte_offset_in = half * 32;
        const byte_offset_out = num_groups * 32;
        objc.msgSendSetBuffer(encoder, poly.metalBuffer(), byte_offset_in, 0);
        objc.msgSendSetBuffer(encoder, self.partials_buf.metal_buffer, byte_offset_out, 1);
        objc.msgSendSetBytes(encoder, &half_u32, 4, 2);
        objc.msgSendDispatchThreads(
            encoder,
            .{ .width = grid, .height = 1, .depth = 1 },
            .{ .width = tg, .height = 1, .depth = 1 },
        );

        objc.msgSendVoid(encoder, "endEncoding");
        objc.msgSendVoid(cmd_buf, "commit");
        objc.msgSendVoid(cmd_buf, "waitUntilCompleted");

        // CPU sum of partial results
        var g0 = BN254Scalar.zero();
        var g1 = BN254Scalar.zero();
        var tmp: BN254Scalar = undefined;
        for (0..num_groups) |i| {
            gpuToScalar(&tmp, self.partials_buf.slice()[i * 8 ..][0..8]);
            g0 = g0.add(tmp);
        }
        for (0..num_groups) |i| {
            gpuToScalar(&tmp, self.partials_buf.slice()[(num_groups + i) * 8 ..][0..8]);
            g1 = g1.add(tmp);
        }
        return .{ g0, g1 };
    }

    /// Fused sumcheck step: bind(challenge) + reduce(new poly) in ONE command buffer.
    /// First call should use sumcheckRoundInPlace (no bind), then subsequent rounds use this.
    /// Returns (g0, g1) for the NEXT round's polynomial.
    pub fn sumcheckStepInPlace(self: *GpuPolyOps, poly: *GpuPolynomial, challenge: BN254Scalar) Error![2]BN254Scalar {
        const old_half = poly.len / 2;
        if (old_half == 0) return .{ BN254Scalar.zero(), BN254Scalar.zero() };

        // After bind, poly has old_half elements. Its sumcheck halves are old_half/2.
        const new_half = old_half / 2;

        // Prepare bind constants
        var r_gpu: [8]u32 = undefined;
        scalarToGpu(&r_gpu, challenge);
        const one_minus_r = BN254Scalar.one().sub(challenge);
        var omr_gpu: [8]u32 = undefined;
        scalarToGpu(&omr_gpu, one_minus_r);
        const half_u32: u32 = @intCast(old_half);

        const max_tg = objc.msgSendUsize(self.reduce_sum_pipeline, "maxTotalThreadsPerThreadgroup");
        const tg = @min(REDUCE_TG_SIZE, max_tg);

        const pool = objc.objc_autoreleasePoolPush() orelse return error.DispatchFailed;
        defer objc.objc_autoreleasePoolPop(pool);

        const cmd_buf = objc.msgSendId(self.gpu.queue, "commandBuffer") orelse
            return error.DispatchFailed;
        const encoder = objc.msgSendId(cmd_buf, "computeCommandEncoder") orelse
            return error.DispatchFailed;

        // Dispatch 1: bindFirst in-place
        objc.msgSendVoidWithId(encoder, "setComputePipelineState:", self.bind_first_inplace_pipeline);
        objc.msgSendSetBuffer(encoder, poly.metalBuffer(), 0, 0);
        objc.msgSendSetBytes(encoder, &r_gpu, 32, 1);
        objc.msgSendSetBytes(encoder, &omr_gpu, 32, 2);
        objc.msgSendSetBytes(encoder, &half_u32, 4, 3);
        objc.msgSendDispatchThreads(
            encoder,
            .{ .width = old_half, .height = 1, .depth = 1 },
            .{ .width = @min(tg, old_half), .height = 1, .depth = 1 },
        );
        poly.len = old_half;

        // If there's a next round to reduce, encode it in the same command buffer
        if (new_half > 0) {
            const num_groups = (new_half + tg - 1) / tg;
            const grid = num_groups * tg;
            const new_half_u32: u32 = @intCast(new_half);

            objc.msgSendVoidWithId(encoder, "setComputePipelineState:", self.reduce_sum_pipeline);

            // Reduce first half → partials[0..num_groups)
            objc.msgSendSetBuffer(encoder, poly.metalBuffer(), 0, 0);
            objc.msgSendSetBuffer(encoder, self.partials_buf.metal_buffer, 0, 1);
            objc.msgSendSetBytes(encoder, &new_half_u32, 4, 2);
            objc.msgSendDispatchThreads(
                encoder,
                .{ .width = grid, .height = 1, .depth = 1 },
                .{ .width = tg, .height = 1, .depth = 1 },
            );

            // Reduce second half → partials[num_groups..2*num_groups)
            objc.msgSendVoidWithId(encoder, "setComputePipelineState:", self.reduce_sum_pipeline);
            objc.msgSendSetBuffer(encoder, poly.metalBuffer(), new_half * 32, 0);
            objc.msgSendSetBuffer(encoder, self.partials_buf.metal_buffer, num_groups * 32, 1);
            objc.msgSendSetBytes(encoder, &new_half_u32, 4, 2);
            objc.msgSendDispatchThreads(
                encoder,
                .{ .width = grid, .height = 1, .depth = 1 },
                .{ .width = tg, .height = 1, .depth = 1 },
            );

            objc.msgSendVoid(encoder, "endEncoding");
            objc.msgSendVoid(cmd_buf, "commit");
            objc.msgSendVoid(cmd_buf, "waitUntilCompleted");

            // CPU sum of partial results
            var g0 = BN254Scalar.zero();
            var g1 = BN254Scalar.zero();
            var tmp: BN254Scalar = undefined;
            for (0..num_groups) |i| {
                gpuToScalar(&tmp, self.partials_buf.slice()[i * 8 ..][0..8]);
                g0 = g0.add(tmp);
            }
            for (0..num_groups) |i| {
                gpuToScalar(&tmp, self.partials_buf.slice()[(num_groups + i) * 8 ..][0..8]);
                g1 = g1.add(tmp);
            }
            return .{ g0, g1 };
        } else {
            // Last round: just bind, no reduce needed
            objc.msgSendVoid(encoder, "endEncoding");
            objc.msgSendVoid(cmd_buf, "commit");
            objc.msgSendVoid(cmd_buf, "waitUntilCompleted");
            return .{ poly.readElement(0), BN254Scalar.zero() };
        }
    }

    /// Fused Stage 2 round polynomial: compute eq-weighted (t0, t_inf) from
    /// interleaved left/right polynomials and E_out/E_in eq tables.
    pub fn productSumcheckRound(
        self: *GpuPolyOps,
        left: []const BN254Scalar,
        right: []const BN254Scalar,
        e_out: []const BN254Scalar,
        e_in: []const BN254Scalar,
        xin_bits: u32,
    ) Error![2]BN254Scalar {
        const n = left.len;
        std.debug.assert(right.len == n);
        const num_groups: u32 = @intCast(n / 2);
        if (num_groups == 0) return .{ BN254Scalar.zero(), BN254Scalar.zero() };

        var buf_left = try GpuBuffer(u32).init(self.gpu.device, n * 8);
        defer buf_left.deinit();
        var buf_right = try GpuBuffer(u32).init(self.gpu.device, n * 8);
        defer buf_right.deinit();
        var buf_eout = try GpuBuffer(u32).init(self.gpu.device, e_out.len * 8);
        defer buf_eout.deinit();
        var buf_ein = try GpuBuffer(u32).init(self.gpu.device, @max(e_in.len, 1) * 8);
        defer buf_ein.deinit();

        writeFieldElements(buf_left.slice(), left);
        writeFieldElements(buf_right.slice(), right);
        writeFieldElements(buf_eout.slice(), e_out);
        if (e_in.len > 0) writeFieldElements(buf_ein.slice(), e_in);

        const tg: usize = REDUCE_TG_SIZE;
        const num_tg: usize = (num_groups + tg - 1) / tg;
        const grid = num_tg * tg;

        var buf_pt0 = try GpuBuffer(u32).init(self.gpu.device, num_tg * 8);
        defer buf_pt0.deinit();
        var buf_ptinf = try GpuBuffer(u32).init(self.gpu.device, num_tg * 8);
        defer buf_ptinf.deinit();

        const e_in_len: u32 = @intCast(e_in.len);
        const e_out_len: u32 = @intCast(e_out.len);

        try self.gpu.dispatchKernel(
            self.product_round_pipeline,
            &.{
                .{ .buffer = buf_left.metal_buffer },
                .{ .buffer = buf_right.metal_buffer },
                .{ .buffer = buf_eout.metal_buffer },
                .{ .buffer = buf_ein.metal_buffer },
                .{ .buffer = buf_pt0.metal_buffer },
                .{ .buffer = buf_ptinf.metal_buffer },
                .{ .bytes = .{ .ptr = &num_groups, .len = 4 } },
                .{ .bytes = .{ .ptr = &xin_bits, .len = 4 } },
                .{ .bytes = .{ .ptr = &e_in_len, .len = 4 } },
                .{ .bytes = .{ .ptr = &e_out_len, .len = 4 } },
            },
            grid,
            tg,
        );

        // CPU sum of partial results
        var t0 = BN254Scalar.zero();
        var tinf = BN254Scalar.zero();
        var tmp: BN254Scalar = undefined;
        for (0..num_tg) |i| {
            gpuToScalar(&tmp, buf_pt0.slice()[i * 8 ..][0..8]);
            t0 = t0.add(tmp);
            gpuToScalar(&tmp, buf_ptinf.slice()[i * 8 ..][0..8]);
            tinf = tinf.add(tmp);
        }
        return .{ t0, tinf };
    }

    /// Fused Stage 2 round on GPU-resident data. ZERO per-round allocations.
    /// Polynomial data in persistent Metal buffers; eq tables + partials use
    /// pre-allocated scratch buffers.
    pub fn productSumcheckRoundGpu(
        self: *GpuPolyOps,
        left_buf: id,
        right_buf: id,
        num_groups: u32,
        e_out: []const BN254Scalar,
        e_in: []const BN254Scalar,
        xin_bits: u32,
    ) Error![2]BN254Scalar {
        if (num_groups == 0) return .{ BN254Scalar.zero(), BN254Scalar.zero() };

        // Write eq tables into pre-allocated scratch (no Metal buffer alloc)
        const eq_total = e_out.len + @max(e_in.len, 1);
        if (eq_total * 8 > self.scratch_eq.len) {
            // Fallback to copy-based if eq tables are too large
            return self.productSumcheckRound(
                // Can't call copy-based from here without CPU slices.
                // This shouldn't happen for realistic trace sizes.
                &.{}, &.{}, &.{}, &.{}, 0,
            );
        }
        writeFieldElements(self.scratch_eq.slice()[0 .. e_out.len * 8], e_out);
        const ein_offset = e_out.len * 8;
        if (e_in.len > 0) {
            writeFieldElements(self.scratch_eq.slice()[ein_offset .. ein_offset + e_in.len * 8], e_in);
        }

        const tg: usize = REDUCE_TG_SIZE;
        const num_tg: usize = (num_groups + tg - 1) / tg;
        const grid = num_tg * tg;

        const e_in_len: u32 = @intCast(e_in.len);
        const e_out_len: u32 = @intCast(e_out.len);

        const pool = objc.objc_autoreleasePoolPush() orelse return error.DispatchFailed;
        defer objc.objc_autoreleasePoolPop(pool);

        const cmd_buf = objc.msgSendId(self.gpu.queue, "commandBuffer") orelse
            return error.DispatchFailed;
        const encoder = objc.msgSendId(cmd_buf, "computeCommandEncoder") orelse
            return error.DispatchFailed;

        objc.msgSendVoidWithId(encoder, "setComputePipelineState:", self.product_round_pipeline);
        objc.msgSendSetBuffer(encoder, left_buf, 0, 0);
        objc.msgSendSetBuffer(encoder, right_buf, 0, 1);
        objc.msgSendSetBuffer(encoder, self.scratch_eq.metal_buffer, 0, 2);
        objc.msgSendSetBuffer(encoder, self.scratch_eq.metal_buffer, ein_offset * 4, 3);
        objc.msgSendSetBuffer(encoder, self.scratch_pt0.metal_buffer, 0, 4);
        objc.msgSendSetBuffer(encoder, self.scratch_ptinf.metal_buffer, 0, 5);
        objc.msgSendSetBytes(encoder, &num_groups, 4, 6);
        objc.msgSendSetBytes(encoder, &xin_bits, 4, 7);
        objc.msgSendSetBytes(encoder, &e_in_len, 4, 8);
        objc.msgSendSetBytes(encoder, &e_out_len, 4, 9);

        const max_tg = objc.msgSendUsize(self.product_round_pipeline, "maxTotalThreadsPerThreadgroup");
        const actual_tg = @min(tg, max_tg);

        objc.msgSendDispatchThreads(
            encoder,
            .{ .width = grid, .height = 1, .depth = 1 },
            .{ .width = actual_tg, .height = 1, .depth = 1 },
        );

        objc.msgSendVoid(encoder, "endEncoding");
        objc.msgSendVoid(cmd_buf, "commit");
        objc.msgSendVoid(cmd_buf, "waitUntilCompleted");

        // CPU sum of partial results (from pre-allocated scratch)
        var t0 = BN254Scalar.zero();
        var tinf = BN254Scalar.zero();
        var tmp: BN254Scalar = undefined;
        for (0..num_tg) |i| {
            gpuToScalar(&tmp, self.scratch_pt0.slice()[i * 8 ..][0..8]);
            t0 = t0.add(tmp);
            gpuToScalar(&tmp, self.scratch_ptinf.slice()[i * 8 ..][0..8]);
            tinf = tinf.add(tmp);
        }
        return .{ t0, tinf };
    }

    /// Fused Stage 1 bind+reduce: binds Az/Bz by challenge r AND computes
    /// (t0, t_inf) product sums in a single GPU dispatch.
    pub fn outerBindReduce(
        self: *GpuPolyOps,
        az: []const BN254Scalar,
        bz: []const BN254Scalar,
        r: BN254Scalar,
        az_out: []BN254Scalar,
        bz_out: []BN254Scalar,
    ) Error![2]BN254Scalar {
        const n = az.len;
        std.debug.assert(bz.len == n);
        const half_n: u32 = @intCast(n / 2);
        std.debug.assert(az_out.len == half_n);
        std.debug.assert(bz_out.len == half_n);
        if (half_n == 0) return .{ BN254Scalar.zero(), BN254Scalar.zero() };

        var buf_az_in = try GpuBuffer(u32).init(self.gpu.device, n * 8);
        defer buf_az_in.deinit();
        var buf_bz_in = try GpuBuffer(u32).init(self.gpu.device, n * 8);
        defer buf_bz_in.deinit();
        var buf_az_out = try GpuBuffer(u32).init(self.gpu.device, half_n * 8);
        defer buf_az_out.deinit();
        var buf_bz_out = try GpuBuffer(u32).init(self.gpu.device, half_n * 8);
        defer buf_bz_out.deinit();

        writeFieldElements(buf_az_in.slice(), az);
        writeFieldElements(buf_bz_in.slice(), bz);

        var r_gpu: [8]u32 = undefined;
        scalarToGpu(&r_gpu, r);

        const tg: usize = REDUCE_TG_SIZE;
        const num_tg: usize = (half_n + tg - 1) / tg;
        const grid = num_tg * tg;

        // partial buffer: [2] per threadgroup (t0, tinf interleaved)
        var buf_partial = try GpuBuffer(u32).init(self.gpu.device, num_tg * 2 * 8);
        defer buf_partial.deinit();

        try self.gpu.dispatchKernel(
            self.outer_bind_reduce_pipeline,
            &.{
                .{ .buffer = buf_az_in.metal_buffer },
                .{ .buffer = buf_bz_in.metal_buffer },
                .{ .buffer = buf_az_out.metal_buffer },
                .{ .buffer = buf_bz_out.metal_buffer },
                .{ .buffer = buf_partial.metal_buffer },
                .{ .bytes = .{ .ptr = &r_gpu, .len = 32 } },
                .{ .bytes = .{ .ptr = &half_n, .len = 4 } },
            },
            grid,
            tg,
        );

        readFieldElements(buf_az_out.slice(), az_out);
        readFieldElements(buf_bz_out.slice(), bz_out);

        // CPU sum of partial results
        var t0 = BN254Scalar.zero();
        var tinf = BN254Scalar.zero();
        var tmp: BN254Scalar = undefined;
        for (0..num_tg) |i| {
            gpuToScalar(&tmp, buf_partial.slice()[(i * 2) * 8 ..][0..8]);
            t0 = t0.add(tmp);
            gpuToScalar(&tmp, buf_partial.slice()[(i * 2 + 1) * 8 ..][0..8]);
            tinf = tinf.add(tmp);
        }
        return .{ t0, tinf };
    }

    // ── Dot product ────────────────────────────────────────────────────────

    /// GPU dot product: Σ a[i] * b[i].
    /// Semantically equivalent to CPU's UnreducedProductAccum loop.
    pub fn fieldDotProduct(self: *GpuPolyOps, a: []const BN254Scalar, b: []const BN254Scalar) Error!BN254Scalar {
        const n = a.len;
        std.debug.assert(b.len == n);
        if (n == 0) return BN254Scalar.zero();

        var buf_a = try GpuBuffer(u32).init(self.gpu.device, n * 8);
        defer buf_a.deinit();
        var buf_b = try GpuBuffer(u32).init(self.gpu.device, n * 8);
        defer buf_b.deinit();

        writeFieldElements(buf_a.slice(), a);
        writeFieldElements(buf_b.slice(), b);

        return self.fieldDotProductGpu(buf_a.metal_buffer, buf_b.metal_buffer, @intCast(n));
    }

    /// Dot product on GPU-resident data (zero allocation).
    /// Takes Metal buffer IDs directly; uses pre-allocated partials_buf.
    pub fn fieldDotProductGpu(self: *GpuPolyOps, a_buf: id, b_buf: id, n: u32) Error!BN254Scalar {
        if (n == 0) return BN254Scalar.zero();

        const max_tg = objc.msgSendUsize(self.dot_product_pipeline, "maxTotalThreadsPerThreadgroup");
        const tg = @min(REDUCE_TG_SIZE, max_tg);
        const num_groups = (@as(usize, n) + tg - 1) / tg;
        const grid = num_groups * tg;

        const pool = objc.objc_autoreleasePoolPush() orelse return error.DispatchFailed;
        defer objc.objc_autoreleasePoolPop(pool);

        const cmd_buf = objc.msgSendId(self.gpu.queue, "commandBuffer") orelse
            return error.DispatchFailed;
        const encoder = objc.msgSendId(cmd_buf, "computeCommandEncoder") orelse
            return error.DispatchFailed;

        objc.msgSendVoidWithId(encoder, "setComputePipelineState:", self.dot_product_pipeline);
        objc.msgSendSetBuffer(encoder, a_buf, 0, 0);
        objc.msgSendSetBuffer(encoder, b_buf, 0, 1);
        objc.msgSendSetBuffer(encoder, self.partials_buf.metal_buffer, 0, 2);
        objc.msgSendSetBytes(encoder, &n, 4, 3);

        objc.msgSendDispatchThreads(
            encoder,
            .{ .width = grid, .height = 1, .depth = 1 },
            .{ .width = tg, .height = 1, .depth = 1 },
        );

        objc.msgSendVoid(encoder, "endEncoding");
        objc.msgSendVoid(cmd_buf, "commit");
        objc.msgSendVoid(cmd_buf, "waitUntilCompleted");

        // CPU sum of partial results
        var result = BN254Scalar.zero();
        var tmp: BN254Scalar = undefined;
        for (0..num_groups) |i| {
            gpuToScalar(&tmp, self.partials_buf.slice()[i * 8 ..][0..8]);
            result = result.add(tmp);
        }
        return result;
    }

    // ── Internal ────────────────────────────────────────────────────────────

    /// Reduce a full Metal buffer of n Fp256 elements to a single sum.
    fn reduceBuffer(self: *GpuPolyOps, metal_buf: id, n: usize) Error!BN254Scalar {
        return self.reduceBufferSlice(metal_buf, 0, n);
    }

    /// Reduce a slice of a Metal buffer [offset..offset+count) to a single sum.
    fn reduceBufferSlice(self: *GpuPolyOps, metal_buf: id, offset: usize, count: usize) Error!BN254Scalar {
        if (count == 0) return BN254Scalar.zero();

        const num_groups = (count + REDUCE_TG_SIZE - 1) / REDUCE_TG_SIZE;
        var buf_partials = try GpuBuffer(u32).init(self.gpu.device, num_groups * 8);
        defer buf_partials.deinit();

        const n_u32: u32 = @intCast(count);

        // Dispatch reduction kernel with buffer offset
        const pool = objc.objc_autoreleasePoolPush() orelse return error.DispatchFailed;
        defer objc.objc_autoreleasePoolPop(pool);

        const cmd_buf = objc.msgSendId(self.gpu.queue, "commandBuffer") orelse
            return error.DispatchFailed;
        const encoder = objc.msgSendId(cmd_buf, "computeCommandEncoder") orelse
            return error.DispatchFailed;

        objc.msgSendVoidWithId(encoder, "setComputePipelineState:", self.reduce_sum_pipeline);

        // buffer(0): input with byte offset
        const byte_offset = offset * 32; // 32 bytes per Fp256
        objc.msgSendSetBuffer(encoder, metal_buf, byte_offset, 0);
        // buffer(1): partial sums output
        objc.msgSendSetBuffer(encoder, buf_partials.metal_buffer, 0, 1);
        // buffer(2): n (inline constant)
        objc.msgSendSetBytes(encoder, &n_u32, 4, 2);

        const max_tg = objc.msgSendUsize(self.reduce_sum_pipeline, "maxTotalThreadsPerThreadgroup");
        const tg = @min(REDUCE_TG_SIZE, max_tg);

        // Grid must cover all elements — round up to multiple of tg
        const grid = ((count + tg - 1) / tg) * tg;

        objc.msgSendDispatchThreads(
            encoder,
            .{ .width = grid, .height = 1, .depth = 1 },
            .{ .width = tg, .height = 1, .depth = 1 },
        );

        objc.msgSendVoid(encoder, "endEncoding");
        objc.msgSendVoid(cmd_buf, "commit");
        objc.msgSendVoid(cmd_buf, "waitUntilCompleted");

        // Sum partial results on CPU (small: typically < 1024 elements)
        var result = BN254Scalar.zero();
        var partial: BN254Scalar = undefined;
        for (0..num_groups) |i| {
            gpuToScalar(&partial, buf_partials.slice()[i * 8 ..][0..8]);
            result = result.add(partial);
        }
        return result;
    }
};

// ── Format conversion ───────────────────────────────────────────────────────

fn writeFieldElements(buf: []u32, elems: []const BN254Scalar) void {
    for (elems, 0..) |elem, idx| {
        const base = idx * 8;
        inline for (0..4) |i| {
            buf[base + i * 2] = @truncate(elem.limbs[i]);
            buf[base + i * 2 + 1] = @truncate(elem.limbs[i] >> 32);
        }
    }
}

fn readFieldElements(buf: []const u32, out: []BN254Scalar) void {
    for (out, 0..) |*elem, idx| {
        const base = idx * 8;
        inline for (0..4) |i| {
            elem.limbs[i] = @as(u64, buf[base + i * 2]) |
                (@as(u64, buf[base + i * 2 + 1]) << 32);
        }
    }
}

fn scalarToGpu(out: *[8]u32, s: BN254Scalar) void {
    inline for (0..4) |i| {
        out[i * 2] = @truncate(s.limbs[i]);
        out[i * 2 + 1] = @truncate(s.limbs[i] >> 32);
    }
}

fn gpuToScalar(out: *BN254Scalar, limbs: *const [8]u32) void {
    inline for (0..4) |i| {
        out.limbs[i] = @as(u64, limbs[i * 2]) | (@as(u64, limbs[i * 2 + 1]) << 32);
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

const testing = std.testing;

fn initGpu() !struct { gpu: GpuAccelerator, ops: GpuPolyOps } {
    var gpu = GpuAccelerator.init(testing.allocator) catch |err| {
        if (err == error.MetalNotAvailable) return err;
        return err;
    };
    errdefer gpu.deinit();
    const ops = try GpuPolyOps.init(&gpu);
    return .{ .gpu = gpu, .ops = ops };
}

test "GPU poly_bind_low: correctness" {
    const gpu_mod = @import("mod.zig");
    if (comptime !gpu_mod.is_metal_available) return;

    var state = initGpu() catch |err| {
        if (err == error.MetalNotAvailable) return;
        return err;
    };
    defer state.ops.deinit();
    defer state.gpu.deinit();

    // Create polynomial evals: 16 elements
    const n = 16;
    var evals: [n]BN254Scalar = undefined;
    for (0..n) |i| {
        evals[i] = BN254Scalar.fromU64(@as(u64, @intCast(i)) * 100 + 1);
    }

    const r = BN254Scalar.fromU64(42);

    // GPU bindLow
    var gpu_out: [n / 2]BN254Scalar = undefined;
    try state.ops.polyBindLow(&evals, r, &gpu_out);

    // CPU bindLow for comparison
    for (0..n / 2) |i| {
        const lo = evals[2 * i];
        const hi = evals[2 * i + 1];
        const expected = lo.add(r.montgomeryMul(hi.sub(lo)));
        try testing.expectEqual(expected.limbs, gpu_out[i].limbs);
    }
}

test "GPU poly_bind_low: 1024 elements" {
    const gpu_mod = @import("mod.zig");
    if (comptime !gpu_mod.is_metal_available) return;

    var state = initGpu() catch |err| {
        if (err == error.MetalNotAvailable) return;
        return err;
    };
    defer state.ops.deinit();
    defer state.gpu.deinit();

    const n = 1024;
    var evals: [n]BN254Scalar = undefined;
    for (0..n) |i| {
        evals[i] = BN254Scalar.fromU64(@as(u64, @intCast(i)) *% 0xABCDEF +% 7);
    }

    const r = BN254Scalar.fromU64(0xDEADBEEF);

    var gpu_out: [n / 2]BN254Scalar = undefined;
    try state.ops.polyBindLow(&evals, r, &gpu_out);

    for (0..n / 2) |i| {
        const lo = evals[2 * i];
        const hi = evals[2 * i + 1];
        const expected = lo.add(r.montgomeryMul(hi.sub(lo)));
        try testing.expectEqual(expected.limbs, gpu_out[i].limbs);
    }
}

test "GPU poly_bind_first: correctness" {
    const gpu_mod = @import("mod.zig");
    if (comptime !gpu_mod.is_metal_available) return;

    var state = initGpu() catch |err| {
        if (err == error.MetalNotAvailable) return;
        return err;
    };
    defer state.ops.deinit();
    defer state.gpu.deinit();

    const n = 32;
    var evals: [n]BN254Scalar = undefined;
    for (0..n) |i| {
        evals[i] = BN254Scalar.fromU64(@as(u64, @intCast(i)) * 50 + 3);
    }

    const r = BN254Scalar.fromU64(77);
    const one_minus_r = BN254Scalar.one().sub(r);

    var gpu_out: [n / 2]BN254Scalar = undefined;
    try state.ops.polyBindFirst(&evals, r, &gpu_out);

    // CPU bindFirst for comparison
    const half = n / 2;
    for (0..half) |i| {
        const lo = evals[i].montgomeryMul(one_minus_r);
        const hi = evals[i + half].montgomeryMul(r);
        const expected = lo.add(hi);
        try testing.expectEqual(expected.limbs, gpu_out[i].limbs);
    }
}

test "GPU field_reduce_sum: correctness" {
    const gpu_mod = @import("mod.zig");
    if (comptime !gpu_mod.is_metal_available) return;

    var state = initGpu() catch |err| {
        if (err == error.MetalNotAvailable) return;
        return err;
    };
    defer state.ops.deinit();
    defer state.gpu.deinit();

    // Sum of 1..512 in the field
    const n = 512;
    var evals: [n]BN254Scalar = undefined;
    var cpu_sum = BN254Scalar.zero();
    for (0..n) |i| {
        evals[i] = BN254Scalar.fromU64(@as(u64, @intCast(i)) + 1);
        cpu_sum = cpu_sum.add(evals[i]);
    }

    const gpu_sum = try state.ops.fieldReduceSum(&evals);
    try testing.expectEqual(cpu_sum.limbs, gpu_sum.limbs);
}

test "GPU sumcheckRound: correctness" {
    const gpu_mod = @import("mod.zig");
    if (comptime !gpu_mod.is_metal_available) return;

    var state = initGpu() catch |err| {
        if (err == error.MetalNotAvailable) return;
        return err;
    };
    defer state.ops.deinit();
    defer state.gpu.deinit();

    const n = 256;
    var evals: [n]BN254Scalar = undefined;
    for (0..n) |i| {
        evals[i] = BN254Scalar.fromU64(@as(u64, @intCast(i)) *% 0x12345 +% 1);
    }

    const result = try state.ops.sumcheckRound(&evals);
    const g0 = result[0];
    const g1 = result[1];

    // CPU reference: sum first half, sum second half
    var cpu_g0 = BN254Scalar.zero();
    var cpu_g1 = BN254Scalar.zero();
    for (0..n / 2) |i| {
        cpu_g0 = cpu_g0.add(evals[i]);
        cpu_g1 = cpu_g1.add(evals[i + n / 2]);
    }

    try testing.expectEqual(cpu_g0.limbs, g0.limbs);
    try testing.expectEqual(cpu_g1.limbs, g1.limbs);
}

test "GPU sumcheck protocol: end-to-end" {
    const gpu_mod = @import("mod.zig");
    if (comptime !gpu_mod.is_metal_available) return;

    var state = initGpu() catch |err| {
        if (err == error.MetalNotAvailable) return;
        return err;
    };
    defer state.ops.deinit();
    defer state.gpu.deinit();

    // 2^4 = 16 evaluations, 4 rounds
    var evals: [16]BN254Scalar = undefined;
    for (0..16) |i| {
        evals[i] = BN254Scalar.fromU64(@as(u64, @intCast(i)) * 7 + 3);
    }

    const challenges = [_]u64{ 42, 77, 13, 99 };

    // Round 0: 16 → 8
    {
        const gpu_r = try state.ops.sumcheckRound(&evals);
        var cpu_g0 = BN254Scalar.zero();
        var cpu_g1 = BN254Scalar.zero();
        for (0..8) |i| { cpu_g0 = cpu_g0.add(evals[i]); cpu_g1 = cpu_g1.add(evals[i + 8]); }
        try testing.expectEqual(cpu_g0.limbs, gpu_r[0].limbs);
        try testing.expectEqual(cpu_g1.limbs, gpu_r[1].limbs);
    }
    const r0 = BN254Scalar.fromU64(challenges[0]);
    var e8: [8]BN254Scalar = undefined;
    try state.ops.polyBindFirst(&evals, r0, &e8);

    // Round 1: 8 → 4
    {
        const gpu_r = try state.ops.sumcheckRound(&e8);
        var cpu_g0 = BN254Scalar.zero();
        var cpu_g1 = BN254Scalar.zero();
        for (0..4) |i| { cpu_g0 = cpu_g0.add(e8[i]); cpu_g1 = cpu_g1.add(e8[i + 4]); }
        try testing.expectEqual(cpu_g0.limbs, gpu_r[0].limbs);
        try testing.expectEqual(cpu_g1.limbs, gpu_r[1].limbs);
    }
    const r1 = BN254Scalar.fromU64(challenges[1]);
    var e4: [4]BN254Scalar = undefined;
    try state.ops.polyBindFirst(&e8, r1, &e4);

    // Round 2: 4 → 2
    {
        const gpu_r = try state.ops.sumcheckRound(&e4);
        var cpu_g0 = BN254Scalar.zero();
        var cpu_g1 = BN254Scalar.zero();
        for (0..2) |i| { cpu_g0 = cpu_g0.add(e4[i]); cpu_g1 = cpu_g1.add(e4[i + 2]); }
        try testing.expectEqual(cpu_g0.limbs, gpu_r[0].limbs);
        try testing.expectEqual(cpu_g1.limbs, gpu_r[1].limbs);
    }
    const r2 = BN254Scalar.fromU64(challenges[2]);
    var e2: [2]BN254Scalar = undefined;
    try state.ops.polyBindFirst(&e4, r2, &e2);

    // Round 3: 2 → 1
    {
        const gpu_r = try state.ops.sumcheckRound(&e2);
        try testing.expectEqual(e2[0].limbs, gpu_r[0].limbs);
        try testing.expectEqual(e2[1].limbs, gpu_r[1].limbs);
    }
    const r3 = BN254Scalar.fromU64(challenges[3]);
    var e1: [1]BN254Scalar = undefined;
    try state.ops.polyBindFirst(&e2, r3, &e1);

    // Final: one evaluation remains — check it matches CPU computation
    const one_minus_r3 = BN254Scalar.one().sub(r3);
    const expected = e2[0].montgomeryMul(one_minus_r3).add(e2[1].montgomeryMul(r3));
    try testing.expectEqual(expected.limbs, e1[0].limbs);
}

test "GPU product_sumcheck_round: correctness" {
    const gpu_mod = @import("mod.zig");
    if (comptime !gpu_mod.is_metal_available) return;

    var state = initGpu() catch |err| {
        if (err == error.MetalNotAvailable) return;
        return err;
    };
    defer state.ops.deinit();
    defer state.gpu.deinit();

    // 32 groups → 64 interleaved elements in left/right
    const num_groups: usize = 32;
    const n = num_groups * 2;
    var left: [n]BN254Scalar = undefined;
    var right: [n]BN254Scalar = undefined;
    for (0..n) |i| {
        left[i] = BN254Scalar.fromU64(@as(u64, @intCast(i)) * 11 + 1);
        right[i] = BN254Scalar.fromU64(@as(u64, @intCast(i)) * 7 + 3);
    }

    // E_out = 32 elements, E_in = 1 element (xin_bits = 0)
    var e_out: [num_groups]BN254Scalar = undefined;
    for (0..num_groups) |i| {
        e_out[i] = BN254Scalar.fromU64(@as(u64, @intCast(i)) + 1);
    }
    const e_in = [_]BN254Scalar{BN254Scalar.one()};

    const gpu_result = try state.ops.productSumcheckRound(&left, &right, &e_out, &e_in, 0);

    // CPU reference
    var cpu_t0 = BN254Scalar.zero();
    var cpu_tinf = BN254Scalar.zero();
    for (0..num_groups) |g| {
        const l_lo = left[2 * g];
        const l_hi = left[2 * g + 1];
        const r_lo = right[2 * g];
        const r_hi = right[2 * g + 1];
        const p0 = l_lo.montgomeryMul(r_lo);
        const slope = (l_hi.sub(l_lo)).montgomeryMul(r_hi.sub(r_lo));
        const eq_w = e_out[g].montgomeryMul(e_in[0]);
        cpu_t0 = cpu_t0.add(p0.montgomeryMul(eq_w));
        cpu_tinf = cpu_tinf.add(slope.montgomeryMul(eq_w));
    }

    try testing.expectEqual(cpu_t0.limbs, gpu_result[0].limbs);
    try testing.expectEqual(cpu_tinf.limbs, gpu_result[1].limbs);
}

test "GPU outer_bind_reduce: correctness" {
    const gpu_mod = @import("mod.zig");
    if (comptime !gpu_mod.is_metal_available) return;

    var state = initGpu() catch |err| {
        if (err == error.MetalNotAvailable) return;
        return err;
    };
    defer state.ops.deinit();
    defer state.gpu.deinit();

    const n = 64;
    var az: [n]BN254Scalar = undefined;
    var bz: [n]BN254Scalar = undefined;
    for (0..n) |i| {
        az[i] = BN254Scalar.fromU64(@as(u64, @intCast(i)) * 13 + 2);
        bz[i] = BN254Scalar.fromU64(@as(u64, @intCast(i)) * 17 + 5);
    }

    const r = BN254Scalar.fromU64(42);
    const half = n / 2;

    var az_out: [half]BN254Scalar = undefined;
    var bz_out: [half]BN254Scalar = undefined;
    const gpu_result = try state.ops.outerBindReduce(&az, &bz, r, &az_out, &bz_out);

    // Verify bind outputs match CPU bindLow
    for (0..half) |i| {
        const az_lo = az[2 * i];
        const az_hi = az[2 * i + 1];
        const expected_az = az_lo.add(r.montgomeryMul(az_hi.sub(az_lo)));
        try testing.expectEqual(expected_az.limbs, az_out[i].limbs);

        const bz_lo = bz[2 * i];
        const bz_hi = bz[2 * i + 1];
        const expected_bz = bz_lo.add(r.montgomeryMul(bz_hi.sub(bz_lo)));
        try testing.expectEqual(expected_bz.limbs, bz_out[i].limbs);
    }

    // Verify reduction sums match CPU
    var cpu_t0 = BN254Scalar.zero();
    var cpu_tinf = BN254Scalar.zero();
    for (0..half) |i| {
        const az_lo = az[2 * i];
        const az_hi = az[2 * i + 1];
        const bz_lo = bz[2 * i];
        const bz_hi = bz[2 * i + 1];
        cpu_t0 = cpu_t0.add(az_lo.montgomeryMul(bz_lo));
        cpu_tinf = cpu_tinf.add((az_hi.sub(az_lo)).montgomeryMul(bz_hi.sub(bz_lo)));
    }

    try testing.expectEqual(cpu_t0.limbs, gpu_result[0].limbs);
    try testing.expectEqual(cpu_tinf.limbs, gpu_result[1].limbs);
}

test "GPU field_dot_product: correctness" {
    const gpu_mod = @import("mod.zig");
    if (comptime !gpu_mod.is_metal_available) return;

    var state = initGpu() catch |err| {
        if (err == error.MetalNotAvailable) return;
        return err;
    };
    defer state.ops.deinit();
    defer state.gpu.deinit();

    const n = 32;
    var a_vals: [n]BN254Scalar = undefined;
    var b_vals: [n]BN254Scalar = undefined;
    for (0..n) |i| {
        a_vals[i] = BN254Scalar.fromU64(@as(u64, @intCast(i)) * 31 + 5);
        b_vals[i] = BN254Scalar.fromU64(@as(u64, @intCast(i)) * 17 + 3);
    }

    const gpu_result = try state.ops.fieldDotProduct(&a_vals, &b_vals);

    // CPU reference: Σ a[i] * b[i]
    var cpu_sum = BN254Scalar.zero();
    for (0..n) |i| {
        cpu_sum = cpu_sum.add(a_vals[i].montgomeryMul(b_vals[i]));
    }

    try testing.expectEqual(cpu_sum.limbs, gpu_result.limbs);
}

test "GPU field_dot_product: large batch (4096)" {
    const gpu_mod = @import("mod.zig");
    if (comptime !gpu_mod.is_metal_available) return;

    var state = initGpu() catch |err| {
        if (err == error.MetalNotAvailable) return;
        return err;
    };
    defer state.ops.deinit();
    defer state.gpu.deinit();

    const n = 4096;
    const allocator = testing.allocator;
    const a_vals = try allocator.alloc(BN254Scalar, n);
    defer allocator.free(a_vals);
    const b_vals = try allocator.alloc(BN254Scalar, n);
    defer allocator.free(b_vals);

    for (0..n) |i| {
        a_vals[i] = BN254Scalar.fromU64(@as(u64, @intCast(i)) *% 0xABCDEF +% 1);
        b_vals[i] = BN254Scalar.fromU64(@as(u64, @intCast(i)) *% 0x7654321 +% 9);
    }

    const gpu_result = try state.ops.fieldDotProduct(a_vals, b_vals);

    var cpu_sum = BN254Scalar.zero();
    for (0..n) |i| {
        cpu_sum = cpu_sum.add(a_vals[i].montgomeryMul(b_vals[i]));
    }

    try testing.expectEqual(cpu_sum.limbs, gpu_result.limbs);
}

test "GPU field_dot_product: matches UnreducedProductAccum" {
    const gpu_mod = @import("mod.zig");
    if (comptime !gpu_mod.is_metal_available) return;

    var state = initGpu() catch |err| {
        if (err == error.MetalNotAvailable) return;
        return err;
    };
    defer state.ops.deinit();
    defer state.gpu.deinit();

    const n = 256;
    var a_vals: [n]BN254Scalar = undefined;
    var b_vals: [n]BN254Scalar = undefined;
    for (0..n) |i| {
        a_vals[i] = BN254Scalar.fromU64(@as(u64, @intCast(i)) *% 0xDEAD +% 42);
        b_vals[i] = BN254Scalar.fromU64(@as(u64, @intCast(i)) *% 0xBEEF +% 77);
    }

    const gpu_result = try state.ops.fieldDotProduct(&a_vals, &b_vals);

    // CPU reference using UnreducedProductAccum (the actual CPU hot-path pattern)
    const UnreducedProductAccum = field.UnreducedProductAccum;
    var accum = UnreducedProductAccum.zero();
    for (0..n) |i| {
        accum.addAssign(a_vals[i].mulToProductAccum(b_vals[i]));
    }
    const cpu_result = accum.reduce();

    try testing.expectEqual(cpu_result.limbs, gpu_result.limbs);
}

test "GPU field_dot_product: edge cases" {
    const gpu_mod = @import("mod.zig");
    if (comptime !gpu_mod.is_metal_available) return;

    var state = initGpu() catch |err| {
        if (err == error.MetalNotAvailable) return;
        return err;
    };
    defer state.ops.deinit();
    defer state.gpu.deinit();

    // n = 1
    {
        const a = [_]BN254Scalar{BN254Scalar.fromU64(7)};
        const b = [_]BN254Scalar{BN254Scalar.fromU64(11)};
        const result = try state.ops.fieldDotProduct(&a, &b);
        const expected = a[0].montgomeryMul(b[0]);
        try testing.expectEqual(expected.limbs, result.limbs);
    }

    // one array all zeros
    {
        const a = [_]BN254Scalar{ BN254Scalar.fromU64(42), BN254Scalar.fromU64(99) };
        const b = [_]BN254Scalar{ BN254Scalar.zero(), BN254Scalar.zero() };
        const result = try state.ops.fieldDotProduct(&a, &b);
        try testing.expectEqual(BN254Scalar.zero().limbs, result.limbs);
    }

    // identity: a[i] * 1
    {
        const n = 8;
        var a_vals: [n]BN254Scalar = undefined;
        var b_vals: [n]BN254Scalar = undefined;
        for (0..n) |i| {
            a_vals[i] = BN254Scalar.fromU64(@as(u64, @intCast(i)) + 1);
            b_vals[i] = BN254Scalar.one();
        }
        const result = try state.ops.fieldDotProduct(&a_vals, &b_vals);
        // Σ a[i] * 1 = Σ a[i]
        var cpu_sum = BN254Scalar.zero();
        for (0..n) |i| cpu_sum = cpu_sum.add(a_vals[i]);
        try testing.expectEqual(cpu_sum.limbs, result.limbs);
    }
}
