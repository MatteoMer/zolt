//! GPU-accelerated Multi-Scalar Multiplication (MSM) for BN254 G1.
//!
//! Dispatches the fused bucket-accumulate + running-sum-reduce kernel to Metal.
//! Each GPU thread processes one (row, window) pair. The CPU handles wNAF
//! digit decomposition, point format conversion, and final window combination.

const std = @import("std");
const device_mod = @import("device.zig");
const buffer_mod = @import("buffer.zig");
const objc = @import("objc.zig");
const field = @import("../field/mod.zig");
const msm_mod = @import("../msm/mod.zig");

const GpuAccelerator = device_mod.GpuAccelerator;
const KernelParam = GpuAccelerator.KernelParam;
const GpuBuffer = buffer_mod.GpuBuffer;
const BN254Scalar = field.BN254Scalar;
const BN254BaseField = field.BN254BaseField;
const AffinePoint = msm_mod.AffinePoint;
const ProjectivePoint = msm_mod.ProjectivePoint;
const BucketPoint = msm_mod.BucketPoint;
const id = objc.id;

/// Scalar bit width for BN254
const SCALAR_BITS: usize = 256;

/// Max wNAF digits: ceil(256/3) + 1 = 87
const MAX_DIGITS: usize = 87;

/// MsmParams layout — must match the MSL struct exactly.
const MsmParams = extern struct {
    num_points: u32,
    num_windows: u32,
    num_buckets: u32,
    num_rows: u32,
};

pub const GpuMsmOps = struct {
    gpu: *GpuAccelerator,
    accumulate_reduce_pipeline: id,

    pub const Error = GpuAccelerator.Error || GpuBuffer(u32).Error || std.mem.Allocator.Error;

    pub fn init(gpu: *GpuAccelerator) GpuAccelerator.Error!GpuMsmOps {
        const pipeline = try gpu.createPipeline("msm_accumulate_reduce");
        return .{
            .gpu = gpu,
            .accumulate_reduce_pipeline = pipeline,
        };
    }

    pub fn deinit(self: *GpuMsmOps) void {
        objc.msgSendVoid(self.accumulate_reduce_pipeline, "release");
    }

    /// Compute one MSM per row: out[row] = sum_j scalars[row*num_cols + j] * bases[j].
    ///
    /// `bases`:    shared G1 affine points (num_cols)
    /// `scalars`:  flat array of all scalars (num_rows * num_cols)
    /// `num_cols`: points per row
    /// `allocator`: for temporary CPU-side allocations
    /// `out`:      one ProjectivePoint per row
    pub fn computeRowCommitments(
        self: *GpuMsmOps,
        bases: []const AffinePoint(BN254BaseField),
        scalars: []const BN254Scalar,
        num_cols: usize,
        allocator: std.mem.Allocator,
        out: []ProjectivePoint(BN254BaseField),
    ) Error!void {
        std.debug.assert(num_cols > 0);
        std.debug.assert(scalars.len % num_cols == 0);
        const num_rows = scalars.len / num_cols;
        std.debug.assert(out.len == num_rows);
        if (num_rows == 0) return;

        // ── Window parameters ──────────────────────────────────────────
        const c = optimalWindowSize(num_cols);
        const num_scalar_windows = (SCALAR_BITS + c - 1) / c;
        const num_windows = num_scalar_windows + 1; // +1 for wNAF carry
        const num_buckets = (@as(usize, 1) << @as(u6, @intCast(c))) / 2;

        // ── CPU: Convert affine points to GPU format ───────────────────
        // Each point = 2 field elements (x, y), each element = 8 × u32.
        const points_u32_count = num_cols * 2 * 8;
        var buf_points = try GpuBuffer(u32).init(self.gpu.device, points_u32_count);
        defer buf_points.deinit();
        writeAffinePoints(buf_points.slice(), bases);

        // ── CPU: wNAF decompose all scalars ────────────────────────────
        // Layout: [num_rows][num_windows][num_cols] as i8
        const digits_count = num_rows * num_windows * num_cols;
        var buf_digits = try GpuBuffer(i8).init(self.gpu.device, digits_count);
        defer buf_digits.deinit();

        const digits_slice = buf_digits.slice();
        for (0..num_rows) |row| {
            for (0..num_cols) |col| {
                const scalar = scalars[row * num_cols + col];
                const raw_limbs = scalar.fromMontgomery().limbs;
                const ds = makeDigits(raw_limbs, c, num_scalar_windows);
                for (0..num_windows) |wi| {
                    digits_slice[(row * num_windows + wi) * num_cols + col] = ds[wi];
                }
            }
        }

        // ── GPU: Allocate scratch + result buffers ─────────────────────
        const num_threads = num_rows * num_windows;
        // Scratch: num_threads * num_buckets * 4 FpBase * 8 u32/FpBase
        const scratch_count = num_threads * num_buckets * 4 * 8;
        var buf_scratch = try GpuBuffer(u32).init(self.gpu.device, scratch_count);
        defer buf_scratch.deinit();
        // Results: num_threads * 4 FpBase * 8 u32/FpBase
        const result_count = num_threads * 4 * 8;
        var buf_results = try GpuBuffer(u32).init(self.gpu.device, result_count);
        defer buf_results.deinit();

        // ── GPU: Dispatch ──────────────────────────────────────────────
        const params = MsmParams{
            .num_points = @intCast(num_cols),
            .num_windows = @intCast(num_windows),
            .num_buckets = @intCast(num_buckets),
            .num_rows = @intCast(num_rows),
        };

        try self.gpu.dispatchKernel(
            self.accumulate_reduce_pipeline,
            &.{
                KernelParam{ .buffer = buf_points.metal_buffer },
                KernelParam{ .buffer = buf_digits.metal_buffer },
                KernelParam{ .buffer = buf_scratch.metal_buffer },
                KernelParam{ .buffer = buf_results.metal_buffer },
                KernelParam{ .bytes = .{ .ptr = &params, .len = @sizeOf(MsmParams) } },
            },
            num_threads,
            0, // auto threadgroup size
        );

        // ── CPU: Read back results and window combination ─────────────
        const results_u32 = buf_results.slice();

        // Temporary storage for per-window XYZZ results for one row.
        // We need at most MAX_DIGITS windows.
        var window_xyzz: [MAX_DIGITS]BucketPoint(BN254BaseField) = undefined;

        // Also need storage for batch affine conversion
        var window_affine: [MAX_DIGITS]AffinePoint(BN254BaseField) = undefined;
        _ = allocator; // allocator reserved for future use if needed

        for (0..num_rows) |row| {
            // Read XYZZ results for all windows in this row
            for (0..num_windows) |wi| {
                const thread_id = row * num_windows + wi;
                const base_offset = thread_id * 4 * 8;
                const x = readBaseFieldElement(results_u32[base_offset..][0..8]);
                const y = readBaseFieldElement(results_u32[base_offset + 8 ..][0..8]);
                const zz = readBaseFieldElement(results_u32[base_offset + 16 ..][0..8]);
                const zzz = readBaseFieldElement(results_u32[base_offset + 24 ..][0..8]);

                // Reconstruct BucketPoint — empty when zz == 0
                const is_empty = zz.isZero();
                window_xyzz[wi] = .{
                    .x = x,
                    .y = y,
                    .zz = zz,
                    .zzz = zzz,
                    .empty = is_empty,
                };
            }

            // Convert window sums to affine (batch inversion)
            for (0..num_windows) |wi| {
                window_affine[wi] = window_xyzz[wi].toAffine();
            }

            // Window combination: MSB → LSB, c doublings between windows
            var result = ProjectivePoint(BN254BaseField).identity();
            var window_idx: usize = num_windows;
            while (window_idx > 0) {
                window_idx -= 1;

                if (!result.isIdentity()) {
                    var k: usize = 0;
                    while (k < c) : (k += 1) {
                        result = result.double();
                    }
                }

                if (!window_affine[window_idx].isIdentity()) {
                    if (result.isIdentity()) {
                        result = ProjectivePoint(BN254BaseField).fromAffine(window_affine[window_idx]);
                    } else {
                        result = result.addAffine(window_affine[window_idx]);
                    }
                }
            }

            out[row] = result;
        }
    }

    /// Compute a single MSM: result = sum_j scalars[j] * bases[j].
    /// Convenience wrapper around computeRowCommitments with num_rows=1.
    pub fn computeSingleMsm(
        self: *GpuMsmOps,
        bases: []const AffinePoint(BN254BaseField),
        scalars: []const BN254Scalar,
        allocator: std.mem.Allocator,
    ) Error!AffinePoint(BN254BaseField) {
        std.debug.assert(bases.len == scalars.len);
        if (bases.len == 0) return AffinePoint(BN254BaseField).identity();

        var result: [1]ProjectivePoint(BN254BaseField) = undefined;
        try self.computeRowCommitments(bases, scalars, bases.len, allocator, &result);
        return result[0].toAffine();
    }
};

// ── wNAF digit decomposition ────────────────────────────────────────────────
//
// Reimplemented here because MSM(F,G).makeDigits is private and generic.
// Identical algorithm: signed-digit decomposition with radix 2^c.

fn makeDigits(limbs: [4]u64, c: usize, num_scalar_windows: usize) [MAX_DIGITS]i8 {
    var digits: [MAX_DIGITS]i8 = undefined;
    const radix: u64 = @as(u64, 1) << @as(u6, @intCast(c));
    const window_mask: u64 = radix - 1;
    const half_radix: u64 = radix >> 1;
    var carry: u64 = 0;

    const num_digits = num_scalar_windows;
    for (0..num_digits) |i| {
        const bit_offset = i * c;
        const u64_idx = bit_offset / 64;
        const bit_idx = @as(u6, @intCast(bit_offset % 64));

        var bit_buf: u64 = if (u64_idx < 4) limbs[u64_idx] >> bit_idx else 0;
        const bit_idx_usize: usize = @as(usize, bit_idx);
        if (bit_idx_usize + c > 64 and u64_idx + 1 < 4) {
            bit_buf |= limbs[u64_idx + 1] << @as(u6, @intCast(64 - bit_idx_usize));
        }

        const coef = carry + (bit_buf & window_mask);
        carry = if (coef >= half_radix) 1 else 0;
        const digit: i32 = @as(i32, @intCast(coef)) - @as(i32, @intCast(carry)) * @as(i32, @intCast(radix));

        digits[i] = @as(i8, @intCast(digit));
    }
    // Final carry becomes an extra digit (0 or 1)
    if (num_digits < MAX_DIGITS) {
        digits[num_digits] = @intCast(carry);
    }
    // Zero remaining entries
    for (num_digits + 1..MAX_DIGITS) |i| {
        digits[i] = 0;
    }
    return digits;
}

/// Choose optimal window size based on input size.
/// Uses ln(n) + 2 heuristic matching upstream MSM(F,G).optimalWindowSize.
fn optimalWindowSize(n: usize) usize {
    if (n < 32) return 3;
    const log2_n = @as(usize, std.math.log2_int(usize, n));
    const c = (log2_n * 69 / 100) + 2;
    return @max(3, @min(c, 20));
}

// ── Format conversion: BN254BaseField ↔ GPU 8×u32 ──────────────────────────
//
// Same bit-splitting as field_ops.zig but for BN254BaseField (Fp).
// Montgomery form is preserved — this is pure bit splitting.
// Little-endian: gpu[2i] = lo32(cpu[i]), gpu[2i+1] = hi32(cpu[i])

fn writeBaseFieldElements(buf: []u32, elems: []const BN254BaseField) void {
    for (elems, 0..) |elem, idx| {
        const base = idx * 8;
        inline for (0..4) |i| {
            buf[base + i * 2] = @truncate(elem.limbs[i]);
            buf[base + i * 2 + 1] = @truncate(elem.limbs[i] >> 32);
        }
    }
}

fn readBaseFieldElement(buf: *const [8]u32) BN254BaseField {
    var result: BN254BaseField = undefined;
    inline for (0..4) |i| {
        result.limbs[i] = @as(u64, buf[i * 2]) | (@as(u64, buf[i * 2 + 1]) << 32);
    }
    return result;
}

/// Write affine points as contiguous FpBase pairs in GPU format.
/// Layout: [num_points * 2] FpBase values (x0, y0, x1, y1, ...)
fn writeAffinePoints(buf: []u32, points: []const AffinePoint(BN254BaseField)) void {
    for (points, 0..) |pt, idx| {
        const base = idx * 2 * 8; // 2 field elements, 8 u32 each
        // x
        inline for (0..4) |i| {
            buf[base + i * 2] = @truncate(pt.x.limbs[i]);
            buf[base + i * 2 + 1] = @truncate(pt.x.limbs[i] >> 32);
        }
        // y
        inline for (0..4) |i| {
            buf[base + 8 + i * 2] = @truncate(pt.y.limbs[i]);
            buf[base + 8 + i * 2 + 1] = @truncate(pt.y.limbs[i] >> 32);
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

const testing = std.testing;

test "GPU Fp base field mul: correctness" {
    const gpu_mod = @import("mod.zig");
    if (comptime !gpu_mod.is_metal_available) return;

    var gpu = device_mod.GpuAccelerator.init(testing.allocator) catch |err| {
        if (err == error.MetalNotAvailable) return;
        return err;
    };
    defer gpu.deinit();

    const pipeline = try gpu.createPipeline("fp_base_mul_test");
    defer objc.msgSendVoid(pipeline, "release");

    const Fp = BN254BaseField;

    // Test values: small numbers, edge cases
    const a_vals = [_]Fp{
        Fp.fromU64(2),
        Fp.fromU64(3),
        Fp.fromU64(0),
        Fp.fromU64(1),
        Fp.fromU64(0xFFFFFFFF),
        Fp.fromU64(0xDEADBEEF),
        Fp.one(),
        Fp.zero(),
    };
    const b_vals = [_]Fp{
        Fp.fromU64(3),
        Fp.fromU64(7),
        Fp.fromU64(5),
        Fp.fromU64(1),
        Fp.fromU64(0xFFFFFFFF),
        Fp.fromU64(42),
        Fp.fromU64(123456789),
        Fp.fromU64(999),
    };

    const n = a_vals.len;
    var buf_a = try GpuBuffer(u32).init(gpu.device, n * 8);
    defer buf_a.deinit();
    var buf_b = try GpuBuffer(u32).init(gpu.device, n * 8);
    defer buf_b.deinit();
    var buf_c = try GpuBuffer(u32).init(gpu.device, n * 8);
    defer buf_c.deinit();

    writeBaseFieldElements(buf_a.slice(), &a_vals);
    writeBaseFieldElements(buf_b.slice(), &b_vals);

    try gpu.dispatchSimple(
        pipeline,
        &.{ buf_a.metal_buffer, buf_b.metal_buffer, buf_c.metal_buffer },
        n,
    );

    for (0..n) |i| {
        const cpu_result = a_vals[i].mul(b_vals[i]);
        const gpu_result = readBaseFieldElement(buf_c.slice()[i * 8 ..][0..8]);
        try testing.expectEqual(cpu_result.limbs, gpu_result.limbs);
    }
}

test "GPU MSM: small correctness" {
    const gpu_mod = @import("mod.zig");
    if (comptime !gpu_mod.is_metal_available) return;

    var gpu = device_mod.GpuAccelerator.init(testing.allocator) catch |err| {
        if (err == error.MetalNotAvailable) return;
        return err;
    };
    defer gpu.deinit();

    var ops = GpuMsmOps.init(&gpu) catch |err| {
        if (err == error.FunctionNotFound) return; // kernel not compiled
        return err;
    };
    defer ops.deinit();

    const Fp = BN254BaseField;
    const G1Affine = AffinePoint(Fp);
    const CpuMSM = msm_mod.MSM(BN254Scalar, Fp);

    const num_cols: usize = 32;
    const num_rows: usize = 2;

    // Generate deterministic test points: repeated doublings of generator
    var bases: [num_cols]G1Affine = undefined;
    bases[0] = G1Affine.generator();
    for (1..num_cols) |i| {
        bases[i] = bases[i - 1].double();
    }

    // Generate deterministic scalars
    var scalars: [num_rows * num_cols]BN254Scalar = undefined;
    for (0..num_rows * num_cols) |i| {
        scalars[i] = BN254Scalar.fromU64(@as(u64, @intCast(i)) *% 0x12345 +% 1);
    }

    // GPU computation
    var gpu_results: [num_rows]ProjectivePoint(Fp) = undefined;
    try ops.computeRowCommitments(
        &bases,
        &scalars,
        num_cols,
        testing.allocator,
        &gpu_results,
    );

    // CPU reference computation
    for (0..num_rows) |row| {
        const row_scalars = scalars[row * num_cols ..][0..num_cols];
        const cpu_affine = CpuMSM.compute(&bases, row_scalars);
        const gpu_affine = gpu_results[row].toAffine();

        try testing.expect(cpu_affine.eql(gpu_affine));
    }
}

test "GPU MSM: large scalars correctness" {
    const gpu_mod_test = @import("mod.zig");
    if (comptime !gpu_mod_test.is_metal_available) return;

    var gpu = device_mod.GpuAccelerator.init(testing.allocator) catch |err| {
        if (err == error.MetalNotAvailable) return;
        return err;
    };
    defer gpu.deinit();

    var ops = GpuMsmOps.init(&gpu) catch |err| {
        if (err == error.FunctionNotFound) return;
        return err;
    };
    defer ops.deinit();

    const Fp = BN254BaseField;
    const G1Affine = AffinePoint(Fp);
    const CpuMSM = msm_mod.MSM(BN254Scalar, Fp);

    // Use 64 points with large (multi-limb) scalars — closer to IPA usage
    const num_cols: usize = 64;

    const bases = try testing.allocator.alloc(G1Affine, num_cols);
    defer testing.allocator.free(bases);
    bases[0] = G1Affine.generator();
    for (1..num_cols) |i| {
        bases[i] = bases[i - 1].double();
    }

    // Multi-limb scalars (not just fromU64)
    const scalars = try testing.allocator.alloc(BN254Scalar, num_cols);
    defer testing.allocator.free(scalars);
    for (0..num_cols) |i| {
        // Create scalars with multiple non-zero limbs
        var s = BN254Scalar.fromU64(@as(u64, @intCast(i)) *% 0xDEADBEEFCAFEBABE +% 1);
        s = s.mul(s); // square to get multi-limb values
        s = s.mul(BN254Scalar.fromU64(@as(u64, @intCast(i)) +% 0x123456789));
        scalars[i] = s;
    }

    var gpu_results: [1]ProjectivePoint(Fp) = undefined;
    try ops.computeRowCommitments(bases, scalars, num_cols, testing.allocator, &gpu_results);
    const gpu_affine = gpu_results[0].toAffine();

    const cpu_affine = CpuMSM.compute(bases, scalars);

    if (!cpu_affine.eql(gpu_affine)) {
        std.debug.print("GPU MSM large-scalar MISMATCH!\n", .{});
        std.debug.print("  CPU x: [{x:0>16}, {x:0>16}, {x:0>16}, {x:0>16}]\n", .{ cpu_affine.x.limbs[0], cpu_affine.x.limbs[1], cpu_affine.x.limbs[2], cpu_affine.x.limbs[3] });
        std.debug.print("  GPU x: [{x:0>16}, {x:0>16}, {x:0>16}, {x:0>16}]\n", .{ gpu_affine.x.limbs[0], gpu_affine.x.limbs[1], gpu_affine.x.limbs[2], gpu_affine.x.limbs[3] });
        return error.TestExpectedEqual;
    }
}

test "GPU MSM: 256-4096 points correctness" {
    const gpu_mod_test = @import("mod.zig");
    if (comptime !gpu_mod_test.is_metal_available) return;

    var gpu = device_mod.GpuAccelerator.init(testing.allocator) catch |err| {
        if (err == error.MetalNotAvailable) return;
        return err;
    };
    defer gpu.deinit();

    var ops = GpuMsmOps.init(&gpu) catch |err| {
        if (err == error.FunctionNotFound) return;
        return err;
    };
    defer ops.deinit();

    const Fp = BN254BaseField;
    const G1Affine = AffinePoint(Fp);
    const CpuMSM = msm_mod.MSM(BN254Scalar, Fp);

    for ([_]usize{ 256, 512, 1024, 4096 }) |num_cols| {
        // Generate test points via repeated doubling
        const bases = try testing.allocator.alloc(G1Affine, num_cols);
        defer testing.allocator.free(bases);
        bases[0] = G1Affine.generator();
        for (1..num_cols) |i| {
            bases[i] = bases[i - 1].double();
        }

        // Generate scalars
        const scalars = try testing.allocator.alloc(BN254Scalar, num_cols);
        defer testing.allocator.free(scalars);
        for (0..num_cols) |i| {
            scalars[i] = BN254Scalar.fromU64(@as(u64, @intCast(i)) *% 0xABCDEF +% 7);
        }

        // GPU
        var gpu_results: [1]ProjectivePoint(Fp) = undefined;
        try ops.computeRowCommitments(bases, scalars, num_cols, testing.allocator, &gpu_results);
        const gpu_affine = gpu_results[0].toAffine();

        // CPU
        const cpu_affine = CpuMSM.compute(bases, scalars);

        if (!cpu_affine.eql(gpu_affine)) {
            std.debug.print("GPU MSM {}-point MISMATCH!\n", .{num_cols});
            return error.TestExpectedEqual;
        }
    }
}
