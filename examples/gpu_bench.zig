const std = @import("std");
const zolt = @import("zolt");
const BN254Scalar = zolt.BN254Scalar;
const gpu_mod = zolt.gpu;

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    var gpu_accel = gpu_mod.GpuAccelerator.init(allocator) catch {
        std.debug.print("No Metal GPU available\n", .{});
        return;
    };
    defer gpu_accel.deinit();
    var gpu_ops = try gpu_mod.GpuPolyOps.init(&gpu_accel);
    defer gpu_ops.deinit();

    std.debug.print("\n=== GPU vs CPU Benchmark (persistent buffers) ===\n\n", .{});

    const sizes = [_]usize{ 1024, 4096, 16384, 65536, 262144 };

    // ── bindLow: in-place GPU vs CPU ────────────────────────────────────────
    for (sizes) |n| {
        const evals = try allocator.alloc(BN254Scalar, n);
        defer allocator.free(evals);
        for (0..n) |i| evals[i] = BN254Scalar.fromU64(@as(u64, @intCast(i)) *% 0xABCDEF +% 7);
        const r = BN254Scalar.fromU64(42);
        const iters: usize = if (n <= 16384) 200 else 50;

        // GPU in-place (persistent buffer)
        var poly = try gpu_mod.GpuPolynomial.initFromCpu(gpu_accel.device, evals);
        defer poly.deinit();
        // Warmup
        try gpu_ops.bindLowInPlace(&poly, r);
        // Re-init for benchmark
        poly.deinit();
        poly = try gpu_mod.GpuPolynomial.initFromCpu(gpu_accel.device, evals);

        var gpu_timer = try std.time.Timer.start();
        for (0..iters) |_| {
            // Re-init each iteration (simulates fresh round data)
            @memcpy(poly.bufs[poly.active].slice()[0 .. n * 8], @as([*]const u32, @ptrCast(@alignCast(evals.ptr)))[0 .. n * 8]);
            poly.len = n;
            try gpu_ops.bindLowInPlace(&poly, r);
        }
        const gpu_ns = gpu_timer.read();
        const gpu_us = @as(f64, @floatFromInt(gpu_ns)) / @as(f64, @floatFromInt(iters)) / 1000.0;

        // CPU
        const out = try allocator.alloc(BN254Scalar, n / 2);
        defer allocator.free(out);
        var cpu_timer = try std.time.Timer.start();
        for (0..iters) |_| {
            for (0..n / 2) |i| {
                const lo = evals[2 * i];
                const hi = evals[2 * i + 1];
                out[i] = lo.add(r.montgomeryMul(hi.sub(lo)));
            }
        }
        const cpu_ns = cpu_timer.read();
        const cpu_us = @as(f64, @floatFromInt(cpu_ns)) / @as(f64, @floatFromInt(iters)) / 1000.0;

        const speedup = cpu_us / gpu_us;
        std.debug.print("bindLow n={d:>7}: GPU {d:>8.0}us  CPU {d:>8.0}us  {d:.1}x\n", .{ n, gpu_us, cpu_us, speedup });
    }

    std.debug.print("\n", .{});

    // ── sumcheckRound: in-place GPU vs CPU ──────────────────────────────────
    for (sizes) |n| {
        const evals = try allocator.alloc(BN254Scalar, n);
        defer allocator.free(evals);
        for (0..n) |i| evals[i] = BN254Scalar.fromU64(@as(u64, @intCast(i)) +% 1);
        const iters: usize = if (n <= 16384) 200 else 50;

        var poly = try gpu_mod.GpuPolynomial.initFromCpu(gpu_accel.device, evals);
        defer poly.deinit();

        var gpu_timer = try std.time.Timer.start();
        for (0..iters) |_| {
            _ = try gpu_ops.sumcheckRoundInPlace(&poly);
        }
        const gpu_ns = gpu_timer.read();
        const gpu_us = @as(f64, @floatFromInt(gpu_ns)) / @as(f64, @floatFromInt(iters)) / 1000.0;

        var cpu_timer = try std.time.Timer.start();
        for (0..iters) |_| {
            var g0 = BN254Scalar.zero();
            var g1 = BN254Scalar.zero();
            for (0..n / 2) |i| { g0 = g0.add(evals[i]); g1 = g1.add(evals[i + n / 2]); }
            std.mem.doNotOptimizeAway(&g0);
            std.mem.doNotOptimizeAway(&g1);
        }
        const cpu_ns = cpu_timer.read();
        const cpu_us = @as(f64, @floatFromInt(cpu_ns)) / @as(f64, @floatFromInt(iters)) / 1000.0;

        const speedup = cpu_us / gpu_us;
        std.debug.print("reduce  n={d:>7}: GPU {d:>8.0}us  CPU {d:>8.0}us  {d:.1}x\n", .{ n, gpu_us, cpu_us, speedup });
    }

    std.debug.print("\n", .{});

    // ── Full sumcheck loop: hybrid GPU/CPU vs pure CPU ────────────────────
    // GPU for large rounds (> threshold), CPU for small rounds.
    for ([_]usize{ 16384, 65536, 262144 }) |n| {
        const evals = try allocator.alloc(BN254Scalar, n);
        defer allocator.free(evals);
        for (0..n) |i| evals[i] = BN254Scalar.fromU64(@as(u64, @intCast(i)) *% 0x12345 +% 1);

        const num_vars = std.math.log2_int(usize, n);
        const iters: usize = 10;

        // Hybrid GPU/CPU: GPU for large rounds, CPU for small
        const GPU_THRESHOLD: usize = 16384;
        var gpu_timer = try std.time.Timer.start();
        for (0..iters) |_| {
            var poly = try gpu_mod.GpuPolynomial.initFromCpu(gpu_accel.device, evals);
            defer poly.deinit();

            // First round: reduce on GPU
            _ = try gpu_ops.sumcheckRoundInPlace(&poly);
            const challenge = BN254Scalar.fromU64(42);

            // GPU rounds (fused bind+reduce)
            var round: usize = 1;
            while (round < num_vars and poly.len / 2 >= GPU_THRESHOLD) : (round += 1) {
                _ = try gpu_ops.sumcheckStepInPlace(&poly, challenge);
            }

            // Transition: read back to CPU
            const cpu_buf = try allocator.alloc(BN254Scalar, poly.len);
            defer allocator.free(cpu_buf);
            if (round < num_vars) {
                // Need to do the bind for the current pending challenge first
                try gpu_ops.bindFirstInPlace(&poly, challenge);
                poly.readAll(cpu_buf[0..poly.len]);
            }
            var cpu_len = poly.len;

            // CPU rounds
            while (round < num_vars) : (round += 1) {
                const hl = cpu_len / 2;
                var g0 = BN254Scalar.zero();
                var g1 = BN254Scalar.zero();
                for (0..hl) |i| { g0 = g0.add(cpu_buf[i]); g1 = g1.add(cpu_buf[i + hl]); }
                std.mem.doNotOptimizeAway(&g0);
                const c = BN254Scalar.fromU64(42);
                const omr = BN254Scalar.one().sub(c);
                for (0..hl) |i| {
                    cpu_buf[i] = cpu_buf[i].montgomeryMul(omr).add(cpu_buf[i + hl].montgomeryMul(c));
                }
                cpu_len = hl;
            }
        }
        const gpu_ns = gpu_timer.read();
        const gpu_us = @as(f64, @floatFromInt(gpu_ns)) / @as(f64, @floatFromInt(iters)) / 1000.0;

        // CPU: full sumcheck loop
        const cpu_evals = try allocator.alloc(BN254Scalar, n);
        defer allocator.free(cpu_evals);
        var cpu_timer = try std.time.Timer.start();
        for (0..iters) |_| {
            @memcpy(cpu_evals, evals);
            var len = n;
            for (0..num_vars) |_| {
                const half_len = len / 2;
                var g0 = BN254Scalar.zero();
                var g1 = BN254Scalar.zero();
                for (0..half_len) |i| { g0 = g0.add(cpu_evals[i]); g1 = g1.add(cpu_evals[i + half_len]); }
                std.mem.doNotOptimizeAway(&g0);
                // bindFirst in-place
                const challenge = BN254Scalar.fromU64(42);
                const omr = BN254Scalar.one().sub(challenge);
                for (0..half_len) |i| {
                    cpu_evals[i] = cpu_evals[i].montgomeryMul(omr).add(cpu_evals[i + half_len].montgomeryMul(challenge));
                }
                len = half_len;
            }
        }
        const cpu_ns = cpu_timer.read();
        const cpu_us = @as(f64, @floatFromInt(cpu_ns)) / @as(f64, @floatFromInt(iters)) / 1000.0;

        const speedup = cpu_us / gpu_us;
        std.debug.print("sumcheck n={d:>7} ({d} rounds): GPU {d:>8.0}us  CPU {d:>8.0}us  {d:.1}x\n", .{ n, num_vars, gpu_us, cpu_us, speedup });
    }

    std.debug.print("\n", .{});

    // ── fieldDotProduct: GPU vs CPU UnreducedProductAccum ────────────────
    for (sizes) |n| {
        const a_vals = try allocator.alloc(BN254Scalar, n);
        defer allocator.free(a_vals);
        const b_vals = try allocator.alloc(BN254Scalar, n);
        defer allocator.free(b_vals);
        for (0..n) |i| {
            a_vals[i] = BN254Scalar.fromU64(@as(u64, @intCast(i)) *% 0xABCDEF +% 7);
            b_vals[i] = BN254Scalar.fromU64(@as(u64, @intCast(i)) *% 0x7654321 +% 3);
        }
        const iters_dp: usize = if (n <= 16384) 200 else 50;

        // GPU dot product
        _ = try gpu_ops.fieldDotProduct(a_vals, b_vals); // warmup
        var gpu_timer = try std.time.Timer.start();
        for (0..iters_dp) |_| {
            _ = try gpu_ops.fieldDotProduct(a_vals, b_vals);
        }
        const gpu_ns = gpu_timer.read();
        const gpu_us = @as(f64, @floatFromInt(gpu_ns)) / @as(f64, @floatFromInt(iters_dp)) / 1000.0;

        // CPU UnreducedProductAccum (the real hot-path pattern)
        const field_mod = @import("zolt").field;
        const UnreducedProductAccum = field_mod.UnreducedProductAccum;
        var cpu_timer = try std.time.Timer.start();
        for (0..iters_dp) |_| {
            var accum = UnreducedProductAccum.zero();
            for (0..n) |i| {
                accum.addAssign(a_vals[i].mulToProductAccum(b_vals[i]));
            }
            const result = accum.reduce();
            std.mem.doNotOptimizeAway(&result);
        }
        const cpu_ns = cpu_timer.read();
        const cpu_us = @as(f64, @floatFromInt(cpu_ns)) / @as(f64, @floatFromInt(iters_dp)) / 1000.0;

        // GPU dot product on persistent buffers (zero-copy, no alloc)
        const GpuPoly = gpu_mod.GpuPolynomial;
        var poly_a = try GpuPoly.initFromCpu(gpu_accel.device, a_vals);
        defer poly_a.deinit();
        var poly_b = try GpuPoly.initFromCpu(gpu_accel.device, b_vals);
        defer poly_b.deinit();

        _ = try gpu_ops.fieldDotProductGpu(poly_a.metalBuffer(), poly_b.metalBuffer(), @intCast(n)); // warmup
        var gpu_zc_timer = try std.time.Timer.start();
        for (0..iters_dp) |_| {
            _ = try gpu_ops.fieldDotProductGpu(poly_a.metalBuffer(), poly_b.metalBuffer(), @intCast(n));
        }
        const gpu_zc_ns = gpu_zc_timer.read();
        const gpu_zc_us = @as(f64, @floatFromInt(gpu_zc_ns)) / @as(f64, @floatFromInt(iters_dp)) / 1000.0;

        const speedup_dp = cpu_us / gpu_us;
        const speedup_zc = cpu_us / gpu_zc_us;
        std.debug.print("dotprod n={d:>7}: GPU(copy) {d:>7.0}us  GPU(zc) {d:>7.0}us  CPU {d:>7.0}us  copy {d:.1}x  zc {d:.1}x\n", .{ n, gpu_us, gpu_zc_us, cpu_us, speedup_dp, speedup_zc });
    }
}
