//! Scaling micro-benchmarks: parallelFor, repeated dispatch, multi-array bind
//!
//! These complement the existing reduce/join benchmarks by testing the patterns
//! that dominate the actual prover hot paths:
//!
//!   1. parallelFor (in-place write) — polynomial binding pattern
//!   2. repeated dispatch — many short parallel regions (sumcheck rounds)
//!   3. multi-array bind — N independent arrays bound in parallel
//!
//! Build: zig build bench-scaling -Doptimize=ReleaseFast

const std = @import("std");
const zolt = @import("zolt");
const MonotonicTimer = zolt.utils.MonotonicTimer;
const bench_io: std.Io = std.Io.Threaded.global_single_threaded.io();
const F = zolt.field.BN254Scalar;
const ThreadPool = zolt.utils.ThreadPool;

const WARMUP = 5;
const ITERS = 50;
const RUNS = 5;

// ====================================================================
// Benchmark 1: parallelFor — in-place u64 write (light work per elem)
// ====================================================================

fn forLightPar(tp: *ThreadPool, data: []u64, n: usize) f64 {
    const Ctx = struct { d: []u64 };
    const ctx = Ctx{ .d = data[0..n] };
    const func = struct {
        fn f(c: Ctx, i: usize) void {
            c.d[i] = c.d[i] *% (c.d[i] +% 1);
        }
    }.f;

    for (0..WARMUP) |_| tp.parallelFor(n, ctx, func);

    var timer = MonotonicTimer.init(bench_io);
    for (0..ITERS) |_| tp.parallelFor(n, ctx, func);
    return @as(f64, @floatFromInt(timer.read())) / @as(f64, @floatFromInt(ITERS)) / 1_000_000.0;
}

fn forLightSeq(data: []u64, n: usize) f64 {
    const d = data[0..n];
    for (0..WARMUP) |_| for (d) |*v| {
        v.* = v.* *% (v.* +% 1);
    };

    var timer = MonotonicTimer.init(bench_io);
    for (0..ITERS) |_| for (d) |*v| {
        v.* = v.* *% (v.* +% 1);
    };
    return @as(f64, @floatFromInt(timer.read())) / @as(f64, @floatFromInt(ITERS)) / 1_000_000.0;
}

// ====================================================================
// Benchmark 2: parallelFor — BN254 field bind (heavy work per elem)
//   out[i] = a[i] * scalar — the polynomial binding pattern
// ====================================================================

fn forHeavyPar(tp: *ThreadPool, a: []F, out: []F, scalar: F, n: usize) f64 {
    const Ctx = struct { a: []const F, out: []F, s: F };
    const ctx = Ctx{ .a = a[0..n], .out = out[0..n], .s = scalar };
    const func = struct {
        fn f(c: Ctx, i: usize) void {
            c.out[i] = c.a[i].mul(c.s);
        }
    }.f;

    for (0..WARMUP) |_| tp.parallelFor(n, ctx, func);

    var timer = MonotonicTimer.init(bench_io);
    for (0..ITERS) |_| tp.parallelFor(n, ctx, func);
    return @as(f64, @floatFromInt(timer.read())) / @as(f64, @floatFromInt(ITERS)) / 1_000_000.0;
}

fn forHeavySeq(a: []const F, out: []F, scalar: F, n: usize) f64 {
    for (0..WARMUP) |_| for (0..n) |i| {
        out[i] = a[i].mul(scalar);
    };

    var timer = MonotonicTimer.init(bench_io);
    for (0..ITERS) |_| for (0..n) |i| {
        out[i] = a[i].mul(scalar);
    };
    return @as(f64, @floatFromInt(timer.read())) / @as(f64, @floatFromInt(ITERS)) / 1_000_000.0;
}

// ====================================================================
// Benchmark 3: Repeated dispatch — call parallelReduce DISPATCH_COUNT
// times at a fixed size. Measures dispatch + wake overhead.
// ====================================================================

const DISPATCH_COUNT: usize = 200;

fn repeatedDispatchPar(tp: *ThreadPool, data: []const u64, n: usize) f64 {
    const Ctx = struct { d: []const u64 };
    const ctx = Ctx{ .d = data[0..n] };
    const mapFn = struct {
        fn f(c: Ctx, s: usize, e: usize) u64 {
            var sum: u64 = 0;
            for (s..e) |i| sum +%= c.d[i] *% (c.d[i] +% 1);
            return sum;
        }
    }.f;
    const redFn = struct {
        fn f(a: u64, b: u64) u64 {
            return a +% b;
        }
    }.f;

    // warmup
    for (0..WARMUP) |_| {
        for (0..DISPATCH_COUNT) |_|
            std.mem.doNotOptimizeAway(tp.parallelReduce(u64, n, @as(u64, 0), ctx, mapFn, redFn));
    }

    var timer = MonotonicTimer.init(bench_io);
    for (0..RUNS) |_| {
        for (0..DISPATCH_COUNT) |_|
            std.mem.doNotOptimizeAway(tp.parallelReduce(u64, n, @as(u64, 0), ctx, mapFn, redFn));
    }
    // Return average time per single dispatch
    return @as(f64, @floatFromInt(timer.read())) / @as(f64, @floatFromInt(RUNS * DISPATCH_COUNT)) / 1_000_000.0;
}

fn repeatedDispatchSeq(data: []const u64, n: usize) f64 {
    const d = data[0..n];
    // warmup
    for (0..WARMUP) |_| {
        for (0..DISPATCH_COUNT) |_| {
            var sum: u64 = 0;
            for (d) |v| sum +%= v *% (v +% 1);
            std.mem.doNotOptimizeAway(sum);
        }
    }

    var timer = MonotonicTimer.init(bench_io);
    for (0..RUNS) |_| {
        for (0..DISPATCH_COUNT) |_| {
            var sum: u64 = 0;
            for (d) |v| sum +%= v *% (v +% 1);
            std.mem.doNotOptimizeAway(sum);
        }
    }
    return @as(f64, @floatFromInt(timer.read())) / @as(f64, @floatFromInt(RUNS * DISPATCH_COUNT)) / 1_000_000.0;
}

// ====================================================================
// Benchmark 4: Multi-array bind — bind NUM_ARRAYS arrays of size T
// using parallelForForce(NUM_ARRAYS, ...) — the stage 2-6 pattern.
// Each task does: arr[j] = arr[j] * scalar for all j in 0..T.
// ====================================================================

const NUM_ARRAYS: usize = 8;

fn multiArrayBindPar(tp: *ThreadPool, arrays: *[NUM_ARRAYS][]F, scalar: F) f64 {
    const Ctx = struct { arrs: *[NUM_ARRAYS][]F, s: F };
    const ctx = Ctx{ .arrs = arrays, .s = scalar };
    const func = struct {
        fn f(c: Ctx, idx: usize) void {
            const arr = c.arrs[idx];
            for (0..arr.len) |j| {
                arr[j] = arr[j].mul(c.s);
            }
        }
    }.f;

    for (0..WARMUP) |_| tp.parallelForForce(NUM_ARRAYS, ctx, func);

    var timer = MonotonicTimer.init(bench_io);
    for (0..ITERS) |_| tp.parallelForForce(NUM_ARRAYS, ctx, func);
    return @as(f64, @floatFromInt(timer.read())) / @as(f64, @floatFromInt(ITERS)) / 1_000_000.0;
}

fn multiArrayBindSeq(arrays: *[NUM_ARRAYS][]F, scalar: F) f64 {
    for (0..WARMUP) |_| {
        for (arrays) |arr| {
            for (0..arr.len) |j| arr[j] = arr[j].mul(scalar);
        }
    }

    var timer = MonotonicTimer.init(bench_io);
    for (0..ITERS) |_| {
        for (arrays) |arr| {
            for (0..arr.len) |j| arr[j] = arr[j].mul(scalar);
        }
    }
    return @as(f64, @floatFromInt(timer.read())) / @as(f64, @floatFromInt(ITERS)) / 1_000_000.0;
}

// ====================================================================
// Benchmark 5: Nested parallelFor — outer parallelForForce(NUM_ARRAYS)
// where each task does an inner parallelFor over T elements.
// This is the actual prover pattern: bind many arrays, each with
// internal parallelism.
// ====================================================================

fn nestedParallelPar(tp: *ThreadPool, arrays: *[NUM_ARRAYS][]F, scalar: F) f64 {
    const InnerCtx = struct { arr: []F, s: F };
    const OuterCtx = struct { arrs: *[NUM_ARRAYS][]F, s: F, tp: *ThreadPool };
    const ctx = OuterCtx{ .arrs = arrays, .s = scalar, .tp = tp };
    const outerFunc = struct {
        fn f(c: OuterCtx, idx: usize) void {
            const inner_ctx = InnerCtx{ .arr = c.arrs[idx], .s = c.s };
            const innerFunc = struct {
                fn g(ic: InnerCtx, j: usize) void {
                    ic.arr[j] = ic.arr[j].mul(ic.s);
                }
            }.g;
            c.tp.parallelFor(c.arrs[idx].len, inner_ctx, innerFunc);
        }
    }.f;

    for (0..WARMUP) |_| tp.parallelForForce(NUM_ARRAYS, ctx, outerFunc);

    var timer = MonotonicTimer.init(bench_io);
    for (0..ITERS) |_| tp.parallelForForce(NUM_ARRAYS, ctx, outerFunc);
    return @as(f64, @floatFromInt(timer.read())) / @as(f64, @floatFromInt(ITERS)) / 1_000_000.0;
}

// ====================================================================
// Main
// ====================================================================

pub fn main() !void {
    const allocator = std.heap.page_allocator;
    const tp = try ThreadPool.init(allocator, bench_io);
    defer tp.deinit();

    std.debug.print("Scaling micro-benchmark (Zig)\n", .{});
    std.debug.print("Threads: {}\n", .{tp.thread_count + 1});

    const sizes = [_]usize{ 1024, 4096, 16384, 65536, 262144, 524288, 1048576 };
    const max_n: usize = 1048576;

    // Allocate u64 data
    const u64_data = try allocator.alloc(u64, max_n);
    defer allocator.free(u64_data);
    for (0..max_n) |i| u64_data[i] = @as(u64, @truncate(i *% 0x9E3779B97F4A7C15 +% 1));

    // Allocate field data
    const fa = try allocator.alloc(F, max_n);
    defer allocator.free(fa);
    const fout = try allocator.alloc(F, max_n);
    defer allocator.free(fout);
    for (0..max_n) |i| {
        fa[i] = F.fromU64(@as(u64, @truncate(i *% 0x9E3779B97F4A7C15 +% 1)));
        fout[i] = F.zero();
    }
    const scalar = F.fromU64(0xDEADBEEFCAFEBABE);

    const header = "{s:>10} {s:>12} {s:>12} {s:>10}\n";
    const sep = "{s:->10} {s:->12} {s:->12} {s:->10}\n";
    const row = "{d:>10} {d:>10.3} ms {d:>10.3} ms {d:>9.2}x\n";

    // --- parallelFor light (u64 in-place write) ---
    std.debug.print("\n--- parallelFor: u64 in-place write ---\n", .{});
    std.debug.print("Config: {d} warmup, {d} iters, {d} runs (min-of-runs)\n\n", .{ WARMUP, ITERS, RUNS });
    std.debug.print(header, .{ "N", "Sequential", "Parallel", "Speedup" });
    std.debug.print(sep, .{ "", "", "", "" });
    for (sizes) |n| {
        var bs: f64 = std.math.inf(f64);
        var bp: f64 = std.math.inf(f64);
        for (0..RUNS) |_| {
            const s = forLightSeq(u64_data, n);
            const p = forLightPar(tp, u64_data, n);
            if (s < bs) bs = s;
            if (p < bp) bp = p;
        }
        std.debug.print(row, .{ n, bs, bp, bs / bp });
    }

    // --- parallelFor heavy (BN254 field bind) ---
    std.debug.print("\n--- parallelFor: BN254 field bind (out[i] = a[i] * scalar) ---\n", .{});
    std.debug.print(header, .{ "N", "Sequential", "Parallel", "Speedup" });
    std.debug.print(sep, .{ "", "", "", "" });
    for (sizes) |n| {
        var bs: f64 = std.math.inf(f64);
        var bp: f64 = std.math.inf(f64);
        for (0..RUNS) |_| {
            const s = forHeavySeq(fa, fout, scalar, n);
            const p = forHeavyPar(tp, fa, fout, scalar, n);
            if (s < bs) bs = s;
            if (p < bp) bp = p;
        }
        std.debug.print(row, .{ n, bs, bp, bs / bp });
    }

    // --- Repeated dispatch ---
    std.debug.print("\n--- Repeated dispatch: {d} calls of parallelReduce (per-call avg) ---\n", .{DISPATCH_COUNT});
    std.debug.print(header, .{ "N", "Sequential", "Parallel", "Speedup" });
    std.debug.print(sep, .{ "", "", "", "" });
    const dispatch_sizes = [_]usize{ 256, 1024, 4096, 16384, 65536, 262144 };
    for (dispatch_sizes) |n| {
        const s = repeatedDispatchSeq(u64_data, n);
        const p = repeatedDispatchPar(tp, u64_data, n);
        std.debug.print(row, .{ n, s, p, s / p });
    }

    // --- Multi-array bind ---
    std.debug.print("\n--- Multi-array bind: {d} arrays, parallelForForce ---\n", .{NUM_ARRAYS});
    std.debug.print(header, .{ "T (each)", "Sequential", "Parallel", "Speedup" });
    std.debug.print(sep, .{ "", "", "", "" });
    const arr_sizes = [_]usize{ 1024, 4096, 16384, 65536, 131072 };
    for (arr_sizes) |t| {
        // Allocate NUM_ARRAYS arrays of size t
        var arrays: [NUM_ARRAYS][]F = undefined;
        for (0..NUM_ARRAYS) |k| {
            arrays[k] = try allocator.alloc(F, t);
            for (0..t) |j| arrays[k][j] = F.fromU64(@as(u64, @truncate((k * t + j) *% 0x9E3779B97F4A7C15 +% 1)));
        }
        defer for (0..NUM_ARRAYS) |k| allocator.free(arrays[k]);

        var bs: f64 = std.math.inf(f64);
        var bp: f64 = std.math.inf(f64);
        for (0..RUNS) |_| {
            const s = multiArrayBindSeq(&arrays, scalar);
            const p = multiArrayBindPar(tp, &arrays, scalar);
            if (s < bs) bs = s;
            if (p < bp) bp = p;
        }
        std.debug.print(row, .{ t, bs, bp, bs / bp });
    }

    // --- Nested parallel (outer ForForce + inner For) ---
    std.debug.print("\n--- Nested parallel: ForForce({d} arrays) x For(T) ---\n", .{NUM_ARRAYS});
    std.debug.print(header, .{ "T (each)", "Sequential", "Nested par", "Speedup" });
    std.debug.print(sep, .{ "", "", "", "" });
    for (arr_sizes) |t| {
        var arrays: [NUM_ARRAYS][]F = undefined;
        for (0..NUM_ARRAYS) |k| {
            arrays[k] = try allocator.alloc(F, t);
            for (0..t) |j| arrays[k][j] = F.fromU64(@as(u64, @truncate((k * t + j) *% 0x9E3779B97F4A7C15 +% 1)));
        }
        defer for (0..NUM_ARRAYS) |k| allocator.free(arrays[k]);

        // Sequential baseline is same as multi-array bind
        var bs: f64 = std.math.inf(f64);
        var bp: f64 = std.math.inf(f64);
        for (0..RUNS) |_| {
            const s = multiArrayBindSeq(&arrays, scalar);
            const p = nestedParallelPar(tp, &arrays, scalar);
            if (s < bs) bs = s;
            if (p < bp) bp = p;
        }
        std.debug.print(row, .{ t, bs, bp, bs / bp });
    }
}
