//! ThreadPool micro-benchmark: measures parallel reduce throughput
//! for direct comparison with the Rayon equivalent (bench_rayon.rs).
//!
//! Build: zig build bench-tp -Doptimize=ReleaseFast
//! Run:   ./zig-out/bin/bench-tp

const std = @import("std");

const MonotonicTimer = struct {
    start_ns: u64,
    pub fn start() error{}!MonotonicTimer {
        return .{ .start_ns = clockNs() };
    }
    pub fn read(self: MonotonicTimer) u64 {
        return clockNs() - self.start_ns;
    }
    pub fn reset(self: *MonotonicTimer) void {
        self.start_ns = clockNs();
    }
    fn clockNs() u64 {
        var ts: std.c.timespec = undefined;
        _ = std.c.clock_gettime(.MONOTONIC, &ts);
        return @intCast(@as(i128, ts.sec) * std.time.ns_per_s + ts.nsec);
    }
};

const zolt = @import("zolt");
const F = zolt.field.BN254Scalar;
const ThreadPool = zolt.utils.ThreadPool;
const UnreducedProductAccum = zolt.field.UnreducedProductAccum;

const WARMUP = 5;
const ITERS = 100;
const RUNS = 5; // take min across runs

fn benchParallelReduce(tp: *ThreadPool, a: []const F, b: []const F, half: usize) f64 {
    const Ctx = struct { a: []const F, b: []const F };
    const ctx = Ctx{ .a = a, .b = b };

    const mapFn = struct {
        fn f(c: Ctx, start: usize, end: usize) [2]F {
            var acc0 = UnreducedProductAccum.zero();
            var acc1 = UnreducedProductAccum.zero();
            for (start..end) |i| {
                acc0.addAssign(c.a[2 * i].mulToProductAccum(c.b[2 * i]));
                acc1.addAssign(c.a[2 * i + 1].mulToProductAccum(c.b[2 * i + 1]));
            }
            return .{ acc0.reduce(), acc1.reduce() };
        }
    }.f;

    const reduceFn = struct {
        fn f(x: [2]F, y: [2]F) [2]F {
            return .{ x[0].add(y[0]), x[1].add(y[1]) };
        }
    }.f;

    const identity = [2]F{ F.zero(), F.zero() };

    for (0..WARMUP) |_| {
        std.mem.doNotOptimizeAway(tp.parallelReduce([2]F, half, identity, ctx, mapFn, reduceFn));
    }

    var timer = MonotonicTimer.start() catch unreachable;
    for (0..ITERS) |_| {
        std.mem.doNotOptimizeAway(tp.parallelReduce([2]F, half, identity, ctx, mapFn, reduceFn));
    }
    const ns = timer.read();
    return @as(f64, @floatFromInt(ns)) / @as(f64, @floatFromInt(ITERS)) / 1_000_000.0;
}

fn benchSequential(a: []const F, b: []const F, half: usize) f64 {
    for (0..WARMUP) |_| {
        var acc0 = UnreducedProductAccum.zero();
        var acc1 = UnreducedProductAccum.zero();
        for (0..half) |i| {
            acc0.addAssign(a[2 * i].mulToProductAccum(b[2 * i]));
            acc1.addAssign(a[2 * i + 1].mulToProductAccum(b[2 * i + 1]));
        }
        std.mem.doNotOptimizeAway(acc0.reduce());
        std.mem.doNotOptimizeAway(acc1.reduce());
    }

    var timer = MonotonicTimer.start() catch unreachable;
    for (0..ITERS) |_| {
        var acc0 = UnreducedProductAccum.zero();
        var acc1 = UnreducedProductAccum.zero();
        for (0..half) |i| {
            acc0.addAssign(a[2 * i].mulToProductAccum(b[2 * i]));
            acc1.addAssign(a[2 * i + 1].mulToProductAccum(b[2 * i + 1]));
        }
        std.mem.doNotOptimizeAway(acc0.reduce());
        std.mem.doNotOptimizeAway(acc1.reduce());
    }
    const ns = timer.read();
    return @as(f64, @floatFromInt(ns)) / @as(f64, @floatFromInt(ITERS)) / 1_000_000.0;
}

pub fn main() !void {
    const allocator = std.heap.page_allocator;
    const tp = try ThreadPool.init(allocator);
    defer tp.deinit();

    std.debug.print("ThreadPool micro-benchmark (Zig)\n", .{});
    std.debug.print("Threads: {}\n", .{tp.thread_count + 1});
    std.debug.print("Workload: parallel reduce Σ a[i]*b[i] with UnreducedProductAccum\n", .{});
    std.debug.print("Config: {d} warmup, {d} iters, {d} runs (min-of-runs)\n\n", .{ WARMUP, ITERS, RUNS });

    const sizes = [_]usize{ 1024, 4096, 16384, 65536, 262144, 524288 };

    std.debug.print("{s:>10} {s:>12} {s:>12} {s:>10}\n", .{ "N (pairs)", "Sequential", "Parallel", "Speedup" });
    std.debug.print("{s:->10} {s:->12} {s:->12} {s:->10}\n", .{ "", "", "", "" });

    for (sizes) |n| {
        const half = n;
        const len = half * 2;

        const a = try allocator.alloc(F, len);
        defer allocator.free(a);
        const b = try allocator.alloc(F, len);
        defer allocator.free(b);

        for (0..len) |i| {
            a[i] = F.fromU64(@as(u64, @truncate(i *% 0x9E3779B97F4A7C15 +% 1)));
            b[i] = F.fromU64(@as(u64, @truncate(i *% 0x517CC1B727220A95 +% 1)));
        }

        // Take minimum across RUNS (eliminates upward noise from scheduling/thermals)
        var best_seq: f64 = std.math.inf(f64);
        var best_par: f64 = std.math.inf(f64);
        for (0..RUNS) |_| {
            const s = benchSequential(a, b, half);
            const p = benchParallelReduce(tp, a, b, half);
            if (s < best_seq) best_seq = s;
            if (p < best_par) best_par = p;
        }
        const speedup = best_seq / best_par;

        std.debug.print("{d:>10} {d:>10.3} ms {d:>10.3} ms {d:>9.2}x\n", .{ n, best_seq, best_par, speedup });
    }
}
