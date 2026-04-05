//! Shared microbenchmark harness for zolt-arith.
//! Zero external dependencies — only std.
//!
//! Usage:
//!   const harness = @import("bench_harness.zig");
//!   _ = harness.run("field_Fp", "mul", harness.Config.field, Fp, struct {
//!       fn call(i: usize) Fp { return a[i % N].mul(b[i % N]); }
//!   }.call);

const std = @import("std");

const MAX_SAMPLES: usize = 101;

pub const Config = struct {
    warmup_iters: usize = 5_000,
    sample_count: usize = 21,
    iters_per_sample: usize = 100_000,

    pub const field: Config = .{
        .warmup_iters = 20_000,
        .sample_count = 21,
        .iters_per_sample = 100_000,
    };

    pub const pairing: Config = .{
        .warmup_iters = 10,
        .sample_count = 21,
        .iters_per_sample = 100,
    };
};

pub const Stats = struct {
    min_ns: f64,
    median_ns: f64,
    mean_ns: f64,
    p99_ns: f64,
    stddev_ns: f64,
    sample_count: usize,
    iters_per_sample: usize,
};

pub fn run(
    comptime group: []const u8,
    comptime op_name: []const u8,
    config: Config,
    comptime T: type,
    comptime body: fn (usize) T,
) Stats {
    const n = @min(config.sample_count, MAX_SAMPLES);
    const iters = config.iters_per_sample;

    // --- Warmup ---
    var sink: T = undefined;
    for (0..config.warmup_iters) |i| {
        sink = body(i);
    }
    std.mem.doNotOptimizeAway(&sink);

    // --- Collect samples ---
    var samples: [MAX_SAMPLES]f64 = undefined;
    for (0..n) |s| {
        var timer = std.time.Timer.start() catch unreachable;
        for (0..iters) |i| {
            sink = body(i);
        }
        std.mem.doNotOptimizeAway(&sink);
        const elapsed_ns: u64 = timer.read();
        samples[s] = @as(f64, @floatFromInt(elapsed_ns)) /
            @as(f64, @floatFromInt(iters));
    }

    // --- Compute stats ---
    const slice = samples[0..n];
    std.sort.pdq(f64, slice, {}, lessThanF64);

    var sum: f64 = 0;
    for (slice) |v| sum += v;
    const mean = sum / @as(f64, @floatFromInt(n));

    var var_sum: f64 = 0;
    for (slice) |v| {
        const d = v - mean;
        var_sum += d * d;
    }
    const stddev = @sqrt(var_sum / @as(f64, @floatFromInt(n)));

    const median = if (n % 2 == 1)
        slice[n / 2]
    else
        (slice[n / 2 - 1] + slice[n / 2]) / 2.0;

    const p99_idx = @as(usize, @intFromFloat(@ceil(0.99 * @as(f64, @floatFromInt(n))))) - 1;

    const stats = Stats{
        .min_ns = slice[0],
        .median_ns = median,
        .mean_ns = mean,
        .p99_ns = slice[p99_idx],
        .stddev_ns = stddev,
        .sample_count = n,
        .iters_per_sample = iters,
    };

    printResult(group, op_name, stats);
    return stats;
}

fn lessThanF64(_: void, a: f64, b: f64) bool {
    return a < b;
}

const FmtTime = struct { val: f64, unit: []const u8 };

fn formatTime(ns: f64) FmtTime {
    if (ns >= 1_000_000.0) return .{ .val = ns / 1_000_000.0, .unit = "ms" };
    if (ns >= 1_000.0) return .{ .val = ns / 1_000.0, .unit = "us" };
    return .{ .val = ns, .unit = "ns" };
}

fn printResult(comptime group: []const u8, comptime op_name: []const u8, s: Stats) void {
    const min_f = formatTime(s.min_ns);
    const med_f = formatTime(s.median_ns);
    const mean_f = formatTime(s.mean_ns);
    const p99_f = formatTime(s.p99_ns);
    const sd_f = formatTime(s.stddev_ns);

    std.debug.print(
        "[BENCH] group={s} op={s} min={d:.1}{s} median={d:.1}{s} mean={d:.1}{s} p99={d:.1}{s} stddev={d:.1}{s} samples={d}x{d}\n",
        .{
            group,          op_name,
            min_f.val,      min_f.unit,
            med_f.val,      med_f.unit,
            mean_f.val,     mean_f.unit,
            p99_f.val,      p99_f.unit,
            sd_f.val,       sd_f.unit,
            s.sample_count, s.iters_per_sample,
        },
    );
}
