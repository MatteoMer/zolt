/// MSM Benchmark: Pippenger MSM for BN254 G1 and G2
/// Tests same sizes as Dory opening proof (2..4096 points)
/// Outputs machine-readable [MSM-BENCH] lines for comparison with arkworks.
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
        const io: std.Io = std.Io.Threaded.global_single_threaded.io();
        return @intCast(@max(0, std.Io.Timestamp.now(io, .boot).nanoseconds));
    }
};

const zolt = @import("zolt");
const field = zolt.field;
const msm_mod = zolt.msm;
const pairing = field.pairing;
const ThreadPool = zolt.utils.ThreadPool;

const Fr = field.BN254Scalar;
const Fp = field.BN254BaseField;
const G1Affine = msm_mod.AffinePoint(Fp);
const G1Projective = msm_mod.ProjectivePoint(Fp);
const G1MSM = msm_mod.MSM(Fr, Fp);
const G2Point = pairing.G2Point;

const dory = zolt.poly.commitment.dory;

const WARMUP = 2;
const MIN_ITERS = 5;
const MIN_TIME_NS: u64 = 500_000_000; // 500ms minimum per size

fn generateG1Bases(alloc: std.mem.Allocator, n: usize) ![]G1Affine {
    var bases = try alloc.alloc(G1Affine, n);
    const gen = G1Affine.generator();
    var proj = G1Projective.fromAffine(gen);
    for (0..n) |i| {
        bases[i] = proj.toAffine();
        proj = proj.double();
    }
    return bases;
}

fn generateG2Bases(alloc: std.mem.Allocator, n: usize) ![]G2Point {
    var bases = try alloc.alloc(G2Point, n);
    const gen = G2Point.generator();
    var acc = pairing.G2Projective.fromAffine(gen);
    for (0..n) |i| {
        bases[i] = acc.toAffine();
        acc = acc.double();
    }
    return bases;
}

fn generateFrScalars(alloc: std.mem.Allocator, n: usize) ![]Fr {
    var scalars = try alloc.alloc(Fr, n);
    // Deterministic pseudo-random scalars
    var seed: u64 = 0xdeadbeef;
    for (0..n) |i| {
        seed = seed *% 6364136223846793005 +% 1442695040888963407;
        const s2 = seed *% 6364136223846793005 +% 1442695040888963407;
        const s3 = s2 *% 6364136223846793005 +% 1442695040888963407;
        const s4 = s3 *% 6364136223846793005 +% 1442695040888963407;
        scalars[i] = Fr.fromU64(seed).add(Fr.fromU64(s2)).add(Fr.fromU64(s3)).add(Fr.fromU64(s4));
        seed = s4;
    }
    return scalars;
}

fn generateI128Scalars(alloc: std.mem.Allocator, n: usize) ![]i128 {
    var scalars = try alloc.alloc(i128, n);
    var seed: u64 = 0xcafebabe;
    for (0..n) |i| {
        seed = seed *% 6364136223846793005 +% 1442695040888963407;
        const hi: u128 = @as(u128, seed) << 64;
        seed = seed *% 6364136223846793005 +% 1442695040888963407;
        const lo: u128 = seed;
        scalars[i] = @bitCast(hi | lo);
    }
    return scalars;
}

fn benchMsmG1(bases: []const G1Affine, scalars: []const Fr, tp: ?*ThreadPool) f64 {
    const n = bases.len;
    _ = n;

    // Warmup
    for (0..WARMUP) |_| {
        _ = G1MSM.computeWithPool(bases, scalars, tp);
    }

    // Timed runs
    var total_ns: u64 = 0;
    var iters: u64 = 0;
    while (iters < MIN_ITERS or total_ns < MIN_TIME_NS) {
        var timer = MonotonicTimer.start() catch unreachable;
        _ = G1MSM.computeWithPool(bases, scalars, tp);
        total_ns += timer.read();
        iters += 1;
    }
    return @as(f64, @floatFromInt(total_ns)) / @as(f64, @floatFromInt(iters)) / 1_000_000.0;
}

fn benchMsmG1I128(bases: []const G1Affine, scalars: []const i128, tp: ?*ThreadPool) f64 {
    // Warmup
    for (0..WARMUP) |_| {
        _ = G1MSM.computeI128(bases, scalars, tp);
    }

    // Timed runs
    var total_ns: u64 = 0;
    var iters: u64 = 0;
    while (iters < MIN_ITERS or total_ns < MIN_TIME_NS) {
        var timer = MonotonicTimer.start() catch unreachable;
        _ = G1MSM.computeI128(bases, scalars, tp);
        total_ns += timer.read();
        iters += 1;
    }
    return @as(f64, @floatFromInt(total_ns)) / @as(f64, @floatFromInt(iters)) / 1_000_000.0;
}

fn benchMsmG2(bases: []const G2Point, scalars: []const Fr, tp: ?*ThreadPool) f64 {
    // Warmup
    for (0..WARMUP) |_| {
        _ = dory.msmG2Bench(Fr, bases, scalars, tp);
    }

    // Timed runs
    var total_ns: u64 = 0;
    var iters: u64 = 0;
    while (iters < MIN_ITERS or total_ns < MIN_TIME_NS) {
        var timer = MonotonicTimer.start() catch unreachable;
        _ = dory.msmG2Bench(Fr, bases, scalars, tp);
        total_ns += timer.read();
        iters += 1;
    }
    return @as(f64, @floatFromInt(total_ns)) / @as(f64, @floatFromInt(iters)) / 1_000_000.0;
}

pub fn main() !void {
    const alloc = std.heap.page_allocator;

    // Initialize thread pool
    var tp = try ThreadPool.init(alloc);
    defer tp.deinit();

    const sizes = [_]usize{ 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144 };
    const max_size = 262144;

    std.debug.print("=== Zolt MSM Benchmark (BN254) ===\n", .{});
    std.debug.print("Threads: {}\n\n", .{tp.thread_count});

    // Pre-generate bases and scalars for max size
    const g1_bases = try generateG1Bases(alloc, max_size);
    const g2_bases = try generateG2Bases(alloc, max_size);
    const fr_scalars = try generateFrScalars(alloc, max_size);
    const i128_scalars = try generateI128Scalars(alloc, max_size);

    // Header
    std.debug.print("{s:>6} │ {s:>10} {s:>10} {s:>10} │ {s:>10} {s:>10} {s:>10}\n", .{
        "N", "G1-Fr", "G1-i128", "G2-Fr", "G1-Fr/P", "G1-i128/P", "G2-Fr/P",
    });
    std.debug.print("───────┼─────────────────────────────────┼─────────────────────────────────\n", .{});

    for (sizes) |n| {
        const b1 = g1_bases[0..n];
        const sf = fr_scalars[0..n];
        const si = i128_scalars[0..n];

        // Sequential (no pool)
        const g1_fr_ms = benchMsmG1(b1, sf, null);
        const g1_i128_ms = benchMsmG1I128(b1, si, null);
        // G2 only up to 8192 (very slow beyond)
        const g2_fr_ms = if (n <= 8192) benchMsmG2(g2_bases[0..n], sf, null) else 0;

        // Parallel (with pool)
        const g1_fr_p_ms = benchMsmG1(b1, sf, tp);
        const g1_i128_p_ms = benchMsmG1I128(b1, si, tp);
        const g2_fr_p_ms = if (n <= 8192) benchMsmG2(g2_bases[0..n], sf, tp) else 0;

        std.debug.print("{d:>6} │ {d:>9.3}  {d:>9.3}  {d:>9.3}  │ {d:>9.3}  {d:>9.3}  {d:>9.3}\n", .{
            n, g1_fr_ms, g1_i128_ms, g2_fr_ms, g1_fr_p_ms, g1_i128_p_ms, g2_fr_p_ms,
        });

        // Machine-readable output
        std.debug.print("[MSM-BENCH] n={d} g1_fr={d:.3} g1_i128={d:.3} g2_fr={d:.3} g1_fr_par={d:.3} g1_i128_par={d:.3} g2_fr_par={d:.3}\n", .{
            n, g1_fr_ms, g1_i128_ms, g2_fr_ms, g1_fr_p_ms, g1_i128_p_ms, g2_fr_p_ms,
        });
    }
}
