const std = @import("std");
const zolt = @import("zolt");

const Fr = zolt.field.BN254Scalar;
const Fp = zolt.field.BN254BaseField;
const S192 = zolt.field.S192;

fn benchField(comptime F: type, comptime name: []const u8) void {
    const n_elems = 128;
    const iters = 2_000_000;

    // Generate pseudo-random elements via fromU64
    var a: [n_elems]F = undefined;
    var b: [n_elems]F = undefined;
    for (0..n_elems) |i| {
        a[i] = F.fromU64(@as(u64, @intCast(i + 1)));
        b[i] = F.fromU64(@as(u64, @intCast(i + 1000)));
    }

    // Warm up
    var sink = F.one();
    for (0..10000) |_| {
        for (0..n_elems) |j| {
            sink = sink.mul(a[j]);
        }
    }
    std.mem.doNotOptimizeAway(&sink);

    // Bench mul throughput (independent ops, use black_box on inputs)
    {
        var timer = std.time.Timer.start() catch unreachable;
        var r: F = undefined;
        for (0..iters) |i| {
            r = a[i % n_elems].mul(b[i % n_elems]);
        }
        const elapsed = timer.read();
        std.mem.doNotOptimizeAway(&r);
        const mul_ns = @as(f64, @floatFromInt(elapsed)) / @as(f64, iters);
        std.debug.print("{s} mul throughput: {d:.1}ns\n", .{ name, mul_ns });
    }

    // Bench square throughput
    {
        var timer = std.time.Timer.start() catch unreachable;
        var r: F = undefined;
        for (0..iters) |i| {
            r = a[i % n_elems].square();
        }
        const elapsed = timer.read();
        std.mem.doNotOptimizeAway(&r);
        const sq_ns = @as(f64, @floatFromInt(elapsed)) / @as(f64, iters);
        std.debug.print("{s} square throughput: {d:.1}ns\n", .{ name, sq_ns });
    }

    // Bench sumOfProducts throughput
    {
        var timer = std.time.Timer.start() catch unreachable;
        var r: F = undefined;
        for (0..iters) |i| {
            const idx = i % n_elems;
            r = F.sumOfProducts(.{ a[idx], b[idx] }, .{ b[idx], a[idx] });
        }
        const elapsed = timer.read();
        std.mem.doNotOptimizeAway(&r);
        const sop_ns = @as(f64, @floatFromInt(elapsed)) / @as(f64, iters);
        std.debug.print("{s} sop throughput: {d:.1}ns\n", .{ name, sop_ns });
    }

    // Bench mul chain (latency)
    {
        var acc = a[0];
        var timer = std.time.Timer.start() catch unreachable;
        for (0..iters) |i| {
            acc = acc.mul(b[i % n_elems]);
        }
        const elapsed = timer.read();
        std.mem.doNotOptimizeAway(&acc);
        const mul_chain_ns = @as(f64, @floatFromInt(elapsed)) / @as(f64, iters);
        std.debug.print("{s} mul chain latency: {d:.1}ns\n", .{ name, mul_chain_ns });
    }

    // Bench square chain (latency)
    {
        var acc = a[0];
        var timer = std.time.Timer.start() catch unreachable;
        for (0..iters) |_| {
            acc = acc.square();
        }
        const elapsed = timer.read();
        std.mem.doNotOptimizeAway(&acc);
        const sq_chain_ns = @as(f64, @floatFromInt(elapsed)) / @as(f64, iters);
        std.debug.print("{s} square chain latency: {d:.1}ns\n", .{ name, sq_chain_ns });
    }

    // Bench add chain (latency)
    {
        var acc = a[0];
        var timer = std.time.Timer.start() catch unreachable;
        for (0..iters) |i| {
            acc = acc.add(b[i % n_elems]);
        }
        const elapsed = timer.read();
        std.mem.doNotOptimizeAway(&acc);
        const add_ns = @as(f64, @floatFromInt(elapsed)) / @as(f64, iters);
        std.debug.print("{s} add chain latency: {d:.1}ns\n", .{ name, add_ns });
    }

    // Bench sub chain (latency)
    {
        var acc = a[0];
        var timer = std.time.Timer.start() catch unreachable;
        for (0..iters) |i| {
            acc = acc.sub(b[i % n_elems]);
        }
        const elapsed = timer.read();
        std.mem.doNotOptimizeAway(&acc);
        const sub_ns = @as(f64, @floatFromInt(elapsed)) / @as(f64, iters);
        std.debug.print("{s} sub chain latency: {d:.1}ns\n", .{ name, sub_ns });
    }
}

fn benchS192() void {
    const n_elems = 128;
    const iters = 2_000_000;

    var a: [n_elems]S192 = undefined;
    var b: [n_elems]S192 = undefined;
    for (0..n_elems) |i| {
        a[i] = S192.fromI128(@as(i128, @intCast(i)) * 0x1234567890ABCDEF + 1);
        b[i] = S192.fromI128(-@as(i128, @intCast(i + 1000)) * 0xFEDCBA9876543210 - 1);
    }

    // Warm up
    var sink = a[0];
    for (0..10000) |_| {
        for (0..n_elems) |j| {
            sink = sink.add(a[j]);
        }
    }
    std.mem.doNotOptimizeAway(&sink);

    // Bench add chain (latency)
    {
        var acc = a[0];
        var timer = std.time.Timer.start() catch unreachable;
        for (0..iters) |i| {
            acc = acc.add(b[i % n_elems]);
        }
        const elapsed = timer.read();
        std.mem.doNotOptimizeAway(&acc);
        const ns = @as(f64, @floatFromInt(elapsed)) / @as(f64, iters);
        std.debug.print("S192 add chain latency: {d:.1}ns\n", .{ns});
    }

    // Bench sub chain (latency)
    {
        var acc = a[0];
        var timer = std.time.Timer.start() catch unreachable;
        for (0..iters) |i| {
            acc = acc.sub(b[i % n_elems]);
        }
        const elapsed = timer.read();
        std.mem.doNotOptimizeAway(&acc);
        const ns = @as(f64, @floatFromInt(elapsed)) / @as(f64, iters);
        std.debug.print("S192 sub chain latency: {d:.1}ns\n", .{ns});
    }

    // Bench mulI32 chain (latency)
    {
        var acc = a[0];
        var timer = std.time.Timer.start() catch unreachable;
        for (0..iters) |i| {
            acc = acc.mulI32(@as(i32, @intCast(i % n_elems)) + 1);
        }
        const elapsed = timer.read();
        std.mem.doNotOptimizeAway(&acc);
        const ns = @as(f64, @floatFromInt(elapsed)) / @as(f64, iters);
        std.debug.print("S192 mulI32 chain latency: {d:.1}ns\n", .{ns});
    }

    // Bench fmaddI32 (latency)
    {
        var acc = a[0];
        var timer = std.time.Timer.start() catch unreachable;
        for (0..iters) |i| {
            S192.fmaddI32(&acc, @as(i32, @intCast(i % n_elems)) + 1, b[i % n_elems]);
        }
        const elapsed = timer.read();
        std.mem.doNotOptimizeAway(&acc);
        const ns = @as(f64, @floatFromInt(elapsed)) / @as(f64, iters);
        std.debug.print("S192 fmaddI32 chain latency: {d:.1}ns\n", .{ns});
    }
}

pub fn main() !void {
    std.debug.print("=== Zolt BN254 field benchmarks ===\n", .{});
    for (0..3) |_| {
        benchField(Fp, "Fp");
        std.debug.print("\n", .{});
    }
    std.debug.print("---\n", .{});
    for (0..3) |_| {
        benchField(Fr, "Fr");
        std.debug.print("\n", .{});
    }
    std.debug.print("---\n", .{});
    for (0..3) |_| {
        benchS192();
        std.debug.print("\n", .{});
    }
}
