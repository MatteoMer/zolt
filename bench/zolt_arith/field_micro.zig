const std = @import("std");
const zolt = @import("zolt");

const Fr = zolt.field.BN254Scalar;
const Fp = zolt.field.BN254BaseField;

const ITERATIONS: usize = 1_000_000;
const WARMUP: usize = 20_000;
const ELEMENTS: usize = 128;

fn fillFieldInputs(comptime F: type, a: *[ELEMENTS]F, b: *[ELEMENTS]F) void {
    for (0..ELEMENTS) |i| {
        a[i] = F.fromU64(@as(u64, @intCast(i + 1)) *% 0x9E3779B185EBCA87 +% 17);
        b[i] = F.fromU64(@as(u64, @intCast(i + 1)) *% 0xD6E8FEB86659FD93 +% 29);
    }
}

fn runUnary(comptime F: type, comptime field_name: []const u8, inputs: *const [ELEMENTS]F, comptime op_name: []const u8, op: fn (F) F) void {
    var sink = F.one();
    for (0..WARMUP) |i| sink = op(inputs[i % ELEMENTS]);
    std.mem.doNotOptimizeAway(&sink);

    var timer = std.time.Timer.start() catch unreachable;
    for (0..ITERATIONS) |i| sink = op(inputs[i % ELEMENTS]);
    const elapsed = timer.read();
    std.mem.doNotOptimizeAway(&sink);

    const ns = @as(f64, @floatFromInt(elapsed)) / @as(f64, ITERATIONS);
    std.debug.print("[FIELD-BENCH] field={s} op={s} ns_per_op={d:.3}\n", .{ field_name, op_name, ns });
}

fn runBinary(comptime F: type, comptime field_name: []const u8, a: *const [ELEMENTS]F, b: *const [ELEMENTS]F, comptime op_name: []const u8, op: fn (F, F) F) void {
    var sink = F.one();
    for (0..WARMUP) |i| sink = op(a[i % ELEMENTS], b[i % ELEMENTS]);
    std.mem.doNotOptimizeAway(&sink);

    var timer = std.time.Timer.start() catch unreachable;
    for (0..ITERATIONS) |i| sink = op(a[i % ELEMENTS], b[i % ELEMENTS]);
    const elapsed = timer.read();
    std.mem.doNotOptimizeAway(&sink);

    const ns = @as(f64, @floatFromInt(elapsed)) / @as(f64, ITERATIONS);
    std.debug.print("[FIELD-BENCH] field={s} op={s} ns_per_op={d:.3}\n", .{ field_name, op_name, ns });
}

fn runSumOfProducts(comptime F: type, comptime field_name: []const u8, a: *const [ELEMENTS]F, b: *const [ELEMENTS]F) void {
    var sink = F.one();
    for (0..WARMUP) |i| {
        const idx = i % ELEMENTS;
        sink = F.sumOfProducts(.{ a[idx], b[idx] }, .{ b[idx], a[idx] });
    }
    std.mem.doNotOptimizeAway(&sink);

    var timer = std.time.Timer.start() catch unreachable;
    for (0..ITERATIONS) |i| {
        const idx = i % ELEMENTS;
        sink = F.sumOfProducts(.{ a[idx], b[idx] }, .{ b[idx], a[idx] });
    }
    const elapsed = timer.read();
    std.mem.doNotOptimizeAway(&sink);

    const ns = @as(f64, @floatFromInt(elapsed)) / @as(f64, ITERATIONS);
    std.debug.print("[FIELD-BENCH] field={s} op=sumOfProducts ns_per_op={d:.3}\n", .{ field_name, ns });
}

fn benchField(comptime F: type, comptime field_name: []const u8) void {
    var a: [ELEMENTS]F = undefined;
    var b: [ELEMENTS]F = undefined;
    fillFieldInputs(F, &a, &b);

    runBinary(F, field_name, &a, &b, "add", struct {
        fn apply(x: F, y: F) F {
            return x.add(y);
        }
    }.apply);
    runBinary(F, field_name, &a, &b, "sub", struct {
        fn apply(x: F, y: F) F {
            return x.sub(y);
        }
    }.apply);
    runBinary(F, field_name, &a, &b, "mul", struct {
        fn apply(x: F, y: F) F {
            return x.mul(y);
        }
    }.apply);
    runUnary(F, field_name, &a, "square", struct {
        fn apply(x: F) F {
            return x.square();
        }
    }.apply);
    runUnary(F, field_name, &a, "inverse", struct {
        fn apply(x: F) F {
            return x.inverse() orelse F.zero();
        }
    }.apply);
    runUnary(F, field_name, &a, "toMontgomery", struct {
        fn apply(x: F) F {
            return x.fromMontgomery().toMontgomery();
        }
    }.apply);
    runUnary(F, field_name, &a, "fromMontgomery", struct {
        fn apply(x: F) F {
            return x.fromMontgomery();
        }
    }.apply);
    runSumOfProducts(F, field_name, &a, &b);
}

pub fn main() !void {
    std.debug.print("=== Zolt-Arith Field Microbench ===\n", .{});
    benchField(Fp, "Fp");
    benchField(Fr, "Fr");
}
