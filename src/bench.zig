//! Benchmarks for Zolt
//!
//! Run with: zig build bench

const std = @import("std");
const zolt = @import("root.zig");

/// Prevent the compiler from optimizing away the computation
fn doNotOptimize(value: anytype) void {
    const ptr: *const volatile @TypeOf(value) = &value;
    _ = ptr.*;
}

pub fn main() !void {
    std.debug.print("Zolt Benchmarks v{s}\n", .{zolt.version});
    std.debug.print("================================\n\n", .{});

    try benchFieldArithmetic();
    try benchBatchOperations();
    try benchPolynomialOperations();
    try benchMSM();
    try benchCommitment();

    std.debug.print("\nBenchmarks complete.\n", .{});
}

fn benchFieldArithmetic() !void {
    std.debug.print("Field Arithmetic (BN254 Scalar):\n", .{});

    const F = zolt.field.BN254Scalar;
    const iterations: usize = 100_000;

    // Addition
    {
        var a = F.fromU64(12345);
        const b = F.fromU64(67890);

        const start = std.time.nanoTimestamp();
        for (0..iterations) |_| {
            a = a.add(b);
        }
        const end = std.time.nanoTimestamp();
        doNotOptimize(a);

        const elapsed_ns: f64 = @floatFromInt(end - start);
        const ns_per_op = elapsed_ns / @as(f64, @floatFromInt(iterations));
        std.debug.print("  Addition:       {d:>7.1} ns/op\n", .{ns_per_op});
    }

    // Subtraction
    {
        var a = F.fromU64(12345);
        const b = F.fromU64(67890);

        const start = std.time.nanoTimestamp();
        for (0..iterations) |_| {
            a = a.sub(b);
        }
        const end = std.time.nanoTimestamp();
        doNotOptimize(a);

        const elapsed_ns: f64 = @floatFromInt(end - start);
        const ns_per_op = elapsed_ns / @as(f64, @floatFromInt(iterations));
        std.debug.print("  Subtraction:    {d:>7.1} ns/op\n", .{ns_per_op});
    }

    // Multiplication
    {
        var a = F.fromU64(12345);
        const b = F.fromU64(67890);

        const start = std.time.nanoTimestamp();
        for (0..iterations) |_| {
            a = a.mul(b);
        }
        const end = std.time.nanoTimestamp();
        doNotOptimize(a);

        const elapsed_ns: f64 = @floatFromInt(end - start);
        const ns_per_op = elapsed_ns / @as(f64, @floatFromInt(iterations));
        std.debug.print("  Multiplication: {d:>7.1} ns/op\n", .{ns_per_op});
    }

    // Squaring
    {
        var a = F.fromU64(12345);

        const start = std.time.nanoTimestamp();
        for (0..iterations) |_| {
            a = a.square();
        }
        const end = std.time.nanoTimestamp();
        doNotOptimize(a);

        const elapsed_ns: f64 = @floatFromInt(end - start);
        const ns_per_op = elapsed_ns / @as(f64, @floatFromInt(iterations));
        std.debug.print("  Squaring:       {d:>7.1} ns/op\n", .{ns_per_op});
    }

    // Inversion
    {
        var a = F.fromU64(12345);
        const inv_iterations = iterations / 100; // Inversion is much slower

        const start = std.time.nanoTimestamp();
        for (0..inv_iterations) |_| {
            a = a.inverse() orelse a;
        }
        const end = std.time.nanoTimestamp();
        doNotOptimize(a);

        const elapsed_ns: f64 = @floatFromInt(end - start);
        const ns_per_op = elapsed_ns / @as(f64, @floatFromInt(inv_iterations));
        std.debug.print("  Inversion:      {d:>7.1} ns/op\n", .{ns_per_op});
    }

    std.debug.print("\n", .{});
}

fn benchBatchOperations() !void {
    std.debug.print("Batch Operations:\n", .{});

    const F = zolt.field.BN254Scalar;
    const BatchOps = zolt.field.BatchOps;
    const allocator = std.heap.page_allocator;

    const sizes = [_]usize{ 64, 256, 1024 };

    for (sizes) |size| {
        const a = try allocator.alloc(F, size);
        defer allocator.free(a);
        const b = try allocator.alloc(F, size);
        defer allocator.free(b);
        const results = try allocator.alloc(F, size);
        defer allocator.free(results);

        for (0..size) |i| {
            a[i] = F.fromU64(@intCast(i + 1));
            b[i] = F.fromU64(@intCast(i + 2));
        }

        // Batch add
        const iterations: usize = 10000;
        {
            const start = std.time.nanoTimestamp();
            for (0..iterations) |_| {
                BatchOps.batchAdd(results, a, b);
            }
            const end = std.time.nanoTimestamp();
            doNotOptimize(results[0]);

            const elapsed_ns: f64 = @floatFromInt(end - start);
            const ns_per_op = elapsed_ns / @as(f64, @floatFromInt(iterations));
            std.debug.print("  Batch add (n={d:>4}):    {d:>8.1} ns/op\n", .{ size, ns_per_op });
        }

        // Inner product
        {
            var result: F = F.zero();
            const start = std.time.nanoTimestamp();
            for (0..iterations) |_| {
                result = BatchOps.innerProduct(a, b);
            }
            const end = std.time.nanoTimestamp();
            doNotOptimize(result);

            const elapsed_ns: f64 = @floatFromInt(end - start);
            const ns_per_op = elapsed_ns / @as(f64, @floatFromInt(iterations));
            std.debug.print("  Inner prod (n={d:>4}):   {d:>8.1} ns/op\n", .{ size, ns_per_op });
        }

        // Batch inverse
        {
            const inv_iterations = iterations / 10;
            const start = std.time.nanoTimestamp();
            for (0..inv_iterations) |_| {
                try BatchOps.batchInverse(results, a, allocator);
            }
            const end = std.time.nanoTimestamp();
            doNotOptimize(results[0]);

            const elapsed_ns: f64 = @floatFromInt(end - start);
            const ns_per_op = elapsed_ns / @as(f64, @floatFromInt(inv_iterations));
            std.debug.print("  Batch inv (n={d:>4}):    {d:>8.1} ns/op\n", .{ size, ns_per_op });
        }
    }

    std.debug.print("\n", .{});
}

fn benchPolynomialOperations() !void {
    std.debug.print("Polynomial Operations:\n", .{});

    const F = zolt.field.BN254Scalar;
    const allocator = std.heap.page_allocator;

    // Create polynomials of different sizes
    const sizes = [_]usize{ 256, 1024, 4096 };

    for (sizes) |size| {
        // Create evaluation vector
        const evals = try allocator.alloc(F, size);
        defer allocator.free(evals);

        for (0..size) |i| {
            evals[i] = F.fromU64(@intCast(i));
        }

        var poly = try zolt.poly.DensePolynomial(F).init(allocator, evals);
        defer poly.deinit();

        // Benchmark evaluation
        const iterations: usize = 100;
        const point = try allocator.alloc(F, poly.num_vars);
        defer allocator.free(point);

        for (0..point.len) |i| {
            point[i] = F.fromU64(@intCast(i + 1));
        }

        var result: F = F.zero();
        const start = std.time.nanoTimestamp();
        for (0..iterations) |_| {
            result = poly.evaluate(point);
        }
        const end = std.time.nanoTimestamp();
        doNotOptimize(result);

        const elapsed_ns: f64 = @floatFromInt(end - start);
        const ns_per_op = elapsed_ns / @as(f64, @floatFromInt(iterations));
        const us_per_op = ns_per_op / 1000.0;

        std.debug.print("  Poly eval (n={d:>4}): {d:>8.1} us/op\n", .{ size, us_per_op });
    }

    std.debug.print("\n", .{});
}

fn benchMSM() !void {
    std.debug.print("Multi-Scalar Multiplication:\n", .{});

    const F = zolt.field.BN254Scalar;
    const Fp = zolt.field.BN254BaseField;
    const Point = zolt.msm.AffinePoint(Fp);
    const MsmOps = zolt.msm.MSM(F, Fp);
    const allocator = std.heap.page_allocator;

    const sizes = [_]usize{ 16, 64, 256 };

    for (sizes) |size| {
        const scalars = try allocator.alloc(F, size);
        defer allocator.free(scalars);
        const points = try allocator.alloc(Point, size);
        defer allocator.free(points);

        // Generate random-ish scalars and points
        for (0..size) |i| {
            scalars[i] = F.fromU64(@intCast(i * 7 + 13));
            // Use generator times i as point
            const scalar_for_point = F.fromU64(@intCast(i + 1));
            const gen = Point.generator();
            const scaled = MsmOps.scalarMul(gen, scalar_for_point);
            points[i] = scaled.toAffine();
        }

        const iterations: usize = 10;
        var result: Point = Point.identity();
        const start = std.time.nanoTimestamp();
        for (0..iterations) |_| {
            result = MsmOps.compute(points, scalars);
        }
        const end = std.time.nanoTimestamp();
        doNotOptimize(result);

        const elapsed_ns: f64 = @floatFromInt(end - start);
        const ns_per_op = elapsed_ns / @as(f64, @floatFromInt(iterations));
        const ms_per_op = ns_per_op / 1_000_000.0;

        std.debug.print("  MSM (n={d:>3}): {d:>8.2} ms/op\n", .{ size, ms_per_op });
    }

    std.debug.print("\n", .{});
}

fn benchCommitment() !void {
    std.debug.print("HyperKZG Commitment:\n", .{});

    const F = zolt.field.BN254Scalar;
    const HyperKZG = zolt.poly.commitment.HyperKZG(F);
    const allocator = std.heap.page_allocator;

    const sizes = [_]usize{ 64, 256, 1024 };

    for (sizes) |size| {
        // Create evaluations
        const evals = try allocator.alloc(F, size);
        defer allocator.free(evals);

        for (0..size) |i| {
            evals[i] = F.fromU64(@intCast(i + 1));
        }

        var params = try HyperKZG.setup(allocator, size);
        defer params.deinit();

        // Benchmark commit
        const iterations: usize = 100;
        {
            var commitment: HyperKZG.Commitment = undefined;
            const start = std.time.nanoTimestamp();
            for (0..iterations) |_| {
                commitment = HyperKZG.commit(&params, evals);
            }
            const end = std.time.nanoTimestamp();
            doNotOptimize(commitment);

            const elapsed_ns: f64 = @floatFromInt(end - start);
            const ns_per_op = elapsed_ns / @as(f64, @floatFromInt(iterations));
            const us_per_op = ns_per_op / 1000.0;

            std.debug.print("  Commit (n={d:>4}):   {d:>8.1} us/op\n", .{ size, us_per_op });
        }
    }

    std.debug.print("\n", .{});
}

test "benchmarks compile" {
    // Just verify it compiles
}
