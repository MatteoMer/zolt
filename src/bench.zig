//! Benchmarks for Zolt
//!
//! Run with: zig build bench

const std = @import("std");
const zolt = @import("root.zig");

pub fn main() !void {
    std.debug.print("Zolt Benchmarks v{s}\n", .{zolt.version});
    std.debug.print("================================\n\n", .{});

    try benchFieldArithmetic();
    try benchBatchOperations();
    try benchPolynomialOperations();
    try benchMSM();
    try benchSumcheck();

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

        const elapsed_ns: f64 = @floatFromInt(end - start);
        const ns_per_op = elapsed_ns / @as(f64, @floatFromInt(iterations));
        std.debug.print("  Addition:       {d:>7.1} ns/op\n", .{ns_per_op});
        _ = a; // prevent optimization
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

        const elapsed_ns: f64 = @floatFromInt(end - start);
        const ns_per_op = elapsed_ns / @as(f64, @floatFromInt(iterations));
        std.debug.print("  Subtraction:    {d:>7.1} ns/op\n", .{ns_per_op});
        _ = a;
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

        const elapsed_ns: f64 = @floatFromInt(end - start);
        const ns_per_op = elapsed_ns / @as(f64, @floatFromInt(iterations));
        std.debug.print("  Multiplication: {d:>7.1} ns/op\n", .{ns_per_op});
        _ = a;
    }

    // Squaring
    {
        var a = F.fromU64(12345);

        const start = std.time.nanoTimestamp();
        for (0..iterations) |_| {
            a = a.square();
        }
        const end = std.time.nanoTimestamp();

        const elapsed_ns: f64 = @floatFromInt(end - start);
        const ns_per_op = elapsed_ns / @as(f64, @floatFromInt(iterations));
        std.debug.print("  Squaring:       {d:>7.1} ns/op\n", .{ns_per_op});
        _ = a;
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

        const elapsed_ns: f64 = @floatFromInt(end - start);
        const ns_per_op = elapsed_ns / @as(f64, @floatFromInt(inv_iterations));
        std.debug.print("  Inversion:      {d:>7.1} ns/op\n", .{ns_per_op});
        _ = a;
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

            const elapsed_ns: f64 = @floatFromInt(end - start);
            const ns_per_op = elapsed_ns / @as(f64, @floatFromInt(iterations));
            std.debug.print("  Batch add (n={d:>4}):    {d:>8.1} ns/op\n", .{ size, ns_per_op });
        }

        // Inner product
        {
            const start = std.time.nanoTimestamp();
            var result: F = F.zero();
            for (0..iterations) |_| {
                result = BatchOps.innerProduct(a, b);
            }
            const end = std.time.nanoTimestamp();

            const elapsed_ns: f64 = @floatFromInt(end - start);
            const ns_per_op = elapsed_ns / @as(f64, @floatFromInt(iterations));
            std.debug.print("  Inner prod (n={d:>4}):   {d:>8.1} ns/op\n", .{ size, ns_per_op });
            _ = result;
        }

        // Batch inverse
        {
            const inv_iterations = iterations / 10;
            const start = std.time.nanoTimestamp();
            for (0..inv_iterations) |_| {
                try BatchOps.batchInverse(results, a, allocator);
            }
            const end = std.time.nanoTimestamp();

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

        const start = std.time.nanoTimestamp();
        var result: F = F.zero();
        for (0..iterations) |_| {
            result = poly.evaluate(point);
        }
        const end = std.time.nanoTimestamp();

        const elapsed_ns: f64 = @floatFromInt(end - start);
        const ns_per_op = elapsed_ns / @as(f64, @floatFromInt(iterations));
        const us_per_op = ns_per_op / 1000.0;

        std.debug.print("  Poly eval (n={d:>4}): {d:>8.1} us/op\n", .{ size, us_per_op });
        _ = result;
    }

    std.debug.print("\n", .{});
}

fn benchMSM() !void {
    std.debug.print("Multi-Scalar Multiplication:\n", .{});

    const F = zolt.field.BN254Scalar;
    const AffinePoint = zolt.msm.AffinePoint;
    const allocator = std.heap.page_allocator;

    const sizes = [_]usize{ 16, 64, 256 };

    for (sizes) |size| {
        const scalars = try allocator.alloc(F, size);
        defer allocator.free(scalars);
        const points = try allocator.alloc(AffinePoint, size);
        defer allocator.free(points);

        // Generate random-ish scalars and points
        for (0..size) |i| {
            scalars[i] = F.fromU64(@intCast(i * 7 + 13));
            // Use generator times i as point
            const scalar_for_point = F.fromU64(@intCast(i + 1));
            const gen = zolt.msm.ProjectivePoint.generator();
            const scaled = gen.scalarMul(scalar_for_point);
            points[i] = scaled.toAffine();
        }

        const iterations: usize = 10;
        const start = std.time.nanoTimestamp();
        for (0..iterations) |_| {
            _ = zolt.msm.pippenger(scalars, points, allocator) catch continue;
        }
        const end = std.time.nanoTimestamp();

        const elapsed_ns: f64 = @floatFromInt(end - start);
        const ns_per_op = elapsed_ns / @as(f64, @floatFromInt(iterations));
        const ms_per_op = ns_per_op / 1_000_000.0;

        std.debug.print("  MSM (n={d:>3}): {d:>8.2} ms/op\n", .{ size, ms_per_op });
    }

    std.debug.print("\n", .{});
}

fn benchSumcheck() !void {
    std.debug.print("Sumcheck Protocol:\n", .{});

    const F = zolt.field.BN254Scalar;
    const allocator = std.heap.page_allocator;

    const sizes = [_]usize{ 64, 256, 1024 };

    for (sizes) |size| {
        // Create polynomial coefficients
        const evals = try allocator.alloc(F, size);
        defer allocator.free(evals);

        for (0..size) |i| {
            evals[i] = F.fromU64(@intCast(i + 1));
        }

        var poly = try zolt.poly.DensePolynomial(F).init(allocator, evals);
        defer poly.deinit();

        // Create sumcheck prover
        const iterations: usize = 10;
        const start = std.time.nanoTimestamp();

        for (0..iterations) |_| {
            var prover = zolt.subprotocols.Sumcheck(F).Prover.init(allocator);
            defer prover.deinit();

            // Generate proof
            for (0..poly.num_vars) |round| {
                const univariate = prover.generateRoundPolynomial(poly.evals, poly.num_vars, round);
                // In real use, verifier would send challenge; here we use deterministic
                const challenge = F.fromU64(@intCast(round + 1));
                _ = challenge;
                _ = univariate;
            }
        }

        const end = std.time.nanoTimestamp();

        const elapsed_ns: f64 = @floatFromInt(end - start);
        const ns_per_op = elapsed_ns / @as(f64, @floatFromInt(iterations));
        const us_per_op = ns_per_op / 1000.0;

        std.debug.print("  Sumcheck prove (n={d:>4}): {d:>8.1} us/op\n", .{ size, us_per_op });
    }

    std.debug.print("\n", .{});
}

test "benchmarks compile" {
    // Just verify it compiles
}
