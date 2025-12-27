//! Benchmarks for Zolt
//!
//! Run with: zig build bench

const std = @import("std");
const zolt = @import("root.zig");

pub fn main() !void {
    std.debug.print("Zolt Benchmarks v{s}\n", .{zolt.version});
    std.debug.print("================\n\n", .{});

    try benchFieldArithmetic();
    try benchPolynomialOperations();

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
        std.debug.print("  Addition: {d:.1} ns/op\n", .{ns_per_op});
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
        std.debug.print("  Subtraction: {d:.1} ns/op\n", .{ns_per_op});
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
        for (0..iterations) |_| {
            _ = poly.evaluate(point);
        }
        const end = std.time.nanoTimestamp();

        const elapsed_ns: f64 = @floatFromInt(end - start);
        const ns_per_op = elapsed_ns / @as(f64, @floatFromInt(iterations));
        const us_per_op = ns_per_op / 1000.0;

        std.debug.print("  Polynomial eval (n={d}): {d:.1} us/op\n", .{ size, us_per_op });
    }

    std.debug.print("\n", .{});
}

test "benchmarks compile" {
    // Just verify it compiles
}
