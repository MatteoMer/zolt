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
    try benchEmulator();
    try benchProver();
    try benchProofSize();
    try benchVerifier();

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

fn benchEmulator() !void {
    std.debug.print("Emulator Execution:\n", .{});

    const allocator = std.heap.page_allocator;
    const common = zolt.common;
    const tracer = zolt.tracer;

    // Program that sums 1 to 100 (more cycles for realistic benchmark)
    const program = [_]u8{
        0x93, 0x00, 0x40, 0x06, // addi x1, x0, 100
        0x13, 0x01, 0x00, 0x00, // addi x2, x0, 0
        // loop:
        0x33, 0x01, 0x11, 0x00, // add x2, x2, x1
        0x93, 0x80, 0xf0, 0xff, // addi x1, x1, -1
        0xe3, 0x9c, 0x00, 0xfe, // bne x1, x0, -8
        0x73, 0x00, 0x00, 0x00, // ecall
    };

    const iterations: usize = 100;
    const config = common.MemoryConfig{ .program_size = 256 };

    // Benchmark emulation
    {
        const start = std.time.nanoTimestamp();
        for (0..iterations) |_| {
            var emu = tracer.Emulator.init(allocator, &config);
            defer emu.deinit();
            emu.max_cycles = 512;
            emu.loadProgram(&program) catch continue;
            emu.run() catch continue;
        }
        const end = std.time.nanoTimestamp();

        const elapsed_ns: f64 = @floatFromInt(end - start);
        const ns_per_op = elapsed_ns / @as(f64, @floatFromInt(iterations));
        const us_per_op = ns_per_op / 1000.0;

        std.debug.print("  Sum 1-100 loop:     {d:>8.1} us/op\n", .{us_per_op});
    }

    std.debug.print("\n", .{});
}

fn benchProver() !void {
    std.debug.print("zkVM Prover:\n", .{});

    const F = zolt.field.BN254Scalar;
    const allocator = std.heap.page_allocator;
    const common = zolt.common;
    const tracer = zolt.tracer;
    const prover = zolt.zkvm.prover;
    const transcripts = zolt.transcripts;

    // Simple program: addi x1, x0, 10; ecall
    const program = [_]u8{
        0x93, 0x00, 0xa0, 0x00, // addi x1, x0, 10
        0x73, 0x00, 0x00, 0x00, // ecall
    };

    const config = common.MemoryConfig{ .program_size = 64 };

    // Run emulation once to get trace
    var emu = tracer.Emulator.init(allocator, &config);
    defer emu.deinit();
    emu.max_cycles = 8;
    try emu.loadProgram(&program);
    try emu.run();

    const log_k: usize = 16;
    const iterations: usize = 3;

    // Benchmark proving
    {
        const start = std.time.nanoTimestamp();
        for (0..iterations) |_| {
            var multi_stage = prover.MultiStageProver(F).init(
                allocator,
                &emu.trace,
                &emu.ram.trace,
                &emu.lookup_trace,
                log_k,
                common.constants.RAM_START_ADDRESS,
            );
            defer multi_stage.deinit();

            var transcript = try transcripts.Transcript(F).init(allocator, "bench");
            defer transcript.deinit();

            _ = multi_stage.prove(&transcript) catch continue;
        }
        const end = std.time.nanoTimestamp();

        const elapsed_ns: f64 = @floatFromInt(end - start);
        const ns_per_op = elapsed_ns / @as(f64, @floatFromInt(iterations));
        const ms_per_op = ns_per_op / 1_000_000.0;

        std.debug.print("  Simple (2 steps):   {d:>8.1} ms/op\n", .{ms_per_op});
    }

    // Benchmark with more steps
    var emu2 = tracer.Emulator.init(allocator, &config);
    defer emu2.deinit();
    emu2.max_cycles = 32;

    // Program with a small loop (4 iterations)
    const loop_program = [_]u8{
        0x93, 0x00, 0x40, 0x00, // addi x1, x0, 4
        0x13, 0x01, 0x00, 0x00, // addi x2, x0, 0
        0x33, 0x01, 0x11, 0x00, // add x2, x2, x1
        0x93, 0x80, 0xf0, 0xff, // addi x1, x1, -1
        0xe3, 0x9c, 0x00, 0xfe, // bne x1, x0, -8
        0x73, 0x00, 0x00, 0x00, // ecall
    };
    try emu2.loadProgram(&loop_program);
    try emu2.run();

    {
        const start = std.time.nanoTimestamp();
        for (0..iterations) |_| {
            var multi_stage = prover.MultiStageProver(F).init(
                allocator,
                &emu2.trace,
                &emu2.ram.trace,
                &emu2.lookup_trace,
                log_k,
                common.constants.RAM_START_ADDRESS,
            );
            defer multi_stage.deinit();

            var transcript = try transcripts.Transcript(F).init(allocator, "bench");
            defer transcript.deinit();

            _ = multi_stage.prove(&transcript) catch continue;
        }
        const end = std.time.nanoTimestamp();

        const elapsed_ns: f64 = @floatFromInt(end - start);
        const ns_per_op = elapsed_ns / @as(f64, @floatFromInt(iterations));
        const ms_per_op = ns_per_op / 1_000_000.0;

        std.debug.print("  Loop (14 steps):    {d:>8.1} ms/op\n", .{ms_per_op});
    }

    std.debug.print("\n", .{});
}

fn benchProofSize() !void {
    std.debug.print("Proof Size:\n", .{});

    const F = zolt.field.BN254Scalar;
    const allocator = std.heap.page_allocator;
    const common = zolt.common;
    const tracer = zolt.tracer;
    const prover = zolt.zkvm.prover;
    const transcripts = zolt.transcripts;

    // Simple program: addi x1, x0, 10; ecall
    const simple_program = [_]u8{
        0x93, 0x00, 0xa0, 0x00, // addi x1, x0, 10
        0x73, 0x00, 0x00, 0x00, // ecall
    };

    const config = common.MemoryConfig{ .program_size = 64 };

    // Run emulation to get trace
    var emu = tracer.Emulator.init(allocator, &config);
    defer emu.deinit();
    emu.max_cycles = 8;
    try emu.loadProgram(&simple_program);
    try emu.run();

    const log_k: usize = 16;

    // Generate proof
    var multi_stage = prover.MultiStageProver(F).init(
        allocator,
        &emu.trace,
        &emu.ram.trace,
        &emu.lookup_trace,
        log_k,
        common.constants.RAM_START_ADDRESS,
    );
    defer multi_stage.deinit();

    var prover_transcript = try transcripts.Transcript(F).init(allocator, "bench");
    defer prover_transcript.deinit();

    const proof = multi_stage.prove(&prover_transcript) catch {
        std.debug.print("  Proof generation failed\n", .{});
        return;
    };

    const size = proof.proofSize();
    const size_bytes = proof.proofSizeBytes();
    const size_kb = @as(f64, @floatFromInt(size_bytes)) / 1024.0;

    std.debug.print("  Simple (2 steps):\n", .{});
    std.debug.print("    Field elements: {d}\n", .{size.total_elements});
    std.debug.print("    Round polys:    {d} (coeffs: {d})\n", .{ size.round_polys, size.poly_coeffs });
    std.debug.print("    Challenges:     {d}\n", .{size.challenges});
    std.debug.print("    Claims:         {d}\n", .{size.claims});
    std.debug.print("    Size:           {d:.2} KB\n", .{size_kb});

    // Now with a loop program
    var emu2 = tracer.Emulator.init(allocator, &config);
    defer emu2.deinit();
    emu2.max_cycles = 32;

    const loop_program = [_]u8{
        0x93, 0x00, 0x40, 0x00, // addi x1, x0, 4
        0x13, 0x01, 0x00, 0x00, // addi x2, x0, 0
        0x33, 0x01, 0x11, 0x00, // add x2, x2, x1
        0x93, 0x80, 0xf0, 0xff, // addi x1, x1, -1
        0xe3, 0x9c, 0x00, 0xfe, // bne x1, x0, -8
        0x73, 0x00, 0x00, 0x00, // ecall
    };
    try emu2.loadProgram(&loop_program);
    try emu2.run();

    var multi_stage2 = prover.MultiStageProver(F).init(
        allocator,
        &emu2.trace,
        &emu2.ram.trace,
        &emu2.lookup_trace,
        log_k,
        common.constants.RAM_START_ADDRESS,
    );
    defer multi_stage2.deinit();

    var prover_transcript2 = try transcripts.Transcript(F).init(allocator, "bench");
    defer prover_transcript2.deinit();

    const proof2 = multi_stage2.prove(&prover_transcript2) catch {
        std.debug.print("  Loop proof generation failed\n", .{});
        return;
    };

    const size2 = proof2.proofSize();
    const size_bytes2 = proof2.proofSizeBytes();
    const size_kb2 = @as(f64, @floatFromInt(size_bytes2)) / 1024.0;

    std.debug.print("  Loop (14 steps):\n", .{});
    std.debug.print("    Field elements: {d}\n", .{size2.total_elements});
    std.debug.print("    Round polys:    {d} (coeffs: {d})\n", .{ size2.round_polys, size2.poly_coeffs });
    std.debug.print("    Challenges:     {d}\n", .{size2.challenges});
    std.debug.print("    Claims:         {d}\n", .{size2.claims});
    std.debug.print("    Size:           {d:.2} KB\n", .{size_kb2});

    std.debug.print("\n", .{});
}

fn benchVerifier() !void {
    std.debug.print("zkVM Verifier:\n", .{});

    const F = zolt.field.BN254Scalar;
    const allocator = std.heap.page_allocator;
    const common = zolt.common;
    const tracer = zolt.tracer;
    const prover = zolt.zkvm.prover;
    const verifier = zolt.zkvm.verifier;
    const transcripts = zolt.transcripts;

    // Simple program: addi x1, x0, 10; ecall
    const program = [_]u8{
        0x93, 0x00, 0xa0, 0x00, // addi x1, x0, 10
        0x73, 0x00, 0x00, 0x00, // ecall
    };

    const config = common.MemoryConfig{ .program_size = 64 };

    // Run emulation to get trace
    var emu = tracer.Emulator.init(allocator, &config);
    defer emu.deinit();
    emu.max_cycles = 8;
    try emu.loadProgram(&program);
    try emu.run();

    const log_k: usize = 16;

    // Generate proof once
    var multi_stage = prover.MultiStageProver(F).init(
        allocator,
        &emu.trace,
        &emu.ram.trace,
        &emu.lookup_trace,
        log_k,
        common.constants.RAM_START_ADDRESS,
    );
    defer multi_stage.deinit();

    var prover_transcript = try transcripts.Transcript(F).init(allocator, "bench");
    defer prover_transcript.deinit();

    const proof = multi_stage.prove(&prover_transcript) catch {
        std.debug.print("  Proof generation failed, skipping verifier benchmark\n", .{});
        return;
    };

    // Benchmark verification (run many times)
    const iterations: usize = 100;
    {
        const start = std.time.nanoTimestamp();
        for (0..iterations) |_| {
            var verify_transcript = try transcripts.Transcript(F).init(allocator, "bench");
            defer verify_transcript.deinit();

            var v = verifier.MultiStageVerifier(F).init(allocator);
            defer v.deinit();

            _ = v.verify(&proof, &verify_transcript) catch continue;
        }
        const end = std.time.nanoTimestamp();

        const elapsed_ns: f64 = @floatFromInt(end - start);
        const ns_per_op = elapsed_ns / @as(f64, @floatFromInt(iterations));
        const us_per_op = ns_per_op / 1000.0;

        std.debug.print("  Verify (2 steps):   {d:>8.1} us/op\n", .{us_per_op});
    }

    // Now with a loop program (more cycles)
    var emu2 = tracer.Emulator.init(allocator, &config);
    defer emu2.deinit();
    emu2.max_cycles = 32;

    const loop_program = [_]u8{
        0x93, 0x00, 0x40, 0x00, // addi x1, x0, 4
        0x13, 0x01, 0x00, 0x00, // addi x2, x0, 0
        0x33, 0x01, 0x11, 0x00, // add x2, x2, x1
        0x93, 0x80, 0xf0, 0xff, // addi x1, x1, -1
        0xe3, 0x9c, 0x00, 0xfe, // bne x1, x0, -8
        0x73, 0x00, 0x00, 0x00, // ecall
    };
    try emu2.loadProgram(&loop_program);
    try emu2.run();

    // Generate proof for loop program
    var multi_stage2 = prover.MultiStageProver(F).init(
        allocator,
        &emu2.trace,
        &emu2.ram.trace,
        &emu2.lookup_trace,
        log_k,
        common.constants.RAM_START_ADDRESS,
    );
    defer multi_stage2.deinit();

    var prover_transcript2 = try transcripts.Transcript(F).init(allocator, "bench");
    defer prover_transcript2.deinit();

    const proof2 = multi_stage2.prove(&prover_transcript2) catch {
        std.debug.print("  Loop proof generation failed, skipping loop verifier benchmark\n", .{});
        return;
    };

    // Benchmark verification for loop program
    {
        const start = std.time.nanoTimestamp();
        for (0..iterations) |_| {
            var verify_transcript = try transcripts.Transcript(F).init(allocator, "bench");
            defer verify_transcript.deinit();

            var v = verifier.MultiStageVerifier(F).init(allocator);
            defer v.deinit();

            _ = v.verify(&proof2, &verify_transcript) catch continue;
        }
        const end = std.time.nanoTimestamp();

        const elapsed_ns: f64 = @floatFromInt(end - start);
        const ns_per_op = elapsed_ns / @as(f64, @floatFromInt(iterations));
        const us_per_op = ns_per_op / 1000.0;

        std.debug.print("  Verify (14 steps):  {d:>8.1} us/op\n", .{us_per_op});
    }

    std.debug.print("\n", .{});
}

test "benchmarks compile" {
    // Just verify it compiles
}
