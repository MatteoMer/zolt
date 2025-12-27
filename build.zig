const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Main library
    const lib = b.addLibrary(.{
        .name = "zolt",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/root.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    b.installArtifact(lib);

    // Main executable (for testing/demo)
    const exe = b.addExecutable(.{
        .name = "zolt",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    b.installArtifact(exe);

    // Run command
    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }
    const run_step = b.step("run", "Run the zolt executable");
    run_step.dependOn(&run_cmd.step);

    // Unit tests for the library
    const lib_unit_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/root.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    const run_lib_unit_tests = b.addRunArtifact(lib_unit_tests);

    // Unit tests for the executable
    const exe_unit_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    const run_exe_unit_tests = b.addRunArtifact(exe_unit_tests);

    // Test step
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_lib_unit_tests.step);
    test_step.dependOn(&run_exe_unit_tests.step);

    // Benchmark step
    const bench_exe = b.addExecutable(.{
        .name = "zolt-bench",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/bench.zig"),
            .target = target,
            .optimize = .ReleaseFast,
        }),
    });
    const run_bench = b.addRunArtifact(bench_exe);
    const bench_step = b.step("bench", "Run benchmarks");
    bench_step.dependOn(&run_bench.step);

    // Example: Field Arithmetic
    const field_example = b.addExecutable(.{
        .name = "example-field",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/field_arithmetic.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zolt", .module = lib.root_module },
            },
        }),
    });
    const run_field_example = b.addRunArtifact(field_example);
    const field_example_step = b.step("example-field", "Run field arithmetic example");
    field_example_step.dependOn(&run_field_example.step);

    // Example: Simple Proof
    const proof_example = b.addExecutable(.{
        .name = "example-proof",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/simple_proof.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zolt", .module = lib.root_module },
            },
        }),
    });
    const run_proof_example = b.addRunArtifact(proof_example);
    const proof_example_step = b.step("example-proof", "Run simple proof example");
    proof_example_step.dependOn(&run_proof_example.step);

    // Example: RISC-V Emulation
    const riscv_example = b.addExecutable(.{
        .name = "example-riscv",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/risc_v_emulation.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zolt", .module = lib.root_module },
            },
        }),
    });
    const run_riscv_example = b.addRunArtifact(riscv_example);
    const riscv_example_step = b.step("example-riscv", "Run RISC-V emulation example");
    riscv_example_step.dependOn(&run_riscv_example.step);
}
