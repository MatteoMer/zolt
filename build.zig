const std = @import("std");

/// Link Metal frameworks on a module (Apple Silicon only).
fn linkMetalFrameworks(module: *std.Build.Module) void {
    const opts: std.Build.Module.LinkFrameworkOptions = .{};
    module.linkFramework("Metal", opts);
    module.linkFramework("CoreGraphics", opts);
    module.linkFramework("Foundation", opts);
}

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const is_apple_silicon = target.result.os.tag == .macos and
        target.result.cpu.arch == .aarch64;

    // Main library
    const lib = b.addLibrary(.{
        .name = "zolt",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/root.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    if (is_apple_silicon) linkMetalFrameworks(lib.root_module);
    b.installArtifact(lib);

    // Export zolt module for dependency consumption
    _ = b.addModule("zolt", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Main executable (for testing/demo)
    const exe = b.addExecutable(.{
        .name = "zolt",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    if (is_apple_silicon) linkMetalFrameworks(exe.root_module);
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
    if (is_apple_silicon) linkMetalFrameworks(lib_unit_tests.root_module);
    const run_lib_unit_tests = b.addRunArtifact(lib_unit_tests);

    // Unit tests for the executable
    const exe_unit_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    if (is_apple_silicon) linkMetalFrameworks(exe_unit_tests.root_module);
    const run_exe_unit_tests = b.addRunArtifact(exe_unit_tests);

    // Test step
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_lib_unit_tests.step);
    test_step.dependOn(&run_exe_unit_tests.step);

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

    // Benchmark: ThreadPool vs Rayon
    const bench_tp = b.addExecutable(.{
        .name = "bench-tp",
        .root_module = b.createModule(.{
            .root_source_file = b.path("bench/threadpool_vs_rayon/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zolt", .module = lib.root_module },
            },
        }),
    });
    b.installArtifact(bench_tp);
    const run_bench_tp = b.addRunArtifact(bench_tp);
    const bench_tp_step = b.step("bench-tp", "Run ThreadPool micro-benchmark");
    bench_tp_step.dependOn(&run_bench_tp.step);

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

    // Example: HyperKZG Commitment
    const hyperkzg_example = b.addExecutable(.{
        .name = "example-hyperkzg",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/hyperkzg_commitment.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zolt", .module = lib.root_module },
            },
        }),
    });
    const run_hyperkzg_example = b.addRunArtifact(hyperkzg_example);
    const hyperkzg_example_step = b.step("example-hyperkzg", "Run HyperKZG commitment example");
    hyperkzg_example_step.dependOn(&run_hyperkzg_example.step);

    // Example: Sumcheck Protocol
    const sumcheck_example = b.addExecutable(.{
        .name = "example-sumcheck",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/sumcheck_protocol.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zolt", .module = lib.root_module },
            },
        }),
    });
    const run_sumcheck_example = b.addRunArtifact(sumcheck_example);
    const sumcheck_example_step = b.step("example-sumcheck", "Run sumcheck protocol example");
    sumcheck_example_step.dependOn(&run_sumcheck_example.step);

    // Example: Full Pipeline
    const pipeline_example = b.addExecutable(.{
        .name = "example-pipeline",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/full_pipeline.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zolt", .module = lib.root_module },
            },
        }),
    });
    const run_pipeline_example = b.addRunArtifact(pipeline_example);
    const pipeline_example_step = b.step("example-pipeline", "Run full proving pipeline example");
    pipeline_example_step.dependOn(&run_pipeline_example.step);

    // Benchmark: Field Arithmetic
    const field_bench = b.addExecutable(.{
        .name = "field-bench",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/field_bench.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zolt", .module = lib.root_module },
            },
        }),
    });
    const run_field_bench = b.addRunArtifact(field_bench);
    const field_bench_step = b.step("bench-field", "Run field arithmetic benchmark");
    field_bench_step.dependOn(&run_field_bench.step);

    // Benchmark: ARM64 field verification
    const arm64_verify = b.addExecutable(.{
        .name = "arm64-verify",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/arm64_verify.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zolt", .module = lib.root_module },
            },
        }),
    });
    const run_arm64_verify = b.addRunArtifact(arm64_verify);
    const arm64_verify_step = b.step("bench-arm64", "Run ARM64 field verification benchmark");
    arm64_verify_step.dependOn(&run_arm64_verify.step);

    // Benchmark: GPU vs CPU
    const gpu_bench = b.addExecutable(.{
        .name = "gpu-bench",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/gpu_bench.zig"),
            .target = target,
            .optimize = .ReleaseFast,
            .imports = &.{
                .{ .name = "zolt", .module = lib.root_module },
            },
        }),
    });
    const run_gpu_bench = b.addRunArtifact(gpu_bench);
    const gpu_bench_step = b.step("bench-gpu", "Run GPU vs CPU benchmark");
    gpu_bench_step.dependOn(&run_gpu_bench.step);

    // Optional: rebuild Metal shaders from .metal sources
    // Usage: zig build metal-shaders
    const metal_step = b.step("metal-shaders", "Rebuild Metal shader library from .metal sources");
    if (is_apple_silicon) {
        const dev_dir = "/Applications/Xcode.app/Contents/Developer";
        const shader_dir = "src/gpu/shaders";
        const include_flag = b.pathFromRoot(shader_dir);

        const compile_smoke = b.addSystemCommand(&.{ "xcrun", "metal", "-c", "-I" });
        compile_smoke.setEnvironmentVariable("DEVELOPER_DIR", dev_dir);
        compile_smoke.addArg(include_flag);
        compile_smoke.addFileArg(b.path(shader_dir ++ "/smoke.metal"));
        compile_smoke.addArg("-o");
        const smoke_air = compile_smoke.addOutputFileArg("smoke.air");

        const compile_field = b.addSystemCommand(&.{ "xcrun", "metal", "-c", "-I" });
        compile_field.setEnvironmentVariable("DEVELOPER_DIR", dev_dir);
        compile_field.addArg(include_flag);
        compile_field.addFileArg(b.path(shader_dir ++ "/field.metal"));
        compile_field.addArg("-o");
        const field_air = compile_field.addOutputFileArg("field.air");

        const compile_poly = b.addSystemCommand(&.{ "xcrun", "metal", "-c", "-I" });
        compile_poly.setEnvironmentVariable("DEVELOPER_DIR", dev_dir);
        compile_poly.addArg(include_flag);
        compile_poly.addFileArg(b.path(shader_dir ++ "/poly.metal"));
        compile_poly.addArg("-o");
        const poly_air = compile_poly.addOutputFileArg("poly.air");

        const compile_msm = b.addSystemCommand(&.{ "xcrun", "metal", "-c", "-I" });
        compile_msm.setEnvironmentVariable("DEVELOPER_DIR", dev_dir);
        compile_msm.addArg(include_flag);
        compile_msm.addFileArg(b.path(shader_dir ++ "/msm.metal"));
        compile_msm.addArg("-o");
        const msm_air = compile_msm.addOutputFileArg("msm.air");

        const metal_link = b.addSystemCommand(&.{ "xcrun", "metallib" });
        metal_link.setEnvironmentVariable("DEVELOPER_DIR", dev_dir);
        metal_link.addFileArg(smoke_air);
        metal_link.addFileArg(field_air);
        metal_link.addFileArg(poly_air);
        metal_link.addFileArg(msm_air);
        metal_link.addArgs(&.{ "-o", b.pathFromRoot(shader_dir ++ "/shaders.metallib") });

        metal_step.dependOn(&metal_link.step);
    }
}
