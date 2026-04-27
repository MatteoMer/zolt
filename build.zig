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

    // Opt-in build of the Rust Jolt verifier staticlib. Off by default so most
    // CI jobs (and quick `zig build` / `zig build test` runs) skip the ~2 min
    // cargo compile. Enable with `zig build -Dverify=true` to get the
    // `zolt verify` command and its extern link to libjolt_verifier.a.
    const enable_verifier = b.option(
        bool,
        "verify",
        "Build and link the Rust Jolt verifier (enables `zolt verify`)",
    ) orelse false;

    const build_options = b.addOptions();
    build_options.addOption(bool, "enable_verifier", enable_verifier);
    const build_options_mod = build_options.createModule();

    const is_apple_silicon = target.result.os.tag == .macos and
        target.result.cpu.arch == .aarch64;

    const is_wasm = target.result.cpu.arch == .wasm32 or
        target.result.cpu.arch == .wasm64;

    // Detect WASM atomics — enables multi-threaded mode for Web Workers
    const has_wasm_atomics = is_wasm and
        std.Target.wasm.featureSetHas(target.result.cpu.features, .atomics);

    // Package dependencies
    const zolt_pool_dep = b.dependency("zolt_pool", .{
        .target = target,
        .optimize = optimize,
    });
    const zolt_pool_mod = zolt_pool_dep.module("zolt_pool");

    const zolt_arith_dep = b.dependency("zolt_arith", .{
        .target = target,
        .optimize = optimize,
    });
    const zolt_arith_mod = zolt_arith_dep.module("zolt_arith");

    // Export zolt module for dependency consumption
    _ = b.addModule("zolt", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "zolt_pool", .module = zolt_pool_mod },
            .{ .name = "zolt_arith", .module = zolt_arith_mod },
        },
    });

    // ── WASM executable target (browser-loadable .wasm module) ─────────
    if (is_wasm) {
        const wasm_mod = b.addExecutable(.{
            .name = "zolt_capi",
            .root_module = b.createModule(.{
                .root_source_file = b.path("src/c_api.zig"),
                .target = target,
                .optimize = optimize,
                .imports = &.{
                    .{ .name = "zolt_pool", .module = zolt_pool_mod },
                    .{ .name = "zolt_arith", .module = zolt_arith_mod },
                },
            }),
        });
        wasm_mod.entry = .disabled;
        wasm_mod.initial_memory = 256 * 1024 * 1024; // 256 MB
        wasm_mod.max_memory = if (target.result.cpu.arch == .wasm64)
            16 * 1024 * 1024 * 1024 // 16 GB (wasm64, wasm-ld max)
        else
            4 * 1024 * 1024 * 1024; // 4 GB (wasm32 max)
        wasm_mod.root_module.export_symbol_names = if (has_wasm_atomics)
            &.{
                "zolt_alloc",
                "zolt_free",
                "zolt_thread_pool_create",
                "zolt_thread_pool_destroy",
                "zolt_load_elf",
                "zolt_load_elf_bytes",
                "zolt_loaded_elf_size",
                "zolt_loaded_elf_destroy",
                "zolt_prove",
                "zolt_proof_result_size",
                "zolt_proof_result_ptr",
                "zolt_proof_result_destroy",
                "zolt_thread_pool_create_wasm",
                "zolt_thread_pool_ptr",
                "zolt_worker_entry",
                // Workers need their own stack and TLS regions
                "__stack_pointer",
                "__tls_base",
                "__tls_size",
                "__tls_align",
                "__wasm_init_tls",
            }
        else
            &.{
                "zolt_alloc",
                "zolt_free",
                "zolt_thread_pool_create",
                "zolt_thread_pool_destroy",
                "zolt_load_elf",
                "zolt_load_elf_bytes",
                "zolt_loaded_elf_size",
                "zolt_loaded_elf_destroy",
                "zolt_prove",
                "zolt_proof_result_size",
                "zolt_proof_result_ptr",
                "zolt_proof_result_destroy",
                "zolt_thread_pool_create_wasm",
                "zolt_thread_pool_ptr",
                "zolt_worker_entry",
            };

        if (has_wasm_atomics) {
            // Enable shared memory for Web Workers + SharedArrayBuffer
            wasm_mod.shared_memory = true;
            wasm_mod.import_memory = true; // JS creates SharedArrayBuffer
            wasm_mod.export_memory = true; // re-export so JS can access via instance.exports.memory
        } else {
            wasm_mod.export_memory = true;
        }
        b.installArtifact(wasm_mod);
    }

    // ── Native-only targets (libraries, executables, tests, benchmarks) ──
    // These require OS threads, filesystem, and can't run on WASM.
    if (!is_wasm) {
        // Main library
        const lib = b.addLibrary(.{
            .name = "zolt",
            .root_module = b.createModule(.{
                .root_source_file = b.path("src/root.zig"),
                .target = target,
                .optimize = optimize,
                .imports = &.{
                    .{ .name = "zolt_pool", .module = zolt_pool_mod },
                    .{ .name = "zolt_arith", .module = zolt_arith_mod },
                },
            }),
        });
        if (is_apple_silicon) linkMetalFrameworks(lib.root_module);
        b.installArtifact(lib);

        // C-API static library (for FFI from Rust/C)
        const capi_lib = b.addLibrary(.{
            .name = "zolt_capi",
            .root_module = b.createModule(.{
                .root_source_file = b.path("src/c_api.zig"),
                .target = target,
                .optimize = optimize,
                .imports = &.{
                    .{ .name = "zolt_pool", .module = zolt_pool_mod },
                    .{ .name = "zolt_arith", .module = zolt_arith_mod },
                },
            }),
        });
        if (is_apple_silicon) linkMetalFrameworks(capi_lib.root_module);
        b.installArtifact(capi_lib);

        // Main executable (for testing/demo)
        const exe = b.addExecutable(.{
            .name = "zolt",
            .root_module = b.createModule(.{
                .root_source_file = b.path("src/main.zig"),
                .target = target,
                .optimize = optimize,
                .imports = &.{
                    .{ .name = "zolt_pool", .module = zolt_pool_mod },
                    .{ .name = "zolt_arith", .module = zolt_arith_mod },
                    .{ .name = "build_options", .module = build_options_mod },
                },
            }),
        });
        if (is_apple_silicon) linkMetalFrameworks(exe.root_module);

        // Build jolt-verifier Rust staticlib once; link it into every Zig
        // compile that needs `extern fn jolt_verify` (the main exe and the
        // exe unit tests, which share src/main.zig as their root).
        // Only created when `-Dverify=true`; otherwise the verify command
        // compiles to a stub and nothing needs libjolt_verifier.a.
        const cargo_build: ?*std.Build.Step.Run = if (enable_verifier) blk: {
            const cmd = b.addSystemCommand(&.{
                "cargo",
                "build",
                "--profile",
                "release-staticlib",
                "--manifest-path",
            });
            cmd.addFileArg(b.path("jolt-verifier/Cargo.toml"));
            break :blk cmd;
        } else null;

        const linkJoltVerifier = struct {
            fn call(
                c: *std.Build.Step.Compile,
                b_: *std.Build,
                tgt: std.Build.ResolvedTarget,
                apple_silicon: bool,
                cargo_step: *std.Build.Step,
            ) void {
                c.root_module.addLibraryPath(.{ .cwd_relative = b_.pathFromRoot("jolt-verifier/target/release-staticlib") });
                c.root_module.linkSystemLibrary("jolt_verifier", .{ .preferred_link_mode = .static });
                c.root_module.linkSystemLibrary("c", .{});
                c.root_module.linkSystemLibrary("m", .{});
                if (tgt.result.os.tag == .linux) {
                    c.root_module.linkSystemLibrary("pthread", .{});
                    c.root_module.linkSystemLibrary("dl", .{});
                    c.root_module.linkSystemLibrary("rt", .{});
                    // Rust's std pulls in panic-unwind symbols (_Unwind_*) that
                    // live in libgcc_s on GNU/Linux; without this, linking a
                    // Rust staticlib on Linux fails with undefined references.
                    c.root_module.linkSystemLibrary("gcc_s", .{});
                } else if (tgt.result.os.tag != .macos) {
                    c.root_module.linkSystemLibrary("pthread", .{});
                }
                if (apple_silicon or tgt.result.os.tag == .macos) {
                    const fw_opts: std.Build.Module.LinkFrameworkOptions = .{};
                    c.root_module.linkFramework("Security", fw_opts);
                    c.root_module.linkFramework("CoreFoundation", fw_opts);
                }
                c.step.dependOn(cargo_step);
            }
        }.call;

        if (cargo_build) |cb| linkJoltVerifier(exe, b, target, is_apple_silicon, &cb.step);

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
                .imports = &.{
                    .{ .name = "zolt_pool", .module = zolt_pool_mod },
                    .{ .name = "zolt_arith", .module = zolt_arith_mod },
                },
            }),
        });
        if (is_apple_silicon) linkMetalFrameworks(lib_unit_tests.root_module);
        const run_lib_unit_tests = b.addRunArtifact(lib_unit_tests);

        // Unit tests for the executable (links jolt-verifier only when -Dverify=true)
        const exe_unit_tests = b.addTest(.{
            .root_module = b.createModule(.{
                .root_source_file = b.path("src/main.zig"),
                .target = target,
                .optimize = optimize,
                .imports = &.{
                    .{ .name = "zolt_pool", .module = zolt_pool_mod },
                    .{ .name = "zolt_arith", .module = zolt_arith_mod },
                    .{ .name = "build_options", .module = build_options_mod },
                },
            }),
        });
        if (is_apple_silicon) linkMetalFrameworks(exe_unit_tests.root_module);
        if (cargo_build) |cb| linkJoltVerifier(exe_unit_tests, b, target, is_apple_silicon, &cb.step);
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

        // Benchmark: Scaling (parallelFor, repeated dispatch, multi-array bind)
        const bench_scaling = b.addExecutable(.{
            .name = "bench-scaling",
            .root_module = b.createModule(.{
                .root_source_file = b.path("bench/threadpool_vs_rayon/bench_scaling.zig"),
                .target = target,
                .optimize = optimize,
                .imports = &.{
                    .{ .name = "zolt", .module = lib.root_module },
                },
            }),
        });
        b.installArtifact(bench_scaling);
        const run_bench_scaling = b.addRunArtifact(bench_scaling);
        const bench_scaling_step = b.step("bench-scaling", "Run scaling micro-benchmark (parallelFor, dispatch, bind)");
        bench_scaling_step.dependOn(&run_bench_scaling.step);

        // Benchmark: MSM (G1/G2 multi-scalar multiplication)
        const bench_msm = b.addExecutable(.{
            .name = "bench-msm",
            .root_module = b.createModule(.{
                .root_source_file = b.path("bench/msm/main.zig"),
                .target = target,
                .optimize = optimize,
                .imports = &.{
                    .{ .name = "zolt", .module = lib.root_module },
                },
            }),
        });
        b.installArtifact(bench_msm);
        const run_bench_msm = b.addRunArtifact(bench_msm);
        const bench_msm_step = b.step("bench-msm", "Run MSM benchmark (G1/G2 Pippenger)");
        bench_msm_step.dependOn(&run_bench_msm.step);

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

        // Release-optimized dep chain for benchmarks (so zolt-arith gets
        // compiled at ReleaseFast instead of Debug, enabling LLVM intrinsics).
        const zolt_pool_dep_release = b.dependency("zolt_pool", .{
            .target = target,
            .optimize = .ReleaseFast,
        });
        const zolt_arith_dep_release = b.dependency("zolt_arith", .{
            .target = target,
            .optimize = .ReleaseFast,
        });
        const zolt_mod_release = b.createModule(.{
            .root_source_file = b.path("src/root.zig"),
            .target = target,
            .optimize = .ReleaseFast,
            .imports = &.{
                .{ .name = "zolt_pool", .module = zolt_pool_dep_release.module("zolt_pool") },
                .{ .name = "zolt_arith", .module = zolt_arith_dep_release.module("zolt_arith") },
            },
        });
        if (is_apple_silicon) linkMetalFrameworks(zolt_mod_release);

        // Benchmark: zolt-arith field microbench (repo-level, optional)
        const zolt_arith_field_micro = b.addExecutable(.{
            .name = "zolt-arith-field-micro",
            .root_module = b.createModule(.{
                .root_source_file = b.path("bench/zolt_arith/field_micro.zig"),
                .target = target,
                .optimize = .ReleaseFast,
                .imports = &.{
                    .{ .name = "zolt", .module = zolt_mod_release },
                },
            }),
        });
        const run_zolt_arith_field_micro = b.addRunArtifact(zolt_arith_field_micro);
        const zolt_arith_field_micro_step = b.step("bench-zolt-arith-field", "Run zolt-arith field microbench");
        zolt_arith_field_micro_step.dependOn(&run_zolt_arith_field_micro.step);

        // Benchmark: zolt-arith pairing microbench (repo-level, optional)
        const zolt_arith_pairing_micro = b.addExecutable(.{
            .name = "zolt-arith-pairing-micro",
            .root_module = b.createModule(.{
                .root_source_file = b.path("bench/zolt_arith/pairing_micro.zig"),
                .target = target,
                .optimize = .ReleaseFast,
                .imports = &.{
                    .{ .name = "zolt", .module = zolt_mod_release },
                },
            }),
        });
        const run_zolt_arith_pairing_micro = b.addRunArtifact(zolt_arith_pairing_micro);
        const zolt_arith_pairing_micro_step = b.step("bench-zolt-arith-pairing", "Run zolt-arith pairing microbench");
        zolt_arith_pairing_micro_step.dependOn(&run_zolt_arith_pairing_micro.step);

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

        // Optional: differential fixture generation outside the package
        const gen_zolt_arith_diff = b.addSystemCommand(&.{
            "cargo",
            "run",
            "--release",
            "--manifest-path",
            "tools/zolt-arith-diff/arkworks-fixtures/Cargo.toml",
            "--",
            "--out-dir",
            "testdata/zolt-arith-diff",
        });
        const gen_zolt_arith_diff_step = b.step("gen-zolt-arith-diff-fixtures", "Generate optional zolt-arith differential fixtures via arkworks");
        gen_zolt_arith_diff_step.dependOn(&gen_zolt_arith_diff.step);

        // Optional: differential fixture verification outside the package
        const zolt_arith_diff_options = b.addOptions();
        zolt_arith_diff_options.addOption([]const u8, "fixtures_root", b.pathFromRoot("testdata/zolt-arith-diff"));

        const zolt_arith_diff_tests = b.addTest(.{
            .root_module = b.createModule(.{
                .root_source_file = b.path("tools/zolt-arith-diff/check.zig"),
                .target = target,
                .optimize = optimize,
                .imports = &.{
                    .{ .name = "zolt", .module = lib.root_module },
                    .{ .name = "diff_config", .module = zolt_arith_diff_options.createModule() },
                },
            }),
        });
        if (is_apple_silicon) linkMetalFrameworks(zolt_arith_diff_tests.root_module);
        const run_zolt_arith_diff_tests = b.addRunArtifact(zolt_arith_diff_tests);
        const zolt_arith_diff_step = b.step("test-zolt-arith-diff", "Run optional zolt-arith differential fixtures");
        zolt_arith_diff_step.dependOn(&run_zolt_arith_diff_tests.step);

        // Optional: rebuild Metal shaders from .metal sources
        // Usage: zig build metal-shaders
        const metal_step = b.step("metal-shaders", "Rebuild Metal shader library from .metal sources");
        if (is_apple_silicon) {
            const dev_dir = "/Applications/Xcode.app/Contents/Developer";
            const shader_dir = "packages/zolt-arith/src/gpu/shaders";
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

        // zig build ci — run all checks that CI enforces
        const ci_step = b.step("ci", "Run all CI checks (test, fmt check, release build)");
        ci_step.dependOn(&run_lib_unit_tests.step);
        ci_step.dependOn(&run_exe_unit_tests.step);

        const fmt_check = b.addFmt(.{
            .paths = &.{
                "src",
                "packages",
                "examples",
                "bench",
                "build.zig",
            },
            .check = true,
        });
        ci_step.dependOn(&fmt_check.step);
    } // end !is_wasm

    // zig build fmt — run zig fmt on all project sources (works for any target)
    const fmt_step = b.step("fmt", "Format all Zig source files");
    const fmt = b.addFmt(.{
        .paths = &.{
            "src",
            "packages",
            "examples",
            "bench",
            "build.zig",
        },
    });
    fmt_step.dependOn(&fmt.step);
}
