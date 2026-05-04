const std = @import("std");

/// Link Metal frameworks on a module (Apple Silicon only).
fn linkMetalFrameworks(module: *std.Build.Module) void {
    const opts: std.Build.Module.LinkFrameworkOptions = .{};
    module.linkFramework("Metal", opts);
    module.linkFramework("CoreGraphics", opts);
    module.linkFramework("Foundation", opts);
}

/// Add an executable that imports the `zolt` module and wire a `zig build <step>` alias for it.
/// Used by examples and microbenchmarks — those targets all share the same shape (one imported
/// module, an optional install, and a run step).
fn addZoltExe(
    b: *std.Build,
    opts: struct {
        name: []const u8,
        source: []const u8,
        zolt_mod: *std.Build.Module,
        target: std.Build.ResolvedTarget,
        optimize: std.builtin.OptimizeMode,
        step_name: []const u8,
        step_desc: []const u8,
        install: bool = false,
    },
) void {
    const exe = b.addExecutable(.{
        .name = opts.name,
        .root_module = b.createModule(.{
            .root_source_file = b.path(opts.source),
            .target = opts.target,
            .optimize = opts.optimize,
            .imports = &.{
                .{ .name = "zolt", .module = opts.zolt_mod },
            },
        }),
    });
    if (opts.install) b.installArtifact(exe);
    const run = b.addRunArtifact(exe);
    const step = b.step(opts.step_name, opts.step_desc);
    step.dependOn(&run.step);
}

/// Source paths fed to `zig fmt` for both the `fmt` and `ci` steps.
const fmt_paths = &[_][]const u8{
    "src",
    "packages",
    "examples",
    "bench",
    "build.zig",
};

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
        const wasm_base_exports = [_][]const u8{
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
        // Workers in atomics/SAB mode need their own stack and TLS regions exposed.
        const wasm_atomics_extra = [_][]const u8{
            "__stack_pointer",
            "__tls_base",
            "__tls_size",
            "__tls_align",
            "__wasm_init_tls",
        };
        wasm_mod.root_module.export_symbol_names = if (has_wasm_atomics)
            &(wasm_base_exports ++ wasm_atomics_extra)
        else
            &wasm_base_exports;

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

        // Example & benchmark executables that import the `zolt` module at
        // the workspace's default optimize level.
        addZoltExe(b, .{
            .name = "example-field",
            .source = "examples/field_arithmetic.zig",
            .zolt_mod = lib.root_module,
            .target = target,
            .optimize = optimize,
            .step_name = "example-field",
            .step_desc = "Run field arithmetic example",
        });
        addZoltExe(b, .{
            .name = "example-proof",
            .source = "examples/simple_proof.zig",
            .zolt_mod = lib.root_module,
            .target = target,
            .optimize = optimize,
            .step_name = "example-proof",
            .step_desc = "Run simple proof example",
        });
        addZoltExe(b, .{
            .name = "bench-tp",
            .source = "bench/threadpool_vs_rayon/main.zig",
            .zolt_mod = lib.root_module,
            .target = target,
            .optimize = optimize,
            .step_name = "bench-tp",
            .step_desc = "Run ThreadPool micro-benchmark",
            .install = true,
        });
        addZoltExe(b, .{
            .name = "bench-scaling",
            .source = "bench/threadpool_vs_rayon/bench_scaling.zig",
            .zolt_mod = lib.root_module,
            .target = target,
            .optimize = optimize,
            .step_name = "bench-scaling",
            .step_desc = "Run scaling micro-benchmark (parallelFor, dispatch, bind)",
            .install = true,
        });
        addZoltExe(b, .{
            .name = "bench-msm",
            .source = "bench/msm/main.zig",
            .zolt_mod = lib.root_module,
            .target = target,
            .optimize = optimize,
            .step_name = "bench-msm",
            .step_desc = "Run MSM benchmark (G1/G2 Pippenger)",
            .install = true,
        });
        addZoltExe(b, .{
            .name = "example-riscv",
            .source = "examples/risc_v_emulation.zig",
            .zolt_mod = lib.root_module,
            .target = target,
            .optimize = optimize,
            .step_name = "example-riscv",
            .step_desc = "Run RISC-V emulation example",
        });
        addZoltExe(b, .{
            .name = "example-hyperkzg",
            .source = "examples/hyperkzg_commitment.zig",
            .zolt_mod = lib.root_module,
            .target = target,
            .optimize = optimize,
            .step_name = "example-hyperkzg",
            .step_desc = "Run HyperKZG commitment example",
        });
        addZoltExe(b, .{
            .name = "example-sumcheck",
            .source = "examples/sumcheck_protocol.zig",
            .zolt_mod = lib.root_module,
            .target = target,
            .optimize = optimize,
            .step_name = "example-sumcheck",
            .step_desc = "Run sumcheck protocol example",
        });
        addZoltExe(b, .{
            .name = "example-pipeline",
            .source = "examples/full_pipeline.zig",
            .zolt_mod = lib.root_module,
            .target = target,
            .optimize = optimize,
            .step_name = "example-pipeline",
            .step_desc = "Run full proving pipeline example",
        });
        addZoltExe(b, .{
            .name = "field-bench",
            .source = "examples/field_bench.zig",
            .zolt_mod = lib.root_module,
            .target = target,
            .optimize = optimize,
            .step_name = "bench-field",
            .step_desc = "Run field arithmetic benchmark",
        });

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

        addZoltExe(b, .{
            .name = "zolt-arith-field-micro",
            .source = "bench/zolt_arith/field_micro.zig",
            .zolt_mod = zolt_mod_release,
            .target = target,
            .optimize = .ReleaseFast,
            .step_name = "bench-zolt-arith-field",
            .step_desc = "Run zolt-arith field microbench",
        });
        addZoltExe(b, .{
            .name = "zolt-arith-pairing-micro",
            .source = "bench/zolt_arith/pairing_micro.zig",
            .zolt_mod = zolt_mod_release,
            .target = target,
            .optimize = .ReleaseFast,
            .step_name = "bench-zolt-arith-pairing",
            .step_desc = "Run zolt-arith pairing microbench",
        });
        addZoltExe(b, .{
            .name = "arm64-verify",
            .source = "examples/arm64_verify.zig",
            .zolt_mod = lib.root_module,
            .target = target,
            .optimize = optimize,
            .step_name = "bench-arm64",
            .step_desc = "Run ARM64 field verification benchmark",
        });
        addZoltExe(b, .{
            .name = "gpu-bench",
            .source = "examples/gpu_bench.zig",
            .zolt_mod = lib.root_module,
            .target = target,
            .optimize = .ReleaseFast,
            .step_name = "bench-gpu",
            .step_desc = "Run GPU vs CPU benchmark",
        });

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
            .paths = fmt_paths,
            .check = true,
        });
        ci_step.dependOn(&fmt_check.step);
    } // end !is_wasm

    // zig build fmt — run zig fmt on all project sources (works for any target)
    const fmt_step = b.step("fmt", "Format all Zig source files");
    const fmt = b.addFmt(.{ .paths = fmt_paths });
    fmt_step.dependOn(&fmt.step);
}
