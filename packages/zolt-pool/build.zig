const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // When targeting WASM with atomics, compile as multi-threaded so
    // `threadlocal` uses proper per-instance TLS via __tls_base, instead
    // of compiling to a fixed address in shared memory.
    const is_wasm = target.result.cpu.arch == .wasm32 or target.result.cpu.arch == .wasm64;
    const has_atomics = is_wasm and
        std.Target.wasm.featureSetHas(target.result.cpu.features, .atomics);

    // Export the zolt_pool module
    _ = b.addModule("zolt_pool", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
        .single_threaded = if (has_atomics) false else null,
    });

    // Tests
    const tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/root.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    const run_tests = b.addRunArtifact(tests);
    const test_step = b.step("test", "Run zolt-pool tests");
    test_step.dependOn(&run_tests.step);
}
