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

    // Get zolt_pool dependency
    const zolt_pool_dep = b.dependency("zolt_pool", .{
        .target = target,
        .optimize = optimize,
    });
    const zolt_pool_mod = zolt_pool_dep.module("zolt_pool");

    // Export the zolt_arith module
    const arith_mod = b.addModule("zolt_arith", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "zolt_pool", .module = zolt_pool_mod },
        },
    });
    if (is_apple_silicon) linkMetalFrameworks(arith_mod);

    // Tests
    const tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/root.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zolt_pool", .module = zolt_pool_mod },
            },
        }),
    });
    if (is_apple_silicon) linkMetalFrameworks(tests.root_module);
    const run_tests = b.addRunArtifact(tests);
    const test_step = b.step("test", "Run zolt-arith tests");
    test_step.dependOn(&run_tests.step);
}
