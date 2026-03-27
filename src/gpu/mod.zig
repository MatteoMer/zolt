//! Metal GPU compute acceleration for Apple Silicon.
//!
//! On macOS + aarch64 (Apple Silicon), this module provides Metal compute
//! dispatch for data-parallel field operations. On all other targets, it
//! compiles to no-op stubs — the CPU code path runs unchanged.

const std = @import("std");
const builtin = @import("builtin");

/// True when targeting Apple Silicon macOS (the only platform with Metal compute).
pub const is_metal_available = builtin.os.tag == .macos and builtin.cpu.arch == .aarch64;

pub const GpuAccelerator = if (is_metal_available)
    @import("device.zig").GpuAccelerator
else
    NoGpu;

pub const GpuBuffer = if (is_metal_available)
    @import("buffer.zig").GpuBuffer
else
    NoGpuBuffer;

pub const GpuFieldOps = if (is_metal_available)
    @import("field_ops.zig").GpuFieldOps
else
    NoGpuFieldOps;

pub const GpuPolyOps = if (is_metal_available)
    @import("poly_ops.zig").GpuPolyOps
else
    NoGpuFieldOps; // same stub pattern

pub const GpuPolynomial = if (is_metal_available)
    @import("poly_ops.zig").GpuPolynomial
else
    struct {};

pub const GpuMsmOps = if (is_metal_available)
    @import("msm_ops.zig").GpuMsmOps
else
    NoGpuFieldOps; // same stub pattern

// ── Stubs for non-Apple targets ────────────────────────────────────────────────

const NoGpu = struct {
    pub const Error = error{GpuUnavailable};

    pub fn init(_: std.mem.Allocator) Error!@This() {
        return error.GpuUnavailable;
    }
    pub fn deinit(_: *@This()) void {}
    pub fn isAvailable() bool {
        return false;
    }
};

fn NoGpuBuffer(comptime T: type) type {
    _ = T;
    return struct {};
}

const NoGpuFieldOps = struct {
    pub const Error = error{GpuUnavailable};
    pub fn init(_: anytype) Error!@This() {
        return error.GpuUnavailable;
    }
    pub fn deinit(_: *@This()) void {}
};

// ── Tests ──────────────────────────────────────────────────────────────────────

test {
    if (comptime is_metal_available) {
        _ = @import("field_ops.zig");
        _ = @import("poly_ops.zig");
        _ = @import("msm_ops.zig");
    }
}

test "GPU smoke test: vector add" {
    if (comptime !is_metal_available) return;

    const device_mod = @import("device.zig");
    const buffer_mod = @import("buffer.zig");
    const objc = @import("objc.zig");

    var gpu = device_mod.GpuAccelerator.init(std.testing.allocator) catch |err| {
        if (err == error.MetalNotAvailable) return; // No GPU at runtime (e.g. VM)
        return err;
    };
    defer gpu.deinit();

    const n: usize = 1024;
    const Buf = buffer_mod.GpuBuffer(u32);

    var buf_a = try Buf.init(gpu.device, n);
    defer buf_a.deinit();
    var buf_b = try Buf.init(gpu.device, n);
    defer buf_b.deinit();
    var buf_c = try Buf.init(gpu.device, n);
    defer buf_c.deinit();

    // Fill inputs on CPU (writes directly to GPU-visible unified memory)
    for (0..n) |i| {
        buf_a.slice()[i] = @intCast(i);
        buf_b.slice()[i] = @intCast(i * 2);
    }

    // Dispatch vector_add kernel on GPU
    const pipeline = try gpu.createPipeline("vector_add");
    defer objc.msgSendVoid(pipeline, "release");

    try gpu.dispatchSimple(
        pipeline,
        &.{ buf_a.metal_buffer, buf_b.metal_buffer, buf_c.metal_buffer },
        n,
    );

    // Verify results on CPU (reads directly from GPU-written unified memory)
    for (0..n) |i| {
        const expected: u32 = @intCast(i + i * 2);
        try std.testing.expectEqual(expected, buf_c.slice()[i]);
    }
}
