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
    struct {
        len: usize = 0,
        pub fn initFromCpu(_: anytype, _: anytype) error{GpuUnavailable}!@This() {
            return error.GpuUnavailable;
        }
        pub fn deinit(_: *@This()) void {}
        pub fn metalBuffer(_: @This()) *anyopaque {
            unreachable;
        }
        pub fn syncToCpu(_: *@This(), _: anytype) void {}
        pub fn readAll(_: @This(), _: anytype) void {}
    };

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

    // Stub device for gpu.device access paths (unreachable on non-Metal)
    const StubDevice = struct { device: *anyopaque = undefined };
    gpu: StubDevice = .{},

    pub fn init(_: anytype) Error!@This() {
        return error.GpuUnavailable;
    }
    pub fn deinit(_: *@This()) void {}

    const BN254Scalar = @import("../field/mod.zig").BN254Scalar;
    const AffinePoint = @import("../msm/mod.zig").AffinePoint;
    const BN254BaseField = @import("../field/mod.zig").BN254BaseField;

    // Stubs for GpuPolyOps methods
    pub fn polyBindLow(_: *@This(), _: []const BN254Scalar, _: BN254Scalar, _: []BN254Scalar) Error!void {
        return error.GpuUnavailable;
    }
    pub fn productSumcheckRoundGpu(_: *@This(), _: *anyopaque, _: *anyopaque, _: u32, _: []const BN254Scalar, _: []const BN254Scalar, _: u32) Error![2]BN254Scalar {
        return error.GpuUnavailable;
    }

    pub fn bindLowInPlace(_: *@This(), _: anytype, _: BN254Scalar) Error!void {
        return error.GpuUnavailable;
    }

    // Stubs for GpuMsmOps methods
    pub fn computeRowCommitmentsI128(_: *@This(), _: []const AffinePoint(BN254BaseField), _: []const i128, _: usize, _: usize, _: []AffinePoint(BN254BaseField)) Error!void {
        return error.GpuUnavailable;
    }
    pub fn computeSingleMsm(_: *@This(), _: []const AffinePoint(BN254BaseField), _: []const BN254Scalar, _: std.mem.Allocator) Error!AffinePoint(BN254BaseField) {
        return error.GpuUnavailable;
    }
};

// ── Tests ──────────────────────────────────────────────────────────────────────

test {
    if (comptime is_metal_available) {
        _ = @import("field_ops.zig");
        _ = @import("poly_ops.zig");
        _ = @import("msm_ops.zig");
    }
}

// GPU smoke test: only available on Metal (macOS aarch64)
// Moved to examples/gpu_bench.zig to avoid libc dependency on Linux builds.
