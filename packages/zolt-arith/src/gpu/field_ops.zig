//! GPU-accelerated BN254 field operations.
//!
//! Dispatches batch field arithmetic (mul, add, sub, neg) to Metal compute
//! kernels. Elements are converted from CPU 4×u64 to GPU 8×u32 limb format
//! at dispatch boundaries — the Montgomery form is unchanged.

const std = @import("std");
const device_mod = @import("device.zig");
const buffer_mod = @import("buffer.zig");
const objc = @import("objc.zig");
const field = @import("../field/mod.zig");

const GpuAccelerator = device_mod.GpuAccelerator;
const GpuBuffer = buffer_mod.GpuBuffer;
const BN254Scalar = field.BN254Scalar;
const id = objc.id;

pub const GpuFieldOps = struct {
    gpu: *GpuAccelerator,
    mul_pipeline: id,
    add_pipeline: id,
    sub_pipeline: id,
    neg_pipeline: id,

    pub const Error = GpuAccelerator.Error || GpuBuffer(u32).Error;

    pub fn init(gpu: *GpuAccelerator) GpuAccelerator.Error!GpuFieldOps {
        const mul_p = try gpu.createPipeline("field_mul_batch");
        errdefer objc.msgSendVoid(mul_p, "release");
        const add_p = try gpu.createPipeline("field_add_batch");
        errdefer objc.msgSendVoid(add_p, "release");
        const sub_p = try gpu.createPipeline("field_sub_batch");
        errdefer objc.msgSendVoid(sub_p, "release");
        const neg_p = try gpu.createPipeline("field_neg_batch");

        return .{
            .gpu = gpu,
            .mul_pipeline = mul_p,
            .add_pipeline = add_p,
            .sub_pipeline = sub_p,
            .neg_pipeline = neg_p,
        };
    }

    pub fn deinit(self: *GpuFieldOps) void {
        objc.msgSendVoid(self.neg_pipeline, "release");
        objc.msgSendVoid(self.sub_pipeline, "release");
        objc.msgSendVoid(self.add_pipeline, "release");
        objc.msgSendVoid(self.mul_pipeline, "release");
    }

    /// out[i] = a[i] * b[i] (Montgomery multiplication on GPU)
    pub fn fieldMulBatch(self: *GpuFieldOps, a: []const BN254Scalar, b: []const BN254Scalar, out: []BN254Scalar) Error!void {
        try self.dispatchBinary(self.mul_pipeline, a, b, out);
    }

    /// out[i] = a[i] + b[i] (modular addition on GPU)
    pub fn fieldAddBatch(self: *GpuFieldOps, a: []const BN254Scalar, b: []const BN254Scalar, out: []BN254Scalar) Error!void {
        try self.dispatchBinary(self.add_pipeline, a, b, out);
    }

    /// out[i] = a[i] - b[i] (modular subtraction on GPU)
    pub fn fieldSubBatch(self: *GpuFieldOps, a: []const BN254Scalar, b: []const BN254Scalar, out: []BN254Scalar) Error!void {
        try self.dispatchBinary(self.sub_pipeline, a, b, out);
    }

    /// out[i] = -a[i] (modular negation on GPU)
    pub fn fieldNegBatch(self: *GpuFieldOps, a: []const BN254Scalar, out: []BN254Scalar) Error!void {
        const n = a.len;
        std.debug.assert(out.len == n);
        if (n == 0) return;

        var buf_a = try GpuBuffer(u32).init(self.gpu.device, n * 8);
        defer buf_a.deinit();
        var buf_out = try GpuBuffer(u32).init(self.gpu.device, n * 8);
        defer buf_out.deinit();

        writeFieldElements(buf_a.slice(), a);

        try self.gpu.dispatchSimple(
            self.neg_pipeline,
            &.{ buf_a.metal_buffer, buf_out.metal_buffer },
            n,
        );

        readFieldElements(buf_out.slice(), out);
    }

    // ── Internal ────────────────────────────────────────────────────────────

    fn dispatchBinary(self: *GpuFieldOps, pipeline: id, a: []const BN254Scalar, b: []const BN254Scalar, out: []BN254Scalar) Error!void {
        const n = a.len;
        std.debug.assert(b.len == n);
        std.debug.assert(out.len == n);
        if (n == 0) return;

        var buf_a = try GpuBuffer(u32).init(self.gpu.device, n * 8);
        defer buf_a.deinit();
        var buf_b = try GpuBuffer(u32).init(self.gpu.device, n * 8);
        defer buf_b.deinit();
        var buf_c = try GpuBuffer(u32).init(self.gpu.device, n * 8);
        defer buf_c.deinit();

        writeFieldElements(buf_a.slice(), a);
        writeFieldElements(buf_b.slice(), b);

        try self.gpu.dispatchSimple(
            pipeline,
            &.{ buf_a.metal_buffer, buf_b.metal_buffer, buf_c.metal_buffer },
            n,
        );

        readFieldElements(buf_c.slice(), out);
    }
};

// ── Format conversion: 4×u64 (CPU) ↔ 8×u32 (GPU) ──────────────────────────
//
// Montgomery form is preserved — this is pure bit splitting.
// Little-endian: gpu[2i] = lo32(cpu[i]), gpu[2i+1] = hi32(cpu[i])

fn writeFieldElements(buf: []u32, elems: []const BN254Scalar) void {
    for (elems, 0..) |elem, idx| {
        const base = idx * 8;
        inline for (0..4) |i| {
            buf[base + i * 2] = @truncate(elem.limbs[i]);
            buf[base + i * 2 + 1] = @truncate(elem.limbs[i] >> 32);
        }
    }
}

fn readFieldElements(buf: []const u32, out: []BN254Scalar) void {
    for (out, 0..) |*elem, idx| {
        const base = idx * 8;
        inline for (0..4) |i| {
            elem.limbs[i] = @as(u64, buf[base + i * 2]) |
                (@as(u64, buf[base + i * 2 + 1]) << 32);
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

const testing = std.testing;

test "GPU field mul: correctness" {
    const gpu_mod = @import("mod.zig");
    if (comptime !gpu_mod.is_metal_available) return;

    var gpu = device_mod.GpuAccelerator.init(testing.allocator) catch |err| {
        if (err == error.MetalNotAvailable) return;
        return err;
    };
    defer gpu.deinit();

    var ops = try GpuFieldOps.init(&gpu);
    defer ops.deinit();

    // Test values: small numbers, edge cases
    const a_vals = [_]BN254Scalar{
        BN254Scalar.fromU64(2),
        BN254Scalar.fromU64(3),
        BN254Scalar.fromU64(0),
        BN254Scalar.fromU64(1),
        BN254Scalar.fromU64(0xFFFFFFFF),
        BN254Scalar.fromU64(0xDEADBEEF),
        BN254Scalar.one(), // R mod p (Montgomery identity)
        BN254Scalar.zero(),
    };
    const b_vals = [_]BN254Scalar{
        BN254Scalar.fromU64(3),
        BN254Scalar.fromU64(7),
        BN254Scalar.fromU64(5),
        BN254Scalar.fromU64(1),
        BN254Scalar.fromU64(0xFFFFFFFF),
        BN254Scalar.fromU64(42),
        BN254Scalar.fromU64(123456789),
        BN254Scalar.fromU64(999),
    };

    var gpu_out: [8]BN254Scalar = undefined;
    try ops.fieldMulBatch(&a_vals, &b_vals, &gpu_out);

    for (0..8) |i| {
        const cpu_result = a_vals[i].montgomeryMul(b_vals[i]);
        try testing.expectEqual(cpu_result.limbs, gpu_out[i].limbs);
    }
}

test "GPU field mul: identity and commutativity" {
    const gpu_mod = @import("mod.zig");
    if (comptime !gpu_mod.is_metal_available) return;

    var gpu = device_mod.GpuAccelerator.init(testing.allocator) catch |err| {
        if (err == error.MetalNotAvailable) return;
        return err;
    };
    defer gpu.deinit();

    var ops = try GpuFieldOps.init(&gpu);
    defer ops.deinit();

    const n = 32;
    var vals: [n]BN254Scalar = undefined;
    var ones: [n]BN254Scalar = undefined;
    for (0..n) |i| {
        vals[i] = BN254Scalar.fromU64(@as(u64, @intCast(i)) * 1000 + 1);
        ones[i] = BN254Scalar.one();
    }

    // a * 1 == a
    var out_identity: [n]BN254Scalar = undefined;
    try ops.fieldMulBatch(&vals, &ones, &out_identity);
    for (0..n) |i| {
        try testing.expectEqual(vals[i].limbs, out_identity[i].limbs);
    }

    // a * b == b * a
    var b_vals: [n]BN254Scalar = undefined;
    for (0..n) |i| {
        b_vals[i] = BN254Scalar.fromU64(@as(u64, @intCast(i)) * 7777 + 42);
    }
    var out_ab: [n]BN254Scalar = undefined;
    var out_ba: [n]BN254Scalar = undefined;
    try ops.fieldMulBatch(&vals, &b_vals, &out_ab);
    try ops.fieldMulBatch(&b_vals, &vals, &out_ba);
    for (0..n) |i| {
        try testing.expectEqual(out_ab[i].limbs, out_ba[i].limbs);
    }
}

test "GPU field add: correctness" {
    const gpu_mod = @import("mod.zig");
    if (comptime !gpu_mod.is_metal_available) return;

    var gpu = device_mod.GpuAccelerator.init(testing.allocator) catch |err| {
        if (err == error.MetalNotAvailable) return;
        return err;
    };
    defer gpu.deinit();

    var ops = try GpuFieldOps.init(&gpu);
    defer ops.deinit();

    const n = 16;
    var a_vals: [n]BN254Scalar = undefined;
    var b_vals: [n]BN254Scalar = undefined;
    for (0..n) |i| {
        a_vals[i] = BN254Scalar.fromU64(@as(u64, @intCast(i)) * 100 + 1);
        b_vals[i] = BN254Scalar.fromU64(@as(u64, @intCast(i)) * 200 + 3);
    }

    var gpu_out: [n]BN254Scalar = undefined;
    try ops.fieldAddBatch(&a_vals, &b_vals, &gpu_out);

    for (0..n) |i| {
        const cpu_result = a_vals[i].add(b_vals[i]);
        try testing.expectEqual(cpu_result.limbs, gpu_out[i].limbs);
    }
}

test "GPU field sub: correctness" {
    const gpu_mod = @import("mod.zig");
    if (comptime !gpu_mod.is_metal_available) return;

    var gpu = device_mod.GpuAccelerator.init(testing.allocator) catch |err| {
        if (err == error.MetalNotAvailable) return;
        return err;
    };
    defer gpu.deinit();

    var ops = try GpuFieldOps.init(&gpu);
    defer ops.deinit();

    const n = 16;
    var a_vals: [n]BN254Scalar = undefined;
    var b_vals: [n]BN254Scalar = undefined;
    for (0..n) |i| {
        a_vals[i] = BN254Scalar.fromU64(@as(u64, @intCast(i)) * 300 + 7);
        b_vals[i] = BN254Scalar.fromU64(@as(u64, @intCast(i)) * 100 + 1);
    }

    var gpu_out: [n]BN254Scalar = undefined;
    try ops.fieldSubBatch(&a_vals, &b_vals, &gpu_out);

    for (0..n) |i| {
        const cpu_result = a_vals[i].sub(b_vals[i]);
        try testing.expectEqual(cpu_result.limbs, gpu_out[i].limbs);
    }

    // Also test subtraction that wraps (a < b)
    try ops.fieldSubBatch(&b_vals, &a_vals, &gpu_out);
    for (0..n) |i| {
        const cpu_result = b_vals[i].sub(a_vals[i]);
        try testing.expectEqual(cpu_result.limbs, gpu_out[i].limbs);
    }
}

test "GPU field neg: correctness" {
    const gpu_mod = @import("mod.zig");
    if (comptime !gpu_mod.is_metal_available) return;

    var gpu = device_mod.GpuAccelerator.init(testing.allocator) catch |err| {
        if (err == error.MetalNotAvailable) return;
        return err;
    };
    defer gpu.deinit();

    var ops = try GpuFieldOps.init(&gpu);
    defer ops.deinit();

    var a_vals = [_]BN254Scalar{
        BN254Scalar.fromU64(0), // neg(0) == 0
        BN254Scalar.fromU64(1),
        BN254Scalar.fromU64(42),
        BN254Scalar.fromU64(0xFFFFFFFFFFFFFFFF),
    };

    var gpu_out: [4]BN254Scalar = undefined;
    try ops.fieldNegBatch(&a_vals, &gpu_out);

    for (0..4) |i| {
        const cpu_result = a_vals[i].neg();
        try testing.expectEqual(cpu_result.limbs, gpu_out[i].limbs);
    }

    // Double negation: neg(neg(a)) == a
    var gpu_out2: [4]BN254Scalar = undefined;
    try ops.fieldNegBatch(&gpu_out, &gpu_out2);
    for (0..4) |i| {
        try testing.expectEqual(a_vals[i].limbs, gpu_out2[i].limbs);
    }
}

test "GPU field mul: larger batch (1024)" {
    const gpu_mod = @import("mod.zig");
    if (comptime !gpu_mod.is_metal_available) return;

    var gpu = device_mod.GpuAccelerator.init(testing.allocator) catch |err| {
        if (err == error.MetalNotAvailable) return;
        return err;
    };
    defer gpu.deinit();

    var ops = try GpuFieldOps.init(&gpu);
    defer ops.deinit();

    const n = 1024;
    var a_vals: [n]BN254Scalar = undefined;
    var b_vals: [n]BN254Scalar = undefined;
    for (0..n) |i| {
        a_vals[i] = BN254Scalar.fromU64(@as(u64, @intCast(i)) *% 0x123456789 +% 1);
        b_vals[i] = BN254Scalar.fromU64(@as(u64, @intCast(i)) *% 0x987654321 +% 7);
    }

    var gpu_out: [n]BN254Scalar = undefined;
    try ops.fieldMulBatch(&a_vals, &b_vals, &gpu_out);

    for (0..n) |i| {
        const cpu_result = a_vals[i].montgomeryMul(b_vals[i]);
        try testing.expectEqual(cpu_result.limbs, gpu_out[i].limbs);
    }
}
