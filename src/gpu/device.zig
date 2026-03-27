//! Metal device initialization, command queue, library loading, and compute dispatch.
//!
//! GpuAccelerator is the single entry point for all GPU operations. It owns the
//! Metal device, command queue, and pre-loaded shader library.

const std = @import("std");
const objc = @import("objc.zig");
const id = objc.id;
const MTLSize = objc.MTLSize;

/// Plain C function exported by Metal.framework.
extern "c" fn MTLCreateSystemDefaultDevice() ?id;

pub const GpuAccelerator = struct {
    device: id,
    queue: id,
    library: id,
    allocator: std.mem.Allocator,

    pub const Error = error{
        MetalNotAvailable,
        CommandQueueFailed,
        LibraryLoadFailed,
        PipelineCreateFailed,
        FunctionNotFound,
        DispatchFailed,
    };

    pub fn init(allocator: std.mem.Allocator) Error!GpuAccelerator {
        const dev = MTLCreateSystemDefaultDevice() orelse return error.MetalNotAvailable;
        errdefer objc.msgSendVoid(dev, "release");

        const queue = objc.msgSendId(dev, "newCommandQueue") orelse
            return error.CommandQueueFailed;
        errdefer objc.msgSendVoid(queue, "release");

        const library = loadMetalLibrary(dev) orelse return error.LibraryLoadFailed;

        return .{
            .device = dev,
            .queue = queue,
            .library = library,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *GpuAccelerator) void {
        objc.msgSendVoid(self.library, "release");
        objc.msgSendVoid(self.queue, "release");
        objc.msgSendVoid(self.device, "release");
    }

    /// Runtime check for Metal GPU availability.
    pub fn isAvailable() bool {
        const dev = MTLCreateSystemDefaultDevice() orelse return false;
        objc.msgSendVoid(dev, "release");
        return true;
    }

    /// Create a compute pipeline state for a named kernel function in the loaded library.
    pub fn createPipeline(self: *GpuAccelerator, comptime name: [:0]const u8) Error!id {
        const pool = objc.objc_autoreleasePoolPush() orelse return error.DispatchFailed;
        defer objc.objc_autoreleasePoolPop(pool);

        const ns_name = objc.nsString(name) orelse return error.FunctionNotFound;
        // NSString from stringWithUTF8String: is autoreleased — do not release it

        const func = objc.msgSendWithId(self.library, "newFunctionWithName:", ns_name) orelse
            return error.FunctionNotFound;
        defer objc.msgSendVoid(func, "release");

        var err: ?id = null;
        const pipeline = objc.msgSendWithIdError(
            self.device,
            "newComputePipelineStateWithFunction:error:",
            func,
            &err,
        ) orelse {
            if (err) |e| objc.msgSendVoid(e, "release");
            return error.PipelineCreateFailed;
        };
        return pipeline;
    }

    /// Dispatch a compute kernel.
    ///
    /// `pipeline`: MTLComputePipelineState from createPipeline()
    /// `buffers`:  Metal buffers to bind at sequential indices starting from 0
    /// `n`:        Total number of threads to dispatch (1D grid)
    pub fn dispatchSimple(
        self: *GpuAccelerator,
        pipeline: id,
        buffers: []const id,
        n: usize,
    ) Error!void {
        const pool = objc.objc_autoreleasePoolPush() orelse return error.DispatchFailed;
        defer objc.objc_autoreleasePoolPop(pool);

        const cmd_buf = objc.msgSendId(self.queue, "commandBuffer") orelse
            return error.DispatchFailed;
        const encoder = objc.msgSendId(cmd_buf, "computeCommandEncoder") orelse
            return error.DispatchFailed;

        objc.msgSendVoidWithId(encoder, "setComputePipelineState:", pipeline);

        for (buffers, 0..) |buf, i| {
            objc.msgSendSetBuffer(encoder, buf, 0, i);
        }

        const max_tg = objc.msgSendUsize(pipeline, "maxTotalThreadsPerThreadgroup");
        const tg_width = @min(max_tg, n);

        objc.msgSendDispatchThreads(
            encoder,
            .{ .width = n, .height = 1, .depth = 1 },
            .{ .width = tg_width, .height = 1, .depth = 1 },
        );

        objc.msgSendVoid(encoder, "endEncoding");
        objc.msgSendVoid(cmd_buf, "commit");
        objc.msgSendVoid(cmd_buf, "waitUntilCompleted");
    }
    /// Kernel parameter: either a Metal buffer or inline constant bytes.
    pub const KernelParam = union(enum) {
        buffer: id,
        bytes: struct { ptr: *const anyopaque, len: usize },
    };

    /// Flexible dispatch with mixed buffer/bytes parameters and explicit threadgroup size.
    ///
    /// Parameters are bound at sequential indices (0, 1, 2, ...).
    /// Use `threadgroup_size = 0` to auto-select (min of maxTotalThreadsPerThreadgroup, grid_size).
    pub fn dispatchKernel(
        self: *GpuAccelerator,
        pipeline: id,
        params: []const KernelParam,
        grid_size: usize,
        threadgroup_size: usize,
    ) Error!void {
        const pool = objc.objc_autoreleasePoolPush() orelse return error.DispatchFailed;
        defer objc.objc_autoreleasePoolPop(pool);

        const cmd_buf = objc.msgSendId(self.queue, "commandBuffer") orelse
            return error.DispatchFailed;
        const encoder = objc.msgSendId(cmd_buf, "computeCommandEncoder") orelse
            return error.DispatchFailed;

        objc.msgSendVoidWithId(encoder, "setComputePipelineState:", pipeline);

        for (params, 0..) |param, i| {
            switch (param) {
                .buffer => |buf| objc.msgSendSetBuffer(encoder, buf, 0, i),
                .bytes => |b| objc.msgSendSetBytes(encoder, b.ptr, b.len, i),
            }
        }

        const max_tg = objc.msgSendUsize(pipeline, "maxTotalThreadsPerThreadgroup");
        const tg = if (threadgroup_size > 0) @min(threadgroup_size, max_tg) else @min(max_tg, grid_size);

        objc.msgSendDispatchThreads(
            encoder,
            .{ .width = grid_size, .height = 1, .depth = 1 },
            .{ .width = tg, .height = 1, .depth = 1 },
        );

        objc.msgSendVoid(encoder, "endEncoding");
        objc.msgSendVoid(cmd_buf, "commit");
        objc.msgSendVoid(cmd_buf, "waitUntilCompleted");
    }
};

/// Load the pre-compiled Metal shader library embedded at build time.
fn loadMetalLibrary(device: id) ?id {
    const metallib_bytes = @embedFile("shaders/shaders.metallib");

    // Wrap embedded bytes in dispatch_data_t (static lifetime — no destructor needed)
    const data = objc.dispatch_data_create(
        metallib_bytes.ptr,
        metallib_bytes.len,
        null,
        null,
    ) orelse return null;
    defer objc.dispatch_release(data);

    var err: ?id = null;
    const library = objc.msgSendWithIdError(device, "newLibraryWithData:error:", data, &err);
    if (err) |e| objc.msgSendVoid(e, "release");
    return library;
}
