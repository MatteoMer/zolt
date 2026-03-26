//! Shared-mode Metal buffer wrapper with zero-copy CPU access.
//!
//! On Apple Silicon, storageModeShared buffers reside in unified memory —
//! both CPU and GPU access the same physical DRAM with no copies.

const objc = @import("objc.zig");
const id = objc.id;

pub fn GpuBuffer(comptime T: type) type {
    return struct {
        metal_buffer: id,
        ptr: [*]T,
        len: usize,

        const Self = @This();

        pub const Error = error{
            BufferAllocFailed,
            BufferContentsFailed,
        };

        /// Allocate a shared-mode Metal buffer for `count` elements of type T.
        /// The returned buffer is immediately accessible by both CPU and GPU.
        pub fn init(device: id, count: usize) Error!Self {
            const byte_len = count * @sizeOf(T);
            // MTLResourceStorageModeShared = 0
            const buf = objc.msgSendNewBuffer(device, byte_len, 0) orelse
                return error.BufferAllocFailed;
            const raw_ptr = objc.msgSendPtr(buf, "contents") orelse {
                objc.msgSendVoid(buf, "release");
                return error.BufferAllocFailed;
            };
            return .{
                .metal_buffer = buf,
                .ptr = @ptrCast(@alignCast(raw_ptr)),
                .len = count,
            };
        }

        pub fn deinit(self: *Self) void {
            objc.msgSendVoid(self.metal_buffer, "release");
        }

        /// Zero-copy CPU-side slice. Reads/writes go directly to GPU-visible memory.
        pub fn slice(self: *const Self) []T {
            return self.ptr[0..self.len];
        }
    };
}
