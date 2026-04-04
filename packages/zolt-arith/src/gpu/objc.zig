//! Minimal Objective-C runtime bindings for Metal integration.
//!
//! Uses direct extern declarations instead of @cImport to avoid SDK header
//! dependencies. All Metal API calls go through objc_msgSend with typed
//! function pointer casts.

// ── Types ──────────────────────────────────────────────────────────────────────

pub const id = *anyopaque;
pub const SEL = *anyopaque;
pub const Class = *anyopaque;

/// Metal's grid/threadgroup size type. Matches MTLSize layout exactly.
pub const MTLSize = extern struct {
    width: usize,
    height: usize,
    depth: usize,
};

// ── Objective-C Runtime ────────────────────────────────────────────────────────

extern "c" fn sel_registerName(name: [*:0]const u8) SEL;
extern "c" fn objc_getClass(name: [*:0]const u8) ?Class;

/// The universal ObjC message dispatch function.
extern "c" fn objc_msgSend() void;

/// Get a typed function pointer to objc_msgSend. Goes through @intFromPtr
/// to prevent LLVM from propagating the extern's declared signature through
/// the cast — without this, ReleaseFast misoptimizes the call sites.
inline fn msgSendFn(comptime F: type) F {
    return @ptrFromInt(@intFromPtr(&objc_msgSend));
}

/// Resolve an Objective-C selector by name.
pub inline fn sel(comptime name: [:0]const u8) SEL {
    return sel_registerName(name.ptr);
}

/// Look up an Objective-C class by name.
pub inline fn getClass(comptime name: [:0]const u8) ?Class {
    return objc_getClass(name.ptr);
}

// ── Typed objc_msgSend wrappers ────────────────────────────────────────────────

/// No arguments, returns object. e.g. [device newCommandQueue]
pub fn msgSendId(target: id, comptime selector_name: [:0]const u8) ?id {
    const F = *const fn (id, SEL) callconv(.c) ?id;
    return msgSendFn(F)(target, sel(selector_name));
}

/// No arguments, void return. e.g. [encoder endEncoding]
pub fn msgSendVoid(target: id, comptime selector_name: [:0]const u8) void {
    const F = *const fn (id, SEL) callconv(.c) void;
    msgSendFn(F)(target, sel(selector_name));
}

/// No arguments, returns raw pointer. e.g. [buffer contents]
pub fn msgSendPtr(target: id, comptime selector_name: [:0]const u8) ?*anyopaque {
    const F = *const fn (id, SEL) callconv(.c) ?*anyopaque;
    return msgSendFn(F)(target, sel(selector_name));
}

/// No arguments, returns usize. e.g. [pipeline maxTotalThreadsPerThreadgroup]
pub fn msgSendUsize(target: id, comptime selector_name: [:0]const u8) usize {
    const F = *const fn (id, SEL) callconv(.c) usize;
    return msgSendFn(F)(target, sel(selector_name));
}

/// One id arg, returns object. e.g. [library newFunctionWithName:name]
pub fn msgSendWithId(target: id, comptime selector_name: [:0]const u8, arg: id) ?id {
    const F = *const fn (id, SEL, id) callconv(.c) ?id;
    return msgSendFn(F)(target, sel(selector_name), arg);
}

/// One id arg, void return. e.g. [encoder setComputePipelineState:pipeline]
pub fn msgSendVoidWithId(target: id, comptime selector_name: [:0]const u8, arg: id) void {
    const F = *const fn (id, SEL, id) callconv(.c) void;
    msgSendFn(F)(target, sel(selector_name), arg);
}

/// [target selector:arg error:&err] -> ?id
pub fn msgSendWithIdError(target: id, comptime selector_name: [:0]const u8, arg: id, err: *?id) ?id {
    const F = *const fn (id, SEL, id, *?id) callconv(.c) ?id;
    return msgSendFn(F)(target, sel(selector_name), arg, err);
}

/// [device newBufferWithLength:len options:opts] -> ?id
pub fn msgSendNewBuffer(target: id, length: usize, options: u64) ?id {
    const F = *const fn (id, SEL, usize, u64) callconv(.c) ?id;
    return msgSendFn(F)(target, sel("newBufferWithLength:options:"), length, options);
}

/// [encoder setBuffer:buf offset:off atIndex:idx]
pub fn msgSendSetBuffer(target: id, buffer: id, offset: usize, index: usize) void {
    const F = *const fn (id, SEL, id, usize, usize) callconv(.c) void;
    msgSendFn(F)(target, sel("setBuffer:offset:atIndex:"), buffer, offset, index);
}

/// [encoder setBytes:bytes length:len atIndex:idx]
pub fn msgSendSetBytes(target: id, bytes: *const anyopaque, length: usize, index: usize) void {
    const F = *const fn (id, SEL, *const anyopaque, usize, usize) callconv(.c) void;
    msgSendFn(F)(target, sel("setBytes:length:atIndex:"), bytes, length, index);
}

/// [encoder dispatchThreads:gridSize threadsPerThreadgroup:tgSize]
pub fn msgSendDispatchThreads(target: id, grid_size: MTLSize, threadgroup_size: MTLSize) void {
    const F = *const fn (id, SEL, MTLSize, MTLSize) callconv(.c) void;
    msgSendFn(F)(target, sel("dispatchThreads:threadsPerThreadgroup:"), grid_size, threadgroup_size);
}

// ── Helpers ────────────────────────────────────────────────────────────────────

/// Create an NSString from a Zig string literal.
/// [NSString stringWithUTF8String:str]
pub fn nsString(str: [*:0]const u8) ?id {
    const NSString = getClass("NSString") orelse return null;
    const F = *const fn (id, SEL, [*:0]const u8) callconv(.c) ?id;
    // Class is also an id (both are *anyopaque)
    return msgSendFn(F)(@ptrCast(NSString), sel("stringWithUTF8String:"), str);
}

// ── libdispatch ────────────────────────────────────────────────────────────────

/// Wraps raw bytes into a dispatch_data_t for Metal library loading.
pub extern "c" fn dispatch_data_create(
    buffer: [*]const u8,
    size: usize,
    queue: ?*anyopaque,
    destructor: ?*anyopaque,
) ?id;

/// Proper release for dispatch objects (use instead of objc release message).
pub extern "c" fn dispatch_release(object: id) void;

/// Push/pop an Objective-C autorelease pool (required for autoreleased objects
/// in non-ObjC programs, especially in ReleaseFast builds).
pub extern "c" fn objc_autoreleasePoolPush() ?*anyopaque;
pub extern "c" fn objc_autoreleasePoolPop(pool: *anyopaque) void;
