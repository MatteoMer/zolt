//! C FFI layer for Zolt proving pipeline.
//!
//! Exposes opaque-handle functions with C calling convention so Rust
//! (or any other language) can drive Zolt in-process without spawning
//! a separate binary.

const std = @import("std");
const zolt = @import("root.zig");
const BN254Scalar = zolt.field.BN254Scalar;

const allocator = std.heap.page_allocator;

// ── Opaque handle types ──────────────────────────────────────────────

const ThreadPoolCtx = struct {
    tp: *zolt.utils.ThreadPool,
};

const LoadedElf = struct {
    tp: *zolt.utils.ThreadPool,
    bytecode: []const u8,
    entry_point: u64,
    base_address: u64,
    text_size: usize,
    program: zolt.host.Program,
};

const ProofResult = struct {
    bytes: []u8,
};

// ── Thread pool lifecycle ────────────────────────────────────────────

export fn zolt_thread_pool_create() callconv(.c) ?*anyopaque {
    const tp = zolt.utils.ThreadPool.init(allocator) catch return null;
    const ctx = allocator.create(ThreadPoolCtx) catch {
        tp.deinit();
        return null;
    };
    ctx.* = .{ .tp = tp };
    return @ptrCast(ctx);
}

export fn zolt_thread_pool_destroy(handle: ?*anyopaque) callconv(.c) void {
    const ctx: *ThreadPoolCtx = cast(ThreadPoolCtx, handle) orelse return;
    ctx.tp.deinit();
    allocator.destroy(ctx);
}

// ── ELF loading ──────────────────────────────────────────────────────

export fn zolt_load_elf(
    tp_handle: ?*anyopaque,
    path_ptr: [*]const u8,
    path_len: usize,
) callconv(.c) ?*anyopaque {
    const tp_ctx: *ThreadPoolCtx = cast(ThreadPoolCtx, tp_handle) orelse return null;
    const path = path_ptr[0..path_len];

    var loader = zolt.host.ELFLoader.init(allocator);
    const program = loader.loadFile(path) catch return null;

    const ctx = allocator.create(LoadedElf) catch {
        var p = program;
        p.deinit();
        return null;
    };
    ctx.* = .{
        .tp = tp_ctx.tp,
        .bytecode = program.bytecode,
        .entry_point = program.entry_point,
        .base_address = program.base_address,
        .text_size = program.text_size,
        .program = program,
    };
    return @ptrCast(ctx);
}

export fn zolt_loaded_elf_size(handle: ?*anyopaque) callconv(.c) usize {
    const ctx: *LoadedElf = cast(LoadedElf, handle) orelse return 0;
    return ctx.bytecode.len;
}

export fn zolt_loaded_elf_destroy(handle: ?*anyopaque) callconv(.c) void {
    const ctx: *LoadedElf = cast(LoadedElf, handle) orelse return;
    var p = ctx.program;
    p.deinit();
    allocator.destroy(ctx);
}

// ── Proving ──────────────────────────────────────────────────────────

export fn zolt_prove(elf_handle: ?*anyopaque) callconv(.c) ?*anyopaque {
    const elf: *LoadedElf = cast(LoadedElf, elf_handle) orelse return null;

    var prover = zolt.zkvm.JoltProver(BN254Scalar).initWithThreadPool(allocator, elf.tp);

    var bundle = prover.proveJoltCompatibleWithDoryAndSrsAtAddress(
        elf.bytecode,
        &[_]u8{},
        null,
        elf.base_address,
        elf.entry_point,
        elf.text_size,
    ) catch return null;
    defer bundle.deinit();

    const bytes = prover.serializeJoltProofWithDory(&bundle) catch return null;

    const result = allocator.create(ProofResult) catch {
        allocator.free(bytes);
        return null;
    };
    result.* = .{ .bytes = bytes };
    return @ptrCast(result);
}

// ── Proof result accessors ───────────────────────────────────────────

export fn zolt_proof_result_size(handle: ?*anyopaque) callconv(.c) usize {
    const result: *ProofResult = cast(ProofResult, handle) orelse return 0;
    return result.bytes.len;
}

export fn zolt_proof_result_destroy(handle: ?*anyopaque) callconv(.c) void {
    const result: *ProofResult = cast(ProofResult, handle) orelse return;
    allocator.free(result.bytes);
    allocator.destroy(result);
}

// ── Helpers ──────────────────────────────────────────────────────────

fn cast(comptime T: type, handle: ?*anyopaque) ?*T {
    const ptr = handle orelse return null;
    return @ptrCast(@alignCast(ptr));
}
