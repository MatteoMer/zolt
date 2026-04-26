//! Shared debug output control for the zkVM prover.
//! Set verbose = true to enable debug prints across all prover stages.

const std = @import("std");
const builtin = @import("builtin");
const zolt_pool = @import("zolt_pool");
pub const is_wasm = zolt_pool.is_wasm;

pub const verbose = false;
pub const bench_timing = false;

/// Platform timer — no-op stub on WASM, real timer on native.
pub const PlatformTimer = if (is_wasm) struct {
    pub fn init(_: std.Io) @This() {
        return .{};
    }
    pub fn read(_: @This()) u64 {
        return 0;
    }
    pub fn reset(_: *@This()) void {}
} else MonotonicTimer;

/// Shared monotonic timer from zolt-pool — takes io: std.Io.
pub const MonotonicTimer = zolt_pool.MonotonicTimer;

/// Platform nanoTimestamp — returns 0 on WASM.
pub fn nanoTimestamp(io: std.Io) i128 {
    if (comptime is_wasm) return 0;
    return std.Io.Timestamp.now(io, .boot).nanoseconds;
}

/// Default Io instance for synchronous file operations (not available on WASM).
pub fn defaultIo() std.Io {
    return std.Io.Threaded.global_single_threaded.io();
}

/// Platform getenv — returns null on WASM (no POSIX environment).
/// Scans the process environment block directly (no libc dependency).
pub fn getenv(key: [*:0]const u8) ?[*:0]const u8 {
    if (comptime is_wasm) return null;
    const key_slice = std.mem.span(key);
    const t = std.Io.Threaded.global_single_threaded;
    const block = t.environ.process_environ.block;
    for (@as([:null]const ?[*:0]const u8, block.slice)) |entry_opt| {
        const entry = std.mem.span(entry_opt orelse continue);
        if (entry.len > key_slice.len and
            entry[key_slice.len] == '=' and
            std.mem.eql(u8, entry[0..key_slice.len], key_slice))
        {
            return (entry_opt.?)[key_slice.len + 1 ..];
        }
    }
    return null;
}

pub fn dbg(comptime fmt: []const u8, args: anytype) void {
    if (verbose) std.debug.print(fmt, args);
}

/// Print a field element as 32-byte little-endian hex: "label[hex...]\n"
pub fn dbgFieldLE(comptime label: []const u8, field: anytype) void {
    if (verbose) {
        const be = field.toBytesBE();
        std.debug.print(label ++ "[", .{});
        for (0..32) |bi| std.debug.print("{x:0>2}", .{be[31 - bi]});
        std.debug.print("]\n", .{});
    }
}

/// Print first 8 bytes of a field element as little-endian hex: "label[hex...]\n"
pub fn dbgFieldLE8(comptime label: []const u8, field: anytype) void {
    if (verbose) {
        const be = field.toBytesBE();
        std.debug.print(label ++ "[", .{});
        for (0..8) |bi| std.debug.print("{x:0>2}", .{be[31 - bi]});
        std.debug.print("]\n", .{});
    }
}
