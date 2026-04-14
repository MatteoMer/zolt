//! Shared debug output control for the zkVM prover.
//! Set verbose = true to enable debug prints across all prover stages.

const std = @import("std");
pub const is_wasm = @import("zolt_pool").is_wasm;

pub const verbose = false;
pub const bench_timing = false;

/// Platform timer — no-op stub on WASM, real timer on native.
pub const PlatformTimer = if (is_wasm) struct {
    pub fn start() error{}!@This() {
        return .{};
    }
    pub fn read(_: @This()) u64 {
        return 0;
    }
    pub fn reset(_: *@This()) void {}
} else std.time.Timer;

/// Platform nanoTimestamp — returns 0 on WASM.
pub fn nanoTimestamp() i128 {
    if (comptime is_wasm) return 0;
    return std.time.nanoTimestamp();
}

/// Platform getenv — returns null on WASM (no POSIX environment).
pub fn getenv(key: []const u8) ?[:0]const u8 {
    if (comptime is_wasm) return null;
    return std.posix.getenv(key);
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
