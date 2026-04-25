//! Shared debug output control for the zkVM prover.
//! Set verbose = true to enable debug prints across all prover stages.

const std = @import("std");
pub const is_wasm = @import("zolt_pool").is_wasm;

pub const verbose = false;
pub const bench_timing = false;

/// Platform timer — no-op stub on WASM, real timer on native using clock_gettime.
pub const PlatformTimer = if (is_wasm) struct {
    pub fn start() error{}!@This() {
        return .{};
    }
    pub fn read(_: @This()) u64 {
        return 0;
    }
    pub fn reset(_: *@This()) void {}
} else MonotonicTimer;

/// Minimal monotonic timer using clock_gettime (replaces std.time.Timer removed in Zig 0.16).
pub const MonotonicTimer = struct {
    start_ns: u64,

    pub fn start() error{}!MonotonicTimer {
        return .{ .start_ns = clockNs() };
    }

    pub fn read(self: MonotonicTimer) u64 {
        return clockNs() - self.start_ns;
    }

    pub fn reset(self: *MonotonicTimer) void {
        self.start_ns = clockNs();
    }

    fn clockNs() u64 {
        var ts: std.c.timespec = undefined;
        _ = std.c.clock_gettime(.MONOTONIC, &ts);
        return @intCast(@as(i128, ts.sec) * std.time.ns_per_s + ts.nsec);
    }
};

/// Platform nanoTimestamp — returns 0 on WASM.
pub fn nanoTimestamp() i128 {
    if (comptime is_wasm) return 0;
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(i128, ts.sec) * std.time.ns_per_s + ts.nsec;
}

/// Default Io instance for synchronous file operations (not available on WASM).
pub fn defaultIo() std.Io {
    return std.Io.Threaded.global_single_threaded.io();
}

/// Platform getenv — returns null on WASM (no POSIX environment).
/// Uses std.c.getenv (direct libc call) since std.posix.getenv was removed in Zig 0.16.
pub fn getenv(key: [*:0]const u8) ?[*:0]const u8 {
    if (comptime is_wasm) return null;
    return std.c.getenv(key);
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
