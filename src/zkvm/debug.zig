//! Shared debug output control for the zkVM prover.
//! Set verbose = true to enable debug prints across all prover stages.

const std = @import("std");

pub const verbose = false;
pub const bench_timing = false;

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
