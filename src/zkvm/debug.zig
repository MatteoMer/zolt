//! Shared debug output control for the zkVM prover.
//! Set verbose = true to enable debug prints across all prover stages.

const std = @import("std");

pub const verbose = false;

pub fn dbg(comptime fmt: []const u8, args: anytype) void {
    if (verbose) std.debug.print(fmt, args);
}
