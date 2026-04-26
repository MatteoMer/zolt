//! Shared monotonic timer for Zolt.
//!
//! Single definition that accepts `io: std.Io` — replaces the per-file
//! copies that each grabbed `global_single_threaded.io()` internally.

const std = @import("std");

pub const MonotonicTimer = struct {
    start_ts: std.Io.Timestamp,
    io: std.Io,

    pub fn init(io: std.Io) MonotonicTimer {
        return .{ .start_ts = std.Io.Timestamp.now(io, .boot), .io = io };
    }

    pub fn read(self: MonotonicTimer) u64 {
        const dur = self.start_ts.durationTo(std.Io.Timestamp.now(self.io, .boot));
        return @intCast(@max(0, dur.nanoseconds));
    }

    pub fn reset(self: *MonotonicTimer) void {
        self.start_ts = std.Io.Timestamp.now(self.io, .boot);
    }
};
