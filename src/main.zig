//! Zolt CLI - A Zig port of the Jolt zkVM
//!
//! This executable provides command-line tools for:
//! - Proving RISC-V program execution
//! - Verifying proofs
//! - Benchmarking

const std = @import("std");
const zolt = @import("root.zig");

pub fn main() void {
    std.debug.print("Zolt zkVM v{s}\n", .{zolt.version});
    std.debug.print("A Zig port of the Jolt zkVM (a16z/jolt)\n\n", .{});

    std.debug.print("Usage: zolt <command> [options]\n", .{});
    std.debug.print("\nCommands:\n", .{});
    std.debug.print("  prove   <elf>    Generate a proof for a RISC-V ELF binary\n", .{});
    std.debug.print("  verify  <proof>  Verify a proof\n", .{});
    std.debug.print("  bench            Run benchmarks\n", .{});
    std.debug.print("\nThis is a work in progress.\n", .{});
}

test "zolt version" {
    const version = zolt.version;
    try std.testing.expect(version.len > 0);
}
