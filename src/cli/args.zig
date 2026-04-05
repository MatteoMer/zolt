//! CLI argument parsing for the Zolt zkVM.
//!
//! Extracts command parsing, help/version output, and argument structs
//! from main.zig for cleaner separation of concerns.

const std = @import("std");
const zolt = @import("../root.zig");

pub const Command = enum {
    help,
    version,
    run,
    prove,
    unknown,
};

pub const RunArgs = struct {
    elf_path: ?[]const u8 = null,
    show_regs: bool = false,
    show_trace: bool = false,
    max_trace_steps: ?usize = null,
    input_file: ?[]const u8 = null,
    input_hex: ?[]const u8 = null,
    show_help: bool = false,
    had_first_arg: bool = false,
};

pub const ProveArgs = struct {
    elf_path: ?[]const u8 = null,
    output_path: ?[]const u8 = null,
    srs_path: ?[]const u8 = null,
    preprocessing_path: ?[]const u8 = null,
    input_hex: ?[]const u8 = null,
    show_help: bool = false,
    had_first_arg: bool = false,
};

pub fn printHelp() void {
    std.debug.print(
        \\Zolt zkVM v{s}
        \\A Zig port of the Jolt zkVM (a16z/jolt)
        \\
        \\USAGE:
        \\    zolt <command> [options]
        \\
        \\COMMANDS:
        \\    help                Show this help message
        \\    version             Show version information
        \\    prove [opts] <elf>  Generate ZK proof for ELF binary
        \\    run [opts] <elf>    Run RISC-V ELF binary in the emulator
        \\
        \\EXAMPLES:
        \\    zolt prove -o proof.bin program.elf         # Generate and save a proof
        \\    zolt run program.elf                        # Execute a RISC-V binary
        \\    zolt run --trace program.elf                # Show execution trace
        \\
        \\For more information, visit: https://github.com/MatteoMer/zolt
        \\
    , .{zolt.version});
}

pub fn printVersion() void {
    std.debug.print("zolt {s}\n", .{zolt.version});
    std.debug.print("zig version: {s}\n", .{@import("builtin").zig_version_string});
}

pub fn parseCommand(arg: []const u8) Command {
    if (std.mem.eql(u8, arg, "help") or std.mem.eql(u8, arg, "-h") or std.mem.eql(u8, arg, "--help")) {
        return .help;
    } else if (std.mem.eql(u8, arg, "version") or std.mem.eql(u8, arg, "-v") or std.mem.eql(u8, arg, "--version")) {
        return .version;
    } else if (std.mem.eql(u8, arg, "run")) {
        return .run;
    } else if (std.mem.eql(u8, arg, "prove")) {
        return .prove;
    }
    return .unknown;
}

/// Parse hex string (with optional 0x prefix) into bytes.
/// Returns null if allocation fails or the hex string contains invalid characters.
pub fn parseHexInput(allocator: std.mem.Allocator, hex: []const u8) ?[]u8 {
    var clean_hex = hex;
    if (std.mem.startsWith(u8, hex, "0x") or std.mem.startsWith(u8, hex, "0X")) {
        clean_hex = hex[2..];
    }
    const buf_len = (clean_hex.len + 1) / 2;
    const buf = allocator.alloc(u8, buf_len) catch return null;
    var i: usize = 0;
    while (i < buf_len) : (i += 1) {
        const start = i * 2;
        const end = @min(start + 2, clean_hex.len);
        buf[i] = std.fmt.parseInt(u8, clean_hex[start..end], 16) catch {
            std.debug.print("Error: invalid hex byte '{s}' in --input-hex\n", .{clean_hex[start..end]});
            allocator.free(buf);
            return null;
        };
    }
    return buf;
}

/// Parse run sub-command arguments from an argument iterator.
/// The first non-flag argument after "run" should be passed as `first_arg`.
pub fn parseRunArgs(args: anytype, first_arg: []const u8) RunArgs {
    var result = RunArgs{};

    if (std.mem.eql(u8, first_arg, "--help") or std.mem.eql(u8, first_arg, "-h")) {
        result.show_help = true;
        return result;
    }

    result.had_first_arg = true;

    // Process first arg
    if (std.mem.startsWith(u8, first_arg, "--")) {
        processRunFlag(&result, first_arg, args);
    } else {
        result.elf_path = first_arg;
    }

    // Process remaining args
    while (args.next()) |next_arg| {
        if (std.mem.startsWith(u8, next_arg, "--")) {
            processRunFlag(&result, next_arg, args);
        } else if (result.elf_path == null) {
            result.elf_path = next_arg;
        }
    }

    return result;
}

fn processRunFlag(result: *RunArgs, flag: []const u8, args: anytype) void {
    if (std.mem.eql(u8, flag, "--regs")) {
        result.show_regs = true;
    } else if (std.mem.eql(u8, flag, "--trace")) {
        result.show_trace = true;
    } else if (std.mem.eql(u8, flag, "--max")) {
        if (args.next()) |n_str| {
            result.max_trace_steps = std.fmt.parseInt(usize, n_str, 10) catch null;
        }
    } else if (std.mem.eql(u8, flag, "--input")) {
        result.input_file = args.next();
    } else if (std.mem.eql(u8, flag, "--input-hex")) {
        result.input_hex = args.next();
    }
}

/// Parse prove sub-command arguments from an argument iterator.
/// The first non-flag argument after "prove" should be passed as `first_arg`.
pub fn parseProveArgs(args: anytype, first_arg: []const u8) ProveArgs {
    var result = ProveArgs{};

    if (std.mem.eql(u8, first_arg, "--help") or std.mem.eql(u8, first_arg, "-h")) {
        result.show_help = true;
        return result;
    }

    result.had_first_arg = true;

    // Process first arg
    if (std.mem.startsWith(u8, first_arg, "-")) {
        processProveFlag(&result, first_arg, args);
    } else {
        result.elf_path = first_arg;
    }

    // Process remaining args
    while (args.next()) |next_arg| {
        if (std.mem.startsWith(u8, next_arg, "-")) {
            processProveFlag(&result, next_arg, args);
        } else if (result.elf_path == null) {
            result.elf_path = next_arg;
        }
    }

    return result;
}

fn processProveFlag(result: *ProveArgs, flag: []const u8, args: anytype) void {
    if (std.mem.eql(u8, flag, "-o") or std.mem.eql(u8, flag, "--output")) {
        result.output_path = args.next();
    } else if (std.mem.eql(u8, flag, "--srs")) {
        result.srs_path = args.next();
    } else if (std.mem.eql(u8, flag, "--export-preprocessing")) {
        result.preprocessing_path = args.next();
    } else if (std.mem.eql(u8, flag, "--input-hex")) {
        result.input_hex = args.next();
    }
}

pub fn printRunHelp() void {
    std.debug.print("Usage: zolt run [options] <elf_file>\n\n", .{});
    std.debug.print("Run a RISC-V ELF binary in the emulator.\n", .{});
    std.debug.print("The emulator supports RV64IMC instructions.\n\n", .{});
    std.debug.print("Options:\n", .{});
    std.debug.print("  --regs           Show final register state\n", .{});
    std.debug.print("  --trace          Show execution trace\n", .{});
    std.debug.print("  --max N          Max trace entries to display (default: 100)\n", .{});
    std.debug.print("  --input FILE     Load input bytes from FILE\n", .{});
    std.debug.print("  --input-hex HEX  Set input as hex bytes (e.g., 0x32 for input 50)\n", .{});
}

pub fn printProveHelp() void {
    std.debug.print("Usage: zolt prove [options] -o <output> <elf_file>\n\n", .{});
    std.debug.print("Generate a ZK proof for a RISC-V ELF binary.\n", .{});
    std.debug.print("This command runs the full proving pipeline:\n", .{});
    std.debug.print("  1. Preprocess (generate SRS and keys)\n", .{});
    std.debug.print("  2. Initialize prover\n", .{});
    std.debug.print("  3. Generate proof using multi-stage sumcheck\n", .{});
    std.debug.print("  4. Save proof to file\n\n", .{});
    std.debug.print("Options:\n", .{});
    std.debug.print("  -o, --output F           Save proof to file F (required)\n", .{});
    std.debug.print("  --srs PATH               Use Dory SRS from PATH (exported by Jolt)\n", .{});
    std.debug.print("  --export-preprocessing P Export Jolt-compatible preprocessing to file P\n", .{});
    std.debug.print("  --input-hex HEX          Set input as hex bytes (e.g., 20 for input 32)\n", .{});
}

test "command parsing" {
    try std.testing.expect(parseCommand("help") == .help);
    try std.testing.expect(parseCommand("-h") == .help);
    try std.testing.expect(parseCommand("--help") == .help);
    try std.testing.expect(parseCommand("version") == .version);
    try std.testing.expect(parseCommand("-v") == .version);
    try std.testing.expect(parseCommand("run") == .run);
    try std.testing.expect(parseCommand("prove") == .prove);
    try std.testing.expect(parseCommand("unknown_cmd") == .unknown);
}

test "parseHexInput" {
    const allocator = std.testing.allocator;

    // Test basic hex
    const result1 = parseHexInput(allocator, "32");
    try std.testing.expect(result1 != null);
    try std.testing.expectEqual(@as(u8, 0x32), result1.?[0]);
    allocator.free(result1.?);

    // Test with 0x prefix
    const result2 = parseHexInput(allocator, "0xFF");
    try std.testing.expect(result2 != null);
    try std.testing.expectEqual(@as(u8, 0xFF), result2.?[0]);
    allocator.free(result2.?);
}
