//! Zolt CLI - A Zig port of the Jolt zkVM
//!
//! This executable provides command-line tools for:
//! - Proving RISC-V programs (Jolt-compatible proofs with Dory commitments)
//! - Running the RISC-V emulator

const std = @import("std");
const zolt = @import("root.zig");
const args_mod = @import("cli/args.zig");
const run_cmd = @import("commands/run.zig");
const prove_cmd = @import("commands/prove.zig");
const verify_cmd = @import("commands/verify.zig");

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;

    var args = init.minimal.args.iterate();

    // Skip program name
    _ = args.skip();

    // Get command
    const cmd_arg = args.next() orelse {
        args_mod.printHelp();
        return;
    };

    const cmd = args_mod.parseCommand(cmd_arg);

    switch (cmd) {
        .help => args_mod.printHelp(),
        .version => args_mod.printVersion(),
        .run => {
            const first_arg = args.next() orelse {
                std.debug.print("Error: run command requires an ELF file path\n", .{});
                std.debug.print("Usage: zolt run [options] <elf_file>\n", .{});
                return;
            };

            const run_args = args_mod.parseRunArgs(&args, first_arg);

            if (run_args.show_help) {
                args_mod.printRunHelp();
                return;
            }

            var input_bytes_owned: ?[]u8 = null;
            defer if (input_bytes_owned) |b| allocator.free(b);

            if (run_args.input_file) |path| {
                const io = init.io;
                input_bytes_owned = std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(256 * 1024 * 1024)) catch |err| {
                    std.debug.print("Failed to read input file: {s}\n", .{@errorName(err)});
                    std.process.exit(1);
                };
            } else if (run_args.input_hex) |hex| {
                input_bytes_owned = args_mod.parseHexInput(allocator, hex);
            }

            if (run_args.elf_path) |path| {
                run_cmd.runEmulator(allocator, path, run_args.show_regs, run_args.show_trace, run_args.max_trace_steps, input_bytes_owned) catch |err| {
                    std.debug.print("Failed to run program: {s}\n", .{@errorName(err)});
                    std.process.exit(1);
                };
            } else {
                std.debug.print("Error: run command requires an ELF file path\n", .{});
                std.debug.print("Usage: zolt run [options] <elf_file>\n", .{});
            }
        },
        .prove => {
            const first_arg = args.next() orelse {
                std.debug.print("Error: prove command requires an ELF file path\n", .{});
                std.debug.print("Usage: zolt prove -o <output> [options] <elf_file>\n", .{});
                return;
            };

            const prove_args = args_mod.parseProveArgs(&args, first_arg);

            if (prove_args.show_help) {
                args_mod.printProveHelp();
                return;
            }

            var input_bytes_owned: ?[]u8 = null;
            defer if (input_bytes_owned) |b| allocator.free(b);

            if (prove_args.input_hex) |hex| {
                input_bytes_owned = args_mod.parseHexInput(allocator, hex);
            }

            if (prove_args.output_path == null) {
                std.debug.print("Error: prove command requires an output path (-o <file>)\n", .{});
                std.debug.print("Usage: zolt prove -o <output> [options] <elf_file>\n", .{});
                return;
            }

            if (prove_args.elf_path) |path| {
                prove_cmd.runProver(allocator, path, prove_args.output_path.?, prove_args.srs_path, prove_args.preprocessing_path, input_bytes_owned) catch |err| {
                    std.debug.print("Failed to generate proof: {s}\n", .{@errorName(err)});
                    std.process.exit(1);
                };
            } else {
                std.debug.print("Error: prove command requires an ELF file path\n", .{});
                std.debug.print("Usage: zolt prove -o <output> [options] <elf_file>\n", .{});
            }
        },
        .verify => {
            const first_arg = args.next() orelse {
                std.debug.print("Error: verify command requires --proof and --preprocessing\n", .{});
                std.debug.print("Usage: zolt verify --proof <file> --preprocessing <file>\n", .{});
                return;
            };

            const verify_args = args_mod.parseVerifyArgs(&args, first_arg);

            if (verify_args.show_help) {
                args_mod.printVerifyHelp();
                return;
            }

            if (verify_args.proof_path == null) {
                std.debug.print("Error: verify command requires a proof file (--proof <file>)\n", .{});
                std.debug.print("Usage: zolt verify --proof <file> --preprocessing <file>\n", .{});
                return;
            }

            if (verify_args.preprocessing_path == null) {
                std.debug.print("Error: verify command requires a preprocessing file (--preprocessing <file>)\n", .{});
                std.debug.print("Usage: zolt verify --proof <file> --preprocessing <file>\n", .{});
                return;
            }

            verify_cmd.runVerifier(allocator, verify_args.proof_path.?, verify_args.preprocessing_path.?) catch |err| {
                std.debug.print("Verification failed: {s}\n", .{@errorName(err)});
                std.process.exit(1);
            };
        },
        .unknown => {
            std.debug.print("Unknown command: {s}\n\n", .{cmd_arg});
            args_mod.printHelp();
        },
    }
}

test "zolt version" {
    const version = zolt.version;
    try std.testing.expect(version.len > 0);
}

test "command parsing" {
    try std.testing.expect(args_mod.parseCommand("help") == .help);
    try std.testing.expect(args_mod.parseCommand("-h") == .help);
    try std.testing.expect(args_mod.parseCommand("--help") == .help);
    try std.testing.expect(args_mod.parseCommand("version") == .version);
    try std.testing.expect(args_mod.parseCommand("-v") == .version);
    try std.testing.expect(args_mod.parseCommand("run") == .run);
    try std.testing.expect(args_mod.parseCommand("prove") == .prove);
    try std.testing.expect(args_mod.parseCommand("verify") == .verify);
    try std.testing.expect(args_mod.parseCommand("unknown_cmd") == .unknown);
}

test "parseHexInput" {
    const allocator = std.testing.allocator;

    // Test basic hex
    const result1 = args_mod.parseHexInput(allocator, "32");
    try std.testing.expect(result1 != null);
    try std.testing.expectEqual(@as(u8, 0x32), result1.?[0]);
    allocator.free(result1.?);

    // Test with 0x prefix
    const result2 = args_mod.parseHexInput(allocator, "0xFF");
    try std.testing.expect(result2 != null);
    try std.testing.expectEqual(@as(u8, 0xFF), result2.?[0]);
    allocator.free(result2.?);
}
