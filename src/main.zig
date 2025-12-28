//! Zolt CLI - A Zig port of the Jolt zkVM
//!
//! This executable provides command-line tools for:
//! - Running the RISC-V emulator
//! - Running benchmarks

const std = @import("std");
const zolt = @import("root.zig");
const BN254Scalar = zolt.field.BN254Scalar;

const Command = enum {
    help,
    version,
    run,
    prove,
    bench,
    decode,
    unknown,
};

fn printHelp() void {
    std.debug.print(
        \\Zolt zkVM v{s}
        \\A Zig port of the Jolt zkVM (a16z/jolt)
        \\
        \\USAGE:
        \\    zolt <command> [options]
        \\
        \\COMMANDS:
        \\    help              Show this help message
        \\    version           Show version information
        \\    run <elf>         Run RISC-V ELF binary in the emulator
        \\    prove <elf>       Generate ZK proof for ELF binary (experimental)
        \\    decode <hex>      Decode a RISC-V instruction (hex)
        \\    bench             Run performance benchmarks
        \\
        \\EXAMPLES:
        \\    zolt run program.elf        # Execute a RISC-V binary
        \\    zolt prove program.elf      # Generate a ZK proof
        \\    zolt decode 0x00a00513      # Decode: li a0, 10
        \\    zolt bench                  # Run benchmarks
        \\
        \\For more information, visit: https://github.com/MatteoMer/zolt
        \\
    , .{zolt.version});
}

fn printVersion() void {
    std.debug.print("zolt {s}\n", .{zolt.version});
    std.debug.print("zig version: {s}\n", .{@import("builtin").zig_version_string});
}

fn parseCommand(arg: []const u8) Command {
    if (std.mem.eql(u8, arg, "help") or std.mem.eql(u8, arg, "-h") or std.mem.eql(u8, arg, "--help")) {
        return .help;
    } else if (std.mem.eql(u8, arg, "version") or std.mem.eql(u8, arg, "-v") or std.mem.eql(u8, arg, "--version")) {
        return .version;
    } else if (std.mem.eql(u8, arg, "run")) {
        return .run;
    } else if (std.mem.eql(u8, arg, "prove")) {
        return .prove;
    } else if (std.mem.eql(u8, arg, "decode")) {
        return .decode;
    } else if (std.mem.eql(u8, arg, "bench")) {
        return .bench;
    }
    return .unknown;
}

fn runEmulator(allocator: std.mem.Allocator, elf_path: []const u8) !void {
    std.debug.print("Loading ELF: {s}\n", .{elf_path});

    // Load the ELF file
    var loader = zolt.host.ELFLoader.init(allocator);
    const program = loader.loadFile(elf_path) catch |err| {
        std.debug.print("Error loading ELF file: {}\n", .{err});
        return err;
    };
    defer {
        var prog = program;
        prog.deinit();
    }

    std.debug.print("Entry point: 0x{x:0>8}\n", .{program.entry_point});
    std.debug.print("Code size: {} bytes\n", .{program.bytecode.len});
    std.debug.print("Base address: 0x{x:0>8}\n", .{program.base_address});

    // Create memory config
    var config = zolt.common.MemoryConfig{
        .program_size = program.bytecode.len,
    };

    // Create emulator
    var emulator = zolt.tracer.Emulator.init(allocator, &config);
    defer emulator.deinit();

    // Load program into memory
    try emulator.loadProgram(program.bytecode);

    // Set entry point PC
    emulator.state.pc = program.entry_point;

    std.debug.print("\nStarting execution...\n", .{});

    // Run the emulator step by step
    var running = true;
    while (running) {
        running = emulator.step() catch |err| {
            std.debug.print("Execution stopped: {}\n", .{err});
            break;
        };
    }

    std.debug.print("\nExecution complete!\n", .{});
    std.debug.print("Cycles executed: {}\n", .{emulator.state.cycle});
    std.debug.print("Final PC: 0x{x:0>8}\n", .{emulator.state.pc});
    std.debug.print("Trace entries: {}\n", .{emulator.trace.len()});
}

fn runProver(allocator: std.mem.Allocator, elf_path: []const u8) !void {
    std.debug.print("Loading ELF for proving: {s}\n", .{elf_path});

    // Load the ELF file
    var loader = zolt.host.ELFLoader.init(allocator);
    const program = loader.loadFile(elf_path) catch |err| {
        std.debug.print("Error loading ELF file: {}\n", .{err});
        return err;
    };
    defer {
        var prog = program;
        prog.deinit();
    }

    std.debug.print("Entry point: 0x{x:0>8}\n", .{program.entry_point});
    std.debug.print("Code size: {} bytes\n", .{program.bytecode.len});

    // Step 1: Preprocess to get proving/verifying keys
    std.debug.print("\nStep 1: Preprocessing...\n", .{});
    var preprocessor = zolt.host.Preprocessing(BN254Scalar).init(allocator);
    preprocessor.setMaxTraceLength(1024);

    var keys = try preprocessor.preprocess(&program);
    defer keys.pk.deinit();
    defer keys.vk.deinit();

    std.debug.print("  SRS degree: {}\n", .{keys.pk.srs.max_degree});
    std.debug.print("  Max trace length: {}\n", .{keys.pk.max_trace_length});

    // Step 2: Execute and collect trace
    std.debug.print("\nStep 2: Executing program...\n", .{});

    var config = zolt.common.MemoryConfig{
        .program_size = program.bytecode.len,
    };

    var emulator = zolt.tracer.Emulator.init(allocator, &config);
    defer emulator.deinit();

    try emulator.loadProgram(program.bytecode);
    emulator.state.pc = program.entry_point;
    emulator.max_cycles = 1024;

    var running = true;
    while (running) {
        running = emulator.step() catch break;
    }

    std.debug.print("  Cycles executed: {}\n", .{emulator.state.cycle});
    std.debug.print("  Trace entries: {}\n", .{emulator.trace.len()});

    // Step 3: Generate proof (experimental - shows the structure)
    std.debug.print("\nStep 3: Generating proof (experimental)...\n", .{});

    // Create the Jolt prover
    const prover = zolt.zkvm.JoltProver(BN254Scalar).init(allocator);

    // In a full implementation, we would:
    // 1. Convert emulator trace to JoltTrace
    // 2. Call prover.prove()
    // 3. Output the proof
    //
    // For now, we demonstrate the preprocessing and trace collection work.
    std.debug.print("  Prover initialized\n", .{});
    std.debug.print("  Note: Full proof generation is experimental\n", .{});

    // Demonstrate that we can access the commitment scheme
    std.debug.print("\nProof system components:\n", .{});
    std.debug.print("  - HyperKZG polynomial commitment\n", .{});
    std.debug.print("  - 6-stage multi-sumcheck prover\n", .{});
    std.debug.print("  - Lasso lookup argument\n", .{});
    std.debug.print("  - 24 lookup tables\n", .{});
    std.debug.print("  - 60+ instruction lookups\n", .{});

    _ = prover;

    std.debug.print("\nProving pipeline structure demonstrated successfully!\n", .{});
}

fn decodeInstruction(hex_str: []const u8) void {
    // Parse the hex instruction
    const instruction = std.fmt.parseInt(u32, hex_str, 0) catch {
        std.debug.print("Error: Invalid hex value: {s}\n", .{hex_str});
        std.debug.print("Example: zolt decode 0x00a00513\n", .{});
        return;
    };

    std.debug.print("Instruction: 0x{x:0>8}\n", .{instruction});

    // Decode using our RISC-V decoder
    const decoded = zolt.zkvm.instruction.DecodedInstruction.decode(instruction);

    std.debug.print("\nDecoded fields:\n", .{});
    std.debug.print("  Opcode:     0x{x:0>2} ({s})\n", .{ @intFromEnum(decoded.opcode), @tagName(decoded.opcode) });
    std.debug.print("  Format:     {s}\n", .{@tagName(decoded.format)});
    std.debug.print("  rd:         x{}\n", .{decoded.rd});
    std.debug.print("  rs1:        x{}\n", .{decoded.rs1});
    std.debug.print("  rs2:        x{}\n", .{decoded.rs2});
    std.debug.print("  funct3:     {}\n", .{decoded.funct3});
    std.debug.print("  funct7:     {}\n", .{decoded.funct7});
    std.debug.print("  immediate:  {d} (0x{x})\n", .{ decoded.imm, @as(u32, @bitCast(decoded.imm)) });
}

fn runBenchmarks() void {
    std.debug.print("Running benchmarks...\n\n", .{});

    // Field arithmetic benchmarks
    std.debug.print("=== Field Arithmetic (BN254 Scalar) ===\n", .{});

    const iterations: u32 = 10000;
    var a = BN254Scalar.fromU64(12345);
    const b = BN254Scalar.fromU64(67890);

    // Multiplication benchmark
    var timer = std.time.Timer.start() catch return;
    for (0..iterations) |_| {
        a = a.mul(b);
    }
    const mul_ns = timer.read();
    std.debug.print("  Multiplication: {} ns/op ({} ops)\n", .{ mul_ns / iterations, iterations });

    // Squaring benchmark
    timer.reset();
    for (0..iterations) |_| {
        a = a.square();
    }
    const sq_ns = timer.read();
    std.debug.print("  Squaring:       {} ns/op ({} ops)\n", .{ sq_ns / iterations, iterations });

    // Addition benchmark
    timer.reset();
    for (0..iterations) |_| {
        a = a.add(b);
    }
    const add_ns = timer.read();
    std.debug.print("  Addition:       {} ns/op ({} ops)\n", .{ add_ns / iterations, iterations });

    // Inverse benchmark (fewer iterations as it's expensive)
    const inv_iterations: u32 = 100;
    timer.reset();
    for (0..inv_iterations) |_| {
        if (a.inverse()) |inv| {
            a = inv;
        }
    }
    const inv_ns = timer.read();
    std.debug.print("  Inverse:        {} ns/op ({} ops)\n", .{ inv_ns / inv_iterations, inv_iterations });

    std.debug.print("\n=== Performance Ratios ===\n", .{});
    std.debug.print("  Mul/Add ratio:  {d:.1}x\n", .{@as(f64, @floatFromInt(mul_ns)) / @as(f64, @floatFromInt(add_ns))});
    std.debug.print("  Sq/Mul ratio:   {d:.2}x\n", .{@as(f64, @floatFromInt(sq_ns)) / @as(f64, @floatFromInt(mul_ns))});
    std.debug.print("  Inv/Mul ratio:  {d:.1}x\n", .{@as(f64, @floatFromInt(inv_ns)) / @as(f64, @floatFromInt(mul_ns)) * @as(f64, @floatFromInt(iterations)) / @as(f64, @floatFromInt(inv_iterations))});

    std.debug.print("\n=== RISC-V Instruction Decoding ===\n", .{});
    const decode_iterations: u32 = 100000;
    const test_instruction: u32 = 0x00a00513; // li a0, 10

    timer.reset();
    for (0..decode_iterations) |i| {
        const inst = test_instruction + @as(u32, @truncate(i & 0xFFF));
        const decoded = zolt.zkvm.instruction.DecodedInstruction.decode(inst);
        std.mem.doNotOptimizeAway(&decoded);
    }
    const decode_ns = timer.read();
    std.debug.print("  Decode:         {} ns/op ({} ops)\n", .{ decode_ns / decode_iterations, decode_iterations });

    std.debug.print("\n=== Summary ===\n", .{});
    std.debug.print("Montgomery mul optimized with CIOS algorithm\n", .{});
    std.debug.print("Squaring uses Karatsuba-like optimization (~25%% fewer muls)\n", .{});
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var args = try std.process.argsWithAllocator(allocator);
    defer args.deinit();

    // Skip program name
    _ = args.skip();

    // Get command
    const cmd_arg = args.next() orelse {
        printHelp();
        return;
    };

    const cmd = parseCommand(cmd_arg);

    switch (cmd) {
        .help => printHelp(),
        .version => printVersion(),
        .run => {
            if (args.next()) |elf_path| {
                try runEmulator(allocator, elf_path);
            } else {
                std.debug.print("Error: run command requires an ELF file path\n", .{});
                std.debug.print("Usage: zolt run <elf_file>\n", .{});
            }
        },
        .prove => {
            if (args.next()) |elf_path| {
                try runProver(allocator, elf_path);
            } else {
                std.debug.print("Error: prove command requires an ELF file path\n", .{});
                std.debug.print("Usage: zolt prove <elf_file>\n", .{});
            }
        },
        .decode => {
            if (args.next()) |hex_str| {
                decodeInstruction(hex_str);
            } else {
                std.debug.print("Error: decode command requires a hex instruction\n", .{});
                std.debug.print("Usage: zolt decode <hex>\n", .{});
                std.debug.print("Example: zolt decode 0x00a00513\n", .{});
            }
        },
        .bench => runBenchmarks(),
        .unknown => {
            std.debug.print("Unknown command: {s}\n\n", .{cmd_arg});
            printHelp();
        },
    }
}

test "zolt version" {
    const version = zolt.version;
    try std.testing.expect(version.len > 0);
}

test "command parsing" {
    try std.testing.expect(parseCommand("help") == .help);
    try std.testing.expect(parseCommand("-h") == .help);
    try std.testing.expect(parseCommand("--help") == .help);
    try std.testing.expect(parseCommand("version") == .version);
    try std.testing.expect(parseCommand("-v") == .version);
    try std.testing.expect(parseCommand("run") == .run);
    try std.testing.expect(parseCommand("prove") == .prove);
    try std.testing.expect(parseCommand("decode") == .decode);
    try std.testing.expect(parseCommand("bench") == .bench);
    try std.testing.expect(parseCommand("unknown_cmd") == .unknown);
}
