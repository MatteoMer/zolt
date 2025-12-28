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
    info,
    run,
    prove,
    srs,
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
        \\    info              Show zkVM capabilities and feature summary
        \\    run [opts] <elf>   Run RISC-V ELF binary in the emulator
        \\    prove <elf>       Generate ZK proof for ELF binary (experimental)
        \\    srs <ptau>        Inspect a Powers of Tau (ptau) file
        \\    decode <hex>      Decode a RISC-V instruction (hex)
        \\    bench             Run performance benchmarks
        \\
        \\EXAMPLES:
        \\    zolt run program.elf        # Execute a RISC-V binary
        \\    zolt prove program.elf      # Generate a ZK proof
        \\    zolt srs file.ptau          # Inspect a PTAU file
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

fn printInfo() void {
    std.debug.print(
        \\Zolt zkVM - Capabilities and Features
        \\======================================
        \\
        \\Version: {s}
        \\License: MIT (porting a16z/jolt, also MIT)
        \\
        \\PROOF SYSTEM:
        \\  Commitment Scheme:    HyperKZG (polynomial commitments)
        \\  Backend:              Spartan (R1CS-based zkSNARK)
        \\  Lookup Arguments:     Lasso (efficient lookups via sumcheck)
        \\  Sumcheck Protocol:    6-stage multi-sumcheck
        \\  Field:                BN254 Scalar (254 bits, ~21 bytes)
        \\
        \\SUMCHECK STAGES:
        \\  1. Outer Spartan     - R1CS constraint verification (degree 3)
        \\  2. RAM RAF           - Read-after-final memory checking (degree 2)
        \\  3. Lasso Lookup      - Instruction lookup verification (degree 2)
        \\  4. Value Evaluation  - Memory consistency (degree 3)
        \\  5. Register          - Register file correctness (degree 2)
        \\  6. Booleanity        - Flag constraint checking (degree 2)
        \\
        \\RISC-V SUPPORT:
        \\  ISA:                  RV32IM / RV64IMC
        \\  Supported Extensions: M (multiply), C (compressed)
        \\  Register Width:       32/64 bits
        \\  Registers:            32 general-purpose (x0-x31)
        \\
        \\LOOKUP TABLES (24 types):
        \\  Bitwise:              AND, OR, XOR, ZERO_EXTEND, SIGN_EXTEND
        \\  Comparison:           LT, LTU, EQ (less than, unsigned, equal)
        \\  Shift:                SLL, SRL, SRA (left/right/arithmetic)
        \\  Arithmetic:           ADD, SUB, MUL, MULH, MULHSU, MULHU
        \\  Division:             DIV, DIVU, REM, REMU
        \\  Utility:              RANGE_CHECK, IDENTITY
        \\
        \\INSTRUCTION FAMILIES (~60 instructions):
        \\  R-type:               ADD, SUB, SLL, SLT, SLTU, XOR, SRL, SRA, OR, AND
        \\  I-type:               ADDI, SLTI, SLTIU, XORI, ORI, ANDI, SLLI, SRLI, SRAI
        \\  Load:                 LB, LH, LW, LBU, LHU, LD, LWU
        \\  Store:                SB, SH, SW, SD
        \\  Branch:               BEQ, BNE, BLT, BGE, BLTU, BGEU
        \\  Jump:                 JAL, JALR
        \\  Upper-imm:            LUI, AUIPC
        \\  M-extension:          MUL, MULH, MULHSU, MULHU, DIV, DIVU, REM, REMU
        \\  C-extension:          16-bit compressed variants
        \\
        \\MEMORY MODEL:
        \\  Address Space:        32-bit / 64-bit addressable
        \\  Memory Regions:       Code, Stack, Heap
        \\  Memory Checking:      RAF (Read-After-Final) verification
        \\
        \\PERFORMANCE (approximate, single-threaded):
        \\  Field multiplication: ~50 ns/op
        \\  Field inversion:      ~13 us/op
        \\  MSM (256 points):     ~0.5 ms/op
        \\  HyperKZG commit:      ~1.5 ms/1024 coefficients
        \\  Proving (2 steps):    ~97 ms
        \\  Verification:         ~600 us (130-165x faster than proving)
        \\  Proof size:           ~6-8 KB (depends on trace length)
        \\
        \\ELF LOADER:
        \\  Formats:              ELF32, ELF64
        \\  Endianness:           Little-endian (RISC-V default)
        \\  Entry Points:         Automatic detection
        \\
        \\For usage: zolt help
        \\
    , .{zolt.version});
}

fn parseCommand(arg: []const u8) Command {
    if (std.mem.eql(u8, arg, "help") or std.mem.eql(u8, arg, "-h") or std.mem.eql(u8, arg, "--help")) {
        return .help;
    } else if (std.mem.eql(u8, arg, "version") or std.mem.eql(u8, arg, "-v") or std.mem.eql(u8, arg, "--version")) {
        return .version;
    } else if (std.mem.eql(u8, arg, "info")) {
        return .info;
    } else if (std.mem.eql(u8, arg, "run")) {
        return .run;
    } else if (std.mem.eql(u8, arg, "prove")) {
        return .prove;
    } else if (std.mem.eql(u8, arg, "srs")) {
        return .srs;
    } else if (std.mem.eql(u8, arg, "decode")) {
        return .decode;
    } else if (std.mem.eql(u8, arg, "bench")) {
        return .bench;
    }
    return .unknown;
}

fn runEmulator(allocator: std.mem.Allocator, elf_path: []const u8, max_cycles: ?u64, show_regs: bool) !void {
    std.debug.print("Loading ELF: {s}\n", .{elf_path});

    // Load the ELF file
    var loader = zolt.host.ELFLoader.init(allocator);
    const program = loader.loadFile(elf_path) catch |err| {
        return err; // Error will be handled by caller
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

    const cycle_limit = max_cycles orelse 16 * 1024 * 1024; // Default 16M cycles
    std.debug.print("Max cycles: {}\n", .{cycle_limit});
    std.debug.print("\nStarting execution...\n", .{});

    // Run the emulator step by step
    var running = true;
    while (running and emulator.state.cycle < cycle_limit) {
        running = emulator.step() catch |err| {
            std.debug.print("Execution stopped: {}\n", .{err});
            break;
        };
    }

    if (emulator.state.cycle >= cycle_limit) {
        std.debug.print("\nCycle limit reached!\n", .{});
    } else {
        std.debug.print("\nExecution complete!\n", .{});
    }
    std.debug.print("Cycles executed: {}\n", .{emulator.state.cycle});
    std.debug.print("Final PC: 0x{x:0>8}\n", .{emulator.state.pc});
    std.debug.print("Trace entries: {}\n", .{emulator.trace.len()});

    // Show final register state if requested
    if (show_regs) {
        std.debug.print("\nFinal Register State:\n", .{});
        var i: u8 = 0;
        while (i < 32) : (i += 1) {
            const val = emulator.state.registers[i];
            if (val != 0) { // Only show non-zero registers
                const reg_name = switch (i) {
                    0 => "zero",
                    1 => "ra  ",
                    2 => "sp  ",
                    3 => "gp  ",
                    4 => "tp  ",
                    5 => "t0  ",
                    6 => "t1  ",
                    7 => "t2  ",
                    8 => "s0  ",
                    9 => "s1  ",
                    10 => "a0  ",
                    11 => "a1  ",
                    12 => "a2  ",
                    13 => "a3  ",
                    14 => "a4  ",
                    15 => "a5  ",
                    16 => "a6  ",
                    17 => "a7  ",
                    18 => "s2  ",
                    19 => "s3  ",
                    20 => "s4  ",
                    21 => "s5  ",
                    22 => "s6  ",
                    23 => "s7  ",
                    24 => "s8  ",
                    25 => "s9  ",
                    26 => "s10 ",
                    27 => "s11 ",
                    28 => "t3  ",
                    29 => "t4  ",
                    30 => "t5  ",
                    31 => "t6  ",
                    else => "x?? ",
                };
                std.debug.print("  x{d:0>2} ({s}): 0x{x:0>16} ({d})\n", .{ i, reg_name, val, val });
            }
        }
    }
}

fn runProver(allocator: std.mem.Allocator, elf_path: []const u8) !void {
    std.debug.print("Zolt zkVM Prover\n", .{});
    std.debug.print("================\n\n", .{});

    // Load the ELF file
    std.debug.print("Loading ELF: {s}\n", .{elf_path});
    var loader = zolt.host.ELFLoader.init(allocator);
    const program = loader.loadFile(elf_path) catch |err| {
        return err; // Error will be handled by caller
    };
    defer {
        var prog = program;
        prog.deinit();
    }

    std.debug.print("  Entry point: 0x{x:0>8}\n", .{program.entry_point});
    std.debug.print("  Code size: {} bytes\n", .{program.bytecode.len});

    // Step 1: Preprocess to get proving/verifying keys
    std.debug.print("\n[1/4] Preprocessing...\n", .{});
    var timer = std.time.Timer.start() catch return;

    var preprocessor = zolt.host.Preprocessing(BN254Scalar).init(allocator);
    preprocessor.setMaxTraceLength(1024);

    var keys = try preprocessor.preprocess(&program);
    defer keys.pk.deinit();
    defer keys.vk.deinit();

    const preprocess_time = timer.read();
    std.debug.print("  SRS degree: {}\n", .{keys.pk.srs.max_degree});
    std.debug.print("  Max trace length: {}\n", .{keys.pk.max_trace_length});
    std.debug.print("  Time: {d:.2} ms\n", .{@as(f64, @floatFromInt(preprocess_time)) / 1_000_000.0});

    // Step 2: Create prover with proving key
    std.debug.print("\n[2/4] Initializing prover...\n", .{});
    timer.reset();

    var prover_inst = zolt.zkvm.JoltProver(BN254Scalar).init(allocator);
    prover_inst.setMaxCycles(1024);
    // Convert host ProvingKey to zkvm ProvingKey
    const zkvm_pk = zolt.zkvm.ProvingKey.fromSRS(keys.pk.srs);
    prover_inst.setProvingKey(zkvm_pk);

    const init_time = timer.read();
    std.debug.print("  Prover initialized with proving key\n", .{});
    std.debug.print("  Time: {d:.2} ms\n", .{@as(f64, @floatFromInt(init_time)) / 1_000_000.0});

    // Step 3: Generate proof
    std.debug.print("\n[3/4] Generating proof...\n", .{});
    std.debug.print("  Running 6-stage multi-sumcheck protocol\n", .{});
    std.debug.print("  Components: HyperKZG, Lasso lookups, 24 tables\n", .{});
    timer.reset();

    var proof = prover_inst.prove(program.bytecode, &[_]u8{}) catch |err| {
        std.debug.print("  Error generating proof: {}\n", .{err});
        return err;
    };
    defer proof.deinit();

    const prove_time = timer.read();
    std.debug.print("  Proof generated successfully!\n", .{});
    std.debug.print("  Time: {d:.2} ms\n", .{@as(f64, @floatFromInt(prove_time)) / 1_000_000.0});

    // Step 4: Verify proof
    std.debug.print("\n[4/4] Verifying proof...\n", .{});
    timer.reset();

    var verifier = zolt.zkvm.JoltVerifier(BN254Scalar).init(allocator);
    // Convert host VerifyingKey to zkvm VerifyingKey
    const zkvm_vk = zolt.zkvm.VerifyingKey{
        .g1 = keys.vk.g1,
        .g2 = keys.vk.g2,
        .tau_g2 = keys.vk.tau_g2,
    };
    verifier.setVerifyingKey(zkvm_vk);

    const verify_result = verifier.verify(&proof, &[_]u8{}) catch |err| {
        std.debug.print("  Error during verification: {}\n", .{err});
        return err;
    };

    const verify_time = timer.read();
    if (verify_result) {
        std.debug.print("  Verification: PASSED\n", .{});
    } else {
        std.debug.print("  Verification: FAILED\n", .{});
    }
    std.debug.print("  Time: {d:.2} ms\n", .{@as(f64, @floatFromInt(verify_time)) / 1_000_000.0});

    // Summary
    std.debug.print("\n================\n", .{});
    std.debug.print("Proof Summary\n", .{});
    std.debug.print("================\n", .{});
    std.debug.print("  Bytecode commitment: {s}\n", .{if (!proof.bytecode_proof.commitment.isZero()) "present" else "none"});
    std.debug.print("  Memory commitment: {s}\n", .{if (!proof.memory_proof.commitment.isZero()) "present" else "none"});
    std.debug.print("  Register commitment: {s}\n", .{if (!proof.register_proof.commitment.isZero()) "present" else "none"});
    std.debug.print("  Stage proofs: {s}\n", .{if (proof.stage_proofs != null) "present" else "none"});

    const total_time = preprocess_time + init_time + prove_time + verify_time;
    std.debug.print("\nTotal time: {d:.2} ms\n", .{@as(f64, @floatFromInt(total_time)) / 1_000_000.0});
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

fn inspectSRS(allocator: std.mem.Allocator, ptau_path: []const u8) !void {
    std.debug.print("SRS File Inspector\n", .{});
    std.debug.print("==================\n\n", .{});

    std.debug.print("Loading PTAU file: {s}\n", .{ptau_path});

    // Load the PTAU file
    const file = std.fs.cwd().openFile(ptau_path, .{}) catch |err| {
        return err; // Error will be handled by caller
    };
    defer file.close();

    const stat = try file.stat();
    std.debug.print("  File size: {} bytes ({d:.2} MB)\n", .{ stat.size, @as(f64, @floatFromInt(stat.size)) / (1024.0 * 1024.0) });

    // Read file contents
    const data = try allocator.alloc(u8, stat.size);
    defer allocator.free(data);

    const bytes_read = try file.readAll(data);
    if (bytes_read != stat.size) {
        std.debug.print("Warning: Only read {} of {} bytes\n", .{ bytes_read, stat.size });
    }

    // Parse the PTAU file
    std.debug.print("\nParsing PTAU format...\n", .{});
    var srs = zolt.poly.commitment.srs.loadFromPtau(allocator, data) catch |err| {
        return err; // Error will be handled by caller
    };
    defer srs.deinit();

    // Display SRS information
    std.debug.print("\n==================\n", .{});
    std.debug.print("SRS Information\n", .{});
    std.debug.print("==================\n", .{});
    std.debug.print("  Power: 2^{d} = {} points\n", .{ srs.power, @as(u64, 1) << @intCast(srs.power) });
    std.debug.print("  Ceremony power: 2^{d}\n", .{srs.ceremony_power});
    std.debug.print("  G1 points: {}\n", .{srs.powers_of_tau_g1.len});
    std.debug.print("  G2 points: {}\n", .{srs.powers_of_tau_g2.len});
    std.debug.print("  Alpha*tau G1 points: {}\n", .{if (srs.alpha_tau_g1) |alpha| alpha.len else 0});
    std.debug.print("  Beta*tau G1 points: {}\n", .{if (srs.beta_tau_g1) |beta| beta.len else 0});

    // Check if we have a valid SRS
    if (srs.powers_of_tau_g1.len > 0) {
        const g1 = srs.powers_of_tau_g1[0];
        std.debug.print("\nFirst G1 point (should be generator):\n", .{});
        std.debug.print("  x: 0x", .{});
        const x_bytes = g1.x.toBytesBE();
        for (x_bytes) |byte| {
            std.debug.print("{x:0>2}", .{byte});
        }
        std.debug.print("\n  y: 0x", .{});
        const y_bytes = g1.y.toBytesBE();
        for (y_bytes) |byte| {
            std.debug.print("{x:0>2}", .{byte});
        }
        std.debug.print("\n", .{});

        // Verify it's on the curve
        if (g1.isOnCurve()) {
            std.debug.print("  Status: On curve (valid)\n", .{});
        } else {
            std.debug.print("  Status: NOT on curve (invalid!)\n", .{});
        }
    }

    // Calculate usable max degree
    const max_degree = srs.powers_of_tau_g1.len;
    std.debug.print("\nUsage:\n", .{});
    std.debug.print("  Max polynomial degree: {}\n", .{max_degree});
    std.debug.print("  Suitable for trace length up to: {} cycles\n", .{max_degree / 2});

    std.debug.print("\nSRS inspection complete.\n", .{});
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
        .info => printInfo(),
        .run => {
            if (args.next()) |arg| {
                if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
                    std.debug.print("Usage: zolt run [options] <elf_file>\n\n", .{});
                    std.debug.print("Run a RISC-V ELF binary in the emulator.\n", .{});
                    std.debug.print("The emulator supports RV64IMC instructions.\n\n", .{});
                    std.debug.print("Options:\n", .{});
                    std.debug.print("  --max-cycles N   Limit execution to N cycles (default: 16M)\n", .{});
                    std.debug.print("  --regs           Show final register state\n", .{});
                } else {
                    // Parse options
                    var elf_path: ?[]const u8 = null;
                    var max_cycles: ?u64 = null;
                    var show_regs = false;

                    // First arg could be an option or the ELF path
                    if (std.mem.startsWith(u8, arg, "--")) {
                        if (std.mem.eql(u8, arg, "--regs")) {
                            show_regs = true;
                        } else if (std.mem.eql(u8, arg, "--max-cycles")) {
                            if (args.next()) |cycles_str| {
                                max_cycles = std.fmt.parseInt(u64, cycles_str, 10) catch null;
                            }
                        }
                    } else {
                        elf_path = arg;
                    }

                    // Parse remaining args
                    while (args.next()) |next_arg| {
                        if (std.mem.startsWith(u8, next_arg, "--")) {
                            if (std.mem.eql(u8, next_arg, "--regs")) {
                                show_regs = true;
                            } else if (std.mem.eql(u8, next_arg, "--max-cycles")) {
                                if (args.next()) |cycles_str| {
                                    max_cycles = std.fmt.parseInt(u64, cycles_str, 10) catch null;
                                }
                            }
                        } else if (elf_path == null) {
                            elf_path = next_arg;
                        }
                    }

                    if (elf_path) |path| {
                        runEmulator(allocator, path, max_cycles, show_regs) catch |err| {
                            std.debug.print("Failed to run program: {s}\n", .{@errorName(err)});
                            std.process.exit(1);
                        };
                    } else {
                        std.debug.print("Error: run command requires an ELF file path\n", .{});
                        std.debug.print("Usage: zolt run [options] <elf_file>\n", .{});
                    }
                }
            } else {
                std.debug.print("Error: run command requires an ELF file path\n", .{});
                std.debug.print("Usage: zolt run [options] <elf_file>\n", .{});
            }
        },
        .prove => {
            if (args.next()) |arg| {
                if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
                    std.debug.print("Usage: zolt prove <elf_file>\n\n", .{});
                    std.debug.print("Generate a ZK proof for a RISC-V ELF binary.\n", .{});
                    std.debug.print("This experimental command runs the full proving pipeline:\n", .{});
                    std.debug.print("  1. Preprocess (generate SRS and keys)\n", .{});
                    std.debug.print("  2. Initialize prover\n", .{});
                    std.debug.print("  3. Generate proof using multi-stage sumcheck\n", .{});
                    std.debug.print("  4. Verify the proof\n", .{});
                } else {
                    runProver(allocator, arg) catch |err| {
                        std.debug.print("Failed to generate proof: {s}\n", .{@errorName(err)});
                        std.process.exit(1);
                    };
                }
            } else {
                std.debug.print("Error: prove command requires an ELF file path\n", .{});
                std.debug.print("Usage: zolt prove <elf_file>\n", .{});
            }
        },
        .srs => {
            if (args.next()) |arg| {
                if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
                    std.debug.print("Usage: zolt srs <ptau_file>\n\n", .{});
                    std.debug.print("Inspect a Powers of Tau (PTAU) ceremony file.\n", .{});
                    std.debug.print("PTAU files are used to provide the trusted setup (SRS)\n", .{});
                    std.debug.print("for polynomial commitment schemes like KZG and HyperKZG.\n", .{});
                } else {
                    inspectSRS(allocator, arg) catch |err| {
                        std.debug.print("Failed to inspect SRS: {s}\n", .{@errorName(err)});
                        std.process.exit(1);
                    };
                }
            } else {
                std.debug.print("Error: srs command requires a PTAU file path\n", .{});
                std.debug.print("Usage: zolt srs <ptau_file>\n", .{});
            }
        },
        .decode => {
            if (args.next()) |arg| {
                if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
                    std.debug.print("Usage: zolt decode <hex>\n\n", .{});
                    std.debug.print("Decode a RISC-V instruction from its hex encoding.\n", .{});
                    std.debug.print("Example: zolt decode 0x00a00513  (decodes: li a0, 10)\n", .{});
                } else {
                    decodeInstruction(arg);
                }
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
    try std.testing.expect(parseCommand("info") == .info);
    try std.testing.expect(parseCommand("run") == .run);
    try std.testing.expect(parseCommand("prove") == .prove);
    try std.testing.expect(parseCommand("decode") == .decode);
    try std.testing.expect(parseCommand("bench") == .bench);
    try std.testing.expect(parseCommand("unknown_cmd") == .unknown);
}
