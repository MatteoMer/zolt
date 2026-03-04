//! Zolt CLI - A Zig port of the Jolt zkVM
//!
//! This executable provides command-line tools for:
//! - Proving and verifying RISC-V programs
//! - Running the RISC-V emulator

const std = @import("std");
const zolt = @import("root.zig");
const BN254Scalar = zolt.field.BN254Scalar;

const Command = enum {
    help,
    version,
    run,
    prove,
    verify,
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
        \\    help                Show this help message
        \\    version             Show version information
        \\    prove [opts] <elf>  Generate ZK proof for ELF binary
        \\    verify <proof>      Verify a proof file
        \\    run [opts] <elf>    Run RISC-V ELF binary in the emulator
        \\
        \\EXAMPLES:
        \\    zolt prove -o proof.bin program.elf         # Generate and save a proof
        \\    zolt verify proof.bin                       # Verify a saved proof
        \\    zolt run program.elf                        # Execute a RISC-V binary
        \\    zolt run --trace program.elf                # Show execution trace
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
    } else if (std.mem.eql(u8, arg, "verify")) {
        return .verify;
    }
    return .unknown;
}

/// Parse hex string (with optional 0x prefix) into bytes.
fn parseHexInput(allocator: std.mem.Allocator, hex: []const u8) ?[]u8 {
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
        buf[i] = std.fmt.parseInt(u8, clean_hex[start..end], 16) catch 0;
    }
    return buf;
}

fn runEmulator(allocator: std.mem.Allocator, elf_path: []const u8, show_regs: bool, show_trace: bool, max_trace_steps: ?usize, input_bytes: ?[]const u8) !void {
    std.debug.print("Loading ELF: {s}\n", .{elf_path});

    // Load the ELF file
    var loader = zolt.host.ELFLoader.init(allocator);
    const program = loader.loadFile(elf_path) catch |err| {
        return err;
    };
    defer {
        var prog = program;
        prog.deinit();
    }

    std.debug.print("Entry point: 0x{x:0>8}\n", .{program.entry_point});
    std.debug.print("Code size: {} bytes\n", .{program.bytecode.len});
    std.debug.print("Base address: 0x{x:0>8}\n", .{program.base_address});

    var config = zolt.common.MemoryConfig{
        .program_size = program.bytecode.len,
        .memory_size = 32768,
    };

    var emulator = zolt.tracer.Emulator.init(allocator, &config);
    defer emulator.deinit();

    try emulator.loadProgramAt(program.bytecode, program.base_address);

    if (input_bytes) |inputs| {
        try emulator.setInputs(inputs);
        std.debug.print("Input bytes: {} bytes\n", .{inputs.len});
        std.debug.print("Input region: 0x{x:0>16} - 0x{x:0>16}\n", .{
            emulator.device.memory_layout.input_start,
            emulator.device.memory_layout.input_end,
        });
    }

    emulator.state.pc = program.entry_point;

    if (show_trace) {
        // Trace mode: run and display execution trace
        std.debug.print("\n", .{});
        var running = true;
        while (running) {
            running = emulator.step() catch break;
        }

        const max_steps = max_trace_steps orelse 100;
        const total_steps = emulator.trace.len();
        const display_steps = @min(total_steps, max_steps);

        std.debug.print("=== Execution Trace ({} of {} steps) ===\n\n", .{ display_steps, total_steps });
        std.debug.print("{s:>6} | {s:>10} | {s:>10} | {s:>12} | {s}\n", .{ "Cycle", "PC", "Instr", "RD Value", "Disasm" });
        std.debug.print("{s:-<6}-+-{s:-<10}-+-{s:-<10}-+-{s:-<12}-+-{s:-<30}\n", .{ "", "", "", "", "" });

        for (0..display_steps) |i| {
            if (emulator.trace.get(i)) |step| {
                const decoded = zolt.zkvm.instruction.DecodedInstruction.decode(step.instruction);
                const mnemonic = blk: {
                    switch (decoded.opcode) {
                        .LUI => break :blk "LUI",
                        .AUIPC => break :blk "AUIPC",
                        .JAL => break :blk "JAL",
                        .JALR => break :blk "JALR",
                        .BRANCH => break :blk "BRANCH",
                        .LOAD => break :blk "LOAD",
                        .STORE => break :blk "STORE",
                        .OP_IMM => break :blk "OP_IMM",
                        .OP => break :blk "OP",
                        .FENCE => break :blk "FENCE",
                        .SYSTEM => break :blk "SYSTEM",
                        .OP_IMM_32 => break :blk "OP_IMM_32",
                        .OP_32 => break :blk "OP_32",
                        _ => break :blk "???",
                    }
                };

                var rd_buf: [16]u8 = undefined;
                const rd_str = if (step.rd_value != 0)
                    std.fmt.bufPrint(&rd_buf, "0x{x:0>8}", .{step.rd_value}) catch "?"
                else
                    std.fmt.bufPrint(&rd_buf, "-", .{}) catch "?";

                var mem_buf: [40]u8 = undefined;
                const mem_str = if (step.memory_addr) |addr| mem_blk: {
                    const mem_val = step.memory_value orelse 0;
                    if (step.is_memory_write) {
                        break :mem_blk std.fmt.bufPrint(&mem_buf, "[0x{x}] <- 0x{x}", .{ addr, mem_val }) catch "";
                    } else {
                        break :mem_blk std.fmt.bufPrint(&mem_buf, "[0x{x}] -> 0x{x}", .{ addr, mem_val }) catch "";
                    }
                } else "";

                std.debug.print("{:>6} | 0x{x:0>8} | 0x{x:0>8} | {s:>12} | {s}", .{
                    step.cycle,
                    step.pc,
                    step.instruction,
                    rd_str,
                    mnemonic,
                });

                switch (decoded.format) {
                    .R => std.debug.print(" x{}, x{}, x{}", .{ decoded.rd, decoded.rs1, decoded.rs2 }),
                    .I => {
                        if (decoded.opcode == .LOAD or decoded.opcode == .JALR) {
                            std.debug.print(" x{}, {}(x{})", .{ decoded.rd, decoded.imm, decoded.rs1 });
                        } else {
                            std.debug.print(" x{}, x{}, {}", .{ decoded.rd, decoded.rs1, decoded.imm });
                        }
                    },
                    .S => std.debug.print(" x{}, {}(x{})", .{ decoded.rs2, decoded.imm, decoded.rs1 }),
                    .B => std.debug.print(" x{}, x{}, {}", .{ decoded.rs1, decoded.rs2, decoded.imm }),
                    .U, .J => std.debug.print(" x{}, 0x{x}", .{ decoded.rd, @as(u32, @bitCast(decoded.imm)) }),
                }

                if (mem_str.len > 0) {
                    std.debug.print("  ; {s}", .{mem_str});
                }

                std.debug.print("\n", .{});
            }
        }

        if (total_steps > max_steps) {
            std.debug.print("\n... {} more steps (use --max N to show more)\n", .{total_steps - max_steps});
        }

        std.debug.print("\n====================\n", .{});
        std.debug.print("Total cycles: {}\n", .{emulator.state.cycle});
        std.debug.print("Final PC: 0x{x:0>8}\n", .{emulator.state.pc});
    } else {
        // Normal run mode
        std.debug.print("\nStarting execution...\n", .{});

        emulator.run() catch |err| {
            std.debug.print("Execution stopped: {}\n", .{err});
        };

        std.debug.print("\nExecution complete!\n", .{});
        std.debug.print("Cycles executed: {}\n", .{emulator.state.cycle});
        std.debug.print("Final PC: 0x{x:0>8}\n", .{emulator.state.pc});
        std.debug.print("Trace entries: {}\n", .{emulator.trace.len()});

        if (show_regs) {
            std.debug.print("\nFinal Register State:\n", .{});
            var i: u8 = 0;
            while (i < 32) : (i += 1) {
                const val = emulator.registers.read(i) catch 0;
                if (val != 0) {
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
}

fn runProver(allocator: std.mem.Allocator, elf_path: []const u8, output_path: []const u8, trace_length_opt: ?u64, srs_path: ?[]const u8, preprocessing_path: ?[]const u8, input_bytes: ?[]const u8) !void {
    std.debug.print("Zolt zkVM Prover\n", .{});
    std.debug.print("================\n\n", .{});

    const trace_length = trace_length_opt orelse 1024;

    // Load the ELF file
    std.debug.print("Loading ELF: {s}\n", .{elf_path});
    var loader = zolt.host.ELFLoader.init(allocator);
    const program = loader.loadFile(elf_path) catch |err| {
        return err;
    };
    defer {
        var prog = program;
        prog.deinit();
    }

    std.debug.print("  Entry point: 0x{x:0>8}\n", .{program.entry_point});
    std.debug.print("  Code size: {} bytes\n", .{program.bytecode.len});
    std.debug.print("  Trace length: {}\n", .{trace_length});
    if (input_bytes) |inputs| {
        std.debug.print("  Input bytes: {} bytes\n", .{inputs.len});
    }

    // Step 1: Preprocess to get proving/verifying keys
    std.debug.print("\n[1/4] Preprocessing...\n", .{});
    var timer = std.time.Timer.start() catch return;

    var preprocessor = zolt.host.Preprocessing(BN254Scalar).init(allocator);
    preprocessor.setMaxTraceLength(trace_length);

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
    const zkvm_pk = zolt.zkvm.ProvingKey.fromSRS(keys.pk.srs);
    prover_inst.setProvingKey(zkvm_pk);

    const init_time = timer.read();
    std.debug.print("  Prover initialized with proving key\n", .{});
    std.debug.print("  Time: {d:.2} ms\n", .{@as(f64, @floatFromInt(init_time)) / 1_000_000.0});

    // Step 3: Generate Jolt-compatible proof with Dory commitments
    std.debug.print("\n[3/4] Generating proof...\n", .{});
    std.debug.print("  Running 6-stage multi-sumcheck protocol\n", .{});
    std.debug.print("  Components: HyperKZG, Lasso lookups, 24 tables\n", .{});
    timer.reset();

    std.debug.print("  Generating Jolt-compatible proof with Dory commitments...\n", .{});

    if (srs_path) |sp| {
        std.debug.print("  Using Jolt SRS from: {s}\n", .{sp});
    }

    var jolt_bundle = prover_inst.proveJoltCompatibleWithDoryAndSrsAtAddress(
        program.bytecode,
        input_bytes orelse &[_]u8{},
        srs_path,
        program.base_address,
        program.entry_point,
    ) catch |err| {
        std.debug.print("  Error generating Jolt-compatible proof: {s}\n", .{@errorName(err)});
        return err;
    };
    defer jolt_bundle.deinit();

    const prove_time = timer.read();
    std.debug.print("  Proof generated successfully!\n", .{});
    std.debug.print("  Time: {d:.2} ms\n", .{@as(f64, @floatFromInt(prove_time)) / 1_000_000.0});

    // Serialize using the bundled Dory commitments
    const jolt_bytes = prover_inst.serializeJoltProofWithDory(&jolt_bundle) catch |err| {
        std.debug.print("  Error serializing Jolt proof with Dory: {}\n", .{err});
        return err;
    };
    defer allocator.free(jolt_bytes);

    std.debug.print("\nSaving proof to: {s}\n", .{output_path});
    const file = std.fs.cwd().createFile(output_path, .{}) catch |err| {
        std.debug.print("  Error creating output file: {}\n", .{err});
        return err;
    };
    defer file.close();
    file.writeAll(jolt_bytes) catch |err| {
        std.debug.print("  Error writing Jolt proof: {}\n", .{err});
        return err;
    };

    std.debug.print("  Format: Jolt (Dory commitments, arkworks-compatible)\n", .{});
    std.debug.print("  Proof size: {} bytes ({d:.2} KB)\n", .{ jolt_bytes.len, @as(f64, @floatFromInt(jolt_bytes.len)) / 1024.0 });
    std.debug.print("  Proof saved successfully!\n", .{});

    // Export preprocessing if requested
    if (preprocessing_path) |pp_path| {
        std.debug.print("\nExporting preprocessing to: {s}\n", .{pp_path});

        const preprocessing = zolt.zkvm.preprocessing;
        const jolt_device = zolt.zkvm.jolt_device;

        const device = jolt_device.JoltDevice.fromEmulator(
            allocator,
            &[_]u8{},
            &[_]u8{},
            false,
            @intCast(program.bytecode.len),
            32768,
        ) catch |err| {
            std.debug.print("  Error creating memory layout: {s}\n", .{@errorName(err)});
            return err;
        };
        var device_mut = device;
        defer device_mut.deinit();

        std.debug.print("  Termination address: 0x{x:0>16}\n", .{device.memory_layout.termination});
        var bytecode_prep = preprocessing.BytecodePreprocessing.preprocess(allocator, program.bytecode, program.entry_point, device.memory_layout.termination) catch |err| {
            std.debug.print("  Error generating bytecode preprocessing: {s}\n", .{@errorName(err)});
            return err;
        };

        const mem_init_entries = try allocator.alloc(struct { u64, u8 }, program.bytecode.len);
        defer allocator.free(mem_init_entries);
        for (program.bytecode, 0..) |byte, i| {
            mem_init_entries[i] = .{ program.entry_point + i, byte };
        }

        var ram_prep = preprocessing.RAMPreprocessing.preprocess(allocator, mem_init_entries) catch |err| {
            std.debug.print("  Error generating RAM preprocessing: {s}\n", .{@errorName(err)});
            bytecode_prep.deinit();
            return err;
        };

        var shared_prep = preprocessing.JoltSharedPreprocessing{
            .bytecode = bytecode_prep,
            .ram = ram_prep,
            .memory_layout = device.memory_layout,
            .max_padded_trace_length = keys.pk.max_trace_length,
        };
        defer shared_prep.deinit();

        const dory = zolt.poly.commitment.dory;
        const DoryCommitmentScheme = dory.DoryCommitmentScheme(zolt.field.BN254Scalar);

        const proof_log_size = jolt_bundle.dory_srs_log_size;
        std.debug.print("  Using SRS log_size={} from proof generation\n", .{proof_log_size});
        var srs = blk: {
            if (srs_path) |srs_file| {
                if (DoryCommitmentScheme.loadFromFile(allocator, srs_file)) |loaded| {
                    break :blk loaded;
                } else |_| {
                    std.debug.print("  Warning: Could not load SRS for verifier setup\n", .{});
                    std.debug.print("  Generating default SRS...\n", .{});
                }
            }
            break :blk DoryCommitmentScheme.setup(allocator, proof_log_size) catch |err| {
                std.debug.print("  Error generating SRS: {s}\n", .{@errorName(err)});
                return err;
            };
        };
        defer srs.deinit();

        var verifier_setup = preprocessing.DoryVerifierSetup.fromSRS(allocator, &srs) catch |err| {
            std.debug.print("  Error creating verifier setup: {s}\n", .{@errorName(err)});
            return err;
        };
        defer verifier_setup.deinit();

        var buffer = std.ArrayListUnmanaged(u8){};
        defer buffer.deinit(allocator);

        verifier_setup.serialize(buffer.writer(allocator)) catch |err| {
            std.debug.print("  Error serializing verifier setup: {s}\n", .{@errorName(err)});
            return err;
        };

        shared_prep.serialize(allocator, buffer.writer(allocator)) catch |err| {
            std.debug.print("  Error serializing shared preprocessing: {s}\n", .{@errorName(err)});
            return err;
        };

        {
            const writer = buffer.writer(allocator);
            try writer.writeAll("ZOLT_RAW\n");
            const raw_words = bytecode_prep.raw_words.items;
            try writer.writeInt(u64, @intCast(raw_words.len), .little);
            for (raw_words) |w| {
                try writer.writeInt(u32, w, .little);
            }
            try writer.writeInt(u64, @intCast(bytecode_prep.pc_map.termination_base_pc), .little);
            std.debug.print("  Appended {} raw instruction words (termination_base_pc={})\n", .{ raw_words.len, bytecode_prep.pc_map.termination_base_pc });
        }

        const pp_file = std.fs.cwd().createFile(pp_path, .{}) catch |err| {
            std.debug.print("  Error creating preprocessing file: {s}\n", .{@errorName(err)});
            return err;
        };
        defer pp_file.close();

        pp_file.writeAll(buffer.items) catch |err| {
            std.debug.print("  Error writing preprocessing: {s}\n", .{@errorName(err)});
            return err;
        };

        std.debug.print("  Preprocessing exported successfully! ({} bytes)\n", .{buffer.items.len});
        std.debug.print("  This file can be loaded by Jolt for cross-verification.\n", .{});

        {
            var ram_buffer = std.ArrayListUnmanaged(u8){};
            defer ram_buffer.deinit(allocator);
            const ram_writer = ram_buffer.writer(allocator);

            try ram_prep.serialize(ram_writer);
            try preprocessing.serializeMemoryLayout(&device.memory_layout, ram_writer);

            const bytecode_K_for_export = zolt.zkvm.computeBytecodeCodeSize(program.bytecode);
            try ram_writer.writeInt(u64, @intCast(bytecode_K_for_export), .little);
            std.debug.print("  bytecode code_size (bytecode_K): {}\n", .{bytecode_K_for_export});

            try ram_writer.writeInt(u64, program.entry_point, .little);
            try ram_writer.writeInt(u64, @intCast(program.bytecode.len), .little);
            try ram_writer.writeAll(program.bytecode);
            std.debug.print("  Exported {} raw program bytes (base=0x{x})\n", .{ program.bytecode.len, program.entry_point });

            try ram_writer.writeInt(u64, @intCast(bytecode_prep.pc_map.termination_base_pc), .little);
            std.debug.print("  termination_base_pc: {}\n", .{bytecode_prep.pc_map.termination_base_pc});

            const ram_path = try std.fmt.allocPrint(allocator, "{s}.ram", .{pp_path});
            defer allocator.free(ram_path);
            const ram_file = try std.fs.cwd().createFile(ram_path, .{});
            defer ram_file.close();
            try ram_file.writeAll(ram_buffer.items);
            std.debug.print("  RAM preprocessing exported to: {s} ({} bytes)\n", .{ ram_path, ram_buffer.items.len });
        }
    }

    const total_time = timer.read();
    std.debug.print("\nTotal time: {d:.2} ms\n", .{@as(f64, @floatFromInt(total_time)) / 1_000_000.0});
}

fn runVerifier(allocator: std.mem.Allocator, proof_path: []const u8) !void {
    std.debug.print("Zolt zkVM Verifier\n", .{});
    std.debug.print("==================\n\n", .{});

    // Load the proof file
    std.debug.print("Loading proof: {s}\n", .{proof_path});
    var timer = std.time.Timer.start() catch return;

    // Detect format first
    const file = std.fs.cwd().openFile(proof_path, .{}) catch |err| {
        std.debug.print("  Error opening proof file: {}\n", .{err});
        return err;
    };
    var header_buf: [64]u8 = undefined;
    const bytes_read = file.readAll(&header_buf) catch |err| {
        std.debug.print("  Error reading proof file: {}\n", .{err});
        file.close();
        return err;
    };
    file.close();

    const format = zolt.zkvm.detectProofFormat(header_buf[0..bytes_read]);
    std.debug.print("  Format: {s}\n", .{format.toString()});

    // Load the full file for auto-detection
    const proof_file = std.fs.cwd().openFile(proof_path, .{}) catch |err| {
        std.debug.print("  Error opening proof file: {}\n", .{err});
        return err;
    };
    defer proof_file.close();
    const stat = try proof_file.stat();
    const data = try allocator.alloc(u8, stat.size);
    defer allocator.free(data);
    _ = try proof_file.readAll(data);

    var proof = zolt.zkvm.readProofAutoDetectFull(BN254Scalar, allocator, data) catch |err| {
        std.debug.print("  Error loading proof: {}\n", .{err});
        return err;
    };
    defer proof.deinit();

    const load_time = timer.read();
    std.debug.print("  Proof loaded successfully!\n", .{});
    std.debug.print("  Load time: {d:.2} ms\n", .{@as(f64, @floatFromInt(load_time)) / 1_000_000.0});

    // Display proof info
    std.debug.print("\nProof Information:\n", .{});
    std.debug.print("  Bytecode commitment: {s}\n", .{if (!proof.bytecode_proof.commitment.isZero()) "present" else "none"});
    std.debug.print("  Memory commitment: {s}\n", .{if (!proof.memory_proof.commitment.isZero()) "present" else "none"});
    std.debug.print("  Register commitment: {s}\n", .{if (!proof.register_proof.commitment.isZero()) "present" else "none"});
    std.debug.print("  Stage proofs: {s}\n", .{if (proof.stage_proofs != null) "present" else "none"});

    if (proof.stage_proofs) |stage_proofs| {
        const size = stage_proofs.proofSize();
        std.debug.print("  Total field elements: {}\n", .{size.total_elements});
        std.debug.print("  Round polynomials: {}\n", .{size.round_polys});
        std.debug.print("  log_t: {}, log_k: {}\n", .{ stage_proofs.log_t, stage_proofs.log_k });
    }

    // Verify the proof
    std.debug.print("\nVerifying proof...\n", .{});
    timer.reset();

    var verifier = zolt.zkvm.JoltVerifier(BN254Scalar).init(allocator);
    verifier.setVerifyingKey(zolt.zkvm.VerifyingKey.init());

    const verify_result = verifier.verify(&proof, &[_]u8{}) catch |err| {
        std.debug.print("  Error during verification: {}\n", .{err});
        return err;
    };

    const verify_time = timer.read();
    std.debug.print("\n==================\n", .{});
    if (verify_result) {
        std.debug.print("Result: PASSED\n", .{});
    } else {
        std.debug.print("Result: FAILED\n", .{});
    }
    std.debug.print("Verification time: {d:.2} ms\n", .{@as(f64, @floatFromInt(verify_time)) / 1_000_000.0});
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
            if (args.next()) |arg| {
                if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
                    std.debug.print("Usage: zolt run [options] <elf_file>\n\n", .{});
                    std.debug.print("Run a RISC-V ELF binary in the emulator.\n", .{});
                    std.debug.print("The emulator supports RV64IMC instructions.\n\n", .{});
                    std.debug.print("Options:\n", .{});
                    std.debug.print("  --regs           Show final register state\n", .{});
                    std.debug.print("  --trace          Show execution trace\n", .{});
                    std.debug.print("  --max N          Max trace entries to display (default: 100)\n", .{});
                    std.debug.print("  --input FILE     Load input bytes from FILE\n", .{});
                    std.debug.print("  --input-hex HEX  Set input as hex bytes (e.g., 0x32 for input 50)\n", .{});
                } else {
                    var elf_path: ?[]const u8 = null;
                    var show_regs = false;
                    var show_trace = false;
                    var max_trace_steps: ?usize = null;
                    var input_file: ?[]const u8 = null;
                    var input_hex: ?[]const u8 = null;

                    if (std.mem.startsWith(u8, arg, "--")) {
                        if (std.mem.eql(u8, arg, "--regs")) {
                            show_regs = true;
                        } else if (std.mem.eql(u8, arg, "--trace")) {
                            show_trace = true;
                        } else if (std.mem.eql(u8, arg, "--max")) {
                            if (args.next()) |n_str| {
                                max_trace_steps = std.fmt.parseInt(usize, n_str, 10) catch null;
                            }
                        } else if (std.mem.eql(u8, arg, "--input")) {
                            input_file = args.next();
                        } else if (std.mem.eql(u8, arg, "--input-hex")) {
                            input_hex = args.next();
                        }
                    } else {
                        elf_path = arg;
                    }

                    while (args.next()) |next_arg| {
                        if (std.mem.startsWith(u8, next_arg, "--")) {
                            if (std.mem.eql(u8, next_arg, "--regs")) {
                                show_regs = true;
                            } else if (std.mem.eql(u8, next_arg, "--trace")) {
                                show_trace = true;
                            } else if (std.mem.eql(u8, next_arg, "--max")) {
                                if (args.next()) |n_str| {
                                    max_trace_steps = std.fmt.parseInt(usize, n_str, 10) catch null;
                                }
                            } else if (std.mem.eql(u8, next_arg, "--input")) {
                                input_file = args.next();
                            } else if (std.mem.eql(u8, next_arg, "--input-hex")) {
                                input_hex = args.next();
                            }
                        } else if (elf_path == null) {
                            elf_path = next_arg;
                        }
                    }

                    var input_bytes_owned: ?[]u8 = null;
                    defer if (input_bytes_owned) |b| allocator.free(b);

                    if (input_file) |path| {
                        const f = std.fs.cwd().openFile(path, .{}) catch |err| {
                            std.debug.print("Failed to open input file: {s}\n", .{@errorName(err)});
                            std.process.exit(1);
                        };
                        defer f.close();
                        const stat = f.stat() catch |err| {
                            std.debug.print("Failed to stat input file: {s}\n", .{@errorName(err)});
                            std.process.exit(1);
                        };
                        input_bytes_owned = allocator.alloc(u8, stat.size) catch null;
                        if (input_bytes_owned) |buf| {
                            _ = f.readAll(buf) catch {
                                std.debug.print("Failed to read input file\n", .{});
                                std.process.exit(1);
                            };
                        }
                    } else if (input_hex) |hex| {
                        input_bytes_owned = parseHexInput(allocator, hex);
                    }

                    if (elf_path) |path| {
                        runEmulator(allocator, path, show_regs, show_trace, max_trace_steps, input_bytes_owned) catch |err| {
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
                    std.debug.print("Usage: zolt prove [options] -o <output> <elf_file>\n\n", .{});
                    std.debug.print("Generate a ZK proof for a RISC-V ELF binary.\n", .{});
                    std.debug.print("This command runs the full proving pipeline:\n", .{});
                    std.debug.print("  1. Preprocess (generate SRS and keys)\n", .{});
                    std.debug.print("  2. Initialize prover\n", .{});
                    std.debug.print("  3. Generate proof using multi-stage sumcheck\n", .{});
                    std.debug.print("  4. Save proof to file\n\n", .{});
                    std.debug.print("Options:\n", .{});
                    std.debug.print("  -o, --output F           Save proof to file F (required)\n", .{});
                    std.debug.print("  --trace-length N         Set trace length for proof system (default: 1024)\n", .{});
                    std.debug.print("  --srs PATH               Use Dory SRS from PATH (exported by Jolt)\n", .{});
                    std.debug.print("  --export-preprocessing P Export Jolt-compatible preprocessing to file P\n", .{});
                    std.debug.print("  --input-hex HEX          Set input as hex bytes (e.g., 20 for input 32)\n", .{});
                } else {
                    var elf_path: ?[]const u8 = null;
                    var trace_length: ?u64 = null;
                    var output_path: ?[]const u8 = null;
                    var srs_path: ?[]const u8 = null;
                    var preprocessing_path: ?[]const u8 = null;
                    var input_hex: ?[]const u8 = null;

                    if (std.mem.startsWith(u8, arg, "-")) {
                        if (std.mem.eql(u8, arg, "--trace-length")) {
                            if (args.next()) |len_str| {
                                trace_length = std.fmt.parseInt(u64, len_str, 10) catch null;
                            }
                        } else if (std.mem.eql(u8, arg, "-o") or std.mem.eql(u8, arg, "--output")) {
                            output_path = args.next();
                        } else if (std.mem.eql(u8, arg, "--srs")) {
                            srs_path = args.next();
                        } else if (std.mem.eql(u8, arg, "--export-preprocessing")) {
                            preprocessing_path = args.next();
                        } else if (std.mem.eql(u8, arg, "--input-hex")) {
                            input_hex = args.next();
                        }
                    } else {
                        elf_path = arg;
                    }

                    while (args.next()) |next_arg| {
                        if (std.mem.startsWith(u8, next_arg, "-")) {
                            if (std.mem.eql(u8, next_arg, "--trace-length")) {
                                if (args.next()) |len_str| {
                                    trace_length = std.fmt.parseInt(u64, len_str, 10) catch null;
                                }
                            } else if (std.mem.eql(u8, next_arg, "-o") or std.mem.eql(u8, next_arg, "--output")) {
                                output_path = args.next();
                            } else if (std.mem.eql(u8, next_arg, "--srs")) {
                                srs_path = args.next();
                            } else if (std.mem.eql(u8, next_arg, "--export-preprocessing")) {
                                preprocessing_path = args.next();
                            } else if (std.mem.eql(u8, next_arg, "--input-hex")) {
                                input_hex = args.next();
                            }
                        } else if (elf_path == null) {
                            elf_path = next_arg;
                        }
                    }

                    var input_bytes_owned: ?[]u8 = null;
                    defer if (input_bytes_owned) |b| allocator.free(b);

                    if (input_hex) |hex| {
                        input_bytes_owned = parseHexInput(allocator, hex);
                    }

                    if (output_path == null) {
                        std.debug.print("Error: prove command requires an output path (-o <file>)\n", .{});
                        std.debug.print("Usage: zolt prove -o <output> [options] <elf_file>\n", .{});
                        return;
                    }

                    if (elf_path) |path| {
                        runProver(allocator, path, output_path.?, trace_length, srs_path, preprocessing_path, input_bytes_owned) catch |err| {
                            std.debug.print("Failed to generate proof: {s}\n", .{@errorName(err)});
                            std.process.exit(1);
                        };
                    } else {
                        std.debug.print("Error: prove command requires an ELF file path\n", .{});
                        std.debug.print("Usage: zolt prove -o <output> [options] <elf_file>\n", .{});
                    }
                }
            } else {
                std.debug.print("Error: prove command requires an ELF file path\n", .{});
                std.debug.print("Usage: zolt prove -o <output> [options] <elf_file>\n", .{});
            }
        },
        .verify => {
            if (args.next()) |arg| {
                if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
                    std.debug.print("Usage: zolt verify <proof_file>\n\n", .{});
                    std.debug.print("Verify a Zolt proof file.\n", .{});
                    std.debug.print("The proof file should be created with 'zolt prove -o <file>'.\n\n", .{});
                    std.debug.print("Example:\n", .{});
                    std.debug.print("  zolt prove -o proof.bin program.elf\n", .{});
                    std.debug.print("  zolt verify proof.bin\n", .{});
                } else {
                    runVerifier(allocator, arg) catch |err| {
                        std.debug.print("Failed to verify proof: {s}\n", .{@errorName(err)});
                        std.process.exit(1);
                    };
                }
            } else {
                std.debug.print("Error: verify command requires a proof file path\n", .{});
                std.debug.print("Usage: zolt verify <proof_file>\n", .{});
            }
        },
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
    try std.testing.expect(parseCommand("verify") == .verify);
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
