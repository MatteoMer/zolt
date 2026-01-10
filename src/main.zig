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
    trace,
    prove,
    verify,
    stats,
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
        \\    trace <elf>       Show execution trace (for debugging)
        \\    prove [opts] <elf> Generate ZK proof for ELF binary
        \\    verify <proof>    Verify a proof file
        \\    stats <proof>     Show detailed proof statistics
        \\    srs <ptau>        Inspect a Powers of Tau (ptau) file
        \\    decode <hex>      Decode a RISC-V instruction (hex)
        \\    bench             Run performance benchmarks
        \\
        \\EXAMPLES:
        \\    zolt run program.elf                    # Execute a RISC-V binary
        \\    zolt trace program.elf                  # Show execution trace
        \\    zolt prove -o proof.bin program.elf     # Generate and save a proof
        \\    zolt verify proof.bin                   # Verify a saved proof
        \\    zolt stats proof.bin                    # Show proof statistics
        \\    zolt srs file.ptau                      # Inspect a PTAU file
        \\    zolt decode 0x00a00513                  # Decode: li a0, 10
        \\    zolt bench                              # Run benchmarks
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
    } else if (std.mem.eql(u8, arg, "trace")) {
        return .trace;
    } else if (std.mem.eql(u8, arg, "prove")) {
        return .prove;
    } else if (std.mem.eql(u8, arg, "verify")) {
        return .verify;
    } else if (std.mem.eql(u8, arg, "stats")) {
        return .stats;
    } else if (std.mem.eql(u8, arg, "srs")) {
        return .srs;
    } else if (std.mem.eql(u8, arg, "decode")) {
        return .decode;
    } else if (std.mem.eql(u8, arg, "bench")) {
        return .bench;
    }
    return .unknown;
}

fn runEmulator(allocator: std.mem.Allocator, elf_path: []const u8, show_regs: bool, input_bytes: ?[]const u8) !void {
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

    // Create memory config matching Jolt's fibonacci example settings
    // This uses memory_size = 32KB to match Jolt-compiled guest's hardcoded addresses
    var config = zolt.common.MemoryConfig{
        .program_size = program.bytecode.len,
        .memory_size = 32768, // Match Jolt fibonacci's memory_size
    };

    // Create emulator
    var emulator = zolt.tracer.Emulator.init(allocator, &config);
    defer emulator.deinit();

    // Load program into memory at the correct base address
    try emulator.loadProgramAt(program.bytecode, program.base_address);

    // Set up inputs if provided
    if (input_bytes) |inputs| {
        try emulator.setInputs(inputs);
        std.debug.print("Input bytes: {} bytes\n", .{inputs.len});
        std.debug.print("Input region: 0x{x:0>16} - 0x{x:0>16}\n", .{
            emulator.device.memory_layout.input_start,
            emulator.device.memory_layout.input_end,
        });
    }

    // Set entry point PC
    emulator.state.pc = program.entry_point;

    std.debug.print("\nStarting execution...\n", .{});

    // Run the emulator until termination
    emulator.run() catch |err| {
        std.debug.print("Execution stopped: {}\n", .{err});
    };

    std.debug.print("\nExecution complete!\n", .{});
    std.debug.print("Cycles executed: {}\n", .{emulator.state.cycle});
    std.debug.print("Final PC: 0x{x:0>8}\n", .{emulator.state.pc});
    std.debug.print("Trace entries: {}\n", .{emulator.trace.len()});

    // Show final register state if requested
    if (show_regs) {
        std.debug.print("\nFinal Register State:\n", .{});
        var i: u8 = 0;
        while (i < 32) : (i += 1) {
            // Read from the RegisterFile, not the VMState
            const val = emulator.registers.read(i) catch 0;
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

fn runProver(allocator: std.mem.Allocator, elf_path: []const u8, trace_length_opt: ?u64, output_path: ?[]const u8, json_format: bool, jolt_format: bool, srs_path: ?[]const u8, preprocessing_path: ?[]const u8, input_bytes: ?[]const u8) !void {
    // srs_path: Optional path to a Jolt-exported Dory SRS file.
    // When provided, uses the same SRS as Jolt for exact commitment compatibility.
    std.debug.print("Zolt zkVM Prover\n", .{});
    std.debug.print("================\n\n", .{});

    const trace_length = trace_length_opt orelse 1024;

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

    // For jolt_format, use the Jolt-compatible proving path directly
    // This generates a proof with Dory commitments in Jolt's format
    if (jolt_format) {
        if (output_path) |path| {
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

            std.debug.print("\nSaving proof to: {s}\n", .{path});
            const file = std.fs.cwd().createFile(path, .{}) catch |err| {
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

                // Generate preprocessing using the same bytecode
                const preprocessing = zolt.zkvm.preprocessing;
                var bytecode_prep = preprocessing.BytecodePreprocessing.preprocess(allocator, program.bytecode, program.entry_point) catch |err| {
                    std.debug.print("  Error generating bytecode preprocessing: {s}\n", .{@errorName(err)});
                    return err;
                };

                // Create memory init from bytecode
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

                // Create memory layout
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
                    bytecode_prep.deinit();
                    ram_prep.deinit();
                    return err;
                };
                var device_mut = device;
                defer device_mut.deinit();

                // Create shared preprocessing (transfer ownership)
                var shared_prep = preprocessing.JoltSharedPreprocessing{
                    .bytecode = bytecode_prep,
                    .ram = ram_prep,
                    .memory_layout = device.memory_layout,
                };
                defer shared_prep.deinit();

                // Create verifier setup from SRS if available
                const dory = zolt.poly.commitment.dory;
                const DoryCommitmentScheme = dory.DoryCommitmentScheme(zolt.field.BN254Scalar);

                // Generate or load SRS for verifier setup
                var srs = blk: {
                    if (srs_path) |srs_file| {
                        if (DoryCommitmentScheme.loadFromFile(allocator, srs_file)) |loaded| {
                            break :blk loaded;
                        } else |_| {
                            std.debug.print("  Warning: Could not load SRS for verifier setup\n", .{});
                            std.debug.print("  Generating default SRS (may not match Jolt exactly)...\n", .{});
                        }
                    }
                    // Generate default SRS
                    break :blk DoryCommitmentScheme.setup(allocator, 20) catch |err| {
                        std.debug.print("  Error generating SRS: {s}\n", .{@errorName(err)});
                        return err;
                    };
                };
                defer srs.deinit();

                // Create verifier setup from SRS
                var verifier_setup = preprocessing.DoryVerifierSetup.fromSRS(allocator, &srs) catch |err| {
                    std.debug.print("  Error creating verifier setup: {s}\n", .{@errorName(err)});
                    return err;
                };
                defer verifier_setup.deinit();

                // Serialize to ArrayList first, then write to file
                var buffer = std.ArrayListUnmanaged(u8){};
                defer buffer.deinit(allocator);

                // Serialize generators (DoryVerifierSetup)
                verifier_setup.serialize(buffer.writer(allocator)) catch |err| {
                    std.debug.print("  Error serializing verifier setup: {s}\n", .{@errorName(err)});
                    return err;
                };

                // Serialize shared preprocessing
                shared_prep.serialize(allocator, buffer.writer(allocator)) catch |err| {
                    std.debug.print("  Error serializing shared preprocessing: {s}\n", .{@errorName(err)});
                    return err;
                };

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
            }

            const total_time = timer.read();
            std.debug.print("\nTotal time: {d:.2} ms\n", .{@as(f64, @floatFromInt(total_time)) / 1_000_000.0});
            return;
        } else {
            std.debug.print("  Error: --jolt-format requires an output path (-o)\n", .{});
            return;
        }
    }

    // Standard Zolt proof generation path
    var proof = prover_inst.prove(program.bytecode, input_bytes orelse &[_]u8{}) catch |err| {
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

    const verify_result = verifier.verify(&proof, input_bytes orelse &[_]u8{}) catch |err| {
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

    // Save proof to file if output path specified (non-jolt-format)
    if (output_path) |path| {
        std.debug.print("\nSaving proof to: {s}\n", .{path});

        if (json_format) {
            // Save as JSON
            zolt.zkvm.writeProofToJsonFile(BN254Scalar, allocator, proof, path) catch |err| {
                std.debug.print("  Error saving proof: {}\n", .{err});
                return err;
            };

            // Calculate proof size
            const json_bytes = try zolt.zkvm.serializeProofToJson(BN254Scalar, allocator, proof);
            defer allocator.free(json_bytes);
            std.debug.print("  Format: JSON\n", .{});
            std.debug.print("  Proof size: {} bytes ({d:.2} KB)\n", .{ json_bytes.len, @as(f64, @floatFromInt(json_bytes.len)) / 1024.0 });
        } else {
            // Save as binary
            zolt.zkvm.writeProofToFile(BN254Scalar, allocator, proof, path) catch |err| {
                std.debug.print("  Error saving proof: {}\n", .{err});
                return err;
            };

            // Calculate proof size
            const proof_bytes = try zolt.zkvm.serializeProof(BN254Scalar, allocator, proof);
            defer allocator.free(proof_bytes);
            std.debug.print("  Format: Binary\n", .{});
            std.debug.print("  Proof size: {} bytes ({d:.2} KB)\n", .{ proof_bytes.len, @as(f64, @floatFromInt(proof_bytes.len)) / 1024.0 });
        }
        std.debug.print("  Proof saved successfully!\n", .{});
    }

    // Export preprocessing if requested
    if (preprocessing_path) |pp_path| {
        std.debug.print("\nExporting preprocessing to: {s}\n", .{pp_path});

        // Generate preprocessing using the same bytecode
        const preprocessing = zolt.zkvm.preprocessing;
        var bytecode_prep = preprocessing.BytecodePreprocessing.preprocess(allocator, program.bytecode, program.entry_point) catch |err| {
            std.debug.print("  Error generating bytecode preprocessing: {s}\n", .{@errorName(err)});
            return err;
        };

        // Create memory init from bytecode
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

        // Create memory layout
        // Use memory_size = 32768 to match Jolt fibonacci example
        const jolt_device = zolt.zkvm.jolt_device;
        const device = jolt_device.JoltDevice.fromEmulator(
            allocator,
            &[_]u8{},
            &[_]u8{},
            false,
            @intCast(program.bytecode.len),
            32768, // Match Jolt fibonacci's memory_size
        ) catch |err| {
            std.debug.print("  Error creating memory layout: {s}\n", .{@errorName(err)});
            bytecode_prep.deinit();
            ram_prep.deinit();
            return err;
        };
        var device_mut = device;
        defer device_mut.deinit();

        // Create shared preprocessing (transfer ownership)
        var shared_prep = preprocessing.JoltSharedPreprocessing{
            .bytecode = bytecode_prep,
            .ram = ram_prep,
            .memory_layout = device.memory_layout,
        };
        defer shared_prep.deinit();

        // Create verifier setup from SRS if available
        const dory = zolt.poly.commitment.dory;
        const DoryCommitmentScheme = dory.DoryCommitmentScheme(zolt.field.BN254Scalar);

        // Generate or load SRS for verifier setup
        var srs = blk: {
            if (srs_path) |srs_file| {
                if (DoryCommitmentScheme.loadFromFile(allocator, srs_file)) |loaded| {
                    break :blk loaded;
                } else |_| {
                    std.debug.print("  Warning: Could not load SRS for verifier setup\n", .{});
                    std.debug.print("  Generating default SRS (may not match Jolt exactly)...\n", .{});
                }
            }
            // Generate default SRS
            break :blk DoryCommitmentScheme.setup(allocator, 20) catch |err| {
                std.debug.print("  Error generating SRS: {s}\n", .{@errorName(err)});
                return err;
            };
        };
        defer srs.deinit();

        // Create verifier setup from SRS
        var verifier_setup = preprocessing.DoryVerifierSetup.fromSRS(allocator, &srs) catch |err| {
            std.debug.print("  Error creating verifier setup: {s}\n", .{@errorName(err)});
            return err;
        };
        defer verifier_setup.deinit();

        // Create full verifier preprocessing (don't own shared - just reference)
        // Note: We need to serialize generators first, then shared preprocessing
        // to match Jolt's JoltVerifierPreprocessing format

        // Serialize to ArrayList first, then write to file
        var buffer = std.ArrayListUnmanaged(u8){};
        defer buffer.deinit(allocator);

        // Serialize generators (DoryVerifierSetup)
        verifier_setup.serialize(buffer.writer(allocator)) catch |err| {
            std.debug.print("  Error serializing verifier setup: {s}\n", .{@errorName(err)});
            return err;
        };

        // Serialize shared preprocessing
        shared_prep.serialize(allocator, buffer.writer(allocator)) catch |err| {
            std.debug.print("  Error serializing shared preprocessing: {s}\n", .{@errorName(err)});
            return err;
        };

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
    }

    const total_time = preprocess_time + init_time + prove_time + verify_time;
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
    // Use default verifying key
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

fn showTrace(allocator: std.mem.Allocator, elf_path: []const u8, max_steps_opt: ?usize, input_bytes: ?[]const u8) !void {
    std.debug.print("Zolt Execution Trace\n", .{});
    std.debug.print("====================\n\n", .{});

    const max_steps = max_steps_opt orelse 100;

    // Load the ELF file
    std.debug.print("Loading ELF: {s}\n", .{elf_path});
    var loader = zolt.host.ELFLoader.init(allocator);
    const program = loader.loadFile(elf_path) catch |err| {
        std.debug.print("Error loading ELF: {}\n", .{err});
        return err;
    };
    defer {
        var prog = program;
        prog.deinit();
    }

    std.debug.print("Entry point: 0x{x:0>8}\n", .{program.entry_point});
    std.debug.print("Code size: {} bytes\n\n", .{program.bytecode.len});

    // Create memory config matching Jolt's fibonacci example settings
    var config = zolt.common.MemoryConfig{
        .program_size = program.bytecode.len,
        .memory_size = 32768, // Match Jolt fibonacci's memory_size
    };

    // Create emulator
    var emulator = zolt.tracer.Emulator.init(allocator, &config);
    defer emulator.deinit();

    // Load program into memory at the correct base address
    try emulator.loadProgramAt(program.bytecode, program.base_address);

    // Set up inputs if provided
    if (input_bytes) |inputs| {
        try emulator.setInputs(inputs);
        std.debug.print("Input bytes: {} bytes = {{ ", .{inputs.len});
        for (inputs) |b| {
            std.debug.print("{x:0>2} ", .{b});
        }
        std.debug.print("}}\n", .{});
        std.debug.print("Input region: 0x{x:0>16} - 0x{x:0>16}\n\n", .{
            emulator.device.memory_layout.input_start,
            emulator.device.memory_layout.input_end,
        });
    }

    // Set entry point PC
    emulator.state.pc = program.entry_point;

    // Run and collect trace
    var running = true;
    while (running) {
        running = emulator.step() catch break;
    }

    const total_steps = emulator.trace.len();
    const display_steps = @min(total_steps, max_steps);

    std.debug.print("=== Execution Trace ({} of {} steps) ===\n\n", .{ display_steps, total_steps });
    std.debug.print("{s:>6} | {s:>10} | {s:>10} | {s:>12} | {s}\n", .{ "Cycle", "PC", "Instr", "RD Value", "Disasm" });
    std.debug.print("{s:-<6}-+-{s:-<10}-+-{s:-<10}-+-{s:-<12}-+-{s:-<30}\n", .{ "", "", "", "", "" });

    for (0..display_steps) |i| {
        if (emulator.trace.get(i)) |step| {
            // Decode instruction for disassembly
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

            // Format RD value (only show if not zero)
            var rd_buf: [16]u8 = undefined;
            const rd_str = if (step.rd_value != 0)
                std.fmt.bufPrint(&rd_buf, "0x{x:0>8}", .{step.rd_value}) catch "?"
            else
                std.fmt.bufPrint(&rd_buf, "-", .{}) catch "?";

            // Format memory access if present
            var mem_buf: [40]u8 = undefined;
            const mem_str = if (step.memory_addr) |addr| blk: {
                const mem_val = step.memory_value orelse 0;
                if (step.is_memory_write) {
                    break :blk std.fmt.bufPrint(&mem_buf, "[0x{x}] <- 0x{x}", .{ addr, mem_val }) catch "";
                } else {
                    break :blk std.fmt.bufPrint(&mem_buf, "[0x{x}] -> 0x{x}", .{ addr, mem_val }) catch "";
                }
            } else "";

            // Print the trace line
            std.debug.print("{:>6} | 0x{x:0>8} | 0x{x:0>8} | {s:>12} | {s}", .{
                step.cycle,
                step.pc,
                step.instruction,
                rd_str,
                mnemonic,
            });

            // Add operand info based on format
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

            // Add memory access info
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
}

fn showProofStats(allocator: std.mem.Allocator, proof_path: []const u8) !void {
    std.debug.print("Zolt Proof Statistics\n", .{});
    std.debug.print("=====================\n\n", .{});

    // Get file info
    const file = std.fs.cwd().openFile(proof_path, .{}) catch |err| {
        std.debug.print("Error opening proof file: {}\n", .{err});
        return err;
    };
    const file_stat = try file.stat();
    var header_buf: [64]u8 = undefined;
    const bytes_read = try file.readAll(&header_buf);
    file.close();

    const format = zolt.zkvm.detectProofFormat(header_buf[0..bytes_read]);

    std.debug.print("File: {s}\n", .{proof_path});
    std.debug.print("Format: {s}\n", .{format.toString()});
    std.debug.print("File size: {} bytes ({d:.2} KB)\n", .{ file_stat.size, @as(f64, @floatFromInt(file_stat.size)) / 1024.0 });

    // Load proof (with full file data for gzip support)
    std.debug.print("\nLoading proof...\n", .{});
    var timer = std.time.Timer.start() catch return;

    const proof_file = std.fs.cwd().openFile(proof_path, .{}) catch |err| {
        std.debug.print("Error opening proof file: {}\n", .{err});
        return err;
    };
    defer proof_file.close();
    const data = try allocator.alloc(u8, file_stat.size);
    defer allocator.free(data);
    _ = try proof_file.readAll(data);

    var proof = zolt.zkvm.readProofAutoDetectFull(BN254Scalar, allocator, data) catch |err| {
        std.debug.print("Error loading proof: {}\n", .{err});
        return err;
    };
    defer proof.deinit();

    const load_time = timer.read();
    std.debug.print("Load time: {d:.2} ms\n", .{@as(f64, @floatFromInt(load_time)) / 1_000_000.0});

    // Section 1: Commitments
    std.debug.print("\n--- COMMITMENTS ---\n", .{});
    std.debug.print("Bytecode commitment:  {s}\n", .{if (!proof.bytecode_proof.commitment.isZero()) "present" else "none"});
    std.debug.print("Memory commitment:    {s}\n", .{if (!proof.memory_proof.commitment.isZero()) "present" else "none"});
    std.debug.print("Register commitment:  {s}\n", .{if (!proof.register_proof.commitment.isZero()) "present" else "none"});

    // Section 2: R1CS Proof
    std.debug.print("\n--- R1CS PROOF ---\n", .{});
    std.debug.print("Tau point size: {}\n", .{proof.r1cs_proof.tau.len});
    std.debug.print("Eval point size: {}\n", .{proof.r1cs_proof.eval_point.len});

    // Section 3: Stage Proofs
    if (proof.stage_proofs) |stage_proofs| {
        std.debug.print("\n--- STAGE PROOFS (Sumcheck) ---\n", .{});
        std.debug.print("log_t (trace length): 2^{} = {} steps\n", .{ stage_proofs.log_t, @as(u64, 1) << @intCast(stage_proofs.log_t) });
        std.debug.print("log_k (address space): 2^{} = {} addresses\n", .{ stage_proofs.log_k, @as(u64, 1) << @intCast(stage_proofs.log_k) });

        const size = stage_proofs.proofSize();
        std.debug.print("\nOverall statistics:\n", .{});
        std.debug.print("  Total field elements: {}\n", .{size.total_elements});
        std.debug.print("  Round polynomials: {}\n", .{size.round_polys});
        std.debug.print("  Polynomial coefficients: {}\n", .{size.poly_coeffs});
        std.debug.print("  Challenges: {}\n", .{size.challenges});
        std.debug.print("  Final claims: {}\n", .{size.claims});

        const stage_names = [_][]const u8{
            "Stage 1 (Spartan)",
            "Stage 2 (RAF)",
            "Stage 3 (Lasso)",
            "Stage 4 (Value)",
            "Stage 5 (Register)",
            "Stage 6 (Booleanity)",
        };

        std.debug.print("\nPer-stage breakdown:\n", .{});
        for (stage_proofs.stage_proofs, 0..) |stage, i| {
            var stage_coeffs: usize = 0;
            for (stage.round_polys.items) |poly| {
                stage_coeffs += poly.len;
            }
            std.debug.print("  {s:20}: {} rounds, {} coeffs, {} claims\n", .{
                stage_names[i],
                stage.round_polys.items.len,
                stage_coeffs,
                stage.final_claims.items.len,
            });
        }

        // Calculate estimated size breakdown
        const field_element_bytes = 32; // 256-bit field elements
        const estimated_stage_bytes = size.total_elements * field_element_bytes;
        const commitment_bytes: usize = 3 * 64; // 3 commitments, 64 bytes each (G1 point)

        std.debug.print("\n--- SIZE BREAKDOWN (estimated) ---\n", .{});
        std.debug.print("Commitments: ~{} bytes\n", .{commitment_bytes});
        std.debug.print("Stage proofs: ~{} bytes ({} field elements)\n", .{ estimated_stage_bytes, size.total_elements });
        std.debug.print("Overhead (headers, metadata): ~{} bytes\n", .{file_stat.size -| (estimated_stage_bytes + commitment_bytes)});
    } else {
        std.debug.print("\n--- STAGE PROOFS ---\n", .{});
        std.debug.print("No stage proofs present (lightweight proof)\n", .{});
    }

    std.debug.print("\n=====================\n", .{});
    std.debug.print("Statistics complete.\n", .{});
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
                    std.debug.print("  --regs           Show final register state\n", .{});
                    std.debug.print("  --input FILE     Load input bytes from FILE\n", .{});
                    std.debug.print("  --input-hex HEX  Set input as hex bytes (e.g., 0x32 for input 50)\n", .{});
                } else {
                    // Parse options
                    var elf_path: ?[]const u8 = null;
                    var show_regs = false;
                    var input_file: ?[]const u8 = null;
                    var input_hex: ?[]const u8 = null;

                    // First arg could be an option or the ELF path
                    if (std.mem.startsWith(u8, arg, "--")) {
                        if (std.mem.eql(u8, arg, "--regs")) {
                            show_regs = true;
                        } else if (std.mem.eql(u8, arg, "--input")) {
                            input_file = args.next();
                        } else if (std.mem.eql(u8, arg, "--input-hex")) {
                            input_hex = args.next();
                        }
                    } else {
                        elf_path = arg;
                    }

                    // Parse remaining args
                    while (args.next()) |next_arg| {
                        if (std.mem.startsWith(u8, next_arg, "--")) {
                            if (std.mem.eql(u8, next_arg, "--regs")) {
                                show_regs = true;
                            } else if (std.mem.eql(u8, next_arg, "--input")) {
                                input_file = args.next();
                            } else if (std.mem.eql(u8, next_arg, "--input-hex")) {
                                input_hex = args.next();
                            }
                        } else if (elf_path == null) {
                            elf_path = next_arg;
                        }
                    }

                    // Load input bytes if specified
                    var input_bytes_owned: ?[]u8 = null;
                    defer if (input_bytes_owned) |b| allocator.free(b);

                    if (input_file) |path| {
                        const file = std.fs.cwd().openFile(path, .{}) catch |err| {
                            std.debug.print("Failed to open input file: {s}\n", .{@errorName(err)});
                            std.process.exit(1);
                        };
                        defer file.close();
                        const stat = file.stat() catch |err| {
                            std.debug.print("Failed to stat input file: {s}\n", .{@errorName(err)});
                            std.process.exit(1);
                        };
                        input_bytes_owned = allocator.alloc(u8, stat.size) catch null;
                        if (input_bytes_owned) |buf| {
                            _ = file.readAll(buf) catch {
                                std.debug.print("Failed to read input file\n", .{});
                                std.process.exit(1);
                            };
                        }
                    } else if (input_hex) |hex| {
                        // Parse hex input (e.g., "32" for byte 50, or "00320000" for multiple bytes)
                        var clean_hex = hex;
                        if (std.mem.startsWith(u8, hex, "0x") or std.mem.startsWith(u8, hex, "0X")) {
                            clean_hex = hex[2..];
                        }
                        const buf_len = (clean_hex.len + 1) / 2;
                        input_bytes_owned = allocator.alloc(u8, buf_len) catch null;
                        if (input_bytes_owned) |buf| {
                            // Parse hex bytes
                            var i: usize = 0;
                            while (i < buf_len) : (i += 1) {
                                const start = i * 2;
                                const end = @min(start + 2, clean_hex.len);
                                buf[i] = std.fmt.parseInt(u8, clean_hex[start..end], 16) catch 0;
                            }
                        }
                    }

                    if (elf_path) |path| {
                        runEmulator(allocator, path, show_regs, input_bytes_owned) catch |err| {
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
        .trace => {
            if (args.next()) |arg| {
                if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
                    std.debug.print("Usage: zolt trace [options] <elf_file>\n\n", .{});
                    std.debug.print("Show the execution trace of a RISC-V ELF binary.\n", .{});
                    std.debug.print("Displays each instruction with PC, opcode, and results.\n\n", .{});
                    std.debug.print("Options:\n", .{});
                    std.debug.print("  --max N           Show at most N trace entries (default: 100)\n", .{});
                    std.debug.print("  --input-hex HEX   Set input as hex bytes (e.g., 03 for input 3)\n", .{});
                } else {
                    var elf_path: ?[]const u8 = null;
                    var max_steps: ?usize = null;
                    var input_hex: ?[]const u8 = null;

                    // First arg could be an option or the ELF path
                    if (std.mem.startsWith(u8, arg, "--")) {
                        if (std.mem.eql(u8, arg, "--max")) {
                            if (args.next()) |n_str| {
                                max_steps = std.fmt.parseInt(usize, n_str, 10) catch null;
                            }
                        } else if (std.mem.eql(u8, arg, "--input-hex")) {
                            input_hex = args.next();
                        }
                    } else {
                        elf_path = arg;
                    }

                    // Parse remaining args
                    while (args.next()) |next_arg| {
                        if (std.mem.startsWith(u8, next_arg, "--")) {
                            if (std.mem.eql(u8, next_arg, "--max")) {
                                if (args.next()) |n_str| {
                                    max_steps = std.fmt.parseInt(usize, n_str, 10) catch null;
                                }
                            } else if (std.mem.eql(u8, next_arg, "--input-hex")) {
                                input_hex = args.next();
                            }
                        } else if (elf_path == null) {
                            elf_path = next_arg;
                        }
                    }

                    // Parse hex input if provided
                    var input_bytes_owned: ?[]u8 = null;
                    defer if (input_bytes_owned) |b| allocator.free(b);

                    if (input_hex) |hex| {
                        var clean_hex = hex;
                        if (std.mem.startsWith(u8, hex, "0x") or std.mem.startsWith(u8, hex, "0X")) {
                            clean_hex = hex[2..];
                        }
                        const buf_len = (clean_hex.len + 1) / 2;
                        input_bytes_owned = allocator.alloc(u8, buf_len) catch null;
                        if (input_bytes_owned) |buf| {
                            for (0..buf_len) |i| {
                                const start = i * 2;
                                const end = @min(start + 2, clean_hex.len);
                                buf[i] = std.fmt.parseInt(u8, clean_hex[start..end], 16) catch 0;
                            }
                        }
                    }

                    if (elf_path) |path| {
                        showTrace(allocator, path, max_steps, input_bytes_owned) catch |err| {
                            std.debug.print("Failed to show trace: {s}\n", .{@errorName(err)});
                            std.process.exit(1);
                        };
                    } else {
                        std.debug.print("Error: trace command requires an ELF file path\n", .{});
                        std.debug.print("Usage: zolt trace [options] <elf_file>\n", .{});
                    }
                }
            } else {
                std.debug.print("Error: trace command requires an ELF file path\n", .{});
                std.debug.print("Usage: zolt trace [options] <elf_file>\n", .{});
            }
        },
        .prove => {
            if (args.next()) |arg| {
                if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
                    std.debug.print("Usage: zolt prove [options] <elf_file>\n\n", .{});
                    std.debug.print("Generate a ZK proof for a RISC-V ELF binary.\n", .{});
                    std.debug.print("This command runs the full proving pipeline:\n", .{});
                    std.debug.print("  1. Preprocess (generate SRS and keys)\n", .{});
                    std.debug.print("  2. Initialize prover\n", .{});
                    std.debug.print("  3. Generate proof using multi-stage sumcheck\n", .{});
                    std.debug.print("  4. Verify the proof\n\n", .{});
                    std.debug.print("Options:\n", .{});
                    std.debug.print("  --trace-length N         Set trace length for proof system (default: 1024)\n", .{});
                    std.debug.print("  -o, --output F           Save proof to file F\n", .{});
                    std.debug.print("  --json                   Output proof in JSON format (human readable)\n", .{});
                    std.debug.print("  --jolt-format            Output proof in Jolt-compatible format for cross-verification\n", .{});
                    std.debug.print("  --srs PATH               Use Dory SRS from PATH (exported by Jolt)\n", .{});
                    std.debug.print("  --export-preprocessing P Export Jolt-compatible preprocessing to file P\n", .{});
                    std.debug.print("  --input-hex HEX          Set input as hex bytes (e.g., 20 for input 32)\n", .{});
                } else {
                    // Parse options
                    var elf_path: ?[]const u8 = null;
                    var trace_length: ?u64 = null;
                    var output_path: ?[]const u8 = null;
                    var json_format = false;
                    var jolt_format = false;
                    var srs_path: ?[]const u8 = null;
                    var preprocessing_path: ?[]const u8 = null;
                    var input_hex: ?[]const u8 = null;

                    // First arg could be an option or the ELF path
                    if (std.mem.startsWith(u8, arg, "-")) {
                        if (std.mem.eql(u8, arg, "--trace-length")) {
                            if (args.next()) |len_str| {
                                trace_length = std.fmt.parseInt(u64, len_str, 10) catch null;
                            }
                        } else if (std.mem.eql(u8, arg, "-o") or std.mem.eql(u8, arg, "--output")) {
                            output_path = args.next();
                        } else if (std.mem.eql(u8, arg, "--json")) {
                            json_format = true;
                        } else if (std.mem.eql(u8, arg, "--jolt-format")) {
                            jolt_format = true;
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

                    // Parse remaining args
                    while (args.next()) |next_arg| {
                        if (std.mem.startsWith(u8, next_arg, "-")) {
                            if (std.mem.eql(u8, next_arg, "--trace-length")) {
                                if (args.next()) |len_str| {
                                    trace_length = std.fmt.parseInt(u64, len_str, 10) catch null;
                                }
                            } else if (std.mem.eql(u8, next_arg, "-o") or std.mem.eql(u8, next_arg, "--output")) {
                                output_path = args.next();
                            } else if (std.mem.eql(u8, next_arg, "--json")) {
                                json_format = true;
                            } else if (std.mem.eql(u8, next_arg, "--jolt-format")) {
                                jolt_format = true;
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

                    // Parse hex input if provided
                    var input_bytes_owned: ?[]u8 = null;
                    defer if (input_bytes_owned) |b| allocator.free(b);

                    if (input_hex) |hex| {
                        var clean_hex = hex;
                        if (std.mem.startsWith(u8, hex, "0x") or std.mem.startsWith(u8, hex, "0X")) {
                            clean_hex = hex[2..];
                        }
                        const buf_len = (clean_hex.len + 1) / 2;
                        input_bytes_owned = allocator.alloc(u8, buf_len) catch null;
                        if (input_bytes_owned) |buf| {
                            var i: usize = 0;
                            while (i < buf_len) : (i += 1) {
                                const start = i * 2;
                                const end = @min(start + 2, clean_hex.len);
                                buf[i] = std.fmt.parseInt(u8, clean_hex[start..end], 16) catch 0;
                            }
                        }
                    }

                    if (elf_path) |path| {
                        runProver(allocator, path, trace_length, output_path, json_format, jolt_format, srs_path, preprocessing_path, input_bytes_owned) catch |err| {
                            std.debug.print("Failed to generate proof: {s}\n", .{@errorName(err)});
                            std.process.exit(1);
                        };
                    } else {
                        std.debug.print("Error: prove command requires an ELF file path\n", .{});
                        std.debug.print("Usage: zolt prove [options] <elf_file>\n", .{});
                    }
                }
            } else {
                std.debug.print("Error: prove command requires an ELF file path\n", .{});
                std.debug.print("Usage: zolt prove [options] <elf_file>\n", .{});
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
        .stats => {
            if (args.next()) |arg| {
                if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
                    std.debug.print("Usage: zolt stats <proof_file>\n\n", .{});
                    std.debug.print("Show detailed statistics about a Zolt proof file.\n", .{});
                    std.debug.print("Includes information about commitments, sumcheck stages,\n", .{});
                    std.debug.print("proof size breakdown, and more.\n\n", .{});
                    std.debug.print("Example:\n", .{});
                    std.debug.print("  zolt stats proof.bin\n", .{});
                    std.debug.print("  zolt stats proof.json\n", .{});
                } else {
                    showProofStats(allocator, arg) catch |err| {
                        std.debug.print("Failed to read proof: {s}\n", .{@errorName(err)});
                        std.process.exit(1);
                    };
                }
            } else {
                std.debug.print("Error: stats command requires a proof file path\n", .{});
                std.debug.print("Usage: zolt stats <proof_file>\n", .{});
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
    try std.testing.expect(parseCommand("trace") == .trace);
    try std.testing.expect(parseCommand("prove") == .prove);
    try std.testing.expect(parseCommand("verify") == .verify);
    try std.testing.expect(parseCommand("stats") == .stats);
    try std.testing.expect(parseCommand("decode") == .decode);
    try std.testing.expect(parseCommand("bench") == .bench);
    try std.testing.expect(parseCommand("unknown_cmd") == .unknown);
}
