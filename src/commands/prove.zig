//! Prove command: Generate a ZK proof for a RISC-V ELF binary.

const std = @import("std");
const zolt = @import("../root.zig");
const BN254Scalar = zolt.field.BN254Scalar;

pub fn runProver(allocator: std.mem.Allocator, elf_path: []const u8, output_path: []const u8, srs_path: ?[]const u8, preprocessing_path: ?[]const u8, input_bytes: ?[]const u8) !void {
    std.debug.print("Zolt zkVM Prover\n", .{});
    std.debug.print("================\n\n", .{});

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
    if (input_bytes) |inputs| {
        std.debug.print("  Input bytes: {} bytes\n", .{inputs.len});
    }

    var timer = std.time.Timer.start() catch return;

    // Initialize prover
    std.debug.print("\n[1/2] Initializing prover...\n", .{});

    const thread_pool = try zolt.utils.ThreadPool.init(allocator);
    defer thread_pool.deinit();

    var prover_inst = zolt.zkvm.JoltProver(BN254Scalar).initWithThreadPool(allocator, thread_pool);

    // Generate Jolt-compatible proof with Dory commitments
    std.debug.print("\n[2/2] Generating proof...\n", .{});
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
        program.text_size,
    ) catch |err| {
        std.debug.print("  Error generating Jolt-compatible proof: {s}\n", .{@errorName(err)});
        return err;
    };
    defer jolt_bundle.deinit();

    const prove_time = timer.read();
    const prove_time_ms = @as(f64, @floatFromInt(prove_time)) / 1_000_000.0;
    std.debug.print("  Proof generated successfully!\n", .{});
    std.debug.print("  Time: {d:.2} ms\n", .{prove_time_ms});
    if (std.posix.getenv("ZOLT_BENCH") != null) {
        std.debug.print("[BENCH] Total time: {d:.1}\n", .{prove_time_ms});
    }

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
        var bytecode_prep = preprocessing.BytecodePreprocessing.preprocessWithTextSize(allocator, program.bytecode, program.entry_point, device.memory_layout.termination, program.text_size) catch |err| {
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
            .max_padded_trace_length = jolt_bundle.proof.trace_length,
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

        var verifier_setup = preprocessing.DoryVerifierSetup.fromSRS(allocator, &srs, thread_pool) catch |err| {
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

        // blindfold_setup: Option<BlindfoldSetup<C>> = None (arkworks serializes as 0u8)
        buffer.writer(allocator).writeByte(0) catch |err| {
            std.debug.print("  Error serializing blindfold_setup: {s}\n", .{@errorName(err)});
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
