//! Verify command: Verify a ZK proof using Jolt's RV64IMAC verifier via linked
//! Rust library. When built without `-Dverify=true`, this compiles to a stub
//! that prints a helpful message and exits — the `extern fn jolt_verify`
//! declaration is gated behind a comptime `if` so it only exists (and only
//! requires libjolt_verifier.a) when the flag is set.

const std = @import("std");
const build_options = @import("build_options");

const impl = if (build_options.enable_verifier) struct {
    extern fn jolt_verify(
        proof_ptr: [*]const u8,
        proof_len: usize,
        preprocessing_ptr: [*]const u8,
        preprocessing_len: usize,
        io_ptr: ?[*]const u8,
        io_len: usize,
    ) callconv(.c) i32;

    pub fn runVerifier(allocator: std.mem.Allocator, proof_path: []const u8, preprocessing_path: []const u8) !void {
        std.debug.print("Zolt zkVM Verifier\n", .{});
        std.debug.print("==================\n\n", .{});

        // Load preprocessing
        std.debug.print("Loading preprocessing: {s}\n", .{preprocessing_path});
        const preprocessing_bytes = blk: {
            const f = std.fs.cwd().openFile(preprocessing_path, .{}) catch |err| {
                std.debug.print("Error: failed to open preprocessing file: {s}\n", .{@errorName(err)});
                return err;
            };
            defer f.close();
            const stat = try f.stat();
            const buf = try allocator.alloc(u8, stat.size);
            const n = try f.readAll(buf);
            if (n != stat.size) {
                std.debug.print("Error: short read on preprocessing file\n", .{});
                allocator.free(buf);
                return error.ShortRead;
            }
            break :blk buf;
        };
        defer allocator.free(preprocessing_bytes);
        std.debug.print("  Preprocessing loaded: {} bytes\n", .{preprocessing_bytes.len});

        // Load proof
        std.debug.print("Loading proof: {s}\n", .{proof_path});
        const proof_bytes = blk: {
            const f = std.fs.cwd().openFile(proof_path, .{}) catch |err| {
                std.debug.print("Error: failed to open proof file: {s}\n", .{@errorName(err)});
                return err;
            };
            defer f.close();
            const stat = try f.stat();
            const buf = try allocator.alloc(u8, stat.size);
            const n = try f.readAll(buf);
            if (n != stat.size) {
                std.debug.print("Error: short read on proof file\n", .{});
                allocator.free(buf);
                return error.ShortRead;
            }
            break :blk buf;
        };
        defer allocator.free(proof_bytes);
        std.debug.print("  Proof loaded: {} bytes\n", .{proof_bytes.len});

        // Try to load program I/O sidecar (<proof_path>.io)
        const io_path = try std.fmt.allocPrint(allocator, "{s}.io", .{proof_path});
        defer allocator.free(io_path);

        var io_bytes: ?[]u8 = null;
        defer if (io_bytes) |b| allocator.free(b);

        if (std.fs.cwd().openFile(io_path, .{})) |f| {
            defer f.close();
            const stat = try f.stat();
            const buf = try allocator.alloc(u8, stat.size);
            const n = try f.readAll(buf);
            if (n == stat.size) {
                io_bytes = buf;
                std.debug.print("  IO sidecar loaded: {s} ({} bytes)\n", .{ io_path, buf.len });
            } else {
                allocator.free(buf);
            }
        } else |_| {
            std.debug.print("  No IO sidecar at {s}; assuming empty I/O\n", .{io_path});
        }

        // Call the Rust verifier
        std.debug.print("\nVerifying...\n", .{});
        var timer = try std.time.Timer.start();

        const io_ptr: ?[*]const u8 = if (io_bytes) |b| b.ptr else null;
        const io_len: usize = if (io_bytes) |b| b.len else 0;

        const result = jolt_verify(
            proof_bytes.ptr,
            proof_bytes.len,
            preprocessing_bytes.ptr,
            preprocessing_bytes.len,
            io_ptr,
            io_len,
        );

        const elapsed = timer.read();
        const elapsed_ms = @as(f64, @floatFromInt(elapsed)) / 1_000_000.0;

        switch (result) {
            0 => {
                std.debug.print("VERIFIED: proof is valid\n", .{});
                std.debug.print("Time: {d:.2} ms\n", .{elapsed_ms});
            },
            1 => {
                std.debug.print("FAILED: proof is invalid\n", .{});
                std.debug.print("Time: {d:.2} ms\n", .{elapsed_ms});
                std.process.exit(1);
            },
            else => {
                std.debug.print("ERROR: deserialization or internal error (code {})\n", .{result});
                std.process.exit(2);
            },
        }
    }
} else struct {
    pub fn runVerifier(_: std.mem.Allocator, _: []const u8, _: []const u8) !void {
        std.debug.print("Error: this build of zolt does not include the Jolt verifier.\n", .{});
        std.debug.print("Rebuild with `zig build -Dverify=true` to enable the verify command.\n", .{});
        return error.VerifierNotEnabled;
    }
};

pub const runVerifier = impl.runVerifier;

const zolt = @import("../root.zig");
const native_verifier = zolt.zkvm.verifier;

/// Run the native Zig verifier (no Rust dependency)
pub fn runNativeVerifier(allocator: std.mem.Allocator, proof_path: []const u8, preprocessing_path: []const u8) !void {
    std.debug.print("Zolt Native Verifier\n", .{});
    std.debug.print("====================\n\n", .{});

    // Load preprocessing
    std.debug.print("Loading preprocessing: {s}\n", .{preprocessing_path});
    const preprocessing_bytes = try readFile(allocator, preprocessing_path);
    defer allocator.free(preprocessing_bytes);
    std.debug.print("  Preprocessing loaded: {} bytes\n", .{preprocessing_bytes.len});

    // Load proof
    std.debug.print("Loading proof: {s}\n", .{proof_path});
    const proof_bytes = try readFile(allocator, proof_path);
    defer allocator.free(proof_bytes);
    std.debug.print("  Proof loaded: {} bytes\n", .{proof_bytes.len});

    // Try to load program I/O sidecar
    const io_path = try std.fmt.allocPrint(allocator, "{s}.io", .{proof_path});
    defer allocator.free(io_path);

    var io_bytes: ?[]u8 = null;
    defer if (io_bytes) |b| allocator.free(b);

    if (std.fs.cwd().openFile(io_path, .{})) |f| {
        defer f.close();
        const stat = try f.stat();
        const buf = try allocator.alloc(u8, stat.size);
        const n = try f.readAll(buf);
        if (n == stat.size) {
            io_bytes = buf;
            std.debug.print("  IO sidecar loaded: {s} ({} bytes)\n", .{ io_path, buf.len });
        } else {
            allocator.free(buf);
        }
    } else |_| {
        std.debug.print("  No IO sidecar at {s}; assuming empty I/O\n", .{io_path});
    }

    // Run native verifier
    std.debug.print("\nVerifying (native)...\n", .{});
    var timer = try std.time.Timer.start();

    const result = native_verifier.verify(
        allocator,
        proof_bytes,
        preprocessing_bytes,
        if (io_bytes) |b| b[0..] else null,
    );

    const elapsed = timer.read();
    const elapsed_ms = @as(f64, @floatFromInt(elapsed)) / 1_000_000.0;

    if (result.valid) {
        std.debug.print("VERIFIED: proof is valid (native)\n", .{});
        std.debug.print("Time: {d:.2} ms\n", .{elapsed_ms});
    } else {
        std.debug.print("FAILED: proof is invalid", .{});
        if (result.failed_stage) |stage| {
            std.debug.print(" (stage {})\n", .{stage});
        } else {
            std.debug.print("\n", .{});
        }
        std.debug.print("Time: {d:.2} ms\n", .{elapsed_ms});
        std.process.exit(1);
    }
}

fn readFile(allocator: std.mem.Allocator, path: []const u8) ![]u8 {
    const f = std.fs.cwd().openFile(path, .{}) catch |err| {
        std.debug.print("Error: failed to open file: {s} ({s})\n", .{ path, @errorName(err) });
        return err;
    };
    defer f.close();
    const stat = try f.stat();
    const buf = try allocator.alloc(u8, stat.size);
    const n = try f.readAll(buf);
    if (n != stat.size) {
        std.debug.print("Error: short read on file {s}\n", .{path});
        allocator.free(buf);
        return error.ShortRead;
    }
    return buf;
}
