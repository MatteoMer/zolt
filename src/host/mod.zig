//! Host interface for Jolt zkVM
//!
//! This module provides the high-level interface for:
//! - Loading RISC-V ELF programs

const std = @import("std");
const Allocator = std.mem.Allocator;
const common = @import("../common/mod.zig");
pub const elf = @import("elf.zig");

/// ELF program loader
pub const ELFLoader = struct {
    allocator: Allocator,

    pub fn init(allocator: Allocator) ELFLoader {
        return .{
            .allocator = allocator,
        };
    }

    /// Load a RISC-V ELF binary from bytes
    pub fn load(self: *ELFLoader, data: []const u8) !Program {
        // Parse the ELF file
        var parsed = try elf.parse(self.allocator, data);
        defer parsed.deinit();

        // Validate it's a RISC-V binary
        if (!parsed.isRiscV()) {
            return error.NotRiscV;
        }

        // Calculate total bytecode size needed
        var max_addr: u64 = 0;
        var min_addr: u64 = std.math.maxInt(u64);
        for (parsed.segments) |segment| {
            if (segment.vaddr < min_addr) {
                min_addr = segment.vaddr;
            }
            const end_addr = segment.vaddr + segment.memsz;
            if (end_addr > max_addr) {
                max_addr = end_addr;
            }
        }

        // Use RAM_START_ADDRESS as base if no segments
        if (min_addr == std.math.maxInt(u64)) {
            min_addr = common.constants.RAM_START_ADDRESS;
            max_addr = min_addr;
        }

        // Allocate bytecode buffer (relative to base address)
        const base_addr = min_addr;
        const bytecode_size = max_addr - base_addr;
        const bytecode = try self.allocator.alloc(u8, bytecode_size);
        errdefer self.allocator.free(bytecode);

        // Initialize to zero (for .bss sections)
        @memset(bytecode, 0);

        // Copy segment data into bytecode buffer
        for (parsed.segments) |segment| {
            const offset = segment.vaddr - base_addr;
            const dest = bytecode[@as(usize, @intCast(offset))..];
            @memcpy(dest[0..segment.data.len], segment.data);
        }

        // Create memory config for layout
        var config = common.MemoryConfig{
            .program_size = bytecode_size,
        };

        // Find the size of the executable segment (.text section).
        // Instructions should only be decoded within this range; bytes beyond
        // this point may be .rodata (constants, lookup tables).
        // ELF segment flags: PF_X = 1 (executable), PF_W = 2 (writable), PF_R = 4 (readable)
        var text_size: usize = bytecode_size;
        for (parsed.segments) |segment| {
            if ((segment.flags & 1) != 0 and segment.data.len > 0) {
                // Executable segment — use its data length as text boundary
                text_size = segment.data.len;
                break;
            }
        }

        return Program{
            .bytecode = bytecode,
            .entry_point = parsed.header.entry,
            .base_address = base_addr,
            .text_size = text_size,
            .memory_layout = common.MemoryLayout.init(&config),
            .allocator = self.allocator,
        };
    }

    /// Load from file path
    pub fn loadFile(self: *ELFLoader, path: []const u8) !Program {
        const file = try std.fs.cwd().openFile(path, .{});
        defer file.close();

        const stat = try file.stat();
        const contents = try self.allocator.alloc(u8, stat.size);
        defer self.allocator.free(contents);

        _ = try file.readAll(contents);

        return self.load(contents);
    }
};

/// A loaded RISC-V program
pub const Program = struct {
    /// Program bytecode (all loadable segments concatenated)
    bytecode: []const u8,
    /// Entry point address
    entry_point: u64,
    /// Base address for the program in memory
    base_address: u64,
    /// Size of the executable code (.text) section in bytes.
    /// Only this portion should be decoded as instructions; bytes
    /// beyond this may be .rodata (constants) or other data.
    text_size: usize,
    /// Memory layout
    memory_layout: common.MemoryLayout,
    allocator: Allocator,

    pub fn deinit(self: *Program) void {
        self.allocator.free(self.bytecode);
    }

    /// Get the program size
    pub fn size(self: *const Program) usize {
        return self.bytecode.len;
    }

    /// Get byte at a given address
    pub fn getByte(self: *const Program, addr: u64) ?u8 {
        if (addr < self.base_address) return null;
        const offset = addr - self.base_address;
        if (offset >= self.bytecode.len) return null;
        return self.bytecode[@as(usize, @intCast(offset))];
    }

    /// Get a slice of bytes at a given address
    pub fn getBytes(self: *const Program, addr: u64, len: usize) ?[]const u8 {
        if (addr < self.base_address) return null;
        const offset = @as(usize, @intCast(addr - self.base_address));
        if (offset + len > self.bytecode.len) return null;
        return self.bytecode[offset .. offset + len];
    }
};
