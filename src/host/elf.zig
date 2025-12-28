//! ELF (Executable and Linkable Format) parser for RISC-V binaries
//!
//! This module parses ELF files to extract program code and metadata
//! needed for RISC-V zkVM execution.

const std = @import("std");
const Allocator = std.mem.Allocator;

/// ELF magic number: 0x7F 'E' 'L' 'F'
pub const ELF_MAGIC = [4]u8{ 0x7F, 'E', 'L', 'F' };

/// ELF class (32-bit or 64-bit)
pub const ElfClass = enum(u8) {
    None = 0,
    Elf32 = 1,
    Elf64 = 2,
    _,
};

/// ELF data encoding (endianness)
pub const ElfData = enum(u8) {
    None = 0,
    Lsb = 1, // Little endian
    Msb = 2, // Big endian
    _,
};

/// ELF machine type
pub const ElfMachine = enum(u16) {
    None = 0,
    RiscV = 0xF3,
    _,
};

/// ELF type
pub const ElfType = enum(u16) {
    None = 0,
    Rel = 1, // Relocatable
    Exec = 2, // Executable
    Dyn = 3, // Shared object
    Core = 4, // Core file
    _,
};

/// ELF section types
pub const SectionType = enum(u32) {
    Null = 0,
    ProgBits = 1, // Program data
    SymTab = 2, // Symbol table
    StrTab = 3, // String table
    Rela = 4, // Relocation with addend
    Hash = 5,
    Dynamic = 6,
    Note = 7,
    NoBits = 8, // .bss
    Rel = 9,
    ShLib = 10,
    DynSym = 11,
    InitArray = 14,
    FiniArray = 15,
    PreInitArray = 16,
    Group = 17,
    SymTabShndx = 18,
    _,
};

/// ELF program header types
pub const SegmentType = enum(u32) {
    Null = 0,
    Load = 1, // Loadable segment
    Dynamic = 2,
    Interp = 3,
    Note = 4,
    ShLib = 5,
    Phdr = 6,
    Tls = 7,
    _,
};

/// Segment flags
pub const SegmentFlags = packed struct {
    execute: bool,
    write: bool,
    read: bool,
    _reserved: u29 = 0,
};

/// ELF Header (common fields)
pub const ElfHeader = struct {
    /// ELF class (32-bit or 64-bit)
    class: ElfClass,
    /// Data encoding (endianness)
    data: ElfData,
    /// ELF version
    version: u8,
    /// OS/ABI identification
    osabi: u8,
    /// ABI version
    abi_version: u8,
    /// Object file type
    elf_type: ElfType,
    /// Machine architecture
    machine: ElfMachine,
    /// Entry point virtual address
    entry: u64,
    /// Program header table offset
    phoff: u64,
    /// Section header table offset
    shoff: u64,
    /// Processor-specific flags
    flags: u32,
    /// ELF header size
    ehsize: u16,
    /// Program header entry size
    phentsize: u16,
    /// Number of program headers
    phnum: u16,
    /// Section header entry size
    shentsize: u16,
    /// Number of section headers
    shnum: u16,
    /// Section name string table index
    shstrndx: u16,
};

/// Program header (segment)
pub const ProgramHeader = struct {
    /// Segment type
    segment_type: SegmentType,
    /// Segment flags
    flags: u32,
    /// Offset in file
    offset: u64,
    /// Virtual address
    vaddr: u64,
    /// Physical address
    paddr: u64,
    /// Size in file
    filesz: u64,
    /// Size in memory
    memsz: u64,
    /// Alignment
    alignment: u64,
};

/// Section header
pub const SectionHeader = struct {
    /// Section name (index into string table)
    name_idx: u32,
    /// Section type
    section_type: SectionType,
    /// Section flags
    flags: u64,
    /// Virtual address
    addr: u64,
    /// Offset in file
    offset: u64,
    /// Section size
    size: u64,
    /// Link to another section
    link: u32,
    /// Additional section info
    info: u32,
    /// Address alignment
    addralign: u64,
    /// Entry size (for sections with fixed-size entries)
    entsize: u64,
};

/// Loaded segment
pub const LoadedSegment = struct {
    /// Virtual address
    vaddr: u64,
    /// Data to load
    data: []const u8,
    /// Size in memory (may be larger than data for .bss)
    memsz: u64,
    /// Flags
    flags: u32,
};

/// Parsed ELF file
pub const ParsedElf = struct {
    /// ELF header
    header: ElfHeader,
    /// Program headers
    program_headers: []ProgramHeader,
    /// Section headers
    section_headers: []SectionHeader,
    /// Loadable segments (data slices into original buffer)
    segments: []LoadedSegment,
    /// Original ELF data
    data: []const u8,
    allocator: Allocator,

    pub fn deinit(self: *ParsedElf) void {
        self.allocator.free(self.program_headers);
        self.allocator.free(self.section_headers);
        self.allocator.free(self.segments);
    }

    /// Get the entry point address
    pub fn entryPoint(self: *const ParsedElf) u64 {
        return self.header.entry;
    }

    /// Get total memory size needed
    pub fn totalMemorySize(self: *const ParsedElf) u64 {
        var max_addr: u64 = 0;
        for (self.segments) |segment| {
            const end_addr = segment.vaddr + segment.memsz;
            if (end_addr > max_addr) {
                max_addr = end_addr;
            }
        }
        return max_addr;
    }

    /// Check if this is a RISC-V executable
    pub fn isRiscV(self: *const ParsedElf) bool {
        return self.header.machine == .RiscV;
    }

    /// Check if this is a 64-bit ELF
    pub fn is64Bit(self: *const ParsedElf) bool {
        return self.header.class == .Elf64;
    }
};

/// Parse an ELF file from bytes
pub fn parse(allocator: Allocator, data: []const u8) !ParsedElf {
    // Validate minimum size
    if (data.len < 16) {
        return error.InvalidElf;
    }

    // Check magic number
    if (!std.mem.eql(u8, data[0..4], &ELF_MAGIC)) {
        return error.InvalidMagic;
    }

    // Read ELF class
    const class: ElfClass = @enumFromInt(data[4]);
    if (class != .Elf32 and class != .Elf64) {
        return error.UnsupportedClass;
    }

    // Read endianness
    const data_encoding: ElfData = @enumFromInt(data[5]);
    if (data_encoding != .Lsb) {
        return error.UnsupportedEndianness; // We only support little endian (RISC-V)
    }

    // Parse header based on class
    const header = switch (class) {
        .Elf64 => try parseHeader64(data),
        .Elf32 => try parseHeader32(data),
        else => return error.UnsupportedClass,
    };

    // Validate machine type (we only support RISC-V)
    if (header.machine != .RiscV) {
        return error.UnsupportedMachine;
    }

    // Parse program headers
    const program_headers = try parseProgramHeaders(allocator, data, &header);
    errdefer allocator.free(program_headers);

    // Parse section headers
    const section_headers = try parseSectionHeaders(allocator, data, &header);
    errdefer allocator.free(section_headers);

    // Extract loadable segments
    const segments = try extractSegments(allocator, data, program_headers);
    errdefer allocator.free(segments);

    return ParsedElf{
        .header = header,
        .program_headers = program_headers,
        .section_headers = section_headers,
        .segments = segments,
        .data = data,
        .allocator = allocator,
    };
}

fn parseHeader64(data: []const u8) !ElfHeader {
    if (data.len < 64) {
        return error.InvalidElf;
    }

    return ElfHeader{
        .class = .Elf64,
        .data = @enumFromInt(data[5]),
        .version = data[6],
        .osabi = data[7],
        .abi_version = data[8],
        .elf_type = @enumFromInt(readU16(data[16..18])),
        .machine = @enumFromInt(readU16(data[18..20])),
        .entry = readU64(data[24..32]),
        .phoff = readU64(data[32..40]),
        .shoff = readU64(data[40..48]),
        .flags = readU32(data[48..52]),
        .ehsize = readU16(data[52..54]),
        .phentsize = readU16(data[54..56]),
        .phnum = readU16(data[56..58]),
        .shentsize = readU16(data[58..60]),
        .shnum = readU16(data[60..62]),
        .shstrndx = readU16(data[62..64]),
    };
}

fn parseHeader32(data: []const u8) !ElfHeader {
    if (data.len < 52) {
        return error.InvalidElf;
    }

    return ElfHeader{
        .class = .Elf32,
        .data = @enumFromInt(data[5]),
        .version = data[6],
        .osabi = data[7],
        .abi_version = data[8],
        .elf_type = @enumFromInt(readU16(data[16..18])),
        .machine = @enumFromInt(readU16(data[18..20])),
        .entry = readU32(data[24..28]),
        .phoff = readU32(data[28..32]),
        .shoff = readU32(data[32..36]),
        .flags = readU32(data[36..40]),
        .ehsize = readU16(data[40..42]),
        .phentsize = readU16(data[42..44]),
        .phnum = readU16(data[44..46]),
        .shentsize = readU16(data[46..48]),
        .shnum = readU16(data[48..50]),
        .shstrndx = readU16(data[50..52]),
    };
}

fn parseProgramHeaders(
    allocator: Allocator,
    data: []const u8,
    header: *const ElfHeader,
) ![]ProgramHeader {
    const count = header.phnum;
    if (count == 0) {
        return allocator.alloc(ProgramHeader, 0);
    }

    var headers = try allocator.alloc(ProgramHeader, count);
    errdefer allocator.free(headers);

    var offset = header.phoff;
    for (0..count) |i| {
        headers[i] = switch (header.class) {
            .Elf64 => try parseProgramHeader64(data, offset),
            .Elf32 => try parseProgramHeader32(data, offset),
            else => return error.UnsupportedClass,
        };
        offset += header.phentsize;
    }

    return headers;
}

fn parseProgramHeader64(data: []const u8, offset: u64) !ProgramHeader {
    const off = @as(usize, @intCast(offset));
    if (off + 56 > data.len) {
        return error.InvalidElf;
    }

    return ProgramHeader{
        .segment_type = @enumFromInt(readU32(data[off .. off + 4])),
        .flags = readU32(data[off + 4 .. off + 8]),
        .offset = readU64(data[off + 8 .. off + 16]),
        .vaddr = readU64(data[off + 16 .. off + 24]),
        .paddr = readU64(data[off + 24 .. off + 32]),
        .filesz = readU64(data[off + 32 .. off + 40]),
        .memsz = readU64(data[off + 40 .. off + 48]),
        .alignment = readU64(data[off + 48 .. off + 56]),
    };
}

fn parseProgramHeader32(data: []const u8, offset: u64) !ProgramHeader {
    const off = @as(usize, @intCast(offset));
    if (off + 32 > data.len) {
        return error.InvalidElf;
    }

    return ProgramHeader{
        .segment_type = @enumFromInt(readU32(data[off .. off + 4])),
        .offset = readU32(data[off + 4 .. off + 8]),
        .vaddr = readU32(data[off + 8 .. off + 12]),
        .paddr = readU32(data[off + 12 .. off + 16]),
        .filesz = readU32(data[off + 16 .. off + 20]),
        .memsz = readU32(data[off + 20 .. off + 24]),
        .flags = readU32(data[off + 24 .. off + 28]),
        .alignment = readU32(data[off + 28 .. off + 32]),
    };
}

fn parseSectionHeaders(
    allocator: Allocator,
    data: []const u8,
    header: *const ElfHeader,
) ![]SectionHeader {
    const count = header.shnum;
    if (count == 0) {
        return allocator.alloc(SectionHeader, 0);
    }

    var headers = try allocator.alloc(SectionHeader, count);
    errdefer allocator.free(headers);

    var offset = header.shoff;
    for (0..count) |i| {
        headers[i] = switch (header.class) {
            .Elf64 => try parseSectionHeader64(data, offset),
            .Elf32 => try parseSectionHeader32(data, offset),
            else => return error.UnsupportedClass,
        };
        offset += header.shentsize;
    }

    return headers;
}

fn parseSectionHeader64(data: []const u8, offset: u64) !SectionHeader {
    const off = @as(usize, @intCast(offset));
    if (off + 64 > data.len) {
        return error.InvalidElf;
    }

    return SectionHeader{
        .name_idx = readU32(data[off .. off + 4]),
        .section_type = @enumFromInt(readU32(data[off + 4 .. off + 8])),
        .flags = readU64(data[off + 8 .. off + 16]),
        .addr = readU64(data[off + 16 .. off + 24]),
        .offset = readU64(data[off + 24 .. off + 32]),
        .size = readU64(data[off + 32 .. off + 40]),
        .link = readU32(data[off + 40 .. off + 44]),
        .info = readU32(data[off + 44 .. off + 48]),
        .addralign = readU64(data[off + 48 .. off + 56]),
        .entsize = readU64(data[off + 56 .. off + 64]),
    };
}

fn parseSectionHeader32(data: []const u8, offset: u64) !SectionHeader {
    const off = @as(usize, @intCast(offset));
    if (off + 40 > data.len) {
        return error.InvalidElf;
    }

    return SectionHeader{
        .name_idx = readU32(data[off .. off + 4]),
        .section_type = @enumFromInt(readU32(data[off + 4 .. off + 8])),
        .flags = readU32(data[off + 8 .. off + 12]),
        .addr = readU32(data[off + 12 .. off + 16]),
        .offset = readU32(data[off + 16 .. off + 20]),
        .size = readU32(data[off + 20 .. off + 24]),
        .link = readU32(data[off + 24 .. off + 28]),
        .info = readU32(data[off + 28 .. off + 32]),
        .addralign = readU32(data[off + 32 .. off + 36]),
        .entsize = readU32(data[off + 36 .. off + 40]),
    };
}

fn extractSegments(
    allocator: Allocator,
    data: []const u8,
    program_headers: []const ProgramHeader,
) ![]LoadedSegment {
    // Count loadable segments
    var load_count: usize = 0;
    for (program_headers) |ph| {
        if (ph.segment_type == .Load) {
            load_count += 1;
        }
    }

    var segments = try allocator.alloc(LoadedSegment, load_count);
    errdefer allocator.free(segments);

    var idx: usize = 0;
    for (program_headers) |ph| {
        if (ph.segment_type == .Load) {
            const file_offset = @as(usize, @intCast(ph.offset));
            const file_size = @as(usize, @intCast(ph.filesz));

            if (file_offset + file_size > data.len) {
                return error.InvalidElf;
            }

            segments[idx] = LoadedSegment{
                .vaddr = ph.vaddr,
                .data = data[file_offset .. file_offset + file_size],
                .memsz = ph.memsz,
                .flags = ph.flags,
            };
            idx += 1;
        }
    }

    return segments;
}

// Little-endian read helpers
fn readU16(data: []const u8) u16 {
    return @as(u16, data[0]) | (@as(u16, data[1]) << 8);
}

fn readU32(data: []const u8) u32 {
    return @as(u32, data[0]) |
        (@as(u32, data[1]) << 8) |
        (@as(u32, data[2]) << 16) |
        (@as(u32, data[3]) << 24);
}

fn readU64(data: []const u8) u64 {
    return @as(u64, data[0]) |
        (@as(u64, data[1]) << 8) |
        (@as(u64, data[2]) << 16) |
        (@as(u64, data[3]) << 24) |
        (@as(u64, data[4]) << 32) |
        (@as(u64, data[5]) << 40) |
        (@as(u64, data[6]) << 48) |
        (@as(u64, data[7]) << 56);
}

// ============================================================================
// Tests
// ============================================================================

test "ELF magic validation" {
    const allocator = std.testing.allocator;

    // Invalid magic
    const invalid_data = [_]u8{ 0x00, 0x00, 0x00, 0x00 } ++ [_]u8{0} ** 60;
    const result = parse(allocator, &invalid_data);
    try std.testing.expectError(error.InvalidMagic, result);
}

test "ELF minimum size validation" {
    const allocator = std.testing.allocator;

    const small_data = [_]u8{ 0x7F, 'E', 'L', 'F' };
    const result = parse(allocator, &small_data);
    try std.testing.expectError(error.InvalidElf, result);
}

test "ELF header parsing helpers" {
    // Test readU16
    const data16 = [_]u8{ 0x34, 0x12 };
    try std.testing.expectEqual(@as(u16, 0x1234), readU16(&data16));

    // Test readU32
    const data32 = [_]u8{ 0x78, 0x56, 0x34, 0x12 };
    try std.testing.expectEqual(@as(u32, 0x12345678), readU32(&data32));

    // Test readU64
    const data64 = [_]u8{ 0xEF, 0xCD, 0xAB, 0x89, 0x67, 0x45, 0x23, 0x01 };
    try std.testing.expectEqual(@as(u64, 0x0123456789ABCDEF), readU64(&data64));
}

test "minimal 32-bit RISC-V ELF parsing" {
    const allocator = std.testing.allocator;

    // A minimal valid 32-bit RISC-V ELF header (52 bytes)
    // with one LOAD program header (32 bytes)
    var elf_data: [256]u8 = undefined;
    @memset(&elf_data, 0);

    // ELF header (52 bytes for 32-bit)
    // Magic: 0x7F "ELF"
    elf_data[0] = 0x7F;
    elf_data[1] = 'E';
    elf_data[2] = 'L';
    elf_data[3] = 'F';
    elf_data[4] = 1; // 32-bit
    elf_data[5] = 1; // Little endian
    elf_data[6] = 1; // Version
    elf_data[7] = 0; // OS/ABI

    // e_type: Executable (2)
    elf_data[16] = 2;
    elf_data[17] = 0;

    // e_machine: RISC-V (0xF3)
    elf_data[18] = 0xF3;
    elf_data[19] = 0;

    // e_version: 1
    elf_data[20] = 1;

    // e_entry: 0x80000000 (entry point)
    elf_data[24] = 0x00;
    elf_data[25] = 0x00;
    elf_data[26] = 0x00;
    elf_data[27] = 0x80;

    // e_phoff: 52 (program header offset)
    elf_data[28] = 52;

    // e_ehsize: 52
    elf_data[40] = 52;

    // e_phentsize: 32
    elf_data[42] = 32;

    // e_phnum: 1
    elf_data[44] = 1;

    // Program header at offset 52 (32 bytes for 32-bit)
    // p_type: PT_LOAD (1)
    elf_data[52] = 1;

    // p_offset: 0x60 (code starts after headers)
    elf_data[56] = 0x60;

    // p_vaddr: 0x80000000
    elf_data[60] = 0x00;
    elf_data[61] = 0x00;
    elf_data[62] = 0x00;
    elf_data[63] = 0x80;

    // p_paddr: 0x80000000
    elf_data[64] = 0x00;
    elf_data[65] = 0x00;
    elf_data[66] = 0x00;
    elf_data[67] = 0x80;

    // p_filesz: 8 (some bytes)
    elf_data[68] = 8;

    // p_memsz: 8
    elf_data[72] = 8;

    // p_flags: RX (5)
    elf_data[76] = 5;

    // p_align: 4
    elf_data[80] = 4;

    // Put some code at offset 0x60
    // addi x1, x0, 42 (0x02a00093)
    elf_data[0x60] = 0x93;
    elf_data[0x61] = 0x00;
    elf_data[0x62] = 0xa0;
    elf_data[0x63] = 0x02;
    // ecall (0x00000073)
    elf_data[0x64] = 0x73;
    elf_data[0x65] = 0x00;
    elf_data[0x66] = 0x00;
    elf_data[0x67] = 0x00;

    // Parse
    var parsed = try parse(allocator, &elf_data);
    defer parsed.deinit();

    // Verify header
    try std.testing.expect(parsed.isRiscV());
    try std.testing.expect(!parsed.is64Bit());
    try std.testing.expectEqual(@as(u64, 0x80000000), parsed.header.entry);

    // Verify segments
    try std.testing.expectEqual(@as(usize, 1), parsed.segments.len);
    try std.testing.expectEqual(@as(u64, 0x80000000), parsed.segments[0].vaddr);
    try std.testing.expectEqual(@as(u64, 8), parsed.segments[0].memsz);
}
