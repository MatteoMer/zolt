//! BytecodePreprocessing - Jolt-compatible bytecode preprocessing
//!
//! Preprocesses RISC-V program bytecode into Jolt's instruction format,
//! handling virtual instruction expansion, W-extension decomposition,
//! and power-of-2 padding.

const std = @import("std");

const zkvm_debug = @import("debug.zig");
const dbg = zkvm_debug.dbg;

const Allocator = std.mem.Allocator;
const jolt_device = @import("jolt_device.zig");
const MemoryLayout = jolt_device.MemoryLayout;
const common = @import("../common/mod.zig");

const sha256_inline = @import("../tracer/sha256_inline.zig");

pub const jolt_instruction = @import("jolt_instruction.zig");
pub const JoltInstruction = jolt_instruction.JoltInstruction;

pub const instruction_decoder = @import("instruction_decoder.zig");
pub const decodeToJoltInstruction = instruction_decoder.decodeToJoltInstruction;

pub const bytecode_pc_mapper = @import("bytecode_pc_mapper.zig");
pub const BytecodePCMapper = bytecode_pc_mapper.BytecodePCMapper;

/// Map CSR address to Jolt virtual register index.
/// mstatus (0x300) → VR 39
/// mtvec   (0x305) → VR 34
/// mscratch(0x340) → VR 35
/// mepc    (0x341) → VR 36
/// mcause  (0x342) → VR 37
/// mtval   (0x343) → VR 38
pub fn csrToVirtualReg(csr_addr: u12) u8 {
    return switch (csr_addr) {
        0x300 => 39, // mstatus
        0x305 => 34, // mtvec
        0x340 => 35, // mscratch
        0x341 => 36, // mepc
        0x342 => 37, // mcause
        0x343 => 38, // mtval
        else => 39, // Unknown CSR → map to mstatus as fallback
    };
}

/// Compute the number of bytecode entries for a CSRRW instruction.
pub fn csrrwEntryCount(rd: u8, rs1: u8) usize {
    if (rd == 0) return 1; // csrw pseudo
    if (rd == rs1) return 3; // need temp
    return 2; // read old, write new
}

/// Compute the number of bytecode entries for a CSRRS instruction.
pub fn csrrsEntryCount(rd: u8, rs1: u8) usize {
    if (rs1 == 0) return 1; // csrr pseudo (read-only)
    if (rd == 0) return 1; // csrs pseudo (set-only)
    if (rd == rs1) return 3; // need temp
    return 2; // read + set
}

/// BytecodePreprocessing - matches Jolt's BytecodePreprocessing
pub const BytecodePreprocessing = struct {
    /// Power-of-2 padded code size
    code_size: usize,
    /// Vector of instructions (serialized as JSON)
    bytecode: std.ArrayListUnmanaged(JoltInstruction),
    /// PC mapper
    pc_map: BytecodePCMapper,
    /// Raw 32-bit instruction words (one per bytecode entry, including NoOp=0 at index 0
    /// and virtual instruction words). Used by Jolt verifier for Zolt-compatible flag computation.
    raw_words: std.ArrayListUnmanaged(u32),
    /// ELF entry point address (e_entry). Serialized after pc_map to match upstream.
    entry_address: u64,

    allocator: Allocator,

    pub fn init(allocator: Allocator) BytecodePreprocessing {
        return .{
            .code_size = 0,
            .bytecode = .{},
            .pc_map = BytecodePCMapper.init(allocator),
            .raw_words = .{},
            .entry_address = 0,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *BytecodePreprocessing) void {
        self.bytecode.deinit(self.allocator);
        self.raw_words.deinit(self.allocator);
        self.pc_map.deinit();
    }

    /// Preprocess bytecode from raw bytes.
    /// `termination_address_opt` is the memory-mapped I/O address where termination is signaled
    /// (from MemoryLayout.termination). This is used to generate the synthetic LUI+ADDI+SD
    /// termination sequence that matches the prover's bytecode table.
    /// If null, uses the default MemoryLayout termination address (0x7FFFC008).
    pub fn preprocess(allocator: Allocator, code_bytes: []const u8, base_address: u64, termination_address_opt: ?u64) !BytecodePreprocessing {
        return preprocessWithTextSize(allocator, code_bytes, base_address, termination_address_opt, code_bytes.len);
    }

    /// Preprocess bytecode, only decoding instructions within the first `text_size` bytes.
    /// Bytes beyond text_size are treated as data (.rodata) and skipped.
    pub fn preprocessWithTextSize(allocator: Allocator, code_bytes: []const u8, base_address: u64, termination_address_opt: ?u64, text_size: usize) !BytecodePreprocessing {
        const termination_address = termination_address_opt orelse 0x7FFFC008; // Default from MemoryLayout with standard 4KB sizes
        var self = BytecodePreprocessing.init(allocator);
        self.entry_address = base_address;
        errdefer self.deinit();

        // Prepend a single NoOp instruction (as Jolt does)
        try self.bytecode.append(allocator, .{
            .variant = .NoOp,
            .address = 0,
            .operands = .{ .None = {} },
            .virtual_sequence_remaining = null,
            .is_first_in_sequence = false,
            .is_compressed = false,
        });
        try self.raw_words.append(allocator, 0); // NoOp = raw word 0

        // Decode instructions within the .text section only
        const decode_limit = @min(text_size, code_bytes.len);
        var offset: usize = 0;
        while (offset < decode_limit) {
            const addr = base_address + offset;

            // Check if compressed (RVC)
            const first_halfword: u16 = std.mem.readInt(u16, code_bytes[offset..][0..2], .little);
            const is_compressed = (first_halfword & 0x3) != 0x3;

            var instruction: u32 = undefined;
            var instr_size: usize = undefined;

            if (first_halfword == 0) {
                // Zero halfword in a code gap — skip 2 bytes, leave as NoOp padding
                offset += 2;
                continue;
            } else if (is_compressed) {
                // 16-bit compressed instruction - expand it
                const zkvm_instruction = @import("instruction/mod.zig");
                instruction = zkvm_instruction.uncompressInstruction(first_halfword, .Bit64);
                instr_size = 2;
            } else {
                // 32-bit instruction
                if (offset + 4 > code_bytes.len) break;
                instruction = std.mem.readInt(u32, code_bytes[offset..][0..4], .little);
                instr_size = 4;
            }

            // Decode and decompose W-extension instructions into virtual sequences
            const jolt_instr = try instruction_decoder.decodeToJoltInstruction(instruction, addr, is_compressed);
            const bytecode_len_before = self.bytecode.items.len;

            // Check if this is a W-extension instruction that needs decomposition
            switch (jolt_instr.variant) {
                .ADDIW => {
                    // ADDIW → ADDI + VirtualSignExtendWord (2-instruction sequence)
                    // Step 1: ADDI(rd, rs1, imm) with virtual_sequence_remaining=1, is_first_in_sequence=true
                    try self.bytecode.append(allocator, .{
                        .variant = .ADDI,
                        .address = addr,
                        .operands = jolt_instr.operands, // Same FormatI operands
                        .virtual_sequence_remaining = 1,
                        .is_first_in_sequence = true,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 2: VirtualSignExtendWord(rd, rd, 0) with virtual_sequence_remaining=0
                    const rd = switch (jolt_instr.operands) {
                        .FormatI => |i| i.rd,
                        else => 0,
                    };
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualSignExtendWord,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = rd, .rs1 = rd, .imm = 0 } },
                        .virtual_sequence_remaining = 0,
                        .is_first_in_sequence = false,
                        .is_compressed = is_compressed,
                    });
                },
                .ADDW => {
                    // ADDW → ADD + VirtualSignExtendWord (2-instruction sequence)
                    // Step 1: ADD(rd, rs1, rs2)
                    try self.bytecode.append(allocator, .{
                        .variant = .ADD,
                        .address = addr,
                        .operands = jolt_instr.operands, // Same FormatR operands
                        .virtual_sequence_remaining = 1,
                        .is_first_in_sequence = true,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 2: VirtualSignExtendWord(rd, rd, 0)
                    const rd = switch (jolt_instr.operands) {
                        .FormatR => |r| r.rd,
                        else => 0,
                    };
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualSignExtendWord,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = rd, .rs1 = rd, .imm = 0 } },
                        .virtual_sequence_remaining = 0,
                        .is_first_in_sequence = false,
                        .is_compressed = is_compressed,
                    });
                },
                .SUBW => {
                    // SUBW → SUB + VirtualSignExtendWord (2-instruction sequence)
                    try self.bytecode.append(allocator, .{
                        .variant = .SUB,
                        .address = addr,
                        .operands = jolt_instr.operands,
                        .virtual_sequence_remaining = 1,
                        .is_first_in_sequence = true,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    const rd = switch (jolt_instr.operands) {
                        .FormatR => |r| r.rd,
                        else => 0,
                    };
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualSignExtendWord,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = rd, .rs1 = rd, .imm = 0 } },
                        .virtual_sequence_remaining = 0,
                        .is_first_in_sequence = false,
                        .is_compressed = is_compressed,
                    });
                },
                .MULW => {
                    // MULW → MUL + VirtualSignExtendWord (2-instruction sequence)
                    try self.bytecode.append(allocator, .{
                        .variant = .MUL,
                        .address = addr,
                        .operands = jolt_instr.operands,
                        .virtual_sequence_remaining = 1,
                        .is_first_in_sequence = true,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    const rd = switch (jolt_instr.operands) {
                        .FormatR => |r| r.rd,
                        else => 0,
                    };
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualSignExtendWord,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = rd, .rs1 = rd, .imm = 0 } },
                        .virtual_sequence_remaining = 0,
                        .is_first_in_sequence = false,
                        .is_compressed = is_compressed,
                    });
                },
                .SLLI => {
                    // SLLI rd, rs1, imm → VirtualMULI rd, rs1, (1 << imm)
                    // Single virtual instruction - standalone 1-entry virtual sequence.
                    // Jolt's finalize() sets vsr=Some(0) and is_first_in_sequence=true.
                    const shift_amount = switch (jolt_instr.operands) {
                        .FormatI => |i| i.imm,
                        else => 0,
                    };
                    const rd = switch (jolt_instr.operands) {
                        .FormatI => |i| i.rd,
                        else => 0,
                    };
                    const rs1 = switch (jolt_instr.operands) {
                        .FormatI => |i| i.rs1,
                        else => 0,
                    };
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualMULI,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = rd, .rs1 = rs1, .imm = @as(u64, 1) << @intCast(shift_amount) } },
                        .virtual_sequence_remaining = 0,
                        .is_first_in_sequence = true,
                        .is_compressed = is_compressed,
                    });
                },
                .SLLIW => {
                    // SLLIW rd, rs1, imm → VirtualMULI rd, rs1, (1 << imm) + VirtualSignExtendWord(rd, rd, 0)
                    const shift_amount = switch (jolt_instr.operands) {
                        .FormatI => |i| i.imm,
                        else => 0,
                    };
                    const rd = switch (jolt_instr.operands) {
                        .FormatI => |i| i.rd,
                        else => 0,
                    };
                    const rs1 = switch (jolt_instr.operands) {
                        .FormatI => |i| i.rs1,
                        else => 0,
                    };
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualMULI,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = rd, .rs1 = rs1, .imm = @as(u64, 1) << @intCast(shift_amount) } },
                        .virtual_sequence_remaining = 1,
                        .is_first_in_sequence = true,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualSignExtendWord,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = rd, .rs1 = rd, .imm = 0 } },
                        .virtual_sequence_remaining = 0,
                        .is_first_in_sequence = false,
                        .is_compressed = is_compressed,
                    });
                },
                .SRLI => {
                    // SRLI rd, rs1, shamt → VirtualSRLI(rd, rs1, bitmask)
                    // Single virtual instruction - standalone 1-entry virtual sequence.
                    // Jolt's finalize() sets vsr=Some(0) and is_first_in_sequence=true.
                    const raw_imm = switch (jolt_instr.operands) {
                        .FormatI => |i| i.imm,
                        else => 0,
                    };
                    const rd = switch (jolt_instr.operands) {
                        .FormatI => |i| i.rd,
                        else => 0,
                    };
                    const rs1_val = switch (jolt_instr.operands) {
                        .FormatI => |i| i.rs1,
                        else => 0,
                    };
                    const shift: u7 = @intCast(raw_imm & 0x3f);
                    const ones: u128 = (@as(u128, 1) << @intCast(64 - @as(u8, shift))) - 1;
                    const bitmask: u64 = @truncate(ones << shift);
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualSRLI,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = rd, .rs1 = rs1_val, .imm = bitmask } },
                        .virtual_sequence_remaining = 0,
                        .is_first_in_sequence = true,
                        .is_compressed = is_compressed,
                    });
                },
                .SRLIW => {
                    // SRLIW rd, rs1, shamt → 3-step sequence:
                    //   Step 1: SLLI(v_rs1, rs1, 32) → VirtualMULI(v_rs1, rs1, 2^32)
                    //   Step 2: VirtualSRLI(rd, v_rs1, bitmask) where shift = shamt + 32
                    //   Step 3: VirtualSignExtendWord(rd, rd, 0)
                    const raw_imm = switch (jolt_instr.operands) {
                        .FormatI => |i| i.imm,
                        else => 0,
                    };
                    const rd = switch (jolt_instr.operands) {
                        .FormatI => |i| i.rd,
                        else => 0,
                    };
                    const rs1_val = switch (jolt_instr.operands) {
                        .FormatI => |i| i.rs1,
                        else => 0,
                    };
                    // Virtual register for intermediate result (register 32 = first virtual register)
                    const v_rs1: u8 = 32;
                    // Compute bitmask: shift = (shamt & 0x1f) + 32
                    const shamt: u7 = @intCast(raw_imm & 0x1f);
                    const total_shift: u7 = shamt + 32;
                    const ones: u128 = (@as(u128, 1) << @intCast(64 - @as(u8, total_shift))) - 1;
                    const bitmask: u64 = @truncate(ones << total_shift);
                    // Step 1: VirtualMULI(v_rs1, rs1, 2^32) - shift left by 32
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualMULI,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = v_rs1, .rs1 = rs1_val, .imm = @as(u64, 1) << 32 } },
                        .virtual_sequence_remaining = 2,
                        .is_first_in_sequence = true,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 2: VirtualSRLI(rd, v_rs1, bitmask)
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualSRLI,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = rd, .rs1 = v_rs1, .imm = bitmask } },
                        .virtual_sequence_remaining = 1,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 3: VirtualSignExtendWord(rd, rd, 0)
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualSignExtendWord,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = rd, .rs1 = rd, .imm = 0 } },
                        .virtual_sequence_remaining = 0,
                        .is_first_in_sequence = false,
                        .is_compressed = is_compressed,
                    });
                },
                .REMUW => {
                    // REMUW → 12-instruction inline sequence (matching Jolt's decomposition)
                    // Virtual registers: a2=32, a3=33, t0=34, t1=35, t2=36, t3=37, t4=38
                    const rs1 = switch (jolt_instr.operands) {
                        .FormatR => |r| r.rs1,
                        else => 0,
                    };
                    const rs2 = switch (jolt_instr.operands) {
                        .FormatR => |r| r.rs2,
                        else => 0,
                    };
                    const rd = switch (jolt_instr.operands) {
                        .FormatR => |r| r.rd,
                        else => 0,
                    };
                    const a2: u8 = 32;
                    const a3: u8 = 33;
                    const t0: u8 = 34;
                    const t1: u8 = 35;
                    const t2: u8 = 36;
                    const t3: u8 = 37;
                    const t4: u8 = 38;

                    // Step 1: VirtualAdvice(a2) → quotient (vsr=11, first)
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualAdvice,
                        .address = addr,
                        .operands = .{ .FormatJ = .{ .rd = a2, .imm = 0 } },
                        .virtual_sequence_remaining = 11,
                        .is_first_in_sequence = true,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 2: VirtualAdvice(a3) → remainder (vsr=10)
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualAdvice,
                        .address = addr,
                        .operands = .{ .FormatJ = .{ .rd = a3, .imm = 0 } },
                        .virtual_sequence_remaining = 10,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 3: VirtualZeroExtendWord(t3, a2) → zero-extend quotient (vsr=9)
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualZeroExtendWord,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = t3, .rs1 = a2, .imm = 0 } },
                        .virtual_sequence_remaining = 9,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 4: VirtualZeroExtendWord(t1, rs1) → zero-extend dividend (vsr=8)
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualZeroExtendWord,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = t1, .rs1 = rs1, .imm = 0 } },
                        .virtual_sequence_remaining = 8,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 5: VirtualZeroExtendWord(t2, rs2) → zero-extend divisor (vsr=7)
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualZeroExtendWord,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = t2, .rs1 = rs2, .imm = 0 } },
                        .virtual_sequence_remaining = 7,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 6: MUL(t0, t3, t2) → quotient * divisor (vsr=6)
                    try self.bytecode.append(allocator, .{
                        .variant = .MUL,
                        .address = addr,
                        .operands = .{ .FormatR = .{ .rd = t0, .rs1 = t3, .rs2 = t2 } },
                        .virtual_sequence_remaining = 6,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 7: VirtualZeroExtendWord(t4, t0) → mask to 32 bits (vsr=5)
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualZeroExtendWord,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = t4, .rs1 = t0, .imm = 0 } },
                        .virtual_sequence_remaining = 5,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 8: VirtualAssertEQ(t4, t0) → assert no overflow (vsr=4)
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualAssertEQ,
                        .address = addr,
                        .operands = .{ .FormatB = .{ .rs1 = t4, .rs2 = t0, .imm = 0 } },
                        .virtual_sequence_remaining = 4,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 9: ADD(t0, t0, a3) → add remainder (vsr=3)
                    try self.bytecode.append(allocator, .{
                        .variant = .ADD,
                        .address = addr,
                        .operands = .{ .FormatR = .{ .rd = t0, .rs1 = t0, .rs2 = a3 } },
                        .virtual_sequence_remaining = 3,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 10: VirtualAssertEQ(t0, t1) → assert dividend = q*d + r (vsr=2)
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualAssertEQ,
                        .address = addr,
                        .operands = .{ .FormatB = .{ .rs1 = t0, .rs2 = t1, .imm = 0 } },
                        .virtual_sequence_remaining = 2,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 11: VirtualAssertValidUnsignedRemainder(a3, t2) → r < d (vsr=1)
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualAssertValidUnsignedRemainder,
                        .address = addr,
                        .operands = .{ .FormatB = .{ .rs1 = a3, .rs2 = t2, .imm = 0 } },
                        .virtual_sequence_remaining = 1,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 12: VirtualSignExtendWord(rd, a3) → sign-extend result (vsr=0, last)
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualSignExtendWord,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = rd, .rs1 = a3, .imm = 0 } },
                        .virtual_sequence_remaining = 0,
                        .is_first_in_sequence = false,
                        .is_compressed = is_compressed,
                    });
                },
                .REMW, .DIVW => {
                    // REMW/DIVW → 21-instruction inline sequence (matching Jolt's decomposition)
                    // Signed division/remainder verification with overflow handling
                    // Virtual registers: a2=32, a3=33, t0=34, t1=35, t2=36, t3=37, t4=38
                    const rs1 = switch (jolt_instr.operands) {
                        .FormatR => |r| r.rs1,
                        else => 0,
                    };
                    const rs2 = switch (jolt_instr.operands) {
                        .FormatR => |r| r.rs2,
                        else => 0,
                    };
                    const rd = switch (jolt_instr.operands) {
                        .FormatR => |r| r.rd,
                        else => 0,
                    };
                    const a2: u8 = 32; // quotient
                    const a3: u8 = 33; // |remainder|
                    const t0: u8 = 34; // adjusted divisor
                    const t1: u8 = 35; // temporary
                    const t2: u8 = 36; // temporary
                    const t3: u8 = 37; // signed remainder
                    const t4: u8 = 38; // sign-extended dividend

                    // Step 1: VirtualAdvice(a2) → quotient (vsr=20, first)
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualAdvice,
                        .address = addr,
                        .operands = .{ .FormatJ = .{ .rd = a2, .imm = 0 } },
                        .virtual_sequence_remaining = 20,
                        .is_first_in_sequence = true,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 2: VirtualAdvice(a3) → |remainder| (vsr=19)
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualAdvice,
                        .address = addr,
                        .operands = .{ .FormatJ = .{ .rd = a3, .imm = 0 } },
                        .virtual_sequence_remaining = 19,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 3: VirtualSignExtendWord(t4, rs1) → sign-extend dividend (vsr=18)
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualSignExtendWord,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = t4, .rs1 = rs1, .imm = 0 } },
                        .virtual_sequence_remaining = 18,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 4: VirtualSignExtendWord(t3, rs2) → sign-extend divisor (vsr=17)
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualSignExtendWord,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = t3, .rs1 = rs2, .imm = 0 } },
                        .virtual_sequence_remaining = 17,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 5: VirtualAssertValidDiv0(t3, a2) → handle div-by-zero (vsr=16)
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualAssertValidDiv0,
                        .address = addr,
                        .operands = .{ .FormatB = .{ .rs1 = t3, .rs2 = a2, .imm = 0 } },
                        .virtual_sequence_remaining = 16,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 6: VirtualChangeDivisorW(t0, t4, t3) → handle overflow (vsr=15)
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualChangeDivisorW,
                        .address = addr,
                        .operands = .{ .FormatR = .{ .rd = t0, .rs1 = t4, .rs2 = t3 } },
                        .virtual_sequence_remaining = 15,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 7: VirtualSignExtendWord(t1, a2) → sign-extend quotient (vsr=14)
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualSignExtendWord,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = t1, .rs1 = a2, .imm = 0 } },
                        .virtual_sequence_remaining = 14,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 8: VirtualAssertEQ(t1, a2) → assert quotient fits 32 bits (vsr=13)
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualAssertEQ,
                        .address = addr,
                        .operands = .{ .FormatB = .{ .rs1 = t1, .rs2 = a2, .imm = 0 } },
                        .virtual_sequence_remaining = 13,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 9: VirtualSRAI(t2, a3, bitmask_31) → sign bit of |remainder| (vsr=12)
                    // SRAI is expanded to VirtualSRAI with bitmask: shift=31, bitmask = ((1<<33)-1) << 31
                    const srai_bitmask: u64 = blk: {
                        const shift_amt: u7 = 31;
                        const ones: u128 = (@as(u128, 1) << @intCast(64 - @as(u8, shift_amt))) - 1;
                        break :blk @truncate(ones << shift_amt);
                    };
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualSRAI,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = t2, .rs1 = a3, .imm = srai_bitmask } },
                        .virtual_sequence_remaining = 12,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 10: VirtualAssertEQ(t2, 0) → assert |remainder| is non-negative (vsr=11)
                    // Note: rs2=0 means comparing against register x0 (always 0)
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualAssertEQ,
                        .address = addr,
                        .operands = .{ .FormatB = .{ .rs1 = t2, .rs2 = 0, .imm = 0 } },
                        .virtual_sequence_remaining = 11,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 11: VirtualSRAI(t2, t4, bitmask_31) → sign bit of dividend (vsr=10)
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualSRAI,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = t2, .rs1 = t4, .imm = srai_bitmask } },
                        .virtual_sequence_remaining = 10,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 12: XOR(t3, a3, t2) → XOR |remainder| with sign mask (vsr=9)
                    try self.bytecode.append(allocator, .{
                        .variant = .XOR,
                        .address = addr,
                        .operands = .{ .FormatR = .{ .rd = t3, .rs1 = a3, .rs2 = t2 } },
                        .virtual_sequence_remaining = 9,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 13: SUB(t3, t3, t2) → t3 = sign-corrected remainder (vsr=8)
                    try self.bytecode.append(allocator, .{
                        .variant = .SUB,
                        .address = addr,
                        .operands = .{ .FormatR = .{ .rd = t3, .rs1 = t3, .rs2 = t2 } },
                        .virtual_sequence_remaining = 8,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 14: MUL(t1, a2, t0) → quotient × adjusted_divisor (vsr=7)
                    try self.bytecode.append(allocator, .{
                        .variant = .MUL,
                        .address = addr,
                        .operands = .{ .FormatR = .{ .rd = t1, .rs1 = a2, .rs2 = t0 } },
                        .virtual_sequence_remaining = 7,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 15: ADD(t1, t1, t3) → + remainder (vsr=6)
                    try self.bytecode.append(allocator, .{
                        .variant = .ADD,
                        .address = addr,
                        .operands = .{ .FormatR = .{ .rd = t1, .rs1 = t1, .rs2 = t3 } },
                        .virtual_sequence_remaining = 6,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 16: VirtualAssertEQ(t1, t4) → assert dividend = q*d + r (vsr=5)
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualAssertEQ,
                        .address = addr,
                        .operands = .{ .FormatB = .{ .rs1 = t1, .rs2 = t4, .imm = 0 } },
                        .virtual_sequence_remaining = 5,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 17: VirtualSRAI(t2, t0, bitmask_31) → sign bit of adjusted divisor (vsr=4)
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualSRAI,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = t2, .rs1 = t0, .imm = srai_bitmask } },
                        .virtual_sequence_remaining = 4,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 18: XOR(t1, t0, t2) → (vsr=3)
                    try self.bytecode.append(allocator, .{
                        .variant = .XOR,
                        .address = addr,
                        .operands = .{ .FormatR = .{ .rd = t1, .rs1 = t0, .rs2 = t2 } },
                        .virtual_sequence_remaining = 3,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 19: SUB(t1, t1, t2) → t1 = abs(divisor) (vsr=2)
                    try self.bytecode.append(allocator, .{
                        .variant = .SUB,
                        .address = addr,
                        .operands = .{ .FormatR = .{ .rd = t1, .rs1 = t1, .rs2 = t2 } },
                        .virtual_sequence_remaining = 2,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 20: VirtualAssertValidUnsignedRemainder(a3, t1) → |r| < |d| (vsr=1)
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualAssertValidUnsignedRemainder,
                        .address = addr,
                        .operands = .{ .FormatB = .{ .rs1 = a3, .rs2 = t1, .imm = 0 } },
                        .virtual_sequence_remaining = 1,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 21: VirtualSignExtendWord(rd, output) → sign-extend result (vsr=0, last)
                    // REMW: output = t3 (signed remainder), DIVW: output = a2 (quotient)
                    const output_reg = if (jolt_instr.variant == .REMW) t3 else a2;
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualSignExtendWord,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = rd, .rs1 = output_reg, .imm = 0 } },
                        .virtual_sequence_remaining = 0,
                        .is_first_in_sequence = false,
                        .is_compressed = is_compressed,
                    });
                },
                .SLL => {
                    // SLL rd, rs1, rs2 → 2-step: VirtualPow2(v0, rs2, 0) + MUL(rd, rs1, v0)
                    const rd = switch (jolt_instr.operands) {
                        .FormatR => |r| r.rd,
                        else => 0,
                    };
                    const rs1 = switch (jolt_instr.operands) {
                        .FormatR => |r| r.rs1,
                        else => 0,
                    };
                    const rs2 = switch (jolt_instr.operands) {
                        .FormatR => |r| r.rs2,
                        else => 0,
                    };
                    const v0: u8 = 40; // first virtual alloc register
                    // Step 1: VirtualPow2(v0, rs2, 0) — compute 2^(rs2 % 64)
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualPow2,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = v0, .rs1 = rs2, .imm = 0 } },
                        .virtual_sequence_remaining = 1,
                        .is_first_in_sequence = true,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 2: MUL(rd, rs1, v0)
                    try self.bytecode.append(allocator, .{
                        .variant = .MUL,
                        .address = addr,
                        .operands = .{ .FormatR = .{ .rd = rd, .rs1 = rs1, .rs2 = v0 } },
                        .virtual_sequence_remaining = 0,
                        .is_first_in_sequence = false,
                        .is_compressed = is_compressed,
                    });
                },
                .SRL => {
                    // SRL rd, rs1, rs2 → 2-step: VirtualShiftRightBitmask(v0, rs2, 0) + VirtualSRL(rd, rs1, v0)
                    const rd = switch (jolt_instr.operands) {
                        .FormatR => |r| r.rd,
                        else => 0,
                    };
                    const rs1 = switch (jolt_instr.operands) {
                        .FormatR => |r| r.rs1,
                        else => 0,
                    };
                    const rs2 = switch (jolt_instr.operands) {
                        .FormatR => |r| r.rs2,
                        else => 0,
                    };
                    const v0: u8 = 40;
                    // Step 1: VirtualShiftRightBitmask(v0, rs2, 0)
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualShiftRightBitmask,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = v0, .rs1 = rs2, .imm = 0 } },
                        .virtual_sequence_remaining = 1,
                        .is_first_in_sequence = true,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 2: VirtualSRL(rd, rs1, v0)
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualSRL,
                        .address = addr,
                        .operands = .{ .FormatR = .{ .rd = rd, .rs1 = rs1, .rs2 = v0 } },
                        .virtual_sequence_remaining = 0,
                        .is_first_in_sequence = false,
                        .is_compressed = is_compressed,
                    });
                },
                .SRA => {
                    // SRA rd, rs1, rs2 → 2-step: VirtualShiftRightBitmask(v0, rs2, 0) + VirtualSRA(rd, rs1, v0)
                    const rd = switch (jolt_instr.operands) {
                        .FormatR => |r| r.rd,
                        else => 0,
                    };
                    const rs1 = switch (jolt_instr.operands) {
                        .FormatR => |r| r.rs1,
                        else => 0,
                    };
                    const rs2 = switch (jolt_instr.operands) {
                        .FormatR => |r| r.rs2,
                        else => 0,
                    };
                    const v0: u8 = 40;
                    // Step 1: VirtualShiftRightBitmask(v0, rs2, 0)
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualShiftRightBitmask,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = v0, .rs1 = rs2, .imm = 0 } },
                        .virtual_sequence_remaining = 1,
                        .is_first_in_sequence = true,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 2: VirtualSRA(rd, rs1, v0)
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualSRA,
                        .address = addr,
                        .operands = .{ .FormatR = .{ .rd = rd, .rs1 = rs1, .rs2 = v0 } },
                        .virtual_sequence_remaining = 0,
                        .is_first_in_sequence = false,
                        .is_compressed = is_compressed,
                    });
                },
                .SRAI => {
                    // SRAI rd, rs1, shamt → VirtualSRAI(rd, rs1, bitmask)
                    // Single virtual instruction (same pattern as SRLI → VirtualSRLI)
                    const raw_imm = switch (jolt_instr.operands) {
                        .FormatI => |i| i.imm,
                        else => 0,
                    };
                    const rd = switch (jolt_instr.operands) {
                        .FormatI => |i| i.rd,
                        else => 0,
                    };
                    const rs1_val = switch (jolt_instr.operands) {
                        .FormatI => |i| i.rs1,
                        else => 0,
                    };
                    const shift: u7 = @intCast(raw_imm & 0x3f);
                    const ones: u128 = (@as(u128, 1) << @intCast(64 - @as(u8, shift))) - 1;
                    const bitmask: u64 = @truncate(ones << shift);
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualSRAI,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = rd, .rs1 = rs1_val, .imm = bitmask } },
                        .virtual_sequence_remaining = 0,
                        .is_first_in_sequence = true,
                        .is_compressed = is_compressed,
                    });
                },
                .LB, .LBU => {
                    // LB/LBU rd, rs1, imm → 8 flat steps
                    const rd = switch (jolt_instr.operands) {
                        .FormatLoad => |l| l.rd,
                        else => 0,
                    };
                    const rs1 = switch (jolt_instr.operands) {
                        .FormatLoad => |l| l.rs1,
                        else => 0,
                    };
                    const imm = switch (jolt_instr.operands) {
                        .FormatLoad => |l| l.imm,
                        else => 0,
                    };
                    const v0: u8 = 40;
                    const v1: u8 = 41;
                    const v2: u8 = 42; // allocated by inner SLL
                    const total_steps: u16 = 7; // vsr counts from 7 down to 0
                    // Step 1: ADDI(v0, rs1, imm)
                    try self.bytecode.append(allocator, .{
                        .variant = .ADDI,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = v0, .rs1 = rs1, .imm = @bitCast(@as(i64, imm)) } },
                        .virtual_sequence_remaining = total_steps,
                        .is_first_in_sequence = true,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 2: ANDI(v1, v0, -8)
                    try self.bytecode.append(allocator, .{
                        .variant = .ANDI,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = v1, .rs1 = v0, .imm = @bitCast(@as(i64, -8)) } },
                        .virtual_sequence_remaining = total_steps - 1,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 3: LD(v1, v1, 0) — MEMORY READ
                    try self.bytecode.append(allocator, .{
                        .variant = .LD,
                        .address = addr,
                        .operands = .{ .FormatLoad = .{ .rd = v1, .rs1 = v1, .imm = 0 } },
                        .virtual_sequence_remaining = total_steps - 2,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 4: XORI(v0, v0, 7)
                    try self.bytecode.append(allocator, .{
                        .variant = .XORI,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = v0, .rs1 = v0, .imm = 7 } },
                        .virtual_sequence_remaining = total_steps - 3,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 5: VirtualMULI(v0, v0, 8) — from SLLI v0, v0, 3
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualMULI,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = v0, .rs1 = v0, .imm = 8 } },
                        .virtual_sequence_remaining = total_steps - 4,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 6: VirtualPow2(v2, v0, 0) — from SLL expansion step 1
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualPow2,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = v2, .rs1 = v0, .imm = 0 } },
                        .virtual_sequence_remaining = total_steps - 5,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 7: MUL(v1, v1, v2) — from SLL expansion step 2
                    try self.bytecode.append(allocator, .{
                        .variant = .MUL,
                        .address = addr,
                        .operands = .{ .FormatR = .{ .rd = v1, .rs1 = v1, .rs2 = v2 } },
                        .virtual_sequence_remaining = total_steps - 6,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 8: VirtualSRAI/VirtualSRLI(rd, v1, bitmask_56)
                    const shift_56: u7 = 56;
                    const ones_56: u128 = (@as(u128, 1) << @intCast(64 - @as(u8, shift_56))) - 1;
                    const bitmask_56: u64 = @truncate(ones_56 << shift_56);
                    if (jolt_instr.variant == .LB) {
                        try self.bytecode.append(allocator, .{
                            .variant = .VirtualSRAI,
                            .address = addr,
                            .operands = .{ .FormatI = .{ .rd = rd, .rs1 = v1, .imm = bitmask_56 } },
                            .virtual_sequence_remaining = 0,
                            .is_first_in_sequence = false,
                            .is_compressed = is_compressed,
                        });
                    } else {
                        // LBU: logical right shift (zero-extend)
                        try self.bytecode.append(allocator, .{
                            .variant = .VirtualSRLI,
                            .address = addr,
                            .operands = .{ .FormatI = .{ .rd = rd, .rs1 = v1, .imm = bitmask_56 } },
                            .virtual_sequence_remaining = 0,
                            .is_first_in_sequence = false,
                            .is_compressed = is_compressed,
                        });
                    }
                },
                .LH, .LHU => {
                    // LH/LHU rd, rs1, imm → 9 flat steps (includes alignment assertion)
                    const rd = switch (jolt_instr.operands) {
                        .FormatLoad => |l| l.rd,
                        else => 0,
                    };
                    const rs1 = switch (jolt_instr.operands) {
                        .FormatLoad => |l| l.rs1,
                        else => 0,
                    };
                    const imm = switch (jolt_instr.operands) {
                        .FormatLoad => |l| l.imm,
                        else => 0,
                    };
                    const v0: u8 = 40;
                    const v1: u8 = 41;
                    const v2: u8 = 42;
                    const total_steps: u16 = 8; // vsr counts from 8 down to 0
                    // Step 1: VirtualAssertHalfwordAlignment(rs1, imm)
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualAssertHalfwordAlignment,
                        .address = addr,
                        .operands = .{ .FormatAssert = .{ .rs1 = rs1, .imm = @as(i64, imm) } },
                        .virtual_sequence_remaining = total_steps,
                        .is_first_in_sequence = true,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 2: ADDI(v0, rs1, imm)
                    try self.bytecode.append(allocator, .{
                        .variant = .ADDI,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = v0, .rs1 = rs1, .imm = @bitCast(@as(i64, imm)) } },
                        .virtual_sequence_remaining = total_steps - 1,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 3: ANDI(v1, v0, -8)
                    try self.bytecode.append(allocator, .{
                        .variant = .ANDI,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = v1, .rs1 = v0, .imm = @bitCast(@as(i64, -8)) } },
                        .virtual_sequence_remaining = total_steps - 2,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 4: LD(v1, v1, 0) — MEMORY READ
                    try self.bytecode.append(allocator, .{
                        .variant = .LD,
                        .address = addr,
                        .operands = .{ .FormatLoad = .{ .rd = v1, .rs1 = v1, .imm = 0 } },
                        .virtual_sequence_remaining = total_steps - 3,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 5: XORI(v0, v0, 6)
                    try self.bytecode.append(allocator, .{
                        .variant = .XORI,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = v0, .rs1 = v0, .imm = 6 } },
                        .virtual_sequence_remaining = total_steps - 4,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 6: VirtualMULI(v0, v0, 8) — from SLLI v0, v0, 3
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualMULI,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = v0, .rs1 = v0, .imm = 8 } },
                        .virtual_sequence_remaining = total_steps - 5,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 7: VirtualPow2(v2, v0, 0) — from SLL step 1
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualPow2,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = v2, .rs1 = v0, .imm = 0 } },
                        .virtual_sequence_remaining = total_steps - 6,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 8: MUL(v1, v1, v2) — from SLL step 2
                    try self.bytecode.append(allocator, .{
                        .variant = .MUL,
                        .address = addr,
                        .operands = .{ .FormatR = .{ .rd = v1, .rs1 = v1, .rs2 = v2 } },
                        .virtual_sequence_remaining = total_steps - 7,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 9: VirtualSRAI/VirtualSRLI(rd, v1, bitmask_48)
                    const shift_48: u7 = 48;
                    const ones_48: u128 = (@as(u128, 1) << @intCast(64 - @as(u8, shift_48))) - 1;
                    const bitmask_48: u64 = @truncate(ones_48 << shift_48);
                    if (jolt_instr.variant == .LH) {
                        try self.bytecode.append(allocator, .{
                            .variant = .VirtualSRAI,
                            .address = addr,
                            .operands = .{ .FormatI = .{ .rd = rd, .rs1 = v1, .imm = bitmask_48 } },
                            .virtual_sequence_remaining = 0,
                            .is_first_in_sequence = false,
                            .is_compressed = is_compressed,
                        });
                    } else {
                        try self.bytecode.append(allocator, .{
                            .variant = .VirtualSRLI,
                            .address = addr,
                            .operands = .{ .FormatI = .{ .rd = rd, .rs1 = v1, .imm = bitmask_48 } },
                            .virtual_sequence_remaining = 0,
                            .is_first_in_sequence = false,
                            .is_compressed = is_compressed,
                        });
                    }
                },
                .LW => {
                    // LW rd, rs1, imm → 8 flat steps (RV64: with alignment assert, SRL, sign-extend)
                    const rd = switch (jolt_instr.operands) {
                        .FormatLoad => |l| l.rd,
                        else => 0,
                    };
                    const rs1 = switch (jolt_instr.operands) {
                        .FormatLoad => |l| l.rs1,
                        else => 0,
                    };
                    const imm = switch (jolt_instr.operands) {
                        .FormatLoad => |l| l.imm,
                        else => 0,
                    };
                    const v0: u8 = 40;
                    const v1: u8 = 41;
                    const v2: u8 = 42; // allocated by inner SRL
                    const total_steps: u16 = 7;
                    // Step 1: VirtualAssertWordAlignment(rs1, imm)
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualAssertWordAlignment,
                        .address = addr,
                        .operands = .{ .FormatAssert = .{ .rs1 = rs1, .imm = @as(i64, imm) } },
                        .virtual_sequence_remaining = total_steps,
                        .is_first_in_sequence = true,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 2: ADDI(v0, rs1, imm)
                    try self.bytecode.append(allocator, .{
                        .variant = .ADDI,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = v0, .rs1 = rs1, .imm = @bitCast(@as(i64, imm)) } },
                        .virtual_sequence_remaining = total_steps - 1,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 3: ANDI(v1, v0, -8)
                    try self.bytecode.append(allocator, .{
                        .variant = .ANDI,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = v1, .rs1 = v0, .imm = @bitCast(@as(i64, -8)) } },
                        .virtual_sequence_remaining = total_steps - 2,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 4: LD(v1, v1, 0)
                    try self.bytecode.append(allocator, .{
                        .variant = .LD,
                        .address = addr,
                        .operands = .{ .FormatLoad = .{ .rd = v1, .rs1 = v1, .imm = 0 } },
                        .virtual_sequence_remaining = total_steps - 3,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 5: VirtualMULI(v0, v0, 8) — from SLLI v0, v0, 3 (NO XORI for LW)
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualMULI,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = v0, .rs1 = v0, .imm = 8 } },
                        .virtual_sequence_remaining = total_steps - 4,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 6: VirtualShiftRightBitmask(v2, v0, 0) — from SRL step 1
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualShiftRightBitmask,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = v2, .rs1 = v0, .imm = 0 } },
                        .virtual_sequence_remaining = total_steps - 5,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 7: VirtualSRL(v1, v1, v2) — from SRL step 2
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualSRL,
                        .address = addr,
                        .operands = .{ .FormatR = .{ .rd = v1, .rs1 = v1, .rs2 = v2 } },
                        .virtual_sequence_remaining = total_steps - 6,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 8: VirtualSignExtendWord(rd, v1, 0)
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualSignExtendWord,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = rd, .rs1 = v1, .imm = 0 } },
                        .virtual_sequence_remaining = 0,
                        .is_first_in_sequence = false,
                        .is_compressed = is_compressed,
                    });
                },
                .LWU => {
                    // LWU rd, rs1, imm → 9 flat steps (with XORI, SLL, SRLI)
                    const rd = switch (jolt_instr.operands) {
                        .FormatLoad => |l| l.rd,
                        else => 0,
                    };
                    const rs1 = switch (jolt_instr.operands) {
                        .FormatLoad => |l| l.rs1,
                        else => 0,
                    };
                    const imm = switch (jolt_instr.operands) {
                        .FormatLoad => |l| l.imm,
                        else => 0,
                    };
                    const v0: u8 = 40;
                    const v1: u8 = 41;
                    const v2: u8 = 42; // from inner SLL
                    const total_steps: u16 = 8;
                    // Step 1: VirtualAssertWordAlignment(rs1, imm)
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualAssertWordAlignment,
                        .address = addr,
                        .operands = .{ .FormatAssert = .{ .rs1 = rs1, .imm = @as(i64, imm) } },
                        .virtual_sequence_remaining = total_steps,
                        .is_first_in_sequence = true,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 2: ADDI(v0, rs1, imm)
                    try self.bytecode.append(allocator, .{
                        .variant = .ADDI,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = v0, .rs1 = rs1, .imm = @bitCast(@as(i64, imm)) } },
                        .virtual_sequence_remaining = total_steps - 1,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 3: ANDI(v1, v0, -8)
                    try self.bytecode.append(allocator, .{
                        .variant = .ANDI,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = v1, .rs1 = v0, .imm = @bitCast(@as(i64, -8)) } },
                        .virtual_sequence_remaining = total_steps - 2,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 4: LD(v1, v1, 0)
                    try self.bytecode.append(allocator, .{
                        .variant = .LD,
                        .address = addr,
                        .operands = .{ .FormatLoad = .{ .rd = v1, .rs1 = v1, .imm = 0 } },
                        .virtual_sequence_remaining = total_steps - 3,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 5: XORI(v0, v0, 4)
                    try self.bytecode.append(allocator, .{
                        .variant = .XORI,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = v0, .rs1 = v0, .imm = 4 } },
                        .virtual_sequence_remaining = total_steps - 4,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 6: VirtualMULI(v0, v0, 8) — from SLLI
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualMULI,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = v0, .rs1 = v0, .imm = 8 } },
                        .virtual_sequence_remaining = total_steps - 5,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 7: VirtualPow2(v2, v0, 0) — from SLL step 1
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualPow2,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = v2, .rs1 = v0, .imm = 0 } },
                        .virtual_sequence_remaining = total_steps - 6,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 8: MUL(v1, v1, v2) — from SLL step 2
                    try self.bytecode.append(allocator, .{
                        .variant = .MUL,
                        .address = addr,
                        .operands = .{ .FormatR = .{ .rd = v1, .rs1 = v1, .rs2 = v2 } },
                        .virtual_sequence_remaining = total_steps - 7,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 9: VirtualSRLI(rd, v1, bitmask_32)
                    const shift_32: u7 = 32;
                    const ones_32: u128 = (@as(u128, 1) << @intCast(64 - @as(u8, shift_32))) - 1;
                    const bitmask_32: u64 = @truncate(ones_32 << shift_32);
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualSRLI,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = rd, .rs1 = v1, .imm = bitmask_32 } },
                        .virtual_sequence_remaining = 0,
                        .is_first_in_sequence = false,
                        .is_compressed = is_compressed,
                    });
                },
                .SB => {
                    // SB rs2, rs1, imm → 13 flat steps
                    const rs1 = switch (jolt_instr.operands) {
                        .FormatS => |s| s.rs1,
                        else => 0,
                    };
                    const rs2 = switch (jolt_instr.operands) {
                        .FormatS => |s| s.rs2,
                        else => 0,
                    };
                    const imm = switch (jolt_instr.operands) {
                        .FormatS => |s| s.imm,
                        else => 0,
                    };
                    const v0: u8 = 40;
                    const v1: u8 = 41;
                    const v2: u8 = 42;
                    const v3: u8 = 43;
                    const v4: u8 = 44; // from inner SLL #1
                    const v5: u8 = 45; // from inner SLL #2
                    const total_steps: u16 = 12;
                    // Step 1: ADDI(v0, rs1, imm)
                    try self.bytecode.append(allocator, .{
                        .variant = .ADDI,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = v0, .rs1 = rs1, .imm = @bitCast(@as(i64, imm)) } },
                        .virtual_sequence_remaining = total_steps,
                        .is_first_in_sequence = true,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 2: ANDI(v1, v0, -8)
                    try self.bytecode.append(allocator, .{
                        .variant = .ANDI,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = v1, .rs1 = v0, .imm = @bitCast(@as(i64, -8)) } },
                        .virtual_sequence_remaining = total_steps - 1,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 3: LD(v2, v1, 0) — MEMORY READ
                    try self.bytecode.append(allocator, .{
                        .variant = .LD,
                        .address = addr,
                        .operands = .{ .FormatLoad = .{ .rd = v2, .rs1 = v1, .imm = 0 } },
                        .virtual_sequence_remaining = total_steps - 2,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 4: VirtualMULI(v3, v0, 8) — from SLLI v3, v0, 3
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualMULI,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = v3, .rs1 = v0, .imm = 8 } },
                        .virtual_sequence_remaining = total_steps - 3,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 5: LUI(v0, 0xff)
                    try self.bytecode.append(allocator, .{
                        .variant = .LUI,
                        .address = addr,
                        .operands = .{ .FormatU = .{ .rd = v0, .imm = 0xff << 12 } },
                        .virtual_sequence_remaining = total_steps - 4,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 6: VirtualPow2(v4, v3, 0) — from SLL(v0, v0, v3) step 1
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualPow2,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = v4, .rs1 = v3, .imm = 0 } },
                        .virtual_sequence_remaining = total_steps - 5,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 7: MUL(v0, v0, v4) — from SLL step 2
                    try self.bytecode.append(allocator, .{
                        .variant = .MUL,
                        .address = addr,
                        .operands = .{ .FormatR = .{ .rd = v0, .rs1 = v0, .rs2 = v4 } },
                        .virtual_sequence_remaining = total_steps - 6,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 8: VirtualPow2(v5, v3, 0) — from SLL(v3, rs2, v3) step 1
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualPow2,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = v5, .rs1 = v3, .imm = 0 } },
                        .virtual_sequence_remaining = total_steps - 7,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 9: MUL(v3, rs2, v5) — from SLL step 2
                    try self.bytecode.append(allocator, .{
                        .variant = .MUL,
                        .address = addr,
                        .operands = .{ .FormatR = .{ .rd = v3, .rs1 = rs2, .rs2 = v5 } },
                        .virtual_sequence_remaining = total_steps - 8,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 10: XOR(v3, v2, v3)
                    try self.bytecode.append(allocator, .{
                        .variant = .XOR,
                        .address = addr,
                        .operands = .{ .FormatR = .{ .rd = v3, .rs1 = v2, .rs2 = v3 } },
                        .virtual_sequence_remaining = total_steps - 9,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 11: AND(v3, v3, v0)
                    try self.bytecode.append(allocator, .{
                        .variant = .AND,
                        .address = addr,
                        .operands = .{ .FormatR = .{ .rd = v3, .rs1 = v3, .rs2 = v0 } },
                        .virtual_sequence_remaining = total_steps - 10,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 12: XOR(v2, v2, v3)
                    try self.bytecode.append(allocator, .{
                        .variant = .XOR,
                        .address = addr,
                        .operands = .{ .FormatR = .{ .rd = v2, .rs1 = v2, .rs2 = v3 } },
                        .virtual_sequence_remaining = total_steps - 11,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 13: SD(v1, v2, 0) — MEMORY WRITE
                    try self.bytecode.append(allocator, .{
                        .variant = .SD,
                        .address = addr,
                        .operands = .{ .FormatS = .{ .rs1 = v1, .rs2 = v2, .imm = 0 } },
                        .virtual_sequence_remaining = 0,
                        .is_first_in_sequence = false,
                        .is_compressed = is_compressed,
                    });
                },
                .SH => {
                    // SH rs2, rs1, imm → 14 flat steps (SB + alignment assertion + 0xffff mask)
                    const rs1 = switch (jolt_instr.operands) {
                        .FormatS => |s| s.rs1,
                        else => 0,
                    };
                    const rs2 = switch (jolt_instr.operands) {
                        .FormatS => |s| s.rs2,
                        else => 0,
                    };
                    const imm = switch (jolt_instr.operands) {
                        .FormatS => |s| s.imm,
                        else => 0,
                    };
                    const v0: u8 = 40;
                    const v1: u8 = 41;
                    const v2: u8 = 42;
                    const v3: u8 = 43;
                    const v4: u8 = 44;
                    const v5: u8 = 45;
                    const total_steps: u16 = 13;
                    // Step 1: VirtualAssertHalfwordAlignment(rs1, imm)
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualAssertHalfwordAlignment,
                        .address = addr,
                        .operands = .{ .FormatAssert = .{ .rs1 = rs1, .imm = @as(i64, imm) } },
                        .virtual_sequence_remaining = total_steps,
                        .is_first_in_sequence = true,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 2: ADDI(v0, rs1, imm)
                    try self.bytecode.append(allocator, .{
                        .variant = .ADDI,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = v0, .rs1 = rs1, .imm = @bitCast(@as(i64, imm)) } },
                        .virtual_sequence_remaining = total_steps - 1,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 3: ANDI(v1, v0, -8)
                    try self.bytecode.append(allocator, .{
                        .variant = .ANDI,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = v1, .rs1 = v0, .imm = @bitCast(@as(i64, -8)) } },
                        .virtual_sequence_remaining = total_steps - 2,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 4: LD(v2, v1, 0) — MEMORY READ
                    try self.bytecode.append(allocator, .{
                        .variant = .LD,
                        .address = addr,
                        .operands = .{ .FormatLoad = .{ .rd = v2, .rs1 = v1, .imm = 0 } },
                        .virtual_sequence_remaining = total_steps - 3,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 5: VirtualMULI(v3, v0, 8) — from SLLI
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualMULI,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = v3, .rs1 = v0, .imm = 8 } },
                        .virtual_sequence_remaining = total_steps - 4,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 6: LUI(v0, 0xffff)
                    try self.bytecode.append(allocator, .{
                        .variant = .LUI,
                        .address = addr,
                        .operands = .{ .FormatU = .{ .rd = v0, .imm = 0xffff << 12 } },
                        .virtual_sequence_remaining = total_steps - 5,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 7: VirtualPow2(v4, v3, 0) — from SLL(v0, v0, v3)
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualPow2,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = v4, .rs1 = v3, .imm = 0 } },
                        .virtual_sequence_remaining = total_steps - 6,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 8: MUL(v0, v0, v4)
                    try self.bytecode.append(allocator, .{
                        .variant = .MUL,
                        .address = addr,
                        .operands = .{ .FormatR = .{ .rd = v0, .rs1 = v0, .rs2 = v4 } },
                        .virtual_sequence_remaining = total_steps - 7,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 9: VirtualPow2(v5, v3, 0) — from SLL(v3, rs2, v3)
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualPow2,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = v5, .rs1 = v3, .imm = 0 } },
                        .virtual_sequence_remaining = total_steps - 8,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 10: MUL(v3, rs2, v5)
                    try self.bytecode.append(allocator, .{
                        .variant = .MUL,
                        .address = addr,
                        .operands = .{ .FormatR = .{ .rd = v3, .rs1 = rs2, .rs2 = v5 } },
                        .virtual_sequence_remaining = total_steps - 9,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 11: XOR(v3, v2, v3)
                    try self.bytecode.append(allocator, .{
                        .variant = .XOR,
                        .address = addr,
                        .operands = .{ .FormatR = .{ .rd = v3, .rs1 = v2, .rs2 = v3 } },
                        .virtual_sequence_remaining = total_steps - 10,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 12: AND(v3, v3, v0)
                    try self.bytecode.append(allocator, .{
                        .variant = .AND,
                        .address = addr,
                        .operands = .{ .FormatR = .{ .rd = v3, .rs1 = v3, .rs2 = v0 } },
                        .virtual_sequence_remaining = total_steps - 11,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 13: XOR(v2, v2, v3)
                    try self.bytecode.append(allocator, .{
                        .variant = .XOR,
                        .address = addr,
                        .operands = .{ .FormatR = .{ .rd = v2, .rs1 = v2, .rs2 = v3 } },
                        .virtual_sequence_remaining = total_steps - 12,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 14: SD(v1, v2, 0) — MEMORY WRITE
                    try self.bytecode.append(allocator, .{
                        .variant = .SD,
                        .address = addr,
                        .operands = .{ .FormatS = .{ .rs1 = v1, .rs2 = v2, .imm = 0 } },
                        .virtual_sequence_remaining = 0,
                        .is_first_in_sequence = false,
                        .is_compressed = is_compressed,
                    });
                },
                .SW => {
                    // SW rs2, rs1, imm → 15 flat steps (RV64)
                    const rs1 = switch (jolt_instr.operands) {
                        .FormatS => |s| s.rs1,
                        else => 0,
                    };
                    const rs2 = switch (jolt_instr.operands) {
                        .FormatS => |s| s.rs2,
                        else => 0,
                    };
                    const imm = switch (jolt_instr.operands) {
                        .FormatS => |s| s.imm,
                        else => 0,
                    };
                    const v0: u8 = 40;
                    const v1: u8 = 41;
                    const v2: u8 = 42;
                    const v3: u8 = 43;
                    const v4: u8 = 44; // from inner SLL #1
                    const v5: u8 = 45; // from inner SLL #2
                    const total_steps: u16 = 14;
                    // Step 1: VirtualAssertWordAlignment(rs1, imm)
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualAssertWordAlignment,
                        .address = addr,
                        .operands = .{ .FormatAssert = .{ .rs1 = rs1, .imm = @as(i64, imm) } },
                        .virtual_sequence_remaining = total_steps,
                        .is_first_in_sequence = true,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 2: ADDI(v0, rs1, imm)
                    try self.bytecode.append(allocator, .{
                        .variant = .ADDI,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = v0, .rs1 = rs1, .imm = @bitCast(@as(i64, imm)) } },
                        .virtual_sequence_remaining = total_steps - 1,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 3: ANDI(v1, v0, -8)
                    try self.bytecode.append(allocator, .{
                        .variant = .ANDI,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = v1, .rs1 = v0, .imm = @bitCast(@as(i64, -8)) } },
                        .virtual_sequence_remaining = total_steps - 2,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 4: LD(v2, v1, 0) — MEMORY READ
                    try self.bytecode.append(allocator, .{
                        .variant = .LD,
                        .address = addr,
                        .operands = .{ .FormatLoad = .{ .rd = v2, .rs1 = v1, .imm = 0 } },
                        .virtual_sequence_remaining = total_steps - 3,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 5: VirtualMULI(v0, v0, 8) — from SLLI v0, v0, 3
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualMULI,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = v0, .rs1 = v0, .imm = 8 } },
                        .virtual_sequence_remaining = total_steps - 4,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 6: ORI(v3, x0, -1) — all 1s
                    try self.bytecode.append(allocator, .{
                        .variant = .ORI,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = v3, .rs1 = 0, .imm = @bitCast(@as(i64, -1)) } },
                        .virtual_sequence_remaining = total_steps - 5,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 7: VirtualSRLI(v3, v3, bitmask_32) — 32-bit mask
                    const shift_32: u7 = 32;
                    const ones_32: u128 = (@as(u128, 1) << @intCast(64 - @as(u8, shift_32))) - 1;
                    const bitmask_32: u64 = @truncate(ones_32 << shift_32);
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualSRLI,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = v3, .rs1 = v3, .imm = bitmask_32 } },
                        .virtual_sequence_remaining = total_steps - 6,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 8: VirtualPow2(v4, v0, 0) — from SLL(v3, v3, v0) step 1
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualPow2,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = v4, .rs1 = v0, .imm = 0 } },
                        .virtual_sequence_remaining = total_steps - 7,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 9: MUL(v3, v3, v4) — shifted 32-bit mask
                    try self.bytecode.append(allocator, .{
                        .variant = .MUL,
                        .address = addr,
                        .operands = .{ .FormatR = .{ .rd = v3, .rs1 = v3, .rs2 = v4 } },
                        .virtual_sequence_remaining = total_steps - 8,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 10: VirtualPow2(v5, v0, 0) — from SLL(v0, rs2, v0) step 1
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualPow2,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = v5, .rs1 = v0, .imm = 0 } },
                        .virtual_sequence_remaining = total_steps - 9,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 11: MUL(v0, rs2, v5) — shifted value
                    try self.bytecode.append(allocator, .{
                        .variant = .MUL,
                        .address = addr,
                        .operands = .{ .FormatR = .{ .rd = v0, .rs1 = rs2, .rs2 = v5 } },
                        .virtual_sequence_remaining = total_steps - 10,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 12: XOR(v0, v2, v0)
                    try self.bytecode.append(allocator, .{
                        .variant = .XOR,
                        .address = addr,
                        .operands = .{ .FormatR = .{ .rd = v0, .rs1 = v2, .rs2 = v0 } },
                        .virtual_sequence_remaining = total_steps - 11,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 13: AND(v0, v0, v3)
                    try self.bytecode.append(allocator, .{
                        .variant = .AND,
                        .address = addr,
                        .operands = .{ .FormatR = .{ .rd = v0, .rs1 = v0, .rs2 = v3 } },
                        .virtual_sequence_remaining = total_steps - 12,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 14: XOR(v2, v2, v0)
                    try self.bytecode.append(allocator, .{
                        .variant = .XOR,
                        .address = addr,
                        .operands = .{ .FormatR = .{ .rd = v2, .rs1 = v2, .rs2 = v0 } },
                        .virtual_sequence_remaining = total_steps - 13,
                        .is_first_in_sequence = false,
                        .is_compressed = false, // Only last entry in sequence gets is_compressed
                    });
                    // Step 15: SD(v1, v2, 0) — MEMORY WRITE
                    try self.bytecode.append(allocator, .{
                        .variant = .SD,
                        .address = addr,
                        .operands = .{ .FormatS = .{ .rs1 = v1, .rs2 = v2, .imm = 0 } },
                        .virtual_sequence_remaining = 0,
                        .is_first_in_sequence = false,
                        .is_compressed = is_compressed,
                    });
                },
                .UNIMPL => {
                    // Check if this is a Jolt inline instruction (opcode 0x0B with FormatInline operands)
                    if (jolt_instr.operands == .FormatInline) {
                        const il = jolt_instr.operands.FormatInline;
                        const is_sha256 = (il.funct7 == 0x00 and (il.funct3 == 0x00 or il.funct3 == 0x01));
                        if (is_sha256) {
                            const initial = (il.funct3 == 0x01);
                            // Build the SHA256 virtual instruction sequence.
                            // The sequence structure is deterministic (independent of register values),
                            // so we use the actual rs1/rs2 from the instruction encoding.
                            var sequence = try sha256_inline.buildSha256Sequence(allocator, il.rs1, il.rs2, initial);
                            defer sequence.deinit(allocator);

                            const seq_len = sequence.items.len;
                            for (sequence.items, 0..) |instr_item, idx| {
                                const vsr: u16 = @intCast(seq_len - 1 - idx);
                                const is_first_step = (idx == 0);
                                const is_last_step = (idx == seq_len - 1);

                                // Map each InlineInstr to a JoltInstruction variant+operands
                                const jolt_entry: JoltInstruction = switch (instr_item.kind) {
                                    .ADD => .{
                                        .variant = .ADD,
                                        .address = addr,
                                        .operands = .{ .FormatR = .{ .rd = instr_item.rd, .rs1 = instr_item.rs1, .rs2 = instr_item.rs2 } },
                                        .virtual_sequence_remaining = vsr,
                                        .is_first_in_sequence = is_first_step,
                                        .is_compressed = if (is_last_step) is_compressed else false,
                                    },
                                    .ADDI => .{
                                        .variant = .ADDI,
                                        .address = addr,
                                        .operands = .{ .FormatI = .{ .rd = instr_item.rd, .rs1 = instr_item.rs1, .imm = @bitCast(instr_item.imm) } },
                                        .virtual_sequence_remaining = vsr,
                                        .is_first_in_sequence = is_first_step,
                                        .is_compressed = if (is_last_step) is_compressed else false,
                                    },
                                    .XOR => .{
                                        .variant = .XOR,
                                        .address = addr,
                                        .operands = .{ .FormatR = .{ .rd = instr_item.rd, .rs1 = instr_item.rs1, .rs2 = instr_item.rs2 } },
                                        .virtual_sequence_remaining = vsr,
                                        .is_first_in_sequence = is_first_step,
                                        .is_compressed = if (is_last_step) is_compressed else false,
                                    },
                                    .XORI => .{
                                        .variant = .XORI,
                                        .address = addr,
                                        .operands = .{ .FormatI = .{ .rd = instr_item.rd, .rs1 = instr_item.rs1, .imm = @bitCast(instr_item.imm) } },
                                        .virtual_sequence_remaining = vsr,
                                        .is_first_in_sequence = is_first_step,
                                        .is_compressed = if (is_last_step) is_compressed else false,
                                    },
                                    .AND => .{
                                        .variant = .AND,
                                        .address = addr,
                                        .operands = .{ .FormatR = .{ .rd = instr_item.rd, .rs1 = instr_item.rs1, .rs2 = instr_item.rs2 } },
                                        .virtual_sequence_remaining = vsr,
                                        .is_first_in_sequence = is_first_step,
                                        .is_compressed = if (is_last_step) is_compressed else false,
                                    },
                                    .ANDI => .{
                                        .variant = .ANDI,
                                        .address = addr,
                                        .operands = .{ .FormatI = .{ .rd = instr_item.rd, .rs1 = instr_item.rs1, .imm = @bitCast(instr_item.imm) } },
                                        .virtual_sequence_remaining = vsr,
                                        .is_first_in_sequence = is_first_step,
                                        .is_compressed = if (is_last_step) is_compressed else false,
                                    },
                                    .OR => .{
                                        .variant = .OR,
                                        .address = addr,
                                        .operands = .{ .FormatR = .{ .rd = instr_item.rd, .rs1 = instr_item.rs1, .rs2 = instr_item.rs2 } },
                                        .virtual_sequence_remaining = vsr,
                                        .is_first_in_sequence = is_first_step,
                                        .is_compressed = if (is_last_step) is_compressed else false,
                                    },
                                    .ANDN => .{
                                        .variant = .ANDN,
                                        .address = addr,
                                        .operands = .{ .FormatR = .{ .rd = instr_item.rd, .rs1 = instr_item.rs1, .rs2 = instr_item.rs2 } },
                                        .virtual_sequence_remaining = vsr,
                                        .is_first_in_sequence = is_first_step,
                                        .is_compressed = if (is_last_step) is_compressed else false,
                                    },
                                    .VirtualMULI => blk_vmuli: {
                                        // sha256_inline.zig stores the shamt in instr_item.imm
                                        // (treated as SLLI by N). Jolt's VirtualMULI expects the
                                        // multiplier (1 << shamt) in the instruction's operand
                                        // imm field. Convert here to match standalone SLLI path.
                                        const shamt_val: u8 = @intCast(instr_item.imm & 0x3F);
                                        const multiplier_val: u64 = @as(u64, 1) << @intCast(shamt_val);
                                        break :blk_vmuli .{
                                            .variant = .VirtualMULI,
                                            .address = addr,
                                            .operands = .{ .FormatI = .{ .rd = instr_item.rd, .rs1 = instr_item.rs1, .imm = multiplier_val } },
                                            .virtual_sequence_remaining = vsr,
                                            .is_first_in_sequence = is_first_step,
                                            .is_compressed = if (is_last_step) is_compressed else false,
                                        };
                                    },
                                    .VirtualSRLI => .{
                                        .variant = .VirtualSRLI,
                                        .address = addr,
                                        .operands = .{ .FormatI = .{ .rd = instr_item.rd, .rs1 = instr_item.rs1, .imm = instr_item.imm } },
                                        .virtual_sequence_remaining = vsr,
                                        .is_first_in_sequence = is_first_step,
                                        .is_compressed = if (is_last_step) is_compressed else false,
                                    },
                                    .VirtualSignExtendWord => .{
                                        .variant = .VirtualSignExtendWord,
                                        .address = addr,
                                        .operands = .{ .FormatI = .{ .rd = instr_item.rd, .rs1 = instr_item.rs1, .imm = 0 } },
                                        .virtual_sequence_remaining = vsr,
                                        .is_first_in_sequence = is_first_step,
                                        .is_compressed = if (is_last_step) is_compressed else false,
                                    },
                                    .VirtualZeroExtendWord => .{
                                        .variant = .VirtualZeroExtendWord,
                                        .address = addr,
                                        .operands = .{ .FormatI = .{ .rd = instr_item.rd, .rs1 = instr_item.rs1, .imm = 0 } },
                                        .virtual_sequence_remaining = vsr,
                                        .is_first_in_sequence = is_first_step,
                                        .is_compressed = if (is_last_step) is_compressed else false,
                                    },
                                    .VirtualROTRIW => .{
                                        .variant = .VirtualROTRIW,
                                        .address = addr,
                                        .operands = .{ .FormatVirtualRightShiftI = .{
                                            .rd = instr_item.rd,
                                            .rs1 = instr_item.rs1,
                                            .imm = instr_item.imm, // rotation amount stored as imm
                                        } },
                                        .virtual_sequence_remaining = vsr,
                                        .is_first_in_sequence = is_first_step,
                                        .is_compressed = if (is_last_step) is_compressed else false,
                                    },
                                    .LD => .{
                                        .variant = .LD,
                                        .address = addr,
                                        .operands = .{ .FormatLoad = .{ .rd = instr_item.rd, .rs1 = instr_item.rs1, .imm = @bitCast(instr_item.imm) } },
                                        .virtual_sequence_remaining = vsr,
                                        .is_first_in_sequence = is_first_step,
                                        .is_compressed = if (is_last_step) is_compressed else false,
                                    },
                                    .SD => .{
                                        .variant = .SD,
                                        .address = addr,
                                        .operands = .{ .FormatS = .{ .rs1 = instr_item.rs1, .rs2 = instr_item.rs2, .imm = @bitCast(instr_item.imm) } },
                                        .virtual_sequence_remaining = vsr,
                                        .is_first_in_sequence = is_first_step,
                                        .is_compressed = if (is_last_step) is_compressed else false,
                                    },
                                };
                                try self.bytecode.append(allocator, jolt_entry);
                            }
                        } else {
                            // Unsupported inline type: append as-is (single UNIMPL entry)
                            try self.bytecode.append(allocator, jolt_instr);
                        }
                    } else {
                        // Regular UNIMPL (unknown opcode): append as-is
                        try self.bytecode.append(allocator, jolt_instr);
                    }
                },
                .CSRRW => {
                    // CSRRW rd, csr, rs1 → decomposed into ADDI virtual sequence
                    // CSR address (bits[31:20]) maps to virtual registers 34-39
                    const rd_val = switch (jolt_instr.operands) {
                        .FormatI => |i| i.rd,
                        else => 0,
                    };
                    const rs1_val = switch (jolt_instr.operands) {
                        .FormatI => |i| i.rs1,
                        else => 0,
                    };
                    const csr_imm = switch (jolt_instr.operands) {
                        .FormatI => |i| i.imm,
                        else => 0,
                    };
                    const csr_addr: u12 = @truncate(csr_imm & 0xFFF);
                    const virtual_reg = csrToVirtualReg(csr_addr);
                    const temp_reg: u8 = 40; // v40 for temp

                    if (rd_val == 0) {
                        // csrw pseudo: ADDI virtual_reg, rs1, 0 (1 step)
                        try self.bytecode.append(allocator, .{
                            .variant = .ADDI,
                            .address = addr,
                            .operands = .{ .FormatI = .{ .rd = virtual_reg, .rs1 = rs1_val, .imm = 0 } },
                            .virtual_sequence_remaining = 0,
                            .is_first_in_sequence = true,
                            .is_compressed = is_compressed,
                        });
                    } else if (rd_val == rs1_val) {
                        // rd == rs1: need temp (3 steps)
                        // Step 1: ADDI temp, rs1, 0
                        try self.bytecode.append(allocator, .{
                            .variant = .ADDI,
                            .address = addr,
                            .operands = .{ .FormatI = .{ .rd = temp_reg, .rs1 = rs1_val, .imm = 0 } },
                            .virtual_sequence_remaining = 2,
                            .is_first_in_sequence = true,
                            .is_compressed = false,
                        });
                        // Step 2: ADDI rd, virtual_reg, 0
                        try self.bytecode.append(allocator, .{
                            .variant = .ADDI,
                            .address = addr,
                            .operands = .{ .FormatI = .{ .rd = rd_val, .rs1 = virtual_reg, .imm = 0 } },
                            .virtual_sequence_remaining = 1,
                            .is_first_in_sequence = false,
                            .is_compressed = false,
                        });
                        // Step 3: ADDI virtual_reg, temp, 0
                        try self.bytecode.append(allocator, .{
                            .variant = .ADDI,
                            .address = addr,
                            .operands = .{ .FormatI = .{ .rd = virtual_reg, .rs1 = temp_reg, .imm = 0 } },
                            .virtual_sequence_remaining = 0,
                            .is_first_in_sequence = false,
                            .is_compressed = is_compressed,
                        });
                    } else {
                        // rd != rs1, rd != 0: 2 steps
                        // Step 1: ADDI rd, virtual_reg, 0
                        try self.bytecode.append(allocator, .{
                            .variant = .ADDI,
                            .address = addr,
                            .operands = .{ .FormatI = .{ .rd = rd_val, .rs1 = virtual_reg, .imm = 0 } },
                            .virtual_sequence_remaining = 1,
                            .is_first_in_sequence = true,
                            .is_compressed = false,
                        });
                        // Step 2: ADDI virtual_reg, rs1, 0
                        try self.bytecode.append(allocator, .{
                            .variant = .ADDI,
                            .address = addr,
                            .operands = .{ .FormatI = .{ .rd = virtual_reg, .rs1 = rs1_val, .imm = 0 } },
                            .virtual_sequence_remaining = 0,
                            .is_first_in_sequence = false,
                            .is_compressed = is_compressed,
                        });
                    }
                },
                .CSRRS => {
                    // CSRRS rd, csr, rs1 → decomposed into ADDI/OR virtual sequence
                    const rd_val = switch (jolt_instr.operands) {
                        .FormatI => |i| i.rd,
                        else => 0,
                    };
                    const rs1_val = switch (jolt_instr.operands) {
                        .FormatI => |i| i.rs1,
                        else => 0,
                    };
                    const csr_imm = switch (jolt_instr.operands) {
                        .FormatI => |i| i.imm,
                        else => 0,
                    };
                    const csr_addr: u12 = @truncate(csr_imm & 0xFFF);
                    const virtual_reg = csrToVirtualReg(csr_addr);
                    const temp_reg: u8 = 40; // v40 for temp

                    if (rs1_val == 0) {
                        // csrr pseudo (read-only): ADDI rd, virtual_reg, 0 (1 step)
                        try self.bytecode.append(allocator, .{
                            .variant = .ADDI,
                            .address = addr,
                            .operands = .{ .FormatI = .{ .rd = rd_val, .rs1 = virtual_reg, .imm = 0 } },
                            .virtual_sequence_remaining = 0,
                            .is_first_in_sequence = true,
                            .is_compressed = is_compressed,
                        });
                    } else if (rd_val == 0) {
                        // csrs pseudo (set-only): OR virtual_reg, virtual_reg, rs1 (1 step)
                        try self.bytecode.append(allocator, .{
                            .variant = .OR,
                            .address = addr,
                            .operands = .{ .FormatR = .{ .rd = virtual_reg, .rs1 = virtual_reg, .rs2 = rs1_val } },
                            .virtual_sequence_remaining = 0,
                            .is_first_in_sequence = true,
                            .is_compressed = is_compressed,
                        });
                    } else if (rd_val == rs1_val) {
                        // rd == rs1: need temp (3 steps)
                        // Step 1: ADDI temp, rs1, 0
                        try self.bytecode.append(allocator, .{
                            .variant = .ADDI,
                            .address = addr,
                            .operands = .{ .FormatI = .{ .rd = temp_reg, .rs1 = rs1_val, .imm = 0 } },
                            .virtual_sequence_remaining = 2,
                            .is_first_in_sequence = true,
                            .is_compressed = false,
                        });
                        // Step 2: ADDI rd, virtual_reg, 0
                        try self.bytecode.append(allocator, .{
                            .variant = .ADDI,
                            .address = addr,
                            .operands = .{ .FormatI = .{ .rd = rd_val, .rs1 = virtual_reg, .imm = 0 } },
                            .virtual_sequence_remaining = 1,
                            .is_first_in_sequence = false,
                            .is_compressed = false,
                        });
                        // Step 3: OR virtual_reg, virtual_reg, temp
                        try self.bytecode.append(allocator, .{
                            .variant = .OR,
                            .address = addr,
                            .operands = .{ .FormatR = .{ .rd = virtual_reg, .rs1 = virtual_reg, .rs2 = temp_reg } },
                            .virtual_sequence_remaining = 0,
                            .is_first_in_sequence = false,
                            .is_compressed = is_compressed,
                        });
                    } else {
                        // rd != rs1, both nonzero: 2 steps
                        // Step 1: ADDI rd, virtual_reg, 0
                        try self.bytecode.append(allocator, .{
                            .variant = .ADDI,
                            .address = addr,
                            .operands = .{ .FormatI = .{ .rd = rd_val, .rs1 = virtual_reg, .imm = 0 } },
                            .virtual_sequence_remaining = 1,
                            .is_first_in_sequence = true,
                            .is_compressed = false,
                        });
                        // Step 2: OR virtual_reg, virtual_reg, rs1
                        try self.bytecode.append(allocator, .{
                            .variant = .OR,
                            .address = addr,
                            .operands = .{ .FormatR = .{ .rd = virtual_reg, .rs1 = virtual_reg, .rs2 = rs1_val } },
                            .virtual_sequence_remaining = 0,
                            .is_first_in_sequence = false,
                            .is_compressed = is_compressed,
                        });
                    }
                },
                .MRET => {
                    // MRET → JALR v40, mepc(vr36), 0 (1 step)
                    const mepc_reg: u8 = 36; // mepc virtual register
                    const temp_reg: u8 = 40; // v40 for return address (unused but matches Jolt)
                    try self.bytecode.append(allocator, .{
                        .variant = .JALR,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = temp_reg, .rs1 = mepc_reg, .imm = 0 } },
                        .virtual_sequence_remaining = 0,
                        .is_first_in_sequence = true,
                        .is_compressed = is_compressed,
                    });
                },
                .AdviceLB, .AdviceLH, .AdviceLW => {
                    // AdviceLB/LH/LW: inline-expanded to VirtualAdvice + SLLI + SRAI
                    // where SLLI → VirtualMULI and SRAI → VirtualSRAI (matching our
                    // internal decomposition). 3-cycle virtual sequence: vsr=2,1,0.
                    const num_bytes: u8 = switch (jolt_instr.variant) {
                        .AdviceLB => 1,
                        .AdviceLH => 2,
                        .AdviceLW => 4,
                        else => unreachable,
                    };
                    const shift: u8 = 64 - num_bytes * 8;
                    const rd_val = switch (jolt_instr.operands) {
                        .FormatI => |i| i.rd,
                        else => 0,
                    };
                    // Step 1: VirtualAdvice(rd)
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualAdvice,
                        .address = addr,
                        .operands = .{ .FormatJ = .{ .rd = rd_val, .imm = @as(u64, num_bytes) } },
                        .virtual_sequence_remaining = 2,
                        .is_first_in_sequence = true,
                        .is_compressed = false,
                    });
                    // Step 2: VirtualMULI(rd, rd, 1<<shift)
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualMULI,
                        .address = addr,
                        .operands = .{ .FormatI = .{ .rd = rd_val, .rs1 = rd_val, .imm = @as(u64, 1) << @intCast(shift) } },
                        .virtual_sequence_remaining = 1,
                        .is_first_in_sequence = false,
                        .is_compressed = false,
                    });
                    // Step 3: VirtualSRAI(rd, rd, bitmask)
                    const ones: u128 = (@as(u128, 1) << @intCast(64 - @as(u8, shift))) - 1;
                    const bitmask: u64 = @truncate(ones << @intCast(shift));
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualSRAI,
                        .address = addr,
                        .operands = .{ .FormatVirtualRightShiftI = .{ .rd = rd_val, .rs1 = rd_val, .imm = bitmask } },
                        .virtual_sequence_remaining = 0,
                        .is_first_in_sequence = false,
                        .is_compressed = is_compressed,
                    });
                },
                .AdviceLD => {
                    // AdviceLD: single VirtualAdvice(rd) cycle (already 64-bit, no sign-extend).
                    const rd_val = switch (jolt_instr.operands) {
                        .FormatI => |i| i.rd,
                        else => 0,
                    };
                    try self.bytecode.append(allocator, .{
                        .variant = .VirtualAdvice,
                        .address = addr,
                        .operands = .{ .FormatJ = .{ .rd = rd_val, .imm = 8 } },
                        .virtual_sequence_remaining = 0,
                        .is_first_in_sequence = true,
                        .is_compressed = is_compressed,
                    });
                },
                else => {
                    // Non-decomposed instructions: append as-is
                    try self.bytecode.append(allocator, jolt_instr);
                },
            }

            // Track raw 32-bit instruction word for each bytecode entry added.
            // All entries from a single ELF instruction share the same raw word.
            const entries_added = self.bytecode.items.len - bytecode_len_before;
            for (0..entries_added) |_| {
                try self.raw_words.append(allocator, instruction);
            }

            offset += instr_size;
        }

        // Build the PC map BEFORE adding termination entries.
        // This ensures termination_base_pc = last_real_instruction_pc + 1,
        // which matches the array index where termination entries will be appended.
        // If we build AFTER, the JAL (address=4) would be counted as a real
        // instruction, making termination_base_pc off by 1.
        try self.pc_map.build(self.bytecode.items);

        // Add termination sequence (LUI + ADDI + SB + JAL) = 4 entries.
        // These must be in the bytecode array BEFORE power-of-2 padding so that
        // code_size accounts for them. This matches computeBytecodeCodeSize which
        // also adds +4 for termination.
        //
        // The termination stores write a sentinel value to the termination address
        // to signal program completion, followed by a JAL-to-self that matches
        // vanilla Jolt's `j .` in _start.
        //
        // LUI x31, upper20(term_addr) - load upper bits of termination address
        // ADDI x30, x0, 1 - load value 1
        // SB x30, lower12(term_addr)(x31) - store value 1 to termination address
        // JAL x0, 0 - infinite loop (j .)
        {
            // Use address=0 for all termination entries (they are virtual, not real ELF instructions)
            // Use the memory layout's termination I/O address (NOT base_address + code_size)
            // to match the prover's bytecode table.
            const term_addr = termination_address;
            const upper20: u32 = @truncate((term_addr >> 12) & 0xFFFFF);
            const lower12: u32 = @truncate(term_addr & 0xFFF);
            const imm_upper7: u32 = (lower12 >> 5) & 0x7F;
            const imm_lower5: u32 = lower12 & 0x1F;

            // Compute instruction words (for raw_words export)
            const lui_word: u32 = (upper20 << 12) | (31 << 7) | 0x37;
            const addi_word: u32 = (1 << 20) | (0 << 15) | (0 << 12) | (30 << 7) | 0x13;
            const sb_word: u32 = (imm_upper7 << 25) | (30 << 20) | (31 << 15) | (0 << 12) | (imm_lower5 << 7) | 0x23;

            // LUI x31 (virtual, vsr=2)
            try self.bytecode.append(allocator, .{
                .variant = .LUI,
                .address = 0,
                .operands = .{ .FormatU = .{ .rd = 31, .imm = @as(u64, upper20) << 12 } },
                .virtual_sequence_remaining = 2,
                .is_first_in_sequence = false,
                .is_compressed = false,
            });
            try self.raw_words.append(allocator, lui_word);

            // ADDI x30, x0, 1 (virtual, vsr=1)
            try self.bytecode.append(allocator, .{
                .variant = .ADDI,
                .address = 0,
                .operands = .{ .FormatI = .{ .rd = 30, .rs1 = 0, .imm = 1 } },
                .virtual_sequence_remaining = 1,
                .is_first_in_sequence = false,
                .is_compressed = false,
            });
            try self.raw_words.append(allocator, addi_word);

            // SD x30, lower12(x31) (anchor, vsr=Some(0))
            // NOTE: We use SD instead of SB because SB is not in Jolt's
            // define_rv32im_trait_impls! macro (circuit_flags panics on SB).
            // SD has the Store flag and is a valid Jolt instruction.
            // The raw_word still encodes the original SB for raw word matching.
            // vsr=Some(0) matches Jolt: VirtualInstruction=true, DoNotUpdatePC=false
            try self.bytecode.append(allocator, .{
                .variant = .SD,
                .address = 0,
                .operands = .{ .FormatS = .{ .rs1 = 31, .rs2 = 30, .imm = @as(i64, @intCast(lower12)) } },
                .virtual_sequence_remaining = 0, // Some(0): last in virtual sequence
                .is_first_in_sequence = false,
                .is_compressed = false,
            });
            try self.raw_words.append(allocator, sb_word);

            // JAL x0, 0 (j . = infinite loop, vsr=None)
            // Matches vanilla Jolt's `j .` in _start after main returns.
            // address=4 (synthetic) provides UPC=4 for SB's constraint 16.
            // vsr=null means VirtualInstruction=false, DoNotUpdatePC=false.
            // Jump=1 disables constraint 16 for JAL→NoOp transition.
            // rd remapped to vr40 (upstream Jolt remaps JAL x0 to virtual register).
            const jal_word: u32 = 0x0000006F;
            try self.bytecode.append(allocator, .{
                .variant = .JAL,
                .address = 4, // Synthetic address: UPC=4 satisfies SB's NextUPC constraint
                .operands = .{ .FormatJ = .{ .rd = 40, .imm = 0 } },
                .virtual_sequence_remaining = null, // Not a virtual sequence
                .is_first_in_sequence = false,
                .is_compressed = false,
            });
            try self.raw_words.append(allocator, jal_word);
        }

        // Pad to next power of 2
        var size = self.bytecode.items.len;
        if (size < 2) size = 2;
        size = std.math.ceilPowerOfTwo(usize, size) catch size;
        self.code_size = size;

        // Pad with NoOps
        while (self.bytecode.items.len < size) {
            try self.bytecode.append(allocator, .{
                .variant = .NoOp,
                .address = 0,
                .operands = .{ .None = {} },
                .virtual_sequence_remaining = null,
                .is_first_in_sequence = false,
                .is_compressed = false,
            });
            try self.raw_words.append(allocator, 0); // NoOp padding = raw word 0
        }

        return self;
    }

    /// Serialize to arkworks format
    pub fn serialize(self: *const BytecodePreprocessing, allocator: Allocator, writer: anytype) !void {
        // code_size as usize (u64)
        try writer.writeInt(u64, @intCast(self.code_size), .little);

        // bytecode: Vec<Instruction>
        // Each instruction is serialized as: u64 length + JSON bytes
        try writer.writeInt(u64, @intCast(self.bytecode.items.len), .little);

        // Reuse a single buffer for JSON serialization to avoid per-instruction allocation
        var json_buf = std.ArrayListUnmanaged(u8){};
        defer json_buf.deinit(allocator);
        try json_buf.ensureTotalCapacity(allocator, 256);

        for (self.bytecode.items) |instr| {
            json_buf.clearRetainingCapacity();
            try instr.writeJsonTo(json_buf.writer(allocator));
            try writer.writeInt(u64, @intCast(json_buf.items.len), .little);
            try writer.writeAll(json_buf.items);
        }

        // pc_map
        try self.pc_map.serialize(writer);

        // entry_address (u64, added in upstream PR #1335)
        try writer.writeInt(u64, self.entry_address, .little);
    }
};

