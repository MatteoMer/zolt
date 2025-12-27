//! Lookup Trace Collector
//!
//! This module provides a lookup trace collector that connects the RISC-V
//! execution tracer to the Lasso lookup argument infrastructure.
//!
//! During execution, each instruction generates one or more lookup queries
//! that are recorded in the lookup trace. This trace is then used by the
//! Lasso prover to generate the lookup argument proof.
//!
//! Reference: jolt-core/src/zkvm/instruction_lookups/

const std = @import("std");
const Allocator = std.mem.Allocator;

const mod = @import("mod.zig");
const lookups = @import("lookups.zig");
const lookup_table = @import("../lookup_table/mod.zig");

const CircuitFlags = mod.CircuitFlags;
const CircuitFlagSet = mod.CircuitFlagSet;
const InstructionFlags = mod.InstructionFlags;
const InstructionFlagSet = mod.InstructionFlagSet;
const LookupTables = mod.LookupTables;
const DecodedInstruction = mod.DecodedInstruction;
const Opcode = mod.Opcode;
const OpFunct3 = mod.OpFunct3;
const OpImmFunct3 = mod.OpImmFunct3;
const BranchFunct3 = mod.BranchFunct3;

/// A single lookup entry in the trace
pub fn LookupEntry(comptime XLEN: comptime_int) type {
    return struct {
        const Self = @This();

        /// CPU cycle at which this lookup occurred
        cycle: usize,
        /// Program counter at this cycle
        pc: u64,
        /// The lookup table being queried
        table: LookupTables(XLEN),
        /// The lookup index (interleaved operands for binary ops)
        index: u128,
        /// The lookup result/output
        result: u64,
        /// Left operand value
        left_operand: u64,
        /// Right operand value
        right_operand: u64,
        /// Circuit flags for constraint generation
        circuit_flags: CircuitFlagSet,
        /// Instruction metadata flags
        instruction_flags: InstructionFlagSet,
        /// Raw RISC-V instruction
        instruction: u32,

        /// Create entry for an ADD instruction
        pub fn fromAdd(cycle: usize, pc: u64, instruction: u32, rs1: u64, rs2: u64) Self {
            const AddLookup = lookups.AddLookup(XLEN);
            const add = AddLookup.init(rs1, rs2);
            return Self{
                .cycle = cycle,
                .pc = pc,
                .table = AddLookup.lookupTable(),
                .index = add.toLookupIndex(),
                .result = add.computeResult(),
                .left_operand = rs1,
                .right_operand = rs2,
                .circuit_flags = AddLookup.circuitFlags(),
                .instruction_flags = AddLookup.instructionFlags(),
                .instruction = instruction,
            };
        }

        /// Create entry for a SUB instruction
        pub fn fromSub(cycle: usize, pc: u64, instruction: u32, rs1: u64, rs2: u64) Self {
            const SubLookup = lookups.SubLookup(XLEN);
            const sub = SubLookup.init(rs1, rs2);
            return Self{
                .cycle = cycle,
                .pc = pc,
                .table = SubLookup.lookupTable(),
                .index = sub.toLookupIndex(),
                .result = sub.computeResult(),
                .left_operand = rs1,
                .right_operand = rs2,
                .circuit_flags = SubLookup.circuitFlags(),
                .instruction_flags = SubLookup.instructionFlags(),
                .instruction = instruction,
            };
        }

        /// Create entry for an AND instruction
        pub fn fromAnd(cycle: usize, pc: u64, instruction: u32, rs1: u64, rs2: u64) Self {
            const AndLookup = lookups.AndLookup(XLEN);
            const and_op = AndLookup.init(rs1, rs2);
            return Self{
                .cycle = cycle,
                .pc = pc,
                .table = AndLookup.lookupTable(),
                .index = and_op.toLookupIndex(),
                .result = and_op.computeResult(),
                .left_operand = rs1,
                .right_operand = rs2,
                .circuit_flags = AndLookup.circuitFlags(),
                .instruction_flags = AndLookup.instructionFlags(),
                .instruction = instruction,
            };
        }

        /// Create entry for an OR instruction
        pub fn fromOr(cycle: usize, pc: u64, instruction: u32, rs1: u64, rs2: u64) Self {
            const OrLookup = lookups.OrLookup(XLEN);
            const or_op = OrLookup.init(rs1, rs2);
            return Self{
                .cycle = cycle,
                .pc = pc,
                .table = OrLookup.lookupTable(),
                .index = or_op.toLookupIndex(),
                .result = or_op.computeResult(),
                .left_operand = rs1,
                .right_operand = rs2,
                .circuit_flags = OrLookup.circuitFlags(),
                .instruction_flags = OrLookup.instructionFlags(),
                .instruction = instruction,
            };
        }

        /// Create entry for a XOR instruction
        pub fn fromXor(cycle: usize, pc: u64, instruction: u32, rs1: u64, rs2: u64) Self {
            const XorLookup = lookups.XorLookup(XLEN);
            const xor_op = XorLookup.init(rs1, rs2);
            return Self{
                .cycle = cycle,
                .pc = pc,
                .table = XorLookup.lookupTable(),
                .index = xor_op.toLookupIndex(),
                .result = xor_op.computeResult(),
                .left_operand = rs1,
                .right_operand = rs2,
                .circuit_flags = XorLookup.circuitFlags(),
                .instruction_flags = XorLookup.instructionFlags(),
                .instruction = instruction,
            };
        }

        /// Create entry for SLT (set less than signed)
        pub fn fromSlt(cycle: usize, pc: u64, instruction: u32, rs1: u64, rs2: u64) Self {
            const SltLookup = lookups.SltLookup(XLEN);
            const slt = SltLookup.init(rs1, rs2);
            return Self{
                .cycle = cycle,
                .pc = pc,
                .table = SltLookup.lookupTable(),
                .index = slt.toLookupIndex(),
                .result = slt.computeResult(),
                .left_operand = rs1,
                .right_operand = rs2,
                .circuit_flags = SltLookup.circuitFlags(),
                .instruction_flags = SltLookup.instructionFlags(),
                .instruction = instruction,
            };
        }

        /// Create entry for SLTU (set less than unsigned)
        pub fn fromSltu(cycle: usize, pc: u64, instruction: u32, rs1: u64, rs2: u64) Self {
            const SltuLookup = lookups.SltuLookup(XLEN);
            const sltu = SltuLookup.init(rs1, rs2);
            return Self{
                .cycle = cycle,
                .pc = pc,
                .table = SltuLookup.lookupTable(),
                .index = sltu.toLookupIndex(),
                .result = sltu.computeResult(),
                .left_operand = rs1,
                .right_operand = rs2,
                .circuit_flags = SltuLookup.circuitFlags(),
                .instruction_flags = SltuLookup.instructionFlags(),
                .instruction = instruction,
            };
        }

        /// Create entry for BEQ (branch if equal)
        pub fn fromBeq(cycle: usize, pc: u64, instruction: u32, rs1: u64, rs2: u64) Self {
            const BeqLookup = lookups.BeqLookup(XLEN);
            const beq = BeqLookup.init(rs1, rs2);
            return Self{
                .cycle = cycle,
                .pc = pc,
                .table = BeqLookup.lookupTable(),
                .index = beq.toLookupIndex(),
                .result = beq.computeResult(),
                .left_operand = rs1,
                .right_operand = rs2,
                .circuit_flags = BeqLookup.circuitFlags(),
                .instruction_flags = BeqLookup.instructionFlags(),
                .instruction = instruction,
            };
        }

        /// Create entry for BNE (branch if not equal)
        pub fn fromBne(cycle: usize, pc: u64, instruction: u32, rs1: u64, rs2: u64) Self {
            const BneLookup = lookups.BneLookup(XLEN);
            const bne = BneLookup.init(rs1, rs2);
            return Self{
                .cycle = cycle,
                .pc = pc,
                .table = BneLookup.lookupTable(),
                .index = bne.toLookupIndex(),
                .result = bne.computeResult(),
                .left_operand = rs1,
                .right_operand = rs2,
                .circuit_flags = BneLookup.circuitFlags(),
                .instruction_flags = BneLookup.instructionFlags(),
                .instruction = instruction,
            };
        }
    };
}

/// Lookup trace collector that records all lookup operations during execution
pub fn LookupTraceCollector(comptime XLEN: comptime_int) type {
    return struct {
        const Self = @This();
        const Entry = LookupEntry(XLEN);

        /// All lookup entries
        entries: std.ArrayListUnmanaged(Entry),
        /// Allocator for dynamic memory
        allocator: Allocator,
        /// Whether to collect lookups (can be disabled for faster emulation)
        enabled: bool,

        pub fn init(allocator: Allocator) Self {
            return Self{
                .entries = .{},
                .allocator = allocator,
                .enabled = true,
            };
        }

        pub fn deinit(self: *Self) void {
            self.entries.deinit(self.allocator);
        }

        /// Clear all entries (for reuse)
        pub fn clear(self: *Self) void {
            self.entries.clearRetainingCapacity();
        }

        /// Enable or disable lookup collection
        pub fn setEnabled(self: *Self, enabled: bool) void {
            self.enabled = enabled;
        }

        /// Record a lookup entry
        pub fn record(self: *Self, entry: Entry) !void {
            if (!self.enabled) return;
            try self.entries.append(self.allocator, entry);
        }

        /// Record lookup for an instruction based on decoded opcode and function
        /// This is the main entry point called by the emulator
        pub fn recordInstruction(
            self: *Self,
            cycle: usize,
            pc: u64,
            instruction: u32,
            decoded: DecodedInstruction,
            rs1_val: u64,
            rs2_val: u64,
        ) !void {
            if (!self.enabled) return;

            switch (decoded.opcode) {
                .OP => {
                    // Check for M extension first
                    if (decoded.funct7 == 0b0000001) {
                        // M extension: MUL, DIV, REM - TODO: implement these lookups
                        // For now, skip these as they require special handling
                        return;
                    }

                    // Standard ALU operations
                    const funct3 = @as(OpFunct3, @enumFromInt(decoded.funct3));
                    const entry: ?Entry = switch (funct3) {
                        .ADD_SUB => blk: {
                            if ((decoded.funct7 & 0x20) != 0) {
                                // SUB
                                break :blk Entry.fromSub(cycle, pc, instruction, rs1_val, rs2_val);
                            } else {
                                // ADD
                                break :blk Entry.fromAdd(cycle, pc, instruction, rs1_val, rs2_val);
                            }
                        },
                        .AND => Entry.fromAnd(cycle, pc, instruction, rs1_val, rs2_val),
                        .OR => Entry.fromOr(cycle, pc, instruction, rs1_val, rs2_val),
                        .XOR => Entry.fromXor(cycle, pc, instruction, rs1_val, rs2_val),
                        .SLT => Entry.fromSlt(cycle, pc, instruction, rs1_val, rs2_val),
                        .SLTU => Entry.fromSltu(cycle, pc, instruction, rs1_val, rs2_val),
                        .SLL, .SRL_SRA => null, // Shift ops - need different handling
                    };
                    if (entry) |e| {
                        try self.entries.append(self.allocator, e);
                    }
                },
                .OP_IMM => {
                    // Immediate ALU operations use immediate as second operand
                    const imm_val: u64 = @bitCast(@as(i64, decoded.imm));
                    const funct3 = @as(OpImmFunct3, @enumFromInt(decoded.funct3));
                    const entry: ?Entry = switch (funct3) {
                        .ADDI => Entry.fromAdd(cycle, pc, instruction, rs1_val, imm_val),
                        .ANDI => Entry.fromAnd(cycle, pc, instruction, rs1_val, imm_val),
                        .ORI => Entry.fromOr(cycle, pc, instruction, rs1_val, imm_val),
                        .XORI => Entry.fromXor(cycle, pc, instruction, rs1_val, imm_val),
                        .SLTI => Entry.fromSlt(cycle, pc, instruction, rs1_val, imm_val),
                        .SLTIU => Entry.fromSltu(cycle, pc, instruction, rs1_val, imm_val),
                        .SLLI, .SRLI_SRAI => null, // Shift ops
                    };
                    if (entry) |e| {
                        try self.entries.append(self.allocator, e);
                    }
                },
                .BRANCH => {
                    // Branch operations
                    const funct3 = @as(BranchFunct3, @enumFromInt(decoded.funct3));
                    const entry: ?Entry = switch (funct3) {
                        .BEQ => Entry.fromBeq(cycle, pc, instruction, rs1_val, rs2_val),
                        .BNE => Entry.fromBne(cycle, pc, instruction, rs1_val, rs2_val),
                        .BLT => Entry.fromSlt(cycle, pc, instruction, rs1_val, rs2_val),
                        .BGE => blk: {
                            // BGE is !(rs1 < rs2), we still record the SLT lookup
                            break :blk Entry.fromSlt(cycle, pc, instruction, rs1_val, rs2_val);
                        },
                        .BLTU => Entry.fromSltu(cycle, pc, instruction, rs1_val, rs2_val),
                        .BGEU => blk: {
                            // BGEU is !(rs1 <u rs2)
                            break :blk Entry.fromSltu(cycle, pc, instruction, rs1_val, rs2_val);
                        },
                        _ => null,
                    };
                    if (entry) |e| {
                        try self.entries.append(self.allocator, e);
                    }
                },
                else => {
                    // LUI, AUIPC, JAL, JALR, LOAD, STORE - no lookup needed
                    // These use range checks or are pure address computation
                },
            }
        }

        /// Get the number of lookup entries
        pub fn len(self: *const Self) usize {
            return self.entries.items.len;
        }

        /// Get entry at index
        pub fn get(self: *const Self, index: usize) ?Entry {
            if (index >= self.entries.items.len) return null;
            return self.entries.items[index];
        }

        /// Get all entries as a slice
        pub fn getEntries(self: *const Self) []const Entry {
            return self.entries.items;
        }

        /// Count lookups by table type
        pub fn countByTable(self: *const Self, table: LookupTables(XLEN)) usize {
            var count: usize = 0;
            for (self.entries.items) |entry| {
                if (entry.table == table) {
                    count += 1;
                }
            }
            return count;
        }

        /// Get statistics about the lookup trace
        pub const Stats = struct {
            total_lookups: usize,
            and_lookups: usize,
            or_lookups: usize,
            xor_lookups: usize,
            sub_lookups: usize,
            range_check_lookups: usize,
            signed_lt_lookups: usize,
            unsigned_lt_lookups: usize,
            equal_lookups: usize,
            not_equal_lookups: usize,
        };

        pub fn getStats(self: *const Self) Stats {
            return Stats{
                .total_lookups = self.entries.items.len,
                .and_lookups = self.countByTable(.And),
                .or_lookups = self.countByTable(.Or),
                .xor_lookups = self.countByTable(.Xor),
                .sub_lookups = self.countByTable(.Sub),
                .range_check_lookups = self.countByTable(.RangeCheck),
                .signed_lt_lookups = self.countByTable(.SignedLessThan),
                .unsigned_lt_lookups = self.countByTable(.UnsignedLessThan),
                .equal_lookups = self.countByTable(.Equal),
                .not_equal_lookups = self.countByTable(.NotEqual),
            };
        }
    };
}

// ============================================================================
// Tests
// ============================================================================

test "lookup entry creation" {
    const Entry = LookupEntry(64);

    // Test ADD entry
    const add_entry = Entry.fromAdd(0, 0x1000, 0x00208033, 10, 20);
    try std.testing.expectEqual(@as(usize, 0), add_entry.cycle);
    try std.testing.expectEqual(@as(u64, 0x1000), add_entry.pc);
    try std.testing.expectEqual(@as(u64, 30), add_entry.result);
    try std.testing.expectEqual(LookupTables(64).RangeCheck, add_entry.table);

    // Test SUB entry
    const sub_entry = Entry.fromSub(1, 0x1004, 0x40208033, 30, 10);
    try std.testing.expectEqual(@as(u64, 20), sub_entry.result);
    try std.testing.expectEqual(LookupTables(64).Sub, sub_entry.table);

    // Test AND entry
    const and_entry = Entry.fromAnd(2, 0x1008, 0x00207033, 0xFF, 0x0F);
    try std.testing.expectEqual(@as(u64, 0x0F), and_entry.result);
    try std.testing.expectEqual(LookupTables(64).And, and_entry.table);
}

test "lookup trace collector basic" {
    const allocator = std.testing.allocator;
    const Collector = LookupTraceCollector(64);
    const Entry = LookupEntry(64);

    var collector = Collector.init(allocator);
    defer collector.deinit();

    // Record some entries
    try collector.record(Entry.fromAdd(0, 0x1000, 0x00208033, 5, 3));
    try collector.record(Entry.fromSub(1, 0x1004, 0x40208033, 10, 4));
    try collector.record(Entry.fromAnd(2, 0x1008, 0x00207033, 0xFF, 0x0F));

    try std.testing.expectEqual(@as(usize, 3), collector.len());

    // Check stats
    const stats = collector.getStats();
    try std.testing.expectEqual(@as(usize, 3), stats.total_lookups);
    try std.testing.expectEqual(@as(usize, 1), stats.range_check_lookups); // ADD
    try std.testing.expectEqual(@as(usize, 1), stats.sub_lookups);
    try std.testing.expectEqual(@as(usize, 1), stats.and_lookups);
}

test "lookup trace collector record instruction" {
    const allocator = std.testing.allocator;
    const Collector = LookupTraceCollector(64);

    var collector = Collector.init(allocator);
    defer collector.deinit();

    // Test ADD instruction: add x1, x2, x3
    // opcode=0x33 (OP), rd=1, rs1=2, rs2=3, funct3=0, funct7=0
    const add_instr: u32 = 0x003100b3;
    const add_decoded = DecodedInstruction.decode(add_instr);
    try collector.recordInstruction(0, 0x1000, add_instr, add_decoded, 10, 20);

    try std.testing.expectEqual(@as(usize, 1), collector.len());
    const entry = collector.get(0).?;
    try std.testing.expectEqual(@as(u64, 30), entry.result);
    try std.testing.expectEqual(LookupTables(64).RangeCheck, entry.table);

    // Test SUB instruction: sub x1, x2, x3
    // opcode=0x33 (OP), rd=1, rs1=2, rs2=3, funct3=0, funct7=0x20
    const sub_instr: u32 = 0x403100b3;
    const sub_decoded = DecodedInstruction.decode(sub_instr);
    try collector.recordInstruction(1, 0x1004, sub_instr, sub_decoded, 30, 10);

    try std.testing.expectEqual(@as(usize, 2), collector.len());
    const sub_entry = collector.get(1).?;
    try std.testing.expectEqual(@as(u64, 20), sub_entry.result);
    try std.testing.expectEqual(LookupTables(64).Sub, sub_entry.table);
}

test "lookup trace collector immediate instructions" {
    const allocator = std.testing.allocator;
    const Collector = LookupTraceCollector(64);

    var collector = Collector.init(allocator);
    defer collector.deinit();

    // Test ADDI instruction: addi x1, x2, 42
    // opcode=0x13 (OP_IMM), rd=1, rs1=2, imm=42, funct3=0
    const addi_instr: u32 = 0x02a10093;
    const addi_decoded = DecodedInstruction.decode(addi_instr);
    try collector.recordInstruction(0, 0x1000, addi_instr, addi_decoded, 10, 0);

    try std.testing.expectEqual(@as(usize, 1), collector.len());
    const entry = collector.get(0).?;
    try std.testing.expectEqual(@as(u64, 52), entry.result); // 10 + 42
}

test "lookup trace collector branch instructions" {
    const allocator = std.testing.allocator;
    const Collector = LookupTraceCollector(64);

    var collector = Collector.init(allocator);
    defer collector.deinit();

    // Test BEQ instruction
    // opcode=0x63 (BRANCH), funct3=0 (BEQ)
    const beq_instr: u32 = 0x00208063;
    const beq_decoded = DecodedInstruction.decode(beq_instr);
    try collector.recordInstruction(0, 0x1000, beq_instr, beq_decoded, 42, 42);

    try std.testing.expectEqual(@as(usize, 1), collector.len());
    const entry = collector.get(0).?;
    try std.testing.expectEqual(@as(u64, 1), entry.result); // Equal
    try std.testing.expectEqual(LookupTables(64).Equal, entry.table);
    try std.testing.expect(entry.instruction_flags.get(.Branch));
}

test "lookup trace collector disabled" {
    const allocator = std.testing.allocator;
    const Collector = LookupTraceCollector(64);
    const Entry = LookupEntry(64);

    var collector = Collector.init(allocator);
    defer collector.deinit();

    // Disable collection
    collector.setEnabled(false);

    // Record should be no-op
    try collector.record(Entry.fromAdd(0, 0x1000, 0x00208033, 5, 3));
    try std.testing.expectEqual(@as(usize, 0), collector.len());

    // Re-enable
    collector.setEnabled(true);
    try collector.record(Entry.fromAdd(0, 0x1000, 0x00208033, 5, 3));
    try std.testing.expectEqual(@as(usize, 1), collector.len());
}
