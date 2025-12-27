//! Instruction Lookup Implementations
//!
//! This module implements the lookup operations for RISC-V instructions
//! using the Lasso lookup argument infrastructure. Each instruction type
//! defines how to compute lookup indices and verify results.
//!
//! The key insight from Jolt is that many RISC-V operations can be decomposed
//! into a small number of lookups into precomputed tables, rather than being
//! computed via arithmetic circuits.
//!
//! Reference: Jolt paper Section 6, jolt-core/src/zkvm/instruction/

const std = @import("std");
const Allocator = std.mem.Allocator;

const mod = @import("mod.zig");
const lookup_table = @import("../lookup_table/mod.zig");

const CircuitFlags = mod.CircuitFlags;
const CircuitFlagSet = mod.CircuitFlagSet;
const InstructionFlags = mod.InstructionFlags;
const InstructionFlagSet = mod.InstructionFlagSet;
const LookupTables = mod.LookupTables;

/// ADD instruction lookup
/// Computes rd = rs1 + rs2 using a single range check lookup
pub fn AddLookup(comptime XLEN: comptime_int) type {
    return struct {
        const Self = @This();

        rs1_val: u64,
        rs2_val: u64,

        /// Create a new ADD lookup from operand values
        pub fn init(rs1_val: u64, rs2_val: u64) Self {
            return Self{
                .rs1_val = rs1_val,
                .rs2_val = rs2_val,
            };
        }

        /// Get the lookup table used by ADD
        /// ADD doesn't need a complex lookup - just verifies the sum is in range
        pub fn lookupTable() LookupTables(XLEN) {
            return .RangeCheck;
        }

        /// Compute the lookup index (interleaved operands)
        pub fn toLookupIndex(self: Self) u128 {
            // For ADD, we need to verify the result is in range
            // The index is the interleaved result
            const result = self.computeResult();
            return @as(u128, result);
        }

        /// Compute the instruction result
        pub fn computeResult(self: Self) u64 {
            // Wrapping add to handle overflow correctly
            return self.rs1_val +% self.rs2_val;
        }

        /// Get circuit flags for ADD
        pub fn circuitFlags() CircuitFlagSet {
            var flags = CircuitFlagSet.init();
            flags.set(.AddOperands);
            flags.set(.WriteLookupOutputToRD);
            return flags;
        }

        /// Get instruction flags for ADD
        pub fn instructionFlags() InstructionFlagSet {
            var flags = InstructionFlagSet.init();
            flags.set(.LeftOperandIsRs1Value);
            flags.set(.RightOperandIsRs2Value);
            return flags;
        }
    };
}

/// SUB instruction lookup
/// Computes rd = rs1 - rs2 using the Sub lookup table
pub fn SubLookup(comptime XLEN: comptime_int) type {
    return struct {
        const Self = @This();

        rs1_val: u64,
        rs2_val: u64,

        pub fn init(rs1_val: u64, rs2_val: u64) Self {
            return Self{
                .rs1_val = rs1_val,
                .rs2_val = rs2_val,
            };
        }

        pub fn lookupTable() LookupTables(XLEN) {
            return .Sub;
        }

        pub fn toLookupIndex(self: Self) u128 {
            // Interleave x and y for lookup
            return lookup_table.interleaveBits(self.rs1_val, self.rs2_val);
        }

        pub fn computeResult(self: Self) u64 {
            return self.rs1_val -% self.rs2_val;
        }

        pub fn circuitFlags() CircuitFlagSet {
            var flags = CircuitFlagSet.init();
            flags.set(.SubtractOperands);
            flags.set(.WriteLookupOutputToRD);
            return flags;
        }

        pub fn instructionFlags() InstructionFlagSet {
            var flags = InstructionFlagSet.init();
            flags.set(.LeftOperandIsRs1Value);
            flags.set(.RightOperandIsRs2Value);
            return flags;
        }
    };
}

/// AND instruction lookup
/// Computes rd = rs1 & rs2 using the And lookup table
pub fn AndLookup(comptime XLEN: comptime_int) type {
    return struct {
        const Self = @This();

        rs1_val: u64,
        rs2_val: u64,

        pub fn init(rs1_val: u64, rs2_val: u64) Self {
            return Self{
                .rs1_val = rs1_val,
                .rs2_val = rs2_val,
            };
        }

        pub fn lookupTable() LookupTables(XLEN) {
            return .And;
        }

        pub fn toLookupIndex(self: Self) u128 {
            return lookup_table.interleaveBits(self.rs1_val, self.rs2_val);
        }

        pub fn computeResult(self: Self) u64 {
            return self.rs1_val & self.rs2_val;
        }

        pub fn circuitFlags() CircuitFlagSet {
            var flags = CircuitFlagSet.init();
            flags.set(.WriteLookupOutputToRD);
            return flags;
        }

        pub fn instructionFlags() InstructionFlagSet {
            var flags = InstructionFlagSet.init();
            flags.set(.LeftOperandIsRs1Value);
            flags.set(.RightOperandIsRs2Value);
            return flags;
        }
    };
}

/// OR instruction lookup
/// Computes rd = rs1 | rs2 using the Or lookup table
pub fn OrLookup(comptime XLEN: comptime_int) type {
    return struct {
        const Self = @This();

        rs1_val: u64,
        rs2_val: u64,

        pub fn init(rs1_val: u64, rs2_val: u64) Self {
            return Self{
                .rs1_val = rs1_val,
                .rs2_val = rs2_val,
            };
        }

        pub fn lookupTable() LookupTables(XLEN) {
            return .Or;
        }

        pub fn toLookupIndex(self: Self) u128 {
            return lookup_table.interleaveBits(self.rs1_val, self.rs2_val);
        }

        pub fn computeResult(self: Self) u64 {
            return self.rs1_val | self.rs2_val;
        }

        pub fn circuitFlags() CircuitFlagSet {
            var flags = CircuitFlagSet.init();
            flags.set(.WriteLookupOutputToRD);
            return flags;
        }

        pub fn instructionFlags() InstructionFlagSet {
            var flags = InstructionFlagSet.init();
            flags.set(.LeftOperandIsRs1Value);
            flags.set(.RightOperandIsRs2Value);
            return flags;
        }
    };
}

/// XOR instruction lookup
/// Computes rd = rs1 ^ rs2 using the Xor lookup table
pub fn XorLookup(comptime XLEN: comptime_int) type {
    return struct {
        const Self = @This();

        rs1_val: u64,
        rs2_val: u64,

        pub fn init(rs1_val: u64, rs2_val: u64) Self {
            return Self{
                .rs1_val = rs1_val,
                .rs2_val = rs2_val,
            };
        }

        pub fn lookupTable() LookupTables(XLEN) {
            return .Xor;
        }

        pub fn toLookupIndex(self: Self) u128 {
            return lookup_table.interleaveBits(self.rs1_val, self.rs2_val);
        }

        pub fn computeResult(self: Self) u64 {
            return self.rs1_val ^ self.rs2_val;
        }

        pub fn circuitFlags() CircuitFlagSet {
            var flags = CircuitFlagSet.init();
            flags.set(.WriteLookupOutputToRD);
            return flags;
        }

        pub fn instructionFlags() InstructionFlagSet {
            var flags = InstructionFlagSet.init();
            flags.set(.LeftOperandIsRs1Value);
            flags.set(.RightOperandIsRs2Value);
            return flags;
        }
    };
}

/// SLT (Set Less Than) instruction lookup
/// Computes rd = (rs1 < rs2) ? 1 : 0 using SignedLessThan table
pub fn SltLookup(comptime XLEN: comptime_int) type {
    return struct {
        const Self = @This();

        rs1_val: u64,
        rs2_val: u64,

        pub fn init(rs1_val: u64, rs2_val: u64) Self {
            return Self{
                .rs1_val = rs1_val,
                .rs2_val = rs2_val,
            };
        }

        pub fn lookupTable() LookupTables(XLEN) {
            return .SignedLessThan;
        }

        pub fn toLookupIndex(self: Self) u128 {
            return lookup_table.interleaveBits(self.rs1_val, self.rs2_val);
        }

        pub fn computeResult(self: Self) u64 {
            // Signed comparison
            const signed_rs1 = @as(i64, @bitCast(self.rs1_val));
            const signed_rs2 = @as(i64, @bitCast(self.rs2_val));
            return if (signed_rs1 < signed_rs2) 1 else 0;
        }

        pub fn circuitFlags() CircuitFlagSet {
            var flags = CircuitFlagSet.init();
            flags.set(.WriteLookupOutputToRD);
            return flags;
        }

        pub fn instructionFlags() InstructionFlagSet {
            var flags = InstructionFlagSet.init();
            flags.set(.LeftOperandIsRs1Value);
            flags.set(.RightOperandIsRs2Value);
            return flags;
        }
    };
}

/// SLTU (Set Less Than Unsigned) instruction lookup
/// Computes rd = (rs1 < rs2) ? 1 : 0 using UnsignedLessThan table
pub fn SltuLookup(comptime XLEN: comptime_int) type {
    return struct {
        const Self = @This();

        rs1_val: u64,
        rs2_val: u64,

        pub fn init(rs1_val: u64, rs2_val: u64) Self {
            return Self{
                .rs1_val = rs1_val,
                .rs2_val = rs2_val,
            };
        }

        pub fn lookupTable() LookupTables(XLEN) {
            return .UnsignedLessThan;
        }

        pub fn toLookupIndex(self: Self) u128 {
            return lookup_table.interleaveBits(self.rs1_val, self.rs2_val);
        }

        pub fn computeResult(self: Self) u64 {
            return if (self.rs1_val < self.rs2_val) 1 else 0;
        }

        pub fn circuitFlags() CircuitFlagSet {
            var flags = CircuitFlagSet.init();
            flags.set(.WriteLookupOutputToRD);
            return flags;
        }

        pub fn instructionFlags() InstructionFlagSet {
            var flags = InstructionFlagSet.init();
            flags.set(.LeftOperandIsRs1Value);
            flags.set(.RightOperandIsRs2Value);
            return flags;
        }
    };
}

/// BEQ (Branch if Equal) instruction lookup
/// Uses Equal table to check if rs1 == rs2
pub fn BeqLookup(comptime XLEN: comptime_int) type {
    return struct {
        const Self = @This();

        rs1_val: u64,
        rs2_val: u64,

        pub fn init(rs1_val: u64, rs2_val: u64) Self {
            return Self{
                .rs1_val = rs1_val,
                .rs2_val = rs2_val,
            };
        }

        pub fn lookupTable() LookupTables(XLEN) {
            return .Equal;
        }

        pub fn toLookupIndex(self: Self) u128 {
            return lookup_table.interleaveBits(self.rs1_val, self.rs2_val);
        }

        pub fn computeResult(self: Self) u64 {
            return if (self.rs1_val == self.rs2_val) 1 else 0;
        }

        pub fn circuitFlags() CircuitFlagSet {
            // Branch doesn't write to RD, so no flags needed
            return CircuitFlagSet.init();
        }

        pub fn instructionFlags() InstructionFlagSet {
            var flags = InstructionFlagSet.init();
            flags.set(.LeftOperandIsRs1Value);
            flags.set(.RightOperandIsRs2Value);
            flags.set(.Branch);
            return flags;
        }
    };
}

/// BNE (Branch if Not Equal) instruction lookup
/// Uses NotEqual table
pub fn BneLookup(comptime XLEN: comptime_int) type {
    return struct {
        const Self = @This();

        rs1_val: u64,
        rs2_val: u64,

        pub fn init(rs1_val: u64, rs2_val: u64) Self {
            return Self{
                .rs1_val = rs1_val,
                .rs2_val = rs2_val,
            };
        }

        pub fn lookupTable() LookupTables(XLEN) {
            return .NotEqual;
        }

        pub fn toLookupIndex(self: Self) u128 {
            return lookup_table.interleaveBits(self.rs1_val, self.rs2_val);
        }

        pub fn computeResult(self: Self) u64 {
            return if (self.rs1_val != self.rs2_val) 1 else 0;
        }

        pub fn circuitFlags() CircuitFlagSet {
            // Branch doesn't write to RD, so no flags needed
            return CircuitFlagSet.init();
        }

        pub fn instructionFlags() InstructionFlagSet {
            var flags = InstructionFlagSet.init();
            flags.set(.LeftOperandIsRs1Value);
            flags.set(.RightOperandIsRs2Value);
            flags.set(.Branch);
            return flags;
        }
    };
}

/// Instruction lookup trace entry
/// Records a single lookup operation for proof generation
pub fn LookupTraceEntry(comptime XLEN: comptime_int) type {
    return struct {
        const Self = @This();

        /// Which cycle this lookup occurred at
        cycle: usize,
        /// The lookup table used
        table: LookupTables(XLEN),
        /// The lookup index (interleaved operands)
        index: u128,
        /// The lookup result
        result: u64,
        /// Circuit flags for this operation
        circuit_flags: CircuitFlagSet,
        /// Instruction flags for this operation
        instruction_flags: InstructionFlagSet,

        /// Create a trace entry from an instruction lookup
        pub fn fromAdd(cycle: usize, add: AddLookup(XLEN)) Self {
            return Self{
                .cycle = cycle,
                .table = add.lookupTable(),
                .index = add.toLookupIndex(),
                .result = add.computeResult(),
                .circuit_flags = AddLookup(XLEN).circuitFlags(),
                .instruction_flags = AddLookup(XLEN).instructionFlags(),
            };
        }

        pub fn fromSub(cycle: usize, sub: SubLookup(XLEN)) Self {
            return Self{
                .cycle = cycle,
                .table = sub.lookupTable(),
                .index = sub.toLookupIndex(),
                .result = sub.computeResult(),
                .circuit_flags = SubLookup(XLEN).circuitFlags(),
                .instruction_flags = SubLookup(XLEN).instructionFlags(),
            };
        }

        pub fn fromAnd(cycle: usize, and_op: AndLookup(XLEN)) Self {
            return Self{
                .cycle = cycle,
                .table = and_op.lookupTable(),
                .index = and_op.toLookupIndex(),
                .result = and_op.computeResult(),
                .circuit_flags = AndLookup(XLEN).circuitFlags(),
                .instruction_flags = AndLookup(XLEN).instructionFlags(),
            };
        }
    };
}

// ============================================================================
// Tests
// ============================================================================

test "add lookup" {
    const add = AddLookup(64).init(10, 20);
    try std.testing.expectEqual(@as(u64, 30), add.computeResult());

    const flags = AddLookup(64).circuitFlags();
    try std.testing.expect(flags.get(.AddOperands));
    try std.testing.expect(flags.get(.WriteLookupOutputToRD));
}

test "sub lookup" {
    const sub = SubLookup(64).init(30, 10);
    try std.testing.expectEqual(@as(u64, 20), sub.computeResult());

    // Test wraparound
    const sub_wrap = SubLookup(64).init(10, 20);
    try std.testing.expectEqual(@as(u64, @bitCast(@as(i64, -10))), sub_wrap.computeResult());
}

test "and lookup" {
    const and_op = AndLookup(64).init(0b1010, 0b1100);
    try std.testing.expectEqual(@as(u64, 0b1000), and_op.computeResult());

    // Verify index is interleaved
    const index = and_op.toLookupIndex();
    const result = lookup_table.uninterleaveBits(index);
    try std.testing.expectEqual(@as(u64, 0b1010), result.x);
    try std.testing.expectEqual(@as(u64, 0b1100), result.y);
}

test "or lookup" {
    const or_op = OrLookup(64).init(0b1010, 0b0101);
    try std.testing.expectEqual(@as(u64, 0b1111), or_op.computeResult());
}

test "xor lookup" {
    const xor_op = XorLookup(64).init(0b1010, 0b1100);
    try std.testing.expectEqual(@as(u64, 0b0110), xor_op.computeResult());
}

test "slt lookup" {
    // Signed comparison
    const slt_pos = SltLookup(64).init(5, 10);
    try std.testing.expectEqual(@as(u64, 1), slt_pos.computeResult());

    const slt_neg = SltLookup(64).init(@bitCast(@as(i64, -5)), 10);
    try std.testing.expectEqual(@as(u64, 1), slt_neg.computeResult());

    const slt_false = SltLookup(64).init(10, 5);
    try std.testing.expectEqual(@as(u64, 0), slt_false.computeResult());
}

test "sltu lookup" {
    const sltu = SltuLookup(64).init(5, 10);
    try std.testing.expectEqual(@as(u64, 1), sltu.computeResult());

    // -1 as unsigned is max value, so it's greater than 10
    const sltu_wrap = SltuLookup(64).init(@bitCast(@as(i64, -1)), 10);
    try std.testing.expectEqual(@as(u64, 0), sltu_wrap.computeResult());
}

test "beq lookup" {
    const beq_true = BeqLookup(64).init(42, 42);
    try std.testing.expectEqual(@as(u64, 1), beq_true.computeResult());

    const beq_false = BeqLookup(64).init(42, 43);
    try std.testing.expectEqual(@as(u64, 0), beq_false.computeResult());

    const flags = BeqLookup(64).instructionFlags();
    try std.testing.expect(flags.get(.Branch));
}

test "bne lookup" {
    const bne_true = BneLookup(64).init(42, 43);
    try std.testing.expectEqual(@as(u64, 1), bne_true.computeResult());

    const bne_false = BneLookup(64).init(42, 42);
    try std.testing.expectEqual(@as(u64, 0), bne_false.computeResult());
}

test "lookup trace entry" {
    const add = AddLookup(64).init(100, 200);
    const entry = LookupTraceEntry(64).fromAdd(5, add);

    try std.testing.expectEqual(@as(usize, 5), entry.cycle);
    try std.testing.expectEqual(LookupTables(64).RangeCheck, entry.table);
    try std.testing.expectEqual(@as(u64, 300), entry.result);
}
