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

/// SLL (Shift Left Logical) instruction lookup
/// Computes rd = rs1 << (rs2 & (XLEN-1))
pub fn SllLookup(comptime XLEN: comptime_int) type {
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
            return .LeftShift;
        }

        pub fn toLookupIndex(self: Self) u128 {
            return lookup_table.interleaveBits(self.rs1_val, self.rs2_val);
        }

        pub fn computeResult(self: Self) u64 {
            const shift_mask: u6 = XLEN - 1;
            const shift: u6 = @truncate(self.rs2_val & @as(u64, shift_mask));
            const mask: u64 = if (XLEN == 64) ~@as(u64, 0) else (@as(u64, 1) << XLEN) - 1;
            return (self.rs1_val << shift) & mask;
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

/// SRL (Shift Right Logical) instruction lookup
/// Computes rd = rs1 >> (rs2 & (XLEN-1)) (logical shift)
pub fn SrlLookup(comptime XLEN: comptime_int) type {
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
            return .RightShift;
        }

        pub fn toLookupIndex(self: Self) u128 {
            return lookup_table.interleaveBits(self.rs1_val, self.rs2_val);
        }

        pub fn computeResult(self: Self) u64 {
            const shift_mask: u6 = XLEN - 1;
            const shift: u6 = @truncate(self.rs2_val & @as(u64, shift_mask));
            const mask: u64 = if (XLEN == 64) ~@as(u64, 0) else (@as(u64, 1) << XLEN) - 1;
            return (self.rs1_val & mask) >> shift;
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

/// SRA (Shift Right Arithmetic) instruction lookup
/// Computes rd = rs1 >> (rs2 & (XLEN-1)) (arithmetic shift, sign-extending)
pub fn SraLookup(comptime XLEN: comptime_int) type {
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
            return .RightShiftArithmetic;
        }

        pub fn toLookupIndex(self: Self) u128 {
            return lookup_table.interleaveBits(self.rs1_val, self.rs2_val);
        }

        pub fn computeResult(self: Self) u64 {
            const shift_mask: u6 = XLEN - 1;
            const shift: u6 = @truncate(self.rs2_val & @as(u64, shift_mask));

            if (XLEN == 64) {
                const signed_val: i64 = @bitCast(self.rs1_val);
                const shifted: i64 = signed_val >> shift;
                return @bitCast(shifted);
            } else {
                const mask: u64 = (@as(u64, 1) << XLEN) - 1;
                const masked = self.rs1_val & mask;
                // Sign extend to full 64 bits, then shift
                const shift_for_sign = 64 - XLEN;
                const signed_val: i64 = @as(i64, @bitCast(masked << @truncate(shift_for_sign))) >> @truncate(shift_for_sign);
                const shifted: i64 = signed_val >> shift;
                return @as(u64, @bitCast(shifted)) & mask;
            }
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

/// SLLI (Shift Left Logical Immediate) instruction lookup
/// Computes rd = rs1 << imm
pub fn SlliLookup(comptime XLEN: comptime_int) type {
    return struct {
        const Self = @This();

        rs1_val: u64,
        imm: u64,

        pub fn init(rs1_val: u64, imm: u64) Self {
            return Self{
                .rs1_val = rs1_val,
                .imm = imm,
            };
        }

        pub fn lookupTable() LookupTables(XLEN) {
            return .LeftShift;
        }

        pub fn toLookupIndex(self: Self) u128 {
            return lookup_table.interleaveBits(self.rs1_val, self.imm);
        }

        pub fn computeResult(self: Self) u64 {
            const shift_mask: u6 = XLEN - 1;
            const shift: u6 = @truncate(self.imm & @as(u64, shift_mask));
            const mask: u64 = if (XLEN == 64) ~@as(u64, 0) else (@as(u64, 1) << XLEN) - 1;
            return (self.rs1_val << shift) & mask;
        }

        pub fn circuitFlags() CircuitFlagSet {
            var flags = CircuitFlagSet.init();
            flags.set(.WriteLookupOutputToRD);
            return flags;
        }

        pub fn instructionFlags() InstructionFlagSet {
            var flags = InstructionFlagSet.init();
            flags.set(.LeftOperandIsRs1Value);
            flags.set(.RightOperandIsImm);
            return flags;
        }
    };
}

/// SRLI (Shift Right Logical Immediate) instruction lookup
/// Computes rd = rs1 >> imm (logical shift)
pub fn SrliLookup(comptime XLEN: comptime_int) type {
    return struct {
        const Self = @This();

        rs1_val: u64,
        imm: u64,

        pub fn init(rs1_val: u64, imm: u64) Self {
            return Self{
                .rs1_val = rs1_val,
                .imm = imm,
            };
        }

        pub fn lookupTable() LookupTables(XLEN) {
            return .RightShift;
        }

        pub fn toLookupIndex(self: Self) u128 {
            return lookup_table.interleaveBits(self.rs1_val, self.imm);
        }

        pub fn computeResult(self: Self) u64 {
            const shift_mask: u6 = XLEN - 1;
            const shift: u6 = @truncate(self.imm & @as(u64, shift_mask));
            const mask: u64 = if (XLEN == 64) ~@as(u64, 0) else (@as(u64, 1) << XLEN) - 1;
            return (self.rs1_val & mask) >> shift;
        }

        pub fn circuitFlags() CircuitFlagSet {
            var flags = CircuitFlagSet.init();
            flags.set(.WriteLookupOutputToRD);
            return flags;
        }

        pub fn instructionFlags() InstructionFlagSet {
            var flags = InstructionFlagSet.init();
            flags.set(.LeftOperandIsRs1Value);
            flags.set(.RightOperandIsImm);
            return flags;
        }
    };
}

/// SRAI (Shift Right Arithmetic Immediate) instruction lookup
/// Computes rd = rs1 >> imm (arithmetic shift, sign-extending)
pub fn SraiLookup(comptime XLEN: comptime_int) type {
    return struct {
        const Self = @This();

        rs1_val: u64,
        imm: u64,

        pub fn init(rs1_val: u64, imm: u64) Self {
            return Self{
                .rs1_val = rs1_val,
                .imm = imm,
            };
        }

        pub fn lookupTable() LookupTables(XLEN) {
            return .RightShiftArithmetic;
        }

        pub fn toLookupIndex(self: Self) u128 {
            return lookup_table.interleaveBits(self.rs1_val, self.imm);
        }

        pub fn computeResult(self: Self) u64 {
            const shift_mask: u6 = XLEN - 1;
            const shift: u6 = @truncate(self.imm & @as(u64, shift_mask));

            if (XLEN == 64) {
                const signed_val: i64 = @bitCast(self.rs1_val);
                const shifted: i64 = signed_val >> shift;
                return @bitCast(shifted);
            } else {
                const mask: u64 = (@as(u64, 1) << XLEN) - 1;
                const masked = self.rs1_val & mask;
                const shift_for_sign = 64 - XLEN;
                const signed_val: i64 = @as(i64, @bitCast(masked << @truncate(shift_for_sign))) >> @truncate(shift_for_sign);
                const shifted: i64 = signed_val >> shift;
                return @as(u64, @bitCast(shifted)) & mask;
            }
        }

        pub fn circuitFlags() CircuitFlagSet {
            var flags = CircuitFlagSet.init();
            flags.set(.WriteLookupOutputToRD);
            return flags;
        }

        pub fn instructionFlags() InstructionFlagSet {
            var flags = InstructionFlagSet.init();
            flags.set(.LeftOperandIsRs1Value);
            flags.set(.RightOperandIsImm);
            return flags;
        }
    };
}

/// MUL (Multiply) instruction lookup
/// Computes rd = (rs1 * rs2)[XLEN-1:0] (low bits of product)
pub fn MulLookup(comptime XLEN: comptime_int) type {
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

        /// MUL uses RangeCheck to verify the result fits
        pub fn lookupTable() LookupTables(XLEN) {
            return .RangeCheck;
        }

        /// The lookup index is the result (to range check it)
        pub fn toLookupIndex(self: Self) u128 {
            return @as(u128, self.computeResult());
        }

        /// Compute the product (low XLEN bits)
        pub fn computeResult(self: Self) u64 {
            // Wrapping multiply to get low bits
            return self.rs1_val *% self.rs2_val;
        }

        pub fn circuitFlags() CircuitFlagSet {
            var flags = CircuitFlagSet.init();
            flags.set(.MultiplyOperands); // This tells the circuit to multiply
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

/// MULH (Multiply High Signed) instruction lookup
/// Computes rd = (rs1 * rs2)[2*XLEN-1:XLEN] (high bits of signed product)
pub fn MulhLookup(comptime XLEN: comptime_int) type {
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
            return .RangeCheck;
        }

        pub fn toLookupIndex(self: Self) u128 {
            return @as(u128, self.computeResult());
        }

        /// Compute the high bits of signed multiply
        pub fn computeResult(self: Self) u64 {
            if (XLEN == 64) {
                const a: i64 = @bitCast(self.rs1_val);
                const b: i64 = @bitCast(self.rs2_val);
                // Use 128-bit multiply
                const product: i128 = @as(i128, a) * @as(i128, b);
                const high_bits: i64 = @truncate(product >> 64);
                return @bitCast(high_bits);
            } else {
                // For 32-bit or smaller XLEN
                const mask: u64 = (@as(u64, 1) << XLEN) - 1;
                const a_masked = self.rs1_val & mask;
                const b_masked = self.rs2_val & mask;

                // Sign-extend to full 64-bit
                const shift = 64 - XLEN;
                const a_signed: i64 = @as(i64, @bitCast(a_masked << @truncate(shift))) >> @truncate(shift);
                const b_signed: i64 = @as(i64, @bitCast(b_masked << @truncate(shift))) >> @truncate(shift);

                // Multiply and get high bits
                const product: i128 = @as(i128, a_signed) * @as(i128, b_signed);
                const high_bits: i64 = @truncate(product >> XLEN);
                return @as(u64, @bitCast(high_bits)) & mask;
            }
        }

        pub fn circuitFlags() CircuitFlagSet {
            var flags = CircuitFlagSet.init();
            flags.set(.MultiplyOperands);
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

/// MULHU (Multiply High Unsigned) instruction lookup
/// Computes rd = (rs1 * rs2)[2*XLEN-1:XLEN] (high bits of unsigned product)
pub fn MulhuLookup(comptime XLEN: comptime_int) type {
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
            return .RangeCheck;
        }

        pub fn toLookupIndex(self: Self) u128 {
            return @as(u128, self.computeResult());
        }

        /// Compute the high bits of unsigned multiply
        pub fn computeResult(self: Self) u64 {
            if (XLEN == 64) {
                // Use 128-bit unsigned multiply
                const product: u128 = @as(u128, self.rs1_val) * @as(u128, self.rs2_val);
                return @truncate(product >> 64);
            } else {
                const mask: u64 = (@as(u64, 1) << XLEN) - 1;
                const a_masked = self.rs1_val & mask;
                const b_masked = self.rs2_val & mask;

                const product: u128 = @as(u128, a_masked) * @as(u128, b_masked);
                return @as(u64, @truncate(product >> XLEN)) & mask;
            }
        }

        pub fn circuitFlags() CircuitFlagSet {
            var flags = CircuitFlagSet.init();
            flags.set(.MultiplyOperands);
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

/// MULHSU (Multiply High Signed-Unsigned) instruction lookup
/// Computes rd = (signed(rs1) * unsigned(rs2))[2*XLEN-1:XLEN]
pub fn MulhsuLookup(comptime XLEN: comptime_int) type {
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
            return .RangeCheck;
        }

        pub fn toLookupIndex(self: Self) u128 {
            return @as(u128, self.computeResult());
        }

        /// Compute the high bits of signed*unsigned multiply
        pub fn computeResult(self: Self) u64 {
            if (XLEN == 64) {
                const a: i64 = @bitCast(self.rs1_val);
                const b: u64 = self.rs2_val;
                // signed * unsigned -> signed 128-bit result
                const product: i128 = @as(i128, a) * @as(i128, @as(i64, @bitCast(b)));
                const high_bits: i64 = @truncate(product >> 64);
                return @bitCast(high_bits);
            } else {
                const mask: u64 = (@as(u64, 1) << XLEN) - 1;
                const a_masked = self.rs1_val & mask;
                const b_masked = self.rs2_val & mask;

                // Sign-extend rs1, keep rs2 unsigned
                const shift = 64 - XLEN;
                const a_signed: i64 = @as(i64, @bitCast(a_masked << @truncate(shift))) >> @truncate(shift);

                // signed * unsigned
                const product: i128 = @as(i128, a_signed) * @as(i128, @as(i64, @bitCast(b_masked)));
                const high_bits: i64 = @truncate(product >> XLEN);
                return @as(u64, @bitCast(high_bits)) & mask;
            }
        }

        pub fn circuitFlags() CircuitFlagSet {
            var flags = CircuitFlagSet.init();
            flags.set(.MultiplyOperands);
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
            const AddLookupType = AddLookup(XLEN);
            return Self{
                .cycle = cycle,
                .table = AddLookupType.lookupTable(),
                .index = add.toLookupIndex(),
                .result = add.computeResult(),
                .circuit_flags = AddLookupType.circuitFlags(),
                .instruction_flags = AddLookupType.instructionFlags(),
            };
        }

        pub fn fromSub(cycle: usize, sub: SubLookup(XLEN)) Self {
            const SubLookupType = SubLookup(XLEN);
            return Self{
                .cycle = cycle,
                .table = SubLookupType.lookupTable(),
                .index = sub.toLookupIndex(),
                .result = sub.computeResult(),
                .circuit_flags = SubLookupType.circuitFlags(),
                .instruction_flags = SubLookupType.instructionFlags(),
            };
        }

        pub fn fromAnd(cycle: usize, and_op: AndLookup(XLEN)) Self {
            const AndLookupType = AndLookup(XLEN);
            return Self{
                .cycle = cycle,
                .table = AndLookupType.lookupTable(),
                .index = and_op.toLookupIndex(),
                .result = and_op.computeResult(),
                .circuit_flags = AndLookupType.circuitFlags(),
                .instruction_flags = AndLookupType.instructionFlags(),
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

test "sll lookup" {
    // 1 << 4 = 16
    const sll1 = SllLookup(64).init(1, 4);
    try std.testing.expectEqual(@as(u64, 16), sll1.computeResult());

    // 0xFF << 8 = 0xFF00
    const sll2 = SllLookup(64).init(0xFF, 8);
    try std.testing.expectEqual(@as(u64, 0xFF00), sll2.computeResult());

    const flags = SllLookup(64).circuitFlags();
    try std.testing.expect(flags.get(.WriteLookupOutputToRD));
}

test "srl lookup" {
    // 256 >> 4 = 16
    const srl1 = SrlLookup(64).init(256, 4);
    try std.testing.expectEqual(@as(u64, 16), srl1.computeResult());

    // 0xFF00 >> 8 = 0xFF
    const srl2 = SrlLookup(64).init(0xFF00, 8);
    try std.testing.expectEqual(@as(u64, 0xFF), srl2.computeResult());
}

test "sra lookup" {
    // Positive number: 256 >> 4 = 16 (same as SRL)
    const sra_pos = SraLookup(64).init(256, 4);
    try std.testing.expectEqual(@as(u64, 16), sra_pos.computeResult());

    // Negative number: -256 >> 4 = -16
    const neg256: u64 = @bitCast(@as(i64, -256));
    const sra_neg = SraLookup(64).init(neg256, 4);
    const expected: u64 = @bitCast(@as(i64, -16));
    try std.testing.expectEqual(expected, sra_neg.computeResult());

    // -1 >> any = -1
    const neg1: u64 = @bitCast(@as(i64, -1));
    const sra_all_ones = SraLookup(64).init(neg1, 10);
    try std.testing.expectEqual(neg1, sra_all_ones.computeResult());
}

test "slli lookup" {
    const slli = SlliLookup(64).init(5, 3);
    try std.testing.expectEqual(@as(u64, 40), slli.computeResult());

    const flags = SlliLookup(64).instructionFlags();
    try std.testing.expect(flags.get(.LeftOperandIsRs1Value));
    try std.testing.expect(flags.get(.RightOperandIsImm));
}

test "srli lookup" {
    const srli = SrliLookup(64).init(40, 3);
    try std.testing.expectEqual(@as(u64, 5), srli.computeResult());
}

test "srai lookup" {
    // Positive
    const srai_pos = SraiLookup(64).init(40, 3);
    try std.testing.expectEqual(@as(u64, 5), srai_pos.computeResult());

    // Negative: -40 >> 3 = -5
    const neg40: u64 = @bitCast(@as(i64, -40));
    const srai_neg = SraiLookup(64).init(neg40, 3);
    const expected: u64 = @bitCast(@as(i64, -5));
    try std.testing.expectEqual(expected, srai_neg.computeResult());
}

test "mul lookup" {
    // 6 * 7 = 42
    const mul1 = MulLookup(64).init(6, 7);
    try std.testing.expectEqual(@as(u64, 42), mul1.computeResult());

    // Negative: -3 * 4 = -12
    const neg3: u64 = @bitCast(@as(i64, -3));
    const mul2 = MulLookup(64).init(neg3, 4);
    const expected: u64 = @bitCast(@as(i64, -12));
    try std.testing.expectEqual(expected, mul2.computeResult());

    // Overflow wraps
    const mul_wrap = MulLookup(64).init(0x8000000000000000, 2);
    try std.testing.expectEqual(@as(u64, 0), mul_wrap.computeResult()); // Overflow to 0

    // Check flags
    const flags = MulLookup(64).circuitFlags();
    try std.testing.expect(flags.get(.MultiplyOperands));
    try std.testing.expect(flags.get(.WriteLookupOutputToRD));
}

test "mulh lookup" {
    // Simple: small numbers, high bits = 0
    const mulh1 = MulhLookup(64).init(100, 200);
    try std.testing.expectEqual(@as(u64, 0), mulh1.computeResult());

    // Large positive * positive
    const mulh2 = MulhLookup(64).init(0x1000000000000000, 0x10);
    try std.testing.expectEqual(@as(u64, 1), mulh2.computeResult());

    // Negative * positive
    const neg_max: u64 = @bitCast(@as(i64, -1));
    const mulh3 = MulhLookup(64).init(neg_max, 2);
    try std.testing.expectEqual(neg_max, mulh3.computeResult()); // -1 * 2 high bits = -1
}

test "mulhu lookup" {
    // Small numbers: high bits = 0
    const mulhu1 = MulhuLookup(64).init(100, 200);
    try std.testing.expectEqual(@as(u64, 0), mulhu1.computeResult());

    // Large numbers
    const mulhu2 = MulhuLookup(64).init(0x1000000000000000, 0x10);
    try std.testing.expectEqual(@as(u64, 1), mulhu2.computeResult());

    // Max * 2 = overflow
    const mulhu3 = MulhuLookup(64).init(0xFFFFFFFFFFFFFFFF, 2);
    try std.testing.expectEqual(@as(u64, 1), mulhu3.computeResult());
}

test "mulhsu lookup" {
    // Positive * positive: same as mulhu
    const mulhsu1 = MulhsuLookup(64).init(100, 200);
    try std.testing.expectEqual(@as(u64, 0), mulhsu1.computeResult());

    // Negative * positive
    const neg1: u64 = @bitCast(@as(i64, -1));
    const mulhsu2 = MulhsuLookup(64).init(neg1, 2);
    try std.testing.expectEqual(neg1, mulhsu2.computeResult()); // -1 * 2 high = -1
}
