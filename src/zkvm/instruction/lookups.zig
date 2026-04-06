//! Instruction Lookup Implementations
//!
//! This module implements the lookup operations for RISC-V instructions
//! using the Shout lookup argument infrastructure. Each instruction type
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

/// Generic two-operand instruction lookup.
/// Covers ALU, branch, shift, and multiply/divide instructions that take (rs1, rs2).
pub fn BinaryLookup(comptime XLEN: comptime_int, comptime cfg: struct {
    table: LookupTables(XLEN),
    /// If true, toLookupIndex returns interleaveBits(rs1, rs2).
    /// If false, returns @as(u128, computeResult(rs1, rs2)).
    interleave: bool = true,
    circuit_flags: CircuitFlagSet,
    instruction_flags: InstructionFlagSet,
    computeResult: *const fn (u64, u64) u64,
    /// Custom index function, overrides interleave behavior if set
    customIndex: ?*const fn (u64, u64, *const fn (u64, u64) u64) u128 = null,
}) type {
    return struct {
        const Self = @This();
        rs1_val: u64,
        rs2_val: u64,

        pub fn init(rs1_val: u64, rs2_val: u64) Self {
            return .{ .rs1_val = rs1_val, .rs2_val = rs2_val };
        }
        pub fn lookupTable() LookupTables(XLEN) {
            return cfg.table;
        }
        pub fn toLookupIndex(self: Self) u128 {
            if (cfg.customIndex) |f| return f(self.rs1_val, self.rs2_val, cfg.computeResult);
            if (cfg.interleave) return lookup_table.interleaveBits(self.rs1_val, self.rs2_val);
            return @as(u128, cfg.computeResult(self.rs1_val, self.rs2_val));
        }
        pub fn computeResult(self: Self) u64 {
            return cfg.computeResult(self.rs1_val, self.rs2_val);
        }
        pub fn circuitFlags() CircuitFlagSet {
            return cfg.circuit_flags;
        }
        pub fn instructionFlags() InstructionFlagSet {
            return cfg.instruction_flags;
        }
    };
}

/// ADD instruction lookup
/// Computes rd = rs1 + rs2 using a single range check lookup
pub fn AddLookup(comptime XLEN: comptime_int) type {
    return BinaryLookup(XLEN, .{
        .table = .RangeCheck,
        .interleave = false,
        .circuit_flags = comptime blk: {
            var f = CircuitFlagSet.init();
            f.set(.AddOperands);
            f.set(.WriteLookupOutputToRD);
            break :blk f;
        },
        .instruction_flags = comptime blk: {
            var f = InstructionFlagSet.init();
            f.set(.LeftOperandIsRs1Value);
            f.set(.RightOperandIsRs2Value);
            break :blk f;
        },
        .computeResult = struct {
            fn f(a: u64, b: u64) u64 {
                return a +% b;
            }
        }.f,
    });
}

/// SUB instruction lookup
/// Computes rd = rs1 - rs2 using the Sub lookup table
pub fn SubLookup(comptime XLEN: comptime_int) type {
    return BinaryLookup(XLEN, .{
        .table = .Sub,
        .circuit_flags = comptime blk: {
            var f = CircuitFlagSet.init();
            f.set(.SubtractOperands);
            f.set(.WriteLookupOutputToRD);
            break :blk f;
        },
        .instruction_flags = comptime blk: {
            var f = InstructionFlagSet.init();
            f.set(.LeftOperandIsRs1Value);
            f.set(.RightOperandIsRs2Value);
            break :blk f;
        },
        .computeResult = struct {
            fn f(a: u64, b: u64) u64 {
                return a -% b;
            }
        }.f,
    });
}

/// AND instruction lookup
/// Computes rd = rs1 & rs2 using the And lookup table
pub fn AndLookup(comptime XLEN: comptime_int) type {
    return BinaryLookup(XLEN, .{
        .table = .And,
        .circuit_flags = comptime blk: {
            var f = CircuitFlagSet.init();
            f.set(.WriteLookupOutputToRD);
            break :blk f;
        },
        .instruction_flags = comptime blk: {
            var f = InstructionFlagSet.init();
            f.set(.LeftOperandIsRs1Value);
            f.set(.RightOperandIsRs2Value);
            break :blk f;
        },
        .computeResult = struct {
            fn f(a: u64, b: u64) u64 {
                return a & b;
            }
        }.f,
    });
}

/// OR instruction lookup
/// Computes rd = rs1 | rs2 using the Or lookup table
pub fn OrLookup(comptime XLEN: comptime_int) type {
    return BinaryLookup(XLEN, .{
        .table = .Or,
        .circuit_flags = comptime blk: {
            var f = CircuitFlagSet.init();
            f.set(.WriteLookupOutputToRD);
            break :blk f;
        },
        .instruction_flags = comptime blk: {
            var f = InstructionFlagSet.init();
            f.set(.LeftOperandIsRs1Value);
            f.set(.RightOperandIsRs2Value);
            break :blk f;
        },
        .computeResult = struct {
            fn f(a: u64, b: u64) u64 {
                return a | b;
            }
        }.f,
    });
}

/// XOR instruction lookup
/// Computes rd = rs1 ^ rs2 using the Xor lookup table
pub fn XorLookup(comptime XLEN: comptime_int) type {
    return BinaryLookup(XLEN, .{
        .table = .Xor,
        .circuit_flags = comptime blk: {
            var f = CircuitFlagSet.init();
            f.set(.WriteLookupOutputToRD);
            break :blk f;
        },
        .instruction_flags = comptime blk: {
            var f = InstructionFlagSet.init();
            f.set(.LeftOperandIsRs1Value);
            f.set(.RightOperandIsRs2Value);
            break :blk f;
        },
        .computeResult = struct {
            fn f(a: u64, b: u64) u64 {
                return a ^ b;
            }
        }.f,
    });
}

/// SLT (Set Less Than) instruction lookup
/// Computes rd = (rs1 < rs2) ? 1 : 0 using SignedLessThan table
pub fn SltLookup(comptime XLEN: comptime_int) type {
    return BinaryLookup(XLEN, .{
        .table = .SignedLessThan,
        .circuit_flags = comptime blk: {
            var f = CircuitFlagSet.init();
            f.set(.WriteLookupOutputToRD);
            break :blk f;
        },
        .instruction_flags = comptime blk: {
            var f = InstructionFlagSet.init();
            f.set(.LeftOperandIsRs1Value);
            f.set(.RightOperandIsRs2Value);
            break :blk f;
        },
        .computeResult = struct {
            fn f(a: u64, b: u64) u64 {
                const sa: i64 = @bitCast(a);
                const sb: i64 = @bitCast(b);
                return if (sa < sb) 1 else 0;
            }
        }.f,
    });
}

/// SLTU (Set Less Than Unsigned) instruction lookup
/// Computes rd = (rs1 < rs2) ? 1 : 0 using UnsignedLessThan table
pub fn SltuLookup(comptime XLEN: comptime_int) type {
    return BinaryLookup(XLEN, .{
        .table = .UnsignedLessThan,
        .circuit_flags = comptime blk: {
            var f = CircuitFlagSet.init();
            f.set(.WriteLookupOutputToRD);
            break :blk f;
        },
        .instruction_flags = comptime blk: {
            var f = InstructionFlagSet.init();
            f.set(.LeftOperandIsRs1Value);
            f.set(.RightOperandIsRs2Value);
            break :blk f;
        },
        .computeResult = struct {
            fn f(a: u64, b: u64) u64 {
                return if (a < b) 1 else 0;
            }
        }.f,
    });
}

/// BEQ (Branch if Equal) instruction lookup
/// Uses Equal table to check if rs1 == rs2
pub fn BeqLookup(comptime XLEN: comptime_int) type {
    return BinaryLookup(XLEN, .{
        .table = .Equal,
        .circuit_flags = CircuitFlagSet.init(),
        .instruction_flags = comptime blk: {
            var f = InstructionFlagSet.init();
            f.set(.LeftOperandIsRs1Value);
            f.set(.RightOperandIsRs2Value);
            f.set(.Branch);
            break :blk f;
        },
        .computeResult = struct {
            fn f(a: u64, b: u64) u64 {
                return if (a == b) 1 else 0;
            }
        }.f,
    });
}

/// BNE (Branch if Not Equal) instruction lookup
/// Uses NotEqual table
pub fn BneLookup(comptime XLEN: comptime_int) type {
    return BinaryLookup(XLEN, .{
        .table = .NotEqual,
        .circuit_flags = CircuitFlagSet.init(),
        .instruction_flags = comptime blk: {
            var f = InstructionFlagSet.init();
            f.set(.LeftOperandIsRs1Value);
            f.set(.RightOperandIsRs2Value);
            f.set(.Branch);
            break :blk f;
        },
        .computeResult = struct {
            fn f(a: u64, b: u64) u64 {
                return if (a != b) 1 else 0;
            }
        }.f,
    });
}

/// BLT (Branch if Less Than - Signed) instruction lookup
/// Uses SignedLessThan table
pub fn BltLookup(comptime XLEN: comptime_int) type {
    return BinaryLookup(XLEN, .{
        .table = .SignedLessThan,
        .circuit_flags = CircuitFlagSet.init(),
        .instruction_flags = comptime blk: {
            var f = InstructionFlagSet.init();
            f.set(.LeftOperandIsRs1Value);
            f.set(.RightOperandIsRs2Value);
            f.set(.Branch);
            break :blk f;
        },
        .computeResult = struct {
            fn f(a: u64, b: u64) u64 {
                const sa: i64 = @bitCast(a);
                const sb: i64 = @bitCast(b);
                return if (sa < sb) 1 else 0;
            }
        }.f,
    });
}

/// BGE (Branch if Greater Than or Equal - Signed) instruction lookup
/// Uses SignedGreaterThanEqual table
pub fn BgeLookup(comptime XLEN: comptime_int) type {
    return BinaryLookup(XLEN, .{
        .table = .SignedGreaterThanEqual,
        .circuit_flags = CircuitFlagSet.init(),
        .instruction_flags = comptime blk: {
            var f = InstructionFlagSet.init();
            f.set(.LeftOperandIsRs1Value);
            f.set(.RightOperandIsRs2Value);
            f.set(.Branch);
            break :blk f;
        },
        .computeResult = struct {
            fn f(a: u64, b: u64) u64 {
                const sa: i64 = @bitCast(a);
                const sb: i64 = @bitCast(b);
                return if (sa >= sb) 1 else 0;
            }
        }.f,
    });
}

/// BLTU (Branch if Less Than Unsigned) instruction lookup
/// Uses UnsignedLessThan table
pub fn BltuLookup(comptime XLEN: comptime_int) type {
    return BinaryLookup(XLEN, .{
        .table = .UnsignedLessThan,
        .circuit_flags = CircuitFlagSet.init(),
        .instruction_flags = comptime blk: {
            var f = InstructionFlagSet.init();
            f.set(.LeftOperandIsRs1Value);
            f.set(.RightOperandIsRs2Value);
            f.set(.Branch);
            break :blk f;
        },
        .computeResult = struct {
            fn f(a: u64, b: u64) u64 {
                return if (a < b) 1 else 0;
            }
        }.f,
    });
}

/// BGEU (Branch if Greater Than or Equal Unsigned) instruction lookup
/// Uses UnsignedGreaterThanEqual table
pub fn BgeuLookup(comptime XLEN: comptime_int) type {
    return BinaryLookup(XLEN, .{
        .table = .UnsignedGreaterThanEqual,
        .circuit_flags = CircuitFlagSet.init(),
        .instruction_flags = comptime blk: {
            var f = InstructionFlagSet.init();
            f.set(.LeftOperandIsRs1Value);
            f.set(.RightOperandIsRs2Value);
            f.set(.Branch);
            break :blk f;
        },
        .computeResult = struct {
            fn f(a: u64, b: u64) u64 {
                return if (a >= b) 1 else 0;
            }
        }.f,
    });
}

/// LUI (Load Upper Immediate) instruction lookup
/// Loads a 20-bit immediate into the upper bits of rd, zeroing lower 12 bits
/// rd = imm << 12
pub fn LuiLookup(comptime XLEN: comptime_int) type {
    return struct {
        const Self = @This();

        imm: i32, // The immediate value (already shifted by 12 from instruction encoding)

        pub fn init(imm: i32) Self {
            return Self{
                .imm = imm,
            };
        }

        pub fn lookupTable() LookupTables(XLEN) {
            // LUI just moves the immediate to rd, no complex lookup needed
            return .RangeCheck;
        }

        pub fn toLookupIndex(self: Self) u128 {
            // The result value is the index for range check
            return @as(u128, self.computeResult());
        }

        pub fn computeResult(self: Self) u64 {
            // For RV64, sign-extend the 32-bit result to 64 bits
            if (XLEN == 64) {
                return @bitCast(@as(i64, self.imm));
            } else {
                return @as(u64, @bitCast(self.imm));
            }
        }

        pub fn circuitFlags() CircuitFlagSet {
            var flags = CircuitFlagSet.init();
            flags.set(.WriteLookupOutputToRD);
            return flags;
        }

        pub fn instructionFlags() InstructionFlagSet {
            var flags = InstructionFlagSet.init();
            flags.set(.RightOperandIsImm);
            return flags;
        }
    };
}

/// AUIPC (Add Upper Immediate to PC) instruction lookup
/// rd = PC + (imm << 12)
pub fn AuipcLookup(comptime XLEN: comptime_int) type {
    return struct {
        const Self = @This();

        pc: u64, // Current program counter
        imm: i32, // The immediate value (already shifted by 12)

        pub fn init(pc: u64, imm: i32) Self {
            return Self{
                .pc = pc,
                .imm = imm,
            };
        }

        pub fn lookupTable() LookupTables(XLEN) {
            // AUIPC is PC + immediate, similar to ADD
            return .RangeCheck;
        }

        pub fn toLookupIndex(self: Self) u128 {
            return @as(u128, self.computeResult());
        }

        pub fn computeResult(self: Self) u64 {
            // PC + sign-extended immediate
            const imm_extended: i64 = @as(i64, self.imm);
            return self.pc +% @as(u64, @bitCast(imm_extended));
        }

        pub fn circuitFlags() CircuitFlagSet {
            var flags = CircuitFlagSet.init();
            flags.set(.AddOperands);
            flags.set(.WriteLookupOutputToRD);
            return flags;
        }

        pub fn instructionFlags() InstructionFlagSet {
            var flags = InstructionFlagSet.init();
            flags.set(.LeftOperandIsPC);
            flags.set(.RightOperandIsImm);
            return flags;
        }
    };
}

/// JAL (Jump and Link) instruction lookup
/// rd = PC + 4 (or PC + 2 for compressed), PC = PC + imm
pub fn JalLookup(comptime XLEN: comptime_int) type {
    return struct {
        const Self = @This();

        pc: u64, // Current program counter
        imm: i32, // Jump offset (already decoded from J-type)
        is_compressed: bool, // Whether this is a compressed instruction

        pub fn init(pc: u64, imm: i32, is_compressed: bool) Self {
            return Self{
                .pc = pc,
                .imm = imm,
                .is_compressed = is_compressed,
            };
        }

        pub fn lookupTable() LookupTables(XLEN) {
            // JAL is PC + 4 (link address), no complex lookup needed
            return .RangeCheck;
        }

        pub fn toLookupIndex(self: Self) u128 {
            return @as(u128, self.computeResult());
        }

        /// Compute the return address (what gets written to rd)
        pub fn computeResult(self: Self) u64 {
            // Return address is PC + instruction size
            const inc: u64 = if (self.is_compressed) 2 else 4;
            return self.pc +% inc;
        }

        /// Compute the jump target address
        pub fn computeTarget(self: Self) u64 {
            const imm_extended: i64 = @as(i64, self.imm);
            return self.pc +% @as(u64, @bitCast(imm_extended));
        }

        pub fn circuitFlags() CircuitFlagSet {
            var flags = CircuitFlagSet.init();
            flags.set(.Jump);
            flags.set(.WriteLookupOutputToRD);
            return flags;
        }

        pub fn instructionFlags() InstructionFlagSet {
            var flags = InstructionFlagSet.init();
            flags.set(.LeftOperandIsPC);
            flags.set(.RightOperandIsImm);
            return flags;
        }
    };
}

/// JALR (Jump and Link Register) instruction lookup
/// rd = PC + 4 (or PC + 2 for compressed), PC = (rs1 + imm) & ~1
pub fn JalrLookup(comptime XLEN: comptime_int) type {
    return struct {
        const Self = @This();

        pc: u64, // Current program counter
        rs1_val: u64, // Base address from register
        imm: i32, // Offset (12-bit sign-extended)
        is_compressed: bool, // Whether this is a compressed instruction

        pub fn init(pc: u64, rs1_val: u64, imm: i32, is_compressed: bool) Self {
            return Self{
                .pc = pc,
                .rs1_val = rs1_val,
                .imm = imm,
                .is_compressed = is_compressed,
            };
        }

        pub fn lookupTable() LookupTables(XLEN) {
            return .RangeCheck;
        }

        pub fn toLookupIndex(self: Self) u128 {
            return @as(u128, self.computeResult());
        }

        /// Compute the return address (what gets written to rd)
        pub fn computeResult(self: Self) u64 {
            const inc: u64 = if (self.is_compressed) 2 else 4;
            return self.pc +% inc;
        }

        /// Compute the jump target address
        pub fn computeTarget(self: Self) u64 {
            const imm_extended: i64 = @as(i64, self.imm);
            const target = self.rs1_val +% @as(u64, @bitCast(imm_extended));
            // Clear the least significant bit
            return target & ~@as(u64, 1);
        }

        pub fn circuitFlags() CircuitFlagSet {
            var flags = CircuitFlagSet.init();
            flags.set(.Jump);
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

/// SLL (Shift Left Logical) instruction lookup
/// Computes rd = rs1 << (rs2 & (XLEN-1))
pub fn SllLookup(comptime XLEN: comptime_int) type {
    return BinaryLookup(XLEN, .{
        .table = .LeftShift,
        .circuit_flags = comptime blk: {
            var f = CircuitFlagSet.init();
            f.set(.WriteLookupOutputToRD);
            break :blk f;
        },
        .instruction_flags = comptime blk: {
            var f = InstructionFlagSet.init();
            f.set(.LeftOperandIsRs1Value);
            f.set(.RightOperandIsRs2Value);
            break :blk f;
        },
        .computeResult = struct {
            fn f(a: u64, b: u64) u64 {
                const shift_mask: u6 = XLEN - 1;
                const shift: u6 = @truncate(b & @as(u64, shift_mask));
                const mask: u64 = if (XLEN == 64) ~@as(u64, 0) else (@as(u64, 1) << XLEN) - 1;
                return (a << shift) & mask;
            }
        }.f,
    });
}

/// SRL (Shift Right Logical) instruction lookup
/// Computes rd = rs1 >> (rs2 & (XLEN-1)) (logical shift)
pub fn SrlLookup(comptime XLEN: comptime_int) type {
    return BinaryLookup(XLEN, .{
        .table = .RightShift,
        .circuit_flags = comptime blk: {
            var f = CircuitFlagSet.init();
            f.set(.WriteLookupOutputToRD);
            break :blk f;
        },
        .instruction_flags = comptime blk: {
            var f = InstructionFlagSet.init();
            f.set(.LeftOperandIsRs1Value);
            f.set(.RightOperandIsRs2Value);
            break :blk f;
        },
        .computeResult = struct {
            fn f(a: u64, b: u64) u64 {
                const shift_mask: u6 = XLEN - 1;
                const shift: u6 = @truncate(b & @as(u64, shift_mask));
                const mask: u64 = if (XLEN == 64) ~@as(u64, 0) else (@as(u64, 1) << XLEN) - 1;
                return (a & mask) >> shift;
            }
        }.f,
    });
}

/// SRA (Shift Right Arithmetic) instruction lookup
/// Computes rd = rs1 >> (rs2 & (XLEN-1)) (arithmetic shift, sign-extending)
pub fn SraLookup(comptime XLEN: comptime_int) type {
    return BinaryLookup(XLEN, .{
        .table = .RightShiftArithmetic,
        .circuit_flags = comptime blk: {
            var f = CircuitFlagSet.init();
            f.set(.WriteLookupOutputToRD);
            break :blk f;
        },
        .instruction_flags = comptime blk: {
            var f = InstructionFlagSet.init();
            f.set(.LeftOperandIsRs1Value);
            f.set(.RightOperandIsRs2Value);
            break :blk f;
        },
        .computeResult = struct {
            fn f(a: u64, b: u64) u64 {
                const shift_mask: u6 = XLEN - 1;
                const shift: u6 = @truncate(b & @as(u64, shift_mask));
                if (XLEN == 64) {
                    const signed_val: i64 = @bitCast(a);
                    return @bitCast(signed_val >> shift);
                } else {
                    const mask: u64 = (@as(u64, 1) << XLEN) - 1;
                    const masked = a & mask;
                    const shift_for_sign = 64 - XLEN;
                    const signed_val: i64 = @as(i64, @bitCast(masked << @truncate(shift_for_sign))) >> @truncate(shift_for_sign);
                    return @as(u64, @bitCast(signed_val >> shift)) & mask;
                }
            }
        }.f,
    });
}

/// SLLI (Shift Left Logical Immediate) instruction lookup
/// Computes rd = rs1 << imm
pub fn SlliLookup(comptime XLEN: comptime_int) type {
    return BinaryLookup(XLEN, .{
        .table = .LeftShift,
        .circuit_flags = comptime blk: {
            var f = CircuitFlagSet.init();
            f.set(.WriteLookupOutputToRD);
            break :blk f;
        },
        .instruction_flags = comptime blk: {
            var f = InstructionFlagSet.init();
            f.set(.LeftOperandIsRs1Value);
            f.set(.RightOperandIsImm);
            break :blk f;
        },
        .computeResult = struct {
            fn f(a: u64, b: u64) u64 {
                const shift_mask: u6 = XLEN - 1;
                const shift: u6 = @truncate(b & @as(u64, shift_mask));
                const mask: u64 = if (XLEN == 64) ~@as(u64, 0) else (@as(u64, 1) << XLEN) - 1;
                return (a << shift) & mask;
            }
        }.f,
    });
}

/// SRLI (Shift Right Logical Immediate) instruction lookup
/// Computes rd = rs1 >> imm (logical shift)
pub fn SrliLookup(comptime XLEN: comptime_int) type {
    return BinaryLookup(XLEN, .{
        .table = .RightShift,
        .circuit_flags = comptime blk: {
            var f = CircuitFlagSet.init();
            f.set(.WriteLookupOutputToRD);
            break :blk f;
        },
        .instruction_flags = comptime blk: {
            var f = InstructionFlagSet.init();
            f.set(.LeftOperandIsRs1Value);
            f.set(.RightOperandIsImm);
            break :blk f;
        },
        .computeResult = struct {
            fn f(a: u64, b: u64) u64 {
                const shift_mask: u6 = XLEN - 1;
                const shift: u6 = @truncate(b & @as(u64, shift_mask));
                const mask: u64 = if (XLEN == 64) ~@as(u64, 0) else (@as(u64, 1) << XLEN) - 1;
                return (a & mask) >> shift;
            }
        }.f,
    });
}

/// SRAI (Shift Right Arithmetic Immediate) instruction lookup
/// Computes rd = rs1 >> imm (arithmetic shift, sign-extending)
pub fn SraiLookup(comptime XLEN: comptime_int) type {
    return BinaryLookup(XLEN, .{
        .table = .RightShiftArithmetic,
        .circuit_flags = comptime blk: {
            var f = CircuitFlagSet.init();
            f.set(.WriteLookupOutputToRD);
            break :blk f;
        },
        .instruction_flags = comptime blk: {
            var f = InstructionFlagSet.init();
            f.set(.LeftOperandIsRs1Value);
            f.set(.RightOperandIsImm);
            break :blk f;
        },
        .computeResult = struct {
            fn f(a: u64, b: u64) u64 {
                const shift_mask: u6 = XLEN - 1;
                const shift: u6 = @truncate(b & @as(u64, shift_mask));
                if (XLEN == 64) {
                    const signed_val: i64 = @bitCast(a);
                    return @bitCast(signed_val >> shift);
                } else {
                    const mask: u64 = (@as(u64, 1) << XLEN) - 1;
                    const masked = a & mask;
                    const shift_for_sign = 64 - XLEN;
                    const signed_val: i64 = @as(i64, @bitCast(masked << @truncate(shift_for_sign))) >> @truncate(shift_for_sign);
                    return @as(u64, @bitCast(signed_val >> shift)) & mask;
                }
            }
        }.f,
    });
}

/// MUL (Multiply) instruction lookup
/// Computes rd = (rs1 * rs2)[XLEN-1:0] (low bits of product)
pub fn MulLookup(comptime XLEN: comptime_int) type {
    return BinaryLookup(XLEN, .{
        .table = .RangeCheck,
        .interleave = false,
        .circuit_flags = comptime blk: {
            var f = CircuitFlagSet.init();
            f.set(.MultiplyOperands);
            f.set(.WriteLookupOutputToRD);
            break :blk f;
        },
        .instruction_flags = comptime blk: {
            var f = InstructionFlagSet.init();
            f.set(.LeftOperandIsRs1Value);
            f.set(.RightOperandIsRs2Value);
            break :blk f;
        },
        .computeResult = struct {
            fn f(a: u64, b: u64) u64 {
                return a *% b;
            }
        }.f,
    });
}

/// MULH (Multiply High Signed) instruction lookup
/// Computes rd = (rs1 * rs2)[2*XLEN-1:XLEN] (high bits of signed product)
pub fn MulhLookup(comptime XLEN: comptime_int) type {
    return BinaryLookup(XLEN, .{
        .table = .RangeCheck,
        .interleave = false,
        .circuit_flags = comptime blk: {
            var f = CircuitFlagSet.init();
            f.set(.MultiplyOperands);
            f.set(.WriteLookupOutputToRD);
            break :blk f;
        },
        .instruction_flags = comptime blk: {
            var f = InstructionFlagSet.init();
            f.set(.LeftOperandIsRs1Value);
            f.set(.RightOperandIsRs2Value);
            break :blk f;
        },
        .computeResult = struct {
            fn f(a: u64, b: u64) u64 {
                if (XLEN == 64) {
                    const sa: i64 = @bitCast(a);
                    const sb: i64 = @bitCast(b);
                    const product: i128 = @as(i128, sa) * @as(i128, sb);
                    const high_bits: i64 = @truncate(product >> 64);
                    return @bitCast(high_bits);
                } else {
                    const mask: u64 = (@as(u64, 1) << XLEN) - 1;
                    const a_masked = a & mask;
                    const b_masked = b & mask;
                    const shift = 64 - XLEN;
                    const a_signed: i64 = @as(i64, @bitCast(a_masked << @truncate(shift))) >> @truncate(shift);
                    const b_signed: i64 = @as(i64, @bitCast(b_masked << @truncate(shift))) >> @truncate(shift);
                    const product: i128 = @as(i128, a_signed) * @as(i128, b_signed);
                    const high_bits: i64 = @truncate(product >> XLEN);
                    return @as(u64, @bitCast(high_bits)) & mask;
                }
            }
        }.f,
    });
}

/// MULHU (Multiply High Unsigned) instruction lookup
/// Computes rd = (rs1 * rs2)[2*XLEN-1:XLEN] (high bits of unsigned product)
pub fn MulhuLookup(comptime XLEN: comptime_int) type {
    return BinaryLookup(XLEN, .{
        .table = .RangeCheck,
        .interleave = false,
        .circuit_flags = comptime blk: {
            var f = CircuitFlagSet.init();
            f.set(.MultiplyOperands);
            f.set(.WriteLookupOutputToRD);
            break :blk f;
        },
        .instruction_flags = comptime blk: {
            var f = InstructionFlagSet.init();
            f.set(.LeftOperandIsRs1Value);
            f.set(.RightOperandIsRs2Value);
            break :blk f;
        },
        .computeResult = struct {
            fn f(a: u64, b: u64) u64 {
                if (XLEN == 64) {
                    const product: u128 = @as(u128, a) * @as(u128, b);
                    return @truncate(product >> 64);
                } else {
                    const mask: u64 = (@as(u64, 1) << XLEN) - 1;
                    const product: u128 = @as(u128, a & mask) * @as(u128, b & mask);
                    return @as(u64, @truncate(product >> XLEN)) & mask;
                }
            }
        }.f,
    });
}

/// MULHSU (Multiply High Signed-Unsigned) instruction lookup
/// Computes rd = (signed(rs1) * unsigned(rs2))[2*XLEN-1:XLEN]
pub fn MulhsuLookup(comptime XLEN: comptime_int) type {
    return BinaryLookup(XLEN, .{
        .table = .RangeCheck,
        .interleave = false,
        .circuit_flags = comptime blk: {
            var f = CircuitFlagSet.init();
            f.set(.MultiplyOperands);
            f.set(.WriteLookupOutputToRD);
            break :blk f;
        },
        .instruction_flags = comptime blk: {
            var f = InstructionFlagSet.init();
            f.set(.LeftOperandIsRs1Value);
            f.set(.RightOperandIsRs2Value);
            break :blk f;
        },
        .computeResult = struct {
            fn f(a: u64, b: u64) u64 {
                if (XLEN == 64) {
                    const sa: i64 = @bitCast(a);
                    const product: i128 = @as(i128, sa) * @as(i128, @as(i64, @bitCast(b)));
                    const high_bits: i64 = @truncate(product >> 64);
                    return @bitCast(high_bits);
                } else {
                    const mask: u64 = (@as(u64, 1) << XLEN) - 1;
                    const shift = 64 - XLEN;
                    const a_signed: i64 = @as(i64, @bitCast((a & mask) << @truncate(shift))) >> @truncate(shift);
                    const product: i128 = @as(i128, a_signed) * @as(i128, @as(i64, @bitCast(b & mask)));
                    const high_bits: i64 = @truncate(product >> XLEN);
                    return @as(u64, @bitCast(high_bits)) & mask;
                }
            }
        }.f,
    });
}

// ============================================================================
// Division and Remainder Lookups (M Extension)
// ============================================================================

/// DIV instruction lookup - signed integer division
/// Computes rd = rs1 / rs2 (signed)
/// Special cases:
/// - Division by zero: returns -1 (all bits set)
/// - Overflow (MIN_INT / -1): returns MIN_INT
pub fn DivLookup(comptime XLEN: comptime_int) type {
    return BinaryLookup(XLEN, .{
        .table = .ValidDiv0,
        .circuit_flags = comptime blk: {
            var f = CircuitFlagSet.init();
            f.set(.WriteLookupOutputToRD);
            break :blk f;
        },
        .instruction_flags = comptime blk: {
            var f = InstructionFlagSet.init();
            f.set(.LeftOperandIsRs1Value);
            f.set(.RightOperandIsRs2Value);
            break :blk f;
        },
        .computeResult = struct {
            fn f(a: u64, b: u64) u64 {
                if (XLEN == 64) {
                    const dividend: i64 = @bitCast(a);
                    const divisor: i64 = @bitCast(b);
                    if (divisor == 0) return @as(u64, @bitCast(@as(i64, -1)));
                    const min_int: i64 = @bitCast(@as(u64, 0x8000000000000000));
                    if (dividend == min_int and divisor == -1) return a;
                    return @bitCast(@divTrunc(dividend, divisor));
                } else if (XLEN == 32) {
                    const mask: u64 = 0xFFFFFFFF;
                    const dividend: i32 = @bitCast(@as(u32, @truncate(a & mask)));
                    const divisor: i32 = @bitCast(@as(u32, @truncate(b & mask)));
                    if (divisor == 0) return 0xFFFFFFFF;
                    const min_int: i32 = @bitCast(@as(u32, 0x80000000));
                    if (dividend == min_int and divisor == -1) return @as(u64, @bitCast(@as(u32, @bitCast(dividend))));
                    return @as(u64, @as(u32, @bitCast(@divTrunc(dividend, divisor))));
                } else {
                    const mask: u64 = 0xFF;
                    const dividend: i8 = @bitCast(@as(u8, @truncate(a & mask)));
                    const divisor: i8 = @bitCast(@as(u8, @truncate(b & mask)));
                    if (divisor == 0) return 0xFF;
                    const min_int: i8 = @bitCast(@as(u8, 0x80));
                    if (dividend == min_int and divisor == -1) return @as(u64, @bitCast(@as(u8, @bitCast(dividend))));
                    return @as(u64, @as(u8, @bitCast(@divTrunc(dividend, divisor))));
                }
            }
        }.f,
        .customIndex = struct {
            fn f(rs1: u64, rs2: u64, compute: *const fn (u64, u64) u64) u128 {
                // Interleave divisor and quotient
                return lookup_table.interleaveBits(rs2, compute(rs1, rs2));
            }
        }.f,
    });
}

/// DIVU instruction lookup - unsigned integer division
/// Computes rd = rs1 / rs2 (unsigned)
/// Division by zero returns MAX_VALUE (all bits set)
pub fn DivuLookup(comptime XLEN: comptime_int) type {
    return BinaryLookup(XLEN, .{
        .table = .ValidDiv0,
        .circuit_flags = comptime blk: {
            var f = CircuitFlagSet.init();
            f.set(.WriteLookupOutputToRD);
            break :blk f;
        },
        .instruction_flags = comptime blk: {
            var f = InstructionFlagSet.init();
            f.set(.LeftOperandIsRs1Value);
            f.set(.RightOperandIsRs2Value);
            break :blk f;
        },
        .computeResult = struct {
            fn f(a: u64, b: u64) u64 {
                if (XLEN == 64) {
                    if (b == 0) return 0xFFFFFFFFFFFFFFFF;
                    return a / b;
                } else if (XLEN == 32) {
                    const mask: u64 = 0xFFFFFFFF;
                    const dividend: u32 = @truncate(a & mask);
                    const divisor: u32 = @truncate(b & mask);
                    if (divisor == 0) return 0xFFFFFFFF;
                    return @as(u64, dividend / divisor);
                } else {
                    const mask: u64 = 0xFF;
                    const dividend: u8 = @truncate(a & mask);
                    const divisor: u8 = @truncate(b & mask);
                    if (divisor == 0) return 0xFF;
                    return @as(u64, dividend / divisor);
                }
            }
        }.f,
        .customIndex = struct {
            fn f(rs1: u64, rs2: u64, compute: *const fn (u64, u64) u64) u128 {
                return lookup_table.interleaveBits(rs2, compute(rs1, rs2));
            }
        }.f,
    });
}

/// REM instruction lookup - signed integer remainder
/// Computes rd = rs1 % rs2 (signed)
/// Division by zero returns the dividend
/// Overflow (MIN_INT % -1) returns 0
pub fn RemLookup(comptime XLEN: comptime_int) type {
    return BinaryLookup(XLEN, .{
        .table = .ValidSignedRemainder,
        .circuit_flags = comptime blk: {
            var f = CircuitFlagSet.init();
            f.set(.WriteLookupOutputToRD);
            break :blk f;
        },
        .instruction_flags = comptime blk: {
            var f = InstructionFlagSet.init();
            f.set(.LeftOperandIsRs1Value);
            f.set(.RightOperandIsRs2Value);
            break :blk f;
        },
        .computeResult = struct {
            fn f(a: u64, b: u64) u64 {
                if (XLEN == 64) {
                    const dividend: i64 = @bitCast(a);
                    const divisor: i64 = @bitCast(b);
                    if (divisor == 0) return a;
                    const min_int: i64 = @bitCast(@as(u64, 0x8000000000000000));
                    if (dividend == min_int and divisor == -1) return 0;
                    return @bitCast(@rem(dividend, divisor));
                } else if (XLEN == 32) {
                    const mask: u64 = 0xFFFFFFFF;
                    const dividend: i32 = @bitCast(@as(u32, @truncate(a & mask)));
                    const divisor: i32 = @bitCast(@as(u32, @truncate(b & mask)));
                    if (divisor == 0) return a & mask;
                    const min_int: i32 = @bitCast(@as(u32, 0x80000000));
                    if (dividend == min_int and divisor == -1) return 0;
                    return @as(u64, @as(u32, @bitCast(@rem(dividend, divisor))));
                } else {
                    const mask: u64 = 0xFF;
                    const dividend: i8 = @bitCast(@as(u8, @truncate(a & mask)));
                    const divisor: i8 = @bitCast(@as(u8, @truncate(b & mask)));
                    if (divisor == 0) return a & mask;
                    const min_int: i8 = @bitCast(@as(u8, 0x80));
                    if (dividend == min_int and divisor == -1) return 0;
                    return @as(u64, @as(u8, @bitCast(@rem(dividend, divisor))));
                }
            }
        }.f,
        .customIndex = struct {
            fn f(rs1: u64, rs2: u64, compute: *const fn (u64, u64) u64) u128 {
                // Interleave remainder and divisor
                return lookup_table.interleaveBits(compute(rs1, rs2), rs2);
            }
        }.f,
    });
}

/// REMU instruction lookup - unsigned integer remainder
/// Computes rd = rs1 % rs2 (unsigned)
/// Division by zero returns the dividend
pub fn RemuLookup(comptime XLEN: comptime_int) type {
    return BinaryLookup(XLEN, .{
        .table = .ValidUnsignedRemainder,
        .circuit_flags = comptime blk: {
            var f = CircuitFlagSet.init();
            f.set(.WriteLookupOutputToRD);
            break :blk f;
        },
        .instruction_flags = comptime blk: {
            var f = InstructionFlagSet.init();
            f.set(.LeftOperandIsRs1Value);
            f.set(.RightOperandIsRs2Value);
            break :blk f;
        },
        .computeResult = struct {
            fn f(a: u64, b: u64) u64 {
                if (XLEN == 64) {
                    if (b == 0) return a;
                    return a % b;
                } else if (XLEN == 32) {
                    const mask: u64 = 0xFFFFFFFF;
                    const dividend: u32 = @truncate(a & mask);
                    const divisor: u32 = @truncate(b & mask);
                    if (divisor == 0) return a & mask;
                    return @as(u64, dividend % divisor);
                } else {
                    const mask: u64 = 0xFF;
                    const dividend: u8 = @truncate(a & mask);
                    const divisor: u8 = @truncate(b & mask);
                    if (divisor == 0) return a & mask;
                    return @as(u64, dividend % divisor);
                }
            }
        }.f,
        .customIndex = struct {
            fn f(rs1: u64, rs2: u64, compute: *const fn (u64, u64) u64) u128 {
                return lookup_table.interleaveBits(compute(rs1, rs2), rs2);
            }
        }.f,
    });
}

// ============================================================================
// Word-Sized Operations (RV64 *W instructions)
// ============================================================================

/// ADDW instruction lookup - add 32-bit values, sign-extend to 64-bit
/// Computes rd = sext((rs1[31:0] + rs2[31:0])[31:0])
pub fn AddwLookup(comptime XLEN: comptime_int) type {
    return BinaryLookup(XLEN, .{
        .table = .RangeCheck,
        .interleave = false,
        .circuit_flags = comptime blk: {
            var f = CircuitFlagSet.init();
            f.set(.AddOperands);
            f.set(.WriteLookupOutputToRD);
            break :blk f;
        },
        .instruction_flags = comptime blk: {
            var f = InstructionFlagSet.init();
            f.set(.LeftOperandIsRs1Value);
            f.set(.RightOperandIsRs2Value);
            break :blk f;
        },
        .computeResult = struct {
            fn f(a: u64, b: u64) u64 {
                const sum32: u32 = @as(u32, @truncate(a)) +% @as(u32, @truncate(b));
                return @bitCast(@as(i64, @as(i32, @bitCast(sum32))));
            }
        }.f,
    });
}

/// SUBW instruction lookup - subtract 32-bit values, sign-extend to 64-bit
/// Computes rd = sext((rs1[31:0] - rs2[31:0])[31:0])
pub fn SubwLookup(comptime XLEN: comptime_int) type {
    return BinaryLookup(XLEN, .{
        .table = .Sub,
        .circuit_flags = comptime blk: {
            var f = CircuitFlagSet.init();
            f.set(.SubtractOperands);
            f.set(.WriteLookupOutputToRD);
            break :blk f;
        },
        .instruction_flags = comptime blk: {
            var f = InstructionFlagSet.init();
            f.set(.LeftOperandIsRs1Value);
            f.set(.RightOperandIsRs2Value);
            break :blk f;
        },
        .computeResult = struct {
            fn f(a: u64, b: u64) u64 {
                const diff32: u32 = @as(u32, @truncate(a)) -% @as(u32, @truncate(b));
                return @bitCast(@as(i64, @as(i32, @bitCast(diff32))));
            }
        }.f,
        .customIndex = struct {
            fn f(rs1: u64, rs2: u64, _: *const fn (u64, u64) u64) u128 {
                return lookup_table.interleaveBits(rs1 & 0xFFFFFFFF, rs2 & 0xFFFFFFFF);
            }
        }.f,
    });
}

/// SLLW instruction lookup - shift left 32-bit, sign-extend to 64-bit
/// Computes rd = sext((rs1[31:0] << (rs2[4:0]))[31:0])
pub fn SllwLookup(comptime XLEN: comptime_int) type {
    return BinaryLookup(XLEN, .{
        .table = .LeftShift,
        .circuit_flags = comptime blk: {
            var f = CircuitFlagSet.init();
            f.set(.WriteLookupOutputToRD);
            break :blk f;
        },
        .instruction_flags = comptime blk: {
            var f = InstructionFlagSet.init();
            f.set(.LeftOperandIsRs1Value);
            f.set(.RightOperandIsRs2Value);
            break :blk f;
        },
        .computeResult = struct {
            fn f(a: u64, b: u64) u64 {
                const a32: u32 = @truncate(a);
                const shift: u5 = @truncate(b);
                const result32: u32 = a32 << shift;
                return @bitCast(@as(i64, @as(i32, @bitCast(result32))));
            }
        }.f,
        .customIndex = struct {
            fn f(rs1: u64, rs2: u64, _: *const fn (u64, u64) u64) u128 {
                return lookup_table.interleaveBits(rs1 & 0xFFFFFFFF, rs2 & 0x1F);
            }
        }.f,
    });
}

/// SRLW instruction lookup - logical shift right 32-bit, sign-extend to 64-bit
/// Computes rd = sext((rs1[31:0] >> (rs2[4:0]))[31:0])
pub fn SrlwLookup(comptime XLEN: comptime_int) type {
    return BinaryLookup(XLEN, .{
        .table = .RightShift,
        .circuit_flags = comptime blk: {
            var f = CircuitFlagSet.init();
            f.set(.WriteLookupOutputToRD);
            break :blk f;
        },
        .instruction_flags = comptime blk: {
            var f = InstructionFlagSet.init();
            f.set(.LeftOperandIsRs1Value);
            f.set(.RightOperandIsRs2Value);
            break :blk f;
        },
        .computeResult = struct {
            fn f(a: u64, b: u64) u64 {
                const a32: u32 = @truncate(a);
                const shift: u5 = @truncate(b);
                const result32: u32 = a32 >> shift;
                return @bitCast(@as(i64, @as(i32, @bitCast(result32))));
            }
        }.f,
        .customIndex = struct {
            fn f(rs1: u64, rs2: u64, _: *const fn (u64, u64) u64) u128 {
                return lookup_table.interleaveBits(rs1 & 0xFFFFFFFF, rs2 & 0x1F);
            }
        }.f,
    });
}

/// SRAW instruction lookup - arithmetic shift right 32-bit, sign-extend to 64-bit
/// Computes rd = sext((rs1[31:0] >>s (rs2[4:0]))[31:0])
pub fn SrawLookup(comptime XLEN: comptime_int) type {
    return BinaryLookup(XLEN, .{
        .table = .RightShiftArithmetic,
        .circuit_flags = comptime blk: {
            var f = CircuitFlagSet.init();
            f.set(.WriteLookupOutputToRD);
            break :blk f;
        },
        .instruction_flags = comptime blk: {
            var f = InstructionFlagSet.init();
            f.set(.LeftOperandIsRs1Value);
            f.set(.RightOperandIsRs2Value);
            break :blk f;
        },
        .computeResult = struct {
            fn f(a: u64, b: u64) u64 {
                const a32: i32 = @bitCast(@as(u32, @truncate(a)));
                const shift: u5 = @truncate(b);
                return @bitCast(@as(i64, a32 >> shift));
            }
        }.f,
        .customIndex = struct {
            fn f(rs1: u64, rs2: u64, _: *const fn (u64, u64) u64) u128 {
                return lookup_table.interleaveBits(rs1 & 0xFFFFFFFF, rs2 & 0x1F);
            }
        }.f,
    });
}

// ============================================================================
// RV64I Immediate Word Operations
// ============================================================================

/// ADDIW instruction lookup - add immediate word
/// Computes rd = sext((rs1[31:0] + sext(imm))[31:0])
pub fn AddiwLookup(comptime XLEN: comptime_int) type {
    return struct {
        const Self = @This();

        rs1_val: u64,
        imm: i32,

        pub fn init(rs1_val: u64, imm: i32) Self {
            return Self{
                .rs1_val = rs1_val,
                .imm = imm,
            };
        }

        pub fn lookupTable() LookupTables(XLEN) {
            return .RangeCheck;
        }

        pub fn toLookupIndex(self: Self) u128 {
            return @as(u128, self.computeResult() & 0xFFFFFFFF);
        }

        /// Compute ADDIW result: sext((rs1[31:0] + imm)[31:0])
        pub fn computeResult(self: Self) u64 {
            const a32: i32 = @bitCast(@as(u32, @truncate(self.rs1_val)));
            const result32: i32 = a32 +% self.imm;
            const extended: i64 = @as(i64, result32);
            return @bitCast(extended);
        }

        pub fn circuitFlags() CircuitFlagSet {
            var flags = CircuitFlagSet.init();
            flags.set(.AddOperands);
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

/// SLLIW instruction lookup - shift left immediate word
/// Computes rd = sext((rs1[31:0] << shamt)[31:0])
pub fn SlliwLookup(comptime XLEN: comptime_int) type {
    return struct {
        const Self = @This();

        rs1_val: u64,
        shamt: u5,

        pub fn init(rs1_val: u64, shamt: u5) Self {
            return Self{
                .rs1_val = rs1_val,
                .shamt = shamt,
            };
        }

        pub fn lookupTable() LookupTables(XLEN) {
            return .LeftShift;
        }

        pub fn toLookupIndex(self: Self) u128 {
            return lookup_table.interleaveBits(self.rs1_val & 0xFFFFFFFF, @as(u64, self.shamt));
        }

        /// Compute SLLIW result
        pub fn computeResult(self: Self) u64 {
            const a32: u32 = @truncate(self.rs1_val);
            const result32: u32 = a32 << self.shamt;
            const signed32: i32 = @bitCast(result32);
            const extended: i64 = @as(i64, signed32);
            return @bitCast(extended);
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

/// SRLIW instruction lookup - logical shift right immediate word
/// Computes rd = sext((rs1[31:0] >> shamt)[31:0])
pub fn SrliwLookup(comptime XLEN: comptime_int) type {
    return struct {
        const Self = @This();

        rs1_val: u64,
        shamt: u5,

        pub fn init(rs1_val: u64, shamt: u5) Self {
            return Self{
                .rs1_val = rs1_val,
                .shamt = shamt,
            };
        }

        pub fn lookupTable() LookupTables(XLEN) {
            return .RightShift;
        }

        pub fn toLookupIndex(self: Self) u128 {
            return lookup_table.interleaveBits(self.rs1_val & 0xFFFFFFFF, @as(u64, self.shamt));
        }

        /// Compute SRLIW result
        pub fn computeResult(self: Self) u64 {
            const a32: u32 = @truncate(self.rs1_val);
            const result32: u32 = a32 >> self.shamt;
            const signed32: i32 = @bitCast(result32);
            const extended: i64 = @as(i64, signed32);
            return @bitCast(extended);
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

/// SRAIW instruction lookup - arithmetic shift right immediate word
/// Computes rd = sext((rs1[31:0] >>s shamt)[31:0])
pub fn SraiwLookup(comptime XLEN: comptime_int) type {
    return struct {
        const Self = @This();

        rs1_val: u64,
        shamt: u5,

        pub fn init(rs1_val: u64, shamt: u5) Self {
            return Self{
                .rs1_val = rs1_val,
                .shamt = shamt,
            };
        }

        pub fn lookupTable() LookupTables(XLEN) {
            return .RightShiftArithmetic;
        }

        pub fn toLookupIndex(self: Self) u128 {
            return lookup_table.interleaveBits(self.rs1_val & 0xFFFFFFFF, @as(u64, self.shamt));
        }

        /// Compute SRAIW result (arithmetic shift preserves sign)
        pub fn computeResult(self: Self) u64 {
            const a32: i32 = @bitCast(@as(u32, @truncate(self.rs1_val)));
            const result32: i32 = a32 >> self.shamt;
            const extended: i64 = @as(i64, result32);
            return @bitCast(extended);
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

/// MULW instruction lookup - multiply 32-bit values, sign-extend to 64-bit
/// Computes rd = sext((rs1[31:0] * rs2[31:0])[31:0])
pub fn MulwLookup(comptime XLEN: comptime_int) type {
    return BinaryLookup(XLEN, .{
        .table = .RangeCheck,
        .interleave = false,
        .circuit_flags = comptime blk: {
            var f = CircuitFlagSet.init();
            f.set(.MultiplyOperands);
            f.set(.WriteLookupOutputToRD);
            break :blk f;
        },
        .instruction_flags = comptime blk: {
            var f = InstructionFlagSet.init();
            f.set(.LeftOperandIsRs1Value);
            f.set(.RightOperandIsRs2Value);
            break :blk f;
        },
        .computeResult = struct {
            fn f(a: u64, b: u64) u64 {
                const product32: u32 = @as(u32, @truncate(a)) *% @as(u32, @truncate(b));
                return @bitCast(@as(i64, @as(i32, @bitCast(product32))));
            }
        }.f,
    });
}

/// DIVW instruction lookup - signed division 32-bit, sign-extend to 64-bit
/// Computes rd = sext((rs1[31:0] /s rs2[31:0])[31:0])
pub fn DivwLookup(comptime XLEN: comptime_int) type {
    return BinaryLookup(XLEN, .{
        .table = .ValidDiv0,
        .circuit_flags = comptime blk: {
            var f = CircuitFlagSet.init();
            f.set(.WriteLookupOutputToRD);
            break :blk f;
        },
        .instruction_flags = comptime blk: {
            var f = InstructionFlagSet.init();
            f.set(.LeftOperandIsRs1Value);
            f.set(.RightOperandIsRs2Value);
            break :blk f;
        },
        .computeResult = struct {
            fn f(a: u64, b: u64) u64 {
                const dividend: i32 = @bitCast(@as(u32, @truncate(a)));
                const divisor: i32 = @bitCast(@as(u32, @truncate(b)));
                if (divisor == 0) return @as(u64, @bitCast(@as(i64, -1)));
                const min_int: i32 = @bitCast(@as(u32, 0x80000000));
                if (dividend == min_int and divisor == -1) return @as(u64, @bitCast(@as(i64, @as(i32, min_int))));
                return @bitCast(@as(i64, @divTrunc(dividend, divisor)));
            }
        }.f,
        .customIndex = struct {
            fn f(rs1: u64, rs2: u64, compute: *const fn (u64, u64) u64) u128 {
                return lookup_table.interleaveBits(rs2 & 0xFFFFFFFF, compute(rs1, rs2) & 0xFFFFFFFF);
            }
        }.f,
    });
}

/// DIVUW instruction lookup - unsigned division 32-bit, sign-extend to 64-bit
/// Computes rd = sext((rs1[31:0] /u rs2[31:0])[31:0])
pub fn DivuwLookup(comptime XLEN: comptime_int) type {
    return BinaryLookup(XLEN, .{
        .table = .ValidDiv0,
        .circuit_flags = comptime blk: {
            var f = CircuitFlagSet.init();
            f.set(.WriteLookupOutputToRD);
            break :blk f;
        },
        .instruction_flags = comptime blk: {
            var f = InstructionFlagSet.init();
            f.set(.LeftOperandIsRs1Value);
            f.set(.RightOperandIsRs2Value);
            break :blk f;
        },
        .computeResult = struct {
            fn f(a: u64, b: u64) u64 {
                const dividend: u32 = @truncate(a);
                const divisor: u32 = @truncate(b);
                if (divisor == 0) return @as(u64, @bitCast(@as(i64, -1)));
                const quotient: u32 = dividend / divisor;
                return @bitCast(@as(i64, @as(i32, @bitCast(quotient))));
            }
        }.f,
        .customIndex = struct {
            fn f(rs1: u64, rs2: u64, compute: *const fn (u64, u64) u64) u128 {
                return lookup_table.interleaveBits(rs2 & 0xFFFFFFFF, compute(rs1, rs2) & 0xFFFFFFFF);
            }
        }.f,
    });
}

/// REMW instruction lookup - signed remainder 32-bit, sign-extend to 64-bit
/// Computes rd = sext((rs1[31:0] %s rs2[31:0])[31:0])
pub fn RemwLookup(comptime XLEN: comptime_int) type {
    return BinaryLookup(XLEN, .{
        .table = .ValidSignedRemainder,
        .circuit_flags = comptime blk: {
            var f = CircuitFlagSet.init();
            f.set(.WriteLookupOutputToRD);
            break :blk f;
        },
        .instruction_flags = comptime blk: {
            var f = InstructionFlagSet.init();
            f.set(.LeftOperandIsRs1Value);
            f.set(.RightOperandIsRs2Value);
            break :blk f;
        },
        .computeResult = struct {
            fn f(a: u64, b: u64) u64 {
                const dividend: i32 = @bitCast(@as(u32, @truncate(a)));
                const divisor: i32 = @bitCast(@as(u32, @truncate(b)));
                if (divisor == 0) return @bitCast(@as(i64, dividend));
                const min_int: i32 = @bitCast(@as(u32, 0x80000000));
                if (dividend == min_int and divisor == -1) return 0;
                return @bitCast(@as(i64, @rem(dividend, divisor)));
            }
        }.f,
        .customIndex = struct {
            fn f(rs1: u64, rs2: u64, compute: *const fn (u64, u64) u64) u128 {
                return lookup_table.interleaveBits(compute(rs1, rs2) & 0xFFFFFFFF, rs2 & 0xFFFFFFFF);
            }
        }.f,
    });
}

/// REMUW instruction lookup - unsigned remainder 32-bit, sign-extend to 64-bit
/// Computes rd = sext((rs1[31:0] %u rs2[31:0])[31:0])
pub fn RemuwLookup(comptime XLEN: comptime_int) type {
    return BinaryLookup(XLEN, .{
        .table = .ValidUnsignedRemainder,
        .circuit_flags = comptime blk: {
            var f = CircuitFlagSet.init();
            f.set(.WriteLookupOutputToRD);
            break :blk f;
        },
        .instruction_flags = comptime blk: {
            var f = InstructionFlagSet.init();
            f.set(.LeftOperandIsRs1Value);
            f.set(.RightOperandIsRs2Value);
            break :blk f;
        },
        .computeResult = struct {
            fn f(a: u64, b: u64) u64 {
                const dividend: u32 = @truncate(a);
                const divisor: u32 = @truncate(b);
                if (divisor == 0) return @bitCast(@as(i64, @as(i32, @bitCast(dividend))));
                const remainder: u32 = dividend % divisor;
                return @bitCast(@as(i64, @as(i32, @bitCast(remainder))));
            }
        }.f,
        .customIndex = struct {
            fn f(rs1: u64, rs2: u64, compute: *const fn (u64, u64) u64) u128 {
                return lookup_table.interleaveBits(compute(rs1, rs2) & 0xFFFFFFFF, rs2 & 0xFFFFFFFF);
            }
        }.f,
    });
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

test "div lookup" {
    // Normal division: 42 / 6 = 7
    const div1 = DivLookup(64).init(42, 6);
    try std.testing.expectEqual(@as(u64, 7), div1.computeResult());

    // Negative dividend: -42 / 6 = -7
    const neg42: u64 = @bitCast(@as(i64, -42));
    const div2 = DivLookup(64).init(neg42, 6);
    const neg7: u64 = @bitCast(@as(i64, -7));
    try std.testing.expectEqual(neg7, div2.computeResult());

    // Division by zero: x / 0 = -1 (all bits set)
    const div3 = DivLookup(64).init(42, 0);
    try std.testing.expectEqual(@as(u64, 0xFFFFFFFFFFFFFFFF), div3.computeResult());

    // Overflow: MIN_INT / -1 = MIN_INT
    const min_int: u64 = 0x8000000000000000;
    const neg1: u64 = @bitCast(@as(i64, -1));
    const div4 = DivLookup(64).init(min_int, neg1);
    try std.testing.expectEqual(min_int, div4.computeResult());
}

test "divu lookup" {
    // Normal division: 42 / 6 = 7
    const divu1 = DivuLookup(64).init(42, 6);
    try std.testing.expectEqual(@as(u64, 7), divu1.computeResult());

    // Large unsigned: 0xFFFFFFFFFFFFFFFF / 2 = 0x7FFFFFFFFFFFFFFF
    const divu2 = DivuLookup(64).init(0xFFFFFFFFFFFFFFFF, 2);
    try std.testing.expectEqual(@as(u64, 0x7FFFFFFFFFFFFFFF), divu2.computeResult());

    // Division by zero: x / 0 = MAX
    const divu3 = DivuLookup(64).init(42, 0);
    try std.testing.expectEqual(@as(u64, 0xFFFFFFFFFFFFFFFF), divu3.computeResult());
}

test "rem lookup" {
    // Normal remainder: 42 % 5 = 2
    const rem1 = RemLookup(64).init(42, 5);
    try std.testing.expectEqual(@as(u64, 2), rem1.computeResult());

    // Negative dividend: -7 % 3 = -1
    const neg7: u64 = @bitCast(@as(i64, -7));
    const rem2 = RemLookup(64).init(neg7, 3);
    const neg1: u64 = @bitCast(@as(i64, -1));
    try std.testing.expectEqual(neg1, rem2.computeResult());

    // Division by zero: x % 0 = x
    const rem3 = RemLookup(64).init(42, 0);
    try std.testing.expectEqual(@as(u64, 42), rem3.computeResult());

    // Overflow: MIN_INT % -1 = 0
    const min_int: u64 = 0x8000000000000000;
    const rem4 = RemLookup(64).init(min_int, neg1);
    try std.testing.expectEqual(@as(u64, 0), rem4.computeResult());
}

test "remu lookup" {
    // Normal remainder: 42 % 5 = 2
    const remu1 = RemuLookup(64).init(42, 5);
    try std.testing.expectEqual(@as(u64, 2), remu1.computeResult());

    // Large unsigned
    const remu2 = RemuLookup(64).init(0xFFFFFFFFFFFFFFFF, 10);
    try std.testing.expectEqual(@as(u64, 5), remu2.computeResult());

    // Division by zero: x % 0 = x
    const remu3 = RemuLookup(64).init(42, 0);
    try std.testing.expectEqual(@as(u64, 42), remu3.computeResult());
}

// ============================================================================
// Word-Sized Instruction Tests (*W instructions for RV64)
// ============================================================================

test "addw lookup" {
    // Simple: 10 + 20 = 30, sign-extended to 64 bits
    const addw1 = AddwLookup(64).init(10, 20);
    try std.testing.expectEqual(@as(u64, 30), addw1.computeResult());

    // Upper bits should be ignored
    const addw2 = AddwLookup(64).init(0xFFFFFFFF00000010, 0xFFFFFFFF00000020);
    try std.testing.expectEqual(@as(u64, 0x30), addw2.computeResult());

    // Result with sign bit: 0x80000000 sign-extended
    const addw3 = AddwLookup(64).init(0x7FFFFFFF, 1);
    const expected3: u64 = @bitCast(@as(i64, @as(i32, @bitCast(@as(u32, 0x80000000)))));
    try std.testing.expectEqual(expected3, addw3.computeResult());
}

test "subw lookup" {
    // Simple: 30 - 10 = 20
    const subw1 = SubwLookup(64).init(30, 10);
    try std.testing.expectEqual(@as(u64, 20), subw1.computeResult());

    // Wraparound: 10 - 20 = -10, sign-extended
    const subw2 = SubwLookup(64).init(10, 20);
    const expected: u64 = @bitCast(@as(i64, -10));
    try std.testing.expectEqual(expected, subw2.computeResult());
}

test "sllw lookup" {
    // Simple: 1 << 4 = 16
    const sllw1 = SllwLookup(64).init(1, 4);
    try std.testing.expectEqual(@as(u64, 16), sllw1.computeResult());

    // Shift into sign bit: 0x40000000 << 1 = 0x80000000 (negative)
    const sllw2 = SllwLookup(64).init(0x40000000, 1);
    const expected: u64 = @bitCast(@as(i64, @as(i32, @bitCast(@as(u32, 0x80000000)))));
    try std.testing.expectEqual(expected, sllw2.computeResult());
}

test "srlw lookup" {
    // Simple: 256 >> 4 = 16
    const srlw1 = SrlwLookup(64).init(256, 4);
    try std.testing.expectEqual(@as(u64, 16), srlw1.computeResult());

    // 0x80000000 >> 1 = 0x40000000 (positive after shift)
    const srlw2 = SrlwLookup(64).init(0x80000000, 1);
    try std.testing.expectEqual(@as(u64, 0x40000000), srlw2.computeResult());
}

test "sraw lookup" {
    // Positive: 256 >> 4 = 16
    const sraw_pos = SrawLookup(64).init(256, 4);
    try std.testing.expectEqual(@as(u64, 16), sraw_pos.computeResult());

    // Negative: 0x80000000 >> 1 = 0xC0000000 (sign-extended)
    const sraw_neg = SrawLookup(64).init(0x80000000, 1);
    const expected: u64 = @bitCast(@as(i64, @as(i32, @bitCast(@as(u32, 0xC0000000)))));
    try std.testing.expectEqual(expected, sraw_neg.computeResult());
}

test "mulw lookup" {
    // Simple: 6 * 7 = 42
    const mulw1 = MulwLookup(64).init(6, 7);
    try std.testing.expectEqual(@as(u64, 42), mulw1.computeResult());

    // Overflow wraps: 0x40000000 * 2 = 0x80000000 (negative result)
    const mulw2 = MulwLookup(64).init(0x40000000, 2);
    const expected: u64 = @bitCast(@as(i64, @as(i32, @bitCast(@as(u32, 0x80000000)))));
    try std.testing.expectEqual(expected, mulw2.computeResult());
}

test "divw lookup" {
    // Normal: 42 / 6 = 7
    const divw1 = DivwLookup(64).init(42, 6);
    try std.testing.expectEqual(@as(u64, 7), divw1.computeResult());

    // Division by zero: returns -1
    const divw2 = DivwLookup(64).init(42, 0);
    try std.testing.expectEqual(@as(u64, @bitCast(@as(i64, -1))), divw2.computeResult());

    // Overflow: MIN_INT / -1 = MIN_INT
    const min32: u64 = 0x80000000;
    const neg1: u64 = @as(u64, @as(u32, @bitCast(@as(i32, -1))));
    const divw3 = DivwLookup(64).init(min32, neg1);
    const expected: u64 = @bitCast(@as(i64, @as(i32, @bitCast(@as(u32, 0x80000000)))));
    try std.testing.expectEqual(expected, divw3.computeResult());
}

test "remw lookup" {
    // Normal: 42 % 5 = 2
    const remw1 = RemwLookup(64).init(42, 5);
    try std.testing.expectEqual(@as(u64, 2), remw1.computeResult());

    // Negative dividend: -7 % 3 = -1
    const neg7_32: u64 = @as(u64, @as(u32, @bitCast(@as(i32, -7))));
    const remw2 = RemwLookup(64).init(neg7_32, 3);
    const expected: u64 = @bitCast(@as(i64, @as(i32, -1)));
    try std.testing.expectEqual(expected, remw2.computeResult());
}

test "blt lookup (signed less than branch)" {
    // Positive: 5 < 10 = true
    const blt1 = BltLookup(64).init(5, 10);
    try std.testing.expectEqual(@as(u64, 1), blt1.computeResult());

    // Positive: 10 < 5 = false
    const blt2 = BltLookup(64).init(10, 5);
    try std.testing.expectEqual(@as(u64, 0), blt2.computeResult());

    // Negative: -5 < 10 = true
    const neg5: u64 = @bitCast(@as(i64, -5));
    const blt3 = BltLookup(64).init(neg5, 10);
    try std.testing.expectEqual(@as(u64, 1), blt3.computeResult());

    // Negative vs Positive: -1 < 0 = true
    const neg1: u64 = @bitCast(@as(i64, -1));
    const blt4 = BltLookup(64).init(neg1, 0);
    try std.testing.expectEqual(@as(u64, 1), blt4.computeResult());

    // Equal: 5 < 5 = false
    const blt5 = BltLookup(64).init(5, 5);
    try std.testing.expectEqual(@as(u64, 0), blt5.computeResult());

    // Check branch flag
    const flags = BltLookup(64).instructionFlags();
    try std.testing.expect(flags.get(.Branch));
}

test "bge lookup (signed greater than or equal branch)" {
    // Positive: 10 >= 5 = true
    const bge1 = BgeLookup(64).init(10, 5);
    try std.testing.expectEqual(@as(u64, 1), bge1.computeResult());

    // Positive: 5 >= 10 = false
    const bge2 = BgeLookup(64).init(5, 10);
    try std.testing.expectEqual(@as(u64, 0), bge2.computeResult());

    // Equal: 5 >= 5 = true
    const bge3 = BgeLookup(64).init(5, 5);
    try std.testing.expectEqual(@as(u64, 1), bge3.computeResult());

    // Negative: 0 >= -1 = true
    const neg1: u64 = @bitCast(@as(i64, -1));
    const bge4 = BgeLookup(64).init(0, neg1);
    try std.testing.expectEqual(@as(u64, 1), bge4.computeResult());

    // Check branch flag
    const flags = BgeLookup(64).instructionFlags();
    try std.testing.expect(flags.get(.Branch));
}

test "bltu lookup (unsigned less than branch)" {
    // Simple: 5 < 10 = true
    const bltu1 = BltuLookup(64).init(5, 10);
    try std.testing.expectEqual(@as(u64, 1), bltu1.computeResult());

    // Simple: 10 < 5 = false
    const bltu2 = BltuLookup(64).init(10, 5);
    try std.testing.expectEqual(@as(u64, 0), bltu2.computeResult());

    // -1 as unsigned is MAX, so MAX < 10 = false
    const max_u64: u64 = @bitCast(@as(i64, -1));
    const bltu3 = BltuLookup(64).init(max_u64, 10);
    try std.testing.expectEqual(@as(u64, 0), bltu3.computeResult());

    // 10 < MAX = true
    const bltu4 = BltuLookup(64).init(10, max_u64);
    try std.testing.expectEqual(@as(u64, 1), bltu4.computeResult());

    // Check branch flag
    const flags = BltuLookup(64).instructionFlags();
    try std.testing.expect(flags.get(.Branch));
}

test "bgeu lookup (unsigned greater than or equal branch)" {
    // Simple: 10 >= 5 = true
    const bgeu1 = BgeuLookup(64).init(10, 5);
    try std.testing.expectEqual(@as(u64, 1), bgeu1.computeResult());

    // Simple: 5 >= 10 = false
    const bgeu2 = BgeuLookup(64).init(5, 10);
    try std.testing.expectEqual(@as(u64, 0), bgeu2.computeResult());

    // Equal: 5 >= 5 = true
    const bgeu3 = BgeuLookup(64).init(5, 5);
    try std.testing.expectEqual(@as(u64, 1), bgeu3.computeResult());

    // -1 as unsigned is MAX >= 10 = true
    const max_u64: u64 = @bitCast(@as(i64, -1));
    const bgeu4 = BgeuLookup(64).init(max_u64, 10);
    try std.testing.expectEqual(@as(u64, 1), bgeu4.computeResult());

    // Check branch flag
    const flags = BgeuLookup(64).instructionFlags();
    try std.testing.expect(flags.get(.Branch));
}

test "lui lookup (load upper immediate)" {
    // lui x5, 0x12345 => rd = 0x12345000
    const lui1 = LuiLookup(64).init(0x12345000);
    try std.testing.expectEqual(@as(u64, 0x12345000), lui1.computeResult());

    // lui with negative immediate (upper bit set)
    const lui2 = LuiLookup(64).init(@as(i32, @bitCast(@as(u32, 0xFFFFF000))));
    const expected: u64 = @bitCast(@as(i64, @as(i32, @bitCast(@as(u32, 0xFFFFF000)))));
    try std.testing.expectEqual(expected, lui2.computeResult());

    // Check flags
    const flags = LuiLookup(64).instructionFlags();
    try std.testing.expect(flags.get(.RightOperandIsImm));

    const circuit_flags = LuiLookup(64).circuitFlags();
    try std.testing.expect(circuit_flags.get(.WriteLookupOutputToRD));
}

test "auipc lookup (add upper immediate to PC)" {
    // auipc x5, 0x1000 => rd = PC + 0x1000000
    const auipc1 = AuipcLookup(64).init(0x1000, 0x1000000);
    try std.testing.expectEqual(@as(u64, 0x1001000), auipc1.computeResult());

    // PC at 0x80000000, with imm 0x1000
    const auipc2 = AuipcLookup(64).init(0x80000000, 0x1000000);
    try std.testing.expectEqual(@as(u64, 0x81000000), auipc2.computeResult());

    // Check flags
    const flags = AuipcLookup(64).instructionFlags();
    try std.testing.expect(flags.get(.LeftOperandIsPC));
    try std.testing.expect(flags.get(.RightOperandIsImm));

    const circuit_flags = AuipcLookup(64).circuitFlags();
    try std.testing.expect(circuit_flags.get(.AddOperands));
    try std.testing.expect(circuit_flags.get(.WriteLookupOutputToRD));
}

test "jal lookup (jump and link)" {
    // jal x1, 100 at PC=0x1000 => rd = 0x1004, target = 0x1064
    const jal1 = JalLookup(64).init(0x1000, 100, false);
    try std.testing.expectEqual(@as(u64, 0x1004), jal1.computeResult()); // link address
    try std.testing.expectEqual(@as(u64, 0x1064), jal1.computeTarget()); // jump target

    // Negative offset: jal x1, -20 at PC=0x2000
    const jal2 = JalLookup(64).init(0x2000, -20, false);
    try std.testing.expectEqual(@as(u64, 0x2004), jal2.computeResult());
    try std.testing.expectEqual(@as(u64, 0x1FEC), jal2.computeTarget()); // 0x2000 - 20

    // Compressed instruction
    const jal_c = JalLookup(64).init(0x1000, 100, true);
    try std.testing.expectEqual(@as(u64, 0x1002), jal_c.computeResult()); // PC + 2 for compressed

    // Check flags
    const circuit_flags = JalLookup(64).circuitFlags();
    try std.testing.expect(circuit_flags.get(.Jump));
    try std.testing.expect(circuit_flags.get(.WriteLookupOutputToRD));

    const inst_flags = JalLookup(64).instructionFlags();
    try std.testing.expect(inst_flags.get(.LeftOperandIsPC));
    try std.testing.expect(inst_flags.get(.RightOperandIsImm));
}

test "jalr lookup (jump and link register)" {
    // jalr x1, 100(x5) with x5=0x1000, PC=0x2000
    // rd = 0x2004, target = (0x1000 + 100) & ~1 = 0x1064
    const jalr1 = JalrLookup(64).init(0x2000, 0x1000, 100, false);
    try std.testing.expectEqual(@as(u64, 0x2004), jalr1.computeResult()); // link address
    try std.testing.expectEqual(@as(u64, 0x1064), jalr1.computeTarget()); // jump target

    // Test clearing of LSB
    // jalr x1, 1(x5) with x5=0x1000 => target = 0x1000 (LSB cleared)
    const jalr2 = JalrLookup(64).init(0x2000, 0x1000, 1, false);
    try std.testing.expectEqual(@as(u64, 0x1000), jalr2.computeTarget()); // (0x1001) & ~1 = 0x1000

    // Negative offset
    const jalr3 = JalrLookup(64).init(0x2000, 0x1000, -100, false);
    try std.testing.expectEqual(@as(u64, 0x0F9C), jalr3.computeTarget()); // 0x1000 - 100 = 0xF9C

    // Compressed instruction
    const jalr_c = JalrLookup(64).init(0x2000, 0x1000, 0, true);
    try std.testing.expectEqual(@as(u64, 0x2002), jalr_c.computeResult()); // PC + 2 for compressed

    // Check flags
    const circuit_flags = JalrLookup(64).circuitFlags();
    try std.testing.expect(circuit_flags.get(.Jump));
    try std.testing.expect(circuit_flags.get(.WriteLookupOutputToRD));

    const inst_flags = JalrLookup(64).instructionFlags();
    try std.testing.expect(inst_flags.get(.LeftOperandIsRs1Value));
    try std.testing.expect(inst_flags.get(.RightOperandIsImm));
}

// ============================================================================
// Load/Store Address Computation Lookups
// ============================================================================

/// Load address computation lookup
/// Computes effective address = rs1 + sign_extended_offset for load instructions
/// (LB, LH, LW, LD, LBU, LHU, LWU)
pub fn LoadAddressLookup(comptime XLEN: comptime_int) type {
    return struct {
        const Self = @This();

        /// Base register value
        rs1_val: u64,
        /// Sign-extended immediate offset
        offset: i32,

        /// Create a new load address lookup
        pub fn init(rs1_val: u64, offset: i32) Self {
            return Self{
                .rs1_val = rs1_val,
                .offset = offset,
            };
        }

        /// Load address uses range check to verify valid address
        pub fn lookupTable() LookupTables(XLEN) {
            return .RangeCheck;
        }

        /// Compute the lookup index
        pub fn toLookupIndex(self: Self) u128 {
            // Use the effective address as the lookup index
            return @as(u128, self.computeResult());
        }

        /// Compute effective address = rs1 + sext(offset)
        pub fn computeResult(self: Self) u64 {
            const signed_offset: i64 = @as(i64, self.offset);
            const signed_base: i64 = @bitCast(self.rs1_val);
            const result: i64 = signed_base +% signed_offset;
            return @bitCast(result);
        }

        pub fn circuitFlags() CircuitFlagSet {
            var flags = CircuitFlagSet.init();
            flags.set(.Load);
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

/// Store address computation lookup
/// Computes effective address = rs1 + sign_extended_offset for store instructions
/// (SB, SH, SW, SD)
pub fn StoreAddressLookup(comptime XLEN: comptime_int) type {
    return struct {
        const Self = @This();

        /// Base register value
        rs1_val: u64,
        /// Sign-extended immediate offset
        offset: i32,

        /// Create a new store address lookup
        pub fn init(rs1_val: u64, offset: i32) Self {
            return Self{
                .rs1_val = rs1_val,
                .offset = offset,
            };
        }

        /// Store address uses range check to verify valid address
        pub fn lookupTable() LookupTables(XLEN) {
            return .RangeCheck;
        }

        /// Compute the lookup index
        pub fn toLookupIndex(self: Self) u128 {
            // Use the effective address as the lookup index
            return @as(u128, self.computeResult());
        }

        /// Compute effective address = rs1 + sext(offset)
        pub fn computeResult(self: Self) u64 {
            const signed_offset: i64 = @as(i64, self.offset);
            const signed_base: i64 = @bitCast(self.rs1_val);
            const result: i64 = signed_base +% signed_offset;
            return @bitCast(result);
        }

        pub fn circuitFlags() CircuitFlagSet {
            var flags = CircuitFlagSet.init();
            flags.set(.Store);
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

/// LB (Load Byte) instruction lookup
/// Loads a byte from memory and sign-extends to XLEN bits
pub fn LbLookup(comptime XLEN: comptime_int) type {
    return struct {
        const Self = @This();

        /// Base register value
        rs1_val: u64,
        /// Sign-extended immediate offset
        offset: i32,
        /// Value loaded from memory (for sign extension verification)
        memory_value: u8,

        /// Create a new LB lookup
        pub fn init(rs1_val: u64, offset: i32, memory_value: u8) Self {
            return Self{
                .rs1_val = rs1_val,
                .offset = offset,
                .memory_value = memory_value,
            };
        }

        /// LB uses sign extension table
        pub fn lookupTable() LookupTables(XLEN) {
            return .SignExtend8;
        }

        /// Compute the lookup index (the byte value to sign-extend)
        pub fn toLookupIndex(self: Self) u128 {
            return @as(u128, self.memory_value);
        }

        /// Compute effective address
        pub fn computeAddress(self: Self) u64 {
            const signed_offset: i64 = @as(i64, self.offset);
            const signed_base: i64 = @bitCast(self.rs1_val);
            const result: i64 = signed_base +% signed_offset;
            return @bitCast(result);
        }

        /// Compute the sign-extended result
        pub fn computeResult(self: Self) u64 {
            const signed_byte: i8 = @bitCast(self.memory_value);
            const extended: i64 = @as(i64, signed_byte);
            return @bitCast(extended);
        }

        pub fn circuitFlags() CircuitFlagSet {
            var flags = CircuitFlagSet.init();
            flags.set(.Load);
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

/// LBU (Load Byte Unsigned) instruction lookup
/// Loads a byte from memory and zero-extends to XLEN bits
pub fn LbuLookup(comptime XLEN: comptime_int) type {
    return struct {
        const Self = @This();

        /// Base register value
        rs1_val: u64,
        /// Sign-extended immediate offset
        offset: i32,
        /// Value loaded from memory
        memory_value: u8,

        /// Create a new LBU lookup
        pub fn init(rs1_val: u64, offset: i32, memory_value: u8) Self {
            return Self{
                .rs1_val = rs1_val,
                .offset = offset,
                .memory_value = memory_value,
            };
        }

        /// LBU uses range check (no sign extension needed)
        pub fn lookupTable() LookupTables(XLEN) {
            return .RangeCheck;
        }

        /// Compute the lookup index
        pub fn toLookupIndex(self: Self) u128 {
            return @as(u128, self.memory_value);
        }

        /// Compute effective address
        pub fn computeAddress(self: Self) u64 {
            const signed_offset: i64 = @as(i64, self.offset);
            const signed_base: i64 = @bitCast(self.rs1_val);
            const result: i64 = signed_base +% signed_offset;
            return @bitCast(result);
        }

        /// Compute the zero-extended result
        pub fn computeResult(self: Self) u64 {
            return @as(u64, self.memory_value);
        }

        pub fn circuitFlags() CircuitFlagSet {
            var flags = CircuitFlagSet.init();
            flags.set(.Load);
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

/// LH (Load Halfword) instruction lookup
/// Loads a 16-bit value from memory and sign-extends to XLEN bits
pub fn LhLookup(comptime XLEN: comptime_int) type {
    return struct {
        const Self = @This();

        /// Base register value
        rs1_val: u64,
        /// Sign-extended immediate offset
        offset: i32,
        /// Value loaded from memory
        memory_value: u16,

        /// Create a new LH lookup
        pub fn init(rs1_val: u64, offset: i32, memory_value: u16) Self {
            return Self{
                .rs1_val = rs1_val,
                .offset = offset,
                .memory_value = memory_value,
            };
        }

        /// LH uses sign extension table for 16-bit values
        pub fn lookupTable() LookupTables(XLEN) {
            return .SignExtend16;
        }

        /// Compute the lookup index (the halfword value to sign-extend)
        pub fn toLookupIndex(self: Self) u128 {
            return @as(u128, self.memory_value);
        }

        /// Compute effective address
        pub fn computeAddress(self: Self) u64 {
            const signed_offset: i64 = @as(i64, self.offset);
            const signed_base: i64 = @bitCast(self.rs1_val);
            const result: i64 = signed_base +% signed_offset;
            return @bitCast(result);
        }

        /// Compute the sign-extended result
        pub fn computeResult(self: Self) u64 {
            const signed_half: i16 = @bitCast(self.memory_value);
            const extended: i64 = @as(i64, signed_half);
            return @bitCast(extended);
        }

        pub fn circuitFlags() CircuitFlagSet {
            var flags = CircuitFlagSet.init();
            flags.set(.Load);
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

/// LHU (Load Halfword Unsigned) instruction lookup
/// Loads a 16-bit value from memory and zero-extends to XLEN bits
pub fn LhuLookup(comptime XLEN: comptime_int) type {
    return struct {
        const Self = @This();

        /// Base register value
        rs1_val: u64,
        /// Sign-extended immediate offset
        offset: i32,
        /// Value loaded from memory
        memory_value: u16,

        /// Create a new LHU lookup
        pub fn init(rs1_val: u64, offset: i32, memory_value: u16) Self {
            return Self{
                .rs1_val = rs1_val,
                .offset = offset,
                .memory_value = memory_value,
            };
        }

        /// LHU uses range check (no sign extension)
        pub fn lookupTable() LookupTables(XLEN) {
            return .RangeCheck;
        }

        /// Compute the lookup index
        pub fn toLookupIndex(self: Self) u128 {
            return @as(u128, self.memory_value);
        }

        /// Compute effective address
        pub fn computeAddress(self: Self) u64 {
            const signed_offset: i64 = @as(i64, self.offset);
            const signed_base: i64 = @bitCast(self.rs1_val);
            const result: i64 = signed_base +% signed_offset;
            return @bitCast(result);
        }

        /// Compute the zero-extended result
        pub fn computeResult(self: Self) u64 {
            return @as(u64, self.memory_value);
        }

        pub fn circuitFlags() CircuitFlagSet {
            var flags = CircuitFlagSet.init();
            flags.set(.Load);
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

/// LW (Load Word) instruction lookup
/// Loads a 32-bit value from memory and sign-extends to 64 bits (on RV64)
pub fn LwLookup(comptime XLEN: comptime_int) type {
    return struct {
        const Self = @This();

        /// Base register value
        rs1_val: u64,
        /// Sign-extended immediate offset
        offset: i32,
        /// Value loaded from memory
        memory_value: u32,

        /// Create a new LW lookup
        pub fn init(rs1_val: u64, offset: i32, memory_value: u32) Self {
            return Self{
                .rs1_val = rs1_val,
                .offset = offset,
                .memory_value = memory_value,
            };
        }

        /// LW uses sign extension table for 32-bit values
        pub fn lookupTable() LookupTables(XLEN) {
            return .SignExtend32;
        }

        /// Compute the lookup index
        pub fn toLookupIndex(self: Self) u128 {
            return @as(u128, self.memory_value);
        }

        /// Compute effective address
        pub fn computeAddress(self: Self) u64 {
            const signed_offset: i64 = @as(i64, self.offset);
            const signed_base: i64 = @bitCast(self.rs1_val);
            const result: i64 = signed_base +% signed_offset;
            return @bitCast(result);
        }

        /// Compute the sign-extended result
        pub fn computeResult(self: Self) u64 {
            const signed_word: i32 = @bitCast(self.memory_value);
            const extended: i64 = @as(i64, signed_word);
            return @bitCast(extended);
        }

        pub fn circuitFlags() CircuitFlagSet {
            var flags = CircuitFlagSet.init();
            flags.set(.Load);
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

/// LWU (Load Word Unsigned) instruction lookup - RV64 only
/// Loads a 32-bit value from memory and zero-extends to 64 bits
pub fn LwuLookup(comptime XLEN: comptime_int) type {
    return struct {
        const Self = @This();

        /// Base register value
        rs1_val: u64,
        /// Sign-extended immediate offset
        offset: i32,
        /// Value loaded from memory
        memory_value: u32,

        /// Create a new LWU lookup
        pub fn init(rs1_val: u64, offset: i32, memory_value: u32) Self {
            return Self{
                .rs1_val = rs1_val,
                .offset = offset,
                .memory_value = memory_value,
            };
        }

        /// LWU uses range check (no sign extension)
        pub fn lookupTable() LookupTables(XLEN) {
            return .RangeCheck;
        }

        /// Compute the lookup index
        pub fn toLookupIndex(self: Self) u128 {
            return @as(u128, self.memory_value);
        }

        /// Compute effective address
        pub fn computeAddress(self: Self) u64 {
            const signed_offset: i64 = @as(i64, self.offset);
            const signed_base: i64 = @bitCast(self.rs1_val);
            const result: i64 = signed_base +% signed_offset;
            return @bitCast(result);
        }

        /// Compute the zero-extended result
        pub fn computeResult(self: Self) u64 {
            return @as(u64, self.memory_value);
        }

        pub fn circuitFlags() CircuitFlagSet {
            var flags = CircuitFlagSet.init();
            flags.set(.Load);
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

/// LD (Load Doubleword) instruction lookup - RV64 only
/// Loads a 64-bit value from memory
pub fn LdLookup(comptime XLEN: comptime_int) type {
    return struct {
        const Self = @This();

        /// Base register value
        rs1_val: u64,
        /// Sign-extended immediate offset
        offset: i32,
        /// Value loaded from memory
        memory_value: u64,

        /// Create a new LD lookup
        pub fn init(rs1_val: u64, offset: i32, memory_value: u64) Self {
            return Self{
                .rs1_val = rs1_val,
                .offset = offset,
                .memory_value = memory_value,
            };
        }

        /// LD uses range check
        pub fn lookupTable() LookupTables(XLEN) {
            return .RangeCheck;
        }

        /// Compute the lookup index
        pub fn toLookupIndex(self: Self) u128 {
            return @as(u128, self.memory_value);
        }

        /// Compute effective address
        pub fn computeAddress(self: Self) u64 {
            const signed_offset: i64 = @as(i64, self.offset);
            const signed_base: i64 = @bitCast(self.rs1_val);
            const result: i64 = signed_base +% signed_offset;
            return @bitCast(result);
        }

        /// Compute the result (no extension needed for 64-bit)
        pub fn computeResult(self: Self) u64 {
            return self.memory_value;
        }

        pub fn circuitFlags() CircuitFlagSet {
            var flags = CircuitFlagSet.init();
            flags.set(.Load);
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

/// SB (Store Byte) instruction lookup
/// Stores the lower 8 bits of rs2 to memory
pub fn SbLookup(comptime XLEN: comptime_int) type {
    return struct {
        const Self = @This();

        /// Base register value
        rs1_val: u64,
        /// Sign-extended immediate offset
        offset: i32,
        /// Value to store
        rs2_val: u64,

        /// Create a new SB lookup
        pub fn init(rs1_val: u64, offset: i32, rs2_val: u64) Self {
            return Self{
                .rs1_val = rs1_val,
                .offset = offset,
                .rs2_val = rs2_val,
            };
        }

        /// SB uses range check
        pub fn lookupTable() LookupTables(XLEN) {
            return .RangeCheck;
        }

        /// Compute the lookup index (the byte to store)
        pub fn toLookupIndex(self: Self) u128 {
            return @as(u128, self.computeResult());
        }

        /// Compute effective address
        pub fn computeAddress(self: Self) u64 {
            const signed_offset: i64 = @as(i64, self.offset);
            const signed_base: i64 = @bitCast(self.rs1_val);
            const result: i64 = signed_base +% signed_offset;
            return @bitCast(result);
        }

        /// Compute the byte value to store
        pub fn computeResult(self: Self) u64 {
            return self.rs2_val & 0xFF;
        }

        pub fn circuitFlags() CircuitFlagSet {
            var flags = CircuitFlagSet.init();
            flags.set(.Store);
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

/// SH (Store Halfword) instruction lookup
/// Stores the lower 16 bits of rs2 to memory
pub fn ShLookup(comptime XLEN: comptime_int) type {
    return struct {
        const Self = @This();

        /// Base register value
        rs1_val: u64,
        /// Sign-extended immediate offset
        offset: i32,
        /// Value to store
        rs2_val: u64,

        /// Create a new SH lookup
        pub fn init(rs1_val: u64, offset: i32, rs2_val: u64) Self {
            return Self{
                .rs1_val = rs1_val,
                .offset = offset,
                .rs2_val = rs2_val,
            };
        }

        /// SH uses range check
        pub fn lookupTable() LookupTables(XLEN) {
            return .RangeCheck;
        }

        /// Compute the lookup index
        pub fn toLookupIndex(self: Self) u128 {
            return @as(u128, self.computeResult());
        }

        /// Compute effective address
        pub fn computeAddress(self: Self) u64 {
            const signed_offset: i64 = @as(i64, self.offset);
            const signed_base: i64 = @bitCast(self.rs1_val);
            const result: i64 = signed_base +% signed_offset;
            return @bitCast(result);
        }

        /// Compute the halfword value to store
        pub fn computeResult(self: Self) u64 {
            return self.rs2_val & 0xFFFF;
        }

        pub fn circuitFlags() CircuitFlagSet {
            var flags = CircuitFlagSet.init();
            flags.set(.Store);
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

/// SW (Store Word) instruction lookup
/// Stores the lower 32 bits of rs2 to memory
pub fn SwLookup(comptime XLEN: comptime_int) type {
    return struct {
        const Self = @This();

        /// Base register value
        rs1_val: u64,
        /// Sign-extended immediate offset
        offset: i32,
        /// Value to store
        rs2_val: u64,

        /// Create a new SW lookup
        pub fn init(rs1_val: u64, offset: i32, rs2_val: u64) Self {
            return Self{
                .rs1_val = rs1_val,
                .offset = offset,
                .rs2_val = rs2_val,
            };
        }

        /// SW uses range check
        pub fn lookupTable() LookupTables(XLEN) {
            return .RangeCheck;
        }

        /// Compute the lookup index
        pub fn toLookupIndex(self: Self) u128 {
            return @as(u128, self.computeResult());
        }

        /// Compute effective address
        pub fn computeAddress(self: Self) u64 {
            const signed_offset: i64 = @as(i64, self.offset);
            const signed_base: i64 = @bitCast(self.rs1_val);
            const result: i64 = signed_base +% signed_offset;
            return @bitCast(result);
        }

        /// Compute the word value to store
        pub fn computeResult(self: Self) u64 {
            return self.rs2_val & 0xFFFFFFFF;
        }

        pub fn circuitFlags() CircuitFlagSet {
            var flags = CircuitFlagSet.init();
            flags.set(.Store);
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

/// SD (Store Doubleword) instruction lookup - RV64 only
/// Stores 64 bits of rs2 to memory
pub fn SdLookup(comptime XLEN: comptime_int) type {
    return struct {
        const Self = @This();

        /// Base register value
        rs1_val: u64,
        /// Sign-extended immediate offset
        offset: i32,
        /// Value to store
        rs2_val: u64,

        /// Create a new SD lookup
        pub fn init(rs1_val: u64, offset: i32, rs2_val: u64) Self {
            return Self{
                .rs1_val = rs1_val,
                .offset = offset,
                .rs2_val = rs2_val,
            };
        }

        /// SD uses range check
        pub fn lookupTable() LookupTables(XLEN) {
            return .RangeCheck;
        }

        /// Compute the lookup index
        pub fn toLookupIndex(self: Self) u128 {
            return @as(u128, self.rs2_val);
        }

        /// Compute effective address
        pub fn computeAddress(self: Self) u64 {
            const signed_offset: i64 = @as(i64, self.offset);
            const signed_base: i64 = @bitCast(self.rs1_val);
            const result: i64 = signed_base +% signed_offset;
            return @bitCast(result);
        }

        /// Compute the value to store
        pub fn computeResult(self: Self) u64 {
            return self.rs2_val;
        }

        pub fn circuitFlags() CircuitFlagSet {
            var flags = CircuitFlagSet.init();
            flags.set(.Store);
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

// ============================================================================
// Load/Store Lookup Tests
// ============================================================================

test "load address computation" {
    const load = LoadAddressLookup(64).init(0x1000, 100);
    try std.testing.expectEqual(@as(u64, 0x1064), load.computeResult());

    // Negative offset
    const load_neg = LoadAddressLookup(64).init(0x1000, -100);
    try std.testing.expectEqual(@as(u64, 0x0F9C), load_neg.computeResult());

    // Check flags
    const circuit_flags = LoadAddressLookup(64).circuitFlags();
    try std.testing.expect(circuit_flags.get(.Load));
}

test "store address computation" {
    const store = StoreAddressLookup(64).init(0x2000, 256);
    try std.testing.expectEqual(@as(u64, 0x2100), store.computeResult());

    // Check flags
    const circuit_flags = StoreAddressLookup(64).circuitFlags();
    try std.testing.expect(circuit_flags.get(.Store));
}

test "lb lookup (load byte signed)" {
    // LB with positive byte
    const lb1 = LbLookup(64).init(0x1000, 0, 0x7F);
    try std.testing.expectEqual(@as(u64, 0x007F), lb1.computeResult());
    try std.testing.expectEqual(@as(u64, 0x1000), lb1.computeAddress());

    // LB with negative byte (sign extension)
    const lb2 = LbLookup(64).init(0x1000, 0, 0x80);
    try std.testing.expectEqual(@as(u64, @bitCast(@as(i64, -128))), lb2.computeResult());

    // LB with offset
    const lb3 = LbLookup(64).init(0x1000, 100, 0x42);
    try std.testing.expectEqual(@as(u64, 0x1064), lb3.computeAddress());
}

test "lbu lookup (load byte unsigned)" {
    // LBU with high bit set - should NOT sign extend
    const lbu = LbuLookup(64).init(0x1000, 0, 0x80);
    try std.testing.expectEqual(@as(u64, 0x0080), lbu.computeResult());
    try std.testing.expectEqual(@as(u64, 0x1000), lbu.computeAddress());
}

test "lh lookup (load halfword signed)" {
    // LH with positive halfword
    const lh1 = LhLookup(64).init(0x1000, 0, 0x7FFF);
    try std.testing.expectEqual(@as(u64, 0x7FFF), lh1.computeResult());

    // LH with negative halfword (sign extension)
    const lh2 = LhLookup(64).init(0x1000, 0, 0x8000);
    try std.testing.expectEqual(@as(u64, @bitCast(@as(i64, -32768))), lh2.computeResult());
}

test "lhu lookup (load halfword unsigned)" {
    const lhu = LhuLookup(64).init(0x1000, 0, 0x8000);
    try std.testing.expectEqual(@as(u64, 0x8000), lhu.computeResult());
}

test "lw lookup (load word signed)" {
    // LW with positive word
    const lw1 = LwLookup(64).init(0x1000, 0, 0x7FFFFFFF);
    try std.testing.expectEqual(@as(u64, 0x7FFFFFFF), lw1.computeResult());

    // LW with negative word (sign extension to 64 bits)
    const lw2 = LwLookup(64).init(0x1000, 0, 0x80000000);
    try std.testing.expectEqual(@as(u64, @bitCast(@as(i64, -2147483648))), lw2.computeResult());
}

test "lwu lookup (load word unsigned)" {
    const lwu = LwuLookup(64).init(0x1000, 0, 0x80000000);
    try std.testing.expectEqual(@as(u64, 0x80000000), lwu.computeResult());
}

test "ld lookup (load doubleword)" {
    const ld = LdLookup(64).init(0x1000, 8, 0xDEADBEEF12345678);
    try std.testing.expectEqual(@as(u64, 0xDEADBEEF12345678), ld.computeResult());
    try std.testing.expectEqual(@as(u64, 0x1008), ld.computeAddress());
}

test "sb lookup (store byte)" {
    const sb = SbLookup(64).init(0x1000, 0, 0x12345678);
    try std.testing.expectEqual(@as(u64, 0x78), sb.computeResult());
    try std.testing.expectEqual(@as(u64, 0x1000), sb.computeAddress());
}

test "sh lookup (store halfword)" {
    const sh = ShLookup(64).init(0x1000, 0, 0x12345678);
    try std.testing.expectEqual(@as(u64, 0x5678), sh.computeResult());
}

test "sw lookup (store word)" {
    const sw = SwLookup(64).init(0x1000, 0, 0x123456789ABCDEF0);
    try std.testing.expectEqual(@as(u64, 0x9ABCDEF0), sw.computeResult());
}

test "sd lookup (store doubleword)" {
    const sd = SdLookup(64).init(0x1000, 16, 0xDEADBEEF12345678);
    try std.testing.expectEqual(@as(u64, 0xDEADBEEF12345678), sd.computeResult());
    try std.testing.expectEqual(@as(u64, 0x1010), sd.computeAddress());
}

// ============================================================================
// RV64I Immediate Word Operation Tests
// ============================================================================

test "addiw lookup (add immediate word)" {
    // Positive result
    const addiw1 = AddiwLookup(64).init(10, 20);
    try std.testing.expectEqual(@as(u64, 30), addiw1.computeResult());

    // Result with upper bits cleared (only lower 32 bits + sign extend)
    const addiw2 = AddiwLookup(64).init(0xFFFFFFFF00000005, 10);
    // Lower 32 bits: 5 + 10 = 15, sign-extended to 64 bits
    try std.testing.expectEqual(@as(u64, 15), addiw2.computeResult());

    // Negative result (wraparound)
    const addiw3 = AddiwLookup(64).init(10, -20);
    // 10 - 20 = -10 sign-extended
    try std.testing.expectEqual(@as(u64, @bitCast(@as(i64, -10))), addiw3.computeResult());

    // Result causing 32-bit overflow, sign-extended
    const addiw4 = AddiwLookup(64).init(0x7FFFFFFF, 1);
    // 0x7FFFFFFF + 1 = 0x80000000 (negative in i32), sign-extended
    try std.testing.expectEqual(@as(u64, @bitCast(@as(i64, -2147483648))), addiw4.computeResult());

    // Check flags
    const circuit_flags = AddiwLookup(64).circuitFlags();
    try std.testing.expect(circuit_flags.get(.AddOperands));
    try std.testing.expect(circuit_flags.get(.WriteLookupOutputToRD));

    const inst_flags = AddiwLookup(64).instructionFlags();
    try std.testing.expect(inst_flags.get(.LeftOperandIsRs1Value));
    try std.testing.expect(inst_flags.get(.RightOperandIsImm));
}

test "slliw lookup (shift left immediate word)" {
    // Basic shift
    const slliw1 = SlliwLookup(64).init(1, 4);
    try std.testing.expectEqual(@as(u64, 16), slliw1.computeResult());

    // Shift with upper bits ignored
    const slliw2 = SlliwLookup(64).init(0xFFFFFFFF00000001, 4);
    // Lower 32 bits: 1 << 4 = 16, sign-extended
    try std.testing.expectEqual(@as(u64, 16), slliw2.computeResult());

    // Shift causing high bit set (negative sign-extend)
    const slliw3 = SlliwLookup(64).init(1, 31);
    // 1 << 31 = 0x80000000 (negative), sign-extended
    try std.testing.expectEqual(@as(u64, @bitCast(@as(i64, -2147483648))), slliw3.computeResult());

    // Check flags
    const inst_flags = SlliwLookup(64).instructionFlags();
    try std.testing.expect(inst_flags.get(.LeftOperandIsRs1Value));
    try std.testing.expect(inst_flags.get(.RightOperandIsImm));
}

test "srliw lookup (logical shift right immediate word)" {
    // Basic shift
    const srliw1 = SrliwLookup(64).init(32, 2);
    try std.testing.expectEqual(@as(u64, 8), srliw1.computeResult());

    // Shift with upper bits ignored
    const srliw2 = SrliwLookup(64).init(0xFFFFFFFF00000020, 2);
    // Lower 32 bits: 0x20 >> 2 = 8, sign-extended
    try std.testing.expectEqual(@as(u64, 8), srliw2.computeResult());

    // Logical shift doesn't preserve sign
    const srliw3 = SrliwLookup(64).init(0x80000000, 1);
    // 0x80000000 >> 1 = 0x40000000 (positive), sign-extended
    try std.testing.expectEqual(@as(u64, 0x40000000), srliw3.computeResult());
}

test "sraiw lookup (arithmetic shift right immediate word)" {
    // Basic shift with positive value
    const sraiw1 = SraiwLookup(64).init(32, 2);
    try std.testing.expectEqual(@as(u64, 8), sraiw1.computeResult());

    // Arithmetic shift preserves sign bit
    const sraiw2 = SraiwLookup(64).init(0x80000000, 1);
    // 0x80000000 (i32: MIN_INT) >> 1 = 0xC0000000 (negative), sign-extended
    try std.testing.expectEqual(@as(u64, @bitCast(@as(i64, -1073741824))), sraiw2.computeResult());

    // Full arithmetic shift of -1
    const sraiw3 = SraiwLookup(64).init(0xFFFFFFFF, 31);
    // -1 >> 31 = -1 (all bits stay 1)
    try std.testing.expectEqual(@as(u64, @bitCast(@as(i64, -1))), sraiw3.computeResult());
}

// ============================================================================
// Virtual Instructions
// ============================================================================

/// VirtualSignExtendWord instruction lookup
/// Sign-extends the lower XLEN/2 bits of the input to full XLEN bits.
/// Used as the second step in W-extension instruction decomposition:
///   ADDIW → ADDI + VirtualSignExtendWord
///   ADDW  → ADD  + VirtualSignExtendWord
///   SUBW  → SUB  + VirtualSignExtendWord
///   MULW  → MUL  + VirtualSignExtendWord
///
/// In Jolt, this maps to SignExtendHalfWordTable (table index 21).
/// The lookup operand is (0, rs1_val as u128), and the result is the sign-extended value.
///
/// Reference: jolt-core/src/zkvm/instruction/virtual_sign_extend_word.rs
pub fn VirtualSignExtendWordLookup(comptime XLEN: comptime_int) type {
    return struct {
        const Self = @This();

        /// The value to sign-extend (rs1 value from the base instruction's result)
        rs1_val: u64,
        /// Whether this instruction is part of a virtual sequence (vsr > 0)
        is_virtual: bool,
        /// Whether this instruction should not update the unexpanded PC
        do_not_update_pc: bool,
        /// Whether this is the first instruction in the sequence
        is_first_in_sequence: bool,
        /// Whether the original instruction was compressed
        is_compressed: bool,
        /// Whether the rd register is not x0
        is_rd_not_zero: bool,

        pub fn init(rs1_val: u64, is_virtual: bool, do_not_update_pc: bool, is_first_in_sequence: bool, is_compressed: bool, is_rd_not_zero: bool) Self {
            return Self{
                .rs1_val = rs1_val,
                .is_virtual = is_virtual,
                .do_not_update_pc = do_not_update_pc,
                .is_first_in_sequence = is_first_in_sequence,
                .is_compressed = is_compressed,
                .is_rd_not_zero = is_rd_not_zero,
            };
        }

        /// VirtualSignExtendWord uses SignExtendHalfWord table (Jolt table index 21)
        pub fn lookupTable() LookupTables(XLEN) {
            return .SignExtendHalfWord;
        }

        /// Lookup index is the rs1 value (the value to sign-extend)
        /// In Jolt: to_lookup_operands returns (0, x as u128 + y as u64 as u128)
        /// where (x, y) = (rs1, 0), so index = rs1
        pub fn toLookupIndex(self: Self) u128 {
            return @as(u128, self.rs1_val);
        }

        /// Compute the sign-extended result
        /// Sign-extends lower XLEN/2 bits to full XLEN bits
        pub fn computeResult(self: Self) u64 {
            const half_word_size = XLEN / 2;
            const lower_half: u64 = self.rs1_val & ((1 << half_word_size) - 1);
            const sign_bit: u64 = (lower_half >> (half_word_size - 1)) & 1;

            if (sign_bit == 1) {
                // Sign extend with 1s
                return lower_half | (((1 << half_word_size) - 1) << half_word_size);
            } else {
                // Sign extend with 0s
                return lower_half;
            }
        }

        /// Circuit flags for VirtualSignExtendWord
        /// Reference: virtual_sign_extend_word.rs circuit_flags()
        pub fn circuitFlags(self: Self) CircuitFlagSet {
            var flags = CircuitFlagSet.init();
            flags.set(.WriteLookupOutputToRD);
            flags.set(.AddOperands);
            if (self.is_virtual) flags.set(.VirtualInstruction);
            if (self.do_not_update_pc) flags.set(.DoNotUpdateUnexpandedPC);
            if (self.is_first_in_sequence) flags.set(.IsFirstInSequence);
            if (self.is_compressed) flags.set(.IsCompressed);
            return flags;
        }

        /// Instruction flags for VirtualSignExtendWord
        /// Reference: virtual_sign_extend_word.rs instruction_flags()
        pub fn instructionFlags(self: Self) InstructionFlagSet {
            var flags = InstructionFlagSet.init();
            flags.set(.LeftOperandIsRs1Value);
            if (self.is_rd_not_zero) flags.set(.IsRdNotZero);
            return flags;
        }
    };
}

/// VirtualMULI (Virtual Multiply Immediate) instruction lookup
/// Used for SLLI decomposition: SLLI rd, rs1, shamt → VirtualMULI rd, rs1, (1 << shamt)
/// Also used as the base step of SLLIW decomposition.
///
/// VirtualMULI is like MUL but with an immediate operand instead of rs2.
/// It uses the RangeCheck table (identity) to verify the result.
///
/// Reference: jolt-core/src/zkvm/instruction/virtual_muli.rs
pub fn VirtualMULILookup(comptime XLEN: comptime_int) type {
    return struct {
        const Self = @This();

        /// rs1 register value
        rs1_val: u64,
        /// Immediate value (the multiplier, e.g., 1 << shamt for SLLI)
        imm_val: u64,
        /// Whether this instruction is part of a virtual sequence (vsr.is_some())
        is_virtual: bool,
        /// Whether this instruction should not update the unexpanded PC (vsr != 0)
        do_not_update_pc: bool,
        /// Whether this is the first instruction in the sequence
        is_first_in_sequence: bool,
        /// Whether the original instruction was compressed
        is_compressed: bool,
        /// Whether the rd register is not x0
        is_rd_not_zero: bool,

        pub fn init(
            rs1_val: u64,
            imm_val: u64,
            is_virtual: bool,
            do_not_update_pc: bool,
            is_first_in_sequence: bool,
            is_compressed: bool,
            is_rd_not_zero: bool,
        ) Self {
            return Self{
                .rs1_val = rs1_val,
                .imm_val = imm_val,
                .is_virtual = is_virtual,
                .do_not_update_pc = do_not_update_pc,
                .is_first_in_sequence = is_first_in_sequence,
                .is_compressed = is_compressed,
                .is_rd_not_zero = is_rd_not_zero,
            };
        }

        /// VirtualMULI uses RangeCheck table (identity table)
        pub fn lookupTable() LookupTables(XLEN) {
            return .RangeCheck;
        }

        /// Lookup index = product (rs1 * imm), used as RangeCheck identity
        /// In Jolt: to_lookup_operands returns (0, x as u128 * y as u64 as u128)
        pub fn toLookupIndex(self: Self) u128 {
            return @as(u128, self.rs1_val) * @as(u128, self.imm_val);
        }

        /// Compute the result: wrapping multiply of rs1 and imm
        /// For XLEN=64: (rs1 as i64).wrapping_mul(imm as i64) as u64
        pub fn computeResult(self: Self) u64 {
            return self.rs1_val *% self.imm_val;
        }

        /// Circuit flags for VirtualMULI
        /// Reference: virtual_muli.rs circuit_flags()
        pub fn circuitFlags(self: Self) CircuitFlagSet {
            var flags = CircuitFlagSet.init();
            flags.set(.MultiplyOperands);
            flags.set(.WriteLookupOutputToRD);
            if (self.is_virtual) flags.set(.VirtualInstruction);
            if (self.do_not_update_pc) flags.set(.DoNotUpdateUnexpandedPC);
            if (self.is_first_in_sequence) flags.set(.IsFirstInSequence);
            if (self.is_compressed) flags.set(.IsCompressed);
            return flags;
        }

        /// Instruction flags for VirtualMULI
        /// Reference: virtual_muli.rs instruction_flags()
        pub fn instructionFlags(self: Self) InstructionFlagSet {
            var flags = InstructionFlagSet.init();
            flags.set(.LeftOperandIsRs1Value);
            flags.set(.RightOperandIsImm);
            if (self.is_rd_not_zero) flags.set(.IsRdNotZero);
            return flags;
        }
    };
}

test "virtual muli lookup" {
    // SLLI x2, x2, 16 → VirtualMULI x2, x2, 65536
    const vmuli1 = VirtualMULILookup(64).init(5, 65536, false, false, false, false, true);
    try std.testing.expectEqual(@as(u64, 5 * 65536), vmuli1.computeResult());
    try std.testing.expectEqual(@as(u128, 5 * 65536), vmuli1.toLookupIndex());

    // Test wrapping multiply (SLLI x1, x1, 63 on large value)
    const vmuli2 = VirtualMULILookup(64).init(3, @as(u64, 1) << 63, false, false, false, false, true);
    try std.testing.expectEqual(@as(u64, 3 *% (@as(u64, 1) << 63)), vmuli2.computeResult());

    // Test circuit flags (non-virtual, standalone SLLI)
    const flags1 = vmuli1.circuitFlags();
    try std.testing.expect(flags1.get(.MultiplyOperands));
    try std.testing.expect(flags1.get(.WriteLookupOutputToRD));
    try std.testing.expect(!flags1.get(.VirtualInstruction));

    // Test circuit flags (virtual, SLLIW base step)
    const vmuli3 = VirtualMULILookup(64).init(5, 4, true, true, true, false, true);
    const flags3 = vmuli3.circuitFlags();
    try std.testing.expect(flags3.get(.MultiplyOperands));
    try std.testing.expect(flags3.get(.WriteLookupOutputToRD));
    try std.testing.expect(flags3.get(.VirtualInstruction));
    try std.testing.expect(flags3.get(.DoNotUpdateUnexpandedPC));
    try std.testing.expect(flags3.get(.IsFirstInSequence));

    // Test instruction flags
    const inst_flags = vmuli1.instructionFlags();
    try std.testing.expect(inst_flags.get(.LeftOperandIsRs1Value));
    try std.testing.expect(inst_flags.get(.RightOperandIsImm));
    try std.testing.expect(inst_flags.get(.IsRdNotZero));
}

test "virtual sign extend word lookup" {
    // Test positive value (no sign extension needed)
    const vsew1 = VirtualSignExtendWordLookup(64).init(42, true, false, false, false, true);
    try std.testing.expectEqual(@as(u64, 42), vsew1.computeResult());

    // Test negative value (sign extension)
    const vsew2 = VirtualSignExtendWordLookup(64).init(0x00000000FFFFFFFF, true, false, false, false, true);
    try std.testing.expectEqual(@as(u64, 0xFFFFFFFFFFFFFFFF), vsew2.computeResult());

    // Test 0x80000000 (negative in 32-bit)
    const vsew3 = VirtualSignExtendWordLookup(64).init(0x80000000, true, false, false, false, true);
    try std.testing.expectEqual(@as(u64, 0xFFFFFFFF80000000), vsew3.computeResult());

    // Test 0x7FFFFFFF (positive in 32-bit)
    const vsew4 = VirtualSignExtendWordLookup(64).init(0x7FFFFFFF, true, false, false, false, true);
    try std.testing.expectEqual(@as(u64, 0x7FFFFFFF), vsew4.computeResult());

    // Test circuit flags
    const flags = vsew1.circuitFlags();
    try std.testing.expect(flags.get(.WriteLookupOutputToRD));
    try std.testing.expect(flags.get(.AddOperands));
    try std.testing.expect(flags.get(.VirtualInstruction));

    // Test instruction flags
    const inst_flags = vsew1.instructionFlags();
    try std.testing.expect(inst_flags.get(.LeftOperandIsRs1Value));
    try std.testing.expect(inst_flags.get(.IsRdNotZero));
}

/// VirtualSRLI (logical right shift via bitmask) lookup computation.
///
/// Reference: jolt-core/src/zkvm/instruction/virtual_srli.rs
///
/// Key properties:
/// - Lookup table: VirtualSRL (table index 26)
/// - Lookup operands: (rs1_value, bitmask_imm) interleaved
/// - Result: the right-shifted value computed using bitmask iteration
/// - Circuit flags: WriteLookupOutputToRD, VirtualInstruction (when vsr.is_some()),
///   DoNotUpdateUnexpandedPC (when vsr!=0), IsFirstInSequence, IsCompressed
/// - Instruction flags: LeftOperandIsRs1Value, RightOperandIsImm, IsRdNotZero
pub fn VirtualSRLILookup(comptime XLEN: comptime_int) type {
    return struct {
        const Self = @This();

        rs1_val: u64,
        bitmask: u64,
        is_virtual: bool,
        do_not_update_pc: bool,
        is_first_in_sequence: bool,
        is_compressed: bool,
        is_rd_not_zero: bool,

        pub fn init(
            rs1_val: u64,
            bitmask: u64,
            is_virtual: bool,
            do_not_update_pc: bool,
            is_first_in_sequence: bool,
            is_compressed: bool,
            is_rd_not_zero: bool,
        ) Self {
            return Self{
                .rs1_val = rs1_val,
                .bitmask = bitmask,
                .is_virtual = is_virtual,
                .do_not_update_pc = do_not_update_pc,
                .is_first_in_sequence = is_first_in_sequence,
                .is_compressed = is_compressed,
                .is_rd_not_zero = is_rd_not_zero,
            };
        }

        /// VirtualSRLI uses VirtualSRL table (table index 26)
        pub fn lookupTable() LookupTables(XLEN) {
            return .VirtualSRL;
        }

        /// Lookup index = interleave(rs1_val, bitmask)
        /// Jolt's to_instruction_inputs returns (rs1, bitmask_imm)
        /// The interleaving puts value bits in even positions, bitmask in odd
        pub fn toLookupIndex(self: Self) u128 {
            return interleaveBits(self.rs1_val, self.bitmask);
        }

        /// Compute the result: logical right shift using bitmask
        /// This matches Jolt's to_lookup_output:
        ///   for each bit from MSB to LSB:
        ///     result = result * (1 + y_i) + x_i * y_i
        pub fn computeResult(self: Self) u64 {
            var result: u64 = 0;
            for (0..XLEN) |i| {
                const bit_pos: u6 = @intCast(XLEN - 1 - i);
                const x_i: u64 = (self.rs1_val >> bit_pos) & 1;
                const y_i: u64 = (self.bitmask >> bit_pos) & 1;
                result = result * (1 + y_i) + x_i * y_i;
            }
            return result;
        }

        /// Circuit flags for VirtualSRLI
        /// Reference: virtual_srli.rs circuit_flags()
        pub fn circuitFlags(self: Self) CircuitFlagSet {
            var flags = CircuitFlagSet.init();
            flags.set(.WriteLookupOutputToRD);
            if (self.is_virtual) flags.set(.VirtualInstruction);
            if (self.do_not_update_pc) flags.set(.DoNotUpdateUnexpandedPC);
            if (self.is_first_in_sequence) flags.set(.IsFirstInSequence);
            if (self.is_compressed) flags.set(.IsCompressed);
            return flags;
        }

        /// Instruction flags for VirtualSRLI
        /// Reference: virtual_srli.rs instruction_flags()
        pub fn instructionFlags(self: Self) InstructionFlagSet {
            var flags = InstructionFlagSet.init();
            flags.set(.LeftOperandIsRs1Value);
            flags.set(.RightOperandIsImm);
            if (self.is_rd_not_zero) flags.set(.IsRdNotZero);
            return flags;
        }
    };
}

fn interleaveBits(x: u64, y: u64) u128 {
    var result: u128 = 0;
    for (0..64) |i| {
        const ii: u6 = @intCast(i);
        result |= @as(u128, (x >> ii) & 1) << @intCast(2 * i);
        result |= @as(u128, (y >> ii) & 1) << @intCast(2 * i + 1);
    }
    return result;
}

test "virtual srli lookup" {
    // SRLI x1, x1, 4 → VirtualSRLI with bitmask
    // shift=4, bitmask = ((1 << 60) - 1) << 4 = 0xFFFFFFFFFFFFFFF0
    const bitmask: u64 = 0xFFFFFFFFFFFFFFF0;
    const vsrli = VirtualSRLILookup(64).init(0xFF, bitmask, false, false, false, false, true);
    // 0xFF >> 4 = 0x0F
    try std.testing.expectEqual(@as(u64, 0x0F), vsrli.computeResult());

    // Test circuit flags
    const flags = vsrli.circuitFlags();
    try std.testing.expect(flags.get(.WriteLookupOutputToRD));
    try std.testing.expect(!flags.get(.VirtualInstruction));
    try std.testing.expect(!flags.get(.MultiplyOperands));
    try std.testing.expect(!flags.get(.AddOperands));

    // Test instruction flags
    const inst_flags = vsrli.instructionFlags();
    try std.testing.expect(inst_flags.get(.LeftOperandIsRs1Value));
    try std.testing.expect(inst_flags.get(.RightOperandIsImm));
    try std.testing.expect(inst_flags.get(.IsRdNotZero));
}

// ============================================================================
// Virtual Instructions for Inline Sequence Decomposition
// ============================================================================

/// VirtualAdvice instruction lookup
/// Injects an oracle-provided value (quotient or remainder) into the register file.
/// Used as the first steps of division/remainder inline sequences.
///
/// Reference: jolt-core/src/zkvm/instruction/virtual_advice.rs
///
/// Key properties:
/// - Lookup table: RangeCheck (identity)
/// - to_lookup_operands: (0, advice) - the advice value is provided by the prover
/// - Circuit flags: Advice, WriteLookupOutputToRD, VirtualInstruction (when vsr.is_some()),
///   DoNotUpdateUnexpandedPC (when vsr!=0), IsFirstInSequence, IsCompressed
/// - Instruction flags: IsRdNotZero
pub fn VirtualAdviceLookup(comptime XLEN: comptime_int) type {
    _ = XLEN;
    return struct {
        const Self = @This();

        /// The advice value (oracle-provided quotient or remainder)
        advice: u64,
        /// Whether this instruction is part of a virtual sequence (vsr.is_some())
        is_virtual: bool,
        /// Whether this instruction should not update the unexpanded PC (vsr != 0)
        do_not_update_pc: bool,
        /// Whether this is the first instruction in the sequence
        is_first_in_sequence: bool,
        /// Whether the original instruction was compressed
        is_compressed: bool,
        /// Whether the rd register is not x0
        is_rd_not_zero: bool,

        pub fn init(
            advice: u64,
            is_virtual: bool,
            do_not_update_pc: bool,
            is_first_in_sequence: bool,
            is_compressed: bool,
            is_rd_not_zero: bool,
        ) Self {
            return Self{
                .advice = advice,
                .is_virtual = is_virtual,
                .do_not_update_pc = do_not_update_pc,
                .is_first_in_sequence = is_first_in_sequence,
                .is_compressed = is_compressed,
                .is_rd_not_zero = is_rd_not_zero,
            };
        }

        /// VirtualAdvice uses RangeCheck table (identity)
        pub fn lookupTable() LookupTables(64) {
            return .RangeCheck;
        }

        /// Lookup index = advice value
        /// In Jolt: to_lookup_operands returns (0, advice as u128)
        pub fn toLookupIndex(self: Self) u128 {
            return @as(u128, self.advice);
        }

        /// Compute the result: the advice value itself
        pub fn computeResult(self: Self) u64 {
            return self.advice;
        }

        /// Circuit flags for VirtualAdvice
        pub fn circuitFlags(self: Self) CircuitFlagSet {
            var flags = CircuitFlagSet.init();
            flags.set(.Advice);
            flags.set(.WriteLookupOutputToRD);
            if (self.is_virtual) flags.set(.VirtualInstruction);
            if (self.do_not_update_pc) flags.set(.DoNotUpdateUnexpandedPC);
            if (self.is_first_in_sequence) flags.set(.IsFirstInSequence);
            if (self.is_compressed) flags.set(.IsCompressed);
            return flags;
        }

        /// Instruction flags for VirtualAdvice
        pub fn instructionFlags(self: Self) InstructionFlagSet {
            var flags = InstructionFlagSet.init();
            if (self.is_rd_not_zero) flags.set(.IsRdNotZero);
            return flags;
        }
    };
}

/// VirtualAssertEQ instruction lookup
/// Asserts that two register values are equal.
/// Used to verify intermediate computations in division/remainder inline sequences.
///
/// Reference: jolt-core/src/zkvm/instruction/virtual_assert_eq.rs
///
/// Key properties:
/// - Lookup table: Equal (table index 6)
/// - to_lookup_operands: (rs1, rs2) interleaved (default behavior)
/// - Circuit flags: Assert, VirtualInstruction (when vsr.is_some()),
///   DoNotUpdateUnexpandedPC (when vsr!=0), IsFirstInSequence, IsCompressed
/// - Instruction flags: LeftOperandIsRs1Value, RightOperandIsRs2Value
pub fn VirtualAssertEQLookup(comptime XLEN: comptime_int) type {
    _ = XLEN;
    return struct {
        const Self = @This();

        /// First register value (rs1)
        rs1_val: u64,
        /// Second register value (rs2)
        rs2_val: u64,
        /// Whether this instruction is part of a virtual sequence
        is_virtual: bool,
        /// Whether this instruction should not update the unexpanded PC
        do_not_update_pc: bool,
        /// Whether this is the first instruction in the sequence
        is_first_in_sequence: bool,
        /// Whether the original instruction was compressed
        is_compressed: bool,

        pub fn init(
            rs1_val: u64,
            rs2_val: u64,
            is_virtual: bool,
            do_not_update_pc: bool,
            is_first_in_sequence: bool,
            is_compressed: bool,
        ) Self {
            return Self{
                .rs1_val = rs1_val,
                .rs2_val = rs2_val,
                .is_virtual = is_virtual,
                .do_not_update_pc = do_not_update_pc,
                .is_first_in_sequence = is_first_in_sequence,
                .is_compressed = is_compressed,
            };
        }

        /// VirtualAssertEQ uses Equal table (table index 6)
        pub fn lookupTable() LookupTables(64) {
            return .Equal;
        }

        /// Lookup index = interleaved(rs1, rs2)
        pub fn toLookupIndex(self: Self) u128 {
            return interleaveBits(self.rs1_val, self.rs2_val);
        }

        /// Compute the result: 1 if equal, 0 if not
        pub fn computeResult(self: Self) u64 {
            return if (self.rs1_val == self.rs2_val) 1 else 0;
        }

        /// Circuit flags for VirtualAssertEQ
        pub fn circuitFlags(self: Self) CircuitFlagSet {
            var flags = CircuitFlagSet.init();
            flags.set(.Assert);
            if (self.is_virtual) flags.set(.VirtualInstruction);
            if (self.do_not_update_pc) flags.set(.DoNotUpdateUnexpandedPC);
            if (self.is_first_in_sequence) flags.set(.IsFirstInSequence);
            if (self.is_compressed) flags.set(.IsCompressed);
            return flags;
        }

        /// Instruction flags for VirtualAssertEQ
        pub fn instructionFlags(_: Self) InstructionFlagSet {
            var flags = InstructionFlagSet.init();
            flags.set(.LeftOperandIsRs1Value);
            flags.set(.RightOperandIsRs2Value);
            return flags;
        }
    };
}

/// VirtualZeroExtendWord instruction lookup
/// Zero-extends lower XLEN/2 bits (masks to 32 bits for XLEN=64).
/// Used to prepare unsigned operands in division/remainder inline sequences.
///
/// Reference: jolt-core/src/zkvm/instruction/virtual_zero_extend_word.rs
///
/// Key properties:
/// - Lookup table: LowerHalfWord (table index 20)
/// - to_lookup_operands: (0, rs1 + 0) = (0, rs1) via AddOperands mode
/// - Circuit flags: WriteLookupOutputToRD, AddOperands, VirtualInstruction (when vsr.is_some()),
///   DoNotUpdateUnexpandedPC (when vsr!=0), IsFirstInSequence, IsCompressed
/// - Instruction flags: LeftOperandIsRs1Value, IsRdNotZero
pub fn VirtualZeroExtendWordLookup(comptime XLEN: comptime_int) type {
    return struct {
        const Self = @This();

        /// The value to zero-extend (rs1)
        rs1_val: u64,
        /// Whether this instruction is part of a virtual sequence
        is_virtual: bool,
        /// Whether this instruction should not update the unexpanded PC
        do_not_update_pc: bool,
        /// Whether this is the first instruction in the sequence
        is_first_in_sequence: bool,
        /// Whether the original instruction was compressed
        is_compressed: bool,
        /// Whether the rd register is not x0
        is_rd_not_zero: bool,

        pub fn init(
            rs1_val: u64,
            is_virtual: bool,
            do_not_update_pc: bool,
            is_first_in_sequence: bool,
            is_compressed: bool,
            is_rd_not_zero: bool,
        ) Self {
            return Self{
                .rs1_val = rs1_val,
                .is_virtual = is_virtual,
                .do_not_update_pc = do_not_update_pc,
                .is_first_in_sequence = is_first_in_sequence,
                .is_compressed = is_compressed,
                .is_rd_not_zero = is_rd_not_zero,
            };
        }

        /// VirtualZeroExtendWord uses LowerHalfWord table (table index 20)
        pub fn lookupTable() LookupTables(XLEN) {
            return .LowerHalfWord;
        }

        /// Lookup index = rs1 value (via AddOperands: (0, rs1 + 0))
        pub fn toLookupIndex(self: Self) u128 {
            return @as(u128, self.rs1_val);
        }

        /// Compute the result: lower XLEN/2 bits (zero-extended)
        pub fn computeResult(self: Self) u64 {
            const half_word_size = XLEN / 2;
            const mask: u64 = (@as(u64, 1) << half_word_size) - 1;
            return self.rs1_val & mask;
        }

        /// Circuit flags for VirtualZeroExtendWord
        pub fn circuitFlags(self: Self) CircuitFlagSet {
            var flags = CircuitFlagSet.init();
            flags.set(.WriteLookupOutputToRD);
            flags.set(.AddOperands);
            if (self.is_virtual) flags.set(.VirtualInstruction);
            if (self.do_not_update_pc) flags.set(.DoNotUpdateUnexpandedPC);
            if (self.is_first_in_sequence) flags.set(.IsFirstInSequence);
            if (self.is_compressed) flags.set(.IsCompressed);
            return flags;
        }

        /// Instruction flags for VirtualZeroExtendWord
        pub fn instructionFlags(self: Self) InstructionFlagSet {
            var flags = InstructionFlagSet.init();
            flags.set(.LeftOperandIsRs1Value);
            if (self.is_rd_not_zero) flags.set(.IsRdNotZero);
            return flags;
        }
    };
}

/// VirtualAssertValidUnsignedRemainder instruction lookup
/// Asserts that remainder < divisor (or divisor == 0).
/// Used in unsigned division/remainder inline sequences to validate the remainder.
///
/// Reference: jolt-core/src/zkvm/instruction/virtual_assert_valid_unsigned_remainder.rs
///
/// Key properties:
/// - Lookup table: ValidUnsignedRemainder (table index 16)
/// - to_lookup_operands: (rs1, rs2) interleaved (default behavior)
/// - Circuit flags: Assert, VirtualInstruction (when vsr.is_some()),
///   DoNotUpdateUnexpandedPC (when vsr!=0), IsFirstInSequence, IsCompressed
/// - Instruction flags: LeftOperandIsRs1Value, RightOperandIsRs2Value
pub fn VirtualAssertValidUnsignedRemainderLookup(comptime XLEN: comptime_int) type {
    _ = XLEN;
    return struct {
        const Self = @This();

        /// Remainder value (rs1)
        rs1_val: u64,
        /// Divisor value (rs2)
        rs2_val: u64,
        /// Whether this instruction is part of a virtual sequence
        is_virtual: bool,
        /// Whether this instruction should not update the unexpanded PC
        do_not_update_pc: bool,
        /// Whether this is the first instruction in the sequence
        is_first_in_sequence: bool,
        /// Whether the original instruction was compressed
        is_compressed: bool,

        pub fn init(
            rs1_val: u64,
            rs2_val: u64,
            is_virtual: bool,
            do_not_update_pc: bool,
            is_first_in_sequence: bool,
            is_compressed: bool,
        ) Self {
            return Self{
                .rs1_val = rs1_val,
                .rs2_val = rs2_val,
                .is_virtual = is_virtual,
                .do_not_update_pc = do_not_update_pc,
                .is_first_in_sequence = is_first_in_sequence,
                .is_compressed = is_compressed,
            };
        }

        /// VirtualAssertValidUnsignedRemainder uses ValidUnsignedRemainder table (index 16)
        pub fn lookupTable() LookupTables(64) {
            return .ValidUnsignedRemainder;
        }

        /// Lookup index = interleaved(remainder, divisor)
        pub fn toLookupIndex(self: Self) u128 {
            return interleaveBits(self.rs1_val, self.rs2_val);
        }

        /// Compute the result: 1 if valid (remainder < divisor or divisor == 0), 0 otherwise
        pub fn computeResult(self: Self) u64 {
            if (self.rs2_val == 0) return 1; // division by zero case
            return if (self.rs1_val < self.rs2_val) 1 else 0;
        }

        /// Circuit flags for VirtualAssertValidUnsignedRemainder
        pub fn circuitFlags(self: Self) CircuitFlagSet {
            var flags = CircuitFlagSet.init();
            flags.set(.Assert);
            if (self.is_virtual) flags.set(.VirtualInstruction);
            if (self.do_not_update_pc) flags.set(.DoNotUpdateUnexpandedPC);
            if (self.is_first_in_sequence) flags.set(.IsFirstInSequence);
            if (self.is_compressed) flags.set(.IsCompressed);
            return flags;
        }

        /// Instruction flags for VirtualAssertValidUnsignedRemainder
        pub fn instructionFlags(_: Self) InstructionFlagSet {
            var flags = InstructionFlagSet.init();
            flags.set(.LeftOperandIsRs1Value);
            flags.set(.RightOperandIsRs2Value);
            return flags;
        }
    };
}

// ============================================================================
// Jolt-Inline Instruction Lookups
// ============================================================================

/// ANDN instruction lookup: rd = rs1 & ~rs2
/// Uses Andn lookup table (table index 3), interleaved operands.
pub fn AndnLookup(comptime XLEN: comptime_int) type {
    return struct {
        const Self = @This();
        rs1_val: u64,
        rs2_val: u64,
        is_virtual: bool,
        do_not_update_pc: bool,
        is_first_in_sequence: bool,
        is_compressed: bool,
        is_rd_not_zero: bool,

        pub fn init(rs1_val: u64, rs2_val: u64, is_virtual: bool, do_not_update_pc: bool, is_first_in_sequence: bool, is_compressed: bool, is_rd_not_zero: bool) Self {
            return Self{ .rs1_val = rs1_val, .rs2_val = rs2_val, .is_virtual = is_virtual, .do_not_update_pc = do_not_update_pc, .is_first_in_sequence = is_first_in_sequence, .is_compressed = is_compressed, .is_rd_not_zero = is_rd_not_zero };
        }

        pub fn lookupTable() LookupTables(XLEN) {
            return .Andn;
        }

        pub fn toLookupIndex(self: Self) u128 {
            return interleaveBits(self.rs1_val, self.rs2_val);
        }

        pub fn computeResult(self: Self) u64 {
            return self.rs1_val & ~self.rs2_val;
        }

        pub fn circuitFlags(self: Self) CircuitFlagSet {
            var flags = CircuitFlagSet.init();
            flags.set(.WriteLookupOutputToRD);
            if (self.is_virtual) flags.set(.VirtualInstruction);
            if (self.do_not_update_pc) flags.set(.DoNotUpdateUnexpandedPC);
            if (self.is_first_in_sequence) flags.set(.IsFirstInSequence);
            if (self.is_compressed) flags.set(.IsCompressed);
            return flags;
        }

        pub fn instructionFlags(self: Self) InstructionFlagSet {
            var flags = InstructionFlagSet.init();
            flags.set(.LeftOperandIsRs1Value);
            flags.set(.RightOperandIsRs2Value);
            if (self.is_rd_not_zero) flags.set(.IsRdNotZero);
            return flags;
        }
    };
}

/// VirtualROTRI instruction lookup: rd = rotate_right(rs1, trailing_zeros(bitmask))
/// Uses VirtualROTR lookup table (table index 27), interleaved operands.
pub fn VirtualROTRILookup(comptime XLEN: comptime_int) type {
    return struct {
        const Self = @This();
        rs1_val: u64,
        bitmask: u64,
        is_virtual: bool,
        do_not_update_pc: bool,
        is_first_in_sequence: bool,
        is_compressed: bool,
        is_rd_not_zero: bool,

        pub fn init(rs1_val: u64, bitmask: u64, is_virtual: bool, do_not_update_pc: bool, is_first_in_sequence: bool, is_compressed: bool, is_rd_not_zero: bool) Self {
            return Self{ .rs1_val = rs1_val, .bitmask = bitmask, .is_virtual = is_virtual, .do_not_update_pc = do_not_update_pc, .is_first_in_sequence = is_first_in_sequence, .is_compressed = is_compressed, .is_rd_not_zero = is_rd_not_zero };
        }

        pub fn lookupTable() LookupTables(XLEN) {
            return .VirtualROTR;
        }

        pub fn toLookupIndex(self: Self) u128 {
            return interleaveBits(self.rs1_val, self.bitmask);
        }

        pub fn computeResult(self: Self) u64 {
            const rotation: u6 = if (self.bitmask == 0) 0 else @intCast(@ctz(self.bitmask));
            return std.math.rotr(u64, self.rs1_val, rotation);
        }

        pub fn circuitFlags(self: Self) CircuitFlagSet {
            var flags = CircuitFlagSet.init();
            flags.set(.WriteLookupOutputToRD);
            if (self.is_virtual) flags.set(.VirtualInstruction);
            if (self.do_not_update_pc) flags.set(.DoNotUpdateUnexpandedPC);
            if (self.is_first_in_sequence) flags.set(.IsFirstInSequence);
            if (self.is_compressed) flags.set(.IsCompressed);
            return flags;
        }

        pub fn instructionFlags(self: Self) InstructionFlagSet {
            var flags = InstructionFlagSet.init();
            flags.set(.LeftOperandIsRs1Value);
            flags.set(.RightOperandIsImm);
            if (self.is_rd_not_zero) flags.set(.IsRdNotZero);
            return flags;
        }
    };
}

/// VirtualROTRIW instruction lookup: rd = rotate_right_32(rs1, trailing_zeros(bitmask))
/// Uses VirtualROTRW lookup table (table index 28), interleaved operands.
pub fn VirtualROTRIWLookup(comptime XLEN: comptime_int) type {
    return struct {
        const Self = @This();
        rs1_val: u64,
        bitmask: u64,
        is_virtual: bool,
        do_not_update_pc: bool,
        is_first_in_sequence: bool,
        is_compressed: bool,
        is_rd_not_zero: bool,

        pub fn init(rs1_val: u64, bitmask: u64, is_virtual: bool, do_not_update_pc: bool, is_first_in_sequence: bool, is_compressed: bool, is_rd_not_zero: bool) Self {
            return Self{ .rs1_val = rs1_val, .bitmask = bitmask, .is_virtual = is_virtual, .do_not_update_pc = do_not_update_pc, .is_first_in_sequence = is_first_in_sequence, .is_compressed = is_compressed, .is_rd_not_zero = is_rd_not_zero };
        }

        pub fn lookupTable() LookupTables(XLEN) {
            return .VirtualROTRW;
        }

        pub fn toLookupIndex(self: Self) u128 {
            return interleaveBits(self.rs1_val, self.bitmask);
        }

        pub fn computeResult(self: Self) u64 {
            const val32: u32 = @truncate(self.rs1_val);
            const bm32: u32 = @truncate(self.bitmask);
            const rotation: u5 = if (bm32 == 0) 0 else @intCast(@min(@as(u6, @intCast(@ctz(bm32))), 31));
            return @as(u64, std.math.rotr(u32, val32, rotation));
        }

        pub fn circuitFlags(self: Self) CircuitFlagSet {
            var flags = CircuitFlagSet.init();
            flags.set(.WriteLookupOutputToRD);
            if (self.is_virtual) flags.set(.VirtualInstruction);
            if (self.do_not_update_pc) flags.set(.DoNotUpdateUnexpandedPC);
            if (self.is_first_in_sequence) flags.set(.IsFirstInSequence);
            if (self.is_compressed) flags.set(.IsCompressed);
            return flags;
        }

        pub fn instructionFlags(self: Self) InstructionFlagSet {
            var flags = InstructionFlagSet.init();
            flags.set(.LeftOperandIsRs1Value);
            flags.set(.RightOperandIsImm);
            if (self.is_rd_not_zero) flags.set(.IsRdNotZero);
            return flags;
        }
    };
}
