//! R1CS Witness Type Definitions
//!
//! Core types used by the R1CS constraint system:
//! - R1CSInputIndex: witness variable indexing (must match Jolt's inputs.rs)
//! - Term / LinearCombination / LC: linear combination algebra
//! - UniformConstraint: equality-conditional constraint form
//!
//! Extracted from constraints.zig for modularity.

const std = @import("std");

/// Index into the witness vector for R1CS inputs
///
/// IMPORTANT: This order MUST match Jolt's ALL_R1CS_INPUTS array in inputs.rs
/// The order is critical for correct R1CS evaluation and claim matching.
///
/// Reference: jolt-core/src/zkvm/r1cs/inputs.rs
pub const R1CSInputIndex = enum(u8) {
    // Matches upstream Jolt's ALL_R1CS_INPUTS order exactly (35 inputs):
    LeftInstructionInput = 0,
    RightInstructionInput = 1,
    Product = 2,
    ShouldBranch = 3,
    PC = 4,
    UnexpandedPC = 5,
    Imm = 6,
    RamAddress = 7,
    Rs1Value = 8,
    Rs2Value = 9,
    RdWriteValue = 10,
    RamReadValue = 11,
    RamWriteValue = 12,
    LeftLookupOperand = 13,
    RightLookupOperand = 14,
    NextUnexpandedPC = 15,
    NextPC = 16,
    NextIsVirtual = 17,
    NextIsFirstInSequence = 18,
    LookupOutput = 19,
    ShouldJump = 20,
    // OpFlags in order matching CircuitFlags enum (14 flags)
    FlagAddOperands = 21,
    FlagSubtractOperands = 22,
    FlagMultiplyOperands = 23,
    FlagLoad = 24,
    FlagStore = 25,
    FlagJump = 26,
    FlagWriteLookupOutputToRD = 27,
    FlagVirtualInstruction = 28,
    FlagAssert = 29,
    FlagDoNotUpdateUnexpandedPC = 30,
    FlagAdvice = 31,
    FlagIsCompressed = 32,
    FlagIsFirstInSequence = 33,
    FlagIsLastInSequence = 34,
    // Additional flags for product virtualization factor polynomials
    // These are derived from instruction fields and needed for Stage 2 factor evaluation
    FlagIsRdNotZero = 35, // 1 if rd register index != 0
    FlagBranch = 36, // 1 if instruction opcode == 0x63 (branch)
    FlagIsNoop = 37, // 1 if this is a noop instruction
    // InstructionFlags for Stage 3 InstructionInput sumcheck
    // These determine which operand values are used for left/right instruction inputs
    FlagLeftOperandIsRs1 = 38, // 1 if left instruction input is rs1 value
    FlagLeftOperandIsPC = 39, // 1 if left instruction input is PC (JAL, AUIPC)
    FlagRightOperandIsRs2 = 40, // 1 if right instruction input is rs2 value (R-type)
    FlagRightOperandIsImm = 41, // 1 if right instruction input is immediate (I-type, etc.)

    pub const NUM_INPUTS = 42;

    pub fn toIndex(self: R1CSInputIndex) usize {
        return @intFromEnum(self);
    }
};

/// Linear combination term
pub const Term = struct {
    input_index: R1CSInputIndex,
    coeff: i128,
};

/// Linear combination of witness variables
pub fn LinearCombination(comptime max_terms: usize) type {
    return struct {
        const Self = @This();

        terms: [max_terms]Term,
        len: usize,
        constant: i128,

        pub fn zero() Self {
            return Self{
                .terms = undefined,
                .len = 0,
                .constant = 0,
            };
        }

        pub fn one() Self {
            return Self{
                .terms = undefined,
                .len = 0,
                .constant = 1,
            };
        }

        pub fn fromConstant(c: i128) Self {
            return Self{
                .terms = undefined,
                .len = 0,
                .constant = c,
            };
        }

        pub fn fromInput(index: R1CSInputIndex) Self {
            var lc = Self{
                .terms = undefined,
                .len = 1,
                .constant = 0,
            };
            lc.terms[0] = .{ .input_index = index, .coeff = 1 };
            return lc;
        }

        pub fn fromInputScaled(index: R1CSInputIndex, coeff: i128) Self {
            var lc = Self{
                .terms = undefined,
                .len = 1,
                .constant = 0,
            };
            lc.terms[0] = .{ .input_index = index, .coeff = coeff };
            return lc;
        }

        /// Helper: convert i128 to field element (handles values > 2^64)
        fn i128ToField(comptime F: type, val: i128) F {
            if (val >= 0) {
                const v: u128 = @intCast(val);
                if (v <= 0xFFFFFFFFFFFFFFFF) {
                    return F.fromU64(@intCast(v));
                } else {
                    // Value > 2^64, use bytes representation
                    var bytes: [16]u8 = undefined;
                    std.mem.writeInt(u128, &bytes, v, .little);
                    return F.fromBytes(&bytes);
                }
            } else {
                const neg_v: u128 = @intCast(-val);
                if (neg_v <= 0xFFFFFFFFFFFFFFFF) {
                    return F.zero().sub(F.fromU64(@intCast(neg_v)));
                } else {
                    // Value > 2^64, use bytes representation
                    var bytes: [16]u8 = undefined;
                    std.mem.writeInt(u128, &bytes, neg_v, .little);
                    return F.zero().sub(F.fromBytes(&bytes));
                }
            }
        }

        /// Evaluate the linear combination given witness values
        pub fn evaluate(self: Self, comptime F: type, witness: []const F) F {
            var result = i128ToField(F, self.constant);

            for (self.terms[0..self.len]) |term| {
                const val = witness[term.input_index.toIndex()];
                const coeff_field = i128ToField(F, if (term.coeff >= 0) term.coeff else -term.coeff);
                const scaled = val.mul(coeff_field);
                if (term.coeff >= 0) {
                    result = result.add(scaled);
                } else {
                    result = result.sub(scaled);
                }
            }

            return result;
        }

        /// Evaluate the linear combination given z values (MLE evaluations)
        /// This is the same as evaluate() but clarifies the intent when used
        /// with MLE evaluations instead of actual witness values.
        pub fn evaluateWithConstant(self: Self, comptime F: type, z: []const F) F {
            return self.evaluate(F, z);
        }
    };
}

/// Type alias for linear combinations with up to 5 terms
pub const LC = LinearCombination(5);

/// R1CS constraint in equality-conditional form: a * b = 0
/// Where b = (left - right), so the constraint is: condition * (left - right) = 0
pub const UniformConstraint = struct {
    /// Guard/condition (typically a boolean flag or sum of flags)
    condition: LC,
    /// Left side of equality
    left: LC,
    /// Right side of equality
    right: LC,

    /// Evaluate constraint: returns condition * (left - right)
    /// Should be zero if constraint is satisfied
    pub fn evaluate(self: UniformConstraint, comptime F: type, witness: []const F) F {
        const cond = self.condition.evaluate(F, witness);
        const l = self.left.evaluate(F, witness);
        const r = self.right.evaluate(F, witness);
        return cond.mul(l.sub(r));
    }

    /// Check if constraint is satisfied (result should be zero)
    pub fn isSatisfied(self: UniformConstraint, comptime F: type, witness: []const F) bool {
        const result = self.evaluate(F, witness);
        return result.eql(F.zero());
    }
};
