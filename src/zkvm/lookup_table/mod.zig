//! Jolt Lookup Tables
//!
//! This module implements the lookup tables used in Jolt for instruction verification.
//! Lookup arguments are THE core technique that makes Jolt efficient.
//!
//! Each lookup table is defined by:
//! 1. materializeEntry(index) - Returns the table entry at a given index
//! 2. evaluateMLE(r) - Evaluates the multilinear extension at point r
//!
//! The key insight from the Jolt papers is that many RISC-V operations can be
//! verified by looking up precomputed values in small tables, rather than computing
//! them via expensive arithmetic circuits.

const std = @import("std");
const Allocator = std.mem.Allocator;
const field = @import("../../field/mod.zig");
const BN254Scalar = field.BN254Scalar;

// Export prefix-suffix decomposition modules
pub const prefixes = @import("prefixes.zig");
pub const identity_poly = @import("identity_poly.zig");
pub const suffixes = @import("suffixes.zig");
pub const prefix_suffix_prover = @import("prefix_suffix_prover.zig");

// Re-export commonly used types
pub const Prefixes = prefixes.Prefixes;
pub const PrefixCheckpoint = prefixes.PrefixCheckpoint;
pub const PrefixCheckpoints = prefixes.PrefixCheckpoints;
pub const LookupBits = prefixes.LookupBits;
pub const IdentityPolynomial = identity_poly.IdentityPolynomial;
pub const OperandPolynomial = identity_poly.OperandPolynomial;
pub const OperandSide = identity_poly.OperandSide;
pub const BindingOrder = identity_poly.BindingOrder;
pub const Suffixes = suffixes.Suffixes;
pub const suffixMle = suffixes.suffixMle;
pub const tableSuffixes = suffixes.tableSuffixes;
pub const AllSuffixPolys = prefix_suffix_prover.AllSuffixPolys;
pub const PrefixCheckpointsState = prefix_suffix_prover.PrefixCheckpointsState;
pub const proverMsgReadChecking = prefix_suffix_prover.proverMsgReadChecking;
pub const RafDecomposition = prefix_suffix_prover.RafDecomposition;
pub const RafPolyType = prefix_suffix_prover.RafPolyType;
pub const initQRaf = prefix_suffix_prover.initQRaf;
pub const proverMsgRaf = prefix_suffix_prover.proverMsgRaf;
pub const ExpandingTable = prefix_suffix_prover.ExpandingTable;
pub const condenseUEvals = prefix_suffix_prover.condenseUEvals;
pub const computeTableValuesAtRAddress = prefix_suffix_prover.computeTableValuesAtRAddress;
pub const NUM_TABLES = prefix_suffix_prover.NUM_TABLES;

/// Utility function to uninterleave bits from an interleaved index.
/// For a 2*XLEN bit index where bits are interleaved as y[0], x[0], y[1], x[1], ...,
/// returns (x, y) where each is an XLEN-bit value.
///
/// Note: This matches Jolt's convention where y bits are at even positions and x bits at odd.
pub fn uninterleaveBits(index: u128) struct { x: u64, y: u64 } {
    // Use Jolt's efficient bit manipulation algorithm
    // x comes from odd bit positions, y from even
    var x_bits: u128 = (index >> 1) & 0x5555_5555_5555_5555_5555_5555_5555_5555;
    var y_bits: u128 = index & 0x5555_5555_5555_5555_5555_5555_5555_5555;

    // Compact x bits into lower half
    x_bits = (x_bits | (x_bits >> 1)) & 0x3333_3333_3333_3333_3333_3333_3333_3333;
    x_bits = (x_bits | (x_bits >> 2)) & 0x0F0F_0F0F_0F0F_0F0F_0F0F_0F0F_0F0F_0F0F;
    x_bits = (x_bits | (x_bits >> 4)) & 0x00FF_00FF_00FF_00FF_00FF_00FF_00FF_00FF;
    x_bits = (x_bits | (x_bits >> 8)) & 0x0000_FFFF_0000_FFFF_0000_FFFF_0000_FFFF;
    x_bits = (x_bits | (x_bits >> 16)) & 0x0000_0000_FFFF_FFFF_0000_0000_FFFF_FFFF;
    x_bits = (x_bits | (x_bits >> 32)) & 0x0000_0000_0000_0000_FFFF_FFFF_FFFF_FFFF;

    // Compact y bits into lower half
    y_bits = (y_bits | (y_bits >> 1)) & 0x3333_3333_3333_3333_3333_3333_3333_3333;
    y_bits = (y_bits | (y_bits >> 2)) & 0x0F0F_0F0F_0F0F_0F0F_0F0F_0F0F_0F0F_0F0F;
    y_bits = (y_bits | (y_bits >> 4)) & 0x00FF_00FF_00FF_00FF_00FF_00FF_00FF_00FF;
    y_bits = (y_bits | (y_bits >> 8)) & 0x0000_FFFF_0000_FFFF_0000_FFFF_0000_FFFF;
    y_bits = (y_bits | (y_bits >> 16)) & 0x0000_0000_FFFF_FFFF_0000_0000_FFFF_FFFF;
    y_bits = (y_bits | (y_bits >> 32)) & 0x0000_0000_0000_0000_FFFF_FFFF_FFFF_FFFF;

    return .{ .x = @truncate(x_bits), .y = @truncate(y_bits) };
}

/// Interleave bits of x and y into a single index.
/// Produces a value where y bits are at even positions and x bits at odd positions.
/// This matches Jolt's convention: result = (spread(x) << 1) | spread(y)
pub fn interleaveBits(x: u64, y: u64) u128 {
    // Use Jolt's efficient bit spreading algorithm
    // Spread x_bits to odd positions
    var x_bits: u128 = @as(u128, x);
    x_bits = (x_bits | (x_bits << 32)) & 0x0000_0000_FFFF_FFFF_0000_0000_FFFF_FFFF;
    x_bits = (x_bits | (x_bits << 16)) & 0x0000_FFFF_0000_FFFF_0000_FFFF_0000_FFFF;
    x_bits = (x_bits | (x_bits << 8)) & 0x00FF_00FF_00FF_00FF_00FF_00FF_00FF_00FF;
    x_bits = (x_bits | (x_bits << 4)) & 0x0F0F_0F0F_0F0F_0F0F_0F0F_0F0F_0F0F_0F0F;
    x_bits = (x_bits | (x_bits << 2)) & 0x3333_3333_3333_3333_3333_3333_3333_3333;
    x_bits = (x_bits | (x_bits << 1)) & 0x5555_5555_5555_5555_5555_5555_5555_5555;

    // Spread y_bits to even positions
    var y_bits: u128 = @as(u128, y);
    y_bits = (y_bits | (y_bits << 32)) & 0x0000_0000_FFFF_FFFF_0000_0000_FFFF_FFFF;
    y_bits = (y_bits | (y_bits << 16)) & 0x0000_FFFF_0000_FFFF_0000_FFFF_0000_FFFF;
    y_bits = (y_bits | (y_bits << 8)) & 0x00FF_00FF_00FF_00FF_00FF_00FF_00FF_00FF;
    y_bits = (y_bits | (y_bits << 4)) & 0x0F0F_0F0F_0F0F_0F0F_0F0F_0F0F_0F0F_0F0F;
    y_bits = (y_bits | (y_bits << 2)) & 0x3333_3333_3333_3333_3333_3333_3333_3333;
    y_bits = (y_bits | (y_bits << 1)) & 0x5555_5555_5555_5555_5555_5555_5555_5555;

    return (x_bits << 1) | y_bits;
}

/// The JoltLookupTable interface for Zig.
/// All lookup tables must implement these methods.
///
/// Usage:
/// ```
/// const MyTable = LookupTable(BN254Scalar, 64);
/// const entry = MyTable.RangeCheck.materializeEntry(42);
/// const mle_eval = MyTable.RangeCheck.evaluateMLE(&r);
/// ```
pub fn LookupTable(comptime F: type, comptime XLEN: comptime_int) type {
    return struct {
        const Self = @This();

        /// RangeCheck: Verifies values are in range [0, 2^XLEN)
        /// materializeEntry(index) = index (mod 2^XLEN)
        pub const RangeCheck = struct {
            /// Materialize the entry at the given index
            pub fn materializeEntry(index: u128) u64 {
                if (XLEN == 64) {
                    return @truncate(index);
                } else {
                    return @truncate(index % (@as(u128, 1) << XLEN));
                }
            }

            /// Evaluate the MLE at point r
            /// For range check, the MLE is: sum_{i=0}^{XLEN-1} 2^{XLEN-1-i} * r[XLEN + i]
            /// where r is of length 2*XLEN
            pub fn evaluateMLE(r: []const F) F {
                std.debug.assert(r.len == 2 * XLEN);

                var result = F.zero();
                inline for (0..XLEN) |i| {
                    const shift: u6 = XLEN - 1 - i;
                    const coeff = F.fromU64(@as(u64, 1) << shift);
                    result = result.add(coeff.mul(r[XLEN + i]));
                }
                return result;
            }

            /// Materialize the entire table (for testing)
            pub fn materialize(allocator: Allocator) ![]u64 {
                const size = @as(usize, 1) << (2 * @min(XLEN, 8));
                const table = try allocator.alloc(u64, size);
                for (0..size) |i| {
                    table[i] = materializeEntry(@intCast(i));
                }
                return table;
            }
        };

        /// And: Bitwise AND of two operands
        /// materializeEntry(interleaved(x, y)) = x & y
        pub const And = struct {
            /// Materialize the entry at the given index
            pub fn materializeEntry(index: u128) u64 {
                const bits = uninterleaveBits(index);
                return bits.x & bits.y;
            }

            /// Evaluate the MLE at point r
            /// For AND, the MLE is: sum_{i=0}^{XLEN-1} 2^{XLEN-1-i} * r[2*i] * r[2*i+1]
            pub fn evaluateMLE(r: []const F) F {
                std.debug.assert(r.len == 2 * XLEN);

                var result = F.zero();
                inline for (0..XLEN) |i| {
                    const shift: u6 = XLEN - 1 - i;
                    const coeff = F.fromU64(@as(u64, 1) << shift);
                    const x_i = r[2 * i];
                    const y_i = r[2 * i + 1];
                    result = result.add(coeff.mul(x_i.mul(y_i)));
                }
                return result;
            }

            /// Materialize the entire table (for testing)
            pub fn materialize(allocator: Allocator) ![]u64 {
                const size = @as(usize, 1) << (2 * @min(XLEN, 8));
                const table = try allocator.alloc(u64, size);
                for (0..size) |i| {
                    table[i] = materializeEntry(@intCast(i));
                }
                return table;
            }
        };

        /// Or: Bitwise OR of two operands
        /// materializeEntry(interleaved(x, y)) = x | y
        pub const Or = struct {
            /// Materialize the entry at the given index
            pub fn materializeEntry(index: u128) u64 {
                const bits = uninterleaveBits(index);
                return bits.x | bits.y;
            }

            /// Evaluate the MLE at point r
            /// For OR: x | y = x + y - x*y (in binary)
            /// MLE is: sum_{i=0}^{XLEN-1} 2^{XLEN-1-i} * (r[2*i] + r[2*i+1] - r[2*i]*r[2*i+1])
            pub fn evaluateMLE(r: []const F) F {
                std.debug.assert(r.len == 2 * XLEN);

                var result = F.zero();
                const one = F.one();

                inline for (0..XLEN) |i| {
                    const shift: u6 = XLEN - 1 - i;
                    const coeff = F.fromU64(@as(u64, 1) << shift);
                    const x_i = r[2 * i];
                    const y_i = r[2 * i + 1];
                    // x | y = x + y - x*y = 1 - (1-x)(1-y)
                    const one_minus_x = one.sub(x_i);
                    const one_minus_y = one.sub(y_i);
                    const or_value = one.sub(one_minus_x.mul(one_minus_y));
                    result = result.add(coeff.mul(or_value));
                }
                return result;
            }

            /// Materialize the entire table (for testing)
            pub fn materialize(allocator: Allocator) ![]u64 {
                const size = @as(usize, 1) << (2 * @min(XLEN, 8));
                const table = try allocator.alloc(u64, size);
                for (0..size) |i| {
                    table[i] = materializeEntry(@intCast(i));
                }
                return table;
            }
        };

        /// Xor: Bitwise XOR of two operands
        /// materializeEntry(interleaved(x, y)) = x ^ y
        pub const Xor = struct {
            /// Materialize the entry at the given index
            pub fn materializeEntry(index: u128) u64 {
                const bits = uninterleaveBits(index);
                return bits.x ^ bits.y;
            }

            /// Evaluate the MLE at point r
            /// For XOR: x ^ y = x(1-y) + y(1-x) = x + y - 2xy
            /// MLE is: sum_{i=0}^{XLEN-1} 2^{XLEN-1-i} * (r[2*i] + r[2*i+1] - 2*r[2*i]*r[2*i+1])
            pub fn evaluateMLE(r: []const F) F {
                std.debug.assert(r.len == 2 * XLEN);

                var result = F.zero();
                const one = F.one();

                inline for (0..XLEN) |i| {
                    const shift: u6 = XLEN - 1 - i;
                    const coeff = F.fromU64(@as(u64, 1) << shift);
                    const x_i = r[2 * i];
                    const y_i = r[2 * i + 1];
                    // x ^ y = x(1-y) + (1-x)y
                    const one_minus_x = one.sub(x_i);
                    const one_minus_y = one.sub(y_i);
                    const xor_value = one_minus_x.mul(y_i).add(x_i.mul(one_minus_y));
                    result = result.add(coeff.mul(xor_value));
                }
                return result;
            }

            /// Materialize the entire table (for testing)
            pub fn materialize(allocator: Allocator) ![]u64 {
                const size = @as(usize, 1) << (2 * @min(XLEN, 8));
                const table = try allocator.alloc(u64, size);
                for (0..size) |i| {
                    table[i] = materializeEntry(@intCast(i));
                }
                return table;
            }
        };

        /// Equal: Checks if two operands are equal
        /// materializeEntry(interleaved(x, y)) = 1 if x == y, else 0
        pub const Equal = struct {
            /// Materialize the entry at the given index
            pub fn materializeEntry(index: u128) u64 {
                const bits = uninterleaveBits(index);
                return if (bits.x == bits.y) 1 else 0;
            }

            /// Evaluate the MLE at point r
            /// For equality, we want prod_{i=0}^{XLEN-1} (r[2*i] * r[2*i+1] + (1-r[2*i])*(1-r[2*i+1]))
            /// This is the MLE of the indicator for x == y
            pub fn evaluateMLE(r: []const F) F {
                std.debug.assert(r.len == 2 * XLEN);

                var result = F.one();
                const one = F.one();

                inline for (0..XLEN) |i| {
                    const x_i = r[2 * i];
                    const y_i = r[2 * i + 1];
                    // For bit i: x_i == y_i iff x_i*y_i + (1-x_i)*(1-y_i) = 1
                    // This equals: 1 - x_i - y_i + 2*x_i*y_i
                    const one_minus_x = one.sub(x_i);
                    const one_minus_y = one.sub(y_i);
                    const eq_bit = x_i.mul(y_i).add(one_minus_x.mul(one_minus_y));
                    result = result.mul(eq_bit);
                }
                return result;
            }

            /// Materialize the entire table (for testing)
            pub fn materialize(allocator: Allocator) ![]u64 {
                const size = @as(usize, 1) << (2 * @min(XLEN, 8));
                const table = try allocator.alloc(u64, size);
                for (0..size) |i| {
                    table[i] = materializeEntry(@intCast(i));
                }
                return table;
            }
        };

        /// UnsignedLessThan: Checks if x < y (unsigned comparison)
        /// materializeEntry(interleaved(x, y)) = 1 if x < y (unsigned), else 0
        pub const UnsignedLessThan = struct {
            /// Materialize the entry at the given index
            pub fn materializeEntry(index: u128) u64 {
                const bits = uninterleaveBits(index);
                return if (bits.x < bits.y) 1 else 0;
            }

            /// Evaluate the MLE at point r
            /// The MLE for less-than is more complex. We use the formula:
            /// x < y iff there exists a most significant bit position j where x[j] = 0 and y[j] = 1,
            /// and for all i > j, x[i] = y[i].
            ///
            /// This can be expressed as:
            /// sum_{j=0}^{XLEN-1} (1-r_x[j]) * r_y[j] * prod_{k=0}^{j-1} eq(r_x[k], r_y[k])
            pub fn evaluateMLE(r: []const F) F {
                std.debug.assert(r.len == 2 * XLEN);

                var result = F.zero();
                var eq_prefix = F.one();
                const one = F.one();

                // Iterate from MSB to LSB
                inline for (0..XLEN) |i| {
                    const x_i = r[2 * i]; // x bit at position (XLEN-1-i), i.e., MSB first
                    const y_i = r[2 * i + 1];

                    // Contribution when this is the first differing bit with x=0, y=1
                    const one_minus_x = one.sub(x_i);
                    const lt_contribution = eq_prefix.mul(one_minus_x).mul(y_i);
                    result = result.add(lt_contribution);

                    // Update equality prefix for next iteration
                    const one_minus_y = one.sub(y_i);
                    const eq_bit = x_i.mul(y_i).add(one_minus_x.mul(one_minus_y));
                    eq_prefix = eq_prefix.mul(eq_bit);
                }

                return result;
            }

            /// Materialize the entire table (for testing)
            pub fn materialize(allocator: Allocator) ![]u64 {
                const size = @as(usize, 1) << (2 * @min(XLEN, 8));
                const table = try allocator.alloc(u64, size);
                for (0..size) |i| {
                    table[i] = materializeEntry(@intCast(i));
                }
                return table;
            }
        };

        /// SignedLessThan: Checks if x < y (signed comparison)
        /// materializeEntry(interleaved(x, y)) = 1 if x < y (signed), else 0
        pub const SignedLessThan = struct {
            /// Materialize the entry at the given index
            pub fn materializeEntry(index: u128) u64 {
                const bits = uninterleaveBits(index);
                // Interpret as signed values
                const x_signed: i64 = @bitCast(bits.x);
                const y_signed: i64 = @bitCast(bits.y);
                return if (x_signed < y_signed) 1 else 0;
            }

            /// Evaluate the MLE at point r
            /// For signed comparison, we use the formula from Jolt:
            /// signed_lt(x, y) = x_sign - y_sign + unsigned_lt(x, y)
            ///
            /// This works because:
            /// - If x negative, y positive: x_sign=1, y_sign=0, so contribution = 1
            /// - If x positive, y negative: x_sign=0, y_sign=1, so contribution = -1
            /// - If same sign: x_sign - y_sign = 0, use unsigned comparison
            pub fn evaluateMLE(r: []const F) F {
                std.debug.assert(r.len == 2 * XLEN);

                const one = F.one();

                // Sign bits are at position 0 (MSB in our interleaved format)
                const x_sign = r[0];
                const y_sign = r[1];

                // Compute unsigned less-than
                var lt = F.zero();
                var eq = F.one();

                inline for (0..XLEN) |i| {
                    const x_i = r[2 * i];
                    const y_i = r[2 * i + 1];

                    const one_minus_x = one.sub(x_i);
                    const one_minus_y = one.sub(y_i);

                    // lt += (1 - x_i) * y_i * eq
                    lt = lt.add(one_minus_x.mul(y_i).mul(eq));

                    // eq *= x_i * y_i + (1 - x_i) * (1 - y_i)
                    eq = eq.mul(x_i.mul(y_i).add(one_minus_x.mul(one_minus_y)));
                }

                // signed_lt = x_sign - y_sign + lt
                return x_sign.sub(y_sign).add(lt);
            }

            /// Materialize the entire table (for testing)
            pub fn materialize(allocator: Allocator) ![]u64 {
                const size = @as(usize, 1) << (2 * @min(XLEN, 8));
                const table = try allocator.alloc(u64, size);
                for (0..size) |i| {
                    table[i] = materializeEntry(@intCast(i));
                }
                return table;
            }
        };

        /// NotEqual: Checks if two operands are not equal
        /// materializeEntry(interleaved(x, y)) = 1 if x != y, else 0
        pub const NotEqual = struct {
            /// Materialize the entry at the given index
            pub fn materializeEntry(index: u128) u64 {
                const bits = uninterleaveBits(index);
                return if (bits.x != bits.y) 1 else 0;
            }

            /// Evaluate the MLE at point r
            /// NotEqual = 1 - Equal
            pub fn evaluateMLE(r: []const F) F {
                const eq_result = Equal.evaluateMLE(r);
                return F.one().sub(eq_result);
            }

            /// Materialize the entire table (for testing)
            pub fn materialize(allocator: Allocator) ![]u64 {
                const size = @as(usize, 1) << (2 * @min(XLEN, 8));
                const table = try allocator.alloc(u64, size);
                for (0..size) |i| {
                    table[i] = materializeEntry(@intCast(i));
                }
                return table;
            }
        };

        /// UnsignedGreaterThanEqual: Checks if x >= y (unsigned)
        /// materializeEntry(interleaved(x, y)) = 1 if x >= y, else 0
        pub const UnsignedGreaterThanEqual = struct {
            /// Materialize the entry at the given index
            pub fn materializeEntry(index: u128) u64 {
                const bits = uninterleaveBits(index);
                return if (bits.x >= bits.y) 1 else 0;
            }

            /// Evaluate the MLE at point r
            /// x >= y = NOT(x < y) = 1 - lt(x, y)
            pub fn evaluateMLE(r: []const F) F {
                const lt_result = UnsignedLessThan.evaluateMLE(r);
                return F.one().sub(lt_result);
            }

            /// Materialize the entire table (for testing)
            pub fn materialize(allocator: Allocator) ![]u64 {
                const size = @as(usize, 1) << (2 * @min(XLEN, 8));
                const table = try allocator.alloc(u64, size);
                for (0..size) |i| {
                    table[i] = materializeEntry(@intCast(i));
                }
                return table;
            }
        };

        /// UnsignedLessThanEqual: Checks if x <= y (unsigned)
        /// materializeEntry(interleaved(x, y)) = 1 if x <= y, else 0
        pub const UnsignedLessThanEqual = struct {
            /// Materialize the entry at the given index
            pub fn materializeEntry(index: u128) u64 {
                const bits = uninterleaveBits(index);
                return if (bits.x <= bits.y) 1 else 0;
            }

            /// Evaluate the MLE at point r
            /// x <= y = x < y OR x == y = lt(x,y) + eq(x,y) - lt*eq (but lt and eq are disjoint)
            /// Simpler: x <= y = NOT(x > y) = NOT(y < x)
            /// We need to swap operands to compute y < x
            pub fn evaluateMLE(r: []const F) F {
                std.debug.assert(r.len == 2 * XLEN);

                // Compute y < x by swapping operands
                var lt = F.zero();
                var eq = F.one();
                const one = F.one();

                inline for (0..XLEN) |i| {
                    // Swap x and y
                    const y_i = r[2 * i]; // This was x
                    const x_i = r[2 * i + 1]; // This was y

                    const one_minus_x = one.sub(x_i);
                    const one_minus_y = one.sub(y_i);

                    lt = lt.add(one_minus_x.mul(y_i).mul(eq));
                    eq = eq.mul(x_i.mul(y_i).add(one_minus_x.mul(one_minus_y)));
                }

                // x <= y = NOT(y < x) = 1 - lt
                return one.sub(lt);
            }

            /// Materialize the entire table (for testing)
            pub fn materialize(allocator: Allocator) ![]u64 {
                const size = @as(usize, 1) << (2 * @min(XLEN, 8));
                const table = try allocator.alloc(u64, size);
                for (0..size) |i| {
                    table[i] = materializeEntry(@intCast(i));
                }
                return table;
            }
        };

        /// SignedGreaterThanEqual: Checks if x >= y (signed)
        /// materializeEntry(interleaved(x, y)) = 1 if x >= y (signed), else 0
        pub const SignedGreaterThanEqual = struct {
            /// Materialize the entry at the given index
            pub fn materializeEntry(index: u128) u64 {
                const bits = uninterleaveBits(index);
                const x_signed: i64 = @bitCast(bits.x);
                const y_signed: i64 = @bitCast(bits.y);
                return if (x_signed >= y_signed) 1 else 0;
            }

            /// Evaluate the MLE at point r
            /// x >= y = NOT(x < y) = 1 - signed_lt(x, y)
            pub fn evaluateMLE(r: []const F) F {
                const lt_result = SignedLessThan.evaluateMLE(r);
                return F.one().sub(lt_result);
            }

            /// Materialize the entire table (for testing)
            pub fn materialize(allocator: Allocator) ![]u64 {
                const size = @as(usize, 1) << (2 * @min(XLEN, 8));
                const table = try allocator.alloc(u64, size);
                for (0..size) |i| {
                    table[i] = materializeEntry(@intCast(i));
                }
                return table;
            }
        };

        /// Movsign: Returns the sign of the operand (most significant bit)
        /// materializeEntry(x) = x >> (XLEN - 1)
        pub const Movsign = struct {
            /// Materialize the entry at the given index
            pub fn materializeEntry(index: u128) u64 {
                const bits = uninterleaveBits(index);
                // Return MSB of x
                return (bits.x >> (XLEN - 1)) & 1;
            }

            /// Evaluate the MLE at point r
            /// Just returns r[0] (the MSB of x in our interleaved format)
            pub fn evaluateMLE(r: []const F) F {
                std.debug.assert(r.len == 2 * XLEN);
                return r[0]; // MSB of x
            }

            /// Materialize the entire table (for testing)
            pub fn materialize(allocator: Allocator) ![]u64 {
                const size = @as(usize, 1) << (2 * @min(XLEN, 8));
                const table = try allocator.alloc(u64, size);
                for (0..size) |i| {
                    table[i] = materializeEntry(@intCast(i));
                }
                return table;
            }
        };

        /// Sub: Subtraction with wrap-around (x - y)
        /// materializeEntry(interleaved(x, y)) = x - y (wrapping)
        pub const Sub = struct {
            /// Materialize the entry at the given index
            pub fn materializeEntry(index: u128) u64 {
                const bits = uninterleaveBits(index);
                return bits.x -% bits.y; // Wrapping subtraction
            }

            /// Evaluate the MLE at point r
            /// Sub MLE is more complex - we can express it in terms of Add
            /// x - y = x + (2^XLEN - y) mod 2^XLEN
            /// For MLE purposes, this doesn't have a simple closed form like AND/XOR
            /// We use a brute-force evaluation for correctness
            pub fn evaluateMLE(r: []const F) F {
                std.debug.assert(r.len == 2 * XLEN);

                // Compute x value contribution
                var x_val = F.zero();
                var y_val = F.zero();

                inline for (0..XLEN) |i| {
                    const shift: u6 = XLEN - 1 - i;
                    const coeff = F.fromU64(@as(u64, 1) << shift);
                    x_val = x_val.add(coeff.mul(r[2 * i]));
                    y_val = y_val.add(coeff.mul(r[2 * i + 1]));
                }

                // Return x - y (field subtraction handles the wrapping correctly in the field)
                return x_val.sub(y_val);
            }

            /// Materialize the entire table (for testing)
            pub fn materialize(allocator: Allocator) ![]u64 {
                const size = @as(usize, 1) << (2 * @min(XLEN, 8));
                const table = try allocator.alloc(u64, size);
                for (0..size) |i| {
                    table[i] = materializeEntry(@intCast(i));
                }
                return table;
            }
        };

        /// Andn: Bitwise AND-NOT (x & ~y)
        /// materializeEntry(interleaved(x, y)) = x & ~y
        pub const Andn = struct {
            /// Materialize the entry at the given index
            pub fn materializeEntry(index: u128) u64 {
                const bits = uninterleaveBits(index);
                return bits.x & ~bits.y;
            }

            /// Evaluate the MLE at point r
            /// x & ~y: For each bit, x_i * (1 - y_i)
            pub fn evaluateMLE(r: []const F) F {
                std.debug.assert(r.len == 2 * XLEN);

                var result = F.zero();
                const one = F.one();

                inline for (0..XLEN) |i| {
                    const shift: u6 = XLEN - 1 - i;
                    const coeff = F.fromU64(@as(u64, 1) << shift);
                    const x_i = r[2 * i];
                    const y_i = r[2 * i + 1];
                    // x & ~y at bit i: x_i * (1 - y_i)
                    result = result.add(coeff.mul(x_i.mul(one.sub(y_i))));
                }
                return result;
            }

            /// Materialize the entire table (for testing)
            pub fn materialize(allocator: Allocator) ![]u64 {
                const size = @as(usize, 1) << (2 * @min(XLEN, 8));
                const table = try allocator.alloc(u64, size);
                for (0..size) |i| {
                    table[i] = materializeEntry(@intCast(i));
                }
                return table;
            }
        };

        /// LeftShift: Logical left shift
        /// materializeEntry(interleaved(x, shift_amount)) = x << (shift_amount % XLEN)
        pub const LeftShift = struct {
            /// Materialize the entry at the given index
            pub fn materializeEntry(index: u128) u64 {
                const bits = uninterleaveBits(index);
                // Shift amount is lower bits of y (only log2(XLEN) bits matter)
                const shift_mask: u6 = XLEN - 1;
                const shift: u6 = @truncate(bits.y & @as(u64, shift_mask));
                // Mask to XLEN bits first, then shift
                const mask: u64 = if (XLEN == 64) ~@as(u64, 0) else (@as(u64, 1) << XLEN) - 1;
                const masked_x = bits.x & mask;
                return (masked_x << shift) & mask;
            }

            /// Evaluate the MLE at point r
            /// This is more complex as it involves bit position dependencies
            pub fn evaluateMLE(r: []const F) F {
                std.debug.assert(r.len == 2 * XLEN);
                // For shift tables, MLE evaluation requires summing over all shift amounts
                // This is a simplified version
                var result = F.zero();

                // For small XLEN (testing), we can enumerate
                if (XLEN <= 8) {
                    const size = @as(usize, 1) << (2 * XLEN);
                    for (0..size) |idx| {
                        const val = materializeEntry(@intCast(idx));
                        // Compute Lagrange basis term
                        var basis = F.one();
                        inline for (0..(2 * XLEN)) |b| {
                            const bit: u1 = @truncate(idx >> b);
                            if (bit == 1) {
                                basis = basis.mul(r[b]);
                            } else {
                                basis = basis.mul(F.one().sub(r[b]));
                            }
                        }
                        result = result.add(F.fromU64(val).mul(basis));
                    }
                }
                return result;
            }

            /// Materialize the entire table (for testing)
            pub fn materialize(allocator: Allocator) ![]u64 {
                const size = @as(usize, 1) << (2 * @min(XLEN, 8));
                const table = try allocator.alloc(u64, size);
                for (0..size) |i| {
                    table[i] = materializeEntry(@intCast(i));
                }
                return table;
            }
        };

        /// RightShift: Logical right shift
        /// materializeEntry(interleaved(x, shift_amount)) = x >> (shift_amount % XLEN)
        pub const RightShift = struct {
            /// Materialize the entry at the given index
            pub fn materializeEntry(index: u128) u64 {
                const bits = uninterleaveBits(index);
                // Shift amount is lower bits of y
                const shift_mask: u6 = XLEN - 1;
                const shift: u6 = @truncate(bits.y & @as(u64, shift_mask));
                // Mask to XLEN bits first, then shift
                const mask: u64 = if (XLEN == 64) ~@as(u64, 0) else (@as(u64, 1) << XLEN) - 1;
                const masked_x = bits.x & mask;
                return masked_x >> shift;
            }

            /// Evaluate the MLE at point r
            pub fn evaluateMLE(r: []const F) F {
                std.debug.assert(r.len == 2 * XLEN);
                var result = F.zero();

                if (XLEN <= 8) {
                    const size = @as(usize, 1) << (2 * XLEN);
                    for (0..size) |idx| {
                        const val = materializeEntry(@intCast(idx));
                        var basis = F.one();
                        inline for (0..(2 * XLEN)) |b| {
                            const bit: u1 = @truncate(idx >> b);
                            if (bit == 1) {
                                basis = basis.mul(r[b]);
                            } else {
                                basis = basis.mul(F.one().sub(r[b]));
                            }
                        }
                        result = result.add(F.fromU64(val).mul(basis));
                    }
                }
                return result;
            }

            /// Materialize the entire table (for testing)
            pub fn materialize(allocator: Allocator) ![]u64 {
                const size = @as(usize, 1) << (2 * @min(XLEN, 8));
                const table = try allocator.alloc(u64, size);
                for (0..size) |i| {
                    table[i] = materializeEntry(@intCast(i));
                }
                return table;
            }
        };

        /// RightShiftArithmetic: Arithmetic right shift (sign-extends)
        /// materializeEntry(interleaved(x, shift_amount)) = (signed) x >> shift_amount
        pub const RightShiftArithmetic = struct {
            /// Materialize the entry at the given index
            pub fn materializeEntry(index: u128) u64 {
                const bits = uninterleaveBits(index);
                // Shift amount is lower bits of y
                const shift_mask: u6 = XLEN - 1;
                const shift: u6 = @truncate(bits.y & @as(u64, shift_mask));

                // Mask to XLEN bits and treat as signed
                const mask: u64 = if (XLEN == 64) ~@as(u64, 0) else (@as(u64, 1) << XLEN) - 1;
                const masked_x = bits.x & mask;

                // Sign extend to i64, then arithmetic shift, then mask back
                if (XLEN == 64) {
                    const signed_x: i64 = @bitCast(masked_x);
                    const shifted: i64 = signed_x >> shift;
                    return @bitCast(shifted);
                } else if (XLEN == 32) {
                    const signed_x: i32 = @truncate(@as(i64, @bitCast(masked_x << (64 - 32))) >> (64 - 32));
                    const shifted: i32 = signed_x >> @as(u5, @truncate(shift));
                    return @as(u64, @as(u32, @bitCast(shifted)));
                } else {
                    // For testing with small XLEN (e.g., 8)
                    const shift_for_sign = 64 - XLEN;
                    const signed_val: i64 = @as(i64, @bitCast(masked_x << @truncate(shift_for_sign))) >> @truncate(shift_for_sign);
                    const shifted: i64 = signed_val >> @as(u6, shift);
                    return @as(u64, @truncate(@as(u64, @bitCast(shifted)))) & mask;
                }
            }

            /// Evaluate the MLE at point r
            pub fn evaluateMLE(r: []const F) F {
                std.debug.assert(r.len == 2 * XLEN);
                var result = F.zero();

                if (XLEN <= 8) {
                    const size = @as(usize, 1) << (2 * XLEN);
                    for (0..size) |idx| {
                        const val = materializeEntry(@intCast(idx));
                        var basis = F.one();
                        inline for (0..(2 * XLEN)) |b| {
                            const bit: u1 = @truncate(idx >> b);
                            if (bit == 1) {
                                basis = basis.mul(r[b]);
                            } else {
                                basis = basis.mul(F.one().sub(r[b]));
                            }
                        }
                        result = result.add(F.fromU64(val).mul(basis));
                    }
                }
                return result;
            }

            /// Materialize the entire table (for testing)
            pub fn materialize(allocator: Allocator) ![]u64 {
                const size = @as(usize, 1) << (2 * @min(XLEN, 8));
                const table = try allocator.alloc(u64, size);
                for (0..size) |i| {
                    table[i] = materializeEntry(@intCast(i));
                }
                return table;
            }
        };

        /// Pow2: Power of 2 table - returns 2^y (useful for shifts)
        /// materializeEntry(y) = 2^y (mod 2^XLEN)
        pub const Pow2 = struct {
            /// Materialize the entry at the given index
            pub fn materializeEntry(index: u128) u64 {
                const y: u6 = @truncate(index & (XLEN - 1));
                const result: u64 = @as(u64, 1) << y;
                if (XLEN == 64) {
                    return result;
                } else {
                    const mask: u64 = (@as(u64, 1) << XLEN) - 1;
                    return result & mask;
                }
            }

            /// Evaluate the MLE at point r (for single-operand table)
            pub fn evaluateMLE(r: []const F) F {
                std.debug.assert(r.len == XLEN);
                var result = F.zero();

                if (XLEN <= 8) {
                    const size = @as(usize, 1) << XLEN;
                    for (0..size) |idx| {
                        const val = materializeEntry(@intCast(idx));
                        var basis = F.one();
                        inline for (0..XLEN) |b| {
                            const bit: u1 = @truncate(idx >> b);
                            if (bit == 1) {
                                basis = basis.mul(r[b]);
                            } else {
                                basis = basis.mul(F.one().sub(r[b]));
                            }
                        }
                        result = result.add(F.fromU64(val).mul(basis));
                    }
                }
                return result;
            }

            /// Materialize the entire table (for testing)
            pub fn materialize(allocator: Allocator) ![]u64 {
                const size = @as(usize, 1) << @min(XLEN, 8);
                const table = try allocator.alloc(u64, size);
                for (0..size) |i| {
                    table[i] = materializeEntry(@intCast(i));
                }
                return table;
            }
        };

        /// SignExtend8: Sign-extend from 8 bits to XLEN bits
        /// Useful for LB (load byte signed) instruction
        pub const SignExtend8 = struct {
            /// Materialize the entry at the given index
            pub fn materializeEntry(index: u128) u64 {
                const byte: u8 = @truncate(index);
                const signed: i8 = @bitCast(byte);
                const extended: i64 = @as(i64, signed);
                if (XLEN == 64) {
                    return @bitCast(extended);
                } else {
                    const mask: u64 = (@as(u64, 1) << XLEN) - 1;
                    return @as(u64, @bitCast(extended)) & mask;
                }
            }

            /// Evaluate the MLE at point r
            pub fn evaluateMLE(r: []const F) F {
                // Only 8 bits of input
                std.debug.assert(r.len >= 8);
                var result = F.zero();

                const size: usize = 256; // 2^8
                for (0..size) |idx| {
                    const val = materializeEntry(@intCast(idx));
                    var basis = F.one();
                    for (0..8) |b| {
                        const bit: u1 = @truncate(idx >> @truncate(b));
                        if (bit == 1) {
                            basis = basis.mul(r[b]);
                        } else {
                            basis = basis.mul(F.one().sub(r[b]));
                        }
                    }
                    result = result.add(F.fromU64(val).mul(basis));
                }
                return result;
            }

            /// Materialize the entire table (for testing)
            pub fn materialize(allocator: Allocator) ![]u64 {
                const table = try allocator.alloc(u64, 256);
                for (0..256) |i| {
                    table[i] = materializeEntry(@intCast(i));
                }
                return table;
            }
        };

        /// SignExtend16: Sign-extend from 16 bits to XLEN bits
        /// Useful for LH (load half signed) instruction
        pub const SignExtend16 = struct {
            /// Materialize the entry at the given index
            pub fn materializeEntry(index: u128) u64 {
                const half: u16 = @truncate(index);
                const signed: i16 = @bitCast(half);
                const extended: i64 = @as(i64, signed);
                if (XLEN == 64) {
                    return @bitCast(extended);
                } else {
                    const mask: u64 = (@as(u64, 1) << XLEN) - 1;
                    return @as(u64, @bitCast(extended)) & mask;
                }
            }

            /// Evaluate the MLE at point r
            pub fn evaluateMLE(r: []const F) F {
                std.debug.assert(r.len >= 16);
                var result = F.zero();

                // For 16-bit input, table is 65536 entries
                // Only practical for small test cases
                if (r.len >= 16) {
                    const size: usize = 65536; // 2^16
                    for (0..size) |idx| {
                        const val = materializeEntry(@intCast(idx));
                        var basis = F.one();
                        for (0..16) |b| {
                            const bit: u1 = @truncate(idx >> @truncate(b));
                            if (bit == 1) {
                                basis = basis.mul(r[b]);
                            } else {
                                basis = basis.mul(F.one().sub(r[b]));
                            }
                        }
                        result = result.add(F.fromU64(val).mul(basis));
                    }
                }
                return result;
            }

            /// Materialize the entire table (for testing, limited size)
            pub fn materialize(allocator: Allocator) ![]u64 {
                const table = try allocator.alloc(u64, 65536);
                for (0..65536) |i| {
                    table[i] = materializeEntry(@intCast(i));
                }
                return table;
            }
        };

        /// SignExtend32: Sign-extend from 32 bits to XLEN bits
        /// Useful for LW (load word signed) on RV64
        pub const SignExtend32 = struct {
            /// Materialize the entry at the given index
            pub fn materializeEntry(index: u128) u64 {
                const word: u32 = @truncate(index);
                const signed: i32 = @bitCast(word);
                const extended: i64 = @as(i64, signed);
                if (XLEN == 64) {
                    return @bitCast(extended);
                } else {
                    const mask: u64 = (@as(u64, 1) << XLEN) - 1;
                    return @as(u64, @bitCast(extended)) & mask;
                }
            }

            /// Evaluate the MLE at point r (too large to materialize fully)
            pub fn evaluateMLE(r: []const F) F {
                // For 32-bit, we cannot enumerate all 2^32 entries
                // This would require a clever closed-form formula
                // Return zero for now - proper implementation would need decomposition
                _ = r;
                return F.zero();
            }

            /// Materialize the entire table - NOT practical for full size
            /// Only use for testing with limited index ranges
            pub fn materialize(allocator: Allocator) ![]u64 {
                // Only materialize first 256 entries for testing
                const table = try allocator.alloc(u64, 256);
                for (0..256) |i| {
                    table[i] = materializeEntry(@intCast(i));
                }
                return table;
            }
        };

        // ========================================================================
        // Division/Remainder Validation Tables
        // ========================================================================

        /// ValidDiv0: Validates division by zero behavior
        /// Returns 1 if (divisor == 0 && quotient == MAX_VALUE) or (divisor != 0)
        /// This enforces RISC-V's division by zero semantics: x / 0 = MAX_VALUE
        pub const ValidDiv0 = struct {
            /// Materialize the entry at the given index
            /// Index format: interleaved(divisor, quotient)
            pub fn materializeEntry(index: u128) u64 {
                const operands = uninterleaveBits(index);
                const divisor = operands.x;
                const quotient = operands.y;

                if (divisor == 0) {
                    // If divisor is zero, quotient must be MAX_VALUE
                    const max_value: u64 = if (XLEN == 64)
                        @as(u64, 0xFFFFFFFFFFFFFFFF)
                    else if (XLEN == 32)
                        @as(u64, 0xFFFFFFFF)
                    else if (XLEN == 8)
                        @as(u64, 0xFF)
                    else
                        (@as(u64, 1) << XLEN) - 1;

                    return if (quotient == max_value) 1 else 0;
                } else {
                    // If divisor is non-zero, any quotient is potentially valid
                    // (actual validity depends on dividend, checked elsewhere)
                    return 1;
                }
            }

            /// Evaluate the MLE at point r
            /// MLE: 1 - divisor_is_zero + is_valid_div_by_zero
            /// where divisor_is_zero = prod(1 - divisor_bit_i)
            ///       is_valid_div_by_zero = prod((1 - divisor_bit_i) * quotient_bit_i)
            pub fn evaluateMLE(r: []const F) F {
                std.debug.assert(r.len >= 2 * XLEN);

                var divisor_is_zero = F.one();
                var is_valid_div_by_zero = F.one();

                // r is interleaved: divisor bits at odd positions, quotient at even
                for (0..XLEN) |i| {
                    const x_i = r[2 * i]; // divisor bit
                    const y_i = r[2 * i + 1]; // quotient bit
                    divisor_is_zero = divisor_is_zero.mul(F.one().sub(x_i));
                    is_valid_div_by_zero = is_valid_div_by_zero.mul(F.one().sub(x_i).mul(y_i));
                }

                // Return: 1 - divisor_is_zero + is_valid_div_by_zero
                return F.one().sub(divisor_is_zero).add(is_valid_div_by_zero);
            }
        };

        /// ValidUnsignedRemainder: Validates that remainder < divisor (or divisor == 0)
        /// Returns 1 if divisor == 0 OR remainder < divisor
        pub const ValidUnsignedRemainder = struct {
            /// Materialize the entry at the given index
            /// Index format: interleaved(remainder, divisor)
            pub fn materializeEntry(index: u128) u64 {
                const operands = uninterleaveBits(index);
                const remainder = operands.x;
                const divisor = operands.y;

                // Valid if divisor == 0 (any remainder allowed) or remainder < divisor
                return if (divisor == 0 or remainder < divisor) 1 else 0;
            }

            /// Evaluate the MLE at point r
            /// MLE: divisor_is_zero + lt (where lt uses lexicographic comparison)
            pub fn evaluateMLE(r: []const F) F {
                std.debug.assert(r.len >= 2 * XLEN);

                var divisor_is_zero = F.one();
                var lt = F.zero();
                var eq = F.one();

                for (0..XLEN) |i| {
                    const x_i = r[2 * i]; // remainder bit
                    const y_i = r[2 * i + 1]; // divisor bit

                    // divisor_is_zero = prod(1 - y_i)
                    divisor_is_zero = divisor_is_zero.mul(F.one().sub(y_i));

                    // lt accumulates when we first find remainder_bit < divisor_bit
                    // while previous bits were equal
                    lt = lt.add(F.one().sub(x_i).mul(y_i).mul(eq));

                    // eq = prod(x_i == y_i) = prod(x_i * y_i + (1-x_i)*(1-y_i))
                    eq = eq.mul(x_i.mul(y_i).add(F.one().sub(x_i).mul(F.one().sub(y_i))));
                }

                return lt.add(divisor_is_zero);
            }
        };

        /// ValidSignedRemainder: Validates signed remainder semantics
        /// Returns 1 if: divisor == 0 OR remainder == 0 OR
        ///              (|remainder| < |divisor| AND sign(remainder) == sign(divisor))
        /// LowerHalfWord: extracts the lower XLEN/2 bits of the input value
        /// For XLEN=64, this masks to lower 32 bits: LowerHalfWord(x) = x mod 2^32
        /// Uses identity (non-interleaved) addressing: r[0..XLEN] = upper bits, r[XLEN..2*XLEN] = lower bits
        pub const LowerHalfWord = struct {
            /// Materialize the entry at the given index
            pub fn materializeEntry(index: u128) u64 {
                const half_word_size = XLEN / 2;
                return @truncate(index % (@as(u128, 1) << half_word_size));
            }

            /// Evaluate the MLE at point r
            /// For identity-path tables, r is split as:
            ///   r[0..XLEN] = "left operand" bits (unused for this table)
            ///   r[XLEN..2*XLEN] = "right operand" bits (the value, MSB first)
            ///
            /// LowerHalfWord keeps only the lower half bits:
            /// result = Σ(i=0..half_word_size-1) 2^(half_word_size-1-i) * r[XLEN + half_word_size + i]
            pub fn evaluateMLE(r: []const F) F {
                std.debug.assert(r.len == 2 * XLEN);
                const half_word_size = XLEN / 2;

                var result = F.zero();
                inline for (0..half_word_size) |i| {
                    const shift: u6 = half_word_size - 1 - i;
                    const coeff = F.fromU64(@as(u64, 1) << shift);
                    result = result.add(coeff.mul(r[XLEN + half_word_size + i]));
                }

                return result;
            }
        };

        /// SignExtendHalfWord: sign-extends the lower XLEN/2 bits to full XLEN
        /// For XLEN=64, this sign-extends a 32-bit value to 64-bit
        /// Uses identity (non-interleaved) addressing: r[0..XLEN] = upper bits, r[XLEN..2*XLEN] = lower bits
        pub const SignExtendHalfWord = struct {
            /// Evaluate the MLE at point r
            /// Matches Jolt's sign_extend_half_word.rs evaluate_mle exactly.
            ///
            /// For identity-path tables, r is split as:
            ///   r[0..XLEN] = "left operand" bits (upper word, MSB first)
            ///   r[XLEN..2*XLEN] = "right operand" bits (the value, MSB first)
            ///
            /// The input value's lower half is in r[XLEN+half_word_size..2*XLEN]
            /// The sign bit is r[XLEN+half_word_size] (MSB of lower half)
            /// Output = lower_half + upper_half * 2^half_word_size
            /// where upper_half has all bits equal to the sign bit.
            pub fn evaluateMLE(r: []const F) F {
                std.debug.assert(r.len == 2 * XLEN);
                const half_word_size = XLEN / 2;

                // Sum for lower half bits (from the second operand, starting at XLEN)
                // r[XLEN + half_word_size + i] for i in 0..half_word_size
                var lower_half = F.zero();
                inline for (0..half_word_size) |i| {
                    const shift: u6 = half_word_size - 1 - i;
                    const coeff = F.fromU64(@as(u64, 1) << shift);
                    lower_half = lower_half.add(coeff.mul(r[XLEN + half_word_size + i]));
                }

                // Sign bit is the MSB of the lower half
                const sign_bit = r[XLEN + half_word_size];

                // Upper half: all bits equal to sign bit
                // sum_{i=0}^{half_word_size-1} 2^{half_word_size-1-i} * sign_bit
                // = sign_bit * (2^0 + 2^1 + ... + 2^{half_word_size-1})
                // = sign_bit * (2^half_word_size - 1)
                var upper_half = F.zero();
                inline for (0..half_word_size) |i| {
                    const shift: u6 = half_word_size - 1 - i;
                    const coeff = F.fromU64(@as(u64, 1) << shift);
                    upper_half = upper_half.add(coeff.mul(sign_bit));
                }

                // Result = lower_half + upper_half * 2^half_word_size
                const upper_shift = F.fromU64(@as(u64, 1) << half_word_size);
                return lower_half.add(upper_half.mul(upper_shift));
            }
        };

        /// HalfwordAlignment: output = 1 if (input % 2 == 0), else 0
        /// Single-operand table: AddOperands combines rs1+imm before lookup.
        /// MLE: f(r) = 1 - r[0] (bit 0 determines halfword alignment)
        pub const HalfwordAlignment = struct {
            pub fn materializeEntry(index: u128) u64 {
                return if (index & 1 == 0) 1 else 0;
            }

            pub fn evaluateMLE(r: []const F) F {
                // Alignment check: output 1 if lowest bit is 0
                // r is in HighToLow order: r[0]=MSB, r[len-1]=LSB
                // MLE = 1 - r_lsb where r_lsb = r[len-1]
                return F.one().sub(r[r.len - 1]);
            }
        };

        /// WordAlignment: output = 1 if (input % 4 == 0), else 0
        /// MLE: f(r) = (1-r_bit0) * (1-r_bit1) where bit0=LSB, bit1=next
        pub const WordAlignment = struct {
            pub fn materializeEntry(index: u128) u64 {
                return if (index & 3 == 0) 1 else 0;
            }

            pub fn evaluateMLE(r: []const F) F {
                // r is in HighToLow: r[len-1]=LSB (bit 0), r[len-2]=bit 1
                return (F.one().sub(r[r.len - 1])).mul(F.one().sub(r[r.len - 2]));
            }
        };

        /// ShiftRightBitmask: output = bitmask with 1s from bit `shift` to bit `XLEN-1`
        /// Input: shift amount (0..XLEN-1). Uses AddOperands (rs1+0=rs1).
        /// output = ((1 << (XLEN - shift)) - 1) << shift
        /// For shift=0: all 1s. For shift=63: only MSB.
        pub const ShiftRightBitmask = struct {
            pub fn materializeEntry(index: u128) u64 {
                const shift: u7 = @truncate(index & (XLEN - 1));
                if (shift == 0) return @as(u64, 0) -% 1; // all 1s
                const ones: u128 = (@as(u128, 1) << @intCast(XLEN - @as(u8, shift))) - 1;
                return @truncate(ones << shift);
            }

            pub fn evaluateMLE(r: []const F) F {
                // Single-operand table: iterate over all XLEN possible shift amounts
                // For XLEN=64 this is only 64 entries (the shift amount is only log2(XLEN) bits)
                std.debug.assert(r.len >= XLEN);
                // The input is the shift amount, which only uses log2(XLEN) bits.
                // But the table is indexed by the full XLEN-bit operand.
                // We need to sum over all 2^XLEN entries, but only the lower log2(XLEN) bits matter.
                // For bits >= log2(XLEN), they don't affect the result. The result is the same
                // regardless of those higher bits since we mask with (XLEN-1).
                // So MLE = (product of (1-r[i]) for i >= log2(XLEN)) * sum_over_shift_amounts
                const log_xlen = if (XLEN == 64) 6 else 5;
                // Factor out high bits: they must all be 0 for a valid shift amount
                var high_factor = F.one();
                for (log_xlen..XLEN) |i| {
                    high_factor = high_factor.mul(F.one().sub(r[i]));
                }
                // Sum over valid shift amounts (0..XLEN-1)
                var sum = F.zero();
                for (0..XLEN) |shift| {
                    const val = materializeEntry(@intCast(shift));
                    var basis = F.one();
                    for (0..log_xlen) |b| {
                        const bit: u1 = @truncate(shift >> @truncate(b));
                        if (bit == 1) {
                            basis = basis.mul(r[b]);
                        } else {
                            basis = basis.mul(F.one().sub(r[b]));
                        }
                    }
                    sum = sum.add(F.fromU64(val).mul(basis));
                }
                return high_factor.mul(sum);
            }
        };

        /// VirtualSRA: shift-right-arithmetic using bitmask encoding
        /// Same as VirtualSRL but with sign extension.
        /// Reference: jolt-core/src/zkvm/lookup_table/virtual_sra.rs
        pub const VirtualSRA = struct {
            pub fn materializeEntry(index: u128) u64 {
                var x: u64 = 0;
                var y: u64 = 0;
                inline for (0..XLEN) |i| {
                    x |= @as(u64, @truncate((index >> (2 * i + 1)) & 1)) << i;
                    y |= @as(u64, @truncate((index >> (2 * i)) & 1)) << i;
                }
                const sign_bit: u64 = (x >> (XLEN - 1)) & 1;
                var entry: u64 = 0;
                inline for (0..XLEN) |i| {
                    const bit_pos = XLEN - 1 - i;
                    const x_i: u64 = (x >> bit_pos) & 1;
                    const y_i: u64 = (y >> bit_pos) & 1;
                    entry = entry *% (1 + y_i) +% (x_i *% y_i) +% (sign_bit *% (1 - y_i));
                }
                return entry;
            }

            /// MLE follows the same interleave convention as VirtualSRL.
            /// r[2*i] corresponds to even bit positions (first arg in interleaveBits),
            /// r[2*i+1] corresponds to odd bit positions (second arg in interleaveBits).
            /// The sign bit is the MSB of the value operand.
            pub fn evaluateMLE(r: []const F) F {
                std.debug.assert(r.len == 2 * XLEN);
                // Follow VirtualSRL's convention: r[2*i] and r[2*i+1] map to the
                // interleaved bit positions. The formula is the same recurrence but
                // with sign_bit * (1 - y_i) added.
                // Sign bit (MSB of value) is at the same relative position as VirtualSRL's MSB
                const sign_bit = r[2 * (XLEN - 1)];
                var result = F.zero();
                inline for (0..XLEN) |i| {
                    const x_i = r[2 * i]; // same convention as VirtualSRL
                    const y_i = r[2 * i + 1]; // same convention as VirtualSRL
                    result = result.mul(F.one().add(y_i)).add(x_i.mul(y_i)).add(sign_bit.mul(F.one().sub(y_i)));
                }
                return result;
            }
        };

        /// VirtualSRL: shift-right-logical using bitmask encoding
        /// The index interleaves value bits (x) and bitmask bits (y).
        /// result = Π_{i=0}^{XLEN-1} (1 + y_i) * ... computed iteratively:
        ///   result = 0
        ///   for i in 0..XLEN: result = result * (1 + y_i) + x_i * y_i
        /// Reference: jolt-core/src/zkvm/lookup_table/virtual_srl.rs
        pub const VirtualSRL = struct {
            /// Materialize the entry at a given interleaved index.
            /// Uninterleaves the index into (value, bitmask), then computes
            /// result = 0; for i in 0..XLEN: result = result * (1 + y_i) + x_i * y_i
            pub fn materializeEntry(index: u128) u64 {
                // Uninterleave matching Jolt convention:
                // interleaveBits128(x, y) puts x at ODD positions, y at EVEN positions
                // So: odd bits → x (value/first arg), even bits → y (bitmask/second arg)
                var x: u64 = 0; // value (from odd positions)
                var y: u64 = 0; // bitmask (from even positions)
                inline for (0..XLEN) |i| {
                    x |= @as(u64, @truncate((index >> (2 * i + 1)) & 1)) << i; // odd positions
                    y |= @as(u64, @truncate((index >> (2 * i)) & 1)) << i; // even positions
                }
                // Compute the lookup table entry iterating MSB to LSB
                var entry: u64 = 0;
                inline for (0..XLEN) |i| {
                    const bit_pos = XLEN - 1 - i;
                    const x_i: u64 = (x >> bit_pos) & 1;
                    const y_i: u64 = (y >> bit_pos) & 1;
                    entry = entry *% (1 + y_i) +% (x_i *% y_i);
                }
                return entry;
            }

            pub fn evaluateMLE(r: []const F) F {
                std.debug.assert(r.len == 2 * XLEN);
                var result = F.zero();
                inline for (0..XLEN) |i| {
                    const x_i = r[2 * i]; // value bit
                    const y_i = r[2 * i + 1]; // bitmask bit
                    result = result.mul(F.one().add(y_i)).add(x_i.mul(y_i));
                }
                return result;
            }
        };

        /// Number of lookup tables in Jolt
        pub const NUM_TABLES: usize = 42;

        /// Evaluate the MLE of any table by index
        /// This matches Jolt's LookupTables enum ordering:
        /// 0: RangeCheck, 1: RangeCheckAligned, 2: And, 3: Andn, 4: Or, 5: Xor,
        /// 6: Equal, 7: SignedGreaterThanEqual, 8: UnsignedGreaterThanEqual, 9: NotEqual,
        /// 10: SignedLessThan, 11: UnsignedLessThan, 12: Movsign, ...
        pub fn evaluateTableMLE(table_index: usize, r: []const F) F {
            return switch (table_index) {
                0 => RangeCheck.evaluateMLE(r),
                1 => blk_rca: {
                    // RangeCheckAligned = RangeCheck with LSB cleared
                    // MLE = Σ_{i=0}^{XLEN-2} 2^{XLEN-1-i} * r[XLEN+i]
                    // Same as RangeCheck but without the LSB term
                    var rca_result = F.zero();
                    const rca_xlen = XLEN;
                    for (0..rca_xlen - 1) |i| {
                        const shift: u6 = @intCast(rca_xlen - 1 - i);
                        const coeff = F.fromU64(@as(u64, 1) << shift);
                        rca_result = rca_result.add(coeff.mul(r[rca_xlen + i]));
                    }
                    break :blk_rca rca_result;
                },
                2 => And.evaluateMLE(r),
                3 => Andn.evaluateMLE(r),
                4 => Or.evaluateMLE(r),
                5 => Xor.evaluateMLE(r),
                6 => Equal.evaluateMLE(r),
                7 => SignedGreaterThanEqual.evaluateMLE(r),
                8 => UnsignedGreaterThanEqual.evaluateMLE(r),
                9 => NotEqual.evaluateMLE(r),
                10 => SignedLessThan.evaluateMLE(r),
                11 => UnsignedLessThan.evaluateMLE(r),
                12 => Movsign.evaluateMLE(r),
                13 => F.zero(), // UpperWord - TODO
                14 => UnsignedLessThanEqual.evaluateMLE(r),
                // 15 was ValidSignedRemainder (removed in PR #1355)
                15 => ValidUnsignedRemainder.evaluateMLE(r),
                16 => ValidDiv0.evaluateMLE(r),
                17 => HalfwordAlignment.evaluateMLE(r),
                18 => WordAlignment.evaluateMLE(r),
                19 => LowerHalfWord.evaluateMLE(r),
                20 => SignExtendHalfWord.evaluateMLE(r),
                21 => Pow2.evaluateMLE(r),
                22 => F.zero(), // Pow2W - TODO
                23 => ShiftRightBitmask.evaluateMLE(r),
                24 => F.zero(), // VirtualRev8W - TODO
                25 => VirtualSRL.evaluateMLE(r),
                26 => VirtualSRA.evaluateMLE(r),
                27 => F.zero(), // VirtualROTR - TODO
                28 => F.zero(), // VirtualROTRW - TODO
                29 => F.zero(), // VirtualChangeDivisor - TODO
                30 => F.zero(), // VirtualChangeDivisorW - TODO
                31 => F.zero(), // MulUNoOverflow - TODO
                32 => F.zero(), // VirtualXORROT32 - TODO
                33 => F.zero(), // VirtualXORROT24 - TODO
                34 => F.zero(), // VirtualXORROT16 - TODO
                35 => F.zero(), // VirtualXORROT63 - TODO
                36 => F.zero(), // VirtualXORROTW16 - TODO
                37 => F.zero(), // VirtualXORROTW12 - TODO
                38 => F.zero(), // VirtualXORROTW8 - TODO
                39 => F.zero(), // VirtualXORROTW7 - TODO
                else => F.zero(),
            };
        }

        /// Materialize a table entry at a given interleaved index (u128).
        /// Returns the u64 table output for the given lookup index.
        /// Used for diagnostics: verifying that combined_vals lookup_output matches the table.
        pub fn materializeTableEntry(table_index: usize, index: u128) u64 {
            // Uninterleave for interleaved-path tables
            const even_bits = blk: {
                var x: u64 = 0;
                inline for (0..XLEN) |i| {
                    x |= @as(u64, @truncate((index >> (2 * i)) & 1)) << i;
                }
                break :blk x;
            };
            const odd_bits = blk: {
                var y: u64 = 0;
                inline for (0..XLEN) |i| {
                    y |= @as(u64, @truncate((index >> (2 * i + 1)) & 1)) << i;
                }
                break :blk y;
            };
            return switch (table_index) {
                // Convention: interleaveBits128(x, y) puts x at odd positions, y at even
                // So: odd_bits = x = first arg (typically rs1/left)
                //     even_bits = y = second arg (typically rs2/right)
                // uninterleave: x=odd_bits, y=even_bits
                0 => @truncate(index), // RangeCheck: identity, lower 64 bits
                1 => @truncate(index & ~@as(u128, 1)), // RangeCheckAligned: lower 64 bits with LSB cleared
                2 => odd_bits & even_bits, // And: x & y
                3 => (~odd_bits) & even_bits, // Andn: ~x & y
                4 => odd_bits | even_bits, // Or: x | y
                5 => odd_bits ^ even_bits, // Xor: x ^ y
                6 => @intFromBool(odd_bits == even_bits), // Equal: x == y
                7 => @intFromBool(@as(i64, @bitCast(odd_bits)) >= @as(i64, @bitCast(even_bits))), // SignedGreaterThanEqual: (signed)x >= (signed)y
                8 => @intFromBool(odd_bits >= even_bits), // UnsignedGreaterThanEqual: x >= y
                9 => @intFromBool(odd_bits != even_bits), // NotEqual: x != y
                10 => @intFromBool(@as(i64, @bitCast(odd_bits)) < @as(i64, @bitCast(even_bits))), // SignedLessThan: (signed)x < (signed)y
                11 => @intFromBool(odd_bits < even_bits), // UnsignedLessThan: x < y
                12 => blk12: { // Movsign: sign(y) ? ~x : x
                    const y_signed: i64 = @bitCast(even_bits);
                    break :blk12 if (y_signed < 0) ~odd_bits else odd_bits;
                },
                13 => blk13: { // UpperWord: upper 32 bits of 128-bit value
                    // For identity-path (MUL/MULHU), index is the raw u128 product
                    // Upper word = bits [127:64] of the full value
                    break :blk13 @truncate(index >> 64);
                },
                16 => ValidUnsignedRemainder.materializeEntry(index), // interleaved(remainder, divisor)
                20 => LowerHalfWord.materializeEntry(index), // identity: lower 32 bits
                21 => blk21: {
                    // SignExtendHalfWord: identity path, sign-extend lower 32 bits to 64 bits
                    const val: u64 = @truncate(index);
                    const lower32: u32 = @truncate(val);
                    const sign_extended: i64 = @as(i64, @as(i32, @bitCast(lower32)));
                    break :blk21 @bitCast(sign_extended);
                },
                17 => blk17: {
                    // ValidDiv0: interleaved(dividend, divisor)
                    // For XLEN=64: truncate to 32-bit signed values
                    // Returns 1 if dividend != INT32_MIN or divisor != -1 (i.e., no overflow)
                    // Also returns 1 if divisor == 0 (div-by-zero handled separately)
                    const dividend_u32: u32 = @truncate(odd_bits); // x = dividend
                    const divisor_u32: u32 = @truncate(even_bits); // y = divisor
                    const dividend_i32: i32 = @bitCast(dividend_u32);
                    const divisor_i32: i32 = @bitCast(divisor_u32);
                    break :blk17 if (dividend_i32 == std.math.minInt(i32) and divisor_i32 == -1) 0 else 1;
                },
                18 => HalfwordAlignment.materializeEntry(index), // identity path
                19 => WordAlignment.materializeEntry(index), // identity path
                22 => Pow2.materializeEntry(index), // identity path
                24 => ShiftRightBitmask.materializeEntry(index), // identity path
                26 => VirtualSRL.materializeEntry(index), // interleaved
                27 => blk27: {
                    // VirtualSRA: interleaved(value, bitmask)
                    // Jolt convention: x=odd_bits=value, y=even_bits=bitmask
                    // SRA output: value AND bitmask, OR sign_extension
                    // The bitmask selects the shifted bits, sign_extension fills upper bits
                    // But for materializeEntry, we compute directly:
                    // value & bitmask | (~bitmask & sign_extend)
                    // where sign_extend = (value >> 63) ? 0xFFFFFFFFFFFFFFFF : 0
                    // Actually, VirtualSRA MLE returns: x & y (like AND), but with sign extension
                    // From Jolt's VirtualSRATable materialize_entry:
                    // let (value, bitmask) = uninterleave_bits(index);
                    // let sign_bit = (value >> (XLEN-1)) & 1;
                    // if sign_bit == 1 { (value & bitmask) | !bitmask } else { value & bitmask }
                    const value = odd_bits; // x
                    const bitmask = even_bits; // y
                    const sign_bit = (value >> (XLEN - 1)) & 1;
                    break :blk27 if (sign_bit == 1) (value & bitmask) | ~bitmask else value & bitmask;
                },
                31 => blk31: {
                    // VirtualChangeDivisorW: interleaved(dividend, divisor)
                    // For XLEN=64: truncate to 32-bit signed values
                    // If dividend == INT32_MIN and divisor == -1: return 1
                    // Otherwise: return divisor sign-extended to 64 bits
                    const dividend_u32: u32 = @truncate(odd_bits); // x = dividend
                    const divisor_u32: u32 = @truncate(even_bits); // y = divisor
                    const dividend_i32: i32 = @bitCast(dividend_u32);
                    const divisor_i32: i32 = @bitCast(divisor_u32);
                    break :blk31 if (dividend_i32 == std.math.minInt(i32) and divisor_i32 == -1)
                        1
                    else
                        @bitCast(@as(i64, divisor_i32));
                },
                else => 0, // Unimplemented tables return 0
            };
        }

        pub const ValidSignedRemainder = struct {
            /// Materialize the entry at the given index
            /// Index format: interleaved(remainder, divisor)
            pub fn materializeEntry(index: u128) u64 {
                const operands = uninterleaveBits(index);
                const x = operands.x; // remainder
                const y = operands.y; // divisor

                if (XLEN == 64) {
                    const remainder: i64 = @bitCast(x);
                    const divisor: i64 = @bitCast(y);

                    if (remainder == 0 or divisor == 0) {
                        return 1;
                    }

                    // Check: |remainder| < |divisor| and same sign
                    const rem_abs = if (remainder < 0) @as(u64, @bitCast(-remainder)) else @as(u64, @bitCast(remainder));
                    const div_abs = if (divisor < 0) @as(u64, @bitCast(-divisor)) else @as(u64, @bitCast(divisor));
                    const rem_sign = remainder >> 63;
                    const div_sign = divisor >> 63;

                    return if (rem_abs < div_abs and rem_sign == div_sign) 1 else 0;
                } else if (XLEN == 32) {
                    const remainder: i32 = @truncate(@as(i64, @bitCast(x)));
                    const divisor: i32 = @truncate(@as(i64, @bitCast(y)));

                    if (remainder == 0 or divisor == 0) {
                        return 1;
                    }

                    const rem_abs = if (remainder < 0) @as(u32, @bitCast(-remainder)) else @as(u32, @bitCast(remainder));
                    const div_abs = if (divisor < 0) @as(u32, @bitCast(-divisor)) else @as(u32, @bitCast(divisor));
                    const rem_sign = remainder >> 31;
                    const div_sign = divisor >> 31;

                    return if (rem_abs < div_abs and rem_sign == div_sign) 1 else 0;
                } else if (XLEN == 8) {
                    const remainder: i8 = @truncate(@as(i64, @bitCast(x)));
                    const divisor: i8 = @truncate(@as(i64, @bitCast(y)));

                    if (remainder == 0 or divisor == 0) {
                        return 1;
                    }

                    const rem_abs = if (remainder < 0) @as(u8, @bitCast(-remainder)) else @as(u8, @bitCast(remainder));
                    const div_abs = if (divisor < 0) @as(u8, @bitCast(-divisor)) else @as(u8, @bitCast(divisor));
                    const rem_sign = remainder >> 7;
                    const div_sign = divisor >> 7;

                    return if (rem_abs < div_abs and rem_sign == div_sign) 1 else 0;
                } else {
                    // Generic fallback
                    return 0;
                }
            }

            /// Evaluate the MLE at point r
            pub fn evaluateMLE(r: []const F) F {
                std.debug.assert(r.len >= 2 * XLEN);

                const x_sign = r[0]; // remainder sign bit
                const y_sign = r[1]; // divisor sign bit

                var remainder_is_zero = F.one().sub(r[0]);
                var divisor_is_zero = F.one().sub(r[1]);
                var positive_remainder_equals_divisor = F.one().sub(x_sign).mul(F.one().sub(y_sign));
                var positive_remainder_less_than_divisor = F.one().sub(x_sign).mul(F.one().sub(y_sign));
                var negative_divisor_equals_remainder = x_sign.mul(y_sign);
                var negative_divisor_greater_than_remainder = x_sign.mul(y_sign);

                for (1..XLEN) |i| {
                    const x_i = r[2 * i];
                    const y_i = r[2 * i + 1];

                    if (i == 1) {
                        positive_remainder_less_than_divisor = positive_remainder_less_than_divisor.mul(F.one().sub(x_i).mul(y_i));
                        negative_divisor_greater_than_remainder = negative_divisor_greater_than_remainder.mul(x_i.mul(F.one().sub(y_i)));
                    } else {
                        positive_remainder_less_than_divisor = positive_remainder_less_than_divisor.add(
                            positive_remainder_equals_divisor.mul(F.one().sub(x_i).mul(y_i))
                        );
                        negative_divisor_greater_than_remainder = negative_divisor_greater_than_remainder.add(
                            negative_divisor_equals_remainder.mul(x_i.mul(F.one().sub(y_i)))
                        );
                    }

                    positive_remainder_equals_divisor = positive_remainder_equals_divisor.mul(
                        x_i.mul(y_i).add(F.one().sub(x_i).mul(F.one().sub(y_i)))
                    );
                    negative_divisor_equals_remainder = negative_divisor_equals_remainder.mul(
                        x_i.mul(y_i).add(F.one().sub(x_i).mul(F.one().sub(y_i)))
                    );
                    remainder_is_zero = remainder_is_zero.mul(F.one().sub(x_i));
                    divisor_is_zero = divisor_is_zero.mul(F.one().sub(y_i));
                }

                return positive_remainder_less_than_divisor
                    .add(negative_divisor_greater_than_remainder)
                    .add(y_sign.mul(remainder_is_zero))
                    .add(divisor_is_zero);
            }
        };
    };
}

// ============================================================================
// Tests
// ============================================================================

test "uninterleave and interleave bits" {
    // Test basic interleaving/uninterleaving
    const x: u64 = 0b1010; // 10
    const y: u64 = 0b1100; // 12

    const interleaved = interleaveBits(x, y);
    const result = uninterleaveBits(interleaved);

    try std.testing.expectEqual(x, result.x);
    try std.testing.expectEqual(y, result.y);
}

test "uninterleave bits pattern" {
    // Test specific bit patterns
    // Jolt format: y bits at even positions, x bits at odd positions
    // interleaved: y[0], x[0], y[1], x[1], ...
    // 0b11_10_01_00 = y=0b1010=10, x=0b1100=12
    const index: u128 = 0b11_10_01_00;
    const result = uninterleaveBits(index);

    try std.testing.expectEqual(@as(u64, 0b1100), result.x);
    try std.testing.expectEqual(@as(u64, 0b1010), result.y);
}

test "RangeCheck materialize" {
    const Table = LookupTable(BN254Scalar, 8);

    try std.testing.expectEqual(@as(u64, 0), Table.RangeCheck.materializeEntry(0));
    try std.testing.expectEqual(@as(u64, 42), Table.RangeCheck.materializeEntry(42));
    try std.testing.expectEqual(@as(u64, 255), Table.RangeCheck.materializeEntry(255));
    // Should wrap for 8-bit
    try std.testing.expectEqual(@as(u64, 0), Table.RangeCheck.materializeEntry(256));
}

test "And materialize" {
    const Table = LookupTable(BN254Scalar, 8);

    // 5 & 3 = 1 (0101 & 0011 = 0001)
    const index = interleaveBits(5, 3);
    try std.testing.expectEqual(@as(u64, 1), Table.And.materializeEntry(index));

    // 0xFF & 0x0F = 0x0F
    const index2 = interleaveBits(0xFF, 0x0F);
    try std.testing.expectEqual(@as(u64, 0x0F), Table.And.materializeEntry(index2));
}

test "Or materialize" {
    const Table = LookupTable(BN254Scalar, 8);

    // 5 | 3 = 7 (0101 | 0011 = 0111)
    const index = interleaveBits(5, 3);
    try std.testing.expectEqual(@as(u64, 7), Table.Or.materializeEntry(index));
}

test "Xor materialize" {
    const Table = LookupTable(BN254Scalar, 8);

    // 5 ^ 3 = 6 (0101 ^ 0011 = 0110)
    const index = interleaveBits(5, 3);
    try std.testing.expectEqual(@as(u64, 6), Table.Xor.materializeEntry(index));
}

test "Equal materialize" {
    const Table = LookupTable(BN254Scalar, 8);

    // 5 == 5 = 1
    const index1 = interleaveBits(5, 5);
    try std.testing.expectEqual(@as(u64, 1), Table.Equal.materializeEntry(index1));

    // 5 == 3 = 0
    const index2 = interleaveBits(5, 3);
    try std.testing.expectEqual(@as(u64, 0), Table.Equal.materializeEntry(index2));
}

test "UnsignedLessThan materialize" {
    const Table = LookupTable(BN254Scalar, 8);

    // 3 < 5 = 1
    const index1 = interleaveBits(3, 5);
    try std.testing.expectEqual(@as(u64, 1), Table.UnsignedLessThan.materializeEntry(index1));

    // 5 < 3 = 0
    const index2 = interleaveBits(5, 3);
    try std.testing.expectEqual(@as(u64, 0), Table.UnsignedLessThan.materializeEntry(index2));

    // 5 < 5 = 0
    const index3 = interleaveBits(5, 5);
    try std.testing.expectEqual(@as(u64, 0), Table.UnsignedLessThan.materializeEntry(index3));
}

test "SignedLessThan materialize" {
    const Table = LookupTable(BN254Scalar, 64);

    // -1 < 1 = 1 (signed)
    const neg_one: u64 = @bitCast(@as(i64, -1));
    const index1 = interleaveBits(neg_one, 1);
    try std.testing.expectEqual(@as(u64, 1), Table.SignedLessThan.materializeEntry(index1));

    // 1 < -1 = 0 (signed)
    const index2 = interleaveBits(1, neg_one);
    try std.testing.expectEqual(@as(u64, 0), Table.SignedLessThan.materializeEntry(index2));

    // -5 < -1 = 1 (signed)
    const neg_five: u64 = @bitCast(@as(i64, -5));
    const index3 = interleaveBits(neg_five, neg_one);
    try std.testing.expectEqual(@as(u64, 1), Table.SignedLessThan.materializeEntry(index3));
}

test "NotEqual materialize" {
    const Table = LookupTable(BN254Scalar, 8);

    // 5 != 5 = 0
    const index1 = interleaveBits(5, 5);
    try std.testing.expectEqual(@as(u64, 0), Table.NotEqual.materializeEntry(index1));

    // 5 != 3 = 1
    const index2 = interleaveBits(5, 3);
    try std.testing.expectEqual(@as(u64, 1), Table.NotEqual.materializeEntry(index2));
}

test "And MLE on boolean hypercube" {
    // Test that MLE agrees with materialize on boolean inputs
    const Table = LookupTable(BN254Scalar, 2);

    // For a 2-bit table, we have 4 variables (2*XLEN)
    // Test all 16 combinations of boolean inputs
    // r array uses MSB-first ordering (like Jolt's index_to_field_bitvector)
    var i: u8 = 0;
    while (i < 16) : (i += 1) {
        var r: [4]BN254Scalar = undefined;
        // MSB-first: r[0] = bit 3, r[1] = bit 2, r[2] = bit 1, r[3] = bit 0
        inline for (0..4) |j| {
            const bit_pos = 4 - 1 - j; // MSB first
            r[j] = if ((i >> bit_pos) & 1 == 1) BN254Scalar.one() else BN254Scalar.zero();
        }

        const mle_result = Table.And.evaluateMLE(&r);
        const expected = Table.And.materializeEntry(i);
        const expected_field = BN254Scalar.fromU64(expected);

        try std.testing.expect(mle_result.eql(expected_field));
    }
}

test "Xor MLE on boolean hypercube" {
    const Table = LookupTable(BN254Scalar, 2);

    var i: u8 = 0;
    while (i < 16) : (i += 1) {
        var r: [4]BN254Scalar = undefined;
        inline for (0..4) |j| {
            const bit_pos = 4 - 1 - j;
            r[j] = if ((i >> bit_pos) & 1 == 1) BN254Scalar.one() else BN254Scalar.zero();
        }

        const mle_result = Table.Xor.evaluateMLE(&r);
        const expected = Table.Xor.materializeEntry(i);
        const expected_field = BN254Scalar.fromU64(expected);

        try std.testing.expect(mle_result.eql(expected_field));
    }
}

test "Or MLE on boolean hypercube" {
    const Table = LookupTable(BN254Scalar, 2);

    var i: u8 = 0;
    while (i < 16) : (i += 1) {
        var r: [4]BN254Scalar = undefined;
        inline for (0..4) |j| {
            const bit_pos = 4 - 1 - j;
            r[j] = if ((i >> bit_pos) & 1 == 1) BN254Scalar.one() else BN254Scalar.zero();
        }

        const mle_result = Table.Or.evaluateMLE(&r);
        const expected = Table.Or.materializeEntry(i);
        const expected_field = BN254Scalar.fromU64(expected);

        try std.testing.expect(mle_result.eql(expected_field));
    }
}

test "Equal MLE on boolean hypercube" {
    const Table = LookupTable(BN254Scalar, 2);

    var i: u8 = 0;
    while (i < 16) : (i += 1) {
        var r: [4]BN254Scalar = undefined;
        inline for (0..4) |j| {
            const bit_pos = 4 - 1 - j;
            r[j] = if ((i >> bit_pos) & 1 == 1) BN254Scalar.one() else BN254Scalar.zero();
        }

        const mle_result = Table.Equal.evaluateMLE(&r);
        const expected = Table.Equal.materializeEntry(i);
        const expected_field = BN254Scalar.fromU64(expected);

        try std.testing.expect(mle_result.eql(expected_field));
    }
}

test "UnsignedLessThan MLE on boolean hypercube" {
    const Table = LookupTable(BN254Scalar, 2);

    var i: u8 = 0;
    while (i < 16) : (i += 1) {
        var r: [4]BN254Scalar = undefined;
        inline for (0..4) |j| {
            const bit_pos = 4 - 1 - j;
            r[j] = if ((i >> bit_pos) & 1 == 1) BN254Scalar.one() else BN254Scalar.zero();
        }

        const mle_result = Table.UnsignedLessThan.evaluateMLE(&r);
        const expected = Table.UnsignedLessThan.materializeEntry(i);
        const expected_field = BN254Scalar.fromU64(expected);

        try std.testing.expect(mle_result.eql(expected_field));
    }
}

test "RangeCheck MLE on boolean hypercube" {
    const Table = LookupTable(BN254Scalar, 2);

    var i: u8 = 0;
    while (i < 16) : (i += 1) {
        var r: [4]BN254Scalar = undefined;
        inline for (0..4) |j| {
            const bit_pos = 4 - 1 - j;
            r[j] = if ((i >> bit_pos) & 1 == 1) BN254Scalar.one() else BN254Scalar.zero();
        }

        const mle_result = Table.RangeCheck.evaluateMLE(&r);
        const expected = Table.RangeCheck.materializeEntry(i);
        const expected_field = BN254Scalar.fromU64(expected);

        try std.testing.expect(mle_result.eql(expected_field));
    }
}

test "Andn MLE on boolean hypercube" {
    const Table = LookupTable(BN254Scalar, 2);

    var i: u8 = 0;
    while (i < 16) : (i += 1) {
        var r: [4]BN254Scalar = undefined;
        inline for (0..4) |j| {
            const bit_pos = 4 - 1 - j;
            r[j] = if ((i >> bit_pos) & 1 == 1) BN254Scalar.one() else BN254Scalar.zero();
        }

        const mle_result = Table.Andn.evaluateMLE(&r);
        const expected = Table.Andn.materializeEntry(i);
        const expected_field = BN254Scalar.fromU64(expected);

        try std.testing.expect(mle_result.eql(expected_field));
    }
}

test "Andn materialize" {
    const Table = LookupTable(BN254Scalar, 8);

    // 0xFF & ~0x0F = 0xF0
    const index1 = interleaveBits(0xFF, 0x0F);
    try std.testing.expectEqual(@as(u64, 0xF0), Table.Andn.materializeEntry(index1));

    // 0x55 & ~0xAA = 0x55 (0101 & ~1010 = 0101)
    const index2 = interleaveBits(0x55, 0xAA);
    try std.testing.expectEqual(@as(u64, 0x55), Table.Andn.materializeEntry(index2));
}

test "UnsignedGreaterThanEqual materialize" {
    const Table = LookupTable(BN254Scalar, 8);

    // 5 >= 3 = 1
    const index1 = interleaveBits(5, 3);
    try std.testing.expectEqual(@as(u64, 1), Table.UnsignedGreaterThanEqual.materializeEntry(index1));

    // 3 >= 5 = 0
    const index2 = interleaveBits(3, 5);
    try std.testing.expectEqual(@as(u64, 0), Table.UnsignedGreaterThanEqual.materializeEntry(index2));

    // 5 >= 5 = 1
    const index3 = interleaveBits(5, 5);
    try std.testing.expectEqual(@as(u64, 1), Table.UnsignedGreaterThanEqual.materializeEntry(index3));
}

test "UnsignedLessThanEqual materialize" {
    const Table = LookupTable(BN254Scalar, 8);

    // 3 <= 5 = 1
    const index1 = interleaveBits(3, 5);
    try std.testing.expectEqual(@as(u64, 1), Table.UnsignedLessThanEqual.materializeEntry(index1));

    // 5 <= 3 = 0
    const index2 = interleaveBits(5, 3);
    try std.testing.expectEqual(@as(u64, 0), Table.UnsignedLessThanEqual.materializeEntry(index2));

    // 5 <= 5 = 1
    const index3 = interleaveBits(5, 5);
    try std.testing.expectEqual(@as(u64, 1), Table.UnsignedLessThanEqual.materializeEntry(index3));
}

test "UnsignedLessThanEqual MLE on boolean hypercube" {
    const Table = LookupTable(BN254Scalar, 2);

    var i: u8 = 0;
    while (i < 16) : (i += 1) {
        var r: [4]BN254Scalar = undefined;
        inline for (0..4) |j| {
            const bit_pos = 4 - 1 - j;
            r[j] = if ((i >> bit_pos) & 1 == 1) BN254Scalar.one() else BN254Scalar.zero();
        }

        const mle_result = Table.UnsignedLessThanEqual.evaluateMLE(&r);
        const expected = Table.UnsignedLessThanEqual.materializeEntry(i);
        const expected_field = BN254Scalar.fromU64(expected);

        try std.testing.expect(mle_result.eql(expected_field));
    }
}

test "SignedGreaterThanEqual materialize" {
    const Table = LookupTable(BN254Scalar, 64);

    // -1 >= 1 = 0 (signed)
    const neg_one: u64 = @bitCast(@as(i64, -1));
    const index1 = interleaveBits(neg_one, 1);
    try std.testing.expectEqual(@as(u64, 0), Table.SignedGreaterThanEqual.materializeEntry(index1));

    // 1 >= -1 = 1 (signed)
    const index2 = interleaveBits(1, neg_one);
    try std.testing.expectEqual(@as(u64, 1), Table.SignedGreaterThanEqual.materializeEntry(index2));

    // 5 >= 5 = 1
    const index3 = interleaveBits(5, 5);
    try std.testing.expectEqual(@as(u64, 1), Table.SignedGreaterThanEqual.materializeEntry(index3));
}

test "Movsign materialize" {
    const Table = LookupTable(BN254Scalar, 64);

    // Positive number: MSB = 0
    const index1 = interleaveBits(100, 0);
    try std.testing.expectEqual(@as(u64, 0), Table.Movsign.materializeEntry(index1));

    // Negative number: MSB = 1
    const neg: u64 = @bitCast(@as(i64, -1));
    const index2 = interleaveBits(neg, 0);
    try std.testing.expectEqual(@as(u64, 1), Table.Movsign.materializeEntry(index2));
}

test "Sub materialize" {
    const Table = LookupTable(BN254Scalar, 64);

    // 10 - 3 = 7
    const index1 = interleaveBits(10, 3);
    try std.testing.expectEqual(@as(u64, 7), Table.Sub.materializeEntry(index1));

    // 3 - 10 = wrapping subtraction in 64-bit
    const index2 = interleaveBits(3, 10);
    const expected: u64 = @as(u64, 3) -% @as(u64, 10);
    try std.testing.expectEqual(expected, Table.Sub.materializeEntry(index2));

    // 5 - 5 = 0
    const index3 = interleaveBits(5, 5);
    try std.testing.expectEqual(@as(u64, 0), Table.Sub.materializeEntry(index3));
}

test "LeftShift materialize" {
    const Table = LookupTable(BN254Scalar, 8);

    // 1 << 0 = 1
    const index0 = interleaveBits(1, 0);
    try std.testing.expectEqual(@as(u64, 1), Table.LeftShift.materializeEntry(index0));

    // 1 << 1 = 2
    const index1 = interleaveBits(1, 1);
    try std.testing.expectEqual(@as(u64, 2), Table.LeftShift.materializeEntry(index1));

    // 1 << 7 = 128
    const index2 = interleaveBits(1, 7);
    try std.testing.expectEqual(@as(u64, 128), Table.LeftShift.materializeEntry(index2));

    // 0xFF << 1 = 0xFE (shifted by 1 in 8-bit)
    const index3 = interleaveBits(0xFF, 1);
    try std.testing.expectEqual(@as(u64, 0xFE), Table.LeftShift.materializeEntry(index3));

    // 5 << 3 = 40
    const index4 = interleaveBits(5, 3);
    try std.testing.expectEqual(@as(u64, 40), Table.LeftShift.materializeEntry(index4));
}

test "RightShift materialize" {
    const Table = LookupTable(BN254Scalar, 8);

    // 128 >> 0 = 128
    const index0 = interleaveBits(128, 0);
    try std.testing.expectEqual(@as(u64, 128), Table.RightShift.materializeEntry(index0));

    // 128 >> 1 = 64
    const index1 = interleaveBits(128, 1);
    try std.testing.expectEqual(@as(u64, 64), Table.RightShift.materializeEntry(index1));

    // 128 >> 7 = 1
    const index2 = interleaveBits(128, 7);
    try std.testing.expectEqual(@as(u64, 1), Table.RightShift.materializeEntry(index2));

    // 0xFF >> 4 = 0x0F
    const index3 = interleaveBits(0xFF, 4);
    try std.testing.expectEqual(@as(u64, 0x0F), Table.RightShift.materializeEntry(index3));

    // 40 >> 3 = 5
    const index4 = interleaveBits(40, 3);
    try std.testing.expectEqual(@as(u64, 5), Table.RightShift.materializeEntry(index4));
}

test "RightShiftArithmetic materialize" {
    const Table = LookupTable(BN254Scalar, 8);

    // Positive number: 64 >> 2 = 16 (same as logical)
    const index1 = interleaveBits(64, 2);
    try std.testing.expectEqual(@as(u64, 16), Table.RightShiftArithmetic.materializeEntry(index1));

    // Negative number (8-bit): 0x80 (-128) >> 1 = 0xC0 (-64)
    // Sign bit extends
    const index2 = interleaveBits(0x80, 1);
    try std.testing.expectEqual(@as(u64, 0xC0), Table.RightShiftArithmetic.materializeEntry(index2));

    // 0xFF (-1 in signed 8-bit) >> 4 = 0xFF (-1)
    const index3 = interleaveBits(0xFF, 4);
    try std.testing.expectEqual(@as(u64, 0xFF), Table.RightShiftArithmetic.materializeEntry(index3));
}

test "Pow2 materialize" {
    const Table = LookupTable(BN254Scalar, 8);

    // 2^0 = 1
    try std.testing.expectEqual(@as(u64, 1), Table.Pow2.materializeEntry(0));

    // 2^1 = 2
    try std.testing.expectEqual(@as(u64, 2), Table.Pow2.materializeEntry(1));

    // 2^4 = 16
    try std.testing.expectEqual(@as(u64, 16), Table.Pow2.materializeEntry(4));

    // 2^7 = 128
    try std.testing.expectEqual(@as(u64, 128), Table.Pow2.materializeEntry(7));
}

test "SignExtend8 materialize" {
    const Table = LookupTable(BN254Scalar, 64);

    // Positive: 100 stays 100
    try std.testing.expectEqual(@as(u64, 100), Table.SignExtend8.materializeEntry(100));

    // Negative: 0x80 (-128) becomes 0xFFFFFFFFFFFFFF80
    const expected_neg: u64 = @bitCast(@as(i64, -128));
    try std.testing.expectEqual(expected_neg, Table.SignExtend8.materializeEntry(0x80));

    // -1 (0xFF) becomes full -1
    const expected_neg1: u64 = @bitCast(@as(i64, -1));
    try std.testing.expectEqual(expected_neg1, Table.SignExtend8.materializeEntry(0xFF));
}

test "SignExtend16 materialize" {
    const Table = LookupTable(BN254Scalar, 64);

    // Positive: 1000 stays 1000
    try std.testing.expectEqual(@as(u64, 1000), Table.SignExtend16.materializeEntry(1000));

    // Negative: 0x8000 (-32768) becomes sign extended
    const expected_neg: u64 = @bitCast(@as(i64, -32768));
    try std.testing.expectEqual(expected_neg, Table.SignExtend16.materializeEntry(0x8000));
}

test "SignExtend32 materialize" {
    const Table = LookupTable(BN254Scalar, 64);

    // Positive: 100000 stays 100000
    try std.testing.expectEqual(@as(u64, 100000), Table.SignExtend32.materializeEntry(100000));

    // Negative: 0x80000000 (-2^31) becomes sign extended
    const expected_neg: u64 = @bitCast(@as(i64, -2147483648));
    try std.testing.expectEqual(expected_neg, Table.SignExtend32.materializeEntry(0x80000000));
}

test "ValidDiv0 materialize" {
    const Table = LookupTable(BN254Scalar, 8);

    // divisor != 0: always valid
    const idx1 = interleaveBits(5, 2); // divisor=5, quotient=2
    try std.testing.expectEqual(@as(u64, 1), Table.ValidDiv0.materializeEntry(idx1));

    // divisor = 0, quotient = MAX (255 for 8-bit): valid
    const idx2 = interleaveBits(0, 255); // divisor=0, quotient=255
    try std.testing.expectEqual(@as(u64, 1), Table.ValidDiv0.materializeEntry(idx2));

    // divisor = 0, quotient != MAX: invalid
    const idx3 = interleaveBits(0, 100); // divisor=0, quotient=100
    try std.testing.expectEqual(@as(u64, 0), Table.ValidDiv0.materializeEntry(idx3));
}

test "ValidDiv0 MLE on boolean hypercube" {
    const Table = LookupTable(BN254Scalar, 2);

    // Test all combinations for 2-bit inputs (16 total)
    var i: u8 = 0;
    while (i < 16) : (i += 1) {
        var r: [4]BN254Scalar = undefined;
        inline for (0..4) |j| {
            const bit_pos = 4 - 1 - j;
            r[j] = if ((i >> bit_pos) & 1 == 1) BN254Scalar.one() else BN254Scalar.zero();
        }

        const mle_result = Table.ValidDiv0.evaluateMLE(&r);
        const expected = Table.ValidDiv0.materializeEntry(i);
        const expected_field = BN254Scalar.fromU64(expected);

        try std.testing.expect(mle_result.eql(expected_field));
    }
}

test "ValidUnsignedRemainder materialize" {
    const Table = LookupTable(BN254Scalar, 8);

    // divisor = 0: always valid
    const idx1 = interleaveBits(42, 0); // remainder=42, divisor=0
    try std.testing.expectEqual(@as(u64, 1), Table.ValidUnsignedRemainder.materializeEntry(idx1));

    // remainder < divisor: valid
    const idx2 = interleaveBits(3, 10); // remainder=3, divisor=10
    try std.testing.expectEqual(@as(u64, 1), Table.ValidUnsignedRemainder.materializeEntry(idx2));

    // remainder >= divisor: invalid
    const idx3 = interleaveBits(10, 5); // remainder=10, divisor=5
    try std.testing.expectEqual(@as(u64, 0), Table.ValidUnsignedRemainder.materializeEntry(idx3));

    // remainder == divisor: invalid
    const idx4 = interleaveBits(7, 7); // remainder=7, divisor=7
    try std.testing.expectEqual(@as(u64, 0), Table.ValidUnsignedRemainder.materializeEntry(idx4));
}

test "ValidUnsignedRemainder MLE on boolean hypercube" {
    const Table = LookupTable(BN254Scalar, 2);

    var i: u8 = 0;
    while (i < 16) : (i += 1) {
        var r: [4]BN254Scalar = undefined;
        inline for (0..4) |j| {
            const bit_pos = 4 - 1 - j;
            r[j] = if ((i >> bit_pos) & 1 == 1) BN254Scalar.one() else BN254Scalar.zero();
        }

        const mle_result = Table.ValidUnsignedRemainder.evaluateMLE(&r);
        const expected = Table.ValidUnsignedRemainder.materializeEntry(i);
        const expected_field = BN254Scalar.fromU64(expected);

        try std.testing.expect(mle_result.eql(expected_field));
    }
}

test "ValidSignedRemainder materialize" {
    const Table = LookupTable(BN254Scalar, 8);

    // divisor = 0: always valid
    const idx1 = interleaveBits(42, 0);
    try std.testing.expectEqual(@as(u64, 1), Table.ValidSignedRemainder.materializeEntry(idx1));

    // remainder = 0: always valid
    const idx2 = interleaveBits(0, 10);
    try std.testing.expectEqual(@as(u64, 1), Table.ValidSignedRemainder.materializeEntry(idx2));

    // Both positive, |remainder| < |divisor|: valid
    const idx3 = interleaveBits(3, 10);
    try std.testing.expectEqual(@as(u64, 1), Table.ValidSignedRemainder.materializeEntry(idx3));

    // Both negative, |remainder| < |divisor|: valid
    // -3 (0xFD) and -10 (0xF6) as 8-bit signed
    const neg3: u64 = @as(u8, @bitCast(@as(i8, -3)));
    const neg10: u64 = @as(u8, @bitCast(@as(i8, -10)));
    const idx4 = interleaveBits(neg3, neg10);
    try std.testing.expectEqual(@as(u64, 1), Table.ValidSignedRemainder.materializeEntry(idx4));

    // Different signs: invalid
    const idx5 = interleaveBits(3, neg10);
    try std.testing.expectEqual(@as(u64, 0), Table.ValidSignedRemainder.materializeEntry(idx5));
}
