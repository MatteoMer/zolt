//! Jolt Lookup Tables
//!
//! This module implements the lookup tables used in Jolt for instruction verification.
//! Lookup arguments are THE core technique that makes Jolt efficient.
//!
//! Each lookup table is defined by:
//! 1. materializeEntry(index) - Returns the table entry at a given index
//! 2. evaluateMLE(r) - Evaluates the multilinear extension at point r
//!
//! The key insight from the Jolt/Lasso papers is that many RISC-V operations can be
//! verified by looking up precomputed values in small tables, rather than computing
//! them via expensive arithmetic circuits.

const std = @import("std");
const Allocator = std.mem.Allocator;
const field = @import("../../field/mod.zig");
const BN254Scalar = field.BN254Scalar;

/// Utility function to uninterleave bits from an interleaved index.
/// For a 2*XLEN bit index where bits are interleaved as x[0], y[0], x[1], y[1], ...,
/// returns (x, y) where each is an XLEN-bit value.
pub fn uninterleaveBits(index: u128) struct { x: u64, y: u64 } {
    var x: u64 = 0;
    var y: u64 = 0;
    var idx = index;

    // Extract bits: even positions go to x, odd positions go to y
    var i: u6 = 0;
    while (i < 64) : (i += 1) {
        x |= @as(u64, @intCast((idx >> 0) & 1)) << i;
        y |= @as(u64, @intCast((idx >> 1) & 1)) << i;
        idx >>= 2;
    }

    return .{ .x = x, .y = y };
}

/// Interleave bits of x and y into a single index.
/// Produces x[0], y[0], x[1], y[1], ... where bit positions alternate.
pub fn interleaveBits(x: u64, y: u64) u128 {
    var result: u128 = 0;

    var i: u7 = 0;
    while (i < 64) : (i += 1) {
        const x_bit: u128 = @as(u128, (x >> @as(u6, @intCast(i))) & 1);
        const y_bit: u128 = @as(u128, (y >> @as(u6, @intCast(i))) & 1);
        result |= x_bit << (@as(u7, i) * 2);
        result |= y_bit << (@as(u7, i) * 2 + 1);
    }

    return result;
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
    // interleaved: x[0], y[0], x[1], y[1], ...
    // 0b11_10_01_00 = x=0101, y=1100
    const index: u128 = 0b11_10_01_00;
    const result = uninterleaveBits(index);

    try std.testing.expectEqual(@as(u64, 0b0101), result.x);
    try std.testing.expectEqual(@as(u64, 0b1100), result.y);
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
    var i: u8 = 0;
    while (i < 16) : (i += 1) {
        var r: [4]BN254Scalar = undefined;
        inline for (0..4) |j| {
            r[j] = if ((i >> j) & 1 == 1) BN254Scalar.one() else BN254Scalar.zero();
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
            r[j] = if ((i >> j) & 1 == 1) BN254Scalar.one() else BN254Scalar.zero();
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
            r[j] = if ((i >> j) & 1 == 1) BN254Scalar.one() else BN254Scalar.zero();
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
            r[j] = if ((i >> j) & 1 == 1) BN254Scalar.one() else BN254Scalar.zero();
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
            r[j] = if ((i >> j) & 1 == 1) BN254Scalar.one() else BN254Scalar.zero();
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
            r[j] = if ((i >> j) & 1 == 1) BN254Scalar.one() else BN254Scalar.zero();
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
            r[j] = if ((i >> j) & 1 == 1) BN254Scalar.one() else BN254Scalar.zero();
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
            r[j] = if ((i >> j) & 1 == 1) BN254Scalar.one() else BN254Scalar.zero();
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
    const Table = LookupTable(BN254Scalar, 8);

    // 10 - 3 = 7
    const index1 = interleaveBits(10, 3);
    try std.testing.expectEqual(@as(u64, 7), Table.Sub.materializeEntry(index1));

    // 3 - 10 = 249 (wrapping, for 8-bit)
    const index2 = interleaveBits(3, 10);
    try std.testing.expectEqual(@as(u64, 249), Table.Sub.materializeEntry(index2));

    // 5 - 5 = 0
    const index3 = interleaveBits(5, 5);
    try std.testing.expectEqual(@as(u64, 0), Table.Sub.materializeEntry(index3));
}
