//! Shared EQ polynomial evaluation utilities.
//!
//! The equality polynomial eq(r, x) = Π_i (r_i * x_i + (1 - r_i)(1 - x_i))
//! evaluates to 1 when r = x and 0 otherwise. Two variants exist depending on
//! the bit ordering convention used by different subsystems.

const std = @import("std");

/// Evaluate eq(r, k) with BIG-ENDIAN bit ordering.
/// r[0] corresponds to the MSB (bit n-1) of k.
/// Used by: Spartan outer, Stage 5 instances, read-write checking.
pub fn computeEqAtPointBE(comptime F: type, r: []const F, k: usize) F {
    const n = r.len;
    var result = F.one();
    for (0..n) |i| {
        const bit = (k >> @intCast(n - 1 - i)) & 1;
        if (bit == 1) {
            result = result.mul(r[i]);
        } else {
            result = result.mul(F.one().sub(r[i]));
        }
    }
    return result;
}

/// Evaluate eq(r, k) with LITTLE-ENDIAN bit ordering.
/// r[i] corresponds to bit i of k (LSB first).
/// Used by: RAM RAF checking, RAM/registers value evaluation.
pub fn computeEqAtPointLE(comptime F: type, r: []const F, k: usize) F {
    var result = F.one();
    for (0..r.len) |i| {
        const bit = (k >> @intCast(i)) & 1;
        if (bit == 1) {
            result = result.mul(r[i]);
        } else {
            result = result.mul(F.one().sub(r[i]));
        }
    }
    return result;
}

test "eq BE basic" {
    const F = @import("zolt_arith").field.BN254Scalar;
    // r = [1, 0] (2 vars), k = 2 (binary 10)
    // eq([1,0], 2) = eq([1,0], [1,0]) = 1*1 * (1-0)*(1-0) = 1
    const r = [_]F{ F.one(), F.zero() };
    const result = computeEqAtPointBE(F, &r, 2);
    try std.testing.expect(result.eql(F.one()));
}

test "eq LE basic" {
    const F = @import("zolt_arith").field.BN254Scalar;
    // r = [0, 1] (2 vars LE), k = 2 (binary 10, so bit0=0, bit1=1)
    // eq([0,1], 2) = (1-0)*(1-0) * 1*1 = 1
    const r = [_]F{ F.zero(), F.one() };
    const result = computeEqAtPointLE(F, &r, 2);
    try std.testing.expect(result.eql(F.one()));
}
