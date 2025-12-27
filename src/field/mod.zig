//! Finite field arithmetic for Jolt
//!
//! This module provides field element types and operations for the cryptographic
//! protocols used in Jolt. The primary field is the BN254 scalar field.

const std = @import("std");

/// Number of bytes in a field element (256 bits = 32 bytes)
pub const FIELD_ELEMENT_BYTES: usize = 32;

/// Number of 64-bit limbs in a field element
pub const NUM_LIMBS: usize = 4;

/// BN254 scalar field modulus
/// p = 21888242871839275222246405745257275088548364400416034343698204186575808495617
pub const BN254_MODULUS: [4]u64 = .{
    0x43e1f593f0000001,
    0x2833e84879b97091,
    0xb85045b68181585d,
    0x30644e72e131a029,
};

/// Montgomery R for BN254 (R = 2^256 mod p)
pub const BN254_R: [4]u64 = .{
    0xac96341c4ffffffb,
    0x36fc76959f60cd29,
    0x666ea36f7879462e,
    0x0e0a77c19a07df2f,
};

/// Montgomery R^2 for BN254 (R^2 = 2^512 mod p)
pub const BN254_R2: [4]u64 = .{
    0x1bb8e645ae216da7,
    0x53fe3ab1e35c59e3,
    0x8c49833d53bb8085,
    0x0216d0b17f4e44a5,
};

/// Montgomery constant: -p^{-1} mod 2^64
/// Used in Montgomery reduction
pub const BN254_INV: u64 = 0xc2e1f593efffffff;

/// JoltField interface - the core trait for field elements
///
/// In Zig, we implement this as a comptime interface check pattern.
pub fn JoltField(comptime Self: type) type {
    return struct {
        pub const num_bytes = FIELD_ELEMENT_BYTES;

        pub fn isJoltField() void {
            // Compile-time check that Self has required methods
            comptime {
                if (!@hasDecl(Self, "zero")) @compileError("JoltField requires zero()");
                if (!@hasDecl(Self, "one")) @compileError("JoltField requires one()");
                if (!@hasDecl(Self, "add")) @compileError("JoltField requires add()");
                if (!@hasDecl(Self, "sub")) @compileError("JoltField requires sub()");
                if (!@hasDecl(Self, "mul")) @compileError("JoltField requires mul()");
                if (!@hasDecl(Self, "inverse")) @compileError("JoltField requires inverse()");
                if (!@hasDecl(Self, "square")) @compileError("JoltField requires square()");
                if (!@hasDecl(Self, "fromU64")) @compileError("JoltField requires fromU64()");
            }
        }
    };
}

/// BN254 scalar field element
/// Stored in Montgomery form: a is represented as a*R mod p
pub const BN254Scalar = struct {
    limbs: [4]u64,

    const Self = @This();

    /// Zero element
    pub fn zero() Self {
        return .{ .limbs = .{ 0, 0, 0, 0 } };
    }

    /// One element (in Montgomery form = R mod p)
    pub fn one() Self {
        return .{ .limbs = BN254_R };
    }

    /// Check if zero
    pub fn isZero(self: Self) bool {
        return self.limbs[0] == 0 and self.limbs[1] == 0 and
            self.limbs[2] == 0 and self.limbs[3] == 0;
    }

    /// Check if one (in Montgomery form)
    pub fn isOne(self: Self) bool {
        return self.limbs[0] == BN254_R[0] and self.limbs[1] == BN254_R[1] and
            self.limbs[2] == BN254_R[2] and self.limbs[3] == BN254_R[3];
    }

    /// Equality check
    pub fn eql(self: Self, other: Self) bool {
        return self.limbs[0] == other.limbs[0] and self.limbs[1] == other.limbs[1] and
            self.limbs[2] == other.limbs[2] and self.limbs[3] == other.limbs[3];
    }

    /// Create from u64 (converts to Montgomery form)
    pub fn fromU64(n: u64) Self {
        var result = Self{ .limbs = .{ n, 0, 0, 0 } };
        // Convert to Montgomery form by multiplying by R^2 and reducing
        result = result.montgomeryMul(.{ .limbs = BN254_R2 });
        return result;
    }

    /// Create from bytes (little-endian, converts to Montgomery form)
    pub fn fromBytes(bytes: []const u8) Self {
        var limbs: [4]u64 = .{ 0, 0, 0, 0 };
        const len = @min(bytes.len, 32);
        var buf: [32]u8 = .{0} ** 32;
        @memcpy(buf[0..len], bytes[0..len]);

        for (0..4) |i| {
            limbs[i] = std.mem.readInt(u64, buf[i * 8 ..][0..8], .little);
        }

        var result = Self{ .limbs = limbs };
        // Convert to Montgomery form
        result = result.montgomeryMul(.{ .limbs = BN254_R2 });
        return result;
    }

    /// Convert from Montgomery form back to standard representation
    pub fn fromMontgomery(self: Self) Self {
        // Multiply by 1 to get a*R * 1 * R^{-1} = a
        return self.montgomeryMul(.{ .limbs = .{ 1, 0, 0, 0 } });
    }

    /// 128-bit multiplication helper
    inline fn mulWide(a: u64, b: u64) u128 {
        return @as(u128, a) * @as(u128, b);
    }

    /// Add with carry
    inline fn addCarry(a: u64, b: u64, carry_in: u64) struct { result: u64, carry: u64 } {
        const sum = @as(u128, a) + @as(u128, b) + @as(u128, carry_in);
        return .{
            .result = @truncate(sum),
            .carry = @truncate(sum >> 64),
        };
    }

    /// Subtract with borrow
    inline fn subBorrow(a: u64, b: u64, borrow_in: u64) struct { result: u64, borrow: u64 } {
        const diff = @as(i128, a) - @as(i128, b) - @as(i128, borrow_in);
        if (diff < 0) {
            return .{
                .result = @truncate(@as(u128, @bitCast(diff + (@as(i128, 1) << 64)))),
                .borrow = 1,
            };
        }
        return .{
            .result = @truncate(@as(u128, @bitCast(diff))),
            .borrow = 0,
        };
    }

    /// Montgomery multiplication: computes a*b*R^{-1} mod p
    pub fn montgomeryMul(self: Self, other: Self) Self {
        // CIOS (Coarsely Integrated Operand Scanning) method
        var t: [5]u64 = .{ 0, 0, 0, 0, 0 };

        inline for (0..4) |i| {
            // t = t + a[i] * b
            var carry: u64 = 0;
            inline for (0..4) |j| {
                const prod = mulWide(self.limbs[i], other.limbs[j]);
                const sum = @as(u128, t[j]) + prod + @as(u128, carry);
                t[j] = @truncate(sum);
                carry = @truncate(sum >> 64);
            }
            const sum_t4 = @as(u128, t[4]) + @as(u128, carry);
            t[4] = @truncate(sum_t4);

            // m = t[0] * N' mod 2^64
            const m = t[0] *% BN254_INV;

            // t = (t + m * N) / 2^64
            carry = 0;
            const prod0 = mulWide(m, BN254_MODULUS[0]);
            const sum0 = @as(u128, t[0]) + prod0;
            carry = @truncate(sum0 >> 64);

            inline for (1..4) |j| {
                const prod = mulWide(m, BN254_MODULUS[j]);
                const sum = @as(u128, t[j]) + prod + @as(u128, carry);
                t[j - 1] = @truncate(sum);
                carry = @truncate(sum >> 64);
            }
            const final_sum = @as(u128, t[4]) + @as(u128, carry);
            t[3] = @truncate(final_sum);
            t[4] = @truncate(final_sum >> 64);
        }

        var result = Self{ .limbs = .{ t[0], t[1], t[2], t[3] } };

        // Final subtraction if result >= p
        if (t[4] != 0 or !result.lessThanModulus()) {
            result = result.subtractModulus();
        }

        return result;
    }

    /// Field addition
    pub fn add(self: Self, other: Self) Self {
        var result: [4]u64 = undefined;
        var carry: u64 = 0;

        inline for (0..4) |i| {
            const ac = addCarry(self.limbs[i], other.limbs[i], carry);
            result[i] = ac.result;
            carry = ac.carry;
        }

        var res = Self{ .limbs = result };
        // Reduce if >= p
        if (carry != 0 or !res.lessThanModulus()) {
            res = res.subtractModulus();
        }
        return res;
    }

    /// Field subtraction
    pub fn sub(self: Self, other: Self) Self {
        var result: [4]u64 = undefined;
        var borrow: u64 = 0;

        inline for (0..4) |i| {
            const sb = subBorrow(self.limbs[i], other.limbs[i], borrow);
            result[i] = sb.result;
            borrow = sb.borrow;
        }

        var res = Self{ .limbs = result };
        if (borrow != 0) {
            res = res.addModulus();
        }
        return res;
    }

    /// Field multiplication
    pub fn mul(self: Self, other: Self) Self {
        return self.montgomeryMul(other);
    }

    /// Field squaring (optimized)
    pub fn square(self: Self) Self {
        // For now, use regular multiplication
        // TODO: Implement optimized squaring
        return self.montgomeryMul(self);
    }

    /// Field negation: -a mod p
    pub fn neg(self: Self) Self {
        if (self.isZero()) return self;
        return (Self{ .limbs = BN254_MODULUS }).sub(self);
    }

    /// Field doubling: 2*a mod p
    pub fn double(self: Self) Self {
        return self.add(self);
    }

    /// Multiplicative inverse using Fermat's little theorem: a^{-1} = a^{p-2} mod p
    pub fn inverse(self: Self) ?Self {
        if (self.isZero()) return null;

        // p - 2 for BN254 scalar field
        // We use binary exponentiation
        const exp_minus_2: [4]u64 = .{
            0x43e1f593efffffff,
            0x2833e84879b97091,
            0xb85045b68181585d,
            0x30644e72e131a029,
        };

        var result = Self.one();
        var base = self;

        inline for (0..4) |i| {
            var bits = exp_minus_2[i];
            var j: usize = 0;
            while (j < 64) : (j += 1) {
                if ((bits & 1) != 0) {
                    result = result.mul(base);
                }
                base = base.square();
                bits >>= 1;
            }
        }

        return result;
    }

    /// Exponentiation: self^exp mod p
    pub fn pow(self: Self, exp: u64) Self {
        if (exp == 0) return Self.one();
        if (exp == 1) return self;

        var result = Self.one();
        var base = self;
        var e = exp;

        while (e > 0) {
            if ((e & 1) != 0) {
                result = result.mul(base);
            }
            base = base.square();
            e >>= 1;
        }

        return result;
    }

    fn lessThanModulus(self: Self) bool {
        var i: usize = 3;
        while (true) : (i -= 1) {
            if (self.limbs[i] < BN254_MODULUS[i]) return true;
            if (self.limbs[i] > BN254_MODULUS[i]) return false;
            if (i == 0) break;
        }
        return false;
    }

    fn subtractModulus(self: Self) Self {
        var result: [4]u64 = undefined;
        var borrow: u64 = 0;

        inline for (0..4) |i| {
            const sb = subBorrow(self.limbs[i], BN254_MODULUS[i], borrow);
            result[i] = sb.result;
            borrow = sb.borrow;
        }

        return .{ .limbs = result };
    }

    fn addModulus(self: Self) Self {
        var result: [4]u64 = undefined;
        var carry: u64 = 0;

        inline for (0..4) |i| {
            const ac = addCarry(self.limbs[i], BN254_MODULUS[i], carry);
            result[i] = ac.result;
            carry = ac.carry;
        }

        return .{ .limbs = result };
    }

    /// Format for printing
    pub fn format(self: Self, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
        _ = fmt;
        _ = options;
        try writer.print("0x{x:0>16}{x:0>16}{x:0>16}{x:0>16}", .{
            self.limbs[3],
            self.limbs[2],
            self.limbs[1],
            self.limbs[0],
        });
    }
};

// Verify BN254Scalar implements JoltField interface
comptime {
    _ = JoltField(BN254Scalar);
}

test "bn254 scalar basic operations" {
    const zero = BN254Scalar.zero();
    const one = BN254Scalar.one();

    try std.testing.expect(zero.isZero());
    try std.testing.expect(one.isOne());
    try std.testing.expect(!zero.isOne());
    try std.testing.expect(!one.isZero());
}

test "bn254 scalar addition and subtraction" {
    const a = BN254Scalar.fromU64(100);
    const b = BN254Scalar.fromU64(50);

    // a + b - b should equal a
    const sum = a.add(b);
    const back = sum.sub(b);
    try std.testing.expect(a.eql(back));

    // a - a should equal zero
    const diff = a.sub(a);
    try std.testing.expect(diff.isZero());
}

test "bn254 scalar multiplication" {
    const one = BN254Scalar.one();
    const a = BN254Scalar.fromU64(7);

    // a * 1 = a
    const prod1 = a.mul(one);
    try std.testing.expect(a.eql(prod1));

    // 1 * a = a
    const prod2 = one.mul(a);
    try std.testing.expect(a.eql(prod2));

    // a * 0 = 0
    const zero = BN254Scalar.zero();
    const prod3 = a.mul(zero);
    try std.testing.expect(prod3.isZero());
}

test "bn254 scalar multiplication correctness" {
    // Test: 3 * 7 = 21
    const three = BN254Scalar.fromU64(3);
    const seven = BN254Scalar.fromU64(7);
    const twenty_one = BN254Scalar.fromU64(21);

    const product = three.mul(seven);
    try std.testing.expect(product.eql(twenty_one));
}

test "bn254 scalar inverse" {
    const a = BN254Scalar.fromU64(7);
    const one = BN254Scalar.one();

    // a * a^{-1} = 1
    if (a.inverse()) |a_inv| {
        const prod = a.mul(a_inv);
        try std.testing.expect(prod.eql(one));
    } else {
        try std.testing.expect(false);
    }
}

test "bn254 scalar power" {
    const two = BN254Scalar.fromU64(2);
    const eight = BN254Scalar.fromU64(8);

    // 2^3 = 8
    const result = two.pow(3);
    try std.testing.expect(result.eql(eight));

    // a^0 = 1
    const one = BN254Scalar.one();
    const pow0 = two.pow(0);
    try std.testing.expect(pow0.eql(one));

    // a^1 = a
    const pow1 = two.pow(1);
    try std.testing.expect(pow1.eql(two));
}
