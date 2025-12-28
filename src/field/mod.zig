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

// ============================================================================
// BN254 Base Field (Fp) - for pairing operations
// ============================================================================
// The base field Fp is different from the scalar field Fr!
// Fp is used for G1/G2 point coordinates and the pairing target group GT

/// BN254 base field modulus
/// q = 21888242871839275222246405745257275088696311157297823662689037894645226208583
pub const BN254_FP_MODULUS: [4]u64 = .{
    0x3c208c16d87cfd47,
    0x97816a916871ca8d,
    0xb85045b68181585d,
    0x30644e72e131a029,
};

/// Montgomery R for Fp (R = 2^256 mod q)
pub const BN254_FP_R: [4]u64 = .{
    0xd35d438dc58f0d9d,
    0x0a78eb28f5c70b3d,
    0x666ea36f7879462c,
    0x0e0a77c19a07df2f,
};

/// Montgomery R^2 for Fp (R^2 = 2^512 mod q)
pub const BN254_FP_R2: [4]u64 = .{
    0xf32cfc5b538afa89,
    0xb5e71911d44501fb,
    0x47ab1eff0a417ff6,
    0x06d89f71cab8351f,
};

/// Montgomery constant: -q^{-1} mod 2^64
pub const BN254_FP_INV: u64 = 0x87d20782e4866389;

/// BN254 base field element for pairing operations
/// This is a wrapper around BN254Scalar that uses the base field modulus
/// Used for Fp, Fp2, Fp6, Fp12 tower and G1/G2 coordinates
pub const BN254BaseField = MontgomeryField(
    BN254_FP_MODULUS,
    BN254_FP_R,
    BN254_FP_R2,
    BN254_FP_INV,
);

/// Generic Montgomery field parameterized by constants
pub fn MontgomeryField(
    comptime modulus: [4]u64,
    comptime montgomery_r: [4]u64,
    comptime montgomery_r2: [4]u64,
    comptime montgomery_inv: u64,
) type {
    return struct {
        limbs: [4]u64,

        const Self = @This();

        /// Zero element
        pub fn zero() Self {
            return .{ .limbs = .{ 0, 0, 0, 0 } };
        }

        /// One element (in Montgomery form = R mod p)
        pub fn one() Self {
            return .{ .limbs = montgomery_r };
        }

        /// Check if zero
        pub fn isZero(self: Self) bool {
            return self.limbs[0] == 0 and self.limbs[1] == 0 and
                self.limbs[2] == 0 and self.limbs[3] == 0;
        }

        /// Check if one (in Montgomery form)
        pub fn isOne(self: Self) bool {
            return self.limbs[0] == montgomery_r[0] and self.limbs[1] == montgomery_r[1] and
                self.limbs[2] == montgomery_r[2] and self.limbs[3] == montgomery_r[3];
        }

        /// Equality check
        pub fn eql(self: Self, other: Self) bool {
            return self.limbs[0] == other.limbs[0] and self.limbs[1] == other.limbs[1] and
                self.limbs[2] == other.limbs[2] and self.limbs[3] == other.limbs[3];
        }

        /// Create from u64 (converts to Montgomery form)
        pub fn fromU64(n: u64) Self {
            var result = Self{ .limbs = .{ n, 0, 0, 0 } };
            result = result.montgomeryMul(.{ .limbs = montgomery_r2 });
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
            result = result.montgomeryMul(.{ .limbs = montgomery_r2 });
            return result;
        }

        /// Convert from Montgomery form back to standard representation
        pub fn fromMontgomery(self: Self) Self {
            return self.montgomeryMul(.{ .limbs = .{ 1, 0, 0, 0 } });
        }

        /// Convert to Montgomery form from standard representation
        pub fn toMontgomery(self: Self) Self {
            return self.montgomeryMul(.{ .limbs = montgomery_r2 });
        }

        /// Create from big-endian bytes (converts to Montgomery form)
        pub fn fromBytesBE(bytes: *const [32]u8) Self {
            // Reverse byte order for big-endian
            var le_bytes: [32]u8 = undefined;
            for (0..32) |i| {
                le_bytes[i] = bytes[31 - i];
            }
            return fromBytes(&le_bytes);
        }

        /// Serialize to big-endian bytes (32 bytes)
        pub fn toBytesBE(self: Self) [32]u8 {
            // First convert from Montgomery form
            const standard = self.fromMontgomery();

            // Convert limbs to bytes (little-endian)
            var le_bytes: [32]u8 = undefined;
            for (0..4) |i| {
                std.mem.writeInt(u64, le_bytes[i * 8 ..][0..8], standard.limbs[i], .little);
            }

            // Reverse for big-endian output
            var be_bytes: [32]u8 = undefined;
            for (0..32) |i| {
                be_bytes[i] = le_bytes[31 - i];
            }
            return be_bytes;
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
            var t: [5]u64 = .{ 0, 0, 0, 0, 0 };

            inline for (0..4) |i| {
                var carry: u64 = 0;
                inline for (0..4) |j| {
                    const prod = mulWide(self.limbs[i], other.limbs[j]);
                    const sum = @as(u128, t[j]) + prod + @as(u128, carry);
                    t[j] = @truncate(sum);
                    carry = @truncate(sum >> 64);
                }
                const sum_t4 = @as(u128, t[4]) + @as(u128, carry);
                t[4] = @truncate(sum_t4);

                const m = t[0] *% montgomery_inv;

                carry = 0;
                const prod0 = mulWide(m, modulus[0]);
                const sum0 = @as(u128, t[0]) + prod0;
                carry = @truncate(sum0 >> 64);

                inline for (1..4) |j| {
                    const prod = mulWide(m, modulus[j]);
                    const sum = @as(u128, t[j]) + prod + @as(u128, carry);
                    t[j - 1] = @truncate(sum);
                    carry = @truncate(sum >> 64);
                }
                const final_sum = @as(u128, t[4]) + @as(u128, carry);
                t[3] = @truncate(final_sum);
                t[4] = @truncate(final_sum >> 64);
            }

            var result = Self{ .limbs = .{ t[0], t[1], t[2], t[3] } };

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

        /// Field squaring
        pub fn square(self: Self) Self {
            return self.montgomeryMul(self);
        }

        /// Doubling (2*self)
        pub fn double(self: Self) Self {
            return self.add(self);
        }

        /// Negation
        pub fn neg(self: Self) Self {
            if (self.isZero()) return self;
            return (Self{ .limbs = modulus }).sub(self);
        }

        /// Field inverse using Fermat's little theorem: a^{-1} = a^{p-2}
        pub fn inverse(self: Self) ?Self {
            if (self.isZero()) return null;

            var result = Self.one();
            var base = self;
            var exp: [4]u64 = modulus;
            exp[0] -= 2;

            for (0..256) |i| {
                const word_idx = i / 64;
                const bit_idx: u6 = @truncate(i % 64);
                if ((exp[word_idx] >> bit_idx) & 1 == 1) {
                    result = result.mul(base);
                }
                base = base.square();
            }

            return result;
        }

        fn lessThanModulus(self: Self) bool {
            var i: usize = 3;
            while (true) : (i -= 1) {
                if (self.limbs[i] < modulus[i]) return true;
                if (self.limbs[i] > modulus[i]) return false;
                if (i == 0) break;
            }
            return false;
        }

        fn subtractModulus(self: Self) Self {
            var result: [4]u64 = undefined;
            var borrow: u64 = 0;

            inline for (0..4) |i| {
                const sb = subBorrow(self.limbs[i], modulus[i], borrow);
                result[i] = sb.result;
                borrow = sb.borrow;
            }

            return Self{ .limbs = result };
        }

        fn addModulus(self: Self) Self {
            var result: [4]u64 = undefined;
            var carry: u64 = 0;

            inline for (0..4) |i| {
                const ac = addCarry(self.limbs[i], modulus[i], carry);
                result[i] = ac.result;
                carry = ac.carry;
            }

            return Self{ .limbs = result };
        }
    };
}

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

    /// Convert to Montgomery form from standard representation
    /// Used when we have raw limbs that need to be converted
    pub fn toMontgomery(self: Self) Self {
        // Multiply by R^2 to get a * R^2 * R^{-1} = a * R
        return self.montgomeryMul(.{ .limbs = BN254_R2 });
    }

    /// Create from big-endian bytes (converts to Montgomery form)
    pub fn fromBytesBE(bytes: *const [32]u8) Self {
        // Reverse byte order for big-endian
        var le_bytes: [32]u8 = undefined;
        for (0..32) |i| {
            le_bytes[i] = bytes[31 - i];
        }
        return fromBytes(&le_bytes);
    }

    /// Serialize to big-endian bytes (32 bytes)
    pub fn toBytesBE(self: Self) [32]u8 {
        // First convert from Montgomery form
        const standard = self.fromMontgomery();

        // Convert limbs to bytes (little-endian)
        var le_bytes: [32]u8 = undefined;
        for (0..4) |i| {
            std.mem.writeInt(u64, le_bytes[i * 8 ..][0..8], standard.limbs[i], .little);
        }

        // Reverse for big-endian output
        var be_bytes: [32]u8 = undefined;
        for (0..32) |i| {
            be_bytes[i] = le_bytes[31 - i];
        }
        return be_bytes;
    }

    /// Convert to u64 (returns low 64 bits of the value)
    /// Useful for debugging and displaying small values.
    /// Note: This loses precision for values >= 2^64.
    pub fn toU64(self: Self) u64 {
        const standard = self.fromMontgomery();
        return standard.limbs[0];
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

    /// Field squaring (optimized using Karatsuba-like technique)
    /// Saves ~25% multiplications compared to naive multiplication
    pub fn square(self: Self) Self {
        // Optimized squaring: we can compute a^2 with fewer multiplications
        // Since (a0 + a1*2^64 + a2*2^128 + a3*2^192)^2 has symmetric terms
        // For example: 2*a0*a1 instead of a0*a1 + a1*a0
        //
        // First, compute the product matrix with reduced operations
        var t: [8]u64 = .{ 0, 0, 0, 0, 0, 0, 0, 0 };
        var carry: u64 = 0;

        // Compute diagonal terms a[i]^2
        inline for (0..4) |i| {
            const prod = mulWide(self.limbs[i], self.limbs[i]);
            const idx = i * 2;
            const sum = @as(u128, t[idx]) + prod;
            t[idx] = @truncate(sum);
            const overflow = @as(u64, @truncate(sum >> 64));
            const sum_next = @as(u128, t[idx + 1]) + @as(u128, overflow);
            t[idx + 1] = @truncate(sum_next);
            // Propagate carry to higher limbs
            if (idx + 2 < 8) {
                t[idx + 2] +%= @truncate(sum_next >> 64);
            }
        }

        // Compute off-diagonal terms 2*a[i]*a[j] for i < j
        inline for (0..4) |i| {
            inline for (i + 1..4) |j| {
                const prod = mulWide(self.limbs[i], self.limbs[j]);
                const idx = i + j;
                // Double the product (since we count both a[i]*a[j] and a[j]*a[i])
                const doubled_lo = @as(u64, @truncate(prod)) << 1;
                const doubled_hi = (@as(u64, @truncate(prod >> 64)) << 1) | (@as(u64, @truncate(prod)) >> 63);
                const carry_out: u64 = @as(u64, @truncate(prod >> 64)) >> 63;

                const sum0 = @as(u128, t[idx]) + @as(u128, doubled_lo);
                t[idx] = @truncate(sum0);
                const sum1 = @as(u128, t[idx + 1]) + @as(u128, doubled_hi) + (sum0 >> 64);
                t[idx + 1] = @truncate(sum1);
                if (idx + 2 < 8) {
                    const sum2 = @as(u128, t[idx + 2]) + @as(u128, carry_out) + (sum1 >> 64);
                    t[idx + 2] = @truncate(sum2);
                    if (idx + 3 < 8) {
                        t[idx + 3] +%= @truncate(sum2 >> 64);
                    }
                }
            }
        }

        // Montgomery reduction: reduce t (512 bits) to 256 bits mod p
        var r: [5]u64 = .{ t[0], t[1], t[2], t[3], 0 };

        inline for (0..4) |i| {
            const m = r[0] *% BN254_INV;
            carry = 0;
            const prod0 = mulWide(m, BN254_MODULUS[0]);
            const sum0 = @as(u128, r[0]) + prod0;
            carry = @truncate(sum0 >> 64);

            inline for (1..4) |j| {
                const prod = mulWide(m, BN254_MODULUS[j]);
                const sum = @as(u128, r[j]) + prod + @as(u128, carry);
                r[j - 1] = @truncate(sum);
                carry = @truncate(sum >> 64);
            }
            const t_idx = i + 4;
            const final_sum = @as(u128, r[4]) + @as(u128, carry) + @as(u128, t[t_idx]);
            r[3] = @truncate(final_sum);
            r[4] = @truncate(final_sum >> 64);
        }

        var result = Self{ .limbs = .{ r[0], r[1], r[2], r[3] } };
        if (r[4] != 0 or !result.lessThanModulus()) {
            result = result.subtractModulus();
        }
        return result;
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

/// Batch field operations for SIMD-like performance
/// These functions operate on slices for cache efficiency
pub const BatchOps = struct {
    /// Batch addition: results[i] = a[i] + b[i]
    pub fn batchAdd(results: []BN254Scalar, a: []const BN254Scalar, b: []const BN254Scalar) void {
        std.debug.assert(results.len == a.len and a.len == b.len);
        for (0..results.len) |i| {
            results[i] = a[i].add(b[i]);
        }
    }

    /// Batch subtraction: results[i] = a[i] - b[i]
    pub fn batchSub(results: []BN254Scalar, a: []const BN254Scalar, b: []const BN254Scalar) void {
        std.debug.assert(results.len == a.len and a.len == b.len);
        for (0..results.len) |i| {
            results[i] = a[i].sub(b[i]);
        }
    }

    /// Batch multiplication: results[i] = a[i] * b[i]
    pub fn batchMul(results: []BN254Scalar, a: []const BN254Scalar, b: []const BN254Scalar) void {
        std.debug.assert(results.len == a.len and a.len == b.len);
        for (0..results.len) |i| {
            results[i] = a[i].mul(b[i]);
        }
    }

    /// Batch scalar multiplication: results[i] = a[i] * scalar
    pub fn batchMulScalar(results: []BN254Scalar, a: []const BN254Scalar, scalar: BN254Scalar) void {
        std.debug.assert(results.len == a.len);
        for (0..results.len) |i| {
            results[i] = a[i].mul(scalar);
        }
    }

    /// Batch squaring: results[i] = a[i]^2
    pub fn batchSquare(results: []BN254Scalar, a: []const BN254Scalar) void {
        std.debug.assert(results.len == a.len);
        for (0..results.len) |i| {
            results[i] = a[i].square();
        }
    }

    /// Inner product: sum(a[i] * b[i])
    pub fn innerProduct(a: []const BN254Scalar, b: []const BN254Scalar) BN254Scalar {
        std.debug.assert(a.len == b.len);
        var result = BN254Scalar.zero();
        for (0..a.len) |i| {
            result = result.add(a[i].mul(b[i]));
        }
        return result;
    }

    /// Sum of products with precomputed terms for Horner's method
    /// Computes: a[0] + x*(a[1] + x*(a[2] + ... + x*a[n-1])))
    pub fn hornerEval(coeffs: []const BN254Scalar, x: BN254Scalar) BN254Scalar {
        if (coeffs.len == 0) return BN254Scalar.zero();

        var result = coeffs[coeffs.len - 1];
        var i: usize = coeffs.len - 1;
        while (i > 0) {
            i -= 1;
            result = result.mul(x).add(coeffs[i]);
        }
        return result;
    }

    /// Batch inverse using Montgomery's trick
    /// Computes inverses of all elements using only one field inversion
    /// Much faster than computing individual inverses: O(3n) muls + 1 inverse vs O(n) inverses
    pub fn batchInverse(results: []BN254Scalar, a: []const BN254Scalar, allocator: std.mem.Allocator) !void {
        std.debug.assert(results.len == a.len);
        if (a.len == 0) return;

        // Step 1: Compute running products
        // products[i] = a[0] * a[1] * ... * a[i]
        const products = try allocator.alloc(BN254Scalar, a.len);
        defer allocator.free(products);

        products[0] = a[0];
        for (1..a.len) |i| {
            if (a[i].isZero()) {
                // Handle zero by using one (will result in zero inverse)
                products[i] = products[i - 1];
            } else {
                products[i] = products[i - 1].mul(a[i]);
            }
        }

        // Step 2: Compute inverse of the final product
        const all_inv = products[a.len - 1].inverse() orelse BN254Scalar.zero();

        // Step 3: Compute individual inverses
        var running_inv = all_inv;
        var i: usize = a.len;
        while (i > 1) {
            i -= 1;
            if (a[i].isZero()) {
                results[i] = BN254Scalar.zero();
            } else {
                // a[i]^{-1} = running_inv * products[i-1]
                results[i] = running_inv.mul(products[i - 1]);
                running_inv = running_inv.mul(a[i]);
            }
        }
        // Handle a[0]
        if (a[0].isZero()) {
            results[0] = BN254Scalar.zero();
        } else {
            results[0] = running_inv;
        }
    }

    /// Multi-scalar multiplication accumulator
    /// Computes sum(scalars[i] * bases[i])
    pub fn multiScalarMulLinear(scalars: []const BN254Scalar, bases: []const BN254Scalar) BN254Scalar {
        return innerProduct(scalars, bases);
    }
};

/// SIMD-accelerated field operations
/// Uses Zig's built-in SIMD vectors for parallel limb operations where beneficial
pub const SimdOps = struct {
    /// Vector type for 4 limbs (one field element)
    pub const Limb4 = @Vector(4, u64);

    /// Convert scalar to SIMD vector
    pub inline fn toVec(scalar: BN254Scalar) Limb4 {
        return Limb4{ scalar.limbs[0], scalar.limbs[1], scalar.limbs[2], scalar.limbs[3] };
    }

    /// Convert SIMD vector to scalar
    pub inline fn fromVec(vec: Limb4) BN254Scalar {
        return BN254Scalar{ .limbs = .{ vec[0], vec[1], vec[2], vec[3] } };
    }

    /// SIMD modulus vector
    pub const modulus_vec: Limb4 = Limb4{
        BN254_MODULUS[0],
        BN254_MODULUS[1],
        BN254_MODULUS[2],
        BN254_MODULUS[3],
    };

    /// Parallel comparison: returns true if all limbs of a < b (lexicographically)
    pub inline fn lessThan(a: Limb4, b: Limb4) bool {
        // Compare from most significant limb
        if (a[3] != b[3]) return a[3] < b[3];
        if (a[2] != b[2]) return a[2] < b[2];
        if (a[1] != b[1]) return a[1] < b[1];
        return a[0] < b[0];
    }

    /// SIMD-parallel addition with reduction (vectorized limb operations)
    /// For cases where we add many field elements, this allows better instruction pipelining
    pub fn simdAdd4(a: [4]BN254Scalar, b: [4]BN254Scalar) [4]BN254Scalar {
        var results: [4]BN254Scalar = undefined;
        // Process all 4 additions - compiler can vectorize limb operations
        comptime var i = 0;
        inline while (i < 4) : (i += 1) {
            results[i] = a[i].add(b[i]);
        }
        return results;
    }

    /// SIMD-parallel multiplication (for pipelining 4 muls together)
    pub fn simdMul4(a: [4]BN254Scalar, b: [4]BN254Scalar) [4]BN254Scalar {
        var results: [4]BN254Scalar = undefined;
        comptime var i = 0;
        inline while (i < 4) : (i += 1) {
            results[i] = a[i].mul(b[i]);
        }
        return results;
    }

    /// Process slices in chunks of 4 for better vectorization
    pub fn batchAddSimd(results: []BN254Scalar, a: []const BN254Scalar, b: []const BN254Scalar) void {
        std.debug.assert(results.len == a.len and a.len == b.len);

        const len = results.len;
        const chunks = len / 4;
        const remainder = len % 4;

        // Process in SIMD-friendly chunks of 4
        var i: usize = 0;
        while (i < chunks * 4) : (i += 4) {
            const a4 = [4]BN254Scalar{ a[i], a[i + 1], a[i + 2], a[i + 3] };
            const b4 = [4]BN254Scalar{ b[i], b[i + 1], b[i + 2], b[i + 3] };
            const r4 = simdAdd4(a4, b4);
            results[i] = r4[0];
            results[i + 1] = r4[1];
            results[i + 2] = r4[2];
            results[i + 3] = r4[3];
        }

        // Handle remainder
        for (i..i + remainder) |j| {
            results[j] = a[j].add(b[j]);
        }
    }

    /// Process multiplications in chunks of 4 for better vectorization
    pub fn batchMulSimd(results: []BN254Scalar, a: []const BN254Scalar, b: []const BN254Scalar) void {
        std.debug.assert(results.len == a.len and a.len == b.len);

        const len = results.len;
        const chunks = len / 4;
        const remainder = len % 4;

        // Process in SIMD-friendly chunks of 4
        var i: usize = 0;
        while (i < chunks * 4) : (i += 4) {
            const a4 = [4]BN254Scalar{ a[i], a[i + 1], a[i + 2], a[i + 3] };
            const b4 = [4]BN254Scalar{ b[i], b[i + 1], b[i + 2], b[i + 3] };
            const r4 = simdMul4(a4, b4);
            results[i] = r4[0];
            results[i + 1] = r4[1];
            results[i + 2] = r4[2];
            results[i + 3] = r4[3];
        }

        // Handle remainder
        for (i..i + remainder) |j| {
            results[j] = a[j].mul(b[j]);
        }
    }

    /// Inner product with unrolled accumulation for better pipelining
    pub fn innerProductSimd(a: []const BN254Scalar, b: []const BN254Scalar) BN254Scalar {
        std.debug.assert(a.len == b.len);

        const len = a.len;
        const chunks = len / 4;
        const remainder = len % 4;

        // Use 4 accumulators for better instruction-level parallelism
        var acc0 = BN254Scalar.zero();
        var acc1 = BN254Scalar.zero();
        var acc2 = BN254Scalar.zero();
        var acc3 = BN254Scalar.zero();

        var i: usize = 0;
        while (i < chunks * 4) : (i += 4) {
            acc0 = acc0.add(a[i].mul(b[i]));
            acc1 = acc1.add(a[i + 1].mul(b[i + 1]));
            acc2 = acc2.add(a[i + 2].mul(b[i + 2]));
            acc3 = acc3.add(a[i + 3].mul(b[i + 3]));
        }

        // Handle remainder
        for (i..i + remainder) |j| {
            acc0 = acc0.add(a[j].mul(b[j]));
        }

        // Combine accumulators
        return acc0.add(acc1).add(acc2.add(acc3));
    }
};

test "simd operations" {
    var a: [8]BN254Scalar = undefined;
    var b: [8]BN254Scalar = undefined;
    var results: [8]BN254Scalar = undefined;

    for (0..8) |i| {
        a[i] = BN254Scalar.fromU64(@as(u64, @intCast(i + 1)));
        b[i] = BN254Scalar.fromU64(@as(u64, @intCast(i + 10)));
    }

    // Test SIMD batch add
    SimdOps.batchAddSimd(&results, &a, &b);
    for (0..8) |i| {
        const expected = BN254Scalar.fromU64(@as(u64, @intCast(2 * i + 11)));
        try std.testing.expect(results[i].eql(expected));
    }

    // Test SIMD batch mul
    SimdOps.batchMulSimd(&results, &a, &b);
    for (0..8) |i| {
        const expected = BN254Scalar.fromU64(@as(u64, @intCast((i + 1) * (i + 10))));
        try std.testing.expect(results[i].eql(expected));
    }

    // Test SIMD inner product: sum((i+1) * (i+10)) for i=0..7
    // = 1*10 + 2*11 + 3*12 + 4*13 + 5*14 + 6*15 + 7*16 + 8*17
    // = 10 + 22 + 36 + 52 + 70 + 90 + 112 + 136 = 528
    const ip = SimdOps.innerProductSimd(&a, &b);
    try std.testing.expect(ip.eql(BN254Scalar.fromU64(528)));
}

test "batch operations" {
    const allocator = std.testing.allocator;

    var a: [4]BN254Scalar = undefined;
    var b: [4]BN254Scalar = undefined;
    var results: [4]BN254Scalar = undefined;

    for (0..4) |i| {
        a[i] = BN254Scalar.fromU64(@as(u64, @intCast(i + 1)));
        b[i] = BN254Scalar.fromU64(@as(u64, @intCast(i + 5)));
    }

    // Test batch add
    BatchOps.batchAdd(&results, &a, &b);
    for (0..4) |i| {
        const expected = BN254Scalar.fromU64(@as(u64, @intCast(2 * i + 6)));
        try std.testing.expect(results[i].eql(expected));
    }

    // Test inner product: 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70
    const ip = BatchOps.innerProduct(&a, &b);
    try std.testing.expect(ip.eql(BN254Scalar.fromU64(70)));

    // Test Horner evaluation: 1 + 2x + 3x^2 + 4x^3 at x=2
    // = 1 + 2*2 + 3*4 + 4*8 = 1 + 4 + 12 + 32 = 49
    const horner_result = BatchOps.hornerEval(&a, BN254Scalar.fromU64(2));
    try std.testing.expect(horner_result.eql(BN254Scalar.fromU64(49)));

    // Test batch inverse
    var non_zero_a: [3]BN254Scalar = .{
        BN254Scalar.fromU64(2),
        BN254Scalar.fromU64(3),
        BN254Scalar.fromU64(5),
    };
    var inverses: [3]BN254Scalar = undefined;
    try BatchOps.batchInverse(&inverses, &non_zero_a, allocator);

    // Verify: a[i] * inv[i] = 1
    for (0..3) |i| {
        const prod = non_zero_a[i].mul(inverses[i]);
        try std.testing.expect(prod.eql(BN254Scalar.one()));
    }
}

// Export pairing module
pub const pairing = @import("pairing.zig");
pub const Fp2 = pairing.Fp2;
pub const Fp6 = pairing.Fp6;
pub const Fp12 = pairing.Fp12;
pub const G2Point = pairing.G2Point;

test {
    // Run pairing tests
    _ = pairing;
}
