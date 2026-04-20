//! Generic wrapper that presents fiat-crypto generated Montgomery field
//! arithmetic through a MontgomeryField-compatible API.
//!
//! This is a separate type from MontgomeryField — the prover continues to
//! use the hand-written asm-backed path; only the verifier uses this
//! formally verified implementation.

const std = @import("std");
const bigint = @import("../bigint.zig");

/// Which field variant to instantiate.
pub const FieldVariant = enum { fr, fp };

/// Build a field element type backed by fiat-crypto generated code.
///
/// The resulting type has the same API surface as `MontgomeryField(4, ...)`
/// so that extension fields, pairing, and serialization code can use either
/// interchangeably.
pub fn FiatField(comptime variant: FieldVariant, comptime modulus: [4]u64, comptime r2: [4]u64, comptime n_prime: u64) type {
    _ = r2;
    _ = n_prime;

    // Select the correct fiat-crypto generated module at comptime.
    const fiat = switch (variant) {
        .fr => @import("bn254_fr_fiat.zig"),
        .fp => @import("bn254_fp_fiat.zig"),
    };

    // Pick the right function names based on variant.
    const fiatMul = switch (variant) {
        .fr => fiat.bn254FrMul,
        .fp => fiat.bn254FpMul,
    };
    const fiatSquare = switch (variant) {
        .fr => fiat.bn254FrSquare,
        .fp => fiat.bn254FpSquare,
    };
    const fiatAdd = switch (variant) {
        .fr => fiat.bn254FrAdd,
        .fp => fiat.bn254FpAdd,
    };
    const fiatSub = switch (variant) {
        .fr => fiat.bn254FrSub,
        .fp => fiat.bn254FpSub,
    };
    const fiatOpp = switch (variant) {
        .fr => fiat.bn254FrOpp,
        .fp => fiat.bn254FpOpp,
    };
    const fiatToMontgomery = switch (variant) {
        .fr => fiat.bn254FrToMontgomery,
        .fp => fiat.bn254FpToMontgomery,
    };
    const fiatFromMontgomery = switch (variant) {
        .fr => fiat.bn254FrFromMontgomery,
        .fp => fiat.bn254FpFromMontgomery,
    };
    const fiatNonzero = switch (variant) {
        .fr => fiat.bn254FrNonzero,
        .fp => fiat.bn254FpNonzero,
    };
    const fiatToBytes = switch (variant) {
        .fr => fiat.bn254FrToBytes,
        .fp => fiat.bn254FpToBytes,
    };
    const fiatFromBytes = switch (variant) {
        .fr => fiat.bn254FrFromBytes,
        .fp => fiat.bn254FpFromBytes,
    };

    return struct {
        limbs: [4]u64,

        const Self = @This();

        pub const LIMB_COUNT: comptime_int = 4;
        pub const MODULUS: [4]u64 = modulus;
        pub const FIELD_ELEMENT_BYTES: usize = 32;

        // -----------------------------------------------------------------
        // Constants
        // -----------------------------------------------------------------

        pub fn zero() Self {
            return .{ .limbs = .{ 0, 0, 0, 0 } };
        }

        pub fn one() Self {
            var result: [4]u64 = undefined;
            fiatToMontgomery(&result, .{ 1, 0, 0, 0 });
            return .{ .limbs = result };
        }

        // -----------------------------------------------------------------
        // Predicates
        // -----------------------------------------------------------------

        pub fn isZero(self: Self) bool {
            return self.limbs[0] == 0 and self.limbs[1] == 0 and
                self.limbs[2] == 0 and self.limbs[3] == 0;
        }

        pub fn isOne(self: Self) bool {
            return eql(self, one());
        }

        pub fn eql(a: Self, b: Self) bool {
            return a.limbs[0] == b.limbs[0] and a.limbs[1] == b.limbs[1] and
                a.limbs[2] == b.limbs[2] and a.limbs[3] == b.limbs[3];
        }

        // -----------------------------------------------------------------
        // Constructors
        // -----------------------------------------------------------------

        pub fn fromU64(n: u64) Self {
            var result: [4]u64 = undefined;
            fiatToMontgomery(&result, .{ n, 0, 0, 0 });
            return .{ .limbs = result };
        }

        pub fn fromBytes(bytes: []const u8) Self {
            var raw: [4]u64 = .{ 0, 0, 0, 0 };
            const len = @min(bytes.len, 32);
            var buf: [32]u8 = .{0} ** 32;
            @memcpy(buf[0..len], bytes[0..len]);
            inline for (0..4) |i| {
                raw[i] = std.mem.readInt(u64, buf[i * 8 ..][0..8], .little);
            }
            var result: [4]u64 = undefined;
            fiatToMontgomery(&result, raw);
            return .{ .limbs = result };
        }

        pub fn fromBytesBE(bytes: *const [32]u8) Self {
            var le_bytes: [32]u8 = undefined;
            for (0..32) |i| {
                le_bytes[i] = bytes[31 - i];
            }
            return fromBytes(&le_bytes);
        }

        pub fn fromRaw(raw: [4]u64) Self {
            var result: [4]u64 = undefined;
            fiatToMontgomery(&result, raw);
            return .{ .limbs = result };
        }

        pub fn toMontgomery(self: Self) Self {
            var result: [4]u64 = undefined;
            fiatToMontgomery(&result, self.limbs);
            return .{ .limbs = result };
        }

        pub fn toRaw(self: Self) [4]u64 {
            var result: [4]u64 = undefined;
            fiatFromMontgomery(&result, self.limbs);
            return result;
        }

        pub fn fromMontgomery(self: Self) Self {
            var result: [4]u64 = undefined;
            fiatFromMontgomery(&result, self.limbs);
            return .{ .limbs = result };
        }

        pub fn toBytesBE(self: Self) [32]u8 {
            var out: [32]u8 = undefined;
            const raw = toRaw(self);
            bigint.toBytesBe(4, raw, &out);
            return out;
        }

        pub fn toBytes(self: Self) [32]u8 {
            var out: [32]u8 = undefined;
            const raw = toRaw(self);
            bigint.toBytesLe(4, raw, &out);
            return out;
        }

        pub fn toU64(self: Self) u64 {
            return toRaw(self)[0];
        }

        // -----------------------------------------------------------------
        // Arithmetic
        // -----------------------------------------------------------------

        pub fn add(a: Self, b: Self) Self {
            var result: [4]u64 = undefined;
            fiatAdd(&result, a.limbs, b.limbs);
            return .{ .limbs = result };
        }

        pub fn sub(a: Self, b: Self) Self {
            var result: [4]u64 = undefined;
            fiatSub(&result, a.limbs, b.limbs);
            return .{ .limbs = result };
        }

        pub fn neg(a: Self) Self {
            if (a.isZero()) return zero();
            var result: [4]u64 = undefined;
            fiatOpp(&result, a.limbs);
            return .{ .limbs = result };
        }

        pub fn mul(a: Self, b: Self) Self {
            var result: [4]u64 = undefined;
            fiatMul(&result, a.limbs, b.limbs);
            return .{ .limbs = result };
        }

        pub fn square(a: Self) Self {
            var result: [4]u64 = undefined;
            fiatSquare(&result, a.limbs);
            return .{ .limbs = result };
        }

        pub fn double(self: Self) Self {
            return self.add(self);
        }

        /// Fused multiply-accumulate: a[0]*b[0] + a[1]*b[1].
        /// Non-fused implementation (3 reductions instead of 2) but
        /// correct and backed by verified primitives.
        pub fn sumOfProducts(a_pair: [2]Self, b_pair: [2]Self) Self {
            return mul(a_pair[0], b_pair[0]).add(mul(a_pair[1], b_pair[1]));
        }

        // -----------------------------------------------------------------
        // Inversion and exponentiation
        // -----------------------------------------------------------------

        /// Modular inverse via Fermat's little theorem: a^{p-2} mod p.
        pub fn inverse(a: Self) ?Self {
            if (a.isZero()) return null;

            const p_minus_two: [4]u64 = comptime blk: {
                var exp: [4]u64 = modulus;
                exp[0] -= 2;
                break :blk exp;
            };

            var result = Self.one();
            var base = a;

            inline for (0..4) |i| {
                var bits = p_minus_two[i];
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

        pub fn inv(a: Self) Self {
            return a.inverse() orelse zero();
        }

        pub fn pow(self: Self, exp: u64) Self {
            if (exp == 0) return one();
            var result = Self.one();
            var base = self;
            var e = exp;
            while (e > 0) {
                if (e & 1 == 1) result = result.mul(base);
                base = base.square();
                e >>= 1;
            }
            return result;
        }

        pub fn powLimbs(a: Self, exponent: [4]u64) Self {
            const top_bit = bigint.bitLen(4, exponent);
            if (top_bit == 0) return one();
            var result = a;
            var i = top_bit - 1;
            while (i > 0) {
                i -= 1;
                result = square(result);
                const limb = i / 64;
                const bit: u6 = @intCast(i % 64);
                if (((exponent[limb] >> bit) & 1) == 1) {
                    result = mul(result, a);
                }
            }
            return result;
        }

        /// Batch inversion using Montgomery's trick.
        pub fn batchInversion(elements: []Self, scratch: []Self) void {
            const n = elements.len;
            if (n == 0) return;
            var acc = one();
            for (0..n) |i| {
                scratch[i] = acc;
                if (!elements[i].isZero()) acc = acc.mul(elements[i]);
            }
            var inv_acc = acc.inverse() orelse unreachable;
            var i: usize = n;
            while (i > 0) {
                i -= 1;
                if (elements[i].isZero()) continue;
                const old = elements[i];
                elements[i] = scratch[i].mul(inv_acc);
                inv_acc = inv_acc.mul(old);
            }
        }

        // -----------------------------------------------------------------
        // Constant-time and utility
        // -----------------------------------------------------------------

        pub fn nonzero(self: Self) bool {
            var out: u64 = undefined;
            fiatNonzero(&out, self.limbs);
            return out != 0;
        }

        pub fn selectznz(cond: u1, a: Self, b: Self) Self {
            // Note: fiat selectznz returns a when cond=0, b when cond=1.
            // We match that: cond=0 → a, cond=1 → b.
            _ = fiatToBytes;
            _ = fiatFromBytes;
            var result: [4]u64 = undefined;
            switch (variant) {
                .fr => fiat.bn254FrSelectznz(&result, cond, a.limbs, b.limbs),
                .fp => fiat.bn254FpSelectznz(&result, cond, a.limbs, b.limbs),
            }
            return .{ .limbs = result };
        }

        // -----------------------------------------------------------------
        // Aliases for MontgomeryField compatibility
        // -----------------------------------------------------------------

        pub const montgomeryMul = mul;
        pub const montMul = mul;

        /// Debug formatter.
        pub fn format(self: Self, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
            _ = fmt;
            _ = options;
            const raw = toRaw(self);
            try writer.writeAll("0x");
            var i: usize = 4;
            while (i > 0) {
                i -= 1;
                try std.fmt.formatInt(raw[i], 16, .lower, .{ .width = 16, .fill = '0' }, writer);
            }
        }
    };
}
