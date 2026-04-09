//! Montgomery field arithmetic over `[N]u64` limbs.
//!
//! `MontgomeryField(modulus, n_limbs, r2, n_prime)` instantiates a
//! finite field `Fp = ℤ / pℤ` with `p = modulus` and stores elements in
//! Montgomery form (`a · R mod p` where `R = 2^(64·N)`).
//!
//! The Montgomery representation lets multiplication be implemented
//! without an expensive `mod p` step at the end of every operation —
//! instead, multiplication uses the CIOS (Coarsely Integrated Operand
//! Scanning) algorithm which interleaves multiplication and reduction.
//! See Acar's "Analyzing and Comparing Montgomery Multiplication
//! Algorithms" (1996) for the canonical description.
//!
//! Constants the caller must precompute (the constructor takes them so
//! the field can stay generic over any modulus):
//!
//!   - `modulus`: `[N]u64`, the prime `p` (little-endian limbs).
//!   - `r2`: `R^2 mod p` where `R = 2^(64·N)`. Used to convert into
//!     Montgomery form via `to_mont(a) = a · R^2 · R^{-1} mod p`.
//!   - `n_prime`: `-p^{-1} mod 2^64`. Drives the per-limb reduction
//!     step inside CIOS.
//!
//! Element layout:
//!
//!   - `Element` is `[N]u64`. The numeric value of `e` is
//!     `(sum_i e[i] * 2^(64·i)) · R^{-1} mod p`.
//!   - Zero is `[0; N]` (Montgomery form is closed under zero).
//!   - One is `R mod p`, which the constructor derives from
//!     `to_mont(1)`.

const std = @import("std");
const bigint = @import("../../bigint.zig");

/// Build a Montgomery field type over the given modulus / N / R^2 / -p^{-1}.
///
/// The factory is comptime so callers get a fresh type per modulus and
/// the inlined arithmetic stays branch-free over the limb count.
pub fn MontgomeryField(
    comptime N: comptime_int,
    comptime modulus: [N]u64,
    comptime r2: [N]u64,
    comptime n_prime: u64,
) type {
    return struct {
        const Self = @This();

        /// One element of the field. Stored in Montgomery form.
        pub const Element = [N]u64;

        /// The prime modulus, exposed so consumers can interrogate it.
        pub const MODULUS: [N]u64 = modulus;

        /// `R^2 mod p` — used to convert raw integers into Montgomery
        /// form via `montMul(raw, R2)`.
        pub const R2: [N]u64 = r2;

        /// Negative inverse of the modulus mod 2^64. Drives the per-
        /// limb reduction in CIOS.
        pub const N_PRIME: u64 = n_prime;

        /// Limb count, exported for byte conversions.
        pub const LIMB_COUNT: comptime_int = N;

        /// The additive identity. Already in Montgomery form (zero is
        /// the same in both representations).
        pub fn zero() Element {
            return .{0} ** N;
        }

        /// The multiplicative identity in Montgomery form: `R mod p`.
        /// Computed as `montMul(1, R2)` so the constructor doesn't have
        /// to embed `R mod p` separately.
        pub fn one() Element {
            var raw_one: [N]u64 = .{0} ** N;
            raw_one[0] = 1;
            return montMul(raw_one, r2);
        }

        /// Constant-shape equality. Walks all limbs.
        pub fn eql(a: Element, b: Element) bool {
            inline for (0..N) |i| {
                if (a[i] != b[i]) return false;
            }
            return true;
        }

        /// `a + b mod p`.
        pub fn add(a: Element, b: Element) Element {
            var sum: [N]u64 = undefined;
            const carry = bigint.add(N, &sum, a, b);
            // If the sum overflowed `R` (carry=1) or is now ≥ p,
            // subtract the modulus to bring it back into range.
            var reduced: [N]u64 = undefined;
            const borrow = bigint.sub(N, &reduced, sum, modulus);
            // Use sum if borrow=1 (sum < p) and carry=0; otherwise use
            // the reduced form. This is a constant-shape select.
            if (carry == 1 or borrow == 0) return reduced;
            return sum;
        }

        /// `a - b mod p`.
        pub fn sub(a: Element, b: Element) Element {
            var diff: [N]u64 = undefined;
            const borrow = bigint.sub(N, &diff, a, b);
            if (borrow == 0) return diff;
            // Borrow: add the modulus back.
            var corrected: [N]u64 = undefined;
            _ = bigint.add(N, &corrected, diff, modulus);
            return corrected;
        }

        /// `-a mod p` = `p - a` (or `0` for zero).
        pub fn neg(a: Element) Element {
            if (bigint.isZero(N, a)) return zero();
            var out: [N]u64 = undefined;
            _ = bigint.sub(N, &out, modulus, a);
            return out;
        }

        /// `a^2 mod p`. Convenience wrapper around `montMul(a, a)`.
        pub fn square(a: Element) Element {
            return montMul(a, a);
        }

        /// Square-and-multiply exponentiation `a^e mod p` where the
        /// exponent is a raw little-endian limb array (not in
        /// Montgomery form). The exponent is walked from MSB to LSB
        /// across all `N` limbs.
        ///
        /// `a` must be in Montgomery form.
        pub fn pow(a: Element, exponent: [N]u64) Element {
            // Find the index of the most-significant set bit so we can
            // skip the leading zeros and avoid an empty `result *= a`
            // step on `a^0`.
            const top_bit = bigint.bitLen(N, exponent);
            if (top_bit == 0) return one();

            var result = a;
            var i = top_bit - 1;
            while (i > 0) {
                i -= 1;
                result = square(result);
                const limb = i / 64;
                const bit = @as(u6, @intCast(i % 64));
                if (((exponent[limb] >> bit) & 1) == 1) {
                    result = montMul(result, a);
                }
            }
            return result;
        }

        /// Field inversion via Fermat's little theorem: `a^{-1} = a^{p-2} mod p`.
        /// Returns `zero` for `zero` (which is mathematically undefined
        /// but lets callers avoid an explicit branch in common code).
        ///
        /// This is much slower than the binary extended Euclidean
        /// algorithm — Fermat does ~381 squarings + ~190 multiplies
        /// for BLS12-381 — but it is constant-time-friendly and trivial
        /// to validate against the cheaper algorithms once they land.
        pub fn inv(a: Element) Element {
            if (bigint.isZero(N, a)) return zero();
            // Compute p - 2 as a raw limb array.
            var p_minus_two: [N]u64 = modulus;
            // The modulus must be odd for this to work; subtract 2 from
            // the lowest limb without underflow.
            std.debug.assert(p_minus_two[0] >= 2);
            p_minus_two[0] -= 2;
            return pow(a, p_minus_two);
        }

        /// Coarsely Integrated Operand Scanning (CIOS) Montgomery
        /// multiplication. Returns `a · b · R^{-1} mod p` — i.e. the
        /// product in Montgomery form.
        ///
        /// The algorithm uses an `[N+2]u64` accumulator (one extra limb
        /// for partial sums and one for the carry-out of the reduction
        /// step). For BLS12-381's 6-limb field this is 8 limbs of
        /// scratch — well within the inlined comptime budget.
        pub fn montMul(a: Element, b: Element) Element {
            var t: [N + 2]u64 = .{0} ** (N + 2);
            inline for (0..N) |i| {
                // Outer-loop multiply: t += a * b[i]
                var carry: u64 = 0;
                inline for (0..N) |j| {
                    const prod = @as(u128, a[j]) * @as(u128, b[i]) + @as(u128, t[j]) + @as(u128, carry);
                    t[j] = @truncate(prod);
                    carry = @intCast(prod >> 64);
                }
                const t_n_sum = @addWithOverflow(t[N], carry);
                t[N] = t_n_sum[0];
                t[N + 1] += @intCast(t_n_sum[1]);

                // Reduce: m = t[0] * n_prime mod 2^64; t = (t + m * p) / 2^64
                const m: u64 = @truncate(@as(u128, t[0]) *% @as(u128, n_prime));
                {
                    const prod0 = @as(u128, m) * @as(u128, modulus[0]) + @as(u128, t[0]);
                    var reduce_carry: u64 = @intCast(prod0 >> 64);
                    inline for (1..N) |j| {
                        const prod = @as(u128, m) * @as(u128, modulus[j]) + @as(u128, t[j]) + @as(u128, reduce_carry);
                        t[j - 1] = @truncate(prod);
                        reduce_carry = @intCast(prod >> 64);
                    }
                    const t_n_red = @addWithOverflow(t[N], reduce_carry);
                    t[N - 1] = t_n_red[0];
                    const t_n1_red = @addWithOverflow(t[N + 1], @as(u64, t_n_red[1]));
                    t[N] = t_n1_red[0];
                    t[N + 1] = t_n1_red[1];
                }
            }
            // Final conditional subtract — t may be in [0, 2p).
            var result: [N]u64 = undefined;
            inline for (0..N) |i| result[i] = t[i];
            var corrected: [N]u64 = undefined;
            const borrow = bigint.sub(N, &corrected, result, modulus);
            // If borrow=0 the subtraction succeeded (result ≥ p), use
            // corrected; otherwise use result. The N+1 carry above
            // also forces the corrected branch.
            if (t[N] != 0 or borrow == 0) return corrected;
            return result;
        }

        /// Convert a raw little-endian integer into Montgomery form.
        pub fn fromRaw(raw: [N]u64) Element {
            return montMul(raw, r2);
        }

        /// Convert from Montgomery form back to a raw little-endian
        /// integer. Implemented as `montMul(value, [1, 0, ...])` which
        /// is `value · 1 · R^{-1} mod p`.
        pub fn toRaw(value: Element) [N]u64 {
            var one_raw: [N]u64 = .{0} ** N;
            one_raw[0] = 1;
            return montMul(value, one_raw);
        }

        /// Read a little-endian byte string into a field element.
        /// The input MUST already be < p; values ≥ p are rejected.
        pub fn fromBytesLeReduced(bytes: []const u8) error{NotInField}!Element {
            const raw = bigint.fromBytesLe(N, bytes);
            if (bigint.cmp(N, raw, modulus) != .lt) return error.NotInField;
            return fromRaw(raw);
        }

        /// Write a field element as a little-endian byte string. The
        /// output is canonical (i.e. fully reduced).
        pub fn toBytesLe(value: Element, out: []u8) void {
            const raw = toRaw(value);
            bigint.toBytesLe(N, raw, out);
        }
    };
}

// ---------------------------------------------------------------------------
// Tests — exercise the field machinery against a small toy modulus that
// is easy to reason about by hand. The BLS12-381 instantiation lives in
// `bls12_381.zig` once it lands.
// ---------------------------------------------------------------------------

const testing = std.testing;

// Toy 4-limb prime: p = 2^255 - 19 (Curve25519's base field). Easy to
// reason about, well-known constants on hand. We use 4 limbs even
// though it fits in 4 to keep the test exercising the same path the
// BLS12-381 6-limb instantiation will use.
//
// Constants computed once via Python:
//   p = 2**255 - 19
//   R = 2**256
//   r2 = (R * R) % p = 1444 = 0x5a4
//   n_prime = (-pow(p, -1, 2**64)) % 2**64 = 0x86bca1af286bca1b
const ED25519_P: [4]u64 = .{
    0xffffffffffffffed,
    0xffffffffffffffff,
    0xffffffffffffffff,
    0x7fffffffffffffff,
};
const ED25519_R2: [4]u64 = .{
    0x00000000000005a4,
    0x0000000000000000,
    0x0000000000000000,
    0x0000000000000000,
};
const ED25519_N_PRIME: u64 = 0x86bca1af286bca1b;

const Ed25519Fp = MontgomeryField(4, ED25519_P, ED25519_R2, ED25519_N_PRIME);

test "Ed25519Fp.zero is identity for add" {
    const zero = Ed25519Fp.zero();
    try testing.expect(Ed25519Fp.eql(Ed25519Fp.add(zero, zero), zero));
    const one = Ed25519Fp.one();
    try testing.expect(Ed25519Fp.eql(Ed25519Fp.add(one, zero), one));
    try testing.expect(Ed25519Fp.eql(Ed25519Fp.add(zero, one), one));
}

test "Ed25519Fp.one is identity for mul" {
    const one = Ed25519Fp.one();
    const a = Ed25519Fp.fromRaw(.{ 0x1234567890abcdef, 0, 0, 0 });
    try testing.expect(Ed25519Fp.eql(Ed25519Fp.montMul(a, one), a));
    try testing.expect(Ed25519Fp.eql(Ed25519Fp.montMul(one, a), a));
}

test "Ed25519Fp.toRaw round-trips fromRaw" {
    const raw: [4]u64 = .{ 0x0102030405060708, 0x0a0b0c0d0e0f0001, 0x1122334455667788, 0x12345678 };
    const m = Ed25519Fp.fromRaw(raw);
    const back = Ed25519Fp.toRaw(m);
    try testing.expectEqual(raw, back);
}

test "Ed25519Fp.add wraps around the modulus" {
    // 1 + (p - 1) = p ≡ 0
    const one = Ed25519Fp.fromRaw(.{ 1, 0, 0, 0 });
    const p_minus_one_raw: [4]u64 = .{
        ED25519_P[0] - 1,
        ED25519_P[1],
        ED25519_P[2],
        ED25519_P[3],
    };
    const p_minus_one = Ed25519Fp.fromRaw(p_minus_one_raw);
    const sum = Ed25519Fp.add(one, p_minus_one);
    try testing.expect(Ed25519Fp.eql(sum, Ed25519Fp.zero()));
}

test "Ed25519Fp.sub borrows correctly" {
    // 0 - 1 = p - 1
    const zero = Ed25519Fp.zero();
    const one_e = Ed25519Fp.fromRaw(.{ 1, 0, 0, 0 });
    const result = Ed25519Fp.sub(zero, one_e);
    const expected = Ed25519Fp.fromRaw(.{
        ED25519_P[0] - 1,
        ED25519_P[1],
        ED25519_P[2],
        ED25519_P[3],
    });
    try testing.expect(Ed25519Fp.eql(result, expected));
}

test "Ed25519Fp.neg + add = zero" {
    const a = Ed25519Fp.fromRaw(.{ 0x1234567890abcdef, 0xabad1deafeedface, 0xdeadbeefcafebabe, 0x0123456789abcdef });
    const neg_a = Ed25519Fp.neg(a);
    const sum = Ed25519Fp.add(a, neg_a);
    try testing.expect(Ed25519Fp.eql(sum, Ed25519Fp.zero()));
}

test "Ed25519Fp.montMul: 2 * 3 = 6" {
    const two = Ed25519Fp.fromRaw(.{ 2, 0, 0, 0 });
    const three = Ed25519Fp.fromRaw(.{ 3, 0, 0, 0 });
    const product = Ed25519Fp.montMul(two, three);
    const six = Ed25519Fp.fromRaw(.{ 6, 0, 0, 0 });
    try testing.expect(Ed25519Fp.eql(product, six));
}

test "Ed25519Fp.montMul: distributive over add (random sample)" {
    // (a + b) * c == a*c + b*c
    const a = Ed25519Fp.fromRaw(.{ 0x1234, 0x5678, 0, 0 });
    const b = Ed25519Fp.fromRaw(.{ 0x9abc, 0xdef0, 0x1234, 0 });
    const c = Ed25519Fp.fromRaw(.{ 0x42, 0, 0, 0 });
    const lhs = Ed25519Fp.montMul(Ed25519Fp.add(a, b), c);
    const rhs = Ed25519Fp.add(Ed25519Fp.montMul(a, c), Ed25519Fp.montMul(b, c));
    try testing.expect(Ed25519Fp.eql(lhs, rhs));
}

test "Ed25519Fp.fromBytesLeReduced rejects values ≥ p" {
    var buf: [32]u8 = .{0xff} ** 32;
    // Top byte must respect the prime ceiling — 0xff is too high.
    try testing.expectError(error.NotInField, Ed25519Fp.fromBytesLeReduced(&buf));

    // 1 < p; should succeed.
    var ok: [32]u8 = .{0} ** 32;
    ok[0] = 1;
    const e = try Ed25519Fp.fromBytesLeReduced(&ok);
    try testing.expect(Ed25519Fp.eql(e, Ed25519Fp.one()));
}

test "Ed25519Fp.square: 5^2 = 25" {
    const five = Ed25519Fp.fromRaw(.{ 5, 0, 0, 0 });
    const sq = Ed25519Fp.square(five);
    const twenty_five = Ed25519Fp.fromRaw(.{ 25, 0, 0, 0 });
    try testing.expect(Ed25519Fp.eql(sq, twenty_five));
}

test "Ed25519Fp.pow: 3^7 = 2187" {
    const three = Ed25519Fp.fromRaw(.{ 3, 0, 0, 0 });
    const result = Ed25519Fp.pow(three, .{ 7, 0, 0, 0 });
    const expected = Ed25519Fp.fromRaw(.{ 2187, 0, 0, 0 });
    try testing.expect(Ed25519Fp.eql(result, expected));
}

test "Ed25519Fp.pow: a^0 = 1" {
    const a = Ed25519Fp.fromRaw(.{ 0x42, 0, 0, 0 });
    const result = Ed25519Fp.pow(a, .{ 0, 0, 0, 0 });
    try testing.expect(Ed25519Fp.eql(result, Ed25519Fp.one()));
}

test "Ed25519Fp.pow: a^1 = a" {
    const a = Ed25519Fp.fromRaw(.{ 0x42, 0xab, 0, 0 });
    const result = Ed25519Fp.pow(a, .{ 1, 0, 0, 0 });
    try testing.expect(Ed25519Fp.eql(result, a));
}

test "Ed25519Fp.inv: a * a^-1 = 1" {
    const a = Ed25519Fp.fromRaw(.{ 0x12345678, 0, 0, 0 });
    const inv_a = Ed25519Fp.inv(a);
    const product = Ed25519Fp.montMul(a, inv_a);
    try testing.expect(Ed25519Fp.eql(product, Ed25519Fp.one()));
}

test "Ed25519Fp.inv: inv(1) = 1" {
    const one = Ed25519Fp.one();
    const inv_one = Ed25519Fp.inv(one);
    try testing.expect(Ed25519Fp.eql(inv_one, one));
}

test "Ed25519Fp.inv: inv(zero) = zero" {
    const z = Ed25519Fp.zero();
    const inv_z = Ed25519Fp.inv(z);
    try testing.expect(Ed25519Fp.eql(inv_z, z));
}

test "Ed25519Fp.toBytesLe round-trips fromBytesLeReduced" {
    var bytes_in: [32]u8 = .{0} ** 32;
    bytes_in[0] = 0x42;
    bytes_in[1] = 0x13;
    const e = try Ed25519Fp.fromBytesLeReduced(&bytes_in);
    var bytes_out: [32]u8 = undefined;
    Ed25519Fp.toBytesLe(e, &bytes_out);
    try testing.expectEqualSlices(u8, &bytes_in, &bytes_out);
}
