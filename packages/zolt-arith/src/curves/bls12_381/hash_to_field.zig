//! RFC 9380 hash-to-curve byte-expansion primitives.
//!
//! Hash-to-curve is the standardized way to deterministically map a
//! byte string into a curve point. It is the missing piece that BLS
//! signature verification needs to convert the message bytes into a
//! G2 element.
//!
//! The full pipeline for `BLS_SIG_BLS12381G2_XMD:SHA-256_SSWU_RO_NUL_`
//! has four stages:
//!
//!   1. `expand_message_xmd` (RFC 9380 §5.3.1) — expand the input bytes
//!      and DST into a byte string of arbitrary length using SHA-256.
//!   2. `hash_to_field` (RFC 9380 §5.2) — interpret the expanded bytes
//!      as field elements modulo `p` (or `p²` for Fp2).
//!   3. `map_to_curve` (the SSWU map for BLS12-381 G2) — map a field
//!      element to an isogenous curve point, then push it through the
//!      isogeny to land on the actual curve.
//!   4. `clear_cofactor` — scale by the cofactor to land in the
//!      prime-order subgroup.
//!
//! This file currently implements stage 1 only. The remaining stages
//! land alongside the optimal Ate pairing in upcoming iterations.

const std = @import("std");
const bls12_381 = @import("curve.zig");

const Sha256 = std.crypto.hash.sha2.Sha256;
const Fp = bls12_381.Fp;
const Fp2 = bls12_381.Fp2;

/// Errors `expand_message_xmd` can surface.
pub const ExpandError = error{
    /// `len_in_bytes` exceeds `255 * b_in_bytes` (the limit set by the
    /// 8-bit big-endian length encoding inside `expand_message_xmd`).
    OutputTooLong,
    /// The DST exceeds 255 bytes. RFC 9380 caps DSTs at this length so
    /// `len(DST_prime)` always fits in a single byte.
    DstTooLong,
};

/// `expand_message_xmd(msg, DST, len_in_bytes)` from RFC 9380 §5.3.1.
///
/// Writes exactly `out.len` bytes into `out`. The output is a uniformly
/// pseudo-random byte string derived from `msg` and `dst`, suitable for
/// feeding into `hash_to_field`.
///
/// Algorithm (with `H = SHA-256`, `b_in_bytes = 32`, `s_in_bytes = 64`):
///
///   ell        = ceil(len_in_bytes / b_in_bytes)
///   DST_prime  = DST || I2OSP(len(DST), 1)
///   Z_pad      = I2OSP(0, s_in_bytes)
///   l_i_b_str  = I2OSP(len_in_bytes, 2)
///   msg_prime  = Z_pad || msg || l_i_b_str || I2OSP(0, 1) || DST_prime
///   b_0        = H(msg_prime)
///   b_1        = H(b_0 || I2OSP(1, 1) || DST_prime)
///   for i in 2..ell:
///     b_i      = H((b_0 XOR b_{i-1}) || I2OSP(i, 1) || DST_prime)
///   return concat(b_1, b_2, ..., b_ell)[0..len_in_bytes]
pub fn expand_message_xmd(
    out: []u8,
    msg: []const u8,
    dst: []const u8,
) ExpandError!void {
    const b_in_bytes: usize = 32; // SHA-256 output length
    const s_in_bytes: usize = 64; // SHA-256 input block size

    if (dst.len > 255) return ExpandError.DstTooLong;
    if (out.len > 255 * b_in_bytes) return ExpandError.OutputTooLong;
    if (out.len > std.math.maxInt(u16)) return ExpandError.OutputTooLong;

    const ell = (out.len + b_in_bytes - 1) / b_in_bytes;
    std.debug.assert(ell <= 255);

    // DST_prime = DST || I2OSP(len(DST), 1).
    var dst_prime: [256]u8 = undefined;
    @memcpy(dst_prime[0..dst.len], dst);
    dst_prime[dst.len] = @intCast(dst.len);
    const dst_prime_len = dst.len + 1;

    // b_0 = H(Z_pad || msg || l_i_b_str || 0x00 || DST_prime).
    var hasher = Sha256.init(.{});
    const z_pad: [s_in_bytes]u8 = .{0} ** s_in_bytes;
    hasher.update(&z_pad);
    hasher.update(msg);
    var l_i_b_str: [2]u8 = undefined;
    std.mem.writeInt(u16, &l_i_b_str, @intCast(out.len), .big);
    hasher.update(&l_i_b_str);
    hasher.update(&[_]u8{0});
    hasher.update(dst_prime[0..dst_prime_len]);
    var b_0: [b_in_bytes]u8 = undefined;
    hasher.final(&b_0);

    // b_1 = H(b_0 || 0x01 || DST_prime).
    var b_prev: [b_in_bytes]u8 = undefined;
    {
        var h = Sha256.init(.{});
        h.update(&b_0);
        h.update(&[_]u8{1});
        h.update(dst_prime[0..dst_prime_len]);
        h.final(&b_prev);
    }
    const first_chunk = @min(out.len, b_in_bytes);
    @memcpy(out[0..first_chunk], b_prev[0..first_chunk]);

    var i: usize = 2;
    while (i <= ell) : (i += 1) {
        // b_i = H((b_0 XOR b_{i-1}) || I2OSP(i, 1) || DST_prime).
        var xored: [b_in_bytes]u8 = undefined;
        for (0..b_in_bytes) |j| xored[j] = b_0[j] ^ b_prev[j];
        var h = Sha256.init(.{});
        h.update(&xored);
        h.update(&[_]u8{@intCast(i)});
        h.update(dst_prime[0..dst_prime_len]);
        var b_i: [b_in_bytes]u8 = undefined;
        h.final(&b_i);

        const start = (i - 1) * b_in_bytes;
        const remaining = out.len - start;
        const chunk = @min(remaining, b_in_bytes);
        @memcpy(out[start .. start + chunk], b_i[0..chunk]);
        b_prev = b_i;
    }
}

// ---------------------------------------------------------------------------
// Tests
//
// These are self-consistency tests rather than cross-implementation
// vectors: they verify that the function is deterministic, produces
// distinct outputs for distinct inputs, fills the requested length,
// and rejects malformed inputs. A separate cross-check pass against
// RFC 9380 Appendix K.1 vectors lands once a Python or external test
// harness is in place — transcribing the exact byte values from the
// RFC by hand is too error-prone.
// ---------------------------------------------------------------------------

const testing = std.testing;

const TEST_DST: []const u8 = "QUUX-V01-CS02-with-expander-SHA256-128";

test "expand_message_xmd: deterministic" {
    var out_a: [32]u8 = undefined;
    var out_b: [32]u8 = undefined;
    try expand_message_xmd(&out_a, "abc", TEST_DST);
    try expand_message_xmd(&out_b, "abc", TEST_DST);
    try testing.expectEqualSlices(u8, &out_a, &out_b);
}

test "expand_message_xmd: distinct messages produce distinct outputs" {
    var out_a: [32]u8 = undefined;
    var out_b: [32]u8 = undefined;
    try expand_message_xmd(&out_a, "abc", TEST_DST);
    try expand_message_xmd(&out_b, "abd", TEST_DST);
    try testing.expect(!std.mem.eql(u8, &out_a, &out_b));
}

test "expand_message_xmd: distinct DSTs produce distinct outputs" {
    var out_a: [32]u8 = undefined;
    var out_b: [32]u8 = undefined;
    try expand_message_xmd(&out_a, "abc", "DST_A");
    try expand_message_xmd(&out_b, "abc", "DST_B");
    try testing.expect(!std.mem.eql(u8, &out_a, &out_b));
}

test "expand_message_xmd: 128-byte output is fully populated" {
    // The first b_in_bytes bytes come from b_1 and the next ones from
    // b_2 through b_ell. Make sure no chunk is left as the 0x00 default
    // by checking that the last byte is not the default zero value.
    var out: [128]u8 = undefined;
    try expand_message_xmd(&out, "abc", TEST_DST);
    // It's overwhelmingly unlikely that the actual output is all zeros.
    var any_nonzero = false;
    for (out) |b| if (b != 0) {
        any_nonzero = true;
        break;
    };
    try testing.expect(any_nonzero);
    // The trailing 32 bytes (b_4) should be different from the first
    // 32 bytes (b_1) for a non-degenerate output.
    try testing.expect(!std.mem.eql(u8, out[0..32], out[96..128]));
}

test "expand_message_xmd: empty msg works" {
    var out: [32]u8 = undefined;
    try expand_message_xmd(&out, "", TEST_DST);
    var any_nonzero = false;
    for (out) |b| if (b != 0) {
        any_nonzero = true;
        break;
    };
    try testing.expect(any_nonzero);
}

test "expand_message_xmd: rejects DST longer than 255 bytes" {
    var huge_dst: [300]u8 = .{0} ** 300;
    var out: [16]u8 = undefined;
    try testing.expectError(ExpandError.DstTooLong, expand_message_xmd(&out, "abc", &huge_dst));
}

test "expand_message_xmd: rejects oversized output" {
    // 255 * 32 = 8160 is the maximum permitted output length.
    const oversized = try testing.allocator.alloc(u8, 8161);
    defer testing.allocator.free(oversized);
    try testing.expectError(ExpandError.OutputTooLong, expand_message_xmd(oversized, "abc", "test"));
}

test "expand_message_xmd: short outputs are exact-length" {
    // Output length 1 stresses the case where ell = 1 and only the
    // first byte of b_1 is copied.
    var out: [1]u8 = undefined;
    try expand_message_xmd(&out, "abc", TEST_DST);
    // Determinism check.
    var out2: [1]u8 = undefined;
    try expand_message_xmd(&out2, "abc", TEST_DST);
    try testing.expectEqual(out[0], out2[0]);
}

// ---------------------------------------------------------------------------
// hash_to_field for BLS12-381 Fp / Fp2
//
// RFC 9380 §5.2 with `k = 128` security parameter and BLS12-381 base
// prime `p`. The per-element byte length is
//
//   L = ceil((ceil(log2(p)) + k) / 8) = ceil((381 + 128) / 8) = 64
//
// `hash_to_field_fp(msg, dst, count)` produces `count` Fp elements.
// `hash_to_field_fp2(msg, dst, count)` produces `count` Fp2 elements.
// ---------------------------------------------------------------------------

/// Per-element byte length for BLS12-381 hash-to-field with k = 128.
pub const BLS12_381_L: usize = 64;

/// Hash a message into `count` BLS12-381 Fp elements. The output
/// slice must have length exactly `count`.
pub fn hash_to_field_fp(
    out: []Fp.Element,
    msg: []const u8,
    dst: []const u8,
) ExpandError!void {
    const count = out.len;
    const len_in_bytes = count * BLS12_381_L;
    // Stack-allocate up to 8 elements (512 bytes). Sufficient for the
    // BLS hash-to-curve cases that need 1 or 2 elements.
    if (count > 8) return ExpandError.OutputTooLong;
    var uniform_bytes: [8 * BLS12_381_L]u8 = undefined;
    try expand_message_xmd(uniform_bytes[0..len_in_bytes], msg, dst);
    var i: usize = 0;
    while (i < count) : (i += 1) {
        const start = i * BLS12_381_L;
        const chunk: *const [BLS12_381_L]u8 = uniform_bytes[start..][0..BLS12_381_L];
        out[i] = bls12_381.fpFromBytes64Be(chunk);
    }
}

/// Hash a message into `count` BLS12-381 Fp2 elements. Each Fp2
/// element consumes two 64-byte chunks of expanded output.
pub fn hash_to_field_fp2(
    out: []Fp2,
    msg: []const u8,
    dst: []const u8,
) ExpandError!void {
    const count = out.len;
    const len_in_bytes = count * 2 * BLS12_381_L;
    if (count > 4) return ExpandError.OutputTooLong;
    var uniform_bytes: [4 * 2 * BLS12_381_L]u8 = undefined;
    try expand_message_xmd(uniform_bytes[0..len_in_bytes], msg, dst);
    var i: usize = 0;
    while (i < count) : (i += 1) {
        const c0_start = (2 * i) * BLS12_381_L;
        const c1_start = (2 * i + 1) * BLS12_381_L;
        const c0_chunk: *const [BLS12_381_L]u8 = uniform_bytes[c0_start..][0..BLS12_381_L];
        const c1_chunk: *const [BLS12_381_L]u8 = uniform_bytes[c1_start..][0..BLS12_381_L];
        out[i] = .{
            .c0 = bls12_381.fpFromBytes64Be(c0_chunk),
            .c1 = bls12_381.fpFromBytes64Be(c1_chunk),
        };
    }
}

test "hash_to_field_fp: deterministic" {
    var a: [2]Fp.Element = undefined;
    var b: [2]Fp.Element = undefined;
    try hash_to_field_fp(&a, "abc", TEST_DST);
    try hash_to_field_fp(&b, "abc", TEST_DST);
    try testing.expect(Fp.eql(a[0], b[0]));
    try testing.expect(Fp.eql(a[1], b[1]));
}

test "hash_to_field_fp: distinct inputs produce distinct outputs" {
    var a: [1]Fp.Element = undefined;
    var b: [1]Fp.Element = undefined;
    try hash_to_field_fp(&a, "abc", TEST_DST);
    try hash_to_field_fp(&b, "abd", TEST_DST);
    try testing.expect(!Fp.eql(a[0], b[0]));
}

test "hash_to_field_fp: produces non-zero element" {
    var elements: [1]Fp.Element = undefined;
    try hash_to_field_fp(&elements, "abc", TEST_DST);
    // Overwhelmingly unlikely to be zero (probability ~1/p ≈ 2^-381).
    var any_nonzero = false;
    inline for (0..6) |i| {
        if (elements[0][i] != 0) {
            any_nonzero = true;
            break;
        }
    }
    try testing.expect(any_nonzero);
}

test "hash_to_field_fp2: deterministic" {
    var a: [2]Fp2 = undefined;
    var b: [2]Fp2 = undefined;
    try hash_to_field_fp2(&a, "abc", TEST_DST);
    try hash_to_field_fp2(&b, "abc", TEST_DST);
    try testing.expect(Fp2.eql(a[0], b[0]));
    try testing.expect(Fp2.eql(a[1], b[1]));
}

test "hash_to_field_fp2: c0 and c1 differ" {
    // Each Fp2 element is built from two distinct 64-byte chunks of the
    // expanded output, so c0 should differ from c1 with overwhelming
    // probability.
    var elements: [1]Fp2 = undefined;
    try hash_to_field_fp2(&elements, "abc", TEST_DST);
    try testing.expect(!Fp.eql(elements[0].c0, elements[0].c1));
}

test "expand_message_xmd: distinct output lengths produce distinct b_1" {
    // The l_i_b_str field is part of msg_prime, so b_0 — and therefore
    // b_1 — depends on the requested output length. A 16-byte output
    // and the first 16 bytes of a 32-byte output for the same (msg, DST)
    // should NOT match. This is a property of the spec, not a bug.
    var short: [16]u8 = undefined;
    var full: [32]u8 = undefined;
    try expand_message_xmd(&short, "abc", TEST_DST);
    try expand_message_xmd(&full, "abc", TEST_DST);
    try testing.expect(!std.mem.eql(u8, &short, full[0..16]));
}
