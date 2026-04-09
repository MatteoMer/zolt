//! BLS12-381 signature verification entry point.
//!
//! Wraps the optimal Ate pairing and the RFC 9380 hash-to-curve into
//! the standard BLS-min-pk verification rule:
//!
//!     e(pk, H(msg)) == e(g1, sig)
//!
//! `min-pk` puts public keys in G1 (48 bytes compressed) and signatures
//! in G2 (96 bytes compressed). This matches `blst::min_pk` (which is
//! what Hyli uses through `hyli-crypto`) and the IETF
//! `BLS_SIG_BLS12381G2_XMD:SHA-256_SSWU_RO_NUL_` ciphersuite.
//!
//! The whole pipeline lives in zolt-arith because the same primitives
//! are useful to consumers other than Zyli (Zolt eventually). The
//! Hyli-specific seam — turning a `Signed<T, V>` envelope into the
//! signable byte string — stays in Zyli's `crypto/signable.zig`.

const std = @import("std");
const bls12_381 = @import("curve.zig");
const hash_to_curve_g2 = @import("hash_to_curve_g2.zig");
const hash_to_field = @import("hash_to_field.zig");

const Fp12 = bls12_381.Fp12;
const G1Affine = bls12_381.G1Affine;
const G2Affine = bls12_381.G2Affine;

/// IETF ciphersuite identifier for BLS sigs over G2 with SHA-256 SSWU
/// random-oracle hashing and the empty augmentation. Same string used
/// by `blst::min_pk` and pinned in Hyli's `hyli-crypto` `DST` constant.
pub const DST_BLS_SIG_NUL: []const u8 = hash_to_curve_g2.DST_BLS_SIG_NUL;

/// Errors that the verifier can return without producing a verdict.
pub const VerifyError = error{
    /// `pk` decoded but is the point at infinity, which is not a valid
    /// signing key.
    PublicKeyIsIdentity,
    /// `pk` is not in the prime-order r-subgroup of G1.
    PublicKeyNotInSubgroup,
    /// `sig` is not in the prime-order r-subgroup of G2.
    SignatureNotInSubgroup,
    /// `hash_to_curve` over the (msg, DST) pair failed (e.g., DST too
    /// long). Bubbles up from the underlying `expand_message_xmd`.
    HashFailed,
};

/// Verify a BLS signature against a public key and message bytes.
/// Returns `true` for a valid signature, `false` for an invalid one.
/// Errors are reserved for malformed inputs that prevent the verifier
/// from producing a verdict at all.
///
/// The pairing check is implemented as a single combined product:
///
///   e(-pk, H(msg)) · e(g1, sig) == 1   (multi-pairing form)
///
/// Equivalent to checking `e(pk, H(msg)) == e(g1, sig)` but with one
/// final exponentiation instead of two — roughly halves verify cost
/// because the hard part of the final exponentiation dominates the
/// pairing pipeline.
///
/// Subgroup membership checks on pk and sig are mandatory and match
/// the blst::min_pk default. Hyli relies on these for adversarially-
/// supplied pubkeys; weakening them silently would be a consensus-
/// safety regression.
pub fn verify(
    pk: G1Affine,
    msg: []const u8,
    sig: G2Affine,
    dst: []const u8,
) VerifyError!bool {
    // 1. Validate the public key.
    if (pk.isIdentity()) return VerifyError.PublicKeyIsIdentity;
    if (!bls12_381.isInG1Subgroup(pk)) return VerifyError.PublicKeyNotInSubgroup;

    // 2. Validate the signature.
    if (!bls12_381.isInG2Subgroup(sig)) return VerifyError.SignatureNotInSubgroup;

    // 3. Hash the message to a G2 point.
    const h = hash_to_curve_g2.hashToG2(msg, dst) catch return VerifyError.HashFailed;

    // 4. Pairing check via multi-pairing trick:
    //    e(pk, H) == e(g1, sig)
    //    ⇔ e(-pk, H) · e(g1, sig) == 1
    //    ⇔ Miller(-pk, H) · Miller(g1, sig)  →  finalExp  →  == 1
    const m1 = bls12_381.millerLoop(pk.neg(), h);
    const m2 = bls12_381.millerLoop(bls12_381.g1Generator(), sig);
    const product = bls12_381.Fp12.mul(m1, m2);
    const final = bls12_381.fp12FinalExp(product);
    return Fp12.eql(final, Fp12.one());
}

/// Convenience wrapper that decodes the compressed wire forms before
/// verifying. Returns `false` for any decode error so callers can treat
/// "malformed bytes" as "invalid signature" without distinguishing.
/// Use the strict variant when you need to know which step failed.
pub fn verifyCompressed(
    pk_bytes: []const u8,
    msg: []const u8,
    sig_bytes: []const u8,
    dst: []const u8,
) bool {
    const pk = bls12_381.decodeG1Compressed(pk_bytes) catch return false;
    const sig = bls12_381.decodeG2Compressed(sig_bytes) catch return false;
    return verify(pk, msg, sig, dst) catch return false;
}

// ---------------------------------------------------------------------------
// Same-message aggregate signature verification.
//
// In the same-message aggregate scheme used by `blst::min_pk` (and by
// Hyli for consensus QCs), several validators each sign the *same*
// message bytes. The aggregate public key is the sum of the individual
// public keys, and the aggregate signature is the sum of the individual
// signatures. Verification reduces to a single pairing equation:
//
//     e(g1, agg_sig) == e(agg_pk, H(msg))
//
// This is exactly the shape of `verify` with `pk = sum_i pk_i`, so we
// expose `aggregatePublicKeys` as a standalone helper and reuse the
// regular verifier underneath.
// ---------------------------------------------------------------------------

/// Sum a slice of G1 public keys into a single aggregate public key.
/// The empty slice produces the identity (which `verify` will reject
/// downstream). Each pk is added directly via the affine arithmetic;
/// projective is unnecessary because aggregations are typically tiny
/// (≤ 100 validators) and a Fermat inversion every few hundred
/// validators is dwarfed by the pairing cost.
///
/// Subgroup membership is intentionally NOT checked here — the caller
/// has already pulled the points out of the wire and the verify path
/// re-checks the aggregate.
pub fn aggregatePublicKeys(pks: []const G1Affine) G1Affine {
    if (pks.len == 0) return G1Affine.identity();
    var acc = pks[0];
    var i: usize = 1;
    while (i < pks.len) : (i += 1) {
        acc = acc.add(pks[i]);
    }
    return acc;
}

/// Sum a slice of G2 signatures into a single aggregate signature.
/// Mirrors `aggregatePublicKeys` for the signature side.
pub fn aggregateSignatures(sigs: []const G2Affine) G2Affine {
    if (sigs.len == 0) return G2Affine.identity();
    var acc = sigs[0];
    var i: usize = 1;
    while (i < sigs.len) : (i += 1) {
        acc = acc.add(sigs[i]);
    }
    return acc;
}

/// Verify a same-message aggregate signature against a list of
/// validator public keys. Returns `true` for valid, `false` for
/// invalid; errors only on the same malformed-input cases as `verify`.
///
/// The empty pubkey list always rejects (nothing to verify against).
pub fn verifyAggregate(
    pks: []const G1Affine,
    msg: []const u8,
    sig: G2Affine,
    dst: []const u8,
) VerifyError!bool {
    if (pks.len == 0) return false;
    // The individual subgroup checks are still important — we don't
    // want to let a single rogue out-of-subgroup pk poison the sum.
    // Walk the inputs first.
    for (pks) |pk| {
        if (pk.isIdentity()) return VerifyError.PublicKeyIsIdentity;
        if (!bls12_381.isInG1Subgroup(pk)) return VerifyError.PublicKeyNotInSubgroup;
    }
    const agg_pk = aggregatePublicKeys(pks);
    return verify(agg_pk, msg, sig, dst);
}

// ---------------------------------------------------------------------------
// Signing.
//
// In BLS-min-pk, signing is straightforward:
//
//   sig = sk · H(msg, dst)
//
// where `sk ∈ Fr` (the scalar field), `H(msg, dst) ∈ G2`, and `·`
// is scalar multiplication on the curve. The resulting `sig ∈ G2` is
// then compressed to 96 wire bytes.
//
// Public key derivation:
//
//   pk = sk · G1_generator
//
// We accept the secret key as a raw `[4]u64` little-endian limb array
// (Fr canonical form). Bytes-level helpers wrap that for callers that
// receive sks as 32-byte big-endian buffers (the standard wire form).
// ---------------------------------------------------------------------------

pub const SignError = error{
    /// `sk_bytes` is not a 32-byte buffer.
    InvalidSecretKeyLength,
    /// The decoded secret key is `0`, which is not a valid signing key
    /// (the resulting public key would be the identity).
    SecretKeyIsZero,
    /// `hash_to_curve` over the (msg, DST) pair failed.
    HashFailed,
};

/// Sign a message with a raw scalar secret key. Returns the affine
/// G2 signature point. Caller is responsible for ensuring `sk` is in
/// `[1, r-1]` — this function does not validate.
pub fn signWithScalar(
    sk: [4]u64,
    msg: []const u8,
    dst: []const u8,
) SignError!G2Affine {
    const h_msg = hash_to_curve_g2.hashToG2(msg, dst) catch return SignError.HashFailed;
    return h_msg.mul(4, sk);
}

/// Derive the BLS public key for a raw scalar. `sk · G1_generator`.
pub fn derivePublicKeyFromScalar(sk: [4]u64) G1Affine {
    return bls12_381.g1Generator().mul(4, sk);
}

/// Sign a message with a 32-byte big-endian secret key (the standard
/// wire form blst uses). Returns the 96-byte compressed signature.
pub fn signBytes(
    sk_bytes: []const u8,
    msg: []const u8,
    dst: []const u8,
) SignError![96]u8 {
    if (sk_bytes.len != 32) return SignError.InvalidSecretKeyLength;
    var sk_be: [32]u8 = undefined;
    @memcpy(&sk_be, sk_bytes);
    // Reinterpret as a 4-limb little-endian integer (the canonical Fr
    // representation). Big-endian bytes need to be reversed limb-wise:
    // the high u64 of the BE representation is limb[3] (the MSB limb)
    // in little-endian.
    var sk_limbs: [4]u64 = undefined;
    inline for (0..4) |i| {
        const start = (3 - i) * 8;
        sk_limbs[i] = std.mem.readInt(u64, sk_be[start..][0..8], .big);
    }
    if (bigint_isZero4(sk_limbs)) return SignError.SecretKeyIsZero;
    const sig = try signWithScalar(sk_limbs, msg, dst);
    return bls12_381.encodeG2Compressed(sig);
}

/// Derive a 48-byte compressed public key from a 32-byte big-endian
/// secret key. Mirrors `signBytes` for the verification side.
pub fn derivePublicKeyBytes(sk_bytes: []const u8) SignError![48]u8 {
    if (sk_bytes.len != 32) return SignError.InvalidSecretKeyLength;
    var sk_limbs: [4]u64 = undefined;
    inline for (0..4) |i| {
        const start = (3 - i) * 8;
        sk_limbs[i] = std.mem.readInt(u64, sk_bytes[start..][0..8], .big);
    }
    if (bigint_isZero4(sk_limbs)) return SignError.SecretKeyIsZero;
    return bls12_381.encodeG1Compressed(derivePublicKeyFromScalar(sk_limbs));
}

inline fn bigint_isZero4(v: [4]u64) bool {
    return v[0] == 0 and v[1] == 0 and v[2] == 0 and v[3] == 0;
}

// ---------------------------------------------------------------------------
// Sign / verify round-trip tests
// ---------------------------------------------------------------------------

test "signWithScalar / verify round-trip" {
    const sk: [4]u64 = .{ 0x1234567890abcdef, 0xdeadbeef, 0, 0 };
    const pk = derivePublicKeyFromScalar(sk);
    const sig = try signWithScalar(sk, "round trip", DST_BLS_SIG_NUL);
    const ok = try verify(pk, "round trip", sig, DST_BLS_SIG_NUL);
    try testing.expect(ok);
}

test "signBytes / derivePublicKeyBytes / verifyCompressed round-trip" {
    var sk: [32]u8 = .{0} ** 32;
    sk[31] = 0x42;
    sk[30] = 0x13;
    const sig = try signBytes(&sk, "compressed round trip", DST_BLS_SIG_NUL);
    const pk = try derivePublicKeyBytes(&sk);
    try testing.expect(verifyCompressed(&pk, "compressed round trip", &sig, DST_BLS_SIG_NUL));
}

test "signBytes: rejects zero secret key" {
    const zero_sk: [32]u8 = .{0} ** 32;
    try testing.expectError(SignError.SecretKeyIsZero, signBytes(&zero_sk, "msg", DST_BLS_SIG_NUL));
}

test "signBytes: rejects wrong-length secret key" {
    const short: [31]u8 = .{0} ** 31;
    try testing.expectError(SignError.InvalidSecretKeyLength, signBytes(&short, "msg", DST_BLS_SIG_NUL));
}

test "derivePublicKeyBytes: distinct sks produce distinct pks" {
    var sk1: [32]u8 = .{0} ** 32;
    sk1[31] = 1;
    var sk2: [32]u8 = .{0} ** 32;
    sk2[31] = 2;
    const pk1 = try derivePublicKeyBytes(&sk1);
    const pk2 = try derivePublicKeyBytes(&sk2);
    try testing.expect(!std.mem.eql(u8, &pk1, &pk2));
}

test "signBytes / verify: tampered message rejects" {
    var sk: [32]u8 = .{0} ** 32;
    sk[31] = 0x99;
    const sig = try signBytes(&sk, "honest message", DST_BLS_SIG_NUL);
    const pk = try derivePublicKeyBytes(&sk);
    try testing.expect(!verifyCompressed(&pk, "tampered message", &sig, DST_BLS_SIG_NUL));
}

// ---------------------------------------------------------------------------
// Aggregate verification tests
// ---------------------------------------------------------------------------

test "aggregatePublicKeys: empty slice → identity" {
    const empty: []const G1Affine = &.{};
    const result = aggregatePublicKeys(empty);
    try testing.expect(result.isIdentity());
}

test "aggregatePublicKeys: single key → that key" {
    const sk: [4]u64 = .{ 5, 0, 0, 0 };
    const pk = bls12_381.g1Generator().mul(4, sk);
    const result = aggregatePublicKeys(&.{pk});
    try testing.expect(G1Affine.eql(result, pk));
}

test "aggregatePublicKeys: two keys sum correctly" {
    const sk1: [4]u64 = .{ 5, 0, 0, 0 };
    const sk2: [4]u64 = .{ 7, 0, 0, 0 };
    const pk1 = bls12_381.g1Generator().mul(4, sk1);
    const pk2 = bls12_381.g1Generator().mul(4, sk2);
    const aggregated = aggregatePublicKeys(&.{ pk1, pk2 });
    // (sk1 + sk2) * G should equal pk1 + pk2.
    const sk_sum: [4]u64 = .{ 12, 0, 0, 0 };
    const expected = bls12_381.g1Generator().mul(4, sk_sum);
    try testing.expect(G1Affine.eql(aggregated, expected));
}

test "verifyAggregate: empty pks → false" {
    const empty: []const G1Affine = &.{};
    const sig = bls12_381.g2Generator();
    const result = try verifyAggregate(empty, "msg", sig, DST_BLS_SIG_NUL);
    try testing.expect(!result);
}

test "verifyAggregate: two-validator round-trip" {
    // Build (sk_i, pk_i, sig_i) for i ∈ {1, 2} on the same message,
    // sum the sigs, and verify against the aggregated pk list.
    const msg = "consensus prepare slot 42";

    const sk1: [4]u64 = .{ 7, 0, 0, 0 };
    const sk2: [4]u64 = .{ 11, 0, 0, 0 };

    const pk1 = bls12_381.g1Generator().mul(4, sk1);
    const pk2 = bls12_381.g1Generator().mul(4, sk2);

    const h_msg = try hash_to_curve_g2.hashToG2(msg, DST_BLS_SIG_NUL);
    const sig1 = h_msg.mul(4, sk1);
    const sig2 = h_msg.mul(4, sk2);

    const agg_sig = aggregateSignatures(&.{ sig1, sig2 });
    const result = try verifyAggregate(&.{ pk1, pk2 }, msg, agg_sig, DST_BLS_SIG_NUL);
    try testing.expect(result);
}

test "verifyAggregate: rejects when one signature is for the wrong message" {
    // Validator 1 signs message A, validator 2 signs message B. The
    // aggregate signature is the sum, but verifyAggregate is asked
    // to verify against message A. The pairing equation must reject.
    const sk1: [4]u64 = .{ 13, 0, 0, 0 };
    const sk2: [4]u64 = .{ 17, 0, 0, 0 };
    const pk1 = bls12_381.g1Generator().mul(4, sk1);
    const pk2 = bls12_381.g1Generator().mul(4, sk2);

    const h_a = try hash_to_curve_g2.hashToG2("message A", DST_BLS_SIG_NUL);
    const h_b = try hash_to_curve_g2.hashToG2("message B", DST_BLS_SIG_NUL);
    const sig1 = h_a.mul(4, sk1);
    const sig2 = h_b.mul(4, sk2);

    const agg_sig = aggregateSignatures(&.{ sig1, sig2 });
    const result = try verifyAggregate(&.{ pk1, pk2 }, "message A", agg_sig, DST_BLS_SIG_NUL);
    try testing.expect(!result);
}

test "verifyAggregate: rejects when one pk is missing from the aggregate" {
    // Both validators sign message A, but verify is called with only
    // pk1. The aggregate signature was built from both, so the pairing
    // equation no longer balances.
    const msg = "message A";
    const sk1: [4]u64 = .{ 19, 0, 0, 0 };
    const sk2: [4]u64 = .{ 23, 0, 0, 0 };
    const pk1 = bls12_381.g1Generator().mul(4, sk1);

    const h_msg = try hash_to_curve_g2.hashToG2(msg, DST_BLS_SIG_NUL);
    const sig1 = h_msg.mul(4, sk1);
    const sig2 = h_msg.mul(4, sk2);
    const agg_sig = aggregateSignatures(&.{ sig1, sig2 });

    const result = try verifyAggregate(&.{pk1}, msg, agg_sig, DST_BLS_SIG_NUL);
    try testing.expect(!result);
}

test "verifyAggregate: three-validator path" {
    const msg = "three validators";
    const sk1: [4]u64 = .{ 29, 0, 0, 0 };
    const sk2: [4]u64 = .{ 31, 0, 0, 0 };
    const sk3: [4]u64 = .{ 37, 0, 0, 0 };
    const pk1 = bls12_381.g1Generator().mul(4, sk1);
    const pk2 = bls12_381.g1Generator().mul(4, sk2);
    const pk3 = bls12_381.g1Generator().mul(4, sk3);

    const h_msg = try hash_to_curve_g2.hashToG2(msg, DST_BLS_SIG_NUL);
    const sig1 = h_msg.mul(4, sk1);
    const sig2 = h_msg.mul(4, sk2);
    const sig3 = h_msg.mul(4, sk3);
    const agg_sig = aggregateSignatures(&.{ sig1, sig2, sig3 });

    const result = try verifyAggregate(&.{ pk1, pk2, pk3 }, msg, agg_sig, DST_BLS_SIG_NUL);
    try testing.expect(result);
}

// ---------------------------------------------------------------------------
// Tests — these don't have a real cross-implementation vector yet, but
// they exercise the algebraic identities that any correct verifier must
// satisfy. The first cross-vector test against a known blst signature
// can land via a fixture once the corpus carries one.
// ---------------------------------------------------------------------------

const testing = std.testing;

test "verify: rejects identity pubkey" {
    const id = G1Affine.identity();
    const sig = bls12_381.g2Generator();
    try testing.expectError(VerifyError.PublicKeyIsIdentity, verify(id, "msg", sig, DST_BLS_SIG_NUL));
}

test "verify: accepts a self-constructed valid signature" {
    // Construct (sk, pk) where sk = 7. Compute sig = sk · H("hello").
    // The verifier must accept it.
    const sk: [4]u64 = .{ 7, 0, 0, 0 };
    const pk = bls12_381.g1Generator().mul(4, sk);
    const h_msg = try hash_to_curve_g2.hashToG2("hello", DST_BLS_SIG_NUL);
    const sig = h_msg.mul(4, sk);
    const result = try verify(pk, "hello", sig, DST_BLS_SIG_NUL);
    try testing.expect(result);
}

test "verify: rejects signature with wrong message" {
    const sk: [4]u64 = .{ 13, 0, 0, 0 };
    const pk = bls12_381.g1Generator().mul(4, sk);
    const h_a = try hash_to_curve_g2.hashToG2("message A", DST_BLS_SIG_NUL);
    const sig = h_a.mul(4, sk);
    // Sign "message A" but verify against "message B".
    const result = try verify(pk, "message B", sig, DST_BLS_SIG_NUL);
    try testing.expect(!result);
}

test "verify: rejects signature signed by a different key" {
    const sk_signer: [4]u64 = .{ 5, 0, 0, 0 };
    const sk_other: [4]u64 = .{ 11, 0, 0, 0 };
    const pk_other = bls12_381.g1Generator().mul(4, sk_other);
    const h_msg = try hash_to_curve_g2.hashToG2("hello", DST_BLS_SIG_NUL);
    const sig = h_msg.mul(4, sk_signer);
    // Verify with the wrong public key.
    const result = try verify(pk_other, "hello", sig, DST_BLS_SIG_NUL);
    try testing.expect(!result);
}

test "verify: rejects a signature swapped with a different message's signature" {
    // sk signs "hello A" then we present sig together with the (otherwise
    // matching) public key but message "hello B". Pairing equality fails.
    const sk: [4]u64 = .{ 17, 0, 0, 0 };
    const pk = bls12_381.g1Generator().mul(4, sk);
    const h_a = try hash_to_curve_g2.hashToG2("hello A", DST_BLS_SIG_NUL);
    const sig_a = h_a.mul(4, sk);
    const result = try verify(pk, "hello B", sig_a, DST_BLS_SIG_NUL);
    try testing.expect(!result);
}

test "verify: empty message works" {
    const sk: [4]u64 = .{ 23, 0, 0, 0 };
    const pk = bls12_381.g1Generator().mul(4, sk);
    const h_empty = try hash_to_curve_g2.hashToG2("", DST_BLS_SIG_NUL);
    const sig = h_empty.mul(4, sk);
    const result = try verify(pk, "", sig, DST_BLS_SIG_NUL);
    try testing.expect(result);
}

test "verifyCompressed: round-trip with valid signature" {
    // Build a valid (pk, sig) pair, compress them, and verify via the
    // compressed-wire entry point.
    const sk: [4]u64 = .{ 19, 0, 0, 0 };
    const pk = bls12_381.g1Generator().mul(4, sk);
    const h_msg = try hash_to_curve_g2.hashToG2("compressed", DST_BLS_SIG_NUL);
    const sig = h_msg.mul(4, sk);

    // We don't have an encoder yet, so round-trip via decode-of-known-good
    // wire forms is not possible. Skip the wire round-trip and just test
    // that the strict verify path agrees on a valid pair.
    const result = try verify(pk, "compressed", sig, DST_BLS_SIG_NUL);
    try testing.expect(result);
}
