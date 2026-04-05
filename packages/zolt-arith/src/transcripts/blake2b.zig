//! Blake2b Fiat-Shamir Transcript for Jolt Compatibility
//!
//! This module implements a Blake2b-based Fiat-Shamir transcript that is
//! byte-compatible with Jolt's Blake2bTranscript (Rust) at commit 97b2c96.
//!
//! Key compatibility requirements:
//! - Uses Blake2b-256 (32-byte output)
//! - 32-byte state with round counter for domain separation
//! - All public methods take a label as first argument
//! - Labels are right-padded to 32 bytes with zeros
//! - Variable-length methods pack label (24 bytes) + length (8 bytes BE) into 32 bytes
//! - Scalars are serialized LE then reversed to BE (EVM format)
//! - Challenges are 128-bit (16 bytes)

const std = @import("std");
const testdata = @import("../testdata.zig");

// Debug output control - set to true to enable verbose debug prints
const debug_verbose = false;
fn dbg(comptime fmt: []const u8, args: anytype) void {
    if (debug_verbose) std.debug.print(fmt, args);
}

const crypto = std.crypto;
const Blake2b256 = crypto.hash.blake2.Blake2b256;
const mem = std.mem;

/// Maximum label length when packed with a length/count.
/// 32 bytes total - 8 bytes for u64 length = 24 bytes for label.
const MAX_LABEL_LEN_WITH_LENGTH: usize = 24;

/// Blake2b-based Fiat-Shamir transcript compatible with Jolt (upstream 97b2c96)
///
/// This matches Jolt's Blake2bTranscript exactly:
/// - 32-byte running state
/// - Round counter (u32) for domain separation
/// - hasher() = Blake2b256(state || packed_round) where packed_round = [0u8; 28] || round.to_be_bytes()
/// - All public methods take a label argument for domain separation
pub fn Blake2bTranscript(comptime F: type) type {
    return struct {
        const Self = @This();

        /// 256-bit running state (matches Jolt's `state: [u8; 32]`)
        state: [32]u8,
        /// Round counter for domain separation (matches Jolt's `n_rounds: u32`)
        n_rounds: u32,

        /// Create a new transcript with a label
        /// Matches Jolt's `fn new(label: &'static [u8]) -> Self`
        pub fn init(label: []const u8) Self {
            std.debug.assert(label.len < 33);

            // Pad label to 32 bytes with zeros on the right
            var padded: [32]u8 = [_]u8{0} ** 32;
            const copy_len = @min(label.len, 32);
            @memcpy(padded[0..copy_len], label[0..copy_len]);

            // Hash the padded label to get initial state
            var h = Blake2b256.init(.{});
            h.update(&padded);
            var initial_state: [32]u8 = undefined;
            h.final(&initial_state);

            return Self{
                .state = initial_state,
                .n_rounds = 0,
            };
        }

        /// Get the hasher with running seed and index added
        /// Matches Jolt's `fn hasher(&self) -> Blake2b256`
        fn hasher(self: *const Self) Blake2b256 {
            var h = Blake2b256.init(.{});
            h.update(&self.state);

            // Create round_data: 28 zero bytes + 4-byte BE round counter
            var round_data: [32]u8 = [_]u8{0} ** 32;
            mem.writeInt(u32, round_data[28..32], self.n_rounds, .big);
            h.update(&round_data);

            return h;
        }

        /// Update the state with a new 32-byte value and increment round counter
        fn updateState(self: *Self, new_state: [32]u8) void {
            self.state = new_state;
            self.n_rounds += 1;
        }

        // =====================================================================
        // Internal (raw) methods - match Jolt's raw_* methods
        // =====================================================================

        /// Raw append a label (right-padded to 32 bytes).
        /// Matches Jolt's `fn raw_append_label(&mut self, label: &'static [u8])`
        fn rawAppendLabel(self: *Self, label: []const u8) void {
            std.debug.assert(label.len < 33);

            var h = self.hasher();

            if (label.len == 32) {
                h.update(label);
            } else {
                var padded: [32]u8 = [_]u8{0} ** 32;
                @memcpy(padded[0..label.len], label);
                h.update(&padded);
            }

            var result: [32]u8 = undefined;
            h.final(&result);
            self.updateState(result);
        }

        /// Raw append label with length packed into 32 bytes.
        /// Label (up to 24 bytes, right-padded) + length (8 bytes, big-endian).
        /// Matches Jolt's `fn raw_append_label_with_len(&mut self, label: &'static [u8], len: u64)`
        fn rawAppendLabelWithLen(self: *Self, label: []const u8, len: u64) void {
            std.debug.assert(label.len <= MAX_LABEL_LEN_WITH_LENGTH);
            var label_buf: [32]u8 = [_]u8{0} ** 32;
            @memcpy(label_buf[0..label.len], label);
            // Zero-pad label portion (already zeroed)
            // Append length as big-endian in last 8 bytes
            mem.writeInt(u64, label_buf[24..32], len, .big);
            self.rawAppendBytes(&label_buf);
        }

        /// Raw append bytes (hash directly).
        /// Matches Jolt's `fn raw_append_bytes(&mut self, bytes: &[u8])`
        fn rawAppendBytes(self: *Self, bytes: []const u8) void {
            var h = self.hasher();
            h.update(bytes);

            var result: [32]u8 = undefined;
            h.final(&result);
            self.updateState(result);
        }

        /// Raw append u64 packed into 32 bytes: [0u8; 24] ++ x.to_be_bytes()
        /// Matches Jolt's `fn raw_append_u64(&mut self, x: u64)`
        fn rawAppendU64(self: *Self, x: u64) void {
            var data: [32]u8 = [_]u8{0} ** 32;
            mem.writeInt(u64, data[24..32], x, .big);

            var h = self.hasher();
            h.update(&data);

            var result: [32]u8 = undefined;
            h.final(&result);
            self.updateState(result);
        }

        /// Raw append scalar: serialize LE, reverse to BE, raw_append_bytes.
        /// Matches Jolt's `fn raw_append_scalar<F>(&mut self, scalar: &F)`
        fn rawAppendScalar(self: *Self, scalar: F) void {
            // Convert from Montgomery form to canonical form
            const standard = scalar.fromMontgomery();

            // Serialize to LE bytes
            var buf: [32]u8 = undefined;
            for (0..4) |i| {
                mem.writeInt(u64, buf[i * 8 ..][0..8], standard.limbs[i], .little);
            }

            // Reverse to BE for EVM compatibility
            var reversed: [32]u8 = undefined;
            for (0..32) |i| {
                reversed[i] = buf[31 - i];
            }

            self.rawAppendBytes(&reversed);
        }

        // =====================================================================
        // Public API - all take a label as first argument
        // =====================================================================

        /// Append a domain-separation label with no associated data.
        /// Matches Jolt's `fn append_label(&mut self, label: &'static [u8])`
        pub fn appendLabel(self: *Self, label: []const u8) void {
            self.rawAppendLabel(label);
        }

        /// Append raw bytes with a label.
        /// Variable-length: label and length packed into single 32-byte word.
        /// Matches Jolt's `fn append_bytes(&mut self, label: &'static [u8], bytes: &[u8])`
        pub fn appendBytes(self: *Self, label: []const u8, bytes: []const u8) void {
            self.rawAppendLabelWithLen(label, bytes.len);
            self.rawAppendBytes(bytes);
        }

        /// Append a u64 value with a label.
        /// Two separate 32-byte words: label (right-padded) + value (left-padded).
        /// Matches Jolt's `fn append_u64(&mut self, label: &'static [u8], x: u64)`
        pub fn appendU64(self: *Self, label: []const u8, x: u64) void {
            self.rawAppendLabel(label);
            self.rawAppendU64(x);
        }

        /// Append a scalar field element with a label.
        /// Matches Jolt's `fn append_scalar(&mut self, label: &'static [u8], scalar: &F)`
        pub fn appendScalar(self: *Self, label: []const u8, scalar: F) void {
            self.rawAppendLabel(label);
            self.rawAppendScalar(scalar);
        }

        /// Append multiple scalars with a label.
        /// Variable-length: label and count packed into single 32-byte word.
        /// Matches Jolt's `fn append_scalars(&mut self, label: &'static [u8], scalars: &[F])`
        pub fn appendScalars(self: *Self, label: []const u8, scalars: []const F) void {
            self.rawAppendLabelWithLen(label, scalars.len);
            for (scalars) |scalar| {
                self.rawAppendScalar(scalar);
            }
        }

        /// Append a serializable object with a label.
        /// Variable-length: label and length packed into single 32-byte word.
        /// Reverses bytes after serialization for EVM compatibility.
        /// Matches Jolt's `fn append_serializable(&mut self, label: &'static [u8], data: &T)`
        pub fn appendSerializable(self: *Self, label: []const u8, comptime len: usize, bytes: *const [len]u8) void {
            self.rawAppendLabelWithLen(label, len);
            // Reverse the bytes to match Jolt's behavior
            var reversed: [len]u8 = undefined;
            for (0..len) |i| {
                reversed[i] = bytes[len - 1 - i];
            }
            self.rawAppendBytes(&reversed);
        }

        /// Append a GT (Fp12) element to the transcript with a label.
        /// Uses append_serializable semantics (label+len packed, bytes reversed).
        /// Matches Jolt's `transcript.append_serializable(b"commitment", &gt_element)`
        pub fn appendGT(self: *Self, label: []const u8, gt: anytype) void {
            const bytes = gt.toBytes();
            self.appendSerializable(label, 384, &bytes);
        }

        // =====================================================================
        // Challenge generation methods (unchanged from old API)
        // =====================================================================

        /// Get 32 challenge bytes from the transcript
        fn challengeBytes32(self: *Self, out: *[32]u8) void {
            var h = self.hasher();
            h.final(out);
            self.updateState(out.*);
        }

        /// Get arbitrary length challenge bytes
        pub fn challengeBytes(self: *Self, out: []u8) void {
            var remaining = out.len;
            var start: usize = 0;

            while (remaining > 32) {
                var full_rand: [32]u8 = undefined;
                self.challengeBytes32(&full_rand);
                @memcpy(out[start..][0..32], &full_rand);
                start += 32;
                remaining -= 32;
            }

            var full_rand: [32]u8 = undefined;
            self.challengeBytes32(&full_rand);
            @memcpy(out[start..][0..remaining], full_rand[0..remaining]);
        }

        /// Get a 128-bit challenge as u128
        /// Matches upstream Jolt: reverse bytes then from_be_bytes(reversed)
        /// = from_le_bytes(original)
        pub fn challengeU128(self: *Self) u128 {
            var buf: [16]u8 = undefined;
            self.challengeBytes(&buf);

            // Upstream: buf.reverse() then u128::from_be_bytes(buf)
            // from_be_bytes(reversed) = from_le_bytes(original)
            return mem.readInt(u128, &buf, .little);
        }

        /// Get a challenge scalar (128-bit, MontU128Challenge-compatible)
        /// Matches Jolt's `challenge_scalar_optimized` which uses MontU128Challenge.
        pub fn challengeScalar(self: *Self) F {
            return self.challengeScalar128Bits();
        }

        /// Get a challenge scalar with proper Montgomery conversion
        /// Matches upstream Jolt's `challenge_scalar` (reverse + from_le_bytes_mod_order = BE of original).
        pub fn challengeScalarFull(self: *Self) F {
            var buf: [16]u8 = undefined;
            self.challengeBytes(&buf);

            // Upstream: reverse bytes then from_le_bytes_mod_order
            // Equivalent to big-endian interpretation of original bytes
            const high: u64 = mem.readInt(u64, buf[0..8], .big);
            const low: u64 = mem.readInt(u64, buf[8..16], .big);

            const standard = F{ .limbs = .{ low, high, 0, 0 } };
            return standard.toMontgomery();
        }

        /// Get a 128-bit challenge scalar (masked to 125 bits for Jolt compatibility)
        /// Matches Jolt's `challenge_scalar_optimized` → MontU128Challenge::new()
        ///
        /// CRITICAL: from_bigint_unchecked does NOT do Montgomery conversion.
        /// The limbs [0, 0, L, H] ARE the Montgomery representation directly.
        pub fn challengeScalar128Bits(self: *Self) F {
            var buf: [16]u8 = undefined;
            self.challengeBytes(&buf);

            // Upstream: buf.reverse() then u128::from_be_bytes(reversed)
            // = from_le_bytes(original)
            const full_value: u128 = mem.readInt(u128, &buf, .little);

            // Mask to 125 bits (MontU128Challenge::new does: value & (u128::MAX >> 3))
            const mask_125: u128 = (1 << 125) - 1;
            const masked_value: u128 = full_value & mask_125;
            const masked_low: u64 = @truncate(masked_value);
            const masked_high: u64 = @truncate(masked_value >> 64);

            // Store [0, 0, L, H] directly WITHOUT toMontgomery - matches from_bigint_unchecked
            return F{ .limbs = .{ 0, 0, masked_low, masked_high } };
        }

        /// Get multiple challenge scalars (uses challengeScalarFull, NOT 125-bit)
        pub fn challengeVector(self: *Self, allocator: std.mem.Allocator, len: usize) ![]F {
            const challenges = try allocator.alloc(F, len);
            for (0..len) |i| {
                challenges[i] = self.challengeScalarFull();
            }
            return challenges;
        }

        /// Compute powers of a challenge scalar (uses challengeScalarFull, NOT 125-bit)
        pub fn challengeScalarPowers(self: *Self, allocator: std.mem.Allocator, len: usize) ![]F {
            const q = self.challengeScalarFull();
            const q_powers = try allocator.alloc(F, len);
            q_powers[0] = F.one();
            for (1..len) |i| {
                q_powers[i] = q_powers[i - 1].mul(q);
            }
            return q_powers;
        }

        /// Debug helper: return current transcript state
        pub fn debugState(self: *const Self) [32]u8 {
            return self.state;
        }
    };
}

// Tests
const testing = std.testing;
const BN254Scalar = @import("../field/mod.zig").BN254Scalar;

test "blake2b rfc7693 vectors" {
    const fixture_text = @embedFile("../testdata/transcripts/blake2b_rfc7693.txt");
    var lines = std.mem.splitScalar(u8, fixture_text, '\n');

    while (lines.next()) |raw_line| {
        const line = testdata.cleanLine(raw_line) orelse continue;
        const fields = try testdata.splitFieldsExact(3, line, '|');

        var input_buf: [256]u8 = undefined;
        const input = try testdata.parseHexIntoBuffer(fields[1], &input_buf);
        const expected_digest = try testdata.parseHexBytesExact(32, fields[2]);

        var hasher = Blake2b256.init(.{});
        hasher.update(input);

        var actual_digest: [32]u8 = undefined;
        hasher.final(&actual_digest);
        try testing.expectEqualSlices(u8, &expected_digest, &actual_digest);
    }
}

test "jolt compatibility: fixture-backed state vectors" {
    const Transcript = Blake2bTranscript(BN254Scalar);
    const fixture_text = @embedFile("../testdata/transcripts/blake2b_jolt_state_vectors.txt");
    var lines = std.mem.splitScalar(u8, fixture_text, '\n');

    while (lines.next()) |raw_line| {
        const line = testdata.cleanLine(raw_line) orelse continue;
        const fields = try testdata.splitFieldsExact(6, line, '|');

        var transcript = Transcript.init(fields[1]);
        const op_kind = fields[2];
        const op_arg = fields[3];

        if (mem.eql(u8, op_kind, "init")) {
            // No-op: expected state is the initialized transcript state.
        } else if (mem.eql(u8, op_kind, "raw_label")) {
            transcript.rawAppendLabel(op_arg);
        } else if (mem.eql(u8, op_kind, "raw_bytes_hex")) {
            var bytes_buf: [256]u8 = undefined;
            const bytes = try testdata.parseHexIntoBuffer(op_arg, &bytes_buf);
            transcript.rawAppendBytes(bytes);
        } else if (mem.eql(u8, op_kind, "raw_scalar_u64")) {
            transcript.rawAppendScalar(BN254Scalar.fromU64(try testdata.parseDecimal(u64, op_arg)));
        } else {
            return error.UnknownTranscriptVectorOperation;
        }

        const expected_state = try testdata.parseHexBytesExact(32, fields[4]);
        try testing.expectEqualSlices(u8, &expected_state, &transcript.state);
        try testing.expectEqual(try testdata.parseDecimal(u32, fields[5]), transcript.n_rounds);
    }
}

test "jolt compatibility: fixture-backed challenge vectors" {
    const Transcript = Blake2bTranscript(BN254Scalar);
    const fixture_text = @embedFile("../testdata/transcripts/blake2b_jolt_challenge_vectors.txt");
    var lines = std.mem.splitScalar(u8, fixture_text, '\n');

    while (lines.next()) |raw_line| {
        const line = testdata.cleanLine(raw_line) orelse continue;
        const fields = try testdata.splitFieldsExact(6, line, '|');

        if (!mem.eql(u8, fields[3], "-")) {
            var transcript_u128 = Transcript.init(fields[1]);
            transcript_u128.rawAppendLabel(fields[2]);
            try testing.expectEqual(try testdata.parseDecimal(u128, fields[3]), transcript_u128.challengeU128());
        }

        if (!mem.eql(u8, fields[4], "-") and !mem.eql(u8, fields[5], "-")) {
            var transcript_limbs = Transcript.init(fields[1]);
            transcript_limbs.rawAppendLabel(fields[2]);
            const challenge = transcript_limbs.challengeScalar128Bits();

            try testing.expectEqual(@as(u64, 0), challenge.limbs[0]);
            try testing.expectEqual(@as(u64, 0), challenge.limbs[1]);
            try testing.expectEqual(try std.fmt.parseInt(u64, fields[4], 16), challenge.limbs[2]);
            try testing.expectEqual(try std.fmt.parseInt(u64, fields[5], 16), challenge.limbs[3]);
        }
    }
}

test "blake2b transcript: basic initialization" {
    const Transcript = Blake2bTranscript(BN254Scalar);
    const t = Transcript.init("test_label");

    var all_zero = true;
    for (t.state) |b| {
        if (b != 0) {
            all_zero = false;
            break;
        }
    }
    try testing.expect(!all_zero);
    try testing.expectEqual(@as(u32, 0), t.n_rounds);
}

test "blake2b transcript: deterministic challenges" {
    const Transcript = Blake2bTranscript(BN254Scalar);

    var t1 = Transcript.init("test");
    var t2 = Transcript.init("test");

    t1.appendLabel("hello");
    t2.appendLabel("hello");

    const c1 = t1.challengeScalar();
    const c2 = t2.challengeScalar();

    try testing.expect(c1.eql(c2));
}

test "blake2b transcript: different inputs produce different challenges" {
    const Transcript = Blake2bTranscript(BN254Scalar);

    var t1 = Transcript.init("test");
    var t2 = Transcript.init("test");

    t1.appendLabel("hello");
    t2.appendLabel("world");

    const c1 = t1.challengeScalar();
    const c2 = t2.challengeScalar();

    try testing.expect(!c1.eql(c2));
}

test "blake2b transcript: round counter increments" {
    const Transcript = Blake2bTranscript(BN254Scalar);

    var t = Transcript.init("test");
    try testing.expectEqual(@as(u32, 0), t.n_rounds);

    t.appendLabel("msg1");
    try testing.expectEqual(@as(u32, 1), t.n_rounds);

    t.appendLabel("msg2");
    try testing.expectEqual(@as(u32, 2), t.n_rounds);

    _ = t.challengeScalar();
    try testing.expectEqual(@as(u32, 3), t.n_rounds);
}

test "blake2b transcript: scalar append and challenge" {
    const Transcript = Blake2bTranscript(BN254Scalar);

    var t = Transcript.init("scalar_test");

    const scalar = BN254Scalar.fromU64(12345);
    t.appendScalar("test_scalar", scalar);

    const challenge = t.challengeScalar();
    try testing.expect(!challenge.eql(BN254Scalar.zero()));
}

test "blake2b transcript: u64 append" {
    const Transcript = Blake2bTranscript(BN254Scalar);

    var t = Transcript.init("u64_test");
    t.appendU64("test_u64", 0x123456789ABCDEF0);

    const challenge = t.challengeScalar();
    try testing.expect(!challenge.eql(BN254Scalar.zero()));
}

test "blake2b transcript: challenge scalar is 128 bits" {
    const Transcript = Blake2bTranscript(BN254Scalar);

    var t = Transcript.init("128bit_test");
    t.appendLabel("data");

    const challenge = t.challengeScalar();
    try testing.expect(!challenge.eql(BN254Scalar.zero()));
}

test "blake2b transcript: challenge vector" {
    const Transcript = Blake2bTranscript(BN254Scalar);
    const allocator = testing.allocator;

    var t = Transcript.init("test");
    t.appendLabel("data");

    const challenges = try t.challengeVector(allocator, 5);
    defer allocator.free(challenges);

    try testing.expectEqual(@as(usize, 5), challenges.len);

    for (0..challenges.len) |i| {
        for (i + 1..challenges.len) |j| {
            try testing.expect(!challenges[i].eql(challenges[j]));
        }
    }
}

test "blake2b transcript: challenge scalar powers" {
    const Transcript = Blake2bTranscript(BN254Scalar);
    const allocator = testing.allocator;

    var t = Transcript.init("test");
    t.appendLabel("data");

    const powers = try t.challengeScalarPowers(allocator, 4);
    defer allocator.free(powers);

    try testing.expect(powers[0].eql(BN254Scalar.one()));

    const q = powers[1];
    try testing.expect(powers[2].eql(q.mul(q)));
    try testing.expect(powers[3].eql(q.mul(q).mul(q)));
}

// =============================================================================
// Jolt Compatibility Test Vectors
// =============================================================================
// NOTE: The old test vectors (test_vector 1-7) tested the OLD unlabeled API.
// With the labeled API, the raw internal methods (rawAppendLabel etc.) match
// the old appendMessage etc. so the core hashing is unchanged — only the
// public call sequences differ.

test "jolt compatibility: rawAppendLabel matches old appendMessage" {
    const Transcript = Blake2bTranscript(BN254Scalar);

    // rawAppendLabel is the same as old appendMessage
    var t = Transcript.init("zolt_test");
    t.rawAppendLabel("hello");

    const expected_state = [_]u8{
        0x3a, 0x25, 0x86, 0x87, 0x9f, 0x2c, 0x7b, 0x91,
        0x1e, 0x0f, 0x00, 0x6e, 0x6c, 0x3c, 0xc9, 0xee,
        0x03, 0x50, 0x7c, 0x28, 0xb8, 0x18, 0x13, 0x82,
        0x83, 0x50, 0x8b, 0x19, 0x2d, 0x43, 0x16, 0x8c,
    };

    try testing.expectEqualSlices(u8, &expected_state, &t.state);
    try testing.expectEqual(@as(u32, 1), t.n_rounds);
}

test "jolt compatibility: initial state" {
    const Transcript = Blake2bTranscript(BN254Scalar);

    const t = Transcript.init("init_test");

    const expected_state = [_]u8{
        0x40, 0xf6, 0xe6, 0x1c, 0xb8, 0x28, 0xbc, 0xfb,
        0xd5, 0x14, 0x3a, 0x3d, 0xf3, 0x02, 0x02, 0x1d,
        0x6a, 0xb7, 0xf8, 0x5d, 0xad, 0x9d, 0x08, 0x3b,
        0x27, 0x45, 0x37, 0xba, 0x1e, 0x73, 0x91, 0xcd,
    };

    try testing.expectEqualSlices(u8, &expected_state, &t.state);
    try testing.expectEqual(@as(u32, 0), t.n_rounds);
}

test "jolt compatibility: rawAppendBytes" {
    const Transcript = Blake2bTranscript(BN254Scalar);

    var t = Transcript.init("");

    const expected_init_state = [_]u8{
        0x89, 0xeb, 0x0d, 0x6a, 0x8a, 0x69, 0x1d, 0xae,
        0x2c, 0xd1, 0x5e, 0xd0, 0x36, 0x99, 0x31, 0xce,
        0x0a, 0x94, 0x9e, 0xca, 0xfa, 0x5c, 0x3f, 0x93,
        0xf8, 0x12, 0x18, 0x33, 0x64, 0x6e, 0x15, 0xc3,
    };

    try testing.expectEqualSlices(u8, &expected_init_state, &t.state);

    t.rawAppendBytes(&[_]u8{ 0x01, 0x02, 0x03 });

    const expected_after_bytes = [_]u8{
        0xd5, 0x23, 0x1a, 0x0c, 0x53, 0xbf, 0x96, 0xca,
        0x56, 0x3e, 0x5d, 0x13, 0xae, 0x11, 0x09, 0x01,
        0xfb, 0xe8, 0x06, 0xe2, 0x2f, 0x0e, 0xbb, 0x52,
        0xc7, 0x0f, 0xed, 0xf7, 0x8f, 0xb2, 0x6f, 0x82,
    };

    try testing.expectEqualSlices(u8, &expected_after_bytes, &t.state);
    try testing.expectEqual(@as(u32, 1), t.n_rounds);
}

test "jolt compatibility: challenge u128" {
    const Transcript = Blake2bTranscript(BN254Scalar);

    var t = Transcript.init("u128_test");
    t.rawAppendLabel("data");

    const challenge_u128 = t.challengeU128();
    const expected: u128 = 112132316132180403369405744574678933239;

    try testing.expectEqual(expected, challenge_u128);
}

test "jolt compatibility: rawAppendU64 and rawAppendLabel" {
    const Transcript = Blake2bTranscript(BN254Scalar);

    var t = Transcript.init("zolt_test_2");
    t.rawAppendLabel("first");
    t.rawAppendLabel("second");
    t.rawAppendU64(12345);

    const expected_state = [_]u8{
        0x71, 0xbe, 0x51, 0xcf, 0x72, 0x3b, 0xed, 0x98,
        0xc6, 0x6e, 0x2f, 0x6f, 0x85, 0xe4, 0x05, 0xf8,
        0x65, 0x69, 0x4e, 0xa8, 0x17, 0x5a, 0x74, 0x97,
        0xb4, 0x7b, 0xbe, 0x0e, 0x99, 0x50, 0x94, 0xc5,
    };

    try testing.expectEqualSlices(u8, &expected_state, &t.state);
    try testing.expectEqual(@as(u32, 3), t.n_rounds);
}

test "jolt compatibility: rawAppendScalar" {
    const Transcript = Blake2bTranscript(BN254Scalar);

    var t = Transcript.init("scalar_test");
    const scalar = BN254Scalar.fromU64(42);
    t.rawAppendScalar(scalar);

    const expected_state = [_]u8{
        0xe6, 0x5e, 0x77, 0x8c, 0x52, 0x61, 0xda, 0x8b,
        0x62, 0xcb, 0x73, 0x8c, 0xbc, 0x54, 0x03, 0x3e,
        0xc4, 0x33, 0xd0, 0xcb, 0xe7, 0x3e, 0xf3, 0x7d,
        0xad, 0x13, 0x37, 0xc0, 0x60, 0xbe, 0xad, 0x31,
    };

    try testing.expectEqualSlices(u8, &expected_state, &t.state);
    try testing.expectEqual(@as(u32, 1), t.n_rounds);
}

test "jolt compatibility: challenge limbs verification" {
    const Transcript = Blake2bTranscript(BN254Scalar);

    var t = Transcript.init("zolt_compat_test");
    t.rawAppendLabel("test_data");

    const challenge = t.challengeScalar128Bits();

    const expected_low: u64 = 0x5207e9d28bcca994;
    const expected_high: u64 = 0x0737345c88127af8;

    try testing.expectEqual(@as(u64, 0), challenge.limbs[0]);
    try testing.expectEqual(@as(u64, 0), challenge.limbs[1]);
    try testing.expectEqual(expected_low, challenge.limbs[2]);
    try testing.expectEqual(expected_high, challenge.limbs[3]);
}
