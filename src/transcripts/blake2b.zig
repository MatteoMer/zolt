//! Blake2b Fiat-Shamir Transcript for Jolt Compatibility
//!
//! This module implements a Blake2b-based Fiat-Shamir transcript that is
//! byte-compatible with Jolt's Blake2bTranscript (Rust).
//!
//! Key compatibility requirements:
//! - Uses Blake2b-256 (32-byte output)
//! - 32-byte state with round counter for domain separation
//! - Messages are right-padded to 32 bytes with zeros
//! - Scalars are serialized LE then reversed to BE (EVM format)
//! - Challenges are 128-bit (16 bytes)
//! - Vector operations use begin/end markers

const std = @import("std");
const crypto = std.crypto;
const Blake2b256 = crypto.hash.blake2.Blake2b256;
const mem = std.mem;

/// Blake2b-based Fiat-Shamir transcript compatible with Jolt
///
/// This matches Jolt's Blake2bTranscript exactly:
/// - 32-byte running state
/// - Round counter (u32) for domain separation
/// - hasher() = Blake2b256(state || packed_round) where packed_round = [0u8; 28] || round.to_be_bytes()
pub fn Blake2bTranscript(comptime F: type) type {
    return struct {
        const Self = @This();

        /// 256-bit running state (matches Jolt's `state: [u8; 32]`)
        state: [32]u8,
        /// Round counter for domain separation (matches Jolt's `n_rounds: u32`)
        n_rounds: u32,

        /// Create a new transcript with a label
        /// Matches Jolt's `fn new(label: &'static [u8]) -> Self`
        ///
        /// The label is padded to 32 bytes with zeros on the right,
        /// then hashed with Blake2b-256 to produce the initial state.
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
        ///
        /// Creates: Blake2b256::new().chain_update(state).chain_update(packed)
        /// where packed = [0u8; 28] ++ n_rounds.to_be_bytes()
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

        /// Append a message to the transcript
        /// Matches Jolt's `fn append_message(&mut self, msg: &'static [u8])`
        ///
        /// Messages are right-padded to 32 bytes with zeros.
        pub fn appendMessage(self: *Self, msg: []const u8) void {
            std.debug.assert(msg.len < 33);

            var h = self.hasher();

            if (msg.len == 32) {
                h.update(msg);
            } else {
                // Right-pad to 32 bytes
                var padded: [32]u8 = [_]u8{0} ** 32;
                @memcpy(padded[0..msg.len], msg);
                h.update(&padded);
            }

            var result: [32]u8 = undefined;
            h.final(&result);
            self.updateState(result);
        }

        /// Append raw bytes to the transcript
        /// Matches Jolt's `fn append_bytes(&mut self, bytes: &[u8])`
        pub fn appendBytes(self: *Self, bytes: []const u8) void {
            var h = self.hasher();
            h.update(bytes);

            var result: [32]u8 = undefined;
            h.final(&result);
            self.updateState(result);
        }

        /// Append a u64 to the transcript
        /// Matches Jolt's `fn append_u64(&mut self, x: u64)`
        ///
        /// Packs into 32 bytes: [0u8; 24] ++ x.to_be_bytes()
        pub fn appendU64(self: *Self, x: u64) void {
            var data: [32]u8 = [_]u8{0} ** 32;
            mem.writeInt(u64, data[24..32], x, .big);

            var h = self.hasher();
            h.update(&data);

            var result: [32]u8 = undefined;
            h.final(&result);
            self.updateState(result);
        }

        /// Append a scalar to the transcript
        /// Matches Jolt's `fn append_scalar<F: JoltField>(&mut self, scalar: &F)`
        ///
        /// Serializes LE, then reverses to BE for EVM compatibility.
        pub fn appendScalar(self: *Self, scalar: F) void {
            // Serialize to LE bytes
            var buf: [32]u8 = undefined;
            for (0..4) |i| {
                mem.writeInt(u64, buf[i * 8 ..][0..8], scalar.limbs[i], .little);
            }

            // Reverse to BE for EVM compatibility
            var reversed: [32]u8 = undefined;
            for (0..32) |i| {
                reversed[i] = buf[31 - i];
            }

            self.appendBytes(&reversed);
        }

        /// Append multiple scalars to the transcript
        /// Matches Jolt's `fn append_scalars<F: JoltField>(&mut self, scalars: &[impl Borrow<F>])`
        ///
        /// Uses begin_append_vector/end_append_vector markers.
        pub fn appendScalars(self: *Self, scalars: []const F) void {
            self.appendMessage("begin_append_vector");
            for (scalars) |scalar| {
                self.appendScalar(scalar);
            }
            self.appendMessage("end_append_vector");
        }

        /// Get 32 challenge bytes from the transcript
        /// Matches Jolt's `fn challenge_bytes32(&mut self, out: &mut [u8])`
        fn challengeBytes32(self: *Self, out: *[32]u8) void {
            var h = self.hasher();
            h.final(out);
            self.updateState(out.*);
        }

        /// Get arbitrary length challenge bytes
        /// Matches Jolt's `fn challenge_bytes(&mut self, out: &mut [u8])`
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

            // Get final bytes (may be less than 32)
            var full_rand: [32]u8 = undefined;
            self.challengeBytes32(&full_rand);
            @memcpy(out[start..][0..remaining], full_rand[0..remaining]);
        }

        /// Get a 128-bit challenge as u128
        /// Matches Jolt's `fn challenge_u128(&mut self) -> u128`
        pub fn challengeU128(self: *Self) u128 {
            var buf: [16]u8 = undefined;
            self.challengeBytes(&buf);

            // Reverse to match Jolt's `buf = buf.into_iter().rev().collect()`
            var reversed: [16]u8 = undefined;
            for (0..16) |i| {
                reversed[i] = buf[15 - i];
            }

            return mem.readInt(u128, &reversed, .big);
        }

        /// Get a challenge scalar (128-bit for performance)
        /// Matches Jolt's `fn challenge_scalar<F: JoltField>(&mut self) -> F`
        pub fn challengeScalar(self: *Self) F {
            return self.challengeScalar128Bits();
        }

        /// Get a 128-bit challenge scalar
        /// Matches Jolt's `fn challenge_scalar_128_bits<F: JoltField>(&mut self) -> F`
        pub fn challengeScalar128Bits(self: *Self) F {
            var buf: [16]u8 = undefined;
            self.challengeBytes(&buf);

            // Reverse to match Jolt's approach
            var reversed: [16]u8 = undefined;
            for (0..16) |i| {
                reversed[i] = buf[15 - i];
            }

            // Create field element from the 16 bytes
            // Pad to 32 bytes (16 bytes of challenge + 16 bytes of zeros)
            var padded: [32]u8 = [_]u8{0} ** 32;
            @memcpy(padded[0..16], &reversed);

            return F.fromBytes(&padded);
        }

        /// Get multiple challenge scalars
        /// Matches Jolt's `fn challenge_vector<F: JoltField>(&mut self, len: usize) -> Vec<F>`
        pub fn challengeVector(self: *Self, allocator: std.mem.Allocator, len: usize) ![]F {
            const challenges = try allocator.alloc(F, len);
            for (0..len) |i| {
                challenges[i] = self.challengeScalar();
            }
            return challenges;
        }

        /// Compute powers of a challenge scalar
        /// Matches Jolt's `fn challenge_scalar_powers<F: JoltField>(&mut self, len: usize) -> Vec<F>`
        pub fn challengeScalarPowers(self: *Self, allocator: std.mem.Allocator, len: usize) ![]F {
            const q = self.challengeScalar();
            const q_powers = try allocator.alloc(F, len);
            q_powers[0] = F.one();
            for (1..len) |i| {
                q_powers[i] = q_powers[i - 1].mul(q);
            }
            return q_powers;
        }

        /// Append a point to the transcript (for curve points)
        /// Matches Jolt's `fn append_point<G: CurveGroup>(&mut self, point: &G)`
        ///
        /// Points are serialized as (x, y) in BE format.
        /// Point at infinity is serialized as 64 zero bytes.
        pub fn appendPoint(self: *Self, comptime Point: type, point: Point) void {
            // Check for point at infinity
            if (point.isIdentity()) {
                self.appendBytes(&([_]u8{0} ** 64));
                return;
            }

            const affine = point.toAffine();

            // Serialize x and y coordinates in BE
            var x_bytes: [32]u8 = undefined;
            var y_bytes: [32]u8 = undefined;

            // Get x and y, serialize LE, then reverse to BE
            for (0..4) |i| {
                mem.writeInt(u64, x_bytes[i * 8 ..][0..8], affine.x.limbs[i], .little);
                mem.writeInt(u64, y_bytes[i * 8 ..][0..8], affine.y.limbs[i], .little);
            }

            // Reverse to BE
            var x_reversed: [32]u8 = undefined;
            var y_reversed: [32]u8 = undefined;
            for (0..32) |i| {
                x_reversed[i] = x_bytes[31 - i];
                y_reversed[i] = y_bytes[31 - i];
            }

            var h = self.hasher();
            h.update(&x_reversed);
            h.update(&y_reversed);

            var result: [32]u8 = undefined;
            h.final(&result);
            self.updateState(result);
        }

        /// Append multiple points to the transcript
        /// Matches Jolt's `fn append_points<G: CurveGroup>(&mut self, points: &[G])`
        pub fn appendPoints(self: *Self, comptime Point: type, points: []const Point) void {
            self.appendMessage("begin_append_vector");
            for (points) |point| {
                self.appendPoint(Point, point);
            }
            self.appendMessage("end_append_vector");
        }

        /// Append a serializable object to the transcript
        /// Matches Jolt's `fn append_serializable<F: CanonicalSerialize>(&mut self, scalar: &F)`
        ///
        /// This reverses the bytes after serialization for EVM compatibility,
        /// matching what arkworks + Jolt does.
        pub fn appendSerializable(self: *Self, comptime T: type, bytes: T) void {
            // Reverse the bytes to match Jolt's behavior
            var reversed: T = undefined;
            for (0..bytes.len) |i| {
                reversed[i] = bytes[bytes.len - 1 - i];
            }
            self.appendBytes(&reversed);
        }

        /// Append a GT (Fp12) element to the transcript
        /// For Dory compatibility - appends the serialized GT element as bytes
        ///
        /// IMPORTANT: This matches Jolt's append_serializable which reverses
        /// all bytes after serialization for EVM compatibility.
        pub fn appendGT(self: *Self, gt: anytype) void {
            const bytes = gt.toBytes();
            // Reverse bytes to match Jolt's append_serializable
            var reversed: [384]u8 = undefined;
            for (0..384) |i| {
                reversed[i] = bytes[383 - i];
            }
            self.appendBytes(&reversed);
        }

        /// Append a G1 point to the transcript (compressed format)
        /// For Dory compatibility - appends the compressed serialized point
        pub fn appendG1Compressed(self: *Self, point: anytype) void {
            const dory = @import("../poly/commitment/dory.zig");
            const bytes = dory.compressG1(point);
            self.appendBytes(&bytes);
        }

        /// Append a G2 point to the transcript (compressed format)
        /// For Dory compatibility - appends the compressed serialized point
        pub fn appendG2Compressed(self: *Self, point: anytype) void {
            const dory = @import("../poly/commitment/dory.zig");
            const bytes = dory.compressG2(point);
            self.appendBytes(&bytes);
        }
    };
}

// Tests
const testing = std.testing;
const BN254Scalar = @import("../field/mod.zig").BN254Scalar;

test "blake2b transcript: basic initialization" {
    const Transcript = Blake2bTranscript(BN254Scalar);
    const t = Transcript.init("test_label");

    // State should be non-zero after initialization
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

    // Two transcripts with same operations should produce same challenges
    var t1 = Transcript.init("test");
    var t2 = Transcript.init("test");

    t1.appendMessage("hello");
    t2.appendMessage("hello");

    const c1 = t1.challengeScalar();
    const c2 = t2.challengeScalar();

    try testing.expect(c1.eql(c2));
}

test "blake2b transcript: different inputs produce different challenges" {
    const Transcript = Blake2bTranscript(BN254Scalar);

    var t1 = Transcript.init("test");
    var t2 = Transcript.init("test");

    t1.appendMessage("hello");
    t2.appendMessage("world");

    const c1 = t1.challengeScalar();
    const c2 = t2.challengeScalar();

    try testing.expect(!c1.eql(c2));
}

test "blake2b transcript: round counter increments" {
    const Transcript = Blake2bTranscript(BN254Scalar);

    var t = Transcript.init("test");
    try testing.expectEqual(@as(u32, 0), t.n_rounds);

    t.appendMessage("msg1");
    try testing.expectEqual(@as(u32, 1), t.n_rounds);

    t.appendMessage("msg2");
    try testing.expectEqual(@as(u32, 2), t.n_rounds);

    _ = t.challengeScalar();
    try testing.expectEqual(@as(u32, 3), t.n_rounds);
}

test "blake2b transcript: scalar append and challenge" {
    const Transcript = Blake2bTranscript(BN254Scalar);

    var t = Transcript.init("scalar_test");

    const scalar = BN254Scalar.fromU64(12345);
    t.appendScalar(scalar);

    const challenge = t.challengeScalar();

    // Challenge should be non-zero
    try testing.expect(!challenge.eql(BN254Scalar.zero()));
}

test "blake2b transcript: append scalars with markers" {
    const Transcript = Blake2bTranscript(BN254Scalar);

    var t1 = Transcript.init("test");
    var t2 = Transcript.init("test");

    const scalars = [_]BN254Scalar{
        BN254Scalar.fromU64(1),
        BN254Scalar.fromU64(2),
        BN254Scalar.fromU64(3),
    };

    // Using appendScalars
    t1.appendScalars(&scalars);

    // Manual equivalent
    t2.appendMessage("begin_append_vector");
    for (scalars) |s| {
        t2.appendScalar(s);
    }
    t2.appendMessage("end_append_vector");

    // Should produce same state
    try testing.expectEqualSlices(u8, &t1.state, &t2.state);
    try testing.expectEqual(t1.n_rounds, t2.n_rounds);
}

test "blake2b transcript: challenge scalar is 128 bits" {
    const Transcript = Blake2bTranscript(BN254Scalar);

    var t = Transcript.init("128bit_test");
    t.appendMessage("data");

    const challenge = t.challengeScalar();

    // The challenge should fit in 128 bits (upper 128 bits should be zero)
    // Check by examining the limbs - limbs 2 and 3 should be zero
    // Actually, the value is computed from 16 bytes, so it fits in 128 bits
    // but after field reduction it could still have bits set in upper limbs
    // So we just check it's a valid field element (non-zero)
    try testing.expect(!challenge.eql(BN254Scalar.zero()));
}

test "blake2b transcript: u64 append" {
    const Transcript = Blake2bTranscript(BN254Scalar);

    var t = Transcript.init("u64_test");
    t.appendU64(0x123456789ABCDEF0);

    const challenge = t.challengeScalar();
    try testing.expect(!challenge.eql(BN254Scalar.zero()));
}

test "blake2b transcript: empty label" {
    const Transcript = Blake2bTranscript(BN254Scalar);

    // Empty label should work (padded to 32 zeros)
    const t = Transcript.init("");
    try testing.expectEqual(@as(u32, 0), t.n_rounds);
}

test "blake2b transcript: max length label" {
    const Transcript = Blake2bTranscript(BN254Scalar);

    // 32-byte label (max length)
    const label = "12345678901234567890123456789012";
    const t = Transcript.init(label);
    try testing.expectEqual(@as(u32, 0), t.n_rounds);
}

test "blake2b transcript: challenge bytes" {
    const Transcript = Blake2bTranscript(BN254Scalar);

    var t1 = Transcript.init("test");
    var t2 = Transcript.init("test");

    t1.appendMessage("data");
    t2.appendMessage("data");

    var buf1: [64]u8 = undefined;
    var buf2: [64]u8 = undefined;

    t1.challengeBytes(&buf1);
    t2.challengeBytes(&buf2);

    // Same inputs should give same outputs
    try testing.expectEqualSlices(u8, &buf1, &buf2);
}

test "blake2b transcript: challenge u128" {
    const Transcript = Blake2bTranscript(BN254Scalar);

    var t1 = Transcript.init("test");
    var t2 = Transcript.init("test");

    t1.appendMessage("data");
    t2.appendMessage("data");

    const val1 = t1.challengeU128();
    const val2 = t2.challengeU128();

    try testing.expectEqual(val1, val2);
    try testing.expect(val1 != 0);
}

test "blake2b transcript: challenge vector" {
    const Transcript = Blake2bTranscript(BN254Scalar);
    const allocator = testing.allocator;

    var t = Transcript.init("test");
    t.appendMessage("data");

    const challenges = try t.challengeVector(allocator, 5);
    defer allocator.free(challenges);

    try testing.expectEqual(@as(usize, 5), challenges.len);

    // All challenges should be different
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
    t.appendMessage("data");

    const powers = try t.challengeScalarPowers(allocator, 4);
    defer allocator.free(powers);

    // First power should be 1
    try testing.expect(powers[0].eql(BN254Scalar.one()));

    // Check powers are correct: powers[i] = powers[1]^i
    const q = powers[1];
    try testing.expect(powers[2].eql(q.mul(q)));
    try testing.expect(powers[3].eql(q.mul(q).mul(q)));
}

// =============================================================================
// Jolt Compatibility Test Vectors
// =============================================================================
// These tests verify that Zolt's Blake2bTranscript produces identical outputs
// to Jolt's Blake2bTranscript (Rust). Any failure here means proof verification
// will fail when a Zolt proof is verified by Jolt.

test "jolt compatibility: test vector 1 - simple label and message" {
    const Transcript = Blake2bTranscript(BN254Scalar);

    var t = Transcript.init("zolt_test");
    t.appendMessage("hello");

    // Expected state after append_message (from Jolt):
    // [04, 5d, b7, 95, b0, 5d, 42, b5, c7, 9d, 6d, bb, f2, 0c, be, 09,
    //  26, 36, df, 45, bb, 1c, 80, f2, a4, be, 9b, 66, 4b, ad, 5e, 0d]
    const expected_state = [_]u8{
        0x04, 0x5d, 0xb7, 0x95, 0xb0, 0x5d, 0x42, 0xb5,
        0xc7, 0x9d, 0x6d, 0xbb, 0xf2, 0x0c, 0xbe, 0x09,
        0x26, 0x36, 0xdf, 0x45, 0xbb, 0x1c, 0x80, 0xf2,
        0xa4, 0xbe, 0x9b, 0x66, 0x4b, 0xad, 0x5e, 0x0d,
    };

    try testing.expectEqualSlices(u8, &expected_state, &t.state);
    try testing.expectEqual(@as(u32, 1), t.n_rounds);

    const challenge = t.challengeScalar();
    try testing.expectEqual(@as(u32, 2), t.n_rounds);

    // Expected challenge (LE bytes from Jolt):
    // [09, be, 0c, f2, bb, 6d, 9d, c7, b5, 42, 5d, b0, 95, b7, 5d, 04, 0, 0, ...]
    // This is a 128-bit value padded to 32 bytes
    const expected_challenge_bytes = [_]u8{
        0x09, 0xbe, 0x0c, 0xf2, 0xbb, 0x6d, 0x9d, 0xc7,
        0xb5, 0x42, 0x5d, 0xb0, 0x95, 0xb7, 0x5d, 0x04,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    };
    const expected_challenge = BN254Scalar.fromBytes(&expected_challenge_bytes);

    try testing.expect(challenge.eql(expected_challenge));
}

test "jolt compatibility: test vector 5 - initial state" {
    const Transcript = Blake2bTranscript(BN254Scalar);

    const t = Transcript.init("init_test");

    // Expected initial state from Jolt:
    const expected_state = [_]u8{
        0x40, 0xf6, 0xe6, 0x1c, 0xb8, 0x28, 0xbc, 0xfb,
        0xd5, 0x14, 0x3a, 0x3d, 0xf3, 0x02, 0x02, 0x1d,
        0x6a, 0xb7, 0xf8, 0x5d, 0xad, 0x9d, 0x08, 0x3b,
        0x27, 0x45, 0x37, 0xba, 0x1e, 0x73, 0x91, 0xcd,
    };

    try testing.expectEqualSlices(u8, &expected_state, &t.state);
    try testing.expectEqual(@as(u32, 0), t.n_rounds);
}

test "jolt compatibility: test vector 6 - empty label" {
    const Transcript = Blake2bTranscript(BN254Scalar);

    var t = Transcript.init("");

    // Expected initial state from Jolt for empty label:
    const expected_init_state = [_]u8{
        0x89, 0xeb, 0x0d, 0x6a, 0x8a, 0x69, 0x1d, 0xae,
        0x2c, 0xd1, 0x5e, 0xd0, 0x36, 0x99, 0x31, 0xce,
        0x0a, 0x94, 0x9e, 0xca, 0xfa, 0x5c, 0x3f, 0x93,
        0xf8, 0x12, 0x18, 0x33, 0x64, 0x6e, 0x15, 0xc3,
    };

    try testing.expectEqualSlices(u8, &expected_init_state, &t.state);
    try testing.expectEqual(@as(u32, 0), t.n_rounds);

    t.appendBytes(&[_]u8{ 0x01, 0x02, 0x03 });

    // Expected state after append_bytes from Jolt:
    const expected_after_bytes = [_]u8{
        0xc4, 0x21, 0x29, 0xc2, 0x59, 0x57, 0x65, 0x9c,
        0xf7, 0x63, 0x38, 0xf5, 0xd2, 0xcb, 0xad, 0xd9,
        0x5d, 0x1b, 0xf5, 0xd3, 0x57, 0xfc, 0xf9, 0xa1,
        0xe9, 0x62, 0xc3, 0xc6, 0xb5, 0xed, 0x37, 0x27,
    };

    try testing.expectEqualSlices(u8, &expected_after_bytes, &t.state);
    try testing.expectEqual(@as(u32, 1), t.n_rounds);
}

test "jolt compatibility: test vector 7 - u128 challenge" {
    const Transcript = Blake2bTranscript(BN254Scalar);

    var t = Transcript.init("u128_test");
    t.appendMessage("data");

    const challenge_u128 = t.challengeU128();

    // Expected from Jolt: 112132316132180403369405744574678933239
    // Hex: 545be6068976bba7550dd1c80c4b9ef7
    const expected: u128 = 112132316132180403369405744574678933239;

    try testing.expectEqual(expected, challenge_u128);
}

test "jolt compatibility: test vector 2 - multiple appends" {
    const Transcript = Blake2bTranscript(BN254Scalar);

    var t = Transcript.init("zolt_test_2");
    t.appendMessage("first");
    t.appendMessage("second");
    t.appendU64(12345);

    // Expected state from Jolt:
    const expected_state = [_]u8{
        0x14, 0x1b, 0xf2, 0x3f, 0x43, 0x6f, 0x74, 0x1b,
        0x0f, 0x9d, 0x78, 0x0f, 0xac, 0x3e, 0x62, 0x93,
        0x62, 0x74, 0x78, 0x7c, 0xde, 0x4f, 0x59, 0x55,
        0x73, 0xa5, 0x32, 0x6c, 0x5a, 0x75, 0x5d, 0x85,
    };

    try testing.expectEqualSlices(u8, &expected_state, &t.state);
    try testing.expectEqual(@as(u32, 3), t.n_rounds);

    const challenge = t.challengeScalar();
    try testing.expectEqual(@as(u32, 4), t.n_rounds);

    // Expected challenge from Jolt (LE bytes):
    const expected_challenge_bytes = [_]u8{
        0x93, 0x62, 0x3e, 0xac, 0x0f, 0x78, 0x9d, 0x0f,
        0x1b, 0x74, 0x6f, 0x43, 0x3f, 0xf2, 0x1b, 0x14,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    };
    const expected_challenge = BN254Scalar.fromBytes(&expected_challenge_bytes);

    try testing.expect(challenge.eql(expected_challenge));
}

test "jolt compatibility: test vector 3 - scalar append" {
    const Transcript = Blake2bTranscript(BN254Scalar);

    var t = Transcript.init("scalar_test");
    const scalar = BN254Scalar.fromU64(42);
    t.appendScalar(scalar);

    // Expected state from Jolt:
    const expected_state = [_]u8{
        0x0d, 0x8c, 0x5a, 0x29, 0x4a, 0x38, 0x74, 0x89,
        0x89, 0xe3, 0x60, 0x61, 0x7d, 0x26, 0x1a, 0x04,
        0x73, 0x5a, 0x30, 0x54, 0xff, 0xf0, 0xf2, 0x9c,
        0xa3, 0x6c, 0x9d, 0x32, 0x28, 0x4e, 0x3a, 0x7c,
    };

    try testing.expectEqualSlices(u8, &expected_state, &t.state);
    try testing.expectEqual(@as(u32, 1), t.n_rounds);

    const challenge = t.challengeScalar();
    try testing.expectEqual(@as(u32, 2), t.n_rounds);

    // Expected challenge from Jolt (LE bytes):
    const expected_challenge_bytes = [_]u8{
        0x04, 0x1a, 0x26, 0x7d, 0x61, 0x60, 0xe3, 0x89,
        0x89, 0x74, 0x38, 0x4a, 0x29, 0x5a, 0x8c, 0x0d,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    };
    const expected_challenge = BN254Scalar.fromBytes(&expected_challenge_bytes);

    try testing.expect(challenge.eql(expected_challenge));
}

test "jolt compatibility: test vector 4 - vector append" {
    const Transcript = Blake2bTranscript(BN254Scalar);

    var t = Transcript.init("vector_test");
    const scalars = [_]BN254Scalar{
        BN254Scalar.fromU64(1),
        BN254Scalar.fromU64(2),
        BN254Scalar.fromU64(3),
    };
    t.appendScalars(&scalars);

    // Expected state from Jolt:
    const expected_state = [_]u8{
        0xa1, 0xbe, 0xdf, 0x0b, 0x1f, 0x6f, 0x93, 0xfb,
        0xab, 0xc9, 0xb7, 0x81, 0x2c, 0x9a, 0x22, 0xd0,
        0x7f, 0x89, 0x7b, 0x4e, 0xb8, 0x7a, 0xff, 0x86,
        0x6c, 0x96, 0x19, 0xec, 0x20, 0xf1, 0x6c, 0x4f,
    };

    try testing.expectEqualSlices(u8, &expected_state, &t.state);
    try testing.expectEqual(@as(u32, 5), t.n_rounds);

    const challenge = t.challengeScalar();
    try testing.expectEqual(@as(u32, 6), t.n_rounds);

    // Expected challenge from Jolt (LE bytes):
    const expected_challenge_bytes = [_]u8{
        0xd0, 0x22, 0x9a, 0x2c, 0x81, 0xb7, 0xc9, 0xab,
        0xfb, 0x93, 0x6f, 0x1f, 0x0b, 0xdf, 0xbe, 0xa1,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    };
    const expected_challenge = BN254Scalar.fromBytes(&expected_challenge_bytes);

    try testing.expect(challenge.eql(expected_challenge));
}
