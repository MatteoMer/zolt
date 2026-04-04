//! Fiat-Shamir transcript for Jolt
//!
//! This module implements the Fiat-Shamir transformation for making
//! interactive proofs non-interactive.
//!
//! Two transcript implementations are provided:
//! - `Transcript`: Keccak-based (original Zolt implementation)
//! - `Blake2bTranscript`: Blake2b-based (Jolt-compatible)
//!
//! For Jolt compatibility, use `Blake2bTranscript`.

const std = @import("std");
const Allocator = std.mem.Allocator;

// Re-export Blake2b transcript for Jolt compatibility
pub const blake2b = @import("blake2b.zig");
pub const Blake2bTranscript = blake2b.Blake2bTranscript;

/// Keccak256 state size in bytes (1600 bits = 200 bytes)
const KECCAK_STATE_SIZE = 200;
/// Rate for Keccak-256 (1088 bits = 136 bytes)
const KECCAK_RATE = 136;
/// Number of rounds in Keccak-f[1600]
const KECCAK_ROUNDS = 24;

/// Round constants for Keccak-f[1600]
const KECCAK_RC: [24]u64 = .{
    0x0000000000000001, 0x0000000000008082, 0x800000000000808a, 0x8000000080008000,
    0x000000000000808b, 0x0000000080000001, 0x8000000080008081, 0x8000000000008009,
    0x000000000000008a, 0x0000000000000088, 0x0000000080008009, 0x000000008000000a,
    0x000000008000808b, 0x800000000000008b, 0x8000000000008089, 0x8000000000008003,
    0x8000000000008002, 0x8000000000000080, 0x000000000000800a, 0x800000008000000a,
    0x8000000080008081, 0x8000000000008080, 0x0000000080000001, 0x8000000080008008,
};

/// Rotation offsets for the rho step
const KECCAK_ROTC: [24]u6 = .{
    1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14,
    27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44,
};

/// Position indices for the pi step
const KECCAK_PILN: [24]usize = .{
    10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4,
    15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1,
};

/// Transcript for Fiat-Shamir transformation
pub fn Transcript(comptime F: type) type {
    return struct {
        const Self = @This();

        /// Internal state buffer
        state: [KECCAK_STATE_SIZE]u8,
        /// Current position in buffer
        position: usize,
        /// Domain separator
        domain: []const u8,
        allocator: Allocator,

        /// Create a new transcript with a domain separator
        pub fn init(allocator: Allocator, domain: []const u8) !Self {
            var transcript = Self{
                .state = [_]u8{0} ** KECCAK_STATE_SIZE,
                .position = 0,
                .domain = domain,
                .allocator = allocator,
            };

            // Initialize with domain separator
            try transcript.appendBytes(domain);

            return transcript;
        }

        pub fn deinit(_: *Self) void {
            // Nothing to free currently
        }

        /// Append a message to the transcript
        pub fn appendMessage(self: *Self, label: []const u8, message: []const u8) !void {
            try self.appendBytes(label);
            try self.appendBytes(message);
        }

        /// Append raw bytes to the transcript
        pub fn appendBytes(self: *Self, data: []const u8) !void {
            for (data) |byte| {
                self.state[self.position] ^= byte;
                self.position += 1;

                if (self.position >= KECCAK_RATE) {
                    self.keccakF();
                    self.position = 0;
                }
            }
        }

        /// Append a field element to the transcript
        pub fn appendScalar(self: *Self, label: []const u8, scalar: F) !void {
            try self.appendBytes(label);

            // Serialize scalar to bytes (little-endian)
            var buf: [32]u8 = undefined;
            for (0..4) |i| {
                std.mem.writeInt(u64, buf[i * 8 ..][0..8], scalar.limbs[i], .little);
            }
            try self.appendBytes(&buf);
        }

        /// Append multiple scalars
        pub fn appendScalars(self: *Self, label: []const u8, scalars: []const F) !void {
            try self.appendBytes(label);
            for (scalars) |scalar| {
                try self.appendScalar("", scalar);
            }
        }

        /// Get a challenge field element
        pub fn challengeScalar(self: *Self, label: []const u8) !F {
            try self.appendBytes(label);

            // Squeeze output
            self.keccakF();

            // Convert first 32 bytes to field element
            var buf: [32]u8 = undefined;
            @memcpy(&buf, self.state[0..32]);

            return F.fromBytes(&buf);
        }

        /// Get multiple challenge scalars
        pub fn challengeScalars(self: *Self, label: []const u8, count: usize, allocator: Allocator) ![]F {
            const challenges = try allocator.alloc(F, count);

            try self.appendBytes(label);

            for (0..count) |i| {
                challenges[i] = try self.challengeScalar("");
            }

            return challenges;
        }

        /// Get challenge bytes
        pub fn challengeBytes(self: *Self, label: []const u8, output: []u8) !void {
            try self.appendBytes(label);

            var remaining = output.len;
            var offset: usize = 0;

            while (remaining > 0) {
                self.keccakF();

                const to_copy = @min(remaining, KECCAK_RATE);
                @memcpy(output[offset..][0..to_copy], self.state[0..to_copy]);

                offset += to_copy;
                remaining -= to_copy;
            }
        }

        /// Keccak-f[1600] permutation - the core of SHA-3
        fn keccakF(self: *Self) void {
            // Convert bytes to 64-bit words (little-endian)
            var st: [25]u64 = undefined;
            for (0..25) |i| {
                st[i] = std.mem.readInt(u64, self.state[i * 8 ..][0..8], .little);
            }

            // Apply 24 rounds
            for (0..KECCAK_ROUNDS) |round| {
                // Theta step
                var bc: [5]u64 = undefined;
                for (0..5) |i| {
                    bc[i] = st[i] ^ st[i + 5] ^ st[i + 10] ^ st[i + 15] ^ st[i + 20];
                }

                for (0..5) |i| {
                    const t = bc[(i + 4) % 5] ^ std.math.rotl(u64, bc[(i + 1) % 5], 1);
                    var j: usize = i;
                    while (j < 25) : (j += 5) {
                        st[j] ^= t;
                    }
                }

                // Rho and Pi steps
                var t = st[1];
                for (0..24) |i| {
                    const j = KECCAK_PILN[i];
                    const tmp = st[j];
                    st[j] = std.math.rotl(u64, t, KECCAK_ROTC[i]);
                    t = tmp;
                }

                // Chi step
                for (0..5) |j| {
                    const row = j * 5;
                    bc[0] = st[row];
                    bc[1] = st[row + 1];
                    bc[2] = st[row + 2];
                    bc[3] = st[row + 3];
                    bc[4] = st[row + 4];

                    for (0..5) |i| {
                        st[row + i] = bc[i] ^ (~bc[(i + 1) % 5] & bc[(i + 2) % 5]);
                    }
                }

                // Iota step
                st[0] ^= KECCAK_RC[round];
            }

            // Convert words back to bytes
            for (0..25) |i| {
                std.mem.writeInt(u64, self.state[i * 8 ..][0..8], st[i], .little);
            }
        }
    };
}

/// Poseidon transcript (alternative to Keccak)
///
/// Poseidon is a ZK-friendly hash function optimized for arithmetic circuits.
/// This implementation uses a simplified Poseidon-like permutation with:
/// - State width: 3 field elements
/// - Full rounds: 8 (with S-boxes on all elements)
/// - Partial rounds: 56 (with S-box on one element)
///
/// Note: For full security, use the proper round constants from the Poseidon paper.
/// This implementation provides the correct structure but uses simplified constants.
pub fn PoseidonTranscript(comptime F: type) type {
    return struct {
        const Self = @This();

        /// Number of full rounds (applied at start and end)
        const FULL_ROUNDS: usize = 8;
        /// Number of partial rounds (applied in the middle)
        const PARTIAL_ROUNDS: usize = 56;
        /// State width (t parameter)
        const STATE_WIDTH: usize = 3;

        state: [STATE_WIDTH]F,
        round_counter: usize,
        allocator: Allocator,

        pub fn init(allocator: Allocator, domain: []const u8) Self {
            // Initialize state with domain separator
            var initial_state: [STATE_WIDTH]F = .{ F.zero(), F.zero(), F.zero() };

            // Mix in domain separator bytes
            if (domain.len > 0) {
                var domain_hash: u64 = 0;
                for (domain) |byte| {
                    domain_hash = domain_hash *% 31 +% byte;
                }
                initial_state[0] = F.fromU64(domain_hash);
            }

            return .{
                .state = initial_state,
                .round_counter = 0,
                .allocator = allocator,
            };
        }

        pub fn deinit(_: *Self) void {}

        /// Absorb a field element
        pub fn absorb(self: *Self, element: F) void {
            self.state[0] = self.state[0].add(element);
            self.permute();
        }

        /// Squeeze a field element
        pub fn squeeze(self: *Self) F {
            self.permute();
            return self.state[0];
        }

        /// Poseidon permutation with full and partial rounds
        fn permute(self: *Self) void {
            // First half of full rounds
            for (0..FULL_ROUNDS / 2) |r| {
                self.fullRound(r);
            }

            // Partial rounds (S-box only on first element)
            for (0..PARTIAL_ROUNDS) |r| {
                self.partialRound(r + FULL_ROUNDS / 2);
            }

            // Second half of full rounds
            for (0..FULL_ROUNDS / 2) |r| {
                self.fullRound(r + FULL_ROUNDS / 2 + PARTIAL_ROUNDS);
            }

            self.round_counter += 1;
        }

        /// Full round: S-boxes on all state elements
        fn fullRound(self: *Self, round: usize) void {
            // Add round constants
            self.addRoundConstants(round);

            // Apply S-box (x^5) to all elements
            for (&self.state) |*s| {
                s.* = self.sbox(s.*);
            }

            // Apply MDS matrix
            self.mdsMatrix();
        }

        /// Partial round: S-box only on first element
        fn partialRound(self: *Self, round: usize) void {
            // Add round constants
            self.addRoundConstants(round);

            // Apply S-box only to first element
            self.state[0] = self.sbox(self.state[0]);

            // Apply MDS matrix
            self.mdsMatrix();
        }

        /// S-box: x^5 (efficient for BN254)
        fn sbox(self: *Self, x: F) F {
            _ = self;
            const x2 = x.mul(x);
            const x4 = x2.mul(x2);
            return x4.mul(x);
        }

        /// Add round constants (simplified - uses deterministic derivation)
        fn addRoundConstants(self: *Self, round: usize) void {
            // Derive round constants deterministically
            // In production, use precomputed constants from the Poseidon paper
            for (0..STATE_WIDTH) |i| {
                const rc = self.deriveRoundConstant(round, i);
                self.state[i] = self.state[i].add(rc);
            }
        }

        /// Derive a round constant deterministically
        fn deriveRoundConstant(_: *Self, round: usize, index: usize) F {
            // Simple deterministic derivation using golden ratio-based constants
            var seed: u64 = 0x9e3779b97f4a7c15;
            seed ^= @as(u64, @intCast(round)) *% 0xc4ceb9fe1a85ec53;
            seed ^= @as(u64, @intCast(index)) *% 0xff51afd7ed558ccd;
            seed ^= seed >> 33;
            seed *%= 0xff51afd7ed558ccd;
            seed ^= seed >> 33;
            return F.fromU64(seed);
        }

        /// MDS matrix multiplication (Cauchy matrix)
        /// For t=3, using a simple circulant structure
        fn mdsMatrix(self: *Self) void {
            const s0 = self.state[0];
            const s1 = self.state[1];
            const s2 = self.state[2];

            // MDS matrix for t=3:
            // [2, 1, 1]
            // [1, 2, 1]
            // [1, 1, 2]
            const two = F.fromU64(2);

            self.state[0] = two.mul(s0).add(s1).add(s2);
            self.state[1] = s0.add(two.mul(s1)).add(s2);
            self.state[2] = s0.add(s1).add(two.mul(s2));
        }
    };
}

test "transcript basic" {
    const F = @import("../field/mod.zig").BN254Scalar;
    const allocator = std.testing.allocator;

    var transcript = try Transcript(F).init(allocator, "test-domain");
    defer transcript.deinit();

    try transcript.appendMessage("label", "message");

    const challenge = try transcript.challengeScalar("challenge");
    _ = challenge;
}

test "keccak-f permutation" {
    const F = @import("../field/mod.zig").BN254Scalar;
    const allocator = std.testing.allocator;

    // Test that two transcripts with same inputs produce same outputs
    var t1 = try Transcript(F).init(allocator, "test");
    defer t1.deinit();
    var t2 = try Transcript(F).init(allocator, "test");
    defer t2.deinit();

    try t1.appendBytes("hello world");
    try t2.appendBytes("hello world");

    const c1 = try t1.challengeScalar("challenge");
    const c2 = try t2.challengeScalar("challenge");

    // Same inputs should produce same challenge
    try std.testing.expect(c1.eql(c2));
}

test "transcript different inputs produce different challenges" {
    const F = @import("../field/mod.zig").BN254Scalar;
    const allocator = std.testing.allocator;

    var t1 = try Transcript(F).init(allocator, "test");
    defer t1.deinit();
    var t2 = try Transcript(F).init(allocator, "test");
    defer t2.deinit();

    try t1.appendBytes("hello");
    try t2.appendBytes("world");

    const c1 = try t1.challengeScalar("challenge");
    const c2 = try t2.challengeScalar("challenge");

    // Different inputs should produce different challenges
    try std.testing.expect(!c1.eql(c2));
}

test "transcript challenge bytes" {
    const F = @import("../field/mod.zig").BN254Scalar;
    const allocator = std.testing.allocator;

    var transcript = try Transcript(F).init(allocator, "test");
    defer transcript.deinit();

    try transcript.appendBytes("test data");

    var output1: [64]u8 = undefined;
    var output2: [64]u8 = undefined;

    var t2 = try Transcript(F).init(allocator, "test");
    defer t2.deinit();
    try t2.appendBytes("test data");

    try transcript.challengeBytes("label", &output1);
    try t2.challengeBytes("label", &output2);

    // Same inputs should produce same output
    try std.testing.expectEqualSlices(u8, &output1, &output2);
}
