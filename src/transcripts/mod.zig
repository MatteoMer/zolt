//! Fiat-Shamir transcript for Jolt
//!
//! This module implements the Fiat-Shamir transformation for making
//! interactive proofs non-interactive. It uses Keccak-f[1600] (SHA-3 core)
//! to derive challenges from the transcript.

const std = @import("std");
const Allocator = std.mem.Allocator;

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
pub fn PoseidonTranscript(comptime F: type) type {
    return struct {
        const Self = @This();

        state: [3]F,
        allocator: Allocator,

        pub fn init(allocator: Allocator, _: []const u8) Self {
            return .{
                .state = .{ F.zero(), F.zero(), F.zero() },
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

        /// Poseidon permutation (placeholder)
        fn permute(self: *Self) void {
            // TODO: Implement proper Poseidon permutation
            self.state[0] = self.state[0].add(self.state[1]);
            self.state[1] = self.state[1].add(self.state[2]);
            self.state[2] = self.state[2].add(self.state[0]);
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
