//! Fiat-Shamir transcript for Jolt
//!
//! This module implements the Fiat-Shamir transformation for making
//! interactive proofs non-interactive. It uses a hash function to
//! derive challenges from the transcript.

const std = @import("std");
const Allocator = std.mem.Allocator;

/// Keccak256 state size
const KECCAK_STATE_SIZE = 200;
const KECCAK_RATE = 136;

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

        /// Keccak-f permutation (simplified placeholder)
        fn keccakF(self: *Self) void {
            // TODO: Implement proper Keccak-f[1600]
            // For now, just do a simple mixing
            var i: usize = 0;
            while (i < KECCAK_STATE_SIZE - 1) : (i += 1) {
                self.state[i] ^= self.state[i + 1];
            }
            self.state[KECCAK_STATE_SIZE - 1] ^= self.state[0];

            // Rotate bytes
            const first = self.state[0];
            for (0..KECCAK_STATE_SIZE - 1) |j| {
                self.state[j] = self.state[j + 1];
            }
            self.state[KECCAK_STATE_SIZE - 1] = first;
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
