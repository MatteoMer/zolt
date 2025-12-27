//! Subprotocols for Jolt
//!
//! This module contains the core cryptographic subprotocols used in Jolt:
//! - Sumcheck protocol
//! - GKR protocol
//! - Memory checking

const std = @import("std");
const Allocator = std.mem.Allocator;
const poly = @import("../poly/mod.zig");

/// Sumcheck protocol
///
/// The sumcheck protocol is used to prove claims of the form:
/// sum_{x in {0,1}^n} f(x) = v
///
/// where f is a multilinear polynomial.
pub fn Sumcheck(comptime F: type) type {
    return struct {
        const Self = @This();

        /// A single round of the sumcheck protocol
        pub const Round = struct {
            /// Univariate polynomial for this round
            poly: poly.UniPoly(F),
        };

        /// Complete sumcheck proof
        pub const Proof = struct {
            /// Claimed sum
            claim: F,
            /// Rounds of the protocol
            rounds: []Round,
            /// Final evaluation point
            final_point: []F,
            /// Final evaluation
            final_eval: F,
            allocator: Allocator,

            pub fn deinit(self: *Proof) void {
                for (self.rounds) |*r| {
                    r.poly.deinit();
                }
                self.allocator.free(self.rounds);
                self.allocator.free(self.final_point);
            }
        };

        /// Prover state
        pub const Prover = struct {
            polynomial: poly.DensePolynomial(F),
            round: usize,
            allocator: Allocator,

            pub fn init(allocator: Allocator, polynomial: poly.DensePolynomial(F)) Prover {
                return .{
                    .polynomial = polynomial,
                    .round = 0,
                    .allocator = allocator,
                };
            }

            /// Generate the next round polynomial
            ///
            /// In round i, we compute the univariate polynomial:
            /// g_i(X_i) = sum_{x_{i+1}, ..., x_n in {0,1}} f(r_1, ..., r_{i-1}, X_i, x_{i+1}, ..., x_n)
            ///
            /// This is a polynomial of degree at most 1 (for multilinear f).
            pub fn nextRound(self: *Prover, allocator: Allocator) !Round {
                // For a multilinear polynomial, the round polynomial has degree 1
                // We need to compute g(X) = sum_{rest} f(X, rest)
                //
                // g(0) = sum over all evaluations where first bit is 0
                // g(1) = sum over all evaluations where first bit is 1
                //
                // Since f is multilinear, g(X) = g(0) + X * (g(1) - g(0))
                // Which in coefficient form is: [g(0), g(1) - g(0)]

                const half = self.polynomial.evaluations.len / 2;

                // g(0) = sum of evaluations with first variable = 0
                var g0 = F.zero();
                for (0..half) |i| {
                    g0 = g0.add(self.polynomial.evaluations[i]);
                }

                // g(1) = sum of evaluations with first variable = 1
                var g1 = F.zero();
                for (0..half) |i| {
                    g1 = g1.add(self.polynomial.evaluations[i + half]);
                }

                // Coefficients: [g(0), g(1) - g(0)]
                const coeffs = try allocator.alloc(F, 2);
                coeffs[0] = g0;
                coeffs[1] = g1.sub(g0);

                return Round{
                    .poly = poly.UniPoly(F){
                        .coeffs = coeffs,
                        .allocator = allocator,
                    },
                };
            }

            /// Receive challenge and update state
            pub fn receiveChallenge(self: *Prover, challenge: F) !void {
                // Bind the current variable to the challenge
                const new_poly = try self.polynomial.bindFirst(challenge);
                self.polynomial.deinit();
                self.polynomial = new_poly;
                self.round += 1;
            }

            /// Check if the protocol is complete
            pub fn isComplete(self: *const Prover) bool {
                return self.polynomial.evaluations.len == 1;
            }

            /// Get the final evaluation (when protocol is complete)
            pub fn getFinalEval(self: *const Prover) F {
                std.debug.assert(self.isComplete());
                return self.polynomial.evaluations[0];
            }
        };

        /// Verifier state
        pub const Verifier = struct {
            claim: F,
            round: usize,
            challenges: []F,
            challenges_len: usize,
            /// Internal counter for deterministic challenge generation
            /// In production, this is derived from Fiat-Shamir transcript
            challenge_counter: u64,
            allocator: Allocator,

            pub fn init(allocator: Allocator, claim: F) Verifier {
                return .{
                    .claim = claim,
                    .round = 0,
                    .challenges = &[_]F{},
                    .challenges_len = 0,
                    .challenge_counter = 0,
                    .allocator = allocator,
                };
            }

            pub fn deinit(self: *Verifier) void {
                if (self.challenges.len > 0) {
                    self.allocator.free(self.challenges);
                }
            }

            /// Verify a round and generate challenge
            pub fn verifyRound(self: *Verifier, round: Round) !F {
                // Check that p(0) + p(1) = current claim
                const p0 = round.poly.evaluate(F.zero());
                const p1 = round.poly.evaluate(F.one());
                const sum = p0.add(p1);

                if (!sum.eql(self.claim)) {
                    return error.SumcheckVerificationFailed;
                }

                // Generate challenge using Fiat-Shamir-style deterministic derivation
                // The challenge is derived from:
                // - Current round number
                // - Round polynomial coefficients
                // - Previous claim
                // This ensures reproducibility for testing while being deterministic
                const challenge = self.deriveChallenge(&round);

                // Grow challenges array
                const new_len = self.challenges_len + 1;
                const new_challenges = try self.allocator.alloc(F, new_len);
                if (self.challenges.len > 0) {
                    @memcpy(new_challenges[0..self.challenges_len], self.challenges);
                    self.allocator.free(self.challenges);
                }
                new_challenges[self.challenges_len] = challenge;
                self.challenges = new_challenges;
                self.challenges_len = new_len;

                // Update claim for next round
                self.claim = round.poly.evaluate(challenge);
                self.round += 1;

                return challenge;
            }

            /// Derive a deterministic challenge from the round polynomial
            /// This implements a simplified Fiat-Shamir transformation
            fn deriveChallenge(self: *Verifier, round: *const Round) F {
                // Hash the round polynomial coefficients with the round number
                // Using a simple but deterministic mixing function
                var hash_state: u64 = 0x9e3779b97f4a7c15; // Golden ratio constant

                // Mix in round number
                hash_state ^= @as(u64, @intCast(self.round));
                hash_state *%= 0xff51afd7ed558ccd;

                // Mix in claim
                for (self.claim.limbs) |limb| {
                    hash_state ^= limb;
                    hash_state *%= 0xc4ceb9fe1a85ec53;
                }

                // Mix in polynomial coefficients
                for (round.poly.coeffs) |coeff| {
                    for (coeff.limbs) |limb| {
                        hash_state ^= limb;
                        hash_state *%= 0xff51afd7ed558ccd;
                        hash_state ^= hash_state >> 33;
                    }
                }

                // Final mixing
                hash_state ^= hash_state >> 33;
                hash_state *%= 0xff51afd7ed558ccd;
                hash_state ^= hash_state >> 33;

                self.challenge_counter += 1;

                return F.fromU64(hash_state);
            }
        };
    };
}

/// Streaming sumcheck for memory-efficient proving
///
/// This is a placeholder for a streaming implementation that processes
/// polynomial evaluations in chunks rather than loading them all into memory.
/// The standard Sumcheck implementation above is sufficient for most use cases.
/// Streaming is beneficial for very large polynomials (2^25+ variables).
pub fn StreamingSumcheck(comptime F: type) type {
    return struct {
        const Self = @This();

        /// Chunk size for streaming evaluation (64KB of field elements)
        pub const CHUNK_SIZE: usize = 65536 / @sizeOf(F);

        chunk_buffer: []F,
        current_chunk: usize,
        total_chunks: usize,
        allocator: Allocator,

        pub fn init(allocator: Allocator, num_vars: usize) !Self {
            const total_evals = @as(usize, 1) << num_vars;
            const total_chunks = (total_evals + CHUNK_SIZE - 1) / CHUNK_SIZE;
            const chunk_buffer = try allocator.alloc(F, CHUNK_SIZE);

            return Self{
                .chunk_buffer = chunk_buffer,
                .current_chunk = 0,
                .total_chunks = total_chunks,
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.chunk_buffer);
        }

        /// Process the next chunk of evaluations
        /// Returns true if more chunks remain
        pub fn processChunk(self: *Self, evals: []const F) bool {
            // Copy evaluations into buffer for processing
            const copy_len = @min(evals.len, self.chunk_buffer.len);
            @memcpy(self.chunk_buffer[0..copy_len], evals[0..copy_len]);

            self.current_chunk += 1;
            return self.current_chunk < self.total_chunks;
        }

        /// Check if streaming is complete
        pub fn isComplete(self: *const Self) bool {
            return self.current_chunk >= self.total_chunks;
        }
    };
}

/// Run the complete sumcheck protocol between prover and verifier
pub fn runSumcheck(comptime F: type, polynomial: poly.DensePolynomial(F), allocator: Allocator) !struct { proof: Sumcheck(F).Proof, result: bool } {
    const SC = Sumcheck(F);

    // Calculate initial claim (sum over boolean hypercube)
    var claim = F.zero();
    for (polynomial.evaluations) |eval| {
        claim = claim.add(eval);
    }

    // Initialize prover and verifier
    var prover = SC.Prover.init(allocator, polynomial);
    var verifier = SC.Verifier.init(allocator, claim);
    defer verifier.deinit();

    const num_rounds = polynomial.num_vars;
    const rounds = try allocator.alloc(SC.Round, num_rounds);
    const challenges = try allocator.alloc(F, num_rounds);

    // Run the protocol
    for (0..num_rounds) |i| {
        // Prover sends round polynomial
        rounds[i] = try prover.nextRound(allocator);

        // Verifier checks and sends challenge
        const challenge = try verifier.verifyRound(rounds[i]);
        challenges[i] = challenge;

        // Prover updates state
        try prover.receiveChallenge(challenge);
    }

    // Get final evaluation
    const final_eval = prover.getFinalEval();

    // Clean up prover's remaining polynomial
    prover.polynomial.deinit();

    // Copy challenges to final_point
    const final_point = try allocator.alloc(F, num_rounds);
    @memcpy(final_point, challenges);
    allocator.free(challenges);

    return .{
        .proof = SC.Proof{
            .claim = claim,
            .rounds = rounds,
            .final_point = final_point,
            .final_eval = final_eval,
            .allocator = allocator,
        },
        .result = verifier.claim.eql(final_eval),
    };
}

test "sumcheck types compile" {
    const F = @import("../field/mod.zig").BN254Scalar;
    const SC = Sumcheck(F);

    // Just verify types compile
    _ = SC.Proof;
    _ = SC.Prover;
    _ = SC.Verifier;
}

test "sumcheck prover round generation" {
    const F = @import("../field/mod.zig").BN254Scalar;
    const allocator = std.testing.allocator;
    const SC = Sumcheck(F);

    // Create a simple polynomial with 2 variables
    // The dense polynomial stores evaluations indexed by binary representation
    // Index 0 = bits 00 -> f(0,0), Index 1 = bits 01 -> f(1,0)
    // Index 2 = bits 10 -> f(0,1), Index 3 = bits 11 -> f(1,1)
    //
    // In bindFirst, we bind the FIRST variable (x_0), which corresponds to the low bit.
    // Low bit = 0: indices 0, 2 (evaluations 1, 3)
    // Low bit = 1: indices 1, 3 (evaluations 2, 4)
    const evals = [_]F{
        F.fromU64(1), // f(0,0) - index 0
        F.fromU64(2), // f(1,0) - index 1
        F.fromU64(3), // f(0,1) - index 2
        F.fromU64(4), // f(1,1) - index 3
    };

    const polynomial = try poly.DensePolynomial(F).init(allocator, &evals);
    var prover = SC.Prover.init(allocator, polynomial);

    // First round: bind x0, sum over x1
    // The prover computes g(X) where:
    // g(0) = sum of evaluations with first bit = 0 = evals[0] + evals[2] = 1 + 3 = 4
    // g(1) = sum of evaluations with first bit = 1 = evals[1] + evals[3] = 2 + 4 = 6
    //
    // Wait, actually our implementation splits at half:
    // First half: evals[0..half] = [1, 2]
    // Second half: evals[half..] = [3, 4]
    // g(0) = 1 + 2 = 3
    // g(1) = 3 + 4 = 7
    var round1 = try prover.nextRound(allocator);
    defer round1.poly.deinit();

    const g0_at_0 = round1.poly.evaluate(F.zero());
    const g0_at_1 = round1.poly.evaluate(F.one());

    // With our half-split approach:
    // g(0) = sum of first half = 1 + 2 = 3
    // g(1) = sum of second half = 3 + 4 = 7
    try std.testing.expect(g0_at_0.eql(F.fromU64(3)));
    try std.testing.expect(g0_at_1.eql(F.fromU64(7)));

    // Verify g(0) + g(1) = total sum = 1 + 2 + 3 + 4 = 10
    const sum = g0_at_0.add(g0_at_1);
    try std.testing.expect(sum.eql(F.fromU64(10)));

    // Apply challenge r0 = 2
    const r0 = F.fromU64(2);
    try prover.receiveChallenge(r0);

    // Second round
    var round2 = try prover.nextRound(allocator);
    defer round2.poly.deinit();

    const g1_at_0 = round2.poly.evaluate(F.zero());
    const g1_at_1 = round2.poly.evaluate(F.one());

    // After binding the first variable to r0=2:
    // new_evals[i] = (1-r0)*old_evals[i] + r0*old_evals[i+half]
    // new_evals[0] = (1-2)*1 + 2*3 = -1 + 6 = 5
    // new_evals[1] = (1-2)*2 + 2*4 = -2 + 8 = 6
    // g1(0) = 5, g1(1) = 6
    try std.testing.expect(g1_at_0.eql(F.fromU64(5)));
    try std.testing.expect(g1_at_1.eql(F.fromU64(6)));

    // Verify g1(0) + g1(1) = g0(r0) = g0(2)
    const expected_claim = round1.poly.evaluate(r0);
    try std.testing.expect(g1_at_0.add(g1_at_1).eql(expected_claim));

    prover.polynomial.deinit();
}

test "sumcheck complete protocol" {
    const F = @import("../field/mod.zig").BN254Scalar;
    const allocator = std.testing.allocator;

    // Create polynomial with 3 variables
    const evals = [_]F{
        F.fromU64(1), F.fromU64(2), F.fromU64(3), F.fromU64(4),
        F.fromU64(5), F.fromU64(6), F.fromU64(7), F.fromU64(8),
    };

    const polynomial = try poly.DensePolynomial(F).init(allocator, &evals);

    var result = try runSumcheck(F, polynomial, allocator);
    defer result.proof.deinit();

    // Verify the sumcheck succeeded
    try std.testing.expect(result.result);

    // Total sum should be 1+2+3+4+5+6+7+8 = 36
    try std.testing.expect(result.proof.claim.eql(F.fromU64(36)));
}
