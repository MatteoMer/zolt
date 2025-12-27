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

            pub fn init(polynomial: poly.DensePolynomial(F)) Prover {
                return .{
                    .polynomial = polynomial,
                    .round = 0,
                };
            }

            /// Generate the next round polynomial
            pub fn nextRound(self: *Prover, _: Allocator) !Round {
                // TODO: Implement sumcheck round
                _ = self;
                @panic("sumcheck prover not yet implemented");
            }

            /// Receive challenge and update state
            pub fn receiveChallenge(self: *Prover, challenge: F) !void {
                // Bind the current variable to the challenge
                const new_poly = try self.polynomial.bindFirst(challenge);
                self.polynomial.deinit();
                self.polynomial = new_poly;
                self.round += 1;
            }
        };

        /// Verifier state
        pub const Verifier = struct {
            claim: F,
            round: usize,
            challenges: std.ArrayList(F),

            pub fn init(allocator: Allocator, claim: F) Verifier {
                return .{
                    .claim = claim,
                    .round = 0,
                    .challenges = std.ArrayList(F).init(allocator),
                };
            }

            pub fn deinit(self: *Verifier) void {
                self.challenges.deinit();
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

                // Generate random challenge (in real impl, from Fiat-Shamir)
                const challenge = F.fromU64(42); // TODO: proper randomness
                try self.challenges.append(challenge);

                // Update claim for next round
                self.claim = round.poly.evaluate(challenge);
                self.round += 1;

                return challenge;
            }
        };
    };
}

/// Streaming sumcheck for memory-efficient proving
pub fn StreamingSumcheck(comptime F: type) type {
    return struct {
        const Self = @This();
        const FieldType = F;

        _marker: ?*const FieldType = null,

        // TODO: Implement streaming sumcheck
        pub fn init() Self {
            return .{};
        }
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
