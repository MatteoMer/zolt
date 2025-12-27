//! Spartan proof system for Jolt
//!
//! Spartan is a zkSNARK for R1CS that achieves O(n) prover time
//! using polynomial commitments and the sumcheck protocol.

const std = @import("std");
const Allocator = std.mem.Allocator;
const poly = @import("../../poly/mod.zig");
const subprotocols = @import("../../subprotocols/mod.zig");

/// Spartan proof for R1CS
pub fn R1CSProof(comptime F: type) type {
    return struct {
        const Self = @This();

        /// Sumcheck proof for the outer sumcheck
        outer_sumcheck: subprotocols.Sumcheck(F).Proof,
        /// Sumcheck proof for the inner sumcheck
        inner_sumcheck: subprotocols.Sumcheck(F).Proof,
        /// Polynomial opening proofs
        openings: []F,
        /// Claims
        claims: []F,

        pub fn deinit(self: *Self, allocator: Allocator) void {
            self.outer_sumcheck.deinit();
            self.inner_sumcheck.deinit();
            allocator.free(self.openings);
            allocator.free(self.claims);
        }
    };
}

/// Spartan prover
pub fn SpartanProver(comptime F: type) type {
    return struct {
        const Self = @This();

        allocator: Allocator,

        pub fn init(allocator: Allocator) Self {
            return .{
                .allocator = allocator,
            };
        }

        /// Generate a Spartan proof for an R1CS instance
        pub fn prove(
            self: *Self,
            _: anytype, // R1CS instance
            _: []const F, // witness
        ) !R1CSProof(F) {
            _ = self;
            @panic("SpartanProver.prove not yet implemented");
        }
    };
}

/// Spartan verifier
pub fn SpartanVerifier(comptime F: type) type {
    return struct {
        const Self = @This();

        allocator: Allocator,

        pub fn init(allocator: Allocator) Self {
            return .{
                .allocator = allocator,
            };
        }

        /// Verify a Spartan proof
        pub fn verify(
            self: *Self,
            _: anytype, // R1CS instance
            _: []const F, // public inputs
            _: *const R1CSProof(F),
        ) !bool {
            _ = self;
            @panic("SpartanVerifier.verify not yet implemented");
        }
    };
}

/// Uniform Spartan (for R1CS with uniform structure)
pub fn UniformSpartan(comptime F: type) type {
    return struct {
        const Self = @This();
        const FieldType = F;

        _marker: ?*const FieldType = null,

        // TODO: Implement uniform Spartan optimizations

        pub fn init() Self {
            return .{};
        }
    };
}

test "spartan types compile" {
    const F = @import("../../field/mod.zig").BN254Scalar;

    // Verify types compile
    _ = R1CSProof(F);
    _ = SpartanProver(F);
    _ = SpartanVerifier(F);
}
