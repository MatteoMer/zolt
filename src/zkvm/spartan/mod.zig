//! Spartan proof system for Jolt
//!
//! Spartan is a zkSNARK for R1CS that achieves O(n) prover time
//! using polynomial commitments and the sumcheck protocol.
//!
//! The main idea is to prove that for all i:
//!   (Az)_i * (Bz)_i = (Cz)_i
//!
//! Using the sumcheck protocol on:
//!   sum_{x in {0,1}^log(m)} eq(tau, x) * [(Az)(x) * (Bz)(x) - (Cz)(x)] = 0

const std = @import("std");
const Allocator = std.mem.Allocator;
const poly = @import("zolt_arith").poly;
const subprotocols = @import("zolt_arith").subprotocols;
const r1cs = @import("../r1cs/mod.zig");


// Streaming outer prover for Jolt compatibility
pub const streaming_outer = @import("streaming_outer.zig");
pub const StreamingOuterProver = streaming_outer.StreamingOuterProver;

// Product virtualization remainder prover for Stage 2
pub const product_remainder = @import("product_remainder.zig");
pub const ProductVirtualRemainderProver = product_remainder.ProductVirtualRemainderProver;

// Stage 3 prover (ShiftSumcheck, InstructionInput, RegistersClaimReduction)
pub const stage3_prover = @import("stage3_prover.zig");
pub const Stage3Prover = stage3_prover.Stage3Prover;
pub const Stage3Result = stage3_prover.Stage3Result;

// Stage 4 orchestrating prover (RegistersRWC + RamValCheck batched sumcheck)
pub const stage4_prover_mod = @import("stage4_prover.zig");
pub const Stage4Prover = stage4_prover_mod.Stage4Prover;
pub const Stage4Result = stage4_prover_mod.Stage4Result;

// Stage 4 Gruen prover (matches Jolt's algorithm exactly)
pub const stage4_gruen_prover = @import("stage4_gruen_prover.zig");
pub const Stage4GruenProver = stage4_gruen_prover.Stage4GruenProver;

// Gruen eq polynomial optimization
pub const gruen_eq = @import("gruen_eq.zig");
pub const GruenSplitEqPolynomial = gruen_eq.GruenSplitEqPolynomial;

// Prefix-suffix optimization for Stage 3
pub const prefix_suffix = @import("prefix_suffix.zig");
pub const Phase1Prover = prefix_suffix.Phase1Prover;

// Stage 5 prover (RegistersValEvaluation, RamRaClaimReduction, LookupsReadRaf)
pub const stage5_prover = @import("stage5_prover.zig");
pub const Stage5BatchedProver = stage5_prover.Stage5BatchedProver;
pub const Stage5Result = stage5_prover.Stage5Result;

// Stage 6 prover (BytecodeReadRaf, HammingBooleanity, Booleanity, RamRaVirtual, LookupsRaVirtual, IncClaimReduction)
pub const stage6_prover = @import("stage6_prover.zig");
pub const Stage6BatchedProver = stage6_prover.Stage6BatchedProver;
pub const Stage6Result = stage6_prover.Stage6Result;

// Stage 7 prover (HammingWeightClaimReduction)
pub const stage7_prover_mod = @import("stage7_prover.zig");
pub const Stage7Prover = stage7_prover_mod.Stage7Prover;
pub const Stage7Result = stage7_prover_mod.Stage7Result;

// Shared sumcheck helper functions
pub const sumcheck_helpers = @import("sumcheck_helpers.zig");


/// Spartan proof for R1CS
pub fn R1CSProof(comptime F: type) type {
    return struct {
        const Self = @This();

        /// Random evaluation point from verifier
        tau: []F,
        /// Sumcheck proof for the outer sumcheck
        sumcheck_proof: subprotocols.Sumcheck(F).Proof,
        /// Claimed evaluations at random point
        /// [A(r), B(r), C(r)] where r is the final sumcheck point
        eval_claims: [3]F,
        /// Final evaluation point (from sumcheck)
        eval_point: []F,
        allocator: Allocator,

        /// Create a placeholder proof (for testing/placeholder purposes)
        pub fn placeholder(allocator: Allocator) !Self {
            const tau = try allocator.alloc(F, 1);
            tau[0] = F.zero();

            const eval_point = try allocator.alloc(F, 1);
            eval_point[0] = F.zero();

            const rounds = try allocator.alloc(subprotocols.Sumcheck(F).Round, 0);
            const final_point = try allocator.alloc(F, 0);

            return Self{
                .tau = tau,
                .sumcheck_proof = .{
                    .claim = F.zero(),
                    .rounds = rounds,
                    .final_point = final_point,
                    .final_eval = F.zero(),
                    .allocator = allocator,
                },
                .eval_claims = .{ F.zero(), F.zero(), F.zero() },
                .eval_point = eval_point,
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.tau);
            self.sumcheck_proof.deinit();
            self.allocator.free(self.eval_point);
        }
    };
}

/// Preprocessed R1CS instance for Spartan
pub fn R1CSShape(comptime F: type) type {
    return struct {
        const Self = @This();

        /// Matrix A in sparse form
        A: r1cs.SparseMatrix(F),
        /// Matrix B in sparse form
        B: r1cs.SparseMatrix(F),
        /// Matrix C in sparse form
        C: r1cs.SparseMatrix(F),
        /// Number of constraints (m)
        num_constraints: usize,
        /// Number of variables (n)
        num_vars: usize,
        /// Number of public inputs
        num_public: usize,
        allocator: Allocator,

        pub fn init(allocator: Allocator, instance: *const r1cs.R1CSInstance(F)) !Self {
            return .{
                .A = try r1cs.SparseMatrix(F).fromR1CS(allocator, instance, .A),
                .B = try r1cs.SparseMatrix(F).fromR1CS(allocator, instance, .B),
                .C = try r1cs.SparseMatrix(F).fromR1CS(allocator, instance, .C),
                .num_constraints = instance.constraints_len,
                .num_vars = instance.num_vars,
                .num_public = instance.num_public,
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            self.A.deinit();
            self.B.deinit();
            self.C.deinit();
        }
    };
}

test "spartan types compile" {
    const F = @import("zolt_arith").field.BN254Scalar;

    // Verify types compile
    _ = R1CSProof(F);
    _ = R1CSShape(F);
}

test {
    // Discover tests in sub-modules (refAllDecls doesn't traverse @import'd modules)
    _ = @import("ra_poly.zig");
    _ = @import("stage6_prover.zig");
    _ = @import("stage5_prover.zig");
    _ = @import("stage5_instances.zig");
    _ = @import("sumcheck_helpers.zig");
}
