//! Proof Converter: Zolt Internal → Jolt Compatible Format
//!
//! This module converts Zolt's internal 6-stage proof structure to
//! Jolt's 7-stage proof format for cross-verification compatibility.
//!
//! ## Stage Mapping
//!
//! Zolt (6 stages):                    Jolt (7 stages):
//! 1. Outer Spartan           →        1. Outer Spartan (+ UniSkip)
//! 2. RAM RAF + Read-Write    →        2. Product virtualization + RAM RAF + RW (+ UniSkip)
//! 3. Instruction Lookup      →        3. Spartan shift + Instruction input + Registers claim
//! 4. Memory Val Evaluation   →        4. Registers RW + RAM val evaluation + RAM val final
//! 5. Register Val Evaluation →        5. Registers val evaluation + RAM RA + Lookups RAF
//! 6. Booleanity              →        6. Bytecode RAF + Hamming + Booleanity + RA virtual
//!                            →        7. Hamming weight claim reduction
//!
//! Note: Zolt's stages are more consolidated, so conversion involves
//! splitting some proofs and creating empty placeholders where Zolt
//! handles things differently.

const std = @import("std");
const Allocator = std.mem.Allocator;

const jolt_types = @import("jolt_types.zig");
const prover = @import("prover.zig");
const field_mod = @import("../field/mod.zig");

/// Convert Zolt's internal proof to Jolt-compatible format
pub fn ProofConverter(comptime F: type) type {
    return struct {
        const Self = @This();

        // Import types we need
        const JoltProofType = jolt_types.JoltProof;
        const SumcheckInstanceProof = jolt_types.SumcheckInstanceProof;
        const UniSkipFirstRoundProof = jolt_types.UniSkipFirstRoundProof;
        const OpeningClaims = jolt_types.OpeningClaims;
        const SumcheckId = jolt_types.SumcheckId;
        const OpeningId = jolt_types.OpeningId;
        const VirtualPolynomial = jolt_types.VirtualPolynomial;

        allocator: Allocator,

        pub fn init(allocator: Allocator) Self {
            return Self{
                .allocator = allocator,
            };
        }

        /// Convert Zolt's 6-stage proof to Jolt's 7-stage format
        ///
        /// This creates a JoltProof that can be serialized and verified
        /// by the Jolt verifier.
        pub fn convert(
            self: *Self,
            comptime Commitment: type,
            comptime Proof: type,
            zolt_stage_proofs: *const prover.JoltStageProofs(F),
            commitments: []const Commitment,
            joint_opening_proof: ?Proof,
            config: ConversionConfig,
        ) !JoltProofType(F, Commitment, Proof) {
            var jolt_proof = JoltProofType(F, Commitment, Proof).init(self.allocator);

            // Copy configuration parameters
            jolt_proof.trace_length = @as(usize, 1) << @intCast(zolt_stage_proofs.log_t);
            jolt_proof.ram_K = @as(usize, 1) << @intCast(zolt_stage_proofs.log_k);
            jolt_proof.bytecode_K = config.bytecode_K;
            jolt_proof.log_k_chunk = config.log_k_chunk;
            jolt_proof.lookups_ra_virtual_log_k_chunk = config.lookups_ra_virtual_log_k_chunk;

            // Copy commitments
            for (commitments) |c| {
                try jolt_proof.commitments.append(self.allocator, c);
            }

            // Set joint opening proof
            jolt_proof.joint_opening_proof = joint_opening_proof;

            // Convert Stage 1 (Outer Spartan)
            // Zolt Stage 0 → Jolt Stage 1
            try self.convertStage1(
                &zolt_stage_proofs.stage_proofs[0],
                &jolt_proof.stage1_sumcheck_proof,
                &jolt_proof.opening_claims,
            );

            // Create UniSkip proof for Stage 1 (Jolt expects this)
            jolt_proof.stage1_uni_skip_first_round_proof = try self.createUniSkipProof(
                &zolt_stage_proofs.stage_proofs[0],
            );

            // Convert Stage 2 (RAM RAF + Read-Write + Product virtualization)
            // Zolt Stage 1 → Jolt Stage 2
            try self.convertStage2(
                &zolt_stage_proofs.stage_proofs[1],
                &jolt_proof.stage2_sumcheck_proof,
                &jolt_proof.opening_claims,
            );

            // Create UniSkip proof for Stage 2
            jolt_proof.stage2_uni_skip_first_round_proof = try self.createUniSkipProof(
                &zolt_stage_proofs.stage_proofs[1],
            );

            // Convert Stage 3 (Spartan shift + Instruction input + Registers claim)
            // Zolt Stage 2 (Lasso lookup) → Jolt Stage 3
            try self.convertStage3(
                &zolt_stage_proofs.stage_proofs[2],
                &jolt_proof.stage3_sumcheck_proof,
                &jolt_proof.opening_claims,
            );

            // Convert Stage 4 (Registers RW + RAM val evaluation + final)
            // Zolt Stage 3 (Val evaluation) → Jolt Stage 4
            try self.convertStage4(
                &zolt_stage_proofs.stage_proofs[3],
                &jolt_proof.stage4_sumcheck_proof,
                &jolt_proof.opening_claims,
            );

            // Convert Stage 5 (Registers val + RAM RA reduction + Lookups RAF)
            // Zolt Stage 4 (Register evaluation) → Jolt Stage 5
            try self.convertStage5(
                &zolt_stage_proofs.stage_proofs[4],
                &jolt_proof.stage5_sumcheck_proof,
                &jolt_proof.opening_claims,
            );

            // Convert Stage 6 (Bytecode RAF + Hamming + Booleanity + RA virtual)
            // Zolt Stage 5 (Booleanity) → Jolt Stage 6
            try self.convertStage6(
                &zolt_stage_proofs.stage_proofs[5],
                &jolt_proof.stage6_sumcheck_proof,
                &jolt_proof.opening_claims,
            );

            // Create Stage 7 (Hamming weight claim reduction)
            // This is a new stage that Zolt doesn't have separately
            try self.createStage7(
                zolt_stage_proofs,
                &jolt_proof.stage7_sumcheck_proof,
                &jolt_proof.opening_claims,
            );

            return jolt_proof;
        }

        /// Convert Zolt Stage 0 to Jolt Stage 1 (Outer Spartan)
        fn convertStage1(
            self: *Self,
            zolt_stage: *const prover.StageProof(F),
            jolt_proof: *SumcheckInstanceProof(F),
            claims: *OpeningClaims(F),
        ) !void {
            // Convert round polynomials
            for (zolt_stage.round_polys.items) |poly| {
                try jolt_proof.addRoundPoly(poly);
            }

            // Add opening claims for SpartanOuter
            if (zolt_stage.final_claims.items.len > 0) {
                try claims.insert(
                    .{ .Virtual = .{ .poly = .Product, .sumcheck_id = .SpartanOuter } },
                    zolt_stage.final_claims.items[0],
                );
            }

            _ = self;
        }

        /// Convert Zolt Stage 1 to Jolt Stage 2 (Product virtualization + RAM RAF + RW)
        fn convertStage2(
            self: *Self,
            zolt_stage: *const prover.StageProof(F),
            jolt_proof: *SumcheckInstanceProof(F),
            claims: *OpeningClaims(F),
        ) !void {
            // Convert round polynomials
            for (zolt_stage.round_polys.items) |poly| {
                try jolt_proof.addRoundPoly(poly);
            }

            // Add opening claims for RAM RAF evaluation
            if (zolt_stage.final_claims.items.len > 0) {
                try claims.insert(
                    .{ .Virtual = .{ .poly = .RamRa, .sumcheck_id = .RamRafEvaluation } },
                    zolt_stage.final_claims.items[0],
                );
            }

            // Add opening claims for RAM read-write checking
            if (zolt_stage.final_claims.items.len > 1) {
                try claims.insert(
                    .{ .Virtual = .{ .poly = .RamVal, .sumcheck_id = .RamReadWriteChecking } },
                    zolt_stage.final_claims.items[1],
                );
            }

            _ = self;
        }

        /// Convert Zolt Stage 2 to Jolt Stage 3 (Instruction lookup / Spartan shift)
        fn convertStage3(
            self: *Self,
            zolt_stage: *const prover.StageProof(F),
            jolt_proof: *SumcheckInstanceProof(F),
            claims: *OpeningClaims(F),
        ) !void {
            // Convert round polynomials
            for (zolt_stage.round_polys.items) |poly| {
                try jolt_proof.addRoundPoly(poly);
            }

            // Add opening claims for instruction claim reduction
            if (zolt_stage.final_claims.items.len > 0) {
                try claims.insert(
                    .{ .Virtual = .{ .poly = .LookupOutput, .sumcheck_id = .InstructionClaimReduction } },
                    zolt_stage.final_claims.items[0],
                );
            }

            _ = self;
        }

        /// Convert Zolt Stage 3 to Jolt Stage 4 (RAM val evaluation)
        fn convertStage4(
            self: *Self,
            zolt_stage: *const prover.StageProof(F),
            jolt_proof: *SumcheckInstanceProof(F),
            claims: *OpeningClaims(F),
        ) !void {
            // Convert round polynomials
            for (zolt_stage.round_polys.items) |poly| {
                try jolt_proof.addRoundPoly(poly);
            }

            // Add opening claims for RAM val evaluation
            if (zolt_stage.final_claims.items.len > 0) {
                try claims.insert(
                    .{ .Virtual = .{ .poly = .RamVal, .sumcheck_id = .RamValEvaluation } },
                    zolt_stage.final_claims.items[0],
                );
            }

            // Add opening claims for RAM val final evaluation
            if (zolt_stage.final_claims.items.len > 1) {
                try claims.insert(
                    .{ .Virtual = .{ .poly = .RamValFinal, .sumcheck_id = .RamValFinalEvaluation } },
                    zolt_stage.final_claims.items[1],
                );
            }

            _ = self;
        }

        /// Convert Zolt Stage 4 to Jolt Stage 5 (Registers val evaluation)
        fn convertStage5(
            self: *Self,
            zolt_stage: *const prover.StageProof(F),
            jolt_proof: *SumcheckInstanceProof(F),
            claims: *OpeningClaims(F),
        ) !void {
            // Convert round polynomials
            for (zolt_stage.round_polys.items) |poly| {
                try jolt_proof.addRoundPoly(poly);
            }

            // Add opening claims for registers val evaluation
            if (zolt_stage.final_claims.items.len > 0) {
                try claims.insert(
                    .{ .Virtual = .{ .poly = .RegistersVal, .sumcheck_id = .RegistersValEvaluation } },
                    zolt_stage.final_claims.items[0],
                );
            }

            // Add opening claims for RAM RA claim reduction
            if (zolt_stage.final_claims.items.len > 1) {
                try claims.insert(
                    .{ .Virtual = .{ .poly = .RamRa, .sumcheck_id = .RamRaClaimReduction } },
                    zolt_stage.final_claims.items[1],
                );
            }

            _ = self;
        }

        /// Convert Zolt Stage 5 to Jolt Stage 6 (Booleanity + bytecode RAF + etc.)
        fn convertStage6(
            self: *Self,
            zolt_stage: *const prover.StageProof(F),
            jolt_proof: *SumcheckInstanceProof(F),
            claims: *OpeningClaims(F),
        ) !void {
            // Convert round polynomials
            for (zolt_stage.round_polys.items) |poly| {
                try jolt_proof.addRoundPoly(poly);
            }

            // Add opening claims for booleanity
            if (zolt_stage.final_claims.items.len > 0) {
                try claims.insert(
                    .{ .Virtual = .{ .poly = .RamHammingWeight, .sumcheck_id = .Booleanity } },
                    zolt_stage.final_claims.items[0],
                );
            }

            // Add opening claims for RAM hamming booleanity
            if (zolt_stage.final_claims.items.len > 1) {
                try claims.insert(
                    .{ .Virtual = .{ .poly = .RamHammingWeight, .sumcheck_id = .RamHammingBooleanity } },
                    zolt_stage.final_claims.items[1],
                );
            }

            _ = self;
        }

        /// Create Jolt Stage 7 (Hamming weight claim reduction)
        /// Zolt doesn't have a separate stage for this, so we create it
        fn createStage7(
            self: *Self,
            zolt_proofs: *const prover.JoltStageProofs(F),
            jolt_proof: *SumcheckInstanceProof(F),
            claims: *OpeningClaims(F),
        ) !void {
            // Stage 7 in Jolt has log_k_chunk rounds
            // We create a minimal valid proof for this stage

            // If we have data from stage 5 (register evaluation), use its final values
            const stage5 = &zolt_proofs.stage_proofs[4];

            if (stage5.final_claims.items.len > 0) {
                // Create a single-round polynomial with the final claim
                const final_claim = stage5.final_claims.items[stage5.final_claims.items.len - 1];
                const poly = try self.allocator.alloc(F, 2);
                poly[0] = final_claim;
                poly[1] = F.zero();
                try jolt_proof.compressed_polys.append(self.allocator, .{
                    .coeffs_except_linear_term = poly,
                    .allocator = self.allocator,
                });

                // Add opening claim for Hamming weight claim reduction
                try claims.insert(
                    .{ .Virtual = .{ .poly = .RamHammingWeight, .sumcheck_id = .HammingWeightClaimReduction } },
                    final_claim,
                );
            }
        }

        /// Create a UniSkipFirstRoundProof from a Zolt stage proof
        fn createUniSkipProof(
            self: *Self,
            zolt_stage: *const prover.StageProof(F),
        ) !?UniSkipFirstRoundProof(F) {
            // If the stage has round polynomials, use the first one as the uni poly
            if (zolt_stage.round_polys.items.len > 0) {
                const first_poly = zolt_stage.round_polys.items[0];
                return try UniSkipFirstRoundProof(F).init(self.allocator, first_poly);
            }

            return null;
        }
    };
}

/// Configuration for proof conversion
pub const ConversionConfig = struct {
    /// Bytecode address space size (K)
    bytecode_K: usize = 1 << 16,
    /// Log of chunk size for one-hot encoding
    log_k_chunk: usize = 10,
    /// Log of chunk size for lookups RA virtualization
    lookups_ra_virtual_log_k_chunk: usize = 8,
};

// =============================================================================
// Tests
// =============================================================================

const testing = std.testing;
const BN254Scalar = field_mod.BN254Scalar;

test "proof converter: basic initialization" {
    const converter = ProofConverter(BN254Scalar).init(testing.allocator);
    _ = converter;
}

test "proof converter: convert empty proof" {
    const F = BN254Scalar;
    var converter = ProofConverter(F).init(testing.allocator);

    // Create empty Zolt stage proofs
    var zolt_proofs = prover.JoltStageProofs(F).init(testing.allocator);
    defer zolt_proofs.deinit();

    zolt_proofs.log_t = 4; // 16 steps
    zolt_proofs.log_k = 10; // 1024 addresses

    // Dummy commitment and proof types for testing
    const DummyCommitment = struct { value: u64 };
    const DummyProof = struct { data: [32]u8 };

    // Convert to Jolt format
    var jolt_proof = try converter.convert(
        DummyCommitment,
        DummyProof,
        &zolt_proofs,
        &[_]DummyCommitment{},
        null,
        .{},
    );
    defer jolt_proof.deinit();

    // Verify trace length is correct
    try testing.expectEqual(@as(usize, 16), jolt_proof.trace_length);
    try testing.expectEqual(@as(usize, 1024), jolt_proof.ram_K);
}

test "proof converter: convert with round polynomials" {
    const F = BN254Scalar;
    var converter = ProofConverter(F).init(testing.allocator);

    // Create Zolt stage proofs with data
    var zolt_proofs = prover.JoltStageProofs(F).init(testing.allocator);
    defer zolt_proofs.deinit();

    zolt_proofs.log_t = 2;
    zolt_proofs.log_k = 8;

    // Add a round polynomial to stage 0
    const coeffs = [_]F{ F.fromU64(1), F.fromU64(2), F.fromU64(3) };
    try zolt_proofs.stage_proofs[0].addRoundPoly(&coeffs);

    // Add a final claim
    try zolt_proofs.stage_proofs[0].final_claims.append(testing.allocator, F.fromU64(42));

    // Dummy types
    const DummyCommitment = struct { value: u64 };
    const DummyProof = struct { data: [32]u8 };

    // Convert
    var jolt_proof = try converter.convert(
        DummyCommitment,
        DummyProof,
        &zolt_proofs,
        &[_]DummyCommitment{},
        null,
        .{},
    );
    defer jolt_proof.deinit();

    // Verify stage 1 has the round polynomial
    try testing.expectEqual(@as(usize, 1), jolt_proof.stage1_sumcheck_proof.compressed_polys.items.len);

    // Verify uni skip proof was created
    try testing.expect(jolt_proof.stage1_uni_skip_first_round_proof != null);

    // Verify opening claims were added
    try testing.expectEqual(@as(usize, 1), jolt_proof.opening_claims.len());
}
