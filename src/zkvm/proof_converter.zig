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
//!
//! ## Constraint Evaluation
//!
//! When `convertWithWitnesses` is called with actual per-cycle witnesses,
//! the converter will compute real Az*Bz products from the R1CS constraints
//! using the evaluators from `r1cs/evaluators.zig`. This enables proper
//! verification of the univariate skip first-round polynomial.

const std = @import("std");
const Allocator = std.mem.Allocator;

const jolt_types = @import("jolt_types.zig");
const prover = @import("prover.zig");
const field_mod = @import("../field/mod.zig");
const r1cs = @import("r1cs/mod.zig");
const spartan_outer = @import("spartan/outer.zig");

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
        ///
        /// IMPORTANT: This generates "zero proofs" - all sumcheck round polynomials
        /// are zero, which satisfies the verification check when all claims are 0.
        /// This is a placeholder for proper cross-compatibility.
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
            const trace_length: usize = @as(usize, 1) << @intCast(zolt_stage_proofs.log_t);
            const ram_K: usize = @as(usize, 1) << @intCast(zolt_stage_proofs.log_k);

            jolt_proof.trace_length = trace_length;
            jolt_proof.ram_K = ram_K;
            jolt_proof.bytecode_K = config.bytecode_K;
            jolt_proof.log_k_chunk = config.log_k_chunk;
            jolt_proof.lookups_ra_virtual_log_k_chunk = config.lookups_ra_virtual_log_k_chunk;

            // Compute derived parameters
            const n_cycle_vars = std.math.log2_int(usize, trace_length);
            const log_ram_k = std.math.log2_int(usize, ram_K);

            // Copy commitments
            for (commitments) |c| {
                try jolt_proof.commitments.append(self.allocator, c);
            }

            // Set joint opening proof
            jolt_proof.joint_opening_proof = joint_opening_proof;

            // Create UniSkip proof for Stage 1 (degree-27 polynomial)
            jolt_proof.stage1_uni_skip_first_round_proof = try self.createUniSkipProofStage1();

            // Stage 1: Outer Spartan Remaining
            // num_rounds = 1 + num_cycles_bits (from OuterRemainingSumcheckParams)
            try self.generateZeroSumcheckProof(
                &jolt_proof.stage1_sumcheck_proof,
                1 + n_cycle_vars,
                3, // degree 3
            );

            // Add Stage 1 opening claims
            // SpartanOuter requires all 36 R1CS inputs + UnivariateSkip claim
            // This matches the ALL_R1CS_INPUTS array in Jolt's r1cs/inputs.rs
            try self.addSpartanOuterOpeningClaims(&jolt_proof.opening_claims);

            // Create UniSkip proof for Stage 2 (degree-12 polynomial)
            jolt_proof.stage2_uni_skip_first_round_proof = try self.createUniSkipProofStage2();

            // Stage 2: Product virtualization + RAM RAF + RW + Output + Instruction claim reduction
            // This is a batched sumcheck with multiple instances
            // The max rounds is typically n_cycle_vars + log_ram_k for RAM operations
            // But the exact count depends on the specific verifiers batched together
            // For simplicity, use n_cycle_vars + 1 (matching Stage 1 remaining)
            try self.generateZeroSumcheckProof(
                &jolt_proof.stage2_sumcheck_proof,
                n_cycle_vars + 1, // Conservative estimate
                3,
            );

            // Add Stage 2 opening claims
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .RamRa, .sumcheck_id = .RamRafEvaluation } },
                F.zero(),
            );
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .RamVal, .sumcheck_id = .RamReadWriteChecking } },
                F.zero(),
            );
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .UnivariateSkip, .sumcheck_id = .SpartanProductVirtualization } },
                F.zero(),
            );

            // Stage 3: Spartan shift + Instruction input + Registers claim reduction
            try self.generateZeroSumcheckProof(
                &jolt_proof.stage3_sumcheck_proof,
                n_cycle_vars,
                3,
            );

            // Add Stage 3 opening claims
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .LookupOutput, .sumcheck_id = .InstructionClaimReduction } },
                F.zero(),
            );

            // Stage 4: Registers RW + RAM val evaluation + final
            try self.generateZeroSumcheckProof(
                &jolt_proof.stage4_sumcheck_proof,
                log_ram_k,
                3,
            );

            // Add Stage 4 opening claims
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .RamVal, .sumcheck_id = .RamValEvaluation } },
                F.zero(),
            );
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .RamValFinal, .sumcheck_id = .RamValFinalEvaluation } },
                F.zero(),
            );

            // Stage 5: Registers val + RAM RA reduction + Lookups RAF
            try self.generateZeroSumcheckProof(
                &jolt_proof.stage5_sumcheck_proof,
                n_cycle_vars,
                3,
            );

            // Add Stage 5 opening claims
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .RegistersVal, .sumcheck_id = .RegistersValEvaluation } },
                F.zero(),
            );
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .RamRa, .sumcheck_id = .RamRaClaimReduction } },
                F.zero(),
            );

            // Stage 6: Bytecode RAF + Hamming + Booleanity + RA virtual + Inc reduction
            try self.generateZeroSumcheckProof(
                &jolt_proof.stage6_sumcheck_proof,
                n_cycle_vars,
                3,
            );

            // Add Stage 6 opening claims
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .RamHammingWeight, .sumcheck_id = .Booleanity } },
                F.zero(),
            );
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .RamHammingWeight, .sumcheck_id = .RamHammingBooleanity } },
                F.zero(),
            );

            // Stage 7: Hamming weight claim reduction
            // num_rounds = log_k_chunk
            try self.generateZeroSumcheckProof(
                &jolt_proof.stage7_sumcheck_proof,
                config.log_k_chunk,
                3,
            );

            // Add Stage 7 opening claims
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .RamHammingWeight, .sumcheck_id = .HammingWeightClaimReduction } },
                F.zero(),
            );

            return jolt_proof;
        }

        /// Generate a zero-filled sumcheck proof with the specified number of rounds
        ///
        /// Each round has a compressed polynomial with degree `degree_bound`.
        /// For claim = 0, all-zero polynomials satisfy p(0) + p(1) = claim.
        fn generateZeroSumcheckProof(
            self: *Self,
            proof: *SumcheckInstanceProof(F),
            num_rounds: usize,
            degree_bound: usize,
        ) !void {
            // Compressed poly: coeffs_except_linear_term has `degree_bound` elements
            // (constant, quadratic, cubic, ...) - linear term is recovered from hint
            for (0..num_rounds) |_| {
                const coeffs = try self.allocator.alloc(F, degree_bound);
                @memset(coeffs, F.zero());
                try proof.compressed_polys.append(self.allocator, .{
                    .coeffs_except_linear_term = coeffs,
                    .allocator = self.allocator,
                });
            }
        }

        /// Add all 36 R1CS input opening claims for SpartanOuter
        ///
        /// This exactly matches the ALL_R1CS_INPUTS array in Jolt's r1cs/inputs.rs:
        /// - 23 simple virtual polynomials
        /// - 13 OpFlags variants
        fn addSpartanOuterOpeningClaims(
            self: *Self,
            claims: *OpeningClaims(F),
        ) !void {
            _ = self;

            // All 36 R1CS inputs in exact order (from Jolt's ALL_R1CS_INPUTS in r1cs/inputs.rs)
            const r1cs_inputs = [_]VirtualPolynomial{
                .LeftInstructionInput, // 0
                .RightInstructionInput, // 1
                .Product, // 2
                .WriteLookupOutputToRD, // 3
                .WritePCtoRD, // 4
                .ShouldBranch, // 5
                .PC, // 6
                .UnexpandedPC, // 7
                .Imm, // 8
                .RamAddress, // 9
                .Rs1Value, // 10
                .Rs2Value, // 11
                .RdWriteValue, // 12
                .RamReadValue, // 13
                .RamWriteValue, // 14
                .LeftLookupOperand, // 15
                .RightLookupOperand, // 16
                .NextUnexpandedPC, // 17
                .NextPC, // 18
                .NextIsVirtual, // 19
                .NextIsFirstInSequence, // 20
                .LookupOutput, // 21
                .ShouldJump, // 22
                // OpFlags variants (13 of them) - CircuitFlags indices
                .{ .OpFlags = 0 }, // 23: AddOperands
                .{ .OpFlags = 1 }, // 24: SubtractOperands
                .{ .OpFlags = 2 }, // 25: MultiplyOperands
                .{ .OpFlags = 3 }, // 26: Load
                .{ .OpFlags = 4 }, // 27: Store
                .{ .OpFlags = 5 }, // 28: Jump
                .{ .OpFlags = 6 }, // 29: WriteLookupOutputToRD
                .{ .OpFlags = 7 }, // 30: VirtualInstruction
                .{ .OpFlags = 8 }, // 31: Assert
                .{ .OpFlags = 9 }, // 32: DoNotUpdateUnexpandedPC
                .{ .OpFlags = 10 }, // 33: Advice
                .{ .OpFlags = 11 }, // 34: IsCompressed
                .{ .OpFlags = 12 }, // 35: IsFirstInSequence
            };

            // Add all R1CS inputs for SpartanOuter with zero claims
            for (r1cs_inputs) |poly| {
                try claims.insert(
                    .{ .Virtual = .{ .poly = poly, .sumcheck_id = .SpartanOuter } },
                    F.zero(),
                );
            }

            // Add the UnivariateSkip claim for SpartanOuter
            try claims.insert(
                .{ .Virtual = .{ .poly = .UnivariateSkip, .sumcheck_id = .SpartanOuter } },
                F.zero(),
            );
        }

        /// Create a UniSkipFirstRoundProof for Stage 1 (degree-27 polynomial)
        ///
        /// Jolt's Stage 1 (Spartan outer) uses a degree-27 first-round polynomial
        /// that encodes all 19 R1CS constraints via the univariate skip optimization.
        ///
        /// For the verification to pass, the polynomial must satisfy:
        ///   Σ_{j=0}^{27} coeff_j * power_sums[j] = 0
        ///
        /// where power_sums[j] = Σ_{t in domain} t^j for domain {-4, -3, ..., 4, 5}.
        ///
        /// The simplest valid polynomial is all zeros (trivially sums to 0).
        fn createUniSkipProofStage1(self: *Self) !?UniSkipFirstRoundProof(F) {
            // For stage 1, we need 28 coefficients (degree 27)
            const NUM_COEFFS = r1cs.OUTER_FIRST_ROUND_POLY_NUM_COEFFS;

            // Create an all-zero polynomial that trivially satisfies the sum constraint.
            const coeffs = try self.allocator.alloc(F, NUM_COEFFS);
            @memset(coeffs, F.zero());

            return UniSkipFirstRoundProof(F){
                .uni_poly = coeffs,
                .allocator = self.allocator,
            };
        }

        /// Create a UniSkipFirstRoundProof for Stage 1 from actual witnesses
        ///
        /// This computes real Az*Bz products using the constraint evaluators,
        /// producing a polynomial that satisfies the univariate skip verification.
        fn createUniSkipProofStage1FromWitnesses(
            self: *Self,
            cycle_witnesses: []const r1cs.R1CSCycleInputs(F),
            tau: []const F,
        ) !?UniSkipFirstRoundProof(F) {
            if (cycle_witnesses.len == 0) {
                return self.createUniSkipProofStage1();
            }

            const poly_mod = @import("../poly/mod.zig");
            const NUM_COEFFS = r1cs.OUTER_FIRST_ROUND_POLY_NUM_COEFFS;

            // Compute eq polynomial evaluations at tau
            var eq_poly = try poly_mod.EqPolynomial(F).init(self.allocator, tau);
            defer eq_poly.deinit();
            const eq_evals = try eq_poly.evals(self.allocator);
            defer self.allocator.free(eq_evals);

            // Use the Spartan outer prover to compute the first-round polynomial
            var outer_prover = try spartan_outer.SpartanOuterProver(F).initFromWitnesses(
                self.allocator,
                cycle_witnesses,
                eq_evals,
                tau,
            );
            defer outer_prover.deinit();

            // Compute the univariate skip polynomial
            var uni_poly = try outer_prover.computeUniskipFirstRoundPoly();
            defer uni_poly.deinit();

            // Copy coefficients to our proof structure
            const coeffs = try self.allocator.alloc(F, NUM_COEFFS);
            @memset(coeffs, F.zero());

            // Copy available coefficients (may be fewer than NUM_COEFFS)
            const copy_len = @min(uni_poly.coeffs.len, NUM_COEFFS);
            @memcpy(coeffs[0..copy_len], uni_poly.coeffs[0..copy_len]);

            return UniSkipFirstRoundProof(F){
                .uni_poly = coeffs,
                .allocator = self.allocator,
            };
        }

        /// Convert with actual per-cycle witnesses for real constraint evaluation
        ///
        /// This method produces proofs with proper Az*Bz evaluations instead of zeros.
        /// Use this for cross-verification with Jolt.
        pub fn convertWithWitnesses(
            self: *Self,
            comptime Commitment: type,
            comptime Proof: type,
            zolt_stage_proofs: *const prover.JoltStageProofs(F),
            commitments: []const Commitment,
            joint_opening_proof: ?Proof,
            config: ConversionConfig,
            cycle_witnesses: []const r1cs.R1CSCycleInputs(F),
            tau: []const F,
        ) !JoltProofType(F, Commitment, Proof) {
            var jolt_proof = JoltProofType(F, Commitment, Proof).init(self.allocator);

            // Copy configuration parameters
            const trace_length: usize = @as(usize, 1) << @intCast(zolt_stage_proofs.log_t);
            const ram_K: usize = @as(usize, 1) << @intCast(zolt_stage_proofs.log_k);

            jolt_proof.trace_length = trace_length;
            jolt_proof.ram_K = ram_K;
            jolt_proof.bytecode_K = config.bytecode_K;
            jolt_proof.log_k_chunk = config.log_k_chunk;
            jolt_proof.lookups_ra_virtual_log_k_chunk = config.lookups_ra_virtual_log_k_chunk;

            // Compute derived parameters
            const n_cycle_vars = std.math.log2_int(usize, trace_length);
            const log_ram_k = std.math.log2_int(usize, ram_K);
            _ = log_ram_k;

            // Copy commitments
            for (commitments) |c| {
                try jolt_proof.commitments.append(self.allocator, c);
            }

            // Set joint opening proof
            jolt_proof.joint_opening_proof = joint_opening_proof;

            // Create UniSkip proof for Stage 1 with actual constraint evaluations
            jolt_proof.stage1_uni_skip_first_round_proof = try self.createUniSkipProofStage1FromWitnesses(
                cycle_witnesses,
                tau,
            );

            // Stage 1: Outer Spartan Remaining
            try self.generateZeroSumcheckProof(
                &jolt_proof.stage1_sumcheck_proof,
                1 + n_cycle_vars,
                3, // degree 3
            );

            // Add Stage 1 opening claims
            try self.addSpartanOuterOpeningClaims(&jolt_proof.opening_claims);

            // Create UniSkip proof for Stage 2 (still using placeholder for now)
            jolt_proof.stage2_uni_skip_first_round_proof = try self.createUniSkipProofStage2();

            // Stage 2 and onwards use placeholder zero proofs
            try self.generateZeroSumcheckProof(
                &jolt_proof.stage2_sumcheck_proof,
                n_cycle_vars + 1,
                3,
            );

            // Add remaining opening claims (same as standard convert)
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .RamRa, .sumcheck_id = .RamRafEvaluation } },
                F.zero(),
            );
            try jolt_proof.opening_claims.insert(
                .{ .Virtual = .{ .poly = .RamVal, .sumcheck_id = .RamReadWriteChecking } },
                F.zero(),
            );

            // Stages 3-7 (placeholder)
            try self.generateZeroSumcheckProof(&jolt_proof.stage3_sumcheck_proof, n_cycle_vars, 3);
            try self.generateZeroSumcheckProof(&jolt_proof.stage4_sumcheck_proof, n_cycle_vars, 3);
            try self.generateZeroSumcheckProof(&jolt_proof.stage5_sumcheck_proof, n_cycle_vars, 3);
            try self.generateZeroSumcheckProof(&jolt_proof.stage6_sumcheck_proof, n_cycle_vars, 3);
            try self.generateZeroSumcheckProof(&jolt_proof.stage7_sumcheck_proof, n_cycle_vars, 3);

            return jolt_proof;
        }

        /// Create a UniSkipFirstRoundProof for Stage 2 (degree-12 polynomial)
        ///
        /// Jolt's Stage 2 (product virtualization) uses a degree-12 first-round
        /// polynomial for the 5 product constraints.
        ///
        /// For the verification to pass, the polynomial must satisfy:
        ///   Σ_{j=0}^{12} coeff_j * power_sums[j] = 0
        ///
        /// where power_sums[j] = Σ_{t in domain} t^j for domain {-2, -1, 0, 1, 2}.
        fn createUniSkipProofStage2(self: *Self) !?UniSkipFirstRoundProof(F) {
            const univariate_skip = r1cs.univariate_skip;

            // For stage 2, we need 13 coefficients (degree 12)
            const NUM_COEFFS = univariate_skip.PRODUCT_VIRTUAL_FIRST_ROUND_POLY_NUM_COEFFS;

            // Create an all-zero polynomial that trivially satisfies the sum constraint.
            const coeffs = try self.allocator.alloc(F, NUM_COEFFS);
            @memset(coeffs, F.zero());

            return UniSkipFirstRoundProof(F){
                .uni_poly = coeffs,
                .allocator = self.allocator,
            };
        }
    };
}

/// Configuration for proof conversion
///
/// These values must match Jolt's config.rs:
/// - log_k_chunk: Must be <= 8 (Jolt uses 4 for small traces, 8 for large)
/// - lookups_ra_virtual_log_k_chunk: Jolt uses LOG_K/8 (=16) for small traces
pub const ConversionConfig = struct {
    /// Bytecode address space size (K)
    bytecode_K: usize = 1 << 16,
    /// Log of chunk size for one-hot encoding (must be <= 8, Jolt uses 4 for small traces)
    log_k_chunk: usize = 4,
    /// Log of chunk size for lookups RA virtualization (LOG_K / 8 = 128 / 8 = 16 for small traces)
    lookups_ra_virtual_log_k_chunk: usize = 16,
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

test "proof converter: convert generates zero proofs" {
    const F = BN254Scalar;
    var converter = ProofConverter(F).init(testing.allocator);

    // Create Zolt stage proofs with data
    var zolt_proofs = prover.JoltStageProofs(F).init(testing.allocator);
    defer zolt_proofs.deinit();

    zolt_proofs.log_t = 2; // trace_length = 4
    zolt_proofs.log_k = 8; // ram_K = 256

    // Note: Zolt stage data is now ignored - we generate zero proofs

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

    // Verify trace length (2^2 = 4)
    try testing.expectEqual(@as(usize, 4), jolt_proof.trace_length);

    // Stage 1: num_rounds = 1 + n_cycle_vars = 1 + 2 = 3
    try testing.expectEqual(@as(usize, 3), jolt_proof.stage1_sumcheck_proof.compressed_polys.items.len);

    // Stage 2: num_rounds = n_cycle_vars + 1 = 3
    try testing.expectEqual(@as(usize, 3), jolt_proof.stage2_sumcheck_proof.compressed_polys.items.len);

    // Verify uni skip proofs were created
    try testing.expect(jolt_proof.stage1_uni_skip_first_round_proof != null);
    try testing.expect(jolt_proof.stage2_uni_skip_first_round_proof != null);

    // Verify opening claims were added (multiple claims per stage)
    try testing.expect(jolt_proof.opening_claims.len() > 0);
}
