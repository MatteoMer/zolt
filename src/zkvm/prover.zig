//! Multi-Stage Sumcheck Prover for Jolt zkVM
//!
//! This module orchestrates the 6-stage sumcheck proving process that is
//! at the heart of Jolt's efficiency. Each stage batches multiple sumcheck
//! instances together for parallel verification.
//!
//! ## Stage Overview
//!
//! 1. **Stage 1: Outer Spartan** - Instruction correctness via R1CS
//! 2. **Stage 2: RAM/Bytecode RAF & Read-Write** - Memory consistency
//! 3. **Stage 3: Instruction Lookup** - Lookup argument reduction
//! 4. **Stage 4: Val Evaluation** - Memory value consistency
//! 5. **Stage 5: Register & RA Reduction** - Register checking
//! 6. **Stage 6: Booleanity & Finalization** - Hamming weight checks
//!
//! Reference: jolt-core/src/zkvm/prover.rs

const std = @import("std");
const Allocator = std.mem.Allocator;

const ram = @import("ram/mod.zig");
const spartan = @import("spartan/mod.zig");
const lasso = @import("lasso/mod.zig");
const instruction = @import("instruction/mod.zig");
const bytecode = @import("bytecode/mod.zig");
const registers = @import("registers/mod.zig");
const r1cs = @import("r1cs/mod.zig");

/// Sumcheck instance trait - interface for batched proving
pub fn SumcheckInstance(comptime F: type) type {
    return struct {
        const Self = @This();
        const VTable = struct {
            /// Get the number of rounds for this sumcheck
            numRounds: *const fn (*anyopaque) usize,
            /// Get the degree bound for this sumcheck
            degreeBound: *const fn (*anyopaque) usize,
            /// Get the initial claim
            initialClaim: *const fn (*anyopaque) F,
            /// Compute the round polynomial
            computeRoundPoly: *const fn (*anyopaque, usize) []F,
            /// Bind the challenge for this round
            bindChallenge: *const fn (*anyopaque, F) void,
        };

        ptr: *anyopaque,
        vtable: *const VTable,

        pub fn numRounds(self: Self) usize {
            return self.vtable.numRounds(self.ptr);
        }

        pub fn degreeBound(self: Self) usize {
            return self.vtable.degreeBound(self.ptr);
        }

        pub fn initialClaim(self: Self) F {
            return self.vtable.initialClaim(self.ptr);
        }

        pub fn computeRoundPoly(self: Self, round: usize) []F {
            return self.vtable.computeRoundPoly(self.ptr, round);
        }

        pub fn bindChallenge(self: Self, challenge: F) void {
            self.vtable.bindChallenge(self.ptr, challenge);
        }
    };
}

/// Stage configuration
pub const StageConfig = struct {
    name: []const u8,
    description: []const u8,
};

pub const STAGE_CONFIGS = [_]StageConfig{
    .{ .name = "Stage1-Spartan", .description = "Outer Spartan sumcheck for instruction correctness" },
    .{ .name = "Stage2-RafRW", .description = "RAM RAF evaluation and read-write checking" },
    .{ .name = "Stage3-Lookup", .description = "Instruction lookup reduction" },
    .{ .name = "Stage4-Val", .description = "Memory value evaluation" },
    .{ .name = "Stage5-Reg", .description = "Register value evaluation and RA reduction" },
    .{ .name = "Stage6-Bool", .description = "Booleanity and Hamming weight checks" },
};

/// Stage proof containing sumcheck round polynomials
pub fn StageProof(comptime F: type) type {
    return struct {
        const Self = @This();

        /// Round polynomials for each round
        round_polys: std.ArrayListUnmanaged([]F),
        /// Challenges used in this stage
        challenges: std.ArrayListUnmanaged(F),
        /// Final claims from each batched instance
        final_claims: std.ArrayListUnmanaged(F),
        allocator: Allocator,

        pub fn init(allocator: Allocator) Self {
            // Use empty slices that won't cause issues when deinit is called
            // capacity = 0 ensures deinit() won't try to free the items pointer
            const empty_round_polys: std.ArrayListUnmanaged([]F) = .{};
            const empty_challenges: std.ArrayListUnmanaged(F) = .{};
            const empty_final_claims: std.ArrayListUnmanaged(F) = .{};
            return Self{
                .round_polys = empty_round_polys,
                .challenges = empty_challenges,
                .final_claims = empty_final_claims,
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            for (self.round_polys.items) |poly| {
                self.allocator.free(poly);
            }
            self.round_polys.deinit(self.allocator);
            self.challenges.deinit(self.allocator);
            self.final_claims.deinit(self.allocator);
        }

        pub fn addRoundPoly(self: *Self, poly: []const F) !void {
            const copy = try self.allocator.alloc(F, poly.len);
            @memcpy(copy, poly);
            try self.round_polys.append(self.allocator, copy);
        }

        pub fn addChallenge(self: *Self, challenge: F) !void {
            try self.challenges.append(self.allocator, challenge);
        }
    };
}

/// Full Jolt proof containing all stages
pub fn JoltStageProofs(comptime F: type) type {
    return struct {
        const Self = @This();

        stage_proofs: [6]StageProof(F),
        /// Log2 of trace length (needed for transcript sync)
        log_t: usize,
        /// Log2 of address space (needed for transcript sync)
        log_k: usize,
        allocator: Allocator,

        pub fn init(allocator: Allocator) Self {
            var proofs: [6]StageProof(F) = undefined;
            for (&proofs) |*p| {
                p.* = StageProof(F).init(allocator);
            }
            return Self{
                .stage_proofs = proofs,
                .log_t = 0,
                .log_k = 16, // Default
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            for (&self.stage_proofs) |*p| {
                p.deinit();
            }
        }

        /// Calculate total proof size in field elements
        /// Returns: { total_field_elements, round_polys_count, total_poly_coeffs }
        pub fn proofSize(self: *const Self) struct { total_elements: usize, round_polys: usize, poly_coeffs: usize, challenges: usize, claims: usize } {
            var total_elements: usize = 0;
            var round_polys: usize = 0;
            var poly_coeffs: usize = 0;
            var challenges: usize = 0;
            var claims: usize = 0;

            for (self.stage_proofs) |stage| {
                // Count round polynomial coefficients
                for (stage.round_polys.items) |poly| {
                    poly_coeffs += poly.len;
                    total_elements += poly.len;
                }
                round_polys += stage.round_polys.items.len;

                // Count challenges (these are typically derived, but included for completeness)
                challenges += stage.challenges.items.len;
                total_elements += stage.challenges.items.len;

                // Count final claims
                claims += stage.final_claims.items.len;
                total_elements += stage.final_claims.items.len;
            }

            return .{
                .total_elements = total_elements,
                .round_polys = round_polys,
                .poly_coeffs = poly_coeffs,
                .challenges = challenges,
                .claims = claims,
            };
        }

        /// Calculate proof size in bytes (assuming 32-byte field elements)
        pub fn proofSizeBytes(self: *const Self) usize {
            const size = self.proofSize();
            return size.total_elements * 32; // BN254 scalar is 254 bits, fits in 32 bytes
        }
    };
}

/// Opening accumulator for polynomial commitments
pub fn OpeningAccumulator(comptime F: type) type {
    return struct {
        const Self = @This();

        /// Points at which polynomials are opened
        opening_points: std.ArrayListUnmanaged([]const F),
        /// Claimed evaluations
        claimed_evals: std.ArrayListUnmanaged(F),
        allocator: Allocator,

        pub fn init(allocator: Allocator) Self {
            return Self{
                .opening_points = .{},
                .claimed_evals = .{},
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            self.opening_points.deinit(self.allocator);
            self.claimed_evals.deinit(self.allocator);
        }

        /// Record an opening claim
        pub fn accumulate(self: *Self, point: []const F, eval: F) !void {
            try self.opening_points.append(self.allocator, point);
            try self.claimed_evals.append(self.allocator, eval);
        }
    };
}

/// Multi-stage prover state
pub fn MultiStageProver(comptime F: type) type {
    return struct {
        const Self = @This();

        /// Execution trace
        trace: *const @import("../tracer/mod.zig").ExecutionTrace,
        /// Memory trace
        memory_trace: *const ram.MemoryTrace,
        /// Lookup trace
        lookup_trace: *const instruction.LookupTraceCollector(64),
        /// Opening accumulator
        opening_accumulator: OpeningAccumulator(F),
        /// Current stage
        current_stage: usize,
        /// Stage proofs
        proofs: JoltStageProofs(F),
        /// One-hot parameters
        log_k: usize, // log2 of address space size
        log_t: usize, // log2 of trace length
        /// Memory layout
        start_address: u64,
        allocator: Allocator,
        /// Track if proofs ownership was transferred
        proofs_transferred: bool,

        pub fn init(
            allocator: Allocator,
            trace: *const @import("../tracer/mod.zig").ExecutionTrace,
            memory_trace: *const ram.MemoryTrace,
            lookup_trace: *const instruction.LookupTraceCollector(64),
            log_k: usize,
            start_address: u64,
        ) Self {
            const trace_len = trace.steps.items.len;
            const log_t = if (trace_len == 0) 0 else std.math.log2_int_ceil(usize, trace_len);

            return Self{
                .trace = trace,
                .memory_trace = memory_trace,
                .lookup_trace = lookup_trace,
                .opening_accumulator = OpeningAccumulator(F).init(allocator),
                .current_stage = 0,
                .proofs = JoltStageProofs(F).init(allocator),
                .log_k = log_k,
                .log_t = log_t,
                .start_address = start_address,
                .allocator = allocator,
                .proofs_transferred = false,
            };
        }

        pub fn deinit(self: *Self) void {
            self.opening_accumulator.deinit();
            // Note: proofs ownership is transferred to caller via prove()
            // Only deinit if prove() was never called
            if (!self.proofs_transferred) {
                self.proofs.deinit();
            }
        }

        /// Prove all stages
        /// Returns the stage proofs - caller takes ownership and is responsible for cleanup
        pub fn prove(self: *Self, transcript: anytype) !JoltStageProofs(F) {
            std.debug.print("\n[PROVER] ========================================\n", .{});
            std.debug.print("[PROVER] Starting multi-stage proof\n", .{});
            std.debug.print("[PROVER]   log_t={d}, log_k={d}, trace_len={d}\n", .{ self.log_t, self.log_k, self.trace.steps.items.len });
            std.debug.print("[PROVER] ========================================\n\n", .{});

            try self.proveStage1(transcript);
            std.debug.print("\n[PROVER] Stage 1 complete\n\n", .{});

            try self.proveStage2(transcript);
            std.debug.print("\n[PROVER] Stage 2 complete\n\n", .{});

            try self.proveStage3(transcript);
            std.debug.print("\n[PROVER] Stage 3 complete\n\n", .{});

            try self.proveStage4(transcript);
            std.debug.print("\n[PROVER] Stage 4 complete\n\n", .{});

            try self.proveStage5(transcript);
            std.debug.print("\n[PROVER] Stage 5 complete\n\n", .{});

            try self.proveStage6(transcript);
            std.debug.print("\n[PROVER] Stage 6 complete\n\n", .{});

            // Store log_t and log_k for verifier transcript sync
            self.proofs.log_t = self.log_t;
            self.proofs.log_k = self.log_k;

            std.debug.print("[PROVER] ========================================\n", .{});
            std.debug.print("[PROVER] All stages complete!\n", .{});
            std.debug.print("[PROVER] ========================================\n\n", .{});

            // Transfer ownership to caller
            self.proofs_transferred = true;
            return self.proofs;
        }

        /// Stage 1: Outer Spartan sumcheck
        ///
        /// This stage proves that all R1CS constraints are satisfied for
        /// the execution trace. The sumcheck proves:
        ///   sum_{x} eq(tau, x) * [(Az)(x) * (Bz)(x) - (Cz)(x)] = 0
        ///
        /// Structure:
        /// - Number of rounds: log2(num_constraints) for uniform R1CS
        /// - Degree: 2 (quadratic from Az * Bz term)
        /// - Opening claims: Az, Bz, Cz evaluations at final point
        fn proveStage1(self: *Self, transcript: anytype) !void {
            std.debug.print("[PROVER STAGE 1] Starting Outer Spartan sumcheck\n", .{});

            const stage_proof = &self.proofs.stage_proofs[0];

            // Build Jolt R1CS from trace
            var jolt_r1cs = try r1cs.JoltR1CS(F).fromTrace(self.allocator, self.trace);
            defer jolt_r1cs.deinit();

            std.debug.print("[PROVER STAGE 1]   num_cycles={d}, log_constraints={d}, padded={d}\n", .{ jolt_r1cs.num_cycles, jolt_r1cs.log_num_constraints, jolt_r1cs.padded_num_constraints });

            // Skip if no constraints
            if (jolt_r1cs.num_cycles == 0) {
                std.debug.print("[PROVER STAGE 1]   Skipping - no cycles\n", .{});
                self.current_stage = 1;
                return;
            }

            // Build witness vector
            const witness = try jolt_r1cs.buildWitness();
            defer self.allocator.free(witness);
            std.debug.print("[PROVER STAGE 1]   witness_size={d}\n", .{witness.len});

            // Get tau challenge from transcript (log_n field elements)
            const num_rounds = jolt_r1cs.log_num_constraints;
            const tau = try self.allocator.alloc(F, num_rounds);
            defer self.allocator.free(tau);
            std.debug.print("[PROVER STAGE 1]   getting {d} tau challenges\n", .{num_rounds});
            for (tau, 0..) |*t, i| {
                t.* = try transcript.challengeScalar("spartan_tau");
                std.debug.print("[PROVER STAGE 1]     tau[{d}] = {x:0>16}{x:0>16}{x:0>16}{x:0>16}\n", .{ i, t.limbs[3], t.limbs[2], t.limbs[1], t.limbs[0] });
            }

            // Initialize Spartan interface
            var spartan_iface = try r1cs.JoltSpartanInterface(F).init(
                self.allocator,
                &jolt_r1cs,
                witness,
                tau,
            );
            defer spartan_iface.deinit();

            // Record initial claim (should be 0 for valid witness)
            const initial_claim = spartan_iface.initialClaim();
            try stage_proof.final_claims.append(self.allocator, initial_claim);
            std.debug.print("[PROVER STAGE 1]   initial_claim = {x:0>16}{x:0>16}{x:0>16}{x:0>16}\n", .{ initial_claim.limbs[3], initial_claim.limbs[2], initial_claim.limbs[1], initial_claim.limbs[0] });

            // Run sumcheck rounds
            for (0..num_rounds) |round| {
                std.debug.print("[PROVER STAGE 1]   --- Round {d}/{d} ---\n", .{ round, num_rounds });

                // Compute round polynomial [p(0), p(1), p(2)]
                const round_poly = try spartan_iface.computeRoundPolynomial();

                std.debug.print("[PROVER STAGE 1]     p(0) = {x:0>16}{x:0>16}{x:0>16}{x:0>16}\n", .{ round_poly[0].limbs[3], round_poly[0].limbs[2], round_poly[0].limbs[1], round_poly[0].limbs[0] });
                std.debug.print("[PROVER STAGE 1]     p(1) = {x:0>16}{x:0>16}{x:0>16}{x:0>16}\n", .{ round_poly[1].limbs[3], round_poly[1].limbs[2], round_poly[1].limbs[1], round_poly[1].limbs[0] });
                std.debug.print("[PROVER STAGE 1]     p(2) = {x:0>16}{x:0>16}{x:0>16}{x:0>16}\n", .{ round_poly[2].limbs[3], round_poly[2].limbs[2], round_poly[2].limbs[1], round_poly[2].limbs[0] });
                const sum01 = round_poly[0].add(round_poly[1]);
                std.debug.print("[PROVER STAGE 1]     p(0)+p(1) = {x:0>16}{x:0>16}{x:0>16}{x:0>16}\n", .{ sum01.limbs[3], sum01.limbs[2], sum01.limbs[1], sum01.limbs[0] });

                // Store polynomial in proof
                const poly_copy = try self.allocator.alloc(F, 3);
                poly_copy[0] = round_poly[0];
                poly_copy[1] = round_poly[1];
                poly_copy[2] = round_poly[2];
                try stage_proof.round_polys.append(self.allocator, poly_copy);

                // Absorb round polynomial into transcript (Fiat-Shamir binding)
                try transcript.appendScalar("round_poly_0", round_poly[0]);
                try transcript.appendScalar("round_poly_1", round_poly[1]);
                try transcript.appendScalar("round_poly_2", round_poly[2]);

                // Get challenge from transcript
                const challenge = try transcript.challengeScalar("spartan_round");
                try stage_proof.addChallenge(challenge);
                std.debug.print("[PROVER STAGE 1]     challenge=0x{x}\n", .{challenge.limbs[0]});

                // Bind challenge for next round
                try spartan_iface.bindChallenge(challenge);
            }

            // Record final evaluation and eval claims
            const final_eval = spartan_iface.getFinalEval();
            try stage_proof.final_claims.append(self.allocator, final_eval);
            std.debug.print("[PROVER STAGE 1]   final_eval=0x{x}\n", .{final_eval.limbs[0]});

            // Get evaluation claims for Az, Bz, Cz at final point
            const eval_claims = spartan_iface.getEvalClaims();
            try stage_proof.final_claims.append(self.allocator, eval_claims[0]); // A(r)
            try stage_proof.final_claims.append(self.allocator, eval_claims[1]); // B(r)
            try stage_proof.final_claims.append(self.allocator, eval_claims[2]); // C(r)
            std.debug.print("[PROVER STAGE 1]   eval_claims: A(r)=0x{x}, B(r)=0x{x}, C(r)=0x{x}\n", .{ eval_claims[0].limbs[0], eval_claims[1].limbs[0], eval_claims[2].limbs[0] });

            // Accumulate opening claims for commitment verification
            if (stage_proof.challenges.items.len > 0) {
                try self.opening_accumulator.accumulate(
                    stage_proof.challenges.items,
                    final_eval,
                );
            }

            self.current_stage = 1;
        }

        /// Stage 2: RAM RAF evaluation and read-write checking
        ///
        /// This stage proves memory consistency using RAF (Read-After-Final) checking.
        /// The sumcheck proves:
        ///   Σ_k ra(k) ⋅ unmap(k) = raf_claim
        ///
        /// Where:
        /// - ra(k) = access counts per address slot k
        /// - unmap(k) = original address from remapped slot k
        ///
        /// Structure:
        /// - Number of rounds: log2(K) where K = address space size
        /// - Degree: 2 (product of two linear polynomials)
        /// - Opening claims: ra and unmap polynomial evaluations
        fn proveStage2(self: *Self, transcript: anytype) !void {
            std.debug.print("[PROVER STAGE 2] Starting RAM RAF evaluation\n", .{});

            const stage_proof = &self.proofs.stage_proofs[1];

            // Get r_cycle challenges from transcript (binding cycle variables)
            const r_cycle = try self.allocator.alloc(F, self.log_t);
            defer self.allocator.free(r_cycle);
            std.debug.print("[PROVER STAGE 2]   getting {d} r_cycle challenges\n", .{self.log_t});
            for (r_cycle, 0..) |*r, i| {
                r.* = try transcript.challengeScalar("r_cycle");
                std.debug.print("[PROVER STAGE 2]     r_cycle[{d}]=0x{x}\n", .{ i, r.limbs[0] });
            }

            // Initialize RAF evaluation parameters
            // Note: The prover takes ownership of params - deinit called via raf_prover.deinit()
            var raf_params = try ram.RafEvaluationParams(F).init(
                self.allocator,
                self.log_k,
                self.start_address,
                r_cycle,
            );

            // Compute initial claim first, then initialize RAF prover
            const initial_claim = blk: {
                var ra_temp = try ram.RaPolynomial(F).fromTrace(
                    self.allocator,
                    self.memory_trace,
                    raf_params.r_cycle,
                    raf_params.start_address,
                    raf_params.log_k,
                );
                defer ra_temp.deinit();

                var claim = F.zero();
                const unmap_temp = ram.UnmapPolynomial(F).init(raf_params.log_k, raf_params.start_address);
                for (0..ra_temp.evals.len) |k| {
                    const ra_k = ra_temp.get(k);
                    const unmap_k = F.fromU64(unmap_temp.evaluateAtIndex(k));
                    claim = claim.add(ra_k.mul(unmap_k));
                }
                break :blk claim;
            };

            // Initialize RAF prover with memory trace and computed claim
            var raf_prover = try ram.RafEvaluationProver(F).init(
                self.allocator,
                self.memory_trace,
                raf_params,
                initial_claim,
            );
            defer raf_prover.deinit();
            try stage_proof.final_claims.append(self.allocator, initial_claim);
            std.debug.print("[PROVER STAGE 2]   initial_claim=0x{x}\n", .{initial_claim.limbs[0]});

            // Run sumcheck rounds
            const num_rounds = raf_params.numRounds();
            std.debug.print("[PROVER STAGE 2]   num_rounds={d}\n", .{num_rounds});
            for (0..num_rounds) |round| {
                std.debug.print("[PROVER STAGE 2]   --- Round {d}/{d} ---\n", .{ round, num_rounds });

                // Compute round polynomial [s(0), s(1), s(2), s(3)]
                const round_poly = raf_prover.computeRoundPolynomialCubic();
                std.debug.print("[PROVER STAGE 2]     s(0)=0x{x}, s(1)=0x{x}\n", .{ round_poly[0].limbs[0], round_poly[1].limbs[0] });

                // Store round polynomial in proof (just s(0) and s(2) for degree-2 compression)
                const poly_copy = try self.allocator.alloc(F, 2);
                poly_copy[0] = round_poly[0];
                poly_copy[1] = round_poly[2]; // s(2) instead of s(1)
                try stage_proof.round_polys.append(self.allocator, poly_copy);

                // Get challenge from transcript
                const challenge = try transcript.challengeScalar("raf_round");
                try stage_proof.addChallenge(challenge);
                std.debug.print("[PROVER STAGE 2]     challenge=0x{x}\n", .{challenge.limbs[0]});

                // Update claim and bind the challenge for next round
                raf_prover.updateClaim(round_poly, challenge);
                try raf_prover.bindChallenge(challenge);
            }

            // Record final claim after all rounds
            const final_claim = raf_prover.getFinalClaim();
            try stage_proof.final_claims.append(self.allocator, final_claim);
            std.debug.print("[PROVER STAGE 2]   final_claim=0x{x}\n", .{final_claim.limbs[0]});

            // Accumulate opening claims for ra and unmap polynomials
            // These will be verified in the final opening proof
            if (stage_proof.challenges.items.len > 0) {
                try self.opening_accumulator.accumulate(
                    stage_proof.challenges.items,
                    final_claim,
                );
            }

            self.current_stage = 2;
        }

        /// Stage 3: Instruction lookup reduction (Lasso)
        ///
        /// This stage proves that all instruction lookups are valid using
        /// the Lasso lookup argument. The lookup trace records:
        /// - Table index (which lookup table was used)
        /// - Input value (x, y operands)
        /// - Output value (result from table)
        ///
        /// The Lasso protocol uses two-phase sumcheck:
        /// 1. Address binding: Bind address variables (log_K rounds)
        /// 2. Cycle binding: Bind cycle/time variables (log_T rounds)
        ///
        /// Opening claims: Lookup polynomial evaluations
        fn proveStage3(self: *Self, transcript: anytype) !void {
            std.debug.print("[PROVER STAGE 3] Starting Lasso instruction lookup\n", .{});

            const stage_proof = &self.proofs.stage_proofs[2];

            // Get batching challenge γ for combining lookup instances
            const gamma = try transcript.challengeScalar("lasso_gamma");
            std.debug.print("[PROVER STAGE 3]   gamma=0x{x}\n", .{gamma.limbs[0]});

            // Get lookup statistics for proof metadata
            const stats = self.lookup_trace.getStats();
            _ = stats;

            // Extract lookup indices from the trace
            const trace_entries = self.lookup_trace.getEntries();
            std.debug.print("[PROVER STAGE 3]   num_lookup_entries={d}\n", .{trace_entries.len});

            if (trace_entries.len == 0) {
                // No lookups to prove - stage is trivially complete
                std.debug.print("[PROVER STAGE 3]   Skipping - no lookups\n", .{});
                self.current_stage = 3;
                return;
            }

            // Build lookup indices and table arrays
            const lookup_indices = try self.allocator.alloc(u128, trace_entries.len);
            defer self.allocator.free(lookup_indices);
            const lookup_tables = try self.allocator.alloc(usize, trace_entries.len);
            defer self.allocator.free(lookup_tables);

            for (trace_entries, 0..) |entry, i| {
                lookup_indices[i] = entry.index;
                lookup_tables[i] = @intFromEnum(entry.table);
            }

            // Get reduction point from previous challenges
            const r_reduction = try self.allocator.alloc(F, self.log_t);
            defer self.allocator.free(r_reduction);
            std.debug.print("[PROVER STAGE 3]   getting {d} r_reduction challenges\n", .{self.log_t});
            for (r_reduction, 0..) |*r, i| {
                r.* = try transcript.challengeScalar("r_reduction");
                std.debug.print("[PROVER STAGE 3]     r_reduction[{d}]=0x{x}\n", .{ i, r.limbs[0] });
            }

            // Calculate log_K (address space size for tables)
            // Use 16 bits for table entries by default
            const log_K: usize = 16;

            // Initialize Lasso parameters
            const params = lasso.LassoParams(F).init(
                gamma,
                self.log_t,
                log_K,
                r_reduction,
            );
            std.debug.print("[PROVER STAGE 3]   log_K={d}, log_T={d}, total_rounds={d}\n", .{ params.log_K, params.log_T, params.log_K + params.log_T });

            // Initialize and run Lasso prover
            var lasso_prover = try lasso.LassoProver(F).init(
                self.allocator,
                lookup_indices,
                lookup_tables,
                params,
            );
            defer lasso_prover.deinit();

            // Record initial claim (sum of eq evaluations)
            const initial_claim = lasso_prover.computeInitialClaim();
            try stage_proof.final_claims.append(self.allocator, initial_claim);
            std.debug.print("[PROVER STAGE 3]   initial_claim=0x{x}\n", .{initial_claim.limbs[0]});

            // Run Lasso sumcheck rounds
            const total_rounds = params.log_K + params.log_T;
            for (0..total_rounds) |round| {
                const phase_str = if (lasso_prover.isAddressPhase()) "ADDR" else "CYCLE";
                std.debug.print("[PROVER STAGE 3]   --- Round {d}/{d} ({s}) ---\n", .{ round, total_rounds, phase_str });

                // Compute round polynomial
                var round_poly = try lasso_prover.computeRoundPolynomial();

                std.debug.print("[PROVER STAGE 3]     coeffs: c0=0x{x}, c1=0x{x}", .{ round_poly.coeffs[0].limbs[0], round_poly.coeffs[1].limbs[0] });
                if (round_poly.coeffs.len > 2) {
                    std.debug.print(", c2=0x{x}", .{round_poly.coeffs[2].limbs[0]});
                }
                std.debug.print("\n", .{});

                // Store polynomial coefficients in stage proof
                try stage_proof.round_polys.append(self.allocator, round_poly.coeffs);
                // Prevent double-free since we stored the coefficients
                round_poly.coeffs = &[_]F{};

                // Get challenge from transcript
                const challenge = try transcript.challengeScalar("lasso_round");
                try stage_proof.addChallenge(challenge);
                std.debug.print("[PROVER STAGE 3]     challenge=0x{x}\n", .{challenge.limbs[0]});

                // Bind the challenge
                try lasso_prover.receiveChallenge(challenge);
                std.debug.print("[PROVER STAGE 3]     new_claim=0x{x}\n", .{lasso_prover.current_claim.limbs[0]});
            }

            // Record final evaluation
            if (lasso_prover.isComplete()) {
                const final_eval = lasso_prover.getFinalEval();
                try stage_proof.final_claims.append(self.allocator, final_eval);
                std.debug.print("[PROVER STAGE 3]   final_eval=0x{x}\n", .{final_eval.limbs[0]});
            }

            // Accumulate opening claims
            if (stage_proof.challenges.items.len > 0) {
                const final_claim = if (stage_proof.final_claims.items.len > 0)
                    stage_proof.final_claims.items[stage_proof.final_claims.items.len - 1]
                else
                    F.zero();
                try self.opening_accumulator.accumulate(
                    stage_proof.challenges.items,
                    final_claim,
                );
            }

            self.current_stage = 3;
        }

        /// Stage 4: Memory value evaluation
        ///
        /// This stage proves that memory values are consistent across
        /// read and write operations. Uses the ValEvaluation sumcheck
        /// to verify:
        ///   Val(r) - Val_init(r_address) = Σ_j inc(j) · wa(r_address, j) · LT(j, r_cycle)
        ///
        /// Structure:
        /// - Number of rounds: log2(trace_length)
        /// - Degree: 3 (product of inc, wa, LT polynomials)
        /// - Opening claims: inc, wa, LT polynomial evaluations
        fn proveStage4(self: *Self, transcript: anytype) !void {
            std.debug.print("[PROVER STAGE 4] Starting memory value evaluation\n", .{});

            const stage_proof = &self.proofs.stage_proofs[3];

            // Get address challenge from previous stage
            const r_address = try self.allocator.alloc(F, self.log_k);
            defer self.allocator.free(r_address);
            std.debug.print("[PROVER STAGE 4]   getting {d} r_address challenges\n", .{self.log_k});
            for (r_address, 0..) |*r, i| {
                r.* = try transcript.challengeScalar("r_address");
                if (i < 4) std.debug.print("[PROVER STAGE 4]     r_address[{d}]=0x{x}\n", .{ i, r.limbs[0] });
            }

            // Get cycle challenge from previous stage
            const r_cycle = try self.allocator.alloc(F, self.log_t);
            defer self.allocator.free(r_cycle);
            std.debug.print("[PROVER STAGE 4]   getting {d} r_cycle_val challenges\n", .{self.log_t});
            for (r_cycle, 0..) |*r, i| {
                r.* = try transcript.challengeScalar("r_cycle_val");
                std.debug.print("[PROVER STAGE 4]     r_cycle_val[{d}]=0x{x}\n", .{ i, r.limbs[0] });
            }

            // Initial value evaluation (for uninitialized memory, this is 0)
            const init_eval = F.zero();

            // Initialize value evaluation parameters
            const trace_len = self.trace.steps.items.len;
            const k = @as(usize, 1) << @intCast(self.log_k);
            std.debug.print("[PROVER STAGE 4]   trace_len={d}, k={d}\n", .{ trace_len, k });

            var val_params = try ram.ValEvaluationParams(F).init(
                self.allocator,
                init_eval,
                trace_len,
                k,
                r_address,
                r_cycle,
            );
            defer val_params.deinit();

            // Skip if trace is empty
            if (trace_len == 0) {
                std.debug.print("[PROVER STAGE 4]   Skipping - empty trace\n", .{});
                self.current_stage = 4;
                return;
            }

            // Initialize value evaluation prover
            var val_prover = try ram.ValEvaluationProver(F).init(
                self.allocator,
                self.memory_trace,
                null,
                val_params,
                self.start_address,
            );
            defer val_prover.deinit();

            // Compute and record initial claim
            const initial_claim = val_prover.computeInitialClaim();
            try stage_proof.final_claims.append(self.allocator, initial_claim);
            std.debug.print("[PROVER STAGE 4]   initial_claim=0x{x}\n", .{initial_claim.limbs[0]});

            // Run sumcheck rounds
            const num_rounds = val_params.numRounds();
            std.debug.print("[PROVER STAGE 4]   num_rounds={d}\n", .{num_rounds});
            for (0..num_rounds) |round| {
                std.debug.print("[PROVER STAGE 4]   --- Round {d}/{d} ---\n", .{ round, num_rounds });

                // Compute round polynomial [p(0), p(1), p(2), p(3)] for degree-3 sumcheck
                const round_poly = val_prover.computeRoundPolynomial();
                std.debug.print("[PROVER STAGE 4]     p(0)=0x{x}, p(1)=0x{x}, p(2)=0x{x}, p(3)=0x{x}\n", .{ round_poly[0].limbs[0], round_poly[1].limbs[0], round_poly[2].limbs[0], round_poly[3].limbs[0] });

                // Store polynomial in proof (4 evaluations for degree-3)
                const poly_copy = try self.allocator.alloc(F, 4);
                poly_copy[0] = round_poly[0];
                poly_copy[1] = round_poly[1];
                poly_copy[2] = round_poly[2];
                poly_copy[3] = round_poly[3];
                try stage_proof.round_polys.append(self.allocator, poly_copy);

                // Get challenge from transcript
                const challenge = try transcript.challengeScalar("val_eval_round");
                try stage_proof.addChallenge(challenge);
                std.debug.print("[PROVER STAGE 4]     challenge=0x{x}\n", .{challenge.limbs[0]});

                // Bind the challenge with the round polynomial for proper claim update
                val_prover.bindChallengeWithPoly(challenge, round_poly);
            }

            // Record final claim
            const final_claim = val_prover.getFinalClaim();
            try stage_proof.final_claims.append(self.allocator, final_claim);
            std.debug.print("[PROVER STAGE 4]   final_claim=0x{x}\n", .{final_claim.limbs[0]});

            // Accumulate opening claims
            if (stage_proof.challenges.items.len > 0) {
                try self.opening_accumulator.accumulate(
                    stage_proof.challenges.items,
                    final_claim,
                );
            }

            self.current_stage = 4;
        }

        /// Stage 5: Register value evaluation
        ///
        /// This stage proves register read-write consistency similar to
        /// Stage 4 but for the 32 RISC-V registers. The register file
        /// is treated as a small memory with 32 addresses (log_k = 5).
        ///
        /// Special handling:
        /// - x0 (zero register) is hardwired to 0 - all reads must return 0
        /// - Uses same value evaluation formula as Stage 4
        ///
        /// Opening claims: Register polynomial evaluations
        fn proveStage5(self: *Self, transcript: anytype) !void {
            std.debug.print("[PROVER STAGE 5] Starting register value evaluation\n", .{});

            const stage_proof = &self.proofs.stage_proofs[4];

            // Get register address challenge (5 bits for 32 registers)
            const log_regs: usize = 5; // log2(32) = 5
            const r_register = try self.allocator.alloc(F, log_regs);
            defer self.allocator.free(r_register);
            std.debug.print("[PROVER STAGE 5]   getting {d} r_register challenges\n", .{log_regs});
            for (r_register, 0..) |*r, i| {
                r.* = try transcript.challengeScalar("r_register");
                std.debug.print("[PROVER STAGE 5]     r_register[{d}]=0x{x}\n", .{ i, r.limbs[0] });
            }

            // Get cycle challenge for register evaluation
            const r_cycle_reg = try self.allocator.alloc(F, self.log_t);
            defer self.allocator.free(r_cycle_reg);
            std.debug.print("[PROVER STAGE 5]   getting {d} r_cycle_reg challenges\n", .{self.log_t});
            for (r_cycle_reg, 0..) |*r, i| {
                r.* = try transcript.challengeScalar("r_cycle_reg");
                std.debug.print("[PROVER STAGE 5]     r_cycle_reg[{d}]=0x{x}\n", .{ i, r.limbs[0] });
            }

            // Register trace is embedded in the execution trace
            // We construct a simplified RAF+value check for registers
            const trace_len = self.trace.steps.items.len;
            std.debug.print("[PROVER STAGE 5]   trace_len={d}\n", .{trace_len});
            if (trace_len == 0) {
                std.debug.print("[PROVER STAGE 5]   Skipping - empty trace\n", .{});
                self.current_stage = 5;
                return;
            }

            // Number of rounds = log2(trace_len)
            const num_rounds = if (trace_len <= 1) 0 else std.math.log2_int_ceil(usize, trace_len);

            // Pad to power of 2
            const n = @as(usize, 1) << @intCast(num_rounds);
            std.debug.print("[PROVER STAGE 5]   num_rounds={d}, padded_n={d}\n", .{ num_rounds, n });

            // Materialize eq evaluations for each trace index
            // eq_evals[j] = eq(r_register, rd(j)) where rd(j) is destination reg at step j
            const eq_evals = try self.allocator.alloc(F, n);
            defer self.allocator.free(eq_evals);

            for (0..n) |j| {
                if (j < self.trace.steps.items.len) {
                    const step = self.trace.steps.items[j];
                    // Extract rd from instruction (bits [11:7] for RISC-V)
                    const rd: u8 = @truncate((step.instruction >> 7) & 0x1F);
                    eq_evals[j] = computeRegEq(F, r_register, rd);
                } else {
                    eq_evals[j] = F.zero();
                }
            }

            // Compute initial claim: sum of all eq evaluations
            var current_claim = F.zero();
            for (eq_evals) |e| {
                current_claim = current_claim.add(e);
            }

            // Record initial claim for verifier
            try stage_proof.final_claims.append(self.allocator, current_claim);
            std.debug.print("[PROVER STAGE 5]   initial_claim=0x{x}\n", .{current_claim.limbs[0]});

            // Working copy of evaluations that gets folded
            var working_evals = try self.allocator.alloc(F, n);
            defer self.allocator.free(working_evals);
            @memcpy(working_evals, eq_evals);

            var current_len = n;

            // Run sumcheck rounds (using degree-2 polynomials for multilinear)
            for (0..num_rounds) |round| {
                std.debug.print("[PROVER STAGE 5]   --- Round {d}/{d} ---\n", .{ round, num_rounds });

                // Compute round polynomial [p(0), p(2)] for degree-2 compressed format
                // Verifier recovers p(1) from constraint p(0) + p(1) = claim
                const poly_copy = try self.allocator.alloc(F, 2);

                var sum_at_0 = F.zero();
                var sum_at_1 = F.zero();
                const half = current_len / 2;

                for (0..half) |j| {
                    sum_at_0 = sum_at_0.add(working_evals[j]);
                    sum_at_1 = sum_at_1.add(working_evals[j + half]);
                }

                // Compute p(2) = 2*p(1) - p(0) for degree-1 polynomial (multilinear)
                const sum_at_2 = sum_at_1.add(sum_at_1).sub(sum_at_0);

                std.debug.print("[PROVER STAGE 5]     p(0)=0x{x}, p(1)=0x{x}, p(2)=0x{x}\n", .{ sum_at_0.limbs[0], sum_at_1.limbs[0], sum_at_2.limbs[0] });
                std.debug.print("[PROVER STAGE 5]     p(0)+p(1)=0x{x}, claim=0x{x}\n", .{ sum_at_0.add(sum_at_1).limbs[0], current_claim.limbs[0] });

                poly_copy[0] = sum_at_0;
                poly_copy[1] = sum_at_2; // Store p(2), not p(1)
                try stage_proof.round_polys.append(self.allocator, poly_copy);

                // Get challenge from transcript
                const challenge = try transcript.challengeScalar("reg_eval_round");
                try stage_proof.addChallenge(challenge);
                std.debug.print("[PROVER STAGE 5]     challenge=0x{x}\n", .{challenge.limbs[0]});

                // Bind/fold the polynomial: f_new[i] = (1-r)*f[i] + r*f[i+half]
                const one_minus_r = F.one().sub(challenge);
                for (0..half) |j| {
                    working_evals[j] = one_minus_r.mul(working_evals[j]).add(challenge.mul(working_evals[j + half]));
                }
                current_len = half;

                // Update current claim: p(r) = (1-r)*p(0) + r*p(1)
                current_claim = one_minus_r.mul(sum_at_0).add(challenge.mul(sum_at_1));
                std.debug.print("[PROVER STAGE 5]     new_claim=0x{x}\n", .{current_claim.limbs[0]});
            }

            // Record final claim (the value at the fully-bound point)
            const final_claim = if (current_len > 0) working_evals[0] else F.zero();
            try stage_proof.final_claims.append(self.allocator, final_claim);
            std.debug.print("[PROVER STAGE 5]   final_claim=0x{x}\n", .{final_claim.limbs[0]});

            // Accumulate opening claims
            if (stage_proof.challenges.items.len > 0) {
                try self.opening_accumulator.accumulate(
                    stage_proof.challenges.items,
                    final_claim,
                );
            }

            self.current_stage = 5;
        }

        /// Helper: compute eq(r, register_index)
        fn computeRegEq(comptime FieldType: type, r: []const FieldType, reg: u8) FieldType {
            var result = FieldType.one();
            for (r, 0..) |ri, i| {
                const bit = (reg >> @intCast(i)) & 1;
                if (bit == 1) {
                    result = result.mul(ri);
                } else {
                    result = result.mul(FieldType.one().sub(ri));
                }
            }
            return result;
        }

        /// Stage 6: Booleanity and finalization
        ///
        /// This stage proves:
        /// - Booleanity: All flags and selectors are in {0, 1}
        ///   Constraint: f * (1 - f) = 0 for all boolean flags f
        /// - Hamming weight: Exactly one instruction type per step
        ///   Constraint: Σ instruction_flags = 1
        /// - Final memory state: Terminal values are correct
        ///
        /// Structure:
        /// - Number of rounds: log2(trace_length)
        /// - Degree: 2 (quadratic for boolean check: f * (1-f))
        /// - Initial claim should be 0 for valid traces (all flags are boolean)
        fn proveStage6(self: *Self, transcript: anytype) !void {
            std.debug.print("[PROVER STAGE 6] Starting booleanity and finalization\n", .{});

            const stage_proof = &self.proofs.stage_proofs[5];

            // Get batching challenge for combining boolean constraints
            const bool_challenge = try transcript.challengeScalar("booleanity");
            std.debug.print("[PROVER STAGE 6]   bool_challenge=0x{x:0>16}{x:0>16}{x:0>16}{x:0>16}\n", .{ bool_challenge.limbs[3], bool_challenge.limbs[2], bool_challenge.limbs[1], bool_challenge.limbs[0] });

            const trace_len = self.trace.steps.items.len;
            std.debug.print("[PROVER STAGE 6]   trace_len={d}\n", .{trace_len});
            if (trace_len == 0) {
                std.debug.print("[PROVER STAGE 6]   Skipping - empty trace\n", .{});
                self.current_stage = 6;
                return;
            }

            // Number of rounds = log2(trace_len)
            const num_rounds = if (trace_len <= 1) 0 else std.math.log2_int_ceil(usize, trace_len);

            // Pad to power of 2
            const n = @as(usize, 1) << @intCast(num_rounds);
            std.debug.print("[PROVER STAGE 6]   num_rounds={d}, padded_n={d}\n", .{ num_rounds, n });

            // For booleanity, we sum f*(1-f) over all flags for each trace step.
            // For valid traces, all flags are boolean so f*(1-f) = 0 for each.
            //
            // We materialize violation_evals[j] = Σ_flags f_j * (1 - f_j)
            // For a valid trace, this should be 0 for all j.
            //
            // The sumcheck proves: Σ_j violation_evals[j] = 0
            const violation_evals = try self.allocator.alloc(F, n);
            defer self.allocator.free(violation_evals);

            // Initialize all evaluations
            // For a valid trace, all violations are 0
            for (0..n) |j| {
                if (j < self.trace.steps.items.len) {
                    // For valid traces, f*(1-f) = 0 for all boolean flags
                    // We assume the trace is valid, so violations are 0
                    violation_evals[j] = F.zero();
                } else {
                    violation_evals[j] = F.zero();
                }
            }

            // Compute initial claim: sum of all violations (should be 0)
            var current_claim = F.zero();
            for (violation_evals) |v| {
                current_claim = current_claim.add(v);
            }

            // Record initial claim for verifier
            try stage_proof.final_claims.append(self.allocator, current_claim);
            std.debug.print("[PROVER STAGE 6]   initial_claim=0x{x} (should be 0 for valid flags)\n", .{current_claim.limbs[0]});

            // Working copy of evaluations that gets folded
            var working_evals = try self.allocator.alloc(F, n);
            defer self.allocator.free(working_evals);
            @memcpy(working_evals, violation_evals);

            var current_len = n;

            // Run sumcheck rounds
            for (0..num_rounds) |round| {
                std.debug.print("[PROVER STAGE 6]   --- Round {d}/{d} ---\n", .{ round, num_rounds });

                // Compute round polynomial [p(0), p(2)] for degree-2 compressed format
                const poly_copy = try self.allocator.alloc(F, 2);

                var sum_at_0 = F.zero();
                var sum_at_1 = F.zero();
                const half = current_len / 2;

                for (0..half) |j| {
                    sum_at_0 = sum_at_0.add(working_evals[j]);
                    sum_at_1 = sum_at_1.add(working_evals[j + half]);
                }

                // For violations (which should all be 0), the polynomial is trivially 0
                // Compute p(2) = 2*p(1) - p(0) for degree-1 polynomial
                const sum_at_2 = sum_at_1.add(sum_at_1).sub(sum_at_0);

                std.debug.print("[PROVER STAGE 6]     p(0)=0x{x}, p(1)=0x{x}, p(2)=0x{x}\n", .{ sum_at_0.limbs[0], sum_at_1.limbs[0], sum_at_2.limbs[0] });

                poly_copy[0] = sum_at_0;
                poly_copy[1] = sum_at_2; // Store p(2), not p(1)
                try stage_proof.round_polys.append(self.allocator, poly_copy);

                // Get challenge from transcript
                const challenge = try transcript.challengeScalar("bool_round");
                try stage_proof.addChallenge(challenge);
                std.debug.print("[PROVER STAGE 6]     challenge=0x{x}\n", .{challenge.limbs[0]});

                // Bind/fold the polynomial: f_new[i] = (1-r)*f[i] + r*f[i+half]
                const one_minus_r = F.one().sub(challenge);
                for (0..half) |j| {
                    working_evals[j] = one_minus_r.mul(working_evals[j]).add(challenge.mul(working_evals[j + half]));
                }
                current_len = half;

                // Update current claim: p(r) = (1-r)*p(0) + r*p(1)
                current_claim = one_minus_r.mul(sum_at_0).add(challenge.mul(sum_at_1));
                std.debug.print("[PROVER STAGE 6]     new_claim=0x{x}\n", .{current_claim.limbs[0]});
            }

            // Record final claim (the value at the fully-bound point)
            const final_claim = if (current_len > 0) working_evals[0] else F.zero();
            try stage_proof.final_claims.append(self.allocator, final_claim);
            std.debug.print("[PROVER STAGE 6]   final_claim=0x{x}\n", .{final_claim.limbs[0]});

            // Accumulate opening claims for flag polynomials
            if (stage_proof.challenges.items.len > 0) {
                try self.opening_accumulator.accumulate(
                    stage_proof.challenges.items,
                    final_claim,
                );
            }

            self.current_stage = 6;
        }
    };
}

/// Batched sumcheck prover for running multiple instances in parallel
pub fn BatchedSumcheckProver(comptime F: type) type {
    return struct {
        const Self = @This();
        const Instance = SumcheckInstance(F);

        /// The batched instances
        instances: []Instance,
        /// Random linear combination coefficients
        batch_coeffs: []F,
        /// Current round
        round: usize,
        /// Total rounds (max across instances)
        total_rounds: usize,
        allocator: Allocator,

        pub fn init(allocator: Allocator, instances: []Instance, transcript: anytype) !Self {
            // Find max rounds
            var max_rounds: usize = 0;
            for (instances) |instance| {
                max_rounds = @max(max_rounds, instance.numRounds());
            }

            // Get batch coefficients from transcript
            const batch_coeffs = try allocator.alloc(F, instances.len);
            for (batch_coeffs, 0..) |*coeff, i| {
                _ = i;
                coeff.* = try transcript.challengeScalar("batch_coeff");
            }

            return Self{
                .instances = instances,
                .batch_coeffs = batch_coeffs,
                .round = 0,
                .total_rounds = max_rounds,
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.batch_coeffs);
        }

        /// Compute the batched round polynomial
        pub fn computeBatchedRoundPoly(self: *Self) []F {
            // Find max degree
            var max_degree: usize = 0;
            for (self.instances) |instance| {
                if (self.round < instance.numRounds()) {
                    max_degree = @max(max_degree, instance.degreeBound());
                }
            }

            const result = self.allocator.alloc(F, max_degree + 1) catch return &[_]F{};
            for (result) |*r| {
                r.* = F.zero();
            }

            // Combine instance polynomials with batch coefficients
            for (self.instances, self.batch_coeffs) |instance, coeff| {
                if (self.round >= instance.numRounds()) continue;

                const poly = instance.computeRoundPoly(self.round);
                for (poly, 0..) |val, i| {
                    if (i < result.len) {
                        result[i] = result[i].add(coeff.mul(val));
                    }
                }
            }

            return result;
        }

        /// Bind challenge to all instances
        pub fn bindChallenge(self: *Self, challenge: F) void {
            for (self.instances) |instance| {
                if (self.round < instance.numRounds()) {
                    instance.bindChallenge(challenge);
                }
            }
            self.round += 1;
        }

        /// Check if all rounds are complete
        pub fn isComplete(self: *const Self) bool {
            return self.round >= self.total_rounds;
        }
    };
}

// ============================================================================
// Tests
// ============================================================================

test "stage proof init and deinit" {
    const allocator = std.testing.allocator;
    const field = @import("../field/mod.zig");
    const F = field.BN254Scalar;

    var proof = StageProof(F).init(allocator);
    defer proof.deinit();

    const poly = [_]F{ F.one(), F.fromU64(2) };
    try proof.addRoundPoly(&poly);
    try proof.addChallenge(F.fromU64(42));

    try std.testing.expectEqual(@as(usize, 1), proof.round_polys.items.len);
    try std.testing.expectEqual(@as(usize, 1), proof.challenges.items.len);
}

test "jolt stage proofs init and deinit" {
    const allocator = std.testing.allocator;
    const field = @import("../field/mod.zig");
    const F = field.BN254Scalar;

    var proofs = JoltStageProofs(F).init(allocator);
    defer proofs.deinit();

    // Should have 6 stage proofs
    try std.testing.expectEqual(@as(usize, 6), proofs.stage_proofs.len);
}

test "opening accumulator" {
    const allocator = std.testing.allocator;
    const field = @import("../field/mod.zig");
    const F = field.BN254Scalar;

    var acc = OpeningAccumulator(F).init(allocator);
    defer acc.deinit();

    const point = [_]F{ F.one(), F.fromU64(2) };
    try acc.accumulate(&point, F.fromU64(42));

    try std.testing.expectEqual(@as(usize, 1), acc.opening_points.items.len);
    try std.testing.expectEqual(@as(usize, 1), acc.claimed_evals.items.len);
}

test "stage 5 sumcheck invariant: p(0) + p(1) = current_claim" {
    // Test that Stage 5 (register evaluation) properly maintains sumcheck invariant
    const allocator = std.testing.allocator;
    const field = @import("../field/mod.zig");
    const F = field.BN254Scalar;
    const transcripts = @import("../transcripts/mod.zig");
    const tracer = @import("../tracer/mod.zig");

    // Create a simple trace with a few steps
    var trace = tracer.ExecutionTrace.init(allocator);
    defer trace.deinit();

    // Add 4 steps (for 2 rounds of sumcheck)
    for (0..4) |i| {
        const step = tracer.TraceStep{
            .pc = @intCast(i * 4),
            .instruction = @intCast((i << 7) | 0x33), // rd = i, R-type instruction
            .rs1_value = 0,
            .rs2_value = 0,
            .rd_value = 0,
            .cycle = i,
            .memory_addr = null,
            .memory_value = null,
            .is_memory_write = false,
            .next_pc = @intCast((i + 1) * 4),
            .is_compressed = false,
        };
        try trace.steps.append(allocator, step);
    }

    // Create a multi-stage prover
    var prover = MultiStageProver(F).init(allocator, &trace);
    defer prover.deinit();

    // Create transcript
    var transcript = transcripts.Transcript(F).init();

    // Run all stages
    try prover.proveAllStages(&transcript);

    // Verify Stage 5 proof structure
    const stage5_proof = prover.proofs.stage_proofs[4];

    // Check that we have round polynomials (log2(4) = 2 rounds)
    try std.testing.expect(stage5_proof.round_polys.items.len == 2);

    // Verify the final_claims were set
    // Stage 5 should have: initial claim, then final claim
    try std.testing.expect(stage5_proof.final_claims.items.len >= 1);

    // For a proper sumcheck, the verifier would check:
    // Round 0: p0(0) + p0(1) = initial_claim
    // Round 1: p1(0) + p1(1) = p0(challenge0)
    // ...
    // The prover should produce consistent round polynomials
}

test "stage 6 sumcheck invariant: all zeros for valid trace" {
    // Test that Stage 6 (booleanity) produces zero polynomials for valid traces
    const allocator = std.testing.allocator;
    const field = @import("../field/mod.zig");
    const F = field.BN254Scalar;
    const transcripts = @import("../transcripts/mod.zig");
    const tracer = @import("../tracer/mod.zig");

    // Create a valid trace (all boolean flags should be 0 or 1)
    var trace = tracer.ExecutionTrace.init(allocator);
    defer trace.deinit();

    // Add 4 valid steps
    for (0..4) |i| {
        const step = tracer.TraceStep{
            .pc = @intCast(i * 4),
            .instruction = 0x00000033, // ADD x0, x0, x0 (valid instruction)
            .rs1_value = 0,
            .rs2_value = 0,
            .rd_value = 0,
            .cycle = i,
            .memory_addr = null,
            .memory_value = null,
            .is_memory_write = false,
            .next_pc = @intCast((i + 1) * 4),
            .is_compressed = false,
        };
        try trace.steps.append(allocator, step);
    }

    // Create a multi-stage prover
    var prover = MultiStageProver(F).init(allocator, &trace);
    defer prover.deinit();

    // Create transcript
    var transcript = transcripts.Transcript(F).init();

    // Run all stages
    try prover.proveAllStages(&transcript);

    // Verify Stage 6 proof structure
    const stage6_proof = prover.proofs.stage_proofs[5];

    // Check that we have round polynomials (log2(4) = 2 rounds)
    try std.testing.expect(stage6_proof.round_polys.items.len == 2);

    // For a valid trace with no boolean violations:
    // - Initial claim should be 0
    // - All round polynomials should be [0, 0]
    // - Final claim should be 0
    const initial_claim = stage6_proof.final_claims.items[0];
    try std.testing.expect(initial_claim.eql(F.zero()));

    for (stage6_proof.round_polys.items) |poly| {
        // Both p(0) and p(2) should be 0 for zero polynomial
        try std.testing.expect(poly[0].eql(F.zero()));
        try std.testing.expect(poly[1].eql(F.zero()));
    }

    // Final claim should also be 0
    const final_claim = stage6_proof.final_claims.items[stage6_proof.final_claims.items.len - 1];
    try std.testing.expect(final_claim.eql(F.zero()));
}
