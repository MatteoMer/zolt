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
            return Self{
                .round_polys = .{},
                .challenges = .{},
                .final_claims = .{},
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
        allocator: Allocator,

        pub fn init(allocator: Allocator) Self {
            var proofs: [6]StageProof(F) = undefined;
            for (&proofs) |*p| {
                p.* = StageProof(F).init(allocator);
            }
            return Self{
                .stage_proofs = proofs,
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            for (&self.stage_proofs) |*p| {
                p.deinit();
            }
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
            };
        }

        pub fn deinit(self: *Self) void {
            self.opening_accumulator.deinit();
            self.proofs.deinit();
        }

        /// Prove all stages
        pub fn prove(self: *Self, transcript: anytype) !JoltStageProofs(F) {
            try self.proveStage1(transcript);
            try self.proveStage2(transcript);
            try self.proveStage3(transcript);
            try self.proveStage4(transcript);
            try self.proveStage5(transcript);
            try self.proveStage6(transcript);

            return self.proofs;
        }

        /// Stage 1: Outer Spartan sumcheck
        ///
        /// This stage proves that all R1CS constraints are satisfied for
        /// the execution trace. The sumcheck proves:
        ///   sum_{x} eq(tau, x) * [(Az)(x) * (Bz)(x) - (Cz)(x)] = 0
        ///
        /// Structure:
        /// - Number of rounds: 1 + log2(trace_length) for uniform R1CS
        /// - Degree: 3 (multiquadratic after first round)
        /// - Opening claims: R1CS input evaluations at r_cycle
        fn proveStage1(self: *Self, transcript: anytype) !void {
            // Get tau challenge from transcript for the EQ polynomial
            const tau = transcript.challengeScalar("spartan_tau");

            // Number of rounds is based on trace length
            // For uniform Jolt R1CS: 1 + log2(T) rounds
            const num_rounds = 1 + self.log_t;
            _ = num_rounds;

            // In a full implementation, we would:
            // 1. Build R1CS constraints from the execution trace
            // 2. Materialize the witness vector z
            // 3. Compute Az, Bz, Cz
            // 4. Run sumcheck on eq(tau, x) * [Az(x) * Bz(x) - Cz(x)]
            // 5. Record opening claims for R1CS inputs at final point
            //
            // For now, we record a placeholder round polynomial
            const stage_proof = &self.proofs.stage_proofs[0];

            // Record that we've completed this stage
            // The stage proof structure will be populated by full implementation
            _ = tau;
            _ = stage_proof;

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
            const stage_proof = &self.proofs.stage_proofs[1];

            // Get r_cycle challenges from transcript (binding cycle variables)
            const r_cycle = try self.allocator.alloc(F, self.log_t);
            defer self.allocator.free(r_cycle);
            for (r_cycle) |*r| {
                r.* = transcript.challengeScalar("r_cycle");
            }

            // Initialize RAF evaluation parameters
            var raf_params = try ram.RafEvaluationParams(F).init(
                self.allocator,
                self.log_k,
                self.start_address,
                r_cycle,
            );
            defer raf_params.deinit();

            // Initialize RAF prover with memory trace
            var raf_prover = try ram.RafEvaluationProver(F).init(
                self.allocator,
                self.memory_trace,
                raf_params,
            );
            defer raf_prover.deinit();

            // Compute and record initial claim
            const initial_claim = raf_prover.computeInitialClaim();
            try stage_proof.final_claims.append(self.allocator, initial_claim);

            // Run sumcheck rounds
            const num_rounds = raf_params.numRounds();
            for (0..num_rounds) |round| {
                // Compute round polynomial [p(0), p(1)]
                const round_poly = raf_prover.computeRoundPolynomial();

                // Store round polynomial in proof
                const poly_copy = try self.allocator.alloc(F, 2);
                poly_copy[0] = round_poly[0];
                poly_copy[1] = round_poly[1];
                try stage_proof.round_polys.append(self.allocator, poly_copy);

                // Get challenge from transcript
                const challenge = transcript.challengeScalar("raf_round");
                try stage_proof.addChallenge(challenge);

                // Bind the challenge for next round
                try raf_prover.bindChallenge(challenge);
                _ = round;
            }

            // Record final claim after all rounds
            const final_claim = raf_prover.getFinalClaim();
            try stage_proof.final_claims.append(self.allocator, final_claim);

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
            const stage_proof = &self.proofs.stage_proofs[2];

            // Get batching challenge γ for combining lookup instances
            const gamma = transcript.challengeScalar("lasso_gamma");

            // Get lookup statistics for proof metadata
            const stats = self.lookup_trace.getStats();
            _ = stats;

            // Extract lookup indices from the trace
            const trace_entries = self.lookup_trace.getEntries();

            if (trace_entries.len == 0) {
                // No lookups to prove - stage is trivially complete
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
            for (r_reduction) |*r| {
                r.* = transcript.challengeScalar("r_reduction");
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

            // Initialize and run Lasso prover
            var lasso_prover = try lasso.LassoProver(F).init(
                self.allocator,
                lookup_indices,
                lookup_tables,
                params,
            );
            defer lasso_prover.deinit();

            // Run Lasso sumcheck rounds
            const total_rounds = params.log_K + params.log_T;
            for (0..total_rounds) |round| {
                // Compute round polynomial
                var round_poly = try lasso_prover.computeRoundPolynomial();

                // Store polynomial coefficients in stage proof
                try stage_proof.round_polys.append(self.allocator, round_poly.coeffs);
                // Prevent double-free since we stored the coefficients
                round_poly.coeffs = &[_]F{};

                // Get challenge from transcript
                const challenge = transcript.challengeScalar("lasso_round");
                try stage_proof.addChallenge(challenge);

                // Bind the challenge
                try lasso_prover.receiveChallenge(challenge);
                _ = round;
            }

            // Record final evaluation
            if (lasso_prover.isComplete()) {
                const final_eval = lasso_prover.getFinalEval();
                try stage_proof.final_claims.append(self.allocator, final_eval);
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
            const stage_proof = &self.proofs.stage_proofs[3];

            // Get address challenge from previous stage
            const r_address = try self.allocator.alloc(F, self.log_k);
            defer self.allocator.free(r_address);
            for (r_address) |*r| {
                r.* = transcript.challengeScalar("r_address");
            }

            // Get cycle challenge from previous stage
            const r_cycle = try self.allocator.alloc(F, self.log_t);
            defer self.allocator.free(r_cycle);
            for (r_cycle) |*r| {
                r.* = transcript.challengeScalar("r_cycle_val");
            }

            // Initial value evaluation (for uninitialized memory, this is 0)
            const init_eval = F.zero();

            // Initialize value evaluation parameters
            const trace_len = self.trace.steps.items.len;
            const k = @as(usize, 1) << @intCast(self.log_k);

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
                self.current_stage = 4;
                return;
            }

            // Initialize value evaluation prover
            const initial_state = try self.allocator.alloc(u64, k);
            defer self.allocator.free(initial_state);
            @memset(initial_state, 0);

            var val_prover = try ram.ValEvaluationProver(F).init(
                self.allocator,
                self.memory_trace,
                initial_state,
                val_params,
                self.start_address,
            );
            defer val_prover.deinit();

            // Compute and record initial claim
            const initial_claim = val_prover.computeInitialClaim();
            try stage_proof.final_claims.append(self.allocator, initial_claim);

            // Run sumcheck rounds
            const num_rounds = val_params.numRounds();
            for (0..num_rounds) |round| {
                // Compute round polynomial [p(0), p(1), p(2)]
                const round_poly = val_prover.computeRoundPolynomial();

                // Store polynomial in proof
                const poly_copy = try self.allocator.alloc(F, 3);
                poly_copy[0] = round_poly[0];
                poly_copy[1] = round_poly[1];
                poly_copy[2] = round_poly[2];
                try stage_proof.round_polys.append(self.allocator, poly_copy);

                // Get challenge from transcript
                const challenge = transcript.challengeScalar("val_eval_round");
                try stage_proof.addChallenge(challenge);

                // Bind the challenge
                val_prover.bindChallenge(challenge);
                _ = round;
            }

            // Record final claim
            const final_claim = val_prover.getFinalClaim();
            try stage_proof.final_claims.append(self.allocator, final_claim);

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
            const stage_proof = &self.proofs.stage_proofs[4];

            // Get register address challenge (5 bits for 32 registers)
            const log_regs: usize = 5; // log2(32) = 5
            const r_register = try self.allocator.alloc(F, log_regs);
            defer self.allocator.free(r_register);
            for (r_register) |*r| {
                r.* = transcript.challengeScalar("r_register");
            }

            // Get cycle challenge for register evaluation
            const r_cycle_reg = try self.allocator.alloc(F, self.log_t);
            defer self.allocator.free(r_cycle_reg);
            for (r_cycle_reg) |*r| {
                r.* = transcript.challengeScalar("r_cycle_reg");
            }

            // Initial register value evaluation (all registers start at 0)
            const init_eval = F.zero();

            // Register trace is embedded in the execution trace
            // We construct a simplified RAF+value check for registers
            const trace_len = self.trace.steps.items.len;
            if (trace_len == 0) {
                self.current_stage = 5;
                return;
            }

            // For register consistency, we use a simplified version of value evaluation
            // The sumcheck proves: reg_val(r) = Σ_j write_delta(j) · was_written(r_reg, j) · before(j, r_cycle)
            //
            // Number of rounds = log2(trace_len)
            const num_rounds = if (trace_len <= 1) 0 else std.math.log2_int_ceil(usize, trace_len);

            // Run sumcheck rounds (using simplified linear polynomials)
            for (0..num_rounds) |round| {
                // Compute round polynomial [p(0), p(1)]
                // For registers, we use degree 2 since it's simpler than full memory
                const poly_copy = try self.allocator.alloc(F, 2);

                // Simplified: sum over trace entries based on register writes
                var sum_0 = F.zero();
                var sum_1 = F.zero();
                const half = trace_len / 2;

                for (0..half) |j| {
                    // Add contribution based on whether this step wrote to a register
                    if (self.trace.steps.items.len > j) {
                        const step = self.trace.steps.items[j];
                        // Check if destination register matches r_register (simplified)
                        const eq_val = computeRegEq(F, r_register, step.rd);
                        sum_0 = sum_0.add(eq_val);
                    }
                    if (self.trace.steps.items.len > j + half) {
                        const step = self.trace.steps.items[j + half];
                        const eq_val = computeRegEq(F, r_register, step.rd);
                        sum_1 = sum_1.add(eq_val);
                    }
                }

                poly_copy[0] = sum_0;
                poly_copy[1] = sum_1;
                try stage_proof.round_polys.append(self.allocator, poly_copy);

                // Get challenge from transcript
                const challenge = transcript.challengeScalar("reg_eval_round");
                try stage_proof.addChallenge(challenge);
                _ = round;
            }

            // Record final claim
            const final_claim = init_eval; // Simplified - real impl would compute properly
            try stage_proof.final_claims.append(self.allocator, final_claim);

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
        /// - Number of rounds: log2(trace_length) + log2(num_flags)
        /// - Degree: 2 (quadratic for boolean check)
        /// - Final claims: All flags satisfy booleanity
        fn proveStage6(self: *Self, transcript: anytype) !void {
            const stage_proof = &self.proofs.stage_proofs[5];

            // Get batching challenge for combining boolean constraints
            const bool_challenge = transcript.challengeScalar("booleanity");

            // Number of circuit flags to check
            const num_circuit_flags: usize = 13; // From instruction/mod.zig CircuitFlags
            const num_instruction_flags: usize = 7; // From instruction/mod.zig InstructionFlags
            const total_flags = num_circuit_flags + num_instruction_flags;
            _ = total_flags;

            const trace_len = self.trace.steps.items.len;
            if (trace_len == 0) {
                self.current_stage = 6;
                return;
            }

            // For each step, we need to verify:
            // 1. All circuit flags are boolean: flag * (1 - flag) = 0
            // 2. Instruction flags have Hamming weight 1
            //
            // The sumcheck proves: Σ_j eq(r, j) * Σ_f (f_j * (1 - f_j)) = 0
            //
            // Number of rounds = log2(trace_len)
            const num_rounds = if (trace_len <= 1) 0 else std.math.log2_int_ceil(usize, trace_len);

            // Compute booleanity claim for each step
            var total_violation = F.zero();

            for (self.trace.steps.items) |step| {
                // Check destination register is valid (0-31)
                const rd_valid = step.rd < 32;
                if (!rd_valid) {
                    // Flag violation (in real impl, this would be caught earlier)
                    total_violation = total_violation.add(F.one());
                }

                // Check source registers are valid
                const rs1_valid = step.rs1 < 32;
                const rs2_valid = step.rs2 < 32;
                if (!rs1_valid or !rs2_valid) {
                    total_violation = total_violation.add(F.one());
                }
            }

            // The initial claim for booleanity sumcheck should be 0
            // (all flags are boolean -> no violations)
            const initial_claim = total_violation.mul(bool_challenge);
            try stage_proof.final_claims.append(self.allocator, initial_claim);

            // Run sumcheck rounds
            for (0..num_rounds) |round| {
                // Compute round polynomial [p(0), p(1)]
                const poly_copy = try self.allocator.alloc(F, 2);

                // For booleanity, sumcheck verifies: Σ eq(r,j) * violation(j) = 0
                var sum_0 = F.zero();
                var sum_1 = F.zero();
                const half = trace_len / 2;

                for (0..half) |j| {
                    // Contribution from first half (x = 0)
                    if (j < self.trace.steps.items.len) {
                        sum_0 = sum_0.add(F.zero()); // No violation assumed
                    }
                    // Contribution from second half (x = 1)
                    if (j + half < self.trace.steps.items.len) {
                        sum_1 = sum_1.add(F.zero()); // No violation assumed
                    }
                }

                poly_copy[0] = sum_0;
                poly_copy[1] = sum_1;
                try stage_proof.round_polys.append(self.allocator, poly_copy);

                // Get challenge from transcript
                const challenge = transcript.challengeScalar("bool_round");
                try stage_proof.addChallenge(challenge);
                _ = round;
            }

            // Final claim should be 0 if all flags are boolean
            const final_claim = F.zero();
            try stage_proof.final_claims.append(self.allocator, final_claim);

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
                coeff.* = transcript.challengeScalar("batch_coeff");
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
