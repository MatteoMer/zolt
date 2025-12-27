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
        fn proveStage1(self: *Self, transcript: anytype) !void {
            _ = transcript;
            // TODO: Implement Spartan outer sumcheck
            // This proves instruction correctness via R1CS
            self.current_stage = 1;
        }

        /// Stage 2: RAM RAF evaluation and read-write checking
        fn proveStage2(self: *Self, transcript: anytype) !void {
            _ = transcript;

            // Initialize RAF evaluation sumcheck
            const r_cycle = try self.allocator.alloc(F, self.log_t);
            defer self.allocator.free(r_cycle);
            for (r_cycle) |*r| {
                r.* = F.fromU64(2); // Placeholder - should come from transcript
            }

            var raf_params = try ram.RafEvaluationParams(F).init(
                self.allocator,
                self.log_k,
                self.start_address,
                r_cycle,
            );
            defer raf_params.deinit();

            var raf_prover = try ram.RafEvaluationProver(F).init(
                self.allocator,
                self.memory_trace,
                raf_params,
            );
            defer raf_prover.deinit();

            // Record initial claim
            const raf_claim = raf_prover.computeInitialClaim();
            _ = raf_claim;

            // TODO: Run sumcheck rounds and accumulate proofs
            self.current_stage = 2;
        }

        /// Stage 3: Instruction lookup reduction
        fn proveStage3(self: *Self, transcript: anytype) !void {
            _ = transcript;
            // TODO: Implement Lasso lookup reduction sumcheck
            // This uses the lookup trace to prove instruction lookups
            _ = self.lookup_trace;
            self.current_stage = 3;
        }

        /// Stage 4: Memory value evaluation
        fn proveStage4(self: *Self, transcript: anytype) !void {
            _ = transcript;
            // TODO: Implement ValEvaluation sumcheck
            self.current_stage = 4;
        }

        /// Stage 5: Register value evaluation
        fn proveStage5(self: *Self, transcript: anytype) !void {
            _ = transcript;
            // TODO: Implement RegisterValEvaluation sumcheck
            self.current_stage = 5;
        }

        /// Stage 6: Booleanity and finalization
        fn proveStage6(self: *Self, transcript: anytype) !void {
            _ = transcript;
            // TODO: Implement booleanity and Hamming weight checks
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
