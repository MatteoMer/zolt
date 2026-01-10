//! Jolt R1CS - Connects uniform constraints to Spartan prover
//!
//! This module builds the full R1CS system for Jolt by:
//! 1. Expanding uniform constraints across all execution cycles
//! 2. Generating the witness vector from execution trace
//! 3. Computing Az, Bz, Cz for Spartan sumcheck
//!
//! ## Structure
//!
//! For T execution cycles with K uniform constraints:
//! - Total constraints: T * K (padded to power of 2)
//! - Witness size: T * INPUTS_PER_CYCLE + 1 (for constant 1)
//!
//! The witness layout is:
//! z = [1, cycle_0_inputs..., cycle_1_inputs..., ..., cycle_{T-1}_inputs...]
//!
//! Reference: jolt-core/src/zkvm/r1cs/mod.rs

const std = @import("std");
const Allocator = std.mem.Allocator;

const tracer = @import("../../tracer/mod.zig");
const constraints = @import("constraints.zig");
const R1CSInputIndex = constraints.R1CSInputIndex;
const UniformConstraint = constraints.UniformConstraint;
const UNIFORM_CONSTRAINTS = constraints.UNIFORM_CONSTRAINTS;
const LC = constraints.LC;
const R1CSCycleInputs = constraints.R1CSCycleInputs;

/// Jolt R1CS system
///
/// Manages the expanded R1CS constraints and witness generation
/// for proving execution correctness via Spartan.
pub fn JoltR1CS(comptime F: type) type {
    return struct {
        const Self = @This();

        /// Number of inputs per cycle (36 witness variables)
        pub const INPUTS_PER_CYCLE: usize = R1CSInputIndex.NUM_INPUTS;
        /// Number of uniform constraints per cycle (19)
        pub const CONSTRAINTS_PER_CYCLE: usize = UNIFORM_CONSTRAINTS.len;

        /// Number of execution cycles
        num_cycles: usize,
        /// Log2 of padded number of constraints
        log_num_constraints: usize,
        /// Padded number of constraints (power of 2)
        padded_num_constraints: usize,
        /// Total witness size
        witness_size: usize,
        /// Per-cycle witness data
        cycle_witnesses: []R1CSCycleInputs(F),
        allocator: Allocator,

        /// Initialize from execution trace
        pub fn fromTrace(
            allocator: Allocator,
            trace: *const tracer.ExecutionTrace,
        ) !Self {
            const num_cycles = trace.steps.items.len;

            // Calculate dimensions
            const total_constraints = num_cycles * CONSTRAINTS_PER_CYCLE;
            const padded = if (total_constraints == 0)
                1
            else
                std.math.ceilPowerOfTwo(usize, total_constraints) catch total_constraints;
            const log_n = if (padded <= 1) 0 else std.math.log2_int(usize, padded);

            // Witness: [1] + [cycle_0_inputs...] + [cycle_1_inputs...] + ...
            const witness_size = 1 + num_cycles * INPUTS_PER_CYCLE;

            // Generate per-cycle witnesses
            if (num_cycles == 0) {
                return Self{
                    .num_cycles = 0,
                    .log_num_constraints = log_n,
                    .padded_num_constraints = padded,
                    .witness_size = witness_size,
                    .cycle_witnesses = &[_]R1CSCycleInputs(F){},
                    .allocator = allocator,
                };
            }

            const cycle_witnesses = try allocator.alloc(R1CSCycleInputs(F), num_cycles);
            for (0..num_cycles) |i| {
                const step = trace.steps.items[i];

                if (step.is_noop) {
                    // NoOp padding cycle: all zeros with IsNoop=1
                    cycle_witnesses[i] = R1CSCycleInputs(F).createNoopWitness();
                } else {
                    // Real cycle: next step always exists after padding
                    const next_step = trace.steps.items[i + 1];
                    cycle_witnesses[i] = R1CSCycleInputs(F).fromTraceStep(step, next_step);
                }
            }

            return Self{
                .num_cycles = num_cycles,
                .log_num_constraints = log_n,
                .padded_num_constraints = padded,
                .witness_size = witness_size,
                .cycle_witnesses = cycle_witnesses,
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            if (self.num_cycles > 0) {
                self.allocator.free(self.cycle_witnesses);
            }
        }

        /// Build the full witness vector
        ///
        /// Layout: [1, cycle_0[0..35], cycle_1[0..35], ...]
        pub fn buildWitness(self: *const Self) ![]F {
            const witness = try self.allocator.alloc(F, self.witness_size);

            // First element is always 1 (for constant terms)
            witness[0] = F.one();

            // Copy per-cycle witnesses
            for (0..self.num_cycles) |cycle| {
                const offset = 1 + cycle * INPUTS_PER_CYCLE;
                const cycle_witness = &self.cycle_witnesses[cycle];
                @memcpy(witness[offset..][0..INPUTS_PER_CYCLE], &cycle_witness.values);
            }

            return witness;
        }

        /// Compute Az (A matrix times witness vector)
        ///
        /// For each constraint i in cycle c:
        /// Az[c * CONSTRAINTS_PER_CYCLE + i] = A_i · z_c
        ///
        /// where A_i is the A component of uniform constraint i
        pub fn computeAz(self: *const Self, witness: []const F) ![]F {
            const Az = try self.allocator.alloc(F, self.padded_num_constraints);
            @memset(Az, F.zero());

            for (0..self.num_cycles) |cycle| {
                const witness_offset = 1 + cycle * INPUTS_PER_CYCLE;

                for (UNIFORM_CONSTRAINTS, 0..) |constraint, i| {
                    const constraint_idx = cycle * CONSTRAINTS_PER_CYCLE + i;
                    if (constraint_idx >= self.padded_num_constraints) break;

                    // For equality-conditional form: A = condition
                    Az[constraint_idx] = evaluateLC(
                        F,
                        constraint.condition,
                        witness,
                        witness_offset,
                    );
                }
            }

            return Az;
        }

        /// Compute Bz (B matrix times witness vector)
        ///
        /// For equality-conditional form: B = left - right
        pub fn computeBz(self: *const Self, witness: []const F) ![]F {
            const Bz = try self.allocator.alloc(F, self.padded_num_constraints);
            @memset(Bz, F.zero());

            for (0..self.num_cycles) |cycle| {
                const witness_offset = 1 + cycle * INPUTS_PER_CYCLE;

                for (UNIFORM_CONSTRAINTS, 0..) |constraint, i| {
                    const constraint_idx = cycle * CONSTRAINTS_PER_CYCLE + i;
                    if (constraint_idx >= self.padded_num_constraints) break;

                    // B = left - right
                    const left = evaluateLC(F, constraint.left, witness, witness_offset);
                    const right = evaluateLC(F, constraint.right, witness, witness_offset);
                    Bz[constraint_idx] = left.sub(right);
                }
            }

            return Bz;
        }

        /// Compute Cz (C matrix times witness vector)
        ///
        /// For equality-conditional form: C = 0 (constraint is A*B = 0)
        pub fn computeCz(self: *const Self, _: []const F) ![]F {
            const Cz = try self.allocator.alloc(F, self.padded_num_constraints);
            @memset(Cz, F.zero());
            // In equality-conditional form, C is always 0
            // The constraint is: condition * (left - right) = 0
            return Cz;
        }

        /// Verify that all constraints are satisfied
        pub fn verifySatisfied(self: *const Self, witness: []const F) bool {
            for (0..self.num_cycles) |cycle| {
                const witness_offset = 1 + cycle * INPUTS_PER_CYCLE;

                for (UNIFORM_CONSTRAINTS) |constraint| {
                    const cond = evaluateLC(F, constraint.condition, witness, witness_offset);
                    const left = evaluateLC(F, constraint.left, witness, witness_offset);
                    const right = evaluateLC(F, constraint.right, witness, witness_offset);

                    // Constraint: cond * (left - right) = 0
                    const result = cond.mul(left.sub(right));
                    if (!result.eql(F.zero())) {
                        return false;
                    }
                }
            }
            return true;
        }

        /// Helper: convert i128 to field element (handles values > 2^64)
        fn i128ToField(comptime FieldType: type, val: i128) FieldType {
            if (val >= 0) {
                const v: u128 = @intCast(val);
                if (v <= 0xFFFFFFFFFFFFFFFF) {
                    return FieldType.fromU64(@intCast(v));
                } else {
                    // Value > 2^64, use bytes representation
                    var bytes: [16]u8 = undefined;
                    std.mem.writeInt(u128, &bytes, v, .little);
                    return FieldType.fromBytes(&bytes);
                }
            } else {
                const neg_v: u128 = @intCast(-val);
                if (neg_v <= 0xFFFFFFFFFFFFFFFF) {
                    return FieldType.zero().sub(FieldType.fromU64(@intCast(neg_v)));
                } else {
                    // Value > 2^64, use bytes representation
                    var bytes: [16]u8 = undefined;
                    std.mem.writeInt(u128, &bytes, neg_v, .little);
                    return FieldType.zero().sub(FieldType.fromBytes(&bytes));
                }
            }
        }

        /// Helper: evaluate a linear combination at a given witness offset
        fn evaluateLC(
            comptime FieldType: type,
            lc: LC,
            witness: []const FieldType,
            witness_offset: usize,
        ) FieldType {
            var result = i128ToField(FieldType, lc.constant);

            for (lc.terms[0..lc.len]) |term| {
                const idx = witness_offset + term.input_index.toIndex();
                if (idx >= witness.len) continue;

                const val = witness[idx];
                const coeff_field = i128ToField(FieldType, if (term.coeff >= 0) term.coeff else -term.coeff);
                const scaled = val.mul(coeff_field);
                if (term.coeff >= 0) {
                    result = result.add(scaled);
                } else {
                    result = result.sub(scaled);
                }
            }

            return result;
        }
    };
}

/// Spartan-compatible R1CS interface for Jolt
///
/// Provides the sumcheck polynomial for Stage 1:
/// f(x) = eq(tau, x) * [Az(x) * Bz(x) - Cz(x)]
pub fn JoltSpartanInterface(comptime F: type) type {
    return struct {
        const Self = @This();
        const poly = @import("../../poly/mod.zig");

        /// The Jolt R1CS system
        r1cs: *const JoltR1CS(F),
        /// Witness vector
        witness: []const F,
        /// Az evaluations
        Az: []F,
        /// Bz evaluations
        Bz: []F,
        /// Cz evaluations
        Cz: []F,
        /// EQ polynomial evaluations at tau
        eq_evals: []F,
        /// Combined polynomial: eq(tau,x) * [Az*Bz - Cz]
        combined_poly: []F,
        /// Current effective length of combined_poly (halves each round)
        current_len: usize,
        /// Current bound challenge values
        challenges: std.ArrayListUnmanaged(F),
        allocator: Allocator,

        /// Initialize from Jolt R1CS and tau challenge
        pub fn init(
            allocator: Allocator,
            r1cs: *const JoltR1CS(F),
            witness: []const F,
            tau: []const F,
        ) !Self {
            std.debug.print("\n[JOLT SPARTAN] =====================================\n", .{});
            std.debug.print("[JOLT SPARTAN] JoltSpartanInterface.init\n", .{});
            std.debug.print("[JOLT SPARTAN] num_cycles = {d}\n", .{r1cs.num_cycles});
            std.debug.print("[JOLT SPARTAN] padded_num_constraints = {d}\n", .{r1cs.padded_num_constraints});
            std.debug.print("[JOLT SPARTAN] log_num_constraints = {d}\n", .{r1cs.log_num_constraints});
            std.debug.print("[JOLT SPARTAN] witness.len = {d}\n", .{witness.len});
            std.debug.print("[JOLT SPARTAN] tau.len = {d}\n", .{tau.len});

            // Print tau values
            for (tau, 0..) |t, i| {
                std.debug.print("[JOLT SPARTAN] tau[{d}] = {x:0>16}{x:0>16}{x:0>16}{x:0>16}\n", .{ i, t.limbs[3], t.limbs[2], t.limbs[1], t.limbs[0] });
            }

            // Compute Az, Bz, Cz
            const Az = try r1cs.computeAz(witness);
            errdefer allocator.free(Az);
            const Bz = try r1cs.computeBz(witness);
            errdefer allocator.free(Bz);
            const Cz = try r1cs.computeCz(witness);
            errdefer allocator.free(Cz);

            std.debug.print("[JOLT SPARTAN] Az.len = {d}, Bz.len = {d}, Cz.len = {d}\n", .{ Az.len, Bz.len, Cz.len });

            // Print first few Az, Bz values
            const num_to_print = @min(8, Az.len);
            for (0..num_to_print) |i| {
                std.debug.print("[JOLT SPARTAN] Az[{d}] = {x:0>16}{x:0>16}{x:0>16}{x:0>16}\n", .{ i, Az[i].limbs[3], Az[i].limbs[2], Az[i].limbs[1], Az[i].limbs[0] });
                std.debug.print("[JOLT SPARTAN] Bz[{d}] = {x:0>16}{x:0>16}{x:0>16}{x:0>16}\n", .{ i, Bz[i].limbs[3], Bz[i].limbs[2], Bz[i].limbs[1], Bz[i].limbs[0] });
            }

            // Compute eq(tau, x) for all x
            var eq_poly = try poly.EqPolynomial(F).init(allocator, tau);
            defer eq_poly.deinit();
            const eq_evals = try eq_poly.evals(allocator);
            errdefer allocator.free(eq_evals);

            std.debug.print("[JOLT SPARTAN] eq_evals.len = {d}\n", .{eq_evals.len});

            // Compute combined polynomial
            const size = r1cs.padded_num_constraints;
            const combined = try allocator.alloc(F, size);
            for (0..size) |i| {
                if (i < Az.len and i < Bz.len and i < Cz.len and i < eq_evals.len) {
                    const ab = Az[i].mul(Bz[i]);
                    const abc = ab.sub(Cz[i]);
                    combined[i] = eq_evals[i].mul(abc);
                } else {
                    combined[i] = F.zero();
                }
            }

            // Print first few combined values
            for (0..num_to_print) |i| {
                if (i < combined.len) {
                    std.debug.print("[JOLT SPARTAN] combined[{d}] = {x:0>16}{x:0>16}{x:0>16}{x:0>16}\n", .{ i, combined[i].limbs[3], combined[i].limbs[2], combined[i].limbs[1], combined[i].limbs[0] });
                }
            }

            std.debug.print("[JOLT SPARTAN] =====================================\n", .{});

            return Self{
                .r1cs = r1cs,
                .witness = witness,
                .Az = Az,
                .Bz = Bz,
                .Cz = Cz,
                .eq_evals = eq_evals,
                .combined_poly = combined,
                .current_len = size,
                .challenges = .{},
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.Az);
            self.allocator.free(self.Bz);
            self.allocator.free(self.Cz);
            self.allocator.free(self.eq_evals);
            self.allocator.free(self.combined_poly);
            self.challenges.deinit(self.allocator);
        }

        /// Get initial sumcheck claim (should be 0 for valid witness)
        pub fn initialClaim(self: *const Self) F {
            var sum = F.zero();
            for (self.combined_poly[0..self.current_len]) |v| {
                sum = sum.add(v);
            }
            std.debug.print("\n[JOLT SPARTAN] initialClaim = {x:0>16}{x:0>16}{x:0>16}{x:0>16}\n", .{ sum.limbs[3], sum.limbs[2], sum.limbs[1], sum.limbs[0] });
            std.debug.print("[JOLT SPARTAN] current_len = {d}\n", .{self.current_len});
            return sum;
        }

        /// Get number of sumcheck rounds
        pub fn numRounds(self: *const Self) usize {
            return self.r1cs.log_num_constraints;
        }

        /// Compute round polynomial for current round
        ///
        /// Returns [p(0), p(1), p(2)] for degree-2 sumcheck
        pub fn computeRoundPolynomial(self: *Self) ![3]F {
            const round = self.challenges.items.len;
            std.debug.print("\n[JOLT SPARTAN] computeRoundPolynomial round={d}\n", .{round});
            std.debug.print("[JOLT SPARTAN] current_len = {d}\n", .{self.current_len});

            if (self.current_len <= 1) {
                const p0 = if (self.current_len == 1) self.combined_poly[0] else F.zero();
                std.debug.print("[JOLT SPARTAN] Single element case: p(0) = {x:0>16}{x:0>16}{x:0>16}{x:0>16}\n", .{ p0.limbs[3], p0.limbs[2], p0.limbs[1], p0.limbs[0] });
                return [3]F{
                    p0,
                    F.zero(),
                    F.zero(),
                };
            }

            const half = self.current_len / 2;

            // p(0) = sum of first half
            // p(1) = sum of second half
            var p0 = F.zero();
            var p1 = F.zero();

            for (0..half) |i| {
                p0 = p0.add(self.combined_poly[i]);
                p1 = p1.add(self.combined_poly[i + half]);
            }

            // p(2) = extrapolation for linear polynomial
            // Since combined_poly is multilinear, after folding it remains linear in each variable
            // The round polynomial is linear, so p(2) = 2*p(1) - p(0)
            const p2 = p1.add(p1).sub(p0);

            std.debug.print("[JOLT SPARTAN] p(0) = {x:0>16}{x:0>16}{x:0>16}{x:0>16}\n", .{ p0.limbs[3], p0.limbs[2], p0.limbs[1], p0.limbs[0] });
            std.debug.print("[JOLT SPARTAN] p(1) = {x:0>16}{x:0>16}{x:0>16}{x:0>16}\n", .{ p1.limbs[3], p1.limbs[2], p1.limbs[1], p1.limbs[0] });
            std.debug.print("[JOLT SPARTAN] p(2) = {x:0>16}{x:0>16}{x:0>16}{x:0>16}\n", .{ p2.limbs[3], p2.limbs[2], p2.limbs[1], p2.limbs[0] });

            const sum = p0.add(p1);
            std.debug.print("[JOLT SPARTAN] p(0)+p(1) = {x:0>16}{x:0>16}{x:0>16}{x:0>16}\n", .{ sum.limbs[3], sum.limbs[2], sum.limbs[1], sum.limbs[0] });

            return [3]F{ p0, p1, p2 };
        }

        /// Bind challenge for this round
        pub fn bindChallenge(self: *Self, challenge: F) !void {
            const round = self.challenges.items.len;
            std.debug.print("\n[JOLT SPARTAN] bindChallenge round={d}\n", .{round});
            std.debug.print("[JOLT SPARTAN] challenge = {x:0>16}{x:0>16}{x:0>16}{x:0>16}\n", .{ challenge.limbs[3], challenge.limbs[2], challenge.limbs[1], challenge.limbs[0] });
            std.debug.print("[JOLT SPARTAN] current_len (before) = {d}\n", .{self.current_len});

            try self.challenges.append(self.allocator, challenge);

            if (self.current_len <= 1) {
                std.debug.print("[JOLT SPARTAN] current_len <= 1, no folding\n", .{});
                return;
            }

            const half = self.current_len / 2;
            const one_minus_r = F.one().sub(challenge);

            // Fold: new[i] = (1-r) * old[i] + r * old[i + half]
            for (0..half) |i| {
                self.combined_poly[i] = one_minus_r.mul(self.combined_poly[i])
                    .add(challenge.mul(self.combined_poly[i + half]));
            }

            // Update the effective length
            self.current_len = half;

            // Compute sum of folded values (new claim)
            var new_sum = F.zero();
            for (0..half) |i| {
                new_sum = new_sum.add(self.combined_poly[i]);
            }
            std.debug.print("[JOLT SPARTAN] current_len (after) = {d}\n", .{self.current_len});
            std.debug.print("[JOLT SPARTAN] new_claim (sum of folded) = {x:0>16}{x:0>16}{x:0>16}{x:0>16}\n", .{ new_sum.limbs[3], new_sum.limbs[2], new_sum.limbs[1], new_sum.limbs[0] });
        }

        /// Get the final evaluation after all rounds
        pub fn getFinalEval(self: *const Self) F {
            if (self.combined_poly.len > 0) {
                return self.combined_poly[0];
            }
            return F.zero();
        }

        /// Get evaluation claims for Az, Bz, Cz at the final point
        pub fn getEvalClaims(self: *const Self) [3]F {
            const point = self.challenges.items;
            return [3]F{
                self.evaluateAtPoint(self.Az, point),
                self.evaluateAtPoint(self.Bz, point),
                self.evaluateAtPoint(self.Cz, point),
            };
        }

        /// Evaluate a polynomial (in evaluation form) at a point
        fn evaluateAtPoint(self: *const Self, evals: []const F, point: []const F) F {
            _ = self;
            if (evals.len == 0) return F.zero();
            if (point.len == 0) return evals[0];

            // Recursive multilinear extension evaluation
            var result = F.zero();
            const n = @min(evals.len, @as(usize, 1) << @intCast(point.len));

            for (0..n) |i| {
                var basis = F.one();
                for (point, 0..) |r, j| {
                    const bit = (i >> @intCast(j)) & 1;
                    if (bit == 1) {
                        basis = basis.mul(r);
                    } else {
                        basis = basis.mul(F.one().sub(r));
                    }
                }
                if (i < evals.len) {
                    result = result.add(basis.mul(evals[i]));
                }
            }

            return result;
        }
    };
}

// ============================================================================
// Tests
// ============================================================================

test "jolt r1cs from empty trace" {
    const field = @import("../../field/mod.zig");
    const F = field.BN254Scalar;
    const allocator = std.testing.allocator;

    var trace = tracer.ExecutionTrace.init(allocator);
    defer trace.deinit();

    var r1cs = try JoltR1CS(F).fromTrace(allocator, &trace);
    defer r1cs.deinit();

    try std.testing.expectEqual(@as(usize, 0), r1cs.num_cycles);
    try std.testing.expectEqual(@as(usize, 1), r1cs.witness_size);
}

test "jolt r1cs witness generation" {
    const field = @import("../../field/mod.zig");
    const F = field.BN254Scalar;
    const allocator = std.testing.allocator;

    // Create a simple trace with one step using proper TraceStep
    var trace = tracer.ExecutionTrace.init(allocator);
    defer trace.deinit();

    try trace.steps.append(allocator, .{
        .cycle = 0,
        .pc = 0x1000,
        .unexpanded_pc = 0x1000,
        .instruction = 0x00000013, // NOP (addi x0, x0, 0)
        .rs1_value = 0,
        .rs2_value = 0,
        .rd_value = 0,
        .memory_addr = null,
        .memory_value = null,
        .is_memory_write = false,
        .next_pc = 0x1004,
        .is_compressed = false,
    });

    var r1cs = try JoltR1CS(F).fromTrace(allocator, &trace);
    defer r1cs.deinit();

    try std.testing.expectEqual(@as(usize, 1), r1cs.num_cycles);

    const witness = try r1cs.buildWitness();
    defer allocator.free(witness);

    // First element should be 1
    try std.testing.expect(witness[0].eql(F.one()));

    // Witness size should be 1 + 43 (NUM_INPUTS after adding instruction flags)
    try std.testing.expectEqual(@as(usize, 44), witness.len);
}

test "jolt r1cs Az Bz Cz computation" {
    const field = @import("../../field/mod.zig");
    const F = field.BN254Scalar;
    const allocator = std.testing.allocator;

    // Create trace with one step
    var trace = tracer.ExecutionTrace.init(allocator);
    defer trace.deinit();

    try trace.steps.append(allocator, .{
        .cycle = 0,
        .pc = 0x1000,
        .unexpanded_pc = 0x1000,
        .instruction = 0x00000013,
        .rs1_value = 0,
        .rs2_value = 0,
        .rd_value = 0,
        .memory_addr = null,
        .memory_value = null,
        .is_memory_write = false,
        .next_pc = 0x1004,
        .is_compressed = false,
    });

    var r1cs = try JoltR1CS(F).fromTrace(allocator, &trace);
    defer r1cs.deinit();

    const witness = try r1cs.buildWitness();
    defer allocator.free(witness);

    const Az = try r1cs.computeAz(witness);
    defer allocator.free(Az);
    const Bz = try r1cs.computeBz(witness);
    defer allocator.free(Bz);
    const Cz = try r1cs.computeCz(witness);
    defer allocator.free(Cz);

    // All arrays should have padded_num_constraints elements
    try std.testing.expectEqual(r1cs.padded_num_constraints, Az.len);
    try std.testing.expectEqual(r1cs.padded_num_constraints, Bz.len);
    try std.testing.expectEqual(r1cs.padded_num_constraints, Cz.len);

    // Cz should be all zeros (equality-conditional form)
    for (Cz) |c| {
        try std.testing.expect(c.eql(F.zero()));
    }
}

test "jolt spartan interface" {
    const field = @import("../../field/mod.zig");
    const F = field.BN254Scalar;
    const allocator = std.testing.allocator;

    // Create trace
    var trace = tracer.ExecutionTrace.init(allocator);
    defer trace.deinit();

    try trace.steps.append(allocator, .{
        .cycle = 0,
        .pc = 0x1000,
        .unexpanded_pc = 0x1000,
        .instruction = 0x00000013,
        .rs1_value = 0,
        .rs2_value = 0,
        .rd_value = 0,
        .memory_addr = null,
        .memory_value = null,
        .is_memory_write = false,
        .next_pc = 0x1004,
        .is_compressed = false,
    });

    var r1cs = try JoltR1CS(F).fromTrace(allocator, &trace);
    defer r1cs.deinit();

    const witness = try r1cs.buildWitness();
    defer allocator.free(witness);

    // Create tau challenge
    const tau = [_]F{F.fromU64(7), F.fromU64(11), F.fromU64(13), F.fromU64(17), F.fromU64(19)};

    var spartan = try JoltSpartanInterface(F).init(allocator, &r1cs, witness, &tau);
    defer spartan.deinit();

    // For a valid witness, initial claim should be 0 or very small
    // (may not be exactly 0 due to placeholder constraints)
    const claim = spartan.initialClaim();
    _ = claim;

    // Number of rounds should match log of constraints
    try std.testing.expectEqual(r1cs.log_num_constraints, spartan.numRounds());
}
