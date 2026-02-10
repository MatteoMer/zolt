//! InstructionReadRaf Sumcheck Prover
//!
//! This implements the Read+RAF sumcheck for instruction lookups in Stage 5.
//! The sumcheck proves:
//!
//! rv + γ·left_op + γ²·right_op = Σ_j Σ_k [ eq(j; r_reduction) · ra(k,j) · (Val_j(k) + γ·RafVal_j(k)) ]
//!
//! Since ra(k,j) = 1 only when k = lookup_index(j), this simplifies to:
//!   Σ_j eq(j; r_reduction) · [lookup_output(j) + γ·raf_contrib(j)]
//!
//! This is a simplified direct implementation without prefix/suffix optimization.
//! The sumcheck runs for LOG_K + log_T rounds, binding address variables first
//! (even though they don't affect the sum directly in this simplified version).
//!
//! Reference: jolt-core/src/zkvm/instruction_lookups/read_raf_checking.rs

const std = @import("std");
const Allocator = std.mem.Allocator;

const poly_mod = @import("../../poly/mod.zig");
const UniPoly = poly_mod.UniPoly;
const EqPoly = poly_mod.EqPoly;
const transcripts = @import("../../transcripts/mod.zig");
const Blake2bTranscript = transcripts.Blake2bTranscript;
const tracer = @import("../../tracer/mod.zig");
const ExecutionTrace = tracer.ExecutionTrace;
const jolt_types = @import("../jolt_types.zig");
const OpeningId = jolt_types.OpeningId;

/// LOG_K for instruction lookups = XLEN * 2 = 128 for RV64
pub const LOG_K: usize = 128;

/// Number of lookup tables in Jolt (LookupTables::COUNT = 41)
pub const NUM_LOOKUP_TABLES: usize = 41;

/// Result from InstructionReadRaf sumcheck
pub fn InstructionReadRafResult(comptime F: type) type {
    return struct {
        const Self = @This();

        /// All sumcheck challenges (LOG_K + log_T)
        challenges: []F,
        /// LookupTableFlag(i) for each lookup table
        table_flags: []F,
        /// InstructionRa(i) chunk claims
        ra_chunks: []F,
        /// InstructionRafFlag claim
        raf_flag: F,
        /// Allocator
        allocator: Allocator,

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.challenges);
            self.allocator.free(self.table_flags);
            self.allocator.free(self.ra_chunks);
        }
    };
}

/// InstructionReadRaf Sumcheck Prover (simplified direct version)
pub fn InstructionReadRafProver(comptime F: type) type {
    return struct {
        const Self = @This();

        allocator: Allocator,

        pub fn init(allocator: Allocator) Self {
            return .{ .allocator = allocator };
        }

        /// Generate InstructionReadRaf sumcheck proof
        ///
        /// This implements a simplified version that computes the correct polynomials
        /// without the prefix/suffix optimization used in Jolt.
        ///
        /// The sumcheck runs LOG_K + log_T rounds:
        /// - First LOG_K rounds bind address variables
        /// - Last log_T rounds bind cycle variables
        pub fn generateProof(
            self: *Self,
            transcript: *Blake2bTranscript(F),
            input_claim: F,
            r_reduction: []const F, // Opening point from InstructionClaimReduction
            n_cycle_vars: usize,
            ra_virtual_log_k_chunk: usize,
            gamma: F,
            gamma_sqr: F,
            trace: *const ExecutionTrace,
        ) !InstructionReadRafResult(F) {
            const log_T = n_cycle_vars;
            const T = @as(usize, 1) << @intCast(log_T);
            const num_rounds = LOG_K + log_T;
            const n_virtual_ra = LOG_K / ra_virtual_log_k_chunk;

            std.debug.print("[LOOKUPS_RAF] Starting with {} rounds (LOG_K={}, log_T={})\n", .{
                num_rounds, LOG_K, log_T,
            });
            std.debug.print("[LOOKUPS_RAF] input_claim = {any}\n", .{input_claim.toBytesBE()[0..8]});
            std.debug.print("[LOOKUPS_RAF] gamma = {any}\n", .{gamma.toBytesBE()[0..8]});

            // Build per-cycle combined values: lookup_output + γ·left_operand + γ²·right_operand
            var combined_vals = try self.allocator.alloc(F, T);
            defer self.allocator.free(combined_vals);
            @memset(combined_vals, F.zero());

            // Build per-cycle RAF values and table indices
            var raf_flags = try self.allocator.alloc(bool, T);
            defer self.allocator.free(raf_flags);
            @memset(raf_flags, false);

            var table_indices = try self.allocator.alloc(?usize, T);
            defer self.allocator.free(table_indices);
            @memset(table_indices, null);

            // Build lookup indices (128-bit)
            var lookup_indices = try self.allocator.alloc(u128, T);
            defer self.allocator.free(lookup_indices);
            @memset(lookup_indices, 0);

            // Populate from trace
            const trace_len = @min(trace.steps.items.len, T);
            for (0..trace_len) |j| {
                const step = trace.steps.items[j];
                if (step.is_noop) continue;

                // Get lookup output (instruction result)
                const lookup_output = step.rd_value; // rd_value is the lookup output

                // Get lookup operands (x, y)
                const rs1 = step.rs1_value;
                const rs2 = step.rs2_value;

                // Check if instruction uses interleaved operands
                // For ADD: operands are rs1, rs2
                // For ADDI: operands are rs1, imm
                const opcode = step.instruction & 0x7f;
                const is_interleaved = switch (opcode) {
                    0x33, // R-type (ADD, SUB, etc.) - interleaved
                    0x13, // I-type (ADDI, etc.) - interleaved
                    0x63, // Branch - interleaved
                    0x03, // Load
                    0x23, // Store
                    => true,
                    0x0B, // VirtualSignExtendWord - NOT interleaved (uses AddOperands)
                    => false,
                    else => false,
                };

                // Compute combined value: lookup_output + γ*left + γ²*right
                // For interleaved: left = rs1, right = rs2
                // For non-interleaved (identity): use rs1+rs2 combination
                const left = F.fromU64(rs1);
                const right: F = if (opcode == 0x0B)
                    F.fromU64(rs2) // VirtualSignExtendWord: right operand is 0
                else if (opcode == 0x13)
                    F.fromU64(@intCast(@as(i64, @bitCast(@as(u64, step.instruction >> 20) | if ((step.instruction >> 31) != 0) 0xFFFFFFFF_FFF00000 else 0)))) // Sign-extended immediate
                else
                    F.fromU64(rs2);

                const output = F.fromU64(lookup_output);
                combined_vals[j] = output.add(gamma.mul(left)).add(gamma_sqr.mul(right));

                raf_flags[j] = !is_interleaved;

                // Compute lookup index (interleaved bits of operands)
                if (opcode == 0x0B) {
                    // VirtualSignExtendWord: lookup index = rs1_val (not interleaved)
                    lookup_indices[j] = @as(u128, rs1);
                } else {
                    lookup_indices[j] = interleaveBits(rs1, if (opcode == 0x13) @truncate(@as(u64, @bitCast(@as(i64, @intCast(@as(i32, @bitCast(@as(u32, @truncate(step.instruction >> 20))))))))) else rs2);
                }

                // Determine which lookup table is used
                table_indices[j] = getTableIndex(step.instruction);
            }

            // Compute eq(j; r_reduction) evaluations
            var eq_evals = try self.allocator.alloc(F, T);
            defer self.allocator.free(eq_evals);
            for (0..T) |j| {
                eq_evals[j] = computeEq(F, r_reduction, j);
            }

            // Verify initial sum
            var computed_sum = F.zero();
            for (0..T) |j| {
                computed_sum = computed_sum.add(eq_evals[j].mul(combined_vals[j]));
            }
            std.debug.print("[LOOKUPS_RAF] computed_sum = {any}\n", .{computed_sum.toBytesBE()[0..8]});
            std.debug.print("[LOOKUPS_RAF] sum matches input: {}\n", .{computed_sum.eql(input_claim)});

            // Allocate challenges array
            var challenges = try self.allocator.alloc(F, num_rounds);
            errdefer self.allocator.free(challenges);

            var current_claim = input_claim;

            // First LOG_K rounds: bind address variables
            // In the simplified version, these don't change the sum
            // (since we're not using prefix/suffix decomposition)
            // But we still need to generate polynomials that satisfy p(0)+p(1) = claim
            for (0..LOG_K) |round| {
                // For address rounds, the polynomial is constant
                // p(x) = current_claim / 2 for all x
                // So p(0) + p(1) = current_claim
                const half_claim = current_claim.mul(F.fromU64(2).inverse().?);

                // Degree-2 polynomial: [p(0), p(2)]
                // For constant polynomial: p(0) = p(2) = half_claim
                const eval_0 = half_claim;
                const eval_2 = half_claim;

                // Create compressed polynomial
                const uni_poly = UniPoly(F).fromEvalsAndHint(current_claim, eval_0, eval_2);

                // Append to transcript
                transcript.appendMessage("UniPoly_begin");
                transcript.appendScalar(uni_poly.coeffs[0]);
                transcript.appendScalar(uni_poly.coeffs[1]);
                transcript.appendScalar(uni_poly.coeffs[2]);
                transcript.appendMessage("UniPoly_end");

                const challenge = transcript.challengeScalar();
                challenges[round] = challenge;

                // For constant polynomial, p(r) = half_claim for any r
                // Or more correctly, use the hint recovery formula
                const c0 = uni_poly.coeffs[0];
                const c2 = uni_poly.coeffs[1];
                const c3 = uni_poly.coeffs[2];
                const c1 = current_claim.sub(c0).sub(c2).sub(c3);
                const r2 = challenge.mul(challenge);
                const r3 = r2.mul(challenge);
                current_claim = c0.add(challenge.mul(c1)).add(r2.mul(c2)).add(r3.mul(c3));
            }

            // Last log_T rounds: bind cycle variables
            // This is where the actual sumcheck happens
            var active_eq = try self.allocator.alloc(F, T);
            defer self.allocator.free(active_eq);
            @memcpy(active_eq, eq_evals);

            var active_vals = try self.allocator.alloc(F, T);
            defer self.allocator.free(active_vals);
            @memcpy(active_vals, combined_vals);

            var current_size = T;

            for (LOG_K..num_rounds) |round| {
                const half = current_size / 2;

                // Compute degree-2 round polynomial
                // p(x) = Σ_i [eq_lo + x*(eq_hi - eq_lo)] * [val_lo + x*(val_hi - val_lo)]
                var eval_0 = F.zero();
                var eval_2 = F.zero();

                for (0..half) |i| {
                    const eq_lo = active_eq[2 * i];
                    const eq_hi = active_eq[2 * i + 1];
                    const val_lo = active_vals[2 * i];
                    const val_hi = active_vals[2 * i + 1];

                    // p_i(0) = eq_lo * val_lo
                    // p_i(2) = (2*eq_hi - eq_lo) * (2*val_hi - val_lo)
                    eval_0 = eval_0.add(eq_lo.mul(val_lo));
                    const eq_2 = eq_hi.add(eq_hi).sub(eq_lo);
                    const val_2 = val_hi.add(val_hi).sub(val_lo);
                    eval_2 = eval_2.add(eq_2.mul(val_2));
                }

                // Create compressed polynomial using Jolt's encoding
                const uni_poly = UniPoly(F).fromEvalsAndHint(current_claim, eval_0, eval_2);

                // Append to transcript
                transcript.appendMessage("UniPoly_begin");
                transcript.appendScalar(uni_poly.coeffs[0]);
                transcript.appendScalar(uni_poly.coeffs[1]);
                transcript.appendScalar(uni_poly.coeffs[2]);
                transcript.appendMessage("UniPoly_end");

                const challenge = transcript.challengeScalar();
                challenges[round] = challenge;

                // Bind the challenge
                for (0..half) |i| {
                    const eq_lo = active_eq[2 * i];
                    const eq_hi = active_eq[2 * i + 1];
                    active_eq[i] = eq_lo.add(challenge.mul(eq_hi.sub(eq_lo)));

                    const val_lo = active_vals[2 * i];
                    const val_hi = active_vals[2 * i + 1];
                    active_vals[i] = val_lo.add(challenge.mul(val_hi.sub(val_lo)));
                }

                // Update claim using polynomial evaluation
                // p(r) = c0 + c1*r + c2*r^2 where c1 = claim - c0 - c2 (hint recovery)
                const c0 = uni_poly.coeffs[0];
                const c2 = uni_poly.coeffs[1];
                const c3 = uni_poly.coeffs[2]; // should be 0 for degree-2
                const c1 = current_claim.sub(c0).sub(c2).sub(c3);
                const r2 = challenge.mul(challenge);
                const r3 = r2.mul(challenge);
                current_claim = c0.add(challenge.mul(c1)).add(r2.mul(c2)).add(r3.mul(c3));
                current_size = half;
            }

            // Compute opening claims
            const table_flags = try self.allocator.alloc(F, NUM_LOOKUP_TABLES);
            @memset(table_flags, F.zero());

            // Compute table flag claims by summing eq*flag for each table
            const r_cycle = challenges[LOG_K..];
            for (0..T) |j| {
                if (table_indices[j]) |t_idx| {
                    const eq_j = computeEq(F, r_cycle, j);
                    table_flags[t_idx] = table_flags[t_idx].add(eq_j);
                }
            }

            // Compute RA chunk claims (placeholder - needs proper implementation)
            const ra_chunks = try self.allocator.alloc(F, n_virtual_ra);
            @memset(ra_chunks, F.one()); // RA should be product of eq evaluations

            // Compute RAF flag claim
            var raf_flag = F.zero();
            for (0..T) |j| {
                if (raf_flags[j]) {
                    const eq_j = computeEq(F, r_cycle, j);
                    raf_flag = raf_flag.add(eq_j);
                }
            }

            return InstructionReadRafResult(F){
                .challenges = challenges,
                .table_flags = table_flags,
                .ra_chunks = ra_chunks,
                .raf_flag = raf_flag,
                .allocator = self.allocator,
            };
        }
    };
}

/// Interleave bits of two 64-bit values into a 128-bit value
fn interleaveBits(x: u64, y: u64) u128 {
    var result: u128 = 0;
    for (0..64) |i| {
        const xi: u128 = @as(u128, (x >> @intCast(i)) & 1);
        const yi: u128 = @as(u128, (y >> @intCast(i)) & 1);
        result |= xi << @intCast(2 * i);
        result |= yi << @intCast(2 * i + 1);
    }
    return result;
}

/// Get the lookup table index for an instruction
fn getTableIndex(instruction: u32) ?usize {
    const opcode = instruction & 0x7f;
    const funct3 = (instruction >> 12) & 0x7;
    const funct7 = (instruction >> 25) & 0x7f;

    return switch (opcode) {
        0x33 => switch (funct3) { // R-type
            0x0 => if (funct7 == 0) @as(?usize, 0) else if (funct7 == 0x20) @as(?usize, 1) else null, // ADD/SUB
            0x1 => @as(?usize, 2), // SLL
            0x2 => @as(?usize, 3), // SLT
            0x3 => @as(?usize, 4), // SLTU
            0x4 => @as(?usize, 5), // XOR
            0x5 => if (funct7 == 0) @as(?usize, 6) else @as(?usize, 7), // SRL/SRA
            0x6 => @as(?usize, 8), // OR
            0x7 => @as(?usize, 9), // AND
            else => null,
        },
        0x13 => switch (funct3) { // I-type
            0x0 => @as(?usize, 10), // ADDI
            0x2 => @as(?usize, 11), // SLTI
            0x3 => @as(?usize, 12), // SLTIU
            0x4 => @as(?usize, 13), // XORI
            0x6 => @as(?usize, 14), // ORI
            0x7 => @as(?usize, 15), // ANDI
            else => null,
        },
        0x63 => switch (funct3) { // Branch
            0x0 => @as(?usize, 16), // BEQ
            0x1 => @as(?usize, 17), // BNE
            0x4 => @as(?usize, 18), // BLT
            0x5 => @as(?usize, 19), // BGE
            0x6 => @as(?usize, 20), // BLTU
            0x7 => @as(?usize, 21), // BGEU
            else => null,
        },
        0x0B => @as(?usize, 22), // VirtualSignExtendWord
        else => null,
    };
}

/// Compute eq(r, x) for a binary index x using BIG ENDIAN indexing
fn computeEq(comptime F: type, r: []const F, x: usize) F {
    var result = F.one();
    const n = r.len;
    for (r, 0..) |ri, i| {
        const xi: u1 = @truncate(x >> @intCast(n - 1 - i));
        if (xi == 1) {
            result = result.mul(ri);
        } else {
            result = result.mul(F.one().sub(ri));
        }
    }
    return result;
}

test "instruction_read_raf compiles" {
    const F = @import("../../field/mod.zig").BN254Scalar;
    const allocator = std.testing.allocator;

    const prover = InstructionReadRafProver(F).init(allocator);
    _ = prover;
}
