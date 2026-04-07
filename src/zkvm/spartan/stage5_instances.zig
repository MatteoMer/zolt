//! Stage 5 Sumcheck Instance Helpers
//!
//! Contains the per-instance compute/bind round polynomial functions
//! and shared helper utilities extracted from stage5_prover.zig.
//!
//! Instance prover helpers:
//! - RegistersValEvaluation: computeRegsValRoundPoly, bindRegsValChallenge
//! - LookupsReadRaf: computeLookupsRoundPoly, bindLookupsChallenge,
//!   computeLookupsRoundPolyWithRa, bindLookupsCycleChallengeWithRa
//! - Shared: bindSinglePolynomial
//!
//! Trace processing and EQ/LT helpers used during initialization.

const std = @import("std");

const zkvm_debug = @import("../debug.zig");
const dbg = zkvm_debug.dbg;
const debug_verbose = zkvm_debug.verbose;

const Allocator = std.mem.Allocator;
const ThreadPool = @import("zolt_pool").ThreadPool;
const GpuPolyOps = @import("zolt_arith").gpu.GpuPolyOps;

const poly_mod = @import("zolt_arith").poly;
const LtPolynomial = @import("zolt_arith").poly.lt_poly.LtPolynomial;
const UniPoly = poly_mod.UniPoly;
const tracer = @import("../../tracer/mod.zig");
const UnreducedProductAccum = @import("zolt_arith").field.UnreducedProductAccum;

/// Comptime-generic namespace for Stage 5 instance helpers.
/// These are the compute/bind round polynomial methods and shared utilities
/// used by Stage5BatchedProver during the batched sumcheck.
pub fn Helpers(comptime F: type) type {
    return struct {
        /// Per-cycle instruction decode for combined_vals, lookup indices, table assignments.
        /// Extracted from the init trace loop for parallel dispatch.
        /// IMPORTANT: This is a hot function called T times in parallel.
        pub fn processTraceCycleCombined(
            step: tracer.TraceStep,
            j: usize,
            combined: []F,
            idx_lo: []u64,
            idx_hi: []u64,
            tbl_ids: []i8,
            is_id: []bool,
            g_raf: F,
            g_raf2: F,
            idx_u128: ?[]u128,
            is_interleaved_out: ?[]bool,
        ) void {
            // NOTE: Do NOT skip NOOPs here! In Jolt, NOOPs (ADDI x0,x0,0) are valid
            // instructions with lookup_table = RangeCheck and is_identity_path = true.
            // Skipping them causes cycle_table_indices and cycle_is_identity_path to be
            // wrong, which corrupts Q arrays, rematerialization, and opening claims.

            const instr = step.instruction;
            const opcode = instr & 0x7f;
            const funct3: u3 = @truncate((instr >> 12) & 0x7);
            const funct7: u7 = @truncate(instr >> 25);

            // Determine left_op, right_op, and lookup_output based on instruction type.
            // This MUST match the verification loop / R1CS witness exactly.
            // Use field arithmetic with signedI64ToField for signed immediates.
            var left_op: F = undefined;
            var right_op: F = undefined;
            var lookup_output: F = undefined;

            // First compute left_input and right_input (same as R1CS)
            const left_is_rs1: bool = switch (opcode) {
                0x33, 0x3b, 0x23, 0x63, 0x13, 0x03, 0x67, 0x1b, 0x0B, 0x2B, 0x6B => true,
                0x5B => (funct3 == 0 or funct3 == 5), // VirtualSRLI/VirtualSRAI only; VirtualHostIO does NOT read rs1
                0x22 => true, // VirtualAssertEQ: left = rs1
                0x42 => true, // VirtualZeroExtendWord: left = rs1
                0x62 => true, // VirtualAssertValidUnsignedRemainder: left = rs1
                // 0x02 (VirtualAdvice): left_is_rs1 = false (instruction_inputs = (0,0))
                else => false,
            };
            const left_is_pc: bool = switch (opcode) {
                0x17, 0x6f => true,
                else => false,
            };
            const right_is_rs2: bool = switch (opcode) {
                0x33, 0x63, 0x3b => true,
                0x22 => (funct3 == 0 or funct3 == 1), // VirtualAssertEQ/ValidDiv0: right = rs2; alignment: right = imm
                0x62 => true, // VirtualAssertValidUnsignedRemainder: right = rs2
                0x5B => (funct3 == 0 or funct3 == 5) and step.rs2_read, // VirtualSRL/VirtualSRA R-type only; VirtualHostIO: false
                else => false,
            };
            const right_is_imm: bool = switch (opcode) {
                0x13, 0x03, 0x67, 0x23, 0x37, 0x17, 0x6f, 0x1b, 0x0B, 0x2B, 0x6B => true,
                0x22 => (funct3 == 2 or funct3 == 3), // alignment assertions: right = imm
                0x5B => (funct3 == 0 or funct3 == 5) and !step.rs2_read, // I-type VirtualSRLI/VirtualSRAI only; VirtualHostIO: false
                else => false,
            };

            // For identity-path AddOperands instructions (ADDI, ADDIW, JAL, JALR, VirtualSignExtendWord),
            // use UNSIGNED u64 immediate to match Jolt's to_lookup_operands() u128 arithmetic.
            // This ensures RightInstructionInput matches between R1CS, Stage 3, and Stage 5.
            const is_identity_add_imm: bool = switch (opcode) {
                0x13 => funct3 == 0, // ADDI
                0x1b => funct3 == 0, // ADDIW
                0x0B => true, // VirtualSignExtendWord
                0x6f => true, // JAL
                0x67 => true, // JALR
                else => false,
            };
            const imm_val = if (opcode == 0x2B) blk: {
                if (funct3 == 0) {
                    // VirtualMULI: IMM = multiplier = 1 << shamt
                    const shamt_raw2: u32 = instr >> 20;
                    const shamt2: u6 = @truncate(shamt_raw2 & 0x3F);
                    const multiplier2: u64 = @as(u64, 1) << shamt2;
                    break :blk F.fromU64(multiplier2);
                } else {
                    // VirtualPow2/VirtualShiftRightBitmask: IMM = 0
                    break :blk F.zero();
                }
            } else if (opcode == 0x5B and (funct3 == 0 or funct3 == 5)) blk: {
                // VirtualSRLI/VirtualSRAI/VirtualSRL/VirtualSRA only (not VirtualHostIO)
                if (step.rs2_read) {
                    // VirtualSRL/VirtualSRA R-type: no immediate (rs2 used instead)
                    break :blk F.zero();
                } else {
                    // VirtualSRLI/VirtualSRAI I-type: IMM = bitmask computed from total shift
                    const total_shift_raw2: u32 = instr >> 20;
                    const total_shift2: u7 = @truncate(total_shift_raw2 & 0x3F);
                    const ones2: u128 = (@as(u128, 1) << @intCast(64 - @as(u8, total_shift2))) - 1;
                    const bitmask2: u64 = @truncate(ones2 << total_shift2);
                    break :blk F.fromU64(bitmask2);
                }
            } else if (opcode == 0x6B) blk: {
                // VirtualROTRI/VirtualROTRIW: IMM = bitmask computed from rotation
                const rot_raw: u32 = instr >> 20;
                if (funct3 == 0) {
                    // VirtualROTRI: 64-bit rotation
                    const rot: u7 = @truncate(rot_raw & 0x3F);
                    const bitmask_6b: u64 = if (rot == 0) 0xFFFFFFFF_FFFFFFFF else blk2: {
                        const ones_6b: u128 = (@as(u128, 1) << @intCast(64 - @as(u8, rot))) - 1;
                        break :blk2 @truncate(ones_6b << @intCast(rot));
                    };
                    break :blk F.fromU64(bitmask_6b);
                } else {
                    // VirtualROTRIW: 32-bit rotation
                    const rot_w: u6 = @truncate(rot_raw & 0x1F);
                    const bitmask_6b_w: u64 = if (rot_w == 0) 0xFFFFFFFF else ((@as(u64, 1) << @intCast(32 - @as(u8, rot_w))) - 1) << @intCast(rot_w);
                    break :blk F.fromU64(bitmask_6b_w);
                }
            } else if (opcode == 0x22 and (funct3 == 2 or funct3 == 3)) blk: {
                // VirtualAssertHalfwordAlignment/WordAlignment: SIGNED IMM encoding
                // Must match R1CS witness (now signed) and Jolt verifier val_poly
                const assert_imm_raw: u32 = @truncate(instr >> 20);
                const assert_imm_signed: i64 = @as(i64, @as(i32, @bitCast(assert_imm_raw << 20)) >> 20);
                if (assert_imm_signed < 0) {
                    break :blk F.fromU64(@intCast(-assert_imm_signed)).neg();
                } else {
                    break :blk F.fromU64(@intCast(assert_imm_signed));
                }
            } else if (is_identity_add_imm) blk: {
                // Use unsigned u64 representation (two's complement) for the immediate.
                // E.g., imm=-1 → F(0xFFFFFFFFFFFFFFFF) instead of F(p-1).
                break :blk F.fromU64(computeUnsignedImmediate(instr));
            } else computeImmediate(instr);

            var left_input: F = F.zero();
            if (left_is_rs1) left_input = F.fromU64(step.rs1_value);
            // FIX: Use unexpanded_pc (raw RISC-V address) not pc (expanded bytecode index)
            // This matches R1CS constraints.zig and Jolt's instruction_input.rs
            if (left_is_pc) left_input = F.fromU64(step.unexpanded_pc);

            var right_input: F = F.zero();
            if (right_is_rs2) right_input = F.fromU64(step.rs2_value);
            if (right_is_imm) right_input = imm_val;

            // Compute LookupOutput = materialize_entry(lookup_index) for the instruction's table.
            // For identity-path (AddOperands/SubtractOperands/MultiplyOperands):
            //   lookup_output = materialize_entry(right_op_raw) = F.fromU64(right_op_raw) for RangeCheck.
            // For interleaved path:
            //   lookup_output = materialize_entry(interleave(left, right)) from the assigned table.
            // Special cases: JAL, JALR, Branch have their own formulas.
            //
            // NOTE: This is computed AFTER the lookup index section below, but we set a
            // preliminary value here and may override it.
            switch (opcode) {
                0x6f => { // JAL: LookupOutput = PC + imm
                    lookup_output = left_input.add(right_input);
                },
                0x67 => { // JALR: LookupOutput = (rs1 + imm) & ~1
                    const target = left_input.add(right_input);
                    const target_u64 = target.toU64() & ~@as(u64, 1);
                    lookup_output = F.fromU64(target_u64);
                },
                0x63 => { // Branch: LookupOutput = condition result (0 or 1)
                    const result: u64 = switch (funct3) {
                        0x0 => if (step.rs1_value == step.rs2_value) 1 else 0,
                        0x1 => if (step.rs1_value != step.rs2_value) 1 else 0,
                        0x4 => if (@as(i64, @bitCast(step.rs1_value)) < @as(i64, @bitCast(step.rs2_value))) 1 else 0,
                        0x5 => if (@as(i64, @bitCast(step.rs1_value)) >= @as(i64, @bitCast(step.rs2_value))) 1 else 0,
                        0x6 => if (step.rs1_value < step.rs2_value) 1 else 0,
                        0x7 => if (step.rs1_value >= step.rs2_value) 1 else 0,
                        else => 0,
                    };
                    lookup_output = F.fromU64(result);
                },
                0x22, 0x62 => {
                    // VirtualAssertEQ and VirtualAssertValidUnsignedRemainder: Assert instructions
                    // LookupOutput = 1 (assertion passed). Matches R1CS computeLookupOutput.
                    lookup_output = F.one();
                },
                else => {
                    // Default: rd_value (will be overridden for ADDIW/ADDW/SUBW below)
                    lookup_output = F.fromU64(step.rd_value);
                },
            }

            // Compute LeftLookupOperand and RightLookupOperand
            switch (opcode) {
                0x33 => { // R-type
                    if (funct7 == 0x01) {
                        // M-extension
                        if (funct3 == 0x0) { // MUL: MultiplyOperands
                            left_op = F.zero();
                            right_op = left_input.mul(right_input); // Product
                        } else if (funct3 == 0x3) { // MULHU: MultiplyOperands
                            left_op = F.zero();
                            right_op = left_input.mul(right_input); // Product
                        } else {
                            // DIVU, REMU, MULHSU, etc.: interleaved
                            left_op = left_input;
                            right_op = right_input;
                        }
                    } else if (funct3 == 0x0 and funct7 == 0x20) {
                        // SUB: SubtractOperands, left=0, right=rs1-rs2+2^64
                        const two_pow_64 = F.fromBytes(&[_]u8{ 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 });
                        left_op = F.zero();
                        right_op = left_input.sub(right_input).add(two_pow_64);
                    } else if (funct3 == 0x0 and funct7 == 0x0) {
                        // ADD: AddOperands, left=0, right=rs1+rs2
                        left_op = F.zero();
                        right_op = left_input.add(right_input);
                    } else {
                        // XOR, AND, OR, SLT, SLTU, SRL, SRA: interleaved operands
                        left_op = left_input;
                        right_op = right_input;
                    }
                },
                0x13 => { // I-type ALU: only ADDI (funct3=0) uses AddOperands
                    // Other I-type ALU instructions (SLLI, SLTI, etc.) use interleaved operands
                    // Note: Jolt expands SLLI/etc to virtual instructions, but Zolt handles them directly
                    if (funct3 == 0) {
                        // ADDI: AddOperands
                        left_op = F.zero();
                        right_op = left_input.add(right_input);
                    } else {
                        // SLLI, SLTI, SLTIU, XORI, SRLI, SRAI, ORI, ANDI: interleaved
                        left_op = left_input;
                        right_op = right_input;
                    }
                },
                0x37 => { // LUI: AddOperands, left_input=0, right_input=imm
                    left_op = F.zero();
                    right_op = left_input.add(right_input);
                },
                0x17 => { // AUIPC: AddOperands, left_input=PC, right_input=imm
                    left_op = F.zero();
                    right_op = left_input.add(right_input);
                },
                0x6f => { // JAL: AddOperands, left_input=PC, right_input=imm
                    left_op = F.zero();
                    right_op = left_input.add(right_input);
                },
                0x67 => { // JALR: AddOperands, left_input=rs1, right_input=imm
                    left_op = F.zero();
                    right_op = left_input.add(right_input);
                },
                0x1b => { // I-type word ALU (ADDIW, SLLIW, SRLIW, SRAIW)
                    // Only ADDIW (funct3=0) uses AddOperands; others use interleaved
                    if (funct3 == 0) {
                        // ADDIW: AddOperands, left=0, right=rs1+imm
                        left_op = F.zero();
                        right_op = left_input.add(right_input);
                    } else {
                        // SLLIW, SRLIW, SRAIW: interleaved
                        left_op = left_input;
                        right_op = right_input;
                    }
                },
                0x3b => { // ADDW/SUBW/VirtualChangeDivisorW
                    // In Jolt, ADDW decomposes to ADD+VirtualSEW, SUBW to SUB+VirtualSEW.
                    // For Zolt's single-cycle model, match the first step's format.
                    if (funct3 == 0 and funct7 == 0) {
                        // ADDW: AddOperands, left=0, right=rs1+rs2
                        left_op = F.zero();
                        right_op = left_input.add(right_input);
                    } else if (funct3 == 0 and funct7 == 0x20) {
                        // SUBW: SubtractOperands, left=0, right=rs1-rs2+2^64
                        const two_pow_64 = F.fromBytes(&[_]u8{ 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 });
                        left_op = F.zero();
                        right_op = left_input.sub(right_input).add(two_pow_64);
                    } else if (funct3 == 6 and funct7 == 0x01) {
                        // VirtualChangeDivisorW: interleaved, left=rs1 as u32 as u64 (truncated), right=rs2
                        // Jolt's to_instruction_inputs: (rs1 as u32 as u64, rs2 as i128)
                        // to_lookup_operands: (rs1 as u32 as u64, rs2 as u64)
                        const rs1_lower32: u64 = step.rs1_value & 0xFFFFFFFF;
                        left_op = F.fromU64(rs1_lower32);
                        right_op = F.fromU64(step.rs2_value);
                    } else {
                        // Other 0x3b variants (not AddOperands/SubtractOperands)
                        left_op = left_input;
                        right_op = right_input;
                    }
                },
                0x0B => { // VirtualSignExtendWord: AddOperands, left=0, right=rs1
                    // Lookup operands: (0, rs1_val + 0) = (0, rs1_val)
                    left_op = F.zero();
                    right_op = left_input.add(right_input); // rs1 + 0 = rs1
                },
                0x2B => { // Virtual I-type: dispatch on funct3
                    if (funct3 == 0) {
                        // VirtualMULI: MultiplyOperands, left=0, right=rs1*imm
                        left_op = F.zero();
                        right_op = left_input.mul(right_input);
                    } else {
                        // VirtualPow2 (funct3=1), VirtualShiftRightBitmask (funct3=2): AddOperands
                        // Lookup operands: (0, rs1 + 0) = (0, rs1)
                        left_op = F.zero();
                        right_op = left_input.add(right_input); // rs1 + 0 = rs1
                    }
                },
                0x03 => { // Load: NOT AddOperands, left=rs1, right=imm
                    // R1CS witness sets: LeftLookupOperand=left_input, RightLookupOperand=right_input
                    left_op = left_input;
                    right_op = right_input;
                },
                0x23 => { // Store: NOT AddOperands, left=rs1, right=imm
                    // R1CS witness sets: LeftLookupOperand=left_input, RightLookupOperand=right_input
                    left_op = left_input;
                    right_op = right_input;
                },
                0x02 => { // VirtualAdvice: Advice flag (identity path)
                    // R1CS: LeftLookupOperand=0, RightLookupOperand=F.fromU128(rd_value)
                    // left_input=0, right_input=0 (no instruction inputs)
                    // The lookup operand is the advice oracle value (rd_value)
                    left_op = F.zero();
                    right_op = F.fromU128(@as(u128, step.rd_value));
                },
                0x22 => { // Virtual assert: dispatch on funct3
                    if (funct3 == 2 or funct3 == 3) {
                        // VirtualAssertHalfwordAlignment/WordAlignment: AddOperands
                        // Lookup operands: (0, rs1 + imm)
                        left_op = F.zero();
                        right_op = left_input.add(right_input); // rs1 + imm
                    } else {
                        // VirtualAssertEQ (funct3=0) / VirtualAssertValidDiv0 (funct3=1): interleaved
                        left_op = left_input;
                        right_op = right_input;
                    }
                },
                0x42 => { // VirtualZeroExtendWord: AddOperands flag (identity path)
                    // R1CS: LeftLookupOperand=0, RightLookupOperand=F.fromU128(rs1_value)
                    // AddOperands: left=0, right=left_input+right_input
                    // Here left_input=rs1, right_input=0, so right=rs1
                    left_op = F.zero();
                    right_op = F.fromU128(@as(u128, step.rs1_value));
                },
                0x62 => { // VirtualAssertValidUnsignedRemainder: Assert flag (interleaved)
                    // R1CS: LeftLookupOperand=left_input(=rs1), RightLookupOperand=right_input(=rs2)
                    left_op = left_input;
                    right_op = right_input;
                },
                else => {
                    // Default: NOT Add+Sub+Mul (includes 0x63 Branch)
                    left_op = left_input;
                    right_op = right_input;
                },
            }

            // Track which lookup table this cycle uses (for flag claims)
            const table_idx = getLookupTableIndex(opcode, funct3, funct7);
            tbl_ids[j] = table_idx;

            // For instructions without a lookup table (Load, Store, SLL, etc.):
            // All three must be zeroed to match the R1CS witness, which sets:
            //   LeftLookupOperand = 0, RightLookupOperand = 0, LookupOutput = 0
            // In Jolt, these instructions decompose into virtual sequences and never
            // appear as raw cycles, so the R1CS witness has all zeros for their operands.
            // The RAF contribution is handled by the global prefix-suffix polynomials
            // during address rounds, NOT by per-cycle combined_vals.
            if (table_idx < 0) {
                lookup_output = F.zero();
                left_op = F.zero();
                right_op = F.zero();
            }

            // combined_vals is rematerialized at round 128 using prefix checkpoint constants
            // (see lines 4382-4422), so skip the expensive field arithmetic here.
            // Only compute for debug verification.
            if (comptime debug_verbose) {
                combined[j] = lookup_output.add(g_raf.mul(left_op)).add(g_raf2.mul(right_op));
            }

            // Determine identity path (not interleaved) based on Jolt's flags:
            //   - AddOperands: ADD, ADDI, ADDIW, ADDW, LUI, AUIPC, JAL, JALR, Load, Store
            //   - SubtractOperands: SUB, SUBW
            //   - MultiplyOperands: MUL, MULHU
            // Identity path instructions use raw operand value as lookup index (NOT interleaved).
            // Interleaved path instructions use interleave_bits(left, right) as lookup index.
            const is_identity_path: bool = switch (opcode) {
                0x33 => blk: {
                    if (funct3 == 0 and funct7 == 0) break :blk true; // ADD (AddOperands)
                    if (funct3 == 0 and funct7 == 0x20) break :blk true; // SUB (SubtractOperands)
                    if (funct7 == 0x01 and funct3 == 0) break :blk true; // MUL (MultiplyOperands)
                    if (funct7 == 0x01 and funct3 == 3) break :blk true; // MULHU (MultiplyOperands)
                    break :blk false;
                },
                0x13 => (funct3 == 0), // ADDI (AddOperands)
                0x0B => true, // VirtualSignExtendWord (AddOperands)
                0x2B => true, // VirtualMULI/Pow2/ShiftRightBitmask: all identity path (MultiplyOperands or AddOperands)
                0x1b => (funct3 == 0), // ADDIW (AddOperands)
                0x3b => blk: {
                    if (funct3 == 0 and funct7 == 0) break :blk true; // ADDW (AddOperands)
                    if (funct3 == 0 and funct7 == 0x20) break :blk true; // SUBW (SubtractOperands)
                    break :blk false;
                },
                0x37 => true, // LUI (AddOperands)
                0x17 => true, // AUIPC (AddOperands)
                0x6f => true, // JAL (AddOperands)
                0x67 => true, // JALR (AddOperands)
                0x02 => true, // VirtualAdvice (Advice flag → identity path)
                0x42 => true, // VirtualZeroExtendWord (AddOperands → identity path)
                0x7B => true, // VirtualRev8W (AddOperands → identity path, single operand rs1)
                0x03 => false, // Load: uses (rs1, imm) format, NOT identity path
                0x23 => false, // Store: uses (rs1, imm) format, NOT identity path
                0x22 => (funct3 == 2 or funct3 == 3), // Alignment assertions: AddOperands (identity); AssertEQ/ValidDiv0: interleaved
                0x62 => false, // VirtualAssertValidUnsignedRemainder: interleaved (rs1, rs2)
                else => false,
            };
            is_id[j] = is_identity_path;

            // Compute lookup operands and index matching Jolt's to_lookup_operands/to_lookup_index.
            // For identity-path: left_op_raw=0, right_op_raw=computed_value, index=computed u128
            //   Jolt's to_lookup_index() for identity-path instructions returns the raw u128 result
            //   (NOT wrapped at 64 bits). E.g., ADD returns x as u128 + y as u64 as u128.
            // For interleaved-path: left_op_raw=rs1, right_op_raw=rs2, index=interleave(left,right)
            var left_op_raw: u64 = undefined;
            var right_op_raw: u64 = undefined;
            // lookup_idx_u128 holds the FULL u128 lookup index (not wrapped at u64)
            var lookup_idx_u128: u128 = undefined;

            if (is_identity_path) {
                left_op_raw = 0;
                // Compute lookup index in u128 to match Jolt's to_lookup_index()
                // Jolt returns the raw computation result, NOT wrapped at 64 bits.
                lookup_idx_u128 = switch (opcode) {
                    // ADD: index = rs1 as u128 + rs2 as u128
                    0x33 => blk128: {
                        if (funct3 == 0 and funct7 == 0) {
                            break :blk128 @as(u128, step.rs1_value) + @as(u128, step.rs2_value);
                        }
                        // SUB: index = rs1 as u128 + (2^64 - rs2 as u128)
                        if (funct3 == 0 and funct7 == 0x20) {
                            break :blk128 @as(u128, step.rs1_value) + (@as(u128, 1) << 64) - @as(u128, step.rs2_value);
                        }
                        // MUL: index = rs1 as u128 * rs2 as u128
                        if (funct7 == 0x01 and funct3 == 0) {
                            break :blk128 @as(u128, step.rs1_value) * @as(u128, step.rs2_value);
                        }
                        // MULHU: index = rs1 as u128 * rs2 as u128
                        if (funct7 == 0x01 and funct3 == 3) {
                            break :blk128 @as(u128, step.rs1_value) * @as(u128, step.rs2_value);
                        }
                        break :blk128 0;
                    },
                    // ADDW/SUBW: same computation as ADD/SUB
                    0x3b => blk128: {
                        if (funct3 == 0 and funct7 == 0) {
                            // ADDW: index = rs1 + rs2 (u128)
                            break :blk128 @as(u128, step.rs1_value) + @as(u128, step.rs2_value);
                        }
                        if (funct3 == 0 and funct7 == 0x20) {
                            // SUBW: index = rs1 + 2^64 - rs2 (u128)
                            break :blk128 @as(u128, step.rs1_value) + (@as(u128, 1) << 64) - @as(u128, step.rs2_value);
                        }
                        break :blk128 0;
                    },
                    // ADDI: index = rs1 + sign_ext(imm) (u128)
                    0x13 => blk128: {
                        const imm12_raw: u32 = @truncate(instr >> 20);
                        const imm_signed: i64 = @as(i64, @as(i32, @bitCast(imm12_raw << 20)) >> 20);
                        const imm_u64: u64 = @bitCast(imm_signed);
                        break :blk128 @as(u128, step.rs1_value) + @as(u128, imm_u64);
                    },
                    // ADDIW: index = rs1 + sign_ext(imm) (u128)
                    0x1b => blk128: {
                        const imm12_raw_w: u32 = @truncate(instr >> 20);
                        const imm_signed_w: i64 = @as(i64, @as(i32, @bitCast(imm12_raw_w << 20)) >> 20);
                        const imm_u64_w: u64 = @bitCast(imm_signed_w);
                        break :blk128 @as(u128, step.rs1_value) + @as(u128, imm_u64_w);
                    },
                    // LUI: index = sign_ext_32_to_64(imm) as u128
                    // Jolt sign-extends the U-type immediate via `as i32 as i64 as u64`
                    0x37 => blk128: {
                        const imm_u32: u32 = instr & 0xFFFFF000;
                        const imm_sext: u64 = @bitCast(@as(i64, @as(i32, @bitCast(imm_u32))));
                        break :blk128 @as(u128, imm_sext);
                    },
                    // AUIPC: index = pc + sign_ext_32_to_64(imm) (u128)
                    0x17 => blk128: {
                        const imm_u32: u32 = instr & 0xFFFFF000;
                        const imm_sext: u64 = @bitCast(@as(i64, @as(i32, @bitCast(imm_u32))));
                        break :blk128 @as(u128, step.unexpanded_pc) + @as(u128, imm_sext);
                    },
                    // JAL: index = pc + sign_ext(imm) (u128)
                    0x6f => blk128: {
                        const imm20: u32 = ((@as(u32, instr >> 31) & 1) << 19) |
                            ((@as(u32, instr >> 12) & 0xFF) << 11) |
                            ((@as(u32, instr >> 20) & 1) << 10) |
                            ((@as(u32, instr >> 21) & 0x3FF));
                        const imm_signed: i64 = @as(i64, @as(i32, @bitCast(imm20 << 12)) >> 11);
                        const imm_u64: u64 = @bitCast(imm_signed);
                        break :blk128 @as(u128, step.unexpanded_pc) + @as(u128, imm_u64);
                    },
                    // JALR: index = rs1 + sign_ext(imm) (u128)
                    0x67 => blk128: {
                        const imm12_raw: u32 = @truncate(instr >> 20);
                        const imm_signed: i64 = @as(i64, @as(i32, @bitCast(imm12_raw << 20)) >> 20);
                        const imm_u64: u64 = @bitCast(imm_signed);
                        break :blk128 @as(u128, step.rs1_value) + @as(u128, imm_u64);
                    },
                    // VirtualSignExtendWord: index = rs1 (the value to sign-extend)
                    0x0B => @as(u128, step.rs1_value),
                    // VirtualRev8W: index = rs1 (single operand, byte-swap-per-32-bit-half)
                    0x7B => @as(u128, step.rs1_value),
                    // VirtualMULI/Pow2/ShiftRightBitmask: dispatch on funct3
                    0x2B => blk128: {
                        if (funct3 == 0) {
                            // VirtualMULI: index = rs1 * multiplier (u128)
                            const shamt_raw3: u32 = instr >> 20;
                            const shamt3: u6 = @truncate(shamt_raw3 & 0x3F);
                            const multiplier3: u64 = @as(u64, 1) << shamt3;
                            break :blk128 @as(u128, step.rs1_value) * @as(u128, multiplier3);
                        } else {
                            // VirtualPow2/VirtualShiftRightBitmask: AddOperands, index = rs1 + 0 = rs1
                            break :blk128 @as(u128, step.rs1_value);
                        }
                    },
                    // VirtualAdvice: index = advice_value (rd_value) — Jolt's to_lookup_index returns second operand
                    0x02 => @as(u128, step.rd_value),
                    // VirtualZeroExtendWord: index = rs1 + 0 = rs1 — Jolt's to_lookup_operands returns (0, x+y) where y=0
                    0x42 => @as(u128, step.rs1_value),
                    // VirtualAssertHalfwordAlignment/WordAlignment (funct3=2,3): AddOperands, index = rs1 + imm (u128)
                    0x22 => blk128: {
                        // Wrapping u64 addition matching tracer's lookup index
                        const imm_u64_22 = computeUnsignedImmediate(instr);
                        break :blk128 @as(u128, step.rs1_value +% imm_u64_22);
                    },
                    else => 0,
                };
                // right_op_raw is the lower 64 bits of the lookup index (for R1CS witness compatibility)
                right_op_raw = @truncate(lookup_idx_u128);

                // CRITICAL: For identity-path instructions, right_op must be the FULL u128
                // lookup index as a field element, matching Jolt's to_lookup_operands() which
                // returns u128 results. This is consistent with the RAF decomposition which
                // uses the u128 index for the identity polynomial evaluation.
                //
                // The R1CS witness also uses u128 values (via computeU128LookupOperand),
                // ensuring consistency between Stage 2 claims and Stage 5 combined_vals.
                right_op = F.fromU128(lookup_idx_u128);
            } else {
                // Interleaved path: left=rs1, right=rs2 (or imm for I-type)
                // VirtualChangeDivisorW (0x3b/f3=6/f7=1): left = rs1 as u32 as u64 (truncated to 32 bits)
                left_op_raw = if (opcode == 0x3b and funct3 == 6 and funct7 == 0x01)
                    step.rs1_value & 0xFFFFFFFF
                else
                    step.rs1_value;
                right_op_raw = switch (opcode) {
                    0x33, 0x3b, 0x63 => step.rs2_value,
                    0x13 => blk: {
                        // I-type: right operand is sign-extended immediate (as u64)
                        const imm12_raw: u32 = @truncate(instr >> 20);
                        const imm_signed: i64 = @as(i64, @as(i32, @bitCast(imm12_raw << 20)) >> 20);
                        break :blk @as(u64, @bitCast(imm_signed));
                    },
                    0x5B => blk5b: {
                        // Only VirtualSRLI/VirtualSRAI (funct3=0/5) reach here;
                        // VirtualHostIO has no lookup table so table_idx < 0 zeros everything.
                        if (step.rs2_read) {
                            // VirtualSRL/VirtualSRA R-type: right operand is rs2
                            break :blk5b step.rs2_value;
                        } else {
                            // VirtualSRLI/VirtualSRAI I-type: right operand is bitmask computed from total shift
                            const ts_raw: u32 = instr >> 20;
                            const ts: u7 = @truncate(ts_raw & 0x3F);
                            const ones_5b: u128 = (@as(u128, 1) << @intCast(64 - @as(u8, ts))) - 1;
                            break :blk5b @truncate(ones_5b << ts);
                        }
                    },
                    0x6B => blk6b: {
                        // VirtualROTRI/VirtualROTRIW: right operand is bitmask from rotation
                        const rot_raw_6b: u32 = instr >> 20;
                        const funct3_6b: u3 = @truncate((instr >> 12) & 0x7);
                        if (funct3_6b == 0) {
                            // VirtualROTRI: 64-bit rotation
                            const rot_6b: u7 = @truncate(rot_raw_6b & 0x3F);
                            if (rot_6b == 0) break :blk6b @as(u64, 0xFFFFFFFF_FFFFFFFF);
                            const ones_6b: u128 = (@as(u128, 1) << @intCast(64 - @as(u8, rot_6b))) - 1;
                            break :blk6b @truncate(ones_6b << @intCast(rot_6b));
                        } else {
                            // VirtualROTRIW: 32-bit rotation
                            const rot_6b_w: u6 = @truncate(rot_raw_6b & 0x1F);
                            if (rot_6b_w == 0) break :blk6b @as(u64, 0xFFFFFFFF);
                            break :blk6b ((@as(u64, 1) << @intCast(32 - @as(u8, rot_6b_w))) - 1) << @intCast(rot_6b_w);
                        }
                    },
                    else => step.rs2_value,
                };
                lookup_idx_u128 = interleaveBits128(left_op_raw, right_op_raw);
            }

            // Use the computed u128 lookup index
            const lookup_idx: u128 = lookup_idx_u128;
            idx_lo[j] = @truncate(lookup_idx);
            idx_hi[j] = @truncate(lookup_idx >> 64);

            // CRITICAL FIX: Instructions without a lookup table should have
            // lookup_index = 0, matching Jolt where to_instruction_inputs() = (0, 0)
            // and interleave(0, 0) = 0.
            if (table_idx < 0) {
                idx_lo[j] = 0;
                idx_hi[j] = 0;
            }

            // NOTE: Do NOT override lookup_output with materializeTableEntry here!
            // The initial lookups_combined_vals must match the R1CS witness polynomials
            // (which use computeLookupOutput = rd_value for most instructions).
            // The address round prefix-suffix decomposition uses table MLEs independently
            // via Q arrays, and combined_vals are rematerialized at the phase transition
            // (init_log_t_rounds) using stored_table_values for the cycle rounds.

            // Merge: compute u128 lookup index and is_interleaved in the same pass
            if (idx_u128) |u128_out| {
                u128_out[j] = (@as(u128, idx_hi[j]) << 64) | idx_lo[j];
            }
            if (is_interleaved_out) |interleaved| {
                interleaved[j] = !is_id[j];
            }
        }
        // (generated by extracting the original sequential loop body)

        /// Compute immediate value from instruction, matching R1CS deriveImmediate
        /// Compute the immediate value as a field element, matching Jolt's per-format encoding.
        ///
        /// CRITICAL: The encoding depends on the RISC-V format type:
        ///   - I-type (FormatI): u64 sign-extended from 12-bit, then u64→i128 zero-extension
        ///     → F.fromU64(sign_extended_u64). This includes 0x13, 0x03, 0x67, 0x1b, 0x73.
        ///   - U-type (FormatU): raw upper 20 bits as u64 → F.fromU64(u32_value)
        ///   - J-type (FormatJ): u64 sign-extended from 21-bit, then u64→i128 zero-extension
        ///     → F.fromU64(sign_extended_u64)
        ///   - S-type (FormatS): i64 sign-extended from 12-bit → i64 as i128 (signed)
        ///     → fieldFromI128(signed_value)
        ///   - B-type (FormatB): i128 sign-extended from 13-bit → signed
        ///     → fieldFromI128(signed_value)
        ///
        /// The reason for the asymmetry: Jolt's FormatI/FormatJ/FormatU store imm as u64,
        /// while FormatS stores imm as i64 and FormatB stores imm as i128. The conversion
        /// to NormalizedOperands.imm (i128) uses `u64 as i128` (zero-extension) for the
        /// unsigned formats, but `i64 as i128` (sign-extension) for the signed formats.
        /// Then `F::from_i128()` is called on the result.
        pub fn computeImmediate(instr: u32) F {
            const opcode: u8 = @truncate(instr & 0x7f);

            switch (opcode) {
                // I-type: imm[11:0] at bits [31:20], sign-extended to i64, then treat as u64
                // Jolt: FormatI.imm is u64, NormalizedOperands.imm = u64 as i128 (zero-ext)
                0x13, 0x03, 0x67, 0x1b, 0x73 => {
                    const imm12: u32 = instr >> 20;
                    const imm_signed: i64 = @as(i64, @as(i32, @bitCast(imm12 << 20)) >> 20);
                    // Treat as unsigned u64 (same bit pattern), matching Jolt's u64 as i128
                    return F.fromU64(@as(u64, @bitCast(imm_signed)));
                },
                // S-type: imm[11:5] at [31:25], imm[4:0] at [11:7], sign-extended
                // Jolt: FormatS.imm is i64, NormalizedOperands.imm = i64 as i128 (sign-ext)
                0x23 => {
                    const imm11_5 = (instr >> 25) & 0x7f;
                    const imm4_0 = (instr >> 7) & 0x1f;
                    const imm12: u32 = (imm11_5 << 5) | imm4_0;
                    const imm_signed: i64 = @as(i64, @as(i32, @bitCast(imm12 << 20)) >> 20);
                    return signedI64ToField(imm_signed);
                },
                // B-type: imm[12|10:5] at [31:25], imm[4:1|11] at [11:7], sign-extended, *2
                // Jolt: FormatB.imm is i128, NormalizedOperands.imm = i128 directly (signed)
                0x63 => {
                    const imm12 = (instr >> 31) & 1;
                    const imm10_5 = (instr >> 25) & 0x3f;
                    const imm4_1 = (instr >> 8) & 0xf;
                    const imm11 = (instr >> 7) & 1;
                    const imm13: u32 = (imm12 << 12) | (imm11 << 11) | (imm10_5 << 5) | (imm4_1 << 1);
                    const imm_signed: i64 = @as(i64, @as(i32, @bitCast(imm13 << 19)) >> 19);
                    return signedI64ToField(imm_signed);
                },
                // U-type: imm[31:12] at [31:12], shifted left by 12, SIGN-EXTENDED to 64 bits
                // Jolt: FormatU.parse does `as i32 as i64 as u64` which sign-extends the
                // 32-bit immediate to 64 bits. E.g., LUI 0xf0f0f → imm = 0xFFFFFFFFF0F0F000.
                0x37, 0x17 => {
                    const imm_upper: u32 = instr & 0xFFFFF000;
                    const sign_extended: i64 = @as(i64, @as(i32, @bitCast(imm_upper)));
                    return F.fromU64(@as(u64, @bitCast(sign_extended)));
                },
                // J-type: imm[20|10:1|11|19:12] at [31:12], sign-extended to i64, then treat as u64
                // Jolt: FormatJ.imm is u64, NormalizedOperands.imm = u64 as i128 (zero-ext)
                0x6f => {
                    const imm20 = (instr >> 31) & 1;
                    const imm10_1 = (instr >> 21) & 0x3ff;
                    const imm11 = (instr >> 20) & 1;
                    const imm19_12 = (instr >> 12) & 0xff;
                    const imm21: u32 = (imm20 << 20) | (imm19_12 << 12) | (imm11 << 11) | (imm10_1 << 1);
                    const imm_signed: i64 = @as(i64, @as(i32, @bitCast(imm21 << 11)) >> 11);
                    // Treat as unsigned u64 (same bit pattern), matching Jolt's u64 as i128
                    return F.fromU64(@as(u64, @bitCast(imm_signed)));
                },
                else => return F.zero(),
            }
        }

        /// Convert signed i64 to field element (handle negative values)
        pub fn signedI64ToField(val: i64) F {
            if (val >= 0) {
                return F.fromU64(@intCast(val));
            } else {
                return F.zero().sub(F.fromU64(@intCast(-val)));
            }
        }

        /// Compute the sign-extended immediate as an UNSIGNED u64 (two's complement).
        /// Used for identity-path AddOperands instructions where the lookup index
        /// is computed as: x as u128 + y as u64 as u128.
        pub fn computeUnsignedImmediate(instr: u32) u64 {
            const opcode: u8 = @truncate(instr & 0x7f);
            switch (opcode) {
                0x13, 0x03, 0x67, 0x1b, 0x22 => { // I-type (including VirtualAssert*)
                    const imm12: u32 = instr >> 20;
                    const imm_signed: i64 = @as(i64, @as(i32, @bitCast(imm12 << 20)) >> 20);
                    return @bitCast(imm_signed);
                },
                0x6f => { // J-type (JAL)
                    const imm20 = (instr >> 31) & 0x1;
                    const imm10_1 = (instr >> 21) & 0x3FF;
                    const imm11 = (instr >> 20) & 0x1;
                    const imm19_12 = (instr >> 12) & 0xFF;
                    const raw = (imm20 << 20) | (imm19_12 << 12) | (imm11 << 11) | (imm10_1 << 1);
                    const imm_signed: i64 = @as(i64, @as(i32, @bitCast(raw << 11)) >> 11);
                    return @bitCast(imm_signed);
                },
                else => return 0,
            }
        }

        /// Compute eq(r, k) for a specific index k
        /// Compute eq(k, r) where r is in BIG_ENDIAN order.
        ///
        /// This matches Jolt's EqPolynomial::evals convention:
        /// - evals[k] = Π_j (bit_{n-1-j}(k) ? r[j] : (1-r[j]))
        /// - Equivalently: bit j of k ↔ r[n-1-j]
        ///
        /// Example for n=2, k=1 (binary 01):
        /// - j=0: bit 1 of k = 0 → (1-r[0])
        /// - j=1: bit 0 of k = 1 → r[1]
        /// - Result: (1-r[0]) * r[1]
        pub fn computeEqAtIndex(r: []const F, k: usize) F {
            const n = r.len;
            var result = F.one();
            for (0..n) |j| {
                // Extract bit (n-1-j) of k: b_j = (k >> (n-1-j)) & 1
                const bj: u1 = @truncate(k >> @intCast(n - 1 - j));
                const rj = r[j]; // r[j] corresponds to bit (n-1-j) of k
                if (bj == 1) {
                    // Use standard F multiplication for full field elements
                    // This matches Stage 2's InstrLookupsProver which also uses F.mul
                    result = result.mul(rj);
                } else {
                    const one_minus_rj = F.one().sub(rj);
                    result = result.mul(one_minus_rj);
                }
            }
            return result;
        }

        /// Build the full EQ table for all indices 0..2^n using parallel forward butterfly.
        /// O(2^n) field multiplications instead of O(n * 2^n) for element-wise computation.
        /// r is in BIG_ENDIAN order: r[0] is MSB.
        /// output must have length >= 2^r.len.
        ///
        /// Algorithm (matches Jolt's EqPolynomial::evals_parallel):
        /// Process r from LSB (r[n-1]) to MSB (r[0]). At each layer, the left/right
        /// halves are independent pairs that can be parallelized.
        pub fn buildFullEqTable(r: []const F, output: []F, tp: ?*ThreadPool) void {
            const n = r.len;
            if (n == 0) {
                output[0] = F.one();
                return;
            }
            // Seed: output[0] = 1
            output[0] = F.one();
            var size: usize = 1;

            // Process from LSB (r[n-1]) to MSB (r[0])
            for (0..n) |i| {
                const ri = r[n - 1 - i];
                const left = output[0..size];
                const right = output[size .. 2 * size];

                // Each (left[j], right[j]) pair is independent
                const PARALLEL_THRESHOLD = 256;
                if (tp != null and size >= PARALLEL_THRESHOLD) {
                    const EqButterflyCtx = struct {
                        l: []F,
                        rr: []F,
                        r_val: F,
                    };
                    const ctx = EqButterflyCtx{ .l = left, .rr = right, .r_val = ri };
                    tp.?.parallelForForce(size, ctx, struct {
                        fn f(c: EqButterflyCtx, j: usize) void {
                            const y = c.l[j].mul(c.r_val);
                            c.rr[j] = y;
                            c.l[j] = c.l[j].sub(y);
                        }
                    }.f);
                } else {
                    for (0..size) |j| {
                        const y = left[j].mul(ri);
                        right[j] = y;
                        left[j] = left[j].sub(y);
                    }
                }
                size *= 2;
            }
        }

        /// Build partial EQ table for first num_vars variables of r.
        /// Output has 2^num_vars entries.
        pub fn buildPartialEqTable(r: []const F, num_vars: usize, output: []F, tp: ?*ThreadPool) void {
            if (num_vars == 0) {
                output[0] = F.one();
                return;
            }
            buildFullEqTable(r[0..num_vars], output, tp);
        }

        /// Compute eq(k, r[0:num_vars]) - partial eq polynomial over first num_vars variables.
        /// This is used in cycle rounds where some variables have been bound.
        ///
        /// r is in BIG_ENDIAN order: r[0] is MSB, r[n-1] is LSB.
        /// For LowToHigh binding of cycle variables:
        /// - After binding k LSB variables, we use r[0:n-k] (the MSB portion)
        /// - k uses bits from the remaining (n-k) variables
        ///
        /// Example with n=8, num_vars=6 (after binding 2 LSB vars):
        /// - k in [0, 2^6) uses bits [0, 6) which correspond to r[0:6]
        /// - bit j of k corresponds to r[5-j] (since r[5] is bit 0, r[0] is bit 5)
        pub fn computeEqAtIndexPartial(r: []const F, k: usize, num_vars: usize) F {
            if (num_vars == 0) return F.one();
            var result = F.one();
            for (0..num_vars) |j| {
                // Extract bit (num_vars-1-j) of k: this is the j-th MSB of k
                const bj: u1 = @truncate(k >> @intCast(num_vars - 1 - j));
                const rj = r[j]; // r[j] corresponds to bit (num_vars-1-j) of k
                if (bj == 1) {
                    // Use standard F multiplication for full field elements
                    result = result.mul(rj);
                } else {
                    const one_minus_rj = F.one().sub(rj);
                    result = result.mul(one_minus_rj);
                }
            }
            return result;
        }

        /// Compute all LT(j, r) evaluations efficiently using Jolt's algorithm
        /// Returns lt_evals where lt_evals[j] = LT(j, r) for all j in [0, 2^n)
        /// r is in BIG_ENDIAN order (MSB first)
        pub fn computeAllLtEvals(allocator: Allocator, r: []const F, tp: ?*ThreadPool) ![]F {
            const n = r.len;
            const size = @as(usize, 1) << @intCast(n);
            var evals = try allocator.alloc(F, size);
            @memset(evals, F.zero());

            // Jolt's lt_evals algorithm with parallel butterfly:
            // Process r from LSB (r[n-1]) to MSB (r[0]).
            // LT formula: right[j] = left[j] * r_i; left[j] += r_i - right[j]

            for (0..n) |i| {
                const ri = r[n - 1 - i]; // Process from LSB to MSB
                const half = @as(usize, 1) << @intCast(i);
                const left = evals[0..half];
                const right = evals[half .. 2 * half];

                const PARALLEL_THRESHOLD = 256;
                if (tp != null and half >= PARALLEL_THRESHOLD) {
                    const LtButterflyCtx = struct {
                        l: []F,
                        rr: []F,
                        r_val: F,
                    };
                    const ctx = LtButterflyCtx{ .l = left, .rr = right, .r_val = ri };
                    tp.?.parallelForForce(half, ctx, struct {
                        fn f(c: LtButterflyCtx, j: usize) void {
                            const y = c.l[j].mul(c.r_val);
                            c.rr[j] = y;
                            c.l[j] = c.l[j].add(c.r_val.sub(y));
                        }
                    }.f);
                } else {
                    for (0..half) |j| {
                        const y = left[j].mul(ri);
                        right[j] = y;
                        left[j] = left[j].add(ri.sub(y));
                    }
                }
            }

            return evals;
        }

        /// Compute LT(j, r_cycle) for index j (legacy single-point version)
        /// LT(x, y) = 1 iff x < y as bitstrings
        /// x is boolean (index j), y is field elements (r_cycle)
        pub fn computeLtAtIndex(r_cycle: []const F, j: usize) F {
            // LT(x, y) = Σ_i (1 - x_i) · y_i · eq(x[i+1:], y[i+1:])
            // where sum runs from MSB to LSB
            var result = F.zero();
            const num_vars = r_cycle.len;

            // Process from MSB (index 0 in BIG_ENDIAN) to LSB
            for (0..num_vars) |i| {
                const ji = (j >> @intCast(num_vars - 1 - i)) & 1; // MSB first
                if (ji == 0) { // (1 - x_i) = 1 only when x_i = 0
                    var contrib = r_cycle[i]; // y_i
                    // Multiply by eq(x[i+1:], y[i+1:])
                    for ((i + 1)..num_vars) |k| {
                        const jk = (j >> @intCast(num_vars - 1 - k)) & 1;
                        const rk = r_cycle[k];
                        if (jk == 1) {
                            contrib = contrib.mul(rk);
                        } else {
                            contrib = contrib.mul(F.one().sub(rk));
                        }
                    }
                    result = result.add(contrib);
                }
            }

            return result;
        }

        /// Compute round polynomial for RegistersValEvaluation
        /// Returns [p(0), p(1), p(2), p(3)] for degree-3 sumcheck
        pub fn computeRegsValRoundPoly(inc: []F, wa: []F, lt: *const LtPolynomial(F), round: usize, tp: ?*ThreadPool) [4]F {
            const n = inc.len >> @intCast(round);
            const half = n / 2;

            if (half == 0) {
                var evals = [_]F{ F.zero(), F.zero(), F.zero(), F.zero() };
                if (n > 0) {
                    evals[0] = inc[0].mul(wa[0]).mul(lt.finalClaim());
                    evals[1] = evals[0];
                    evals[2] = evals[0];
                }
                return evals;
            }

            const LtPoly = LtPolynomial(F);
            const Ctx = struct {
                inc_p: []F,
                wa_p: []F,
                lt_p: *const LtPoly,
            };
            const ctx = Ctx{ .inc_p = inc, .wa_p = wa, .lt_p = lt };
            const identity = [_]F{ F.zero(), F.zero(), F.zero(), F.zero() };

            const mapFn = struct {
                fn f(c: Ctx, start: usize, end: usize) [4]F {
                    var r_u: [4]UnreducedProductAccum = .{UnreducedProductAccum.zero()} ** 4;
                    for (start..end) |i| {
                        const inc_0 = c.inc_p[2 * i];
                        const wa_0 = c.wa_p[2 * i];
                        const lt_0 = c.lt_p.getBoundCoeff(2 * i);
                        const inc_1 = c.inc_p[2 * i + 1];
                        const wa_1 = c.wa_p[2 * i + 1];
                        const lt_1 = c.lt_p.getBoundCoeff(2 * i + 1);

                        r_u[0].addAssign(inc_0.mul(wa_0).mulToProductAccum(lt_0));
                        r_u[1].addAssign(inc_1.mul(wa_1).mulToProductAccum(lt_1));
                        r_u[2].addAssign(inc_1.add(inc_1).sub(inc_0).mul(wa_1.add(wa_1).sub(wa_0)).mulToProductAccum(lt_1.add(lt_1).sub(lt_0)));
                        r_u[3].addAssign(inc_1.sub(inc_0).mul(wa_1.sub(wa_0)).mulToProductAccum(lt_1.sub(lt_0)));
                    }
                    return [4]F{ r_u[0].reduce(), r_u[1].reduce(), r_u[2].reduce(), r_u[3].reduce() };
                }
            }.f;
            const reduceFn = struct {
                fn f(a: [4]F, b: [4]F) [4]F {
                    return [4]F{ a[0].add(b[0]), a[1].add(b[1]), a[2].add(b[2]), a[3].add(b[3]) };
                }
            }.f;

            if (tp) |pool| {
                return pool.parallelReduce([4]F, half, identity, ctx, mapFn, reduceFn);
            }
            return mapFn(ctx, 0, half);
        }

        /// Bind challenge for RegistersValEvaluation polynomials
        pub fn bindRegsValChallenge(inc: []F, wa: []F, lt: *LtPolynomial(F), round: usize, r: F, tp: ?*ThreadPool, gpu: ?*GpuPolyOps) void {
            const n = inc.len >> @intCast(round);
            const half = n / 2;
            if (half == 0) return;

            // Bind LtPolynomial (operates on sqrt(T)-sized sub-arrays internally)
            lt.bind(r);

            // Bind inc and wa arrays (parallelize across 2 independent arrays)
            const arrays = [_][]F{ inc, wa };

            if (gpu) |g| {
                if (half >= 16384) {
                    for (arrays) |arr| {
                        g.polyBindLow(arr[0 .. half * 2], r, arr[0..half]) catch {
                            for (0..half) |i| {
                                arr[i] = arr[2 * i].add(r.mul(arr[2 * i + 1].sub(arr[2 * i])));
                            }
                        };
                    }
                } else {
                    for (0..half) |i| {
                        inc[i] = inc[2 * i].add(r.mul(inc[2 * i + 1].sub(inc[2 * i])));
                        wa[i] = wa[2 * i].add(r.mul(wa[2 * i + 1].sub(wa[2 * i])));
                    }
                }
            } else if (tp) |pool| {
                if (half >= 256) {
                    const BindCtx = struct { inc: []F, wa: []F, rv: F, h: usize };
                    const ctx = BindCtx{ .inc = inc, .wa = wa, .rv = r, .h = half };
                    pool.parallelForForce(2, ctx, struct {
                        fn f(c: BindCtx, arr_idx: usize) void {
                            const arr = if (arr_idx == 0) c.inc else c.wa;
                            for (0..c.h) |i| {
                                arr[i] = arr[2 * i].add(c.rv.mul(arr[2 * i + 1].sub(arr[2 * i])));
                            }
                        }
                    }.f);
                } else {
                    for (0..half) |i| {
                        inc[i] = inc[2 * i].add(r.mul(inc[2 * i + 1].sub(inc[2 * i])));
                        wa[i] = wa[2 * i].add(r.mul(wa[2 * i + 1].sub(wa[2 * i])));
                    }
                }
            } else {
                for (0..half) |i| {
                    inc[i] = inc[2 * i].add(r.mul(inc[2 * i + 1].sub(inc[2 * i])));
                    wa[i] = wa[2 * i].add(r.mul(wa[2 * i + 1].sub(wa[2 * i])));
                }
            }

            // Zero out upper half (inc and wa only; LtPolynomial handles its own state)
            for (half..n) |i| {
                inc[i] = F.zero();
                wa[i] = F.zero();
            }
        }

        /// Compute round polynomial for LookupsReadRaf (cycle rounds only)
        /// This computes Σ_j eq_reduction(j) * combined_vals(j)
        /// Returns [p(0), p(1), p(2), p_inf] for degree-2 polynomial (product of 2 linears)
        pub fn computeLookupsRoundPoly(eq_evals: []F, combined: []F, round: usize, tp: ?*ThreadPool) [4]F {
            const n = eq_evals.len >> @intCast(round);
            const half = n / 2;

            if (half == 0) {
                var evals = [_]F{ F.zero(), F.zero(), F.zero(), F.zero() };
                if (n > 0) {
                    const c = eq_evals[0].mul(combined[0]);
                    evals[0] = c;
                    evals[1] = c;
                    evals[2] = c;
                }
                return evals;
            }

            const Ctx = struct { eq: []F, comb: []F };
            const ctx = Ctx{ .eq = eq_evals, .comb = combined };
            const identity = [_]F{ F.zero(), F.zero(), F.zero(), F.zero() };

            const mapFn = struct {
                fn f(c: Ctx, start: usize, end: usize) [4]F {
                    var r_u: [4]UnreducedProductAccum = .{UnreducedProductAccum.zero()} ** 4;
                    for (start..end) |i| {
                        const eq_0 = c.eq[2 * i];
                        const eq_1 = c.eq[2 * i + 1];
                        const c_0 = c.comb[2 * i];
                        const c_1 = c.comb[2 * i + 1];
                        r_u[0].addAssign(eq_0.mulToProductAccum(c_0));
                        r_u[1].addAssign(eq_1.mulToProductAccum(c_1));
                        r_u[2].addAssign(eq_1.add(eq_1).sub(eq_0).mulToProductAccum(c_1.add(c_1).sub(c_0)));
                        r_u[3].addAssign(eq_1.sub(eq_0).mulToProductAccum(c_1.sub(c_0)));
                    }
                    return [4]F{ r_u[0].reduce(), r_u[1].reduce(), r_u[2].reduce(), r_u[3].reduce() };
                }
            }.f;
            const reduceFn = struct {
                fn f(a: [4]F, b: [4]F) [4]F {
                    return [4]F{ a[0].add(b[0]), a[1].add(b[1]), a[2].add(b[2]), a[3].add(b[3]) };
                }
            }.f;

            if (tp) |pool| {
                return pool.parallelReduce([4]F, half, identity, ctx, mapFn, reduceFn);
            }
            return mapFn(ctx, 0, half);
        }

        /// Bind challenge for LookupsReadRaf polynomials (cycle rounds) - legacy version
        pub fn bindLookupsChallenge(eq_evals: []F, combined: []F, round: usize, r: F, tp: ?*ThreadPool, gpu: ?*GpuPolyOps) void {
            const n = eq_evals.len >> @intCast(round);
            const half = n / 2;
            if (half == 0) return;

            const arrays = [_][]F{ eq_evals, combined };

            if (gpu) |g| {
                if (half >= 16384) {
                    for (arrays) |arr| {
                        g.polyBindLow(arr[0 .. half * 2], r, arr[0..half]) catch {
                            for (0..half) |i| {
                                arr[i] = arr[2 * i].add(r.mul(arr[2 * i + 1].sub(arr[2 * i])));
                            }
                        };
                    }
                } else {
                    for (0..half) |i| {
                        eq_evals[i] = eq_evals[2 * i].add(r.mul(eq_evals[2 * i + 1].sub(eq_evals[2 * i])));
                        combined[i] = combined[2 * i].add(r.mul(combined[2 * i + 1].sub(combined[2 * i])));
                    }
                }
            } else if (tp) |pool| {
                if (half >= 256) {
                    const BindCtx = struct { eq: []F, comb: []F, rv: F, h: usize };
                    const ctx = BindCtx{ .eq = eq_evals, .comb = combined, .rv = r, .h = half };
                    pool.parallelForForce(2, ctx, struct {
                        fn f(c: BindCtx, arr_idx: usize) void {
                            const arr = if (arr_idx == 0) c.eq else c.comb;
                            for (0..c.h) |i| {
                                arr[i] = arr[2 * i].add(c.rv.mul(arr[2 * i + 1].sub(arr[2 * i])));
                            }
                        }
                    }.f);
                } else {
                    for (0..half) |i| {
                        eq_evals[i] = eq_evals[2 * i].add(r.mul(eq_evals[2 * i + 1].sub(eq_evals[2 * i])));
                        combined[i] = combined[2 * i].add(r.mul(combined[2 * i + 1].sub(combined[2 * i])));
                    }
                }
            } else {
                for (0..half) |i| {
                    eq_evals[i] = eq_evals[2 * i].add(r.mul(eq_evals[2 * i + 1].sub(eq_evals[2 * i])));
                    combined[i] = combined[2 * i].add(r.mul(combined[2 * i + 1].sub(combined[2 * i])));
                }
            }

            // Zero out upper half
            for (half..n) |i| {
                eq_evals[i] = F.zero();
                combined[i] = F.zero();
            }
        }

        /// Compute round polynomial for LookupsReadRaf with ra_weights (cycle rounds)
        /// This computes Σ_j eq(j) * ra(j) * combined(j)
        /// Returns [p(0), p(1), p(2), p_inf] for degree-3 polynomial (product of 3 linears)
        pub fn computeLookupsRoundPolyWithRa(eq_evals: []F, ra_weights: []F, combined: []F, round: usize, tp: ?*ThreadPool) [4]F {
            const n = eq_evals.len >> @intCast(round);
            const half = n / 2;

            if (half == 0) {
                var evals = [_]F{ F.zero(), F.zero(), F.zero(), F.zero() };
                if (n > 0) {
                    const c = eq_evals[0].mul(ra_weights[0]).mul(combined[0]);
                    evals[0] = c;
                    evals[1] = c;
                    evals[2] = c;
                }
                return evals;
            }

            const Ctx = struct { eq: []F, ra: []F, comb: []F };
            const ctx = Ctx{ .eq = eq_evals, .ra = ra_weights, .comb = combined };
            const identity = [_]F{ F.zero(), F.zero(), F.zero(), F.zero() };

            const mapFn = struct {
                fn f(c: Ctx, start: usize, end: usize) [4]F {
                    var r_u: [4]UnreducedProductAccum = .{UnreducedProductAccum.zero()} ** 4;
                    for (start..end) |i| {
                        const eq_0 = c.eq[2 * i];
                        const eq_1 = c.eq[2 * i + 1];
                        const ra_0 = c.ra[2 * i];
                        const ra_1 = c.ra[2 * i + 1];
                        const c_0 = c.comb[2 * i];
                        const c_1 = c.comb[2 * i + 1];
                        r_u[0].addAssign(eq_0.mul(ra_0).mulToProductAccum(c_0));
                        r_u[1].addAssign(eq_1.mul(ra_1).mulToProductAccum(c_1));
                        r_u[2].addAssign(eq_1.add(eq_1).sub(eq_0).mul(ra_1.add(ra_1).sub(ra_0)).mulToProductAccum(c_1.add(c_1).sub(c_0)));
                        r_u[3].addAssign(eq_1.sub(eq_0).mul(ra_1.sub(ra_0)).mulToProductAccum(c_1.sub(c_0)));
                    }
                    return [4]F{ r_u[0].reduce(), r_u[1].reduce(), r_u[2].reduce(), r_u[3].reduce() };
                }
            }.f;
            const reduceFn = struct {
                fn f(a: [4]F, b: [4]F) [4]F {
                    return [4]F{ a[0].add(b[0]), a[1].add(b[1]), a[2].add(b[2]), a[3].add(b[3]) };
                }
            }.f;

            if (tp) |pool| {
                return pool.parallelReduce([4]F, half, identity, ctx, mapFn, reduceFn);
            }
            return mapFn(ctx, 0, half);
        }

        /// Bind challenge for LookupsReadRaf polynomials with ra_weights (cycle rounds)
        pub fn bindLookupsCycleChallengeWithRa(eq_evals: []F, ra_weights: []F, combined: []F, round: usize, r: F, tp: ?*ThreadPool, gpu: ?*GpuPolyOps) void {
            const n = eq_evals.len >> @intCast(round);
            const half = n / 2;
            if (half == 0) return;

            const arrays = [_][]F{ eq_evals, ra_weights, combined };

            if (gpu) |g| {
                if (half >= 16384) {
                    for (arrays) |arr| {
                        g.polyBindLow(arr[0 .. half * 2], r, arr[0..half]) catch {
                            for (0..half) |i| {
                                arr[i] = arr[2 * i].add(r.mul(arr[2 * i + 1].sub(arr[2 * i])));
                            }
                        };
                    }
                } else {
                    for (0..half) |i| {
                        eq_evals[i] = eq_evals[2 * i].add(r.mul(eq_evals[2 * i + 1].sub(eq_evals[2 * i])));
                        ra_weights[i] = ra_weights[2 * i].add(r.mul(ra_weights[2 * i + 1].sub(ra_weights[2 * i])));
                        combined[i] = combined[2 * i].add(r.mul(combined[2 * i + 1].sub(combined[2 * i])));
                    }
                }
            } else if (tp) |pool| {
                if (half >= 256) {
                    const BindCtx = struct { eq: []F, ra: []F, comb: []F, rv: F, h: usize };
                    const ctx = BindCtx{ .eq = eq_evals, .ra = ra_weights, .comb = combined, .rv = r, .h = half };
                    pool.parallelForForce(3, ctx, struct {
                        fn f(c: BindCtx, arr_idx: usize) void {
                            const arr = switch (arr_idx) {
                                0 => c.eq,
                                1 => c.ra,
                                2 => c.comb,
                                else => unreachable,
                            };
                            for (0..c.h) |i| {
                                arr[i] = arr[2 * i].add(c.rv.mul(arr[2 * i + 1].sub(arr[2 * i])));
                            }
                        }
                    }.f);
                } else {
                    for (0..half) |i| {
                        eq_evals[i] = eq_evals[2 * i].add(r.mul(eq_evals[2 * i + 1].sub(eq_evals[2 * i])));
                        ra_weights[i] = ra_weights[2 * i].add(r.mul(ra_weights[2 * i + 1].sub(ra_weights[2 * i])));
                        combined[i] = combined[2 * i].add(r.mul(combined[2 * i + 1].sub(combined[2 * i])));
                    }
                }
            } else {
                for (0..half) |i| {
                    eq_evals[i] = eq_evals[2 * i].add(r.mul(eq_evals[2 * i + 1].sub(eq_evals[2 * i])));
                    ra_weights[i] = ra_weights[2 * i].add(r.mul(ra_weights[2 * i + 1].sub(ra_weights[2 * i])));
                    combined[i] = combined[2 * i].add(r.mul(combined[2 * i + 1].sub(combined[2 * i])));
                }
            }

            // Zero out upper half
            for (half..n) |i| {
                eq_evals[i] = F.zero();
                ra_weights[i] = F.zero();
                combined[i] = F.zero();
            }
        }

        /// Bind challenge for a single polynomial (used for per-chunk ra weights)
        pub fn bindSinglePolynomial(poly: []F, round: usize, r: F, tp: ?*ThreadPool, gpu: ?*GpuPolyOps) void {
            _ = tp; // Single polynomial can't be parallelized across arrays
            const n = poly.len >> @intCast(round);
            const half = n / 2;
            if (half == 0) return;

            if (gpu) |g| {
                if (half >= 16384) {
                    g.polyBindLow(poly[0 .. half * 2], r, poly[0..half]) catch {
                        for (0..half) |i| {
                            poly[i] = poly[2 * i].add(r.mul(poly[2 * i + 1].sub(poly[2 * i])));
                        }
                    };
                } else {
                    for (0..half) |i| {
                        poly[i] = poly[2 * i].add(r.mul(poly[2 * i + 1].sub(poly[2 * i])));
                    }
                }
            } else {
                for (0..half) |i| {
                    poly[i] = poly[2 * i].add(r.mul(poly[2 * i + 1].sub(poly[2 * i])));
                }
            }

            // Zero out upper half
            for (half..n) |i| {
                poly[i] = F.zero();
            }
        }
    };
}

/// Evaluate LeftOperandPolynomial at r
/// LeftOperand(r) = Σ_{i=0}^{n/2-1} r[2i] * 2^(n/2-1-i)
/// For LOG_K=128: sum of even-indexed r values with powers of 2
pub fn evaluateLeftOperand(comptime F: type, r: []const F) F {
    const n = r.len;
    std.debug.assert(n % 2 == 0);
    var result = F.zero();
    var power = F.one();
    // Process from LSB to MSB of result
    var i: usize = n / 2;
    while (i > 0) {
        i -= 1;
        result = result.add(r[2 * i].mul(power));
        power = power.add(power); // power *= 2
    }
    return result;
}

/// Evaluate RightOperandPolynomial at r
/// RightOperand(r) = Σ_{i=0}^{n/2-1} r[2i+1] * 2^(n/2-1-i)
/// For LOG_K=128: sum of odd-indexed r values with powers of 2
pub fn evaluateRightOperand(comptime F: type, r: []const F) F {
    const n = r.len;
    std.debug.assert(n % 2 == 0);
    var result = F.zero();
    var power = F.one();
    // Process from LSB to MSB of result
    var i: usize = n / 2;
    while (i > 0) {
        i -= 1;
        const idx = 2 * i + 1;
        const term = r[idx].mul(power);
        result = result.add(term);
        // Debug: print first few and last few iterations
        if (n == 128 and (i < 3 or i >= 61)) {
            if (comptime debug_verbose) {
                dbg("[RIGHT_OP_DEBUG] i={d}: r[{d}]={x}, power={x}, term={x}, result={x}\n", .{
                    i,                          idx,                          r[idx].toBytesBE()[16..32].*, power.toBytesBE()[16..32].*,
                    term.toBytesBE()[16..32].*, result.toBytesBE()[16..32].*,
                });
            }
        }
        power = power.add(power); // power *= 2
    }
    if (comptime debug_verbose) {
        dbg("[RIGHT_OP_DEBUG] final result = {x}\n", .{result.toBytesBE()[16..32].*});
    }
    return result;
}

/// Evaluate IdentityPolynomial at r
/// Identity(r) = Σ_{i=0}^{n-1} r[i] * 2^(n-1-i)
/// This treats r as bits of a binary number
pub fn evaluateIdentity(comptime F: type, r: []const F) F {
    const n = r.len;
    var result = F.zero();
    var power = F.one();
    // Process from LSB to MSB
    var i: usize = n;
    while (i > 0) {
        i -= 1;
        const term = r[i].mul(power);
        result = result.add(term);
        // Debug: print first few and last few iterations
        if (n == 128 and (i < 4 or i >= 124)) {
            if (comptime debug_verbose) {
                dbg("[IDENTITY_DEBUG] i={d}: r[{d}]={x}, power={x}, term={x}, result={x}\n", .{
                    i,                          i,                            r[i].toBytesBE()[16..32].*, power.toBytesBE()[16..32].*,
                    term.toBytesBE()[16..32].*, result.toBytesBE()[16..32].*,
                });
            }
        }
        power = power.add(power); // power *= 2
    }
    if (comptime debug_verbose) {
        dbg("[IDENTITY_DEBUG] final result = {x}\n", .{result.toBytesBE()[16..32].*});
    }
    return result;
}

/// Compute eq(r, s) for two field element vectors
/// eq(r, s) = Π_i (r[i]*s[i] + (1-r[i])*(1-s[i]))
pub fn computeEqPolynomial(comptime F: type, r: []const F, s: []const F) F {
    std.debug.assert(r.len == s.len);
    var result = F.one();
    for (r, s) |ri, si| {
        // eq_i = ri*si + (1-ri)*(1-si) = 1 - ri - si + 2*ri*si
        const ri_si = ri.mul(si);
        const term = F.one().sub(ri).sub(si).add(ri_si).add(ri_si);
        result = result.mul(term);
    }
    return result;
}

/// Compute eq(k, r) where r is in BIG_ENDIAN order.
///
/// This matches Jolt's EqPolynomial::evals convention:
/// - evals[k] = Π_j (bit_{n-1-j}(k) ? r[j] : (1-r[j]))
/// - Equivalently: bit j of k ↔ r[n-1-j]
pub fn computeEqAtPoint(comptime F: type, r: []const F, k: u64) F {
    return @import("../eq_utils.zig").computeEqAtPointBE(F, r, @intCast(k));
}

/// Interleave bits of two 64-bit values into a 128-bit value
/// Matches Jolt's interleave_bits(even_bits, odd_bits):
///   - x (even_bits) goes to ODD positions (1, 3, 5, ...)
///   - y (odd_bits) goes to EVEN positions (0, 2, 4, ...)
/// In Jolt: `interleave_bits(x, y)` returns `(spread(x) << 1) | spread(y)`
pub fn interleaveBits128(x: u64, y: u64) u128 {
    var result: u128 = 0;
    for (0..64) |i| {
        const xi: u128 = @as(u128, (x >> @intCast(i)) & 1);
        const yi: u128 = @as(u128, (y >> @intCast(i)) & 1);
        // x at odd positions (2i+1), y at even positions (2i) - matches Jolt
        result |= xi << @intCast(2 * i + 1);
        result |= yi << @intCast(2 * i);
    }
    return result;
}

/// Get bit `bit_index` from a 128-bit value stored as (lo, hi)
/// bit_index 0 is LSB, bit_index 127 is MSB
pub fn getBit128(lo: u64, hi: u64, bit_index: usize) u1 {
    if (bit_index < 64) {
        return @truncate(lo >> @intCast(bit_index));
    } else {
        return @truncate(hi >> @intCast(bit_index - 64));
    }
}

/// Get the lookup table index for an instruction
/// Returns -1 if no lookup table is used, otherwise returns table index 0-41
/// Based on Jolt's LookupTables enum ordering:
///   0: RangeCheck, 1: RangeCheckAligned, 2: And, 3: Andn, 4: Or, 5: Xor,
///   6: Equal, 7: SignedGreaterThanEqual, 8: UnsignedGreaterThanEqual,
///   9: NotEqual, 10: SignedLessThan, 11: UnsignedLessThan, 12: Movsign,
///   13: UpperWord, 14: LessThanEqual, 15-17: Valid*Remainder/Div0,
///   18-19: HalfwordAlignment/WordAlignment, 20-21: LowerHalfWord/SignExtendHalfWord,
///   22-23: Pow2/Pow2W, 24: ShiftRightBitmask, 25: VirtualRev8W,
///   26: VirtualSRL, 27: VirtualSRA, 28: VirtualROTR, 29: VirtualROTRW,
///   30-31: VirtualChangeDivisor/W, 32: MulUNoOverflow, 33-40: VirtualXORROT*
pub fn getLookupTableIndex(opcode: u32, funct3: u32, funct7: u32) i8 {
    return switch (opcode) {
        0x33 => blk: { // R-type
            if (funct3 == 0 and funct7 == 0) break :blk 0; // ADD -> RangeCheckTable
            if (funct3 == 0 and funct7 == 0x20) break :blk 0; // SUB -> RangeCheckTable
            if (funct3 == 7) break :blk 2; // AND -> AndTable
            if (funct3 == 6) break :blk 4; // OR -> OrTable
            if (funct3 == 4) break :blk 5; // XOR -> XorTable
            if (funct3 == 1) break :blk -1; // SLL -> uses virtual decomposition
            if (funct3 == 5 and funct7 == 0) break :blk 25; // SRL -> VirtualSRLTable
            if (funct3 == 5 and funct7 == 0x20) break :blk 26; // SRA -> VirtualSRATable
            if (funct7 == 0x01 and funct3 == 0) break :blk 0; // MUL -> RangeCheckTable
            if (funct7 == 0x01 and funct3 == 3) break :blk 13; // MULHU -> UpperWordTable
            if (funct3 == 2) break :blk 10; // SLT -> SignedLessThanTable
            if (funct3 == 3) break :blk 11; // SLTU -> UnsignedLessThanTable
            break :blk -1;
        },
        0x13 => blk: { // I-type
            if (funct3 == 0) break :blk 0; // ADDI -> RangeCheckTable
            if (funct3 == 7) break :blk 2; // ANDI -> AndTable
            if (funct3 == 6) break :blk 4; // ORI -> OrTable
            if (funct3 == 4) break :blk 5; // XORI -> XorTable
            if (funct3 == 1) break :blk -1; // SLLI -> uses virtual decomposition
            if (funct3 == 5 and (funct7 & 0x40) == 0) break :blk 25; // SRLI -> VirtualSRLTable
            if (funct3 == 5 and (funct7 & 0x40) != 0) break :blk 26; // SRAI -> VirtualSRATable
            if (funct3 == 2) break :blk 10; // SLTI -> SignedLessThanTable
            if (funct3 == 3) break :blk 11; // SLTIU -> UnsignedLessThanTable
            break :blk -1;
        },
        0x1b => blk: { // OP-IMM-32
            if (funct3 == 0) break :blk 0; // ADDIW -> RangeCheckTable
            break :blk -1;
        },
        0x3b => blk: { // OP-32
            if (funct3 == 0 and funct7 == 0) break :blk 0; // ADDW -> RangeCheckTable
            if (funct3 == 0 and funct7 == 0x20) break :blk 0; // SUBW -> RangeCheckTable
            if (funct3 == 6 and funct7 == 0x01) break :blk 30; // VirtualChangeDivisorW -> VirtualChangeDivisorWTable
            break :blk -1;
        },
        0x63 => blk: { // B-type (branches)
            if (funct3 == 0) break :blk 6; // BEQ -> EqualTable
            if (funct3 == 1) break :blk 9; // BNE -> NotEqualTable
            if (funct3 == 4) break :blk 10; // BLT -> SignedLessThanTable
            if (funct3 == 5) break :blk 7; // BGE -> SignedGreaterThanEqualTable
            if (funct3 == 6) break :blk 11; // BLTU -> UnsignedLessThanTable
            if (funct3 == 7) break :blk 8; // BGEU -> UnsignedGreaterThanEqualTable
            break :blk -1;
        },
        0x0B => 20, // VirtualSignExtendWord -> SignExtendHalfWordTable
        0x2B => blk2b: { // Virtual I-type
            if (funct3 == 1) break :blk2b 21; // VirtualPow2 -> Pow2Table
            if (funct3 == 2) break :blk2b 23; // VirtualShiftRightBitmask -> ShiftRightBitmaskTable
            break :blk2b 0; // VirtualMULI (funct3=0) -> RangeCheckTable
        },
        0x5B => blk5b: { // Virtual shift right (funct3=0/5 only)
            if (funct3 == 5) break :blk5b 26; // VirtualSRAI -> VirtualSRATable
            if (funct3 == 0) break :blk5b 25; // VirtualSRLI -> VirtualSRLTable
            break :blk5b -1; // VirtualHostIO (other funct3) -> no lookup table
        },
        0x7B => 24, // VirtualRev8W (internal synthetic opcode) -> VirtualRev8WTable
        0x02 => 0, // VirtualAdvice -> RangeCheckTable
        0x22 => blk22: { // Virtual assert
            if (funct3 == 1) break :blk22 16; // VirtualAssertValidDiv0 -> ValidDiv0Table
            if (funct3 == 2) break :blk22 17; // VirtualAssertHalfwordAlignment -> HalfwordAlignmentTable
            if (funct3 == 3) break :blk22 18; // VirtualAssertWordAlignment -> WordAlignmentTable
            break :blk22 6; // VirtualAssertEQ -> EqualTable (funct3=0)
        },
        0x42 => 19, // VirtualZeroExtendWord -> LowerHalfWordTable
        0x62 => 15, // VirtualAssertValidUnsignedRemainder -> ValidUnsignedRemainderTable
        0x6B => blk6b: { // VirtualROTRI/VirtualROTRIW
            if (funct3 == 0) break :blk6b 27; // VirtualROTRI -> VirtualROTRTable
            break :blk6b 28; // VirtualROTRIW -> VirtualROTRWTable
        },
        0x37 => 0, // LUI -> RangeCheckTable
        0x17 => 0, // AUIPC -> RangeCheckTable
        0x6f => 0, // JAL -> RangeCheckTable
        0x67 => 1, // JALR -> RangeCheckAlignedTable
        0x03 => -1, // Load -> None (no lookup table)
        0x23 => -1, // Store -> None (no lookup table)
        else => -1,
    };
}

test "operand polynomial evaluation" {
    const F = @import("zolt_arith").field.BN254Scalar;

    // Simple test: r = [1, 0, 0, 1] (4 vars, LOG_K=4)
    // Left operand uses r[0], r[2] = 1, 0 → 1*2 + 0*1 = 2
    // Right operand uses r[1], r[3] = 0, 1 → 0*2 + 1*1 = 1
    const r = [_]F{ F.one(), F.zero(), F.zero(), F.one() };

    const left = evaluateLeftOperand(F, &r);
    const right = evaluateRightOperand(F, &r);

    // Left: r[0]*2 + r[2]*1 = 1*2 + 0*1 = 2
    try std.testing.expectEqual(F.fromU64(2), left);
    // Right: r[1]*2 + r[3]*1 = 0*2 + 1*1 = 1
    try std.testing.expectEqual(F.fromU64(1), right);
}
