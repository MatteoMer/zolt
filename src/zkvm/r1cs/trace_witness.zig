//! Direct TraceStep → CompactWitness + RawR1CSInputs builder.
//!
//! Builds integer witnesses directly from the execution trace without
//! going through Montgomery-encoded field elements. This eliminates the
//! ~150ms Montgomery round-trip (encode in fromTraceStepWithPCMap →
//! decode in compactFromFieldWitness) for 524k-trace programs.
//!
//! All arithmetic is pure integer — no F.fromU64(), no fromMontgomery().

const std = @import("std");
const Allocator = std.mem.Allocator;

const tracer = @import("../../tracer/mod.zig");
const TraceStep = tracer.TraceStep;
const ExecutionTrace = tracer.ExecutionTrace;
const preprocessing = @import("../preprocessing.zig");
const BytecodePCMapper = preprocessing.BytecodePCMapper;
const constraints = @import("constraints.zig");
const evaluators = @import("evaluators.zig");
const CompactWitness = evaluators.CompactWitness;
const RawR1CSInputs = evaluators.RawR1CSInputs;
const field_mod = @import("zolt_arith").field;
const S192 = field_mod.S192;
const ThreadPool = @import("zolt_pool").ThreadPool;

const FIRST_GROUP_SIZE = evaluators.FIRST_GROUP_SIZE;
const SECOND_GROUP_SIZE = evaluators.SECOND_GROUP_SIZE;

/// Build CompactWitness + RawR1CSInputs arrays directly from the execution trace.
/// Parallel over cycles. No Montgomery encoding/decoding.
pub fn buildFromTrace(
    trace: *const ExecutionTrace,
    pc_map: *const BytecodePCMapper,
    trace_length: usize,
    allocator: Allocator,
    thread_pool: ?*ThreadPool,
) !struct { compact: []CompactWitness, raw: []RawR1CSInputs } {
    const compact = try allocator.alloc(CompactWitness, trace_length);
    errdefer allocator.free(compact);
    const raw = try allocator.alloc(RawR1CSInputs, trace_length);
    errdefer allocator.free(raw);

    const num_real = trace.steps.items.len;
    const steps = trace.steps.items;

    const Ctx = struct {
        steps: []const TraceStep,
        pc_map: *const BytecodePCMapper,
        compact: []CompactWitness,
        raw: []RawR1CSInputs,
        num_real: usize,
        trace_length: usize,
    };

    const ctx = Ctx{
        .steps = steps,
        .pc_map = pc_map,
        .compact = compact,
        .raw = raw,
        .num_real = num_real,
        .trace_length = trace_length,
    };

    const mapFn = struct {
        fn f(c: Ctx, i: usize) void {
            if (i >= c.num_real) {
                // Beyond trace: pure NoOp padding
                c.compact[i] = CompactWitness.noop();
                c.raw[i] = RawR1CSInputs.noop();
                return;
            }
            const step = c.steps[i];
            if (step.is_noop or step.is_termination_store) {
                // NoOp/termination: fall back to field path for exact match
                // These are rare (a handful per trace) so cost is negligible
                @setEvalBranchQuota(10000);
                const F = field_mod.BN254Scalar;
                const ws = if (step.is_termination_store) blk_ts: {
                    const ns: ?TraceStep = if (i + 1 < c.num_real) c.steps[i + 1] else null;
                    break :blk_ts constraints.R1CSCycleInputs(F).createTerminationStoreWitness(step, ns, c.pc_map);
                } else constraints.R1CSCycleInputs(F).createNoopWitness();
                const wslice = ws.asSlice();
                c.compact[i] = evaluators.compactFromFieldWitnessPublic(F, wslice);
                c.raw[i] = evaluators.rawFromFieldWitnessPublic(F, wslice);
                return;
            }
            const next_step: ?TraceStep = if (i + 1 < c.num_real) c.steps[i + 1] else null;

            // Fast integer path for standard instructions.
            // Virtual/complex instructions fall back to field path for correctness.
            const op: u8 = @truncate(step.instruction & 0x7F);
            const needs_field_fallback = switch (op) {
                0x3B => @as(u3, @truncate((step.instruction >> 12) & 0x7)) == 6, // VirtualChangeDivisorW
                else => false,
            };

            // Also fall back for VirtualAssert alignment (0x22 funct3=2/3)
            const f3_op: u3 = @truncate((step.instruction >> 12) & 0x7);
            const needs_field_fallback_ext = needs_field_fallback or (op == 0x22 and (f3_op == 2 or f3_op == 3));
            if (needs_field_fallback_ext) {
                @setEvalBranchQuota(10000);
                const F = field_mod.BN254Scalar;
                const ws = constraints.R1CSCycleInputs(F).fromTraceStepWithPCMap(step, next_step, c.pc_map);
                const wslice = ws.asSlice();
                c.compact[i] = evaluators.compactFromFieldWitnessPublic(F, wslice);
                c.raw[i] = evaluators.rawFromFieldWitnessPublic(F, wslice);
            } else if (!needs_field_fallback_ext) {
                const result = compactAndRawFromTraceStep(step, next_step, c.pc_map);
                c.compact[i] = result.compact;
                c.raw[i] = result.raw;
            }
        }
    }.f;

    if (thread_pool) |tp| {
        tp.parallelFor(trace_length, ctx, mapFn);
    } else {
        for (0..trace_length) |i| {
            mapFn(ctx, i);
        }
    }

    return .{ .compact = compact, .raw = raw };
}

/// Build CompactWitness + RawR1CSInputs from a single TraceStep using pure integer arithmetic.
fn compactAndRawFromTraceStep(
    step: TraceStep,
    next_step: ?TraceStep,
    pc_map: *const BytecodePCMapper,
) struct { compact: CompactWitness, raw: RawR1CSInputs } {
    // ── Instruction decode ──
    const instr = step.instruction;
    const opcode: u8 = @truncate(instr & 0x7F);
    const funct3: u3 = @truncate((instr >> 12) & 0x7);
    const funct7: u7 = @truncate(instr >> 25);
    const is_load = (opcode == 0x03);
    const is_store = (opcode == 0x23);
    _ = is_load or is_store; // used implicitly
    const is_branch = (opcode == 0x63);
    const is_jump = (opcode == 0x6F or opcode == 0x67);
    const has_table = constraints.hasLookupTable(opcode, funct3, funct7);
    const writes_to_rd = step.rd_written and step.rd_index != 0 and !is_store and !is_branch;

    // ── Immediate (integer-only) ──
    const imm: i128 = decodeImmediateInt(instr, step);

    // ── Register values ──
    const reads_rs1 = readsRs1(opcode, step);
    const reads_rs2 = readsRs2(opcode, funct3, step);
    const rs1: u64 = if (reads_rs1) step.rs1_value else 0;
    const rs2: u64 = if (reads_rs2) step.rs2_value else 0;

    // ── Memory values ──
    const mem_val: u64 = step.memory_value orelse 0;
    const mem_pre_val: u64 = step.memory_pre_value orelse 0;
    var ram_addr: u64 = 0;
    var ram_read: u64 = 0;
    var ram_write: u64 = 0;
    var rd_write: u64 = 0;

    if (is_load) {
        // RamAddress = Rs1 + Imm (integer wrapping)
        ram_addr = rs1 +% @as(u64, @bitCast(@as(i64, @truncate(imm))));
        ram_read = mem_val;
        ram_write = mem_val;
        rd_write = mem_val;
    } else if (is_store) {
        ram_addr = rs1 +% @as(u64, @bitCast(@as(i64, @truncate(imm))));
        ram_read = mem_pre_val;
        ram_write = step.rs2_value;
        rd_write = 0;
    } else {
        if (is_jump) {
            const compressed_offset: u64 = if (step.is_compressed) 2 else 0;
            rd_write = step.unexpanded_pc +% 4 -% compressed_offset;
        } else if (writes_to_rd) {
            rd_write = step.rd_value;
        }
    }

    // ── Operand flags ──
    const flags = extractOperandFlags(opcode, funct3, funct7, step, has_table);

    // ── Instruction inputs ──
    const upc = step.unexpanded_pc;
    const left_input: u64 = if (flags.left_is_rs1) rs1 else if (flags.left_is_pc) upc else 0;
    // right_input: for identity-add, use unsigned imm (u64). For others, use signed or rs2.
    const right_input_i128: i128 = if (flags.right_is_rs2)
        @as(i128, rs2)
    else if (flags.right_is_imm)
        imm
    else
        0;
    // Product must handle u64 × u64 which can reach 2^128 (exceeds i128).
    // Compute sign-magnitude product as S192.
    const product_s192_val: S192 = blk_prod: {
        if (right_input_i128 == 0 or left_input == 0) break :blk_prod S192.zero();
        if (right_input_i128 > 0) {
            const mag = @as(u128, left_input) * @as(u128, @intCast(right_input_i128));
            break :blk_prod s192FromU128(mag);
        } else {
            const mag = @as(u128, left_input) * @as(u128, @intCast(-right_input_i128));
            break :blk_prod s192FromU128(mag).neg();
        }
    };

    // ── Lookup output (integer) ──
    const lookup_out: u64 = computeLookupOutputInt(step, funct3);

    // ── Lookup operands ──
    var left_lookup: u64 = 0;
    var right_lookup_u128: u128 = 0;
    var is_identity_path = false;
    if (has_table) {
        if (flags.flag_add or flags.flag_sub or flags.flag_mul or flags.flag_advice) {
            // Identity path: lookup operand computed from u128 formula
            is_identity_path = true;
            left_lookup = 0;
            right_lookup_u128 = computeU128LookupOperandInt(instr, step);
        } else {
            left_lookup = left_input;
            // For interleaved: right_lookup = right_input as u64
            right_lookup_u128 = if (right_input_i128 >= 0)
                @intCast(right_input_i128)
            else
                @bitCast(@as(i128, @as(i64, @truncate(right_input_i128))));
        }
    }
    // Special case: VirtualChangeDivisorW (0x3B funct3=6 funct7=1) uses truncated rs1
    if (opcode == 0x3B and funct3 == 6 and funct7 == 0x01) {
        left_lookup = step.rs1_value & 0xFFFFFFFF;
        right_lookup_u128 = step.rs2_value;
        is_identity_path = false;
    }

    // ── PC values ──
    const pc: u64 = @intCast(pc_map.getPCForStep(step));
    const next_is_real_noop = if (next_step) |ns| ns.is_noop and !ns.is_termination_store else true;
    const next_pc: u64 = if (next_is_real_noop) 0 else @intCast(pc_map.getPCForStep(next_step.?));
    const next_upc: u64 = if (next_is_real_noop) 0 else next_step.?.unexpanded_pc;

    // ── Circuit flags (boolean) ──
    const is_compressed = step.is_compressed;
    const vi = (step.virtual_sequence_remaining > 0 and !step.is_termination_store) or
        step.is_last_in_sequence or isVirtualOpcode(opcode);
    const dont_update = step.virtual_sequence_remaining > 0 and !step.is_termination_store;
    const is_last = step.is_last_in_sequence and opcode == 0x67;
    const next_is_virtual = if (next_step) |ns| blk: {
        const nop = ns.is_noop;
        const no: u8 = @truncate(ns.instruction & 0x7F);
        break :blk (!nop and ns.virtual_sequence_remaining > 0) or
            (!nop and ns.is_last_in_sequence) or
            isVirtualOpcode(no);
    } else false;
    const next_is_first = if (next_step) |ns| !ns.is_noop and ns.is_first_in_sequence else false;
    const should_jump_bool = flags.flag_jump and !isNoopStep(next_step);
    const should_branch_bool: bool = if (is_branch and has_table) lookup_out != 0 else false;

    // ── CompactWitness Az guards ──
    const b2i = struct {
        fn f(b: bool) i8 {
            return if (b) 1 else 0;
        }
    }.f;
    const load_i: i8 = b2i(is_load);
    const store_i: i8 = b2i(is_store);
    const add_i: i8 = b2i(flags.flag_add);
    const sub_i: i8 = b2i(flags.flag_sub);
    const mul_i: i8 = b2i(flags.flag_mul);

    var cw: CompactWitness = undefined;
    cw._pad = .{0} ** 5;
    cw.az_first = .{
        1 - load_i - store_i, // FG0: not_load_store
        load_i, // FG1: load
        load_i, // FG2: load
        store_i, // FG3: store
        add_i + sub_i + mul_i, // FG4: add_sub_mul
        1 - add_i - sub_i - mul_i, // FG5: not_add_sub_mul
        b2i(flags.flag_assert), // FG6: assert
        b2i(should_jump_bool), // FG7: should_jump
        b2i(vi) - b2i(is_last), // FG8: vi - is_last
        b2i(next_is_virtual) - b2i(next_is_first), // FG9: next_vi - next_first
    };
    cw.az_second = .{
        load_i + store_i, // SG0
        add_i, // SG1
        sub_i, // SG2
        mul_i, // SG3
        1 - add_i - sub_i - mul_i - b2i(flags.flag_advice), // SG4
        b2i(flags.flag_write_lookup), // SG5
        b2i(flags.flag_jump), // SG6
        b2i(should_branch_bool), // SG7
        1 - b2i(should_branch_bool) - b2i(flags.flag_jump), // SG8
    };

    // ── CompactWitness Bz magnitudes ──
    const dont_update_u64: u64 = if (dont_update) 1 else 0;
    const is_compressed_u64: u64 = if (is_compressed) 1 else 0;

    cw.bz_first = .{
        @as(i128, ram_addr),
        @as(i128, ram_read) - @as(i128, ram_write),
        @as(i128, ram_read) - @as(i128, rd_write),
        @as(i128, rs2) - @as(i128, ram_write),
        @as(i128, left_lookup),
        @as(i128, left_lookup) - @as(i128, left_input),
        @as(i128, lookup_out) - 1,
        @as(i128, next_upc) - @as(i128, lookup_out),
        @as(i128, next_pc) - @as(i128, pc) - 1,
        1 - @as(i128, dont_update_u64),
    };

    // For S192 Bz: need right_lookup as S192 and product as S192
    const rl_s192 = s192FromU128(right_lookup_u128);
    const product_s192 = product_s192_val;
    const imm_s192 = s192FromI128(imm);
    const ri_s192 = s192FromI128(right_input_i128);

    cw.bz_second = .{
        S192.fromU64(ram_addr).sub(S192.fromU64(rs1)).sub(imm_s192),
        rl_s192.sub(S192.fromU64(left_input)).sub(ri_s192),
        rl_s192.sub(S192.fromU64(left_input)).add(ri_s192).sub(S192.fromU128(@as(u128, 1) << 64)),
        rl_s192.sub(product_s192),
        rl_s192.sub(ri_s192),
        S192.fromU64(rd_write).sub(S192.fromU64(lookup_out)),
        S192.fromU64(rd_write).sub(S192.fromU64(upc)).sub(S192.fromU64(4)).add(S192.fromU64(2 * is_compressed_u64)),
        S192.fromU64(next_upc).sub(S192.fromU64(upc)).sub(imm_s192),
        S192.fromU64(next_upc).sub(S192.fromU64(upc)).sub(S192.fromU64(4)).add(S192.fromU64(4 * dont_update_u64)).add(S192.fromU64(2 * is_compressed_u64)),
    };

    // ── RawR1CSInputs ──
    var raw: RawR1CSInputs = undefined;
    raw.u64_values = .{
        left_input, pc,        upc,         ram_addr, rs1,     rs2,        rd_write,
        ram_read,   ram_write, left_lookup, next_upc, next_pc, lookup_out,
    };
    raw.signed_values = .{ right_input_i128, imm };
    raw.wide_values = .{ product_s192, s192FromU128(right_lookup_u128) };
    raw.bool_flags = .{
        should_branch_bool, // ShouldBranch
        next_is_virtual, // NextIsVirtual
        next_is_first, // NextIsFirstInSequence
        should_jump_bool, // ShouldJump
        flags.flag_add, // FlagAddOperands
        flags.flag_sub, // FlagSubtractOperands
        flags.flag_mul, // FlagMultiplyOperands
        is_load, // FlagLoad
        is_store, // FlagStore
        flags.flag_jump, // FlagJump
        flags.flag_write_lookup, // FlagWriteLookupOutputToRD
        vi, // FlagVirtualInstruction
        flags.flag_assert, // FlagAssert
        dont_update, // FlagDoNotUpdateUnexpandedPC
        flags.flag_advice, // FlagAdvice
        is_compressed, // FlagIsCompressed
        step.is_first_in_sequence, // FlagIsFirstInSequence
        is_last, // FlagIsLastInSequence
        step.rd_index != 0 and !is_store and !is_branch, // FlagIsRdNotZero
        is_branch, // FlagBranch
        false, // FlagIsNoop (real cycles are never noop)
        flags.left_is_rs1, // FlagLeftOperandIsRs1
        flags.left_is_pc, // FlagLeftOperandIsPC
        flags.right_is_rs2, // FlagRightOperandIsRs2
        flags.right_is_imm, // FlagRightOperandIsImm
    };

    return .{ .compact = cw, .raw = raw };
}

// ── Helpers ──

const OperandFlags = struct {
    left_is_rs1: bool,
    left_is_pc: bool,
    right_is_rs2: bool,
    right_is_imm: bool,
    flag_add: bool,
    flag_sub: bool,
    flag_mul: bool,
    flag_jump: bool,
    flag_assert: bool,
    flag_advice: bool,
    flag_write_lookup: bool,
};

fn extractOperandFlags(opcode: u8, funct3: u3, funct7: u7, step: TraceStep, has_table: bool) OperandFlags {
    var f = OperandFlags{
        .left_is_rs1 = false,
        .left_is_pc = false,
        .right_is_rs2 = false,
        .right_is_imm = false,
        .flag_add = false,
        .flag_sub = false,
        .flag_mul = false,
        .flag_jump = false,
        .flag_assert = false,
        .flag_advice = false,
        .flag_write_lookup = false,
    };
    if (!has_table) return f;

    switch (opcode) {
        0x33 => { // R-type: left=rs1, right=rs2 always
            f.left_is_rs1 = true;
            f.right_is_rs2 = true; // ALL R-type have right=rs2
            f.flag_write_lookup = true;
            if (funct7 == 0x01 and (funct3 == 0 or funct3 == 3)) {
                f.flag_mul = true; // MUL, MULHU
            } else if (funct3 == 0 and funct7 == 0x20) {
                f.flag_sub = true; // SUB
            } else if (funct3 == 0 and funct7 == 0) {
                f.flag_add = true; // ADD
            }
        },
        0x13 => { // I-type
            f.left_is_rs1 = true;
            f.right_is_imm = true; // ALL I-type ALU use imm as right operand
            f.flag_write_lookup = true;
            if (funct3 == 0) {
                f.flag_add = true;
            } // ADDI also has flag_add
        },
        0x6F => {
            f.left_is_pc = true;
            f.right_is_imm = true;
            f.flag_jump = true;
            f.flag_add = true;
        },
        0x67 => {
            f.left_is_rs1 = true;
            f.right_is_imm = true;
            f.flag_jump = true;
            f.flag_add = true;
        },
        0x63 => {
            f.left_is_rs1 = true;
            f.right_is_rs2 = true;
        },
        0x37 => {
            f.right_is_imm = true;
            f.flag_add = true;
            f.flag_write_lookup = true;
        },
        0x17 => {
            f.left_is_pc = true;
            f.right_is_imm = true;
            f.flag_add = true;
            f.flag_write_lookup = true;
        },
        0x1B => {
            f.left_is_rs1 = true;
            f.right_is_imm = true;
            f.flag_add = true;
            f.flag_write_lookup = true;
        },
        0x3B => { // OP-32: left=rs1, right=rs2 always
            f.left_is_rs1 = true;
            f.right_is_rs2 = true;
            f.flag_write_lookup = true;
            if (funct3 == 0 and funct7 == 0) {
                f.flag_add = true;
            } // ADDW
            else if (funct3 == 0 and funct7 == 0x20) {
                f.flag_sub = true;
            } // SUBW
        },
        0x0B => {
            f.left_is_rs1 = true;
            f.flag_add = true;
            f.flag_write_lookup = true;
        },
        0x2B => {
            f.left_is_rs1 = true;
            f.right_is_imm = true;
            f.flag_write_lookup = true;
            if (funct3 == 0) {
                f.flag_mul = true;
            } else {
                f.flag_add = true;
            }
        },
        0x5B => {
            f.left_is_rs1 = true;
            f.flag_write_lookup = true;
            if (step.rs2_read) {
                f.right_is_rs2 = true;
            } else {
                f.right_is_imm = true;
            }
        },
        0x02 => {
            f.flag_advice = true;
            f.flag_write_lookup = true;
        },
        0x22 => {
            f.left_is_rs1 = true;
            f.flag_assert = true;
            if (funct3 == 0 or funct3 == 1) {
                f.right_is_rs2 = true;
            } else if (funct3 == 2 or funct3 == 3) {
                f.right_is_imm = true;
                f.flag_add = true;
            }
        },
        0x42 => {
            f.left_is_rs1 = true;
            f.flag_add = true;
            f.flag_write_lookup = true;
        },
        0x62 => {
            f.left_is_rs1 = true;
            f.right_is_rs2 = true;
            f.flag_assert = true;
        },
        else => {},
    }
    return f;
}

fn readsRs1(opcode: u8, step: TraceStep) bool {
    _ = step;
    return switch (opcode) {
        0x13, 0x03, 0x67, 0x1b, 0x33, 0x3b, 0x23, 0x63, 0x0B, 0x2B, 0x5B, 0x22, 0x42, 0x62 => true,
        else => false,
    };
}

fn readsRs2(opcode: u8, funct3: u3, step: TraceStep) bool {
    return switch (opcode) {
        0x33, 0x3b, 0x23, 0x63 => true,
        0x22 => funct3 == 0 or funct3 == 1,
        0x5B => step.rs2_read,
        0x62 => true,
        else => false,
    };
}

/// Decode immediate as integer (no field ops). Matches fromTraceStepWithPCMap exactly.
fn decodeImmediateInt(instr: u32, step: TraceStep) i128 {
    const opcode: u8 = @truncate(instr & 0x7F);
    // Virtual instruction special cases
    if (opcode == 0x2B) {
        const f3: u3 = @truncate((instr >> 12) & 0x7);
        if (f3 == 0) {
            const shamt: u6 = @truncate((instr >> 20) & 0x3F);
            return @as(i128, @as(u64, 1) << shamt);
        }
        return 0;
    }
    if (opcode == 0x5B) {
        if (step.rs2_read) return 0;
        const total_shift: u7 = @truncate((instr >> 20) & 0x3F);
        const ones: u128 = (@as(u128, 1) << @intCast(64 - @as(u8, total_shift))) - 1;
        const bitmask: u64 = @truncate(ones << total_shift);
        return @as(i128, bitmask);
    }
    if (opcode == 0x22) {
        const f3_22: u3 = @truncate((instr >> 12) & 0x7);
        if (f3_22 == 2 or f3_22 == 3) {
            const imm12_raw: u32 = @truncate(instr >> 20);
            const imm_signed: i64 = @as(i64, @as(i32, @bitCast(imm12_raw << 20)) >> 20);
            return @as(i128, imm_signed);
        }
        return 0;
    }
    // Identity-add path: store as UNSIGNED u64 representation
    const is_identity_add = switch (opcode) {
        0x13 => (instr >> 12) & 0x7 == 0,
        0x1b => (instr >> 12) & 0x7 == 0,
        0x6f, 0x67 => true,
        else => false,
    };
    if (is_identity_add) {
        return @as(i128, computeUnsignedImmediate(instr));
    }
    // Signed path for Load/Store/Branch/U-type
    return deriveImmediateInt(instr);
}

/// Derive signed immediate from instruction (integer-only version of deriveImmediate)
fn deriveImmediateInt(instr: u32) i128 {
    const opcode: u8 = @truncate(instr & 0x7F);
    switch (opcode) {
        0x13, 0x67, 0x1b => { // I-type unsigned encoding
            const imm12: u32 = instr >> 20;
            const imm_signed: i64 = @as(i64, @as(i32, @bitCast(imm12 << 20)) >> 20);
            return @as(i128, @as(u64, @bitCast(imm_signed)));
        },
        0x03 => { // LOAD: signed
            const imm12: u32 = instr >> 20;
            const imm_signed: i64 = @as(i64, @as(i32, @bitCast(imm12 << 20)) >> 20);
            return @as(i128, imm_signed);
        },
        0x23 => { // STORE: S-type
            const imm4_0 = (instr >> 7) & 0x1F;
            const imm11_5 = (instr >> 25) & 0x7F;
            const imm_raw = (imm11_5 << 5) | imm4_0;
            if (imm_raw & 0x800 != 0) return -@as(i128, (~imm_raw + 1) & 0xFFF);
            return @as(i128, imm_raw);
        },
        0x63 => { // BRANCH: B-type
            const imm12 = (instr >> 31) & 0x1;
            const imm10_5 = (instr >> 25) & 0x3F;
            const imm4_1 = (instr >> 8) & 0xF;
            const imm11 = (instr >> 7) & 0x1;
            const imm_raw = (imm12 << 12) | (imm11 << 11) | (imm10_5 << 5) | (imm4_1 << 1);
            if (imm_raw & 0x1000 != 0) return -@as(i128, (~imm_raw + 1) & 0x1FFF);
            return @as(i128, imm_raw);
        },
        0x6F => { // JAL: J-type
            const imm20 = (instr >> 31) & 0x1;
            const imm10_1 = (instr >> 21) & 0x3FF;
            const imm11 = (instr >> 20) & 0x1;
            const imm19_12 = (instr >> 12) & 0xFF;
            const imm_raw = (imm20 << 20) | (imm19_12 << 12) | (imm11 << 11) | (imm10_1 << 1);
            if (imm_raw & 0x100000 != 0) return -@as(i128, (~imm_raw + 1) & 0x1FFFFF);
            return @as(i128, imm_raw);
        },
        0x37, 0x17 => { // U-type
            const imm_u32: u32 = instr & 0xFFFFF000;
            const imm_sext: u64 = @bitCast(@as(i64, @as(i32, @bitCast(imm_u32))));
            return @as(i128, imm_sext);
        },
        else => return 0,
    }
}

/// Compute LookupOutput as integer (no field ops)
fn computeLookupOutputInt(step: TraceStep, funct3: u3) u64 {
    const opcode: u8 = @truncate(step.instruction & 0x7F);
    const funct7: u7 = @truncate(step.instruction >> 25);
    if (!constraints.hasLookupTable(opcode, funct3, funct7)) return 0;

    switch (opcode) {
        0x6F => { // JAL
            const imm = constraints.decodeJTypeImmediate(step.instruction);
            const pc_i64: i64 = @intCast(step.unexpanded_pc);
            return @bitCast(pc_i64 +% imm);
        },
        0x67 => { // JALR
            const imm = constraints.decodeITypeImmediate(step.instruction);
            const rs1_i64: i64 = @intCast(step.rs1_value);
            return @as(u64, @bitCast(rs1_i64 +% imm)) & ~@as(u64, 1);
        },
        0x63 => { // Branch condition
            const rs1 = step.rs1_value;
            const rs2 = step.rs2_value;
            const taken: bool = switch (funct3) {
                0x0 => rs1 == rs2,
                0x1 => rs1 != rs2,
                0x4 => @as(i64, @bitCast(rs1)) < @as(i64, @bitCast(rs2)),
                0x5 => @as(i64, @bitCast(rs1)) >= @as(i64, @bitCast(rs2)),
                0x6 => rs1 < rs2,
                0x7 => rs1 >= rs2,
                else => false,
            };
            return if (taken) 1 else 0;
        },
        0x22, 0x62 => return 1, // Assert: always 1
        else => return step.rd_value,
    }
}

/// Compute u128 lookup operand (integer version of computeU128LookupOperand)
fn computeU128LookupOperandInt(instr: u32, step: TraceStep) u128 {
    const opcode: u8 = @truncate(instr & 0x7F);
    const funct3 = (instr >> 12) & 0x7;
    const funct7 = (instr >> 25) & 0x7F;

    switch (opcode) {
        0x33 => {
            if (funct7 == 0x01 and (funct3 == 0 or funct3 == 3)) {
                return @as(u128, step.rs1_value) * @as(u128, step.rs2_value);
            }
            if (funct7 == 0x20 and funct3 == 0) {
                return @as(u128, step.rs1_value) + (@as(u128, 1) << 64) - @as(u128, step.rs2_value);
            }
            if (funct3 == 0 and funct7 == 0) {
                return @as(u128, step.rs1_value) + @as(u128, step.rs2_value);
            }
            return 0;
        },
        0x13 => {
            if (funct3 == 0) {
                const imm_u64 = computeUnsignedImmediate(instr);
                return @as(u128, step.rs1_value) + @as(u128, imm_u64);
            }
            return 0;
        },
        0x1b => {
            if (funct3 == 0) {
                const imm_u64 = computeUnsignedImmediate(instr);
                return @as(u128, step.rs1_value) + @as(u128, imm_u64);
            }
            return 0;
        },
        0x37 => {
            const imm_u32: u32 = instr & 0xFFFFF000;
            const imm_sext: u64 = @bitCast(@as(i64, @as(i32, @bitCast(imm_u32))));
            return @as(u128, imm_sext);
        },
        0x17 => {
            const imm_u32: u32 = instr & 0xFFFFF000;
            const imm_sext: u64 = @bitCast(@as(i64, @as(i32, @bitCast(imm_u32))));
            return @as(u128, step.unexpanded_pc) + @as(u128, imm_sext);
        },
        0x6F => {
            const imm_u64 = computeUnsignedImmediate(instr);
            return @as(u128, step.unexpanded_pc) + @as(u128, imm_u64);
        },
        0x67 => {
            const imm_u64 = computeUnsignedImmediate(instr);
            return @as(u128, step.rs1_value) + @as(u128, imm_u64);
        },
        0x3B => {
            if (funct3 == 0 and funct7 == 0) {
                return @as(u128, step.rs1_value) + @as(u128, step.rs2_value);
            }
            if (funct3 == 0 and funct7 == 0x20) {
                return @as(u128, step.rs1_value) + (@as(u128, 1) << 64) - @as(u128, step.rs2_value);
            }
            return 0;
        },
        0x0B => return @as(u128, step.rs1_value),
        0x2B => {
            const f3: u3 = @truncate(funct3);
            if (f3 == 0) {
                const shamt: u6 = @truncate((instr >> 20) & 0x3F);
                const multiplier: u64 = @as(u64, 1) << shamt;
                return @as(u128, step.rs1_value) * @as(u128, multiplier);
            }
            return @as(u128, step.rs1_value);
        },
        0x42 => return @as(u128, step.rs1_value),
        0x22 => {
            const f3_22: u3 = @truncate(funct3);
            if (f3_22 == 2 or f3_22 == 3) {
                const imm_u64 = computeUnsignedImmediate(instr);
                return @as(u128, step.rs1_value) + @as(u128, imm_u64);
            }
            return 0;
        },
        0x02 => return @as(u128, step.rd_value),
        else => return 0,
    }
}

/// Compute unsigned u64 representation of sign-extended immediate.
/// Pure integer — matches R1CSCycleInputs.computeUnsignedImmediate.
fn computeUnsignedImmediate(instr: u32) u64 {
    const opcode = instr & 0x7F;
    switch (opcode) {
        0x13, 0x03, 0x67, 0x1b, 0x22 => {
            const imm12: u32 = instr >> 20;
            const imm_signed: i64 = @as(i64, @as(i32, @bitCast(imm12 << 20)) >> 20);
            return @bitCast(imm_signed);
        },
        0x6F => {
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

fn isVirtualOpcode(opcode: u8) bool {
    return switch (opcode) {
        0x0B, 0x2B, 0x5B, 0x02, 0x22, 0x42, 0x62 => true,
        else => false,
    };
}

fn isNoopStep(step_opt: ?TraceStep) bool {
    const step = step_opt orelse return false;
    if (step.is_noop) return true;
    if (@as(u8, @truncate(step.instruction & 0x7F)) == 0x13) {
        const rd: u8 = @truncate((step.instruction >> 7) & 0x1F);
        const rs1: u8 = @truncate((step.instruction >> 15) & 0x1F);
        const f3: u8 = @truncate((step.instruction >> 12) & 0x7);
        if (rd == 0 and rs1 == 0 and f3 == 0) {
            const imm_val: i32 = @bitCast(@as(u32, @truncate(step.instruction >> 20)));
            if (imm_val == 0) return true;
        }
    }
    return false;
}

fn s192FromI128(v: i128) S192 {
    if (v >= 0) {
        const mag: u128 = @intCast(v);
        return S192{ .magnitude = .{ @truncate(mag), @truncate(mag >> 64), 0 }, .is_positive = true };
    } else {
        const mag: u128 = @intCast(-v);
        return S192{ .magnitude = .{ @truncate(mag), @truncate(mag >> 64), 0 }, .is_positive = false };
    }
}

fn s192FromU128(v: u128) S192 {
    return S192{ .magnitude = .{ @truncate(v), @truncate(v >> 64), 0 }, .is_positive = true };
}
