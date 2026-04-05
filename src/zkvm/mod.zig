//! Jolt zkVM - Zero-knowledge Virtual Machine
//!
//! This module implements the core zkVM functionality:
//! - RISC-V instruction execution
//! - Bytecode handling
//! - Memory and register checking
//! - R1CS constraint system
//! - Spartan proof system

const std = @import("std");

const zkvm_debug = @import("debug.zig");
const dbg = zkvm_debug.dbg;
const debug_verbose = zkvm_debug.verbose;

const Allocator = std.mem.Allocator;
const common = @import("../common/mod.zig");
const field = @import("zolt_arith").field;
const tracer = @import("../tracer/mod.zig");
const transcripts = @import("zolt_arith").transcripts;
const msm = @import("zolt_arith").msm;
const poly_commitment = @import("zolt_arith").poly.commitment;
const Dory = poly_commitment.dory;
const Fp = field.BN254BaseField;
const Fr = field.BN254Scalar;

pub const bytecode = @import("bytecode/mod.zig");
pub const claim_reductions = @import("claim_reductions/mod.zig");
pub const commitment_types = @import("commitment_types.zig");
pub const instruction = @import("instruction/mod.zig");
pub const jolt_device = @import("jolt_device.zig");
pub const jolt_types = @import("jolt_types.zig");
pub const jolt_serialization = @import("jolt_serialization.zig");
pub const preprocessing = @import("preprocessing.zig");
pub const jolt_prover = @import("jolt_prover.zig");
pub const shout = @import("shout/mod.zig");
pub const lookup_table = @import("lookup_table/mod.zig");
pub const r1cs = @import("r1cs/mod.zig");
pub const ram = @import("ram/mod.zig");
pub const registers = @import("registers/mod.zig");
pub const spartan = @import("spartan/mod.zig");

// Re-export commitment types
pub const PolyCommitment = commitment_types.PolyCommitment;
pub const OpeningProof = commitment_types.OpeningProof;


// Proving pipeline (extracted from mod.zig)
pub const proving_pipeline = @import("proving_pipeline.zig");
pub const computeBytecodeCodeSize = proving_pipeline.computeBytecodeCodeSize;

/// RISC-V register indices
pub const Register = enum(u8) {
    // Standard RISC-V registers
    zero = 0, // x0 - hardwired zero
    ra = 1, // x1 - return address
    sp = 2, // x2 - stack pointer
    gp = 3, // x3 - global pointer
    tp = 4, // x4 - thread pointer
    t0 = 5, // x5 - temporary
    t1 = 6, // x6 - temporary
    t2 = 7, // x7 - temporary
    s0 = 8, // x8/fp - saved/frame pointer
    s1 = 9, // x9 - saved
    a0 = 10, // x10 - argument/return
    a1 = 11, // x11 - argument/return
    a2 = 12, // x12 - argument
    a3 = 13, // x13 - argument
    a4 = 14, // x14 - argument
    a5 = 15, // x15 - argument
    a6 = 16, // x16 - argument
    a7 = 17, // x17 - argument
    s2 = 18, // x18 - saved
    s3 = 19, // x19 - saved
    s4 = 20, // x20 - saved
    s5 = 21, // x21 - saved
    s6 = 22, // x22 - saved
    s7 = 23, // x23 - saved
    s8 = 24, // x24 - saved
    s9 = 25, // x25 - saved
    s10 = 26, // x26 - saved
    s11 = 27, // x27 - saved
    t3 = 28, // x28 - temporary
    t4 = 29, // x29 - temporary
    t5 = 30, // x30 - temporary
    t6 = 31, // x31 - temporary
    _,

    pub fn fromIndex(index: u8) Register {
        return @enumFromInt(index);
    }

    pub fn toIndex(self: Register) u8 {
        return @intFromEnum(self);
    }
};

/// VM state during execution
pub const VMState = struct {
    /// Program counter
    pc: u64,
    /// Register file
    registers: [32]u64,
    /// Current instruction
    instruction: u32,
    /// Cycle count
    cycle: u64,

    pub fn init(entry_point: u64) VMState {
        var state = VMState{
            .pc = entry_point,
            .registers = [_]u64{0} ** 32,
            .instruction = 0,
            .cycle = 0,
        };
        // x0 is always zero
        state.registers[0] = 0;
        return state;
    }

    /// Read a register value
    pub fn readReg(self: *const VMState, reg: Register) u64 {
        const idx = reg.toIndex();
        if (idx == 0) return 0; // x0 is hardwired to zero
        return self.registers[idx];
    }

    /// Write a register value
    pub fn writeReg(self: *VMState, reg: Register, value: u64) void {
        const idx = reg.toIndex();
        if (idx == 0) return; // x0 is read-only
        self.registers[idx] = value;
    }
};

/// Jolt prover (delegates to proving_pipeline)
pub const JoltProver = proving_pipeline.JoltProver;


test {
    // Discover tests in sub-modules
    _ = @import("spartan/mod.zig");
}

test "vm state basic operations" {
    var state = VMState.init(0x80000000);

    // x0 should always be zero
    try std.testing.expectEqual(@as(u64, 0), state.readReg(.zero));

    // Write to x0 should be ignored
    state.writeReg(.zero, 42);
    try std.testing.expectEqual(@as(u64, 0), state.readReg(.zero));

    // Write to other registers should work
    state.writeReg(.a0, 123);
    try std.testing.expectEqual(@as(u64, 123), state.readReg(.a0));
}

test "register enum" {
    try std.testing.expectEqual(@as(u8, 0), Register.zero.toIndex());
    try std.testing.expectEqual(@as(u8, 1), Register.ra.toIndex());
    try std.testing.expectEqual(@as(u8, 2), Register.sp.toIndex());
    try std.testing.expectEqual(@as(u8, 10), Register.a0.toIndex());
}

// ============================================================================
// R1CS-Spartan Integration Tests
// ============================================================================

test "r1cs-spartan: witness generation and Az Bz Cz computation" {
    const F = field.BN254Scalar;
    const allocator = std.testing.allocator;

    // Create execution trace manually to test R1CS integration
    var trace = tracer.ExecutionTrace.init(allocator);
    defer trace.deinit();

    // Add a few execution steps using proper TraceStep structure
    try trace.steps.append(allocator, .{
        .cycle = 0,
        .pc = 0x1000,
        .unexpanded_pc = 0x1000,
        .instruction = 0x00500093, // ADDI x1, x0, 5
        .rs1_value = 0,
        .rs2_value = 0,
        .rd_pre_value = 0, // x1 was 0 before
        .rd_value = 5,
        .memory_addr = null,
        .memory_pre_value = null,
        .memory_value = null,
        .is_memory_write = false,
        .next_pc = 0x1004,
        .is_compressed = false,
    });

    try trace.steps.append(allocator, .{
        .cycle = 1,
        .pc = 0x1004,
        .unexpanded_pc = 0x1004,
        .instruction = 0x00A00113, // ADDI x2, x0, 10
        .rs1_value = 0,
        .rs2_value = 0,
        .rd_pre_value = 0, // x2 was 0 before
        .rd_value = 10,
        .memory_addr = null,
        .memory_pre_value = null,
        .memory_value = null,
        .is_memory_write = false,
        .next_pc = 0x1008,
        .is_compressed = false,
    });

    // Build JoltR1CS and test witness generation
    var jolt_r1cs = try r1cs.JoltR1CS(F).fromTrace(allocator, &trace);
    defer jolt_r1cs.deinit();

    try std.testing.expectEqual(@as(usize, 2), jolt_r1cs.num_cycles);

    // Build witness
    const witness = try jolt_r1cs.buildWitness();
    defer allocator.free(witness);

    // First element should be 1
    try std.testing.expect(witness[0].eql(F.one()));

    // Test Az, Bz, Cz computation
    const Az = try jolt_r1cs.computeAz(witness);
    defer allocator.free(Az);
    const Bz = try jolt_r1cs.computeBz(witness);
    defer allocator.free(Bz);
    const Cz = try jolt_r1cs.computeCz(witness);
    defer allocator.free(Cz);

    // Cz should be all zeros (equality-conditional form)
    for (Cz) |c| {
        try std.testing.expect(c.eql(F.zero()));
    }

    // Verify proper array sizes
    try std.testing.expectEqual(jolt_r1cs.padded_num_constraints, Az.len);
    try std.testing.expectEqual(jolt_r1cs.padded_num_constraints, Bz.len);
    try std.testing.expectEqual(jolt_r1cs.padded_num_constraints, Cz.len);

    // Note: Full constraint satisfaction requires proper instruction decoding
    // and consistent witness values. This test verifies the structure is correct.
}

test "sparse onehot joint poly equivalence with dense witness" {
    const F = field.BN254Scalar;

    // Simulate Stage 8 joint poly construction with small dimensions:
    // instruction_d=2, ram_d=1, bytecode_d=1 (4 total onehot arrays)
    const instruction_d = 2;
    const ram_d = 1;
    const bytecode_d = 1;
    const k_chunk = 4;
    const trace_length = 4;
    const total_poly_size = k_chunk * trace_length; // 16

    // Onehot indices: Zolt stores [InstructionRa, RamRa, BytecodeRa]
    var inst_ra0 = [_]?u8{ 1, 0, 3, 2 };
    var inst_ra1 = [_]?u8{ 0, 2, 1, 3 };
    var ram_ra0 = [_]?u8{ 3, 1, 0, 2 };
    var bc_ra0 = [_]?u8{ 2, 3, 0, 1 };
    const onehot_indices = [_][]?u8{
        &inst_ra0, // [0] InstructionRa chunk 0
        &inst_ra1, // [1] InstructionRa chunk 1
        &ram_ra0, //  [2] RamRa chunk 0
        &bc_ra0, //   [3] BytecodeRa chunk 0
    };

    // Gamma powers — Jolt order: [0]=RamInc, [1]=RdInc, [2..4]=InstructionRa, [4]=BytecodeRa, [5]=RamRa
    const gamma_base = F.fromU64(7); // arbitrary
    var gamma_powers: [6]F = undefined;
    gamma_powers[0] = gamma_base;
    for (1..6) |i| {
        gamma_powers[i] = gamma_powers[i - 1].mul(gamma_base);
    }

    // Dense witness polys: [0]=RdInc, [1]=RamInc (both length trace_length, padded to total_poly_size)
    var rd_inc = [_]F{ F.fromU64(10), F.fromU64(20), F.zero(), F.fromU64(30) } ++ ([_]F{F.zero()} ** (total_poly_size - 4));
    var ram_inc = [_]F{ F.fromU64(5), F.zero(), F.fromU64(15), F.fromU64(25) } ++ ([_]F{F.zero()} ** (total_poly_size - 4));
    var witness_polys = [_][]F{ &rd_inc, &ram_inc };

    // --- Build joint_poly via sparse path (same logic as Stage 8) ---
    var joint_sparse = [_]F{F.zero()} ** total_poly_size;

    // RamInc: gamma_powers[0]
    for (0..@min(witness_polys[1].len, total_poly_size)) |j| {
        if (!witness_polys[1][j].eql(F.zero())) {
            joint_sparse[j] = joint_sparse[j].add(witness_polys[1][j].mul(gamma_powers[0]));
        }
    }
    // RdInc: gamma_powers[1]
    for (0..@min(witness_polys[0].len, total_poly_size)) |j| {
        if (!witness_polys[0][j].eql(F.zero())) {
            joint_sparse[j] = joint_sparse[j].add(witness_polys[0][j].mul(gamma_powers[1]));
        }
    }
    // InstructionRa: gamma_powers[2..2+inst_d], onehot[0..inst_d]
    for (0..instruction_d) |i| {
        const gamma = gamma_powers[2 + i];
        const oh_idx = onehot_indices[i];
        for (0..trace_length) |cycle| {
            if (oh_idx[cycle]) |addr| {
                const j = @as(usize, addr) * trace_length + cycle;
                joint_sparse[j] = joint_sparse[j].add(gamma);
            }
        }
    }
    // BytecodeRa: gamma_powers[2+inst_d..2+inst_d+bc_d], onehot[inst_d+ram_d..] (Zolt order)
    for (0..bytecode_d) |i| {
        const gamma = gamma_powers[2 + instruction_d + i]; // Jolt: BytecodeRa before RamRa
        const oh_arr_idx = instruction_d + ram_d + i; // Zolt: BytecodeRa after RamRa
        const oh_idx = onehot_indices[oh_arr_idx];
        for (0..trace_length) |cycle| {
            if (oh_idx[cycle]) |addr| {
                const j = @as(usize, addr) * trace_length + cycle;
                joint_sparse[j] = joint_sparse[j].add(gamma);
            }
        }
    }
    // RamRa: gamma_powers[2+inst_d+bc_d..], onehot[inst_d..inst_d+ram_d] (Zolt order)
    for (0..ram_d) |i| {
        const gamma = gamma_powers[2 + instruction_d + bytecode_d + i]; // Jolt: RamRa after BytecodeRa
        const oh_arr_idx = instruction_d + i; // Zolt: RamRa before BytecodeRa
        const oh_idx = onehot_indices[oh_arr_idx];
        for (0..trace_length) |cycle| {
            if (oh_idx[cycle]) |addr| {
                const j = @as(usize, addr) * trace_length + cycle;
                joint_sparse[j] = joint_sparse[j].add(gamma);
            }
        }
    }

    // --- Build joint_poly via dense expansion (ground truth) ---
    var joint_dense = [_]F{F.zero()} ** total_poly_size;

    // Dense witness polys contribute the same way
    for (0..@min(witness_polys[1].len, total_poly_size)) |j| {
        if (!witness_polys[1][j].eql(F.zero())) {
            joint_dense[j] = joint_dense[j].add(witness_polys[1][j].mul(gamma_powers[0]));
        }
    }
    for (0..@min(witness_polys[0].len, total_poly_size)) |j| {
        if (!witness_polys[0][j].eql(F.zero())) {
            joint_dense[j] = joint_dense[j].add(witness_polys[0][j].mul(gamma_powers[1]));
        }
    }

    // Dense onehot expansion: expand each onehot array into k_chunk*trace_length dense poly,
    // then accumulate with the correct gamma (Jolt gamma order, not Zolt storage order)
    // Jolt gamma order: InstructionRa[0..inst_d], BytecodeRa[0..bc_d], RamRa[0..ram_d]
    // For InstructionRa: gamma_idx = 2+i, onehot_idx = i
    for (0..instruction_d) |i| {
        const gamma = gamma_powers[2 + i];
        for (0..trace_length) |cycle| {
            if (onehot_indices[i][cycle]) |addr| {
                const j = @as(usize, addr) * trace_length + cycle;
                joint_dense[j] = joint_dense[j].add(gamma);
            }
        }
    }
    // For BytecodeRa: gamma_idx = 2+inst_d+i, onehot_idx = inst_d+ram_d+i
    for (0..bytecode_d) |i| {
        const gamma = gamma_powers[2 + instruction_d + i];
        for (0..trace_length) |cycle| {
            if (onehot_indices[instruction_d + ram_d + i][cycle]) |addr| {
                const j = @as(usize, addr) * trace_length + cycle;
                joint_dense[j] = joint_dense[j].add(gamma);
            }
        }
    }
    // For RamRa: gamma_idx = 2+inst_d+bc_d+i, onehot_idx = inst_d+i
    for (0..ram_d) |i| {
        const gamma = gamma_powers[2 + instruction_d + bytecode_d + i];
        for (0..trace_length) |cycle| {
            if (onehot_indices[instruction_d + i][cycle]) |addr| {
                const j = @as(usize, addr) * trace_length + cycle;
                joint_dense[j] = joint_dense[j].add(gamma);
            }
        }
    }

    // Verify element-wise equality
    for (0..total_poly_size) |j| {
        try std.testing.expect(joint_sparse[j].eql(joint_dense[j]));
    }
}

// Include tests from submodules
test {
    // Force preprocessing tests to be included
    _ = preprocessing;
    _ = preprocessing.dory_verifier_setup;
}
