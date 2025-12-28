//! Lookup Trace Collector
//!
//! This module provides a lookup trace collector that connects the RISC-V
//! execution tracer to the Lasso lookup argument infrastructure.
//!
//! During execution, each instruction generates one or more lookup queries
//! that are recorded in the lookup trace. This trace is then used by the
//! Lasso prover to generate the lookup argument proof.
//!
//! Reference: jolt-core/src/zkvm/instruction_lookups/

const std = @import("std");
const Allocator = std.mem.Allocator;

const mod = @import("mod.zig");
const lookups = @import("lookups.zig");
const lookup_table = @import("../lookup_table/mod.zig");

const CircuitFlags = mod.CircuitFlags;
const CircuitFlagSet = mod.CircuitFlagSet;
const InstructionFlags = mod.InstructionFlags;
const InstructionFlagSet = mod.InstructionFlagSet;
const LookupTables = mod.LookupTables;
const DecodedInstruction = mod.DecodedInstruction;
const Opcode = mod.Opcode;
const OpFunct3 = mod.OpFunct3;
const OpImmFunct3 = mod.OpImmFunct3;
const BranchFunct3 = mod.BranchFunct3;

/// A single lookup entry in the trace
pub fn LookupEntry(comptime XLEN: comptime_int) type {
    return struct {
        const Self = @This();

        /// CPU cycle at which this lookup occurred
        cycle: usize,
        /// Program counter at this cycle
        pc: u64,
        /// The lookup table being queried
        table: LookupTables(XLEN),
        /// The lookup index (interleaved operands for binary ops)
        index: u128,
        /// The lookup result/output
        result: u64,
        /// Left operand value
        left_operand: u64,
        /// Right operand value
        right_operand: u64,
        /// Circuit flags for constraint generation
        circuit_flags: CircuitFlagSet,
        /// Instruction metadata flags
        instruction_flags: InstructionFlagSet,
        /// Raw RISC-V instruction
        instruction: u32,

        /// Create entry for an ADD instruction
        pub fn fromAdd(cycle: usize, pc: u64, instruction: u32, rs1: u64, rs2: u64) Self {
            const AddLookup = lookups.AddLookup(XLEN);
            const add = AddLookup.init(rs1, rs2);
            return Self{
                .cycle = cycle,
                .pc = pc,
                .table = AddLookup.lookupTable(),
                .index = add.toLookupIndex(),
                .result = add.computeResult(),
                .left_operand = rs1,
                .right_operand = rs2,
                .circuit_flags = AddLookup.circuitFlags(),
                .instruction_flags = AddLookup.instructionFlags(),
                .instruction = instruction,
            };
        }

        /// Create entry for a SUB instruction
        pub fn fromSub(cycle: usize, pc: u64, instruction: u32, rs1: u64, rs2: u64) Self {
            const SubLookup = lookups.SubLookup(XLEN);
            const sub = SubLookup.init(rs1, rs2);
            return Self{
                .cycle = cycle,
                .pc = pc,
                .table = SubLookup.lookupTable(),
                .index = sub.toLookupIndex(),
                .result = sub.computeResult(),
                .left_operand = rs1,
                .right_operand = rs2,
                .circuit_flags = SubLookup.circuitFlags(),
                .instruction_flags = SubLookup.instructionFlags(),
                .instruction = instruction,
            };
        }

        /// Create entry for an AND instruction
        pub fn fromAnd(cycle: usize, pc: u64, instruction: u32, rs1: u64, rs2: u64) Self {
            const AndLookup = lookups.AndLookup(XLEN);
            const and_op = AndLookup.init(rs1, rs2);
            return Self{
                .cycle = cycle,
                .pc = pc,
                .table = AndLookup.lookupTable(),
                .index = and_op.toLookupIndex(),
                .result = and_op.computeResult(),
                .left_operand = rs1,
                .right_operand = rs2,
                .circuit_flags = AndLookup.circuitFlags(),
                .instruction_flags = AndLookup.instructionFlags(),
                .instruction = instruction,
            };
        }

        /// Create entry for an OR instruction
        pub fn fromOr(cycle: usize, pc: u64, instruction: u32, rs1: u64, rs2: u64) Self {
            const OrLookup = lookups.OrLookup(XLEN);
            const or_op = OrLookup.init(rs1, rs2);
            return Self{
                .cycle = cycle,
                .pc = pc,
                .table = OrLookup.lookupTable(),
                .index = or_op.toLookupIndex(),
                .result = or_op.computeResult(),
                .left_operand = rs1,
                .right_operand = rs2,
                .circuit_flags = OrLookup.circuitFlags(),
                .instruction_flags = OrLookup.instructionFlags(),
                .instruction = instruction,
            };
        }

        /// Create entry for a XOR instruction
        pub fn fromXor(cycle: usize, pc: u64, instruction: u32, rs1: u64, rs2: u64) Self {
            const XorLookup = lookups.XorLookup(XLEN);
            const xor_op = XorLookup.init(rs1, rs2);
            return Self{
                .cycle = cycle,
                .pc = pc,
                .table = XorLookup.lookupTable(),
                .index = xor_op.toLookupIndex(),
                .result = xor_op.computeResult(),
                .left_operand = rs1,
                .right_operand = rs2,
                .circuit_flags = XorLookup.circuitFlags(),
                .instruction_flags = XorLookup.instructionFlags(),
                .instruction = instruction,
            };
        }

        /// Create entry for SLT (set less than signed)
        pub fn fromSlt(cycle: usize, pc: u64, instruction: u32, rs1: u64, rs2: u64) Self {
            const SltLookup = lookups.SltLookup(XLEN);
            const slt = SltLookup.init(rs1, rs2);
            return Self{
                .cycle = cycle,
                .pc = pc,
                .table = SltLookup.lookupTable(),
                .index = slt.toLookupIndex(),
                .result = slt.computeResult(),
                .left_operand = rs1,
                .right_operand = rs2,
                .circuit_flags = SltLookup.circuitFlags(),
                .instruction_flags = SltLookup.instructionFlags(),
                .instruction = instruction,
            };
        }

        /// Create entry for SLTU (set less than unsigned)
        pub fn fromSltu(cycle: usize, pc: u64, instruction: u32, rs1: u64, rs2: u64) Self {
            const SltuLookup = lookups.SltuLookup(XLEN);
            const sltu = SltuLookup.init(rs1, rs2);
            return Self{
                .cycle = cycle,
                .pc = pc,
                .table = SltuLookup.lookupTable(),
                .index = sltu.toLookupIndex(),
                .result = sltu.computeResult(),
                .left_operand = rs1,
                .right_operand = rs2,
                .circuit_flags = SltuLookup.circuitFlags(),
                .instruction_flags = SltuLookup.instructionFlags(),
                .instruction = instruction,
            };
        }

        /// Create entry for BEQ (branch if equal)
        pub fn fromBeq(cycle: usize, pc: u64, instruction: u32, rs1: u64, rs2: u64) Self {
            const BeqLookup = lookups.BeqLookup(XLEN);
            const beq = BeqLookup.init(rs1, rs2);
            return Self{
                .cycle = cycle,
                .pc = pc,
                .table = BeqLookup.lookupTable(),
                .index = beq.toLookupIndex(),
                .result = beq.computeResult(),
                .left_operand = rs1,
                .right_operand = rs2,
                .circuit_flags = BeqLookup.circuitFlags(),
                .instruction_flags = BeqLookup.instructionFlags(),
                .instruction = instruction,
            };
        }

        /// Create entry for BNE (branch if not equal)
        pub fn fromBne(cycle: usize, pc: u64, instruction: u32, rs1: u64, rs2: u64) Self {
            const BneLookup = lookups.BneLookup(XLEN);
            const bne = BneLookup.init(rs1, rs2);
            return Self{
                .cycle = cycle,
                .pc = pc,
                .table = BneLookup.lookupTable(),
                .index = bne.toLookupIndex(),
                .result = bne.computeResult(),
                .left_operand = rs1,
                .right_operand = rs2,
                .circuit_flags = BneLookup.circuitFlags(),
                .instruction_flags = BneLookup.instructionFlags(),
                .instruction = instruction,
            };
        }

        /// Create entry for BLT (branch if less than signed)
        pub fn fromBlt(cycle: usize, pc: u64, instruction: u32, rs1: u64, rs2: u64) Self {
            const BltLookup = lookups.BltLookup(XLEN);
            const blt = BltLookup.init(rs1, rs2);
            return Self{
                .cycle = cycle,
                .pc = pc,
                .table = BltLookup.lookupTable(),
                .index = blt.toLookupIndex(),
                .result = blt.computeResult(),
                .left_operand = rs1,
                .right_operand = rs2,
                .circuit_flags = BltLookup.circuitFlags(),
                .instruction_flags = BltLookup.instructionFlags(),
                .instruction = instruction,
            };
        }

        /// Create entry for BGE (branch if greater than or equal signed)
        pub fn fromBge(cycle: usize, pc: u64, instruction: u32, rs1: u64, rs2: u64) Self {
            const BgeLookup = lookups.BgeLookup(XLEN);
            const bge = BgeLookup.init(rs1, rs2);
            return Self{
                .cycle = cycle,
                .pc = pc,
                .table = BgeLookup.lookupTable(),
                .index = bge.toLookupIndex(),
                .result = bge.computeResult(),
                .left_operand = rs1,
                .right_operand = rs2,
                .circuit_flags = BgeLookup.circuitFlags(),
                .instruction_flags = BgeLookup.instructionFlags(),
                .instruction = instruction,
            };
        }

        /// Create entry for BLTU (branch if less than unsigned)
        pub fn fromBltu(cycle: usize, pc: u64, instruction: u32, rs1: u64, rs2: u64) Self {
            const BltuLookup = lookups.BltuLookup(XLEN);
            const bltu = BltuLookup.init(rs1, rs2);
            return Self{
                .cycle = cycle,
                .pc = pc,
                .table = BltuLookup.lookupTable(),
                .index = bltu.toLookupIndex(),
                .result = bltu.computeResult(),
                .left_operand = rs1,
                .right_operand = rs2,
                .circuit_flags = BltuLookup.circuitFlags(),
                .instruction_flags = BltuLookup.instructionFlags(),
                .instruction = instruction,
            };
        }

        /// Create entry for BGEU (branch if greater than or equal unsigned)
        pub fn fromBgeu(cycle: usize, pc: u64, instruction: u32, rs1: u64, rs2: u64) Self {
            const BgeuLookup = lookups.BgeuLookup(XLEN);
            const bgeu = BgeuLookup.init(rs1, rs2);
            return Self{
                .cycle = cycle,
                .pc = pc,
                .table = BgeuLookup.lookupTable(),
                .index = bgeu.toLookupIndex(),
                .result = bgeu.computeResult(),
                .left_operand = rs1,
                .right_operand = rs2,
                .circuit_flags = BgeuLookup.circuitFlags(),
                .instruction_flags = BgeuLookup.instructionFlags(),
                .instruction = instruction,
            };
        }

        /// Create entry for LUI (load upper immediate)
        pub fn fromLui(cycle: usize, pc: u64, instruction: u32, imm: i32) Self {
            const LuiLookup = lookups.LuiLookup(XLEN);
            const lui = LuiLookup.init(imm);
            return Self{
                .cycle = cycle,
                .pc = pc,
                .table = LuiLookup.lookupTable(),
                .index = lui.toLookupIndex(),
                .result = lui.computeResult(),
                .left_operand = 0, // No rs1 for LUI
                .right_operand = @as(u64, @bitCast(@as(i64, imm))),
                .circuit_flags = LuiLookup.circuitFlags(),
                .instruction_flags = LuiLookup.instructionFlags(),
                .instruction = instruction,
            };
        }

        /// Create entry for AUIPC (add upper immediate to PC)
        pub fn fromAuipc(cycle: usize, pc: u64, instruction: u32, imm: i32) Self {
            const AuipcLookup = lookups.AuipcLookup(XLEN);
            const auipc = AuipcLookup.init(pc, imm);
            return Self{
                .cycle = cycle,
                .pc = pc,
                .table = AuipcLookup.lookupTable(),
                .index = auipc.toLookupIndex(),
                .result = auipc.computeResult(),
                .left_operand = pc,
                .right_operand = @as(u64, @bitCast(@as(i64, imm))),
                .circuit_flags = AuipcLookup.circuitFlags(),
                .instruction_flags = AuipcLookup.instructionFlags(),
                .instruction = instruction,
            };
        }

        /// Create entry for JAL (jump and link)
        pub fn fromJal(cycle: usize, pc: u64, instruction: u32, imm: i32, is_compressed: bool) Self {
            const JalLookup = lookups.JalLookup(XLEN);
            const jal = JalLookup.init(pc, imm, is_compressed);
            return Self{
                .cycle = cycle,
                .pc = pc,
                .table = JalLookup.lookupTable(),
                .index = jal.toLookupIndex(),
                .result = jal.computeResult(),
                .left_operand = pc,
                .right_operand = @as(u64, @bitCast(@as(i64, imm))),
                .circuit_flags = JalLookup.circuitFlags(),
                .instruction_flags = JalLookup.instructionFlags(),
                .instruction = instruction,
            };
        }

        /// Create entry for JALR (jump and link register)
        pub fn fromJalr(cycle: usize, pc: u64, instruction: u32, rs1: u64, imm: i32, is_compressed: bool) Self {
            const JalrLookup = lookups.JalrLookup(XLEN);
            const jalr = JalrLookup.init(pc, rs1, imm, is_compressed);
            return Self{
                .cycle = cycle,
                .pc = pc,
                .table = JalrLookup.lookupTable(),
                .index = jalr.toLookupIndex(),
                .result = jalr.computeResult(),
                .left_operand = rs1,
                .right_operand = @as(u64, @bitCast(@as(i64, imm))),
                .circuit_flags = JalrLookup.circuitFlags(),
                .instruction_flags = JalrLookup.instructionFlags(),
                .instruction = instruction,
            };
        }

        /// Create entry for SLL (shift left logical)
        pub fn fromSll(cycle: usize, pc: u64, instruction: u32, rs1: u64, rs2: u64) Self {
            const SllLookup = lookups.SllLookup(XLEN);
            const sll = SllLookup.init(rs1, rs2);
            return Self{
                .cycle = cycle,
                .pc = pc,
                .table = SllLookup.lookupTable(),
                .index = sll.toLookupIndex(),
                .result = sll.computeResult(),
                .left_operand = rs1,
                .right_operand = rs2,
                .circuit_flags = SllLookup.circuitFlags(),
                .instruction_flags = SllLookup.instructionFlags(),
                .instruction = instruction,
            };
        }

        /// Create entry for SRL (shift right logical)
        pub fn fromSrl(cycle: usize, pc: u64, instruction: u32, rs1: u64, rs2: u64) Self {
            const SrlLookup = lookups.SrlLookup(XLEN);
            const srl = SrlLookup.init(rs1, rs2);
            return Self{
                .cycle = cycle,
                .pc = pc,
                .table = SrlLookup.lookupTable(),
                .index = srl.toLookupIndex(),
                .result = srl.computeResult(),
                .left_operand = rs1,
                .right_operand = rs2,
                .circuit_flags = SrlLookup.circuitFlags(),
                .instruction_flags = SrlLookup.instructionFlags(),
                .instruction = instruction,
            };
        }

        /// Create entry for SRA (shift right arithmetic)
        pub fn fromSra(cycle: usize, pc: u64, instruction: u32, rs1: u64, rs2: u64) Self {
            const SraLookup = lookups.SraLookup(XLEN);
            const sra = SraLookup.init(rs1, rs2);
            return Self{
                .cycle = cycle,
                .pc = pc,
                .table = SraLookup.lookupTable(),
                .index = sra.toLookupIndex(),
                .result = sra.computeResult(),
                .left_operand = rs1,
                .right_operand = rs2,
                .circuit_flags = SraLookup.circuitFlags(),
                .instruction_flags = SraLookup.instructionFlags(),
                .instruction = instruction,
            };
        }

        /// Create entry for SLLI (shift left logical immediate)
        pub fn fromSlli(cycle: usize, pc: u64, instruction: u32, rs1: u64, imm: u64) Self {
            const SlliLookup = lookups.SlliLookup(XLEN);
            const slli = SlliLookup.init(rs1, imm);
            return Self{
                .cycle = cycle,
                .pc = pc,
                .table = SlliLookup.lookupTable(),
                .index = slli.toLookupIndex(),
                .result = slli.computeResult(),
                .left_operand = rs1,
                .right_operand = imm,
                .circuit_flags = SlliLookup.circuitFlags(),
                .instruction_flags = SlliLookup.instructionFlags(),
                .instruction = instruction,
            };
        }

        /// Create entry for SRLI (shift right logical immediate)
        pub fn fromSrli(cycle: usize, pc: u64, instruction: u32, rs1: u64, imm: u64) Self {
            const SrliLookup = lookups.SrliLookup(XLEN);
            const srli = SrliLookup.init(rs1, imm);
            return Self{
                .cycle = cycle,
                .pc = pc,
                .table = SrliLookup.lookupTable(),
                .index = srli.toLookupIndex(),
                .result = srli.computeResult(),
                .left_operand = rs1,
                .right_operand = imm,
                .circuit_flags = SrliLookup.circuitFlags(),
                .instruction_flags = SrliLookup.instructionFlags(),
                .instruction = instruction,
            };
        }

        /// Create entry for SRAI (shift right arithmetic immediate)
        pub fn fromSrai(cycle: usize, pc: u64, instruction: u32, rs1: u64, imm: u64) Self {
            const SraiLookup = lookups.SraiLookup(XLEN);
            const srai = SraiLookup.init(rs1, imm);
            return Self{
                .cycle = cycle,
                .pc = pc,
                .table = SraiLookup.lookupTable(),
                .index = srai.toLookupIndex(),
                .result = srai.computeResult(),
                .left_operand = rs1,
                .right_operand = imm,
                .circuit_flags = SraiLookup.circuitFlags(),
                .instruction_flags = SraiLookup.instructionFlags(),
                .instruction = instruction,
            };
        }

        /// Create entry for MUL (multiply low)
        pub fn fromMul(cycle: usize, pc: u64, instruction: u32, rs1: u64, rs2: u64) Self {
            const MulLookup = lookups.MulLookup(XLEN);
            const mul = MulLookup.init(rs1, rs2);
            return Self{
                .cycle = cycle,
                .pc = pc,
                .table = MulLookup.lookupTable(),
                .index = mul.toLookupIndex(),
                .result = mul.computeResult(),
                .left_operand = rs1,
                .right_operand = rs2,
                .circuit_flags = MulLookup.circuitFlags(),
                .instruction_flags = MulLookup.instructionFlags(),
                .instruction = instruction,
            };
        }

        /// Create entry for MULH (multiply high signed)
        pub fn fromMulh(cycle: usize, pc: u64, instruction: u32, rs1: u64, rs2: u64) Self {
            const MulhLookup = lookups.MulhLookup(XLEN);
            const mulh = MulhLookup.init(rs1, rs2);
            return Self{
                .cycle = cycle,
                .pc = pc,
                .table = MulhLookup.lookupTable(),
                .index = mulh.toLookupIndex(),
                .result = mulh.computeResult(),
                .left_operand = rs1,
                .right_operand = rs2,
                .circuit_flags = MulhLookup.circuitFlags(),
                .instruction_flags = MulhLookup.instructionFlags(),
                .instruction = instruction,
            };
        }

        /// Create entry for MULHU (multiply high unsigned)
        pub fn fromMulhu(cycle: usize, pc: u64, instruction: u32, rs1: u64, rs2: u64) Self {
            const MulhuLookup = lookups.MulhuLookup(XLEN);
            const mulhu = MulhuLookup.init(rs1, rs2);
            return Self{
                .cycle = cycle,
                .pc = pc,
                .table = MulhuLookup.lookupTable(),
                .index = mulhu.toLookupIndex(),
                .result = mulhu.computeResult(),
                .left_operand = rs1,
                .right_operand = rs2,
                .circuit_flags = MulhuLookup.circuitFlags(),
                .instruction_flags = MulhuLookup.instructionFlags(),
                .instruction = instruction,
            };
        }

        /// Create entry for MULHSU (multiply high signed-unsigned)
        pub fn fromMulhsu(cycle: usize, pc: u64, instruction: u32, rs1: u64, rs2: u64) Self {
            const MulhsuLookup = lookups.MulhsuLookup(XLEN);
            const mulhsu = MulhsuLookup.init(rs1, rs2);
            return Self{
                .cycle = cycle,
                .pc = pc,
                .table = MulhsuLookup.lookupTable(),
                .index = mulhsu.toLookupIndex(),
                .result = mulhsu.computeResult(),
                .left_operand = rs1,
                .right_operand = rs2,
                .circuit_flags = MulhsuLookup.circuitFlags(),
                .instruction_flags = MulhsuLookup.instructionFlags(),
                .instruction = instruction,
            };
        }

        /// Create entry for DIV (signed division)
        pub fn fromDiv(cycle: usize, pc: u64, instruction: u32, rs1: u64, rs2: u64) Self {
            const DivLookup = lookups.DivLookup(XLEN);
            const div = DivLookup.init(rs1, rs2);
            return Self{
                .cycle = cycle,
                .pc = pc,
                .table = DivLookup.lookupTable(),
                .index = div.toLookupIndex(),
                .result = div.computeResult(),
                .left_operand = rs1,
                .right_operand = rs2,
                .circuit_flags = DivLookup.circuitFlags(),
                .instruction_flags = DivLookup.instructionFlags(),
                .instruction = instruction,
            };
        }

        /// Create entry for DIVU (unsigned division)
        pub fn fromDivu(cycle: usize, pc: u64, instruction: u32, rs1: u64, rs2: u64) Self {
            const DivuLookup = lookups.DivuLookup(XLEN);
            const divu = DivuLookup.init(rs1, rs2);
            return Self{
                .cycle = cycle,
                .pc = pc,
                .table = DivuLookup.lookupTable(),
                .index = divu.toLookupIndex(),
                .result = divu.computeResult(),
                .left_operand = rs1,
                .right_operand = rs2,
                .circuit_flags = DivuLookup.circuitFlags(),
                .instruction_flags = DivuLookup.instructionFlags(),
                .instruction = instruction,
            };
        }

        /// Create entry for REM (signed remainder)
        pub fn fromRem(cycle: usize, pc: u64, instruction: u32, rs1: u64, rs2: u64) Self {
            const RemLookup = lookups.RemLookup(XLEN);
            const rem = RemLookup.init(rs1, rs2);
            return Self{
                .cycle = cycle,
                .pc = pc,
                .table = RemLookup.lookupTable(),
                .index = rem.toLookupIndex(),
                .result = rem.computeResult(),
                .left_operand = rs1,
                .right_operand = rs2,
                .circuit_flags = RemLookup.circuitFlags(),
                .instruction_flags = RemLookup.instructionFlags(),
                .instruction = instruction,
            };
        }

        /// Create entry for REMU (unsigned remainder)
        pub fn fromRemu(cycle: usize, pc: u64, instruction: u32, rs1: u64, rs2: u64) Self {
            const RemuLookup = lookups.RemuLookup(XLEN);
            const remu = RemuLookup.init(rs1, rs2);
            return Self{
                .cycle = cycle,
                .pc = pc,
                .table = RemuLookup.lookupTable(),
                .index = remu.toLookupIndex(),
                .result = remu.computeResult(),
                .left_operand = rs1,
                .right_operand = rs2,
                .circuit_flags = RemuLookup.circuitFlags(),
                .instruction_flags = RemuLookup.instructionFlags(),
                .instruction = instruction,
            };
        }

        // ========================================================================
        // Word-Sized Operations (*W instructions for RV64)
        // ========================================================================

        /// Create entry for ADDW (32-bit add, sign-extend)
        pub fn fromAddw(cycle: usize, pc: u64, instruction: u32, rs1: u64, rs2: u64) Self {
            const AddwLookup = lookups.AddwLookup(XLEN);
            const addw = AddwLookup.init(rs1, rs2);
            return Self{
                .cycle = cycle,
                .pc = pc,
                .table = AddwLookup.lookupTable(),
                .index = addw.toLookupIndex(),
                .result = addw.computeResult(),
                .left_operand = rs1,
                .right_operand = rs2,
                .circuit_flags = AddwLookup.circuitFlags(),
                .instruction_flags = AddwLookup.instructionFlags(),
                .instruction = instruction,
            };
        }

        /// Create entry for SUBW (32-bit subtract, sign-extend)
        pub fn fromSubw(cycle: usize, pc: u64, instruction: u32, rs1: u64, rs2: u64) Self {
            const SubwLookup = lookups.SubwLookup(XLEN);
            const subw = SubwLookup.init(rs1, rs2);
            return Self{
                .cycle = cycle,
                .pc = pc,
                .table = SubwLookup.lookupTable(),
                .index = subw.toLookupIndex(),
                .result = subw.computeResult(),
                .left_operand = rs1,
                .right_operand = rs2,
                .circuit_flags = SubwLookup.circuitFlags(),
                .instruction_flags = SubwLookup.instructionFlags(),
                .instruction = instruction,
            };
        }

        /// Create entry for SLLW (32-bit shift left, sign-extend)
        pub fn fromSllw(cycle: usize, pc: u64, instruction: u32, rs1: u64, rs2: u64) Self {
            const SllwLookup = lookups.SllwLookup(XLEN);
            const sllw = SllwLookup.init(rs1, rs2);
            return Self{
                .cycle = cycle,
                .pc = pc,
                .table = SllwLookup.lookupTable(),
                .index = sllw.toLookupIndex(),
                .result = sllw.computeResult(),
                .left_operand = rs1,
                .right_operand = rs2,
                .circuit_flags = SllwLookup.circuitFlags(),
                .instruction_flags = SllwLookup.instructionFlags(),
                .instruction = instruction,
            };
        }

        /// Create entry for SRLW (32-bit logical shift right, sign-extend)
        pub fn fromSrlw(cycle: usize, pc: u64, instruction: u32, rs1: u64, rs2: u64) Self {
            const SrlwLookup = lookups.SrlwLookup(XLEN);
            const srlw = SrlwLookup.init(rs1, rs2);
            return Self{
                .cycle = cycle,
                .pc = pc,
                .table = SrlwLookup.lookupTable(),
                .index = srlw.toLookupIndex(),
                .result = srlw.computeResult(),
                .left_operand = rs1,
                .right_operand = rs2,
                .circuit_flags = SrlwLookup.circuitFlags(),
                .instruction_flags = SrlwLookup.instructionFlags(),
                .instruction = instruction,
            };
        }

        /// Create entry for SRAW (32-bit arithmetic shift right, sign-extend)
        pub fn fromSraw(cycle: usize, pc: u64, instruction: u32, rs1: u64, rs2: u64) Self {
            const SrawLookup = lookups.SrawLookup(XLEN);
            const sraw = SrawLookup.init(rs1, rs2);
            return Self{
                .cycle = cycle,
                .pc = pc,
                .table = SrawLookup.lookupTable(),
                .index = sraw.toLookupIndex(),
                .result = sraw.computeResult(),
                .left_operand = rs1,
                .right_operand = rs2,
                .circuit_flags = SrawLookup.circuitFlags(),
                .instruction_flags = SrawLookup.instructionFlags(),
                .instruction = instruction,
            };
        }

        /// Create entry for MULW (32-bit multiply, sign-extend)
        pub fn fromMulw(cycle: usize, pc: u64, instruction: u32, rs1: u64, rs2: u64) Self {
            const MulwLookup = lookups.MulwLookup(XLEN);
            const mulw = MulwLookup.init(rs1, rs2);
            return Self{
                .cycle = cycle,
                .pc = pc,
                .table = MulwLookup.lookupTable(),
                .index = mulw.toLookupIndex(),
                .result = mulw.computeResult(),
                .left_operand = rs1,
                .right_operand = rs2,
                .circuit_flags = MulwLookup.circuitFlags(),
                .instruction_flags = MulwLookup.instructionFlags(),
                .instruction = instruction,
            };
        }

        /// Create entry for DIVW (32-bit signed division, sign-extend)
        pub fn fromDivw(cycle: usize, pc: u64, instruction: u32, rs1: u64, rs2: u64) Self {
            const DivwLookup = lookups.DivwLookup(XLEN);
            const divw = DivwLookup.init(rs1, rs2);
            return Self{
                .cycle = cycle,
                .pc = pc,
                .table = DivwLookup.lookupTable(),
                .index = divw.toLookupIndex(),
                .result = divw.computeResult(),
                .left_operand = rs1,
                .right_operand = rs2,
                .circuit_flags = DivwLookup.circuitFlags(),
                .instruction_flags = DivwLookup.instructionFlags(),
                .instruction = instruction,
            };
        }

        /// Create entry for DIVUW (32-bit unsigned division, sign-extend)
        pub fn fromDivuw(cycle: usize, pc: u64, instruction: u32, rs1: u64, rs2: u64) Self {
            const DivuwLookup = lookups.DivuwLookup(XLEN);
            const divuw = DivuwLookup.init(rs1, rs2);
            return Self{
                .cycle = cycle,
                .pc = pc,
                .table = DivuwLookup.lookupTable(),
                .index = divuw.toLookupIndex(),
                .result = divuw.computeResult(),
                .left_operand = rs1,
                .right_operand = rs2,
                .circuit_flags = DivuwLookup.circuitFlags(),
                .instruction_flags = DivuwLookup.instructionFlags(),
                .instruction = instruction,
            };
        }

        /// Create entry for REMW (32-bit signed remainder, sign-extend)
        pub fn fromRemw(cycle: usize, pc: u64, instruction: u32, rs1: u64, rs2: u64) Self {
            const RemwLookup = lookups.RemwLookup(XLEN);
            const remw = RemwLookup.init(rs1, rs2);
            return Self{
                .cycle = cycle,
                .pc = pc,
                .table = RemwLookup.lookupTable(),
                .index = remw.toLookupIndex(),
                .result = remw.computeResult(),
                .left_operand = rs1,
                .right_operand = rs2,
                .circuit_flags = RemwLookup.circuitFlags(),
                .instruction_flags = RemwLookup.instructionFlags(),
                .instruction = instruction,
            };
        }

        /// Create entry for REMUW (32-bit unsigned remainder, sign-extend)
        pub fn fromRemuw(cycle: usize, pc: u64, instruction: u32, rs1: u64, rs2: u64) Self {
            const RemuwLookup = lookups.RemuwLookup(XLEN);
            const remuw = RemuwLookup.init(rs1, rs2);
            return Self{
                .cycle = cycle,
                .pc = pc,
                .table = RemuwLookup.lookupTable(),
                .index = remuw.toLookupIndex(),
                .result = remuw.computeResult(),
                .left_operand = rs1,
                .right_operand = rs2,
                .circuit_flags = RemuwLookup.circuitFlags(),
                .instruction_flags = RemuwLookup.instructionFlags(),
                .instruction = instruction,
            };
        }
    };
}

/// Lookup trace collector that records all lookup operations during execution
pub fn LookupTraceCollector(comptime XLEN: comptime_int) type {
    return struct {
        const Self = @This();
        const Entry = LookupEntry(XLEN);

        /// All lookup entries
        entries: std.ArrayListUnmanaged(Entry),
        /// Allocator for dynamic memory
        allocator: Allocator,
        /// Whether to collect lookups (can be disabled for faster emulation)
        enabled: bool,

        pub fn init(allocator: Allocator) Self {
            return Self{
                .entries = .{},
                .allocator = allocator,
                .enabled = true,
            };
        }

        pub fn deinit(self: *Self) void {
            self.entries.deinit(self.allocator);
        }

        /// Clear all entries (for reuse)
        pub fn clear(self: *Self) void {
            self.entries.clearRetainingCapacity();
        }

        /// Enable or disable lookup collection
        pub fn setEnabled(self: *Self, enabled: bool) void {
            self.enabled = enabled;
        }

        /// Record a lookup entry
        pub fn record(self: *Self, entry: Entry) !void {
            if (!self.enabled) return;
            try self.entries.append(self.allocator, entry);
        }

        /// Record lookup for an instruction based on decoded opcode and function
        /// This is the main entry point called by the emulator
        pub fn recordInstruction(
            self: *Self,
            cycle: usize,
            pc: u64,
            instruction: u32,
            decoded: DecodedInstruction,
            rs1_val: u64,
            rs2_val: u64,
        ) !void {
            if (!self.enabled) return;

            switch (decoded.opcode) {
                .OP => {
                    // Check for M extension first
                    if (decoded.funct7 == 0b0000001) {
                        // M extension: MUL, MULH, MULHSU, MULHU, DIV, DIVU, REM, REMU
                        const entry: Entry = switch (decoded.funct3) {
                            0b000 => Entry.fromMul(cycle, pc, instruction, rs1_val, rs2_val), // MUL
                            0b001 => Entry.fromMulh(cycle, pc, instruction, rs1_val, rs2_val), // MULH
                            0b010 => Entry.fromMulhsu(cycle, pc, instruction, rs1_val, rs2_val), // MULHSU
                            0b011 => Entry.fromMulhu(cycle, pc, instruction, rs1_val, rs2_val), // MULHU
                            0b100 => Entry.fromDiv(cycle, pc, instruction, rs1_val, rs2_val), // DIV
                            0b101 => Entry.fromDivu(cycle, pc, instruction, rs1_val, rs2_val), // DIVU
                            0b110 => Entry.fromRem(cycle, pc, instruction, rs1_val, rs2_val), // REM
                            0b111 => Entry.fromRemu(cycle, pc, instruction, rs1_val, rs2_val), // REMU
                        };
                        try self.entries.append(self.allocator, entry);
                        return;
                    }

                    // Standard ALU operations
                    const funct3 = @as(OpFunct3, @enumFromInt(decoded.funct3));
                    const entry: ?Entry = switch (funct3) {
                        .ADD_SUB => blk: {
                            if ((decoded.funct7 & 0x20) != 0) {
                                // SUB
                                break :blk Entry.fromSub(cycle, pc, instruction, rs1_val, rs2_val);
                            } else {
                                // ADD
                                break :blk Entry.fromAdd(cycle, pc, instruction, rs1_val, rs2_val);
                            }
                        },
                        .AND => Entry.fromAnd(cycle, pc, instruction, rs1_val, rs2_val),
                        .OR => Entry.fromOr(cycle, pc, instruction, rs1_val, rs2_val),
                        .XOR => Entry.fromXor(cycle, pc, instruction, rs1_val, rs2_val),
                        .SLT => Entry.fromSlt(cycle, pc, instruction, rs1_val, rs2_val),
                        .SLTU => Entry.fromSltu(cycle, pc, instruction, rs1_val, rs2_val),
                        .SLL => Entry.fromSll(cycle, pc, instruction, rs1_val, rs2_val),
                        .SRL_SRA => blk: {
                            if ((decoded.funct7 & 0x20) != 0) {
                                // SRA (arithmetic)
                                break :blk Entry.fromSra(cycle, pc, instruction, rs1_val, rs2_val);
                            } else {
                                // SRL (logical)
                                break :blk Entry.fromSrl(cycle, pc, instruction, rs1_val, rs2_val);
                            }
                        },
                    };
                    if (entry) |e| {
                        try self.entries.append(self.allocator, e);
                    }
                },
                .OP_IMM => {
                    // Immediate ALU operations use immediate as second operand
                    const imm_val: u64 = @bitCast(@as(i64, decoded.imm));
                    const funct3 = @as(OpImmFunct3, @enumFromInt(decoded.funct3));
                    const entry: ?Entry = switch (funct3) {
                        .ADDI => Entry.fromAdd(cycle, pc, instruction, rs1_val, imm_val),
                        .ANDI => Entry.fromAnd(cycle, pc, instruction, rs1_val, imm_val),
                        .ORI => Entry.fromOr(cycle, pc, instruction, rs1_val, imm_val),
                        .XORI => Entry.fromXor(cycle, pc, instruction, rs1_val, imm_val),
                        .SLTI => Entry.fromSlt(cycle, pc, instruction, rs1_val, imm_val),
                        .SLTIU => Entry.fromSltu(cycle, pc, instruction, rs1_val, imm_val),
                        .SLLI => blk: {
                            // Shift amount is in the lower bits of imm
                            const imm_u32: u32 = @bitCast(@as(i32, @truncate(decoded.imm)));
                            const shamt: u64 = @as(u64, imm_u32 & 0x3F);
                            break :blk Entry.fromSlli(cycle, pc, instruction, rs1_val, shamt);
                        },
                        .SRLI_SRAI => blk: {
                            // Shift amount is in the lower bits of imm
                            const imm_u32: u32 = @bitCast(@as(i32, @truncate(decoded.imm)));
                            const shamt: u64 = @as(u64, imm_u32 & 0x3F);
                            if ((decoded.funct7 & 0x20) != 0) {
                                // SRAI (arithmetic)
                                break :blk Entry.fromSrai(cycle, pc, instruction, rs1_val, shamt);
                            } else {
                                // SRLI (logical)
                                break :blk Entry.fromSrli(cycle, pc, instruction, rs1_val, shamt);
                            }
                        },
                    };
                    if (entry) |e| {
                        try self.entries.append(self.allocator, e);
                    }
                },
                .BRANCH => {
                    // Branch operations - use dedicated branch lookups
                    const funct3 = @as(BranchFunct3, @enumFromInt(decoded.funct3));
                    const entry: ?Entry = switch (funct3) {
                        .BEQ => Entry.fromBeq(cycle, pc, instruction, rs1_val, rs2_val),
                        .BNE => Entry.fromBne(cycle, pc, instruction, rs1_val, rs2_val),
                        .BLT => Entry.fromBlt(cycle, pc, instruction, rs1_val, rs2_val),
                        .BGE => Entry.fromBge(cycle, pc, instruction, rs1_val, rs2_val),
                        .BLTU => Entry.fromBltu(cycle, pc, instruction, rs1_val, rs2_val),
                        .BGEU => Entry.fromBgeu(cycle, pc, instruction, rs1_val, rs2_val),
                        _ => null,
                    };
                    if (entry) |e| {
                        try self.entries.append(self.allocator, e);
                    }
                },
                .OP_32 => {
                    // 32-bit integer register-register operations (RV64 only)
                    // Check for M extension first
                    if (decoded.funct7 == 0b0000001) {
                        // RV64M word operations: MULW, DIVW, DIVUW, REMW, REMUW
                        const entry: Entry = switch (decoded.funct3) {
                            0b000 => Entry.fromMulw(cycle, pc, instruction, rs1_val, rs2_val), // MULW
                            0b100 => Entry.fromDivw(cycle, pc, instruction, rs1_val, rs2_val), // DIVW
                            0b101 => Entry.fromDivuw(cycle, pc, instruction, rs1_val, rs2_val), // DIVUW
                            0b110 => Entry.fromRemw(cycle, pc, instruction, rs1_val, rs2_val), // REMW
                            0b111 => Entry.fromRemuw(cycle, pc, instruction, rs1_val, rs2_val), // REMUW
                            else => Entry.fromAddw(cycle, pc, instruction, rs1_val, rs2_val), // fallback
                        };
                        try self.entries.append(self.allocator, entry);
                        return;
                    }

                    // Standard RV64I word operations: ADDW, SUBW, SLLW, SRLW, SRAW
                    const entry: Entry = switch (decoded.funct3) {
                        0b000 => blk: {
                            if ((decoded.funct7 & 0x20) != 0) {
                                break :blk Entry.fromSubw(cycle, pc, instruction, rs1_val, rs2_val); // SUBW
                            } else {
                                break :blk Entry.fromAddw(cycle, pc, instruction, rs1_val, rs2_val); // ADDW
                            }
                        },
                        0b001 => Entry.fromSllw(cycle, pc, instruction, rs1_val, rs2_val), // SLLW
                        0b101 => blk: {
                            if ((decoded.funct7 & 0x20) != 0) {
                                break :blk Entry.fromSraw(cycle, pc, instruction, rs1_val, rs2_val); // SRAW
                            } else {
                                break :blk Entry.fromSrlw(cycle, pc, instruction, rs1_val, rs2_val); // SRLW
                            }
                        },
                        else => Entry.fromAddw(cycle, pc, instruction, rs1_val, rs2_val), // fallback
                    };
                    try self.entries.append(self.allocator, entry);
                },
                .LUI => {
                    // Load upper immediate
                    const entry = Entry.fromLui(cycle, pc, instruction, decoded.imm);
                    try self.entries.append(self.allocator, entry);
                },
                .AUIPC => {
                    // Add upper immediate to PC
                    const entry = Entry.fromAuipc(cycle, pc, instruction, decoded.imm);
                    try self.entries.append(self.allocator, entry);
                },
                .JAL => {
                    // Jump and link
                    const is_compressed = false; // Standard instructions are not compressed
                    const entry = Entry.fromJal(cycle, pc, instruction, decoded.imm, is_compressed);
                    try self.entries.append(self.allocator, entry);
                },
                .JALR => {
                    // Jump and link register
                    const is_compressed = false;
                    const entry = Entry.fromJalr(cycle, pc, instruction, rs1_val, decoded.imm, is_compressed);
                    try self.entries.append(self.allocator, entry);
                },
                else => {
                    // LOAD, STORE - memory operations handled separately
                },
            }
        }

        /// Get the number of lookup entries
        pub fn len(self: *const Self) usize {
            return self.entries.items.len;
        }

        /// Get entry at index
        pub fn get(self: *const Self, index: usize) ?Entry {
            if (index >= self.entries.items.len) return null;
            return self.entries.items[index];
        }

        /// Get all entries as a slice
        pub fn getEntries(self: *const Self) []const Entry {
            return self.entries.items;
        }

        /// Count lookups by table type
        pub fn countByTable(self: *const Self, table: LookupTables(XLEN)) usize {
            var count: usize = 0;
            for (self.entries.items) |entry| {
                if (entry.table == table) {
                    count += 1;
                }
            }
            return count;
        }

        /// Get statistics about the lookup trace
        pub const Stats = struct {
            total_lookups: usize,
            and_lookups: usize,
            or_lookups: usize,
            xor_lookups: usize,
            sub_lookups: usize,
            range_check_lookups: usize,
            signed_lt_lookups: usize,
            unsigned_lt_lookups: usize,
            equal_lookups: usize,
            not_equal_lookups: usize,
        };

        pub fn getStats(self: *const Self) Stats {
            return Stats{
                .total_lookups = self.entries.items.len,
                .and_lookups = self.countByTable(.And),
                .or_lookups = self.countByTable(.Or),
                .xor_lookups = self.countByTable(.Xor),
                .sub_lookups = self.countByTable(.Sub),
                .range_check_lookups = self.countByTable(.RangeCheck),
                .signed_lt_lookups = self.countByTable(.SignedLessThan),
                .unsigned_lt_lookups = self.countByTable(.UnsignedLessThan),
                .equal_lookups = self.countByTable(.Equal),
                .not_equal_lookups = self.countByTable(.NotEqual),
            };
        }
    };
}

// ============================================================================
// Tests
// ============================================================================

test "lookup entry creation" {
    const Entry = LookupEntry(64);

    // Test ADD entry
    const add_entry = Entry.fromAdd(0, 0x1000, 0x00208033, 10, 20);
    try std.testing.expectEqual(@as(usize, 0), add_entry.cycle);
    try std.testing.expectEqual(@as(u64, 0x1000), add_entry.pc);
    try std.testing.expectEqual(@as(u64, 30), add_entry.result);
    try std.testing.expectEqual(LookupTables(64).RangeCheck, add_entry.table);

    // Test SUB entry
    const sub_entry = Entry.fromSub(1, 0x1004, 0x40208033, 30, 10);
    try std.testing.expectEqual(@as(u64, 20), sub_entry.result);
    try std.testing.expectEqual(LookupTables(64).Sub, sub_entry.table);

    // Test AND entry
    const and_entry = Entry.fromAnd(2, 0x1008, 0x00207033, 0xFF, 0x0F);
    try std.testing.expectEqual(@as(u64, 0x0F), and_entry.result);
    try std.testing.expectEqual(LookupTables(64).And, and_entry.table);
}

test "lookup trace collector basic" {
    const allocator = std.testing.allocator;
    const Collector = LookupTraceCollector(64);
    const Entry = LookupEntry(64);

    var collector = Collector.init(allocator);
    defer collector.deinit();

    // Record some entries
    try collector.record(Entry.fromAdd(0, 0x1000, 0x00208033, 5, 3));
    try collector.record(Entry.fromSub(1, 0x1004, 0x40208033, 10, 4));
    try collector.record(Entry.fromAnd(2, 0x1008, 0x00207033, 0xFF, 0x0F));

    try std.testing.expectEqual(@as(usize, 3), collector.len());

    // Check stats
    const stats = collector.getStats();
    try std.testing.expectEqual(@as(usize, 3), stats.total_lookups);
    try std.testing.expectEqual(@as(usize, 1), stats.range_check_lookups); // ADD
    try std.testing.expectEqual(@as(usize, 1), stats.sub_lookups);
    try std.testing.expectEqual(@as(usize, 1), stats.and_lookups);
}

test "lookup trace collector record instruction" {
    const allocator = std.testing.allocator;
    const Collector = LookupTraceCollector(64);

    var collector = Collector.init(allocator);
    defer collector.deinit();

    // Test ADD instruction: add x1, x2, x3
    // opcode=0x33 (OP), rd=1, rs1=2, rs2=3, funct3=0, funct7=0
    const add_instr: u32 = 0x003100b3;
    const add_decoded = DecodedInstruction.decode(add_instr);
    try collector.recordInstruction(0, 0x1000, add_instr, add_decoded, 10, 20);

    try std.testing.expectEqual(@as(usize, 1), collector.len());
    const entry = collector.get(0).?;
    try std.testing.expectEqual(@as(u64, 30), entry.result);
    try std.testing.expectEqual(LookupTables(64).RangeCheck, entry.table);

    // Test SUB instruction: sub x1, x2, x3
    // opcode=0x33 (OP), rd=1, rs1=2, rs2=3, funct3=0, funct7=0x20
    const sub_instr: u32 = 0x403100b3;
    const sub_decoded = DecodedInstruction.decode(sub_instr);
    try collector.recordInstruction(1, 0x1004, sub_instr, sub_decoded, 30, 10);

    try std.testing.expectEqual(@as(usize, 2), collector.len());
    const sub_entry = collector.get(1).?;
    try std.testing.expectEqual(@as(u64, 20), sub_entry.result);
    try std.testing.expectEqual(LookupTables(64).Sub, sub_entry.table);
}

test "lookup trace collector immediate instructions" {
    const allocator = std.testing.allocator;
    const Collector = LookupTraceCollector(64);

    var collector = Collector.init(allocator);
    defer collector.deinit();

    // Test ADDI instruction: addi x1, x2, 42
    // opcode=0x13 (OP_IMM), rd=1, rs1=2, imm=42, funct3=0
    const addi_instr: u32 = 0x02a10093;
    const addi_decoded = DecodedInstruction.decode(addi_instr);
    try collector.recordInstruction(0, 0x1000, addi_instr, addi_decoded, 10, 0);

    try std.testing.expectEqual(@as(usize, 1), collector.len());
    const entry = collector.get(0).?;
    try std.testing.expectEqual(@as(u64, 52), entry.result); // 10 + 42
}

test "lookup trace collector branch instructions" {
    const allocator = std.testing.allocator;
    const Collector = LookupTraceCollector(64);

    var collector = Collector.init(allocator);
    defer collector.deinit();

    // Test BEQ instruction
    // opcode=0x63 (BRANCH), funct3=0 (BEQ)
    const beq_instr: u32 = 0x00208063;
    const beq_decoded = DecodedInstruction.decode(beq_instr);
    try collector.recordInstruction(0, 0x1000, beq_instr, beq_decoded, 42, 42);

    try std.testing.expectEqual(@as(usize, 1), collector.len());
    const entry = collector.get(0).?;
    try std.testing.expectEqual(@as(u64, 1), entry.result); // Equal
    try std.testing.expectEqual(LookupTables(64).Equal, entry.table);
    try std.testing.expect(entry.instruction_flags.get(.Branch));
}

test "lookup trace collector disabled" {
    const allocator = std.testing.allocator;
    const Collector = LookupTraceCollector(64);
    const Entry = LookupEntry(64);

    var collector = Collector.init(allocator);
    defer collector.deinit();

    // Disable collection
    collector.setEnabled(false);

    // Record should be no-op
    try collector.record(Entry.fromAdd(0, 0x1000, 0x00208033, 5, 3));
    try std.testing.expectEqual(@as(usize, 0), collector.len());

    // Re-enable
    collector.setEnabled(true);
    try collector.record(Entry.fromAdd(0, 0x1000, 0x00208033, 5, 3));
    try std.testing.expectEqual(@as(usize, 1), collector.len());
}
