//! Witness generation from execution trace
//!
//! Converts an execution trace into witnesses for R1CS constraints.
//! Each trace step generates witness values for the circuit.

const std = @import("std");
const Allocator = std.mem.Allocator;
const zkvm = @import("../zkvm/mod.zig");
const tracer = @import("mod.zig");

const TraceStep = tracer.TraceStep;
const ExecutionTrace = tracer.ExecutionTrace;

/// Witness generation from execution trace
///
/// Converts an execution trace into witnesses for R1CS constraints.
/// Each trace step generates witness values for the circuit.
pub fn WitnessGenerator(comptime F: type) type {
    return struct {
        const Self = @This();

        allocator: Allocator,

        pub fn init(allocator: Allocator) Self {
            return .{ .allocator = allocator };
        }

        /// Witness layout for a single CPU step
        /// This matches Jolt's circuit structure
        pub const StepWitness = struct {
            /// Program counter
            pc: F,
            /// Instruction (as field element)
            instruction: F,
            /// Source register 1 value
            rs1: F,
            /// Source register 2 value
            rs2: F,
            /// Destination register value
            rd: F,
            /// Immediate value
            imm: F,
            /// Memory address (0 if no memory access)
            memory_addr: F,
            /// Memory value
            memory_value: F,
            /// Next PC
            next_pc: F,
            /// Opcode breakdown bits (for instruction decoding constraints)
            opcode_bits: [7]F,
            /// Register index bits for rs1
            rs1_idx_bits: [5]F,
            /// Register index bits for rs2
            rs2_idx_bits: [5]F,
            /// Register index bits for rd
            rd_idx_bits: [5]F,
        };

        /// Generate witness for a single trace step
        pub fn generateStepWitness(self: *Self, step: TraceStep) StepWitness {
            _ = self;
            const decoded = zkvm.instruction.DecodedInstruction.decode(step.instruction);

            // Convert values to field elements
            const pc = F.fromU64(step.pc);
            const instruction_f = F.fromU64(step.instruction);
            const rs1 = F.fromU64(step.rs1_value);
            const rs2 = F.fromU64(step.rs2_value);
            const rd = F.fromU64(step.rd_value);
            const imm: F = if (decoded.imm >= 0)
                F.fromU64(@intCast(decoded.imm))
            else
                F.zero().sub(F.fromU64(@intCast(-decoded.imm)));
            const memory_addr = F.fromU64(step.memory_addr orelse 0);
            const memory_value = F.fromU64(step.memory_value orelse 0);
            const next_pc = F.fromU64(step.next_pc);

            // Extract opcode bits
            var opcode_bits: [7]F = undefined;
            const opcode_val = step.instruction & 0x7F;
            for (0..7) |i| {
                opcode_bits[i] = if ((opcode_val >> @intCast(i)) & 1 == 1) F.one() else F.zero();
            }

            // Extract register index bits
            var rs1_idx_bits: [5]F = undefined;
            var rs2_idx_bits: [5]F = undefined;
            var rd_idx_bits: [5]F = undefined;

            for (0..5) |i| {
                rs1_idx_bits[i] = if ((decoded.rs1 >> @intCast(i)) & 1 == 1) F.one() else F.zero();
                rs2_idx_bits[i] = if ((decoded.rs2 >> @intCast(i)) & 1 == 1) F.one() else F.zero();
                rd_idx_bits[i] = if ((decoded.rd >> @intCast(i)) & 1 == 1) F.one() else F.zero();
            }

            return .{
                .pc = pc,
                .instruction = instruction_f,
                .rs1 = rs1,
                .rs2 = rs2,
                .rd = rd,
                .imm = imm,
                .memory_addr = memory_addr,
                .memory_value = memory_value,
                .next_pc = next_pc,
                .opcode_bits = opcode_bits,
                .rs1_idx_bits = rs1_idx_bits,
                .rs2_idx_bits = rs2_idx_bits,
                .rd_idx_bits = rd_idx_bits,
            };
        }

        /// Generate full witness vector from trace
        ///
        /// Returns a flat array of field elements suitable for R1CS verification
        pub fn generateWitness(self: *Self, trace: *const ExecutionTrace) ![]F {
            const steps = trace.steps.items;
            if (steps.len == 0) {
                return self.allocator.alloc(F, 0);
            }

            // Witness layout:
            // [1 (constant), public_inputs..., step_witnesses...]
            //
            // Each step has the following witness elements:
            // pc, instruction, rs1, rs2, rd, imm, memory_addr, memory_value, next_pc,
            // 7 opcode bits, 5+5+5 register bits = 31 elements per step

            const elements_per_step: usize = 9 + 7 + 15; // 31 elements

            // Public inputs: initial PC, final PC, cycle count
            const public_inputs_count: usize = 3;

            const total_elements = 1 + public_inputs_count + (steps.len * elements_per_step);
            const witness = try self.allocator.alloc(F, total_elements);

            var idx: usize = 0;

            // Constant 1
            witness[idx] = F.one();
            idx += 1;

            // Public inputs
            witness[idx] = F.fromU64(steps[0].pc); // Initial PC
            idx += 1;
            witness[idx] = F.fromU64(steps[steps.len - 1].next_pc); // Final PC
            idx += 1;
            witness[idx] = F.fromU64(steps.len); // Cycle count
            idx += 1;

            // Step witnesses
            for (steps) |step| {
                const sw = self.generateStepWitness(step);

                witness[idx] = sw.pc;
                idx += 1;
                witness[idx] = sw.instruction;
                idx += 1;
                witness[idx] = sw.rs1;
                idx += 1;
                witness[idx] = sw.rs2;
                idx += 1;
                witness[idx] = sw.rd;
                idx += 1;
                witness[idx] = sw.imm;
                idx += 1;
                witness[idx] = sw.memory_addr;
                idx += 1;
                witness[idx] = sw.memory_value;
                idx += 1;
                witness[idx] = sw.next_pc;
                idx += 1;

                // Opcode bits
                for (sw.opcode_bits) |bit| {
                    witness[idx] = bit;
                    idx += 1;
                }

                // Register index bits
                for (sw.rs1_idx_bits) |bit| {
                    witness[idx] = bit;
                    idx += 1;
                }
                for (sw.rs2_idx_bits) |bit| {
                    witness[idx] = bit;
                    idx += 1;
                }
                for (sw.rd_idx_bits) |bit| {
                    witness[idx] = bit;
                    idx += 1;
                }
            }

            return witness;
        }

        /// Generate memory checking witness
        ///
        /// Creates offline memory checking tuples for all memory accesses
        pub fn generateMemoryWitness(self: *Self, trace: *const ExecutionTrace) !MemoryWitness(F) {
            const steps = trace.steps.items;

            // Use unmanaged ArrayList for Zig 0.15
            var reads: std.ArrayListUnmanaged(MemoryTuple(F)) = .empty;
            var writes: std.ArrayListUnmanaged(MemoryTuple(F)) = .empty;

            for (steps, 0..) |step, i| {
                if (step.memory_addr) |addr| {
                    const tuple = MemoryTuple(F){
                        .address = F.fromU64(addr),
                        .value = F.fromU64(step.memory_value orelse 0),
                        .timestamp = F.fromU64(@intCast(i)),
                    };

                    if (step.is_memory_write) {
                        try writes.append(self.allocator, tuple);
                    } else {
                        try reads.append(self.allocator, tuple);
                    }
                }
            }

            return MemoryWitness(F){
                .reads = try reads.toOwnedSlice(self.allocator),
                .writes = try writes.toOwnedSlice(self.allocator),
                .allocator = self.allocator,
            };
        }
    };
}

/// Memory access tuple for offline memory checking
pub fn MemoryTuple(comptime F: type) type {
    return struct {
        address: F,
        value: F,
        timestamp: F,
    };
}

/// Full memory witness for offline memory checking
pub fn MemoryWitness(comptime F: type) type {
    return struct {
        const Self = @This();

        reads: []MemoryTuple(F),
        writes: []MemoryTuple(F),
        allocator: Allocator,

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.reads);
            self.allocator.free(self.writes);
        }

        /// Total number of memory operations
        pub fn totalOps(self: *const Self) usize {
            return self.reads.len + self.writes.len;
        }
    };
}

test "witness generation from trace" {
    const field = @import("zolt_arith").field;
    const F = field.BN254Scalar;
    const common = @import("../common/mod.zig");
    const allocator = std.testing.allocator;

    var config = common.MemoryConfig{ .program_size = 1024 };
    var emu = tracer.Emulator.init(allocator, &config);
    defer emu.deinit();

    // Simple program: addi x1, x0, 42; addi x2, x1, 8
    const program = [_]u8{
        0x93, 0x00, 0xa0, 0x02, // addi x1, x0, 42
        0x13, 0x81, 0x80, 0x00, // addi x2, x1, 8
    };
    try emu.loadProgram(&program);

    // Execute both instructions
    _ = try emu.step();
    _ = try emu.step();

    // Generate witness
    var wg = WitnessGenerator(F).init(allocator);
    const witness = try wg.generateWitness(&emu.trace);
    defer allocator.free(witness);

    // Verify witness has expected size
    // 1 + 3 public inputs + 2 steps * 31 elements = 66
    try std.testing.expectEqual(@as(usize, 66), witness.len);

    // Verify constant 1
    try std.testing.expect(witness[0].eql(F.one()));

    // Verify cycle count = 2
    try std.testing.expect(witness[3].eql(F.fromU64(2)));
}

test "memory witness generation" {
    const field = @import("zolt_arith").field;
    const F = field.BN254Scalar;
    const common = @import("../common/mod.zig");
    const allocator = std.testing.allocator;

    var config = common.MemoryConfig{ .program_size = 1024 };
    var emu = tracer.Emulator.init(allocator, &config);
    defer emu.deinit();

    // Set up for store instruction
    try emu.registers.write(1, 0x1000); // x1 = address
    try emu.registers.write(2, 42); // x2 = value to store

    // sw x2, 0(x1)
    // Encoding: imm[11:5]=0, rs2=2, rs1=1, funct3=010, imm[4:0]=0, opcode=0100011
    const sw_instr: u32 = 0x0020a023; // sw x2, 0(x1)
    const program = std.mem.asBytes(&sw_instr);
    try emu.loadProgram(program);

    _ = try emu.step();

    // Generate memory witness
    var wg = WitnessGenerator(F).init(allocator);
    var mem_wit = try wg.generateMemoryWitness(&emu.trace);
    defer mem_wit.deinit();

    // Should have 1 write (stores also generate a read for the word at that address)
    try std.testing.expectEqual(@as(usize, 1), mem_wit.writes.len);
    try std.testing.expectEqual(@as(usize, 1), mem_wit.reads.len);

    // Verify write tuple
    try std.testing.expect(mem_wit.writes[0].address.eql(F.fromU64(0x1000)));
}
