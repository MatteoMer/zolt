//! Stage 6 Helper Functions (extracted from stage6_prover.zig)
//!
//! Contains utility functions shared across Stage 6, Stage 7, and instance provers:
//! - Eq polynomial table construction (computeEqTable, computeEqTableParallel)
//! - Polynomial coefficient conversion (addEvalsAsMonomialToCoeffs, etc.)
//! - Bit manipulation and RISC-V decoding (interleaveBits, extractChunkMSB, etc.)
//! - Background memory deallocation (dropInBackground)

const std = @import("std");

const Allocator = std.mem.Allocator;
const ThreadPool = @import("zolt_pool").ThreadPool;

const poly_mod = @import("zolt_arith").poly;
const UniPoly = poly_mod.UniPoly;

const tracer = @import("../../tracer/mod.zig");
const bytecode_entry_mod = @import("bytecode_entries.zig");
const hasLookupTable = bytecode_entry_mod.hasLookupTable;

/// Free a large allocation on a detached background thread so the caller doesn't block.
/// Falls back to synchronous free if thread spawn fails.
/// Supports flat slices ([]T) and slices-of-slices ([][]T).
pub fn dropInBackground(allocator: Allocator, slice: anytype) void {
    const T = @TypeOf(slice);
    const SpawnCtx = struct { alloc: Allocator, ptr: T };
    const info = @typeInfo(T);
    const is_slice_of_slices = comptime blk: {
        if (info != .pointer) break :blk false;
        const child_info = @typeInfo(info.pointer.child);
        break :blk (child_info == .pointer and child_info.pointer.size == .slice);
    };
    const ctx = SpawnCtx{ .alloc = allocator, .ptr = slice };
    const thread = std.Thread.spawn(.{}, struct {
        fn run(c: SpawnCtx) void {
            if (is_slice_of_slices) {
                for (c.ptr) |inner| c.alloc.free(inner);
            }
            c.alloc.free(c.ptr);
        }
    }.run, .{ctx}) catch {
        // Fallback: free synchronously if spawn fails
        if (is_slice_of_slices) {
            for (slice) |inner| allocator.free(inner);
        }
        allocator.free(slice);
        return;
    };
    thread.detach();
}

// =============================================================================
// Helper: Convert evaluations to monomial coefficients and add batch*coeffs to combined_coeffs
// =============================================================================
// Converts [p(0), p(1), ..., p(d)] (Vandermonde evals) to monomial [c0, c1, ..., cd]
// using finite differences for small degrees (d <= 3), then adds batch * c_i to combined_coeffs[i].
pub fn addEvalsAsMonomialToCoeffs(comptime F: type, combined_coeffs: []F, polys: []const F, n_evals: usize, batch_coeff: F) void {
    if (n_evals == 1) {
        // Degree 0: c0 = p(0)
        combined_coeffs[0] = combined_coeffs[0].add(batch_coeff.mul(polys[0]));
    } else if (n_evals == 2) {
        // Degree 1: c0 = p(0), c1 = p(1) - p(0)
        const c0 = polys[0];
        const c1 = polys[1].sub(polys[0]);
        combined_coeffs[0] = combined_coeffs[0].add(batch_coeff.mul(c0));
        combined_coeffs[1] = combined_coeffs[1].add(batch_coeff.mul(c1));
    } else if (n_evals == 3) {
        // Degree 2: c0 = p(0), c2 = (p(2) - 2p(1) + p(0)) / 2, c1 = p(1) - p(0) - c2
        const inv2 = UniPoly(F).INV2;
        const c0 = polys[0];
        const c2 = polys[2].sub(polys[1]).sub(polys[1]).add(polys[0]).mul(inv2);
        const c1 = polys[1].sub(polys[0]).sub(c2);
        combined_coeffs[0] = combined_coeffs[0].add(batch_coeff.mul(c0));
        combined_coeffs[1] = combined_coeffs[1].add(batch_coeff.mul(c1));
        combined_coeffs[2] = combined_coeffs[2].add(batch_coeff.mul(c2));
    } else if (n_evals == 4) {
        // Degree 3: finite differences
        const inv2 = UniPoly(F).INV2;
        const inv6 = F.fromU64(6).inverse().?;
        const c0 = polys[0];
        const d1 = polys[1].sub(polys[0]);
        const d2 = polys[2].sub(polys[1]);
        const d3 = polys[3].sub(polys[2]);
        const dd1 = d2.sub(d1);
        const dd2 = d3.sub(d2);
        const c3 = dd2.sub(dd1).mul(inv6);
        const c2 = dd1.mul(inv2).sub(c3.mul(F.fromU64(3)));
        const c1 = d1.sub(c2).sub(c3);
        combined_coeffs[0] = combined_coeffs[0].add(batch_coeff.mul(c0));
        combined_coeffs[1] = combined_coeffs[1].add(batch_coeff.mul(c1));
        combined_coeffs[2] = combined_coeffs[2].add(batch_coeff.mul(c2));
        combined_coeffs[3] = combined_coeffs[3].add(batch_coeff.mul(c3));
    } else {
        // General case: use Newton forward differences with static buffer
        // Supports up to degree 15 (16 eval points)
        std.debug.assert(n_evals <= 16);
        var dd: [16]F = undefined;
        for (0..n_evals) |i| dd[i] = polys[i];

        // Build forward difference table: dd[k] = k-th order forward difference at 0
        // After processing, dd[k] = Δ^k p(0)
        var coeffs_buf: [16]F = undefined;
        coeffs_buf[0] = dd[0]; // Δ^0 = p(0)

        var order: usize = 1;
        while (order < n_evals) : (order += 1) {
            // Compute order-th forward differences in-place
            var i = n_evals - 1;
            while (i >= order) : (i -= 1) {
                dd[i] = dd[i].sub(dd[i - 1]);
                if (i == order) break;
            }
            coeffs_buf[order] = dd[order]; // Δ^order p(0)
        }

        // Convert Newton forward differences to monomial coefficients
        // Newton form: p(x) = Σ_k Δ^k p(0) * C(x, k)
        // where C(x, k) = x(x-1)...(x-k+1) / k!
        // We need to convert to monomial c0 + c1*x + c2*x^2 + ...
        // Use the fact that Δ^k p(0) / k! is the leading coefficient contribution
        // Actually, the simplest approach for general n: use the Vandermonde solver result
        // which is already available via fromEvalsVandermonde. But since this is a non-allocating
        // path, we use Sterling numbers of the first kind.
        //
        // Actually for the general case, let's just compute monomial coefficients directly
        // from the forward differences using the Stirling number relationship.
        // c_j = Σ_{k=j}^{d} S1(k, j) * Δ^k p(0) / k!
        // This is complex. For now, fall back to evaluating the Newton form at integer points
        // and using the same approach as vandermondeToCompressed for n > 4.
        //
        // Simpler: we have forward differences. Convert via the standard formula:
        // The Newton forward difference interpolation gives:
        // c_k = Σ_{j=0}^{k} (-1)^{k-j} C(k,j) * Δ^j p(0) / ... no, this is circular.
        //
        // Let's just directly use finite-difference-to-monomial conversion:
        // Start with Newton basis coefficients dd[0..n] = [Δ^0 p(0)/0!, Δ^1 p(0)/1!, ...]
        // and convert to monomial via the standard algorithm.

        // Divide by factorials to get Newton basis coefficients
        var fact = F.one();
        for (1..n_evals) |k| {
            fact = fact.mul(F.fromU64(@intCast(k)));
            coeffs_buf[k] = coeffs_buf[k].mul(fact.inverse().?);
        }

        // Convert Newton basis to monomial: c(x) = Σ a_k * x*(x-1)*...*(x-k+1)
        // Process from highest to lowest degree, expanding x*(x-1)*...*(x-k+1) into monomials.
        // Use the recurrence: multiply running polynomial by (x - k) at each step.
        var mono: [16]F = .{F.zero()} ** 16;
        mono[0] = coeffs_buf[0];

        for (1..n_evals) |k| {
            // We need to add coeffs_buf[k] * x*(x-1)*...*(x-k+1) to mono
            // Build the falling factorial x*(x-1)*...*(x-k+1) incrementally
            // ff[k] = ff[k-1] * (x - (k-1))
            // We maintain ff_mono[0..k] = monomial coefficients of x*(x-1)*...*(x-k+1)
            // Start: ff_mono = [0, 1] for x
            // Multiply by (x - j) for j = 1, 2, ..., k-1
            var ff: [16]F = .{F.zero()} ** 16;
            ff[1] = F.one(); // x
            for (1..k) |j| {
                // Multiply ff by (x - j): new[i] = ff[i-1] - j*ff[i]
                const neg_j = F.zero().sub(F.fromU64(@intCast(j)));
                var i_rev = j + 1;
                while (i_rev > 0) {
                    i_rev -= 1;
                    const prev = if (i_rev > 0) ff[i_rev - 1] else F.zero();
                    ff[i_rev] = prev.add(neg_j.mul(ff[i_rev]));
                }
            }
            // Add coeffs_buf[k] * ff to mono
            for (0..k + 1) |i| {
                mono[i] = mono[i].add(coeffs_buf[k].mul(ff[i]));
            }
        }

        // Add batch * mono to combined_coeffs
        for (0..n_evals) |i| {
            combined_coeffs[i] = combined_coeffs[i].add(batch_coeff.mul(mono[i]));
        }
    }
}

// =============================================================================
// Helper: Add variable-length instance evals to combined_evals with interpolation (LEGACY)
// =============================================================================
// All evaluation arrays use Vandermonde format: [p(0), p(1), ..., p(d)]
// (evaluations at consecutive integer points, no p_inf)
pub fn addInstanceEvalsToCombibed(comptime F: type, combined_evals: []F, polys: []const F, batch_coeff: F, num_evals: usize) void {
    const inst_n_evals = polys.len;

    if (inst_n_evals >= num_evals) {
        // Instance has enough eval points - just add the first num_evals
        for (0..num_evals) |k| {
            combined_evals[k] = combined_evals[k].add(batch_coeff.mul(polys[k]));
        }
    } else {
        // Instance has fewer eval points - need Lagrange interpolation for missing points
        // polys format (Vandermonde): [p(0), p(1), ..., p(inst_n_evals-1)]
        // Need to interpolate p(inst_n_evals), ..., p(num_evals-1)

        // Add known evaluation points
        for (0..inst_n_evals) |k| {
            combined_evals[k] = combined_evals[k].add(batch_coeff.mul(polys[k]));
        }

        // Lagrange interpolation for missing points
        for (inst_n_evals..num_evals) |k| {
            const x = F.fromU64(@intCast(k));
            var lagrange_val = F.zero();
            for (0..inst_n_evals) |m| {
                var basis = F.one();
                const xm = F.fromU64(@intCast(m));
                for (0..inst_n_evals) |n| {
                    if (n != m) {
                        const xn = F.fromU64(@intCast(n));
                        basis = basis.mul(x.sub(xn)).mul(xm.sub(xn).inverse().?);
                    }
                }
                lagrange_val = lagrange_val.add(basis.mul(polys[m]));
            }
            combined_evals[k] = combined_evals[k].add(batch_coeff.mul(lagrange_val));
        }
    }
}

/// Add fixed-size instance evaluations to combined (for degree-3 instances like Hamming)
// All evaluation arrays use Vandermonde format: [p(0), p(1), ..., p(d)]
pub fn addFixedEvalsToCombibed(comptime F: type, combined_evals: []F, polys: []const F, n_polys: usize, batch_coeff: F, num_evals: usize) void {
    if (n_polys >= num_evals) {
        // Instance has enough eval points - add the first num_evals
        for (0..num_evals) |k| {
            combined_evals[k] = combined_evals[k].add(batch_coeff.mul(polys[k]));
        }
    } else {
        // Instance has fewer eval points - need Lagrange interpolation for missing points
        for (0..n_polys) |k| {
            combined_evals[k] = combined_evals[k].add(batch_coeff.mul(polys[k]));
        }

        // Lagrange interpolation for missing points
        for (n_polys..num_evals) |k| {
            const x = F.fromU64(@intCast(k));
            var lagrange_val = F.zero();
            for (0..n_polys) |m| {
                var basis = F.one();
                const xm = F.fromU64(@intCast(m));
                for (0..n_polys) |n| {
                    if (n != m) {
                        const xn = F.fromU64(@intCast(n));
                        basis = basis.mul(x.sub(xn)).mul(xm.sub(xn).inverse().?);
                    }
                }
                lagrange_val = lagrange_val.add(basis.mul(polys[m]));
            }
            combined_evals[k] = combined_evals[k].add(batch_coeff.mul(lagrange_val));
        }
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Compute eq polynomial table: eq(r, j) for all j in [0, 2^n_vars)
/// r is in BIG_ENDIAN order (r[0] is the most significant variable)
pub fn computeEqTable(comptime F: type, allocator: Allocator, r: []const F, n_vars: usize) ![]F {
    return computeEqTableParallel(F, allocator, r, n_vars, null);
}

/// Compute eq polynomial table with optional parallel inner loops.
/// Same as computeEqTable but parallelizes large levels via ThreadPool.
pub fn computeEqTableParallel(comptime F: type, allocator: Allocator, r: []const F, n_vars: usize, pool: ?*ThreadPool) ![]F {
    const size: usize = @as(usize, 1) << @intCast(n_vars);
    var table = try allocator.alloc(F, size);

    table[0] = F.one();

    for (0..n_vars) |i| {
        const r_i = r[i];
        const cur_size: usize = @as(usize, 1) << @intCast(i);

        if (pool != null and cur_size >= 256) {
            // Parallel: forward iteration, writes to disjoint halves [0..cur_size) and [cur_size..2*cur_size)
            const Ctx = struct {
                tbl: []F,
                ri: F,
                cs: usize,
            };
            const ctx = Ctx{ .tbl = table, .ri = r_i, .cs = cur_size };
            pool.?.parallelForForce(cur_size, ctx, struct {
                fn f(c: Ctx, j: usize) void {
                    const x = c.tbl[j];
                    const y = x.mul(c.ri);
                    c.tbl[j + c.cs] = y;
                    c.tbl[j] = x.sub(y);
                }
            }.f);
        } else {
            // Sequential: backward iteration (original)
            var j: usize = cur_size;
            while (j > 0) {
                j -= 1;
                const x = table[j];
                const y = x.mul(r_i);
                table[j + cur_size] = y;
                table[j] = x.sub(y);
            }
        }
    }

    return table;
}

/// Convert signed i128 to field element
pub fn fieldFromI128(comptime F: type, val: i128) F {
    if (val >= 0) {
        return F.fromU128(@intCast(val));
    } else {
        return F.fromU128(@intCast(-val)).neg();
    }
}

/// Extract chunk from address value using MSB-first ordering (matching Jolt)
/// chunk_idx=0 is the most significant chunk
pub fn extractChunkMSB(addr: u64, chunk_idx: usize, total_chunks: usize, log_k_chunk: usize) usize {
    // Jolt: shift = log_k_chunk * (d - 1 - chunk_idx)
    const shift_amount = log_k_chunk * (total_chunks - 1 - chunk_idx);
    if (shift_amount >= 64) return 0;
    const shift: u6 = @intCast(shift_amount);
    const mask: u64 = (@as(u64, 1) << @intCast(log_k_chunk)) - 1;
    return @intCast((addr >> shift) & mask);
}

/// Interleave bits of two 64-bit values to form a 128-bit lookup index
/// Matches Jolt's interleave_bits(even_bits, odd_bits): result = (even << 1) | odd
/// So even_bits (rs1) go to odd bit positions (1,3,5,...,127)
/// and odd_bits (rs2) go to even bit positions (0,2,4,...,126)
pub fn interleaveBits(rs1: u64, rs2: u64) u128 {
    // Spread rs1 bits to odd positions
    var x: u128 = @intCast(rs1);
    x = (x | (x << 32)) & 0x0000_0000_FFFF_FFFF_0000_0000_FFFF_FFFF;
    x = (x | (x << 16)) & 0x0000_FFFF_0000_FFFF_0000_FFFF_0000_FFFF;
    x = (x | (x << 8)) & 0x00FF_00FF_00FF_00FF_00FF_00FF_00FF_00FF;
    x = (x | (x << 4)) & 0x0F0F_0F0F_0F0F_0F0F_0F0F_0F0F_0F0F_0F0F;
    x = (x | (x << 2)) & 0x3333_3333_3333_3333_3333_3333_3333_3333;
    x = (x | (x << 1)) & 0x5555_5555_5555_5555_5555_5555_5555_5555;

    // Spread rs2 bits to even positions
    var y: u128 = @intCast(rs2);
    y = (y | (y << 32)) & 0x0000_0000_FFFF_FFFF_0000_0000_FFFF_FFFF;
    y = (y | (y << 16)) & 0x0000_FFFF_0000_FFFF_0000_FFFF_0000_FFFF;
    y = (y | (y << 8)) & 0x00FF_00FF_00FF_00FF_00FF_00FF_00FF_00FF;
    y = (y | (y << 4)) & 0x0F0F_0F0F_0F0F_0F0F_0F0F_0F0F_0F0F_0F0F;
    y = (y | (y << 2)) & 0x3333_3333_3333_3333_3333_3333_3333_3333;
    y = (y | (y << 1)) & 0x5555_5555_5555_5555_5555_5555_5555_5555;

    return (x << 1) | y;
}

/// Decode sign-extended immediate from RISC-V instruction encoding, returned as u64 (two's complement).
/// This matches Jolt's `to_instruction_inputs()` which sign-extends the immediate value.
pub fn decodeImmediateU64(instr: u32) u64 {
    const opcode: u8 = @truncate(instr & 0x7f);
    switch (opcode) {
        // I-type: imm[11:0] at bits [31:20], sign-extended
        0x13, 0x03, 0x67, 0x1b, 0x73 => {
            const imm12: u32 = instr >> 20;
            const imm_signed: i64 = @as(i64, @as(i32, @bitCast(imm12 << 20)) >> 20);
            return @bitCast(imm_signed);
        },
        // S-type: imm[11:5] at [31:25], imm[4:0] at [11:7], sign-extended
        0x23 => {
            const imm11_5 = (instr >> 25) & 0x7f;
            const imm4_0 = (instr >> 7) & 0x1f;
            const imm12: u32 = (imm11_5 << 5) | imm4_0;
            const imm_signed: i64 = @as(i64, @as(i32, @bitCast(imm12 << 20)) >> 20);
            return @bitCast(imm_signed);
        },
        // B-type: imm[12|10:5] at [31:25], imm[4:1|11] at [11:7], sign-extended, *2
        0x63 => {
            const imm12 = (instr >> 31) & 1;
            const imm10_5 = (instr >> 25) & 0x3f;
            const imm4_1 = (instr >> 8) & 0xf;
            const imm11 = (instr >> 7) & 1;
            const imm13: u32 = (imm12 << 12) | (imm11 << 11) | (imm10_5 << 5) | (imm4_1 << 1);
            const imm_signed: i64 = @as(i64, @as(i32, @bitCast(imm13 << 19)) >> 19);
            return @bitCast(imm_signed);
        },
        // U-type: imm[31:12] at [31:12], shifted left by 12, SIGN-EXTENDED to 64 bits
        // Matches Jolt's FormatU.parse: `as i32 as i64 as u64`
        0x37, 0x17 => {
            const imm_upper: u32 = instr & 0xFFFFF000;
            return @bitCast(@as(i64, @as(i32, @bitCast(imm_upper))));
        },
        // J-type: imm[20|10:1|11|19:12] at [31:12], sign-extended, *2
        0x6f => {
            const imm20 = (instr >> 31) & 1;
            const imm10_1 = (instr >> 21) & 0x3ff;
            const imm11 = (instr >> 20) & 1;
            const imm19_12 = (instr >> 12) & 0xff;
            const imm21: u32 = (imm20 << 20) | (imm19_12 << 12) | (imm11 << 11) | (imm10_1 << 1);
            const imm_signed: i64 = @as(i64, @as(i32, @bitCast(imm21 << 11)) >> 11);
            return @bitCast(imm_signed);
        },
        else => return 0,
    }
}

/// Compute the 128-bit lookup index for a trace step.
///
/// This matches Jolt's per-instruction `to_lookup_index()` method:
/// - AddOperands instructions (ADD, ADDI, etc.): returns raw sum as u128 (NO interleaving)
/// - SubtractOperands instructions (SUB, SUBW): returns raw shifted difference as u128
/// - MultiplyOperands instructions (MUL, MULHU): returns raw product as u128
/// - Standard instructions (XOR, AND, OR, SLT, branches): returns interleave_bits(x, y)
/// - No-lookup instructions (Load, Store, SLL, SRL): returns 0
/// - NoOp cycles: returns 0
pub fn computeLookupIndex(step: tracer.TraceStep) u128 {
    if (step.is_noop and !step.is_termination_store) return 0;

    const instr = step.instruction;
    const opcode: u8 = @truncate(instr & 0x7f);
    const funct3: u3 = @truncate((instr >> 12) & 0x7);
    const funct7: u7 = @truncate(instr >> 25);

    // Check if instruction has a lookup table at all
    if (!hasLookupTable(opcode, funct3, funct7)) return 0;

    // Virtual opcodes: handle specially since they don't follow standard RISC-V encoding
    if (opcode == 0x0B) {
        // VirtualSignExtendWord: AddOperands → rs1 + 0 = rs1
        // Jolt's to_lookup_index() returns rs1 directly (no interleaving)
        return @as(u128, step.rs1_value);
    }
    if (opcode == 0x2B) {
        if (funct3 == 0) {
            // VirtualMULI: MultiplyOperands → rs1 * (1 << shamt)
            const shamt_raw: u32 = instr >> 20;
            const shamt: u6 = @truncate(shamt_raw & 0x3F);
            const multiplier: u128 = @as(u128, 1) << shamt;
            return @as(u128, step.rs1_value) * multiplier;
        } else {
            // VirtualPow2 (funct3=1), VirtualShiftRightBitmask (funct3=2): AddOperands → rs1 + 0 = rs1
            return @as(u128, step.rs1_value);
        }
    }
    if (opcode == 0x5B) {
        if (step.rs2_read) {
            // VirtualSRL/VirtualSRA R-type: interleaved(rs1_value, rs2_value)
            return interleaveBits(step.rs1_value, step.rs2_value);
        } else {
            // VirtualSRLI/VirtualSRAI I-type: interleaved(rs1_value, bitmask)
            const total_shift_raw: u32 = instr >> 20;
            const total_shift: u7 = @truncate(total_shift_raw & 0x3F);
            const ones: u128 = (@as(u128, 1) << @intCast(64 - @as(u8, total_shift))) - 1;
            const bitmask: u64 = @truncate(ones << total_shift);
            return interleaveBits(step.rs1_value, bitmask);
        }
    }
    if (opcode == 0x02) {
        // VirtualAdvice: the lookup index is the advice value (rd_value)
        // Jolt's to_lookup_index() returns the second operand which is the advice value
        return @as(u128, step.rd_value);
    }
    if (opcode == 0x22) {
        if (funct3 == 2 or funct3 == 3) {
            // VirtualAssertHalfwordAlignment/WordAlignment: AddOperands → rs1 + imm
            const imm_raw: u32 = instr >> 20;
            const imm_signed: i64 = @as(i64, @as(i32, @bitCast(imm_raw << 20)) >> 20);
            return @as(u128, step.rs1_value +% @as(u64, @bitCast(imm_signed)));
        } else {
            // VirtualAssertEQ (funct3=0) / VirtualAssertValidDiv0 (funct3=1): interleaved
            return interleaveBits(step.rs1_value, step.rs2_value);
        }
    }
    if (opcode == 0x42) {
        // VirtualZeroExtendWord: AddOperands → rs1 + 0 = rs1
        // Jolt's to_lookup_index() returns rs1 directly (like SignExtendWord)
        return @as(u128, step.rs1_value);
    }
    if (opcode == 0x62) {
        // VirtualAssertValidUnsignedRemainder: interleaved(rs1_value, rs2_value)
        // LeftOperandIsRs1Value, RightOperandIsRs2Value → interleave
        return interleaveBits(step.rs1_value, step.rs2_value);
    }

    // Determine left_input and right_input (matching Jolt's to_instruction_inputs)
    const left_is_rs1: bool = switch (opcode) {
        0x33, 0x3b, 0x23, 0x63, 0x13, 0x03, 0x67, 0x1b => true,
        else => false,
    };
    const left_is_pc: bool = switch (opcode) {
        0x17, 0x6f => true,
        else => false,
    };
    const right_is_rs2: bool = switch (opcode) {
        0x33, 0x63, 0x3b => true,
        else => false,
    };
    const right_is_imm: bool = switch (opcode) {
        0x13, 0x03, 0x67, 0x23, 0x37, 0x17, 0x6f, 0x1b => true,
        else => false,
    };

    var left_input: u64 = 0;
    if (left_is_rs1) left_input = step.rs1_value;
    if (left_is_pc) left_input = step.unexpanded_pc;

    var right_input: u64 = 0;
    if (right_is_rs2) right_input = step.rs2_value;
    if (right_is_imm) right_input = decodeImmediateU64(instr);

    // Now compute the lookup index based on the instruction's operand mode
    switch (opcode) {
        0x33 => { // R-type
            if (funct7 == 0x01) {
                // M-extension
                if (funct3 == 0x0) {
                    // MUL: MultiplyOperands → raw product
                    return @as(u128, left_input) * @as(u128, right_input);
                } else if (funct3 == 0x3) {
                    // MULHU: MultiplyOperands → raw product
                    return @as(u128, left_input) * @as(u128, right_input);
                } else {
                    // Other M-ext: interleaved
                    return interleaveBits(left_input, right_input);
                }
            } else if (funct7 == 0x20 and funct3 == 0x0) {
                // SUB: SubtractOperands → x + (2^64 - y)
                return @as(u128, left_input) + (@as(u128, 1) << 64) - @as(u128, right_input);
            } else if (funct7 == 0 and funct3 == 0x0) {
                // ADD: AddOperands → raw sum
                return @as(u128, left_input) + @as(u128, right_input);
            } else {
                // Other R-type (AND, OR, XOR, SLT, SLTU): interleaved
                return interleaveBits(left_input, right_input);
            }
        },
        0x13 => { // I-type ALU
            if (funct3 == 0) {
                // ADDI: AddOperands → raw sum
                return @as(u128, left_input) + @as(u128, right_input);
            } else {
                // SLLI, SLTI, SLTIU, XORI, SRLI, SRAI, ORI, ANDI: interleaved
                return interleaveBits(left_input, right_input);
            }
        },
        0x37 => { // LUI: AddOperands → immediate directly (left=0)
            return @as(u128, left_input) + @as(u128, right_input);
        },
        0x17 => { // AUIPC: AddOperands → PC + imm
            return @as(u128, left_input) + @as(u128, right_input);
        },
        0x6f => { // JAL: AddOperands → PC + imm
            return @as(u128, left_input) + @as(u128, right_input);
        },
        0x67 => { // JALR: AddOperands → rs1 + imm
            return @as(u128, left_input) + @as(u128, right_input);
        },
        0x1b => { // I-type word ALU
            if (funct3 == 0) {
                // ADDIW: AddOperands → raw sum
                return @as(u128, left_input) + @as(u128, right_input);
            } else {
                // SLLIW, SRLIW, SRAIW: interleaved
                return interleaveBits(left_input, right_input);
            }
        },
        0x3b => { // OP-32
            if (funct3 == 0 and funct7 == 0) {
                // ADDW: AddOperands → raw sum
                return @as(u128, left_input) + @as(u128, right_input);
            } else if (funct3 == 0 and funct7 == 0x20) {
                // SUBW: SubtractOperands → x + (2^64 - y)
                return @as(u128, left_input) + (@as(u128, 1) << 64) - @as(u128, right_input);
            } else {
                // Other 0x3b: interleaved
                return interleaveBits(left_input, right_input);
            }
        },
        0x63 => { // Branch: interleaved
            return interleaveBits(left_input, right_input);
        },
        else => {
            // Default: interleaved
            return interleaveBits(left_input, right_input);
        },
    }
}

/// Get lookup index chunk from trace step.
/// This matches Jolt's lookup_index_chunk with instruction_shifts (MSB-first ordering).
/// Uses the instruction-type-aware computeLookupIndex to correctly handle
/// AddOperands, SubtractOperands, and MultiplyOperands instructions.
pub fn getLookupChunkInterleaved(step: tracer.TraceStep, chunk_idx: usize, log_k_chunk: usize, instruction_d: usize) usize {
    // Build the correct 128-bit lookup index based on instruction type
    const lookup_index = computeLookupIndex(step);

    // MSB-first: shift = log_k_chunk * (instruction_d - 1 - chunk_idx)
    const shift_amount = log_k_chunk * (instruction_d - 1 - chunk_idx);
    if (shift_amount >= 128) return 0;
    const shift: u7 = @intCast(shift_amount);
    const mask: u128 = (@as(u128, 1) << @intCast(log_k_chunk)) - 1;
    return @intCast((lookup_index >> shift) & mask);
}

// =============================================================================
// Tests
// =============================================================================

const testing = std.testing;
const BN254Scalar = @import("zolt_arith").field.BN254Scalar;

test "split-eq factorization: eq_lo * eq_hi = eq_full" {
    // Verify the core split-eq identity:
    //   eq(r, x) = eq(r_lo, x_lo) * eq(r_hi, x_hi)
    // where x = x_lo + x_hi << prefix_n_vars
    //
    // computeEqTable takes BE input r[0..n], output table[j] has bit i → r[i].
    // For x = x_lo | (x_hi << prefix_n_vars):
    //   bits 0..prefix_n_vars-1 (x_lo) → r_be[0..prefix_n_vars]
    //   bits prefix_n_vars..n_vars-1 (x_hi) → r_be[prefix_n_vars..n_vars]
    const allocator = testing.allocator;
    const F = BN254Scalar;

    const n_vars = 4;
    const prefix_n_vars = 2;
    const suffix_n_vars = 2;
    const T: usize = 1 << n_vars;
    const prefix_len: usize = 1 << prefix_n_vars;
    const suffix_len: usize = 1 << suffix_n_vars;

    // Full BE challenge
    var r_be = [4]F{ F.fromU64(17), F.fromU64(31), F.fromU64(7), F.fromU64(53) };
    const eq_full = try computeEqTable(F, allocator, &r_be, n_vars);
    defer allocator.free(eq_full);

    // Split: prefix (x_lo bits) uses r_be[0..prefix_n_vars]
    var r_lo_be = [2]F{ r_be[0], r_be[1] };
    const eq_lo = try computeEqTable(F, allocator, &r_lo_be, prefix_n_vars);
    defer allocator.free(eq_lo);

    // Suffix (x_hi bits) uses r_be[prefix_n_vars..n_vars]
    var r_hi_be = [2]F{ r_be[2], r_be[3] };
    const eq_hi = try computeEqTable(F, allocator, &r_hi_be, suffix_n_vars);
    defer allocator.free(eq_hi);

    // Verify: eq_full[x] == eq_lo[x_lo] * eq_hi[x_hi] for all x
    for (0..T) |x| {
        const x_lo = x & (prefix_len - 1);
        const x_hi = x >> prefix_n_vars;
        const product = eq_lo[x_lo].mul(eq_hi[x_hi]);
        try testing.expect(eq_full[x].eql(product));
    }

    // Also verify: Σ_{x_hi} f(x_lo, x_hi) * eq_hi[x_hi] correctly folds suffix dimension
    var folded = [_]F{F.zero()} ** prefix_len;
    for (0..prefix_len) |x_lo| {
        for (0..suffix_len) |x_hi| {
            const x = x_lo + (x_hi << prefix_n_vars);
            folded[x_lo] = folded[x_lo].add(eq_hi[x_hi].mul(F.fromU64(@intCast(x))));
        }
    }
    // Verify: Σ_x_lo P[x_lo] * folded[x_lo] == Σ_x eq_full[x] * f(x)
    var sum_pq = F.zero();
    for (0..prefix_len) |x_lo| {
        sum_pq = sum_pq.add(eq_lo[x_lo].mul(folded[x_lo]));
    }
    var sum_direct = F.zero();
    for (0..T) |x| {
        sum_direct = sum_direct.add(eq_full[x].mul(F.fromU64(@intCast(x))));
    }
    try testing.expect(sum_pq.eql(sum_direct));
}

test "split-eq bind Phase 1 then Phase 2 matches flat eq bind" {
    // Verify that binding a split eq (Phase 1 prefix, then Phase 2 suffix)
    // produces the same result as binding the flat eq table.
    const allocator = testing.allocator;
    const F = BN254Scalar;

    const n_vars = 4;
    const prefix_n_vars = 2;
    const suffix_n_vars = 2;
    const prefix_len: usize = 1 << prefix_n_vars;
    const suffix_len: usize = 1 << suffix_n_vars;

    var r_be = [4]F{ F.fromU64(5), F.fromU64(13), F.fromU64(3), F.fromU64(19) };
    const challenges = [4]F{ F.fromU64(7), F.fromU64(11), F.fromU64(2), F.fromU64(17) };

    // Build flat eq table and bind sequentially
    var eq_flat = try computeEqTable(F, allocator, &r_be, n_vars);
    defer allocator.free(eq_flat);

    var flat_len: usize = 1 << n_vars;
    for (challenges) |ch| {
        const half = flat_len / 2;
        for (0..half) |j| {
            eq_flat[j] = eq_flat[2 * j].add(ch.mul(eq_flat[2 * j + 1].sub(eq_flat[2 * j])));
        }
        flat_len = half;
    }
    const flat_final = eq_flat[0];

    // Split: prefix uses r_be[0..prefix_n_vars], suffix uses r_be[prefix_n_vars..]
    var r_lo_be = [2]F{ r_be[0], r_be[1] };
    var eq_lo = try computeEqTable(F, allocator, &r_lo_be, prefix_n_vars);
    defer allocator.free(eq_lo);

    var r_hi_be = [2]F{ r_be[2], r_be[3] };
    var eq_hi = try computeEqTable(F, allocator, &r_hi_be, suffix_n_vars);
    defer allocator.free(eq_hi);

    // Phase 1: bind prefix rounds on eq_lo
    var lo_len = prefix_len;
    for (0..prefix_n_vars) |round| {
        const half = lo_len / 2;
        for (0..half) |j| {
            eq_lo[j] = eq_lo[2 * j].add(challenges[round].mul(eq_lo[2 * j + 1].sub(eq_lo[2 * j])));
        }
        lo_len = half;
    }
    const eq_lo_scalar = eq_lo[0];

    // Phase 2: scale eq_hi by eq_lo scalar and bind suffix rounds
    for (0..suffix_len) |j| {
        eq_hi[j] = eq_hi[j].mul(eq_lo_scalar);
    }
    var hi_len = suffix_len;
    for (0..suffix_n_vars) |round| {
        const half = hi_len / 2;
        for (0..half) |j| {
            eq_hi[j] = eq_hi[2 * j].add(challenges[prefix_n_vars + round].mul(eq_hi[2 * j + 1].sub(eq_hi[2 * j])));
        }
        hi_len = half;
    }
    const split_final = eq_hi[0];

    try testing.expect(flat_final.eql(split_final));
}

test "P*Q sum matches flat polynomial sum" {
    // Verify that Σ P[x_lo] * Q[x_lo] == Σ_x eq(r, x) * f(x)
    // where Q[x_lo] = Σ_{x_hi} eq_hi(r_hi, x_hi) * f(x_lo, x_hi)
    // This is the IncClaimReduction Phase 1 correctness property.
    const allocator = testing.allocator;
    const F = BN254Scalar;

    const n_vars = 6;
    const prefix_n_vars = 3;
    const suffix_n_vars = 3;
    const T: usize = 1 << n_vars;
    const prefix_len: usize = 1 << prefix_n_vars;
    const suffix_len: usize = 1 << suffix_n_vars;

    var r_be = [6]F{
        F.fromU64(3),  F.fromU64(7),  F.fromU64(11),
        F.fromU64(17), F.fromU64(23), F.fromU64(29),
    };

    const eq_full = try computeEqTable(F, allocator, &r_be, n_vars);
    defer allocator.free(eq_full);

    // Prefix uses r_be[0..prefix_n_vars], suffix uses r_be[prefix_n_vars..]
    var r_lo_be = [3]F{ r_be[0], r_be[1], r_be[2] };
    const eq_lo = try computeEqTable(F, allocator, &r_lo_be, prefix_n_vars);
    defer allocator.free(eq_lo);

    var r_hi_be = [3]F{ r_be[3], r_be[4], r_be[5] };
    const eq_hi = try computeEqTable(F, allocator, &r_hi_be, suffix_n_vars);
    defer allocator.free(eq_hi);

    // f(x) = x^2 + 3x + 1 (arbitrary polynomial for testing)
    var f_vals = try allocator.alloc(F, T);
    defer allocator.free(f_vals);
    for (0..T) |x| {
        const xf = F.fromU64(@intCast(x));
        f_vals[x] = xf.mul(xf).add(F.fromU64(3).mul(xf)).add(F.one());
    }

    // Q[x_lo] = Σ_{x_hi} eq_hi[x_hi] * f(x_lo + x_hi << prefix_n_vars)
    var Q = try allocator.alloc(F, prefix_len);
    defer allocator.free(Q);
    for (0..prefix_len) |x_lo| {
        Q[x_lo] = F.zero();
        for (0..suffix_len) |x_hi| {
            const x = x_lo + (x_hi << prefix_n_vars);
            Q[x_lo] = Q[x_lo].add(eq_hi[x_hi].mul(f_vals[x]));
        }
    }

    // Σ P[x_lo] * Q[x_lo]
    var sum_pq = F.zero();
    for (0..prefix_len) |x_lo| {
        sum_pq = sum_pq.add(eq_lo[x_lo].mul(Q[x_lo]));
    }

    // Σ eq_full[x] * f(x)
    var sum_direct = F.zero();
    for (0..T) |x| {
        sum_direct = sum_direct.add(eq_full[x].mul(f_vals[x]));
    }

    try testing.expect(sum_pq.eql(sum_direct));
}

test "P*Q Phase 1 sumcheck round polynomial matches flat" {
    // Verify that the Phase 1 round polynomial from the P*Q factorization
    // produces the same evaluations as computing from the flat polynomial.
    const allocator = testing.allocator;
    const F = BN254Scalar;

    const n_vars = 4;
    const prefix_n_vars = 2;
    const suffix_n_vars = 2;
    const T: usize = 1 << n_vars;
    const prefix_len: usize = 1 << prefix_n_vars;
    const suffix_len: usize = 1 << suffix_n_vars;

    var r_be = [4]F{ F.fromU64(5), F.fromU64(13), F.fromU64(3), F.fromU64(19) };

    // Build flat polynomial: poly[x] = eq(r, x) * f(x)
    const eq_full = try computeEqTable(F, allocator, &r_be, n_vars);
    defer allocator.free(eq_full);

    // f(x) = x + 1
    var poly = try allocator.alloc(F, T);
    defer allocator.free(poly);
    for (0..T) |x| {
        poly[x] = eq_full[x].mul(F.fromU64(@intCast(x + 1)));
    }

    // Flat round 1: p(0) = Σ poly[2j], p(1) = Σ poly[2j+1]
    var flat_p0 = F.zero();
    var flat_p1 = F.zero();
    for (0..T / 2) |j| {
        flat_p0 = flat_p0.add(poly[2 * j]);
        flat_p1 = flat_p1.add(poly[2 * j + 1]);
    }

    // Split: P * Q version (prefix = r_be[0..2], suffix = r_be[2..4])
    var r_lo_be = [2]F{ r_be[0], r_be[1] };
    const P = try computeEqTable(F, allocator, &r_lo_be, prefix_n_vars);
    defer allocator.free(P);

    var r_hi_be = [2]F{ r_be[2], r_be[3] };
    const eq_hi = try computeEqTable(F, allocator, &r_hi_be, suffix_n_vars);
    defer allocator.free(eq_hi);

    var Q = try allocator.alloc(F, prefix_len);
    defer allocator.free(Q);
    for (0..prefix_len) |x_lo| {
        Q[x_lo] = F.zero();
        for (0..suffix_len) |x_hi| {
            const x = x_lo + (x_hi << prefix_n_vars);
            Q[x_lo] = Q[x_lo].add(eq_hi[x_hi].mul(F.fromU64(@intCast(x + 1))));
        }
    }

    // Phase 1 round 1: p(t) = Σ_{x_lo} P(x_lo, t) * Q(x_lo, t)
    // P(x_lo, 0) = P[2*x_lo], P(x_lo, 1) = P[2*x_lo+1] (standard MLE bind)
    // Q same structure
    var split_p0 = F.zero();
    var split_p1 = F.zero();
    const half = prefix_len / 2;
    for (0..half) |j| {
        split_p0 = split_p0.add(P[2 * j].mul(Q[2 * j]));
        split_p1 = split_p1.add(P[2 * j + 1].mul(Q[2 * j + 1]));
    }

    try testing.expect(flat_p0.eql(split_p0));
    try testing.expect(flat_p1.eql(split_p1));
}

test "HammingBooleanity split-eq: Phase 1 sum matches flat" {
    // HammingBooleanity computes Σ_x eq(r, x) * H(x) * (H(x) - 1)
    // Verify split-eq Phase 1 round poly matches flat computation.
    const allocator = testing.allocator;
    const F = BN254Scalar;

    const n_vars = 4;
    const prefix_n_vars = 2;
    const suffix_n_vars = 2;
    const T: usize = 1 << n_vars;
    const prefix_len: usize = 1 << prefix_n_vars;
    const suffix_len: usize = 1 << suffix_n_vars;

    var r_be = [4]F{ F.fromU64(11), F.fromU64(23), F.fromU64(7), F.fromU64(41) };

    // Build flat eq
    const eq_full = try computeEqTable(F, allocator, &r_be, n_vars);
    defer allocator.free(eq_full);

    // H(x) = some test values (simulating Hamming weight or similar)
    var H = [16]F{
        F.fromU64(0), F.fromU64(1), F.fromU64(1), F.fromU64(2),
        F.fromU64(1), F.fromU64(2), F.fromU64(2), F.fromU64(3),
        F.fromU64(1), F.fromU64(2), F.fromU64(2), F.fromU64(3),
        F.fromU64(2), F.fromU64(3), F.fromU64(3), F.fromU64(4),
    };

    // Flat sum: Σ eq(r,x) * H(x) * (H(x) - 1) for degree 3 sumcheck
    // Round 1: p(t) at t=0 and t=1
    var flat_p0 = F.zero();
    var flat_p1 = F.zero();
    for (0..T / 2) |j| {
        flat_p0 = flat_p0.add(eq_full[2 * j].mul(H[2 * j]).mul(H[2 * j].sub(F.one())));
        flat_p1 = flat_p1.add(eq_full[2 * j + 1].mul(H[2 * j + 1]).mul(H[2 * j + 1].sub(F.one())));
    }

    // Split-eq: prefix = r_be[0..2], suffix = r_be[2..4]
    var r_lo_be = [2]F{ r_be[0], r_be[1] };
    const eq_lo = try computeEqTable(F, allocator, &r_lo_be, prefix_n_vars);
    defer allocator.free(eq_lo);

    var r_hi_be = [2]F{ r_be[2], r_be[3] };
    const eq_hi = try computeEqTable(F, allocator, &r_hi_be, suffix_n_vars);
    defer allocator.free(eq_hi);

    // Split round 1 (prefix dimension, bit 0):
    // p(t) = Σ_{x_lo_rest, x_hi} eq_lo(x_lo_rest, t) * eq_hi(x_hi) * H * (H-1)
    // At t=0: sum over even x_lo indices; at t=1: sum over odd x_lo indices
    var split_p0 = F.zero();
    var split_p1 = F.zero();
    const half_lo = prefix_len / 2;
    for (0..half_lo) |j_lo| {
        for (0..suffix_len) |j_hi| {
            const x0 = 2 * j_lo + (j_hi << prefix_n_vars);
            const x1 = 2 * j_lo + 1 + (j_hi << prefix_n_vars);
            const eq_term = eq_lo[2 * j_lo].mul(eq_hi[j_hi]);
            const eq_term1 = eq_lo[2 * j_lo + 1].mul(eq_hi[j_hi]);
            split_p0 = split_p0.add(eq_term.mul(H[x0]).mul(H[x0].sub(F.one())));
            split_p1 = split_p1.add(eq_term1.mul(H[x1]).mul(H[x1].sub(F.one())));
        }
    }

    try testing.expect(flat_p0.eql(split_p0));
    try testing.expect(flat_p1.eql(split_p1));
}

test "IncClaimReduction Phase 1→2 transition: folded suffix matches flat" {
    // Verify that the Phase 1→2 transition math produces the same result as flat computation.
    // All eq tables use LE convention (matching the actual prover which reverses BE→LE first).
    const allocator = testing.allocator;
    const F = BN254Scalar;

    const n_vars = 4;
    const prefix_n_vars = 2;
    const suffix_n_vars = 2;
    const T: usize = 1 << n_vars;
    const prefix_len: usize = 1 << prefix_n_vars;
    const suffix_len: usize = 1 << suffix_n_vars;

    const gamma = F.fromU64(13);
    const challenges = [2]F{ F.fromU64(7), F.fromU64(11) }; // prefix sumcheck challenges

    // 4 opening points in LE order (simulates the prover's reversed BE→LE points).
    // In the prover: r_cycle_rev[i] = r_cycle_be[n_vars - 1 - i].
    // Here we just define them directly in LE.
    var points_le: [4][4]F = undefined;
    points_le[0] = .{ F.fromU64(23), F.fromU64(5), F.fromU64(17), F.fromU64(3) };
    points_le[1] = .{ F.fromU64(19), F.fromU64(2), F.fromU64(11), F.fromU64(7) };
    points_le[2] = .{ F.fromU64(37), F.fromU64(31), F.fromU64(29), F.fromU64(13) };
    points_le[3] = .{ F.fromU64(53), F.fromU64(47), F.fromU64(43), F.fromU64(41) };

    // Build full eq tables for each point (LE input to computeEqTable)
    var eq_full: [4][]F = undefined;
    for (0..4) |i| {
        eq_full[i] = try computeEqTable(F, allocator, &points_le[i], n_vars);
    }
    defer for (0..4) |i| allocator.free(eq_full[i]);

    // Flat approach: eq_ram[x] = eq_0[x] + gamma*eq_1[x], eq_rd[x] = eq_2[x] + gamma*eq_3[x]
    // Then bind prefix variables with challenges to get suffix-sized arrays.
    var flat_eq_ram = try allocator.alloc(F, T);
    defer allocator.free(flat_eq_ram);
    var flat_eq_rd = try allocator.alloc(F, T);
    defer allocator.free(flat_eq_rd);
    for (0..T) |x| {
        flat_eq_ram[x] = eq_full[0][x].add(gamma.mul(eq_full[1][x]));
        flat_eq_rd[x] = eq_full[2][x].add(gamma.mul(eq_full[3][x]));
    }

    // Bind prefix_n_vars rounds (round 0 binds bit 0, round 1 binds bit 1)
    var flat_len: usize = T;
    for (challenges) |ch| {
        const half = flat_len / 2;
        for (0..half) |j| {
            flat_eq_ram[j] = flat_eq_ram[2 * j].add(ch.mul(flat_eq_ram[2 * j + 1].sub(flat_eq_ram[2 * j])));
            flat_eq_rd[j] = flat_eq_rd[2 * j].add(ch.mul(flat_eq_rd[2 * j + 1].sub(flat_eq_rd[2 * j])));
        }
        flat_len = half;
    }

    // Split approach: eq_lo from first prefix_n_vars LE vars, eq_hi from the rest.
    // This mirrors the prover's init which does:
    //   P[i] = computeEqTable(rev_lo, prefix_n_vars) where rev_lo[k] = points_be[n-1-k]
    //   eq_hi[i] = computeEqTable(rev_hi, suffix_n_vars) where rev_hi[k] = points_be[suffix-1-k]
    // In LE terms: lo = points_le[0..prefix_n_vars], hi = points_le[prefix_n_vars..n_vars]
    var eq_hi: [4][]F = undefined;
    for (0..4) |i| {
        var r_hi: [2]F = undefined;
        for (0..suffix_n_vars) |k| r_hi[k] = points_le[i][prefix_n_vars + k];
        eq_hi[i] = try computeEqTable(F, allocator, &r_hi, suffix_n_vars);
    }
    defer for (0..4) |i| allocator.free(eq_hi[i]);

    // Prefix scalars: eq(challenges, point_lo_i) where point_lo = points_le[0..prefix_n_vars]
    var eq_prefix_scalars: [4]F = undefined;
    for (0..4) |i| {
        var result = F.one();
        for (0..prefix_n_vars) |k| {
            const a = challenges[k];
            const b = points_le[i][k];
            const prod = a.mul(b);
            result = result.mul(prod.add(prod).add(F.one()).sub(a.add(b)));
        }
        eq_prefix_scalars[i] = result;
    }

    // Build split eq arrays and compare
    for (0..suffix_len) |x_hi| {
        const split_ram = eq_prefix_scalars[0].mul(eq_hi[0][x_hi]).add(gamma.mul(eq_prefix_scalars[1].mul(eq_hi[1][x_hi])));
        const split_rd = eq_prefix_scalars[2].mul(eq_hi[2][x_hi]).add(gamma.mul(eq_prefix_scalars[3].mul(eq_hi[3][x_hi])));
        try testing.expect(flat_eq_ram[x_hi].eql(split_ram));
        try testing.expect(flat_eq_rd[x_hi].eql(split_rd));
    }

    // Also verify the inc folding: Σ_{x_lo} eq_prefix[x_lo] * f(x_lo, x_hi) matches
    // flat bind of f(x) over prefix variables.
    const eq_prefix_table = try computeEqTable(F, allocator, &challenges, prefix_n_vars);
    defer allocator.free(eq_prefix_table);

    // f(x) = x + 1 (synthetic)
    var f_vals = try allocator.alloc(F, T);
    defer allocator.free(f_vals);
    for (0..T) |x| f_vals[x] = F.fromU64(@intCast(x + 1));

    // Flat bind of f over prefix
    var f_flat = try allocator.alloc(F, T);
    defer allocator.free(f_flat);
    @memcpy(f_flat, f_vals);
    var f_len: usize = T;
    for (challenges) |ch| {
        const half = f_len / 2;
        for (0..half) |j| {
            f_flat[j] = f_flat[2 * j].add(ch.mul(f_flat[2 * j + 1].sub(f_flat[2 * j])));
        }
        f_len = half;
    }

    // Split fold: Σ_{x_lo} eq_prefix[x_lo] * f(x_lo + x_hi << prefix_n_vars)
    for (0..suffix_len) |x_hi| {
        var acc = F.zero();
        for (0..prefix_len) |x_lo| {
            const x = x_lo + (x_hi << prefix_n_vars);
            acc = acc.add(eq_prefix_table[x_lo].mul(f_vals[x]));
        }
        try testing.expect(f_flat[x_hi].eql(acc));
    }
}

test "BytecodeReadRaf split-eq F_s: inner*outer matches flat eq pushforward" {
    // Verify F_s[pc] = Σ_c eq(r_cycle, c) * δ(PC(c)=pc) is the same whether computed
    // via a flat T-sized eq table or via the split-eq double loop with touched-PC tracking.
    const allocator = testing.allocator;
    const F = BN254Scalar;

    const n_vars = 4;
    const T: usize = 1 << n_vars;
    const lo_bits = n_vars / 2;
    const hi_bits = n_vars - lo_bits;
    const in_len: usize = 1 << lo_bits;
    const out_len: usize = 1 << hi_bits;
    const bytecode_K: usize = 8;

    // PC map: cycle c → pc_idx (some synthetic mapping)
    var pc_map_arr: [T]usize = undefined;
    for (0..T) |c| {
        pc_map_arr[c] = (c * 3 + 1) % bytecode_K;
    }

    // r_cycle in LE order (r[0]→LSB, as used by computeEqTable)
    var r_le = [4]F{ F.fromU64(5), F.fromU64(17), F.fromU64(31), F.fromU64(43) };

    // Method 1: Flat computation with full T-sized eq table
    const eq_flat = try computeEqTable(F, allocator, &r_le, n_vars);
    defer allocator.free(eq_flat);

    var F_s_flat: [bytecode_K]F = .{F.zero()} ** bytecode_K;
    for (0..T) |c| {
        F_s_flat[pc_map_arr[c]] = F_s_flat[pc_map_arr[c]].add(eq_flat[c]);
    }

    // Method 2: Split-eq double loop (same algorithm as BytecodeReadRafProver.init)
    // Split LE points into lo and hi halves

    var r_lo_arr = [2]F{ r_le[0], r_le[1] };
    const E_lo = try computeEqTable(F, allocator, &r_lo_arr, lo_bits);
    defer allocator.free(E_lo);

    var r_hi_arr = [2]F{ r_le[2], r_le[3] };
    const E_hi = try computeEqTable(F, allocator, &r_hi_arr, hi_bits);
    defer allocator.free(E_hi);

    var F_s_split: [bytecode_K]F = .{F.zero()} ** bytecode_K;
    var inner_buf: [bytecode_K]F = .{F.zero()} ** bytecode_K;
    var touched_buf: [bytecode_K]usize = undefined;
    var touched_set: [bytecode_K]bool = .{false} ** bytecode_K;

    for (0..out_len) |c_hi| {
        var touched_count: usize = 0;

        for (0..in_len) |c_lo| {
            const c = c_lo + (c_hi << @intCast(lo_bits));
            const pc = pc_map_arr[c];
            if (!touched_set[pc]) {
                touched_set[pc] = true;
                touched_buf[touched_count] = pc;
                touched_count += 1;
            }
            inner_buf[pc] = inner_buf[pc].add(E_lo[c_lo]);
        }

        const e_hi_val = E_hi[c_hi];
        for (0..touched_count) |ti| {
            const pc = touched_buf[ti];
            F_s_split[pc] = F_s_split[pc].add(e_hi_val.mul(inner_buf[pc]));
            inner_buf[pc] = F.zero();
            touched_set[pc] = false;
        }
    }

    for (0..bytecode_K) |k| {
        try testing.expect(F_s_flat[k].eql(F_s_split[k]));
    }
}

test "IncClaimReduction full multi-round: split P/Q matches flat across phase transition" {
    // Full multi-round sumcheck simulation for IncClaimReduction:
    // Phase 1 (prefix rounds on P/Q) → transition → Phase 2 (suffix rounds on dense arrays).
    // The sumcheck is degree 2 (product of two linear factors: eq × inc).
    // We keep the factors separate in the flat reference to properly evaluate the degree-2
    // round polynomial at 3 points [s(0), s(1), s(2)].
    const allocator = testing.allocator;
    const F = BN254Scalar;

    const n_vars = 6;
    const prefix_n_vars = 3;
    const suffix_n_vars = 3;
    const T: usize = 1 << n_vars;
    const prefix_len: usize = 1 << prefix_n_vars;
    const suffix_len: usize = 1 << suffix_n_vars;

    const gamma = F.fromU64(13);
    const gamma_sqr = gamma.mul(gamma);

    // 4 opening points in LE order
    const points_le = [4][6]F{
        .{ F.fromU64(3), F.fromU64(7), F.fromU64(11), F.fromU64(17), F.fromU64(23), F.fromU64(29) },
        .{ F.fromU64(5), F.fromU64(13), F.fromU64(19), F.fromU64(31), F.fromU64(37), F.fromU64(41) },
        .{ F.fromU64(2), F.fromU64(43), F.fromU64(47), F.fromU64(53), F.fromU64(59), F.fromU64(61) },
        .{ F.fromU64(67), F.fromU64(71), F.fromU64(73), F.fromU64(79), F.fromU64(83), F.fromU64(89) },
    };

    // Synthetic inc values
    var ram_inc_vals: [T]F = undefined;
    var rd_inc_vals: [T]F = undefined;
    for (0..T) |x| {
        ram_inc_vals[x] = F.fromU64(@intCast(x + 1));
        rd_inc_vals[x] = F.fromU64(@intCast(2 * x + 3));
    }

    // Build flat eq tables
    var eq_full: [4][]F = undefined;
    for (0..4) |i| {
        eq_full[i] = try computeEqTable(F, allocator, @constCast(&points_le[i]), n_vars);
    }
    defer for (0..4) |i| allocator.free(eq_full[i]);

    // Flat: keep eq and inc separate (4 eq arrays, 2 inc arrays) for degree-2 round poly
    var flat_ram_inc = try allocator.alloc(F, T);
    defer allocator.free(flat_ram_inc);
    var flat_rd_inc = try allocator.alloc(F, T);
    defer allocator.free(flat_rd_inc);
    @memcpy(flat_ram_inc, &ram_inc_vals);
    @memcpy(flat_rd_inc, &rd_inc_vals);

    // --- Split approach: build P, Q arrays ---
    var P: [4][]F = undefined;
    var eq_hi: [4][]F = undefined;
    for (0..4) |i| {
        var r_lo: [3]F = undefined;
        for (0..prefix_n_vars) |k| r_lo[k] = points_le[i][k];
        P[i] = try computeEqTable(F, allocator, &r_lo, prefix_n_vars);

        var r_hi: [3]F = undefined;
        for (0..suffix_n_vars) |k| r_hi[k] = points_le[i][prefix_n_vars + k];
        eq_hi[i] = try computeEqTable(F, allocator, &r_hi, suffix_n_vars);
    }
    defer for (0..4) |i| {
        allocator.free(P[i]);
        allocator.free(eq_hi[i]);
    };

    var Q: [4][]F = undefined;
    for (0..4) |i| {
        Q[i] = try allocator.alloc(F, prefix_len);
        for (0..prefix_len) |x_lo| {
            var acc = F.zero();
            for (0..suffix_len) |x_hi| {
                const x = x_lo + (x_hi << prefix_n_vars);
                const inc_val = if (i < 2) ram_inc_vals[x] else rd_inc_vals[x];
                acc = acc.add(eq_hi[i][x_hi].mul(inc_val));
            }
            Q[i][x_lo] = acc;
        }
    }
    defer for (0..4) |i| allocator.free(Q[i]);

    const gamma_cub = gamma_sqr.mul(gamma);
    const weights = [4]F{ F.one(), gamma, gamma_sqr, gamma_cub };

    var flat_len: usize = T;
    var p_len: usize = prefix_len;
    var challenges: [6]F = undefined;
    var in_phase2 = false;

    var p2_ram_inc: ?[]F = null;
    defer if (p2_ram_inc) |a| allocator.free(a);
    var p2_rd_inc: ?[]F = null;
    defer if (p2_rd_inc) |a| allocator.free(a);
    var p2_eq_ram: ?[]F = null;
    defer if (p2_eq_ram) |a| allocator.free(a);
    var p2_eq_rd: ?[]F = null;
    defer if (p2_eq_rd) |a| allocator.free(a);
    var p2_len: usize = 0;

    for (0..n_vars) |round| {
        const r = F.fromU64(@intCast(round * 7 + 3));
        challenges[round] = r;

        const flat_half = flat_len / 2;

        // --- Flat round poly (degree 2): 3 evaluation points ---
        // s(t) = Σ_j [ (eq_0(t) + γ·eq_1(t))·ram_inc(t) + γ²·(eq_2(t) + γ·eq_3(t))·rd_inc(t) ]
        var flat_evals: [3]F = .{ F.zero(), F.zero(), F.zero() };
        for (0..flat_half) |j| {
            // Values at t=0, t=1, t=2
            var eq_ram_at: [3]F = undefined;
            var eq_rd_at: [3]F = undefined;
            var ram_at: [3]F = undefined;
            var rd_at: [3]F = undefined;
            for (0..3) |t| {
                const tf = F.fromU64(@intCast(t));
                inline for (0..4) |k| {
                    const v0 = eq_full[k][2 * j];
                    const v1 = eq_full[k][2 * j + 1];
                    const interp = v0.add(tf.mul(v1.sub(v0)));
                    if (k == 0) eq_ram_at[t] = interp;
                    if (k == 1) eq_ram_at[t] = eq_ram_at[t].add(gamma.mul(interp));
                    if (k == 2) eq_rd_at[t] = interp;
                    if (k == 3) eq_rd_at[t] = eq_rd_at[t].add(gamma.mul(interp));
                }
                const r0 = flat_ram_inc[2 * j];
                const r1 = flat_ram_inc[2 * j + 1];
                ram_at[t] = r0.add(tf.mul(r1.sub(r0)));
                const d0 = flat_rd_inc[2 * j];
                const d1 = flat_rd_inc[2 * j + 1];
                rd_at[t] = d0.add(tf.mul(d1.sub(d0)));
            }
            for (0..3) |t| {
                flat_evals[t] = flat_evals[t].add(
                    ram_at[t].mul(eq_ram_at[t]).add(gamma_sqr.mul(rd_at[t].mul(eq_rd_at[t]))),
                );
            }
        }

        // --- Split round poly ---
        var split_evals: [3]F = .{ F.zero(), F.zero(), F.zero() };

        if (!in_phase2) {
            const half = p_len / 2;
            for (0..half) |j| {
                for (0..3) |t| {
                    const tf = F.fromU64(@intCast(t));
                    var term = F.zero();
                    for (0..4) |k| {
                        const p0 = P[k][2 * j];
                        const p1 = P[k][2 * j + 1];
                        const q0 = Q[k][2 * j];
                        const q1 = Q[k][2 * j + 1];
                        const p_t = p0.add(tf.mul(p1.sub(p0)));
                        const q_t = q0.add(tf.mul(q1.sub(q0)));
                        term = term.add(weights[k].mul(p_t.mul(q_t)));
                    }
                    split_evals[t] = split_evals[t].add(term);
                }
            }
        } else {
            const half = p2_len / 2;
            for (0..half) |j| {
                for (0..3) |t| {
                    const tf = F.fromU64(@intCast(t));
                    const ram_t = p2_ram_inc.?[2 * j].add(tf.mul(p2_ram_inc.?[2 * j + 1].sub(p2_ram_inc.?[2 * j])));
                    const eq_r_t = p2_eq_ram.?[2 * j].add(tf.mul(p2_eq_ram.?[2 * j + 1].sub(p2_eq_ram.?[2 * j])));
                    const rd_t = p2_rd_inc.?[2 * j].add(tf.mul(p2_rd_inc.?[2 * j + 1].sub(p2_rd_inc.?[2 * j])));
                    const eq_d_t = p2_eq_rd.?[2 * j].add(tf.mul(p2_eq_rd.?[2 * j + 1].sub(p2_eq_rd.?[2 * j])));
                    split_evals[t] = split_evals[t].add(
                        ram_t.mul(eq_r_t).add(gamma_sqr.mul(rd_t.mul(eq_d_t))),
                    );
                }
            }
        }

        for (0..3) |t| {
            try testing.expect(flat_evals[t].eql(split_evals[t]));
        }

        // --- Bind all arrays ---
        // Flat: bind 4 eq arrays + 2 inc arrays
        for (0..flat_half) |j| {
            for (0..4) |k| {
                eq_full[k][j] = eq_full[k][2 * j].add(r.mul(eq_full[k][2 * j + 1].sub(eq_full[k][2 * j])));
            }
            flat_ram_inc[j] = flat_ram_inc[2 * j].add(r.mul(flat_ram_inc[2 * j + 1].sub(flat_ram_inc[2 * j])));
            flat_rd_inc[j] = flat_rd_inc[2 * j].add(r.mul(flat_rd_inc[2 * j + 1].sub(flat_rd_inc[2 * j])));
        }
        flat_len = flat_half;

        if (!in_phase2) {
            if (p_len == 2) {
                // Transition to Phase 2
                const eq_prefix = try computeEqTable(F, allocator, challenges[0 .. round + 1], prefix_n_vars);
                defer allocator.free(eq_prefix);

                var eq_prefix_scalars: [4]F = undefined;
                for (0..4) |i| {
                    var result = F.one();
                    for (0..prefix_n_vars) |k| {
                        const a = challenges[k];
                        const b = points_le[i][k];
                        const prod = a.mul(b);
                        result = result.mul(prod.add(prod).add(F.one()).sub(a.add(b)));
                    }
                    eq_prefix_scalars[i] = result;
                }

                p2_eq_ram = try allocator.alloc(F, suffix_len);
                p2_eq_rd = try allocator.alloc(F, suffix_len);
                for (0..suffix_len) |x_hi| {
                    p2_eq_ram.?[x_hi] = eq_prefix_scalars[0].mul(eq_hi[0][x_hi]).add(
                        gamma.mul(eq_prefix_scalars[1].mul(eq_hi[1][x_hi])),
                    );
                    p2_eq_rd.?[x_hi] = eq_prefix_scalars[2].mul(eq_hi[2][x_hi]).add(
                        gamma.mul(eq_prefix_scalars[3].mul(eq_hi[3][x_hi])),
                    );
                }

                p2_ram_inc = try allocator.alloc(F, suffix_len);
                p2_rd_inc = try allocator.alloc(F, suffix_len);
                for (0..suffix_len) |x_hi| {
                    var acc_ram = F.zero();
                    var acc_rd = F.zero();
                    for (0..prefix_len) |x_lo| {
                        const x = x_lo + (x_hi << prefix_n_vars);
                        acc_ram = acc_ram.add(eq_prefix[x_lo].mul(ram_inc_vals[x]));
                        acc_rd = acc_rd.add(eq_prefix[x_lo].mul(rd_inc_vals[x]));
                    }
                    p2_ram_inc.?[x_hi] = acc_ram;
                    p2_rd_inc.?[x_hi] = acc_rd;
                }
                p2_len = suffix_len;
                in_phase2 = true;
            } else {
                const half = p_len / 2;
                for (0..4) |k| {
                    for (0..half) |j| {
                        P[k][j] = P[k][2 * j].add(r.mul(P[k][2 * j + 1].sub(P[k][2 * j])));
                        Q[k][j] = Q[k][2 * j].add(r.mul(Q[k][2 * j + 1].sub(Q[k][2 * j])));
                    }
                }
                p_len = half;
            }
        } else {
            const half = p2_len / 2;
            for (0..half) |j| {
                p2_ram_inc.?[j] = p2_ram_inc.?[2 * j].add(r.mul(p2_ram_inc.?[2 * j + 1].sub(p2_ram_inc.?[2 * j])));
                p2_rd_inc.?[j] = p2_rd_inc.?[2 * j].add(r.mul(p2_rd_inc.?[2 * j + 1].sub(p2_rd_inc.?[2 * j])));
                p2_eq_ram.?[j] = p2_eq_ram.?[2 * j].add(r.mul(p2_eq_ram.?[2 * j + 1].sub(p2_eq_ram.?[2 * j])));
                p2_eq_rd.?[j] = p2_eq_rd.?[2 * j].add(r.mul(p2_eq_rd.?[2 * j + 1].sub(p2_eq_rd.?[2 * j])));
            }
            p2_len = half;
        }
    }

    // Final scalar: split must match flat
    const flat_final = flat_ram_inc[0].mul(
        eq_full[0][0].add(gamma.mul(eq_full[1][0])),
    ).add(gamma_sqr.mul(flat_rd_inc[0].mul(
        eq_full[2][0].add(gamma.mul(eq_full[3][0])),
    )));
    const split_final = p2_ram_inc.?[0].mul(p2_eq_ram.?[0]).add(
        gamma_sqr.mul(p2_rd_inc.?[0].mul(p2_eq_rd.?[0])),
    );
    try testing.expect(flat_final.eql(split_final));
}

test "HammingBooleanity full multi-round: split-eq matches flat across phase transition" {
    // Full multi-round sumcheck simulation for HammingBooleanity:
    // Phase 1 (prefix rounds with factored eq_lo·eq_hi) → transition → Phase 2 (merged eq).
    // Verifies every round polynomial matches the flat (unsplit) computation.
    const allocator = testing.allocator;
    const F = BN254Scalar;

    const n_vars = 6;
    const prefix_n_vars = 3;
    const suffix_n_vars = 3;
    const T: usize = 1 << n_vars;
    const prefix_len: usize = 1 << prefix_n_vars;
    const suffix_len: usize = 1 << suffix_n_vars;

    // r_cycle in LE order
    var r_le = [6]F{
        F.fromU64(5),  F.fromU64(13), F.fromU64(3),
        F.fromU64(19), F.fromU64(7),  F.fromU64(11),
    };

    // H values: simulate Hamming weight (binary values for booleanity test)
    var H_flat: [T]F = undefined;
    var H_split: [T]F = undefined;
    for (0..T) |x| {
        // Mix of 0 and 1 with some non-boolean values to make test interesting
        const v: u64 = if (x % 5 == 0) 0 else if (x % 3 == 0) 1 else @intCast(x % 4);
        H_flat[x] = F.fromU64(v);
        H_split[x] = F.fromU64(v);
    }

    // Flat eq table
    var eq_flat = try computeEqTable(F, allocator, &r_le, n_vars);
    defer allocator.free(eq_flat);

    // Split eq tables
    var r_lo: [3]F = undefined;
    for (0..prefix_n_vars) |k| r_lo[k] = r_le[k];
    var eq_lo = try computeEqTable(F, allocator, &r_lo, prefix_n_vars);
    defer allocator.free(eq_lo);

    var r_hi: [3]F = undefined;
    for (0..suffix_n_vars) |k| r_hi[k] = r_le[prefix_n_vars + k];
    const eq_hi = try computeEqTable(F, allocator, &r_hi, suffix_n_vars);
    defer allocator.free(eq_hi);

    var flat_len: usize = T;
    var split_h_len: usize = T;
    var lo_len: usize = prefix_len;
    var in_phase2 = false;

    // Phase 2 state
    var eq_merged: ?[]F = null;
    defer if (eq_merged) |a| allocator.free(a);
    var merged_len: usize = 0;

    for (0..n_vars) |round| {
        const r = F.fromU64(@intCast(round * 11 + 2));
        const two = F.fromU64(2);
        const three = F.fromU64(3);

        // --- Flat round poly: [s(0), s(1), s(2), s(3)] ---
        const flat_half = flat_len / 2;
        var flat_evals: [4]F = .{ F.zero(), F.zero(), F.zero(), F.zero() };
        for (0..flat_half) |j| {
            const h0 = H_flat[2 * j];
            const h1 = H_flat[2 * j + 1];
            const h_delta = h1.sub(h0);
            const e0 = eq_flat[2 * j];
            const e1 = eq_flat[2 * j + 1];
            const e_delta = e1.sub(e0);

            flat_evals[0] = flat_evals[0].add(e0.mul(h0.mul(h0).sub(h0)));
            flat_evals[1] = flat_evals[1].add(e1.mul(h1.mul(h1).sub(h1)));

            const h_at_2 = h0.add(two.mul(h_delta));
            const e_at_2 = e0.add(two.mul(e_delta));
            flat_evals[2] = flat_evals[2].add(e_at_2.mul(h_at_2.mul(h_at_2).sub(h_at_2)));

            const h_at_3 = h0.add(three.mul(h_delta));
            const e_at_3 = e0.add(three.mul(e_delta));
            flat_evals[3] = flat_evals[3].add(e_at_3.mul(h_at_3.mul(h_at_3).sub(h_at_3)));
        }

        // --- Split round poly ---
        var split_evals: [4]F = .{ F.zero(), F.zero(), F.zero(), F.zero() };

        if (!in_phase2) {
            // Phase 1: double loop with factored eq = eq_lo(x_lo) * eq_hi(x_hi)
            const half_lo = lo_len / 2;
            for (0..suffix_len) |j_outer| {
                const eq_hi_val = eq_hi[j_outer];
                for (0..half_lo) |j_inner| {
                    const j = j_inner + j_outer * half_lo;
                    const h0 = H_split[2 * j];
                    const h1 = H_split[2 * j + 1];
                    const h_delta = h1.sub(h0);

                    const eq_lo_0 = eq_lo[2 * j_inner];
                    const eq_lo_1 = eq_lo[2 * j_inner + 1];
                    const e0 = eq_lo_0.mul(eq_hi_val);
                    const e1 = eq_lo_1.mul(eq_hi_val);
                    const e_delta = e1.sub(e0);

                    split_evals[0] = split_evals[0].add(e0.mul(h0.mul(h0).sub(h0)));
                    split_evals[1] = split_evals[1].add(e1.mul(h1.mul(h1).sub(h1)));

                    const h_at_2 = h0.add(two.mul(h_delta));
                    const e_at_2 = e0.add(two.mul(e_delta));
                    split_evals[2] = split_evals[2].add(e_at_2.mul(h_at_2.mul(h_at_2).sub(h_at_2)));

                    const h_at_3 = h0.add(three.mul(h_delta));
                    const e_at_3 = e0.add(three.mul(e_delta));
                    split_evals[3] = split_evals[3].add(e_at_3.mul(h_at_3.mul(h_at_3).sub(h_at_3)));
                }
            }
        } else {
            // Phase 2: flat loop with merged eq
            const half = split_h_len / 2;
            for (0..half) |j| {
                const h0 = H_split[2 * j];
                const h1 = H_split[2 * j + 1];
                const h_delta = h1.sub(h0);
                const e0 = eq_merged.?[2 * j];
                const e1 = eq_merged.?[2 * j + 1];
                const e_delta = e1.sub(e0);

                split_evals[0] = split_evals[0].add(e0.mul(h0.mul(h0).sub(h0)));
                split_evals[1] = split_evals[1].add(e1.mul(h1.mul(h1).sub(h1)));

                const h_at_2 = h0.add(two.mul(h_delta));
                const e_at_2 = e0.add(two.mul(e_delta));
                split_evals[2] = split_evals[2].add(e_at_2.mul(h_at_2.mul(h_at_2).sub(h_at_2)));

                const h_at_3 = h0.add(three.mul(h_delta));
                const e_at_3 = e0.add(three.mul(e_delta));
                split_evals[3] = split_evals[3].add(e_at_3.mul(h_at_3.mul(h_at_3).sub(h_at_3)));
            }
        }

        // All 4 evaluation points must match
        for (0..4) |k| {
            try testing.expect(flat_evals[k].eql(split_evals[k]));
        }

        // --- Bind ---
        // Flat: bind eq and H
        for (0..flat_half) |j| {
            eq_flat[j] = eq_flat[2 * j].add(r.mul(eq_flat[2 * j + 1].sub(eq_flat[2 * j])));
            H_flat[j] = H_flat[2 * j].add(r.mul(H_flat[2 * j + 1].sub(H_flat[2 * j])));
        }
        flat_len = flat_half;

        // Split: bind H always, plus eq_lo or merged eq
        const split_half = split_h_len / 2;
        for (0..split_half) |j| {
            H_split[j] = H_split[2 * j].add(r.mul(H_split[2 * j + 1].sub(H_split[2 * j])));
        }
        split_h_len = split_half;

        if (!in_phase2) {
            const half_lo = lo_len / 2;
            for (0..half_lo) |j| {
                eq_lo[j] = eq_lo[2 * j].add(r.mul(eq_lo[2 * j + 1].sub(eq_lo[2 * j])));
            }
            lo_len = half_lo;

            // Transition when eq_lo reaches length 1
            if (half_lo == 1) {
                const eq_lo_scalar = eq_lo[0];
                // Merge: eq_merged[j_hi] = eq_lo_scalar * eq_hi[j_hi]
                eq_merged = try allocator.alloc(F, suffix_len);
                for (0..suffix_len) |j| {
                    eq_merged.?[j] = eq_lo_scalar.mul(eq_hi[j]);
                }
                merged_len = suffix_len;
                in_phase2 = true;
            }
        } else {
            // Phase 2: bind merged eq
            const half = merged_len / 2;
            for (0..half) |j| {
                eq_merged.?[j] = eq_merged.?[2 * j].add(r.mul(eq_merged.?[2 * j + 1].sub(eq_merged.?[2 * j])));
            }
            merged_len = half;
        }
    }

    // Final scalars must match
    try testing.expect(H_flat[0].eql(H_split[0]));
    try testing.expect(eq_flat[0].eql(eq_merged.?[0]));
}
