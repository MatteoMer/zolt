const std = @import("std");

/// SHA-256 initial hash values (first 32 bits of the fractional parts
/// of the square roots of the first 8 primes).
pub const BLOCK: [8]u64 = .{
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
};

/// SHA-256 round constants (first 32 bits of the fractional parts
/// of the cube roots of the first 64 primes).
pub const K: [64]u64 = .{
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
};

/// Operand: either an immediate constant or a register reference.
/// Used for constant folding: when both operands of a binary op are
/// Imm, the result is computed at build time with no instruction emitted.
pub const Value = union(enum) {
    imm: u64,
    reg: u8,
};

/// Instruction kind for the emulator to dispatch execution.
/// NOTE: These are the DECOMPOSED virtual forms, matching Jolt's inline_sequence().
/// SLLI -> VirtualMULI, SRLI -> VirtualSRLI, SRLIW -> 3-step sequence.
pub const InstrKind = enum {
    ADD,
    ADDI,
    XOR,
    XORI,
    AND,
    ANDI,
    OR,
    ANDN,
    LD,
    SD,
    VirtualMULI, // Decomposed from SLLI: rd = rs1 * (1 << imm)
    VirtualSRLI, // Decomposed from SRLI: rd = rs1 & bitmask (bitmask in imm)
    VirtualSignExtendWord, // Sign-extend lower 32 bits
    VirtualZeroExtendWord,
    VirtualROTRIW,
};

/// Describes a single virtual instruction in the inline sequence.
pub const InlineInstr = struct {
    /// Destination register (0 for stores).
    rd: u8,
    /// Source register 1.
    rs1: u8,
    /// Source register 2 (R-type) or 0.
    rs2: u8,
    /// Immediate value (I-type, or bitmask for ROTRIW, or offset for LD/SD).
    imm: u64,
    /// Instruction kind for the emulator to dispatch.
    kind: InstrKind,
};

/// The result list type returned by buildSha256Sequence.
pub const InlineInstrList = std.ArrayList(InlineInstr);

// ---------------------------------------------------------------------------
// Virtual register assignments (deterministic, hardcoded)
// Mirrors Jolt's allocate_for_inline() which starts at register 48.
// ---------------------------------------------------------------------------

/// Working state A..H -> registers 48..55
const STATE_BASE: u8 = 48;
/// Message schedule W[0..15] -> registers 56..71
const MESSAGE_BASE: u8 = 56;
/// Initial values (non-initial mode) -> registers 72..79
const IV_BASE: u8 = 72;
/// Per-round temporaries -> registers 80..83
const TEMP_T1: u8 = 80;
const TEMP_T2: u8 = 81;
const TEMP_SS: u8 = 82;
const TEMP_SS2: u8 = 83;

/// Zig equivalent of Rust's i32::rem_euclid(8).
/// Always returns a value in 0..7.
inline fn rem_euclid_8(v: i32) u8 {
    return @intCast(@mod(v, 8));
}

/// Zig equivalent of Rust's i32::rem_euclid(16).
/// Always returns a value in 0..15.
inline fn rem_euclid_16(v: i32) u8 {
    return @intCast(@mod(v, 16));
}

/// Builds the SHA-256 inline instruction sequence.
///
/// `allocator`: memory allocator for the returned list.
/// `rs1`: architectural register holding the pointer to the 8 x u32 state.
/// `rs2`: architectural register holding the pointer to the 16 x u32 input.
/// `initial`: when true, use BLOCK constants as the initial hash values;
///            when false, load IV from memory at rs1.
///
/// The caller owns the returned list and must call `list.deinit(allocator)`.
pub fn buildSha256Sequence(
    allocator: std.mem.Allocator,
    rs1: u8,
    rs2: u8,
    initial: bool,
) !InlineInstrList {
    var b = Builder{
        .seq = .empty,
        .round = 0,
        .rs1 = rs1,
        .rs2 = rs2,
        .initial = initial,
    };
    errdefer b.seq.deinit(allocator);

    // Pre-allocate: ~2200 instructions for initial, ~2300 for non-initial
    try b.seq.ensureTotalCapacity(allocator, 2400);

    // 1. If !initial: load IV from memory at rs1 (4 x load_paired_u32_dirty)
    if (!initial) {
        for (0..4) |i| {
            b.loadPairedU32Dirty(
                rs1,
                @as(u64, @intCast(i * 8)),
                IV_BASE + @as(u8, @intCast(i * 2)),
                IV_BASE + @as(u8, @intCast(i * 2 + 1)),
            );
        }
    }

    // 2. Load input from memory at rs2 (8 x load_paired_u32_dirty)
    for (0..8) |i| {
        b.loadPairedU32Dirty(
            rs2,
            @as(u64, @intCast(i * 8)),
            MESSAGE_BASE + @as(u8, @intCast(i * 2)),
            MESSAGE_BASE + @as(u8, @intCast(i * 2 + 1)),
        );
    }

    // 3. 64 rounds of SHA-256 compression
    for (0..64) |_| {
        b.doRound();
    }

    // 4. Final IV addition
    b.finalAddIv();

    // 5. Store output to rs1 (4 x store_paired_u32)
    const pairs = [4][2]u8{
        .{ 'A' - 'A', 'B' - 'A' },
        .{ 'C' - 'A', 'D' - 'A' },
        .{ 'E' - 'A', 'F' - 'A' },
        .{ 'G' - 'A', 'H' - 'A' },
    };
    for (pairs, 0..) |pair, i| {
        b.storePairedU32(
            rs1,
            @as(u64, @intCast(i * 8)),
            b.vr(pair[0]),
            b.vr(pair[1]),
        );
    }

    // 6. Zero all allocated inline registers.
    // Mirrors Jolt's finalize_inline() which emits ADDI rd, x0, 0
    // for every register tracked by pending_clearing_inline.
    if (initial) {
        // state (48..55) + message (56..71) = 24 regs, temps (80..83) = 4 regs
        for (48..72) |r| {
            b.emit(.{ .rd = @intCast(r), .rs1 = 0, .rs2 = 0, .imm = 0, .kind = .ADDI });
        }
        for (80..84) |r| {
            b.emit(.{ .rd = @intCast(r), .rs1 = 0, .rs2 = 0, .imm = 0, .kind = .ADDI });
        }
    } else {
        // state (48..55) + message (56..71) + iv (72..79) + temps (80..83) = 36 regs
        for (48..84) |r| {
            b.emit(.{ .rd = @intCast(r), .rs1 = 0, .rs2 = 0, .imm = 0, .kind = .ADDI });
        }
    }

    return b.seq;
}

// ---------------------------------------------------------------------------
// Internal builder
// ---------------------------------------------------------------------------

const Builder = struct {
    seq: InlineInstrList,
    round: i32,
    rs1: u8,
    rs2: u8,
    initial: bool,

    // -----------------------------------------------------------------------
    // Emit helpers
    // -----------------------------------------------------------------------

    fn emit(self: *Builder, instr: InlineInstr) void {
        self.seq.appendAssumeCapacity(instr);
    }

    // -----------------------------------------------------------------------
    // Binary operation with constant folding (mirrors InstrAssembler::bin)
    // -----------------------------------------------------------------------

    /// Generic binary op. Returns the result as a Value.
    /// When both operands are Imm, constant-folds with `fold` and emits nothing.
    fn bin(
        self: *Builder,
        a: Value,
        b: Value,
        rd: u8,
        r_kind: InstrKind,
        i_kind: InstrKind,
        fold: *const fn (u64, u64) u64,
    ) Value {
        switch (a) {
            .reg => |r1| switch (b) {
                .reg => |r2| {
                    self.emit(.{ .rd = rd, .rs1 = r1, .rs2 = r2, .imm = 0, .kind = r_kind });
                    return .{ .reg = rd };
                },
                .imm => |imm_val| {
                    self.emit(.{ .rd = rd, .rs1 = r1, .rs2 = 0, .imm = imm_val, .kind = i_kind });
                    return .{ .reg = rd };
                },
            },
            .imm => |a_imm| switch (b) {
                .reg => |r2| {
                    // Swap: emit I-type with the register as rs1
                    self.emit(.{ .rd = rd, .rs1 = r2, .rs2 = 0, .imm = a_imm, .kind = i_kind });
                    return .{ .reg = rd };
                },
                .imm => |b_imm| {
                    return .{ .imm = fold(a_imm, b_imm) };
                },
            },
        }
    }

    fn add(self: *Builder, a: Value, b: Value, rd: u8) Value {
        return self.bin(a, b, rd, .ADD, .ADDI, &foldAdd);
    }

    fn xorOp(self: *Builder, a: Value, b: Value, rd: u8) Value {
        return self.bin(a, b, rd, .XOR, .XORI, &foldXor);
    }

    fn andOp(self: *Builder, a: Value, b: Value, rd: u8) Value {
        return self.bin(a, b, rd, .AND, .ANDI, &foldAnd);
    }

    // -----------------------------------------------------------------------
    // Fold functions
    // -----------------------------------------------------------------------

    fn foldAdd(x: u64, y: u64) u64 {
        return x +% y;
    }
    fn foldXor(x: u64, y: u64) u64 {
        return x ^ y;
    }
    fn foldAnd(x: u64, y: u64) u64 {
        return x & y;
    }

    // -----------------------------------------------------------------------
    // Shift / rotate helpers
    // -----------------------------------------------------------------------

    /// Logical right shift immediate on a 32-bit word (SRLIW decomposition in 64-bit mode).
    /// In Jolt, SRLIW decomposes to 3 steps:
    ///   1. VirtualMULI(v_rs1, rs1, shamt=32)  -- SLLI by 32 to position lower 32 bits
    ///   2. VirtualSRLI(rd, v_rs1, bitmask)     -- shift right by (shamt+32) with bitmask
    ///   3. VirtualSignExtendWord(rd, rd, 0)     -- sign-extend lower 32 bits
    /// Uses register 40 as intermediate (Jolt's allocator.allocate() for instruction VRs).
    fn srli(self: *Builder, rs1_val: Value, shamt: u5, rd: u8) Value {
        if (shamt == 0) {
            return self.xorOp(rs1_val, .{ .imm = 0 }, rd);
        }
        switch (rs1_val) {
            .reg => |r| {
                const INSTR_VR: u8 = 40; // Jolt's allocator.allocate() first register
                // Step 1: VirtualMULI(v_rs1, rs1, shamt=32) -- equivalent to SLLI rs1, 32
                self.emit(.{ .rd = INSTR_VR, .rs1 = r, .rs2 = 0, .imm = 32, .kind = .VirtualMULI });
                // Step 2: VirtualSRLI(rd, v_rs1, bitmask) -- shift right by (shamt+32)
                const total_shift: u7 = @as(u7, shamt) + 32;
                const ones: u128 = (@as(u128, 1) << @intCast(64 - @as(u8, total_shift))) - 1;
                const bitmask: u64 = @truncate(ones << total_shift);
                self.emit(.{ .rd = rd, .rs1 = INSTR_VR, .rs2 = 0, .imm = bitmask, .kind = .VirtualSRLI });
                // Step 3: VirtualSignExtendWord(rd, rd, 0)
                self.emit(.{ .rd = rd, .rs1 = rd, .rs2 = 0, .imm = 0, .kind = .VirtualSignExtendWord });
                return .{ .reg = rd };
            },
            .imm => |val| {
                const val32: u32 = @truncate(val);
                return .{ .imm = @as(u64, val32 >> shamt) };
            },
        }
    }

    /// 32-bit rotate right by `shamt`.
    /// Emits VirtualROTRIW with the bitmask encoding.
    /// Mirrors InstrAssembler::rotri32 for Xlen::Bit64.
    fn rotri32(self: *Builder, rs1_val: Value, shamt: u5, rd: u8) Value {
        if (shamt == 0) {
            return self.xorOp(rs1_val, .{ .imm = 0 }, rd);
        }
        switch (rs1_val) {
            .reg => |r| {
                // Bitmask encoding: ones in bit positions [shamt .. 31],
                // i.e., ((1 << (32 - shamt)) - 1) << shamt
                const shift_amt: u6 = @intCast(shamt);
                const ones: u64 = (@as(u64, 1) << (32 - shift_amt)) - 1;
                const mask: u64 = ones << shift_amt;
                self.emit(.{
                    .rd = rd,
                    .rs1 = r,
                    .rs2 = 0,
                    .imm = mask,
                    .kind = .VirtualROTRIW,
                });
                return .{ .reg = rd };
            },
            .imm => |val| {
                const val32: u32 = @truncate(val);
                return .{ .imm = @as(u64, std.math.rotr(u32, val32, shamt)) };
            },
        }
    }

    /// Composite ROTRi ^ ROTRj on a 32-bit word.
    /// Mirrors InstrAssembler::rotri_xor_rotri32.
    fn rotriXorRotri32(self: *Builder, rs1_val: Value, imm1: u5, imm2: u5, rd: u8, scratch: u8) Value {
        const r1 = self.rotri32(rs1_val, imm1, scratch);
        const r2 = self.rotri32(rs1_val, imm2, rd);
        return self.xorOp(r1, r2, rd);
    }

    // -----------------------------------------------------------------------
    // SHA-256 helper functions
    // -----------------------------------------------------------------------

    /// Sigma_0: ROTR2(x) ^ ROTR13(x) ^ ROTR22(x)
    fn shaSigma0(self: *Builder, rs1_val: Value, rd: u8, ss: u8) Value {
        const rotri_xor = self.rotriXorRotri32(rs1_val, 2, 13, rd, ss);
        const rotri_22 = self.rotri32(rs1_val, 22, ss);
        return self.xorOp(rotri_xor, rotri_22, rd);
    }

    /// Sigma_1: ROTR6(x) ^ ROTR11(x) ^ ROTR25(x)
    fn shaSigma1(self: *Builder, rs1_val: Value, rd: u8, ss: u8) Value {
        const rotri_xor = self.rotriXorRotri32(rs1_val, 6, 11, rd, ss);
        const rotri_25 = self.rotri32(rs1_val, 25, ss);
        return self.xorOp(rotri_xor, rotri_25, rd);
    }

    /// sigma_0 (word schedule): ROTR7(x) ^ ROTR18(x) ^ SHR3(x)
    fn shaWordSigma0(self: *Builder, rs1_reg: u8, rd: u8, ss: u8) void {
        _ = self.rotriXorRotri32(.{ .reg = rs1_reg }, 7, 18, rd, ss);
        _ = self.srli(.{ .reg = rs1_reg }, 3, ss);
        _ = self.xorOp(.{ .reg = rd }, .{ .reg = ss }, rd);
    }

    /// sigma_1 (word schedule): ROTR17(x) ^ ROTR19(x) ^ SHR10(x)
    fn shaWordSigma1(self: *Builder, rs1_reg: u8, rd: u8, ss: u8) void {
        _ = self.rotriXorRotri32(.{ .reg = rs1_reg }, 17, 19, rd, ss);
        _ = self.srli(.{ .reg = rs1_reg }, 10, ss);
        _ = self.xorOp(.{ .reg = rd }, .{ .reg = ss }, rd);
    }

    /// Ch(E, F, G) = (E & F) ^ (~E & G)
    fn shaCh(self: *Builder, e: Value, f: Value, g: Value, rd: u8, ss: u8) Value {
        const e_and_f = self.andOp(e, f, ss);
        switch (e) {
            .reg => |r1| switch (g) {
                .reg => |r3| {
                    // Use ANDN: rd = G & ~E
                    self.emit(.{ .rd = rd, .rs1 = r3, .rs2 = r1, .imm = 0, .kind = .ANDN });
                    const neg_e_and_g: Value = .{ .reg = rd };
                    return self.xorOp(e_and_f, neg_e_and_g, rd);
                },
                else => {},
            },
            else => {},
        }
        // Fallback for immediate values (used in first few rounds)
        const neg_e = self.xorOp(e, .{ .imm = 0xFFFFFFFF }, rd);
        const neg_e_and_g = self.andOp(neg_e, g, rd);
        return self.xorOp(e_and_f, neg_e_and_g, rd);
    }

    /// Maj(A, B, C) = (A & B) ^ (A & C) ^ (B & C)
    ///             = (B & C) ^ (A & (B ^ C))
    fn shaMaj(self: *Builder, a: Value, b: Value, c: Value, rd: u8, ss: u8) Value {
        const b_and_c = self.andOp(b, c, ss);
        const b_xor_c = self.xorOp(b, c, rd);
        const a_and_bxc = self.andOp(a, b_xor_c, rd);
        return self.xorOp(b_and_c, a_and_bxc, rd);
    }

    // -----------------------------------------------------------------------
    // State variable access
    // -----------------------------------------------------------------------

    /// Returns Value (Imm or Reg) for the named working variable (0=A .. 7=H).
    /// When `initial` is true, uses BLOCK constants for early rounds until
    /// the variable has been computed.
    fn vri(self: *const Builder, shift_idx: u8) Value {
        if (self.initial) {
            const should_use_imm = switch (self.round) {
                0 => true,
                1 => (shift_idx != 0 and shift_idx != 4), // not A, not E
                2 => (shift_idx != 0 and shift_idx != 1 and shift_idx != 4 and shift_idx != 5), // not A,B,E,F
                3 => (shift_idx == 3 or shift_idx == 7), // only D, H
                else => false,
            };
            if (should_use_imm) {
                const si: i32 = @intCast(shift_idx);
                const idx = rem_euclid_8(si - self.round);
                return .{ .imm = BLOCK[idx] };
            }
        }
        return .{ .reg = self.vr(shift_idx) };
    }

    /// Maps working variable index (0=A..7=H) to its current register.
    /// For non-initial mode, early rounds use iv[] registers for values
    /// not yet computed.
    fn vr(self: *const Builder, shift_idx: u8) u8 {
        if (!self.initial) {
            const should_use_iv = switch (self.round) {
                0 => true,
                1 => (shift_idx != 0 and shift_idx != 4), // not A, not E
                2 => (shift_idx != 0 and shift_idx != 1 and shift_idx != 4 and shift_idx != 5),
                3 => (shift_idx == 3 or shift_idx == 7), // only D, H
                else => false,
            };
            if (should_use_iv) {
                const si: i32 = @intCast(shift_idx);
                const idx = rem_euclid_8(si - self.round);
                return IV_BASE + idx;
            }
        }
        const si: i32 = @intCast(shift_idx);
        const idx = rem_euclid_8(-self.round + si);
        return STATE_BASE + idx;
    }

    /// Register holding W[round + shift] (mod 16).
    fn w(self: *const Builder, shift: i32) u8 {
        const idx = rem_euclid_16(self.round + shift);
        return MESSAGE_BASE + idx;
    }

    // -----------------------------------------------------------------------
    // Message schedule update
    // -----------------------------------------------------------------------

    /// For rounds >= 16, updates W[t] = sigma1(W[t-2]) + W[t-7] + sigma0(W[t-15]) + W[t-16]
    fn updateW(self: *Builder, ss: [2]u8) void {
        if (self.round < 16) return;

        // sigma0(W[t-15])
        self.shaWordSigma0(self.w(-15), ss[0], ss[1]);
        // W[t-16] += sigma0
        _ = self.add(.{ .reg = self.w(-16) }, .{ .reg = ss[0] }, self.w(-16));
        // W[t-16] += W[t-7]
        _ = self.add(.{ .reg = self.w(-7) }, .{ .reg = self.w(-16) }, self.w(-16));
        // sigma1(W[t-2])
        self.shaWordSigma1(self.w(-2), ss[0], ss[1]);
        // W[t-16] += sigma1  => now W[t]
        _ = self.add(.{ .reg = self.w(-16) }, .{ .reg = ss[0] }, self.w(-16));
    }

    // -----------------------------------------------------------------------
    // Load / store helpers (64-bit paired u32 mode)
    // -----------------------------------------------------------------------

    /// Load two u32 values packed in a single u64 doubleword.
    /// vr_lo gets the low 32 bits (with junk in upper 32), vr_hi gets the high 32 bits.
    fn loadPairedU32Dirty(self: *Builder, base: u8, offset: u64, vr_lo: u8, vr_hi: u8) void {
        // LD vr_lo, offset(base)
        self.emit(.{ .rd = vr_lo, .rs1 = base, .rs2 = 0, .imm = offset, .kind = .LD });
        // SRLI vr_hi, vr_lo, 32 -> decomposes to VirtualSRLI in Jolt
        const shift: u7 = 32;
        const ones: u128 = (@as(u128, 1) << @intCast(64 - @as(u8, shift))) - 1;
        const bitmask: u64 = @truncate(ones << shift);
        self.emit(.{ .rd = vr_hi, .rs1 = vr_lo, .rs2 = 0, .imm = bitmask, .kind = .VirtualSRLI });
    }

    /// Store two u32 values by packing them into a single u64 doubleword.
    /// Clobbers both input registers.
    fn storePairedU32(self: *Builder, base: u8, offset: u64, vr_lo: u8, vr_hi: u8) void {
        // VirtualZeroExtendWord vr_lo, vr_lo (mask to 32 bits)
        self.emit(.{ .rd = vr_lo, .rs1 = vr_lo, .rs2 = 0, .imm = 0, .kind = .VirtualZeroExtendWord });
        // SLLI vr_hi, vr_hi, 32 -> decomposes to VirtualMULI in Jolt
        self.emit(.{ .rd = vr_hi, .rs1 = vr_hi, .rs2 = 0, .imm = 32, .kind = .VirtualMULI });
        // OR vr_lo, vr_hi, vr_lo
        self.emit(.{ .rd = vr_lo, .rs1 = vr_hi, .rs2 = vr_lo, .imm = 0, .kind = .OR });
        // SD base, vr_lo, offset
        self.emit(.{ .rd = 0, .rs1 = base, .rs2 = vr_lo, .imm = offset, .kind = .SD });
    }

    // -----------------------------------------------------------------------
    // Round logic
    // -----------------------------------------------------------------------

    fn doRound(self: *Builder) void {
        std.debug.assert(self.round < 64);

        const t1_val = self.computeT1();
        const t2_val = self.computeT2();
        const old_d = self.vri(3); // 'D' - 'A' = 3

        self.applyRoundUpdate(t1_val, t2_val, old_d);
    }

    /// Compute T1 = H + K[round] + Sigma1(E) + Ch(E,F,G) + W[round]
    fn computeT1(self: *Builder) Value {
        const t1 = TEMP_T1;
        const ss = TEMP_SS;
        const ss2 = TEMP_SS2;

        // H + K[round] -- done first because H is Imm longest
        const h_add_k = self.add(.{ .imm = K[@intCast(self.round)] }, self.vri(7), t1); // H = index 7

        // Sigma1(E)
        const sigma_1 = self.shaSigma1(self.vri(4), ss, ss2); // E = index 4

        const add_sigma_1 = self.add(h_add_k, sigma_1, t1);

        // Ch(E, F, G)
        const ch = self.shaCh(self.vri(4), self.vri(5), self.vri(6), ss, ss2); // E=4, F=5, G=6

        const add_ch = self.add(add_sigma_1, ch, t1);

        // Update W (message schedule)
        self.updateW(.{ ss, ss2 });

        // + W[round]
        return self.add(add_ch, .{ .reg = self.w(0) }, t1);
    }

    /// Compute T2 = Sigma0(A) + Maj(A,B,C)
    fn computeT2(self: *Builder) Value {
        const t2 = TEMP_T2;
        const ss = TEMP_SS;
        const ss2 = TEMP_SS2;

        // Sigma0(A)
        const sigma_0 = self.shaSigma0(self.vri(0), t2, ss); // A = index 0

        // Maj(A, B, C)
        const maj = self.shaMaj(self.vri(0), self.vri(1), self.vri(2), ss, ss2); // A=0, B=1, C=2

        return self.add(sigma_0, maj, t2);
    }

    /// Apply the round update: new_A = T1 + T2, new_E = old_D + T1,
    /// then advance the round counter (which rotates the state mapping).
    fn applyRoundUpdate(self: *Builder, t1: Value, t2: Value, old_d: Value) void {
        self.round += 1;
        // After incrementing round, vr('A') / vr('E') point to new positions
        _ = self.add(t1, t2, self.vr(0)); // new A
        _ = self.add(t1, old_d, self.vr(4)); // new E
    }

    // -----------------------------------------------------------------------
    // Final IV addition
    // -----------------------------------------------------------------------

    fn finalAddIv(self: *Builder) void {
        if (!self.initial) {
            for (0..8) |i| {
                const idx: u8 = @intCast(i);
                _ = self.add(self.vri(idx), .{ .reg = IV_BASE + idx }, self.vr(idx));
            }
        } else {
            for (0..8) |i| {
                const idx: u8 = @intCast(i);
                _ = self.add(self.vri(idx), .{ .imm = BLOCK[i] }, self.vr(idx));
            }
        }
    }
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "buildSha256Sequence initial produces non-empty sequence" {
    const allocator = std.testing.allocator;
    var seq = try buildSha256Sequence(allocator, 10, 11, true);
    defer seq.deinit(allocator);
    // Should have a substantial number of instructions (> 2000)
    try std.testing.expect(seq.items.len > 2000);
}

test "buildSha256Sequence non-initial produces non-empty sequence" {
    const allocator = std.testing.allocator;
    var seq = try buildSha256Sequence(allocator, 10, 11, false);
    defer seq.deinit(allocator);
    // Non-initial has additional IV loads, so should be longer
    try std.testing.expect(seq.items.len > 2000);
}

test "buildSha256Sequence initial ends with register zeroing" {
    const allocator = std.testing.allocator;
    var seq = try buildSha256Sequence(allocator, 10, 11, true);
    defer seq.deinit(allocator);

    // Last 28 instructions should be ADDI rd, x0, 0 (register zeroing)
    // Registers 48..71 (24) + 80..83 (4) = 28
    const items = seq.items;
    const zeroing_count: usize = 28;
    for (0..zeroing_count) |j| {
        const instr = items[items.len - zeroing_count + j];
        try std.testing.expectEqual(InstrKind.ADDI, instr.kind);
        try std.testing.expectEqual(@as(u8, 0), instr.rs1);
        try std.testing.expectEqual(@as(u64, 0), instr.imm);
    }
}

test "buildSha256Sequence non-initial ends with register zeroing" {
    const allocator = std.testing.allocator;
    var seq = try buildSha256Sequence(allocator, 10, 11, false);
    defer seq.deinit(allocator);

    // Last 36 instructions should be ADDI rd, x0, 0 (register zeroing)
    // Registers 48..83 = 36
    const items = seq.items;
    const zeroing_count: usize = 36;
    for (0..zeroing_count) |j| {
        const instr = items[items.len - zeroing_count + j];
        try std.testing.expectEqual(InstrKind.ADDI, instr.kind);
        try std.testing.expectEqual(@as(u8, 0), instr.rs1);
        try std.testing.expectEqual(@as(u64, 0), instr.imm);
    }
}

test "buildSha256Sequence initial starts with message loads" {
    const allocator = std.testing.allocator;
    var seq = try buildSha256Sequence(allocator, 10, 11, true);
    defer seq.deinit(allocator);

    // First 8 pairs of (LD, SRLI) = 16 instructions loading message
    for (0..8) |i| {
        const ld_instr = seq.items[i * 2];
        try std.testing.expectEqual(InstrKind.LD, ld_instr.kind);
        try std.testing.expectEqual(@as(u8, 11), ld_instr.rs1); // rs2 = input pointer

        const srli_instr = seq.items[i * 2 + 1];
        try std.testing.expectEqual(InstrKind.SRLI, srli_instr.kind);
        try std.testing.expectEqual(@as(u64, 32), srli_instr.imm);
    }
}

test "buildSha256Sequence non-initial starts with IV loads then message loads" {
    const allocator = std.testing.allocator;
    var seq = try buildSha256Sequence(allocator, 10, 11, false);
    defer seq.deinit(allocator);

    // First 4 pairs of (LD, SRLI) = 8 instructions loading IV from rs1
    for (0..4) |i| {
        const ld_instr = seq.items[i * 2];
        try std.testing.expectEqual(InstrKind.LD, ld_instr.kind);
        try std.testing.expectEqual(@as(u8, 10), ld_instr.rs1); // rs1 = state pointer
    }

    // Next 8 pairs of (LD, SRLI) = 16 instructions loading message from rs2
    for (0..8) |i| {
        const ld_instr = seq.items[8 + i * 2];
        try std.testing.expectEqual(InstrKind.LD, ld_instr.kind);
        try std.testing.expectEqual(@as(u8, 11), ld_instr.rs1); // rs2 = input pointer
    }
}

test "constant folding in initial mode round 0" {
    const allocator = std.testing.allocator;
    var seq = try buildSha256Sequence(allocator, 10, 11, true);
    defer seq.deinit(allocator);

    // Basic sanity: the sequence length should be reasonable
    try std.testing.expect(seq.items.len > 2000);
    try std.testing.expect(seq.items.len < 3000);
}

test "rem_euclid_8 correctness" {
    try std.testing.expectEqual(@as(u8, 7), rem_euclid_8(-1));
    try std.testing.expectEqual(@as(u8, 0), rem_euclid_8(0));
    try std.testing.expectEqual(@as(u8, 1), rem_euclid_8(1));
    try std.testing.expectEqual(@as(u8, 0), rem_euclid_8(8));
    try std.testing.expectEqual(@as(u8, 6), rem_euclid_8(-2));
    try std.testing.expectEqual(@as(u8, 1), rem_euclid_8(-7));
    try std.testing.expectEqual(@as(u8, 0), rem_euclid_8(-8));
    try std.testing.expectEqual(@as(u8, 4), rem_euclid_8(-60));
}

test "rem_euclid_16 correctness" {
    try std.testing.expectEqual(@as(u8, 15), rem_euclid_16(-1));
    try std.testing.expectEqual(@as(u8, 0), rem_euclid_16(0));
    try std.testing.expectEqual(@as(u8, 0), rem_euclid_16(16));
    try std.testing.expectEqual(@as(u8, 14), rem_euclid_16(-2));
}
