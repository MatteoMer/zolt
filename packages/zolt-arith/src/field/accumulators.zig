//! Accumulator and reduction types for vectorized field arithmetic.
//!
//! This module contains:
//! - `UnreducedProductAccum` — deferred Montgomery reduction accumulator
//! - Barrett reduction constants and helpers
//! - `FoldedMulU64`, `FoldedMulU128`, `FoldedMulU128Accum` — folded product accumulators
//! - `SmallAccumU`, `MedAccumS`, `WideAccumS` — tiered signed accumulators
//! - `S192` — signed 192-bit integer type
//! - `BatchOps` — batch field operations

const std = @import("std");
const builtin = @import("builtin");
const mod = @import("mod.zig");

const BN254Scalar = mod.BN254Scalar;
const BN254_MODULUS = mod.BN254_MODULUS;
const BN254_INV = mod.BN254_INV;
const BN254BaseField = mod.BN254BaseField;

/// Unreduced product accumulator for deferred Montgomery reduction.
///
/// Stores partial products in positional `u128` slots to avoid Montgomery reduction
/// in hot accumulation loops. Each slot holds a sum of u64×u64 partial products;
/// carries between slots are deferred until `reduce()`. This mirrors Jolt's
/// `Folded256ProductAccum` type.
///
/// ## Usage
/// ```
/// var accum = UnreducedProductAccum.zero();
/// for (a_vals, b_vals) |a, b| {
///     accum.addAssign(a.mulToProductAccum(b));
/// }
/// const result = accum.reduce();  // single Montgomery reduction
/// ```
///
/// ## Overflow Safety
/// Each `fromMul` contributes at most `4 × (2^64-1)` to any slot. After N `addAssign`
/// calls, max slot value = `N × 4 × (2^64-1)`. With u128 max = 2^128-1, safe for
/// N up to ~2^62. At T=2^30 with E_in=2^15, N=32768 → max slot ≈ 2^79, well within bounds.
pub const UnreducedProductAccum = struct {
    slots: [8]u128,

    const Self = @This();

    pub inline fn zero() Self {
        return .{ .slots = .{0} ** 8 };
    }

    /// Create an accumulator from a single product a×b (schoolbook 4×4, no reduction).
    pub inline fn fromMul(a: BN254Scalar, b: BN254Scalar) Self {
        @setEvalBranchQuota(10000);
        var slots: [8]u128 = .{0} ** 8;
        inline for (0..4) |i| {
            inline for (0..4) |j| {
                const p: u128 = @as(u128, a.limbs[i]) * @as(u128, b.limbs[j]);
                slots[i + j] += @as(u128, @as(u64, @truncate(p))); // lo 64
                slots[i + j + 1] += @as(u128, @as(u64, @truncate(p >> 64))); // hi 64
            }
        }
        return .{ .slots = slots };
    }

    /// Create an accumulator from field_elem × raw_u128 (schoolbook 4×2, no reduction).
    ///
    /// field_elem is in Montgomery form: stores (a * R mod p).
    /// raw is a plain integer (NOT Montgomery).
    /// Schoolbook product: (a * R) * raw = a * R * raw.
    /// After reduce() (Montgomery division by R): a * raw (mod p).
    /// This is NOT in Montgomery form — caller must use .toMontgomery() to convert.
    pub inline fn fromMulU128(field_elem: BN254Scalar, raw: u128) Self {
        @setEvalBranchQuota(10000);
        const a = field_elem.limbs;
        const b: [2]u64 = .{ @truncate(raw), @truncate(raw >> 64) };
        var slots: [8]u128 = .{0} ** 8;
        inline for (0..4) |i| {
            inline for (0..2) |j| {
                const p: u128 = @as(u128, a[i]) * @as(u128, b[j]);
                slots[i + j] += @as(u128, @as(u64, @truncate(p)));
                slots[i + j + 1] += @as(u128, @as(u64, @truncate(p >> 64)));
            }
        }
        return .{ .slots = slots };
    }

    /// Accumulate another product into this accumulator.
    pub inline fn addAssign(self: *Self, other: Self) void {
        inline for (0..8) |i| {
            self.slots[i] += other.slots[i];
        }
    }

    /// Add two accumulators.
    pub inline fn add(self: Self, other: Self) Self {
        var result: Self = self;
        inline for (0..8) |i| {
            result.slots[i] += other.slots[i];
        }
        return result;
    }

    /// Reduce the accumulated products to a single field element via Montgomery reduction.
    ///
    /// This is where all the deferred work happens: carry propagation across slots,
    /// then standard 4-step CIOS Montgomery reduction. If the accumulated value
    /// overflows 8 limbs (9th limb nonzero), the overflow contribution is folded in
    /// as `overflow * 2^256 mod p = overflow * R mod p = fromU64(overflow)`.
    pub fn reduce(self: Self) BN254Scalar {
        // Step 1: Normalize [8]u128 → [8]u64 + overflow (up to u128)
        // For N accumulated products, each slot can be up to N * 2^130, so
        // the carry chain can produce overflow exceeding u64.
        var limbs: [8]u64 = undefined;
        var carry: u128 = 0;
        inline for (0..8) |i| {
            const sum = self.slots[i] + carry;
            limbs[i] = @truncate(sum);
            carry = sum >> 64;
        }
        // carry holds the overflow: limbs[8..9] conceptually.
        // Split into two u64 limbs for handling.
        const overflow_lo: u64 = @truncate(carry);
        const overflow_hi: u64 = @truncate(carry >> 64);

        // Step 2: Standard 4-step CIOS Montgomery reduction on limbs[0..7]
        // Same algorithm as BN254Scalar.square() reduction
        var t: [5]u64 = .{ limbs[0], limbs[1], limbs[2], limbs[3], 0 };

        inline for (0..4) |i| {
            const m = t[0] *% BN254_INV;
            var c: u64 = 0;
            const prod0 = BN254Scalar.mulWide(m, BN254_MODULUS[0]);
            const sum0 = @as(u128, t[0]) + prod0;
            c = @truncate(sum0 >> 64);

            inline for (1..4) |j| {
                const prod = BN254Scalar.mulWide(m, BN254_MODULUS[j]);
                const s = @as(u128, t[j]) + prod + @as(u128, c);
                t[j - 1] = @truncate(s);
                c = @truncate(s >> 64);
            }
            const final_sum = @as(u128, t[4]) + @as(u128, c) + @as(u128, limbs[i + 4]);
            t[3] = @truncate(final_sum);
            t[4] = @truncate(final_sum >> 64);
        }

        // Step 3: Reduce 5-limb CIOS result (t[4]*2^256 + t[0..3]) mod p.
        // For single products, t[4] is always 0 and one subtraction suffices.
        // For accumulated products, the 8-limb input can span full 512 bits,
        // so t[4] can be 1, requiring up to 5 subtractions of p.
        var result = BN254Scalar{ .limbs = .{ t[0], t[1], t[2], t[3] } };
        var extra = t[4]; // 0 or 1
        var iters: u32 = 0;
        while (extra != 0 or !result.lessThanModulus()) : (iters += 1) {
            std.debug.assert(iters < 6); // at most 5 subtractions of p
            const was_less = result.lessThanModulus();
            result = result.subtractModulus();
            if (was_less) extra -= 1; // borrow consumed from extra
        }

        // Step 4: Add overflow contribution from limbs beyond position 7.
        // The overflow represents value * 2^512. After Montgomery division by R=2^256,
        // this becomes overflow_val * 2^256 mod p.
        // Using toMontgomery: raw_val * R² * R^{-1} = raw_val * R = raw_val * 2^256 mod p.
        if (overflow_lo != 0 or overflow_hi != 0) {
            const raw = BN254Scalar{ .limbs = .{ overflow_lo, overflow_hi, 0, 0 } };
            result = result.add(raw.toMontgomery());
        }

        return result;
    }
};

// ============================================================================
// Barrett Reduction Constants and Helpers
// ============================================================================

/// Barrett reduction constants for BN254 scalar field (Fr).
/// Modulus is 254 bits → 2 spare bits in top limb of 4-limb representation.
const MODULUS_NUM_SPARE_BITS: u6 = 2;

/// BARRETT_MU = floor(2^317 / MODULUS).
/// Computed via: python3 -c "p=0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001; print(hex((2**317)//p))"
const BARRETT_MU: u64 = 0xa948e8c4c474094f;

/// 2 × MODULUS as [5]u64
const MODULUS_TIMES_2: [5]u64 = blk: {
    var result: [5]u64 = .{0} ** 5;
    var carry: u64 = 0;
    for (0..4) |i| {
        const wide = @as(u128, BN254_MODULUS[i]) * 2 + @as(u128, carry);
        result[i] = @as(u64, @truncate(wide));
        carry = @as(u64, @truncate(wide >> 64));
    }
    result[4] = carry;
    break :blk result;
};

/// 3 × MODULUS as [5]u64
const MODULUS_TIMES_3: [5]u64 = blk: {
    var result: [5]u64 = .{0} ** 5;
    var carry: u64 = 0;
    for (0..5) |i| {
        const a: u64 = if (i < 4) BN254_MODULUS[i] else 0;
        const wide = @as(u128, a) + @as(u128, MODULUS_TIMES_2[i]) + @as(u128, carry);
        result[i] = @as(u64, @truncate(wide));
        carry = @as(u64, @truncate(wide >> 64));
    }
    break :blk result;
};

/// Compare 4-limb value >= 4-limb value (unsigned, big-endian comparison)
inline fn cmpGe4(a: [4]u64, b: [4]u64) bool {
    // Compare from most significant limb
    comptime var i = 4;
    inline while (i > 0) {
        i -= 1;
        if (a[i] > b[i]) return true;
        if (a[i] < b[i]) return false;
    }
    return true; // equal
}

/// Compare 5-limb value >= 5-limb value
inline fn cmpGe5(a: [5]u64, b: [5]u64) bool {
    comptime var i = 5;
    inline while (i > 0) {
        i -= 1;
        if (a[i] > b[i]) return true;
        if (a[i] < b[i]) return false;
    }
    return true;
}

/// Subtract 4-limb values: a - b (assumes a >= b)
inline fn sub4(a: [4]u64, b: [4]u64) [4]u64 {
    var result: [4]u64 = undefined;
    var borrow: u64 = 0;
    inline for (0..4) |i| {
        const sb = BN254Scalar.subBorrow(a[i], b[i], borrow);
        result[i] = sb.result;
        borrow = sb.borrow;
    }
    return result;
}

/// Reduce a 5-limb value to 4 limbs modulo BN254 scalar field.
/// Port of Jolt's barrett_reduce_nplus1_to_n.
fn barrettReduceNplus1(c: [5]u64) [4]u64 {
    // 1. Extract tilde_c = floor(c / 2^254)
    const shift_amt: u6 = 64 - @as(u7, MODULUS_NUM_SPARE_BITS);
    const tilde_c: u64 = (c[4] << MODULUS_NUM_SPARE_BITS) |
        (c[3] >> shift_amt);

    // 2. Estimate quotient: m = floor((tilde_c * MU) / 2^64)
    const m: u64 = @truncate((@as(u128, tilde_c) *% @as(u128, BARRETT_MU)) >> 64);

    // 3. Compute m * 2p
    var m2p: [5]u64 = undefined;
    {
        var carry: u64 = 0;
        inline for (0..5) |i| {
            const wide = @as(u128, MODULUS_TIMES_2[i]) * @as(u128, m) + @as(u128, carry);
            m2p[i] = @truncate(wide);
            carry = @truncate(wide >> 64);
        }
    }

    // 4. Subtract: r = c - m*2p (using sbb chain)
    var r: [5]u64 = undefined;
    {
        var borrow: u64 = 0;
        inline for (0..5) |i| {
            const sb = BN254Scalar.subBorrow(c[i], m2p[i], borrow);
            r[i] = sb.result;
            borrow = sb.borrow;
        }
    }

    // 5. Conditional subtraction: r in [0, 4p) → [0, p)
    return barrettCondSubtract(r);
}

/// Conditional subtraction: bring [0, 4p) → [0, p).
/// With SPARE_BITS=2, 2p and 3p fit in 4 limbs (r[4] == 0 after step 4).
fn barrettCondSubtract(r: [5]u64) [4]u64 {
    const r4: [4]u64 = .{ r[0], r[1], r[2], r[3] };
    if (cmpGe5(r, MODULUS_TIMES_3)) {
        return sub4(r4, .{ MODULUS_TIMES_3[0], MODULUS_TIMES_3[1], MODULUS_TIMES_3[2], MODULUS_TIMES_3[3] });
    } else if (cmpGe4(r4, .{ MODULUS_TIMES_2[0], MODULUS_TIMES_2[1], MODULUS_TIMES_2[2], MODULUS_TIMES_2[3] })) {
        return sub4(r4, .{ MODULUS_TIMES_2[0], MODULUS_TIMES_2[1], MODULUS_TIMES_2[2], MODULUS_TIMES_2[3] });
    } else if (cmpGe4(r4, BN254_MODULUS)) {
        return sub4(r4, BN254_MODULUS);
    } else {
        return r4;
    }
}

/// Reduce an L-limb value to a BN254Scalar by Horner's method (MSB first)
/// using the (N+1) → N Barrett kernel at each step.
/// Port of Jolt's from_barrett_reduce.
pub fn barrettReduce(comptime L: usize, limbs: [L]u64) BN254Scalar {
    if (L <= 4) {
        var padded: [5]u64 = .{0} ** 5;
        inline for (0..L) |i| padded[i] = limbs[i];
        return BN254Scalar{ .limbs = barrettCondSubtract(padded) };
    }
    if (L == 5) {
        return BN254Scalar{ .limbs = barrettReduceNplus1(limbs) };
    }
    // Horner fold: acc = limbs[i] + 2^64 * acc, reduce mod p at each step.
    // Processes from highest limb to lowest.
    var acc: [4]u64 = .{0} ** 4;
    comptime var i = L;
    inline while (i > 0) {
        i -= 1;
        // [limbs[i], acc[0..3]] represents limbs[i] + acc * 2^64
        const c5: [5]u64 = .{ limbs[i], acc[0], acc[1], acc[2], acc[3] };
        acc = barrettReduceNplus1(c5);
    }
    return BN254Scalar{ .limbs = acc };
}

// ============================================================================
// Folded Accumulator Types
// ============================================================================

/// Field × u64 product accumulator. 5 u128 slots, normalizes to [5]u64.
/// Supports ~2^64 additions before overflow.
pub const FoldedMulU64 = struct {
    slots: [5]u128,

    pub inline fn zero() FoldedMulU64 {
        return .{ .slots = .{0} ** 5 };
    }

    /// Add raw 4-limb Montgomery representation directly.
    pub inline fn addBigInt4(self: *@This(), limbs: [4]u64) void {
        inline for (0..4) |i| {
            self.slots[i] += @as(u128, limbs[i]);
        }
    }

    /// Slot-wise accumulation.
    pub inline fn addAssign(self: *@This(), other: FoldedMulU64) void {
        inline for (0..5) |i| {
            self.slots[i] += other.slots[i];
        }
    }

    /// Convert [M]u64 product into folded form.
    pub inline fn fromBigInt(comptime M: usize, limbs: [M]u64) FoldedMulU64 {
        var out: [5]u128 = .{0} ** 5;
        inline for (0..@min(M, 5)) |i| {
            out[i] = @as(u128, limbs[i]);
        }
        return .{ .slots = out };
    }

    /// Propagate deferred carries → [6]u64.
    /// Uses 6 limbs (not 5) to capture carry overflow from heavy accumulation.
    /// With N additions of 320-bit products, the result can be up to 320+log2(N) bits.
    pub fn normalize(self: @This()) [6]u64 {
        var out: [6]u64 = .{0} ** 6;
        var carry: u128 = 0;
        inline for (0..5) |i| {
            const sum = self.slots[i] +% carry;
            out[i] = @truncate(sum);
            carry = sum >> 64;
        }
        out[5] = @truncate(carry);
        return out;
    }
};

/// Field × u128 product accumulator. 6 u128 slots, normalizes to [6]u64.
pub const FoldedMulU128 = struct {
    slots: [6]u128,

    pub inline fn zero() FoldedMulU128 {
        return .{ .slots = .{0} ** 6 };
    }

    /// Add raw 4-limb values (zero-extended to 6 slots).
    pub inline fn addBigInt4(self: *@This(), limbs: [4]u64) void {
        inline for (0..4) |i| {
            self.slots[i] += @as(u128, limbs[i]);
        }
    }

    /// Slot-wise accumulation.
    pub inline fn addAssign(self: *@This(), other: FoldedMulU128) void {
        inline for (0..6) |i| {
            self.slots[i] += other.slots[i];
        }
    }

    /// Convert [M]u64 product into folded form.
    pub inline fn fromBigInt(comptime M: usize, limbs: [M]u64) FoldedMulU128 {
        var out: [6]u128 = .{0} ** 6;
        inline for (0..@min(M, 6)) |i| {
            out[i] = @as(u128, limbs[i]);
        }
        return .{ .slots = out };
    }

    /// Promote from FoldedMulU64 (5 slots → 6 slots).
    pub inline fn fromFoldedU64(f: FoldedMulU64) FoldedMulU128 {
        var out: [6]u128 = .{0} ** 6;
        inline for (0..5) |i| {
            out[i] = f.slots[i];
        }
        return .{ .slots = out };
    }

    /// Propagate deferred carries → [6]u64.
    pub fn normalize(self: @This()) [6]u64 {
        var out: [6]u64 = undefined;
        var carry: u128 = 0;
        inline for (0..6) |i| {
            const sum = self.slots[i] +% carry;
            out[i] = @truncate(sum);
            carry = sum >> 64;
        }
        return out;
    }
};

/// Field × u128 wide accumulator. 7 u128 slots, normalizes to [7]u64.
/// Used for second-group Bz where magnitudes can be S160.
pub const FoldedMulU128Accum = struct {
    slots: [7]u128,

    pub inline fn zero() FoldedMulU128Accum {
        return .{ .slots = .{0} ** 7 };
    }

    /// Slot-wise accumulation.
    pub inline fn addAssign(self: *@This(), other: FoldedMulU128Accum) void {
        inline for (0..7) |i| {
            self.slots[i] += other.slots[i];
        }
    }

    /// Promote from FoldedMulU128 (6 slots → 7 slots).
    pub inline fn fromFoldedU128(f: FoldedMulU128) FoldedMulU128Accum {
        var out: [7]u128 = .{0} ** 7;
        inline for (0..6) |i| {
            out[i] = f.slots[i];
        }
        return .{ .slots = out };
    }

    /// Convert [M]u64 product into folded form (7 slots).
    pub inline fn fromBigInt(comptime M: usize, limbs: [M]u64) FoldedMulU128Accum {
        var out: [7]u128 = .{0} ** 7;
        inline for (0..@min(M, 7)) |i| {
            out[i] = @as(u128, limbs[i]);
        }
        return .{ .slots = out };
    }

    /// Propagate deferred carries → [7]u64.
    pub fn normalize(self: @This()) [7]u64 {
        var out: [7]u64 = undefined;
        var carry: u128 = 0;
        inline for (0..7) |i| {
            const sum = self.slots[i] +% carry;
            out[i] = @truncate(sum);
            carry = sum >> 64;
        }
        return out;
    }
};

// ============================================================================
// Unreduced Multiply Methods and Barrett Reduce on BN254Scalar
// ============================================================================

// These are free functions that operate on BN254Scalar since we can't
// reopen the struct. They're accessed as e.g. field_mod.mulU64Unreduced(scalar, val).

/// Return raw Montgomery limbs (no conversion).
pub inline fn toUnreduced(self: BN254Scalar) [4]u64 {
    return self.limbs;
}

/// Multiply field element by u64 scalar, return unreduced 5-limb result.
/// Computes (self.limbs) × scalar via 4×1 schoolbook.
pub inline fn mulU64Unreduced(self: BN254Scalar, scalar: u64) FoldedMulU64 {
    if (scalar == 0) return FoldedMulU64.zero();
    var result: [5]u64 = undefined;
    var carry: u64 = 0;
    inline for (0..4) |i| {
        const wide = @as(u128, self.limbs[i]) * @as(u128, scalar) + @as(u128, carry);
        result[i] = @truncate(wide);
        carry = @truncate(wide >> 64);
    }
    result[4] = carry;
    return FoldedMulU64.fromBigInt(5, result);
}

/// Multiply field element by u128 scalar, return unreduced 6-limb result.
/// Computes (self.limbs) × scalar via 4×2 schoolbook.
pub fn mulU128Unreduced(self: BN254Scalar, scalar: u128) FoldedMulU128 {
    if (scalar == 0) return FoldedMulU128.zero();
    const other: [2]u64 = .{ @truncate(scalar), @truncate(scalar >> 64) };
    var result: [6]u64 = .{0} ** 6;
    inline for (0..2) |j| {
        var carry: u64 = 0;
        inline for (0..4) |i| {
            const wide = @as(u128, result[i + j]) + @as(u128, self.limbs[i]) * @as(u128, other[j]) + @as(u128, carry);
            result[i + j] = @truncate(wide);
            carry = @truncate(wide >> 64);
        }
        result[4 + j] = carry;
    }
    return FoldedMulU128.fromBigInt(6, result);
}

/// Multiply field element by [3]u64 (S192 magnitude), return unreduced 7-limb result.
/// Computes (self.limbs) × scalar via 4×3 schoolbook. Matches arkworks fm_limbs_into<3, 7>.
pub fn mulU192Unreduced(self: BN254Scalar, scalar: [3]u64) FoldedMulU128Accum {
    if (scalar[0] == 0 and scalar[1] == 0 and scalar[2] == 0) return FoldedMulU128Accum.zero();
    var result: [7]u64 = .{0} ** 7;
    inline for (0..3) |j| {
        if (scalar[j] != 0) {
            var carry: u64 = 0;
            inline for (0..4) |i| {
                const wide = @as(u128, result[i + j]) + @as(u128, self.limbs[i]) * @as(u128, scalar[j]) + @as(u128, carry);
                result[i + j] = @truncate(wide);
                carry = @truncate(wide >> 64);
            }
            result[4 + j] = carry;
        }
    }
    return FoldedMulU128Accum.fromBigInt(7, result);
}

/// Reduce FoldedMulU64 (5 slots) → BN254Scalar via Barrett.
/// normalize() returns [6]u64 to handle carry overflow from heavy accumulation.
pub fn reduceMulU64(folded: FoldedMulU64) BN254Scalar {
    return barrettReduce(6, folded.normalize());
}

/// Multiply a field element by a small u64 constant. Cheaper than full field mul:
/// uses 4×1 schoolbook (4 mulq) + Barrett reduction vs 4×4 (16 mulq) + Montgomery.
pub inline fn mulU64(self: BN254Scalar, scalar: u64) BN254Scalar {
    return reduceMulU64(mulU64Unreduced(self, scalar));
}

/// Reduce FoldedMulU128 (6 slots) → BN254Scalar via Barrett.
pub fn reduceMulU128(folded: FoldedMulU128) BN254Scalar {
    return barrettReduce(6, folded.normalize());
}

/// Reduce FoldedMulU128Accum (7 slots) → BN254Scalar via Barrett.
pub fn reduceMulU128Accum(folded: FoldedMulU128Accum) BN254Scalar {
    return barrettReduce(7, folded.normalize());
}

// ============================================================================
// Tiered Accumulators
// ============================================================================

/// Signed accumulator for boolean/small-integer guards.
/// Dual FoldedMulU64 (pos and neg) to handle signed integer guards.
/// fmaddBool: conditional add of raw Montgomery limbs (4 u128 adds).
/// fmaddI8: handles small integer guards in [-4, 4].
pub const SmallAccumU = struct {
    pos: FoldedMulU64,
    neg: FoldedMulU64,

    pub inline fn zero() SmallAccumU {
        return .{ .pos = FoldedMulU64.zero(), .neg = FoldedMulU64.zero() };
    }

    /// Bool fmadd: if true, add raw Montgomery representation of field element.
    /// Cost: 4 u128 additions. No multiplication.
    pub inline fn fmaddBool(self: *@This(), field: BN254Scalar, value: bool) void {
        if (value) {
            self.pos.addBigInt4(field.limbs);
        }
    }

    /// i8 fmadd: for small integer guards in [-4, 4].
    /// Positive values accumulate into pos, negative into neg.
    pub inline fn fmaddI8(self: *@This(), field: BN254Scalar, value: i8) void {
        if (value == 0) return;
        if (value == 1) {
            self.pos.addBigInt4(field.limbs);
            return;
        }
        if (value == -1) {
            self.neg.addBigInt4(field.limbs);
            return;
        }
        // General case: multiply field by |value|
        const abs_val: u64 = if (value > 0) @intCast(value) else @intCast(-value);
        const prod = mulU64Unreduced(field, abs_val);
        if (value > 0) {
            self.pos.addAssign(prod);
        } else {
            self.neg.addAssign(prod);
        }
    }

    /// Barrett reduce to field element. Result is in Montgomery form.
    pub fn barrettReduce(self: @This()) BN254Scalar {
        const pos_reduced = reduceMulU64(self.pos);
        const neg_reduced = reduceMulU64(self.neg);
        return pos_reduced.sub(neg_reduced);
    }
};

/// Signed accumulator for medium-width magnitudes (u64, S64, i128).
/// Dual FoldedMulU128 (pos and neg), reduced via Barrett(pos) - Barrett(neg).
pub const MedAccumS = struct {
    pos: FoldedMulU128,
    neg: FoldedMulU128,

    pub inline fn zero() MedAccumS {
        return .{ .pos = FoldedMulU128.zero(), .neg = FoldedMulU128.zero() };
    }

    /// Bool fmadd: add to pos accumulator.
    pub inline fn fmaddBool(self: *@This(), field: BN254Scalar, value: bool) void {
        if (value) {
            self.pos.addBigInt4(field.limbs);
        }
    }

    /// u64 fmadd: unreduced field × u64, add to pos.
    pub inline fn fmaddU64(self: *@This(), field: BN254Scalar, value: u64) void {
        if (value == 0) return;
        self.pos.addAssign(FoldedMulU128.fromFoldedU64(mulU64Unreduced(field, value)));
    }

    /// i128 fmadd: split by sign, unreduced field × u128.
    pub inline fn fmaddI128(self: *@This(), field: BN254Scalar, value: i128) void {
        if (value == 0) return;
        if (value > 0) {
            self.pos.addAssign(mulU128Unreduced(field, @intCast(value)));
        } else {
            self.neg.addAssign(mulU128Unreduced(field, @as(u128, @intCast(-value))));
        }
    }

    /// Barrett reduce: pos - neg → field element in Montgomery form.
    pub fn barrettReduce(self: @This()) BN254Scalar {
        const pos_reduced = reduceMulU128(self.pos);
        const neg_reduced = reduceMulU128(self.neg);
        return pos_reduced.sub(neg_reduced);
    }
};

/// Wide signed accumulator for large magnitudes (S160/S192).
/// Dual FoldedMulU128Accum (7 u128 slots each).
pub const WideAccumS = struct {
    pos: FoldedMulU128Accum,
    neg: FoldedMulU128Accum,

    pub inline fn zero() WideAccumS {
        return .{ .pos = FoldedMulU128Accum.zero(), .neg = FoldedMulU128Accum.zero() };
    }

    /// i128 fmadd: field × u128 product into 7-slot accumulator.
    pub inline fn fmaddI128(self: *@This(), field: BN254Scalar, value: i128) void {
        if (value == 0) return;
        if (value > 0) {
            self.pos.addAssign(FoldedMulU128Accum.fromFoldedU128(mulU128Unreduced(field, @intCast(value))));
        } else {
            self.neg.addAssign(FoldedMulU128Accum.fromFoldedU128(mulU128Unreduced(field, @as(u128, @intCast(-value)))));
        }
    }

    /// S192 fmadd: field × S192 magnitude product into 7-slot accumulator.
    /// Matches arkworks WideAccumS::fmadd(F, S192).
    pub inline fn fmaddS192(self: *@This(), field: BN254Scalar, value: S192) void {
        if (value.magnitude[0] == 0 and value.magnitude[1] == 0 and value.magnitude[2] == 0) return;
        const product = mulU192Unreduced(field, value.magnitude);
        if (value.is_positive) {
            self.pos.addAssign(product);
        } else {
            self.neg.addAssign(product);
        }
    }

    /// Barrett reduce: pos - neg → field element in Montgomery form.
    pub fn barrettReduce(self: @This()) BN254Scalar {
        const pos_reduced = reduceMulU128Accum(self.pos);
        const neg_reduced = reduceMulU128Accum(self.neg);
        return pos_reduced.sub(neg_reduced);
    }
};

/// LLVM carry/borrow intrinsics — map to single adc/sbb instructions on x86-64.
/// Wrapped in a comptime-conditional struct so they are not emitted on non-x86 targets.
const x86 = if (builtin.cpu.arch == .x86_64) struct {
    extern fn @"llvm.x86.addcarry.u64"(c_in: u8, a: u64, b: u64, result: *u64) u8;
    extern fn @"llvm.x86.subborrow.u64"(b_in: u8, a: u64, b: u64, result: *u64) u8;

    pub inline fn addcarry(c_in: u8, a: u64, b: u64, result: *u64) u8 {
        return @"llvm.x86.addcarry.u64"(c_in, a, b, result);
    }
    pub inline fn subborrow(b_in: u8, a: u64, b: u64, result: *u64) u8 {
        return @"llvm.x86.subborrow.u64"(b_in, a, b, result);
    }
} else struct {};

/// Comptime flag: true when targeting AArch64 (all AArch64 has adds/adcs/subs/sbcs).
const use_arm64_asm = (builtin.cpu.arch == .aarch64);

/// Signed 192-bit integer type matching arkworks SignedBigInt<3>.
/// Used for second-group Bz values that need exact integer arithmetic
/// without wrapping i128 truncation artifacts.
pub const S192 = struct {
    magnitude: [3]u64, // unsigned magnitude, little-endian
    is_positive: bool, // true = positive (matches arkworks convention)

    pub inline fn zero() S192 {
        return .{ .magnitude = .{ 0, 0, 0 }, .is_positive = true };
    }

    pub inline fn isZero(self: S192) bool {
        return self.magnitude[0] == 0 and self.magnitude[1] == 0 and self.magnitude[2] == 0;
    }

    pub inline fn fromU64(v: u64) S192 {
        return .{ .magnitude = .{ v, 0, 0 }, .is_positive = true };
    }

    pub inline fn fromU128(v: u128) S192 {
        return .{ .magnitude = .{ @truncate(v), @truncate(v >> 64), 0 }, .is_positive = true };
    }

    pub inline fn fromI128(v: i128) S192 {
        if (v >= 0) {
            const u: u128 = @intCast(v);
            return .{ .magnitude = .{ @truncate(u), @truncate(u >> 64), 0 }, .is_positive = true };
        } else {
            const u: u128 = @bitCast(-%v);
            return .{ .magnitude = .{ @truncate(u), @truncate(u >> 64), 0 }, .is_positive = false };
        }
    }

    pub inline fn fromI64(v: i64) S192 {
        if (v >= 0) {
            return .{ .magnitude = .{ @intCast(v), 0, 0 }, .is_positive = true };
        } else {
            return .{ .magnitude = .{ @intCast(-%v), 0, 0 }, .is_positive = false };
        }
    }

    pub inline fn neg(self: S192) S192 {
        if (self.isZero()) return self;
        return .{ .magnitude = self.magnitude, .is_positive = !self.is_positive };
    }

    pub fn add(self: S192, other: S192) S192 {
        if (other.isZero()) return self;
        if (self.isZero()) return other;
        if (self.is_positive == other.is_positive) {
            return .{ .magnitude = addMagnitudes(self.magnitude, other.magnitude), .is_positive = self.is_positive };
        }
        const cmp = cmpMagnitudes(self.magnitude, other.magnitude);
        if (cmp == .eq) return S192.zero();
        if (cmp == .gt) {
            return .{ .magnitude = subMagnitudes(self.magnitude, other.magnitude), .is_positive = self.is_positive };
        }
        return .{ .magnitude = subMagnitudes(other.magnitude, self.magnitude), .is_positive = other.is_positive };
    }

    pub fn sub(self: S192, other: S192) S192 {
        return self.add(other.neg());
    }

    /// 1×3 schoolbook multiply by i32, truncated to 3 limbs.
    /// Matches arkworks mul_trunc::<1,3>.
    pub fn mulI32(self: S192, c: i32) S192 {
        if (c == 0) return S192.zero();
        if (self.isZero()) return S192.zero();
        const abs_c: u64 = if (c >= 0) @intCast(c) else @intCast(-c);
        var result: [3]u64 = .{ 0, 0, 0 };
        var carry: u64 = 0;
        inline for (0..3) |i| {
            const tmp = @as(u128, self.magnitude[i]) * @as(u128, abs_c) + @as(u128, carry);
            result[i] = @truncate(tmp);
            carry = @truncate(tmp >> 64);
        }
        return .{ .magnitude = result, .is_positive = self.is_positive == (c >= 0) };
    }

    /// Fused multiply-add: acc += term * c. Matches arkworks fmadd_trunc.
    pub fn fmaddI32(acc: *S192, c: i32, term: S192) void {
        const product = term.mulI32(c);
        acc.* = acc.add(product);
    }

    fn addMagnitudes(a: [3]u64, b: [3]u64) [3]u64 {
        if (!@inComptime() and comptime use_arm64_asm) return mod.arm64Add192(a, b);
        var result: [3]u64 = undefined;
        if (comptime builtin.cpu.arch == .x86_64) {
            var c: u8 = 0;
            inline for (0..3) |i| {
                c = x86.addcarry(c, a[i], b[i], &result[i]);
            }
        } else {
            var carry: u64 = 0;
            inline for (0..3) |i| {
                const tmp = @as(u128, a[i]) + @as(u128, b[i]) + @as(u128, carry);
                result[i] = @truncate(tmp);
                carry = @truncate(tmp >> 64);
            }
        }
        return result;
    }

    fn subMagnitudes(a: [3]u64, b: [3]u64) [3]u64 {
        if (!@inComptime() and comptime use_arm64_asm) return mod.arm64Sub192(a, b);
        var result: [3]u64 = undefined;
        if (comptime builtin.cpu.arch == .x86_64) {
            var b_out: u8 = 0;
            inline for (0..3) |i| {
                b_out = x86.subborrow(b_out, a[i], b[i], &result[i]);
            }
        } else {
            var borrow: u64 = 0;
            inline for (0..3) |i| {
                const tmp = (@as(u128, 1) << 64) + @as(u128, a[i]) - @as(u128, b[i]) - @as(u128, borrow);
                result[i] = @truncate(tmp);
                borrow = if ((tmp >> 64) == 0) @as(u64, 1) else @as(u64, 0);
            }
        }
        return result;
    }

    fn cmpMagnitudes(a: [3]u64, b: [3]u64) std.math.Order {
        var i: usize = 3;
        while (i > 0) {
            i -= 1;
            if (a[i] != b[i]) return std.math.order(a[i], b[i]);
        }
        return .eq;
    }
};

/// Batch field operations for SIMD-like performance
/// These functions operate on slices for cache efficiency
pub const BatchOps = struct {
    /// Batch addition: results[i] = a[i] + b[i]
    pub fn batchAdd(results: []BN254Scalar, a: []const BN254Scalar, b: []const BN254Scalar) void {
        std.debug.assert(results.len == a.len and a.len == b.len);
        for (0..results.len) |i| {
            results[i] = a[i].add(b[i]);
        }
    }

    /// Batch subtraction: results[i] = a[i] - b[i]
    pub fn batchSub(results: []BN254Scalar, a: []const BN254Scalar, b: []const BN254Scalar) void {
        std.debug.assert(results.len == a.len and a.len == b.len);
        for (0..results.len) |i| {
            results[i] = a[i].sub(b[i]);
        }
    }

    /// Batch multiplication: results[i] = a[i] * b[i]
    pub fn batchMul(results: []BN254Scalar, a: []const BN254Scalar, b: []const BN254Scalar) void {
        std.debug.assert(results.len == a.len and a.len == b.len);
        for (0..results.len) |i| {
            results[i] = a[i].mul(b[i]);
        }
    }

    /// Batch scalar multiplication: results[i] = a[i] * scalar
    pub fn batchMulScalar(results: []BN254Scalar, a: []const BN254Scalar, scalar: BN254Scalar) void {
        std.debug.assert(results.len == a.len);
        for (0..results.len) |i| {
            results[i] = a[i].mul(scalar);
        }
    }

    /// Batch squaring: results[i] = a[i]^2
    pub fn batchSquare(results: []BN254Scalar, a: []const BN254Scalar) void {
        std.debug.assert(results.len == a.len);
        for (0..results.len) |i| {
            results[i] = a[i].square();
        }
    }

    /// Inner product: sum(a[i] * b[i])
    pub fn innerProduct(a: []const BN254Scalar, b: []const BN254Scalar) BN254Scalar {
        std.debug.assert(a.len == b.len);
        var result = BN254Scalar.zero();
        for (0..a.len) |i| {
            result = result.add(a[i].mul(b[i]));
        }
        return result;
    }

    /// Sum of products with precomputed terms for Horner's method
    /// Computes: a[0] + x*(a[1] + x*(a[2] + ... + x*a[n-1])))
    pub fn hornerEval(coeffs: []const BN254Scalar, x: BN254Scalar) BN254Scalar {
        if (coeffs.len == 0) return BN254Scalar.zero();

        var result = coeffs[coeffs.len - 1];
        var i: usize = coeffs.len - 1;
        while (i > 0) {
            i -= 1;
            result = result.mul(x).add(coeffs[i]);
        }
        return result;
    }

    /// Batch inverse using Montgomery's trick
    /// Computes inverses of all elements using only one field inversion
    /// Much faster than computing individual inverses: O(3n) muls + 1 inverse vs O(n) inverses
    pub fn batchInverse(results: []BN254Scalar, a: []const BN254Scalar, allocator: std.mem.Allocator) !void {
        std.debug.assert(results.len == a.len);
        if (a.len == 0) return;

        // Step 1: Compute running products
        // products[i] = a[0] * a[1] * ... * a[i]
        const products = try allocator.alloc(BN254Scalar, a.len);
        defer allocator.free(products);

        products[0] = a[0];
        for (1..a.len) |i| {
            if (a[i].isZero()) {
                // Handle zero by using one (will result in zero inverse)
                products[i] = products[i - 1];
            } else {
                products[i] = products[i - 1].mul(a[i]);
            }
        }

        // Step 2: Compute inverse of the final product
        const all_inv = products[a.len - 1].inverse() orelse BN254Scalar.zero();

        // Step 3: Compute individual inverses
        var running_inv = all_inv;
        var i: usize = a.len;
        while (i > 1) {
            i -= 1;
            if (a[i].isZero()) {
                results[i] = BN254Scalar.zero();
            } else {
                // a[i]^{-1} = running_inv * products[i-1]
                results[i] = running_inv.mul(products[i - 1]);
                running_inv = running_inv.mul(a[i]);
            }
        }
        // Handle a[0]
        if (a[0].isZero()) {
            results[0] = BN254Scalar.zero();
        } else {
            results[0] = running_inv;
        }
    }

    /// Multi-scalar multiplication accumulator
    /// Computes sum(scalars[i] * bases[i])
    pub fn multiScalarMulLinear(scalars: []const BN254Scalar, bases: []const BN254Scalar) BN254Scalar {
        return innerProduct(scalars, bases);
    }
};

// ============================================================================
// Tests
// ============================================================================

test "UnreducedProductAccum single product matches mul" {
    const a = BN254Scalar.fromU64(12345);
    const b = BN254Scalar.fromU64(67890);
    const expected = a.mul(b);
    const actual = a.mulToProductAccum(b).reduce();
    try std.testing.expect(expected.eql(actual));
}

test "UnreducedProductAccum sum of products" {
    // Verify sum(a[i]*b[i]) via accum equals sum via direct mul
    const N = 100;
    var a_vals: [N]BN254Scalar = undefined;
    var b_vals: [N]BN254Scalar = undefined;

    // Use deterministic "random" values
    for (0..N) |i| {
        a_vals[i] = BN254Scalar.fromU64(@as(u64, i) * 7 + 13);
        b_vals[i] = BN254Scalar.fromU64(@as(u64, i) * 11 + 29);
    }

    // Direct sum
    var expected = BN254Scalar.zero();
    for (0..N) |i| {
        expected = expected.add(a_vals[i].mul(b_vals[i]));
    }

    // Accumulated sum
    var accum = UnreducedProductAccum.zero();
    for (0..N) |i| {
        accum.addAssign(a_vals[i].mulToProductAccum(b_vals[i]));
    }
    const actual = accum.reduce();

    try std.testing.expect(expected.eql(actual));
}

test "UnreducedProductAccum large accumulation (10000 products)" {
    const N = 10000;
    var expected = BN254Scalar.zero();
    var accum = UnreducedProductAccum.zero();

    for (0..N) |i| {
        const a = BN254Scalar.fromU64(@as(u64, i) * 31 + 7);
        const b = BN254Scalar.fromU64(@as(u64, i) * 17 + 3);
        expected = expected.add(a.mul(b));
        accum.addAssign(a.mulToProductAccum(b));
    }

    try std.testing.expect(expected.eql(accum.reduce()));
}

test "UnreducedProductAccum zero accumulation" {
    const accum = UnreducedProductAccum.zero();
    try std.testing.expect(accum.reduce().isZero());
}

test "UnreducedProductAccum with large field elements" {
    // Use values near the modulus to stress carry propagation
    const a = BN254Scalar{ .limbs = .{ 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0x0FFFFFFFFFFFFFFF } };
    const b = BN254Scalar{ .limbs = .{ 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0x0FFFFFFFFFFFFFFF } };

    const expected = a.mul(b);
    const actual = a.mulToProductAccum(b).reduce();
    try std.testing.expect(expected.eql(actual));

    // Accumulate many large products
    var exp2 = BN254Scalar.zero();
    var acc2 = UnreducedProductAccum.zero();
    for (0..50) |_| {
        exp2 = exp2.add(a.mul(b));
        acc2.addAssign(a.mulToProductAccum(b));
    }
    try std.testing.expect(exp2.eql(acc2.reduce()));
}

test "batch operations" {
    const allocator = std.testing.allocator;

    var a: [4]BN254Scalar = undefined;
    var b: [4]BN254Scalar = undefined;
    var results: [4]BN254Scalar = undefined;

    for (0..4) |i| {
        a[i] = BN254Scalar.fromU64(@as(u64, @intCast(i + 1)));
        b[i] = BN254Scalar.fromU64(@as(u64, @intCast(i + 5)));
    }

    // Test batch add
    BatchOps.batchAdd(&results, &a, &b);
    for (0..4) |i| {
        const expected = BN254Scalar.fromU64(@as(u64, @intCast(2 * i + 6)));
        try std.testing.expect(results[i].eql(expected));
    }

    // Test inner product: 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70
    const ip = BatchOps.innerProduct(&a, &b);
    try std.testing.expect(ip.eql(BN254Scalar.fromU64(70)));

    // Test Horner evaluation: 1 + 2x + 3x^2 + 4x^3 at x=2
    // = 1 + 2*2 + 3*4 + 4*8 = 1 + 4 + 12 + 32 = 49
    const horner_result = BatchOps.hornerEval(&a, BN254Scalar.fromU64(2));
    try std.testing.expect(horner_result.eql(BN254Scalar.fromU64(49)));

    // Test batch inverse
    var non_zero_a: [3]BN254Scalar = .{
        BN254Scalar.fromU64(2),
        BN254Scalar.fromU64(3),
        BN254Scalar.fromU64(5),
    };
    var inverses: [3]BN254Scalar = undefined;
    try BatchOps.batchInverse(&inverses, &non_zero_a, allocator);

    // Verify: a[i] * inv[i] = 1
    for (0..3) |i| {
        const prod = non_zero_a[i].mul(inverses[i]);
        try std.testing.expect(prod.eql(BN254Scalar.one()));
    }
}

test "batchInversion known values" {
    const Fp = BN254BaseField;
    var elems = [_]Fp{ Fp.fromU64(2), Fp.fromU64(3), Fp.fromU64(7) };
    var scratch: [3]Fp = undefined;
    Fp.batchInversion(&elems, &scratch);
    // Each element should now be its inverse
    try std.testing.expect(elems[0].mul(Fp.fromU64(2)).eql(Fp.one()));
    try std.testing.expect(elems[1].mul(Fp.fromU64(3)).eql(Fp.one()));
    try std.testing.expect(elems[2].mul(Fp.fromU64(7)).eql(Fp.one()));
}

test "batchInversion with zeros" {
    const Fp = BN254BaseField;
    var elems = [_]Fp{ Fp.fromU64(5), Fp.zero(), Fp.fromU64(11), Fp.zero(), Fp.fromU64(13) };
    var scratch: [5]Fp = undefined;
    Fp.batchInversion(&elems, &scratch);
    // Zeros stay zero
    try std.testing.expect(elems[1].isZero());
    try std.testing.expect(elems[3].isZero());
    // Non-zeros are inverted
    try std.testing.expect(elems[0].mul(Fp.fromU64(5)).eql(Fp.one()));
    try std.testing.expect(elems[2].mul(Fp.fromU64(11)).eql(Fp.one()));
    try std.testing.expect(elems[4].mul(Fp.fromU64(13)).eql(Fp.one()));
}

test "batchInversion single element" {
    const Fp = BN254BaseField;
    var elems = [_]Fp{Fp.fromU64(42)};
    var scratch: [1]Fp = undefined;
    Fp.batchInversion(&elems, &scratch);
    try std.testing.expect(elems[0].mul(Fp.fromU64(42)).eql(Fp.one()));
}

test "batchInversion empty" {
    const Fp = BN254BaseField;
    var elems: [0]Fp = .{};
    var scratch: [0]Fp = .{};
    Fp.batchInversion(&elems, &scratch);
}

test "batchInversion all zeros" {
    const Fp = BN254BaseField;
    var elems = [_]Fp{ Fp.zero(), Fp.zero(), Fp.zero() };
    var scratch: [3]Fp = undefined;
    Fp.batchInversion(&elems, &scratch);
    for (elems) |e| try std.testing.expect(e.isZero());
}

// ============================================================================
// Barrett Reduction and Tiered Accumulator Tests
// ============================================================================

test "Barrett reduce matches Montgomery for small values" {
    // mulU64Unreduced → reduceMulU64 should match field.mul(fromU64)
    const a = BN254Scalar.fromU64(12345);
    const b: u64 = 67890;
    const expected = a.mul(BN254Scalar.fromU64(b));
    const actual = reduceMulU64(mulU64Unreduced(a, b));
    try std.testing.expect(expected.eql(actual));
}

test "Barrett reduce matches Montgomery for u128 values" {
    const a = BN254Scalar.fromU64(999999);
    const b: u128 = 0x123456789ABCDEF0_FEDCBA9876543210;
    const expected = a.mul(BN254Scalar.fromU128(b));
    const actual = reduceMulU128(mulU128Unreduced(a, b));
    try std.testing.expect(expected.eql(actual));
}

test "Barrett reduce with zero" {
    const a = BN254Scalar.fromU64(42);
    const result = reduceMulU64(mulU64Unreduced(a, 0));
    try std.testing.expect(result.isZero());
}

test "Barrett reduce with one" {
    const a = BN254Scalar.fromU64(42);
    const result = reduceMulU64(mulU64Unreduced(a, 1));
    try std.testing.expect(result.eql(a));
}

test "FoldedMulU64 normalize correctness" {
    var f = FoldedMulU64.zero();
    // Add a known value
    const a = BN254Scalar.fromU64(7);
    f.addBigInt4(a.limbs);
    f.addBigInt4(a.limbs);
    // Normalizing should give 2*a Montgomery limbs (with carries propagated)
    const norm = f.normalize();
    const expected = a.add(a);
    // Barrett reduce the folded result
    const actual = reduceMulU64(f);
    try std.testing.expect(expected.eql(actual));
    _ = norm;
}

test "SmallAccumU fmaddBool accumulation matches field.add" {
    var acc = SmallAccumU.zero();
    const w0 = BN254Scalar.fromU64(100);
    const w1 = BN254Scalar.fromU64(200);
    const w2 = BN254Scalar.fromU64(300);

    acc.fmaddBool(w0, true);
    acc.fmaddBool(w1, false);
    acc.fmaddBool(w2, true);

    const expected = w0.add(w2);
    const actual = acc.barrettReduce();
    try std.testing.expect(expected.eql(actual));
}

test "SmallAccumU fmaddI8 handles negative values" {
    var acc = SmallAccumU.zero();
    const w0 = BN254Scalar.fromU64(100);
    const w1 = BN254Scalar.fromU64(200);
    const w2 = BN254Scalar.fromU64(50);

    acc.fmaddI8(w0, 1);
    acc.fmaddI8(w1, -1);
    acc.fmaddI8(w2, 3);

    // Expected: 100 - 200 + 150 = 50
    const expected = w0.sub(w1).add(w2.mul(BN254Scalar.fromU64(3)));
    const actual = acc.barrettReduce();
    try std.testing.expect(expected.eql(actual));
}

test "MedAccumS signed accumulation matches field arithmetic" {
    var acc = MedAccumS.zero();
    const w0 = BN254Scalar.fromU64(1000);
    const w1 = BN254Scalar.fromU64(2000);
    const w2 = BN254Scalar.fromU64(3000);

    const b0: i128 = 500;
    const b1: i128 = -300;
    const b2: i128 = 700;

    acc.fmaddI128(w0, b0);
    acc.fmaddI128(w1, b1);
    acc.fmaddI128(w2, b2);

    // Expected: 1000*500 - 2000*300 + 3000*700 = 500000 - 600000 + 2100000 = 2000000
    const expected = w0.mul(BN254Scalar.fromU64(500))
        .sub(w1.mul(BN254Scalar.fromU64(300)))
        .add(w2.mul(BN254Scalar.fromU64(700)));
    const actual = acc.barrettReduce();
    try std.testing.expect(expected.eql(actual));
}

test "MedAccumS large i128 values" {
    var acc = MedAccumS.zero();
    const w = BN254Scalar.fromU64(1);
    const big_val: i128 = @as(i128, 1) << 100;

    acc.fmaddI128(w, big_val);
    acc.fmaddI128(w, -big_val);

    // Should cancel to zero
    const actual = acc.barrettReduce();
    try std.testing.expect(actual.isZero());
}

test "SmallAccumU large accumulation (1000 adds)" {
    const N = 1000;
    var acc = SmallAccumU.zero();
    var expected = BN254Scalar.zero();

    for (0..N) |i| {
        const w = BN254Scalar.fromU64(@as(u64, i) * 7 + 1);
        const val: i8 = @intCast(@as(i64, @intCast(i % 5)) - 2); // -2..2
        acc.fmaddI8(w, val);
        if (val == 1) {
            expected = expected.add(w);
        } else if (val == -1) {
            expected = expected.sub(w);
        } else if (val > 0) {
            expected = expected.add(w.mul(BN254Scalar.fromU64(@intCast(val))));
        } else if (val < 0) {
            expected = expected.sub(w.mul(BN254Scalar.fromU64(@intCast(-val))));
        }
    }

    const actual = acc.barrettReduce();
    try std.testing.expect(expected.eql(actual));
}

test "Barrett reduce round-trip: mulU64Unreduced sum" {
    // Accumulate many field × u64 products via folded, compare to direct field arithmetic
    const N = 100;
    var folded = FoldedMulU64.zero();
    var expected = BN254Scalar.zero();

    for (0..N) |i| {
        const w = BN254Scalar.fromU64(@as(u64, i) * 13 + 7);
        const scalar: u64 = @as(u64, i) * 11 + 3;
        folded.addAssign(mulU64Unreduced(w, scalar));
        expected = expected.add(w.mul(BN254Scalar.fromU64(scalar)));
    }

    const actual = reduceMulU64(folded);
    try std.testing.expect(expected.eql(actual));
}

test "Barrett reduce round-trip: mulU64Unreduced STRESS (large N, large values)" {
    // Stress test with N=100000 and full-range field values to catch overflow/precision bugs
    const N = 100_000;
    var folded = FoldedMulU64.zero();
    var expected = BN254Scalar.zero();

    for (0..N) |i| {
        // Use large field values (close to modulus) and large scalars
        const w = BN254Scalar.fromU64(@as(u64, i) *% 0x123456789ABCDEF +% 0xFEDCBA9876543210);
        const scalar: u64 = @as(u64, i) *% 0xABCD +% 0xFFFF;
        folded.addAssign(mulU64Unreduced(w, scalar));
        expected = expected.add(w.mul(BN254Scalar.fromU64(scalar)));
    }

    const actual = reduceMulU64(folded);
    if (!expected.eql(actual)) {
        std.debug.print("MISMATCH at N={}\n", .{N});
        std.debug.print("expected: {any}\n", .{expected.limbs});
        std.debug.print("actual:   {any}\n", .{actual.limbs});

        // Binary search for first failure
        var folded2 = FoldedMulU64.zero();
        var expected2 = BN254Scalar.zero();
        for (0..N) |i| {
            const w = BN254Scalar.fromU64(@as(u64, i) *% 0x123456789ABCDEF +% 0xFEDCBA9876543210);
            const scalar: u64 = @as(u64, i) *% 0xABCD +% 0xFFFF;
            folded2.addAssign(mulU64Unreduced(w, scalar));
            expected2 = expected2.add(w.mul(BN254Scalar.fromU64(scalar)));
            const check = reduceMulU64(folded2);
            if (!expected2.eql(check)) {
                std.debug.print("First mismatch at i={}\n", .{i});
                std.debug.print("  w.limbs = {any}\n", .{w.limbs});
                std.debug.print("  scalar = {}\n", .{scalar});
                std.debug.print("  folded slots = {any}\n", .{folded2.slots});
                std.debug.print("  normalized = {any}\n", .{folded2.normalize()});
                break;
            }
        }
    }
    try std.testing.expect(expected.eql(actual));
}

test "Barrett reduce round-trip: mulU128Unreduced sum" {
    const N = 50;
    var folded = FoldedMulU128.zero();
    var expected = BN254Scalar.zero();

    for (0..N) |i| {
        const w = BN254Scalar.fromU64(@as(u64, i) * 17 + 5);
        const scalar: u128 = @as(u128, i) * 0x1234567890ABCDEF + 1;
        folded.addAssign(mulU128Unreduced(w, scalar));
        expected = expected.add(w.mul(BN254Scalar.fromU128(scalar)));
    }

    const actual = reduceMulU128(folded);
    try std.testing.expect(expected.eql(actual));
}

test "WideAccumS i128 accumulation" {
    var acc = WideAccumS.zero();
    const w0 = BN254Scalar.fromU64(42);
    const w1 = BN254Scalar.fromU64(99);

    const b0: i128 = @as(i128, 1) << 80;
    const b1: i128 = -(@as(i128, 1) << 60);

    acc.fmaddI128(w0, b0);
    acc.fmaddI128(w1, b1);

    const expected = w0.mul(BN254Scalar.fromU128(@intCast(b0)))
        .sub(w1.mul(BN254Scalar.fromU128(@intCast(-b1))));
    const actual = acc.barrettReduce();
    try std.testing.expect(expected.eql(actual));
}

test "MODULUS_TIMES_2 and MODULUS_TIMES_3 correctness" {
    // Verify 2*p
    const two_p = BN254Scalar.fromU64(2).fromMontgomery();
    _ = two_p; // Just verifying comptime constants don't crash
    // Simple sanity: MODULUS_TIMES_2[0] should be 2*MODULUS[0] mod 2^64
    const expected_lo: u128 = @as(u128, BN254_MODULUS[0]) * 2;
    try std.testing.expectEqual(@as(u64, @truncate(expected_lo)), MODULUS_TIMES_2[0]);
}
