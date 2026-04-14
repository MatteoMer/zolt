//! Generic Montgomery field arithmetic over `[N]u64` limbs.
//!
//! `MontgomeryField(N, modulus, r2, n_prime)` returns a concrete type
//! whose elements are `struct { limbs: [N]u64 }` in Montgomery form.
//! The factory is comptime-generic over the limb count, so both 4-limb
//! BN254 and 6-limb BLS12-381 fields flow through the same code.
//!
//! The API surface matches the OG zolt-arith `BN254Scalar` / `BN254BaseField`
//! shapes (struct wrapper, `.limbs` access, method-call convention) so
//! existing callers in `field/`, `poly/`, `msm/`, etc. continue to work
//! when the BN254 types are aliased to instantiations of this factory.
//!
//! Aliases (e.g. `montMul` for `montgomeryMul`, `fromRaw` for
//! `toMontgomery`) let the BLS12-381 side use the same type under the
//! names established in the standalone `zolt-arith` package that zyli
//! consumed before the consolidation.

const std = @import("std");
const builtin = @import("builtin");
const bigint = @import("../bigint.zig");

// Import the N=4 asm backends from field/mod.zig (ARM64 + x86 helpers).
const asm_mod = @import("../field/mod.zig");

/// Comptime flag: x86-64 BMI2+ADX available for fast Montgomery mul.
const use_asm_mul = blk: {
    if (builtin.cpu.arch != .x86_64) break :blk false;
    const features = builtin.cpu.features;
    break :blk features.isEnabled(@intFromEnum(std.Target.x86.Feature.bmi2)) and
        features.isEnabled(@intFromEnum(std.Target.x86.Feature.adx));
};

/// Comptime flag: AArch64 (adds/adcs/subs/sbcs always available).
const use_arm64_asm = (builtin.cpu.arch == .aarch64);

/// LLVM x86 carry/borrow intrinsics — produce single adc/sbb instructions.
/// These work in Release builds (LLVM lowers them as intrinsics) but fail in
/// Debug (linker can't resolve the symbol). Guard all uses with !@inComptime().
// In Debug mode the linker can't resolve LLVM intrinsic symbols (they're
// only inlined by LLVM in optimized builds). So we only declare them when
// the module is compiled with optimizations. The bench build.zig creates
// a ReleaseFast dep chain for zolt-arith, so benches get the fast path.
const x86_has_intrinsics = builtin.cpu.arch == .x86_64 and builtin.mode != .Debug;
const x86_intrinsics = if (x86_has_intrinsics) struct {
    extern fn @"llvm.x86.addcarry.u64"(c_in: u8, a: u64, b: u64, result: *u64) u8;
    extern fn @"llvm.x86.subborrow.u64"(b_in: u8, a: u64, b: u64, result: *u64) u8;

    pub inline fn addcarry(c_in: u8, a: u64, b: u64, result: *u64) u8 {
        return @"llvm.x86.addcarry.u64"(c_in, a, b, result);
    }
    pub inline fn subborrow(b_in: u8, a: u64, b: u64, result: *u64) u8 {
        return @"llvm.x86.subborrow.u64"(b_in, a, b, result);
    }
} else struct {};

/// Build a Montgomery field type over the given modulus.
///
/// Parameters (all comptime):
///   - `N`: number of u64 limbs
///   - `modulus`: `[N]u64`, the prime `p` (little-endian)
///   - `r2`: `R^2 mod p` where `R = 2^(64·N)`
///   - `n_prime`: `-p^{-1} mod 2^64`
pub fn MontgomeryField(
    comptime N: comptime_int,
    comptime modulus: [N]u64,
    comptime r2: [N]u64,
    comptime n_prime: u64,
) type {
    return struct {
        limbs: [N]u64,

        const Self = @This();

        /// The numeric limb count, exported for byte-conversion logic.
        pub const LIMB_COUNT: comptime_int = N;
        pub const MODULUS: [N]u64 = modulus;
        pub const R2: [N]u64 = r2;
        pub const N_PRIME: u64 = n_prime;

        // -----------------------------------------------------------------
        // Constants
        // -----------------------------------------------------------------

        /// Zero element (additive identity).
        pub fn zero() Self {
            return .{ .limbs = .{0} ** N };
        }

        /// One element (multiplicative identity) in Montgomery form.
        /// Computed as `montMul(1, R^2)` = `1 * R^2 * R^{-1}` = `R mod p`.
        pub fn one() Self {
            var raw_one: [N]u64 = .{0} ** N;
            raw_one[0] = 1;
            return montgomeryMul(.{ .limbs = raw_one }, .{ .limbs = r2 });
        }

        // -----------------------------------------------------------------
        // Predicates
        // -----------------------------------------------------------------

        pub fn isZero(self: Self) bool {
            inline for (0..N) |i| {
                if (self.limbs[i] != 0) return false;
            }
            return true;
        }

        pub fn isOne(self: Self) bool {
            const o = one();
            return eql(self, o);
        }

        pub fn eql(a: Self, b: Self) bool {
            inline for (0..N) |i| {
                if (a.limbs[i] != b.limbs[i]) return false;
            }
            return true;
        }

        // -----------------------------------------------------------------
        // Constructors
        // -----------------------------------------------------------------

        /// Create from a small u64, converting to Montgomery form.
        pub fn fromU64(n: u64) Self {
            var raw: [N]u64 = .{0} ** N;
            raw[0] = n;
            return montgomeryMul(.{ .limbs = raw }, .{ .limbs = r2 });
        }

        /// Create from u128, converting to Montgomery form.
        pub fn fromU128(n: u128) Self {
            var raw: [N]u64 = .{0} ** N;
            raw[0] = @truncate(n);
            if (N > 1) raw[1] = @truncate(n >> 64);
            return montgomeryMul(.{ .limbs = raw }, .{ .limbs = r2 });
        }

        /// Create from little-endian bytes, converting to Montgomery form.
        pub fn fromBytes(bytes: []const u8) Self {
            var raw: [N]u64 = .{0} ** N;
            const len = @min(bytes.len, N * 8);
            var buf: [N * 8]u8 = .{0} ** (N * 8);
            @memcpy(buf[0..len], bytes[0..len]);
            inline for (0..N) |i| {
                raw[i] = std.mem.readInt(u64, buf[i * 8 ..][0..8], .little);
            }
            return montgomeryMul(.{ .limbs = raw }, .{ .limbs = r2 });
        }

        /// Create from big-endian bytes (32 / 48 / 64 byte pubkeys etc.)
        pub fn fromBytesBE(bytes: *const [N * 8]u8) Self {
            var le_bytes: [N * 8]u8 = undefined;
            for (0..N * 8) |i| {
                le_bytes[i] = bytes[N * 8 - 1 - i];
            }
            return fromBytes(&le_bytes);
        }

        /// Convert a raw little-endian limb array INTO Montgomery form.
        pub fn fromRaw(raw: [N]u64) Self {
            return mul(.{ .limbs = raw }, .{ .limbs = r2 });
        }

        /// Alias used by the OG callers.
        pub fn toMontgomery(self: Self) Self {
            return mul(self, .{ .limbs = r2 });
        }

        /// Convert FROM Montgomery form back to raw little-endian limbs.
        pub fn toRaw(self: Self) [N]u64 {
            var one_raw: [N]u64 = .{0} ** N;
            one_raw[0] = 1;
            return mul(self, .{ .limbs = one_raw }).limbs;
        }

        /// Alias used by the OG callers.
        pub fn fromMontgomery(self: Self) Self {
            var one_raw: [N]u64 = .{0} ** N;
            one_raw[0] = 1;
            return mul(self, .{ .limbs = one_raw });
        }

        /// Serialize to big-endian bytes, returning by value.
        /// This matches the OG `BN254Scalar.toBytesBE()` signature.
        pub fn toBytesBE(self: Self) [N * 8]u8 {
            var out: [N * 8]u8 = undefined;
            const raw = toRaw(self);
            bigint.toBytesBe(N, raw, &out);
            return out;
        }

        // -----------------------------------------------------------------
        // Utility helpers used by accumulators.zig and other low-level code
        // -----------------------------------------------------------------

        /// Wide multiply of two u64 words. Utility, not a field operation.
        pub inline fn mulWide(a: u64, b: u64) u128 {
            return @as(u128, a) * @as(u128, b);
        }

        /// Add with carry — public alias of the internal helper.
        pub const addCarry = addCarryFn;

        /// Subtract with borrow — public alias of the internal helper.
        pub const subBorrow = subBorrowFn;

        // -----------------------------------------------------------------
        // Arithmetic
        // -----------------------------------------------------------------

        /// Carry-chain add helper. Uses LLVM adc intrinsic on x86, u128 elsewhere.
        inline fn addCarryFn(aa: u64, bb: u64, carry_in: u64) struct { result: u64, carry: u64 } {
            if (!@inComptime() and comptime x86_has_intrinsics) {
                var result: u64 = undefined;
                const c = x86_intrinsics.addcarry(@truncate(carry_in), aa, bb, &result);
                return .{ .result = result, .carry = c };
            }
            const s = @as(u128, aa) + @as(u128, bb) + @as(u128, carry_in);
            return .{ .result = @truncate(s), .carry = @truncate(s >> 64) };
        }

        /// Borrow-chain sub helper. Uses LLVM sbb intrinsic on x86, u128 elsewhere.
        inline fn subBorrowFn(aa: u64, bb: u64, borrow_in: u64) struct { result: u64, borrow: u64 } {
            if (!@inComptime() and comptime x86_has_intrinsics) {
                var result: u64 = undefined;
                const b_out = x86_intrinsics.subborrow(@truncate(borrow_in), aa, bb, &result);
                return .{ .result = result, .borrow = b_out };
            }
            const wide_a = @as(u128, aa);
            const wide_b = @as(u128, bb) + @as(u128, borrow_in);
            const diff = wide_a -% wide_b;
            return .{ .result = @truncate(diff), .borrow = @truncate(diff >> 127) };
        }

        /// Modular addition `(a + b) mod p`.
        pub inline fn add(a: Self, b: Self) Self {
            @setEvalBranchQuota(10000);
            if (N == 4 and !@inComptime() and comptime use_arm64_asm) {
                var res = Self{ .limbs = asm_mod.arm64Add256(a.limbs, b.limbs) };
                if (!res.lessThanModulus()) res = res.subtractModulus();
                return res;
            }
            var result: [N]u64 = undefined;
            var carry: u64 = 0;
            inline for (0..N) |i| {
                const ac = addCarryFn(a.limbs[i], b.limbs[i], carry);
                result[i] = ac.result;
                carry = ac.carry;
            }
            var res = Self{ .limbs = result };
            if (carry != 0 or !res.lessThanModulus()) res = res.subtractModulus();
            return res;
        }

        /// Modular subtraction `(a - b) mod p`.
        pub inline fn sub(a: Self, b: Self) Self {
            @setEvalBranchQuota(10000);
            if (N == 4 and !@inComptime() and comptime use_arm64_asm)
                return .{ .limbs = asm_mod.arm64SubMod256(a.limbs, b.limbs, modulus) };
            var result: [N]u64 = undefined;
            var borrow: u64 = 0;
            inline for (0..N) |i| {
                const sb = subBorrowFn(a.limbs[i], b.limbs[i], borrow);
                result[i] = sb.result;
                borrow = sb.borrow;
            }
            if (borrow != 0) {
                // Add modulus back
                var c: u64 = 0;
                inline for (0..N) |i| {
                    const ac = addCarryFn(result[i], modulus[i], c);
                    result[i] = ac.result;
                    c = ac.carry;
                }
            }
            return .{ .limbs = result };
        }

        /// Modular negation `-a mod p`.
        pub inline fn neg(a: Self) Self {
            if (a.isZero()) return zero();
            return (Self{ .limbs = modulus }).sub(a);
        }

        /// Squaring with asm dispatch for N=4.
        pub inline fn square(a: Self) Self {
            if (N == 4 and !@inComptime()) {
                if (comptime use_arm64_asm) {
                    const mod_arr: [4]u64 = modulus;
                    var r = Self{ .limbs = asm_mod.arm64MontgomerySquare256(&a.limbs, &mod_arr, n_prime) };
                    if (!r.lessThanModulus()) r = r.subtractModulus();
                    return r;
                }
                // x86 BMI2+ADX square uses the mul path (mul(a,a) is fine
                // since the x86 mul is already fast; dedicated square asm
                // can be added later for ~15% additional gain).
                if (comptime use_asm_mul) return x86MontgomeryMul4(a, a);
            }
            return montgomeryMul(a, a);
        }

        /// Exponentiation `a^e mod p` by square-and-multiply.
        /// `e` is a raw little-endian limb array (NOT in Montgomery form).
        pub fn powLimbs(a: Self, exponent: [N]u64) Self {
            const top_bit = bigint.bitLen(N, exponent);
            if (top_bit == 0) return one();
            var result = a;
            var i = top_bit - 1;
            while (i > 0) {
                i -= 1;
                result = square(result);
                const limb = i / 64;
                const bit: u6 = @intCast(i % 64);
                if (((exponent[limb] >> bit) & 1) == 1) {
                    result = montgomeryMul(result, a);
                }
            }
            return result;
        }

        /// Modular inverse via Fermat's little theorem, returning null for zero.
        pub fn inverse(a: Self) ?Self {
            if (a.isZero()) return null;

            // Square-and-multiply: a^{p-2} mod p
            // Runtime loop to keep code compact (inline for unrolls N*64
            // iterations, blowing the icache for N >= 4).
            const p_minus_two: [N]u64 = comptime blk: {
                var exp: [N]u64 = modulus;
                exp[0] -= 2;
                break :blk exp;
            };

            var result = Self.one();
            var base = a;

            inline for (0..N) |i| {
                var bits = p_minus_two[i];
                var j: usize = 0;
                while (j < 64) : (j += 1) {
                    if ((bits & 1) != 0) {
                        result = result.mul(base);
                    }
                    base = base.square();
                    bits >>= 1;
                }
            }
            return result;
        }

        /// Non-optional inverse (returns zero for zero). Matches the
        /// standalone zolt-arith naming convention used by BLS12-381 code.
        pub fn inv(a: Self) Self {
            return a.inverse() orelse zero();
        }

        /// CIOS Montgomery multiplication: `a * b * R^{-1} mod p`.
        /// Uses [N+1]u64 accumulator matching the original factory.
        pub fn montgomeryMul(a: Self, b: Self) Self {
            var t: [N + 1]u64 = .{0} ** (N + 1);

            inline for (0..N) |i| {
                var carry: u64 = 0;
                inline for (0..N) |j| {
                    const prod = @as(u128, a.limbs[i]) * @as(u128, b.limbs[j]);
                    const s = @as(u128, t[j]) + prod + @as(u128, carry);
                    t[j] = @truncate(s);
                    carry = @truncate(s >> 64);
                }
                const sum_tn = @as(u128, t[N]) + @as(u128, carry);
                t[N] = @truncate(sum_tn);

                // Reduce: m = t[0] * n_prime mod 2^64
                const m = t[0] *% n_prime;
                carry = 0;
                {
                    const prod0 = @as(u128, m) * @as(u128, modulus[0]);
                    const s0 = @as(u128, t[0]) + prod0;
                    carry = @truncate(s0 >> 64);
                }
                inline for (1..N) |j| {
                    const prod = @as(u128, m) * @as(u128, modulus[j]);
                    const s = @as(u128, t[j]) + prod + @as(u128, carry);
                    t[j - 1] = @truncate(s);
                    carry = @truncate(s >> 64);
                }
                const final_sum = @as(u128, t[N]) + @as(u128, carry);
                t[N - 1] = @truncate(final_sum);
                t[N] = @truncate(final_sum >> 64);
            }

            var result = Self{ .limbs = undefined };
            inline for (0..N) |i| result.limbs[i] = t[i];
            if (t[N] != 0 or !result.lessThanModulus()) return result.subtractModulus();
            return result;
        }

        /// Alias matching the standalone's name.
        pub const montMul = montgomeryMul;

        /// R^2 as a field element (for compatibility with the OG
        /// BN254Scalar.rSquared). Returns `fromRaw(R2)`.
        pub fn rSquared() Self {
            return fromRaw(r2);
        }

        // -----------------------------------------------------------------
        // Aliases and JoltField-compatible surface
        // -----------------------------------------------------------------

        /// Field multiplication with asm dispatch for N=4.
        pub inline fn mul(a: Self, b: Self) Self {
            if (N == 4 and !@inComptime()) {
                if (comptime use_arm64_asm) {
                    const mod_arr: [4]u64 = modulus;
                    var r = Self{ .limbs = asm_mod.arm64MontgomeryMul256(&a.limbs, &b.limbs, &mod_arr, n_prime) };
                    if (!r.lessThanModulus()) r = r.subtractModulus();
                    return r;
                }
                // x86 BMI2+ADX: use the asm from the old MontgomeryField factory
                if (comptime use_asm_mul) return x86MontgomeryMul4(a, b);
            }
            return montgomeryMul(a, b);
        }

        /// x86 BMI2+ADX CIOS Montgomery mul (N=4 only).
        fn x86MontgomeryMul4(a_self: Self, b_other: Self) Self {
            const a = a_self.limbs;
            const b = b_other.limbs;
            const mod_arr: [4]u64 = modulus;
            var out0: u64 = undefined;
            var out1: u64 = undefined;
            var out2: u64 = undefined;
            var out3: u64 = undefined;
            asm volatile (
                \\xorq %%rcx, %%rcx
                \\movq (%%rdi), %%rdx
                \\mulxq (%%rsi), %%r8, %%r9
                \\mulxq 8(%%rsi), %%rax, %%r10
                \\adcxq %%rax, %%r9
                \\mulxq 16(%%rsi), %%rax, %%r11
                \\adcxq %%rax, %%r10
                \\mulxq 24(%%rsi), %%rax, %%rcx
                \\movq $0, %%r13
                \\adcxq %%rax, %%r11
                \\adcxq %%r13, %%rcx
                \\movq %%rbx, %%rdx
                \\mulxq %%r8, %%rdx, %%rax
                \\mulxq (%%r14), %%rax, %%r13
                \\adcxq %%r8, %%rax
                \\adoxq %%r13, %%r9
                \\mulxq 8(%%r14), %%rax, %%r13
                \\adcxq %%rax, %%r9
                \\adoxq %%r13, %%r10
                \\mulxq 16(%%r14), %%rax, %%r13
                \\adcxq %%rax, %%r10
                \\adoxq %%r13, %%r11
                \\mulxq 24(%%r14), %%rax, %%r8
                \\movq $0, %%r13
                \\adcxq %%rax, %%r11
                \\adoxq %%rcx, %%r8
                \\adcxq %%r13, %%r8
                \\movq 8(%%rdi), %%rdx
                \\mulxq (%%rsi), %%rax, %%r13
                \\adcxq %%rax, %%r9
                \\adoxq %%r13, %%r10
                \\mulxq 8(%%rsi), %%rax, %%r13
                \\adcxq %%rax, %%r10
                \\adoxq %%r13, %%r11
                \\mulxq 16(%%rsi), %%rax, %%r13
                \\adcxq %%rax, %%r11
                \\adoxq %%r13, %%r8
                \\mulxq 24(%%rsi), %%rax, %%rcx
                \\movq $0, %%r13
                \\adcxq %%rax, %%r8
                \\adoxq %%r13, %%rcx
                \\adcxq %%r13, %%rcx
                \\movq %%rbx, %%rdx
                \\mulxq %%r9, %%rdx, %%rax
                \\mulxq (%%r14), %%rax, %%r13
                \\adcxq %%r9, %%rax
                \\adoxq %%r13, %%r10
                \\mulxq 8(%%r14), %%rax, %%r13
                \\adcxq %%rax, %%r10
                \\adoxq %%r13, %%r11
                \\mulxq 16(%%r14), %%rax, %%r13
                \\adcxq %%rax, %%r11
                \\adoxq %%r13, %%r8
                \\mulxq 24(%%r14), %%rax, %%r9
                \\movq $0, %%r13
                \\adcxq %%rax, %%r8
                \\adoxq %%rcx, %%r9
                \\adcxq %%r13, %%r9
                \\movq 16(%%rdi), %%rdx
                \\mulxq (%%rsi), %%rax, %%r13
                \\adcxq %%rax, %%r10
                \\adoxq %%r13, %%r11
                \\mulxq 8(%%rsi), %%rax, %%r13
                \\adcxq %%rax, %%r11
                \\adoxq %%r13, %%r8
                \\mulxq 16(%%rsi), %%rax, %%r13
                \\adcxq %%rax, %%r8
                \\adoxq %%r13, %%r9
                \\mulxq 24(%%rsi), %%rax, %%rcx
                \\movq $0, %%r13
                \\adcxq %%rax, %%r9
                \\adoxq %%r13, %%rcx
                \\adcxq %%r13, %%rcx
                \\movq %%rbx, %%rdx
                \\mulxq %%r10, %%rdx, %%rax
                \\mulxq (%%r14), %%rax, %%r13
                \\adcxq %%r10, %%rax
                \\adoxq %%r13, %%r11
                \\mulxq 8(%%r14), %%rax, %%r13
                \\adcxq %%rax, %%r11
                \\adoxq %%r13, %%r8
                \\mulxq 16(%%r14), %%rax, %%r13
                \\adcxq %%rax, %%r8
                \\adoxq %%r13, %%r9
                \\mulxq 24(%%r14), %%rax, %%r10
                \\movq $0, %%r13
                \\adcxq %%rax, %%r9
                \\adoxq %%rcx, %%r10
                \\adcxq %%r13, %%r10
                \\movq 24(%%rdi), %%rdx
                \\mulxq (%%rsi), %%rax, %%r13
                \\adcxq %%rax, %%r11
                \\adoxq %%r13, %%r8
                \\mulxq 8(%%rsi), %%rax, %%r13
                \\adcxq %%rax, %%r8
                \\adoxq %%r13, %%r9
                \\mulxq 16(%%rsi), %%rax, %%r13
                \\adcxq %%rax, %%r9
                \\adoxq %%r13, %%r10
                \\mulxq 24(%%rsi), %%rax, %%rcx
                \\movq $0, %%r13
                \\adcxq %%rax, %%r10
                \\adoxq %%r13, %%rcx
                \\adcxq %%r13, %%rcx
                \\movq %%rbx, %%rdx
                \\mulxq %%r11, %%rdx, %%rax
                \\mulxq (%%r14), %%rax, %%r13
                \\adcxq %%r11, %%rax
                \\adoxq %%r13, %%r8
                \\mulxq 8(%%r14), %%rax, %%r13
                \\adcxq %%rax, %%r8
                \\adoxq %%r13, %%r9
                \\mulxq 16(%%r14), %%rax, %%r13
                \\adcxq %%rax, %%r9
                \\adoxq %%r13, %%r10
                \\mulxq 24(%%r14), %%rax, %%r11
                \\movq $0, %%r13
                \\adcxq %%rax, %%r10
                \\adoxq %%rcx, %%r11
                \\adcxq %%r13, %%r11
                : [_r0] "={r8}" (out0),
                  [_r1] "={r9}" (out1),
                  [_r2] "={r10}" (out2),
                  [_r3] "={r11}" (out3),
                : [_a] "{rdi}" (&a),
                  [_b] "{rsi}" (&b),
                  [_mod] "{r14}" (&mod_arr),
                  [_inv] "{rbx}" (n_prime),
                : .{ .rax = true, .rcx = true, .rdx = true, .r13 = true, .cc = true, .memory = true });
            var result = Self{ .limbs = .{ out0, out1, out2, out3 } };
            if (!result.lessThanModulus()) result = result.subtractModulus();
            return result;
        }

        /// Doubling (2*self).
        pub inline fn double(self: Self) Self {
            return self.add(self);
        }

        /// Serialize to little-endian bytes, returning by value.
        pub fn toBytes(self: Self) [N * 8]u8 {
            var out: [N * 8]u8 = undefined;
            const raw = toRaw(self);
            bigint.toBytesLe(N, raw, &out);
            return out;
        }

        /// Extract the low u64 from the raw (non-Montgomery) form.
        pub fn toU64(self: Self) u64 {
            return toRaw(self)[0];
        }

        /// Field element byte count.
        pub const FIELD_ELEMENT_BYTES: usize = N * 8;

        // -----------------------------------------------------------------
        // Helpers for reduction / unreduced paths (used by OG callers)
        // -----------------------------------------------------------------

        /// Check if the limbs are strictly less than the modulus (MSB-first).
        /// Unrolled comparison matching the original MontgomeryField factory.
        inline fn lessThanModulus(self: Self) bool {
            // Unrolled MSB-first comparison — const modulus enables LLVM to
            // emit cmpq with immediates.
            comptime var idx: usize = N;
            inline while (idx > 0) {
                idx -= 1;
                if (self.limbs[idx] != modulus[idx]) return self.limbs[idx] < modulus[idx];
            }
            return self.limbs[0] < modulus[0];
        }

        /// Unconditional subtraction of the modulus.
        pub inline fn subtractModulus(self: Self) Self {
            @setEvalBranchQuota(10000);
            if (N == 4 and !@inComptime() and comptime use_arm64_asm)
                return .{ .limbs = asm_mod.arm64Sub256(self.limbs, modulus) };
            var result: [N]u64 = undefined;
            var borrow: u64 = 0;
            inline for (0..N) |i| {
                const sb = subBorrowFn(self.limbs[i], modulus[i], borrow);
                result[i] = sb.result;
                borrow = sb.borrow;
            }
            return .{ .limbs = result };
        }

        /// Addition without final reduction. Result in [0, 2p).
        pub inline fn addNoReduce(self: Self, other: Self) Self {
            @setEvalBranchQuota(10000);
            var result: [N]u64 = undefined;
            var carry: u64 = 0;
            inline for (0..N) |i| {
                const ac = addCarryFn(self.limbs[i], other.limbs[i], carry);
                result[i] = ac.result;
                carry = ac.carry;
            }
            var res = Self{ .limbs = result };
            if (carry != 0) res = res.subtractModulus();
            return res;
        }

        /// Reduce from [0, 2p) to [0, p).
        pub inline fn reduce(self: Self) Self {
            if (!self.lessThanModulus()) return self.subtractModulus();
            return self;
        }

        // (inverseOpt removed — `inverse()` itself returns `?Self` now)

        // -----------------------------------------------------------------
        // Extended arithmetic (used by zolt's polynomial / MSM / accumulator code)
        // -----------------------------------------------------------------

        /// Fused multiply-accumulate: a[0]*b[0] + a[1]*b[1] with 2 reductions
        /// instead of 3 (vs separate mul + mul + add). Interleaved CIOS.
        pub inline fn sumOfProducts(a_pair: [2]Self, b_pair: [2]Self) Self {
            if (N == 4 and !@inComptime() and comptime use_arm64_asm) {
                const mod_arr: [4]u64 = modulus;
                return .{ .limbs = asm_mod.arm64SumOfProducts256(
                    &a_pair[0].limbs, &a_pair[1].limbs,
                    &b_pair[0].limbs, &b_pair[1].limbs,
                    &mod_arr, n_prime,
                ) };
            }
            var t: [N + 1]u64 = .{0} ** (N + 1);

            inline for (0..N) |i| {
                var carry1: u64 = 0;
                inline for (0..2) |pair| {
                    var carry: u64 = 0;
                    inline for (0..N) |j| {
                        const prod = @as(u128, a_pair[pair].limbs[i]) * @as(u128, b_pair[pair].limbs[j]);
                        const s = @as(u128, t[j]) + prod + @as(u128, carry);
                        t[j] = @truncate(s);
                        carry = @truncate(s >> 64);
                    }
                    const s_tn = @as(u128, t[N]) + @as(u128, carry) + @as(u128, carry1);
                    t[N] = @truncate(s_tn);
                    carry1 = @truncate(s_tn >> 64);
                }

                // Montgomery reduction step (shared)
                const m = t[0] *% n_prime;
                var carry: u64 = 0;
                {
                    const prod0 = @as(u128, m) * @as(u128, modulus[0]) + @as(u128, t[0]);
                    carry = @truncate(prod0 >> 64);
                }
                inline for (1..N) |j| {
                    const prod = @as(u128, m) * @as(u128, modulus[j]) + @as(u128, t[j]) + @as(u128, carry);
                    t[j - 1] = @truncate(prod);
                    carry = @truncate(prod >> 64);
                }
                const final_sum = @as(u128, t[N]) + @as(u128, carry);
                t[N - 1] = @truncate(final_sum);
                t[N] = @as(u64, @truncate(final_sum >> 64)) +% carry1;
            }

            var result: Self = undefined;
            inline for (0..N) |i| result.limbs[i] = t[i];
            if (t[N] != 0 or !result.lessThanModulus()) result = result.subtractModulus();
            return result;
        }

        /// Multiply by a signed 128-bit integer.
        pub fn mulI128(self: Self, val: i128) Self {
            if (val == 0) return Self.zero();
            if (val == 1) return self;
            if (val == -1) return self.neg();
            if (val > 0) {
                return self.mulU128(@intCast(val));
            } else {
                return self.mulU128(@intCast(-val)).neg();
            }
        }

        fn mulU128(self: Self, val: u128) Self {
            if (val == 0) return Self.zero();
            if (val == 1) return self;
            const low: u64 = @truncate(val);
            const high: u64 = @truncate(val >> 64);
            var other = Self.fromU64(low);
            if (high != 0) {
                const high_fe = Self.fromU64(high);
                var two_64 = Self.fromU64(1);
                for (0..64) |_| two_64 = two_64.double();
                other = other.add(high_fe.mul(two_64));
            }
            return self.mul(other);
        }

        /// Exponentiation with a u64 exponent.
        pub fn pow(self: Self, exp: u64) Self {
            if (exp == 0) return one();
            var result = Self.one();
            var base = self;
            var e = exp;
            while (e > 0) {
                if (e & 1 == 1) result = result.mul(base);
                base = base.square();
                e >>= 1;
            }
            return result;
        }

        /// Batch inversion using Montgomery's trick.
        /// Inverts n elements in-place with 1 inversion + 3(n-1) muls.
        pub fn batchInversion(elements: []Self, scratch: []Self) void {
            const n = elements.len;
            if (n == 0) return;
            var acc = one();
            for (0..n) |i| {
                scratch[i] = acc;
                if (!elements[i].isZero()) acc = acc.mul(elements[i]);
            }
            var inv_acc = acc.inverse() orelse unreachable;
            var i: usize = n;
            while (i > 0) {
                i -= 1;
                if (elements[i].isZero()) continue;
                const old = elements[i];
                elements[i] = scratch[i].mul(inv_acc);
                inv_acc = inv_acc.mul(old);
            }
        }

        /// Optimized multiplication by a 128-bit value in the top 2 limbs
        /// of a [N]u64 array. Only the limbs at indices [N-2] and [N-1] are
        /// non-zero. Uses 2 iterations of CIOS instead of N.
        pub fn mulHiBigIntU128(self: Self, hi_limbs: [N]u64) Self {
            // Process only the top 2 non-zero limbs
            const limb_n2 = hi_limbs[N - 2];
            const limb_n1 = hi_limbs[N - 1];

            var r: [N]u64 = .{0} ** N;

            // Iteration for limb at index N-2
            inline for ([_]u64{ limb_n2, limb_n1 }) |limb_val| {
                var carry1: u64 = 0;
                {
                    const prod0 = mulWide(self.limbs[0], limb_val);
                    const sum0 = @as(u128, r[0]) + prod0 + @as(u128, carry1);
                    r[0] = @truncate(sum0);
                    carry1 = @truncate(sum0 >> 64);
                }
                const k = r[0] *% n_prime;
                var carry2: u64 = 0;
                {
                    const red0 = mulWide(k, modulus[0]);
                    const red_sum0 = @as(u128, r[0]) + red0;
                    carry2 = @truncate(red_sum0 >> 64);
                }
                inline for (1..N) |j| {
                    const prod_j = mulWide(self.limbs[j], limb_val);
                    const new_rj = @as(u128, r[j]) + prod_j + @as(u128, carry1);
                    const new_rj_trunc: u64 = @truncate(new_rj);
                    carry1 = @truncate(new_rj >> 64);

                    const red_j = mulWide(k, modulus[j]);
                    const red_sum_j = @as(u128, new_rj_trunc) + red_j + @as(u128, carry2);
                    r[j - 1] = @truncate(red_sum_j);
                    carry2 = @truncate(red_sum_j >> 64);

                    r[j] = new_rj_trunc;
                }
                r[N - 1] = carry1 +% carry2;
            }

            var result = Self{ .limbs = r };
            if (!result.lessThanModulus()) result = result.subtractModulus();
            return result;
        }

        /// Unreduced product accumulator — defers Montgomery reduction
        /// across multiple multiply-accumulate steps. The generic version
        /// uses `[2*N]u128` slots matching the OG `UnreducedProductAccum`.
        pub const ProductAccum = struct {
            slots: [2 * N]u128,

            pub inline fn zero() @This() {
                return .{ .slots = .{0} ** (2 * N) };
            }

            /// Schoolbook N×N without reduction.
            pub inline fn fromMul(a: Self, b: Self) @This() {
                @setEvalBranchQuota(10000);
                var slots: [2 * N]u128 = .{0} ** (2 * N);
                inline for (0..N) |i| {
                    inline for (0..N) |j| {
                        const p: u128 = @as(u128, a.limbs[i]) * @as(u128, b.limbs[j]);
                        slots[i + j] += @as(u128, @as(u64, @truncate(p)));
                        slots[i + j + 1] += @as(u128, @as(u64, @truncate(p >> 64)));
                    }
                }
                return .{ .slots = slots };
            }

            /// Schoolbook N×2 (field_elem × raw u128) without reduction.
            pub inline fn fromMulU128(field_elem: Self, raw: u128) @This() {
                @setEvalBranchQuota(10000);
                const a = field_elem.limbs;
                const b: [2]u64 = .{ @truncate(raw), @truncate(raw >> 64) };
                var slots: [2 * N]u128 = .{0} ** (2 * N);
                inline for (0..N) |i| {
                    inline for (0..2) |j| {
                        const p: u128 = @as(u128, a[i]) * @as(u128, b[j]);
                        slots[i + j] += @as(u128, @as(u64, @truncate(p)));
                        slots[i + j + 1] += @as(u128, @as(u64, @truncate(p >> 64)));
                    }
                }
                return .{ .slots = slots };
            }

            pub inline fn addAssign(self: *@This(), other: @This()) void {
                inline for (0..(2 * N)) |i| {
                    self.slots[i] += other.slots[i];
                }
            }

            pub inline fn add(self: @This(), other: @This()) @This() {
                var result = self;
                inline for (0..(2 * N)) |i| {
                    result.slots[i] += other.slots[i];
                }
                return result;
            }

            /// Reduce accumulated products to a field element via Montgomery
            /// reduction. Handles overflow from heavy accumulation (up to
            /// ~2^62 products).
            pub fn reduce(self: @This()) Self {
                // Step 1: Normalize [2N]u128 → [2N]u64 + overflow
                var limbs_wide: [2 * N]u64 = undefined;
                var carry: u128 = 0;
                inline for (0..(2 * N)) |i| {
                    const s = self.slots[i] + carry;
                    limbs_wide[i] = @truncate(s);
                    carry = s >> 64;
                }
                const overflow_lo: u64 = @truncate(carry);
                const overflow_hi: u64 = @truncate(carry >> 64);

                // Step 2: CIOS Montgomery reduction, folding upper limbs
                var t: [N + 1]u64 = undefined;
                inline for (0..N) |i| t[i] = limbs_wide[i];
                t[N] = 0;

                inline for (0..N) |i| {
                    const m = t[0] *% n_prime;
                    var c: u64 = 0;
                    {
                        const prod0 = @as(u128, m) * @as(u128, modulus[0]) + @as(u128, t[0]);
                        c = @truncate(prod0 >> 64);
                    }
                    inline for (1..N) |j| {
                        const prod = @as(u128, m) * @as(u128, modulus[j]) + @as(u128, t[j]) + @as(u128, c);
                        t[j - 1] = @truncate(prod);
                        c = @truncate(prod >> 64);
                    }
                    const final_sum = @as(u128, t[N]) + @as(u128, c) + @as(u128, limbs_wide[i + N]);
                    t[N - 1] = @truncate(final_sum);
                    t[N] = @truncate(final_sum >> 64);
                }

                // Step 3: Multi-subtract for accumulated overflow
                var result = Self{ .limbs = undefined };
                inline for (0..N) |i| result.limbs[i] = t[i];
                var extra = t[N];
                var iters: u32 = 0;
                while (extra != 0 or !result.lessThanModulus()) : (iters += 1) {
                    std.debug.assert(iters < N + 2);
                    const was_less = result.lessThanModulus();
                    result = result.subtractModulus();
                    if (was_less) extra -= 1;
                }

                // Step 4: Add overflow contribution
                if (overflow_lo != 0 or overflow_hi != 0) {
                    var raw: [N]u64 = .{0} ** N;
                    raw[0] = overflow_lo;
                    if (N > 1) raw[1] = overflow_hi;
                    result = Self.add(result, Self.toMontgomery(.{ .limbs = raw }));
                }

                return result;
            }
        };

        /// Create an unreduced product accumulator from `self * other`.
        pub inline fn mulToProductAccum(self: Self, other: Self) ProductAccum {
            return ProductAccum.fromMul(self, other);
        }

        /// Debug formatter for field elements.
        pub fn format(self: Self, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
            _ = fmt;
            _ = options;
            const raw = toRaw(self);
            try writer.writeAll("0x");
            var i: usize = N;
            while (i > 0) {
                i -= 1;
                try std.fmt.formatInt(raw[i], 16, .lower, .{ .width = 16, .fill = '0' }, writer);
            }
        }
    };
}


// =========================================================================
// Tests
// =========================================================================

const testing = std.testing;

// Toy 4-limb prime: p = 2^255 - 19 (Curve25519's base field).
const ED25519_P: [4]u64 = .{
    0xffffffffffffffed, 0xffffffffffffffff,
    0xffffffffffffffff, 0x7fffffffffffffff,
};
const ED25519_R2: [4]u64 = .{ 0x5a4, 0, 0, 0 };
const ED25519_N_PRIME: u64 = 0x86bca1af286bca1b;
const Ed25519Fp = MontgomeryField(4, ED25519_P, ED25519_R2, ED25519_N_PRIME);

// BLS12-381 Fp (6-limb) constants for cross-checking the N=6 path.
const BLS12_381_FP_MODULUS: [6]u64 = .{
    0xb9feffffffffaaab, 0x1eabfffeb153ffff,
    0x6730d2a0f6b0f624, 0x64774b84f38512bf,
    0x4b1ba7b6434bacd7, 0x1a0111ea397fe69a,
};
const BLS12_381_FP_R2: [6]u64 = .{
    0xf4df1f341c341746, 0x0a76e6a609d104f1,
    0x8de5476c4c95b6d5, 0x67eb88a9939d83c0,
    0x9a793e85b519952d, 0x11988fe592cae3aa,
};
const BLS12_381_FP_N_PRIME: u64 = 0x89f3fffcfffcfffd;
const Bls12381Fp = MontgomeryField(6, BLS12_381_FP_MODULUS, BLS12_381_FP_R2, BLS12_381_FP_N_PRIME);

// -- N=4 tests (Ed25519 base field) --

test "N=4 zero is additive identity" {
    const z = Ed25519Fp.zero();
    const o = Ed25519Fp.one();
    try testing.expect(Ed25519Fp.eql(Ed25519Fp.add(z, z), z));
    try testing.expect(Ed25519Fp.eql(Ed25519Fp.add(o, z), o));
    try testing.expect(Ed25519Fp.eql(Ed25519Fp.add(z, o), o));
}

test "N=4 one is multiplicative identity" {
    const o = Ed25519Fp.one();
    const a = Ed25519Fp.fromRaw(.{ 0x1234567890abcdef, 0, 0, 0 });
    try testing.expect(Ed25519Fp.eql(Ed25519Fp.montgomeryMul(a, o), a));
    try testing.expect(Ed25519Fp.eql(Ed25519Fp.montgomeryMul(o, a), a));
}

test "N=4 toRaw round-trips fromRaw" {
    const raw: [4]u64 = .{ 0x0102030405060708, 0x0a0b0c0d0e0f0001, 0x1122334455667788, 0x12345678 };
    const m = Ed25519Fp.fromRaw(raw);
    try testing.expectEqual(raw, Ed25519Fp.toRaw(m));
}

test "N=4 add wraps around modulus" {
    const o = Ed25519Fp.fromRaw(.{ 1, 0, 0, 0 });
    const pm1 = Ed25519Fp.fromRaw(.{ ED25519_P[0] - 1, ED25519_P[1], ED25519_P[2], ED25519_P[3] });
    try testing.expect(Ed25519Fp.eql(Ed25519Fp.add(o, pm1), Ed25519Fp.zero()));
}

test "N=4 sub borrows correctly" {
    const z = Ed25519Fp.zero();
    const o = Ed25519Fp.fromRaw(.{ 1, 0, 0, 0 });
    const result = Ed25519Fp.sub(z, o);
    const expected = Ed25519Fp.fromRaw(.{ ED25519_P[0] - 1, ED25519_P[1], ED25519_P[2], ED25519_P[3] });
    try testing.expect(Ed25519Fp.eql(result, expected));
}

test "N=4 mul: 2 * 3 = 6" {
    const two = Ed25519Fp.fromRaw(.{ 2, 0, 0, 0 });
    const three = Ed25519Fp.fromRaw(.{ 3, 0, 0, 0 });
    const six = Ed25519Fp.fromRaw(.{ 6, 0, 0, 0 });
    try testing.expect(Ed25519Fp.eql(Ed25519Fp.montgomeryMul(two, three), six));
}

test "N=4 inv: a * a^-1 = 1" {
    const a = Ed25519Fp.fromRaw(.{ 0x12345678, 0, 0, 0 });
    const inv_a = Ed25519Fp.inverse(a) orelse return error.SkipZigTest;
    try testing.expect(Ed25519Fp.eql(Ed25519Fp.montgomeryMul(a, inv_a), Ed25519Fp.one()));
}

test "N=4 .limbs access" {
    const o = Ed25519Fp.one();
    // The struct wrapper provides `.limbs` for backward compat.
    try testing.expect(o.limbs[0] != 0 or o.limbs[1] != 0 or o.limbs[2] != 0 or o.limbs[3] != 0);
}

// -- N=6 tests (BLS12-381 base field) --

test "N=6 zero is additive identity" {
    const z = Bls12381Fp.zero();
    const o = Bls12381Fp.one();
    try testing.expect(Bls12381Fp.eql(Bls12381Fp.add(z, z), z));
    try testing.expect(Bls12381Fp.eql(Bls12381Fp.add(o, z), o));
}

test "N=6 one is multiplicative identity" {
    const o = Bls12381Fp.one();
    const a = Bls12381Fp.fromRaw(.{ 42, 0, 0, 0, 0, 0 });
    try testing.expect(Bls12381Fp.eql(Bls12381Fp.montgomeryMul(a, o), a));
}

test "N=6 toRaw round-trips fromRaw" {
    const raw: [6]u64 = .{ 0x0102030405060708, 0x0a0b0c0d0e0f0001, 0x1122334455667788, 0x12345678, 0xabcd, 0 };
    const m = Bls12381Fp.fromRaw(raw);
    try testing.expectEqual(raw, Bls12381Fp.toRaw(m));
}

test "N=6 add wraps around modulus" {
    const o = Bls12381Fp.fromRaw(.{ 1, 0, 0, 0, 0, 0 });
    const pm1 = Bls12381Fp.fromRaw(.{
        BLS12_381_FP_MODULUS[0] - 1,
        BLS12_381_FP_MODULUS[1],
        BLS12_381_FP_MODULUS[2],
        BLS12_381_FP_MODULUS[3],
        BLS12_381_FP_MODULUS[4],
        BLS12_381_FP_MODULUS[5],
    });
    try testing.expect(Bls12381Fp.eql(Bls12381Fp.add(o, pm1), Bls12381Fp.zero()));
}

test "N=6 mul: 2 * 3 = 6" {
    const two = Bls12381Fp.fromRaw(.{ 2, 0, 0, 0, 0, 0 });
    const three = Bls12381Fp.fromRaw(.{ 3, 0, 0, 0, 0, 0 });
    const six = Bls12381Fp.fromRaw(.{ 6, 0, 0, 0, 0, 0 });
    try testing.expect(Bls12381Fp.eql(Bls12381Fp.montgomeryMul(two, three), six));
}

test "N=6 inv: a * a^-1 = 1" {
    const a = Bls12381Fp.fromRaw(.{ 0x42, 0, 0, 0, 0, 0 });
    const inv_a = Bls12381Fp.inverse(a) orelse return error.SkipZigTest;
    try testing.expect(Bls12381Fp.eql(Bls12381Fp.montgomeryMul(a, inv_a), Bls12381Fp.one()));
}

test "N=6 montMul alias works" {
    const two = Bls12381Fp.fromRaw(.{ 2, 0, 0, 0, 0, 0 });
    const three = Bls12381Fp.fromRaw(.{ 3, 0, 0, 0, 0, 0 });
    const via_mont = Bls12381Fp.montMul(two, three);
    const via_full = Bls12381Fp.montgomeryMul(two, three);
    try testing.expect(Bls12381Fp.eql(via_mont, via_full));
}

// -- Cross-N test: distributive property --

test "N=4 distributive: (a+b)*c == a*c + b*c" {
    const a = Ed25519Fp.fromRaw(.{ 0x1234, 0x5678, 0, 0 });
    const b = Ed25519Fp.fromRaw(.{ 0x9abc, 0xdef0, 0x1234, 0 });
    const c = Ed25519Fp.fromRaw(.{ 0x42, 0, 0, 0 });
    const lhs = Ed25519Fp.montgomeryMul(Ed25519Fp.add(a, b), c);
    const rhs = Ed25519Fp.add(Ed25519Fp.montgomeryMul(a, c), Ed25519Fp.montgomeryMul(b, c));
    try testing.expect(Ed25519Fp.eql(lhs, rhs));
}

test "N=6 distributive: (a+b)*c == a*c + b*c" {
    const a = Bls12381Fp.fromRaw(.{ 0x1234, 0x5678, 0, 0, 0, 0 });
    const b = Bls12381Fp.fromRaw(.{ 0x9abc, 0xdef0, 0x1234, 0, 0, 0 });
    const c = Bls12381Fp.fromRaw(.{ 0x42, 0, 0, 0, 0, 0 });
    const lhs = Bls12381Fp.montgomeryMul(Bls12381Fp.add(a, b), c);
    const rhs = Bls12381Fp.add(Bls12381Fp.montgomeryMul(a, c), Bls12381Fp.montgomeryMul(b, c));
    try testing.expect(Bls12381Fp.eql(lhs, rhs));
}
