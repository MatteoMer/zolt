#ifndef FP_COMMON_H
#define FP_COMMON_H

#include <metal_stdlib>
using namespace metal;

// ── BN254 base field Fp: 8×32-bit limbs, Montgomery form ────────────────────
//
// This is the BASE FIELD (point coordinates), NOT the scalar field.
// Same CIOS algorithm as field_common.h but with different modulus/constants.
//
// q = 21888242871839275222246405745257275088696311157297823662689037894645226208583

struct FpBase {
    uint32_t limbs[8];
};

// BN254 base field modulus q (little-endian 8×32-bit)
constant uint32_t FP_MODULUS[8] = {
    0xd87cfd47, 0x3c208c16, 0x6871ca8d, 0x97816a91,
    0x8181585d, 0xb85045b6, 0xe131a029, 0x30644e72
};

// Montgomery R = 2^256 mod q
constant uint32_t FP_MONT_R[8] = {
    0xc58f0d9d, 0xd35d438d, 0xf5c70b3d, 0x0a78eb28,
    0x7879462c, 0x666ea36f, 0x9a07df2f, 0x0e0a77c1
};

// Montgomery R² = 2^512 mod q
constant uint32_t FP_MONT_R2[8] = {
    0x538afa89, 0xf32cfc5b, 0xd44501fb, 0xb5e71911,
    0x0a417ff6, 0x47ab1eff, 0xcab8351f, 0x06d89f71
};

// Montgomery constant: -q^{-1} mod 2^32
constant uint32_t FP_MONT_INV = 0xe4866389;

// ── Helpers ──────────────────────────────────────────────────────────────────

static FpBase fpb_zero() {
    FpBase z;
    for (int i = 0; i < 8; i++) z.limbs[i] = 0;
    return z;
}

static FpBase fpb_one() {
    FpBase one;
    for (int i = 0; i < 8; i++) one.limbs[i] = FP_MONT_R[i];
    return one;
}

static bool fpb_is_zero(FpBase a) {
    for (int i = 0; i < 8; i++) {
        if (a.limbs[i] != 0) return false;
    }
    return true;
}

static bool fpb_eq(FpBase a, FpBase b) {
    for (int i = 0; i < 8; i++) {
        if (a.limbs[i] != b.limbs[i]) return false;
    }
    return true;
}

static bool fpb_gte_modulus(FpBase a) {
    for (int i = 7; i >= 0; i--) {
        if (a.limbs[i] > FP_MODULUS[i]) return true;
        if (a.limbs[i] < FP_MODULUS[i]) return false;
    }
    return true;
}

static FpBase fpb_sub_modulus(FpBase a) {
    uint32_t borrow = 0;
    for (int i = 0; i < 8; i++) {
        uint64_t diff = (uint64_t)a.limbs[i] - (uint64_t)FP_MODULUS[i] - (uint64_t)borrow;
        a.limbs[i] = (uint32_t)diff;
        borrow = (uint32_t)(diff >> 63) & 1;
    }
    return a;
}

static FpBase fpb_add_modulus(FpBase a) {
    uint32_t carry = 0;
    for (int i = 0; i < 8; i++) {
        uint64_t sum = (uint64_t)a.limbs[i] + (uint64_t)FP_MODULUS[i] + (uint64_t)carry;
        a.limbs[i] = (uint32_t)sum;
        carry = (uint32_t)(sum >> 32);
    }
    return a;
}

// ── fpb_mul: CIOS Montgomery multiplication (8×32-bit) ──────────────────────

static FpBase fpb_mul(FpBase a, FpBase b) {
    uint64_t t[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};

    for (int i = 0; i < 8; i++) {
        uint64_t carry = 0;
        for (int j = 0; j < 8; j++) {
            uint64_t prod = (uint64_t)a.limbs[j] * (uint64_t)b.limbs[i] + t[j] + carry;
            t[j] = prod & 0xFFFFFFFF;
            carry = prod >> 32;
        }
        t[8] += carry;

        uint32_t m = (uint32_t)t[0] * FP_MONT_INV;
        uint64_t prod = (uint64_t)m * (uint64_t)FP_MODULUS[0] + t[0];
        carry = prod >> 32;
        for (int j = 1; j < 8; j++) {
            prod = (uint64_t)m * (uint64_t)FP_MODULUS[j] + t[j] + carry;
            t[j - 1] = prod & 0xFFFFFFFF;
            carry = prod >> 32;
        }
        uint64_t sum = t[8] + carry;
        t[7] = sum & 0xFFFFFFFF;
        t[8] = sum >> 32;
    }

    FpBase result;
    for (int i = 0; i < 8; i++) result.limbs[i] = (uint32_t)t[i];
    if (t[8] != 0 || fpb_gte_modulus(result)) {
        result = fpb_sub_modulus(result);
    }
    return result;
}

// ── fpb_square: Montgomery squaring (reuses mul for now) ────────────────────

static FpBase fpb_square(FpBase a) {
    return fpb_mul(a, a);
}

// ── fpb_add: modular addition ───────────────────────────────────────────────

static FpBase fpb_add(FpBase a, FpBase b) {
    FpBase result;
    uint32_t carry = 0;
    for (int i = 0; i < 8; i++) {
        uint64_t sum = (uint64_t)a.limbs[i] + (uint64_t)b.limbs[i] + (uint64_t)carry;
        result.limbs[i] = (uint32_t)sum;
        carry = (uint32_t)(sum >> 32);
    }
    if (carry != 0 || fpb_gte_modulus(result)) {
        result = fpb_sub_modulus(result);
    }
    return result;
}

// ── fpb_sub: modular subtraction ────────────────────────────────────────────

static FpBase fpb_sub(FpBase a, FpBase b) {
    FpBase result;
    uint32_t borrow = 0;
    for (int i = 0; i < 8; i++) {
        uint64_t diff = (uint64_t)a.limbs[i] - (uint64_t)b.limbs[i] - (uint64_t)borrow;
        result.limbs[i] = (uint32_t)diff;
        borrow = (uint32_t)(diff >> 63) & 1;
    }
    if (borrow != 0) {
        result = fpb_add_modulus(result);
    }
    return result;
}

// ── fpb_neg: modular negation ───────────────────────────────────────────────

static FpBase fpb_neg(FpBase a) {
    if (fpb_is_zero(a)) return a;
    FpBase result;
    uint32_t borrow = 0;
    for (int i = 0; i < 8; i++) {
        uint64_t diff = (uint64_t)FP_MODULUS[i] - (uint64_t)a.limbs[i] - (uint64_t)borrow;
        result.limbs[i] = (uint32_t)diff;
        borrow = (uint32_t)(diff >> 63) & 1;
    }
    return result;
}

// ── fpb_double: a + a ───────────────────────────────────────────────────────

static FpBase fpb_double(FpBase a) {
    return fpb_add(a, a);
}

#endif // FP_COMMON_H
