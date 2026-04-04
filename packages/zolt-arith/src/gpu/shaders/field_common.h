#ifndef FIELD_COMMON_H
#define FIELD_COMMON_H

#include <metal_stdlib>
using namespace metal;

// ── BN254 scalar field: 8×32-bit limbs, Montgomery form ─────────────────────
//
// All elements are stored as 8 little-endian uint32_t limbs in Montgomery
// representation: a is stored as a*R mod p, where R = 2^256.
//
// The CIOS algorithm uses 64-bit intermediates for carry propagation.
// Apple Silicon GPUs support native uint64_t since MSL 2.2.

struct Fp256 {
    uint32_t limbs[8];
};

// BN254 scalar field modulus p (little-endian 8×32-bit)
constant uint32_t MODULUS[8] = {
    0xf0000001, 0x43e1f593, 0x79b97091, 0x2833e848,
    0x8181585d, 0xb85045b6, 0xe131a029, 0x30644e72
};

// Montgomery R² = 2^512 mod p (for converting to Montgomery form)
constant uint32_t MONT_R2[8] = {
    0xae216da7, 0x1bb8e645, 0xe35c59e3, 0x53fe3ab1,
    0x53bb8085, 0x8c49833d, 0x7f4e44a5, 0x0216d0b1
};

// Montgomery constant: -p^{-1} mod 2^32
constant uint32_t MONT_INV = 0xefffffff;

// ── Helpers ──────────────────────────────────────────────────────────────────

static Fp256 fp_zero() {
    Fp256 z;
    for (int i = 0; i < 8; i++) z.limbs[i] = 0;
    return z;
}

// Montgomery R = 2^256 mod p (the multiplicative identity in Montgomery form)
constant uint32_t MONT_R[8] = {
    0x4ffffffb, 0xac96341c, 0x9f60cd29, 0x36fc7695,
    0x7879462e, 0x666ea36f, 0x9a07df2f, 0x0e0a77c1
};

static Fp256 fp_one() {
    Fp256 one;
    for (int i = 0; i < 8; i++) one.limbs[i] = MONT_R[i];
    return one;
}

static bool fp_gte_modulus(Fp256 a) {
    for (int i = 7; i >= 0; i--) {
        if (a.limbs[i] > MODULUS[i]) return true;
        if (a.limbs[i] < MODULUS[i]) return false;
    }
    return true;
}

static Fp256 fp_sub_modulus(Fp256 a) {
    uint32_t borrow = 0;
    for (int i = 0; i < 8; i++) {
        uint64_t diff = (uint64_t)a.limbs[i] - (uint64_t)MODULUS[i] - (uint64_t)borrow;
        a.limbs[i] = (uint32_t)diff;
        borrow = (uint32_t)(diff >> 63) & 1;
    }
    return a;
}

static Fp256 fp_add_modulus(Fp256 a) {
    uint32_t carry = 0;
    for (int i = 0; i < 8; i++) {
        uint64_t sum = (uint64_t)a.limbs[i] + (uint64_t)MODULUS[i] + (uint64_t)carry;
        a.limbs[i] = (uint32_t)sum;
        carry = (uint32_t)(sum >> 32);
    }
    return a;
}

// ── fp_mul: CIOS Montgomery multiplication (8×32-bit) ───────────────────────

static Fp256 fp_mul(Fp256 a, Fp256 b) {
    uint64_t t[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};

    for (int i = 0; i < 8; i++) {
        uint64_t carry = 0;
        for (int j = 0; j < 8; j++) {
            uint64_t prod = (uint64_t)a.limbs[j] * (uint64_t)b.limbs[i] + t[j] + carry;
            t[j] = prod & 0xFFFFFFFF;
            carry = prod >> 32;
        }
        t[8] += carry;

        uint32_t m = (uint32_t)t[0] * MONT_INV;
        uint64_t prod = (uint64_t)m * (uint64_t)MODULUS[0] + t[0];
        carry = prod >> 32;
        for (int j = 1; j < 8; j++) {
            prod = (uint64_t)m * (uint64_t)MODULUS[j] + t[j] + carry;
            t[j - 1] = prod & 0xFFFFFFFF;
            carry = prod >> 32;
        }
        uint64_t sum = t[8] + carry;
        t[7] = sum & 0xFFFFFFFF;
        t[8] = sum >> 32;
    }

    Fp256 result;
    for (int i = 0; i < 8; i++) result.limbs[i] = (uint32_t)t[i];
    if (t[8] != 0 || fp_gte_modulus(result)) {
        result = fp_sub_modulus(result);
    }
    return result;
}

// ── fp_add: modular addition ────────────────────────────────────────────────

static Fp256 fp_add(Fp256 a, Fp256 b) {
    Fp256 result;
    uint32_t carry = 0;
    for (int i = 0; i < 8; i++) {
        uint64_t sum = (uint64_t)a.limbs[i] + (uint64_t)b.limbs[i] + (uint64_t)carry;
        result.limbs[i] = (uint32_t)sum;
        carry = (uint32_t)(sum >> 32);
    }
    if (carry != 0 || fp_gte_modulus(result)) {
        result = fp_sub_modulus(result);
    }
    return result;
}

// ── fp_sub: modular subtraction ─────────────────────────────────────────────

static Fp256 fp_sub(Fp256 a, Fp256 b) {
    Fp256 result;
    uint32_t borrow = 0;
    for (int i = 0; i < 8; i++) {
        uint64_t diff = (uint64_t)a.limbs[i] - (uint64_t)b.limbs[i] - (uint64_t)borrow;
        result.limbs[i] = (uint32_t)diff;
        borrow = (uint32_t)(diff >> 63) & 1;
    }
    if (borrow != 0) {
        result = fp_add_modulus(result);
    }
    return result;
}

// ── fp_neg: modular negation ────────────────────────────────────────────────

static Fp256 fp_neg(Fp256 a) {
    bool is_zero = true;
    for (int i = 0; i < 8; i++) {
        if (a.limbs[i] != 0) { is_zero = false; break; }
    }
    if (is_zero) return a;

    Fp256 result;
    uint32_t borrow = 0;
    for (int i = 0; i < 8; i++) {
        uint64_t diff = (uint64_t)MODULUS[i] - (uint64_t)a.limbs[i] - (uint64_t)borrow;
        result.limbs[i] = (uint32_t)diff;
        borrow = (uint32_t)(diff >> 63) & 1;
    }
    return result;
}

#endif // FIELD_COMMON_H
