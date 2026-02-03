# Zolt-Jolt Compatibility Implementation

## Status: Session 27 - INVESTIGATING eq_eval_r_reduction MISMATCH

## Current Issue

**Stage 5 sumcheck fails because Zolt's polynomial evaluations differ from what Jolt's verifier expects.**

- Zolt's `output_claim`: `[ed, a5, f6, bf, ...]`
- Jolt's `expected_claim`: `[b2, 8f, 91, 24, ...]`

The expected_claim is computed from 3 instances:
1. RegistersValEvaluation
2. RamRaClaimReduction
3. InstructionReadRaf

Instance 2 (InstructionReadRaf) uses `eq_eval_r_reduction` which multiplies the result.

## Key Finding

**r_reduction values are CORRECTLY MATCHING between Zolt and Jolt!**

Zolt stores r_reduction as Challenge (Montgomery form internally):
```
r_reduction_be[0] limbs = [0, 0, low=b000efc0b0898d0d, high=0734140b1b8aa484]
```

Jolt serializes r_reduction[0] as:
```
[0d, 8d, 89, b0, c0, ef, 00, b0, 84, a4, 8a, 1b, 0b, 14, 34, 07]
```

When reversed (LE to BE): `0734140b1b8aa484b000efc0b0898d0d`

This matches Zolt's `high=0x0734140b1b8aa484`, `low=0xb000efc0b0898d0d`.

## The Problem

Jolt computes `eq_eval_r_reduction = EqPolynomial::mle(&r_reduction, &r_cycle_prime)` = `[8c, 9f, a7, ab, ...]`

Zolt computes `lookups_current_scalar` after 8 cycle rounds using:
```zig
// eq(w, r) = 1 - w - r + 2*w*r
const w_i = r_reduction[n_cycle_vars - 1 - lookups_round];
const prod_w_r = w_i.mulHiBigIntU128(challenge.limbs);
const one_minus_r_scalar = F.one().sub(challenge);
const eq_factor = one_minus_r_scalar.sub(w_i).add(prod_w_r).add(prod_w_r);
lookups_current_scalar = lookups_current_scalar.mul(eq_factor);
```

The formula is correct (`1 - w - r + 2wr`). The order is correct (both pair r_reduction elements with reversed cycle challenges).

## Hypothesis

The issue might be in how Zolt performs the Challenge arithmetic operations:
1. `F.one().sub(challenge)` - does this correctly compute `1 - r` where r is a Challenge?
2. `w_i.mulHiBigIntU128(challenge.limbs)` - does this correctly compute `w * r`?

Need to verify these operations match Jolt's behavior.

## Next Steps

1. Add detailed debug in Zolt to print intermediate values during eq_factor computation
2. Compare with Jolt's arithmetic (Challenge * Challenge and F - Challenge)
3. Verify the Montgomery form handling is correct in both systems

## Test Commands

```bash
# Jolt verification with debug
cd jolt && cargo test -p jolt-core --features zolt-debug --lib test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture

# Zolt proof generation
./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format -o logs/zolt_proof_dory.bin --export-preprocessing logs/zolt_preprocessing.bin
```
