# Zolt-Jolt Compatibility TODO

## Current Status: Session 31 - January 2, 2026

**All 702 tests pass**

### Issue: Transcript still diverges - r0 challenge is WRONG

Despite fixing memory layout constants, the r0 challenge from the transcript is different:
- Zolt r0: `7193532858613009548167626689752378790` (very small, ~128 bits)
- Jolt r0: `6919882260122427158724897727024710502508333642996197054262116261168391078818` (full 254 bits)

This means something appended to the transcript BEFORE the r0 challenge differs:
1. Fiat-Shamir preamble (memory layout, inputs, outputs, panic, ram_K, trace_length)
2. Polynomial commitments (Dory GT elements)
3. UniSkip polynomial coefficients

### Root Cause Analysis

The transcript divergence causes:
1. Wrong r0 challenge
2. Wrong uni_skip_claim = UniSkip(r0)
3. Wrong current_claim for remaining sumcheck
4. Wrong polynomial evaluations in all subsequent rounds
5. Verification failure

### Items to Investigate

1. **Dory commitment serialization** - GT element toBytes() might differ from arkworks
2. **Commitment order** - Different order of commitments appended to transcript
3. **Missing data** - Jolt might append additional data before stage 1
4. **Coefficient format** - UniSkip polynomial coefficients might be serialized differently

### Completed Work (This Session)

1. [x] Fixed memory layout constants (128MB memory, 4KB stack)
2. [x] Added debug tracing for round polynomial computation
3. [x] Identified that r0 challenge differs (transcript divergence)
4. [x] Confirmed uni_skip_claim is wrong because r0 is wrong
5. [x] Verified Gruen's cubic polynomial formula is correct
6. [x] Verified bind() operation is correct
7. [x] Verified interpolation and evaluation formulas are correct

### Next Steps

1. [ ] Add transcript state debugging to compare bytes appended
2. [ ] Verify GT element serialization (384 bytes, reversed)
3. [ ] Compare UniSkip polynomial coefficients with Jolt
4. [ ] Check if any preprocessing data is missing from transcript

### Blocking Issues

1. **Dory proof generation panic** - index out of bounds in openWithRowCommitments
   - `params.g2_vec` has length 64 but `current_len` is 128
   - Need to fix SRS loading or verify polynomial degree bounds

## Verified Correct
- [x] Blake2b transcript implementation format
- [x] Field serialization (Arkworks format)
- [x] Memory layout constants match Jolt
- [x] UniSkip polynomial generation logic
- [x] R1CS constraint definitions (19 constraints, 2 groups)
- [x] Split eq polynomial factorization (E_out/E_in tables)
- [x] Lagrange kernel L(tau_high, r0)
- [x] MLE evaluation for opening claims
- [x] Gruen cubic polynomial formula
- [x] bind() operation (correct eq factor)
- [x] Polynomial evaluation (Horner's method)
- [x] Interpolation (Vandermonde inverse)
- [x] evalsToCompressed produces correct compressed coefficients
- [x] All 702 Zolt tests pass

## Test Commands
```bash
# Zolt tests
zig build test --summary all

# Generate proof with debug output
zig build -Doptimize=Debug && ./zig-out/bin/zolt prove path/to/elf --jolt-format -o /tmp/proof.bin

# Jolt verification test
cd /Users/matteo/projects/jolt
cargo test --package jolt-core test_verify_zolt_proof -- --ignored --nocapture
```
