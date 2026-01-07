# Zolt-Jolt Compatibility - Stage 2 Progress

## Current Status
- Stage 1: PASSING ✅
- Stage 2: FAILING ❌ (sumcheck output_claim mismatch)
- Stage 3+: Not reached yet

## CRITICAL FINDING (Session 5)

The ProductVirtualRemainder round 0 values are **completely different** between Zolt prover and a separate Jolt prover run:

**Jolt (LE first 8 bytes):**
- t0 = b9 40 9a c7 e0 4b e9 a7
- t_inf = 4a d3 0c 19 c0 9d 72 8f
- claim = 3f 5a 2a 38 c3 d0 6c be

**Zolt (LE last 8 bytes of BE):**
- t0 = 63 69 06 26 07 42 54 07
- t_inf = 93 ba a2 28 94 e0 5c c5
- claim = 60 f7 55 cd 48 ef c9 c3

**Important Note**: This comparison was between Jolt's fibonacci example (using Jolt SDK) and Zolt's fibonacci.elf. These are DIFFERENT PROGRAMS with different traces, so different values are expected.

The actual issue is that when Jolt verifies a Zolt-generated proof, the values computed from the proof don't match what the verifier expects.

## Root Cause Hypothesis

The previous_claim for ProductVirtualRemainder round 0 should be the UniSkip output from Stage 2. If this value differs, all subsequent computations will differ.

Possible causes:
1. UniSkip computation uses different challenges or r0 values
2. Tau values (from Stage 1) differ between what Zolt computes vs what Jolt expects
3. The left/right polynomials are constructed from different trace data
4. The fused polynomial values at each cycle don't match

## Verified Components
1. ✅ EqPolynomial.evals produces correct partition of unity (sum = 1)
2. ✅ Witness values are populated from trace correctly
3. ✅ Gruen polynomial computation matches Jolt's formula
4. ✅ Stage 1 passes (bytecode, R1CS compatible)

## Key Files
- `src/zkvm/proof_converter.zig` - Stage 2 batched sumcheck generation
- `src/zkvm/spartan/product_remainder.zig` - ProductVirtualRemainder prover

## Next Steps
1. **Compare Stage 2 input_claim** - the UniSkip claim that starts ProductVirtualRemainder
2. **Compare tau values** - the challenges used for split_eq
3. **Compare left[0]/right[0]** - first values of fused polynomials
4. **Trace exact byte values** during Jolt verification to see what Jolt expects vs what Zolt provides

## Debug Commands
```bash
# Build and generate proof
zig build -Doptimize=ReleaseFast
./zig-out/bin/zolt prove --jolt-format -o /tmp/proof.bin examples/fibonacci.elf 2>&1 | grep -E "PRODUCT round"

# Test with Jolt
cd /Users/matteo/projects/jolt
cargo test --package jolt-core test_verify_zolt_proof -- --ignored --nocapture 2>&1 | tail -100
```
