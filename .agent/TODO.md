# Zolt-Jolt Compatibility TODO

## Current Status: Session 32 - January 2, 2026

**All 702 Zolt tests pass**

### Analysis This Session

1. **Proof Generation Works**: Zolt successfully generates proofs (`zig build && ./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format -o /tmp/zolt_proof.bin`)

2. **Verification Fails at Stage 1**: The sumcheck output_claim doesn't match expected_output_claim
   - output_claim = 18149181199645709635565994144274301613989920934825717026812937381996718340431
   - expected = 9784440804643023978376654613918487285551699375196948804144755605390806131527

3. **Eq Factor is Correct**: The cross-verification test confirms that `prover_eq_factor == verifier_eq_factor`

4. **Az/Bz MLE Match**: The test shows "Az MLE match: true, Bz MLE match: true"

5. **Root Cause Analysis**:
   - Expected formula: `expected = tau_high_bound_r0 * tau_bound_r_tail_reversed * inner_sum_prod`
   - Prover formula: `output_claim = eq_factor * sumcheck_accumulated_az_bz`
   - The eq factors match, but `sumcheck_accumulated_az_bz != inner_sum_prod`
   - This means the sumcheck prover is computing Az*Bz differently than what the verifier expects

### Hypothesis

The sumcheck prover computes:
```
Σ_cycle eq(tau, cycle) * Az(cycle) * Bz(cycle)  // Sum over cycles at each round
```

After binding to random point r, this becomes:
```
eq(tau, r) * MLE(Az * Bz, r)
```

But the verifier expects:
```
eq(tau, r) * MLE(Az, r) * MLE(Bz, r)
```

**Key insight**: `MLE(Az * Bz, r) ≠ MLE(Az, r) * MLE(Bz, r)` in general!

The sumcheck should compute `MLE(Az, r) * MLE(Bz, r)`, NOT the MLE of the product.

### The Problem

Looking at Jolt's outer sumcheck, it uses a **multiquadratic polynomial** representation where:
- `t'(X)` encodes Az and Bz separately
- The round polynomial is built by multiplying the projections of Az and Bz

In Zolt, the streaming round computes:
```zig
const prod_0 = az_g0.mul(bz_g0);  // Product at position 0
const prod_inf = slope_az.mul(slope_bz);  // Product of slopes
```

This is correct! The Gruen construction uses:
- t'(0) = Az(0) * Bz(0)
- t'(∞) = (Az(1) - Az(0)) * (Bz(1) - Bz(0))

Which gives a polynomial that evaluates to Az(X) * Bz(X) at any point X.

So the structure is correct. The bug must be elsewhere...

### Next Steps

1. Compare Jolt's streaming round polynomial coefficients with Zolt's
2. Check if the constraint group blending with r_stream is correct
3. Verify the index mapping (x_out, x_in) -> cycle_idx is correct
4. Check if there's an issue with how E_out/E_in tables are being used

### Test Commands
```bash
# Run Zolt tests
zig build test --summary all

# Generate proof
zig build -Doptimize=ReleaseFast && ./zig-out/bin/zolt prove examples/fibonacci.elf \
  --jolt-format -o /tmp/zolt_proof.bin

# Jolt verification tests
cd /Users/matteo/projects/jolt
cargo test --package jolt-core test_verify_zolt_proof -- --ignored --nocapture
```

## Verified Correct Components

### Transcript
- [x] Blake2b transcript format matches Jolt
- [x] Challenge scalar computation (128-bit, no masking)
- [x] Field serialization (Arkworks LE format)

### Polynomial Computation
- [x] Gruen cubic polynomial formula
- [x] Split eq polynomial factorization (E_out/E_in)
- [x] bind() operation (eq factor computation)
- [x] Lagrange interpolation
- [x] evalsToCompressed format

### RISC-V & R1CS
- [x] R1CS constraint definitions (19 constraints, 2 groups)
- [x] UniSkip polynomial generation
- [x] Memory layout constants match Jolt
- [x] R1CS input ordering matches Jolt's ALL_R1CS_INPUTS

### All Tests Pass
- [x] 702/702 Zolt tests pass
