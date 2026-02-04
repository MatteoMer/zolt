# Zolt-Jolt Compatibility Implementation

## Status: Session 71 - Stage 1 Sumcheck Investigation

## Current Issue: Stage 1 Sumcheck Verification Failure

### Root Cause Analysis (DEEP DIVE)

The prover's output_claim doesn't match the verifier's expected_claim:
- **output_claim**: `[8f, 49, 4e, 9d, ...]` (from prover's sumcheck)
- **expected_claim**: `[f5, 77, db, ef, ...]` (verifier's computation)

Both use the **same eq_factor**: `[f7, be, 45, d2, b1, 33, ...]` ✅

The mismatch is in Az*Bz:
- **Prover's implied Az*Bz**: `[ad, ed, 4d, d6, 9a, ...]` (final_claim / eq_factor)
- **Verifier's Az*Bz**: `[6e, d1, 32, 0b, 4a, ...]` (from inner_sum_prod)

### Understanding the Protocol

Stage 1 proves:
```
Σ_{x_constr, x_cycle} eq(τ, x) * Az(x_constr, x_cycle) * Bz(x_constr, x_cycle) = 0
```

The sumcheck has two phases:
1. **UniSkip round**: Handles constraint dimension via degree-27 polynomial over domain {-4,...,5}
2. **Streaming round**: Binds r_stream to blend first/second constraint groups
3. **Cycle rounds**: Bind r_cycle to the cycle dimension (8 rounds for 256 cycles)

After binding:
- `r_constr = (r_stream, r0)`
- `r_cycle = (r_1, ..., r_8)`

Expected output claim:
```
eq(τ, r) * Az(r) * Bz(r)
```

### Verifier's Computation

```rust
// Lagrange weights at r0 for constraint groups
w = LagrangePolynomial::evals(r0);

// z = R1CS input MLE evaluations at r_cycle
z = r1cs_input_evals;  // 36 values

// First group Az/Bz using Lagrange blend
az_g0 = Σ_i w[i] * lc_a[i].dot_product(z)
bz_g0 = Σ_i w[i] * lc_b[i].dot_product(z)

// Second group Az/Bz
az_g1 = Σ_i w[i] * lc_a[i].dot_product(z)
bz_g1 = Σ_i w[i] * lc_b[i].dot_product(z)

// Blend with r_stream
az_final = az_g0 + r_stream * (az_g1 - az_g0)
bz_final = bz_g0 + r_stream * (bz_g1 - bz_g0)

return az_final * bz_final
```

Key insight: `lc_a[i].dot_product(z)` computes the constraint Az using MLE-evaluated R1CS inputs.

### Prover's Current Implementation

Looking at `JoltSpartanInterface`:
1. Computes Az[constraint_idx] and Bz[constraint_idx] for all (cycle, constraint) pairs
2. Creates `combined_poly[x] = eq(tau,x) * Az[x] * Bz[x]`
3. Does standard sumcheck on combined_poly

This is correct for the sumcheck structure. The issue must be in how the challenges are used.

### Hypothesis: Challenge Ordering or Binding Issue

The debug output shows:
- Prover's rounds match expected behavior (p(0)+p(1) = claim)
- Challenge values are derived correctly
- eq_factor matches between prover and verifier

BUT the final implied Az*Bz doesn't match what the verifier computes from R1CS input evals.

This suggests the prover's sumcheck polynomial structure differs from what the verifier expects.

### Possible Issues

1. **Index ordering**: Zolt might use different bit ordering (high-to-low vs low-to-high) than Jolt
2. **Constraint grouping**: The UniSkip interleaves constraint groups, but cycle rounds might not handle this correctly
3. **R1CS input ordering**: Zolt's witness structure might differ from Jolt's expectations

### Next Steps

1. Add debug to print prover's Az/Bz at the final challenge point
2. Compare with verifier's az_g0, bz_g0, az_g1, bz_g1
3. Check if the constraint-to-cycle indexing matches between prover and verifier

### Test Commands

```bash
zig build -Doptimize=ReleaseFast
./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format -o /tmp/zolt_proof_dory.bin --export-preprocessing /tmp/zolt_preprocessing.bin --trace-length 64
cd /home/vivado/projects/jolt && cargo test -p jolt-core --features zolt-debug --lib test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture
```

## Files Modified This Session

- `.agent/TODO.md`: Updated with deep analysis
