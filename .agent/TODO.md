# Zolt-Jolt Compatibility TODO

## Current Status: Session 33 - January 2, 2026

**All 702 Zolt tests pass**

### Fixes Applied This Session

1. **Fixed Constraint 8 (RightLookupSub)**: Added missing 2^64 constant (`lc.constant = 0x10000000000000000`)
2. **Fixed i128ToField**: Updated helpers in both `constraints.zig` and `jolt_r1cs.zig` to handle values > 2^64 using bytes representation
3. **Fixed SUB witness generation**: Now correctly computes `LeftInput - RightInput + 2^64`
4. **Constraint Audit**: Verified all 19 constraints match Jolt exactly (via Task agent)

### Current Issue: Stage 1 Sumcheck Still Fails

Verification error:
```
output_claim:          17460831489166246924525756229519258101479150040426090837402456066148331301615
expected_output_claim: 5659454754207529901068046179760836365067409963211356691285388103257563690698
```

The sumcheck round polynomials pass individually (p(0) + p(1) = claim), but the final claim doesn't match what the verifier computes.

### Analysis

The expected_output_claim is computed by Jolt as:
```rust
result = tau_high_bound_r0 * tau_bound_r_tail_reversed * inner_sum_prod

Where:
- inner_sum_prod = Az_final * Bz_final
- Az_final = az_g0 + r_stream * (az_g1 - az_g0)
- Bz_final = bz_g0 + r_stream * (bz_g1 - bz_g0)
- az_g0 = Σᵢ w[i] * lc_a[i].dot_product(z)  // z = R1CS input MLE evaluations
- bz_g0 = Σᵢ w[i] * lc_b[i].dot_product(z)
```

The issue is that:
1. **Constraints match** - all 19 constraint definitions are identical between Zolt and Jolt
2. **Eq factor matches** - `tau_high_bound_r0 * tau_bound_r_tail_reversed` is correct
3. **inner_sum_prod mismatch** - The Az·Bz product computed by the prover differs from what verifier expects

### Possible Root Causes

1. **R1CS input MLE evaluations**: The opening claims for the 36 R1CS inputs might be computed incorrectly
   - Need to verify `computeClaimedInputs` matches Jolt's MLE evaluation at r_cycle

2. **Lagrange weights mismatch**: The Lagrange basis weights L_i(r0) might differ
   - Jolt uses domain {-4, -3, ..., 4, 5} (symmetric around 0.5)
   - Need to verify Zolt uses the same domain and evaluation

3. **r_cycle orientation**: The opening point might use wrong byte order
   - Jolt: `r_cycle = sumcheck_challenges[1..].reverse()` (big-endian)
   - Need to verify Zolt converts correctly

4. **Streaming vs Linear phase**: The linear phase materialization might have issues
   - Jolt materializes Az/Bz once, then binds with `bound_poly_var_bot`
   - Zolt implements this but might have indexing bugs

### Next Steps

1. Add debug output to compare:
   - `r1cs_input_evals[0..3]` between Zolt's opening claims and Jolt's expectation
   - `az_g0, bz_g0, az_g1, bz_g1` values at verification time
   - Lagrange weights `w[0..9]` at r0

2. Create a minimal test case that:
   - Uses a single cycle
   - Prints intermediate values from both Zolt prover and Jolt verifier
   - Identifies exact divergence point

### Test Commands
```bash
# Run Zolt tests
zig build test --summary all

# Generate proof
zig build -Doptimize=ReleaseFast && ./zig-out/bin/zolt prove examples/fibonacci.elf \
  --jolt-format -o /tmp/zolt_proof_dory.bin

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
- [x] R1CS constraint definitions (19 constraints, 2 groups) - FIXED
- [x] Constraint 8 (RightLookupSub) now has 2^64 constant
- [x] UniSkip polynomial generation
- [x] Memory layout constants match Jolt
- [x] R1CS input ordering matches Jolt's ALL_R1CS_INPUTS

### All Tests Pass
- [x] 702/702 Zolt tests pass
