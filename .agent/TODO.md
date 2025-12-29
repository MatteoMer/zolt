# Zolt-Jolt Compatibility TODO

## Current Status: Stage 1 Output Claim Mismatch (Session 20)

### Summary
All Stage 1 sumcheck rounds pass (p(0)+p(1) = claim), but the **expected output claim** doesn't match. The ratio is ~1.338 consistently.

### Key Discovery (Session 20)
- Fixed R1CSInputIndex order to match Jolt's ALL_R1CS_INPUTS exactly
- Jolt's Lagrange kernel includes `L(τ_high, r0)` as scaling factor in round polynomials
- Zolt's `StreamingOuterProver` uses `initWithScaling` with lagrange_tau_r0
- The issue is likely in how the two provers (SpartanOuterProver and StreamingOuterProver) coordinate

### Key Values (Latest Run)
- output_claim (sumcheck): 21176670064311113248327121399637823341669491654917035040693110982193526510099
- expected (R1CS): 15830891598945306629010829910964994017594280764528826029442912827815044293203
- Ratio: ~1.338

### Root Cause Hypothesis
Two separate provers create the proof:
1. `SpartanOuterProver` - creates UniSkip polynomial from witnesses + full tau
2. `StreamingOuterProver` - creates remaining rounds from witnesses + tau_low + lagrange_tau_r0

These may be computing inconsistent values because:
- Different eq polynomial factorizations
- The UniSkip polynomial evaluation at r0 may not match what StreamingOuter expects

### Next Steps
1. Add debug output comparing lagrange_tau_r0 with Jolt's tau_high_bound_r0
2. Verify split_eq.current_scalar includes Lagrange kernel
3. Compare t_zero and t_infinity values per round
4. Ensure the two provers produce a consistent sumcheck

### Expected Output Claim Formula (Jolt Verifier)
```rust
expected = tau_high_bound_r0 * tau_bound_r_tail * inner_sum_prod

inner_sum_prod = az_final * bz_final
az_final = az_g0 + r_stream * (az_g1 - az_g0)
bz_final = bz_g0 + r_stream * (bz_g1 - bz_g0)

az_g0 = Σ w[i] * lc_a[i].dot_product(z)
bz_g0 = Σ w[i] * lc_b[i].dot_product(z)

z = [r1cs_input_evals..., 1]  // 37 elements
```

Where:
- `L(τ_high, r0)` = Lagrange kernel (scaling factor for round polys)
- `r_tail_reversed = sumcheck_challenges.reversed()` (includes r_stream)
- R1CS inputs evaluated at `challenges[1..]` (excludes r_stream)

## Completed

### Phase 1-5: Core Infrastructure
1. Transcript Compatibility - Blake2b
2. Proof Structure - 7-stage
3. Serialization - Arkworks format
4. Commitment - Dory with Jolt SRS
5. Verifier Preprocessing Export

### Stage 1 Fixes (Sessions 11-20)
- Big-endian EqPolynomial.evals()
- R1CSInputIndex order matches Jolt
- All 656 Zolt tests pass

## Test Commands

```bash
# Zolt tests
zig build test --summary all

# Generate proof
zig build -Doptimize=ReleaseFast
./zig-out/bin/zolt prove examples/sum.elf --jolt-format -o /tmp/zolt_proof_dory.bin --export-preprocessing /tmp/zolt_preprocessing.bin

# Jolt verification test
cd /Users/matteo/projects/jolt
cargo test --package jolt-core test_debug_stage1_verification -- --ignored --nocapture
```
