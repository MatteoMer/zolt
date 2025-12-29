# Zolt-Jolt Compatibility TODO

## Current Status: Stage 1 Output Claim Mismatch (Session 19)

### Summary
All Stage 1 sumcheck rounds pass (p(0)+p(1) = claim), but the **expected output claim** doesn't match. The ratio changed from ~1.2 to ~1.34 after fixing EqPolynomial to use big-endian indexing.

### Key Values (Latest Run)
- output_claim (sumcheck): 21176670064311113248327121399637823341669491654917035040693110982193526510099
- expected (R1CS): 15830891598945306629010829910964994017594280764528826029442912827815044293203
- Ratio: ~1.338

### Verified Components ✓
1. EqPolynomial.evals() uses big-endian indexing
2. split_eq E tables use big-endian indexing
3. computeCubicRoundPoly matches Jolt's gruen_poly_deg_3
4. r_cycle = challenges[1..] reversed (10 elements)
5. tau_bound_r_tail uses all 11 challenges reversed
6. Constraint definitions match Jolt
7. First/Second group indices match Jolt
8. Lagrange polynomial domain matches Jolt (-4..5)
9. Domain size (10 for first group) matches

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

### Remaining Investigation Areas

1. **Opening Claims (r1cs_input_evals)**
   - The z vector comes from proof.opening_claims
   - If Zolt's MLE evaluations are wrong, expected will be wrong
   - Check: `R1CSInputEvaluator.computeClaimedInputs` output
   - Compare specific values with what Jolt's prover would produce

2. **Witness Extraction (R1CSCycleInputs)**
   - Zolt: `R1CSCycleInputs.fromTraceStep`
   - Jolt: `R1CSCycleInputs::from_trace`
   - Verify the witness values for specific fields match

3. **Constraint Dot Product**
   - Verify lc.dot_product computes same as Zolt's LC.evaluate
   - Check constant terms are handled correctly

4. **Debug Approach**
   - Add debug output to print first cycle's Az_g0, Bz_g0 in Zolt prover
   - Compare with Jolt's computed az_g0, bz_g0 from the verifier
   - Track down which component differs

### Files to Examine
- Zolt witness extraction: `src/zkvm/r1cs/constraints.zig:R1CSCycleInputs.fromTraceStep`
- Zolt MLE evaluation: `src/zkvm/r1cs/evaluation.zig:computeClaimedInputs`
- Jolt witness extraction: `jolt-core/src/zkvm/r1cs/inputs.rs:from_trace`
- Jolt MLE evaluation: `jolt-core/src/zkvm/r1cs/evaluation.rs:compute_claimed_inputs`

## Completed

### Phase 1-5: Core Infrastructure
1. Transcript Compatibility - Blake2b
2. Proof Structure - 7-stage
3. Serialization - Arkworks format
4. Commitment - Dory with Jolt SRS
5. Verifier Preprocessing Export

### Stage 1 Fixes (Sessions 11-19)
6-33. [Previous fixes...]
34. Big-endian EqPolynomial.evals()

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
