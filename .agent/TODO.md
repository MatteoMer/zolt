# Zolt-Jolt Compatibility Implementation

## Status: IN PROGRESS - Stage 5 Sumcheck Mismatch

## Session 131 Summary

### Key Finding: The eq_prefix Fix is Mathematically Correct

After extensive analysis, I confirmed that:

1. **Jolt's EqPolynomial::evals convention**: bit (n-1-j) of index k ↔ r[j]
   - Example: for n=2, k=1 (binary 01):
     - j=0: bit 1 = 0 → (1-r[0])
     - j=1: bit 0 = 1 → r[1]
     - Result: (1-r[0]) * r[1]
   - This matches Zolt's MSB-first convention (no change needed)

2. **eq_prefix decomposition**: For the pair (2j, 2j+1):
   - eq(2j, r) = eq_prefix(j) * (1 - r[-1])
   - eq(2j+1, r) = eq_prefix(j) * r[-1]
   - eq_prefix = eq(2j, r) / (1 - r[-1]) = eq(j, r[:-1])

3. **Jolt's GruenSplitEqPolynomial structure**:
   - E_out * E_in gives partial eq WITHOUT the current round variable (w_last)
   - current_scalar accumulates eq factors from ALREADY-BOUND variables
   - After current_scalar multiplication, sum_evals = partial_eq * current_scalar

4. **Zolt's bound eq values**:
   - After binding round k with challenge c_k:
     - lookups_eq_evals[j] = eq_partial(j) * accumulated_eq
   - The accumulated_eq is automatically included through the binding process

### Why It Still Fails

The sumcheck polynomial coefficients are computed correctly in terms of the eq_prefix structure, but there must be some other subtle difference:

1. Possibly the binding order or how challenges are indexed
2. Possibly something in finishMlesProductSumFromEvals
3. Possibly how the claim is computed at each round

### Debug Information

From verification output:
```
output_claim:   [84, 83, e6, 0a, 81, 4f, 33, 12, ...]
expected_claim: [c6, 19, df, ae, 44, 5b, ac, 2e, ...]
```

Individual instances match but batched sumcheck fails.

### Files Modified

1. `/home/vivado/projects/zolt/src/zkvm/spartan/stage5_prover.zig`:
   - Added eq_prefix computation for cycle rounds (lines 2816-2848)
   - Factors out eq(X, r_round) from pairs[0] to match Jolt's convention

### Next Steps

1. Add detailed debug output comparing:
   - Zolt's eq_prefix values vs Jolt's E_out * E_in * current_scalar
   - sum_evals at each cycle round
   - The full polynomial coefficients before compression

2. Check if there's an issue with:
   - The claim value passed to finishMlesProductSumFromEvals
   - The r_round value used in the function
   - The interpolation or multiplication step

### Test Commands

```bash
# Build Zolt
zig build -Doptimize=ReleaseFast

# Generate proof
./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format --export-preprocessing /tmp/zolt_preprocessing.bin -o /tmp/zolt_proof_dory.bin

# Verify with Jolt
cp /tmp/zolt_*.bin /home/vivado/projects/jolt/
cd /home/vivado/projects/jolt
cargo test --package jolt-core --features zolt-debug test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture
```
