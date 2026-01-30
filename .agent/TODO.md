# Zolt-Jolt Compatibility Implementation

## Status: IN PROGRESS - Stage 5 Sumcheck Mismatch

## Verified Stages
- Stage 1: PASSED ✅
- Stage 2: PASSED ✅
- Stage 3: PASSED ✅
- Stage 4: PASSED ✅
- Stage 5: FAILING ❌ (output_claim != expected_output_claim)
- Stage 6: Not tested yet
- Stage 7: Not tested yet

## Current Issue: Stage 5 Sumcheck Mismatch

The Stage 5 sumcheck verification fails with:
- `output_claim = 14207773099973851432316380405455832148939247972351308198786873286672527593212`
- `expected_output_claim = 18244124058491777017643072331864832508434028736769776383365323382525894304545`

### Investigation Summary

1. **Opening points are stored correctly** - Debug confirmed that the `RegistersVal` opening point from Stage 4 is stored in the accumulator with correct values.

2. **Serialization red herring** - Earlier debug showed zeros because `F::Challenge::serialize_compressed()` behaves differently than `F::serialize_compressed()`. The actual values are correct.

3. **LT polynomial algorithm matches Jolt** - Both compute `evals[j] = x + r - x*r` and `evals[half+j] = x*r` iterating from LSB to MSB.

4. **Binding order matches** - Both use `LowToHigh` binding: `Z_new[i] = (1-r)*Z[2i] + r*Z[2i+1]`

5. **Toom-Cook encoding** - Zolt computes `[p(0), p(1), p(2), p_inf]` where `p_inf = (inc_1-inc_0)*(wa_1-wa_0)*(lt_1-lt_0)`. This matches Jolt.

### Possible Remaining Issues

1. **Input claim mismatch** - The Stage 5 input claim may not match what Jolt expects
2. **WA polynomial** - The write-address polynomial may be computed differently
3. **Inc polynomial** - The increment polynomial may be computed differently
4. **Transcript divergence** - Though Stages 1-4 pass, there may be a subtle transcript difference

### Next Steps

1. Compare Stage 5 round 0 coefficients between Zolt and Jolt
2. Verify inc/wa polynomial values at index 0
3. Check if the batching coefficient is correct
4. Verify the input_claim for Stage 5 matches exactly

## Test Commands

```bash
# Generate proof
cd /home/vivado/projects/zolt
./zig-out/bin/zolt prove examples/fibonacci.elf \
  --jolt-format \
  -o /tmp/zolt_proof_dory.bin

# Verify with Jolt
cd /home/vivado/projects/zolt/jolt
cargo test --package jolt-core test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture
```

## Key Files
- `src/zkvm/spartan/stage5_prover.zig` - Zolt Stage 5 prover
- `jolt-core/src/zkvm/registers/val_evaluation.rs` - Jolt Stage 5 verifier
- `jolt-core/src/poly/lt_poly.rs` - Jolt LT polynomial

## Commits Made This Session
- `85d485d` - fix(stage5): correct constant polynomial p_inf to be zero
