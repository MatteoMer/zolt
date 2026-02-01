# Zolt-Jolt Compatibility Implementation

## Status: IN PROGRESS - Stage 5 Cycle Round Polynomial Computation Issue

## Current Session Progress (Session 88)

### What's Working
- ✅ Stage 5 cycle rounds (128-135) produce degree-10 polynomials
- ✅ `evalLinearProd10` correctly evaluates product of 10 linear factors at [1, 2, ..., 9, ∞]
- ✅ `fromEvalsToom` correctly interpolates from Toom-Cook evaluations
- ✅ Sumcheck property `p(0) + p(1) = claim` holds for all cycle rounds
- ✅ Initial batched_claim for Stage 5 matches between Zolt and Jolt verifier

### What's Not Working
- ❌ Final output_claim doesn't match expected_claim after all 136 rounds
- Output claim: `[db, d3, 94, 5e, a2, ae, d7, 0d, ...]`
- Expected claim: `[2a, f2, 1c, 73, 3c, 5a, b3, 61, ...]`

### Key Observations
1. Rounds 134 and 135 have `p(1) = 0`, meaning all evaluated products at X=1 are zero
2. This might be correct (T=256 = 2^8, so after 6 rounds, we have only 4 cycles left, after 7 rounds only 2 cycles left)
3. The sumcheck property still holds, so the polynomial computation is internally consistent

### Possible Issues to Investigate
1. **Split-Eq vs Direct Eq**: My approach uses 10 linear factors directly (eq + val + 8 ra). Jolt uses:
   - 9 linear factors (e_in absorbed into val + 8 ra)
   - Then multiplies by e_out, current_scalar
   - Then `finish_mles_product_sum_from_evals` adds eq(X, r_round) factor

2. **r_round Indexing**: For cycle rounds, Jolt uses:
   - r_round = r_reduction[current_index - 1] (LowToHigh)
   - For cycle round 0: r_round = r_reduction[7]
   - For cycle round 7: r_round = r_reduction[0]

3. **Variable Binding Order**: The eq polynomial uses BIG_ENDIAN (MSB first):
   - Bit 0 (LSB) of cycle index corresponds to r_reduction[7]
   - Bit 7 (MSB) of cycle index corresponds to r_reduction[0]

### Implementation Files
- `/home/vivado/projects/zolt/src/zkvm/spartan/stage5_prover.zig` - Cycle round computation (lines 1060-1250)
- `/home/vivado/projects/zolt/src/poly/mod.zig` - Added `evalLinearProd10`, `fromEvalsToom`, `toCompressed`

### Test Commands
```bash
# Build
zig build -Doptimize=ReleaseFast

# Generate proof
./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format --export-preprocessing /tmp/zolt_preprocessing.bin -o /tmp/zolt_proof_dory.bin

# Verify with Jolt
cd /home/vivado/projects/jolt
cargo test --package jolt-core --features zolt-debug test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture
```

### Next Steps for Future Session
1. Add debug output comparing individual cycle round polynomials between Zolt and Jolt prover
2. Check if the expected_claim computation in Jolt verifier matches the sumcheck output
3. Consider implementing `finishMlesProductSumFromEvals` approach instead of direct 10-factor product
4. Verify the batch coefficient handling for Instance 2 during cycle rounds

## Previous Session Work (Session 87)
- ✅ Per-chunk ra_weights tracking during address rounds
- ✅ Proper cycle round binding: bind chunks separately, compute product after binding
- ✅ ra_product == lookups_ra_weights[0] after all binding (invariant now holds)
- ✅ Proper table_flags and raf_flag computation from eq(r_cycle', j) sums
