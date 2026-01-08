# Zolt-Jolt Compatibility - Current Status

## Summary (Session 29 - Updated)

**Stage 1**: PASSES with Zolt preprocessing ✓
**Stage 2**: PASSES with Zolt preprocessing ✓
**Stage 3**: FAILS - Round polynomials produce wrong output_claim
**Stage 4-7**: Untested (blocked on Stage 3)

### Latest Progress: Gamma Values Fixed! ✓

The gamma values now match between Zolt and Jolt:
- shift_gamma[1] = 167342415292111346589945515279189495473 ✓
- instr_gamma and reg_gamma also match ✓

The fix: Stage 3 uses Jolt's `challenge_scalar` (NOT `challenge_scalar_optimized`),
which does NOT apply 125-bit masking. Zolt was using `challengeScalar` with masking.
Now using `challengeScalarFull` which properly converts the 128-bit value.

### Current Issue: Sumcheck Round Polynomials

```
Stage 3 output_claim:          12471530361947505251693797544524218046917141118486157199773065282661666309311
Stage 3 expected_output_claim: 5306394876981572062168920462043199017393461233968153260958946411171098925214
```

The round polynomials are not computing the correct function. The expected_output_claim
is computed from opening claims, which are now correct. So the issue must be in the
Stage 3 prover's round polynomial computation.

### Next Steps

1. [ ] Debug Stage 3 round polynomial computation
2. [ ] Verify eq/eq+1 polynomial evaluations are correct
3. [ ] Verify the batched sumcheck claim computation
4. [ ] Check that witness data is being used correctly

### Testing Commands

```bash
# Generate proof with preprocessing
./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format \
  --export-preprocessing /tmp/zolt_preprocessing.bin \
  -o /tmp/zolt_proof_dory.bin --srs /tmp/jolt_dory_srs.bin

# Test with Zolt preprocessing
cd /Users/matteo/projects/jolt/jolt-core && \
  cargo test test_verify_zolt_proof_with_zolt_preprocessing --release -- --ignored --nocapture
```

### Key Insights

1. **125-bit masking difference**: Stage 1-2 sumcheck challenges use `challenge_scalar_optimized`
   with 125-bit masking. Stage 3 uses `challenge_scalar` WITHOUT masking.

2. **Transcript state matches**: After Stage 2 cache_openings, transcript states are identical.

3. **Claim values match**: All 17 Stage 2 cache_openings claims match between Zolt and Jolt.
