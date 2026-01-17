# Zolt-Jolt Compatibility TODO

## Current Progress

| Stage | Internal (Zolt) | Cross-verify (Jolt) | Notes |
|-------|-----------------|---------------------|-------|
| 1 | ✅ PASS | ❌ FAIL | Streaming outer prover bug |
| 2 | ✅ PASS | - | - |
| 3 | ✅ PASS | - | - |
| 4 | ✅ PASS | Montgomery fix applied |
| 5 | ✅ PASS | - | - |
| 6 | ✅ PASS | - | - |

## Session 42 Progress (2026-01-17)

### Bug Identified: Streaming Outer Prover Az*Bz Mismatch

**Root Cause:**
The streaming outer prover (`src/zkvm/spartan/streaming_outer.zig`) generates sumcheck polynomials that are internally consistent but encode the WRONG Az*Bz computation.

**Evidence:**
- Round polynomial coefficients ARE non-zero ✓
- batching_coeff MATCHES Jolt (185020165269464640985840804826040774859) ✓
- Opening claims are computed with actual evaluations ✓
- BUT output_claim differs from expected_output_claim

**Values:**
- Jolt's `inner_sum_prod (Az*Bz)`: 14279035532130326282759614533689080459036208928223103610768541756919699764986
- Proof's `output_claim`: 10634556229438437044377436930505224242122955378672273598842115985622605632100
- Expected: 17179895292250284685319038554266025066782411088335892517994655937988141579529

### Key Comparison: Constraint Evaluation

**Jolt's `extrapolate_from_binary_grid_to_tertiary_grid`:**
1. Uses `R1CSCycleInputs::from_trace` for witness values
2. Uses `R1CSEval::from_cycle_inputs` for evaluator
3. Uses `eval.fmadd_first_group_at_r` (selector=0)
4. Uses `eval.fmadd_second_group_at_r` (selector=1)
5. Uses specialized accumulators (`Acc5U`, `Acc6S`, `Acc7S`)

**Zolt's `materializeLinearPhasePolynomials`:**
1. Uses `cycle_witnesses` from R1CSConstraintGenerator
2. Uses `UNIFORM_CONSTRAINTS` array for evaluation
3. Uses `FIRST_GROUP_INDICES` / `SECOND_GROUP_INDICES` for constraint grouping
4. Uses basic field multiplication

**Likely Issue:**
The constraint group indexing or evaluation logic differs between Zolt and Jolt.

### Test Results
- **714/714 unit tests pass** ✅
- Cross-verification fails at Stage 1 sumcheck

### Next Steps

1. **Compare constraint group indices**: Verify FIRST_GROUP_INDICES matches Jolt
2. **Debug Az/Bz values**: Add per-cycle logging to compare with Jolt
3. **Check constraint evaluation**: Ensure condition/left/right evaluation matches
4. **Verify selector logic**: Check if (full_idx & 1) selector is correct

### Commands

```bash
# Generate proof with Jolt's SRS
zig build run -- prove examples/fibonacci.elf --jolt-format --srs /tmp/jolt_dory_srs.bin --export-preprocessing /tmp/zolt_preprocessing.bin -o /tmp/zolt_proof_dory.bin

# Test cross-verification
cd /Users/matteo/projects/jolt/jolt-core && cargo test test_verify_zolt_proof_with_zolt_preprocessing --release -- --nocapture --ignored
```

## Previous Sessions Summary

- **Session 41**: Fixed Stage 4 Montgomery conversion, fixed proof serialization
- **Session 40**: Fixed Stage 2 synthetic termination write
- **Earlier**: Fixed Stage 3 prefix-suffix, Stage 1 NextPC=0 for NoOp
