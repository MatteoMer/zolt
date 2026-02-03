# Zolt-Jolt Compatibility Implementation

## Status: Session 22 - Stage 5 Sumcheck Mismatch Identified

## Progress This Session

### Fixes Implemented
1. **SumcheckId**: Fixed to 24 variants (added AdviceClaimReductionCyclePhase, AdviceClaimReduction)
2. **Proof Config**: Fixed serialization (rw_config, one_hot_config as u8 fields, dory_layout)
3. **All proof components deserialize correctly** ✓

### Stage 5 Sumcheck Debug Analysis

Using `--features zolt-debug`, got detailed comparison:

**Mismatch Found:**
- Zolt prover `output_claim`: `[ed, a5, f6, bf, 30, c4, 10, f8, ...]`
- Jolt verifier `expected_claim`: `[b2, 8f, 91, 24, 33, 0c, b4, 56, ...]`

**Stage 5 has 3 sumcheck instances:**
1. `RegistersValEvaluation`
2. `RamRaClaimReduction`
3. `InstructionReadRaf`

**Verifier's expected output claims (batched with coefficients):**
- Instance 0 claim*coeff: `[f2, 1e, ae, a8, ...]`
- Instance 1 claim*coeff: `[c3, 3a, cb, a7, ...]`
- Instance 2 claim*coeff: `[fe, 35, 18, c4, ...]`

## Root Cause Investigation

The issue is in how Zolt's Stage 5 prover computes the polynomial evaluations. Key areas to check:

### 1. RegistersValEvaluation
- `inc_claim`: `[39, 22, ab, 81, ...]`
- `wa_claim`: `[1f, 1c, 42, 45, ...]`
- `lt_eval`: `[f4, 1f, 17, b4, ...]`
- `result`: `[74, f7, 8e, 8c, ...]`

### 2. RamRaClaimReduction
- `ra_claim_reduced`: `[ef, 55, 4a, 31, ...]`
- `expected_output_claim`: `[c9, 1b, b9, ac, ...]`

### 3. InstructionReadRaf
- `ra_claim`: `[01, 93, 87, 0f, ...]`
- `raf_flag_claim`: `[c4, 03, 95, 05, ...]`
- `final_result`: `[02, ad, 67, 08, ...]`

## Next Steps for Next Session

1. **Add debug logging to Zolt prover** for Stage 5 instance evaluations
2. **Compare polynomial coefficient encoding** - check Montgomery form conversion
3. **Verify eq polynomial evaluation** - this is used in all 3 instances
4. **Check batching coefficient computation** - might differ from Jolt

## Key Files
- Zolt Stage 5 prover: `src/zkvm/proof_converter.zig` lines 2400+
- Jolt Stage 5 verifier: `jolt-core/src/zkvm/verifier.rs` verify_stage5()
- Jolt Stage 5 debug output: enabled with `--features zolt-debug`

## Test Commands
```bash
# Run with debug output
cd jolt && cargo test -p jolt-core --features zolt-debug --lib test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture
```

## Files
- `logs/zolt_proof_dory.bin`: 59,083 bytes
- `logs/zolt_preprocessing.bin`: 26,356 bytes

## Test Results
- 714/714 Zolt tests pass ✓
- Proof deserializes in Jolt ✓
- Stages 1-4 verification: pass (no error until Stage 5)
- **Stage 5: FAIL** - sumcheck output claim mismatch

SESSION_ENDING - Good progress made. Deserialization fully working, verification reaches Stage 5 with clear debug output showing the mismatch.
