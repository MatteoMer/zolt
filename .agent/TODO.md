# Zolt-Jolt Compatibility TODO

## Current Progress

### Stages 1-3: PASS
### Stage 4: IN PROGRESS - Batching Implemented, Transcript Divergence Issue

## Stage 4 Investigation (Updated 2026-01-11)

### ROOT CAUSE IDENTIFIED: Missing Batching Coefficient - IMPLEMENTED

**Status:** Batching coefficient implemented in Stage 4 prover, but verification still fails due to transcript state divergence.

**What was done:**
1. Added `batching_coeff` parameter to Stage 4 prover
2. Prover now multiplies round polynomial coefficients by batching_coeff before appending to transcript
3. Prover tracks batched claims throughout sumcheck

### CURRENT ISSUE: Transcript State Divergence

**Observation:**
- Zolt's `r0` batching coefficient ≠ Jolt's `coeff_0`
- But the `input_claim` values are IDENTICAL:
  ```
  Zolt: rd_write_value = [58, 70, 2a, 66, ...] ✓
  Jolt: rd_wv_claim    = [58, 70, 2a, 66, ...] ✓

  Zolt: rs1_value = [61, 30, 4c, 9d, ...] ✓
  Jolt: rs1_rv_claim = [61, 30, 4c, 9d, ...] ✓

  Zolt: input_claim_registers = [6b, dc, 1e, a9, ...] ✓
  Jolt: result = [6b, dc, 1e, a9, ...] ✓
  ```
- And the `gamma` values are also IDENTICAL (first 64 bits):
  ```
  Zolt: 0x25d645b7560ee63c
  Jolt: 0x25d645b7560ee63c ✓
  ```

**Implication:**
Since gamma and input_claim match, but r0/coeff_0 don't match, the transcript state must diverge BETWEEN:
1. gamma derivation (after Stage 3)
2. Batching coefficient derivation (start of Stage 4 sumcheck)

**Possible Divergence Points:**
1. `verifier_accumulate_advice()` in Jolt - called after gamma, before BatchedSumcheck
   - This might append advice-related data to transcript for programs with advice
   - For fibonacci (no advice), this should be a no-op

2. RAM verifier initialization - Jolt creates RamValEvaluationSumcheckVerifier and ValFinalSumcheckVerifier
   - These don't take transcript as parameter, so shouldn't modify it

3. Different Stage 3 claim reduction handling?

### Debug Values

**Stage 4 Batching:**
```
Zolt r0 = { 117, 117, 84, 63, 252, 182, 234, 189 }
Jolt coeff_0 = 34501835073289124989986282317166852958 (0x19f4cfcd902ed538614231d1e6cebb5e)
```

**Stage 4 Sumcheck:**
```
output_claim = 7776653789547607390977439245232485355949181060166366194138714198419271251763
expected_output_claim = 3237708061191035719840323018238722061055447934202833244355840997888460025119
```

### Next Steps

1. **Add transcript hash debug** - Print transcript internal state (hash) in both Zolt and Jolt right before deriving batching coefficient
2. **Trace Stage 3 → Stage 4 transition** - Ensure all transcript messages match exactly
3. **Check for missing transcript appends** - Verify Zolt appends all the same data as Jolt between stages

### Code Changes Made

**Stage 4 Prover (`stage4_prover.zig`):**
- Added `batching_coeff: F` field to prover struct
- Modified `initWithClaims` to accept batching coefficient
- Prover multiplies round polynomial coefficients by batching_coeff before appending to transcript
- Prover tracks batched claims: `current_claim = batching_coeff * unbatched_claim`

**Proof Converter (`proof_converter.zig`):**
- Derives batching coefficients r0, r1, r2 from transcript (same pattern as Jolt)
- Passes r0 to Stage 4 prover

### Previous Fixes Still Valid

1. **r_cycle from Stage 3** - Correctly passed and stored in opening accumulator
2. **Opening claims** - Correctly serialized and retrieved
3. **eq polynomial computation** - Working correctly (params.r_cycle has correct non-zero values)
4. **gamma values** - Match between prover and verifier

### Previous Fixes Applied

1. **Sumcheck polynomial computation** - Now uses pointwise multiplication of univariate restriction evals
2. **Challenge ordering** - Stage 3 challenges passed correctly for claim computation
3. **Register values** - Rs1Value/Rs2Value set to 0 for instructions that don't read them
4. **K value** - Confirmed K=128 (32 RISC-V + 96 virtual registers)
5. **Batching coefficient** - Stage 4 prover now applies batching coefficient

## Testing
```bash
bash scripts/build_verify.sh
```
