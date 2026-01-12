# Zolt-Jolt Compatibility TODO

## Current Progress

### Stages 1-3: PASS
### Stage 4: IN PROGRESS - Batched Sumcheck Output Mismatch

## Stage 4 Investigation (Updated 2026-01-12)

### VERIFIED: normalize_opening_point Works Correctly

Added debug to Jolt's `normalize_opening_point` in `read_write_checking.rs`. Confirmed:
- T = 256, phase1_num_rounds = 8, phase2_num_rounds = 7
- sumcheck_challenges.len = 15
- r_cycle is correctly reconstructed (8 elements, non-zero)
- The function correctly reverses phase1 challenges to get r_cycle

Example output:
```
[NORMALIZE] T = 256, phase1_num_rounds = 8, phase2_num_rounds = 7
[NORMALIZE]   r_cycle[0] = [ac, a9, ed, b9, 97, 43, db, 44]
[NORMALIZE]   r_cycle[7] = [9a, 95, 69, 5e, 07, 04, 10, de]  (reversed sumcheck_challenges[0])
```

### CURRENT ISSUE: Batched Sumcheck Output Mismatch

**Symptom:**
```
output_claim:          4025718365397880246610377225086562173672770992931618085272964253447434290014
expected_output_claim: 12140057478814186156378889252409437120722846392760694384171609527722202919821
```

**Instance Breakdown (from Jolt verifier):**
- Instance 0 (RegistersReadWriteChecking): expected_claim = 4801..., coeff = 53772..., weighted = 12140...
- Instance 1 (RamValEvaluation): expected_claim = 0
- Instance 2 (RamValFinal): expected_claim = 0

**Key Observation:** The ratio expected/output ≈ 3x but not exactly. This suggests the round polynomial evaluations in Zolt prover differ from what Jolt verifier expects.

### Root Cause Hypothesis: Challenge Ordering in Sumcheck

Jolt uses **LowToHigh sparse sumcheck** binding:
- Round 0 binds the LAST variable (highest bit)
- Challenges are stored as [r_{n-1}, r_{n-2}, ..., r_0]
- normalize_opening_point reverses them to get [r_0, r_1, ..., r_{n-1}]

Zolt uses **dense sumcheck** binding:
- Round 0 binds the FIRST variable (lowest bit)
- Challenges are stored as [r_0, r_1, ..., r_{n-1}]
- When Jolt reverses, it gets WRONG order [r_{n-1}, ..., r_0]

This affects how eq(r_cycle_stage4, r_cycle_stage3) is computed in expected_output_claim.

### TODO: Next Steps

1. **Compare batching coefficients** between Zolt and Jolt
   - Check `[ZOLT] STAGE4: batching_coeff` vs Jolt's coefficients
   - They should match exactly (same transcript state → same coefficients)

2. **Check challenge ordering in Stage 4 prover**
   - Zolt stores challenges in `stage4_r_sumcheck[round_idx] = challenge`
   - This is [c0, c1, ..., c14] (HighToLow binding)
   - But sumcheck polynomial evaluation expects LowToHigh?

3. **Verify the eq polynomial evaluation**
   - In `expected_output_claim`: `eq(r_cycle_stage4, r_cycle_stage3)`
   - r_cycle_stage4 = reversed challenges from Stage 4
   - r_cycle_stage3 = challenges from Stage 3 (stored in params)
   - Are both in same endianness?

4. **Add debug to compare round-by-round**
   - Print each round's claim update in Zolt
   - Compare to what verifier expects at each round

### Files to Investigate

**Zolt:**
- `src/zkvm/proof_converter.zig` lines 1670-1822 (Stage 4 batched sumcheck)
- `src/zkvm/spartan/stage4_prover.zig` (RegistersReadWriteChecking prover)

**Jolt:**
- `jolt-core/src/zkvm/registers/read_write_checking.rs` (normalize_opening_point, expected_output_claim)
- `jolt-core/src/subprotocols/sumcheck.rs` (BatchedSumcheck::verify)

## Testing
```bash
bash scripts/build_verify.sh  # Output now goes to logs/ directory
```
