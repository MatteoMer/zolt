# Zolt-Jolt Compatibility TODO

## Current Progress

### Stages 1-3: PASS
### Stage 4: IN PROGRESS - Claims Match, Batched Sumcheck Diverges

## Stage 4 Investigation (Updated 2026-01-12)

### FIXED: Eq Polynomial Endianness Issue

**Root Cause:** The eq polynomial was changed from little-endian to big-endian ordering, with corresponding challenge reversal. This broke the Stage 4 sumcheck because the polynomial indexing didn't match the execution trace indexing.

**Fix Applied:**
1. Reverted `computeEqEvals` to LE ordering (r[0] binds LSB of index)
2. Removed challenge reversal - pass Stage 3 challenges directly

**Result:** Stage 4 claims now match perfectly:
```
[STAGE4] Simple rd_wv MLE sum = { 218, 190, 87, 33, ... }
[STAGE4] Expected rd_wv from Stage 3 = { 218, 190, 87, 33, ... } ✓ MATCH

[STAGE4] Simple rs1_v MLE sum = { 4, 11, 232, 188, ... }
[STAGE4] Expected rs1_v from Stage 3 = { 4, 11, 232, 188, ... } ✓ MATCH

[STAGE4] Simple rs2_v MLE sum = { 115, 117, 41, 122, ... }
[STAGE4] Expected rs2_v from Stage 3 = { 115, 117, 41, 122, ... } ✓ MATCH
```

### CURRENT ISSUE: Batched Sumcheck Output Mismatch

**Symptom:**
```
output_claim:          4025718365397880246610377225086562173672770992931618085272964253447434290014
expected_output_claim: 12140057478814186156378889252409437120722846392760694384171609527722202919821
```

**Likely Cause:** Transcript state divergence when deriving batching coefficients. The batching coefficients (coeff_0, coeff_1, coeff_2) are derived from the transcript after appending input claims. If the transcript state differs between Zolt prover and Jolt verifier at this point, the coefficients will differ.

**Investigation Areas:**
1. Compare transcript state before deriving batching coefficients
2. Verify input_claim values appended to transcript match exactly (full 32-byte BE)
3. Check if RamValEvaluation/RamValFinal input claims (currently 0) should be non-zero

### Key Insight from Debugging

The previous hypothesis about "input claim values differ beyond first 8 bytes" was misleading. The actual issue was:
- The eq polynomial bit ordering (LE vs BE) determines how indices map to challenge points
- Stage 3 sumcheck binds variables in LE order (first challenge binds LSB)
- Stage 4's eq polynomial must use the same ordering to correctly evaluate eq(r_cycle, j)

### Code Structure

**Eq Polynomial Ordering:**
- LE (correct): `r[0]` corresponds to LSB of index, `r[n-1]` to MSB
- BE (incorrect for our use): `r[0]` corresponds to MSB of index

**Stage 3 → Stage 4 Challenge Flow:**
1. Stage 3 sumcheck produces challenges `[r_0, r_1, ..., r_{n-1}]`
2. `r_0` was bound first (to dimension 0, which is LSB)
3. Stage 4 uses these challenges directly with LE eq polynomial

## Testing
```bash
bash scripts/build_verify.sh
```
