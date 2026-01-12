# Zolt-Jolt Compatibility TODO

## Current Progress

### Stages 1-3: PASS
### Stage 4: IN PROGRESS - Input Claim Value Mismatch Identified

## Stage 4 Investigation (Updated 2026-01-11)

### NEW ROOT CAUSE: Input Claim Values Differ Beyond First 8 Bytes

**Status:** The batching coefficient derivation order was fixed, but the actual input_claim values being appended to the transcript differ between Zolt and Jolt.

**Latest Finding:**
- The first 8 bytes of component claims (rd_write_value, rs1_value, rs2_value) match
- The gamma values match (0x25d645b7560ee63c)
- BUT the full 32-byte input_claim values differ when appended to transcript:
  ```
  Zolt claim BE first 8:  [0x29, 0x14, 0x12, 0xa5, 0x6a, 0x0e, 0x23, 0x80]
  Jolt claim BE first 8:  [0x00, 0x19, 0x76, 0x26, 0x3d, 0x11, 0xdb, 0x57]
  ```

**Implication:**
The full 32-byte claim values (rd_write_value, rs1_value, rs2_value) likely differ beyond their first 8 bytes. Zolt's Stage 3 claims appear to be larger values (non-zero MSBs), while Jolt's claims may be smaller values (leading zeros in BE).

**Investigation Areas:**
1. Check if Zolt's MLE evaluations at Stage 3 challenges are computing correct claim values
2. Verify the full 32-byte claim values match between Zolt and Jolt
3. Confirm Montgomery form handling is consistent

### Previous Fixes Applied

1. **Batching coefficient derivation order** - Fixed to append ALL claims first, then derive ALL coefficients (matching Jolt's BatchedSumcheck)
2. **Stage 3 cache_openings** - Already handled by Stage 3 prover (no duplicate appends needed)
3. **Transcript implementation** - Confirmed to match Jolt's (Montgomery→Canonical→LE→reverse to BE)

### Code Structure Understanding

**Stage 3 → Stage 4 Flow in Jolt:**
1. Stage 3 BatchedSumcheck completes
2. cache_openings() called for each Stage 3 verifier (16 appends total)
3. Stage 4 gamma derived (RegistersReadWriteCheckingParams::new)
4. verifier_accumulate_advice() - no-op for Fibonacci
5. Stage 4 BatchedSumcheck: append 3 input claims, derive 3 batching coefficients

**Zolt's Current Implementation:**
1. Stage 3 prover generates round polys and appends 16 cache_openings claims
2. Stage 4 gamma derived
3. Compute input_claim = rd_write_value + gamma*rs1_value + gamma²*rs2_value
4. Append 3 claims (input_claim, 0, 0), derive 3 coefficients (r0, r1, r2)

### Debug Values (Current)

**Stage 4 Input Claims (after computation):**
```
Zolt input_claim LE first 8: [6b, dc, 1e, a9, c7, be, 21, 90]
Jolt input_claim result:     [6b, dc, 1e, a9, c7, be, 21, 90] ✓ (match!)
```

**BUT when appended to transcript (full 32-byte BE):**
```
Zolt: starts with [0x29, 0x14, 0x12, ...]
Jolt: starts with [0x00, 0x19, 0x76, ...]
```

This discrepancy suggests the FULL claim values differ even though the first 8 LE bytes match.

### Next Steps

1. **Print full 32-byte claim values** in both Zolt and Jolt to compare
2. **Check if Jolt's claims are small values** (fitting in <128 bits, hence leading zeros)
3. **Verify Zolt's Stage 3 MLE evaluations** are computing correct claim values
4. **Compare Montgomery representations** if necessary

## Testing
```bash
bash scripts/build_verify.sh
```
