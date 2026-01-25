# Actual Issue: Proof Serialization/Deserialization Mismatch

## Date: 2026-01-25

## Summary

The cross-verification failure at Stage 4 is **NOT** due to:
- ❌ Missing `execution_trace` or `memory_trace`
- ❌ RAM instances contributing non-zero when they shouldn't
- ❌ Stage 4 proof_converter code not executing

The issue **IS** due to:
- ✅ A mismatch in how Zolt serializes vs how Jolt deserializes the proof structure

## Evidence

### Zolt IS Generating Stage 4 Correctly

From the build logs:
```
[ZOLT STAGE4] Round 0: c0 = {...}, c2 = {...}
[ZOLT STAGE4] Round 1: c0 = {...}, c2 = {...}
...
[ZOLT STAGE4] Round 14: c0 = {...}, c2 = {...}
[ZOLT STAGE4] Final batched_claim = {...}
[ZOLT STAGE4 CLAIMS] val_claim bytes = {...}
[ZOLT STAGE4 CLAIMS] rd_write_value_claim = {...}
```

All 15 rounds are generated (correct for Fibonacci), and all opening claims are computed.

### Jolt's Verifier Reads Wrong Value

From the cross-verification logs:
```
[JOLT] STAGE1_FINAL: output_claim = 4025718365397880246610377225086562173672770992931618085272964253447434290014
...
[JOLT BATCHED] output_claim = 4025718365397880246610377225086562173672770992931618085272964253447434290014
[JOLT BATCHED] expected_output_claim (sum) = 12140057478814186156378889252409437120722846392760694384171609527722202919821
```

The `output_claim` used by Jolt's Stage 4 verifier is **identical** to Stage 1's final claim. This means Jolt is reading the wrong sumcheck proof or the proofs are misaligned.

## Hypothesis

There are several possibilities:

### 1. Missing or Extra Sumcheck Proofs

- Zolt might not be writing all the required sumcheck proofs (Stages 1-7)
- Or Zolt might be writing them in the wrong order
- Result: When Jolt tries to read Stage 4, it actually reads Stage 1's data

### 2. Incorrect Round Count

- Some stage might have the wrong number of rounds
- This could cause Jolt to read past the end of one proof and into the next
- Result: Deserialization offset error

### 3. Sumcheck Serialization Format Mismatch

- Zolt might be using a different compression or encoding
- Result: Jolt can't properly parse the structure

## Next Steps to Debug

1. **Check Stage Count**: Verify that all 7 stage sumcheck proofs are being written
   ```bash
   grep "stage[1-7]_sumcheck_proof" build log
   ```

2. **Verify Round Counts**: Check that each stage has the correct number of rounds:
   - Stage 1: Should match ProductSumcheck rounds
   - Stage 2: Should match batched instance max rounds
   - Stage 3: Should match n_cycle_vars
   - Stage 4: Should be 15 for Fibonacci (log2(256) + log2(128))
   - Stages 5-7: Various

3. **Add Serialization Debug**: Print the size/structure of each sumcheck proof before serialization

4. **Compare with Jolt**: Generate a Jolt proof for Fibonacci and compare the proof structure byte-by-byte

## Recommended Fix Approach

1. Add detailed logging to the proof serialization:
   ```zig
   std.debug.print("[SERIALIZE] Stage1 sumcheck: {} rounds\n", .{stage1_proof.compressed_polys.items.len});
   std.debug.print("[SERIALIZE] Stage2 sumcheck: {} rounds\n", .{stage2_proof.compressed_polys.items.len});
   ...
   ```

2. Compare with what Jolt expects for each stage

3. Fix any mismatches in round counts or proof structure

## Files to Investigate

- `src/zkvm/proof_converter.zig`: Where sumcheck proofs are generated
- `src/zkvm/mod.zig`: Where proofs are serialized
- `src/zkvm/serialization/jolt_proof_serialization.zig`: Serialization logic
