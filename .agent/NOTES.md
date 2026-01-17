# Zolt-Jolt Cross-Verification Progress

## Session 41 Summary - Montgomery Fix & Serialization (2026-01-17)

### Major Accomplishments

1. **Fixed Stage 4 Montgomery Conversion** ✅
   - Root cause: Jolt's MontU128Challenge stores [0, 0, L, H] as BigInt representation
   - When converted to Fr: represents 2^128 * original_value (NOT the original 125-bit value)
   - OLD Zolt behavior: directly stored [0, 0, L, H] as Montgomery limbs (WRONG)
   - FIX: Store [0, 0, L, H] as standard form, then call toMontgomery()
   - Result: All 6 Zolt internal verification stages now PASS

2. **Fixed Proof Serialization Format** ✅
   - Issue: Using `--jolt` flag instead of `--jolt-format`
   - Issue: Proof was missing the claims count header
   - Result: Jolt can now deserialize Zolt proofs successfully

### Current Blocker: Commitment Mismatch

Cross-verification fails at Stage 1 because:
- Zolt and Jolt produce **different Dory commitments** for the polynomials
- Commitments are appended to the Fiat-Shamir transcript
- Different commitments → different transcript states → different challenges
- Different challenges → sumcheck verification fails

**Error:**
```
output_claim:          14184556905709553188252266513419299337140543956476510328335642523536328101946
expected_output_claim: 4850043052968955019670601823786826972287752356873852008418437649167667199602
```

### Next Steps to Complete Cross-Verification

For full cross-verification to work, Zolt needs to:
1. **Option A**: Implement Dory commitment scheme that produces identical GT elements
   - Multi-scalar multiplication with G1/G2 points from SRS
   - Multi-pairing computation to produce GT commitment
2. **Option B**: Use shared commitment data
   - Have Jolt export commitments, Zolt reuses them
   - Only proves sumcheck execution matches, not commitment computation

### Stage Status After Fixes

| Stage | Internal (Zolt) | Cross-verify (Jolt) | Notes |
|-------|-----------------|---------------------|-------|
| 1 | ✅ PASS | ❌ Commit mismatch | Transcript diverges from commitments |
| 2 | ✅ PASS | - | - |
| 3 | ✅ PASS | - | - |
| 4 | ✅ PASS | - | Montgomery fix applied |
| 5 | ✅ PASS | - | - |
| 6 | ✅ PASS | - | - |

---

## Session 40 Summary - Stage 4 Investigation (2026-01-17)

### Stage 2 Fix
Removed synthetic termination write from memory trace. In Jolt, the termination bit
is set directly in val_final during OutputSumcheck, NOT in the execution/memory trace.
The RWC sumcheck only includes actual LOAD/STORE instructions.

### Stage 4 Deep Investigation

#### Verified Matches
1. **Transcript state**: IDENTICAL between Zolt and Jolt at all checkpoints
2. **Challenge bytes**: IDENTICAL (f5 ce c4 8c b0 64 ba b5 ce 4d a4 2a db 38 f8 ac)
3. **Input claims**: ALL THREE match exactly
4. **Batching coefficients**: MATCH
5. **Polynomial coefficients in proof**: MATCH

### Root Cause Analysis

The challenge in Zolt was stored as:
```zig
result = F{ .limbs = .{ 0, 0, masked_low, masked_high } };
```

This is NOT proper Montgomery form. Jolt's MontU128Challenge stores the same bytes
but interprets them as a BigInt that gets converted to Montgomery via from_bigint_unchecked.

**Fix applied in Session 41**: Convert properly using toMontgomery().

---

## Previous Sessions

### Stage 3 Fix (Session 35)
- Fixed prefix-suffix decomposition convention (r_hi/r_lo)

### Stage 1 Fix
- Fixed NextPC = 0 issue for NoOp padding

---

## Technical References

- Jolt MontU128Challenge: `jolt-core/src/field/challenge/mont_ark_u128.rs`
- Jolt BatchedSumcheck verify: `jolt-core/src/subprotocols/sumcheck.rs:180`
- Zolt Blake2b transcript: `src/transcripts/blake2b.zig`
- Zolt Stage 4 proof: `src/zkvm/proof_converter.zig` line ~1700
