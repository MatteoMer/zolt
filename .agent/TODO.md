# Zolt-Jolt Compatibility TODO

## Current Progress

| Stage | Status | Notes |
|-------|--------|-------|
| 1 | ✅ PASS | Univariate skip sumcheck |
| 2 | ✅ PASS | RWC - removed synthetic termination write |
| 3 | ✅ PASS | RegistersClaimReduction |
| 4 | ✅ PASS | **FIXED** - Montgomery conversion fix |
| 5 | ✅ PASS | RegistersValEvaluation |
| 6 | ✅ PASS | RAM evaluation |

## Session 41 Progress (2026-01-17)

### MAJOR FIX: Stage 4 Montgomery Conversion
1. ✅ Fixed Stage 4 sumcheck verification failure
   - **Root cause:** Jolt's MontU128Challenge stores [0, 0, low, high] as a BigInt representation
   - When converted to Fr via `from_bigint_unchecked(BigInt([0, 0, L, H]))`:
     - BigInt([0, 0, L, H]) represents L*2^128 + H*2^192 (NOT L + H*2^64!)
     - This is then converted to proper Montgomery form
   - **OLD Zolt behavior:** Directly stored [0, 0, low, high] as Montgomery limbs
     - This represented a DIFFERENT value ((0 + 0*2^64 + L*2^128 + H*2^192) / R mod p)
   - **FIX:** Store [0, 0, low, high] as STANDARD form, then call toMontgomery()
     - This matches Jolt's from_bigint_unchecked behavior exactly
   - Commit: 54200fa

2. ✅ All Zolt internal stages now pass (1-6)
   ```
   [VERIFIER] Stage 1 PASSED
   [VERIFIER] Stage 2 PASSED
   [VERIFIER] Stage 3 PASSED
   [VERIFIER] Stage 4 PASSED
   [VERIFIER] Stage 5 PASSED
   [VERIFIER] Stage 6 PASSED
   [VERIFIER] All stages PASSED!
   ```

### Remaining Issue: Proof Serialization
- Jolt proof: 66761 bytes (/tmp/fib_proof.bin)
- Zolt proof: 11345 bytes (/tmp/zolt_proof_dory.bin)
- The Zolt proof is much smaller - missing data for Jolt deserialization
- Error: "Not enough bytes for field element!" at claim 340 (offset 11342)
- This is a SEPARATE issue from the sumcheck verification

## Next Steps
1. Fix proof serialization to match Jolt's expected format
   - Compare JoltProof structure between Jolt and Zolt
   - Ensure all commitments, opening claims, and stage proofs are serialized
   - Check sumcheck proof coefficient count and format
2. Run cross-verification test again after serialization fix
3. Verify all 578+ tests pass with the Montgomery fix

## Session 40 Progress (2026-01-17)

### Completed
1. ✅ Fixed Stage 2 RWC mismatch
   - Root cause: Zolt was adding synthetic termination write to memory trace
   - Jolt only sets termination bit in val_final, NOT in execution trace
   - Fix: Removed recordTerminationWrite() calls from tracer
   - Commit: 5cec222

## Commit History
- 54200fa: fix: convert challenge scalar to proper Montgomery form (Stage 4 fix)
- 5cec222: fix: remove synthetic termination write from memory trace (Stage 2 fix)
- 51b5f1b: docs: update TODO with Stage 4 investigation progress
- ad6eb9d: docs: update notes with Stage 4 investigation findings
