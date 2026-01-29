# Zolt-Jolt Compatibility: Current Status

## Status: Deserialization Complete, Verification In Progress ⏳

## Session 73 Summary (2026-01-29)

### Major Fix: SumcheckId Mismatch

**Problem Found:**
- Zolt had 24 SumcheckId values, Jolt has 22
- This caused OpeningId base offsets to be wrong:
  - Zolt: 0/24/48/72 (UNTRUSTED/TRUSTED/COMMITTED/VIRTUAL)
  - Jolt: 0/22/44/66

**Fix Applied:**
- Removed `AdviceClaimReductionCyclePhase` and `AdviceClaimReduction`
- Renumbered `IncClaimReduction` to 20, `HammingWeightClaimReduction` to 21
- COUNT now 22 to match Jolt exactly

### Proof Serialization Fixed

**Problems Found:**
1. Missing 4 advice proof options (only had 1 for commitment)
2. Configuration written as mix of u8 and usize, should be 5 usizes

**Fixes Applied:**
- Added all 5 advice proof options (4 proofs + 1 commitment, all None)
- Changed configuration to exactly 5 usizes matching Jolt's struct

### Current State

**Proof Deserialization: COMPLETE ✓**
```
=== Step 1: Claims ===
OK: 91 claims

=== Step 2: Commitments (Vec<GT>) ===
OK: 37 commitments

=== Steps 3-11: Sumcheck Stages ===
All OK

=== Step 12: Dory Opening Proof ===
OK: 5 rounds, nu=4, sigma=5

=== Steps 13-17: Advice Proofs & Commitment ===
All OK (None values)

=== Steps 18-22: Configuration ===
trace_length: 256
ram_K: 65536
bytecode_K: 65536
log_k_chunk: 4
lookups_ra_virtual_log_k_chunk: 16

=== COMPLETE ===
Final position: 40544, Total: 40544
```

**Verification: FAILS at Stage 2**
```
output_claim:          21381532812498647026951017256069055058409470421711163232531942150439292669264
expected_output_claim: 7589737359806175897404235347050845364246073571786737297475678711983129582270
```

### Next Steps

The proof deserializes completely but verification fails at Stage 2 sumcheck. This means:

1. **The sumcheck polynomial values in the proof don't match verifier expectations**
2. This could be caused by:
   - Transcript state mismatch between Zolt prover and Jolt verifier
   - Incorrect polynomial computations in Stage 2
   - Wrong claims being accumulated

To debug:
1. Add transcript state logging to Zolt's Stage 2 prover
2. Compare challenge derivation at each sumcheck round
3. Trace the batched claims and coefficients

### Files Modified This Session

1. `src/zkvm/jolt_types.zig` - Fixed SumcheckId enum (22 values)
2. `src/zkvm/mod.zig` - Fixed serialization:
   - All 5 advice proof options
   - Configuration as 5 usizes

## Previous Sessions

### Session 72 (2026-01-28)
- 714/714 unit tests passing
- Stage 3 sumcheck verified mathematically correct
- Opening claims storage/retrieval verified

### Session 71 (2026-01-28)
- Instance 0 (RegistersRWC) verified correct
- Synthetic termination write discovery
