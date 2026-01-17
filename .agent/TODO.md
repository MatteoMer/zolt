# Zolt-Jolt Compatibility TODO

## Current Progress

| Stage | Status | Notes |
|-------|--------|-------|
| 1 | ✅ PASS | Univariate skip sumcheck |
| 2 | ✅ PASS | RWC now passes (removed synthetic termination write) |
| 3 | ✅ PASS | RegistersClaimReduction |
| 4 | ❌ FAIL | Sumcheck output_claim != expected_output_claim |
| 5-7 | ⏳ Blocked | Waiting for Stage 4 |

## Session 40 Summary (2026-01-17)

### Completed
1. ✅ Fixed Stage 2 RWC mismatch
   - Root cause: Zolt was adding synthetic termination write to memory trace
   - Jolt only sets termination bit in val_final, NOT in execution trace
   - Fix: Removed recordTerminationWrite() calls from tracer
   - Stage 2 now passes!

### Current Issue: Stage 4 RegistersReadWriteChecking

**Symptom:**
- Jolt verification fails at Stage 4
- output_claim = 19271728596168755243423895321875251085487803860811927729070795448153376555895
- expected_output_claim = 5465056395000139767713092380206826725893519464559027111920075372240160609265

**Analysis:**
- Stage 4 has 15 rounds (LOG_K=7 + log_T=8)
- Initial claim: 6327777426086953187456610298387046691641445481470372610958906907463930640269
- Zolt's eq_val and combined values MATCH Jolt's
- But final sumcheck claim doesn't match expected

**Key Debug Values:**
- Zolt eq_val_le matches Jolt eq_val bytes exactly
- Zolt combined matches Jolt combined bytes exactly
- The issue is in the sumcheck polynomial computation

**Hypothesis:**
The Stage 4 prover polynomial computation might have a bug. Even though
individual claims match, the round polynomial evaluations might be wrong.

### Next Steps
1. Compare Stage 4 round 0 coefficients between Zolt and Jolt
2. Trace through polynomial evaluation for first few rounds
3. Check if eq polynomial is applied correctly in round evaluation

## Commit History
- 5cec222: fix: remove synthetic termination write from memory trace (Stage 2 fix)
