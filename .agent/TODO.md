# Zolt-Jolt Compatibility: Current Status

## Status: Debugging Stage 3 Initial Claim ⏳

## Session Summary (2026-01-29)

### Key Findings

**714/714 Unit Tests Pass!** All Zolt unit tests are passing.

**Stage 1 is CORRECT!** Verified in previous sessions.

**Stage 3 Flow Verified:**
- Transcript state at Stage 3 boundary: `{ 218, 190, 38, 231, ... }` matches expected
- Individual input_claim formulas match Jolt exactly:
  - ShiftSumcheck: `NextUnexpandedPC + γ*NextPC + γ²*NextIsVirtual + γ³*NextIsFirst + γ⁴*(1-NextIsNoop)`
  - InstructionInput: `(RightOuter + γ*LeftOuter) + γ²*(RightProduct + γ*LeftProduct)`
  - RegistersClaimReduction: `RdWriteValue + γ*Rs1Value + γ²*Rs2Value`
- Batching coefficient derivation matches Jolt (uses `challenge_scalar` = full 128-bit)
- Opening claims are stored and retrieved correctly

### Current Investigation

The Stage 3 initial_claim mismatch may be due to:
1. **Outdated TODO notes** - The expected Jolt value `[da, a3, 24, 84, ...]` needs verification
2. **r_cycle evaluation point** - Verify the MLE evaluation point matches what Jolt expects
3. **Need Jolt verifier** - Without running Jolt's verifier, we can't confirm the actual error

### Technical Details

**Zolt Stage 3 Values:**
```
transcript_state: { 218, 190, 38, 231, 136, 156, 76, 190 }
input_claim[0] (Shift): { 30, 39, 195, 164, 59, 14, 143, 21, ... }
input_claim[1] (InstrInput): { 90, 193, 204, 241, 164, 156, 192, 62, ... }
input_claim[2] (Registers): { 75, 0, 96, 142, 99, 112, 107, 174, ... }
batching_coeff[0]: { 40, 209, 4, 96, 132, 232, 161, 190, ... }
current_claim (ROUND_0): { 7, 141, 69, 74, 209, 62, 182, 63, ... }
```

**Individual instance verification (ROUND_0):**
- shift_p0+p1 = shift_claim ✓
- instr_p0+p1 = instr_claim ✓
- reg_p0+p1 = reg_claim ✓

### Next Steps

1. **Install Dependencies**: Need pkg-config/libssl-dev to run Jolt verification
2. **Run Jolt Verifier**: Get actual error message from Jolt
3. **Compare with fresh Jolt run**: Generate a reference proof with Jolt and compare byte-by-byte

## Completed

- [x] Implemented Stage 1 UniSkip + Remaining sumcheck prover
- [x] Verified Stage 1 matches Jolt exactly
- [x] Implemented Stage 3 RegistersClaimReduction prover
- [x] Verified Stage 3 input_claim formulas match Jolt
- [x] Verified Stage 3 batching coefficient derivation matches Jolt
- [x] All 714 unit tests passing

## In Progress

- [ ] Run Jolt verifier to get actual error
- [ ] Compare proof bytes with Jolt-generated reference
