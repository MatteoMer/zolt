# Zolt-Jolt Compatibility: Stage 3 Initial Claim Debug

## Status: Stage 3 initial_claim Mismatch ⏳

## Session Summary (2026-01-29)

### Key Findings

**Stage 1 is CORRECT!** Zolt's Stage 1 matches Jolt exactly:
- Stage 1 initial_claim: `db b1 f8 a9 eb ed 61 41 ac 8c fb 3c 60 1a 7a bc ...` ✓
- Stage 1 round 0 coefficients match ✓
- UniSkip polynomial and challenges computed correctly ✓

### Current Issue: Stage 3 Initial Claim Mismatch

The Stage 3 initial_claim differs:
- **Jolt's verifier**: `[da, a3, 24, 84, db, 59, f8, 88, ...]`
- **Zolt's prover**: `[07, 8d, 45, 4a, d1, 3e, b6, 3f, ...]`

The batched initial_claim is computed as:
```
batched_claim = Σ coeff[i] * input_claim[i]
```

Where Stage 3 instances are:
- `input_claim[0]` = SpartanShift (from SpartanOuter opening claims)
- `input_claim[1]` = InstructionInputVirtualization (from SpartanOuter opening claims)
- `input_claim[2]` = RegistersClaimReduction (from SpartanOuter opening claims)

### Debug Data

**Zolt Stage 3 Pre Values:**
```
transcript_state: { 218, 190, 38, 231, ... } = da be 26 e7 ...
input_claim[0] (Shift): { 30, 39, 195, 164, ... } = 1e 27 c3 a4 ...
input_claim[1] (InstrInput): { 90, 193, 204, 241, ... } = 5a c1 cc f1 ...
input_claim[2] (Registers): { 75, 0, 96, 142, ... } = 4b 00 60 8e ...
batching_coeff[0]: { 40, 209, 4, 96, ... } = 28 d1 04 60 ...
```

### Root Cause Hypothesis

The mismatch could be due to:
1. Different opening claims from Stage 2 (stored vs computed by verifier)
2. Different transcript state when sampling Stage 3 batching coefficients
3. Incorrect Stage 3 input_claim retrieval from opening claims

### Next Steps

1. **Compare opening claims**: Add debug to show what Jolt reads from the proof vs what Zolt stored
2. **Compare transcript states**: Verify transcript is identical at Stage 2 end / Stage 3 start
3. **Verify input_claim formula**: Check that Stage 3 instances use correct opening claim lookups
4. **Install dependencies**: Need pkg-config/libssl-dev to run Jolt verification tests

## Completed

- [x] Implemented Stage 1 UniSkip + Remaining sumcheck prover
- [x] Verified Stage 1 matches Jolt exactly
- [x] Implemented Stage 3 RegistersClaimReduction prover
- [x] Traced Stage 3 initial_claim mismatch
- [x] Collected detailed debug data for Stage 3

## In Progress

- [ ] Debug Stage 3 input_claim computation
- [ ] Compare what opening claims Jolt verifier reads
- [ ] Verify transcript consistency at Stage 3 boundary
