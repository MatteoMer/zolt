# Zolt-Jolt Compatibility: Stage 3 Initial Claim Debug

## Status: Stage 3 initial_claim Mismatch ⏳

## Session Summary (2026-01-29)

### Key Finding

**Stage 1 is CORRECT!** Zolt's Stage 1 initial_claim and round coefficients match Jolt exactly:
- Stage 1 initial_claim: `db b1 f8 a9 eb ed 61 41 ac 8c fb 3c 60 1a 7a bc ...` ✓
- Stage 1 round 0 c0: `11 f5 09 43 df 7c 85 e0 ...` ✓
- Stage 1 round 0 c2: `15 0e c5 ba cb f8 0a 6b ...` ✓
- Stage 1 round 0 c3: `ed c8 c8 65 78 27 38 6e ...` ✓

### Current Issue: Stage 3 Initial Claim Mismatch

The Stage 3 initial_claim differs between Zolt and Jolt:
- **Jolt's verifier**: `[da, a3, 24, 84, db, 59, f8, 88, ...]`
- **Zolt's prover**: `[07, 8d, 45, 4a, d1, 3e, b6, 3f, ...]`

### Analysis

The Stage 3 initial_claim is computed as:
```
batched_claim = Σ coeff[i] * input_claim[i]
```

Where:
- `input_claim[0]` = SpartanShift input claim (from opening claims)
- `input_claim[1]` = InstructionInputVirtualization input claim
- `input_claim[2]` = RegistersClaimReduction input claim
- `coeff[i]` = batching coefficients from transcript

The mismatch could be due to:
1. Different input claims from Stage 2 opening claims
2. Different transcript state when sampling batching coefficients
3. Different batching coefficient computation

### Debug Values from Zolt Stage 3

```
input_claim[0] (Shift): { 30, 39, 195, 164, 59, 14, 143, 21, ... } = 1e 27 c3 a4 3b 0e 8f 15 ...
input_claim[1] (InstrInput): { 90, 193, 204, 241, 164, 156, 192, 62, ... } = 5a c1 cc f1 a4 9c c0 3e ...
input_claim[2] (Registers): { 75, 0, 96, 142, 99, 112, 107, 174, ... } = 4b 00 60 8e 63 70 6b ae ...
batching_coeff[0]: { 40, 209, 4, 96, 132, 232, 161, 190, ... } = 28 d1 04 60 84 e8 a1 be ...
```

### Next Steps

1. Compare transcript states between Stage 2 end and Stage 3 start
2. Verify Stage 2 opening claims are being stored correctly
3. Check batching coefficient computation matches Jolt's approach
4. Trace the input_claim values through Stage 2 verification

## Completed

- [x] Implemented Stage 1 UniSkip + Remaining sumcheck prover
- [x] Verified Stage 1 matches Jolt exactly
- [x] Implemented Stage 3 RegistersClaimReduction prover
- [x] Traced Stage 3 initial_claim mismatch

## In Progress

- [ ] Debug Stage 3 input claim computation
- [ ] Verify Stage 2 opening claims storage
- [ ] Fix transcript state consistency
