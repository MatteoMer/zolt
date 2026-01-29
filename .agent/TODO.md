# Zolt-Jolt Compatibility: Stage 3/4 Challenge Debug

## Status: Investigating Stage 3 Input Claim Mismatch ⏳

## Session Summary (2026-01-29)

### Key Finding

The Stage 4 sumcheck fails because `params.r_cycle` (Stage 3 challenges) doesn't match between Zolt prover and Jolt verifier.

### Root Cause Chain

1. **Stage 3 initial claims differ**:
   - Jolt's verifier computes Stage 3 initial_claim: `[da, a3, 24, 84, db, 59, f8, 88, ...]`
   - Zolt's prover produces Stage 3 current_claim: `[07, 8d, 45, 4a, d1, 3e, b6, 3f, ...]`

2. **Stage 3 input claims come from Stage 1's SpartanOuter**:
   - `RegistersClaimReduction.input_claim` uses:
     - `RdWriteValue` @ `SpartanOuter`
     - `Rs1Value` @ `SpartanOuter`
     - `Rs2Value` @ `SpartanOuter`
   - These are computed from R1CS evaluations at Stage 1's r_cycle

3. **Stage 1 initial_claim mismatch**:
   - Jolt expects: `[db, b1, f8, a9, eb, ed, 61, 41, ...]` with 9 rounds
   - Zolt produces: `0x0` (all zeros!) with 13 rounds
   - **This is suspicious** - zero initial claim suggests the R1CS sum is zero

### Analysis

The Zolt Stage 1 sumcheck produces all-zero round polynomials because:
- Either Az ⊙ Bz = Cz for all rows (R1CS is trivially satisfied everywhere)
- Or there's a bug in how the R1CS evaluation polynomial is computed

But Jolt expects a NON-ZERO initial claim, which means Jolt's prover computes a non-zero R1CS sum.

### Comparison Data

#### Zolt Stage 3 r_cycle (passed to Stage 4):
```
r_cycle_be[0] = { 00...00 4e 55 11 b5 4e 49 62 7f 3d e9 51 ee bb 38 23 1f }
r_cycle_be[1] = { 00...00 38 65 34 87 6f 99 63 13 2d d6 e8 95 cf 0d 32 17 }
...
```

#### Jolt's verifier params.r_cycle (from Stage 3):
```
params.r_cycle[0]: [00..00, 3c, 22, 3b, 5f, f7, 37, 45, fa, a6, 03, e4, e5, bc, ec, c8, 18]
params.r_cycle[1]: [00..00, b3, bc, 28, 96, 5b, 3c, ff, 82, 63, f2, ea, 64, d9, c6, 89, 00]
...
```

These are completely different because the Stage 3 sumcheck produced different round polynomials.

### Next Steps

1. **Debug Stage 1 R1CS polynomial**:
   - Verify Az, Bz, Cz are computed correctly
   - Check the eq polynomial is initialized properly with tau challenges
   - Ensure the combined polynomial f(x) = Σ eq(τ,x) * (Az(x) * Bz(x) - Cz(x)) is correct

2. **Verify transcript consistency**:
   - Compare transcript states at each stage between Zolt and Jolt
   - Ensure commitments are appended in identical order

3. **Compare Stage 1 output**:
   - The final claim after Stage 1 sumcheck
   - The r_cycle challenges produced

## Completed

- [x] Implemented Stage 4 RegistersReadWriteChecking with Gruen optimization
- [x] Implemented Stage 3 RegistersClaimReduction prover
- [x] Added extensive debug output
- [x] Identified that Stage 1 initial_claim = 0 in Zolt but non-zero in Jolt
- [x] Traced the claim mismatch back to Stage 1

## In Progress

- [ ] Debug Stage 1 R1CS polynomial evaluation
- [ ] Fix the zero initial_claim issue
- [ ] Verify transcript consistency
