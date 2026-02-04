# Zolt-Jolt Compatibility Implementation

## Status: Session 51+ - Stage 4 Transcript Divergence

## Current Investigation - Stage 4 Verification Failure

### What We Know

1. **Stage 3 passes** - The r_cycle values from Stage 3 match correctly
2. **evaluateCubicAtChallengeFromEvals FIXED** - Now uses toomCookToCoeffs instead of wrong Lagrange points
3. **Round polynomial coefficients MATCH** - Zolt's c0, c2, c3 match what Jolt reads for Round 0
4. **But challenges DIVERGE** - Zolt computes different challenge from same coefficients

### The Root Cause

The transcript state at Stage 4 start is different between Zolt and Jolt:
- Jolt Stage 4 batching coeff[0]: `[53, dd, 21, 20, ...]`
- Zolt Stage 4 batching coeff[0]: `[4b, fe, 92, ...]` (different!)

This means the transcript diverged BEFORE Stage 4's first round polynomial was appended.

### Possible Sources of Divergence

1. **Stage 3 sumcheck round polynomials** - might have different coefficients
2. **Stage 3 final claims** - might be different
3. **Input claims for Stage 4** - Already verified they match for RegistersRWC input

### Key Finding: Coefficients Match But Transcript Diverges

Looking at Jolt's Stage 4 Round 0 coefficients:
```
[0]: [2d, 33, 27, 97, 4c, 65, 6f, 11, ...] = c0
[1]: [c4, 63, ef, aa, 26, a1, 4c, 9f, ...] = c2
[2]: [bf, 56, 16, 45, 1d, c8, e6, e3, ...] = c3
```

Zolt's coefficients (from full_coeffs):
- c0 = [2d, 33, 27, 97, ...] ✓ MATCHES
- c2 = [c4, 63, ef, aa, ...] ✓ MATCHES

But the challenges differ because the transcript state BEFORE appending these coefficients is already different.

### Next Steps

1. Check Stage 3's round polynomial coefficients - do they match?
2. Check Stage 3's output claims - do they match?
3. Find where transcript diverges between Zolt and Jolt
4. Fix the divergence source

### Test Commands

```bash
# Build Zolt
zig build -Doptimize=ReleaseFast

# Generate proof
./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format -o /tmp/zolt_proof_dory.bin --export-preprocessing /tmp/zolt_preprocessing.bin --trace-length 64

# Verify with Jolt
cd /home/vivado/projects/jolt && cargo test -p jolt-core --features zolt-debug --lib test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture
```

## Completed This Session

1. Fixed `evaluateCubicAtChallengeFromEvals` - was using wrong Lagrange interpolation points (0,1,2,3) instead of Toom-Cook format (p(0), p(1), p(2), p_inf)
2. Verified Stage 4 Round 0 coefficients match what Jolt expects
3. Identified that transcript divergence happens BEFORE Stage 4 Round 0

## SESSION_ENDING

Context running low. The key finding is that the round polynomial coefficients themselves are correct, but the transcript state has already diverged before Stage 4 starts. Next session should focus on finding where the transcript divergence originates (likely in Stage 3).
