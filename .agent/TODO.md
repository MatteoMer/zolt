# Zolt-Jolt Compatibility Implementation

## Status: Session 38 - Stage 4 PASSES! Stage 5 Investigation

## Major Progress!

**Stage 4 is now passing!** Verification progresses to Stage 5 (InstructionReadRaf), which fails.

## Current Failure: Stage 5 (InstructionReadRaf)

### Verification Details
```
Sumcheck verification failed!
  output_claim:   [af, 51, 7b, 30, ff, 29, 91, 26, 92, 26, 11, 23, ...]
  expected_claim: [f0, c1, c7, e7, 7e, fd, c3, 3b, f3, 7e, 52, 31, ...]
Verification failed: Stage 5
```

### Good News: r_reduction and ra_claims MATCH!

Confirmed matching values between Zolt and Jolt:

**r_reduction[0]** (from Stage 2 InstructionClaimReduction):
- Jolt: `[0d, 8d, 89, b0, c0, ef, 00, b0, 84, a4, 8a, 1b, 0b, 14, 34, 07]`
- Zolt: `limbs = [b000efc0b0898d0d, 0734140b1b8aa484]` → Same bytes! ✓

**ra_claims[0]** (InstructionRa(0) at InstructionReadRaf):
- Both: `[69, 16, 50, 9b, b0, 0d, 7a, 4e, 25, 9d, b6, 8b, 53, 6e, 2b, 3d, ...]` ✓

### Stage 5 Instance Details

The expected_claim is computed as sum of:
- **Instance 0** (RegistersValEvaluation): `36 6f 00 34 ...` * coeff[0]
- **Instance 1** (RamRaClaimReduction): `48 d9 da ee ...` * coeff[1]
- **Instance 2** (InstructionReadRaf): `73 79 ec b4 ...` * coeff[2]

### Remaining Issue

The **output_claim** (from evaluating sumcheck polynomials at challenges) doesn't match **expected_claim** (from computing individual instance claims).

This suggests either:
1. Round polynomial coefficients differ between prover/verifier
2. Challenge values diverge at some round

### Next Investigation Steps

1. **Compare Stage 5 round 0 polynomial**
   - Zolt: What coefficients are generated?
   - Jolt: What coefficients are read from proof?

2. **Check transcript state at Stage 5 start**
   - Both should have identical state after Stage 4 claims are appended

3. **Compare challenges at rounds 128-135** (cycle variables)
   - These determine r_cycle_prime for InstructionReadRaf

### Working Commands

```bash
# Build optimized (~13 sec proof generation)
zig build -Doptimize=ReleaseFast

# Generate proof
./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format -o /tmp/zolt_proof_dory.bin --export-preprocessing /tmp/zolt_preprocessing.bin --trace-length 64

# Verify
cd /home/vivado/projects/jolt && cargo test -p jolt-core --features zolt-debug --lib test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture
```

### Session Summary

- **Stage 4 now passes!** (RegistersRWC, RamValEvaluation, ValFinal)
- Stage 5 fails at final sumcheck claim verification
- Confirmed r_reduction and ra_claims match between Zolt and Jolt
- Issue is output_claim vs expected_claim mismatch
- Next: Debug round polynomial coefficients and challenge derivation

SESSION_ENDING - Stage 4 passes! Stage 5 r_reduction/ra_claims verified matching. Output vs expected claim mismatch remains.
