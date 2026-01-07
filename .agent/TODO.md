# Zolt-Jolt Compatibility - Stage 2 Progress

## Current Status
Stage 2 sumcheck verification improving but still failing.

### Latest Values (after claim update fix)
- output_claim:          20291704288663086458385602321745330294718053661243679658093460943559986613657
- expected_output_claim: 15745090188359599137189004113283834746953666719330265610972696622859550686329

### Tests
- All 712 Zolt tests pass
- Jolt verification failing at Stage 2

## Completed Fixes

### 1. Factor Flags in R1CS (commit fd6e5b7)
- Added FlagIsRdNotZero: 1 if rd register index != 0
- Added FlagBranch: 1 if instruction opcode == 0x63
- Added FlagIsNoop: 1 if this is a noop instruction
- NUM_INPUTS increased from 36 to 39

### 2. Claim Update for ProductVirtualRemainder (commit 85898bb)
- CRITICAL: Added updateClaim(evals, challenge) before bindChallenge()
- Without this, the hint s(1) = current_claim - s(0) uses wrong claim
- This significantly improved the output_claim value

### 3. R_cycle Reversal for BIG_ENDIAN (commit 95b565a)
- Challenges reversed for factor evaluation
- Matches Jolt's OpeningPoint<BIG_ENDIAN> convention

## Remaining Issues

The output_claim and expected_output_claim are now within ~30% of each other (20291e75 vs 15745e75) but still don't match.

Possible causes:
1. **Round polynomial computation** - The cubic polynomial s(X) might have errors
2. **Split_eq binding** - The eq polynomial binding might differ from Jolt
3. **Endianness** - Subtle ordering differences in tau or other parameters

## Debugging Strategy

1. **Add more debug output** - Print intermediate values in both Zolt and Jolt
2. **Compare round by round** - Verify each round's polynomial matches
3. **Test with smaller trace** - Use 4-cycle trace for easier debugging

## Commands

```bash
# Build and test
zig build test --summary all

# Generate proof
./zig-out/bin/zolt prove --jolt-format -o /tmp/proof.bin examples/fibonacci.elf

# Test with Jolt
cp /tmp/proof.bin /tmp/zolt_proof_dory.bin
cd /Users/matteo/projects/jolt
cargo test --package jolt-core test_verify_zolt_proof -- --ignored --nocapture
```
