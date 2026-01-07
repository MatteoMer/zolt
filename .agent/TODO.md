# Zolt-Jolt Compatibility - Session 17 Progress

## Status Summary

### ✅ COMPLETED
- All 712 internal tests pass
- Stage 1 passes Jolt verification completely
- Stage 2 sumcheck constraint (s(0)+s(1) = claim) passes all rounds
- All 5 instance provers integrated and updating claims correctly

### ❌ REMAINING ISSUE
- Stage 2 output_claim ≠ expected_output_claim
- The sumcheck ROUNDS are correct (claims propagate properly)
- The FINAL claim doesn't match what verifier expects from polynomial openings

## Stage 2 Analysis

### Current Numbers
- Zolt output_claim: 21589049388974437284381767714841252907584727603323336789751151772545883922031
- Jolt expected_output_claim: 9898116330490506910341296015176916681340621516181642175616333993492433350690

### Verified Components (MATCH ✅):
- fused_left: 6890326872915039705262170125493404951295898969136690654065828419097758694280
- fused_right: 16403093111254811232256366631936514937767139743124672059436405855732128935088

### Expected Output Claim Formula (Instance 0):
```
expected = tau_high_bound_r0 * tau_bound_r_tail_reversed * fused_left * fused_right
```

Where:
- tau_high_bound_r0 = 20155157908076722556502616784924511038746646565368255767444635160140374572918
- tau_bound_r_tail_reversed = 20142916058911915487407898995168112567948371721580244419193443500655839812742

### Root Cause Analysis
The fused_left/fused_right (opening claims) MATCH between Zolt and Jolt.
The tau_high_bound_r0 (Lagrange kernel) should match if r0 matches.
The tau_bound_r_tail_reversed depends on:
1. tau_low values (should be identical from transcript)
2. r_tail_reversed = reversed(r_cycle challenges)

The issue is likely in:
1. Challenge ordering (reversed vs not)
2. Split_eq binding direction

## Fixes Made This Session

1. Fixed lagrangeC2 formula in output_check.zig:
   - OLD: c2 = (s(0) - 2*s(1) + s(2)) / 2 (WRONG - gives c2 + 3*c3)
   - NEW: c2 = (2*s(0) - 5*s(1) + 4*s(2) - s(3)) / 2 (CORRECT)

2. Added bindChallenge/updateClaim for InstructionLookups prover

3. Fixed claim_before capture for ProductVirtualRemainder

## Next Steps

1. Verify tau values match between Zolt and Jolt
2. Check challenge ordering in EqPolynomial::mle evaluation
3. Verify split_eq binding direction (LowToHigh)
4. Compare r_cycle challenges between prover and verifier

## Verification Commands

```bash
# Build and test Zolt
zig build test --summary all

# Generate Jolt-compatible proof
./zig-out/bin/zolt prove --jolt-format -o /tmp/zolt_proof_dory.bin examples/fibonacci.elf

# Verify with Jolt
cd /Users/matteo/projects/jolt/jolt-core
cargo test test_verify_zolt_proof -- --ignored --nocapture
```
