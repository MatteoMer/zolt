# Zolt-Jolt Compatibility Implementation

## Status: IN PROGRESS - Stage 5 Expected Output Claim Mismatch

## Current Issue

Stage 5 sumcheck round polynomials are CORRECT (verified by comparing coefficients), but the final `expected_output_claim` computed by the verifier doesn't match the prover's `output_claim`.

### Evidence

**Sumcheck round polynomial verification:**
- Round 0 coeff[0]: Zolt=`08 90 1d df ...`, Jolt=`[08, 90, 1d, df, ...]` ✓ MATCH
- Round 0 coeff[1]: Zolt=`17 34 7e 8f ...`, Jolt=`[17, 34, 7e, 8f, ...]` ✓ MATCH
- Challenges match exactly between Zolt and Jolt

**Sumcheck final claim verification:**
- output_claim (from proof): `[ce, b6, 9e, 59, 52, e3, 16, ed, ...]`
- expected_claim (computed): `[b7, c0, 2e, 8e, c4, 3e, bd, 5d, ...]`
- THESE DON'T MATCH!

**Per-instance expected_output_claim from Jolt:**
- Instance 0 (RegistersValEvaluation): `[4a, 87, c9, 11, ...]` × batch0
- Instance 1 (RamRaClaimReduction): `[5d, c0, 20, b8, ...]` × batch1
- Instance 2 (InstructionReadRaf): `[10, fe, 87, dc, ...]` × batch2

### Root Cause Hypothesis

The prover's sumcheck polynomial correctly produces the output_claim, but one or more of the instance `expected_output_claim` formulas evaluates to a different value than what the prover's polynomial implies.

This suggests a mismatch in:
1. How the prover updates instance polynomials during binding
2. How the verifier formula computes expected_output_claim

### Verified Components (All Match)

1. **InstructionRa claims** ✓ - All 8 chunks match
2. **ra_product** ✓ - Product matches (`[7f, 43, eb, f8, ...]`)
3. **LookupTableFlag claims** ✓ - All 42 flags match
4. **raf_claim** ✓ - Formula computation matches (`[92, 5a, 0b, 79, ...]`)
5. **eq_eval_r_reduction** ✓ - Matches (`[f6, fa, a0, 07, ...]`)
6. **left/right/identity prefix evaluations** ✓ - All match
7. **batching coefficients** ✓ - batch0, batch1, batch2 all match
8. **input_claims** ✓ - RegistersVal and other Stage 5 inputs match
9. **Round polynomial coefficients** ✓ - First 3 rounds match exactly
10. **Sumcheck challenges** ✓ - All 136 challenges match

### Next Steps

1. Debug what Zolt prover computes for each instance's contribution to the final polynomial
2. Compare Instance 0/1/2 expected values between Zolt prover and Jolt verifier
3. The discrepancy might be in:
   - RegistersValEvaluation (inc*wa*lt product)
   - RamRaClaimReduction (eq_combined * ra_claim)
   - InstructionReadRaf (eq*ra*(val+gamma*raf))

### Key Files

- `/home/vivado/projects/zolt/src/zkvm/spartan/stage5_prover.zig` - Stage 5 prover
- `/home/vivado/projects/jolt/jolt-core/src/subprotocols/sumcheck.rs` - Jolt sumcheck verifier
- `/home/vivado/projects/jolt/jolt-core/src/zkvm/instruction_lookups/read_raf_checking.rs` - InstructionReadRaf verifier

## Test Commands

```bash
# Build Zolt
zig build -Doptimize=ReleaseFast

# Generate proof
./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format --export-preprocessing /tmp/zolt_preprocessing.bin -o /tmp/zolt_proof_dory.bin

# Verify with Jolt
cp /tmp/zolt_*.bin /home/vivado/projects/jolt/
cd /home/vivado/projects/jolt
cargo test --package jolt-core --features zolt-debug test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture
```

## Session Summary

This session made significant progress:
1. Confirmed Stage 5 input claims now match (fixed from previous session)
2. Verified all opening claims match (InstructionRa, LookupTableFlag, etc.)
3. Verified round polynomial coefficients are identical
4. Identified that the issue is in expected_output_claim formula evaluation
5. Narrowed down to per-instance expected_output_claim computation mismatch
