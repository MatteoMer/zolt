# Zolt-Jolt Compatibility: Stage 4 Internal Check Issue

## Status: In Progress (Session 69)

## Current Issue

The Stage 4 sumcheck internal consistency check is failing:
```
[ZOLT STAGE4 FINAL DEBUG] Match? false
```

This check compares:
- `batched_claim`: The final claim from sumcheck (after all binding rounds)
- `coeff[0] * regs_current_claim`: Expected claim from polynomial evaluations

They should be equal but they're not.

## Debug Output Analysis

From the proof generation:
```
[ZOLT STAGE4 FINAL DEBUG] coeff[0] * regs_current = { 112, 237, 127, 101, ...
[ZOLT STAGE4 FINAL DEBUG] batched_claim = { 231, 126, 223, 191, ...
[ZOLT STAGE4 FINAL DEBUG] Match? false
```

## Session 69 Findings

1. **Jolt dependency issues**: Cannot build Jolt due to missing OpenSSL/pkg-config on this system
2. **The internal check failure is in Stage 4 prover** (not transcript or serialization issue)
3. **The prover uses Stage4GruenProver** with Gruen optimization for eq polynomial

## Root Cause Candidates

1. **evalFromHint formula mismatch**: The hint-based evaluation might not match Jolt's
2. **Polynomial binding order**: The challenge binding might be in wrong order
3. **eq polynomial computation**: The GruenSplitEqPolynomial might produce wrong values
4. **Phase configuration**: The phase1/phase2/phase3 round split might be incorrect

## Key Files

- `/home/vivado/projects/zolt/src/zkvm/proof_converter.zig:2032-2362` - Stage 4 sumcheck loop
- `/home/vivado/projects/zolt/src/zkvm/spartan/stage4_gruen_prover.zig` - Stage 4 prover
- `/home/vivado/projects/zolt/jolt/jolt-core/src/zkvm/registers/read_write_checking.rs:815-914` - Jolt verifier

## Next Steps

1. **Debug the sumcheck binding loop**:
   - Add more detailed logging for each round
   - Compare `batched_claim` evolution with expected
   - Verify `evalFromHint` produces correct values

2. **Verify polynomial evaluations**:
   - Check if `regs_current_claim` is being computed correctly
   - Verify eq polynomial evaluations match Jolt

3. **Compare with Jolt prover** (if dependency issues can be resolved):
   - Run Jolt prover with same input
   - Compare round-by-round output

## Session Summary

- Started investigating Stage 4 internal check failure
- Analyzed proof_converter.zig Stage 4 implementation
- Could not run cross-verification test due to OpenSSL build dependency
- Need to resolve internal consistency check before cross-verification can work
