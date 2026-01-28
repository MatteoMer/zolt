# Zolt-Jolt Compatibility: Status Update

## Status: ROOT CAUSE FOUND - LT POLYNOMIAL MISMATCH

## Summary

Zolt can now:
1. Generate proofs for RISC-V programs (`./zig-out/bin/zolt prove`)
2. Verify proofs internally (`./zig-out/bin/zolt verify`) - ALL 6 STAGES PASS
3. Export proofs in Jolt-compatible format (`--jolt-format`)
4. Export preprocessing for Jolt verifier (`--export-preprocessing`)
5. Pass all 714 unit tests ✓
6. **Proof successfully deserializes in Jolt** (when matching preprocessing)
7. **Preprocessing successfully deserializes in Jolt** ✓

## Cross-Verification Status

Verification fails at Stage 4: `Sumcheck verification failed`

## ROOT CAUSE IDENTIFIED (Session 72)

### LT Polynomial Mismatch

The LT (Less-Than) polynomial computation doesn't match between Zolt's prover and Jolt's verifier:

```
[ZOLT LT DEBUG] Computing LT(r, r_cycle):
  lt_eval_computed (Jolt formula) = { 32, 36, 41, 131, 38, 192, 145, 187, ... }
  lt_eval_prover (from binding)   = { 45, 226, 211, 44, 147, 119, 45, 90, ... }
  Match? false
```

**This is the root cause of the Stage 4 verification failure.**

### The Issue

The LT polynomial is used in `ValEvaluation::expected_output_claim()` to compute `inc * wa * lt`. Since `lt` doesn't match between prover and verifier, the expected output claim doesn't match the sumcheck output claim.

Looking at the debug output:
- x values (from Stage 4 sumcheck challenges): `{ 19, 204, 144, 7, ... }`, etc.
- y values (from Stage 2 r_cycle): `{ 84, 119, 25, 226, ... }`, etc.

### Likely Causes

1. **r_cycle point mismatch**: The prover's LtPolynomial is bound with a different r_cycle point than what the verifier uses.

2. **Endianness issue**: The LT computation iterates MSB-to-LSB, but there may be a mismatch in how the challenge bytes are interpreted.

3. **normalize_opening_point issue**: The verifier's `normalize_opening_point` may construct r_cycle differently than what the prover used.

### Next Steps

1. **Compare r_cycle points** - Check that the r_cycle used to initialize LtPolynomial in the prover matches what the verifier reconstructs via normalize_opening_point.

2. **Check LtPolynomial::bind ordering** - The LtPolynomial uses LowToHigh binding order. Verify this matches Jolt's expectation.

3. **Verify challenge interpretation** - Ensure the 32-byte challenges are interpreted the same way in both systems.

### Relevant Code Locations

Zolt:
- `src/zkvm/ram/val_evaluation.zig` - LtPolynomial initialization and binding
- `src/zkvm/proof_converter.zig` - Stage 4 LT debug code (around line 2380)

Jolt:
- `jolt-core/src/zkvm/ram/val_evaluation.rs` - LT computation formula (lines 386-391):
  ```rust
  for (x, y) in zip(&r.r, &r_cycle.r) {
      lt_eval += (F::one() - x) * y * eq_term;
      eq_term *= F::one() - x - y + *x * y + *x * y;
  }
  ```

### Debug Command

```bash
./zig-out/bin/zolt prove examples/fibonacci.elf \
    --export-preprocessing /tmp/zolt_preprocessing_new.bin \
    -o /tmp/native_proof.bin \
    --jolt-format /tmp/zolt_proof_jolt.bin 2>&1 | grep "LT DEBUG"
```

## Previous Session Progress

- Fixed termination write inclusion
- Fixed start_address in proof_converter
- Verified serialization format matches
- All 5 Stage 4 sumcheck instances match their individual provers
- Batched claim computation is correct

## Files

- Proof generation command completed, check `/tmp/` for output files
- Need to copy to `/home/vivado/projects/zolt/logs/` before running Jolt test

SESSION_ENDING - Root cause found: LT polynomial mismatch. Need to debug r_cycle point and binding order.
