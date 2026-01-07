# Zolt-Jolt Compatibility - Status Update

## Session 15 - Final Summary

### Current Status
- ✅ All 712 internal tests pass (`zig build test`)
- ✅ Stage 1 passes Jolt verification
- ✅ Stage 2 UniSkip r0 now matches Jolt
- ❌ Stage 2 batched sumcheck fails

### Critical Finding: Stage 2 UniSkip Fixed!
After extensive debugging, the Stage 2 UniSkip transcript is now correctly aligned:
- All 13 polynomial coefficients match between Zolt and Jolt
- r0 = `8768758914789955585787902790032491769856779696899125603611137465800193155946` (matches!)

### Remaining Issue: Stage 2 Batched Sumcheck

The batched sumcheck has 5 instances. Only instances 0 and 3 have proper provers:

| Instance | Name | Status | Input Claim | Expected Output |
|----------|------|--------|-------------|-----------------|
| 0 | ProductVirtualRemainder | ✅ Implemented | non-zero | fused products |
| 1 | RamRafEvaluation | ❌ Zero fallback | non-zero | 0 |
| 2 | RamReadWriteChecking | ❌ Zero fallback | non-zero | 0 |
| 3 | OutputSumcheck | ✅ Implemented | 0 | 0 |
| 4 | InstructionLookupsClaimReduction | ❌ Zero fallback | non-zero | 0 |

### Sumcheck Claim Errors
- Round 0: `s(0)+s(1) != old_claim`
- Round 23-25: `s(0)+s(1) != old_claim`

These errors occur because instances 1, 2, 4 have non-zero input claims but contribute zero polynomials.

### What's Needed

1. **RamRafEvaluation Prover** (instance 1)
   - Evaluates `eq(r_address, x)` polynomial
   - Reference: `jolt-core/src/zkvm/ram/raf_evaluation.rs`
   - 16 rounds

2. **RamReadWriteChecking Prover** (instance 2)
   - Most complex - 3 phases
   - Reference: `jolt-core/src/zkvm/ram/read_write_checking.rs`
   - 26 rounds

3. **InstructionLookupsClaimReduction Prover** (instance 4)
   - 2 phases: prefix-suffix sumcheck + regular sumcheck
   - Reference: `jolt-core/src/zkvm/claim_reductions/instruction_lookups.rs`
   - 10 rounds

### Alternative Approach
For the fibonacci program specifically, RAM operations are minimal. The provers might be simplified if the actual polynomial evaluations are zero. However, the sumcheck protocol requires proper polynomial contributions even if they evaluate to zero at the end.

## Verification Steps

To reproduce:
```bash
# Generate Zolt proof
./zig-out/bin/zolt prove --jolt-format -o /tmp/zolt_proof_dory.bin examples/fibonacci.elf

# Test with Jolt
cd /Users/matteo/projects/jolt/jolt-core
cargo test test_verify_zolt_proof -- --ignored --nocapture
```

## Key Files
- `src/zkvm/proof_converter.zig` - Main proof conversion and Stage 2 batched sumcheck
- `src/zkvm/r1cs/univariate_skip.zig` - UniSkip polynomial construction
- `src/zkvm/spartan/product_remainder.zig` - ProductVirtualRemainder prover
- `src/zkvm/ram/output_check.zig` - OutputSumcheck prover
