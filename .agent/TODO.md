# Zolt-Jolt Compatibility - Current Status

## Summary

**Stage 1**: PASSES (outer sumcheck) ✓
**Stage 2**: PASSES (product virtualization + RAM RAF) ✓
**Stage 3**: FAILS - Claims present, but sumcheck verification fails (zero polynomials)
**Stage 4-7**: Untested (blocked on Stage 3)

## Progress This Session

### 1. Fixed Dory Opening Proof (DONE)
- Fixed asymmetric matrix handling (sigma > nu case)
- Updated Jolt SRS export to 16 variables (256 points)

### 2. Added All Opening Claims (DONE)
- SpartanShift: UnexpandedPC, PC, OpFlags(VirtualInstruction, IsFirstInSequence), InstructionFlags(IsNoop)
- InstructionInputVirtualization: 8 claims for left/right operand handling
- RegistersClaimReduction: RdWriteValue, Rs1Value, Rs2Value
- BytecodeReadRaf: InstructionRafFlag, InstructionRa
- RegistersReadWriteChecking: RegistersVal, Rs1Ra, Rs2Ra, RdWa
- RamValEvaluation, RamValFinalEvaluation, RamRaClaimReduction, RamRafEvaluation: RamRa
- Booleanity, RamHammingBooleanity, HammingWeightClaimReduction: RamHammingWeight

### 3. Current Issue: Stage 3 Sumcheck Verification

The Jolt verifier output shows:
```
output_claim:          3605979267482843492618018818811131090814373229214467976717812727899800934418
expected_output_claim: 1846872701798109175261071120538427009056470961050860597433873141898176138550
Verification failed: Stage 3
```

This is expected because:
- Zolt generates placeholder zero polynomials for stages 3-7
- The verifier computes `expected_output_claim` from the claims
- The `output_claim` from the zero polynomials doesn't match

## Next Steps

1. [ ] Implement real SpartanShift sumcheck prover
2. [ ] Implement InstructionInputVirtualization sumcheck prover
3. [ ] Implement RegistersClaimReduction sumcheck prover
4. [ ] Implement remaining stage 3-7 sumcheck provers

## Verification Status

The test now passes in the sense that:
- Proof can be deserialized
- All claims are present
- The verification logic runs (doesn't panic)
- But sumcheck verification fails (expected with zero polynomials)

## Testing Commands

```bash
# Build Zolt
zig build

# Generate proof
./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format --export-preprocessing /tmp/zolt_preprocessing.bin -o /tmp/zolt_proof_dory.bin --srs /tmp/jolt_dory_srs.bin

# Run Jolt verification
cd /Users/matteo/projects/jolt/jolt-core && cargo test test_verify_zolt_proof_with_zolt_preprocessing --release -- --ignored --nocapture
```

## Commits Made

1. 60c1ad1 - debug: Found OutputSumcheck zero-check failure root cause
2. b74cb76 - fix: Set panic/termination bits in val_final + correct ELF base address
3. fc2c8cc - docs: Update TODO with progress on OutputSumcheck fix
4. 0bae6fc - fix: Dory opening proof for asymmetric matrix sizes
5. 15b2d47 - feat: Add all required opening claims for Jolt stages 3-7
