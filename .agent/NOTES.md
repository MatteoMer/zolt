# Zolt-Jolt Cross-Verification Progress

## Session 15 Final Summary

### Achievements
1. **Stage 2 UniSkip Fixed** - All 13 polynomial coefficients match between Zolt and Jolt
2. **Transcript Alignment** - Stage 2 r0 = `8768758914789955585787902790032491769856779696899125603611137465800193155946` matches
3. **Stage 1 Verified** - Passes Jolt verification completely

### Remaining Work: Stage 2 Batched Sumcheck Provers

The Stage 2 batched sumcheck has 5 instances. Only 2 have proper provers:

| Instance | Name | Rounds | Implementation Status |
|----------|------|--------|----------------------|
| 0 | ProductVirtualRemainder | 10 | ✅ Implemented |
| 1 | RamRafEvaluation | 16 | ❌ Zero fallback |
| 2 | RamReadWriteChecking | 26 | ❌ Zero fallback |
| 3 | OutputSumcheck | 16 | ✅ Implemented |
| 4 | InstructionLookupsClaimReduction | 10 | ❌ Zero fallback |

### Instance Details

#### Instance 1: RamRafEvaluation
- Evaluates `eq(r_address, x)` for RAM access validation
- Input claim: RamAddress opening from SpartanOuter
- Reference: `jolt-core/src/zkvm/ram/raf_evaluation.rs`

#### Instance 2: RamReadWriteChecking
- Most complex - 3 phases
- Validates RAM read/write consistency
- Input claim: RamReadValue + gamma * RamWriteValue
- Reference: `jolt-core/src/zkvm/ram/read_write_checking.rs`

#### Instance 4: InstructionLookupsClaimReduction
- 2 phases: prefix-suffix sumcheck + regular sumcheck
- Reduces instruction lookup claims
- Input claim: LookupOutput + gamma * (LeftOperand + RightOperand)
- Reference: `jolt-core/src/zkvm/claim_reductions/instruction_lookups.rs`

### Key Insight

The sumcheck protocol requires that:
```
s(0) + s(1) = old_claim
```

at every round. Instances with non-zero input claims must produce proper polynomial evaluations to satisfy this constraint. Simply contributing zeros is incorrect.

### Test Commands
```bash
# Generate Zolt proof
./zig-out/bin/zolt prove --jolt-format -o /tmp/zolt_proof_dory.bin examples/fibonacci.elf

# Test with Jolt
cd /Users/matteo/projects/jolt/jolt-core
cargo test test_verify_zolt_proof -- --ignored --nocapture
```

### Technical Notes

1. **Batching Coefficients**: Each instance is scaled by a batching coefficient sampled from the transcript
2. **Round Distribution**: Different instances become active at different rounds based on their polynomial sizes
3. **Phase Transitions**: Some provers (like InstructionLookupsClaimReduction) have phase transitions mid-sumcheck

### Files to Implement

1. `src/zkvm/ram/raf_evaluation.zig` - New file for RamRafEvaluation prover
2. `src/zkvm/ram/read_write_checking.zig` - New file for RamReadWriteChecking prover
3. `src/zkvm/claim_reductions/instruction_lookups.zig` - New file for InstructionLookupsClaimReduction prover
