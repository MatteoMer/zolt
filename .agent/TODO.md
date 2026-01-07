# Zolt-Jolt Compatibility - Session 15 Complete

## Status Summary

### ✅ COMPLETED
- All 712 internal tests pass
- Stage 1 passes Jolt verification completely
- Stage 2 UniSkip r0 matches Jolt (transcript aligned)
- Stage 2 UniSkip polynomial coefficients verified identical

### ❌ REMAINING
- Stage 2 batched sumcheck fails (3 missing provers)

## Detailed Analysis

### Stage 2 Batched Sumcheck Architecture

The Stage 2 batched sumcheck verifies 5 instances in parallel:

| Instance | Prover | Rounds | Status |
|----------|--------|--------|--------|
| 0 | ProductVirtualRemainder | 10 | ✅ Implemented |
| 1 | RamRafEvaluation | 16 | ❌ Missing |
| 2 | RamReadWriteChecking | 26 | ❌ Missing |
| 3 | OutputSumcheck | 16 | ✅ Implemented |
| 4 | InstructionLookupsClaimReduction | 10 | ❌ Missing |

### Error Analysis

Current errors:
```
Round 0: s(0)+s(1) != old_claim
Round 23: s(0)+s(1) != old_claim
Round 24: s(0)+s(1) != old_claim
Round 25: s(0)+s(1) != old_claim
```

Root cause: Instances 1, 2, 4 have non-zero input claims but contribute zero polynomials.

### Implementation Requirements

Each missing prover requires:

1. **RamRafEvaluation** (`jolt-core/src/zkvm/ram/raf_evaluation.rs`)
   - Eq polynomial evaluation over RAM addresses
   - 16 rounds of sumcheck
   - Input: RamAddress claim from SpartanOuter

2. **RamReadWriteChecking** (`jolt-core/src/zkvm/ram/read_write_checking.rs`)
   - 3-phase prover (most complex)
   - Validates RAM read/write consistency
   - 26 rounds total

3. **InstructionLookupsClaimReduction** (`jolt-core/src/zkvm/claim_reductions/instruction_lookups.rs`)
   - 2-phase prover (prefix-suffix + regular)
   - Reduces lookup operand claims
   - 10 rounds total

### Key Files

Modified:
- `src/zkvm/proof_converter.zig` - Stage 2 proof generation
- `src/zkvm/r1cs/univariate_skip.zig` - UniSkip polynomial construction

To Create:
- `src/zkvm/ram/raf_evaluation.zig`
- `src/zkvm/ram/read_write_checking.zig`
- `src/zkvm/claim_reductions/instruction_lookups.zig`

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

## Progress Metrics

- Tests: 712/712 passing ✅
- Stage 1: PASS ✅
- Stage 2 UniSkip: PASS ✅
- Stage 2 Batched Sumcheck: FAIL ❌ (missing 3 provers)
