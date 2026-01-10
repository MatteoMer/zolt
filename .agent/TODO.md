# Zolt-Jolt Compatibility TODO

## Current Progress

### ✅ Stage 1 - PASSES
- Outer Spartan sumcheck with univariate skip
- All round polynomials verified

### ✅ Stage 2 - PASSES
- Batched sumcheck with 5 instances
- ProductVirtualRemainder, RamRAF, RamRWC, OutputSumcheck, InstructionLookupsClaimReduction

### ✅ Stage 3 - PASSES
- Shift, InstructionInput, RegistersClaimReduction sumchecks
- Fixed InstructionInput witness consistency
- Fixed RegistersClaimReduction prefix challenge ordering

### 🔄 Stage 4 - IN PROGRESS
- Fixed round count: LOG_K + log2(T) = 7 + 8 = 15 rounds
- **BLOCKING**: Requires dense polynomial commitments:
  - `CommittedPolynomial::RdInc` - Register increment (post_value - pre_value)
  - `CommittedPolynomial::RamInc` - RAM increment
  - These are committed polynomials requiring Dory commitments

### ❌ Stages 5-7 - PENDING
- Stage 5: Address checking
- Stage 6: RAF evaluation
- Stage 7: Final RAM evaluation

## Key Technical Findings

### Register Count
- REGISTER_COUNT = 32 (RISCV) + 96 (Virtual) = 128
- LOG_K = log2(128) = 7
- Stage 4 max_rounds = LOG_K + n_cycle_vars

### Dense vs Virtual Polynomials
- **Virtual**: Computed on-the-fly from witness, no commitment
- **Dense/Committed**: Requires Dory commitment, stored in proof.commitments[]
- Stage 4 uses both virtual (RegistersVal, Rs1Ra, etc.) and dense (RdInc, RamInc)

### Next Steps
1. Generate proper Dory commitments for RdInc/RamInc polynomials
2. Ensure commitment indices match Jolt's expectations
3. Continue to Stages 5-7

## Progress Summary

| Stage | Status | Notes |
|-------|--------|-------|
| 1 | ✅ PASSES | Using ZOLT preprocessing |
| 2 | ✅ PASSES | |
| 3 | ✅ PASSES | Fixed instruction/registers claims |
| 4 | 🔄 IN PROGRESS | Round count fixed, needs dense commitments |
| 5-7 | Blocked | |

## Key Files
- `src/zkvm/spartan/stage3_prover.zig` - Stage 3 prover with InstructionInput
- `src/zkvm/r1cs/constraints.zig` - computeInstructionInputs, setFlagsFromInstruction
- `src/zkvm/proof_converter.zig` - Proof generation and stage rounds
- Jolt: `jolt-core/src/zkvm/registers/read_write_checking.rs` - RegistersReadWriteChecking
- Jolt: `jolt-core/src/zkvm/witness.rs` - RdInc/RamInc polynomial generation

## Testing
```bash
bash scripts/build_verify.sh
```
