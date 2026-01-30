# Zolt-Jolt Compatibility Implementation

## Status: IN PROGRESS - Stage 5 Claim Value Mismatch

## Current Issue (2026-01-30)

### Progress Made
1. **Fixed: Missing `--jolt-format` flag** - The prover was not using the Jolt-compatible proof generation path
2. **Fixed: Stage 4 Phase Configuration** - Changed phase1 from `log_t/2` to `log_t` to match Jolt
3. **Stage 4 PASSES** - RegistersReadWriteChecking with 15 rounds works correctly!
4. **Fixed: Stage 5 Round Count** - Changed from 8 to 136 rounds (LookupsReadRaf max)
5. **Fixed: Stage 6 Round Count** - Changed from 8 to 24 rounds (BytecodeReadRaf max)
6. **Fixed: Missing Claims** - Added LookupTableFlag(0-41), InstructionRa(0-7), InstructionRafFlag

### Current Error
```
=== SUMCHECK VERIFICATION FAILED ===
output_claim:          9219502725919403917352040447078840562485657953396409742770278624303131450233
expected_output_claim: 0
r_sumcheck len: 136
```
Stage 5 sumcheck structure is correct (136 rounds pass) but the claims are zeros instead of real values.

### Verification Results
- Stage 1: PASSED ✅
- Stage 2: PASSED ✅
- Stage 3: PASSED ✅
- Stage 4: PASSED ✅ (15 rounds, RegistersReadWriteChecking)
- Stage 5: FAILED ❌ (claims are zero - need to compute real values)
- Stage 6: NOT YET REACHED
- Stage 7: NOT YET REACHED

## Files Modified This Session

### `src/zkvm/proof_converter.zig`
- Line ~2652: Changed Stage 5 from `n_cycle_vars` to `lookups_log_k + n_cycle_vars` (136 rounds)
- Line ~2688: Changed Stage 6 from `n_cycle_vars` to `bytecode_log_k + n_cycle_vars` (24 rounds)
- Added LookupTableFlag(0-41) claims for InstructionReadRaf
- Added InstructionRa(0-7) claims for InstructionReadRaf
- Added InstructionRafFlag claim for InstructionReadRaf

### Key Architecture Notes

### Jolt Stage 5 Structure
Stage 5 is a **batched sumcheck** with 3 instances:
1. `RegistersValEvaluationSumcheckVerifier` (8 rounds = log_T)
2. `RamRaClaimReductionSumcheckVerifier` (24 rounds = log_K + log_T = 16 + 8)
3. `LookupsReadRafSumcheckVerifier` (136 rounds = LOG_K + log_T = 128 + 8)

Where:
- `LOG_K` for lookups = XLEN * 2 = 64 * 2 = 128
- `log_K` for RAM = 16 (from one_hot_params.ram_k)
- `log_T` = 8 (trace length = 256)

The batched sumcheck uses **max_num_rounds** = max(8, 24, 136) = 136 rounds!

## Test Commands

### Generate Jolt-compatible Proof (MUST use --jolt-format!)
```bash
cd /home/vivado/projects/zolt
./zig-out/bin/zolt prove examples/fibonacci.elf \
  --jolt-format \
  --export-preprocessing logs/zolt_preprocessing.bin \
  -o logs/zolt_proof_dory.bin
```

### Verify with Jolt
```bash
cd /home/vivado/projects/zolt/jolt
cargo test --package jolt-core test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture
```

## Next Steps

1. **Implement Real Stage 5 Prover**
   - RegistersValEvaluation: compute real claims from trace
   - RamRaClaimReduction: compute real claims from trace
   - LookupsReadRaf: compute real claims from trace

2. **Implement Real Stage 6 Prover**
   - BytecodeReadRaf
   - RamHammingBooleanity
   - Booleanity
   - RamRaVirtual
   - LookupsRaVirtual
   - IncClaimReduction

3. **Implement Real Stage 7 Prover**
   - HammingWeightClaimReduction

4. **Final verification**
   - Once all stages generate correct proofs, full verification should pass
