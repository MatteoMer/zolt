# Zolt-Jolt Compatibility Implementation

## Status: IN PROGRESS - Stage 5 Round Count Mismatch

## Current Issue (2026-01-30)

### Progress Made
1. **Fixed: Missing `--jolt-format` flag** - The prover was not using the Jolt-compatible proof generation path
2. **Fixed: Stage 4 Phase Configuration** - Changed phase1 from `log_t/2` to `log_t` to match Jolt
3. **Stage 4 PASSES** - RegistersReadWriteChecking with 15 rounds works correctly!

### Current Error
```
assertion `left == right` failed
  left: 8
 right: 136
```
In `sumcheck.rs:335` - Stage 5 sumcheck proof has wrong number of rounds.

### Root Cause
Jolt's Stage 5 is a **batched sumcheck** with 3 instances:
1. `RegistersValEvaluationSumcheckVerifier` (8 rounds = log_T)
2. `RamRaClaimReductionSumcheckVerifier` (136 rounds!!)
3. `LookupsReadRafSumcheckVerifier`

Zolt generates Stage 5 with only 8 rounds (RegistersValEvaluation), not considering the other instances!

### Verification Results
- Stage 1: PASSED ✅
- Stage 2: PASSED ✅
- Stage 3: PASSED ✅
- Stage 4: PASSED ✅ (15 rounds, RegistersReadWriteChecking)
- Stage 5: FAILED ❌ (wrong round count)

## Files Modified This Session

### `src/zkvm/proof_converter.zig`
- Line 2990: Changed `phase1_num_rounds = n_cycle_vars / 2` to `phase1_num_rounds = n_cycle_vars`
- Line 3828: Changed `phase1 = n_cycle_vars / 2` to `phase1 = n_cycle_vars`

### `src/zkvm/jolt_types.zig`
- Line 41: Changed `ram_phase1 = log_t / 2` to `ram_phase1 = log_t`
- Line 43: Changed `reg_phase1 = log_t / 2` to `reg_phase1 = log_t`

### Key Discovery: --jolt-format flag
The `--jolt-format` command line flag is REQUIRED to use the Jolt-compatible proof generation path!
Without it, Zolt generates proofs with its internal format (different stage layout).

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

1. **Implement Stage 5 Batched Sumcheck**
   - Add RamRaClaimReductionSumcheckProver
   - Add LookupsReadRafSumcheckProver
   - Batch all 3 instances together with correct round count (136 rounds)

2. **Check Stage 6 and Stage 7**
   - Stage 6: BytecodeReadRaf + RamHammingBooleanity + Booleanity
   - Stage 7: HammingWeightClaimReduction

3. **Verify remaining stages**
   - Once all stages generate correct proofs, full verification should pass

## Key Architecture Notes

### Jolt Stage Layout (from verifier.rs)
- Stage 1: SpartanOuter + UniSkip
- Stage 2: Batched (multiple instances)
- Stage 3: SpartanShift
- Stage 4: RegistersReadWriteChecking + RamValEval + RamValFinal (batched)
- Stage 5: RegistersValEval + RamRaReduction + LookupsReadRaf (batched)
- Stage 6: BytecodeReadRaf + RamHammingBooleanity + Booleanity (batched)
- Stage 7: HammingWeightClaimReduction
- Stage 8: Dory batch opening

### Zolt Internal Stage Layout (different!)
- Stage 1-3: Similar
- Stage 4: Value evaluation (8 rounds)
- Stage 5: Register evaluation (8 rounds)
- Stage 6: Booleanity (8 rounds)

The mismatch is because Zolt's internal verifier has a different stage layout than Jolt!
The proof_converter.zig is supposed to translate, but it's not generating the batched stages correctly.
