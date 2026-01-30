# Zolt-Jolt Compatibility Implementation

## Status: IN PROGRESS - Stage 5 Sumcheck Implementation

## Current Issue (2026-01-30)

### Progress Made
1. **Fixed: Missing `--jolt-format` flag** - The prover was not using the Jolt-compatible proof generation path
2. **Fixed: Stage 4 Phase Configuration** - Changed phase1 from `log_t/2` to `log_t` to match Jolt
3. **Stage 4 PASSES** - RegistersReadWriteChecking with 15 rounds works correctly!
4. **Fixed: Stage 5 Round Count** - Changed from 8 to 136 rounds (LookupsReadRaf max)
5. **Fixed: Stage 6 Round Count** - Changed from 8 to 24 rounds (BytecodeReadRaf max)
6. **Fixed: Missing Claims** - Added LookupTableFlag(0-41), InstructionRa(0-7), InstructionRafFlag

### Root Cause Analysis (2026-01-30)

**The issue**: Stage 5 gets non-zero input claims from Stage 4's accumulator but we generate zero sumcheck round polynomials.

- input_claim[0] (RegistersValEvaluation) = 20196670024706610341728276844931391924934592974175535367959454787282160553899
- input_claim[1] (RamRaClaimReduction) = 16410442144988038954986615472772880745324464916492580913716405392685466979654
- input_claim[2] (LookupsReadRaf) = 9299828901037110504125985581408576613022125108259561907120516744221579828954

All three expected_output_claims are 0 (because we provide zero opening claims), but the zero sumcheck produces non-zero output_claim due to hint-based reconstruction.

**Solution**: Implement actual Stage 5 provers that correctly compute sumcheck round polynomials to reduce the input claims.

### Progress on Stage 5 Implementation

1. **RegistersValEvaluationProver** - Created `src/zkvm/registers/val_evaluation.zig`
   - Computes: Σ_j inc(j) · wa(j) · LT(j, r_cycle)
   - Need to integrate into proof_converter

2. **RamRaClaimReduction** - TODO
   - 3-phase prover: address → cycle1 → cycle2
   - Batches 4 RA claims from earlier stages

3. **LookupsReadRaf** - TODO
   - 128 + 8 = 136 rounds
   - Instruction lookup RAF checking

### Verification Results
- Stage 1: PASSED ✅
- Stage 2: PASSED ✅
- Stage 3: PASSED ✅
- Stage 4: PASSED ✅ (15 rounds, RegistersReadWriteChecking)
- Stage 5: FAILED ❌ (need real prover implementation)
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

### Immediate Priority: Stage 5 Implementation

The core challenge is implementing a batched sumcheck prover that:
1. Handles different round counts per instance (8, 24, 136)
2. Computes actual round polynomials from witness data
3. Produces opening claims that match the sumcheck reduction

**Implementation Plan:**

1. **Create BatchedSumcheckProver framework**
   - Generic batched sumcheck that handles multiple instances
   - Instance start/end based on remaining rounds
   - Polynomial accumulation with batching coefficients

2. **Implement RegistersValEvaluationProver** (partially done)
   - Located at: `src/zkvm/registers/val_evaluation.zig`
   - Need to integrate with proof_converter
   - Need to extract rd_inc, rd_addresses from trace

3. **Implement RamRaClaimReductionProver**
   - 3-phase structure: PhaseAddress → PhaseCycle1 → PhaseCycle2
   - Reference: jolt-core/src/zkvm/claim_reductions/ram_ra.rs
   - Batches 4 RA claims into single opening

4. **Implement LookupsReadRafProver**
   - 136 rounds (128 address + 8 cycle)
   - Instruction lookup verification
   - Reference: jolt-core/src/zkvm/bytecode/read_raf_checking.rs

### After Stage 5

5. **Implement Stage 6 Provers**
   - BytecodeReadRaf
   - RamHammingBooleanity, Booleanity
   - RamRaVirtual, LookupsRaVirtual
   - IncClaimReduction

6. **Implement Stage 7 Prover**
   - HammingWeightClaimReduction

7. **Final verification**
   - Once all stages generate correct proofs, full verification should pass

## Technical Notes

### Why Zero Sumcheck Doesn't Work

Zero-coefficient round polynomials combined with hint-based reconstruction:
```
p(x) = H*x  (where H = previous claim)
p(r) = H*r
output_claim = input_claim * r1 * r2 * ... * rn ≠ 0
```

But expected_output_claim = 0 (if we provide zero opening claims).
Hence verification fails.
