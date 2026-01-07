# Zolt-Jolt Compatibility - Stage 2 Progress

## Current Status
- Stage 1: PASSING ✅
- Stage 2: FAILING ❌ (OutputSumcheck val_final_claim != val_io_eval)
- Stage 3+: Not reached yet

## Session 7 Final Progress

### Verified Components (ALL MATCH!)
1. ✅ Stage 2 input_claims (all 5 match Jolt)
2. ✅ Stage 2 gamma_rwc and gamma_instr
3. ✅ Batching coefficients
4. ✅ Polynomial coefficients (c0, c2, c3) for all 26 rounds
5. ✅ Sumcheck challenges for all 26 rounds
6. ✅ 8 factor evaluations (LeftInstructionInput, RightInstructionInput, etc.)
7. ✅ Factor claims inserted into proof correctly

### ROOT CAUSE DEEP DIVE

#### The Problem

OutputSumcheck (instance 3) has a NON-ZERO `expected_output_claim` even though `input_claim = 0`.

The `expected_output_claim` formula is:
```
expected_output_claim = eq_eval * io_mask_eval * (val_final_claim - val_io_eval)
```

Where:
- `val_final_claim = 0` (currently set by Zolt)
- `val_io_eval = MLE(ProgramIOPolynomial, r_address_prime) ≠ 0`

The non-zero `val_io_eval` comes from **the termination bit**. Even with empty inputs/outputs,
`ProgramIOPolynomial` sets `coeffs[termination_index] = 1` for successful execution.

#### Why val_io_eval is Non-Zero

For a correctly executing program:
- `program_io.inputs = []` (empty)
- `program_io.outputs = []` (empty)
- `program_io.panic = false`
- Therefore: `coeffs[termination_index] = 1`

The MLE evaluation at random point `r_address_prime`:
```
val_io_eval = eq(termination_index, r_address_prime) * 1 ≠ 0
```

#### The Fix Required

For OutputSumcheck to pass, we need:
```
val_final_claim = val_io_eval
```

This means computing `val_io_eval = MLE(termination_bit_poly, r_address_prime)` where:
1. `termination_bit_poly` is a polynomial with a single `1` at `termination_index`
2. `r_address_prime = challenges[10..26]` (last 16 challenges from Stage 2 sumcheck)

The computation is: `eq(termination_index, r_address_prime)` where `eq` is the Lagrange basis.

### Implementation Status

✅ Stage2Result now includes `challenges` slice for use in val_io computation
✅ ConversionConfig extended with `memory_layout` field
✅ Imports added for jolt_device and constants
❌ val_io_eval computation not yet implemented

### Next Steps

1. Pass memory_layout through ConversionConfig
2. Implement `computeValIoEval(memory_layout, challenges)`:
   - Get `termination_index` from memory layout
   - Compute `eq(termination_index, r_address_prime)`
   - Set `RamValFinal at RamOutputCheck = result`
3. Test Stage 2 verification passes

## Commits
- `abe09a4`: Fixed input_claims and gamma sampling
- `5033064`: Debug - polynomial coefficients match
- `78a09cf`: Deep dive - termination bit is the issue
- `68db1c2`: WIP structure improvements for OutputSumcheck
