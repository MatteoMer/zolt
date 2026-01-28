# Zolt-Jolt Compatibility: Stage 4 Input Claim Issue

## Current Problem

Stage 4 verification fails because the transcript diverges at the input claims:
- Jolt verifier expects: `instance[1]` (RamValEval) = NON-ZERO
- Zolt prover sends: `instance[1]` (RamValEval) = ZERO

But for Fibonacci (no RAM ops), both should be ZERO! So why does Jolt compute non-zero?

## Key Findings (Session 68)

### 1. Opening Points Are Reconstructed, Not Stored

Both Jolt prover/verifier reconstruct opening points from sumcheck challenges:
- Jolt's proof serialization skips `_opening_point` (only stores key + claim)
- Verifier initializes with `OpeningPoint::default()`, then populates during `cache_openings`
- `normalize_opening_point(sumcheck_challenges)` reconstructs `r_address` and `r_cycle`

Zolt's format is correct (only stores key + claim).

### 2. The `init_eval` Computation Flow

For RamValEvaluation's input_claim:
```
input_claim = claimed_evaluation - init_eval
```

Where:
- `claimed_evaluation` = proof.opening_claims[RamVal @ RamReadWriteChecking]
- `init_eval` = computed during verification from:
  - advice_contributions (from commitment openings)
  - val_init_public_eval (bytecode + inputs, computed directly)

### 3. For Programs Without RAM Ops (Fibonacci)

- `claimed_evaluation` = MLE(initial_ram) @ r_address
- `init_eval` = MLE(bytecode + inputs) @ r_address
- Since initial_ram = bytecode + inputs (+ advice if any), these should be EQUAL
- `input_claim = 0` is CORRECT for Fibonacci!

### 4. The Mismatch

The NOTES indicate Jolt computes non-zero while Zolt sends zero. Possible causes:
1. `r_address` reconstruction differs between Jolt verifier and Zolt prover
2. Jolt's `eval_initial_ram_mle` includes something extra that Zolt's `computeInitialRamEval` doesn't
3. The `claimed_evaluation` stored in the proof differs from what Jolt expects

### 5. Termination Bit Workaround Issue

Zolt's RWC prover has a workaround that adds termination bit to `val_init`:
```zig
// In read_write_checking.zig
val_init[termination_index] = F.one();  // WORKAROUND
```

But Jolt's `initial_ram` does NOT include termination bit (only final_ram does).

This creates a mismatch:
- `rwc_val_claim` = MLE(initial_ram + termination) @ r_address (Zolt)
- But Jolt expects: `rwc_val_claim` = MLE(initial_ram) @ r_address

The workaround comment claims it makes input_claim=0, but that's wrong if `computeInitialRamEval` doesn't also add termination bit.

## Recommended Fix

**Remove the termination bit workaround** from RWC prover:
1. Jolt's prover does NOT add termination to initial RAM
2. Jolt's emulator records termination as a real RAM write (creates sparse entry)
3. Zolt's tracer currently skips termination write (see tracer/mod.zig comments)

Correct approach:
1. Either have Zolt's tracer record termination write like Jolt
2. Or remove the workaround entirely and accept that for no-RAM programs, both rwc_val_claim and init_eval are equal (giving input_claim = 0)

## Next Steps

1. **Remove termination bit workaround** from RWC prover's val_init
2. **Test with Fibonacci** - input_claim should be 0 for both Zolt and Jolt
3. **If Jolt still expects non-zero**, investigate what Jolt's preprocessing includes

## Files Involved

- `/home/vivado/projects/zolt/src/zkvm/proof_converter.zig` - Stage 4 input claim computation
- `/home/vivado/projects/zolt/src/zkvm/ram/read_write_checking.zig` - RWC prover with termination workaround (lines 224-236)
- `/home/vivado/projects/zolt/src/zkvm/mod.zig` - buildInitialRamMap function

## Session Notes

- Spent significant time understanding Jolt's opening point handling
- Opening points are NOT in serialized proof - reconstructed during verification
- For no-RAM programs, input_claim = 0 is mathematically correct
- The NOTES may have been from a different test case with actual RAM operations
