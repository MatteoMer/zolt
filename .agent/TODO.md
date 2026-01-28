# Zolt-Jolt Compatibility: Stage 4 Input Claim Issue

## Status: In Progress

## Recent Changes (Session 68)

1. **Removed termination bit workaround from RWC prover** (`cee6b7e`)
   - RWC's val_init now matches Jolt's initial RAM (no termination bit)
   - For programs without RAM ops, `input_claim_val_eval = 0` should be correct

2. **Identified OutputSumcheck workaround** still in place
   - Sets termination in BOTH val_init and val_final
   - This makes OutputSumcheck internally consistent but breaks ValFinal input_claim

## Root Cause Analysis

### The Termination Bit Problem

**Jolt's approach:**
- Emulator records termination write as a real RISC-V store instruction
- This creates a sparse matrix entry: `inc=1`, `wa=term_addr`
- `val_init[term] = 0` (initial RAM)
- `val_final[term] = 1` (after termination write)
- OutputSumcheck: `val_final - val_init = 1 = Σ inc * wa` ✓

**Zolt's approach (with workaround):**
- Tracer does NOT record termination write
- No sparse entry for termination
- Workaround sets BOTH: `val_init[term] = 1`, `val_final[term] = 1`
- OutputSumcheck: `val_final - val_init = 0 = Σ inc * wa` ✓ (internally consistent)

**The problem:**
- `output_val_final_claim` includes termination=1 (from OutputSumcheck's val_final)
- `init_eval_for_val_final` computed from `config.initial_ram` has termination=0
- `input_claim_val_final = 1 - 0 = termination_contribution ≠ 0`

But the NOTES say Zolt sends zeros... need to verify actual debug output.

## Proper Fix Options

1. **Have Zolt's tracer record termination write** (like Jolt)
   - Modify `src/tracer/mod.zig` to emit a store instruction for termination
   - This would make sparse matrix entries correct
   - OutputSumcheck's val_final would naturally have termination=1

2. **Remove ALL termination workarounds** and accept that:
   - For no-RAM programs, all input_claims = 0
   - Need to verify Jolt also produces zeros for such programs

3. **Align val_final/val_init computation** with what verifier expects
   - Ensure `output_val_final_claim` and `init_eval_for_val_final` use same basis

## Next Steps

1. Add debug output to print actual values of:
   - `output_val_final_claim` from OutputSumcheck
   - `init_eval_for_val_final` from `computeInitialRamEval`
   - `input_claim_val_final` before appending to transcript

2. Verify if Jolt truly expects non-zero for Fibonacci
   - May need to run Jolt on Fibonacci and check debug output
   - The NOTES might be from a different test case

3. If Jolt expects zeros for Fibonacci:
   - Ensure both workarounds are properly consistent
   - val_init in OutputSumcheck should match `computeInitialRamEval`

## Files Involved

- `/home/vivado/projects/zolt/src/zkvm/ram/read_write_checking.zig` - RWC workaround (REMOVED)
- `/home/vivado/projects/zolt/src/zkvm/ram/output_check.zig` - OutputSumcheck workaround (STILL PRESENT)
- `/home/vivado/projects/zolt/src/zkvm/proof_converter.zig` - input_claim computation
- `/home/vivado/projects/zolt/src/tracer/mod.zig` - Emulator termination handling

## Session Summary

- Understood Jolt's opening point handling (reconstructed, not stored)
- Understood Jolt's verifier RECOMPUTES init_eval from preprocessing
- Removed RWC termination workaround
- Need to address OutputSumcheck termination workaround
- Need to verify actual debug output to confirm hypothesis
