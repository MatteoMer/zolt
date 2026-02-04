# Zolt-Jolt Compatibility Implementation

## Status: Session 57 - Termination Write Fix Applied

## Progress This Session

### Key Discovery: Termination Write Missing
- Found that Zolt was filtering out termination/panic writes from the Inc polynomial
- Jolt DOES include termination writes in the trace (guest program writes via `core::ptr::write_volatile`)
- Fixed by:
  1. Re-enabled termination write recording in tracer/mod.zig
  2. Removed the filter in val_evaluation.zig that skipped termination/panic addresses

### Results After Fix
- ValFinal now shows non-zero `inc_claim` and `wa_claim` (previously both were 0)
- Stage 4 sumcheck passes the round polynomial checks (p(0)+p(1) = claim)
- However, the **final output claim** still doesn't match expected

### Current Issue: Output Claim Mismatch
```
output_claim:   [42, c4, df, c9, ...]
expected_claim: [c2, d9, 64, f7, ...]
```

The expected claim is computed as:
- Instance 0 (RegistersRWC): claim * coeff[0]
- Instance 1 (ValEval): claim * coeff[1]
- Instance 2 (ValFinal): claim * coeff[2]
Sum of these should equal output_claim from sumcheck.

### Potential Root Cause
The "hint mechanism" for ValEval/ValFinal may be computing incorrect polynomials:
- When instance becomes active, I set `p(1) = individual_claim - p(0)`
- This ensures `p(0) + p(1) = claim` for the round check
- But the prover polynomial evaluations might not match verifier expectations

### Files Modified This Session

- `/home/vivado/projects/zolt/src/tracer/mod.zig`:
  - Re-enabled `recordTerminationWrite` function
  - Records termination write to RAM trace and memory state

- `/home/vivado/projects/zolt/src/zkvm/ram/val_evaluation.zig`:
  - Removed the filter that skipped termination/panic writes from Inc polynomial

## Next Steps

1. **Debug the output claim computation** - Add more debug output to trace:
   - What are the individual claims after final round?
   - Do they match what the verifier expects?
   - Is the prover polynomial evaluation consistent with stored openings?

2. **Check if hint mechanism is correct** for active instances:
   - Compare Zolt's polynomial construction with Jolt's `from_evals_and_hint`
   - Verify the modified p(1) and p(2) values are self-consistent

3. **Verify opening claims** match between prover and verifier:
   - inc_claim, wa_claim, lt_eval at the final sumcheck point
   - These determine the expected_output_claim

### Test Commands

```bash
# Build Zolt
zig build -Doptimize=ReleaseFast

# Generate proof
./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format -o /tmp/zolt_proof_dory.bin --export-preprocessing /tmp/zolt_preprocessing.bin --trace-length 64

# Verify with Jolt
cd /home/vivado/projects/jolt && cargo test -p jolt-core --features zolt-debug --lib test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture
```

### Key Files

- `/home/vivado/projects/zolt/src/zkvm/proof_converter.zig` - Stage 4 batched sumcheck
- `/home/vivado/projects/zolt/src/zkvm/ram/val_evaluation.zig` - ValEval prover + Inc polynomial
- `/home/vivado/projects/zolt/src/zkvm/ram/val_final.zig` - ValFinal prover
- `/home/vivado/projects/zolt/src/tracer/mod.zig` - Emulator + termination write
