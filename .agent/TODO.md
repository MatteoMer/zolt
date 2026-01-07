# Zolt-Jolt Compatibility - Session 18 Progress

## Status Summary

### ✅ COMPLETED
- All 712 internal tests pass
- Stage 1 passes Jolt verification completely
- Stage 2 sumcheck constraint (s(0)+s(1) = claim) passes ALL 26 rounds
- All 5 instance provers integrated and producing round polynomials
- Individual RWC opening claims (ra, val, inc) now computed
- Fixed double-free bug in InstructionLookupsProver
- Fixed computeEq for big-endian interpretation

### ❌ REMAINING ISSUE
Stage 2 verification fails because:
- `output_claim` ≠ `expected_output_claim`
- Difference is ~80e75 (a large number, similar to one instance contribution)

**Root Cause Investigation**:
The expected_output_claim is computed from our opening claims:
```
eq_eval_cycle * ra_claim * (val_claim + gamma * (val_claim + inc_claim))
```

Our claims may not correspond to the actual polynomial values that our
sumcheck prover is computing. Need to trace through polynomial construction.

## Instance Contributions (from Jolt debug output)

For Stage 2:
- Instance 0 (ProductVirtual): 8369935295803813061188230597022735237
- Instance 1 (RAF): 7819199229764417167347615577448458590
- Instance 2 (RWC): 5221874132503095678699145889033794949
- Instance 3 (OutputSumcheck): 0
- Instance 4 (InstructionLookups): 0

## Next Steps

1. **Trace RWC polynomial construction** - Verify sparse matrix entries match actual memory trace
2. **Compare with Jolt's expected values** - Check if our ra/val/inc evaluations match what Jolt expects
3. **Debug Instance 0 (ProductVirtual)** - May also have incorrect contribution

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
