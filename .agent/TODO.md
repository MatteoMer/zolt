# Zolt-Jolt Compatibility TODO

## Current Status: January 6, 2026 - Session 61

**STATUS: All 712 Zolt tests pass, but Jolt verification fails**

### Summary

The core algorithms are correct (all unit tests pass), but when running with real execution data, the verification fails because `expected_output_claim` doesn't match `output_claim`.

### Key Findings

1. **All 712 Zolt tests pass** - Including the cross-verification test
2. **Opening claims serialize correctly** - Bytes in proof match what Jolt reads
3. **R1CS constraint order matches** - FIRST/SECOND_GROUP_INDICES match Jolt
4. **Transcript compatibility verified** - All states and challenges match
5. **Tau factor matches** - `lagrange_tau_r0` is identical

### The Remaining Issue

When running the real proof:
- Zolt output_claim: `6773516909001919453588788632964349915676722363381828976724283873891965463518`
- Jolt expected: `2434835346226335308248403617389563027378536009995292689449902135365447362920`

From the tests, we know:
- `tau_factor = tau_high_bound_r0 * tau_bound_r_tail_reversed` matches
- `Az/Bz MLE` computations work correctly in unit tests
- `inner_sum_prod` from prover should match verifier's

### Potential Root Causes

1. **Witness generation** - The real `fromTraceStep` may produce different values than Jolt
2. **Cycle indexing** - How cycles map to MLE indices in real execution
3. **Padding** - How trace padding is handled in real execution
4. **Integration bug** - Something in `proof_converter.zig` doesn't match the tested paths

### Next Steps for Future Sessions

1. **Add debug output in real proof generation** to print Az/Bz values for first few cycles
2. **Compare witness values** between Zolt's trace and Jolt's trace for same program
3. **Check trace padding** - ensure padding zeros are in the right places
4. **Verify cycle count** - ensure num_cycles matches between prover and verifier

### Test Commands

```bash
# All tests pass
zig build test

# Generate real proof
./zig-out/bin/zolt prove /tmp/jolt-guest-targets/fibonacci-guest-fib/riscv64imac-unknown-none-elf/release/fibonacci-guest --jolt-format --input-hex 32 --export-preprocessing /tmp/zolt_preprocessing.bin -o /tmp/zolt_proof_dory.bin

# Jolt verification (fails)
cd /Users/matteo/projects/jolt
cargo test --package jolt-core test_verify_zolt_proof -- --ignored --nocapture
```

### Files to Investigate

1. `/Users/matteo/projects/zolt/src/zkvm/r1cs/constraints.zig` - `fromTraceStep` witness generation
2. `/Users/matteo/projects/zolt/src/zkvm/proof_converter.zig` - Integration of components
3. `/Users/matteo/projects/jolt/jolt-core/src/zkvm/r1cs/inputs.rs` - Jolt's `from_trace` for comparison
