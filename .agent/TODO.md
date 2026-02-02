# Zolt-Jolt Compatibility Implementation

## Status: IN PROGRESS - Opening Claims Mismatch

## Session 138 Progress

### Fixed: Polynomial Evaluation Bug

The Stage 5 sumcheck was using incorrect Lagrange interpolation that treated `p_inf` as `p(3)`.
Fixed by using `UniPoly.evaluateToomCookAt()` which correctly converts Toom-Cook evaluations
to coefficients and uses Horner's method.

### Current Issue: Opening Claims Expected Output

The sumcheck now produces the correct `output_claim`, but it doesn't match `expected_claim`.

From Jolt verification:
```
output_claim:   [84, 83, e6, 0a, 81, 4f, 33, 12, ...]  <- matches Zolt's final batched claim!
expected_claim: [c6, 19, df, ae, 44, 5b, ac, 2e, ...]  <- computed from opening claims
```

The output_claim is computed correctly by the sumcheck (Zolt and Jolt agree).
The expected_claim is computed by Jolt's verifier from the opening claims.

This means either:
1. The opening claims Zolt is putting in the proof are incorrect
2. The challenges used to evaluate the opening claims differ between prover and verifier

### Evidence

Opening claims (like LookupTableFlag) match between Zolt's proof and Jolt's reading:
- Zolt: `LookupTableFlag(0) = { 79, 32, 121, 16, ..., 98, 243, 35, 204, 159, 229, 92, 67, 252, 30, 69, 248, 51, 37, 28, 11 }`
- Jolt: `table_flag[0] = [62, f3, 23, cc, 9f, e5, 5c, 43, fc, 1e, 45, f8, 33, 25, 1c, 0b]`

The last 16 bytes match exactly, confirming serialization is correct.

### Next Steps

1. Check if the challenge values used for evaluating opening claims match between prover and verifier
2. Look at how `expected_output_claim()` is computed for each Stage 5 instance
3. Verify the ra_claim computation in InstructionReadRaf

### Test Commands

```bash
# Build Zolt
zig build -Doptimize=ReleaseFast

# Generate proof
./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format --export-preprocessing /tmp/zolt_preprocessing.bin -o /tmp/zolt_proof_dory.bin

# Verify with Jolt
cp /tmp/zolt_*.bin /home/vivado/projects/jolt/
cd /home/vivado/projects/jolt
cargo test --package jolt-core --features zolt-debug test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture
```

## Previous Sessions Summary

- Session 133-137: Various debugging and fixes
- Session 138:
  - Fixed polynomial evaluation bug (using Horner's method)
  - Identified opening claims mismatch as next issue

## Key Files

- `/home/vivado/projects/zolt/src/zkvm/spartan/stage5_prover.zig` - Stage 5 prover
- `/home/vivado/projects/zolt/src/poly/mod.zig` - UniPoly with evaluateToomCookAt
- `/home/vivado/projects/jolt/jolt-core/src/zkvm/instruction_lookups/read_raf_checking.rs` - Instance 2 verification
- `/home/vivado/projects/jolt/jolt-core/src/subprotocols/sumcheck.rs` - Sumcheck verifier
