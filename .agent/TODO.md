# Zolt-Jolt Compatibility Implementation

## Status: Session 79 - Stage 3 FIXED! Stage 5 Now Fails

## Current Issue: Stage 5 Sumcheck Output/Expected Claim Mismatch

### Stage 5 Details
Stage 5 consists of 3 batched sumcheck instances:
1. **RegistersValEvaluation** - Register value evaluation (degree 3, ~6 rounds)
2. **RamRaClaimReduction** - RAM read address claim reduction (degree 2, log_K + log_T rounds)
3. **InstructionReadRafSumcheck** - Instruction lookups read+RAF checking (128 + 6 = 134 rounds)

Max num_rounds = 134 (InstructionReadRafSumcheck dominates)

### Error
```
Sumcheck verification failed!
  output_claim:   [d8, e4, b0, e7, 00, aa, 9b, d6, ...]
  expected_claim: [eb, 4e, 32, 94, 8e, 12, 6a, 42, ...]
Verification failed: Stage 5
```

### Next Steps
1. Add debug output to Stage 5 to identify which instance(s) fail
2. Check if there's a similar Phase 2 bug in the Stage 5 prover
3. Verify Stage 5 round polynomials match between Zolt and Jolt

## Fix Applied This Session (Session 79)

### CRITICAL FIX: Phase 2 computeRoundEvalsPhase2 used wrong array length
- **Bug**: Both ShiftSumcheck and RegistersClaimReduction used `eq_outer.len / 2` (or `eq.len / 2`)
  as the loop bound in `computeRoundEvalsPhase2`. This is the original allocation size (suffix_size),
  which never changes after allocation.
- **Correct**: Should use `self.current_witness_size / 2`, which tracks the active size that shrinks
  after each `bindPhase2` call.
- **Symptom**: After the first Phase 2 bind (round 3→4), the loop iterated over stale data in array
  positions beyond the active range, producing incorrect round polynomial evaluations.
- **Detection**: Per-round Phase 2 verification showed `match=false` at ROUND_4 (after round 4 bind).
- **Files fixed**: `src/zkvm/spartan/stage3_prover.zig` (both ShiftSumcheck and RegistersClaimReduction)

### Result: Stage 1 ✅, Stage 2 ✅, Stage 3 ✅, Stage 4 ✅, Stage 5 ❌

## Verified Working Components

- ✅ Blake2b transcript matches between Zolt and Jolt
- ✅ Tau challenges generated correctly (count = 8)
- ✅ UncompressedUniPoly_begin transcript state matches
- ✅ r0 values match between Zolt prover and Jolt verifier
- ✅ UniSkip check_sum_evals passes
- ✅ No R^2 scaling (correctly removed)
- ✅ Zero constraint violations
- ✅ Stage 1 sumcheck verification PASSES
- ✅ Stage 2 sumcheck verification PASSES
- ✅ Opening claims (R1CS input MLE evaluations) match
- ✅ Stage 3 sumcheck verification PASSES (shift, instr, reg all match)
- ✅ Stage 4 implicitly passes (Stage 5 error occurs after it)
- ❌ Stage 5 sumcheck output_claim ≠ expected_claim

## Debug Commands

```bash
# Build Zolt
zig build -Doptimize=ReleaseFast

# Generate proof
./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format -o /tmp/zolt_proof_dory.bin --export-preprocessing /tmp/zolt_preprocessing.bin --trace-length 64

# Verify with Jolt
cd /home/vivado/projects/jolt && cargo test -p jolt-core --features zolt-debug --lib test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture
```
