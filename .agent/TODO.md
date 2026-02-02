# Zolt-Jolt Compatibility Implementation

## Status: IN PROGRESS - Stage 5 sumcheck output doesn't match expected

## Session 6 Progress (Continuation)

### Root Cause Identified!

**The `r_reduction` values from Stage 3 (InstructionClaimReduction) don't match between Zolt and Jolt!**

This is the actual root cause of the Stage 5 verification failure.

### Evidence

Jolt's `r_reduction[0]` (bytes 16-31 LE):
```
0d 8d 89 b0 c0 ef 00 b0 84 a4 8a 1b 0b 14 34 07
```

Zolt's `r_reduction[0]` (toBytesBE()[16..32]):
```
a2 70 af 2a 26 b8 57 9a 38 19 e3 4d 5f 35 02 0d
```

These are COMPLETELY DIFFERENT values!

### What We Verified As Matching

1. **Polynomial coefficients at round 128**: MATCH ✓
2. **Challenges for rounds 128-135**: ALL MATCH ✓
3. **ra_claims (InstructionRa)**: MATCH ✓
4. **table_flags (LookupTableFlag)**: MATCH ✓
5. **raf_flag (InstructionRafFlag)**: MATCH ✓
6. **Instance 0 (RegistersValEvaluation) claims**: MATCH ✓
7. **Instance 1 (RamRaClaimReduction) claims**: MATCH ✓
8. **Stage 5 input claims**: MATCH ✓

### Why This Causes Failure

The expected_claim for Instance 2 (InstructionReadRaf) is:
```
expected = eq(r_reduction, r_cycle') * ra_claim * (val_claim + gamma * raf_claim)
```

The `eq(r_reduction, r_cycle')` term uses `r_reduction` from Stage 3. If Zolt's `r_reduction` differs from what Jolt computes, then:
1. The eq polynomial in the sumcheck uses different values
2. The expected_claim formula uses Jolt's computed r_reduction
3. These don't match → verification fails

### What This Means

The polynomial coefficients match because Zolt computes the sumcheck correctly using ITS r_reduction values. The challenges match because they're derived from matching coefficients via transcript.

But the expected_claim uses Jolt's RECOMPUTED r_reduction from the transcript. Since Zolt's Stage 3 coefficients must differ from what Jolt expects, Jolt's verifier derives different r_reduction values than what Zolt used to compute the Stage 5 polynomial.

### Next Steps

1. **Investigate Stage 3 (InstructionClaimReduction) sumcheck**
   - Compare polynomial coefficients at each round
   - Find where the divergence starts
   - This sumcheck has 8 rounds (n_cycle_vars)

2. **Check what feeds into Stage 3**
   - The input claim for InstructionClaimReduction
   - The polynomial structure

3. **Trace the transcript state**
   - Compare transcript state before Stage 3 between Zolt and Jolt
   - Any earlier mismatch will cause all subsequent challenges to differ

### Key Files for Stage 3 Investigation

- `/home/vivado/projects/zolt/src/zkvm/claim_reductions/instruction_lookups.zig`
- `/home/vivado/projects/jolt/jolt-core/src/zkvm/instruction_lookups/claim_reduction.rs`

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
