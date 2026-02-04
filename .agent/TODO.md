# Zolt-Jolt Compatibility Implementation

## Status: Session 50+ - Stage 4 Batched Sumcheck Mismatch

## Current Investigation - Stage 4 Verification Failure

### What We Know

1. **Stage 3 passes** - The r_cycle values match correctly
2. **Stage 4 r_cycle initialization is correct** - Zolt uses the same r_cycle as Jolt
3. **Stage 4 round polynomials are DIFFERENT** - Zolt's coefficients don't match Jolt's

### The Mismatch

Jolt Stage 4 Round 0 coefficients (from verification):
```
[0]: [37, 2d, 28, 8e, 4c, 71, 68, 11, 2c, c9, 23, a2, 70, a9, 6c, 1b]
```

Zolt Stage 4 Round 0 coefficients (from prover):
```
c0 = { 21, 195, 222, 234, 101, 75, 191, 73, 85, 84, 9, 224, 170, 42, 91, 10 }
```

These are COMPLETELY different. This means the round polynomial computation is wrong.

### Stage 4 Batched Sumcheck Structure

Stage 4 is a batched sumcheck with 3 instances:
1. `RegistersReadWriteChecking` - the main registers checking
2. `RamValEvaluation` - for RAM value evaluation
3. `RamValFinal` - for final RAM check

The batched sumcheck combines them:
```
batched_poly(X) = coeff[0] * regs_poly(X) + coeff[1] * val_eval_poly(X) + coeff[2] * val_final_poly(X)
```

### Key Question

Is Zolt computing all three instance polynomials correctly? Looking at the code, Zolt seems to be computing only the RegistersReadWriteChecking polynomial and the other two are zero. But they shouldn't be zero - they should have actual polynomial evaluations.

### Next Steps

1. Check how Zolt computes RamValEvaluation polynomial
2. Check how Zolt computes RamValFinal polynomial
3. Verify the batching coefficients match Jolt
4. Ensure the polynomial evaluations are combined correctly

### Files to Check

- `/home/vivado/projects/zolt/src/zkvm/proof_converter.zig` - Stage 4 proof generation
- `/home/vivado/projects/jolt/jolt-core/src/zkvm/registers/val_evaluation.rs` - RamValEvaluation
- `/home/vivado/projects/jolt/jolt-core/src/zkvm/ram/val_final.rs` - RamValFinal

## Test Commands

```bash
# Build Zolt
zig build -Doptimize=ReleaseFast

# Generate proof
./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format -o /tmp/zolt_proof_dory.bin --export-preprocessing /tmp/zolt_preprocessing.bin --trace-length 64

# Verify with Jolt
cd /home/vivado/projects/jolt && cargo test -p jolt-core --features zolt-debug --lib test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture
```

## SESSION_ENDING

Context running low. The key finding is that Stage 4 round polynomials don't match because:
1. Zolt is only computing RegistersReadWriteChecking (Instance 0)
2. Instances 1 (RamValEvaluation) and 2 (RamValFinal) appear to produce zero contributions
3. But Jolt's expected output_claim shows non-zero contributions from all three

Next session should focus on:
- Understanding what RamValEvaluation and RamValFinal compute
- Implementing proper polynomial evaluations for these instances
- Ensuring the batched sumcheck correctly combines all three
