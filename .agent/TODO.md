# Zolt-Jolt Compatibility Implementation

## Status: Session 101 - Diagnosing Challenge Point Mismatch

### Current Issue

Stage 5 sumcheck verification fails because Instance 2 (LookupsReadRaf) prover polynomial chain evaluation differs from verifier's expected formula.

**Root Cause Identified:**
The `left_op_claim` and `right_op_claim` are computed at `r_spartan` (Stage 2 challenges), but Stage 5 uses `r_reduction` (Stage 3 challenges) for the eq polynomial. These are DIFFERENT challenge vectors!

- `r_spartan_for_instr[0] = { 43, 70, 202, 83...` (hex 0x2b, 0x46...)
- `r_reduction_be[0] = ...a2e00e6d3d591508...` (completely different)

### Fixes Applied This Session

1. **unexpanded_pc fix**: Changed Stage 5 to use `step.unexpanded_pc` instead of `step.pc` for left_input when `left_is_pc = true`. This matches R1CS constraints.zig.

2. **0x13 opcode fix**: Changed Stage 5 to treat ALL 0x13 opcodes (I-type ALU) as AddOperands, matching R1CS constraints.zig. Previously, only ADDI (funct3=0) was treated as AddOperands.

### What's Verified Working

- Per-cycle witness values now MATCH between Stage 2 and Stage 5
  - j=0: left=0, right=0x8000 ✓
  - j=1: left=0, right=0x8001 ✓
  - j=2: left=0, right=0x8011 ✓ (was 0x10 before fix)
  - j=3: left=0, right=0x80000044 ✓
  - j=4: left=0, right=0x9 ✓

- `output_sum (Σ eq*output)` MATCHES `rv_claim` ✓

### What's Still Broken

- `left_sum (Σ eq*left)` ≠ `left_op_claim`
- `right_sum (Σ eq*right)` ≠ `right_op_claim`
- This causes `computed_sum` ≠ `lookups_input`

### Why The Mismatch Exists

In Jolt's architecture:
1. Stage 2 (Batched Sumcheck): InstructionLookupsClaimReduction computes claims at its own sumcheck challenges
2. Stage 3 (InstructionClaimReduction): A SEPARATE sumcheck that takes Stage 2's output claims
3. Stage 5 (InstructionReadRaf): Uses `r_reduction` from InstructionClaimReduction (Stage 3)

The claims are stored with `SumcheckId::InstructionClaimReduction`, meaning they should be evaluated at the InstructionClaimReduction point.

**The Question:** What exactly IS `r_reduction`?
- It should be the InstructionClaimReduction sumcheck challenges
- But in Zolt, it's coming from Stage 3, while the claims are computed in Stage 2

### Next Steps

1. Understand exactly where `r_reduction` should come from
2. Either:
   a. Make the claims be computed at `r_reduction` point, OR
   b. Use `r_spartan` in Stage 5 instead of `r_reduction`
3. Check how Jolt's InstructionReadRafSumcheckProver gets its params.r_reduction

### Test Commands

```bash
# Build and generate proof
zig build -Doptimize=ReleaseFast
./zig-out/bin/zolt prove examples/fibonacci.elf -o /home/vivado/projects/jolt/zolt_proof.bin --trace-length 64 --jolt-format --export-preprocessing /home/vivado/projects/jolt/zolt_preprocessing.bin

# Test with Jolt
cd /home/vivado/projects/jolt
cargo test -p jolt-core test_verify_zolt_proof_with_zolt_preprocessing --release --features zolt-debug -- --ignored --nocapture 2>&1 | tail -100
```
