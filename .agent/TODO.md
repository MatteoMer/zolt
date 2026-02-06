# Zolt-Jolt Compatibility Implementation

## Status: Session 103 - Fixed Per-Cycle Claims, Sumcheck Polynomial Still Mismatch

### Progress Made

1. **Fixed eq polynomial multiplication** - Changed `mulHiBigIntU128` to standard `mul()` for F field elements
2. **Fixed opcode handling for 0x13 (I-type ALU)**:
   - Only funct3=0 (ADDI) uses AddOperands
   - Other I-type ALU (SLLI, SLTI, etc.) use interleaved operands
3. **Fixed 0x1b and 0x3b handling** in both switches in Stage 5
4. **Updated R1CS constraints.zig** to match Stage 5's opcode handling

### Current Status

Per-cycle MLE claims NOW MATCH between Stage 2 and Stage 5:
- `output_sum` = `rv_claim` ✓
- `left_sum` = `left_op_claim` ✓
- `right_sum` = `right_op_claim` ✓

### Remaining Issue

Stage 5 sumcheck verification still fails:
- `output_claim` (prover): `9b66e75f29b6733f22cb13e2d582630c...`
- `expected_claim` (verifier): `ce7ef7a72c81030ea8e899fcc2d52002...`

The sumcheck polynomial values don't produce the correct expected output. This means the sumcheck polynomial coefficients are being computed incorrectly somewhere.

### Investigation Points

1. **Sumcheck polynomial computation**: The prover's round polynomials might not be correct
2. **Batching coefficients**: The three instances are batched together, batching might be wrong
3. **Scaling factors**: Each instance has different number of variables, scaling might be off
4. **Challenge extraction**: The challenges from transcript might not match Jolt's expectations

### Debug Evidence

From Jolt verifier output:
```
[SUM DEBUG] expected_output_claim (sum of all): [ce, 7e, f7, a7, ...]
[SUM DEBUG] manual f0+f1+f2: [73, b1, a9, 6a, ...]  <- Different from expected!
```

The `manual f0+f1+f2` (sum of individual instance claims) doesn't match `expected_output_claim`. This suggests the individual instance claims computed by Jolt's verifier don't add up correctly, OR the prover's batched sum is wrong.

### Next Steps

1. Compare Stage 5 sumcheck polynomial coefficients round-by-round with Jolt expectations
2. Verify the batching coefficient computation matches Jolt
3. Check if the scaling factors for shorter instances are correct
4. Debug the first few sumcheck rounds to find where divergence starts

### Test Commands

```bash
# Build and generate proof
zig build -Doptimize=ReleaseFast
./zig-out/bin/zolt prove examples/fibonacci.elf -o /home/vivado/projects/jolt/zolt_proof.bin --trace-length 64 --jolt-format --export-preprocessing /home/vivado/projects/jolt/zolt_preprocessing.bin

# Test with Jolt
cd /home/vivado/projects/jolt
cargo test -p jolt-core test_verify_zolt_proof_with_zolt_preprocessing --release --features zolt-debug -- --ignored --nocapture 2>&1 | tail -100
```
