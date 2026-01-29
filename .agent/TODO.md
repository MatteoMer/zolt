# Zolt-Jolt Compatibility: Current Status

## Status: Stage 2 Sumcheck Proof Polynomials Wrong 🔴

## Session 77 Summary (2026-01-29)

### Progress Made

1. **Fixed config serialization** - trace_length, ram_K, bytecode_K, ReadWriteConfig, OneHotConfig, DoryLayout now match Jolt format exactly

2. **Proof deserialization works** - 91 opening claims, 37 commitments parsed correctly

3. **Stage 1 passes** - Outer Spartan sumcheck verification passes

4. **Found Stage 2 root cause**: The `RamAddress` claim at `SpartanOuter` is correctly ZERO (fibonacci has no loads/stores, so the polynomial is identically zero)

5. **Real issue identified**: The sumcheck output_claim doesn't match expected_claim:
   ```
   output_claim:   [50, 8d, 70, 43, ...]
   expected_claim: [38, d1, cc, 37, ...]
   ```

### Detailed Analysis

Verified via proof parsing:
```
Claim 50: Virtual(poly=29, sumcheck=0) = ZERO  (RamAddress at SpartanOuter)
```
This is CORRECT - fibonacci has no memory operations, so RamAddress polynomial is identically zero.

The expected_claim is computed from 5 instance contributions:
- Instance 0: ProductVirtualRemainder - non-zero contribution
- Instance 1: RamRafEvaluation - zero (correct - no loads/stores)
- Instance 2: RamReadWriteChecking - zero (correct - no memory ops)
- Instance 3: OutputSumcheck - zero (correct)
- Instance 4: InstructionClaimReduction - non-zero contribution

The sumcheck polynomial rounds produce an `output_claim`, but this doesn't match what the verifier computes from the instance expected_output_claims.

**This means the Stage 2 sumcheck polynomials computed by Zolt are wrong.**

### Key Files to Investigate

1. `/home/vivado/projects/zolt/src/zkvm/proof_converter.zig` lines 2700-3200 - Stage 2 batched sumcheck generation
2. Look at how Stage 2 polynomials are computed vs how Jolt computes them

### How Jolt Computes Instance Expected Output Claims

For each instance at r_final (the sumcheck challenges):
1. `ProductVirtualRemainder::expected_output_claim(r)` - evaluates factor polys at r
2. `RamRafEvaluation::expected_output_claim(r)` - evaluates unmap * raf at r
3. `RamReadWriteChecking::expected_output_claim(r)` - evaluates read/write checking
4. `OutputSumcheck::expected_output_claim(r)` - evaluates program I/O check
5. `InstructionClaimReduction::expected_output_claim(r)` - evaluates instruction lookup reduction

The verifier sums these (weighted by batching coeffs) and expects it to match output_claim from proof.

### Next Steps

1. [ ] Add debug prints to Zolt's Stage 2 polynomial computation
2. [ ] Compare Zolt's final round polynomial evaluation with Jolt's expected values
3. [ ] Verify batching coefficients match
4. [ ] Verify each instance's polynomial is computed correctly

### Commits Made
- `0baedb0` - fix: correct Jolt config serialization format

### How to Run Tests

```bash
# Jolt verification test
cd /home/vivado/projects/jolt
ZOLT_LOGS_DIR=/home/vivado/projects/zolt/logs cargo test --features "minimal,zolt-debug" --no-default-features -p jolt-core test_verify_zolt_proof_with_zolt_preprocessing -- --nocapture --ignored

# Zolt proof generation
cd /home/vivado/projects/zolt
zig build -Doptimize=ReleaseSafe && ./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format --export-preprocessing logs/zolt_preprocessing.bin -o logs/zolt_proof_dory.bin
```

## Technical Details
- trace_length: 256
- ram_K: 65536
- bytecode_K: 65536
- Stage 2 max_rounds: 24 (log_ram_k + n_cycle_vars = 16 + 8)

---

## Previous Progress

### Session 76
- Fixed ZOLT header issue using `--jolt-format`

### Earlier Sessions
- Fixed SumcheckId mismatch
- Verified factor polynomial ordering
