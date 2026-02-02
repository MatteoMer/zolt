# Zolt-Jolt Compatibility Implementation

## Status: IN PROGRESS - Stage 5 Verification Mismatch

## Current Issue

Stage 5 sumcheck fails: output_claim from proof doesn't match expected_claim computed by verifier.

### Evidence

**Sumcheck verification failed:**
```
output_claim:   [ce, b6, 9e, 59, 52, e3, 16, ed, 32, a4, 41, 28, 30, 87, d9, 5c, ...]
expected_claim: [b7, c0, 2e, 8e, c4, 3e, bd, 5d, 3a, 49, 78, 90, c1, e8, b8, ed, ...]
```

### Verified Components (All Match Between Zolt and Jolt)

1. **InstructionRa claims** ✓ - All 8 chunks match
2. **ra_product** ✓ - Product of InstructionRa chunks matches
3. **LookupTableFlag claims** ✓ - All 42 flags match (only 0, 1, 9 are non-zero for Fibonacci)
4. **raf_claim** ✓ - Formula `(1-raf_flag)*(left+gamma*right) + raf_flag*gamma*identity` matches
5. **eq_eval_r_reduction** ✓ - Matches between Zolt and Jolt
6. **left/right/identity prefix evaluations** ✓ - Match between Zolt and Jolt
7. **batching coefficients** ✓ - batch0, batch1, batch2 all match
8. **input_claims** ✓ - RegistersVal and other Stage 5 inputs now match

### Suspected Issue

The sumcheck polynomial evaluations at each round might be producing incorrect round polynomials, leading to a different final output_claim even though all the final opening claims are correct.

The expected_output_claim for each instance is:
1. Instance 0 (RegistersValEvaluation): `inc_claim * wa_claim * lt_claim`
2. Instance 1 (RamRaClaimReduction): `eq_combined * ra_claim * ...`
3. Instance 2 (InstructionReadRaf): `eq_r_reduction * ra_claim * (val_claim + gamma * raf_claim)`

The prover needs to compute round polynomials such that:
- At each round r, p_r(X) has degree ≤ degree_bound
- p_r(0) + p_r(1) = previous_claim (or input_claim for r=0)
- The final evaluation matches expected_output_claim

### Next Steps

1. Add debug output to compare round polynomial coefficients between Zolt prover and Jolt verifier
2. Check if the polynomial degree or structure differs
3. Verify the sumcheck polynomial construction in stage5_prover.zig

### Key Files

- `/home/vivado/projects/zolt/src/zkvm/spartan/stage5_prover.zig` - Stage 5 prover
- `/home/vivado/projects/jolt/jolt-core/src/subprotocols/sumcheck.rs` - Jolt sumcheck verifier
- `/home/vivado/projects/jolt/jolt-core/src/zkvm/instruction_lookups/read_raf_checking.rs` - InstructionReadRaf verifier

## Test Commands

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

## Previous Session Findings

1. Fixed suffix MLEs (LsbSuffix, Pow2Suffix, etc.) to return 1 when len==0
2. Fixed Stage 5 input claim reading - now correctly reads from opening_claims
3. Verified expanding table implementations are identical between Zolt and Jolt
4. Confirmed phase structure matches (16 phases, 8 chunks for small traces)
5. Verified Stages 1-4 pass verification
