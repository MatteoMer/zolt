# Zolt-Jolt Compatibility Implementation

## Status: IN PROGRESS - Stage 5 sumcheck output doesn't match expected

## Session 6 Progress

### Fixed Issues
1. **ra_chunks computation**: Changed from recomputing claims using eq(r_cycle', j) * ra_chunk_weights[i][j]
   to using ra_chunk_weights[i][0] after binding (matching Jolt's final_sumcheck_claim())

### Current Issue: Sumcheck output_claim doesn't match expected_claim

The verification failure shows:
```
output_claim:   [c8, d4, 1b, fc, ...]  <- What the sumcheck polynomial chain produces
expected_claim: [e5, b7, 3b, 32, ...]  <- Computed from eq * ra_claim * (val + gamma * raf)
```

The expected_claim formula is:
```
eq_eval_r_reduction * ra_claim * (val_claim + gamma * raf_claim)
```

### What's Working
- ra_chunks claims are correctly serialized and match what Jolt receives
- Sumcheck property p(0)+p(1)=claim holds for all 8 cycle rounds
- Table flags, raf_flag claims appear to match

### What's Still Wrong
- The polynomial coefficients themselves must be wrong (even though p(0)+p(1)=claim)
- This means the polynomial p(X) has wrong shape - different from what Jolt expects

### Key Hypothesis

The issue is likely in how the polynomial is computed during cycle rounds:
1. The eq_prefix extraction via dividing by (1-r_round) might not match Jolt's split-eq approach
2. The product polynomial structure might be different

### Investigation Needed

1. Compare first cycle round (128) polynomial coefficients between Zolt and Jolt
2. Verify the eq_prefix computation matches the expected structure
3. Check if combined_vals rematerialization is correct
4. Compare the claim update chain: how claim evolves through rounds

### Debug Values Comparison

Jolt's ra_claims (LE, first 16 bytes):
```
ra_claims[0] = [a5, 5e, c7, 72, 66, 8e, 13, 27, 21, 0d, f3, 0e, 35, 26, 9b, 11]
```

Zolt's ra_chunks (BE, first 16 bytes):
```
ra_chunks[0] = [17, 9b, 26, 35, 0e, f3, 0d, 21, 27, 13, 8e, 66, 72, c7, 5e, a5]
```

These are the SAME values (endianness reversed) - serialization is correct!

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

### Key Files

- `/home/vivado/projects/zolt/src/zkvm/spartan/stage5_prover.zig` - Stage 5 prover
- `/home/vivado/projects/zolt/src/poly/mod.zig` - UniPoly with finishMlesProductSumFromEvals
- `/home/vivado/projects/jolt/jolt-core/src/zkvm/instruction_lookups/read_raf_checking.rs` - InstructionReadRaf prover/verifier
- `/home/vivado/projects/jolt/jolt-core/src/subprotocols/mles_product_sum.rs` - Jolt's finish function
