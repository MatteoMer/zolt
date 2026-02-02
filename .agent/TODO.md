# Zolt-Jolt Compatibility Implementation

## Status: IN PROGRESS - Stage 5 Transcript Divergence

## Session 125 Summary

### Progress Made

1. **Fixed combined_vals rematerialization bug**
   - Cycles without lookup tables were getting `combined_val = 0`
   - Jolt ALWAYS adds RAF contribution regardless of table
   - Fixed: Now cycles without tables get `raf_interleaved` or `raf_identity`

2. **Verified eq_evals match**
   - After eq_evals reinitialization fix, `eq_evals[0] == eq_eval_r_reduction` ✓
   - The eq polynomial is now correct after all bindings

3. **Verified ra_chunk computation logic is correct**
   - Bit extraction: `bit_index = LOOKUPS_LOG_K - 1 - round` correctly processes MSB first
   - Chunk assignment: `chunk_idx = round / 16` correctly groups 16 rounds per chunk
   - Factor computation: `eq(bit, challenge) = (1-r) if bit=0 else r` is correct
   - For cycles with all-zero high bits, ra_chunks 0-3 are uniform (expected behavior)

### Current Issue

**Transcript divergence causes different challenges between Zolt and Jolt**

Debug shows Stage 5 Round 0 challenges are different:
```
Zolt Round 0: 1a4f09881ff874890d8d8d3810780e797ca25a17c12902cb92c0d5878c3b73da
Jolt r[0]:    15b2ebced7ca0d488e1f5913aabdd05a
```

This causes all subsequent rounds to diverge, leading to wrong ra_chunk values.

### Root Cause Analysis

The transcript state must have diverged before Stage 5 Round 0. Possible causes:

1. **Earlier stage polynomials are different**
   - Stages 1-4 append polynomial coefficients to transcript
   - If any coefficient differs, transcript state changes

2. **Instance 0 or 1 polynomials are different**
   - Stage 5 is a batched sumcheck with 3 instances
   - Instances 0 (RegistersValEvaluation) and 1 (RamRaClaimReduction) run first
   - If their polynomials differ, challenges for Instance 2 diverge

3. **Polynomial compression format mismatch**
   - Jolt uses specific compressed format `[c0, c2, c3, ..., cd]` (skipping c1)
   - Any format difference changes transcript

### Key Observation

The ra_chunk values after all bindings are completely different:
```
Zolt ra_chunks[0] = 72c54fffc84783cff5628ecc74b37775
Jolt ra_claims[0] = 12109e5de8bae83db5b8fca2612309a8
```

But the ra_chunk computation LOGIC is verified correct. The difference comes from using different challenges during address rounds.

### Next Steps

1. **Trace transcript state at Stage 5 start**
   - Compare transcript hash/state between Zolt and Jolt
   - Identify which previous message caused divergence

2. **Verify Instance 0 and 1 polynomials match**
   - RegistersValEvaluation: check inc_evals, wa_evals, lt_evals
   - RamRaClaimReduction: check the PhaseCycle polynomial computation

3. **Check polynomial format**
   - Verify compressed format matches exactly
   - Check byte order of coefficients

### Key Files

**Zolt:**
- `src/zkvm/spartan/stage5_prover.zig` - Stage 5 batched sumcheck prover
- `src/zkvm/proof_converter.zig` - Proof generation orchestration

**Jolt:**
- `jolt-core/src/zkvm/instruction_lookups/read_raf_checking.rs` - InstructionReadRaf sumcheck

### Test Commands

```bash
# Build Zolt
zig build -Doptimize=ReleaseFast

# Generate proof
./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format --export-preprocessing logs/zolt_preprocessing.bin -o logs/zolt_proof_dory.bin

# Copy to /tmp for Jolt test
cp logs/zolt_*.bin /tmp/

# Verify with Jolt
cd /home/vivado/projects/jolt
cargo test --package jolt-core --features zolt-debug test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture
```
