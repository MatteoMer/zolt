# Zolt-Jolt Compatibility Implementation

## Status: IN PROGRESS - Stage 5 Prefix-Suffix Debugging

## Session 99 Progress

### Task: Debug prefix-suffix decomposition Stage 5 verification failure

### Completed This Session

1. **Implemented RafDecomposition** (`prefix_suffix_prover.zig`):
   - Created `RafDecomposition` struct with Q accumulators
   - Implemented `initQRaf` for fused initialization of left/right/identity Q arrays
   - Implemented `proverMsgRaf` computing γ*left + γ²*(identity + right) evaluations
   - Added `uninterleaveBitsLeft/Right` helpers for operand extraction

2. **Added fromU128 to BN254Scalar** (`field/mod.zig`):
   - BN254Scalar now supports creating field elements from 128-bit values

3. **Integrated prefix-suffix in stage5_prover.zig**:
   - Initialize RAF decompositions for left/right/identity at phase 0
   - Call `proverMsgReadChecking` + `proverMsgRaf` in address rounds
   - Add `suffix_polys.bindAll` and RAF binding after each challenge
   - Add prefix checkpoint updates every 2 rounds
   - Add phase transitions every 16 rounds with Q reinitialization

### Current Status

- Stages 1-4: PASS
- Stage 5: FAIL - Sumcheck verification failure
  - output_claim doesn't match expected_claim
  - The prover runs through all 136 rounds but produces wrong final claim

### What Needs Investigation

1. **proverMsgReadChecking evaluation logic** - May have bugs in:
   - How prefix MLEs are computed at c=0 and c=2
   - How suffixes are combined with prefixes
   - The quadratic interpolation formula

2. **proverMsgRaf evaluation logic** - May have bugs in:
   - How Q accumulators are structured
   - The summation over half-indices
   - The γ*left + γ²*(right + identity) combination

3. **Phase transition handling**:
   - When Q arrays are reinitialized
   - How u_evals should be updated through expanding tables

4. **Prefix MLE implementations in prefixes.zig**:
   - Many prefixes return F.zero() placeholder
   - Need full implementations matching Jolt

### What Needs to Be Done

The core issue is that stage5_prover.zig address rounds use bit-splitting instead of prefix-suffix:

**Current (wrong):**
```zig
for (0..T) |j| {
    const bit = getBit128(lookups_indices_lo[j], lookups_indices_hi[j], bit_index);
    const contrib = lookups_eq_evals[j].mul(lookups_ra_weights[j]).mul(lookups_combined_vals[j]);
    if (bit == 0) p0 = p0.add(contrib);
    else p1 = p1.add(contrib);
}
const eval_0 = p0;
const eval_2 = p1.add(p1).sub(p0);
```

**Required (Jolt approach):**
```zig
const read_checking = proverMsgReadChecking(F, round, &suffix_polys, &prefix_checkpoints, r_x);
const raf = proverMsgRaf(F, round, &identity_ps, &left_ps, &right_ps, gamma, gamma_sqr);
const eval_0 = read_checking[0].add(raf[0]);
const eval_2 = read_checking[1].add(raf[1]);

// After round:
suffix_polys.bindAll(challenge);
if (round % 2 == 1) {
    prefix_checkpoints.update(challenges[round-1], challenge, round, suffix_len);
}
// Phase transition every 16 rounds
```

### Remaining Implementation

1. **proverMsgRaf function** - Not yet implemented. Needs:
   - Identity polynomial prefix-suffix decomposition
   - Left/Right operand prefix-suffix decompositions
   - Computation: `γ*left + γ²*(right + identity)` via prefix-suffix

2. **Stage 5 integration** (stage5_prover.zig lines 1262-1362):
   - Replace bit-splitting with proverMsgReadChecking + proverMsgRaf
   - Add suffix_polys.bindAll(challenge) after each round
   - Add prefix checkpoint updates every 2 rounds
   - Add phase transitions every 16 rounds
   - Keep cycle rounds (128-135) as-is (already correct)

3. **Complete prefix MLE implementations** (prefixes.zig):
   - Many prefixes return F.zero() placeholder
   - Need full implementations matching Jolt

4. **Complete tableCombine for tables 15-41**:
   - Need to add proper formulas for all remaining tables

### Test Commands

```bash
# Build
zig build -Doptimize=ReleaseFast

# Generate proof
./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format -o /tmp/zolt_proof_dory.bin

# Verify with Jolt
cd /home/vivado/projects/jolt
cargo test --package jolt-core --features zolt-debug test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture
```

### Current Verification Status

- Stages 1-4: PASS
- Stage 5: FAIL (polynomial degree mismatch - need prefix-suffix for correct evaluations)

### Key Files

- `src/zkvm/spartan/stage5_prover.zig` - Main prover needing integration
- `src/zkvm/lookup_table/prefix_suffix_prover.zig` - proverMsgReadChecking, tableCombine
- `src/zkvm/lookup_table/prefixes.zig` - Prefix MLE implementations
- `src/zkvm/lookup_table/suffixes.zig` - Suffix MLE implementations
- `src/zkvm/lookup_table/identity_poly.zig` - Identity/Operand polynomials

### Commits This Session

- `d7d3f51` - feat: improve tableCombine with comprehensive table formulas
