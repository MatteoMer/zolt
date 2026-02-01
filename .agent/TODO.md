# Zolt-Jolt Compatibility Implementation

## Status: IN PROGRESS - Stage 5 Prefix-Suffix Integration

## Session 98 Progress

### Task: Replace bit-splitting with prefix-suffix decomposition in Stage 5

### Completed This Session

1. **Analyzed Jolt's prefix-suffix decomposition** - Deep understanding of:
   - How `prover_msg_read_checking` computes prefix MLE at c=0 and c=2
   - How `table.combine()` combines prefix/suffix evaluations
   - How RAF contribution is computed via identity/operand decompositions

2. **Improved tableCombine function** (`prefix_suffix_prover.zig`):
   - Added proper formulas for tables 0-14 matching Jolt exactly
   - Implemented signed/unsigned comparison formulas with MSB prefix handling
   - Tables 15-41 have placeholder (need to be completed)

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
