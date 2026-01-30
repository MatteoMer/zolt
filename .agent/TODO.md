# Zolt-Jolt Compatibility Implementation

## Status: IN PROGRESS - Stage 5 Debugging

## Verified Stages
- Stage 1: PASSED ✅
- Stage 2: PASSED ✅
- Stage 3: PASSED ✅
- Stage 4: PASSED ✅
- Stage 5: PARTIAL (sum check matches, final claim mismatch) 🔄
- Stage 6: Not tested yet
- Stage 7: Not tested yet

## Current Session Progress

### Fixed Issue
- **computeEqAtIndex bit ordering** - Was using LSB-first extraction but Jolt uses MSB-first (big-endian) indexing
  - Old: `ki = (k >> i) & 1` with `ri = r[n-1-i]`
  - New: `bj = (k >> (n-1-j)) & 1` with `rj = r[j]`

### Current Status
Stage 5 RegistersValEvaluation sum check now passes:
```
[STAGE5] Sum check: computed_sum = { 44, 166, 232, 254, 202, 91, 155, 217, ... }
[STAGE5] Sum check: regs_val_input = { 44, 166, 232, 254, 202, 91, 155, 217, ... }
[STAGE5] Sum check: match = true
```

But the final sumcheck output claim doesn't match verifier's expectation:
```
output_claim:          9634238360255972074564063771795547071448922862312878542074078713134022512917
expected_output_claim: 12526348194846811955338446430006011584675142908683096698746245038486546873528
```

### Analysis
The verifier computes `expected_output_claim = inc_claim * wa_claim * LT(r_normalized, r_cycle)`

Where:
- `inc_claim` and `wa_claim` are retrieved from the proof (stored by prover's cache_openings)
- `LT(r_normalized, r_cycle)` is computed independently by verifier
- `r_normalized` = reversed sumcheck challenges (LITTLE_ENDIAN → BIG_ENDIAN)

The issue is likely in how the LT polynomial evaluates after binding. After binding with challenges
in LowToHigh order, `lt[0]` should equal `LT(challenges_reversed, r_cycle)`.

### Next Steps
1. Add debug output to verify:
   - What is `lt[0]` after all bindings?
   - What does verifier compute for `LT(r_normalized, r_cycle)`?
   - Are `inc_claim` and `wa_claim` correct?

2. Verify the binding order matches Jolt exactly:
   - Jolt binds variables LowToHigh (bit 0 = LSB first)
   - After binding, the evaluation point is in reversed (BIG_ENDIAN) order

3. Check if there's an off-by-one error in the batched sumcheck round handling

## Test Commands

```bash
# Generate proof
cd /home/vivado/projects/zolt
./zig-out/bin/zolt prove examples/fibonacci.elf \
  --jolt-format \
  -o /tmp/zolt_proof_dory.bin

# Verify with Jolt
cd /home/vivado/projects/zolt/jolt
cargo test --package jolt-core test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture
```

## Key Files
- `src/zkvm/spartan/stage5_prover.zig` - Stage 5 batched sumcheck
- `src/zkvm/proof_converter.zig:2663-2686` - r_cycle/r_address extraction
- `jolt-core/src/zkvm/registers/val_evaluation.rs` - Jolt's prover reference
- `jolt-core/src/poly/lt_poly.rs` - LtPolynomial implementation
