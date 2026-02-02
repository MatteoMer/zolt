# Zolt-Jolt Compatibility Implementation

## Status: IN PROGRESS - ra_claims Values Mismatch

## Session 128 Summary

### Key Findings

1. **Polynomial coefficients MATCH** - The batched sumcheck polynomial for round 0 matches between Zolt prover and Jolt verifier

2. **Round 128 challenge MATCHES** - Verified that `f79e052f4a48e5103da274f0c5d379ef` matches between both

3. **Operand evaluations MATCH** - left_operand_eval, right_operand_eval, identity_eval all match:
   - `left_op_eval = b2450f205a45b0cf95a97f152808af6f` (Zolt BE)
   - `left_operand_eval = [6f, af, 08, 28, ...]` (Jolt LE) → matches when converted

4. **ra_claims DO NOT MATCH**:
   - Zolt: `ra_chunks[0] = 90fa96e636b607e1e46f2c8bff8e00be`
   - Jolt: `ra_claims[0] = [a5, 5e, c7, 72, 66, 8e, 13, 27, ...]` → `119b26350ef30d2127138e6672c75ea5` in BE

### The Core Issue

The ra_claims are the final sumcheck claims for the virtual ra polynomials after all 8 cycle binding rounds. Even though:
- Initial materialization appears correct
- Binding challenges appear correct

The final values don't match.

### Possible Root Causes

1. **Binding logic issue during cycle rounds** - The polynomial binding might differ
2. **Challenge order/endianness** - How challenges are applied to polynomials
3. **Off-by-one in polynomial access** - Accessing wrong indices after binding

### Debug Information

**Zolt ra_chunk_weights after materialization (cycle 0, chunk 0):**
- v0_p0 = `408e25165ab4e1eaa3ad60d7db172356`
- v0_p1 = `45df4a65017fd3a782c45b7dc305c1cb`
- product = `90fa96e636b607e1e46f2c8bff8e00be` (initial ra_chunk[0][0])

After 8 binding rounds, all cycles had the same value (constant polynomial) so:
- final ra_chunks[0] = `90fa96e636b607e1e46f2c8bff8e00be` (unchanged)

But Jolt expects: `119b26350ef30d2127138e6672c75ea5`

### Next Steps

1. **Add debug to Jolt** to print ra_polys values immediately after init_log_t_rounds() materialization
2. **Compare expanding table values** between Zolt and Jolt at round 128
3. **Verify challenge application order** in cycle round binding

### Files Modified This Session

- `src/zkvm/spartan/stage5_prover.zig`:
  - Removed bit-reversal in ra_chunk_weights materialization
  - Added detailed debug output for ra computation
  - Added r_address_prime debug prints

### Test Commands

```bash
# Build Zolt
zig build -Doptimize=ReleaseFast

# Generate proof with debug
./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format --export-preprocessing logs/zolt_preprocessing.bin -o logs/zolt_proof_dory.bin

# Copy and verify
cp logs/zolt_*.bin /tmp/
cd /home/vivado/projects/jolt
cargo test --package jolt-core --features zolt-debug test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture
```
