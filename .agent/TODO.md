# Zolt-Jolt Compatibility Implementation

## Status: Session 27 - DEEP INVESTIGATION NEEDED

## Current Issue

**The Challenge values after normalize_opening_point don't match the original sumcheck_challenges.**

### Debug Output Analysis

InstructionClaimReduction cache_openings receives 8 challenges. After normalize_opening_point (which reverses):

```
sumcheck_challenges[0] as F = [e5, 14, 65, ea, ..., 04, 5f, 60, 72, ...]
sumcheck_challenges[7] as F = [0d, 02, 35, 5f, ..., e4, 6f, ff, 58, ...]

After reversal:
opening_point.r[0] = [..., 0d, 8d, 89, b0, c0, ef, 00, b0, 84, a4, 8a, 1b, 0b, 14, 34, 07]
opening_point.r[7] = [..., 41, 9e, 1e, ff, f9, d6, df, 8e, 6d, 56, 26, 2e, ba, 1a, c1, 03]
```

Expected: r[0] = challenges[7] = `[..., e4, 6f, ff, 58, ...]`
Actual: r[0] = `[..., 0d, 8d, 89, b0, ...]`

These don't match!

### Hypothesis

The `challenges.to_vec()` in normalize_opening_point might be doing more than a simple copy. Or there's some state mutation happening between the two debug prints.

### Investigation Needed

1. Check if sumcheck_challenges is being modified between debug print and normalize_opening_point
2. Check if OpeningPoint::new does any transformation
3. Check if match_endianness does more than reverse
4. Verify the Challenge serialization format

### Files Modified

- jolt-core/src/zkvm/claim_reductions/instruction_lookups.rs (debug output)
- jolt-core/src/zkvm/instruction_lookups/read_raf_checking.rs (debug output)

### Next Steps

1. Add more debug between every step in normalize_opening_point
2. Print the actual memory representation of challenges before and after
3. Check if there's a transcript mutation happening concurrently

## Files to Check

- Jolt: jolt-core/src/poly/opening_proof.rs (OpeningPoint implementation)
- Jolt: jolt-core/src/zkvm/claim_reductions/instruction_lookups.rs

## Test Commands
```bash
cd jolt && cargo test -p jolt-core --features zolt-debug --lib test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture
```
