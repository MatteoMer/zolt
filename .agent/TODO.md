# Zolt-Jolt Compatibility Implementation

## Status: IN PROGRESS - Stage 5 Opening Point Issue

## Verified Stages
- Stage 1: PASSED ✅
- Stage 2: PASSED ✅
- Stage 3: PASSED ✅
- Stage 4: PASSED ✅
- Stage 5: FAILING ❌ (opening_point is all zeros)
- Stage 6: Not tested yet
- Stage 7: Not tested yet

## Current Session Progress

### Fixed in This Session
1. **Fixed Stage 5 Toom-Cook encoding** - Changed from evaluation at x=3 to evaluation at infinity for the degree-3 sumcheck polynomials. This matches Jolt's `UniPoly::from_evals_toom` format.
   - Added `toomCookToCompressed()` function in `src/poly/mod.zig`
   - Updated `computeRegsValRoundPoly()` in stage5_prover.zig to compute eval_at_inf correctly
   - Fixed constant polynomial handling (p_inf = 0 for constant polynomials)

### Current Issue: Stage 5 Opening Point is All Zeros
The Stage 5 verifier gets `r_cycle` = all zeros from the opening accumulator.

**Root Cause Analysis:**
1. Stage 4 verifier's `cache_openings()` is called
2. It calls `normalize_opening_point(sumcheck_challenges)` to compute the opening point
3. The `sumcheck_challenges` being passed are all zeros!
4. This causes `opening_point` to be all zeros
5. When Stage 5 retrieves `RegistersVal` opening, it gets zeros for r_cycle

**Debug evidence:**
- `[STAGE4 VERIFIER cache_openings] opening_point.len = 15`
- All `opening_point[i] bytes = [00, 00, 00, 00, 00, 00, 00, 00]`
- But Stage 4's computed r_cycle values are non-zero (printed earlier in the flow)

**Hypothesis:**
The batched sumcheck verifier may not be correctly passing the challenges to `cache_openings()`. There could be an issue with:
1. How the batched sumcheck collects challenges across instances
2. How the challenges are passed to the Registers instance's cache_openings
3. A timing issue where cache_openings is called before challenges are populated

### Next Steps
1. Add debug output to the batched sumcheck verifier to see what challenges it passes
2. Check if RegistersReadWriteCheckingVerifier receives the correct challenges
3. Trace the challenge flow from transcript -> batched sumcheck -> instance cache_openings

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
- `src/zkvm/spartan/stage5_prover.zig` - Stage 5 batched sumcheck (FIXED Toom-Cook encoding)
- `src/poly/mod.zig` - Added toomCookToCompressed() function
- `jolt-core/src/zkvm/registers/read_write_checking.rs` - Stage 4 cache_openings (added debug)
- `jolt-core/src/zkvm/registers/val_evaluation.rs` - Stage 5 verifier (added debug)

## Commits Made
- Previous session: "fix(stage5): correct eq polynomial bit ordering for Jolt compatibility"
- Previous session: "fix(stage5): use compressed UniPoly transcript format"
- This session: Toom-Cook encoding fix (needs commit)

## Technical Notes

### Toom-Cook Encoding for Degree-3 Sumcheck
Jolt uses Toom-Cook style evaluation points: `[p(0), p(1), p(2), p(inf)]`
- `p(inf)` = leading coefficient (c3 for cubic polynomial)
- For product of linear polynomials: `f_inf = f_1 - f_0`

### Constant Polynomial in Batched Sumcheck
For a constant polynomial `p(x) = c`:
- `p(0) = p(1) = p(2) = c`
- `p(inf) = 0` (no x^3 term)

Previously Zolt was setting `p(3) = c` which is wrong for Toom-Cook.
