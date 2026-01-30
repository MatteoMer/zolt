# Zolt-Jolt Compatibility Implementation

## Status: IN PROGRESS - Stage 5 Output Claim Mismatch

## Verified Stages
- Stage 1: PASSED ✅
- Stage 2: PASSED ✅
- Stage 3: PASSED ✅
- Stage 4: PASSED ✅
- Stage 5: FAILING ❌ (output_claim != expected_output_claim)
- Stage 6: Not tested yet
- Stage 7: Not tested yet

## Current Session Progress

### Fixed in This Session
1. **Fixed Stage 5 transcript format** - Changed from `UncompressedUniPoly_begin/end` (4 coefficients) to `UniPoly_begin/end` (3 compressed coefficients) to match Jolt's BatchedSumcheck format.

### Current Issue: Stage 5 Output Claim Mismatch
The sumcheck internally passes (computed_sum = regs_val_input), but the final output_claim doesn't match expected_output_claim:
- `output_claim = 14207773099973851432316380405455832148939247972351308198786873286672527593212`
- `expected_output_claim = 18244124058491777017643072331864832508434028736769776383365323382525894304545`

The mismatch comes from Instance 0 (RegistersValEvaluation):
- Zolt's `inc_claim` and `wa_claim` values differ from what Jolt expects
- The verifier computes `expected_output_claim = inc_claim * wa_claim * LT(r_normalized, r_cycle)`

### Investigation Notes
1. Binding order verified: Both Zolt and Jolt use `LowToHigh` binding (`Z[i] = (1-r)*Z[2i] + r*Z[2i+1]`)
2. Transcript format fixed: Now using compressed UniPoly format
3. Gamma values match at Stage 4 start
4. Stage 4 passes verification

### Next Steps
1. Compare inc_claim and wa_claim values between Zolt prover and what Jolt verifier reads from proof
2. Check if the polynomial binding is happening in the correct rounds (offset calculation in batched sumcheck)
3. Verify that the r_cycle used for polynomial construction matches r_cycle from Stage 4
4. Check the normalize_opening_point logic matches between Zolt and Jolt

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
- `src/zkvm/spartan/stage5_prover.zig` - Stage 5 batched sumcheck (FIXED transcript format)
- `src/zkvm/proof_converter.zig:2663-2686` - r_cycle/r_address extraction
- `jolt-core/src/zkvm/registers/val_evaluation.rs` - Jolt's verifier reference
- `jolt-core/src/poly/opening_proof.rs` - OpeningAccumulator implementation

## Commits Made
- Previous session: "fix(stage5): correct eq polynomial bit ordering for Jolt compatibility"
- This session: Fixed Stage 5 transcript format (UncompressedUniPoly -> UniPoly)
