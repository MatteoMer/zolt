# Zolt-Jolt Compatibility - Iteration 15 Status

## Summary

**Status**: Stage 1 passes, Stage 2 sumcheck is VALID but `expected_output_claim` differs from `output_claim`.

### Current Issue: expected_output_claim Mismatch

The Stage 2 batched sumcheck proof is mathematically valid:
- All 26 rounds pass (c0, c2, c3 coefficients match)
- All 26 challenges match
- Final `output_claim` matches

But `expected_output_claim ≠ output_claim`, meaning the polynomial evaluation differs from what the verifier expects.

| Value | Zolt | Jolt | Match |
|-------|------|------|-------|
| gamma_rwc | 31086377837778175205123147017089894504 | Same | ✓ |
| r_address challenge bytes | Identical | Same | ✓ |
| val_final_claim | 17708184114734783145538053377514369906907256976835332190588297692773985493533 | Same | ✓ |
| All 26 round polynomials | Byte-identical | Same | ✓ |
| output_claim | 13123490541784894264218864301865646689101148350774762798288422615780802764028 | Same | ✓ |
| expected_output_claim | N/A | 13551736511186635527939534124733318337862614044088180116386301103911465144413 | **Mismatch** |

### Completed ✓
1. All 712 Zolt internal tests pass
2. Stage 1 verification PASSES in Jolt
3. Fixed computeEqEvals to use big-endian ordering (matching Jolt)
4. gamma_rwc matches ✓
5. r_address challenge bytes match ✓ (previous hypothesis was wrong)
6. tau_high matches (21765436148454510711826271285135850355242651027995807066391230491147133393157) ✓
7. input_claims match ✓
8. factor_evals match ✓
9. fused_left and fused_right match ✓

### Instance Contributions (Stage 2)
- Instance 0 (ProductVirtual): claim=19366854058847837639268755478203018132153606224021885848136854669519817243621
- Instance 1 (RAF): claim=0
- Instance 2 (RWC): claim=0
- Instance 3 (Output): claim=16569652859076173421498202873716701161554115357020607472481625342580247939354
- Instance 4 (Instruction): claim=1281769312034380278185881668539414319935475418875412314915363312819906677592

### The Paradox

Sumcheck verification guarantees that `output_claim` equals the sum of polynomial evaluations at the challenge point. The opening claims should be those evaluations.

If `output_claim ≠ expected_output_claim`, one of these must be true:
1. The opening claims in the proof don't match what was used in prover computation
2. The verifier computes expected_output_claim differently than the prover
3. There's a normalization/endianness issue in how tau_low or r_tail_reversed are computed

### Focus: ProductVirtualRemainder (Instance 0)

The expected_output_claim for Instance 0 is computed as:
```
tau_high_bound_r0 * tau_bound_r_tail_reversed * fused_left * fused_right
```

Where:
- `tau_high_bound_r0` = Lagrange kernel of tau_high at r0
- `tau_bound_r_tail_reversed` = Eq MLE of (tau_low, r_tail_reversed)
- `fused_left` and `fused_right` = computed from factor_evals (MATCH ✓)

### Next Steps

1. **Compare tau_low values** between Zolt and Jolt
2. **Compare r_tail_reversed computation** - how is it derived from Stage 2 challenges?
3. **Compare tau_bound_r_tail_reversed** - the Eq polynomial evaluation
4. **Compare tau_high_bound_r0** - Lagrange kernel evaluation

### Hypothesis

The issue is likely in:
- How `r_tail_reversed` is computed from the sumcheck challenges
- Or how `tau_low` is set/used
- Or the Eq polynomial evaluation `eq(tau_low, r_tail_reversed)`

### Files to Investigate
- Zolt: `src/zkvm/proof_converter.zig` (Stage 2 generation)
- Jolt: `jolt-core/src/zkvm/spartan/product.rs` (ProductVirtualRemainder verifier)
- Jolt: `jolt-core/src/subprotocols/sumcheck.rs` (batched verification)

### Tests Status
- All 712 tests pass ✓
- Stage 1 passes ✓
- Stage 2 sumcheck rounds pass ✓
- Stage 2 expected_output_claim fails ✗
