# Zolt-Jolt Compatibility: Stage 4 Investigation

## Completed
- Fixed `merge()` function in GruenSplitEqPolynomial to match Jolt's implementation
- eq_eval now matches between Zolt prover and Jolt verifier: `d8 d0 62 f3 fe 93 6a 58 ...`
- Opening claims (val_claim, rs1_ra_claim, etc.) match between Zolt and Jolt

## Current Status: Stage 4 Verification Still Fails

The sumcheck is internally consistent (p(0)+p(1)=claim every round), but the final output doesn't match expected:
- Sumcheck output_claim: `73 57 fa 78 7c 9c da 6d ...`
- Expected claim: `b9 89 d9 c7 c5 a4 3c a4 ...`

### Analysis
1. **eq_eval matches** - The merge() fix works correctly
2. **combined values match** - Both compute `27 56 d8 bd ...`
3. **Individual claims match** - val_claim, rs1_ra_claim, etc. are identical
4. **expected_output matches** - Instance 0 expected = `4c 51 88 8e ...`

### Key Observation
The sumcheck polynomial evaluations after binding don't equal `eq_eval * combined`:
- Prover computes: `merged_eq[0] * (ra[0]*val[0] + wa[0]*(val[0]+inc[0]))`
- But this doesn't equal the expected `eq_eval * combined_from_claims`

### Possible Causes
1. **Polynomial binding mismatch**: The polynomials (val, ra, wa, inc) may not be bound correctly to produce the correct final evaluations
2. **Index/ordering issue**: The polynomial evaluations at index [0] after binding may not correspond to the same point as the claims
3. **Phase transition bug**: Something in Phase 1→2→3 transitions may be corrupting polynomial state

### Debug Values Needed
Compare these values:
- `val_poly[0]` after binding vs `val_claim` from getFinalClaims()
- `rs1_ra_poly[0]` vs `rs1_ra_claim`
- `rs2_ra_poly[0]` vs `rs2_ra_claim`
- `rd_wa_poly[0]` vs `rd_wa_claim`
- `inc_poly[0]` vs `inc_claim`

If these differ, the binding logic is wrong.
If these match but the sumcheck output differs from expected, there's a deeper issue with how the round polynomials are computed vs how expected claims are evaluated.

## Next Steps
1. Add debug to compare polynomial values after binding with stored claims
2. Trace through one complete example to verify polynomial binding
3. Compare Zolt's Phase 1/2/3 binding with Jolt's RegistersReadWriteCheckingProver

## Files Modified
- `/home/vivado/projects/zolt/src/zkvm/spartan/gruen_eq.zig` - Fixed merge()
- `/home/vivado/projects/zolt/src/poly/mod.zig` - Added evalsSliceWithScaling()
