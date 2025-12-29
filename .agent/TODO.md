# Zolt-Jolt Compatibility TODO

## Current Status: Stage 1 Sumcheck Verification (Session 31 continued)

### Issue Summary
The Stage 1 sumcheck output_claim doesn't match expected_output_claim.

```
output_claim:          11331697095435039208873616544229270298263565208265409364435501006937104790550
expected_output_claim: 12484965348201065871489189011985428966546791723664683385883331440930509110658
```

### Verified Correct
- [x] Blake2b transcript (byte-for-byte match)
- [x] UniSkip polynomial coefficients
- [x] Compressed poly format [c0, c2, c3]
- [x] interpolateDegree3 and evalsToCompressed
- [x] computeCubicRoundPoly formula matches Jolt's gruen_poly_deg_3
- [x] Streaming round uses SUM-OF-PRODUCTS structure
- [x] Cycle rounds use selector = full_idx & 1
- [x] r_grid update logic matches Jolt's HalfSplitSchedule
- [x] E_out/E_in table construction (big-endian indexing)
- [x] split_eq.bind() formula matches Jolt
- [x] All 656 Zolt tests pass

### Key Insight
The eq polynomial and Az*Bz computation SHOULD be producing the same result, but the implied eq values show a systematic mismatch:
```
eq_prover:   18044952575580162354061690119060873415990550067227758637828342229490813546300
eq_verifier:  4684424059850432415474566298431286528152551688654307742749281091517521660901
```

### Investigation Paths

1. **Transcript Challenge Derivation**
   - Verify Zolt and Jolt produce the same challenges for the same input
   - Check if there's a difference in how input_claim is appended

2. **full_idx Bit Layout**
   - Zolt: base_idx | x_val_shifted | r_idx where step_idx = full_idx >> 1
   - Verify this matches Jolt's indexing exactly

3. **r_grid Weight Structure**
   - r_grid[r_idx] contains weights for (group, cycle_bits)
   - Bit 0 of r_idx corresponds to r_stream (group factor)
   - Verify the weight accumulation is correct

4. **Az*Bz Product Computation**
   - Jolt uses separate bz accumulators per group, then sums
   - Zolt accumulates bz from selected group into single grid
   - Mathematically equivalent, but verify

### Files Under Investigation
- src/zkvm/spartan/streaming_outer.zig (computeRemainingRoundPolyMultiquadratic)
- src/poly/split_eq.zig (bind, computeCubicRoundPoly)
- src/zkvm/proof_converter.zig (Stage 1 proof generation)

### Next Steps
1. Add debug output to compare challenges at each round
2. Compare t_zero and t_infinity values
3. Verify the uni_skip_claim is computed correctly
4. Check if the input_claim is appended to transcript correctly before getting batching_coeffs

## Test Commands
```bash
# Zolt tests
zig build test --summary all

# Generate proof
zig build -Doptimize=ReleaseFast && ./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format -o /tmp/zolt_proof_dory.bin

# Jolt verification test
cd /Users/matteo/projects/jolt
cargo test --package jolt-core test_verify_zolt_proof -- --ignored --nocapture
```

## Completed Milestones
- [x] Blake2b transcript implementation
- [x] Field serialization (Arkworks format)
- [x] UniSkip polynomial generation
- [x] Stage 1 remaining rounds sumcheck
- [x] R1CS constraint definitions
- [x] Split eq polynomial factorization
- [x] Lagrange kernel computation
- [x] Opening claims with MLE evaluation
- [x] Streaming round sum-of-products structure
- [x] Cycle rounds multiquadratic method
- [x] All 656 Zolt tests pass
