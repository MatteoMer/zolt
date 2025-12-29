# Zolt-Jolt Compatibility TODO

## Current Status: Stage 1 Sumcheck Verification (Session 31)

### Issue Summary
The Stage 1 sumcheck output_claim still doesn't match expected_output_claim after fixing the sum-of-products structure.

```
output_claim:          11331697095435039208873616544229270298263565208265409364435501006937104790550
expected_output_claim: 12484965348201065871489189011985428966546791723664683385883331440930509110658
```

### Key Observation
The implied eq values show a systematic mismatch:
```
eq_prover:   18044952575580162354061690119060873415990550067227758637828342229490813546300
eq_verifier:  4684424059850432415474566298431286528152551688654307742749281091517521660901
ratio: 11242438927315634885064942370832368827610827657470043311206407684400448931447
```

The ratio is not a simple integer, suggesting the eq polynomial is being computed differently.

### Recent Fix Applied
- Fixed streaming round to use SUM-OF-PRODUCTS structure (not product-of-sums)
- For each cycle, we now:
  1. Compute Az/Bz for BOTH constraint groups
  2. Multiply pointwise (Az_g0*Bz_g0, slope_Az*slope_Bz)
  3. Weight by E_out[x_out] * E_in[x_in] and accumulate

### Verified Correct
- [x] tau_low extraction: tau_low = tau[0..tau.len-1]
- [x] Split eq initialization with m = tau_low.len / 2 = 5
- [x] E_out and E_in both have 32 entries (2^5)
- [x] head_in_bits = 5 for streaming round
- [x] Cycle index = (out_idx << head_in_bits) | in_idx
- [x] Lagrange kernel scaling factor applied
- [x] Sum-of-products structure for streaming round

### Remaining Investigation Needed

1. **Eq Polynomial Binding Order**
   - Zolt: bind() uses tau[current_index - 1], decrements current_index
   - Jolt: same pattern, but need to verify variable ordering matches

2. **E_out/E_in Table Construction**
   - Zolt builds tables with tau[k] for outer, tau[m+k] for inner
   - Need to verify big-endian indexing matches Jolt's EqPolynomial::evals

3. **Final Current Scalar Value**
   - After all binds, current_scalar should equal eq(tau_low, r_reversed)
   - The verifier computes: eq(tau_low, [r_n, ..., r_1, r_stream])
   - Need to trace if prover computes the same

4. **getWindowEqTables Return Values**
   - For streaming round: E_out[32], E_in[32], head_in_bits=5
   - Need to verify these match Jolt's E_out_in_for_window(1)

### Next Steps
1. [ ] Add debug output to trace current_scalar after each bind
2. [ ] Compare E_out/E_in values with Jolt
3. [ ] Check if there's an off-by-one in the tau indexing
4. [ ] Verify r_reversed construction matches Jolt

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
- [x] Split eq polynomial factorization (tau_low)
- [x] Lagrange kernel computation
- [x] Opening claims with MLE evaluation
- [x] Store tau_high separately
- [x] Constraint group combination matches Jolt
- [x] Sum-of-products structure for streaming round
- [x] All 656 Zolt tests pass
