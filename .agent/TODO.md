# Zolt-Jolt Compatibility TODO

## Current Status: Stage 1 Sumcheck Verification (Session 28)

### Latest Findings
- Fixed tau split issue: now passing full tau (len 12) instead of tau_low (len 11)
- This caused output_claim to change: old ~11.3e75, new ~13.3e75
- But verification still fails (output_claim ≠ expected_output_claim)
- Ratio is now ~0.75 instead of ~0.91

```
output_claim:          13253494829145889902934155105755742414822642232978678139117322728009984608729
expected_output_claim: 17615172925949730609183894067324551214200634089653146239852636572846820428381
```

### What Was Fixed
- [x] Pass full tau to split_eq (m = tau.len/2 = 6 for len=12)
- [x] E_out now has 64 entries (2^6), E_in has 32 entries (2^5) - matches Jolt

### Verified Working
- [x] All 656 Zolt tests pass
- [x] Proof deserialization (all 48 opening claims parsed)
- [x] Jolt preprocessing loads correctly
- [x] Verifier instance creation succeeds
- [x] 128-bit challenges in Montgomery form
- [x] EqPolynomial evaluation
- [x] r_cycle computation (challenges[1..] reversed)
- [x] UniSkip verification passes
- [x] Individual round equations (p(0)+p(1)=claim)
- [x] Tau split now matches Jolt (m = tau.len/2)

### Remaining Issues
The sumcheck output_claim doesn't match expected_output_claim.

Expected formula (Jolt outer.rs lines 421-452):
```
expected = tau_high_bound_r0 * tau_bound_r_tail_reversed * inner_sum_prod

where:
- tau_high_bound_r0 = L(tau_high, r0) - Lagrange kernel
- tau_bound_r_tail_reversed = eq(tau_low, reversed(sumcheck_challenges))
- inner_sum_prod = Az(rx_constr) * Bz(rx_constr)
- rx_constr = [r_stream, r0]
```

Possible remaining issues:
1. **R1CS witness values** - Zolt generates witnesses from its emulator, may differ from Jolt
2. **Inner sum product computation** - The evaluate_inner_sum_product_at_point formula
3. **Constraint group handling** - The interleaved group structure (full_idx >> 1, & 1)
4. **Binding order** - How challenges are bound in the streaming prover

### Next Investigation Steps
1. Compare Zolt's Az/Bz values at a specific cycle with Jolt's
2. Debug the inner_sum_prod computation
3. Check if the constraint evaluation matches Jolt's R1CSEval

## Test Commands
```bash
# Zolt tests
zig build test --summary all

# Generate proof
zig build && ./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format -o /tmp/zolt_proof_dory.bin

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
- [x] Challenge Montgomery form conversion
- [x] Tau split fix (pass full tau, not tau_low)
