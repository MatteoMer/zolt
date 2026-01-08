# Zolt-Jolt Compatibility - Deep Debugging Session

## Current Status: Instance 3 (OutputSumcheck) Bug Investigation

### CRITICAL BUG IDENTIFIED

Instance 3 (OutputSumcheck/RamOutputCheck) is producing a non-zero final claim even though it should be zero:

- Input claim = 0 (correct, Jolt expects 0)
- Expected final claim = 0 (Jolt)
- Actual final claim = non-zero (Zolt)

This causes the expected_output_claim mismatch because Instance 3's contribution (coeff * claim) becomes non-zero when it should be 0.

### Understanding the Protocol

OutputSumcheck proves a **zero-check**:
```
Σ_k eq(r_address, k) * io_mask(k) * (Val_final(k) - Val_io(k)) = 0
```

- `input_claim = 0` always (this is a zero-check)
- `final_claim = eq(r_address, r') * io_mask(r') * (Val_final(r') - Val_io(r'))`
  - where r' = sumcheck challenges (normalized to BIG_ENDIAN)

### Jolt's claim update mechanism

The verifier uses `eval_from_hint`:
```rust
c1 = hint - 2*c0 - c2 - c3
P(r) = c0 + c1*r + c2*r^2 + c3*r^3
```

where `hint = s(0) + s(1) = previous_claim`.

### Potential Root Causes

1. **OutputSumcheckProver polynomial not identically zero**
   - If io_mask ≠ 0 somewhere AND Val_final ≠ Val_io there, the polynomial is non-zero
   - The `v = Val_final - Val_io` must be 0 in the entire IO region

2. **Claim update using wrong hint value**
   - The prover's `updateClaim` should use the previous claim as hint
   - Currently it uses Lagrange to recompute c2, c3 from evaluations, which SHOULD be equivalent

3. **Endianness mismatch in challenge handling**
   - Jolt normalizes opening points: LITTLE_ENDIAN → BIG_ENDIAN
   - Zolt must do the same

### Next Steps

1. Add debug output to OutputSumcheckProver to see:
   - Round-by-round s(0), s(1), s(2), s(3) values
   - Whether they're all zero (as expected)
   - If not, which index k has non-zero contribution

2. Verify that Val_final[k] = Val_io[k] for all k in IO region

3. Check if the io_mask is correctly computed

### Files:
- Zolt OutputSumcheck: src/zkvm/ram/output_check.zig
- Zolt proof converter: src/zkvm/proof_converter.zig (Stage 2 batched sumcheck)
- Jolt OutputSumcheck: jolt-core/src/zkvm/ram/output_check.rs

---

## Previous Investigation Summary

### What MATCHES between Zolt and Jolt:
1. Stage 1 sumcheck proof - all rounds pass ✓
2. Stage 2 initial batched_claim ✓
3. Stage 2 batching_coeffs (all 5) ✓
4. Stage 2 input_claims for all 5 instances ✓
5. Stage 2 tau_high ✓
6. Stage 2 r0 ✓
7. ALL 26 Stage 2 round coefficients (c0, c2, c3) ✓
8. ALL 26 Stage 2 challenges ✓
9. Final output_claim ✓
10. All factor claims match ✓
11. Instance 4 claims match ✓
12. fused_left and fused_right ✓

### The Problem:
- output_claim: 6490144552088470893406121612867210580460735058165315075507596046977766530265
- expected_output_claim: 15485190143933819853706813441242742544529637182177746571977160761342770740673

### Expected contribution breakdown (from Jolt):
- Instance 0 (ProductVirtual): contribution ≠ 0
- Instance 1 (RAF): contribution = 0
- Instance 2 (RWC): contribution = 0
- Instance 3 (Output): contribution = 0 (expected)
- Instance 4 (Instruction): contribution ≠ 0
