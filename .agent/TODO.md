# Zolt-Jolt Compatibility TODO

## Current Status: Session 30 - January 1, 2026

**All 702 tests pass**

### ROOT CAUSE IDENTIFIED

**Issue:** Stage 1 sumcheck output_claim ≠ expected_output_claim

The core problem is that Zolt's linear phase implementation doesn't match Jolt's architecture.

**Jolt's Approach:**
1. Streaming phase: Uses r_grid to weight trace contributions
2. Linear phase: Materializes Az/Bz polynomials, then BINDS them each round with `bound_poly_var_bot()`

**Zolt's Approach (Incorrect):**
1. All rounds: Tries to use r_grid, recomputes from trace each round
2. Never binds Az/Bz polynomials

### Active Work

**Fix Required:** Implement Jolt-style linear phase

1. [x] **Add DensePolynomial.bindLow()** - Polynomial binding for linear sumcheck ✅ DONE
2. [ ] **Add Az/Bz storage to StreamingOuterProver** - Store bound polynomials
3. [ ] **Materialize polynomials at linear phase start** - Compute Az/Bz for all cycles at switchover
4. [ ] **Modify linear round computation** - Use bound polynomials instead of trace
5. [ ] **Bind Az/Bz each linear round** - Call bindLow() on each challenge
6. [ ] **Test with Jolt verifier**

### Implementation Notes

**Linear Phase Flow (from Jolt):**
```rust
// At linear phase start (switchover):
let (az, bz) = fused_materialise_polynomials_general_with_multiquadratic(shared, window_size);

// Each linear round:
fn ingest_challenge(&mut self, shared: &mut Self::Shared, r_j: F::Challenge, _round: usize) {
    shared.split_eq_poly.bind(r_j);
    // CRITICAL: Bind Az and Bz polynomials
    self.az.bind_parallel(r_j, BindingOrder::LowToHigh);
    self.bz.bind_parallel(r_j, BindingOrder::LowToHigh);
}

// For computing round poly, use bound az/bz values, NOT trace:
let az_value = self.az.Z[idx];  // Use bound polynomial
let bz_value = self.bz.Z[idx];  // Not trace lookup!
```

### Debug Values (Latest Run)
```
output_claim:          18149181199645709635565994144274301613989920934825717026812937381996718340431
expected_output_claim: 9784440804643023978376654613918487285551699375196948804144755605390806131527

Implicit prover inner_sum = output / eq_factor = 4998170542285690794587036959401462214732012225407648253956536381087950469724
Expected inner_sum_prod = 18008138052294660670516952860372938358542359888052020571951954839855384564920
Ratio = ~3.6x (not a simple factor - confirms structural mismatch)
```

### Verified Components
- ✅ DensePolynomial.bindLow() matches Jolt's bound_poly_var_bot()
- ✅ eq factor: `prover_eq_factor == verifier_eq_factor`
- ✅ Individual sumcheck rounds pass (p(0) + p(1) = claim)
- ✅ R1CS constraint and input ordering matches Jolt
- ✅ Az and Bz MLE computations match
- ✅ Streaming round logic is correct
- ✅ r_grid ExpandingTable matches Jolt

## Pending Tasks
- [ ] Implement linear phase with bound polynomials (BLOCKING)
- [ ] Complete remaining stages (2-7) proof generation
- [ ] Create end-to-end verification test with Jolt verifier

## Verified Correct
- [x] Blake2b transcript implementation
- [x] Field serialization (Arkworks format)
- [x] UniSkip polynomial generation
- [x] R1CS constraint definitions (19 constraints, 2 groups)
- [x] Split eq polynomial factorization (E_out/E_in tables)
- [x] Lagrange kernel L(tau_high, r0)
- [x] MLE evaluation for opening claims
- [x] Gruen cubic polynomial formula
- [x] r_cycle computation (big-endian, excluding r_stream)
- [x] eq polynomial factor matches verifier
- [x] Streaming round sum-of-products structure
- [x] Transcript flow matching Jolt
- [x] ExpandingTable (r_grid) matches Jolt
- [x] DensePolynomial.bindLow() implementation
- [x] All 702 Zolt tests pass

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
