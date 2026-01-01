# Zolt-Jolt Compatibility TODO

## Current Status: Session 30 - January 1, 2026

**All 702 tests pass**

### Issue: Stage 1 sumcheck output_claim ≠ expected_output_claim

The infrastructure for the linear phase is now in place, but the indexing needs work.

### Completed Work (This Session)

1. [x] **DensePolynomial.bindLow()** - Matches Jolt's bound_poly_var_bot()
2. [x] **Az/Bz polynomial storage** - Added to StreamingOuterProver
3. [x] **Materialization at switchover** - Creates polynomials with correct size (2^linear_rounds)
4. [x] **Binding infrastructure** - Calls bindLow() each linear round
5. [x] **Linear phase code path** - Separate from streaming phase

### Remaining Issue

The linear phase computation still produces wrong values. The indexing into the bound polynomials doesn't match Jolt's approach.

**Current behavior:**
- Materialization creates polynomial of size 2^6 = 64 for 6 linear rounds
- Each linear round binds one variable
- But the VALUES being read from the polynomial don't match what the verifier expects

**Jolt's approach (from compute_evaluation_grid_from_polynomials_parallel):**
```rust
// Jolt uses pre-bound Az/Bz directly indexed by position
for j in 0..grid_size {
    let full_idx = grid_size * i + j;
    local_ans[idx] += az.Z[full_idx] * bz.Z[full_idx] * E_out[i];
}
```

**What needs fixing:**
1. The materialization should create Az/Bz with the same structure as Jolt
2. The linear phase indexing should directly read from az_poly[idx], bz_poly[idx]
3. The E_out/E_in multiplication should happen during computation, not materialization

### Debug Values (Latest Run)
```
output_claim:          18149181199645709635565994144274301613989920934825717026812937381996718340431
expected_output_claim: 9784440804643023978376654613918487285551699375196948804144755605390806131527
(Still unchanged after linear phase changes - indexing issue)
```

## Pending Tasks
- [ ] Fix linear phase indexing to match Jolt's compute_evaluation_grid_from_polynomials_parallel
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
- [x] Linear phase infrastructure (az_poly, bz_poly, binding)
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
