# Zolt-Jolt Compatibility TODO

## Current Status: Session 28 - December 31, 2024

**All 657 tests pass** (656 original + 1 new cross-verification test)

### Session 28 Accomplishments

#### ✅ Verified Key Components
1. **r_cycle computation**: `challenges[1..]` reversed to big-endian matches Jolt's `normalize_opening_point`
2. **eq polynomial**: Prover's `current_scalar` equals `L(tau_high, r0) * eq(tau_low, r_tail_reversed)`
3. **Az/Bz blending**: `final = g0 + r_stream * (g1 - g0)` matches Jolt
4. **Cross-verification test passes**: `prover_eq_factor == verifier_eq_factor`

#### Key Formula (from Jolt's expected_output_claim)
```
expected = tau_high_bound_r0 * tau_bound_r_tail_reversed * inner_sum_prod

Where:
- tau_high_bound_r0 = L(tau_high, r0) = Lagrange kernel at UniSkip challenge
- tau_bound_r_tail_reversed = eq(tau_low, [r_n, ..., r_1, r_stream])
- inner_sum_prod = Az_final * Bz_final (from R1CS input MLE evaluations)
- r_cycle = challenges[1..] reversed (excludes r_stream, used for R1CS inputs)

Important: r_tail_reversed includes ALL sumcheck challenges (including r_stream)
           r_cycle for R1CS inputs excludes r_stream
```

### Remaining Investigation
The eq factor (tau_high_bound_r0 * tau_bound_r_tail_reversed) matches between prover and verifier.
The remaining question is whether inner_sum_prod matches:
- Prover computes: `Σ_cycles eq(tau_low, cycle) * Az(cycle) * Bz(cycle)` via streaming sumcheck
- Verifier computes: `Az(r_stream, r0, z(r_cycle)) * Bz(r_stream, r0, z(r_cycle))` using opening claims

### Next Steps
1. Debug inner_sum_prod computation (Az*Bz from opening claims vs prover)
2. Create end-to-end verification test with Jolt verifier
3. Complete remaining stages (2-7) proof generation

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
- [x] eq polynomial factor matches verifier (new cross-verification test)

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
- [x] Transcript flow matching Jolt
- [x] All 657 Zolt tests pass
- [x] Cross-verification test for eq factors (new)
