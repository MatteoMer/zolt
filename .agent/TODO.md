# Zolt-Jolt Compatibility TODO

## Current Status: Session 29 - December 31, 2024

**All 656 tests pass**

### Active Investigation

**Issue:** Stage 1 sumcheck output_claim ≠ expected_output_claim

**Debug Values (Latest Run):**
```
output_claim:          18149181199645709635565994144274301613989920934825717026812937381996718340431
expected_output_claim: 9784440804643023978376654613918487285551699375196948804144755605390806131527
tau_high_bound_r0:     10811398959691251178446374398729567517345474364208367822721217673367518413943
tau_bound_r_tail:      19441068294701806481650633278345574000268469546428153638327506282094641388680
inner_sum_prod:        18008138052294660670516952860372938358542359888052020571951954839855384564920
```

**Verified:**
- ✅ eq factor: `prover_eq_factor == verifier_eq_factor` (cross-verification test passes)
- ✅ Individual sumcheck rounds pass
- ✅ R1CS constraint and input ordering matches Jolt

**Suspected Issue:**
The Az*Bz computation in cycle rounds may be using the wrong approach:
- After streaming round binds r_stream, should cycle rounds use combined Az/Bz?
- Current code uses `selector = full_idx & 1` to pick constraint group

### Next Steps
1. Compare cycle round Az/Bz computation with Jolt's approach
2. Check if `computeCycleAzBzForGroup` should be `computeCycleAzBzProductCombined` in cycle rounds

## Pending Tasks
- [ ] Debug inner_sum_prod (Az*Bz) computation
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
- [x] All 656 Zolt tests pass

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
