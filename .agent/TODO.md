# Zolt-Jolt Compatibility TODO

## Current Status: Stage 1 Output Claim Mismatch

### Session 19 Summary

**Fixed:**
- Changed EqPolynomial.evals() to use big-endian indexing (matching Jolt)

**Current Values:**
- output_claim (from sumcheck): 21176670064311113248327121399637823341669491654917035040693110982193526510099
- expected (from R1CS): 15830891598945306629010829910964994017594280764528826029442912827815044293203
- Ratio: ~1.338 (changed from ~1.2 after eq fix)

**Key Observations:**
1. All 11 sumcheck rounds pass (p(0) + p(1) = claim verified) ✓
2. UniSkip domain sum passes ✓
3. The mismatch is in the expected output claim only

**Verified Components:**
- EqPolynomial.evals() uses big-endian indexing ✓
- split_eq E tables use big-endian indexing ✓
- computeCubicRoundPoly matches Jolt's gruen_poly_deg_3 ✓
- r_cycle = challenges[1..] reversed (10 elements) ✓
- tau_bound_r_tail uses all 11 challenges reversed ✓

**Remaining Investigation:**
1. Check if t_zero and t_infinity values match Jolt's prover
2. Verify constraint group indices are correct
3. Check if the cycle witness extraction matches Jolt's R1CSCycleInputs

### Key Formula (from Jolt verifier):
```rust
expected = tau_high_bound_r0 * tau_bound_r_tail * inner_sum_prod

where:
- tau_high_bound_r0 = L(tau_high, r0)
- tau_bound_r_tail = eq(tau_low, [r_10, ..., r_1, r_stream])
- inner_sum_prod = az_final * bz_final
  - az_final = az_g0 + r_stream * (az_g1 - az_g0)
  - bz_final = bz_g0 + r_stream * (bz_g1 - bz_g0)
```

## Completed

### Phase 1-5: Core Infrastructure
1. Transcript Compatibility - Blake2b
2. Proof Structure - 7-stage
3. Serialization - Arkworks format
4. Commitment - Dory with Jolt SRS
5. Verifier Preprocessing Export

### Stage 1 Fixes (Sessions 11-19)
6-33. [Previous fixes...]
34. Big-endian EqPolynomial.evals()

## Test Commands

```bash
# Zolt tests
zig build test --summary all

# Generate proof
zig build -Doptimize=ReleaseFast
./zig-out/bin/zolt prove examples/sum.elf --jolt-format -o /tmp/zolt_proof_dory.bin

# Jolt verification test
cd /Users/matteo/projects/jolt
cargo test --package jolt-core test_debug_stage1_verification -- --ignored --nocapture
```
