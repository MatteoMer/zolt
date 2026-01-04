# Zolt-Jolt Compatibility TODO

## Current Status: Session 51 - January 4, 2026

**FIXED: Round number offset**

Session 51 fixed the round offset issue:
- Jolt's `cache_openings` appends `uni_skip_claim` to transcript BEFORE `BatchedSumcheck::verify`
- Added extra `transcript.appendScalar(uni_skip_claim)` after r0 to match
- UniPoly_begin now at round=59 in both Zolt and Jolt
- Round polynomial coefficients match exactly
- Sumcheck challenges match exactly

**CURRENT ISSUE: Sumcheck output_claim mismatch**

The sumcheck still fails with:
```
output_claim:          1981412718113544531505000459902467367241081743372122430443746733682840647343
expected_output_claim: 5570169908849902992653081094926679248864263885808703143417188980283623941035
```

All individual components match:
- UniPoly_begin at round 59 (both)
- c0, c2, c3 coefficients match byte-for-byte
- First challenge matches byte-for-byte

Need to investigate:
1. How the verifier computes expected_output_claim
2. Whether there's an issue with the final output_claim computation
3. Whether opening claims in the proof are correct

---

## IMMEDIATE NEXT STEPS

### 1. Debug expected_output_claim Computation

The verifier computes expected_output_claim from:
```rust
let tau_high_bound_r0 = LagrangePolynomial::lagrange_kernel(&r0, &tau_high);
let tau_bound_r_tail_reversed = EqPolynomial::mle(tau_low, &r_tail_reversed);
let inner_sum_prod = key.evaluate_inner_sum_product_at_point(rx_constr, r1cs_input_evals);
expected_output_claim = tau_high_bound_r0 * tau_bound_r_tail_reversed * inner_sum_prod * batching_coeff;
```

Compare these intermediate values between Zolt prover and Jolt verifier.

### 2. Verify Opening Claims

The proof contains `opening_claims` which the verifier uses. Ensure these values are correct:
- `Virtual(UnivariateSkip, SpartanOuter)` should equal uni_skip_claim

---

## Test Commands

```bash
# Build and generate proof
zig build -Doptimize=ReleaseFast
./zig-out/bin/zolt prove /tmp/jolt-guest-targets/fibonacci-guest-fib/riscv64imac-unknown-none-elf/release/fibonacci-guest --jolt-format --input-hex 32 -o /tmp/zolt_proof_dory.bin

# Jolt verification test
cd /Users/matteo/projects/jolt
cargo test --package jolt-core test_verify_zolt_proof -- --ignored --nocapture
```

---

## Previous Sessions Summary

- **Session 51**: Fixed round offset by adding cache_openings appendScalar; challenges now match
- **Session 50**: Found round number offset between Zolt and Jolt after r0
- **Session 49**: Fixed from_bigint_unchecked interpretation - tau values now match
- **Session 48**: Fixed challenge limb ordering, round polynomials now match
- **Session 47**: Fixed LookupOutput for JAL/JALR, UniSkip first-round now passes
- **Session 46**: Fixed memory_size mismatch, transcript states now match
- **Session 45**: Fixed RV64 word operations, fib(50) now works
