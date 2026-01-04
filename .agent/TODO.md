# Zolt-Jolt Compatibility TODO

## Current Status: Session 52 - January 5, 2026

**INVESTIGATING: Gruen polynomial claim divergence**

Session 52 deep investigation findings:
- eq_factor (split_eq.current_scalar) matches Jolt exactly ✓
- Az_final * Bz_final (from bound polynomials) matches Jolt's inner_sum_prod exactly ✓
- r1cs_input_evals (opening claims) match exactly ✓
- BUT: output_claim / eq_factor ≠ Az_final * Bz_final ✗

**ROOT CAUSE IDENTIFIED:**

The Gruen polynomial q(X) is constructed to satisfy s(0)+s(1)=previous_claim, but it's NOT equivalent to the multiquadratic bound polynomial. Specifically:
- q(0) = t_zero = Σ eq * Az(0) * Bz(0) ✓
- q(∞) = t_infinity = Σ eq * slope_products ✓
- q(1) is SOLVED from constraint, NOT from t(1) ✗

This means q(r) ≠ bound_t_prime(r), causing the claim to diverge from eq_factor * (Az*Bz at bound point).

**VALUES:**
```
Zolt eq_factor:              11957315549363330504202442373139802627411419139285673324379667683258896529103
Jolt tau_high * tau_bound:   11957315549363330504202442373139802627411419139285673324379667683258896529103 ✓

Zolt az_final * bz_final:    12979092390518645131981692805702461345196587836340614110145230289986137758183
Jolt inner_sum_prod:         12979092390518645131981692805702461345196587836340614110145230289986137758183 ✓

Zolt implied_inner_sum_prod: 15784999673434232655471753340953239083388838864127013231339270095339506918519
(This should equal az_final * bz_final but doesn't!)
```

---

## IMMEDIATE NEXT STEPS

### 1. Investigate Gruen Polynomial Claim Tracking

The Gruen polynomial construction uses:
1. t_zero = t'(0) - sum of weighted Az(0)*Bz(0)
2. t_infinity = t'(∞) - sum of weighted slope products
3. q(1) = (previous_claim - l(0)*t_zero) / l(1)

But q(1) is NOT the same as Σ eq * Az(1) * Bz(1)!

Need to verify if Jolt's implementation has the same issue or if there's something different in how they handle this.

### 2. Compare with Jolt's Claim Tracking

Check if Jolt's prover uses a different method to ensure:
- output_claim = eq_factor * Az_final * Bz_final

Or if there's additional state that needs to be tracked.

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

## Key Debug Output

At round 11 (last round):
```
[ZOLT] ROUND 11 REBUILD: E_out.len = 1, E_in.len = 1
[ZOLT] ROUND 11 AFTER REBUILD: t_prime[0] = az[0]*bz[0] (matches!)
[ZOLT] ROUND 11: t_zero = (correct value)
```

But after the round polynomial is evaluated:
- implied_inner_prod = output_claim / eq_factor ≠ az_final * bz_final

---

## Previous Sessions Summary

- **Session 52**: Deep investigation - eq_factor and Az*Bz match but claim doesn't
- **Session 51**: Fixed round offset by adding cache_openings appendScalar; challenges now match
- **Session 50**: Found round number offset between Zolt and Jolt after r0
- **Session 49**: Fixed from_bigint_unchecked interpretation - tau values now match
- **Session 48**: Fixed challenge limb ordering, round polynomials now match
- **Session 47**: Fixed LookupOutput for JAL/JALR, UniSkip first-round now passes
- **Session 46**: Fixed memory_size mismatch, transcript states now match
- **Session 45**: Fixed RV64 word operations, fib(50) now works
