# Zolt-Jolt Compatibility TODO

## Current Status: Session 55 - January 5, 2026

**FIXED: Batching coefficient Montgomery form bug** (Session 53)
**REMAINING: rebuildTPrimePoly produces incorrect t_prime values**

Session 55 findings:
- Verified polynomial coefficients (c0, c2, c3) match exactly at all rounds ✓
- Verified challenges match exactly (derived from same Fiat-Shamir transcript) ✓
- Verified eq_factor computation is correct (lagrange_tau_r0 * product of eq bindings) ✓
- Verified scaling model is consistent (prover unscaled, verifier scaled by batching_coeff) ✓
- **BUG FOUND: rebuildTPrimePoly produces wrong t_prime values**
  - At ROUND 2 after rebuild: t_prime[0] = {29, 42, 85, 202, ...}
  - EXPECTED t_prime[0] = az[0]*bz[0] = {6, 206, 182, 204, ...} (different!)
  - This causes claim to drift from true bound polynomial over rounds
- The debug "EXPECTED" is misleading - t_prime[0] should be weighted sum Σ E*az*bz, not just az[0]*bz[0]
- But the actual issue is that buildTPrimePoly indexing may not match bound Az/Bz state

Session 54 findings:
- Polynomial coefficients (c0, c2, c3) match exactly between Zolt and Jolt ✓
- Claim propagation diverges: prover tracks unscaled, verifier tracks scaled
- Changed q(2)/q(3) to use Jolt's recurrence relations (no effect - algebraically equivalent)
- The implied_inner_sum_prod (output_claim / eq_factor) does NOT equal az_final * bz_final
- Root cause: Gruen polynomial q(X) drifts from true bound polynomial t'(X) over rounds

Session 53 findings:
- Fixed batching_coeff bug: was using `[0, 0, low, high]` (MontU128Challenge style) instead of proper Montgomery form
- Added `challengeScalarFull()` for batching coefficients which properly converts to Montgomery form
- Initial claim now matches Jolt exactly ✓
- Sumcheck still fails due to Gruen polynomial claim divergence (original issue)

---

## Fixed in Session 53

### Batching Coefficient Montgomery Form

**Problem:**
- `uni_skip_claim` matched between Zolt and Jolt ✓
- `batching_coeff` did NOT match:
  - Zolt: 3585365310819910961476179832490187488669617511825727803093062673748144578813 (256-bit)
  - Jolt: 38168636090528866393074519943917698662 (128-bit)
- `initial_claim = uni_skip_claim * batching_coeff` therefore didn't match

**Root Cause:**
Jolt uses TWO different challenge scalar methods:
1. `challenge_scalar_optimized()` → returns `F::Challenge` (MontU128Challenge) with `[0, 0, low, high]` representation
2. `challenge_scalar()` (via `challenge_vector`) → returns proper `F` via `F::from_bytes()` with Montgomery conversion

Zolt was using the `[0, 0, low, high]` representation for BOTH, which is correct for sumcheck challenges but WRONG for batching coefficients.

**Fix:**
- Added `challengeScalarFull()` that properly converts to Montgomery form
- Updated `proof_converter.zig` to use `challengeScalarFull()` for batching_coeff
- Now: batching_coeff = 38168636090528866393074519943917698662 (matches Jolt!)
- Now: initial_claim = 21674923214564316833547681277109851767489952526125883853786217589527714841889 (matches Jolt!)

---

## Remaining Issue: Gruen Polynomial Claim Divergence

The sumcheck still fails:
```
output_claim:          3156099394088378331739429618582031493604140997965859776862374574205175751175
expected_output_claim: 6520563849248945342410334176740245598125896542821607373002483479060307387386
```

### Root Cause (from Session 52)

The Gruen polynomial q(X) is constructed to satisfy s(0)+s(1)=previous_claim, but it's NOT equivalent to the multiquadratic bound polynomial:
- q(0) = t_zero = Σ eq * Az(0) * Bz(0) ✓
- q(∞) = t_infinity = Σ eq * slope_products ✓
- q(1) is SOLVED from constraint, NOT from t(1) ✗

This means q(r) ≠ bound_t_prime(r), causing the claim to diverge.

### Next Steps

1. **Fix rebuildTPrimePoly indexing** - The function calls buildTPrimePoly which assumes a specific layout. After Az/Bz are bound, the indices shift. Need to verify:
   - E_out.len * E_in.len * grid_size == az.boundLen() (sizes match)
   - The index formula `grid_size * i + j` accesses correct values after binding

2. **Compare Jolt's compute_evaluation_grid_from_polynomials_parallel** - This is Jolt's equivalent. Key formula:
   ```rust
   let index = grid_size * i + j;
   az_grid[j] = az[index];
   ```
   Verify Zolt uses same indexing scheme.

3. **Add per-round t_zero/t_infinity values** - Print these from Jolt prover (already has debug) and compare with Zolt.

4. **Test with minimal trace** - Try with 2 cycles to reduce debugging complexity.

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

- **Session 55**: Found rebuildTPrimePoly bug - t_prime[0] values don't match expected after rebuild
- **Session 54**: Verified coefficients match, confirmed claim drift issue
- **Session 53**: Fixed batching_coeff Montgomery form bug; initial_claim now matches
- **Session 52**: Deep investigation - eq_factor and Az*Bz match but claim doesn't
- **Session 51**: Fixed round offset by adding cache_openings appendScalar; challenges now match
- **Session 50**: Found round number offset between Zolt and Jolt after r0
- **Session 49**: Fixed from_bigint_unchecked interpretation - tau values now match
- **Session 48**: Fixed challenge limb ordering, round polynomials now match
- **Session 47**: Fixed LookupOutput for JAL/JALR, UniSkip first-round now passes
- **Session 46**: Fixed memory_size mismatch, transcript states now match
- **Session 45**: Fixed RV64 word operations, fib(50) now works
