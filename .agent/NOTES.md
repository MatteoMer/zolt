# Zolt-Jolt Compatibility Notes

## Current Status (Session 22 - December 29, 2024)

### Stage 1 Index Structure Analysis

**Key Finding**: Jolt's streaming sumcheck uses a complex index structure that Zolt may not be matching exactly.

In Jolt's `fused_materialise_polynomials_round_zero`:
```rust
for (i, (az_chunk, bz_chunk)) in az.par_chunks_exact_mut(grid_size).zip(...).enumerate() {
    // grid_size = 1 << window_size
    while j < grid_size {
        let full_idx = grid_size * i + j;
        let time_step_idx = full_idx >> 1;  // Cycle index
        let selector = full_idx & 1;  // Constraint group
        // ...
    }
    // Final weight by E_out[i]
}
```

The key insight is that:
1. `i` is the combined (x_out, x_in) index
2. `j` is within the window (handling the streaming variable)
3. `time_step_idx = full_idx >> 1` means cycle indices are shared across constraint groups
4. `selector = full_idx & 1` selects the constraint group (0 or 1)

In Zolt's streaming round, we iterate directly over cycles, which may not match this structure.

### Verified in Session 22

1. ✅ Lagrange kernel L(tau_high, r0) - symmetric, matches Jolt
2. ✅ tau_low extraction matches Jolt
3. ✅ split_eq initialization with tau_low and lagrange_tau_r0 scaling
4. ✅ bind() function matches Jolt's eq formula
5. ✅ r_cycle_big_endian computation matches Jolt's normalize_opening_point
6. ✅ R1CS input evaluations use correct eq polynomial

### Still Investigating

- Index structure in computeRemainingRoundPoly for streaming round
- How E_out/E_in indices map to cycle indices
- Whether Zolt's factorized eq is correct for the streaming round

---

## Session 21 - December 29, 2024

### Stage 1 Verification Issue

**Problem**: All sumcheck rounds pass (p(0) + p(1) = claim), but final output_claim ≠ expected_output_claim

**Latest Values:**
- output_claim = 7120341815860535077792666425421583012196152296139946730075156877231654137396
- expected = 2000541294615117218219795634222435854478303422072963760833200542270573423153
- Ratio (integer): ~3.56
- Ratio (field mod p): 8358532945086661905360953846561390757679463074586057649783072130911153544533

### Verified Components (Session 21)

1. ✅ Lagrange kernel L(tau_high, r0) - symmetric, order doesn't matter
2. ✅ split_eq initialization with tau_low
3. ✅ bind() updates current_scalar correctly
4. ✅ computeCubicRoundPoly uses current_scalar for linear eq factor
5. ✅ R1CS constraint group indices match Jolt exactly
6. ✅ t'(∞) formula: product of slopes (Az(1)-Az(0)) * (Bz(1)-Bz(0))
7. ✅ Individual sumcheck rounds pass verification
8. ✅ Jolt uses LinearOnlySchedule (switch_over = 0)

### Key Insight from Session 21

Jolt uses `LinearOnlySchedule::new()` which sets `switch_over_point() = 0`, meaning ALL rounds use linear mode. However, Zolt's implementation has special "streaming round" logic for round 1. This distinction may not matter for correctness, but worth noting.

### Expected Output Claim Formula

```
expected = tau_high_bound_r0 * tau_bound_r_tail_reversed * inner_sum_prod

tau_high_bound_r0 = L(τ_high, r0) = 18796872752198882468706643523486633226658657759867260826380105601287106614970
tau_bound_r_tail = eq(τ_low, r_tail_reversed) = 13330793061469069248256603124694999981346909330093035829032594788030901377683
inner_sum_prod = Az_final * Bz_final = 12743996023445103930025687297173833157935883282725550257061179867498976368827

Az_final = az_g0 + r_stream * (az_g1 - az_g0)
Bz_final = bz_g0 + r_stream * (bz_g1 - bz_g0)

az_g0 = 7543887623553796639022762555706702067884745212652401071075907851931007375221
bz_g0 = 7385987366068817964405961970983403310894645058260096091751841775851519432644
az_g1 = 21642654990609751288487014314124631874918846676589706504157890394950123239009
bz_g1 = 3918541254077008785088751528944340709929299267734721029349343619431186429976
```

### Next Steps

1. Add debug output to Zolt's streaming round:
   - Print t_zero and t_infinity
   - Print E_out and E_in table values for first few entries
   - Print current_scalar at each round

2. Compare Lagrange weights L_i(r0) between Zolt and Jolt

3. Create minimal test with known correct values

### Test Commands

```bash
# Jolt verification test
cd /Users/matteo/projects/jolt
cargo test --package jolt-core test_debug_stage1_verification -- --ignored --nocapture

# Zolt tests
cd /Users/matteo/projects/zolt
zig build test --summary all
```

---

## Previous Sessions

### Session 20 - tau_low fix

Fixed UniSkip to use tau_low instead of full tau, avoiding double-counting τ_high.

### Session 19 - EqPolynomial big-endian fix

Fixed EqPolynomial.evals() to use big-endian indexing.

### Session 17-18 - E_out/E_in tables

Fixed split_eq E tables to use big-endian indexing.

---

## Architecture Notes

### Sumcheck Structure

Stage 1 has:
- 1 UniSkip round (produces r0)
- 1 + num_cycle_vars remaining rounds

For trace_length = 1024:
- num_cycle_vars = 10
- num_rows_bits = 12
- tau.len = 12
- tau_low.len = 11
- Remaining rounds = 11
- r_tail_reversed = [r_10, r_9, ..., r_1, r_stream]

### Big-Endian Convention

From Jolt's eq_poly.rs:
```
evals(r)[i] = eq(r, b₀…b_{n-1})
where i has MSB b₀ and LSB b_{n-1}
```
