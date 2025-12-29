# Zolt-Jolt Compatibility TODO

## Current Status: Az*Bz Mismatch (Session 20-24)

### Summary
All Stage 1 sumcheck rounds pass (p(0)+p(1) = claim). The core issue is that the prover's `output_claim` doesn't match the verifier's `expected_output_claim`.

**Key Values:**
- `output_claim` (from sumcheck): 7120341815860535077792666425421583012196152296139946730075156877231654137396
- `expected_output_claim` (from R1CS): 2000541294615117218219795634222435854478303422072963760833200542270573423153
- **Ratio**: ~3.56

### Session 24 Analysis

Traced through Jolt's outer.rs vs Zolt's streaming_outer.zig in detail:

1. **Index Structure**: Both use the same index structure for streaming round:
   - `full_idx = (out_idx * e_in_len + in_idx) * jlen + j`
   - `step_idx = full_idx >> 1` (cycle index)
   - `selector = full_idx & 1` (constraint group)

2. **Multiquadratic Values**: Both compute:
   - `t_prime[0] = Σ eq * Az_g0 * Bz_g0`
   - `t_prime[∞] = Σ eq * (Az_g1 - Az_g0) * (Bz_g1 - Bz_g0)`

3. **Lagrange Weighting**: Both use the same `lagrange_evals_r0[i]` for constraint `i`.

4. **Split Eq Tables**: Both use the same head_out_bits/head_in_bits split calculation.

5. **Verified Components**:
   - ✅ Constraint group indices match (FIRST_GROUP_INDICES, SECOND_GROUP_INDICES)
   - ✅ E_out/E_in factorization formula matches
   - ✅ computeCubicRoundPoly formula matches gruen_poly_deg_3
   - ✅ bind() function updates current_scalar correctly
   - ✅ Lagrange kernel is symmetric

### Suspected Root Cause

The verifier's `inner_sum_prod` is computed from:
```rust
let z = r1cs_input_evals.to_vec();  // From opening claims
z.push(F::one());
az_g0 += w[i] * lc_a.dot_product(&z, z_const_col);
```

This uses the **R1CS INPUT EVALUATIONS** from the proof's opening claims, NOT the trace data directly.

The prover's output_claim comes from the sumcheck over the trace, which should evaluate to the same thing at the binding point.

**Theory**: The issue might be in how the opening claims are being generated. The R1CS input evaluations in the proof might not match the actual polynomial evaluations at the sumcheck point.

### Key Test Output Values
```
tau_high_bound_r0 = 18796872752198882468706643523486633226658657759867260826380105601287106614970
tau_bound_r_tail = 13330793061469069248256603124694999981346909330093035829032594788030901377683
inner_sum_prod = 12743996023445103930025687297173833157935883282725550257061179867498976368827

expected = tau_high_bound_r0 * tau_bound_r_tail * inner_sum_prod
         = 2000541294615117218219795634222435854478303422072963760833200542270573423153

az_g0 = 7543887623553796639022762555706702067884745212652401071075907851931007375221
bz_g0 = 7385987366068817964405961970983403310894645058260096091751841775851519432644
az_g1 = 21642654990609751288487014314124631874918846676589706504157890394950123239009
bz_g1 = 3918541254077008785088751528944340709929299267734721029349343619431186429976
```

### Next Steps

1. **Verify Opening Claims Generation**
   - Check how Zolt computes the R1CS input evaluations
   - Ensure they're evaluated at the correct point (r_cycle_big_endian)

2. **Add Debug Output in Zolt**
   - Print current_scalar at each round
   - Print t_zero, t_infinity before computeCubicRoundPoly
   - Print final Az, Bz values at the bound point

3. **Compare eq Factor Accumulation**
   - Verify current_scalar after all bindings equals L(tau_high, r0) * eq(tau_low, r_all)

### Formula Reference

**Expected Output Claim** (from verifier):
```
expected = L(tau_high, r0) * eq(tau_low, r_tail_reversed) * Az_final * Bz_final

Where:
- r_tail_reversed = [r_n, ..., r_1, r_stream]
- rx_constr = [r_stream, r0]
- Az_final = az_g0 + r_stream * (az_g1 - az_g0)
- Bz_final = bz_g0 + r_stream * (bz_g1 - bz_g0)
- az_gX = Σ_i L_i(r0) * lc_a[i].dot_product(z)
- z = [r1cs_input_evals..., 1]
```

**Prover Output Claim** (from sumcheck):
```
output_claim = current_scalar * eq(tau_curr, r_final) * q(r_final)

Where current_scalar accumulates:
- Initially: L(tau_high, r0)
- After bind(r_i): current_scalar *= eq(tau[i], r_i)
```

## Test Commands

```bash
# Zolt tests
zig build test --summary all

# Generate proof
zig build -Doptimize=ReleaseFast
./zig-out/bin/zolt prove examples/sum.elf --jolt-format -o /tmp/zolt_proof_dory.bin --export-preprocessing /tmp/zolt_preprocessing.bin

# Jolt verification test
cd /Users/matteo/projects/jolt
cargo test --package jolt-core test_debug_stage1_verification -- --ignored --nocapture
```
