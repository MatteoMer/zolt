# Zolt-Jolt Compatibility TODO

## Current Status: Az*Bz Product Mismatch (Session 24 - BREAKTHROUGH)

### Summary
All Stage 1 sumcheck rounds pass (p(0)+p(1) = claim). The eq factor accumulation is CORRECT.

**ROOT CAUSE IDENTIFIED**: The prover's Az*Bz product differs from the verifier's expectation.

**Key Values:**
- `output_claim` (from sumcheck): 7120341815860535077792666425421583012196152296139946730075156877231654137396
- `expected_output_claim` (from R1CS): 2000541294615117218219795634222435854478303422072963760833200542270573423153
- **eq_factor** matches between prover and verifier: `tau_high_bound_r0 * tau_bound_r_tail` ✅
- **inner_sum_prod DIFFERS**:
  - Prover's Az*Bz: `6845670145302814045138444113000749599157896909649021689277739372381215505241`
  - Verifier's Az*Bz: `12743996023445103930025687297173833157935883282725550257061179867498976368827`

### Session 24 Analysis

**Key Breakthrough**: Verified eq factor accumulation is CORRECT!
- Computed prover's eq factor = `Product_{i=0}^{10} eq(tau[10-i], r_i)` = `13330793061469069248256603124694999981346909330093035829032594788030901377683`
- Computed verifier's eq factor = `eq(tau_low, r_tail_reversed)` = `13330793061469069248256603124694999981346909330093035829032594788030901377683`
- **These match!**

**Verified Components**:
1. ✅ eq factor accumulation formula
2. ✅ tau ordering and binding
3. ✅ Lagrange kernel initialization
4. ✅ bind() function implementation
5. ✅ Constraint group indices match
6. ❌ Az*Bz product computation

### Root Cause Analysis

The verifier computes:
```rust
// z = [r1cs_input_evals..., 1]
// az_gX = Σ_i L_i(r0) * lc_a[i].dot_product(z)
// Az_final = az_g0 + r_stream * (az_g1 - az_g0)
// Bz_final = bz_g0 + r_stream * (bz_g1 - bz_g0)
// inner_sum_prod = Az_final * Bz_final
```

The prover's sumcheck computes:
```
output_claim = eq_factor * Az*Bz (at sumcheck binding point)
```

Since eq_factor matches, the difference must be in how Az*Bz is computed.

### Possible Issues

1. **Lagrange weights applied differently**
   - Prover uses `lagrange_evals_r0[i]` for each constraint
   - Verifier uses `w[i] * lc_a[i].dot_product(z)`
   - These should give same result if constraint evaluation matches

2. **Constraint evaluation order**
   - Prover iterates over cycles, evaluating constraints
   - Verifier uses the R1CS input evaluations from opening claims
   - Opening claims are MLE evaluations at r_cycle_big_endian

3. **Group combination formula**
   - Both use: `final = g0 + r_stream * (g1 - g0)`
   - But might be applied differently

### Next Steps

1. **Compare Lagrange weights at r0**
   - Print Zolt's `lagrange_evals_r0[i]` for i=0..9
   - Compare with Jolt's `w[i]` from test output

2. **Compare single-cycle Az*Bz**
   - For cycle 0, print both group's Az and Bz
   - Compare with what Jolt would compute for same cycle

3. **Trace constraint evaluation**
   - For constraint 0, print condition and magnitude values
   - Verify they match Jolt's evaluation

### Key Test Output Values
```
Verifier (from R1CS inputs):
az_g0 = 7543887623553796639022762555706702067884745212652401071075907851931007375221
bz_g0 = 7385987366068817964405961970983403310894645058260096091751841775851519432644
az_g1 = 21642654990609751288487014314124631874918846676589706504157890394950123239009
bz_g1 = 3918541254077008785088751528944340709929299267734721029349343619431186429976
inner_sum_prod = 12743996023445103930025687297173833157935883282725550257061179867498976368827

Prover (derived from sumcheck):
inner_sum_prod_prover = 6845670145302814045138444113000749599157896909649021689277739372381215505241

eq_factor (matches):
tau_high_bound_r0 = 18796872752198882468706643523486633226658657759867260826380105601287106614970
tau_bound_r_tail = 13330793061469069248256603124694999981346909330093035829032594788030901377683
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
