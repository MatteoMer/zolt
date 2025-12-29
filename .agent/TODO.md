# Zolt-Jolt Compatibility TODO

## Current Status: Stage 1 Output Claim Mismatch (Session 20-21)

### Summary
All Stage 1 sumcheck rounds pass (p(0)+p(1) = claim), but the **expected output claim** doesn't match the **output claim from sumcheck walk**.

### Latest Test Results
- **output_claim**: 7120341815860535077792666425421583012196152296139946730075156877231654137396
- **expected_output_claim**: 2000541294615117218219795634222435854478303422072963760833200542270573423153
- **Ratio**: ~3.56

### Verified Components (Session 21)
1. ✅ Lagrange kernel computation order matches Jolt (symmetric: K(x,y) = K(y,x))
2. ✅ split_eq initialization with tau_low is correct
3. ✅ bind() function updates current_scalar correctly
4. ✅ Round polynomial uses current_scalar for eq factor
5. ✅ R1CS input claims are non-zero and reasonable
6. ✅ Challenge ordering and reversal logic matches
7. ✅ Prover: `lagrangeKernel(r0, tau_high)` matches Jolt's `lagrange_kernel(&r0, &tau_high)`
8. ✅ Verifier: `lagrange_kernel(tau_high, r0)` (symmetric, so equivalent)

### Suspected Root Causes

1. **Az/Bz computation point mismatch**
   - Prover evaluates constraints at each cycle with Lagrange weights at r0
   - Verifier uses R1CS input claims + rx_constr = [r_stream, r0]
   - The full evaluation should be at (r_stream, r0, r_cycle_big_endian)

2. **Split eq table factorization issue**
   - E_out and E_in tables may not be combining correctly
   - The head_in_bits/head_out_bits split needs verification

3. **R1CS input evaluation endianness**
   - Zolt reverses challenges to get r_cycle_big_endian
   - This should match Jolt's match_endianness() conversion

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

### Next Debugging Steps

1. **Add debug output in Zolt prover**
   - Print eq_val for each cycle during streaming round
   - Print t_zero and t_infinity before cubic poly construction
   - Print current_scalar at each round

2. **Compare Lagrange weights**
   - Print Zolt's lagrange_evals_r0[i] for i=0..9
   - Compare with Jolt's w[i] values from test output

3. **Trace Az/Bz computation**
   - Verify constraint evaluation matches Jolt's dot_product
   - Check if Lagrange weights are applied correctly

### Expected Output Claim Formula
```
expected = tau_high_bound_r0 * tau_bound_r_tail * inner_sum_prod
         = L(τ_high, r0) * eq(τ_low, r_tail_reversed) * Az_final * Bz_final

Az_final = az_g0 + r_stream * (az_g1 - az_g0)
Bz_final = bz_g0 + r_stream * (bz_g1 - bz_g0)
inner_sum_prod = Az_final * Bz_final
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
