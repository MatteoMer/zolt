# Zolt-Jolt Compatibility TODO

## Current Status: Stage 1 Output Claim Mismatch (Session 20)

### Summary
All Stage 1 sumcheck rounds pass (p(0)+p(1) = claim), but the **expected output claim** doesn't match the **output claim from sumcheck walk**.

### Key Fixes Applied This Session
1. ✅ R1CSInputIndex order matches Jolt's ALL_R1CS_INPUTS
2. ✅ UniSkip polynomial uses tau_low for eq (not full tau, avoiding double-counting τ_high)

### Latest Test Results
After tau_low fix:
- **output_claim**: 7120341815860535077792666425421583012196152296139946730075156877231654137396
- **expected_output_claim**: 2000541294615117218219795634222435854478303422072963760833200542270573423153
- **Ratio**: ~3.56 (improved from ~1.338)

### Key Discoveries

1. **Lagrange Kernel Placement**
   - s1(Y) = L(τ_high, Y) * t1(Y) (UniSkip includes Lagrange)
   - Remaining rounds: current_scalar = L(τ_high, r0) (also includes Lagrange)
   - Verifier: expected = tau_high_bound_r0 * tau_bound_r_tail * inner_sum_prod
   - This is CORRECT - the kernel appears in both prover and verifier consistently

2. **Constraint Group Selection**
   - In linear phase, selector = full_idx & 1 (LSB of index)
   - r_idx represents bound streaming challenges
   - selector comes from bound r_stream, not a separate variable

3. **Streaming Window vs Linear Phase**
   - switch_over = num_remaining_rounds / 2
   - Streaming window phase: rounds 1 to switch_over, r_grid is updated
   - Linear phase: rounds > switch_over, r_grid is fixed
   - Both phases should use same formula (window_size = 1 for linear phase)

### Remaining Investigation

1. **Debug polynomial computation**
   - Add print statements to trace t_zero, t_infinity, eq_val through rounds
   - Compare with Jolt's computed values

2. **Verify Az/Bz at challenge point**
   - The prover computes Az/Bz via constraints + Lagrange weights
   - The verifier computes via opening claims + linear combinations
   - These should produce identical values

3. **Check r_grid update logic**
   - Verify r_grid.update() produces correct eq evaluations
   - Ensure binding order matches Jolt's LowToHigh order

### Expected Output Claim Formula
```
expected = tau_high_bound_r0 * tau_bound_r_tail * inner_sum_prod
         = L(τ_high, r0) * eq(τ_low, r_tail_reversed) * Az_final * Bz_final

Az_final = az_g0 + r_stream * (az_g1 - az_g0)
Bz_final = bz_g0 + r_stream * (bz_g1 - bz_g0)
inner_sum_prod = Az_final * Bz_final
```

## Commits This Session
1. `fix: align R1CSInputIndex enum order with Jolt's ALL_R1CS_INPUTS`
2. `fix: use tau_low for eq polynomial in UniSkip, not full tau`
3. `chore: update TODO with detailed analysis of Stage 1 verification`

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
