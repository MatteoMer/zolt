# Zolt-Jolt Compatibility TODO

## Current Status: January 6, 2026 - Session 61

**STATUS: Opening claims are CORRECT, but expected_output_claim doesn't match**

### Key Discovery

The opening claims (r1cs_input_evals) in the proof file are CORRECT and match what Jolt reads. The issue is that Jolt's `expected_output_claim` (computed from R1CS matrices) doesn't match Zolt's `output_claim` (from sumcheck).

### Debug Data

**Zolt's output_claim:** `6773516909001919453588788632964349915676722363381828976724283873891965463518`

**Jolt's expected_output_claim:** `2434835346226335308248403617389563027378536009995292689449902135365447362920`

**Jolt's computation:**
```
expected_output_claim = tau_high_bound_r0 * tau_bound_r_tail_reversed * inner_sum_prod
where:
  tau_high_bound_r0 = 4727117031941196407385944657435434079136516688275922352384973912379816609693
  tau_bound_r_tail_reversed = 13259834722364943624192342920911916962118484982517771566822504207737030243809
  inner_sum_prod = 1221027240042985780108460212824162278077143256096887971142513640043566180374
```

### Root Cause Analysis

The sumcheck output_claim should equal:
```
output_claim = eq(tau, r) * Az(r) * Bz(r)
```

The verifier's expected_output_claim is:
```
expected = tau_factor * inner_sum_prod
        = tau_factor * Az_eval * Bz_eval
```

where Az_eval and Bz_eval are computed from the R1CS matrices and r1cs_input_evals.

The discrepancy suggests either:
1. **R1CS constraints differ** between Zolt and Jolt's preprocessing
2. **R1CS evaluation differs** - how Zolt computes Az*Bz vs how Jolt evaluates
3. **Preprocessing mismatch** - Jolt's verifier key may have different R1CS sparse matrices

### ✅ Confirmed Working

1. **Transcript Compatibility**: All transcript states match
2. **Preamble**: Matches
3. **Dory Commitments**: Match
4. **Tau Derivation**: All 12 values match
5. **UniSkip Polynomial**: All coefficients match
6. **All Sumcheck Challenges**: Match exactly
7. **Opening Claims Serialization**: Values in proof file are correct

### ❌ Current Issue

The R1CS evaluation produces different results:
- Zolt computes `Az*Bz` during sumcheck → produces output_claim
- Jolt verifier computes `Az_eval * Bz_eval` from R1CS matrices → produces expected_output_claim
- These should be equal but aren't

### Next Steps

1. **Compare R1CS constraints** - Verify Zolt's R1CS matches Jolt's exactly
2. **Check R1CS sparse matrix representation** - How Zolt represents vs how Jolt evaluates
3. **Verify preprocessing export** - Ensure Zolt exports R1CS correctly
4. **Add debug to R1CS evaluation** - Print Az_eval, Bz_eval from both sides

### Files to Investigate

- `/Users/matteo/projects/zolt/src/zkvm/r1cs/constraints.zig` - Zolt's R1CS constraints
- `/Users/matteo/projects/jolt/jolt-core/src/zkvm/r1cs/constraints.rs` - Jolt's R1CS constraints
- `/Users/matteo/projects/jolt/jolt-core/src/zkvm/r1cs/evaluation.rs` - Jolt's R1CS evaluation
- `/Users/matteo/projects/zolt/src/zkvm/spartan/streaming_outer.zig` - Zolt's sumcheck

## Test Commands

```bash
# Build and generate proof
zig build -Doptimize=ReleaseFast
./zig-out/bin/zolt prove /tmp/jolt-guest-targets/fibonacci-guest-fib/riscv64imac-unknown-none-elf/release/fibonacci-guest --jolt-format --input-hex 32 --export-preprocessing /tmp/zolt_preprocessing.bin -o /tmp/zolt_proof_dory.bin

# Jolt verification test
cd /Users/matteo/projects/jolt
cargo test --package jolt-core test_verify_zolt_proof -- --ignored --nocapture

# Debug proof format
cargo test --package jolt-core test_debug_zolt_format -- --ignored --nocapture
```
