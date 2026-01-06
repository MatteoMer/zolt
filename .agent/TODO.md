# Zolt-Jolt Compatibility TODO

## Current Status: January 6, 2026 - Session 61

**STATUS: Opening claims are CORRECT, R1CS constraint order verified, output_claim mismatch persists**

### Key Findings

1. **Opening claims in proof file are CORRECT** - Verified that bytes written by Zolt match exactly what Jolt reads
2. **R1CS constraint order matches** - Verified FIRST_GROUP_INDICES and SECOND_GROUP_INDICES match Jolt's
3. **Individual constraints look correct** - Compared RamAddrEqRs1PlusImmIfLoadStore, matches Jolt

### The Issue

Despite all the above matching, Jolt's `expected_output_claim` (computed from R1CS matrices) doesn't match Zolt's `output_claim` (from sumcheck):

- **Zolt output_claim:** `6773516909001919453588788632964349915676722363381828976724283873891965463518`
- **Jolt expected:** `2434835346226335308248403617389563027378536009995292689449902135365447362920`

### Jolt's Computation

```
expected_output_claim = tau_high_bound_r0 * tau_bound_r_tail_reversed * inner_sum_prod
where:
  tau_high_bound_r0 = 4727117031941196407385944657435434079136516688275922352384973912379816609693
  tau_bound_r_tail_reversed = 13259834722364943624192342920911916962118484982517771566822504207737030243809
  inner_sum_prod = 1221027240042985780108460212824162278077143256096887971142513640043566180374
```

`inner_sum_prod` is computed by `evaluate_inner_sum_product_at_point(rx_constr, r1cs_input_evals)` which:
1. Computes Lagrange weights `w` at `r0` over the domain {-4, ..., 5}
2. For each group, sums `w[i] * lc_a.dot_product(z) * w[i] * lc_b.dot_product(z)` over constraints
3. Blends groups using `r_stream`

### Potential Issues to Investigate

1. **Lagrange basis evaluation** - Is Zolt's Lagrange kernel at r0 matching Jolt's?
2. **Group blending formula** - Is `(1-r_stream)*g0 + r_stream*g1` matching Jolt's?
3. **Constraint equation evaluation** - Are the `lc.dot_product(z)` values the same?
4. **rx_constr order** - Is `[r_stream, r0]` or `[r0, r_stream]`?

### Debug Data

All sumcheck challenges match exactly between Zolt and Jolt:
- r0 = 14367833564280337454825687001197154633344501482915202122217190070888490391598
- r_stream (r_sumcheck[0]) = 401701074988603179123933526844662105332873635826937971775978583225973524867

### Test Commands

```bash
# Build and generate proof
zig build -Doptimize=ReleaseFast
./zig-out/bin/zolt prove /tmp/jolt-guest-targets/fibonacci-guest-fib/riscv64imac-unknown-none-elf/release/fibonacci-guest --jolt-format --input-hex 32 --export-preprocessing /tmp/zolt_preprocessing.bin -o /tmp/zolt_proof_dory.bin

# Jolt verification test
cd /Users/matteo/projects/jolt
cargo test --package jolt-core test_verify_zolt_proof -- --ignored --nocapture
```
