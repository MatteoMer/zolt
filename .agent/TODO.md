# Zolt-Jolt Compatibility TODO

## Current Status: Session 56 - January 6, 2026

**STATUS: Deep debugging - comparing az/bz group values**

### Latest Debug Values from Jolt Verifier

```
r_stream = 403453513528045288561805897703363889300374796249554293903824070113366263553
r0 = 9956720580376218385912772229440842054804584136250960391272783931628043427152
az_g0 = 7784823065365355150329898612995661848060669911039813026488804049763027733277
bz_g0 = 17295306602198664491678763456607120003277220072842593579631007088433375750847
az_g1 = 1819540214425536184764541679660107604875800049998949590309674377395141916961
bz_g1 = 7283832083962387142906994059148710843912079143868948223318847447205362507259
az_final = 18118550305323548719991270363692126563213624719115355489360394285597267782328
bz_final = 189353324837795743110663437939657740667583580724624394002588005226092056829
inner_sum_prod = 3428564744352329898278095955238265070037131657307455691194697055242544749299
```

### Test Results

```
output_claim:          3156099394088378331739429618582031493604140997965859776862374574205175751175
expected_output_claim: 6520563849248945342410334176740245598125896542821607373002483479060307387386
```

The sumcheck `output_claim` doesn't match `expected_output_claim`.

### Root Cause Analysis

The verifier computes:
```rust
expected_output_claim = tau_high_bound_r0 * tau_bound_r_tail_reversed * inner_sum_prod
```

Where `inner_sum_prod = az_final * bz_final` comes from:
```rust
az_final = az_g0 + r_stream * (az_g1 - az_g0)
bz_final = bz_g0 + r_stream * (bz_g1 - bz_g0)
```

And `az_g0`, `az_g1`, `bz_g0`, `bz_g1` are computed by:
```rust
for i in 0..FIRST_GROUP.len() {
    az_g0 += w[i] * lc_a[i].dot_product(&z, z_const_col);
    bz_g0 += w[i] * lc_b[i].dot_product(&z, z_const_col);
}
```

**The prover must produce a final sumcheck claim that equals this formula.**

### Next Steps

1. Add debug output to Zolt prover to compute the same az/bz values
2. Compare with Jolt's verifier values to find where they diverge
3. Fix the computation that differs

### Files to Add Debug Output

- `src/zkvm/spartan/streaming_outer.zig` - Add function to compute az_g0/az_g1/bz_g0/bz_g1 at the bound point
- Need to compute: `Σ w[i] * constraint_i.condition * z(r_cycle)` for Az
- And: `Σ w[i] * (constraint_i.left - constraint_i.right) * z(r_cycle)` for Bz

---

## Completed Tasks

- [x] Compare R1CS input ordering between Zolt and Jolt (they match!)
- [x] Compare R1CS constraint ordering between Zolt and Jolt (they match!)
- [x] Verify transcript challenges match (they do!)
- [x] Verify r1cs_input_evals match between prover and verifier (they do!)
- [x] Fix batching coefficient Montgomery form bug (Session 53)
- [x] Identify root cause: missing univariate skip optimization (partially)
- [x] Add debug output to Jolt verifier to see az/bz group values

## In Progress

- [ ] Add debug output to Zolt prover to compute same values
- [ ] Compare Zolt az_g0/bz_g0/az_g1/bz_g1 with Jolt values

---

## Test Commands

```bash
# Build and generate proof
zig build -Doptimize=ReleaseFast
./zig-out/bin/zolt prove /tmp/jolt-guest-targets/fibonacci-guest-fib/riscv64imac-unknown-none-elf/release/fibonacci-guest --jolt-format --input-hex 32 --export-preprocessing /tmp/zolt_preprocessing.bin -o /tmp/zolt_proof_dory.bin

# Jolt verification test (with Jolt preprocessing)
cd /Users/matteo/projects/jolt
cargo test --package jolt-core test_verify_zolt_proof -- --ignored --nocapture
```

---

## Key Formulas

### Verifier's inner_sum_prod

```rust
// Lagrange weights at r0 over base domain {-4..5}
let w = LagrangePolynomial::evals(&r0);

// Group 0 (10 constraints)
for i in 0..FIRST_GROUP.len() {
    az_g0 += w[i] * lc_a[i].dot_product(&z, z_const_col);
    bz_g0 += w[i] * lc_b[i].dot_product(&z, z_const_col);
}

// Group 1 (9 constraints)
for i in 0..SECOND_GROUP.len() {
    az_g1 += w[i] * lc_a[i].dot_product(&z, z_const_col);
    bz_g1 += w[i] * lc_b[i].dot_product(&z, z_const_col);
}

// Blend with r_stream
az_final = az_g0 + r_stream * (az_g1 - az_g0)
bz_final = bz_g0 + r_stream * (bz_g1 - bz_g0)

inner_sum_prod = az_final * bz_final
```

### Prover's Final Claim

```
output_claim = current_claim after all sumcheck rounds

// This should equal:
// sum over all (row, x_cycle) of: eq(tau, row||x_cycle) * Az(row, x_cycle) * Bz(row, x_cycle)
// bound to the challenge point [r0, r_stream, r_1, ..., r_n]
```

### Expected Relationship

```
output_claim == tau_high_bound_r0 * tau_bound_r_tail_reversed * inner_sum_prod
```
