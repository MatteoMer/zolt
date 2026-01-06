# Zolt-Jolt Compatibility TODO

## Current Status: Session 56 - January 6, 2026

**ROOT CAUSE FOUND: Missing Univariate Skip Implementation**

### New Finding (Session 56)

The sumcheck verification fails because Zolt's prover uses **standard multilinear sumcheck**
but Jolt's verifier expects the **univariate skip optimization** in the first round.

**Evidence:**
```
=== SUMCHECK VERIFICATION FAILED ===
output_claim:          3156099394088378331739429618582031493604140997965859776862374574205175751175
expected_output_claim: 6520563849248945342410334176740245598125896542821607373002483479060307387386
```

The `expected_output_claim` is computed by `evaluate_inner_sum_product_at_point` which:
1. Gets Lagrange weights `w` over the univariate-skip base domain (10 points: -4 to 5)
2. Computes `az_g0, bz_g0` by dotting **first group** constraints (10) with `z`
3. Computes `az_g1, bz_g1` by dotting **second group** constraints (9) with `z`
4. Blends: `az_final = az_g0 + r_stream * (az_g1 - az_g0)`
5. Returns `az_final * bz_final`

But Zolt's prover computes a standard sumcheck over the full 19 constraints without this structure.

### What Needs to Be Implemented

1. **First Round (UniSkip)**: Prover must:
   - Evaluate both constraint groups at Lagrange-weighted points
   - Send coefficients for the univariate polynomial over the base domain
   - Use `r_stream` to blend the two groups in subsequent rounds

2. **Constraint Grouping**: Match Jolt exactly:
   - **First group (10 constraints)**: indices 1,2,3,4,5,6,11,14,17,18
   - **Second group (9 constraints)**: indices 0,7,8,9,10,12,13,15,16

3. **Verify Claims Match**: After implementing univariate skip, the verifier's
   `inner_sum_prod = evaluate_inner_sum_product_at_point(...)` should equal
   what the prover produces.

---

## Completed Tasks

- [x] Compare R1CS input ordering between Zolt and Jolt (they match!)
- [x] Compare R1CS constraint ordering between Zolt and Jolt (they match!)
- [x] Verify transcript challenges match (they do!)
- [x] Verify r1cs_input_evals match between prover and verifier (they do!)
- [x] Fix batching coefficient Montgomery form bug (Session 53)
- [x] Identify root cause: missing univariate skip optimization

## In Progress

- [ ] Implement univariate skip first round in Zolt's Spartan prover
- [ ] Use Lagrange polynomial over domain [-4, 5] for constraint evaluation
- [ ] Split constraints into first group (10) and second group (9)
- [ ] Blend groups with r_stream challenge after first round

---

## Reference Files

### Jolt (implementation to match)
- `jolt-core/src/zkvm/r1cs/evaluation.rs` - Grouped constraint evaluation
- `jolt-core/src/zkvm/r1cs/key.rs` - `evaluate_inner_sum_product_at_point`
- `jolt-core/src/subprotocols/univariate_skip.rs` - UniSkip targets

### Zolt (files to modify)
- `src/zkvm/r1cs/jolt_r1cs.zig` - JoltSpartanInterface (needs univariate skip)
- `src/zkvm/r1cs/evaluation.zig` - Constraint evaluation helpers
- `src/zkvm/r1cs/constraints.zig` - First/second group indices

---

## Key Constants from Jolt

```rust
pub const OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE: usize = 10;  // domain {-4..5}
pub const OUTER_UNIVARIATE_SKIP_DEGREE: usize = 5;        // polynomial degree

// First group labels (10 constraints with boolean guards)
R1CS_CONSTRAINTS_FIRST_GROUP_LABELS = [
    RamAddrEqZeroIfNotLoadStore,       // constraint 1
    RamReadEqRamWriteIfLoad,           // constraint 2
    RamReadEqRdWriteIfLoad,            // constraint 3
    Rs2EqRamWriteIfStore,              // constraint 4
    LeftLookupZeroUnlessAddSubMul,     // constraint 5
    LeftLookupEqLeftInputOtherwise,    // constraint 6
    AssertLookupOne,                   // constraint 11
    NextUnexpPCEqLookupIfShouldJump,   // constraint 14
    NextPCEqPCPlusOneIfInline,         // constraint 17
    MustStartSequenceFromBeginning,    // constraint 18
]

// Second group: constraints 0,7,8,9,10,12,13,15,16
```

---

## Test Commands

```bash
# Build and generate proof
zig build -Doptimize=ReleaseFast
./zig-out/bin/zolt prove /tmp/jolt-guest-targets/fibonacci-guest-fib/riscv64imac-unknown-none-elf/release/fibonacci-guest --jolt-format --input-hex 32 --export-preprocessing /tmp/zolt_preprocessing.bin -o /tmp/zolt_proof_dory.bin

# Jolt verification test (with Zolt preprocessing)
cd /Users/matteo/projects/jolt
cargo test --package jolt-core test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture

# Jolt verification test (with Jolt preprocessing)
cargo test --package jolt-core test_verify_zolt_proof -- --ignored --nocapture
```

---

## Previous Sessions Summary

- **Session 56**: Found root cause - missing univariate skip optimization
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
