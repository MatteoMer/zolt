# Zolt zkVM Implementation Plan

## Current Status: Session 55 - January 5, 2026

**GOAL**: Make Zolt proofs verifiable by Jolt with Jolt's preprocessing.

---

## Session 55 Findings: R1CS Constraint Key Mismatch

### Key Discovery

**Test with Zolt preprocessing: PASSES**
**Test with Jolt preprocessing: FAILS**

This proves:
1. The sumcheck claim propagation is CORRECT
2. The Gruen polynomial construction is CORRECT
3. The issue is **R1CS constraint key mismatch** between Zolt and Jolt

### How Verification Works

The verifier computes `expected_output_claim` using:
```rust
let inner_sum_prod = self.key.evaluate_inner_sum_product_at_point(rx_constr, r1cs_input_evals);
expected_output_claim = eq_factor * inner_sum_prod * batching_coeff;
```

Where `self.key` is the R1CS sparse constraint key from **preprocessing**.

When Zolt generates its own preprocessing, the constraint key matches what was used during proving, so verification passes.
When using Jolt's preprocessing, the constraint key differs, so verification fails.

### Root Cause

Zolt's R1CS constraints differ from Jolt's. The `evaluate_inner_sum_product_at_point` function evaluates:
- `Az(rx) = sum_y A(rx, y) * z(y)`
- `Bz(rx) = sum_y B(rx, y) * z(y)`
- `inner_sum_prod = Az * Bz`

If A and B matrices differ, the product differs, causing verification failure.

### Next Steps

#### Step 1: Compare R1CS Constraints

Compare Zolt's `UNIFORM_CONSTRAINTS` with Jolt's to identify differences:
- Constraint count (should be same)
- Constraint ordering (must match exactly)
- Linear combination coefficients for condition/left/right

**Zolt file**: `src/zkvm/r1cs/constraints.zig`
**Jolt file**: `/Users/matteo/projects/jolt/jolt-core/src/r1cs/constraints.rs`

#### Step 2: Compare Witness Mapping

Ensure the witness input variables (ALL_R1CS_INPUTS) map to the same indices:
- LeftInstructionInput, RightInstructionInput, Product, etc.
- The order and indices must match exactly

**Zolt file**: `src/zkvm/r1cs/inputs.zig`
**Jolt file**: `/Users/matteo/projects/jolt/jolt-core/src/r1cs/inputs.rs`

#### Step 3: Verify Constraint Evaluation

Add debug output to compare Az, Bz evaluations at each constraint:
- Print Az*Bz product for a test witness
- Compare with Jolt's evaluation

#### Step 4: Fix Mismatches

Once differences are identified, update Zolt's constraints to match Jolt exactly.

### Workaround (Already Working)

If using Zolt-generated preprocessing with Zolt proofs, verification passes.
This can be used as a fallback while fixing the constraint mismatch.

### Key Files

**Zolt:**
- `src/zkvm/spartan/streaming_outer.zig` - streaming outer prover
- `src/poly/split_eq.zig` - GruenSplitEqPolynomial
- `src/poly/multiquadratic.zig` - MultiquadraticPolynomial
- `src/zkvm/proof_converter.zig` - proof generation with transcript

**Jolt:**
- `/Users/matteo/projects/jolt/jolt-core/src/zkvm/spartan/outer.rs` - outer sumcheck
- `/Users/matteo/projects/jolt/jolt-core/src/poly/split_eq_poly.rs` - GruenSplitEqPolynomial
- `/Users/matteo/projects/jolt/jolt-core/src/poly/multiquadratic_poly.rs` - MultiquadraticPolynomial

### Test Commands

```bash
# Build Zolt
zig build -Doptimize=ReleaseFast

# Generate proof with Zolt (with debug output)
./zig-out/bin/zolt prove /tmp/jolt-guest-targets/fibonacci-guest-fib/riscv64imac-unknown-none-elf/release/fibonacci-guest --jolt-format --input-hex 32 -o /tmp/zolt_proof.bin 2>&1 | tee /tmp/zolt_prover.log

# Verify with Jolt (with debug output)
cd /Users/matteo/projects/jolt
cargo test --package jolt-core test_verify_zolt_proof -- --ignored --nocapture 2>&1 | tee /tmp/jolt_verify.log

# Compare logs
diff <(grep 't_zero\|t_infinity\|eq_factor' /tmp/zolt_prover.log) <(grep 't_zero\|t_infinity\|eq_factor' /tmp/jolt_verify.log)
```

### Success Criteria

- `output_claim == expected_output_claim` in Jolt verification
- Stage 1 sumcheck passes without error
- Proof verified successfully by Jolt

---

## Previous Sessions Summary

- **Session 54**: Investigated claim propagation - coefficients match, claims diverge
- **Session 53**: Fixed batching_coeff Montgomery form bug; initial_claim now matches
- **Session 52**: Deep investigation - eq_factor and Az*Bz match but claim doesn't
- **Session 51**: Fixed round offset by adding cache_openings appendScalar
- **Session 50**: Found round number offset between Zolt and Jolt
- **Session 49**: Fixed from_bigint_unchecked interpretation
- **Session 48**: Fixed challenge limb ordering
- **Session 47**: Fixed LookupOutput for JAL/JALR
- **Session 46**: Fixed memory_size mismatch
- **Session 45**: Fixed RV64 word operations

---

## Architecture Reference

### Sumcheck Flow (Stage 1 - Outer Spartan)

1. **UniSkip Round**: Produces r0, uni_skip_claim = poly(r0)
2. **Remaining Rounds** (1 + num_cycle_vars):
   - Compute t_zero, t_infinity from multiquadratic t_prime_poly
   - Build Gruen polynomial: q(0)=t_zero, q(∞)=t_infinity, q(1) from constraint
   - Compute cubic s(X) = l(X) * q(X) where l is linear eq polynomial
   - Send scaled coefficients c0*α, c2*α, c3*α to proof
   - Get challenge r from transcript
   - Update claim: new_claim = s(r)
   - Bind all polynomials (split_eq, t_prime, az, bz)

### Batched Sumcheck Scaling

- Prover tracks UNSCALED claims internally
- Prover scales output coefficients by batching_coeff
- Verifier tracks SCALED claims
- Verifier recovers c1 from: `c1 = scaled_claim - 2*c0 - c2 - c3`

### Gruen Polynomial Construction

```
l(X) = eq_0 + (eq_1 - eq_0) * X       // Linear eq polynomial
q(0) = t_zero                         // From actual polynomial
q(∞) = t_infinity                     // From actual polynomial
q(1) = (claim - l(0)*q(0)) / l(1)     // Derived from constraint
s(X) = l(X) * q(X)                    // Cubic round polynomial
```

### Critical Binding Order

After each round challenge r:
1. `split_eq.bind(r)` - Update eq factor
2. `t_prime_poly.bind(r)` - Bind multiquadratic
3. `az_poly.bindLow(r)`, `bz_poly.bindLow(r)` - Bind Az/Bz

---

## Project Overview

### Phase 1 Complete: Core zkVM
- All 578 tests pass
- 9 C example programs working
- Full CLI (run/trace/prove/verify/stats)
- Binary and JSON proof serialization

### Phase 2: Jolt Compatibility (IN PROGRESS)
- Transcript alignment (Blake2b) ✓
- Proof structure (7 stages + UniSkip) ✓
- Serialization (arkworks format) ✓
- Batching coefficient fix ✓
- **Gruen polynomial claim divergence** ← CURRENT ISSUE

### Components Status ✅
- BN254 field arithmetic, extension fields
- HyperKZG/Dory polynomial commitments
- Spartan R1CS prover/verifier
- Lasso lookup arguments (24 tables)
- 6-stage sumcheck orchestration
- RISC-V emulator (RV64IMC)
- ELF loader (ELF32/ELF64)
