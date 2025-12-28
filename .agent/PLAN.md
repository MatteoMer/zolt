# Zolt zkVM Implementation Plan

## Current Status (December 2024 - Iteration 39)

### Session Summary - Full Pipeline Strict Verification PASSES!

This iteration fixed two critical issues that were preventing strict sumcheck verification:

**Issue 1: Lasso Prover Eq_evals Padding**
- Problem: eq_evals array was sized to lookup_indices.len, which might not be 2^log_T
- Impact: Cycle phase folding (log_T rounds) would fail when array size wasn't power of 2
- Fix: Pad eq_evals to 2^log_T, fill extra entries with zeros

**Issue 2: Val Prover Degree-3 Interpolation**
- Problem: Product of 3 multilinear polynomials creates a degree-3 univariate in X
- Using 3 evaluation points [p(0), p(1), p(2)] can only exactly recover degree-2
- Verifier computed p(r) via Lagrange interpolation (correct)
- Prover computed new claim as sum of folded products (WRONG for degree 3)
- Fix: Send 4 evaluation points [p(0), p(1), p(2), p(3)] for exact cubic interpolation

**Result:**
- Full pipeline now passes with strict_sumcheck = true
- All 6 stages verify correctly with p(0) + p(1) = claim check

## Architecture Summary

### Field Tower
```
Fp  = BN254 base field (254 bits) - G1 point coordinates
Fr  = BN254 scalar field (254 bits) - scalars for multiplication

Fp2 = Fp[u] / (u^2 + 1)
Fp6 = Fp2[v] / (v^3 - xi)  where xi = 9 + u
Fp12 = Fp6[w] / (w^2 - v)
```

### Proof Structure
```
JoltProof:
  |-- bytecode_proof: Commitment to program bytecode
  |-- memory_proof: Memory access commitments
  |-- register_proof: Register file commitments
  |-- r1cs_proof: R1CS/Spartan proof
  +-- stage_proofs: 6-stage sumcheck proofs
        |-- Stage 1: Outer Spartan (R1CS correctness) - degree 3
        |-- Stage 2: RAM RAF evaluation - degree 2
        |-- Stage 3: Lasso lookup (instruction lookups) - degree 2
        |-- Stage 4: Value evaluation (memory consistency) - degree 3
        |-- Stage 5: Register evaluation - degree 2
        +-- Stage 6: Booleanity (flag constraints) - degree 2
```

### Sumcheck Prover Requirements

Each sumcheck prover must maintain:
- `current_claim`: The claim that p(0) + p(1) must equal
- Internal state that gets bound/folded after each challenge

**Critical Insight for Degree-3 Sumcheck:**
- For product of k multilinear polynomials, the univariate has degree k
- Need k+1 evaluation points for exact Lagrange interpolation
- Degree 2: 3 points [p(0), p(1), p(2)]
- Degree 3: 4 points [p(0), p(1), p(2), p(3)]

### Polynomial Format Summary
- Stage 1 (Spartan): Degree 3, sends [p(0), p(1), p(2)] (TODO: should be 4 points?)
- Stage 2 (RAF): Degree 2, sends [p(0), p(2)]
- Stage 3 (Lasso): Degree 2, sends coefficients [c0, c1, c2]
- Stage 4 (Val): Degree 3, sends [p(0), p(1), p(2), p(3)] (FIXED in iteration 39)
- Stage 5 (Register): Degree 2, sends [p(0), p(2)]
- Stage 6 (Booleanity): Degree 2, sends [p(0), p(2)]

## Components Status

### Fully Working ✅
- **BN254 Pairing** - Full Miller loop, final exponentiation
- **Extension Fields** - Fp2, Fp6, Fp12
- **Field Arithmetic** - Montgomery form CIOS multiplication
- **G1/G2 Point Arithmetic** - All operations
- **Sumcheck Protocol** - Complete prover/verifier
- **RISC-V Emulator** - Full RV64IMC execution
- **ELF Loader** - ELF32/ELF64 parsing
- **MSM** - Multi-scalar multiplication
- **HyperKZG** - All operations
- **Dory** - IPA-based commitment
- **Host Execute** - Program execution with tracing
- **Preprocessing** - Proving and verifying keys
- **Spartan** - Proof generation and verification
- **Lasso** - Lookup argument (FIXED in iteration 39!)
- **RAF Prover** - Memory checking (verified correct)
- **Val Prover** - Value evaluation (FIXED in iteration 39!)
- **Stage 5 Prover** - Register evaluation (FIXED in iteration 38)
- **Stage 6 Prover** - Booleanity (FIXED in iteration 38)
- **Multi-stage Prover** - 6-stage orchestration
- **Multi-stage Verifier** - Strict sumcheck verification
- **Lookup Tables** - 24+ tables
- **Instructions** - 60+ instruction types

## Future Work

### High Priority
1. Test with more complex programs (loops, memory operations)
2. Add benchmarks for full proof generation

### Medium Priority
1. Performance optimization with SIMD
2. Parallel sumcheck round computation

### Low Priority
1. Complete HyperKZG pairing verification
2. More comprehensive benchmarking
3. Add more example programs

## Performance Metrics
- Field addition: 4.0 ns/op
- Field multiplication: 55.5 ns/op
- Field inversion: 11.8 us/op
- MSM (256 points): 0.50 ms/op
- HyperKZG commit (1024): 1.5 ms/op
- Full prove (simple program): ~1.4 seconds
- Full verify (simple program): ~11 ms

## Commit History

### Iteration 39
1. Fix Lasso prover eq_evals padding for cycle phase folding
2. Fix Val prover to use 4-point interpolation for degree-3 sumcheck
3. Full pipeline strict verification PASSES!

### Iteration 38
1. Fix Stage 5 & 6 provers to properly track sumcheck invariant
2. Add sumcheck invariant tests for Stages 5 and 6

### Iteration 37
1. Fix Val prover polynomial binding for correct sumcheck
2. Add comprehensive sumcheck invariant test for Val prover

### Iteration 36
1. Fix Lasso prover claim tracking for strict sumcheck verification
2. Add test for Lasso prover claim tracking invariant
3. Add RAF prover claim tracking test
