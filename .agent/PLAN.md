# Zolt zkVM Implementation Plan

## Current Status (December 2024 - Iteration 37)

### Session Summary - Val Prover Fix

This iteration fixed the Val (value evaluation) prover polynomial binding issue.

**Problem:**
The Val evaluation prover was only binding the `inc` polynomial after each sumcheck
challenge, but not the `wa` (write-address) and `lt` (less-than) polynomials.
This caused the sumcheck invariant p(0) + p(1) = current_claim to fail.

**Solution:**
1. Materialize all three polynomial evaluations (inc, wa, lt) upfront
2. Bind all three polynomials together after each challenge using linear interpolation
3. Track current_claim properly through sumcheck rounds
4. New claim = sum over folded evaluations after binding

**Test Results:**
- All 554 tests pass
- Val prover sumcheck invariant test: PASS

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
        |-- Stage 1: Outer Spartan (R1CS correctness)
        |-- Stage 2: RAM RAF evaluation
        |-- Stage 3: Lasso lookup (instruction lookups)
        |-- Stage 4: Value evaluation (memory consistency)
        |-- Stage 5: Register evaluation
        +-- Stage 6: Booleanity (flag constraints)
```

### Sumcheck Prover Requirements
Each sumcheck prover must maintain:
- `current_claim`: The claim that p(0) + p(1) must equal
- Internal state that gets bound/folded after each challenge

After round i:
1. Compute polynomial p_i(X) where p_i(0) + p_i(1) = current_claim
2. Send p_i to verifier (in various formats: coefficients, evaluations, compressed)
3. Receive challenge r from verifier
4. Update internal state: bind variable i to r
5. Update current_claim = p_i(r)

### Polynomial Format Summary
- Stage 1 (Spartan): Degree 3, sends [p(0), p(1), p(2)]
- Stage 2 (RAF): Degree 2, sends [p(0), p(2)]
- Stage 3 (Lasso): Degree 2, sends coefficients [c0, c1, c2]
- Stage 4 (Val): Degree 3, sends [p(0), p(1), p(2)]
- Stage 5 (Register): Degree 2, sends [p(0), p(2)]
- Stage 6 (Booleanity): Degree 2, sends [p(0), p(2)]

## Components Status

### Fully Working
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
- **Lasso** - Lookup argument (claim tracking fixed in iteration 36!)
- **RAF Prover** - Memory checking (verified correct)
- **Val Prover** - Value evaluation (claim tracking fixed in iteration 37!)
- **Multi-stage Prover** - 6-stage orchestration
- **Lookup Tables** - 24+ tables
- **Instructions** - 60+ instruction types

## Future Work

### High Priority
1. Test full pipeline with strict verification mode (all stages)
2. Investigate test interference issue (e2e test causes other tests to fail)

### Medium Priority
1. Performance optimization with SIMD
2. Parallel sumcheck round computation

### Low Priority
1. More comprehensive benchmarking
2. Add more example programs

## Performance Metrics
- Field addition: 4.0 ns/op
- Field multiplication: 55.5 ns/op
- Field inversion: 11.8 us/op
- MSM (256 points): 0.50 ms/op
- HyperKZG commit (1024): 1.5 ms/op

## Commit History

### Iteration 37
1. Fix Val prover polynomial binding for correct sumcheck
2. Add comprehensive sumcheck invariant test for Val prover

### Iteration 36
1. Fix Lasso prover claim tracking for strict sumcheck verification
2. Add test for Lasso prover claim tracking invariant
3. Add RAF prover claim tracking test
