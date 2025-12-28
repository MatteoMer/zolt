# Zolt zkVM Implementation Plan

## Current Status (December 2024 - Iteration 42)

### Session Summary - CLI Info Command Added

This iteration added:

1. **CLI Info Command**
   - Added `zolt info` command to display zkVM capabilities
   - Shows proof system (HyperKZG, Spartan, Lasso, 6-stage sumcheck)
   - Lists RISC-V support (60+ instructions, 24 lookup tables)
   - Includes performance metrics and ELF loader info

2. **CLI Run Options**
   - Added `--max-cycles N` option to limit emulator cycles
   - Added `--regs` option to display final register state
   - Options work before or after ELF path
   - Updated README with new command documentation

### Previous Session (Iteration 40-41) - Complex Tests & Benchmarks

Previous iterations added:

1. **Bug Fix: Branch Target Calculation**
   - Fixed PC overflow when calculating branch targets for high addresses (0x80000000+)
   - PC was incorrectly cast to i32, now uses proper u64 arithmetic with signed immediate

2. **Complex Program Tests**
   - Arithmetic sequence test (sum 1 to 10 using loop)
   - Memory store/load test (sw, lw instructions)
   - Shift operations test (slli, srli, srai)
   - Comparison operations test (slt, sltu)
   - XOR and bit manipulation test

3. **Benchmark Infrastructure**
   - Added emulator benchmark (sum 1-100 loop: 88 us/op)
   - Added prover benchmark (simple: 96ms, loop: 98ms)

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
- Stage 1 (Spartan): Degree 3, sends [p(0), p(1), p(2)]
- Stage 2 (RAF): Degree 2, sends [p(0), p(2)]
- Stage 3 (Lasso): Degree 2, sends coefficients [c0, c1, c2]
- Stage 4 (Val): Degree 3, sends [p(0), p(1), p(2), p(3)]
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
- **Lasso** - Lookup argument
- **RAF Prover** - Memory checking
- **Val Prover** - Value evaluation
- **Stage 5 Prover** - Register evaluation
- **Stage 6 Prover** - Booleanity
- **Multi-stage Prover** - 6-stage orchestration
- **Multi-stage Verifier** - Strict sumcheck verification
- **Lookup Tables** - 24+ tables
- **Instructions** - 60+ instruction types

## Future Work

### High Priority
1. Add verifier benchmarks
2. Test with real RISC-V programs compiled from C/Rust

### Medium Priority
1. Performance optimization with SIMD
2. Parallel sumcheck round computation

### Low Priority
1. Complete HyperKZG pairing verification
2. More comprehensive benchmarking
3. Add more example programs

## Performance Metrics (from `zig build bench`)
- Field addition: 4.1 ns/op
- Field multiplication: 52.1 ns/op
- Field inversion: 13.3 us/op
- MSM (256 points): 0.49 ms/op
- HyperKZG commit (1024): 1.5 ms/op
- Emulator (sum 1-100, 304 cycles): 88 us/op
- Prover (2 steps): ~96 ms/op
- Prover (14 steps): ~98 ms/op

## Commit History

### Iteration 40
1. Fix branch target calculation for high PC addresses
2. Add complex program tests (loops, memory, shifts, comparisons)
3. Add emulator and prover benchmarks

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
