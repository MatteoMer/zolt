# Zolt-Jolt Compatibility - Stage 2 Debugging

## Current Status
Stage 2 sumcheck verification FAILING - output_claim != expected_output_claim

### Latest Values
- output_claim:          19330663220293480209656412701223293038883720376917369484599270483629981507895
- expected_output_claim: 11465411700650658245294917902500914114009241845259440237908447388745039084178

### Tests
- All 712 Zolt tests pass
- Factor flags now correctly tracked (IsRdNotZero, BranchFlag, IsNoop)

## Root Cause Investigation

### Eq Polynomial Ordering Issue

The verifier computes:
```rust
let r_tail_reversed: Vec<F::Challenge> = sumcheck_challenges.iter().rev().copied().collect();
let tau_bound_r_tail_reversed = EqPolynomial::mle(tau_low, &r_tail_reversed);
```

This is `Eq(tau_low, [r_{n-1}, ..., r_1, r_0])`.

The prover uses split_eq with LowToHigh binding:
- Round 0: bind variable 0 to r_0
- Round 1: bind variable 1 to r_1
- ...

After all rounds, the prover has `Eq(tau_low, [r_0, r_1, ..., r_{n-1}])`.

These are NOT the same:
- Prover: Π_i (tau[i] * r_i + ...)
- Verifier: Π_i (tau[i] * r_{n-1-i} + ...)

### Key Question
How does Jolt handle this mismatch? The split_eq polynomial must do something to account for the reversal.

Possible explanations:
1. The split_eq structure naturally handles the reversal through its variable layout
2. There's a compensating transformation somewhere in the code
3. The factor evaluations are computed with the same reversal, canceling out

### Next Steps

1. **Deep dive into split_eq binding**
   - Trace exactly what happens when bind() is called
   - Verify if the accumulated scalar accounts for reversal

2. **Compare with Jolt prover directly**
   - Print intermediate values from Jolt's ProductVirtualRemainderProver
   - Compare with Zolt's values

3. **Simplify test case**
   - Use 4-cycle trace for easier debugging
   - Compare round polynomials between Zolt and Jolt

## Factor Polynomial Status

All 8 factors now correctly extracted:
- [x] LeftInstructionInput - from R1CS input
- [x] RightInstructionInput - from R1CS input
- [x] IsRdNotZero - from FlagIsRdNotZero (rd != 0)
- [x] WriteLookupOutputToRDFlag - from R1CS input
- [x] JumpFlag - from R1CS input
- [x] LookupOutput - from R1CS input
- [x] BranchFlag - from FlagBranch (opcode == 0x63)
- [x] NextIsNoop - from next cycle's FlagIsNoop

## Commands

```bash
# Build and test
zig build test --summary all

# Generate proof
./zig-out/bin/zolt prove --jolt-format -o /tmp/proof.bin examples/fibonacci.elf

# Test with Jolt
cp /tmp/proof.bin /tmp/zolt_proof_dory.bin
cd /Users/matteo/projects/jolt
cargo test --package jolt-core test_verify_zolt_proof -- --ignored --nocapture
```
