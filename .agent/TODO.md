# Zolt-Jolt Compatibility - Stage 2 Debugging

## Current Status
Stage 2 sumcheck verification FAILING - output_claim != expected_output_claim

### Test Results
- All 712 Zolt tests pass
- Stage 2 verification fails with:
  - output_claim: 12410090867396887919100369163684640024029021363712395692233063153596180695984
  - expected_output_claim: varies based on r_cycle endianness

## Issues Identified

### 1. Factor Polynomial Computation
The 8 factor polynomials need correct values:
- LeftInstructionInput (u64) ✓
- RightInstructionInput (i128) ✓
- IsRdNotZero - **WRONG**: using RdWriteValue != 0, should be rd_index != 0
- WriteLookupOutputToRDFlag ✓
- JumpFlag ✓
- LookupOutput ✓
- BranchFlag - **WRONG**: using ShouldBranch heuristic, should be instruction opcode 0x63
- NextIsNoop - **WRONG**: using PC heuristic, should be next instruction's IsNoop flag

### 2. Round Polynomial Computation
ProductVirtualRemainderProver computes:
```
s(X) = Σ eq(tau, x) * fused_left(x, X) * fused_right(x, X)
```
This may have endianness issues with tau or challenge binding.

### 3. Endianness
- Tried reversing r_cycle for BIG_ENDIAN (like Jolt's OpeningPoint)
- Still failing, expected_output_claim changes but doesn't match

## Next Steps

1. **Fix factor value extraction**
   - Add rd_index to R1CS witnesses for IsRdNotZero
   - Add instruction opcode for BranchFlag
   - Add IsNoop flag for NextIsNoop

2. **Verify round polynomial formula**
   - Compare Jolt's ProductVirtualRemainderProver round computation
   - Check eq polynomial binding order

3. **Debug with smaller trace**
   - Use 4-cycle program for easier debugging
   - Print intermediate values on both sides

## Technical Details

### Jolt's expected_output_claim formula:
```rust
tau_high_bound_r0 * tau_bound_r_tail_reversed * fused_left * fused_right
```

Where:
- tau_high = tau[tau.len - 1]
- tau_low = tau[0..tau.len - 1]
- r_tail_reversed = sumcheck_challenges reversed
- fused_left = Σ w[i] * factor_left[i](r_cycle)
- fused_right = Σ w[i] * factor_right[i](r_cycle)
- w[i] = Lagrange weights at r0 over 5-point domain

### Product Constraints (5 total):
1. Product = LeftInstructionInput * RightInstructionInput
2. WriteLookupOutputToRD = IsRdNotZero * OpFlags(WriteLookupOutputToRD)
3. WritePCtoRD = IsRdNotZero * OpFlags(Jump)
4. ShouldBranch = LookupOutput * InstructionFlags(Branch)
5. ShouldJump = OpFlags(Jump) * (1 - NextIsNoop)

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
