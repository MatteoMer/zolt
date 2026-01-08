# Zolt-Jolt Compatibility - Iteration 11 Summary

## Current Status

### Tests
- All Zolt internal tests pass with exit code 0
- Zolt's 6-stage internal verifier PASSES for all stages:
  - Stage 1 (Spartan): PASSED
  - Stage 2 (RAM RAF): PASSED
  - Stage 3 (Lasso): PASSED
  - Stage 4 (Value): PASSED
  - Stage 5 (Register): PASSED
  - Stage 6 (Booleanity): PASSED

### Proof Serialization
- Jolt-format proof serialization works ✓
- Deserialization test in Jolt PASSES (`test_deserialize_zolt_proof`)
- Proof bytes are correctly formatted for arkworks

### Cross-Verification Issue
When attempting to verify Zolt proof with Jolt's verifier:
- Stage 1 sumcheck fails with output_claim mismatch
- **Root cause**: Different programs being proven
  - Zolt uses `fibonacci.elf` (C code, fib(10)=55)
  - Jolt uses `fibonacci-guest` (Rust, fib(50), 128-bit arithmetic, SDK runtime)

### Key Findings

1. **Program Compatibility**: For cross-verification to work, Zolt must prove **the exact same ELF** that Jolt uses.

2. **Jolt Guest ELF Location**:
   ```
   /tmp/jolt-guest-targets/fibonacci-guest-fib/riscv64imac-unknown-none-elf/release/fibonacci-guest
   ```
   - Format: ELF 64-bit LSB, RISC-V, RVC, soft-float ABI, statically linked

3. **Execution Differences**:
   - Jolt guest uses `SYSTEM` (ecall) instructions for I/O
   - Jolt guest expects specific memory layout (input/output regions)
   - Jolt SDK uses postcard serialization for inputs

## Next Steps

### Option A: Run Jolt Guest in Zolt
1. Add support for Jolt's I/O convention (input bytes at 0x7fff8000)
2. Handle SYSTEM calls correctly
3. Match memory layout exactly

### Option B: Create Compatible Test Program
1. Create a minimal Jolt guest that:
   - Uses same build toolchain
   - Has same SDK dependencies
   - Uses same memory layout
2. Have Jolt export the ELF, preprocessing, and proof
3. Use Zolt to prove the same ELF and compare

### Option C: Debug Transcript Divergence
1. Restore Jolt debug output (git stash pop)
2. Compare transcript state at each step
3. Find exact point of divergence

## Technical Details

### Proof Sizes
- Zolt internal format: ~10KB for 256 cycles
- Jolt format (Dory): ~32KB for 256 cycles
- Jolt native proof: ~67KB for 50 iterations

### Key Values from Last Run
- Zolt output_claim: 13341694253320792207898818675544641744423324780186566512606436749524497154341
- Jolt expected: 13993936375866540635487266661744168976503279552720049959007000136664747058730

The mismatch is fundamental - these are different programs producing different traces.
