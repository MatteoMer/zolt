# Zolt-Jolt Compatibility Implementation

## Status: Stage 6 sumcheck runs with real provers, but verification fails

### Current Issue: Stage 6 Sumcheck Mismatch

Stage 6 runs with real provers for ALL instances, but output_claim != expected_claim.
The mismatch is caused by:

1. **BytecodeReadRaf (Instance 0)**: Val polynomials are all zeros (need bytecode preprocessing data)
2. **BytecodeReadRaf r_cycles**: The 5 r_cycles passed to BytecodeReadRaf may not match Jolt's 5 stages:
   - Stage 1 (SpartanOuter): r_spartan_original - CORRECT
   - Stage 2 (SpartanProductVirtualization): Using r_cycle_stage2_rw - WRONG (that's RamReadWriteChecking)
   - Stage 3 (SpartanShift): Using r_cycle_stage4_val - WRONG (that's RamValEvaluation)
   - Stage 4 (RegistersReadWriteChecking): r_cycle_stage4_regs - CORRECT
   - Stage 5 (RegistersValEvaluation): r_cycle_stage5_regs_val - CORRECT
3. **IncClaimReduction claim tracking**: Using approximate claim halving instead of proper eval

### What's Working
- IncClaimReduction (Instance 5): Real prover with trace data
- HammingBooleanity (Instance 1): Real prover
- RamRaVirtual (Instance 3): Real prover with memory layout remapping
- LookupsRaVirtual (Instance 4): Real prover with interleaved bit extraction
- BytecodeReadRaf (Instance 0): Real prover BUT with wrong Val data and wrong r_cycles
- Booleanity (Instance 2): Zero polynomial (correct for valid traces)

### What Needs Fixing
1. BytecodeReadRaf r_cycles: Need to derive from the opening accumulator:
   - r_cycle_2 = ProductVirtualization r_cycle (from SpartanProductVirtualization/OutputSumcheck)
   - r_cycle_3 = SpartanShift r_cycle (from InstructionInputVirtualization)
   Need to check which opening points these correspond to in proof_converter.zig

2. BytecodeReadRaf Val polynomials: Need bytecode preprocessing to compute per-stage Val values

3. IncClaimReduction claim tracking: The p(1) = instance_claim - p(0) recovery requires
   the EXACT instance claim, but we're using halving approximation.
   Should compute from the round poly evaluation at the challenge point.

### Implementation Order
1. Fix IncClaimReduction claim tracking (most impactful, simplest fix)
2. Fix BytecodeReadRaf r_cycles (need to trace through Jolt's opening accumulator)
3. Compute BytecodeReadRaf Val polynomials from bytecode data
4. Fix Stage 7 if needed
5. End-to-end test

### Test Commands
```bash
zig build -Doptimize=ReleaseFast
./zig-out/bin/zolt prove examples/fibonacci.elf --trace-length 64 -o /tmp/zolt_proof.bin --jolt-format --export-preprocessing /tmp/zolt_preprocessing.bin
cd /home/vivado/projects/jolt && RAYON_NUM_THREADS=1 cargo run --release --features zolt-debug --manifest-path examples/fibonacci/Cargo.toml -- --verify-zolt-proof /tmp/zolt_proof.bin --zolt-preprocessing /tmp/zolt_preprocessing.bin.ram
```

### Key Technical Details

**Lookup index construction**: Uses INTERLEAVED bits (not concatenation)
- rs1 bits go to odd positions (1,3,5,...,127)
- rs2 bits go to even positions (0,2,4,...,126)
- Result = (spread(rs1) << 1) | spread(rs2)

**Chunk extraction ordering**: MSB-first
- chunk_idx=0 extracts the most significant bits
- shift = log_k_chunk * (d - 1 - chunk_idx)

**RAM address remapping**: getLowestAddress() + division by 8
- remapped = (addr - getLowestAddress()) / 8

**LookupsRaVirtual r_address**: NOT reversed (from InstructionReadRaf which doesn't reverse address)
**RamRaVirtual r_address**: IS reversed (from RamRaClaimReduction which reverses both)
