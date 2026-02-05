# Zolt-Jolt Compatibility Implementation

## Status: Session 81 - Termination Store Implemented, Stage 2 Fails

## Current Issue: Stage 2 Sumcheck Output/Expected Claim Mismatch

### Changes Made This Session
1. **Termination step changed from NoOp to real Store instruction**
   - `src/tracer/mod.zig`: `recordTerminationWrite` now creates a real Store trace step
     - instruction=0x00000023 (SB), rs1_value=termination_addr, rs2_value=1
     - is_noop=true (so previous JAL sees NextIsNoop=1, making ShouldJump=0)
     - is_termination_store=true (new field, so generateWitness uses Store witness)
   - RAM trace entry and memory state update restored
   - TraceStep struct: added `is_termination_store` field (default false)
   - padWithNoop: fixed to not skip padding when last step is termination_store

2. **R1CS witness for termination Store**
   - `src/zkvm/r1cs/constraints.zig`: Added `createTerminationStoreWitness()`
     - Calls fromTraceStep() then overrides:
       - FlagDoNotUpdateUnexpandedPC = 1 (constraint 16: 0 = 0+4-4 = 0)
       - FlagIsNoop = 1 (for product factor NextIsNoop consistency)
   - `generateWitness`: checks is_termination_store BEFORE is_noop
   - `src/zkvm/r1cs/jolt_r1cs.zig`: Same ordering fix

3. **Removed synthetic write injection from val_final prover**
   - `src/zkvm/proof_converter.zig`: Replaced `initWithSyntheticWrites` with simple `init()`
   - RAM trace now naturally includes the termination write

4. **Stage 5 prover updated**
   - `src/zkvm/spartan/stage5_prover.zig`: Changed `is_noop` checks to `is_noop and !is_termination_store`
   - This ensures the termination Store is processed (not skipped) in the combined_vals computation

### Results
- Zero R1CS constraint violations
- Stage 1: PASSES ✅
- Stage 2: FAILS ❌ (output_claim ≠ expected_claim)
- Stages 3-7: Not yet reached

### Stage 2 Failure Analysis
Stage 2 fails at the expected_output_claim check after all sumcheck rounds.
The 5 instances:
- Instance 0: ProductVirtualRemainder
- Instance 1: RamRafEvaluation
- Instance 2: RamReadWriteChecking
- Instance 3: OutputSumcheck
- Instance 4: InstructionClaimReduction

The termination Store witness has:
- FlagStore=1, FlagIsNoop=1, FlagDoNotUpdateUPC=1
- Rs1Value=termination_addr, Rs2Value=1
- RamAddress=termination_addr, RamWriteValue=1
- LeftInstructionInput=termination_addr (from left_is_rs1=1)
- LeftLookupOperand=termination_addr
- These non-zero values affect the MLE evaluations used in Stage 2

### Known Issue (Not Blocking Stage 2)
Zolt has a general bug where Store/Load instructions use non-zero instruction inputs:
- Zolt: left_is_rs1=1 for Store → LeftInstructionInput = rs1_value
- Jolt: left_is_rs1=0 for Store → LeftInstructionInput = 0
This doesn't break stages because it's internally consistent (R1CS + Stage 5 both use same wrong values).
But it could be relevant if the termination Store interacts differently.

### Possible Causes for Stage 2 Failure
1. The termination Store's FlagStore=1 might affect the product virtualization remainder differently
2. The R1CS polynomial MLE at the random point might not be consistent between prover and verifier
3. The `base_evals` for the product sumcheck might not correctly include the Store circuit flags

### Next Steps
1. Debug Stage 2 by comparing prover and verifier claims for each instance
2. Check if the issue is in the product virtualization (Instance 0)
3. Consider if FlagStore needs to be 0 for the termination step

## Debug Commands

```bash
# Build Zolt
zig build -Doptimize=ReleaseFast

# Generate proof
./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format -o /tmp/zolt_proof_dory.bin --export-preprocessing /tmp/zolt_preprocessing.bin --trace-length 64

# Verify with Jolt
cd /home/vivado/projects/jolt && cargo test -p jolt-core --features zolt-debug --lib test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture
```
