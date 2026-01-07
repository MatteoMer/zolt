# Zolt-Jolt Compatibility - Session Progress

## Current Status

### ✅ COMPLETED
- Stage 1 passes Jolt verification completely
- All 712 internal tests pass
- R1CS witness generation fixed:
  - RamReadValue for stores now uses pre-value (not zero)
- RWC prover improvements:
  - r_cycle uses correct slice: tau[0..n_cycle_vars]
  - eq computation uses BIG_ENDIAN order (MSB first)
  - inc = new_value - prev_value (signed difference)
  - val_coeff = pre-value for writes, value for reads
  - inc polynomial folds during Phase 1 binding
  - Phase 2 uses eq_evals[0] and inc[0] as scalars
- Memory trace separation:
  - Instruction fetches use untraced reads (bytecode commitment, not RAM trace)
  - Program loading uses untraced writes (initial state, not execution trace)
- ELF loading fixed:
  - proveJoltCompatibleWithDoryAndSrsAtAddress accepts base_address and entry_point
  - Programs loaded at correct address (not hardcoded RAM_START_ADDRESS)

### ✅ RWC SUMCHECK FIXED
- RWC (Instance 2) now produces claim=0 for fibonacci (correct - no RAM operations)
- total_sum and current_claim now match!

### ❌ IN PROGRESS - Stage 2 ProductVirtual Mismatch

**The Issue:**
```
output_claim:          21156486024890420644021865284593324425941707831228671205398077315638214271613
expected_output_claim: 17764994705888080168286558985700513012579491205889951058836940863591331707629
```

Instance breakdown:
- Instance 0 (ProductVirtual): claim=10905316..., contribution=17764994... (problem here)
- Instance 1 (RegistersVal): claim=0
- Instance 2 (RWC): claim=0 ✅ FIXED
- Instance 3 (OutputCheck): claim=0
- Instance 4 (InstructionClaimReduction): claim=0

## Next Steps

1. Investigate ProductVirtual polynomial computation
2. Compare how Zolt vs Jolt compute ProductVirtual claims
3. Check polynomial ordering or coefficient issues

## Verification Commands

```bash
# Build and test Zolt
zig build test --summary all

# Generate Jolt-compatible proof
./zig-out/bin/zolt prove --jolt-format -o /tmp/zolt_proof_dory.bin examples/fibonacci.elf

# Verify with Jolt
cd /Users/matteo/projects/jolt/jolt-core
cargo test test_verify_zolt_proof -- --ignored --nocapture
```

## Code Locations
- RWC prover: `/Users/matteo/projects/zolt/src/zkvm/ram/read_write_checking.zig`
- Jolt RWC: `/Users/matteo/projects/jolt/jolt-core/src/zkvm/ram/read_write_checking.rs`
- R1CS witness: `/Users/matteo/projects/zolt/src/zkvm/r1cs/constraints.zig`
- Proof converter: `/Users/matteo/projects/zolt/src/zkvm/proof_converter.zig`
