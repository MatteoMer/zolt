# Zolt-Jolt Compatibility Implementation

## Status: Session 68 - Termination Write Fix Applied

## Previous Issue: R1CS vs Memory Trace Inconsistency

Stage 5 sumcheck was failing due to fundamental inconsistency between R1CS trace and memory trace.

### Root Cause Analysis (COMPLETED)

In Jolt:
1. The SDK macro ALWAYS generates a termination write via `core::ptr::write_volatile(termination_bit as *mut u8, 1)`
2. This becomes an actual SB (Store Byte) instruction in the trace
3. The SB instruction sets the Store flag and RamAddress = termination_addr in R1CS
4. The RAF claim includes this address

In Zolt (before fix):
1. For bare-metal programs (like fibonacci), there was NO termination write in the trace
2. We were injecting a synthetic RAM write but NOT a full trace cycle
3. This caused R1CS RamAddress claim = 0 but RAM trace had the termination access
4. RAF evaluation sumcheck failed due to inconsistency

### Fix Applied

Modified `src/tracer/mod.zig` to inject a FULL synthetic trace cycle for termination:
- Creates a synthetic SB instruction with opcode 0x23
- Sets rs1_value = termination_addr to satisfy R1CS constraint: RamAddress = Rs1 + Imm
- Adds the cycle to both execution trace AND RAM trace
- Now R1CS and RAM trace are consistent

### Current State

1. ✅ Fixed termination write - now injects full R1CS cycle
2. ❌ Stage 1 sumcheck still fails (PRE-EXISTING issue, not caused by termination fix)
3. The Stage 1 failure was present before the termination fix

### Stage 1 Failure Investigation Needed

The Stage 1 (Spartan Outer) sumcheck verification fails with:
```
output_claim:   [8f, 49, 4e, 9d, ca, ...]
expected_claim: [f5, 77, db, ef, 9d, ...]
```

This indicates the R1CS constraint evaluation doesn't match between prover and verifier.

Possible causes:
1. R1CS constraint encoding mismatch
2. Witness generation issues
3. Transcript state divergence
4. Different handling of padding cycles

### Next Steps

1. Debug Stage 1 R1CS constraint evaluation
2. Compare Zolt's R1CS inputs with Jolt's expected format
3. Check if the synthetic termination cycle creates R1CS constraint violations

## Test Commands

```bash
zig build -Doptimize=ReleaseFast
./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format -o /tmp/zolt_proof_dory.bin --export-preprocessing /tmp/zolt_preprocessing.bin --trace-length 64
cd /home/vivado/projects/jolt && cargo test -p jolt-core --features zolt-debug --lib test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture
```

## Files Modified This Session

- `src/tracer/mod.zig`:
  - `recordTerminationWrite()` now injects full trace cycle (not just RAM entry)
  - Synthetic SB instruction with rs1_value = termination_addr
  - Ensures R1CS and RAF consistency
