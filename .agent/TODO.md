# Zolt-Jolt Compatibility TODO

## Current Progress

| Stage | Status | Notes |
|-------|--------|-------|
| 1 | ✅ PASS | Univariate skip sumcheck |
| 2 | ❌ FAIL | RWC expected claim mismatch - investigating |
| 3 | ⏳ Blocked | Waiting for Stage 2 |
| 4 | ⏳ Blocked | Waiting for Stage 3 |
| 5-7 | ⏳ Blocked | Waiting for Stage 4 |

## Session 39 Summary (2026-01-17)

### Completed
1. ✅ Implemented AddressMajor ordering for Phase 2
   - Sort entries by (address, cycle) at start of Phase 2
   - Group by column pairs with checkpoint tracking
   - Compute [s(0), s(2)] and derive s(1) from current_claim
2. ✅ Implemented Phase 2 entry binding
   - bindAddressMajorPair for (Some, Some) case
   - bindAddressMajorEvenOnly for (Some, None) case
   - bindAddressMajorOddOnly for (None, Some) case

### Issue Found: R1CS Witness Missing Memory Operations

**ROOT CAUSE:**
- Fibonacci program has NO STORE/LOAD instructions
- The only "memory write" is the synthetic termination write
- Synthetic write is added to MemoryTrace but NOT to ExecutionTrace
- R1CS witness builds from ExecutionTrace
- Result: RamReadValue=0, RamWriteValue=0 for all cycles

**Evidence:**
- `[R1CS GEN] Total steps with memory access: 0`
- `claim[13] = { 0, 0, ... }` (RamReadValue)
- `claim[14] = { 0, 0, ... }` (RamWriteValue)

**But Jolt expects non-zero:**
- Instance 2 expected_claim = 3148303805315997521479349691467259099534742698741779098822247733559209807773

### Stage 2 Verification Results
- output_claim = 9372091488520543937914220410023422887748128142472392185375157440884454716621
- expected_output_claim = 1649620375365432227061373494601203837685206271295948652652812281106374890120
- MISMATCH

### Next Steps
1. [ ] Investigate how Jolt handles termination for programs without memory ops
2. [ ] Check if synthetic termination write should affect R1CS RamWriteValue claim
3. [ ] Verify all 5 Stage 2 instance claims match Jolt's expectations
4. [ ] Test AddressMajor Phase 2 implementation with program that has actual memory ops

## Files Modified This Session
- `src/zkvm/ram/read_write_checking.zig`
  - Complete rewrite of computePhase2Polynomial for AddressMajor ordering
  - Added computePhase2Evals, computePhase2EvalsEvenOnly, computePhase2EvalsOddOnly
  - Added bindEntriesAddressMajor for Phase 2 binding
  - Added bindAddressMajorPair, bindAddressMajorEvenOnly, bindAddressMajorOddOnly

## Testing Commands

```bash
# Build and run prover
zig build && ./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format --export-preprocessing logs/zolt_preprocessing.bin -o logs/zolt_proof_dory.bin --srs /tmp/jolt_dory_srs.bin

# Run Jolt verifier
cd /Users/matteo/projects/jolt && ZOLT_LOGS_DIR=/Users/matteo/projects/zolt/logs cargo test --package jolt-core test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture
```
