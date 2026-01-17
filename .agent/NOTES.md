# Zolt-Jolt Cross-Verification Progress

## Session 39 Summary - Phase 2 AddressMajor Implementation (2026-01-17)

### Current Status
- **Stage 1: PASSES** ✓
- **Stage 2: FAILS** - investigating RWC expected_claim mismatch

### Key Changes This Session

1. **Implemented AddressMajor ordering for Phase 2**
   - Sort entries by (address, cycle) at start of Phase 2
   - Group by column pairs (2k, 2k+1)
   - Track checkpoints per column pair as entries are processed
   - Compute [s(0), s(2)] and derive s(1) from current_claim

2. **Implemented Phase 2 entry binding**
   - `bindEntriesAddressMajor` function
   - `bindAddressMajorPair` for (Some, Some) case
   - `bindAddressMajorEvenOnly` for (Some, None) case
   - `bindAddressMajorOddOnly` for (None, Some) case

### Major Discovery: R1CS Witness Missing Memory Operations

**ROOT CAUSE FOUND:**
- Fibonacci program has NO actual STORE/LOAD instructions!
- Program terminates via infinite loop detection at cycle 54
- The only "memory write" is the synthetic termination write added to MemoryTrace
- This synthetic write is NOT reflected in ExecutionTrace steps
- R1CS witness is built from ExecutionTrace
- Result: RamReadValue=0, RamWriteValue=0 for ALL cycles

**Evidence from logs:**
```
[R1CS GEN] Total steps with memory access: 0
[ZOLT] OPENING_CLAIMS: claim[13] = { 0, 0, ... }  (RamReadValue)
[ZOLT] OPENING_CLAIMS: claim[14] = { 0, 0, ... }  (RamWriteValue)
```

### Unexpected Result: Jolt expects non-zero RWC claim

**Jolt's expected_claim for Instance 2 (RWC):**
```
expected_claim = 3148303805315997521479349691467259099534742698741779098822247733559209807773
```

This is VERY non-zero, despite:
- Zolt's proof having zero for RamReadValue and RamWriteValue
- The formula being `rv_claim + gamma * wv_claim`

**Possible explanations:**
1. Jolt's fibonacci test might use a different program with actual memory ops
2. Jolt might handle termination differently
3. There might be a serialization mismatch in how claims are stored/read

### Investigation Path

The next step is to add debug output to Jolt's verifier to trace:
1. What values are read from the proof for RamReadValue/RamWriteValue
2. What gamma is used
3. How expected_claim is actually computed

### Files Modified
- `src/zkvm/ram/read_write_checking.zig`
  - Complete rewrite of `computePhase2Polynomial`
  - Added `computePhase2Evals`, `computePhase2EvalsEvenOnly`, `computePhase2EvalsOddOnly`
  - Added `bindEntriesAddressMajor`
  - Added binding helper functions

---

## Session 38 Summary - RWC Phase 2 Investigation (2026-01-17)

### Key Finding: Phase 2 Uses AddressMajor Iteration

**Jolt's Phase 2:**
- Entries sorted by column (address), then by row (cycle)
- Iterates: for each column pair (2k, 2k+1), merge entries by cycle
- Uses `even_checkpoint` and `odd_checkpoint` that track state across cycles
- Checkpoints come from val_init and are updated as entries are processed

**Zolt's Phase 2 (was):**
- Entries remained in cycle-major order
- Iterated directly over entries
- Used bound val_init as static checkpoints
- Missing the cycle-by-cycle state tracking

### Memory Trace vs Execution Trace
- MemoryTrace: Tracks actual RAM accesses (cycle 54: addr=2049, write value=1)
- ExecutionTrace: Records trace steps with optional memory_addr/memory_value
- R1CS witness is built from ExecutionTrace, not MemoryTrace

---

## Previous Sessions Summary

### Stage 4 Issue (from Session 36)
Stage 4 was failing with val_final mismatch - now deferred until Stage 2 is fixed.

### Stage 3 Fix (from Session 35)
- Fixed prefix-suffix decomposition convention (r_hi/r_lo)
- Stage 3 now passes

### Stage 1 Fix
- Fixed NextPC = 0 issue for NoOp padding
- Stage 1 passes

---

## Technical References

- Jolt RamReadWriteChecking: `jolt-core/src/zkvm/ram/read_write_checking.rs`
- Jolt BatchedSumcheck: `jolt-core/src/subprotocols/sumcheck.rs`
- Zolt RWC Prover: `src/zkvm/ram/read_write_checking.zig`
- Zolt Stage 2: `src/zkvm/proof_converter.zig` (generateStage2BatchedSumcheck)
