# Zolt-Jolt Compatibility TODO

## 🎯 Current Task: Fix Stage 4 Sumcheck - Pre/Post Value Tracking

**Status:** Deserialization ✅ | Ra Polynomials ✅ (generated) | Stage 4 Verification ❌ (incorrect values)

### Problem

Ra polynomials are now generating real values from trace data, but Stage 4 sumcheck verification still fails:

```
[JOLT BATCHED] output_claim          = 13790373438827639882557683572286534321489361070389115930961142260387674941556
[JOLT BATCHED] expected_output_claim = 12640480056023150955589545284889516342512199511163763258096899648096280534264
=== SUMCHECK VERIFICATION FAILED ===
Difference: ~1.15e57 (about 5% of field modulus)
```

### Root Cause Analysis

**Hypothesis**: Pre/post value tracking for RdInc/RamInc doesn't match Jolt's approach.

**Key Observations**:
1. Jolt's `cycle.rd_write()` returns `(rd_index, pre_value, post_value)` for each cycle
2. Zolt's `TraceStep` only stores `rd_value` (post-execution value)
3. Current implementation **manually tracks** register state across cycles to compute pre-values
4. This may introduce discrepancies in how increments are computed

**Investigation Needed**:
- Does Jolt store both pre/post values in each Cycle struct?
- How does Jolt's tracer capture the "before" state of rd?
- Should Zolt's TraceStep be extended to include both rd_pre_value and rd_post_value?
- Are we computing increments per-cycle correctly, or should we be using register trace data directly?

### What Was Implemented (Session 58)

**Location**: `src/zkvm/mod.zig:1621-1770`

1. ✅ **RdInc** - Register destination increment polynomial (lines 1621-1661)
   - Formula: `rd_inc[i] = post_value[rd] - pre_value[rd]`
   - Tracks per-register state across trace using hashmap
   - Handles negative increments via field negation
   - Size: `reg_poly_size` (padded to power of 2)

2. ✅ **RamInc** - RAM increment polynomial (lines 1663-1705)
   - Formula: `ram_inc[i] = post_value[addr] - pre_value[addr]`
   - Tracks per-address state across memory trace
   - Only computes increments for memory writes
   - Size: `memory_poly_size` (padded to power of 2)

3. ✅ **InstructionRa[0..31]** - Instruction read address chunks (lines 1707-1721)
   - Extract lookup index chunks using OneHot decomposition
   - Formula: `chunk[i] = (index >> shift[i]) & mask`
   - 32 polynomials for LOG_K=128, log_k_chunk=4

4. ✅ **RamRa[0..ram_d-1]** - RAM read address chunks (lines 1723-1747)
   - Extract RAM address chunks using OneHot decomposition
   - Number of chunks depends on log2(ram_k)

5. ✅ **BytecodeRa[0..bytecode_d-1]** - Bytecode read address chunks (lines 1749-1770)
   - Extract PC chunks using OneHot decomposition
   - Number of chunks depends on log2(bytecode_k)

### Next Steps to Fix Stage 4

1. **Investigate Jolt's Cycle struct**
   - Find how Jolt stores rd pre/post values in `/Users/matteo/projects/jolt/tracer/src/instruction/mod.rs`
   - Compare with Zolt's TraceStep struct at `src/tracer/mod.zig:12-41`

2. **Option A: Extend Zolt's TraceStep**
   - Add `rd_pre_value: u64` field to TraceStep
   - Capture register value BEFORE instruction execution
   - Use direct pre/post values instead of manual tracking

3. **Option B: Use Register Trace Directly**
   - Instead of TraceStep.rd_value, use `emulator.registers.trace`
   - Match Jolt's approach of querying register trace for pre/post values
   - May need to correlate trace entries by timestamp/cycle

4. **Verify Against Jolt Source**
   - Study `jolt-core/src/zkvm/witness.rs:69-77` (RdInc implementation)
   - Understand exactly what `cycle.rd_write().unwrap_or_default()` returns
   - Match the computation precisely

---

## Current Progress

| Stage | Internal (Zolt) | Cross-verify (Jolt) | Notes |
|-------|-----------------|---------------------|-------|
| 1 | ✅ PASS | ✅ PASS | Outer sumcheck working |
| 2 | ✅ PASS | ✅ PASS | Memory checking working |
| 3 | ✅ PASS | ✅ PASS | Bytecode checking working |
| 4 | ✅ PASS | ❌ FAIL | **ACTIVE**: Ra polys generated but values incorrect (~5% off) |
| 5 | ✅ PASS | - | Blocked by Stage 4 |
| 6 | ✅ PASS | - | Blocked by Stage 4 |

**Stage 4 Failure Details**:
- Before Session 58: Ra polynomials were all zeros → `output_claim = 1.3e57`
- After Session 58: Ra polynomials have real values → `output_claim = 1.4e58` (different!)
- This confirms polynomials are being generated, but computation method differs from Jolt

**Recent Progress**:
- Session 57: ✅ Commitment serialization - all 37 commitments generated and serialized
- Session 58: ✅ Ra polynomial implementation - all polynomials use real trace data (no longer zeros)

---

## Commands

```bash
# Generate proof (without SRS for faster iteration)
zig build run -- prove examples/fibonacci.elf --jolt-format --export-preprocessing /tmp/zolt_preprocessing.bin -o /tmp/zolt_proof_dory.bin

# Test cross-verification
cd /Users/matteo/projects/jolt/jolt-core && cargo test test_verify_zolt_proof_with_zolt_preprocessing --release -- --nocapture --ignored

# Run Zolt tests
zig build test

# Quick build check
zig build
```

## Success Criteria

- ✅ Deserialization works (all 37 commitments read correctly)
- ✅ Ra polynomials generate non-zero values from trace data
- ✅ All 5 polynomial types implemented (RdInc, RamInc, InstructionRa, RamRa, BytecodeRa)
- ⚠️ Stage 4 sumcheck passes with Jolt verifier (**values computed but incorrect**)
- ⏳ All 578+ Zolt tests pass (not yet tested)
- ⏳ Zolt proof fully verifies with Jolt for Fibonacci example (blocked by Stage 4)

**Remaining Work**: Fix pre/post value computation in RdInc/RamInc to match Jolt's exact approach

---

## Recent Session History

**Session 58 (2026-01-25)**: ⚡ Implemented Ra polynomial generation - RdInc, RamInc, InstructionRa, RamRa, BytecodeRa all generate real values from trace. Stage 4 still fails but with different output_claim, indicating progress. Need to investigate pre/post value tracking.

**Session 57 (2026-01-25)**: ✅ Fixed serialization - implemented all 37 commitments (RdInc, RamInc, InstructionRa[], RamRa[], BytecodeRa[])

**Session 56 (2026-01-25)**: 🔥 Identified root cause - commitment count mismatch (5 vs 37)

**Session 55 (2026-01-24)**: Fixed double-batching in Stage4GruenProver

**Session 52-54**: Cross-verification setup, deep code audit

See `.agent/SERIALIZATION_BUG_FOUND.md` and `.agent/BUG_FOUND.md` for detailed analysis

---

## Key File References

### Zolt Implementation
- **Ra Polynomial Generation**: `src/zkvm/mod.zig:1621-1770`
  - RdInc: lines 1621-1661
  - RamInc: lines 1663-1705
  - InstructionRa: lines 1707-1721
  - RamRa: lines 1723-1747
  - BytecodeRa: lines 1749-1770
- **Trace Structure**: `src/tracer/mod.zig:12-41` (TraceStep struct)
- **Register Trace**: `src/zkvm/registers/mod.zig:19-70` (RegisterAccess, RegisterTrace)

### Jolt Reference Implementation
- **Witness Generation**: `/Users/matteo/projects/jolt/jolt-core/src/zkvm/witness.rs:56-210`
  - RdInc: lines 69-77
  - RamInc: lines 79-89
  - InstructionRa: lines 91-100
  - BytecodeRa: lines 101-110
  - RamRa: lines 111-124
- **Cycle Structure**: `/Users/matteo/projects/jolt/tracer/src/instruction/mod.rs`
  - `fn rd_write()`: returns `(rd_index, pre_value, post_value)`
  - `fn ram_access()`: returns RAMAccess enum with pre/post values