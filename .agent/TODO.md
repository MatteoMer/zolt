# Zolt-Jolt Compatibility TODO

## 🎯 Current Task: Debug Stage 4 - Pre/Post Values Captured But Still Failing

**Status:** Deserialization ✅ | Ra Polynomials ✅ | Pre/Post Capture ✅ | Stage 4 Verification ❌ (still wrong)

### Problem

Stage 4 sumcheck verification still fails with the SAME values even after implementing pre/post value capture:

```
[JOLT BATCHED] output_claim          = 13790373438827639882557683572286534321489361070389115930961142260387674941556
[JOLT BATCHED] expected_output_claim = 12640480056023150955589545284889516342512199511163763258096899648096280534264
=== SUMCHECK VERIFICATION FAILED ===
Difference: ~1.15e57 (about 5% of field modulus)
```

### What Was Fixed (Session 59 - Today)

**Implemented proper pre/post value tracking**:

1. ✅ **Extended TraceStep** (`src/tracer/mod.zig:12-43`)
   - Added `rd_pre_value: u64` - captures rd value BEFORE instruction execution
   - Added `memory_pre_value: ?u64` - captures memory value BEFORE write
   - Both fields now populated from trace data, matching Jolt's Cycle struct

2. ✅ **Enhanced RAMState** (`src/zkvm/ram/mod.zig`)
   - Modified `MemoryAccess` to store both `pre_value` and `value` (post)
   - Updated `write()` and `writeByte()` to capture pre-value during write
   - RAM trace now records both values, just like Jolt's RAMWrite

3. ✅ **Fixed Tracer** (`src/tracer/mod.zig:266-310`)
   - Captures `rd_pre_value` before execute() at line 266
   - Retrieves `memory_pre_value` from RAM trace after write at lines 286-295
   - Both values correctly populated in TraceStep

4. ✅ **Updated Ra Polynomial Generation**
   - `buildRdIncPolynomial`: Now uses direct `rd_pre_value` and `rd_value` from trace
   - `buildRamIncPolynomial`: Uses `memory_pre_value` and `memory_value` from trace
   - No more manual state tracking - matches Jolt's approach exactly

### Debug Evidence

Pre/post values ARE being captured correctly:
```
[RDINC DEBUG] cycle=0, rd=x2, pre=0, post=32768, inc=32768
[RDINC DEBUG] cycle=1, rd=x2, pre=32768, post=32769, inc=1
[RDINC DEBUG] cycle=2, rd=x2, pre=32769, post=2147549184, inc=2147516415
[RDINC DEBUG] Total rd writes: 52, Non-zero increments: 39
[RAMINC DEBUG] Total writes: 0, Non-zero increments: 0
```

**Observations**:
- RdInc pre-values are CHANGING correctly (0 → 32768 → 32769)
- Fibonacci example has NO memory writes (RamInc is all zeros)
- 39 non-zero register increments out of 52 writes
- Yet output_claim is still wrong by same amount!

### Next Investigation Steps

The pre/post values are correct, but the output_claim is unchanged. Possible issues:

1. **Field Arithmetic** - Are we handling negative increments correctly?
   - Check signed vs unsigned conversion at `@intCast`
   - Verify field negation for negative increments

2. **Polynomial Indexing** - Are increments at correct cycle indices?
   - Verify `poly[i]` matches cycle index in trace
   - Check padding/alignment with poly_size

3. **Batching/Evaluation** - Is Stage 4 batching different?
   - Compare how Zolt vs Jolt evaluates RdInc polynomial
   - Check if there's an off-by-one in how Instance 0 is computed

4. **Other Ra Polynomials** - InstructionRa/RamRa/BytecodeRa might be wrong
   - Instance 0 is RdInc, but Instance 1/2 might affect batching
   - Check if InstructionRa chunks are computed correctly

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