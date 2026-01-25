# Zolt-Jolt Compatibility TODO

## 🎯 Current Task: Debug Stage 4 - Input Claim Mismatch

**Status:** Stages 1-3 ✅ PASSING | Stage 4 ❌ Input claims incorrect

### ROOT CAUSE FOUND (Session 61 - 2026-01-25)

**The Problem:**
Stage 4 batched sumcheck verification fails because the **input claims are incorrect**.

**Verified Facts:**
- ✅ Batching coefficients MATCH between Zolt and Jolt (all 3 coefficients identical)
- ✅ eq polynomial computation is CORRECT (BIG_ENDIAN fix from Session 60)
- ❌ Input claims for Stage 4's three instances are WRONG

**Input Claim Mismatch:**

For Fibonacci (no RAM writes), the expected input claims are:
- Instance 0 (RegistersReadWriteChecking): 8494940868042831272571427889592148129715827118309988888518489912562301393374
- Instance 1 (RamValEvaluation): 0
- Instance 2 (RamValFinalEvaluation): 0

But Zolt computes:
- Instance 0: 10960129572097163177603722996998750391162218193231933792862644774061483523224 ❌
- Instance 1: 16843726827876190648710859579071819473340754364270059307512129120184648645607 ❌ (should be 0)
- Instance 2: 5258723638175825215483753966464390100826417414032932059867770167991589922285 ❌ (should be 0)

**Result:**
- Initial batched_claim is computed from wrong input claims
- This propagates through all 15 sumcheck rounds
- Final output_claim = 13790373438827639882557683572286534321489361070389115930961142260387674941556 (9% too high)
- Expected output_claim = 12640480056023150955589545284889516342512199511163763258096280534264 ✓ (Jolt's verifier value)

**The Issue:**
Looking at proof_converter.zig:1756-1757:
```zig
const input_claim_val_eval = stage2_result.rwc_val_claim.sub(val_init_eval);
const input_claim_val_final = stage2_result.output_val_final_claim.sub(stage2_result.output_val_init_claim);
```

For Fibonacci (no RAM writes):
- `rwc_val_claim` ≠ `val_init_eval` (but should be equal!)
- `output_val_final_claim` ≠ `output_val_init_claim` (but should be equal!)

These claims represent RAM state evaluations. For a program with no RAM writes, initial and final RAM states should be identical, making both differences zero.

**Verification:**
With CORRECT input claims (0, 0, 0 becomes 8494940..., 0, 0):
- Initial batched_claim would be: 12640480056023150955589545284889516342512199511163763258096280534264
- This EXACTLY matches Jolt verifier's expected_output_claim! ✅

**Additional Finding:**
- Jolt's PROVER also computes Instance 0 input_claim = 10960129... (same as Zolt)
- But Jolt's VERIFIER expects Instance 0 = 8494940...
- This suggests the input_claim formula might be different between prover and verifier in Jolt
- OR there's something about how Stage 4 instances are set up that we're missing

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

### Analysis Complete - Root Cause Identified (Session 60)

**CRITICAL FINDING**: The issue is in **opening point construction and endianness**!

Jolt's `normalize_opening_point` (read_write_checking.rs:138-173):
1. Splits sumcheck challenges: `[phase1_cycle, phase2_address, phase3]`
2. Reverses each phase: `phase1.rev()`, `phase2.rev()`, `phase3.rev()`
3. **Reorders to**: `[r_address, r_cycle]` = `[phase2.rev(), phase1.rev()]`
4. Returns opening point in BIG_ENDIAN: `[addr_{LOG_K-1}..addr_0, cycle_{log_T-1}..cycle_0]`

Verifier's expected_output_claim (read_write_checking.rs:138+):
```rust
let r = normalize_opening_point(sumcheck_challenges);
let (_, r_cycle) = r.split_at(LOG_K);  // Extract last log_T elements
let eq_val = EqPolynomial::mle_endian(&r_cycle, &params.r_cycle);  // Both BIG_ENDIAN
output_claim = eq_val * combined
```

**Zolt's Current Behavior**:
- ✅ Binding order correct: cycle vars first, then address vars
- ✅ Polynomial indexing correct: `idx = k * T + j`
- ✅ GruenSplitEqPolynomial uses correct BIG_ENDIAN for r_cycle
- ❌ **Missing**: Proper opening point construction for verification

**Detailed Analysis**: See `.agent/STAGE4_CHALLENGE_ORDERING_ANALYSIS.md`

### Fix Applied - BIG_ENDIAN eq polynomial (Session 60)

**CRITICAL FIX IMPLEMENTED** in `proof_converter.zig:2051`:
- Changed from LITTLE_ENDIAN to BIG_ENDIAN for eq polynomial computation
- Converts both r_cycle_sumcheck and stage3_r_cycle to BIG_ENDIAN before calling `mle()`
- Matches Jolt's `RegistersReadWriteCheckingVerifier::expected_output_claim` behavior

**Results**:
- ✅ eq_val now MATCHES between Zolt and Jolt: `[b3, f2, 8c, 0b, 5c, 46, bf, 7d, ...]`
- ✅ combined value MATCHES: `[7a, a0, c5, 44, 6d, 20, 7e, 64, ...]`
- ❌ expected_output still has slight mismatch (different starting at byte 13)
- ❌ Batched output_claim still fails verification

**Current Status**:
- Zolt expected_output: `[de, d9, 51, 0a, d7, 14, 22, 42, 0c, 18, c6, f7, 69, f1, 54, cd, ...]`
- Jolt expected: `[de, d9, 51, 0a, d7, 14, 22, 42, 0c, 18, c6, f7, 51, f4, cd, 90, ...]`
- First 12 bytes match! Difference starts at byte 13

### Root Cause Found - Transcript Divergence! (Session 60 cont.)

**CRITICAL DISCOVERY**: The batching coefficients don't match between Zolt and Jolt!

**Verified Matches**:
- ✅ Input claims ALL MATCH:
  - Instance 0 (RegistersRWC): `10960129572097163177603722996998750391162218193231933792862644774061483523224`
  - Instance 1 (RamValEval): `16843726827876190648710859579071819473340754364270059307512129120184648645607`
  - Instance 2 (RamValFinal): `5258723638175825215483753966464390100826417414032932059867770167991589922285`

**Batching Coefficients Mismatch** ❌:
```
Instance 0:
  Zolt:  163320999741960436325050883603688376856
  Jolt:  32951307615119296988740679619783089786

Instance 1:
  Zolt:  176932814840088648024187126714333406679
  Jolt:  285977201933363026445562837187426589829

Instance 2:
  Zolt:  336461337500628531265857785923175776380
  Jolt:  164827475813667398540528739772213502205
```

**Transcript State Comparison**:
- Zolt after gamma: `{ ee 49 80 e3 36 3b de 88 }`
- Jolt before input claims: `{ 66 a9 f8 b1 cf, b8 4b 45 }` (**DIFFERENT!**)

This proves the transcript states diverge BEFORE sampling batching coefficients, even though:
- ✅ Input claims are correct
- ✅ eq polynomial is correct
- ✅ gamma matches

### Next Steps to Fix

1. **Investigate RAM claim computation in Stage 2**
   - Check how `rwc_val_claim` is computed in Stage 2 RWC sumcheck
   - Verify `val_init_eval` computation (should come from initial RAM state)
   - Compare with Jolt's computation in `/Users/matteo/projects/jolt/jolt-core/src/zkvm/ram/read_write_checking.rs`

2. **Check ValFinal claim computation**
   - Investigate `output_val_final_claim` and `output_val_init_claim`
   - These come from Stage 2's OutputSumcheck
   - Should both evaluate the same polynomial (RamVal) at different points

3. **Verify for no-RAM case**
   - Fibonacci has NO memory writes (only reads of program data)
   - Confirm that Jolt also gets input_claim = 0 for Instances 1 and 2
   - Check if there's special handling for programs without RAM operations

4. **Root cause options**
   - Option A: RAM trace includes spurious writes
   - Option B: Initial/final RAM state evaluations are at wrong points
   - Option C: The formula `rwc_val_claim - val_init_eval` is incorrect

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

**Session 61 (2026-01-25)**: 🎯 **ROOT CAUSE FOUND** - Stage 4 input claims are incorrect! For Fibonacci (no RAM), Instances 1&2 should have input_claim=0 but Zolt computes non-zero values. Instance 0 also has wrong value. The batching coefficients DO match now after Session 60's eq polynomial fix. Issue is in Stage 2's RAM claim computation.

**Session 60 (2026-01-25)**: ✅ Fixed eq polynomial endianness (BIG_ENDIAN) - this made batching coefficients match Jolt! But input claims still wrong.

**Session 59 (2026-01-25)**: ✅ Implemented proper pre/post value tracking in TraceStep and RAMState.

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