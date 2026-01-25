# Zolt-Jolt Compatibility TODO

## 🎯 Current Task: Debug Stage 4 - Sumcheck Serialization

**Status:** Stages 1-2-3 ✅ PASSING | Stage 4 ❌ Proof serialization mismatch

### CRITICAL BUGS FIXED (Session 61 - 2026-01-25) 🎉

**BUG #1: val_init Iteration After Binding**

**Root Cause:**
- After Phase 2 (address binding) completes, `val_init` is fully bound to a single value at `val_init[0]`
- But `getOpeningClaims()` was iterating over all 65536 elements
- Stale data from indices 4099, 4101, 4102, 4104, 4108 (the original 13 values) **contaminated the result**!

**Why This Happened:**
- Jolt progressively **replaces** the coefficient array after each bind: `self.Z = bound_Z` with `len` halved
- After 16 rounds: `len = 1`, and `final_sumcheck_claim()` asserts this and returns `Z[0]`
- Zolt only updated `val_init[0]` during binding, leaving the full 65536-element array allocated
- The loop `for (0..K)` iterated over stale data!

**The Fix:** `src/zkvm/ram/read_write_checking.zig:1178-1191`
```zig
// OLD (WRONG):
for (0..@min(K, self.val_init.len)) |k| {
    val_claim = val_claim.add(eq_addr.mul(self.val_init[k])); // Iterates stale data!
}

// NEW (CORRECT):
const val_claim = self.val_init[0]; // Just use the bound value!
```

**Result:**
- ✅ `rwc_val_claim` now EQUALS `val_init_eval`
- ✅ Instance 1 (RamValEvaluation) input_claim = 0 ✅
- ✅ Instance 2 (RamValFinalEvaluation) input_claim = 0 ✅

---

**BUG #2: inc Iteration After Binding**

**Root Cause:**
- SAME ISSUE as val_init! After Phase 1 (cycle binding), `inc` is fully bound to `inc[0]`
- But `getOpeningClaims()` was iterating over all 256 elements with stale data

**The Fix:** `src/zkvm/ram/read_write_checking.zig:1210-1216`
```zig
// OLD (WRONG):
for (0..@min(T, self.inc.len)) |j| {
    inc_claim = inc_claim.add(eq_cycle.mul(self.inc[j])); // Iterates stale data!
}

// NEW (CORRECT):
const inc_claim = self.inc[0]; // Just use the bound value!
```

**Result:**
- ✅ **Stage 1 (SpartanOuter) now PASSES!**
- ✅ **Stage 2 (Batched sumcheck with RWC) now PASSES!**

---

**Current Status After Fixes:**

| Stage | Status | Notes |
|-------|--------|-------|
| 1 | ✅ PASS | Fixed by inc bug fix |
| 2 | ✅ PASS | Fixed by inc bug fix |
| 3 | ✅ PASS | Was already working |
| 4 | ❌ FAIL | Proof serialization mismatch |
| 5-7 | ⏸️ Blocked | Waiting for Stage 4 |

**Remaining Issue:**

Stage 4 sumcheck has a **batched claim computation mismatch**:
- Zolt computes final batched_claim: **26477452956988969139049508369983257081713655557682043515982918407251104571143**
- Jolt reads output_claim from proof: **3222202605336969917752560428519260639714485547316395735332587040365989955898**
- Jolt verifier expects (Instance 0 weighted): **14040906104615165865748342028504031406407271517016916344822754339107148783910**

**Deep Investigation (Session 62)**:

✅ **Polynomial coefficients are CORRECT**:
- Verified Round 0-14: ALL coefficients (c0, c2, c3) match exactly between Zolt and Jolt
- Round 0: c0, c2, c3 all match
- Round 14: c0, c2, c3 all match
- The polynomial evaluations being written to the proof are correct!

✅ **Expected output values match**:
- eq_val (LE bytes) matches: `[e4, 11, 97, 09, b9, 11, b8, a5, ...]`
- combined (LE bytes) matches: `[8f, f1, 64, 8c, 0b, 34, 15, a9, ...]`
- expected_output = eq_val * combined is computed correctly by both

❌ **BUT: Final sumcheck claim is wrong**:
- After 15 rounds of sumcheck, Jolt computes output_claim = 3222...
- This should equal the expected_output_claim = 16312... (before batching coefficient)
- Then Instance 0 weighted = expected_output_claim * coeff should = 14040...

**BREAKTHROUGH (Session 62 - Part 2):**

✅ **ALL 15 sumcheck rounds match PERFECTLY!**:
- Initial batched_claim: `8353800845797471892957852304028707864816912644711435457539773061910360616203` ✅ Match
- Round 0 next_claim: `9794045122350346090318168103879095805179191807002823692559248174472879681882` ✅ Match
- Rounds 1-13: ALL MATCH ✅
- Round 14 (final): `3222202605336969917752560428519260639714485547316395735332587040365989955898` ✅ Match

✅ **eq_val * combined computation is CORRECT**:
- Computed: `16312594108896115280983001186205331480915541977747372249472626050588421392719`
- Expected: `16312594108896115280983001186205331480915541977747372249472626050588421392719` ✅ Match!

❌ **BUT: Final sumcheck output doesn't match expected weighted sum**:
- Sumcheck output_claim: `3222...`
- Jolt expects: `coeff[0] * 16312... = 14040...`
- These DON'T match!

**ROOT CAUSE FOUND**: Instance 2 (RamValFinalEvaluation) has **non-zero input_claim**!
- Instance 0 input_claim: Non-zero (correct)
- Instance 1 input_claim: `0` ✅
- Instance 2 input_claim: `5258723638175825215483753966464390100826417414032932059867770167991589922285` ❌ (SHOULD BE 0!)

But Jolt expects Instance 2's **expected_claim = 0**. This means:
1. Instance 2's input_claim is WRONG (should be 0, not 5258...)
2. OR Instance 2 correctly evaluates to 0 at the final point despite non-zero input
3. The non-zero input_claim is contaminating the batched sumcheck!

**Critical**: `input_claim_val_final = output_val_final_claim - output_val_init_claim`
This computation is producing a non-zero value when it should be 0!

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

### Next Steps to Fix Stage 4

1. **Investigate Sumcheck Proof Serialization**
   - Find where `jolt_proof.stage4_sumcheck_proof.rounds` are populated
   - Verify each round's polynomial coefficients are written correctly
   - Check if the batched sumcheck structure matches Jolt's expectations

2. **Compare Proof Structure**
   - Examine what Zolt writes to the proof for Stage 4
   - Compare with what Jolt's verifier expects to read
   - Look for field ordering or structure mismatches

3. **Verify Batched Sumcheck Logic**
   - Ensure the 3 instances are batched correctly
   - Check that Instance 0 has 15 rounds, Instances 1&2 have 8 rounds
   - Verify the scaling/padding logic for different round counts

4. **Check Final Claim Computation**
   - The computed batched_claim (26477...) is very different from what's in the proof (3222...)
   - This suggests either:
     - The wrong value is being written to the proof
     - OR the final claim calculation is incorrect
     - OR there's an off-by-one in which round's claim is used

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
| 1 | ✅ PASS | ✅ PASS | **FIXED Session 61** - inc binding bug |
| 2 | ✅ PASS | ✅ PASS | **FIXED Session 61** - inc binding bug |
| 3 | ✅ PASS | ✅ PASS | Bytecode checking working |
| 4 | ✅ PASS | ❌ FAIL | **ACTIVE**: Proof serialization mismatch |
| 5 | ✅ PASS | - | Blocked by Stage 4 |
| 6 | ✅ PASS | - | Blocked by Stage 4 |

**Stage 4 Current Issue**:
- ✅ Batching coefficients MATCH
- ✅ Instances 1 & 2 input_claims = 0 (fixed by val_init bug)
- ✅ Stage 4 sumcheck computation works (computes batched_claim = 26477...)
- ❌ BUT: Jolt reads different value (3222...) from the serialized proof
- This is a **proof structure/serialization issue**, not a computation error

**Major Fixes (Session 61)**:
- Fixed val_init iteration bug → Instances 1&2 now have input_claim = 0 ✅
- Fixed inc iteration bug → Stages 1&2 now PASS ✅

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

**Session 61 (2026-01-25)**: 🎉 **MAJOR BREAKTHROUGH** - Fixed TWO critical bugs in RWC getOpeningClaims():
1. val_init iteration bug: After binding, only val_init[0] is valid but code iterated over all 65536 elements (stale data contamination!)
2. inc iteration bug: Same issue - only inc[0] is valid after binding but code iterated over all 256 elements
**RESULT: Stages 1 & 2 now PASS! ✅** Instances 1&2 input_claims now correctly = 0. Remaining issue: Stage 4 proof serialization mismatch.

**Session 60 (2026-01-25)**: ✅ Fixed eq polynomial endianness (BIG_ENDIAN) - this made batching coefficients match Jolt!

**Session 59 (2026-01-25)**: ✅ Implemented proper pre/post value tracking in TraceStep and RAMState.

**Session 58 (2026-01-25)**: ⚡ Implemented Ra polynomial generation - RdInc, RamInc, InstructionRa, RamRa, BytecodeRa all generate real values from trace.

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