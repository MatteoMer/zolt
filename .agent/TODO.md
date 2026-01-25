# Zolt-Jolt Compatibility TODO

## ✅ SERIALIZATION BUG FIXED - Session 57 (2026-01-25)

**COMMITMENT SERIALIZATION BUG IS RESOLVED!**

### What Was Fixed

**Issue #2 (Primary Root Cause): Wrong Number of Commitments**
- **Problem**: Writing only 5 commitments instead of ~37
- **Solution**: Implemented all commitment polynomials:
  - RdInc, RamInc (2 base) ✅
  - InstructionRa[0..31] (32 for LOG_K=128, log_k_chunk=4) ✅
  - RamRa[0..ram_d-1] (varies by program) ✅
  - BytecodeRa[0..bytecode_d-1] (varies by program) ✅
- **Files Modified**:
  - `src/zkvm/jolt_types.zig:713` - Changed `dory_commitments` from `[5]GT` to `[]GT` (dynamic)
  - `src/zkvm/mod.zig:864-960` - Build all 37 commitment polynomials with correct OneHot params
  - `src/zkvm/mod.zig:1410-1416` - Serialize all commitments dynamically

**Issue #1 (Clarified): GT Element Size**
- **Investigation Result**: GT elements are ALWAYS 384 bytes in arkworks (no compression)
- **Verification**: Confirmed via Jolt test - both compressed and uncompressed are 384 bytes
- **No Fix Needed**: Original implementation was correct

### Verification Results

**Fibonacci example (instruction_d=32, bytecode_d=2, ram_d=1):**
- ✅ Generating 37 commitments (2 + 32 + 1 + 2)
- ✅ Serializing 37 commitments (14,208 bytes for commitments alone)
- ✅ Jolt successfully deserializes all commitments
- ✅ Passes Stages 1-3 verification
- ✅ Progresses to Stage 4 sumcheck (deserialization working!)

**Remaining Issue**: Stage 4 sumcheck verification fails due to placeholder zero polynomials
- This is a **proof correctness issue**, not a serialization issue
- The Ra polynomial implementations need actual values, not zeros
- Deserialization and format are now correct ✓

### Next Steps

The serialization format is now correct. The next task is to implement proper Ra polynomial generation instead of placeholder zeros. This is tracked separately as it's a proof generation issue, not a serialization bug

---

## Executive Summary (Previous Understanding - NOW SUPERSEDED)

**Status:** 3/6 stages passing cross-verification (Stages 1-3 ✅, Stage 4 ❌)

**BUG FOUND & FIXED:** Double-batching in Stage4GruenProver
- **Location**: `stage4_gruen_prover.zig:430-434`
- **Issue**: Round polynomials batched twice (transcript + storage)
- **Fix**: Store unbatched polynomials
- **Result**: Individual register prover now correct (p(0)+p(1)=claim ✓)

**Previous Hypothesis (INCORRECT):** Combined polynomial in proof_converter
- This was NOT the issue - the polynomials are actually correct!
- The real issue is serialization offset causing Jolt to read wrong data

**See:** `.agent/BUG_FOUND.md` for previous polynomial analysis (still valuable for understanding Stage 4 implementation)

---

## Current Progress

| Stage | Internal (Zolt) | Cross-verify (Jolt) | Notes |
|-------|-----------------|---------------------|-------|
| 1 | ✅ PASS | ✅ PASS | Fixed MontU128Challenge |
| 2 | ✅ PASS | ✅ PASS | - |
| 3 | ✅ PASS | ✅ PASS | - |
| 4 | ✅ PASS | ❌ FAIL | Round poly coefficients diverge |
| 5 | ✅ PASS | - | Blocked by Stage 4 |
| 6 | ✅ PASS | - | Blocked by Stage 4 |

## Session 55 Progress - BUG FOUND! (2026-01-24)

### Major Breakthrough: Found and Fixed Double-Batching Bug

**Bug Found:**
- Location: `stage4_gruen_prover.zig`, lines 430-434
- Issue: Round polynomials were being batched TWICE
  1. First batched when appending to transcript (for Fiat-Shamir)
  2. Then batched AGAIN when storing in `round_polys`
- This caused coefficients to be multiplied by `batching_coeff²` instead of `batching_coeff`

**Fix Applied:**
```zig
// OLD (buggy):
var batched_poly = round_poly;
for (0..4) |i| {
    batched_poly.coeffs[i] = round_poly.coeffs[i].mul(self.batching_coeff);
}
round_polys[round] = batched_poly;

// NEW (fixed):
round_polys[round] = round_poly;  // Store unbatched
```

**Verification:**
- ✅ Individual register prover polynomial is now CORRECT
- ✅ Sumcheck relation holds: `p(0) + p(1) = current_claim`
- ✅ All formulas verified correct through deep code audit
- ❌ Cross-verification still fails (different issue)

**Remaining Problem:**
The proof_converter combines 3 instances:
1. Register RW prover (CORRECT ✓)
2. RAM val evaluation (suspect ❌)
3. Val final evaluation (suspect ❌)

The combined polynomial produces wrong output_claim, even though register prover is correct.
This suggests instances 2 & 3 are contributing non-zero values when they should be zero for programs without RAM.

**Investigation Methodology:**
1. Deep code audit comparing all Zolt vs Jolt implementations
2. Added extensive debug logging for coefficients, evaluations, and claims
3. Traced through proof_converter to find where coefficients diverge
4. Discovered mismatch between gruenPolyDeg3 output and serialization
5. Found double-batching by tracing coefficient storage

**See:** `.agent/BUG_FOUND.md` for complete analysis

---

## Commands

```bash
# Generate proof with Jolt's SRS
zig build run -- prove examples/fibonacci.elf --jolt-format --srs /tmp/jolt_dory_srs.bin --export-preprocessing /tmp/zolt_preprocessing.bin -o /tmp/zolt_proof_dory.bin

# Test cross-verification
cd /Users/matteo/projects/jolt/jolt-core && cargo test test_verify_zolt_proof_with_zolt_preprocessing --release -- --nocapture --ignored

# Run Zolt tests
zig build test
```

## Success Criteria
- All 578+ Zolt tests pass
- Zolt proof verifies with Jolt verifier for Fibonacci example
- No modifications needed on Jolt side

## Files to Study
- Jolt prover: `/Users/matteo/projects/jolt/jolt-core/src/zkvm/registers/read_write_checking.rs`
- Jolt sparse matrix: `/Users/matteo/projects/jolt/jolt-core/src/subprotocols/read_write_matrix/`
- Jolt Gruen eq: `/Users/matteo/projects/jolt/jolt-core/src/poly/split_eq_poly.rs`
- Zolt prover: `/Users/matteo/projects/zolt/src/zkvm/spartan/stage4_prover.zig`

## Previous Sessions Summary (Archived)

**Session 57 (2026-01-25)**: ✅ FIXED commitment serialization - implemented all 37 commitments (RdInc, RamInc, InstructionRa[], RamRa[], BytecodeRa[])
**Session 56 (2026-01-25)**: 🔥 ROOT CAUSE FOUND - Commitment serialization mismatch (5 vs 37 commitments)
**Session 55 (2026-01-24)**: Fixed double-batching bug in Stage4GruenProver; register prover now correct but combined polynomial still fails
**Session 54 (2026-01-24)**: Deep code audit - confirmed ALL polynomial implementations match Jolt exactly
**Session 53 (2026-01-24)**: Coefficient analysis - documented Stage 4 intermediate values
**Session 52 (2026-01-24)**: Cross-verification test shows Stages 1-3 pass, Stage 4 fails due to coefficient mismatch
**Sessions 47-51**: Deep investigation of Stage 4 implementation, E_out/E_in tables, and polynomial computation
**Sessions 43-46**: Fixed Stage 1 MontU128Challenge, eq polynomial, and input_claim handling

For detailed session notes, see previous commits or individual analysis files in `.agent/`
