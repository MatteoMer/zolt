# Zolt-Jolt Compatibility TODO

## 🔥 CRITICAL BUG FOUND - Session 56 (2026-01-25)

**ROOT CAUSE IDENTIFIED: Commitment Serialization Mismatch**

The cross-verification failure is NOT due to incorrect polynomial computation. The issue is in the **serialization format**:

### Issue #1: Uncompressed GT Elements (384 bytes vs ~288 bytes)
- **Location**: `src/zkvm/mod.zig:1360`, `src/field/pairing.zig:635`
- **Problem**: Writing GT elements uncompressed (12 × 32 = 384 bytes)
- **Expected**: Compressed format (~288 bytes per arkworks spec)
- **Impact**: +480 bytes offset for 5 commitments alone!

### Issue #2: Wrong Number of Commitments (5 vs ~37)
- **Location**: `src/zkvm/mod.zig:1360`
- **Problem**: Writing only 5 commitments
- **Expected**: `2 + instruction_d + ram_d + bytecode_d ≈ 37` commitments
  - RdInc, RamInc (2 base)
  - InstructionRa[0..31] (32 for LOG_K=128, log_k_chunk=4)
  - RamRa[0..ram_d-1] (2-5 depending on program)
  - BytecodeRa[0..bytecode_d-1] (1-2 depending on program)
- **Impact**: Missing ~9,000 bytes of commitment data!

### Combined Impact
When Jolt deserializes:
1. Reads commitments: expects ~37 × 288 bytes, gets 5 × 384 bytes
2. **Deserialization offset is wrong by ~9,000 bytes**
3. When reading Stage 4 sumcheck, actually reads Stage 1 data!
4. Stage 4 output_claim equals Stage 1 final claim (evidence of misalignment)

**See:** `.agent/SERIALIZATION_BUG_FOUND.md` for complete analysis

**Files to Fix:**
1. `src/field/pairing.zig` - Add `toBytesCompressed()` for Fp12/GT
2. `src/zkvm/jolt_serialization.zig` - Add `writeGTCompressed()`
3. `src/zkvm/proof_converter.zig` - Track all committed polynomials
4. `src/zkvm/mod.zig` - Serialize all commitments in compressed format

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

**Session 56 (2026-01-25)**: 🔥 ROOT CAUSE FOUND - Commitment serialization mismatch (5 vs 37 commitments, uncompressed vs compressed GT)
**Session 55 (2026-01-24)**: Fixed double-batching bug in Stage4GruenProver; register prover now correct but combined polynomial still fails
**Session 54 (2026-01-24)**: Deep code audit - confirmed ALL polynomial implementations match Jolt exactly
**Session 53 (2026-01-24)**: Coefficient analysis - documented Stage 4 intermediate values
**Session 52 (2026-01-24)**: Cross-verification test shows Stages 1-3 pass, Stage 4 fails due to coefficient mismatch
**Sessions 47-51**: Deep investigation of Stage 4 implementation, E_out/E_in tables, and polynomial computation
**Sessions 43-46**: Fixed Stage 1 MontU128Challenge, eq polynomial, and input_claim handling

For detailed session notes, see previous commits or individual analysis files in `.agent/`
