# Zolt-Jolt Cross-Verification Progress

## Session 70 Summary - Stage 4 Final Claim Mismatch (2026-01-28)

### Key Finding: regs_current_claim ≠ eq_scalar * combined

After applying the Phase 2/3 fix for `from_evals_and_hint` pattern:
- Sumcheck constraint `p(0)+p(1)=claim` is satisfied at EVERY round ✓
- No constraint violations detected

But the final claim doesn't match expected output:
```
eq_scalar = { 226, 200, 110, 80, ... }
combined (ra*val + wa*(val+inc)) = { 113, 49, 238, 72, ... }
expected (eq * combined) = { 165, 170, 243, 243, ... }
regs_current_claim = { 85, 93, 128, 139, ... }  ← DIFFERENT!
```

### Understanding the Discrepancy

The sumcheck produces:
- `batched_claim` = evolution through rounds (via `p(challenge)` at each round)
- `expected_output` = `eq_scalar * combined` computed AFTER sumcheck

Both should be equal, but they're not. Even though:
- Individual polynomial values are bound correctly
- `eq_scalar = merged_eq[0]` is computed correctly
- `combined` formula matches Jolt's formula

### Root Cause Analysis

The issue is NOT in:
- ✓ Sumcheck constraint enforcement (verified satisfied)
- ✓ Polynomial formula (matches Jolt)
- ✓ Individual claim computations

The issue IS likely in:
- How round polynomial coefficients are computed
- How the eq polynomial is incorporated in each round
- Variable binding order vs MLE evaluation order

### Technical Details

Phase 1 uses `gruenPolyDeg3` which:
1. Computes linear eq polynomial: `l(X) = eq_eval_0 + slope*X`
2. Computes quadratic body polynomial: `q(X) = c + d*X + e*X^2`
3. Returns cubic: `s(X) = l(X) * q(X)`

Phase 2/3 use dense merged_eq and don't multiply by eq factor directly - they use the `from_evals_and_hint` pattern to recover `p(1) = claim - p(0)`.

The discrepancy suggests the polynomial being summed doesn't correctly represent the RegistersRWC instance.

### Next Steps

1. Verify that Phase 2/3 polynomial computation includes eq factor
2. Compare round-by-round with Jolt's prover output
3. Check if `regs_current_claim` should equal `eq * combined` or something else

---

## Session 69 Summary - Stage 4 Internal Consistency Bug (2026-01-28)

### Key Finding: Internal Sumcheck Consistency Failure

The Stage 4 sumcheck internal consistency check was failing because:
```
regs_current_claim ≠ expected_output
```

Even though both use the same:
- eq polynomial value: `{ 217, 23, 239, 206, 84, 218, 209, 78, ... }`
- combined value: `{ 40, 229, 105, 197, ... }`

### Fix Applied: Phase 2/3 from_evals_and_hint Pattern

Modified `stage4_gruen_prover.zig` Phase 2 and Phase 3 to:
1. Compute only `p(0)` and `p(2)` directly
2. Recover `p(1) = previous_claim - p(0)` using the hint
3. This ensures `p(0)+p(1)=claim` constraint is always satisfied

This fix resolved the constraint violations but not the final claim mismatch.

---

## Session 68 Summary - Stage 4 Fix Applied (2026-01-28)

### Fix Applied: Removed Termination Bit Workaround

**Commit:** `cee6b7e` - Remove termination bit workaround from RWC prover

**Analysis:**
1. Jolt's initial RAM state does NOT include termination or panic bits
2. These bits are only in the final RAM state (val_final), used by OutputSumcheck
3. Zolt's workaround was incorrectly adding termination bit to val_init
4. This caused `rwc_val_claim` to include termination when it shouldn't

---

## Session 67 Summary - Stage 4 Input Claim Mismatch (2026-01-27)

### Critical Finding: Stage 4 Transcript Divergence

**Root Cause Identified:**
Stage 4 sumcheck verification fails because the transcript diverges. The input claims appended to the transcript by Zolt don't match what Jolt's verifier computes.

For programs WITHOUT RAM operations (like Fibonacci):
- `claimed_evaluation` at any point = the initial RAM value at that point
- `init_eval` = same initial RAM value at that point (from preprocessing)
- So `input_claim = 0` is CORRECT for RamValEval and ValFinal

---

## Key Files for Investigation

1. `proof_converter.zig:2032-2362` - Main Stage 4 sumcheck loop
2. `stage4_gruen_prover.zig:538-555` - `computeRoundPolynomialGruen` dispatcher
3. `stage4_gruen_prover.zig:561` - `phase1ComputeMessage` (Gruen optimization)
4. `stage4_gruen_prover.zig:764` - `phase2ComputeMessage`
5. `stage4_gruen_prover.zig:853` - `phase3ComputeMessage`
6. `gruen_eq.zig:214` - `gruenPolyDeg3` function
