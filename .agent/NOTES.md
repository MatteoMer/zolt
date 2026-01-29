# Zolt-Jolt Cross-Verification Progress

## Session 72 Summary - Stage 3 Sumcheck Verification (2026-01-29)

### Key Findings

#### Stage 3 Sumcheck is Mathematically Correct ✓

All 8 rounds of Stage 3 sumcheck satisfy the constraint `p(0) + p(1) = claim`:
- Round 0: p0+p1 = current_claim ✓
- Round 1: p0+p1 = current_claim ✓
- ...all rounds verified...

This means the prover is producing a valid sumcheck proof for its chosen claims.

#### Individual Instance Claims Match

At each round, the batched claim correctly decomposes:
- shift_p0+p1 = shift_claim ✓
- instr_p0+p1 = instr_claim ✓
- reg_p0+p1 = reg_claim ✓

#### What This Means

The sumcheck mechanics are correct. If Jolt's verifier rejects the proof, it must be because:
1. The **initial input_claims** differ from what Jolt would compute
2. The **opening claims** stored in the proof differ from what Jolt expects
3. The **transcript state** differs, causing different challenge derivation

### Next Steps

1. Run Jolt verifier to see exact error message
2. Compare the serialized opening claims in proof with what Jolt expects
3. Verify transcript state at Stage 3 boundary matches Jolt exactly

---

## Session 71 Summary - Stage 4 Batched Sumcheck Debug (2026-01-28)

### Key Findings

#### 1. Instance 0 (RegistersRWC) is CORRECT
- After all 15 rounds, `regs_current_claim` matches `expected_output (eq * combined)`
- The eq polynomial (`merged_eq[0]`) matches `eq_val_be` computed via MLE
- This validates the Stage 4 sumcheck is fundamentally working for Instance 0

#### 2. The Problem: Batched Claim Mismatch
```
batched_claim (sumcheck output) = { 13, 174, 120, 9, 233, 120, ... }
total_expected (from openings) = { 18, 61, 142, 143, 28, 54, ... }
Do they match? false
```

Instance 0's contribution is correct, but the total doesn't match.

#### 3. Synthetic Termination Write Discovery
- Fibonacci has NO actual RAM operations
- But Zolt injects a "synthetic termination write" at address `0x7fffc008`
- This write is at cycle 54, setting value from 0 to 1
- RWC prover uses `start_address=0x7fff8000` and SEES this write
- Committed RamInc uses `start_address=0x80000000` and SKIPS it
- ValEvaluation/ValFinal provers use `start_address=0x7fff8000` and SEE it

#### 4. ValEvaluation Polynomial Behavior Anomaly
- At all active rounds, `val_eval_evals[1] = 0`
- With only one nonzero entry at cycle 54, this is only expected for some rounds
- 54 = 0b00110110, so after round 0, index becomes 27 (odd)
- At round 1, index 27 should be in UPPER half → `evals[1] ≠ 0`
- But debug shows `evals[1] = 0` → suggests folding issue or zero `wa/lt` values

### Root Cause Hypothesis

The ValEvaluation prover's `inc_evals` array may not be correctly tracking the termination write through the folding process. Possible issues:
1. The `wa_evals[54]` or `lt_evals[54]` are zero, causing product to be zero
2. The folding logic has a bug for sparse polynomials
3. There's a mismatch between address-indexed and cycle-indexed arrays

### Next Steps

1. Add debug output to trace `inc_evals`, `wa_evals`, `lt_evals` at cycle 54
2. Verify that `wa_evals[54] = eq(r_address, 2049)` is nonzero
3. Verify that `lt_evals[54] = LT(54, r_cycle)` is nonzero
4. If both are nonzero, trace through folding to see where value goes

---

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
