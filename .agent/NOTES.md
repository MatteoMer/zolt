# Zolt-Jolt Cross-Verification Progress

## Session 69 Summary - Stage 4 Internal Consistency Bug (2026-01-28)

### Key Finding: Internal Sumcheck Consistency Failure

The Stage 4 sumcheck internal consistency check is failing because:
```
regs_current_claim ≠ expected_output
```

Even though both use the same:
- eq polynomial value: `{ 217, 23, 239, 206, 84, 218, 209, 78, ... }`
- combined value: `{ 40, 229, 105, 197, ... }`
- expected output (eq * combined): `{ 98, 177, 18, 229, ... }` (LE)

But `regs_current_claim` (from round-by-round evaluation) gives:
`{ 47, 105, 199, 69, ... }` (LE) - DIFFERENT!

### Root Cause Hypothesis

The issue is in how `regs_current_claim` evolves through the sumcheck rounds. It's computed as:
```zig
regs_current_claim = evaluateCubicAtChallengeFromEvals(regs_evals, challenge);
```

Where `regs_evals` comes from `regs_prover.computeRoundEvals()`.

The `computeRoundPolynomialGruen` function computes round polynomials in 3 phases:
1. Phase 1: Cycle vars via Gruen eq optimization
2. Phase 2: Address vars (eq not bound)
3. Phase 3: Remaining cycle vars via merged dense eq

The error likely occurs in one of these phases - the polynomial computation doesn't correctly represent the sumcheck invariant.

### Critical Debug Output Analysis

```
[ZOLT STAGE4 FINAL DEBUG] Match? false
```

This is `batched_claim.eql(batching_coeffs[0].mul(regs_current_claim))` - they don't match.

Since RAM instances (1 and 2) have zero claims:
```
Instance 1 expected = inc*wa = { 0, 0, 0, 0, ...
Instance 2 expected = inc*wa = { 0, 0, 0, 0, ...
```

The batched_claim should equal `coeff[0] * final_claim_0`.

### Key Files for Investigation

1. `proof_converter.zig:2053-2228` - Main Stage 4 sumcheck loop
2. `stage4_gruen_prover.zig:538-555` - `computeRoundPolynomialGruen` dispatcher
3. `stage4_gruen_prover.zig:561` - `phase1ComputeMessage` (Gruen optimization)
4. `stage4_gruen_prover.zig` - `phase2ComputeMessage` and `phase3ComputeMessage`

### Blocked: Jolt Build Issues

Could not run cross-verification test due to missing OpenSSL/pkg-config packages.
System doesn't have sudo access to install dependencies.

---

## Session 68 Summary - Stage 4 Fix Applied (2026-01-28)

### Fix Applied: Removed Termination Bit Workaround

**Commit:** `cee6b7e` - Remove termination bit workaround from RWC prover

**Analysis:**
1. Jolt's initial RAM state does NOT include termination or panic bits
2. These bits are only in the final RAM state (val_final), used by OutputSumcheck
3. Zolt's workaround was incorrectly adding termination bit to val_init
4. This caused `rwc_val_claim` to include termination when it shouldn't

**Expected Result:**
For programs without RAM operations (like Fibonacci):
- `rwc_val_claim` = MLE(initial_ram) @ r_address
- `init_eval` = MLE(initial_ram) @ r_address (computed by verifier)
- `input_claim = 0` (correct for both Zolt and Jolt)

**Key Insight:**
Opening points are NOT stored in serialized proof - both prover and verifier
reconstruct them from sumcheck challenges using `normalize_opening_point`.
Zolt's proof format (only storing key + claim) is correct.

---

## Session 67 Summary - Stage 4 Input Claim Mismatch (2026-01-27)

### Critical Finding: Stage 4 Transcript Divergence

**Root Cause Identified:**
Stage 4 sumcheck verification fails because the transcript diverges. The input claims appended to the transcript by Zolt don't match what Jolt's verifier computes.

**Jolt Verifier Input Claims (from debug):**
- `instance[0]` (RegistersRWC): `46 1e 5e 08 ac 95 3a 5e ...`
- `instance[1]` (RamValEval): `e8 73 76 d5 1c c0 a2 16 ...` ← NON-ZERO
- `instance[2]` (ValFinal): `ed e5 a3 0e 38 7f ec 60 ...` ← NON-ZERO

**Zolt Prover Input Claims:**
- `instance[0]` (RegistersRWC): `46 1e 5e 08 ac 95 3a 5e ...` ✓ MATCHES
- `instance[1]` (RamValEval): `00 00 00 00 ...` ✗ SENDS ZEROS
- `instance[2]` (ValFinal): `00 00 00 00 ...` ✗ SENDS ZEROS

**Why the Mismatch:**

For RamValEvaluation, the input_claim formula is:
```
input_claim = claimed_evaluation - init_eval
```

Where:
- `claimed_evaluation` = `RamVal @ RamReadWriteChecking` (from proof opening_claims)
- `init_eval` = computed by verifier from preprocessing:
  - `untrusted_contribution + trusted_contribution + val_init_public_eval`
  - For Fibonacci with no advice: `init_eval = eval_initial_ram_mle(bytecode, _, r_address)`

Zolt computes:
```zig
const input_claim_val_eval = stage2_result.rwc_val_claim.sub(val_init_eval);
```

Where `val_init_eval = stage2_result.output_val_init_claim` which is `RamValInit @ RamOutputCheck`.

**The Problem:**
- `RamValInit @ RamOutputCheck` is evaluated at a DIFFERENT `r_address` than `RamVal @ RamReadWriteChecking`
- These come from different sumchecks with different opening points
- So `rwc_val_claim - output_val_init_claim = 0` (they're the same value)
- But Jolt's verifier recomputes `init_eval` at `r_address` from `RamVal @ RamReadWriteChecking`

**The Fix Required:**
Zolt needs to compute `init_eval` at the SAME `r_address` that the verifier will use:
1. Get `r_address` from `RamVal @ RamReadWriteChecking` opening point
2. Compute `init_eval = eval_initial_ram_mle(bytecode, r_address)` using that point
3. Use that for input_claim subtraction

For programs WITHOUT RAM operations (like Fibonacci):
- `claimed_evaluation` at any point = the initial RAM value at that point
- `init_eval` = same initial RAM value at that point (from preprocessing)
- So `input_claim = 0` is CORRECT for RamValEval and ValFinal

But the issue is Jolt's verifier might compute a different `init_eval` from preprocessing than what the prover stored.

### Round Polynomial Coefficients MATCH

Verified that Stage 4 polynomial coefficients are identical between Zolt and Jolt:
- Round 0 c0: `d5 62 57 63 00 61 a6 5a ...` ✓
- Round 0 c2: `2c 24 e0 32 3d a9 9e a0 ...` ✓
- Round 0 c3: `2e e1 3f 2d 96 7d 4c 2e ...` ✓

The polynomial coefficients are NOT the issue - it's the transcript input claims.

### Final Claims Also MATCH

All final claims from Stage 4 match between Zolt and Jolt:
- val_claim ✓
- rs1_ra_claim ✓
- rs2_ra_claim ✓
- rd_wa_claim ✓
- inc_claim ✓

---

## Previous Session Notes

(See previous session summaries below)
