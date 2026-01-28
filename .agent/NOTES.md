# Zolt-Jolt Cross-Verification Progress

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
