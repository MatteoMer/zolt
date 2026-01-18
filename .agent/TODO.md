# Zolt-Jolt Compatibility TODO

## Current Progress

| Stage | Internal (Zolt) | Cross-verify (Jolt) | Notes |
|-------|-----------------|---------------------|-------|
| 1 | ✅ PASS | ✅ PASS | Fixed MontU128Challenge |
| 2 | ✅ PASS | ✅ PASS | - |
| 3 | ✅ PASS | ✅ PASS | - |
| 4 | ✅ PASS | ❌ FAIL | Sumcheck challenges diverge |
| 5 | ✅ PASS | - | Blocked by Stage 4 |
| 6 | ✅ PASS | - | Blocked by Stage 4 |

## Session 47 Progress (2026-01-18)

### Latest Findings (Updated)

**FIXED - Transcript Synchronization:**
- gamma_stage4 bytes match between Zolt and Jolt ✓
- input_claim_registers bytes match exactly ✓
- input_claim_val_eval now correctly derived from accumulator (not polynomial sum) ✓
- input_claim_val_final now correctly derived from accumulator ✓
- Padding fix applied: val_poly for padding cycles now filled ✓
- Transcript state after input claims: `[75, 0f, 4a, 12, 44, 5c, d0, 24]` matches ✓
- Batching coefficients now match ✓

**REMAINING ISSUE - Round Polynomial Coefficients:**
Jolt Stage 4 Round 0 coefficients (LE bytes):
- c0_bytes = [175, 203, 5, 251, 28, 188, 99, 10, ...]
- c2_bytes = [229, 122, 188, 188, 236, 68, 132, 131, ...]
- c3_bytes = [151, 106, 90, 172, 115, 202, 59, 141, ...]

Zolt Stage 4 Round 0 coefficients:
- c0 = { 251, 39, 236, 197, 111, 171, 135, 176, ... }
- c2 = { 196, 91, 223, 7, 40, 209, 118, 126, ... }
- c3 = { 237, 164, 171, 42, 199, 9, 20, 191, ... }

These are COMPLETELY different. The fundamental issue is in how
RegistersReadWriteChecking computes its round polynomials.

**Root Cause:**
Zolt's Stage 4 prover uses dense iteration, while Jolt uses:
1. GruenSplitEqPolynomial with LowToHigh binding
2. ReadWriteMatrixCycleMajor sparse representation
3. gruen_poly_deg_3 conversion from quadratic coefficients to cubic evaluations

The mathematical polynomial should be the same, but the computation differs.

**Next Steps:**
1. Compare Round 0 computation step-by-step between Zolt and Jolt
2. Verify eq polynomial values match at cycle indices
3. Check if iteration order (k, j) matches
4. Consider porting Jolt's sparse+Gruen algorithm to Zolt

---

### Deep Code Analysis

**Jolt Stage 4 Prover Structure:**
1. Phase 1: Binds all cycle variables (log_T rounds) using:
   - `GruenSplitEqPolynomial` (splits eq into E_out * E_in)
   - `ReadWriteMatrixCycleMajor` (sparse matrix representation)
   - `prover_message_contribution` computes **quadratic coefficients** [q(0), q_quadratic]
   - `gruen_poly_deg_3` converts to cubic polynomial evaluations

2. Phase 2: Binds all address variables (LOG_K rounds) using:
   - `ReadWriteMatrixAddressMajor`
   - Dense merged_eq polynomial

3. Phase 3: Dense computation for any remaining variables

**Key Insight: Coefficient vs Evaluation Representation**
Jolt's `compute_evals` returns `[eval_0, eval_infty]`:
- `eval_0` = contribution at X=0
- `eval_infty` = coefficient of X in the linear polynomial (the "slope")

This is NOT `[p(0), p(1)]` but rather polynomial coefficients!

The `gruen_poly_deg_3` then converts these quadratic coefficients to actual evaluations.

**Zolt's Approach:**
1. Computes actual evaluations at X=0, 1, 2, 3 directly
2. Then interpolates to get coefficients

**The Problem:**
The final polynomial should be mathematically the same, but Jolt's sparse matrix handles missing entries differently:
- Missing entries have implicit ra=0, wa=0
- Val is inferred from `prev_val` and `next_val` of adjacent entries

Zolt's dense representation:
- Explicitly stores all entries
- Zero entries still contribute to the sum (but with 0 contribution)

### Next Steps
1. Run side-by-side comparison of Round 0 computation
2. Verify eq polynomial values match between Zolt and Jolt
3. Check if sparse vs dense handling causes the difference
4. Consider porting Gruen optimization to Zolt

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

## Previous Sessions Summary
- **Session 47**: Deep analysis of Jolt's Gruen optimization and sparse matrix
- **Session 46**: Confirmed values match but round poly coefficients differ
- **Session 45**: Fixed Stage 4 polynomial sums, identified input_claim mismatch
- **Session 44**: Fixed eq polynomial analysis, identified index reversal issue
- **Session 43**: Fixed Stage 1 MontU128Challenge conversion
