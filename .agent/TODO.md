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
