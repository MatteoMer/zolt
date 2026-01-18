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

## Session 46 Progress (2026-01-18)

### Deep Analysis of Stage 4 Failure

**Key Findings - Values that MATCH:**
- gamma matches between Zolt and Jolt ✓
- input_claim matches ✓
- params.r_cycle (Stage 3 challenges) matches ✓
- val_claim, rs1_ra_claim, rs2_ra_claim, rd_wa_claim, inc_claim all match ✓
- Sumcheck univariate checks pass (round polynomials are internally consistent)

**Key Finding - The Root Cause:**
The sumcheck CHALLENGES diverge between Zolt prover and Jolt verifier:
```
Zolt's r_cycle_sumcheck_le[0]: [4a, ad, 41, d4, 07, 40, 75, 65]
Jolt's r_cycle[7] (should match): [41, fa, bd, 56, c6, 07, e2, a7]
```
These are completely different values!

**Why Challenges Diverge:**
1. Challenges are derived from transcript using Fiat-Shamir
2. Transcript incorporates round polynomial coefficients
3. If Zolt's round polys differ from Jolt's expected, challenges diverge
4. The sumcheck still "passes" because each round is internally consistent
5. But the POLYNOMIAL being proven is different from what Jolt expects

**The Fundamental Difference:**
Jolt's Stage 4 prover uses:
- `GruenSplitEqPolynomial` with LowToHigh binding for eq
- `ReadWriteMatrixCycleMajor` sparse matrix for ra/wa/val
- Complex sparse ops in `prover_message_contribution`

Zolt's Stage 4 prover uses:
- Dense `computeEqEvalsBE` for eq polynomial
- Dense arrays for val_poly, rd_wa_poly, etc.
- Direct iteration over all (j, k) pairs

While mathematically equivalent, subtle differences in:
1. Sparse vs dense handling
2. Implicit zero handling
3. Indexing conventions
...cause different round polynomial coefficients.

### What Needs to Happen

To fix Stage 4, Zolt's round polynomial coefficients MUST exactly match Jolt's.
Options:
1. **Port Jolt's sparse matrix representation** (complex but correct)
2. **Find the specific computation difference** (debugging approach)
3. **Add debug output to Jolt's prover** to compare coefficient by coefficient

### Verified Matches
- gamma bytes match ✓
- input_claim bytes match ✓
- params.r_cycle bytes match ✓
- All 5 opening claims bytes match ✓

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
- Zolt prover: `/Users/matteo/projects/zolt/src/zkvm/spartan/stage4_prover.zig`

## Previous Sessions Summary
- **Session 46**: Confirmed values match but round poly coefficients differ
- **Session 45**: Fixed Stage 4 polynomial sums, identified input_claim mismatch
- **Session 44**: Fixed eq polynomial analysis, identified index reversal issue
- **Session 43**: Fixed Stage 1 MontU128Challenge conversion
