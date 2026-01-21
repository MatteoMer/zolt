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

## Session 51 Progress (2026-01-21)

### Added Detailed Debug Tracing for E_out/E_in Comparison

**New Debug Output Added:**
- E_out and E_in table structure (lengths, first 4 entries)
- current_scalar and current_w values
- Per-contribution x_out, x_in, E_out, E_in values
- Pair 0 (first j pair) detailed contributions for all registers

**Zolt Debug Output (Round 0):**
```
n=8, m=4
E_out.len=16, E_in.len=8
E_out[0] = { 63, 200, 23, 119, 168, 156, 128, 140, ... }
E_in[0] = { 105, 57, 186, 110, 51, 166, 185, 214, ... }

Pair 0 contributions for k=2:
  ra_even={0,0,0,0,0,0,0,0}, wa_even={1,0,0,0,0,0,0,0}, val_even={0,0,0,0,0,0,0,0}
  ra_odd={8,216,230,198,206,150,22,128}, wa_odd={1,0,0,0,0,0,0,0}, val_odd={0,128,0,0,0,0,0,0}
  c_0={0,128,0,0,0,0,0,0}, c_X2={0,0,4,108,115,99,103,75}
```

**Next Steps:**
1. Get matching Jolt debug output for same fibonacci.elf trace
2. Compare E_out/E_in table values between Zolt and Jolt
3. Identify exact point of divergence in q_0/q_X2 accumulation

---

## Session 50 Progress (2026-01-21)

### Deep Investigation of Stage 4 q_0/q_X2 Mismatch

**Verified Components (all match between Zolt and Jolt):**
- [x] input_claim_registers (converted and compared byte-by-byte)
- [x] gamma value from transcript
- [x] r_cycle_be values (all 8 values match params.r_cycle exactly)
- [x] Polynomial formulas: c_0 = ra_even*val_even + wa_even*(val_even+inc_0)
- [x] Polynomial formulas: c_X2 = ra_slope*val_slope + wa_slope*(val_slope+inc_slope)
- [x] E_in/E_out table indexing logic

**Debug Output Comparison:**

Zolt Round 0:
```
q_0 = { 208, 184, 198, 139, 236, 153, 15, 1, ... }
q_X2 = { 108, 114, 119, 247, 43, 73, 247, 11, ... }
previous_claim = { 51, 114, 162, 234, 94, 186, 234, 164, ... }
```

Jolt Round 0 (from prover):
```
q_0 = [230, 40, 53, 41, 4, 58, 165, 153, ...]
q_X2 = [160, 132, 230, 84, 30, 20, 87, 220, ...]
```

**Key Finding:**
The q_0 and q_X2 values are DIFFERENT even though all the input parameters appear to match.
This means the difference is in how the intermediate values are computed/accumulated.

**Polynomial Values (Zolt, register k=2):**
```
j=0: val=0, ra=0, wa=1
j=1: val=32768 (0x8000), ra=gamma, wa=1
j=2: val=32769 (0x8001), ra=gamma, wa=1
j=3: val=65536 (0x10000), ra=0, wa=0
```

**Remaining Suspects:**
1. GruenSplitEqPolynomial E_in/E_out table VALUES (not indexing)
2. current_scalar updates during binding
3. The w[] array initialization from r_cycle
4. Subtle difference in how sparse vs dense iteration combines contributions

**Next Investigation Steps:**
1. Add debug to compare E_out[0] and E_in[0] values with Jolt
2. Trace through the first contribution (k=2, j=0,1) in both systems
3. Verify w[] array in GruenSplitEqPolynomial matches Jolt

---

## Session 49 Progress (2026-01-21)

### Key Finding: Coefficient Comparison Shows Complete Mismatch

**Zolt Round 0 coeffs (serialized):**
```
coeffs[0] = { 167, 2, 138, 133, 197, 33, 242, 168, ... }
coeffs[1] = { 213, 211, 179, 236, 114, 215, 98, 219, ... }
coeffs[2] = { 28, 7, 243, 236, 36, 154, 133, 63, ... }
```

**Jolt Round 0 coeffs (from fibonacci example):**
```
coeffs[0] = [201, 189, 36, 128, 253, 37, 79, 102, ...]
coeffs[1] = [196, 49, 84, 111, 247, 168, 4, 113, ...]
coeffs[2] = [40, 7, 138, 70, 39, 98, 253, 198, ...]
```

The coefficients are **completely different** - not even close.

**Resolution:**
- Internal Zolt verification uses a DIFFERENT Stage 4 prover than proof_converter
- proof_converter uses Stage4GruenProver for Jolt compatibility
- The Stage4GruenProver IS producing non-zero coefficients correctly

---

## Session 48 Progress (2026-01-20)

### Deep Investigation of Stage 4 Round Polynomial Mismatch

**Verified Components (all match between Zolt and Jolt):**
- [x] gamma value from transcript
- [x] r_cycle_be passed to GruenSplitEqPolynomial
- [x] ra_poly contains correct gamma factors (gamma for rs1, gamma^2 for rs2)
- [x] E_in/E_out table computation logic
- [x] gruenPolyDeg3 formula (mathematically equivalent to Jolt's gruen_poly_deg_3)
- [x] interpolateDegree3 and evalsToCompressed formulas
- [x] Cycle-to-(x_out, x_in, x_last) decomposition

**Key Debug Findings:**
```
Jolt verifier output:
- r_cycle from sumcheck: [c1, 1d, 68, 46, ...] (derived from proof coefficients)
- params.r_cycle: [80, fc, d1, 50, ...] (expected from preprocessing)
```

The sumcheck-derived r_cycle differs from params.r_cycle because Zolt's
round polynomial coefficients produce different Fiat-Shamir challenges.

**Zolt Internal Check Passes:**
- p(0) + p(1) = batched_claim ✓ (sumcheck relation satisfied)
- But the polynomial itself is DIFFERENT from what Jolt expects

**Analysis of Individual Contributions:**
For k=2, j_pair=(0,1) at cycle 0:
- ra_even = 0 (no read), wa_even = 1 (write)
- val_even = 0, inc_0 = 32768
- c_0 = 0*0 + 1*(0+32768) = 32768 ✓

The per-entry contributions appear correct, but the accumulated
q_0 and q_X2 differ from Jolt's expected values.

**Remaining Hypotheses:**
1. E_in/E_out table indexing might differ in edge cases
2. The inc_poly handling after binding might not match Jolt's sparse iteration
3. current_T/current_K update timing might be off

**Next Investigation Steps:**
1. Add side-by-side E_out[0..4] and E_in[0..4] comparison
2. Create minimal test case with single register write
3. Compare intermediate values at each accumulation step

---

## Session 47 Progress (2026-01-18)

### Previous Findings

**FIXED - Transcript Synchronization:**
- gamma_stage4 bytes match between Zolt and Jolt ✓
- input_claim_registers bytes match exactly ✓
- input_claim_val_eval now correctly derived from accumulator (not polynomial sum) ✓
- input_claim_val_final now correctly derived from accumulator ✓
- Padding fix applied: val_poly for padding cycles now filled ✓
- Transcript state after input claims: `[75, 0f, 4a, 12, 44, 5c, d0, 24]` matches ✓
- Batching coefficients now match ✓

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
