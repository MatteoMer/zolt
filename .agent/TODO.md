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

## Session 54 Progress - Deep Code Audit (2026-01-24)

### Investigation: Option 3 - Deep Code Audit

**Goal:** Line-by-line comparison of Zolt and Jolt Stage 4 implementations.

**Scope Audited:**
1. Polynomial formulas (c_0, c_X2 computation)
2. val_poly semantics
3. x_in/x_out indexing logic
4. evalsCached table building
5. Sparse vs dense handling
6. gruenPolyDeg3 conversion
7. Coefficient interpolation

**Findings:**
- ✅ ALL core implementations match exactly!
- ✅ Formulas: `c_0 = ra_even*val_even + wa_even*(val_even+inc_0)` - IDENTICAL
- ✅ Formulas: `c_X2 = ra_slope*val_slope + wa_slope*(val_slope+inc_slope)` - IDENTICAL
- ✅ val_poly stores value BEFORE cycle - MATCHES
- ✅ x_in/x_out computation: `x_in = i & x_bitmask` - MATCHES
- ✅ evalsCached loop: `curr[i] = scalar * w[k]; curr[i-1] = scalar - curr[i]` - IDENTICAL
- ✅ gruenPolyDeg3: All 60+ lines match exactly
- ✅ fromEvals (Lagrange interpolation): MATCHES

**Key Differences Found:**
1. **Sparse vs Dense**: Jolt uses sparse matrix (only touched registers), Zolt iterates all registers
   - Analysis: Should be mathematically equivalent (untouched registers contribute 0)
2. **Binding Order**: Both bind LSB-to-MSB, but Zolt reverses r_cycle array first
   - Zolt: Reverses to r_cycle_be, then binds from index n-1 to 0
   - Jolt: Uses LowToHigh binding with non-reversed array, binds from index n-1 to 0
   - Analysis: Should be equivalent

**Conclusion:**
Since ALL implementations match, the bug must be extremely subtle:
- Possible accumulation error (double-counting, wrong order)
- Possible coordinate/indexing edge case
- Possible field arithmetic precision issue

**See:** `.agent/deep_code_audit.md` for full line-by-line comparison

**Next Actions:**
Given that all code matches, we need a different approach:
1. **Add contribution-level logging**: Log each E_combined * c_0 addition to q_0
2. **Compare first 5 contributions**: Check if accumulation diverges early
3. **Check for off-by-one errors**: Verify loop bounds, index calculations
4. **Test minimal case**: Create 2-cycle trace, manually verify

---

## Session 53 Progress - Coefficient Analysis (2026-01-24)

### Investigation: Comparison Challenge

**Goal:** Compare Zolt and Jolt Stage 4 Round 0 coefficients to understand divergence.

**Findings:**
1. **Jolt fibonacci-guest coefficients captured:**
   - Test: `fib_e2e_dory` (fibonacci-guest with input 100u32)
   - Round 0 coeffs: `[31, ed, 7b, ...]`, `[54, e9, 89, ...]`, `[4a, 84, 37, ...]`
   - **Problem:** This is a DIFFERENT program from Zolt's fibonacci.elf!

2. **Zolt fibonacci.elf characteristics:**
   - Bare-metal program that computes fib(10) = 55
   - No inputs, no Jolt framework
   - Trace length: 256 (padded)
   - Cannot directly compare coefficients with Jolt's fibonacci-guest

3. **Attempted Solution:**
   - Tried to create Jolt test to prove same fibonacci.elf
   - Encountered API compatibility issues (Jolt uses host/guest framework)
   - Would need significant adaptation to support bare-metal ELF

4. **Fresh Zolt Proof Generation:**
   - Generated new proof with full debug output
   - Captured detailed Stage 4 intermediate values:
     - q_0 = `{ 142, 134, 24, 23, 198, 184, 119, 182, ... }`
     - q_X2 = `{ 142, 219, 181, 75, 210, 72, 66, 115, ... }`
     - E_out[0] = `{ 71, 160, 211, 72, 66, 155, 28, 51, ... }`
     - E_in[0] = `{ 212, 155, 232, 122, 38, 129, 172, 11, ... }`
   - Per-contribution debug shows computation details

5. **Documentation:**
   - Created `.agent/stage4_analysis.md` with comprehensive analysis
   - Documents inputs, intermediate values, hypotheses, and next steps

**Next Steps:**

**Option 1 (Recommended): Manual Verification**
- Take first contribution (k=2, j_pair=(0,1)) values from debug output
- Manually compute expected q_0/q_X2 contribution using Jolt formulas
- Compare with Zolt's computation to find divergence point

**Option 2: Create Minimal Test Case**
- Create 2-cycle fibonacci in both Zolt and Jolt
- Manually compute all expected values
- Compare step-by-step

**Option 3: Deep Code Audit**
- Line-by-line comparison of stage4_gruen_prover.zig vs read_write_checking.rs
- Focus on GruenSplitEqPolynomial and phase1_compute_message
- Check for subtle formula differences

---

## Session 52 Progress - Cross-Verification Test (2026-01-24)

### Cross-Verification Result: STAGE 4 FAILURE

Successfully generated proof and ran Jolt cross-verification test:
```bash
cd /Users/matteo/projects/jolt/jolt-core && \
  cargo test test_verify_zolt_proof_with_zolt_preprocessing --release -- --nocapture --ignored
```

**Verification Result:**
- Stages 1-3: PASS
- Stage 4: FAIL (Sumcheck verification failed)

**Root Cause Analysis:**

1. **r_cycle mismatch:**
   - Jolt r_cycle[0] from sumcheck: `[a1, d3, 18, 73, 0b, 81, f7, 4f]`
   - Jolt params.r_cycle[0] (expected): `[7e, 54, 44, cd, b8, 4f, c4, f6]`
   - These should be IDENTICAL but are completely different!

2. **eq_val != expected:**
   - eq_val = 14547277650427858... (eq evaluated at sumcheck-derived r_cycle)
   - combined = 11536200169523487...
   - expected = 8138691954662080... (from claims)
   - eq_val * combined != expected, causing verification failure

3. **Why they differ:**
   - Zolt's round polynomial coefficients are different from Jolt's
   - Different coefficients → different Fiat-Shamir challenges → different r_cycle
   - This causes eq(r_cycle_expected, r_cycle_actual) != 1

**Key Observation:**
The Zolt serialization output shows:
```
[SERIALIZATION] Stage4 Round 0 coeffs:
  coeffs[0] = { 167, 2, 138, 133, 197, 33, 242, 168, ... }
  coeffs[1] = { 213, 211, 179, 236, 114, 215, 98, 219, ... }
  coeffs[2] = { 28, 7, 243, 236, 36, 154, 133, 63, ... }
```

These coefficients generate the sumcheck challenges. Since they differ from Jolt's native prover output, the challenges diverge immediately.

### Action Plan to Fix Stage 4

**Priority 1: Compare Round Polynomial Coefficients**
1. Add debug output to Jolt's `read_write_checking.rs` to print Round 0 coefficients
2. Run Jolt fibonacci example: `cd jolt && cargo run --example fibonacci`
3. Compare Jolt's coeffs with Zolt's: `[167, 2, 138, ...]`, `[213, 211, ...]`, `[28, 7, 243, ...]`

**Priority 2: Trace Polynomial Values**
1. Add debug to print first 4 entries of val_poly, ra_poly, wa_poly in both systems
2. Verify gamma, gamma_sq values match exactly
3. Check register value tracking logic (val_poly should have value BEFORE cycle)

**Priority 3: E_out/E_in Table Comparison**
1. Compare E_out[0..4] and E_in[0..4] between Zolt and Jolt
2. Verify w_out, w_in, w_last slicing is correct
3. Check that evalsCached produces same results

**Priority 4: q_0/q_X2 Accumulation**
1. Add detailed tracing for first contribution (k=2, j=0)
2. Verify: c_0 = ra_even*val_even + wa_even*(val_even+inc_0)
3. Verify: c_X2 = ra_slope*val_slope + wa_slope*(val_slope+inc_slope)
4. Check E_combined = E_out[x_out] * E_in[x_in] matches

**Priority 5: Create Minimal Test Case**
1. Create 2-cycle fibonacci test with known expected values
2. Manually compute expected q_0 and q_X2
3. Verify step-by-step against both implementations

### Commands Used

Generate proof:
```bash
zig build run -- prove examples/fibonacci.elf --jolt-format \
  --export-preprocessing /tmp/zolt_preprocessing.bin \
  -o /tmp/zolt_proof_dory.bin
```

Cross-verify:
```bash
cd /Users/matteo/projects/jolt/jolt-core && \
  cargo test test_verify_zolt_proof_with_zolt_preprocessing --release -- --nocapture --ignored
```

---

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
