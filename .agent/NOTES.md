# Zolt-Jolt Cross-Verification Progress

## Session 66 Summary - SumcheckId Fix (2026-01-27)

### Critical Bug Fixed: Missing SumcheckId Variants

**Root Cause:**
Zolt's SumcheckId enum had 22 variants, but Jolt has 24. The missing variants were:
- `AdviceClaimReductionCyclePhase` (id=20)
- `AdviceClaimReduction` (id=21)

This caused all OpeningId encoding bases to be off by 2:
- Old: TRUSTED_ADVICE_BASE=22, COMMITTED_BASE=44, VIRTUAL_BASE=66
- New: TRUSTED_ADVICE_BASE=24, COMMITTED_BASE=48, VIRTUAL_BASE=72

**Impact:**
Every Committed and Virtual claim in the proof was serialized with the wrong fused byte,
causing Jolt deserializer to misinterpret the claims entirely.

**Fix:** Commit 280c687 - Added missing SumcheckId variants

### Proof Structure Verification

After the fix, proof structure parses correctly:
- 91 claims (3127 bytes)
- 37 commitments (384 bytes each = 14208 bytes)
- Stage 1 UniSkip: 28 coeffs
- Stage 1-7 sumcheck proofs with correct round counts

### Remaining Work

Need to run Jolt verifier to confirm the fix resolves Stage 4 mismatch.
Currently blocked by missing OpenSSL/pkg-config for Jolt compilation.

---

## Session 52 Summary - Debug Output Working (2026-01-24)

### Key Finding: Debug Output Now Showing

Fixed the issue with missing debug output - the SRS file was missing! When running without `--srs`:
```
zig build run -- prove examples/fibonacci.elf --jolt-format -o /tmp/zolt_proof.bin
```

All debug output now appears:

**Zolt Stage 4 Round 0 (q_0, q_X2):**
```
q_0 = { 142, 134, 24, 23, 198, 184, 119, 182, ... }
q_X2 = { 142, 219, 181, 75, 210, 72, 66, 115, ... }
regs_evals[0] = { 136, 205, 113, 48, 248, 104, 6, 240 }
regs_evals[1] = { 188, 135, 97, 158, 228, 141, 236, 192 }
regs_evals[2] = { 58, 166, 205, 230, 72, 241, 230, 251 }
regs_evals[3] = { 55, 198, 84, 181, 234, 185, 133, 134 }
```

**E_out/E_in Table Values (Round 0):**
```
E_out[0] = { 71, 160, 211, 72, 66, 155, 28, 51, ... }
E_out[1] = { 41, 243, 104, 67, 21, 89, 78, 158, ... }
E_in[0] = { 212, 155, 232, 122, 38, 129, 172, 11, ... }
E_in[1] = { 146, 223, 209, 85, 252, 189, 104, 204, ... }
```

**First Contribution (k=2, j_pair=(0,1)):**
```
EVEN: ra={ 0, 0, 0, 0, 0, 0, 0, 0 }, wa={ 1, 0, 0, 0, 0, 0, 0, 0 }, val={ 0, 0, 0, 0, 0, 0, 0, 0 }
ODD:  ra={ 165, 220, 86, 216, 147, 169, 75, 108 }, wa={ 1, 0, 0, 0, 0, 0, 0, 0 }, val={ 0, 128, 0, 0, 0, 0, 0, 0 }
inc_0={ 0, 128, 0, 0, 0, 0, 0, 0 }, inc_slope={ 2, 128, 255, 239, 147, 245, 225, 67 }
```

### Cross-Verification Test Results

Successfully ran Jolt verifier on Zolt-generated proof:

**Command:**
```bash
zig build run -- prove examples/fibonacci.elf --jolt-format \
  --export-preprocessing /tmp/zolt_preprocessing.bin \
  -o /tmp/zolt_proof_dory.bin

cd /Users/matteo/projects/jolt/jolt-core && \
  cargo test test_verify_zolt_proof_with_zolt_preprocessing --release -- --nocapture --ignored
```

**Results:**
- ✅ Stage 1 (Outer Spartan): PASS
- ✅ Stage 2 (Lasso lookups): PASS
- ✅ Stage 3 (Register claims): PASS
- ❌ Stage 4 (Register RWC): FAIL

**Failure Analysis:**
```
output_claim:          3110039466046447483900250223050551234127534443712699458605485358576992771996
expected_output_claim: 1714094508039949549364454035354307069625501160663719062284025310540203859155
```

The verification fails because:
1. Zolt's Stage 4 round polynomial coefficients differ from Jolt's
2. Different coefficients → different Fiat-Shamir challenges
3. r_cycle from sumcheck: `[a1, d3, 18, 73, ...]` (actual)
4. r_cycle from params: `[7e, 54, 44, cd, ...]` (expected)
5. Since r_cycle differs, `eq(r_expected, r_actual) * combined != expected`

**Zolt's Round 0 Coefficients:**
```
coeffs[0] = { 167, 2, 138, 133, 197, 33, 242, 168, ... }
coeffs[1] = { 213, 211, 179, 236, 114, 215, 98, 219, ... }
coeffs[2] = { 28, 7, 243, 236, 36, 154, 133, 63, ... }
```

These must match Jolt's coefficients exactly for the proof to verify.

### Next Steps

1. **Compare with Jolt native prover** - Run Jolt's fibonacci example with debug output to see their Round 0 coefficients
2. **Step-by-step comparison** - Compare q_0 and q_X2 accumulation between Zolt and Jolt
3. **Investigate val_poly construction** - Verify that register value tracking matches Jolt's implementation
4. **Check E_out/E_in computation** - Ensure GruenSplitEqPolynomial produces identical table values
5. **Test with minimal trace** - Create 2-cycle test case to isolate the divergence

The key insight is that while all INPUTS to Stage 4 match (gamma, r_cycle, claims), the polynomial computation itself diverges. This points to either:
- Incorrect polynomial value construction (val_poly, ra_poly, wa_poly)
- Different E_out/E_in table values from GruenSplitEqPolynomial
- Accumulation bugs in q_0/q_X2 computation
- Subtle difference in how sparse entries are handled

---

## Session 49 Summary - Stage 4 Deep Investigation (2026-01-18)

### Key Finding: All Inputs Match, But Round Polynomials Differ

After extensive debugging, verified that ALL inputs to Stage 4 prover match between Zolt and Jolt:

1. **r_cycle_be matches params.r_cycle byte-for-byte**:
   - r_cycle_be[0] = [80, fc, d1, 50, 52, 28, 5d, 74, ...] ✓
   - r_cycle_be[7] = [91, ed, 3b, 94, 29, 37, 61, e9, ...] ✓

2. **Input claim matches**:
   - First 8 bytes: [33, 72, a2, ea, 5e, ba, ea, a4] ✓

3. **Opcode-based register handling matches Jolt's per-instruction logic**:
   - RS1 reads: 0x13, 0x03, 0x67, 0x1b, 0x33, 0x3b, 0x23, 0x63 ✓
   - RS2 reads: 0x33, 0x3b, 0x23, 0x63 ✓
   - RD writes: all except 0x23 (Store), 0x63 (Branch) ✓

4. **Formula verification**:
   - c_0 = ra_0*val_0 + wa_0*(val_0+inc_0) ✓
   - c_X2 = ra_slope*val_slope + wa_slope*(val_slope+inc_slope) ✓

### The Mystery

Despite all inputs being correct:
- Zolt's internal check passes: p(0) + p(1) = current_claim ✓
- But output_claim (19271728...) ≠ expected_output_claim (5465056...)

The round polynomial coefficients are DIFFERENT, producing different challenges.

### Remaining Hypotheses

1. **E_in/E_out table values**: Tables might be computed with different values
2. **Trace interpretation**: val_poly might have different actual values
3. **Accumulation precision**: Jolt uses unreduced Montgomery, Zolt uses reduced
4. **Edge case handling**: Empty sparse entries vs dense zeros

### Debug Output Added
- `[STAGE4_GRUEN_INIT]` prints all r_cycle_be values for comparison
- E_in.len, E_out.len, num_x_in_bits verification

---

## Session 48 Summary - Stage 4 Deep Debugging (2026-01-18)

### Key Finding: Both Gruen and Original Provers Fail!

Tested both implementations:
1. `stage4_gruen_prover.zig` - Gruen-optimized (matches Jolt's approach)
2. `stage4_prover.zig` - Original dense implementation

**BOTH produce the same failure pattern**, indicating the issue is NOT in the Gruen optimization itself.

### What's Working
1. **Claims match**: val_claim, rs1_ra_claim, rs2_ra_claim, rd_wa_claim, inc_claim all match between Zolt and Jolt
2. **Gamma matches**: Verified byte-by-byte
3. **r_cycle (Stage 3 challenges) matches**: Verified byte-by-byte
4. **Transcript state at Stage 4 start appears to match**: `[8d, f6, 85, 47, 1a, b6, 8d, c2]`
5. **Internal sumcheck relation satisfied**: p(0)+p(1)=claim passes for all rounds

### What's Failing
The r_sumcheck challenges derived from round polynomials don't match r_cycle:
- Jolt expected r_cycle[0]: `[80, fc, d1, 50, 52, 28, 5d, 74]`
- Actual r_sumcheck[0]: Completely different values

This causes eq(r_cycle, r_sumcheck) != 1, leading to verification failure.

### Fix Applied: inc_slope = 0
Changed `phase1ComputeMessage` in `stage4_gruen_prover.zig`:
```zig
// OLD: c_X2 = ra_slope*val_slope + wa_slope*(val_slope+inc_slope)
// NEW: c_X2 = ra_slope*val_slope + wa_slope*val_slope
```

This matches Jolt's behavior where `inc_evals = [inc_coeff, 0]` (constant, no slope).
**Result**: Still fails, so this wasn't the only issue.

### Hypotheses for Root Cause

1. **eq polynomial indexing**: The dense eq_evals might be indexed differently than expected
2. **Binding order implementation**: LowToHigh binding might have subtle differences
3. **E_out/E_in table construction**: The evalsCached function might differ from Jolt's
4. **gruenPolyDeg3 computation**: Might have subtle numerical differences

### Next Steps
1. Add step-by-step debugging to compare eq polynomial values
2. Create a minimal test case (2 cycles, 1 register) to verify formulas
3. Compare E_out_current/E_in_current return values between Zolt and Jolt
4. Verify gruenPolyDeg3 output matches Jolt's implementation exactly

---

## Session 47 Summary - Stage 4 Detailed Analysis (2026-01-18)

### Latest Findings

1. **Padding Fix Applied** (commit 35dd23b):
   - Fixed padding cycles not having final register values
   - This was causing incorrect polynomial extrapolation for pairs straddling real/padding boundary

2. **val_eval and val_final claims = 0**: This is CORRECT for fibonacci since there are no memory operations (only registers). Instance 1 and 2 having expected_claim=0 is expected.

3. **RegistersReadWriteChecking mismatch persists**:
   - output_claim = 3159763944798181886722852590115930947586131532755679042258164540994444897089
   - expected_output_claim = 4857024169349606329580068783301423991985019660972366542411131427015650777104

4. **r_cycle divergence confirmed**:
   - r_cycle from Stage 4 sumcheck (derived from coefficients) is COMPLETELY different from params.r_cycle
   - This causes eq_val mismatch, leading to verification failure

### Zolt Round 0 Output
```
p(0) = { 9, 229, 235, 172, 204, 112, 210, 123, ... } (BE)
p(1) = { 10, 128, 241, 96, 234, 213, 124, 124, ... } (BE)
p(0)+p(1) = batched_claim ✓
challenge = { 26, 202, 171, 10, 84, 123, 125, 173, ... } (BE)
```

### Jolt Expected r_cycle (from params)
```
r_cycle[0] bytes: [80, fc, d1, 50, 52, 28, 5d, 74]
r_cycle[1] bytes: [69, e1, d4, 1c, db, 43, 0f, 41]
...
```

### Jolt Received r_cycle (from Stage 4 sumcheck)
```
r_cycle[0] bytes: [c5, 35, 86, 31, 18, a9, 39, 17]
r_cycle[1] bytes: [00, f1, d2, 03, 03, 37, 7d, fd]
...
```

These don't match, confirming the round polynomial coefficients produce wrong challenges.

---

## Session 46 Summary - Stage 4 Round Polynomial Mismatch (2026-01-18)

### Verified Facts
1. **gamma matches** between Zolt and Jolt ✓
2. **input_claim matches** ✓
3. **params.r_cycle (Stage 3 challenges) matches** ✓
4. **All 5 opening claims match** (val, rs1_ra, rs2_ra, rd_wa, inc) ✓
5. **Sumcheck univariate checks pass** (internally consistent)

### The Problem
Round 0 polynomial coefficients are completely different:

**Jolt's round 0 coefficients (from native prover):**
```
coeffs[0] = [c9, bd, 24, 80, fd, 25, 4f, 66, ...]
coeffs[1] = [c4, 31, 54, 6f, f7, a8, 04, 71, ...]
coeffs[2] = [28, 07, 8a, 46, 27, 62, fd, c6, ...]
coeffs[3] = [64, ab, e8, 66, ad, 84, 4b, b8, ...]
```

**Zolt's round 0 coefficients:**
```
coeffs[0] = [77, bb, 08, ec, 88, 15, 06, 86, ...]
coeffs[1] = [a9, 39, 38, 6a, dc, db, 0f, 2a, ...]
coeffs[2] = [d6, 96, 40, 95, c4, 1f, e8, b7, ...]
```

### Root Cause: Different Prover Implementations

**Jolt's Stage 4 Prover:**
1. Uses `GruenSplitEqPolynomial` with `LowToHigh` binding
2. Uses `ReadWriteMatrixCycleMajor` sparse matrix representation
3. Computes via `prover_message_contribution` on sparse entries only
4. Applies Gruen optimization: eq = E_out * E_in

**Zolt's Stage 4 Prover:**
1. Uses `computeEqEvalsBE` dense eq polynomial
2. Uses dense arrays: `val_poly[k * T + j]`
3. Iterates over all (j, k) pairs directly
4. Now also has Gruen implementation (still fails)

### Key Structural Differences

1. **Sparse vs Dense**:
   - Jolt: Only stores non-zero entries (sparse matrix)
   - Zolt: Dense arrays with many zeros

2. **Eq Polynomial**:
   - Jolt: GruenSplitEqPolynomial (splits into E_in and E_out for efficiency)
   - Zolt: GruenSplitEqPolynomial (Zig port) or dense array of size T

3. **Round Poly Computation**:
   - Jolt: `prover_message_contribution` processes sparse entries
   - Zolt: Direct iteration over all pairs

### Why They Compute Different Polynomials

Both should represent the same polynomial:
```
P(j, k) = eq(r_cycle, j) * combined(j, k)
```

But the round polynomial coefficients differ because:
1. Jolt's Gruen optimization reorders the summation
2. Different sparse matrix processing vs dense iteration
3. Possibly different zero-handling

**This causes the sumcheck challenges to diverge**, which causes eq_val to differ, which causes the verification to fail.

### Files to Study
- Jolt sparse matrix: `jolt-core/src/subprotocols/read_write_matrix/`
- Jolt Gruen eq: `jolt-core/src/poly/split_eq_poly.rs`
- Jolt prover: `jolt-core/src/zkvm/registers/read_write_checking.rs`
- Zolt provers: `src/zkvm/spartan/stage4_prover.zig`, `stage4_gruen_prover.zig`
- Zolt Gruen eq: `src/zkvm/spartan/gruen_eq.zig`

### Potential Fixes
1. **Full port**: Continue porting Jolt's exact implementation
2. **Debug deeper**: Add step-by-step comparison to find exact divergence
3. **Minimal test case**: Create simple 2-cycle test to verify formulas

---

## Session 41 Summary - Montgomery Fix & Serialization (2026-01-17)

### Major Accomplishments

1. **Fixed Stage 4 Montgomery Conversion** ✅
   - Root cause: Jolt's MontU128Challenge stores [0, 0, L, H] as BigInt representation
   - When converted to Fr: represents 2^128 * original_value (NOT the original 125-bit value)
   - OLD Zolt behavior: directly stored [0, 0, L, H] as Montgomery limbs (WRONG)
   - FIX: Store [0, 0, L, H] as standard form, then call toMontgomery()
   - Result: All 6 Zolt internal verification stages now PASS

2. **Fixed Proof Serialization Format** ✅
   - Issue: Using `--jolt` flag instead of `--jolt-format`
   - Issue: Proof was missing the claims count header
   - Result: Jolt can now deserialize Zolt proofs successfully

### Stage Status (as of Session 48)

| Stage | Internal (Zolt) | Cross-verify (Jolt) | Notes |
|-------|-----------------|---------------------|-------|
| 1 | ✅ PASS | ✅ PASS | Fixed MontU128Challenge |
| 2 | ✅ PASS | ✅ PASS | - |
| 3 | ✅ PASS | ✅ PASS | - |
| 4 | ✅ PASS | ❌ FAIL | Round poly coefficients differ |
| 5 | ✅ PASS | - | Blocked by Stage 4 |
| 6 | ✅ PASS | - | Blocked by Stage 4 |

---

## Technical References

- Jolt MontU128Challenge: `jolt-core/src/field/challenge/mont_ark_u128.rs`
- Jolt BatchedSumcheck verify: `jolt-core/src/subprotocols/sumcheck.rs:180`
- Jolt Stage 4 prover: `jolt-core/src/zkvm/registers/read_write_checking.rs`
- Jolt sparse matrix: `jolt-core/src/subprotocols/read_write_matrix/`
- Jolt Gruen eq: `jolt-core/src/poly/split_eq_poly.rs`
- Zolt Blake2b transcript: `src/transcripts/blake2b.zig`
- Zolt Stage 4 provers: `src/zkvm/spartan/stage4_prover.zig`, `stage4_gruen_prover.zig`
- Zolt Gruen eq: `src/zkvm/spartan/gruen_eq.zig`
