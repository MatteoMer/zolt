# Zolt-Jolt Cross-Verification Progress

## Session 34 Summary - Stage 3 Sumcheck Invariant Bug (2026-01-09)

### Issue Found
The Stage 3 round polynomials (c0, c2, c3) match Jolt exactly, but the polynomial evaluation diverges after round 0.

**Root Cause:** The sumcheck invariant `p(0) + p(1) = previous_claim` is NOT being satisfied.

### Investigation Steps
1. Discovered that the combined polynomial evaluations don't match the current_claim
2. Added debug to directly compute p(1) = Σ_j P[2j+1] * Q[2j+1]
3. Even with direct computation, `p(0) + p(1) ≠ previous_claim`

### The Real Issue
The P/Q buffer construction is fundamentally incorrect. The formula:
```
H(X) = Σ_j P[2j+X] * Q[2j+X]
```

Should satisfy `H(0) + H(1) = input_claim`. But this requires the P and Q buffers to be constructed such that:
```
Σ_j (P[2j]*Q[2j] + P[2j+1]*Q[2j+1]) = initial_input_claim
```

The issue is in how we build P and Q during initialization. We need to match Jolt's exact construction from `EqPlusOnePrefixSuffixPoly`.

### Key Insight from Jolt's Code
Looking at shift.rs:
- `p.bind()` and `q.bind()` are called but P and Q come from `EqPlusOnePrefixSuffixPoly::new(...)`
- The P buffers come from eq+1 evaluated on the prefix
- The Q buffers come from witness values weighted by suffix evaluations

The critical piece is that Jolt computes Q as:
```rust
Q[x_lo] = Σ_{x_hi} witness(x_lo, x_hi) * suffix(x_hi)
```

Where suffix is the eq polynomial evaluated over suffix variables.

### Next Steps
1. Review Jolt's EqPlusOnePrefixSuffixPoly::new() to understand P/Q construction
2. Verify our suffix polynomial evaluation matches
3. Ensure Q buffer accumulation formula is correct

---

## Session 33 Summary - Stage 3 Opening Claims Debugging (2026-01-08)

### Major Progress
- **Round polynomials now match Jolt exactly** for all 10 rounds ✓
- c0, c2, c3 coefficients are byte-for-byte identical
- Challenges match, next_claim values match

### Current Issue: Final Opening Claims Mismatch
After all sumcheck rounds, the expected_output_claim doesn't match output_claim:
- output_claim: 14932952239165959208391594932228026048117817561820246612828087537182840868903
- expected_output_claim: 5752395372260583840012327349377493683770217164924623153552748626861123188406

The expected_output_claim is computed from the witness MLE evaluations:
```
expected = Σ coeff[i] * instance_claim[i]
```

Where instance_claim[i] is the final claim for each sumcheck instance (Shift, InstructionInput, Registers).

### Implementation Changes Made
1. Added `current_witness_size` tracking to ShiftPrefixSuffixProver and RegistersPrefixSuffixProver
2. Bind witness MLEs in both Phase 1 and Phase 2
3. Update size tracking after each binding

### Remaining Investigation
The witness MLE binding produces different final values than Jolt expects. Possible causes:
1. Index ordering in initial witness storage
2. Binding order mismatch with Jolt
3. Incorrect witness values being stored

### Next Steps
1. Add debug output to compare intermediate witness MLE values
2. Verify eq(r_prefix, i) computation
3. Consider if partial evaluation approach is needed instead of direct binding

---

## Session 32 Summary - Stage 3 Prefix-Suffix Optimization Required (2026-01-08)

### Critical Discovery: Round Polynomials Are Different!

After extensive debugging, we discovered that **Zolt and Jolt produce mathematically different (but both valid) round polynomials**.

**Both polynomials satisfy the sumcheck property `p(0) + p(1) = previous_claim`**, but they are different polynomials.

### The Root Cause

Jolt's Stage 3 prover uses a **prefix-suffix decomposition optimization** (EqPlusOnePrefixSuffixPoly):
- Splits the point r into (r_hi, r_lo)
- Computes eq+1((r_hi, r_lo), (y_hi, y_lo)) = prefix_0 * suffix_0 + prefix_1 * suffix_1
- Uses this to efficiently compute round polynomials

Zolt's Stage 3 prover uses a **naive direct computation**:
- Iterates over all indices computing the sumcheck directly
- Produces valid round polynomials, but with different coefficients

### Why This Breaks Verification

1. Zolt produces round polynomial p_zolt(X) with coefficients [c0_z, c1_z, c2_z, c3_z]
2. Jolt prover would produce p_jolt(X) with coefficients [c0_j, c1_j, c2_j, c3_j]
3. Both satisfy p(0) + p(1) = claim, but c0_z ≠ c0_j, etc.
4. Since Fiat-Shamir challenges are derived from the polynomial coefficients, different coefficients → different challenges
5. Different challenges → different final output claims

### Evidence

Round 0 comparison:
- Zolt c0: 15162749667655265946555954462559066615162111224393573091137614644230810640633
- Jolt c0: 7091024619817638108434831413024896929049773929476085946006737149609972313435
- Completely different!

But both satisfy sumcheck property:
- Zolt: shift_p0 + shift_p1 = shift_claim ✓
- Zolt: combined_eval0 + combined_eval1 = current_claim ✓

### Required Fix (COMPLETED)

To achieve compatibility, Zolt **must implement the same prefix-suffix optimization** as Jolt:

1. ✓ Implement `EqPlusOnePrefixSuffixPoly` decomposition (added to poly/mod.zig)
2. ✓ Implement Phase1Prover (prefix-suffix sumcheck rounds)
   - P buffers: prefix polynomials from decomposition
   - Q buffers: accumulated sums over trace weighted by suffix
   - Round polynomial: g(X) = Σ P(2i || 2i+1)[X] * Q(2i || 2i+1)[X]
3. ✓ Implement Phase2Prover (regular sumcheck after transition)
   - Triggered when prefix_size == 2
   - Materialize full eq+1 polynomial from prefix_0_eval * suffix + prefix_1_eval * suffix
4. ✓ Match the exact computation formula and binding order

This has been implemented (500+ lines of code). Round polynomials now match!

---

## Earlier Sessions

(See previous session summaries below for detailed history)

---

## Technical References

- Jolt ProductVirtual: `jolt-core/src/zkvm/spartan/product.rs`
- Jolt BatchedSumcheck: `jolt-core/src/subprotocols/sumcheck.rs`
- Jolt ShiftSumcheck: `jolt-core/src/zkvm/spartan/shift.rs`
- Jolt Phase2Prover: `jolt-core/src/zkvm/spartan/shift.rs:558-669`
- Zolt Stage 3 prover: `src/zkvm/spartan/stage3_prover.zig`
