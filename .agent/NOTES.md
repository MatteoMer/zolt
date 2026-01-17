# Zolt-Jolt Cross-Verification Progress

## Session 36 Summary - Stage 4 Verification Failure Analysis (2026-01-17)

### Current Status
- **Stage 1: PASSES** ✓
- **Stage 2: PASSES** ✓
- **Stage 3: PASSES** ✓
- **Stage 4: FAILS** - output_claim ≠ expected_output_claim

### Stage 4 Failure Analysis

The Stage 4 sumcheck (RegistersReadWriteChecking) verification fails because:
- output_claim = 17388657012463501289028458753211654741331023344380174987495804360843521599428
- expected_output_claim = 13087017565662880187225932472750198262574881263786061171449896105154105298773

### Expected Output Claim Formula (from Jolt)
```rust
let (_, r_cycle) = r.split_at(LOG_K);  // r_cycle from Stage 4 challenges (BE)
let eq_val = EqPolynomial::mle_endian(&r_cycle, &self.params.r_cycle);  // eq(stage4, stage3)

// Claims from accumulator (from Zolt's proof)
let rd_write_value_claim = rd_wa_claim * (inc_claim + val_claim);
let rs1_value_claim = rs1_ra_claim * val_claim;
let rs2_value_claim = rs2_ra_claim * val_claim;

let combined = rd_write_value_claim + gamma * (rs1_value_claim + gamma * rs2_value_claim);
expected = eq_val * combined
```

### Key Observations

1. **r_cycle values differ (correctly)**:
   - r_cycle (from sumcheck) = Stage 4's normalized cycle challenges
   - params.r_cycle (stored) = Stage 3's normalized opening point
   These SHOULD be different - they're from different stages.

2. **eq polynomial computation**:
   - Jolt: `mle_endian(r4_be, r3_be)` pairs positionally
   - Zolt: `mle(r4_le, r3_le)` also pairs positionally
   - Since eq(a_be, b_be) = eq(a_le, b_le), these should match!

3. **Variable binding order matches**:
   - Jolt Phase 1: log_T cycle rounds (LowToHigh)
   - Jolt Phase 2: LOG_K address rounds (LowToHigh)
   - Zolt: Same order (cycle first, then address)

4. **Possible root cause**:
   - The claims (val_claim, rd_wa_claim, etc.) stored by Zolt might not match what Jolt expects
   - The polynomial structure might differ from Jolt's expectation

### Next Steps
1. Add debug to compare Zolt's final claims with Jolt's received claims
2. Verify the eq_val matches between Zolt and Jolt
3. Check if the polynomial contribution formula matches

---

## Session 35 Summary - Stage 1 Fix + Prefix-Suffix Convention (2026-01-10)

### Stage 1 Fix (SUCCESS)
Stage 1 was failing because `NextPC = 0` for last cycle broke MLE polynomial consistency.

**Root Cause:** Setting `NextPC = 0` when next is NoOp changes the MLE polynomial, causing the sumcheck claimed sum to not match actual polynomial evaluation.

**Fix Applied:**
```zig
// Even when next is NoOp, use step.next_pc to preserve MLE consistency
inputs.values[R1CSInputIndex.NextPC.toIndex()] = F.fromU64(step.next_pc);
```

The constraint `if ShouldJump => NextPC == LookupOutput` is still satisfied because:
- `ShouldJump = Jump * (1 - next_is_noop) = 0` when next is NoOp
- So the constraint is trivially satisfied regardless of NextPC value

### Prefix-Suffix r_hi/r_lo Convention Fix
Discovered that Zolt had the prefix-suffix decomposition backwards from Jolt.

**Jolt Convention:**
- r_hi = r[0..mid] (first half) → used for **SUFFIX**
- r_lo = r[mid..n] (second half) → used for **PREFIX**
- prefix_size = 2^len(r_lo), suffix_size = 2^len(r_hi)

**Zolt Had (WRONG):**
- r_hi → PREFIX
- r_lo → SUFFIX

**Fixed in:**
- `ShiftPrefixSuffixProver.init()`
- `ShiftPrefixSuffixProver.transitionToPhase2()`
- `RegistersPrefixSuffixProver.init()`

### Other Changes
- Added `unexpanded_pc` field to TraceStep (for future virtual sequence support)
- Added `is_noop` flag and `padWithNoop()` for trace padding

### Current Status
- **Stage 1: PASSES** ✓
- **Stage 2: PASSES** ✓
- **Stage 3: FAILS** - prefix-suffix fix applied but `grand_sum != input_claim`
- **Stages 4-7: Blocked**

### Commits
- `02fcac3` - fix: Stage 1 verification + NoOp padding + prefix-suffix decomposition
- `cd2df80` - chore: add build_verify.sh script

---

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
