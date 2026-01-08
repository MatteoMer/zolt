# Zolt-Jolt Cross-Verification Progress

## Session 22 Summary - Deep Investigation of expected_output_claim Mismatch

### Current Status

Stage 2 verification passes all round checks but fails at the final expected_output_claim comparison.

### What MATCHES between Zolt and Jolt:
1. Stage 1 sumcheck proof - all rounds pass ✓
2. Stage 2 initial batched_claim ✓
3. Stage 2 batching_coeffs (all 5) ✓
4. Stage 2 input_claims for all 5 instances ✓
5. Stage 2 tau_high ✓
6. Stage 2 r0 ✓
7. ALL 26 Stage 2 round coefficients (c0, c2, c3) ✓
8. ALL 26 Stage 2 challenges ✓
9. Final output_claim ✓
10. All factor claims (LeftInstructionInput, RightInstructionInput, IsRdNotZero, etc.) ✓
11. Instance 4 claims (lookup_output, left_operand, right_operand) ✓
12. fused_left and fused_right ✓
13. gamma_instr ✓
14. Stage 1 challenges (r0 through r10) - bytes match exactly ✓
15. Stage 2 challenges for rounds 16-25 - bytes match exactly ✓
16. r_spartan (Stage 1 opening point, normalized) ✓

### The Problem:
- output_claim: 6490144552088470893406121612867210580460735058165315075507596046977766530265
- expected_output_claim: 15485190143933819853706813441242742544529637182177746571977160761342770740673

### Expected contribution breakdown:
- Instance 0 (ProductVirtual): contribution=4498967682475391509859569585405531136164526664964613766755402335917970683628
- Instance 1 (RAF): contribution=0
- Instance 2 (RWC): contribution=0
- Instance 3 (Output): contribution=0
- Instance 4 (Instruction): contribution=10986222461458428343847243855837211408365110517213132805221758425424800057045

### The Paradox

The sumcheck proof is VALID (all constraints satisfied):
- s(0) + s(1) = claim at every round ✓
- All round polynomials pass degree bounds ✓
- Final output_claim matches verifier computation ✓

But expected_output_claim ≠ output_claim, meaning the polynomial evaluation differs from what the verifier expects based on the stored claims.

This should be mathematically impossible if all inputs match.

### Remaining Hypothesis

The issue may be in how Instance 4 (InstructionClaimReduction) computes expected_output_claim:

```
Eq(opening_point, r_spartan) * (lookup_output + gamma * left + gamma_sqr * right)
```

Even though all individual components match:
- opening_point = Stage 2's last 10 challenges (normalized to BE)
- r_spartan = Stage 1's cycle challenges (normalized to BE)
- claims match

There might be a subtle difference in:
1. How the Eq polynomial pairs variables (which a[i] with which b[i])
2. The normalization of opening points (the reversal logic)
3. How the claims are associated with the wrong sumcheck instance

### Key Debug Finding

Challenge byte representations match exactly between Zolt and Jolt:
- STAGE1_ROUND_1 through STAGE1_ROUND_10 bytes match
- STAGE2_ROUND_16 through STAGE2_ROUND_25 bytes match

But Jolt's Debug output for Challenge type shows internal Montgomery form representation, which is different from the actual value. This initially caused confusion but is not a real mismatch.

### Files Involved
- Zolt: src/zkvm/proof_converter.zig (Stage 2 generation, r_spartan computation)
- Jolt: jolt-core/src/zkvm/spartan/product.rs (ProductVirtualRemainder)
- Jolt: jolt-core/src/zkvm/claim_reductions/instruction_lookups.rs (InstructionClaimReduction)
- Jolt: jolt-core/src/subprotocols/sumcheck.rs (batched verification)

### Next Steps

1. Add debug output to Jolt's expected_output_claim computation for Instance 4
2. Compare the Eq(opening_point, r_spartan) value between Zolt and Jolt
3. Verify the opening_point normalization matches exactly
4. Consider that the issue might be in Instance 0, not Instance 4

---

## Session 21 Summary - evalFromHint Fix and tau_high Divergence

### Key Fix: Stage 2 Claim Update

Fixed the Stage 2 sumcheck claim evolution to use `evalFromHint` instead of Lagrange interpolation.

**Problem:** The prover was using Lagrange interpolation from `combined_evals` to compute `next_claim`, but the verifier uses `eval_from_hint` which reconstructs c1 from the hint and computes P(r) directly from coefficients.

**Root Cause:** Different evaluation sets [s0, s1, s2, s3] can produce the same compressed coefficients [c0, c2, c3] but give different Lagrange evaluations at the challenge point.

**Fix:** In `proof_converter.zig`, changed line 2127 from:
```zig
batched_claim = evaluateCubicAtChallengeFromEvals(combined_evals, challenge);
```
to:
```zig
batched_claim = evalFromHint(compressed, old_claim, challenge);
```

**Result:** Stage 2 output_claim now matches Jolt's computation exactly:
- Initial batched claim: matches
- Final output_claim: matches (11948928263400051798463901278432764058724926493141863520413443728531572654384)
- Round 16+ next_claim: matches byte-for-byte
- fused_left/fused_right: matches exactly

### Remaining Issue: tau_high for Stage 2

The expected_output_claim still differs because tau_high for Stage 2 is different:
- Zolt tau_high: 55597861199438361161714452967226452302444674035205491421209262082033450074888
- Jolt tau_high: 3964043112274501458186604711136127524303697198496731069976411879372059241338

This causes expected_output_claim mismatch because:
```
expected = tau_high_bound_r0 * tau_bound_r_tail_reversed * fused_left * fused_right * batching_coeff
```

### Root Cause of tau_high Divergence

The transcript state differs at the point of tau_high sampling (between Stage 1 completion and Stage 2 start). This is caused by:

1. **Opening claims order mismatch**: After Stage 1 completes, both Zolt and Jolt append opening claims to the transcript. The order must match exactly.

2. **Jolt's order**:
   - OuterUniSkip verifier calls cache_openings → appends UnivariateSkip claim
   - OuterRemainingSumcheck verifier calls cache_openings → appends 36 R1CS input claims

3. **Zolt's current order** (in addSpartanOuterOpeningClaimsWithEvaluations):
   - Appends 36 R1CS input claims
   - Does NOT append UnivariateSkip (claims it was already appended earlier)

The issue is that Zolt's ordering of transcript appends doesn't match Jolt's exact sequence.

### Next Steps

1. Trace the exact order of transcript appends in Jolt's verification flow
2. Match Zolt's cache_openings to append claims in the same order
3. Verify UnivariateSkip claim is appended at the correct position
4. Re-run verification

### Files Changed
- `src/zkvm/proof_converter.zig` - Fixed evalFromHint usage

---

## Session 20 Summary - Stage 1 tau Mismatch

### Critical Finding: tau Values Differ

The Stage 1 `tau` values sampled from the transcript differ between Zolt and Jolt:
- Zolt tau[0] = 759180986986426178918369937758990285438920684980950670784015002138546871782
- Jolt tau[0] = 5099858431598424328551710934302323790717024242011569457717680063482014863998

ALL 12 tau values differ! This is the root cause of Stage 1 verification failure.

### Why This Matters

The `tau` vector is sampled via Fiat-Shamir from the transcript AFTER:
1. Proof configuration values are appended
2. Polynomial commitments are appended

The transcript state before tau sampling must be identical between prover and verifier.

### Verification Flow (Expected)
1. Verifier appends proof config to transcript (trace_length, ram_K, bytecode_K, etc.)
2. Verifier appends commitments to transcript
3. Verifier samples tau = transcript.challenge_vector(num_rows_bits)
4. Verifier runs Stage 1 sumcheck verification
5. At end: output_claim should equal expected_output_claim

### What DOES Match

Despite tau mismatch, these values match:
1. **r_cycle (sumcheck challenges)** - The round polynomials produce matching challenges
2. **R1CS input claims** - The MLE evaluations at r_cycle match exactly
3. **Round polynomial coefficients (c0, c2, c3)** - All match byte-for-byte

### The Actual Problem

The mismatch chain:
1. Transcript state differs → tau values differ
2. tau values differ → eq(τ, x) polynomial differs
3. eq polynomial differs → expected_output_claim differs
4. expected_output_claim ≠ output_claim → verification fails

### What Zolt Does Before Stage 1

Looking at `convertWithTranscript`:
1. Copies commitments
2. Creates UniSkip proof from witnesses and tau
3. Generates streaming outer sumcheck proof with transcript

But the tau is RECEIVED from the caller, not sampled from transcript. The issue is that Zolt uses its own tau, but Jolt's verifier samples tau from the commitments in the proof!

### Solution

The tau vector must be sampled from the transcript by both prover and verifier:
1. Prover appends commitments to transcript
2. Prover samples tau from transcript
3. Prover uses this tau for sumcheck
4. Verifier receives proof with commitments
5. Verifier appends same commitments to its transcript
6. Verifier samples same tau from transcript
7. Tau values match!

The issue is that Zolt's prover receives tau as a parameter, but it should be sampling tau from the transcript after appending commitments.

### Next Steps

1. Fix Zolt to sample tau from transcript AFTER appending commitments
2. Ensure commitment serialization matches Jolt's format exactly
3. Re-run verification

---

## Session 19 Summary - Stage 2 Investigation

### Major Findings

1. **Fixed Instance 4 (InstructionLookupsClaimReduction) endianness bug**
   - `computeEq` in `instruction_lookups.zig` was using LITTLE ENDIAN bit indexing
   - But `r_spartan` (from `tau[0..n_cycle_vars]`) is in BIG ENDIAN format
   - Fixed by changing `x >> i` to `x >> (n - 1 - i)`

2. **All Instance 0 (ProductVirtual) components now match between Zolt and Jolt:**
   - `split_eq.current_scalar` matches `tau_high_bound_r0 * tau_bound_r_tail_reversed`
   - `fused_left` matches exactly
   - `fused_right` matches exactly
   - `left * right * eq` matches expected

3. **Current Issue: Batched sumcheck output_claim diverges**
   - Expected (computed from components): 19828484771497821494602704840470477639244539279836761038780805731500438199328
   - Zolt output_claim: 5584134810285329217002595006333176637104372627852824503579688439906349437652
   - The final polynomial values are correct, but the sumcheck claim evolution is wrong

### Debugging Progress

The individual components match:
```
Zolt left:  3020136264963051235489773022866837256495459151625256950341582263426242632602
Jolt fused_left:  3020136264963051235489773022866837256495459151625256950341582263426242632602

Zolt right: 9255024100601318676668993040097161032104347331651195994818994872862207439177
Jolt fused_right: 9255024100601318676668993040097161032104347331651195994818994872862207439177

Zolt eq:    20475033914414057635964920496637706243132929093097161521099370097473395544235
Jolt eq (L * Eq): 20475033914414057635964920496637706243132929093097161521099370097473395544235
```

But the sumcheck claim diverges, suggesting the round polynomial computation in the batched sumcheck has an issue.

### Key Insight

Instance 4 fix propagated through Fiat-Shamir:
- Before fix: Instance 4 contributed a wrong non-zero claim
- After fix: Instance 4 contributes 0 (correct)
- But the challenges changed throughout, so the verification point changed

### Suspected Root Cause

The issue is likely in how the batched sumcheck combines round polynomials from the 5 instances, specifically in `proof_converter.zig:generateStage2BatchedSumcheckProof`.

The claim update logic might not be tracking the sumcheck evolution correctly.

### Files Changed
- `src/zkvm/claim_reductions/instruction_lookups.zig` - Fixed `computeEq` endianness

### Deep Investigation Results

**Verified working:**
- Initial batched claim: `17546890048469121259959092810946017101193047322015505312565639058039619540320` ✓
- Claim at round 16 (when ProductVirtual starts): `14588867119323547528518416779260997343214321137351904230316926948679952311830` ✓
- All 26 challenges match exactly between Zolt and Jolt ✓
- tau_stage2 (r_cycle) matches Jolt's r_cycle ✓
- Polynomial evaluation at each round: `s(challenge)` computed correctly ✓
- Soundness constraint `s(0) + s(1) = claim` satisfied at every round ✓

**The paradox:**
All individual values match, yet the final claim diverges. This should be mathematically impossible.

The sumcheck produces a VALID proof (all constraints satisfied), but the final claim (5584...) doesn't match expected (19828...).

**Potential root causes to investigate:**
1. The round polynomial coefficients (c0, c1, c2, c3) might differ from what Jolt's verifier computes
2. There might be a subtle bug in how the Gruen split_eq weights are applied
3. The ProductVirtual prover's t0/t_inf computation might differ from Jolt's

### Next Steps

1. Add Jolt-side debug output to compare per-round polynomial coefficients
2. Verify the Gruen eq table construction matches Jolt's exactly
3. Compare t0/t_inf intermediate values between Zolt and Jolt
4. Consider testing with a simpler program (fewer cycles) to reduce complexity

## Technical References

- Jolt ProductVirtual: `jolt-core/src/zkvm/spartan/product.rs`
- Jolt BatchedSumcheck: `jolt-core/src/subprotocols/sumcheck.rs`
- Zolt Stage 2 prover: `src/zkvm/proof_converter.zig:generateStage2BatchedSumcheckProof`
- Zolt split_eq: `src/poly/split_eq.zig`
