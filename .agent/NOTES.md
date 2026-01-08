# Zolt-Jolt Cross-Verification Progress

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
