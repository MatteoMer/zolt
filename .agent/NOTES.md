# Zolt-Jolt Cross-Verification Progress

## Session 23 Summary - Fixed Double-Prove Issue

### Major Progress

1. **Fixed double-prove bug in --jolt-format mode**
   - Previously, main.zig called `prove()` then `proveJoltCompatibleWithDoryAndSrsAtAddress()`
   - This caused memory corruption and state issues
   - Fixed by using a single code path for jolt-format

2. **Jolt-format proof generation now works**
   - Proof size: 32,966 bytes
   - Preprocessing size: 93,456 bytes
   - Stage 1 verification passes!
   - Stage 2 verification fails at expected_output_claim

### Current Stage 2 Issue

```
output_claim:          6490144552088470893406121612867210580460735058165315075507596046977766530265
expected_output_claim: 21082316018007705420574862052777536378229816237024856529615376176100426073034
```

Stage 2 batches 5 instances:
1. ProductVirtualRemainderVerifier
2. RamRafEvaluation
3. RamReadWriteChecking
4. OutputCheck
5. InstructionClaimReduction

Each instance's `expected_output_claim` is computed from opening claims stored in the accumulator.

### ProductVirtualRemainderVerifier expected_output_claim

This computes:
```
tau_high_bound_r0 * tau_bound_r_tail_reversed * fused_left * fused_right
```

Where:
- `tau_high_bound_r0` = Lagrange kernel of tau_high at r0
- `tau_bound_r_tail_reversed` = eq(tau_low, r_tail_reversed)
- `fused_left` = w[0]*l_inst + w[1]*is_rd_not_zero + w[2]*is_rd_not_zero + w[3]*lookup_out + w[4]*j_flag
- `fused_right` = w[0]*r_inst + w[1]*wl_flag + w[2]*j_flag + w[3]*branch_flag + w[4]*(1-next_is_noop)

The 8 factor claims come from Zolt's `factor_evals` array:
- [0] LeftInstructionInput
- [1] RightInstructionInput
- [2] IsRdNotZero
- [3] WriteLookupOutputToRDFlag
- [4] JumpFlag
- [5] LookupOutput
- [6] BranchFlag
- [7] NextIsNoop

### Potential Issues

1. **Opening point mismatch**: The factor_evals are computed at a challenge point from Stage 2. The exact normalization (endianness, reversal) must match Jolt.

2. **R1CS input index mismatch**: Zolt's `R1CSInputIndex` enum might not map to the same witness positions as Jolt's.

3. **Instance ordering**: The 5 instances must be in the exact same order.

4. **Claim batching**: The scaling by 2^(max_rounds - instance_rounds) must be correct.

### Files Changed
- `src/main.zig` - Fixed double-prove issue

### Next Steps

1. Add debug output to compare per-instance expected_output_claim between Zolt and Jolt
2. Verify r_cycle normalization matches
3. Check Instance 0's expected_output_claim in isolation
4. Verify batching coefficient computation matches

---

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

---

## Technical References

- Jolt ProductVirtual: `jolt-core/src/zkvm/spartan/product.rs`
- Jolt BatchedSumcheck: `jolt-core/src/subprotocols/sumcheck.rs`
- Zolt Stage 2 prover: `src/zkvm/proof_converter.zig:generateStage2BatchedSumcheckProof`
- Zolt split_eq: `src/poly/split_eq.zig`
