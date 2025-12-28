# Zolt Implementation Notes

## Current Status (2024-12-28)

### What Works
- **Proof Serialization**: Byte-perfect Arkworks compatibility
- **Transcript**: Blake2b matches Jolt exactly
- **Proof Structure**: All 7 stages, correct round counts
- **SRS Loading**: arkworks format with flag bit handling
- **G1/G2 MSM**: Matches Jolt exactly
- **Dory Commitment**: GT elements match exactly
- **R1CS Constraints**: All 19 constraints satisfied (Az*Bz = 0)
- **UniSkip Polynomial**: Correctly all zeros for satisfied constraints

### Current Issue: Stage 1 Sumcheck Verification Fails

Error: "Sumcheck verification failed" in Stage 1

### Latest Analysis (December 28, 2024 Iteration 2)

**Test Output Shows**:
- UniSkip polynomial has all zero coefficients (correct for satisfied constraints)
- Round polynomials have non-zero values from streaming prover
- Opening claims have computed MLE evaluations

**Verification Equation**:
```
expected_output_claim = tau_high_bound_r0 * tau_bound_r_tail_reversed * inner_sum_prod
```

Where:
- `tau_high_bound_r0 = L(tau_high, r0)` - Lagrange kernel
- `tau_bound_r_tail_reversed = eq(tau_low, r_cycle_reversed)`
- `inner_sum_prod = Az(rx)*z(rx) * Bz(rx)*z(rx)` - computed from R1CS input evals

**The Issue**:
The streaming prover's round polynomials are computed with incorrect/placeholder logic.
Even though the sum is zero, the intermediate round polynomials must be computed correctly
using the Gruen optimization (gruen_poly_deg_3).

**What Needs Fixing**:
1. `streaming_outer.zig::computeRemainingRoundPoly()` - must use Gruen optimization
2. Challenge derivation - must use transcript consistently
3. MLE evaluation point - r_cycle must be computed from transcript challenges

---

## Previous Stage 1 Analysis

### Key Files
- `/jolt-core/src/subprotocols/sumcheck.rs:200-252` - BatchedSumcheck::verify
- `/jolt-core/src/zkvm/spartan/outer.rs:438-453` - OuterRemainingSumcheckVerifier::cache_openings
- `/jolt-core/src/zkvm/spartan/outer.rs:407-436` - expected_output_claim

### Next Steps

1. **Understand Opening Accumulator Protocol**
   - How does Jolt's prover populate the opening accumulator?
   - What values does `append_virtual` expect?
   - How are claimed evaluations verified?

2. **Match Prover Protocol Exactly**
   - The sumcheck polynomials encode specific evaluations
   - The opening claims must match what the verifier recomputes
   - Need to trace through Jolt's prover to understand the relationship

---

## R1CS Constraint Fix (Completed)

### Problem
Constraint 16 (NextUnexpPCUpdateOtherwise) was failing:
- Az = 1 (condition true: 1 - ShouldBranch - Jump = 1)
- Bz = 255 (NextUnexpandedPC != UnexpandedPC + 4)

### Root Cause
Compressed instructions (RVC) advance PC by 2, not 4. The `IsCompressed` flag wasn't
being set because we were checking the expanded instruction (always has 0x3 in low bits).

### Fix
1. Added `is_compressed` field to TraceStep
2. Emulator sets this field when fetching instruction
3. Witness generator uses trace.is_compressed instead of checking instruction bits

Result: All 19 R1CS constraints now satisfied.

---

## Previous Implementation Notes

### Dory Commitment (Complete)

Successfully matching Jolt's Dory:
- MSM over G1 points matches exactly
- Pairing to GT element works
- arkworks-compatible serialization

### Blake2b Transcript (Complete)

- 32-byte state with round counter
- Messages right-padded to 32 bytes
- Scalars serialized LE then reversed to BE
- 128-bit challenges

### Proof Structure (Complete)

7 stages matching Jolt:
- Stage 1: Outer Spartan (UniSkip + sumcheck)
- Stage 2: Product virtualization
- Stages 3-7: Various claim reductions

---

## File Locations

### SRS File
`/tmp/jolt_dory_srs.bin` - arkworks format

### Key Source Files

| File | Purpose |
|------|---------|
| `src/poly/commitment/dory.zig` | Dory commitment scheme |
| `src/zkvm/proof_converter.zig` | Proof format conversion |
| `src/zkvm/r1cs/constraints.zig` | R1CS constraint definitions |
| `src/tracer/mod.zig` | Execution trace with is_compressed |
