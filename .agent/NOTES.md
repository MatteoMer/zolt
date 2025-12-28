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

UniSkip passes but batched sumcheck fails. See analysis below.

---

## Stage 1 Sumcheck Verification Investigation

### Root Cause Analysis

The issue is in how Jolt's `BatchedSumcheck::verify` works:

1. **Input claim** is retrieved from the opening accumulator:
   - `input_claim = accumulator.get_virtual_polynomial_opening(UnivariateSkip, SpartanOuter)`
   - This should be the UniSkip polynomial evaluation at challenge r0

2. **Verification loop**:
   - claim = 0 (our stored value)
   - For each round, computes `e = compressed_poly.eval_from_hint(&e, &r_i)`
   - With all-zero polynomials and hint=0, output_claim = 0

3. **Expected output claim**:
   - Calls `sumcheck.cache_openings(...)` which modifies the accumulator
   - Then `sumcheck.expected_output_claim(...)` computes from R1CS input evaluations
   - Uses `inner_sum_prod = A(rx)*z(rx) * B(rx)*z(rx)`
   - For satisfied constraints, this should be 0

**The Problem**: The `cache_openings` call modifies the opening accumulator by calling
`append_virtual` for each R1CS input. This expects the proof to contain the claimed
evaluations that match what the prover would have computed.

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
