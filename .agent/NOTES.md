# Zolt-Jolt Cross-Verification Progress

## Session 25 Summary - Factor Claim Mismatch Analysis (2026-01-08)

### Key Discovery

The Stage 2 sumcheck fails because **factor claims don't match** between Zolt and Jolt.

### Factor Claims Comparison (Fibonacci test)

| Factor | Description | Match | Difference |
|--------|-------------|-------|------------|
| 0 | LeftInstructionInput | ✓ Exact match | 0 |
| 1 | RightInstructionInput | ✓ Exact match | 0 |
| 2 | IsRdNotZero | ✗ | ~4194304 (2^22) |
| 3 | WriteLookupOutputToRD | ✗ | significant |
| 4 | Jump | ✓ Exact match | 0 |
| 5 | LookupOutput | ✗ | significant |
| 6 | Branch | ✗ | significant |
| 7 | NextIsNoop | ✗ | ~129 |

### Observations

1. **Factors 0, 1, 4 match exactly** - These are computed correctly:
   - LeftInstructionInput: MLE evaluation works
   - RightInstructionInput: MLE evaluation works
   - Jump flag: Circuit flag computation works

2. **Factors 2, 3, 5, 6, 7 don't match** - Small-ish differences suggest:
   - The witness values themselves are different for some cycles
   - NOT an endianness issue (factors 0, 1, 4 would fail too)
   - NOT an MLE computation issue (same algorithm used)

### Root Cause Hypothesis

The witness values for these flags differ between Zolt and Jolt:
- `IsRdNotZero` - How Zolt determines rd != 0
- `WriteLookupOutputToRD` - How Zolt sets this circuit flag
- `LookupOutput` - How Zolt computes lookup results
- `Branch` - How Zolt determines branch flag
- `NextIsNoop` - How Zolt determines if next instruction is noop

### How Jolt Computes These Values

From `ProductCycleInputs::from_trace`:
```rust
// is_rd_not_zero: instruction_flags[InstructionFlags::IsRdNotZero]
// write_lookup_output_to_rd_flag: flags_view[CircuitFlags::WriteLookupOutputToRD]
// lookup_output: LookupQuery::to_lookup_output(cycle)
// branch_flag: instruction_flags[InstructionFlags::Branch]
// not_next_noop: !trace[t+1].instruction_flags()[InstructionFlags::IsNoop] (or false for last)
```

### Next Steps

1. Debug witness generation in Zolt's `R1CSCycleInputs::fromTraceStep`
2. Verify each flag is set consistently with Jolt's semantics
3. Compare raw trace data to ensure instruction parsing matches

### Files to Investigate
- `src/zkvm/r1cs/constraints.zig`: `R1CSCycleInputs.fromTraceStep()`, `setFlagsFromInstruction()`
- `src/zkvm/proof_converter.zig`: `computeProductFactorEvaluations()`

### Root Cause Identified

**Zolt doesn't track virtual instruction sequences.**

In Jolt, complex instructions are expanded into virtual steps:
- `CircuitFlags::VirtualInstruction = true` for virtual steps
- Each virtual step has specific flag values based on the instruction type
- `IsNoop` flag is set based on the instruction type, not just opcode

Zolt's `setFlagsFromInstruction` only knows the opcode byte, not:
- Whether this is a virtual instruction step
- Whether this is a compressed instruction
- Whether this is the first in a sequence
- The specific instruction type (AND vs OR vs XOR, etc.)

This causes incorrect values for:
- Factor 2 (IsRdNotZero): Virtual steps may have different rd handling
- Factor 3 (WriteLookupOutputToRD): Each instruction type sets this differently
- Factor 5 (LookupOutput): Depends on instruction-specific computation
- Factor 6 (Branch): Needs InstructionFlags::Branch, not just opcode
- Factor 7 (NextIsNoop): Needs IsNoop flag on instruction, not just opcode check

### Fix Required

1. **Enhance Zolt's trace format** to include:
   - Virtual instruction sequence flag
   - Instruction type enum (not just opcode)
   - Per-instruction circuit_flags array

2. **Update `R1CSCycleInputs.fromTraceStep`** to:
   - Read flags from the trace instead of computing from opcode
   - Handle virtual instruction sequences properly
   - Compute IsNoop based on instruction type

3. **Update `computeProductFactorEvaluations`** to use the correct flag values

---

## Session 24 Summary - Deep Stage 2 Analysis

### Key Finding
Stage 2 sumcheck proof is mathematically valid - all rounds pass:
- All 26 rounds have matching c0, c2, c3 coefficients
- All 26 challenges match
- Final output_claim matches: 13123490541784894264218864301865646689101148350774762798288422615780802764028

BUT expected_output_claim differs from output_claim, indicating opening claims mismatch.

### Detailed Comparison Results

| Value | Zolt | Jolt | Match |
|-------|------|------|-------|
| gamma_rwc | 31086377837778175205123147017089894504 | Same | ✓ |
| r_address challenge bytes | Identical | Same | ✓ |
| val_final_claim | 17708184114734783145538053377514369906907256976835332190588297692773985493533 | Same | ✓ |
| All 26 round polynomials | Byte-identical | Same | ✓ |
| output_claim | 13123490541784894264218864301865646689101148350774762798288422615780802764028 | Same | ✓ |
| expected_output_claim | N/A | 13551736511186635527939534124733318337862614044088180116386301103911465144413 | **Mismatch** |

### Instance Contributions (Stage 2)
- Instance 0 (ProductVirtual): claim=19366854058847837639268755478203018132153606224021885848136854669519817243621
- Instance 1 (RAF): claim=0
- Instance 2 (RWC): claim=0
- Instance 3 (Output): claim=16569652859076173421498202873716701161554115357020607472481625342580247939354
- Instance 4 (Instruction): claim=1281769312034380278185881668539414319935475418875412314915363312819906677592

### The Paradox
Sumcheck verification should guarantee that output_claim equals the sum of polynomial evaluations at the challenge point. The opening claims are supposed to be those evaluations.

If output_claim ≠ expected_output_claim, one of these must be true:
1. The opening claims in the proof don't match what was used in prover computation
2. The verifier computes expected_output_claim differently than the prover
3. There's a normalization/endianness issue in how claims are serialized/deserialized

### Next Investigation
Need to add debug output to Jolt's ProductVirtualRemainderVerifier to see exactly which factor_evals it reads from the accumulator and compare with what Zolt sends.

---

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
