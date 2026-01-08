# Zolt-Jolt Cross-Verification Progress

## Session 28 Summary - Stage 3 Verification Testing (2026-01-08)

### Key Finding

**With Zolt preprocessing, Stages 1-2 PASS, Stage 3 FAILS**

```
Verification failed: Stage 3
output_claim:          20577841778877219275547658846017849540037880489939953429278770536770306134083
expected_output_claim: 5338855503670768469593231154982602907961044230168061695638087912869958288945
```

### Testing Commands

```bash
# Generate proof with preprocessing export
./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format \
  --export-preprocessing /tmp/zolt_preprocessing.bin \
  -o /tmp/zolt_proof_dory.bin --srs /tmp/jolt_dory_srs.bin

# Run Jolt verification with Zolt preprocessing
cd /Users/matteo/projects/jolt/jolt-core && \
  cargo test test_verify_zolt_proof_with_zolt_preprocessing --release -- --ignored --nocapture
```

### Preprocessing File Mismatch Issue

The test `test_verify_zolt_proof` (uses Jolt preprocessing) fails at Stage 1 because:
- Jolt's preprocessing has different commitment parameters than Zolt
- This causes tau values to differ, making expected_output_claim mismatch

**Solution**: Use Zolt's exported preprocessing with Jolt verifier.

### Stage 3 Prover Status

Implemented Stage 3 batched sumcheck prover with:
- ShiftMLEs, InstructionInputMLEs, RegistersMLEs structs
- eq and eq+1 polynomial evaluation tables
- Round polynomial computation for all 3 instances
- Proper transcript flow (gamma derivation, input claims, batching)

**Still failing** because round polynomials don't compute correct values.

### Next Steps

1. Debug Stage 3 round polynomial computation
2. Verify eq+1 evaluation formula is correct
3. Check MLE building from trace matches Jolt expectations
4. Verify transcript operations match Jolt exactly

### Technical Notes on Instruction Flags

InstructionInput MLEs need these flags for each cycle:
- `LeftOperandIsRs1Value`: 1 if left = rs1
- `LeftOperandIsPC`: 1 if left = PC (JAL, AUIPC)
- `RightOperandIsRs2Value`: 1 if right = rs2 (R-type instructions)
- `RightOperandIsImm`: 1 if right = imm (I-type, Load, Store, etc.)

Current approach: Compare `LeftInstructionInput` value to rs1/pc to determine flag.
This should work since `computeInstructionInputs` in constraints.zig sets these
based on opcode. But edge cases (rs1 == 0 == pc for first cycle) may cause issues.

---

## Session 27 Summary - Stage 3 Prover Implementation (2026-01-08)

### Key Discovery: Sumcheck Verification

The sumcheck verification in Jolt uses `eval_from_hint` which **doesn't explicitly check p(0) + p(1) = claim**. Instead:

```rust
linear_term = hint - 2*c0 - c2 - c3 - ...
p(r) = c0 + linear_term * r + c2 * r² + c3 * r³ + ...
```

This means:
- Zero polynomials (all coeffs = 0) don't cause immediate verification failure
- linear_term is recovered as `claim` when all coeffs are 0
- Final output claim ≈ claim * r_1 * r_2 * ... * r_n

This is why Stage 3 verification fails at the **final claim comparison**, not at round verification.

### Files Created

1. **`src/poly/mod.zig`** - Added:
   - `EqPlusOnePolynomial(F)` - eq+1(x, y) = 1 iff y = x + 1 (binary increment)
   - `EqPolynomial.mle()` - static evaluation method

2. **`src/zkvm/spartan/stage3_prover.zig`** - Stage 3 prover framework:
   - `Stage3Prover` struct with `generateStage3Proof` method
   - `ShiftMLEs`, `InstructionInputMLEs`, `RegistersMLEs` structs
   - MLE building from R1CSCycleInputs

### Stage 3 Understanding

Stage 3 is a batched sumcheck with 3 instances (all n_cycle_vars rounds):

1. **ShiftSumcheck** (degree 2)
   - Proves: `Σ eq+1(r_outer, j) * (upc + γ*pc + γ²*virt + γ³*first) + γ⁴*(1-noop) * eq+1(r_prod, j)`
   - Opening claims at SpartanShift: UnexpandedPC, PC, VirtualInstruction, IsFirstInSequence, IsNoop

2. **InstructionInputSumcheck** (degree 3)
   - Proves: `(eq(r, r_stage1) + γ²*eq(r, r_stage2)) * (right + γ*left)`
   - Where left = left_is_rs1 * rs1 + left_is_pc * upc
   - Where right = right_is_rs2 * rs2 + right_is_imm * imm
   - Opening claims at InstructionInputVirtualization: 8 flag/value pairs

3. **RegistersClaimReduction** (degree 2)
   - Proves: `eq(r, r_spartan) * (rd + γ*rs1 + γ²*rs2)`
   - Opening claims at RegistersClaimReduction: RdWriteValue, Rs1Value, Rs2Value

### Commits This Session

1. fdb7698 - feat: Add EqPlusOnePolynomial and Stage 3 prover framework
2. a1a2580 - docs: Update TODO with Stage 3 architecture analysis
3. 90d4c53 - fix: Update Stage 3 prover to use R1CSInputIndex correctly

### Current Status

| Stage | Status | Details |
|-------|--------|---------|
| 1 | ✓ PASSES | Outer sumcheck works |
| 2 | ✓ PASSES | Product virtualization works |
| 3 | ✗ FAILS | Zero polys give wrong output_claim |
| 4-7 | Blocked | Waiting on Stage 3 |

### Next Steps

1. Integrate Stage 3 prover into proof_converter
2. Implement proper round polynomial computation with eq/eq+1 weighting
3. Test Stage 3 verification
4. Implement Stages 4-7

---

## Session 26 Summary - Stage 3+ Claims and Sumcheck Verification (2026-01-08)

### Major Progress

1. **Fixed Dory Opening Proof** - Asymmetric matrix handling (sigma > nu case)
2. **Updated Jolt SRS export** - Now generates 16-variable SRS (256 G1/G2 points)
3. **Added All Opening Claims** - Stages 3-7 now have all required claims

### Current Status

| Stage | Status | Details |
|-------|--------|---------|
| 1 | ✓ PASSES | Outer sumcheck verification works |
| 2 | ✓ PASSES | Product virtualization + RAM RAF works |
| 3 | ✗ FAILS | Claims present, sumcheck verification fails |
| 4-7 | Blocked | Waiting on Stage 3 |

### Stage 3 Verification Failure Analysis

The Jolt verifier output shows:
```
output_claim:          3605979267482843492618018818811131090814373229214467976717812727899800934418
expected_output_claim: 1846872701798109175261071120538427009056470961050860597433873141898176138550
Verification failed: Stage 3
```

**Root Cause**: Zolt generates placeholder zero polynomials for stages 3-7, but the verifier computes `expected_output_claim` from the claims, which includes:

```
gamma[4] * (1 - is_noop_claim) * eq_plus_one_r_product
```

With `is_noop_claim = 0` (our zero claim):
- `(1 - 0) = 1`
- `gamma[4] * 1 * eq_plus_one_r_product ≠ 0`

This makes `expected_output_claim` non-zero, but zero polynomials produce `output_claim = 0`.

### Required Implementation

To complete compatibility, we need real sumcheck provers for:

**Stage 3** (highest priority):
1. **SpartanShift** - Shift polynomial: `f_shift(j) = f(j+1)`
2. **InstructionInputVirtualization** - Left/right operand computation
3. **RegistersClaimReduction** - Register claim reduction

**Stage 4-7** (lower priority):
- RegistersReadWriteChecking
- RamValEvaluation, RamValFinalEvaluation
- RegistersValEvaluation
- RamRaClaimReduction, RamRafEvaluation
- RamHammingBooleanity, Booleanity
- HammingWeightClaimReduction

### Commits Made This Session

1. 0bae6fc - fix: Dory opening proof for asymmetric matrix sizes
2. 15b2d47 - feat: Add all required opening claims for Jolt stages 3-7

---

## Previous Sessions Summary

(See below for detailed history)

---

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

### Progress Made (2026-01-08)

**Fixed padding cycle handling:**
- Added explicit handling for padding cycles in `computeProductFactorEvaluations`
- For padding cycles (NoOp), only Factor 7 (NextIsNoop) = 1, all others = 0
- This fixed Factor 7 mismatch

**After padding fix:**
- Factors 0, 1, 3, 4, 7 all match ✓
- Factors 2 (IsRdNotZero), 5 (LookupOutput), 6 (Branch) still differ

### Remaining Issues

The remaining differences are because Jolt's factor values depend on:
1. **Instruction type** (not just opcode) - e.g., ADD vs ADDI have different behaviors
2. **Virtual instruction sequences** - some instructions expand into multiple virtual steps
3. **Compressed instructions** - instructions may be first-in-sequence or continuations

### Fix Required

1. **Enhance Zolt's trace format** to include:
   - Instruction type enum (matching Jolt's Cycle enum variants)
   - Virtual instruction flag
   - First-in-sequence flag
   - Per-instruction computed LookupOutput value

2. **Update `R1CSCycleInputs.fromTraceStep`** to:
   - Use instruction type to compute flags correctly
   - Compute LookupOutput based on instruction semantics
   - Compute IsRdNotZero based on instruction type, not just rd field

3. **Update `computeProductFactorEvaluations`** to use the correct flag values

---

## Technical References

- Jolt ProductVirtual: `jolt-core/src/zkvm/spartan/product.rs`
- Jolt BatchedSumcheck: `jolt-core/src/subprotocols/sumcheck.rs`
- Jolt ShiftSumcheck: `jolt-core/src/zkvm/spartan/shift.rs`
- Zolt Stage 2 prover: `src/zkvm/proof_converter.zig:generateStage2BatchedSumcheckProof`
- Zolt split_eq: `src/poly/split_eq.zig`
