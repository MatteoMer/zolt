# Zolt-Jolt Compatibility - Current Status

## Summary (Session 30 - Updated)

**Stage 1**: PASSES ✓
**Stage 2**: PASSES ✓
**Stage 3**: FAILS - Round polynomial computation incorrect

### Latest Finding: Input Claims Match!

All Stage 3 input claims are **CORRECT** and match Jolt:

| Claim | Zolt (first 8 bytes BE) | Jolt (first 8 bytes BE) | Match |
|-------|-------------------------|-------------------------|-------|
| shift_input_claim | matches | matches | ✓ |
| instr_input_claim | matches | matches | ✓ |
| reg_input_claim | matches | matches | ✓ |

The individual Next* claims also match:
- NextUnexpandedPC = 5016914920442655063139027353295106901665615638715450801907420320438791241677 ✓
- NextPC = same as above ✓
- NextIsVirtual = 0 ✓
- NextIsFirstInSequence = 0 ✓
- NextIsNoop = 14175110745294312468493177356540255929141240160643613108653122477912496566260 ✓

Gamma powers also match:
- gamma_powers[0] = 1 ✓
- gamma_powers[1] = 167342415292111346589945515279189495473 ✓

### ROOT CAUSE

The Stage 3 sumcheck verification fails because the **round polynomials** are generating an incorrect final `output_claim`:

- `output_claim` (from round polys) = 1673574733889313935270617916743060218503432297743197831652540070081185994486
- `expected_output_claim` (from opening claims) = 21327743636063891625108510123531019119449360721408058830639529124915741467777

The round polynomial computation in `stage3_prover.zig` is producing wrong evaluations.

### Verified So Far

1. ✓ Input claims are correct (NextUnexpandedPC, NextPC, etc.)
2. ✓ Gamma powers are correct
3. ✓ Binding formula is correct: `new[i] = (1-r)*old[2i] + r*old[2i+1]`
4. ✓ Round polynomial approach: compute p(0), p(2), derive p(1) from claim
5. ✓ Coefficient interpolation formula (finite differences for degree 3)

### Investigation Needed

1. **eq+1 polynomial evaluations**: Are they computed correctly at each index?
   - Current: Using `EqPlusOnePolynomial.mle(r, j_bits)` for each j
   - Jolt: Uses `EqPlusOnePrefixSuffixPoly` with prefix-suffix optimization
   - These should be equivalent mathematically

2. **MLE values from trace**: Are the witness values extracted correctly?
   - ShiftMLEs: unexpanded_pc, pc, is_virtual, is_first_in_sequence, is_noop
   - InstructionInputMLEs: left_is_rs1, rs1_value, etc.
   - RegistersMLEs: rd_write_value, rs1_value, rs2_value

3. **Product formula**: Is the round polynomial formula correct?
   - Shift: eq+1_outer * val + gamma^4 * (1-noop) * eq+1_product
   - Instr: (eq_outer + gamma^2 * eq_product) * (right + gamma * left)
   - Reg: eq * (rd + gamma*rs1 + gamma^2*rs2)

### Next Steps

1. Add debug output for round 0 values to compare with Jolt
2. Print eq+1 evaluations at indices 0, 1, 2, 3 for first round
3. Print MLE values at same indices
4. Compare computed p(0), p(2) with expected

### Key Files

- `/Users/matteo/projects/zolt/src/zkvm/spartan/stage3_prover.zig` - Round polynomial generation
- `/Users/matteo/projects/jolt/jolt-core/src/zkvm/spartan/shift.rs` - ShiftSumcheck implementation
- `/Users/matteo/projects/jolt/jolt-core/src/zkvm/spartan/instruction_input.rs` - InstructionInputSumcheck
- `/Users/matteo/projects/jolt/jolt-core/src/zkvm/claim_reductions/registers.rs` - RegistersClaimReduction
- `/Users/matteo/projects/jolt/jolt-core/src/poly/eq_plus_one_poly.rs` - eq+1 polynomial
