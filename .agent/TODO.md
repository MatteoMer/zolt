# Zolt-Jolt Compatibility - Current Status

## Summary (Session 30 - Updated)

**Stage 1**: PASSES ✓
**Stage 2**: PASSES ✓
**Stage 3**: FAILS - Round polynomial computation incorrect

### Latest Finding: Input Claims Match!

All Stage 3 input claims are **CORRECT** and match Jolt:

| Claim | Zolt (first 8 bytes BE) | Jolt (first 8 bytes BE) | Match |
|-------|-------------------------|-------------------------|-------|
| shift_input_claim | [37, 74, 168, ...] | [37, 74, 168, ...] | ✓ |
| instr_input_claim | [26, 147, 254, ...] | [26, 147, 254, ...] | ✓ |
| reg_input_claim | [11, 151, 170, ...] | [11, 151, 170, ...] | ✓ |

The individual Next* claims also match:
- NextUnexpandedPC = 5016914920442655063139027353295106901665615638715450801907420320438791241677 ✓
- NextPC = 5016914920442655063139027353295106901665615638715450801907420320438791241677 ✓
- NextIsVirtual = 0 ✓
- NextIsFirstInSequence = 0 ✓
- NextIsNoop = 14175110745294312468493177356540255929141240160643613108653122477912496566260 ✓

### ROOT CAUSE

The Stage 3 sumcheck verification fails because the **round polynomials** are generating an incorrect final `output_claim`:

- `output_claim` (from round polys) = 9239339117021878410508390355265032343017124036393535225297324005737789124360
- `expected_output_claim` (from opening claims) = 17717112342689802773312615643940766507739260162469446821836707546402020189794

The round polynomial computation in `stage3_prover.zig` is not correctly implementing the batched sumcheck polynomial function.

### Next Steps

1. [ ] Debug Stage 3 round polynomial computation
2. [ ] Verify eq/eq+1 polynomial evaluations
3. [ ] Verify MLE evaluations from cycle_witnesses
4. [ ] Check round polynomial formula matches Jolt's ShiftSumcheck, InstructionInputSumcheck, RegistersClaimReduction

### Key Files

- `/Users/matteo/projects/zolt/src/zkvm/spartan/stage3_prover.zig` - Round polynomial generation
- `/Users/matteo/projects/jolt/jolt-core/src/zkvm/spartan/shift.rs` - ShiftSumcheck implementation
- `/Users/matteo/projects/jolt/jolt-core/src/zkvm/spartan/instruction_input.rs` - InstructionInputSumcheck
