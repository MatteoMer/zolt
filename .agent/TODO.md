# Zolt-Jolt Compatibility - Current Status

## Summary

**Stage 1**: PASSES (outer sumcheck) ✓
**Stage 2**: PASSES (product virtualization + RAM RAF) ✓
**Stage 3**: FAILS - Zero polynomials produce wrong output claim
**Stage 4-7**: Untested (blocked on Stage 3)

## Session 27 Progress (2026-01-08)

### Completed
1. ✓ Added EqPlusOnePolynomial to poly/mod.zig
2. ✓ Added static EqPolynomial.mle() method
3. ✓ Created stage3_prover.zig framework
4. ✓ Analyzed Stage 3 expected_output_claim formulas
5. ✓ Fixed Stage 3 prover to use R1CSInputIndex correctly
6. ✓ Documented sumcheck eval_from_hint behavior

### Key Discovery

The sumcheck verification in Jolt uses `eval_from_hint` which:
- Recovers linear term as `hint - 2*c0 - c2 - c3 - ...`
- Evaluates `p(r) = c0 + linear_term * r + c2 * r² + ...`
- For zero polynomials: `p(r) = claim * r`

This means zero polynomials don't cause round verification failure, but produce wrong final output.

### Stage 3 Architecture

Stage 3 is a batched sumcheck with 3 instances (all n_cycle_vars rounds):

1. **ShiftSumcheck** (degree 2)
   - Expected output: `Σ γ[i] * claim[i] * eq+1(r_outer, r) + γ[4] * (1-noop) * eq+1(r_prod, r)`
   - Needs EqPlusOnePolynomial for shift relations

2. **InstructionInputSumcheck** (degree 3)
   - Expected output: `(eq(r, r_stage1) + γ² * eq(r, r_stage2)) * (right + γ * left)`
   - Needs instruction flag inference from operand values

3. **RegistersClaimReduction** (degree 2)
   - Expected output: `eq(r, r_spartan) * (rd + γ * rs1 + γ² * rs2)`
   - Straightforward register value MLE evaluation

### Files Created/Modified

- `src/poly/mod.zig`: Added EqPlusOnePolynomial, EqPolynomial.mle()
- `src/zkvm/spartan/mod.zig`: Export stage3_prover
- `src/zkvm/spartan/stage3_prover.zig`: Stage 3 prover framework (incomplete)

### Commits This Session

1. fdb7698 - feat: Add EqPlusOnePolynomial and Stage 3 prover framework
2. a1a2580 - docs: Update TODO with Stage 3 architecture analysis
3. 90d4c53 - fix: Update Stage 3 prover to use R1CSInputIndex correctly
4. e13a4c6 - docs: Document sumcheck eval_from_hint behavior

## Next Steps

### Immediate (Stage 3)
1. [ ] Implement proper round polynomial computation with eq/eq+1 weighting
2. [ ] Integrate Stage 3 prover into proof_converter
3. [ ] Handle transcript flow matching Jolt exactly
4. [ ] Test Stage 3 verification

### Later (Stages 4-7)
1. [ ] Implement Stage 4: Bytecode + Instruction claim reduction
2. [ ] Implement Stage 5: RAM RAF + RAM Val evaluations
3. [ ] Implement Stage 6: Registers Val + RAM claim reductions
4. [ ] Implement Stage 7: Hamming weight claim reduction

## Testing Commands

```bash
# Build Zolt
zig build

# Run tests (714 should pass)
zig build test --summary all

# Generate proof
./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format --export-preprocessing /tmp/zolt_preprocessing.bin -o /tmp/zolt_proof_dory.bin --srs /tmp/jolt_dory_srs.bin

# Run Jolt verification
cd /Users/matteo/projects/jolt/jolt-core && cargo test test_verify_zolt_proof_with_zolt_preprocessing --release -- --ignored --nocapture
```

## Key Insights

1. **Sumcheck verification is lenient**: Uses eval_from_hint, doesn't check p(0)+p(1)=claim
2. **Zero polynomials propagate claims**: Final output = claim * Π(r_i)
3. **EqPlusOne is essential**: ShiftSumcheck needs eq+1(x,y) = 1 iff y = x+1
4. **Instruction flags matter**: InstructionInput needs left/right operand sources
