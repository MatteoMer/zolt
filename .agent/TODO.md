# Zolt-Jolt Compatibility - Current Status

## Summary (Session 30 - Final Update)

**Stage 1**: PASSES with Zolt preprocessing ✓
**Stage 2**: PASSES with Zolt preprocessing ✓
**Stage 3**: FAILS - Input claims don't match

### ROOT CAUSE IDENTIFIED

The Stage 3 verification fails because the **input claims are different** between Zolt and Jolt:

| Claim | Zolt (first 8 bytes BE) | Jolt (first 8 bytes BE) |
|-------|-------------------------|-------------------------|
| shift_input_claim | [37, 74, 168, 47, ...] | [38, 105, 107, 105, ...] |
| instr_input_claim | [26, 147, 254, 101, ...] | [27, 247, 250, 1, ...] |
| reg_input_claim | [11, 151, 170, 39, ...] | [31, 1, 206, 114, ...] |

### Diagnosis

1. **Transcript state is CORRECT going into Stage 3**:
   - Both Zolt and Jolt have state `25 57 93 8e d9 16 ec 86` ✓

2. **Gamma values are CORRECT**:
   - shift_gamma[1] = 167342415292111346589945515279189495473 ✓

3. **Input claims are computed from opening_claims**:
   - `shift_input_claim = gamma^0 * NextUnexpandedPC + gamma^1 * NextPC + gamma^2 * NextIsVirtual + gamma^3 * NextIsFirstInSequence + gamma^4 * (1 - NextIsNoop)`
   - These Next* values come from Stage 1's `cache_openings` (SpartanOuter)

4. **The Next* claims stored in opening_claims are WRONG**:
   - They're computed from cycle_witnesses at r_cycle
   - But the computation or storage is producing incorrect values

### Next Steps

1. [ ] Debug what values are being stored for NextUnexpandedPC, NextPC, etc. in opening_claims
2. [ ] Compare with Jolt's expected values for these claims
3. [ ] Fix the R1CS input evaluation for Next* polynomials

### Key Files

- `/Users/matteo/projects/zolt/src/zkvm/proof_converter.zig` - addSpartanOuterOpeningClaimsWithEvaluations()
- `/Users/matteo/projects/zolt/src/zkvm/spartan/stage3_prover.zig` - computeShiftInputClaim()
- `/Users/matteo/projects/zolt/src/zkvm/r1cs/constraints.zig` - R1CS witness generation
- `/Users/matteo/projects/zolt/src/zkvm/r1cs/mod.zig` - R1CSInputEvaluator

### Testing Commands

```bash
# Generate proof
./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format \
  --export-preprocessing /tmp/zolt_preprocessing.bin \
  -o /tmp/zolt_proof_dory.bin --srs /tmp/jolt_dory_srs.bin

# Test verification
cd /Users/matteo/projects/jolt/jolt-core && \
  cargo test test_verify_zolt_proof_with_zolt_preprocessing --release -- --ignored --nocapture
```
