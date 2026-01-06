# Zolt-Jolt Compatibility TODO

## Current Status: Session 59 - January 6, 2026

**STATUS: Sumcheck polynomials match perfectly! Opening claims mismatch.**

### Key Finding

The entire sumcheck (UniSkip + remaining 11 rounds) produces IDENTICAL polynomials and challenges:
- All 11 rounds: c0, c2, c3 coefficients match byte-for-byte
- All challenges match
- Output claim matches: `7379936223227643720496096556404058095654400826692293621824406143059361739906`
- But expected_output_claim differs: `18262841792610895119506382142558123804551187240428090173583850498893311403655`

The verifier's expected_output_claim = `lagrange_tau_r0 * tau_bound_r_tail * inner_sum_prod`

### Root Cause Analysis

- `lagrange_tau_r0` matches ✓
- `tau_bound_r_tail` matches (since challenges match) ✓
- `inner_sum_prod` does NOT match!

The verifier computes `inner_sum_prod` from `r1cs_input_evals` (the opening claims):
- Zolt r1cs_input_evals[0]: `5231928340169930126114200659211794492272793854802256682327597965133309506589`
- Jolt r1cs_input_evals[0]: `13323732181978876592732594325445545041189112629920616992481060020772236071179`

The opening claims don't match because Zolt generates R1CS witnesses from `TraceStep` while
Jolt generates them from its internal `Cycle` structure with different semantics.

### Changes Completed This Session

1. ✅ Fixed batching coefficient - removed incorrect 125-bit masking
2. ✅ Verified all sumcheck coefficients match byte-for-byte
3. ✅ Verified all challenges match

### Previous Fixes

- ✅ UniSkip polynomial passes domain sum check
- ✅ Remaining sumcheck rounds produce correct polynomials
- ✅ Batching coefficient now matches (was masked to 125 bits incorrectly)

### Next Steps

1. **Fix R1CS witness generation** - The `fromTraceStep` function needs to match Jolt's witness semantics
2. **Verify witness values match** - Add debug output to compare cycle_witnesses[0] with Jolt's values
3. **Consider trace format compatibility** - Zolt's TraceStep may not contain all fields Jolt expects
4. **Study Jolt's `LookupQuery::to_instruction_inputs`** - This is how Jolt extracts LeftInstructionInput and RightInstructionInput from cycles
5. **Consider shared trace format** - Zolt may need to use the same trace format as Jolt

### Technical Note on Why Sumcheck Matches but Opening Claims Don't

The sumcheck uses **constraint-weighted evaluations**:
```
Az(x) = Σ_constraint w[c] * constraint[c].condition(witness[x])
```

The opening claims use **raw witness values**:
```
r1cs_input_evals[i] = Σ_x eq(r, x) * witness[x].values[i]
```

These are different! The sumcheck only cares that Az*Bz = 0 (the R1CS is satisfied),
which depends on the constraint RELATIONSHIPS being correct. The opening claims
require the ABSOLUTE VALUES to match Jolt's semantics.

The sumcheck matches because:
- Zolt satisfies the same R1CS constraints as Jolt
- The witness values may differ but still satisfy Az*Bz = 0

The opening claims differ because:
- Zolt's `fromTraceStep` extracts different values than Jolt's `from_trace`
- Key differences likely in: LeftInstructionInput, RightInstructionInput, Product

---

## Technical Details

### Verification Flow

```
1. Prover generates sumcheck proof (polynomials) - CORRECT ✓
2. Prover generates opening claims (r1cs_input_evals) - MISMATCH ✗
3. Verifier uses opening claims to compute expected_output_claim - FAILS
```

### Files Involved

- `src/zkvm/r1cs/constraints.zig:fromTraceStep` - Generates R1CS witnesses
- `src/zkvm/r1cs/evaluation.zig:computeClaimedInputs` - Computes MLE at r_cycle
- `src/zkvm/proof_converter.zig:addSpartanOuterOpeningClaimsWithEvaluations` - Serializes claims

---

## Test Commands

```bash
# Build and generate proof
zig build -Doptimize=ReleaseFast
./zig-out/bin/zolt prove /tmp/jolt-guest-targets/fibonacci-guest-fib/riscv64imac-unknown-none-elf/release/fibonacci-guest --jolt-format --input-hex 32 --export-preprocessing /tmp/zolt_preprocessing.bin -o /tmp/zolt_proof_dory.bin

# Jolt verification test
cd /Users/matteo/projects/jolt
cargo test --package jolt-core test_verify_zolt_proof -- --ignored --nocapture
```

---

## Previous Sessions Summary

- Session 58: UniSkip passes domain sum check
- Session 57: Identified SECOND_GROUP missing from UniSkip
- Sessions 51-56: Various fixes (batching, round offset, transcript)
