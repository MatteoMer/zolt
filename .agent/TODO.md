# Zolt-Jolt Compatibility - Stage 2 Progress

## Current Status
- Stage 1: PASSING ✅
- Stage 2: FAILING ❌ (sumcheck output_claim mismatch)
- Stage 3+: Not reached yet

## CRITICAL PROGRESS (Session 6)

### Factor Evaluations Now Match! ✅

After analysis, confirmed that the factor evaluations (l_inst, r_inst, is_rd_not_zero, etc.)
NOW MATCH between Zolt and Jolt:
- Jolt l_inst: 2007879717122260674263092286963027290004407599772686769801689448225287510618
- Zolt l_inst: 2007879717122260674263092286963027290004407599772686769801689448225287510618

### Sumcheck Polynomials Still Wrong ❌

The Stage 2 sumcheck produces `output_claim` that doesn't match `expected_output_claim`.

**Current values:**
- `output_claim`: 6712305349781773213915836621366914034919475189156389350844584049700714314367
- `expected_output_claim`: 8321209767613183988201183581193448103173376364360119728613327078546122551176

### Root Cause Analysis

Stage 2 in Jolt is a **BATCHED** sumcheck combining multiple instances:
1. ProductVirtualRemainderVerifier (26 rounds)
2. RamRafEvaluationSumcheckVerifier
3. RamReadWriteCheckingVerifier
4. OutputSumcheckVerifier
5. InstructionLookupsClaimReductionSumcheckVerifier
6. RamRaClaimReductionSumcheckVerifier
7. RegistersValEvaluationSumcheckVerifier
8. RegistersReadWriteCheckingVerifier

Each instance has different num_rounds and input_claims. The batched polynomial is:
```
P(x) = Σ_i (coeff_i * scale_i * poly_i(x))
```

Where scale_i = 2^(max_rounds - rounds_i) for smaller instances.

### Key Problem

The challenge mismatch between prover and verifier:
- Zolt challenge[25] LE = `[a4, 59, ea, 4a, e8, 2c, e8, 55]`
- Jolt r_tail_reversed[0] = `[e1, e7, ef, 63, 65, 8e, 4c, bf]`

These should be identical if the sumcheck polynomials are correct.

## Next Steps

1. **Check batching structure** - Verify Zolt batches all Stage 2 instances correctly
2. **Compare initial claims** - Each batched instance needs correct input_claim
3. **Verify scaling factors** - 2^(max_rounds - num_rounds) must be applied
4. **Trace round polynomial computation** - Compare formula with Jolt's implementation

## Verified Components
1. ✅ Field element serialization (LE bytes match arkworks)
2. ✅ Factor evaluations (l_inst, r_inst, etc. match)
3. ✅ Transcript synchronization through Stage 1
4. ✅ tau_high sampling matches
5. ✅ Opening claims (factor evals) read correctly from proof

## Debug Commands
```bash
# Generate proof
./zig-out/bin/zolt prove --jolt-format -o /tmp/zolt_proof_dory.bin examples/fibonacci.elf 2>&1 | grep "STAGE2"

# Verify with Jolt (shows batched instances)
cd /Users/matteo/projects/jolt
cargo test --package jolt-core test_verify_zolt_proof -- --ignored --nocapture 2>&1 | grep -E "input_claim|num_rounds"
```
