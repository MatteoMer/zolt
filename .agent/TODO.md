# Zolt-Jolt Compatibility - Current Status

## Summary (Session 31 - Updated)

**Stage 1**: PASSES
**Stage 2**: PASSES
**Stage 3**: FAILS - Opening claims mismatch for Shift and InstructionInput

### Latest Finding: Round Polynomials ARE Correct!

All Stage 3 sumcheck round polynomials **MATCH** between Zolt and Jolt:

| Round | c0 | c2 | c3 | Challenge |
|-------|----|----|----|-----------|
| 0 | MATCH | MATCH | MATCH | MATCH |
| 1 | MATCH | MATCH | MATCH | MATCH |
| 2 | MATCH | MATCH | MATCH | MATCH |
| 3 | MATCH | MATCH | MATCH | MATCH |
| 4 | MATCH | MATCH | MATCH | MATCH |
| ... | ... | ... | ... | ... |

The sumcheck protocol is working correctly. The issue is the **opening claims**.

### ROOT CAUSE: Opening Claims Mismatch

The Stage 3 verification fails because the final MLE evaluations (opening claims) don't match:

| Instance | Zolt Claim | Jolt Expected | Match |
|----------|------------|---------------|-------|
| Shift | 13328834370005231... | 9669677241730825... | **NO** |
| InstructionInput | 3842266989647484... | 10802936892837509... | **NO** |
| Registers | 6520606900337296... | 6520606900337296... | **YES** |

```
output_claim:          6108158350971909917768492919875926703216279897287968891477060522070057021008
expected_output_claim: 10978770909765126050242131433050941553872066487109074337453920367876979372731
```

### Analysis

Since the **Registers** claim matches but **Shift** and **InstructionInput** don't:

1. The core MLE binding mechanism is correct (Registers works)
2. The eq polynomial evaluations are likely correct
3. The issue is specific to **Shift** and **InstructionInput** `finalClaims()` computation
4. Possibly related to eq_plus_one handling or the specific MLE structure

### Investigation Needed

1. **ShiftMLEs.finalClaims()** - Check how the bound values are extracted
   - Are the eq_plus_one values being combined correctly?
   - Is the gamma^4 * (1-noop) term computed right?

2. **InstrInputMLEs.finalClaims()** - Check the left/right operand combination
   - Are the flag values (left_is_rs1, left_is_pc, etc.) correct after binding?
   - Is the (eq_outer + gamma^2 * eq_product) weighting correct?

3. **eq_plus_one polynomial** - Verify the shift formula
   - eq_plus_one(r, j) should equal eq(r, j+1)
   - Check if binding order affects the final value

### Verified So Far

1. Round polynomial coefficients (c0, c2, c3) - ALL MATCH
2. Challenges derived from transcript - ALL MATCH
3. Batching coefficients - ALL MATCH
4. Initial batched claim - MATCHES
5. Registers opening claim - MATCHES
6. Shift opening claim - MISMATCH
7. InstructionInput opening claim - MISMATCH

### Next Steps

1. Debug `ShiftMLEs.finalClaims()` to compare with Jolt's expected value
2. Debug `InstrInputMLEs.finalClaims()` similarly
3. Add debug output for the final bound MLE values
4. Compare eq_plus_one final evaluations between Zolt and Jolt

### Key Files

- `/Users/matteo/projects/zolt/src/zkvm/spartan/stage3_prover.zig` - Opening claim computation
- `/Users/matteo/projects/jolt/jolt-core/src/zkvm/spartan/shift.rs` - ShiftSumcheck finalClaims
- `/Users/matteo/projects/jolt/jolt-core/src/zkvm/spartan/instruction_input.rs` - InstructionInput finalClaims
- `/Users/matteo/projects/jolt/jolt-core/src/zkvm/claim_reductions/registers.rs` - Registers finalClaims
