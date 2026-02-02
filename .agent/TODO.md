# Zolt-Jolt Compatibility Implementation

## Status: Stage 5 Operand Evaluation Investigation

## Session 12 Progress (Current)

### Critical Finding UPDATED

The Stage 5 sumcheck verification fails because `expected_claim != output_claim`.

**Previous incorrect hypothesis**: r_address_prime values differ

**Corrected understanding**:
- Raw sumcheck challenges MATCH (verified)
- r_address_prime values MATCH when interpreted correctly
  - Jolt's debug output for `r[i]` serializes the Challenge type directly (which shows the raw limbs including zeros)
  - When converted to F via `.into()`, the values match Zolt's values
- Operand evaluation formulas are mathematically equivalent

**Actual mismatch**:
- right_op_eval: Zolt=`60f99a29...` vs Jolt=`609f9a29...` (byte 5: `f9` vs `9f`)
- identity_eval: Zolt=`3705a51b...` vs Jolt=`37055a1b...` (byte 6: `a5` vs `5a`)

### Hypotheses to Investigate

1. **mul_u128 vs mul(power)**: Jolt uses `r[i].into().mul_u128(1 << k)` while Zolt uses `r[i].mul(power)` where power is accumulated by doubling. These should be equivalent but may have subtle differences.

2. **Challenge -> F conversion**: The `.into()` conversion in Jolt converts MontU128Challenge to Fr. Need to verify Zolt does the same thing correctly.

3. **Power accumulation**: Zolt computes power by repeated doubling in the field. Any error in doubling would compound over 64-127 iterations.

### Verified

1. Round 0 challenge bytes MATCH:
   - Zolt: `7ca25a17c12902cb92c0d5878c3b73da` (BE)
   - Jolt: `7ca25a17c12902cb92c0d5878c3b73da` (BE)

2. All 136 sumcheck challenges MATCH

3. Output claim from polynomial evaluation MATCHES

4. Operand formulas are equivalent:
   - Left: `sum_{i=0}^{63} r[2i] * 2^(63-i)`
   - Right: `sum_{i=0}^{63} r[2i+1] * 2^(63-i)`
   - Identity: `sum_{i=0}^{127} r[i] * 2^(127-i)`

### Commands

Generate Jolt-format proof:
```bash
./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format --export-preprocessing logs/zolt_preprocessing.bin -o logs/zolt_proof_dory.bin
```

Cross-verify:
```bash
cp logs/zolt_proof_dory.bin /tmp/ && cp logs/zolt_preprocessing.bin /tmp/
cd ../jolt && cargo test -p jolt-core --lib test_verify_zolt_proof_with_zolt_preprocessing --features zolt-debug -- --ignored --nocapture
```

### Previous Sessions

- Session 1-8: Initial implementation, transcript ordering, MLE evaluations
- Session 9: MontU128Challenge multiplication fix - internal verification PASSED
- Session 10: Cross-verification debugging, input claims match, polynomial mismatch
- Session 11: Deep investigation - all components match but expected_claim still differs
- Session 12: Verified r_address_prime values match. Issue is in mul_u128 vs mul(power) computation?

### Next Step

Add debug output to Zolt's evaluateLeftOperand to print intermediate values and compare with Jolt. The byte-level difference (`f9` vs `9f`, `a5` vs `5a`) suggests a nibble or byte order issue within a single computation step rather than a completely wrong formula.

## Files Modified This Session

None yet - still debugging
