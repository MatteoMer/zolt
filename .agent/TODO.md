# Zolt-Jolt Compatibility Implementation

## Status: Session 14 - Stage 5 Verification Failing

## Session 14 Investigation

### Current Issue

Stage 5 sumcheck verification fails:
- `output_claim: [63, 47, 60, 9d, ...]` (from Zolt's sumcheck polynomial)
- `expected_claim: [a9, 4a, 89, dd, ...]` (computed by Jolt's verifier)

### Debug Analysis

Jolt's verifier computes expected_claim as sum of three instances:
1. Instance 0: RegistersValEvaluation
2. Instance 1: RamRaClaimReduction
3. Instance 2: InstructionReadRaf

For Instance 2 (InstructionReadRaf), Jolt's formula:
```
eq_eval_r_reduction * ra_claim * (val_claim + gamma * raf_claim)
```

Where:
- raf_claim = (1 - raf_flag) * (left_op + gamma * right_op) + raf_flag * gamma * identity

Jolt's computed values:
- left_operand_eval:  [23, bd, 14, a1, ...]
- right_operand_eval: [2b, 88, fa, 0a, ...]
- identity_poly_eval: [24, 1e, 72, 5d, ...]
- final_result: [b6, fd, fb, 64, ...]

### Key Findings

1. **Operand evaluation functions match Jolt**:
   - Left: `r[2*i] * 2^(n/2-1-i)` (even indices)
   - Right: `r[2*i+1] * 2^(n/2-1-i)` (odd indices)
   - Identity: `r[i] * 2^(n-1-i)`

2. **Challenge normalization**:
   - Jolt's `normalize_opening_point` reverses r_cycle but NOT r_address
   - Need to verify Zolt does the same

3. **ra_chunk claims need verification**:
   - Jolt expects specific ra_claims[0..7] values
   - Zolt computes ra_chunks[i] = ra_chunk_weights[i][0] after binding

### Next Steps

1. Add debug output to Zolt's Stage 5 to print:
   - left_op_eval, right_op_eval, identity_eval at final point
   - ra_chunks[i] for i=0..7
   - eq_r_reduction
   - val_claim and raf_claim

2. Compare these values with Jolt's expectations

3. Identify which component(s) differ

### Commands

Generate proof:
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
- Session 12: Verified r_address_prime values match. Added operand eval debug
- Session 13: Fixed suffix_len overflow, fixed leftMsbUpdateCheckpoint, Stage 5 PASSED internally
