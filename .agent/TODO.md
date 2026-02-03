# Zolt-Jolt Compatibility Implementation

## Status: Session 18 - Stage 5 Cycle Round Polynomial Investigation

## Current Issue

Stage 5 verification fails with output_claim != expected_claim:
```
output_claim:   [ed, a5, f6, bf, 30, c4, 10, f8, ...]  (from sumcheck round 135)
expected_claim: [b2, 8f, 91, 24, 33, 0c, b4, 56, ...]  (from opening claims)
```

## Session 18 Findings

### Code Structure Understanding

The Stage 5 sumcheck has 136 rounds:
- **Rounds 0-127**: Address rounds (prefix-suffix decomposition)
- **Rounds 128-135**: Cycle rounds (standard sumcheck over bound polynomials)

Three instances are batched:
1. **Instance 0 (RegistersValEvaluation)**: Active for 8 cycle rounds, degree-3 polynomial
2. **Instance 1 (RamRaClaimReduction)**: Active for 24 rounds (16 address + 8 cycle), degree-2 polynomial
3. **Instance 2 (LookupsReadRaf)**: Active all 136 rounds, degree-9 polynomial during cycle rounds

### Cycle Round Polynomial Computation (Round >= 128)

For each cycle round:
1. Instance 0+1 compute polynomials via existing code (lines 1575-2015)
2. Instance 2 computes 9-factor product: eq * (8 ra_chunks) * combined_val
3. Polynomials are combined into `combined_coeffs` (degree-10)
4. Compressed coefficients [c0, c2, c3, ...] are serialized

### Jolt's Cycle Round Coefficients (from test output)

Round 128:
```
c0: [05, b5, df, f2, d3, 49, ca, d8, ...]
c2: [54, c2, 7c, f5, aa, 45, 65, 80, ...]
c3: [49, 36, 44, b5, 9d, f1, de, 64, ...]
```

### Key Insight: Degree Mismatch Comment

Line 2991 has a misleading comment:
```
// Instance 0 and 1 contribute at most degree-3 (constant polynomials for most rounds)
```

This is INCORRECT for cycle rounds. During cycle rounds:
- Instance 0 is degree-3 (product of 3 linear factors)
- Instance 1 is degree-2 (PhaseCycle2)
- Instance 2 is degree-9 (product of 9 linear factors)

Total combined polynomial degree = max(3, 9) = 9, not 10.

Wait - Jolt says degree = ra_polys.len() = 8+1 = 9, but the code creates `combined_coeffs` with 11 elements (degree-10). This might be the bug!

### POSSIBLE BUG: Degree Calculation

Zolt allocates 11 coefficients (degree-10):
```zig
var combined_coeffs = try self.allocator.alloc(F, 11);  // line 3000
```

But the actual degree should be:
- eq factor: degree-1
- ra factors: 8 factors of degree-1 each = degree-8 contribution
- combined_val: degree-0 (already evaluated)
- Total: degree-9

The polynomial has degree 9, requiring only 10 coefficients, not 11!

### Hypothesis

The extra coefficient is causing the polynomial to not match Jolt's structure. Zolt might be producing degree-10 polynomials when Jolt expects degree-9.

## Next Steps

1. **Verify degree**: Check `full_coeffs.len` after `finishMlesProductSumFromEvals` - should be 10 (degree-9)
2. **Check Jolt's degree**: In read_raf_checking.rs, verify n_evals = ra_polys.len() + 1 = 9, giving degree 9
3. **Fix allocation**: If confirmed, change `combined_coeffs` to 10 elements
4. **Test**: Generate new proof and cross-verify

## Test Commands

```bash
# Generate proof
./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format --export-preprocessing logs/zolt_preprocessing.bin -o logs/zolt_proof_dory.bin

# Cross-verify
cp logs/zolt_proof_dory.bin /tmp/ && cp logs/zolt_preprocessing.bin /tmp/
cd ../jolt && cargo test -p jolt-core --lib test_verify_zolt_proof_with_zolt_preprocessing --features zolt-debug -- --ignored --nocapture
```

## Session History

- Session 1-8: Initial implementation, transcript ordering
- Session 9: MontU128Challenge multiplication fix
- Session 10-11: Cross-verification debugging
- Session 12: Verified r_address_prime challenges match
- Session 13: Fixed suffix_len overflow
- Session 14: Internal verification passes
- Session 15: Confirmed opening claims match
- Session 16: Fixed LowerWord/UpperWord/LowerHalfWord suffix MLEs
- Session 17: Verified all opening claims match
- Session 18: **Discovered potential degree mismatch** (degree-10 vs degree-9)
