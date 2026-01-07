# Zolt-Jolt Compatibility - Stage 2 Progress

## Current Status
- Stage 1: PASSING ✅
- Stage 2: FAILING ❌ (sumcheck output_claim mismatch)
- Stage 3+: Not reached yet

## Current Issue
Stage 2 sumcheck output_claim doesn't match expected_output_claim:
- `output_claim`: 7092315906761387499155192332297668483214634895487230792733107015466920310794
- `expected_output_claim`: 11535186225949250426807989625067498736367003469117527002588759500768361489976

## Root Cause Analysis

The sumcheck verification fails because the prover's `output_claim` doesn't match the verifier's `expected_output_claim`.

### Expected Output Claim Formula (Jolt Verifier)
```
expected = L(τ_high, r0) * Eq(τ_low, r_tail_reversed) * fused_left * fused_right
```

The factor claims (fused_left/fused_right components) are appended to the transcript. The verifier reads these and computes expected_output_claim.

### Verified Components
1. ✅ EqPolynomial.evals produces correct partition of unity (sum = 1)
2. ✅ Witness values are populated from trace correctly
3. ✅ Gruen polynomial computation matches Jolt's formula
4. ✅ Challenges are reversed correctly for BIG_ENDIAN indexing
5. ✅ Batched sumcheck structure follows Jolt's pattern

### Suspected Issues
1. **Factor claims differ** - Zolt's factor evaluations don't match Jolt's transcript bytes
   - Zolt: `{ 4, 112, 107, ...}` (BE)
   - Jolt: `[05, f8, e7, ...]` (LE first 8 bytes)
   - These represent different field element values

2. **Possible causes**:
   - Witness values differ from Jolt's trace values
   - Eq polynomial evaluation uses different indexing
   - Challenge ordering differs subtly

## Recent Session Findings

### Session 5
- Fixed EqPolynomial.evals to match Jolt's rev().step_by(2) iteration
- Fixed computeRoundPolynomial to use interleaved format
- Added debug output for eq_evals and witness values
- **Finding**: eq_sum = 1 (partition of unity correct)
- **Finding**: witness[0][LeftInstructionInput] = 0 (correct for LUI instruction)

## Key Files
- `src/zkvm/proof_converter.zig` - Stage 2 batched sumcheck generation
- `src/zkvm/spartan/product_remainder.zig` - ProductVirtualRemainder prover
- `src/poly/split_eq.zig` - Gruen polynomial and eq tables
- `src/poly/mod.zig` - EqPolynomial implementation
- `src/zkvm/r1cs/constraints.zig` - Witness generation from trace

## Debug Commands
```bash
# Build and test
zig build test --summary all

# Generate proof with debug output
zig build -Doptimize=ReleaseFast
./zig-out/bin/zolt prove --jolt-format -o /tmp/proof.bin examples/fibonacci.elf 2>&1 | grep -E "FACTOR_EVALS|GRUEN"

# Test with Jolt
cd /Users/matteo/projects/jolt
cargo test --package jolt-core test_verify_zolt_proof -- --ignored --nocapture 2>&1 | tail -100
```

## Next Steps
1. Add debug output to Jolt's `compute_claimed_factors` to print eq_one[0..2], eq_two[0..2], and trace values
2. Compare exact field element values between Zolt and Jolt
3. Check if RightInstructionInput uses signed vs unsigned representation correctly
4. Verify the first few challenges match between Zolt and Jolt transcripts
