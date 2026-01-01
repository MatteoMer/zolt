# Zolt-Jolt Compatibility TODO

## Current Status: Session 31 - January 2, 2026

**All 702 tests pass**

### Issue: Commitment/Preprocessing Mismatch

The r0 challenge differs between Zolt and Jolt because:
1. Zolt generates its own Dory commitments for the polynomials
2. Jolt's verifier preprocessing contains Jolt's expected commitments
3. The transcript includes these commitments
4. Different commitments → different transcript state → different challenges

**Solution Options:**
1. **Zolt generates both proof AND preprocessing** - The preprocessing contains the commitments that were appended to transcript. If Zolt exports both, they'll match.
2. **Use Jolt's SRS and matching commitment algorithm** - Ensure byte-for-byte identical commitments

### Progress This Session

1. [x] Fixed 125-bit mask in challenge scalar (was incorrect)
2. [x] Fixed memory layout constants (128MB memory, 4KB stack)
3. [x] Identified that Dory commitments differ between Zolt and Jolt
4. [x] Understood that proof+preprocessing must come from same source
5. [x] All polynomial evaluation formulas verified correct

### Key Finding

The transcript divergence is NOT a bug in Zolt's transcript implementation. It's because:
- Jolt preprocessing was generated with Jolt's Dory commitments
- Zolt proof was generated with Zolt's Dory commitments
- These commitments are appended to transcript before challenges
- Different commitments → different challenges

### Next Steps

1. [ ] Test with Zolt-generated preprocessing AND proof together
2. [ ] Ensure Zolt's Dory commitment output matches arkworks format exactly
3. [ ] Verify GT element serialization byte-by-byte with Jolt

### Blocking Issues

1. **Dory proof generation panic** - index out of bounds in openWithRowCommitments
   - `params.g2_vec` has length 64 but `current_len` is 128
   - Need to fix SRS loading or verify polynomial degree bounds

## Verified Correct
- [x] Blake2b transcript implementation format
- [x] Field serialization (Arkworks format)
- [x] Memory layout constants match Jolt
- [x] Challenge scalar computation (no 125-bit mask)
- [x] UniSkip polynomial generation logic
- [x] R1CS constraint definitions (19 constraints, 2 groups)
- [x] Split eq polynomial factorization (E_out/E_in tables)
- [x] Lagrange kernel L(tau_high, r0)
- [x] MLE evaluation for opening claims
- [x] Gruen cubic polynomial formula
- [x] bind() operation (correct eq factor)
- [x] Polynomial evaluation (Horner's method)
- [x] Interpolation (Vandermonde inverse)
- [x] evalsToCompressed produces correct compressed coefficients
- [x] All 702 Zolt tests pass

## Test Commands
```bash
# Zolt tests
zig build test --summary all

# Generate proof AND preprocessing from Zolt
zig build -Doptimize=Debug && ./zig-out/bin/zolt prove path/to/elf \
  --jolt-format \
  --export-preprocessing /tmp/zolt_preprocessing.bin \
  -o /tmp/zolt_proof_dory.bin

# Jolt verification with Zolt preprocessing
cd /Users/matteo/projects/jolt
cargo test --package jolt-core test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture
```
