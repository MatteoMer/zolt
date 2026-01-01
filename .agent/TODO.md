# Zolt-Jolt Compatibility TODO

## Current Status: Session 31 - January 2, 2026

**All 702 tests pass**

### Key Findings This Session

1. **Fixed 125-bit mask bug** - Zolt was incorrectly masking challenge scalars to 125 bits. Jolt's `challenge_scalar_128_bits` uses the full 128 bits. Fixed.

2. **Fixed memory layout constants** - Updated to match Jolt (128MB memory, 4KB stack).

3. **Verified polynomial computation is correct** - The Gruen cubic polynomial formula, bind() operation, interpolation, and evaluation all match Jolt exactly.

4. **Identified root cause of verification failure** - The proof and preprocessing MUST come from the same source. When Zolt generates a proof with its own Dory commitments, but verification uses Jolt's preprocessing (with Jolt's commitments), the transcript diverges because commitments are part of the Fiat-Shamir.

### Solution Path Forward

The proof + preprocessing must match:
- **Option A**: Zolt generates both proof AND preprocessing → they'll use the same commitments
- **Option B**: Ensure Zolt produces byte-identical Dory commitments to Jolt (requires matching SRS and commitment algorithm)

### Blocking Issues

1. **Dory proof generation panic** - index out of bounds when polynomial size exceeds SRS size
   - Need to ensure SRS is large enough for all polynomials being opened
   - Current error: `g2_vec.len = 64` but `current_len = 128`

## Summary of Verified Correct Components

### Transcript
- [x] Blake2b transcript format matches Jolt
- [x] Challenge scalar computation (128-bit, no masking)
- [x] Field serialization (Arkworks LE format)
- [x] Message and scalar append operations

### Polynomial Computation
- [x] Gruen cubic polynomial formula
- [x] Split eq polynomial factorization (E_out/E_in)
- [x] bind() operation (eq factor computation)
- [x] Lagrange interpolation (Vandermonde inverse)
- [x] Horner's method for evaluation
- [x] evalsToCompressed format

### RISC-V & R1CS
- [x] R1CS constraint definitions (19 constraints, 2 groups)
- [x] UniSkip polynomial generation
- [x] Memory layout constants match Jolt

### All Tests Pass
- [x] 702/702 Zolt tests pass

## Test Commands
```bash
# Run Zolt tests
zig build test --summary all

# Generate proof (may fail at Dory opening due to SRS size)
zig build -Doptimize=ReleaseFast && ./zig-out/bin/zolt prove path/to/elf \
  --jolt-format -o /tmp/zolt_proof.bin

# Jolt verification tests
cd /Users/matteo/projects/jolt
cargo test --package jolt-core test_verify_zolt_proof -- --ignored --nocapture
```

## Next Session Priority

1. Fix Dory SRS size issue (ensure SRS matches polynomial size)
2. Test with Zolt-generated preprocessing + proof together
3. Verify GT element serialization byte-by-byte
