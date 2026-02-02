# Zolt-Jolt Compatibility Implementation

## Status: IN PROGRESS - Stage 5 Prefix-Suffix Decomposition Mismatch

## Session 113 Summary

### Progress Made

1. **Fixed `interleaveBits128` to match Jolt's convention:**
   - OLD: x at even positions (0, 2, 4, ...), y at odd positions (1, 3, 5, ...)
   - NEW: x (left) at ODD positions, y (right) at EVEN positions
   - This matches Jolt's `interleave_bits(even_bits, odd_bits)` which does `(spread(x) << 1) | spread(y)`

2. **Verified suffix_len calculation is correct:**
   - Phase 0: suffix_len = 128 - (0+1)*8 = 120
   - This matches Jolt's `k.split((phases - 1 - phase) * log_m)` = k.split(120)

3. **Verified initial claims match:**
   - Zolt Stage 5 initial batched claim: `990578d0e96c66a0a2c80e472d900a1cb2dac0db537eaae82e1b48bc1760fb00`
   - Jolt Stage 5 initial claim: same value
   - Claims are identical, so the issue is in polynomial computation

4. **Identified that polynomial coefficients still differ:**
   - Jolt expects c0: [e2, ee, 6f, c7, ...]
   - Zolt produces c0: [02, 27, ff, 26, ...]

### Current Investigation

The suffix polynomial initialization and prefix MLE computation appear to follow the same logic as Jolt, but the resulting polynomial coefficients are different. The issue is somewhere in:

1. **Suffix MLE evaluation** - How we compute suffix_mle(suffix_bits) for each suffix type
2. **Prefix MLE evaluation** - How we compute prefix_mle(checkpoints, r_x, c, b, round)
3. **Table combine functions** - How we combine prefix and suffix evaluations

Key observation from debug output:
```
[SUFFIX INIT] cycle j=0 t_idx=0 k=0x00000000000000000000000040000000 prefix_bits=0 suffix_len=120
  suffix[0]=One t=1 idx=0
  suffix[1]=LowerWord t=32768 idx=0
```

The LowerWord suffix at `k=0x40000000` returns 32768 (0x8000). Let me verify this is correct:
- k = 0x40000000 = bit 30 set
- After uninterleaving: right operand = bits at even positions
- Bit 30 at position 30 (even) → right operand bit 15 set
- right & 0xFFFFFFFF = 0x8000 = 32768 ✓

This seems correct!

### Next Steps

1. **Add detailed debug to compare prefix MLE evaluations:**
   - Print prefixes_c0 and prefixes_c2 for round 0, b=0
   - Compare with expected values from Jolt

2. **Verify the tableCombine functions:**
   - Check that RangeCheck combine is: `prefixes[LowerWord] * one + lower_word`
   - Verify the multiplication and addition order

3. **Check the LowerWord prefix evaluation:**
   - In round 0 (j=0), LowerWord should return 0 (ignores first XLEN rounds)
   - The contribution should come only from the suffix

4. **Consider adding a test case:**
   - Create a minimal test that computes one read_checking round
   - Compare output with Jolt's expected values

### Key Components

1. **Prefix-Suffix Decomposition** (`src/zkvm/lookup_table/`)
   - `prefixes.zig` - All 46 prefix types with MLE implementations
   - `suffixes.zig` - All suffix types with MLE implementations
   - `prefix_suffix_prover.zig` - Q polynomial accumulation and proverMsg functions

2. **Stage 5 Prover** (`src/zkvm/spartan/stage5_prover.zig`)
   - Three-instance batched sumcheck (136 rounds)
   - Instance 0: RegistersValEvaluation (8 rounds)
   - Instance 1: RamRaClaimReduction (24 rounds)
   - Instance 2: LookupsReadRaf (136 rounds) - prefix-suffix decomposition

### Test Commands
```bash
# Build
zig build -Doptimize=ReleaseFast

# Generate proof
timeout 600 ./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format -o logs/zolt_proof_dory.bin

# Verify with Jolt
cd /home/vivado/projects/jolt
cargo test --package jolt-core --features zolt-debug test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture
```

### Debug Output Analysis

From Zolt proof generation (round 0):
```
[STAGE5 COEFF ROUND 0] c0 = 0227ff26f6fc2e8d99f99d71df1d9008927616895c839a61b0e8249c7e779386
[STAGE5 COEFF ROUND 0] claim = 00fb6017bc481b2ee8aa7e53dbc0dab21c0a902d470ec8a2a0666ce9d0780599
[STAGE5 COEFF ROUND 0] inst01_p0 = 24e9f37c8fe20a5bcca11f8a09893313f0cae1922f47764faf4b569bfd3067ae
[STAGE5 COEFF ROUND 0] inst2_eval0 = 1f31af1cf6c3199f163b7a1eca41ff90d4d10aa7489b7bb3642a41bb48bf9a99
```

The issue is likely in `inst2_eval0` (Instance 2's read_checking + RAF contribution).
