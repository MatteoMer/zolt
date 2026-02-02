# Zolt-Jolt Compatibility Implementation

## Status: IN PROGRESS - Stage 5 Batched Sumcheck Debugging

## Session 113 Summary (continued)

### Key Discovery

Found that Jolt's BatchedSumcheck uses a specific pattern for inactive instances:
1. Initial individual claims are scaled by 2^(max_rounds - instance_rounds)
2. Each inactive round, the polynomial is CONSTANT with value `previous_claim / 2`
3. This results in claims being halved each round until the instance becomes active

### Current Analysis

For round 0 with Instance 0 (RegistersValEvaluation, 8 rounds):
- max_rounds = 136, instance_rounds = 8
- offset = 128 (instance starts at round 128)
- Round 0 is inactive (round < offset)
- Initial claim = input_claim * 2^128
- Polynomial = constant with value (input_claim * 2^128) / 2 = input_claim * 2^127
- This matches my `scale = remaining_rounds - num_rounds - 1 = 127`

So the scaling formula is correct!

### Remaining Issues

1. **Polynomial coefficients still differ** - even though the logic seems correct, the output doesn't match

2. **Need to verify:**
   - Batch coefficients (batch0, batch1, batch2) are derived from transcript correctly
   - Instance input claims match what Jolt expects
   - The polynomial combination formula is correct

3. **Debug output shows:**
   ```
   [STAGE5 COEFF ROUND 0] c0 = 0227ff26f6fc2e8d99f99d71df1d9008927616895c839a61b0e8249c7e779386
   [STAGE5 COEFF ROUND 0] inst01_p0 = 24e9f37c8fe20a5bcca11f8a09893313f0cae1922f47764faf4b569bfd3067ae
   [STAGE5 COEFF ROUND 0] inst2_eval0 = 1f31af1cf6c3199f163b7a1eca41ff90d4d10aa7489b7bb3642a41bb48bf9a99
   ```

   inst01_p0 and inst01_p2 should be equal (constant polynomial), and they are based on scaled claims.

### Next Steps

1. **Compare transcript state** - verify batch coefficients match Jolt
2. **Compare instance input claims** - check that regs_val_input matches Jolt
3. **Add Jolt-side debug** - print the actual polynomial values Jolt computes for round 0
4. **Check if interleave fix cascaded** - the interleave fix might affect how lookup indices are computed, which affects Q polynomial initialization

### Files Modified This Session

- `src/zkvm/spartan/stage5_prover.zig` - Fixed interleaveBits128, added debug
- `src/zkvm/lookup_table/prefix_suffix_prover.zig` - Added Q value debug

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

## Key Formula Reference

### Jolt's Interleave Format
```
interleave_bits(x, y) = (spread(x) << 1) | spread(y)
```
- x (left operand) → ODD bit positions (1, 3, 5, ...)
- y (right operand) → EVEN bit positions (0, 2, 4, ...)

### Inactive Instance Polynomial
For round k where instance hasn't started (k < offset):
- current_claim[k] = input_claim * 2^(max_rounds - instance_rounds) / 2^k
- polynomial = constant with value current_claim[k] / 2
- p(0) = p(1) = p(2) = current_claim[k] / 2
- scale_exponent = max_rounds - instance_rounds - 1 - k

### RangeCheck Table Combine
```
combine(prefixes, suffixes) = prefixes[LowerWord] * one + lower_word
```
Where:
- one = Q[One][b_idx] (suffix polynomial for Suffixes::One)
- lower_word = Q[LowerWord][b_idx] (suffix polynomial for Suffixes::LowerWord)
- prefixes[LowerWord] = 0 for j < XLEN (first 64 rounds)

SESSION_ENDING - saved progress to TODO.md
