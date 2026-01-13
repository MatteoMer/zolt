# Stage 4 Investigation Log

**Status:** Stage 4 fails with `output_claim != expected_output_claim`

## Validated Facts

1. **Stage 3 challenges MATCH** between Zolt and Jolt:
   - Zolt ROUND_0: `[21, 7a, 93, 9f, 1e, bf, 00, 7b]`
   - Jolt challenge[0]: `[21, 7a, 93, 9f, 1e, bf, 00, 7b]`

2. **The "zeros" in r_cycle debug are expected** - MontU128Challenge stores as `[0, 0, low, high]`, so first 8 bytes are always 0

3. **Transcript states appear to match** at key points

4. **Stage 4 error values:**
   ```
   output_claim:          4025718365397880246610377225086562173672770992931618085272964253447434290014
   expected_output_claim: 12140057478814186156378889252409437120722846392760694384171609527722202919821
   ```

## Things Tried

- [x] Reversed r_cycle ordering (LE→BE) - didn't help
- [x] Increased Dory SRS size - not the issue
- [x] Compared Stage 3 challenge bytes - they match
- [x] Analyzed MontU128Challenge structure - understood the [0,0,low,high] format
- [x] **Tried Montgomery conversion fix** - Changed `challengeScalar128Bits()` to use `toMontgomery()` - **DID NOT HELP**

## Current Status

The Montgomery hypothesis was WRONG. Converting to proper Montgomery form did not fix the issue.

The problem is likely in how Stage 4 polynomial values are computed, not in challenge representation.

## Next Steps

1. **Compare Stage 4 polynomial evaluations:**
   - Add debug to print rd_wv, rs1_v, rs2_v MLE sums
   - Compare Zolt's computed values vs expected from Stage 3

2. **Check gamma derivation in Stage 4:**
   - Verify gamma is derived identically

3. **Trace eq polynomial evaluation:**
   - Print both inputs to eq_eval
   - Compare intermediate values

## Key Files

- `src/zkvm/spartan/stage4_prover.zig` - Stage 4 prover (likely bug location)
- `src/zkvm/ram/read_write_checking.zig` - RWC polynomial construction
- `jolt-core/src/zkvm/registers/read_write_checking.rs` - Jolt Stage 4 verifier
