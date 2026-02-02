# Zolt-Jolt Compatibility Implementation

## Status: IN PROGRESS - Stage 5 Expected Claim Mismatch

## Session 129 Summary

### MAJOR BREAKTHROUGH: ra_claims NOW MATCH!

After extensive debugging, the InstructionRa chunks are now correctly computed:
- Zolt: `InstructionRa(0) = 119b26350ef30d2127138e6672c75ea5`
- Jolt: `ra_claims[0] = [a5, 5e, c7, 72, 66, 8e, 13, 27, ...]` (LE) → same value!

The fix was understanding that binding transforms the initial value.

### All Individual Components MATCH:

1. **Instance 0 (RegistersValEvaluation)**: ✓
   - `inc_claim * wa_claim * lt_eval` matches verifier's expected computation
   - Value: `2e133a8eb83ed52d70c91f47fe5c8d8c118ac4a7969d53212f19f47fa1b9a265`

2. **Instance 1 (RamRaClaimReduction)**: ✓
   - `ra_claim_reduced` matches: `0956072d38428d511d5342e39c916fe33a948b3e88eff3755eb97c124ab471ff`

3. **Instance 2 (InstructionReadRaf)**: ✓
   - Final result matches: `[b1, 30, 62, bc, a8, dd, 8c, d2, 53, f9, 1f, 69, f5, 7b, e1, 72]`

4. **Batching coefficients**: ✓
   - `batch0 = d123dc2a0dee8c5ade56bfc5643d9704`
   - `batch1 = 149a2e91d4267f14003f2da6a0192a50`
   - `batch2 = 313aa709c6314e254d58c99ee2755045`

5. **Initial claim**: ✓
   - `00fb6017bc481b2ee8aa7e53dbc0dab21c0a902d470ec8a2a0666ce9d0780599`

6. **Polynomial coefficients (Round 0)**: ✓
   - `c0 = 0227ff26f6fc2e8d99f99d71df1d9008`
   - `c2 = 154d5f5a181d7a180ac943e18f7e3417`

### The Remaining Issue

Despite all components matching individually, the verification fails:
```
output_claim:   [43, 99, b3, 5d, b4, 2d, 6b, ae, bf, 91, 7b, dd, d4, 96, ac, ce, ...]
expected_claim: [a5, 06, a9, 32, b6, e9, 68, 44, 5c, f2, 37, 59, df, fa, 57, c6, ...]
```

Where:
- `output_claim` = final sumcheck polynomial evaluation at challenges (computed by verifier from proof)
- `expected_claim` = sum of instance claims weighted by batching coefficients (computed by verifier from opening claims)

### Hypothesis

The issue may be:
1. Challenge computation differs slightly at some intermediate round
2. Polynomial coefficient compression/decompression has a subtle bug
3. Round offset handling in the batched sumcheck

### Next Steps

1. Add per-round verification: compare `e` value after each round
2. Check if polynomial `eval_from_hint` gives correct results
3. Verify compressed polynomial format matches between Zolt and Jolt
4. Compare intermediate claims during the 136-round sumcheck

### Files Modified

- `src/zkvm/proof_converter.zig`: Added debug for InstructionRa insertion
- `src/zkvm/jolt_serialization.zig`: Added serialization debug
- `jolt-core/src/zkvm/claim_reductions/ram_ra.rs`: Added eq_combined debug
- `jolt-core/src/subprotocols/sumcheck.rs`: Fixed Stage 5 batching coeff label

### Test Commands

```bash
# Build Zolt
zig build -Doptimize=ReleaseFast

# Generate proof
./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format --export-preprocessing logs/zolt_preprocessing.bin -o logs/zolt_proof_dory.bin

# Verify
cp logs/zolt_*.bin /tmp/
cd /home/vivado/projects/jolt
cargo test --package jolt-core --features zolt-debug test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture
```
