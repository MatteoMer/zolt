# Zolt-Jolt Compatibility TODO

## Current Progress

| Stage | Status | Notes |
|-------|--------|-------|
| 1 | ✅ PASS | Univariate skip sumcheck |
| 2 | ✅ PASS | Batched sumcheck (5 instances) |
| 3 | ✅ PASS | Challenges match between Zolt and Jolt |
| 4 | ❌ FAIL | Transcript diverges at Round 0 challenge |
| 5-6 | ⏳ Blocked | Waiting for Stage 4 |

## Stage 4 Investigation Summary

### What's CONFIRMED working:
- ✅ Gamma matches (0xBC6DB96DA7E28854F05610B16FA99513)
- ✅ Batching coefficients match
- ✅ Input claims match
- ✅ MLE sums match Stage 3 claims (rd_wv, rs1_v, rs2_v)
- ✅ Polynomial combination logic is correct in proof_converter.zig

### The bug:
**Transcript diverges between Zolt prover and Jolt verifier at Stage 4 Round 0:**
- Zolt Round 0 challenge: `2b 6e f8 0c e5 44 d9 18 ...`
- Jolt Round 0 challenge: `9a 95 69 5e 07 04 10 de`

### Current hypothesis:
The compressed polynomial bytes being **serialized to the proof** may differ from what Jolt's verifier expects to read and append to its transcript.

## Next Steps

1. Compare exact bytes Zolt writes to proof vs what Jolt reads
2. Check field element serialization format (LE vs BE, Montgomery)
3. Verify CompressedUniPoly serialization in jolt_types.zig

## Testing

```bash
bash scripts/build_verify.sh
```

## See Also

- `.agent/stage4_investigation.md` - detailed investigation log
