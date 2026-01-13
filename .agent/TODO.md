# Zolt-Jolt Compatibility TODO

## Current Progress

| Stage | Status | Notes |
|-------|--------|-------|
| 1 | ✅ PASS | Univariate skip sumcheck |
| 2 | ✅ PASS | Batched sumcheck (5 instances) |
| 3 | ✅ PASS | Challenges match between Zolt and Jolt |
| 4 | ❌ FAIL | output_claim != expected_output_claim |
| 5-6 | ⏳ Blocked | Waiting for Stage 4 |

## Next Step

The Montgomery hypothesis was **WRONG** - tried it and it didn't help.

Focus should shift to **Stage 4 polynomial construction**:
1. Compare rd_wv, rs1_v, rs2_v MLE evaluations
2. Check gamma derivation matches
3. Trace eq polynomial inputs

## Testing

```bash
bash scripts/build_verify.sh
```

## See Also

- `.agent/stage4_investigation.md` - detailed investigation log
