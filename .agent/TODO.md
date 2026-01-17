# Zolt-Jolt Compatibility TODO

## Current Progress

| Stage | Status | Notes |
|-------|--------|-------|
| 1 | ✅ PASS | Univariate skip sumcheck |
| 2 | ❌ FAIL | RamReadWriteChecking formula still mismatching |
| 3 | ⏳ Blocked | Waiting for Stage 2 |
| 4 | ⏳ Blocked | Waiting for Stage 3 |
| 5-7 | ⏳ Blocked | Waiting for Stage 4 |

## Session 38 Progress - Stage 2 RWC Debugging (2026-01-17)

### Changes Made
1. **Fixed inc binding to LowToHigh order** - Jolt binds inc with LowToHigh, not HighToLow
   - `bound[i] = (1-r)*coeff[2*i] + r*coeff[2*i+1]`
2. **Fixed eq_evals binding to LowToHigh order** - Same as inc, must match Jolt's merged_eq binding
3. **Entry binding is correct** - Verified cycle halving matches Jolt

### Current Issue
The Stage 2 batched sumcheck still fails:
```
output_claim:          1876659233941903643546444704109340869460452145833020843094669927539534033863
expected_output_claim: 11897100216018413347250923153440684137462725884555934880339441864313184199375
```

Instance 2 (RWC) expected_claim = 8008895879423396486994993074856700301517848836219751932700329139803832082995 (non-zero)

### Analysis
- Entry at cycle 54, address 2049 (RAM termination write)
- inc[54] = 1 (write of value 1)
- After Phase 1: entry.cycle = 0, ra_coeff = eq(54, r_sumcheck), inc[0] = eq(54, r_sumcheck)
- Phase 2 runs for 16 address rounds

### Remaining Investigation
1. Check if Phase 2 polynomial computation is correct
2. Verify eq_addr computation in Phase 2 matches Jolt
3. Compare round polynomials between Zolt and Jolt directly
4. Check if gamma_rwc is being used correctly

## Testing Commands

```bash
# Build and run prover
zig build && ./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format --export-preprocessing logs/zolt_preprocessing.bin -o logs/zolt_proof_dory.bin --srs /tmp/jolt_dory_srs.bin

# Run Jolt verifier
cd /Users/matteo/projects/jolt && ZOLT_LOGS_DIR=/Users/matteo/projects/zolt/logs cargo test --package jolt-core test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture
```
