# Zolt-Jolt Compatibility Implementation

## Status: Session 93 - Stage 5 RAF drift investigation

## Current Issue: Stage 5 sumcheck verification fails

### Root Cause Analysis (Session 93)

The drift occurs because `lookups_claim` (polynomial evaluation chain) diverges from the materialized sum.

**Key discovery**: At cycle round start, the formula is:
```
Σ_j eq(j, r_red) * combined_val[j] * Π_chunk ra_chunk[j]
```

Where:
- `eq(j, r_red)` = eq polynomial over cycle variables
- `combined_val[j]` = table_mle(r_addr) + raf(r_addr) = SCALAR per cycle
- `ra_chunk[j]` = `eq(k[j][chunk_bits], r_addr[chunk_bits])` = expanding table evaluation

This SHOULD equal `lookups_claim` at the end of address rounds, but it doesn't:
- `materialized_sum (WITH ra)` = `e728cffa1af93851e97fbac6cb36aca0`
- `lookups_claim` = `f21f2ce546c92b0f7c9ad5e065cec05a`

**Possible causes**:
1. **ra_weights wrong**: The expanding table values or indexing differs from Jolt
2. **combined_vals wrong**: The rematerialization (table + raf) differs from Jolt
3. **lookups_claim wrong**: The polynomial evaluation chain through address rounds is incorrect

### Debug Results

**Initial claim (round 0)**: MATCHES
```
total_sum(eq*combined_vals) = lookups_claim = af0ee294043c5efdb7e3d1fb851c28c5
```

**Phase transitions**: DRIFT IMMEDIATELY
```
Phase 1: brute_sum = 51d640..., lookups_claim = 8c6b57...
```

**Key insight**: The brute_sum uses ORIGINAL combined_vals, but lookups_claim evolves through polynomial evaluation. During address rounds, the claim should track:
```
Σ_j condensed_eq[j] * prefix_eval * suffix_eval
```

NOT:
```
Σ_j condensed_eq[j] * original_combined_vals[j]
```

The prefix-suffix decomposition handles the polynomial evolution internally through Q arrays.

### Next Steps (Priority Order)

1. **Compare ra_weights with Jolt**
   - Add debug to Jolt's prover to print `ra_polys[chunk][j]` for first few cycles
   - Compare with Zolt's `ra_chunk_weights[chunk][j]`

2. **Compare combined_vals rematerialization**
   - Print Jolt's `combined_val_poly[j]` for first few cycles
   - Verify table_values_at_r_addr match
   - Verify raf_interleaved and raf_identity match

3. **Verify expanding table indexing**
   - The k_bound extraction: `k >> suffix_bits & m_mask`
   - The phase counting: 16 phases with log_m=8 for small traces

### Results
- Stage 1: PASSES ✅
- Stage 2: PASSES ✅
- Stage 3: PASSES ✅
- Stage 4: PASSES ✅
- Stage 5: FAILS ❌ (RAF polynomial mismatch after 128 rounds)
- Stages 6-7: Not reached

### Build/Test Commands
```bash
zig build -Doptimize=ReleaseFast
./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format -o /tmp/zolt_proof_dory.bin --export-preprocessing /tmp/zolt_preprocessing.bin --trace-length 64
cd /home/vivado/projects/jolt && cargo test -p jolt-core --features zolt-debug --lib test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture
```
