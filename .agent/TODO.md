# Zolt-Jolt Compatibility Implementation

## Status: Session 93 - Stage 5 RAF drift investigation (SESSION_ENDING)

## Current Issue: Stage 5 sumcheck verification fails

### Root Cause Analysis (Session 93 - Final Summary)

**Key Finding**: The polynomial evaluation chain (`lookups_claim`) is internally consistent throughout address rounds, but diverges from the materialized sum at cycle round start.

**Polynomial chain verification**:
- Round 0: `p(0) + p(1) = claim` ✓
- ...
- Round 127: `p(0) + p(1) = claim` ✓
- `lookups_claim` at R127 end: `f21f2ce546c92b0f7c9ad5e065cec05a`

**Materialized sum at cycle start**:
- `materialized_sum (WITH ra)` = `e728cffa1af93851e97fbac6cb36aca0`
- `sum_no_ra (WITHOUT ra)` = `39edfd2288a73b78ae89c459d4ef76a9`
- Neither matches `lookups_claim`!

**The formula being verified**:
```
Σ_j eq(j, r_red) * eq(k[j], r_addr) * combined_val[j]
```
Where:
- `eq(j, r_red)` = `lookups_eq_evals[j]` (reinitialized for cycle rounds)
- `eq(k[j], r_addr)` = `lookups_ra_weights[j]` = `Π_chunk ra_chunk[j]`
- `combined_val[j]` = rematerialized `table_mle(r_addr) + raf(r_addr)`

**Probable causes** (to investigate next session):
1. **ra_weights indexing**: The k_bound extraction `k >> suffix_bits & m_mask` might differ from Jolt
2. **combined_vals formula**: The `table_values_at_r_addr` or `raf_interleaved/raf_identity` computation might differ
3. **Phase counting**: With 16 phases (small trace), the expanding table structure differs from 8-phase case

### Next Steps (Priority)

1. **Add debug to Jolt prover** to print `ra_polys[chunk][j]` and `combined_val[j]` for first 5 cycles at cycle round start
2. **Compare values** between Zolt and Jolt:
   - `ra_chunk_weights[chunk][j]` vs `ra_polys[chunk][j]`
   - `lookups_combined_vals[j]` vs `combined_val_poly[j]`
   - `table_values_at_r_addr[t]` vs Jolt's table values
   - `raf_interleaved` and `raf_identity` vs Jolt's values
3. **Trace expanding table** through phases to verify indexing

### Debug Data Captured

**ra_chunk_weights (first 4 cycles)**:
```
j=0: [0080c722ae1a53de, 7d4fae9bcda84b1d, b7cf95c59c9b0de3, 4908ed2da0be0001, 1eb60dd5d30480d3, 287f540f8a24c69c, 24094594c9ac3db4, 36d19bb46c67d36f]
j=1: [...same first 7...]                                                                                                                            ff9d94cfd387853f]
j=2: [...same first 6...]                                                                                     fecf1e5eb19835e5, b945d6aa245225e7]
j=3: [...same first 6...]                                                                                     fecf1e5eb19835e5, 75760263c6faf21a]
```

**combined_vals (first 5 cycles)**:
```
j=0: 091decbe7fa1960c (identity path, table 0)
j=1: 091decbe7fa1960c (identity path, table 0)
j=2: 507c0b0cdf461a11 (interleaved, no table)
j=3: 091decbe7fa1960c (identity path, table 0)
j=4: 091decbe7fa1960c (identity path, table 0)
```

### Test Results
- Stage 1: PASSES ✅
- Stage 2: PASSES ✅
- Stage 3: PASSES ✅
- Stage 4: PASSES ✅
- Stage 5: FAILS ❌ (RAF polynomial drift)
- Stages 6-7: Not reached

### Files Modified This Session
- `src/zkvm/spartan/stage5_prover.zig`: Added drift checks and CLAIM_CHAIN debug
- `.agent/NOTES.md`: Updated with session findings
- `jolt/jolt-core/src/poly/prefix_suffix.rs`: Added `debug_Q()` accessor

### Build/Test Commands
```bash
zig build -Doptimize=ReleaseFast
./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format -o /tmp/zolt_proof_dory.bin --export-preprocessing /tmp/zolt_preprocessing.bin --trace-length 64
cd /home/vivado/projects/jolt && cargo test -p jolt-core --features zolt-debug --lib test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture
```
