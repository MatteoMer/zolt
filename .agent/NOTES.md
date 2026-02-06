# Session 93 Notes - Stage 5 RAF Polynomial Drift Deep Dive

## Summary

Continuing investigation of Stage 5 sumcheck verification failure. The issue is drift between `lookups_claim` (polynomial evaluation chain) and `materialized_sum` (direct computation) at the transition from address rounds to cycle rounds.

## Key Findings

### 1. Initial Claim Matches
At round 0:
- `total_sum(eq*combined_vals) = lookups_claim = af0ee294043c5efdb7e3d1fb851c28c5`
- Match: TRUE

### 2. Sumcheck Property Holds Throughout
- For all 128 rounds: `p(0) + p(1) = claim` ✓
- The sumcheck polynomial is internally consistent

### 3. Drift at Cycle Round Transition (Round 128)
```
materialized_sum (direct) = e728cffa1af93851e97fbac6cb36aca0
lookups_claim (poly chain) = f21f2ce546c92b0f7c9ad5e065cec05a
MISMATCH!
```

### 4. Jolt Verification Expected vs Zolt Output
- Zolt's output_claim after 136 rounds: `ad4ede5afd49bd1a1a104b8c7d8f2da0...`
- Jolt's expected_claim (sum of instances): `09f9ba2becbd9928e9433175d5401c41...`

## Technical Analysis

### The Address Round Formula
During address rounds 0-127:
- The polynomial is computed via prefix-suffix decomposition (Q arrays)
- `proverMsgRaf` computes `eval_0` and `eval_2` from Q arrays
- `eval_1 = claim - eval_0` (sumcheck property)
- `inst2_at_r = p(challenge)` becomes the new `lookups_claim`

### The Cycle Round Transition (Round 128)
Jolt's `init_log_t_rounds()`:
1. Materializes `ra_polys` from expanding tables
2. Materializes `combined_val_polynomial` from prefix checkpoints
3. Sets up multilinear polynomials over (address, cycle) variables

Zolt's equivalent:
1. Rematerializes `lookups_combined_vals` using prefix checkpoints
2. Reinitializes `lookups_eq_evals` for cycle rounds
3. Materializes `ra_chunk_weights` from expanding tables

### The Cycle Round Sum Formula
```
Σ_j eq_cycle(j, r_red) * combined_val[j] * eq_addr(k[j], r_addr)
```
Where:
- `eq_addr(k[j], r_addr)` = `∏_chunk ra[chunk][j]` = `lookups_ra_weights[j]`

## Possible Causes of Drift

1. **Expanding table values differ** - The eq accumulation might compute differently

2. **condenseUEvals issue** - The u_evals condensation at phase boundaries might differ

3. **Polynomial evaluation chain divergence** - The `inst2_at_r` computation from Q arrays might accumulate differently than the materialized sum

4. **Phase counting** - With small traces (T=64), we have 16 phases of 8 rounds each, not 8 phases of 16 rounds

## Debug Data

### Phase Configuration (T=64)
- num_phases = 16 (since log_T < 24)
- log_m = 128 / 16 = 8 rounds per phase
- Phase transitions occur after rounds 7, 15, 23, ..., 119, 127

### Expanding Table Values at Round 128
```
phase[0] v[0] = e63db0efdf987b32171fd744f8155e63
phase[1] v[0] = 671556bc65446d21a443335978dab0b8
phase[2] v[0] = b26e52c2...
phase[3] v[0] = 2ea9ff4b...
```

## Next Steps

1. Add debug to Jolt's prover to print expanding table values at round 128
2. Compare the polynomial evaluation chain step-by-step with Jolt
3. Check if the Q array binding produces the same running claim
4. Verify that condenseUEvals produces the same u_evals as Jolt

## Files Modified This Session

- `/home/vivado/projects/jolt/jolt-core/src/poly/prefix_suffix.rs`: Added `debug_Q()` accessor for Q arrays
- `/home/vivado/projects/jolt/jolt-core/src/zkvm/instruction_lookups/read_raf_checking.rs`: Updated debug to use `debug_Q()` accessor

## Test Commands

```bash
# Build and run Zolt
zig build -Doptimize=ReleaseFast
./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format -o /tmp/zolt_proof_dory.bin --export-preprocessing /tmp/zolt_preprocessing.bin --trace-length 64 2>&1 | tee /tmp/zolt_stage5_debug.log

# Run Jolt verification
cd /home/vivado/projects/jolt && cargo test -p jolt-core --features zolt-debug --lib test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture 2>&1 | tee /tmp/jolt_verify_debug.log
```
