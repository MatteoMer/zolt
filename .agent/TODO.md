# Zolt-Jolt Compatibility Implementation

## Status: Session 70 - Stage 1 Sumcheck Investigation - ROOT CAUSE FOUND

## Previous Issue: R1CS vs Memory Trace Inconsistency
FIXED by injecting a full synthetic trace cycle for termination writes.

## Current Issue: Stage 1 Sumcheck Verification Failure

### ROOT CAUSE IDENTIFIED

**The sumcheck polynomial chain is CORRECT** - prover's claim tracking matches verifier exactly.

**The issue is the constraint polynomial Az*Bz computation!**

### Detailed Comparison

| Component | Prover (Zolt) | Verifier (Jolt) | Status |
|-----------|---------------|-----------------|--------|
| eq_factor | [f7, be, 45, d2, b1, 33, ...] | [f7, be, 45, d2, b1, 33, ...] | ✅ MATCH |
| final_claim | [be, 81, 99, 16, ...] | N/A (derived) | N/A |
| output_claim | [8f, 49, 4e, 9d, ...] | [8f, 49, 4e, 9d, ...] | ✅ MATCH |
| implied Az*Bz | [ad, ed, 4d, d6, 9a, ...] | [6e, d1, 32, 0b, 4a, ...] | ❌ MISMATCH |

### The Mismatch

Prover's implied `Az*Bz = final_claim / eq_factor`:
- `[ad, ed, 4d, d6, 9a, 9a, fb, 64, ...]`

Verifier's `inner_sum_prod` from R1CS evaluation:
- `[6e, d1, 32, 0b, 4a, 61, ec, ac, ...]`

### Jolt's inner_sum_prod Components

From debug output:
```
rx_constr: r_stream=[95, b7, 17, cd, ...], r0=[94, 4b, dd, 50, ...]
az_g0 = [55, 82, 0a, 57, ...]
bz_g0 = [60, bc, ff, e5, ...]
az_g1 = [77, 55, 35, 64, ...]
bz_g1 = [43, 8b, cd, af, ...]
az_final = az_g0 + r_stream * (az_g1 - az_g0) = [4d, 13, d1, af, ...]
bz_final = bz_g0 + r_stream * (bz_g1 - bz_g0) = [37, d9, 37, 60, ...]
inner_sum_prod = az_final * bz_final = [6e, d1, 32, 0b, ...]
```

### What's Different

The prover's sumcheck polynomial evaluates to a different Az*Bz than what the verifier recomputes using:
1. R1CS input evaluations (from opening claims)
2. Lagrange weights at r0
3. r_stream blending

This means either:
1. The prover's constraint matrices (FIRST_GROUP, SECOND_GROUP) differ from verifier
2. The prover's witness MLE evaluations at r_cycle differ from opening claims
3. The Lagrange basis weights computation differs
4. The r_stream blending formula differs

### Next Steps (Session 71)

1. Add debug to Zolt's prover to print az_g0, bz_g0, az_g1, bz_g1 after all rounds
2. Compare Lagrange weight computation between Zolt and Jolt
3. Verify R1CS input evaluations match what's in opening claims
4. Check if FIRST_GROUP_INDICES and SECOND_GROUP_INDICES match Jolt exactly

### Key Code Locations

**Zolt:**
- `src/zkvm/spartan/streaming_outer.zig`: Az/Bz computation
- `src/zkvm/r1cs/constraints.zig`: FIRST_GROUP_INDICES, SECOND_GROUP_INDICES
- `src/zkvm/r1cs/evaluation.zig`: R1CS evaluation

**Jolt:**
- `jolt-core/src/zkvm/r1cs/key.rs`: evaluate_inner_sum_product_at_point
- `jolt-core/src/zkvm/r1cs/constraints.rs`: R1CS_CONSTRAINTS_FIRST_GROUP, R1CS_CONSTRAINTS_SECOND_GROUP

## Test Commands

```bash
zig build -Doptimize=ReleaseFast
./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format -o /tmp/zolt_proof_dory.bin --export-preprocessing /tmp/zolt_preprocessing.bin --trace-length 64
cd /home/vivado/projects/jolt && cargo test -p jolt-core --features zolt-debug --lib test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture
```

## Files Modified This Session

- `src/zkvm/proof_converter.zig`: Added debug for lagrange_tau_r0, final_claim, implied Az*Bz
- `/home/vivado/projects/jolt/jolt-core/src/zkvm/r1cs/key.rs`: Added inner_sum_prod component debug
- `/home/vivado/projects/jolt/jolt-core/src/zkvm/spartan/outer.rs`: Added eq_factor product debug
- `.agent/TODO.md`: Updated with root cause analysis
