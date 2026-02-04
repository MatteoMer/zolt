# Zolt-Jolt Compatibility Implementation

## Status: Session 41 - Stage 5 P/Q Binding Fix + Opening Claims Investigation

## Major Progress!

**Fixed Instance 1 P/Q binding during cycle rounds!**
- Added P_raf, P_rw, P_val, Q_raf, Q_rw, Q_val binding in cycle rounds path
- Round polynomial coefficients for rounds 129+ now match Jolt
- Transcript challenges now match Jolt
- output_claim now matches Jolt (when accounting for endianness)

## Current Issue: Opening Claims Mismatch

The sumcheck polynomial evaluation (output_claim) matches, but expected_claim doesn't.
This means the opening claims are being computed incorrectly.

### Debug Findings

1. **Polynomial coefficients**: ✓ Match between Zolt and Jolt
2. **Challenges**: ✓ Match between Zolt and Jolt
3. **output_claim**: ✓ Match (after endianness conversion)
4. **expected_claim**: ✗ Doesn't match

### Expected Claim Formula

```
expected_claim = batch0 * inst0_claim + batch1 * inst1_claim + batch2 * inst2_claim
```

Where:
- inst0_claim = inc_claim * wa_claim * lt_eval (Instance 0: RegistersValEvaluation)
- inst1_claim = eq_combined * ra_claim (Instance 1: RamRaClaimReduction)
- inst2_claim = eq * ra * (val + gamma * raf) (Instance 2: InstructionReadRaf)

### Hypothesis: H_prime Initialization Issue

Looking at PhaseCycle2 initialization in Zolt:
- H_prime is computed from cycle_challenges[0..prefix_n_vars]
- cycle_challenges are set AFTER the polynomial computation
- This might cause H_prime to be initialized with wrong values

### Key Code Locations

**Instance 1 (RamRaClaimReduction):**
- PhaseCycle polynomial: lines 1823-1901 (PhaseCycle1) and 1902-2053 (PhaseCycle2)
- H_prime initialization: lines 1907-1945
- Binding code (my fix): lines 3163-3273

**ra_claim computation:**
- Zolt: line 3704, returns `ram_ra_claim` from computeEqAtIndex sum
- Jolt: Uses `H_prime.final_sumcheck_claim()` which should be H_prime[0] after binding

### Next Steps

1. **Verify H_prime[0] after final binding**
   - Print H_prime[0] after round 135
   - Compare with Jolt's ra_claim_reduced

2. **Check if cycle_challenges timing is correct**
   - Verify cycle_challenges[0..3] are set before PhaseCycle2 init

3. **Consider using H_prime[0] instead of computeEqAtIndex sum for ra_claim**

### Working Commands

```bash
# Build optimized (~13 sec proof generation)
zig build -Doptimize=ReleaseFast

# Generate proof
./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format -o /tmp/zolt_proof_dory.bin --export-preprocessing /tmp/zolt_preprocessing.bin --trace-length 64

# Verify
cd /home/vivado/projects/jolt && cargo test -p jolt-core --features zolt-debug --lib test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture
```

### Session Summary

- Fixed Instance 1 P/Q binding during cycle rounds (commit 2c97a1a)
- Verified polynomial coefficients and challenges now match
- Identified that opening claims (specifically ra_claim) might be wrong
- H_prime[0] appears to be 0, which shouldn't be the case

SESSION_ENDING - Fixed P/Q binding. Need to investigate H_prime initialization and ra_claim computation for Instance 1.
