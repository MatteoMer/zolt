# Zolt-Jolt Compatibility Implementation

## Status: IN PROGRESS - RamRaClaimReduction Cycle Rounds Implementation

## Session 121 Summary

### Progress Made
1. **Implemented RamRaClaimReduction PhaseAddress sumcheck**
   - Added sparse state tracking: ram_addresses, ram_cycles, ram_G_A, ram_G_B
   - Precompute G_A[i] = eq_raf(c_i) + γ·eq_val(c_i), G_B[i] = eq_rw(c_i) + γ·eq_val(c_i)
   - Initialize B_1 = eq(r_address_raf, k), B_2 = eq(r_address_rw, k) polynomials
   - Initialize ram_ra_F expanding table for tracking eq(r_addr_reduced, k)

2. **PhaseAddress rounds (0-15 of RamRaClaimReduction = rounds 112-127)**
   - For each round, iterate over RAM accesses
   - Compute polynomial evals based on address bit being bound
   - Bind B_1, B_2 and update ram_ra_F after each challenge

3. **Added helper function `computeEqAtPoint`**
   - Computes eq(r, k) for a specific point k given r in BIG_ENDIAN order

### Remaining Issue
**PhaseCycle implementation is incorrect**

The cycle round implementation (rounds 128-135) is using precomputed G_A/G_B which have FULL cycle eq values baked in. However, during the sumcheck, we need to bind the eq polynomials incrementally.

For Fibonacci:
- claim_raf = 0, claim_rw = 0
- claim_val_final ≠ 0, claim_val_eval ≠ 0
- ram_ra_input = γ*claim_val_final + γ³*claim_val_eval

The verifier reports:
- Instance 1 expected claim mismatch (Stage 5 sumcheck fails)

### Proper Cycle Round Algorithm
Looking at Jolt's ram_ra.rs:

**PhaseCycle1** (first log_T/2 cycle rounds):
- Uses P/Q buffer structure for prefix-suffix optimization
- P_x[c_lo] = eq(r_cycle_x_lo, c_lo)
- Q_x[c_lo] = Σ_{c_hi} H[c_lo, c_hi] · eq_x_hi(c_hi)
- Polynomial: coeff_raf * P_raf * Q_raf + coeff_rw * P_rw * Q_rw + coeff_val * P_val * Q_val

**PhaseCycle2** (last log_T/2 cycle rounds):
- Dense sumcheck over H_prime[c_hi] = Σ_{c_lo} H[c_lo,c_hi] · eq(r_prefix, c_lo)
- Uses eq_raf_hi, eq_rw_hi, eq_val_hi polynomials

For sparse implementation, we need to:
1. Track α_combined for each access after PhaseAddress completes
2. Track eq_cycle contributions that need to be bound during cycle rounds
3. Properly compute polynomial at each cycle round

### Files Modified This Session
- `/home/vivado/projects/zolt/src/zkvm/spartan/stage5_prover.zig`
  - Added RamRaClaimReduction state initialization (ram_addresses, ram_cycles, ram_G_A, ram_G_B)
  - Added B_1, B_2 eq polynomials for address rounds
  - Added ram_ra_F expanding table
  - Implemented PhaseAddress polynomial computation
  - Added state binding after each challenge
  - Added `computeEqAtPoint` helper function

### Test Commands
```bash
# Build Zolt
zig build -Doptimize=ReleaseFast

# Generate proof
./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format --export-preprocessing logs/zolt_preprocessing.bin -o logs/zolt_proof_dory.bin

# Copy to /tmp for Jolt test
cp logs/zolt_*.bin /tmp/

# Verify with Jolt
cd /home/vivado/projects/jolt
cargo test --package jolt-core --features zolt-debug test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture
```

## Next Steps
1. Fix PhaseCycle implementation:
   - Option A: Implement proper P/Q buffer structure for cycle rounds
   - Option B: For sparse traces, compute polynomial directly from bound challenges

2. The key insight for sparse implementation:
   - After PhaseAddress, compute α_combined[i] = α₁·scale_raf + γ²·α₂·scale_rw (per access)
   - Track which cycle eq factors need binding during cycle rounds
   - eq_cycle_raf, eq_cycle_rw, eq_cycle_val need separate tracking

3. Test with Jolt verifier

## SESSION_ENDING

Progress saved. PhaseAddress implemented correctly but PhaseCycle needs proper eq binding. Next session should fix the cycle round polynomial computation.

