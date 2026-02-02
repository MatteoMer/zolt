# Zolt-Jolt Compatibility Implementation

## Status: IN PROGRESS - Stage 5 RamRaClaimReduction Implementation Needed

## Session 118 Summary

### Root Cause Identified
The Stage 5 verification fails because:

1. **Instance 1 (RamRaClaimReduction) is NOT implemented** for active rounds!
   - In `stage5_prover.zig` lines 1301-1304, when Instance 1 becomes active:
   ```zig
   } else {
       // Zero polynomial for now (TODO: implement RamRaClaimReduction)
       // This is correct if ram_ra_input = 0
   }
   ```
   - This outputs ZERO contribution when the instance becomes active (rounds 112-135)

2. **But ram_ra_input is NOT zero!** Debug shows:
   - `claim_raf (RamRafEvaluation) = 0`
   - `claim_val_final (RamValFinalEvaluation) = non-zero`
   - `claim_rw (RamReadWriteChecking) = 0`
   - `claim_val_eval (RamValEvaluation) = non-zero` (same as val_final!)
   - `ram_ra_input = γ*claim_val_final + γ³*claim_val_eval ≠ 0`

3. **ram_ra_claim is hardcoded to F.zero()** at line 2130:
   ```zig
   .ram_ra_claim = F.zero(),
   ```
   This should be the final reduced claim from the RamRaClaimReduction sumcheck.

### Verification Failure
```
output_claim:   [42, f3, 7b, 3a, c3, 1b, 09, f8, ...]  <- from polynomial evaluations
expected_claim: [44, 17, 5e, 31, f5, bc, 6f, 87, ...]  <- from expected output claims

Instance contributions to expected_claim:
- Instance 0 (RegistersValEvaluation): claim * batch0
- Instance 1 (RamRaClaimReduction): 0 * batch1 = 0 (WRONG - ram_ra_claim=0)
- Instance 2 (LookupsReadRaf): claim * batch2
```

### RamRaClaimReduction Sumcheck Details (from Jolt)
Proves: `Σ_{k,c} eq_combined(k, c) · ra(k, c) = input_claim`

Where:
- `eq_combined(k, c) = eq(r_addr_1, k)·(eq_raf(c) + γ·eq_val(c)) + γ²·eq(r_addr_2, k)·(eq_rw(c) + γ·eq_val(c))`
- `input_claim = claim_raf + γ·claim_val_final + γ²·claim_rw + γ³·claim_val_eval`

Three phases:
- **PhaseAddress**: First 16 rounds binding address variables (log_K rounds)
- **PhaseCycle1**: First 4 cycle rounds using prefix-suffix optimization (log_T/2 rounds)
- **PhaseCycle2**: Last 4 cycle rounds using dense sumcheck (log_T/2 rounds)

For Fibonacci: 16 + 8 = 24 rounds total, active during Stage 5 rounds 112-135.

### Files Modified This Session
- `/home/vivado/projects/zolt/src/zkvm/spartan/stage5_prover.zig`: Added debug for ram_ra_input components
- `/home/vivado/projects/zolt/.agent/TODO.md`: Updated with findings

### Verified Components
1. Batching coefficients match between Zolt and Jolt ✓
2. Input claims match ✓
3. Challenges match during cycle rounds (128-135) ✓
4. Transcript synchronization works ✓
5. Instance 0 and Instance 2 polynomial contributions appear correct ✓

### What Needs to Be Done
1. **Implement RamRaClaimReduction sumcheck polynomial computation** for rounds 112-135
   - PhaseAddress: Dense sumcheck over eq_combined * ra where ra is the RAM address polynomial
   - PhaseCycle1/2: Prefix-suffix then dense sumcheck over cycle variables

2. **Compute actual ram_ra_claim** - This is `H_prime.final_sumcheck_claim()` from Jolt's PhaseCycle2State

### Alternative: Skip RamRaClaimReduction for No-RAM Programs?
For Fibonacci which has no RAM operations, we might be able to:
1. Check if trace has no RAM ops
2. Set ram_ra_input = 0 artificially
3. Output zero polynomials for Instance 1

But this is a workaround, not a fix. The proper fix is implementing RamRaClaimReduction.

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

## SESSION_ENDING - Context Running Low

Progress saved to TODO.md. The main finding is that RamRaClaimReduction sumcheck (Instance 1 of Stage 5) is not implemented for active rounds, causing verification failure when ram_ra_input ≠ 0.
