# Zolt-Jolt Compatibility Implementation

## Status: Session 44 - Multiple Bugs Fixed, One Remaining

## Summary of Fixes in This Session

### 1. Fixed Instance 2 (Lookups) Claim Tracking ✓
**Location:** `src/zkvm/spartan/stage5_prover.zig` around line 2203

**Problem:** After each round, `lookups_claim` was being recomputed from raw arrays that hadn't been properly bound, causing divergence from the actual polynomial evolution.

**Fix:** Update `lookups_claim` by evaluating Instance 2's polynomial at the challenge point:
```zig
// Evaluate Instance 2's degree-2 polynomial at challenge
const inst2_c0 = eval_0_inst2;
const inst2_c2 = eval_2_inst2.sub(eval_1_inst2).sub(eval_1_inst2).add(eval_0_inst2).mul(F.fromU64(2).inverse().?);
const inst2_c1 = eval_1_inst2.sub(eval_0_inst2).sub(inst2_c2);
const inst2_at_r = inst2_c0.add(inst2_c1.mulHiBigIntU128(challenge.limbs)).add(inst2_c2.mul(r2));
lookups_claim = inst2_at_r;
```

Also removed the incorrect recomputation at line 2486-2490.

### 2. Fixed Instance 1 (RamRaClaimReduction) PhaseAddress Polynomial ✓
**Location:** `src/zkvm/spartan/stage5_prover.zig` around line 1703

**Problem:** The code was incorrectly adding both contrib_0 and contrib_1 to both eval_0 and eval_1 (swapping when k_m=1).

**Fix:** Each access contributes to exactly ONE of eval_0 or eval_1 based on its address bit:
```zig
if (k_m == 0) {
    eval_0 = eval_0.add(contrib_0);  // Access at even address
} else {
    eval_1 = eval_1.add(contrib_1);  // Access at odd address
}
```

### 3. Remaining Issue: Placeholder Opening Claims
**Location:** `src/zkvm/proof_converter.zig` around line 154

**Problem:** The RamRa virtual opening claims (claim_raf, claim_val_final, claim_rw, claim_val_eval) are set to zero as placeholders instead of being computed from the actual trace.

**Impact:** `ram_ra_input` = γ*claim_val_final + γ³*claim_val_eval (since claim_raf and claim_rw are 0), but the polynomial computed from trace data produces a different sum.

**To Fix:** Need to compute the actual MLE evaluations for:
- `claim_raf = Σ_{accesses at (k,c)} eq(r_address_raf, k) * eq(r_cycle_raf, c)`
- `claim_val_final = ...` (similar with appropriate randomness)
- `claim_rw = ...`
- `claim_val_eval = ...`

These are evaluations of the ra polynomial (which is 1 at access locations) at various random points from earlier stages.

## Verification Results

After the fixes:
- Rounds 0-111: `p(0)+p(1) = claim` ✓ (Instance 0, 1 pre-active + Instance 2)
- Rounds 112-127: `p(0)+p(1) ≠ claim` ✗ (Instance 1 active with wrong claims)

The sumcheck polynomial computation is now correct, but the input claim `ram_ra_input` doesn't match because the opening claims are placeholders.

## Test Commands

```bash
# Build
zig build -Doptimize=ReleaseFast

# Generate proof
./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format -o /tmp/zolt_proof_dory.bin --export-preprocessing /tmp/zolt_preprocessing.bin --trace-length 64

# Verify with Jolt
cd /home/vivado/projects/jolt && cargo test -p jolt-core --features zolt-debug --lib test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture
```

## Architecture Notes

### Instance 1 (RamRaClaimReduction) Structure
- Total rounds: 24 (log_K=16 address + log_T=8 cycle)
- PhaseAddress: rounds 0-15, bind address variables
- PhaseCycle1: rounds 16-19, use P*Q prefix-suffix
- PhaseCycle2: rounds 20-23, bind suffix with H'*eq_hi

### Opening Claim Formula
```
ram_ra_input = claim_raf + γ·claim_val_final + γ²·claim_rw + γ³·claim_val_eval
```

Each claim_X is the evaluation of the ra polynomial MLE at (r_address_X, r_cycle_X).

For single access at (addr=2049, cycle=54):
```
claim_raf = eq(r_address_raf, 2049) * eq(r_cycle_raf, 54)
```

But currently this is set to 0 as a placeholder.
