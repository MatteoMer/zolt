# Zolt-Jolt Compatibility - Stage 2 Progress

## Current Status
- Stage 1: PASSING ✅
- Stage 2: FAILING ❌ (claim mismatch after tau_stage2 fix)
- Stage 3+: Not reached yet

### Latest Values (after tau_stage2 fix)
- output_claim:          14867822501945056124760436278298769782043474221052097072713539181667688027183
- expected_output_claim: 7131681515739848144364178122568074008762470833002613723202290098936339019660

### Tests
- All 712 Zolt tests pass
- Jolt verification failing at Stage 2

## Completed Fixes

### 1. Factor Flags in R1CS (commit fd6e5b7)
- Added FlagIsRdNotZero: 1 if rd register index != 0
- Added FlagBranch: 1 if instruction opcode == 0x63
- Added FlagIsNoop: 1 if this is a noop instruction
- NUM_INPUTS increased from 36 to 39

### 2. Claim Update for ProductVirtualRemainder (commit 85898bb)
- CRITICAL: Added updateClaim(evals, challenge) before bindChallenge()
- Without this, the hint s(1) = current_claim - s(0) uses wrong claim
- This significantly improved the output_claim value

### 3. R_cycle Reversal for BIG_ENDIAN (commit 95b565a)
- Challenges reversed for factor evaluation
- Matches Jolt's OpeningPoint<BIG_ENDIAN> convention

### 4. Correct tau for Stage 2 (commit 7b9d285)
- Stage 2 tau should be [r_cycle_stage1..., tau_high_stage2]
- NOT the original tau from Stage 1
- Matches Jolt's ProductVirtualUniSkipParams::new

## Remaining Issues

The expected_output_claim for ProductVirtualRemainder is computed as:
```
tau_high_bound_r0 * tau_bound_r_tail_reversed * fused_left * fused_right
```

Where:
- fused_left = w[0]*l_inst + w[1]*is_rd_not_zero + w[2]*is_rd_not_zero + w[3]*lookup_out + w[4]*j_flag
- fused_right = w[0]*r_inst + w[1]*wl_flag + w[2]*j_flag + w[3]*branch_flag + w[4]*(1-next_is_noop)
- w[0..4] = Lagrange weights at r0 over 5-point domain
- tau_bound_r_tail_reversed = eq(tau_low, r_cycle_reversed)

The sumcheck round polynomials need to be consistent with this formula.

## Debugging Strategy

1. **Check round polynomial computation** - My current approach computes evaluations directly, not using Gruen's polynomial
2. **Verify tau_low ordering** - tau_low for Stage 2 should be [r_1, r_2, ..., r_n] (cycle challenges from Stage 1)
3. **Compare eq polynomial evaluations** - Ensure eq(tau_low, x) matches Jolt

## Commands

```bash
# Build and test
zig build test --summary all

# Generate proof
./zig-out/bin/zolt prove --jolt-format -o /tmp/proof.bin examples/fibonacci.elf

# Test with Jolt
cp /tmp/proof.bin /tmp/zolt_proof_dory.bin
cd /Users/matteo/projects/jolt
cargo test --package jolt-core test_verify_zolt_proof -- --ignored --nocapture
```
