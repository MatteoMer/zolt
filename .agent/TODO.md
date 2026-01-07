# Zolt-Jolt Compatibility - Session Progress

## Current Status

### ✅ COMPLETED
- Stage 1 passes Jolt verification completely
- All 712 internal tests pass
- All opening claims verified to match Jolt exactly:
  - l_inst, r_inst, is_rd_not_zero, next_is_noop, fused_left, fused_right ✓
  - ra_claim, val_claim, inc_claim ✓
- Instance 0 (ProductVirtualRemainder) final claim matches exactly

### ❌ IN PROGRESS
Stage 2 fails because Instance 2 (RWC) final claim doesn't match.

**Root Cause Identified:**
Our RWC prover's eq polynomial handling is incorrect. The issue is:
1. Phase 1 eq polynomial computation was fixed to properly bind sumcheck challenges
2. Phase 2 still uses stale eq_evals that don't account for binding
3. The overall structure of how we compute round polynomials may not match Jolt's sparse matrix approach

**Current Difference:**
- Our RWC final: 17925181248966282971112807010799772681208014801023116248823233609842789352688
- Jolt expected:  11216823976254905917561036500968546773134980482196871908475958474138871482864

## Next Steps

1. Study Jolt's RWC prover more carefully - understand the GruenSplitEqPolynomial and matrix structure
2. Fix Phase 2 to properly handle the eq polynomial after all cycle variables are bound
3. Verify the sparse matrix iteration matches Jolt's approach
4. Fix the double-free memory bug

## Verification Commands

```bash
# Build and test Zolt
zig build test --summary all

# Generate Jolt-compatible proof
./zig-out/bin/zolt prove --jolt-format -o /tmp/zolt_proof_dory.bin examples/fibonacci.elf

# Verify with Jolt
cd /Users/matteo/projects/jolt/jolt-core
cargo test test_verify_zolt_proof -- --ignored --nocapture
```

## Code Locations
- RWC prover: `/Users/matteo/projects/zolt/src/zkvm/ram/read_write_checking.zig`
- Jolt RWC: `/Users/matteo/projects/jolt/jolt-core/src/zkvm/ram/read_write_checking.rs`
