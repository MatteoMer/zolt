# Zolt-Jolt Compatibility Implementation

## Status: Session 56 - Stage 4 Batched Sumcheck Debugging

## Progress This Session

### Key Discovery: Batched Sumcheck Individual Claim Tracking
- Fixed the batched sumcheck implementation to properly track individual claims for each instance
- For inactive instances (before offset round), the claim evolves as: `claim' = claim / 2`
- The constant polynomial `H(X) = claim/2` is used for inactive rounds
- This matches Jolt's "front-loaded" batching approach

### Current Issue: ValFinal Prover Produces Zero Polynomials
After fixing the claim tracking, the sumcheck still fails at Round 7 (when val_eval/val_final activate):
```
[ZOLT STAGE4 CHECK FAIL] Round 7: p(0)+p(1) != batched_claim!
  p(0) = non-zero
  p(1) = 0
```

The val_final prover produces all-zero evaluations (`val_final_evals[0] = val_final_evals[1] = 0`).

**Root cause analysis:**
- ValFinal sumcheck proves: `Σ_j Inc(j) * wa(r_address, j) = input_claim`
- For Fibonacci (no RAM writes): `Inc(j) = 0` for all j
- So the polynomial sum is 0, but `input_claim = Val_final - Val_init ≠ 0` (due to termination bit)

**Question:** How does Jolt handle this apparent inconsistency?
- The input_claim in Jolt is also non-zero (`[4e, e5, 52, 92, ...]`)
- But the polynomial being summed is all zeros
- Need to investigate how Jolt's prover handles this case

### Files Modified This Session

- `/home/vivado/projects/zolt/src/zkvm/proof_converter.zig`:
  - Added `individual_claims` array to track each instance's claim across rounds
  - Fixed inactive instance contribution: uses `claim/2` constant polynomial
  - Fixed claim updates: inactive instances halve their claim each round

## Next Steps

1. **Investigate Jolt's ValFinal prover** to understand how it handles the case where
   the polynomial is all zeros but the input_claim is non-zero

2. **Check if the input_claim formula is different** between prover and verifier sides

3. **Possible resolution**: The input_claim might need to account for the fact that
   the sumcheck is proving something different than the raw polynomial sum

### Test Commands

```bash
# Build Zolt
zig build -Doptimize=ReleaseFast

# Generate proof
./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format -o /tmp/zolt_proof_dory.bin --export-preprocessing /tmp/zolt_preprocessing.bin --trace-length 64

# Verify with Jolt
cd /home/vivado/projects/jolt && cargo test -p jolt-core --features zolt-debug --lib test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture
```

### Key Files

- `/home/vivado/projects/zolt/src/zkvm/proof_converter.zig` - Stage 4 batched sumcheck
- `/home/vivado/projects/zolt/src/zkvm/ram/val_final.zig` - ValFinal prover
- `/home/vivado/projects/jolt/jolt-core/src/zkvm/ram/val_final.rs` - Jolt's ValFinal impl
