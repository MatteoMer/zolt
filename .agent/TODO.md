# Zolt-Jolt Compatibility Implementation

## Status: Session 107 - Found transcript divergence in Stage 2

### Critical Finding

**The transcript state diverges between Zolt prover and Jolt verifier starting at Stage 2, Round 0.**

Jolt verifier's Stage 2 challenge[0] (derived from verifying proof):
```
[00, 00, ..., ca, e3, 15, 54, c5, e0, 25, 42, 7d, 85, 67, cf, 78, d9, 73, 1e]
```

Zolt prover's Stage 2 challenge[0] (used during proving):
```
{ 119, 212, 208, 55, ... }  = 77, d4, d0, 37, ...
```

These are completely different! This means the transcript state at the end of Stage 1 is different.

### Root Cause

The Jolt verifier re-derives ALL challenges by hashing the proof's round polynomials. If Zolt's serialized proof produces different transcript hashes, challenges diverge.

Possible causes:
1. **Stage 1 round polynomial mismatch**: The coefficients Zolt SENDS don't match what it USED internally
2. **Serialization format mismatch**: Field elements serialized differently than Jolt expects
3. **Missing transcript data**: UniSkip proof or other data not matching
4. **Coefficient order**: Coefficients might be in wrong order

### Stage 4 Failure Explained

Stage 4 fails as a CONSEQUENCE of Stage 2 divergence:
1. Jolt verifier derives different Stage 2 challenges than Zolt used
2. `cache_openings` stores wrong r_cycle in accumulator
3. Stage 4's ValEvaluation retrieves wrong r_cycle
4. `LT(r, r_cycle)` computes wrong value → sumcheck mismatch

### Next Steps

1. Compare Stage 1 round polynomial coefficients between:
   - What Zolt serializes to the proof
   - What Jolt reads from the proof
2. Check transcript state after commitments are added (before Stage 1)
3. Verify UniSkip first round proof handling matches

### Test Commands

```bash
# Generate proof
./zig-out/bin/zolt prove examples/fibonacci.elf --trace-length 64 -o /tmp/zolt_proof.bin --jolt-format 2>&1 | head -200

# Verify with Jolt
cd ../jolt && cargo run --release --features zolt-debug --manifest-path examples/fibonacci/Cargo.toml -- --verify-zolt-proof /tmp/zolt_proof.bin 2>&1 | grep -E "Stage.*1.*Round.*0|coeff" | head -20
```

### Previous Sessions

- Session 107: Found transcript divergence at Stage 2, Round 0
- Session 106: Verified challenge representation is correct
- Session 105: Identified ValEval/ValFinal input_claim mismatch
- Session 104: Fixed proof serialization format (--jolt-format flag)
