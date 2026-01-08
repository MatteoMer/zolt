# Zolt-Jolt Compatibility - Stage 2 Round 21+ Divergence

## Current Status: Investigating Round 21 Soundness Failure

### What Works (Verified)
- [x] Stage 1 sumcheck proof generation matches Jolt
- [x] Stage 2 initial batched_claim matches
- [x] Stage 2 input_claims for all 5 instances match
- [x] Stage 2 batching_coeffs match
- [x] Stage 2 tau_high matches (transcript state alignment confirmed)
- [x] Round 0-20 coefficients (c0, c2, c3) match Jolt
- [x] Round 0-20 challenges match Jolt
- [x] Round 21 c0, c2, c3 coefficients match Jolt

### Current Problem: Round 21+ Soundness Failure

At round 21, Zolt's `combined_evals[0] + combined_evals[1]` does NOT equal `old_claim`:
- old_claim = 19220444789179267739542873996735695198911802587512205544750095726067823719616
- Zolt s(0)+s(1) = 16986136980477769455519983177407049670037175512723289430257197891323522321127

Yet the compressed coefficients (c0, c2, c3) match Jolt exactly!

### Root Cause Analysis

The compressed coefficients (c0, c2, c3) are correct, but the combined_evals are wrong.
This suggests the issue is in how we convert from compressed coefficients back to evaluations.

The compressed format stores [c0, c2, c3] where c1 is implied by: c1 = claim - 2*c0 - c2 - c3

When computing combined_evals, we should be summing evaluations from all active instances,
not reconstructing them from compressed coefficients.

### Key Insight

At round 21:
- Instance 0 (ProductVirtual): Active (started at round 16)
- Instance 1 (RAF): Active (started at round 10)
- Instance 2 (RWC): Active (started at round 0)
- Instance 3 (Output): Active (started at round 10)
- Instance 4 (Instr): Active (started at round 16)

All instances are active. The combined_evals should be the sum of each instance's evals.

The issue might be in:
1. The ProductVirtualRemainder prover's round polynomial computation
2. Or one of the other instance provers contributing wrong values
3. Or the individual claims not being updated correctly between rounds

### Files to Investigate
- `src/zkvm/proof_converter.zig` - Stage 2 batched sumcheck generation
- `src/zkvm/spartan/product_remainder.zig` - ProductVirtualRemainder prover
- `src/zkvm/ram/read_write_checking.zig` - RWC prover
- `src/zkvm/claim_reductions/instruction_lookups.zig` - Instruction lookups prover

### Test Commands
```bash
# Build and test Zolt
zig build test --summary all

# Generate proof with debug output
./zig-out/bin/zolt prove --jolt-format -o /tmp/zolt_proof_dory.bin examples/fibonacci.elf 2>&1 | grep -E "STAGE2_ROUND_21|CLAIM.*round 21"

# Verify with Jolt
cd /Users/matteo/projects/jolt/jolt-core
cargo test test_verify_zolt_proof -- --ignored --nocapture 2>&1 | grep "STAGE2_ROUND_21"
```

### Next Steps

1. Check each instance's individual contribution at round 21
2. Verify instance claim updates are using correct polynomial evaluations
3. Trace through the ProductVirtualRemainder prover for rounds 16-21
4. Verify soundness constraint s(0)+s(1)=claim holds for each individual instance
