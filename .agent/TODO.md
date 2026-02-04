# Zolt-Jolt Compatibility Implementation

## Status: Session 51+ - Transcript Divergence Investigation

## Current Investigation - Stage 4 Verification Failure

### Key Finding: Transcript Diverges Early

The root cause of Stage 4 failure is that the transcript diverges much earlier than Stage 4:

1. **Stage 3 challenges don't match**:
   - Jolt Stage 3 Round 0: `[dc, b1, f1, 16, ...]`
   - Zolt Stage 3 Round 0: `[11, 90, 20, c3, ...]` (completely different!)

2. **Stage 1 challenges are different** (need to verify)

3. **Stage 4 batching coefficients don't match** (consequence of earlier divergence):
   - Jolt: `[53, dd, 21, 20, ...]`
   - Zolt: different

### What Was Fixed This Session

1. **`evaluateCubicAtChallengeFromEvals`** - Was using Lagrange interpolation for points {0, 1, 2, 3}, but input is in Toom-Cook format `[p(0), p(1), p(2), p_inf]`. Now correctly converts to coefficients using `toomCookToCoeffs` first.

2. **Stage 4 Round 0 coefficients** - Verified that Zolt's c0, c2, c3 values MATCH what Jolt reads. The polynomial computation itself is correct.

### Root Cause Analysis

The transcript includes all proof data from all stages. If any earlier stage produces different polynomial coefficients, the transcript state diverges and all subsequent challenges differ.

**Likely sources of divergence:**
1. Stage 1 (R1CS sumcheck) - polynomial coefficients might differ
2. Stage 2 (Product virtualization) - polynomial coefficients might differ
3. Stage 3 (Lasso lookup) - polynomial coefficients might differ
4. Or earlier: preprocessing data, initial commitments, etc.

### How to Debug

Need to compare Zolt and Jolt round polynomial coefficients starting from Stage 1 Round 0:

```bash
# Get Zolt's Stage 1 Round 0 coefficients
./zig-out/bin/zolt prove ... 2>&1 | grep "STAGE1_ROUND_0"

# Compare with what Jolt verifier reads from proof
# (need to add debug output in Jolt's sumcheck verifier for Stage 1)
```

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

- `/home/vivado/projects/zolt/src/zkvm/proof_converter.zig` - All stage proof generation
- `/home/vivado/projects/zolt/src/zkvm/spartan/stage4_gruen_prover.zig` - Stage 4 sumcheck
- `/home/vivado/projects/jolt/jolt-core/src/subprotocols/sumcheck.rs` - Jolt sumcheck verifier

## Next Session Should Focus On

1. Add debug output to compare Stage 1 Round 0 coefficients between Zolt and Jolt
2. Find the first round where coefficients diverge
3. Fix that stage's polynomial computation
4. Repeat until all stages match

## SESSION_ENDING

Substantial progress made:
- Fixed evaluateCubicAtChallengeFromEvals bug
- Confirmed Stage 4 polynomial coefficients are correct
- Identified that transcript diverges before Stage 4 (likely Stage 1, 2, or 3)
- Next step is to find which stage first produces wrong coefficients
