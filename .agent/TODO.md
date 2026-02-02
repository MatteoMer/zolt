# Zolt-Jolt Compatibility Implementation

## Status: IN PROGRESS - Found Polynomial Evaluation Bug

## Session 138 Progress

### ROOT CAUSE IDENTIFIED: Lagrange Interpolation Bug

**The Stage 5 sumcheck mismatch is caused by incorrect polynomial evaluation in Zolt.**

In `stage5_prover.zig`, the code computes round polynomials with evaluations `[p(0), p(1), p(2), p_inf]`
where `p_inf = c3` is the leading coefficient (evaluation at infinity).

However, the Lagrange interpolation at lines 300-320 treats these as `[p(0), p(1), p(2), p(3)]`:

```zig
const L3 = r.mul(r_1).mul(r_2).mul(six.inverse().?);  // Lagrange basis for x=3
current_batched_claim = p0.mul(L0).add(p1.mul(L1)).add(p2.mul(L2)).add(p3.mul(L3));
```

**This is wrong!** `combined_poly[3]` is `p_inf = c3`, NOT `p(3)`!

### The Fix

Option A: Use Horner's method on coefficients (matching Jolt's prover):
1. Convert Toom-Cook evaluations `[p(0), p(1), p(2), p_inf]` to coefficients `[c0, c1, c2, c3]`
2. Evaluate at challenge using Horner's method: `c0 + r*(c1 + r*(c2 + r*c3))`

Option B: Convert to proper evaluation points first:
1. Compute `p(3) = c0 + 3*c1 + 9*c2 + 27*c3` from coefficients
2. Then use Lagrange interpolation with `[p(0), p(1), p(2), p(3)]`

Option A is preferred as it matches Jolt's approach.

### How Jolt Does It

In `sumcheck.rs` line 117:
```rust
individual_claims.iter_mut().zip(univariate_polys.into_iter())
    .for_each(|(claim, poly)| *claim = poly.evaluate(&r_j));
```

Where `poly.evaluate()` uses Horner's method on coefficients (see `unipoly.rs` lines 175-192).

### Challenge Format (Already Correct)

Zolt's `challengeScalar128Bits()` already produces the correct format:
- Returns `F{ .limbs = .{ 0, 0, masked_low, masked_high } }`
- WITHOUT calling `toMontgomery()` (matches Jolt's `from_bigint_unchecked`)

The challenge format is NOT the issue - the polynomial evaluation is.

### Key Files to Modify

1. `/home/vivado/projects/zolt/src/zkvm/spartan/stage5_prover.zig` - Fix polynomial evaluation

### Test Commands

```bash
# Build Zolt
zig build -Doptimize=ReleaseFast

# Generate proof
./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format --export-preprocessing /tmp/zolt_preprocessing.bin -o /tmp/zolt_proof_dory.bin

# Verify with Jolt
cp /tmp/zolt_*.bin /home/vivado/projects/jolt/
cd /home/vivado/projects/jolt
cargo test --package jolt-core --features zolt-debug test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture
```

## Previous Sessions Summary

- Session 133: Confirmed challenges match, identified table MLE mismatch
- Session 134: Confirmed table MLEs match at basic level, identified ra_claims mismatch
- Session 135: Confirmed ra_claims serialization is correct, issue is sumcheck polynomial values
- Session 136: Identified prefix-suffix decomposition as potential issue
- Session 137: Identified challenge format as potential issue (but actually correct)
- Session 138: **Found root cause: Lagrange interpolation bug treating p_inf as p(3)**

## Key Files

- `/home/vivado/projects/zolt/src/zkvm/spartan/stage5_prover.zig` - Stage 5 prover (contains bug)
- `/home/vivado/projects/zolt/src/poly/mod.zig` - UniPoly with interpolation functions
- `/home/vivado/projects/jolt/jolt-core/src/poly/unipoly.rs` - Jolt's polynomial evaluation
- `/home/vivado/projects/jolt/jolt-core/src/subprotocols/sumcheck.rs` - Jolt's sumcheck prover
