# Zolt-Jolt Compatibility TODO

## Current Status: Session 61 - January 6, 2026

**STATUS: Sumcheck Passes, Opening Claims Wrong**

### ✅ Confirmed Working

1. **Transcript Compatibility**: Keccak256 transcript perfectly matches Jolt
2. **Preamble**: Memory layout, inputs, outputs, panic, ram_K, trace_length all match
3. **Dory Commitments**: GT element serialization matches
4. **Tau Derivation**: All 12 tau values match exactly
5. **UniSkip Polynomial**: All 28 coefficients match exactly
6. **r0 Challenge**: Matches exactly after UniSkip polynomial appended
7. **Batching Coefficient**: Matches exactly
8. **All Sumcheck Rounds (0-10)**: Transcript states match at every round
9. **All Sumcheck Challenges**: Every challenge matches between Zolt and Jolt
10. **Output Claim**: Zolt's output_claim (scaled) matches Jolt's exactly

### ❌ Current Issue: Opening Claims Wrong

**The Problem:**

Verification fails because Zolt's opening claims (r1cs_input_evals) don't match Jolt's expected values:

| Input | Zolt Value | Jolt Expected |
|-------|------------|---------------|
| LeftInstructionInput | 111671539584291221291839977129744267341086976218214076725283565100174391436821 | 9906325628809186578319307187123224683532195550528121667958235291543194428406 |

**Root Cause:**

Opening claims are MLE evaluations at r_cycle. The issue is in how `computeClaimedInputs` evaluates the MLE.

**Key Observations:**

1. **r_cycle construction is correct** - Challenges are reversed properly from LITTLE_ENDIAN to BIG_ENDIAN
2. **Witness values look correct** - First cycle has sensible values
3. **The EqPolynomial evaluation may be wrong** - Either the indexing or the accumulation

### Next Steps

1. **Add debug in MLE evaluation** - Print eq_evals for specific cycles and compare with Jolt
2. **Verify cycle-to-hypercube mapping** - Check if cycle 0 maps to (0,0,...,0) or something else
3. **Compare EqPolynomial semantics** - Ensure Zolt's "big-endian indexing" matches Jolt's `mle_endian`

### Technical Details

**Jolt's r_cycle (from outer.rs):**
```rust
let r_cycle = challenges[1..].to_vec();  // Skip r0
OpeningPoint::<LITTLE_ENDIAN, F>::new(r_cycle).match_endianness()  // Reverse for BIG_ENDIAN
```

**Zolt's r_cycle (from proof_converter.zig):**
```zig
const cycle_challenges = all_challenges[1..];  // Skip r_stream
for (0..cycle_challenges.len) |i| {
    r_cycle_big_endian[i] = cycle_challenges[cycle_challenges.len - 1 - i];  // Reverse
}
```

The construction appears identical but MLE evaluation produces different results.

### Files to Investigate

- `/Users/matteo/projects/zolt/src/zkvm/r1cs/evaluation.zig` - `computeClaimedInputs`
- `/Users/matteo/projects/zolt/src/poly/mod.zig` - `EqPolynomial.evals`
- `/Users/matteo/projects/jolt/jolt-core/src/poly/multilinear.rs` - Jolt's MLE semantics

## Test Commands

```bash
# Build and generate proof
zig build -Doptimize=ReleaseFast
./zig-out/bin/zolt prove /tmp/jolt-guest-targets/fibonacci-guest-fib/riscv64imac-unknown-none-elf/release/fibonacci-guest --jolt-format --input-hex 32 --export-preprocessing /tmp/zolt_preprocessing.bin -o /tmp/zolt_proof_dory.bin

# Jolt verification test
cd /Users/matteo/projects/jolt
cargo test --package jolt-core test_verify_zolt_proof -- --ignored --nocapture
```
