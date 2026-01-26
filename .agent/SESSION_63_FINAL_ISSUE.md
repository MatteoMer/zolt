# Session 63 Final Issue: Stage 4 Proof Serialization Bug

## Date
2026-01-26

## Summary

After successfully fixing Instances 1 and 2 with the termination bit workaround, we discovered that **Stage 4 verification still fails** due to a proof serialization/deserialization mismatch.

## Current Status

✅ **Instance Claims - ALL VALID:**
- Instance 0 (RegistersRWC): Zolt computes `input_claim_registers = {4, 68, 118, 36...}` (non-zero)
- Instance 1 (RamValEvaluation): `expected_claim = 0` in Jolt verifier ✅
- Instance 2 (RamValFinalEvaluation): `expected_claim = 0` in Jolt verifier ✅

❌ **Final Output Claim - MISMATCH:**
```
Zolt computes:  Final batched_claim = 11401707430797178799080047434534659298949363068571747577761899350971462131125
Jolt reads:     output_claim        = 2794768927403232170685203001712134750206965869554042859404932801547924672323
Jolt expects:   expected_output     = 19036722498929976088547735251378923562016308482664214076291639064331774676064
```

**All three values are different!**

## Root Cause Analysis

### How Jolt's Batched Sumcheck Works

1. **Input Claims are NOT stored in the proof**
   - The `SumcheckInstanceProof` only contains `compressed_polys` (polynomial coefficients)
   - Input claims are recomputed by the verifier from `opening_accumulator` (populated from `proof.opening_claims`)

2. **Output Claim is NOT stored either**
   - The verifier computes `output_claim` by evaluating all sumcheck round polynomials
   - Starting with the batched input_claim, it applies each round's polynomial evaluation
   - The final value after all rounds is the `output_claim`

3. **Verification Check**
   ```rust
   let output_claim = proof.verify(batched_input_claim, ...);  // Computed from compressed_polys
   let expected_output_claim = sum of (instance.expected_output_claim() * coeff);
   assert_eq!(output_claim, expected_output_claim);
   ```

### The Mystery Value

The value `2794...` that Jolt reads as `output_claim` does NOT appear anywhere in Zolt's prover output. This means:

**Option A**: Jolt is evaluating the compressed_polys incorrectly
**Option B**: The compressed_polys in the proof don't match what Zolt computed
**Option C**: There's a deserialization bug where Jolt reads wrong data

### Historical Context

This bug **existed before** our termination bit fixes:
- Commit `da51b7d` (before fixes): Final batched_claim = `6203...` ≠ Jolt reads `2794...`
- Current commit (after fixes): Final batched_claim = `11401...` ≠ Jolt reads `2794...`

Both commits fail Stage 4 with the same Jolt-side `output_claim = 2794...`, but different Zolt-side values.

## What We've Ruled Out

❌ **Not an input_claim issue** - All 3 instances have correct expected_claims
❌ **Not a termination bit issue** - Bug existed before our fixes
❌ **Not a transcript issue** - input_claims are correctly appended to transcript
❌ **Not an opening_claims issue** - Those are separate from output_claim

## Suspected Issues

### 1. Compressed Polynomial Serialization

The `compressed_polys` might not be serialized/deserialized correctly. Each round should have coefficients `[c0, c2, c3]` (for degree-3 polynomials), but perhaps:
- Zolt writes coefficients in wrong order
- Byte endianness mismatch (LE vs BE)
- Missing or extra coefficients

### 2. Round Count Mismatch

Stage 4 should have:
- Instance 0 (RegistersRWC): 15 rounds (log_registers=7 + log_T=8)
- Instance 1 (RamValEval): 8 rounds (log_K=8 for RAM addresses)
- Instance 2 (RamValFinal): 8 rounds

If the proof structure has wrong round counts, evaluation would fail.

### 3. Evaluation Formula Mismatch

Jolt evaluates: `new_claim = compressed_poly.eval_from_hint(&current_claim, &challenge)`

If Zolt's `eval_from_hint` implementation differs from Jolt's, the claims would diverge.

## Next Steps to Debug

### Step 1: Verify Compressed Polynomial Contents

Add debug output to see what gets written vs read:

```zig
// In proof_converter.zig, after addRoundPoly:
std.debug.print("[ZOLT STAGE4 SERIALIZE] Round {}: c0={any}, c2={any}, c3={any}\n",
    .{round_idx, compressed[0].toBytesBE(), compressed[1].toBytesBE(), compressed[2].toBytesBE()});
```

```rust
// In Jolt's sumcheck.rs verify(), print what's read:
eprintln!("[JOLT STAGE4 DESERIALIZE] Round {}: c0={:?}, c2={:?}, c3={:?}",
    i, coeffs[0], coeffs[1], coeffs[2]);
```

Compare to see if they match.

### Step 2: Check Proof File Structure

Examine the binary proof file to see if Stage 4 data is in the right location:

```bash
hexdump -C /tmp/zolt_proof_dory.bin | less
```

Look for the compressed polynomial coefficients and verify their positions.

### Step 3: Add Eval Tracing

Trace the evaluation step-by-step:

```rust
// In Jolt's verify():
for each round:
    let poly_at_0 = compressed_poly.eval(0);
    let poly_at_1 = compressed_poly.eval(1);
    eprintln!("Round {}: p(0)={}, p(1)={}, sum={}", i, poly_at_0, poly_at_1, poly_at_0 + poly_at_1);
    assert_eq!(poly_at_0 + poly_at_1, current_claim, "Sumcheck constraint failed");
```

This would catch if the polynomial coefficients don't satisfy the sumcheck constraint.

### Step 4: Compare Jolt Prover vs Zolt Prover

Generate a proof with Jolt's native prover and compare the Stage 4 compressed_polys with Zolt's:

```bash
# Generate Jolt proof
cd /Users/matteo/projects/jolt/jolt-core
cargo run --release --bin jolt prove examples/fibonacci.elf -o /tmp/jolt_proof.bin

# Add debug output to compare compressed_polys
```

## Files to Investigate

1. **Zolt Side:**
   - `src/zkvm/proof_converter.zig:1948` - where `addRoundPoly()` is called
   - `src/zkvm/spartan/stage4_gruen_prover.zig` - registers sumcheck prover
   - `src/poly/uni_poly.zig` - polynomial evaluation/compression

2. **Jolt Side:**
   - `/Users/matteo/projects/jolt/jolt-core/src/subprotocols/sumcheck.rs:150-220` - verify() method
   - `/Users/matteo/projects/jolt/jolt-core/src/poly/unipoly.rs` - CompressedUniPoly
   - `/Users/matteo/projects/jolt/jolt-core/src/zkvm/batched_sumcheck.rs` - batched verification

## Conclusion

The termination bit fixes for Instances 1 and 2 are **working correctly**. The remaining Stage 4 failure is a **separate serialization bug** that existed before our changes. The proof structure for Stage 4's compressed polynomials is either:
1. Being written incorrectly by Zolt
2. Being read incorrectly by Jolt
3. Using a different evaluation formula

This requires careful byte-level debugging of the proof serialization format.
