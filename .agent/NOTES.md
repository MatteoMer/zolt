# Zolt-Jolt Compatibility Notes

## Current Status (Session 6 - January 7, 2026)

### Summary

**Stage 1 PASSES, Stage 2 FAILS at sumcheck verification**

The Stage 2 batched sumcheck produces `output_claim` that doesn't match the verifier's `expected_output_claim`.

### CRITICAL FINDING: Serialization Mismatch

The polynomial coefficients written by Zolt are read as DIFFERENT values by Jolt!

**Zolt writes (Stage 2, round 25, c0):**
- BE bytes: `{41, 233, 194, 132, ...}`
- Value: 18957844668819946272...

**Jolt reads (STAGE1_ROUND_25, c0):**
- LE bytes: `[218, 112, 200, 225, ...]`
- Value: 14124309671825385295...

These are COMPLETELY DIFFERENT values, indicating a serialization/deserialization format mismatch.

### What Works

1. ✅ tau_high values match (transcript synchronized)
2. ✅ Stage 1 challenges match between prover/verifier
3. ✅ Basic field element serialization (toBytes produces LE)

### Root Cause Hypothesis

The proof binary is being read at the wrong offset, OR there's a structural mismatch between how Zolt serializes and how Jolt deserializes.

### Jolt Proof Structure (expected order)

```
1. opening_claims (BTreeMap<OpeningId, F>)
2. commitments (Vec<Commitment>)
3. stage1_uni_skip_first_round_proof
4. stage1_sumcheck_proof (Vec<CompressedUniPoly>)
5. stage2_uni_skip_first_round_proof
6. stage2_sumcheck_proof (Vec<CompressedUniPoly>)
7-11. stage3-7_sumcheck_proof
12. joint_opening_proof
13-17. Optional proofs/commitments
18-22. Configuration values (usize)
```

### Next Steps

1. Add byte-offset tracking to Jolt's deserializer
2. Compare exact byte positions for each section
3. Verify CompressedUniPoly serialization matches arkworks format
4. Check if sumcheck proof sections are at correct offsets

### Debugging Commands

```bash
# Generate proof with debug
./zig-out/bin/zolt prove --jolt-format -o /tmp/proof.bin examples/fibonacci.elf 2>&1 | grep "STAGE2_BATCHED round 25"

# Verify with Jolt
cd /Users/matteo/projects/jolt
cargo test --package jolt-core test_verify_zolt_proof -- --ignored --nocapture 2>&1 | grep "STAGE1_ROUND_25"
```

---

## Previous Session Analysis

The `expected_output_claim` in Jolt is computed as:
```
L(tau_high, r0) * eq(tau_low, r_reversed) * fused_left(claims) * fused_right(claims)
```

This formula was verified to be correct, but the issue is now identified as a serialization problem, not a computation problem.
