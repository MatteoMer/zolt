# Zolt-Jolt Compatibility TODO

## Current Status: Session 40 - January 2, 2026

**712 tests pass. CRITICAL: GT serialization likely causing transcript divergence.**

---

## CRITICAL Finding (Session 40)

### The Problem

The batching coefficient and lagrange_tau_r0 don't match between Zolt and Jolt. This indicates that the transcript state diverged BEFORE Stage 1 remaining sumcheck began.

### Root Cause Analysis

The transcript divergence happens because **GT (Fp12) element serialization** may not match between Zolt and Jolt:

1. **Zolt's `appendGT`**:
   - Calls `gt.toBytes()` which serializes 12 Fp elements in little-endian
   - Order: c0.c0.c0, c0.c0.c1, c0.c1.c0, c0.c1.c1, c0.c2.c0, c0.c2.c1, c1.c0.c0, c1.c0.c1, c1.c1.c0, c1.c1.c1, c1.c2.c0, c1.c2.c1
   - Reverses all 384 bytes
   - Appends to transcript

2. **Jolt's `append_serializable`**:
   - Calls `serialize_uncompressed` (arkworks)
   - Reverses all bytes
   - Appends to transcript

If arkworks uses a different field element ordering within Fp12, or different byte order, the hashes will differ.

### Data Points

**Zolt prover values:**
- `lagrange_tau_r0`: 16352479363158949757474927920495789621963005842526293440633700861589541710157
- `batching_coeff`: 337824298732027351174516659111631235902

**Jolt verifier values:**
- `tau_high_bound_r0`: 8028489090661391714608006371229486480224032252478234922314677496455554319506
- `batching_coeff`: 174319264625250476236973977450622404778

### Verification Needed

1. Confirm arkworks Fp12 serialization order
2. Compare byte-by-byte output of Zolt's GT serialization vs arkworks
3. Fix any ordering discrepancies

---

## Next Steps

1. **Create a GT serialization test**:
   - Compute a known GT value (e.g., pairing of known points)
   - Compare Zolt's serialization with arkworks serialization byte-by-byte
   - Identify any ordering or format differences

2. **Fix GT serialization to match arkworks**

3. **Re-run Jolt verification test**

---

## Test Commands

```bash
# All tests pass
zig build test --summary all

# Generate proof with debug output
zig build -Doptimize=ReleaseFast
./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format -o /tmp/zolt_proof_dory.bin

# Jolt verification (fails at Stage 1)
cd /Users/matteo/projects/jolt
cargo test --package jolt-core test_verify_zolt_proof -- --ignored --nocapture
```
