# Zolt-Jolt Compatibility TODO

## Current Status: Session 31 - January 2, 2026

**All 702 tests pass**

### Issue: Transcript state diverges between Zolt and Jolt

The debug output shows completely different challenges:
- Zolt's r0 challenge: `5140254016165682325926826379386165292`
- Jolt's r0 challenge: `6919882260122427158724897727024710502508333642996197054262116261168391078818`

This means the transcript state BEFORE the stage 1 sumcheck is different.

### Root Cause Analysis

The r0 challenge is derived from the transcript after:
1. Initial label ("Jolt")
2. All polynomial commitments (GT elements for Dory)
3. UniSkip first-round polynomial coefficients

Since the challenges differ, one of these must be serialized differently:
- GT element serialization (384 bytes, reversed)
- UniSkip polynomial coefficients
- Missing or extra data in transcript

### Completed Work (This Session)

1. [x] Added detailed debug output to compare Zolt and Jolt values
2. [x] Verified round polynomial coefficient computation
3. [x] Confirmed interpolation formula is correct

### Investigation Needed

1. **Compare GT element bytes** - Verify toBytes() matches arkworks CanonicalSerialize
2. **Trace full transcript sequence** - Log every append before stage 1
3. **Compare UniSkip coefficients** - Verify these match between Zolt and Jolt

### Blocking Issues

1. **Dory proof generation panic** - index out of bounds in openWithRowCommitments
   - `params.g2_vec` has length 64 but `current_len` is 128
   - Need to fix SRS loading or verify polynomial degree bounds

## Pending Tasks
- [ ] Fix transcript divergence (root cause of verification failure)
- [ ] Fix Dory proof generation (SRS size issue)
- [ ] Complete remaining stages (2-7) proof generation
- [ ] Create end-to-end verification test with Jolt verifier

## Verified Correct
- [x] Blake2b transcript implementation format
- [x] Field serialization (Arkworks format)
- [x] UniSkip polynomial generation logic
- [x] R1CS constraint definitions (19 constraints, 2 groups)
- [x] Split eq polynomial factorization (E_out/E_in tables)
- [x] Lagrange kernel L(tau_high, r0)
- [x] MLE evaluation for opening claims
- [x] Gruen cubic polynomial formula
- [x] r_cycle computation (big-endian, excluding r_stream)
- [x] eq polynomial factor matches verifier
- [x] Streaming round sum-of-products structure
- [x] ExpandingTable (r_grid) matches Jolt
- [x] DensePolynomial.bindLow() implementation
- [x] Linear phase infrastructure (az_poly, bz_poly, binding)
- [x] All 702 Zolt tests pass

## Test Commands
```bash
# Zolt tests
zig build test --summary all

# Generate proof
zig build -Doptimize=ReleaseFast && ./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format -o /tmp/zolt_proof_dory.bin

# Jolt verification test
cd /Users/matteo/projects/jolt
cargo test --package jolt-core test_verify_zolt_proof -- --ignored --nocapture
```
