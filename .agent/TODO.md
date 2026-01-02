# Zolt-Jolt Compatibility TODO

## Current Status: Session 40 - January 2, 2026

**712 tests pass. CRITICAL: transcript state diverges BEFORE Stage 1 remaining sumcheck.**

---

## CRITICAL Finding (Session 40)

### Debug Output Comparison

**Zolt prover values:**
- `lagrange_tau_r0`:  16352479363158949757474927920495789621963005842526293440633700861589541710157
- `batching_coeff`:   337824298732027351174516659111631235902

**Jolt verifier values:**
- `tau_high_bound_r0`: 8028489090661391714608006371229486480224032252478234922314677496455554319506
- `batching_coeff`:    174319264625250476236973977450622404778

### Root Cause Analysis

The **batching_coeff mismatch** proves that the transcript state diverged BEFORE Stage 1 remaining sumcheck. The batching coefficient is generated after:

1. Commitments appended to transcript ← **LIKELY CULPRIT**
2. tau = challenge_vector(num_rows_bits)
3. UniSkip polynomial appended
4. r0 = challengeScalar()
5. uni_skip_claim appended
6. batching_coeff = challengeScalar() ← DIFFERS

### Hypothesis

**Commitments are not being appended to transcript correctly.**

Looking at the code in `convertWithTranscript`, I don't see commitments being appended to transcript. The Jolt verifier expects commitments to be hashed before tau is generated.

Without commitments in the transcript:
- Zolt generates different tau values
- Different tau_high leads to different lagrange_tau_r0
- Different transcript state leads to different batching_coeff

### Next Steps (Priority Order)

1. **Find where Jolt appends commitments to transcript**
2. **Add commitment serialization to Zolt's transcript**
3. **Verify tau values match after fix**
4. **Verify batching_coeff matches after fix**

---

## Earlier Fixes Applied

1. ✅ Batching coefficient applied to round polynomials
2. ✅ Prover internal state uses unscaled claims
3. ✅ Round polynomials scaled for transcript hash

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
