# Zolt-Jolt Compatibility TODO

## Current Status: Session 50 - January 3, 2026

**FIXED: from_bigint_unchecked interpretation**

Session 49 fixed the challenge representation:
- arkworks' `from_bigint_unchecked` treats input as ALREADY in Montgomery form
- Zolt now stores `[0, 0, low, high]` directly as limbs (no toMontgomery call)
- tau values now match between Zolt and Jolt

**CURRENT ISSUE: Round number offset**

After r0, there's a 1-round offset in transcript operations:

| Operation | Zolt round | Jolt round |
|-----------|------------|------------|
| r0 challenge | 55 | 55 |
| UniPoly_begin | 58 | 59 |
| UniPoly_end | 62 | 63 |
| 1st sumcheck challenge | 63 | 64 |

This suggests Jolt does one extra operation between r0 and Stage 1 sumcheck.

---

## IMMEDIATE NEXT STEPS

### 1. Debug Transcript State Divergence

Compare transcript states at each operation between r0 and UniPoly_begin:
- Jolt: UniPoly_begin at round=59, state_before=[1c, b7, 03, 0d, 14, 2d, 44, 65]
- Zolt: UniPoly_begin at round=58, need to capture state_before

**Key Question**: What operation does Jolt do at round 56-58 that Zolt is missing?

### 2. Check Batched Sumcheck Flow

Jolt's BatchedSumcheck::prove:
1. For each sumcheck instance: `transcript.append_scalar(&input_claim)`
2. Get batching_coeffs: `transcript.challenge_vector(len)`
3. Process round polynomials

Zolt's proof_converter.zig (lines 442-446):
1. `transcript.appendScalar(uni_skip_claim)` - this is round 56
2. `batching_coeff = transcript.challengeScalar()` - this is round 57

Does Jolt do something extra here?

### 3. Add Debug Output

Add state_before logging to Zolt's appendMessage for UniPoly_begin to compare with Jolt's [1c, b7, 03, 0d, 14, 2d, 44, 65].

---

## Test Commands

```bash
# Build and generate proof
zig build -Doptimize=ReleaseFast
./zig-out/bin/zolt prove /tmp/jolt-guest-targets/fibonacci-guest-fib/riscv64imac-unknown-none-elf/release/fibonacci-guest --jolt-format --input-hex 32 -o /tmp/zolt_proof_dory.bin

# Jolt verification test
cd /Users/matteo/projects/jolt
cargo test --package jolt-core test_verify_zolt_proof -- --ignored --nocapture
```

---

## Previous Sessions Summary

- **Session 50**: Found round number offset between Zolt and Jolt after r0
- **Session 49**: Fixed from_bigint_unchecked interpretation - tau values now match
- **Session 48**: Fixed challenge limb ordering, round polynomials now match
- **Session 47**: Fixed LookupOutput for JAL/JALR, UniSkip first-round now passes
- **Session 46**: Fixed memory_size mismatch, transcript states now match
- **Session 45**: Fixed RV64 word operations, fib(50) now works
