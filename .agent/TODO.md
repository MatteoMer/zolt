# Zolt-Jolt Compatibility TODO

## Current Status: Session 48 - January 3, 2026

**FIXED: Challenge limb ordering bug in transcript**

Session 48 fixed a critical bug where 128-bit challenges were stored in the wrong limbs:
- **Bug**: In `src/transcripts/blake2b.zig`, challenges were stored as `[0, 0, low, high]`
- **Fix**: Changed to `[low, high, 0, 0]` - the 128-bit value belongs in limbs[0:1], not limbs[2:3]
- **Impact**: The old code made challenge values 2^128 times too large

**CURRENT STATUS: Round polynomials match, but challenges still diverge**

After the fix:
- Round polynomial coefficients (c0, c2, c3) now match byte-for-byte between Zolt and Jolt
- Transcript states match at many checkpoints during proof generation
- BUT the computed challenges still differ between Zolt prover and Jolt verifier

---

## Key Finding: Challenge Representation Issue

The remaining issue is likely in how challenges are USED, not how they're stored:

1. Jolt's `MontU128Challenge` stores value as `[0, 0, low, high]` internally
2. When converted to Fr via `Into<Fr>`, it uses `from_bigint_unchecked(BigInt::new([0, 0, low, high]))`
3. This creates an Fr representing `(low * 2^128 + high * 2^192) mod p`
4. Zolt's challenge is stored as `[low, high, 0, 0]` and converted to Montgomery form

The challenge VALUES in Zolt are correct (128-bit), but Jolt may expect them in the "shifted" format for certain operations.

---

## IMMEDIATE NEXT STEPS

### 1. Verify Challenge Usage in Sumcheck

Check how challenges are used in Jolt's sumcheck:
```rust
// In sumcheck.rs line 349:
let r_i: F::Challenge = transcript.challenge_scalar_optimized::<F>();
// This returns MontU128Challenge, NOT Fr

// When used in eval_from_hint:
let new_e = self.compressed_polys[i].eval_from_hint(&e, &r_i);
```

The `eval_from_hint` takes `&r_i` which is a `MontU128Challenge`. Does it use the challenge directly or convert to Fr first?

### 2. Check eval_from_hint Implementation

Look at `jolt-core/src/poly/unipoly.rs`:
```rust
pub fn eval_from_hint<C: Into<F> + Copy>(&self, hint: &F, r: &C) -> F
```

The `C: Into<F>` means the challenge is converted to Fr. The conversion uses `from_bigint_unchecked([0, 0, low, high])`.

### 3. Fix: Zolt Must Use Same Shifted Representation

**CRITICAL**: When Zolt binds a challenge `r` in polynomial evaluation, it must use the SAME field element value that Jolt uses:

- Jolt: `Fr::from_bigint([0, 0, low, high])` = `(low * 2^128 + high * 2^192) mod p`
- Zolt current: `Fr.toMontgomery([low, high, 0, 0])` = `(low + high * 2^64) * R mod p`

These produce DIFFERENT field element values!

### 4. Proposed Fix

In `src/transcripts/blake2b.zig`, change:
```zig
// WRONG - gives (low + high * 2^64) as a field element
const raw = F{ .limbs = .{ masked_low, masked_high, 0, 0 } };
const result = raw.toMontgomery();

// CORRECT - gives (low * 2^128 + high * 2^192) as a field element
// This matches Jolt's from_bigint([0, 0, low, high])
const raw = F{ .limbs = .{ 0, 0, masked_low, masked_high } };
const result = raw.toMontgomery();
```

Wait - this is what we HAD before! The issue is that `toMontgomery()` and Jolt's `from_bigint` may do different things.

### 5. ACTUAL FIX NEEDED

The real fix is to understand the difference between:
- Zolt: `F.toMontgomery([0, 0, low, high])`
- Jolt: `Fr::from_bigint_unchecked(BigInt::new([0, 0, low, high]))`

In arkworks, `from_bigint` takes a standard integer BigInt and converts to Montgomery form.
In Zolt, `toMontgomery()` takes limbs in standard form and converts to Montgomery form.

Both SHOULD give the same result for the same input limbs. Need to verify this.

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

## Session 48 Debug Output Summary

After the fix, comparing Round 0 values:

**c0 bytes (MATCH):**
- Zolt: `{ 187, 132, 95, 37, 236, 70, 160, 14, ... }`
- Jolt: `[187, 132, 95, 37, 236, 70, 160, 14, ...]`

**Challenge (DON'T MATCH):**
- Zolt (128-bit, LE padded): `{ 204, 58, 130, 67, ..., 0, 0, 0, 0 }`
- Jolt (full Fr): `[84, 142, 218, 8, 93, 53, 21, 159, ...]`

The Jolt challenge has non-zero bytes in ALL 32 positions because it's a full Fr, while Zolt only has 16 non-zero bytes (128-bit challenge).

---

## Previous Sessions Summary

- **Session 48**: Fixed challenge limb ordering, round polynomials now match, challenges still diverge
- **Session 47**: Fixed LookupOutput for JAL/JALR, UniSkip first-round now passes
- **Session 46**: Fixed memory_size mismatch, transcript states now match
- **Session 45**: Fixed RV64 word operations, fib(50) now works
