# Zolt-Jolt Compatibility Implementation

## Status: CRITICAL FIX IDENTIFIED - Challenge multiplication mismatch

## Session 8 Progress (Current)

### Root Cause Found!

Jolt uses an **intentionally inconsistent** interpretation for MontU128Challenge:

1. **For Addition/Subtraction** (`F + challenge`, `F - challenge`):
   - Uses `Into<F>::into(challenge)` first
   - This calls `from_bigint_unchecked([0,0,L,H])`
   - Stores `[0,0,L,H]` as Montgomery form → field value = `(L*2^128 + H*2^192) / R`

2. **For Multiplication** (`F * challenge`):
   - Uses `mul_by_hi_2limbs(L, H)` directly
   - Treats `(L, H)` as raw integer → field value = `L*2^128 + H*2^192`
   - NO division by R!

### Why This Matters

When computing `eq(r, s) = Π_i (r[i]*s[i] + (1-r[i])*(1-s[i]))`:

Jolt computes:
- `r[i] * s[i]` using `mul_by_hi_2limbs` → `(L_r*2^128 + H_r*2^192) * (L_s*2^128 + H_s*2^192)`
- `1 - r[i]` using standard subtraction → `1 - (L_r*2^128 + H_r*2^192) / R`

But Zolt was computing:
- `r[i] * s[i]` using standard Montgomery mul → `((L_r*2^128 + H_r*2^192)/R) * ((L_s*2^128 + H_s*2^192)/R)`
- This gives DIFFERENT results!

### The Fix

Zolt needs to match Jolt's EXACT behavior:

1. **Challenge storage**: Store as `[0, 0, L, H]` directly (✓ already done)

2. **Challenge multiplication**: Use `mulHiBigIntU128(challenge.limbs)` instead of `mul(challenge)`
   - Applies to: `computeEqAtIndex`, `split_eq.zig`, expanding table updates, etc.

3. **Challenge subtraction** (`F.one() - challenge`):
   - The current Zolt code does `F.one().sub(challenge)` which treats challenge as Montgomery form
   - But since multiplication uses the raw interpretation, the subtraction should ALSO use raw interpretation?
   - Actually NO - Jolt's subtraction DOES convert first, so both differ by factor R
   - This is INTENTIONAL in Jolt and creates a non-standard eq polynomial!

### Key Files to Modify

1. `src/zkvm/spartan/stage5_prover.zig`:
   - `computeEqAtIndex()` at line 3565: Change `result.mul(rj)` to `result.mulHiBigIntU128(rj.limbs)`

2. `src/poly/split_eq.zig`:
   - Lines 143-144, 166-167: Change `prev[i].mul(tau_k)` and `prev[i].mul(one_minus_tau_k)`
   - Need to use `mulHiBigIntU128` for challenge multiplications

3. `src/utils/expanding_table.zig`:
   - Any challenge multiplication there

4. All sumcheck provers that multiply with challenges

### Verification Steps

1. After fix, rebuild: `zig build -Doptimize=ReleaseFast`
2. Generate proof: `./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format --export-preprocessing /tmp/zolt_preprocessing.bin -o /tmp/zolt_proof_dory.bin`
3. Copy and verify: `cp /tmp/zolt_*.bin /home/vivado/projects/jolt/ && cd /home/vivado/projects/jolt && cargo test --package jolt-core --features zolt-debug test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture`

### Wait - Re-Analysis Needed!

Looking more carefully at Jolt's eq_poly.rs evals_serial:
```rust
for i in (0..size).rev().step_by(2) {
    let scalar = evals[i / 2];
    evals[i] = scalar * r[j];      // F * Challenge → uses mul_by_hi_2limbs
    evals[i - 1] = scalar - evals[i];  // F - F → standard field subtraction!
}
```

So the computation is:
- `scalar * r[j]` = `scalar * (L*2^128 + H*2^192)` (using mul_by_hi_2limbs, treats r[j] as raw int)
- `scalar - (scalar * r[j])` = `scalar * (1 - (L*2^128 + H*2^192))`

NOT `scalar * (1 - r[j])` where r[j] is converted first!

This means the eq polynomial Jolt computes is:
- For challenge r[j] with raw value c = L*2^128 + H*2^192:
- eq factor = b*c + (1-b)*(1-c) when b=1 or b=0
- But this is NOT standard eq polynomial because c is NOT in [0,1]!

Actually, this IS valid because we're working in a field where c is a random field element.
The eq polynomial is: eq(r, b) = Π_i (b_i * r_i + (1-b_i)*(1-r_i))

When b_i = 1: factor = r_i
When b_i = 0: factor = 1 - r_i

In Jolt's implementation with challenges:
- b_i = 1: factor = mul_by_hi_2limbs result = raw_challenge_value
- b_i = 0: factor = scalar - (scalar * challenge) = scalar * (1 - raw_challenge_value)

This IS consistent! The subtraction `scalar - (scalar * r[j])` equals `scalar * (1 - r[j])` where r[j] is the RAW value.

So the eq polynomial uses raw_challenge_value, not the Montgomery-converted value.

### Conclusion

Zolt must use `mulHiBigIntU128` for ALL challenge multiplications in eq polynomial computation.
The "field value" of a challenge is `L*2^128 + H*2^192` (NOT divided by R).

But wait - this contradicts the `Into<F>` implementation which DOES use `from_bigint_unchecked`!

Let me check if `Into<F>` is ever actually called in the hot path...

Actually, looking at the macro again:
- `F - Challenge` uses `self - Into::<$f>::into(rhs)` (line 218)
- But in eq_poly.rs, the subtraction is `scalar - evals[i]` which is `F - F`, not `F - Challenge`!

So the flow is:
1. `scalar * r[j]` → Challenge → F via mul_by_hi_2limbs → stored in evals[i]
2. `scalar - evals[i]` → F - F → standard subtraction

The Challenge is NEVER subtracted directly from F in the eq computation! It's always multiplied first, then the result is subtracted.

This makes sense and is consistent!

### Final Fix Plan

In Zolt:
- Change all `field.mul(challenge)` to `field.mulHiBigIntU128(challenge.limbs)`
- Do NOT change subtraction handling
- The eq polynomial will then match Jolt's interpretation

## Files Changed This Session

(none yet - analysis complete, implementation next)

## Test Commands

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
