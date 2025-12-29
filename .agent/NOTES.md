# Zolt-Jolt Compatibility Notes

## Current Status (December 29, 2024)

### Session 13 - Stage 1 Remaining Rounds Deep Dive

**Status**: UniSkip passes, all sumcheck rounds pass (p(0)+p(1)=claim), but final output_claim ≠ expected.

**Key Finding - Multiquadratic Product of Slopes:**
The Gruen method requires:
- `t'(0)` = Σ eq * Az(0) * Bz(0) = sum of products at position 0
- `t'(∞)` = Σ eq * (Az(1) - Az(0)) * (Bz(1) - Bz(0)) = product of SLOPES

This is NOT `t'(1) - t'(0)` (slope of product), but the PRODUCT OF SLOPES!

**Previous Bug:** Was computing `t_infinity = t_one - t_zero` (slope of product).
**Fix:** Now computing `slope_az * slope_bz` (product of slopes).

**Remaining Issue - Cycle Round Implementation:**
Current cycle round implementation has fundamental bugs:
1. Computing `(Σ eq * Az) * (Σ eq * Bz)` instead of `Σ (eq * Az * Bz)`
2. Not properly using r_grid weights for bound challenges
3. Splitting cycles into halves physically instead of via eq weights

**Jolt's Approach:**
- Streaming window: Computes grid[j] for j ∈ {0, 1} (window positions)
- Each grid position sums over k ∈ 0..klen where klen = 2^(num_bound_challenges)
- scaled_w[k] = Lagrange(r0) * r_grid[k] combines constraint weighting with bound challenges
- After multiquadratic expansion, products are computed at each grid position

**Two Possible Approaches:**
1. **Streaming approach**: Properly implement Jolt's streaming sumcheck with r_grid weighting
2. **Materialization approach**: After streaming round, materialize Az/Bz as dense polynomials

The materialization approach (like Jolt's LinearStage) may be simpler to get correct.

---

### Session 12 - Stage 1 Remaining Rounds Debugging

**Progress Made:**
1. Fixed constraint group indices to match Jolt's FIRST_GROUP/SECOND_GROUP ordering
2. Added ExpandingTable (r_grid) for tracking bound challenge weights
3. Identified multiquadratic slope computation bug

---

## Previous Session (December 28, 2024, Session 11)

### Session 11 - MAJOR BREAKTHROUGH! 🎉

**Stage 1 UniSkip Claims Now Match!**

After extensive debugging, we identified and fixed the challenge derivation to be
compatible with Jolt's `MontU128Challenge` system.

#### Root Cause Analysis

Jolt uses a special `MontU128Challenge` type for 128-bit challenges:
1. `MontU128Challenge::new(value)` stores `[0, 0, low, high]` where `low = value & 0xFFFFFFFFFFFFFFFF` and `high = value >> 64`
2. When converted to `Fr` via `.into()`, it uses `from_bigint_unchecked([0, 0, low, high])` which stores the raw BigInt without Montgomery conversion
3. In polynomial evaluation, `power * coeff` does Montgomery multiplication where `power` is raw and `coeff` is Montgomery form:
   - `REDC(raw * mont) = REDC(raw * val * R) = raw * val mod p`

The key insight: When multiplying a raw BigInt `[0, 0, low, high]` with a Montgomery-form Fr, the result is correct because:
- The raw value interpreted from `[0, 0, low, high]` is `low * 2^128 + high * 2^192`
- After Montgomery reduction, this gives the correct mathematical result

#### The Fix

1. **Challenge Format**: Store challenges as `[0, 0, low, high]` (raw BigInt format)
   ```zig
   return F{ .limbs = .{ 0, 0, low, high } };
   ```

2. **Byte Interpretation**: Parse reversed 16-byte array as big-endian u128 (matching Rust's `u128::from_be_bytes`)
   ```zig
   const val: u128 = mem.readInt(u128, &reversed, .big);
   ```

3. **Polynomial Evaluation**: Use raw multiplication for challenge evaluation
   ```zig
   // result (Montgomery) * x (raw) = correct result
   result = result.montgomeryMul(x).add(coeffs[i]);
   ```

#### Verification

```
r0 and r0_fr evaluations match: true
Claims match: true
✓ Stage 1 UniSkip verification PASSED!
```

---

## Working Components ✅

| Component | Status | Notes |
|-----------|--------|-------|
| Blake2b Transcript | ✅ Working | State and n_rounds match |
| Challenge Derivation | ✅ Working | MontU128Challenge-compatible |
| Dory Commitment | ✅ Working | GT elements match, MSM correct |
| Proof Structure | ✅ Working | 7 stages, claims, all parse |
| Serialization | ✅ Working | Byte-level compatible |
| UniSkip Algorithm | ✅ Working | Domain sum = 0, claims match |
| Preprocessing Export | ✅ Working | Full JoltVerifierPreprocessing |
| DoryVerifierSetup | ✅ Working | Precomputed pairings |

---

## Session History

### Session 11
- Fixed MontU128Challenge-compatible challenge derivation
- Challenge format: `[0, 0, low, high]` for 128-bit challenges
- Byte interpretation: big-endian u128
- Stage 1 UniSkip claims now match!

### Session 10
- Fixed Montgomery form serialization in `appendScalar`
- Transcript states now match at r0 derivation
- Identified challenge format mismatch

### Session 9
- Stage 1 UniSkip verification passes (domain sum)
- Fixed Lagrange interpolation bug

### Session 7-8
- DoryVerifierSetup implementation
- Full preprocessing export

---

## Key Technical Details

### MontU128Challenge Representation

```
Mathematical value: v = low + high * 2^64 (128-bit integer)
BigInt storage:     [0, 0, low, high] (NOT Montgomery form)
BigInt value:       low * 2^128 + high * 2^192 (as a BigInt)

When used in multiplication:
- coeff is in Montgomery form: coeff_mont = coeff_val * R mod p
- REDC([0,0,l,h] * coeff_mont) = REDC(bigint_val * coeff_val * R)
                                = bigint_val * coeff_val mod p

This works because Montgomery reduction cancels the R factor,
and the multiplication by the bigint value is correct.
```

### Challenge Derivation

```zig
// 1. Get 16 bytes from transcript
var buf: [16]u8 = undefined;
self.challengeBytes(&buf);

// 2. Reverse bytes
var reversed: [16]u8 = undefined;
for (0..16) |i| {
    reversed[i] = buf[15 - i];
}

// 3. Interpret as big-endian u128
const val: u128 = mem.readInt(u128, &reversed, .big);

// 4. Mask to 125 bits (matching MontU128Challenge::new)
const val_masked = val & ((std.math.maxInt(u128)) >> 3);

// 5. Store in Jolt-compatible format
const low: u64 = @truncate(val_masked);
const high: u64 = @truncate(val_masked >> 64);
return F{ .limbs = .{ 0, 0, low, high } };
```

---

## Commands

```bash
# Test Zolt (all 632 tests)
zig build test --summary all

# Generate proof
zig build -Doptimize=ReleaseFast
./zig-out/bin/zolt prove examples/sum.elf --jolt-format -o /tmp/zolt_proof_dory.bin

# Run Jolt debug test
cd /Users/matteo/projects/jolt
cargo test --package jolt-core test_debug_stage1_verification -- --ignored --nocapture
```

---

## Next Steps

1. Verify Stage 1 remaining sumcheck rounds (after UniSkip)
2. Implement Stage 2-7 verification tests
3. Full end-to-end proof verification
4. Performance optimization
