# Performance Plan: Closing the Gap with Jolt

**Current state (collatz):** Zolt 3196ms vs Jolt 1548ms (2.06x)

Flamegraph comparison shows 4 clear bottlenecks accounting for ~1650ms of the gap.

---

## 1. Fp2 Lazy Reduction (~1041ms gap, 55% of Zolt time)

Zolt spends 1766ms on Fp2/Fp6/Fp12 arithmetic vs Jolt's 725ms. `Fp2.mul` is the #1 function by samples.

**Problem:** Every Fp2 mul/add/sub does immediate modular reduction. In sequences like Fp6.mul or Miller loop line functions, intermediate results are added together before the final value is needed — those intermediate reductions are wasted.

**Solution:** Implement "lazy reduction" Fp2:
- Fp2.mul computes `a0*b0`, `a1*b1`, `(a0+a1)*(b0+b1)` using unreduced intermediates (allow values up to 2p)
- Only reduce at the end of a compound operation
- Arkworks does this via `Fp2ConfigWrapper` with `mul_assign` that delays reductions
- Also optimize `mul_by_034` and `mul_by_01` sparse Fp12/Fp6 multiplications

**Impact:** ~30-40% reduction in pairing arithmetic. Affects pairings, GLV G2 scalar muls, and all G2 operations.

**Expected savings:** ~400-500ms on collatz

---

## 2. Batch Affine MSM (~584ms gap)

Zolt spends 686ms on MSM vs Jolt's 102ms. Zolt uses projective coordinates in Pippenger bucket accumulation.

**Problem:** Projective point addition requires ~12 field muls per add. Each bucket accumulation does many adds. The projective→affine conversion at the end also costs inversions.

**Solution:** Implement batch affine addition using Montgomery's trick:
- Accumulate points in affine coordinates
- Use batch inversion (one inversion + O(n) muls for n additions) instead of per-addition projective ops
- This is Jolt's `batch_g1_additions_multi` approach
- Affine add is ~3 field muls + 1 inversion, but batched the inversion is amortized to ~3 muls

**Also:** Implement G2 Pippenger MSM (currently naive loop, 147ms alone).

**Impact:** MSM is 21% of Zolt time. Batch affine could bring it to Jolt-level (~100-150ms).

**Expected savings:** ~500ms on collatz

---

## 3. Sumcheck Optimization (~227ms gap)

Zolt's stages 4-6 take ~256ms vs Jolt's ~29ms. That's ~8x slower.

**Problem:** Likely a combination of:
- Memory access patterns (cache misses on large polynomial arrays)
- Insufficient parallelism in inner loops (bind operations)
- Jolt uses rayon for fine-grained parallelism across all sumcheck rounds

**Solution:** Profile stages 4-6 individually and identify:
- Whether bind operations can be parallelized (Stage 4 Gruen is 130ms alone)
- Whether polynomial evaluation can use SIMD or batch operations
- Whether memory layout can be improved for cache locality

**Expected savings:** ~150-200ms on collatz

---

## 4. Fr Field Arithmetic (~235ms gap)

Zolt shows 235ms in Fr (scalar field) operations that don't appear as a separate category in Jolt.

**Problem:** Despite x86 asm Montgomery mul, Fr operations are still visible in the profile. Possibly:
- Jolt's Fr ops are fully inlined and attributed to callers
- Zolt has unnecessary Fr operations (e.g., conversions, redundant normalizations)
- The asm mul may not be inlined across compilation units

**Solution:**
- Ensure `montgomeryMul` is properly inlined at all call sites
- Profile which callers are triggering Fr operations
- Check for redundant toMontgomery/fromMontgomery conversions
- Consider `@setRuntimeSafety(false)` in hot paths if safety checks are showing up

**Expected savings:** ~100-150ms on collatz

---

## Summary

| Optimization | Gap | Expected Savings | Difficulty |
|---|---|---|---|
| Fp2 lazy reduction | 1041ms | 400-500ms | Medium |
| Batch affine MSM | 584ms | 400-500ms | Medium-Hard |
| Sumcheck optimization | 227ms | 150-200ms | Medium |
| Fr field arithmetic | 235ms | 100-150ms | Easy-Medium |
| **Total** | **~1650ms** | **~1050-1350ms** | |

If all optimizations land, collatz would go from 3196ms → ~1850-2150ms (1.2-1.4x Jolt), down from 2.06x.
