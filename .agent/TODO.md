# Zolt-Jolt Compatibility Implementation

## Status: Session 46 - Stage 5 Output Claim Mismatch (Deep Analysis)

## Key Discovery

The Stage 5 verification fails because:
1. **Initial claim matches** ✓
2. **Coefficients match** ✓
3. **Challenges match** ✓
4. **Batching coefficients match** ✓
5. **Input claims match** ✓
6. **BUT new_claim after Round 0 differs**

### Zolt Round 0 Results
- claim = `99 05 78 d0 e9 6c 66 a0 a2 c8 0e 47 2d 90 0a 1c b2 da c0 db 53 7e aa e8 2e 1b 48 bc 17 60 fb 00` (LE)
- new_claim = `0a ad ae 99 93 5a b0 7d 49 23 af 46 92 b6 95 49 8b a8 47 be 64 bd 1c 76 c2 cd 0b f1 92 34 45 21` (LE)

### Jolt Round 0 Results
- claim (verified same as Zolt)
- new_claim = `36 ef d9 e1 08 37 de ed e0 e9 93 8d 21 52 47 ed 71 11 f1 85 98 aa 31 4e 5f 19 8b 02 7c 77 ea 12` (LE)

### Root Cause Hypothesis

The polynomial evaluation formula is:
```
new_claim = c0 + c1*r + c2*r² + c3*r³
where c1 = claim - 2*c0 - c2 - c3
```

Both Zolt and Jolt use this formula, but Zolt's internal tracking produces a different result than what Jolt computes. This suggests:

1. **The polynomial evaluation in Zolt doesn't match Jolt's** - specifically the `c1 * r` multiplication
2. Zolt uses `c1.mulHiBigIntU128(challenge.limbs)` for Field * Challenge
3. Jolt uses `*x * linear_term` which is Challenge * Field

The multiplication order and method may differ!

## Hypothesis to Test

Check if `F * Challenge` in Zolt produces the same result as `Challenge * F` in Jolt.

In Jolt:
- `*x * linear_term` = Challenge * F
- Uses the optimized multiplication from Challenge impl

In Zolt:
- `c1.mulHiBigIntU128(challenge.limbs)` = F * (high 128-bit BigInt)
- This might have a different semantic than Jolt's multiplication

## Files of Interest

- `/home/vivado/projects/zolt/src/zkvm/spartan/stage5_prover.zig:2316` - Zolt's claim update
- `/home/vivado/projects/jolt/jolt-core/src/poly/unipoly.rs:455` - Jolt's eval_from_hint
- `/home/vivado/projects/zolt/src/field/mod.zig:337` - Zolt's mulHiBigIntU128

## Next Steps

1. Add detailed debug to Zolt's round 0 computation showing:
   - c0, c2, c3 values (already logged)
   - c1 value (claim - 2*c0 - c2 - c3)
   - challenge value
   - c1 * challenge result
   - Final new_claim computation

2. Compare step-by-step with Jolt's computation

3. Fix the multiplication to match Jolt's semantics

## Commands

```bash
zig build -Doptimize=ReleaseFast
./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format -o /tmp/zolt_proof_dory.bin --export-preprocessing /tmp/zolt_preprocessing.bin --trace-length 64
cd /home/vivado/projects/jolt && cargo test -p jolt-core --features zolt-debug --lib test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture
```
