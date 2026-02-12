# Zolt → Jolt Verification Progress

## Current Status
**7/8 example programs pass Stage 5. 6/8 pass all 8 stages.**

### Verified Programs (ALL 8 stages pass):
- ✅ fibonacci.elf (trace-length 64, 128, 4096)
- ✅ factorial.elf (trace-length 64)
- ✅ sum.elf (trace-length 64)
- ✅ signed.elf (trace-length 64)
- ✅ collatz.elf (trace-length 128)
- ✅ bitwise.elf (trace-length 32)

### Programs with verification failures:
- ❌ primes.elf - Stage 6 (sumcheck mismatch in memory checking)
- ❌ gcd.elf - Stage 6 (sumcheck mismatch in memory checking)

**Both primes and gcd use REMUW/DIVUW instructions (virtual instruction decomposition).**
**All 6 passing programs do NOT use division/remainder instructions.**

## Session 12 Fixes

### FIXED: Stage 5 primes.elf + gcd.elf
**Root cause:** Missing uninterleave + short-circuit check in `LeftOperandIsZero` and `RightOperandIsZero` prefix MLE functions.

Jolt's implementations call `b.uninterleave()` and return `F::zero()` if the relevant operand bits are non-zero (because the IsZero prefix = Π(1-x_i), which is 0 if any bit is 1). Zolt was missing this check, causing incorrect polynomial evaluations during PS sumcheck rounds.

**Fix:** Added `b.uninterleave()` + short-circuit to both functions in `src/zkvm/lookup_table/prefixes.zig`.
**Commit:** 8e8c90b

## Stage 6 Failure Analysis

### What Stage 6 does:
Stage 6 is the memory read/write checking (grand product) stage. It verifies:
1. Register file reads/writes are consistent
2. RAM reads/writes are consistent
3. Program counter increments correctly

### Error details:
- For BOTH primes and gcd: `SUM_S6 DEBUG` shows `expected_output_claim ≠ output_claim`
- The sumcheck itself runs fine (coefficients are self-consistent)
- The final expected_output_claim (computed from opening claims) doesn't match

### Key observation:
- All 6 passing programs do NOT use division/remainder (REMUW/DIVUW)
- Both failing programs DO use REMUW/DIVUW
- REMUW decomposes into 12 virtual instruction steps
- The virtual instruction decomposition IS implemented in Zolt's tracer + preprocessing
- The issue must be in how these virtual instructions interact with Stage 6

### Likely root causes:
1. **Bytecode expansion**: The preprocessing may not correctly expand REMUW into 12 bytecode entries
2. **PC mapping**: Virtual instruction PCs may not match what Jolt's verifier expects
3. **Circuit flags**: VirtualInstruction, DoNotUpdateUnexpandedPC, IsFirstInSequence flags may be incorrect
4. **Register mapping**: Virtual registers (a2=32, a3=33, t0-t4=34-38) may not be encoded correctly
5. **Instruction witness**: The lookup outputs for assert/advice instructions may be wrong

### Next steps:
1. Compare Zolt's REMUW bytecode entries with what Jolt expects
2. Check if the PC mapping for virtual instructions matches
3. Verify circuit flag encoding for virtual instruction sequences
4. Add diagnostic output to Stage 6 to identify which component mismatches

## Test Commands
```bash
cd /home/vivado/projects/zolt && zig build -Doptimize=ReleaseFast
./zig-out/bin/zolt prove examples/primes.elf --trace-length 4096 -o /tmp/zolt_primes_proof.bin --jolt-format --export-preprocessing /tmp/zolt_primes_preprocessing.bin --srs /tmp/jolt_dory_srs.bin
cd /home/vivado/projects/jolt && RAYON_NUM_THREADS=1 cargo run --release --features zolt-debug --manifest-path examples/fibonacci/Cargo.toml -- --verify-zolt-proof /tmp/zolt_primes_proof.bin --zolt-preprocessing /tmp/zolt_primes_preprocessing.bin.ram
```
