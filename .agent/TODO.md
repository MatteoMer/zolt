# Zolt → Jolt Verification Progress

## Current Status
**7/8 example programs pass all 8 stages. 1 needs REMW/DIVW decomposition.**

### Verified Programs (ALL 8 stages pass):
- ✅ fibonacci.elf (trace-length 64, 128, 4096)
- ✅ factorial.elf (trace-length 64)
- ✅ sum.elf (trace-length 64)
- ✅ signed.elf (trace-length 64)
- ✅ collatz.elf (trace-length 128)
- ✅ bitwise.elf (trace-length 32)
- ✅ primes.elf (trace-length 4096) - **FIXED in Session 15!**

### Programs with verification failures:
- ❌ gcd.elf - Needs REMW/DIVW 21-step decomposition (currently only REMUW/DIVUW 12-step is implemented)

## Session 15 Fixes (primes.elf - ALL 8 stages now pass)

### Fix 1: Stage 6 IncClaimReduction virtual register support
**Root cause:** IncClaimReduction was using `(step.instruction >> 7) & 0x1f` (5-bit truncation, only registers 0-31) and `register_values: [32]u64`, while stage4_gruen_prover (the actual Stage 4 implementation) uses `step.rd_index` (u8, registers 0-127) and `register_values: [128]u64`.

For virtual instruction steps with registers 32+, Stage 6 was truncating to 0-31, causing wrong pre_value lookups and wrong inc values.

**Fix:** Changed IncClaimReduction to use `step.rd_index`, `step.rd_written`, and `register_values: [128]u64`.
**File:** `src/zkvm/spartan/stage6_prover.zig`

### Fix 2: Stage 6 computeLookupIndex missing virtual opcode handlers
**Root cause:** `computeLookupIndex` only handled virtual opcodes 0x0B, 0x2B, 0x5B. Missing handlers for 0x02 (VirtualAdvice), 0x22 (VirtualAssertEQ), 0x42 (VirtualZeroExtendWord), 0x62 (VirtualAssertValidUnsignedRemainder) caused them to fall through to standard RISC-V decoding where `left_is_rs1`/`right_is_rs2` switches didn't include these opcodes, returning `interleaveBits(0, 0) = 0`.

**Fix:** Added explicit handlers for all 4 missing virtual opcodes.
**File:** `src/zkvm/spartan/stage6_prover.zig`

### Fix 3: Dory buildRdIncPolynomial virtual register support
**Root cause:** Same bug as Fix 1 but in the Dory witness polynomial builder. `buildRdIncPolynomial` was using `register_values: [32]u64` and instruction bit extraction instead of `step.rd_index`/`step.rd_written` with 128 registers. This caused the Dory-committed polynomial to differ from the Stage 6 opening claims.

**Fix:** Changed to use `step.rd_index`, `step.rd_written`, and `register_values: [128]u64`.
**File:** `src/zkvm/mod.zig`

## Pending Tasks
- [ ] Commit and push fixes
- [ ] Regression test 6 previously-passing programs
- [ ] Implement REMW/DIVW 21-step decomposition for gcd.elf
- [ ] Test all 8 programs pass verification

## Key Technical Notes
- Stage 4 uses `stage4_gruen_prover` (NOT `stage4_prover`) - critical for matching register tracking
- Virtual opcodes: 0x02 (VirtualAdvice), 0x0B (VirtualSignExtendWord), 0x22 (VirtualAssertEQ), 0x2B (VirtualMULI), 0x42 (VirtualZeroExtendWord), 0x5B (VirtualSRLI), 0x62 (VirtualAssertValidUnsignedRemainder)
- `buildInstructionRaPolynomial` already uses `computeLookupIndex` (auto-picks up virtual opcode fix)

## Test Commands
```bash
cd /home/vivado/projects/zolt && zig build -Doptimize=ReleaseFast
./zig-out/bin/zolt prove examples/primes.elf --trace-length 4096 -o /tmp/zolt_primes_proof.bin --jolt-format --export-preprocessing /tmp/zolt_primes_preprocessing.bin --srs /tmp/jolt_dory_srs.bin
cd /home/vivado/projects/jolt && RAYON_NUM_THREADS=1 cargo run --release --features zolt-debug --manifest-path examples/fibonacci/Cargo.toml -- --verify-zolt-proof /tmp/zolt_primes_proof.bin --zolt-preprocessing /tmp/zolt_primes_preprocessing.bin.ram
```
