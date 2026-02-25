# Zolt → Jolt Cross-Verification Progress

## STATUS: ✅ COMPLETE — 8/8 PROGRAMS VERIFIED

### Success Criteria:
1. ✅ `zig build test` passes all tests (exit code 0)
2. ✅ Zolt generates proofs for all 8 example programs
3. ✅ **All 8 proofs verified by Jolt** ✅
4. ✅ No modifications needed on the Jolt side (only zolt-debug feature path)

### All Programs Passing (8/8):
1. ✅ fibonacci — All 8 stages pass
2. ✅ collatz — All 8 stages pass
3. ✅ factorial — All 8 stages pass
4. ✅ sum — All 8 stages pass
5. ✅ signed — All 8 stages pass
6. ✅ primes — All 8 stages pass
7. ✅ gcd — All 8 stages pass
8. ✅ bitwise — All 8 stages pass

### Last Full Verification: Feb 25, 2026 (Iteration 1546 — Re-confirmed)
All 8 programs confirmed: proof generation + Jolt verification passing.
- fibonacci: ✅ All 8 stages pass (5.6s)
- collatz: ✅ All 8 stages pass (22.9s)
- factorial: ✅ All 8 stages pass (5.6s)
- sum: ✅ All 8 stages pass (5.5s)
- signed: ✅ All 8 stages pass (5.5s)
- primes: ✅ All 8 stages pass (27.2s)
- gcd: ✅ All 8 stages pass (10.2s)
- bitwise: ✅ All 8 stages pass (11.0s)
- `zig build test` exit code: 0

### Automated Testing
- Integration test script: `scripts/verify_all.sh`
  - `./scripts/verify_all.sh` — Full prove + Jolt verify for all 8 programs
  - `./scripts/verify_all.sh --quick` — Prove only (no Jolt verification)
  - `./scripts/verify_all.sh fibonacci gcd` — Test specific programs

## BUGS FIXED (Summary)

### Bug: LUI/AUIPC Imm encoding mismatch (bitwise Stage 6)
- **Root cause**: Val_poly for LUI/AUIPC truncated the immediate to u32 before
  converting to field element, but the R1CS witness used the full 64-bit sign-extended value.
- **Fix**: Removed the LUI/AUIPC special case in val_poly Imm encoding.

### Bug: SignExtension suffix MLE (GCD Stage 5)
- Suffix type for sign-extension instructions had wrong MLE evaluation.

### Bug: LeftOperandMsb prefix MLE (GCD Stage 5)
- Prefix MLE for left-operand MSB had an incorrect computation.

### Bug: Suffix ordering for ValidSignedRemainder/ValidUnsignedRemainder/DivByZero
- Suffix ordering didn't match Jolt's expected order.

### Bug: LookupBits.uninterleave() MSB drop
- Dropped MSB on odd-length inputs.

### Bug: REMW/DIVW trace emission
- Missing 21-step trace emission for REMW/DIVW instructions.

## KEY FILES
- `src/zkvm/spartan/stage6_prover.zig` — BytecodeReadRaf val_poly construction
- `src/zkvm/proof_converter.zig` — Opening claims computation, R1CS witness
- `src/zkvm/r1cs/constraints.zig` — R1CS Imm encoding (deriveImmediate)
- `src/zkvm/lookup_table/prefix_suffix_prover.zig` — Prefix/suffix MLE evaluation

## BUILD & TEST COMMANDS
```bash
# Build Zolt
zig build -Doptimize=ReleaseFast

# Run all Zig tests
zig build test

# Run full cross-verification (all 8 programs)
./scripts/verify_all.sh

# Quick mode (prove only, no Jolt verification)
./scripts/verify_all.sh --quick

# Manual single-program test
./zig-out/bin/zolt prove examples/<program>.elf --jolt-format -o /tmp/zolt_proof_dory.bin --export-preprocessing /tmp/zolt_preprocessing.bin
cd /home/vivado/projects/jolt && cargo test --package jolt-core --features zolt-debug zolt_compat_test::tests::test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture
```
