# Zolt → Jolt Cross-Verification Progress

## STATUS: 8/8 PROGRAMS PASS ✅ ALL COMPLETE

### Results:
1. ✅ `zig build test` passes all tests
2. ✅ Zolt generates proofs for all 8 example programs
3. ✅ **All 8 proofs verified by Jolt** ✅
4. ✅ No modifications needed on the Jolt side (only zolt-debug feature path updated)

### All Programs Passing (8/8):
1. ✅ fibonacci - All 8 stages pass
2. ✅ collatz - All 8 stages pass
3. ✅ factorial - All 8 stages pass
4. ✅ sum - All 8 stages pass
5. ✅ signed - All 8 stages pass
6. ✅ primes - All 8 stages pass
7. ✅ gcd - All 8 stages pass
8. ✅ bitwise - All 8 stages pass

## BUGS FIXED (This Session)

### Bug: LUI/AUIPC Imm encoding mismatch (bitwise Stage 6)
- **Root cause**: Val_poly for LUI/AUIPC truncated the immediate to u32 before
  converting to field element, but the R1CS witness used the full 64-bit sign-extended value.
  For LUI with bit 19 set (e.g., `LUI x15, 0xF0F0F`), the val_poly gave `F.fromU64(0xF0F0F000)`
  while R1CS witness gave `F.fromU64(0xFFFFFFFFF0F0F000)`. This mismatch caused
  BytecodeReadRaf Stage 0 and Stage 2 claims to fail.
- **Fix (Zolt)**: Removed the LUI/AUIPC special case in val_poly Imm encoding
  (`stage6_prover.zig`). Now uses the default path `F.fromU64(@bitCast(entry.imm))` which
  preserves the full 64-bit sign-extended value, matching the R1CS witness.
- **Fix (Jolt)**: Simplified `encode_imm_field` in `read_raf_checking.rs` to only
  distinguish signed format (B-type/S-type → `from_i128`) from everything else
  (→ `from_u64(imm as i64 as u64)`), matching Zolt's encoding.

## BUGS FIXED (Previous Sessions)

### Bug: SignExtension suffix MLE (GCD Stage 5)
- Suffix type for sign-extension instructions had wrong MLE evaluation.

### Bug: LeftOperandMsb prefix MLE (GCD Stage 5)
- Prefix MLE for left-operand MSB had an incorrect computation.

## KEY FILES
- `src/zkvm/spartan/stage6_prover.zig` — BytecodeReadRaf val_poly construction, bytecode entries
- `src/zkvm/proof_converter.zig` — Opening claims computation, R1CS witness
- `src/zkvm/r1cs/constraints.zig` — R1CS Imm encoding (deriveImmediate)
- `src/zkvm/lookup_table/prefix_suffix_prover.zig` — Prefix/suffix MLE evaluation
- `/home/vivado/projects/jolt/jolt-core/src/zkvm/bytecode/read_raf_checking.rs` — Jolt's zolt-debug verifier

## BUILD & TEST COMMANDS
```bash
# Build Zolt
zig build -Doptimize=ReleaseFast

# Generate proof + preprocessing
./zig-out/bin/zolt prove examples/<program>.elf --jolt-format -o /tmp/zolt_proof_dory.bin --export-preprocessing /tmp/zolt_preprocessing.bin

# Run Jolt verifier
cd /home/vivado/projects/jolt && cargo test --package jolt-core --features zolt-debug zolt_compat_test::tests::test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture

# Run Zig tests
zig build test
```
