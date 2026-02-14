# Zolt → Jolt Cross-Verification Progress

## COMPLETED ✅ — 6/8 Programs Pass All 8 Stages!

### Passing Programs:
1. ✅ fibonacci - All 8 stages pass
2. ✅ collatz - All 8 stages pass
3. ✅ factorial - All 8 stages pass
4. ✅ sum - All 8 stages pass
5. ✅ signed - All 8 stages pass
6. ✅ primes - All 8 stages pass (log_size=16, sigma=8, nu=8)

### Failing Programs:
7. ❌ bitwise - Stage 5 fails (prover self-check Stage 4 passes, but Stage 5 RAF mismatch)
8. ❌ gcd - Stage 4 fails (prover self-check Stage 4 ALSO fails: batched_claim ≠ total_expected)

### Fix History:
- Stages 1-4: R1CS, operands, serialization, preprocessing
- Stage 5: SumcheckId enum + config serialization format
- Stage 6: Booleanity gamma sampling + BytecodeReadRaf Val[3] rd=0 handling
- Stage 7: Sumcheck passes (no issues)
- Stage 8: Dory g2_0 mismatch (SRS log_size fix - Feb 14, 2026)

## REMAINING WORK 🔄

### gcd.elf - Stage 4 Prover Bug
- The prover's own Stage 4 self-check fails: `batched_claim != total_expected`
- This means the Stage 4 sumcheck proof itself is incorrectly generated
- gcd has more bytecode (188 bytes, bytecode_K=256) vs collatz (68 bytes, bytecode_K=32)
- May be related to bytecode_K affecting the grand product computation

### bitwise.elf - Stage 5 RAF Mismatch
- Stage 4 self-check passes, but Stage 5 fails at verification
- `[CORRECT_RAF R1] matches raf_evals[0]: false` - RAF evaluations don't match for row 1+
- Row 0 matches but subsequent rows don't
- May be related to how bitwise instructions (AND, OR, XOR, SLL, SRL, SRA) are decomposed
- Bitwise ops exercise different lookup tables than arithmetic/branch ops

### Cleanup
- [ ] Remove debug prints from dory.zig, mod.zig, main.zig
- [ ] Remove debug modifications from Jolt's dory-pcs crate
- [ ] Documentation

## KEY FILES
- Proof: /home/vivado/projects/zolt/logs/zolt_proof_dory.bin
- Preprocessing: /home/vivado/projects/zolt/logs/zolt_preprocessing.bin
- Jolt verifier test: /home/vivado/projects/jolt/jolt-core/src/zolt_compat_test.rs

## BUILD & TEST COMMANDS
```bash
# Build Zolt
zig build -Doptimize=ReleaseFast

# Generate proof + preprocessing
./zig-out/bin/zolt prove examples/collatz.elf --jolt-format -o /tmp/zolt_proof_dory.bin --export-preprocessing /tmp/zolt_preprocessing.bin

# Run Jolt verifier
cd /home/vivado/projects/jolt && cargo test --package jolt-core --features zolt-debug zolt_compat_test::tests::test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture
```
