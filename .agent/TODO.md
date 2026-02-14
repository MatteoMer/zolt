# Zolt → Jolt Cross-Verification Progress

## COMPLETED ✅ — ALL 8/8 Programs Pass All 8 Stages!

### Passing Programs:
1. ✅ fibonacci - All 8 stages pass
2. ✅ collatz - All 8 stages pass
3. ✅ factorial - All 8 stages pass
4. ✅ sum - All 8 stages pass
5. ✅ signed - All 8 stages pass
6. ✅ primes - All 8 stages pass (log_size=16, sigma=8, nu=8)
7. ✅ bitwise - All 8 stages pass (FIXED: LUI sign-extension + materializeTableEntry override)
8. ✅ gcd - All 8 stages pass (FIXED: rs1_read comment bug + LUI sign-extension)

### Fix History:
- Stages 1-4: R1CS, operands, serialization, preprocessing
- Stage 5: SumcheckId enum + config serialization format
- Stage 6: Booleanity gamma sampling + BytecodeReadRaf Val[3] rd=0 handling
- Stage 7: Sumcheck passes (no issues)
- Stage 8: Dory g2_0 mismatch (SRS log_size fix - Feb 14, 2026)
- Stage 4 (gcd): rs1_read accidentally commented out in tracer for REMW/DIVW
- Stage 5 (bitwise+gcd): LUI/AUIPC U-type immediate sign-extension fix +
  removed materializeTableEntry override in combined_vals (Feb 11, 2026)

### Cleanup (Feb 14, 2026):
- ✅ Set debug_verbose=false in prefix_suffix_prover.zig and stage5_prover.zig
- ✅ Gated unconditional debug prints behind debug_verbose in dory.zig
- ✅ Replaced unconditional std.debug.print with dbg() in stage6_prover.zig
- ✅ Replaced unconditional std.debug.print with dbg() in proof_converter.zig
- ✅ Replaced unconditional std.debug.print with dbg() in zkvm/mod.zig
- ✅ Replaced unconditional std.debug.print with dbg() in prefix_suffix_prover.zig
- ✅ All 8 programs verified after cleanup

## KEY FILES
- Proof: /tmp/zolt_proof_dory.bin
- Preprocessing: /tmp/zolt_preprocessing.bin
- Jolt verifier test: /home/vivado/projects/jolt/jolt-core/src/zolt_compat_test.rs

## BUILD & TEST COMMANDS
```bash
# Build Zolt
zig build -Doptimize=ReleaseFast

# Generate proof + preprocessing
./zig-out/bin/zolt prove examples/collatz.elf --jolt-format -o /tmp/zolt_proof_dory.bin --export-preprocessing /tmp/zolt_preprocessing.bin

# Run Jolt verifier
cd /home/vivado/projects/jolt && cargo test --package jolt-core --features zolt-debug zolt_compat_test::tests::test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture

# Test all 8 programs
for prog in fibonacci collatz factorial sum signed primes bitwise gcd; do
  echo "=== $prog ===" && \
  ./zig-out/bin/zolt prove examples/$prog.elf --jolt-format -o /tmp/zolt_proof_dory.bin --export-preprocessing /tmp/zolt_preprocessing.bin 2>/dev/null && \
  cd /home/vivado/projects/jolt && cargo test --package jolt-core --features zolt-debug zolt_compat_test::tests::test_verify_zolt_proof_with_zolt_preprocessing -- --ignored 2>&1 | grep 'test result' && \
  cd /home/vivado/projects/zolt
done
```
