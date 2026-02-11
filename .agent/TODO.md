# Zolt → Jolt Verification Progress

## Current Status
**🔧 IN PROGRESS — Stages 1-7 ALL PASS, Stage 8 (Dory final check) fails**

### What's been fixed:
1. **VirtualMULI R1CS flags** (constraints.zig):
   - VirtualInstruction ALWAYS true for opcode 0x2B (Jolt: vsr.is_some()=true)
   - IsFirstInSequence=true when vsr=0 (standalone SLLI)
   - NextIsVirtual includes opcode 0x2B
   - NextIsFirstInSequence includes standalone VirtualMULI case

2. **Lookup index consistency** (proof_converter.zig + stage6_prover.zig):
   - All lookup index computations now use centralized `computeLookupIndex()`
   - Stage 6 Booleanity G tables (Phase 1 init)
   - Stage 6 Booleanity transitionToPhase2 (Phase 2 H tables)
   - Stage 6 LookupsRaVirtual init
   - Stage 7 G table builder
   - Witness polynomial builder (buildInstructionRaPolynomial)
   - Root cause: virtual opcodes 0x0B/0x2B were handled differently in each copy

3. **Virtual opcode handling in computeLookupIndex**:
   - 0x0B (VirtualSignExtendWord): lookup_index = rs1_value (AddOperands)
   - 0x2B (VirtualMULI): lookup_index = rs1_value * (1 << shamt) (MultiplyOperands)

4. All previous fixes (Val poly encoding, rd=0, Vandermonde format, etc.)

### Current state:
- Stages 1-5: PASS ✅
- Stage 6 (Batched Sumcheck): PASS ✅
- Stage 7 (HammingWeightClaimReduction): PASS ✅
- Stage 8 (Dory commitment opening): VMV D2 check passes, final check FAILS ❌

### Next step:
- Debug Stage 8 Dory final check failure
- This is likely a pre-existing Dory issue (was never reached before since Stage 6 was failing)
- VMV D2 passes → initial setup is OK
- Final check fails → reduction rounds or final scalar product message may have issues
- Need to compare Dory transcript state between Zolt and Jolt

## Test Commands
```bash
# Build Zolt
cd /home/vivado/projects/zolt && zig build -Doptimize=ReleaseFast

# Generate proof
./zig-out/bin/zolt prove examples/fibonacci.elf --trace-length 64 -o /tmp/zolt_proof.bin --jolt-format --export-preprocessing /tmp/zolt_preprocessing.bin --srs /tmp/jolt_dory_srs.bin

# Verify with Jolt (debug mode)
cd /home/vivado/projects/jolt && RAYON_NUM_THREADS=1 cargo run --release --features zolt-debug --manifest-path examples/fibonacci/Cargo.toml -- --verify-zolt-proof /tmp/zolt_proof.bin --zolt-preprocessing /tmp/zolt_preprocessing.bin.ram

# Run Zig tests
cd /home/vivado/projects/zolt && zig build test
```
