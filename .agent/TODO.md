# Zolt → Jolt Verification Progress

## Current Status
**✅ Core verification pipeline works! 4/8 example programs pass all 8 stages.**

### Verified Programs (ALL 8 stages pass):
- ✅ fibonacci.elf (trace-length 64, 128)
- ✅ factorial.elf (trace-length 64)
- ✅ sum.elf (trace-length 64)
- ✅ signed.elf (trace-length 64)

### Programs with verification failures (instruction-specific issues):
- ❌ gcd.elf - Stage 6 (uses `divw`, `remw` — division/remainder instructions)
- ❌ collatz.elf - Stage 4 (uses `slliw`, `srliw`, `bltu` — W-ext shifts + unsigned branch)
- ❌ bitwise.elf - Stage 4 (uses `xor`, `slliw`, `srliw` — XOR + W-ext shifts)
- ❌ primes.elf - Stage 5 (uses `remuw`, `bgeu`, `bne`, `beq` — unsigned rem, branch variants)

### What's been accomplished:

1. **bytecode_K preprocessing export fix** (Iteration 5):
   - Fixed `main.zig` to use `computeBytecodeCodeSize()` instead of `bytecode_prep.code_size`
   - The former accounts for +3 termination store entries (LUI/ADDI/SB), the latter doesn't
   - Made `computeBytecodeCodeSize` public for use from main.zig
   - This fixed factorial.elf verification (was crashing in Stage 6 with index OOB)

2. **VirtualMULI R1CS flags** (constraints.zig):
   - VirtualInstruction ALWAYS true for opcode 0x2B
   - IsFirstInSequence=true when vsr=0 (standalone SLLI)
   - NextIsVirtual includes opcode 0x2B
   - NextIsFirstInSequence includes standalone VirtualMULI case

3. **Lookup index consistency** (proof_converter.zig + stage6_prover.zig + mod.zig):
   - All lookup index computations now use centralized `stage6_prover.computeLookupIndex()`

4. **Virtual opcode handling in computeLookupIndex**:
   - 0x0B (VirtualSignExtendWord): lookup_index = rs1_value (AddOperands)
   - 0x2B (VirtualMULI): lookup_index = rs1_value * (1 << shamt) (MultiplyOperands)

5. **Debug print cleanup** (multiple iterations):
   - Gated ~3400+ std.debug.print calls behind compile-time `debug_verbose = false`

6. **Fixed OOM in integration test** (iteration 3):
   - Added MAX_CYCLES (1M) limit to emulator run()
   - Fixed test programs with proper termination (jal x0, 0 self-loop)

7. **Fixed pre-existing test failures** (iteration 3):
   - All 726/726 tests pass

### Known Issues:
- **Segfault during cleanup**: After proof generation succeeds, deallocation may SIGSEGV. Pre-existing.
- **Instruction coverage**: Programs using divw/remw/remuw/xor/slliw/srliw need further work.

## Next Steps for Failing Programs:
- collatz & bitwise: Likely SRLIW/SLLIW virtual decomposition issues in Stage 4
- gcd: DIVW/REMW not yet implemented as virtual instruction decomposition
- primes: REMUW not implemented, branch instruction handling may be incomplete

## Test Commands
```bash
cd /home/vivado/projects/zolt && zig build -Doptimize=ReleaseFast
./zig-out/bin/zolt prove examples/fibonacci.elf --trace-length 64 -o /tmp/zolt_proof.bin --jolt-format --export-preprocessing /tmp/zolt_preprocessing.bin --srs /tmp/jolt_dory_srs.bin
cd /home/vivado/projects/jolt && RAYON_NUM_THREADS=1 cargo run --release --features zolt-debug --manifest-path examples/fibonacci/Cargo.toml -- --verify-zolt-proof /tmp/zolt_proof.bin --zolt-preprocessing /tmp/zolt_preprocessing.bin.ram
```

## Success Criteria:
- [x] `zig build test` passes all tests (726/726)
- [x] Zolt can generate a proof for example programs
- [x] The proof can be loaded and verified by Jolt's verifier (4 programs verified)
- [x] No modifications needed on the Jolt side (only zolt-debug feature for logging)
