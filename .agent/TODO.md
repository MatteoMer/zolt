# Zolt zkVM Implementation TODO

## Current Status

**Project Status: SIGNIFICANT PROGRESS - ECALL HANDLING IMPLEMENTED 🔄**

### Session 5 Summary

#### Completed
1. **Format Compatibility** ✅ - Proof format fully compatible with Jolt
2. **I/O Region Support** ✅ - Loads from 0x7fffa000 work correctly
3. **CLI Input Options** ✅ - `--input-hex` works to provide guest input
4. **ECALL Handling** ✅ - Jolt SDK ECALLs are now handled properly

#### In Progress
5. **Guest Execution Debugging** 🔄 - Program runs but loops indefinitely

### What Changed

**ECALL Handling (Major Fix)**

Before: Zolt terminated on ALL ECALL instructions
After: Zolt recognizes Jolt SDK ECALLs and continues execution:
- `JOLT_CYCLE_TRACK_ECALL_NUM (0xC7C1E)` - Skip and continue
- `JOLT_PRINT_ECALL_NUM` - Skip and continue
- Unknown ECALLs - Continue (for SDK internal calls)
- Termination requests - Terminate as expected

**Result**: fibonacci-guest now runs 100000+ cycles instead of stopping at 21!

### Remaining Issue

The program runs past the first ECALL but then loops indefinitely:
```
  [ECALL] a0=0x8000009a, a7=0x0, PC=0x800000b0
    -> Unknown ECALL, continuing

Cycle limit reached!
Cycles executed: 100000
Final PC: 0x800000cc
```

The loop counter (x14) reaches 49 (= n-1 for n=50) but the loop continues.
This might be due to:
1. Input not being parsed correctly as a u32 (only 1 byte provided)
2. Loop termination condition issue
3. Some other instruction execution difference

### Test Results

- Zolt tests: 622/622 PASS ✅
- Jolt deserialization: PASS ✅
- Guest execution: INCOMPLETE (loops)

---

## Commits This Session

1. `docs: update compatibility status - format fully verified`
2. `feat(tracer): add I/O region support for Jolt guest programs`
3. `feat(cli): add --input and --input-hex options for guest programs`
4. `docs: update TODO with I/O region progress`
5. `docs: identify ECALL handling as root cause of execution mismatch`
6. `feat(tracer): add Jolt SDK ECALL handling`

---

## Next Steps

1. **Fix input parsing**: The input 50 should be read as a u32 (4 bytes) not just 1 byte
   - Postcard encodes u32 differently than raw bytes
   - May need to provide full postcard-encoded input

2. **Debug loop termination**: Compare instruction execution with Jolt
   - Add trace comparison mode
   - Check branch conditions

3. **Alternative approach**: Test with no-input guest programs first
   - memory-ops-guest doesn't need input
   - Simpler to debug

---

## Commands

```bash
# Build and test
zig build test --summary all

# Run Jolt guest with input
zig build -Doptimize=ReleaseFast
./zig-out/bin/zolt run --input-hex 32 --max-cycles 100000 --regs /path/to/guest.elf
```
