# Zolt zkVM Implementation TODO

## Current Status (Jolt Compatibility Phase)

**Project Status: FORMAT COMPATIBLE ✅, EXECUTION INTEGRATION BLOCKED 🔄**

### Session 5 Findings

#### Root Cause Identified

The execution mismatch is caused by **ECALL handling**:

1. **Zolt**: Treats ALL ECALL instructions as termination
2. **Jolt**: Has special ECALL handling for:
   - `JOLT_CYCLE_TRACK_ECALL_NUM (0xC7C1E)`: Cycle tracking - continue execution
   - `JOLT_PRINT_ECALL_NUM`: Print/output - continue execution
   - Other ECALLs: Normal trap handling

When the fibonacci-guest calls `start_cycle_tracking("fib_loop")`, it issues an ECALL with a0=0xC7C1E. Jolt continues, but Zolt terminates.

#### What Works
1. **I/O Region**: Loads from 0x7fffa000 correctly return input bytes ✅
2. **Input Setting**: `--input-hex 32` correctly sets input to 50 ✅
3. **Instruction Execution**: All RV64IMC instructions execute correctly ✅
4. **Format Compatibility**: Proofs deserialize correctly in Jolt ✅

#### What Needs Work
1. **ECALL Handling**: Need to recognize Jolt SDK ECALLs and not terminate
2. **Cycle Tracking**: Optional - could skip or implement

### Next Steps

**To implement Jolt SDK ECALL handling:**
```zig
// In tracer/mod.zig, when handling ECALL:
const a0 = self.registers.read(10);
const JOLT_CYCLE_TRACK_ECALL_NUM: u32 = 0xC7C1E;
const JOLT_PRINT_ECALL_NUM: u32 = 0x...;

if (a0 == JOLT_CYCLE_TRACK_ECALL_NUM) {
    // Cycle tracking - ignore and continue
    self.state.pc += 4; // Skip past ECALL
    return; // Don't terminate
} else if (a0 == JOLT_PRINT_ECALL_NUM) {
    // Print - handle output if needed
    self.state.pc += 4;
    return;
} else {
    // Check for termination or panic
    return error.Ecall;
}
```

---

## Test Results Summary

### Zolt: 622/622 tests PASS ✅

### Jolt Cross-Verification
| Test | Status |
|------|--------|
| `test_deserialize_zolt_proof` | ✅ PASS |
| `test_verify_zolt_proof` | ⚠️ Blocked (execution mismatch) |

---

## Session 5 Commits

1. `docs: update compatibility status - format fully verified`
2. `feat(tracer): add I/O region support for Jolt guest programs`
3. `feat(cli): add --input and --input-hex options for guest programs`
4. `docs: update TODO with I/O region progress`

---

## Debug Info

Fibonacci guest execution trace (with input 50):
```
Cycle 10: LOAD x11, 0(x10) ; [0x7fffa000] -> 0x32 (50)
Cycle 11: ANDI x14, x11, 127 ; x14 = 50
Cycle 12: BGE x11, x0, 78 ; branch taken (50 >= 0)
...
Cycle 21: ECALL (a0 = 0x8000009a) -> Zolt terminates
```

Jolt continues past the ECALL because it recognizes it as a Jolt SDK call.

---

## Commands Reference

```bash
# Run with input
./zig-out/bin/zolt run --input-hex 32 --regs /path/to/guest.elf

# Run Jolt fibonacci to compare
cd /Users/matteo/projects/jolt/examples/fibonacci
cargo run --release -- --save

# Check Jolt trace file size (should be ~22KB for fib(50))
ls -la /tmp/fib_trace.bin
```
