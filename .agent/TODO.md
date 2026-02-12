# Zolt → Jolt Verification Progress

## Current Status
**✅ 6/8 example programs pass all 8 verification stages.**

### Verified Programs (ALL 8 stages pass):
- ✅ fibonacci.elf (trace-length 64, 128)
- ✅ factorial.elf (trace-length 64)
- ✅ sum.elf (trace-length 64)
- ✅ signed.elf (trace-length 64)
- ✅ collatz.elf (trace-length 128)
- ✅ bitwise.elf (trace-length 32) — FIXED in this session

### Programs with verification failures:
- ❌ gcd.elf - Stage 6 (uses `divw`, `remw` — needs inline sequence decomposition)
- ❌ primes.elf - Stage 6 (uses `remuw` — needs inline sequence decomposition)

## Stage 5 Fix (COMPLETED)

### Root Cause
For instructions where `rd_value ≠ materializeTableEntry(lookup_key)`, the initial claim
`Σ_j eq(j) * combined_vals[j]` diverged from the address round prefix-suffix decomposition.

Examples of affected instructions:
- LUI (0x37): rd_value = sign_extend_32_to_64(imm<<12), table = lower64(key)
- JAL (0x6f): lookup_output was pc+imm, but table entry differs
- VirtualSRLI (0x5b): rd_value differs from table MLE

### Fix Applied
In `stage5_prover.zig`, replaced instruction-specific lookup_output overrides
(previously only ADDIW/ADDW/SUBW) with a GENERAL fix:
```zig
if (table_idx >= 0) {
    lookup_output = F.fromU64(Table.materializeTableEntry(table_idx, lookup_key));
}
```
This uses the actual table entry for ALL instructions with a valid lookup table.

Also added `materializeEntry()` for VirtualSRL table and a general `materializeTableEntry()`
function in `mod.zig`.

## Stage 6 Root Cause Analysis

### Problem
gcd.elf and primes.elf fail at Stage 6 because they use division/remainder instructions
(DIVW, REMW, REMUW) that Zolt doesn't decompose into virtual instruction sequences.

### Background
In Jolt, complex M-extension instructions are decomposed into multi-step inline sequences:
- DIVW → 21 virtual instructions (VirtualAdvice, VirtualSignExtendWord, MUL, SUB, etc.)
- REMW → 21 virtual instructions
- REMUW → 12 virtual instructions (VirtualAdvice, VirtualZeroExtendWord, MUL, etc.)
- DIVUW → 14 virtual instructions
- DIV → 18 virtual instructions
- REM → 18 virtual instructions
- DIVU → 11 virtual instructions
- REMU → 10 virtual instructions

Zolt currently executes these as single trace steps, creating a fundamental mismatch
in trace structure, bytecode, and all subsequent proofs.

### What's Needed
1. Implement inline sequence expansion for division/remainder instructions in the tracer
2. Add new virtual instruction types: VirtualAdvice, VirtualAssertEQ, VirtualAssertValidDiv0,
   VirtualChangeDivisorW, VirtualZeroExtendWord, VirtualAssertValidUnsignedRemainder,
   VirtualAssertMulUNoOverflow, VirtualAssertLTE, VirtualMovsign
3. Add new lookup tables: ValidDiv0Table, VirtualChangeDivisorWTable/Table,
   ValidUnsignedRemainderTable, LowerHalfWordTable, LTETable,
   VirtualAssertMulUNoOverflowTable, MovsignTable
4. Implement execution semantics for each virtual instruction
5. Handle register file updates across the virtual sequence
6. Generate correct bytecode entries for each step in the sequence

### Key Files
- `src/tracer/mod.zig` - Instruction execution and trace generation
- `src/zkvm/preprocessing.zig` - Bytecode decomposition
- `src/zkvm/instruction/lookup_trace.zig` - Lookup trace generation
- `src/zkvm/instruction/lookups.zig` - Lookup computation

### Instructions Using Division/Remainder in Examples
- gcd.elf: divw, remw, mulw
- primes.elf: remuw, mulw

## Test Commands
```bash
cd /home/vivado/projects/zolt && zig build -Doptimize=ReleaseFast
./zig-out/bin/zolt prove examples/bitwise.elf --trace-length 32 -o /tmp/zolt_bitwise_proof.bin --jolt-format --export-preprocessing /tmp/zolt_bitwise_preprocessing.bin --srs /tmp/jolt_dory_srs.bin
cd /home/vivado/projects/jolt && RAYON_NUM_THREADS=1 cargo run --release --features zolt-debug --manifest-path examples/bitwise/Cargo.toml -- --verify-zolt-proof /tmp/zolt_bitwise_proof.bin --zolt-preprocessing /tmp/zolt_bitwise_preprocessing.bin.ram
```
