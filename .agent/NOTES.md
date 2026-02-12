# Debugging Notes

## Stage 5 Fix (Session 5) - COMPLETED

### Root Cause Found
The Stage 5 bug for collatz, bitwise, and primes was NOT in the prefix-suffix decomposition
or eval_2 computation. The actual root cause was:

**lookup_output in combined_vals used rd_value instead of materializeTableEntry(lookup_key)**

For many instruction types, `rd_value ≠ materializeTableEntry(key)`:
- LUI: rd_value = sign_extend_32_to_64(imm<<12), but table entry = lower64(key)
- JAL: lookup_output was pc+imm, but table entry at key differs
- VirtualSRLI: rd_value differs from table MLE output

The initial claim `Σ_j eq(j) * combined_vals[j]` used `lookup_output = rd_value` (wrong),
while the address round prefix-suffix decomposition correctly used table MLEs. This caused
the sumcheck chain to diverge from the initial claim.

### Fix
Replaced instruction-specific overrides (only ADDIW/ADDW/SUBW) with a general fix:
```zig
if (table_idx >= 0) {
    lookup_output = F.fromU64(Table.materializeTableEntry(table_idx, lookup_idx));
}
```

Also implemented:
- `VirtualSRL.materializeEntry()` for the VirtualSRL table
- General `materializeTableEntry()` function dispatching to all 30+ tables

### Previous Hypothesis Was Wrong
The earlier sessions hypothesized that the prefix MLE or suffix values were computed
incorrectly at c=2. This was incorrect - the prefix-suffix decomposition was always
correct. The bug was in the initial combined_vals computation.

## Stage 6 Analysis (Session 5) - IN PROGRESS

### Root Cause
gcd.elf (Stage 6 failure) and primes.elf (Stage 6 failure) use division/remainder
instructions that Zolt doesn't decompose into virtual instruction sequences:

- gcd.elf uses: divw (21 steps), remw (21 steps), mulw (2 steps)
- primes.elf uses: remuw (12 steps), mulw (2 steps)

In Jolt, these are expanded into complex inline sequences:
- DIVW/REMW → 21 virtual instructions each
- REMUW → 12 virtual instructions
- Each uses VirtualAdvice, VirtualSignExtendWord, VirtualAssertEQ, etc.

Zolt currently executes these as single trace steps, creating a fundamental mismatch
in trace structure, bytecode, and all subsequent proofs.

### Required New Virtual Instructions
- VirtualAdvice - Oracle-provided values (quotient, remainder)
- VirtualAssertEQ - Assert two values are equal
- VirtualAssertValidDiv0 - Validate division by zero handling
- VirtualChangeDivisorW/VirtualChangeDivisor - Handle overflow cases
- VirtualZeroExtendWord - Zero-extend to 32 bits
- VirtualAssertValidUnsignedRemainder - Verify |remainder| < |divisor|
- VirtualAssertMulUNoOverflow - Verify unsigned multiply doesn't overflow
- VirtualAssertLTE - Less than or equal assertion
- VirtualMovsign - Extract sign bit

### Required New Lookup Tables
- ValidDiv0Table (index 17)
- ValidUnsignedRemainderTable (index 16)
- VirtualChangeDivisorWTable (index 29)
- VirtualChangeDivisorTable
- LowerHalfWordTable
- LTETable
- VirtualAssertMulUNoOverflowTable
- MovsignTable

### Scope
This is a multi-day effort involving:
1. Tracer: Multi-step execution with virtual register tracking
2. Preprocessing: Bytecode expansion for all division variants
3. Lookup tables: New table implementations and materializeEntry functions
4. R1CS constraints: New circuit flags for virtual instructions
5. All 8 proof stages must handle the expanded traces correctly
