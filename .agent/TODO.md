# Zolt-Jolt Compatibility - Status Update

## Current Status
- **Stage 1: PASSING ✅** - Sumcheck output_claim matches
- **Stage 2: Preprocessing mismatch** - Different programs have different tau values
- **Preprocessing Export: Format Mismatch** - SRS size differs (11 vs 13 entries)
- All Zolt tests pass (712/712)

## What Works
1. **Proof serialization/deserialization** ✅ - Jolt can deserialize Zolt proofs
2. **Virtual polynomial claims** ✅ - All values match exactly:
   - LeftInstructionInput ✅
   - RightInstructionInput ✅
   - IsRdNotZero ✅
   - NextIsNoop ✅
3. **OutputSumcheck** ✅ - val_final_claim == val_io_eval (difference = 0)
4. **Stage 1 sumcheck** ✅ - output_claim is correct

## Root Cause

### Problem 1: Different Programs
- Zolt's fibonacci.elf: RV32 (32-bit RISC-V)
- Jolt's fibonacci guest: RV64IMAC (64-bit RISC-V)
- Different bytecode → Different tau values → Different expected claims

### Problem 2: Different SRS
- Zolt generates its own SRS with `DorySRS.setup(allocator, 20)`
- Jolt uses `DoryGlobals` with runtime-computed dimensions
- Different SRS → Different commitment verification values

### Observed Values
- Zolt tau[10]: `10082273819133654128073895258316729483365029118085832455614804789697107726281`
- Jolt tau_high: `1724079782492782403949918631195347939403999634829548103697761600182229454970`

## Solution Path

For full cross-verification, we need:

### Option 1: Shared SRS (Production Approach)
1. Generate a universal SRS through trusted setup
2. Both Zolt and Jolt use the same SRS
3. Export/import SRS in compatible format

### Option 2: Same Program Binary
1. Compile a RV64IMAC program that both can execute
2. Zolt needs RV64 support (currently RV32)

### Option 3: Jolt Preprocessing for Zolt ELF
1. Add Jolt API to preprocess external ELF files
2. Use Jolt-generated preprocessing with Zolt proof

## Session 10 Commits
- `362e8ab`: fix(output-sumcheck): Reverse r_address_prime for big-endian evaluation

## Technical Details

### Preprocessing Format Comparison
```
Zolt: 0b 00 00 00 00 00 00 00 (11 delta_1l entries)
Jolt: 0d 00 00 00 00 00 00 00 (13 delta_1l entries)
```

The different array lengths indicate different SRS dimensions.

### Instruction Serialization
Both use JSON format via serde_json, but field names must match exactly.
Zolt's preprocessing export includes BytecodePreprocessing, RAMPreprocessing, MemoryLayout, and DoryVerifierSetup.

## Next Steps

1. **Align SRS Generation**
   - Match Zolt's SRS seed/parameters with Jolt's
   - Or implement SRS import/export

2. **Add RV64 Support to Zolt**
   - Would allow proving Jolt-compiled programs
   - Significant effort but clean solution

3. **Jolt API Extension**
   - Add function to preprocess arbitrary ELF
   - Use that for cross-verification testing
