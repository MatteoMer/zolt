# Stage 6 Debugging Progress

## W-Extension Fix (DONE)
- Fixed `from_raw_words` to properly handle W-extension decomposition
- Uses pattern matching on Instruction enum variants (VirtualSignExtendWord, VirtualMULI)
- For base instructions of W-ext sequences: maps opcode (0x1b→0x13, 0x3b→0x33) and sets virtual flags
- All bytecode entries now match between Zolt prover and Jolt verifier

## rd=0 Fix (DONE)
- RISC-V x0 is hardwired zero register, can never be written
- Jolt verifier maps rd_raw==0 to None (zero contribution to val_polys)
- Fixed 3 locations in stage6_prover.zig to map rd==0 to sentinel 255
- All 5 stages of val_polys now match 100%

## Round Polynomial Format Fix (DONE)
- Converted from Toom-Cook format to Vandermonde format
- Changed all computeRoundPoly functions
- Updated compression/evaluation helpers

## Stage 6 Instance Layout
- Instance 0: BytecodeReadRaf (14 rounds, degree 3)
- Instance 1: HammingBooleanity (8 rounds, degree 3)
- Instance 2: Booleanity (12 rounds, degree 3)
- Instance 3: RamRaVirtual (8 rounds, degree 5)
- Instance 4: LookupsRaSumcheck (8 rounds, degree 5)
- Instance 5: IncClaimReduction (8 rounds, degree 2)

## Current Issue: ALL Stage Claims Mismatch (INVESTIGATING)

### Key Finding: ALL 5 stages of BytecodeReadRaf have `Σ_k F_s[k]*val[k] ≠ opening_claim`

Debug evidence (from /tmp/zolt_stderr7.txt):
- Stage 0 (SpartanOuter): sc != ext, F_s_sum=1 ✓, direct==recomp ✓
- Stage 1 (ProductVirt): sc != ext
- Stage 2 (SpartanShift): sc != ext
- Stage 3 (RegistersRWC): sc != ext
- Stage 4 (RegistersValEval): sc != ext

### Field-level Comparison
- Stage 1: address MATCHES, imm MISMATCHES, most circuit_flags MISMATCH
- Stage 2: Jump MATCHES, Branch MATCHES, IsRdNotZero MISMATCH, WriteLookupToRD MISMATCH

### Key Investigation Results

1. **Streaming round**: Does NOT affect opening claims. r_cycle uses only cycle variables (sumcheck_challenges[1..]), excluding streaming challenge. Opening claims are simple `Σ_c eq(r_cycle,c)*poly(c)`.

2. **Jolt's prover uses EXTERNAL claims** (from opening accumulator), not recomputed from arrays. Array recomputation is test-only assertion.

3. **Val_polys match `compute_val_polys_zolt`** byte-for-byte. This is a MODIFIED version with Zolt-specific encodings.

4. **Imm encoding differs**: Zolt uses per-opcode encoding (fromU64 bitcast for ADDI/JAL etc, truncated u32 for LUI). Standard Jolt uses from_i128 universally. BUT Zolt's R1CS ALSO uses the same per-opcode encoding (computeUnsignedImmediate). So val_polys and R1CS should agree.

5. **Flag differences**: compute_val_polys_zolt uses raw instruction bits; compute_val_polys uses Instruction trait methods. These can produce different flag values.

### Theories to Test
- The R1CS constraint system might not use exactly the same flag computation as the val_poly builder
- There might be cycles where the R1CS witness produces different flag values than the bytecode table entry
- The issue might be in how virtual instructions (VirtualSignExtendWord, VirtualMULI) contribute to the R1CS witness vs the bytecode table

### Next Step
Add diagnostic for Stage 2 (ProductVirtualization, 4 flags only) to compare:
1. `Σ_c eq(r_cycle_s1, c) * R1CS_IsRdNotZero(c)`
2. `Σ_k F_s[k] * bytecode_IsRdNotZero(k)`
3. Opening claim for IsRdNotZero from SpartanProductVirtualization

If (1) == (3) and (1) != (2), then the bytecode table's flag differs from R1CS for some cycles.
If (1) != (3), then the R1CS witness differs from the opening claim.
