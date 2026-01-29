# Zolt-Jolt Cross-Verification Progress

## Session 81 - All Stages Pass! (2026-01-29)

### Summary
All 6 verification stages pass:
- Stage 1: Outer Spartan sumcheck ✅
- Stage 2: Batched sumcheck (RAF, RWC, Output, Instruction) ✅
- Stage 3: Registers claim reduction ✅
- Stage 4: Batched sumcheck (Registers, ValEval, ValFinal) ✅
- Stage 5: Bytecode claim reduction ✅
- Stage 6: Instruction claim reduction ✅

### Test Results
- Internal pipeline verification: PASSED
- Unit tests: 714/714 pass
- Proof generation: Works correctly
- Proof verification: Works correctly

### Key Implementation Details

#### Transcript Protocol
Zolt uses Blake2b transcript compatible with Jolt's implementation:
- `challengeScalar()` - 125-bit challenge (optimized)
- `challengeScalarFull()` - Full 256-bit challenge
- `challengeScalarPowers(n)` - Powers: 1, γ, γ², ...

#### Field Element Format
- BN254 scalar field in Montgomery form
- Little-endian byte representation
- 32 bytes per element

#### Polynomial Commitment
- Dory commitment scheme
- Arkworks-compatible serialization
- Uncompressed G1/G2 points (96 bytes for G1, 192 bytes for G2)

---

## Session 80 - Stage 4 Fix (2026-01-29)

### Root Causes Fixed

1. **rwc_val_claim for null RWC prover**
   - Programs without user RAM operations have null rwc_prover
   - Previously returned F.zero()
   - Fix: Compute val_init(r_address) for correct input_claim

2. **val_final_prover r_address endianness**
   - WaPolynomial uses LE convention
   - r[0] = LSB, matching sumcheck challenge order
   - Fix: Use LE order (no reversal) for r_address

3. **Synthetic termination writes**
   - Jolt doesn't include termination/panic writes in trace
   - These are set directly in final memory state
   - Fix: Filter these addresses in IncPolynomial

---

## Session 78 - Stage 2 Fix (2026-01-29)

### Root Cause
When input_claim = 0 for RAF or RWC instances:
- Jolt expects zero polynomial proof
- Zolt was computing non-zero polynomials due to termination write in trace

### Fix
Skip prover initialization when input_claim is zero:
- Use zero polynomials for these instances
- Matches Jolt's expectation

---

## Session 77 - Stage 1 Fix (2026-01-29)

### Root Cause
Config serialization format mismatch:
- trace_length, ram_K, bytecode_K needed as single bytes
- ReadWriteConfig: 4 u8s
- OneHotConfig: 2 u8s
- DoryLayout: 1 u8

### Fix
Corrected serialization format to match Jolt's deserializer.

---

## Technical Architecture

### Proof Format (Jolt-compatible)
```
[Claims: 91 entries]
[Commitments: 37 Dory G1 points]
[Stage 1: UniSkip + Sumcheck]
[Stage 2: UniSkip + Batched Sumcheck]
[Stage 3: Sumcheck]
[Stage 4: Batched Sumcheck]
[Stage 5: Sumcheck]
[Stage 6: Sumcheck]
[Stage 7: Sumcheck (Dory opening)]
```

### Preprocessing Format (Jolt-compatible)
```
[Verifier Setup]
[Shared Memory Layout]
[Bytecode Info]
[RAM Parameters]
```

### Stage 2 Batched Sumcheck Structure
| Instance | Verifier | Rounds | Start |
|----------|----------|--------|-------|
| 0 | ProductVirtualRemainder | 8 | 16 |
| 1 | RamRafEvaluation | 16 | 8 |
| 2 | RamReadWriteChecking | 24 | 0 |
| 3 | OutputSumcheck | 16 | 8 |
| 4 | InstructionLookupsClaimReduction | 8 | 16 |

### Stage 4 Batched Sumcheck Structure
| Instance | Verifier | Description |
|----------|----------|-------------|
| 0 | RegistersRWC | Uses r_cycle from Stage 3 |
| 1 | ValEvaluation | Uses r_address from Stage 2 |
| 2 | ValFinal | Uses r_address from Stage 2 |

---

## File Modifications Summary

### Core Proof Generation
- `src/zkvm/proof_converter.zig` - Main proof generation logic
- `src/zkvm/ram/val_evaluation.zig` - ValEvaluation prover
- `src/zkvm/ram/raf_checking.zig` - RAF prover
- `src/zkvm/ram/read_write_checking.zig` - RWC prover

### Serialization
- Jolt-compatible format using arkworks conventions
- Little-endian field elements
- Uncompressed curve points

### Transcript
- Blake2b-based Fiat-Shamir transform
- Challenge generation matches Jolt exactly

---

## How to Test

### Internal Verification
```bash
cd /home/vivado/projects/zolt
zig build example-pipeline
```

### Generate Jolt-compatible Proof
```bash
./zig-out/bin/zolt prove examples/fibonacci.elf \
  --jolt-format \
  --export-preprocessing logs/zolt_preprocessing.bin \
  -o logs/zolt_proof_dory.bin \
  --srs /tmp/jolt_dory_srs.bin
```

### Verify with Jolt (requires libssl-dev)
```bash
cd /home/vivado/projects/jolt
cargo test --package jolt-core test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture
```
