# Zolt → Jolt Cross-Verification Progress

## STATUS: Upstream alignment in progress

### Current State (Mar 6 2026)
- Migrating from MatteoMer/jolt fork to vanilla a16z/jolt upstream
- All transcript labels, proof format, R1CS constraints updated
- Proof deserialization passes through all fields against upstream verifier
- **NEXT**: Full end-to-end verification test against upstream jolt-verifier

## COMPLETED: Upstream Alignment Changes

### Transcript Labels (Fiat-Shamir domain separation)
- All blake2b transcript calls now include labels matching upstream
- Labels: sumcheck_claim, sumcheck_poly, opening_claim, dory_serde, dory_group, etc.
- appendScalars: removed begin/end markers, uses rawAppendLabelWithLen
- Preamble: max_input_size, max_output_size, heap_size, inputs, outputs, panic, ram_K, trace_length

### Proof Serialization Format
- Reordered: commitments first, then stages with enum discriminant bytes
- bytecode_K removed
- opening_claims moved after untrusted_advice_commitment

### Type System Updates
- SumcheckId: merged RamValEvaluation+RamValFinalEvaluation→RamValCheck (COUNT=23)
- VirtualPolynomial: removed WritePCtoRD/WriteLookupOutputToRD, byte values match upstream
- InstructionFlags: 6 variants only (removed fork-only IsRdNotZero)
- MemoryLayout: memory_size→heap_size, memory_end→heap_end

### R1CS & Circuit Fixes
- Constraint 13: JAL/JALR RdWriteValue fix for rd=x0
- PRODUCT_UNIQUE_FACTOR_VIRTUALS: reordered to match upstream 8-entry ordering
- Opening claims: corrected polynomial IDs (OpFlags indices, InstructionFlags(Branch))

### Infrastructure
- jolt-verifier/ crate using upstream a16z/jolt with --diagnose mode
- Preprocessing: blindfold_setup None byte appended

## REMAINING TODO

### Must Do
- [ ] Test full end-to-end verification (prove fibonacci → verify with jolt-verifier)
- [ ] Debug any remaining verification failures
- [ ] Test all 8 programs against upstream verifier
- [ ] Clean up debug_verbose flags and diagnostic prints in stage5/6 provers

### Nice to Have
- [ ] Remove the jolt/ fork directory (replace fully with jolt-verifier/)

## BUILD & TEST COMMANDS
```bash
# Build Zolt
zig build -Doptimize=ReleaseFast

# Prove and export
./zig-out/bin/zolt prove examples/<program>.elf --jolt-format -o /tmp/proof.bin --export-preprocessing /tmp/preproc.bin
cp /tmp/proof.bin /tmp/zolt_proof_dory.bin
cp /tmp/preproc.bin /tmp/zolt_preprocessing.bin
cp /tmp/preproc.bin.ram /tmp/zolt_preprocessing.bin.ram

# Verify with upstream jolt-verifier
cd jolt-verifier && cargo run --release -- --proof /tmp/zolt_proof_dory.bin --preprocessing /tmp/zolt_preprocessing.bin

# Diagnose deserialization issues
cd jolt-verifier && cargo run --release -- --proof /tmp/zolt_proof_dory.bin --preprocessing /tmp/zolt_preprocessing.bin --diagnose
```

## KEY FILES
- `src/zkvm/proof_converter.zig` — Main proof conversion logic
- `src/zkvm/jolt_types.zig` — Proof types, SumcheckId, VirtualPolynomial, OpeningId
- `src/zkvm/jolt_serialization.zig` — Proof serialization
- `src/zkvm/r1cs/constraints.zig` — R1CS constraint definitions
- `src/transcripts/blake2b.zig` — Blake2b transcript with labels
- `src/zkvm/spartan/stage5_prover.zig` — InstructionReadRaf sumcheck
- `src/zkvm/spartan/stage6_prover.zig` — BytecodeReadRaf val_poly
- `jolt-verifier/src/main.rs` — Standalone upstream verifier
