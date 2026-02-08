# Zolt → Jolt Verification Progress

## Current Status
Stages 1-2 PASS, Stage 3 FAILS (Spartan Shift + InstructionInput + RegistersClaim)

## Completed
- [x] Store Imm as unsigned u64 for identity-path AddOperands instructions
- [x] Verify Stage 1 R1CS constraint 7 passes
- [x] Verify Stage 5 output_claim matches
- [x] Fix IncClaimReduction w1/w2 mismatch (RdInc polynomial)
- [x] Fix BytecodeReadRaf - update raw_words export for 3 termination entries
- [x] Fix BytecodeReadRaf - update Jolt verifier for 3 termination entries
- [x] Fix bytecode entry k=0 flags (DoNotUpdateUnexpandedPC + IsNoop)
- [x] Fix termination R1CS witness flags (VirtualInstruction for LUI/ADDI only, not SB)
- [x] Fix `populateEntryFromInstruction` to reset flags (clear NoOp defaults)

## In Progress
- [ ] Debug Stage 3 failure after termination flag fixes

## Key Changes Made This Session

### 1. Bytecode entry initialization (stage6_prover.zig)
- All entries initialized with `DoNotUpdateUnexpandedPC=true, IsNoop=true` (matching Jolt's NoOp)
- `populateEntryFromInstruction` now resets all flags before setting instruction-specific ones

### 2. Termination bytecode entries (stage6_prover.zig)
- LUI (vsr=2): VirtualInstruction=true, DoNotUpdateUnexpandedPC=true
- ADDI (vsr=1): VirtualInstruction=true, DoNotUpdateUnexpandedPC=true
- SB (vsr=0): DoNotUpdateUnexpandedPC=true ONLY (no VirtualInstruction - would violate constraint 17)

### 3. Termination R1CS witness (constraints.zig)
- Dummy noop: DoNotUpdateUnexpandedPC=1, FlagIsNoop=1 (unchanged, matches k=0 NoOp entry)
- LUI/ADDI (vsr>0): DoNotUpdateUnexpandedPC=1, FlagVirtualInstruction=1, FlagIsNoop=0
- SB (vsr=0): DoNotUpdateUnexpandedPC=1, FlagVirtualInstruction=0, FlagIsNoop=0

### 4. Jolt verifier (read_raf_checking.rs)
- `noop()` now sets DoNotUpdateUnexpandedPC=true, IsNoop=true
- `termination_entry_virtual()` for LUI/ADDI (VirtualInstruction + DoNotUpdateUnexpandedPC)
- `termination_entry_anchor()` for SB (DoNotUpdateUnexpandedPC only)

## Stage 3 Investigation
- Stage 3 = Spartan Shift + Instruction Input + Registers Claim Reduction
- 3 batched instances, 8 rounds
- output_claim != expected_claim after final round
- R1CS constraints all satisfied (0 violations)
- Need to investigate: is there a mismatch in shift/instruction/register polynomial evaluations?
- Possible cause: the FlagIsNoop change for LUI/ADDI/SB affects the shift sumcheck's
  is_noop polynomial, which was previously 1 for these cycles and is now 0.
  This changes the shift sumcheck claim and the opening claims flowing forward.

## Pending
- [ ] Implement Stage 7 (HammingWeightClaimReduction)
- [ ] End-to-end verification test

## Test Commands
```bash
cd /home/vivado/projects/zolt && zig build -Doptimize=ReleaseFast
./zig-out/bin/zolt prove examples/fibonacci.elf --trace-length 64 -o /tmp/zolt_proof.bin --jolt-format --export-preprocessing /tmp/zolt_preprocessing.bin
cd /home/vivado/projects/jolt && RAYON_NUM_THREADS=1 cargo run --release --features zolt-debug --manifest-path examples/fibonacci/Cargo.toml -- --verify-zolt-proof /tmp/zolt_proof.bin --zolt-preprocessing /tmp/zolt_preprocessing.bin.ram
```
