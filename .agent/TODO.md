# Zolt → Jolt Cross-Verification Progress

## STATUS: ALL 8 PROGRAMS FULLY VERIFIED (Stages 1-5 PASS)

### Current State (Mar 5 2026)
- All 8 programs prove + verify ALL stages against Jolt
- **Stage 5 FIXED**: NOP/padding cycles now properly handled in InstructionReadRaf sumcheck

## COMPLETED: Stage 5 Fix (InstructionReadRaf NOP handling)

### Root Cause
NOOPs (ADDI x0,x0,0) and padding cycles were skipped entirely in the lookup processing
loop of stage5_prover.zig. This caused:
1. `cycle_table_indices[j]` = -1 instead of 0 (RangeCheck)
2. `cycle_is_identity_path[j]` = false instead of true
3. Missing contributions to Q arrays during 128 address rounds
4. Missing RAF contributions during cycle round rematerialization at round 128
5. Wrong opening claims (table flags, raf flag didn't include NOP/padding)

### Fix (3 changes in stage5_prover.zig)
1. Removed `continue` for NOOPs in lookup processing loop — let them process normally
   through the ADDI code path (sets table_idx=0/RangeCheck, is_identity_path=true)
2. Added padding cycle handling (trace_len..T) — set table_idx=0, is_identity_path=true
3. Fixed rematerialization to add RAF for ALL cycles (not just those with tables),
   matching Jolt's init_log_t_rounds() behavior
4. Fixed opening claims to include ALL T cycles (removed trace_len skip)

## COMPLETED: Termination Sequence Fix (Option A: JAL-to-self)

### What Changed
The termination sequence now has 4 entries (was 3): NoOp, LUI, ADDI, SB, **JAL**

**SB anchor (tbpc+2)**: Changed from `VI=false, DNUPC=true` to `VI=true, DNUPC=false`
- Now matches vanilla Jolt's `circuit_flags()` for SD with `vsr=Some(0)`

**JAL-to-self (tbpc+3)**: New entry `JAL x0, 0` at address=4
- Jump=1 disables constraint 16 for JAL→NoOp transition

## Jolt Version Gap (secondary issue)

Fork (`MatteoMer/jolt`) is 71 commits behind `a16z/jolt` upstream (`807c360d`).
Key upstream changes: `IsLastInSequence` (37th R1CS input), BlindFold ZK, x0 fix.

## BUILD & TEST COMMANDS
```bash
# Build Zolt
zig build -Doptimize=ReleaseFast

# Prove and export
./zig-out/bin/zolt prove examples/<program>.elf --jolt-format -o /tmp/proof.bin --export-preprocessing /tmp/preproc.bin
cp /tmp/proof.bin /tmp/zolt_proof_dory.bin
cp /tmp/preproc.bin /tmp/zolt_preprocessing.bin
cp /tmp/preproc.bin.ram /tmp/zolt_preprocessing.bin.ram

# Verify with Jolt (all stages pass)
cd jolt && cargo test --package jolt-core --features zolt-debug \
  "test_verify_zolt_proof_with_zolt" -- --include-ignored
```

## KEY FILES
- `src/zkvm/spartan/stage5_prover.zig` — InstructionReadRaf sumcheck (FIXED)
- `src/zkvm/spartan/stage6_prover.zig` — BytecodeReadRaf val_poly construction
- `src/zkvm/lookup_table/prefix_suffix_prover.zig` — Read-checking and RAF helpers
- `src/zkvm/r1cs/constraints.zig` — R1CS constraints, witness generation
- `src/zkvm/preprocessing.zig` — Preprocessing export
- `jolt/jolt-core/src/zkvm/instruction_lookups/read_raf_checking.rs` — Reference Stage 5 prover
