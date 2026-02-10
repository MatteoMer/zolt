# Zolt → Jolt Verification Progress

## Current Status
Stages 1-6 PASS! Stage 7 (HammingWeightClaimReduction) FAILS.

## Stage 7 Analysis

### Structure
Stage 7 is HammingWeightClaimReduction - a single batched sumcheck over `log_k_chunk=4` rounds.
For each `ra_i` (N=38 total: 32 InstructionRa + 2 BytecodeRa + 4 RamRa), it fuses:
1. HammingWeight: `G_i(k) should sum to 1` for instruction/bytecode, or `ram_hw_factor` for RAM
2. Booleanity: `G_i(ρ) * eq(r_addr_bool, ρ) = bool_claim_i`
3. Virtualization: `G_i(ρ) * eq(r_addr_virt_i, ρ) = virt_claim_i`

Initial claim = Σ_i γ^{3i} * hw_claim_i + γ^{3i+1} * bool_claim_i + γ^{3i+2} * virt_claim_i

### Failures
- Indices 0-14: all match (InstructionRa chunks 0-14)
- Index 15: G*eq_virt and G*eq_bool mismatch
- Indices 24-37: all mismatch (InstructionRa 24-31, BytecodeRa 0-1, RamRa 0-3)
- Round polynomials: p(0)+p(1) ≠ claim at ALL rounds (SANITY check fails from R0)
- input_claim different from prover expected (prover expected=[8abde91164b66235], sumcheck output=[09a3ad4e0ab587c4])

### Root Cause Hypothesis
The G tables for some indices don't correctly represent `G_i(k) = Σ_j eq(r_cycle, j) * ra_i(k, j)`.
Since the eq_cycle_bool_phase2 (BE ordering) changed, the G tables in Stage 7 might also need
the same ordering.

Actually, the Stage 7 G tables are SEPARATE from Stage 6's booleanity G tables.
The Stage 7 G tables use eq(r_cycle_stage6, j) with the r_cycle from Stage 6's booleanity opening.
These are constructed in the Stage 7 prover code, not reused from Stage 6.

Need to check:
1. Are the Stage 7 G tables using the correct eq_cycle?
2. Are the virt_claims and bool_claims correctly read from the proof?
3. Is the eq_bool evaluation correct? (should be eq(r_addr_bool, ρ) where ρ = stage7 challenges)
4. Is the eq_virt evaluation correct?

### Key Data
- Transcript matches before Stage 7: both show { 69 25 be 13 37 59 0f 37 }
- gamma matches: [b4,09,5c,71,3b,fb,72,f6]
- N=38, log_k_chunk=4, k_chunk=16, T=256

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
- [x] Fix NoOp bytecode entry is_interleaved flag
- [x] Stage 6 InstructionRafFlag mismatch debugged and fixed
- [x] Fix eq_cycle table direction (use BE, not LE) -- then unified to single table
- [x] Compute booleanity_ra_claims properly (non-zero ra_i(ρ) values)
- [x] Fix Booleanity Phase 1→Phase 2 transition (unified eq_cycle table)
- [x] Stage 6 PASSES

## In Progress
- [ ] Fix Stage 7 HammingWeightClaimReduction sumcheck
  - G table verification mismatches for some indices
  - Round polynomial p(0)+p(1) ≠ claim at all rounds
  - Need to investigate G table construction and eq evaluations

## Pending
- [ ] End-to-end verification test (all stages pass)

## Test Commands
```bash
cd /home/vivado/projects/zolt && zig build -Doptimize=ReleaseFast
./zig-out/bin/zolt prove examples/fibonacci.elf --trace-length 64 -o /tmp/zolt_proof.bin --jolt-format --export-preprocessing /tmp/zolt_preprocessing.bin
cd /home/vivado/projects/jolt && RAYON_NUM_THREADS=1 cargo run --release --features zolt-debug --manifest-path examples/fibonacci/Cargo.toml -- --verify-zolt-proof /tmp/zolt_proof.bin --zolt-preprocessing /tmp/zolt_preprocessing.bin.ram
```
