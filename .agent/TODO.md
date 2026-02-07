# Zolt-Jolt Compatibility Implementation

## Status: STAGES 1-5 PASS ✅ — Stage 6 needs real sumcheck proofs

### Completed
1. ✅ Fix polynomial chain divergence in Stage 5 (RAF prefix MLE materialization)
2. ✅ Fix bytecode_K mismatch (hardcoded 65536 → computed from decoded instructions)
3. ✅ Add missing Stage 6 opening claims (BytecodeRa, InstructionRa, RamRa)
4. ✅ Export bytecode code_size in preprocessing for Jolt verifier override

### Current Issue: Stage 6 Sumcheck

Stage 6 is a batched sumcheck with 6 instances. Currently generates zero proofs.
The initial_claim is NON-ZERO because instances 0 (BytecodeReadRaf), 3 (RamRaVirtual),
4 (LookupsRaVirtual), and 5 (IncClaimReduction) have non-zero input_claims from
the opening accumulator (set by Stages 1-5).

**Instance input_claims:**
- Instance 0 (BytecodeReadRaf): NON-ZERO (0x2e3ea1...) — RLC of virtual poly openings from stages 1-5
- Instance 1 (HammingBooleanity): ZERO — zero-check
- Instance 2 (Booleanity): ZERO — zero-check
- Instance 3 (RamRaVirtual): NON-ZERO (0x2388a2...) — from Stage 5 RamRaClaimReduction
- Instance 4 (LookupsRaVirtual): NON-ZERO (0x1a73e2...) — RLC of InstructionRa openings from Stage 5
- Instance 5 (IncClaimReduction): NON-ZERO (0x1a20f4...) — combines RamInc/RdInc from stages 2,4,5

**Parameters (fibonacci, bytecode_K=32, T=256):**
- stage6_max_rounds = 13 (5 + 8)
- max_degree = 5
- bytecode_d = 2, ram_d = 4, instruction_d = 32

### Implementation Needed

To generate correct Stage 6 proofs, need to implement:

1. **BytecodeReadRafSumcheckProver** (13 rounds, degree 3)
   - Polynomial: Σ_i γ^i · [Val_i(addr) + RAF_i] · eq(r_cycle_i, x) · ∏ ra_j
   - Needs: bytecode val polynomials, eq evaluations, RA chunks

2. **HammingBooleanitySumcheckProver** (8 rounds, degree 3)
   - Polynomial: eq(r_cycle, x) · (H(x)² - H(x))
   - Zero for valid traces, but still needs proper eq factors

3. **BooleanitySumcheckProver** (12 rounds, degree 3)
   - Polynomial: eq(r_addr||r_cycle, x) · Σ γ^{2i} · (ra_i² - ra_i)
   - Zero for valid traces, but still needs proper eq factors

4. **RamRaVirtualSumcheckProver** (8 rounds, degree 5)
   - Polynomial: eq(r_cycle, x) · ∏ ra_i(x)
   - Needs: RAM address chunks, eq evaluation

5. **LookupsRaVirtualSumcheckProver** (8 rounds, degree 5)
   - Polynomial: eq(r_cycle, x) · Σ γ^i · ∏_j ra_{i·M+j}(x)
   - Needs: instruction address chunks, eq evaluation

6. **IncClaimReductionSumcheckProver** (8 rounds, degree 2)
   - Polynomial: RamInc(x) · eq_combined + γ² · RdInc(x) · eq_combined
   - Needs: RamInc/RdInc values, eq evaluations from stages 2,4,5

### Architecture Decision
These provers need to be implemented in proof_converter.zig as part of the
`convertWithTranscript` function. Each prover materializes its polynomial
evaluations over the trace, then runs the standard sumcheck protocol.

### Test Commands
```bash
zig build -Doptimize=ReleaseFast
./zig-out/bin/zolt prove examples/fibonacci.elf --trace-length 64 -o /tmp/zolt_proof.bin --jolt-format --export-preprocessing /tmp/zolt_preprocessing.bin
cd /home/vivado/projects/jolt && RAYON_NUM_THREADS=1 cargo run --release --features zolt-debug --manifest-path examples/fibonacci/Cargo.toml -- --verify-zolt-proof /tmp/zolt_proof.bin --zolt-preprocessing /tmp/zolt_preprocessing.bin.ram
```

### Remaining Tasks
1. [IN PROGRESS] Implement Stage 6 batched sumcheck provers
2. [PENDING] Implement Stage 7 (HammingWeightClaimReduction)
3. [PENDING] Clean up debug prints
4. [PENDING] Verify end-to-end proof generation (all stages pass)
5. [PENDING] Run full test suite (578+ tests)
