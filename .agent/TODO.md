# Zolt-Jolt Compatibility Implementation

## Status: IN PROGRESS - Stage 5 LookupsReadRaf Sumcheck Needed

## Progress Summary

### Completed
- ✅ SumcheckId COUNT mismatch (22 → 24) - Added missing `AdviceClaimReductionCyclePhase` (20) and `AdviceClaimReduction` (21)
- ✅ Config serialization format - Fixed to use proper u8 fields for ReadWriteConfig, OneHotConfig, DoryLayout
- ✅ Claims deserialization works correctly
- ✅ Stage 1-4 pass verification (confirmed via debug output)
- ✅ RegistersValEvaluation sumcheck (Stage 5 Instance 0) implemented with trace data
- ✅ RamRaClaimReduction (Stage 5 Instance 1) works (zero for Fibonacci without RAM)

### Current Issue: Stage 5 Instance 2 (LookupsReadRaf)

Stage 5 verification fails at sumcheck final claim check:
```
Sumcheck verification failed!
  output_claim:   [0b, 9c, ac, 06, ...]
  expected_claim: [d7, 15, 11, 32, ...]
```

The "constant polynomial" approach for LookupsReadRaf doesn't work because:
1. Constant polynomials reduce `lookups_input` to a non-zero tiny value after 136 rounds
2. The `expected_output_claim` formula computes a different value from opening claims
3. These must match for verification to pass

### What's Needed: Full LookupsReadRaf Implementation

The LookupsReadRaf sumcheck proves:
```
rv(r_reduction) + γ·left_op(r_reduction) + γ²·right_op(r_reduction)
= Σ_j Σ_k [ eq(j; r) · ra(k, j) · (Val(k) + γ · RafVal(k)) ]
```

Where:
- `eq(j; r)` = equality polynomial over log_T cycle variables
- `ra(k, j)` = selector (1 when cycle j's lookup key equals k)
- `Val(k)` = lookup table value at address k
- `RafVal(k)` = RAF operand contribution

The sumcheck has 136 rounds (128 address + 8 cycle variables).

Final opening claims must be computed:
- `InstructionRa(0..7)` - 8 virtual RA polynomial chunks
- `LookupTableFlag(0..41)` - 42 lookup table selectors
- `InstructionRafFlag` - RAF operand flag

### Simplified Approach for Fibonacci

Since Fibonacci only uses a few instructions (LUI, ADDI, ADD, JAL, BNE, etc.):
1. Most LookupTableFlag claims can be zero (unused tables)
2. Only compute ra for the tables actually used
3. Can batch operations over fewer actual lookups

### Test Commands

```bash
# Generate proof
./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format -o /tmp/zolt_proof_dory.bin

# Verify with Jolt (with debug output)
cd /home/vivado/projects/jolt
cargo test --package jolt-core --features zolt-debug test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture
```

### Key Files
- `src/zkvm/spartan/stage5_prover.zig` - Stage 5 prover (needs LookupsReadRaf)
- `src/zkvm/spartan/instruction_read_raf.zig` - Stub exists, needs implementation
- `jolt-core/src/zkvm/instruction_lookups/read_raf_checking.rs` - Jolt reference implementation

### Debug Findings

From Jolt debug output:
- Stage 4 RamValEvaluation shows `inc_claim = 0` (correct - Fibonacci has no RAM writes)
- Stage 4 RamValFinal shows non-zero inc_claim (final termination write)
- Stage 5 initial_claim is non-zero (lookups_input contributes)
- After 136 rounds, output_claim ≠ expected_claim

SESSION_ENDING - Context is running low. Next session should focus on implementing LookupsReadRaf sumcheck in `instruction_read_raf.zig`.
