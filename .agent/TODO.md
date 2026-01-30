# Zolt-Jolt Compatibility Implementation

## Status: IN PROGRESS - Stage 5 Sumcheck Verification Failing

## Progress Summary

### Fixed Issues
- ✅ SumcheckId COUNT mismatch (22 vs 24) - Added missing `AdviceClaimReductionCyclePhase` and `AdviceClaimReduction`
- ✅ Config serialization format - Changed from usizes to proper u8 fields for ReadWriteConfig, OneHotConfig, DoryLayout
- ✅ Claims deserialization now works correctly

### Verified Stages
- Stage 1: Deserializes correctly
- Stage 2: Deserializes correctly
- Stage 3: Deserializes correctly
- Stage 4: Deserializes correctly
- **Stage 5: FAILS - "Sumcheck verification failed"**
- Stage 6: Not tested yet
- Stage 7: Not tested yet

## Current Issue: Stage 5 Sumcheck Verification

The Stage 5 batched sumcheck has 3 instances:
1. **RegistersValEvaluation** (8 rounds) - Uses trace data
2. **RamRaClaimReduction** (24 rounds) - Zero for Fibonacci (no RAM ops)
3. **LookupsReadRaf** (136 rounds) - **NOT PROPERLY IMPLEMENTED**

### Why It Fails

The current "constant polynomial" approach for LookupsReadRaf doesn't work:

1. We send constant polynomials: `p(x) = claim/2` for all x
2. This satisfies `p(0) + p(1) = claim` at each round
3. After 136 rounds: `output_claim = lookups_input * 2^(-136)` (very small but non-zero)
4. But `expected_output_claim = eq_eval * ra_claim * (val_claim + gamma * raf_claim)`
5. With all claims set to zero, `expected_output_claim = 0`
6. `output_claim ≠ expected_output_claim` → verification fails

### What's Needed

The LookupsReadRaf sumcheck proves:
```
rv(r_reduction) + γ·left_op(r_reduction) + γ²·right_op(r_reduction)
= Σ_j Σ_k [ eq(j; r_reduction) · ra(k, j) · (Val_j(k) + γ · RafVal_j(k)) ]
```

To implement properly:
1. Build `ra` polynomial from trace (lookup table indices per cycle)
2. Build `Val` polynomial (lookup table values)
3. Build `RafVal` polynomial (RAF operand contributions)
4. Run sumcheck over 136 variables (128 address + 8 cycle)
5. Compute opening claims: InstructionRa(0..7), LookupTableFlag(0..41), InstructionRafFlag

### Files to Modify
- `src/zkvm/spartan/stage5_prover.zig` - Main Stage 5 prover
- `src/zkvm/spartan/instruction_read_raf.zig` - LookupsReadRaf implementation (stub exists)

### Alternative Approaches
1. **Full implementation**: Implement proper LookupsReadRaf sumcheck (~500-1000 lines)
2. **Simplified for Fibonacci**: Only compute for ADD/ADDI tables, zero others
3. **Skip Stage 5**: Mark as partial pass, continue to other stages

## Test Commands

```bash
# Generate proof
./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format -o /tmp/zolt_proof_dory.bin

# Verify with Jolt
cd /home/vivado/projects/jolt
cargo test --package jolt-core test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture
```

## Key Files
- `src/zkvm/spartan/stage5_prover.zig` - Zolt Stage 5 prover
- `src/zkvm/proof_converter.zig` - Proof generation and conversion
- `jolt-core/src/zkvm/instruction_lookups/read_raf_checking.rs` - Jolt LookupsReadRaf reference
- `jolt-core/src/subprotocols/sumcheck.rs` - Jolt batched sumcheck verifier
