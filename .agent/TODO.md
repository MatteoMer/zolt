# Zolt-Jolt Compatibility Implementation

## Status: Session 38 - PROGRESS! Stage 4 Now Passes, Stage 5 Fails

## Major Progress This Session!

**Stage 4 is now passing!** The verification has progressed to **Stage 5** (InstructionReadRaf), which now fails.

## Current Failure: Stage 5 (InstructionReadRaf)

### Error Details
```
Sumcheck verification failed!
  output_claim:   [af, 51, 7b, 30, ff, 29, 91, 26, ...]
  expected_claim: [f0, c1, c7, e7, 7e, fd, c3, 3b, ...]
Verification failed: Stage 5
```

### Stage 5 Architecture
Stage 5 batches THREE sumcheck instances:

1. **RegistersValEvaluation** (8 rounds)
   - Degree: 3
   - Proves register value consistency

2. **RamRaClaimReduction** (16 rounds = log_K + log_T)
   - Degree: 2
   - Consolidates four RAM RA claims

3. **InstructionReadRaf** (136 rounds = LOG_K + log_T = 128 + 8)
   - Degree: 10 (8 virtual RA polynomials + 2)
   - Main instruction lookups sumcheck
   - LOG_K = XLEN * 2 = 128 for 64-bit instructions
   - **This is the component that's failing**

### Key Variables in InstructionReadRaf
- **r_reduction**: Original cycle evaluation point from InstructionClaimReduction (Stage 2)
- **r_cycle_prime**: Last 8 challenges from InstructionReadRaf sumcheck (reversed)
- **eq_eval_r_reduction**: `EqPolynomial::mle(r_reduction, r_cycle_prime)` - bridges Stage 2 to Stage 5

### Potential Causes of Mismatch
1. **Transcript divergence from earlier stage**
2. **r_reduction from Stage 2 not matching**
3. **Virtual polynomial claims mismatch (ra_claims, table_flag_claims)**
4. **Challenge normalization/reversal issues**

### Working Test Commands

```bash
# Build optimized (MUCH faster!)
zig build -Doptimize=ReleaseFast

# Generate proof (~13 seconds with optimization)
./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format -o /tmp/zolt_proof_dory.bin --export-preprocessing /tmp/zolt_preprocessing.bin --trace-length 64

# Verify with Jolt
cd /home/vivado/projects/jolt && cargo test -p jolt-core --features zolt-debug --lib test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture
```

### Next Steps for Next Session

1. **Compare r_reduction values** between Zolt and Jolt
   - Zolt stores this in Stage 2 (InstructionClaimReduction)
   - Jolt retrieves it from opening_accumulator

2. **Compare ra_claims product**
   - Check each ra_claim[0..7] matches
   - Verify the product computation

3. **Check table_flag_claims**
   - Non-zero for tables 0, 1, 9 based on instruction usage
   - Verify table MLE evaluations match

4. **Debug eq_eval_r_reduction**
   - Compare r_reduction (8 elements)
   - Compare r_cycle_prime (8 elements)
   - Verify eq polynomial evaluation matches

### Key Files

**Zolt Stage 5:**
- `src/zkvm/spartan/stage5_prover.zig` (LOOKUPS_LOG_K=128 at line 45)
- `src/zkvm/proof_converter.zig` (Stage 5 around line 2780)

**Jolt Stage 5:**
- `jolt-core/src/zkvm/verifier.rs` (lines 383-413)
- `jolt-core/src/zkvm/instruction_lookups/read_raf_checking.rs` (expected_output_claim at lines 1326-1476)

### Session Summary

- Built optimized Zolt binary (22MB vs 48MB, 10x faster proof generation)
- Generated fresh proof + preprocessing files
- **Stage 4 now passes!** (RegistersRWC, RamValEvaluation, ValFinal)
- Stage 5 fails at InstructionReadRaf sumcheck
- Root cause likely in r_reduction or virtual polynomial claims

SESSION_ENDING - Major progress: Stage 4 passes! Stage 5 (InstructionReadRaf) now fails. Next session: debug r_reduction and virtual polynomial claims comparison.
