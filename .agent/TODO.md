# Stage 2 Implementation - Current State

## Status: Ready for Integration

### Completed Components

1. **ProductVirtualRemainderProver** (`src/zkvm/spartan/product_remainder.zig`)
   - Skeleton implemented with:
     - `init(allocator, r0, tau, uni_skip_claim, cycle_witnesses)`
     - `numRounds()` - returns n_cycle_vars
     - `computeRoundPolynomial()` - returns compressed [c0, c2, c3]
     - `bindChallenge(challenge)` - binds left/right/split_eq
     - `computeOpeningClaims()` - computes 8 factor evaluations

2. **BatchedSumcheckProver** (`src/zkvm/batched_sumcheck.zig`)
   - Infrastructure for combining 5 instances
   - Handles different round counts via scaling
   - Properly combines polynomials with batching coefficients

3. **Import added** to proof_converter.zig

### Integration Task

The Stage 2 sumcheck proof generation in `convertWithTranscript` needs to:

1. After Stage 2 UniSkip completes (have: r0, uni_skip_claim, tau):

2. Initialize 5 sumcheck instances:
   ```
   instances[0] = ProductVirtualRemainder(r0, tau, uni_skip_claim, witnesses)
   instances[1] = ZeroInstance(log_ram_k rounds)  // RamRafEvaluation
   instances[2] = ZeroInstance(log_ram_k + n_cycle_vars rounds)  // RamReadWriteChecking
   instances[3] = ZeroInstance(log_ram_k rounds)  // OutputSumcheck
   instances[4] = ZeroInstance(n_cycle_vars rounds)  // InstructionLookupsClaimReduction
   ```

3. Run batched sumcheck protocol:
   - Append 5 input_claims to transcript
   - Sample 5 batching coefficients
   - For each round:
     - Compute combined polynomial from all instances
     - Append to transcript
     - Sample challenge
     - Bind all instances
   - Output: round_polys[], challenges[]

4. Cache opening claims:
   - ProductVirtualRemainder: 8 factor evaluations at r_cycle
   - Others: zeros

### Key Files to Modify

- `src/zkvm/proof_converter.zig` (lines ~1110-1122)
  - Replace `generateZeroSumcheckProof` with real batched sumcheck
  - Add proper opening claims computation

### Testing

After integration:
```bash
# Generate proof and verify with Jolt
./zig-out/bin/zolt prove --jolt-format -o /tmp/proof.bin examples/simple.elf
cd /Users/matteo/projects/jolt && cargo test test_verify_zolt_proof
```

### Why This Should Work

For programs without RAM/lookups:
- ProductVirtualRemainder: non-zero input, real polynomials, real opening claims
- RamRafEvaluation: zero input → zero output → zero expected
- RamReadWriteChecking: zero input → zero output → zero expected
- OutputSumcheck: zero input (always) → zero output → zero expected
- InstructionLookupsClaimReduction: zero input → zero output → zero expected

The verification equation `output_claim == expected_output_claim` passes when:
- Real instance: real sumcheck matches real expected
- Zero instance: 0 == 0

## Next Session Tasks

1. Implement `generateStage2SumcheckProof` function
2. Wire up ProductVirtualRemainder prover
3. Create ZeroSumcheckInstance for other 4 instances
4. Test end-to-end with Jolt verifier
