# Stage 2 Implementation - Status

## Current Status: Stage 2 Wired Up, Ready for Testing

### Completed ✅

1. **ProductVirtualRemainderProver** (`src/zkvm/spartan/product_remainder.zig`)
   - Fuses 5 product constraints into left/right polynomials
   - Computes cubic round polynomials [c0, c2, c3]
   - Extracts 8 unique factor polynomial values per cycle

2. **BatchedSumcheckProver** (`src/zkvm/batched_sumcheck.zig`)
   - Combines 5 instances with different round counts
   - Handles batching protocol matching Jolt exactly

3. **generateStage2BatchedSumcheckProof** in proof_converter.zig
   - Replaces zero sumcheck with real batched sumcheck
   - ProductVirtualRemainder uses real prover
   - Other 4 instances contribute zero claims

4. **All 712+ tests pass**

### In Progress 🔄

1. **Test with Jolt verifier**
   - Generate proof: `./zig-out/bin/zolt prove --jolt-format -o /tmp/proof.bin examples/simple.elf`
   - Verify with Jolt: `cd /Users/matteo/projects/jolt && cargo test test_verify_zolt_proof`

### Known Issues / TODOs 📋

1. **ProductVirtualRemainder prover**
   - Polynomial decompression may need fixing (lines 1364-1377 in proof_converter.zig)
   - The evalsToCompressed / compressed format handling

2. **Opening claims for ProductVirtualRemainder**
   - Need to compute 8 factor evaluations at r_cycle
   - Currently set to zeros, need real values

3. **Other 4 Stage 2 instances**
   - Currently use zero claims (valid for simple programs)
   - For full programs, need to implement:
     - RamRafEvaluation
     - RamReadWriteChecking
     - OutputSumcheck
     - InstructionLookupsClaimReduction

### Test Commands

```bash
# Run Zolt tests
zig build test --summary all

# Generate proof
zig build && ./zig-out/bin/zolt prove --jolt-format -o /tmp/proof.bin examples/simple.elf

# Verify with Jolt (in jolt directory)
cd /Users/matteo/projects/jolt
cargo test test_verify_zolt_proof -- --ignored --nocapture
```

### Stage 2 Architecture Reference

```
Stage 2 = BatchedSumcheck([
    ProductVirtualRemainder,           // n_cycle_vars rounds, degree 3
    RamRafEvaluation,                  // log_ram_k rounds, degree 2
    RamReadWriteChecking,              // log_ram_k + n_cycle_vars rounds (MAX)
    OutputSumcheck,                    // log_ram_k rounds, degree 3
    InstructionLookupsClaimReduction,  // n_cycle_vars rounds, degree 2
])
```

### Why Zero Claims Should Work (for simple programs)

For programs without RAM/lookups:
- ProductVirtualRemainder: non-zero input, real polynomials
- RamRafEvaluation: zero input → zero output → 0 == 0 ✓
- RamReadWriteChecking: zero input → zero output → 0 == 0 ✓
- OutputSumcheck: zero input (always) → zero output → 0 == 0 ✓
- InstructionLookupsClaimReduction: zero input → zero output → 0 == 0 ✓
