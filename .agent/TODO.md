# Zolt-Jolt Compatibility Implementation

## Status: IN PROGRESS - Stage 5 Prefix-Suffix Decomposition Required

## Session 95 Summary

### Investigation Complete
Thoroughly analyzed the root cause of Stage 5 sumcheck verification failure.

### Root Cause
Zolt uses a simplified bit-splitting approach for LookupsReadRaf address rounds that produces degree-1 polynomials, while Jolt uses prefix-suffix decomposition producing degree-2 polynomials with different value semantics.

### Key Differences Documented
1. **Polynomial Degree**: Jolt=2, Zolt=1 during address rounds
2. **Value Computation**: Jolt uses table MLE evaluations, Zolt uses raw lookup results
3. **RAF Computation**: Jolt evaluates LeftOperand/RightOperand/Identity polynomials at r_address, Zolt uses concrete operand values

### Files Analyzed
- Jolt read_raf_checking.rs: Main LookupsReadRaf implementation
- Jolt lookup_table/mod.rs: 42 lookup table definitions
- Jolt prefixes/mod.rs: 45+ prefix types
- Jolt identity_poly.rs: Identity, OperandPolynomial implementations
- Jolt suffixes/mod.rs: Suffix definitions

### Required Implementation

To achieve Jolt compatibility, Zolt needs:

1. **Identity/Operand Polynomials** (simpler)
   - `IdentityPolynomial`: `Σ r[i] * 2^(n-1-i)` evaluation
   - `OperandPolynomial`: Left/Right operand evaluations
   - These are used for RAF computation

2. **Prefix-Suffix Decomposition** (complex)
   - 45+ prefix types with MLE evaluation and checkpoint updates
   - Suffix types for each lookup table
   - `combine()` function for each table

3. **Address Round Polynomial Computation**
   - Must match Jolt's `compute_prefix_suffix_prover_message()`
   - Returns evaluations at X∈{0, 2} for degree-2 interpolation

4. **Cycle Round Polynomial Computation**
   - After address rounds, materialize `combined_val_polynomial`
   - Use table MLE evaluations, not raw lookup results

### Incremental Approach

For Fibonacci program specifically:
- Uses ADD, ADDI, ADDW, ADDIW, BNE, JAL, JALR, LUI instructions
- All use "AddOperands" flag → identity RAF path (not interleaved)
- Could potentially implement minimal subset first

### Files to Create
- `src/zkvm/lookup_tables/mod.zig` - Table definitions
- `src/zkvm/lookup_tables/prefixes/` - All 45+ prefix implementations
- `src/zkvm/lookup_tables/suffixes/` - Suffix implementations
- `src/zkvm/spartan/prefix_suffix.zig` - PrefixSuffixDecomposition

### Estimated Effort
Several days of focused implementation work.

### Test Commands

```bash
# Build
zig build -Doptimize=ReleaseFast

# Generate proof
./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format -o /tmp/zolt_proof_dory.bin

# Verify with Jolt (currently fails at Stage 5)
cd /home/vivado/projects/jolt
cargo test --package jolt-core --features zolt-debug test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture
```

### Current Verification Status

```
Sumcheck verification failed!
  output_claim:   [d9, 50, 6a, 6e, 69, 84, 32, f8, ...]
  expected_claim: [bb, 2a, d3, 8c, 2c, 8c, 44, d3, ...]
```

Stages 1-4: PASS
Stage 5: FAIL (polynomial mismatch due to missing prefix-suffix decomposition)

### Session Notes

See `.agent/NOTES.md` for detailed technical analysis of the prefix-suffix decomposition requirements.
