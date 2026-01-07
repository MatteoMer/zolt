# Zolt-Jolt Cross-Verification Progress

## Session 16 Summary

### Achievements
1. **RAF Prover Integrated** - Instance 1 now produces proper cubic [s(0), s(1), s(2), s(3)] output
2. **Instance Timing Analysis** - Documented when each instance becomes active
3. **Memory Trace Propagation** - Added memory_trace to ConversionConfig

### Critical Discovery: Instance 2 Starts at Round 0!

The batched sumcheck instance timing is:
- Instance 0 (ProductVirtualRemainder): 10 rounds → starts at round 16
- Instance 1 (RamRafEvaluation): 16 rounds → starts at round 10 ✅
- **Instance 2 (RamReadWriteChecking): 26 rounds → starts at round 0!** ❌
- Instance 3 (OutputSumcheck): 16 rounds → starts at round 10 ✅
- Instance 4 (InstructionLookupsClaimReduction): 10 rounds → starts at round 16

**This means Instance 2 is the BLOCKER** - it has a non-zero input claim from the very first round, but our fallback produces incorrect polynomials.

## RamReadWriteChecking Architecture (from Jolt)

This is the most complex prover in Stage 2 - a 3-phase sparse matrix sumcheck:

### Phase 1: Cycle-Major (Rounds 0 to log_T-1)
- Entries sorted by (cycle, address)
- Uses Gruen split-eq optimization
- Binds cycle variables

### Phase 2: Address-Major (Rounds log_T to log_T+log_K-1)
- Re-sorts entries by (address, cycle)
- Binds address variables
- More efficient for address dimension

### Phase 3: Dense (Remaining rounds if any)
- Materializes to dense polynomials
- Standard sumcheck

### The Polynomial Being Proved
```
Σ_{k,j} eq(r_cycle, j) * ra(k,j) * (Val(k,j) + γ*(Val(k,j) + inc(j))) = rv_claim + γ*wv_claim
```

Where:
- k indexes addresses [0, K)
- j indexes cycles [0, T)
- ra(k,j) = 1 if address k accessed at cycle j
- Val(k,j) = memory value at address k before cycle j
- inc(j) = value increment at cycle j (write_val - read_val, or 0 for reads)

## Implementation Status

| Instance | Prover | Status | Notes |
|----------|--------|--------|-------|
| 0 | ProductVirtualRemainder | ✅ | Working |
| 1 | RafEvaluationProver | ✅ | Integrated this session |
| 2 | RamReadWriteChecking | ❌ | **BLOCKER** - complex 3-phase prover |
| 3 | OutputSumcheckProver | ✅ | Working |
| 4 | InstructionLookupsClaimReduction | ❌ | Needs implementation |

## Key Files Modified This Session

- `src/zkvm/proof_converter.zig` - RAF prover integration
- `src/zkvm/ram/raf_checking.zig` - Added `computeRoundPolynomialCubic()`
- `src/zkvm/mod.zig` - Pass memory_trace through config
- `src/zkvm/prover.zig` - Updated RAF prover API usage

## Test Commands
```bash
# Generate Zolt proof
./zig-out/bin/zolt prove --jolt-format -o /tmp/zolt_proof_dory.bin examples/fibonacci.elf

# Test with Jolt
cd /Users/matteo/projects/jolt/jolt-core
cargo test test_verify_zolt_proof -- --ignored --nocapture
```

## Next Session Priority

1. **Implement RamReadWriteCheckingProver** - This is the critical blocker
   - Need sparse matrix construction from memory trace
   - Phase transitions (cycle→address major)
   - Gruen's split-eq optimization
   - ~500 lines of code

2. **Implement InstructionLookupsClaimReductionProver** - Secondary
   - 2-phase prover
   - Simpler than RAM checking
   - ~200 lines of code

## Technical References

- Jolt RAM checking: `jolt-core/src/zkvm/ram/read_write_checking.rs`
- Gruen optimization: `jolt-core/src/poly/split_eq_poly.rs`
- Sparse matrix: `jolt-core/src/subprotocols/read_write_matrix/`
