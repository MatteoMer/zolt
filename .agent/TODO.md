# Zolt zkVM Implementation TODO

## Completed ✅ (This Session - Iteration 4)

### Lookup Trace Integration
- [x] LookupEntry for storing lookup operation data
- [x] LookupTraceCollector for recording lookups during execution
- [x] Integration with Emulator.step() - automatic lookup recording
- [x] Statistics collection for lookup analysis
- [x] Integration test verifying lookup trace collection

### Memory RAF Checking (`src/zkvm/ram/raf_checking.zig`)
- [x] RafEvaluationParams: Sumcheck parameters
- [x] RaPolynomial: ra(k) = Σ_j eq(r_cycle, j) · 1[address(j) = k]
- [x] UnmapPolynomial: Converts remapped address to original
- [x] RafEvaluationProver: Prover with round polynomial computation
- [x] RafEvaluationVerifier: Verifier with challenge generation
- [x] Helper functions: computeEqEvals, computeEqAtPoint

## Completed ✅ (Previous Sessions)

### Phase 1: Lookup Arguments
- [x] Lookup table infrastructure (14 tables)
- [x] LookupBits utility for bit manipulation
- [x] Fixed interleave/uninterleave to match Jolt conventions
- [x] Fixed MLE tests with MSB-first ordering

### Lasso Infrastructure (`src/zkvm/lasso/`)
- [x] ExpandingTable for EQ polynomial accumulation
- [x] SplitEqPolynomial (Gruen's optimization)
- [x] PrefixSuffixDecomposition with suffix/prefix types
- [x] PrefixPolynomial, SuffixPolynomial, PrefixRegistry
- [x] LassoProver with two-phase sumcheck
- [x] LassoVerifier with round verification
- [x] Integration tests for Lasso protocol

### Instruction Flags and Lookups
- [x] CircuitFlags enum (13 flags)
- [x] InstructionFlags enum (7 flags)
- [x] CircuitFlagSet and InstructionFlagSet types
- [x] LookupTables(XLEN) enum with materializeEntry()
- [x] InstructionLookup, Flags, LookupQuery interfaces
- [x] AddLookup, SubLookup, AndLookup, OrLookup, XorLookup
- [x] SltLookup, SltuLookup for comparisons
- [x] BeqLookup, BneLookup for branches

## Next Up (Future Iterations)
- [ ] Multi-stage sumcheck orchestration (7 stages)
- [ ] Val evaluation sumcheck (memory value checking)
- [ ] Read-write checking sumcheck
- [ ] Fix HyperKZG verification (current stub returns true)
- [ ] Add real BN254 curve constants
- [ ] Implement execute(), prove(), verify()

## Files Added This Session

### Lookup Trace Integration
- `src/zkvm/instruction/lookup_trace.zig` - LookupEntry, LookupTraceCollector

### RAF Checking
- `src/zkvm/ram/raf_checking.zig` - Full RAF sumcheck infrastructure

## Summary
This iteration focused on:
1. **Lookup Trace Integration**: Connect execution tracer to Lasso infrastructure
2. **RAF Memory Checking**: Implement RAF evaluation sumcheck prover/verifier

All 261 tests pass.
