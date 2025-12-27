# Zolt zkVM Implementation TODO

## Completed ✅ (This Session - Iteration 3)

### Lasso Infrastructure (`src/zkvm/lasso/`)
- [x] ExpandingTable for EQ polynomial accumulation
- [x] SplitEqPolynomial (Gruen's optimization)
- [x] PrefixSuffixDecomposition with suffix/prefix types
- [x] PrefixPolynomial, SuffixPolynomial, PrefixRegistry
- [x] LassoProver with two-phase sumcheck (address + cycle binding)
- [x] LassoVerifier with round verification
- [x] Integration tests for Lasso protocol

### Instruction Lookups (`src/zkvm/instruction/lookups.zig`)
- [x] AddLookup, SubLookup for arithmetic
- [x] AndLookup, OrLookup, XorLookup for bitwise operations
- [x] SltLookup, SltuLookup for comparisons (signed/unsigned)
- [x] BeqLookup, BneLookup for branch equality
- [x] LookupTraceEntry for trace recording

## Completed ✅ (Previous Sessions)
- [x] Lookup table infrastructure (14 tables)
- [x] LookupBits utility for bit manipulation
- [x] Fixed interleave/uninterleave to match Jolt conventions
- [x] Fixed MLE tests with MSB-first ordering
- [x] CircuitFlags enum (13 flags)
- [x] InstructionFlags enum (7 flags)
- [x] CircuitFlagSet and InstructionFlagSet types
- [x] LookupTables(XLEN) enum with materializeEntry()
- [x] InstructionLookup, Flags, LookupQuery interfaces

## Next Up (Future Iterations)
- [ ] Connect instruction lookups to execution tracer
- [ ] Memory RAF (Read-After-Final) checking
- [ ] Multi-stage sumcheck orchestration (7 stages)
- [ ] Fix HyperKZG verification
- [ ] Add real BN254 curve constants
- [ ] Implement execute(), prove(), verify()

## Files Added This Session

### Lasso Module
- `src/zkvm/lasso/mod.zig` - Module exports
- `src/zkvm/lasso/expanding_table.zig` - ExpandingTable for EQ
- `src/zkvm/lasso/split_eq.zig` - Gruen's SplitEqPolynomial
- `src/zkvm/lasso/prefix_suffix.zig` - Prefix/suffix decomposition
- `src/zkvm/lasso/prover.zig` - LassoProver implementation
- `src/zkvm/lasso/verifier.zig` - LassoVerifier implementation
- `src/zkvm/lasso/integration_test.zig` - End-to-end tests

### Instruction Lookups
- `src/zkvm/instruction/lookups.zig` - 9 instruction implementations

## Summary
This iteration focused on:
1. **Lasso Data Structures**: ExpandingTable, SplitEqPolynomial, PrefixSuffixDecomposition
2. **Lasso Protocol**: LassoProver (two-phase sumcheck) and LassoVerifier
3. **Instruction Lookups**: ADD, SUB, AND, OR, XOR, SLT, SLTU, BEQ, BNE

All 256 tests pass.
