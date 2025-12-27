# Zolt zkVM Implementation TODO

## Completed ✅ (This Session)
- [x] Lookup table infrastructure (14 tables)
- [x] LookupBits utility for bit manipulation
- [x] Fixed interleave/uninterleave to match Jolt conventions
- [x] Fixed MLE tests with MSB-first ordering
- [x] CircuitFlags enum (13 flags)
- [x] InstructionFlags enum (7 flags)
- [x] CircuitFlagSet and InstructionFlagSet types
- [x] LookupTables(XLEN) enum with materializeEntry()
- [x] InstructionLookup, Flags, LookupQuery interfaces
- [x] ExpandingTable for EQ polynomial accumulation
- [x] SplitEqPolynomial (Gruen's optimization)
- [x] PrefixSuffixDecomposition infrastructure
- [x] PrefixPolynomial, SuffixPolynomial, PrefixRegistry
- [x] LassoProver with two-phase sumcheck
- [x] LassoVerifier with round verification

## Next Up
- [ ] Create SumcheckAdapter to integrate Lasso with existing sumcheck
- [ ] Create an end-to-end test that runs prover + verifier together
- [ ] Implement lookup queries for ADD, SUB, AND instructions
- [ ] Connect instruction lookups to table evaluation
- [ ] Generate R1CS constraints per instruction

## Later (Phase 3+)
- [ ] Memory RAF checking
- [ ] Multi-stage sumcheck orchestration (7 stages)
- [ ] Fix HyperKZG verification
- [ ] Add real BN254 curve constants
- [ ] Implement execute(), prove(), verify()

## Session Summary (Iteration 3)
This iteration focused on implementing the core Lasso infrastructure:

1. **ExpandingTable** (`lasso/expanding_table.zig`)
   - Incrementally builds EQ polynomial evaluations
   - Doubles in size each round (O(2^i) for round i)
   - Used during address binding phase

2. **SplitEqPolynomial** (`lasso/split_eq.zig`)
   - Gruen's optimization for EQ evaluation
   - Factors eq(w,x) into outer and inner components
   - Caches prefix tables for efficient inner products

3. **PrefixSuffixDecomposition** (`lasso/prefix_suffix.zig`)
   - SuffixType enum with field evaluations (And, Or, Xor, etc.)
   - PrefixType enum (LowerWord, UpperWord, Eq, etc.)
   - PrefixPolynomial with binding and caching
   - PrefixRegistry for sharing prefixes

4. **LassoProver** (`lasso/prover.zig`)
   - LassoParams: gamma batching, log_T/log_K, reduction point
   - Two-phase protocol:
     - Address binding (LOG_K rounds)
     - Cycle binding (log_T rounds)
   - Computes round polynomials using EQ accumulation

5. **LassoVerifier** (`lasso/verifier.zig`)
   - Verifies g(0) + g(1) = claim for each round
   - Derives Fiat-Shamir challenges
   - Checks final evaluation

All 256 tests pass.
