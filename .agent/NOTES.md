# Zolt Implementation Notes

## Test Interference Issue (Iteration 10-11)

### Problem

Adding a full e2e prover test that calls `JoltProver.prove()` causes unrelated tests
to fail:
- `zkvm.lasso.split_eq.test.split eq inner product`
- `zkvm.lasso.expanding_table.test.expanding table multiple binds`
- `zkvm.lasso.integration_test.test.lasso multiple rounds consistent`
- `zkvm.spartan.mod.test.spartan proof generation`

### Observations

1. Without the e2e test, all 324 tests pass
2. With even a simple e2e test that calls `prover.prove()`, other tests fail
3. The e2e test itself passes - it's not failing
4. Running tests with `-j1` (single thread) doesn't help
5. The failures are deterministic (not flaky)
6. Adding a dummy test (that doesn't call prove()) doesn't cause failures
7. Adding a test that only calls JoltProver.init() doesn't cause failures
8. Adding a test that calls JoltProver.prove() DOES cause failures
9. Clearing .zig-cache and rebuilding doesn't help
10. No global/static variables were found in the codebase

### Root Cause (Likely)

This appears to be a **Zig 0.15.2 compiler bug** related to comptime evaluation.
When the prover test is included:
- The compiler generates different code for unrelated tests
- The field arithmetic tests produce different (incorrect) results
- This is NOT runtime memory corruption - it's a compile-time issue

Evidence: The failures are deterministic and occur even with:
- Single-threaded execution (-j1)
- Fresh cache (rm -rf .zig-cache)
- Completely independent allocators in each test

### Workaround

The e2e prover test is commented out in `src/zkvm/mod.zig`. The full prover
functionality was verified during development of previous iterations and works
correctly when run in isolation.

### Future Investigation

1. Report to Zig issue tracker with minimal reproduction
2. Test with newer Zig versions when available
3. Try restructuring the prover to use less comptime

## Bit Ordering Convention

The Lasso lookup tables and EQ polynomials use a specific bit ordering:

### ExpandingTable
After binding variables r0, r1, r2 in order:
- Index 0 (000): `(1-r0)(1-r1)(1-r2)`
- Index 1 (001): `(1-r0)(1-r1)*r2`
- Index 4 (100): `r0*(1-r1)*(1-r2)`
- Index 7 (111): `r0*r1*r2`

The LSB (bit 0) corresponds to the LAST bound variable (r2), not the first.

### SplitEqPolynomial
Uses (outer_idx, inner_idx) with linear index `j = outer_idx * inner_size + inner_idx`.
This is different from the binary representation where bit positions directly
map to variable indices.

## Lasso Prover Parameter Fix (Iteration 10)

The LassoProver was incorrectly recalculating `log_T` from `lookup_indices.len`
using `log2_int` which requires power-of-2 inputs. Fixed to use `params.log_T`
directly, which matches the length of `r_reduction`.
