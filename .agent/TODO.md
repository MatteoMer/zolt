# Zolt-Jolt Compatibility Implementation

## Status: IN PROGRESS - Stage 5 RAF/Operand Evaluation Mismatch

## Session 112 Summary

### Progress Made

1. **Fixed OperandPolynomial binding parity bug in `operandPrefixEvals`:**
   - Line 953 was inverted: `if (is_left) !is_even_round else is_even_round`
   - Should be: `if (is_left) is_even_round else !is_even_round`
   - This matches Jolt's OperandPolynomial::bind() logic:
     - LeftOperand binds when num_bound_vars is EVEN
     - RightOperand binds when num_bound_vars is ODD

2. **Verified understanding of Jolt's operand evaluation formulas:**
   - `identity_poly_eval = Σ(i=0 to 127) r_address[i] * 2^(127-i)`
   - `left_operand_eval = Σ(i=0 to 63) r_address[2i] * 2^(63-i)` (even indices)
   - `right_operand_eval = Σ(i=0 to 63) r_address[2i+1] * 2^(63-i)` (odd indices)

### Current Issue

**Stage 5 sumcheck verification still fails after the binding parity fix.**

The polynomial coefficients Zolt produces are different from what Jolt expects:
- Jolt reads: `[e2, ee, 6f, c7, ...]` for c0 in round 0
- Zolt produces: `[25, f7, 18, 0c, ...]` (different values)

This causes different challenges to be derived, leading to different operand evaluations.

The fix I made only affects how bound_value accumulates during cycle rounds (128-135), not the polynomial coefficients during address rounds (0-127).

### Root Cause Analysis

The Stage 5 address rounds use prefix-suffix decomposition. The computed polynomials depend on:
1. **Suffix polynomials (Q)** - accumulated from `u_evals[j] * suffix_mle(suffix_bits[j])`
2. **Prefix polynomials** - depend on checkpoint values and current round

The mismatch suggests either:
1. The suffix polynomial initialization is wrong
2. The prefix MLE evaluation is wrong
3. The table combine functions are wrong
4. The Q binding is wrong

### Next Steps

1. **Debug suffix polynomial initialization:**
   - Verify `AllSuffixPolys.initPhase()` correctly computes Q[b]
   - Check that cycle_table_indices maps cycles to correct tables
   - Verify suffix_mle returns correct values

2. **Debug prefix MLE evaluation:**
   - Verify each prefix type's prefixMle function matches Jolt's
   - Check checkpoint initialization and updates

3. **Add detailed debug output:**
   - Print Q values after initialization
   - Print prefix evaluations for round 0
   - Compare with expected values from Jolt

4. **Consider simplification:**
   - Test with a simpler program that uses only one table type
   - Verify the basic prefix-suffix machinery works for that case

### Key Components

1. **Prefix-Suffix Decomposition** (`src/zkvm/lookup_table/`)
   - All 46 prefix types implemented in `prefixes.zig`
   - All 43 suffix types implemented in `suffixes.zig`
   - Table-specific combine functions in `prefix_suffix_prover.zig`
   - RAF decomposition for left/right/identity operands

2. **Stage 5 Prover** (`src/zkvm/spartan/stage5_prover.zig`)
   - Three-instance batched sumcheck
   - Instance 0: RegistersValEvaluation (8 rounds)
   - Instance 1: RamRaClaimReduction (24 rounds) - simplified for fibonacci
   - Instance 2: LookupsReadRaf (136 rounds) with prefix-suffix decomposition
   - Phase transitions every 8 rounds (16 phases total)

### Test Commands
```bash
# Build
zig build -Doptimize=ReleaseFast

# Generate proof
timeout 600 ./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format -o logs/zolt_proof_dory.bin

# Verify with Jolt
cd /home/vivado/projects/jolt
cargo test --package jolt-core --features zolt-debug test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture
```
