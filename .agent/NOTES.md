# Zolt-Jolt Compatibility Notes

## Current Status (Session 65 - January 7, 2026)

### Summary

**Stage 1 PASSES, Stage 2 UniSkip PASSES, Stage 2 Sumcheck FAILS (zeros need real implementation)**

Key finding from Jolt analysis: **Cannot cheat with zeros!**
- input_claim for ProductVirtualRemainder is non-zero (from Stage 1)
- If round polynomials are zeros, output_claim = input_claim ≠ 0
- If opening claims are zeros, expected_output_claim = 0
- Verification fails: output_claim ≠ expected_output_claim

### Completed Work

1. **ProductVirtualRemainderProver skeleton** (`src/zkvm/spartan/product_remainder.zig`)
   - Fuses 5 product constraints into left/right polynomials using Lagrange weights
   - Computes cubic round polynomials [s(0), s(2), s(3)]
   - Extracts 8 unique factor polynomial values per cycle

2. **BatchedSumcheckProver infrastructure** (`src/zkvm/batched_sumcheck.zig`)
   - Combines multiple sumcheck instances with different round counts
   - Handles batching protocol: scale claims, sample coefficients, combine polynomials

3. **All 712+ tests passing**

### Stage 2 Architecture

Stage 2 batches 5 sumcheck instances:
1. **ProductVirtualRemainder**: n_cycle_vars rounds, degree 3
   - Input claim = uni_skip_claim (from Stage 2 UniSkip)
   - Must implement: fused left/right from 5 product constraints

2. **RamRafEvaluation**: log_ram_k rounds, degree 2
   - Input claim = raf_claim from RamAddress @ SpartanOuter
   - Can be zero for programs without RAM

3. **RamReadWriteChecking**: log_ram_k + n_cycle_vars rounds (MAX), degree 3
   - Input claim = rv_claim + γ * wv_claim from RAM read/write values
   - Can be zero for programs without RAM

4. **OutputSumcheck**: log_ram_k rounds, degree 3
   - Input claim = 0 (ALWAYS - this is a zero-check)

5. **InstructionLookupsClaimReduction**: n_cycle_vars rounds, degree 2
   - Input claim = lookup_output + γ * left + γ² * right
   - Can be zero for trivial programs

### Verification Flow

After batched sumcheck completes:
1. `output_claim` = final polynomial evaluation after all rounds
2. `expected_output_claim` = Σᵢ αᵢ * instance[i].expected_output_claim(r_sumcheck)
3. Check: `output_claim == expected_output_claim`

Each instance's `expected_output_claim` fetches opening claims from accumulator and evaluates the polynomial relation.

### Next Steps

1. Wire ProductVirtualRemainder into proof_converter.zig
2. Implement proper transcript operations for Stage 2 batched sumcheck
3. Compute real opening claims from witnesses
4. For other 4 instances, need either:
   - Real implementations (ideal)
   - Zero contributions if their Stage 1 inputs are zero (valid for simple programs)

---

## Previous Sessions

See earlier notes for Stage 1 fixes including:
- Montgomery form serialization
- Batching coefficient handling
- UniSkip polynomial construction
- Opening claims witness matching
