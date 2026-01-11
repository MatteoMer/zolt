# Zolt-Jolt Compatibility TODO

## Current Progress

### ✅ Stage 1 - PASSES
### ✅ Stage 2 - PASSES
### ✅ Stage 3 - PASSES

### 🔄 Stage 4 - DEBUGGING IN PROGRESS

**✅ FIXED Issues:**
1. Challenge ordering: Stage 3 challenges passed as-is (no reversal needed)
2. RdWriteValue: Set to 0 for non-writing instructions (STORE, BRANCH, rd=0)
3. Rs1Value/Rs2Value: Set to 0 for instructions that don't read rs1/rs2 (LUI, JAL, AUIPC)
4. Input claim mismatch: Fixed by ensuring R1CS witness matches Jolt's behavior
5. K value: Confirmed K=128 (32 RISC-V + 96 virtual registers), LOG_K=7

**✅ VERIFIED CORRECT:**
- Simple MLE computation matches Stage 3 claims ✓
- Computed input claim == Expected input claim from Stage 3 ✓
- Round 0 sumcheck passes ✓

**🔴 CURRENT BUG: Product Polynomial Binding**

Round 0 passes, but Round 1+ fails. Root cause identified:

The Stage 4 polynomial is a PRODUCT of simpler polynomials:
```
P(k,j) = eq(r,j) * [rd_wa(k,j)*(inc(j)+val(k,j)) + γ*rs1_ra(k,j)*val(k,j) + ...]
```

**The Problem:**
After binding variable 0 with challenge r, I was binding each component separately then multiplying:
```
rd_wa' * val' ≠ (rd_wa * val)'  // These are NOT equal!
```

**Jolt's Solution (from Rust code analysis):**
Jolt uses "sumcheck evaluations" - for each index, it:
1. Computes univariate restriction evals for each component: `rd_wa_evals[0,1,2]`, `val_evals[0,1,2]`, etc.
2. Multiplies pointwise: `result[i] = rd_wa_evals[i] * val_evals[i]`
3. Does NOT use standard binding formula on the product

**Next Steps:**
1. Rewrite `computeRoundPolynomial` to compute sumcheck_evals for each component separately
2. Multiply the evals pointwise to get correct round polynomial
3. Remove the incorrect binding-then-multiply approach

## Key Files
- `src/zkvm/spartan/stage4_prover.zig` - Stage 4 prover (needs sumcheck_evals rewrite)
- `src/zkvm/r1cs/constraints.zig` - R1CS witness (Rs1Value/Rs2Value fix applied)

## Reference: Jolt Implementation
- `jolt-core/src/zkvm/registers/read_write_checking.rs` - Phase 3 sumcheck logic
- `jolt-core/src/poly/multilinear_polynomial.rs` - sumcheck_evals function
- `jolt-core/src/subprotocols/read_write_matrix/registers.rs` - Sparse matrix for Phases 1/2

## Testing
```bash
bash scripts/build_verify.sh
```
