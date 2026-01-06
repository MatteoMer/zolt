# Stage 2 Implementation Plan

## Current Status: Stage 1 PASSES, Stage 2 placeholder zeros fail verification

**Goal:** Replace placeholder zeros with real sumcheck proofs for all 5 Stage 2 instances.

---

## Progress

### Completed
- [x] Stage 1 streaming outer prover (PASSES)
- [x] Stage 1 UniSkip polynomial
- [x] Stage 2 UniSkip polynomial (PASSES)
- [x] ProductVirtualRemainderProver skeleton (src/zkvm/spartan/product_remainder.zig)
- [x] BatchedSumcheckProver infrastructure (src/zkvm/batched_sumcheck.zig)
- [x] All 712+ tests pass

### In Progress
- [ ] Wire ProductVirtualRemainderProver into proof_converter.zig
- [ ] Test with real sumcheck proofs

### Pending
- [ ] Implement RamRafEvaluationSumcheckProver
- [ ] Implement RamReadWriteCheckingProver
- [ ] Implement OutputSumcheckProver
- [ ] Implement InstructionLookupsClaimReductionProver

---

## Stage 2 Architecture

```
Stage 2 = BatchedSumcheck([
    ProductVirtualRemainderProver,           // n_cycle_vars rounds, degree 3
    RamRafEvaluationSumcheckProver,          // log_ram_k rounds, degree 2
    RamReadWriteCheckingProver,              // log_ram_k + n_cycle_vars rounds, degree 3
    OutputSumcheckProver,                    // log_ram_k rounds, degree 3
    InstructionLookupsClaimReductionProver,  // n_cycle_vars rounds, degree 2
])
```

**Batching Protocol:**
1. Append all 5 input_claims to transcript
2. Sample 5 batching coefficients: α₀, α₁, α₂, α₃, α₄
3. Scale claims: `scaled_claim[i] = claim[i] * 2^(max_rounds - rounds[i])`
4. Compute batched_claim = Σᵢ αᵢ * scaled_claim[i]
5. Run `max_rounds` sumcheck rounds (max = log_ram_k + n_cycle_vars)
6. Each round: combine univariate polynomials h(z) = Σᵢ αᵢ * hᵢ(z)

---

## Files Created

| File | Status |
|------|--------|
| `src/zkvm/spartan/product_remainder.zig` | Created |
| `src/zkvm/batched_sumcheck.zig` | Created |
| `src/zkvm/spartan/mod.zig` | Modified - added exports |
| `src/zkvm/mod.zig` | Modified - added batched_sumcheck |

---

## Test Commands

```bash
# Run all Zolt tests
zig build test --summary all

# Expected: 712+ tests pass
```

---

## Key Insights

1. **Stage 2 has 5 instances with different round counts:**
   - ProductVirtualRemainder: n_cycle_vars rounds
   - RamRafEvaluation: log_ram_k rounds
   - RamReadWriteChecking: log_ram_k + n_cycle_vars rounds (MAX)
   - OutputSumcheck: log_ram_k rounds
   - InstructionLookupsClaimReduction: n_cycle_vars rounds

2. **The batching handles different rounds by:**
   - Scaling claims by 2^(max - rounds[i])
   - Returning constant polynomials before instance starts

3. **Opening claims needed:**
   - 8x PRODUCT_UNIQUE_FACTOR_VIRTUALS @ SpartanProductVirtualization
   - RamRa @ RamRafEvaluation, RamReadWriteChecking
   - RamVal, RamInc @ RamReadWriteChecking
   - RamValFinal, RamValInit @ RamOutputCheck
   - LookupOutput, LeftLookupOperand, RightLookupOperand @ InstructionClaimReduction
