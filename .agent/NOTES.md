# Zolt-Jolt Cross-Verification Progress

## Session 85 - Stage 5 Analysis Complete (2026-01-30)

### Key Finding
Stage 5 requires implementing 3 instances:
1. **RegistersValEvaluation** (8 rounds) - IMPLEMENTED ✅
2. **RamRaClaimReduction** (24 rounds) - Zero OK (no RAM in Fibonacci)
3. **LookupsReadRaf** (136 rounds) - NOT IMPLEMENTED ❌

### Root Cause of Failure
```
batched_claim - p(0)+p(1) = batch2 * lookups_input  (exactly!)
```

Instance 2 (LookupsReadRaf) has num_rounds = max_rounds = 136, so:
- It never has a "constant phase" where we can scale input_claim
- Every round is "active" and requires actual polynomial computation
- We send zero polynomials, but lookups_input is non-zero

### Verification of RegistersValEvaluation
1. ✅ Sum check: computed_sum = regs_val_input (match = true)
2. ✅ LT polynomial: lt[0] matches LT_verifier
3. ✅ Final product: inc*wa*lt matches expected_product
4. ❌ Overall Stage 5 fails due to Instance 2

### What LookupsReadRaf Needs
From read_raf_checking.rs:
```rust
rv(r_reduction) + γ·left_op(r_reduction) + γ²·right_op(r_reduction)
  = Σ_j Σ_k [ eq(j; r_reduction) · ra(k, j) · (Val_j(k) + γ · RafVal_j(k)) ]
```

Requires:
- Prefix/suffix decomposition for 128-bit address space
- Lookup table MLE evaluations
- RAF polynomial handling
- Complex batching with γ

### Commits
- 6b7de26: Stage 5 - Fix RegistersValEvaluation, document LookupsReadRaf missing

---

## Session 84 Summary
- Identified Stage 5 failure: output_claim ≠ expected_output_claim
- Fixed constant polynomial scaling (use scaled_input_claim, not half_claim)
- Verified RegistersValEvaluation implementation is correct

## Key Files
- `src/zkvm/spartan/stage5_prover.zig` - Stage 5 batched sumcheck prover
- `jolt-core/src/zkvm/registers/val_evaluation.rs` - Jolt's RegistersValEvaluation
- `jolt-core/src/zkvm/instruction_lookups/read_raf_checking.rs` - LookupsReadRaf reference
- `jolt-core/src/subprotocols/sumcheck.rs` - Batched sumcheck verifier
