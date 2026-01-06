# Stage 2 Implementation Plan

## Current Status: Stage 1 PASSES, Stage 2 placeholder zeros fail verification

**Goal:** Replace placeholder zeros with real sumcheck proofs for all 5 Stage 2 instances.

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

## Incremental Implementation Order

Implement and verify one sumcheck instance at a time:

### Phase 1: ProductVirtualRemainderProver
- [ ] Create `src/zkvm/spartan/product_remainder.zig`
- [ ] Implement degree-3 sumcheck over n_cycle_vars rounds
- [ ] Wire witnesses (Left/Right factors) from execution trace
- [ ] Update `proof_converter.zig` to generate real proofs
- [ ] Test with Jolt verifier → Debug → Pass

### Phase 2: RamRafEvaluationSumcheckProver
- [ ] Complete `src/zkvm/ram/raf_checking.zig` (already partial)
- [ ] Implement degree-2 sumcheck over log_ram_k rounds
- [ ] Wire ra and unmap polynomials from memory trace
- [ ] Test → Debug → Pass

### Phase 3: InstructionLookupsClaimReductionProver
- [ ] Create `src/zkvm/claim_reductions/instruction_lookups.zig`
- [ ] Implement degree-2 sumcheck over n_cycle_vars rounds
- [ ] Wire lookup_output, left_operand, right_operand from lookup trace
- [ ] Test → Debug → Pass

### Phase 4: OutputSumcheckProver
- [ ] Create `src/zkvm/ram/output_check.zig`
- [ ] Implement degree-3 sumcheck over log_ram_k rounds (zero-check)
- [ ] Wire val_init, val_final, io_mask from RAM state
- [ ] Test → Debug → Pass

### Phase 5: RamReadWriteCheckingProver
- [ ] Create `src/zkvm/ram/read_write_checking.zig`
- [ ] Implement 3-phase sparse sumcheck (most complex)
- [ ] Wire inc, val, ra polynomials
- [ ] Test → Debug → Pass → Stage 2 Complete!

---

## Prover Details

### 1. ProductVirtualRemainderProver
**Input:** UniSkip params (τ, r₀), 5 product witnesses, Left/Right factors
**Algorithm:**
```
input_claim = uni_skip_output_claim
For round j in [0..n_cycle_vars):
    h_j(z) = Σ_{b∈{0,1}^remaining} L(τ_high, r₀) × Eq(τ_low, [r..., z, b...]) × left(z,b) × right(z,b)
```
**Opening Claims:** LeftInstructionInput, RightInstructionInput, InstructionFlags(IsRdNotZero), OpFlags(WriteLookupOutputToRD), OpFlags(Jump), LookupOutput, InstructionFlags(Branch), NextIsNoop

### 2. RamRafEvaluationSumcheckProver
**Input:** r_cycle, memory trace, start_address
**Algorithm:**
```
ra(k) = Σⱼ eq(r_cycle, j) × [address(j) = k]
input_claim = Σₖ ra(k) × unmap(k)
For round j in [0..log_ram_k):
    h_j(z) = Σ_{b} eq(r_addr, [z,b]) × ra(z,b) × unmap(z,b)
```
**Opening Claims:** RamRa at [r_address || r_cycle]

### 3. RamReadWriteCheckingProver (3-phase)
**Input:** γ, r_cycle, memory trace (addresses, values, increments)
**Algorithm:**
```
input_claim = rv_claim + γ × wv_claim
Phase 1 (cycle-major, log_t rounds): sparse
Phase 2 (address-major, log_k rounds): sparse
Phase 3 (dense, remaining vars)
```
**Opening Claims:** RamVal, RamRa at [r_address || r_cycle], RamInc at r_cycle

### 4. OutputSumcheckProver
**Input:** program I/O, val_init, val_final, io_mask
**Algorithm:**
```
input_claim = 0 (zero-check)
For round j in [0..log_ram_k):
    h_j(z) = Σ_{b} eq(r_addr, [z,b]) × io_mask(z,b) × (val_final - val_io)
```
**Opening Claims:** RamValFinal, RamValInit at r_address

### 5. InstructionLookupsClaimReductionProver
**Input:** γ, lookup claims, r_spartan, lookup polynomials
**Algorithm:**
```
input_claim = output_claim + γ × left_claim + γ² × right_claim
For round j in [0..n_cycle_vars):
    h_j(z) = Σ_{b} eq(r_spartan, [z,b]) × (output + γ × left + γ² × right)
```
**Opening Claims:** LookupOutput, LeftLookupOperand, RightLookupOperand at r_cycle

---

## Files to Create/Modify

| File | Action |
|------|--------|
| `src/zkvm/batched_sumcheck.zig` | Create - infrastructure |
| `src/zkvm/spartan/product_remainder.zig` | Create |
| `src/zkvm/ram/raf_checking.zig` | Extend |
| `src/zkvm/ram/read_write_checking.zig` | Create |
| `src/zkvm/ram/output_check.zig` | Create |
| `src/zkvm/claim_reductions/instruction_lookups.zig` | Create |
| `src/zkvm/prover.zig` | Modify - integrate |
| `src/zkvm/proof_converter.zig` | Modify - serialization |

---

## Test Commands

```bash
# Generate Zolt proof
./zig-out/bin/zolt prove --jolt-format -o /tmp/zolt_proof_dory.bin examples/fibonacci.elf

# Run Jolt verification
cd /Users/matteo/projects/jolt
cargo test --package jolt-core test_verify_zolt_proof -- --ignored --nocapture
```
