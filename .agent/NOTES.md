# Zolt-Jolt Cross-Verification Progress

## Session 75 Summary - Challenge Type Analysis (2026-01-29)

### Challenge Type Mapping Verified

| Jolt Function | Returns | Zolt Equivalent | Use Case |
|---------------|---------|-----------------|----------|
| `challenge_scalar::<F>()` | Fr (Montgomery) | `challengeScalarFull()` | Batching coeffs, gamma values |
| `challenge_scalar_optimized::<F>()` | MontU128Challenge (125-bit, `[0,0,L,H]`) | `challengeScalar()` | tau_high, r0, sumcheck r_i |
| `challenge_vector(n)` | Vec<Fr> | n × `challengeScalarFull()` | Batching coeffs |
| `challenge_vector_optimized(n)` | Vec<MontU128Challenge> | n × `challengeScalar()` | r_address |
| `challenge_scalar_powers(n)` | Vec<Fr> (1, q, q², ...) | `challengeScalarPowers()` | Gamma powers |

### Stage 2 Challenge Sampling Order (Verified Correct)

1. `ProductVirtualUniSkipParams::new` → `challenge_scalar_optimized` → `tau_high_stage2`
2. UniSkip proof: append poly → `challenge_scalar_optimized` → `r0_stage2`
3. UniSkip `cache_openings`: `append_virtual(uni_skip_claim)`
4. `RamReadWriteCheckingParams::new` → `challenge_scalar` → `gamma_rwc`
5. `OutputSumcheckParams::new` → `challenge_vector_optimized(log_k)` → `r_address`
6. `InstructionLookupsClaimReductionSumcheckParams::new` → `challenge_scalar` → `gamma_instr`
7. `BatchedSumcheck::verify` → append input_claims → `challenge_vector(5)` → batching_coeffs

**Zolt uses matching functions for all of these ✓**

### Remaining Hypothesis: Opening Claims Storage

The verifier retrieves factor evaluations from opening_claims map using:
- `(VirtualPolynomial::X, SumcheckId::SpartanProductVirtualization)`

If Zolt stores these at incorrect keys or with wrong values, the expected formula will compute wrong.

Factor polynomials checked (need values verification):
- InstructionOutput @ SpartanProductVirtualization
- IsRdNotZero @ SpartanProductVirtualization
- WriteLookupOutputToRD @ SpartanProductVirtualization
- Jump @ SpartanProductVirtualization
- Branch @ SpartanProductVirtualization
- NextIsNoop @ SpartanProductVirtualization

---

## Session 74 Summary - Stage 2 Deep Dive (2026-01-29)

### Key Finding: Zolt Prover is INTERNALLY CONSISTENT

**Evidence:**
- `STAGE2_FINAL: output_claim` = `{ 181, 30, 249, 122, ... }` (LE bytes)
- `expected_batched (from provers)` = `{ 35, 43, 4, 85, ... }` (BE bytes)
- Converting LE→BE: These ARE the same value ✓

**Implication:** The prover computes correct round polynomials that evaluate to what it expects. The issue is that Jolt's verifier computes a DIFFERENT expected value.

### Stage 2 Architecture Analysis

| Instance | Verifier | Rounds | Start | input_claim |
|----------|----------|--------|-------|-------------|
| 0 | ProductVirtualRemainder | 8 | 16 | uni_skip_claim |
| 1 | RamRafEvaluation | 16 | 8 | RamAddress@SpartanOuter |
| 2 | RamReadWriteChecking | 24 | 0 | RamReadValue + γ*RamWriteValue |
| 3 | OutputSumcheck | 16 | 8 | 0 |
| 4 | InstructionLookupsClaimReduction | 8 | 16 | LookupOutput + γ*Left + γ²*Right |

### Instance 0 (ProductVirtualRemainder) Expected Formula

```
expected = tau_high_bound_r0 * eq(tau_low, r_tail_reversed) * fused_left * fused_right
```

Where:
- `fused_left = w[0]*l_inst + w[1]*is_rd_not_zero + w[2]*is_rd_not_zero + w[3]*lookup_out + w[4]*j_flag`
- `fused_right = w[0]*r_inst + w[1]*wl_flag + w[2]*j_flag + w[3]*branch_flag + w[4]*(1-next_is_noop)`
- `w[i]` = Lagrange weights at r0 over domain [-2,-1,0,1,2]

### Verified Correct

1. **OpeningId indices match** - SumcheckId enum has 22 values matching Jolt
2. **Factor claim indices match**:
   - InstructionFlags::IsRdNotZero = 6 ✓
   - InstructionFlags::Branch = 4 ✓
   - OpFlags::Jump = 5 ✓
   - OpFlags::WriteLookupOutputToRD = 6 ✓
3. **Claims stored at correct sumcheck_ids** - SpartanProductVirtualization for product factors

### Suspected Root Cause: Transcript Divergence

The transcript state before tau_high sampling determines what tau_high will be.
- Zolt: `state before tau_high = { 37, 204, 55, 100, 179, 84, 234, 62 }`
- Jolt: Unknown (needs verification)

If these differ, tau_high differs, and all subsequent Stage 2 computations will be wrong.

Transcript depends on:
1. Initial preamble (polynomial commitments, bytecode hash)
2. Stage 1 univariate skip
3. Stage 1 sumcheck rounds
4. Stage 1 cache_openings (36 R1CS input claims)

### Blocking Issue

Cannot run Jolt tests - missing system dependencies:
```
pkg-config: command not found
openssl-sys build failed
```

### Instance Current Claims from Zolt

All instances match internally:
- inst0 prover.current_claim = `{ 27, 32, 238, 77, ... }` (ProductVirtual)
- inst1 prover.current_claim = `{ 16, 16, 53, 226, ... }` (RamRaf)
- inst2 prover.current_claim = `{ 26, 54, 142, 194, ... }` (RamRWC)
- inst3 prover.current_claim = `{ 45, 42, 247, 142, ... }` (Output)
- inst4 prover.current_claim = `{ 43, 14, 190, 152, ... }` (InstrLookups)

---

## Session 73 Summary - Deserialization Complete! (2026-01-29)

### Critical Fix: SumcheckId Mismatch

**Root Cause:** Zolt had 24 SumcheckId values, Jolt has 22.

The extra values were:
- `AdviceClaimReductionCyclePhase = 20`
- `AdviceClaimReduction = 21`

**Fix:** Removed extra values, renumbered:
- `IncClaimReduction = 20`
- `HammingWeightClaimReduction = 21`
- `COUNT = 22`

### Proof Serialization Fixes

1. **Missing advice proofs**: Only had 1 (commitment), needed all 5
2. **Configuration format**: Was writing mix of u8/usize, now 5 usizes

### Deserialization Result: COMPLETE SUCCESS

All 40544 bytes parse correctly.

---

## Previous Sessions

### Session 72 (2026-01-28)
- 714/714 unit tests passing
- Stage 3 sumcheck mathematically correct
- Opening claims storage verified

### Session 71 (2026-01-28)
- Instance 0 (RegistersRWC) verified correct
- Synthetic termination write discovery

### Session 70 (2026-01-28)
- Stage 4 final claim mismatch found
- Phase 2/3 from_evals_and_hint pattern applied

### Session 69 (2026-01-28)
- Internal sumcheck consistency failure
- Fixed Phase 2/3 polynomial computation

### Session 68 (2026-01-28)
- Removed termination bit workaround from RWC prover
