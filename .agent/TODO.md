# Zolt-Jolt Compatibility: Current Status

## Status: Stage 2 Verification Failure 🔴

## Session 76 Summary (2026-01-29)

### Major Progress: Proof Deserialization Fixed!

**Working:**
- ✅ Proof deserialization (all 7 stages, 91 opening claims, 37 commitments)
- ✅ Stage 1 (OuterRemainingSumcheck) verification passes
- ✅ Preprocessing loading from both compressed and uncompressed formats

**Failing:**
- ❌ Stage 2 sumcheck verification - expected_output_claim mismatch

### Stage 2 Error Details

```
output_claim:          15906954023365202249122192714132265766544458757312739318826275235085359324853
expected_output_claim: 11386433087960536582639845443917888291615956842149860534020066572649924103188
```

Stage 2 is a batched sumcheck with 5 instances:
1. ProductVirtualRemainderVerifier (n_cycle_vars rounds)
2. RamRafEvaluationSumcheckVerifier (log_ram_k rounds)
3. RamReadWriteCheckingVerifier (log_ram_k + n_cycle_vars rounds - max!)
4. OutputSumcheckVerifier (log_ram_k rounds)
5. InstructionLookupsClaimReductionSumcheckVerifier (n_cycle_vars rounds)

Instance expected_claims from Jolt verifier:
- Instance 0: 13162261949552616826381676439296451788601018621847815047898609024679378399536
- Instance 1: 0 (ProductVirtualRemainder - zeros for simple program)
- Instance 2: 18644577964730782764190402023295937512886863831479486117757364935258160752617
- Instance 3: 20429994345422184441049916064819417529654187996943582502607189827673613930347
- Instance 4: 19475503802839692994087720930790055641780649560464270942640555911588860270906

### Root Cause Analysis

The expected_output_claim is computed from:
1. Opening claims stored in proof (factor evaluations at r_cycle point)
2. Batching coefficients derived from transcript
3. Weighted sum of all instance claims

The mismatch indicates Zolt's Stage 2 sumcheck proof has:
- Incorrect round polynomials, OR
- Incorrect opening claims for the verifier's computation, OR
- Transcript divergence causing different batching coefficients

### Next Steps (Priority Order)

1. [ ] Compare Zolt's Stage 2 round polynomials with what Jolt computes
2. [ ] Debug the factor polynomial evaluations for each instance
3. [ ] Verify the r_cycle point used for MLE evaluations
4. [ ] Check if Stage 2's opening claims match Jolt's expected format

### Technical Details
- trace_length: 256 (padded from 54 actual cycles)
- n_cycle_vars: 8
- log_ram_k: 16
- Stage 2: 24 rounds

### Commits
- `db0e57e3` - feat: add uncompressed deserialization support to Serializable trait

---

## Session 75 Summary (2026-01-29)

### Key Finding: Challenge Type Mapping Verified Correct

| Jolt Function | Zolt Function | Masking | Montgomery |
|---------------|---------------|---------|------------|
| `challenge_scalar_optimized` | `challengeScalar()` | 125-bit | No (raw `[0,0,L,H]`) |
| `challenge_scalar` | `challengeScalarFull()` | None | Yes (proper Fr) |
| `challenge_vector_optimized` | n × `challengeScalar()` | 125-bit | No |
| `challenge_vector` | n × `challengeScalarFull()` | None | Yes |

### Stage 2 Challenge Sampling Order (Verified Matches Jolt)

1. `ProductVirtualUniSkipParams::new` → `challenge_scalar_optimized` → `tau_high_stage2`
2. UniSkip proof: append poly → `challenge_scalar_optimized` → `r0_stage2`
3. UniSkip `cache_openings`: `append_virtual(uni_skip_claim)`
4. `RamReadWriteCheckingParams::new` → `challenge_scalar` → `gamma_rwc`
5. `OutputSumcheckParams::new` → `challenge_vector_optimized(log_k)` → `r_address`
6. `InstructionLookupsClaimReductionSumcheckParams::new` → `challenge_scalar` → `gamma_instr`
7. `BatchedSumcheck::verify` → append input_claims → `challenge_vector(5)` → batching_coeffs

### Factor Polynomial Order (Verified Matches `PRODUCT_UNIQUE_FACTOR_VIRTUALS`)

- [0] LeftInstructionInput
- [1] RightInstructionInput
- [2] InstructionFlags(IsRdNotZero) = index 6
- [3] OpFlags(WriteLookupOutputToRD) = index 6
- [4] OpFlags(Jump) = index 5
- [5] LookupOutput
- [6] InstructionFlags(Branch) = index 4
- [7] NextIsNoop

---

## Test Status

- ✅ 714/714 unit tests passing
- ✅ Proof serialization/deserialization working
- ✅ Stage 1 verification passing
- ❌ Stage 2 verification failing
