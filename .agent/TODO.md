# Zolt-Jolt Compatibility: Current Status

## Status: Stage 2 Verification Failure - Challenge Type Analysis Complete 🔴

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

### Blocking Issue

Cannot run Jolt tests to compare transcript state:
```
pkg-config: command not found
openssl-sys build failed
```

Need: `sudo apt-get install pkg-config libssl-dev`

### Remaining Hypothesis

Since challenge types and order are correct, the issue must be:
1. **Transcript state divergence** - Some bytes being appended differently
2. **Factor evaluation values** - MLE computation at r_cycle differs
3. **Opening claim storage** - Values stored at wrong keys

### Next Steps (Priority Order)

1. [ ] **HIGH PRIORITY**: Install system deps to run Jolt with `zolt-debug`
2. [ ] Compare transcript state hex dump at tau_high sampling point
3. [ ] Compare batching_coeffs[0..4] between Zolt and Jolt
4. [ ] Compare factor_evals with Jolt's ProductVirtualEval::compute_claimed_factors

### Test Status

- ✅ 714/714 unit tests passing
- ❌ Integration test OOM killed (signal 9)

---

## Session 74 Summary (2026-01-29)

### Key Finding: Zolt's Prover is INTERNALLY CONSISTENT

**Evidence:**
- Stage 2 `output_claim` (sumcheck evaluation) = `expected_batched` (prover formula) ✓
- All 5 instance claims match internally ✓
- Factor claims stored at correct (poly, sumcheck_id) pairs ✓

**Implication:** Jolt's verifier must be computing different expected_output_claim.

### Verified Correct

1. **SumcheckId enum** - 22 values matching Jolt
2. **Factor claim indices**:
   - InstructionFlags::IsRdNotZero = 6 ✓
   - InstructionFlags::Branch = 4 ✓
   - OpFlags::Jump = 5 ✓
   - OpFlags::WriteLookupOutputToRD = 6 ✓
3. **R1CS input ordering** - `R1CS_VIRTUAL_POLYS` matches Jolt's `ALL_R1CS_INPUTS`
4. **Transcript message labels**:
   - "UniPoly_begin/end" for CompressedUniPoly ✓
   - "UncompressedUniPoly_begin/end" for UniPoly ✓
5. **Scalar encoding** - LE to BE reversal matches ✓

### Suspected Issue: Transcript State Divergence

Transcript state before tau_high sampling:
- Zolt: `{ 37, 204, 55, 100, 179, 84, 234, 62 }`

---

## Previous Sessions

### Session 73 (2026-01-29)
- Fixed SumcheckId mismatch (22 values, not 24)
- Fixed proof serialization (5 advice options, 5 usize config)
- Proof deserializes completely ✓

### Session 72 (2026-01-28)
- 714/714 unit tests passing
- Stage 3 mathematically verified
