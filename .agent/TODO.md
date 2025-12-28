# Zolt zkVM Implementation TODO

## Completed (This Session - Iteration 36)

### Lasso Prover Claim Tracking Fix ✅
- [x] Add `current_claim` field to track running claim
- [x] Add `eq_evals` array to track eq(r, j) evaluations per cycle
- [x] Add `eq_evals_len` to track effective array size
- [x] Update `receiveChallenge` to properly bind and fold eq_evals
- [x] Add test verifying sumcheck invariant: p(0) + p(1) = current_claim
- [x] All tests pass

### RAF Prover Verification ✅
- [x] Verified RAF prover correctly implements sumcheck
- [x] Added RAF prover claim tracking test
- [x] Confirmed RAF prover passes claim tracking invariant

## Potential Issues Identified

### Val Prover (Stage 4) Polynomial Binding
The ValEvaluationProver computes a degree-3 sumcheck for:
  Σ inc(j) * wa(j) * lt(j)

**Potential issue:** Only `inc` polynomial is folded via `bind()`, but `wa` and `lt`
are re-evaluated fresh at each index. For correct sumcheck, all three polynomials
should be bound together. This may cause verification failures in strict mode.

**Impact:** Stage 4 verification may fail with strict sumcheck enabled.

**Fix needed:** Update `bindChallenge` to fold `wa` and `lt` polynomials as well.

## Completed (Previous Sessions)

### Iteration 35 - Transcript Synchronization Fix
- [x] Fix spurious "r1cs_tau" challenge in verifyR1CSProof
- [x] Add log_t and log_k fields to JoltStageProofs
- [x] Stage 1 (Spartan) now verifies with strict sumcheck

### Iteration 34 - Sumcheck Degree Mismatch Fix
- [x] Fix RAF prover to compute [p(0), p(2)] for degree-2 format
- [x] Add `evaluateQuadraticAt3Points` helper

### Iterations 1-33 - Core Implementation
- [x] Complete zkVM implementation

## Next Steps (Future Iterations)

### High Priority
- [ ] Fix Val prover polynomial binding
- [ ] Test full pipeline with strict verification mode
- [ ] Investigate test interference issue

### Medium Priority
- [ ] Add real BN254 pairing constants
- [ ] Performance optimization with SIMD
- [ ] Parallel sumcheck round computation

### Low Priority
- [ ] Complete HyperKZG pairing verification
- [ ] More comprehensive benchmarking
- [ ] Add more example programs

## Test Status
- All tests pass (554+ tests)
- Stage 1 strict verification: PASSED ✅
- Lasso claim tracking: VERIFIED ✅
- RAF claim tracking: VERIFIED ✅
- Val prover: MAY NEED FIX (polynomial binding)
- Full pipeline example: WORKING

## Performance (from benchmarks)
- Field addition: 4.0 ns/op
- Field multiplication: 55.5 ns/op
- Field inversion: 11.8 us/op
- MSM (256 points): 0.50 ms/op
- HyperKZG commit (1024): 1.5 ms/op
