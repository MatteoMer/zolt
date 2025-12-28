# Zolt zkVM Implementation TODO

## Completed (This Session - Iteration 36)

### Lasso Prover Claim Tracking Fix ✅
- [x] Identify Lasso prover wasn't maintaining sumcheck invariant
- [x] Add `current_claim` field to track running claim
- [x] Add `eq_evals` array to track eq(r, j) evaluations per cycle
- [x] Add `eq_evals_len` to track effective array size (shrinks during folding)
- [x] Update `computeAddressRoundPoly` to use eq_evals for sums
- [x] Update `computeCycleRoundPoly` to use eq_evals_len
- [x] Update `receiveChallenge` to:
  - Bind eq_evals with (1-r) or r factors during address phase
  - Fold eq_evals in half during cycle phase
  - Recompute current_claim as sum of (folded) eq_evals
- [x] Add test verifying sumcheck invariant: p(0) + p(1) = current_claim
- [x] All 554 tests pass

## Key Insight from This Session

The Lasso prover wasn't maintaining the sumcheck claim invariant:
- For each round: p(0) + p(1) = current_claim
- After receiving challenge r: new_claim = p(r)

The fix required:
- Tracking current_claim explicitly
- Maintaining eq_evals array with cycle eq evaluations
- Properly binding/folding eq_evals when receiving challenges
- Recomputing current_claim after each challenge

## Completed (Previous Sessions)

### Iteration 35 - Transcript Synchronization Fix
- [x] Fix spurious "r1cs_tau" challenge in verifyR1CSProof
- [x] Add log_t and log_k fields to JoltStageProofs
- [x] Stage 1 (Spartan) now verifies with strict sumcheck

### Iteration 34 - Sumcheck Degree Mismatch Fix
- [x] Fix RAF prover to compute [p(0), p(2)] for degree-2 format
- [x] Add `evaluateQuadraticAt3Points` helper

### Iterations 1-33 - Core Implementation
- [x] BN254 field and curve arithmetic
- [x] Extension fields (Fp2, Fp6, Fp12)
- [x] Pairing with Miller loop and final exponentiation
- [x] HyperKZG commit, open, verify with batch support
- [x] Dory commit, open, verify with IPA
- [x] Sumcheck protocol
- [x] RISC-V emulator (RV64IMC)
- [x] ELF loader
- [x] MSM operations
- [x] Spartan proof generation and verification
- [x] Lasso lookup argument prover/verifier
- [x] Multi-stage prover (6 stages)
- [x] Host execute
- [x] Preprocessing
- [x] 50+ lookup tables
- [x] Complete RV64IM instruction coverage (60+ instructions)

## Next Steps (Future Iterations)

### High Priority
- [ ] Test Stage 3+ with strict verification mode
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
- Full pipeline example: WORKING

## Performance (from benchmarks)
- Field addition: 4.0 ns/op
- Field multiplication: 55.5 ns/op
- Field inversion: 11.8 us/op
- MSM (256 points): 0.50 ms/op
- HyperKZG commit (1024): 1.5 ms/op
