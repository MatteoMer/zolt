# Zolt-Jolt Compatibility TODO

## 🎯 Current Status: Stage 4 Verification Failing (Round Polys Correct!)

**Session 3 (2026-01-27)**

### ✅ Completed Fixes

1. **Fixed proof header serialization**
   - Changed from incorrect advice fields + usize config to correct format
   - Fixed in `mod.zig::serializeJoltProofWithDory`
   - Verification now reaches Stage 4

2. **Proof header deserializes correctly:**
   - Trace length: 256 ✓
   - RAM K: 65536 ✓
   - Bytecode K: 65536 ✓
   - 37 Dory commitments ✓
   - 91 opening claims ✓

3. **Stage 4 Round Polynomials are CORRECT:**
   - Verified coefficient bytes match between Zolt serialized proof and Jolt deserialized proof
   - Round 0: [635762d5, 32e0242c, 2d3fe12e] matches in both systems

### 🔴 Current Issue: Stage 4 Sumcheck Verification Fails

**Error:** "Stage 4: Sumcheck verification failed"

**Investigation Complete:**
- Stage 4 round polynomial coefficients are correct
- The issue is likely:
  1. **Transcript state mismatch** - Different Fiat-Shamir challenges
  2. **Input claim computation** - Verifier computes different input_claim
  3. **Batching coefficient mismatch** - Different coefficients for combining sumchecks

**Key Debug Output from Zolt:**
```
[ZOLT STAGE4] input_claim_registers_BE = { 4, 68, 118, 36, ... }
[ZOLT STAGE4] input_claim_val_eval_BE = { 0, 0, ... }  (zeros - no RAM)
[ZOLT STAGE4] input_claim_val_final_BE = { 0, 0, ... }  (zeros - no RAM)
[ZOLT STAGE4] batching_coeff[0]_BE = { 0, 0, ..., 24, 154, 37, 79, ... }
[ZOLT STAGE4] Initial batched_claim (BE) = { 24, 197, 195, 16, ... }
```

**Hypothesis:**
The Jolt verifier computes `input_claim` for each sumcheck instance from the `opening_accumulator`. If the claims stored in the proof don't match what the verifier expects, verification fails.

### 📋 Next Steps

1. **Add debug to Jolt verifier** to print:
   - Input claims for each Stage 4 instance
   - Batching coefficients computed from transcript
   - Running claim at each round
   - Compare with Zolt's values

2. **Check transcript state alignment:**
   - Verify transcript is in same state before Stage 4
   - Compare gamma challenge between Zolt and Jolt

3. **Verify opening_claims lookup:**
   - Ensure claims are stored with correct OpeningId keys
   - Check Rs1Value, Rs2Value, RdWriteValue claims match

### Commands

```bash
# Generate proof
zig build -Doptimize=ReleaseFast run -- prove examples/fibonacci.elf --jolt-format -o /tmp/zolt_proof_dory.bin

# Copy files to expected paths
cp /tmp/zolt_proof_dory.bin /tmp/zolt_proof_dory.bin
cp /tmp/zolt_preprocessing.bin /tmp/jolt_verifier_preprocessing.dat

# Test verification
cd /home/vivado/projects/jolt && cargo test --package jolt-core --no-default-features --features minimal test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture

# Debug Stage 4
cd /home/vivado/projects/jolt && cargo test --package jolt-core --no-default-features --features minimal test_debug_stage4_verification -- --ignored --nocapture
```

### Key Files

- `/home/vivado/projects/zolt/src/zkvm/mod.zig` - serializeJoltProofWithDory
- `/home/vivado/projects/zolt/src/zkvm/proof_converter.zig` - Stage 4 proof generation
- `/home/vivado/projects/jolt/jolt-core/src/zkvm/verifier.rs` - Stage 4 verification
- `/home/vivado/projects/jolt/jolt-core/src/subprotocols/sumcheck.rs` - BatchedSumcheck::verify

### Success Criteria

- [x] Proof header deserializes correctly
- [x] Claims parse correctly
- [x] Commitments parse correctly
- [x] Stage 4 round polynomials serialize correctly
- [ ] Stage 4 input claims match verifier expectations
- [ ] Stage 4 sumcheck verification passes
- [ ] Full Jolt verification passes
- [ ] `zig build test` passes
