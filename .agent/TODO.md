# Zolt-Jolt Compatibility TODO

## 🎯 Current Status: Stage 4 eq_eval Mismatch (Transcript Divergence)

**Session 69 (2026-01-27)**

### ✅ Previous Fixes (from earlier sessions)

1. **Fixed proof header serialization** - Verification reaches Stage 4
2. **Stage 4 Round Polynomials are CORRECT** - Coefficient bytes match
3. **Fixed ValEvaluation and ValFinal** - Different init_eval points
4. **Stage 4 Input Claims now match** - All 3 instances (RegistersRWC, RamValEval, ValFinal)

### 🔴 Current Issue: Stage 4 Challenges Differ Between Prover and Verifier

**Symptom:** Stage 4 fails with `output_claim != expected_claim` because eq_eval values differ.

**Root Cause:** The Fiat-Shamir transcript produces different challenges!

**Debug Evidence:**
```
Zolt Stage 4 challenge #7: low=0x233d551037bcd329, high=0x04fd755bc3f284a6
Jolt r_cycle[0] (should be c7): low=0x03967406_1269aa55, high=0x022c6820_0a4a1842
```

These are COMPLETELY different values! This means the proof data diverges somewhere.

**Important:** Stage 3 challenges DO match perfectly, so the issue starts in Stage 4.

### 📊 Investigation Findings

1. **Understanding Jolt's normalize_opening_point:**
   - Phase 1: 8 rounds (cycle vars via Gruen)
   - Phase 2: 7 rounds (address vars via Gruen)
   - Phase 3: 0 rounds (for our 256-step trace)
   - r_cycle = [c7, c6, ..., c0] (challenge #7 through #0, reversed)
   - r_address = [c14, c13, ..., c8]

2. **Zolt's Current Approach is Correct (Ordering):**
   - Takes first 8 challenges, reverses → matches Jolt's r_cycle construction

3. **The Transcript Diverges:**
   Since both implementations use Fiat-Shamir, the proof data being appended must differ.

   Possible causes:
   - Round polynomial coefficients computed incorrectly
   - Different serialization of round poly coefficients
   - Missing or extra data being appended

### 📋 Next Steps

1. **Compare Round 0 coefficients exactly:**
   - Print c0, c2, c3 being appended in Zolt Stage 4 Round 0
   - Print what Jolt verifier reads from the proof
   - They should be identical bytes

2. **Check Stage 4 proof serialization:**
   - 3 coefficients per round: c0, c2, c3
   - Verify byte ordering

3. **Debug transcript state:**
   - Print Zolt's transcript state after each Stage 4 round
   - Compare with what Jolt's verifier's transcript produces

### Commands

```bash
# Generate proof
zig build -Doptimize=ReleaseFast run -- prove examples/fibonacci.elf --jolt-format -o /tmp/zolt_proof_dory.bin

# Copy files
cp /tmp/zolt_preprocessing.bin /tmp/jolt_verifier_preprocessing.dat

# Test verification with debug
cd /home/vivado/projects/jolt && cargo test --package jolt-core --no-default-features --features "minimal,zolt-debug" test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture
```

### Key Files

- `/home/vivado/projects/zolt/src/zkvm/proof_converter.zig` - Stage 4 proof generation (line 2038+)
- `/home/vivado/projects/zolt/src/zkvm/spartan/stage4_gruen_prover.zig` - Gruen eq prover
- `/home/vivado/projects/jolt/jolt-core/src/zkvm/registers/read_write_checking.rs` - Jolt verifier
- `/home/vivado/projects/jolt/jolt-core/src/subprotocols/sumcheck.rs` - Sumcheck verification

### Success Criteria

- [x] Proof header deserializes correctly
- [x] Stage 4 round polynomials serialize correctly (format)
- [x] ValEvaluation uses correct r_address (RWC point)
- [x] ValFinal uses correct r_address (OutputCheck point)
- [x] Stage 4 input claims match verifier expectations (all 3 instances)
- [ ] **Stage 4 challenges match between prover/verifier** ← CURRENT BLOCKER
- [ ] eq_eval computation matches Jolt verifier
- [ ] Stage 4 sumcheck verification passes
- [ ] Full Jolt verification passes
