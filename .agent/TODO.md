# Zolt-Jolt Compatibility Implementation

## Status: Session 21 - Dory Proof Serialization Issue

## Current Progress

### Session 21 Fixes
1. **SumcheckId**: Fixed to 22 variants to match Jolt (committed)
   - UNTRUSTED_ADVICE_BASE = 0
   - TRUSTED_ADVICE_BASE = 22
   - COMMITTED_BASE = 44
   - VIRTUAL_BASE = 66

2. **Proof Config**: All 5 config fields are usize (8 bytes each)

### Current Issue: Dory Proof at Position 45183

The stepwise deserialization test shows all components pass until the Dory opening proof:
- Claims: OK (142 claims)
- Commitments: OK
- Stage 1-7 sumchecks: OK
- Dory opening proof: FAILS with memory allocation error (tries to allocate 508GB!)

**Root Cause Analysis**:
- Position 45183 should be the start of the Dory proof
- Bytes at 45183: `9592160e8cd56e98...` - doesn't look like a GT element
- Bytes before 45183: all zeros (suspicious!)
- The first 8 bytes are being interpreted as num_rounds = 10983951338711716501

**Expected Dory proof format (from dory-pcs crate ark_serde.rs)**:
1. VMV message: c (GT=384 bytes), d2 (GT=384 bytes), e1 (G1=32 compressed)
2. num_rounds (u32)
3. First messages per round: d1_left, d1_right, d2_left, d2_right (all GT), e1_beta (G1), e2_beta (G2)
4. Second messages per round: c_plus, c_minus (GT), e1_plus, e1_minus (G1), e2_plus, e2_minus (G2)
5. Final message: e1 (G1), e2 (G2)
6. nu (u32), sigma (u32)

**Investigation needed**:
- Is `serializeJoltProofWithDory` (mod.zig line 1464-1499) actually writing the Dory proof?
- Check if `writeDoryProof` is being called
- Compare proof file size before/after the fix

### Verified Working
- Claims parsing: All 142 claims deserialize successfully
- Commitments: Vec<GT> deserializes correctly
- All 7 stage sumchecks: Deserialize correctly
- UniSkip proofs: Deserialize correctly

### Next Steps
1. Add debug print to verify Dory proof is written
2. Check if the serialization code path is correct
3. May need to check `jolt_serialization.zig` writeDoryProof function

## Key Files
- Zolt Dory serialization: `src/zkvm/jolt_serialization.zig` lines 148-185
- Zolt proof serialization: `src/zkvm/mod.zig` lines 1464-1499
- Jolt Dory format: dory-pcs crate `ark_serde.rs` lines 252-295

## Test Commands
```bash
# Generate proof and test
./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format -o logs/zolt_proof_dory.bin
cp logs/zolt_proof_dory.bin /tmp/
cd jolt && cargo test -p jolt-core --lib test_stepwise_deserialize -- --ignored --nocapture
```
