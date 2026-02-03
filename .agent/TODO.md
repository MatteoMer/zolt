# Zolt-Jolt Compatibility Implementation

## Status: Session 32 - Stage 5 InstructionReadRaf Verification Analysis

## CRITICAL FINDING

**Stage 5 sumcheck verification fails** at Instance 2 (InstructionReadRaf) with output_claim mismatch:
```
output_claim:   [ed, a5, f6, bf, 30, c4, 10, f8, 59, ce, db, ef, ee, 23, 2f, 96]... (LE)
expected_claim: [b2, 8f, 91, 24, 33, 0c, b4, 56, b9, 08, 89, 4c, fd, af, 54, 11]... (LE)
```

## Analysis Summary

### What's Working
1. **Polynomial coefficients match** - Rounds 0, 1, 2 sumcheck polynomials are identical
2. **InstructionRa claims serialize correctly** - High 16 bytes match between Zolt and Jolt
3. **Serialization format is correct** - LE field element encoding matches arkworks

### What's NOT Working
1. **ra_claim product differs** between Zolt and Jolt
   - Zolt: `BC 8E CE 3A 67 EF E8 02 58 90 C4 0A 65 C7 60 D0 63 BD C4 05 76 E5 5B 37 E5 44 9D DD E7 09 B6 29` (LE)
   - Jolt expects: `01 93 87 0f d1 f4 08 b0 6c 71 28 a5 7d 64 9d f1` (LE hi16 only)

### Root Cause Hypothesis

The individual InstructionRa[i] claims have matching **high 16 bytes** but different **low 16 bytes**. This causes:
1. The claims pass high-byte comparison
2. But the full field element multiplication produces different products

### Expected Output Claim Formula
```
expected_output_claim = eq_eval_r_reduction * ra_claim * (val_claim + gamma * raf_claim)
```

Where:
- `ra_claim = Π_{i=0}^{7} InstructionRa(i)` (product of 8 RA chunk claims)
- `val_claim = Σ_{i=0}^{41} LookupTableFlag(i) * table_i_eval`
- `raf_claim = (1 - raf_flag) * (left_op + gamma * right_op) + raf_flag * gamma * identity`

## Debug Data Collected

### Jolt ra_chunk claims (LE hi16 only):
```
ra_claims[0] = [18, d1, 65, 32, 94, 21, 95, 0a, 35, fb, 24, fd, bc, 79, 55, 19]
ra_claims[1] = [c4, 2c, b0, 4d, 2b, 6d, a0, 74, 70, 75, 5a, 16, 1e, 10, 10, 1f]
ra_claims[2] = [40, 1f, b9, 2a, d1, 23, 36, 7e, 30, 5f, 2e, 01, fe, 16, 79, 2c]
ra_claims[3] = [8d, a1, 41, 0d, a6, cb, f4, 03, 29, 00, 38, 0e, 59, 52, 84, 2c]
ra_claims[4] = [63, 09, ea, b4, 9e, 0a, e3, 09, b9, e8, 0f, 47, a7, 16, 59, 06]
ra_claims[5] = [49, 31, 9a, 89, 4d, b5, 27, e7, c7, a5, 1c, d7, cf, 1d, d4, 26]
ra_claims[6] = [19, d8, 05, 58, a1, b9, 3c, 69, 1d, c6, 97, 0c, 53, f3, 28, 0e]
ra_claims[7] = [2b, 2d, f7, 59, f1, 94, 04, 6d, 90, 42, 93, e8, ab, 3c, d9, 22]
```

### Zolt InstructionRa claims (BE full, need to reverse for LE):
```
[0] = { 25, 85, 121, 188, 253, 36, 251, 53, 10, 149, 33, 148, 50, 101, 209, 24, 61, 43, 110, 83, 139, 182, 157, 37, 78, 122, 13, 176, 155, 80, 22, 105 }
```
Hi16 match ✅, but need to verify full 32 bytes match

## Next Steps

1. [ ] Add full 32-byte debug output in Jolt's expected_output_claim
2. [ ] Compare full InstructionRa claim bytes (all 32 bytes) between Zolt and Jolt
3. [ ] If low 16 bytes differ, investigate ra_chunk_weights binding
4. [ ] Check if expanding table values match at cycle round start

## Test Commands

```bash
# Generate proof
./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format -o logs/zolt_proof_dory.bin --export-preprocessing logs/zolt_preprocessing.bin 2>&1 | tee /tmp/zolt_debug.log

# Verify with Jolt (debug mode)
cd jolt && cargo test -p jolt-core --features zolt-debug --lib test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture
```

## Files Modified This Session

- `/home/vivado/projects/jolt/jolt-core/src/zkvm/instruction_lookups/read_raf_checking.rs` - Added full 32-byte debug for ra_claims
- `/home/vivado/projects/zolt/src/zkvm/spartan/stage5_prover.zig` - Added full LE debug for ra_chunks

## Session Progress

- [x] Identify Stage 5 verification failure at Instance 2
- [x] Confirmed polynomial coefficients match
- [x] Verified InstructionRa high 16 bytes match
- [x] Analyzed expected_output_claim formula
- [x] Identified ra_claim product mismatch
- [ ] Debug full 32-byte InstructionRa claims
- [ ] Fix Stage 5 opening claims
