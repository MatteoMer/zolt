# Zolt-Jolt Compatibility Implementation

## Status: Session 34 - Serialization Complete, Stage 1 Sumcheck Mismatch

## MAJOR MILESTONE - Proof Deserialization Working!

All serialization format issues are fixed. The Zolt proof now deserializes correctly in Jolt and verification begins.

### Serialization Fixes This Session

1. **Config field types** (Session 33): Fixed `one_hot_config` and `rw_config` to write as u8 instead of usize

2. **Option field count** (Session 34): Removed 4 extra Option bytes
   - JoltProof has ONLY `untrusted_advice_commitment: Option<PCS::Commitment>`
   - Zolt was incorrectly writing 5 Option bytes (for non-existent advice proof fields)
   - Fixed to write only 1 Option byte for `untrusted_advice_commitment`
   - Proof size: 64143 -> 64139 bytes (-4 bytes)

### Current Status

- **Serialization: COMPLETE**
- **Proof deserialization: WORKING**
- **Verifier creation: WORKING**
- **Stage 1 verification: FAILING (sumcheck mismatch)**

### Stage 1 Sumcheck Verification Analysis

The verifier outputs:
```
initial_claim: [f2, 98, aa, 2d, ...]  (from transcript state)
first round coeffs: [56, cb, ...], [9e, d4, ...], [aa, 70, ...]

After 11 rounds:
  output_claim:   [22, 3d, 0b, 1d, ...]  (computed from proof)
  expected_claim: [73, 61, e4, f4, ...]  (from verifier instances)
```

The mismatch indicates one of:
1. Zolt's sumcheck polynomial coefficients are incorrect
2. Zolt's transcript state differs from Jolt's
3. Commitment contributions to transcript don't match

## Next Steps

1. [ ] **Compare transcript states** - Add debug to both Zolt and Jolt to compare transcript state at Stage 1 start
2. [ ] **Verify commitment order** - Ensure Zolt appends commitments to transcript in same order as Jolt
3. [ ] **Compare sumcheck polynomial computation** - Verify Zolt computes Stage 1 sumcheck polys correctly

## Test Commands

```bash
# Generate Zolt proof using Jolt's fibonacci guest
./zig-out/bin/zolt prove /tmp/jolt-guest-targets/fibonacci-guest-fib/riscv64imac-unknown-none-elf/release/fibonacci-guest --jolt-format -o /tmp/zolt_proof_dory.bin --trace-length 1024 --input-hex 32

# Use Jolt's preprocessing (critical)
cp /tmp/jolt_verifier_preprocessing.dat /tmp/zolt_preprocessing.bin

# Run Jolt verifier with debug output
cd /home/vivado/projects/jolt && cargo test -p jolt-core --features zolt-debug --lib test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture
```

## Files Modified This Session

- `src/zkvm/mod.zig` - Removed extra Option bytes in both serialization paths (COMMITTED)

## Key Discoveries

1. JoltProof has only ONE optional field: `untrusted_advice_commitment`
2. The 4 "advice proof" fields mentioned in code comments don't exist in JoltProof struct
3. Jolt's Instruction type uses JSON serialization inside arkworks CanonicalSerialize
4. Stage 1 is "SpartanOuter" - the outer sumcheck for the Spartan R1CS proof

## Proof Structure (Working)

```
Opening Claims: 142 claims
Commitments: 39 GT elements (384 bytes each)
Stage 1 UniSkip: 28 coeffs
Stage 1 Sumcheck: 11 rounds
Stage 2 UniSkip: 13 coeffs
Stage 2 Sumcheck: 26 rounds
Stage 3 Sumcheck: 10 rounds
Stage 4 Sumcheck: 17 rounds
Stage 5 Sumcheck: 138 rounds
Stage 6 Sumcheck: 26 rounds
Stage 7 Sumcheck: 4 rounds
Joint Opening Proof: Dory proof
untrusted_advice_commitment: None (1 byte)
trace_length: 1024 (8 bytes)
ram_K: 65536 (8 bytes)
bytecode_K: 65536 (8 bytes)
rw_config: 4 u8 fields
one_hot_config: 2 u8 fields
dory_layout: 1 u8
```

Total: 64139 bytes
