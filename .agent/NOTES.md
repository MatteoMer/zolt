# Zolt Implementation Notes

## Current Status (2024-12-28)

### What Works
- **Proof Serialization**: Byte-perfect Arkworks compatibility
- **Transcript**: Blake2b matches Jolt exactly
- **Opening Claims**: Non-zero MLE evaluations computed correctly
- **Proof Structure**: All 7 stages, correct round counts
- **SRS Loading**: arkworks format with flag bit handling
- **G1 MSM**: Row commitments match Jolt exactly
- **G2 Generator**: Matches arkworks exactly
- **G2 Points**: Coordinates from SRS match exactly

### What's Failing
- **Pairing Function**: e(G1_gen, G2_gen) produces different result from Jolt
  - Zolt: f5 a1 b9 0d 00 81 ca 8f 26 1e 63 72 1d f6 f7 1b...
  - Jolt: 95 0e 87 9d 73 63 1f 5e b5 78 85 89 eb 5f 7e f8...
  - Issue is in Miller loop or final exponentiation

---

## Latest Progress: Dory Commitment Debugging

### arkworks Flag Bit Fix (Iteration 20)

arkworks `serialize_uncompressed` stores metadata flags in top 2 bits of last byte:
- bit 7: y-sign flag
- bit 6: infinity flag

Fix: `y_limbs[3] &= 0x3FFFFFFFFFFFFFFF` to clear flags before use.

Without this fix, y coordinates were larger than field modulus, causing
Montgomery conversion failures and broken curve arithmetic.

### Verified Components

| Component | Status | Notes |
|-----------|--------|-------|
| G1 points on curve | ✅ | All 4 G1 points valid |
| G2 generator | ✅ | Exact match with arkworks |
| G2[0] coordinates | ✅ | All 4 Fp components match |
| Row 0 MSM | ✅ | Exact match |
| Row 1 MSM | ✅ | Exact match |
| Pairing | ❌ | Different result |

### Test Commands

```bash
# Export Jolt SRS
cd /Users/matteo/projects/jolt
cargo test --package jolt-core test_export_dory_srs -- --ignored --nocapture

# Run Jolt debug test
cargo test --package jolt-core test_export_dory_commitment_debug -- --ignored --nocapture

# Run Zolt tests
cd /Users/matteo/projects/zolt
zig build test
```

---

## Pairing Debugging Plan

The pairing function `e: G1 × G2 → GT` uses:
1. Miller loop
2. Final exponentiation

Since G1 and G2 inputs match exactly, the difference is in:
- Miller loop line function computations
- Fp2/Fp6/Fp12 tower arithmetic
- Final exponentiation algorithm

Next steps:
1. Compare intermediate Miller loop values
2. Check Fp2 multiplication constants (e.g., ξ = 9 + u)
3. Verify final exponentiation formula

---

## Previous Implementation Notes

### Univariate Skip Implementation (Iteration 11) - SUCCESS

Successfully implemented Jolt's univariate skip optimization for stages 1-2:

1. **univariate_skip.zig** - Core module with:
   - Constants matching Jolt (NUM_R1CS_CONSTRAINTS=19, DEGREE=9, NUM_COEFFS=28)
   - `buildUniskipFirstRoundPoly()` - Produces degree-27 polynomial from extended evals
   - `LagrangePolynomial` - Interpolation on extended symmetric domain
   - `uniskipTargets()` - Compute extended evaluation points

2. **spartan/outer.zig** - Spartan outer prover:
   - `SpartanOuterProver` with univariate skip support
   - `computeUniskipFirstRoundPoly()` - Generates proper first-round polynomial

3. **proof_converter.zig** - Updated to generate proper-degree polynomials:
   - Stage 1: `createUniSkipProofStage1()` - 28 coefficients (degree 27)
   - Stage 2: `createUniSkipProofStage2()` - 13 coefficients (degree 12)

### Blake2b Transcript Compatibility (Complete)

Successfully implemented Blake2b transcript matching Jolt's implementation:
- 32-byte state with round counter
- Messages right-padded to 32 bytes
- Scalars serialized LE then reversed to BE (EVM format)
- 128-bit challenges
- Vector operations with begin/end markers

All 7 test vectors from Jolt verified to match.

### Dory Commitment Implementation (In Progress)

**Location**: `src/poly/commitment/dory.zig`

1. **DoryCommitmentScheme** - Matches Jolt's DoryCommitmentScheme
   - `setup(allocator, max_num_vars)` - Generate SRS using "Jolt Dory URS seed"
   - `commit(params, evals)` - Commit polynomial to GT element
   - `loadFromFile()` - Load arkworks-format SRS from file
   - DorySRS with G1/G2 generators
   - DoryCommitment = GT = Fp12

2. **GT (Fp12) Serialization** - Added to `src/field/pairing.zig`
   - `Fp12.toBytes()` - 384 bytes arkworks format (12 × 32 bytes)
   - `Fp12.fromBytes()` - Deserialize from arkworks format
   - Serialization order: c0.c0.c0, c0.c0.c1, ..., c1.c2.c1

### Cross-Verification Status

**Jolt successfully deserializes Zolt proofs!**

```
cargo test --package jolt-core test_deserialize_zolt_proof -- --ignored --nocapture

Successfully deserialized Zolt proof!
  Trace length: 8
  RAM K: 65536
  Bytecode K: 65536
  Commitments: 5
```

### Test Status

All Zolt tests pass:
```
zig build test --summary all
Build Summary: tests passed
```

Cross-verification tests (Jolt):
- `test_deserialize_zolt_proof`: PASS
- `test_debug_zolt_format`: PASS
- `test_export_dory_srs`: PASS
- `test_export_dory_commitment_debug`: PASS (MSM matches, pairing differs)
- `test_verify_zolt_proof`: FAIL (commitment mismatch due to pairing)

---

## File Locations

### SRS File
`/tmp/jolt_dory_srs.bin` - 1000 bytes for max_num_vars=3

### Format
```
Header: "JOLT_DORY_SRS_V1" (16 bytes)
max_num_vars: u64 (8 bytes)
g1_count: u64 (8 bytes)
G1 points: 4 * 64 bytes = 256 bytes (uncompressed affine)
g2_count: u64 (8 bytes)
G2 points: 4 * 128 bytes = 512 bytes (uncompressed affine)
h1: 64 bytes
h2: 128 bytes
```

### Key Source Files

| File | Purpose |
|------|---------|
| `src/poly/commitment/dory.zig` | Dory commitment scheme |
| `src/field/pairing.zig` | BN254 pairing (needs fix) |
| `src/field/mod.zig` | Montgomery field arithmetic |
| `src/msm/mod.zig` | Multi-scalar multiplication |
