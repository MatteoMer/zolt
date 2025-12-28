# Zolt-Jolt Compatibility TODO

## Phase 1: Transcript Compatibility ✅ COMPLETE
- [x] Create Blake2bTranscript in Zolt
- [x] Port Blake2b-256 hash function
- [x] Implement 32-byte state with round counter
- [x] Match Jolt's append/challenge methods exactly
- [x] Test vector validation - same inputs produce same challenges

## Phase 2: Proof Structure Refactoring ✅ COMPLETE
- [x] Restructure JoltProof in zkvm/mod.zig
- [x] Add 7 explicit stage proof fields
- [x] Match stage ordering with Jolt
- [x] Opening claims structure for batched verification

## Phase 3: Serialization Alignment ✅ COMPLETE
- [x] Implement arkworks-compatible field element serialization
- [x] Remove ZOLT magic header (pure arkworks format)
- [x] Match usize encoding (u64 little-endian)
- [x] GT/G1/G2 point serialization in arkworks format
- [x] Dory commitment serialization

## Phase 4: Commitment Scheme ✅ COMPLETE
- [x] Complete Dory implementation with Jolt-compatible SRS
- [x] SRS loading from Jolt-exported files
- [x] MSM with same point format as arkworks
- [x] Pairing operations matching arkworks

## Phase 5: Verifier Preprocessing Export ✅ COMPLETE
- [x] DoryVerifierSetup structure with precomputed pairings
- [x] delta_1l, delta_1r, delta_2l, delta_2r, chi computation
- [x] Full GT element serialization (Fp12 -> 12 * 32 bytes)
- [x] G1/G2 point serialization with flags
- [x] JoltVerifierPreprocessing (generators + shared)
- [x] CLI --export-preprocessing includes verifier setup

## Phase 6: Integration Testing 🚧 IN PROGRESS

### Proof/Preprocessing Deserialization ✅
- [x] Jolt can deserialize Zolt proof in --jolt-format
- [x] Opening claims: 48 entries, all valid
- [x] Commitments: 5 GT elements, all valid
- [x] Sumcheck proofs: structure matches
- [x] DoryVerifierSetup parses correctly
- [x] Full JoltVerifierPreprocessing::deserialize_uncompressed works

### Stage 1 UniSkip Verification ❌ FAILING
**Root Cause Identified:**
The polynomial does not sum to zero over the evaluation domain.

Debug test output:
```
UniSkip proof analysis:
  uni_poly degree: 27
  uni_poly num coeffs: 28
  All coefficients zero: false

Domain sum check:
  Input claim (expected domain sum): 0
  Power sums array (first 5): [10, 5, 85, 125, 1333]
  Computed domain sum: 5449091566537931454238289696340678230446604034401384111412173487916094860136
  Sum equals input_claim: false
```

**Issues Found:**
1. `streaming_outer.zig:interpolateFirstRoundPoly()` is BROKEN
   - Currently just copies evaluations as coefficients
   - Should do proper Lagrange interpolation + Lagrange kernel multiplication

2. Constraint evaluations may produce non-zero values
   - Need to verify that valid R1CS witnesses produce Az*Bz = 0

---

## Current Status: FIX INTERPOLATION

The `interpolateFirstRoundPoly` function in `src/zkvm/spartan/streaming_outer.zig` needs to:

1. Take extended domain evaluations `t1_vals[i]` for i in symmetric window around 0
2. Interpolate to get `t1(Y)` polynomial coefficients
3. Compute Lagrange kernel `L(τ_high, Y)` coefficients
4. Multiply polynomials to get `s1(Y) = L(τ_high, Y) * t1(Y)`
5. Return coefficients of `s1(Y)`

Reference: `jolt-core/src/subprotocols/univariate_skip.rs:build_uniskip_first_round_poly`

---

## Next Steps

1. **Fix `interpolateFirstRoundPoly`**
   - Implement proper Lagrange interpolation
   - Multiply by Lagrange kernel L(τ_high, Y)

2. **Verify R1CS constraints**
   - For a valid execution, Az(x)*Bz(x) should be 0 for all x
   - Check that constraint evaluators are correct

3. **Run e2e test again**

---

## Commands

```bash
# Run all tests
zig build test --summary all

# Build release
zig build -Doptimize=ReleaseFast

# Generate proof in Jolt format
./zig-out/bin/zolt prove examples/sum.elf \
    --jolt-format \
    --export-preprocessing /tmp/zolt_preprocessing.bin \
    -o /tmp/zolt_proof_dory.bin

# Run Jolt debug test
cd /Users/matteo/projects/jolt
cargo test --package jolt-core test_debug_stage1_verification -- --ignored --nocapture

# Run Jolt e2e test
cargo test --package jolt-core test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture
```

## File Sizes
- Proof (Jolt format): 30.9 KB (30,926 bytes)
- Preprocessing: 62.2 KB (62,223 bytes)
