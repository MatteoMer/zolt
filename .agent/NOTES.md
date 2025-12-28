# Zolt-Jolt Compatibility Notes

## Current Status (December 28, 2024, Session 10)

### Session 10 Progress

**Major Progress on Transcript Compatibility!**

1. **Fixed Montgomery Form Issue**: `appendScalar` was serializing Montgomery form
   limbs directly instead of converting to canonical form first. Fixed by adding
   `fromMontgomery()` call before serialization.

2. **Transcript States Now Match**: After the Montgomery form fix, the transcript
   states match exactly at the point of r0 derivation:
   - Jolt state: `[51, 28, c0, 92, ab, 81, 34, c6, ...]`
   - Zolt state: `5128c092ab8134c6178d3b18...`

3. **Remaining Issue**: Despite matching states, the r0 challenge values differ:
   - Jolt r0: `3203159906685754656633863192913202159923849199052541271036524843387280424960`
   - Zolt raw bytes: `c9e8fb5d94a6ff9206ac6f469cec1467`
   - Expected (from Jolt): `c9e8fb5d94a6ff9206ac6f469cec1407`
   - Difference: Last byte is `67` vs `07`

### Analysis of Challenge Mismatch

The Blake2b hasher includes `n_rounds` in the hash input:
```
hasher() = Blake2b256(state || [0u8; 28] || n_rounds.to_be_bytes())
```

If `n_rounds` differs between Zolt and Jolt at the same state, the hash outputs
will differ. This is the likely cause.

When deriving r0:
- Zolt: n_rounds = 55
- Jolt: n_rounds = ? (needs verification)

### Root Cause Hypothesis

The state can match even if n_rounds differs if:
- Different sequences of operations led to the same final state
- But the number of operations (and thus n_rounds) differs

For example:
- Path A: 55 operations → state X
- Path B: 50 operations → state X (via different intermediate states)

This shouldn't happen if both follow the same protocol... unless there's a subtle
difference in how operations are counted or performed.

### Next Steps

1. Add n_rounds logging to track counter at each major step
2. Create minimal reproducible test case
3. Compare operation counts between Zolt prover and Jolt verifier

---

## Working Components ✅

| Component | Status | Notes |
|-----------|--------|-------|
| Blake2b Transcript | ⚠️ Almost | State matches, n_rounds may differ |
| Dory Commitment | ✅ Working | GT elements match, MSM correct |
| Proof Structure | ✅ Working | 7 stages, claims, all parse |
| Serialization | ✅ Working | Byte-level compatible |
| UniSkip Algorithm | ✅ Working | Domain sum = 0 verified |
| Preprocessing Export | ✅ Working | Full JoltVerifierPreprocessing |
| DoryVerifierSetup | ✅ Working | Precomputed pairings |

---

## Session History

### Session 10
- Fixed Montgomery form serialization in `appendScalar`
- Transcript states now match at r0 derivation
- Identified n_rounds counter mismatch as likely cause

### Session 9
- Stage 1 UniSkip verification passes
- Fixed Lagrange interpolation bug

### Session 7-8
- DoryVerifierSetup implementation
- Full preprocessing export

---

## Key Files

### Zolt
- `src/transcripts/blake2b.zig` - Blake2b transcript
- `src/zkvm/proof_converter.zig` - Proof conversion with transcript
- `src/field/mod.zig` - Field element serialization

### Jolt (Reference)
- `jolt-core/src/transcripts/blake2b.rs` - Reference transcript
- `jolt-core/src/zolt_compat_test.rs` - Compatibility tests

---

## Commands

```bash
# Test Zolt (all 632 tests)
zig build test --summary all

# Generate proof
zig build -Doptimize=ReleaseFast
./zig-out/bin/zolt prove examples/sum.elf --jolt-format \
    --export-preprocessing /tmp/zolt_preprocessing.bin \
    -o /tmp/zolt_proof_dory.bin

# Run Jolt debug test
cd /Users/matteo/projects/jolt
cargo test --package jolt-core test_debug_stage1_verification -- --ignored --nocapture
```
