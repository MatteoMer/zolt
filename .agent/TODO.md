# Zolt-Jolt Compatibility - Iteration 14 Status

## Summary

**Status**: Stage 1 passes, Stage 2 has transcript mismatch in r_address sampling.

### New Finding
The previous "EQ polynomial ordering" hypothesis was incorrect. The actual issue is **r_address values differ between Zolt and Jolt**.

### Completed ✓
1. All 712 Zolt internal tests pass
2. Stage 1 verification PASSES in Jolt
3. Fixed computeEqEvals to use big-endian ordering (matching Jolt)
4. gamma_rwc matches between Zolt and Jolt ✓

### Current Issue: r_address Mismatch

Despite correct round counts and gamma_rwc matching, r_address values differ:
- Zolt r_address[0]: 19039565092020683038185790567646070878007400533548331450519833455249204308507
- Jolt r_address[0]: 7962179938130852157218692959767706856444757081102233151385452154497670316032
- gamma_rwc: MATCHES ✓ (72439255043403958777314423659479935078)

### Transcript Flow Analysis

Zolt transcript operations for Stage 2 batched sumcheck:
1. `challengeScalarFull` round=179 → gamma_rwc (MATCHES JOLT ✓)
2. `challengeScalar128Bits` rounds 180-195 → r_address (16 challenges)
3. `challengeScalarFull` round=196 → gamma_instr

The round count is correct (16 r_address challenges). But the FIRST r_address challenge differs.

### Hypothesis

The issue is likely in how bytes are interpreted after gamma_rwc:
- Jolt's `challenge_scalar` uses 16 bytes → reversed → `F::from_bytes` (from_le_bytes_mod_order)
- Jolt's `challenge_scalar_optimized` uses 16 bytes → reversed → u128 → masked to 125 bits → MontU128Challenge

The critical question: What are the RAW bytes for r_address[0]?

In Jolt:
- challenge_u128 gets 16 bytes, reverses them, interprets as BE u128
- MontU128Challenge::from(u128) masks to 125 bits and stores in Montgomery form

In Zolt:
- challengeScalar128Bits gets 16 bytes, reverses them, masks to 125 bits, stores in Montgomery form

### Next Step

Add debug output to compare the raw 16 bytes extracted for the first r_address challenge in both Zolt and Jolt.

### Files with Debug Output
- `src/transcripts/blake2b.zig` - Added debug for challengeScalar128Bits
- `jolt-core/src/zkvm/ram/output_check.rs` - Added r_address debug
- `jolt-core/src/zkvm/ram/read_write_checking.rs` - Added gamma debug

### Tests Status
- All 578+ tests pass ✓
- Stage 1 passes ✓
- Stage 2 OutputSumcheck fails (r_address mismatch)
