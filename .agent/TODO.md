# Zolt-Jolt Compatibility Implementation

## Status: Session 24 - Stage 5 Deep Dive

## Current Issue

Stage 5 verification fails - sumcheck output_claim doesn't match expected_claim.

**From Jolt debug (Session 24):**
- output_claim: `[ed, a5, f6, bf, 30, c4, 10, f8, 59, ce, db, ef, ...]`
- expected_claim: `[b2, 8f, 91, 24, 33, 0c, b4, 56, b9, 08, 89, 4c, ...]`

## Key Findings This Session

### 1. MontU128Challenge Format
- Stored as `[0, 0, low, high]` where low and high are u64 limbs
- Serializes as 32 bytes: 16 zeros + 16 value bytes (low LE, high LE)
- Zolt's `challengeScalar128Bits()` produces the same format

### 2. r_reduction Source
- Comes from Stage 2 InstructionClaimReduction sumcheck (Instance 4)
- Stored in `OpeningPoint.r` as `Vec<F::Challenge>`
- After `normalize_opening_point`, reversed to BIG_ENDIAN order
- Zolt extracts from `stage2_result.challenges[16..24]` and reverses

### 3. Jolt's Stage 5 Debug Output
```
r_reduction[0]: [00, 00, ..., 0d, 8d, 89, b0, c0, ef, 00, b0, 84, a4, 8a, 1b, 0b, 14, 34, 07]
```
This is the Challenge serialization with low=0xb000efc0b0898d0d, high=0x0734140b1b8aa484

### 4. Opening Claims Match
- `InstructionRa(0)` claim from proof: `[18, d1, 65, 32, ...]`
- `ra_claims[0]` from verifier: `[18, d1, 65, 32, ...]`
- These match, so opening claims are correct!

### 5. Round-by-Round Verification
- Stage 5 has 136 rounds (128 address + 8 cycle)
- Round 0 coefficients from Zolt proof match what verifier sees
- Round 135 new_claim = output_claim = `[ed, a5, f6, bf, ...]`

## Diagnosis

The sumcheck polynomial computation in Zolt produces a different final claim than expected. Since opening claims match, the issue is in how Zolt computes the polynomial during each round.

### Most Likely Causes:
1. **eq polynomial computation difference** - How `eq(r_reduction, r_cycle_prime)` is computed
2. **combined_val formula mismatch** - The table_val + raf_contribution formula
3. **Polynomial binding order** - LOW_TO_HIGH vs some other order

## Next Steps

1. **Add more debug output to Zolt's Stage 5** - Print round-by-round values
2. **Compare Zolt's round 128-135 coefficients** with Jolt's debug output
3. **Verify eq_prefix extraction** - Check `eq_0 / (1 - r_round)` formula
4. **Check combined_val computation** - Compare with Jolt's cycle round formula

## Test Commands
```bash
# Jolt verification with debug
cd jolt && cargo test -p jolt-core --features zolt-debug --lib test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture 2>&1 | grep -A 50 "Stage5"

# NOTE: Zolt test OOMs on this machine - use smaller test or remote run
```

## Key Files
- Zolt Stage 5: `src/zkvm/spartan/stage5_prover.zig`
- Jolt Stage 5 verifier: `jolt-core/src/zkvm/verifier.rs`
- Jolt InstructionReadRaf: `jolt-core/src/zkvm/instruction_lookups/read_raf_checking.rs`
- r_reduction extraction: `src/zkvm/proof_converter.zig:1662-1687`

SESSION_ENDING - Investigation of MontU128Challenge format and r_reduction source complete. The sumcheck polynomial computation in Zolt's Stage 5 prover produces different values than expected. Next session should compare Zolt's internal computations with Jolt's cycle round formula.
