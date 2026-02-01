# Zolt-Jolt Compatibility Implementation

## Status: IN PROGRESS - Stage 5 Expected Output Claim Mismatch

## Current Session Progress (Session 91)

### Key Fix: r_reduction source corrected!

**BUG FOUND AND FIXED**: Zolt was using `stage3_result.challenges` for `r_reduction`, but the correct source is `stage2_result.challenges` (last n_cycle_vars = 8 challenges from Stage 2's InstructionClaimReduction).

**Fix applied** in `proof_converter.zig`:
- Changed `r_reduction_be` to be extracted from Stage 2 challenges (indices 16-23 for max_rounds=24, n_cycle_vars=8)
- r_reduction comes from Stage 2 InstructionClaimReduction, NOT Stage 3

### Verification: eq_r_reduction now matches!
- Zolt's eq_r_reduction: `8349e6eb71ecb0c07088c5aa7c4d7b5a` (BE)
- Jolt's eq_eval_r_reduction: `[5a, 7b, 4d, 7c, aa, c5, 88, 70, c0, b0, ec, 71, eb, e6, 49, 83]` (LE)
- These are the SAME value, just reversed endianness!

### Remaining Issue: Stage 5 expected_output_claim still doesn't match

The sumcheck rounds all pass, but the final claim comparison fails:
- `output_claim:   [bd, 7a, 64, 13, 7c, 97, 3f, 42, ...]`
- `expected_claim: [b2, d7, 91, f3, d5, d1, 0e, 0e, ...]`

The expected_claim is computed from:
```
expected = eq_eval_r_reduction * ra_claim * (val_claim + gamma * raf_claim)
```

The mismatch suggests one of these opening claims doesn't match:
- `val_claim` - Lookup value polynomial at opening point
- `ra_claim` - Product of ra virtual selector evaluations
- `raf_claim` - RAF flag polynomial evaluation

### Next Steps
1. Debug val_claim, ra_claim, raf_claim values between Zolt and Jolt
2. Verify that Zolt's InstructionReadRaf prover is computing these claims correctly
3. Check if the claims are being serialized correctly to the proof

### Files Changed This Session
- `/home/vivado/projects/zolt/src/zkvm/proof_converter.zig`:
  - Fixed r_reduction_be to use Stage 2 challenges (InstructionClaimReduction)
  - Added debug output for r_reduction limbs vs toBytesBE

### Technical Context
- Stage 5 has 3 batched instances: RegistersValEvaluation (8 rounds), RamRaClaimReduction (24 rounds), LookupsReadRaf (136 rounds)
- Instance 2 (LookupsReadRaf) expected_output_claim = eq_r_reduction * ra_claim * (val_claim + gamma * raf_claim)
- r_reduction comes from Stage 2's InstructionClaimReduction challenges (indices 16-23 in BIG_ENDIAN after reversal)
- r_cycle_prime comes from Stage 5's cycle rounds (challenges 128-135)

### Test Commands
```bash
# Build
zig build -Doptimize=ReleaseFast

# Generate proof
./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format --export-preprocessing /tmp/zolt_preprocessing.bin -o /tmp/zolt_proof_dory.bin

# Verify with Jolt
cd /home/vivado/projects/jolt
cargo test --package jolt-core --features zolt-debug test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture
```
