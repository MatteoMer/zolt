# Zolt-Jolt Compatibility: Current Status

## Status: Stage 2 Verification Failure - Transcript Analysis Needed 🔴

## Session 74 Summary (2026-01-29)

### Key Finding: Zolt's Prover is INTERNALLY CONSISTENT

**Evidence:**
- Stage 2 `output_claim` (sumcheck evaluation) = `expected_batched` (prover formula) ✓
- All 5 instance claims match internally ✓
- Factor claims stored at correct (poly, sumcheck_id) pairs ✓

**Implication:** Jolt's verifier must be computing different expected_output_claim.

### Verified Correct

1. **SumcheckId enum** - 22 values matching Jolt
2. **Factor claim indices**:
   - InstructionFlags::IsRdNotZero = 6 ✓
   - InstructionFlags::Branch = 4 ✓
   - OpFlags::Jump = 5 ✓
   - OpFlags::WriteLookupOutputToRD = 6 ✓
3. **R1CS input ordering** - `R1CS_VIRTUAL_POLYS` matches Jolt's `ALL_R1CS_INPUTS`
4. **Transcript message labels**:
   - "UniPoly_begin/end" for CompressedUniPoly ✓
   - "UncompressedUniPoly_begin/end" for UniPoly ✓
5. **Scalar encoding** - LE to BE reversal matches ✓

### Stage 1 Sumcheck Verified Working

Zolt generates real Stage 1 round polynomials with:
- 9 rounds (1 streaming + 8 cycle vars)
- Non-zero coefficients (c0, c2, c3)
- Challenges derived from transcript

Example:
```
[ZOLT] STAGE1_ROUND_0: c0 = { 179, 252, 58, 169, ... }
[ZOLT] STAGE1_ROUND_0: challenge = { 66, 175, 255, 237, ... }
```

### Suspected Issue: Transcript State Divergence

Transcript state before tau_high sampling:
- Zolt: `{ 37, 204, 55, 100, 179, 84, 234, 62 }`

If Jolt's verifier has different state here, everything downstream will fail.

Possible causes:
1. Stage 1 UniSkip polynomial encoding differs
2. Stage 1 sumcheck round polynomial encoding differs
3. Stage 1 cache_openings claim values differ
4. Some transcript operation differs

### Blocking Issue

Cannot run Jolt tests directly:
```
pkg-config: command not found
openssl-sys build failed
```

### Next Steps

1. [ ] **HIGH PRIORITY**: Get system dependencies installed to run Jolt tests with `zolt-debug`
2. [ ] Compare Stage 1 UniSkip polynomial bytes between Zolt and Jolt
3. [ ] Compare Stage 1 round polynomial bytes
4. [ ] Compare Stage 1 final claim values

### Commands

```bash
# Generate proof
./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format -o proof.bin --srs /tmp/jolt_dory_srs.bin

# Verify with Jolt (needs openssl)
cd /home/vivado/projects/jolt && cargo test --features zolt-debug test_verify_zolt_proof -- --ignored --nocapture
```

---

## Previous Sessions

### Session 73 (2026-01-29)
- Fixed SumcheckId mismatch (22 values, not 24)
- Fixed proof serialization (5 advice options, 5 usize config)
- Proof deserializes completely ✓

### Session 72 (2026-01-28)
- 714/714 unit tests passing
- Stage 3 mathematically verified
