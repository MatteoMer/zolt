# Zolt-Jolt Compatibility: Status Update

## Status: DEBUGGING PROOF DESERIALIZATION / STAGE 4 VALIDATION

## Summary

Zolt can now:
1. Generate proofs for RISC-V programs (`./zig-out/bin/zolt prove`)
2. Verify proofs internally (`./zig-out/bin/zolt verify`) - ALL 6 STAGES PASS
3. Export proofs in Jolt-compatible format (`--jolt-format`)
4. Export preprocessing for Jolt verifier (`--export-preprocessing`)
5. Pass all 714 unit tests ✓
6. **Proof successfully deserializes in Jolt** (when matching preprocessing)
7. **Preprocessing successfully deserializes in Jolt** ✓

## Cross-Verification Status

Verification fails at Stage 4: `Sumcheck verification failed`

## Current Issue Analysis (Session 72)

### Two Separate Issues Identified

#### Issue 1: Proof Deserialization Failure
When proof/preprocessing are out of sync:
```
Failed to deserialize proof: the input buffer contained invalid data
```
**Fix**: Always regenerate proof and preprocessing together

#### Issue 2: Stage 4 inc_claim/wa_claim Zero in Jolt

Jolt's verifier receives ZERO for `inc_claim` and `wa_claim`:

```
ValEvaluation expected_output_claim debug:
  inc_claim: [00, 00, 00, 00, ...]  <-- ALL ZEROS!
  wa_claim: [00, 00, 00, 00, ...]   <-- ALL ZEROS!
  lt_eval: [a1, b1, 48, 02, ...]    <-- Non-zero (computed correctly)
  result (inc*wa*lt): [00, 00, ...]  <-- Zero because inc*wa = 0
```

But Zolt computes NONZERO values:
```
[ZOLT STAGE4 VALEVAL DEBUG]
  val_eval_openings.inc_eval = { 9, 53, 218, 196, ... }   <-- NONZERO!
  val_eval_openings.wa_eval = { 17, 245, 253, 169, ... }  <-- NONZERO!
```

### Serialization Format Analysis

Verified that OpeningId serialization format matches between Zolt and Jolt:
- UNTRUSTED_ADVICE_BASE = 0
- TRUSTED_ADVICE_BASE = 24
- COMMITTED_BASE = 48
- VIRTUAL_BASE = 72

For `CommittedPolynomial::RamInc` (value 1) with `SumcheckId::RamValEvaluation` (value 10):
- fused = 48 + 10 = 58
- poly = 1

Both Zolt and Jolt use the same encoding.

### Sumcheck Instance Matching Verified

All 5 instances match at the end of Stage 4:
```
[ZOLT DEBUG] inst0 MATCH: true
[ZOLT DEBUG] inst1 MATCH: true
[ZOLT DEBUG] inst2 MATCH: true
[ZOLT DEBUG] inst3 MATCH: true
[ZOLT DEBUG] inst4 MATCH: true
[ZOLT DEBUG] expected_batched (from provers) = { 6, 218, 102, ... }
[ZOLT DEBUG] actual batched = { 6, 218, 102, ... }
[ZOLT DEBUG] MATCH: true
```

The sumcheck round polynomials are correct, but the expected output claim fails
because Jolt receives zero for the opening claims.

### Next Steps

1. **Regenerate proof/preprocessing** - Currently running, ~5 minutes
2. **Add debug to Jolt deserialization** - Print the raw bytes being read for RamInc claims
3. **Compare byte-by-byte** - hexdump both Zolt output and check against Jolt's expected format
4. **Check BTreeMap iteration order** - Verify the claims are written in the order Jolt expects

### Debug Code Added

In `src/zkvm/jolt_types.zig` - Added debug output for RamInc claims during serialization:
```zig
// In OpeningClaims.serialize():
switch (entry.id) {
    .Committed => |c| {
        switch (c.poly) {
            .RamInc => {
                std.debug.print("[SERIALIZE DEBUG] Claim {}: RamInc/{} = {any}\n", ...);
            },
            ...
        }
    },
    ...
}
```

### Test Commands

```bash
# Generate proof and preprocessing (must be run together)
./zig-out/bin/zolt prove examples/fibonacci.elf \
    --export-preprocessing /home/vivado/projects/zolt/logs/zolt_preprocessing.bin \
    -o /home/vivado/projects/zolt/logs/zolt_proof_dory.bin

# Run Jolt verification with debug
cd /home/vivado/projects/jolt && \
cargo test --no-default-features --features minimal,zolt-debug --release \
    -p jolt-core test_verify_zolt_proof_with_zolt_preprocessing \
    -- --ignored --nocapture

# Hexdump opening claims section
xxd /home/vivado/projects/zolt/logs/zolt_proof_dory.bin | head -100
```

### Key Files

- `src/zkvm/jolt_types.zig` - OpeningId and OpeningClaims serialization
- `src/zkvm/proof_converter.zig` - Stage 4 claim insertion (lines 2486-2512)
- Jolt: `jolt-core/src/poly/opening_proof.rs` - OpeningId deserialization
- Jolt: `jolt-core/src/zkvm/proof_serialization.rs` - Claims deserialization

## Currently Running

Prover is regenerating proof and preprocessing:
```bash
./zig-out/bin/zolt prove examples/fibonacci.elf \
    --export-preprocessing /home/vivado/projects/zolt/logs/zolt_preprocessing.bin \
    -o /home/vivado/projects/zolt/logs/zolt_proof_dory.bin
```

Check with: `pgrep -f "zig-out/bin/zolt" && tail -20 /tmp/prover_debug.log`

SESSION_ENDING - Prover still running, need to verify deserialization and debug inc_claim/wa_claim issue
