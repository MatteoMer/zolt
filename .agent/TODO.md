# Zolt-Jolt Compatibility: Status Update

## Status: DEBUGGING STAGE 4 inc_claim/wa_claim ZERO ISSUE

## Summary

Zolt can now:
1. Generate proofs for RISC-V programs (`./zig-out/bin/zolt prove`)
2. Verify proofs internally (`./zig-out/bin/zolt verify`) - ALL 6 STAGES PASS
3. Export proofs in Jolt-compatible format (`--jolt-format`)
4. Export preprocessing for Jolt verifier (`--export-preprocessing`)
5. Pass all 714 unit tests ✓
6. **Proof successfully deserializes in Jolt** ✓
7. **Preprocessing successfully deserializes in Jolt** ✓

## Cross-Verification Status

Verification fails at Stage 4: `Sumcheck verification failed`

## Current Issue Analysis (Session 72)

### Key Finding

Jolt's verifier is receiving ZERO values for `inc_claim` and `wa_claim` when computing ValEvaluation and ValFinal expected outputs:

```
ValEvaluation expected_output_claim debug:
  inc_claim: [00, 00, 00, 00, ...]  <-- ALL ZEROS!
  wa_claim: [00, 00, 00, 00, ...]   <-- ALL ZEROS!
  lt_eval: [a1, b1, 48, 02, ...]    <-- Non-zero (computed correctly)
  result (inc*wa*lt): [00, 00, ...]  <-- Zero because inc*wa = 0

ValFinal expected_output_claim debug:
  inc_claim: [00, 00, 00, 00, ...]  <-- ALL ZEROS!
  wa_claim: [00, 00, 00, 00, ...]   <-- ALL ZEROS!
  result (inc*wa): [00, 00, ...]     <-- Zero
```

BUT Zolt is computing NONZERO values:

```
[ZOLT STAGE4 VALEVAL DEBUG]
  val_eval_openings.inc_eval = { 9, 53, 218, 196, ... }   <-- NONZERO!
  val_eval_openings.wa_eval = { 17, 245, 253, 169, ... }  <-- NONZERO!
  val_eval_openings.lt_eval = { 45, 226, 211, 44, ... }   <-- NONZERO!
```

### Potential Issues

1. **Opening key mismatch**: Zolt serializes with key `(CommittedPolynomial::RamInc, SumcheckId::RamValEvaluation)`
   but Jolt might be looking up with a different key due to:
   - Enum ordering differences
   - Serialization format mismatch

2. **Value serialization**: The claim values are converted from Montgomery form using `fromMontgomery()`,
   but there might be an issue with how the bytes are written or interpreted.

### Serialization Debug Added

Added debug output to `jolt_types.zig` to print RamInc claims during serialization:
```zig
// In OpeningClaims.serialize():
// Debug: Print RamInc claims
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

### Next Steps

1. **Wait for prover to complete** - Currently running with debug output
2. **Compare serialized bytes** - Check if Zolt writes nonzero bytes for RamInc claims
3. **Verify Jolt lookup** - Debug how Jolt's `get_committed_polynomial_opening` finds claims
4. **Check BTreeMap ordering** - Jolt uses BTreeMap which orders by key; verify our ordering matches

### Sumcheck Instance Matches

All individual instances match after 15 rounds:
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

But the **expected_output_claim** computation fails because inc_claim and wa_claim are zero in Jolt.

### Key Files

- `/home/vivado/projects/zolt/src/zkvm/jolt_types.zig` - OpeningClaims serialization
  - Line 643-660: serialize() function with debug output added

- `/home/vivado/projects/zolt/src/zkvm/proof_converter.zig` - Stage 4 claim insertion
  - Line 2486-2489: RamInc/RamValEvaluation claim insertion

- `/home/vivado/projects/jolt/jolt-core/src/zkvm/ram/val_evaluation.rs` - Jolt verifier
  - Line 393-396: get_committed_polynomial_opening for inc_claim

### Test Commands

```bash
# Generate proof with debug output
./zig-out/bin/zolt prove examples/fibonacci.elf \
    --export-preprocessing /tmp/zolt_preprocessing.bin \
    -o /tmp/zolt_proof_dory.bin 2>&1 | grep "SERIALIZE DEBUG\|RamInc"

# Run Jolt verification with zolt-debug feature
cd /home/vivado/projects/jolt && \
cargo test --no-default-features --features minimal,zolt-debug --release \
    -p jolt-core test_verify_zolt_proof_with_zolt_preprocessing \
    -- --ignored --nocapture
```

## Files Generated

- `/home/vivado/projects/zolt/logs/zolt_proof_dory.bin` - Jolt-compatible proof
- `/home/vivado/projects/zolt/logs/zolt_preprocessing.bin` - Verifier preprocessing
