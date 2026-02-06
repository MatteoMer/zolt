# Zolt-Jolt Compatibility Implementation

## Status: Session 106 - r_cycle mismatch identified in ValEvaluation

### Key Finding

Challenge representation is CORRECT! We verified:
1. Challenge limbs `[0, 0, low, high]` match Jolt exactly
2. Arithmetic operations (sub, mul) produce identical results
3. The "quasi-Montgomery" representation is correct

### Current Issue: Stage 4 Verification Failure

The Stage 4 (Gruen prover) verification fails because `expected_output_claim` doesn't match.

**Root cause appears to be in ValEvaluation**: The verifier's ValEvaluation computes input_claim using r_cycle from Stage 2's RamReadWriteChecking opening, but there's a mismatch:

From Jolt debug output:
```
r_cycle (from Stage 2 RamVal opening):
  r_cycle[0]: [56, 18, 3f, 29, b6, 41, 44, 9a, cc, 27, 7d, c1, 4d, 21, ea, 2b]

r (from Stage 4 sumcheck challenges, normalized):
  r[0]: [d8, c9, 49, 15, 0d, ea, 97, 30, ff, 2a, 0e, 6b, ed, 64, c8, 0d]
```

These are completely different! The ValEvaluation gets r_address/r_cycle from the RamVal@RamReadWriteChecking opening, which is:
- Computed by the verifier using `normalize_opening_point(stage2_challenges)`
- Stored in `VerifierOpeningAccumulator` during Stage 2 verification
- Retrieved by ValEvaluation via `get_virtual_polynomial_opening`

### Hypothesis

The issue might be that Zolt's Stage 4 is computing r_address/r_cycle differently than how Jolt's verifier expects. The verifier re-computes these from the Stage 2 challenges during verification, not from the proof data.

Key insight from Jolt's `cache_openings`:
```rust
let opening_point = self.params.normalize_opening_point(sumcheck_challenges);
accumulator.append_virtual(transcript, VirtualPolynomial::RamVal, SumcheckId::RamReadWriteChecking, opening_point.clone(), claim);
```

The verifier calls this during `verify_claims` to store the opening point. So the opening point is derived from the Stage 2 sumcheck challenges.

### Investigation Needed

1. Check if Zolt's Stage 4 uses the SAME challenges that the verifier uses for `normalize_opening_point`
2. The verifier's `cache_openings` is called after Stage 2 sumcheck verification
3. Zolt may be using different challenge indices or order

### Files to Check

- `/home/vivado/projects/jolt/jolt-core/src/zkvm/ram/read_write_checking.rs` - cache_openings, normalize_opening_point
- `/home/vivado/projects/zolt/src/zkvm/proof_converter.zig` - Stage 4 Gruen prover r_address computation

### Test Commands

```bash
# Generate proof
zig build run -Doptimize=ReleaseFast -- prove examples/fibonacci.elf --jolt-format -o /tmp/zolt_proof.bin --srs /tmp/jolt_dory_srs.bin

# Verify with debug output
cd ../jolt && cargo run --release --features zolt-debug --manifest-path examples/fibonacci/Cargo.toml -- --verify-zolt-proof /tmp/zolt_proof.bin 2>&1 | tail -300
```

## Previous Sessions

- Session 106: Verified challenge representation is correct; identified r_cycle mismatch in ValEvaluation
- Session 105: Identified ValEval/ValFinal input_claim mismatch
- Session 104: Fixed proof serialization format (--jolt-format flag)
- Session 103-101: Various Stage 5 fixes
