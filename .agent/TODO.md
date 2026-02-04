# Zolt-Jolt Compatibility Implementation

## Status: Session 37 - Transcript Divergence Investigation

## Critical Finding: Challenge Mismatch

The Stage 4 verification fails because Zolt's prover and Jolt's verifier compute different sumcheck challenges from Stage 2. This causes:

1. Different `r_address` points after normalize_opening_point()
2. Different `val_init(r_address)` evaluations
3. Non-zero `input_claim = rwc_val_claim - init_eval` when it should be zero

### Key Values (from debug)

**Zolt's rwc_val_claim (BE):**
```
[1d, f5, f8, 9c, 2e, b5, 56, 72, a4, 26, d4, f5, 6f, 65, 79, ba, 2e, c8, 3c, 7a, b4, f8, 65, 8a, eb, b6, 21, 13, 18, 26, c4, 60]
```

**Jolt's init_eval (LE):**
```
[ad, 81, 92, 5b, 94, d8, 3d, 01, 60, 8c, 94, f3, ef, e0, 65, 7d, 5c, 4a, e7, fa, ba, 9e, 3d, c6, 87, a7, f9, b1, 02, da, e4, 14]
```

These are completely different field values.

### Root Cause Theory

The sumcheck verifier recomputes challenges by:
1. Reading round polynomial from proof
2. Appending to transcript
3. Deriving challenge

If the transcript state diverges at any point, all subsequent challenges differ.

### Transcript Format Understanding

Both Zolt and Jolt use the same format for appending round polynomials:
```
append_message("UniPoly_begin")
for each coefficient:
    append_scalar(coefficient)
append_message("UniPoly_end")
```

### Challenge Format Understanding

Both use MontU128Challenge with `[0, 0, low, high]` format:
- `from_bigint_unchecked([0, 0, low, high])` treats input as Montgomery form
- Serialization converts to standard form: `[0, 0, low, high] * R^(-1) mod p`

### Next Steps for Next Session

1. **Add byte-level transcript comparison**
   - Print exact bytes appended to transcript in both Zolt and Jolt
   - Compare at each stage boundary

2. **Compare Stage 2 round polynomial coefficients**
   - Print coefficients from Zolt's proof
   - Print coefficients read by Jolt's verifier
   - Verify they match byte-for-byte

3. **Trace transcript state divergence point**
   - Start from Stage 2 beginning
   - Compare state after each operation
   - Identify exact point where states diverge

### Test Commands

```bash
# Generate Zolt proof with debug
./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format -o /tmp/zolt_proof_dory.bin --trace-length 1024 2>&1 | tee /tmp/zolt_debug.log

# Verify with Jolt debug
cd /home/vivado/projects/jolt && cargo test -p jolt-core --features zolt-debug --lib test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture 2>&1 | tee /tmp/jolt_debug.log
```

### Key Files

- Zolt transcript: `src/transcripts/blake2b.zig`
- Zolt proof converter: `src/zkvm/proof_converter.zig` (Stage 2 at line 2984)
- Jolt transcript: `jolt/jolt-core/src/transcripts/blake2b.rs`
- Jolt sumcheck: `jolt/jolt-core/src/subprotocols/sumcheck.rs`
- Jolt UniPoly: `jolt/jolt-core/src/poly/unipoly.rs` (append_to_transcript at line 479)

### Previous Session Findings

- Session 36: Found MontU128Challenge format `[0, 0, low, high]`
- Session 35: Stage 4 instance analysis (RegistersRWC, RamValEvaluation, RamValFinal)
- Jolt native tests pass (`fib_e2e_dory`)

SESSION_ENDING - Made significant progress understanding the transcript divergence issue. Next session should add detailed byte-level comparison between Zolt prover and Jolt verifier transcript operations.
