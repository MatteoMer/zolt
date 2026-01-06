# Zolt-Jolt Compatibility TODO

## Current Status: January 6, 2026 - Session 62

**STATUS: STAGE 1 PASSES - Stage 2 fails at univariate skip**

### Session 62 Progress

**STAGE 1 NOW PASSES!**

Root cause of Stage 1 failure was **Fiat-Shamir preamble mismatch**:
- Jolt verifier was loading I/O from `/tmp/fib_io_device.bin` with inputs=[32] and outputs=[e1,f2,cc,f1,2e]
- Zolt was proving with empty inputs/outputs (bare-metal fibonacci.elf doesn't use Jolt I/O convention)
- This caused transcript divergence from the very beginning

**Fix applied:**
- Modified Jolt test `test_verify_zolt_proof` to use empty I/O (`JoltDevice::default()`)
- This matches Zolt's bare-metal execution

**Current error:**
```
Verification failed: Stage 2 univariate skip first round
Caused by: ProductVirtual uni-skip first-round verification failed
```

### Comparison Script

Created `scripts/compare_sumcheck.py` for debugging:
- Compares Fiat-Shamir preamble values
- Shows all values in hex with both BE/LE interpretations
- Compares commitments, initial claims, round polynomials
- Identifies root cause of mismatches

### What Now Matches

| Component | Status |
|-----------|--------|
| Preamble (all fields) | ✅ |
| Commitments (all 5) | ✅ |
| Stage 1 sumcheck | ✅ |
| Stage 2 uni-skip | ❌ |

### Next Steps

1. Debug Stage 2 univariate skip first round
2. Compare Stage 2 values between Zolt and Jolt
3. The ProductVirtual uni-skip verification is failing

### Test Commands

```bash
# Generate Zolt proof (with empty I/O)
./zig-out/bin/zolt prove --jolt-format -o /tmp/zolt_proof_dory.bin examples/fibonacci.elf

# Run Jolt verification
cargo test --package jolt-core test_verify_zolt_proof -- --ignored --nocapture

# Run comparison script
python3 scripts/compare_sumcheck.py /tmp/zolt.log /tmp/jolt.log
```

### Previous Session Notes (Session 61)

Previous issue was off-by-one in split_eq tau handling - that may still be relevant for Stage 2.
