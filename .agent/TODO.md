# Zolt-Jolt Compatibility TODO

## Current Progress

| Stage | Status | Notes |
|-------|--------|-------|
| 1 | ✅ PASS | Univariate skip sumcheck |
| 2 | ❌ FAIL | RamReadWriteChecking round polynomial mismatch |
| 3 | ⏳ Blocked | Waiting for Stage 2 |
| 4 | ⏳ Blocked | Waiting for Stage 3 |
| 5-7 | ⏳ Blocked | Waiting for Stage 4 |

## Stage 2 Root Cause Analysis (Session 37)

### Key Discovery: Fibonacci Has 1 Memory Operation!
- Entry at cycle=54, addr=2049 (RAM address 0x80004008)
- This is for output/termination mechanism
- The RWC sumcheck is NOT a zero polynomial!

### Stage 2 Failure Details
```
output_claim:          8531846635557760858493086388539389736059592909629786934607653033526197973299
expected_output_claim: 17808130426384425926507399004264546912094784764713076233980989102782648691939
```

### Root Cause: GruenSplitEqPolynomial Optimization Required

**The Problem:**

Jolt's RamReadWriteChecking sumcheck uses the Dao-Thaler + Gruen optimization
(GruenSplitEqPolynomial) which produces specific round polynomial coefficients.

Zolt's naive direct computation produces mathematically equivalent polynomials,
but with DIFFERENT coefficient representations. Since Fiat-Shamir challenges
depend on the exact coefficients:
- Different coefficients → different challenges
- Different challenges → different final claims
- Different claims → verification fails

This is the EXACT same issue we faced with Stage 3's Shift sumcheck (Session 32),
which was fixed by implementing the prefix-suffix optimization.

**Jolt's Implementation:**
- Uses GruenSplitEqPolynomial from `jolt-core/src/poly/split_eq_poly.rs`
- 800+ lines implementing Dao-Thaler + Gruen optimization
- Paper reference: https://eprint.iacr.org/2024/1210.pdf

### Solution Required

Implement GruenSplitEqPolynomial for Zolt's RWC prover to match Jolt's
round polynomial coefficients exactly.

Key components:
1. Split eq(w, x) into three parts: E_out, E_in, current variable
2. Cache prefix eq tables for efficient computation
3. Use `gruen_poly_deg_3` formula for cubic round polynomials
4. Match binding order (LowToHigh)

### Debug Output Added
```
[ZOLT] STAGE2 RWC: entries.len = 1
[ZOLT] STAGE2 RWC: entry[0]: cycle=54, addr=2049
[ZOLT] STAGE2 RWC: ra_claim = non-zero
[ZOLT] STAGE2 RWC: val_claim = non-zero
[ZOLT] STAGE2 RWC: inc_claim = non-zero
```

## Testing Commands

```bash
# Build and run prover
zig build && ./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format --export-preprocessing logs/zolt_preprocessing.bin -o logs/zolt_proof_dory.bin --srs /tmp/jolt_dory_srs.bin

# Run Jolt verifier
cd /Users/matteo/projects/jolt && ZOLT_LOGS_DIR=/Users/matteo/projects/zolt/logs cargo test --package jolt-core test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture
```

## Files to Modify

1. `src/zkvm/ram/read_write_checking.zig` - Implement GruenSplitEqPolynomial optimization
2. May need new `src/poly/split_eq_poly.zig` for shared implementation
