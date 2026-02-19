# Debugging Notes - Session 36 (Feb 18, 2026)

## KEY FINDINGS THIS SESSION

### 1. All tableCombine formulas VERIFIED to match (ALL 41 tables)
- Specifically checked GCD-specific tables 15, 16, 17, 31

### 2. All suffix orderings and MLE implementations match

### 3. Q suffix polynomial initialization matches Jolt

### 4. Everything individual component-level matches, yet GCD still fails

### 5. Right half Q polynomials are ALL ZERO for GCD at round 0

## NEXT APPROACH
Compare actual compressed polynomial coefficients (c0, c2) at round 0 between Jolt and Zolt.
If round 0 coefficients match, bug is in later rounds.
If they differ, need to find which instance contribution differs.

## BUILD COMMANDS
```bash
zig build -Doptimize=ReleaseFast
./zig-out/bin/zolt prove examples/gcd.elf --jolt-format -o /tmp/zolt_proof_dory.bin --export-preprocessing /tmp/zolt_preprocessing.bin
cd /home/vivado/projects/jolt && cargo test --package jolt-core --features zolt-debug zolt_compat_test::tests::test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture
```
