# Zolt-Jolt Compatibility TODO

## Current Status: Stage 1 Sumcheck Verification (Session 27)

### Latest Findings
- 128-bit challenges are now converted to Montgomery form
- Output claim and expected claim still don't match (ratio ~0.91)
- Issue is NOT in challenge format or eq polynomial evaluation

```
output_claim:          11331697095435039208873616544229270298263565208265409364435501006937104790550
expected_output_claim: 12484965348201065871489189011985428966546791723664683385883331440930509110658
```

### Verified Working
- [x] All 656 Zolt tests pass
- [x] Proof deserialization (all 48 opening claims parsed)
- [x] Jolt preprocessing loads correctly
- [x] Verifier instance creation succeeds
- [x] 128-bit challenges in Montgomery form
- [x] EqPolynomial evaluation
- [x] r_cycle computation (challenges[1..] reversed)
- [x] UniSkip verification passes
- [x] Individual round equations (p(0)+p(1)=claim)

### Likely Issues
The mismatch is likely in one of:
1. **R1CS witness values** - Do Zolt's witnesses match what Jolt expects for the same program?
2. **Streaming sumcheck computation** - Is Az*Bz being accumulated correctly?
3. **Trace organization** - Are cycles/constraints ordered the same way?

### Next Steps
1. Debug R1CS witnesses - compare values between Zolt and Jolt
2. Add intermediate value logging to streaming sumcheck
3. Verify trace structure matches Jolt's expectations

## Test Commands
```bash
# Zolt tests
zig build test --summary all

# Generate proof
zig build && ./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format -o /tmp/zolt_proof_dory.bin

# Jolt verification test
cd /Users/matteo/projects/jolt
cargo test --package jolt-core test_verify_zolt_proof -- --ignored --nocapture
```

## Completed Milestones
- [x] Blake2b transcript implementation
- [x] Field serialization (Arkworks format)
- [x] UniSkip polynomial generation
- [x] Stage 1 remaining rounds sumcheck
- [x] R1CS constraint definitions
- [x] Split eq polynomial factorization
- [x] Lagrange kernel computation
- [x] Opening claims with MLE evaluation
- [x] Challenge Montgomery form conversion
