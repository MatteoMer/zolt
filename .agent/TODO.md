# Zolt-Jolt Compatibility TODO

## Current Status
Stage 3 prefix-suffix optimization implemented but needs verification

## Current Task
- [ ] Debug Stage 3 ShiftSumcheck prefix-suffix implementation
  - Algorithm structure is correct
  - Need to verify P and Q buffer initialization matches Jolt exactly
  - Key issue: Must match Jolt's exact decomposition formula

## Implementation Done
- [x] Implement Stage 3 prover structure
- [x] Implement ShiftPrefixSuffixProver with 4 (P,Q) pairs
- [x] Implement InstructionInputProver (direct computation)
- [x] Implement RegistersPrefixSuffixProver with 1 (P,Q) pair

## Pending
- [ ] Test Stage 3 against Jolt verifier with debug output comparison
- [ ] Implement Stages 4-7 sumcheck provers
- [ ] Full proof verification test

## Key Formulas

### EqPlusOnePrefixSuffixPoly Decomposition
```
eq+1((r_hi, r_lo), (y_hi, y_lo)) =
    eq+1(r_lo, y_lo) * eq(r_hi, y_hi) +
    is_max(r_lo) * is_min(y_lo) * eq+1(r_hi, y_hi)
```

Where:
- r is split at mid: r_hi = r[0..mid], r_lo = r[mid..n]
- prefix_0[j] = eq+1(r_lo, j)
- suffix_0[j] = eq(r_hi, j)
- prefix_1[0] = is_max(r_lo), all other indices = 0
- suffix_1[j] = eq+1(r_hi, j)
- is_max(x) = product of x[i] for all i (= eq((1)^n, x))

### Q Buffer Construction
```
Q[x_lo] = Σ_{x_hi} witness(x) * suffix[x_hi]
where x = x_lo + (x_hi << prefix_n_vars)
```

### Phase 1 Round Polynomial
```
H(X) = Σ_pairs Σ_{i < half} P[i](X) * Q[i](X)
where:
  P[i](X) = P[2*i] + X * (P[2*i+1] - P[2*i])  (linear interpolation)
  Q[i](X) = Q[2*i] + X * (Q[2*i+1] - Q[2*i])

Evaluations returned:
  evals[0] = H(0) = Σ P[2*i] * Q[2*i]
  evals[1] = H(2) = Σ (2*P[2*i+1] - P[2*i]) * (2*Q[2*i+1] - Q[2*i])
```

### Binding (LowToHigh)
```
new[i] = old[2*i] + r * (old[2*i+1] - old[2*i])
```

## Notes
- Jolt uses BIG_ENDIAN opening points: r[0] is MSB
- Sumcheck binds LowToHigh (index 0, 1, 2, ...)
- Phase 1: Bind prefix variables (first prefix_n_vars rounds)
- Phase 2: Bind suffix variables (remaining rounds)
- Transition when prefix buffer size == 2 (1 variable left)

## Overall Progress

| Stage | Status | Notes |
|-------|--------|-------|
| 1 | ✓ PASSES | Outer Spartan sumcheck works |
| 2 | ✓ PASSES | Product virtualization works |
| 3 | In Progress | Prefix-suffix optimization implemented |
| 4-7 | Blocked | Waiting on Stage 3 |

## Testing Commands

```bash
# Generate proof
./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format \
  --export-preprocessing /tmp/zolt_preprocessing.bin \
  -o /tmp/zolt_proof_dory.bin --srs /tmp/jolt_dory_srs.bin

# Verify with Jolt
cd /Users/matteo/projects/jolt && cargo test -p jolt-core \
  test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture

# Run Zolt tests
zig build test
```
