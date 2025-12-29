# Zolt-Jolt Compatibility TODO

## Current Status: Stage 1 Sumcheck Verification (Session 30 continued)

### Issue Summary
The Stage 1 sumcheck output_claim is approximately 1.52x the expected_output_claim.

```
output_claim:          9012353217108547355521203237420571194773588078319073875630688947404723196510
expected_output_claim: 13679333034475978057044998736602529104372847349794408879776350094438916822983
ratio: ~1.52 (close to 3/2)
```

### Verified Correct
- [x] tau_low extraction: tau_low = tau[0..tau.len-1]
- [x] tau_high storage: stored separately for first-round Lagrange kernel
- [x] Split eq initialization with m = tau_low.len / 2 = 5
- [x] E_out and E_in both have 32 entries (2^5)
- [x] Opening claims correctly read (LeftInstructionInput, RightInstructionInput, etc.)
- [x] Gruen polynomial construction formula matches Jolt
- [x] Constraint group combination: az_final = az_g0 + r_stream * (az_g1 - az_g0)
- [x] Lagrange kernel is symmetric: lagrange_kernel(x, y) = lagrange_kernel(y, x)
- [x] interpolateDegree3 and evalsToCompressed are correct

### Possible Issues (Need Investigation)
1. **Eq polynomial accumulation** - current_scalar may accumulate differently
2. **Variable binding order** - may be LSB-first vs MSB-first mismatch
3. **E_in/E_out table indexing** - bit ordering in tables
4. **r_grid weights** - may affect streaming round computation

### Debug Data
```
tau_high_bound_r0:          10811398959691251178446374398729567517345474364208367822721217673367518413943
tau_bound_r_tail_reversed:  14125962160514979480084655056025740513842614775637540494241851278366681977590
inner_sum_prod:             2309286487192234790818585784375390705895687692935567051482424615198743651405
r0:                         11697823044030274153413191308021134249784459050027548723260376992474506918271
rx_constr[0] (r_stream):    2116934244015962401899708433894043262098824377802461496614220883524162579815
sumcheck_challenges.len:    11
```

### Next Steps
1. Add debug output to print t_zero/t_infinity at each round
2. Compare current_scalar value at each round with Jolt
3. Verify eq table values match Jolt's
4. Check if r_grid is being updated correctly

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
- [x] Split eq polynomial factorization (tau_low)
- [x] Lagrange kernel computation
- [x] Opening claims with MLE evaluation
- [x] Challenge Montgomery form conversion
- [x] Store tau_high separately
- [x] Constraint group combination matches Jolt
