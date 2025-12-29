# Zolt-Jolt Compatibility TODO

## Current Status: Stage 1 Sumcheck Verification (Session 29)

### Latest Findings
The streaming round computation now iterates over (x_out, x_in) pairs correctly:
- group = in_idx & 1 (LSB of E_in index is group selector)
- cycle = (out_idx << num_in_prime_bits) | (in_idx >> 1)

After this fix, output claim changed significantly - now ~3x the expected claim.

```
output_claim:          21666336742712548502319747532501161924289540611986917453312419397056388736753
expected_output_claim: 6886947417280328868350115979630985812677709302801764445050583343275239326794
ratio: ~3.15x
```

### What Was Fixed This Session
- [x] Pass full tau to split_eq (m = tau.len/2 = 6 for len=12)
- [x] E_out now has 64 entries (2^6), E_in has 32 entries (2^5)
- [x] Streaming round index mapping: group from in_idx & 1, cycle from combining out_idx and in_idx >> 1

### Verified Formula Match
The Gruen poly construction matches Jolt:
- q_constant = t'(0) = grid_az[0] * grid_bz[0]
- q_quadratic_coeff = t'(∞) = slope_az * slope_bz
- Previous claim = s(0) + s(1) from sumcheck
- The cubic polynomial s(X) = l(X) * q(X) is computed correctly

### Possible Remaining Issues
1. **E_in bit ordering** - LSB is group selector, but need to verify MSB-first vs LSB-first table build
2. **Split eq table build** - may have big-endian vs little-endian issue in tau variable ordering
3. **current_scalar in split_eq** - after UniSkip binding, does it include the Lagrange kernel properly?
4. **r_grid weights** - may need to be integrated into the eq weight computation

### Next Investigation Steps
1. Add debug output to print E_out/E_in lengths and first few values
2. Print grid_az/grid_bz values to compare with Jolt
3. Check if current_scalar matches Jolt's after UniSkip binding
4. Verify the r_grid is being used correctly (for klen > 1 cases)

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
- [x] Tau split fix (pass full tau, not tau_low)
- [x] Streaming round index structure (group in LSB of in_idx)
