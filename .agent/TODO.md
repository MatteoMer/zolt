# Zolt-Jolt Compatibility Implementation

## Status: IN PROGRESS - Stage 5 Debugging

## Current Issue (2026-01-30)

### Progress Made
1. **Fixed: Inactive instance polynomial computation** - Changed from `p(x) = claim` to `p(x) = claim/2` to satisfy sumcheck invariant
2. **Stages 1-4 PASS** ✅

### Stage 5 Failure Analysis

**Test output:**
- Instance 0 (RegistersValEvaluation): input_claim = 20196670... (non-zero!)
- Instance 1 (RamRaClaimReduction): input_claim = 16410442... (non-zero!)
- Instance 2 (LookupsReadRaf): input_claim = 9299828... (non-zero!)

**Verification mismatch:**
- output_claim = 11423923503841537814317862404954452055449709571592944381151147753965436781440
- expected_output_claim = 13492124496804440808267970253322341201831628230026493905363906851561539710310

**Root cause:** Input claims are non-zero because we have actual register/RAM operations. We implement RegistersValEvaluation with real trace data, but:
1. RamRaClaimReduction uses zero polynomials (TODO)
2. LookupsReadRaf uses zero polynomials (TODO)

### RegistersValEvaluation Analysis

The expected_output_claim for RegistersValEvaluation is computed as:
```
expected = inc_claim * wa_claim * lt_eval
```

Where:
- `inc_claim` = RdInc polynomial evaluated at sumcheck challenges r'
- `wa_claim` = RdWa virtual polynomial at (r_address, r')
- `lt_eval` = LT(r', r_cycle) computed by verifier

Instance 0 shows:
- expected_claim = 17580643... (non-zero)

This is the product of our opening claims and the LT evaluation. We're providing some non-zero inc_claim and wa_claim from our trace-based prover.

### Suspected Issues

1. **LT polynomial endianness** - Need to verify bit ordering in LT(j, r_cycle) matches Jolt
2. **r_address/r_cycle extraction** - Are we correctly extracting from Stage 4 challenges?
3. **Register mapping** - rd field (5-bit) vs Jolt's 128-slot register file (7-bit)

## Test Commands

### Generate Jolt-compatible Proof
```bash
cd /home/vivado/projects/zolt
./zig-out/bin/zolt prove examples/fibonacci.elf \
  --jolt-format \
  --export-preprocessing logs/zolt_preprocessing.bin \
  -o logs/zolt_proof_dory.bin
cp logs/*.bin /tmp/
```

### Verify with Jolt
```bash
cd /home/vivado/projects/zolt/jolt
cargo test --package jolt-core test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture
```

## Next Steps

1. Add detailed debug output to both Zolt prover and Jolt verifier to compare:
   - r_address and r_cycle values
   - LT polynomial evaluations at specific indices
   - Round polynomial values for RegistersValEvaluation

2. Once RegistersValEvaluation passes, implement:
   - RamRaClaimReduction (24 rounds, 3-phase)
   - LookupsReadRaf (136 rounds)

3. Then Stage 6 and 7
