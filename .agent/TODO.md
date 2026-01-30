# Zolt-Jolt Compatibility Implementation

## Status: IN PROGRESS - Stage 5 Sumcheck Implementation

## Verified Stages
- Stage 1: PASSED ✅
- Stage 2: PASSED ✅
- Stage 3: PASSED ✅
- Stage 4: PASSED ✅
- Stage 5: FAILED ❌ (current focus)
- Stage 6: Not tested yet
- Stage 7: Not tested yet

## Stage 5 Root Cause Analysis (2026-01-30)

### Problem Identified
The Stage 5 batched sumcheck fails because:
1. `output_claim = 17054937...` (from our sumcheck polynomials)
2. `expected_output_claim = 18413512...` (from opening claims)
3. Difference: ~1.3e75 (significant)

### Why It Fails
- Instance 0 (RegistersValEvaluation) has non-zero expected_claim = 1225620...
- Instance 1 (RamRaClaimReduction) has expected_claim = 0
- Instance 2 (LookupsReadRaf) has expected_claim = 0

The verifier computes:
```
expected_output_claim = inc_claim * wa_claim * lt_eval
```

Where `inc_claim` and `wa_claim` are read from our serialized opening claims.

### Current Implementation Issues
1. **RegistersValEvaluation** (8 rounds):
   - Sumcheck polynomials are partially implemented
   - But we're using zero polynomials for RamRa/Lookups which are active the whole time
   - This affects transcript state divergence

2. **RamRaClaimReduction** (24 rounds):
   - Uses zero polynomials when active ❌
   - Input claim is non-zero but expected_claim is 0 (correct for fibonacci)

3. **LookupsReadRaf** (136 rounds):
   - Uses zero polynomials when active ❌
   - Input claim is non-zero but expected_claim is 0 (correct for fibonacci)

### The Key Insight
For fibonacci, RamRa and Lookups produce `expected_claim = 0` because:
- The opening claims we provide for these are all zeros
- `expected_output_claim = Σ (claim_i * prod_of_opening_factors) = 0`

So **the polynomial sum should reduce to 0** but our zero polynomials don't correctly participate in the batched sumcheck:
- When inactive: must produce `p(0) + p(1) = scaled_input_claim`
- When active: must produce actual polynomial that sums to 0

### Solution Path
1. Keep using zero polynomials for RamRa and Lookups when active (since their contribution is 0)
2. BUT ensure the inactive-to-active transition is correct
3. The main issue is RegistersValEvaluation - need to correctly compute:
   - Round polynomials that reduce input_claim to expected_output_claim
   - Final opening claims `inc_claim` and `wa_claim` that make verifier happy

### Batched Sumcheck Math
For max_rounds = 136:
- Rounds 0-111: Only LookupsReadRaf active (128 rounds)
- Rounds 112-127: LookupsReadRaf + RamRaClaimReduction active (16+8 rounds)
- Rounds 128-135: All three active (8 rounds each)

Initial batched claim:
```
batched_claim = batch0 * 2^128 * regs_input + batch1 * 2^112 * ram_ra_input + batch2 * lookups_input
```

## Next Steps (Priority Order)

1. **Verify Stage 5 is using trace-aware prover**
   - Check that config.execution_trace is not null
   - Add debug output to confirm path taken

2. **Fix RegistersValEvaluation sumcheck**
   - Compute LT polynomial correctly
   - Ensure inc_evals and wa_evals are populated from trace
   - Verify round polynomial p(0) + p(1) = current_claim

3. **Verify RamRa and Lookups zero polynomials**
   - Since expected_claim = 0, zero polynomials should work
   - But must ensure transcript appends match Jolt

## Test Commands

```bash
# Generate proof
cd /home/vivado/projects/zolt
./zig-out/bin/zolt prove examples/fibonacci.elf \
  --jolt-format \
  --export-preprocessing logs/zolt_preprocessing.bin \
  -o logs/zolt_proof_dory.bin
cp logs/*.bin /tmp/

# Verify with Jolt
cd /home/vivado/projects/zolt/jolt
cargo test --package jolt-core test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture
```

## Key Files
- `src/zkvm/spartan/stage5_prover.zig` - Stage 5 batched sumcheck prover
- `src/zkvm/proof_converter.zig` - Main proof generation orchestration
- `jolt-core/src/subprotocols/sumcheck.rs` - Jolt's batched sumcheck reference
