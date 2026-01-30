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

## Stage 5 Analysis (2026-01-30)

### Structure
Stage 5 is a batched sumcheck with 3 instances:
1. **RegistersValEvaluation** (8 rounds): Proves Val(r) = Σ_j inc(j)·wa(r_addr,j)·LT(r_cycle,j)
2. **RamRaClaimReduction** (24 rounds): Batches 4 RA claims using gamma
3. **LookupsReadRaf** (136 rounds): Instruction lookup verification

### Problem
All three instances have **NON-ZERO input claims**:
- RegistersValEvaluation: 20196670024706610341728276844931391924934592974175535367959454787282160553899
- RamRaClaimReduction: 16410442144988038954986615472772880745324464916492580913716405392685466979654
- LookupsReadRaf: 9299828901037110504125985581408576613022125108259561907120516744221579828954

### Current Implementation Status
- RegistersValEvaluation: Partially implemented with trace data (may have bugs)
- RamRaClaimReduction: Uses zero polynomials when active ❌
- LookupsReadRaf: Uses zero polynomials when active ❌

### Why Zero Polynomials Fail
When an instance becomes active with non-zero current_claim:
- Sumcheck requires p(0) + p(1) = current_claim
- Zero polynomial gives p(0) + p(1) = 0 ≠ current_claim
- Verification fails

### Fix Applied
Fixed inactive instance computation from `p(x) = claim` to `p(x) = claim/2` so that p(0) + p(1) = claim (the sumcheck invariant).

## Next Steps (Priority Order)

1. **Debug RegistersValEvaluation**
   - Add detailed debug output comparing Zolt prover vs Jolt verifier
   - Check r_address/r_cycle extraction from Stage 4
   - Verify LT polynomial bit ordering

2. **Implement RamRaClaimReduction**
   - 3-phase prover: PhaseAddress → PhaseCycle1 → PhaseCycle2
   - Reference: jolt-core/src/zkvm/claim_reductions/ram_ra.rs
   - Batches 4 RA claims using gamma

3. **Implement LookupsReadRaf**
   - 136 rounds (128 address + 8 cycle)
   - Reference: jolt-core/src/zkvm/instruction_lookups/read_raf_checking.rs
   - Uses prefix-suffix decomposition

4. **Implement Stage 6 and 7 provers**

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
