# Zolt-Jolt Compatibility - Session Progress

## Current Status

### Completed
- Stage 1 passes Jolt verification completely
- All 712 internal tests pass
- R1CS witness generation fixed
- RWC prover improvements (eq endianness, inc computation, etc.)
- Memory trace separation (instruction fetches untraced, program loading untraced)
- ELF loading fixed (correct base_address and entry_point)
- RWC SUMCHECK FIXED: Instance 2 now produces claim=0 (correct for fibonacci)
- Instance 0 (ProductVirtualRemainder): Factor evaluations match (l_inst, r_inst, fused_left, fused_right)
- Fixed claim update in batched_sumcheck.zig (properly recover c1)
- Fixed array length halving in InstructionLookups bindChallenge
- Fixed double-free in RAF prover

### In Progress - Stage 2 Instance 4 (InstructionLookupsClaimReduction)

**The Issue:**
Stage 2 batched sumcheck fails. The final `output_claim` doesn't match `expected_output_claim`.
- output_claim: 6490144552088470893406121612867210580460735058165315075507596046977766530265
- expected_output_claim: 4498967682475391509859569585405531136164526664964613766755402335917970683628

Instance 4 (InstructionLookupsClaimReduction) is producing non-zero round polynomials when its expected_output_claim is 0.

**Root Cause Analysis:**

The `expected_output_claim` for Instance 4 is:
```
eq(opening_point, r_spartan) * (lookup_output + gamma*left_op + gamma^2*right_op) = 0
```

This is 0 because `eq(opening_point, r_spartan) = 0`:
- `opening_point` = Stage 2 sumcheck challenges (normalized for Instance 4)
- `r_spartan` = Stage 1 SpartanOuter opening point for LookupOutput

The problem is Zolt's prover uses a different `r_spartan` than Jolt's verifier expects:
- Jolt r_spartan[0]: 13291784373217047195973552552972929364606691179869648085526756428005221859328
- Zolt r_spartan[0]: 19195855984845646906582597310221268809296176676350499148162414452517558935024

**Investigation Findings:**

1. Stage 1 challenges match between Zolt prover and Jolt verifier (verified byte-by-byte)
2. The r_spartan formula is: `reverse(sumcheck_challenges[1..])` (skip r_stream, then reverse for BIG_ENDIAN)
3. Zolt is applying this formula to Stage 1 prover's challenges
4. BUT Jolt's InstructionLookupsClaimReductionSumcheckParams retrieves r_spartan from opening_accumulator (set during Stage 1)
5. The values don't match - there may be additional processing or state updates

**Instance Status:**
- Instance 0 (ProductVirtual): ✓ Factor claims match
- Instance 1 (RamRafEvaluation): claim=0 (correct for fibonacci)
- Instance 2 (RWC): claim=0 (correct)
- Instance 3 (OutputCheck): claim=0 (correct)
- Instance 4 (InstructionClaimReduction): ✗ Non-zero when expected=0

## Next Steps

1. Add debug output to Jolt's `OuterLinearStage::cache_openings` to see what r_spartan it stores
2. Compare with r_spartan retrieved by `InstructionLookupsClaimReductionSumcheckParams::new`
3. Trace through the opening accumulator to find where r_spartan is set/modified
4. Alternatively: Investigate if Instance 4 can use a simpler "reduce to zero" approach

## Recent Changes

- Added `r_spartan_for_instr` parameter to `generateStage2BatchedSumcheckProof`
- Added debug output for r_spartan in Jolt's `InstructionLookupsClaimReductionSumcheckParams`
- Disabled InstructionLookupsProver (using fallback) - but this doesn't fix the issue

## Verification Commands

```bash
# Build and test Zolt
zig build test --summary all

# Generate Jolt-compatible proof
./zig-out/bin/zolt prove --jolt-format -o /tmp/zolt_proof_dory.bin examples/fibonacci.elf

# Verify with Jolt
cd /Users/matteo/projects/jolt/jolt-core
cargo test test_verify_zolt_proof -- --ignored --nocapture
```

## Code Locations
- ProductVirtual prover: `/Users/matteo/projects/zolt/src/zkvm/spartan/product_remainder.zig`
- Jolt ProductVirtual: `/Users/matteo/projects/jolt/jolt-core/src/zkvm/spartan/product.rs`
- Split Eq polynomial: `/Users/matteo/projects/zolt/src/poly/split_eq.zig`
- Proof converter: `/Users/matteo/projects/zolt/src/zkvm/proof_converter.zig`
- InstructionLookups: `/Users/matteo/projects/zolt/src/zkvm/claim_reductions/instruction_lookups.zig`
- Jolt InstructionLookups: `/Users/matteo/projects/jolt/jolt-core/src/zkvm/claim_reductions/instruction_lookups.rs`
