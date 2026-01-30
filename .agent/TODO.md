# Zolt-Jolt Compatibility Implementation

## Status: IN PROGRESS - Stage 5 Requires LookupsReadRaf Prover

## Verified Stages
- Stage 1: PASSED ✅
- Stage 2: PASSED ✅
- Stage 3: PASSED ✅
- Stage 4: PASSED ✅
- Stage 5: PARTIAL - RegistersValEvaluation WORKS, LookupsReadRaf NOT IMPLEMENTED
- Stage 6: Not tested yet
- Stage 7: Not tested yet

## Current Issue: Stage 5 Batched Sumcheck Missing Instance 2

The Stage 5 batched sumcheck has 3 instances:
1. **RegistersValEvaluation** (8 rounds) - IMPLEMENTED ✅
2. **RamRaClaimReduction** (24 rounds) - Sends zero (OK for programs without RAM ops)
3. **LookupsReadRaf** (136 rounds) - NOT IMPLEMENTED ❌

### Why It Fails

The verification fails because:
1. `lookups_input` is non-zero (Fibonacci has instruction lookups like ADD, ADDI)
2. Instance 2 has 136 rounds = max_rounds, so there's no constant phase
3. We send zero polynomials for Instance 2
4. The batched sumcheck verifier expects `p(0)+p(1) = batched_claim` at each round
5. But `batched_claim` includes `batch2 * lookups_input` which our zero polynomial doesn't match

### Debug Output
```
[STAGE5 DEBUG] Instance 2 missing contribution:
  lookups_input = { 35, 47, 148, 88, ... } (non-zero!)
  batch2 * lookups_input = { 1, 179, 75, 232, ... }
  batched_claim - p(0)+p(1) = { 1, 179, 75, 232, ... }  (same!)
```

### What Works
1. ✅ RegistersValEvaluation polynomial computation (inc * wa * lt)
2. ✅ Sum check: `Σ_j inc(j) * wa(j) * lt(j) = regs_val_input`
3. ✅ LT polynomial evaluation matches verifier
4. ✅ Final product `inc*wa*lt` matches expected_product

### What Needs Implementation: LookupsReadRaf

The LookupsReadRaf sumcheck proves:
```
rv(r_reduction) + γ·left_op(r_reduction) + γ²·right_op(r_reduction)
  = Σ_j Σ_k [ eq(j; r_reduction) · ra(k, j) · (Val_j(k) + γ · RafVal_j(k)) ]
```

This requires:
1. Prefix/suffix decomposition for 128-bit address space
2. Lookup table evaluations
3. RAF (Read-And-Flag) polynomial handling
4. Complex batching with γ

### Options for Progress

1. **Skip Stage 5 for now**: Mark verification as partial pass for stages 1-4
2. **Test with programs without lookups**: Create a minimal program that has `lookups_input = 0`
3. **Implement LookupsReadRaf**: Major undertaking (~1000 lines of code)

### Recommended Next Steps
1. Document current progress (stages 1-4 pass, stage 5 partial)
2. Consider implementing LookupsReadRaf if full verification is needed
3. Or focus on other improvements (Stage 6, 7, commitment schemes)

## Test Commands

```bash
# Generate proof
./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format -o /tmp/zolt_proof_dory.bin

# Verify with Jolt
cargo test --package jolt-core test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture
```

## Key Files
- `src/zkvm/spartan/stage5_prover.zig` - Zolt Stage 5 prover
- `src/zkvm/proof_converter.zig` - Proof generation and conversion
- `jolt-core/src/zkvm/registers/val_evaluation.rs` - Jolt Stage 5 verifier
- `jolt-core/src/zkvm/instruction_lookups/read_raf_checking.rs` - LookupsReadRaf reference
- `jolt-core/src/subprotocols/sumcheck.rs` - Jolt batched sumcheck verifier
