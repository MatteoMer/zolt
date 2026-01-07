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
- **Instance 4 (InstructionLookupsClaimReduction) endianness FIXED**

### In Progress - Stage 2 Batched Sumcheck Output Claim Mismatch

**The Issue:**
All individual components match between Zolt and Jolt:
- fused_left matches
- fused_right matches
- split_eq.current_scalar matches (L * Eq)
- Challenges match exactly

But the batched sumcheck output_claim diverges:
```
Expected:     19828484771497821494602704840470477639244539279836761038780805731500438199328
Zolt output:  5584134810285329217002595006333176637104372627852824503579688439906349437652
```

**Root Cause Investigation:**
The issue is in the sumcheck claim evolution, not the final polynomial values.
The round polynomial computation or claim update formula has a bug.

**Instance Status:**
- Instance 0 (ProductVirtual): Components match, claim evolution wrong
- Instance 1 (RegistersVal): claim=0 (inactive for fibonacci)
- Instance 2 (RWC): claim=0 (correct)
- Instance 3 (OutputCheck): claim=0
- Instance 4 (InstructionClaimReduction): claim=0 (FIXED endianness)

## Next Steps

1. Add per-round debug output comparing Zolt vs Jolt polynomial values
2. Verify the compressed -> evals -> Lagrange eval flow is correct
3. Check if round polynomial coefficients match between Zolt and Jolt
4. Investigate claim update: `new_claim = s(challenge)` computation

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
