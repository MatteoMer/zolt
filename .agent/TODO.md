# Zolt-Jolt Compatibility Implementation

## Status: Session 19 - Stage 5 Sumcheck Debug (SESSION_ENDING)

## Current Progress

### Completed
1. ✅ Fixed SumcheckId mismatch (22 variants, not 24)
2. ✅ Fixed proof serialization format:
   - Added 4 advice proof option bytes
   - Fixed config to use 5 usize values (trace_length, ram_K, bytecode_K, log_k_chunk, lookups_ra_virtual_log_k_chunk)
   - Removed rw_config and dory_layout
3. ✅ Fixed proof deserialization (uses compressed format)
4. ✅ Proof now loads and verification runs!

### Current Issue: Stage 5 Sumcheck

Stage 5 verification fails with output_claim != expected_claim:
```
output_claim:          4685578738422804254959705568336941759457054412197980855802774129638288827885
expected_output_claim: 12188497878882392561462773456725953651449308138029928163257319185678381191090
```

From debug output:
- Instance 2 expected_claim = 5228641186838112443815018829560347135294696928443125671497986510902740364546
- Instance 2 coeff = 65436712948050808574557213493451640901
- Instance 2 weighted = 5388269998998530687089403994657448571284391738532206803112685224477058414078

The expected_output_claim is: `sum(instance_i.expected_claim * batch_coeff_i)`

The output_claim comes from sumcheck polynomial evaluation at the challenges.

## Stage 5 Architecture

Stage 5 is a batched sumcheck with 3 instances:
1. **Instance 0 (RegistersValEvaluation)**: 8 rounds (cycle only), degree-3
   - Uses: RdInc, RdWa claims
   - r_cycle from RegistersReadWriteChecking

2. **Instance 1 (RamRaClaimReduction)**: 24 rounds (16 address + 8 cycle), degree-2
   - Uses: RamRa(i) claims for all chunks
   - r_cycle from RamRaClaimReduction

3. **Instance 2 (LookupsReadRaf)**: 136 rounds (128 address + 8 cycle), degree-10 during cycle
   - Uses: InstructionRa(i), LookupTableFlag(i), InstructionRafFlag claims
   - r_reduction from InstructionClaimReduction

## Likely Root Causes

1. **Opening claims mismatch**: The values computed by Zolt for the opening claims might not match what Jolt expects
2. **Challenge ordering**: The r_reduction/r_cycle points might be in wrong endianness
3. **Polynomial computation**: The cycle round polynomial might have incorrect eq_prefix or ra_chunk computation

## Next Session Tasks

1. Add debug output comparing:
   - Zolt's Instance 2 expected_claim computation
   - Jolt's Instance 2 expected_claim from opening claims

2. Verify opening claims match:
   - InstructionRa(i) for i in 0..8 at the correct evaluation points
   - LookupTableFlag(i) for all tables
   - InstructionRafFlag

3. Check r_reduction ordering - is it BigEndian or LowToHigh?

## Test Commands

```bash
# Generate proof
./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format --export-preprocessing logs/zolt_preprocessing.bin -o logs/zolt_proof_dory.bin

# Cross-verify
cp logs/zolt_proof_dory.bin /tmp/ && cp logs/zolt_preprocessing.bin /tmp/
cd jolt && cargo test -p jolt-core --lib test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture
```

## Session History

- Session 1-8: Initial implementation, transcript ordering
- Session 9: MontU128Challenge multiplication fix
- Session 10-11: Cross-verification debugging
- Session 12: Verified r_address_prime challenges match
- Session 13: Fixed suffix_len overflow
- Session 14: Internal verification passes
- Session 15: Confirmed opening claims match
- Session 16: Fixed LowerWord/UpperWord/LowerHalfWord suffix MLEs
- Session 17: Verified all opening claims match
- Session 18: Discovered potential degree mismatch
- Session 19: **Fixed serialization, proof loads, Stage 5 sumcheck fails**
