# Zolt-Jolt Compatibility Implementation

## Status: Session 32 - Stage 5 InstructionReadRaf Verification Analysis

## KEY DISCOVERY - DIFFERENT RUNS PRODUCED DIFFERENT VALUES

The log files `/tmp/zolt_debug.log` (16:34) and `logs/jolt_verify.log` (13:49) are from DIFFERENT proof generations. This explains why individual InstructionRa claims appeared to mismatch - they were comparing different runs.

## CRITICAL FINDING

**Stage 5 sumcheck verification fails** at Instance 2 (InstructionReadRaf) with output_claim mismatch:
```
output_claim:   [ed, a5, f6, bf, 30, c4, 10, f8, 59, ce, db, ef, ee, 23, 2f, 96]... (LE)
expected_claim: [b2, 8f, 91, 24, 33, 0c, b4, 56, b9, 08, 89, 4c, fd, af, 54, 11]... (LE)
```

## Analysis Summary

### What's Working
1. **Polynomial coefficients match** - Rounds 0, 1, 2 sumcheck polynomials are identical
2. **InstructionRa claims serialize correctly** - When comparing same run, high 16 bytes match
3. **Serialization format is correct** - LE field element encoding matches arkworks

### What's NOT Working
1. **ra_claim product differs** between Zolt's computed product and what Jolt expects
2. The `expected_output_claim` computation uses the wrong ra_claim value

### Expected Output Claim Formula
```
expected_output_claim = eq_eval_r_reduction * ra_claim * (val_claim + gamma * raf_claim)
```

Where:
- `ra_claim = Π_{i=0}^{7} InstructionRa(i)` (product of 8 RA chunk claims)
- `val_claim = Σ_{i=0}^{41} LookupTableFlag(i) * table_i_eval`
- `raf_claim = (1 - raf_flag) * (left_op + gamma * right_op) + raf_flag * gamma * identity`

## Current State

- Fresh proof generation started at 19:53, still running
- Need to wait for it to complete, then run Jolt verifier immediately
- Compare InstructionRa claims from the SAME proof run

## Next Steps

1. [ ] Wait for fresh proof generation to complete
2. [ ] Run Jolt verifier immediately on the fresh proof
3. [ ] Compare InstructionRa claims from same run (full 32 bytes)
4. [ ] Identify which claim(s) differ and why
5. [ ] Fix the ra_chunk_weights computation if needed

## Test Commands

```bash
# Wait for fresh proof to complete, then:
cp /tmp/fresh_proof.bin logs/zolt_proof_dory.bin
cp /tmp/fresh_preprocessing.bin logs/zolt_preprocessing.bin

# Verify with Jolt
cd jolt && cargo test -p jolt-core --features zolt-debug --lib test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture 2>&1 | tee /tmp/fresh_jolt.log

# Compare ra_claims
grep "ra_chunks" /tmp/fresh_zolt.log
grep "ra_claims" /tmp/fresh_jolt.log
```

## Files Modified This Session

- `/home/vivado/projects/jolt/jolt-core/src/zkvm/instruction_lookups/read_raf_checking.rs` - Added full 32-byte debug for ra_claims
- `/home/vivado/projects/zolt/src/zkvm/spartan/stage5_prover.zig` - Added full LE debug for ra_chunks

## Background Process

Zolt proof generation running as PID 489770:
```
./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format -o /tmp/fresh_proof.bin --export-preprocessing /tmp/fresh_preprocessing.bin
```

## Session Progress

- [x] Identify Stage 5 verification failure at Instance 2
- [x] Confirmed polynomial coefficients match
- [x] Verified InstructionRa high 16 bytes match (same run)
- [x] Analyzed expected_output_claim formula
- [x] Identified logs were from different runs
- [x] Started fresh proof generation
- [ ] Complete fresh proof generation
- [ ] Run Jolt verifier on fresh proof
- [ ] Debug full 32-byte InstructionRa claims (same run)
- [ ] Fix Stage 5 opening claims

SESSION_ENDING - Waiting for fresh proof generation to complete (PID 489770)
