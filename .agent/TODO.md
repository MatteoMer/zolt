# Zolt-Jolt Compatibility: Current Status

## Status: Stage 4 Fails (Stages 1-3 PASS!) 🟡

## Session 79 Summary (2026-01-29)

### ROOT CAUSE FOUND: Stage 4 input_claim mismatch

**The Issue:**
Stage 4 sumcheck fails because `input_claim_val_eval` doesn't match the prover's actual polynomial sum.

**Detailed Analysis:**

1. **Stage 4 sumcheck consistency check fails at Round 7**
   - `p(0) + p(1) != batched_claim` starting from Round 7
   - Round 7 is when ValEvaluation and ValFinal instances start participating
   - Rounds 0-6 pass the check

2. **Root cause trace:**
   ```
   input_claim_val_eval = rwc_val_claim - init_eval_for_val_eval
                       = 0 - init_eval
                       = -init_eval (large field element)
   ```

   But the prover computes:
   ```
   Σ_j inc[j] * wa[j] * lt[j] = non-zero positive value
   ```

3. **Why rwc_val_claim = 0:**
   - `rwc_prover` is NULL because `input_claims[2] = 0`
   - `input_claims[2] = ram_read_value_claim + gamma * ram_write_value_claim`
   - Both `ram_read_value_claim` and `ram_write_value_claim` are **ZERO**
   - These come from Stage 1 factor evaluations (indices 13 and 14)

4. **But Fibonacci DOES have a RAM write:**
   - At cycle 54, writes result `1` to address `0x7FFFC008` (public output)
   - The `IncPolynomial` correctly sees this: `inc=1, old_val=0, new_val=1`
   - But Stage 1's R1CS factors don't include it

5. **Key debug output:**
   ```
   [VALEVAL_INIT] Write at idx=2049, timestamp=54, old_val=0, new_val=1, inc=1
   [ZOLT] OPENING_CLAIMS: claim[13] = { 0, 0, 0, 0, ...}  // RamReadValue = 0
   [ZOLT] OPENING_CLAIMS: claim[14] = { 0, 0, 0, 0, ...}  // RamWriteValue = 0
   [ZOLT STAGE2 RWC: rwc_prover is_null = true
   [ZOLT STAGE4] rwc_val_claim_BE = { 0, 0, 0, 0, ...}
   ```

### Next Steps (in priority order)

1. **[ ] Fix Stage 1 RamReadValue/RamWriteValue extraction**
   - Check `generateWitness()` in R1CS constraint generator
   - Factor indices 13 (RamReadValue) and 14 (RamWriteValue)
   - These should contain non-zero values when RAM operations occur

2. **[ ] Verify how Jolt computes these factors**
   - Check Jolt's `spartan_product.rs`
   - Understand what RamReadValue and RamWriteValue represent

3. **[ ] Run verification after fix**

### Files to Investigate:

- `/home/vivado/projects/zolt/src/zkvm/r1cs/constraint_gen.zig` - R1CS constraint generator
- `/home/vivado/projects/zolt/src/zkvm/proof_converter.zig:700-800` - Factor evaluation mapping
- Jolt's `spartan_product.rs` - How RAM factors are evaluated

### How to Run Tests

```bash
# Zolt proof generation
cd /home/vivado/projects/zolt
zig build -Doptimize=ReleaseSafe && ./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format --export-preprocessing logs/zolt_preprocessing.bin -o logs/zolt_proof_dory.bin

# Jolt verification test
cd /home/vivado/projects/jolt
ZOLT_LOGS_DIR=/home/vivado/projects/zolt/logs cargo test --features "minimal,zolt-debug" --no-default-features -p jolt-core test_verify_zolt_proof_with_zolt_preprocessing -- --nocapture --ignored
```

---

## Previous Sessions

### Session 78 (2026-01-29)
- Fixed Stage 2 issue by skipping prover initialization when input_claim is zero
- Stages 1-3 now pass
- Stage 4 fails with r_cycle mismatch (now understood to be input_claim mismatch)

### Session 77
- Fixed config serialization format
- Stage 1 passes
- Stage 2 fails: output_claim != expected_claim

### Session 76
- Fixed ZOLT header issue using `--jolt-format`
