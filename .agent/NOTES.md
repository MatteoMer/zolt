# Zolt-Jolt Cross-Verification Progress

## Session 37 Summary - Stage 2 RWC Debugging (2026-01-17)

### Current Status
- **Stage 1: PASSES** ✓
- **Stage 2: FAILS** - output_claim ≠ expected_output_claim

### Key Discovery: Fibonacci Has 1 Memory Operation!
- Entry at cycle=54, addr=2049 (RAM address 0x80004008)
- This is likely for output/termination mechanism
- The RWC sumcheck is NOT a zero polynomial!

### Stage 2 Failure Details
```
output_claim:          8531846635557760858493086388539389736059592909629786934607653033526197973299
expected_output_claim: 17808130426384425926507399004264546912094784764713076233980989102782648691939
```

### Stage 2 Batched Sumcheck (5 Instances)
1. ProductVirtualRemainder - 8 rounds, input_claim = uni_skip_claim
2. RamRafEvaluation - 16 rounds, input_claim = 0 (RAM address at SpartanOuter = 0)
3. RamReadWriteChecking - 24 rounds, input_claim = 0 (RamReadValue + gamma*RamWriteValue = 0)
4. OutputSumcheck - 16 rounds, input_claim = 0
5. InstructionLookupsClaimReduction - 8 rounds, input_claim = non-zero

### RWC Instance Analysis
The RWC has 1 entry: cycle=54, addr=2049

Expected output claim formula (from Jolt):
```rust
eq_eval_cycle * ra_claim * (val_claim + gamma * (val_claim + inc_claim))
```

Zolt's opening claims after Stage 2:
- ra_claim (non-zero): evaluates to eq(r_addr, 2049) * eq(r_cycle, 54)
- val_claim (non-zero): MLE of Val at (r_addr, r_cycle)
- inc_claim (non-zero): MLE of Inc at r_cycle

### Potential Issues
1. The round polynomials may not exactly match Jolt's formula
2. Opening claims might be computed at wrong point
3. eq polynomial endianness differences
4. gamma_rwc synchronization between Zolt and Jolt

### Debug Output Added
```
[ZOLT] STAGE2 RWC: entries.len = 1
[ZOLT] STAGE2 RWC: entry[0]: cycle=54, addr=2049
[ZOLT] STAGE2 RWC: ra_claim = non-zero
[ZOLT] STAGE2 RWC: val_claim = non-zero
[ZOLT] STAGE2 RWC: inc_claim = non-zero
```

### Next Steps
1. Compare gamma_rwc between Zolt and Jolt
2. Verify eq_eval_cycle computation
3. Check if round polynomial matches expected formula
4. Debug the RWC prover round-by-round

---

## Previous Sessions Summary

### Stage 4 Issue (from Session 36)
Stage 4 was failing with val_final mismatch - now deferred until Stage 2 is fixed.

### Stage 3 Fix (from Session 35)
- Fixed prefix-suffix decomposition convention (r_hi/r_lo)
- Stage 3 now passes

### Stage 1 Fix
- Fixed NextPC = 0 issue for NoOp padding
- Stage 1 passes

---

## Technical References

- Jolt RamReadWriteChecking: `jolt-core/src/zkvm/ram/read_write_checking.rs`
- Jolt BatchedSumcheck: `jolt-core/src/subprotocols/sumcheck.rs`
- Zolt RWC Prover: `src/zkvm/ram/read_write_checking.zig`
- Zolt Stage 2: `src/zkvm/proof_converter.zig` (generateStage2BatchedSumcheck)
