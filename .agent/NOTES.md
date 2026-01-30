# Zolt-Jolt Cross-Verification Progress

## Session 84 - Stage 5 Debugging (2026-01-30)

### Key Finding
Stage 5 RegistersValEvaluation fails because:
```
computed_sum = Σ_j inc(j) * wa(j) * LT(j, r_cycle)
regs_val_input = RegistersVal @ RegistersReadWriteChecking (from Stage 4)
computed_sum ≠ regs_val_input
```

### Debug Output
```
[STAGE5] Sum check: computed_sum = { 11, 81, 53, ... }
[STAGE5] Sum check: regs_val_input = { 44, 166, 232, ... }
[STAGE5] Sum check: match = false
```

### Analysis
The prover computes `inc * wa * LT` over all trace cycles and sums them.
But this sum doesn't equal the input claim from Stage 4.

Possible causes:
1. **r_cycle is wrong** - The r_cycle from Stage 4 may not match what the verifier expects
2. **r_address is wrong** - Similarly for r_address used in wa computation
3. **inc computation is wrong** - rd_inc values from trace may be wrong
4. **wa computation is wrong** - eq(r_address, rd) computation may be wrong
5. **LT computation is wrong** - LT(j, r_cycle) computation may have bit order issues

### Stage 4 Architecture
Stage 4 is a batched sumcheck with:
- RegistersRWC: 15 rounds (7 address + 8 cycle, 3-phase Gruen)
- ValEvaluation: 8 rounds (cycle only)
- ValFinal: 8 rounds (cycle only)
- max_rounds = 15

Phase structure for RegistersRWC:
- Phase 1 (rounds 0-7): Bind cycle vars (8 rounds)
- Phase 2 (rounds 8-14): Bind address vars (7 rounds)
- Phase 3: None (0 rounds)

So:
- `r_cycle = reverse(challenges[0..8])` → BIG_ENDIAN
- `r_address = reverse(challenges[8..15])` → BIG_ENDIAN

### Next Steps
1. Add debug to print first few inc/wa/lt evaluations
2. Compare against what Jolt prover computes
3. Verify r_cycle and r_address match between prover and verifier
4. Check bit ordering in LT and eq computations

### Key Files
- `src/zkvm/spartan/stage5_prover.zig` - Stage 5 batched sumcheck prover
- `src/zkvm/proof_converter.zig:2663-2686` - r_cycle/r_address extraction
- `jolt-core/src/zkvm/registers/val_evaluation.rs` - Jolt's prover reference

---

## Session 83 Summary
- Identified Stage 5 failure: output_claim ≠ expected_output_claim
- Expected claims: Instance 0 = non-zero, Instance 1 = 0, Instance 2 = 0
- The issue is in RegistersValEvaluation polynomial computation
