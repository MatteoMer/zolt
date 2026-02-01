# Zolt-Jolt Compatibility Implementation

## Status: IN PROGRESS - Stage 5 Sumcheck Debugging

## Session 106 Summary (continued)

### Completed
1. **Fixed shift overflow in rightShiftPrefixMle** (committed)
   - When `y_u32 == 0`, `@ctz()` returns 32 which causes overflow in `x_u32 >> 32`
   - Added bounds check to return 0 when shift amount >= 32
   - Also fixed in `rightShiftWPrefixMle`

2. **Analysis Complete**
   - Analyzed Jolt's prefix-suffix decomposition algorithm
   - Verified bit ordering (HighToLow binding, MSB first)
   - Verified suffix_len formula matches Jolt
   - Verified fromEvalsAndHint implementation matches Jolt

### Current Issue
Stage 5 (InstructionReadRaf) sumcheck verification fails:
- output_claim (from Zolt): `[eb, 1c, 1a, 7c, 50, c5, 1b, 64, ...]`
- expected_claim (from Jolt): `[76, 19, 2f, 98, 45, 38, 7b, 09, ...]`

### Code Paths Verified
1. `proverMsgReadChecking` - computes eval_0 and eval_2 for read-checking
2. `proverMsgRaf` - computes eval_0 and eval_2 for RAF contribution
3. `tableCombine` - combines prefix and suffix for each table (41+ cases)
4. `fromEvalsAndHint` - interpolates degree-2 polynomial from eval_0, eval_2
5. `initPhase` - initializes Q (suffix) polynomials for each phase

### Possible Root Causes
1. **Prefix checkpoint computation** - Are checkpoints being updated correctly?
2. **Suffix MLE computation** - Are suffix MLE values correct for each table?
3. **Q polynomial initialization** - Are cycles being bucketed correctly?
4. **RAF operand computation** - Is identity/left/right contribution computed correctly?
5. **Phase transition** - Is condenseUEvals working correctly between phases?

### Next Debugging Steps
1. Add detailed logging to compare first round polynomial values with Jolt
2. Print Q[0][b] and Q[1][b] values at start of phase 0
3. Print prefix checkpoint values after each update
4. Verify cycle_table_indices mapping is correct

### Test Commands
```bash
# Build
zig build -Doptimize=ReleaseFast

# Generate proof
./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format -o /tmp/zolt_proof.bin

# Verify with Jolt
cd /home/vivado/projects/jolt
cargo test --package jolt-core --features zolt-debug test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture
```

### Key Files
- `src/zkvm/spartan/stage5_prover.zig` - Main Stage 5 prover
- `src/zkvm/lookup_table/prefix_suffix_prover.zig` - Prefix-suffix decomposition
- `src/zkvm/lookup_table/prefixes.zig` - All 40+ prefix implementations
- `src/zkvm/lookup_table/suffixes.zig` - Suffix MLE implementations

### Jolt Formula Recap
Stage 5 proves:
```
Σ_j Σ_k eq(j; r_reduction) · ra(k,j) · (Val_j(k) + γ·RafVal_j(k)) = input_claim
```

Where:
- Val_j(k) = table.evaluate_mle(k) for the lookup table used at cycle j
- RafVal_j(k) = (1-flag)*(left_op + γ*right_op) + flag*γ*identity

Expected output at end:
```
expected = eq(r_cycle', r_reduction) * ra(r_address) * (val_claim + γ*raf_claim)
```
