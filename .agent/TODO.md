# Zolt-Jolt Compatibility Implementation

## Status: IN PROGRESS - Stage 5 Sumcheck Mismatch

## Session 102 Summary

### Analysis from jolt-rust-expert

The Stage 5 verification fails because:
1. **Output claim mismatch**: Zolt produces different sumcheck output than Jolt expects
2. **Root cause identified**: The `val_claim` computation might be wrong or missing

The expected_output_claim formula in Jolt is:
```
expected_output_claim = eq(r_reduction, r_cycle_prime)
                      * ra_claim
                      * (val_claim + γ * raf_claim)
```

Where:
- `eq_eval`: eq(r_reduction, r_cycle_prime_reversed)
- `ra_claim`: ∏_{i=0}^{7} InstructionRa(i) (product of 8 chunk claims)
- `val_claim`: Σ_{i=0}^{41} table_i(r_address) * LookupTableFlag(i)
- `raf_claim`: (1-raf_flag)*(left+γ*right) + raf_flag*γ*identity

### Current Error Output

```
Sumcheck verification failed!
  output_claim:   [eb, 1c, 1a, 7c, 50, c5, 1b, 64, dd, 58, 39, 41, a8, d8, 94, 28, ...]
  expected_claim: [76, 19, 2f, 98, 45, 38, 7b, 09, b3, 3c, 7f, 8b, b0, ac, cd, b0, ...]
```

### Investigation Needed

1. **Verify sumcheck polynomial computation during address rounds**:
   - Line 1310: `proverMsgReadChecking()` - computes read-checking eval_0 and eval_2
   - Line 1314: `proverMsgRaf()` - computes RAF contribution
   - These compute the polynomial evaluations at X=0 and X=2

2. **Verify the prefix-suffix decomposition is correct**:
   - `tableCombine()` function at lines 307-407 - combines prefixes and suffixes
   - `suffixMle()` function - computes suffix MLE values
   - `prefixMle()` function - computes prefix MLE values

3. **Verify the expanding table condensation**:
   - Line 1416: `condenseUEvals()` - multiplies u_evals by expanding table
   - This was identified as missing earlier and added

### Test Commands

```bash
# Build
zig build -Doptimize=ReleaseFast

# Generate proof
./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format -o /tmp/zolt_proof_dory.bin

# Verify with Jolt
cd /home/vivado/projects/jolt
cargo test --package jolt-core --features zolt-debug test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture
```

### Key Files

- `src/zkvm/spartan/stage5_prover.zig` - Stage 5 prover
- `src/zkvm/lookup_table/prefix_suffix_prover.zig` - Prefix-suffix decomposition
- `src/zkvm/lookup_table/prefixes.zig` - Prefix MLE implementations
- `src/zkvm/lookup_table/suffixes.zig` - Suffix MLE implementations
- `/home/vivado/projects/jolt/jolt-core/src/zkvm/instruction_lookups/read_raf_checking.rs` - Jolt's Stage 5

### Previous Sessions

- Session 101: Implemented all 43 suffix MLEs, added expanding table condensation
- Session 100: Fixed prefix-suffix decomposition state management
- Session 95-99: Investigated sumcheck mismatch, identified prefix-suffix issue
