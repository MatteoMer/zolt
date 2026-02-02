# Zolt-Jolt Compatibility Implementation

## Status: IN PROGRESS - Stage 5 RAF/Operand Evaluation Mismatch

## Session 111 Summary

### Progress Made

1. **Fixed integer overflow issues in prefix MLE functions:**
   - Added `fieldPow2` function to handle 2^exp where exp >= 64
   - Updated `operandPrefixEvals`, `identityPrefixEvals` in prefix_suffix_prover.zig
   - Updated `signExtensionPrefixMle` and `signExtensionUpdateCheckpoint` in prefixes.zig

2. **Fixed RafDecomposition phase transition bug:**
   - Added `resetForPhase` method to restore Q_size to initial value (256)
   - Q_size was shrinking to 1 after each phase (8 rounds of binding)
   - This caused crashes at phase 9 when initQRaf tried to access Q arrays

3. **Proof generation now completes successfully:**
   - All 16 phases complete
   - All 128 address rounds + 8 cycle rounds complete
   - Proof serialization works

4. **Verified RAF bound_value computation is correct:**
   - Added debug output comparing computed operand/identity evals from challenges
   - `left_prefix`, `right_prefix`, `identity_prefix` all match computed values from challenges
   - Formula: `left = Σᵢ r[2i] · 2^(63-i)` for i=0..63

### Current Issue

**Stage 5 sumcheck output_claim doesn't match expected_claim.**

From Jolt verification:
```
output_claim:   [eb, 1c, 1a, 7c, ...]
expected_claim: [76, 19, 2f, 98, ...]
```

The expected_claim formula is:
```
expected = eq_eval * ra_claim * (val_claim + γ * raf_claim)
where raf_claim = (1-raf_flag)*(left_op + γ*right_op) + raf_flag*γ*identity
```

**Key Finding:** The operand/identity polynomial evaluations differ between Zolt and Jolt:
- Zolt's `left_prefix`: `7287e555d5caa2d03f9e811705385608`
- Jolt's `left_operand_eval`: `1bd2a26586...` (completely different)

**Root Cause Analysis:**
The operand polynomial evaluations are computed from the Stage 5 sumcheck challenges:
```
left_operand_eval = Σᵢ challenges[2i] · 2^(63-i)
```

Both Zolt and Jolt use the same formula, but the **challenges are different**.

This means the transcript states have diverged somewhere before Stage 5. The sumcheck verification passes for all 136 rounds because p(0)+p(1)=claim holds, but the underlying challenges are different because the polynomial coefficients we send don't match what Jolt expects.

### Debug Output

Zolt's first 4 challenges:
```
challenges[0] = a5d819c34e687fc91d6178237b565315
challenges[1] = db0ab34a3cbd5d94d904ca0182db69f4
challenges[2] = ba3c74f02f939d8fb9aab3d15183396d
challenges[3] = d7e1bec3e58b449a822ec1fb513857a1
```

### Next Steps

1. **Debug transcript divergence:**
   - Add more detailed transcript state logging
   - Compare transcript states at the START of Stage 5 between Zolt and Jolt
   - Identify exactly where the transcripts diverge

2. **Check polynomial coefficient output:**
   - Verify the sumcheck polynomial coefficients sent during address rounds
   - The coefficients determine the transcript state and thus the challenges

3. **Verify suffix MLE values:**
   - The address round polynomials use suffix MLE values
   - If these are wrong, the coefficients will be wrong

4. **Check table value computation:**
   - The `val_claim` contribution comes from lookup table evaluations
   - Verify the table MLE values are computed correctly

### Key Components

1. **Prefix-Suffix Decomposition** (`src/zkvm/lookup_table/`)
   - All 46 prefix types implemented in `prefixes.zig`
   - All 43 suffix types implemented in `suffixes.zig`
   - Table-specific combine functions in `prefix_suffix_prover.zig`
   - RAF decomposition for left/right/identity operands

2. **Stage 5 Prover** (`src/zkvm/spartan/stage5_prover.zig`)
   - Three-instance batched sumcheck
   - Instance 0: RegistersValEvaluation (8 rounds)
   - Instance 1: RamRaClaimReduction (24 rounds) - simplified for fibonacci
   - Instance 2: LookupsReadRaf (136 rounds) with prefix-suffix decomposition
   - Phase transitions every 8 rounds (16 phases total)
   - Expanding table condensation working

### Test Commands
```bash
# Build
zig build -Doptimize=ReleaseFast

# Generate proof
timeout 600 ./zig-out/bin/zolt prove examples/fibonacci.elf --jolt-format -o logs/zolt_proof_dory.bin

# Verify with Jolt
cd /home/vivado/projects/jolt
cargo test --package jolt-core --features zolt-debug test_verify_zolt_proof_with_zolt_preprocessing -- --ignored --nocapture
```
