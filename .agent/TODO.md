# Zolt-Jolt Compatibility TODO

## Current Progress

### Stages 1-3: PASS
### Stage 4: ROOT CAUSE IDENTIFIED

## Stage 4 Root Cause: eq Polynomial Endianness Mismatch

**FINDING**: The polynomial opening claims all match perfectly, but the eq polynomial evaluation differs!

- Zolt eq_claim: `{ 205, 98, 173, 194, ... }`
- Jolt eq_eval: `{ 27, 82, 75, 12, ... }`

### The Issue

Jolt stores `r_cycle` (from Stage 3) in **BIG_ENDIAN** format:
- Stage 3 sumcheck produces challenges `[r0, r1, r2, ...]` where `r0` binds LSB (little-endian/binding order)
- Jolt's `normalize_opening_point` **REVERSES** them to `[..., r2, r1, r0]` for BIG_ENDIAN
- Result: `r_cycle[0]` = MSB, `r_cycle[n-1]` = LSB

Zolt passes Stage 3 challenges **directly without reversing**:
```zig
// WRONG: Using challenges in binding order (LITTLE_ENDIAN)
stage3_result.challenges  // [r0, r1, ...] where r0=LSB
```

### The Fix

Reverse the Stage 3 challenges before passing to Stage 4:

```zig
// Reverse challenges from Stage 3 to convert from LITTLE_ENDIAN to BIG_ENDIAN
const r_cycle = try self.allocator.alloc(F, stage3_result.challenges.len);
for (0..stage3_result.challenges.len) |i| {
    r_cycle[i] = stage3_result.challenges[stage3_result.challenges.len - 1 - i];
}
```

### Location to Fix

File: `src/zkvm/proof_converter.zig` lines 1575-1586

Change from:
```zig
var stage4_prover = Stage4ProverType.initWithClaims(
    ...
    stage3_result.challenges,  // WRONG: little-endian order
    ...
)
```

To:
```zig
var stage4_prover = Stage4ProverType.initWithClaims(
    ...
    reversed_challenges,  // CORRECT: big-endian order
    ...
)
```

## Previous Fixes Applied

1. **Sumcheck polynomial computation** - Now uses pointwise multiplication of univariate restriction evals
2. **Challenge ordering** - Stage 3 challenges passed correctly for claim computation
3. **Register values** - Rs1Value/Rs2Value set to 0 for instructions that don't read them
4. **K value** - Confirmed K=128 (32 RISC-V + 96 virtual registers)

## Testing
```bash
bash scripts/build_verify.sh
```
