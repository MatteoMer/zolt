# Zolt-Jolt Compatibility TODO

## Current Progress

### Stages 1-3: PASS
### Stage 4: IN PROGRESS - RWC val_claim Bug Found

## Stage 4 Investigation (Updated 2026-01-12)

### VERIFIED CORRECT

1. **normalize_opening_point Works Correctly** - r_cycle is correctly reconstructed
2. **gamma matches between Zolt and Jolt** - Value: 250464584498748727615350532275610162451
3. **Round polynomial coefficients (c0, c2) match** - Verified byte-by-byte
4. **Instance 0 input_claim (RegistersReadWriteChecking) matches**
5. **Transcript protocol is correct** - No advice appends for fibonacci

### ROOT CAUSE FOUND: RWC val_claim is Zero

**The Bug:**
Zolt's Stage 2 RWC (Ram Read-Write Checking) prover returns ALL ZEROS for `val_claim`:
```
[ZOLT] STAGE2 RWC: val_claim = { 0, 0, 0, 0, ... }
```

But Jolt expects a **non-zero** val_claim even for fibonacci with no RAM operations!

**Why val_claim Should Be Non-Zero:**

1. The `val` polynomial represents the memory state matrix `Val(k, j)` = value at address k before cycle j
2. Even with NO RAM operations, the `val` polynomial is initialized from `val_init` (initial RAM state)
3. Initial RAM contains non-zero values: bytecode words, program inputs
4. When evaluated at a random point `(r_address, r_cycle)`, this gives a non-zero result

**Jolt's Expected Values:**
- Instance 1 (RamValEvaluation) input_claim: 1063618430616024228610198021464576543949054712450992483279253898727921684198
- This is `val_claim - init_eval` where val_claim = evaluation of initial RAM at random point

**The Fix Needed:**

Zolt's RWC prover needs to:
1. Build the `val` polynomial from `val_init` (initial RAM state)
2. For fibonacci (no RAM ops): `val[k, j] = val_init[k]` for all k, j
3. After sumcheck binding, evaluate `val` at the final random point
4. Return this non-zero evaluation as `val_claim`

### Transcript Divergence Chain

1. gamma = challenge ✓ (matches)
2. append(input_claim_registers) ✓ (matches)
3. append(input_claim_val_eval) ✗ (MISMATCH - val_claim is wrong)
4. append(input_claim_val_final) ✗ (MISMATCH - downstream from wrong claims)
5. All subsequent batching coefficients and challenges diverge

### Files to Fix

**Zolt RWC Prover:**
- The RWC prover needs to properly compute val_claim from val_init
- Location: wherever `getOpeningClaims()` returns val_claim

**Reference Jolt Files:**
- `/jolt-core/src/zkvm/ram/read_write_checking.rs:616-645` - cache_openings sets RamVal
- `/jolt-core/src/subprotocols/read_write_matrix/ram.rs:229-277` - val polynomial construction
- `/jolt-core/src/zkvm/ram/val_evaluation.rs:160-166` - input_claim computation

## Next Steps

1. **Find Zolt's RWC prover** - locate where val_claim is computed
2. **Build val polynomial from val_init** - replicate initial RAM across all cycles
3. **Evaluate val at final sumcheck point** - this gives the non-zero val_claim
4. **Verify input_claim_val_eval matches Jolt**

## Testing
```bash
bash scripts/build_verify.sh  # Output goes to logs/ directory
```
