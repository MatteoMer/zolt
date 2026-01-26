# Session 63 Verification: What Were The Actual Values?

## The Question
Were Instances 1 and 2 ACTUALLY working before Session 63 changes?

## Findings at Commit 64f041e (Session 61)

**From TODO.md at that commit:**
```
But Zolt computes:
- Instance 0: 10960129572097163177603722996998750391162218193231933792862644774061483523224 ❌
- Instance 1: 16843726827876190648710859579071819473340754364270059307512129120184648645607 ❌ (should be 0)
- Instance 2: 5258723638175825215483753966464390100826417414032932059867770167991589922285 ❌ (should be 0)
```

**Conclusion**: At commit 64f041e, BOTH Instance 1 AND 2 were NON-ZERO (broken)!

## Current Status (After Session 63)

**From Zolt prover output:**
```
[ZOLT STAGE4] input_claim_val_eval_BE = { 36, 195, 249, 110, ... } (NON-ZERO)
[ZOLT STAGE4] input_claim_val_final_BE = { 0, 0, 0, 0, ... } (ZERO)

[ZOLT STAGE4] Instance 0 (RegistersReadWriteChecking): { 23, 212, 69, 170, ... } (NON-ZERO - expected)
[ZOLT STAGE4] Instance 1 (RamValEvaluation): { 11, 250, 253, 152, ... } (NON-ZERO - BROKEN!)
[ZOLT STAGE4] Instance 2 (RamValFinalEvaluation): { 0, 0, 0, 0, ... } (ZERO - WORKING!)
```

**Conclusion**:
- Instance 1: STILL BROKEN ❌
- Instance 2: FIXED ✅

## What About Jolt's Values?

Checked Jolt test output but couldn't find explicit Stage 4 instance input_claim values printed.

**From Jolt's earlier Stage 1 batching (NOT Stage 4):**
```
[JOLT BATCHED] Instance 1 expected_claim = 0
[JOLT BATCHED] Instance 2 expected_claim = 0
```

These are from Stage 1, not Stage 4. Need to find where Jolt prints Stage 4 instance values from its PROVER.

## CRITICAL INSIGHT

**The TODO was WRONG about the state before Session 63!**

The TODO said "Instance 1 and 2 working" but the actual commit history shows:
- At 64f041e (Session 61): Instance 1 = NON-ZERO ❌, Instance 2 = NON-ZERO ❌
- There was NO commit between Session 61 and Session 63 that fixed them!

**So the truth is:**
- BEFORE Session 63: Instance 1 BROKEN, Instance 2 BROKEN
- AFTER Session 63: Instance 1 STILL BROKEN, Instance 2 FIXED

## Why Session 63 Changes Helped Instance 2

### Change 3: Bytecode Copying
```zig
for (0..K) |k| {
    if ((k < io_start or k >= io_end) && k != termination && k != panic) {
        if (val_final[k].eql(F.zero()) && !val_init[k].eql(F.zero())) {
            val_final[k] = val_init[k];  // ← Copy bytecode!
        }
    }
}
```

**Why this fixed Instance 2:**
- initial_ram has bytecode (from buildInitialRamMap line 132)
- final_ram (emulator.memory) does NOT have bytecode
- Copying val_init → val_final for bytecode region makes them match
- After MLE: `output_val_final_claim` = `output_val_init_claim` ✓

### Change 2: Termination Bit
```zig
val_init[termination_index] = F.one();
val_final[termination_index] = F.one();
```

**Why this also helped Instance 2:**
- Eliminates the termination bit difference between val_init and val_final

## Why Instance 1 Is Still Broken

**Instance 1 needs:** `rwc_val_claim` = `val_init_eval`

**Current state:**
- `rwc_val_claim`: From RWC's val_init (termination=0, from initial_ram)
- `val_init_eval`: From OutputSumcheck's val_init (termination=1, after Session 63 change)
- **Result**: They don't match! ❌

**The mismatch:**
```
rwc_val_claim: uses initial_ram without termination bit workaround
val_init_eval: uses OutputSumcheck WITH termination bit workaround
Difference: NON-ZERO ❌
```

## Solution for Instance 1

**Option A**: Add termination bit to RWC's val_init (same workaround as OutputSumcheck)
**Option B**: Revert to computing val_init_eval from initial_ram (same source as RWC)

Both would make `rwc_val_claim` = `val_init_eval`, fixing Instance 1.

## Next Steps

1. Decide on fix approach (Option A or B)
2. Implement fix for Instance 1
3. Verify BOTH Instance 1 AND 2 work together
4. Find Jolt's actual Stage 4 prover values to compare
